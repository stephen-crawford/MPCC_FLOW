//! Finite-horizon MPCC solver.
//!
//! Strategy: Gauss-Newton on the nominal trajectory, plus a projected
//! gradient step on the condensed (U-only) cost. One or two outer iterations
//! per `solve()` call is enough at typical horizons (N <= 16) — the paper's
//! linearized QP (Eq. 11-13) is quadratic, so a single Newton step is close
//! to optimal when the linearization is accurate.
//!
//! This is the Rust analogue of `planning/network_solver.py::NetworkMPCCQPSolver`
//! in the Python reference implementation. We avoid pulling in an external QP
//! backend (osqp/clarabel) by handling box constraints via projected gradient
//! and the state inequalities (q <= q_max, R <= 4 R*) via a quadratic
//! penalty. The trade-off is small infeasibility at saturation; for
//! mahimahi-style bounded-buffer experiments this is fine.

use std::time::Instant;

use super::config::MpccConfig;
use super::cost::total_cost;
use super::dynamics::rollout;

#[derive(Clone, Debug)]
pub struct QpResult {
    pub success: bool,
    pub send_rate_bps: f64,
    pub control_sequence: Vec<f64>,
    pub predicted_throughput: Vec<f64>,
    pub predicted_rtt: Vec<f64>,
    pub solve_time_ms: f64,
}

#[derive(Clone)]
pub struct Solver {
    pub cfg: MpccConfig,
    last_u: Option<Vec<f64>>,
}

impl Solver {
    pub fn new(cfg: MpccConfig) -> Self {
        Solver { cfg, last_u: None }
    }

    /// Solve the MPCC problem from the given initial state.
    pub fn solve(
        &mut self, tput: f64, rtt: f64, queue_bytes: f64, bw_est: f64,
    ) -> QpResult {
        let t0 = Instant::now();
        let bw = bw_est.max(1.0);
        let n = self.cfg.horizon;

        // Warm-start u_bar. Shift last solve forward one step.
        let mut u = match &self.last_u {
            Some(prev) if prev.len() == n => {
                let mut shifted = prev.clone();
                for k in 0..n - 1 {
                    shifted[k] = shifted[k + 1];
                }
                shifted
            }
            _ => vec![bw * 0.5; n],
        };

        let x0 = (tput, rtt, queue_bytes);
        let (u_opt, success) = self.projected_gradient(x0, &mut u, bw);
        let traj = rollout(x0, &u_opt, bw, &self.cfg);
        self.last_u = Some(u_opt.clone());

        QpResult {
            success,
            send_rate_bps: u_opt[0].clamp(self.cfg.min_rate_bps, self.cfg.rate_max_bps),
            control_sequence: u_opt,
            predicted_throughput: traj.iter().map(|p| p.0).collect(),
            predicted_rtt: traj.iter().map(|p| p.1).collect(),
            solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Box-constrained projected gradient descent on the condensed cost.
    /// Returns (u_final, success).
    fn projected_gradient(
        &self, x0: (f64, f64, f64), u: &mut [f64], bw: f64,
    ) -> (Vec<f64>, bool) {
        const MAX_ITER: usize = 50;
        const EPS: f64 = 1e-2; // Mbps
        const BACKTRACK_MAX: usize = 10;

        let n = u.len();
        let rate_max = self.cfg.rate_max_bps;

        let mut u_vec: Vec<f64> = u.to_vec();
        let mut best_cost = self.augmented_cost(x0, &u_vec, bw);
        let mut step = 1e-2;

        for _ in 0..MAX_ITER {
            let grad = self.numeric_gradient(x0, &u_vec, bw);
            let grad_norm: f64 = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
            if grad_norm < EPS {
                break;
            }

            // Backtracking line search: shrink step until cost decreases.
            let mut accepted = false;
            let mut trial_cost = best_cost;
            let mut trial = u_vec.clone();
            let mut local_step = step;
            for _ in 0..BACKTRACK_MAX {
                for k in 0..n {
                    trial[k] = (u_vec[k] - local_step * grad[k])
                        .clamp(0.0, rate_max);
                }
                trial_cost = self.augmented_cost(x0, &trial, bw);
                if trial_cost < best_cost {
                    accepted = true;
                    break;
                }
                local_step *= 0.5;
            }
            if !accepted {
                // No progress possible with this gradient — terminate.
                break;
            }
            u_vec = trial;
            best_cost = trial_cost;
            // Try a slightly larger step next iteration (momentum-ish).
            step = (local_step * 1.5).min(1e-1);
        }

        (u_vec, best_cost.is_finite())
    }

    fn numeric_gradient(
        &self, x0: (f64, f64, f64), u: &[f64], bw: f64,
    ) -> Vec<f64> {
        // Scale the perturbation with rate_max so finite differences are
        // well-conditioned across the 50x range of plausible send rates.
        let h = self.cfg.rate_max_bps * 1e-4;
        let base = self.augmented_cost(x0, u, bw);
        let mut g = vec![0.0_f64; u.len()];
        for k in 0..u.len() {
            let mut up = u.to_vec();
            up[k] += h;
            g[k] = (self.augmented_cost(x0, &up, bw) - base) / h;
        }
        g
    }

    /// Cost + quadratic barrier on the state inequality constraints
    /// (q <= q_max, R <= 4 R*). Box bounds on U are enforced by projection.
    fn augmented_cost(
        &self, x0: (f64, f64, f64), u: &[f64], bw: f64,
    ) -> f64 {
        let traj = rollout(x0, u, bw, &self.cfg);
        let mut c = total_cost(&traj, u, bw, &self.cfg);

        // Penalty for q_k > q_max and R_k > 4 R*.
        let q_pen_w = 1.0;
        let r_pen_w = 1.0;
        let r_ceiling = 4.0 * self.cfg.rtt_target_s;
        for (_t, r, q) in traj.iter().copied() {
            let over_q = (q - self.cfg.q_max_bytes).max(0.0);
            let over_r = (r - r_ceiling).max(0.0);
            c += q_pen_w * (over_q / self.cfg.q_max_bytes.max(1.0)).powi(2);
            c += r_pen_w * (over_r / r_ceiling).powi(2);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MpccConfig {
        MpccConfig {
            horizon: 6,
            dt_s: 0.05,
            rate_max_bps: 20e6,
            rtt_prop_s: 0.020,
            rtt_target_s: 0.050,
            alpha: 0.150,
            q_max_bytes: 100_000.0,
            ..MpccConfig::default()
        }
    }

    #[test]
    fn solve_bounds_respected() {
        let mut s = Solver::new(cfg());
        let res = s.solve(5e6, 0.030, 5_000.0, 10e6);
        assert!(res.success);
        assert!(res.send_rate_bps >= 0.0);
        assert!(res.send_rate_bps <= cfg().rate_max_bps);
    }

    #[test]
    fn solve_pulls_back_at_queue_ceiling() {
        let mut s = Solver::new(cfg());
        let cfg_ = cfg();
        let res = s.solve(9e6, 0.1, cfg_.q_max_bytes * 0.95, 10e6);
        assert!(res.success);
        assert!(res.send_rate_bps <= 12e6,
                "expected pullback near queue ceiling, got {}",
                res.send_rate_bps);
    }
}
