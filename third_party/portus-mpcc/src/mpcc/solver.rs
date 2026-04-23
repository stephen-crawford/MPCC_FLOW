//! Finite-horizon MPCC solver with path-parameter decision variables.
//!
//! Decision vector Z = [u_0, ..., u_{N-1}, v_0, ..., v_{N-1}] in R^{2N}.
//! Theta is built from (theta_0, V) via theta_{k+1} = theta_k + dt * v_k and
//! is used to evaluate the reference curve Gamma(theta_k); this is the
//! classical Liniger MPCC formulation, not the degenerate theta = T/C
//! collapse.
//!
//! Strategy: projected gradient on the condensed (Z-only) augmented cost.
//! Box bounds on U and V are enforced by projection; the scalar state
//! inequalities (q <= q_max, R <= 4 R*, theta in [0,1]) are handled by a
//! quadratic barrier. Two to three outer iterations at N<=16 are enough for
//! the RTT-scale control loop.

use std::time::Instant;

use super::config::MpccConfig;
use super::cost::total_cost;
use super::dynamics::rollout;

#[derive(Clone, Debug)]
pub struct QpResult {
    pub success: bool,
    pub send_rate_bps: f64,
    pub control_sequence: Vec<f64>,
    pub v_theta_sequence: Vec<f64>,
    pub theta_sequence: Vec<f64>,
    pub predicted_throughput: Vec<f64>,
    pub predicted_rtt: Vec<f64>,
    pub solve_time_ms: f64,
}

#[derive(Clone)]
pub struct Solver {
    pub cfg: MpccConfig,
    last_u: Option<Vec<f64>>,
    last_v: Option<Vec<f64>>,
    /// theta_0 to use on the next solve (receding-horizon shift of theta_1).
    next_theta0: f64,
}

impl Solver {
    pub fn new(cfg: MpccConfig) -> Self {
        Solver { cfg, last_u: None, last_v: None, next_theta0: 0.0 }
    }

    /// Solve the MPCC problem from the given initial state.
    ///
    /// Uses a two-start projected gradient: one from the receding-horizon
    /// shift of the last solve (continuity) and one from an aggressive
    /// (u = bw, v = v_theta_max) start that pushes the planner into the
    /// high-throughput basin. Returns whichever start reaches lower cost.
    pub fn solve(
        &mut self, tput: f64, rtt: f64, queue_bytes: f64, bw_est: f64,
    ) -> QpResult {
        let t0 = Instant::now();
        let bw = bw_est.max(1.0);
        let n = self.cfg.horizon;
        let v_max = self.cfg.v_theta_max;
        let theta_0 = self.next_theta0.clamp(0.0, 1.0);
        let x0 = (tput, rtt, queue_bytes);

        // Candidate A: shifted previous solve (or default high-rate/high-v
        // on the first call so cold-start still finds the good basin).
        let u_shifted: Vec<f64> = match &self.last_u {
            Some(prev) if prev.len() == n => {
                let mut s = prev.clone();
                for k in 0..n - 1 { s[k] = s[k + 1]; }
                s
            }
            _ => vec![bw; n],
        };
        let v_shifted: Vec<f64> = match &self.last_v {
            Some(prev) if prev.len() == n => {
                let mut s = prev.clone();
                for k in 0..n - 1 { s[k] = s[k + 1]; }
                s
            }
            _ => vec![v_max; n],
        };

        // Candidate B: aggressive full-rate / full-progress start. This is
        // the classic MPCC "cruise at the frontier" reference and reliably
        // falls into the optimum basin even when the shifted warm start is
        // in a degenerate local min.
        let u_hot = vec![bw; n];
        let v_hot = vec![v_max; n];

        let mut candidates: Vec<(Vec<f64>, Vec<f64>)> = vec![(u_shifted, v_shifted)];
        if self.last_u.is_some() {
            candidates.push((u_hot, v_hot));
        }

        let mut best_u: Option<Vec<f64>> = None;
        let mut best_v: Option<Vec<f64>> = None;
        let mut best_cost = f64::INFINITY;

        for (mut u_c, mut v_c) in candidates {
            let _ok = self.projected_gradient(x0, &mut u_c, &mut v_c, theta_0, bw);
            let c = self.augmented_cost(x0, &u_c, &v_c, theta_0, bw);
            if c.is_finite() && c < best_cost {
                best_cost = c;
                best_u = Some(u_c);
                best_v = Some(v_c);
            }
        }

        let (u, v) = match (best_u, best_v) {
            (Some(u), Some(v)) => (u, v),
            _ => (vec![bw * 0.5; n], vec![0.0; n]),   // degenerate fallback
        };

        let theta = theta_seq(theta_0, &v, self.cfg.dt_s);
        let traj = rollout(x0, &u, bw, &self.cfg);

        self.last_u = Some(u.clone());
        self.last_v = Some(v.clone());
        self.next_theta0 = theta.get(1).copied().unwrap_or(theta_0).clamp(0.0, 1.0);

        QpResult {
            success: best_cost.is_finite(),
            send_rate_bps: u[0].clamp(self.cfg.min_rate_bps, self.cfg.rate_max_bps),
            control_sequence: u,
            v_theta_sequence: v,
            theta_sequence: theta,
            predicted_throughput: traj.iter().map(|p| p.0).collect(),
            predicted_rtt: traj.iter().map(|p| p.1).collect(),
            solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Projected gradient descent on the condensed cost over (U, V).
    fn projected_gradient(
        &self,
        x0: (f64, f64, f64),
        u: &mut [f64],
        v: &mut [f64],
        theta_0: f64,
        bw: f64,
    ) -> bool {
        const MAX_ITER: usize = 50;
        const EPS: f64 = 1e-2;
        const BACKTRACK_MAX: usize = 10;

        let n = u.len();
        let rate_max = self.cfg.rate_max_bps;
        let v_max = self.cfg.v_theta_max;

        let mut best = self.augmented_cost(x0, u, v, theta_0, bw);
        let mut step = 1e-2;

        for _ in 0..MAX_ITER {
            let (g_u, g_v) = self.numeric_gradient(x0, u, v, theta_0, bw);
            let gn: f64 = (g_u.iter().map(|x| x * x).sum::<f64>()
                         + g_v.iter().map(|x| x * x).sum::<f64>()).sqrt();
            if gn < EPS { break; }

            let mut accepted = false;
            let mut local = step;
            let mut trial_u = u.to_vec();
            let mut trial_v = v.to_vec();
            let mut trial_cost = best;
            for _ in 0..BACKTRACK_MAX {
                for k in 0..n {
                    trial_u[k] = (u[k] - local * g_u[k]).clamp(0.0, rate_max);
                    trial_v[k] = (v[k] - local * g_v[k]).clamp(0.0, v_max);
                }
                trial_cost = self.augmented_cost(x0, &trial_u, &trial_v, theta_0, bw);
                if trial_cost < best { accepted = true; break; }
                local *= 0.5;
            }
            if !accepted { break; }
            u.copy_from_slice(&trial_u);
            v.copy_from_slice(&trial_v);
            best = trial_cost;
            step = (local * 1.5).min(1e-1);
        }

        best.is_finite()
    }

    fn numeric_gradient(
        &self,
        x0: (f64, f64, f64),
        u: &[f64],
        v: &[f64],
        theta_0: f64,
        bw: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let h_u = self.cfg.rate_max_bps * 1e-4;
        let h_v = self.cfg.v_theta_max.max(1e-3) * 1e-4;
        let base = self.augmented_cost(x0, u, v, theta_0, bw);

        let mut gu = vec![0.0_f64; u.len()];
        for k in 0..u.len() {
            let mut up = u.to_vec();
            up[k] += h_u;
            gu[k] = (self.augmented_cost(x0, &up, v, theta_0, bw) - base) / h_u;
        }
        let mut gv = vec![0.0_f64; v.len()];
        for k in 0..v.len() {
            let mut vp = v.to_vec();
            vp[k] += h_v;
            gv[k] = (self.augmented_cost(x0, u, &vp, theta_0, bw) - base) / h_v;
        }
        (gu, gv)
    }

    /// Cost + quadratic barrier on state inequalities (q, R) and on theta in
    /// [0, 1]. Box bounds on U, V handled by projection.
    fn augmented_cost(
        &self,
        x0: (f64, f64, f64),
        u: &[f64],
        v: &[f64],
        theta_0: f64,
        bw: f64,
    ) -> f64 {
        let traj = rollout(x0, u, bw, &self.cfg);
        let theta = theta_seq(theta_0, v, self.cfg.dt_s);

        let mut c = total_cost(&traj, u, &theta, v, bw, &self.cfg);

        let q_pen_w = 1.0;
        let r_pen_w = 1.0;
        let t_pen_w = 1.0;
        let r_ceiling = 4.0 * self.cfg.rtt_target_s;

        for (_t, r, q) in traj.iter().copied() {
            let over_q = (q - self.cfg.q_max_bytes).max(0.0);
            let over_r = (r - r_ceiling).max(0.0);
            c += q_pen_w * (over_q / self.cfg.q_max_bytes.max(1.0)).powi(2);
            c += r_pen_w * (over_r / r_ceiling).powi(2);
        }
        for &th in &theta {
            let over = (th - 1.0).max(0.0);
            let under = (-th).max(0.0);
            c += t_pen_w * (over * over + under * under);
        }
        c
    }
}

/// Build theta_0..theta_N from theta_0 and the v_theta sequence. Values are
/// clamped at each step so the decision sequence is feasible even if the
/// caller passed an initial v_theta outside [0, v_theta_max].
fn theta_seq(theta_0: f64, v: &[f64], dt: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len() + 1);
    let mut cur = theta_0.clamp(0.0, 1.0);
    out.push(cur);
    for &vk in v {
        cur = (cur + dt * vk).clamp(0.0, 1.0);
        out.push(cur);
    }
    out
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
            v_theta_max: 5.0,
            w_theta: 10.0,
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
        for &th in &res.theta_sequence {
            assert!((0.0..=1.0).contains(&th), "theta out of [0,1]: {}", th);
        }
        for &vk in &res.v_theta_sequence {
            assert!((0.0..=cfg().v_theta_max).contains(&vk),
                    "v out of range: {}", vk);
        }
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

    #[test]
    fn solve_makes_progress_along_curve() {
        // With a positive progress weight the controller should pick v_0 > 0
        // rather than parking at the starting theta.
        let mut s = Solver::new(cfg());
        let res = s.solve(1e6, 0.025, 1_000.0, 10e6);
        assert!(res.success);
        assert!(res.v_theta_sequence[0] > 0.0,
                "expected v_0 > 0 with w_theta > 0, got {}",
                res.v_theta_sequence[0]);
    }
}
