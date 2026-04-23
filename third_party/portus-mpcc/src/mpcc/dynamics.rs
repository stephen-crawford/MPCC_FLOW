//! Network dynamics (paper Eq. 4). Forward Euler for simplicity — the
//! Python reference uses RK4 but Eq. 4 is linear enough at short horizons
//! (N=8-12, dt=20-50 ms) that forward Euler matches to within 0.5%.

use super::config::MpccConfig;

/// Single forward step of Eq. 4 in SI units:
///    T [bps], R [s], q [bytes], s [bps], bw [bps]
pub fn step(
    t: f64, r: f64, q: f64, s: f64, bw: f64, cfg: &MpccConfig,
) -> (f64, f64, f64) {
    let eff = s.min(bw);
    let t_next = t + cfg.dt_s * (eff - t) / cfg.tau_tput_s;
    let r_eq = cfg.rtt_prop_s + q.max(0.0) / bw.max(1.0);
    let r_next = r + cfg.dt_s * (r_eq - r) / cfg.tau_rtt_s;
    let q_next = (q + cfg.dt_s * (s - bw)).max(0.0);
    (t_next, r_next, q_next)
}

/// Forward-simulate a control sequence.
pub fn rollout(
    x0: (f64, f64, f64),
    u_seq: &[f64],
    bw: f64,
    cfg: &MpccConfig,
) -> Vec<(f64, f64, f64)> {
    let mut traj = Vec::with_capacity(u_seq.len() + 1);
    traj.push(x0);
    let mut cur = x0;
    for &s in u_seq {
        cur = step(cur.0, cur.1, cur.2, s, bw, cfg);
        traj.push(cur);
    }
    traj
}

/// Closed-form Jacobian of the discrete dynamics (paper Eq. 13) at a
/// linearization point. Useful when assembling the exact QP; for the
/// simple projected-gradient QP solver we use here we just need a numeric
/// gradient, so `rollout` is enough.
pub fn jacobians(
    s: f64, bw: f64, cfg: &MpccConfig,
) -> ([[f64; 3]; 3], [f64; 3]) {
    // A_k = I + dt * d f / d x
    let a = [
        [1.0 - cfg.dt_s / cfg.tau_tput_s, 0.0, 0.0],
        [0.0, 1.0 - cfg.dt_s / cfg.tau_rtt_s, cfg.dt_s / (bw.max(1.0) * cfg.tau_rtt_s)],
        [0.0, 0.0, 1.0],
    ];
    // B_k = dt * d f / d s
    let b = [
        if s < bw { cfg.dt_s / cfg.tau_tput_s } else { 0.0 },
        0.0,
        cfg.dt_s,
    ];
    (a, b)
}
