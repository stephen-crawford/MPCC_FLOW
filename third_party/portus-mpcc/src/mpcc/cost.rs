//! Stage cost (paper Eq. 7-9).

use super::config::MpccConfig;
use super::reference::errors;

/// Total MPCC cost over the horizon for a given control sequence and
/// forward-simulated trajectory.
pub fn total_cost(
    traj: &[(f64, f64, f64)],
    u_seq: &[f64],
    bw: f64,
    cfg: &MpccConfig,
) -> f64 {
    let bw_safe = bw.max(1.0);
    let fair_share = bw_safe / cfg.n_flows.max(1) as f64;
    let mut cost = 0.0;

    for (t, r, _q) in traj.iter().copied() {
        let (e_c, e_l) = errors(t, r, bw_safe, cfg.rtt_prop_s, cfg.alpha);
        cost += cfg.w_contour * e_c * e_c;
        cost += cfg.w_lag * e_l * e_l;

        let excess = ((r - cfg.rtt_target_s) / cfg.rtt_target_s).max(0.0);
        cost += cfg.w_delay * excess * excess;

        // Kleinrock power (reward, subtracted).
        cost -= cfg.w_power * (t / bw_safe) / (r / cfg.rtt_prop_s + 1e-6);
    }

    for k in 0..u_seq.len() {
        if k > 0 {
            let d = u_seq[k] - u_seq[k - 1];
            cost += cfg.w_du * (d / cfg.rate_max_bps).powi(2);
        }
        if cfg.w_fair > 0.0 && cfg.n_flows > 1 {
            let d = u_seq[k] - fair_share;
            cost += cfg.w_fair * (d / cfg.rate_max_bps).powi(2);
        }
    }

    cost
}
