//! Stage cost for the path-parameter MPCC formulation.
//!
//! Contouring and lag errors are evaluated at the *decision-variable* theta_k,
//! not at theta_k = T_k/C. Progress along the curve is rewarded via
//! `- w_theta * v_theta_k`, giving the controller a genuine reason to advance.

use super::config::MpccConfig;
use super::reference::errors_at_theta;

/// Total MPCC cost over the horizon.
///
/// * `traj`      — forward-simulated `(T, R, q)` states, length N+1.
/// * `u_seq`     — send rates, length N.
/// * `theta_seq` — path parameters, length N+1.
/// * `v_seq`     — progress rates, length N.
pub fn total_cost(
    traj: &[(f64, f64, f64)],
    u_seq: &[f64],
    theta_seq: &[f64],
    v_seq: &[f64],
    bw: f64,
    cfg: &MpccConfig,
) -> f64 {
    let bw_safe = bw.max(1.0);
    let fair_share = bw_safe / cfg.n_flows.max(1) as f64;
    let mut cost = 0.0;

    for (k, (t, r, _q)) in traj.iter().copied().enumerate() {
        let theta_k = theta_seq[k];
        let (e_c, e_l) = errors_at_theta(t, r, theta_k, bw_safe, cfg.rtt_prop_s, cfg.alpha);
        cost += cfg.w_contour * e_c * e_c;
        cost += cfg.w_lag * e_l * e_l;

        let excess = ((r - cfg.rtt_target_s) / cfg.rtt_target_s).max(0.0);
        cost += cfg.w_delay * excess * excess;

        // Kleinrock power (reward, subtracted).
        cost -= cfg.w_power * (t / bw_safe) / (r / cfg.rtt_prop_s + 1e-6);
    }

    for k in 0..u_seq.len() {
        // Progress reward.
        cost -= cfg.w_theta * v_seq[k];

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
