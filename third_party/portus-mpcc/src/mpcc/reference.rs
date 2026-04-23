//! Reference curve in throughput-delay space (paper Eq. 5-6):
//!     Gamma(theta) = ( C * theta, R_0 + alpha * theta^2 )
//! with unit tangent
//!     t_hat(theta) = (C, 2 alpha theta) / |(C, 2 alpha theta)|
//!
//! The path parameter `theta` is treated as an independent optimization
//! variable (see solver.rs), not as a function of the current throughput.
//! This preserves the geometric tracking problem — contouring error stays
//! orthogonal to the curve, lag error stays along it.

/// Reference operating point at progress theta.
pub fn gamma(theta: f64, bw: f64, rtt_prop: f64, alpha: f64) -> (f64, f64) {
    let theta = theta.clamp(0.0, 1.0);
    (bw * theta, rtt_prop + alpha * theta * theta)
}

/// Unit tangent to the reference curve at theta.
pub fn tangent(theta: f64, bw: f64, alpha: f64) -> (f64, f64) {
    let theta = theta.clamp(0.0, 1.0);
    let tt = bw;
    let tr = 2.0 * alpha * theta;
    let n = (tt * tt + tr * tr + 1e-8).sqrt();
    (tt / n, tr / n)
}

/// Contouring and lag errors at an operating point (tput, rtt) relative to the
/// reference point Gamma(theta). `theta` is the independent path parameter;
/// the caller supplies it (typically from the solver's decision variable,
/// falling back to `tput/C` only in diagnostic/metric code paths).
pub fn errors_at_theta(
    tput: f64,
    rtt: f64,
    theta: f64,
    bw: f64,
    rtt_prop: f64,
    alpha: f64,
) -> (f64, f64) {
    let (ref_t, ref_r) = gamma(theta, bw, rtt_prop, alpha);
    let (tx, ty) = tangent(theta, bw, alpha);
    let dx = tput - ref_t;
    let dy = rtt - ref_r;
    let e_c = ty * dx - tx * dy;
    let e_l = tx * dx + ty * dy;
    (e_c, e_l)
}
