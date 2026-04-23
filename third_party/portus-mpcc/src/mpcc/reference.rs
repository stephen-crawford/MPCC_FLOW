//! Reference curve in throughput-delay space (paper Eq. 5-6):
//!     Gamma(theta) = ( C * theta, R_0 + alpha * theta^2 )
//! with unit tangent
//!     t_hat(theta) = (C, 2 alpha theta) / |(C, 2 alpha theta)|

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

/// Contouring and lag errors (paper Eq. 6a, 6b) at an operating point.
pub fn errors(
    tput: f64, rtt: f64, bw: f64, rtt_prop: f64, alpha: f64,
) -> (f64, f64) {
    let theta = (tput / bw.max(1.0)).clamp(0.0, 1.0);
    let (ref_t, ref_r) = gamma(theta, bw, rtt_prop, alpha);
    let (tx, ty) = tangent(theta, bw, alpha);
    let dx = tput - ref_t;
    let dy = rtt - ref_r;
    let e_c = ty * dx - tx * dy;
    let e_l = tx * dx + ty * dy;
    (e_c, e_l)
}
