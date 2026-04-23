//! Config shared across the MPCC planner. Mirrors the YAML schema read by
//! the Python `network_solver.py` so the two implementations can be
//! compared head-to-head.

#[derive(Clone, Debug)]
pub struct MpccConfig {
    // Planner
    pub horizon: usize,
    pub dt_s: f64,

    // Network
    pub rtt_prop_s: f64,
    pub tau_rtt_s: f64,
    pub tau_tput_s: f64,
    pub rate_max_bps: f64,
    pub alpha: f64,
    pub rtt_target_s: f64,
    pub q_max_bytes: f64,
    pub n_flows: usize,
    pub min_rate_bps: f64,
    /// Upper bound on the path progress rate v_theta (units: 1/s). Default
    /// lets the optimizer cover the entire curve in ~1/v_theta_max seconds
    /// under unbounded cost; the real trajectory will be further limited by
    /// the network dynamics (tau_T, etc.).
    pub v_theta_max: f64,

    // Weights (paper Sec. IV)
    pub w_contour: f64,
    pub w_lag: f64,
    pub w_delay: f64,
    pub w_power: f64,
    pub w_du: f64,
    pub w_fair: f64,
    /// Progress reward weight. Multiplies v_theta_k inside the stage cost (as
    /// a subtracted term) so the controller has a genuine reason to advance
    /// along the reference curve. Must be > 0 or the controller parks at
    /// theta=0 (zero throughput).
    pub w_theta: f64,

    /// Enable BBR-style probe-BW pacing (1.25× / 0.75× / 1.0××6 cycle on
    /// top of the solver's rate) plus a sliding-window max-filter on
    /// delivery rate. Needed on cellular traces where the link capacity
    /// swings beneath `rate_max_bps`. Off by default: on stable wired /
    /// LEO links the perturbations have no slack to probe and hurt
    /// throughput.
    pub probe_bw: bool,
}

impl Default for MpccConfig {
    fn default() -> Self {
        MpccConfig {
            horizon: 8,
            dt_s: 0.020,
            rtt_prop_s: 0.020,
            tau_rtt_s: 0.1,
            tau_tput_s: 0.1,
            rate_max_bps: 50e6,
            alpha: 0.150,
            rtt_target_s: 0.050,
            q_max_bytes: 100_000.0,
            n_flows: 1,
            min_rate_bps: 1e5,
            v_theta_max: 5.0,
            w_contour: 50.0,
            w_lag: 1.0,
            w_delay: 100.0,
            w_power: 0.1,
            w_du: 0.5,
            w_fair: 0.0,
            w_theta: 10.0,
            probe_bw: false,
        }
    }
}

impl MpccConfig {
    /// Parse a YAML config file with the same schema as mpcc_flow's
    /// `config/CONFIG_NETWORK.yml`. Returns defaults on any missing key.
    #[cfg(feature = "yaml")]
    pub fn from_yaml_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let parsed: serde_yaml::Value = serde_yaml::from_str(&text)?;
        let mut cfg = Self::default();

        if let Some(planner) = parsed.get("planner") {
            if let Some(v) = planner.get("horizon").and_then(|x| x.as_u64()) {
                cfg.horizon = v as usize;
            }
            if let Some(v) = planner.get("timestep").and_then(|x| x.as_f64()) {
                cfg.dt_s = v;
            }
        }
        if let Some(net) = parsed.get("network") {
            for (key, dest) in [
                ("rtt_prop", &mut cfg.rtt_prop_s),
                ("tau_rtt", &mut cfg.tau_rtt_s),
                ("tau_tput", &mut cfg.tau_tput_s),
                ("rate_max", &mut cfg.rate_max_bps),
                ("alpha", &mut cfg.alpha),
                ("rtt_target", &mut cfg.rtt_target_s),
                ("q_max", &mut cfg.q_max_bytes),
                ("v_theta_max", &mut cfg.v_theta_max),
            ] {
                if let Some(v) = net.get(key).and_then(|x| x.as_f64()) {
                    *dest = v;
                }
            }
            if let Some(v) = net.get("n_flows").and_then(|x| x.as_u64()) {
                cfg.n_flows = v as usize;
            }
            if let Some(v) = net.get("probe_bw").and_then(|x| x.as_bool()) {
                cfg.probe_bw = v;
            }
        }
        if let Some(w) = parsed.get("weights") {
            for (key, dest) in [
                ("contour_weight", &mut cfg.w_contour),
                ("contouring_lag_weight", &mut cfg.w_lag),
                ("delay_weight", &mut cfg.w_delay),
                ("power_weight", &mut cfg.w_power),
                ("acceleration_weight", &mut cfg.w_du),
                ("fairness_weight", &mut cfg.w_fair),
                ("theta_progress_weight", &mut cfg.w_theta),
            ] {
                if let Some(v) = w.get(key).and_then(|x| x.as_f64()) {
                    *dest = v;
                }
            }
        }
        Ok(cfg)
    }
}
