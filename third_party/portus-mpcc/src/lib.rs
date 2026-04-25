//! MPCC congestion control algorithm, ported from mpcc_flow's Python
//! `planning/network_solver.py` to a portus/Rust CCP datapath.
//!
//! Per-RTT on_report loop:
//!   1. Decode the datapath report -> (acked, sacked, loss, timeout, rtt,
//!      inflight).
//!   2. Estimate network state:  T (throughput), R (RTT), q (queue delay).
//!   3. Linearize the Eq. (4) dynamics at the previous solve's nominal U,
//!      assemble a finite-horizon QP matching Eq. (7-9), and solve it with
//!      the box-constrained projected gradient solver in `qp.rs`.
//!   4. Update the CCP Cwnd field with u*[0] * rtt, bounded by the iperf
//!      pacing regime.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use log::{debug, info, warn};
use portus::ipc::Ipc;
use portus::lang::Scope;
use portus::{CongAlg, Datapath, DatapathInfo, DatapathTrait, Flow, Report};

// MpccFlow is parameterised by the IPC type rather than storing a boxed
// DatapathTrait: Datapath<I> holds Rc<> internally and therefore cannot be
// boxed as `dyn DatapathTrait + Send`. Mirrors the nimbus crate's pattern
// (see nimbus/src/lib.rs: NimbusFlow<T>).

pub mod mpcc;

pub use mpcc::config::MpccConfig;
pub use mpcc::solver::QpResult;

const DATAPATH_PROGRAM: &str = r#"
(def (Report
    (volatile acked 0)
    (volatile sacked 0)
    (volatile loss 0)
    (volatile timeout false)
    (volatile rtt 0)
    (volatile inflight 0)
))
(when true
    (:= Report.inflight Flow.packets_in_flight)
    (:= Report.rtt Flow.rtt_sample_us)
    (:= Report.acked (+ Report.acked Ack.bytes_acked))
    (:= Report.sacked (+ Report.sacked Ack.packets_misordered))
    (:= Report.loss Ack.lost_pkts_sample)
    (:= Report.timeout Flow.was_timeout)
    (fallthrough)
)
(when (|| Report.timeout (> Report.loss 0))
    (report)
    (:= Micros 0)
)
(when (> Micros Flow.rtt_sample_us)
    (report)
    (:= Micros 0)
)
"#;

#[derive(Clone, Default)]
pub struct MpccAlgorithm {
    pub config: MpccConfig,
}

impl MpccAlgorithm {
    pub fn new(config: MpccConfig) -> Self {
        MpccAlgorithm { config }
    }
}

impl<I: Ipc> CongAlg<I> for MpccAlgorithm {
    type Flow = MpccFlow<I>;

    fn name() -> &'static str {
        "mpcc"
    }

    fn datapath_programs(&self) -> HashMap<&'static str, String> {
        let mut h = HashMap::new();
        h.insert("default", DATAPATH_PROGRAM.to_string());
        h
    }

    fn new_flow(&self, mut control: Datapath<I>, info: DatapathInfo) -> Self::Flow {
        let scope = control.set_program("default", None).expect("set_program");
        let init_cwnd = (info.mss as f64 * 10.0) as u32;
        control
            .update_field(&scope, &[("Cwnd", init_cwnd)])
            .expect("initial cwnd");
        MpccFlow::new(control, scope, info, self.config.clone())
    }
}

/// Sliding-window length for the BBR-style BtlBw max-filter (samples).
/// At ~20 ms report cadence, 30 samples ≈ 600 ms of history — long enough
/// for a cellular fade/peak cycle to re-contribute its peak before the
/// filter forgets it, but short enough that the filter still tracks a
/// shifted baseline capacity after a sustained drop. Previously 10 (≈200 ms)
/// which caused C_est to collapse during fades and starve the planner.
const BW_WINDOW: usize = 30;
/// Phases in the probe-BW cycle: one RTT of up-gain, one RTT of drain,
/// then four RTTs of 1.0× steady. Tightened from BBR's 8-RTT cadence
/// (6 steady phases) because cellular capacity swings faster than wired
/// links; shorter cycle gives the max-filter fresh peak samples more
/// often. Gains are configurable via `MpccConfig::probe_up_gain` /
/// `probe_down_gain`.
const PROBE_CYCLE: u32 = 6;
/// Cold-start phases where we hold pacing gain at 1.0 so the BW filter
/// populates with at least a few honest samples before we start probing.
const COLD_START_REPORTS: u32 = 4;

fn pacing_gain(phase: u32, up: f64, down: f64) -> f64 {
    match phase % PROBE_CYCLE {
        0 => up,
        1 => down,
        _ => 1.0,
    }
}

pub struct MpccFlow<T: Ipc> {
    control: Datapath<T>,
    scope: Scope,
    info: DatapathInfo,
    solver: mpcc::solver::Solver,
    last_report: Option<Instant>,
    last_rate_bps: f64,
    rtt_prop_us: Option<u32>,
    /// Per-ACK delivery-rate samples for the BBR-style BtlBw max-filter.
    /// Only used when ``cfg.probe_bw = true``; otherwise we keep the legacy
    /// monotonic-max ``bw_peak_bps``.
    bw_samples: VecDeque<f64>,
    /// Legacy monotonic max of observed delivery rate, seeded at rate_max so
    /// Γ(θ) has a well-defined scale from the first RTT. Used whenever
    /// ``cfg.probe_bw = false``.
    bw_peak_bps: f64,
    /// Monotonically-increasing report counter; pacing_gain cycles on this.
    phase: u32,
    cfg: MpccConfig,
}

impl<T: Ipc> MpccFlow<T> {
    fn new(
        control: Datapath<T>,
        scope: Scope,
        info: DatapathInfo,
        cfg: MpccConfig,
    ) -> Self {
        let init_rate_bps = (info.init_cwnd as f64 * 8.0) / 0.1; // 10 mss / 100 ms
        let bw_peak_bps = cfg.rate_max_bps.max(init_rate_bps);
        MpccFlow {
            control,
            scope,
            info,
            solver: mpcc::solver::Solver::new(cfg.clone()),
            last_report: None,
            last_rate_bps: init_rate_bps,
            rtt_prop_us: None,
            bw_samples: VecDeque::with_capacity(BW_WINDOW),
            bw_peak_bps,
            phase: 0,
            cfg,
        }
    }

    /// bw_est (bps): max of the last BW_WINDOW per-ACK delivery rates,
    /// or rate_max_bps during cold start. Floor 100 kbps so divisions in
    /// the dynamics never see ~0.
    fn bw_est_bps(&self) -> f64 {
        if self.bw_samples.is_empty() {
            return self.cfg.rate_max_bps.max(1e5);
        }
        self.bw_samples
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
            .max(1e5)
    }
}

impl<T: Ipc> Flow for MpccFlow<T> {
    fn on_report(&mut self, _sock_id: u32, m: Report) {
        let acked: u32 = m.get_field("Report.acked", &self.scope).unwrap_or(0) as u32;
        let rtt_us: u32 = m.get_field("Report.rtt", &self.scope).unwrap_or(0) as u32;
        let loss: u32 = m.get_field("Report.loss", &self.scope).unwrap_or(0) as u32;
        let timeout: u32 = m.get_field("Report.timeout", &self.scope).unwrap_or(0) as u32;
        let _inflight: u32 =
            m.get_field("Report.inflight", &self.scope).unwrap_or(0) as u32;

        if rtt_us == 0 {
            return;
        }
        if timeout > 0 {
            self.on_timeout();
            return;
        }
        if loss > 0 {
            self.on_loss();
            return;
        }

        let now = Instant::now();
        let dt_s = self
            .last_report
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(self.cfg.dt_s);
        self.last_report = Some(now);

        // --- Estimators --------------------------------------------------
        let rtt_s = rtt_us as f64 / 1e6;
        self.rtt_prop_us = Some(
            self.rtt_prop_us
                .map(|r| r.min(rtt_us))
                .unwrap_or(rtt_us),
        );
        let rtt_prop_s = self.rtt_prop_us.unwrap() as f64 / 1e6;

        let tput_bps = (acked as f64 * 8.0) / dt_s.max(1e-3);
        let bw_est_bps = if self.cfg.probe_bw {
            // Per-ACK delivery-rate sample feeds the BBR-style max-filter.
            // The filter gives us a C that follows actual link capacity; the
            // probe-BW pacing gain below gives the filter fresh high-water
            // samples whenever the link has slack so C can grow after fades.
            if tput_bps.is_finite() && tput_bps > 0.0 {
                self.bw_samples.push_back(tput_bps);
                while self.bw_samples.len() > BW_WINDOW {
                    self.bw_samples.pop_front();
                }
            }
            self.bw_est_bps()
        } else {
            // Legacy monotonic-max: stable on wired/LEO, but cannot shrink
            // below the configured rate_max. That's the paper's shipped
            // behavior and the baseline for Tables 1 & 3.
            self.bw_peak_bps = self.bw_peak_bps.max(tput_bps);
            self.bw_peak_bps.max(1.0)
        };

        // Queue delay (seconds) from RTT inflation above rtt_prop; converted
        // into bytes at the bandwidth estimate, matching network_solver.py.
        let q_delay_s = (rtt_s - rtt_prop_s).max(0.0);
        let q_bytes = q_delay_s * bw_est_bps / 8.0;

        // --- Solve ------------------------------------------------------
        let result = self.solver.solve(tput_bps, rtt_s, q_bytes, bw_est_bps);
        let base_rate_bps = if result.success {
            result.send_rate_bps
        } else {
            debug!("MPCC solve failed, holding rate {}", self.last_rate_bps);
            self.last_rate_bps
        };

        // Probe-BW pacing gain: one RTT up at 1.25×, one down at 0.75×,
        // six at 1.0× (BBR's ProbeBW cadence). Only enabled when the
        // config opts in (cellular.yml sets probe_bw: true); on stable
        // wired / LEO links the perturbations have no slack to probe for.
        let gain = if self.cfg.probe_bw && self.phase >= COLD_START_REPORTS {
            pacing_gain(self.phase, self.cfg.probe_up_gain, self.cfg.probe_down_gain)
        } else {
            1.0
        };
        self.phase = self.phase.wrapping_add(1);
        let rate_bps = (base_rate_bps * gain)
            .clamp(self.cfg.min_rate_bps, self.cfg.rate_max_bps);
        self.last_rate_bps = rate_bps;

        // cwnd (bytes) = send_rate (bps) * rtt_prop (s) / 8.
        //
        // Critical: use rtt_prop, NOT the current rtt_s. Using rtt_s is a
        // positive-feedback loop — if the queue fills, RTT inflates, cwnd
        // scales up, the sender pushes harder, and the queue grows further,
        // turning MPCC into a loss-based controller despite the solver
        // having already chosen a rate <= bw. Pinning cwnd to the BDP
        // (rate * rtt_prop) keeps the queue bounded by the solver's
        // queue-penalty choice of rate.
        let cwnd_bytes = ((rate_bps * rtt_prop_s) / 8.0) as u32;
        let cwnd_bytes = cwnd_bytes.max((self.info.mss * 2) as u32);

        // One info-level line per solve. Parsed by scripts/network/paper_metrics.py
        // to recover solve-time (§V.D) and the rate/throughput/rtt time-series.
        info!(
            "mpcc_step solve_ms={:.3} rate_bps={:.0} tput_bps={:.0} rtt_us={} q_bytes={:.0} success={}",
            result.solve_time_ms,
            rate_bps,
            tput_bps,
            rtt_us,
            q_bytes,
            result.success as u8,
        );

        if let Err(e) = self.control.update_field(&self.scope, &[("Cwnd", cwnd_bytes)]) {
            warn!("failed to update Cwnd: {:?}", e);
        }
    }
}

impl<T: Ipc> MpccFlow<T> {
    fn on_loss(&mut self) {
        // AIMD multiplicative decrease.
        self.last_rate_bps = (self.last_rate_bps * 0.5).max(self.cfg.min_rate_bps);
        let cwnd = ((self.last_rate_bps * self.cfg.dt_s) / 8.0) as u32;
        let cwnd = cwnd.max((self.info.mss * 2) as u32);
        let _ = self.control.update_field(&self.scope, &[("Cwnd", cwnd)]);
    }

    fn on_timeout(&mut self) {
        self.last_rate_bps = self.cfg.min_rate_bps;
        let cwnd = (self.info.mss * 2) as u32;
        let _ = self.control.update_field(&self.scope, &[("Cwnd", cwnd)]);
    }
}
