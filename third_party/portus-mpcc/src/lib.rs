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

use std::collections::HashMap;
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

pub struct MpccFlow<T: Ipc> {
    control: Datapath<T>,
    scope: Scope,
    info: DatapathInfo,
    solver: mpcc::solver::Solver,
    last_report: Option<Instant>,
    last_rate_bps: f64,
    rtt_prop_us: Option<u32>,
    bw_peak_bps: f64,
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
        // Seed bw_peak_bps at rate_max_bps (the paper's link capacity C) so
        // that the reference curve Γ(θ) has a well-defined scale from the
        // first RTT. Without this seed, bw_peak bootstraps at ~0, the
        // dynamics clip every predicted s at the tiny bootstrap, the cost
        // has no gradient above the clip, and the solver collapses to
        // min_rate_bps. We refine bw_peak downward over the run via RTT/queue
        // feedback in the solver rather than by tracking observed tput.
        let bw_peak_bps = cfg.rate_max_bps.max(init_rate_bps);
        MpccFlow {
            control,
            scope,
            info,
            solver: mpcc::solver::Solver::new(cfg.clone()),
            last_report: None,
            last_rate_bps: init_rate_bps,
            rtt_prop_us: None,
            bw_peak_bps,
            cfg,
        }
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
        self.bw_peak_bps = self.bw_peak_bps.max(tput_bps);
        let bw_est_bps = self.bw_peak_bps.max(1.0);

        // Queue delay (seconds) from RTT inflation above rtt_prop; converted
        // into bytes at the bandwidth estimate, matching network_solver.py.
        let q_delay_s = (rtt_s - rtt_prop_s).max(0.0);
        let q_bytes = q_delay_s * bw_est_bps / 8.0;

        // --- Solve ------------------------------------------------------
        let result = self.solver.solve(tput_bps, rtt_s, q_bytes, bw_est_bps);
        let rate_bps = if result.success {
            result.send_rate_bps
        } else {
            debug!("MPCC solve failed, holding rate {}", self.last_rate_bps);
            self.last_rate_bps
        };
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
