//! MPCC CCP binary entry point.
//!
//! Invocation:
//!     sudo ./mpcc_cca --ipc netlink
//!     sudo ./mpcc_cca --ipc unix
//!
//! Run `sudo ccp_kernel_load ipc=0` first (see nimbus-measurement's
//! `ccp-kernel`). Then set the kernel CCA to `ccp`:
//!     sudo sysctl -w net.ipv4.tcp_congestion_control=ccp

use clap::Parser;
use portus::algs::ipc_valid;
use portus_mpcc::{MpccAlgorithm, MpccConfig};

#[derive(Parser)]
#[command(name = "mpcc_cca")]
#[command(about = "MPCC congestion control as a portus/CCP datapath")]
struct Args {
    #[arg(long, default_value = "netlink", value_parser = ipc_string_parser)]
    ipc: String,

    #[arg(long, default_value_t = 8)]
    horizon: usize,

    #[arg(long, default_value_t = 0.020)]
    dt_s: f64,

    /// Solver mode: currently informational — this binary always runs the
    /// Gauss-Newton + projected-gradient condensed QP. `--solver nlp` is
    /// accepted for CLI compatibility with the Python stub but unused.
    #[arg(long, default_value = "qp")]
    solver: String,

    #[arg(long)]
    config: Option<String>,
}

fn ipc_string_parser(v: &str) -> Result<String, String> {
    ipc_valid(v.to_string()).map(|_| v.to_string())
}

fn main() {
    // Default to portus_mpcc=info so per-step solve-time lines are emitted
    // without needing RUST_LOG in the environment (which sudo may strip).
    // Callers can still override via RUST_LOG at runtime.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("portus_mpcc=info"),
    ).init();
    let args = Args::parse();

    let mut cfg = if let Some(path) = args.config.as_deref() {
        #[cfg(feature = "yaml")]
        {
            MpccConfig::from_yaml_file(path).unwrap_or_else(|e| {
                eprintln!("failed to load {path}: {e}; using defaults");
                MpccConfig::default()
            })
        }
        #[cfg(not(feature = "yaml"))]
        {
            eprintln!("portus-mpcc built without `yaml` feature — ignoring --config");
            MpccConfig::default()
        }
    } else {
        MpccConfig::default()
    };
    cfg.horizon = args.horizon;
    cfg.dt_s = args.dt_s;

    let alg = MpccAlgorithm::new(cfg);

    match args.ipc.as_str() {
        "netlink" => {
            #[cfg(target_os = "linux")]
            {
                use portus::ipc::netlink::Socket;
                use portus::ipc::BackendBuilder;
                use portus::ipc::Blocking;
                let b = Socket::<Blocking>::new()
                    .map(|sk| BackendBuilder { sock: sk })
                    .expect("netlink ipc initialization");
                portus::RunBuilder::new(b)
                    .default_alg(alg)
                    .run()
                    .expect("portus run");
            }
            #[cfg(not(target_os = "linux"))]
            {
                eprintln!("netlink IPC requires Linux");
                std::process::exit(1);
            }
        }
        "unix" => {
            use portus::ipc::unix::Socket;
            use portus::ipc::BackendBuilder;
            use portus::ipc::Blocking;
            let b = Socket::<Blocking>::new("portus")
                .map(|sk| BackendBuilder { sock: sk })
                .expect("unix ipc initialization");
            portus::RunBuilder::new(b)
                .default_alg(alg)
                .run()
                .expect("portus run");
        }
        other => {
            eprintln!("unsupported ipc {}", other);
            std::process::exit(1);
        }
    }
}
