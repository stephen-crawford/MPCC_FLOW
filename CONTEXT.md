# mpcc_flow — development context (for agents and humans)

This file records the **project objective, research context, and how to run the project** on this machine. Update it when the workflow changes.

## Project Objective

Design a **Model Predictive Contouring Control (MPCC) algorithm for congestion control on cellular networks**. The core idea is to repurpose the MPCC trajectory-optimization framework -- originally for robot/vehicle motion planning -- as a congestion controller that regulates sending rate over time-varying wireless links.

### Research Goals

1. **MPCC as congestion control**: formulate the sending-rate control problem as contouring control along a reference path in throughput-delay space, with the MPC horizon optimizing future send rates subject to network dynamics constraints.

2. **Sliding-window feedback with AIMD**: the algorithm must operate using a sliding congestion window as the primary control variable, with **additive increase** during normal operation and **multiplicative decrease** on congestion signals (loss, ECN, delay threshold). This ensures compatibility with the AIMD paradigm that underpins TCP fairness.

3. **Provable convergence**: the algorithm must come with formal convergence guarantees -- showing that the sending rate converges to a stable operating point that maximizes throughput while bounding queuing delay, and that multiple concurrent flows converge to a fair allocation.

4. **Coexistence with legacy CCAs**: the strategy must handle both access points running the MPCC algorithm and access points running traditional/legacy congestion control. Specifically, it must fairly share bandwidth when competing with:
   - **ABC** (A Simple Explicit Congestion Controller for Wireless Networks) -- an explicit, access-point-based scheme [Goyal et al., NSDI 2020]
   - **Sprout** (Stochastic Forecasts Achieve High Throughput and Low Delay over Cellular Networks) -- a receiver-side forecast-based protocol [Winstein et al., NSDI 2013]
   - **Verus** (Adaptive Congestion Control for Unpredictable Cellular Networks) -- a delay-profile-based adaptive protocol [Zaki et al., SIGCOMM 2015]
   - Standard TCP variants (Cubic, BBR, Vegas, Reno)
   - Network-layer schemes (CoDel, ECN)

5. **Cellular-network-aware**: the algorithm should account for the unique properties of cellular links -- bursty scheduling, rapidly varying capacity, stochastic packet loss unrelated to congestion, deep per-user queues at base stations, and asymmetric uplink/downlink behavior.

### How the Existing MPCC Stack Maps to Congestion Control

| MPCC (vehicle) | MPCC (congestion control) |
|----------------|--------------------------|
| Vehicle state (x, y, psi, v) | Network state (cwnd, RTT, throughput, queue_delay) |
| Control inputs (acceleration, steering) | Rate adjustments (window increment/decrement, pacing rate) |
| Reference path | Target operating curve in throughput-delay space |
| Contouring error (lateral deviation) | Delay error (deviation from target delay) |
| Lag error (progress along path) | Throughput deficit (below target utilization) |
| Obstacle constraints | Congestion constraints (queue overflow, loss threshold, fairness) |
| Scenario-based constraints | Stochastic bandwidth prediction scenarios |
| Goal objective | Convergence to fair-share rate |
| Horizon (H steps) | Prediction window (H future RTTs) |
| CasADi/IPOPT solver | Same or lighter-weight QP for real-time feasibility |

### Comparison Algorithms (Papers Analyzed)

Detailed paper summaries are in `docs/papers/`:

| Paper | Key Technique | Our Advantage |
|-------|--------------|---------------|
| **Sprout** (NSDI'13) | Bayesian forecast of Poisson link rate; window = 5th-percentile delivery forecast | MPCC optimizes over horizon; handles multi-flow; formal convergence |
| **Verus** (SIGCOMM'15) | Learned delay profile (W,D curve); epsilon-epoch exploration; AIMD | MPCC uses predictive dynamics model instead of reactive profile; optimization-derived deltas |
| **TURBO** (NINeS'26) | Utility-aware ILP for bandwidth allocation across AV services; QUIC transport | MPCC provides continuous (not step-function) utility; incorporates network dynamics model |
| **LeoCC** (SIGCOMM'25) | Reconfiguration-aware network model; dual BW estimator; robust rate controller for LEO | MPCC generalizes event-aware adaptation; optimizes over horizon; provides convergence proof |
| **ABC** (NSDI'20) | Explicit rate signaling from access point; simple target-rate formula | Our MPCC works end-to-end (no AP modification) AND coexists with ABC-equipped APs |

### Design Principles

- **Optimization-based**: replace heuristic window-adjustment rules with a solved optimization problem (the MPC) that accounts for predicted network evolution
- **Constraint-driven safety**: use the MPC constraint mechanism to enforce hard bounds (max queuing delay, min throughput, fairness bounds) rather than hoping the heuristic stays safe
- **Modular**: reuse the existing pympc module/constraint/objective architecture -- swap vehicle dynamics for network dynamics, contouring cost for throughput-delay cost
- **Testable via Mahimahi**: validate using real cellular traces (already in the repo) via Mahimahi's mm-link, with optional Nimbus/CCP integration for kernel-bypass deployment

## Repository root

```
/home/stephen/mpcc_flow
```

All commands below assume `cd` to this directory unless noted.

## Python environment (required for tests)

Use a **local venv** (PEP 668 / system Python may block global `pip install`).

```bash
cd /home/stephen/mpcc_flow
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[dev]"
```

- **Editable install** pulls in: `pympc`, `planning`, `modules`, `solver`, `utils`, `network_emulation`.
- **Dev extras** include: `pytest`, `pytest-cov`, `pytest-xdist`, `ruff`, `black`, `mypy`, `pre-commit`.

Re-activate in new shells:

```bash
source /home/stephen/mpcc_flow/.venv/bin/activate
```

## Run the full test suite

From repo root (after `make venv` once):

```bash
cd /home/stephen/mpcc_flow
make test
```

Or manually:

```bash
source .venv/bin/activate
pytest tests/
pytest tests/ -q
```

With coverage (optional):

```bash
pytest tests/ --cov=pympc --cov=planning --cov=modules --cov=solver --cov=network_emulation --cov=utils --cov-report=term-missing
```

**What the tests need:** Core unit tests need **Python deps only**. **Integration / e2e** tests use real `mm-link`, traces, and Nimbus when installed; they **skip** if something is missing (except you may see **1 skip** for `iperf3` until `sudo apt install iperf3`).

**Last verified run:** 59 tests (`tests/test_network_stack_e2e.py` adds mm-link `/bin/true` e2e, `third_party` path checks, optional iperf3/CCP).

### `third_party/` (Mahimahi + Nimbus inside the repo)

Symlinks (not copies) so checkouts stay in one place:

```bash
./third_party/setup_symlinks.sh   # expects ../mahimahi and ../nimbus next to mpcc_flow
```

Path order: **`MAHIMAHI_ROOT` / `NIMBUS_ROOT` env** → **`third_party/`** → **`~/mahimahi`**, **`~/nimbus`**.

Shell smoke test (mm-link e2e + pytest subset):

```bash
./scripts/network/verify_stack.sh
```

### Mahimahi / Nimbus integration tests

These run **automatically** with `pytest tests/` or `make test`. They **skip** if tools are missing, and **execute** when present:

| Test area | Needs | Env / action |
|-----------|--------|----------------|
| `mm-link` usage | Built/installed `mm-link` | `PATH` or `MAHIMAHI_MM_LINK` |
| `mm-graph` | Script under `MAHIMAHI_ROOT/scripts` | `MAHIMAHI_ROOT` or `third_party/mahimahi` |
| Cellular traces | `*.up` / `*.down` pairs | `…/traces` |
| `mm-link` argv | Above + traces | Same |
| **mm-link e2e** | `mm-link U D -- /bin/true` | `tests/test_network_stack_e2e.py` |
| **third_party defaults** | Symlinks present | Unset `MAHIMAHI_ROOT`/`NIMBUS_ROOT` to test |
| Nimbus `--help` | `target/release/nimbus` | `cargo build --release` |
| Nimbus source layout | `Cargo.toml` + `src/lib.rs` | `NIMBUS_ROOT` or `third_party/nimbus` |
| `iperf3` (optional) | Throughput scripts | Skipped in pytest if not installed |
| CCP module (optional) | Live Nimbus on TCP | Skipped if `lsmod` has no `ccp` |

Run only integration tests:

```bash
pytest tests/test_mahimahi_nimbus_integration.py -v
pytest -m mahimahi -v
pytest -m nimbus -v
```

Force-skip Nimbus binary test (e.g. CI without Rust build):

```bash
SKIP_NIMBUS_TESTS=1 pytest tests/
```

## CLI entry points (after `pip install -e .`)

```bash
pympc --help
mpcc-network check
```

`mpcc-network` resolves **Mahimahi** / **Nimbus** paths via env (`MAHIMAHI_ROOT`, `NIMBUS_ROOT`); see `network_emulation/README.md`.

## System / tooling outside Python

| Tool | Role | Needed for pytest? |
|------|------|--------------------|
| Mahimahi (`mm-link`, traces) | Cellular emulation | Only integration/e2e tests |
| Rust (`rustup`, `cargo`) | Build Nimbus / `cxxbridge-cmd` | Only Nimbus tests |
| `iperf3` | Network experiments | Optional pytest (skip if missing) |
| CCP kernel module | Live Nimbus | Optional pytest (skip if missing) |

## Config file location

`utils/utils.py` loads **`/home/stephen/mpcc_flow/config/CONFIG.yml`** (relative to repo). If you move the repo, paths still resolve from the package layout.

## Makefile targets

| Target | Action |
|--------|--------|
| `make venv` | Create `.venv` and `pip install -e ".[dev]"` |
| `make test` | Run `pytest tests/` |
| `make test-integration` | `tests/test_mahimahi_nimbus_integration.py` |
| `make test-e2e` | Integration + `tests/test_network_stack_e2e.py` |
| `make test-cov` | Tests + coverage report |
| `make lint` | `ruff check` on packages + tests |

## Troubleshooting

- **`pytest` not found:** Activate `.venv` or use `/home/stephen/mpcc_flow/.venv/bin/pytest`.
- **Import errors:** Re-run `pip install -e ".[dev]"` from repo root.
- **CasADi install issues:** Use the same Python as the venv (`which python`).
