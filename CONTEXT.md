# mpcc_flow — development context (for agents and humans)

This file records the **project objective, research context, and how to run the project** on this machine. Update it when the workflow changes.

**Paper draft (source of truth for the research narrative):** `paper_state.tex` — *Model Predictive Contouring Control for Congestion Control in Time-Varying Networks* (IEEE-style draft). When this file and the paper disagree, **refresh this file** after changing the TeX.

## Project Objective

Develop **Model Predictive Contouring Control (MPCC) for end-host congestion control**: treat desirable operating points as a **reference curve in throughput–RTT space**, and optimize the sender’s rate over a finite horizon using a small network-state model. The motivating setting is **time-varying paths** (wired, cellular, LEO-inspired), as in the paper’s Mahimahi evaluation story—not only static cellular links.

The repository still contains the original **vehicle MPCC** stack (`pympc/`, `planning/`, …). Congestion control is split across two cooperating layers:

- **`planning/network_solver.py`** + **`planning/mpcc_controller.py`**: network dynamics, contouring cost, **QP (default)** or NLP solver — this is what **`third_party/portus-mpcc`** mirrors for the CCP datapath (`solver_type="qp"` in `MPCCController`).
- **`congestion_control/mpcc_cc.py`**: a **self-contained** MPCC CCA for the **Python link emulator** (`congestion_control/emulation/`). **Default** ``mpc_solver="qp"`` calls **`planning/network_solver.NetworkMPCCQPSolver`** (CasADi condensed QP, paper-aligned delay/fairness scaling). Optional ``mpc_solver="slsqp"`` keeps SciPy SLSQP on an RK4 rollout for debugging. Maps optimum to **cwnd**; used by `scripts/run_comparison.py` and `tests/test_congestion_control.py`.

### Research goals (aligned with `paper_state.tex`)

1. **MPCC formulation**: unified objective with contouring error, lag error, hinge delay penalty, Kleinrock-style throughput–delay term, rate smoothness, and optional **multi-flow fairness** penalty toward \(C/n\).

2. **Network model**: state \(x=[\hat T,\hat R,q]^\top\), control \(u=[s]\) (sending rate), continuous-time dynamics as in the paper; discretization / estimation details are documented in code (the paper still has a TODO to lock TeX to implementation).

3. **Tractable control**: the deployed story is a **condensed QP** over the joint decision vector \(Z=[s_0,\dots,s_{N-1},\,v_{\theta,0},\dots,v_{\theta,N-1}]\) of size \(2N\). The path parameter \(\theta_k\) and its progress rate \(v_{\theta,k}\) are **decision variables** (Liniger-style MPCC); \(\theta_{k+1}=\theta_k+\Delta t\,v_{\theta,k}\) with \(\theta_k\in[0,1]\), \(v_{\theta,k}\in[0,v_{\theta,\max}]\). Contouring/lag errors use the symbolic \(\theta_k\), not \(\hat T_k/C\) (the latter collapses the cost to a 1-D RTT penalty). Progress is rewarded via \(-w_\theta v_{\theta,k}\) in the stage cost.
   - **Python** (`planning/network_solver.NetworkMPCCQPSolver`): CasADi **`qrqp`** backend. Decision vector is rescaled to \((U/\text{rate\_max},\,V/v_{\theta,\max})\) so both halves are \(\mathcal O(1)\) and the Hessian is well-conditioned. Full cost Hessian is symmetrized and **spectrum-shifted to PSD** when indefinite (the quadratic residuals \(e_c^2,e_\ell^2\) can pick up concave 2nd-order contributions). **Two-point multistart** per solve — receding-horizon shift of last \(U,V\) plus an aggressive \((s=C,v_\theta=v_{\theta,\max})\) cruise — keeps one QP step from falling into the bad local minimum at \(\theta\approx\hat T/C\). Linearized constraints for \(q\le q_{\max}\), \(\hat R\le 4R^*\), \(\theta\in[0,1]\).
   - **Rust** (`third_party/portus-mpcc/src/mpcc/solver.rs`): same \(Z\), but **projected gradient with numeric finite differences** instead of an assembled QP. Quadratic barriers on the state inequalities, same two-point multistart, backtracking line search.
   - **Per-ACK solve time** on the paper matrix: **25 µs mean, 88 µs p99** over 11,292 measured solves. Warm-start across ACKs is a receding-horizon shift of \(U\), \(V\), and \(\theta_0\leftarrow\text{prev }\theta_1\).
   - `congestion_control/mpcc_cc.py` defaults to this CasADi QP (`mpc_solver="qp"`); optional `"nlp"` uses IPOPT for offline debugging, `"slsqp"` keeps a SciPy RK4-rolled program.
   - **cwnd fix** (CCP datapath): `cwnd = s*·rtt_prop` (min observed RTT), **not** `s*·rtt_now`. The latter is a positive-feedback loop (queue grows → RTT grows → cwnd grows → queue grows).
   - Build the Rust binary with `cargo build --release` inside `third_party/portus-mpcc/` after cloning `ccp-project/portus` at `../../../portus`. Integration tests expect **`scripts/network/configs/paper/{wired,cellular,leo,fairness}.yml`** for the paper matrix.

4. **Evaluation**: the paper targets **Mahimahi** scenarios (wired 50 Mbps, bundled LTE traces, LEO-style handover trace, multi-flow fairness) and kernel/user-space baselines (CUBIC, Reno, BBR, Nimbus, Sprout, Verus). **In this tree**, `congestion_control/emulation/` provides a **Python trace-driven emulator** and `scripts/run_comparison.py` runs **MPCC vs Sprout / Verus / ABC** (and variants) on synthetic and optional real Mahimahi trace files—not the full kernel CCA matrix from the paper.

5. **TCP-friendly hooks**: the Python MPCC CCA maps the optimized rate to **cwnd** and applies **multiplicative decrease on loss** (AIMD-style loss response) for interoperability with the emulation and window-based stacks; this is a transport-layer detail beyond the paper’s core MPCC objective.

6. **Legacy coexistence (design intent)**: comparisons and literature notes still cover **Sprout**, **Verus**, **ABC**, and TCP-family baselines; the paper additionally positions **Nimbus** and notes **ABC** is cited but omitted from the Mahimahi matrix without AP-side marking support.

### How the Existing MPCC Stack Maps to Congestion Control

| MPCC (vehicle) | MPCC (congestion control) |
|----------------|--------------------------|
| Vehicle state (x, y, psi, v) | Network state (cwnd, RTT, throughput, queue_delay) |
| Control inputs (acceleration, steering) | Rate adjustments (window increment/decrement, pacing rate) |
| Reference path | Target operating curve in throughput-delay space |
| Contouring error (lateral deviation) | Delay error (deviation from target delay) |
| Lag error (progress along path) | Throughput deficit (below target utilization) |
| Obstacle constraints | Congestion constraints (queue overflow, loss threshold, fairness) |
| Scenario-based constraints | Optional hard caps (queue, RTT) in the finite-horizon program |
| Goal objective | Fair-share bias via \(\ell_{\mathrm{fair}}\) when \(n>1\) |
| Horizon (H steps) | `horizon` steps at `dt_s` (default 8 × 20 ms-class steps) |
| CasADi/IPOPT solver | Vehicle stack + **network** MPCC (`planning/network_solver`); emulator CC default **QP** |

### Comparison Algorithms (Papers Analyzed)

Detailed paper summaries are in `docs/papers/`:

| Paper | Key Technique | Our Advantage |
|-------|--------------|---------------|
| **Sprout** (NSDI'13) | Bayesian forecast of Poisson link rate; window = 5th-percentile delivery forecast | Paper + repo: baseline class; repo includes an emulation-oriented implementation |
| **Verus** (SIGCOMM'15) | Learned delay profile (W,D curve); epsilon-epoch exploration; AIMD | Same: baseline + emulation in `congestion_control/verus.py` |
| **TURBO** (NINeS'26) | Utility-aware ILP for bandwidth allocation across AV services; QUIC transport | Related reading in `docs/papers/`; not a first-class baseline in `paper_state.tex` |
| **LeoCC** (SIGCOMM'25) | LEO-aware modeling and rate control | Motivation for time-varying / LEO scenarios; paper uses a **synthetic handover trace** (generator named in TeX as `leo_trace.py`—**not present in this repo yet**) |
| **ABC** (NSDI'20) | Explicit rate signaling from access point | Paper: in related work; Mahimahi matrix excludes faithful ABC without AP marking. Repo: simplified **ABC emulation** in `congestion_control/abc_cc.py` for comparisons |

### Design principles

- **Optimization-based**: finite-horizon cost over predicted states (see `congestion_control/mpcc_cc.py`).
- **Constraint-driven**: rate bounds, queue cap, and \(\hat R \le 4R^\*\) (CasADi QP/NLP in `planning/network_solver`; linearized inequalities in the QP pass).
- **Modular**: vehicle MPCC (`pympc/`, …) stays separate from **`congestion_control/`** emulation and CCAs.
- **Testable**: pytest covers emulated links and CCAs (`tests/test_congestion_control.py`). Full **mm-link + iperf3 + kernel CCA** runs are scripted under **`scripts/network/`** (`run_cc_experiments.py`, `run_paper_matrix.sh`, …) and need local **Mahimahi**, traces, and (for MPCC on the wire) a built **`third_party/portus-mpcc`** binary; see `network_emulation/README.md` and `third_party/README.md`.

### Implementation status vs `paper_state.tex` (verified 2026-04-22)

| Topic | Paper (`paper_state.tex`) | This repository |
|--------|---------------------------|-----------------|
| State / dynamics / ref. curve / \(e_c,e_\ell\) / stage cost | Yes (Sec. methodology) | **Matches** in `congestion_control/mpcc_cc.py` (comments cite equations) |
| Fairness term \(\ell_{\mathrm{fair}}\), constraints | Yes | **Matches** (same structure) |
| Horizon / \(\Delta t\) | \(N\in\{8,10,12\}\), 20 ms control interval in results | Default **`horizon=8`**, **`dt_s=0.02`**; YAMLs under `scripts/network/configs/paper/` tune planner horizons and weights |
| **Solver** | QP over \(Z=[s;v_\theta]\) (size \(2N\)), warm-started per ACK, 25 µs mean / 88 µs p99 deployed | **`planning/network_solver.NetworkMPCCQPSolver`** (CasADi `qrqp`, spectrum-shifted Hessian, two-point multistart) + **`third_party/portus-mpcc`** (projected gradient with barriers + multistart; `src/mpcc/solver.rs`). **`congestion_control/mpcc_cc.py`** defaults to the **same CasADi QP**; optional **`mpc_solver="nlp"`** (IPOPT) / **`"slsqp"`** (SciPy RK4) |
| **Deployment** | Rust `portus-mpcc` / CCP | Crate at **`third_party/portus-mpcc/`**; optional until `cargo build --release` |
| **LEO trace** | handover trace generator | **`network_emulation/leo_trace.py`** (TeX names `leo_trace.py` at repo root; same utility via `python -m network_emulation.leo_trace`) |
| **Paper YAML weights** | per-scenario weights | **`scripts/network/configs/paper/{wired,cellular,leo,fairness}.yml`** |
| Baseline matrix | CUBIC, Reno, BBR, Nimbus, Sprout, Verus (kernel / authors’ binaries) | **Mahimahi matrix:** `scripts/network/` + kernel CCAs. **Python emulation:** `congestion_control/emulation` + `scripts/run_comparison.py` (**Sprout, Verus, ABC**, MPCC) |

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
