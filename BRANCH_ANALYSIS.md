# Branch Analysis: `main` vs `network_dynamics`

## Branch Relationship

```
main          @ 5aaad54  "MPCC Flow Working"
network_dynamics @ 3bb1f9b  "Add MPCC congestion control with NLP/QP solvers, Portus CCP integration, and multi-flow fairness"
                           (1 commit ahead of main, 0 behind)
```

`network_dynamics` is a strict superset of `main` -- it contains everything in `main` plus **27 new/modified files** adding **3,235 lines**. No existing files were deleted or rewritten; only `.gitignore` was modified.

---

## What `main` Provides (Shared Foundation)

The `main` branch implements a complete **Model Predictive Contouring Control (MPCC)** framework for autonomous vehicle trajectory planning, along with an initial sketch of congestion-control ideas.

### Core MPC Planning Stack

| Layer | Key Files | Purpose |
|-------|-----------|---------|
| **Public API** | `pympc/__init__.py`, `factory.py`, `runner.py`, `cli.py` | `create_planner()`, `run_mpc()`, CLI entry points |
| **Planning** | `planning/planner.py`, `dynamic_models.py` | MPC loop, second-order unicycle + bicycle dynamics, RK4 integration |
| **Objectives** | `modules/objectives/contouring_objective.py`, `goal_objective.py` | Contouring/lag error minimization, goal reaching |
| **Constraints** | `modules/constraints/obstacle_constraint.py`, `linearized_constraints.py`, `contouring_constraints.py` | Scenario-based and linearized collision avoidance, road boundaries |
| **Solver** | `solver/casadi_solver.py`, `warmstart.py`, `performance_optimizations.py` | CasADi/IPOPT NLP, warm-starting, fast/balanced/safe modes |
| **Types** | `planning/types/` (state, trajectory, obstacles, path, scenario) | State, ReferencePath, Obstacle, Prediction dataclasses |
| **Utilities** | `utils/math_tools.py`, `utils.py`, `const.py` | TKSpline, Halfspace, logging, config helpers |
| **Config** | `pympc/config.py`, `pympc/registry.py`, `config/CONFIG.yml` | Dataclass config hierarchy, constraint-type registry, YAML defaults |

### Initial Network/CC Code (on `main`)

| File | Purpose |
|------|---------|
| `congestion_control/mpcc_cc.py` | MPCC-based CCA using lightweight numpy QP (BBR-inspired probing, AIMD compatibility) |
| `congestion_control/base.py` | Abstract CongestionController (AckInfo, LossInfo, CCAState) |
| `congestion_control/network_model.py` | RTTEstimator, BandwidthEstimator, DeliveryRateEstimator |
| `congestion_control/abc_cc.py`, `sprout.py`, `verus.py` | Comparison algorithm stubs |
| `network_emulation/mahimahi.py` | Mahimahi mm-link integration for cellular traces |
| `network_emulation/nimbus_ccp.py` | Nimbus CCP algorithm support |
| `network_emulation/cellular_traces.py` | Trace file management (Verizon/TMobile/ATT LTE) |

### Test & Build Infrastructure

- **pytest** with markers: `slow`, `integration`, `mahimahi`, `nimbus`, `optional_tooling`
- **Makefile**: `venv`, `test`, `test-cov`, `test-integration`, `lint`
- **pyproject.toml**: hatchling build, Python 3.10+, CasADi/NumPy/SciPy/PyYAML/Matplotlib deps
- **Documentation**: `README.md`, `CONTEXT.md`, `docs/papers/` (Sprout, Verus, TURBO, LeoCC, ABC summaries)

---

## What `network_dynamics` Adds

The `network_dynamics` branch extends the project with a **self-contained network congestion control stack** that reuses the existing MPCC planning framework. The new code is designed to be non-invasive -- it adds new files without modifying the existing vehicle planning code.

### 1. Network Dynamics Model (`planning/network_dynamics.py`)

A `NetworkDynamicsModel` subclass of `DynamicsModel` that maps network state into the vehicle MPCC state interface:

| MPCC Slot | Network Meaning | Units |
|-----------|-----------------|-------|
| `x` | Throughput | bytes/s |
| `y` | RTT | seconds |
| `psi` | Unused (heading) | radians (held at 0) |
| `v` | Queue occupancy | bytes |
| `spline` | Arc-length progress (utilization-driven) | dimensionless |

**Continuous dynamics:**
- `dtput/dt = (effective_rate - tput) / tau_tput` (EMA throughput)
- `drtt/dt = (rtt_prop + queue/bw - rtt) / tau_rtt` (RTT tracks queueing delay)
- `dq/dt = send_rate - bw` (queue fill/drain)
- Integrated via RK4, with `queue >= 0` enforced

**Key parameters:** `rtt_prop=25ms`, `tau_rtt=0.1s`, `tau_tput=0.1s`, `rate_max=50Mbps`, `q_max=500KB`

### 2. Dual Solvers (`planning/network_solver.py`)

Two solver implementations with identical cost structure:

**`NetworkMPCCSolver` (NLP)** -- Full CasADi/IPOPT nonlinear program:
- Decision variables: send rates `U[0..N-1]`, predicted states `X_tput`, `X_rtt`, `X_q`
- Dynamics constraints built as equality constraints per timestep
- State/control bounds enforced as inequality constraints
- Warm-started from previous solution

**`NetworkMPCCQPSolver` (QP)** -- Lightweight L-BFGS-B via SciPy:
- Forward-simulates trajectory in NumPy (no symbolic graph)
- Same cost function evaluated numerically
- Much faster (~sub-ms) at the cost of approximation quality
- Suitable for real-time per-RTT control

**Shared cost function:**
```
J = sum over k:
    w_contour * e_c(k)^2          -- contouring error (RTT deviation from reference curve)
  + w_lag * e_l(k)^2              -- lag error (throughput deficit)
  + w_delay * max(rtt - target, 0)^2  -- delay penalty
  - 0.1 * (tput/bw) / (rtt/rtt_prop) -- throughput/delay ratio reward
  + w_du * (U[k] - U[k-1])^2     -- smoothness
  + w_fair * (U[k] - fair_share)^2   -- multi-flow fairness (optional)
```

The **reference curve** is parametric: `Gamma(theta) = (bw * theta, rtt_prop + alpha * theta^2)` where `theta in [0,1]` is utilization. The contouring/lag errors are computed using the tangent-normal decomposition of the displacement from this curve, identically to the vehicle planner's contouring objective.

### 3. Reference Trajectory (`planning/network_reference.py`)

- `generate_network_reference()`: Creates a `ReferencePath` in throughput-RTT space with proper arc-length parameterization and TKSpline interpolation
- `AdaptiveNetworkReference`: Re-generates the reference when bandwidth estimate changes by >10%, avoiding unnecessary recomputation

### 4. Bandwidth Estimation (`planning/network_estimator.py`)

Two estimator modes:
- **EMA**: Exponential moving average of observed throughput (alpha=0.3, 10ms measurement window)
- **Sprout-style**: Bayesian Kalman filter modeling arrivals as Poisson process with Brownian motion drift. Provides probabilistic forecasts at a given percentile (5th by default) for conservative rate setting

### 5. Controller (`planning/mpcc_controller.py`)

`MPCCController`: Wraps NLP or QP solver with:
- Loss-triggered multiplicative decrease (halve rate on any loss)
- RTT-based fallback when solver fails (decrease if RTT > 3x propagation, hold if 1.5-3x, probe up if < 1.5x)

### 6. Network Problem Factory (`planning/network_problem.py`)

`NetworkMPCProblem` and `create_network_planner()`: Wires the network dynamics model into the existing `Planner` class, reusing `ContouringObjective` directly. Disables obstacle/goal modules (not meaningful for network control).

### 7. UDP Transport Layer (`network_emulation/transport/`)

**`UDPSender`**: Token-bucket paced sender with:
- Threaded send loop (rate-controlled) and ACK receive loop
- Binary protocol: `!QdQ` header (seq, timestamp, payload_size)
- ACK callback feeds into bandwidth estimator
- CSV packet logging

**`UDPReceiver`**: Simple echo server:
- Receives packets, sends ACKs with receive timestamps
- Tracks out-of-order delivery
- Standalone CLI: `python -m network_emulation.transport.receiver`

### 8. CCP/Portus Integration (`network_emulation/mpcc_cca.py`)

`MPCCFlow` / `MPCCAlgorithm`: Implements `portus.AlgBase` for kernel-level deployment:
- Datapath program reports ACKs, loss, RTT, inflight per RTT
- On each report: estimates queue from (RTT - rtt_prop) * throughput, calls `MPCCController.step()`, updates kernel Cwnd
- Supports netlink and unix IPC

### 9. Baseline Algorithms (`baselines/`)

Standardized `BaselineAlgorithm` interface with `ExperimentResult` dataclass:

| Baseline | Implementation | External Dependency |
|----------|----------------|---------------------|
| **TCP Cubic** | `cubic.py` | iperf3 + mm-link |
| **PCC Vivace** | `pcc_vivace.py` | PCC-Uspace binaries |
| **Sprout** | `sprout.py` | sproutbt2 binary |

All baselines run inside Mahimahi shells for apples-to-apples comparison.

### 10. Experiment Infrastructure (`scripts/network/`)

**`simulate_closed_loop.py`**: Runs MPCC (NLP + QP) and AIMD against synthetic bandwidth traces:
- 4 scenarios: `constant`, `step`, `cellular` (sinusoidal fading), `ramp`
- 6-panel comparison plot: rate, RTT, queue, throughput-delay space, metrics bar chart
- Measures per-step solve time

**`run_cc_experiments.py`**: Full experiment suite:
- Runs MPCC + all baselines over multiple traces/delay configurations
- Collects avg throughput, p95 RTT, median RTT, loss rate, power metric
- Generates comparison tables and plots

**`visualize_dynamics.py`**: Network dynamics visualization (phase portraits, transient response)

**`setup_baselines.sh`**: Builds Sprout, PCC-Uspace, and verifies Mahimahi

### 11. Network Configuration (`config/CONFIG_NETWORK.yml`)

Dedicated YAML config for network mode: horizon=15, timestep=50ms, EMA estimator, balanced performance, all obstacle/goal modules disabled.

### 12. Tests

| Test File | Coverage |
|-----------|----------|
| `test_network_dynamics.py` | NetworkDynamicsModel state propagation, RK4 integration, queue non-negativity |
| `test_network_reference.py` | Reference curve generation, adaptive update, arc-length parameterization |
| `test_mpcc_cca.py` | Portus CCA wrapper, datapath program, observation handling |
| `test_cc_integration.py` | End-to-end: controller + dynamics + estimator closed-loop |

---

## Architectural Comparison

### How network_dynamics Reuses the Vehicle Planning Stack

The key architectural insight is that the network dynamics model maps throughput/RTT into the same `(x, y, psi, v, spline)` state interface that `ContouringObjective` expects. This means:

1. **ContouringObjective** computes contouring/lag errors identically -- the only change is what `x` and `y` represent
2. **ReferencePath** + **TKSpline** work unchanged -- the reference curve is just in a different coordinate space
3. **Planner** orchestration works as-is via `NetworkMPCProblem`
4. **CasADi solver** handles the NLP with the same IPOPT backend

What's bypassed:
- Obstacle constraints (no collision avoidance in network domain)
- Goal objective (convergence is implicit in contouring)
- Vehicle-specific dynamics (replaced by queue/RTT dynamics)
- Multiple robot discs (not applicable)

### What's New vs What Existed

| Capability | `main` | `network_dynamics` |
|------------|--------|-------------------|
| Network dynamics model (CasADi) | -- | `NetworkDynamicsModel` with RK4 |
| NLP solver for CC | `congestion_control/mpcc_cc.py` (numpy QP) | `NetworkMPCCSolver` (CasADi/IPOPT) |
| QP solver for CC | -- | `NetworkMPCCQPSolver` (SciPy L-BFGS-B) |
| Reference trajectory | Implicit in `mpcc_cc.py` | Explicit `ReferencePath` in throughput-RTT space |
| Bandwidth estimation | `network_model.py` (RTT/BW/DeliveryRate) | `network_estimator.py` (EMA + Sprout Kalman filter) |
| Contouring cost reuse | -- | Full reuse of `ContouringObjective` |
| UDP transport | -- | Sender + Receiver with token-bucket pacing |
| CCP/Portus integration | `nimbus_ccp.py` (stub) | `mpcc_cca.py` (full `AlgBase` implementation) |
| Baseline wrappers | `abc_cc.py`, `sprout.py`, `verus.py` (in-process stubs) | `baselines/` (Cubic, Vivace, Sprout via Mahimahi) |
| Closed-loop simulation | -- | `simulate_closed_loop.py` with 4 scenarios |
| Experiment runner | `scripts/run_comparison.py` | `scripts/network/run_cc_experiments.py` |
| Multi-flow fairness | Per-flow AIMD in `mpcc_cc.py` | `w_fair * (rate - fair_share)^2` in NLP cost |

### Design Philosophy Differences

**`main`'s CC approach** (`congestion_control/mpcc_cc.py`):
- Self-contained numpy QP, does not use CasADi
- BBR-inspired probing phases (CRUISE/PROBE_UP/DRAIN)
- AIMD compatibility as hard constraint
- Designed for lightweight per-RTT execution

**`network_dynamics`' CC approach** (`planning/network_solver.py`):
- Uses the full CasADi/IPOPT NLP (or SciPy QP fallback)
- Reuses the vehicle MPCC contouring framework directly
- Multi-flow fairness as soft cost term
- Reference curve as explicit parametric path in throughput-RTT space
- Heavier solve but richer optimization structure

---

## File Inventory: New in `network_dynamics`

```
M .gitignore                               (+13 lines)

A planning/network_dynamics.py             (131 lines) -- CasADi network dynamics model
A planning/network_solver.py               (268 lines) -- NLP + QP solvers
A planning/network_problem.py              (137 lines) -- Problem setup and planner factory
A planning/network_reference.py            (90 lines)  -- Reference trajectory generation
A planning/network_estimator.py            (153 lines) -- EMA + Sprout bandwidth estimators
A planning/mpcc_controller.py              (66 lines)  -- Controller wrapper

A network_emulation/mpcc_cca.py            (133 lines) -- Portus/CCP integration
A network_emulation/transport/__init__.py  (4 lines)
A network_emulation/transport/sender.py    (170 lines) -- UDP rate-paced sender
A network_emulation/transport/receiver.py  (139 lines) -- UDP ACK echo receiver

A baselines/__init__.py                    (43 lines)  -- BaselineAlgorithm interface
A baselines/cubic.py                       (73 lines)  -- TCP Cubic via iperf3
A baselines/pcc_vivace.py                  (83 lines)  -- PCC Vivace wrapper
A baselines/sprout.py                      (79 lines)  -- Sprout wrapper

A config/CONFIG_NETWORK.yml                (61 lines)  -- Network-mode configuration

A scripts/__init__.py                      (0 lines)
A scripts/network/__init__.py              (0 lines)
A scripts/network/run_cc_experiments.py    (331 lines) -- Full experiment suite
A scripts/network/setup_baselines.sh       (79 lines)  -- Build baseline tools
A scripts/network/simulate_closed_loop.py  (267 lines) -- Closed-loop MPCC vs AIMD sim
A scripts/network/visualize_dynamics.py    (187 lines) -- Dynamics visualization

A tests/test_network_dynamics.py           (194 lines)
A tests/test_network_reference.py          (118 lines)
A tests/test_mpcc_cca.py                   (160 lines)
A tests/test_cc_integration.py             (115 lines)

A utils/cc_plotting.py                     (141 lines) -- CC-specific plotting utilities
```

**Total: 27 files, +3,235 lines**

---

## Summary

`main` provides the general-purpose MPCC trajectory planning framework plus an initial, self-contained congestion control prototype using numpy. `network_dynamics` builds on this by creating a proper network dynamics model that plugs into the existing CasADi solver and contouring objective, adds a full UDP transport layer, CCP kernel integration, baseline comparison wrappers, and a closed-loop simulation + experiment pipeline. The two CC approaches coexist: `main`'s lightweight numpy QP in `congestion_control/mpcc_cc.py` and `network_dynamics`' CasADi NLP in `planning/network_solver.py`.
