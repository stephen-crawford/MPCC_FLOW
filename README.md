# mpcc_flow

A **minimal Python Model Predictive Contouring Control (MPCC)** stack: second-order unicycle dynamics, reference path contouring, goal tracking, and obstacle avoidance via **scenario** (sampled half-spaces) or **linearized** half-space constraints. The CasADi/IPOPT backend solves the nonlinear program each step.

Use this repo as a **starting point for experiments**—for example, swapping the vehicle model or cost for **network flow / rate-control fairness** tests while keeping the same MPC shell.

## Install

```bash
cd mpcc_flow
pip install -e .
```

Requires Python 3.10+, CasADi, NumPy, SciPy, PyYAML, Matplotlib.

## Quick API

```python
from pympc import create_default_config, create_planner, run_mpc
import numpy as np

config = create_default_config("scenario")  # or "linearized"
# ... build initial_state, reference_path, obstacles, goal ...
planner = create_planner(initial_state, reference_path, obstacles, goal, config)
result = run_mpc(planner)
```

Configuration can also be loaded with `pympc.config.load_config("path/to.yml")` or environment variables (`PYMPC_PLANNER_HORIZON`, etc.).

## Layout

| Path | Role |
|------|------|
| `pympc/` | Public API (`create_planner`, `run_mpc`, config, logging) |
| `planning/` | State, dynamics, planner loop |
| `modules/` | Objectives (goal, contouring) and constraints |
| `solver/` | CasADi NLP formulation |
| `utils/` | Math helpers, shared config read |
| `config/CONFIG.yml` | Default YAML merged with code defaults |
| `network_emulation/` | Mahimahi + Nimbus paths, cellular traces, `mm-link` helpers |
| `scripts/network/` | Example shell scripts (e.g. iperf under `mm-link`) |
| `third_party/` | Symlinks to Mahimahi + Nimbus (`setup_symlinks.sh`) |
| `third_party/README.md` | Path resolution and build notes |
| `scripts/network/verify_stack.sh` | Shell smoke test (mm-link e2e + pytest subset) |

## Cellular links & congestion-control-style experiments

Experiments use **Mahimahi** (`mm-link` with real **cellular traces** from your checkout—Verizon/TMobile/ATT LTE, etc.) for the bottleneck, and optionally **Nimbus** as a **CCP** congestion-control algorithm (Rust binary built from your Nimbus repo; see [CCP guide](https://ccp-project.github.io/guide)).

Defaults expect checkouts at `~/mahimahi` and `~/nimbus`. Override with `MAHIMAHI_ROOT` and `NIMBUS_ROOT`.

```bash
mpcc-network check
mpcc-network list-traces
mpcc-network mm-link-cmd Verizon-LTE-short iperf3 -c 10.0.0.2 -t 30
# See network_emulation/README.md and scripts/network/cellular_iperf_experiment.sh
```

## Tests

```bash
make test
# or: source .venv/bin/activate && pytest tests/
```

Developer setup, venv, coverage, and **Mahimahi/Nimbus integration tests** are in **`CONTEXT.md`**. Run `make test-integration` to exercise only those checks.

## Origin

Structured after the [TUD-AMR `mpc_planner`](https://github.com/tud-amr/mpc_planner) ideas, reduced to the pieces needed for MPCC-style trajectory optimization in Python.
