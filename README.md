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


# RL ENV: mahimahi-gym

`mahimahi-gym` provides four Gymnasium scenario environments for congestion
control experiments. Each scenario can run as a fast in-process simulator for
local PPO training, or through a real Mahimahi `mm-link` subprocess controller
for network-emulation experiments.

The current code is intentionally centered on these four registered scenarios:

| ID | Class | Purpose | Action |
| --- | --- | --- | --- |
| `MahiWiredBottleneck-v0` | `WiredBottleneckEnv` | Stable wired bottleneck with fixed bandwidth and low delay. | one send rate in Mbps |
| `MahiCellularBursty-v0` | `CellularBurstyEnv` | Cellular-style bursty capacity and delay variation. | one send rate in Mbps |
| `MahiLeoSatellite-v0` | `LeoSatelliteEnv` | Abrupt bandwidth and delay shifts inspired by LEO links. | one send rate in Mbps |
| `MahiMultiFlowFairness-v0` | `MultiFlowFairnessEnv` | Multi-flow fairness over a shared bottleneck. | one send rate per flow |

## Repository Layout

The main source files are:

```text
mahimahi_gym/scenarios.py      # four scenario environments and Mahimahi scenario wrappers
mahimahi_gym/mm_link.py        # JSON-lines subprocess adapter for real mm-link runs
examples/train_ppo.py          # Stable-Baselines3 PPO training script for the four scenarios
examples/mm_link_json_controller.py
                              # example controller used by real-Mahimahi mode
tests/                         # scenario and subprocess-adapter tests
docs/environment_designs.tex   # design notes for the environments
```

There is no JAX trainer or legacy mobile-trace environment in the current code.

## Install

For development and tests:

```bash
uv pip install -e ".[dev]"
pytest
```

For PPO training:

```bash
uv pip install -e ".[dev,rl]"
```

The Python package itself depends only on `gymnasium` and `numpy`. PPO training
adds `stable-baselines3`, `torch`, and `matplotlib`.

If you need the bundled Mahimahi source checkout, initialize the submodule after
cloning:

```bash
git submodule update --init --recursive mahimahi
```

## Quick Gym Usage

Importing `mahimahi_gym` registers the four Gymnasium environments:

```python
import gymnasium as gym
import mahimahi_gym

env = gym.make(
    "MahiWiredBottleneck-v0",
    episode_steps=100,
)

obs, info = env.reset(seed=7)
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

By default, `gym.make(...)` uses the in-process simulator. Pass
`use_mahimahi=True` to run the same scenario shape through the real `mm-link`
adapter.

## PPO Training

Use `examples/train_ppo.py` for all four scenarios.

Fast simulator mode:

```bash
uv run --extra rl python examples/train_ppo.py \
  --env-id MahiWiredBottleneck-v0 \
  --no-use-mahimahi \
  --episode-steps 100 \
  --timesteps 50000
```

Other simulator examples:

```bash
uv run --extra rl python examples/train_ppo.py --env-id MahiCellularBursty-v0 --no-use-mahimahi --episode-steps 20 --timesteps 50000
uv run --extra rl python examples/train_ppo.py --env-id MahiLeoSatellite-v0 --no-use-mahimahi --episode-steps 20 --timesteps 50000
uv run --extra rl python examples/train_ppo.py --env-id MahiMultiFlowFairness-v0 --no-use-mahimahi --n-flows 3 --episode-steps 100 --timesteps 50000
```

Multi-seed PPO with a 95% confidence interval:

```bash
uv run --extra rl python examples/train_ppo.py \
  --env-id MahiWiredBottleneck-v0 \
  --no-use-mahimahi \
  --episode-steps 100 \
  --timesteps 50000 \
  --seeds 1,2,3,4,5
```

The script writes:

- trained models under `models/`
- per-seed monitor CSV files under `runs/`
- return plots under `plots/`
- a seed summary CSV at `runs/ppo_seed_summary.csv`

To train with the MPCC-style reward instead of the default throughput-delay-loss
reward:

```bash
uv run --extra rl python examples/train_ppo.py \
  --env-id MahiWiredBottleneck-v0 \
  --no-use-mahimahi \
  --reward-mode mpcc \
  --episode-steps 100 \
  --timesteps 50000
```

The MPCC weights can be tuned from the CLI, for example:

```bash
uv run --extra rl python examples/train_ppo.py \
  --env-id MahiMultiFlowFairness-v0 \
  --no-use-mahimahi \
  --reward-mode mpcc \
  --mpcc-contour-weight 0.001 \
  --mpcc-lag-weight 0.001 \
  --mpcc-delay-weight 0.0001 \
  --mpcc-power-weight 1.0 \
  --mpcc-smoothing-weight 0.1 \
  --mpcc-fairness-weight 0.01
```

## Scenario API

### Single-Flow Scenarios

`MahiWiredBottleneck-v0`, `MahiCellularBursty-v0`, and
`MahiLeoSatellite-v0` use one action:

```text
[send_rate_mbps]
```

Their normalized observation has seven values:

```text
send_rate, throughput, capacity, propagation_delay, queue_delay, loss, queue_occupancy
```

The default single-flow reward is:

```text
throughput_mbps - delay_penalty * rtt_ms - loss_penalty * loss_fraction
```

With `reward_mode="mpcc"`, the environment tracks a reference curve in the
throughput-RTT plane:

```text
Gamma(theta) = (C theta, R0 + alpha theta^2)
```

The Gym reward is the negative MPCC stage cost:

```text
-(w_c e_c^2 + w_l e_l^2 + w_d [R_hat - R*]_+^2
  + w_u (s_k - s_{k-1})^2 / s_max^2
  + w_loss loss_fraction^2
  - w_p ((T_hat / C) / (R_hat / R0)))
```

The `info` dictionary includes MPCC diagnostics such as `mpcc_contour_error`,
`mpcc_lag_error`, `mpcc_power`, `mpcc_smoothing_penalty`, and
`mpcc_stage_cost`.

### Multi-Flow Fairness Scenario

`MahiMultiFlowFairness-v0` uses one action per flow:

```text
[rate_flow_1, rate_flow_2, ..., rate_flow_N]
```

Its normalized observation contains:

```text
per-flow send rates, per-flow throughputs, capacity, queue occupancy, loss, Jain fairness
```

The reward combines total throughput, Jain fairness, queueing delay, and loss:

```text
sum(throughputs) + fairness_weight * Jain_fairness
  - delay_penalty * queue_delay_ms
  - loss_penalty * loss_fraction
```

With `reward_mode="mpcc"`, the multi-flow environment uses the same
throughput-RTT curve objective on aggregate throughput and adds a fair-share
penalty:

```text
w_f sum_i (s_i - C / n)^2
```

## Real Mahimahi Mode

When a scenario is constructed with `use_mahimahi=True`, it delegates to the
subprocess adapter in `mahimahi_gym/mm_link.py`. The adapter launches:

```bash
mm-link uplink.trace downlink.trace -- <controller command>
```

The default controller command is:

```bash
python examples/mm_link_json_controller.py
```

The environment sends one JSON action per step to the controller's stdin:

```json
{"type": "action", "step": 0, "send_rates_mbps": [4.0]}
```

The controller must print one JSON measurement per action to stdout:

```json
{"throughput_mbps": 3.8, "rtt_ms": 45.0, "queue_delay_ms": 5.0, "loss_fraction": 0.0, "capacity_mbps": 10.0}
```

The controller may also report fields such as `throughputs_mbps`,
`jain_fairness`, `reward`, or `done`.

### Default Real-Mahimahi Traces

| ID | Default uplink trace | Default downlink trace |
| --- | --- | --- |
| `MahiWiredBottleneck-v0` | `mahimahi/12mbps.trace` | `mahimahi/12mbps.trace` |
| `MahiCellularBursty-v0` | `mahimahi/traces/TMobile-LTE-driving.up` | `mahimahi/traces/TMobile-LTE-driving.down` |
| `MahiLeoSatellite-v0` | `mahimahi/traces/Verizon-LTE-short.up` | `mahimahi/traces/Verizon-LTE-short.down` |
| `MahiMultiFlowFairness-v0` | `mahimahi/12mbps.trace` | `mahimahi/12mbps.trace` |

Real Mahimahi example:

```bash
uv run --extra rl python examples/train_ppo.py \
  --env-id MahiCellularBursty-v0 \
  --episode-steps 20 \
  --timesteps 50000 \
  --controller-cmd "python examples/mm_link_json_controller.py"
```

To override traces:

```bash
uv run --extra rl python examples/train_ppo.py \
  --env-id MahiWiredBottleneck-v0 \
  --uplink-trace path/to/uplink.trace \
  --downlink-trace path/to/downlink.trace
```

Run real Mahimahi training from a normal host shell, not from inside an
interactive `mm-link` shell. The adapter places the controller process inside
the Mahimahi network namespace.

Mahimahi trace reminder: `mm-link` traces are packet-delivery schedules. Each
line is a delivery opportunity timestamp for an MTU-sized packet, not a simple
`duration_ms capacity_mbps` row.

## Installing Mahimahi

`mm-link` depends on Linux networking features such as network namespaces, so
Ubuntu, a Linux server, or a Linux VM is the recommended setup.

On Ubuntu, first try:

```bash
sudo apt update
sudo apt install mahimahi
```

Optional cellular traces:

```bash
sudo apt install mahimahi-traces
```

Verify the install:

```bash
which mm-link
mm-link --help
```

If `mm-link` asks for IP forwarding:

```bash
sudo sysctl -w net.ipv4.ip_forward=1
```

If the package is unavailable, build Mahimahi from source:

```bash
sudo apt update
sudo apt install \
  make \
  autoconf \
  libtool \
  iproute2 \
  iptables \
  dnsmasq \
  apache2 \
  apache2-dev \
  protobuf-compiler \
  pkg-config \
  libssl-dev \
  libxcb-present-dev \
  libpangomm-2.48-dev

cd mahimahi
./autogen.sh
./configure
make
sudo make install
```

If `mm-link` exits with `mm-link: needs to be installed setuid root`, repair the
installed binary:

```bash
sudo chown root:root /usr/local/bin/mm-link
sudo chmod 4755 /usr/local/bin/mm-link
```

Quick smoke test:

```bash
seq 0 1000 > 12mbps.trace
mm-link 12mbps.trace 12mbps.trace -- /bin/true
```

## Development Checks

Run the current test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest
```

The tests cover the four scenario environments and the JSON-lines subprocess
adapter used by real Mahimahi mode.

## Next Integration Step

The scenario wrappers already bridge PPO to real Mahimahi. The remaining
project-specific work is to replace `examples/mm_link_json_controller.py` with a
controller that launches your actual sender/receiver, applies the chosen rate,
and reports measured throughput, RTT, queueing delay, and loss as JSON.
