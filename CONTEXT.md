# mpcc_flow — development context (for agents and humans)

This file records **how to run the project and tests** on this machine. Update it when the workflow changes.

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
