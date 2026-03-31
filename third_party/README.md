# Third-party tooling (Mahimahi + Nimbus)

Sources stay **outside** this repo; we **link** them in so paths default to `mpcc_flow/third_party/...`.

## Layout (recommended)

Siblings under the same parent directory:

```text
parent/
  mpcc_flow/          ← this repository
  mahimahi/           ← Mahimahi checkout (build + install mm-link)
  nimbus/             ← Nimbus (ccp_nimbus) Rust workspace
```

From `mpcc_flow`:

```bash
./third_party/setup_symlinks.sh
```

This creates **relative** symlinks:

- `third_party/mahimahi` → `../../mahimahi`
- `third_party/nimbus` → `../../nimbus`

If your checkouts live elsewhere, set:

- `MAHIMAHI_ROOT`
- `NIMBUS_ROOT`

(or adjust the symlinks by hand).

## Path resolution (`network_emulation.paths`)

1. `MAHIMAHI_ROOT` / `NIMBUS_ROOT` if set.
2. Else `mpcc_flow/third_party/mahimahi` or `third_party/nimbus` if that path exists.
3. Else `~/mahimahi` and `~/nimbus`.

## Build reminders

- **Mahimahi:** `./autogen.sh && ./configure && make && sudo make install` (see upstream README).
- **Nimbus:** `cargo build --release` → `target/release/nimbus`.
- **Live CCP:** kernel/datapath per [CCP guide](https://ccp-project.github.io/guide) (optional for pytest; required for real TCP control).

## Verify

```bash
./scripts/network/verify_stack.sh
# or
make test-integration
.venv/bin/pytest tests/test_network_stack_e2e.py -v
```
