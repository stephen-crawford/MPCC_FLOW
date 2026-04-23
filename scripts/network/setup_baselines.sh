#!/usr/bin/env bash
# Build every external tool needed to run the MPCC paper's Mahimahi
# experiments from a fresh Linux box. Run this ONCE on the host that will
# actually execute the experiments — not on a shared HPC login node.
#
# Handles:
#   mahimahi          - mm-link/mm-delay emulator + trace format (C++)
#   iperf3            - stock throughput client/server (apt)
#   ccp-kernel        - kernel CCP datapath module (from nimbus-measurement)
#   portus            - Rust CCP user-space runtime (library, shared with
#                       nimbus + portus-mpcc)
#   portus-mpcc       - our Rust MPCC CCP binary (third_party/portus-mpcc/)
#   nimbus            - reference Nimbus CCP binary (from nimbus-measurement)
#   sprout (sproutbt2) - NSDI'13 sprout reference implementation
#   verus             - SIGCOMM'15 Verus reference implementation
#   pyportus          - Python binding to portus (optional fallback for MPCC)
#
# Environment overrides:
#   MAHIMAHI_ROOT, NIMBUS_MEASUREMENT_ROOT, PORTUS_ROOT, SPROUT_ROOT, VERUS_ROOT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"
mkdir -p "$THIRD_PARTY"

MAHIMAHI_ROOT="${MAHIMAHI_ROOT:-$(dirname "$REPO_ROOT")/mahimahi}"
NIMBUS_MEASUREMENT_ROOT="${NIMBUS_MEASUREMENT_ROOT:-$(dirname "$REPO_ROOT")/nimbus-measurement}"
PORTUS_ROOT="${PORTUS_ROOT:-$(dirname "$REPO_ROOT")/portus}"
SPROUT_ROOT="${SPROUT_ROOT:-$THIRD_PARTY/sprout}"
VERUS_ROOT="${VERUS_ROOT:-$THIRD_PARTY/verus}"

echo "== MPCC baselines setup =="
echo "  REPO_ROOT:               $REPO_ROOT"
echo "  MAHIMAHI_ROOT:           $MAHIMAHI_ROOT"
echo "  NIMBUS_MEASUREMENT_ROOT: $NIMBUS_MEASUREMENT_ROOT"
echo "  PORTUS_ROOT:             $PORTUS_ROOT"
echo "  SPROUT_ROOT:             $SPROUT_ROOT"
echo "  VERUS_ROOT:              $VERUS_ROOT"
echo

# ---------------------------------------------------------------------------
# 0. apt packages
# ---------------------------------------------------------------------------
if command -v apt-get &>/dev/null; then
    echo "--- apt packages ---"
    sudo apt-get update -qq
    sudo apt-get install -y \
        build-essential autoconf automake libtool pkg-config \
        libboost-all-dev libssl-dev libxcb-present-dev libcairo2-dev \
        libpango1.0-dev iptables dnsmasq-base apache2-dev apache2-utils \
        apache2-bin debhelper iperf iperf3 libprotobuf-dev protobuf-compiler \
        git curl python3 python3-pip python3-venv \
        libpcap-dev libglib2.0-dev libevent-dev libnl-3-dev libnl-genl-3-dev
fi

# ---------------------------------------------------------------------------
# 1. Rust
# ---------------------------------------------------------------------------
if ! command -v cargo &>/dev/null; then
    echo "--- installing Rust via rustup ---"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
else
    echo "Rust already installed: $(rustc --version)"
fi
export PATH="$HOME/.cargo/bin:$PATH"

# cxxbridge-cmd — required by the nimbus-measurement mahimahi fork's
# Rust-C++ queue FFI (configure step fails otherwise).
if ! command -v cxxbridge &>/dev/null; then
    echo "--- installing cxxbridge-cmd ---"
    cargo install cxxbridge-cmd
fi

# ---------------------------------------------------------------------------
# 2. portus (CCP runtime) — library only
# ---------------------------------------------------------------------------
if [ ! -d "$PORTUS_ROOT" ]; then
    echo "--- cloning portus ---"
    git clone https://github.com/ccp-project/portus.git "$PORTUS_ROOT"
fi
(cd "$PORTUS_ROOT" && cargo build --release)

# ---------------------------------------------------------------------------
# 3. mahimahi — nimbus-measurement ships a fork with trace-driven queues
# ---------------------------------------------------------------------------
if [ ! -d "$MAHIMAHI_ROOT" ]; then
    echo "--- cloning mahimahi ---"
    git clone https://github.com/ravinet/mahimahi.git "$MAHIMAHI_ROOT"
fi
if ! command -v mm-link &>/dev/null; then
    echo "--- building mahimahi ---"
    pushd "$MAHIMAHI_ROOT"
    ./autogen.sh
    ./configure
    make -j"$(nproc)"
    sudo make install
    popd
fi
# Mahimahi requires kernel IP forwarding.
sudo sysctl -w net.ipv4.ip_forward=1 || true

# Link into third_party for the Python helpers.
ln -sfn "$MAHIMAHI_ROOT" "$THIRD_PARTY/mahimahi"

# ---------------------------------------------------------------------------
# 4. nimbus-measurement (CCP kernel module + Nimbus + empirical-traffic-gen)
# ---------------------------------------------------------------------------
if [ ! -d "$NIMBUS_MEASUREMENT_ROOT" ]; then
    echo "--- cloning nimbus-measurement ---"
    git clone https://github.com/ccp-project/nimbus-measurement.git "$NIMBUS_MEASUREMENT_ROOT"
fi
pushd "$NIMBUS_MEASUREMENT_ROOT"
git submodule update --init --recursive

# 4a. ccp-kernel
if [ -d ccp-kernel ]; then
    echo "--- building ccp-kernel ---"
    (cd ccp-kernel && make)
fi

# 4b. Nimbus (CCP user-space binary)
if [ -d nimbus ]; then
    echo "--- building nimbus ---"
    (cd nimbus && cargo build --release)
fi

# 4c. empirical-traffic-gen
if [ -d empirical-traffic-gen ]; then
    echo "--- building empirical-traffic-gen ---"
    (cd empirical-traffic-gen && make)
fi
popd

ln -sfn "$NIMBUS_MEASUREMENT_ROOT/nimbus" "$THIRD_PARTY/nimbus" || true

# ---------------------------------------------------------------------------
# 5. portus-mpcc (our MPCC CCP binary)
# ---------------------------------------------------------------------------
MPCC_CRATE="$THIRD_PARTY/portus-mpcc"
if [ -d "$MPCC_CRATE" ]; then
    echo "--- building portus-mpcc ---"
    (cd "$MPCC_CRATE" && cargo build --release)
    echo "    mpcc_cca -> $MPCC_CRATE/target/release/mpcc_cca"
fi

# ---------------------------------------------------------------------------
# 6. Sprout — sproutbt2 reference binary
# ---------------------------------------------------------------------------
if [ ! -d "$SPROUT_ROOT" ]; then
    echo "--- cloning sprout ---"
    git clone https://github.com/keithw/alfalfa.git "$SPROUT_ROOT"
fi
if [ ! -f "$SPROUT_ROOT/src/examples/sproutbt2" ]; then
    echo "--- building sprout ---"
    pushd "$SPROUT_ROOT"
    ./autogen.sh
    ./configure
    make -j"$(nproc)" || echo "sprout build failed — protobuf / boost versions vary; see sprout README"
    popd
fi

# ---------------------------------------------------------------------------
# 7. Verus — verus_client / verus_server
# ---------------------------------------------------------------------------
if [ ! -d "$VERUS_ROOT" ]; then
    echo "--- cloning verus ---"
    git clone https://github.com/yzaki/verus.git "$VERUS_ROOT" || \
        git clone https://github.com/aliceliu/Verus.git "$VERUS_ROOT" || \
        echo "WARNING: could not clone verus. Check the ocaml/boost dependencies before retrying."
fi
if [ -d "$VERUS_ROOT" ]; then
    echo "--- building verus ---"
    (cd "$VERUS_ROOT" && make -j"$(nproc)" || echo "verus build failed — see Verus README")
fi

# ---------------------------------------------------------------------------
# 8. pyportus (optional Python fallback for the MPCC CCP algorithm)
# ---------------------------------------------------------------------------
if [ -d "$PORTUS_ROOT/python" ]; then
    if ! python3 -c "import pyportus" 2>/dev/null; then
        echo "--- building pyportus (Python fallback) ---"
        pushd "$PORTUS_ROOT/python"
        if ! command -v maturin &>/dev/null; then
            python3 -m pip install --user maturin
            export PATH="$HOME/.local/bin:$PATH"
        fi
        maturin develop --release || echo "pyportus build failed — optional, Rust binary is preferred"
        popd
    fi
fi

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
echo
echo "== Verification =="
echo "mm-link:       $(command -v mm-link 2>/dev/null || echo MISSING)"
echo "mm-delay:      $(command -v mm-delay 2>/dev/null || echo MISSING)"
echo "iperf3:        $(command -v iperf3 2>/dev/null || echo MISSING)"
echo "portus (lib):  $(ls -la $PORTUS_ROOT/target/release/*.rlib 2>/dev/null | head -1 || echo MISSING)"
echo "nimbus bin:    $(ls -la $NIMBUS_MEASUREMENT_ROOT/nimbus/target/release/nimbus 2>/dev/null || echo MISSING)"
echo "ccp_kernel:    $(ls -la $NIMBUS_MEASUREMENT_ROOT/ccp-kernel/ccp.ko 2>/dev/null || echo MISSING)"
echo "mpcc_cca:      $(ls -la $MPCC_CRATE/target/release/mpcc_cca 2>/dev/null || echo MISSING)"
echo "sproutbt2:     $(ls -la $SPROUT_ROOT/src/examples/sproutbt2 2>/dev/null || echo MISSING)"
echo "verus_client:  $(ls -la $VERUS_ROOT/client/verus_client 2>/dev/null || echo MISSING)"
echo "verus_server:  $(ls -la $VERUS_ROOT/server/verus_server 2>/dev/null || echo MISSING)"
echo "pyportus:      $(python3 -c 'import pyportus; print(pyportus.__file__)' 2>/dev/null || echo NOT INSTALLED)"
echo
echo "Next: python -m scripts.network.run_cc_experiments check"
