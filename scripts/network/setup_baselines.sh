#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"

echo "=== Building networking stack for MPCC congestion control ==="
echo "Repo root: $REPO_ROOT"
echo ""

MAHIMAHI_SRC="${MAHIMAHI_ROOT:-$THIRD_PARTY/mahimahi}"
PORTUS_SRC="${PORTUS_ROOT:-$(dirname "$REPO_ROOT")/portus}"

if [ ! -d "$MAHIMAHI_SRC" ]; then
    echo "ERROR: mahimahi not found at $MAHIMAHI_SRC"
    echo "Clone it: git clone https://github.com/akshayknarayan/mahimahi.git"
    exit 1
fi

if [ ! -d "$PORTUS_SRC" ]; then
    echo "WARNING: portus not found at $PORTUS_SRC"
    echo "Clone it: git clone https://github.com/ccp-project/portus.git"
fi

echo "--- Building Mahimahi ---"
if command -v mm-link &>/dev/null; then
    echo "mm-link already installed."
elif [ -f "$MAHIMAHI_SRC/src/frontend/mm-link" ]; then
    echo "mm-link already built (not installed system-wide)."
else
    echo "Building mahimahi from source..."
    cd "$MAHIMAHI_SRC"
    ./autogen.sh
    ./configure
    make -j"$(nproc)"
    echo "Built. For system-wide: sudo make install"
    cd "$REPO_ROOT"
fi

echo ""
echo "--- Building Portus Python bindings (pyportus) ---"
if python3 -c "import pyportus" 2>/dev/null; then
    echo "pyportus already installed."
elif [ -d "$PORTUS_SRC/python" ]; then
    echo "Building pyportus..."
    cd "$PORTUS_SRC/python"
    if command -v maturin &>/dev/null; then
        maturin develop --release
    else
        echo "Installing maturin..."
        pip3 install maturin
        maturin develop --release
    fi
    echo "pyportus installed."
    cd "$REPO_ROOT"
else
    echo "SKIP: portus/python not found."
fi

echo ""
echo "--- Verifying ---"
echo "mm-link:  $(command -v mm-link 2>/dev/null || echo "$MAHIMAHI_SRC/src/frontend/mm-link")"
echo "mm-delay: $(command -v mm-delay 2>/dev/null || echo "$MAHIMAHI_SRC/src/frontend/mm-delay")"
python3 -c "import pyportus; print('pyportus: OK')" 2>/dev/null || echo "pyportus: NOT INSTALLED"

echo ""
echo "Traces available:"
ls "$MAHIMAHI_SRC/traces/"*.up 2>/dev/null | while read f; do basename "$f" .up; done

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run experiments:"
echo "  mpcc-cc check"
echo "  mpcc-cc run --traces Verizon-LTE-short --algorithms mpcc cubic --delays 25 50"
echo ""
echo "To run MPCC standalone as CCP algorithm:"
echo "  python -m network_emulation.mpcc_cca --ipc netlink"
