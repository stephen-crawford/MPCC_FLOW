#!/usr/bin/env bash
# Symlink Mahimahi and Nimbus into third_party/ using paths relative to this repo.
# Assumes ``mahimahi`` and ``nimbus`` live next to ``mpcc_flow`` (same parent directory):
#   parent/
#     mpcc_flow/
#     mahimahi/
#     nimbus/
#
# Usage: ./third_party/setup_symlinks.sh
set -euo pipefail
cd "$(dirname "$0")"
ln -sfn ../../mahimahi mahimahi
ln -sfn ../../nimbus nimbus
echo "Linked $(pwd)/mahimahi -> ../../mahimahi"
echo "Linked $(pwd)/nimbus -> ../../nimbus"
echo "Clear MAHIMAHI_ROOT / NIMBUS_ROOT to use these defaults from network_emulation.paths"
