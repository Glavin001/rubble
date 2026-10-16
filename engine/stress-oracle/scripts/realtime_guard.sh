#!/usr/bin/env bash
# Autoresearch guard: every kept change must
#  1. pass the GPU accuracy gate's quick tier (13 cases, established methods),
#  2. add no micro-tier failure (two predate this work: coupled_impulse, coupled_sleeping).
# (Its PhysX-coupled realtime_peak step stayed in the PhysX repository: rubble replaces
# PhysX as the rigid-body engine.)
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -q --release -p stress-gpu --bins 2>/dev/null
STRESS_METHODS=established target/release/stress-gpu-gate --build --quick > /dev/null 2>&1
STRESS_METHODS=established target/release/stress-gpu-gate --quick 2>&1 | tail -1 | grep -q "13 of 13 cases pass"
STRESS_METHODS=established target/release/stress-gpu-gate --build --micro > /dev/null 2>&1
fails=$(STRESS_METHODS=established target/release/stress-gpu-gate --micro 2>&1 | grep -c "FAIL" || true)
[ "$fails" -le 2 ]
echo "guard: pass (quick gate 13/13, micro failures $fails)"
