#!/usr/bin/env bash
# Idempotent install of the pinned OpenSeesPy oracle environment, with hash checking.
#
#   oracles/opensees/install.sh [VENV]
#
# VENV defaults to $ORACLE_VENV, else `oracle-env/venv` next to the PhysX checkout
# (/home/user/oracle-env/venv for /home/user/PhysX, the environment the goldens used).
#
# Creates a CPython 3.12 venv with uv if it does not exist, then installs
# requirements.txt with --require-hashes (a no-op when the pinned versions are already
# there; other packages in a shared venv are left alone), and finally verifies the
# installed OpenSees shared library against the hash recorded in the pinned wheel.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT_PARENT="$(cd "$HERE/../../../../.." && pwd)"
VENV="${1:-${ORACLE_VENV:-$CHECKOUT_PARENT/oracle-env/venv}}"
PY="$VENV/bin/python"

if ! command -v uv >/dev/null 2>&1; then
    echo "install.sh: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 1
fi
if [ ! -x "$PY" ]; then
    uv venv --python 3.12 "$VENV"
fi
"$PY" -I -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || { echo "install.sh: $VENV is not CPython 3.12 (wheel hashes are pinned for cp312)" >&2; exit 1; }

uv pip install --python "$PY" --require-hashes -r "$HERE/requirements.txt"

# Verify the binary that will actually run (RECORD hash of openseespylinux 3.8.0.0).
"$PY" -I - <<'PYEOF'
import hashlib, importlib.metadata as md, os, sys
import openseespylinux
want = {"openseespy": "3.8.0.0", "openseespylinux": "3.8.0.0", "numpy": "2.5.3"}
for name, v in want.items():
    got = md.version(name)
    if got != v:
        sys.exit(f"install.sh: {name} {got} installed, {v} pinned")
so = os.path.join(os.path.dirname(openseespylinux.__file__), "opensees.so")
h = hashlib.sha256(open(so, "rb").read()).hexdigest()
pinned = "b1bfa97ceb8f0f5d3c3c49eb498ac4e9dc57f86971f0f54c03074968a0f830a3"
if h != pinned:
    sys.exit(f"install.sh: opensees.so sha256 {h} != pinned {pinned}")
import openseespy.opensees as ops
print(f"OpenSeesPy {md.version('openseespy')} (OpenSees {ops.version()}) ok, opensees.so sha256 {h}")
PYEOF
