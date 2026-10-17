#!/usr/bin/env bash
# Install the pinned Kratos DEM oracle runtime into the oracle venv.
#   PY=/home/user/oracle-env/venv/bin/python ./install.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/user/oracle-env/venv/bin/python}
# KratosMultiphysics-all==10.3.1 is not installable (a dependency wheel is missing on PyPI);
# only the core and DEMApplication wheels are needed here.
uv pip install --python "$PY" --require-hashes --no-deps -r "$HERE/requirements.txt"
"$PY" -I -c "import KratosMultiphysics, KratosMultiphysics.DEMApplication as d; print('Kratos DEM OK', d.__file__)"
