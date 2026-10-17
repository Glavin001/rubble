#!/usr/bin/env bash
# Build and install the LMGC90 Python package (pylmgc90: chipy + pre) into the oracle venv.
#
# compas_lmgc90 (on PyPI) is NOT used: its wrapper hard-codes a single contact law for
# every pair (see README.md), which cannot separate mortar joints from frictional contacts.
# pylmgc90 is not on PyPI, so it is built from the official LMGC90 user release zip.
#
#   PY=/home/user/oracle-env/venv/bin/python WORK=/home/user/oracle-runs/lmgc90 ./install.sh
set -euo pipefail

PY=${PY:-/home/user/oracle-env/venv/bin/python}
WORK=${WORK:-/home/user/oracle-runs/lmgc90}
ZIP_URL=https://lmgc90.pages-git-xen.lmgc.univ-montp2.fr/lmgc90_dev/downloads/lmgc90_user_2026.rc1.zip
ZIP_SHA256=a31902406f83d8e371957cf04e5d7b8227ab01443c12966a0dcf9fca71473499
NUMPY=2.5.3

# System toolchain (Ubuntu 24.04 noble): Fortran compiler, SWIG, BLAS/LAPACK.
if ! command -v gfortran >/dev/null || ! command -v swig >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gfortran=4:13.2.0-7ubuntu1 swig=4.2.0-2ubuntu1 liblapack-dev libblas-dev
fi

mkdir -p "$WORK/dl" "$WORK/srcdist" "$WORK/wheel"
if [ ! -f "$WORK/dl/lmgc90_user_2026.rc1.zip" ]; then
  curl -sSL --max-time 600 -o "$WORK/dl/lmgc90_user_2026.rc1.zip" "$ZIP_URL"
fi
echo "$ZIP_SHA256  $WORK/dl/lmgc90_user_2026.rc1.zip" | sha256sum -c -
if [ ! -d "$WORK/srcdist/lmgc90_user_2026.rc1" ]; then
  "$PY" -I -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$WORK/dl/lmgc90_user_2026.rc1.zip" "$WORK/srcdist"
fi

uv pip install --python "$PY" "numpy==$NUMPY"
# Rigid-body build: no MUMPS, no MatLib (FE material library), no HDF5. Python modules chipy/pre/post.
CMAKE_BUILD_PARALLEL_LEVEL=${JOBS:-2} uv build --wheel --python "$PY" --out-dir "$WORK/wheel" \
  -C cmake.define.LMGC90_SPARSE_LIBRARY=none \
  -C cmake.define.LMGC90_MATLIB_VERSION=none \
  -C cmake.define.LMGC90_WITH_HDF5=OFF \
  -C build-dir="$WORK/build" \
  "$WORK/srcdist/lmgc90_user_2026.rc1"
# The wheel metadata still says 2025rc3 (stale pyproject.toml in the 2026.rc1 zip).
uv pip install --python "$PY" --no-deps --reinstall "$WORK"/wheel/pylmgc90-2025rc3-cp312-cp312-linux_x86_64.whl
sha256sum "$WORK"/wheel/pylmgc90-*.whl
"$PY" -I -c "from pylmgc90 import chipy, pre; print('pylmgc90 OK', chipy.__file__)"
