# LMGC90 oracle: pinned versions

| Item | Version / hash | Licence |
|---|---|---|
| LMGC90 source | `lmgc90_user_2026.rc1.zip` from <https://lmgc90.pages-git-xen.lmgc.univ-montp2.fr/lmgc90_dev/downloads/lmgc90_user_2026.rc1.zip> (link on the official user wiki, <https://git-xen.lmgc.univ-montp2.fr/lmgc90/lmgc90_user/-/wikis/download_and_install>), sha256 `a31902406f83d8e371957cf04e5d7b8227ab01443c12966a0dcf9fca71473499` | CeCILL v1.1 (French GPL-compatible free-software licence; `src/Licence_CeCILL_V1.1-US.txt`). We only run it as an external tool; nothing of LMGC90 is vendored into this repository. |
| pylmgc90 wheel (built locally by `install.sh`) | `pylmgc90-2025rc3-cp312-cp312-linux_x86_64.whl`, sha256 of our build `b22e491d41db2e0397e9cd8263238df88497bd7caa6281e41dc9c5727dbae9eb` (not bit-reproducible across machines; the source zip hash above is the pin). The "2025rc3" in the wheel name is a stale `pyproject.toml` version inside the 2026.rc1 zip. | CeCILL v1.1 |
| Build options | `LMGC90_SPARSE_LIBRARY=none`, `LMGC90_MATLIB_VERSION=none`, `LMGC90_WITH_HDF5=OFF`, `LMGC90_ENABLE_DOC=OFF`, `LMGC90_ENABLE_TESTING=OFF`, ChiPy + pre + post on, no OpenMP (single thread) | |
| Build toolchain | Ubuntu 24.04: gcc/gfortran 13.3.0 (`gfortran 4:13.2.0-7ubuntu1`), `swig 4.2.0-2ubuntu1`, `liblapack-dev/libblas-dev 3.12.0-3build1.1`, CMake 3.28.3, scikit-build-core 1.1.1 (uv build isolation), uv 0.11.32 | |
| Python | CPython 3.12.3 (`/home/user/oracle-env/venv`) | PSF |
| numpy | 2.5.3, `numpy-2.5.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` sha256 `b7e18c623bb5c95acb3b3328861272816ba199fb531921c5d6d0b675f1fde9e3` | BSD-3-Clause |

pylmgc90's declared runtime dependencies `scipy`, `matplotlib`, `vtk`, `h5py` are not
needed for this oracle (installed with `--no-deps`; only the optional vtk display
warning is printed on import).

## Not used, and why

* `compas-lmgc90==0.1.11` (PyPI, MIT, bundles an LMGC90 core): its Fortran wrapper
  (`src/wrap_lmgc90_compas.f90`, `set_see_tables`) registers exactly one see table,
  `RBDY3 POLYR REDxx <-> RBDY3 POLYR REDxx -> 'iqsc0'`, and every POLYR gets colour
  `REDxx`; there is no initial-velocity setter (only driven DOFs) and no SPHER
  contactor. A cohesive law would therefore apply to every contact (including the
  impactor and contacts that form after cracking would need a law with zero
  initial cohesion), joint classes with different stiffness cannot coexist, and the
  impactor cannot be launched freely. The full pylmgc90 (chipy) exposes all of this.
* `pylmgc90` is not on PyPI (404 at <https://pypi.org/pypi/pylmgc90/json>, checked 2026-10-09).
