# Kratos DEM oracle: pinned versions

Installed by `install.sh` from `requirements.txt` (`uv pip install --require-hashes`).

| Package | Version | Wheel (PyPI) sha256 | Licence |
|---|---|---|---|
| KratosMultiphysics | 10.4.4 | `kratosmultiphysics-10.4.4-cp312-cp312-manylinux_2_28_x86_64.whl` `bff506b4ff8c06bc3fdd8427f29076ac8772f608e2c994fde190e100d5995f3c` | BSD-4-Clause (PyPI metadata) |
| KratosDEMApplication | 10.4.4 | `kratosdemapplication-10.4.4-cp312-cp312-manylinux_2_28_x86_64.whl` `620eb237bf75f6dcda9dac0ac323906435c1de70c9a698df1ced4f00d0d4ea6f` | BSD-4-Clause |
| numpy | 2.5.3 | `numpy-2.5.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` `b7e18c623bb5c95acb3b3328861272816ba199fb531921c5d6d0b675f1fde9e3` | BSD-3-Clause |
| scipy | 1.18.1 | `scipy-1.18.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` `f55fa87b6c612ecd6b058f167c53231b1d14e412efe361d3d6e38b3631c73218` | BSD-3-Clause |
| Python | CPython 3.12.3 | `/home/user/oracle-env/venv` | PSF |

Source used to read the constitutive laws (not built): tag `v10.4.4` of
<https://github.com/KratosMultiphysics/Kratos> (commit
`4f78f8a4a9dc3d92d5e1405ed3e7d10d05ec5b32`),
`applications/DEMApplication/custom_constitutive/DEM_parallel_bond_bilinear_damage_CL.cpp`.

Also present in the venv but not used by this oracle: KratosStructuralMechanicsApplication,
KratosLinearSolversApplication, KratosConstitutiveLawsApplication 10.4.4 (FEM-DEM
"KratosMultiphysics-all==10.3.1" was not installable: a dependency wheel is missing).

Runtime: OpenMP build; the pipeline sets `OMP_NUM_THREADS=1` (Kratos reports
"Maximum number of threads: 1").
