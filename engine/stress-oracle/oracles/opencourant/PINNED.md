# OpenCourant oracle — pinned versions

| item | value |
|---|---|
| solver | OpenCourant (fork of OpenRadioss), Linux x86-64 prebuilt, GNU compiler build |
| release tag | `latest-20261006` |
| source commit | `33e685176cccf0c539a3ce07aa2096985a284e2a` (reported by `starter_linux64_gf -version`, build date Oct 7 2026, reader `20260710_d899773e`) |
| asset | `OpenCourant_linux64.zip` |
| asset sha256 | `9d67531de156dd9beba05fbfe710dcdc2bcecbdf3dc3a12642cf85dcece80081` |
| URL | https://github.com/OpenCourant/OpenCourant/releases/download/latest-20261006/OpenCourant_linux64.zip |
| executables used | `exec/starter_linux64_gf`, `exec/engine_linux64_gf` (SMP/OpenMP, double precision), `exec/th_to_csv_linux64_gf`, `exec/anim_to_vtk_linux64_gf` |
| environment | `OPENCOURANT_PATH=<install>/OpenCourant`, `LD_LIBRARY_PATH=$OPENCOURANT_PATH/extlib/hm_reader/linux64:$OPENCOURANT_PATH/extlib/h3d/lib/linux64`, `RAD_CFG_PATH=$OPENCOURANT_PATH/hm_cfg_files`, `OMP_NUM_THREADS`/`-nt` = 1 or 2 |
| Python | `/home/user/oracle-env/venv/bin/python` 3.12, numpy (only numpy and the standard library are used) |
| input deck format | `/BEGIN` version 2026, units g / cm / s |

OpenCourant keeps only its three newest builds on the release page.  `install.sh`
downloads exactly this tag, verifies the hash, and reuses a cached copy of the zip in
`<prefix>/opencourant-dl/` when its hash matches; if the tag has been pruned upstream,
keep that cached zip (the hash above is the identity of the build).

Source code consulted for the model choices (read-only, same commit): 
`starter/source/materials/mat/mat169/hm_read_mat169.F90`,
`engine/source/materials/mat/mat169/sigeps169_connect.F90`,
`engine/source/elements/solid/sconnect/*` (TYPE43 cohesive solid),
`engine/source/output/th/bcs1th.F` (reaction TH = accumulated impulse),
`engine/source/user_interface/suforc3.F` (no `/SECT` accumulation for TYPE43 elements).
