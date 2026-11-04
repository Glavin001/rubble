# PhysX harness (temporary, measurement only)

Small programs that drive the PhysX native GPU destruction stage directly, linked against a
locally installed SDK, to recreate test scenarios and compare rubble's `blast_gpu` with PhysX on
identical inputs. No vibe-land code is involved: only the PhysX install.

- **Target SDK:** `~/Development/PhysX/out/install/garage-multihull`, PhysX `a08abef50`, the build
  vibe-land runs (see `../../findings/blast-gpu-port-target.md`).
- **Never part of rubble's build, tests or CI.** Rubble does not depend on these; results they
  produce are recorded in the findings, not committed as test fixtures.
- **GPU runs one at a time**, each under a kill timer (`perl -e 'alarm 300; exec @ARGV' ...`).

```bash
cmake -S engine/stress-oracle/tools/physx-harness -B <build dir> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPHYSX_SDK=$HOME/Development/PhysX/out/install/garage-multihull
cmake --build <build dir>
perl -e 'alarm 300; exec @ARGV' <build dir>/smoke gpu   # exit 0: the install links and its GPU stage runs
```

| Program | What |
|---|---|
| `smoke` | PhysX's own package-consumer test (`tests/destruction/package-consumer/main.cpp` at `a08abef50`), native scene only: a two-chunk structure configured, solved, loaded by gravity, reconfigured and cleared on the GPU |

`cmake/CuMetalConfig.cmake` points CMake at the CuMetal runtime the SDK bundles
(`lib/libcumetal.dylib`), so the harness links the runtime the SDK was built and warmed with.
