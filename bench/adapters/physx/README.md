# bench-physx: NVIDIA PhysX 5 adapter

`bench-physx` runs a benchmark scene on PhysX. The Rust binary (`src/main.rs`) implements
`bench_core::Engine`. A small C++ shim (`src/physx_shim.cpp`) does the PhysX work through an
`extern "C"` API: create the world from flat body/joint arrays plus a settings struct, step
(`simulate(dt)` then `fetchResults(true)`), read the state in scene order, read counters,
destroy. The shim uses only the public PhysX 5 API. It compiles unchanged against two sources
of PhysX:

| Source | Cargo features | Versions | Backends | Platforms |
|---|---|---|---|---|
| `physx-sys` crate builds PhysX from source | `vendored` (default) | 5.1.3 (physx-sys 0.11.5) | CPU | macOS (incl. Apple silicon), Linux, Windows |
| Official [NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) SDK, built by you | `--no-default-features` (+ `gpu`) with `PHYSX_SDK_DIR` | 5.x (tested against 5.9.0 headers) | CPU, CUDA | Linux x86-64/aarch64, Windows x64 |

Command line: the shared harness flags (`--scene --out --poses --mode --threads --sleep
--iterations --set k=v --max-seconds --label --quiet`). See `crates/bench-core/src/cli.rs`.

```sh
cd bench
cargo build --release -p bench-physx
./target/release/bench-physx --scene scenes/out/wall_smash_2k.scene.json --out results/dev/physx-wall.json --threads 1
./target/release/bench-judge --result results/dev/physx-wall.json --scene scenes/out/wall_smash_2k.scene.json
python3 -I adapters/physx/check_initial.py scenes/out/wall_smash_2k.scene.json results/dev/physx-wall.poses.bin
```

## macOS: PhysX 5.10 with GPU dynamics on Metal (CuMetal package) — preferred

The sibling `PhysX` repo installs a package whose CUDA GPU module is compiled for Apple
Metal by CuMetal (sibling `cuda-metal`). `bench/run.py` builds against it automatically
when `../PhysX/out/install/macos-cumetal/release` exists:

```sh
cd bench
PHYSX_CUMETAL_DIR=$HOME/Development/PhysX/out/install/macos-cumetal/release \
  cargo build --release -p bench-physx --no-default-features --features gpu
./target/release/bench-physx --scene scenes/out/brick_pyramid_12.scene.json --out r.json --set gpu=on   # GPU (Metal)
./target/release/bench-physx --scene scenes/out/brick_pyramid_12.scene.json --out r.json --set gpu=off  # CPU, same PhysX
```

- It links what the package's `PhysXDestruction::NativeScene` CMake target links: the
  static PhysX libraries plus `libcumetal.dylib`, with `PX_CUMETAL=1`. PhysX `dlopen`s
  `libPhysXGpuActivity_64.dylib` through the binary's rpath to the package's `lib/`.
- `libcumetal` reads precompiled pipelines from `lib/cumetal-pipeline-archive`. Startup was
  about 0.2 s here. Kernels missing from the archive compile into `CUMETAL_CACHE_DIR`,
  which `run.py` sets to `bench/results/cumetal-cache`.
- PhysX sets `CUMETAL_USE_METAL_DEVICE_ADDRESSES=1` itself, and CuMetal warns that this
  removes cross-stream concurrency.
- The package profile is `rigid-demo-experimental`: no articulations, cloth, particles,
  convex-core or GPU SDF. Boxes, spheres and spherical joints are all the suite needs.
- Never rebuild or install into the PhysX or cuda-metal trees from here. Their own
  `AGENTS.md` and handoff docs govern that work. This adapter only reads the installed
  package.

## macOS (or any host): vendored PhysX 5.1, CPU

You only need a Rust toolchain and a C++ compiler (Xcode command line tools on macOS).
`physx-sys` compiles PhysX 5.1.3 with `cc`. That takes about 15 s to a few minutes,
depending on the machine. `build.rs` finds physx-sys's sources with `cargo metadata`. If that
fails, it falls back to `$CARGO_HOME/registry/src/*/physx-sys-*`, and you can override it with
`PHYSX_SYS_DIR`. It then compiles the shim against `physx/physx/include` with the same defines
physx-sys uses: `PX_PHYSX_STATIC_LIB DISABLE_CUDA_PHYSX NDEBUG=1 PX_SUPPORT_PVD=1`,
`-std=c++14 -fno-strict-aliasing`. The PhysX archive is linked by physx-sys itself. The adapter
never calls physx-sys's generated Rust API.

## Linux / Windows: official SDK 5.x, CPU and GPU

### 1. Build the SDK

```sh
git clone --branch 110.1-omni-and-physx-5.9.0 --depth 1 https://github.com/NVIDIA-Omniverse/PhysX.git
cd PhysX/physx
./generate_projects.sh linux-gcc          # or linux-clang; linux-aarch64-gcc on ARM
                                          # *-cpu-only presets skip CUDA entirely
make -C compiler/linux-gcc-release -j     # use -release (or -profile); not debug/checked
```

- Requirements (from the SDK's `documentation/platformreadme`): CMake ≥ 3.21, Python ≥ 3.5,
  curl, clang or gcc, glibc ≥ 2.31. GPU presets also need the CUDA Toolkit 12.8 (`nvcc` on
  `PATH`, or `CUDACXX=/usr/local/cuda-12.8/bin/nvcc`). If a newer compiler stops the build on
  warnings, remove `-Werror` from `source/compiler/cmake/linux/CMakeLists.txt`.
- Linux presets build static libraries by default (`PX_GENERATE_STATIC_LIBRARIES=True`). They
  land in `physx/bin/linux.x86_64/release/` (or `linux.aarch64`): `libPhysXExtensions_static_64.a`,
  `libPhysX_static_64.a`, `libPhysXPvdSDK_static_64.a`, `libPhysXCommon_static_64.a`,
  `libPhysXFoundation_static_64.a`. The GPU runtime `libPhysXGpu_64.so` goes in the same
  directory. The packman step downloads it, or, from 5.6 on, it can be built from source with
  CUDA.
- Windows: `generate_projects.bat vc17win64` (or `vc16win64`, or `vc17win64-cpu-only`), then
  build the `release` configuration of `compiler/vc17win64/PhysXSDK.sln`. The default
  Windows preset builds DLLs (`PhysX_64.dll`, `PhysXCommon_64.dll`, `PhysXFoundation_64.dll`,
  plus `PhysXGpu_64.dll`) with import libraries, and the dynamic CRT (`/MD`, which Rust needs).
  `PhysXExtensions_static_64.lib` and `PhysXPvdSDK_static_64.lib` are always static. If you
  build static libraries instead, keep `NV_USE_STATIC_WINCRT=False` so the CRT matches Rust's.

### 2. Build the adapter against it

```sh
cd bench
export PHYSX_SDK_DIR=/path/to/PhysX/physx
cargo build --release -p bench-physx --no-default-features                 # CPU
cargo build --release -p bench-physx --no-default-features --features gpu  # CPU + CUDA
```

`build.rs` compiles the shim with `-I $PHYSX_SDK_DIR/include -std=c++17 -DNDEBUG`. It adds
`-DPX_PHYSX_STATIC_LIB` when it links the static libraries, and `PX_CHECKED`/`PX_PROFILE` when
the library directory is `checked`/`profile`. It links, in GNU-ld order, `PhysXExtensions`,
`PhysX`, `PhysXPvdSDK`, `PhysXCommon` and `PhysXFoundation`. For each one it picks
`*_static_64` if that file exists, else `*_64` (import library or `.so`). On Linux it also
links `stdc++ dl pthread rt m` and sets an rpath of `$ORIGIN:<lib dir>`. It looks for the
libraries in the first `$PHYSX_SDK_DIR/bin/*/{release,profile,checked}` that contains them.
Overrides:

- `PHYSX_LIB_DIR=<dir>`: library directory.
- `PHYSX_LIBS=a,b,c`: library names in link order. Use a `static:` / `dylib:` prefix to force
  the link kind.

Building with both the `vendored` feature and `PHYSX_SDK_DIR` is an error. You would get two
copies of PhysX.

### 3. Run (GPU)

- `libPhysXGpu_64.so` / `PhysXGpu_64.dll` is not linked. PhysX loads it at runtime when the
  CUDA context manager is created:
  - **Linux**: `dlopen("libPhysXGpu_64.so")` by bare name. It is found next to the binary
    (`$ORIGIN` rpath), in the SDK bin directory (rpath), or on `LD_LIBRARY_PATH`.
  - **Windows**: put `PhysXGpu_64.dll` (and, for DLL builds, the other PhysX DLLs) next to
    `bench-physx.exe`, or on `PATH`.
- You need an NVIDIA GPU (Volta / sm_70 or newer for 5.9) and a display driver compatible with
  CUDA 12.8. The CUDA toolkit is not needed at runtime.
- Device selection: env `PHYSX_GPU_DEVICE=N` (SDK behaviour), or `--set gpu_device=N` on SDKs
  whose `PxCudaContextManagerDesc` has `deviceOrdinal` (not 5.1).
- With `gpu` built, GPU dynamics is on by default (`--set gpu=off` runs the same binary on the
  CPU). If PhysX cannot create a valid CUDA context, or the scene silently falls back to CPU
  dynamics, the run fails instead of reporting CPU numbers as GPU. The reported fields are
  backend `cuda`, device = `PxCudaContextManager::getDeviceName()`, and threads = the CPU
  dispatcher's worker count (PhysX still runs parts of the pipeline on the CPU).
- GPU scene setup:
  - `PxSceneFlag::eENABLE_GPU_DYNAMICS`
  - `broadPhaseType = eGPU`
  - `gpuMaxNumPartitions = 8`
  - `gpuDynamicsConfig` sized from the scene's body count n. The large scenes have about 12
    contacts per body, so the defaults allow twice that, with PhysX's own defaults as the
    floor:

    | Field | Default | `--set` override |
    |---|---|---|
    | maxRigidContactCount | max(24n, 512Ki) | `gpu_contacts=N` |
    | maxRigidPatchCount | max(8n + 2·joints, 80Ki) | `gpu_patches=N` |
    | foundLostPairsCapacity | max(16n, 256Ki) | `gpu_found_lost=N` |
    | foundLostAggregatePairsCapacity | 1024 | `gpu_agg_found_lost=N` |
    | totalAggregatePairsCapacity | 1024 | `gpu_agg_pairs=N` |
    | heapCapacity | max(n KiB, 64 MiB), rounded up to a power of two | `gpu_heap_mb=N` |
    | collisionStackSize | max(n·512 B, 64 MiB) | `gpu_stack_mb=N` |
    | tempBufferCapacity | max(n·256 B, 16 MiB) | `gpu_temp_mb=N` |

    `--set gpu_partitions=N` overrides `gpuMaxNumPartitions`. If PhysX reports a buffer
    overflow (on stderr; `physxErrors` in the counters), raise the matching value.

Check the GPU path on a machine without CUDA (it has only been compile-checked; see below):

```sh
INC=$PHYSX_SDK_DIR/include   # or physx-sys's physx/physx/include (5.1)
c++ -fsyntax-only -std=c++17 -DBENCH_PHYSX_GPU -DBENCH_PHYSX_FORCE_GPU_DECLS \
    -DPX_PHYSX_STATIC_LIB -DNDEBUG=1 -isystem "$INC" adapters/physx/src/physx_shim.cpp
```

`BENCH_PHYSX_FORCE_GPU_DECLS` forces `PX_SUPPORT_GPU_PHYSX=1`. PhysX hides the CUDA
declarations on macOS, so this is the only way to type-check the GPU code there. Use it only
for this check.

## Fairness settings

These match the rapier adapter's choices. Every value is recorded in the result's
`engine.config`.

- **Solver**: `PxSolverType::eTGS`. `--iterations` = position iterations (default 4, PhysX's
  TGS default). Velocity iterations default to 1 (`--set vel_iters=N`). Both are set on every
  actor with `setSolverIterationCounts`. `--set solver=pgs` switches to PGS.
- **Scene**:
  - `eENABLE_PCM` on (the PhysX 5 default).
  - Broadphase `eABP` on the CPU (`--set broadphase=sap|abp|mbp|pabp`; MBP gets a 4×4 region
    grid over the padded scene bounds).
  - CCD, stabilization and enhanced determinism off.
  - Patch friction (default).
  - Default `PxTolerancesScale` (length 1, speed 10). That gives bounceThreshold 2,
    frictionOffsetThreshold 0.04 and frictionCorrelationDistance 0.025 (all recorded).
- **Materials**: one per distinct scene friction. Static = dynamic = scene friction,
  restitution 0. Friction and restitution combine mode `eAVERAGE`. three-avbd uses sqrt(a·b);
  the suite's coefficients are 0.5/0.6, so the two differ by <0.5%.
- **Shapes**: `PxBoxGeometry(half extents)` and `PxSphereGeometry(r)`, one exclusive shape per
  actor. Default contactOffset 0.02 and restOffset 0 (recorded).
- **Convex hulls** (scene format v2): every hull's points (body frame) are cooked with
  `PxCreateConvexMesh` (`cooking/PxCooking.h`, `PxCookingParams` from the physics' tolerances,
  `eCOMPUTE_CONVEX`, vertex and polygon limit 64; with GPU dynamics also `buildGPUData`, plus
  `eGPU_COMPATIBLE` on SDKs that still have the flag, i.e. ≤ 5.3). Each hull becomes one
  exclusive `PxConvexMeshGeometry` shape (scale 1, identity local pose); a body of several
  hulls is one actor with several shapes. Mass and inertia stay the scene's (never
  recomputed from the shapes). PhysX's quickhull drops points it considers non-extreme or
  near-coplanar (plane tolerance), so cooked hulls can be up to ~2 mm smaller than the input;
  `config.hullCookingResult` records vertex/polygon counts, how many hulls changed and the
  farthest input point left outside its cooked hull. Under GPU dynamics PhysX refuses GPU data
  for some very oblong hulls (a warning each, counted in `physxErrors`) and collides those on
  the CPU; the count is in `hullCookingResult`.
- **Ground plane** (v2): a `PxRigidStatic` at the scene pose with a `PxPlaneGeometry` whose
  local pose is `PxTransformFromPlaneEquation(n, -n·p)` expressed in the actor frame (PhysX's
  plane normal is +x; the scene's is body +y).
- **Dynamics**:
  - `setMass(mass)`, `setMassSpaceInertiaTensor(principal inertia)`, identity centre-of-mass
    pose.
  - Linear and angular damping 0. PhysX's default angular damping is 0.05.
  - `setMaxAngularVelocity(1e4)` (`--set max_ang_vel=X`).
  - Initial linear and angular velocity from the scene.
  - maxDepenetrationVelocity and maxLinearVelocity are left at PhysX's defaults: 1e32 and 1e16
    in 5.1, recorded. `--set max_depen_vel=X` exists for experiments; it is not used by default.
- **Statics**: `PxRigidStatic` for scene bodies with `is_static`.
- **Joints**:
  - `PxSphericalJointCreate` with local frames at the scene anchors (identity rotation).
  - `PxConstraintFlag::eCOLLISION_ENABLED` follows the scene's `collide` flag. It is false in
    every suite scene, which is also PhysX's default.
  - `PxInitExtensions` is called.
- **Sleeping**: off by default. Each dynamic gets `setSleepThreshold(0)`, so the energy test
  (`energy >= threshold`) always passes and PhysX resets the wake counter every step. The wake
  counter is also set to 1e10 after insertion. The `awake` and `activeDynamicBodies` counters
  confirm that every dynamic body stays active. `--sleep on` uses PhysX's defaults (threshold
  0.005, wake counter reset 0.4 s).
- **Threads**: `PxDefaultCpuDispatcherCreate(N)` with N = `--threads` (0 = all logical cores),
  in the default `eWAIT_FOR_WORK` mode (`--set dispatch=wait|yield|spin`). `EngineInfo.threads`
  = N.
- **Step**: `simulate(dt)` then `fetchResults(true)` with the scene dt. The step is not
  pipelined.
- **State**: `getGlobalPose()` (pose = body frame, since the CoM pose is identity), plus linear
  and angular velocity. Bodies come back in scene order, in the canonical Y-up frame, with
  quaternions as (x, y, z, w). PhysX is used Y-up directly, so no conversion is needed.
  Quaternions are passed through bit-exact; they are renormalised only if they are visibly off
  unit length.

Counters, read at each checkpoint and never timed: `awake` (dynamics with `!isSleeping()`),
plus these from `PxSimulationStatistics`:

- `activeDynamicBodies`, `activeConstraints`, `axisSolverConstraints`
- `contactPairs` / `contactPairsTouching` / `contactPairsCacheHits` (discrete pairs)
- `newPairs` / `lostPairs` / `newTouches` / `lostTouches`
- `partitions`

It also reports `physxErrors`, the number of errors and warnings PhysX has reported (each is
also printed to stderr). The statistics describe the last `simulate`, so they read 0 at
checkpoint 0.

## Version caveats

- On macOS the engine is PhysX **5.1.3** (physx-sys 0.11.5, the newest physx-sys). The official
  SDK used on Linux/Windows is 5.x, e.g. **5.9.0**, which is newer. Between them the TGS solver,
  the GPU pipeline and some defaults changed:
  - `PxgDynamicsMemoryConfig` became `PxGpuDynamicsMemoryConfig`, with a 64-bit
    `tempBufferCapacity`.
  - The friction type became patch-only.
  - TGS external forces can now be applied every iteration.

  Compare CPU numbers across these versions with care. `engine.version` and `config.source`
  record which one ran.
- The shim compiles (`-fsyntax-only`, CPU and `BENCH_PHYSX_GPU`) against both the 5.1.3 headers
  and the `110.1-omni-and-physx-5.9.0` headers. Only the vendored 5.1.3 CPU path has been built,
  linked and run here (Apple M3 Max). The official-SDK link step and the GPU path have not been
  run.
