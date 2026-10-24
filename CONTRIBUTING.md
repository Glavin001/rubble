# Contributing to Rubble

Thanks for working on Rubble. This guide covers the repository, how to build
and test it, and the rules every change follows. For using the engine from a
game or simulation, see the [README](README.md) and
[engine/docs/public-api.md](engine/docs/public-api.md).

## Ground rules

These apply to every change, by people and by coding agents alike
([AGENTS.md](AGENTS.md) has the full text).

- **Scale:** judge performance at 10,000+ active rigid bodies.
- **Physics quality is not traded for speed.** Make lossless optimizations
  (the same results with less work). Don't simplify collision shapes, drop
  contacts or solver iterations, coarsen timesteps, loosen tolerances or skip
  work whose result could differ. If a limit is genuinely wanted, make it an
  explicit engine-wide setting that fails with a clear error when exceeded,
  or fix the data at its source (for example, re-export from Shardwright).
- **Destruction is forward-only.** Fracture is decided before or within a
  step, never by re-simulating one: no save-and-restore of the rigid engine
  and no correction pass. The correction pass (`GpuWorld::correct_frame`)
  exists only as a measurement behind `STRESS_CORRECTION_LIMIT=1`; don't
  enable it in a profile or in production code.
- **Grow with flags, don't delete paths.** New behaviour is opt-in, and a
  world that doesn't use a feature must produce byte-identical results.
- **One source.** GPU code is Slang, host logic is Rust, and JavaScript stays
  a thin layer with no duplicated logic.
- **`inspiration/` is reference only.** It is gitignored and absent in CI and
  fresh clones. Read it to understand an algorithm, then write our own tests
  from its definition and physical invariants. Never import it or record its
  outputs.
- **Node:** the version is pinned in `.nvmrc`; run `nvm use` before any
  `node` or `npm` command.
- **Benchmarks run one at a time.** Only one engine, page or server on the
  GPU while measuring.

## Working on the public API

The game-facing API is the `rubble` crate
([engine/crates/engine-world](engine/crates/engine-world)) and its C ABI
([engine/crates/engine-ffi](engine/crates/engine-ffi/README.md)). The design,
the cross-engine review it comes from and the phase plan are in
[engine/docs/public-api.md](engine/docs/public-api.md) and
[engine/docs/api-review.md](engine/docs/api-review.md).

- **Lossless first.** Each feature lands with a test showing bodies stay
  byte-identical (`bytemuck::cast_slice(download_bodies())`) when it is unused,
  and when its output is read-only (see `tests/outputs.rs` and
  `tests/queries.rs`).
- **Run GPU tests one at a time:**

  ```bash
  cd engine
  cargo test --release -p engine-world -p engine-ffi -- --test-threads=1
  ```

- **The C header is generated.** After changing the ABI, regenerate
  `include/rubble.h` and commit it; `tests/header.rs` fails when it is stale:

  ```bash
  cd engine
  RUBBLE_FFI_BLESS=1 cargo test --release -p engine-ffi
  ```

- **Coordinates:** every new ABI field that carries a vector, rotation or
  length goes through `src/coords.rs` with the right class (polar vector,
  pseudovector, rotation, length), and gets a round-trip test.
- **The quick start compiles:** the README example is
  `engine/crates/engine-world/examples/quickstart.rs`; keep them in sync.
- **Format only your files.** Parts of the tree aren't `rustfmt`-clean, so
  run `rustfmt <your files>` rather than `cargo fmt` across the workspace.

## Repository layout

| Path | What | Docs |
|------|------|------|
| `engine/` | **The engine.** Slang AVBD solver for Metal, WebGPU (wgpu and the browser) and CUDA; convex hulls, D6 joints, vehicles, fracture lab, sleeping. Its own Cargo workspace | [engine/README.md](engine/README.md), [engine/DECISIONS.md](engine/DECISIONS.md) |
| `engine/crates/engine-webgpu/` | Browser demos: `demo.html` (scenes and joints), `vehicle.html`, `fracture/` (fracture lab) | [README](engine/crates/engine-webgpu/README.md) |
| `engine/stress-oracle/` | Destruction stress solver: the CPU f64 reference (`stress-ref`), the GPU solver (`stress-gpu`) and its coupling to the engine (`stress-rubble`), viewers and benchmarks. Its own Cargo workspace | [README](engine/stress-oracle/README.md), [improvement tracker](engine/docs/stress-solver-tracker.md) |
| `bench/` | Cross-engine rigid-body benchmark (rubble, three-avbd, rapier, PhysX) | [bench/README.md](bench/README.md) |
| `crates/`, `web/` | **Legacy.** The original wgpu rigid-body crates (2D/3D), WASM build and web demos: the root Cargo workspace | [below](#legacy-wgpu-crates-crates-web) |

## Building from source

Install **[Git LFS](https://git-lfs.com/)** before cloning (`git lfs install`), or run
`git lfs pull` after. The fracture-lab assets (`engine/crates/engine-webgpu/fracture/assets/*.glb`,
about 118 MB) are stored in LFS; without it they are small pointer files and the fracture
lab, its tests and the fracture viewer cannot load them. CI rejects new files over 5 MB
that are not in LFS.

You also need Rust stable, `slangc` (the pinned version is in
[engine/TOOLCHAIN.md](engine/TOOLCHAIN.md)), and for the browser demos the
`wasm32-unknown-unknown` target. The web build below downloads the pinned `slangc` and
`wasm-pack` itself when they are not installed.

### Browser demos (WebGPU)

From the repository root, build the site (the same build as Vercel and CI; Linux or
macOS, no GPU needed) and serve it:

```bash
bash engine/scripts/build-web.sh
```

```bash
python3 -m http.server 8080 -d engine/web-dist
```

Then open `http://localhost:8080/` in a WebGPU browser (Chrome or Edge on desktop). The
build runs `wgsl_export` (Slang kernels to WGSL, the vehicle wasm), the fracture-lab
wasm and `scene_export` (the Rust-built scenes as GPU snapshots), then copies the pages
into `engine/web-dist/`. `demo.html` takes `?scene=`: `ring110k`, `wall`, `joints`,
`ragdolls10k`, `legacyPyramid`, `legacyGrid10k`, and the rest of its scene menu.

### Legacy scenes on the new engine

[`engine-scenes/src/legacy.rs`](engine/crates/engine-scenes/src/legacy.rs) rebuilds the
legacy web demo's 3D scenes (Empty, Ground, Dynamic Friction, Static Friction, Pyramid,
Stack, Stack Ratio, Scatter, Scatter Boxes, Grid Boxes, Slanted Grid, 10k Grid) body for
body: the same sizes, masses, positions, velocities and per-body friction, legacy
gravity (−9.81) and 12 iterations. Legacy's random scenes used an unseeded RNG; the port
uses a fixed seed. They are the `legacy ·` entries in `demo.html`, and natively:

```bash
(cd engine && cargo run -p engine-scenes --release --bin demo_viewer -- "10k Grid")
```

`100k Grid` (`?scene=legacyGrid100k`) is the 10k grid scaled to 100,000 cubes
(50 × 40 × 50). It isn't in the legacy demo; it uses the same cubes and spacing at the
10,000+ body scale the engine targets.

### Native (macOS / Metal)

```bash
cd engine
cargo test -p engine-metal --release -- --test-threads=1
cargo run -p engine-scenes --release --bin demo_viewer -- "Wall Smash (2k)"
cargo run -p engine-viewer --release --bin fracture_viewer -- grid --shoot
```

The stress solver's viewer, from `engine/stress-oracle`:

```bash
cd engine/stress-oracle
cargo run -p stress-rubble --release --bin destruction-view -- --scene house_ram
```

See [engine/README.md](engine/README.md) for every test, bench and viewer, and
[engine/crates/engine-webgpu/README.md](engine/crates/engine-webgpu/README.md) for the
WebGPU regression checks.

## Legacy wgpu crates (`crates/`, `web/`)

The original wgpu rigid-body engine (2D and 3D), with its WASM build, Vite web demos and
native viewer. The new engine in `engine/` replaces it; it is kept for reference and
comparison. The root workspace commands below build and test it; its CI jobs are in
[.github/workflows/legacy.yml](.github/workflows/legacy.yml) and run only when started
by hand (Actions → Legacy CI). Its web demo is no longer deployed.

### Prerequisites

1. Install **Rust stable** with [rustup](https://rustup.rs/) (matches CI).
2. For **GPU tests**, you need a stack wgpu can use (see Linux or macOS below). For **rust-GPU shader work**, `rust-toolchain.toml` pins nightly; everyday `cargo check`, tests, `fmt`, and `clippy` use **stable** like CI.

#### Linux (Vulkan / lavapipe)

For software Vulkan (e.g. headless or no discrete GPU), aligned with CI:

1. Install Mesa Vulkan drivers and X11/keyboard libraries needed by the native viewer:

```bash
sudo apt-get update
sudo apt-get install -y mesa-vulkan-drivers libxkbcommon-x11-0 libxcb-xkb1
```

2. If you need lavapipe explicitly, set the ICD and backend:

```bash
LVP_ICD=$(find /usr/share/vulkan/icd.d -name 'lvp_icd*' 2>/dev/null | head -1)
[ -n "$LVP_ICD" ] && export VK_ICD_FILENAMES="$LVP_ICD"
export WGPU_BACKEND=vulkan
```

#### macOS

1. Usually no extra drivers. To force Metal:

```bash
export WGPU_BACKEND=metal
```

### Build

From the repository root:

1. Build the workspace:

```bash
cargo build --workspace
```

2. Same check CI runs:

```bash
cargo check --workspace --all-targets
```

### Tests

1. Install [cargo-nextest](https://nexte.st/) once:

```bash
cargo install cargo-nextest --locked
```

2. Run tests. Set `WGPU_BACKEND` as in the Linux or macOS sections (`vulkan` with lavapipe on Linux, `metal` on macOS if needed):

```bash
cargo nextest run --workspace --profile ci
```

3. Run the heavy physics bug-hunt lane. This executes the ignored long-horizon sweep and chaos tests:

```bash
cargo nextest run --workspace --profile heavy --run-ignored ignored-only
```

4. Run the current known-failure scenarios explicitly. These are skipped by default in the normal lane, but they still live in-tree and can be re-run to track progress:

```bash
RUBBLE_RUN_KNOWN_FAILURES=1 cargo nextest run --workspace --profile ci
```

5. Without nextest:

```bash
cargo test --workspace
```

6. Optional host-side Rust coverage for the default lane:

```bash
cargo install cargo-llvm-cov --locked
cargo llvm-cov --workspace --lcov --output-path target/llvm-cov/lcov.info
```

Coverage reports measure Rust-side orchestration and control flow. They do not directly measure WGSL or SPIR-V shader execution, so keep treating the invariant tests as the source of truth for GPU correctness.

### Format and Clippy

1. Add components once:

```bash
rustup component add rustfmt clippy
```

2. Check formatting:

```bash
cargo fmt --all -- --check
```

3. Run Clippy:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

On Linux, CI installs `mesa-vulkan-drivers` for Clippy as well so GPU-linked crates behave like the test job.

### Native Viewer

The `rubble-viewer` crate provides a native desktop window (winit + wgpu) for real-time physics visualization — no browser or WASM needed.

```bash
cargo run -p rubble-viewer --example demo_3d --release
cargo run -p rubble-viewer --example demo_2d --release
```

Controls (3D): left-drag to orbit, right-drag to pan, scroll to zoom. Controls (2D): left-drag to pan, scroll to zoom.

See [crates/rubble-viewer/examples/](crates/rubble-viewer/examples/) for the demo source.

### Web and WASM (optional)

The legacy web build is **[web/build.sh](web/build.sh)** (`npm ci` plus `npm run build:all`); the manual Legacy CI workflow uses it. Granular steps live in [web/package.json](web/package.json) (`wasm`, `build`, `build:all`). **wasm-pack** comes from that package’s [devDependencies](web/package.json), not `cargo install`.

1. Install **Node 20** (same major version as CI).

2. Add the WebAssembly Rust target:

```bash
rustup target add wasm32-unknown-unknown
```

3. From the **repository root**, build (installs npm deps and runs the WASM + Vite pipeline):

```bash
./web/build.sh
```

For a quick rebuild after you already have `web/node_modules` (skips a full `npm ci`):

```bash
npm --prefix web run build:all
```

4. Other useful commands from the repository root:

| Goal | Command |
|------|---------|
| Dev server | `npm --prefix web run dev` |
| WASM only | `npm --prefix web run wasm` |
| Typecheck only | `npm --prefix web run typecheck` |
| E2E (install browsers once, then tests) | `npm --prefix web run playwright:install` then `npm --prefix web test` |

Use `cd web` and the same `npm run …` names if you prefer not to use `--prefix web`.

#### Chrome WebGPU without a GPU

On headless Linux or machines without a discrete GPU, Chrome ships a bundled
SwiftShader Vulkan driver that can emulate WebGPU in software. Launch Chrome
with these extra flags to enable it:

```bash
google-chrome \
  --enable-unsafe-webgpu \
  --enable-features=Vulkan \
  --use-vulkan=swiftshader \
  --use-angle=swiftshader \
  --disable-vulkan-fallback-to-gl-for-testing \
  --disable-gpu-sandbox
```

The Playwright E2E config (`web/playwright.config.ts`) already includes these
flags, so `npm --prefix web test` works without extra setup.

> **Note:** E2E tests serve from `web/dist/`. Run `./web/build.sh` (or
> `npm --prefix web run build:all`) before the first `npm --prefix web test`.

## CI at a glance

[.github/workflows/ci.yml](.github/workflows/ci.yml), on pushes to `main` and on pull
requests:

| Job | What runs |
|-----|-----------|
| Web demos | [engine/scripts/build-web.sh](engine/scripts/build-web.sh) on Linux (the Vercel build), then the JS tests of the WebGPU host, vehicle and fracture modules; uploads the site |
| Engine host tests | `cargo test` of `engine-schema`, `engine-exec`, `engine-scenes` (host build), `engine-vehicle` and `engine-fracture`: no GPU |
| Large files | [.github/scripts/check-large-files.sh](.github/scripts/check-large-files.sh): no raw file over 5 MB, every `*.glb` an LFS pointer |

GPU tests (Metal, wgpu, the stress solver) need a real adapter and run locally; see
[engine/README.md](engine/README.md). CI does no performance testing: software GPUs
don't predict real ones. The legacy workspace's jobs are in
[legacy.yml](.github/workflows/legacy.yml), manual only.

Vercel builds the same site from [vercel.json](vercel.json): production from `main`
([rubble-tau.vercel.app](https://rubble-tau.vercel.app)) and a preview for every pull
request. A Linux build takes about 10 minutes. Pushes that change nothing the site uses
(docs, the stress solver, the legacy crates) skip the build (`ignoreCommand`). The project
needs **Git LFS enabled** (Project Settings → Git) so the fracture assets are real files;
that setting can't live in `vercel.json`.
