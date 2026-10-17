# rubble

GPU rigid-body physics with destruction, in Rust and [Slang](https://shader-slang.org/).
One Slang source for the solver compiles to Metal, WebGPU (native wgpu and the browser)
and CUDA. The engine lives in [`engine/`](engine/README.md).

- **Solver:** AVBD (augmented vertex block descent) on the GPU, with island sleeping
- **Shapes:** boxes, spheres, capsules, convex hulls and compound bodies
- **Joints:** one D6 joint record (ball, hinge, slider, fixed, distance, spring, motors,
  limits, breakable); ragdolls and chains
- **Vehicles:** raycast wheels solved inside AVBD, drivetrains, tanks, motorcycles,
  parts you can break off
- **Destruction:** Shardwright fracture assets, a GPU stress solver that breaks bonds
  under load, and a CPU f64 stress reference to check it against
- **Scale:** aimed at 10,000+ active bodies; the browser demos run a 110k-brick ring and
  506k-body gables scenes

<table>
<tr>
<td><img src="engine/docs/media/combined-2026-10/demo-ring110k.jpg" alt="110k-brick ring" width="400"><br><sub><code>demo.html?scene=ring110k</code>: 110,441 bricks in Chrome WebGPU, about 10 ms/step</sub></td>
<td><img src="engine/docs/media/combined-2026-10/demo-ragdolls10k.jpg" alt="10k ragdolls" width="400"><br><sub><code>demo.html?scene=ragdolls10k</code>: 11,002 bodies, 10,001 D6 joints</sub></td>
</tr>
<tr>
<td><img src="engine/docs/media/combined-2026-10/fracture-lab.jpg" alt="Fracture lab" width="400"><br><sub><code>fracture/</code>: a Shardwright house after five shots</sub></td>
<td><img src="engine/docs/media/combined-2026-10/vehicle-traffic.jpg" alt="Vehicle traffic" width="400"><br><sub><code>vehicle.html?scene=traffic</code>: 49 drivable vehicles</sub></td>
</tr>
<tr>
<td><img src="engine/docs/media/combined-2026-10/demo-wall-smash.jpg" alt="Wall smash" width="400"><br><sub><code>demo.html?scene=wall</code>: wall smash, sleeping bodies in blue</sub></td>
<td><img src="engine/docs/media/combined-2026-10/stress-destruction-view.jpg" alt="Stress solver destruction view" width="400"><br><sub><code>destruction-view</code>: GPU stress solver, chunks coloured by bond load</sub></td>
</tr>
</table>

More screenshots: [engine/docs/media/combined-2026-10](engine/docs/media/combined-2026-10/README.md).

## Repository layout

| Path | What | Docs |
|------|------|------|
| `engine/` | **The engine.** Slang AVBD solver for Metal, WebGPU (wgpu and the browser) and CUDA; convex hulls, D6 joints, vehicles, fracture lab, sleeping. Its own Cargo workspace | [engine/README.md](engine/README.md), [engine/DECISIONS.md](engine/DECISIONS.md) |
| `engine/crates/engine-webgpu/` | Browser demos: `demo.html` (scenes and joints), `vehicle.html`, `fracture/` (fracture lab) | [README](engine/crates/engine-webgpu/README.md) |
| `engine/stress-oracle/` | Destruction stress solver: the CPU f64 reference (`stress-ref`), the GPU solver (`stress-gpu`) and its coupling to the engine (`stress-rubble`), viewers and benchmarks. Its own Cargo workspace | [README](engine/stress-oracle/README.md), [improvement tracker](engine/docs/stress-solver-tracker.md) |
| `bench/` | Cross-engine rigid-body benchmark (rubble, three-avbd, rapier, PhysX) | [bench/README.md](bench/README.md) |
| `crates/`, `web/` | **Legacy.** The original wgpu rigid-body crates (2D/3D), WASM build and web demos: the root Cargo workspace | [below](#legacy-wgpu-crates-crates-web) |

## Quick start

Install **[Git LFS](https://git-lfs.com/)** before cloning (`git lfs install`), or run
`git lfs pull` after. The fracture-lab assets (`engine/crates/engine-webgpu/fracture/assets/*.glb`,
about 118 MB) are stored in LFS; without it they are small pointer files and the fracture
lab, its tests and the fracture viewer cannot load them. CI rejects new files over 5 MB
that are not in LFS.

You also need Rust stable, `slangc` (the pinned version is in
[engine/TOOLCHAIN.md](engine/TOOLCHAIN.md)), and for the browser demos the
`wasm32-unknown-unknown` target and [wasm-pack](https://rustwasm.github.io/wasm-pack/).

### Browser demos (WebGPU)

From the repository root:

```bash
(cd engine && cargo run -p engine-webgpu --release --bin wgsl_export)
```

```bash
wasm-pack build engine/crates/engine-fracture-wasm --release --target web --out-dir ../engine-webgpu/generated/fracture-wasm --no-typescript
```

```bash
(cd engine && cargo run -p engine-scenes --release --bin scene_export)
```

```bash
python3 -m http.server 8080 -d engine/crates/engine-webgpu
```

Then open `http://localhost:8080/demo.html`, `/vehicle.html` or `/fracture/` in a
WebGPU browser (Chrome or Edge on desktop). `wgsl_export` compiles the Slang kernels to
WGSL and builds the vehicle wasm; `scene_export` writes the joint scenes (it needs
Metal today). Scenes take `?scene=`: `ring110k`, `wall`, `joints`, `ragdolls10k`, and
more in the demo's scene menu.

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
comparison. The root workspace commands below and the CI jobs in
[.github/workflows/ci.yml](.github/workflows/ci.yml) build and test it.

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

The canonical production build is **[web/build.sh](web/build.sh)** — same script as **CI** and **Vercel** (`npm ci` plus `npm run build:all`). Granular steps live in [web/package.json](web/package.json) (`wasm`, `build`, `build:all`). **wasm-pack** comes from that package’s [devDependencies](web/package.json), not `cargo install`.

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

| Job | What runs |
|-----|-----------|
| Check | `cargo check --workspace --all-targets` |
| Test | `cargo nextest run --workspace --profile ci` with `WGPU_BACKEND=vulkan` |
| Format | `cargo fmt --all -- --check` |
| Clippy | `cargo clippy --workspace --all-targets -- -D warnings` |
| WASM | `./web/build.sh` (see [web/build.sh](web/build.sh), [web/package.json](web/package.json)) |
| E2E | Playwright on `web/dist` |
| Large files | [.github/scripts/check-large-files.sh](.github/scripts/check-large-files.sh): no raw file over 5 MB, every `*.glb` an LFS pointer |
