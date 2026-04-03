# rubble

GPU-accelerated rigid-body physics in Rust ([wgpu](https://github.com/gfx-rs/wgpu)).

- **Status**: [TODO.md](TODO.md) (Rubble spec checklist)
- **CI**: [.github/workflows/ci.yml](.github/workflows/ci.yml) on pushes/PRs to `main`

## Prerequisites

1. Install **Rust stable** with [rustup](https://rustup.rs/) (matches CI).
2. For **GPU tests**, you need a stack wgpu can use (see Linux or macOS below). For **rust-GPU shader work**, `rust-toolchain.toml` pins nightly; everyday `cargo check`, tests, `fmt`, and `clippy` use **stable** like CI.

### Linux (Vulkan / lavapipe)

For software Vulkan (e.g. headless or no discrete GPU), aligned with CI:

1. Install Mesa Vulkan drivers:

```bash
sudo apt-get update
sudo apt-get install -y mesa-vulkan-drivers
```

2. If you need lavapipe explicitly, set the ICD and backend:

```bash
LVP_ICD=$(find /usr/share/vulkan/icd.d -name 'lvp_icd*' 2>/dev/null | head -1)
[ -n "$LVP_ICD" ] && export VK_ICD_FILENAMES="$LVP_ICD"
export WGPU_BACKEND=vulkan
```

### macOS

1. Usually no extra drivers. To force Metal:

```bash
export WGPU_BACKEND=metal
```

## Build

From the repository root:

1. Build the workspace:

```bash
cargo build --workspace
```

2. Same check CI runs:

```bash
cargo check --workspace --all-targets
```

## Tests

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

## Format and Clippy

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

## Native Viewer

The `rubble-viewer` crate provides a native desktop window (winit + wgpu) for real-time physics visualization — no browser or WASM needed.

```bash
cargo run -p rubble-viewer --example demo_3d --release
cargo run -p rubble-viewer --example demo_2d --release
```

Controls (3D): left-drag to orbit, right-drag to pan, scroll to zoom. Controls (2D): left-drag to pan, scroll to zoom.

See [crates/rubble-viewer/examples/](crates/rubble-viewer/examples/) for the demo source.

## Web and WASM (optional)

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

## CI at a glance

| Job | What runs |
|-----|-----------|
| Check | `cargo check --workspace --all-targets` |
| Test | `cargo nextest run --workspace --profile ci` with `WGPU_BACKEND=vulkan` |
| Format | `cargo fmt --all -- --check` |
| Clippy | `cargo clippy --workspace --all-targets -- -D warnings` |
| WASM | `./web/build.sh` (see [web/build.sh](web/build.sh), [web/package.json](web/package.json)) |
| E2E | Playwright on `web/dist` |
