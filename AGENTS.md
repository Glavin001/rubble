# AGENTS.md

- Assume our target is 10,000+ active rigid bodies when calculating performance at scale
- Node: the repo pins its version in `.nvmrc`. Run `nvm use` (or rely on an
  nvm `cd` hook) before any `node`/`npm` command.
- Never make tests, builds or runtime code depend on `inspiration/`: it is
  gitignored and absent in CI, fresh clones and worktrees. Read it to
  understand an algorithm, then write our own tests from its definition and
  physical invariants (see `engine/crates/engine-webgpu/tests/spatial_order.test.mjs`
  and `scene_builders.test.mjs`); do not import or record its outputs.
- Performance work must not trade away physics quality or accuracy. Never make
  lossy changes on our own initiative: simplifying or decimating collision
  shapes, dropping contacts or solver iterations, coarsening timesteps,
  loosening tolerances, or skipping work whose result could differ. Prefer
  lossless optimizations (same results, less work). When a limit or trade-off
  is genuinely wanted:
  - make it an explicit, engine-wide configuration that fails with a clear
    error when input exceeds it, never something that silently degrades input;
  - or fix it at the source (for example, re-export assets from Shardwright with
    the desired settings) rather than altering data at load time.
  Measuring a lossy option to report its effect is fine; adopting it needs the
  user's decision.
- Destruction is forward-only. Fracture is decided before or within a step and
  never by re-simulating one: no save-and-restore of the rigid engine and no
  "internal correction" pass that undoes a step and solves it again (PhysX's
  `internalCorrectionLimit`). A strike's breaks are predicted at the end of the
  frame before it and split before the engine steps. This is the routing check
  plus a predictor such as `threshold_predict` or the GPU punch
  (`engine/stress-oracle/findings/rewind-scaling.md` design C, and the charter's
  M5 routing design).
  - The correction pass (`GpuWorld::correct_frame`) exists only as a measurement,
    behind `STRESS_CORRECTION_LIMIT=1`, off by default.
  - Do not enable it in a profile or in production code.

## Cursor Cloud specific instructions

> All standard build/test/lint/web commands and prerequisites are in `README.md`.
> This section covers **only** Cloud-Agent-specific setup and caveats.

### Shell environment

Every shell session must export these before any Rust command:

```bash
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
```

These are already appended to `~/.bashrc` by the initial setup, so new
interactive shells inherit them. For non-interactive shells (e.g. background
commands), export them explicitly.

### Running the native viewer

The virtual X11 display is at `DISPLAY=:1`. Launch the viewer with:

```bash
DISPLAY=:1 cargo run -p rubble-viewer --example demo_3d --release
```

### Chrome WebGPU in this VM

The system Chrome `.desktop` entry has been patched with the SwiftShader
WebGPU flags documented in `README.md → Chrome WebGPU without a GPU`.
If Chrome is already running without those flags, kill it and relaunch so
the web demos render properly.

### Update script scope

The VM update script installs: `mesa-vulkan-drivers`, `libxkbcommon-x11-0`,
`libxcb-xkb1`, Rust stable + `rustfmt` + `clippy`, `wasm32-unknown-unknown`
target, `cargo-nextest`, and `npm ci` for `web/`. It does **not** start any
services. See `README.md` for how to run builds, tests, and servers.
