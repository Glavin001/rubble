# AGENTS.md

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
