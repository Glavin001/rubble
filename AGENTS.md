# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Rubble is a GPU-accelerated rigid-body physics engine (2D + 3D) in Rust with wgpu. It includes a native viewer (winit + egui), a WASM bridge, and a web demo frontend (Vite + Three.js). There are no databases, Docker containers, or backend servers.

### Environment variables

These three variables must be exported before running Rust builds, tests, or clippy:

```
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
```

`RUSTUP_TOOLCHAIN=stable` overrides the `rust-toolchain.toml` nightly pin (which is only for `rubble-shaders` spirv work).

### Key commands

All commands documented in `README.md`. Quick reference:

| Task | Command |
|------|---------|
| Build | `cargo build --workspace` |
| Test (Rust) | `cargo nextest run --workspace --profile ci` |
| Format check | `cargo fmt --all -- --check` |
| Clippy | `cargo clippy --workspace --all-targets -- -D warnings` |
| Web build | `./web/build.sh` (from repo root) |
| Web dev server | `npm --prefix web run dev` |
| E2E tests | `npm --prefix web run playwright:install && npm --prefix web test` |

### Gotchas

- The `rust-toolchain.toml` pins nightly-2025-11-13 for spirv shader work. **Always** set `RUSTUP_TOOLCHAIN=stable` for normal development, CI mirrors this.
- GPU tests require `mesa-vulkan-drivers` (lavapipe software Vulkan). Without it, wgpu cannot acquire an adapter and all GPU tests fail.
- `wasm-pack` is provided via npm devDependencies in `web/package.json`, not through `cargo install`.
- Playwright E2E tests require `web/dist/` to exist (run `./web/build.sh` first). The Playwright config starts its own `vite preview` server on port 4173.
- Playwright uses Chromium with `--enable-unsafe-webgpu --use-vulkan=swiftshader` flags for CPU-emulated WebGPU. Install browsers once with `npm --prefix web run playwright:install`.
- The native viewer (`rubble-viewer`) needs `libxkbcommon-x11-0` and `libxcb-xkb1` installed (`sudo apt-get install -y libxkbcommon-x11-0 libxcb-xkb1`). With DISPLAY=:1 and lavapipe, the viewer renders live physics on the virtual X11 display.

### Chrome WebGPU (SwiftShader) for web demos

The default Chrome in this VM does **not** enable WebGPU. To view the Rubble web demos in Chrome, launch it with these additional flags:

```
--enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=swiftshader --use-angle=swiftshader --disable-vulkan-fallback-to-gl-for-testing --disable-gpu-sandbox
```

These flags have been added to `/usr/share/applications/google-chrome.desktop`. If Chrome was already running without them, kill it and relaunch.

Playwright E2E tests already include these flags in `web/playwright.config.ts`, so `npm --prefix web test` works out of the box.
