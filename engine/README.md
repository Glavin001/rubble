# GPU AVBD engine (`engine/`)

New 3D GPU rigid-body physics engine. **Not** an extension of the legacy `crates/rubble*` tree.

- **Physics:** shared Slang (`slang/physics`, `slang/kernels`) → Metal / SPIR-V / CUDA
- **Host:** Rust (`crates/engine-*`)
- **Proof target:** Apple Silicon Metal (MacBook M3 Max)
- **Sleep:** required, island-scoped
- **CCD:** stretch goal (`ccd_classify` / `ccd_sweep`)

See [TOOLCHAIN.md](TOOLCHAIN.md) for slangc and pinned inspiration revisions.
For the browser demo and automated scale/correctness checks, see
[Slang/WebGPU regression checks](crates/engine-webgpu/README.md).
Vehicles (raycast wheels solved inside AVBD, drivetrain, tanks and motorcycles, severable parts, drivable
WebGPU demo `vehicle.html`): see [docs/vehicles.md](docs/vehicles.md).

## Quick start (Metal)

```bash
export PATH="$HOME/.local/bin:$PATH"
cd engine
cargo test -p engine-schema
cargo test -p engine-metal -- --nocapture
cargo test -p engine-oracles -- --nocapture
cargo run -p engine-metal --release --bin abi_probe
cargo run -p engine-metal --release --bin scale_bench

# three-avbd-inspired demos (Y-up)
cargo run -p engine-scenes --release --bin demo_viewer -- --list
cargo run -p engine-scenes --release --bin demo_viewer -- "Wall Smash (2k)"
cargo run -p engine-scenes --release --bin demo_viewer -- "Brick Ring (28k)" --steps 120
cargo run -p engine-scenes --release --bin demo_viewer -- "Chain Mail (1.6k)"
```

## Demo scenes

Ported from `inspiration/pinned/three-avbd` paper/GPU showcases (Z-up → Y-up):

| Group | Scenes |
|-------|--------|
| Paper scenes (GPU) | Brick Ring (28k/110k), Brick Walls (27k), Wall Smash (2k), Breakable Wall (600), Chain Mail (1.6k), Heavy Pendulum |
| Scale tests (GPU) | Box Pile (4k/32k), Box Columns (100k), Jointed Drop (small), Sphere Pile (1k/10k) |
| Contacts & friction | Pyramid, Stack |
| Joints & springs | Rope, Heavy Rope |

## Layout

| Path | Role |
|------|------|
| `slang/` | Single numerical source |
| `crates/engine-schema` | ABI, handles, commands |
| `crates/engine-exec` | Phase graph |
| `crates/engine-metal` | M3 proof backend + benches |
| `crates/engine-scenes` | Demo/benchmark scene builders + `demo_viewer` |
| `crates/engine-oracles` | three-avbd / Newton / Jolt / demo3d adapters |
| `crates/engine-webgpu` | Slang→SPIR-V→WGSL (fair Chrome bar vs three-avbd); `wgsl_smoke`; `vehicle.html` |
| `crates/engine-vehicle` | Vehicle descriptors, presets, GPU record packing, host bookkeeping |
| `crates/engine-cuda` | CUDA source emission |
| `crates/engine-viewer` | Buffer-shared draw |
| `conformance/` | Oracle scene schema + adapter JSON |
