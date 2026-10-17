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
For single-file Shardwright asset import, see the
[fracture GLB / WebGPU demo](crates/engine-webgpu/fracture/README.md).
Convex hulls and compound bodies (packed hull ABI, broadphase, cooperative SAT,
CPU/GPU verification) are described in [docs/convex-hulls.md](docs/convex-hulls.md).

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

## Native fracture viewer (macOS / Metal)

The native counterpart of the [fracture lab](crates/engine-webgpu/fracture/README.md):
every scene of the lab's Scene menu, built and stepped by the same Rust host logic
(`crates/engine-fracture`) and the same Slang kernels, compiled Slang → MSL with the
lab's `ENGINE_WGSL` buffer layout. See *Shared architecture* in the lab's README.

```bash
cargo run -p engine-viewer --release --bin fracture_viewer                  # house, level 0
cargo run -p engine-viewer --release --bin fracture_viewer -- grid --shoot   # 49 houses, ball fired
cargo run -p engine-viewer --release --bin fracture_viewer -- arch --level 1 --bonds
cargo run -p engine-viewer --release --bin fracture_viewer -- my.fracture.glb
cargo test -p engine-viewer --release --test fracture_regression           # arch impact + every scene
cargo run -p engine-viewer --release --example fracture_compare -- arch 0 240
cargo run -p engine-viewer --release --example fracture_snapshot -- suburb 150 suburb.bmp
```

Keys: 1–7 scenes (house, 20-house row, 7×7 grid, suburb, cuboid suburb, cuboid grid,
arch) · `[` `]` level · drop a `.glb` on the window · click to shoot · F fire the
wrecking ball (rows, grids, suburbs) · drag to orbit · scroll to zoom · Space
start/pause · R reset · B bond graph · K breakable bonds (experimental, preloaded as in
the lab) · C colour by sleep state. Live TPS/FPS and counters are in the window title.
`fracture_compare` runs the deterministic parity shot (compare with `parity.mjs` in the
browser); `fracture_snapshot` renders a scene offscreen to a BMP.

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
| `crates/engine-exec` | Phase graph, backend trait, and `rigid_plan`: the one step graph every host replays (DECISIONS 8et) |
| `crates/engine-metal` | M3 proof backend + benches; `MetalSim` replays `rigid_plan` natively |
| `crates/engine-scenes` | Demo/benchmark scene builders + `demo_viewer`; `scene_export` writes the browser's joint scenes |
| `crates/engine-oracles` | three-avbd / Newton / Jolt / demo3d adapters |
| `crates/engine-webgpu` | Slang→SPIR-V→WGSL (fair Chrome bar vs three-avbd); `wgsl_smoke`; `demo.html`, `vehicle.html`, `fracture/` lab |
| `crates/engine-vehicle` | Vehicle descriptors, presets, GPU record packing, host bookkeeping |
| `crates/engine-vehicle-wasm` | wasm-bindgen face of `engine-vehicle` for the browser vehicle demo |
| `crates/engine-fracture` | Fracture-lab host logic shared by the browser and Metal hosts: Shardwright GLB reader, scene builders, tick plan; stress cooking |
| `crates/engine-fracture-wasm` | wasm-bindgen bindings of `engine-fracture` for the browser fracture lab |
| `crates/engine-slang-build` | Runs `slangc` offline and embeds the Metal / SPIR-V / CUDA artifacts |
| `crates/engine-cuda` | CUDA source emission |
| `crates/engine-viewer` | Buffer-shared draw; native Metal `fracture_viewer` |
| `conformance/` | Oracle scene schema + adapter JSON |
| `stress-oracle/` | Destruction stress solver (own workspace): CPU reference, GPU solver, rubble coupling, `destruction-view`; see its README and [docs/stress-solver-tracker.md](docs/stress-solver-tracker.md) |
| `docs/` | Design notes, benchmarks and baselines (`docs/baselines/`), joints (`docs/JOINTS.md`), convex hulls (`docs/convex-hulls.md`) |

Large assets (the fracture GLBs) are in Git LFS: run `git lfs install` before cloning, or
`git lfs pull` after.
