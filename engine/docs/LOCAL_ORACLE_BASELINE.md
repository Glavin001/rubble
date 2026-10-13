# Local oracle baselines (this machine)

**Host:** Apple M3 Max, macOS (darwin 23.x).  
**Policy:** Accuracy first. Speed bars from **local** runs only — never published ms from other GPUs.

## Top oracles selected

| Rank | Oracle | Role | Local status |
|------|--------|------|--------------|
| 1 | `inspiration/pinned/three-avbd` | Primary GPU AVBD + CPU f64/CpuRef | CPU tests **81/81 pass** (Node 24). Stack settles. Headless Dawn GPU **blocked** (dawn.node for macOS 26). Chrome WebGPU timing still TODO. |
| 2 | `inspiration/pinned/avbd-demo3d` | Official C++ numerical / params | Collide/manifold source for our SAT port. |
| 3 | (optional) `avbd-sandbox` GPU | Second GPU orchestration check | After Chrome three-avbd GPU baseline. |

## Commands (local)

```bash
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm use 24
cd inspiration/pinned/three-avbd && pnpm test

cd engine
cargo test -p engine-metal --release --test oracle_accuracy --test avbd_slice
cargo run -p engine-scenes --release --bin scene_fps
```

## Accuracy vs three-avbd (local)

| Case | three-avbd CPU (local) | engine Metal (local) |
|------|------------------------|----------------------|
| Stack 10, 900 frames, spaced drop | PASS (z≈i+1−margin, xy=0) | **PASS** (`oracle_accuracy`) |
| Pyramid finite | PASS | PASS |
| Rope joint bound | PASS | PASS |
| Seated sphere on plane | n/a | PASS (`avbd_slice`) |

### Correctness fixes landed

1. Step order NP→warmStart (C0 at x⁻)
2. Contact ABI 128 B; cReg + margin; massPenalty; island sleep
3. O(1) contact warm-start hash (was O(n²) linear scan)

## Speed — our Metal `scene_fps` (local M3 Max)

Interactive gate: p50 ≤ 8.33 ms. Smash-peak rings/walls/columns are `stress`.

| Scene | p50 ms | note |
|-------|--------|------|
| Wall Smash (2k) @ 8 it | ~7–8 | PASS |
| Chain Mail (1.6k) | ~1.3 | PASS |
| Sphere Pile (10k) | ~3.0 | PASS |
| Stack / Rope / Pyramid | &lt; 6 | PASS |
| Brick Ring (28k) | ~17 | stress (smash peak) |
| Brick Ring (110k) | ~83 | stress |
| Box Columns (100k) | ~63 | stress |

**vs three-avbd GPU:** blocked until local Chrome/Dawn numbers exist. Do not gate on M4 Max published FINDINGS.

## three-avbd CPU log

See `LOCAL_three_avbd_cpu_test.log` (81 passed, 2026-10-13).
