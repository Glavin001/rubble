# Local oracle baselines (this machine)

**Host:** Apple M3 Max, macOS (darwin 23.x).  
**Policy:** Accuracy first. Speed bars from **local** runs only — never published ms from other GPUs.

## Top oracles selected

| Rank | Oracle | Role | Local status |
|------|--------|------|--------------|
| 1 | `inspiration/pinned/three-avbd` | Primary GPU AVBD + CPU f64/CpuRef | CPU tests **81/81 pass** (Node 24). Stack settles. Chrome WebGPU bench3d **recorded** (vite needs `@rolldown/binding-darwin-arm64`). |
| 2 | `inspiration/pinned/avbd-demo3d` | Official C++ numerical / params | Collide/manifold source for our SAT port. |
| 3 | (optional) `avbd-sandbox` GPU | Second GPU orchestration check | After closing the Chrome delta on smash scenes. |

## Commands (local)

```bash
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm use 24
cd inspiration/pinned/three-avbd && pnpm test
# GPU bench (install binding if vite fails): pnpm add -D @rolldown/binding-darwin-arm64@1.2.10
pnpm dev   # http://127.0.0.1:5317/bench3d.html — small+steady tiers

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
| Box Columns 100k settle | GPU colours≈3, stands | **PASS** after CSR 2^19 + island gate + seated y=0.5 |
| Seated sphere on plane | n/a | PASS (`avbd_slice`) |

### Correctness fixes landed

1. Step order NP→warmStart (C0 at x⁻)
2. Contact ABI 128 B; cReg + margin; massPenalty; island sleep
3. O(1) contact warm-start hash (was O(n²) linear scan)
4. CSR Hillis–Steele offsets through 2^19 (was exploding >65k bodies)
5. JP smallest-free colours; island sleep gate always; asleep early-out

## Speed — three-avbd Chrome WebGPU (local M3 Max, 2026-10-13)

Adapter: `apple · metal-3`. Source: `LOCAL_three_avbd_gpu_bench.json`.

**Measurement note:** three-avbd reports two numbers. `wallMs` pipelines a batch of steps then one queue drain (CPU encode overlaps GPU). `gpuTotal` is isolated per-step GPU timestamp time. Our `scene_fps` p50 is `commit`+`wait_until_completed` per step — comparable to **gpuTotal**, not wallMs. Pipelining on Metal needs double-buffered host counter/scratch writes (deferred).

| Scene | wallMs | gpuTotal | contacts | colours |
|-------|--------|----------|----------|---------|
| Wall smash 2k @ 10 it | 2.86 | 10.81 | 19 675 | 17 |
| Chain mail 1.6k @ 10 it | 1.13 | 3.15 | 9 | 5 |
| Brick ring 9k @ 4 it | 1.75 | **5.51** | 45 554 | 10 |
| Settled pile 32k @ 4 it | 3.14 | 11.8 | 120 445 | 11 |
| Box columns 100k @ 4 it | 3.50 | 9.57 | 400 000 | 3 |

## Speed — our Metal `scene_fps` (local M3 Max)

Interactive gate: p50 ≤ 8.33 ms. Ring smash uses lead=60 (three-avbd). Settled columns use lead=120.

| Scene | p50 ms | note |
|-------|--------|------|
| Wall Smash (2k) @ 8 it | ~1.6–2.1 | **beats** oracle wallMs 2.86 and gpuTotal 10.81 |
| Brick Ring (9k) @ 4 it | ~4.4–7 | PASS ≥120 FPS; contacts ~58k after gridCellSize fix; was ~3.3 / ~52k — recover in perf pass |
| Brick Walls (27k) | ~7.2 | PASS |
| Chain Mail (1.6k) | ~1.2 | PASS |
| Box Columns (100k) settled | ~0 (asleep early-out) | PASS (plane@0.5, centres at 1+layer) |
| Box Pile (32k) | STRESS ~28–44 ms | accuracy: @3600 deep~400, among-hashed reuse~84%, KE~230, ~1.7k sleeping; speed still open vs oracle settled 11.8 gpuTotal |
| Box Pile (4k) | ~1.4–2 ms PASS | @600: KE~7, deep~0; among-hashed reuse ~87%@600 / ~95%@1200; ~300 sleeping with shallow island gate |
| Stack / Rope / Pyramid | &lt; 3 | PASS |
| Brick Ring (28k) lead=60 | ~27 | stress (smash peak) |
| Brick Ring (110k) | ~140+ | stress |

## three-avbd CPU log

See `LOCAL_three_avbd_cpu_test.log` (81 passed, 2026-10-13).
