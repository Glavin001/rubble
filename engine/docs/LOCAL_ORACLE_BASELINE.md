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

## Fair speed bar (policy)

| Comparison | Status |
|------------|--------|
| **three-avbd Chrome WebGPU ↔ our Slang→WGSL in Chrome WebGPU** | **Fair oracle bar** (same API, same browser). Target. |
| our native Metal `scene_fps` ↔ three-avbd Chrome `gpuTotal` | Cross-API hint only — useful for M3 interactive gate, **not** the claim that we beat their WebGPU stack. |
| our Metal ↔ three-avbd wallMs | Unfair until we have a body/state ring (their wallMs pipelines encode∥GPU). |

Commands for the fair path (full physics step still stubbed; WGSL load proven):

```bash
cd engine
cargo run -p engine-webgpu --release --bin wgsl_smoke    # SPIR-V→WGSL + native wgpu load
cargo run -p engine-webgpu --release --bin wgsl_export   # write generated/wgsl/*.wgsl
cd crates/engine-webgpu && python3 -m http.server 8765
# Chrome (foreground tab): http://127.0.0.1:8765/chrome_wgsl_load.html
# Next: full step + same bench-cases as three-avbd bench3d (gpuTotal / wallMs)
```

## Speed — three-avbd Chrome WebGPU (local M3 Max, 2026-10-13)

Adapter: `apple · metal-3`. Source: `LOCAL_three_avbd_gpu_bench.json` (small+steady) and `LOCAL_three_avbd_paper_large_steady.json` (paper+large+steady).

**Measurement note:** three-avbd reports two numbers. `wallMs` pipelines a batch of steps then one queue drain (CPU encode overlaps GPU). `gpuTotal` is isolated per-step GPU timestamp time. Our Metal `scene_fps` p50 is `commit`+`wait_until_completed` per step — a **cross-API** proxy for gpuTotal, not the fair WebGPU bar.

| Scene | wallMs | gpuTotal | contacts | colours |
|-------|--------|----------|----------|---------|
| Wall smash 2k @ 10 it | 2.86 | 10.81 | 19 675 | 17 |
| Chain mail 1.6k @ 10 it | 1.13 | 3.15 | 9 | 5 |
| Brick ring 9k @ 4 it | 1.75 | **5.51** | 45 554 | 10 |
| Brick ring 110k @ 4 it | 10.97 | 11.47 | 1 170 611 | 11 |
| Jointed drop 34k @ 10 it | 7.23 | 6.95 | 108 507 | 30 |
| Brick gables 506k @ 3 it | 29.99 | 29.62 | 4 102 693 | 9 |
| Settled pile 32k @ 4 it | 3.58 | 3.41 | 126 119 | 13 |
| Box columns 100k @ 4 it | 4.10 | 4.00 | 400 000 | 3 |

### Settled brick pyramid (oracle hold — local Chrome)

`inspiration/pinned/three-avbd/settle-pyramid.html?levels=30` (reuseContacts+startAtRest): after 180 steps **maxZ 14.98/15.25**, **KE≈0.06**, contacts stable ~140 480, colours 13. Our Metal Pyramid 30 still collapses (height→~10, KE thousands) — JP rounds floored to 16 (was 4–8) and priority matched; remaining gap is reuse (~1% vs ~94%) / warm-start churn.

## Speed — our Metal `scene_fps` (local M3 Max)

Interactive gate: p50 ≤ 8.33 ms. Ring smash uses lead=60 (three-avbd). Settled columns use lead=120.

| Scene | p50 ms | note |
|-------|--------|------|
| Wall Smash (2k) @ 8 it | ~1.6–2.1 | **beats** oracle wallMs 2.86 and gpuTotal 10.81 |
| Brick Ring (9k) @ 4 it | ~4.4–7 | PASS ≥120 FPS; contacts ~58k after gridCellSize fix; was ~3.3 / ~52k — recover in perf pass |
| Brick Walls (27k) | ~7.2 | PASS |
| Chain Mail (1.6k) | ~1.2 | PASS |
| Box Columns (100k) settled | ~0 (asleep early-out) | PASS (plane@0.5, centres at 1+layer) |
| Box Pile (32k) | STRESS **~8.2–9.8 ms** | beats oracle settled gpuTotal 11.8; occasionally ≤8.33; @1200 deep≈0, ~32k sleeping |
| Box Pile (4k) | ~1.4–2 ms PASS | @600: KE~5, deep≈0–5, majority sleeping |
| Stack / Rope / Pyramid | &lt; 3 | PASS |
| Brick Ring (28k) lead=60 | ~27 | stress (smash peak) |
| Brick Ring (110k) | ~140+ | stress |

## three-avbd CPU log

See `LOCAL_three_avbd_cpu_test.log` (81 passed, 2026-10-13).
