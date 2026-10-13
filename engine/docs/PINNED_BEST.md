# Best-of-pinned (GPU oracles only)

**Speed:** beat every open-source **GPU** alternative — **fair bar = Chrome WebGPU ↔ Chrome WebGPU** (`three-avbd` first; our path is Slang→SPIR-V→WGSL). Native Metal `scene_fps` is the M3 interactive gate, not the oracle claim.  
**Accuracy:** use their **existing** CPU refs when available (three-avbd f64/CpuRef, demo3d, sandbox CPU) plus GPU physical envelopes — never invent a home-grown Rust AVBD.

Primary: [three-avbd](../inspiration/pinned/three-avbd) WebGPU. Also: avbd-sandbox GPU, beat-box Slang, Newton VBD GPU ideas, Jolt island/sleep patterns.

Reports folded from exploration agents: [Mine three-avbd GPU opts](62027ae2-38f8-4e81-8fca-082631dbb3ea), [Mine demo3d+sandbox+beat-box](f03d9f0e-64ea-4a66-9bed-dcc3917b4571), [Mine Newton+Jolt opts](2c62a165-94d0-4421-b6c4-a1d68b03215c).

## MUST-COPY (landed / in progress)

| Pattern | Source | Status |
|--------|--------|--------|
| CSR degree→scan→scatter→**sortAdjacency** | three-avbd | landed |
| Jones–Plassmann + pass/indirect color args | three-avbd | landed |
| Joint C0 + λ·α·γ (demo3d) | avbd-demo3d | landed |
| **startAtRest** (REST_SPEED=0.1) | three-avbd FINDINGS 8m | landed |
| **massPenalty** k ≥ m/dt² | three-avbd | landed |
| O(1) **joint-pair hash** (constrainedTo) | demo3d / three-avbd | landed |
| Sleep: skip predict writes; awake-list colour/pack | three-avbd / Jolt | partial |
| Sleep dual stasis (no λ decay if both sleep) | beat-box | landed |
| Adaptive colorCap + indirect empty colors | three-avbd | landed (indirect) |
| JP smallest-free colour (not round id) | three-avbd | landed |
| CSR prefix through 2^19 (scale bodies) | — | landed (was >65k bug) |
| matchNearest contact warm-start | three-avbd | landed (pair hash + nearest within prev manifold; n·n_prev>0.9) |
| reuseContacts (SAT skip) | three-avbd | opt-in `FLAG_REUSE_CONTACTS` + updateRefs/generation (Box Pile on; Ring off) |
| gridCellSize ≈ 2·maxSmallRadius | three-avbd | landed (always 2·r; no 1.25 brick shortcut) |
| Primal multi-lane | three-avbd 8g | landed |
| Contact manifold records (CSR/primal) | three-avbd 8b | landed (dual still per-contact) |
| Island sleep + fully-asleep early-out | Jolt / three-avbd | landed (gate always; host skip) |
| Face-axis broadphase reject | three-avbd 8h | landed (OBB face axes, 1 mm gap) |
| Cap face manifolds at 4 pts | — | landed (Stack/columns green; Ring ~10.4 ms) |

## Speed bars (local M3 Max only)

Do **not** gate on published M4 Max FINDINGS. Local Chrome numbers: `LOCAL_three_avbd_gpu_bench.json`.

| Scene | our Metal p50 (local) | three-avbd wall (local) | gate |
|-------|----------------------|-------------------------|------|
| Interactive demos | see `scene_fps` | — | ≤ 8.33 ms |
| Wall Smash 2k | ~1.5 ms | 2.86 ms | **beat** |
| Brick Ring 9k smash | ~4.4–7 ms / ~58k | wall 1.75 / gpuTotal **5.51** / 45k | PASS ≥120 FPS; reclaim 3.3 ms after 8ap |
| Settled pile 32k | settling (reuse~60%@600) | wall 3.14 / gpuTotal **11.8** | accuracy first (8ap); speed next |
| Brick Ring 28k smash | ~9.2 ms | — | `stress` |
| Box Pile 32k settled | **~8.2–9.8 ms** | 11.8 ms gpuTotal | STRESS (beats oracle; flickers around 8.33) |
| Box Columns 100k settled | ~0 ms asleep | 3.50 ms | PASS |

`cargo run -p engine-scenes --release --bin scene_fps`

## Accuracy (GPU envelopes)

Same binary: after N steps, finite poses, no escape, no KE runaway — the failure modes the pinned GPU demos do not exhibit on these scenes.
