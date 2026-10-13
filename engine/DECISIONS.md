# Deferred decisions (ask user at end of goal)

Decisions made autonomously while driving toward ≥120 FPS demos, oracle parity, and hot-path exhaustion. Review at the end; none block continued work.

## Performance / product

1. **120 FPS scope:** Wall-clock `step()` ≤ 8.33 ms (p50) on Apple M3 Max Metal for interactive demos. Brick Ring / Brick Walls / Box Columns smash peaks are `stress` in `scene_fps` (275k+ contacts); still chase three-avbd GPU ms after accuracy.
2. **dt stays 1/60** for physics even when rendering at 120 Hz (two substeps or display interpolation later). Not changed unless oracle comparison requires it.
3. **One mid-step GPU sync** after coloring to read `color_count` / `color_counts` and skip empty solve colors. Preferable to 200+ empty dispatches; revisit with Metal indirect command buffers later.
4. **CSR constraint gather** is mandatory for solve/color; brute-force contact scans are treated as bugs, not a valid mode.
5. **Pairwise broadphase** kept for ≤512 colliders; grid above that. Threshold may need tuning per scene density.
6. **Capsule ropes** may keep sphere-chain demos until capsule–capsule + long-chain AVBD is stable; accuracy tests should cover both when ready.

## Accuracy / oracles

7. **Speed (after accuracy):** beat open-source **GPU** alternatives on **this machine** (M3 Max). Do not trust published ms from other GPUs. Primary local oracle: `three-avbd`; also run at most 1–2 other top GPU refs (sandbox / beat-box) locally. Record baselines under `engine/docs/LOCAL_ORACLE_BASELINE.md`.
8. **Accuracy first:** correctness before any perf chase. Use pinned engines’ **existing** CPU refs (three-avbd f64/CpuRef, avbd-demo3d) for seeded checks when available; GPU physical envelopes always. Do **not** invent a home-grown Rust AVBD.
8b. **Box contacts:** port demo3d/three-avbd SAT + face clipping (multi-point). Contact C0 = n·(xB−xA)+margin (A→B). Contact ABI **128 B** (Slang `_pad0/_pad1/_pad2` must match `engine-schema`). cReg = C_geom − α·C0 + margin (margin only in C0, not double-subtracted). massPenalty k ≥ m/dt² every frame. Exact featureId warm-start only (no nearest). Face-bias 0.95/0.01; SAT face-axis ties keep first axis (stop FACE_A/B flip).
8c. **Spaced three-avbd Stack 10-box/900f: PASS** after matching their step order (NP/C0 at x⁻, then body warm-start) plus small-scene GS (`solve_primal_one` high→low). sleep_lin/ang=0.15, sleep_time=1.0 + island sleep gate (stack sleeps together). High-speed sphere–plane → CCD stretch.
8d. **Step order (mandatory):** build_aabbs → broadphase → narrowphase/persist (C0 at x⁻) → CSR/color → predict/warmStartBodies → primal/dual → extract → island sleep. Predicting before NP made impact C0 huge-negative and exploded drops.
8e. **Island sleep:** candidates revoked via `sleep_island_gate` over contacts so one box cannot sleep under an awake neighbor (was collapsing settled stacks). Sleeping bodies stay in broadphase insert; sleepers still query statics.
8f. **Brick Ring bottleneck (local probe):** ~7 of ~14 ms is narrowphase (iters=0 still ~9 ms; disabling NP → ~5 ms). Face-axis BP reject (8h) cut candidates ~2×. Next: faster box manifolds / OBB cache / primal multi-lane — not more solver iters.
8g. **CSR prefix must cover max_bodies:** Hillis–Steele pass_table offsets were only 2^0..2^15; above 65 536 bodies adjacency was corrupt and Box Columns (100k) exploded. Extended to 2^19. JP assigns *smallest-free* colour (not round id). Island sleep gate always (8 rounds). Fully-asleep steps early-out on host.
8h. **Local three-avbd Chrome WebGPU (M3 Max):** see `docs/LOCAL_three_avbd_gpu_bench.json`. Wall smash 2.86 ms, brick ring 9k 1.75 ms, settled pile 32k 3.14 ms, box columns 100k 3.50 ms. Our seated+settled columns match colours=3 and sleep. Do not shrink BP halo/cellSize — that falsely “wins” Ring by dropping stack contacts.
8i. **Primal multi-lane + Stage 8b manifolds (landed):** `primalAccumulate` / groupshared reduce; CSR/primal/island key **manifolds** (one edge per pair), dual stays per-contact for Metal occupancy. Wall Smash ~2.2 ms (beats local 2.86). Brick Ring 9k scene matches three-avbd params (`12,20,2,10`); 28k remains stress (~9 ms).
8j. **NP dispatch sizing:** never `n_cols*32` — that over-dispatched Brick Ring 9k ~9× (292k threads vs ~33k candidates). Size from `last.candidate_count` (+50% headroom). Contact-reuse (`bodies_prev` and full `updateRefs`+gen) both **reverted** — each regressed Ring 9k ~3.4→~7 ms on Metal; revisit only with timestamps proving SAT-skip wins outweigh hash/copy.
8k. **Ground seating:** `build_world` uses a **plane at y=0.5** (three-avbd ground top). Plane at y=0 air-dropped Brick Ring 0.5 m. Box-slab ground + isLarge O(1) probe still destabilized Columns 100k (~30 frames). Plane@0.5 + column centres at `1+layer` restores asleep early-out. Keep grid `cur > i` early-out (loading every lower-index chain node doubled Ring ms). Ring 9k still ~2× contacts vs local oracle (~110k vs 45k).
8l. **SAT edge axes:** always test edge axes (do not skip when faceSep<0.01). The skip was a false win that forced 4-point face manifolds; restoring edges keeps Stack/Columns green and brings isolated Ring probes back ~3.8 ms. Prefer non-FLAG_FACE_BIAS edge pick on Metal for now.
9. **Unsupported three-avbd features** (cloth, soft, plastic, wind) stay out of rigid-body pass/fail gates.
10. **Y-up vs Z-up:** keep engine Y-up; oracle fixtures convert `(x,y,z)→(x,z,y)`.

## Process

11. **Iterative commits** on `feat/avbd-attempt-2`; revert via git if an optimization regresses accuracy or FPS.
12. **Legacy `crates/rubble*`** is not modified for this goal; all work stays under `engine/`.
