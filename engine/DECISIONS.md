# Deferred decisions (ask user at end of goal)

Decisions made autonomously while driving toward ≥120 FPS demos, oracle parity, and hot-path exhaustion. Review at the end; none block continued work.

## Performance / product

1. **120 FPS scope:** Interpreted as wall-clock `step()` ≤ 8.33 ms (p50) on Apple M3 Max Metal for every shipped `engine-scenes` demo, including Brick Ring 110k / Box Columns 100k. If a scene is intentionally a stress case above realtime, mark it `stress` in the scene list rather than silently lowering the bar.
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
9. **Unsupported three-avbd features** (cloth, soft, plastic, wind) stay out of rigid-body pass/fail gates.
10. **Y-up vs Z-up:** keep engine Y-up; oracle fixtures convert `(x,y,z)→(x,z,y)`.

## Process

11. **Iterative commits** on `feat/avbd-attempt-2`; revert via git if an optimization regresses accuracy or FPS.
12. **Legacy `crates/rubble*`** is not modified for this goal; all work stays under `engine/`.
