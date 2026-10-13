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

7. **Primary numerical oracle:** three-avbd @ pinned SHA; avbd-demo3d for small seeded stacks; Newton/Jolt for qualitative settle only until adapters grow metrics.
8. **Unsupported three-avbd features** (cloth, soft, plastic, wind) stay out of rigid-body pass/fail gates.
9. **Y-up vs Z-up:** keep engine Y-up; oracle fixtures convert `(x,y,z)→(x,z,y)`.

## Process

10. **Iterative commits** on `feat/avbd-attempt-2`; revert via git if an optimization regresses accuracy or FPS.
11. **Legacy `crates/rubble*`** is not modified for this goal; all work stays under `engine/`.
