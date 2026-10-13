# Best-of-pinned architecture (must copy)

Synthesized from `inspiration/pinned/{three-avbd,avbd-demo3d,avbd-sandbox,beat-box,newton,JoltPhysics}` while closing the ≥120 FPS / oracle-parity goal. Update as we land each item.

## MUST-COPY (three-avbd PLAN + FINDINGS + GPU solver)

Source of truth for GPU AVBD at scale on Apple Silicon: `three-avbd` @ pinned SHA.

| Pattern | Why | Where in oracle |
|--------|-----|-----------------|
| **Per-body CSR adjacency** (degree → prefix scan → scatter → sort) | Primal/color must not scan all contacts | `avbd2d/gpu/wgsl-topology.ts` |
| **Jones-Plassmann colouring** + color buckets | Parallel-safe body colours | same + `soa/coloring.ts` |
| **Indirect / args buffer per colour** | `iters × colours` dispatch overhead dominates on M1/M-series; never write pass id into a shared buffer after encode | `layout.ts` `makeArgsWGSL`, `solver.ts` |
| **Pass constant for colour/round** (`pc.color`) | Mid-encoder CPU overwrites of one scratch slot make every dispatch see the *last* value | `wgsl-solve.ts` `@group(1)` pass |
| **No hot-path readback** | HUD stats async / one frame late | PLAN §5 |
| **Contact persistence** pair key + feature id, λ·α·γ, k·γ | Energy injection otherwise | `wgsl-collision.ts`, demo3d `manifold.cpp` |
| **Uniform grid broadphase** + separate static/plane list; face-axis reject | Ring scenes have many AABB overlaps that never touch | FINDINGS Stage 8h |
| **Colour cap ≥8**, grow on clashes, ~3 spare; graph = touching pairs not broadphase pairs | Cap-too-low → silent Jacobi clashes → pyramid collapse | FINDINGS Stage 8g |
| **Primal multi-lane** (2/4/8 threads/body by colour size) | Primal is latency-bound on contact chains | FINDINGS Stage 8g |
| **Size manifolds from touching pairs** | Don't allocate 4× broadphase pairs | FINDINGS Stage 8f |
| **Workgroup 64**, compact SoA, f32 | Bandwidth-bound dual on M4 Max | PLAN §5 |
| **α=0.99, γ=0.999, β_lin/ang, k ramp** | Official demo defaults | `avbd-demo3d/source/solver.cpp` |

## MUST-COPY (avbd-demo3d numerical)

- Body-centric primal: gather forces for one body → 6×6 (split 3×3 LDLᵀ in demo) → apply Δz with trust region.
- Dual updates λ and penalty after all primals in the iteration.
- Joint warm-start: `λ *= α·γ`, `penalty = clamp(penalty·γ, min, max)`.
- Adaptive inertial warm-start on positions (accel weight vs gravity).

## TRANSFER (Jolt / Newton) — ideas, not full ports

| Idea | Map to our GPU path |
|------|---------------------|
| Island sleep + wake on new contact with awake body | Compact awake list; skip sleeping in broadphase/solve (already partial) |
| Broadphase layers (moving vs static) | Keep planes/statics out of hash; test dynamics vs static list |
| Contact cache / manifold reuse with bounds | Feature-id hash persistence (three-avbd hashInsert) |
| Deterministic constraint order | Sort each body's CSR list (three-avbd `sortAdjacency`) |

## Pitfalls our engine had (fix to match pinned)

1. **Scratch[4]/[5] rewritten every dispatch before `commit`** → all color/solve passes see only the last index. Fix: pass-table bind offset or args buffer (three-avbd style).
2. **O(bodies × contacts) gather in primal/color** → replace with CSR.
3. **Dispatching `max_colors` including empties × iterations** → bucket + indirect args; cap colours ~8–16.
4. **Grid hash sharing buffers with color lists** → OK only if phases don't overlap; keep adj storage distinct after broadphase.
5. **Sleep events stuck at 0 in benches** → island sleep must actually exclude work (awake compaction).

## Perf targets (from three-avbd on M4 Max, paper on 4090)

| Scene | three-avbd (M4 Max) | Paper (4090) | Our gate |
|-------|---------------------|--------------|----------|
| Brick ring 110k, 4 it | ~10.4 ms | 9.8 ms | ≤8.33 ms @ 120 FPS stretch; document if stress |
| Box columns 100k | ~3 ms | — | ≤8.33 ms |
| Wall smash | multi-lane was −28–53% | — | ≤8.33 ms |

User goal: **all demo scenes ≥120 FPS** on M3 Max → p50 step ≤ 8.33 ms. Stress scenes may be flagged in DECISIONS.md if physics fidelity requires more iters than realtime allows.

## Landed (this goal)

| Item | Status |
|------|--------|
| CSR adjacency (degree→prefix→scatter→sort) | landed |
| Pass-table bind offsets (fix scratch mid-encode bug) | landed |
| Skip empty colours after mid-step colour-count sync | landed (indirect path retained, unused) |
| demo3d joint C0 + λ·α·γ warm-start | landed |
| Binding-order doc for Slang→Metal | landed |

### scale_bench wall-clock on M3 Max (after CSR)

| scene | gpu_p50 | 120 FPS (≤8.33ms) |
|-------|---------|-------------------|
| pile_1k | 2.3 ms | yes |
| pile_10k | 3.8 ms | yes |
| collapse_400 | 0.7 ms | yes |
| wall_smash_2k | 1.1 ms | yes |
| rope_128 | 3.8 ms | yes |
| asleep_50k_awake_1k | 17 ms | **no — need awake compaction** |
