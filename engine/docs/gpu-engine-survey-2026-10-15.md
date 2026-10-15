# GPU physics engine survey for many convex hulls (2026-10-15)

Source-read survey of three-avbd (pinned), box3d, dimforge nexus and the PhysX GPU
pipeline. It asks what helps a scene with ~100k awake convex fragments that our
engine does not already do. Context is the 49-house fracture grid (155k bodies,
246k hulls): its peak is ~34 ms/step on an M3 Max with ~105k awake and ~800k
contacts.

| Kernel | ms |
|---|---|
| solve_primal_color | 9.7 |
| convex_sat_large | 7.6 |
| broadphase_query | 5.3 |
| convex_manifold | ~4 |
| solve_dual | 2.1 |
| color_bodies | 1.5 |
| convex_sat_small | 1.2 |

## What we already have

three-avbd's ideas are ported:
- startAtRest, stick friction anchors, mass penalty;
- Gauss-map edge pruning, face bias;
- contact reuse, whole-step manifold reuse;
- Morton order, sparse colour tail, sorted adjacency.

Nexus (LBVH rebuilt every frame, GJK/EPA, coloured soft TGS, no sleeping) has no
technique we lack. Its tiled, fused colour solver resembles our sparse path. Its
broadphase compaction (8 pairs per thread, one atomic per workgroup) we measured:
batching candidates per thread gave no gain (5.3 → 5.4 ms).

## Lossless ideas we do not have

1. **Separated-pair SAT cache** (box3d `b3CollideHullsAtWidth`, `convex_manifold.c:1271`;
   `b3SATCache` per contact).
   - Re-test last frame's separating feature first. A positive separation proves
     no contact, so the full SAT is skipped.
   - Lossless with a small margin (only skip when separation > ~0.1 mm), because
     our SAT reports "no contact" exactly when some axis separates.
   - box3d's second half (accept a rebuilt manifold when the cached feature's
     separation moved < linear slop) is a heuristic, not lossless.
   - Needs a per-pair cache that survives frames, including separated pairs.
   - Saving depends on the separated share of SAT pairs (not yet measured).
2. **Fat AABBs + moved-only broadphase updates** (box3d `b3DynamicTree_MarkProxyMoved`,
   margin min(5 cm, extent/8); PhysX incremental GPU SAP updates only changed
   handles).
   - Exact if pairs are filtered by the tight AABB afterwards.
   - Fits a persistent pair set rather than our rebuilt hash grid.
3. **Per-compound broadphase proxies with a child midphase** (PhysX aggregates,
   `aggregate.cu`). One proxy per multi-hull body; child pairs come from a
   bipartite or self pass. Our fragments are mostly single-hull, so the gain is
   small here.
4. **Partitioning instead of colouring** (PhysX `PxgConstraintPartition.cpp:919`:
   a 32-bit mask per body, edges take the lowest free bit).
   - It is incremental from persistent pairs, unlike our per-step JP recolour.
   - PhysX's multi-slab mass-splitting average is Jacobi-like (lossy). One slab
     is equivalent to colouring.

## Debris that never sleeps

At peak ~17% of debris keeps creeping (median 2.4 cm/s, 0.13 rad/s, monotonic,
higher in piles). Island sleep then keeps ~135k bodies awake and the step cost
stays near peak.

How the references settle piles:
- **box3d**
  - Soft step: a biased Push stage, then an unbiased Relax.
  - Friction solved last, at one weighted manifold centre, plus rolling
    resistance.
  - Contact recycling within 10x linear slop: narrowphase is skipped and only the
    separation is updated from fixed anchors.
  - Sleep velocity = max(|v| + |w x extent|, 0.5 * max position correction / dt),
    threshold 0.05 m/s for 0.5 s.
- **PhysX**
  - Friction patches with body-space anchors are reused while the normal and
    anchors match; reuse prevents creep.
  - Wake counter over accumulated velocity.
  - Optional stabilization damps settled bodies and lowers their gravity (lossy).

All of these change the physics (solver formulation, friction model, contact
reuse tolerance or sleep criteria), so they need a product decision.

Tested and rejected here:
- the real per-body radius in the start-at-rest test (|v| + |w| r with r from the
  colliders instead of 0.5 m): 135k → 141k awake at 12 s, no help;
- 32 lanes per large SAT pair: no change;
- 16 primal lanes: slower than 8.
