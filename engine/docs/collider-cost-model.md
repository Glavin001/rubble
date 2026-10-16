# Collider cost model (2026-10-15)

Relative GPU cost of collider shapes and shape pairs, for choosing colliders.
One use is to let a convex decomposition trade "one hull" against "several
boxes".

- **Units.** All numbers are measured on an Apple M3 Max in Chrome WebGPU.
  Absolute nanoseconds will differ on other GPUs; the ratios are what to use.
- **Raw data.** `collider-cost-model.json`.
- **Benchmark.** `engine/crates/engine-webgpu/fracture/shape-cost.mjs`; see
  "Reproduce" at the end.

## Supported shapes

| Shape | Notes |
|---|---|
| Sphere | |
| Capsule | |
| Box | |
| Convex hull | Up to 64 vertices, 128 faces and 192 edges |
| Plane | Static ground |

A body can carry several colliders (a compound). Each touching collider pair
gets its own contact manifold.

How each pair is handled:

| Pair | Path | Max contacts |
|---|---|---|
| Sphere/sphere, sphere/plane, sphere/box, sphere/capsule, capsule/capsule | Closed form, one thread | 1 (capsule/capsule: 1) |
| Box/box, box/plane | Box SAT and clipping, one thread | 4 |
| Any pair with a hull (box/hull, hull/hull, hull vs sphere/capsule/plane) | Cooperative hull path: classify, cached-axis check, SAT, manifold | 4 |
| **Box/capsule, capsule/plane** | **No dedicated test.** Falls back to bounding spheres, so the contacts are wrong | 1 |

Avoid capsules touching boxes or planes until those pairs get real tests.

## Measured cost

Each scene has one static base with one awake body resting on it, so every
pair stays in contact. The numbers are ns per pair per tick.

"Moving" re-tests the pair every tick (contact reuse off), as for debris in
motion. "Resting" has contact reuse on, which skips the shape test while the
pair hasn't moved.

| Pair (base/top) | Contacts | Total (moving) | Narrowphase (moving) | Total (resting) | Relative to box/box (moving) |
|---|---|---|---|---|---|
| box/capsule ⚠ | 1 | 19 | 0.2 | | (wrong contacts) |
| box/sphere | 1 | 37 | 3 | | 0.63 |
| hull 8v/sphere | 1 | 38 | 5 | | 0.66 |
| hull 32v/sphere | 1 | 39 | 5 | | 0.67 |
| hull 32v/capsule | 2 | 47 | 8 | | 0.81 |
| **box/box** | 4 | **58** | 11 | 56 | **1.00** |
| box/hull 8v | 3.25 | 69 | 26 | | 1.19 |
| hull 8v/hull 8v | 3 | 70 | 26 | | 1.21 |
| box/hull 16v | 4 | 74 | 27 | | 1.28 |
| hull 16v/hull 16v | 4 | 87 | 38 | 56 | 1.50 |
| box/hull 32v | 4 | 139 | 88 | | 2.38 |
| hull 32v/hull 32v | 4 | 151 | 103 | | 2.60 |
| box/hull 64v | 4 | 183 | 140 | | 3.16 |
| hull 64v/hull 64v | 4 | 289 | 241 | 56 | 4.98 |
| box / 4-box compound (4 pairs) | 16 | 148 | 45 | 143 | 2.55 |
| hull 8v / 4-box compound (4 pairs) | 12 | 161 | 87 | | 2.77 |

Hull "Nv" means N vertices; a box as a hull has 8. The hulls are prisms with
the box's footprint and height. The resting column was measured for four
pairs only.

## The model

The cost per awake tick is roughly:

```
cost ≈ Σ awake bodies              22 ns            (6x6 solve per body, 8 iterations)
     + Σ colliders on awake bodies 3 ns             (bounds + broadphase)
     + Σ touching collider pairs   NP(pair) + 3.5 ns × contacts + 3 ns
```

`NP(pair)` is the narrowphase cost of one touching collider pair:

| Pair | NP (moving) |
|---|---|
| sphere vs box or hull | 3-5 |
| capsule vs hull | 8 |
| box vs box | 11 |
| hull vs hull or box, both ≤ 16 vertices (≤ 32 faces, ≤ 48 edges: the small SAT bucket) | 26-38 |
| hull vs hull or box, larger | ≈ 1.9 × (Va + Vb), with a box counting as Va = 8 |
| any pair resting with contact reuse on | ≈ 8-9, whatever the shapes |

Contacts per pair: sphere 1, capsule 2, box or hull up to 4.

Notes:

- **Sleeping bodies cost almost nothing.** Their pairs aren't tested or solved.
- **Shape complexity costs only while pieces move against each other.** At rest
  with reuse, a 64-vertex hull pair costs the same as a box pair.
- **AABB-overlapping but separated pairs are mostly cheap.** The OBB face test
  and the cached separating axis drop most of them before the SAT. They were
  not measured separately here.
- **The broadphase scales with collider count**, not with shape complexity.

## For convex decomposition

Compare candidate colliders for one piece by the pairs they create with each
neighbour it touches while moving:

- **One hull with V vertices** gives one manifold per neighbour:
  `NP(hull, neighbour) + 4 × 3.5 + 3`.
- **k boxes** give one manifold per box that touches the neighbour:
  `Σ over touching boxes (11 + 4 × 3.5 + 3) ≈ 28 ns each` (box neighbour),
  plus 3 ns per box in the broadphase.

Against a box neighbour, with all four boxes touching:

| Collider for the piece | Cost per touching neighbour |
|---|---|
| hull ≤ 16 vertices | ~41-55 ns |
| hull 32 vertices | ~105 ns |
| hull 64 vertices | ~157 ns |
| 4 boxes, all touching | ~112 ns |
| 4 boxes, only 2 touching | ~56 ns |

So:

- Prefer one hull of up to about 16 vertices over several boxes.
- Around 32 vertices a hull costs about the same as four touching boxes.
- Beyond about 40 vertices, a few boxes (or two smaller hulls of 16 vertices or
  fewer) are cheaper, especially when only some of them touch any one
  neighbour.
- Spheres are the cheapest collider: 1 contact and a closed-form test.
- Every extra collider adds broadphase work and, when it touches, a whole
  manifold of up to 4 contacts. Contact count drives solver cost (≈ 3.5 ns per
  contact, on top of ≈ 22 ns per body).

## Reproduce

```sh
nvm use
cd engine/crates/engine-webgpu/fracture
node shape-cost.mjs . out.json 0                                  # moving (reuse off)
COMBOS=box/box,hull16/hull16 node shape-cost.mjs . out.json 1     # resting (reuse on)
node headless.mjs pairs a=hull8 b=box4 count=8192                 # one scene
```

- Shapes: `box`, `sphere`, `capsule`, `hullN` (an N-gon prism with 2N
  vertices, N = 3-32), `box4` (four-box compound).
- Scenes are built by `engine-fracture` `pairs.rs` (wasm `Scene.pairs`).
