# Self-collision of a deformable vehicle: the final design (2026-11-14)

Branch `perf/selfcol-per-pair`. The goal is the GPU's physical cost for a car's
self-collision (the node–triangle contacts of jbeam-semantics-spec.md §36) at 2 kHz, for
one car and for hundreds, with the results of today's path bit for bit (and so the CPU
reference == GPU kernel contract). The design is chosen from the standard GPU
collision pipeline first; the two non-standard schemes of the brief are kept as
comparison rows only.

Machine: M3 Max (40 GPU cores), native wgpu on Metal and headless Chrome. Car: the
generated sedan `sedan_new` as the vehicle lab runs it (548 nodes, 3,581 beams, 528
collision triangles, anti-clip and own-travel boxes on), the mountain at t 34–45 s.

## 1. What the standard pipeline is

Every high-performance GPU collision pipeline for many primitives has the same three
stages (references below):

1. **Broadphase, data-parallel and sort-based**: a uniform grid / spatial hash built by a
   counting or radix sort with prefix sums (Teschner et al. 2003, *Optimized Spatial
   Hashing for Collision Detection of Deformable Objects*; Green 2008/2010, *Particle
   Simulation using CUDA*), or an LBVH built or refit per pass (Lauterbach et al. 2009;
   Karras 2012). It emits a **compacted candidate-pair buffer**.
2. **A conservative skin, so the candidates are reused**: the Verlet / cell list of
   molecular dynamics (Verlet 1967; Allen & Tildesley §5.3; LAMMPS, HOOMD-blue): the
   candidates are every pair within the interaction range plus a skin, rebuilt at a fixed
   cadence or when the largest displacement since the build exceeds half the skin.
   Substepped solvers generate candidates once a frame or every k substeps and reuse them
   in every substep (Macklin et al. 2019, *Small Steps in Physics Simulation*).
3. **A data-parallel narrowphase**: one thread per candidate pair, the dispatch sized
   from the candidate count (indirect dispatch), over all GPU cores, for all objects at
   once, so cost follows the total number of primitives in the scene, not the number of
   objects.

Cloth self-collision adds two textbook tools: normal-cone curvature culling (Volino &
Thalmann 1994) and BVTT-front tracking with spatial-hash pair culling (Tang et al. 2018,
*I-Cloth*; *PSCC*). Sweep-and-prune with temporal coherence (Baraff 1992; Cohen et al.
1995, I-COLLIDE; Liu et al. 2010 on the GPU) is the other classic broadphase.

## 2. Today, against that standard

Today's path already has stages 1 and 2 in a form a molecular-dynamics code would
recognise, but not stage 3, and it runs all three inside the car's own workgroup:

| Stage | Today | Standard |
|---|---|---|
| Broadphase | Per car: counting sort of the 548 candidates into 1,024 hash buckets (Teschner), in the car's workgroup; the walk lists each triangle's (triangle, node) pairs inside its padded box, **2 lanes a triangle, serial over ≤ 64 cells** | the same sort; the emission spread over the whole GPU |
| Skin | The pair list (`pairReuseList`, margin M = 0.1 m), reused while the tracked nodes' displacement *spread* plus the boxes' growth stays within M (lab-mountain-perf.md §5, §6, §8): the half-skin rule made translation-invariant, with per-piece groups. **Its state lives in workgroup memory, so every frame starts with a build**, and with surface contact on (`EB_FRAME_SEG`, a dispatch a step) **every step builds** (`pairReuseBegin` runs at each segment's start: a performance bug found by reading the
kernel, not an accuracy one; not yet confirmed by a measurement) | the same rule; state in storage |
| Narrowphase | **All 10–13k listed pairs re-tested every substep by the car's 512 lanes**: ≈ 22 pairs a lane, each a chain of ≈ 5 dependent loads (liveness, box, node, `tryContact`'s loads, the 9 geometry records), time-sliced on **one of the 40 GPU cores** | one thread a pair, all cores |
| Scope | a workgroup a car; a contact group of k cars on one workgroup | all cars in one dispatch |

What it costs (native, ms per 34-substep frame, t 36–44 s; the census of
`vehicle_lab --tri-census`, ablations 2^8 / 2^26 / 2^27 / 2^30 of `0edf14b9`):

| | ms a frame |
|---|---|
| Whole frame | 15.3–18.2 |
| Without self-collision (2^8) | 4.3 |
| List builds (walk, in the workgroup) | ≈ 3.5 |
| Listed-pair tests | ≈ 6.5 (geometry loads 1.5, retired/exclusion checks 0.9, box and own travel 0.6, the hit stage `selfClip` 3.5) |

Per substep: 11.0–13.3k pairs listed, 8.8–12.8k inside the triangle's box, 4.1–5.3k past
the own-travel box, **47–75 hits**. List reuse 63–86 % of grid steps (5–12 builds a
frame): main-body spread (the body tumbles at 4–15 rad/s and deforms) and, at 34–37 s,
the torn pieces.

**The departure that costs the time** is stage 3 inside one workgroup: 6.5 ms is ≈ 190 µs
a substep for ≈ 11.5k independent tests, i.e. one core walking them 22 deep. The same
tests as one thread each over 40 cores are ≈ 290 threads a core, a single wave, whose
latency is one test's chain. The builds (3.5 ms) are the same departure for stage 1's
emission: the walk's lanes take whole triangles serially.

## 3. Measured dispatch and barrier costs (from `perf/contact-groups`)

`examples/contact_dispatch_cost.rs` and `tools/contact-groups/dispatch.mjs` (that
branch's `out/dispatch_native2.txt`, `out/dispatch_chrome2.txt`), a 512-lane kernel bound
as the explicit frame is:

| | native (wgpu/Metal) | headless Chrome |
|---|---|---|
| a dependent dispatch, no workgroup zeroing | 1.3–2.4 µs (1 / 40 / 128 workgroups) | 1.4–2.2 µs (5.0 µs: 128 workgroups of a vehicle-sized array) |
| the same with naga's workgroup zeroing, vehicle-sized array | **680–1,770 µs** | (Tint zeroes in parallel: included above) |
| one `sync()` | 0.19–0.25 µs | 0.18–0.19 µs |

So a dispatch boundary per substep costs 34 × ≈ 2 µs ≈ **0.07 ms a frame** (natively only
with zeroing off, as that branch proves safe for the split frame), and a barrier is 10×
cheaper than the audit assumed. A per-substep narrowphase across workgroups is affordable;
it is not affordable with zeroing natively (0.7 ms a dispatch).

## 4. The candidates compared

Tests and ms are per 34-substep frame of one car late on the mountain unless stated;
"tests" are narrowphase pair tests a substep. First-principles costs use §3 and a test's
dependent-load chain of ≈ 5 levels at 0.3–0.6 µs (lab-mountain-perf.md; audit §1.2).

| | Design | Tests a substep | Expected self-collision ms | Proof obligations (exactness) | Risk |
|---|---|---|---|---|---|
| **S** | **Standard: today's Verlet list (same superset proof) built by a data-parallel walk across workgroups when the skin check fails, its state in storage; narrowphase one thread a listed pair over all cores by indirect dispatch, every substep; all cars in the same two dispatches** | 11–13k (unchanged), spread over 40 cores | tests: 34 × (dispatch ≈ 2 µs + one chain ≈ 2–4 µs) ≈ **0.15–0.2**; builds: 5–12 × (≈ 2 + 3–5 µs) ≈ **0.05–0.1**; empty indirect dispatches on reuse steps ≈ 0.05; the grid's counting sort stays in the car's workgroup (548 keys) ≈ 0.1. Total ≈ **0.3–0.5** (from ≈ 11) | none new: the same list and the same `tryContact` on the same state; the hits are applied in the specification's order by `selfApply` whatever order they were found in (fixed-point sums, order-free side memory `clipRecord` with storage atomics); every value the cut dispatches exchange goes through storage (as contact-groups-design.md §5.2 for the group path) | the solo frame becomes a dispatch a substep (state saved between steps: contact-groups' `EB_SOLO_STEP`); native zeroing must be off for these modules (contact-groups' proof); Chrome dispatch cost per frame ≈ 0.15–0.25 ms |
| L | LBVH (Karras 2012) over the 528 triangles, refit or rebuilt per substep, 548 node queries | box-overlapping pairs: the same 8.8–12.8k (the box test is the filter either way) | build: Morton codes, a 4-pass radix sort, hierarchy, bottom-up bounds ≈ 6–8 dependent phases a substep; refit alone ≈ 10 levels; traversal ≈ 10 levels × 2 loads ≈ 6 µs latency. ≥ the grid's cost at this size, no fewer tests | the same superset proof (box vs box) | none, but no gain: LBVHs pay for very uneven or very many primitives; these are 548 + 528 near-uniform ones, the grid's case (Teschner) |
| SAP | Sweep-and-prune with temporal coherence (insertion sort) or GPU SAP (a sort a substep + sweep) | the same box pairs (one-axis sweep then the box) | insertion sort is serial; the GPU form is a 1,076-key sort a substep, ≥ the counting sort it replaces | the same | no gain |
| NC | Normal-cone culling / BVTT front (Volino & Thalmann; I-Cloth) | — | — | normal cones prove a low-curvature surface patch cannot intersect *itself*; §36's contacts are nodes in a band of triangles (with spawn exclusions), which a cone does not exclude, so culling would drop specified contacts (lossy). The BVTT front is the coherent pair list, which S already has | not exact |
| a | Per-pair safe-until skipping (brief): per pair (gap, extent, stamp), per-node off-rigid path, skipped until the bound could close the gap | ≈ hundreds tested, but all 11–13k records still scanned each substep (≈ 22 a lane in the workgroup) | scan ≈ 0.4–0.5 + tests; builds unchanged (≈ 3.5) | new: the rotation term a·D per pair, reach growth, the crossing test's lateral reach (whose barycentric interpolation amplifies node travel by the triangle's aspect ratio, so slivers must be excluded), f32 rounding of each | high (a new proof, non-standard); its gain over S is ≤ S's whole test cost (≈ 0.2 ms) |
| b | Candidacy in the co-rotating frame (a fitted rigid motion a step, Kabsch) + per-row refresh | as today | removes the rotation share of the builds; with S a build is ≈ 5–7 µs, so ≤ 0.1 ms to gain | new: the fit's orthogonality and rounding, rows' stable slots | medium–high for ≤ 0.1 ms |

**Why S meets the exactness rule:** it changes where and in what order the same pure
tests run, not which pairs are tested or what a test computes. The list is today's
list (its superset proof, lab-mountain-perf.md §5 and §6, is unchanged); the tests are
today's `pairReuseTest` → `tryContact`; their only effects are order-free (fixed-point
atomic sums, `clipRecord`'s two-smallest storage atomics, idempotent flags) or are put in
the specification's order before use (`selfApply` orders the hit list). Neither (a) nor
(b) is needed: the standard design meets the rule at 2 kHz.

**d (contact groups):** yes, self-collision rides the same structure. `perf/contact-groups`
already runs a solo vehicle's frame a step a dispatch (`EB_SOLO_STEP`, the pair list's
state saved between steps) and spreads the vehicle-contact walk and pair tests over
workgroups ("tests, walk" and "tests, pairs", 16 workgroups a group). Self-collision adds
its own walk and pairs dispatches beside those, over every vehicle at once. Its state
persistence is merged, not reimplemented.

## 5. The chosen design (S), its schedule

Per frame, for the vehicles that run it (a solo car without surface contact first; the
frame kernel is unchanged for everything else, and the old path stays selectable):

| Dispatch | Workgroups | Does |
|---|---|---|
| step s (parameter copy s) | one a vehicle | step s−1's rest (`selfApply` of its hits, the slides, the closing barrier), then step s up to its self-collision: coupling, drive, nodes, beams, the grid's counting sort and the reuse decision (`pairReuseDecide`); the pair list's state and the hit flag saved for the next (contact-groups' `soloStepSave`) |
| self walk s (indirect) | over every vehicle's tested triangles; none when every vehicle reuses its list | a building vehicle's triangles each list their pairs (`pairReuseList`; the grid's offsets in storage) |
| self pairs s (indirect) | one thread a listed pair over every vehicle | `pairReuseTest` → `tryContact` (and the volatile triangles' tests) |

The frame's end runs after the last pairs dispatch (the last step's rest, the frame end,
as `EB_SOLO_STEP` does). Built (`slang/kernels/explicit_beams_selfsplit.slang`, host
`ExplicitBeamsGpu::set_self_split`, `RUBBLE_SELF_SPLIT=0` off):

- **Step dispatch** (`explicit_beams_self_step`): `explicitBeamsFrameStep` with
  `EB_SELF_SPLIT`. At its start (step k > 0) step k − 1's rest: its hit flag rebuilt
  from the saved `gsHit` and the hit list's count (every hit counts there), `selfApply`,
  the surface step (off in these frames), the slides, the closing barrier; and a walk's
  list recorded as `pairReuseTests` records a build's (valid, count, statistics). Then
  step k as the frame runs it up to its tests: a reusing step leaves its listed count for
  the pairs dispatch and tests its volatile triangles in the workgroup (and, past the
  list's room, walks with its tests as before); a building step writes the node build
  records, the volatile list and separations (`piecesBuild` without its triangles' lists)
  and the grid's offsets to storage for the walk dispatch.
- **Walk dispatch** (`explicit_beams_self_walk`): every building vehicle's (triangle,
  share) threads, 8 shares a triangle, list their pairs (`pairReuseList`'s arithmetic, the
  count a storage atomic) and test each as the pairs dispatch would (`pairReuseTest`), so a
  list past its room is still tested.
- **Pairs dispatch** (`explicit_beams_self_pairs`): a thread a listed pair of every
  reusing vehicle.
- Indirect arguments in a buffer of their own (a buffer cannot be storage and indirect in
  one dispatch), maxima over the vehicles, clamped in the kernel (pairs ≤ TRI_PAIR_CAP /
  LANES workgroups, walk ≤ 32 with a strided loop past it), cleared a step ahead.
- Used for frames on the per-vehicle frame kernel alone (no vehicle contact, sleep,
  surface contact, wide passes, test variants or diagnostic pipelines).

**A bug found in the merge of `perf/contact-groups` with this branch's base** (neither
branch alone has it): the coupling's candidate list (`cplListBegin`, lab-skin) clears its
frame maxima at each dispatch's start, and `EB_SOLO_STEP` makes every step a dispatch,
so its skin check would see only the current step's shape travel. Fixed here for
`EB_SOLO_STEP` (cleared at the frame's start only, as the whole frame does). The split
group path's members (`splitMember`, checked 2026-11-15 after merging the latest
`perf/contact-groups`): their coupling step (`splitCoupleStep`) predated the list, so a
member stepped without it (`cplCountBy`, no `cplListShapes`) and nothing cleared its
maxima at the group's frame start (`splitGroupBegin`). The results were the same (the
list is lossless), but the list's state differed from `contactGroup`'s. Fixed: the member
runs `coupleStepMembers`' calls (the list's maxima and count pass) and `splitGroupBegin`
clears the maxima; test `explicit_vehicles::the_group_split_with_the_coupling_is_bit_exact`.
The fold's mark (`cplFoldMark`, workgroup memory) runs at every dispatch of the solo step,
which is right: workgroup memory does not survive a dispatch; groups never fold (neither
`contactGroup` nor `splitMember` marks it).

Test plan (each bitwise against the in-workgroup path, every state word, every frame):
spin at 20 rad/s free and into a wall; a piece torn off mid-list; retired nodes; a
triangle larger than the grid's 64 cells; a fast node through a thin gap; the mountain
crash 36–44 s natively (`--ab-switch`). The CPU reference does not change (it is the walk,
which the list already equals), so its contract is checked by the existing suites.

## 6. Results

Filled in as the phases land.
