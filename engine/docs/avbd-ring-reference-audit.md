# 110k brick ring: pinned three-avbd / Slang WebGPU audit

Audit date: 2026-10-14. Reference checkout HEAD: `2bb1a31`; Rubble HEAD: `501485f`, with substantial working-tree changes. The reference checkout also has local changes in its benchmark scene/case files. This audit compares the **actual working-tree sources**, not just those commits. Source locations below identify functions as well as line numbers because implementation work is continuing in parallel.

This was a read-only implementation audit. No GPU tests or browser timing workloads were run by the auditor. The only executed experiment was CPU enumeration of the grid hash. Proposed speedups are hypotheses unless explicitly identified as historical measurements from the reference's own documentation. They are not measurements of Rubble on the user's machine.

## Outcome and priorities

There are several concrete reference advantages left to adopt. The most urgent are **pair uniqueness and coverage defects**, followed by unnecessary generated memory/atomic operations. Compact contact storage and stable color packing offer larger architectural opportunities. A fixed 8.333 ms physics budget must include collision detection, persistence, graph construction, coloring, all four primal/dual iterations, and velocity extraction at the full active workload.

| Priority | Finding | Why it matters | Status when found |
| --- | --- | --- | --- |
| P0 | Large impactor pairs can be duplicated or missed depending on collider order | Changes the actual constraint system and invalidates reordered-scene comparisons | Parent reproduced count 2, implemented fix; focused GPU test passes |
| P0 | Colliding neighbor-cell hashes can emit a small-body pair repeatedly | Doubles constraints even with unique bodies; wasted work and altered support | Parent added exact-cell filter; focused GPU test passes |
| P1 | Primal uses atomic loads for immutable CSR/color data | Extra hot-path operations and weaker optimization opportunities | Parent implemented plain-read aliases; compiler/GPU focused tests pass, speed unmeasured |
| P1 | Primal stores the entire body when only pose changes | 112-byte logical store versus 28 bytes of changed fields | Parent implemented pose-only stores; compiler/GPU focused tests pass, speed unmeasured |
| P1 | Default contact-reuse diagnostics update globally shared atomics | Every pair attempts one shared diagnostic increment | Removed unused increments; generated-IR regression passes; overall speedup unestablished |
| P1 | 128-byte contacts duplicate pair geometry; reference uses 64-byte points | More memory, cache lines and traffic in every iteration | Verified ABI difference |
| P1 | Color buckets use global atomic append; reference preserves index locality | Scatters neighboring body reads even when body order is good | Stable tile packing implemented and scale tests pass; timing evaluation in progress |
| P2 | CSR sorts append IDs, not physical neighbor IDs | Sorting does not make accumulation order deterministic | Verified correctness/reproducibility gap |
| P2 | Always 32 color rounds and 32 primal dispatches/iteration | Empty command and host encoding overhead | Verified; safe reduction needs a GPU-correct coverage plan |
| P2 | Linked bucket chains versus reference counting-sort grid | Dependent random loads and weaker collision handling | Verified; wholesale replacement needs phase evidence |
| P2 | Full contact dispatch for hash insertion/persistence versus pair-centric reference | Work and repeated loads scale with points rather than manifolds | Verified; grouping has occupancy tradeoffs |
| P2 | Live demo awaits a readback every rendered frame | CPU/GPU serialization and misleading live FPS relative to physics throughput | Verified; distinct from batched physics timing |

The parent has already pruned unused counter snapshots, implemented incremental coloring with compaction, avoided empty sleep propagation, and added an explicit physics-only benchmark mode. During the audit the parent also implemented the first four findings above. Their original mechanisms remain documented below; they are **newly implemented and not yet measured**, not remaining implementation recommendations. Do not declare 120 FPS or overall parity from this audit.

## Scene and scale: largely aligned, with remaining numerical differences

The actual large ring has **110,440 bricks + one dynamic sphere + one static ground = 110,442 total bodies**. Some reference comments still say 110,332 bricks; trust the builder and direct comparison test, not that prose. `tests/scene_builders.test.mjs` checks every brick's course height, ring radius, step profile, tangential orientation, mass and inertia, each ring's even spacing, maximal brick count, phase and running-bond offset, and the impactor's resting contact and rolling-without-slip velocity. It also checks the ground and rolling contact velocity.

Reference `src/avbd3d/bench-scenes.ts:14-17,53-57,88-123` builds a radius-80 ring, 40 courses, tier 4, ten rows at the base; bricks have full sizes 1 × 0.5 × 0.5, mass 0.25, 2% circumferential gaps and 0.52 m radial pitch. Ball radius is 8, density 10, speed 30, center at reference `[0,-95.2,8.5]`; ground is 360 × 360 × 1 centered at zero. Rubble `scene-builders.mjs:1-36` and `demo.html:568-605` match after the Y/Z reflection. Rotation and angular velocity correctly change sign under that reflection; the sphere rolls along +Z at angular velocity `[3.75,0,0]`. Ground top is Y=0.5. The deep ground used by columns is intentionally separate and should remain intact.

Parameters match in the important ring defaults: dt=1/60, gravity=-10, four iterations, alpha=.99, gamma=.999, betaLin=10000, margin=.01, friction=.5, contact reuse enabled. Reference evidence: `gpu/solver.ts:202-213`, `ref/solver.ts` default parameters, and `bench-cases.ts:82`. Rubble evidence: `demo.html:164-168,330-335`.

Remaining **verified numerical differences**, not proven causes of ring slowdown:

* `predictOne`, `world_step.slang:166-174`, uses radius **0.5 for every body** in the slow-body criterion. Reference `wgsl-solve.ts:294-297` uses each body's actual radius. A brick's radius is about .612372; the ball's is 8. The moving ball is nowhere near the rest threshold initially, but slow rotating debris can cross it differently. Store a true per-body bound; test angular-only bodies on both sides of the .1 m/s cutoff, including an 8 m sphere.
* Rubble caps tangent penalties at **1e6**; reference caps all three at **1e10** (`demo.html:333`; reference `layout.ts:142-145`, `wgsl-solve.ts:603-606`). Instrument actual ring tangent maxima before changing this. A focused test must ramp the two tangents independently and preserve cone behavior; columns and friction oracles remain gates.
* Rubble equilibrates LDLT and uses pivot floor **1e-6**, clamps linear update to .175 m, angular update to .5, and inertial angular residual to .5 (`common.slang:313-568`, `world_step.slang:2033-2037,2169-2181`). Reference uses unscaled explicit LDLT with a diagonal-relative **1e-5** floor and applies the raw finite update (`wgsl-solve.ts:441-502`). These change hard-impact behavior. Do not remove protections for benchmark speed without independent SPD/ill-conditioned dense-solve tests and long-scene checks.

Current contact geometry already includes the reference's eight-point clipping, face-over-edge bias, all edge-axis SAT tests, quaternion cross-product rotation, sphere step-start lever arms, mass-scaled normal penalty, two independent friction penalties, unclamped cone test for penalty ramping and both friction Hessian rows. See `box_collide.slang:319-379`, `world_step.slang:1451-1462,2062-2108,2350-2408`, versus reference `wgsl-collision.ts:645-674,689-694,842-884` and `wgsl-solve.ts:180-213,423-435,588-612`. These should not be rediscovered as missing features.

## P0: broadphase ownership and hash collisions

### Large impactor duplication and order dependence

At audit start, `world_step.slang::broadphase_insert:464-469` puts the radius-8 ball in the large-dynamic list because radius > cell size (~1.224867). Every intersecting lower-index brick emits its ball pair in the list loop (`broadphase_query:543-557`). The same brick then tests the last collider in the endpoint fallback (`622-649`) and emits it again. `emitCandidate:360-378` and `narrowphase:974-995` do not deduplicate.

Conversely, if the large ball is placed **before** a small brick, the list loop's unconditional `dynamic && j<i` filter suppresses the brick's query. The ball is not in the grid, and its own query only covers ±1 cells around its center, not its radius-8 surface. Surface contacts several cells away are missed. This is a reason to fix ownership **before retrying Morton order**; it does not establish that every earlier Morton experiment failed for this reason.

Reference `wgsl-collision.ts::findPairs:234-273` distinguishes small/large query bodies. Small bodies always probe the large list; the ID filter applies only to large-large dynamic pairs. It has no first/last-collider special case. `findFixedLarge:277-310` supplies the remaining fixed-small/large-dynamic combinations.

**Minimal regression:** force the >512 grid path with 513 colliders: ground at 0, 510 separated filler bricks, a test brick at `[0,8.5,7.9]`, and sphere at `[0,8.5,0]`, radius 8. Run sphere-last and sphere-before-brick permutations. Require exactly one unordered candidate, manifold and sphere contact, and compare pair sets to independent brute force. Include large dynamic boxes and large-large ownership. A failing message should name both collider IDs, classification, emitted multiplicity and whether an endpoint fallback or ID filter is implicated.

**Parent follow-up:** the focused GPU regression reproduced two pairs before the fix, then passed sphere-last and sphere-first orders. Small-to-large ownership is now independent of IDs, large queriers skip small-grid scans, and the endpoint fallback is static-only. Larger ownership/capacity coverage remains useful; performance impact is unmeasured.

### Hash collision duplication

The linked grid stores only a hashed bucket head. `broadphase_query:566-619` walks all 27 neighboring **hash buckets**, without verifying that each candidate belongs to the requested coordinate cell. If two of the 27 cells hash to one bucket, the same eligible candidate is emitted twice. Geometry rejection removes distant false positives but cannot remove this duplication.

`gridHash:288-292` produces the same value for cells `(0,1,-1)` and `(0,-1,1)`. CPU enumeration found 144 repeated-bucket events among neighborhoods in ring-area integer cells; this number is **not a measured duplicate-pair count**. A concrete touching fixture uses lower-ID brick A at `[0,.75,0]`, higher-ID B at `[0,1.25,-.1]`, half sizes `[.5,.25,.25]`, with >512 colliders. A's cell is `(0,0,0)`; B's is `(0,1,-1)`, and the other aliased neighbor cell causes another traversal.

Reference `wgsl-collision.ts:253-261` stores each body's cell and explicitly rejects `cj != cell` before testing the pair. The parent has now added the corresponding exact-cell filter and the aliased-neighbor fixture passes. Broader coverage should include negative coordinates, distant same-hash cells, exact-touch boxes, and pair multiplicity against brute force. Capacity flags must remain zero. This is a correctness fix even if its speed effect is small.

### Generalization limits

Rubble currently allows only 16 large dynamic entries and eight planes, silently excluding additional entries from their lists; large statics are recovered only at indices 0 and last (`world_step.slang:443-469,543,561,622`). The ring needs one sphere and one ground, so these capacities are adequate **for this scene**, but they are not a general alternative to reference classification. Reference `radii.ts:12-13,46-55` has a 64-body large set and increases the small threshold when necessary, preserving coverage. Add explicit overflow reporting or a conservative fallback before advertising general mixed-size parity.

## P1: remove avoidable GPU work before redesigning the solver

### Immutable topology is still atomic in the primal

Reference `wgsl-solve.ts:23-29` binds adjacency, colors and counters as read-only plain `u32`. Rubble's `world_step.slang:2043-2048,2273` reads CSR ranges, every CSR ID and color count through `ALOAD`. Exported `generated/wgsl/solve_primal_color.wgsl:722-736,1615` confirms actual `atomicLoad`, not merely a source abstraction. These data are complete before solving; they do not need atomic reads in that phase.

The parent implemented WGSL-only **`RWStructuredBuffer<uint>` aliases at the same bindings 14/17**, read only by the primal; compiler/GPU focused validation passes. This avoids atomic operations while retaining the existing writable storage layout. The global atomic resources must stay atomic in construction/coloring kernels, and unused-resource elimination must remove the opposite declaration in every entry. Do not simultaneously bind overlapping read/write aliases. If later changed to `StructuredBuffer<uint>` / `var<storage, read>`, update host layout access to `read-only-storage`; the demo currently declares all nonuniform bindings writable (`demo.html:714-745`). Preserve writable counters for `pivotFloorHits`.

**Tests:** parse both primal entry modules with Naga and assert no atomic loads from the CSR/color-count bindings, then validate every other entry to catch duplicate bindings and layout errors. Preserve serial-versus-parallel lane tests, seeded body state and complete CSR prefix coverage. Measure primal GPU time separately. No claimed gain until measured.

### Full-body stores and hot body layout

At audit start, `primalApply:2161-2183` changed position and orientation but wrote `gBodies[bodyIdx]=b`. Generated WGSL emitted all 12 field stores (`1493-1504`), including velocities, mass, inertia, flags and initial pose. The reference `finishBody:499-502` writes only pose. The parent has now implemented two field stores, removing **84 logical bytes per body per iteration**, approximately 37.1 MB for four iterations at 110,441 active bodies, before cache/coalescing effects. This is a byte calculation, not a bandwidth or timing measurement; compiler and independent Newton/lane tests pass.

Reference also places current and initial poses in the first 64 body bytes (`layout.ts:169-179`). Rubble puts initial pose at byte 80 (`common.slang:5-21`), and primal loads full body structs. Removing dead fields from pair helpers can reduce live values and avoid dependent loads; actual compiler lowering must be inspected. Keep ABI changes separate from field-only stores because render code, native hosts and fracture code share it.

**Tests:** exact unchanged bytes for all non-pose fields after a primal solve; dense f64 Newton oracle for changed pose; generated-store-field whitelist; lane widths 1/2/4/8; live render ABI test. Field stores are a safer first step than body-layout migration.

### Diagnostic atomic hotspots in the default path

Rubble increments `gScratchAtomics[18..20]` for reuse hit/miss/generation-gate outcomes (`world_step.slang:902,909,969`), and `[21..23]` for reference-pose changes (`865-867`). These are global contended counters, unrelated to required physics counts. Reference `reuseContacts` and `refsWGSL` have no such default increments (`wgsl-collision.ts:32-46,696-706`). Gate them behind explicit diagnostics or reduce per workgroup only when requested. Required counts, capacity flags and correctness reporting must remain enabled.

**Tests:** diagnostics on/off produce identical contacts and seeded body states; diagnostic mode reports exact expected path counts; default generated modules contain no writes to those diagnostic offsets. A direct count microbenchmark can validate the mechanism, but only full-ring A/B establishes relevance.

### Contact ABI and pair-centric warm start

Reference `layout.ts:74-88,195-213` uses a 32-byte manifold plus **64-byte point**, storing IDs, normal and friction once per pair and deriving tangents. Rubble `common.slang:56-91` has the same 32-byte manifold **plus a 128-byte point that repeats IDs, normal, friction and two tangent vectors**. At 1.285M points this is about 82 MB additional storage per current set, or 164 MB across two sets. Four-iteration primal and dual repeatedly touch these records. A compact internal point format is the largest clear memory opportunity; preserving the public 128-byte debug/schema format by conversion is possible.

Reference hash insertion dispatches one invocation per previous manifold (`wgsl-collision.ts:347-362`), with count/generation available directly. Rubble dispatches over all previous contacts, loads current and previous head candidates, and only then returns for non-heads (`world_step.slang:1242-1280`). Rubble's separate `persist_contacts` repeats body/collider/hash work per point (`1300-1462`); reference narrowphase matches and warm-starts an entire pair together (`wgsl-collision.ts:821-884`). Migrate pair metadata and points coherently, rather than only cutting record bytes.

**Tests:** exact ABI/packing round trip; all eight distinct corners, including duplicate feature IDs; two independently ramped tangent penalties; generation boundaries and scene reset; a reused pair versus freshly regenerated pair; simultaneous hash collisions; large tiled dispatches; columns180, ring480, gables240 and dense Newton/Metal oracles. Preserve nearest-anchor disambiguation in Rubble: reference's first matching feature is not automatically stronger than our corrected duplicate-feature matching.

**Tradeoff:** Reference dual is one thread per manifold (`wgsl-solve.ts:588-623`), loading pair bodies once; Rubble deliberately uses per-point dual (`world_step.slang:2325-2411`) for occupancy. Earlier grouped-dual experiments were discarded. Compact storage is independently valuable; do not blindly restore grouped dual or claim it must be faster on Metal/WebGPU.

## Graph, locality and reproducibility

Reference shared `src/avbd2d/gpu/wgsl-topology.ts` (imported by the 3D solver), lines 120–150, sorts adjacency by other body plus secondary physical key. Rubble `csr_sort:1749-1779` sorts joint/manifold **append IDs**. Atomic append order changes across runs, so this is not a stable physical summation order. Reference also uses ping-pong Jones–Plassmann rounds (`wgsl-topology.ts:225-254`), whereas Rubble's round reads/writes the same atomic color array (`world_step.slang:1822-1871`): valid colorings can converge at different rates depending on within-dispatch visibility. Neither observation alone proves a coloring clash; both limit reproducible timing and trajectory comparison.

**Proposal/tests:** sort by neighbor identity (collider identity where multiple colliders per body require it), then a stable manifold secondary key. Feed shuffled manifold append orders and require identical physical adjacency sequences and single-primal results. Repeat scene resets with fixed capacity and compare checkpoints. If deterministic coloring is required, test ping-pong rounds independently; preserve the current compact pending-body list to avoid restoring full-body work unnecessarily.

Reference `colorCount/colorScatter`, `wgsl-topology.ts:256-320`, builds per-workgroup histograms and stable ranks, then prefix-scans/scatters contiguous index-ordered buckets. Rubble `pack_color_lists:1981-2000` uses a global atomic counter per color and appends from an atomically constructed awake list. This forfeits spatial coherence even before Morton ordering. Stable packing also reduces contention on the two huge ring colors. It is a more targeted experiment than a whole body reorder.

Reference `solver.ts:409-439,635` defaults to Morton body order. Rubble ring retains builder order (`demo.html:604-605`). Reference historical numbers in `FINDINGS.md:828-847` demonstrate benefits on its pile, not a promised ring improvement here. Fix pair ownership first, test stable packing second, and only then retry Morton with an explicit identity map and full-contact-set comparison.

**Tests for packing/order:** each active body appears exactly once; bucket counts sum to active count; each edge has different endpoint colors; IDs monotonic inside stable buckets; arbitrary awake-list order and reset; no changes to colliders/ball identity, original-order regression metrics or warm-start keys. Measure graph time plus primal time, since a more expensive pack can still win overall. No sleeping or reduced body count may be used to manufacture the gain.

The reference adapts its color cap and rounds after counter reads (`solver.ts:1680-1706`). Rubble issues 32 JP rounds and **128 primal indirect commands per four-iteration step** (`demo.html:919-936`), even if many dispatch counts are zero. Current compact pending-color dispatch already reduces actual JP threads substantially. Lowering the host color cap using a stale previous count can omit newly created colors during ten-step benchmark batches; retaining GPU coverage is mandatory. Adaptive command coverage needs a tested fallback/tail, not just `last.color`.

The reference's sparse tail fusion is **opt-in and unsuitable for this full active ring policy**: its host restricts fusion to ≤1,024 active bodies (`solver.ts:1460-1472`), and `FINDINGS.md` ends with a real replay divergence that caused the feature to be disabled by default. Do not copy it as a proven safe optimization.

## Collision grid, hash and shader scheduling

Reference builds a counting-sort grid: count, prefix scan, scatter, explicit cell filter (`wgsl-collision.ts:99-114,234-273`). Rubble uses an atomic-exchange linked list (`world_step.slang:471-477`) and a fixed 131,072-slot hash, versus reference table size at least twice body capacity (`solver.ts:607-608`). Dependent pointer chasing and repeated AABB loads are plausible bottlenecks. Reference has extra scan/clear work; replacing the grid only makes sense after the pair correctness repairs and phase measurements.

Rubble precomputes 80-byte world OBBs in `build_aabbs:269-282`; reference computes axes from quaternion in `shapeOf` (`wgsl-collision.ts:145-170`). This is a **Rubble strength/tradeoff**, not an omission: cached geometry saves arithmetic but adds bandwidth and storage. Test a fused/cache-free variant only if collision timing warrants it.

Reference warm-start hash uses exchange-with-displaced-occupant insertion, carrying the evicted value along the probe chain (`wgsl-collision.ts:340-362`), and probes up to table size. Rubble uses nested bounded weak-CAS retries and 64-slot search (`world_step.slang:1242-1297`, `atomics.slang:15-31`). The original weak-CAS theory is not sufficient evidence for any remaining instability. The exchange algorithm is not equivalent to simply overwriting a slot; a prior incorrect exchange experiment already regressed Metal. If revisited, prove every stored key remains reachable and report insertion exhaustion explicitly. Test adversarial identical initial hashes, wraparound, duplicates, high load, all backends and the exact weak-CAS success contract.

Both primal kernels use 64 threads and 1/2/4/8 lanes/body. Reference practical thresholds are 2 lanes from 32768, 4 from 4096, 8 below, with one lane effectively disabled at 2^31 (`solver.ts:483-487`, `layout.ts:319-329`). Rubble uses one lane at 65536 (`world_step.slang::primalLanesFor`). Typical ring's large colors can be below that threshold; measure bucket distribution before blaming the threshold. Preserve lane-equivalence tests. The current Rubble reduction already transfers only 21 lower-triangle entries and six RHS values from 32 senders; blindly copying the reference's padded `Acc` shared memory can be worse.

Rubble's shared source deliberately supports colliders, joints, sleeping, commands and multiple backends. The ring has no joints and no sleepers. Per-ring specialization may remove dead branches, collider indirection and body flag checks, but only if generated instruction/register evidence justifies the added variants. The current host already skips zero-joint phases. No capability should silently disappear from the general backend.

## Host loop, memory and fair measurements

Reference `GpuSolver3D.step` submits without waiting (`solver.ts:1450-1590`); periodic viewer reads are asynchronous (`gpu/sim.ts:142-167`). Rubble live `frame` submits physics plus render and then awaits `staging.mapAsync` before scheduling the next tick (`demo.html:1253-1288`). That is a latency-sensitive serialized loop. A staging ring with infrequent nonblocking HUD reads can improve live smoothness independently of physics throughput; retain asynchronous error handling and do not recycle mapped buffers. GPU counts already drive dispatch, so HUD freshness must not become a dependency again.

Rubble's `sceneCap` uses rounded fixed capacities and one collider allocation sized for twice the bodies (`demo.html:102-118`), while reference sizes contact/manifold/hash capacity from a CPU initial touching-pair estimate and grows before overflow (`solver.ts:157-161,646-680,770-803,1710ff`). The 128-byte point format dominates Rubble's allocation. Reference lower allocation is an advantage for adapter limits and residency, but shrinking a buffer does not itself reduce active solve traffic. Preserve headroom and explicit overflow failures; never drop contacts to achieve 8.333 ms.

Reference's ring benchmark uses lead=60, four iterations, then `compare(rounds:2,batches:5,perBatch:10,warmup:0)` (`bench-cases.ts:82,187-214`). The helper adds three untimed steps before each round (`avbd2d/gpu/timing.ts:24-43`). Thus timed windows are steps 64–113 and 117–166, not arbitrary settled phases. Parent's new `physics-benchmark.mjs` mirrors this and retains raw batches. Reference final reported contacts/colors follow ten additional profiled steps, so they are not exactly the timed-window endpoint unless reported separately.

A publishable comparison must:

1. Use the same adapter/browser/power state, one active benchmark workload, no concurrent native GPU tests or another live demo. Record competing work; contaminated A/B is diagnostic only.
2. Rebuild/reset each solver to the same scene and state, run the same lead and exact timed windows, keep all 110,441 dynamics active, four iterations and dt=1/60. Do not compare one implementation after collapse against the other's intact ring.
3. Record raw batch times, both round medians, best-round statistic for reference compatibility, adapter/UA, source and shader hashes, wall-clock time, memory capacity, initial/final contact/manifold/awake/color/overflow counts. Benchmark correctness assertions must abort invalid runs.
4. Alternate independent fresh runs A/B/A or B/A/B. Repeat until controls are reasonably stable; variation is a result to report, not hide behind the best sample.
5. Separate queued **physics throughput**, isolated GPU phase timestamps, and live render frame FPS. Timestamps from extra pass boundaries can perturb scheduling. Never equate ten-step batch means with a guaranteed maximum individual-step latency.
6. Require correctness checks outside timing: full ring480 including reset, columns180 including100k without sleep, gables240 as shared-shader regression, lane equivalence, dense f64 Newton, Metal `oracle_accuracy` / `avbd_slice` / WGSL parity. A number ≤8.333 ms does not independently establish parity.

The parent is running these measurements and validations serially. No timing result in this audit certifies ≥120 FPS. Historical M4 Max results in the reference's `FINDINGS.md` are useful evidence for optimization hypotheses, but cannot substitute for the current M3 Max/browser measurements.

## Suggested implementation sequence

1. Finish unique, order-independent pairs and hash-cell identity, with GPU brute-force diagnostics. These repair physical work before measuring it.
2. Land isolated low-risk generated-code reductions: body pose stores, readonly primal topology, opt-in diagnostic atomics. Preserve columns and oracles after each meaningful change.
3. Measure stable color packing; fix physical CSR ordering for reproducibility. Profile collision/primal/dual on the same full-ring windows.
4. Design compact internal contacts and pair metadata with explicit ABI migration boundaries. This is the strongest remaining memory opportunity, but must be kept separate from speculative math changes and from the independent fracture work.
5. Optimize the measured residual: counting-sort grid, pair-centric persistence, lane thresholds or safe command scheduling. Reject experiments that merely shift cost or loosen physics.

Useful existing test entry points: `tests/scene_builders.test.mjs`, `src/gpu_tests.rs`, `tests/ring_hold.rs`, `ring-regression.mjs`, `tests/columns_hold.rs`, `tests/gables_hold.rs`, `engine-metal/tests/oracle_accuracy.rs`, `engine-metal/tests/avbd_slice.rs`. Failures should state the physical invariant, offending body/pair/field, expected/actual value and likely violated stage contract. Avoid tests that only search source text when a generated IR or actual GPU invariant can be asserted.

## Validation update from implementation work

After the audit-driven pair and primal changes, all 19 WebGPU library tests pass,
including generated-WGSL view/store checks and the independent dense Newton oracle.
The full native ring480, gables240 and four columns tests also pass without relaxed
bounds. Browser and Metal verification are tracked in `avbd-regression-notes.md`.
These results establish regression coverage, not a measured speedup or 120 FPS.

The follow-up implementation removes the unused reuse/reference-pose diagnostic
atomics entirely; a generated-IR regression rejects their buffer in production
entries. A diagnostic A/B/A did not establish an overall speedup (see
`avbd-ring-reuse-statistics-ab-2026-10-14.json`). Stable body-index color packing
now uses tile histograms, exclusive prefixes and scatter in Slang, with focused
WGSL/Metal coverage. All 21 WebGPU library tests, four columns cases, gables240
and ring480 pass. Browser ring480/reset and all Metal oracles also pass. The packing A/B/A
measured 227.20/225.70/229.61 ms under competing load, which does not establish
a speedup; see `avbd-ring-stable-packing-ab-2026-10-14.json`.

A subsequent opt-in solver-only compact-point experiment is implemented and tested
in `avbd-compact-solver.md`. It retains canonical persistence/debug records and
therefore adds scratch allocation. Native and browser correctness gates pass, but
contended compact/canonical/compact timings of 232.81/233.59/247.20 ms do not show a
repeatable gain. Canonical remains the default. Full compact persistent storage
and fair isolated performance remain outstanding; the ABI difference is not yet
resolved by this intermediate experiment.
