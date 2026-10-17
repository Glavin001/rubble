# AVBD regression evidence

Reference: `inspiration/pinned/three-avbd`, `brickRing(solver)` and
`GpuSolver3D`, four iterations, dt=1/60, default GPU parameters. Measurements on
2026-10-14 used Chromium WebGPU on the local Apple GPU. Map reference body IDs
through `gpu.refToGpu` before reading GPU snapshots: the reference sorts uploads
by Morton order. Positions swap Y/Z; quaternions and angular velocities must also
change sign because that swap reverses handedness.

The pinned builder creates **110,440 bricks plus one sphere and one ground body**.
Its older comment saying 110,332 bricks is stale. Gables has 505,920 bricks, two
spheres and one ground body. Tests compare actual builders, not those comments.

## Ring physical baseline

Flanks are bricks whose initial |x| > 40 m. Drift is absolute vertical movement
from the initial pose. The ball travels along +Z in Rubble (+Y in the reference).
Values vary slightly with GPU scheduling; the tests allow numerical variation.

| Completed steps | Reference max flank drift (m) | Mean flank drift (m) | Ball forward coordinate (m) | Ball height (m) |
|---:|---:|---:|---:|---:|
| 1 | 0.00076 | 0.00045 | -94.70 | 8.500 |
| 30 | 0.245 | 0.0585 | -81.00 | 9.037 |
| 60 | 0.277 | 0.0917 | -67.60 | 9.022 |
| 120 | 0.313 | 0.1027 | -41.33 | 8.482 |
| 240 | 7.790 | 0.2344 | 10.06 | 8.487 |
| 480 | 19.533 | 1.0159 | 105.46 | 8.464 |

Some distant bricks fall after impact propagates. Requiring every flank brick to
stay within 0.5 m for all eight seconds would reject the reference itself. The
regression instead requires max drift <0.5 m for the first two seconds and mean
drift <0.2/0.5/1.6 m at two/four/eight seconds. The ball must clear both walls
(forward coordinate >90 m) and stay near ground level at the sampled steps.

The first reference frame has 1,284,737 contacts. Rubble after exact-touch and
full-polygon fixes has 1,285,073 (small floating-point/SAT ordering differences).
The test accepts 1.25–1.32 million, catching lost exact-touch contacts and polygon
truncation while permitting those differences.

## Reproduced failures

- CSR finalize overwrote inclusive values while another workgroup still read
  them. The isolated 25,001-body test found offset 24,992 =124,953 instead of
  124,960. Finalization now writes the other buffer, then publishes separately.
- Quantized feature IDs aliased on a large ground slab. Warm start selected one
  corner for every contact, collapsing the support polygon. Equal-ID matches
  now choose the closest pair of local anchors.
- A 2^22-thread hash clear was clamped at 65,535 workgroups, leaving its last 64
  entries stale. Two-dimensional dispatch covers the entire range and protects
  the sentinel beyond its logical end.
- Previous-frame estimates under-dispatched newly generated contacts. GPU
  indirect arguments now use current candidate, contact, manifold and awake counts.
- Ring yaw and rolling angular velocity did not reverse handedness on Y/Z
  exchange. Scene tests compare all bodies with the pinned implementation.
- Rotated face polygons were capped at their first four vertices. An octagonal
  overlap test now requires all eight unique corners and a centered centroid.
- Expanded quaternion rotation scaled the rotation axis by the rounded squared
  quaternion norm. Exactly touching rotated boxes lost all contacts. The
  cross-product form preserves the axis, matching three-avbd.
- Friction penalties were tested after projection onto the friction cone and
  shared one scalar for both tangents. A saturated test row increased from 10 to
  ~4010 in one update. The raw force now determines eligibility and the two
  tangents ramp independently; the second penalty uses the existing padding word
  at byte 124, preserving the 128-byte Contact ABI.

These are correctness gates. Passing them does not establish the broader 120 FPS
or three-avbd performance objective; measure that separately on active scenes.

## Additional concurrency regression

Checking only poses missed an unfinished coloring at step 60: body 56,964 still
had color `0xffffffff`, and `pack_color_lists` silently assigned it to color zero.
The host now runs 32 JP rounds, independently of the number of colors, and submits
the entire palette through GPU indirect arguments. Packing reports `OVERFLOW_COLORS`
for any unfinished or out-of-range color. The full ring test reads every active
body color and contact manifold at its checkpoints and rejects shared colors on
connected active bodies, with body and manifold IDs in the failure.

## Performance and playback status

### Full-size gables startup and impact

The native wgpu 506k fixture initially stalled in `Queue::write_buffer` staging
allocation while inserting each body and collider separately, before physics
started. A process stack sample identified the allocation wait. Bulk insertion
now uploads current poses, previous poses, and colliders in three contiguous
writes. A GPU round-trip test covers appending after existing records, empty
batches, collider associations, and initialization of previous poses.

With bulk upload, the 505,922-active-body startup test completes and verifies:

- Exactly 3,916,802 first-frame contacts: 1,088 gables × (870 inter-course
  supports + 30 ground supports) × four points, plus two sphere contacts.
- No overflow, all bodies awake, finite state, and valid contact coloring.
- Maximum brick height drift at steps 1/10/20: 0.000828/0.034250/0.175831 m.

The test now continues through 240 impact steps. A fresh pinned-reference capture
is saved in `avbd-gables-reference.json`, including source hashes; the reproducible
capture page is `crates/engine-webgpu/tests/reference-gables.html`. It maps sorted
GPU body IDs back through `refToGpu` and converts Z-up positions to Y-up.

At step 240, the 442,680 bricks in columns untouched by either ball have:

| Implementation | Mean height drift | Maximum height drift | Ball forward positions |
| --- | ---: | ---: | --- |
| pinned three-avbd | 0.117816 m | 0.328042 m | −74.0336 / −74.0800 m |
| native wgpu | 0.118901 m | 0.328536 m | −74.3753 / −74.0789 m |
| browser WebGPU, batches of 10 | 0.118080 m | 0.438074 m | −74.3666 / −74.1437 m |

The native and browser regressions require mean unhit-wall drift <0.2 m and
maximum drift <0.6 m, plus both impactors within 1.5 m of reference forward
positions, within 1 m of their original lateral line, and at heights 4.6–4.95 m.
These bounds allow GPU ordering variation while rejecting collapsed remote walls,
stalled/reversed balls, and floor penetration. Every sampled body must remain
finite, every step must remain awake and overflow-free, and the native test also
checks every active contact's coloring at checkpoints. The browser verifies reset
and GPU-only warm-start metadata between readbacks.

These results establish bounded agreement through 240 steps. They do not establish
longer-duration agreement or fair performance parity.
The large gables demo now also disables sleep during ordinary playback/benchmarking,
matching the reference's active workload rather than timing settled sleeping walls.

### Solver checks and timing

An independent contact-primal regression assembles the dense 6×6 Newton system
in f64 on the CPU and solves it with Gaussian elimination. It compares the GPU's
translation and rotation for both contact endpoints, off-center anchors, unequal
tangent penalties, and sticking/sliding forces. Deliberately removing the second
tangent Hessian row fails with the endpoint, friction regime, pose component and
expected/actual values. This complements cooperative-versus-serial tests, which
cannot detect a mathematical error shared by both GPU paths.

A trial that shortened contact Jacobian lifetimes passed correctness but was
reverted pending controlled timing. Its browser run overlapped a separate active
Chrome fracture session; the observed 171.86 ms is not a fair performance result.

Two concrete performance differences remain to measure: Rubble's Contact record
is 128 bytes versus the reference's 64-byte per-point record plus shared manifold
data; and Rubble switches to one primal lane at 65,536 bodies per color while the
pinned reference's threshold is 2^31 (two lanes above 32,768 in practical scenes).
Neither difference by itself proves a speedup from changing it.

Incremental coloring preserves valid colors and compacts only local priority
maxima before invalidating new conflicts. The history and compacted snapshot
occupy separate rows beyond packed color lists; first-step reset ignores history.
Pending body IDs temporarily use the CCD buffer, which velocity extraction clears
before sleep/CCD processing. Counter byte 52 and indirect slot 69 publish the
pending workload. Hosts opt in with scratch/uniform word 26; the separate fracture
wrapper can retain its existing layout and full recoloring behavior.

The focused GPU test covers stable graphs (zero pending dispatch), newly added
contacts, waking neighbors, invalid/absent history, scene reset, mixed joint and
manifold adjacency, and shuffled awake-list indices. It checks independent edge
color validity and untouched history during initialization, then verifies history
publication and the reported color count. It also requires isolated/local-maximum
bodies to compact toward smaller colors. Skipping compaction is not an accepted
shortcut: the initial reuse-only trial failed gables at step 240 with 2.045 m
maximum untouched-wall drift, versus the unchanged 0.6 m limit.

With compaction, native gables at step 240 has mean/max untouched-wall drift
0.118633/0.330484 m and only 285 pending bodies out of 505,922 active. Native ring
passes 480 steps with 165 pending bodies at its final checkpoint. Browser ring
also passes 480 steps and reset: final mean flank drift 1.2133 m, ball forward
position 104.4138 m, no overflow. Metal's slice/accuracy oracles and 16k/25k/100k
column parity checks remain green. Browser gables also passes 240 steps and reset:
final mean/max untouched-wall drift is 0.117733/0.330029 m, both balls remain within
the reference bounds, and all 505,922 dynamic bodies remain awake with no overflow.

Browser diagnostic A/B/A runs (`bench=1`, `coloring=rebuild/reuse/rebuild`) measured
248.11/139.80/343.41 ms per step. Graph phase medians were 123.14/27.07/118.95 ms.
A separate fracture GPU workload was active, and the control's large drift makes
these unsuitable as isolated speedup or three-avbd parity evidence. They support
continuing the experiment; fair performance and the 120 FPS objective remain open.

The color-argument writer now bounds its count-buffer reads by the configured
palette. Previously it launched 64 threads and read 64 entries even when hosts
allocated only 32. WGSL robust access can conceal that invalid read; native Metal
does not provide the same guarantee. Focused tests on both backends exercise
palettes of 1, 7 and 32, empty buckets, and canaries in all unused color/work
argument slots. The WebGPU test additionally covers each primal lane threshold.
The test fails on the old shader at palette 1, slot 1 and passes with the bound.
After this fix, all 14 WebGPU library tests, the direct Metal bounds test, four
column tests, the 240-step gables test, the 480-step ring test, seven AVBD slice
tests and five accuracy oracles pass. This is correctness evidence, not evidence
that the observed live-playback slowdown or the performance gap against
three-avbd has been resolved.

CSR construction now uses three block-scan passes: scan 256 bodies per workgroup,
scan the block totals, then add block offsets and publish scatter cursors. The
previous global scan required roughly 20 full-buffer passes at ring scale. GPU
tests compare all entries through 524,288 bodies with variable, zero, and boundary
spike degrees, and check that the end sentinel is untouched. Metal must launch
complete 64-lane groups, including padding; its ordinary 32-lane SIMD-width
launcher is invalid for this cooperative kernel. A direct Metal regression tests
that contract without allowing corrupt offsets to reach scatter.

The browser's 480-step ring check passes with this scan, but the observed total
physics time remains 20.85 ms (graph phase 1.38 ms). This is not evidence that
the overall performance gap is closed.

The latest retained browser measurements are approximately 20.8 ms per physics
step versus 10.2 ms for the pinned three-avbd ring on this machine. Correctness
passes do not close that performance gap. Retained changes reduce shared-memory
traffic in the primal reduction, gather reused contacts in the parallel persistence
pass, and write only changed fields in the dual update. A serial-versus-cooperative
primal test and a reused-contact gather test cover their numerical invariants.

Normal demo playback no longer triggers a benchmark that pauses and resets the
scene. Use `&bench=1` explicitly for measurement. `&check=ring` intentionally runs
480 physics steps without continuous rendering and stops at its result; it now
labels that behavior and links to uninterrupted playback. Never time either
implementation while another GPU test or demo is running.

## 2026-10-14 audit-driven ring changes

The full comparison is in `avbd-ring-reference-audit.md`. The latest implementation
removes unused atomic counter snapshots, reads completed primal topology without
atomics, publishes only changed body pose fields, and uses GPU-written zero-sized
sleep dispatches when no sleep candidates exist. Browser playback no longer maps
an unused body-read buffer. These changes preserve the physical workload.

The audit also found actual broadphase errors: duplicate last-body impactor pairs,
missing surface pairs when the large body came first, and duplicate small pairs
when neighboring cells shared a hash bucket. The focused >512-collider GPU test
failed with two brick/ball pairs before repair; it now requires one candidate and
manifold, one sphere contact (or four box-face contacts), and no overflow.
Initial ring contacts are now 1,284,737 rather than 1,285,073: 336 duplicate points
were removed, not truncated by capacity. Full native ring480 passes: final mean
flank drift 1.1372 m, ball forward 105.50082 m, all 110,441 awake, no overflow. Native
gables240 passes with unchanged bounds: untouched-wall mean/max drift
0.118634/0.330526 m. All four columns cases and all 19 library checks pass.

Before the latest pair/primal changes, a fresh same-browser diagnostic measured
Rubble 239.95 ms/step and pinned three-avbd 220.90 ms/step. Both are dramatically
slower than earlier isolated measurements. The GPU reported 100% utilization even
when our browser benchmark and native test processes were idle; no system thermal
or performance warning was recorded. These numbers are contaminated diagnostics,
not evidence of a speedup, fair parity, or an 8.333 ms result.

`bench=only` now runs the pinned 60-step lead-in and 2×5×10 timing window, records
every batch plus both medians/adapter/browser/workload metadata, then stops. The
benchmark test verifies GPU completion is awaited, CPU encoding is included,
pre-existing work/untimed steps are excluded, and slow batches remain visible even
when the best median is below 8.333 ms. Measurements reject nonfinite clocks and
device failures. Raw batch averages do not establish maximum individual-step
latency; correctness checks and isolated A/B/A measurements remain separate gates.

After the pair/primal changes, all Metal checks also pass: two library scan/dispatch
checks, seven `avbd_slice` tests, three 16k/25k/100k WGSL parity tests, and five
`oracle_accuracy` tests. Maximum column position difference remains 0.000779 m.

The final Chrome ring run passes 480 steps and scene reset: final mean flank drift
1.219147 m, ball forward 105.217781 m, all 110,441 bodies awake and overflow 0.
Its diagnostic best-round median is 223.12 ms; second-round median 236.80 ms.
Raw batches, phase medians, hardware/browser metadata, source hashes and explicit
isolation caveats are saved in `avbd-ring-diagnostic-2026-10-14.json`. The sleep
phase measures 0.07 ms in this run, but current conditions do not establish an
isolated overall speedup. The 120 FPS / three-avbd parity objective remains open.

## Stable color packing and production statistics (2026-10-14)

Removed six unused shared diagnostic increments from reference-pose update and
contact reuse. The generated-IR regression fails on the previous entries and
requires both production shaders to omit the diagnostic scratch binding. A/B/A
results are preserved in `avbd-ring-reuse-statistics-ab-2026-10-14.json`; controls
and workload phases drifted, so these data do not establish an overall speedup.

Stable color buckets now preserve body-index order using per-tile histograms,
exclusive prefixes and scatter. WGSL tests independently derive expected buckets
and prefixes, reverse the awake list, cover static/sleeping bodies, invalid colors,
empty/partial tiles and history/capacity sentinels. Metal checks complete 64-thread
launches. The old atomic packing remains available as `packing=atomic` for browser
comparisons. Neither mode changes active body count or solver iterations.

All 21 WebGPU library checks pass. The four columns tests, gables240 and ring480
pass with unchanged bounds. Gables unhit-wall mean/max drift is 0.118610/0.329382 m.
Ring final mean flank drift is 1.1956 m, ball forward position 105.31321 m; all
110,441 bodies remain awake and no overflow occurs. Metal's three library checks,
seven physics slice checks, three columns parity checks and five accuracy oracles
pass. Maximum column position discrepancy remains 0.000779 m. The eight JavaScript
scene/timing tests also pass. Browser validation and packing timing are recorded below.

The packing A/B/A (atomic/stable/atomic) measured best-round medians of
227.20/225.70/229.61 ms; second-round medians were 245.58/250.32/250.40 ms.
These contended diagnostics do not establish a stable-packing speedup or 120 FPS.
`avbd-ring-stable-packing-ab-2026-10-14.json` retains all batch means, both round
medians, separately profiled phases, adapter/browser metadata and source/shader
hashes. All runs used 110,441 awake bodies, four iterations and zero overflow.
No native test or compiler workload overlapped these runs; other GPU work was
not isolated. Stable ordering is retained with a tested atomic control available,
while its performance value remains open.

The stable-packing browser run passes all 480 steps and scene reset. Initial
contacts are 1,284,737; final mean flank drift is 1.253236 m and the ball reaches
105.415550 m. All 110,441 dynamics remain awake, overflow is zero and final color
count is 12. The final checkpoint is included in the packing diagnostic JSON.
The broader 120 FPS / three-avbd parity objective remains open.

## Compact solver contact experiment (2026-10-14)

An opt-in 64-byte solver point format retains the public 128-byte persistence ABI.
See `avbd-compact-solver.md` for fields, allocation costs and migration boundaries.
25 focused WGSL checks pass across the full library run and added cooperative
oracle check. Compact-mode columns180 (four cases), gables240 and ring480 pass;
Metal library/slice/columns parity/accuracy gates also pass. The browser ring480
and reset pass with mean flank drift 1.170003 m and ball forward 105.235405 m, all
110,441 awake and zero overflow. Ten JS scene/timing/profiling checks pass.

The profiler had a compact-only reporting bug, reproduced in a new automated
orchestration test and repaired. The failed report is excluded. Three fresh runs
with the repaired profiler (compact/canonical/compact) measured
232.81/233.59/247.20 ms. These are contended diagnostics, not a demonstrated gain;
canonical remains the default. Raw samples and source/shader hashes are in
`avbd-ring-compact-contact-ab-2026-10-14.json`. The 120 FPS / three-avbd parity
objective remains open, including an isolated performance comparison.

## Paired profiling and physical adjacency (2026-10-14)

The earlier reports labelled “contended” did not establish the source of the
slowdown. Treat them as diagnostics with unverified isolation. Current driver
utilization alone cannot identify a competing process. The profiler now retains
raw, same-submission GPU phase/span and CPU completion measurements instead of
only a string of separately aggregated phase medians. Six profiler orchestration
tests plus three batch-timing tests pass, including rejection of mismatched phase
counts/reversed timestamps and resource cleanup after readback failure.

Physical adjacency sorting orders contacts by neighboring body and unordered
collider pair, independent of atomic manifold append order. Small rows use
insertion sort; larger rows use a heap with chunked sweeps to avoid the WGSL
65,536-iteration safety cap. The new GPU fixture first failed on append ordering
and now passes shuffled/reversed append permutations, compound collider pairs,
joints, short rows, 65,537-contact rows, and row canaries. The full native ring480
check passes: 1,284,737 initial contacts, final mean flank drift 1.1830 m and ball
position [0.145814, 8.493586, 105.39549], all 110,441 dynamic bodies awake. Columns,
gables and Metal validation of this new ordering were pending at this checkpoint
(and passed in the post-reboot validation below). This is not
a demonstrated performance improvement.

After Codex restarted, the unchanged pinned reference ring measured 216.77 ms/step
and Rubble's canonical/append-order control measured 227.20 ms/step (second-round
median 243.20). The native ring regression had completed before both browser runs;
no native tests or compiler work overlapped them. Thus stopping that test did not
restore the earlier baseline. No specific competing external GPU process has
been established. Raw Rubble batches and paired phase samples are retained in
`avbd-ring-perf-retry-2026-10-14.json`. In the same step, the summed phase durations
can exceed the first-to-last timestamp span: step 174 is 571.736 vs 322.437 ms, with
327.9 ms CPU completion. The phase values must not be treated as additive cost
percentages; independently aggregated medians alone do not explain this discrepancy.

## Independent dispatch/memory diagnostic

Added browser `dispatch-diagnostic.html` and native `gpu_dispatch_probe`, sharing
a tiny WGSL increment shader. Both use the pinned batch timing protocol and
validate every word after timing (8,388,608 words for the 32 MiB case). The probes
contain no physics or Slang. A default-64-thread browser run takes 0.19 ms for
128 tiny dispatches but 11.71 ms for one full 32 MiB update. Native wgpu/Metal
measures 0.189 and 11.936 ms respectively. A 256-thread native group does not
restore the larger workload (12.799 ms). Later browser batches remain variable:
2.79–21.59 ms for the single-dispatch case, with the page visible and focused.
The controls exclude simple dispatch overhead as the explanation for hundreds
of milliseconds of physics work, but do not establish a hardware fault or
external GPU contention. There is no historical baseline for this new probe.
Selected medians and raw large-workload samples are retained in
`avbd-dispatch-diagnostic-2026-10-14.json`. Native and browser runs were serial.

## Recovery after full Mac reboot (2026-10-14)

Serial browser runs after the user restarted macOS restored normal-scale timing:
three-avbd ring 10.22 ms/step, Rubble append-order control 19.99 ms/step, and
Rubble physical-order default 20.29 ms/step. Before reboot the paired reference
and append control measured 216.77 and 227.20 ms respectively. No native GPU
tests or compiler work overlapped these runs. Both Rubble variants retained
110,441 awake dynamic bodies, four iterations, canonical contacts, stable color
packing, and zero overflow. The reboot recovery does not identify the original
cause. Physical ordering has no demonstrated speed benefit in this comparison.

Current default phase medians: primal 6.42, narrowphase 4.85, dual 3.60, grid
1.64, warm start 1.38, and graph 1.31 ms. These are later split-pass profiling
samples, not the batched throughput window. The 120 FPS goal still requires
8.33 ms/step and remains unmet. Batch samples, metadata and source hashes are
in `avbd-ring-post-reboot-2026-10-14.json`. This timing retry does not replace
the outstanding columns/gables/Metal correctness checks for physical sorting.

## Compact ring default after repeatable measurements

With normal performance restored, full-active compact/canonical/compact browser
runs measured 17.93/19.98/17.90 ms per step. Compact points now default on for
`ring110k`; `contacts=canonical` retains the control and other scenes remain
opt-in. The extra scratch costs 128 MiB at ring capacity. The shipped default
passed browser ring480 plus reset and then measured 18.10 ms, with 110,441 awake
bodies at both ends of the timed window and zero overflow.

Native compact library26, columns4, ring480 and gables240 pass with physical
adjacency. Metal library3, slice7, columns parity3 and accuracy5 pass; maximum
columns position delta remains 0.000779 m. Expanded compact publication coverage
passes static/moving/sleeping/sphere cases in both contact halves and requires
inactive endpoints to preserve all canonical bits. Ten timing/profiling and five
body-for-body scene parity JS tests pass. Benchmark workload guards now reject
reduced active counts, iterations, altered dt, absent collision work and overflow
with stage-specific diagnostics, outside the timing intervals. Import versioning
prevents existing browser caches from retaining the old benchmark module API.

Raw batches, workload counters, source hashes and regression summaries are in
`avbd-ring-compact-post-reboot-2026-10-14.json`. This is about a 10% wall-time
reduction, not 120 FPS or reference speed parity. Persistent contact compaction
and pair-level persistence remain major opportunities; canonical persistence
is still 128 bytes per point.

The canonical path also passes all four columns cases and gables240 with physical
adjacency: 505,922 awake, unhit-wall mean/max drift 0.118615/0.508853 m, zero
overflow. The remaining work is performance optimization and broader objective
validation, not a known failure of these regression gates.

## Persistence scheduling experiments

A shared point-update function allows one thread per manifold to load immutable
pair bodies once. A compact variant also writes solver points from the warm-start
result, preserving the tag clear but eliminating the separate packing pass.
Neither changes warm-start matching, nearest-corner disambiguation or decay.
Both variants remain opt-in. Serial manifold/contact/manifold full-ring runs
measured 17.98/17.80/17.93 ms; fused/contact/fused measured 17.53/17.52/17.49 ms.
There is no repeatable total-wall advantage. Lower fused primal phase time partly
reflects moving preparation into warm start, not a faster primal solve.

All 28 focused GPU/compiler checks pass. New tests compare canonical publication
and live compact state across 67 ragged manifolds, empty ranges, dead points,
holes, partial workgroups, both halves and inactive endpoints; cold fields and
canaries must match exactly. Computed residuals allow tight 1e-6 reassociation
after an observed one-ULP backend difference. The existing independent corner
identity/once-only decay fixture exercises both schedulers. Browser fused ring480
plus reset and native fused ring480, gables240 and all four columns cases pass
without changing any scene acceptance bounds. Measurements and validation are in
`avbd-ring-persistence-ab-2026-10-14.json`.

Shared-source Metal library3/slice7/accuracy5/columns parity3 also pass after this
refactor; maximum columns position delta is 0.000213 m. Canonical per-point
ring480 and gables240 pass too. Eight new JS tests execute the shipped encoder
for canonical/compact/manifold/fused modes, with and without profiling. They
require tag clearing before fused persistence, correct current-count dispatch
slots, no redundant fused repack, final-iteration publication and all 128 primal
commands. The GPU range fixture uses two distinct collider pairs and endpoint
transforms so incorrect reuse of pair bodies cannot pass.

## Spatial upload with preserved scene identities

`order=morton` applies the pinned 10-bit Morton curve in reference Z-up axes,
keeping ground at GPU index zero for renderer compatibility. Original body IDs
remain unchanged in scene data; upload permutes current and previous body halves
and paired colliders, and regression readbacks undo the permutation. Six tests
compare the full 110k/506k permutations to the actual pinned routine, restore
every readback word, reject corrupt maps and run the shipped upload code to
verify renderer/collider/body alignment. All 24 timing/profiling/scheduling/spatial
JS tests pass. Shared Slang and native backend code did not change in this step.

Serial Morton/construction/Morton ring timings are 17.63/17.78/17.74 ms at full
active scale. Grid samples improve from 1.64 to 1.51 ms, but total-wall differences
are below 1% and comparable to run variation. Construction order remains default.
Browser Morton ring480/reset passes (mean flank drift 1.243084 m, ball forward
105.342758 m), as does gables240/reset (505,922 awake, exact 3,916,802 initial
contacts, unhit mean/max drift 0.117736/0.327027 m). Both have zero overflow.
Raw timings, workload counters and hashes are in
`avbd-ring-morton-ab-2026-10-14.json`. This does not close the 120 FPS objective.

## Clipping cost and an independent compact-gables issue (2026-10-14)

Passing clipping scratch by reference, with eight vertices, reduces the full active
browser ring from 17.73 ms to 14.91–15.12 ms per step in serial new/old/new runs.
Narrowphase drops from 4.52 ms to about 2.0 ms. All eight support points remain;
compiler IR and exact-touch/rotated-face tests protect that contract. The later
split-pass timestamp samples are diagnostic, separate from the throughput window.
See [batches and controls](avbd-ring-clipping-ab-2026-10-14.json).

Browser ring480 and gables240/reset pass. Native canonical gables finishes with
mean untouched-wall drift 0.118626 m, maximum 0.330369 m. Native compact gables
(`ENGINE_COMPACT_CONTACTS=1`) fails the existing 0.60 m maximum at exactly one
brick: body 164354, column 5, initial `[-71.92, 4.25, -102.5]`, final
`[-71.53714, 3.5324783, -100.593346]`, drift 0.717522 m. The original by-value
16-vertex clipping control reproduces the same trajectory and failure, as do
by-reference 8/16-vertex variants. This is an open compact-gables issue, not evidence
that all layouts pass. The test now reports the offending brick, column, initial
and current position, velocity, and number of bricks over the limit. No bounds
were relaxed. Ring performance remains below the 120 FPS requirement.

## Cached solver pairs (2026-10-14)

Compact primal/dual now read a 32-byte pair cache containing body IDs and the
sphere flag, avoiding collider indirection on each solver iteration. Both the
separate packing pass and fused persistence refresh it. Canonical manifolds keep
their collider IDs, so compound-shape identity and persistence remain intact.
The ring pays an additional 64 MiB allocation. Cached/control/cached timing is
14.78/14.97/14.78 ms at the unchanged full active scale: a small ~1.3% gain.
Phase samples reduce primal/dual from 5.24/2.62 to 5.05/2.49 ms, but these later
split-pass measurements are separate from the throughput batches.

Browser ring480/reset reproduces the prior checkpoint values exactly (final mean
flank drift 1.181787 m; ball forward 105.191246 m). Native compact ring480 and all
four columns checks pass, as do Metal accuracy/slice and columns parity. Added
tests exercise nonidentity/compound collider mappings, reversed endpoints,
sphere lever arms, dead points, mixed joints and contacts, all lane widths and
both contact halves. Compiler checks verify the cache ABI and removal of collider
bindings. Raw batch means and validation scope are recorded in
[the comparison](avbd-ring-cached-pairs-ab-2026-10-14.json). The pre-existing native
compact-gables drift failure remains open; no correctness bounds were relaxed.
