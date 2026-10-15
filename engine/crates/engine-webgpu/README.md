# Slang / WebGPU regression checks

Run Rust commands from `engine/`. Use a real GPU adapter: these tests deliberately
fail when no adapter is available instead of silently passing a skipped physics check.
On macOS, wgpu selects Metal. Set the environment required by the root `AGENTS.md`.

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo test -p engine-webgpu --release --lib --test columns_hold --test ring_hold --test gables_hold --test render_abi -- --nocapture --test-threads=1
cargo test -p engine-metal --release --lib -- --nocapture --test-threads=1
cargo test -p engine-metal --release --test oracle_accuracy --test avbd_slice --test columns_wgpu_parity --test webgpu_parity -- --nocapture --test-threads=1
cargo test -p engine-scenes --release --lib scene_conversion
```

The Metal packages require macOS. The WebGPU tests use the available wgpu backend.
Serial execution prevents scale tests from competing for GPU memory and bandwidth.

From the repository root, use Node 24 (or a Node version with TypeScript transform
support) for scene parity. This imports the pinned three-avbd builder directly and
compares every ring/gables body, including rotations, masses, inertia and velocities:

```sh
node --test engine/crates/engine-webgpu/tests/scene_builders.test.mjs
node --test engine/crates/engine-webgpu/tests/physics_benchmark.test.mjs engine/crates/engine-webgpu/tests/demo_profiling.test.mjs engine/crates/engine-webgpu/tests/persistence_schedule.test.mjs
node --test engine/crates/engine-webgpu/tests/spatial_order.test.mjs
```

The small GPU tests isolate CSR prefix publication, duplicate warm-start feature
IDs, dispatches beyond 65,535 workgroups, current-frame indirect counts, complete
rotated support polygons, exact-touch quaternion roundoff and friction penalty
updates. Coloring tests reject unfinished palettes and shared colors on active contact edges. A primal test compares 1/2/4/8-lane reductions with serial accumulation, including partial workgroups. Failure messages name the violated invariant and the affected slot/body.
The CSR tests compare every offset and scatter cursor, cover partial blocks and
empty adjacency, and protect the end sentinel. A direct Metal test catches a
32-lane or unpadded launch before bad offsets can reach a scene's scatter pass.
The contact-primal test independently assembles and solves the Newton system in
f64 on the CPU, covering both endpoints and sticking/sliding friction. It catches
shared GPU math errors that serial-versus-cooperative comparisons would miss.

The scale tests check every body, not just maximum height. Columns run for 180
steps, including a 100k-body case with sleep disabled. The 110k ring runs for 480
steps with sleep disabled, checks intact flanks before the impact spreads, then
checks aggregate collapse and the ball's passage through both walls. See
[`../../docs/avbd-regression-notes.md`](../../docs/avbd-regression-notes.md) for the
reference measurements used to choose those bounds.

Gables has a full-size 240-step impact gate: 505,922 active bodies, exactly
3,916,802 initial support contacts, valid contact coloring, no overflow, and
finite poses/velocities. Unhit columns must remain standing, and both balls must
stay within the recorded three-avbd trajectory bounds. The initial 20 steps also
check every brick before impact. The fixture uses
`add_bodies`/`add_colliders` to avoid exhausting staging resources with more than
a million tiny uploads before the first GPU submission.

## Browser verification

After changing Slang, regenerate the ignored WGSL assets:

```sh
cargo run -p engine-webgpu --release --bin wgsl_export
python3 -m http.server 8765 --bind 127.0.0.1 --directory crates/engine-webgpu
```

Open `http://127.0.0.1:8765/demo.html?scene=ring110k`. Reload after each export.
Normal playback runs continuously without benchmark pauses or resets. To measure,
add `&bench=1`; playback pauses after the impact, then restarts after measurement.
In that mode the console emits `BENCH` with wall time, active bodies, contact count and overflow.
Compare the same scene, iteration count, lead-in and batch protocol on the same
browser/GPU, with no competing GPU work. Sleeping-scene timings cannot establish
active-body performance parity. Never accept overflow or missing contacts as a
performance improvement.

For repeatable measurement without live playback before/after it, use
`?scene=ring110k&bench=only`. It performs the same 60-step lead-in, two rounds of
three untimed steps plus five ten-step batches, then stops. The `BENCH` report
includes every batch, both round medians, the pinned-compatible best median,
adapter/browser metadata and body/iteration counts. `allBatchesWithin120Hz` is
false if any batch exceeds 8.33 ms, even when the best median meets the target.
The start/end workload snapshots include every color-bucket size, read outside
the timed window. Their sum must match the GPU awake count; a mismatch aborts
the benchmark with a coverage diagnostic. Use this distribution when tuning
cooperative solve lanes instead of inferring bucket sizes from the color count.
It measures physics submission plus GPU completion, without rendering or CPU
readbacks inside the timed interval. Correctness must pass separately. Run both
implementations sequentially with other GPU work idle; repeat the control to
detect clock/load drift. The reference reports 110,442 total bodies; our HUD
reports 110,441 dynamic bodies, excluding the same static ground.

Compiler regressions inspect Naga IR to reject unrelated atomic counter reads,
atomic topology reads in the primal solve, duplicate topology bindings, and
full-body stores where only position/orientation change. The broadphase GPU
fixture exceeds the 512-collider grid threshold and checks unique candidate,
manifold and contact counts for a large ball placed first or last, and for two
neighboring cells sharing a hash bucket. A separate sleep-dispatch test covers
empty work, neighbor vetoes, sleep events and subsequent wakeful frames.
The production reuse shaders must also omit the unused shared diagnostic counter
buffer. Stable color-packing tests compare every bucket against an independent CPU
filter, with reversed awake lists, sleeping/static bodies, partial tiles, invalid
colors, history rows and capacity sentinels. A native Metal test requires complete
64-thread groups for the cooperative histogram pass.
Cooperative reduction tests exercise all four lane widths in canonical, compact
and persistent storage. They place 67 independently constrained bodies at each
bucket's tail, cover every scratch slot and partial workgroups, compare with a
serial solve, repeat three times and check static padding remains unchanged.
Failures identify the layout, lane width, bucket slot, body and pose component.
The moving-contact oracle independently computes the full `J·dq` residual in
f64 and solves the dense Newton system by Gaussian elimination. It covers both
endpoints, translated/rotated reference poses, quaternion hemisphere changes,
static/sleeping/awake neighbors and sphere reference lever arms in all three
storage layouts. It also checks dual forces against an independent cone
projection. Omitting angular motion was verified to trigger a diagnostic failure.
The oracle now also uses a fully oblique contact frame with unequal penalties and
asserts that all three translational cross-terms are nonzero. This covers matrix
indexing errors that an axis-plane normal cannot expose, for 288 moving primal
and 144 independent dual cases. Failures include the contact-basis case.
Deliberately omitting the yz matrix term was verified to fail in the oblique
case with a 1.92 mm pose error; the production term is restored.
Mixed joint/contact coverage also compares ordered, interleaved and reversed adjacency
against one ordered reference across both contact halves and all lane widths. It
includes empty manifolds and dead points; a stale cursor after a joint was verified
to fail with the edge order, body, component, and actual/reference pose.

The 110k ring now uses separate dispatches for colors 0–9 and a single-workgroup
solver for the remaining colors. The GPU's current color count bounds the tail;
each bucket is processed in eight-body chunks with storage synchronization between
colors. `sparse=32` retains the separate-color control. Other scenes keep that control
by default; native testing can opt in with `ENGINE_SPARSE_COLORS=10` (capacities above
524,288 bodies retain separate dispatches to respect the WGSL loop bound).
Paired full-ring browser runs measured **11.86 / 12.19 / 11.88 ms** for
sparse/control/sparse, approximately 2.5% less wall time. All 110,441 dynamic bodies
remained active and both workload snapshots matched exactly. This is about 84
physics steps/s, still below 120. See [timings and validation](../../docs/avbd-ring-sparse-colors-ab-2026-10-14.json).
Tests compare dependent colors, empty buckets, a 67-body bucket spanning nine
chunks, partial lanes, and four iterations across all three layouts. Mixed
contact/joint coverage also runs the sparse kernel. Shader validation protects
buffer synchronization; browser tests require every tail to run exactly once
before dual, using exactly one workgroup.

The browser has an automated full-scene check too:

`http://127.0.0.1:8765/demo.html?scene=ring110k&check=ring`

It checks 480 active steps in ten-step batches and validates scene reset.
Use `?scene=gables506k&check=gables` for the 240-step full-size gables check.
Add `&bench=1` to either check to measure wall time after validation. Rendering
intentionally pauses during a check and stops at its result.
Use the page's live-demo link to resume ordinary playback. The page must finish with `PASS`; the console emits `RING_CHECK` or `GABLES_CHECK` samples
and a final `REGRESSION` report. Add `&batch=1` to drain the queue after each step.
Both modes intentionally avoid CPU counter readbacks between checkpoints, so they
exercise GPU warm-start metadata rather than relying on cached host counts.

Coloring reuses valid colors by default and compacts local priority maxima before
repairing conflicts. To run the diagnostic control, append `&coloring=rebuild` to
a demo/check/benchmark URL; `&coloring=reuse` explicitly selects the default path.
Benchmark reports record this choice. Compare the same scene, iteration count and
active workload, and isolate competing GPU work before claiming a speedup. The
current diagnostic A/B/A measurements are documented in
`../../docs/avbd-regression-notes.md`; they do not establish three-avbd parity.

Color buckets use stable body-index order by default. `&packing=atomic` selects
the old append path for controlled comparisons; `&packing=stable` explicitly
selects the default. Reports include the packing choice. Compare total physics
time as well as graph/primal phases: stable packing adds two dispatches and a
tile-prefix buffer, so better locality alone does not establish a speedup.

Persistent 64-byte solver points plus 32-byte metadata are the default for
`ring110k`. Both contact history halves use that layout, eliminating point packing
and final dual copyback. `&contacts=compact` selects the previous scratch layout;
`&contacts=canonical` selects the original 128-byte contacts. Other scenes keep
the canonical default and opt in with `&contacts=persistent` or `&contacts=compact`.
Native WGSL tests select these with `ENGINE_PERSISTENT_CONTACTS=1` or
`ENGINE_COMPACT_CONTACTS=1`, respectively; native Metal remains canonical.
Reports include `contactLayout`. The older compact mode uses 64-byte solver scratch, then publishes
canonical warm-start/debug state on the final dual iteration. It adds allocation
and conversion work (128 MiB point scratch plus 64 MiB cached-pair scratch for ring). Post-reboot serial
compact/canonical/compact ring runs measured 17.93/19.98/17.90 ms per step,
about 10% less wall time. This remains above the 8.33 ms target.
After the clipping improvement, caching pair body IDs and sphere flags once per
step gives 14.78/14.97/14.78 ms in cached/control/cached runs. The gain is small
(about 1.3%); it does not establish 120 FPS or reference parity. Compact primal
and dual no longer bind collider records. Tests cover compound collider mappings,
reversed sphere endpoints, mixed joints/contacts, all lane widths and both contact
halves. The fused preparation test poisons the pair cache before each path and
checks every pair plus a tail canary, preventing stale data from hiding a missing
refresh. See [timings and validation](../../docs/avbd-ring-cached-pairs-ab-2026-10-14.json).
See [the compact layout and test contract](../../docs/avbd-compact-solver.md).
The persistent path measured 12.76/14.75/12.71 ms in persistent/compact/persistent
full-ring runs, with identical timed-window counters. A fresh default-mode retry
measured 12.69 ms (about 79 physics steps/sec), still above the 8.33 ms target.
It saves 256 MiB of ring contact allocation relative to compact scratch, while
retaining full feature and generation IDs. Its browser ring480 and gables240
checks pass. Native persistent ring/columns pass, but native persistent gables
fails the unchanged unhit-wall gate (five bricks over 0.60 m, maximum 3.00 m).
This backend discrepancy is unresolved; gables retains its canonical default.
The subsequent linked-grid change fetches each next link before testing geometry
and carries an awake-dynamic flag in the link word, allowing early rejection of
pairs owned by the lower-index body. Sleeping neighbors remain discoverable at
either index. Clean optimized/original/optimized ring runs measure
12.26/12.74/12.32 ms (about 81 physics steps/sec), with identical workload counters
and no extra allocation. A 641-body CPU all-pairs oracle checks all 6,084 expected
touching pairs; GPU tests also protect link membership, signed cell hashes,
partial workgroups, sleeping endpoints and large-body ownership. The counting-sort
grid experiment was slower and removed. See
[grid timings and validation](../../docs/avbd-ring-grid-filter-ab-2026-10-14.json).
Run the complete correctness commands above with the environment variable set,
then compare fresh canonical/compact/canonical browser benchmark runs with no
other GPU work. Both modes must retain the full active scene and unchanged bounds.
Benchmark reports preserve counters before and after the timed window and reject
wrong body counts, iteration counts, timesteps, sleeping impact scenes, empty
collision work, or overflow. These guards supplement full-scene accuracy tests.

Persistence scheduling experiments remain opt-in: `&persistence=manifold` loads
pair bodies once, while `&persistence=fused` also prepares compact solver points
during warm start (requires compact contacts). Native WGSL equivalents are
`ENGINE_MANIFOLD_PERSISTENCE=1` and `ENGINE_MANIFOLD_PERSISTENCE=fused`, with
`ENGINE_COMPACT_CONTACTS=1` for the latter. The default remains per-contact.
Neither demonstrated a repeatable full-ring wall-time gain; see
[the measured comparisons](../../docs/avbd-ring-persistence-ab-2026-10-14.json).
The fused path still clears tags before persistence to protect dead reserved
slots, then skips the separate solver-point packing pass. Tests compare every
canonical word and live compact field across 67 ragged manifolds, both buffer
halves, inactive endpoints and tail canaries. Only computed floating values
allow 1e-6 relative/absolute roundoff; metadata and inactive records remain exact.

`&order=morton` uploads the demo in the pinned reference's spatial order, with
ground kept at GPU index zero for the renderer. Construction order remains the
default: ring Morton/construction/Morton timings of 17.63/17.78/17.74 ms do not
establish a material throughput improvement. Source bodies retain their original
identities; both GPU body halves and colliders use the permutation, and regression
readbacks undo it before checking walls and impactors. Full ring480 and gables240
plus reset pass. Six new tests compare full-scale ordering against the actual
pinned Morton function, restore every body word, and execute the shipped upload
to check collider/body/render pairing. See
[ordering measurements](../../docs/avbd-ring-morton-ab-2026-10-14.json).

The timestamp profiler also emits `phaseSamples`: ten raw samples pairing phase
durations, first-to-last GPU timestamp span, CPU encoding time and submission-to-
readback time from the same step. Each sample records its step and pass count.
These split-pass, individually drained samples follow the timed batch window and
can perturb execution; their phase medians must not be added to infer batch wall
time. The orchestration tests reject missing phase boundaries, reversed timestamps
and readback failures, and verify that earlier queued work is excluded from the
CPU interval and temporary GPU resources are released on failure.

`&profile=detail` separates reference-pose updates, hash clearing, hash insertion,
and collision generation in the later timestamp samples. Ordinary timing batches
keep their existing pass boundaries. A zero phase may be timestamp quantization,
not absent work.

`&profile=graph` instead separates awake-list construction, adjacency degrees,
prefix sums, scatter, physical sorting, color initialization, color repair,
color packing and prediction. It keeps the same physics throughput window;
the extra timestamp boundaries apply only to the later diagnostic steps.
Tests execute the shipped encoder to verify the kernels charged to each stage,
solver timestamp offsets, resource cleanup and single-pass throughput scheduling.

Stable color packing now computes tile prefixes cooperatively, with one complete
64-thread workgroup per color. Each lane owns a disjoint tile range, exchanges its
range total through a workgroup scan, then writes exclusive prefixes in place.
Body-index order is unchanged. `&prefix=serial` retains the original browser
control; native WebGPU uses the cooperative version and Metal keeps its existing
serial host path. Full-ring candidate/control/repeat timings were
11.50/11.89/11.61 ms, with identical workload and color-bucket snapshots.
The independent CPU scan regression covers ring/gables tile counts, ragged lane
ranges, empty colors, zero-body reset, extra workgroups and untouched buffer tails.
Both prefix kernels also run through the stable packing/order regression.
See [the paired measurements](../../docs/avbd-ring-parallel-color-prefix-2026-10-14.json).

A separate reuse-pass experiment did not materially improve ring throughput and
was removed. Its diagnostic readback found 44% of candidates reusable at step 166;
see [the collision comparison](../../docs/avbd-ring-clipping-ab-2026-10-14.json).

The clipping kernel uses eight-vertex scratch passed by reference, matching the
reference implementation's maximum support polygon. A compiler IR test rejects
by-value array arguments or an incorrect scratch capacity; geometric regressions
require all eight distinct support points, including exact-touch cases.

## Isolating browser/driver overhead

`dispatch-diagnostic.html` runs a tiny increment shader, with no physics or Slang.
It compares empty submissions, 1/32/128 dispatches, separate compute passes, and
a 32 MiB streaming buffer. Every output element is checked against the exact
submitted dispatch count; missing/duplicated work invalidates the result. All
workloads finish and the device is destroyed before the report is displayed.
The page records visibility/focus and every timing batch. These numbers are not
physics FPS and do not establish the scene's 120 Hz requirement.

`cargo run -p engine-webgpu --release --bin gpu_dispatch_probe` runs the same
WGSL and timing protocol through native wgpu, with the same full-buffer checks.
Use `GPU_PROBE_WORKGROUP=256` (native) or `?group=256` (browser) to compare a
256-thread group against the default 64. Run native and browser probes serially.
The probe uses its own device/buffers and does not modify the physics backend.
