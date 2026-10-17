# Experimental compact AVBD solver contacts

The ring emits roughly 1.28 million contact points. The canonical 128-byte Contact
repeats collider IDs, normal, friction and tangent vectors already available from
its manifold. `compact_solve.slang` exports separate solver entries that use a
64-byte hot record while retaining the canonical persistence/debug ABI.

| Byte | Field |
| --- | --- |
| 0 | anchor A (3 floats), normal reference residual |
| 16 | anchor B (3 floats), first tangent reference residual |
| 32 | multipliers (3 floats), second tangent reference residual |
| 48 | three independent penalties, pair index plus stick bit |

The pair index occupies 31 bits; `0x7fffffff` means dead. The top bit holds static
friction state. Full feature IDs and full generation bits remain untouched in the
canonical Contact: neither is truncated or repurposed. Normal/friction come from
the manifold, and tangents use the same deterministic basis as contact creation.
This format is internal to generated contacts, not an alternate public API for
arbitrary independently supplied tangent frames.

After warm start and before the first solve, a contact-count dispatch clears every
compact tag to dead; a manifold-count dispatch packs the live ranges. The clear
protects reserved holes and dead manifolds, including reuse of a previous larger
frame. Manifold contact starts include the canonical ping-pong offset; compact
indices subtract that offset. Current bodies and manifold metadata are shared.
Each dual iteration updates compact forces, penalties and stick state. Only the
final iteration (pass uniform 1) publishes the eight changed canonical words,
including final normal residual. The remaining canonical fields retain their
original bits. Zero iterations leave canonical warm-start output unchanged.

The full `ring110k` browser scene now defaults to the persistent storage described
below. The older scratch path is selected with `contacts=compact`; `contacts=canonical` is the explicit
comparison control. Native WGSL uses `ENGINE_COMPACT_CONTACTS=1`. Native WebGPU also provides
`WgpuSim::new_with_compact_contacts(capacity, true)` for isolated tests. Existing entries and other scenes keep canonical contacts. Controls must use identical
scene, color packing, active workload, iteration count and benchmark windows.

This first experiment reduces repeated solver traffic, **not total allocation**:
it adds 64 bytes per contact capacity and two dispatches per step. It deliberately
keeps the old layout for a direct numerical/performance comparison. Retain it as a
production default only with isolated total-wall-time evidence. If worthwhile,
the next migration is compact persistent points plus pair metadata, eliminating
the duplicate allocation and conversion passes. That later migration must retain
full feature identity, generations, nearest-anchor disambiguation and external
readback compatibility.

Regression coverage:

- Naga checks 64-byte stride, all eight field offsets, absence of canonical contact
  binding in compact primal entries, and the ten-storage-buffer browser limit.
- Both primal entries retain the plain topology/pose-only-store checks.
- An independent CPU f64 Newton solve checks both endpoints and sliding/sticking
  cases with the reconstructed tangent frame.
- Four dual iterations compare every canonical word with the existing solver in
  both ping-pong halves. Intermediate iterations must leave canonical bytes
  unchanged. Final publication must preserve all feature/generation/geometry bits
  and both dead holes and dead points within live manifolds.
- Full columns180, ring480, gables240, Metal accuracy/slice/parity, browser reset
  and controlled timing remain the acceptance gates. A green small test does not
  establish full-scene correctness or the 120 FPS goal.

## Initial validation, 2026-10-14

All 24 focused WebGPU library checks pass. Compact-mode columns180 passes all
four cases, including 100k without sleep. Gables240 passes with 505,922 awake
bodies, initial contact count 3,916,802, unhit-wall mean/max drift 0.118613/0.330658 m.
Ring480 passes with 110,441 awake bodies, initial contacts 1,284,737, final mean
flank drift 1.2008 m and ball forward position 105.66325 m. All overflow flags
remain zero. Metal validation also passes: three library checks, seven physics slice tests,
three columns parity tests (maximum position delta 0.000779 m), and five accuracy
oracles. Browser validation and fair performance remain separate gates.

The browser compact ring also passes all 480 steps and scene reset, with final
mean flank drift 1.170003 m, ball forward 105.235405 m, all 110,441 awake and
zero overflow. It starts with 1,284,737 contacts, as the canonical ring does.

The additional compact cooperative-primal test passes the same independent dense
oracle with a partial workgroup. The profiler orchestration test reproduces and
prevents a bug where compact dispatches were accidentally inserted into timestamp
reduction; both canonical and compact profiling now encode one step and release
all timestamp resources. All ten JS scene/timing/profiling tests pass.

The repaired-profiler compact/canonical/compact comparison measured
232.81/233.59/247.20 ms (second-round medians 241.56/253.57/253.48 ms). Raw batches,
phase profiles and source hashes are saved in
`avbd-ring-compact-contact-ab-2026-10-14.json`. Conditions were not isolated and
there is no repeatable total-wall improvement. These earlier measurements did not justify default adoption. It provides a tested migration boundary for future
compact persistent storage; it does not meet the 8.333 ms target.

## Post-reboot validation and ring adoption

After the whole Mac restarted, serial compact/canonical/compact full-ring runs
measured 17.93/19.98/17.90 ms with physical adjacency and identical active scale.
Both compact runs reduced wall time by about 10%; primal/dual medians fell from
6.49/3.60 ms to about 5.4/2.69 ms. Full browser ring480 plus reset passes; native
compact ring480, gables240, all four columns tests and 26 focused library tests
pass. Shared-source Metal library, slice, accuracy and columns parity checks pass
(maximum columns position delta 0.000779 m). Publication coverage now also checks
rotating spheres, two moving bodies, two sleepers, and static/sleeping pairs in
both buffer halves; inactive endpoints must retain every canonical word.

The ring demo adopts compact points; other scenes remain opt-in until measured.
The scratch cost is 128 MiB at ring capacity, and persistent contacts are still
128 bytes. Full persistent compaction remains a future opportunity. This does
not meet 120 FPS or reference performance parity. Raw batch means and source
hashes: `avbd-ring-compact-post-reboot-2026-10-14.json`.

## Persistent hot/cold storage

The ring default now retains the 64-byte solver point across both history halves.
Binding 4 contains a 32-byte cold record: collider IDs, full 32-bit feature and
generation IDs, normal and friction. Binding 27 contains the two hot halves;
binding 28 remains the current-step 32-byte solver pair cache. At ring capacity,
contact storage falls from 512 MiB canonical history plus 128 MiB point scratch
to 128 MiB cold history plus 256 MiB hot history. The 64 MiB pair cache is unchanged.
This saves 256 MiB but is still larger than a fully deduplicated pair/point layout.

The persistent tag uses low 30 bits for manifold index plus one (zero is dead),
bit 30 for transient reuse, and bit 31 for sticking. During narrowphase reuse,
the cold feature word temporarily holds the previous index plus two. Persistence
consumes that marker before publishing the full feature ID. Dead reserved points
must remain dead, and the compact scratch-clear kernel must never erase live
persistent tags. Preparing solver pairs still runs, but point packing and final
dual copyback do not. Canonical entries and their debug ABI are unchanged.

Tests cover both halves, holes, regenerated/reused/separated pairs, generation
changes, high-bit features, duplicate-feature nearest-anchor matching, and
exactly-once force/penalty decay. A dual fixture compares every hot float after
each of four iterations with the old compact path, checks untouched cold bytes
and previous history, and includes compound colliders, reversed sphere endpoints,
sleeping/static bodies and sliding/sticking contacts. Serial and cooperative
primal tests use an independent f64 Newton oracle. Generated Naga checks protect
strides, full-width IDs, binding limits and absence of dual copyback; JavaScript
tests protect routing, double-history allocation and scheduling.

Serial persistent/compact/persistent browser ring timings are 12.76/14.75/12.71 ms,
with identical workload counters. The default-mode retry is 12.69 ms. Browser
ring480/reset and native persistent ring480/columns pass. Browser persistent
gables240 passes with unhit mean/max drift 0.117252/0.327617 m and 505,922 awake
bodies. **Native persistent gables fails**: five unhit bricks exceed 0.60 m,
maximum 3.001130 m at body 318697, column 10. This is worse than the older compact
failure (one brick, 0.717522 m); its cause is unresolved. Native canonical gables
still passes, mean/max 0.118626/0.330369 m. Keep the gables default canonical and
retain all bounds. Metal library, accuracy, slice and columns parity gates pass
after the shared storage refactor (18 tests, max column position delta 0.000107 m).

The ring remains about 79 physics steps/sec, below 120. These results do not
establish reference performance parity or solve the native gables discrepancy.
