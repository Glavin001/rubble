# Stress fracture: GPU-resident coupling

rubble's AVBD engine and stress-gpu stay two modules on one device. This plan removes
the host from the frame: no blocking readback inside a frame, one submission per frame,
and island routing, freezing, contact planning and splits done in kernels. Approved
2026-10-16 (items 8 and the "one submission" part of the recommendations).

## Where a frame leaves the GPU today

Measured on house-1story hit by a car, 250 frames (`stress_bench --only house_car`,
`sample` on the main thread):

| share | what |
|---|---|
| ~35 % | waiting for the GPU's own stress substeps (real work) |
| 18 % | `GpuSolver::build`: re-packing every record from the f64 mirror (`stress_layout::pack`), new buffers, load terms, a `Scene` clone |
| 11 % | per-segment `snapshot`: a full download plus a clone of the mirror |
| ~10 % | contact planning on the mirror (`contacts::plan`, `engaged_contact_dt`) |
| ~14 % | readbacks inside `step` (per batch, bond history on halts, `sync_motion`) |
| ~4 % | rubble calls: `download_bodies` per step, per substep contacts, `motion`, `set_motion`, `sync_bodies` with every chunk's shape |

rubble's AVBD step itself costs 2.4-4.7 ms per step with a handful of bodies (GPU
execution of ~100 latency-bound dispatches), which is what makes the floor drop and the
sub-stepped free-wall ram slower than PhysX GPU.

## Status (2026-10-16)

- **Stage 1 and Tier 0 done, bit-identical.**
  - Host waste is out of every frame.
  - Quiet clusters are left out of the GPU tables, and their routing and step data are reused.
  - rubble's body read-backs are reused.
  - Hulls are cooked once per distinct shape.
  - Routing and the contact plan no longer loop over every pair.
  - Setup work is parallel and proportional to the scene.
- **Measured.**
  - Idle 64-house town: 210 → 2.4 ms per steady-state frame.
  - 256 houses with four rammed: 815 → 514 ms per frame.
  - 256-house setup: 55+ → 15 s.
- **Stages 2–3 not started.**
  - The measured value of stage 2 (one submission, shared buffers) is now small: host waits are under 5 % of a single-house destruction frame.
  - Stage 3 (planning, packing and splits on the GPU) is what the many-house destruction frames need.

## Stages

Each stage is gated before the next starts.

**Stage 1: host waste out, results bit-identical.** Gate: every scene of the accuracy and
town sets gives the same end-state digest (`stress_bench` `digest`: cluster motion and
membership, chunk hidden states, impactors, FNV over their exact values) as before, for the
scenes whose digest is stable across repeats; the others stay within their spread.
- no `Scene` clone per `build`; the mirror's structures shared (`Arc`, copy on write), so
  snapshots stop deep-copying topology (done);
- static records (`ChunkStatic` geometry, bond statics, hulls, CSR) packed once per
  topology version and kept on the GPU; `build` rewrites only the plan- and load-dependent
  parts, into existing buffers when they fit;
- rubble: body states kept host-side from the last readback instead of a
  `download_bodies` per step and per removed body; contacts of all substeps of a frame
  collected on the GPU and read once; the step's counters read once per frame;
- `sync_bodies` keyed by a per-cluster version bumped on splits, instead of rebuilding
  every chunk's shape every frame.

**Stage 2: one submission per frame, shared buffers.** Gate: digests equal to stage 1
where stage 1's were stable.
- stress-gpu reads rubble's body buffer (pose, velocity) and contact rows directly through
  a coupling bind group: driven clusters follow rubble bodies, rubble contact impulses
  become chunk loads (collider → (cluster, chunk) table written when bodies are made);
- island bodies frozen and released by kernels writing rubble's body buffer;
- the batch halts (`travel_check`, splits) become GPU flags that make later dispatches of
  the frame no-ops for halted islands; the host records the whole frame (rubble substeps
  with indirect sizes, stress batches up to the frame's substep budget, the hand-back) and
  reads one small status record at the end; a halt that needs host work (a split until
  stage 3) is serviced then and the frame continues from the halt, which is what the
  current loop does after its readback.

**Stage 3: routing, planning and splits in kernels; the mirror becomes optional.**
Gate: the accuracy set's outcomes within the run-to-run spread of the GPU stress solve
(2026-10-16 benchmark), the integrated tests, `free_wall_ram_needs_no_rewind`'s exact
momentum, b5 push-over / punch-through, and b8 within the oracle's spread. Host decisions
move from f64 to compensated f32, so this stage cannot be bit-identical; routing is kept
conservative (swept bounds inflated by their f32 error bound) so no impact the f64 test
would route is missed.
- routing: per-cluster swept bounds, `rigid_assumption_fails` per candidate pair from
  per-cluster constants (wave round trip, weakest bond capacity, contact modulus) written
  at split time, transitive closure by label propagation;
- contact planning: chunk swept boxes through rubble's broadphase grid, the certified
  contact step (`engaged_contact_dt`) as a reduction;
- splits: connected components over unbroken bonds by label propagation (as stress-gpu
  #17/#30), new clusters' mass properties and frames by reductions, rubble bodies from a
  free-slot list with colliders re-assigned (chunk hulls are already in rubble's hull
  arena, so nothing is re-cooked);
- the mirror is updated asynchronously (non-blocking map) for observations and tests only.

## Stages 2 and 3: the resident frame (plan of 2026-10-17)

Your direction: the whole coupled frame stays on the GPU; the host configures it,
submits it and reads one small status record. Built in phases, each landing working and
measured before the next.

### Where a coupled frame crosses to the host today (measured inventory)

- **rubble** (`RubbleEngine::step`): per engine substep a blocking counter read (80 B),
  a blocking contact read (`last_contacts`, rows × 128 B) and a blocking body read
  (bodies × 112 B); `motion`, `set_motion` and `sync_bodies` read bodies again; the
  collider → (structure, chunk) table exists only on the host (`owners`).
- **stress-gpu** (`GpuWorld::step_frame_coupled`): a full `download` at frame start and at
  the write-back (state, islands, impactors, loads, ledgers, the whole `bond_dyn`); two
  table rebuilds (`refresh` → `build`, re-packing from the f64 mirror); one blocking
  read-back per substep batch; host splits with a download and a full rebuild each;
  contact plans on the host; the quasi-static tail's sync and host commit.
- **Host decisions:** routing (`select_islands`), the load filter (`LoadFilter`), contact
  planning (`contacts::plan`, `engaged_contact_dt`), splits (`split_cluster`), the quiet
  set and the frame verification (`fractured_since`).

### Phases

1. **Shared buffers and the collider table.** `WgpuSim` exposes its body, collider and
   contact buffers and the last step's contact range; the coupling uploads a
   collider → GPU chunk table (and body → island) whenever either side's layout changes.
2. **Engine contacts to stress loads on the GPU.** After each rubble substep, an ingest
   kernel reads that substep's contact rows (GPU → GPU, no host) and accumulates each
   chunk's impulse, point and closing speed in a fixed order (rows grouped by chunk:
   deterministic sums, no float atomics); a second kernel applies the load filter's
   sustained / pulse split (`LoadFilter`) into the stress solver's load slots. Replaces
   `last_contacts`, the host filter and the per-frame rebuild for loads.
3. **rubble without readbacks:** `step_submit` (counts forwarded on the GPU) for every
   engine substep; CCD substep counts from a GPU bound or a conservative host bound.
4. **Poses and write-back on the GPU:** driven clusters' rigid frames written into the
   island records from rubble's body buffer; island bodies written back into it
   (freeze/release by kernel); no `download`/`refresh` per frame.
5. **One submission per frame:** rubble's substeps, the ingest, the stress substeps (the
   batch's halt flags already make later dispatches no-ops) and the quasi-static solves
   and commits (`equilibrate_resident`, `commit_resident`) encoded together; one small
   status read (halts, splits, unconverged counts) at the end.
6. **Stage 3, splits on the GPU:** new clusters from the device connectivity labels
   (`island_connected_*`), mass properties by reductions, new rubble bodies from a
   free-slot list with colliders re-pointed; then routing (`select_islands` per cluster
   from GPU bounds) and contact planning in kernels; the f64 mirror synced only on demand
   (observations, tests).

### Status

- **Phase 1 (done):** `WgpuSim::{body,collider,contact,counters}_buffer`,
  `last_contact_offset`, `capacity`. The collider table is `stress_gpu::ingest::ColliderInfo`:
  (GPU chunk, body code, kind, chunk side taken) plus the other side's mass, modulus and
  radius.
- **Phase 2a (done): the contact ingest.** `shaders/coupling.slang`,
  `stress_gpu::ingest::EngineIngest`.
  - `ingest_rows` runs once per engine substep, one thread per row. It writes two records
    per row at fixed slots (substep, row, side).
  - A counting sort groups the frame's records by chunk (`ingest_count`, `ingest_scan`,
    `ingest_scatter`). Atomics are used on integer counts only.
  - `ingest_merge` puts each chunk's records back into slot order, then merges them per
    other body in order of first appearance, the rule of stress-ref `manifolds`. The
    order and the sums are deterministic.
  - Checked against the host by `stress-rubble/tests/ingest_gpu.rs`
    (`RubbleEngine::attach_ingest_check`). The test runs each step through the GPU ingest
    on rubble's own buffers and compares the result with the host's manifolds from the
    same rows. On the block-pile scene, 72 manifolds match with relative impulse error
    ≤ 7e-8, point error ≤ 6e-8 m and closing-speed error ≤ 3e-7 m/s. That is f32
    accumulation; the host's chain is f32 then f64.
  - Off by default; the coupled frame does not use it yet.
  - Record memory is 2 × rows × substeps × 64 B (8 MB per substep at the default 65,536
    contacts). Phase 2b merges after every substep into running manifolds, so the memory
    stays O(rows).
- **The resident quasi-static frame (done, opt-in): phases 2b, 3 and 4 for the
  quasi-static coupling.** `GpuWorld::with_resident_engine` (stress_bench:
  `STRESS_RESIDENT=1`) takes it under the fast frame's conditions.
  - **Engine.** rubble runs its substeps with `step_submit`. Counts are forwarded on the
    GPU and nothing is read back. Each substep's rows go into the ingest
    (`stress_gpu::resident::ResidentEngine`, implemented by `RubbleEngine` and shared with
    the world's shell through `SharedEngine`).
  - **Loads.** Three kernels in `shaders/resident.slang`:
    - `resident_poses` writes driven clusters' poses into the island records from the
      engine's frame-start bodies.
    - `resident_filter` is `LoadFilter` with `frame_average_terms`. Sustained loads are
      summed in the host's key order. Impact pulses are averaged over the frame; those
      that outlast it are carried in `STRESS_RESIDENT_PULSES` slots per chunk (default 4).
      Overflow fails the frame rather than dropping a load.
    - `resident_static_loads` is `static_loads` per solved island.
  - **Solve and commit.** `equilibrate_resident_with` (loads formed by the kernels above
    in the same submission), then `commit_resident`.
  - **The cascade under GPU control (phase 5).** `GpuSolver::resident_cascade` with
    `shaders/cascade.slang` encodes `fast_static_tail`'s passes, each one as
    `equilibrate_resident` and `commit_resident` run it:
    - the layout's island records are copied in on the GPU;
    - the unconverged solves are restored by a kernel;
    - a pass's verdict (damage grew, nothing disconnected) decides on the GPU whether
      the next pass runs.

    Passes go in windows of 4 per submission. One command buffer holding all 200 passes
    of `max_cascade` failed on Metal, and empty passes still cost dispatches.
  - **One read per frame.** The frame reads the cascade's status record, rubble's step
    counters (`WgpuSim::drained`, which replaces `drain`'s own read) and the pulse slots'
    status, together in one `read_ranges`. A cascade longer than its window, or a split,
    adds a read.
  - **Splits.** A split syncs the mirror, including the engine's motion
    (`GpuWorld::sync`), splits on the host, rebuilds the tables (the frame's filtered loads
    and pulses carried by chunk) and resumes the cascade at the next pass.
  - **Same results as the fast frame, bit for bit.** Measured on house_ram (qs and blast
    configurations, 25 bonds broken), town16_one_ram (qs, 8 broken), the fractured
    high-rise rammed (qs) and Vibe Town idle. The f32 load path reproduces the host's
    f64 → f32 loads exactly on these scenes. Covered by `tests/resident_gpu.rs`: the
    cracking house, and a weak cantilever that splits at its root, whose fragments fall
    onto the ground. Damage, clusters and fragment positions match bit for bit.
  - **Vibe Town idle** (281 clusters, authored, qs, budget 2 × 32, interleaved runs on a
    shared GPU): fast frame median 35–51 ms; **resident 16–21 ms with host control of the
    cascade, 16–17.5 ms with the GPU cascade**. Engine preparation falls from 2.6 to
    0.6 ms, and the download (2.3–2.8 ms) and host loads are gone. What remains (~16 ms)
    is the static solve on the GPU, the next item.
  - **The repeat gate (lossless).** `cascade.slang` `cascade_gate` replaces quiet
    exclusion for resident frames. An unconverged solve leaves its island as it found
    it: the state is restored and nothing is committed. A later solve of that island with
    the same loads, bit for bit, would return the same verdict, so it is skipped.
    `STRESS_QS_REPEAT_GATE=0` turns it off. `tests/resident_gpu.rs`
    `repeat_gate_is_lossless` checks it.
    - Vibe Town idle with the gate (budget 2 × 32): **median 4.2–4.3 ms**, against 16 ms
      without it and 35 ms for the fast frame. The digest is unchanged.
    - Vibe Town rammed: median 7.6 ms, p95 9.9, max 41 (19 / 23 / 50 without the gate;
      fast 39 / 46 / 65). The gated and ungated digests are the same.
    - The fast frame's digest differs from the resident one there, and only in the
      mirror. Every chunk state and cluster pose is equal; one impactor velocity component
      is 0.0 against −0.0. The fast frame's mirror takes the engine's motion through
      `interpolate(start, end, 1)`, while the resident sync copies the end.
  - **Not covered yet:**
    - Impact islands (tiered and explicit frames) still take the host path. They need
      `resident_filter`'s pulses as load terms (`load_range.zw`) and phase 6.
    - Splits are on the host.
    - Every dispatch writes small uniforms. Those are host → GPU writes, not read-backs.

### Island frames: what is left on the host (measured 2026-10-17)

Fractured high-rise 10F rammed, tiered, authored, island frames (M3 Max). A frame is
~100 ms:
- **GPU substeps: ~80 ms** (1,188 substeps).
- **Host: ~20 ms.**
  - Segment set-up: 9.5 ms. Four segments per frame, each with a snapshot of 1.2–1.7 ms
    (a full download plus a mirror clone), a full table `build` of 1.0–1.3 ms for the
    segment's contact plan, and 0.4 ms of preview.
  - Read-backs per batch: 1.5–4 ms.
  - Encode: 1.6 ms.
  - Download at the write-back: 2 ms.
  - Write-back: 1.5 ms.
  - Refresh: 0.8 ms.

The download exists because the host contact plan (`contacts::plan`) reads chunk
positions and velocities. Residency here therefore needs the plan on the GPU. Phases:

- **6a. Contact plan on the GPU.** Per-chunk motion and budgets; cluster-pair culling;
  chunk-pair sweeps with `may_overlap`; impactor and ground candidates. Compacted in the
  host plan's order. Checked against `contacts::plan` (the same candidate sets; f32
  rounding may add borderline candidates, which is conservative).
  **Done (2026-10-17):** `shaders/plan.slang`, `stress_gpu::gpu_plan::GpuPlanner`,
  checked by `STRESS_GPU_PLAN_CHECK=1` beside every host plan
  (`tests/gpu_plan.rs`). Measured on house_ram, wall_ram, b5_v20 and house_car:
  - 90k host pairs, none missed, in the host's order;
  - 2 borderline extra pairs on house_car;
  - ground and impactor candidates and rotation checks identical;
  - budgets within 1.6e-6.
- **6b. Plan to tables on the GPU.** The contact index, chunk contact ranges, budgets and
  planned poses. Pair memory moves into a GPU hash keyed by chunk pair. No `build` per
  segment.
- **6c. Segment snapshots as GPU copies.** The step bound (`plan_contact_dt`) becomes a
  GPU reduction read with the batch status.
- **6d. Routing on the GPU.** `select_islands` from engine contacts and bounds. Island
  bodies are frozen and released in rubble's body buffer by kernel. The write-back is
  done by kernel.
- **6e. Splits on the GPU.** From the connectivity labels; new rubble bodies from a
  free-slot list. The f64 mirror is synced only on demand.

Gate per phase: the opt-in resident frame's outcomes within the GPU solve's run-to-run
spread and the oracle gates on the realtime, accuracy and pack sets; the default path
bit-identical (`scripts/destruction_peak.py guard`) until the resident frame is adopted.
