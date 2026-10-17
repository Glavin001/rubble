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

Gate per phase: the opt-in resident frame's outcomes within the GPU solve's run-to-run
spread and the oracle gates on the realtime, accuracy and pack sets; the default path
bit-identical (`scripts/destruction_peak.py guard`) until the resident frame is adopted.
