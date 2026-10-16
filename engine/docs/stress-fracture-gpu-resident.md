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
