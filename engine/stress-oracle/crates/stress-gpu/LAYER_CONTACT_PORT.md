# Porting the proven layer contact to the GPU

Design, 2026-10-12. This is REALTIME_ROADMAP Phase 6, the proven contact. It was
prompted by three energy leaks found in 10 s runs:

- **The GPU statics certificate.** Fixed in `08ae1a2ee`.
- **The origin-dependent rigid drift fold.** Fixed in `a3546af9d`.
- **The established contact's handover to PhysX** (failing test `4a9047c8f`). Its
  10 % corner pull-in, hand-picked stiffness shares and born-deep threshold are the
  heuristics IMPROVEMENTS.md's `layer_contact` entry already replaced in the oracle.
  The user chose to port the oracle's proven contact, not to patch the heuristic.
  The PhysX side transfers the layer's physical state, `Stick.indent` (`ad3dedc6f`).

Paths:

- `ref/` is `crates/stress-ref/src/`
- `gpu/` is `crates/stress-gpu/src/`
- `sl/` is `crates/stress-gpu/shaders/`

## 1. How the GPU handles the proven switches

| Switch | Oracle | GPU today |
|---|---|---|
| `layer_contact` | `ref/world.rs` `apply_layer_contacts`, `hand_over_split_joints`, `layer_force`, `pair_layer_contact`, `pair_elastic_with`; `ref/polytope.rs` | **Rejected**: `gpu/world.rs` `unsupported()` |
| `scaled_step_bound` | `ref/solver.rs` (Gershgorin, mass-normalised) | Delegated to the mirror's `chunk_stable_dts` and `substep_dt` |
| `exact_frame_transfer` | `ref/solver.rs` `split_cluster` | Delegated: every split runs on the host mirror |
| `contact_crushing` | `ref/world.rs`, only inside `pair_layer_contact` | Inert without layer contact; `methods.rs` rejects it with layer contact until it is ported |
| `tangent_statics` | `ref/statics.rs`: secant for `SECANT_STEPS`, then a globalised tangent re-solve | Delegated (phase 0, done): the GPU secant runs at most `SECANT_STEPS` (`StaticOptions::secant_steps`); a cluster it leaves unconverged is solved by the mirror's `equilibrate_or_keep` and uploaded (`gpu/solver.rs` `equilibrate`, `tests/tangent_statics.rs`). Was a silent hazard: an unconverged GPU solve kept its state where the oracle re-solves |
| `lattice_contact_step` | `ref/world.rs` (lattice bound, with layer contact only) | Inert without layer contact (the oracle's `contact_stable_dt` needs both); `methods.rs` rejects it with layer contact. Once ported, coupled island frames need it: `plan_contact_dt` calls the penalty-network `engaged_contact_dt` |
| `implicit_contact_steps` | `ref/world.rs` (implicit mode) | Rejected (implicit mode) |

`frame_contact_step` (not proven) acts in the oracle only with `layer_contact` (`ref/world.rs`
`step_frame`), so it is inert while layer contact is rejected; GpuWorld never calls
`World::step_frame`, so it is classified Inert-without `layer_contact` and rejected with it.

**Done:** `gpu/methods.rs` classifies every name in `Methods::NAMES` as Implemented,
Delegated, Inert-without or Rejected; a test iterates over the names so that no switch can
go unclassified, and `unsupported()` rejects exactly the scenes `methods::unhonoured` lists.

## 2. The layer contact

### Geometry (`ref/polytope.rs`)

- **Polytopes.** Each chunk is a hull or a cuboid, stored as a boundary representation
  (faces with their vertices).
- **Clipping (`clip_into`).** Each face keeps its inside part. Crossings are
  interpolated from the inside end, so two faces that share an edge get the same
  crossing bit for bit.
- **The cap (`cap_polygon`).** Points are deduplicated by their bits, sorted by integer
  `total_cmp` keys, and joined by Andrew's monotone chain.
- **Overlap.** `intersect` is `F_b` sequential clips.
- **Field cells (`field_cells_in`).** Optionally recede each body by its indent. The
  candidate faces are those whose minimum depth is at most the smallest maximum depth.
  Bisector clips between candidates form the cells. Each cell gives its mass
  properties, interface integrals and peak depth.

### Elastic and contact law

**Elastic law (`pair_elastic_with`).**

- `k''_n` and `k''_t` come from `layer_moduli` and each body's thickness along the
  cell normal.
- Force: `(n V + interface)·½k_n`.
- Moment: includes the turning term.
- Stored energy: `½k_n ∫s dV`.
- Crushing peaks: `(max_depth + indent, k_n)`.

**Contact law (`layer_force`).**

- **Normal.** `f_el − c_n v_n`, clamped at zero, plus a rocking dashpot.
- **Shear.** Turn to the new normal, resize to `k''_t A`, strain by `−v_t dt`, add the
  dashpot, then return-map onto the Coulomb cap.
- **Torsion.** The same with `k''_t J` and the cap `μ f_n r̄`.

Nothing in a substep iterates to convergence. Every loop is bounded by hull data.
Only the split handover bisects, and it runs on the host.

### State

`Stick` per `ContactKey`:

- **Keys.** `Ground(s,c)`, `ImpactorGround(i)`, `Impactor(i,s,c)` and
  `Pair(sa,ca,sb,cb)`, keyed by chunk identity so they survive a split's renumbering.
- **`shear`, `torque`.** Mindlin–Deresiewicz cohort layers `(d, Vec<(k_i, d_i)>)`.
- **`normal`, `indent`, `crushed`.**

Lifecycle:

- Every substep, `previous = take(sticks)`; only evaluated pairs are reinserted.
- An empty overlap drops the whole Stick.
- Overlap with no force keeps `indent` and `crushed` and clears the layers.
- Only pairs get an indent. It only ratchets up, through crushing or a split handover,
  and is forgotten when the pair parts.

### The cohort stack: exact, with no fixed size

- **What is already compressed exactly.** The top cohort merges when its `d_i == d`
  (bit-exact in f32 too). Force and energy need only the moments
  `Σk_i`, `Σk_i d_i` and `Σk_i|d_i|²`; the pops need the individual cohorts.
- **No exact fixed-size form exists.** After N steps of sticking growth with distinct
  strains, the pops' forces determine every `(k_i, d_i)`. Turning-point (Preisach)
  compression is exact only for constant `d` during growth, which is the top-merge.
- **The exact bound from the loading history.** Each layer gets at most one push per
  substep, and slip or zero force collapses the stack to one cohort. So the depth is at
  most the count at the batch start plus the batch's substeps.
- **Consequence.** The GPU holds every cohort in an arena sized per batch. The batch
  length is a scheduling knob, not physics.
- **Any fixed cap or merge is a model change.** It would have to be proven in the oracle
  behind its own switch. The candidate merge is one whose energy change
  `½k_ik_j/(k_i+k_j)|d_i−d_j|²` lies below the f32 rounding of the layer's energy;
  such a merge never creates energy. Experiment E1 decides whether it is needed.

### Ground and impactors

- **Ground.** One clip against `z ≤ height`, with no field cells.
- **Box impactor.** `pair_elastic` plus a crush cap, which is a global sum over the
  impactor's candidates.
- **Sphere impactor.** `ball_elastic`, `ball_cells` and `with_ball` (closed-form discs);
  the largest separate code path.
- **Impactor on the ground.** `sphere_cap`, or a half-space clip.

### Cost on the real packs

| Pack | Hull data |
|---|---|
| House | prisms of 6–14 points, ≤ 9 faces, ≤ 7 vertices per face |
| Brick | cuboids |
| Fractured | 12–670 points per hull; imported through `hull_points_reduced` to ≤ 64 vertices, about 34 faces |

Bounds from `MAX_HULL_VERTICES = 64`:

- faces ≤ 124;
- a cell's faces ≤ F_a + F_b + F + C − 1;
- clip incidences grow by at most `3F` per clip.

| Case | Cell faces | Incidences |
|---|---|---|
| Box pair | 23 | 126 |
| House prism | 35 | 198 |
| Typical fractured | 135 | 800 |
| Hull limit | 495 | 2,958 (47 KB) |

The oracle spends about 45 µs of CPU per pair, mostly on allocation.

## 3. GPU mapping

### Kernels

- **New kernels, under proven methods only.**
  - `layer_contacts` covers pairs, ground and box-impactor candidates, each writing a
    force slot.
  - `layer_impactor_shares` computes the elastic record that the crush cap needs.
- **What stays as it is.** `contact_forces` and `ground_contact` remain for established.
  `contact_sum_at` only sums slots. `island_frame` is untouched apart from a
  contact-work term for the oracle's work-based ledger.
- **Work per pair.** One 32-lane SIMD group per pair, as today:
  - lanes run over faces inside each clip;
  - `WavePrefixSum` compacts the kept faces;
  - the cap polygon is sorted with a SIMD bitonic sort on the oracle's integer keys,
    then joined by a one-lane monotone chain;
  - the clips themselves stay sequential, and so do the cells;
  - lanes run over fan triangles for mass properties, and over cohorts with a fixed
    partition, so results are deterministic.

### Workspace

The workspace is derived per pair by the host from the two hulls, not tuned.

- **Threadgroup variant.** 2 pairs of 16 KB each, enough for boxes (about 8 KB) and house
  prisms (about 11 KB).
- **Device variant.** Fractured hulls use a per-pair slab in scratch.
- **No local arrays,** which would spill.
- **Hull upload.** It must add face-vertex index lists; today only vertices, face planes
  and centroids go up.

### State on the device

- **Fixed record.** 3 float4 per contact:
  - (normal, indent);
  - (shear d, twist d);
  - (crushed, counts, flags).
- **Cohort arena.** Shear entries are float4 (k, d); twist entries are float2.
- **Capacity.** The count at the batch start plus the batch's substeps. The host
  compacts the arena at download; the copy is exact.

### Host side

- The host keeps the oracle's own `HashMap<ContactKey, Stick>`, made public in stress-ref
  with f32 pack and unpack, in place of `gpu/contacts.rs` `pair_memory`.
- The existing snapshot, restore, segment redo and frame redo carry it unchanged.
- **Split handover.** After a host `split_cluster`, the oracle's `hand_over_split_joints`
  runs on the shell, with sticks swapped in, before the next substep. It must be
  extracted as a public function.

## 4. f32 robustness

- **The depth floor.** Rounding in vertex positions is about ε·h (6e-8 m at h = 0.5 m).
  The resting elastic sink is about 3e-8 m, the same floor the established GPU contact
  has.
- **Mitigation.** Work in body A's frame, with B transformed by the compensated relative
  pose (`world_diff`, `rotation_err`, TwoSum where it pays). This is a named
  approximation, "f32 pair geometry", measured by E3.
- **Predicates that need care.**
  - Side signs and their exact zeros: identical expressions and no FMA contraction
    differences.
  - The interpolation's operand order.
  - Cap deduplication and the hull's `cross ≤ 0` test.
  - Candidate and gap tests, which use the same vertices as the clip.
  - Bisectors of nearly parallel faces: a second-order effect, to be named.
  - The zero guards.
  - `turned`'s flip test.

## 5. Step bounds

- **Standalone world.** It uses `shell.substep_dt()`, the oracle's own lattice bound
  with the damped factor, so nothing changes.
- **Coupled island frames.** `engaged_contact_dt` is the penalty network.
  - Its block norm already covers the tangential layers, because
    `‖k_n nnᵀ + k_t P‖ = k_n` when E' ≥ G.
  - It omits the rotational coupling that layer contact makes physical: lever arms,
    rocking, torsion, and the dashpots of every mode.
- **Re-derivation.** Add `engaged_layer_contact_dt` to stress-ref.
  - Each contact contributes a 12×12 block
    `K ≼ k''_max (A_max G_tᵀG_t + J_max G_rᵀG_r) + K_geo`.
  - The block is bounded over the plan as `pair_stiffness_within` does.
  - The step comes from the 6-DOF Collatz–Wielandt `certified_radius` and the
    `certified_damped_step` fixed point.
  - Stars, such as a ram on many chunks, must still be handled.
  - The plan, budget and redo machinery carries over unchanged.
  - Until the bound is proven, island frames take the oracle's global proven bound.

## 6. Handover to PhysX

The CPU side is implemented (`ad3dedc6f`).

- **The engine interface.** The RigidEngine methods `set_rest_overlaps` and
  `rest_overlaps`. The world calls `hand_rest_overlaps_to_engine` in `coupled_prepare`
  and `take_rest_overlaps_from_engine` in `coupled_write_back`.
- **What the GPU must do:**
  - hold the oracle's sticks and swap them into the shell around both calls;
  - save and restore the shell's `Coupling::{handed, kept}` with the frame redo;
  - never plan pairs that involve driven clusters.

## 7. Phases (a failing test first in each)

0. **Hazards.**
   - Add `gpu/methods.rs` with its test over `Methods::NAMES`.
   - Under `tangent_statics`, the GPU secant runs at most `SECANT_STEPS`. Clusters it
     fails go to the host `equilibrate_or_keep`, else the scene is rejected.
   - `frame_contact_step`: inert without layer contact, rejected with it (`methods.rs`).
   - Micro case: b9 adaptive with `STRESS_METHODS=-layer_contact`.
1. **stress-ref API.**
   - Make `ContactKey`, `Stick` and `Layer` public, with f32 pack and unpack.
   - Add a sticks swap and a public handover function.
   - The GPU host state moves to sticks; established output must stay bit-identical
     (gate 13/13, micro snapshots unchanged).
2. **Polytope in Slang** (`sl/polytope_test.slang`). Compare against `polytope.rs` over
   random poses (box, prism, coplanar grid, bisectors through edges): volume, centroid,
   area, and watertightness within a rounding bound derived from conditioning.
3. **Field cells and `pair_elastic`.** Compare against the oracle, plus the
   finite-difference gradient test.
4. **`layer_force` with the arena.** Replay cohort sequences recorded from the oracle.
5. **Ground end to end** (the first relaxation of `unsupported`).
   - Principle micro cases: the resting sink of 3.00e-8 m, stick without creep, sliding,
     and a stopping distance of 0.1019 m in 0.204 s.
   - Coverage bits: `LAYER_GROUND`, `LAYER_SLIP`, `COHORT_POP`.
6. **Pairs, crushing and the host split handover.**
   - Micro cases: an elastic collision conserving energy, a split handover, a crushing
     hinge.
   - The work ledger.
7. **Impactors.** Boxes with the two-pass cap, then impactor on ground, then spheres.
8. **`engaged_layer_contact_dt` in stress-ref.** Proven with the examples and the star
   test, then switched on in the GPU.
9. **Coupled indent handover.** Gated by stress-physx `handover_energy` on the GPU world.
10. **Gates.**
    - `stress-gpu-gate --methods proven|established|both`, with a proven quick tier.
    - The guard runs both method sets.
    - Micro cases under proven.
    - An updated kernel budget.

**Risk:**

- Medium for boxes and houses.
- High for fractured hulls (device workspace, many clips).
- The main correctness risk is f32 precision at rest.
- The main engineering risk is the cohort arena.

**Performance estimate:** about 6–11 µs of dependent latency per touching pair.

| Case | Contact kernel per substep | Per substep total |
|---|---|---|
| Today, established | 20–25 µs | — |
| House fracture frame, layer contact | about 30–60 µs | 80–110 µs (+30–50 %) |
| Fractured packs | about 100–250 µs | — |

The proven step changes the substep count separately.

## 8. Experiments

| | Question | Decides |
|---|---|---|
| E1 | Cohort depth histogram in the oracle (b9, wall ram, house ram, packs); share mergeable under the rounding criterion | Arena size; whether a merge switch is needed |
| E2 | Clips and field cells per pair (oracle `--profile`) | Per-pair cost; workspace split |
| E3 | f32 depth floor: resting sink and force error from 1e-8 to 1e-2·h; plain against compensated pose | The "f32 pair geometry" approximation |
| E4 | Engaged layer bound against the lattice bound against observed stability limits; substeps on the rams | The coupled step bound |
| E5 | Threadgroup against device workspace; lanes over faces against lanes over cells | The workspace layout |
| E6 | Time per substep on the house and wall frames against 20–25 µs | The performance cost |
| E7 | Failure rate of GPU secant solves under proven methods | Host fallback against a GPU tangent re-solve |
