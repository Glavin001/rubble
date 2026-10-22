# Learned impact-outcome calls (`learned-impact-outcome/4`)

**Two layers, two owners** (agreed with P1 on 2026-10-21; P1's reply is in the shared notes,
`p1-data-contract-reply.md`):

| Layer | Version | What | Owner |
|---|---|---|---|
| frame record | `learned-impact-frames/1` (`FRAMES_SCHEMA.md`, P1 branch `claude/learned-impact-p1-data-aa822c`) | one scenario's whole life cycle, lossless, frame by frame: state at every frame boundary, the routing's causal inputs, the oracle's outcome, per-bond energies, the energy ledger, QS and punch baselines, keys, splits, ensemble id/member | P1 recorder / generator |
| **calls** (this document) | `learned-impact-outcome/4` | the model's view: one record per routed island per frame, a typed graph of nondimensional, invariant features plus targets | the Rust call builder `stress-learned/src/outcome_calls.rs` (the single feature builder P3 ports; proposed as its own session, not yet started); its output must pass `schema.py`'s validator. Its feature code takes a *frame-start state* (keys → pose, v, ω, joint state, live bonds): a recorded frame is one source, a model-decoded state (pushforward, DAgger, closed loop) another |
| decoded state | `learned-impact-decoded-state/1` (`DECODED_STATE.md`) | the model's decoded outcome of one call, keyed by stable keys: what DAgger restarts and the closed-loop harness apply | P2 writes (`outcome/state_io.py`), P1 (DAgger) and the rollout harness read |

The trainer reads only this layer; the frame record stays the archival truth, so a feature
change regenerates calls without re-running the oracle.

For the plan in `docs/learned-stress-solver/learned-stress-solver-plan.md` and the boundary
P0 validated (`findings/p0-feasibility-2026-10-20.md`, step 3):

> one call per routed impact island per engine frame; inputs are the frame-start state;
> outputs are breaks, carried damage, each piece's frame-end pose, linear and angular
> velocity, and the impactor's; applied at window end with the island's contacts filtered.

It is unrelated to the older `learned-impact/1` (`SCHEMA.md`), which stays as it is.

The Python mirror of this document is `python/learned_impact/outcome/schema.py`
(dataclasses, column names, loader and validator). **If the two disagree, that is a bug in
one of them; the version string changes whenever either changes meaning.**

### Where each part comes from in the frame record

| Calls field | Event-recording source |
|---|---|
| island membership (one call per routed island) | `frame_chunk_routed`, `frame_impactor_routed` (debris-reach routing, behind its flag, adds structures) |
| raw node state, `y_*_v_end`, `y_*_dx`, `y_*_drot` | `state_*` rows `f` and `f + 1` (Δx = Δpos; Δθ from the quaternions) |
| `y_chunk_fragment` | connected components of `state_bond_connected` row `f + 1` |
| bond, cracked edges, frames, offsets | `static_bond_endpoints`, `static_bond_geometry`, `state_bond_connected` |
| bond joint columns | `state_bond_joint` (damage, crush, utilization) |
| `y_bond_peak_util`, `y_bond_broke`, `y_bond_break_bin` | `frame_bond_peak_util`, `frame_bond_break_bin` (bin + 1; −1 → 0) |
| impactor contact edges | `contact_i32`, `contact_f32` |
| `log_qs_hertz_util`, punch columns | `frame_bond_qs_prior_util`, `frame_chunk_punch_plug`, `frame_bond_punch` |
| variant-B groups, `base_qs_*`, `base_punch_*` | `qs_*`, `punch_*` |
| energy | `frames.json` ledger (`KE_in/out` exact; booked contact terms diagnostic only) |
| `y_bond_dissipated`, `D_bond_joint`, `D_bond_damping` | `frame_bond_dissipated [F,E,2]` (recorder revision ≥ 5): (joint = fracture + cracked-face friction + rebar, **not pure fracture**; dashpot), per frame, differenced in f64. Records before revision 5: difference `state_bond_energy [F+1,E,3]` instead (agrees to 1e-5 on M5) |
| `phase` | `frame_phase`: P1 writes pre_contact / contact / secondary / separation / settling; `blast` arrives in P5, and quiet non-impact frames are `pre_contact` or `settling` |
| `y_impactor_regime` | full-run label per impactor, relative to the remnant (largest fragment): perforate / arrest / rebound / deflect |
| `split`, `split_group`, `layout_family` | `context.json` + `outcome_sampling.partition` |
| `ensemble_id`, `ensemble_member`, `perturbation` | `context.json` → `ensemble = {id, member, size, perturbation: {impactors: [{offset_m, speed_scale}], weibull_seed}}` (8 members on val/test/sealed and 15% of train); aggregation is `learned_impact.outcome.ensemble` only |
| `split`, `split_group` | `context.json` |

## Rules that hold everywhere

1. **Every model input is written by the recorder.** Python never derives an input feature
   (single source: the Rust feature builder the recorder uses is the one P3 ports to the
   engine). The only things Python builds are the BSMS hierarchy (from the pristine bond
   graph, at load time, as the plan says), its coarse-edge fibre features (the pooled
   centre-of-mass offset in the source node's body frame ÷ L, and its length ÷ L, computed
   inside the model from the raw `chunk_pos` / `chunk_q_world`; P3 ports these with the
   hierarchy) and batching.
2. **Inputs are invariant**: no absolute position, no world-frame vector, no per-bond or
   per-chunk identity. Vectors are in a local frame (a body frame for nodes, the edge frame
   for edges). Every feature is a ratio to a physical scale of the call (`scales` below),
   never normalised per graph.
3. **Raw arrays are kept for the decode and the metrics** (masses, world inertia, positions
   relative to the island's centre of mass, world velocities, frames). They are never model
   inputs; the loader keeps them in a separate namespace.
4. **Fail loud.** A missing required array, a wrong shape, a non-finite value or an unknown
   column is an error, never a silent default. Optional groups have explicit availability
   flags (below); a variant that needs a group the record lacks refuses to train.
5. Units are SI. Time bins split the frame `[0, dt)` into `n_time_bins = 8` equal bins.

## Directory layout (per event)

```
<event_id>/
  manifest.json   # event metadata, feature-column lists, availability flags, full-run outcome
  calls.json      # one entry per call: metadata, scales, energies, array catalog
  arrays.bin      # every array of every call (and the pristine graph), concatenated
```

`arrays.bin` may instead be stored as `arrays.bin.zst`: one zstd frame of exactly the same
bytes (lossless; offsets refer to the decompressed bytes). Exactly one of the two exists.
`arrays.bin` is little-endian, row-major, no headers; each array starts at an 8-byte-aligned
offset. A catalog entry is `{"dtype": "f32"|"i32"|"i64", "shape": [...], "offset": bytes}`.
Python reads it with `np.memmap`. Three files per event keep 30k events × ~10 calls
manageable (one `.bin` per array would be ~10⁷ files).

## `manifest.json`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | exactly `"learned-impact-outcome/4"` |
| `event_id` | string | directory basename |
| `recipe_id` | string | generator recipe (replayable) |
| `family` | string | e.g. `free_wall`, `wall_row`, `wall_opening` |
| `layout_family` | string | layout id; whole layouts are held out |
| `split_group` | string | partition key; prefixes, cadence variants, branches keep the parent's |
| `split` | string | `train` / `val` / `test` / `sealed` (assigned by the generator from `split_group`) |
| `ensemble_id` | string or null | members of one ensemble share it |
| `ensemble_member` | int or null | 0-based |
| `ensemble_size` | int | 1 when not an ensemble |
| `perturbation` | object | `{offset_m: [3], speed_scale, weibull_seed}` |
| `seed` | u64 | |
| `oracle` | object | `{config: "realtime", methods, contact: "penalty"|"layer", hash}` |
| `scene_hash` | string | FNV-1a of the scene JSON |
| `recorder_revision` | string | git commit of the recorder |
| `frame_dt` | f64 | 1/60 |
| `n_time_bins` | int | 8 |
| `gravity` | [3] f64 | world, for the decode only |
| `truncated` | bool | hit the 10 s cap: excluded from outcome metrics |
| `n_calls` | int | |
| `columns` | object | the feature column names of each feature array, in order (see below); the loader checks them against `schema.py` exactly |
| `available` | object | `{qs_hertz, punch, qs_frame, punch_frame, blast, cracked_edges, structure_contacts, bond_energy_split}` booleans |
| `source` | string, optional | where the oracle outcome came from: `"standalone"` (stress-ref alone; the bulk v1 data, kept as pretraining) or `"coupled"` (stress-ref + rubble, production's inputs). Absent means standalone. The trainer mixes sources by weight (`--dataset path@w,…`) |
| `outcome` | object | full-run outcome (below) |
| `structures` | array | per structure `{structure_id, n_chunks, n_bonds}`; the pristine graph is in `arrays.bin` as `structure/<id>/…` |

**Full-run outcome** (`manifest.outcome`), from the whole recorded life cycle: per impactor
`{key, v_exit: [3], speed_exit, regime: "arrest"|"perforate"|"rebound"|"deflect"}`
(regime labelled geometrically); `n_fragments`, `largest_fraction`, `fragment_masses`
(sorted, kg), `retained_anchored_fraction`, `collapsed`, `sim_time`, `frames`.

### Pristine structure graph (`structure/<id>/…`, once per event)

Used to build the BSMS hierarchy once per intact structure and to align keys across calls.

| name | dtype | shape | notes |
|---|---|---|---|
| `chunk_key` | i64 | `[Ns]` | stable chunk key, `(structure_id << 32) | chunk_index` |
| `bond_key` | i64 | `[Bs]` | stable bond key, `(structure_id << 32) | bond_index` |

`chunk_index` is the chunk's position within its structure: `c` of the frame record's
`context.json` `chunk_keys [s, c]`. `bond_index` is the scene's own bond index, `source` of
`bond_keys [s, source, a, b]` (in today's unrefined walls it equals the bond's rank within
the structure, but `source` survives any filtering). Keys are identical across ensemble
members, branches and restarts of one scene. One Rust key function in `stress-learned`,
shared by the call builder and the recorder's `--restart`, defines them.
| `bond_ends` | i32 | `[Bs,2]` | local chunk indices into `chunk_key` |
| `chunk_rest_pos` | f32 | `[Ns,3]` | rest COM positions in any fixed frame; used only to choose BSMS seeds (nearest to each cluster's centroid, a rotation-invariant choice) |

## `calls.json`: one entry per call

| Field | Type | Notes |
|---|---|---|
| `call_index` | int | order in the event |
| `frame` | int | engine frame index |
| `island` | int | island index within the frame |
| `prev_call` | int or null | the same island's call in the previous frame (lineage for pushforward) |
| `phase` | string | `pre_contact`, `contact`, `blast`, `separation`, `secondary`, `settling`, `quiet` |
| `transition` | bool | first frame of a new phase, a first break, or a regime change |
| `routed` | bool | false for the non-impact frames recorded for variant C |
| `anchored` | bool | island contains any anchored chunk (momentum not conserved) |
| `counts` | object | `{chunks, impactors, bond, cracked, impactor_contact, structure_contact}`: edge counts are **undirected** |
| `scales` | object | below |
| `energy` | object | below (J, f64) |
| `arrays` | object | name → catalog entry |

### `scales` (every feature divides by one of these)

| key | meaning |
|---|---|
| `L` | median chunk size (m): cube root of median chunk volume |
| `m` | median chunk mass (kg) |
| `v` | max(5 m/s routing speed, the predicted contacts' closing speeds, every island chunk's and impactor's \|v_i − v_com\| and \|ω_i\|·R_i with R_i = \|half extents\|) (from `/3`) |
| `c` | P-wave speed `sqrt(E/ρ)` of the island's dominant material (m/s) |
| `f_t` | median bond tensile strength (Pa) |
| `E` | median modulus (Pa) |
| `GA` | median bond `G_f·A` (J), tension mode |
| `dt` | frame length (s) |
| `E_in` | `KE_in + W_gravity + W_external` of the call (J); floor `½·m·v²` so it is never 0 |

## Nodes

Node order in every call: chunks `[0, Nc)`, then impactors `[Nc, Nc+Ni)`. The island's
**global node** is the `global_feat` row. Edges index the concatenated node list.

**Body frame.** Each node carries its world orientation `q_world` (unit quaternion, w-first).
Node vector features and node outputs are expressed in that body frame, so a rotation of the
whole scene leaves every feature unchanged and rotates every decoded world vector.

### Chunk arrays

| name | dtype | shape | notes |
|---|---|---|---|
| `chunk_key` | i64 | `[Nc]` | stable key |
| `chunk_feat` | f32 | `[Nc, 33]` | columns below |
| `chunk_qs_feat` | f32 | `[Nc, 6]` | variant B: `blast_parity` QS frame Δv, Δω in body frame (÷v, ×L/v); zero if `available.qs_frame` is false |
| raw `chunk_mass` | f32 | `[Nc]` | kg |
| raw `chunk_inertia_body` | f32 | `[Nc,3]` | principal inertia (kg m²), body axes |
| raw `chunk_q_world` | f32 | `[Nc,4]` | body → world |
| raw `chunk_pos` | f32 | `[Nc,3]` | world axes, relative to the island centre of mass (incl. impactors) |
| raw `chunk_v`, `chunk_w` | f32 | `[Nc,3]` | world, frame start |
| raw `chunk_anchored` | i32 | `[Nc]` | 1 fixed/pinned |

`chunk_feat` columns (`CHUNK_FEAT_COLUMNS` in `schema.py`):

| # | name | definition |
|---|---|---|
| 0 | `log_mass` | ln(mass/m) |
| 1–3 | `inertia_0..2` | principal inertia ÷ (mass·L²) |
| 4–6 | `half_extent_0..2` | ÷ L |
| 7 | `support_fixed` | 0/1 |
| 8 | `support_pinned` | 0/1 |
| 9 | `bond_damage_mean` | mean damage of its live bonds |
| 10 | `bond_damage_max` | |
| 11 | `n_live_bonds` | live bonds ÷ 6 |
| 12 | `n_broken_bonds` | broken pristine bonds ÷ 6 |
| 13 | `hops_to_contact` | bond-graph hops to the nearest predicted impactor contact ÷ 16 (cap 4 → 64 hops); 4 if none |
| 14 | `wave_arrival` | (t_contact + geodesic distance / c) ÷ dt, capped at 4 |
| 15 | `hops_to_anchor` | hops to the nearest anchored chunk ÷ 16; 4 if none |
| 16–18 | `gravity_body_0..2` | gravity in body frame ÷ 9.81 |
| 19–21 | `v_rel_body_0..2` | (v − v_com,island) in body frame ÷ v |
| 22–24 | `w_body_0..2` | ω in body frame × L ÷ v |
| 25 | `punch_in_plug` | 0/1: the punch's plug contains this chunk |
| 26–31 | `blast_0..5` | load-agnostic pressure features (plan P5): p_r·A ÷ (f_t·A_bond), impulse in frame ÷ (mass·v), arrival ÷ dt, duration ÷ dt, clearing ÷ dt, shadowed fraction. **Zero for impacts** |
| 32 | `in_contact_any` | 0/1: any contact edge touches it |

**Carried columns** (rewritten from the model's own decoded output in pushforward, noised in
training; `CARRIED_COLUMNS` in `schema.py`): 19–24. Everything else in a successor call
(topology, damage summaries, geometry) stays the oracle's; DAgger covers those.

### Impactor arrays

| name | dtype | shape | notes |
|---|---|---|---|
| `impactor_key` | i64 | `[Ni]` | the impactor's index in the scene's impactor list (stable across members, branches and restarts; the same key function as chunks and bonds) |
| `impactor_feat` | f32 | `[Ni, 16]` | columns below |
| `impactor_qs_feat` | f32 | `[Ni, 6]` | variant B: QS-frame Δv, Δω (as chunks) |
| raw `impactor_mass`, `impactor_inertia_body`, `impactor_q_world`, `impactor_pos`, `impactor_v`, `impactor_w` | f32 | as chunks | |

`impactor_feat` columns (`IMPACTOR_FEAT_COLUMNS`):

| # | name | definition |
|---|---|---|
| 0 | `log_mass_island` | ln(m_imp ÷ m_island) |
| 1 | `log_mass_chunk` | ln(m_imp ÷ m) |
| 2–4 | `inertia_0..2` | principal ÷ (m_imp·R²) |
| 5 | `log_radius` | ln(R ÷ L) |
| 6 | `log_modulus` | ln(E_imp ÷ E) |
| 7 | `crushable` | 0/1 |
| 8–10 | `gravity_body_0..2` | ÷ 9.81 |
| 11–13 | `v_rel_body_0..2` | (v − v_com,island) in body frame ÷ v |
| 14 | `speed_over_c` | ‖v_rel‖ ÷ c |
| 15 | `in_contact_any` | 0/1 |

Impactor ω enters through its contact edges (spin in the contact frame), not here.
Carried: 11–14.

### Global node (`global_feat`, f32 `[1, 12]`)

| # | name | definition |
|---|---|---|
| 0 | `log_energy_ratio` | ln(Σ impactor KE ÷ Σ_island G_f·A) |
| 1 | `log_energy_ratio_plug` | ln(Σ impactor KE ÷ Σ_plug G_f·A); 0 if no punch |
| 2 | `log_dt_c_over_L` | ln(dt·c ÷ L) (wave transits per frame; ≈ 500–700 raw on walls) |
| 3 | `dt_v_over_L` | dt·v ÷ L |
| 4 | `n_impactors` | ÷ 4 |
| 5 | `log_n_chunks` | ln(Nc) ÷ 10 |
| 6 | `gravity_mag` | ÷ 9.81 |
| 7 | `anchored` | 0/1 |
| 8 | `blast_log_tnt` | ln(W ÷ 1 kg); 0 for impacts |
| 9 | `blast_z` | Z (m/kg^⅓) ÷ 10; 0 for impacts |
| 10 | `blast_impulse` | pulse impulse ÷ (m_island·v); 0 for impacts |
| 11 | `v_over_c` | v ÷ c |

## Edges

Four edge types, each with its own encoder. **Every edge is stored as two directed twins**:
rows `2k` and `2k+1` are the same physical edge, with endpoints swapped and every
direction-dependent feature recomputed from the swapped side (the MeshGraphNets
convention). Per-edge targets are indexed by the undirected edge `k`.

**Direction convention** (normative; pushforward re-derives the carried columns with it,
and `tests/test_outcome_pushforward.py` checks it against recorded calls):

- `T_frame[k]` = rows `(t1, t2, n)` in world axes for row `2k`, `n` pointing from its src
  to its dst; row `2k+1` uses the reversed right-handed frame `(−t1, t2, −n)`.
- Bond and cracked rows: `offset` = (bond point − src COM) in the row's frame ÷ L;
  `v_rel_edge` = row frame · (v_dst(p) − v_src(p)) ÷ v, where v_x(p) = v_x + ω_x × (p − x_x)
  is the velocity of body x at the bond point p.
- Contact rows: `rel` = v_src − v_dst (COM velocities); `v_n` = rel·n ÷ v (positive
  closing), `v_t_i` = rel·t_i ÷ v, `v_n_over_c` = rel·n ÷ c, all in the row's frame; `spin` =
  the impactor's ω in the row's frame × R ÷ v (the same body on both twins).
- `rot_src`, `rot_dst` use the row's frame.

Each type `T ∈ {bond, cracked, impactor_contact, structure_contact}` has:

| name | dtype | shape | notes |
|---|---|---|---|
| `T_src`, `T_dst` | i32 | `[2Et]` | node indices |
| `T_key` | i64 | `[Et]` | undirected stable key: the bond key for `bond`/`cracked`; for contacts `(key_a·1_000_003) ^ key_b` with `key_a < key_b` |
| `T_feat` | f32 | `[2Et, F_T]` | columns below |
| raw `T_frame` | f32 | `[Et, 3, 3]` | rows `t1, t2, n` in world axes, `n` pointing src→dst of row `2k` |

**Edge-frame orientation features.** Every type ends with 12 columns `rot_src_0..5`,
`rot_dst_0..5`: the first two columns of `R_body(src)ᵀ·R_edge` and of `R_body(dst)ᵀ·R_edge`
(the 6-D rotation representation), so the network can carry vectors between body frames
without seeing a world frame.

### `bond_feat` (`[2E, 41]`), live bonds (intact or damaged)

| # | name | definition |
|---|---|---|
| 0 | `log_area` | ln(A ÷ L²) |
| 1–3 | `offset_0..2` | bond point relative to src COM, edge frame, ÷ L |
| 4–9 | `log_stiffness_0..5` | ln(k ÷ k_ref) per DOF; k_ref = E·A/L (axial, shear), E·A·L (angular) |
| 10–12 | `log_strength_0..2` | ln(strength ÷ f_t): tension, shear, compression |
| 13–15 | `log_ga_0..2` | ln(G_f·A ÷ GA) per mode |
| 16 | `weibull` | strength factor |
| 17 | `damage` | |
| 18 | `crush` | |
| 19 | `fail_margin` | failure margin at frame start, q = max over the families of κ ÷ κ_fail (below); < 1 for a live bond |
| 20 | `log1p_utilization` | ln(1 + current `JointState.utilization`); the undamaged-stress failure index, which exceeds 1 once softening starts (up to ≈ 200 raw) |
| 21 | `log1p_peak_util_prev` | ln(1 + peak utilisation over the previous frame); 0 on the first call |
| 22 | `rebar` | 0/1 |
| 23 | `log_qs_hertz_util` | ln(QS utilization with the impact as a Hertz-peak load), converged `solve_static_all`; 0 if `available.qs_hertz` false |
| 24 | `punch_plug_bond` | 0/1: holds the punch's plug |
| 25 | `log_punch_capacity_ratio` | ln(m_eff·v/τ ÷ plug capacity); 0 off-plug |
| 26–28 | `v_rel_edge_0..2` | (v_dst − v_src at the bond point) in edge frame ÷ v |
| 29–40 | `rot_src_0..5`, `rot_dst_0..5` | |

**Failure margin** (`fail_margin`, `y_bond_fail_margin`), from `stress_ref::joint`: per family
(tension/shear with `kappa`, `ductility`; compression/buckling with `kappa_c`,
`ductility_c`), κ_fail = max(r, 1) for a brittle law with softening (D = 1 iff κ ≥ r; r ≤ 1
snaps at κ > 1), κ_fail = (r + 1)/2 for a ductile law, κ_fail = 1 with softening off; the
family's q = κ ÷ κ_fail, and the bond's q is the larger of the two. With rebar the bond
disconnects only when the rebar also ruptures, so its q = min(q_concrete,
rebar_work ÷ rupture_work). q ≥ 1 exactly when the bond fails, which makes the censored
threshold τ = 1 exact (the plan's peak-utilisation test is not: see "Changes in /2").

Carried: 17, 19, 21, 26–28. **Variant B group** `bond_qs_feat` `[2E, 3]`:
`log_qs_frame_peak_util`, `qs_frame_broke` (0/1), `qs_frame_break_time` (bin ÷ 8, 0 if none).

### `cracked_feat` (`[2Ec, 22]`): broken bonds whose faces are still unilateral contacts

A broken bond is a cracked edge only while its gap along the bond normal is ≤ `CONTACT_MARGIN` = 0.25 L (the structure-contact margin) **and** the two chunks overlap across the face along both t1 and t2 (from `/3`).

`log_area`, `offset_0..2`, `gap` (÷ L, signed, negative = overlap), `friction`,
`v_rel_edge_0..2` (÷ v), `frames_since_broken` (÷ 8, cap 1), `rot_src_0..5`, `rot_dst_0..5`.
Carried: 6–8 (`v_rel_edge`).

### `impactor_contact_feat` (`[2Ei, 30]`): predicted impactor contacts from the routing sweep

Row `2k` is the twin whose src is the impactor.

| # | name | definition |
|---|---|---|
| 0 | `v_n` | approach velocity, normal, ÷ v (positive closing) |
| 1–2 | `v_t_0..1` | tangential ÷ v |
| 3 | `v_n_over_c` | normal approach ÷ c |
| 4 | `log_mass_ratio` | ln(m_other ÷ m_self) |
| 5 | `log_radius` | ln(R_impactor ÷ L) |
| 6 | `log_modulus` | ln(E_impactor ÷ E_chunk) |
| 7–9 | `spin_0..2` | impactor ω in the contact frame × R ÷ v |
| 10 | `t_contact` | time of contact ÷ dt; −1 if already in contact at frame start |
| 11 | `log_energy_ratio` | ln(E_k ÷ Σ_plug G_f·A); 0 if no plug |
| 12 | `log_hertz_force` | ln(Hertz peak force ÷ (f_t·median bond area)) |
| 13 | `hertz_duration` | Hertz contact time ÷ dt |
| 14 | `is_self_impactor` | 1 on the twin whose src is the impactor |
| 15 | `overlap` | penetration ÷ L at frame start, 0 if none |
| 16 | `friction` | coefficient |
| 17 | `restitution` | coefficient |
| 18–29 | `rot_src_0..5`, `rot_dst_0..5` | |

Carried: 0–3, 7–9.

### `structure_contact_feat` (`[2Es, 22]`): contacts between chunks of different structures or fragments inside the island

`v_n`, `v_t_0..1`, `v_n_over_c`, `log_mass_ratio`, `t_contact`, `overlap`, `friction`,
`log_area` (contact patch ÷ L², 0 if unknown), `same_structure` (0/1: two fragments of one
structure), `rot_src_0..5`, `rot_dst_0..5`. Carried: 0–3.

## Targets (per call; all from the oracle over this frame)

### Per bond (undirected live bond `k`, `[E]`)

| name | dtype | notes |
|---|---|---|
| `y_bond_fail_margin` | f32 | peak failure margin q over the frame (running max over the substep evaluations); ≥ 1 exactly for bonds that broke (censored there). **The break target** |
| `y_bond_peak_util` | f32 | peak `JointState.utilization` over the frame (`record_peak_utilization`); auxiliary: ≥ 1 is the onset of softening, not failure |
| `y_bond_broke` | i32 | 0/1 |
| `y_bond_break_bin` | i32 | 0 none, 1..8 time bin of the break |
| `y_bond_fail_mode` | i32 | 0 none, 1 tension, 2 shear, 3 compression, 4 buckling, 5 rebar |
| `y_bond_ddamage` | f32 | damage_end − damage_start (≥ 0; 0 for broken) |
| `y_bond_dissipated` | f32 | J, Δ`JointState.dissipated` over the frame (fracture + bond damping) |
| `y_bond_support_path` | i32 | optional, 0/1: on a support load path (QS gravity utilisation ≥ 0.1); default 0 |
| `y_bond_break_freq` | f32 | **optional**: ensemble break frequency, present when the call is ensemble-aligned (below) |

### Per chunk (`[Nc]`)

| name | dtype | shape | notes |
|---|---|---|---|
| `y_chunk_v_end`, `y_chunk_w_end` | f32 | `[Nc,3]` | world, frame end, each chunk's own body in the oracle (unreduced) |
| `y_chunk_dx` | f32 | `[Nc,3]` | COM displacement over the frame, world |
| `y_chunk_drot` | f32 | `[Nc,3]` | rotation over the frame, rotation vector, world |
| `y_chunk_fragment` | i32 | `[Nc]` | fragment id at frame end (connected components of intact bonds) |

### Per impactor (`[Ni]`)

`y_impactor_v_end`, `y_impactor_w_end`, `y_impactor_dx`, `y_impactor_drot` (f32 `[Ni,3]`,
world), `y_impactor_regime` (i32: 0 arrest, 1 perforate, 2 rebound, 3 deflect, −1 undecided;
the **full-run** geometric label of that impactor), optional `y_impactor_perforate_freq`
(f32, ensemble).

### Energy (`calls.json` → `energy`, J, f64)

| key | target? | notes |
|---|---|---|
| `KE_in`, `KE_out` | yes (exact) | translational + rotational, island + impactors |
| `W_gravity`, `W_external` | input to the bound | work over the frame; `W_external` is the blast work, 0 for impacts |
| `E_elastic_start`, `E_elastic_end` | yes | stored bond energy |
| `D_bond` | yes | Σ `y_bond_dissipated` (must match it within 1e-6 relative) |
| `D_bond_joint`, `D_bond_damping` | yes if `available.bond_energy_split` | split of `D_bond`: joint (fracture + cracked-face friction + rebar) and bond dashpot |
| `D_fracture_floor` | diagnostic | Σ_broken G_f·A |
| `D_contact_booked`, `D_friction_booked` | **diagnostic only, never a target**: P0 found the penalty contact books its dashpot quadrature, not its work (ledger residual +0.3–4.7% E0) |
| `ledger_residual` | diagnostic | (KE_out + E_el_end + D_total − KE_in − E_el_start − W) ÷ E_in |

The trainer's energy targets are therefore `KE_loss = KE_in + W_gravity + W_external − KE_out`
(exact), `D_bond` (and its split) and `E_elastic_end`.

### Baseline outcomes on the same state (three-way report, variant B base)

`base_qs_*` (`blast_parity` QS frame) and `base_punch_*` (the **CPU** punch: `World::predict_punch`'s
decision rule, then the `blast_parity` QS frame, under the oracle's contact, not rubble's GPU
rigid engine), each when available:
`_bond_broke` (i32 `[E]`), `_bond_peak_util` (f32 `[E]`), `_chunk_v_end`, `_chunk_w_end`
(f32 `[Nc,3]`), `_impactor_v_end` (f32 `[Ni,3]`). Raw, for metrics; variant B's *inputs*
come only from the `*_qs_feat` groups.

## Ensembles

Members are separate events with the same `ensemble_id`. A call is **ensemble-aligned** when
every member has a call with the same `call_index`, the same island chunk-key set and the
same live-bond key set (typically the first impact call). For aligned calls the
`y_*_freq` arrays are written into every member's call (aggregated over members by bond and
impactor key), plus `ens_impactor_v_end` (f32 `[M, Ni, 3]`, all members, member order) for
afCRPS. Unaligned calls omit them. Either the recorder or the Python tool
`learned_impact.outcome.ensemble` (which writes the same arrays in place) may produce them;
the arrays are identical either way.

## Changes in `/2` (2026-10-21)

Found by the output-side ceiling (`ceiling.py`) on P1's smoke calls: pushing the oracle's own
outcome through the deployed decode, τ = 1 on peak utilisation misclassified 8,338 val
bonds against 4,595 real breaks. Utilisation is the failure index of the *undamaged* stress,
so ≥ 1 marks the onset of softening; a bond fails when κ reaches κ_fail. Every other
stage (encoding, momentum projection, the 5% KE bound) reproduced the oracle exactly.
- added `fail_margin` (bond feature 19, carried) and `y_bond_fail_margin` (the break target);
- `utilization` → `log1p_utilization`, `peak_util_prev` → `log1p_peak_util_prev`, and
  global `dt_c_over_L` → `log_dt_c_over_L` (raw values up to ≈ 200 and ≈ 720 on the smoke
  calls);
- the reference speed `v` for a call without an impactor is the routing speed, 5 m/s, as
  `/1` already specified (the smoke converter floored it at 0.05 m/s, which put
  `v_rel_body` up to ≈ 90).

## Changes in `/3` (2026-10-21)

- **Reference speed:** `v` now also covers the island's own motion: 5 m/s at least, plus every body's speed relative to the island centre of mass. Before, it was the predicted contacts' closing speed alone: 0.03–0.06 m/s in 10% of calls while debris moved at 40–80 m/s, so some features reached about 600. Now `v` has median 5 and max 76 m/s. Chunk `v_rel_body` ≤ 1.0 and cracked `v_rel_edge` ≤ 1.7. Amplitude still reaches the model through the `/c` features.
- **Cracked edges:** a face must be within the contact margin and overlap laterally to count (it was gap < 0.5 L along the normal only).

## Changes in `/4` (2026-10-21): hidden deformation carried across calls

Why: P0's closed loop shows the reduced state drops each chunk's elastic deformation at the
handoff. The free wall's reduced-state ceiling fails fragment motion, and keeping the
deformation passes it. Anchored walls under gravity lose their prestress every routed frame
and over-crack. Definitions are P0's, from what `STRESS_DECODED_DEFORMATION=1` restores
(`world_handoff.rs` `apply_decoded_with`). They are provisional until P0's cl-eval-v2
confirms them.

**Frame: each chunk's own body (its cluster, the fragment), not the island.** The body's
rigid motion is the momentum-exact fit of its chunks:
- COM = Σ m·x / M;
- R_body = exp(mass-mean drot)·R0;
- v_body = P / M;
- ω_body = I_w⁻¹ · Σ (r × m v + I_c w).

So the hidden parts carry zero net linear and angular momentum per body by construction.

**Hidden state per chunk (12 numbers), exactly the solver's `ChunkState` (u, th, v, w):**
- `u` = R_body⁻¹(x − X_body) − c_rest;
- `θ`: the chunk's world orientation = R_body · exp(θ) · R_rest;
- `v_def` = R_body⁻¹(v − v_body − ω_body × r);
- `ω_def` = R_body⁻¹(w − ω_body).
- For anchored bodies v_body = 0.

The recorder has u and θ (`state_chunk_hidden [n,6]`). The rates follow from
`state_chunk_vel` / `angvel` and the body fit (P1).

**What changes:**
- **Inputs only (`chunk_feat` gains 12 columns).** `def_u_body_0..2` ÷ u_c, where u_c is
  the mean tensile failure elongation of the chunk's live bonds, so 1 ≈ at failure;
  `def_rot_body_0..2` ÷ (u_c ÷ R_i); `def_v_body_0..2` ÷ `v`; `def_w_body_0..2` · R_i ÷ `v`.
  The raw `/3` pose and velocity arrays already include the hidden parts. What the model
  can't see today is the split: the body frame the deformation is measured from. SI
  copies go in carried arrays `chunk_def` (f32 `[Nc,12]`), for pushforward.
- **No new targets or outputs.** The per-chunk `y_chunk_v_end` / `w_end` / `dx` / `drot`
  (unreduced) and the chunk head's 12 outputs already determine all four at frame end; the
  host derives them from the fit. Decoded state stays `/1`. For the loss only, the builder
  adds `y_chunk_def_end` (f32 `[Nc,12]`, from the frame-end boundary's hidden state): an
  exact target, since a fit from f32 world arrays would lose µm-scale u.
- **Body per chunk:** `chunk_body` (i32 `[Nc]`, the frame-start cluster index, local to
  the call) and `chunk_body_q` (f32 `[Nc,4]`, that body's world rotation, w-first like `chunk_q_world`). Pushforward
  needs them to re-derive v_def / ω_def from predicted velocities in the body frame.
- **Input scale u_c:** the mean over the chunk's live bonds of f_t·A ÷ k_n (failure force
  over normal stiffness), floored at 1e-9 m; a chunk with no live bonds takes the island
  median. Also carried as `chunk_def_uc` (f32 `[Nc]`).
- **Decode:** it must not fold a body to rigid; it doesn't today. The per-island momentum
  projection stays.
- **Bond prestress:** not boundary state. A bond's force follows from both ends' (u, θ) and
  the rest geometry. A derived `def_strain` bond column (elongation ÷ failure elongation)
  may be added if it helps learning.
- **KE bound:** unchanged. KE_out already includes the hidden velocities; elastic energy
  would be a refinement.
- **Loss (P2):** `def_motion` is the KE norm (½ m |Δv_def error|² + ½ ωᵀ I ω on the ω_def
  error) of the frame-end hidden rates about each target body's fit, ÷ E_in and normalised
  by its zero predictor. Without it, the deformation part is a tiny share of the motion
  term. A strain-energy-norm `def_pose` on u is a later option.
- **Pushforward:** rewrites the hidden-state inputs from the model's decoded chunk state
  through the same fit.

**Bond internal state (answered by P0).** P0's cl-eval-v2 reset every surviving bond's
joint history at the call boundary, keeping only damage. The reset covers κ, crushing,
ductility, fatigue, plastic offsets, rebar and strain rate (`STRESS_DECODED_RESET_JOINTS=1`,
candidate `ceiling_def_nojoint`). It gives the same gate numbers as `ceiling_def` on both
families. So damage is the only bond state the boundary needs, and `/4` has no
joint-history channel.

## Versioning

`learned-impact-outcome/4`. Any change of meaning, order or shape bumps the version; the
loader accepts exactly one version. Additive optional arrays may be added within a version
only if old readers ignore them and they appear in this document.

## A call inside a closed-loop provider request

P0's harness (`learned_p0 ceiling --variant provider --provider files`,
`PROVIDER_PROTOCOL.md` on P0's branch) writes `D/req/<f:06>_<i>/request.json`. Once the call
builder exists, it also writes the call there as an ordinary single-call event directory in
this format: `manifest.json` (with the structures' pristine graphs), `calls.json` with one
entry, and `arrays.bin`. `request.json` then sets `call_record` to that directory's path,
relative to the request directory, e.g. `"call"`. Targets are absent in closed loop (no
oracle runs), so the provider loads such records with the validator's `require_targets=False`.
The provider is `python -m learned_impact.outcome.provider`.
