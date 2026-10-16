# OpenCourant (OpenRadioss) oracle

Explicit continuum FE oracle for the stress-solver validation scenes.  Every chunk is a
block of elastic hexahedra, every scene bond is a layer of failing cohesive elements, and
everything is driven only by the scene file (`stress-scene/1`).

```bash
oracles/opencourant/install.sh                      # pinned build, sha256-verified (see PINNED.md)
python -I oracles/opencourant/test_export.py        # exporter smoke test on b1 (no solver)
python -I oracles/opencourant/pipeline.py scenes/b1_bond_tension.json [--seed N] \
       [--elements-per-chunk K] [--cohesive-thickness F] [--threads T]
# -> golden/<scene>/opencourant[_seedN].json + provenance_opencourant[_seedN].json
```

`pipeline.py` = `export.py` (scene -> starter/engine decks + `map.json`) -> `run.py`
(starter, engine, `th_to_csv`, `anim_to_vtk`) -> `observe.py` (CSV/VTK -> observation).
Run directories live in `/home/user/oracle-runs/opencourant/<scene>` (override with
`--rundir` or `OPENCOURANT_RUNS`).  Per-scene hints can be put in the scene's
`oracle.opencourant` object (`elements_per_chunk`, `cohesive_thickness`,
`contact_gap_fraction`, `contact_damping`, `impact_gap`, `dt_safety`, `threads`, ...);
CLI flags override them.

## Model

| scene item | OpenRadioss model |
|---|---|
| chunk (box) | 8-node hexahedra, `K` (default 2) along the body's smallest chunk extent on each axis (flat chunks get flat elements; grids conform across bonds), `/PROP/SOLID` Isolid=24, `/MAT/LAW1` with the chunk material's E (x `stiffness_scale`), nu, rho |
| bond | layer of 8-node cohesive solids `/PROP/TYPE43` + `/MAT/LAW169` over the bond patch, one per pair of matching chunk-face elements |
| bond strength | `TENMAX = f_t * weibull`, `SHRMAX = c * weibull`, `SHT_SL = friction` (Mohr-Coulomb: shear strength `c + mu * compression`), `GCTEN = G_f,tension`, `GCSHR = G_f,shear`, `SHRP = 0` (no plateau) |
| no bond | chunk faces are not connected |
| contact | `/INTER/TYPE7` self contact of each body (all chunk faces, friction = joint friction, constant gap = 0.5 t_c), body-body and impactor-body TYPE7 in both directions (gap 0.2 mm, reduced to half the initial clearance if smaller; friction = scene `contact_friction` or min of the two materials, as in stress-ref; with `contact_restitution` = 1 these contacts are undamped, VIS_S = 1e-6, otherwise the Radioss default 5 % of critical) |
| `support: fixed` | `/BCS 111 111` on every node of the chunk (`pinned` is held the same way and noted) |
| rigid impactor | hexahedral box (or cube-to-ball mapped sphere) of nominal density inside an `/RBODY` (Ikrem=1) whose main node carries the scene mass and inertia; `/INIVEL` |
| body velocity | `/INIVEL/TRA` on the body's nodes |
| gravity | `/GRAV` on all nodes, applied at t = 0 |
| point force | `/CLOAD` on all nodes of the chunk with lumped-mass shares (`f_i = m_i (a + alpha x r_i)`: rigid-body distribution incl. the moment of an off-centre point) |
| pressure | `/PLOAD` on the exposed (not covered by another chunk) element faces of the selected chunks whose outward normal is `face_normal`, `/FUNCT` sampled from the TimeFunction |

### Why LAW169 cohesive elements

Candidates in this build were (source read at the pinned commit):

* `/MAT/LAW169` (ARUP adhesive, cohesive solids `/PROP/TYPE43`): independent tension and
  shear checks — damage starts when `sigma_n >= TENMAX` **or**
  `tau >= SHRMAX - SHT_SL * sigma_n` (the power-law interaction only gates the check, so
  the initiation surface is the tension cutoff plus Mohr-Coulomb shear, exactly the
  reference's max-type criterion) — followed by linear softening in opening / sliding
  displacement to `2 G_c / strength`, i.e. the dissipated energy per area equals the
  fracture energy.  Fully failed integration points are switched off and the element is
  deleted when all four have failed.  **Chosen.**
* `/MAT/LAW116` / `/MAT/LAW117` (mixed-mode cohesive laws, zero-thickness capable): power-law
  mixed-mode interaction and no pressure-dependent shear strength (no Mohr-Coulomb term).
* `/INTER/TYPE2` with rupture or merged nodes with `/FAIL`: no fracture-energy control per
  interface, or no way to keep unbonded faces apart.

LAW169 has two quirks that shape the model:

1. **Finite thickness.**  The engine divides by the geometric element thickness
   (`sigma += d_delta * E_constrained / thick0`), so zero-thickness cohesive elements are
   impossible.  The faces of a chunk that carry a bond are pulled in by `t_c / 2`
   (`t_c = 0.2 x element size` by default, affine mapping of the chunk's node grid) and the
   layer (density of the chunks) fills the gap.  Its modulus is chosen so that the series
   compliance chunk + layer equals the scene bond's `L / E_bond` (equal to the chunk
   modulus when bond and chunk materials match; softer for mortar joints).  A cracked
   joint can close by `t_c / 2` before the self-contact (gap `0.5 t_c`) takes over; with
   the contact gap equal to the full `t_c` the contact acted across intact joints and its
   friction dissipated a lot of energy spuriously (tested on b5 v40), so it is not used.
2. **Unit-length assumptions.**  The starter enforces `G_c >= sigma_max^2 / E` and adds
   `E * area` to the nodal stiffness, i.e. it assumes a layer thickness of one length unit.
   The decks therefore use g / cm / s: the layers are 0.5 - 1 cm thick, the stiffness used
   for the nodal time step is right, and the energy check never alters the scene's G_f (in
   SI it would silently raise the shear fracture energy of b1's rock 7x).  The cohesive
   layer also limits the stable step, which the exporter estimates and caps with `/DTIX`.

### Outputs and post-processing

* `chunk_displacement` / `chunk_velocity`: `/TH/NODE` DX..VZ of all chunk nodes,
  lumped-mass weighted mean.
* `section_force`: OpenRadioss `/SECT` does not accumulate forces of TYPE43 elements
  (`suforc3.F` has no section call), so the crossing bonds' cohesive elements are recorded
  with `/TH/BRIC` (element mean stress, global frame) and integrated over their faces:
  force on the `+normal` side `= -/+ (sigma . n) A`; moments use element centroids
  (the within-element part of a bending moment is not captured — only relevant for b2/b8).
* `reaction`: `/TH/NODE REAC*` is the *accumulated reaction impulse* (`bcs1th.F` adds
  `-A m dt` every cycle); the observer differentiates it over each sample interval.  Only
  nodes on faces of supported chunks that are not glued to another supported chunk are
  recorded.
* `impactor_*`: `/TH/RBODY` X..VZ of the rigid body.
* final chunk state from the last animation frame (`anim_to_vtk`): lumped-mass mean
  velocity and displacement; a bond is intact while at least one of its cohesive
  elements is not deleted; `detached` = not connected through intact bonds to a supported
  chunk (heaviest piece for unsupported bodies); `flags.any_bond_broken` = some bond with
  all its cohesive elements deleted; `flags.collapse` = some chunk of a supported body
  detached; `values.bond_dissipation` = final internal energy of all cohesive layers
  (includes elastic energy still stored in intact layers; exact for b1).
* TH sampling: half the scene's `sample_interval` (or `duration/2000` for scenes without
  one); no `lo/hi` envelopes.

## Validation

**b1_bond_tension** (2 rock cubes, force ramp; expected `f_t A = 10 kN`, `G_f A = 1 J`):

| mesh (elements per chunk edge) | failure force | dissipated | engine time |
|---|---|---|---|
| 1 | 9.997 kN (-0.03 %) | 1.002 J | 3 s |
| **2 (golden)** | **9.655 kN (-3.5 %)** | **1.005 J (+0.5 %)** | 11 s |
| 4 | 9.400 kN (-6.0 %) | 0.978 J | 29 s |

The energy is mesh independent within 2 %; the peak force drops with refinement because
the continuum resolves the stress concentration at the edges of a joint against a fully
held block (Poisson contraction of the pulled chunk), which a rigid-chunk bond cannot have.

**b3_bar_wave** (elastic): wave speed 714.4 m/s (analytic 707.1, +1.0 %: the thin
constrained cohesive layers are slightly stiffer than the uniaxial bar), free-end velocity
ratio 2.002, incident compression -995.6 N, reflected tension 991.6 N — all within 1.6 %
of stress-ref and of the analytic values.

## Limitations

* No crushing of joints or chunks (LAW169 compression is elastic; chunks are elastic), no
  shear cap, no rebar, no strain-rate effects (none of the oracle scenes use DIF), no bond
  dashpots.  Rigid-impactor contact pressures (rho c v: 18 MPa at 2 m/s, 355 MPa at
  40 m/s on concrete) therefore load the joints far harder than in stress-ref, whose
  bonds crush at `f_c` and whose impactor contact is damped: the FE breaks many more bonds
  around a hard impact.
* Mesh sensitivity of the wall scenes is real: b5 v40 is a `hole` with 1 element per chunk
  edge and a base failure (`push_over`) with 2; b5 v10 likewise.  The goldens use 2 (v40,
  v10) and 1 (v02, for run time: 0.3 s of simulated time would take ~22 min with 2).
* Gravity is applied suddenly at t = 0 (no prestress phase): the self-weight stresses
  oscillate around their static value (0 to 2x, < 0.1 MPa) — negligible against the
  strengths, visible in the reaction probes.
* Scene features not exported: scripted events (`remove_chunks`, `remove_supports`),
  ground plane, blast loads, rotated chunks/impactors, crushable impactors, refined
  (level > 0) chunks, rebar.  The exporter raises instead of approximating.
* Weibull seeds: `--seed N` derives the scene with `stress-ref with-seed` (binary from
  `--stress-ref` or `$STRESS_REF`, default `target/release/stress-ref`) and writes
  `opencourant_seedN.json` with `seed = N`.  LAW169 has no per-element strength, so bonds
  are grouped into one cohesive part (material) per distinct strength set: with Weibull
  strengths every bond gets its own part with `TENMAX = f_t * weibull` and
  `SHRMAX = c * weibull` (verified on b9_panel_high_weibull: 838 bonds -> 838 parts).
  The many small element groups make such runs ~2.5x slower (b9: 210 s instead of 82 s).

## Goldens (seed 0) and cost

Single machine, 4 shared cores, `-nt 2` for the larger runs (engine wall time):

| scene | mesh | elements (cohesive) | engine | cycles |
|---|---|---|---|---|
| b1_bond_tension | 2 | 20 (4) | 11 s | 121k |
| b3_bar_wave | 2 | 1,196 (396) | 3 s | 2.3k |
| b5_wall_impact_v02 | **1** | 2,938 (2,018) | 196 s | 109k |
| b5_wall_impact_v10 | 2 | 15,432 (8,072) | 639 s | 50k |
| b5_wall_impact_v40 | 2 | 15,432 (8,072) | 174 s | 21k |
| b6_spall (redesigned, planar plate) | 2 | 37,824 (21,696) | 125 s | 2.1k |
| b7_masonry_v04 / v15 | 2 | 4,580 (1,804) | 23 / 28 s | 12k / 14k |
| b9_panel_low / high | 2 | 6,872 (3,352) | 79 / 82 s | 27k |

Mesh sensitivity seen on the walls (1 vs 2 elements per chunk edge): v40 `hole` (hole
area 1.06 m^2, speed lost 4.93 m/s) vs `push_over` (base joints fail, 5.97 m/s); v10
`hole` (0.66 m^2, 1.31 m/s) vs `push_over` (1.63 m/s).  b7 and b9 agree with stress-ref
on every gated metric.

b6 (steel plate body at 3 m/s on a 0.5 m slab): the back face peaks at 4.83 m/s at 182 us
(1-D analytic 4.94; 3 elements per edge: 4.89 m/s at 178 us).  The reflected tension
cracks the y-joints on several planes (y = 0.25 - 0.45 m, 473 broken bonds) and the back
layer leaves at ~3.0 m/s against ~1.3 m/s for the front, but at 1 ms the cracked planes
are still held by ligaments at the four slab corners, where the release waves from both
lateral free faces keep the tension below f_t.  The connectivity criterion therefore
reports no detached chunk (spall_occurs false), whereas the rigid-chunk reference (no
lateral release) separates whole planes.  The same holds with 3 elements per edge
(474 broken bonds, no detachment).
