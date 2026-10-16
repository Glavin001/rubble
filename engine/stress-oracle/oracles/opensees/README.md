# OpenSees oracle (classical elastic frame analysis)

Gives elastic structural-dynamics reference values for the chunked cantilever (`b2_*`)
and for the frame column-removal scenes (`b8_frame_*`). Versions, hashes and license
are in [PINNED.md](PINNED.md).

```bash
oracles/opensees/install.sh                                   # pinned env, hash-checked, idempotent
PY=/home/user/oracle-env/venv/bin/python
$PY -I oracles/opensees/pipeline.py scenes/b8_frame_sudden.json [--seed N]
$PY -I oracles/opensees/test_export.py                        # plain-assert checks (~10 s)
./target/release/stress-ref compare scenes/X.json ours.json golden/X/opensees.json
```

Outputs: `golden/<scene>/opensees[_seedN].json` (compact `stress-observation/1`) and
`provenance_opensees[_seedN].json`, which records the versions, wheel hashes, the
installed `opensees.so` hash, the scene and deck sha256, the command and the wall time.
The deck (`model.json`) and the raw response arrays (`raw.npz`) go to
`$OPENSEES_ORACLE_WORK` or to `oracle-runs/opensees/<scene>/` next to the PhysX checkout.
For `--seed N` the pipeline first re-derives the scene with `stress-ref with-seed`.

## Files

- `export.py`: `export_model(scene) -> dict` is a pure function. It also has
  `build(model, ops)`, which creates the OpenSees domain.
- `run.py`: eigenvalues, the static prestress, Newmark integration and removal events.
  It records the raw element end forces and the node displacement, velocity and reaction.
- `observe.py`: computes probes, the bond failure surface, bodies, flags, values and notes.
- `pipeline.py`: the single entry point.

## Model

- **Nodes:** one 6-DOF node per level-0 chunk, at the chunk centre. Each node carries the
  box's lumped mass and its rotary inertia about the centre. `fixed` holds all 6 DOFs;
  `pinned` holds the translations.
- **Elements:** one `elasticBeamColumn` per bond, from centre a to centre b.
  - `E` and `G = E/(2(1+nu))` come from the bond material, times `stiffness_scale`.
  - `A` is the bond area and `J` is `derived.torsion_constant`.
  - `Iz = derived.i_t1` and `Iy = derived.i_t2`.
  - The element uses `geomTransf Linear` with `vecxz = tangent`, so local z = t1 and
    local y = -t2.
  - The element is Euler-Bernoulli: it has no shear deformation.
- **Loads:** gravity is applied as nodal `m g` (also on held nodes, where it goes into the
  reaction). Point forces become a nodal force plus a moment about the centre. Each
  TimeFunction becomes an exact `Path` series; curved functions are sampled on the
  step grid.
- **Analysis:**
  1. Intact eigenvalues (ARPACK).
  2. Static solve under gravity and the t=0 loads (`gravity_prestress`).
  3. Newmark average acceleration with `dt = sample_interval / 4` (2.5e-4 s here;
     override with `scene.oracle.opensees.substeps`). Probes are sampled every
     `sample_interval`, with lo/hi envelopes over the substeps.
- **Damping:** each element gets its own stiffness-proportional term,
  `beta_e = 2 zeta / sqrt(kn / m_red)`. This is the reference solver's bond dashpot
  (zeta at the bond's own axial frequency), so the global modes are almost undamped,
  as in ours. Rayleigh damping at the first two modes would damp the 3-20 Hz frame
  modes about 100x more and would lower the dynamic peaks; it was not used. Reported
  forces exclude the damping force, as do the failure checks in `joint.rs`.
- **`remove_chunks`:** at the event time the runner first measures the force and moment
  that each element touching a removed chunk exerts on its kept node (`-ops.eleForce`).
  Then it deletes those elements, the removed nodes and their load patterns. The
  measured loads are applied again, ramping linearly to zero over `duration`. With
  `duration = 0` nothing is applied, which matches `ReplacementLoad::factor`.
- **Probes:**
  - `section_force` sums, over the crossing elements, the end force on the `+normal`
    node with its moment about `point`. That equals the force at the bond centroid,
    because no load acts along an element. `normal` is tension positive.
  - `chunk_displacement` is measured from the undeformed position, so the static
    deflection shows at t=0.
  - `reaction` comes from `ops.nodeReaction`, which is the force the support exerts on
    the structure.
- **Failure:** at every analysis step, every element gets the generalised bond force at
  its centroid. The runner evaluates `joint.rs` `stress_measures` and `failure_indices`
  on it, with strengths times `derived.weibull`; DIF and static-fatigue factors are not
  applied.
  - `flags.any_bond_cracked` (and `any_bond_broken`, which is identical here because an
    elastic model cannot tell damage onset from disconnection) is true when any index
    reaches 1 at any time.
  - `values.max_failure_index` and `first_crack_time` (= `first_failure_time`) are
    recorded, and the notes say where.
  - The torsion modulus is `derived.torsion_modulus` when present, otherwise it is
    recomputed from `width` as in `bond.rs`.
  - The analysis stays elastic: nothing detaches, and `collapse` is false by construction.
  - Removed chunks are reported with `removed: true` and zero state.
- **Not supported:** impactors, ground, pressure and blast loads, rebar, rotated
  chunks or bodies, and initial velocities. These raise `UnsupportedScene` rather than
  being dropped.

## Status (2026-10-09)

| scene | OpenSees wall | ours wall | result |
|---|---|---|---|
| b2_cantilever_n10 / b2_cantilever / b2_cantilever_n40 | 0.9 / 1.5 / 2.7 s | 0.25 / 0.6 / 2.2 s | all metrics pass, diff <= 0.15 % |
| b8_frame_gradual | 9.1 s | 5.8 s | all pass, diff <= 0.13 %; both uncracked (max index 0.8990 ours / 0.8993 OpenSees) |
| b8_frame_sudden_elastic | 4.1 s | ~2.4 s | all pass: peaks within 2.3 %, max index 1.722 ours / 1.736 OpenSees |
| b8_frame_sudden (fracturing) | 3.8 s | 2.4 s | `before` values pass; `first_failure` (`any_bond_cracked`) passes, first crack at 0.0639 s ours / 0.064 s OpenSees; `collapse` is report-only (ours true, the elastic oracle false) |

Cross-check of RBSN against beam elements: the static tip deflection of the RBSN chain
is `P h^3/EI (n^3/3 - n/12) + n P h/(G A)`. This closed form reproduces ours exactly and
is within 0.09 / 0.09 / 0.14 % of the OpenSees value `P L^3/3EI` for n = 10 / 20 / 40.

The peak metrics of sudden removal live on `b8_frame_sudden_elastic`. Ours fractures
in `b8_frame_sudden`, and after the first crack the histories cannot be compared with an
elastic oracle. A timestep study (dt = 5e-4, 2.5e-4, 1.25e-4 s) moves the OpenSees peaks
by up to 4 % for `axial_col0` and 2 % for the beam moment, because the peak sits in a
fast axial transient.
