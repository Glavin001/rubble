# LMGC90 oracle (non-smooth contact dynamics, rigid blocks + cohesive joints)

```
oracles/lmgc90/install.sh                                   # once: build + install pylmgc90
python -I oracles/lmgc90/test_export.py                     # exporter checks (no solver)
python -I oracles/lmgc90/pipeline.py scenes/b7_masonry_v04.json [--seed N]
```

`pipeline.py` runs `export.py` (scene -> `DATBOX/` + `manifest.json`), `run.py`
(chipy time loop -> `raw.npz`, `run.json`) and `observe.py` (-> `stress-observation/1`)
as separate `python -I` processes, single-threaded, and writes
`golden/<scene>/lmgc90[_seedN].json` plus `provenance_lmgc90[_seedN].json`. Scratch
decks live in `/home/user/oracle-runs/lmgc90/<scene>/deck`.

## Model

LMGC90 (2026.rc1) NSCD: Moreau–Jean theta scheme (theta = 0.5, dt = 1e-4 s),
non-linear Gauss–Seidel contact solver (`QM/16` norm, tol 1.666e-4, up to 10 000
iterations per step; unconverged steps are counted in the observation notes).

| Scene | LMGC90 |
|---|---|
| chunk (box) | rigid `RBDY3` with one `POLYR` (8 vertices, 12 triangles), density of the chunk material |
| `support: fixed` | all six velocity DOFs driven to 0 |
| impactor sphere | rigid `RBDY3` with a `SPHER` contactor; density = mass / sphere volume |
| scene bond (mortar joint) | `IQS_EXPO_CZM` interaction on the face-to-face contact points of the two bricks (4 points per joint, `PRPRx_UseCpF2fExplicitDetection`, faces shrunk by 1e-3) |
| unbonded chunk faces | `IQS_CLB` Coulomb contact, mu = min of the two chunk materials' friction |
| impactor contact | `RST_CLB`, normal restitution 0.2, mu = min(impactor, brick) friction = 0 |
| gravity | scene gravity; 0.01 s settling with the impactor invisible (`RBDY3_SetInvisible`), then it is made visible at its scene position with its scene velocity (t = 0) |

### Joint parameter mapping (per bond class; every b7 bond has weibull = 1)

`IQS_EXPO_CZM` = Signorini unilateral contact + mixed-mode cohesive zone with
exponential softening (damage `beta`, 1 intact -> 0 broken) + Coulomb friction.

| our joint (`crates/stress-ref/src/joint.rs`) | LMGC90 parameter |
|---|---|
| normal stiffness `kn = E A / L` | `cn = kn / A` (N/m^3; 1.5e10 head, 3e10 bed, 2e10 half-brick head) |
| shear stiffness `ks = G A / L` | `ct = ks / A` |
| tensile strength `f_t x weibull` | `s1` (0.3 MPa) |
| cohesion `c x weibull` | `s2` (0.4 MPa) |
| `G_f` tension / shear | `G1` = 10 J/m^2, `G2` = 100 J/m^2 |
| friction `mu` (intact: `c + mu sigma`; cracked: Coulomb) | `stfr = dyfr = 0.75` with `tact_behav_SetCZMwithInitialFriction(0)`: friction `(1-beta) mu_d + beta mu_s = mu` is active on the *intact* joint as well, so shear capacity is the cohesive traction plus `mu` times the contact pressure (a Mohr–Coulomb envelope), and the broken joint keeps Coulomb friction |
| compressive strength 8 MPa, buckling | not represented (rigid unilateral contact in compression; never reached here) |
| linear softening (`brittle`) | exponential softening to `eta = 0.01`, same fracture energy |
| tension/bending/shear interaction (`N/A + |M|/S`, max of tension and shear indices) | point-wise: each of the 4 contact points has its own cohesive state; bending appears as different openings of the points; mixed mode by LMGC90's elliptic interpolation of `s1/s2`, `G1/G2` |

LMGC90 assigns contact laws per (candidate colour, antagonist colour) pair, not per
contact. The exporter therefore colours chunks by (support, size, course parity) and
verifies that every colour pair carries exactly one bond class; per-bond strength
scatter (Weibull) is rejected with an error rather than averaged. Bonds between two
fixed chunks are not modelled (both sides driven; LMGC90's cohesive local solver
divides by a zero Delassus term there). `IQS_EXPO_CZM` makes a contact cohesive only
at the first time step (`init_CZM_3D`, `NSTEP == 1`), so contacts that appear later
(cracked bricks re-touching, impactor) are purely frictional.

### Post-processing

A scene bond is intact while at least one of its LMGC90 contact points still has
`beta > 0` (internal variable 5). Chunks connected to a fixed chunk through intact
bonds are attached; all others are detached (`fragment` = connected component).
Chunk velocity = rigid-body velocity at end_time. `ball_velocity` is projected on
the probe axis every step (exact `lo/hi` envelope), sampled every `sample_interval`.

## Limitations

* Rigid blocks: no brick deformation/cracking (matches the reference solver's rigid
  chunks); joint compliance is entirely in the cohesive zone.
* Joint bending is discretised by 4 contact points at the face corners: the
  bending stiffness and the first-crack moment are ~3x those of a linear stress
  distribution (`I_eff = A w^2/4` instead of `A w^2/12`).
* Exponential instead of linear softening; mixed-mode criterion is LMGC90's
  (elliptic in `s1`, `s2`), not our max(tension, shear) index.
* Restitution of the impactor contact (0.2) is not in the scene; chosen equal to the
  reference solver's internal contact constant (`IMPACT_RESTITUTION`) so both
  oracles share the assumption.
* Contacts between chunks of a bonded colour pair that form after cracking use the
  mortar friction (0.75) rather than the brick friction (0.7).
* NSCD is implicit/impulsive: the impact force history is resolved only at dt = 1e-4 s;
  the ball velocity probe is exact at step ends.
* Single scene body, sphere impactor, axis-aligned boxes, no loads/events/ground,
  no rebar (the exporter refuses anything else).

## b7 results and sensitivity (2026-10-09, single thread, ~245 s per scene)

| scene | detached chunks | broken bonds | debris_speed (m/s) | ball_speed_lost (m/s) | NLGS unconverged steps |
|---|---|---|---|---|---|
| b7_masonry_v04 | 1 (struck brick) | 14 / 432 | 3.70 | 4.05 | 0 / 1000 |
| b7_masonry_v15 | 94 | 126 / 432 | 0.68 | 9.51 | 1 / 1000 |

v04 sensitivity (scratch runs, not goldens): restitution 0 instead of 0.2 -> 8 broken
bonds, 1 brick, debris_speed 0.15, ball_speed_lost 3.87; dt = 5e-5 instead of 1e-4 ->
9 broken bonds, 1 brick, debris_speed 0.11, ball_speed_lost 3.92. The punched-brick
outcome and the ball speed are robust; the v04 `debris_speed` (mean over a single
rebounding brick) is not and should not be used as a pass/fail reference.
