# Kratos DEM oracle (bonded-particle model, DEMApplication 10.4.4)

```
oracles/kratos/install.sh                                   # pinned wheels (requirements.txt, hashes)
python -I oracles/kratos/test_export.py                     # exporter checks (no solver)
python -I oracles/kratos/pipeline.py scenes/b7_masonry_v04.json [--seed N]
```

`pipeline.py` runs `export.py` (scene -> `wallDEM.mdpa`, `MaterialsDEM.json`,
`ProjectParametersDEM.json`, `manifest.json`), `run.py` (DEMAnalysisStage time loop ->
`raw.npz`, `run.json`) and `observe.py` (-> `stress-observation/1`) as separate
`python -I` processes with `OMP_NUM_THREADS=1`, and writes
`golden/<scene>/kratos[_seedN].json` plus `provenance_kratos[_seedN].json`. Scratch
decks: `/home/user/oracle-runs/kratos/<scene>/deck`.

## Model

Kratos DEM continuum strategy (`continuum_sphere_strategy`,
`SphericContinuumParticle3D`, symplectic Euler, particle rotation on, explicit
dt = 3.0e-6 s for b7).

| Scene | Kratos DEM |
|---|---|
| chunk (box) | regular cubic pack of spheres, spacing `s` = half the smallest chunk edge (0.05 m: 16 spheres per brick, 8 per half brick, 2560 in total), radius `s/2`, density scaled so the pack has the chunk's mass |
| inside a chunk | `DEM_parallel_bond_bilinear_damage_Linear`, `IS_UNBREAKABLE`, normal stiffness per area 5x the stiffest joint (`brick_stiffness_ratio`) |
| scene bond (mortar joint) | the particle bonds crossing it (area / s^2 of them: 4 per 0.1x0.1 joint, 8 at the footing), `DEM_parallel_bond_bilinear_damage_Linear` |
| `support: fixed` | all particles of the chunk have fixed velocity and angular velocity (0) |
| impactor | one sphere particle of the scene radius, density = mass / volume, not bonded (COHESIVE_GROUP 0) |
| contacts (unbonded / after failure) | `DEM_D_Linear_classic`, E and nu of the chunk material, friction = joint friction for bonded colour pairs (cracked joint) else min chunk friction, impactor friction = min(steel, brick) = 0, restitution 0.2 |
| gravity | scene gravity; 0.01 s settling with footing fixed and impactor held, then the impactor is released with its scene velocity (t = 0) |

### Joint parameter mapping (per unit joint area; every b7 bond has weibull = 1)

`DEM_parallel_bond_bilinear_damage` (C. Shang, CIMNE): bonded part with tension
cutoff `BOND_SIGMA_MAX`, shear strength `BOND_TAU_ZERO + BOND_INTERNAL_FRICC x sigma`
(sigma = compressive normal stress, coefficient, not angle, in this law), linear
softening whose area is the fracture energy (`FRACTURE_ENERGY_NORMAL/TANGENTIAL`),
damage coupled as max(normal, tangential); unbonded part in parallel (Linear classic
contact + Coulomb friction) keeps compression and friction after failure.

With `A_b = pi r^2` the particle-bond area and `s^2` the joint area each particle
bond represents:

| our joint | Kratos particle bond |
|---|---|
| `f_t x weibull` | `BOND_SIGMA_MAX = f_t s^2 / A_b` |
| `c x weibull` | `BOND_TAU_ZERO = c s^2 / A_b` |
| `mu` | `BOND_INTERNAL_FRICC = mu` (intact), `STATIC/DYNAMIC_FRICTION = mu` (cracked) |
| `G_f` tension / shear | `FRACTURE_ENERGY_* = G_f s^2 / A_b` (energy per particle bond = `G_f s^2`) |
| `kn = E A / L`, `ks = G A / L` | joint bond `kn_j, kt_j` chosen so that the chain brick-centre -> joint -> brick-centre (`L/s - 1` intra-brick bonds in series with the joint bond) has compliance `1/(kn/A)` and `1/(ks/A)` per unit area; `BOND_YOUNG_MODULUS = kn_j s / A_b`, `BOND_KNKS_RATIO = kn_j/kt_j` |
| `damping_ratio` | `DAMPING_GAMMA` (bond viscous damping) |
| compressive strength, buckling | not represented (only the unbonded contact acts in compression beyond the bond) |
| bending/torsion (`M/S` in the tension index) | emerges from the 4 (8) particle bonds per joint; the law's own bond moments do not contribute to failure in this version |

Kratos picks the bond law of a particle pair from the material relation of the two
particles' materials. Chunks are coloured by (support, size, course parity, position
parity in the course) so that relation (i, i) only holds intra-chunk bonds and every
(i, j) carries one joint class; the exporter verifies this, verifies that initial
neighbours exist only inside chunks and across scene bonds, and refuses per-bond
parameter scatter (Weibull) instead of averaging it.

### Post-processing

`ContactMeshOption` keeps one `ParticleContactElement` per initial bond whose
`CONTACT_FAILURE` id is refreshed every step; the runner accumulates "ever failed" at
every sample. A scene bond is intact while at least one of its particle bonds never
failed; chunks connected to a fixed chunk through intact bonds are attached, the
others detached. Chunk velocity/displacement = mass-weighted mean of its particles.

## Limitations

* Bricks are stiff but not rigid (series compliance compensated); they cannot crack.
* 4 particle bonds per joint: crack initiation and bending capacity are
  discretised (no linear stress profile, no `M/S` term in the failure index);
  partially cracked joints count as connected.
* Explicit dt is limited by particle rotation (5e-6 s was unstable, 3e-6 s used:
  ~33 000 steps, ~12 min per scene single-threaded).
* Restitution 0.2 of unbonded/impactor contacts is not in the scene (same value as
  the reference solver's contact constant and the LMGC90 oracle).
* Gravity is applied suddenly at the start of settling (no static pre-solve); the
  residual kinetic energy at release is reported in the notes.
* Kratos FEM-DEM (`KratosMultiphysics-all`) was not installable, so no continuum
  brick fracture model; DEM-continuum is the most faithful available option.
* Single body, sphere impactor, axis-aligned boxes that are multiples of the
  particle size, no loads/events/ground/rebar (the exporter refuses anything else).

## b7 results (2026-10-09, single thread)

| scene | detached chunks | broken / partially cracked joints | debris_speed (m/s) | ball_speed_lost (m/s) | wall time |
|---|---|---|---|---|---|
| b7_masonry_v04 | 121 | 133 / 158 of 432 | 0.067 | 3.88 | 710 s |
| b7_masonry_v15 | 157 (all) | 291 / 88 of 432 | 0.39 | 13.10 | 666 s |
