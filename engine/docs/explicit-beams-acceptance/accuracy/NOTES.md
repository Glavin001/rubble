# Accuracy runs: findings (2026-11-02)

Produced by `run.sh` here, one vehicle per run. Every run has 1 trial and no
quiet-machine gate, so the timings in these results are not acceptance timings.

| Vehicle | Idle | Drive | Wall |
|---|---|---|---|
| Sedan | pass | pass | pass |
| Agora S | pass | pass | node RMS fails (2149 mm against a 958 mm limit); beam counts pass |
| Agora L | pass | pass | node RMS fails (195 mm against a 138 mm limit); beam counts pass |
| JBeam car | broken count 3, against RoR's 5 widened to 4–6 | pass | node RMS fails (1355 against 1195 mm); broken 303, against 305–381 |

## Agora S wall: why the GPU misses

**Evidence:**
- **Deterministic:** three repeats gave identical results.
- **Not the solver rules:** on the same scene the CPU reference breaks 19
  beams, as RoR does. The GPU breaks 21 and ends 2.1 m from the CPU, about its
  distance from RoR.
- **Origin matters:** spawned at the origin, the GPU and CPU agree through the
  whole crash: 19 and 19 broken, 86 mm apart after 2.5 s.
- **Rest, by spawn distance** (GPU against CPU, after 0.2 s):

  | Spawn distance | Node RMS |
  |---|---|
  | 0–50 m | 0.1 mm |
  | 64 m and beyond | 1.2–1.5 mm |

  The jump is at the float exponent boundary (64 m) and does not shrink after
  the origin rebase.

**What this shows:**
- Spawning far from the origin rounds the vehicle's geometry. All engines see
  the same rounded data: the GPU is uploaded from the CPU's state.
- The GPU's arithmetic (Metal, compiled through wgpu with default compile
  options, which allow fast math and FMA contraction) amplifies that rounding
  differently from the CPU's.
- The scaling scenes place vehicles about 190 m out (the fleet grid), so the
  comparisons run in that regime.

**Next:** find which GPU operations differ, and make them match without
changing the physics.

## Agora S wall: the source is self-collision (CPU reference against RoR)

Scene: RoR's `free` scene, spawned at (189.45, 0.5, 100), moving at −27.78 m/s
along x, into the box (174.06, 3, 101.25) with half extents (0.5, 3, 2.25), drag
off, tyres rolling. Commands are below.

**Node RMS, CPU against RoR:**

| | 0.6 s (impact) | 1 s | 2.5 s | Broken beams |
|---|---|---|---|---|
| As specified | 10 mm | 266 mm | 5.2 m | 21 against RoR's 21 |
| Self-collision off on both sides (RoR `selfcol=0`) | 0.005 mm | 2.6 mm | 35 mm | 19 against 19 |

RoR against itself, spawn speed 0.1 mm/s higher, with self-collision: 24 mm at
2.5 s. Everything else in the solver matches RoR through the crash, so the gap
is in self-collision.

**Ruled out** (experiment switches tested locally, not kept):
- box face choice and the order of the ground and box contacts. A stiff-cube
  probe matches RoR to within RoR's own noise: 16 mm against 17 mm at 1 s.
- the triangle throttle;
- treating tyre nodes as candidates (530 mm);
- skipping back-side hits;
- always taking the front normal, or always the back normal;
- the self-collision friction surface;
- skipping the resting contacts below.

**Lead:** contacter nodes 60 and 62 lie exactly in the planes of collision
triangles (1, 96, 61) and (63, 97, 79) from spawn. They hit every step with
δ ≈ 0, and Rubble's side test (the sign of δ) flips the contact normal from
step to step. The divergence from RoR grows from these contacts after the bus
lands at about 0.12 s. RoR evidently chooses the contact side differently for
nodes in a triangle's plane. Determine its behaviour with black-box probes (a
triangle and a contacter in its plane, then with neighbours on either side),
then specify and implement it clean-room.

**Commands** (from the repository root):

```bash
inspiration/ror-harness/bin/ror_ref <agoras.truck> free pos=189.45,0.5,100 vel=-27.78,0,0 \
  box=174.06,3,101.25,0.5,3,2.25 drag=0 roll=1 [selfcol=0] secs=2.5 out=r.csv
cargo run --release -p engine-softbody --example ror_compare -- free <agoras.truck> r.csv \
  pos=189.45,0.5,100 vel=-27.78,0,0 box=174.06,3,101.25,0.5,3,2.25 drag=0 roll=1 secs=2.5
```

**RoR quirk:** RoR's static boxes do not register at negative coordinates. A
cube and a bus passed through boxes placed there. Keep probe scenes at positive
coordinates.
