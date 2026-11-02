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

## Agora S wall: a chaotic crash (CPU reference against RoR)

Corrects an earlier version of this section (cb43a476). A shell quoting error
had dropped the wall, the tyre rolling and the duration from some RoR runs, so
it reported RoR against itself as 24 mm and the CPU against RoR as 5.2 m. Both
figures were wrong.

Scene: RoR's `free` scene, spawned at (189.45, 0.5, 100), moving at −27.78 m/s
along x, into the box (174.06, 3, 101.25) with half extents (0.5, 3, 2.25), drag
off, tyres rolling, 2.5 s.

**RoR against itself, one nudge at a time** (node RMS at 2.5 s, broken beams):

| Nudge | Node RMS | Broken |
|---|---|---|
| x + 2e-5 m | 119 mm | 21 |
| y + 1e-5 m | 864 mm | 19 |
| z + 1e-5 m | 678 mm | 19 |
| vx + 0.1 mm/s | 895 mm | 19 |
| vy + 0.1 mm/s | 1409 mm | 21 |

**The CPU reference against RoR:** 947 mm, 21 broken. That is inside RoR's own
spread.

**Self-collision off on both sides:** the CPU against RoR is 35 mm; RoR against
itself, nudged by +0.1 mm/s, is 74 mm.

**Self-collision behaves like RoR's.** A clean-room probe study ran RoR's
program only, without its source: triangle windings, both approach sides,
contacters in the plane (22 random cases), tunnelling, the contact range, and
the throttle. Rubble's rule matched RoR at every decision point. RoR pushes a
node toward the side it is on now (the sign of δ, with δ = 0 going to the
winding normal), with no memory of the previous side and no vote among
neighbours. A resting table with no contacters is also within RoR's own noise:
0.057 mm against 0.197 mm at 1 s.

**What this means for acceptance item 2:**
- The wall crashes are chaotic. One nudge at the level of float rounding moves
  RoR's own end state by 0.1–1.4 m and changes its broken count between 19 and
  21.
- A limit of 1.5 times a single RoR-against-itself sample is therefore noisy.
  Under the same test, RoR fails against itself with the vy nudge: 1409 mm
  against 1.5 × 895 mm.
- The GPU's 2149 mm in the acceptance run is above every RoR sample above, so it
  needs a distribution, not a single sample: several nudged runs on each engine.

**Commands** (from the repository root; RoR's arguments must be separate
words):

```bash
inspiration/ror-harness/bin/ror_ref <agoras.truck> free pos=189.45,0.5,100 vel=-27.78,0,0 \
  box=174.06,3,101.25,0.5,3,2.25 drag=0 roll=1 secs=2.5 [selfcol=0] out=r.csv
cargo run --release -p engine-softbody --example ror_compare -- free <agoras.truck> r.csv \
  pos=189.45,0.5,100 vel=-27.78,0,0 box=174.06,3,101.25,0.5,3,2.25 drag=0 roll=1 secs=2.5
```

**RoR quirk:** RoR's static boxes do not register at negative coordinates.
Keep probe scenes at positive coordinates.

## Agora S wall: spread over speed nudges (single vehicle)

**GPU, acceptance scene** (`explicit_beams_scaling ... scenarios=impact counts=1
speed=<s>`): Rubble GPU against RoR at the same speed, plus RoR's own
sensitivity sample (RoR at that speed against itself 0.1 mm/s faster).

| speed (m/s) | GPU against RoR | Broken (GPU / RoR) | Travel (GPU / RoR, m) | RoR against itself |
|---|---|---|---|---|
| 27.7798 | 2027 mm | 21 / 19 | 11.03 / 9.07 | 70 mm |
| 27.7799 | 1971 mm | 21 / 19 | 11.04 / 9.15 | 346 mm |
| 27.78 | 2149 mm | 21 / 19 | 10.99 / 8.95 | 639 mm |
| 27.7801 | 663 mm | 19 / 19 | 9.37 / 8.91 | 630 mm |
| 27.7802 | 1788 mm | 20 / 19 | 10.60 / 9.00 | 518 mm |
| 27.7803 | 566 mm | 19 / 19 | 8.96 / 9.01 | 33 mm |

The GPU splits into two outcomes: travel close to RoR's (2 of 6), or about 2 m
further with 20–21 broken beams (4 of 6). RoR's travel stays within 8.9–9.15 m
in all six runs.

**CPU, `free` scene,** each against the RoR base run at −27.78 m/s:

| vx (m/s) | CPU against RoR base | CPU broken | RoR at this speed against RoR base |
|---|---|---|---|
| −27.78 | 947 mm | 21 | 0 mm |
| −27.7801 | 986 mm | 21 | 894 mm |
| −27.7802 | 1425 mm | 19 | 188 mm |
| −27.7803 | 207 mm | 21 | 822 mm |
| −27.7799 | 817 mm | 19 | 827 mm |
| −27.7798 | 1399 mm | 19 | 639 mm |

**Reading:**
- The CPU's spread overlaps RoR's.
- The GPU's spread looks shifted toward the 2 m-further outcome. The GPU and
  CPU agree at the origin (86 mm through this crash) but not about 190 m out,
  so the open suspect is GPU float rounding far from the origin.
- Six samples per engine is too few to settle either point.
