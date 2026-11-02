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
