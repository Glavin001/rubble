# Accuracy rule for acceptance item 2 (decided 2026-11-02)

## Why the rule changed

The original rule compared one Rubble run with one RoR run. Its limit was 1.5
times a single RoR-against-itself sample, with the spawn speed 0.1 mm/s higher.
That cannot judge chaotic cases: wall crashes, and a vehicle that breaks beams
under its own weight.
- Float-level nudges move RoR's own end state by 0.1–1.4 m (Agora S wall).
- They change its broken-beam count from 19 to 21 (Agora S wall) and from 9 to 17
  (JBeam car at idle).
- RoR fails the original rule against itself. See
  [accuracy/NOTES.md](accuracy/NOTES.md).

The user asked for the scientifically sound choice. This rule replaces the
single-sample rule for every scenario, chaotic or not. It keeps the original 1.5x
and 1 mm tolerances and the 10% widening of counts.

## The rule (per vehicle and scenario)

1. **Runs:** K = 8 one-vehicle runs per engine. Run k is spawned 0.1 mm/s × k
   faster, at the same place and on the same scene file for both engines.
2. **RoR's spread:** node RMS between every pair of RoR runs (28 pairs). Take
   its median, M_RR.
3. **Rubble against RoR:** node RMS from every Rubble run to every RoR run (64
   pairs). Take its median, M_XR.
4. **Pass on distance:** M_XR ≤ 1.5 · M_RR, or M_XR ≤ 1 mm. For non-chaotic
   cases M_RR is tiny, so the 1 mm floor applies, as before.
5. **Pass on beam counts:** the median broken count of Rubble's runs lies within
   [0.9 · min, 1.1 · max] of RoR's runs, rounded outward. The same applies to the
   bent count.
6. **Fleets:** a fleet's first and last vehicle (trial 0, moved by its spawn
   offset to the one-vehicle spawn) have a median node RMS to the RoR runs within
   the same limit as step 4.
7. **Report:** a bootstrap 95% interval of the ratio M_XR / M_RR (2,000
   resamples of each engine's runs, fixed seed) is reported next to the verdict.

**Why medians and pairwise distances:** if Rubble behaves like a draw from
RoR's own distribution, M_XR ≈ M_RR, so the ratio is about 1. A systematic
error pushes the ratio above 1. Medians resist the occasional far-off chaotic
outcome.

**Implementation:** `explicit_beams_scaling` (`nudges=`, section Accuracy), run
by [accuracy/run.sh](accuracy/run.sh). The earlier single-sample results are kept
in `accuracy/single-sample/`.
