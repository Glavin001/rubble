# RoR harness rebuilt mid-run (2026-11-02 20:57)

During the `agora_s/idle` piece, the harness binary (`ror_ref`) was rebuilt to add
`commands`, `wings` and `fusedrag`, all with RoR's own code, for the real repository
cars. So the piece's RoR timings come from two builds.

Agora S has none of these sections. Its node positions are byte-identical between
the two builds (`rest`, `wall`, `free`).

A head-to-head timing, alternating the builds, 3 rounds each, on this piece's own
`scene_idle_64.txt` (64 buses, 60 warm frames, then 240 timed frames of 17 steps):

| round | old build mean / p99 (ms) | new build mean / p99 (ms) |
|---|---|---|
| 1 | 1.938 / 2.212 | 1.816 / 1.904 |
| 2 | 1.833 / 1.906 | 1.832 / 1.922 |
| 3 | 1.832 / 1.900 | 1.808 / 1.880 |

The difference is within run-to-run noise, and the new build is not slower. So
mixing the two builds does not favour Rubble. Raw results are `ab_*.json` here.
