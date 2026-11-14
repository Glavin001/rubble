# Fleet benchmark: many generated sedans

The gate for the vehicle frame's performance work (2026-11-14). `engine/tools/explicit-2khz/fleet.sh`
runs `examples/explicit_vehicles_bench` with the generated sedan (`sedan_new`, JBeam mode
on its pressure wheels: 804 nodes, 6,429 beams, 528 collision triangles), native wgpu on
Metal (M3 Max), GPU time of each 1/60 s World step from submit to completion, 3 s.
The page's own numbers (headless Chrome, raycast wheels: 548 nodes) are in
`lab-mountain-perf.md`. Target: p99 under 16.7 ms.

A run that overlaps another GPU job reads far too high (32 cars apart: 132 ms once,
14 ms repeated on an idle GPU): check the GPU's utilisation first.

## Baseline (`fe3b9074`, before the pair list's reuse)

| Cars | Apart: median / p99 (ms) | Crash town: median / p99 (ms) |
|---|---|---|
| 1 | 11.7 / 12.9 | 14.9 / 15.6 |
| 2 | 11.9 / 13.1 | |
| 4 | 12.1 / 13.2 | 15.1 / 45.5 |
| 8 | 12.7 / 13.8 | 66.6 / 71.0 |
| 16 | 13.0 / 14.5 | 72.6 / 112.9 |
| 24 | 13.5 / 15.0 | |
| 32 | 13.9–14.2 / 15.4–16.3 | fails: more colliders than the coupling's 1,024 shapes |

"Apart": the cars couple with nothing and do not meet (`EV_NO_CONTACT=1 EV_NO_VV=1`).
The World without vehicles is 2.0–2.4 ms of each figure.

What it shows:
- **Separate cars cost one car's latency.** 1 to 32 cars: 11.7 to 14 ms. Each car is its
  own workgroup; the GPU is far from full.
- **Cars in contact are the wall.** In the crash town the cars drive into houses and each
  other: 8 cars take 67 ms, 16 take 73 ms (p99 113). Contact groups step several cars in
  one workgroup.
- 32 cars in the town run out of coupling shapes at step 70 (a capacity error, as designed).
