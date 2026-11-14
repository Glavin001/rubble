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

## After the reused pair list and own-travel boxes (`e356f58a`, 2026-11-14)

Both are in `lab-mountain-perf.md` §5. This bench's cars run the reuse (on by default,
lossless); own-travel boxes are the lab's setting and are off here.

| Cars | Apart: median / p99 (ms), before → after | Crash town: median / p99 (ms), before → after |
|---|---|---|
| 1 | 11.7 / 12.9 → 8.9 / 10.0 | 14.9 / 15.6 → 12.1 / 13.7 |
| 4 | 12.1 / 13.2 → not run | 15.1 / 45.5 → 12.3 / 47.1 |
| 8 | 12.7 / 13.8 → 9.6 / 10.8 | 66.6 / 71.0 → 67.6 / 71.4 |
| 32 | 13.9–14.2 / 15.4–16.3 → 11.4 / 12.5 | fails (coupling shapes), as before |

(The GPU showed 92 % utilisation just before the "after" runs; they are no lower than a
quiet run would be.) Separate cars gained 2.6–2.9 ms each. Cars in contact did not:
contact groups keep the walk every step, and are the next work.

Headless Chrome, one car in the lab (raycast wheels), GPU ms per World step, before → after:
parked 6.9 → 4.5; mountain free fall 7.8 → 5.0, 8–16 s 11.5–12.6 → 9.6–10.1, 20 s
13.8 → 11.1 (60 steps/s until 22 s, was 19 s), 26 s on 17–20 → 18–22 (40 steps/s, unchanged:
the detached nodes in free fall invalidate the list most steps).
