# The explicit 2 kHz solver on the GPU: lossless cost work (perf/explicit-2khz)

M3 Max, shared with other agents. Tools: `engine/tools/explicit-2khz/` (`bounded.sh`,
`ab.sh` interleaved A/B with a bitwise check, `etk_probe.sh`, `verify.sh` the parity
suites, `wg_mem.py` workgroup memory per module). Base: `fix/contact-ub` 1861035c with
`feat/vehicle-sleep` 398e2f85 merged (the hang fix), the same harness.

## Where the frame's time goes (before)

The explicit frame alone (`explicit_frame_probe`, 33 steps, after 60 World steps at
60 Hz, median of 9):

| Vehicle | Frame | Ablation |
|---|---|---|
| ETK K-series (585 nodes, 4,304 beams, 508 triangles; coupling off) | 13.5–14.1 ms | self-collision tests off (`EB_ABLATE` 256): 3.0 ms; walk stopped after the triangle geometry: 3.5 ms; no hit test past the box: 8.7 ms; no duplicate-bucket check: 12.3 ms; no anti-clip crossing test: 12.1 ms |
| Page car (crash town, 100 proxy shapes after the strike) | 2.94 ms | coupling off: 0.72 ms |

So the ETK's frame is two thirds self-collision (the grid walk and its hit tests), and
the page car's three quarters coupling. Neither is the beams' force accumulation.

## Workgroup memory (the brief's step 1)

`wg_mem.py` on the built WGSL: the frame modules declare 29.9–31.0 KB of workgroup
memory, the contact modules up to 31.5 KB, against Metal's 32 KB. A 470–585-node car's
positions and velocities are 15–19 KB. They do not fit beside the grid counts and
offsets (8 KB), the tested-triangle list (8 KB), the pair list (4 KB) and the coupling's
copies (8.3 KB), all live within a step.
