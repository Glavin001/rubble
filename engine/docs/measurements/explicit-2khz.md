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

## Gather instead of scatter (the brief's step 2): already the design

The beams' forces are not accumulated by atomics. The beam phase writes each beam's
force on each end into that end's incidence entry (`gBeams[incBase + slot]`, the CSR of
beam ends built at upload), and the node phase sums a node's entries in beam order
(`beamSum`): a fixed order, the CPU's, no atomics. The fixed-point atomics (`gAcc`) carry
only contact forces (self-collision hits, vehicle contact, the coupling's reactions),
whose integer sums are order-free by construction. Nothing to change.

## Packing small vehicles (step 5): no idle lanes

Fleets of up to 32 vehicles run 512 lanes a vehicle, larger ones 256. The cars measured
have 470–585 nodes and 1,500–4,300 beams: every phase spreads over all lanes (the beam
phase 3–9 passes a lane), so no lanes are idle to pack another vehicle into.

## Spreading a vehicle over workgroups (step 4): exists as the wide passes

`explicit_beams_wide_*` already run one vehicle across the GPU, a dispatch per phase per
step (6 a step with centred damping, about 200 a frame). They do not run the coupling,
aero, raycast wheels or surface contact, so the town's cars (coupled) cannot use them.
