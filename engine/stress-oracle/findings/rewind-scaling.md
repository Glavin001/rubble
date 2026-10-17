# Rewind scaling: when fracture needs corrections, and what they cost

Question (the user's): in which cases do rewinds, internal correction passes and
re-simulation become necessary? Does PhysX with corrections off fail there, reach some
quality at a low correction limit, and solve it at a high limit like 100? And does
rubble's pipeline get there in one pass at any size, because it simulates the impact
through instead of trying and correcting?

Short answer, measured:

- **Corrections are needed whenever the struck body breaks during the contact.** A rigid
  engine solves the contact against the whole body, but the body really exchanges
  momentum only with the piece that breaks free. This is the rule, not the exception, for
  fast impacts on breakable things: a ram into a wall, a projectile through several walls
  or houses, debris into the next structure.
- **The earlier PhysX design (Blast-style correction passes) does not converge with more
  passes.** Each pass re-solves the whole frame, so the work grows with the limit, but the
  outcome plateaus far from the truth.
  - One wall at 40 m/s: limits 0 / 4 / 16 / 100 leave the ram at 10.9 / 10.9 / 12.7 /
    15.6 m/s (truth 35.0), with 23 / 24 / 32 / 141 fragments (truth 6).
  - The frame of impact takes 0.6 / 2.9 / 9.8 / 60 s.
  - Two walls: the ram ends at 4.9 / 4.8 / 5.8 / 7.0 m/s (truth 23.6) for 9 / 25 / 61 /
    194 full-frame solves.
  - So limit 100 does not "solve it". A correction is computed from PhysX's
    impulse against the still-intact rigid body, so more corrections repeat the wrong
    answer more times.
  - The pass count does scale with the number of failure fronts in a frame: more walls
    in one frame need more passes, and each pass costs a whole frame of the whole scene.
- **The oracle's coupled frame (routing before the step, plus verify-and-redo) and
  rubble's pipeline (routing, no redo) reach the oracle's outcome in one pass.** No frame
  was redone in any wall-row run: 1 to 8 walls, 40 to 160 m/s, PhysX or rubble as the
  engine.
  - Routing puts every wall the ram can reach within the frame into the frame's impact
    island before the engine steps, so there is nothing to correct.
  - The redo is a backstop for routing misses. It fired in 2 of the scenes measured:
    once in the striker scene below, and 6 times in b8's sudden column removal.

## The three designs, as the code does them

### A. Blast-style correction passes (the earlier PhysX integration)

`PhysX/destruction/stress-oracle/crates/stress-physx/src/lib.rs`, `PhysxDestruction::step`
(the native pipeline's `internalCorrectionLimit`):

```
checkpoint PhysX motion
loop {
    PhysX.step(dt)                                  // rigid contact vs whole bodies
    trial = stress.clone(); out = trial.step(PhysX contact impulses)
    verdict = bonds broken within one chunk wave-transit of the first break
    if verdict empty or pass == correction_limit { accept trial; break }
    restore PhysX motion; stress.accept_verdict(verdict); install fragments
    checkpoint again; pass += 1                     // re-solve the whole frame
}
```

- **Cost.** Every pass is a whole frame for the whole scene: a PhysX step of every body
  and a stress solve of every structure, on a clone. A frame with `p` corrections costs
  `p + 1` frames.
- **Passes needed.** At least the number of failure fronts in the frame. The front window
  is the wave transit across one chunk (0.1 m of concrete: 28 µs), against a frame of
  16.7 ms. A plug takes many fronts to separate: the free wall first splits at pass 17
  (REALTIME_ROADMAP A10). Every further wall the projectile reaches within the frame
  adds its own fronts. And with `N` walls, each pass also costs about `N` times more.
  So the work for a frame grows like (fronts per frame) × (scene size).
- **Why more passes do not converge.** Each trial's loads are PhysX's contact impulses
  against the current rigid bodies. Until the plug is already free, that is the whole
  wall: a perfectly inelastic whole-wall collision, about 5.8× the real impulse, spread
  over the wall (A10).
  - First-front acceptance takes 1–3 bonds per pass, which separate nothing, so the
    same wrong trial repeats.
  - Accepting the trial's whole fracture (`front_window = ∞`) stops after at most 8
    passes, at a fixed point that is wrong: the wall is shattered by the whole-wall load
    (1 wall: 23.8 m/s, 167 fragments; truth 35.0 m/s, 6).
  - The fixed point of the iteration is not the physical answer, so raising the limit
    cannot reach it.
- **Finer engine steps help, at multiplied cost** (table T3). With 64 PhysX steps per
  frame and limit 16, the ram leaves at 30.2 m/s, but the wall still breaks into 137
  pieces (truth 6), for 637 solves instead of 9. The trial gets closer to the truth only
  as the engine's step approaches the contact's time scale (microseconds). That is the
  step the stress solve takes inside an impact island, but there it is taken once, and
  only for the bodies in the island.

### B. The oracle's coupled frame: routing, then verify-and-redo

`crates/stress-ref/src/world_coupling.rs`:

- `select_islands`: before the engine steps, bounds are swept over the frame. A pair
  whose swept bounds overlap is owned by the stress solve when `rigid_assumption_fails`:
  the Hertz contact time is shorter than the struck body's wave round trip, or the peak
  force exceeds its weakest bond. Owned pairs seed islands, and every body whose swept
  bounds overlap an island joins it (union-find).
- The engine then steps with island bodies frozen. The stress solve simulates the
  islands chunk by chunk at microsecond substeps.
- `step_frame_coupled` then checks: if a body the engine drove fractured during the
  frame, the whole frame is rewound (world snapshot plus `engine.restore()`) and
  simulated again, with the fractured bodies forced into islands
  (`forced.extend(broke)`). The loop ends when no driven body outside `forced` broke.
  Each round adds only the bodies that broke in the previous attempt. A chain of `K`
  misses in one frame, where each attempt reveals the next body breaking, therefore
  costs up to `K` redos, each a full frame.
- In practice `K` stays small. A miss needs a body that is at rest (no sweep) at the
  frame start, is set moving within the frame by a contact the routing calls rigid, and
  then breaks something its frame-start bounds did not overlap. Touching chains are
  already joined by the union. `stress-gpu`'s `GpuWorld::with_engine` runs the same loop
  on the GPU (`redo_fractured_frames = true`).

### C. rubble's pipeline: routing, no redo, promotion

This is `GpuWorld::with_engine` with `stress_rubble::RubbleEngine::with_device` and
`redo_fractured_frames = false`.

- It uses the same routing as B, and never rewinds a frame.
- A fracture under the engine's loads stands as computed, and is counted in
  `unverified_fractures`.
- That structure is promoted: all of its clusters are island seeds from the next frame
  for as long as it keeps breaking (`PROMOTE_FRAMES`).
- The cost of a frame is one pass. The island part grows with the bodies in the island,
  not with the scene.

## Scenes (reproducible)

`crates/stress-rubble/tests/rewind_scenes/mod.rs`:

- **`wall_row(n, speed, gap)`**: `n` free concrete walls, each 1.0 × 0.2 × 1.1 m (10 × 2
  × 11 chunks, 528 kg; the free wall of A10/A12), 0.1 m apart along the ram's path. A
  200 kg steel ram at `speed` starts 5 cm short of the first wall. No gravity, A12's
  real-time configuration (established methods, certified true step, contact safety
  0.35, stress safety 0.5, adaptive, 1/60 s frames), 0.15 s (9 frames).
  - At 40 m/s the ram covers 0.67 m per frame, about 2 walls per frame. At 80 m/s it
    covers about 4, at 160 m/s about 9.
  - The first wall is punched (a plug). Plug and debris then shatter the walls behind:
    the oracle breaks 130 or more pieces from wall 2 on.
- **`striker(speed)`**: a routing miss on purpose. The ram strikes a resting steel bar
  (0.2 × 1.0 × 0.2 m, 314 kg, one chunk, no bonds) whose far end is 1 cm short of one free
  wall.
  - The ram's sweep does not reach the wall, the bar is at rest at the frame start, and a
    bond-less body never makes the rigid assumption fail, so no island is predicted.
  - Within the frame, the engine carries the momentum through the bar into the wall.

Harnesses:

- `crates/stress-rubble/examples/rewind_scaling.rs`: oracle, rubble, rubble with redo.
- The untracked
  `PhysX/destruction/stress-oracle/crates/stress-physx/examples/rewind_scaling.rs`:
  Blast passes with any limit (`--sub` for PhysX steps per frame), whole-trial
  acceptance, the coupled frame with PhysX CPU on the reference, and PhysX GPU on the GPU
  stress solver. Its `wall_row` and `striker` are the same scenes, restated.

Raw results (JSON lines, per-frame times included) are in `findings/rewind-scaling-data/`.
`tests/rewind_scaling.rs` keeps the 2-wall case as a test: no rewind, and the oracle's
outcome.

```
cargo run --release -p stress-rubble --features gpu --example rewind_scaling -- --solver oracle --walls 1,2,4,8 --speed 40
cargo run --release -p stress-rubble --features gpu --example rewind_scaling -- --solver rubble-gpu --walls 1,2,4,8 --speed 40
# PhysX repository, with the GPU SDK:
DYLD_LIBRARY_PATH=$SDK/lib target/physx-gpu/release/examples/rewind_scaling --solver blast --walls 1,2,4,8 --limits 0,1,4,16,64,100
```

## Results

Cells read "ram exit speed (m/s) / fragments". The oracle is the standalone stress-ref
world (CPU f64, chunk-level contact throughout). Blast cells add (full-frame solves over
the 9 frames, worst frame). "Redone" counts whole frames rewound and simulated again.

How to read the times: most runs went concurrently on a 16-core machine (up to ~15 CPU
jobs and one GPU job at a time), so their absolute times are inflated. Compare solve
counts, and times within a column; T5 has the quieter rerun.

#### T1 (40 m/s): ram exit speed / fragments

| walls | oracle (truth) | rubble, no rewind | rubble with redo: redone | PhysX coupled frame (CPU ref): redone | PhysX GPU coupled | Blast-style limit 0 | limit 4 | limit 16 | limit 100 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 35.0 / 6 | 35.1 / 6 (0.3 s worst frame) | 0, 35.1 / 6 | 0, 35.0 / 6 | 35.1 / 6 | 10.9 / 23 (9 solves, 1 s) | 10.9 / 24 (17 solves, 3 s) | 12.7 / 32 (41 solves, 10 s) | 15.6 / 141 (121 solves, 60 s) |
| 2 | 23.6 / 133 | 23.4 / 137 (1.2 s worst frame) | 0, 23.9 / 131 | 0, 23.6 / 133 | 23.4 / 137 | 4.9 / 240 (9 solves, 1 s) | 4.8 / 89 (25 solves, 6 s) | 5.8 / 221 (61 solves, 18 s) | 7.0 / 194 (194 solves, 124 s) |
| 4 | 12.1 / 392 | 11.7 / 407 (3.9 s worst frame) | 0, 11.5 / 403 | 0, 12.0 / 384 | 11.2 / 400 | — | — | — | — |
| 8 | 5.6 / 672 | 6.3 / 662 (7.1 s worst frame) | 0, 5.5 / 675 | 0, 6.5 / 616 | — | — | — | — | — |

#### T1 (80 m/s): ram exit speed / fragments

| walls | oracle (truth) | rubble, no rewind | rubble with redo: redone | PhysX coupled frame (CPU ref): redone | PhysX GPU coupled | Blast-style limit 0 | limit 4 | limit 16 | limit 100 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 70.3 / 3 | 70.2 / 7 (0.3 s worst frame) | 0, 70.2 / 7 | 0, 70.3 / 3 | 70.2 / 7 | 22.3 / 34 (9 solves, 1 s) | 22.2 / 41 (18 solves, 4 s) | 22.3 / 42 (43 solves, 12 s) | 41.2 / 205 (114 solves, 48 s) |
| 2 | 66.0 / 9 | 51.8 / 162 (1.8 s worst frame) | 0, 52.0 / 167 | 0, 66.0 / 9 | 52.0 / 167 | 10.3 / 413 (9 solves, 2 s) | 8.3 / 410 (25 solves, 8 s) | 10.4 / 383 (66 solves, 21 s) | 19.2 / 256 (221 solves, 128 s) |
| 4 | 31.8 / 446 | 28.1 / 474 (7.6 s worst frame) | 0, 27.9 / 471 | 0, 31.8 / 446 | 27.9 / 472 | — | — | — | — |
| 8 | 12.6 / 1096 | 11.1 / 1126 (14.7 s worst frame) | — | 0, 12.4 / 1079 | — | — | — | — | — |

#### T1 (160 m/s): ram exit speed / fragments

| walls | oracle (truth) | rubble, no rewind | rubble with redo: redone | PhysX coupled frame (CPU ref): redone | PhysX GPU coupled | Blast-style limit 0 | limit 4 | limit 16 | limit 100 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 142.0 / 3 | 141.7 / 3 (1.1 s worst frame) | 0, 141.7 / 3 | 0, 142.0 / 3 | 141.7 / 3 | — | — | — | — |
| 2 | 125.3 / 66 | 130.1 / 6 (1.3 s worst frame) | 0, 130.1 / 6 | 0, 125.3 / 66 | 130.1 / 6 | — | — | — | — |
| 4 | 79.4 / 385 | 73.9 / 440 (13.1 s worst frame) | 0, 74.0 / 443 | 0, 79.4 / 385 | 74.0 / 443 | — | — | — | — |
| 8 | 29.2 / 1163 | 39.4 / 1041 (34.6 s worst frame) | — | 0, 29.2 / 1163 | — | — | — | — | — |

#### T2. Blast-style correction limit sweep (first-front acceptance)

| walls | speed | limit | ram (m/s) | fragments | bonds | corrections | full-frame solves | worst frame (s) | total (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 40 | 0 | 10.93 | 23 | 101 | 0 | 9 | 0.6 | 5 |
| 1 | 40 | 1 | 10.91 | 21 | 108 | 2 | 11 | 1.2 | 6 |
| 1 | 40 | 4 | 10.93 | 24 | 106 | 8 | 17 | 2.9 | 9 |
| 1 | 40 | 16 | 12.70 | 32 | 133 | 32 | 41 | 9.8 | 22 |
| 1 | 40 | 64 | 14.74 | 107 | 314 | 69 | 78 | 35.0 | 40 |
| 1 | 40 | 100 | 15.59 | 141 | 382 | 112 | 121 | 60.4 | 70 |
| 2 | 40 | 0 | 4.90 | 240 | 649 | 0 | 9 | 1.0 | 7 |
| 2 | 40 | 1 | 3.77 | 256 | 709 | 4 | 13 | 2.0 | 10 |
| 2 | 40 | 4 | 4.83 | 89 | 303 | 16 | 25 | 5.5 | 26 |
| 2 | 40 | 16 | 5.85 | 221 | 656 | 52 | 61 | 17.6 | 58 |
| 2 | 40 | 64 | 7.08 | 166 | 513 | 147 | 156 | 85.7 | 181 |
| 2 | 40 | 100 | 6.96 | 194 | 570 | 185 | 194 | 124.0 | 210 |
| 1 | 80 | 0 | 22.30 | 34 | 158 | 0 | 9 | 0.9 | 6 |
| 1 | 80 | 4 | 22.21 | 41 | 163 | 9 | 18 | 3.8 | 12 |
| 1 | 80 | 16 | 22.28 | 42 | 169 | 34 | 43 | 11.6 | 25 |
| 1 | 80 | 100 | 41.18 | 205 | 492 | 105 | 114 | 48.2 | 51 |
| 2 | 80 | 0 | 10.29 | 413 | 987 | 0 | 9 | 1.6 | 6 |
| 2 | 80 | 4 | 8.29 | 410 | 984 | 16 | 25 | 7.7 | 21 |
| 2 | 80 | 16 | 10.35 | 383 | 956 | 57 | 66 | 20.5 | 51 |
| 2 | 80 | 100 | 19.15 | 256 | 734 | 212 | 221 | 127.9 | 223 |

#### T2b. Blast-style passes accepting the whole trial fracture (40 m/s)

| walls | sub | limit | ram (m/s) | fragments | bonds | corrections | solves |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 17.91 | 78 | 228 | 2 | 11 |
| 1 | 1 | 4 | 23.68 | 162 | 417 | 4 | 13 |
| 1 | 1 | 16 | 23.81 | 167 | 426 | 8 | 17 |
| 1 | 1 | 64 | 23.81 | 167 | 426 | 8 | 17 |
| 1 | 1 | 100 | 23.81 | 167 | 426 | 8 | 17 |
| 1 | 4 | 1 | 17.58 | 103 | 308 | 2 | 38 |
| 1 | 4 | 4 | 27.66 | 137 | 366 | 5 | 41 |
| 1 | 4 | 16 | 27.68 | 141 | 370 | 8 | 44 |
| 1 | 16 | 1 | 22.82 | 90 | 296 | 4 | 148 |
| 1 | 16 | 4 | 26.56 | 113 | 334 | 11 | 155 |
| 1 | 16 | 16 | 27.19 | 144 | 385 | 13 | 157 |
| 1 | 64 | 1 | 10.50 | 67 | 211 | 7 | 583 |
| 1 | 64 | 4 | 30.01 | 173 | 451 | 17 | 593 |
| 1 | 64 | 16 | 30.13 | 156 | 418 | 28 | 604 |
| 2 | 1 | 1 | 8.61 | 300 | 792 | 3 | 12 |
| 2 | 1 | 4 | 13.32 | 304 | 821 | 8 | 17 |
| 2 | 1 | 16 | 12.47 | 341 | 882 | 15 | 24 |
| 2 | 1 | 64 | 12.47 | 341 | 882 | 15 | 24 |
| 2 | 1 | 100 | 12.47 | 341 | 882 | 15 | 24 |
| 4 | 1 | 1 | 3.77 | 418 | 1116 | 5 | 14 |
| 4 | 1 | 4 | 3.63 | 358 | 1005 | 12 | 21 |
| 4 | 1 | 16 | 3.15 | 473 | 1217 | 24 | 33 |
| 4 | 1 | 64 | 3.15 | 473 | 1217 | 24 | 33 |
| 4 | 1 | 100 | 3.15 | 473 | 1217 | 24 | 33 |
| 8 | 1 | 1 | 2.99 | 386 | 1046 | 5 | 14 |
| 8 | 1 | 4 | 2.43 | 352 | 1001 | 14 | 23 |
| 8 | 1 | 16 | 2.35 | 468 | 1285 | 30 | 39 |
| 8 | 1 | 64 | 2.35 | 468 | 1285 | 30 | 39 |
| 8 | 1 | 100 | 2.35 | 468 | 1285 | 30 | 39 |

#### T3. PhysX steps per frame x correction limit (1 wall, 40 m/s, first-front)

| PhysX steps per frame | limit | ram (m/s) | fragments | bonds | solves |
|---|---|---|---|---|---|
| 1 | 0 | 10.93 | 23 | 101 | 9 |
| 1 | 1 | 10.91 | 21 | 108 | 11 |
| 1 | 4 | 10.93 | 24 | 106 | 17 |
| 1 | 16 | 12.70 | 32 | 133 | 41 |
| 1 | 64 | 14.74 | 107 | 314 | 78 |
| 1 | 100 | 15.59 | 141 | 382 | 121 |
| 4 | 0 | 12.32 | 18 | 121 | 36 |
| 4 | 4 | 12.39 | 19 | 124 | 42 |
| 4 | 16 | 18.91 | 66 | 245 | 68 |
| 16 | 0 | 11.54 | 40 | 185 | 144 |
| 16 | 4 | 13.43 | 63 | 229 | 160 |
| 16 | 16 | 26.34 | 71 | 253 | 189 |
| 64 | 0 | 10.80 | 40 | 170 | 576 |
| 64 | 4 | 25.55 | 164 | 428 | 597 |
| 64 | 16 | 30.19 | 137 | 376 | 637 |

#### T4. Striker (routing miss): ram / fragments / broken bonds

| speed | oracle | PhysX coupled frame (CPU ref): redone | rubble with redo: redone | rubble, no rewind: unverified | Blast limit 0 | limit 4 | limit 16 | limit 100 |
|---|---|---|---|---|---|---|---|---|
| 20 | 5.00 / 10 / 58 | 0, 3.57 / 4 / 33 | 0, -47.81 / 91 / 296 | 0, -47.81 / 88 / 294 | — | — | — | — |
| 40 | 8.63 / 21 / 123 | 1, 8.63 / 21 / 123 | 1, 8.93 / 19 / 120 | 1, -38.70 / 42 / 178 | 8.10 / 10 / 61 (9 solves) | 8.06 / 8 / 55 (13 solves) | 10.70 / 41 / 164 (25 solves) | 13.00 / 107 / 320 (57 solves) |

#### T5. Cost at 40 m/s: median / worst frame (ms); whole 9-frame run (s)

| walls | oracle CPU | PhysX coupled, CPU reference (concurrent run) | PhysX GPU coupled (GPU stress) | rubble (GPU stress, no rewind) | Blast limit 16 (concurrent run) | Blast limit 100 (concurrent run) |
|---|---|---|---|---|---|---|
| 1 | 286 / 313; 2.6 | 302 / 328; 2.7 | 65 / 270; 0.8 | 96 / 259; 1.2 | 471 / 9819; 22.2 | 471 / 60396; 69.8 |
| 2 | 620 / 884; 6.0 | 651 / 899; 6.2 | 315 / 1103; 3.8 | 325 / 1145; 4.0 | 936 / 17602; 57.7 | 816 / 124030; 209.9 |
| 4 | 1810 / 2845; 17.1 | 1736 / 2911; 17.1 | 915 / 3227; 12.3 | 909 / 3258; 12.6 | — | — |
| 8 | 5555 / 6336; 47.5 | 5691 / 6024; 47.0 | 3112 / 5350; 27.9 | 3050 / 4789; 27.8 | — | — |

Notes on the tables:

- **Missing Blast cells.** First-front Blast runs with 4 and 8 walls (40 m/s) and with 4
  walls (80 m/s) had not finished even their limit-0 pass after 15 minutes. With 2 walls
  that pass took 7.5 s. They were stopped, so those cells are empty.
- **T1 rubble cells** come from the concurrent run. T5 is a rerun of the 40 m/s row with
  only three background CPU jobs; its outcomes agree to within 1 m/s and 3 % of the
  fragments.
- **T5's Blast columns** come from the concurrent runs and are slower for that reason.
  Even so, their worst frames (10–124 s) are the passes, not the contention: the same
  run's quiet frames take 0.4–0.9 s.
- **T4 Blast** at 40 m/s: limit 0 (8.10 m/s, 10 fragments) is the closest to the oracle.
  Here PhysX's rigid partner for the wall really is the bar, so the rigid impulse is
  nearly right, and corrections only add fragments (limit 100: 107).

## What the tables show

1. **The work of correction passes grows with the limit; the quality does not follow.**
   - One wall at 40 m/s, limits 0 → 100: 9 → 121 full-frame solves, and a worst frame of
     0.6 → 60 s. Ram 10.9 → 15.6 m/s (truth 35.0); fragments 23 → 141 (truth 6).
   - Two walls: 9 → 194 solves, worst frame 1.0 → 124 s; ram 4.9 → 7.0 m/s (truth 23.6).
   - At 80 m/s, limit 100 gets one wall to 41 m/s (truth 70), at 205 fragments (truth 3).
   - The passes pin at the limit in every frame in which the ram is in a wall: per-frame
     corrections at limit 16 are 16, 16, then 0 for one wall, and 16, 16, 16, 4 for two
     walls. At limit 100, two walls still use 100 in the first frame. More walls keep
     more frames at the limit, and each of those passes re-solves a larger scene
     (`corrections_per_frame` in the data).
2. **Whole-trial acceptance converges in a few passes to a wrong fixed point** (T2b):
   1 wall 23.8 m/s / 167 fragments, 2 walls 12.5 / 341, 4 walls 3.2 / 473. Limits above
   8–24 change nothing.
3. **Finer PhysX steps move the exit speed towards the truth only at multiplied cost, and
   never the fracture** (T3).
4. **Routing before the step removes the need for corrections.**
   - The coupled frame with PhysX on the CPU reference matches the standalone oracle to
     the digit for 1–2 walls at every speed, and for 4 walls at 80 and 160 m/s. At 40
     m/s with 4 walls it gives 12.0 / 384 against 12.1 / 392. It redid 0 frames.
   - rubble (no rewind) and rubble with redo switched on also redo 0 frames.
   - At 40 m/s rubble is within 0.9 m/s and 4 % of the fragments of the oracle for 1–8
     walls.
5. **The pass count is engine-independent: it is set by routing.** The outcome is not
   quite: see the GPU solver gap below.
6. **Cost stays one pass at any size** (T5).
   - rubble's median frame is 96 / 325 / 909 / 3050 ms for 1 / 2 / 4 / 8 walls. The CPU
     oracle takes 286 / 620 / 1810 / 5555 ms.
   - The cost grows with what the islands contain, not with a pass count.
   - PhysX GPU on the same GPU stress solver costs the same as rubble: the islands
     dominate.
   - Blast-style passes multiply the frame of impact instead: 0.5 s → 60 s (1 wall) and
     1 s → 124 s (2 walls) at limit 100.

## Where corrections do occur with routing, and what promotion does (limits of our approach)

- **Routing misses (striker).** The coupled frame with PhysX redid 1 frame at 40 m/s and
  then matched the oracle exactly (8.63 m/s, 21 fragments, 123 bonds). rubble with redo
  also redid 1 frame (8.93 m/s, 19 fragments, 120 bonds).
  - rubble without redo let the wall's fracture under the engine's load stand
    (`unverified_fractures` 1) and promoted the wall from the next frame.
  - Its outcome is wrong, but for a different reason (next bullet): the ram rebounds at
    −38.7 m/s.
  - At 20 m/s the wall did not break within the engine's frame, so no redo fired in any
    design. The coupled frame with PhysX then under-breaks the wall (3.57 m/s, 4
    fragments; oracle 5.00 m/s, 10): a routing miss without a redo is a one-frame
    rigid-contact error.
- **rubble's engine-to-island hand-over of interpenetration (found here, not fixed).**
  - In the striker scene, rubble's engine resolves ram → bar → wall in frame 1 as one
    rigid stack (all about 3.3 m/s at 20 m/s), with the bodies interpenetrating.
  - In frame 2 they become an island, and the stress solve's penalty contact releases
    that overlap explosively. The ram leaves at −47.8 m/s, more kinetic energy than the
    scene started with (`REWIND_SCALING_TRACE=1` prints the per-frame velocities).
  - PhysX in the same coupled frame does not do this (3.57 m/s).
  - So this is the overlap hand-over between rubble's engine and an island, not
    fracture timing. It is a follow-up for the engine, and it also explains why the
    striker scene cannot yet show promotion's own error in isolation.
- **Cascades (b8 sudden column removal, 0.25 s).** rubble with redo redid 6 frames (13
  fragments / 13 bonds). Without redo: 3 unverified fractures, then promotion (10 / 10).
  The oracle gives 11 / 11 and PhysX GPU with redo 14 / 14 (`docs/stress-fracture.md`).
  None of the other 11 accuracy scenes redid a frame. Promotion is exact only from the
  frame after a miss; the missed frame keeps the rigid-load fracture.
- **The GPU stress solver at high speed (engine-independent, not a rewind effect).**
  - At 80 m/s with 2 walls, rubble gives 51.8 m/s / 162 fragments. PhysX GPU on the
    same GPU stress solver gives 52.0 / 167. The CPU reference with PhysX matches the
    oracle (66.0 / 9).
  - At 160 m/s with 2 walls it is the other way round: 130.1 / 6 on the GPU against
    125.3 / 66.
  - So the stress-gpu island solve departs from the reference when walls are crossed
    within a frame at ≥ 80 m/s. Both engines agree, so it is not the engine or the
    absence of rewinds. It is a fidelity item for stress-gpu (these runs used the cost
    model's segment horizons, not `STRESS_GPU_FIXED_HORIZON`).
- **Debris after the islands end** belongs to the engine, so long rows drift a little from
  the oracle: 8 walls at 40 m/s, rubble 6.3 m/s / 662 against 5.6 / 672.

## Answer to the question

- **Cases where rewinds are really necessary.** Any contact in which the struck body
  breaks while the contact lasts: the rigid contact then picks the wrong partner (the
  whole body instead of the piece that breaks off).
  - It gets worse with more breakable bodies crossed within one frame (rows of walls or
    houses, debris into the next structure).
  - It gets worse with speed relative to the frame rate: more fronts per frame.
  - It gets worse with lower strength relative to the impact (more fronts per body).
- **Without corrections, PhysX fails** by perfectly inelastic whole-body impacts: the
  ram stops (10.9 m/s against 35 for one wall, 4.9 against 23.6 for two).
- **With corrections, it does not get "solved at 100" in these cases.** The correction
  re-solves the frame from the same rigid-body loads. The pass count, and the work, grow
  with the failure fronts in the frame, but the outcome converges to the wrong answer:
  either barely moving (first-front acceptance) or a shattered wall (whole-trial
  acceptance).
- **Getting closer needs the engine itself to step at the contact's time scale, with
  passes on top.** 64 PhysX steps per frame, each allowed 16 corrections, got one wall
  to 30 m/s, still 23 times the true fragment count. That took 637 full-scene solves
  for 9 frames, against 9.
  - Every one of those solves covers the whole scene, so for `N` walls the work is about
    (engine steps + passes) × `N` per frame.
  - Getting to the oracle this way is computationally infeasible, as the user expected,
    and it still would not fix the fragments.
- **rubble decides before the step** which contacts the rigid assumption cannot handle,
  and simulates exactly those bodies through the impact at the stress solver's own step.
  It reaches the oracle's outcome in one pass, with 0 redone frames at every N and speed
  measured. Its remaining errors are routing misses (one frame, then promotion), the
  hand-over and GPU-solver items above, and engine-owned debris.
