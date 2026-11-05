# BeamNG-quality destructible vehicles: acceptance criteria

The goal (user, 2026-11-05): BeamNG-quality destructible vehicles in Rubble, authored and imported, at GPU scale, from one library on every platform.

## Non-negotiables

- **One library.** Rubble covers rigid bodies, stress-solved destruction, prefracture and soft-body vehicles in one system, usable together.
  - The code is single source in Rust and Slang.
  - Thin clients (JS, C and others) carry no duplicated logic.
  - The same solution runs on WebGPU (browser and native wgpu), Metal and CUDA.
- **Explicit node–beam solver at a fixed 2 kHz,** the same as BeamNG and RoR.
  - No changes to the step, tolerances or contact model.
  - Static contact is node-only.
  - Destruction is forward-only.
- **Clean room.**
  - Learn from BeamNG and RoR public docs, papers and data files only; never read or copy their code.
  - Downloads (papers, PDFs, mods) are allowed into the gitignored `inspiration/` folder, are never committed, and have their licences recorded.
- **Grounded assertions.** Every physical assertion comes from measured data (NHTSA, NCAP, IIHS, FMVSS, published papers) or from exact physics (energy, momentum).
  - No invented thresholds.
  - Results land inside the measured spread, judged on medians over nearby speeds.
- **Workflow.** Work in worktrees and branches, commit and push each verified step, and finish with a PR to main.

## A. Authored showcase vehicle (Rubble's own generator, editable from code)

- **A1. BeamNG-like construction.**
  - Every panel is its own part: doors, bonnet, boot lid, bumpers, fenders, glass panes, mirrors, lamps.
  - Each part is attached only by hinge, latch or bolt hardware modelled as break groups.
  - Seals are damping-only or compression-only.
  - At least 10% of beams are compression-only SUPPORT beams.
  - Part masses are within ±10% of sheet area × gauge × density.
  - Each part has its own collision skin: no collision triangle outlives the joint it spans. The skin is the closed lattice. Each triangle across a shut line breaks (spec §18) when the beams of its edges across the line break, or when its parts' nearest hinge or latch group lets go. A detached part keeps its own triangles. Reason, from the 80 km/h two-vehicle T-bone with contact from every node (striker nodes inside the struck skin, of 1,647; 2026-11-05):

    | Collision skin | Striker nodes inside |
    |---|---|
    | Spanning triangles dropped (gaps up to 0.52 m) | 600+ (single-sim run, file contacters only) |
    | Per-part pieces, 4.5 mm hems, seal strips (closed) | 98–123 |
    | Closed lattice, spanning triangles breakable | at most 6 of 1,141 (no tyre nodes), 0 dead triangles |

    Cutting the skin at every shut line leaves hem pieces and slivers whose neighbours pass each other. The lattice's spanning triangles do not.
  - **The T-bone leak criterion (test `t`, refined 2026-11-05).** At most 1 % of the striker's nodes have penetrated the struck car's *live* skin: a node counts when its net crossings of live collision triangles since spawn, inward less outward, are not zero (`explicit_skin::LeakTracker`, `jbeam-semantics-spec.md` §10.3, fifth pass). A node that passes through the opening left by dead shut-line triangles (§18: the door has come away there) is in a real opening and does not count. This replaces "inside the struck skin by its winding number, dead triangles included", which counted those openings; the test still reports that number beside the new one. With nothing dead the two agree.
  - Several thousand beams in total.
- **A2. Stable.**
  - The rest and post-crash settle tests pass.
  - Momentum closes to rounding.
  - Energy never rises beyond the at-rest chatter.
- **A3. Frontal, 56 km/h NCAP.** All of these sit inside the measured NHTSA spreads in `crash-reference-targets.md`:
  - residual crush and dynamic crush;
  - time of maximum crush;
  - peak g at the rear-seat crossmember, 100 Hz filtered;
  - restitution;
  - Campbell b1 and the 50/56 km/h scaling;
  - door-aperture change;
  - A-pillar and firewall movement.
- **A4. Side.**
  - The NCAP 75° side pole's residual crush is in band.
  - The FMVSS 214 static door crush passes.
  - The FMVSS 214 dynamic rule holds: the struck door may deform but does not separate, and non-struck doors stay closed.
- **A5. Roof.** The FMVSS 216a strength-to-weight ratio is inside the measured spread.
- **A6. Hardware.**
  - Hinge and latch strengths come from measured FMVSS 206 compliance data.
  - Doors stay on in the side pole and in the 56 km/h frontal.
  - A door separates in a severe T-bone (for example car-to-car at 80 km/h, 90°) when its hinge and latch loads exceed their measured strength.
  - Detached parts stay physical: they fall at g and never pass through the ground or the car.
- **A7. Rear body and boot lid.** In the 56 km/h frontal, the boot lid's separation from the deck opening stays within the limit derived from SAE bulb-seal compression.
- **A8. Pole pass-through.**
  - No pass-through at 30 km/h, in both the front and side sweeps.
  - Higher speeds are judged by the section plastic-collapse criterion.
- **A9. Glass.**
  - Glass is brittle: it breaks at the σ/E strain, with no yield.
  - Glass, lamps and mirrors break and detach only on physics events.

## B. JBeam compatibility (community BeamNG vehicles)

- **B1.** `jbeam-semantics-spec.md` documents these, and the solver implements them on the CPU (Rust) and the GPU (Slang):
  - NORMAL, SUPPORT, BOUNDED, ANISOTROPIC and PRESSURED beams;
  - break groups, including several per element and `breakGroupType`;
  - triangle breaking;
  - deform groups;
  - rebound and velocity-split damping;
  - pressure wheels;
  - rails and slidenodes;
  - LBEAMs;
  - precompression and deform limits.
- **B2.** Unused features are bit-identical to before: existing tests are unchanged.
- **B3.** At least 3 full community JBeam vehicles, with flexbodies:
  - load and stand stably;
  - survive a 56 km/h wall crash with plausible, stable deformation and part detachment;
  - have no unsupported physics feature in the census.
- **B4.**
  - Flexbodies (`.dae` meshes) and props render skinned to their node groups.
  - Deform groups switch materials (broken glass, dead lamps).
  - Detached parts carry their visuals.

## C. Visual fidelity (authored and imported)

- **C1.** The visual damage state equals the physics state every frame, checked automatically.
- **C2. Panels show substance.**
  - Inner panels and real sheet thickness show at torn edges.
  - Creases stay sharp.
  - No stair-step or sawtooth tears.
  - No stretched collision ghost sheets.
- **C3. Debris from physics events.**
  - Sparks on scrapes, glass shards on pane breaks, chips on hard impacts.
  - Paint scratches and gloss loss from contact and plastic strain.
- **C4.** The authored car's body passes the zebra highlight check: continuous reflection lines on the bonnet, sides and deck.
- **C5.** Render cost is at most 0.5 ms per car (10 cars, `?perf=1`), with a far LOD.
- **C6.** The user signs off a crash-lab video recorded with `crash-lab-record.mjs`. It shows the frontal, the side pole, a T-bone with door detachment, the drop, and the beam-strain view.

## D. Crash lab (test environment)

- **D1. Controls and views.**
  - One unified scene list: presets plus every scene, with per-scene sliders that persist.
  - Slow motion at exactly 20×, an impact timer, replay and scrub.
  - Beam colouring by strain, damage or part.
  - Skeleton, ghost and body views.
  - Node grab-and-pull.
- **D1. Scenes.** Wall, side pole, FMVSS 214 barrier, car-to-car T-bone, head-on, small overlap, drop and door pull.
- **D2.** `crash-lab-check.mjs` passes for the authored car, the procedural sedan, and each community vehicle in B3.

## E. Scale and platforms

- **E1. Physics GPU time** in the 56 km/h impact scenario, authored medium car, on the M3 Max. These two targets are the user's call:
  - at least 100 vehicles with p99 ≤ 8.33 ms at 120 Hz;
  - at least 256 vehicles at 60 Hz.
- **E2.** Vehicles collide with each other and with prefractured destructible buildings in the same world: one demo scene plus a test.
- **E3. CPU↔GPU parity tests pass on:**
  - WebGPU, native and Chrome;
  - Metal (`vehicle_parity`);
  - CUDA, or the CUDA Slang target builds and runs its parity test where the hardware exists.

## Done

All of A–E is verified by committed tests and scripts on a branch. The docs are updated (`vehicle-authoring.md`, `jbeam-semantics-spec.md`, `crash-reference-targets.md`). A PR to main lists the results. The user has signed off C6.
