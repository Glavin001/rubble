---
name: physics-evidence-first
description: Decide whether physics behaviour is actually wrong before changing a solver, importer or content, and pick the right reference to judge it against. Use when a crash, deformation, contact or vehicle "looks wrong", when tempted to tune a constant, when comparing to BeamNG / Rigs of Rods / real cars, or when designing verification tests.
---

# Evidence before physics changes

"It looks wrong" is a hypothesis, not a bug report. On this project the ETK was called "too strong" (15 cm of crush at 113 km/h through a brick wall). Measurement showed:
- the coupling was correct;
- the importer matched an independent parse of all 2,767 beams;
- loose 25 kg bricks simply can't push hard enough: about 100 kN against the front's yield of about 600 kN.

Tuning would have broken correct physics.

## Order of trust (strongest first)
1. **Closed forms and maths:** small analytic cases (`engine-softbody/tests/invariants_small.rs`).
2. **Conservation and invariants:**
   - momentum and angular momentum;
   - an energy ledger that never rises (`energy_ledger.rs`);
   - Newton III across every coupling;
   - frame, rebase and timestep invariance.
3. **Implementations agree:** CPU reference against WebGPU, Metal and CUDA, bit-exact or bounded.
4. **Code paths agree:** the same scenario through two code paths must agree. A native static wall against the same wall as an immovable World rigid body; a drop onto native ground against one onto a rigid slab.
5. **Black-box ground truth, on content tuned for that solver:** Rigs of Rods' unmodified solver on Rigs of Rods trucks (`inspiration/ror-harness`, local only; never a build dependency).
6. **BeamNG:** closed source. Only its public JBeam documentation, as a field-by-field spec (`engine/docs/jbeam-semantics-spec.md`), unless someone records BeamNG itself (e.g. BeamNGpy on Windows).
7. **Real-world data, only under matching conditions:** e.g. NHTSA rigid barrier at 56 km/h with a CFC60 filter. A sanity band, not ground truth.
8. **Visual judgement, last.**

## Rules
- **Name the reference before calling something wrong,** and check it's valid for this content and this scene. A brick wall isn't a crash barrier; our own authored car isn't a reference.
- **Content tuned for another solver may behave differently in ours.** JBeam is tuned for BeamNG, and BeamNG ≠ Rigs of Rods. If the maths is sound, fix authoring or conversion at the source, never the solver for one car.
- **Importer changes need a provable mismatch** with the public docs, plus a test derived from the doc text. Report them as a proposal.
- **No lossy changes on your own initiative** (`AGENTS.md`). Measuring a lossy option is fine; adopting one is the user's call.
- **Keep three things apart in reports:** proven facts, measurements and hypotheses.
- **Clean-room:** never read or copy BeamNG or Rigs of Rods source. Write the spec, have someone else implement it, and compare black-box at the end.

The full matrix is `engine/docs/verification-plan.md`; open questions are `engine/docs/handoff/known-bugs.md`.
