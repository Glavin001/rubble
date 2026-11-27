# How to decide: the owner's rules for physics and engine trade-offs

Written 2026-11-26 from the owner's own words over the continuous-contact work, reviewed by
Fable, so that agents and people make these decisions without asking. **Decide by these rules,
act, and report the decision and its evidence first.** Ask only in the cases listed under "When
to ask".

## Hard rules: never traded

1. **No energy created.** Created energy is a bug, never a trade-off: it runs away and is
   obvious to the eye. A contact pass's change of mechanical energy (kinetic + gravitational +
   stored in the beams, per group, per step) is at most zero to rounding; known-solution tests
   keep at most 1.001 of their energy; bodies at rest gain nothing. A default that creates energy
   is reverted at once to the last one that does not.
2. **Momentum conserved** (linear to rounding, angular within the tests' bar). No truncation,
   freezing or scaling of motion as a failsafe.
3. **No crossings.** Collision meshes never pass through each other (0 crossings, a hard test).
   Contact never holds cars together: no binding that loses momentum, no zone or freeze that
   outlasts the impact. Cars held together by their own bent metal are fine.
4. **No cushions.** Contact keeps the true surfaces apart; it never holds bodies apart by a
   thickness beyond them and never absorbs a crash's energy by design (AGENTS.md, "Contact
   never stands in for metal").
5. **Forward only**: no re-simulating a step (AGENTS.md).

## Graded goals, in order

6. **Real time selects among accurate methods.** It rules out cost multipliers (32× sub-steps,
   3-5× more contact primitives), and it never buys a breach of 1-5 or an undisclosed lossy cut
   (fewer rounds, capped ACCD, a pass every N steps: AGENTS.md). If no method meets both yet,
   the accurate one stays on, slower, the cost is reported, and draws drop before the physics
   changes. Lossless speed-ups first; one with no measured gain is not kept.
7. **Physically accurate and visually realistic, both.** Neither is bought with the other: fit
   the collision skin to the paint, never the paint to the lattice (the approved look stays);
   node count and mass are not the knob, but placement, diagonals and skin fit are free.
8. **Energy removal: numerical residue within limits.** Removal cannot run away (at worst things
   slowly come to rest), so it is judged by what it changes. Over 8 runs:
   - contact's net removal at most 5 % of the pre-impact kinetic energy per crash (fail above),
     target at most 2 %; the elastic bars keep at least 0.98, rebound within 5 % of exact;
   - crush, permanent set and rebound within 5 % of the physics reference (a symmetric head-on's
     rigid-wall equivalent) or the scene's own 8-run spread;
   - the recorded value is a ratchet: it only goes down. Between 2 and 5 %: ship, report, keep
     fixing properly, never patch.

## What "accurate" means

- **Validate against physics, not against our own earlier output**: known solutions (elastic
  bars, conservation laws, the rigid-wall equivalent, published crash ranges such as NHTSA
  restitution 0.13-0.20). A run with crossings is not a reference.
- **Established engineering, not research**: crash codes, non-smooth contact dynamics, BeamNG
  practice; cite the method. BeamNG is a source of established methods and the tie-break between
  options that pass every rule above; where BeamNG and physics differ, physics wins.
- **Meaningful, not just measurable**: quantify an error against a reference and show it
  (video, stills, numbers: crush, permanent set, rebound, beams broken), then judge.
- **Be explicit about compromises**: every report lists each one with its number.
- Measure on the generated sedan, 8 runs a crash.

## Visual against physical

- The collision mesh is the hard bar. The drawn paint may overlap a little while the collision
  meshes are apart, held by a guard: 7 cm during a hit, 3 cm from 0.3 s, over 1 cm at most 0.2 s.
  These are ceilings on today's worst case; the target is millimetres. After each generator fix,
  lower them to the worst of 8 runs plus 5 mm. Never raise them to pass a change.
- The generator is the living rulebook for physically accurate vehicles: change it freely and
  codify the reason with the change.

## How to work

- **Proper fixes over interim patches**: a cap or correction that hides an error is not the
  answer.
- **Measure before deciding**, and decide from the numbers.
- **Ask Fable** (the expert model; the owner asked for it) for an independent design or review on
  hard physics and GPU questions; give it the code and the measurements, not the conclusion.
- Jobs bounded to 5 minutes; commit and push each verified step.

## When to ask the owner

Only when a measured option would: breach or relax a hard rule or raise a guard or tolerance;
adopt a lossy change; change a car's approved look; or meet a conflict this page does not order.
Ask once, with the trade-off table, a recommendation, and what continues meanwhile. Everything
else: decide, act, report first.

## Worked examples from 2026-11-26

| question | decision |
|---|---|
| Hold the full paint thickness in contact (contact absorbs up to 765 kJ)? | No: a cushion (4). |
| Cap restitution at 1.25 and raise the paint guard to 8 cm? | No: it hides a fault in the response's form and raises a guard. |
| e = 1 on the whole response that adds energy to the elastic bars? | Revert at once (1). |
| Crashing cars at a 32× smaller step? | No: a cost multiplier (6). |
| Fit the paint to the node lattice? | No: changes the approved look (7). |
| 3-5× more contact primitives on the visible surface? | No: a cost multiplier (6). |
| Contact removes 18 %: 21 % less crush and cars stuck together against the rigid-wall reference? | Fix: binding (3) and visible removal (8). |
| Paint overlaps a few cm in the hardest crash, collision meshes apart? | Accept within the guard, aim lower. |
| Subcycled penalty keeps 1.017 of the bars' energy? | Energy created: revert (1). |
| Restitution for true impacts? | e = 1 only on a pair newly closing, e = 0 on pressed contact, never above 1, never above the energy before the pass; adopt only if the bars meet rule 8, else e = 0 and report. |
| No method yet both accurate and real time for a hard two-car crash? | The accurate one stays on, slower, reported; lossless speed-ups first. |
