# Testing the destruction solvers

How we find, reproduce and prove fixes, decided 2026-10-12.

Three energy-injection bugs were found by watching 10 s videos and then diagnosed by
hand over hours:

- **The GPU statics certificate** accepted divergence as convergence.
- **The rigid drift fold** rotated about the body origin, not the fold's centre.
- **The established contact's engine handover** let PhysX push apart overlaps the world
  holds at rest.

The guard simulated at most 1.5 s and checked no physical invariant, so all three passed
it. This document is the method we use instead.

## Rules

1. **Check invariants every frame, in every run.** The checks below run in tests, the
   gate, the micro tier and the guard. Production runs can switch them on. A violation
   stops the run at its first bad frame, with a report.
2. **Reproduce in seconds, not frames.**
   - A long run is needed at most once, to find a bug.
   - The bug then becomes a short test: a captured state replayed for one frame (e.g.
     `stress-gpu/tests/data/house_ram_settle_frame62.json`), a micro case, or a
     fixed-horizon scene of a few frames.
   - Capture and replay of the whole world state at the last good frame is planned, so
     every violation yields its fixture automatically.
3. **Test the principle, not the symptom.** A test states a physical property that must
   hold, with a bound derived from the physics, never tuned to pass. Examples:
   - An observable doesn't depend on the arbitrary body origin (metamorphic).
   - A handover to the engine adds no energy.
   - A static solve is "converged" only when the oracle would say so.
   - No chunk is faster than sqrt(2 E_available / m).
4. **A failing test comes first.** For every bug: commit the test, show it failing on the
   current code, then commit the fix and show it passing. The failure stays reproducible
   from history.
5. **Runs must be reproducible.** Tests use `STRESS_GPU_FIXED_HORIZON=1`. The cost
   model's wall-clock-driven horizons are otherwise the only nondeterminism on the GPU
   side. PhysX GPU contact is nondeterministic under load: a known issue, kept out of
   gates.
6. **Time budgets.**

   | Tier | Budget | Examples |
   |---|---|---|
   | Unit and micro tests | < 1 s each | `statics_vs_reference`, `frame_fold`, micro cases |
   | Fixture replays, short scenes | < 10 s | captured settles, `handover_energy` |
   | Guard (`scripts/realtime_guard.sh`) | < 10 min | reference spread, quick gate, micro |
   | Soak (10 s scenes, towns) | background only | never blocks a decision; each failure becomes a fixture |

   Nothing a decision depends on takes longer than about 10 minutes. The oracle under
   layer contact on fractured hulls takes about 9 s per frame and is a soak, never a gate.
7. **Classify every method switch.** Each switch in `stress_ref::scene::Methods` is
   Implemented, Delegated, Inert-without or Rejected on the GPU, enforced by a test over
   `Methods::NAMES`. A switch the GPU does not honour must reject the scene, never run
   silently.

## The invariant checks

These live in one module shared by the oracle `World`, `GpuWorld` (through its f64
mirror) and the stress-physx tests. Every bound is derived; the derivation is in the code.

| Check | Bound | Would have caught |
|---|---|---|
| Finite state | exact: poses, velocities, hidden state, bond forces | bug 1 (inf residual) |
| Energy budget per body | KE ≤ E_available (initial mechanical energy and prestress, plus external work): speed ≤ sqrt(2E/m), height ≤ what E allows | bugs 1, 2, 3 |
| Total energy | KE + PE above ground ≤ E_available. The closure (mechanical + dissipated) is pass/fail only with a derived ε, otherwise a diagnostic | bugs 1, 2, 3 |
| Certificate consistency | a converged static solve has a finite residual at or below its tolerance, re-checked on the host | bug 1 |
| Frame invariance (tests) | translating the body origin or rigidly rotating the scene changes observables only by rounding | bug 2 |
| Momentum (where applicable) | conserved without gravity, ground or engine, apart from impactor exchange | — |

### The report

The first violation reports:

- the frame and the check;
- the chunk or cluster (structure, mass, size, material, cluster size, state);
- its value against the bound;
- what happened to it this frame (split, settle, handover, island);
- its neighbours;
- the energy ledger.

The checks run outside the timed region of `realtime_peak`, so they do not change
PEAK_MS.

## Two-level gating

This uses the quality profiles in [ORACLE_QUALITY.md](ORACLE_QUALITY.md).

- **Level 1, the accuracy cost.** The full f64 oracle against the oracle at a reduced
  quality profile. This measures what each named approximation costs.
- **Level 2, the implementation.** The GPU against the oracle at the matching profile,
  at a tight tolerance. A GPU bug can no longer hide inside "approximation".

Chaotic scenes (fracture, contact) are compared as distributions over ensembles, as
`gate.rs` and `fidelity.py` already do, never trajectory by trajectory.
