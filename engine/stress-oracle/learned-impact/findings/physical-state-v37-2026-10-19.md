# Explicit physical-state experiments, V37

The physical graph is the baseline temporal memory. A learned transition should advance chunk deformation/rotation and velocities together with material history; force, stored energy and motion must describe the same transition. Additional learned temporal memory is optional and must earn its cost. The detailed input/output and ownership contract is in [PHYSICAL_STATE_TRANSITION.md](../PHYSICAL_STATE_TRANSITION.md).

These experiments establish an accurate linear elastic baseline and expose missing recorded state. **They do not repair the failed V36 destruction model, establish nonlinear native accuracy, or meet the whole-world 5 ms target. No checkpoint is promoted.**

## What changed and what the measurements mean

| Experiment | Result | Decision |
| --- | --- | --- |
| Suppress first-frame damage growth where a future oracle mask says it should be unchanged, then run seven oracle continuation frames | House00 final breaks 394→399; house12 276→268. The same model forces remain. | A deliberately oracle-assisted diagnostic. Reject a damage gate alone as the next fix; this is not causal inference. |
| Explicit displacement/velocity, generic multiscale network, energy/momentum projection | Two seeds: 32-step normalized state MSE 77,749→0.253 and 4,303→0.272. Energy remains bounded, wave phase remains inaccurate. | Conservation helps stability but does not establish accuracy. Valid only for free undamped elastic chains. |
| Shared learned linear operators for displacement and velocity | Sequence-based validation selects degree 16, 32 stored coefficients. Unseen 32/64/128-node chains: 32-step normalized MSE 4.61e-15–6.07e-15; max relative energy drift 1.89e-7. | Keep as a physical capability baseline. It is linear system identification with known mechanical structure, not a successful nonlinear neural solver or an equal-optimizer-budget comparison. |
| Same coefficients on heterogeneous scalar graphs | New grids, supports, branches and disconnected systems: max relative state energy error 3.36e-14 after 32 steps. | The graph, mass, stiffness and support inputs transfer within the declared spectral/time domain. |
| Constant external forces, changing independently each interval | Four new graph cases: max relative work-balance defect 9.42e-8 after 32 steps. | Forces and work follow the same propagator. Prescribed forces are not live contacts. |
| Native six-DOF graph geometry and full inertia | Deformations, generalized forces and nodal forces match the oracle's own calculations. Four 24-chunk graphs retain max relative state energy error 4.11e-14 over 32 transitions against a dense modal teacher. | Frozen small-motion linear mechanics validated, including rotations, fixed/pinned supports and rotated anisotropic inertia. No damping, fracture or finite corotation yet. |
| Float32 inference on those native linear graphs | Max relative state energy error 5.66e-9; max work-balance defect 1.79e-5 over 32 transitions. | Promising numerical precision within this narrow domain; measured on CPU, not a GPU throughput result. |
| Exact physical chunk/contact recording | All 13 existing files remain byte-identical over two house frames, 298 phases and 365,379 response rows. New sidecars retain raw f64 state and ordered contact histories. | Keep the lossless recording path. Existing models do not yet consume it; this is not a complete World restart. |

Relative state energy error means the mechanical energy of the displacement/velocity **error**, divided by the teacher state's mechanical energy. It is quadratic, not a percent trajectory accuracy or a fracture score. Work-balance defect uses initial mechanical energy plus accumulated absolute external work as its denominator. Normalized state MSE uses training-only scales. Detailed numbers, source hashes and experiment scopes are in [the result summary](physical-state-v37-2026-10-19.json).

## Why the structured baseline works

For frozen elastic mechanics, `M x'' + K x = f`. The graph supplies `K = Bᵀ diag(k) B`; `B` maps chunk translation/rotation into the six bond deformations. Mass and full inertia supply `M`. After mass normalization, one shared operator pair advances both physical fields:

```
x_next = C(L) x + S(L) v
v_next = -L S(L) x + C(L) v
```

`C` and `S` are fitted Chebyshev polynomials, using training transition pairs. The teacher uses eigenvectors; inference uses sparse graph operations and the fitted coefficients, without eigenvectors or oracle calls. Rigid modes are preserved by construction. Constant-force displacement uses `(I-C)/L`, evaluated without dividing by eigenvalues or inverting a matrix. Its zero-mode constraint is enforced during fitting; incompatible coefficients reject forced inference.

This deliberately encodes known physics instead of asking independent neural output heads to rediscover mutually consistent motion, forces and energy. Persisted state is displacement/rotation and velocity, with no hidden temporal vector. Nonlinear material evolution and live contacts have not been learned by this experiment.

## The performance finding is a limitation, not a speedup

Degree 16 still means sequential graph work, and displacement, velocity and forcing require multiple field evaluations. Its declared dimensionless spectral domain permits only about 0.073–0.122 ms per call on the small stiff concrete fixtures. Covering an illustrative 16.67 ms frame would require 137–230 calls using the conservative spectrum bound. The existing 665-chunk/1,299-bond house00 graph permits about 0.569 ms, or 30 calls. These are supported-domain calculations, not certified universal accuracy bounds or lower bounds on all possible architectures.

The final target remains ≤5 ms for the entire world with hundreds of structures, hundreds of thousands of bonds and ≥10,000 active rigid bodies. Hardware and physical duration per budgeted update are still unspecified. No component timing here establishes that target. The next architecture must demonstrate useful accuracy per unit of total computation, including contacts, event timing, topology, state transfer and integration. A larger physical interval cannot be justified by a small parameter count alone.

The repo's previous full modal pre-window also matters: [its accuracy passed and its cost failed](../../findings/modal-prewindow.md). Replacing local steps with dense exact modes is not a newly discovered performance solution.

## A specific physical-memory requirement

The oracle's contact layer retains ordered cohorts of material as contact area grows and shrinks. Two layers can have the same displacement, aggregate stiffness, force and energy, but opposite forces after the newest cohort leaves. A new native test demonstrates this. A few current aggregate response values therefore do not encode this contact law's complete state.

`World::contact_history()` exposes those ordered shear/twist cohorts, normal, indentation, crushing and pair-overlap history without changing the world. `RuntimeGraph::capture_with_physical_history()` additionally retains exact raw chunk state, full inertia, mass, support constraints, identities and flags. `learned_impact_phase_record ... --physical-history` writes optional boundary sidecars; the old snapshots and phase arrays retain their existing meaning. In the two-frame house check, the 2 ms boundary holds 189 contacts, 1,283 shear cohorts and 2,006 twist cohorts. No cohorts are pooled or dropped.

The stress model need not own the engine's contact state if the engine computes contact forces during the coupled advance. It does need an interface that receives the changing interactions causally. The sidecars make the oracle history inspectable and available for the eventual coupled training contract; they do not prescribe duplicating the contact engine inside the network.

## Reproduction and remaining gates

Use the existing Python environment. The scalar fitting CLI chooses capacity using independent 32-step validation sequences:

```sh
python -m learned_impact.wave_operator --force-consistent --out /tmp/new-forced-operator
```

From `engine/stress-oracle`, with the required Rust environment variables set, export each of `free`, `fixed`, `pinned` and `rotated` using:

```sh
cargo run --release -p stress-learned --bin learned_impact_elastic_graph -- \
  --fixture fixed --out /tmp/native-fixed.json
```

Then run `python -m learned_impact.elastic_body_audit --weights /tmp/new-forced-operator/operator.pt --graph /tmp/native-fixed.json --out /tmp/new-native-audit.json`. Outputs must be new paths. `--scene PATH` exports an existing intact, unreinforced native initial graph. Dense modal validation is intended for small graphs; sparse inference does not use that teacher.

199 Python tests and 11 targeted Rust tests pass at this checkpoint. The next gates are damping and nonlinear constitutive consistency, live contact/actor feedback, fracture event timing, coordinate changes at splits, independent long rollouts and complete-engine timing. Full capture/restore/continue parity remains open. New corpus collection must preserve the required state before large-scale retraining; existing frame-start snapshots alone do not contain all of it.
