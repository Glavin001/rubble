# Physical state transition contract

The learned solver should advance the same physical state that the reference solver advances, using fewer computations. Its primary memory is the persisted physical graph. The baseline has no additional learned temporal memory. An optional latent state must demonstrate a benefit against this baseline and have an explicit initialization and restart contract.

The target is **at most 5 ms for the entire world at a 120 Hz test cadence (8.33 ms simulated time per update)**, with many simultaneous impacts across hundreds of structures, hundreds of thousands of bonds, and at least 10,000 active rigid bodies. The user expects PhysX likely to run at 60 Hz and requested the 120 Hz assumption to leave GPU headroom. Test that integration separately; this requirement does not change production timesteps. A structure, physical chunk, graph node and active rigid body are different counts. The final benchmark must report all of them. Final target hardware remains unspecified; local M3 Max measurements are provisional.

## Transition boundary

`physical state at t + current interactions + known prescribed forcing + interval → physical state at t + interval`

The output state becomes the next input state. Intermediate network activations are scratch computation, not required temporal memory. Network passes need not correspond to physical timesteps.

The stress transition owns the internal deformation and material response. The engine owns rigid motion, contacts, support constraints and topology application. Both sides must exchange compatible impulses and work while the interval evolves. Unknown future oracle contacts, fracture decisions or actor motion are never inference inputs. A frame-end state alone does not define the trajectory or the projectile impulse during the interval.

There are two interface scopes to keep distinct. A stress-only transition can receive current engine-computed nodal forces/torques or impulses; the engine retains its contact memory. A predictor that advances the entire coupled impact through a longer interval also needs a causal way to obtain changing contacts and actor motion during that interval. One current contact-force sample cannot describe an unknown future loading history. The model need not duplicate the engine's contact law or own its history when the engine supplies those evolving interactions.

## Persisted physical state

| Owner | State and units | Oracle source |
| --- | --- | --- |
| Physical chunk | Internal displacement `u` in m, internal rotation `th` in rad, their rates `v` in m/s and `w` in rad/s; active/removed flags; cluster identity; applied inertia scale | `solver.rs::ChunkState` |
| Rigid cluster | Stable identity, member chunks and live interfaces, frame pose, center-of-mass velocity, angular velocity, mass and inertia, support and motion ownership | `solver.rs::Cluster` |
| Material bond | Damage/crush/fatigue, maximum loading indices, ductility initialization, plastic offsets, rebar slip/plasticity/work/rupture, filtered strain rate and its governing-stress history, material mode and connectivity | `joint.rs::JointState` |
| Runtime interface | Current geometry/frame, stiffness, damping and material parameters; alive/source identity; previous total/elastic response where native stepping or transfer uses it | `solver.rs::RtBond` |
| Contact | Contact identity, shear and torsion histories, normal/frame, indentation and crush; the reference layer law also retains ordered material cohorts as contact area changes | `world.rs::ContactKey`, `Stick`, `Layer` |
| External actor | Pose, linear/angular velocity, mass/inertia, material/crush state and motion ownership | `world.rs::Impactor` |
| World and coupling | Physical time, pending prescribed events/loads, frame transfers/replacement loads, contact-filter impulses/pulses and ownership | `World`, `ReplacementLoad`, `api.rs::ContactLoadFilter` |

Static properties and evolving properties should be stored separately. Supports can change through explicit events and must therefore have a defined update path. Stable IDs own histories. Removing a structural connection does not automatically remove its compressive/frictional contact interaction.

The propagating wave state resides in the distribution of deformation and velocity. Stress alone does not identify it. Opposite velocities at the same displacement produce different future waves. Plastic slip and damage are additional physical history; they cannot be reconstructed from the current shape alone.

The oracle's contact law provides a stronger witness: two layers with identical current displacement, stiffness, net force and stored energy can have different ordered material cohorts. When their area shrinks, removing the newest cohort produces opposite subsequent forces. `contact_history.rs` tests this directly. Explicit physical state therefore includes the required history, sometimes variable-length history, rather than only a list of current measured responses.

## Quantities to derive consistently

Forces, stresses, stored energy, impulses and motion must describe one transition. In a linear elastic example, bond extension comes from chunk displacement, force is stiffness times extension, and stored energy is one half stiffness times squared extension. Predicting all three independently creates avoidable contradictions.

For nonlinear material, the corresponding relationship depends on the current internal variables and contact mode. The learned part may predict their evolution or a structured response operator, while known geometry, mass/inertia, incidence and frame transforms remain deterministic. A free elastic system conserves energy; damping, damage and contact require their actual work and dissipation accounts. The energy projection used in the free-chain probe must not be applied to destruction as if energy were constant.

The native signed `overshoot` bookkeeping residual is distinct from nonnegative physical dissipation. Replacing that residual with a clipped value changes the reference's contract. Its meaning and the concentrated initial transient must be handled explicitly.

Fracture is a consequence of the material-state transition and its event timing. A separate fracture classification score remains useful for diagnosis, but cannot substitute for correct motion or material state.

## Current recording gaps

The native phase corpus records f64 chunk deformation/rate and applied loads at every phase, full bond history/response and work, and frame snapshots for visualization. It does not yet provide a complete self-contained physical-state restart. The frame bridge contains legacy float32 chunk features.

The optional `learned_impact_phase_record ... --physical-history` path now adds `physical-history/000000.json` and a file at each subsequent boundary. It retains raw f64 chunk displacement/rotation/velocity, full inertia, mass, supports, identity and flags, plus ordered contact shear/twist cohorts, indentation/crushing and pair-overlap history. Version `native-physical-history/2` additionally records every captured bond's current reference frame/levers, nominal stiffness and damping, and each cluster's actual mass-damping coefficient and inertia-relief rule. Nominal mechanics still require the nonlinear joint law and its history; they are not the current effective tangent. This closes an observed alias: the legacy model features can be identical when runtime damping differs. No missing damping is implicitly zero-filled. Joint history and rigid poses remain in the companion snapshot. Capture is read-only; the existing learned model does not consume these new files yet. Earlier datasets cannot recover this missing contact history from their old snapshots alone; deterministic replay can collect it. This addition is still not a complete World restart or a claim that all necessary coupling/scheduling state is exposed.

Existing oracle branches reconstruct complete native state by deterministic replay of the scene and saved predictions, then verify another continuation. This is a valid reconstruction mechanism for those branches, but does not prove that the current graph input alone is a sufficient physical state.

Before bulk collection for the new transition, add a versioned state contract and verify: capture/restore/continue agrees with uninterrupted execution; two physically different histories are distinguishable; stable IDs and coordinate transforms survive splits; contact histories survive restart; and the model never reads future phase observations as current inputs. Scratch buffers, caches and numerical integration history must be distinguished from physical state. Restarting the same numerical integrator can require additional numerical state even when a different integrator does not.

## Experiments and acceptance

The first explicit-state probes persist only displacement and velocity because these are a complete state for their free linear dynamics. Forces and energy are derived. They are capability tests, not a full implementation of this contract.

- A generic multiscale network with a per-island energy projection remains bounded over 32 transitions, but still loses wave phase. Conservation does not establish accuracy.
- A shared pair of learned propagation operators advances displacement and velocity together. Coefficients are fitted on training transitions and used unchanged on unseen states. New chain lengths, heterogeneous elastic graphs and fixed supports test representation and generalization within the declared spectral/time domain.
- Nonlinear material history, live contact, fracture, 3-D rotation, coordinate transfer and actual engine performance remain separate acceptance stages.

The six-DOF extension now checks native initial geometry, translation/rotation coupling and full mass/inertia against the oracle's own bond calculations. Its trajectories use a frozen linear system and an independent dense modal teacher. This validates small-motion mechanics, not finite corotational rotation or nonlinear native destruction. See `findings/physical-state-v37-2026-10-19.md` for measured results and the unresolved computation cost.

The damping extension uses separate physical stiffness and damping graph operators, including the native mass dashpot where its frozen linear form is valid. The operators generally do not commute; treating damping as a scalar fade or deleting modal cross-coupling is an approximation. Dense small-graph reference transitions and an independent RK4 check validate this capability, not a fast learned inference implementation. The persisted state is still physical displacement and velocity, with no additional temporal latent.

At the requested 8.33 ms interval, native damping can make the endpoint map much more compressible than its early response. A measured heterogeneous fixture needs all 216 mechanical state directions at 0.5 ms for the tested 5% worst-case state-amplitude bound, while its 8.33 ms endpoint needs 30 directions for a 1% bound. These are per-graph linear diagnostics, not a shared-network capacity guarantee. No modes are discarded in the solver. Keep a detailed local transient path and assess intermediate force peaks, work and material history alongside endpoints; eventual vibration decay does not undo earlier fracture.

Selection should use independent sequences, not only one-step error. Measure physical state error, wave phase, momentum with external reactions, energy/work balance, first damage and fracture timing, fragment masses, and actor penetration/rebound/trajectory. The final native evaluation needs a fresh holdout, meaningful post-impact duration and the entire-world timing budget. No component result alone qualifies the model for deployment.
