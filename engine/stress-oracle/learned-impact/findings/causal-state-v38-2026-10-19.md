# Causal physical-state model V38 — 2026-10-19

Built and trained seven candidates (A–G), with independent native rollouts and no oracle resets. The retained artifacts are [F/F32 and G](../experimental/causal-state-v38/README.md). **No checkpoint is promoted; the efficient learned destruction-solver goal remains incomplete.**

F has 13,895 learned parameters and G has 14,023. Each used 60,000 updates on 1,471,809 sampled rows from five small wall scenarios and one corrective trajectory (4,411,271 raw rows). The network updates first-class material history; the physical graph stores its memory. Native mechanical integration and graph propagation remain numerical. This is a learned material-state component, not a trained coarse graph solver.

F32 results below cover 100 ms of independently evolving native physics. The listed ram error is time-weighted trajectory vector RMSE. The model receives only its own current state. All acceptance runs have zero oracle stress queries and zero engine fallback evaluations.

| Case | Oracle/model breaks | Eventual bond F1 | Ram error | Final chunk RMSE | Full small-fixture gate |
|---|---:|---:|---:|---:|---|
| Original heavy impact (training scene, longer horizon) | 85/93 | 94.4% | 12.39 mm | 4.04 cm | Fail |
| Changed speed (validation) | 87/92 | 96.1% | 1.85 mm | 1.99 cm | Pass |
| Combined offset/strength (validation) | 65/80 | 77.2% | 11.77 mm | 3.49 cm | Fail |
| Light ram rebound (training scene) | 0/0 | 100.0% | 0.00 mm | 0.00 cm | Pass |
| Combined case first unseen by F | 74/96 | 83.5% | 18.44 mm | 6.85 cm | Fail |
| Original heavy impact at 120 Hz | 87/93 | 93.3% | 9.88 mm | 3.64 cm | Fail |

The five primary cases give F/F32 2/5 full passes. G gives 1/5, despite reducing original-case breaks to 87 versus 85 and offset-case breaks to 72 versus 65. G is retained as an objective experiment, not an overall improvement. All five F and G cases match the ram direction and original-wall-plane clearance outcome, but these two outcomes alone are not a realism acceptance gate. The original full wall/house scenes are not validated by these small-wall results.

The fixed research gate requires eventual break F1 ≥0.9, break count within 10%, largest-fragment mass fraction within five percentage points, final chunk RMSE below 2 cm, projectile velocity RMSE below 5% of initial speed, matching direction and wall clearance, and no mechanical energy above initial energy plus 1% in these unforced scenes. These are provisional small-fixture checks, not universal production tolerances. Thresholds were not relaxed to pass a candidate.

The fresh combined case was first tested after F was frozen. Its later G evaluation is reused validation, not another untouched test. The speed and offset cases informed model selection; their labels/snapshots were not used as training inputs.

**What changed and what failed.**

- A used independently regressed force/history outputs. It stayed finite on two heavy impacts but overfractured (105 versus 85/87 bonds).
- B added corrective labels and trained longer. Local fits improved, but the training rollout developed 59.4 m chunk error and the other speed became nonfinite at 0.304 ms. Rejected.
- C replaced exponential residual decoding with linear decoding and added a light impact. Raw-force decoding still broke 103 bonds where the oracle broke zero. Rejected as a runtime candidate.
- A shared energy/force decoder made the light impact stable with no false fractures. Known yield admissibility and irreversible physical history further reduced excess heavy-impact damage.
- D focused the loss on state and added dimensionless physical features. E predicted absolute damage/ductility instead of repeatedly adding small biased increments, improving the original break count from 96 to 88 versus 85.
- F added three training scenes varying mass, speed, location and strength, and removed random couplings from constant training features. This improved one validation gate but did not solve combined-parameter generalization.
- G separated new fracture events from already-complete damage labels and added physical completion margins. Break counts improved in some cases, while trajectory/fragment errors did not consistently improve.

**Evidence that learning matters.** Freezing learned history updates, while retaining the same native mechanics and physical decoder, produces zero breaks versus 85 reference breaks and sends the ram backward. Its ram trajectory error is 489.6 mm. F produces passage with 12.4 mm error in this particular small case. This is an ablation demonstrating causal contribution, not a competitive numerical-solver baseline.

**Validation and data.** 205 Python tests and the Rust force/energy invariant test pass. Native f64 export parity is within 2.28e-13 against independent NumPy arithmetic. FP32 dense layers retain the same break counts and gate classifications in six measured runs; mechanics and normalization remain f64. This is not GPU parity. Recording labels does not perturb the world: all 31 compared snapshots are byte-identical. The new row index was checked for 14,526 unique, ordered native-phase/bond identities, with teacher/reference snapshots identical. Earlier recorded training corpora lack these indices and must be regenerated for sequence-unrolled training. Data hashes, scene definitions and training configuration are saved with the checkpoints.

**Remaining work.** The network is a local constitutive/history component. It does not learn chunk motion propagation or replace the many numerical substeps. At 120 Hz the small-wall rollout still has 22,249 provider calls over 100 ms. Dense inference is currently CPU code. No Metal/CUDA/WebGPU model integration, ≥10,000-active-body test, or ≤5 ms whole-world performance result exists for this candidate. The model also lacks validated rebar, fatigue, rate effects, buckling, varied geometry, blasts and repeated impacts.

The next architectural step must learn a coarse graph state transition, with live-contact response and physical-history supervision across its own evolving trajectories. Sequence/closed-loop objectives are needed for fragment motion; phase regression plus a few corrective recordings is not enough. Retain the numerical solver as the reference and viable fallback while testing that step. Do not equate this trained component with completion of the runtime goal.

Full local experiments: `runs/causal-state-v38-2026-10-19/`. Machine-readable outcomes: [evaluation JSON](causal-state-v38-2026-10-19.json). Portable experimental weights and reproduction commands: [checkpoint directory](../experimental/causal-state-v38/README.md).
