# Recurrent multiscale solver prototype results

19 October 2026. Existing work was checkpointed in `6622e05a` before this implementation. This is the first implementation stage of the [architecture research review](learned-solver-research-2026-10-19.md).

The new processor and recurrent execution path work, and controlled experiments support multiscale communication and retained history. **The trained native model remains rejected:** it still over-fractures severely, and no complete real-time solver has been demonstrated.

## Implementation

`multiscale.py` provides stable-ID connected matching, latent restriction/prolongation, local bond processing and explicit caller-owned edge memory. All original physical chunks and output bonds remain present. Hierarchy construction uses current material connectivity, and its exact cache invalidates on changed identities, endpoints, masks or depth.

Memory follows stable bond IDs through ordering and retirement, rejects scene/clock reuse and remains differentiable across training steps. Training uses complete chronological parent episodes with four-frame truncated backpropagation. It never resets recurrent memory between those windows. It does reset at the beginning of each independent episode.

`recurrent_trajectory.py` combines this processor with the previously audited causal feature encoders and complete-response decoder. The candidate has **145,824 parameters**, versus 191,492 in the previous complete model. Legacy attention, message-passing modules and unused endpoint heads are removed. The controls retain the same allocated modules/initialization, with recurrence or hierarchy disabled; unused control parameters are not evidence of equal executed compute.

Native inference carries memory between the model world's own frames. Causal boundary kinematics are now saved under `inference-inputs/`, alongside native snapshots and forecasts, so predictions and memory can be replayed without an oracle. `native_comparison.py` checks matching source graphs, identical initial states, shared clocks, finite states and zero analytical stress fallback before reporting comparisons.

## Capability checks

- A distant-node sensitivity test confirms that the hierarchy transmits information that two local graph passes cannot reach.
- Permuting rows while preserving identities leaves predictions unchanged within numerical tolerance.
- Independent components do not influence each other, including when their hierarchy depths differ. This test exposed and fixed extra coarse updates being applied to short components beside longer ones.
- Material breaks remove structural communication paths. This is specifically a material graph: closed cracks can still carry contact forces and require a separate contact representation.
- Memory survives reordering, drops retired identities and initializes newly encountered identities without inheriting another bond's state.
- A native oracle test reaches the same current geometry through opposite sliding histories and obtains opposite friction forces. Explicit plastic history distinguishes these cases; the test does not establish that learned recurrence is mandatory.
- The analytic wave reference conserves energy and momentum and composes two half-intervals into one full interval to float64 tolerance.

## Controlled wave experiment

A free 32-node linear spring chain supplies exact modal transition targets. Mass is one per node, stiffness is 100, and each prediction advances 0.8 seconds. Training, validation and test use independent initial-state seeds; geometry and material are fixed. Both processors receive 1,000 optimizer updates, width 32 and the same initialization/minibatch schedule within each seed. Checkpoints are selected on validation state error.

The decoder produces paired momentum exchanges and paired displacement corrections. This enforces total linear momentum and center-of-mass evolution for each isolated chain, but does not enforce energy conservation or represent 3-D angular momentum.

| Training seed | Local normalized test MSE | Multiscale normalized test MSE | Reduction |
| --- | ---: | ---: | ---: |
| 0 | 0.275512 | 0.135175 | 50.94% |
| 1 | 0.276897 | 0.147691 | 46.66% |

The ballistic baseline is 0.940481 under the same normalization. Maximum total-momentum errors remain below `2.8e-7` in float32.

**The subsequent energy audit fails.** Mean absolute relative energy changes are about 361% and 360% for the local models, and 92% and 162% for the multiscale models. Energy was not a training or checkpoint-selection criterion. These unchanged-checkpoint results demonstrate better transition approximation, not a validated physical simulator. Independently predicting displacement and velocity corrections can satisfy momentum while violating the energy relationship between them.

## Matched native training experiment

All three arms use the existing verified ten-house native-phase corpus: six training parents, two validation parents and two test parents, each eight frames long. No native physical scenarios were added in this stage. Each arm starts randomly from seed zero, uses width 48, memory width 16, 16 response knots, 60 epochs and **720 optimizer updates**. The complete-response loss and native application contract are unchanged to isolate processor changes. Selection minimizes the common chronological validation objective.

| Processor | Selected epoch | Validation objective | Validation force RMSE N | Test force RMSE N |
| --- | ---: | ---: | ---: | ---: |
| Local, no recurrence | 54 | 0.529762 | 2015.90 | 2523.33 |
| Multiscale, no recurrence | 60 | 0.535928 | 2022.03 | 2336.19 |
| Multiscale, recurrent | 59 | 0.497407 | 1867.39 | 2280.32 |

The recurrent candidate improves validation force error by **7.37%** and test force error by **9.63%** relative to the matched local control. Relative to the multiscale stateless control, test force error improves about 2.39%. This is one native-training seed and a narrow corpus. It is not a matched comparison against the earlier pretrained V31 model, which used a different optimization budget and initialization.

These scores use oracle physical states in chronological order, with learned recurrent memory. They do not measure independent motion.

## Independent native outcomes

Every arm was run on training house00 and validation houses12/16 for eight milliseconds. All nine runs completed with zero oracle resets and zero analytical stress evaluations. All fail structural acceptance.

| Case | Oracle breaks | Recurrent model breaks | Oracle largest mass | Recurrent largest mass | Actor final position error |
| --- | ---: | ---: | ---: | ---: | ---: |
| House00 | 324 | 1258 | 66.17% | 5.97% | 2.753 mm |
| House12 | 181 | 1255 | 88.14% | 5.97% | 2.702 mm |
| House16 | 176 | 1259 | 81.89% | 5.97% | 1.808 mm |

The local and multiscale stateless controls also over-fracture severely. Actor errors vary by case; recurrence improves house12 relative to the local control but worsens houses00/16. Small actor errors over this short horizon still conceal catastrophic structural errors.

This rejects the trained model and the assumption that changing the graph processor alone fixes destruction. It does not identify contact feedback as the sole cause: force forecasting, coupled state/work outputs, objective balance and dataset coverage remain confounded failure paths.

## Runtime and numerical parity

The component benchmark uses width 48, eight hierarchy levels, 10,000 graph nodes and 19,989 edges. These are synthetic graph counts, **not 10,000 active rigid bodies**. It excludes feature construction, the recurrent cell, response heads, engine contacts, native integration and topology application.

| Measurement | Result |
| --- | ---: |
| CPU processor median, one thread, isolated measurement | 32.62 ms |
| Metal processor median, synchronized | 4.35 ms |
| Metal processor maximum of ten samples | 6.33 ms |
| CPU hierarchy construction during Metal benchmark | 24.15 ms |
| Exact cached hierarchy lookup | 0.032 ms |
| Initial device transfer/setup | 22.66 ms |
| Maximum absolute Metal versus CPU latent difference | `9.54e-7` |

The cache helps only when topology is unchanged. All eight recorded oracle frames in the parity check changed topology and required rebuilding. Graph construction during fracture bursts therefore remains a real cost. A 4.35 ms processor call also cannot, by itself, meet real time when called once per 1 ms of physical time. We still need a validated accelerated transition, appropriate feedback cadence and a resident implementation measured as part of the whole engine.

For each trained arm, all eight oracle-input predictions and recurrent states match the frozen training source bit-for-bit after the cache addition. Repeating the recurrent house12 native run with caching preserves all nine snapshots and eight forecasts byte-for-byte. Replaying its saved model-generated inputs reproduces all eight predictions exactly without oracle data.

## Validation and remaining architecture work

Validation completed with **187 Python tests and 13 targeted Rust tests passing**. The additional native run validates the cache and recording change against a complete independent trajectory. No production checkpoint was promoted and no oracle timestep, contacts, solver tolerances or collision shapes were reduced.

The next implementation stage needs to address these specific gaps:

1. **Coupled temporal outputs.** The wave audit shows why momentum structure alone is insufficient. Forces, motion and work need a consistent transition rather than unrelated endpoint corrections.
2. **Live within-interval feedback.** The current native candidate still forecasts once at frame start. Its memory updates once per interval, so it cannot revise that forecast from newly evolving contacts during the interval. A controlled nominal-response/local-feedback experiment is the next comparison, not an assumed fix.
3. **Separate material and contact connectivity.** The current hierarchy uses material connections. Projectile descriptors remain in the existing feature encoders; a complete dynamic contact-manifold graph has not been implemented. Closed cracked interfaces and new fragment contacts must not disappear from the physical information path.
4. **Work representation and objective balance.** The known concentrated initial signed-work transient remains in the old smooth-rate head for this controlled experiment. Complete the initial-transient/cumulative-work representation with exact native accounting before mixing corrective sequences again.
5. **Recurrent corrective data.** Corrective branches need their causal warmup prefix or a verified memory snapshot. The existing isolated branch records cannot simply be shuffled into recurrent training with zero hidden state.
6. **Deployment cost.** Preserve exact caching, then measure topology rebuilding, complete model execution and the actual engine at 10,000+ active bodies. GPU component timings do not establish that target.

## Reproduction artifacts

The local artifact root is `learned-impact/runs/recurrent-multiscale-v36-2026-10-19/`. It contains frozen training sources and hashes, source-dataset audit hashes, configurations, checkpoints, histories, selected metrics, wave checkpoints, independent native snapshots/forecasts/logs, physical audits, latency results and replay parity evidence. Run artifacts remain under the existing gitignored data policy; the code, tests and this result summary are tracked.

See [the compact result record](recurrent-multiscale-v36-2026-10-19.json) for checkpoint hashes and comparison numbers, and [the Python README](../python/README.md) for commands.
