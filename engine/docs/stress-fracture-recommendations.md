# Stress-fracture performance: what needs your decision

Every change adopted so far keeps the physics as the oracle defines it: rubble's pipeline
matches PhysX GPU's outcomes (with its rewinds) in 11 of 12 accuracy scenes and is
closer to the oracle in the twelfth, with no frame rewound. The levers below cost
accuracy, change an asset, change the engine's behaviour, or are large enough to need a
go-ahead. Measured effects are cited; unmeasured ones are marked.

## Where the time goes (rubble's pipeline, house-1story rammed)

About 77 % of a destruction frame is the stress solver's explicit substeps inside impact
islands: ~2,900 substeps per 1/60 s frame at ~44 µs each (three latency-bound kernels).
The rest: host splits and readbacks ~7 %, rubble's rigid steps and fragment-body creation
a few percent (more when a fast body needs collision substeps: 14 % on the free-wall ram).
Two alternatives to explicit windows were built and measured and both lose on cost at
equal fidelity: AVBD as the stress solver (8–32× more work on walls,
`stress-oracle/findings/avbd-stress.md`) and an exact modal pre-window (2–30× slower,
`findings/modal-prewindow.md`). So the substep count and the cost per substep are what
remain to cut.

## Levers that change accuracy (measure-then-decide)

1. **Stress step safety 0.9 instead of 0.5.** About 1.6–2× fewer substeps where the bond
   bound sets the step. Not converged: the house ram drifts to 2.62 m/s against 3.04
   (REALTIME_ROADMAP A12 #20). Recommendation: keep 0.5.
2. **Sliver elimination in the assets.** Statically condense the few tiny, stiff chunks
   that set the explicit step: house-1story ×4.5 fewer substeps by removing 16 of 319
   chunks (A3). It changes the model (those chunks follow their neighbours
   quasi-statically) and is a Shardwright/asset decision; it must be gated against the
   oracle with `fidelity.py` first.
3. **A calibrated critical-speed router.** Below a per-material critical impact speed,
   resolve impacts quasi-statically instead of in an explicit window (push-over is right
   there; punch-through starts at 15–20 m/s on b5, IMPROVEMENTS.md). Large savings for
   debris rain and slow pushes; wrong if the calibration misses a case. Not built.
4. **Debris release certificates.** Let fragments that can no longer break anything leave
   the stress solve early (A12 remaining levers). Exact only with a rigorous bound; the
   simple weakest-bond test is not conservative under lever arms (A12 #19).
5. **Stiffness or mass scaling (E × 0.01, selective mass scaling).** Big substep cuts; shifts
   the impulsive/quasi-static boundary (README). Not recommended for fidelity.

## Levers that change engine behaviour

6. **Let coupled rigid bodies sleep.** The coupling keeps rubble's bodies awake so their
   support reactions stay visible to the stress solve; rubble's own sleep would cut idle
   rigid cost in large towns. Needs a rule that a sleeping body's loads are frozen into
   the stress solve (they are unchanged while it sleeps). Not measured yet at scale.
7. **Swept collision (CCD) in rubble's Rust host instead of substeps.** rubble's
   `ccd_classify`/`ccd_sweep` kernels exist but the Rust host never dispatches them, so the
   coupling substeps the whole rigid scene for a fast body (~2.6 ms per rubble step;
   14 % of the free-wall ram). Wiring them changes how fast contacts are resolved (as
   PhysX's CCD does); a conditional-substep shortcut was tried and rejected (it saved
   7 % and was not lossless, because the substep count applies to every body).

## Large engineering steps (lossless, need a go-ahead for scope)

8. **GPU-resident coupling.** Today the host still selects islands, plans contacts per
   segment and splits clusters on a CPU mirror (≈7 % of a destruction frame, more with
   many fractures). Moving routing, contact planning and splits into kernels removes the
   host round trips and makes the pipeline scale with activity only.
9. **Kernel fusion / per-island multirate steps.** The three per-substep kernels are
   latency bound (~15 µs each). A persistent one-threadgroup pipeline was slower before
   (A12 #15); per-island step sizes pay in mixed towns (A5). Both are research items.
10. **Cross-platform determinism of asset bakes.** Bakes are byte-identical across runs and
    thread counts on this Mac; Shardwright's CI does not check `building_lite` across
    OS/arch. Adding it there (and pinning its toolchain) makes rebuilt assets reproducible
    everywhere.
