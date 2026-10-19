# V38 trained physical-state component — experimental

These are actual trained neural checkpoints. **None is a promoted destruction solver, and none demonstrates a runtime speedup.** This experiment establishes a causal learned material-state update inside the native reference world's independent contact/motion simulation. Graph propagation and native integration substeps remain numerical.

`model-f.json` has 13,895 parameters. `model-f-f32.json` has the same trained weights and explicitly uses FP32 dense-layer arithmetic; feature preparation, normalization, physical decoding and native mechanics remain f64. `model-g.json` has 14,023 parameters and a revised new-fracture-event objective. G reduces some excess fractures but does not consistently improve motion. Both are retained for comparison, not selected as production replacements. The weights came from CPU PyTorch training, 60,000 AdamW updates, seed 0, batch 2,048. Dataset and artifact hashes are in `provenance.json`.

**Architecture and state.** The model is a bond-local MLP, F: 79→64→64→71, G: 81→64→64→71, with ReLU hidden layers. It is not a trained graph transformer or a learned coarse-time propagator. Inputs are current bond deformation and rate, physical joint history, actual stiffness/damping, material/geometry coefficients, dt, and dimensionless loading/energy features. Neither frame index, bond identity, future loads nor oracle frames are inference features.

Learned outputs update damage, crushing, ductility, sliding/torsional plastic offsets, dissipation and discrete transition decisions. These physical values persist on the actual bonds and become the next inputs. There is no hidden learned memory vector. A fixed physical decoder derives force and stored energy from the same predicted history. It retains the elastic law, exact compressed-patch geometry, known yield admissibility, irreversible history maxima, stress-rate filtering, damping and native integration. This is a hybrid material-state surrogate; the expensive propagation-through-time problem is not replaced here.

Supported experimental scope: brittle softening with exact cracked-patch contact, patch-after-damage and exact rate filtering; no rebar, dynamic increase factor, static fatigue or member buckling. Unsupported laws fail explicitly. Use `energy` mode with these checkpoints: unused raw-force heads were not trained and are rejected as a runtime decoder.

**Data.** Five oracle wall scenarios plus one corrective model trajectory supply 4,411,271 phase rows. Training samples 1,471,809 rows, with a maximum of 300,000 per recording. The five scenes vary ram speed, mass, horizontal impact position and strength. All use a small 64-chunk, 112-bond wall. Validation changes speed and combined impact/material parameters. These recordings do not constitute geometry/material/use-case diversity sufficient for deployment.

Each `rows.f64` row stores 69 current input values, 42 physical baseline values, 42 oracle target values and 29 transition labels. Unset ductility has explicit presence bits. Snapshot files save each frame's poses, velocities, physical bond history, connectivity, forces, energies and projectile state for independent evaluation and visualization. The final recorder also writes `rows.index.u64`: native substep, structure and bond identity. Earlier A–G training recordings predate that index; regenerate them before sequence-unrolled training rather than inventing row identities. Current training is phase-supervised with corrective on-policy examples, not end-to-end rollout differentiation.

**Verification.** The force decoder matches the reference response at the same physical state and passes a finite-difference energy-gradient test. Exported f64 forward inference agrees with an independent NumPy implementation within 2.28e-13. FP32 dense inference was separately exercised in six independent runs; it retains the reported break counts and pass/fail classifications. This is not GPU-backend parity. All 205 Python tests and the Rust physical decoder test pass. Adding observational training labels leaves all 31 compared trajectory snapshots byte-identical; indexed teacher/reference recording also leaves snapshots unchanged.

**Results and limits.** Read `evaluation.json` or `../../findings/causal-state-v38-2026-10-19.md`. F passes 2 of 5 small-fixture accuracy gates; G passes 1 of 5. Heavy ram passage and light ram rebound agree with the oracle across those cases, but fragment motion and changed impact/material combinations still fail. A frozen-history ablation produces zero fractures and ram rebound where the trained model produces passage. Thus the learned state updates materially affect the outcome, but this is not sufficient validation of realism.

At a 120 Hz outer cadence, the tested small wall still calls the stress provider 22,249 times over 100 ms. No native substeps were removed. There is no Metal/CUDA/WebGPU implementation or whole-world benchmark for this model. The ≤5 ms target for 10,000+ active rigid bodies and hundreds of thousands of bonds is **not achieved**. Chunk count is not active rigid-body count.

Run from the repository root, with the existing Python environment installed:

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo build --release -p stress-learned --bin learned_impact_causal --manifest-path engine/stress-oracle/Cargo.toml
CAUSAL_RUN=$(mktemp -d /tmp/causal-state.XXXXXX)
engine/stress-oracle/target/release/learned_impact_causal fixture "$CAUSAL_RUN/scene.json" 16 50 0 1
engine/stress-oracle/target/release/learned_impact_causal reference "$CAUSAL_RUN/scene.json" "$CAUSAL_RUN/oracle" 100
engine/stress-oracle/learned-impact/python/.venv/bin/python -m learned_impact.causal_evaluate \
  --binary engine/stress-oracle/target/release/learned_impact_causal \
  --scene "$CAUSAL_RUN/scene.json" --oracle "$CAUSAL_RUN/oracle" \
  --weights engine/stress-oracle/learned-impact/experimental/causal-state-v38/model-f-f32.json \
  --out "$CAUSAL_RUN/model" --frames 100
```

Use `teacher SCENE NEW_OUTPUT FRAMES` to record new training labels. `collect-energy SCENE NEW_OUTPUT FRAMES WEIGHTS` collects oracle labels on a model's own states without applying those labels. Such collection is explicitly marked as training, records nonzero oracle-query counts, and does not produce an acceptance plan. `reference` saves oracle trajectories without the large training tensor. Learned acceptance runs read only a scene and checkpoint, start identically, and use their own subsequent contacts, motion and bond history.

The next required work is sequence/closed-loop physical-state training and a learned coarse graph transition that reduces integration work while retaining live contact response. More local regression epochs or a faster implementation of this MLP alone do not establish the requested efficient stress solver.
