# P0 feasibility of the learned impact-outcome solver (2026-10-20)

Phase P0 of `docs/learned-stress-solver/learned-stress-solver-plan.md`: no learning, only the
tests that could kill or redirect the plan before any training. Every number below comes from
runs in this worktree on the M3 Max (one GPU job at a time; CPU ensembles in parallel). The
machine-readable summary is [`p0-feasibility-2026-10-20.json`](p0-feasibility-2026-10-20.json);
the GPU probe's raw lines are in [`p0-data/`](p0-data/). How to reproduce is at the end.

## Gates at a glance

| Step | Gate | Verdict | Evidence (detail below) |
|---|---|---|---|
| 0 | Foundation fixes ported; no digest moves | **PASS** | Guard 32/32 identical; stress-ref, stress-gpu (fixed horizon, established), stress-rubble (gpu) suites all pass. No re-recording needed |
| 0 | Ledger fit for an energy cap / label | **FAIL as a cap, usable with care** | Balance residual +0.3% to +4.7% E0 (energy gained, penalty-contact bookkeeping); the fracture floor ΣG·A is 0.1–0.5% E0, below that residual; negative overshoot exists but is ≈1e-6 E0 |
| 1 | Pinned oracle reproduces 35.0/6 and 23.6/133 | **PASS**, with a caveat | 35.034 m/s / 6 / 31 bonds; 23.622 / 133 / 401. **Both are knife-edge symmetric cases**: ±1 mm moves them to 32.0/78 and 22.9/200 |
| 1 | GPU explicit window agrees on the free wall; wall-row gap explained | **PASS / explained** | Full-feature routed GPU passes the free-wall ensemble; the 31.65/41 gap is a routing miss (≈ 4.3 m/s, 51 fragments) plus `blast_parity_routed`'s material switches (≈ 3.1 m/s, 32 fragments); f32 adds ≈ 0.3 m/s |
| 2 | Chaos floor measured | **DONE** | 16-member ensembles × 2 perturbation sets × 7 walls (224 runs) + late-scene ensembles |
| 3 | Best boundary variant inside the chaos spread on both M5 scenes and beats the punch | **PASS** (window-end poses, reduced state, 1/60, contact filtered) | Passes free wall and wall row (both scene timings); the punch fails every scene (skill 1.8–7.9 on ram, 4 vs 77 fragments) |
| 4 | Some ≥ 8-round config ≤ 2 ms at 10k chunks | **FAIL as measured** | Best: Metal 3.0 ms (W32 f16, 8 rounds, tiled), Chrome WebGPU 5.3 ms. M5-sized islands (220 chunks) take 0.95–1.3 ms (Metal) / 1.4–1.9 ms (Chrome) |
| 6 | Baselines on the same metrics | **DONE** | Zero-break, blast_parity, QS, routed island, punch, 4 ms window, full-feature GPU routing, per-bond GBDT: none passes both M5 scenes |

**What this means for the plan.** The boundary works with a perfect model, so training is
not ruled out by the handoff. The cost gate fails by about 1.5× on Metal and 2.7× in Chrome
at 10k chunks with the widths the plan proposes, so the model size and the kernel need
decisions before P2 (options at the end). The pinned "truth" must become the ensemble.

## Step 0: foundation fixes, guard, suites, ledger

- **Ported from `6622e05a`** (on `claude/stress-web-demo`; this worktree branches from
  `main` at `160126f1`, where the fixes were absent): `Polytope::mass_properties` accumulates
  the determinants and divides once (no cutoff), with its test
  `tiny_positive_volumes_are_retained_without_a_cutoff`; `serde_json` `float_roundtrip` in
  stress-ref.
- **Guard** (`scripts/destruction_peak.py guard`): 32/32 digests identical. **Suites**:
  `cargo test --release -p stress-ref --no-fail-fast`;
  `STRESS_GPU_FIXED_HORIZON=1 STRESS_METHODS=established cargo test --release -p stress-gpu -- --test-threads=1`;
  `STRESS_GPU_FIXED_HORIZON=1 cargo test --release -p stress-rubble --features gpu -- --test-threads=1`:
  all pass. No digest moved, so nothing is re-recorded.
- **Ledger diagnostic added** (no physics change): `EnergyLedger::softening_overshoot_negative`,
  the negative part of the signed overshoot.
- **The signed overshoot residual is real but negligible**: −0.012 J (free wall) to −1.5 J
  (b5), at most 6.5e-6 of E0. Positive overshoot is 6–13% of all dissipation.
- **The balance does not close on the pinned configuration**: (E_mech + D − E0)/E0 =
  +0.3% to +4.7% (wall row members up to +4.7%). The pinned methods use the penalty contact
  (`Methods::established`, `layer_contact` off), which books its dashpot quadrature rather
  than its work; `tests/conservation.rs` documents this gap as not shrinking with the step.
  Contact dissipation is 73–90% of everything booked.
- **The fracture-work floor is too small to matter**: ΣG·A over broken bonds is 0.1–0.5% of
  E0; the broken bonds book 6–28× more than it. It holds everywhere, but it is below the
  ledger's own residual.
- **Consequences for the plan's decode and losses**:
  - the energy bound can only be KE_out ≤ KE_in + W_gravity with a tolerance of at least the
    measured residual (≈5% E0); the `E_fracture,min` tightening buys nothing here;
  - `L_energy` should not use the booked contact dissipation as a target under this method
    set. Use kinetic-energy changes (exact) and per-bond dissipation, or move the oracle to
    the layer contact (a method change: the user's decision).

## Step 1: the pinned oracle

**Configuration** (`stress_rubble::scenes::realtime`, the A12 real-time configuration):
`Methods::established()` with `true_step`, contact safety 0.35, stress safety 0.5,
`SolveMode::Adaptive`, 1/60 s frames, `oracle_concrete(None)` (no Weibull scatter),
`elastic_steel` ram, no gravity, 0.15 s, the standalone `World` (CPU f64).

| Scene (harness name) | Scene hash (FNV-1a of the scene JSON) | Ram / fragments / broken bonds |
|---|---|---|
| free wall (`free_wall`, rewind-scaling's `wall_row(1)`) | `3657bced087a0e78` | 35.034 / 6 / 31 (truth 35.0 / 6) |
| wall row (`wall_row`, `wall_row(2)`) | `f9665b3652139750` | 23.622 / 133 / 401 (truth 23.6 / 133) |
| `free_wall_late` (`wall_ram_late`) | `f015401c2636260e` | 35.099 / 6 / 24 |
| `wall_row_late` (`wall_row_late(2)`) | `f0c85ad7cdb2fee3` | 24.029 / 128 / 397 |
| `free_wall_late2`, `wall_row_late2` (stress_bench `*_late2`) | `037a9756df315416`, `bfd4bcffe8bdf35d` | (ensembles only) |

**The truths are knife-edge cases.** The ram's 0.2 m face is exactly aligned with a 2 × 2 block
of 0.1 m chunks. Speed ±0.1% or a point offset of ±1 µm keep the clean plug (35.0 / 6), but ±1 mm
across the face (either axis) breaks the symmetry and the wall shatters: the free wall's
ensemble is 31.98 m/s [30.97, 33.12] and 78 fragments [67, 95]; the unperturbed run is
outside it. The same holds for the wall row (22.95 [21.81, 24.25], 200 [178, 220] against
133), the 25 and 60 m/s walls and the thick wall. The anchored b5 wall and the edge hit are
inside their spreads. **Acceptance should be against the ensemble**, not the single
symmetric run: a model that reproduces 35.0/6 would be overfit to an alignment accident.

**The GPU explicit window on the free wall**: the full-feature routed GPU
(`gpu_realtime`, rewind-scaling's "rubble, no rewind") passes the free-wall ensemble
(32.67 m/s, 72 fragments; skill 1.05 / 1.03 / 1.19). `blast_parity_routed` gets the speed
(32.01) but over-fractures (115 fragments, sizes skill 1.49).

**The wall-row gap (31.65/41 routed GPU against 23.6/133)**, decomposed on the unperturbed
`wall_row_late` (CPU f64 oracle 24.03 / 128):

1. **A routing miss in the strike frame: 24.0 → 28.3 m/s, 128 → 77 fragments**, on the CPU
   f64 stress coupled to rubble (`learned_p0 ceiling --variant coupled`). In frame 2 the
   island is {ram, wall 1} (the ram's sweep reaches only wall 1). The plug the ram punches
   out mid-frame reaches wall 2 within the same frame, while wall 2 is the engine's. So
   wall 2 takes a rigid exchange. Nothing broke under the engine, so the redo did not fire.
   This is the striker scene's miss, made by a body that was at rest at the frame's start.
   The 5 cm `wall_row` has no miss: its first frame's sweep already covers both walls, and
   the coupled run reproduces 23.62 / 133 exactly.
2. **`blast_parity_routed`'s material switches: 28.3/77 → 31.65/41.** Softening, Weibull,
   crack contact, rate effects, static fatigue, buckling, rebar and dashpots are off, and
   `blast_model` is on, inside the islands too. The full-feature GPU routing gives 28.59 / 73
   on the same scene.
3. **f32 on the GPU: about 0.3 m/s** (28.3 → 28.6, same features).

On the perturbed ensembles the speed gap mostly closes: routed GPU 23.99 m/s against 23.32 on
`wall_row_late2`. The fragments stay wrong (274 against 198). **GPU labels from
`blast_parity_routed` are not oracle labels**; the routing must include structures the
launched debris reaches within the frame.

## Step 2: the chaos floor

16 members per scene: the impact point moved by up to ±1 mm across the face, the speed
scaled by up to ±0.1% (`kin`), and with Weibull strengths of modulus 8 seeded per member
(`kin_weibull`). The table is in the appendix. Points to note:

- the spreads are wide. Ram speed IQR is 0.5–2.8 m/s; fragments range over a factor of 1.2–10.
  1 − ARI between two members is 0.47–0.65, so the partition differs in about half its pairs.
  60–84% of the bonds that ever break are uncertain (0.1 < p < 0.9);
- Weibull scatter adds little on top of the ±1 mm offset;
- **calibration of the skill rule**: a held-out oracle member, scored against the other 15
  exactly as a candidate is, has a mean skill ratio of 1.0–1.42 and passes the per-member
  rule in only 62–81% of cases. So the plan's literal 1.25 bar is failed by the oracle
  itself on some scenes. The tables give both verdicts, literal and calibrated
  (≤ max(1.25, the oracle's own mean)).

## Step 3: the perfect-model ceiling

The harness couples the f64 oracle to rubble's engine (`World::with_engine`). At every frame
the routing flags (`predict_islands`), a standalone copy of the world steps the frame: this
is the "model", the oracle itself. Its outcome is then applied as each variant says
(`stress_ref::world::handoff`):

- `window_full`: the frame-end state adopted whole;
- `window_reduced`: the same, folded to rigid bodies (momentum exact, the hidden
  deformation dropped). This is the state a model's decode hands the engine;
- `pre_filtered`: the breaks, the damage and the frame-end velocities applied at the
  frame-start poses, then a ballistic frame (the call's contacts kept from the engine);
- `pre_engine`: the same, then the engine steps the frame with its contacts.

Every variant ran on the same 16 perturbed members as the oracle ensemble, at 1/60 and (for
two variants) 1/120.

- **`window_reduced` at 1/60 passes on all six scene sets** (both M5 scenes at both strike
  timings, plus the `*_late2` pair). On `free_wall_late2` it passes the calibrated rule only:
  sizes 1.27 against the oracle's own 1.42. Its paired distance to the member with the same
  inputs is 0.00–0.40 m/s. **The reduced state is enough**: dropping the hidden deformation
  costs nothing measurable.
- **Pre-impact poses fail on the wall rows** (`pre_filtered`: skill on ram 1.9 on
  `wall_row`, 3.5 on `wall_row_late`). The pieces follow their end velocity from where they
  started, not the path they took, so they reach wall 2 at the wrong place and time.
  **The call must output the frame-end poses** (an integrated displacement per body),
  not only velocities applied before the engine steps.
- **The frame's contact must be filtered**: `pre_engine` fails everywhere (ram skill
  1.6–3.2, fragment tails up to 167 against 95).
- **1/120 frames do not help**: no variant improves.
- **One call per frame is the right count**. On the wall rows the routing flags every frame
  (11–12 calls in 12 frames), so the call is not a single impact-frame event.
- **It beats the punch everywhere**: the punch gives 22.9 m/s and 4 fragments on the free
  wall (oracle 32.3 / 77) and 23.1 / 18 on the wall row (23.3 / 198); 0% of members pass.

**Boundary as validated**: one call per routed island per frame. Inputs: the frame-start
state. Outputs: breaks, carried damage, each piece's frame-end pose, linear and angular
velocity, and the impactor's. Applied at window end with the island's contacts filtered for
that frame; the rest is rigid bodies. The island must include every structure the launched
debris can reach within the frame (step 1's miss).

## Step 4: the GPU cost envelope

A random-weight MeshGraphNet of the plan's shape (`stress_gpu::gnn_probe`,
`shaders/gnn_probe.slang`; all weights in one buffer, ≤ 8 storage buffers per stage):

- an encoder, then a U-processor on a bi-stride hierarchy (BSMS pooling of the bond graph,
  2 coarse levels): 12 rounds (3 / 2 / 2 / 2 / 3) or 8 (2 / 1 / 2 / 1 / 2);
- pool and unpool between levels, then a decoder;
- wall-shaped islands of 220, 1,000, 2,000 and 10,000 chunks; widths 32 / 64 / 128; f32 and f16.

Three ways to dispatch a round:

- `fused`: one dispatch; each node runs the edge MLP over its incoming edges in turn;
- `split`: the edge MLP edge-parallel, then the node update;
- `tiled`: GEMM tiles for both MLPs, with a CSR sum between them.

A CPU reference test (`tests/gnn_probe.rs`) checks every kernel's outputs at every width,
through the hierarchy, on a batch of two islands. So the timings are of real work. Times are
submit-to-done wall times (median of 30, after 5 warm-up calls). GPU timestamps are in the
data, but on Apple they are noisy.

- **Metal (Slang → metallib)**: the best ≥ 8-round call at 10k chunks is **3.0 ms** (W32, f16,
  8 rounds, tiled; 12 rounds 4.2 ms). W64 takes 9.3–14.7 ms and W128 40–62 ms.
- **WebGPU, Chrome** (the in-app Chromium, Dawn/Tint, same GPU, `stress-web` running the same
  Rust and Slang): best 5.3 ms at about 10k chunks (50 × 220, W32, 12 rounds); one 10k
  island takes 6.7 ms (8 rounds). WGSL through Naga natively: 5.1 ms.
- **M5-sized islands are cheap**: 220 chunks at W32 take 0.95–1.31 ms (Metal) and 1.4–1.9 ms
  (Chrome), and W64 1.65–2.4 ms (Metal).
- **The batch knee**: up to about 20 islands of 220 chunks (4,400 chunks), the cost barely
  grows (Metal W32, 12 rounds: 1.31 / 1.32 / 2.07 ms for 1 / 5 / 20; Chrome 1.7 / 1.8 / 2.8). Beyond that it grows with total
  chunks × width², about 1 TMAC/s (≈ 14% of the M3 Max's fp32 peak).
- **f16 buys 0–10%**: its storage and multiply-adds are in half, but the tiled kernel
  accumulates in float.
- **Two portability findings** (both rewrites lossless; the CPU reference passes):
  - Chrome's Tint rejected the module twice for barriers in non-uniform control flow,
    which Naga had accepted. The fixes: no early return on a lane predicate, and a uniform
    loop bound. Rule for P3: run the WGSL through Tint/Chrome, not only Naga.
  - Slang emits a 4 × 4 accumulator array as a dynamically indexed local array in WGSL.
    Naga spills it (27.6 ms against 3.2 ms for the same tiled kernel). Explicit `float4`
    rows fix it (5.1 ms).
- **Why the gate fails, and the levers left**:
  - the coarse levels carry more edges than the fine one: 54k fine, then 73k and 78k from
    BSMS's two-hop joins on a 3-D grid. That puts about 60% of the edge work on coarse
    levels;
  - the kernel is at about 14% of peak.
  - Lossless levers:
    - project h_src and h_dst through the edge MLP's first layer once per node, not per
      edge: about 2× fewer edge FLOPs;
    - a better GEMM, using subgroups where WebGPU has them;
    - fusing the CSR sum into the node MLP.
  - Architecture levers (the user's choice, P2 ablations): sparser coarse edges, or 8
    rounds at W32.
  - Estimated together, W32 with 8 rounds reaches about 1–1.5 ms on Metal. Chrome stays
    about 1.7× slower.

## Step 6: baselines (same members, same metrics)

| Baseline | Free wall (`free_wall_late2`) | Wall row (`wall_row_late2`) |
|---|---|---|
| oracle ensemble | 32.28 m/s, 77 fragments | 23.32 m/s, 198 fragments |
| zero-break (fracture off) | −5.5 m/s rebound, 1 | −5.5, 2 |
| `blast_parity` (QS, today's default) | 11.0 (inelastic whole-wall), 1 | 10.2, 122 |
| QS with the full model's features (`--config qs`) | 8.6, 1 | 8.6, 2 |
| `blast_parity_routed` (explicit island, M5) | 32.0, 115: fail on sizes | 24.0, 274: fail |
| GPU punch (`STRESS_IMPACT_PUNCH=gpu`) | 22.9, 4: fail | 23.1, 18: fail |
| 4 ms impact window | 30.9, 101: fail | 20.8 [0.1, 31.4], 289: fail |
| full-feature routed GPU (`gpu_realtime`) | 32.7, 72: **pass** | 23.6, 191: fail (ram skill 2.5) |
| perfect-model ceiling (`window_reduced`) | 32.2, 78: pass (calibrated) | 23.0, 185: **pass** |

**Per-bond GBDT** (breaks only; impact-relative features; no ram speed):

- leave one scene out, the fragment count is in range on 3 of 7 scenes (wall row 115
  against 178–220) and bond F1 is 0.67–0.89;
- trained on members of the same scene, F1 is 0.89–0.95 and the fragment count is in range
  on 6 of 7.

A tree over physics features does not generalise from 7 scenes. The bar for the graph model
is set by the ceiling, not by the tree.

"QS with Hertz peak loads" (the plan's prior feature) does not exist in the engine. The QS
baseline here uses the engine's contact loads. **Not done**: a converged Hertz-peak QS solve.

## Decisions this needs (not taken here)

1. **Acceptance target**: the ensemble, not the symmetric single run (and possibly
   re-stating M5's 35.0/6 and 23.6/133 as their ensembles).
2. **The skill rule**: the literal 1.25 bar, or one calibrated by the oracle's own held-out
   member.
3. **The cost gate**: accept W32 with 8 rounds and the kernel work above, relax the gate
   for the browser (Chrome is about 1.7× Metal), or shrink the coarse graph.
4. **The energy cap**: KE ≤ E_in + W_g with about 5% tolerance under the penalty contact, or
   the layer contact for labels (a method change).
5. **Routing**: extend the island by the launched debris' reach (closes step 1's miss for
   every path, learned or not).

## Reproduce

```bash
cd engine/stress-oracle
CARGO_TARGET_DIR=target-p0 cargo build --release -p stress-rubble --example learned_p0
CARGO_TARGET_DIR=target-p0 cargo build --release -p stress-gpu --bin stress_gpu_gnn_probe
python3 learned-impact/scripts/job.py p0-chaos -- python3 -u learned-impact/scripts/p0_runs.py learned-impact/runs/p0-chaos/out --jobs 12 --chaos
python3 learned-impact/scripts/job.py p0-gnn -- target-p0/release/stress_gpu_gnn_probe --sweep all --kernel all
cd learned-impact/python && .venv/bin/python -m learned_impact.p0_report ../runs ../findings/p0-feasibility-2026-10-20.json ../runs/p0-tables.md
```

The ceiling and baseline specs (`ceiling_ens_spec.json`, `base_*_spec.json`) are generated
by the snippets recorded in `learned-impact/scripts/p0_specs.py`. The browser probe is
`learned-impact/web-probe/index.html`, served from `engine/`, over a `wasm-pack` build of
`stress-web`. Every job writes `runs/<job>/status.json` and `progress.log`.

## Appendix: tables (generated by `p0_report.py`)

### Chaos floor (16 members; ram speed m/s, fragments, largest fraction: median [min, max])

| Scene | Perturbation | Ram exit speed | IQR | Fragments | Largest fraction | Bonds ever / uncertain | Typical 1−ARI | Unperturbed run |
|---|---|---|---|---|---|---|---|---|
| b5_v20 | kin | 16.79 [16.49, 17.02] | 0.11 | 118 [102, 156] | 0.209 [0.167, 0.346] | 1386 / 866 | 0.50 | 17.03 / 106 (inside) |
| b5_v20 | kin_weibull | 16.82 [16.38, 17.23] | 0.20 | 122 [83, 157] | 0.203 [0.167, 0.365] | 1431 / 943 | 0.52 | 17.03 / 106 (inside) |
| free_wall_edge | kin | 31.65 [31.25, 32.55] | 0.65 | 65 [40, 76] | 0.166 [0.127, 0.223] | 436 / 298 | 0.47 | 32.65 / 46 (inside) |
| free_wall_edge | kin_weibull | 31.76 [31.12, 32.34] | 0.58 | 59 [49, 80] | 0.164 [0.127, 0.218] | 448 / 308 | 0.49 | 32.65 / 46 (outside) |
| free_wall | kin | 31.98 [30.97, 33.12] | 0.48 | 78 [67, 95] | 0.157 [0.100, 0.214] | 462 / 341 | 0.54 | 35.03 / 6 (outside) |
| free_wall | kin_weibull | 32.16 [30.92, 33.25] | 0.94 | 82 [53, 105] | 0.150 [0.091, 0.173] | 459 / 328 | 0.56 | 35.03 / 6 (outside) |
| free_wall_v25 | kin | 19.70 [17.63, 21.67] | 1.22 | 50 [9, 87] | 0.209 [0.109, 0.473] | 418 / 341 | 0.52 | 21.77 / 6 (outside) |
| free_wall_v25 | kin_weibull | 19.76 [17.70, 21.72] | 1.73 | 50 [11, 69] | 0.220 [0.114, 0.318] | 426 / 347 | 0.51 | 21.77 / 6 (outside) |
| free_wall_v60 | kin | 50.55 [47.18, 52.49] | 2.01 | 74 [13, 121] | 0.168 [0.123, 0.405] | 469 / 392 | 0.64 | 52.72 / 6 (outside) |
| free_wall_v60 | kin_weibull | 50.24 [47.88, 52.55] | 1.94 | 72 [10, 124] | 0.175 [0.100, 0.545] | 472 / 409 | 0.65 | 52.72 / 6 (outside) |
| thick_wall | kin | 26.01 [23.88, 29.08] | 2.81 | 140 [90, 158] | 0.132 [0.100, 0.179] | 736 / 488 | 0.49 | 29.88 / 103 (outside) |
| thick_wall | kin_weibull | 26.09 [24.42, 29.85] | 2.00 | 140 [88, 165] | 0.127 [0.094, 0.182] | 741 / 494 | 0.50 | 29.88 / 103 (outside) |
| wall_row | kin | 22.95 [21.81, 24.25] | 0.82 | 200 [178, 220] | 0.082 [0.059, 0.116] | 930 / 562 | 0.49 | 23.62 / 133 (outside) |
| wall_row | kin_weibull | 22.92 [22.14, 24.14] | 1.22 | 200 [161, 214] | 0.073 [0.057, 0.114] | 935 / 570 | 0.52 | 23.62 / 133 (outside) |

### Perfect-model ceiling (variant ensemble against the oracle ensemble, same 16 members)

Gate: literal = the plan's rule (skill ≤ 1.25 on ram, partition, sizes; medians inside the oracle's range); calibrated = skill ≤ max(1.25, a held-out oracle member's own mean skill on that scene).

| Scene @ frame | Variant | Ram median [min, max] | Fragments median [min, max] | Skill ram / partition / sizes | Gate literal / calibrated | Members passing | Paired Δv, 1−ARI |
|---|---|---|---|---|---|---|---|
| free_wall@1_120 | **oracle** | 32.17 [30.89, 32.98] | 79 [64, 102] | 1 (by definition) | | | |
| free_wall@1_120 | pre_filtered | 32.34 [31.13, 32.76] | 87 [65, 112] | 1.16 / 0.98 / 1.11 | PASS / - | 81% | 0.16, 0.04 |
| free_wall@1_120 | window_reduced | 32.17 [30.89, 32.98] | 79 [63, 101] | 1.18 / 0.94 / 1.01 | PASS / - | 94% | 0.00, 0.00 |
| free_wall@1_60 | **oracle** | 31.98 [30.97, 33.12] | 78 [67, 95] | 1 (by definition) | | | |
| free_wall@1_60 | coupled | 31.98 [30.97, 33.12] | 78 [67, 95] | 1.20 / 0.95 / 0.95 | PASS / PASS | 100% | 0.00, 0.00 |
| free_wall@1_60 | pre_engine | 32.06 [30.97, 38.46] | 94 [69, 167] | 3.20 / 1.08 / 1.73 | fail / fail | 56% | 0.10, 0.15 |
| free_wall@1_60 | pre_filtered | 32.02 [30.97, 33.13] | 78 [67, 94] | 1.20 / 0.95 / 0.95 | PASS / PASS | 100% | 0.00, 0.00 |
| free_wall@1_60 | window_full | 31.98 [30.97, 33.12] | 78 [67, 95] | 1.20 / 0.95 / 0.95 | PASS / PASS | 100% | 0.00, 0.00 |
| free_wall@1_60 | window_reduced | 31.98 [30.97, 33.12] | 79 [67, 95] | 1.20 / 0.95 / 0.95 | PASS / PASS | 100% | 0.00, 0.00 |
| free_wall_late2@1_60 | **oracle** | 32.28 [30.72, 34.92] | 77 [6, 98] | 1 (by definition) | | | |
| free_wall_late2@1_60 | gpu_blast_parity | 11.00 [10.94, 11.09] | 1 [1, 1] | 16.41 / 1.54 / 3.39 | fail / fail | 0% | 21.28, 1.00 |
| free_wall_late2@1_60 | gpu_punch | 22.87 [20.29, 23.39] | 4 [3, 5] | 7.92 / 1.44 / 3.06 | fail / fail | 0% | 10.42, 0.98 |
| free_wall_late2@1_60 | gpu_qs | 8.60 [8.58, 8.63] | 1 [1, 1] | 18.25 / 1.54 / 3.39 | fail / fail | 0% | 23.68, 1.00 |
| free_wall_late2@1_60 | gpu_realtime | 32.67 [30.72, 34.95] | 72 [6, 99] | 1.05 / 1.03 / 1.19 | PASS / PASS | 75% | 0.89, 0.48 |
| free_wall_late2@1_60 | gpu_routed | 32.01 [30.87, 33.43] | 115 [59, 149] | 0.94 / 1.18 / 1.49 | fail / fail | 25% | 1.05, 0.77 |
| free_wall_late2@1_60 | gpu_window4ms | 30.87 [28.50, 34.96] | 101 [7, 146] | 1.85 / 1.10 / 1.54 | fail / fail | 19% | 1.31, 0.65 |
| free_wall_late2@1_60 | window_reduced | 32.24 [31.03, 34.92] | 78 [7, 94] | 1.10 / 1.00 / 1.27 | fail / PASS | 75% | 0.05, 0.12 |
| free_wall_late2@1_60 | zerobreak | -5.49 [-5.70, -5.42] | 1 [1, 1] | 28.92 / 1.54 / 3.39 | fail / fail | 0% | 37.82, 1.00 |
| free_wall_late@1_120 | **oracle** | 32.27 [30.67, 34.95] | 78 [6, 101] | 1 (by definition) | | | |
| free_wall_late@1_120 | pre_filtered | 32.07 [30.56, 34.44] | 79 [18, 105] | 1.32 / 0.93 / 1.05 | fail / - | 75% | 0.28, 0.18 |
| free_wall_late@1_120 | window_reduced | 32.15 [30.71, 34.95] | 75 [6, 101] | 1.25 / 0.97 / 1.21 | fail / - | 88% | 0.04, 0.04 |
| free_wall_late@1_60 | **oracle** | 32.33 [30.78, 34.92] | 76 [6, 101] | 1 (by definition) | | | |
| free_wall_late@1_60 | coupled | 32.42 [30.71, 34.92] | 74 [6, 99] | 1.18 / 0.99 / 1.19 | PASS / PASS | 75% | 0.06, 0.02 |
| free_wall_late@1_60 | pre_engine | 31.70 [26.79, 34.36] | 98 [22, 161] | 1.56 / 1.04 / 1.31 | fail / fail | 50% | 1.05, 0.26 |
| free_wall_late@1_60 | pre_filtered | 32.33 [30.21, 34.42] | 88 [24, 127] | 1.28 / 0.99 / 1.09 | fail / fail | 81% | 0.40, 0.22 |
| free_wall_late@1_60 | window_full | 32.42 [30.71, 34.92] | 74 [6, 99] | 1.18 / 0.99 / 1.19 | PASS / PASS | 75% | 0.06, 0.02 |
| free_wall_late@1_60 | window_reduced | 32.42 [30.70, 34.92] | 71 [7, 97] | 1.17 / 1.01 / 1.19 | PASS / PASS | 75% | 0.06, 0.05 |
| wall_row@1_120 | **oracle** | 22.96 [21.67, 24.06] | 194 [180, 216] | 1 (by definition) | | | |
| wall_row@1_120 | pre_filtered | 22.57 [20.57, 24.56] | 202 [178, 252] | 1.34 / 0.96 / 1.27 | fail / - | 69% | 0.48, 0.10 |
| wall_row@1_120 | window_reduced | 22.95 [21.66, 24.05] | 193 [182, 215] | 1.00 / 0.93 / 1.04 | PASS / - | 100% | 0.05, 0.02 |
| wall_row@1_60 | **oracle** | 22.95 [21.81, 24.25] | 200 [178, 220] | 1 (by definition) | | | |
| wall_row@1_60 | coupled | 22.95 [21.81, 24.25] | 200 [178, 220] | 1.08 / 0.94 / 1.03 | PASS / PASS | 100% | 0.00, 0.00 |
| wall_row@1_60 | pre_engine | 22.78 [14.69, 24.10] | 220 [189, 257] | 2.21 / 0.99 / 1.52 | fail / fail | 50% | 0.16, 0.13 |
| wall_row@1_60 | pre_filtered | 22.14 [19.75, 23.02] | 224 [188, 298] | 1.91 / 1.05 / 1.88 | fail / fail | 44% | 0.83, 0.11 |
| wall_row@1_60 | window_full | 22.95 [21.81, 24.25] | 200 [178, 220] | 1.08 / 0.94 / 1.03 | PASS / PASS | 100% | 0.00, 0.00 |
| wall_row@1_60 | window_reduced | 22.95 [21.81, 24.25] | 200 [177, 218] | 1.08 / 0.94 / 1.03 | PASS / PASS | 88% | 0.00, 0.00 |
| wall_row_late2@1_60 | **oracle** | 23.32 [21.92, 24.70] | 198 [129, 224] | 1 (by definition) | | | |
| wall_row_late2@1_60 | gpu_blast_parity | 10.19 [9.66, 10.82] | 122 [52, 267] | 19.60 / 1.51 / 2.14 | fail / fail | 0% | 12.90, 0.87 |
| wall_row_late2@1_60 | gpu_punch | 23.11 [19.71, 24.77] | 18 [12, 23] | 1.78 / 1.55 / 8.62 | fail / fail | 0% | 0.96, 0.93 |
| wall_row_late2@1_60 | gpu_qs | 8.60 [8.59, 8.64] | 2 [2, 2] | 22.03 / 1.58 / 9.69 | fail / fail | 0% | 14.72, 0.95 |
| wall_row_late2@1_60 | gpu_realtime | 23.62 [21.57, 28.42] | 191 [66, 226] | 2.51 / 1.03 / 1.55 | fail / fail | 75% | 0.91, 0.45 |
| wall_row_late2@1_60 | gpu_routed | 23.99 [22.92, 28.00] | 274 [182, 308] | 2.04 / 1.26 / 2.37 | fail / fail | 0% | 0.71, 0.74 |
| wall_row_late2@1_60 | gpu_window4ms | 20.83 [0.10, 31.35] | 289 [43, 399] | 8.70 / 1.31 / 3.12 | fail / fail | 6% | 3.62, 0.75 |
| wall_row_late2@1_60 | window_reduced | 22.98 [21.67, 23.94] | 185 [118, 211] | 1.12 / 0.96 / 1.19 | PASS / PASS | 94% | 0.33, 0.21 |
| wall_row_late2@1_60 | zerobreak | -5.49 [-5.70, -5.42] | 2 [2, 2] | 43.27 / 1.58 / 9.69 | fail / fail | 0% | 28.85, 0.95 |
| wall_row_late@1_120 | **oracle** | 23.09 [21.48, 24.49] | 199 [136, 214] | 1 (by definition) | | | |
| wall_row_late@1_120 | pre_filtered | 24.07 [20.73, 26.29] | 206 [108, 240] | 2.92 / 1.07 / 1.19 | fail / - | 25% | 1.59, 0.34 |
| wall_row_late@1_120 | window_reduced | 23.01 [21.59, 24.29] | 198 [121, 211] | 1.13 / 1.01 / 1.19 | PASS / - | 88% | 0.36, 0.16 |
| wall_row_late@1_60 | **oracle** | 23.47 [22.59, 24.86] | 190 [122, 219] | 1 (by definition) | | | |
| wall_row_late@1_60 | coupled | 23.98 [22.06, 27.26] | 188 [88, 209] | 2.10 / 1.01 / 1.29 | fail / fail | 69% | 0.71, 0.25 |
| wall_row_late@1_60 | pre_engine | 24.62 [20.20, 25.62] | 190 [111, 284] | 2.16 / 1.06 / 1.18 | fail / fail | 44% | 1.07, 0.37 |
| wall_row_late@1_60 | pre_filtered | 25.57 [18.17, 27.85] | 189 [158, 241] | 3.48 / 1.08 / 1.00 | fail / fail | 31% | 2.35, 0.40 |
| wall_row_late@1_60 | window_full | 23.56 [22.53, 24.92] | 188 [127, 213] | 1.01 / 0.99 / 1.14 | PASS / PASS | 100% | 0.20, 0.10 |
| wall_row_late@1_60 | window_reduced | 23.19 [22.20, 24.40] | 188 [118, 211] | 1.06 / 0.96 / 1.09 | PASS / PASS | 81% | 0.40, 0.16 |

#### A held-out oracle member scored as a candidate (mean / p90 / max skill ratio)

| Scene | Ram | Partition | Sizes | Members passing the per-member rule |
|---|---|---|---|---|
| free_wall_late | 1.28 / 1.99 / 2.26 | 1.06 / 1.23 / 1.66 | 1.37 / 2.16 / 4.65 | 62% |
| free_wall | 1.32 / 2.48 / 3.02 | 1.01 / 1.10 / 1.25 | 1.02 / 1.25 / 1.48 | 75% |
| wall_row_late | 1.12 / 1.52 / 2.27 | 1.05 / 1.24 / 1.50 | 1.26 / 1.91 / 3.76 | 81% |
| wall_row | 1.20 / 2.07 / 2.56 | 1.01 / 1.12 / 1.16 | 1.11 / 1.60 / 2.03 | 81% |
| free_wall_late2 | 1.23 / 2.01 / 2.07 | 1.06 / 1.38 / 1.57 | 1.42 / 2.68 / 4.09 | 62% |
| wall_row_late2 | 1.25 / 2.14 / 2.71 | 1.02 / 1.19 / 1.46 | 1.36 / 2.12 / 3.84 | 81% |

### Energy ledger on the pinned configuration (unperturbed run; kin ensemble range)

| Scene | Balance residual (E+D−E0)/E0 | Negative overshoot (J) | Negative overshoot / E0 | Fracture floor ΣGA (J) | Booked by broken bonds (J) | Floor / E0 | Contact share of D |
|---|---|---|---|---|---|---|---|
| b5_v20 | +0.0167 (kin +0.0176 to +0.0270) | -0.598 (kin min -1.49) | -6.5e-06 | 615.6 | 4861 | 3.0e-03 | 0.73 |
| free_wall | +0.0063 (kin +0.0127 to +0.0205) | -0.0122 (kin min -0.209) | -1.3e-06 | 37.2 | 1061 | 2.0e-03 | 0.84 |
| free_wall_edge | +0.0227 (kin +0.0162 to +0.0310) | -0.18 (kin min -0.422) | -2.6e-06 | 238.8 | 3012 | 1.8e-03 | 0.75 |
| free_wall_v25 | +0.0189 (kin +0.0193 to +0.0367) | -0.0593 (kin min -0.051) | -8.2e-07 | 34.8 | 1099 | 3.7e-03 | 0.73 |
| free_wall_v60 | +0.0028 (kin +0.0043 to +0.0121) | -0.0011 (kin min -0.598) | -1.7e-06 | 28.8 | 1013 | 8.8e-04 | 0.90 |
| thick_wall | +0.0152 (kin +0.0194 to +0.0270) | -0.013 (kin min -0.409) | -2.6e-06 | 525.6 | 2854 | 3.5e-03 | 0.81 |
| wall_row | +0.0201 (kin +0.0387 to +0.0473) | -0.0126 (kin min -0.244) | -1.5e-06 | 481.2 | 6243 | 4.6e-03 | 0.81 |

### GPU cost envelope (wall ms, submit to done, median of 30; 10k chunks unless noted)

- `chrome_webgpu`: best ≥ 8-round config at 10k chunks: W32 f32 u12 tiled: **5.30 ms** wall (4.72 ms GPU timestamps)
- `metal_metallib`: best ≥ 8-round config at 10k chunks: W32 f16 u8 tiled: **3.01 ms** wall (2.15 ms GPU timestamps)
- `metal_metallib_contended`: best ≥ 8-round config at 10k chunks: W32 f16 u8 tiled: **3.00 ms** wall (2.06 ms GPU timestamps)
- `metal_metallib_tiled_v2`: best ≥ 8-round config at 10k chunks: W32 f16 u8 tiled: **3.04 ms** wall (2.44 ms GPU timestamps)
- `metal_wgsl_naga`: best ≥ 8-round config at 10k chunks: W32 f16 u8 tiled: **5.12 ms** wall (4.45 ms GPU timestamps)
- `metal_wgsl_naga_arrays`: best ≥ 8-round config at 10k chunks: W32 f32 u8 tiled: **27.58 ms** wall (17.90 ms GPU timestamps)

| Source | Chunks | W | Precision | Rounds | Fused | Split | Tiled |
|---|---|---|---|---|---|---|---|
| metal_metallib | 220 | 32 | f16 | 8 | 1.63 | 0.77 | 1.31 |
| metal_metallib | 220 | 32 | f16 | 12 | 2.22 | 1.00 | 1.24 |
| metal_metallib | 220 | 32 | f32 | 8 | 1.70 | 0.83 | 0.92 |
| metal_metallib | 220 | 32 | f32 | 12 | 2.27 | 1.58 | 1.21 |
| metal_metallib | 220 | 64 | f16 | 8 | 2.97 | 1.36 | 2.11 |
| metal_metallib | 220 | 64 | f16 | 12 | 3.92 | 1.98 | 2.71 |
| metal_metallib | 220 | 64 | f32 | 8 | 3.08 | 1.97 | 1.98 |
| metal_metallib | 220 | 64 | f32 | 12 | 4.11 | 2.73 | 2.33 |
| metal_metallib | 220 | 128 | f16 | 8 | 5.58 | 3.27 | 5.29 |
| metal_metallib | 220 | 128 | f16 | 12 | 8.05 | 4.25 | 8.47 |
| metal_metallib | 220 | 128 | f32 | 8 | 7.65 | 3.68 | 5.26 |
| metal_metallib | 220 | 128 | f32 | 12 | 10.81 | 7.69 | 6.19 |
| metal_metallib | 1000 | 32 | f16 | 8 | 1.74 | 1.16 | 1.04 |
| metal_metallib | 1000 | 32 | f16 | 12 | 2.31 | 1.74 | 1.33 |
| metal_metallib | 1000 | 32 | f32 | 8 | 1.79 | 1.41 | 1.06 |
| metal_metallib | 1000 | 32 | f32 | 12 | 2.36 | 1.73 | 1.30 |
| metal_metallib | 1000 | 64 | f16 | 8 | 3.50 | 2.63 | 1.74 |
| metal_metallib | 1000 | 64 | f16 | 12 | 4.84 | 3.86 | 2.50 |
| metal_metallib | 1000 | 64 | f32 | 8 | 3.74 | 3.14 | 1.80 |
| metal_metallib | 1000 | 64 | f32 | 12 | 5.34 | 4.74 | 2.60 |
| metal_metallib | 1000 | 128 | f16 | 8 | 10.55 | 8.01 | 6.07 |
| metal_metallib | 1000 | 128 | f16 | 12 | 15.82 | 12.47 | 9.23 |
| metal_metallib | 1000 | 128 | f32 | 8 | 13.16 | 11.00 | 6.34 |
| metal_metallib | 1000 | 128 | f32 | 12 | 19.79 | 15.47 | 9.30 |
| metal_metallib | 2000 | 32 | f32 | 12 | 3.63 | 2.96 | 2.78 |
| metal_metallib | 2000 | 64 | f32 | 12 | 10.09 | 8.81 | 5.69 |
| metal_metallib | 2000 | 128 | f32 | 12 | 37.84 | 31.75 | 17.93 |
| metal_metallib | 10000 | 32 | f16 | 8 | 7.91 | 7.86 | 3.01 |
| metal_metallib | 10000 | 32 | f16 | 12 | 11.49 | 11.40 | 4.27 |
| metal_metallib | 10000 | 32 | f32 | 8 | 8.05 | 8.04 | 3.24 |
| metal_metallib | 10000 | 32 | f32 | 12 | 11.54 | 11.15 | 4.53 |
| metal_metallib | 10000 | 64 | f16 | 8 | 25.27 | 24.86 | 9.28 |
| metal_metallib | 10000 | 64 | f16 | 12 | 37.11 | 36.29 | 13.55 |
| metal_metallib | 10000 | 64 | f32 | 8 | 27.66 | 26.92 | 10.10 |
| metal_metallib | 10000 | 64 | f32 | 12 | 40.80 | 38.65 | 14.34 |
| metal_metallib | 10000 | 128 | f16 | 8 | 91.00 | 99.09 | 40.00 |
| metal_metallib | 10000 | 128 | f16 | 12 | 138.56 | 139.50 | 58.79 |
| metal_metallib | 10000 | 128 | f32 | 8 | 107.01 | 115.38 | 41.24 |
| metal_metallib | 10000 | 128 | f32 | 12 | 158.71 | 160.62 | 60.67 |
| metal_metallib_tiled_v2 | 220 | 32 | f16 | 8 | - | - | 0.95 |
| metal_metallib_tiled_v2 | 220 | 32 | f16 | 12 | - | - | 1.27 |
| metal_metallib_tiled_v2 | 220 | 32 | f32 | 8 | - | - | 0.95 |
| metal_metallib_tiled_v2 | 220 | 32 | f32 | 12 | - | - | 1.31 |
| metal_metallib_tiled_v2 | 220 | 64 | f16 | 8 | - | - | 1.65 |
| metal_metallib_tiled_v2 | 220 | 64 | f16 | 12 | - | - | 2.33 |
| metal_metallib_tiled_v2 | 220 | 64 | f32 | 8 | - | - | 1.70 |
| metal_metallib_tiled_v2 | 220 | 64 | f32 | 12 | - | - | 2.36 |
| metal_metallib_tiled_v2 | 220 | 128 | f16 | 8 | - | - | 3.99 |
| metal_metallib_tiled_v2 | 220 | 128 | f16 | 12 | - | - | 6.31 |
| metal_metallib_tiled_v2 | 220 | 128 | f32 | 8 | - | - | 4.13 |
| metal_metallib_tiled_v2 | 220 | 128 | f32 | 12 | - | - | 5.73 |
| metal_metallib_tiled_v2 | 1000 | 32 | f16 | 8 | - | - | 0.97 |
| metal_metallib_tiled_v2 | 1000 | 32 | f16 | 12 | - | - | 1.35 |
| metal_metallib_tiled_v2 | 1000 | 32 | f32 | 8 | - | - | 1.00 |
| metal_metallib_tiled_v2 | 1000 | 32 | f32 | 12 | - | - | 1.34 |
| metal_metallib_tiled_v2 | 1000 | 64 | f16 | 8 | - | - | 1.83 |
| metal_metallib_tiled_v2 | 1000 | 64 | f16 | 12 | - | - | 2.53 |
| metal_metallib_tiled_v2 | 1000 | 64 | f32 | 8 | - | - | 1.84 |
| metal_metallib_tiled_v2 | 1000 | 64 | f32 | 12 | - | - | 2.61 |
| metal_metallib_tiled_v2 | 1000 | 128 | f16 | 8 | - | - | 5.92 |
| metal_metallib_tiled_v2 | 1000 | 128 | f16 | 12 | - | - | 8.70 |
| metal_metallib_tiled_v2 | 1000 | 128 | f32 | 8 | - | - | 6.17 |
| metal_metallib_tiled_v2 | 1000 | 128 | f32 | 12 | - | - | 9.13 |
| metal_metallib_tiled_v2 | 2000 | 32 | f32 | 12 | - | - | 1.71 |
| metal_metallib_tiled_v2 | 2000 | 64 | f32 | 12 | - | - | 4.51 |
| metal_metallib_tiled_v2 | 2000 | 128 | f32 | 12 | - | - | 16.45 |
| metal_metallib_tiled_v2 | 10000 | 32 | f16 | 8 | - | - | 3.04 |
| metal_metallib_tiled_v2 | 10000 | 32 | f16 | 12 | - | - | 4.20 |
| metal_metallib_tiled_v2 | 10000 | 32 | f32 | 8 | - | - | 3.15 |
| metal_metallib_tiled_v2 | 10000 | 32 | f32 | 12 | - | - | 4.44 |
| metal_metallib_tiled_v2 | 10000 | 64 | f16 | 8 | - | - | 9.28 |
| metal_metallib_tiled_v2 | 10000 | 64 | f16 | 12 | - | - | 13.44 |
| metal_metallib_tiled_v2 | 10000 | 64 | f32 | 8 | - | - | 10.19 |
| metal_metallib_tiled_v2 | 10000 | 64 | f32 | 12 | - | - | 14.66 |
| metal_metallib_tiled_v2 | 10000 | 128 | f16 | 8 | - | - | 40.31 |
| metal_metallib_tiled_v2 | 10000 | 128 | f16 | 12 | - | - | 59.21 |
| metal_metallib_tiled_v2 | 10000 | 128 | f32 | 8 | - | - | 41.73 |
| metal_metallib_tiled_v2 | 10000 | 128 | f32 | 12 | - | - | 61.54 |
| metal_wgsl_naga | 220 | 32 | f16 | 8 | - | - | 1.25 |
| metal_wgsl_naga | 220 | 32 | f16 | 12 | - | - | 1.77 |
| metal_wgsl_naga | 220 | 32 | f32 | 8 | - | - | 1.25 |
| metal_wgsl_naga | 220 | 32 | f32 | 12 | - | - | 1.59 |
| metal_wgsl_naga | 220 | 64 | f16 | 8 | - | - | 2.26 |
| metal_wgsl_naga | 220 | 64 | f16 | 12 | - | - | 3.17 |
| metal_wgsl_naga | 220 | 64 | f32 | 8 | - | - | 2.25 |
| metal_wgsl_naga | 220 | 64 | f32 | 12 | - | - | 3.06 |
| metal_wgsl_naga | 220 | 128 | f16 | 8 | - | - | 6.43 |
| metal_wgsl_naga | 220 | 128 | f16 | 12 | - | - | 9.07 |
| metal_wgsl_naga | 220 | 128 | f32 | 8 | - | - | 5.96 |
| metal_wgsl_naga | 220 | 128 | f32 | 12 | - | - | 8.02 |
| metal_wgsl_naga | 1000 | 32 | f16 | 8 | - | - | 1.95 |
| metal_wgsl_naga | 1000 | 32 | f16 | 12 | - | - | 2.68 |
| metal_wgsl_naga | 1000 | 32 | f32 | 8 | - | - | 1.83 |
| metal_wgsl_naga | 1000 | 32 | f32 | 12 | - | - | 2.02 |
| metal_wgsl_naga | 1000 | 64 | f16 | 8 | - | - | 3.05 |
| metal_wgsl_naga | 1000 | 64 | f16 | 12 | - | - | 4.43 |
| metal_wgsl_naga | 1000 | 64 | f32 | 8 | - | - | 3.28 |
| metal_wgsl_naga | 1000 | 64 | f32 | 12 | - | - | 3.81 |
| metal_wgsl_naga | 1000 | 128 | f16 | 8 | - | - | 8.69 |
| metal_wgsl_naga | 1000 | 128 | f16 | 12 | - | - | 12.86 |
| metal_wgsl_naga | 1000 | 128 | f32 | 8 | - | - | 9.29 |
| metal_wgsl_naga | 1000 | 128 | f32 | 12 | - | - | 12.54 |
| metal_wgsl_naga | 2000 | 32 | f32 | 12 | - | - | 2.60 |
| metal_wgsl_naga | 2000 | 64 | f32 | 12 | - | - | 6.30 |
| metal_wgsl_naga | 2000 | 128 | f32 | 12 | - | - | 21.51 |
| metal_wgsl_naga | 10000 | 32 | f16 | 8 | - | - | 5.12 |
| metal_wgsl_naga | 10000 | 32 | f16 | 12 | - | - | 7.04 |
| metal_wgsl_naga | 10000 | 32 | f32 | 8 | - | - | 5.24 |
| metal_wgsl_naga | 10000 | 32 | f32 | 12 | - | - | 7.50 |
| metal_wgsl_naga | 10000 | 64 | f16 | 8 | - | - | 13.20 |
| metal_wgsl_naga | 10000 | 64 | f16 | 12 | - | - | 19.03 |
| metal_wgsl_naga | 10000 | 64 | f32 | 8 | - | - | 14.01 |
| metal_wgsl_naga | 10000 | 64 | f32 | 12 | - | - | 19.74 |
| metal_wgsl_naga | 10000 | 128 | f16 | 8 | - | - | 53.03 |
| metal_wgsl_naga | 10000 | 128 | f16 | 12 | - | - | 77.73 |
| metal_wgsl_naga | 10000 | 128 | f32 | 8 | - | - | 53.61 |
| metal_wgsl_naga | 10000 | 128 | f32 | 12 | - | - | 80.71 |
| chrome_webgpu | 220 | 32 | f16 | 8 | 2.30 | 1.30 | 1.40 |
| chrome_webgpu | 220 | 32 | f16 | 12 | 3.00 | 1.60 | 1.90 |
| chrome_webgpu | 220 | 32 | f32 | 8 | 2.30 | 1.40 | 1.60 |
| chrome_webgpu | 220 | 32 | f32 | 12 | 3.00 | 1.50 | 1.70 |
| chrome_webgpu | 220 | 64 | f16 | 8 | 3.90 | 1.70 | 5.70 |
| chrome_webgpu | 220 | 64 | f16 | 12 | 4.80 | 2.60 | 6.40 |
| chrome_webgpu | 220 | 64 | f32 | 8 | 3.80 | 2.20 | 4.80 |
| chrome_webgpu | 220 | 64 | f32 | 12 | 5.60 | 2.70 | 2.90 |
| chrome_webgpu | 220 | 128 | f16 | 8 | 7.20 | 4.60 | 9.00 |
| chrome_webgpu | 220 | 128 | f16 | 12 | 10.00 | 6.40 | 10.50 |
| chrome_webgpu | 220 | 128 | f32 | 8 | 9.10 | 5.10 | 8.80 |
| chrome_webgpu | 220 | 128 | f32 | 12 | 13.40 | 6.30 | 8.30 |
| chrome_webgpu | 1000 | 32 | f16 | 8 | 2.60 | 1.90 | 1.90 |
| chrome_webgpu | 1000 | 32 | f16 | 12 | 3.30 | 2.60 | 2.40 |
| chrome_webgpu | 1000 | 32 | f32 | 8 | 2.60 | 2.20 | 3.00 |
| chrome_webgpu | 1000 | 32 | f32 | 12 | 3.20 | 2.50 | 2.90 |
| chrome_webgpu | 1000 | 64 | f16 | 8 | 5.40 | 4.20 | 6.00 |
| chrome_webgpu | 1000 | 64 | f16 | 12 | 7.70 | 6.00 | 5.40 |
| chrome_webgpu | 1000 | 64 | f32 | 8 | 5.50 | 5.10 | 5.40 |
| chrome_webgpu | 1000 | 64 | f32 | 12 | 7.60 | 6.30 | 4.00 |
| chrome_webgpu | 1000 | 128 | f16 | 8 | 16.90 | 11.80 | 11.40 |
| chrome_webgpu | 1000 | 128 | f16 | 12 | 24.30 | 17.70 | 14.80 |
| chrome_webgpu | 1000 | 128 | f32 | 8 | 17.80 | 12.70 | 11.10 |
| chrome_webgpu | 1000 | 128 | f32 | 12 | 25.00 | 19.90 | 11.80 |
| chrome_webgpu | 2000 | 32 | f32 | 12 | 5.20 | 4.30 | 3.30 |
| chrome_webgpu | 2000 | 64 | f32 | 12 | 15.60 | 12.50 | 5.40 |
| chrome_webgpu | 2000 | 128 | f32 | 12 | 54.20 | 45.90 | 19.90 |
| chrome_webgpu | 10000 | 32 | f16 | 8 | 13.30 | 11.80 | 6.70 |
| chrome_webgpu | 10000 | 32 | f16 | 12 | 18.10 | 17.30 | 8.40 |
| chrome_webgpu | 10000 | 32 | f32 | 8 | 14.30 | 11.90 | 6.80 |
| chrome_webgpu | 10000 | 32 | f32 | 12 | 18.80 | 17.20 | 9.00 |
| chrome_webgpu | 10000 | 64 | f16 | 8 | 41.60 | 39.90 | 15.50 |
| chrome_webgpu | 10000 | 64 | f16 | 12 | 55.80 | 57.40 | 23.10 |
| chrome_webgpu | 10000 | 64 | f32 | 8 | 39.90 | 41.80 | 17.20 |
| chrome_webgpu | 10000 | 64 | f32 | 12 | 58.20 | 59.90 | 21.70 |
| chrome_webgpu | 10000 | 128 | f16 | 8 | 169.70 | 142.60 | 58.70 |
| chrome_webgpu | 10000 | 128 | f16 | 12 | 244.00 | 213.60 | 86.10 |
| chrome_webgpu | 10000 | 128 | f32 | 8 | 180.60 | 168.00 | 62.90 |
| chrome_webgpu | 10000 | 128 | f32 | 12 | 261.20 | 256.40 | 88.30 |

#### Batch scaling (tiled, f32, 12 rounds; wall ms)

| Source | Islands × chunks | W32 | W64 | W128 |
|---|---|---|---|---|
| metal_metallib_tiled_v2 | 1 × 220 | 1.31 | 2.36 | 5.73 |
| metal_metallib_tiled_v2 | 5 × 220 | 1.32 | 2.62 | 9.15 |
| metal_metallib_tiled_v2 | 20 × 220 | 2.07 | 6.16 | 23.97 |
| metal_metallib_tiled_v2 | 50 × 220 | 3.95 | 12.39 | 51.55 |
| metal_metallib_tiled_v2 | 1 × 1000 | 1.34 | 2.61 | 9.13 |
| metal_metallib_tiled_v2 | 5 × 1000 | 2.45 | 6.88 | 27.66 |
| metal_metallib_tiled_v2 | 20 × 1000 | 6.91 | 22.57 | 95.98 |
| metal_metallib_tiled_v2 | 50 × 1000 | 15.40 | 52.51 | 242.99 |
| metal_metallib_tiled_v2 | 1 × 2000 | 1.71 | 4.51 | 16.45 |
| metal_metallib_tiled_v2 | 5 × 2000 | 4.39 | 14.42 | 60.96 |
| metal_metallib_tiled_v2 | 20 × 2000 | 14.89 | 50.72 | 220.68 |
| metal_metallib_tiled_v2 | 50 × 2000 | 34.66 | 122.03 | 563.47 |
| chrome_webgpu | 1 × 220 | 1.70 | 2.90 | 8.30 |
| chrome_webgpu | 5 × 220 | 1.80 | 3.20 | 11.50 |
| chrome_webgpu | 20 × 220 | 2.80 | 8.00 | 32.20 |
| chrome_webgpu | 50 × 220 | 5.30 | 15.90 | 70.70 |
| chrome_webgpu | 1 × 1000 | 2.90 | 4.00 | 11.80 |
| chrome_webgpu | 5 × 1000 | 3.90 | 10.10 | 37.00 |
| chrome_webgpu | 20 × 1000 | 9.30 | 29.40 | 133.80 |
| chrome_webgpu | 50 × 1000 | 22.20 | 72.40 | 321.20 |
| chrome_webgpu | 1 × 2000 | 3.30 | 5.40 | 19.90 |
| chrome_webgpu | 5 × 2000 | 6.00 | 17.70 | 83.70 |
| chrome_webgpu | 20 × 2000 | 21.40 | 74.90 | 309.50 |
| chrome_webgpu | 50 × 2000 | 51.90 | 169.00 | 752.10 |

### Per-bond GBDT baseline (breaks only)

| Scene | Split | Fragments (ensemble min/median/max) | In range | Largest in range | Bond P / R / F1 vs consensus | Brier vs frequency |
|---|---|---|---|---|---|---|
| free_wall | leave_one_scene_out | 60 (67/78/95) | False | True | 0.97 / 0.76 / 0.85 | 0.018 |
| wall_row | leave_one_scene_out | 115 (178/200/220) | False | False | 0.97 / 0.70 / 0.81 | 0.044 |
| free_wall_v25 | leave_one_scene_out | 79 (9/50/87) | True | True | 0.61 / 0.96 / 0.75 | 0.051 |
| free_wall_v60 | leave_one_scene_out | 58 (13/74/121) | True | False | 0.94 / 0.74 / 0.83 | 0.017 |
| free_wall_edge | leave_one_scene_out | 62 (40/65/76) | True | False | 0.72 / 0.62 / 0.67 | 0.113 |
| thick_wall | leave_one_scene_out | 164 (90/140/158) | False | True | 0.89 / 0.89 / 0.89 | 0.022 |
| b5_v20 | leave_one_scene_out | 181 (102/118/156) | False | False | 0.67 / 0.77 / 0.71 | 0.064 |
| free_wall | in_scene | 91 (67/78/95) | True | True | 0.95 / 0.91 / 0.93 | 0.007 |
| wall_row | in_scene | 218 (178/200/220) | True | True | 0.96 / 0.94 / 0.95 | 0.007 |
| free_wall_v25 | in_scene | 34 (9/50/87) | True | True | 0.96 / 0.88 / 0.92 | 0.006 |
| free_wall_v60 | in_scene | 83 (13/74/121) | True | True | 0.93 / 0.88 / 0.90 | 0.008 |
| free_wall_edge | in_scene | 52 (40/65/76) | True | False | 0.97 / 0.89 / 0.93 | 0.007 |
| thick_wall | in_scene | 151 (90/140/158) | True | True | 0.92 / 0.91 / 0.92 | 0.008 |
| b5_v20 | in_scene | 96 (102/118/156) | False | False | 0.93 / 0.85 / 0.89 | 0.009 |
