# Explicit multirate integration of the bond network (`sim.methods.multirate`)

Written 2026-10-16. Question (REALTIME_ROADMAP A1): one to three light chunks ("slivers")
set the explicit step of every mixed building, and A1 estimated that letting each chunk run
at its own step would cut the substep work ×1.9–4.3 (×18 on minas-tirith). The roadmap
listed multirate as research. `findings/stress-literature-review.md` (branch
`claude/interesting-robinson-21197b`) found explicit–explicit subcycling established:
asynchronous variational integrators (AVI) conserve momentum and have convergence proofs,
and clustered local time stepping runs on GPUs (SeisSol). Can the oracle take it, and what
does it buy?

Code: `crates/stress-ref/src/solver.rs` (`open_multirate_window`, `bond_rates`,
`chunk_native_dts`, `firing_bonds`), `world.rs` (`open_multirate_window`,
`chunk_contact_dts`), `scene.rs` (`Methods::multirate`). Off by default; with it off every
path is the previous one (default test suite unchanged; with every bond at rate 1 the
switch is bit-identical to single rate, `tests/multirate.rs`).

## Verdict

- **It works as an oracle method.** The integrator is AVI at power-of-two step ratios:
  each bond runs at the rate of its finer chunk and kicks both chunks with a balanced
  impulse. It keeps exactly what single rate keeps: linear momentum to round-off through
  impact and fracture, angular momentum as single rate does (the difference vanishes with
  the substep), energy converging to zero error at first order, and no energy drift over
  1e5 undamped substeps at safety 0.9.
- **Fidelity.** Catalogue: 34/35 scenes pass every gameplay metric. The one diff
  (s_house_car, `where`) is inside the reference's own spread: single rate against
  itself at safety 0.45 and 0.4 fails the same test. On `true_step`: 35/35.
- **Packs.**
  - house-1story, house-2story and villa-savoye rammed pass every metric with ×4.8, ×4.3
    and ×4.0 fewer bond evaluations. The Shardwright projectile passes too.
  - The Shardwright ram cannot be judged: its single-rate reference creates energy
    (3.7e9 J of contact dissipation against a 28.8 kJ ram).
  - It is a step change, so its accuracy is that of the requested safety at every bond.
    Single rate over-resolves the heavy chunks.
  - On b6_spall (a wave problem) the spall speed at the default safety 0.5 moves +8 %
    (3.59 → 3.88 m/s; OpenCourant 3.02). At safety ≤ 0.25 the two agree within 0.6 % and
    converge together.
- **Work.** At the oracle's default step (Gershgorin, contact combined), bond evaluations
  fall ×4.6 on house-1story, ×4.2 on house-2story, ×4.1 on villa-savoye and ×1.13 on the
  Shardwright building. On a rammed house-1story the solver substep runs ×2.5 faster.
  - The standalone world's debris contacts, which this does not touch, are 83 % of that
    run.
- **Against A1's column: A1 is an upper bound.** A1 binned every chunk at its own step.
  A sliver's neighbours cannot do that: the bond to the sliver must run at the sliver's
  rate, which is what keeps momentum exact, so the neighbour is kicked at that rate too.
  At A1's own substep (the GPU's power-iteration true step):

  | Pack | Bond evaluations | Chunk updates | A1's column |
  |---|---|---|---|
  | houses | ×2.1–2.2 | ×1.4–1.5 | ×3.0–3.4 |
  | parking-garage, park-432, petronas, villa-savoye, neighbourhood | ×1.6–2.6 | ×1.3–1.9 | ×1.9–3.8 |
  | skyline | ×2.95 | ×2.12 | ×4.34 |
  | minas-tirith | ×15.7 | ×9.1 | ×18.1 |
- **GPU: only batched.** A model on the M3 Max measurements (below) gives:
  - ×1.03–1.3 on one awake house, which is latency-bound: SeisSol's 1.3× case;
  - ×1.7–1.9 on 100 houses batched;
  - ×7–13 on minas-tirith, in proportion to how cheap a launch is (dispatch, CUDA graph
    node or persistent-kernel barrier).
- **Not adopted.** It is a step change (the coarse bonds' truncation error grows with
  their step) and needs the user's decision (AGENTS.md).

## Method

### The integrator

Single rate is symplectic Euler on the hidden 6-DOF chunk motion: at each substep, every
bond is evaluated at the state, every chunk is kicked by `dt` times its forces, then
drifts by `dt` times its new velocity.

Multirate keeps that loop and changes what a bond contributes:

- **Rates.** At the start of each frame (`World::step_frame_once`, `n` substeps of `dt`)
  every chunk gets its own stable step `d_c`, with the safety factor (`chunk_native_dts`):
  - its Gershgorin step (`chunk_stable_dts`);
  - floored at the structure's certified step under `true_step`;
  - combined with the chunk's own contact bound against the stiffest material (the term
    `contact_stable_dt` takes the minimum of), as `substep_dt` combines the global two
    (`1/d² = 1/d_s² + 1/d_c²` under `scaled_step_bound`).
  A bond's rate `r_b` is the largest power of two with `r_b dt ≤ min(d_a, d_b)`
  (`bond_rates`): the rate of its finer chunk.
- **Firing.** A bond is evaluated at substeps `0, r_b, 2 r_b, …` of the frame. When it
  fires, its interval is `w = min(r_b − i mod r_b, n − i)` substeps (to its next firing
  or the frame's end). Its force and moment on both chunks are applied as the impulse of
  that interval, `w dt f`, within the substep. The joint law runs with its own interval
  `w dt` (rate effects, fatigue, the dashpot's dissipation).
- **Every substep, every chunk.** Each chunk takes its external, inertial (floating
  frame) and mass-damping loads and drifts. The world (contacts, events, probes) is
  unchanged.
- **Every frame starts again with all bonds due**, so a frame boundary is a
  synchronisation point and the substep may change between frames.
- **Topology.** Splits keep each bond's schedule. Damage, cracks and splits only lower
  stiffness, so the rates stay stable. Rates are recomputed each frame (after refinement,
  for instance).

This is Lew et al.'s asynchronous variational integrator: each bond is an element with its
own time step; between its events the chunks drift; at an event it delivers its impulse.
External loads form a further element at the finest step. With every element at one step
it is the single-rate integrator, bit for bit (weights of 1).

**Momentum.** A bond's impulse is the single-rate bond force times `w dt`, applied to both
of its chunks at the same moment. Its balance is unchanged: equal and opposite forces, and
moments balanced about the deformed configuration with `corotational_bonds`. No other
term changes. So the momentum budget is single rate's, term by term.

**Energy.** The scheme is variational, so its energy error stays bounded with no drift.
At frame ends every bond's interval closes. There, the half-interval lag of leapfrog's
velocities makes the energy reading first order, with the coarse bonds' interval in place
of the substep. Elastic energy is read at the current state for bonds waiting for their
next evaluation (`elastic_energy`).

**Rule for interface bonds.** The bond between a sliver and a heavy neighbour runs at the
sliver's rate. The sliver's stability depends on that bond's stiffness against its own
small mass, and the impulse must reach both chunks at once. So the heavy neighbour is
kicked at the fine rate: as a GPU chunk kernel sees it, it is a fine chunk. That is the
gap between A1's per-chunk column and the bond rule.

**Not covered.**
- Implicit mode: the explicit windows hand Newmark per-substep accelerations, so the
  method is inert there.
- The engine-coupled frame (`step_frame_coupled`) runs single rate.
- `stress-gpu` rejects the switch (`methods.rs`).

### Tools

```bash
cargo test --release -p stress-ref --test multirate                     # 7 tests, ~3 min
cargo run --release --example multirate_work -- [--methods true_step] <scene | pack> ...
cargo run --release --example multirate_scenes -- OUT 0.5 packs/house-1story.json ...
python3 scripts/fidelity.py --bin target/release/stress-ref --b multirate --out OUT [--scenes OUT]
python3 scripts/fidelity.py ... --b-set sim.methods.multirate=true      # scenes that pin sim.methods (Shardwright)
target/release/stress-gpu-steps --levers <packs>                         # A1 with the "bond rule" line
python3 findings/multirate_gpu_model.py
```

- `fidelity.py` gained `--scenes DIR` and a bond-evaluation cost column. Observations
  carry `bond_evaluations`.
- Scenes that set `sim.methods` themselves ignore `STRESS_METHODS`; the Shardwright
  import sets it (layer contact off), hence `--b-set` there.

## Conservation (`tests/multirate.rs`)

The test body is a free concrete wall of 0.1 m cubes in zero gravity (10 × 2 × 10),
spinning slowly. Its fifth column is slivers: concrete at 1/25 of its density, bonded
with concrete, so their own step is a fifth of the others'. A steel sphere strikes beside
the slivers. Bonds take rates 1 (the slivers' and their neighbours') and 4, and multirate
makes 36 % of single rate's bond evaluations.

| Test | Multirate | Single rate, same scene |
|---|---|---|
| Linear momentum, 40 m/s impact with fracture | drift ≤ 2.2e-4 of the round-off bound | (`conservation.rs`: to round-off) |
| Angular momentum, co-rotational, 40 m/s, against single rate | difference 2.3e-7, 7.7e-8, 2.9e-8, 1.2e-8 of `\|L0\|` at safety 0.5…0.0625: → 0 | own drift 1.5e-5, constant |
| Energy, elastic impact at 5 m/s, safety 0.5…0.016 | −8.6e-4 … −5.1e-5: first order → 0 | −1.8e-3 … −5.6e-5 |
| Energy, free undamped vibration, frames of 64 substeps | 7.1e-4, 3.8e-4, 1.9e-4, 9.4e-5: → 0 | 2.2e-4 … 3.0e-5 |
| Energy drift, free vibration at safety 0.9, 1.6e5 substeps | first tenth 1.5e-3, last tenth 1.0e-3: no drift | 4.3e-4 → 4.9e-4 |
| One rate everywhere (uniform wall, fracture) | bit-identical to single rate | |
| Rates | slivers and their bonds at 1, cubes at 4 | |

**Found on the way (single rate, pre-existing).** On this wall the angular momentum of
single rate drifts by 1.5e-5 of `|L0|` whatever the substep: 1.7e-5 without spin, and a
geometric sliver column instead of a light one gives about 1e-5. On the uniform wall
(`conservation.rs`) the drift vanishes with the substep. So the drift needs unequal chunk
masses. Unexplained; see follow-ups. Multirate shares it exactly, which is why its test is
differential.

**Impulse-method resonance** (Biesiadecki and Skeel 1993) is the known risk of
multiple-time-stepping: a slow force applied as an impulse at a period near a multiple of
half a fast period. The 0.9-safety run is the test for it. Its energy error does not grow.
The free-vibration test at frames of 494 substeps showed one artefact of measurement, not
of the scheme: with an odd number of substeps per frame a rate-2 bond's last interval is
one substep, which changes the frame-end reading. Hence fixed frames of 64.

## Fidelity (`scripts/fidelity.py`, A single rate, B multirate)

### Catalogue (35 scenes, default methods)

34/35 pass every gameplay metric. Multirate acts on five scenes; the others have every
bond at rate 1 and are bit-identical.

| Scene | Bond evaluations B/A | Gameplay | Outcome A → B |
|---|---|---|---|
| b6_spall | ×0.65 | all ok | 432 → 432 broken, 5 → 5 fragments, debris 2.65 → 2.64 m/s; spall speed 3.59 → 3.88 m/s (below) |
| s_house_car | ×0.52 | `where` DIFF | 123 → 122 fragments, debris 0.79 → 0.70 m/s, 315 → 429 detached (Jaccard 0.73) |
| s_house | ×0.97 | all ok | intact |
| s_house_blast | ×0.97 | all ok | 59 → 59 fragments |
| s_house_settlement | ×0.98 | all ok | 47 → 49 fragments |

CPU over the catalogue: A 809 s, B 804 s. The five scenes are a small share of it.

**s_house_car is chaotic at this level.** Detached sets of single rate and multirate at
safety 0.5, 0.45 and 0.4 (Jaccard, the `where` test passes at ≥ 0.8):

| | A 0.45 | A 0.40 | B 0.50 | B 0.45 | B 0.40 |
|---|---|---|---|---|---|
| A 0.50 (315 detached) | 0.76 | 0.77 | 0.73 | 0.81 | 0.74 |
| A 0.45 (352) | | 0.86 | 0.82 | 0.82 | 0.82 |
| A 0.40 (409) | | | 0.95 | 0.71 | 0.96 |

Single rate fails its own `where` test against itself 10 % and 20 % finer. Detached
counts: single rate 315–409, multirate 327–429.

**b6_spall: spall speed (m/s), not a fidelity metric, against the substep:**

| Safety | 0.5 | 0.45 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|---|
| single rate | 3.586 | 3.580 | 3.600 | 3.613 | 3.620 |
| multirate | 3.878 | 3.881 | 3.582 | 3.600 | 3.614 |

- b6's bonds take rates 1 (1,488) and 2 (3,936), so most of the bar runs at twice the
  substep.
- At 0.5 the spall layer leaves 8 % faster. Single rate moves 0.4 % from 0.5 to 0.25.
- From 0.25 the two runs converge together.
- So on a wave problem multirate at the default safety is less accurate than single rate
  at the same safety. At safety 0.25 it costs 1.27× single rate's bond work at 0.5
  (twice the substeps, ×0.65 the bonds), so b6 gains nothing.

### On `true_step` (A `true_step`, B `true_step,multirate`)

35/35 pass every metric. On the certified step the houses' bonds all stay at rate 1, and
only b6_spall takes coarse rates: ×0.65 bond evaluations, the same outcome as above.
CPU A 697 s, B 695 s.

### Packs (0.5 s each, default methods)

Generated by `examples/multirate_scenes`. Each pack is rammed by stress-gpu's ram (400 kg
at 12 m/s into the −x face). The Shardwright two-storey building (L2, 1,182 chunks; baked
and exported per `HANDOFF_SHARDWRIGHT.md`) is rammed and hit by frac-sim's projectile.

| Scene | Bond evaluations B/A | Gameplay | Outcome A → B |
|---|---|---|---|
| house-1story rammed | ×0.21 | all ok | 209 → 209 broken, 40 → 39 fragments, debris 0.74 → 0.72 m/s, first crack the same |
| house-2story rammed | ×0.23 | all ok | 216 → 216 broken, 37 → 40 fragments, debris 0.03 → 0.04 m/s, first crack the same |
| villa-savoye rammed | ×0.25 | all ok | 323 → 306 broken, 21 → 22 fragments, debris 1.13 → 1.31 m/s, first crack the same |
| Shardwright projectile | ×0.88 | all ok | 18 → 18 broken, 10 → 10 fragments, debris 3.14 → 3.34 m/s |
| Shardwright ram | ×0.92 | **void**: `where`, `pieces`, `debris` DIFF against a reference that creates energy | 2,321 → 480 broken, 555 → 89 fragments, debris 8.57 → 2.15 m/s |

**The Shardwright ram cannot be judged.** Its single-rate reference creates energy, as
`SHARDWRIGHT_DESTRUCTION.md` found for this case. It books 3.65e9 J of contact
dissipation against a 28.8 kJ ram (the building's potential is of order 1e6 J):

| Run | Broken | Fragments | Contact dissipation (J) | Bond dissipation (J) |
|---|---|---|---|---|
| single rate, safety 0.5 | 2,321 | 555 | 3.65e9 | 6.31e5 |
| single rate, 0.45 | 1,830 | 363 | 1.69e8 | 3.40e5 |
| single rate, 0.25 | 1,580 | 323 | 3.22e8 | 2.90e5 |
| multirate, 0.5 | 480 | 89 | 4.00e4 | 3.11e4 |
| multirate, 0.45 | 473 | 80 | 3.73e4 | 3.12e4 |
| multirate, 0.25 | 691 | 113 | 6.93e4 | 6.12e4 |

- Single rate blows up at every safety tried. Multirate's books stay at the scale of the
  ram's energy.
- That is no evidence that multirate is right here: the blow-up is the compound-hull
  penalty contact's, which multirate leaves unchanged.
- Why multirate's runs escape it is not known (follow-ups). The projectile case, which
  does not blow up, passes.

Wall time on one thread: house-1story 10,320 → 9,751 s, house-2story 13,090 → 9,097 s,
under varying machine load. The standalone world's debris contacts dominate these runs.
The houses' multirate sides were run beside `fidelity.py` (same binary, settings and
`judge`), to halve the wait.

## Work per frame

### At the oracle's own step (`examples/multirate_work`)

Per 60 Hz frame, after the prestress, from the rates the integrator assigns. Ratios are
single rate over multirate.
- **Bonds:** bond evaluations.
- **Chunks:** chunk updates, if each chunk is updated at its finest bond's rate (the GPU
  design; the oracle drifts every chunk every substep).
- **A1-style:** A1's column at this substep.
- **W/o contact:** the bond ratio without the chunks' contact bounds.

Default methods (Gershgorin step):

| Pack | Substeps | Binds | Bonds | Chunks | A1-style | W/o contact | Bonds per rate |
|---|---|---|---|---|---|---|---|
| house-1story | 33,720 | stress | ×4.56 | ×2.27 | ×8.46 | ×6.09 | 1:113 2:66 4:76 8:38 16:137 32:447 |
| house-2story | 35,595 | stress | ×4.23 | ×2.15 | ×6.64 | ×5.74 | 1:251 2:140 4:35 8:127 16:397 32:584 64:118 |
| villa-savoye | 19,537 | contact | ×4.05 | ×2.45 | ×4.31 | ×10.26 | 1:250 2:232 4:287 8:1038 16:657 |
| veneer-house-l0 | 609,909 | stress | ×3.18 | ×1.77 | ×6.89 | ×3.20 | 1:68 … 256:4 |
| Shardwright two-storey (L2) | 3,170 | stress | ×1.13 | ×1.05 | ×1.33 | ×1.89 | 1:2797 2:798 4:40 |
| rig-portal | 2,174 | stress | ×1.26 | ×1.00 | ×1.57 | ×1.64 | 1:23 2:16 |
| brick-building, comp-wall-bay, rig-toppled | | contact | ×1.00 | ×1.00 | ×1.00–1.04 | ×1.04–1.73 | all 1 |

With `true_step`, the oracle's certified bound sets each cluster's stress step and the
global contact bound binds every pack but the Shardwright building:

| Pack | Substeps | Bonds | Chunks | A1-style | W/o contact |
|---|---|---|---|---|---|
| house-1story | 18,567 | ×3.48 | ×1.73 | ×5.70 | ×7.47 |
| house-2story | 22,390 | ×3.41 | ×1.71 | ×4.96 | ×4.97 |
| villa-savoye | 16,016 | ×3.75 | ×2.38 | ×4.06 | ×9.54 |
| veneer-house-l0 | 72,121 | ×1.24 | ×1.13 | ×1.56 | ×1.26 |
| Shardwright two-storey | 3,170 | ×1.13 | ×1.05 | ×1.33 | ×1.89 |

Measured, house-1story rammed, 3 frames, one thread (`--profile`):

| | Single rate | Multirate |
|---|---|---|
| Bond evaluations | 80.8 M | 17.5 M (×4.60) |
| Bond law | 27.7 s | 6.5 s |
| Chunk frame loads and integration | 6.4 s | 5.7 s |
| Solver substep in all | 37.2 s | 14.8 s (×2.51) |
| Contacts (standalone debris) | 180.8 s | 154.0 s |

### At A1's substep (`stress-gpu-steps --levers`, bond rule line)

This is A1's own setting:
- the power-iteration true step (+3 %) at safety 0.9, no contact term (the engine-coupled
  path has none outside impact islands);
- the bond rule evaluated at that substep, with Gershgorin chunk steps at the same safety.

| Pack | Chunks | True substeps | A1 multirate (per chunk) | Bond rule: bond evaluations | Bond rule: chunk updates | Bonds per rate |
|---|---|---|---|---|---|---|
| brick-building | 248 | 179 | ×1.00 | ×1.00 | ×1.00 | all 1 |
| house-1story | 319 | 1,264 | ×3.36 | ×2.14 | ×1.40 | 1:252 2:98 4:351 8:176 |
| house-2story | 599 | 1,358 | ×3.03 | ×2.21 | ×1.50 | 1:443 2:218 4:586 8:401 16:4 |
| parking-garage | 3,177 | 934 | ×2.71 | ×2.34 | ×1.55 | |
| villa-savoye | 1,045 | 931 | ×1.90 | ×1.62 | ×1.31 | 1:1062 2:611 4:438 8:353 |
| neighbourhood | 8,877 | 1,360 | ×3.76 | ×2.29 | ×1.66 | |
| park-432 | 24,452 | 1,271 | ×2.86 | ×2.44 | ×1.94 | |
| petronas | 24,766 | 1,369 | ×2.89 | ×2.57 | ×1.89 | |
| skyline | 62,525 | 1,944 | ×4.34 | ×2.95 | ×2.12 | |
| minas-tirith | 49,049 | 6,766 | ×18.1 | ×15.66 | ×9.14 | 1:1513 2:4237 4:7921 8:12797 16:12633 32:51665 64:73744 128:4782 256:46 |
| Shardwright two-storey | 1,182 | 586 | ×1.03 | ×1.01 | ×1.00 | 1:3550 2:85 |

- A1's column reproduces exactly.
- The bond rule keeps 61–89 % of it in bond evaluations and 42–69 % in chunk updates.
- The Shardwright building has nothing to gain: its top mode is one 1.4 kg brick-wall
  sliver whose neighbours are themselves near the step.

## GPU design (not built)

**What the oracle method needs from a GPU.**
- The scheme is the oracle's with one change: coarse chunks are updated only at their own
  rate.
- In the oracle every chunk drifts and takes its external loads every substep. On the GPU
  that is a full chunk pass per substep, which would keep one launch per substep for every
  chunk and give back most of the gain.
- For anchored islands (buildings) the chunk loads are gravity, scripted loads and the
  engine's contact impulses. Within a frame these are constant or known in closed form
  (the pulse filter), so their integral over a chunk's interval is exact.
- Free islands (fragments) take inertial loads that change every substep. They run at
  their own finest rate, which still gives per-island stepping.
- This variant is an oracle switch to gate first (`fidelity.py`), like this one.

**Data layout (per frame, all awake islands batched).**
- **Global substep `dt`.** Fix it per scene class (a power-of-two fraction of the frame)
  so that rates are static per asset. Rates then change only when topology does, and are
  conservative after damage.
- **Level buckets.** Bonds and chunks are stored in level order (rate 1 first) per asset,
  with per-level offsets; a chunk's level is its finest bond's. Waking or sleeping an
  island adds or removes its ranges from per-level indirect-dispatch arguments (prefix
  sums over awake islands).
- **Per-island stepping comes free.** An island whose finest chunk allows `4 dt` simply has
  no bonds at levels 1–2.

**Schedule.**
- Substep `i` fires every level whose rate divides `i`. In level order that is a prefix of
  the buckets: one bond dispatch and one chunk dispatch per substep, over a length that
  varies with `i`.
- The schedule is static per frame (`n` and the levels), so it can be captured once and
  replayed:
  - CUDA: a graph of the frame's dispatches, or a cooperative persistent kernel with a
    grid barrier per substep;
  - Metal: an indirect command buffer encoded on the GPU (no grid barrier: forward
    progress is not guaranteed, PLAN.md).
- Splits keep the current design: the island stops and the host splits; each bond keeps
  its schedule.

**Cost model** (`findings/multirate_gpu_model.py`).
- Each substep is one launch of latency L plus its work.
- Throughput is 1.47 ns per bond evaluation (G8: a 24k-chunk island at 139 µs per
  substep, less the 24 µs floor). A chunk update costs 0.58 of a bond (the oracle's
  profile).
- Rates are at the true step (the bond rule above).
- L is 24 µs (today's full-model floor, A6), 5 µs (the intact-kernel target) or 2 µs (a
  graph node or barrier).

| Case | L = 24 µs | 5 µs | 2 µs |
|---|---|---|---|
| 1 house-1story | ×1.03 (32 → 31 ms) | ×1.13 | ×1.27 |
| 10 house-1story | ×1.24 | ×1.59 | ×1.76 |
| 100 house-1story | ×1.74 (228 → 131 ms) | ×1.90 | ×1.93 |
| 100 house-2story | ×1.89 | ×2.00 | ×2.02 |
| 100 villa-savoye | ×1.51 | ×1.54 | ×1.54 |
| minas-tirith | ×7.1 (2.1 s → 301 ms) | ×11.6 | ×13.0 |

- **One house is latency-bound.** Every substep pays L whatever its work, so cutting work
  gains ×1.03–1.3: SeisSol's 1.3× when each cluster is too small to fill the GPU.
- **Batched, the fine level fills the GPU.** 100 houses put 25k bonds in level 1 alone,
  and the gain approaches the bond rule's work ratio (×2.1). That is SeisSol's ×9.75 on
  an A100 with CUDA graphs, scaled to our ratio.
- **Lowering L multiplies with multirate.** On minas-tirith multirate leaves the frame
  latency-bound (half its substeps fire only the 1,513 level-1 bonds), so the graph or
  persistent kernel is what turns ×7 into ×13.
- **Rejected in the model: the fine levels in a persistent threadgroup per building**
  (barriers of 0.1–0.2 µs, A9; the device launches only every fourth substep).
  - The levels below rate 4 hold 40 % of a house's bonds and 5,750 of minas-tirith's, and
    one core runs them about 40× slower than the device.
  - It wins only where launches are expensive and the buildings many: ×1.86 against ×1.74
    for 100 houses at 24 µs.
  - It loses everywhere else: one villa-savoye ×0.27, minas-tirith ×1.3 against ×7.1.
  - It would pay where the fine level is a handful of chunks.

## Follow-ups

- **Adoption is the user's decision.** It is a step change. Its evidence: the gates
  above, and the b6 spall speed at the default safety.
- **The interface rule costs a third of A1's gain** (×2.1 against ×3.4 on house-1story at
  the true step). Sliver treatment at the asset (re-export from Shardwright, or sliver
  elimination as its own switch, A5) removes the slivers instead of subcycling them, and
  would combine with this.
- **The rates come from Gershgorin rows,** looser than the certified bound by about 1.6×
  (`true_step`). A per-chunk certified bound would move more bonds to coarse levels.
- **The chunk-rate external-load variant** (the GPU design) is an oracle switch to gate
  first.
- **Single rate's angular-momentum drift on mass-heterogeneous bodies** (1.5e-5 of `|L0|`,
  not vanishing with the substep; the uniform wall's does) is unexplained.
- **Why multirate escapes the Shardwright ram's contact blow-up** at every safety tried
  (single rate does not) is not known.
- **The engine-coupled frame** (`step_frame_coupled`) could open the same window per
  engine step.
