# Multi-time-step explicit–implicit coupling for struck structures

Written 2026-10-16. Follow-up to `avbd-stress.md` and the literature review
(`stress-literature-review.md`, branch `claude/interesting-robinson-21197b`, §3).

**Question.** An explicit impact window integrates every chunk of a struck structure at
the true step. Can it run only a *hot zone* explicitly at its true step and leave the rest
to a much larger Newmark step, as in Gravouil–Combescure (GC) multi-time-step domain
decomposition? The rest must stay nonlinear, so that it can still crack. Any lossy
variant is measured here, not adopted.

## Verdict

**No for the gate cases; at most break-even on the house.** Measured on CPU, f64, one
thread, against the explicit truth's own spread:

- **A sound trigger makes the whole b5 wall and the free wall explicit within about 1 ms
  of contact.** In the explicit truth, 79–91 % of b5's bonds and 85 % of the free wall's
  pass utilisation 0.5, and 47–79 % pass 1.0 (the onset of softening). About half of b5's
  bonds pass 0.5 within 1 ms: the impact's stress wave loads most of a 2 × 2 m wall to
  near its strength. A perfect-foresight trigger (each bond hot one coarse step before the
  truth first takes it past 0.5) still needs 78–90 % of b5's explicit work and 85 % of the
  free wall's. That bounds any sound hot zone on these cases to **at most a 1.1–1.3×
  saving**, before the exterior's own cost. Measured with the prototype: hot zone = every
  chunk by +1 ms, explicit work 1.0–1.4× the full explicit solve (look-ahead passes and
  rollbacks included), wall time 1.1–6.8×.
- **On the house the hot zone stays partial**: 240 of 300 chunks by +20 ms, 61 % of the
  truth's bonds below 0.5 to 40 ms. With the nonlinear Newmark exterior, the sound
  trigger **passes the gate at m = 4, 30 and 100** (h up to 130 µs, 100× the true step).
  No accepted step had a miss. Explicit work is 0.51–0.60 of the full solve from first
  contact. Wall time from contact: 1.92× at m = 4, 1.09× at m = 30, **0.78× at m = 100**.
  So on CPU it is a ~1.3× gain at best. On a GPU it is worse: a sparse direct
  factorisation per repartition (86–136 of them) against a 2× cut in explicit bond
  passes.
- **Fidelity under the sound trigger is mostly inside the truth's spread, never better
  than it.** Results:
  - b5 10–30 m/s all pass at m = 30; at m = 4 only 15 and 30 pass, at m = 100 only 20;
  - b5 40 m/s fails at every ratio (448–514 chunks out against 32–386: extra push-over);
  - the free wall and the house pass at every ratio.
  The failures come where the stress wave crosses several rings in one coarse step, so
  rollbacks run out (m = 30 and 100 on b5). The wave sets the step: h ≲ (ring width) /
  (wave speed) ≈ 0.1 m / 3,500 m/s ≈ 30 µs while the wave is in the structure.
- **An exterior that is allowed to crack is not viable.** With contact closure alone
  (distance-like escalation, no utilisation trigger), the Newmark exterior at 256 µs
  softens bonds far from the impact (b5's base and mid-wall) and diverges on all five b5
  speeds and the free wall. It passes the house (11 exterior cracks).
- **The linear Craig–Bampton exterior is worse than Newmark everywhere.** It passes the
  free wall and b5 10, 15 and 30 at m = 4 and 30, and fails b5 20 and 40. It fails the
  house badly at m = 4 and 30 (598–762 broken against 154–159), and more modes do not
  help (10 / 40 / 150 modes: 640 / 598 / 677). Its ring copies drift 4 mm apart there,
  against ≤ 0.02 mm for Newmark on the house (≤ 0.4 mm on b5). It costs 2–180× the truth, because every
  repartition and rollback is a new reduction (constraint modes, then a shift-inverted
  subspace iteration).
- **The coupling itself needed care** (§ Coupling). As commonly stated (interface
  operator weighted by k/m), GC is weakly unstable: spectral radius 1.001–1.005 per coarse
  step. The constant-operator form drains a cantilever's vibration to zero. The form used
  here carries the last multiplier as a load in the exterior's free problem. It is
  non-expansive at every ratio and tracks the cantilever (f₁ within 0.02 %), losing
  1–7 % of the energy over 2 s through GC's interface dissipation.

**For the in-engine design:** keep impact windows fully explicit for walls and houses of
this size. The scheme can pay only where the damaged region is a small part of a large
structure, and even then the gain is bounded by the hot fraction (≈ 2× fewer bond
passes on the house). That gain is paid for with direct factorisations, rollbacks and an
interface solve per substep. The cheaper levers stay the roadmap's: per-island stepping,
the contact skin list (review §5), and multirate explicit–explicit (review §2). The
latter keeps one integrator family and no factorisations.

## Method

**Prototype:**
- `crates/stress-ref/src/avbd/mts.rs` (`MtsWorld`) and `avbd/skyline.rs` (a profile
  Cholesky with reverse Cuthill–McKee);
- driver `examples/mts_compare.rs`;
- all behind the new `mts` feature (`faer` for the small dense eigensolves).
Default builds and behaviour are unchanged. Edits outside the new files:
- `AvbdWorld` derives `Clone`;
- `AvbdWorld` has a `detect_mask` field, `None` by default, that the detector consults.

**World.** The AVBD study's chunk-level world (`avbd.rs`), which reproduces the oracle's
established methods:
- the oracle's damageable joints with true stiffness and co-rotational kinematics;
- sample-point penalty contact;
- clusters as connected components;
- its explicit integrator, the truth replica (`avbd-stress.md` validates it against the
  oracle).

**Partition by bonds.**
- *Core*: escalated chunks.
- *Hot zone*: the core, a one-bond ring around it, the impactors and every contact. It
  runs the replica's explicit substep at the certified true step (`methods.true_step`):
  8.55 µs on b5 and the free wall, 1.30 µs on the house.
- *Exterior bonds*: those with a free end outside the hot zone.
- *Ring chunks* carry bonds of both kinds and are duplicated. Mass and inertia are split
  by each zone's share of the chunk's axial bond stiffness, so each copy keeps the
  chunk's k/m and neither zone's stability changes. One 6-DOF multiplier per ring chunk
  holds the two copies to one velocity.

**Exterior.**
- `Newmark` (default): average acceleration at h = m·dt on every exterior chunk (as
  `implicit.rs`). It is nonlinear: each bond's full law is evaluated at the trial
  deformation from its step-start history, so damage enters inside the iteration (frozen
  damage fails, `avbd-stress.md`). It is solved by modified Newton on the true residual
  (tolerance 1e-8) with a profile Cholesky of `K_t + 2/h C + 4/h² M`, refactored when the
  residual stalls or the partition changes. An indefinite tangent is retried with a
  raised diagonal; past that the run is marked diverged.
- `CraigBampton`: the exterior reduced about its state at each repartition to the
  interface DOFs (static constraint modes) plus 40 fixed-interface modes (shift-inverted
  subspace iteration, M-orthonormalised). It is linear, so it cannot crack; its bonds'
  utilisation is evaluated from the reconstructed state, for the trigger only.

**Coupling (GC with a multiplier predictor).** Per coarse step from t_n:
1. The hot zone's force pass at t_n.
2. The exterior's *free* Newmark step, with the multiplier at t_n, λ₀, held as its
   interface load.
3. For each substep k (time t_k), the condensed interface problem
   `(dt/2 D + S) λ_k = (1 − k/m) V_B(t_n) + (k/m) V_B^free + S λ₀ − v_A^free(t_k)`:
   - continuity of the hot copies' integer-time velocity (central-difference form of the
     symplectic substep) with the exterior's interpolated free velocity, less its full
     link response;
   - `D` is the hot copies' inverse mass, `S = 2/h C K_eff⁻¹ Cᵀ`;
   - `S` is constant over the coarse step: one symmetric eigendecomposition per
     factorisation makes each substep's solve two dense mat-vecs.
4. A look-ahead force pass at t_{n+1}, evaluated but not committed and identical to the
   next substep's own, gives the end multiplier.
5. The exterior's *link* step with λ_end − λ₀; the exterior's bonds are committed at the
   accepted state.

At a repartition the ring copies are merged at the integer time (momentum-weighted). The
first multiplier is then the one that gives each hot copy its exterior copy's
acceleration.

**Escalation** into the core at each coarse step's start, never back:
- *contact closure*: anything within two coarse steps of travel of touching anything;
- *detachment* from the main piece, or any damaged bond;
- *utilisation and rate*: a bond whose utilisation, extrapolated two coarse steps at its
  current rate, plus an influence bound, reaches **u_hot = 0.5**. The influence bound is
  `Σ_c |∂u/∂d_c| Σ_i |(J K_eff⁻¹ Cᵀ)_ci| Δλ_i`, with Δλ twice the last coarse step's
  multiplier change.

The test runs on every exterior bond's own solved stress. A base bond loaded through a
moment arm (leverage) is therefore caught by its own utilisation and rate, not by its
distance from the impact.

**Verification (sound by construction).** After each coarse step, any of these is a
*miss*: an exterior bond at u ≥ u_hot, a damaged exterior bond, or a contact touching the
exterior. `Verify::Rollback` restores the step's start, forces the missed chunks into the
core and redoes the step (up to 60 times). No accepted step then has a crack or a contact
outside the hot zone, unless the 60 attempts run out; the cost columns count those
attempts. "Exterior may crack" (`--u-hot 1e9 --verify count --no-influence`) keeps only
contact closure and detachment, the distance-like trigger.

**Cost accounting.**
- *Explicit work*: bond evaluations in the hot zone, look-ahead passes and rolled-back
  attempts included, over the full explicit solve's (every active bond, every substep),
  from the hot zone's first chunk.
- *Wall time*: one thread, from the same instant, against the all-explicit run of the
  same code (`--all-hot`, bit-identical to the replica; a unit test checks it).
- *Exterior work* is reported separately: Newton iterations, factorisations, rollbacks.

**Gate.** Each case is gated against the spread of three truths: the oracle (established
methods), the replica at the oracle substep, and the replica at the true step. Outcomes
are read at the scene's end (house: 40 ms).
- ✓ inside the truth spread; ~ within a quarter of the spread (or 0.2 % of the impact
  speed for the ram, 10 % of the count for chunks out) beyond it; ✗ outside.
- A run **passes** when chunks out, fast chunks and ram exit speed are not ✗ and it did
  not diverge. These are the regime metrics; broken-bond counts are chaotic at the
  regime boundary, as `avbd-stress.md` found.
- `where` is `scripts/fidelity.py`'s Jaccard of the detached set: the best against the
  three truths, ≥ 0.8 to pass.
- First cracks agree to the substep in every run: the hot zone starts with the impact,
  so Δ = 0.

## Why the hot zone fills b5: the truth's own utilisation map

`mts_compare --solver umap`: each bond's first time past each utilisation level, in the
replica at the true step.

| case | bonds | u ≥ 0.25 | u ≥ 0.5 | u ≥ 1.0 | u ≥ 0.5 by 0.5 / 1 / 2 ms | first contact |
|---|---|---|---|---|---|---|
| b5 10 | 2,018 | 94 % | 88 % | 61 % | 640 / 1,351 / 1,658 | 0.13 ms |
| b5 20 | 2,018 | 97 % | 91 % | 79 % | 644 / 1,124 / 1,453 | 0.08 ms |
| b5 40 | 2,018 | 91 % | 79 % | 47 % | 613 / 1,068 / 1,399 | 0.04 ms |
| free wall | 508 | 96 % | 85 % | 50 % | (+0.5 ms from contact) 390 by +0.7 ms | 1.27 ms |
| house (to 40 ms) | 877 | 57 % | 39 % | 28 % | 162 by +3 ms, 341 by +23 ms | 16.7 ms |

Perfect-foresight lower bound on explicit work, from first contact to the end. Each
bond is hot from one coarse step h before the truth first passes u_hot. Bond level, with
no ring, no contact margin and no exterior cost:

| case | u_hot 0.25 | u_hot 0.5 | u_hot 1.0 (not sound) |
|---|---|---|---|
| b5 10 / 20 / 40 | 0.94 / 0.96 / 0.91 | 0.88 / 0.90 / 0.78 | 0.61 / 0.76 / 0.46 |
| free wall | 0.96 | 0.85 | 0.50 |
| house | 0.41 | 0.29 | 0.22 |

(Bounds at h = 0.85 ms; they change by ≤ 0.04 from h = 0.1 to 4.3 ms.) The measured
house work (0.51–0.60) is about twice this bound. The difference is the ring, the
contact-closure margin and the trigger's two-step horizon.

## Coupling validation (b2 cantilever, root explicit, tip in the exterior)

Chunks 1–6 in the core, 7 on the ring, 8–20 in Newmark. Fixed partition, tip load
released at 0.05 s, 2 s of free vibration; explicit truth f₁ = 2.7148 Hz.

| coupling | m = 1 | 2 | 10 | 50 | 200 |
|---|---|---|---|---|---|
| midpoint-velocity, momentum-exact "mean multiplier" (first attempt) | unstable | | 2.7147 Hz | unstable | unstable |
| GC, operator weighted by k/m | 2.7148 Hz (exact) | unstable (5.4 Hz) | unstable (41 Hz) | unstable | unstable |
| GC, constant operator, no predictor | exact | unstable | vibration drained to 0 | to 0 | to 0 |
| **GC, constant operator, multiplier predictor (used)** | exact, ΔE −0.01 % | 2.7148, ΔE −1.0 % | 2.7147, −1.7 % | 2.7151, −3.2 % | 2.7142, −7.1 % |
| same with Craig–Bampton (5 or 20 modes) | | | 2.7145–2.7147 | 2.7148–2.7151 | 2.7142 |

Copy drift ≤ 1.4e-5 m; velocity gap ≤ 5e-17 m/s. The exterior alone (no hot zone) gives
2.7138–2.7158 Hz up to m = 200 (3.9 ms). A drift-closing velocity bias
`(x_hot − x_ext)/h` destabilised the coupling and is off by default.

On a 1-D spring chain (explicit central difference against trapezoidal Newmark, ring
mass split 0.5), the spectral radius of one coarse step is:

| coupling | m = 1 | 2 | 3 | 5 | 10 | 50 |
|---|---|---|---|---|---|---|
| operator weighted by k/m | 1.0000 | 1.0011–1.0017 | 1.0000–1.0019 | 1.0006–1.0010 | 1.0012–1.0039 | 1.0000–1.0053 |
| constant operator | 1.0000 | 1.0001–1.0020 | 1.0000–1.0001 | 1.0000 | 1.0000 | 1.0000 |
| constant + predictor | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

(Ranges over dt = 0.1–0.95 of the explicit limit.) The constant operator without the
predictor is inconsistent for a sustained interface force. Its free problem omits the
force, and the constant operator then drives the hot side by (1 − k/m)·Sλ: harmless for
the chain's pulse, fatal for the cantilever's sustained bending moment. Two unit tests
(`--features mts`) cover the essentials: all-hot is bit-identical to the replica, and
the coupling tracks the cantilever to 2 % of the amplitude.

## Gate

Truth spread (broken / out (fast) / ram exit speed in m/s, along the impact):

| case | oracle (2.5 µs) | replica, oracle substep | replica, true step | true step |
|---|---|---|---|---|
| b5 10 | 249 / 800 (40) / 8.36 | 350 / 800 (40) / 7.95 | 314 / 800 (47) / 8.10 | 8.55 µs |
| b5 15 | 437 / 800 (54) / 12.42 | 380 / 800 (52) / 12.46 | 382 / 800 (50) / 12.48 | 8.55 µs |
| b5 20 | 541 / 630 (50) / 17.00 | 483 / 566 (50) / 17.09 | 485 / 664 (48) / 17.02 | 8.55 µs |
| b5 30 | 123 / 480 (32) / 26.98 | 310 / 800 (34) / 26.54 | 567 / 800 (38) / 26.26 | 8.55 µs |
| b5 40 | 64 / 32 (32) / 36.00 | 71 / 32 (32) / 35.85 | 209 / 386 (32) / 35.85 | 8.55 µs |
| free wall | 24 / 8 (8) / 35.05 | 27 / 8 (8) / 34.75 | 25 / 8 (8) / 34.69 | 8.55 µs |
| house (40 ms) | 154 / 39 (0) / 3.26 | 155 / 43 (0) / 2.86 | 159 / 44 (0) / 2.64 | 1.30 µs |

Each cell: broken / out (fast) / ram, each marked against the spread, then `where`.

| case | Newmark, sound, m 4 | Newmark, sound, m 30 | Newmark, sound, m 100 | Newmark, exterior may crack, m 30 | Craig-Bampton, sound, m 4 | Craig-Bampton, sound, m 30 |
|---|---|---|---|---|---|---|
| b5 10 | 341✓ / 800✓ (52✗) / 8.12✓, where 1.00: **fail** | 282✓ / 800✓ (40✓) / 8.31✓, where 1.00: **pass** | 484✗ / 800✓ (52✗) / 8.03✓, where 1.00: **fail** | 195~ / 62✗ (40✓) / 8.44~, where 0.08: **diverged** | 276✓ / 800✓ (40✓) / 8.34✓, where 1.00: **pass** | 272✓ / 800✓ (48~) / 8.28✓, where 1.00: **pass** |
| b5 15 | 416✓ / 800✓ (48~) / 12.51~, where 1.00: **pass** | 352~ / 800✓ (50✓) / 12.49~, where 1.00: **pass** | 429✓ / 448✗ (50✓) / 12.51✗, where 0.56: **fail** | 54✗ / 32✗ (32✗) / 13.29✗, where 0.04: **diverged** | 360~ / 800✓ (48~) / 12.47✓, where 1.00: **pass** | 391✓ / 800✓ (50✓) / 12.42✓, where 1.00: **pass** |
| b5 20 | 368~ / 800✗ (42✗) / 17.33✗, where 0.83: **fail** | 485✓ / 514~ (48✓) / 17.07✓, where 0.91: **pass** | 511✓ / 550~ (50✓) / 17.01✓, where 0.92: **pass** | 64✗ / 32✗ (32✗) / 18.11✗, where 0.06: **diverged** | 516✓ / 592✓ (43~) / 17.26✗, where 0.94: **fail** | 245✗ / 428✗ (42✗) / 17.54✗, where 0.76: **fail** |
| b5 30 | 134✓ / 480✓ (32✓) / 26.89✓, where 1.00: **pass** | 132✓ / 480✓ (32✓) / 26.87✓, where 1.00: **pass** | 591~ / 600✓ (48✗) / 26.16~, where 0.78: **fail** | 48~ / 32✗ (32✓) / 27.58✗, where 0.07: **diverged** | 542✓ / 707✓ (36✓) / 26.33✓, where 0.88: **pass** | 124✓ / 480✓ (32✓) / 26.90✓, where 1.00: **pass** |
| b5 40 | 182✓ / 448✗ (32✓) / 35.84~, where 0.86: **fail** | 290✗ / 454✗ (36✗) / 35.84~, where 0.85: **fail** | 541✗ / 514✗ (42✗) / 35.24✗, where 0.75: **fail** | 48~ / 32✓ (32✓) / 36.53✗, where 1.00: **diverged** | 272✗ / 800✗ (32✓) / 35.71✗, where 0.48: **fail** | 340✗ / 538✗ (34~) / 35.41✗, where 0.72: **fail** |
| free wall | 36✗ / 8✓ (8✓) / 34.61~, where 1.00: **pass** | 24✓ / 8✓ (8✓) / 34.69✓, where 1.00: **pass** | 32~ / 8✓ (8✓) / 34.69✓, where 1.00: **pass** | 20~ / 8✓ (8✓) / 34.78✓, where 1.00: **diverged** | 37✗ / 8✓ (8✓) / 34.61~, where 1.00: **pass** | 26✓ / 8✓ (8✓) / 34.69✓, where 1.00: **pass** |
| house | 160~ / 44✓ (0✓) / 2.64✓, where 1.00: **pass** | 161~ / 44✓ (0✓) / 2.65✓, where 1.00: **pass** | 154✓ / 44✓ (0✓) / 2.88✓, where 1.00: **pass** | 159✓ / 44✓ (0✓) / 2.64✓, where 1.00: **pass** | 762✗ / 281✗ (25✗) / 3.69✗, where 0.16: **fail** | 598✗ / 192✗ (0✓) / 3.52✗, where 0.23: **fail** |

**Accepted misses** (steps accepted after 60 rollbacks without a clean attempt, and the
chunks missed):
- Newmark, m = 4: none in any case.
- Newmark, m = 30: one step on b5 20–40 and the free wall (14–62 chunks).
- Newmark, m = 100: one step on every b5 case and the free wall (56–132 chunks).
- House: none at any ratio.
- Craig–Bampton: one step in most cases.

In every case the step that ran out is the first one after contact, while the stress
wave crosses the structure.

## Cost

From the hot zone's first chunk, rollbacks and look-ahead passes included. h is the coarse step, with the ratio m in brackets.

| case | variant | h (m) | explicit work | wall vs truth | hot chunks at +0.5 / 1 / 5 ms / end (of) | Newton its | factorisations | rollbacks |
|---|---|---|---|---|---|---|---|---|
| b5 10 | Newmark, sound, m 4 | 34 µs (4) | 1.20 | 2.11× | 572 / 800 / 800 / 800 (800) | 48 | 24 | 1 |
| b5 10 | Newmark, sound, m 30 | 256 µs (30) | 1.00 | 1.13× | 446 / 800 / 800 / 800 (800) | 10 | 6 | 3 |
| b5 10 | Newmark, sound, m 100 | 855 µs (100) | 1.23 | 3.48× | 446 / 800 / 800 / 800 (800) | 61 | 61 | 60 |
| b5 10 | Newmark, exterior may crack, m 30 | 256 µs (30) | 0.43 | 0.67× | 320 / 634 / 634 / 634 (800) | 149 | 54 | 0 | diverged
| b5 10 | Craig-Bampton, sound, m 4 | 34 µs (4) | 1.23 | 7.88× | 524 / 800 / 800 / 800 (800) | 0 | 97 | 72 |
| b5 10 | Craig-Bampton, sound, m 30 | 256 µs (30) | 1.28 | 16.78× | 522 / 714 / 800 / 800 (800) | 0 | 126 | 122 |
| b5 15 | Newmark, sound, m 4 | 34 µs (4) | 1.20 | 1.99× | 608 / 800 / 800 / 800 (800) | 47 | 21 | 1 |
| b5 15 | Newmark, sound, m 30 | 256 µs (30) | 1.00 | 1.07× | 634 / 800 / 800 / 800 (800) | 8 | 5 | 2 |
| b5 15 | Newmark, sound, m 100 | 855 µs (100) | 1.18 | 2.12× | 352 / 800 / 800 / 800 (800) | 61 | 61 | 60 |
| b5 15 | Newmark, exterior may crack, m 30 | 256 µs (30) | 0.18 | 0.39× | 176 / 248 / 248 / 248 (800) | 70 | 24 | 0 | diverged
| b5 15 | Craig-Bampton, sound, m 4 | 34 µs (4) | 1.21 | 8.69× | 550 / 800 / 800 / 800 (800) | 0 | 40 | 15 |
| b5 15 | Craig-Bampton, sound, m 30 | 256 µs (30) | 1.51 | 42.78× | 340 / 640 / 800 / 800 (800) | 0 | 246 | 241 |
| b5 20 | Newmark, sound, m 4 | 34 µs (4) | 1.19 | 2.27× | 648 / 800 / 800 / 800 (800) | 45 | 23 | 0 |
| b5 20 | Newmark, sound, m 30 | 256 µs (30) | 1.11 | 3.35× | 656 / 800 / 800 / 800 (800) | 65 | 63 | 60 |
| b5 20 | Newmark, sound, m 100 | 855 µs (100) | 1.39 | 4.96× | 272 / 798 / 800 / 800 (800) | 67 | 63 | 61 |
| b5 20 | Newmark, exterior may crack, m 30 | 256 µs (30) | 0.16 | 2.77× | 136 / 227 / 227 / 227 (800) | 121 | 55 | 0 | diverged
| b5 20 | Craig-Bampton, sound, m 4 | 34 µs (4) | 1.27 | 20.18× | 560 / 800 / 800 / 800 (800) | 0 | 92 | 68 |
| b5 20 | Craig-Bampton, sound, m 30 | 256 µs (30) | 1.69 | 129.38× | 368 / 634 / 800 / 800 (800) | 0 | 194 | 189 |
| b5 30 | Newmark, sound, m 4 | 34 µs (4) | 1.20 | 2.23× | 620 / 800 / 800 / 800 (800) | 46 | 23 | 1 |
| b5 30 | Newmark, sound, m 30 | 256 µs (30) | 1.09 | 4.72× | 658 / 800 / 800 / 800 (800) | 65 | 63 | 60 |
| b5 30 | Newmark, sound, m 100 | 855 µs (100) | 1.42 | 5.63× | 278 / 800 / 800 / 800 (800) | 64 | 62 | 61 |
| b5 30 | Newmark, exterior may crack, m 30 | 256 µs (30) | 0.12 | 0.74× | 136 / 136 / 136 / 136 (800) | 61 | 19 | 0 | diverged
| b5 30 | Craig-Bampton, sound, m 4 | 34 µs (4) | 1.27 | 19.78× | 566 / 800 / 800 / 800 (800) | 0 | 95 | 69 |
| b5 30 | Craig-Bampton, sound, m 30 | 256 µs (30) | 1.88 | 177.15× | 376 / 650 / 800 / 800 (800) | 0 | 247 | 242 |
| b5 40 | Newmark, sound, m 4 | 34 µs (4) | 1.20 | 3.04× | 642 / 800 / 800 / 800 (800) | 48 | 24 | 0 |
| b5 40 | Newmark, sound, m 30 | 256 µs (30) | 1.10 | 5.84× | 598 / 800 / 800 / 800 (800) | 66 | 63 | 60 |
| b5 40 | Newmark, sound, m 100 | 855 µs (100) | 1.41 | 6.84× | 280 / 800 / 800 / 800 (800) | 61 | 61 | 60 |
| b5 40 | Newmark, exterior may crack, m 30 | 256 µs (30) | 0.12 | 0.71× | 136 / 136 / 136 / 136 (800) | 61 | 16 | 0 | diverged
| b5 40 | Craig-Bampton, sound, m 4 | 34 µs (4) | 1.27 | 22.73× | 598 / 800 / 800 / 800 (800) | 0 | 95 | 69 |
| b5 40 | Craig-Bampton, sound, m 30 | 256 µs (30) | 1.68 | 152.60× | 340 / 638 / 800 / 800 (800) | 0 | 192 | 187 |
| free wall | Newmark, sound, m 4 | 34 µs (4) | 1.25 | 2.29× | 220 / 220 / 220 / 220 (220) | 486 | 223 | 0 |
| free wall | Newmark, sound, m 30 | 256 µs (30) | 1.05 | 1.53× | 72 / 220 / 220 / 220 (220) | 122 | 65 | 61 |
| free wall | Newmark, sound, m 100 | 855 µs (100) | 1.08 | 4.46× | 48 / 172 / 220 / 220 (220) | 3615 | 1808 | 60 |
| free wall | Newmark, exterior may crack, m 30 | 256 µs (30) | 0.18 | 0.03× | 48 / 80 / 80 / 80 (220) | 72 | 33 | 0 | diverged
| free wall | Craig-Bampton, sound, m 4 | 34 µs (4) | 1.25 | 3.93× | 220 / 220 / 220 / 220 (220) | 0 | 75 | 66 |
| free wall | Craig-Bampton, sound, m 30 | 256 µs (30) | 1.05 | 3.40× | 72 / 190 / 220 / 220 (220) | 0 | 68 | 63 |
| house | Newmark, sound, m 4 | 5 µs (4) | 0.60 | 1.92× | 28 / 49 / 117 / 231 (300) | 10136 | 136 | 10 |
| house | Newmark, sound, m 30 | 39 µs (30) | 0.51 | 1.09× | 27 / 47 / 117 / 237 (300) | 1647 | 118 | 9 |
| house | Newmark, sound, m 100 | 130 µs (100) | 0.53 | 0.78× | 20 / 42 / 112 / 242 (300) | 514 | 86 | 10 |
| house | Newmark, exterior may crack, m 30 | 39 µs (30) | 0.37 | 0.59× | 26 / 36 / 111 / 198 (300) | 1644 | 90 | 0 |
| house | Craig-Bampton, sound, m 4 | 5 µs (4) | 0.89 | 3.57× | 290 / 299 / 299 / 299 (300) | 0 | 96 | 67 |
| house | Craig-Bampton, sound, m 30 | 39 µs (30) | 0.90 | 2.35× | 41 / 285 / 299 / 299 (300) | 0 | 33 | 10 |

The hot zone over time from first contact (core / hot / ring chunks):

| case, variant | +0.25 ms | +0.5 ms | +1 ms | +2 ms | +5 ms | +10 ms | +20 ms | end |
|---|---|---|---|---|---|---|---|---|
| b5 20, Newmark m 30 | 144 / 300 / 100 | 534 / 656 / 58 | 750 / 800 / 0 | 800 / 800 / 0 | | | | 800 |
| free wall, Newmark m 30 | 4 / 16 / 12 | 32 / 72 / 32 | 216 / 220 / 0 | 220 / 220 / 0 | | | | 220 |
| house, Newmark m 100 | 2 / 10 / 8 | 10 / 20 / 10 | 21 / 42 / 14 | 40 / 76 / 17 | 82 / 112 / 20 | 125 / 192 / 42 | 166 / 240 / 46 | 172 / 242 / 43 |

Step ratio reached with the gate passed:
- house m = 100: h = 130 µs, 100× the true step;
- free wall m = 100: 855 µs;
- b5 10–30: m = 30 (256 µs) in all four, m = 100 (855 µs) in one (b5 20);
- b5 40: none.
On b5 and the free wall, though, the ratio buys nothing: the exterior is empty after
1 ms.

Exterior cost on the house, Newmark m = 100, wall time split: hot zone 27.5 s, Newton
0.7 s (514 iterations, 1–4 per coarse step), factorisations 4.5 s (86 of them, one per
repartition), escalation and verification 1.1 s.

## Limits

- CPU f64 prototype on the chunk-level world, one thread per run. No GPU path; the GPU
  remarks are structural (sparse direct factorisations, rollbacks and per-substep dense
  interface solves against explicit passes), not measured.
- Single runs per configuration. b5 at 20–40 m/s is chaotic (truth spread up to 3× in
  broken bonds), so regime, fast chunks and ram speed are the robust comparisons.
- The house only to 40 ms (impact and first failures), as in `avbd-stress.md`.
- The hot zone never shrinks. Fragments at rest stay explicit, which favours nothing
  here: the gate cases end with most of each structure hot.
- Escalation parameters are this prototype's: u_hot 0.5, horizon 2 steps, contact margin
  2 steps, one ring, influence growth 2. A larger u_hot cannot be sound (the exterior
  then meets softening bonds), and the bounds table shows what u_hot = 1.0 would allow
  at best.
- Craig–Bampton's failure on the house is measured but not explained. It holds at rest,
  passes b2, the free wall and b5 10, 15 and 30 at m = 4, and does not improve with more
  modes (10 / 40 / 150 modes at m = 30: 640 / 598 / 677 broken). Re-linearising about a dynamic state at each of its ~30
  repartitions is the likely cause; a defect in the prototype is not ruled out.
- The midpoint ("mean multiplier") variant, the weighted-operator GC and the drift bias
  were tried and dropped as unstable (tables above); the code keeps only the form used.

## Reproduce

```sh
cd engine/stress-oracle
cargo test --release -p stress-ref --features mts --lib avbd
cargo run --release --features mts -p stress-ref --example mts_compare -- \
    --case house_1story_ram --end 0.04 --solver mts --true-step --ratio 100 --series s.json
#   --solver oracle | explicit [--true-step] | mts | umap
#   mts: --ratio M  --exterior newmark|cb [--modes K]  --u-hot U  --horizon H
#        --verify rollback|count [--max-rollbacks N]  --no-influence  --all-hot
#        --seed 1-6 --no-escalation (fixed partition)  --energy  --drift  --fine-scale S
```
Cases: `b5_wall_impact_vNN` (any speed), `ram_into_free_wall`, `house_1story_ram`,
`b2_cantilever` and any catalogue scene the chunk world supports. The run logs behind the
tables (JSON lines and per-coarse-step series) are not committed.
