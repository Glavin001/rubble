# Modal pre-window for impacts (prototype)

Branch `proto/modal-prewindow` (from `feat/stress-fracture`). Code: `crates/stress-ref/src/modal.rs`
(`world::modal`, feature `modal`), gate `examples/modal_prewindow.rs`, tests
`tests/modal_prewindow.rs`. Default builds and behaviour are unchanged: the module, its `faer`
dependency, the example and the tests exist only with `--features modal`.

```sh
cargo test --release --features modal -p stress-ref --test modal_prewindow      # 5 tests, ~4 s
cargo run --release --features modal --example modal_prewindow -- [--methods proven] [--serial] [--case NAME] \
    [--json out.json] [--window-only] [--safety-scale S] [--zoh]
```

The full established gate takes about 75 minutes. The b5 basis is built once per process.

## Verdict

**The accuracy gate passes. The cost gate fails.**

* **Accuracy.**
  * The verdicts match in all 17 established-method cases and all 6 proven-method cases. There
    are no false negatives.
  * Every first damage onset falls on the explicit solver's own substep: 0.0000 ms difference
    in all established cases, and at most 0.0041 ms (b5 0.5 m/s, proven).
  * Broken bonds, fragments and impactor exit velocity after the handover stay within the
    explicit solver's own sensitivity. That sensitivity is measured by the same explicit run
    with the impactor moved back 1 um, and with 0.9x the step. One exception: the b5 1 m/s ram
    rebounds at -0.81 m/s against -0.54 m/s in all three explicit runs.
* **Cost.** Wall time over the same span:
  * The window is 2-23x slower than the explicit substeps it replaces, with 16 threads against
    a serial explicit solver.
  * On one core it is 4-30x slower.
  * b5's modes take 170 s to precompute (480 s on one core) and 0.74 GB.
* **Why it does not pay.** Damaging impacts reach onset 16-500 steps into the window, so the
  window replaces 0.1-2% of a run's substeps. A no-damage hit still has to be stepped through
  its contact, with every bond checked every step, at several times the explicit cost. The
  certificate then replaced 5-83% of the remaining run in 3 of the 5 no-damage cases.

The idea is sound and exact. As specified (exact, all modes, bonds checked at the contact
resolution), it is not a speed-up on CPU for clusters of the size measured here (220-840
chunks).

## Method

While a struck cluster is intact and every bond is below its damage onset, its hidden
dynamics are linear: `M x'' + C x' + K x = f(t)`.

* **Assembly.** `K = sum B^T diag(k) B` and `C = sum B^T diag(c) B` over the cluster's live bonds,
  with `B` the bond kinematics of `BondGeometry::kinematics`, `k` the six bond stiffnesses and `c`
  the bond dashpots. Rayleigh mass damping `alpha M` is added on unsupported chunks when the
  scene uses it. `M` holds each chunk's mass and inertia; supported DOFs are removed. A test
  checks `-(K x + C v)` against the explicit solver's own `chunk_loads` of every bond, to
  `1e-12`.
* **Damping is not classical.** Each bond's dashpot is `beta_b k` with
  `beta_b = 2 zeta / sqrt(k_n / m_red)`. Bonds to b5's fixed row see the chunk's full mass as
  reduced mass, the others half, so `C~ = Phi^T C Phi` has off-diagonals up to 6% of its
  diagonals (b5: `6.0e-2`; the free wall of equal chunks is classical, `1e-13`). The modes are
  therefore taken in state space:
  1. `M = L L^T` chunk by chunk. The undamped modes of `L^-1 K L^-T` give `Omega` and
     `Phi = L^-T Q`. A free cluster's six rigid modes are dropped: the floating frame carries
     rigid motion, and dashpots do no work on it.
  2. `s = [Omega eta; eta']` gives `s' = A s + [0; Phi^T f]` with
     `A = [[0, Omega], [-Omega, -C~]]`. This form is well scaled: entries are about `omega`, not
     `omega^2`.
  3. A real eigendecomposition `A = U D U^-1` with the explicit inverse `U^-1`. The inverse gives
     exact biorthogonality, also for repeated frequencies.
  4. One complex coordinate per conjugate pair: `c' = lambda c + b^T f`, `x = Re(sum psi c)`,
     `x' = Re(sum psi lambda c)`.
* **Classical damping** (`C~` diagonal to `1e-10`) takes a closed form per mode instead. Each
  2 x 2 block `[[0, w], [-w, -c]]` has right eigenvector `(w, lambda)` and left eigenvector
  `(-w/lambda, 1) lambda/(lambda^2 - w^2)`. faer's general eigensolver gave nearly parallel
  eigenvectors (inverse residual `1e-7`) on the free wall's repeated frequencies. faer's
  symmetric divide-and-conquer was worse there (`Q^T K Q` off-diagonal `1e-4`). The symmetric
  step therefore uses the tridiagonal QR algorithm. Every basis reports residuals for a
  pseudo-random vector: b5 `1.7e-14`, `1.9e-13`, `6.7e-15`.
* **Stepping.** The window advances on the world's own substep `h` and frames.
  1. Candidates are the chunks whose margin-inflated box (2 mm) can overlap an impactor. Their
     hidden state is rebuilt from the modal sums over their rows only and written into the
     solver.
  2. The world's own `apply_contacts` computes the contact: penalty contact, or layer contact
     under `proven`, with the same state and memory.
  3. The contact acts as an impulse `h F` at the start of the step, followed by the exact free
     response: `c <- e^(lambda h) (c + h b^T F) + (e^(lambda h) - 1)/lambda b^T f_const`. This is
     the explicit solver's own kick-drift staggering, which the impactor update also uses.
     Holding `F` constant over the step (zero-order hold) lags it by half a step against the
     explicit solver. On the free-wall ram, the hidden displacement at the handover then
     differed by 20%, against 0.3% with the impulse.
  4. Constant loads (gravity, and the start state's tiny static residual) are integrated
     exactly.
  5. The impactor and a free cluster's rigid body use the world's own update.
* **Onset.** Bonds are checked in blocks of 32 steps. One GEMM, `Re(Psi C)`, gives every DOF at
  every step of the block. `JointModel::evaluate` then runs on every bond at every step,
  committing its history exactly as the explicit substep does: strain-rate filter, static
  fatigue, dormant contact-patch slip, utilization. The first step at which any bond reports
  damage (or `utilization >= handover_utilization`, default 1) ends the window. The world is
  rolled back to the start of that step: modal state, impactors, rigid body, contact memory,
  time and frame. The full hidden state `(u, theta, v, w)` of every chunk is written in, and the
  explicit solver takes that step itself. A bisection on the closed-form trajectory also reports
  the continuous crossing time, with the rate factor frozen over the step.
* **No-damage certificate** (anchored clusters, contact quiet for 200 steps, retried every 2 ms).
  For later times with no contact, `|c_j(t) - c_j,inf| <= |c_j(now)  - c_j,inf|` because
  `Re lambda <= 0`. Each bond component is therefore bounded by `|q - q_inf| <= sum_j |D_bj| |c_j - c_j,inf|`
  (`D = k B psi`). The failure index is bounded from these boxes with conservative strength:
  no rate increase, and static fatigue at its worst over the remaining time. The impactors'
  ballistic paths, swept with the same displacement bound and the margin, must not overlap any
  chunk. If every bond's bound stays below 0.95 and no re-contact is possible, no explicit
  window is needed.

### Exactness

For the linear system with piecewise-constant (or impulsive) loads, the update is the exact
solution up to floating point. `tests/modal_prewindow.rs::modal_update_is_exact` advances an
initial velocity and a constant load over 0.2 ms in one step and in seven unequal steps. Against
RK4 at 1/100 of the fastest radian, the error is `1e-9`, which is RK4's own error. It is the same
for 1 and 7 steps, on an anchored wall with non-classical damping and on a free wall (rigid modes
dropped). The window differs from the explicit solver only by the explicit solver's
time-discretisation error. On the free-wall ram, halving the Courant safety three times halved
the zero-order-hold state difference each time (`1.96e-1, 9.8e-2, 4.9e-2, 2.5e-2`), as the
explicit solver's first-order error does.

## Results

Machine: Apple M3 Max (16 cores), release build. The modal window runs parallel (faer GEMMs,
rayon); the explicit solver is serial at these sizes (its pool starts at 4096 items).
"E" is the explicit truth, "M" the modal path (window then handover), "Ex" the explicit solver
with the impactor 1 um further back, and "Edt" the explicit solver at 0.9x the Courant safety.
"peak index" is the largest failure index reached when nothing breaks. "state diff" is the
relative difference of hidden displacement and velocity between the window's end state and the
explicit solver at the same time.

### Established methods (`Methods::established()`), all cases

| case | first crack E / M (ms) | diff (ms) | broken E / M / Ex / Edt | fragments E / M / Ex / Edt | exit v E / M / Ex / Edt (m/s) | peak index E / M | window | state diff x / v |
|---|---|---|---|---|---|---|---|---|
| b5 10 m/s | 0.1319 / 0.1319 | 0.0000 | 249 / 267 / 296 / 285 | 24 / 32 / 42 / 45 | 8.364 / 8.163 / 8.294 / 8.225 | - / 0.926 | handover at 0.1294 ms (bond 810, crossing 0.1282 ms) | 1.6e-03 / 2.1e-03 |
| b5 15 m/s | 0.0921 / 0.0921 | 0.0000 | 437 / 376 / 389 / 392 | 87 / 81 / 82 / 84 | 12.424 / 12.479 / 12.442 / 12.425 | - / 0.950 | handover at 0.0896 ms (bond 810, crossing 0.0877 ms) | 1.7e-03 / 1.5e-03 |
| b5 20 m/s | 0.0722 / 0.0722 | 0.0000 | 541 / 520 / 564 / 593 | 108 / 103 / 120 / 111 | 16.998 / 17.008 / 16.923 / 17.030 | - / 0.794 | handover at 0.0697 ms (bond 810, crossing 0.0694 ms) | 2.1e-03 / 1.6e-03 |
| b5 30 m/s | 0.0523 / 0.0523 | 0.0000 | 123 / 135 / 123 / 120 | 5 / 7 / 5 / 5 | 26.980 / 26.986 / 26.980 / 26.982 | - / 0.860 | handover at 0.0498 ms (bond 810, crossing 0.0484 ms) | 2.4e-03 / 1.7e-03 |
| b5 40 m/s | 0.0423 / 0.0423 | 0.0000 | 64 / 72 / 164 / 64 | 3 / 3 / 6 / 3 | 35.999 / 35.993 / 36.008 / 36.002 | - / 0.787 | handover at 0.0398 ms (bond 810, crossing 0.0388 ms) | 2.7e-03 / 2.2e-03 |
| free wall ram 40 m/s | 1.2664 / 1.2664 | 0.0000 | 24 / 24 / 24 / 30 | 6 / 6 / 6 / 6 | 35.053 / 35.052 / 35.053 / 34.972 | - / 0.889 | handover at 1.2639 ms (bond 206, crossing 1.2621 ms) | 2.8e-03 / 2.2e-03 |
| b5 0.05 m/s | none / none | - | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | -0.021 / -0.021 / -0.021 / -0.021 | 0.425 / 0.426 | certified no damage at 286.536 ms (bound 0.429) | 3.1e-04 / 3.9e-03 |
| b5 0.1 m/s | none / none | - | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | -0.041 / -0.041 / -0.041 / -0.041 | 0.868 / 0.869 | certified no damage at 158.479 ms (bound 0.949) | 9.9e-04 / 2.7e-03 |
| b5 0.2 m/s | 6.3187 / 6.3187 | 0.0000 | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | -0.081 / -0.081 / -0.081 / -0.081 | - / 0.999 | handover at 6.3163 ms (bond 29, crossing 6.3143 ms) | 1.5e-03 / 8.2e-03 |
| b5 0.5 m/s | 2.2323 / 2.2323 | 0.0000 | 40 / 40 / 40 / 40 | 2 / 2 / 2 / 2 | -0.107 / -0.107 / -0.107 / -0.107 | - / 0.990 | handover at 2.2299 ms (bond 1155, crossing 2.2293 ms) | 5.7e-03 / 4.6e-03 |
| b5 1 m/s | 1.1398 / 1.1398 | 0.0000 | 185 / 177 / 183 / 183 | 14 / 12 / 14 / 14 | -0.542 / -0.811 / -0.539 / -0.537 | - / 0.951 | handover at 1.1373 ms (bond 1057, crossing 1.1373 ms) | 6.0e-03 / 8.1e-03 |
| b5 2 m/s | 0.5923 / 0.5923 | 0.0000 | 289 / 319 / 229 / 285 | 35 / 43 / 28 / 34 | 0.587 / 0.550 / 0.659 / 0.540 | - / 0.989 | handover at 0.5898 ms (bond 715, crossing 0.5890 ms) | 6.2e-03 / 5.7e-03 |
| debris 10 cm concrete 1 m/s | none / none | - | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | -0.102 / -0.103 / -0.094 / -0.098 | 0.445 / 0.458 | no onset to the end (50.000 ms) | 2.6e-02 / 7.3e-02 |
| debris 2 cm steel 5 m/s | none / none | - | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | -0.683 / -0.686 / -0.673 / -0.709 | 0.156 / 0.156 | certified no damage at 8.601 ms (bound 0.044) | 1.9e-03 / 3.3e-02 |
| debris 10 cm concrete 5 m/s | 1.2219 / 1.2219 | 0.0000 | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | -0.027 / -0.027 / -0.027 / -0.028 | - / 0.946 | handover at 1.2195 ms (bond 1007, crossing 1.2176 ms) | 4.1e-03 / 1.2e-02 |
| debris 10 cm concrete 20 m/s | 0.3111 / 0.3111 | 0.0000 | 4 / 4 / 4 / 4 | 1 / 1 / 1 / 1 | -1.168 / -1.197 / -1.257 / -1.262 | - / 0.771 | handover at 0.3086 ms (bond 1007, crossing 0.3070 ms) | 3.1e-03 / 1.9e-03 |
| debris 5 cm steel 10 m/s | 0.3733 / 0.3733 | 0.0000 | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | -1.973 / -1.970 / -1.966 / -2.035 | - / 0.849 | handover at 0.3708 ms (bond 1007, crossing 0.3704 ms) | 3.8e-03 / 2.1e-03 |

**The assets.**
* b5: 840 chunks, 4,800 free DOFs, 4,800 complex modes. Non-classical damping, `C~`
  off-diagonals 6.0e-2 of the diagonals. Modal damping ratios 0.0000-0.0186.
* Free wall: 220 chunks, 1,314 flexible DOFs, classical damping.

**The no-damage cases.**
* b5 at 0.05 and 0.1 m/s:
  * The window carries the explicit solver's peak failure index to 3 digits (0.425/0.426 and
    0.868/0.869).
  * The certificate holds at 287 ms and 158 ms of the 300 ms run. It waits until the ram,
    rebounding at 2-4 cm/s, has cleared the wall by the 2 mm margin.
* Debris:
  * 2 cm steel at 5 m/s: the certificate holds 8.6 ms into a 50 ms run.
  * 10 cm concrete at 1 m/s: the index bound passed (0.12), but the cube lingers against the
    wall (10,139 contact steps) and the re-contact sweep never clears it, so the window ran to
    the end.
* The b5 0.2 m/s case cracks one bond (index 0.999 just before) and is caught at the explicit
  solver's step.

**State at the handover.**
* The displacement agrees with the explicit solver to 0.2-0.6%.
* The velocity agrees to 0.2-0.8%. The window hands over the step-mean velocity
  `(x_n - x_(n-1)) / h`, the explicit solver's staggered `v_(n-1/2)`.
* Handing over the exact velocity `x'(t_n)` instead left 2-4% velocity differences. The run was
  then outside the explicit solver's spread on b5 40 m/s (456 broken bonds and 91 fragments,
  against 64 and 3).

**Holding the contact force over the step (zero-order hold)** instead of the impulse form gave
the same verdicts and onsets. The hidden displacement then differed by 8-20% at the handover
(0.2-0.6% with the impulse), converging only to first order in the step.

### Proven methods (`Methods::proven()`, layer contact and co-rotational bonds), subset

The "Edt" column was not run for this table (it repeats "Ex"). This run's window times were
taken while other jobs shared the machine.

| case | first crack E / M (ms) | diff (ms) | broken E / M / Ex / Edt | fragments E / M / Ex / Edt | exit v E / M / Ex / Edt (m/s) | peak index E / M | window | state diff x / v |
|---|---|---|---|---|---|---|---|---|
| b5 10 m/s | 0.1200 / 0.1200 | 0.0000 | 113 / 113 / 113 / 113 | 4 / 4 / 4 / 4 | 8.341 / 8.340 / 8.341 / 8.341 | - / 0.558 | handover at 0.1158 ms (bond 810, crossing 0.1153 ms) | 5.0e-03 / 4.6e-03 |
| b5 40 m/s | 0.0414 / 0.0414 | 0.0000 | 49 / 50 / 48 / 48 | 3 / 3 / 3 / 3 | 36.926 / 36.948 / 36.977 / 36.977 | - / 0.780 | handover at 0.0372 ms (bond 810, crossing 0.0337 ms) | 6.9e-03 / 6.2e-03 |
| free wall ram 40 m/s | 1.2629 / 1.2629 | 0.0000 | 20 / 20 / 20 / 20 | 3 / 3 / 3 / 3 | 36.026 / 36.038 / 36.027 / 36.027 | - / 0.788 | handover at 1.2588 ms (bond 206, crossing 1.2552 ms) | 7.0e-03 / 6.1e-03 |
| b5 0.1 m/s | none / none | - | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | -0.034 / -0.033 / -0.034 / -0.034 | 0.890 / 0.891 | certified no damage at 151.435 ms (bound 0.794) | 1.1e-03 / 2.7e-02 |
| b5 0.5 m/s | 2.1428 / 2.1387 | 0.0041 | 40 / 40 / 40 / 40 | 2 / 2 / 2 / 2 | -0.099 / -0.099 / -0.099 / -0.099 | - / 0.986 | handover at 2.1345 ms (bond 1055, crossing 2.1328 ms) | 1.4e-02 / 1.5e-02 |
| debris 2 cm steel 5 m/s | none / none | - | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | -0.610 / -0.614 / -0.593 / -0.593 | 0.181 / 0.181 | certified no damage at 6.947 ms (bound 0.051) | 1.7e-03 / 4.6e-02 |

Under layer contact the explicit solver is far less chaotic, and the modal path reproduces it
almost exactly: b5 10 m/s 113/4/8.340 against 113/4/8.341. This holds although co-rotational
bonds are linearised in the window's dynamics, and the onset check uses the co-rotational
kinematics.

### Cost

**Window against the explicit solver over the same span, established, 16 threads.** Stage
columns are seconds.

| case | window steps | explicit substeps (run) | window s | explicit s same span | ratio | ms/step modal | ms/substep explicit | contact | modal sums | GEMM | law | certificate | handover |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b5 10 m/s | 52 | 39927 | 0.170 | 0.019 | 8.8 | 3.261 | 0.371 | 0.032 | 0.029 | 0.037 | 0.003 | 0.000 | 0.060 |
| b5 15 m/s | 36 | 39927 | 0.169 | 0.014 | 12.5 | 4.682 | 0.375 | 0.028 | 0.031 | 0.038 | 0.002 | 0.000 | 0.057 |
| b5 20 m/s | 28 | 13394 | 0.142 | 0.011 | 13.3 | 5.061 | 0.379 | 0.031 | 0.015 | 0.021 | 0.002 | 0.000 | 0.061 |
| b5 30 m/s | 20 | 13394 | 0.133 | 0.008 | 17.6 | 6.672 | 0.379 | 0.023 | 0.017 | 0.022 | 0.001 | 0.000 | 0.062 |
| b5 40 m/s | 16 | 13394 | 0.137 | 0.006 | 23.3 | 8.592 | 0.368 | 0.033 | 0.016 | 0.020 | 0.001 | 0.000 | 0.060 |
| free wall ram 40 m/s | 504 | 199380 | 0.093 | 0.046 | 2.0 | 0.185 | 0.092 | 0.006 | 0.017 | 0.046 | 0.006 | 0.000 | 0.014 |
| b5 0.05 m/s | 115136 | 120546 | 164.153 | 39.335 | 4.2 | 1.426 | 0.342 | 10.202 | 51.133 | 76.051 | 4.075 | 18.286 | 0.000 |
| b5 0.1 m/s | 63680 | 120546 | 113.141 | 22.419 | 5.0 | 1.777 | 0.352 | 8.458 | 36.586 | 49.677 | 2.926 | 12.035 | 0.000 |
| b5 0.2 m/s | 2538 | 120546 | 3.440 | 0.905 | 3.8 | 1.355 | 0.357 | 0.288 | 1.011 | 1.509 | 0.076 | 0.389 | 0.038 |
| b5 0.5 m/s | 896 | 119730 | 2.371 | 0.336 | 7.1 | 2.646 | 0.375 | 0.205 | 0.760 | 0.966 | 0.049 | 0.219 | 0.098 |
| b5 1 m/s | 457 | 119679 | 1.066 | 0.164 | 6.5 | 2.332 | 0.358 | 0.091 | 0.307 | 0.409 | 0.023 | 0.135 | 0.068 |
| b5 2 m/s | 237 | 119679 | 0.392 | 0.084 | 4.7 | 1.655 | 0.354 | 0.039 | 0.112 | 0.155 | 0.010 | 0.000 | 0.062 |
| debris 10 cm concrete 1 m/s | 20091 | 20091 | 21.947 | 7.520 | 2.9 | 1.092 | 0.374 | 1.870 | 4.828 | 12.050 | 0.633 | 1.402 | 0.000 |
| debris 2 cm steel 5 m/s | 3456 | 20091 | 3.895 | 1.193 | 3.3 | 1.127 | 0.345 | 0.299 | 0.767 | 2.039 | 0.095 | 0.497 | 0.000 |
| debris 10 cm concrete 5 m/s | 490 | 20091 | 0.586 | 0.168 | 3.5 | 1.196 | 0.343 | 0.049 | 0.128 | 0.302 | 0.015 | 0.000 | 0.059 |
| debris 10 cm concrete 20 m/s | 124 | 20091 | 0.192 | 0.043 | 4.5 | 1.552 | 0.345 | 0.015 | 0.028 | 0.074 | 0.005 | 0.000 | 0.059 |
| debris 5 cm steel 10 m/s | 149 | 20091 | 0.238 | 0.051 | 4.7 | 1.598 | 0.339 | 0.022 | 0.039 | 0.095 | 0.006 | 0.000 | 0.059 |

**One core** (`--serial`: faer sequential, no rayon), established. The explicit numbers come
from the same process:

| case | window steps | explicit substeps (run) | window s | explicit s same span | ratio | ms/step modal | ms/substep explicit | contact | modal sums | GEMM | law | certificate | handover |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b5 10 m/s | 52 | 6697 | 1.018 | 0.035 | 29.3 | 19.567 | 0.668 | 0.042 | 0.290 | 0.363 | 0.036 | 0.000 | 0.185 |
| free wall ram 40 m/s | 504 | 6646 | 0.206 | 0.049 | 4.2 | 0.409 | 0.096 | 0.007 | 0.014 | 0.137 | 0.029 | 0.000 | 0.013 |
| b5 0.5 m/s | 896 | 6697 | 8.978 | 0.447 | 20.1 | 10.020 | 0.498 | 0.177 | 2.203 | 4.097 | 0.291 | 2.085 | 0.086 |
| debris 2 cm steel 5 m/s | 3456 | 6697 | 17.255 | 1.355 | 12.7 | 4.993 | 0.392 | 0.197 | 0.703 | 10.942 | 0.754 | 4.526 | 0.000 |

**Precompute** (per asset, independent of the impact):

| asset | DOFs | 16 threads | 1 core | memory |
|---|---|---|---|---|
| b5 wall | 4,800 | 170 s: undamped eigen 37, state-space eigen 123, inverse 5 | 478 s: 48, 358, 39 | 737 MB (`psi`, `b`) |
| free wall | 1,314 | 0.9 s (classical closed form) | 1.2 s | 56 MB |

**Explicit substeps the handover saves.**
* Damaging impacts: the window covers 16-2,538 substeps of 13,394-199,380 per run (0.1-2%).
  This is the time from t = 0 to onset; on b5 most of it is the ram's approach.
* No-damage impacts: the certificate removes the rest of the run.
  * b5 0.05 m/s: 5% of the run.
  * b5 0.1 m/s: 47%.
  * Debris 2 cm steel: 83%.
  * Debris 10 cm concrete 1 m/s: none (no certificate).

**Where the window's time goes.** Per step, b5 costs 1.1-2.6 ms in long windows and 3-9 ms in
the short ones, where the handover's full-state write and the first candidate setup dominate.
The explicit solver costs 0.34-0.38 ms. Two terms set this:
* **Bond checks.** Each check needs every DOF from the modes: a dense `n x modes` product, about
  92 MFLOP per step on b5. The explicit solver needs a sparse sweep of about 1,800 joint
  evaluations. The checks are about 50% of the window.
* **Modal sums at the contact rows.** About 6 x 20 rows x 4,800 modes per step: 30% of the
  window.

The contact itself is the world's (10%), and the certificate's `|D| |c|` bound adds 10% in long
windows.

**Checking less often (bounded envelopes between checks) does not rescue the cost.** This is
reasoned, not measured. A rigorous bound on a bond force's change over an interval is
`sum_j |D_bj| |c_j'|`. It has no cancellation between modes, so it loses causality: a bond the
stress wave has not yet reached gets the same loose bound as the struck one. The bound costs a
row of `|D|` (`O(modes)`) per bond, as much as an exact check. The b5 step is close to the top
mode's stability limit (`|lambda| h ~ 1`), so the bound grows by a top-mode amplitude per step
and stays tight only for a few steps. Skipping enough checks to beat the explicit solver needs
skip lengths in the tens of steps for every bond.

## Limits

* **Scope.**
  * Explicit solve mode only.
  * One bonded cluster in the world.
  * Intact bonds without rebar: the cluster is refused otherwise.
  * No scripted events, engine coupling or refinement.
  * The certificate is offered for anchored clusters without a ground plane.
* **Free clusters** are linearised about their rigid motion. The floating frame's centrifugal,
  Coriolis and gyroscopic loads are second order and left out. The rigid body takes the full
  contact load through the world's own update, and inertia relief is exact in the flexible
  modes. On the free-wall ram this matched the explicit solver to 0.2% in state and identically
  in outcome.
* **Co-rotational bonds (proven)** are linearised in the dynamics; the onset check uses their
  kinematics. The rotations before onset are about 1e-4, and the proven runs matched closely.
* **Damping** is exact for any bond dashpots plus Rayleigh mass damping. The price is a real
  eigendecomposition of size `2n`: 123 s of b5's 170 s. Classical damping skips it.
* **Bond history** (strain-rate filter, fatigue, dormant contact-patch slip) is exact because
  the joint law runs at every step. That is also why the window costs what it does. A cheaper
  window would have to reconstruct this history at the handover.
* **Contact.** A 2 mm margin bounds which chunks need their hidden state each step. The largest
  hidden displacement seen at a contact row before onset was 0.44 mm.
* **The certificate is conservative.**
  * It needs the contact quiet for 200 steps.
  * The ballistic sweep must clear every chunk by the margin.
  * The bound `sum |D| |c - c_inf|` adds mode amplitudes without phase, so it only passes at
    0.95 when the actual peak is well below.
  * Static fatigue is taken at its worst over the remaining time.
* **Refinement** (multi-level chunks) is not supported: refining changes `K` and `M`, so the
  basis would have to be rebuilt.
* **The basis is per asset and per method set.** Rayleigh damping rescales the dashpots, so
  proven and established need separate bases. It is valid only while every bond of the cluster
  is intact; after any damage, the clusters (and their bases) change.

## What would make it pay

All of these are untested here:
* Truncating modes with a certified error, which gives up exactness.
* Batching the checks on a GPU, where the dense GEMMs are cheap and the explicit solver's
  per-substep latency is the bottleneck (REALTIME_ROADMAP A2/A6).
* Precomputing per-asset influence bounds that keep causality, so far bonds need no checks
  until the wave can reach them.

On this CPU and at these sizes, the explicit window remains the cheaper exact method.
