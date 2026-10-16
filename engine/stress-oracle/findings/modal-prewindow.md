# Modal pre-window for impacts (prototype)

Branch `proto/modal-prewindow` (from `feat/stress-fracture`). Code: `crates/stress-ref/src/modal.rs`
(`world::modal`, feature `modal`), gate `examples/modal_prewindow.rs`, tests
`tests/modal_prewindow.rs`. Default builds and behaviour are unchanged: the module, its `faer`
dependency, the example and the tests exist only with `--features modal`.

```sh
cargo test --release --features modal -p stress-ref --test modal_prewindow      # 5 tests, ~4 s
cargo run --release --features modal --example modal_prewindow -- [--methods proven] [--serial] [--case NAME]
```

## Verdict

__VERDICT__

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

__RESULTS__

## Limits

__LIMITS__
