# Explicit beams: aerodynamics (`wings`, `fusedrag`) and `commands` without input

Behavioural specification for adding three `.truck` features to Rubble's explicit
node–beam solver (`explicit-beams.md`): command beams with no driver input, the
fuselage drag (`fusedrag`) and wing panels (`wings`). It extends that document. Terms
not defined here (step h = 0.0005 s, frame of 33 steps, node pass, beam pass, origin
and relative positions, approximate power `pow≈`, gate G, strength S, kinds and types)
mean what they mean there.

The reference is Rigs of Rods (RoR), compared as a black box. The goal is agreement
to float tolerance, so this document gives every formula with its constants, the
precision each sub-expression is evaluated in, and its order relative to the other
passes. Several behaviours are quirks of the reference, not physics. They are marked
**(quirk)** and must be reproduced in the RoR-compatible profile.

Notation: all quantities are IEEE single precision (f32) unless marked **[f64]**. An
expression marked [f64] is evaluated in double precision from f32 inputs (each f32
input widened exactly), and its result is rounded to f32 once, at the point stated.
Vector operations are component-wise f32; a dot product is `x·x' + y·y' + z·z'`,
summed left to right; |v| = sqrt(v·v) with a correctly rounded f32 sqrt; the cross
product u × w = (u.y·w.z − u.z·w.y, u.z·w.x − u.x·w.z, u.x·w.y − u.y·w.x).
"Normalise v" means: L = |v|; if L > 0, multiply each component by (1/L) (the
reciprocal computed in f32 first); if L = 0, leave v unchanged.

---

## 0. Summary of what changes in a step

The existing step (node pass, beam pass, self-collision) becomes:

| order | pass | reads | writes |
|---|---|---|---|
| once per frame, before the origin rebase | **wing break check** (§3.8) | node world positions (end of previous frame) | wing broken flags |
| 1 | **node pass** (existing). Its force restart adds the fuselage-drag vector **D** stored by the previous step **instead of** the air drag and turbulence, on vehicles that have a `fusedrag` line (§2.4) | D from step n−1 | v, x, F |
| 1a | **wings**, each wing in file order (§3) | velocities and positions just integrated in this step | adds to F |
| 1b | **fuselage drag**: compute D (§2) | the front node's just-integrated velocity and world height | stores D for the next step |
| 1c | **commands** (§1.4): change rest lengths; with no input only self-centering beams move | rest lengths | rest lengths |
| 2 | **beam pass** (existing) | | adds to F |
| 3 | **self-collision** (existing) | | adds to F |

The node pass integrates with the force accumulated over the previous step. So wing
forces, like beam forces, act on the step after the state they were computed from.
The fuselage drag is a further step late: it is computed from the velocities of step
n (pass 1b) and added during the restart of step n+1, so it is first integrated in
step n+2. In the first step after spawn (or reset), D = 0.

---

## 1. `commands` and `commands2`

### 1.1 File format (the parts that matter)

`commands2`: `n1, n2, shorten_rate, lengthen_rate, max_contraction, max_extension,
contract_key, extend_key [, options [, description [, start_delay, stop_delay,
start_fn, stop_fn [, affect_engine [, needs_engine [, plays_sound]]]]]]`.

`commands` is the same line with a single rate (used as both shorten and lengthen
rate) and one fewer field. Lines from both sections join **one list in file order**.

* `max_contraction`, `max_extension`: dimensionless multiples of the spawn length.
* Rates are in metres per second (the rest length changes by rate·h per step at
  full command; see §1.4).
* Options, one character each: `n` filler; `i` invisible (no physics effect); `r`
  rope; `c` self-centering; `f` "not faster" (an engine-coupling limiter); `p`
  one-press; `o` one-press-to-centre. Conflicts: the first of `o`, `p`, `c` in the
  string wins and the other two of those are dropped. Unknown letters are ignored.
* Defaults: affect_engine = 1.0, needs_engine = true, plays_sound = true, no inertia.
* `description`, `plays_sound`, the key numbers and the inertia only affect input
  handling, sound and UI. With no input they have no physical effect (§1.4).

### 1.2 The beam each line spawns

One beam per line, between n1 and n2, using the **beam defaults in force at that line**
(`set_beam_defaults` and its scale factors):

| property | value |
|---|---|
| kind | **plain**, or **rope** with option `r` |
| type | **actuator**: it never yields (no plastic deformation in compression or tension) |
| stiffness k | default springiness × springiness scale |
| damping c | default damping × damping scale |
| strength S | default breaking threshold × breaking scale (**scaled**; unlike plain beams, whose spawn strength is the unscaled threshold) |
| compression and tension thresholds, gate G | the same **deform threshold** a plain beam would get at this line (the existing loader's rules: 400 000 N when unset; raised to 400 000 unless `enable_advanced_deformation` is on; at least the "creak" value 100 000 unless a plastic coefficient was given; × the deform scale). **G = that threshold, not S** (see §1.3) |
| plastic coefficient | from the defaults (unused because the beam never yields) |
| rest length L, reference length L_ref | both = |x_n1 − x_n2| at spawn (spawned f32 positions) |
| detacher group | the current detacher group at that line |

**Spawn order**: command beams are created after `beams`, `shocks`, `shocks2` and
`shocks3`, and before `hydros`, `triggers` and `ropes`. The existing spec's order
"beams, shocks, hydros" becomes "beams, shocks, **commands**, hydros".

For the "on a collision triangle with fewer than 3 live beams" break exception, a
non-rope command beam counts as a live plain beam. A rope command beam does not
(only beams with no special kind are counted).

### 1.3 Forces, compression, tension, breaking

Command beams go through the ordinary beam pass with no special rules:

* **Force**: F = −k·e − c·u (e = length − L), as for any beam.
* **Rope** (`r`): when e < 0 (shorter than L), k ← 0 and c ← 0.1·c for that step,
  exactly the rope rule. Under tension it is a normal spring–damper.
* **No yield**: the actuator type skips both yield branches. L only changes through
  §1.4.
* **Break**: the break test sits inside the gate test. The beam breaks when |F| > G
  **and** |F| > S, that is when |F| > max(G, S). This uses the same collision-triangle
  exception (if an end saves it, S ← 2·G) and the same detacher-group cascade as any
  beam. **(quirk)** Because G is the deform threshold, which defaults to 400 000 N
  or more, a command beam whose break threshold is lower does not break until
  |F| > G.
  * *Note for the implementer*: the existing explicit-beams spec sets G = S for
    actuators. RoR gives hydros G = deform threshold too (both are spawned through
    the same beam-creation path). Check `hydros` against this rule as well.

### 1.4 Behaviour over time with no input

Each step, after the wings and fuselage drag and before the beam pass, the command
pass runs. With no driver input every key's command value is 0. Then:

* **Plain command beams, and the `p` and `o` options**: nothing changes. The
  one-press state machines only advance on a key press. The inertia filter, starting
  at rest with input 0, outputs 0. The length update multiplies L by exactly 1.0.
  L stays at the spawn length for ever.
* **`needs_engine`, `affect_engine`, `f`**: these only gate or scale motion. With no
  motion they have no effect. (When a command does move, its speed is multiplied by
  the engine's crank factor if affect_engine > 0. With no engine that factor is 1.
  Vehicles whose engine would be involved are out of scope.)
* **Self-centering (`c`) beams can move with no input. (quirk)** They are driven
  toward the centre ratio C = (max_extension − max_contraction)/2 + max_contraction
  (f32; the formula is symmetric in the two limits, so it holds whichever is larger).
  If the limits are not symmetric about 1.0, C ≠ 1 and the beam starts moving at
  spawn.

#### Self-centering algorithm

Each command line owns one shared state: mode m ∈ {−1, 0, +1}, initially 0. It has
two entries, a contraction entry (direction −1, rate = shorten_rate, boundary =
max_contraction, on the contract key) and an extension entry (direction +1, rate =
lengthen_rate, boundary = max_extension, on the extend key).

The pass visits key numbers in ascending order, from 1 to the maximum key. Under each
key it visits that key's entries in file order. So the contraction entry is processed
before the extension entry exactly when contract_key < extend_key. Several commands
may share keys; their entries interleave by key.

For each entry with direction σ of a self-centering line (no input):

1. If L_ref = 0 or L = 0: skip this entry entirely.
2. r = L / L_ref (f32). If |r − C| < 0.0001 (f32 subtraction, compared with 1e-4 in
   f64): m ← 0. Otherwise: let m_old = m; m ← −1 if r > C, else +1; if m_old ≠ 0 and
   m_old ≠ m (it overshot): L ← C·L_ref (f32) and m ← 0.
3. Recompute r = L / L_ref. The entry may move only if
   (σ = +1 and r < max_extension) or (σ = −1 and r > max_contraction), strictly.
4. If it may move: the command value v is 1 when σ·m > 0, else 0 (no input).
   Then L ← L × (1 + σ·rate·v·cf·h / L), where cf = 1 with no engine. Evaluate
   rate·v·cf·h·(1/L) in f32, left to right as written (rate·v, ·cf, ·h, then ÷L),
   add it to 1.0 in [f64], multiply by L in [f64], and round to f32. With v = 0 this
   is L × 1.0, an exact no-op.

Consequence: a self-centering beam spawned at r = 1 with C ≠ 1 changes L by rate·h
per step, at the rate of the side it moves toward, until |r − C| < 1e-4 or it
overshoots and snaps to C·L_ref. Then it holds.

Nothing else in the command pass touches beams when there is no input. (Rotators
are a separate feature and out of scope.)

---

## 2. `fusedrag`

### 2.1 File format

`fusedrag` takes one of two forms:

* `front_node, rear_node, width [, airfoil]`
* `front_node, rear_node, autocalc [, area_coefficient [, airfoil]]`

`airfoil` defaults to `NACA0009.afl` and `area_coefficient` defaults to 1.0. If the
section has several lines, the last one wins.

### 2.2 Spawn-time quantities

* **Front node P**: `front_node`.
* **(quirk)** The rear node is parsed and then ignored. The "back" end of the
  fuselage axis is also P, so the axis has zero length.
* **Width W**:
  * explicit form: W = `width`, used as given, including 0 or a negative value;
  * `autocalc`: W = (z_max − z_min) · (y_max − y_min) · area_coefficient, multiplied
    left to right in f32.
    * The extents are the minimum and maximum of the **node coordinates as written in
      the file**, before any spawn translation.
    * They are taken over every node defined in `nodes` / `nodes2`. Cinecam and tyre
      nodes are not included.
    * They start at min = 1000 and max = −1000.
    * W is an area in m², but it is used below as a length, so W² appears. **(quirk)**
* **Airfoil**: the `.afl` table (§4) is loaded. A missing file gives all-zero tables.
* **N**: the vehicle's total node count, including cinecam and tyre nodes.

### 2.3 The fuselage-drag vector D (pass 1b, every step)

If W ≤ 0, nothing is computed and D keeps its value, which is 0 since spawn.
Otherwise:

1. Airflow a = −v_P, where v_P is P's velocity just integrated in this step's node
   pass. A = |a|. There is **no wind** anywhere in RoR: "wind" is always just the
   negated velocity.
2. Axis length ℓ = 0 (the axis runs from P to P), so the fuselage area s = ℓ·W = 0.
3. The airfoil's drag coefficient c_x is looked up. Because the axis is zero, the
   angle-of-attack computation always yields 0. (The rotation between a zero vector
   and anything is the identity; see §3.4.) So c_x = c_d(index 1800), the α = 0
   entry (0.0052 for NACA0009). It is multiplied by s = 0, so **the airfoil has no
   effect on the force.** Its only role is that a `fusedrag` line exists (§2.4).
4. Air density ρ_F at height y = world y of P (origin.y + rel.y):
   * t = 1.0 − 0.0065·y / 288.1 [f64: (0.0065·y)/288.1, then 1.0 − that], rounded
     to f32. Note **288.1**; the wings use 288.15.
   * p = 101325 × pow≈(t, 5.24947), in f32, where pow≈ is the bit-level approximation
     defined in explicit-beams.md and 5.24947 is rounded to f32.
   * ρ_F = p × f32(0.0000120896) (an f32 product).
   * At y = 0 this gives ρ = 1.2249787… → f32.
5. Scalar k_D = ((c_x·s) + W·W·0.5) · 0.5 · ρ_F · A / N:
   * c_x·s and W·W are f32 products;
   * (W·W)·0.5 and every later operation are [f64], evaluated left to right in the
     order shown (N widened exactly);
   * the result is rounded to f32.
6. D = k_D · a (component-wise f32). Store D.

The force is the same vector for every node, so the total is N·D, which is
(W²/4)·ρ·|v|·(−v). Each node's acceleration depends on its own mass.

### 2.4 Applying D: node pass, force restart

In the node pass, for every node, after integration, the force restart becomes:

* F ← (0, m·gravity, 0); then
* **if the vehicle has a `fusedrag` line (whatever W is): F ← F + D.** No air drag
  (−0.05·s·v) and no turbulence are applied, and **no turbulence noise draws are
  consumed**, so this vehicle's noise sequence does not advance.
* otherwise: the existing air-drag and turbulence rules apply.

This applies to every node, immovable ones included (their force is never used).

**(quirk) A `fusedrag` line with W ≤ 0 removes all air drag from the vehicle and adds
nothing.** The Viper ships with exactly this (`fusedrag` / `98, 73, 0.0,
NACA0009.afl`): its nodes get gravity and contact forces only, with no 0.05 drag and
no turbulence.

### 2.5 Edge cases

* v_P = 0: A = 0, so D = 0.
* D lags by one step (§0). At spawn and after a reset, D = 0.

---

## 3. `wings`

### 3.1 File format

16 to 22 fields:

* 8 node references, in this order:

  | slot | position | slot | position |
  |---|---|---|---|
  | 0 | front-left-down (FLD) | 4 | back-left-down (BLD) |
  | 1 | front-right-down (FRD) | 5 | back-right-down (BRD) |
  | 2 | front-left-up (FLU) | 6 | back-left-up (BLU) |
  | 3 | front-right-up (FRU) | 7 | back-right-up (BRU) |

* 8 texture coordinates: **irrelevant to physics**; ignore them;
* control-surface letter (default `n`);
* chord point (default −1);
* minimum deflection in degrees (default −1);
* maximum deflection in degrees (default −1);
* airfoil file name (no default: a wing without one is rejected and spawns nothing);
* efficacy coefficient (default 1.0).

Slots may repeat nodes. The documented 2-, 3- and 4-node layouts are `A,B,A,B,A,B,A,B`,
`A,B,A,B,E,E,E,E` and `A,B,A,B,E,F,E,F`; the Viper uses the 4-node layout. Forces are
added once per slot, so a node in two slots gets the slot force twice (§3.6).

Control-surface letters: `n` none; `a` right aileron; `b` left aileron; `f` flap; `e`
elevator; `r` rudder; `c`/`d` right/left elevon; `g`/`h` right/left flaperon; `i`/`j`
right/left ruddervator; `S`/`T` right/left stabilator; `U`/`V` right/left taileron.
Any other letter is treated as `n`.

* *Has control*: every letter except `n`, `S`, `T`, `U`, `V`.
* *Stabilator*: `S`, `T`, `U`, `V`.

### 3.2 Spawn-time quantities per wing

* **Chord ratio** c_r = chord point; **efficacy** E = efficacy coefficient.
* **Deflection** δ = 0 (degrees).
* **Airfoil**: the `.afl` table (§4). Each wing loads its own; identical file, identical
  table.
* **Break reference** S_ref, from spawn positions (relative = world at spawn):
  * T₁ = 2.0·|(FRD − FLD) × (BLD − FLD)| and T₂ = 2.0·|(FRD − FLD) × (BRD − FRD)|,
    each product 2.0 × f32 evaluated [f64] and rounded to f32;
  * S_ref = max(T₁, T₂), taking T₂ unless T₁ > T₂;
  * S_ref ← S_ref·S_ref (f32).
* **Wing chains and induced drag.** Process wings in file order and group them into
  chains. A wing continues the current chain when its FRD node is the same node as the
  previous wing's FLD node; otherwise it starts a new chain.
  * **Chain area** A_c: the sum, in file order and in f32, of each member's area
    ½·(|(FRD − FLD) × (BLD − FLD)| + |(FRD − BRD) × (BLD − BRD)|), computed from spawn
    world positions (the two lengths are added in f32, then × 0.5).
  * **Chain span** b_c = |FRD(first wing of chain) − FLD(last wing of chain)|, from
    spawn positions.
  * When a chain closes (at a discontinuity, or after the last wing), its first wing
    gets induced drag with (b_c, A_c, side = right) and then its last wing gets
    (b_c, A_c, side = left).
  * In a one-wing chain these are the same wing and the second assignment wins, so a
    lone wing has induced drag on the **left** side, with its own span |FRD − FLD| and
    its own area.
  * Wings strictly inside a chain have no induced drag.
* **Propeller wash**: only vehicles with turboprops or turbojets have it. Those
  vehicles are out of scope; for every in-scope vehicle the wash term is zero.

### 3.3 Per-step inputs (pass 1a)

Skip the wing if it is broken (§3.8). Otherwise read the velocities and **relative**
positions of its nodes just integrated in this step's node pass, and the **world**
height of FLD:

* airflow w = −(v_FLD + v_FRD)·0.5, with speed V = |w|;
* chord vector c = ((x_BLD − x_FLD) + (x_BRD − x_FRD))·0.5, with chord length C = |c|;
* span vector b = ((x_FRD − x_FLD) + (x_BRD − x_BLD))·0.5, with span length B = |b|;
* lift direction (not normalised) l = b × (−w);
* panel area s = B·C;
* normal n = c × b, normalised.

### 3.4 Angle of attack α (degrees)

1. **Project the airflow onto the chord plane.**
   * q = n × c. Normalise q; it stays zero if its length is 0. q is the unit normal of
     the plane spanned by n and c.
   * Let the 3×3 matrix M have entries M_ij = δ_ij − q_i·q_j: the diagonal terms are
     1 − q_i·q_i, the off-diagonal terms −q_i·q_j.
   * p = M·(−w), with each component summed in column order
     (M_i0·g_x + M_i1·g_y + M_i2·g_z, where g = −w).
2. **Rotation from c to −p.** Let u = c and t = −p. Build a rotation quaternion in
   this exact way:
   * a = sqrt((u·u)·(t·t)) in f32. β = a + (t·u).
   * **Identity case**: if |β − 2a| ≤ 1.1920929e-7 (f32 machine epsilon, an absolute
     test) or a = 0, the rotation is the identity, so the angle is 0 and the axis is
     (1, 0, 0).
   * **Opposite case**: else if β < 1e-6·a, set β = 0. The axis is (−u.y, u.x, 0) when
     |u.x| > |u.z|, else (0, −u.z, u.y).
   * **General case**: otherwise the axis is u × t.
   * Quaternion Q = (w_q = β, x, y, z = axis). Normalise Q:
     * N_Q = sqrt(w_q² + x² + y² + z²), summed in that order;
     * multiply every component by (1/N_Q).
3. **Angle and axis from Q.**
   * σ² = x² + y² + z². If σ² > 0: θ = 2·acos(w_q), where acos returns 0 for
     w_q ≥ 1 and π for w_q ≤ −1. The axis is (x, y, z)·(1/sqrt(σ²)).
   * Otherwise θ = 0 and the axis is (1, 0, 0).
   * Convert θ to degrees: α = θ × r2d, where r2d = f32(180) / f32(4·atan(1)) is
     computed in f32 (≈ 57.29578).
4. **Sign.** If axis·b > 0, then α ← −α.

Mathematically, α is the signed angle between the chord and the in-plane airflow, in
[−180°, 180°]. The steps above are the reference evaluation; follow them to stay on the
same side of the table's 0.1° bins. The opposite case gives 180° (or −180°); both table
ends hold identical NACA0009 values.

### 3.5 Coefficients

* **Normal wing**: (c_z, c_x, c_m) = lookup(α, c_r, δ).
* **Stabilator**: (c_z, c_x, c_m) = lookup(α − δ, c_r, 0).

lookup(a, c_r, d) (all angles in degrees):

1. **Wrap a.**
   * k = trunc(a / 360.0) [f64 division, truncated to an integer];
   * a' = a − f32(k·360);
   * if a' > 180: a' −= 360; if a' < −180: a' += 360.
2. **Lift and moment index**: i = trunc((a' + 180)·10) in f32. This is floor, since the
   argument is ≥ 0. **No interpolation at run time**: the value of the 0.1° bin at or
   below a' is used.
3. **Drag index.**
   * a_d = a' + 1.15·(1.0 − c_r)·d [f64], rounded to f32;
   * wrap a_d the same way as a';
   * i_d = trunc((a_d + 180)·10).
4. **Values.** Let sg = −1 if d < 0, else +1.
   * c_z = CL[i] − 0.66·sg·(1.0 − c_r)·sqrt(|d|);
   * c_x = CD[i_d] + 0.00015·(1.0 − c_r)·d·d;
   * c_m = CM[i] + 0.20·sg·(1.0 − c_r)·sqrt(|d|).

   Each expression is [f64], rounded to f32, with sqrt(|d|) taken in f32.

With no input d = 0, so the shift terms are exactly 0 and (c_z, c_x, c_m) =
(CL[i], CD[i], CM[i]) whatever c_r is.

### 3.6 Forces

Air density ρ_W at height y = world y of FLD:

* t = 1.0 − 0.0065·y / 288.15 [f64], rounded to f32;
* p = 101325 × pow≈(t, 5.24947) in f32;
* ρ_W = p × 0.0000120896, where the constant is f64, the product is [f64], and the
  result is rounded to f32. This can differ by one ulp from the fuselage's f32 constant.

Then compute:

* **Profile drag**: W_f = k_d·w, with k_d = c_x·0.5·ρ_W·V·s [f64 → f32].
* **Induced drag** (only if the wing has it):
  * I = k_i·w, with k_i = (c_x·c_x·0.25·ρ_W·V·A_c·A_c) / (3.14159·b_c·b_c);
  * c_x·c_x is f32; the rest is [f64], each side evaluated left to right, then the
    quotient is rounded to f32.
* **Lift**: W_f ← W_f + k_l·l, with k_l = c_z·0.5·ρ_W·V·C [f64 → f32]. Lift is
  perpendicular to the span and to the airflow, and its magnitude scales with
  B·V·sin(angle between b and w).
* **Moment scalar**: μ = −c_m·0.5·ρ_W·V·V·s [f64 → f32].
* **Front slot force**: f₁ = W_f·e₁ + n·(E·μ).
  * e₁ = E·0.75/4 [f64 → f32].
  * The moment factor E·μ is an f32 product; the source divides it by (4·0.25), which
    is exactly 1.
* **Back slot force**: f₂ = W_f·e₂ − n·((E·μ)/3).
  * e₂ = E·0.25/4 [f64 → f32].
  * (E·μ)/3 is an f32 division, from (4·0.75) = 3.

Add the forces to the node forces in this order (repeated slots add repeatedly):

1. induced drag I, if present:
   * left side: to BLU, then BLD;
   * right side: to BRU, then BRD;
2. f₁ to FLD, FLU, FRD, FRU, in that order;
3. f₂ to BLD, BLU, BRD, BRU, in that order.

Totals:

* profile drag plus lift sums to E·W_f, split 75 % front and 25 % back (the
  centre of pressure at quarter chord);
* the moment pair is **not** force-free **(quirk)**: net (4 − 4/3)·E·μ along n.

### 3.7 Order among wings

Wings run in file order. Within the step they come after the node pass and before the
fuselage drag, commands and beams. Airbrakes, aero engines and screw propellers would
run before the wings in this slot; they are out of scope.

### 3.8 Breaking

A wing can break; it never stalls in any special way (stall is only what the table
holds). The check is **not per step**:

* It runs **once per frame, before that frame's steps** (before the origin rebase),
  on the node **world** positions left by the previous frame's last step. It also runs
  once at spawn.
* With X = FRD − FLD, Z_L = BLD − FLD and Z_R = BRD − FRD (world, f32):

  broken ← broken OR (|X × Z_L|² > S_ref) OR (|X × Z_R|² > S_ref)

  |·|² is the squared length, without a square root.
* Breaking is permanent for a locally simulated vehicle. (Network-replicated copies
  re-evaluate the flag without the OR; that case is out of scope.)
* A broken wing applies no force at all, including induced drag.
* In effect, a wing breaks when either half-parallelogram's doubled area exceeds twice
  the larger spawn value, i.e. when the panel area has roughly doubled.

### 3.9 Control deflection with no input

The deflection δ is set once per frame, before the steps, alongside the break check:

* input value s ∈ [−1, 1] (or a flap setting);
* δ = −s·min_deflection when s < 0, else δ = s·max_deflection.

With no input, all of these are 0:

* aileron, rudder and elevator inputs (an autopilot exists only on vehicles with aero
  engines, and with no mode engaged it also outputs 0);
* the flap setting, flap notch 0.

So δ = 0 for every surface type. The minimum and maximum deflection and the chord point
then have no effect, because every term they enter is multiplied by d = 0.

### 3.10 Edge cases

* V = 0: every force is ±0 and adding −0 leaves forces unchanged. α evaluates to 0
  (identity case, since a = 0).
* Degenerate panels (B = 0 or C = 0): s = 0. If c × b = 0, n stays zero, q = 0, M = I
  and p = −w.
* The Viper's wing panels are flat (each "up" node equals its "down" node). This is
  legal: the "ratio" quantities that would divide by the thickness are only used for
  control-surface visuals.

---

## 4. Airfoil data (`.afl`)

### 4.1 Where it comes from

RoR ships four tables in its source tree under `resources/airfoils/`:

* `NACA0009.afl` (used by the Viper and the `fusedrag` default);
* `Clark-Y.afl`;
* `NACA64.1.412.afl`;
* `NACA64.3.618.afl`.

They are packed into the game's resources.

* **License**: there is no separate notice in that directory. The RoR repository is
  GPL-3.0 (`COPYING` at its root), so treat the files as GPL-3.0 data. Load them at
  run time from a configured local path (for example the local RoR checkout). Do not
  copy them into Rubble.
* **Checksum**: NACA0009.afl, 24 814 bytes, SHA-256
  `43265ec42c720f332d427d722d60866b1ce84ce9c4fdaf169824b1994fb1523f`.
* **Name lookup**: a wing or fusedrag names the file by its bare file name.

### 4.2 Format

The format is X-Plane's airfoil format:

* a few header lines (`A`, `700 version`, `1234 device type code`, and one line of 30
  numbers). **All are ignored.**
* a line starting with the five characters `alpha` (`alpha cl cd cm:`);
* rows of `angle  cl  cd  cm`, whitespace separated, angles in degrees with exactly
  one decimal.

NACA0009 holds:

* integer degrees from −180 to −20 and from 20 to 180;
* 0.1° steps from −19.9 to 19.9.

That is 721 rows in all.

### 4.3 Building the tables

There are three tables, CL, CD and CM, each with 3601 f32 entries (index i ↔ angle
(i − 1800)/10 degrees). All start at 0. Ignore lines before the `alpha` line and empty
lines. For each later row:

1. **Index.** Parse the angle as an integer part and a one-digit fraction separated by
   `.`.
   * Until a row whose integer part and fraction are both 0 has been seen, the fraction
     is negated, so rows before the 0.0 row are taken as negative angles.
   * Index = 10·(integer part) + fraction + 1800.
   * For the shipped files (one decimal, with a 0.0 row) this is
     **index = 1800 + round(10·angle)**: "−0.5" → 1795, "−18.6" → 1614.
   * The integer part "−0" parses as 0, and the negation rule still gives the right
     sign.
2. **Values.** Store cl, cd and cm (parsed to f32) at that index.
3. **Fill gaps.** If the previous stored index j and this one, i, differ by more than
   1, fill j+1 … i−1 linearly, in f32:
   value[j+1+k] = value[j] + ((f32)(k+1)·(value[i] − value[j])) / (f32)(i − j),
   for k = 0 … i−j−2, operations in that order.
4. **Stop.** Stop reading after storing index 3600.

Entries before the first row or after the last stay 0. A missing file leaves all three
tables zero, and the wing or fuselage then makes only the zero-coefficient forces.

---

## 5. Frame and step ordering (consolidated)

Per frame (1/60 s, 33 steps of 0.5 ms):

1. For each vehicle: set wing deflections (0 with no input) and run the **wing break
   check** (§3.8) on the current world positions.
2. **Origin rebase** (existing).
3. 33 steps; each step, per vehicle:
   1. **Node pass** (existing): contacts, integration, then force restart with gravity
      and either + D (`fusedrag` vehicles) or air drag and turbulence (others).
   2. **Wings**, in file order: forces from this step's integrated velocities and
      positions.
   3. **Fuselage drag**: compute and store D from the front node's integrated velocity
      and height. It is used at the next step's restart.
   4. **Commands**: rest-length updates (with no input, self-centering beams only).
   5. **Beam pass**, then **self-collision** (existing).

Velocities used:

* wings use the velocity of step n (after step n's integration); their force is
  integrated in step n+1;
* the fuselage drag uses the same velocity, but its force enters at step n+1's restart
  and is integrated in step n+2;
* the command pass reads no velocities.

---

## 6. Verification plan (black-box parity against RoR)

The current RoR reference harness (`ror_ref`) does not step aero or commands, and it
refuses `commands` sections. Its owner, not the implementer, must first extend it to
run RoR's own:

* wing pass;
* fuselage-drag pass and its restart term;
* command pass;
* per-frame wing deflection and break update;

and to export per-node forces after any pass. Then compare:

1. **Airfoil tables**:
   * dump RoR's 3 × 3601 NACA0009 (and Clark-Y) tables; they must match exactly;
   * lookups at angles {−540, −180, −179.95, −20.05, −0.05, 0, 0.04999, 19.95, 179.99,
     180, 360.1} with d = 0, and with d = ±10°, c_r = 0.7: exact.
2. **One wing panel, one step**:
   * an 8-distinct-node box panel, 1 m span × 1 m chord × 0.05 m thick, each node 1 kg;
   * gravity, ground and beams off; all eight nodes given the same velocity;
   * speeds {0, 5, 30, 80} m/s;
   * angles of attack from −30° to +30° in 2.5° steps, plus ±89°, ±91° and 180°, with
     yaw 0 and 15°;
   * altitudes y ∈ {0, 1000, 5000} m;
   * efficacy {1.0, 0.76};
   * compare the per-node forces after the wing pass, to f32 tolerance (a few ulp).
3. **Induced drag and chaining**: three wings chained by shared FRD/FLD nodes, plus one
   lone wing:
   * which wings carry induced drag;
   * on which side;
   * b_c and A_c, through forces.
4. **Viper-layout panel** (4 distinct nodes, repeated slots): duplicate-slot
   accumulation, with efficacy 0.76 and 0.020.
5. **Wing break**: stretch a panel with prescribed node motion until its area doubles.
   Force must go to zero from the first frame whose start-of-frame check sees it, not
   mid-frame, and never come back.
6. **Fusedrag, explicit width**:
   * a 10-node block with `fusedrag` width 1.5, no ground, gravity off, launched at
     (30, 0, 0) m/s;
   * trajectory of every node for 2 s;
   * check D = 0 in step 1 and the one-step lag.
   Repeat at y = 500 and 3000 m (density path, constant 288.1).
7. **Fusedrag, width 0** (the Viper case): the same block with `width 0.0` must show no
   drag at all. Compare against the same block without `fusedrag`, whose 0.05 drag with
   turbulence off must differ.
8. **Fusedrag autocalc**: block with known y/z extents and coefficient 0.5; check W
   through the force magnitude.
9. **Commands, no input**:
   * Viper idle on flat ground, 5 s: command-beam rest lengths unchanged, node
     positions match;
   * a pull test where the beam force lies between S and G: in RoR it must **not** break;
   * a rope command under compression (k = 0, c × 0.1).
10. **Self-centering** with limits (0.8, 1.5) (C = 1.15) and unequal rates: L(t) per
    step, including the final snap or stop, for both key orders (contract key < extend
    key and the reverse).
11. **Integration**: Viper with wings and fusedrag, dropped from 0.5 m and pushed to
    40 m/s, 10 s. Compare node positions to the existing RoR-compatible tolerance;
    compare total aerodynamic force per frame.
