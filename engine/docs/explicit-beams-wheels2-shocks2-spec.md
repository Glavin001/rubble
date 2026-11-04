# Explicit beams: `meshwheels`, `meshwheels2` and `shocks2`

This is a behavioural specification for adding three `.truck` sections to Rubble's explicit node–beam solver:

* `meshwheels`: a soft tyre with a visual rim mesh;
* `meshwheels2`: a soft tyre whose rim beams are stiffer than its tread;
* `shocks2`: a progressive shock with separate compression and extension rates.

It extends `explicit-beams.md`, which already specifies plain `wheels` (soft tyres), `shocks` and the step. It also extends `explicit-beams-aero-commands-spec.md`, which covers wings, fusedrag and commands. Terms not defined here mean what they mean in those documents. That includes:

* the step h = 0.0005 s;
* the node pass and the beam pass;
* the gate G and the strength S;
* the kinds (plain, limited spring, support, rope) and the types (deformable, actuator, virtual);
* the limited-spring rule;
* the tyre ring rule.

The reference is Rigs of Rods (RoR), judged as a black box. The goal is agreement to float tolerance, at spawn as well as over time:

* node positions, masses and flags, in the same index order;
* beams in the same index order, with the same parameters;
* the same per-step forces.

Behaviours that are artefacts of the reference rather than physics are marked **(quirk)**. The RoR-compatible profile must reproduce them.

Notation: all arithmetic is IEEE single precision (f32). An expression `a + b·c` is evaluated left to right as written, with each product rounded before the sum. No fused multiply–add is used anywhere in this document. `min(x, 1)` means "1 if 1 < x, else x", so a NaN x passes through. `max(a, b)` means "b if a < b, else a".

---

## 0. Summary of the differences

| section | nodes per wheel/line | beams per wheel/line | differs from `wheels` in |
|---|---|---|---|
| `meshwheels` | 2n tyre nodes | 8n, or 9n with a rigidity node | argument layout only (tyre radius first; extra rim radius, side, mesh and material arguments, all visual) |
| `meshwheels2` | 2n tyre nodes | 8n, or 9n with a rigidity node | argument layout as `meshwheels`; the 4 rim beams per ray take spring and damping from `set_beam_defaults`; the two axle-to-own-ring spokes get long bound 0.15 instead of 0 |
| `shocks2` | none | 1 | a new beam kind with a velocity-direction-dependent, position-progressive k and c, with hard or soft bump limits |

The `meshwheels2` "rim" is **not** a separate ring of rim nodes. It is the same 2n-node tread ring as `wheels`. Only the ring's own reinforcement beams take the rim spring. `wheels2` and `flexbodywheels` build a real rim ring (4n nodes); see §3.

---

## 1. `meshwheels` and `meshwheels2`: file format

Both sections have the same 16 comma- or space-separated arguments. A line with fewer than 16 is rejected; it produces no wheel, no nodes and no legacy node numbers.

| # | name | type | physics use |
|---|---|---|---|
| 0 | tyre radius R | float, m | **yes**: the ring radius |
| 1 | rim radius | float, m | none (visual mesh scale only) |
| 2 | width | float | none: the physical width is always the axle-node distance |
| 3 | rays n | int | yes |
| 4, 5 | axle nodes | node refs | yes |
| 6 | rigidity node | node ref, or `9999` for none | yes |
| 7 | braking | int 0–4 (any other value means 0, with an error logged) | none with no input |
| 8 | propulsion | int 0–2 (any other value means 0) | none with no input |
| 9 | reference arm node | node ref (required; RoR dereferences it unconditionally) | none with no input (wheel reaction torque only) |
| 10 | mass M | float, kg | yes |
| 11 | spring kₜ | float | yes |
| 12 | damping cₜ | float | yes |
| 13 | side | char `l` or `r` (anything else means `l`, with a warning) | none (mesh mirroring) |
| 14 | mesh name | string | none |
| 15 | material name | string | none |

Compared with `wheels` (radius, width, rays, n1, n2, snode, braked, propulsed, arm, mass, spring, damping, face material, band material), the mesh sections insert the rim radius after the tyre radius. They replace the two materials with side, mesh and material.

Each line also captures, at its position in the file, the node defaults (`set_node_defaults`) and the beam defaults (`set_beam_defaults`, `set_beam_defaults_scale`) in force at that line, exactly as `wheels` does. Beam defaults that no directive has set yet are:

| field | value |
|---|---|
| springiness | 9 000 000 |
| damping | 12 000 |
| deform | 400 000 |
| break | 1 000 000 |
| plastic coefficient | 0 |
| scales | all 1 |

A negative springiness or damping in `set_beam_defaults` is replaced by 9 000 000 or 12 000 respectively.

## 2. `meshwheels` and `meshwheels2`: spawn

### 2.1 Where in the order

Generated nodes and beams are appended **section by section, not in file order**:

1. truck nodes (numbered, then named);
2. cinecam nodes and beams;
3. every `wheels` line;
4. every `wheels2` line;
5. every `meshwheels` line;
6. every `meshwheels2` line;
7. every `flexbodywheels` line;
8. then `beams`, `shocks`, `shocks2`, `shocks3`, commands and hydros, each section in file order.

So a `meshwheels2` line written before a `wheels` line still spawns after it. Within a section, lines go in file order. Each wheel line appends its 2n nodes and then its beams before the next wheel line starts. Within the whole file, the beam order is therefore:

* cinecam beams;
* all `wheels` beams;
* all `meshwheels` beams;
* all `meshwheels2` beams;
* then the file's beams and so on.

**Legacy node numbers.** Files that address nodes by number may name generated tyre nodes. Number assignment and resolution work like this:

* The parser hands out legacy numbers **in file order** as it meets each node-generating line:
  * a `meshwheels` or `meshwheels2` line takes 2n consecutive numbers;
  * the axle rings alternate A, B, A, B…;
  * whether or not a rigidity node is given has no effect on the count.
* A legacy number resolves to the node's real index:
  * start from the count of nodes from all sections spawned before its section, in the section order above;
  * add its index among the generated nodes of its own section.
* The existing loader already does this for `wheels`; extend its section offsets:
  * `meshwheels` sits after `wheels` and `wheels2`;
  * `meshwheels2` sits after `meshwheels`.

### 2.2 Axle nodes and the ring

These follow exactly the tyre ring rule of `explicit-beams.md`, with R = the tyre radius (argument 0):

* A1 is the axle node of lower z (the file's first on a tie), A2 the other. The two must differ, or the line is rejected.
* axis = normalize(A2 − A1). The first spoke is s = normalize(axis × (1,0,0)), or normalize(axis × (0,1,0)) when the first has squared length < 1e-12; then s ← R·s.
* The rotation is θ = −360°/(2n), as an f32 quaternion.
* For i in 0..n:
  * outer node Oᵢ = A1 + s, then rotate s;
  * inner node Iᵢ = A2 + s, then rotate s.
* Node index = base + 2i for Oᵢ and base + 2i + 1 for Iᵢ, where base is the node count before the line.

Positions are computed from the axle nodes' spawned (relative) positions.

### 2.3 Node properties (identical for both sections and for `wheels`)

Every generated node gets:

* **mass** M/(2n), evaluated as `M / (2·n)` with 2·n converted to f32 first. It is final: the node-mass recalculation skips tyre nodes, and `minimass` does not apply to them.
* **flags**: contacter, tyre. Ground contact stays **on**. The no-ground-contact option and every other `set_node_defaults` option except buoyancy (`b`) are ignored for generated nodes. They are not loaded or override-mass nodes.
* **friction coefficient** = the node-defaults friction at the line (default 1). The volume and surface coefficients come from the same defaults. The width does **not** enter the friction (unlike `wheels2` tyre nodes, §3).

### 2.4 Beams, per ray i (next = (i + 1) mod n), in this order

Let (kₜ, cₜ) be the tyre spring and damping (arguments 11, 12). The rim pair (kᵣ, cᵣ) depends on the section:

* **`meshwheels`**: kᵣ = kₜ and cᵣ = cₜ. The section is physically identical to `wheels`.
* **`meshwheels2`**:
  * kᵣ is the beam-defaults springiness at the line, **unscaled**: `set_beam_defaults_scale`'s spring factor is **not** applied. **(quirk)**
  * cᵣ is the beam-defaults damping, also unscaled.
  * With no `set_beam_defaults` before the line, these are 9 000 000 and 12 000.

| order | ends | k, c | kind | bounds |
|---|---|---|---|---|
| 1 | (A1, Oᵢ) | kₜ, cₜ | limited spring | sb = 0.66, lb = 0 (`meshwheels`) or **0.15** (`meshwheels2`) |
| 2 | (A2, Iᵢ) | kₜ, cₜ | limited spring | as row 1 |
| 3 | (A2, Oᵢ) | kₜ, cₜ | plain | — |
| 4 | (A1, Iᵢ) | kₜ, cₜ | plain | — |
| 5 | (Oᵢ, Iᵢ) | kᵣ, cᵣ | plain | — |
| 6 | (Oᵢ, O_next) | kᵣ, cᵣ | plain | — |
| 7 | (Iᵢ, I_next) | kᵣ, cᵣ | plain | — |
| 8 | (Iᵢ, O_next) | kᵣ, cᵣ | plain | — |
| 9 (only with a rigidity node P) | (P, Oᵢ) if \|P − A1\| < \|P − A2\| (strict, spawned relative positions), else (P, Iᵢ) | kₜ, cₜ | plain, **virtual** type | — |

Notes on the table:

* The first end listed is the beam's node a (force sign convention as in `explicit-beams.md`).
* The bounds are f32 constants: 0.66 is f32(0.66), and 0.15 is f32(0.15) (converted from the double literal 0.15).

Every beam of the wheel, rows 1–9, shares these values:

* **Type**: deformable, except row 9 (virtual).
* **Detacher group**: 0.
* **Rest length**: L = L₀ = |a − b| from the spawned f32 relative positions.
* **Strength**: S = the beam-defaults break threshold at the line, **unscaled** (the break scale is not applied).
* **Thresholds**: Tc = Tt = G = the deform threshold computed exactly as for `beams`, which is:
  1. d = 400 000;
  2. if `set_beam_defaults` was used: d = its deform value, raised to 400 000 unless `enable_advanced_deformation` preceded it;
  3. the creak floor: if d < 100 000 then d = 100 000, unless the plastic coefficient was user-given and ≥ 0;
  4. multiply by the deform scale of `set_beam_defaults_scale`.
* **Plastic coefficient**: from the beam defaults.
* **Limited springs** (rows 1–2): kℓ = 9 000 000 and cℓ = 12 000. These are constants, not the beam defaults: a wheel spoke is not an actuator. Limited springs never yield, but they can break.

`meshwheels2`'s long bound of 0.15 means that the axle-to-own-ring spokes behave as follows:

* they harden only once stretched more than 0.15·L;
* `wheels` and `meshwheels` spokes harden on **any** stretch (lb = 0);
* all three harden under compression beyond 0.66·L.

### 2.5 Effects on other nodes' masses

Wheel beams take part in the dry-mass distribution like any non-virtual beam (`explicit-beams.md` defers this to the existing loader, which must now count these beams too):

* each beam adds L₀/2 to the total length, once per end that is **not** a tyre node;
* so rows 1–4 contribute L₀/2 each, all on the axle-node side;
* rows 5–8 contribute nothing;
* row 9 is virtual and is excluded.

The total length is an f32 running sum over all beams in beam-index order. Matching the beam order above therefore matters at float tolerance. Then each axle node gets L₀·dry_mass/len/2 per such beam.

### 2.6 Visual-only and run-time-inert arguments

The following arguments have no effect on physics:

* rim radius, width, side, mesh and material;
* the tuning overrides of rim radius and media.

The following matter only with driver input or the parking brake, which the black-box scenes keep off:

* braking;
* propulsion;
* the reference arm node.

With zero torque, RoR's wheel pass adds exactly zero force. These wheels need nothing beyond `wheels`.

---

## 3. Relation to `wheels2` (and `flexbodywheels`)

`wheels2` shares no node or beam builder with `meshwheels2`, despite the name:

* it builds 4n nodes, n·2 rim nodes then n·2 tyre nodes;
* its rim ring starts at (0, R_rim, 0) rather than the perpendicular spoke, and is rotated by −360°/n;
* its tyre ring is offset by −180°/n;
* its rim nodes weigh M/(4n);
* its tyre nodes weigh 0.67·M/(2n) and have friction 2·width;
* it has 24n beams, or 25n with a rigidity node;
* it has its own rim and tyre spring arguments.

It is a separate feature and out of scope here. A file containing `wheels2` or `flexbodywheels` must still fail with a clear "unsupported section" error. It must not be treated as `meshwheels2`.

---

## 4. `shocks2`: file format

The line has 13 required arguments and an optional 14th. A line with fewer than 13 is rejected.

| # | name | symbol |
|---|---|---|
| 0, 1 | nodes a, b | |
| 2 | spring in (compression) | k_in |
| 3 | damping in | c_in |
| 4 | spring-in progression factor | p_k,in |
| 5 | damping-in progression factor | p_c,in |
| 6 | spring out (extension) | k_out |
| 7 | damping out | c_out |
| 8 | spring-out progression factor | p_k,out |
| 9 | damping-out progression factor | p_c,out |
| 10 | short bound (max contraction) | sb |
| 11 | long bound (max extension) | lb |
| 12 | precompression | π |
| 13 | options (optional string) | |

**Options** are a string of characters; each character is handled as follows:

* `i`: invisible;
* `s`: soft bump;
* `m`: metric bounds;
* `M`: absolute metric bounds;
* `n` and `v`: no-ops;
* any other character is ignored, with a warning.

**Validation:**

* A line with any numeric field (2–12) below −0.8 is discarded.
* **(quirk)** After a discarded line, the next `shocks2` line in the same module is not validated at all, and is kept even if invalid.
* There is no other clamp. A precompression of 0 is accepted; the docs' "default 1.0" is only advice, since the argument is required.

The line also captures the beam defaults and the current `detacher_group` at its position.

## 5. `shocks2`: spawn

Each line appends one beam (after all `beams` and `shocks` beams, in file order) and one shock record:

1. **Bounds.** Start with sb, lb from the file, and apply the options in this order:
   * `m`: let ℓ_w = |a − b| from the nodes' **world** (absolute) spawn positions, in f32. Then sb ← sb/ℓ_w and lb ← lb/ℓ_w.
   * `M` (applied after `m` if both are given, **(quirk)**: then it acts on the already-divided values): with the same ℓ_w,
     * sb ← (ℓ_w − sb)/ℓ_w and lb ← (lb − ℓ_w)/ℓ_w;
     * then, if lb < 0, lb ← 0;
     * then, if sb > 1, sb ← 1.
2. **Beam.**
   * Kind: shock2. Type: actuator (never yields).
   * Detacher group: the current group.
   * Strength: S = 4 × the beam-defaults break threshold, **unscaled**, as an f32 product.
   * Gate: G = the deform threshold by the rule of §2.4 (it is *not* S). The plastic coefficient comes from the beam defaults (it has no effect).
   * L₀ = |a − b| from the spawned f32 **relative** positions (not ℓ_w).
   * Then L = L₀·π, and the reference (spawn) length is also L₀·π.
   * The beam's nominal k and c are k_in and c_in. They are fully overwritten every step (§6), so they matter only for reporting.
3. **Shock record.**
   * k_in, c_in, k_out, c_out and the four progression factors, from the file.
   * The bump spring k_b and bump damping c_b: the beam-defaults springiness and damping at the line, **unscaled** (default 9 000 000 and 12 000).
   * The soft-bump flag.

In the dry-mass distribution the shock beam is non-virtual and counts with its precompressed length L₀·π. It counts for both ends unless an end is a tyre node.

The `m`/`M` conversions use the un-precompressed world length. The step uses the precompressed L. **(quirk)**

## 6. `shocks2`: the force law (every step, inside the beam pass)

There is no separate shock pass. The rule below replaces step 2 ("kind rules") of the beam pass in `explicit-beams.md` for a shock2 beam. It runs at the beam's index in the normal beam loop, so it reads the positions and velocities produced by this step's node pass.

From step 1 of the beam pass: e = ℓ − L (positive when extended), and the relative speed u = (v_a − v_b)·d·ι (positive when the ends separate).

### 6.1 Base rates (velocity direction picks the side)

* If **u > 0** (extending):
  * k = k_out and c = c_out;
  * λ = 1 if lb = 0, else λ = min(r·r, 1) with r = e / (lb·L);
  * k ← k + (p_k,out·k)·λ, then c ← c + (p_c,out·c)·λ.
* Otherwise (**u ≤ 0**, including u = 0, compressing):
  * k = k_in and c = c_in;
  * λ = 1 if sb = 0, else λ = min(r·r, 1) with r = e / (sb·L);
  * k ← k + (p_k,in·k)·λ, then c ← c + (p_c,in·c)·λ.

Properties of the base rates:

* The side is chosen by velocity, not by position. A shock that is compressed but moving outwards uses the *out* rates, with λ measured against lb.
* With a zero bound, the progression is always fully on (λ = 1). **(quirk)**
* r·r makes λ independent of the sign of e.
* The products `lb·L` and `sb·L` are f32. So is the division.

### 6.2 Hard bump (no `s` option)

If e > lb·L or e < (−sb)·L, then k = k_b and c = c_b. This is a replacement, not an interpolation, unlike `shocks`. It applies whatever the direction of u.

### 6.3 Soft bump (`s` option)

This replaces §6.2. Let L′ = L·0.8, e_l = lb·L′, e_s = (−sb)·L′. The cases below are tested in order.

* **Near or past the long bound**, if e > e_l:
  1. Recompute the out rates with progression exactly as in §6.1's extending case, whatever u is.
  2. Then compute the bump factor μ: 1 if lb = 0, else μ = min(q·q, 1) with q = ((e − e_l)·5) / (lb·L).
  3. Apply k ← k + (k + 100)·p_k,out·μ and c ← c + (c + 100)·p_c,out·μ. Evaluate each as `k + (((k + 100)·p)·μ)`.
  4. Then, if u < 0, set k = k_in and c = c_in, the bare rates with no progression.
* **Near or past the short bound**, else if e < e_s:
  1. Recompute the in rates with progression exactly as in §6.1's compressing case.
  2. Then compute μ: 1 if sb = 0, else μ = min(q·q, 1) with q = ((e − e_s)·5) / (sb·L).
  3. Apply k ← k + (k + 100)·**p_k,out**·μ and c ← c + (c + 100)·**p_c,out**·μ. The *out* factors are used here. **(quirk)**
  4. Then, if u > 0, set k = k_out and c = c_out (bare).
* **Otherwise**: keep the §6.1 values.

Then, in all three cases, if e > lb·L or e < (−sb)·L: k ← max(k, k_b) and c ← max(c, c_b).

### 6.4 Force, yield and break

The rest is the ordinary beam pass with these k and c:

* **Force**: F = (−k)·e − c·u, and F is stored as the beam's last force.
* **Yield**: none. The beam is an actuator, so L never changes.
* **Break**: as for any beam. If |F| > G and |F| > S, the beam breaks (F ← 0), unless the collision-triangle exception saves it. In that case S ← 2·G.
  * For that exception, the live-beam count excludes all bounded beams (shocks, limited springs, supports, ropes), as in `explicit-beams.md`.
  * A breaking master of detacher group g > 0 breaks its whole group.
* **Apply**: as usual.

### 6.5 State

These values are fixed at spawn and never change during a step:

* L and the reference length;
* the bounds;
* the rates and the bump rates.

The only per-step state is the beam's broken flag and its last force; nothing is integrated. RoR's active-stabilizer pass (`shocks` with `L`/`R` options) never touches `shocks2` beams. A broken shock2 stays broken and is skipped.

### 6.6 Worked checks (f32)

Take L = 0.5 with:

* k_in = 50 000, c_in = 3 000, k_out = 40 000, c_out = 2 000;
* p_k,in = 1, p_c,in = 0.5, p_k,out = 0.5, p_c,out = 2;
* sb = 0.2, lb = 0.3, k_b = 9e6, c_b = 12 000.

| case | e | u | expected k | expected c |
|---|---|---|---|---|
| at rest | 0 | 0 | 50 000 (λ = 0) | 3 000 |
| compressing, half bound | −0.05 | −1 | 50 000 + 50 000·0.25 = 62 500 | 3 000 + 1 500·0.25 = 3 375 |
| extending, at long bound | 0.15 | 0.5 | 40 000 + 20 000·1 = 60 000 | 2 000 + 4 000·1 = 6 000 |
| hard bump past short bound | −0.11 | −1 | 9e6 | 12 000 |
| compressed but moving out | −0.05 | 0.2 | λ = (−0.05/0.15)² = 0.111…; 40 000 + 20 000·λ | 2 000 + 4 000·λ |

---

## 7. Position in the step

There is nothing new in the frame or step order:

1. node pass;
2. wings;
3. fusedrag;
4. commands;
5. **beam pass**, which now includes the shock2 rule and the meshwheel beams;
6. self-collision.

In RoR's step, the wheel, stabilizer, hydro and command passes run between the node pass and the beam pass. With no input they add no force and change no `shocks2` or tyre state. The shock2 rule reads the state just integrated in the same step. Its force, like every beam force, is integrated in the next step's node pass.

**GPU.** A shock2 beam is just another beam in the beam loop. It needs these extra per-beam fields:

* k_in, c_in, k_out, c_out;
* the four progression factors;
* k_b, c_b;
* a soft-bump bit.

Alternatively, a side table indexed from the beam can hold them. The branches are per-lane data branches with no barrier inside, so Chrome's uniform-control-flow rule is unaffected. Meshwheel beams are ordinary plain and limited-spring beams; only their spawn parameters differ.

---

## 8. Verification plan (black-box parity against RoR)

**Harness work first.** The RoR reference harness (`inspiration/ror-harness`, `ror_ref`) needs extending by its owner. RoR's code must stay verbatim:

* extract `ProcessMeshWheel`, `ProcessMeshWheel2`, `ProcessShock2` and `Actor::CalcShocks2`;
* stub `CreateMeshWheelVisuals` and beam visuals;
* keep the parking brake off and give no input;
* add an export of the node table (index, position, mass, flags, friction) and the beam table (index, ends, k, c, L, reference length, bounds, kind, type, G, Tc, Tt, S, detacher group, shock parameters), written right after spawn and node-mass recalculation;
* add a per-step export of per-beam force.

Then:

1. **Single `meshwheels2` wheel spawn.** Use two axle nodes on a fixed chassis triangle, with n ∈ {4, 12, 24} and R = 0.45:
   * with and without a rigidity node;
   * axle nodes listed in both orders, including equal z;
   * axis along z, along x (fallback spoke) and skewed.

   Compare node by node (positions bitwise or within 1 ulp; mass; contacter, tyre and ground flags; friction) and beam by beam (ends, order, every parameter). Repeat with:
   * `set_beam_defaults 700000, 350, 60000000, 80000000` before the line, to show kᵣ/cᵣ;
   * `set_beam_defaults_scale 2, 2, 2, 2`, to show the unscaled rim spring and break threshold but the scaled deform threshold.
2. **`meshwheels` versus `wheels`.** The same geometry written as `wheels` and as `meshwheels` must give an identical node table, beam table and 5 s trajectory. A `meshwheels2` line with beam defaults equal to its tyre spring must differ only in the 0.15 long bound.
3. **Section order and legacy numbers.** Use one file with `meshwheels2`, `wheels` and `meshwheels` lines in that file order, numbered nodes, and a `beams` line naming a generated tyre node by legacy number. The node order must be wheels, then meshwheels, then meshwheels2. The beam must attach to the same physical node as in RoR.
4. **Masses.** Use a car with `meshwheels2` and `shocks2` (precompression 0.9) and check every axle and chassis node mass. This checks that wheel spokes count on the axle side, rim beams not at all, and shocks with their precompressed length.
5. **`shocks2` element at set rates.** Use two 1000 kg nodes with gravity, ground and drag off and node b immovable. Prescribe node a's velocity along the axis at constant rates u ∈ {−2, −0.5, 0, +0.5, +2} m/s. Sweep e from −1.2·sb·L to +1.2·lb·L. Compare F for every step to f32 tolerance, for:
   * hard bounds, and soft bounds (`s`);
   * zero bounds (always-full progression);
   * zero progression factors;
   * options `m`, `M`, `mM`, including an `M` line whose long bound is shorter than the spawn length, which clamps to 0;
   * precompression {1, 0.8, 1.2};
   * the reversal points: just inside and just outside e_l, e_s, lb·L and −sb·L, with u of both signs.
6. **`shocks2` break.** Use beam defaults with break 1000 and a pull ramp. The beam must break on the first step with |F| > max(G, 4000). Break it as a detacher-group master and check that the group goes with it.
7. **Validation.** Give one `shocks2` section a line with spring_in = −1 followed by another invalid line. RoR must drop the first and keep the second (the quirk). Rubble must match, or fail loudly if its loader rejects the second line by design.
8. **Integration.** Use a sedan with four `meshwheels2` and four `shocks2` (soft bump on the rear), dropped from 0.3 m onto flat ground for 5 s, then a 30 km/h curb strike. Compare all node trajectories to the existing RoR-compatible tolerance, and compare the broken and bent beam counts.
