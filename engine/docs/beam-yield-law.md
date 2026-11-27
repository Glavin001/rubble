# A beam yield law that lets a lattice member crush: design, CPU prototype, measurements

Written 2026-11-27 (branch `crash-ref/beam-law`, from `claude/crash-reference-suite` at
`3ab7bd268`). The problem is `front-load-path.md` §4 and §6: a real thin-walled member peaks, folds
at a lower near-constant force over about 73 % of its length, densifies, and keeps its bending
strength throughout; our JBeam beam only hardens, cannot shorten past 0.1 m, and a lattice cell of
corner chords set to the member's folding force has nothing left to resist bending, so bays turn
instead of folding.

**Status:** a CPU prototype behind opt-in per-beam keys and a generator switch, both off by
default; no kernel changed. Facts, measurements and estimates are marked. Every number here is our
own, from the code and runs named.

Code:
- solver `engine-softbody/src/explicit_crush.rs` (the law, crush cells, the GPU refusal); hooks in
  `explicit_jbeam_ext.rs` (`attach_ext`, `ext_beam`, `jbeam_ext_force_with`), `jbeam.rs`
  (`read_keys`), `desc.rs` (`JbeamBeam::crush`); `engine-exec` `ExplicitUpload::with_sleep`;
- ledger `crash_reference/metrics/beams.rs` (`Plastic::of`, `CellPlastic`), `metrics.rs`;
- calculator `engine-vehiclegen/src/crush_law.rs`, `thin_wall/crush.rs` (`box_axial_peak`,
  `fold_half_length`); generator switch `Physics::front_rules.crush_law`
  (`gen2/crush_law.rs`, the second face diagonals in `gen2/load_path.rs`);
- tests `engine-softbody-lab/tests/beam_yield_law.rs`; tools `tools/beam-law/` (`identity.sh`,
  `plot.py`); plots `out/beam-law/` (copy in `~/Development/rubble-artifacts/beam-law/`).

Reproduce (from `engine/`):

```
cargo test --release -p engine-softbody-lab --test beam_yield_law                      # asserted (11)
cargo test --release -p engine-softbody-lab --test beam_yield_law -- --ignored --nocapture  # curves, off-centre table
tools/beam-law/identity.sh base; (change); tools/beam-law/identity.sh after; diff -r out/beam-law/ident/{base,after}
tools/sections/front_ab.sh lpx load_path crush_law; python3 -I tools/sections/front_ab.py off lp lpx
python3 -I tools/sections/load_path_spans.py off lp lpx
tools/sections/load_path_hard.sh lpx load_path crush_law; python3 -I tools/sections/front_hard_rules.py lpx
$PY tools/beam-law/plot.py off lp lpx         # PY: a venv with matplotlib
```

## 1. Answer first: do existing parameters suffice?

**Partly, and not for the part that matters.** Tested on one beam through the solver
(`existing_parameters_peak_then_plateau_by_a_fuse`):

- **Peak then plateau: yes, with two beams.** A NORMAL beam flat at the plateau Q
  (`deformLimitStress` = `beamDeform` = Q) in parallel with a "fuse" NORMAL beam (`beamDeform`
  FLT_MAX, `beamStrength` P − Q) peaks at P = 50.0 kN and drops to 20.0 kN in one step when the
  fuse breaks (measured). The drop is a jump (no softening length), the fuse also breaks in tension,
  and it doubles the beams.
- **Separate compressive and tensile yield: not in JBeam.** The beam has `tc` and `tt`, but JBeam
  has one `beamDeform` for both; only the `.truck` extension `rubble_tension_yield` sets Tt alone
  (spec §18.4). A NORMAL beam in parallel with a near-tension-only ANISOTROPIC beam gives Tt > Tc
  at the cost of a second beam (not needed: see §3.2).
- **A lower least rest length: no.** `deformLimit` can only raise the floor; `MIN_REST` = 0.1 m is
  a solver constant.
- **Bending (and shear) strength while a cell folds: impossible with any per-beam law** (proof
  §3.5). This is what turns the bays, so new solver code was needed.

## 2. Survey: what is established

Sources are cited by author, year or document; those we could not check here are marked
*unverified* (none of the papers was at hand; nothing was read from BeamNG or Rigs of Rods code).

1. **Axial collapse of thin-walled members** (peak, plateau, densification). The mean folding
   force of square tubes (Wierzbicki & Abramowicz 1983; Abramowicz & Jones 1984, 1986; already in
   `thin_wall::crush`), the effective crush distance 0.73 of the fold (Abramowicz 1983), and the
   initial peak being a buckling/ultimate load of the walls, reached in the first millimetres, then
   a drop to the folding force as the first lobe forms (Jones, *Structural Impact*, ch. 7,
   *unverified chapter*). We take the peak as the effective-width ultimate load σ_y·A_eff (AISI
   S100, EN 1993-1-5 §4.4: confident as the stub-column ultimate load; as a crush peak an
   estimate) and the drop over the first fold's half-length H = 0.99 t^(1/3) c^(2/3) (Wierzbicki &
   Abramowicz 1983, *coefficient unverified*).
2. **Lumped and "hybrid" crash models.** Kamal's lumped mass–spring model (1970, *unverified*)
   represents each crush zone by a nonlinear spring whose force–deflection curve comes from a
   component test: peak, plateau, densification, with unloading along an elastic slope. Hybrid
   models (lumped masses with beam elements) add moment–rotation curves for bending hinges.
3. **Bending collapse.** Kecman (1983) gives the moment of a rectangular tube's plastic hinge: a
   maximum set by the compression flange's effective width, then falling with rotation. We have
   only an upper estimate in its spirit (`box_bending_collapse`: σ_y·Z with the flange at its
   effective width), *not Kecman's formula*.
4. **Macro-elements** (Abramowicz, "Thin-walled structures as impact energy absorbers", Thin-Walled
   Structures 41, 2003, *unverified title*): a member is a beam whose axial crush and hinge
   rotations are separate generalised degrees of freedom, each with its own collapse curve. Crash
   codes carry the same idea as 6-DOF discrete beams with one load curve per DOF (LS-DYNA's
   nonlinear plastic discrete beam materials, *unverified material numbers*). **This is how such
   models keep bending strength while collapsing axially: the generalised forces yield
   separately.** Our crush cell (§3.5) is that, mapped onto four chords.
5. **Unloading and reloading.** Elastic unloading along the initial stiffness from the current
   point, reloading elastically to the current yield, no energy returned beyond the elastic part:
   the standard return mapping of rate-independent plasticity (Simo & Hughes, *Computational
   Inelasticity*, 1998). A fold does not unfold: the compressive set only grows.
6. **Rate.** Cowper & Symonds (1957) on the flow stress, applied to the whole folding force
   (Abramowicz & Jones 1984): the load path's `real_crash` factor already scales the plateau; the
   prototype scales the peak, moments and shear the same way (an estimate for the peak).
7. **JBeam's parameters in our implementation** (`jbeam-semantics-spec.md` §3–5, §19, the code):
   `beamDeform` = Tc = Tt; `beamStrength` the break; `deformLimit` / `deformLimitExpansion` floor
   and ceiling of the rest length (fractions of L₀; at a limit the beam is a spring about it);
   `deformLimitStress` caps the hardening (`min(cap.max(tc))`: it can hold the threshold flat, never
   lower it); `beamPrecompression` moves the rest length; `breakGroup` breaks together;
   BOUNDED's limit springs and SUPPORT's compression-only spring never yield as crush (bounded) or
   yield as NORMAL in compression (support); `deformationTriggerRatio` only flags a deform group
   (visuals). None lowers the yield force with set, none couples beams.

## 3. Candidate laws, with their proofs

Notation: k spring, L rest, ℓ length, e = ℓ − L, F = −k·e − D(u) (compression positive),
s the compressive plastic set (m), P peak, Q plateau, d drop length, L₀ spawn rest.

### 3.1 (a) Existing parameters (no solver change)

The fuse pair of §1: F = min-of-states sum; energy: the plateau beam's ∫Q ds; the fuse's ½S²/k is
removed at the break (a break stores nothing afterwards; no energy created; the ledger loses it to
"broken"). Bytes: a second beam record (the GPU's beam record). Unload/reload: the plateau beam's
JBeam rule. It cannot hold bending (§3.5) and the hardening default must be capped by
`deformLimitStress` for a flat plateau.

### 3.2 (b) Compressive yield separate from tensile

The beam already has Tc and Tt and the yield rule keeps them apart (spec §18.4); the prototype
reads `rubbleTensionYield` into Tt (no new state). Use: a wall in tension does not buckle, so a
chord's Tt is its squash share, and **a face diagonal's Tt is its wall's plastic shear over the
diagonal's sine while its Tc stays at its small axial share** (§5.3): this is what gives a cell its
shear strength while it folds. Energy: the JBeam rule's (each yield moves the rest toward the
current length).

### 3.3 (c) Peak, drop, plateau, densification ("the crush law")

Capacity C(s) = P + (Q − P)·s/d for s < d, Q for s ≥ d. Per step, compression only (e < 0, the
elastic force g = −k·e, damping not read): if g > C(s), the set grows by Δ solving g − kΔ = C(s+Δ)
(closed form on each piece: Δ = (g − C(s))/(k − H) on the drop, H = (P − Q)/d; (g − Q)/k on the
plateau), the rest moves by −Δ, never below the floor L_min = max((1 − `deformLimit`)·L₀,
`rubbleMinRest`) (densified: a spring about L_min), and the force applied is the post-flow elastic
force −k(ℓ − L) − D(u) (no lag, no halfway rule). Tension: the JBeam rule with Tt. State: s (4 B).
Parameters: Q, d, L_min, Tt (16 B; P is `beamDeform`).

- **Plastic work** W = ∫C ds ≥ 0; each yield moves the rest toward the current length, so the
  elastic energy at fixed positions only falls. Closed cycles (load, unload, reload, reverse into
  tension past Tt, crush again, back to the start length): work put in 20,301.5 J, stored at the
  end 844.0 J, ledger 19,452.8 J, residual 4.8 J (0.02 %) (`closed_cycles_create_no_energy`).
  Reload stays on the plateau (no second peak).
- **The drop** is a softening (negative tangent −kH/(k − H)). Two conditions: no snap-back (H < k),
  and the tangent's magnitude no larger than k, so the explicit step's stability bound is not
  moved: H ≤ k/2, i.e. **d ≥ 2(P − Q)/k** (enforced at spawn; a drop steeper fails with the length it
  needs). Where does the drop's energy go: the elastic energy above Q is taken as plastic work over
  the drop (½(P² − Q²)/k of it at fixed length), never returned. Measured at crash rate (20 kg at
  15.6 m/s on an anchored 0.18 m beam, 0.5 ms steps): the step's invariant energy (½mv² + ½k·e·e₋,
  constant to 0.01 J in an elastic control run) **never rises in any step** (largest one-step rise
  0.000 J) at H = k/2 and k/8; no chatter (no sign change of the force while closing); the force
  never passes P (one step closes 7.8 mm = 39 kN at k 5 MN/m, so the peak is reached within two
  steps). The ledger's ∫C ds counts 2 % more than the steps take (the JBeam rule's own: 1.6 %), a
  sampling error of the ledger, not energy created.
- **Momentum**: the force is along the beam, equal and opposite.

### 3.4 (d) A lower least rest length

Per beam `rubbleMinRest` (default the solver's 0.1 m), with an engine-wide floor
`MIN_REST_FLOOR` = 0.04 m: a file below it fails to spawn with the reason. **Why 0.1 m was
there:** it is in the solver's first version (`785e5c91a`, 2026-11-01, "Rubble's fixed-step
node-beam solver"), listed in `explicit-beams.md` as a parameter without a derivation; its origin
is presumably the Rigs of Rods model the solver started from (*unverified*: we do not read its
source). Nothing in our history measured it.

**Inversion margin.** A chord locked at L_min must stop its nodes' closing before they meet:
(1) one step's closing v·h must be well under L_min; (2) the spring about L_min must absorb the
pair's relative energy within L_min: v·√(μ/k) < L_min (μ the pair's reduced mass). Closing speeds
of the suite: S12 11.1 m/s → 5.6 mm a step; S11 15.6 → 7.8 mm (the previous round's ≤ 16 mm with
node speeds up to twice the approach); S07 52.8 → 26.4 mm; 200 km/h 55.6 → 27.8 mm. So 0.04 m is
over one step's closing at every speed of the suite (1.4 steps at 200 km/h). Criterion (2) is not
met by a lone chord at 200 km/h (k 1.3–1.8 MN/m on the sedan, μ 1–2 kg: 41–69 mm); in the car each
station has four chords, eight diagonals and the face ties locking together. **Not verified at
190–200 km/h on the car** (§6.4 has S07 and S10 run 0).

### 3.5 (e) Bending (and shear) strength while a cell folds

**No per-beam law can do it (proof).** Take a cell of two chord pairs A and B at lever h, each
pair shortening at δ̇_A, δ̇_B. A member with folding force P_m and collapse moment M_c dissipates
P_m·(δ̇_A + δ̇_B)/2 + M_c·|δ̇_A − δ̇_B|/h. Uniform crush (δ̇_A = δ̇_B) needs pair A's force to be
P_m/2; a one-sided shortening (δ̇_B = 0: the "turn") needs P_m/2 + M_c/h from pair A. A
rate-independent law gives a beam a force from its own history only, and pair A's history is the
same in both motions up to that instant, so the two values must be equal: M_c = 0. So with independent chords the cell turns at any eccentricity: the
mechanism "near pair shortens, far pair stays" costs only the near pair's plateau, and an axial
load P at eccentricity e activates it at P = P_m/(1 + 2e/h) < P_m. (Diagonals cannot fix it: under
that motion they shorten or stay, like the chords.) Options without nodes:

1. **Chords for bending plus an axial fuse member**: any member in parallel adds to the crush
   force; in series needs a node. Rejected.
2. **Offset chords at y* = Z/A** (section-matching §2): matches σ_y·Z at full squash, but while
   folding every chord is at yield: no reserve. Rejected.
3. **Face members that carry moment but not crush**: shown above to see the turn as shortening.
   Rejected for bending; **adopted for shear** (3.2: X diagonals, tension at the wall's shear).
4. **A crush cell** (adopted): the four chords' rest-length changes split into orthonormal modes
   Φ (axial ½(1,1,1,1); bending about the strong and weak principal axes, from the chords' spawn
   positions; the remaining warping mode). Each mode m has its generalised force G_m = Φ_m·f and
   flows on its own: the axial mode by the chords' crush law summed (C_G(s_G) = ½N(s_G/2)) and their
   tension ½ΣTt; the bending modes at M/lever (lever Φ_m·y = h for a rectangle), perfectly plastic;
   warping at the smaller. Equal chord springs are required (then the modes are k-orthogonal and
   decouple exactly; the spawn fails otherwise). The cell then delivers:
   - axial: P (peak) → Q (plateau) → densified, the sum of the chords' laws;
   - bending about either axis: up to M_strong, M_weak, **independent of the axial state**
     (measured: crushed to the plateau, then rotated: the moment rises to 6.39 kN·m = M_c and holds
     while the axial force stays at the plateau within 0.3 %; independent chords at the same crush
     give only their couple at fixed shortening, 3.63 kN·m, and turn under load);
   - axial-and-bending interaction: none (an upper estimate; Kecman's hinge falls with rotation and
     a folding section keeps about (1 − (P_m/N_p)²) of its moment: not modelled);
   - shear and torsion: from the X diagonals (3.2), not the cell.
   The rule "fold if e < M_c/N, bend if more" follows: the near-pair mechanism costs P_m/2·δ̇ +
   M_c·δ̇/h, so the cell folds axially while the load's eccentricity is under M_c/N_pl. For the
   Camry's crash zone that is 16.9/120 = 0.14 m; for the sedan's box cell 11.5/81.8 = 0.14 m.

   Energy: stored ½kΣ(ℓ − r)² = ½kΣ_m(ε_m − π_m)² (orthonormal); each mode's flow has its force's
   sign (return mapping), so the plastic work Σ G_m Δπ_m ≥ 0; densification stops a mode's flow at
   the first chord at its floor (the mode is then elastic: no energy created, some held). Momentum:
   every force is along a chord. A chord that breaks dissolves the cell (the others carry on alone,
   each on the crush law at the cell's chord set). Ledger: `CellPlastic` takes each sample's Δr,
   splits it by Φ, and integrates each mode's capacity.

## 4. The prototype (CPU only) and the evidence that nothing changed by default

- **Opt-in data**: JBeam keys on NORMAL beams (Rubble extensions): `rubbleCrushPlateau` (N; turns
  the law on), `rubbleCrushDrop` (m), `rubbleMinRest` (m), `rubbleTensionYield` (N);
  `rubbleCrushCell` (name: four chords) with `rubbleCellMoment` [strong, weak] (N·m). An unmodified
  JBeam file has none: the parser leaves `JbeamBeam::crush` None, no extension record is made for
  them, and the beam runs the JBeam rule unchanged. Stored in the beam's extension record
  (`BeamExt::crush`, mask bits 16 and 32).
- **Default unchanged**, evidence: `crash_fleet_digests` passes (every scenario's fleet, word for
  word, all runs and contact settings; `BeamExt` prints exactly as `derive(Debug)` did when the law
  is unused, asserted by `a_record_without_the_law_prints_as_derive_printed_it`); the 8-run reports
  of S11 and S02 × discrete/edge/continuous (48 reports, `tools/beam-law/identity.sh`) are byte for
  byte those of the base commit; the off sedan's 66 generated files match
  `rubble-artifacts/sections/gen-before-33b57f525/SHA256SUMS`; the `off` and `lp` rows of
  `front-load-path.md` §4 reproduce exactly.
- **GPU refusal**: `explicit_crush::assert_cpu_only`, called from
  `engine_exec::explicit_plan::ExplicitUpload::with_sleep`, the packer every GPU executor (WebGPU,
  Metal, CUDA) uploads through: a fleet with a crush beam panics with "runs on the CPU reference
  only, no GPU kernel implements it" (tested).
- **Generator**: `Physics::front_rules.crush_law` (and `RUBBLE_FRONT_RULES=load_path,crush_law`),
  acting only with `load_path`: each bay of the crash box and rail is a crush cell, values from the
  section (`engine_vehiclegen::crush_law::member_crush`) times the character's multiple and rate as
  the load path already applies: peak at the effective-width squash load less the diagonals'
  share, plateau as the load path's chords, drop max(H, 2.05(P − Q)/k), densified at `deformLimit`
  0.73 with `rubbleMinRest` 0.04 m, tension at the squash share, collapse moments; each long face
  gets its second diagonal, both flat in compression at the load path's quarter share and in
  tension at the wall's plastic shear over the sine (τ_y·depth·t, τ_y = σ_y/√3, EN 1993-1-1
  §6.2.6's V_pl; thin webs buckle in shear first, so an upper estimate). The cell's four springs are
  set to the least of theirs (a cell needs one). Strengths are raised to 1.1 × the beam's own
  tensile yield where lower (a beam should not break below its yield).
  Sedan values (`beamng_feel`, FRONT_RULES_LOG): box chords 49.0 → 15.3 kN over 37.6 mm (H 22.8
  mm; the drop is set by k 1.84 MN/m), tension 82.6 kN, moments 11.50 / 7.21 kN·m; rail bays
  67.3 → 20.8 kN over 73.4 mm (H 24.2 mm; k 1.30 MN/m), tension 99.1, moments 14.15 / 9.06 kN·m.
  **The softened springs (2 kHz) spread the drop over 1.6–3 × the fold's half-length.**

## 5. Measurements, beam to car (CPU)

### 5.1 One beam (quasi-static, a quarter of the section; `one_beam_force_shortening.png`)

Sedan crash box (70 × 110 × 1.5 mm HSLA, 0.18 m) and rail bay (× 1.8 mm, 0.38 m), static values
from the calculator (peak = effective-width squash, plateau = Abramowicz–Jones, stroke 0.73 L):

| member, one chord | law | densifies at | plastic work | target ∫ to 73 % |
|---|---|---|---|---|
| crash box (peak 30.1, plateau 11.4 kN) | JBeam rule (hardens) | 0.085 m (47 %), then a spring | 1.20 kJ | 1.49 kJ |
| | JBeam rule, flat | 0.083 m (46 %) | 0.91 kJ | |
| | crush law | 0.134 m (75 %) | 1.71 kJ (the peak's 0.2 kJ above) | |
| rail bay (41.2 / 15.4 kN) | JBeam rule (hardens) | 0.292 m (77 %), 3.3 × at the end | 7.66 kJ | 4.27 kJ |
| | JBeam rule, flat | 0.281 m | 4.27 kJ | |
| | crush law | 0.281 m | 4.58 kJ | |

The box chord loses 0.05 m of stroke to `MIN_REST` under the JBeam rule and stores the rest of its
crush as spring energy (returned); the crush law reaches 75 %.

### 5.2 One cell under an off-centre load (`off_centre_cell.png`; ignored test `off_centre_cells`)

A 1000 kg ram carrying 0.6 of the cell's plateau work pushes the front plate through a pin at
height e above its centre; rear face anchored. Shortening of the near (top) and far (bottom) chord
pairs at the end, plate pitch; M/N the cell's reserve eccentricity (box 140.6 mm, rail 127.6 mm):

| cell | e / (M/N) | crush cell + X diagonals | independent crush chords + X | crush cell, one diagonal a face | JBeam flat |
|---|---|---|---|---|---|
| crash box | 0 | 43 / 44 mm, 0.00 rad | 13 / 74, 0.50 | 1 / 2, 0.18 | 22 / 7, 0.53 |
| | 0.3 | **33 / 34, 0.01** | 77 / 1, −0.59 | −1 / −1, −0.69 | 51 / 1, −0.64 |
| | 1.0 | 38 / 33, 0.01 | 96 / 0, −0.67 | | 45 / 0, −0.69 |
| | 2.0 | 35 / 26, −0.09 | 63 / 3, −0.49 | | 30 / 0, −0.45 |
| rail bay | 0 | 109 / 110, 0.05 | 114 / 101, −0.06 | 59 / 59, −0.07 | 69 / 68, 0.24 |
| | 0.3 | **92 / 88, −0.03** | 192 / 13, 0.00 | 66 / 66, 0.46 | 122 / 10, −0.05 |
| | 1.0 | 83 / 66, −0.15 | 158 / 31, −0.78 | | 120 / 0, −0.60 |
| | 2.0 | 84 / 29, −0.50 | 112 / 4, −0.77 | | 69 / −5, −0.65 |

**Yes: a member now folds along its length under an off-centre load within its reserve**, but only
with both parts: the crush cell (bending) **and** the faces' shear (X diagonals in tension). With
one diagonal a face at a quarter of the crush force (today's cell) the cell shears: the plate turns
with the chords at their length, even at e = 0 (the pin's own instability is enough): **this is the
"bays lose their span while their chords keep their length" of `front-load-path.md` §4.1.** The
generated sedan's box-bay diagonals hold about 6 kN of shear against the box walls' 65 kN (×1.8
`beamng_feel`: 116 kN). Past M/N the cell bends (global bending: physical).

### 5.3 The generated sedan (load path + crush law, `lpx`; run 0, edge contact)

Push (the audit's slow push), each bay's shortening along the car / one chord's length:

| push | box a1–a2 | rail a2–a4 | rail a4–a5 | force (kN) |
|---|---|---|---|---|
| lp 0.4 | 0.017 / 0.180 | 0.104 / 0.375 | 0.132 / 0.451 | 111 |
| lpx 0.4 | 0.032 / 0.172 | 0.067 / 0.375 | 0.137 / 0.446 | 137 |
| lp 0.6 | 0.112 / 0.179 | 0.347 / 0.377 | 0.082 / 0.455 | 267 |
| lpx 0.6 | 0.054 / 0.172 | 0.114 / 0.369 | 0.258 / 0.444 | 288 |

**In the car the rail still forms a Z** (bay a2–a4's front rises 0.18–0.24 m, a4–a5's falls
0.27–0.38 m, the kink at a4, the subframe's station), with less pitch of the first rail bay than
`lp` and the chords' mean length near rest: the cells now hinge in bending at their collapse moment
(14 kN·m at `beamng_feel`) rather than shear for free. *Hypothesis, not measured mode by mode*: the
moment at a4 (the subframe's vertical load and the push's offset from the rail) exceeds the small
generated rail's collapse moment (70 × 110 × 1.8 against the Camry's 83 × 145 × 2.06, M_p 16.2 kN·m
static); a per-mode split of `CellPlastic` over S11 would settle it.

S11 (56 km/h wall) and S12 (40 km/h) beside the Camry tests (`front_ab.py off lp lpx`;
`s11_s12_overlay.png`):

| quantity | before (off) | load path (lp) | load path + crush law (lpx) | test 8545 | test 7520 |
|---|---|---|---|---|---|
| S11 t_m (ms) | 44.4 | 73.3 | **76.5** | 70.9 | 75.9 |
| S11 dynamic crush (m) | 0.394 | 0.700 | **0.702** | 0.663 | 0.685 |
| S11 residual crush (m) | 0.199 | 0.539 | **0.559** | 0.554 | 0.579 |
| S11 rebound (m/s) / restitution | 5.34 / 0.342 | 3.84 / 0.246 | **2.91 / 0.186** | 1.50 / 0.095 | 0.89 / 0.057 |
| S11 peak wall force (kN) at (ms) | 661 at 20 | 491 at 44 | 493 at 48 | 921 at 29.5 | 829 at 30.1 |
| S11 4 ms / CFC 60 peak (g) | 50.0 / 59.1 | 42.8 / 55.0 | 41.2 / 50.6 | 36.8 / 39.8 | 40.7 / 45.0 |
| S11 force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 255 / 538 / 644 / – | 228 / 309 / 228 / 187 | 234 / 333 / 243 / 213 | 142 / 210 / 294 at 0.1/0.2/0.3 (8545) | |
| S11 elastic at the stop / 0.3 s later (kJ) | 47.3 / 9.7 | 25.8 / 4.4 | 29.0 / 14.8 | – | – |
| S11 released by zone (kJ) | cabin 16.6, front 18.9 | cabin 12.7, front 6.8 | cabin 8.5, front 4.5 | – | – |
| S11 cabin plastic share; door opening peak (mm) | 0.061; 151 | 0.027; 99 | 0.021; 96 | – | – |
| S11 engine rear to firewall, least (m) | 0.227 | 0.137 | 0.132 (not reached) | reached (≈ 55 ms) | |
| S12 t_m (ms) / crush (m) | 39.5 / 0.275 | 98.6 / 0.553 | 94.9 / 0.513 | 69.3 / 0.475 | 74.7 / 0.489 |
| S12 restitution / residual (m) | 0.484 / 0.099 | 0.186 / 0.437 | 0.193 / 0.332 | 0.149 / – | 0.107 / – |
| S12 peak wall force (kN) | 576 at 32 | 268 at 20 | 268 at 16 | – | – |

Mean deceleration to t_m (lpx) = 15.6 m/s / 76.5 ms = 20.8 g (Camry 21.0–22.7 g). The parameters
were set from the sections, not fitted.

## 6. Hard rules over 8 runs, gates

### 6.1 Hard rules (8 runs × discrete/edge/continuous, CPU, `crash_reference::verdict`)

`tools/sections/load_path_hard.sh lpx load_path crush_law`; `front_verdicts.py lpx`,
`front_hard_rules.py lpx`. The 4x4 (S02) is unchanged by the switch (no load path on it).

| variant | settings judged | new failures | known failures that now pass in every run |
|---|---|---|---|
| lp (front-load-path.md §5) | 15 | S08 edge: no energy created 1/8; S12 continuous: no energy created 1/8 | S08 continuous: the common centre of mass stays still |
| **lpx** | 15 | S08 edge: no energy created 1/8 (1.9 kJ, bar 340 J) | S08 continuous: centre of mass; **S11 discrete, edge, continuous: restitution within the 56 km/h sedan fleet's** |

| scenario | contact | energy created, max (J) | band | into the wall, max (m) | crossings, max | restitution | dyn. crush (m) | residual (m) |
|---|---|---|---|---|---|---|---|---|
| S11 | discrete / edge | 0.0 | ≤ 177 | 0.001 | – | 0.186–0.201 | 0.691–0.706 | 0.527–0.565 |
| S11 | continuous | 0.0 | ≤ 177 | 0.011 | – | 0.222–0.233 | 0.687–0.704 | 0.528–0.542 |
| S12 | discrete / edge | 0.0 | ≤ 89 | 0.000 | – | 0.148–0.204 | 0.495–0.533 | 0.321–0.401 |
| S12 | continuous | 0.0 | ≤ 89 | 0.006 | – | 0.152–0.246 | 0.468–0.502 | 0.278–0.357 |
| S01 | discrete / edge | 6.1 | ≤ 230 | 0.012 | – | 0.197–0.211 | 0.917–0.948 | 0.009–0.035 |
| S01 | continuous | 7.3 | ≤ 230 | 0.001 | – | 0.184–0.215 | 0.892–0.939 | 0.020–0.053 |
| S02 | discrete / edge | 0.3 | | 0.003 | – | 0.186–0.206 | 0.502–0.520 | 0.042–0.216 |
| S02 | continuous | 24.5 | | 0.002 | – | 0.187–0.208 | 0.493–0.515 | 0.101–0.231 |
| S08 | discrete | 242.3 | ≤ 340 | 0.000 | 101 | 0.115–0.149 | 1.100–1.114 | −0.039–−0.018 |
| S08 | edge | **1905.5 (1 run)** | ≤ 340 | 0.000 | 93 | 0.088–0.131 | 1.084–1.121 | −0.029–−0.007 |
| S08 | continuous | 9.8 | ≤ 340 | 0.000 | 0 | 0.071–0.168 | 1.025–1.078 | −0.031–0.001 |

S08 edge's one run is the edge contact's known position step that every deeper-folding front
reaches (lp: 1.45 kJ; lessons §25.8), not the beam law: the crush law's own steps never add energy
(§3.3) and its plastic work is never negative. S12 continuous's 107 J of `lp` is gone. The S08
discrete/edge crossings counts are of the same order as `lp`'s (99 / 40): the edge contact's,
reported, not judged here.

### 6.2 Gates

`tools/sections/front_gates.sh lpx load_path,crush_law au:every_design au:sedan_energy
au:gen_sedan_paint` (the gates that broke for `lp`), against `lp` (front-load-path.md §5):

| test | before (off) | lp | lpx |
|---|---|---|---|
| `every_design` (≤ 5 mm parked) | pass | fails: Mach 1 `subf_rr` 11.6 mm | fails: Mach 1 `subf_rr` 11.7 mm (the load path's subframe mounts, not the law: the Mach 1 has no crush cells) |
| `sedan_energy` `every_part_torn_off_and_spun_adds_no_energy` | pass | fails: hub cap FL at 5 m/s +21 J (bar 20.8 J) | the same 21 J |
| `gen_sedan_paint::accept_head_60` (50 mm) | pass | fails, 51 mm | **passes** |
| `gen_sedan_paint::accept_head_60_pressed` | pass, 21 mm | fails, 70 mm | fails: worst 48 mm (bar 50), but 28 mm from 0.3 s (bar 10 mm) |
| `gen_sedan_paint::own_paint_ends_apart` (listed) | fails | fails | fails (listed) |

**The hub cap's 21 J is not this law's accounting**: it is the same with and without the crush law,
and bisected by the load path's steps (`LOAD_PATH_STEPS`) it comes with `cells` alone (cells only:
21 J; `engine` alone or `mounts` alone: pass): the stations' corner nodes and the lattice's springs
fitted again to the node budget change the car the flung cap lands on, 0.2 J past its bar. A
front-rules issue to fix in the load path, not here.

**At 190 and 200 km/h** (S07, S10, run 0, edge): finite, energy gain 112.2 J and 0.0 J, but
identical for off, lp and lpx: those scenarios do not run the generated sedan's front, so they say
nothing about the inversion margin of §3.4 (unverified).

## 7. What the real change would take (no code)

- **Data on the GPU.** The law: per beam 16 B static (Q, d, L_min, Tt; P is the beam's own Tc) and
  4 B state (s), in the extension record (static E-records and the state record the kernel already
  indexes: `extIndex`, `extBase`, `extState`, spec §30): a beam with the law is an extension beam
  (`TYPE_EXT`). A cell: one record per cell on its first chord, about 112 B static (four chord
  indices 16 B, Φ 64 B in f32, three capacities and the axial curve 28 B, k 4 B) and 8 B state
  (s_G, alive), the other chords a 4 B index to it. At 10,000+ bodies (AGENTS.md's scale; the
  explicit vehicles' beams are counted in the hundreds of thousands): with every front member of a
  car on cells (6 cells, 48 law beams) a car adds about 6 × 120 B + 48 × 20 B ≈ 1.7 kB; 100 cars
  170 kB; a vehicle without the keys adds nothing.
- **Kernels that touch the beam law**: `slang/kernels/explicit_beams_jbeams.slang`
  (`jbeamUpdate`, `jbeamExtUpdate`: the yield step), `explicit_beams_beam_phase.slang` (the beam
  phase's dispatch to them), `explicit_beams_centred.slang` (centred damping's beam phase, which
  calls `jbeamUpdate`), `explicit_beams.slang` (the module that includes them); host
  `engine_exec::explicit_plan` (packing the records), the CUDA and Metal executors through the
  same packer.
- **Arithmetic**: the law adds, per law beam per step, one compare and on a flowing step one
  division and a few multiply-adds (the drop piece or the plateau): branch on "flowing", which the
  JBeam rule already has. A cell: per cell per step, 4 lengths (already computed per chord), 4 × 4
  mode projections (16 FMA), per mode a compare and on flow a division and 4 FMA, the floor clip
  (≤ 4 compares): about 60 flops, run by one lane per cell; its chords' lanes must skip
  (a scan-with-skip would serialise: give the cell's lane its chords as a list, the lesson
  `simt-scan-with-skip-runs-serial`). Not branch-free; the branches are per beam/cell state.
- **Bit-for-bit plan** (AGENTS.md's ladder): new feature paths in separate functions behind an early
  return (`metal-fast-math-codegen`); rung 1 fixtures with the switch (a new `--ab-switch`), CPU ==
  GPU every state word with bit-exact results on; rung 1.5 the scenario matrix (every row bit for
  bit and no row slower: the default path is untouched, a vehicle without the keys runs no new
  code); rung 2 Chrome's module validation and `compile_cost.py` (new code behind `#ifdef` or its
  own pass: `contact-modules-at-compile-limit`); rung 3 `perf-verify.sh`, engine-metal parity, the
  Chrome A/B; then remove `assert_cpu_only`'s refusal for the ported features only.
- **Existing vehicles**: none changes unless its files carry the keys (proved on the CPU here; on
  the GPU by the matrix).
- **JBeam compatibility**: an extension. Proposed keys: `rubbleCrushPlateau`, `rubbleCrushDrop`,
  `rubbleMinRest`, `rubbleTensionYield`, `rubbleCrushCell`, `rubbleCellMoment`. BeamNG ignores
  unknown keys (JBeam options are a key–value table): an unmodified file behaves as before here,
  and a Rubble file still loads in BeamNG with the JBeam rule (a car made for the crush law would
  be too weak there: its beamDeform is the peak but the plateau is lost).
- **Risks**: the crush cell's equal-spring requirement (the generator equalises to the least);
  the softened springs spread the drop over 37–73 mm; the cell's bending is perfectly plastic (no
  Kecman decay, no N–M interaction: an upper estimate of the moment after the hinge forms); the
  peak, the fold length and the wall shear are estimates; a cell's face distortion (the stations'
  ties) is not part of the cell; the inversion margin at 190–200 km/h is unverified.

## 8. Recommendation

Confident:
- The crush law (§3.3) is sound: dissipative by construction, its closed cycles create nothing,
  its drop is stable at 0.5 ms with d ≥ 2(P − Q)/k, its ledger matches ∫C ds; and the per-beam
  `rubbleMinRest` gives the crash box its stroke (75 % against 47 %).
- A lattice cell needs **two** things the JBeam beam cannot give to fold instead of turning:
  bending strength independent of the axial crush (the crush cell; proved impossible per beam) and
  shear strength independent of it (tension-strong, compression-weak face diagonals: Tc ≠ Tt,
  which the beam already has). The off-centre ram shows both are needed and together sufficient
  inside the reserve eccentricity M/N.
- Nothing changes by default (digests, 48 reports, the sedan's files).

Not confident:
- **The car does not yet fold bay by bay**: the rail now hinges in bending at the subframe station
  instead of shearing; the generated rail's section is small and its subframe load enters at a4.
  The next step is the generator, not the solver: the rail's sections from the Camry's (§1.3 of
  `front-load-path.md`), the rear rail to the firewall as cells, and a per-mode split of the cells'
  work over S11.
- S11 restitution improved (0.246 → 0.186) but is still twice the Camry's; the wall force's peak is
  half the Camry's and late (the engine's stack, `front-load-path.md` §6); S12 is still too soft
  (95 ms against 69–75).
- The estimates (peak as the effective-width squash load, the fold length, Kecman's moment as σ_y·Z
  with an effective flange, the wall shear as V_pl) need their sources checked.

**Recommendation:** keep the switch off. Adopt the crush law and the per-beam least rest length as
the solver feature to port (they are needed by any member that folds and they are cheap), with the
crush cell as an opt-in extension; port only after the generator shows the front folding with
them (the remaining defects are content: section sizes and the rail's rear joint).
