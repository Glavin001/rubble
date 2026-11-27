# Section matching: what a node-beam cell must be to stand in for a member

How a lattice segment of chords and diagonals reproduces a real member's stiffness and
strength, what our JBeam beam law does with a yield force over travel, and where the two
cannot agree. Written 2026-11-27 for the crash reference work (`crash-reference.md` §4: the
generated sedan beside the real Camry tests). The audit that applies it to the sedan is
[sedan-front-audit.md](sedan-front-audit.md).

Code:
- the section calculator `engine_vehiclegen::thin_wall` (`thin_wall/mod.rs`, `crush.rs`);
- the beam-law tests `engine-softbody-lab/tests/section_beam_law.rs` (fast tests assert
  every statement in section 3 below; the `#[ignore]`d one drives the sedan's own chords);
- the oracle `engine/tools/sections/oracle.py` (local only).

## 1. The section calculator

A cross-section is a set of walls, each a midline polyline (open or closed) with a
thickness: a box, a C-channel, a hat on its closing plate, two stampings welded flange to
flange (one wall each, stacked where they overlap). Each wall becomes quadrilateral pieces
that meet on their **mitre lines**, so a corner's steel is counted once and a closed
rectangular midline (b − t) × (h − t) is exactly the b × h tube of wall t. Every quantity is
then exact for those polygons: area, centroid, second moments and product by Green's
theorem; the plastic neutral axis by bisection on the exact clipped area; the plastic
modulus Z = ∫|s − s_p| dA from the two halves' first moments. The only approximation is the
polygon itself (a round tube is its chords; error (π/n)²/6 in area).

With a material: squash load σ_y·A, first-yield and fully plastic moments σ_y·W and σ_y·Z,
axial stiffness EA/L, bending rigidity EI, mass ρA.

**Tests** (`thin_wall/tests.rs`, all pass): a rectangular tube against its closed forms (A,
I_x, I_y, W, Z_x, Z_y to 1e-12 / 1e-9); a circular tube (to 5e-8, the chords); translation,
rotation and mirror invariance of A, I₁, I₂ and Z about turned axes; the truck's C 266 × 90 × 8
(A 3,440 mm², I 3.4387e7 mm⁴, Z 3.1076e5 mm³, M_p 155.4 kN·m at 500 MPa, shape factor 1.20);
the Camry rail's crush numbers (below); `Member::crush_n` equal bit for bit to the calculator
(it now calls it).

**Oracle** (`sectionproperties` 3.x, a 2D finite-element section analysis, in a throwaway
virtualenv under `engine/out/sections/venv`; run `oracle.py`): on ten sections (rectangular and
round tubes, the C, a hat open and on its plate, a two-part closed rail, a lipped channel with
a sloped web, the Camry rail, the sedan's crash box and rail) our A, I_x, I_y, W_x, W_y, Z_x, Z_y
agree with it to **1e-14 relative** on straight-walled sections and to **1e-8** on the round
tube and the sloped channel (the oracle's 1e-7 mm snapping grid). Both compute the same
region, so the agreement checks our integrals and plastic axis, not the mitre model. The
classic thin-wall midline sums (line integrals, no corner or t³ terms) differ from the exact
region by 1e-4 to 6e-3 on I and up to **4 %** on the C's weak-axis Z (its flanges' corners
counted on the midline): the O(t/b) terms the exact region avoids.

### Crush and collapse formulas (`thin_wall::crush`)

| Formula | Source | Confidence |
|---|---|---|
| P_m = 13.06 σ₀ t^(5/3) C^(1/3), C the mean side | Abramowicz & Jones 1984 (Int. J. Impact Eng. 2) and 1986 (4): P_m/M₀ = 52.22 (C/t)^(1/3), M₀ = σ₀t²/4 | **Confident**: it is 38.12/0.73, the superfolding element divided by the effective crush distance, an identity that checks both numbers |
| P_m = 9.56 σ₀ b^(1/3) t^(5/3) | Wierzbicki & Abramowicz 1983 (J. Appl. Mech. 50) | The paper's 38.12/4 is 9.53; 9.56 is the commonly quoted value. **To be verified** (0.3 %) |
| Effective crush distance δ_e/2H = 0.73 | Abramowicz 1983 (Int. J. Impact Eng. 1) | Confident for the fold; its use as a member's usable stroke (0.70–0.75 of the length before the folds stack) is common practice (e.g. Jones, *Structural Impact*), **to be verified** |
| Rectangle as a square of mean side (b + h)/2 | common practice | Not part of the derivation; fine near aspect 1–2 |
| ε̇ ≈ 0.33 V/C in the folds | Abramowicz & Jones 1984 | Form recalled, **coefficient to be verified** |
| σ_d/σ₀ = 1 + (ε̇/D)^(1/q) | Cowper & Symonds 1957 | Confident. D = 40.4 s⁻¹, q = 5 (mild steel, confident) overstate high-strength steel; the CCSA Camry deck's own C = 8,000 s⁻¹, p = 8 is what we use for automotive grades |
| Winter effective width ρ = (1 − 0.22/λ)/λ | AISI S100, EN 1993-1-5 §4.4 | Confident |
| Box bending collapse: σ_y·Z with the compression flange at its effective width | **Not** Kecman 1983's formula (not at hand): an upper estimate in its spirit | **Estimate, to be checked against Kecman** |

No top-hat formula (White, Jones & Abramowicz 1999) is implemented: we could not state its
coefficients with confidence. A hat on its closing plate is computed as a box of the same
mean side, as the generator does, and marked so.

**Regression of the implementation** (not of the truth): the Camry rail at its crash box, 92 ×
166 mm, 1.69 mm, σ_y 360, σ₀ 499 MPa: P_m 57 kN at σ_y, **79 kN at σ₀**, × 1.515 (deck rate
scaling at 15.6 m/s, ε̇ = 40 s⁻¹) = **120 kN**; the mild-steel pair gives × 2.0. Two of them, 240
kN, is the Camry tests' first plateau (observation, `fe-import-spike.md` §4).

## 2. A lattice cell against a member

A bay of length L with N chords along the member at section points (x_i, y_i), spring k_i,
yield F_i, plus diagonals, between end faces held in shape (section ties: in our cars they are
shorter than `MIN_REST` and so never yield, or short and stiff).

| Member property | Real section | Lattice cell (chords only) |
|---|---|---|
| Axial stiffness | EA/L | Σ k_i (+ Σ k_d cos²θ_d for diagonals) |
| Bending rigidity about x | EI_x | L·Σ k_i (y_i − ȳ_k)² |
| Squash (all yielded, axial) | σ_y A | Σ F_i (+ Σ F_d cos θ_d) |
| Mean folding force | P_m (§1) ≈ 0.25–0.45 σ_y A for car members | Σ F_i (the law cannot fold: §3) |
| Fully plastic moment | σ_y Z | Σ F_i |y_i − y_p| (y_p: the equal-force axis) |

**What can be satisfied together.** EA and EI together need the chords' stiffness to have the
section's radius of gyration: Σk y²/Σk = I/A. Squash and M_p together need Σ F|y − y_p| / Σ F =
Z/A. With four chords at the corners of an h-deep section both ratios are fixed at h²/4 and h/2,
the values of a section whose steel is all at its extreme fibres. A thin-walled section's
steel is not: its webs carry steel at every height. So with corner chords:

- match the squash (Σ F = σ_y A) and the plastic moment is too high by **A·h/(2Z)**: a b × h box
  (thin) gives (b + h)/(b + h/2), 1.33 square, 1.46 for the sedan's 70 × 110 crash box (A 531 mm²,
  Z 19,979 mm³); the truck's C gives 1.47 (229 kN·m against 155, `generated-vehicles-lessons.md`
  §22);
- match M_p and the squash is too low by the inverse (the truck: 1,168 kN against 1,720);
- match EA and EI is too high by A h²/(4I): 1.74 on the sedan's crash box.

Options, each a generator rule:
1. **More chords**, spread over the walls by steel area (mid-web chords on the deep sides): a
   chord per wall segment, at the segment's centroid, with σ_y × its area, reproduces A, I and Z
   to O(1/n²). Costs nodes (mass per node falls, so the node budget's springs fall too:
   `generated-vehicles-lessons.md` §2-3).
2. **Chords inset to the section's own lever arm**: four chords at ±y* with y* = Z/A carry both
   σ_y A and σ_y Z exactly (EI is then A·y*², which is not I unless y*² = I/A; the 70 × 110 box
   gets y* = 0.68 h/2 and EI 0.81 of the section's). The chords' nodes are physics only (inside the skin), so this
   does not touch the look; but the collision skin must not ride on them.
3. **Choose by role**, which is what the crash needs: a crash box or a rail ahead of the
   firewall crushes axially, so its cell's axial yield is matched to its **mean folding force at
   crash rate** (not its squash), over the **stroke** it really has (0.73 of its length), and its
   bending is allowed to be 1.3-1.5 × strong; a sill, a B-pillar, a bumper beam or a truck's frame
   rail is loaded in bending, so its chords are matched to M_p (or the collapse moment, §1) and
   its squash comes out low, which does not matter for that role.

Diagonals add to every row of the table (the four-chord test below: +842 kN axially, +107 kN·m
in bending on the truck cell), so a cell's chord forces have to be set with its diagonals'
share counted, or the diagonals' yields kept below the chords' (as the generator's 0.7 ×
skip-beam rule does for lengthwise skips, but not for bracing).

## 3. What the solver's beam law does with a yield force over travel

The JBeam NORMAL beam in `explicit_beams.rs` `jbeam_force_filtered` (no damping here), as the
JBeam spawn sets it (`push_jbeam_beam`: `plastic` 0, thresholds tc = tt = `beamDeform`, hardening
cap `limit_k` = `deformLimitStress`, default ∞; compression floor `short_bound` = (1 −
`deformLimit`)·L₀, none when unset; `MIN_REST` = 0.1 m on every beam). Every statement below is
asserted by `section_beam_law.rs` on the solver's own function.

**Monotone compression of one beam** (`one_beam_yields_hardens_and_locks`: L₀ 0.5 m, k 5e6 N/m,
tc₀ 50 kN, deformLimit 0.7):
1. Elastic, F = k·s, until F = tc₀ at s = tc₀/k (10 mm here; for the sedan's crash-box chord
   10.1 mm, 5.6 % strain: steel yields at 0.17 %).
2. Then the rest length moves so the elastic force is exactly the threshold, and the threshold
   hardens as **tc = tc₀ · L₀ / L_rest** (`beam.tc * old / beam.rest` each yield). Under steady
   shortening the force is **F(s) = tc₀ L₀ / (L₀ − s + F/k)**: it rises the whole way, 1.25 tc₀ at
   20 % shortening, 2 tc₀ at 50 %, 3.33 tc₀ at 70 %. Measured to 0.2 %.
3. It stops yielding at **L_rest = max((1 − deformLimit) L₀, MIN_REST)**: 0.15 m here (3.33 tc₀,
   locked at s = 0.383 m = 77 % of L₀ with the elastic part). Past that it is a plain spring of
   stiffness k about the locked rest length (250 kN at s = 0.40 m).
4. A beam of rest under 0.333 m reaches **MIN_REST before deformLimit 0.7**:
   (`a_short_beam_locks_at_min_rest`) a 0.18 m beam (the sedan's crash-box chords) crushes 44 % of
   its length and locks at **1.8 tc₀**; one under 0.1 m never yields in compression.
5. **Energy.** Plastic work to a rest L is tc₀ L₀ ln(L₀/L) (measured to 1 %): to 0.3 L₀, 1.20 tc₀
   L₀, against 0.70 tc₀ L₀ for a flat plateau over the same travel, so the **mean force over the
   stroke is 1.72 tc₀**. Per metre of crush: the sedan's crash-box chord 54 kJ/m (tc₀ 40.9 kN),
   its rail chords 139-151 kJ/m (tc₀ 111 and 92 kN). Elastic energy at the threshold is ½tc²/k and
   is all given back on unloading (measured: ½F²/k to 2 %); at the lock it is ½(3.33 tc₀)²/k, 11 ×
   the energy at first yield.
6. **Tension** (`tension_hardens_and_eats_the_strength`): tt hardens as tt₀ L/L₀ and every
   increment of set takes k × the set off the strength, so a beam breaks after a set of about
   (S − tt)/k: 2 cm for a 50 kN / 150 kN beam of k 5e6. Tensile ductility is nearly nil
   (hinges, bolts and a rail's tension side tear after centimetres).

**A real member's curve** is a peak near σ_y A (or its buckling load) at a few millimetres, then
a lower, wavy plateau at P_m (0.25-0.45 of the peak for car members) over about 73 % of its
length, then densification. In our law:
- **The peak then lower plateau cannot be represented.** Under monotone compression the force
  never falls: tc only grows. The only load drop the solver has is a break (a beam removed). A
  member set to P_m loses the initial peak (a few ms, small energy); one set to the peak never
  comes down.
- **The flat plateau is not the default**: hardening as 1/L makes the force climb 1.8 × (a 0.18 m
  chord) to 3.3 × (deformLimit 0.7) over the stroke. JBeam's own `deformLimitStress` caps it:
  `limit_k` = tc₀ makes tc flat (the solver's `min(limit_k.max(tc))`; the metrics kit's
  `Plastic::advance` integrates the capped law). The gen2 rigger writes no `deformLimitStress` on
  shell beams today (∞).
- **The stroke** ends at MIN_REST or deformLimit, whichever is longer as a rest: members shorter
  than 0.37 m cannot reach 73 % of their length; the crash box's 0.18 m bay reaches 44 %.
- **Densification** is the lock: past it, a stiff spring (k), which is what a stacked fold is,
  but its energy is elastic and returns.
- **Elastic storage is the softening's price.** At a fixed 2 kHz step every beam is softened
  steel (`generated-vehicles-lessons.md` §2), so a beam at its threshold holds ½tc²/k, two to
  three orders of magnitude more than the real member at the same force, and gives it all back.
  The sedan's crash-box chord holds 206 J at first yield, its rail chord 1.5 kJ, a subframe mount
  beam (157 kN, k 1.5e6) 8.2 kJ.

**One four-chord cell** (`a_four_chord_cell_axially_and_in_bending`, the truck rail's lattice:
chords 430 kN at 0.266 × 0.09 m, one diagonal a face at 215 kN, 0.95 m bay, ties never yield):
axially the chords yield at 1,720 kN and the diagonals add 842 kN as they yield (2,586 kN at
30 mm, then hardening: 3,115 kN at 0.2 m); in bending about the strong axis the chords alone
give 2 F h = 229 kN·m and the cell reaches **336 kN·m** with its diagonals, 2.2 × the C section's
M_p of 155 kN·m.

## 4. Rules that follow (proposals, for the generator)

1. A crush member's cell: Σ (chord yields + diagonal axial shares) = its **mean crush force at
   crash rate** (§1), and nothing else in its bay carries axial load it would not carry in the
   car (braces across the bay to other members are the usual culprit).
2. Its chords long enough to crush 73 % (≥ 0.37 m with MIN_REST 0.1 m), or the stroke shortfall
   stated; `deformLimit` 0.73 and `deformLimitStress` = its yield, so the plateau is flat.
3. Stops and supports in a crush zone (SUPPORT beams, deform ∞) are springs: give them a yield
   below the member's, or they store and return energy.
4. Bending members by M_p (or collapse moment), axial members by P_m; state which in the
   member's definition.
