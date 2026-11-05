# JBeam semantics in the explicit solver ("JBeam mode")

Specification of how Rubble's explicit node–beam solver (`engine_softbody::explicit_beams`,
CPU reference; `engine_webgpu::ExplicitBeamsGpu`, `slang/kernels/explicit_beams.slang`)
simulates a vehicle loaded from BeamNG JBeam files. It complements
`engine/docs/explicit-beams.md`, which specifies the solver and its Rigs of Rods (RoR)
compatible rules; everything not stated here is as there.

**Clean room.** Written from BeamNG's public documentation only
(documentation.beamng.com, and the old wiki.beamng.com pages, which now redirect to
it). No BeamNG code (game Lua, BeamNGpy, the Blender JBeam Editor's physics) and no
Rigs of Rods source was read. The only BeamNG files read are JBeam data: what the
test vehicles (the BX coupé body, the legocar) actually use. Where the documentation
is silent or ambiguous the choice made is stated and marked **Choice**.

Sources (all retrieved 2026-11-03):

* [B] Beams: https://documentation.beamng.com/modding/vehicle/sections/beams/
* [S] Support beams: https://documentation.beamng.com/modding/vehicle/sections/beams/support/
* [BD] Bounded beams: https://documentation.beamng.com/modding/vehicle/sections/beams/bounded/
* [A] Anisotropic beams: https://documentation.beamng.com/modding/vehicle/sections/beams/anisotropic/
* [L] L-beams: https://documentation.beamng.com/modding/vehicle/sections/beams/l-beam/
* [N] Nodes: https://documentation.beamng.com/modding/vehicle/sections/nodes/
* [T] Triangles: https://documentation.beamng.com/modding/vehicle/sections/triangles/
* [C] Couplers: https://documentation.beamng.com/modding/vehicle/sections/nodes/couplers/
* [TB] Torsion bars: https://documentation.beamng.com/modding/vehicle/sections/torsionbars/
* [TH] Torsion hydros: https://documentation.beamng.com/modding/vehicle/sections/torsionhydros/
* [I] Introduction to JBeam: https://documentation.beamng.com/modding/vehicle/intro_jbeam/
* [CI] Common JBeam issues: https://documentation.beamng.com/modding/vehicle/intro_jbeam/common_issues/
* [GM] Ground models: https://documentation.beamng.com/modding/levels/level_formats/groundmodels/
  (retrieved 2026-11-05).
* [PN] BeamNG.drive alpha 0.11 patch notes ("The coast is clear"):
  https://beamng.com/game/news/patch/the-coast-is-clear/ (retrieved 2026-11-05).
* [W] the former wiki page "Beams" (https://wiki.beamng.com/Beams, now a redirect to
  documentation.beamng.com; its text as quoted by search results: deformLimit "limits
  by how much of original length the beam can deform in compression",
  deformLimitExpansion "in expansion", "the most frequently used").

## 1. Why a JBeam mode

Rubble's JBeam importer (`jbeam::load_jbeam`) builds a `SoftVehicleDesc`. Before this
work the explicit solver only ran it through `.truck` text
(`truck_export::desc_to_truck`, then `ExplicitBeams::from_truck`), which turns JBeam
into RoR semantics and loses JBeam's: passive break groups became active detacher
groups, bounded beams became RoR shocks (no bound zone, a 12000 N·s/m limit damping
where the file says 0), precompression was dropped on every beam but bounded ones,
supports took RoR's break rules, couplers and torsion bars were dropped, every node
self-collided, and beam lengths went through RoR's fast inverse square root. On the BX
coupé body that broke the doors off at rest within 8 ms: a door detent, a bounded beam
with `beamLimitDamp: 0` precompressed to 1.9×, got RoR's 12000 N·s/m limit damping,
passed its 8000 N strength, and (written as an active detacher group) took both hinge
groups with it.

**Choice: a native desc → solver path.** `ExplicitBeams::from_desc_jbeam(desc, tyres,
…)` (and `from_jbeam_native(files, pc, …)`) spawns the desc's nodes and beams directly,
with JBeam rules carried by new beam kinds (`JbeamPlain`, `JbeamSupport`,
`JbeamBounded`, `JbeamCoupler`, `JbeamTorsion` + `JbeamTorsionPair`). The importer keeps
the raw JBeam values a beam needs in `SoftBeam::jbeam` (`desc::JbeamBeam`), coupler
pairs in `SoftVehicleDesc::couplers`, torsion bars and torsion hydros in `elements`.
The `.truck` path (`from_truck`, `from_jbeam`) is unchanged: RoR content and the RoR
rules are bit for bit what they were (§13). Carrying the semantics through `.truck`
text was rejected: `.truck` has no syntax for most of them, and every extension would
have been a Rubble-only dialect of RoR's format.

The one part still taken from the `.truck` path is wheels: BeamNG's pressure wheels
are spawned as the `.truck` path's 12-ray soft tyres (`TyreDefaults::for_vehicle`),
after the desc's nodes and beams (§12).

## 2. Common rules

Units: metres, kilograms, seconds, newtons; JBeam's frame (+X left, +Y rear, +Z up)
becomes Rubble's (+X left, +Y up, +Z forward) as before. The step is the solver's fixed
0.5 ms (BeamNG's documentation gives no physics rate; 2000 Hz is the solver's).

* **Lengths are exact** (ℓ = √(d·d)). The fast inverse square root is a RoR
  compatibility rule; its ≈0.2 % length error strains a beam of 2 MN/m by kilonewtons
  at rest, which shook the legocar's front axle apart. `from_desc_jbeam` turns
  `fast_inv_sqrt` off for the vehicle whatever the caller passes.
* **Beam force** (as in the solver): with e = ℓ − L and u the extension rate,
  F = −k·e − D(u), D the damping force (c·u unless a rule below changes it); F > 0
  pushes the ends apart. "All standard beam arguments apply to every specialized type"
  [B].
* **Precompression** [B]: the rest length at spawn is L = p·ℓ₀ for every beam type
  (`beamPrecompression` p, default 1; `precompressionRange` r overrides it:
  L = ℓ₀ + r). ℓ₀ is the spawn distance of the two nodes, L₀ = L the spawn rest
  length. `beamPrecompressionTime` (a ramp) is not simulated: the length is set at
  spawn (a warning when a file uses it; neither test vehicle does).
* **Breaking**: when |F| passes `beamStrength` S the beam breaks [B], [I]. The yield
  and break tests start at G = min(`beamDeform`, S) (the RoR rule starts them at the
  deform threshold only, so a beam with an infinite deform threshold never broke).
  **No break exception**: RoR keeps a beam whose end is on a collision triangle with
  fewer than three live plain beams; BeamNG documents nothing like it.

## 3. NORMAL beams (kind `JbeamPlain`)

Force as §2. **Deformation** [B], [I]: "Once deformed, the beam will no longer return
back to its original shape"; the documentation gives no law. **Choice:** the solver's
own yield rule (`explicit-beams.md` §2, step 4, with plastic coefficient 0): past the
deform threshold the rest length moves so the force returns halfway to the threshold,
and the threshold hardens by the length ratio (Tc ← Tc·L_old/L in compression,
Tt ← Tt·L/L_old in tension); a stretched beam's strength drops by k × its plastic
elongation. `deformLimitStress` [B] "caps the growth of beamDeform as the beam
deforms": the hardened thresholds never pass max(cap, their current value).

**Deform limits** [B], [W]. `deformLimit` d limits "how far the beam can deform in
compression, as a fraction of original length"; `deformLimitExpansion` dₑ the same in
expansion (default: `deformLimit`). **Choice (ambiguous):** the limit is on the
plastic change, as a fraction of the spawn rest length: L ≥ (1 − d)·L₀ and
L ≤ (1 + dₑ)·L₀. The other reading, a limit on L/L₀ itself (L ≤ dₑ·L₀), cannot be what
the BX means by `deformLimitExpansion: 0.01` on its latch beams (a rest length capped
at 1 % of the original), and it would make the documented default (dₑ = d, d < 1 in
compression) forbid any stretching. At a limit the rest length stops there and the
beam is elastic about it: F = −k·(ℓ − L_limit) − D(u) (no halfway rule). Values below
zero or non-numeric mean no limit.

**HYDRO** beams (the `hydros` section and `|HYDRO` rows, now loaded) have no input in
this solver (no steering or controller): their rest length stays at L₀. They are
`JbeamPlain` beams with their own spring, damping, deform and strength.

## 4. SUPPORT beams (kind `JbeamSupport`) [S], [B]

"Support beams are beams that only resist to forces in compression"; with
precompression they "only start resisting forces when being compressed smaller than
their precompressed length". `beamLongBound` b: "When this bound is exceeded, the beam
automatically breaks"; b = 0.5 breaks at 150 % and b = 1 at 200 % of the spawn length.

* e ≤ 0: a NORMAL beam (force, deformation in compression, break past S).
* e > 0: **no force** (spring and damping both zero; **Choice**: RoR keeps a tenth of
  the damping, BeamNG's "only resist in compression" reads as none).
* ℓ > (1 + b)·ℓ₀ breaks it. **Choice (ambiguous):** ℓ₀ is the geometric spawn
  length, not the precompressed rest length (a precompressed support's rest is shorter
  than its nodes' distance; measuring from it, b = 0 would break a support precompressed
  below 1 at spawn). Default b = 1 [S]. A length break is a break: it sets off the
  beam's break group like any other. `beamLongExtent` (an absolute length) is not
  simulated (warning; unused by the test vehicles).

## 5. BOUNDED beams (kind `JbeamBounded`) [BD]

"The properties change progressively from the initial beam properties to the bound
properties when reaching the limits"; springs and limits are position terms, damping a
velocity term.

* Bounds: s = `beamShortBound`, l = `beamLongBound` as fractions of the rest length L
  (`shortBoundRange` / `longBoundRange`, in metres, override them: divided by L).
  **Choice:** L is the precompressed rest length (the docs say "ratio of spawn
  length"; the BX door detents, precompressed 1.9× with bounds of ±0.1, only work as
  detents about the precompressed length). Defaults 1.
* Overshoot o = e − l·L if e > l·L, −e − s·L if e < −s·L, else 0. Blend
  σ = min(o / z, 1) with z = `boundZone` (m, default 1: "deformation after reaching the
  bound before full limit properties apply"); σ = 1 when z ≤ 0.
* k = k₀ + (k_lim − k₀)·σ and D = D₀(u) + (c_lim·u − D₀(u))·σ, with k_lim =
  `beamLimitSpring` and c_lim = `beamLimitDamp`, both default 1.0 as documented.
* Slow and fast damping: D₀(u) = c·u while |u| ≤ v_s (`beamDampVelocitySplit`, default
  ∞), else sign(u)·(c·v_s + c_f·(|u| − v_s)) with c_f = `beamDampFast` (default c).
  **Choice:** continuous at the split ("the force will always keep going up as the speed
  increases"). The rebound values (`beamDampRebound`, `beamDampReboundFast`,
  `beamDampVelocitySplitRebound`, `beamLimitDampRebound`) are not simulated (warning;
  unused by the test vehicles).
* Bounded beams never deform (**Choice**; the docs do not say; RoR's shocks do not
  either). They break past S.

With the BX door detent (k = 800, bounds ±0.1, zone 0.01 m, limit spring and damping
0, precompression 1.9) this gives zero force at rest, where the `.truck` path gave
36 kN.

## 6. Damping cut-off (`dampCutoffHz`) [B]

What BeamNG documents (retrieved 2026-11-05; no developer forum post gives the law):

* [B]: "Limits the vibration frequency (Hz) above which damping applies. Only applies
  to normal, bounded, and l-beams"; it is used "mostly with suspension components and
  other critical parts of the jbeam to help run higher damping before instability";
  "Expensive to run, should only be used where needed".
* [PN] the patch notes of alpha 0.11, which added it (to normal and bounded beams):
  "This option simulates band limited damping, increasing the amount of damping that
  can be stably used".

So the cut-off is a low-pass on the damping, and its one stated purpose is
**stability: more damping than the plain law can run**. The BX and the legocar put
it on damper beams and engine mounts (250–1000 Hz) and the legocar's steering dampers
(1000 Hz); the Astral R on its stiff rear control arms (500 Hz); pressure wheels on
their tread and periphery beams (500 Hz).

**Rule:** the damping uses y, the extension rate u through the discrete RC low-pass
at the cut-off f_c: a = 1 − exp(−2π·f_c·h), y_n = a·u_n + (1 − a)·y_{n−1} (the CPU and
GPU record (u, y, a, b) with b = a − 1, y = a·u − b·y₋₁). The state starts at zero.

**Why this law (analysis, 2026-11-05).** One beam mode of angular frequency ω, mass
m, damping c, on the solver's step (symplectic Euler: v_{n+1} = v_n + h·(−ω²x_n −
(c/m)·y_n), x_{n+1} = x_n + h·v_{n+1}, y = H(z)·v). Write Ω = ω·h and γ = c·h/m.
Undamped, the mode turns by θ per step with 2 − 2 cos θ = Ω² (θ = 2 arcsin(Ω/2); the
step's limit Ω = 2 is θ = π, the Nyquist rate). The damping reads v_n = (x_n −
x_{n−1})/h, a backward difference: in the mode it lags the centred rate by θ/2. A
damping force removes energy only while its total lag stays under 90°:
θ/2 + φ_H(θ) < π/2. So any filter's lag adds to one the step already has, and the
margin at a fast mode is small (at Ω = 1.77, θ/2 = 62°).

* The §6 law of 2026-11-03 was the bilinear-transform low-pass,
  H(z) = a·(1 + z⁻¹)/(1 + b·z⁻¹). Its zero at the Nyquist rate is a factor
  (1 + z⁻¹), which lags θ/2 by itself; at f_c = 500 Hz on the 2 kHz step b = 0 and
  H is exactly (1 + z⁻¹)/2. The total lag is then θ, past 90° for θ > π/2: every mode
  with Ω > √2 is **anti-damped**, however small c is. Below 500 Hz the denominator
  lags more (anti-damped from Ω = 1.08 at 250 Hz), above it less (1.86 at 900 Hz).
* The RC low-pass lags little near the Nyquist rate (−arg(1 − (1 − a)e^{−iθ}), 9° at
  Ω = 1.77 and 500 Hz, falling to 0 at θ = π) and never adds anti-damping.

Eigenvalues of the one-mode map (spectral radius ≤ 1 for every Ω up to a bound;
the largest γ = c·h/m that stays stable):

| modes up to | cut-off | plain damping | RC low-pass | bilinear (before) |
|---|---|---|---|---|
| Ω ≤ 0.5 | 250 / 500 / 1000 Hz | 1.87 | 5.01 / 2.85 / 2.04 | 1.57 / 1.75 / 1.85 |
| Ω ≤ 1.0 | 250 / 500 / 1000 Hz | 1.50 | 4.01 / 2.28 / 1.63 | 0.29 / 1.00 / 1.42 |
| Ω ≤ 1.5 | 250 / 500 / 1000 Hz | 0.87 | 2.34 / 1.33 / 0.95 | 0 / 0 / 0.69 |

The RC law does what [PN] says the option is for, more stable damping than the
plain law at every cut-off; the bilinear filter allowed less. (Script: the session
scratchpad's `law.py`, a 4×4 linear map per (Ω, γ); not in the repository.) The
bilinear filter's one advantage, no damping at all at the Nyquist rate, is covered
since 2026-11-05 by §28's backward-Euler rule for dampers with c·h/μ > 1.

**Effect on the vehicles** (`examples/jbeam_accept.rs`, CPU, 4 s at rest then a rigid
wall at 56 km/h for 1.5 s; bilinear → RC):

| vehicle | rest: KE at 4 s, max node speed | rest: centre of mass 1 → 4 s | wall: broken / bent at 1.5 s |
|---|---|---|---|
| BX body | 2.8 → 2.2 J, 3.1 → 1.8 m/s | 9.1 → 8.2 cm (both: still settling) | 72 / 346 → 76 / 355 |
| legocar | 79 → 59 J, 5.2 → 3.8 m/s | 11.2 → 9.0 cm | 540 / 11 → 482 / 10 |
| Monster supercar | 315 → 163 J | 57 → 43 cm sideways (it rolls; before and after) | 0 / 37 → 0 / 35 |
| Astral R | 87 broken, 141 m/s → 0 broken, 3.3 m/s | — → 3.3 cm | (diverged) → 213 / 267 |

The legocar's wall run has one contact event of its loose bricks at 1.5 s (a 1.3 kg
brick node at 50 m/s for one sample; 13 kJ at 1.5 s, 681 J at 2.0 s); the bilinear
run has its own at other times (22 m/s at 1.0 s). The break count of a car that comes
apart into bricks is chaotic.

## 7. Couplers (kind `JbeamCoupler`) [C]

Coupled nodes are "considered one and snap together while connected"; they part when
the force passes the coupler strength, or when their `breakGroup` breaks
(`breakGroupType` 1: broken by the group only). Couplers start uncoupled unless
`couplerStartRadius` (`couplingStartRadius` in advanced coupler controllers) sets them
active at spawn. The importer reads advanced coupler controllers' `couplerNodes` tables
(`cid1`, `cid2`, `autoCouplingStrength`, `couplingStartRadius`, `breakGroup`); a pair
within its start radius at spawn is coupled; others are left out (warning). Node-level
`couplerTag` couplers are not read.

**Choice (no stiffness is documented):** an isotropic zero-length spring–damper,
F_a = −k·(x_a − x_b) − c·(v_a − v_b), with k = 0.16·μ/h² and c = √(k·μ) (μ the pair's
reduced mass): ω·h = 0.4 and a damping ratio of 0.5, as stiff as the fixed step allows
with margin. It breaks when |F| > strength. The BX's doors, bonnet and boot are held
by five such latches; without them the boot lid rattled on its supports.

## 8. Break groups [B], [T], [C], [TB]

* A beam (or coupler, torsion bar) in group g with `breakGroupType` 0 breaks every
  element of g when it breaks, and is broken when g breaks. With type 1 it is broken by
  g but never sets it off. ("Type 1 for torsionbars is exactly the same as having no
  breakGroup on them at all" [TB]: the importer drops the group.) This is the solver's
  detacher-group mechanism with passive members (negative group numbers), which already
  existed for RoR; the `.truck` writer could not express passive members.
* JBeam modifier rows carry `breakGroupType` and `breakGroup` to every following row
  until reset; the BX's door detents and limiters thereby are passive members of the
  hinge groups, as the data says.
* A beam in several groups (a table) uses the first only (warning; the docs call it
  "very rarely used"; neither test vehicle does).
* **Not simulated: triangle breaking.** Triangles stop colliding when a side beam or
  their group breaks [T]; here collision triangles stay (as on the `.truck` path).

## 9. Torsion bars and torsion hydros (kinds `JbeamTorsion`, `JbeamTorsionPair`) [TB], [TH]

Four nodes: lever end 1, axis 2–3, lever end 4. `spring` (N·m/rad), `damp`
(N·m·s/rad), `deform` and `strength` (N·m), `precompressionAngle` (rad), `breakGroup`.
**Choice (no law is documented, the units are torsional):** φ is the dihedral angle of
planes 1-2-3 and 2-3-4 about 2→3, φ₀ the spawn angle plus `precompressionAngle`, and
the torque τ = −k·wrap(φ − φ₀) − c·φ̇ acts as forces τ·∂φ/∂xᵢ on the four nodes (the
Blondel–Karplus gradient). Past the deform torque φ₀ follows φ so that k·|φ − φ₀| stays
at it (perfect plasticity; no hardening is documented). Past the strength it breaks.
A torsion hydro has no input here: it is a torsion bar at its spawn angle (the legocar's
door hydros hold the doors shut this way).

"If a beam located between 2 consecutive nodes of a torsion bar breaks, the torsion bar
itself will break" [TB]. Implemented through break groups: each beam between
consecutive nodes (1–2, 2–3, 3–4) that has no group gets a group of its own, and the
torsion element watches the groups of its links; it breaks the step after one of them
breaks (the GPU reads the last step's group slots, so both engines break it then). A
link that is a passive group member breaking on its own is not seen (warning).
`spring2`/`damp2` (anisotropic torsion bars) and `precompressionTime` are not simulated
(warning; unused by the test configurations).

Each element takes two beam records (nodes 1–2, then 3–4) so that its four node forces
use the solver's incidence entries; the second record has no force of its own.

## 10. Self-collision [N], [CI]

`selfCollision` (default **false** [N]) makes a node collide with its own vehicle's
triangles; `collision: false` turns all its collision off (and fixed nodes have none).
The `.truck` path made every colliding node a self-collision contacter.

**Choice (exclusion):** a node skips the collision triangles whose three nodes share
its first node `group` (the importer's `self_group`, the rule Rubble's BeamNG-parity
solver already uses). Without it a panel's middle node lies in the plane of its own
neighbouring triangles: the BX boot lid's centre node sat 3 cm from its own panel and
was kicked by the self-collision throttle every eighth step. Collision triangles,
range (0.03 m) and contact rules are otherwise the solver's.

### 10.1 Self-collision pairs that start inside the contact range (2026-11-05)

[N]: a node starts feeling a triangle "at around 2.5cm"; [CI]: "nodes that spawn within
triangles, and get pushed out on spawn" (named as a cause of spawn loads and
instability); BeamNG documents no rule for them otherwise. The solver's contact is a
projection (it cancels the node's force and removes a fifth of the depth per step),
not a spring, so a push-out at spawn is as strong as whatever holds the node: the
Astral R's rear bumper centre node `rb5` (lying on the diffuser's triangle edge)
chattered at 8 m/s, and the forklift's beacon, whose base nodes start 1 cm above the
roll hoop's triangles, broke its 3.8 kN mounts at 52 ms.

**Choice:** a self-collision node that starts inside a collision triangle's contact
range (inside it, within the 3 cm range of its plane, and not excluded by §10's
group rule) never collides with that triangle (`ExplicitBeams::spawn_exclusions`,
`tri_excl`; at most `TRI_EXCL_CAP` = 16 per triangle). JBeam mode always; the
`.truck` path only through `opt_in_full_contact` (§10.2). GPU: per triangle (start,
count) at `triExcl` (params word 69; 0 when no vehicle has any), the nodes in
`gLists`; `tryContact` skips them. Effect: the Astral R excludes 17 pairs (rest KE
3.5 J at 4 s, was 18.7 J); the forklift 12 (it rests with nothing broken); the legocar
rests at 18 J (was 82 J).

### 10.2 Vehicle contact from every node (2026-11-05)

[N]: "The `collision` argument affects all types of collision". In JBeam mode every
node with `collision` is a contacter against other vehicles' triangles (as before).
The `.truck` path's contacters are RoR's (its contacter flag, or a node on a
triangle): a car's frame nodes are not, so once its skin crumples they pass into the
other car. `ExplicitBeams::opt_in_full_contact` (JSON `fullContact: true` on a
vehicle of `explicitSim`) makes every node without `collision: false`
(`NODE_NO_GROUND`) a contacter, and computes §10.1's exclusions; RoR content that does
not opt in is unchanged (bit for bit). A `.truck` file opts in itself with the line
`rubble_full_contact` (**Rubble extension, not RoR**; the showcase car's authoring
writes it; RoR files without it are read as before). The vehicle-contact kernels take the contact
point lists as data, so every backend runs it.

### 10.3 Two-body vehicle contact (JBeam mode, 2026-11-05)

The vehicle-contact response (`explicit-beams-vehicle-contact-spec.md` §6, RoR's)
cancels the contacter's **whole** force, as against a static surface, uses the
weighted mean m_t = Σ w_k·m_k as the triangle point's mass, and sizes every hit as if
it were alone. With the FMVSS 214 barrier (0.8 kg face and 4.7 kg back-plate nodes)
on the showcase car's skin (0.07–0.3 kg nodes) each of the three threw light nodes
off: a 0.07 kg node at weight 0.77 beside a 10 kg node at 0.14 has m_t = 1.5 kg
by the mean, 0.12 kg by its impulse response; the cancelled column force of a
heavy node (8–20 kN) went whole onto that node (Δv = 150 m/s in one step); and up to
six back-plate nodes hit one triangle in a step, each removing the approach on its
own. The car's nodes reached 300–370 m/s at the high resolution (CPU and GPU) and
diverged at the medium one before the barrier's shear was rebuilt (§24 below).

**Rule** (exact physics of the lumped masses; `explicit_contact.rs`, `vvTry`) in a
hit where the owner or the partner has `ExplicitBeams::two_body_contact`: on in JBeam
mode (so any pair with the FMVSS 214 barrier); off on both vehicles of a pair, RoR's
rule, bit for bit. **Not yet the `.truck` default:** made the default, it raised the
showcase car's 80 km/h T-bone at the door middle to 62 of 1,141 striker nodes inside
the struck skin (6 with RoR's rule; the limit is 1 %), so it waits on the leak (below).

* **Mass splitting** (Tonge, Benevolenski and Voroshilov, SIGGRAPH 2012, as the
  World's rigid proxies, `explicit-vehicles-in-world.md` §1): in one owner's turn
  every node counts its two-body hits (as the contacter, or as a vertex of the hit's
  triangle) and is k sub-bodies, inverse mass k/m each; an immovable node has none.
* **Triangle point**: inverse mass Σ w_k²·k_k/m_k (w_k = 0 left out): the reaction
  is shared out as w_k·J, so the point's velocity changes by J·Σ w_k²·k_k/m_k. The
  reduced mass m = 1/(k_h/m_h + Σ w_k²·k_k/m_k).
* **Cancelled force**: (F_h/m_h − Σ w_k·F_k/m_k)·m, the pair's relative acceleration
  from the forces at the contact pass's start (every node's accumulator before any
  vehicle contact of the step) times the reduced mass: the force that zeroes their
  relative acceleration, normal and tangential (friction's slip force), in place of
  RoR's whole F_h. The approach and penetration terms use m. Friction as before.
* GPU: per-vehicle flag bit 7; the forces at the pass's start in gBeams at `vvSnap`
  (written at the pass's start), the counts in gAcc at `vvSplit` (per node its
  triangle count, then `VV_PART` point slots: the small groups' owners count side by
  side), cleared after each turn. A turn whose hit list overflows applies at once
  with the counts found so far (as before, out of the specification's order).

Effect (MDB at 53.9 km/h, 27° crab, showcase car, GPU, fastest node over 1.5 s):
low / medium / high resolution 93 / 34 / 372 m/s before (diverging at medium the
day before), 30–48 / 47–66 / 42–46 m/s after; nothing non-finite; the vehicles'
kinetic plus gravitational energy never rises (`explicit_scenes.rs`).

**Energy (`energy_ledger.rs`).** In the showcase T-bone at 80 km/h, 0.7 m off the
door middle, RoR's rule does +1.18 MJ of friction and +0.18 MJ of normal work on the
two cars (striker 379.7 kJ): the static-surface projection puts the contacter's whole
force, normal and tangential, on light triangles, and the rattling nodes that results
(30–66 m/s under a 15 m/s barrier) are what the air and the beams then soak up. With
this rule every contact term is negative and the beams dissipate 115 kJ of the 338 kJ
lost.

**Leak (80 km/h sedan T-bone, every node a contacter, CPU, five offsets −0.3 to
0.45 m, most striker nodes inside the struck skin of 463):** RoR's rule 63 in total
(28 at −0.3 m); this rule's first form (cancel the share m_t/(m_h + m_t) of F_h
alone) 84; this rule 64. Every leaked node is a contacter over a struck triangle that
has crossed its whole ±3 cm band (none through gaps, none untested: the throttle off
changes nothing); one traced node entered a triangle's prism sideways, in front of
it, and the side vote (its neighbours behind the plane) pushed it through. Open
(task 3): a side kept per contacter–triangle pair cut it to 57 only.

**Leak, second pass (2026-11-05, showcase car, CPU, TB_DZ 0 / 0.43 / 0.7 m, most
striker nodes inside of 1,141).** Today's car: RoR's rule 9 / 78 / 28, this rule 91 /
144 / 88. Tried, all reverted (`explicit_skin.rs` keeps the side test as a
measurement; `TB_SIDECHECK`, `TB_TRACE`, `TB_RULE` in `showcase_car.rs`):

* *Exact side* (closest feature of the closed skin by angle-weighted pseudo-normals,
  Bærentzen and Aanæs 2005; the lattice is one closed, oriented component): it
  agrees with the winding number for 80–90 % of striker nodes within 2 cm of the
  skin (the rest sit at folds of the crushed, self-intersecting skin). As the side
  of RoR's prism hits (an inside point pushed out through its nearest hit, along
  the outward normal): this rule 52 / 104 / 72; RoR's rule diverges (NaN).
* *Closest-feature contact* (one contact per point at its closest feature, face,
  edge or vertex, pushed out the way it came in; no prism gaps): this rule 33 / 99 /
  77; RoR's rule diverges (an edge or vertex hit puts the contacter's whole force on
  one or two light skin nodes, §10.3's failure).
* The throttle off: unchanged (the fast-approach reset tests every triangle while
  it matters). Holes: winding over the live triangles only counts as many.

**Mechanism (traced, this rule):** the nodes are not misplaced by the side; they
grind through. A striker node in contact approaches at 2–6 m/s every step with a
1.5–2.5 kN contact force on it at a reduced mass of 0.03–0.06 kg: the struck
lattice's triangles are coarse beside the striker's nodes, so each struck node
takes part in many hits, and the mass splitting (one Jacobi pass) leaves each hit
a small share; without the splitting the pass diverges, and a naive iterated
Jacobi (4 passes) gives 13 / 78 / 34 but diverges at 16. Open: a converged contact
solve (a Gauss–Seidel or accumulated-impulse iteration per turn) is what the leak
asks for; the side test alone does not fix it.

**Leak, third pass: a converged solve (2026-11-05, showcase car after the
body-in-white merge, 1,157 striker nodes, so the limit is 11; CPU; TB_DZ 0 / 0.43 /
0.7 m; most striker nodes inside).** Today's car: RoR's rule 11 / 129 / 60, this rule
106 / 190 / 131. Tried, all reverted (the patch is not in the repository):
* *Iterated mass-split Jacobi per owner's turn* (accumulated normal forces, the
  turn's forces summed in fixed point so any order gives the same sums, each hit moved
  by its split reduced mass times the residual of its relative normal acceleration,
  the contact law's target as the fixed point). With §10.3's w²·k splitting it
  diverges (16 passes: 4 / 50 / 0 with thousands of beams broken, NaN): a node with a
  small weight in one hit and large ones in others is under-split. Splitting by the
  summed weights S (inverse mass Σ w·S/m, which bounds every row of the hits' coupling
  by its diagonal, Gershgorin) converges: 8 / 32 / 128 passes 7 / 99 / 43, 7 / 97 /
  77, 16 / 64 / 64 (diverging at 128).
* *The same over the whole contact pass* (every owner's two-body hits solved together;
  the walks read no force, so walking every owner first changes nothing for RoR's
  rule): 16 passes 7 / 98 / 18, 64 passes 6 / 89 / 9 (diverging at 0.7 m). Each pass's
  largest change after 64 passes is still 0.1–0.5 % of the total.
* *Energy (`energy_ledger.rs`, T-bone 0.7 m):* the single pass does −11 kJ normal and
  −34 kJ friction work; converged, the contact law's own terms put energy in: the
  depth push 0.2·p/h and the cancelled force on separating hits (the law holds a
  separating pair's velocity) do +22 kJ normal work at 16 passes, +58 kJ normal and
  +1.57 MJ friction at 64. Without the push: +9.5 / +15 kJ normal, and the leak 56 /
  105 / 35 (16 passes). With an inelastic target (relative normal velocity ≥ 0, no
  push): −13 kJ normal, −15 kJ friction at 16 passes (friction +35 kJ at 64), leak
  35 / 106 / 73.
* **So the leak at 0.43 m is not the solve's convergence:** every variant leaves
  89–106 striker nodes inside, converged or not, against 129 with RoR's rule. A
  converged two-body solve fixes the door middle (6–7, inside the limit) and
  helps 0.7 m, but RoR's contact law iterated to convergence is not dissipative.
  **Open:** where the 0.43 m nodes get in (they are no longer hits: about 70 hits a
  step at the peak); two-body stays JBeam mode's default only.

**Leak, fourth pass: candidate generation traced (2026-11-05, same car, RoR's rule,
CPU; `showcase_car.rs` test t with `TB_LEAKED=1` (each leaked node classed by the
nearest struck triangle whose prism holds it) and `TB_HTRACE=<node>` (per step every
struck triangle whose prism holds the node: δ, in band, its contact throttle, dead)).**
* Not the throttle (every traced triangle had wait 0), not a hit cap (the CPU has
  none; 40–90 hits a step is the contact patch), not spawn exclusions (vehicle
  contact has none), not the range, not contact-point lists (every leaked node is a
  contact point).
* **Half are over broken triangles.** At 0.43 m the latch lets go at 0.05 s and 36–39
  struck triangles die (§18: the body–door shut-line triangles, nodes on `body` and
  `door_f_l`); a dead triangle has no contact (as specified), but the leak metric's
  winding number counts it as skin. Leaked nodes by their nearest prism's triangle:
  0.43 m 47 live / 39 dead at 0.1 s, 66 / 63 at 0.2 s, 40 / 44 at 2 s; 0.7 m 37 / 22
  at 0.2 s; at the door middle (0 m, nothing broken) all 7–11 live. Node 318 (0.43 m)
  sat in the band of triangles 352, 354, 312 (all body–door, all dead from 0.055 s)
  and crossed with no hit.
* **The live half cross at an edge.** Node 394 (0.43 m) was in contact with triangle
  572 for 25 steps, its δ between +2 cm and −0.5 cm; at step 170 it left 572's prism
  across an edge that no neighbouring prism covers (a convex fold of the crushed
  skin) and had no hit for two steps; it re-entered 572's prism at δ = −1.6 cm, was a
  hit once more, and the next step was at −2.1 cm, out of the band (the response
  pushed it inward: with δ < 0 the side vote reads its neighbours behind the plane),
  then −8 cm. Edge and vertex regions between prisms are the gap (§10.3's
  closest-feature attempt closed them but put RoR's whole force on one or two light
  nodes).
* **Open:** (1) whether a node over a dead shut-line triangle is a leak at all (the
  door has opened there) or the metric should take the live skin plus the opening;
  (2) the prism gap at folds, with a side kept per contacter from its last hit.

**Leak, fifth pass: the metric corrected (2026-11-05; a refinement of the acceptance
criterion, `vehicle-acceptance-criteria.md` A1).** The old metric counted a striker
node as leaked when the generalised winding number of the struck car's whole skin,
dead triangles included, held it inside (|w| > 0.5). Once the latch lets go at
0.43 m and 0.7 m, the body–door shut-line triangles die (§18) and stop contacting, as
specified, so the door has opened there; a striker node in that gap is in a real
opening, but the winding number still counted it. **The criterion now:** a node has
leaked when it has penetrated the live skin, that is, when its net crossings of the
struck skin's live triangles since spawn, inward less outward (by the closed
lattice's outward orientation), are not zero. `explicit_skin::LeakTracker` follows
it step by step: a step's crossings are the integer jump of the node's winding
number over the nearby triangles (each triangle's solid angle changes by about 4π
when the node passes through it), put to the live or the dead triangles by their
larger share; a crossing of a dead triangle is not counted. With nothing dead it is
the winding number's own count (door middle, RoR's rule: 13 and 13 at the peak). A
node with net −1 went in through an opening and out through the skin, or through a
triangle the crush turned over; both are penetrations and count. The test also
reports the deepest leaked node's distance to the live skin, how many are past the
contact band, the crossings through dead triangles, and the old winding count.

Today's showcase car (1,165 striker nodes, limit 11; `showcase_car.rs` test `t`, CPU;
TB_DZ 0 / 0.43 / 0.7 m; most striker nodes leaked at a sampled frame, the old
winding count in brackets):

| Rule | 0 m | 0.43 m | 0.7 m |
|---|---|---|---|
| RoR's | 13 (13) | 43 (149) | 85 (96) |
| Two-body (§10.3) | 101 (118) | 134 (211) | 135 (135) |

Through dead triangles: 0 / 960 / 1,085 crossings with RoR's rule (nodes moving in
and out of the opening). Traced leaked nodes at 0.7 m (`TB_XTRACE=<nodes>`) cross
live triangles around the fold of the crushed door (572, 564, 566, 568, 606) in and
out every few steps: they sit on the plane inside the contact band, where the hit's
side vote decides.

**Leak, sixth pass: where the crossings start; the prism-gap fixes measured
(2026-11-05, same car, CPU, corrected metric).** `showcase_car.rs` test `t` now classes
every inward crossing of a live triangle by where the node was against that triangle at
the contact pass before (positions after the last step, whose hits set this step's
forces). It also classes each hit's side (by the skin's outward orientation) and the
leaked nodes at the worst frame by their last inward crossing.

| Rule, offset | Inward crossings: hit / beyond band / outside the prism | Leaked by last crossing: hit / beyond / outside / out only | Hit crossings: vote pushed in / pushed out |
|---|---|---|---|
| RoR's, 0 m | 121 / 4 / 11 | 9 / 2 / 2 / 0 | 14 / 107 |
| RoR's, 0.43 m | 295 / 30 / 21 | 26 / 2 / 5 / 10 | 116 / 179 |
| RoR's, 0.7 m | 532 / 35 / 13 | 74 / 6 / 5 / 0 | 217 / 315 |
| Two-body, 0 m | 314 / 0 / 23 | 97 / 0 / 4 / 0 | 15 / 299 |
| Two-body, 0.43 m | 575 / 1 / 6 | 128 / 1 / 4 / 1 | 239 / 336 |
| Two-body, 0.7 m | 517 / 3 / 17 | 124 / 1 / 10 / 0 | 260 / 257 |

**So the prism gaps are not the leak.** 2–10 leaked nodes per case last crossed from
outside the prism; 80–95 % were hits of the triangle they crossed, and every hit
crossing came from in front. Of those, 30–50 % at the offsets were pushed in by the side
vote (the node in front, its neighbours behind the plane). The rest were pushed out and
crossed anyway. In a trace (node 397 against triangle 606, 0.7 m), the hit swings by
±1.5 cm a step for ten steps before the node goes through: a contact that rings rather
than holds.

Measured, all reverted (patches in the session scratchpad, not in the repository; most
leaked / 1,165; "div." = the run went non-finite):

| Variant | RoR's rule 0 / 0.43 / 0.7 m | Two-body 0 / 0.43 / 0.7 m |
|---|---|---|
| Today | 13 / 43 / 85 | 101 / 134 / 135 |
| Side memory: last hit's push normal for 20 steps, else the vote | 30 / div. / div. | 111 / 101 / 116 |
| Side memory, else the side of δ | 23 / div. / 48 | 114 / 120 / 108 |
| The side of δ, no vote | 27 / 91 / div. | 154 / 171 / 176 |
| Prisms extended across live edges to the dihedral's bisector plane (weights clamped, renormalised; 6–11 % of hits are new) | 13 / div. / div. | 82 / 118 / 161 |

The vote is also what recovers a penetrated node: without it, outward crossings fall
from 133 to 16 (RoR's rule, 0 m). Extended prisms under RoR's rule put the whole force
of an edge hit on one or two light nodes, the closest-feature attempt's failure, and
diverge. **Not adopted;** two-body stays JBeam mode's default only, because it leaks
more than RoR's rule at every offset. **Open:** the ringing hit. The response
over-corrects the light struck triangle when many striker nodes hit it at once (RoR's
rule sizes each hit as if it were alone). Two-body's splitting under-corrects, and a
converged solve injects energy (third pass). This is a contact-response question, not a
candidate-generation one.

**Leak, seventh pass: the one-tier car (`SHOWCASE_STYLE=sedan1`, 2026-11-05, 1,052
striker nodes, limit 10; CPU, live-skin metric; most leaked, the winding count in
brackets).** RoR's rule 23 (42) / 7 (38) / 4 (5) at 0 / 0.43 / 0.7 m; two-body 13 (57) /
11 (37) / 11 (31). With 1.3 kg body nodes, the hit crossings that leak are a handful,
not hundreds. The count is chaotic at this level. Near the door middle, RoR's rule
gives 9 / 23 / 13 at −0.03 / 0 / +0.03 m; near 0.43 m it gives 12 / 7 / 4 at 0.40 /
0.43 / 0.46 m. Every case is within about twice the limit, and some are inside it.

*Each hit's triangle mass split among the turn's hits on that triangle* (RoR's rule:
m_t / n_t in the reduced mass, n_t the triangle's hits in the owner's turn; measured,
reverted). sedan1: 17 / 10 / 18 at −0.03 / 0 / +0.03 m, 4 / 19 / 5 at 0.40 / 0.43 /
0.46 m, 7 at 0.7 m. The two-tier car: 15 / 72 / 95 (was 13 / 43 / 85). No systematic
gain on either car, so not adopted.

## 11. What the BX and the legocar use, and what is left

Feature census (`examples/jbeam_census.rs`) of the two test configurations: NORMAL,
SUPPORT (530 and 438 rows), BOUNDED (41 and 4), `breakGroupType` 1 (68 and 13),
`beamPrecompression`, `deformLimitExpansion` (876 BX rows), `boundZone`,
`beamLimitSpring`/`beamLimitDamp`, `longBoundRange`/`shortBoundRange`, `beamDampFast`
and `beamDampVelocitySplit` (legocar), `dampCutoffHz`, hydros, couplers (BX), torsion
bars and torsion hydros, `selfCollision`, triangle and quad `breakGroup`.

Not simulated (warnings name each when a file uses it): triangle breaking and
`disableTriangleBreaking`; ANISOTROPIC beams (run as NORMAL; [A]: `springExpansion`,
`dampExpansion`, `transitionZone`); PRESSURED beams (run as NORMAL); LBEAMs [L]; rails
and slidenodes; pressure wheels (soft tyres instead, §12); `beamDampRebound` and the
other rebound values; `beamPrecompressionTime`; `beamLongExtent`; `spring2`/`damp2`;
couplers not coupled at spawn and node-level couplers; several break groups per
element; deform groups (visual); flexbodies, props, triangle aerodynamics
(`dragCoef`, `liftCoef`), `groundModel`s, node friction models and powertrains.

## 12. Wheels

BeamNG's `pressureWheels` generate a pressurised tyre. That model is not implemented:
the JBeam mode spawns each wheel as the `.truck` path's soft tyre (12 rays, mass,
spring and damping from `TyreDefaults::for_vehicle`), on the same axle nodes, radius
and width, after the desc's nodes and beams. A faithful pressure wheel is the largest
remaining gap for driving BeamNG content.

## 13. GPU layout and RoR compatibility

The kernel runs the JBeam kinds in their own functions (`jbeamUpdate`,
`jtorsionUpdate`) behind an early return in `beamUpdate`; the RoR kinds' code is
unchanged. Records per beam (kinds ≥ 6): (k, c, limit k or deform-stress cap, limit c),
(short, long, bound zone or 0, damping filter record + 1 or 0); for plain and support
beams the rest-length limits and the support's break length are in metres; bounded
beams keep the velocity split and fast damping in the Tc and Tt state slots (they never
yield); torsion elements keep φ₀ in the rest slot and their watched groups in the
second static record. Damping filters are gBeams records (u₋₁, y₋₁, a, b) after the
vehicle-contact runs. A node's self-collision group rides in its force record's .w and
a triangle's in its throttle record's .z; the node pass and the throttle now keep
those words instead of writing zeros, which changes nothing for RoR content (zeros).
`explicit_beams_bitwise` (the current kernel against the frozen reference on the
repository's sedans and the Viper, E36, Mustang and Agora S) stays bit for bit.

## 14. Verification (2026-11-03, M3 Max)

Vehicles: BeamNG's BX coupé body (`bx` with `rubble_body_only.pc`: 452 nodes, 3203
desc beams, 3228 solver records with 5 couplers and 10 torsion elements, 559
triangles, no wheels) and the legocar (`agenty_legocar` `start.pc`: 757 + 96 tyre
nodes, 5888 records, 4 wheels as soft tyres). Neither is in the repository; the
opt-in tests read them from `RUBBLE_JBEAM_BX` and `RUBBLE_JBEAM_DIR`.
`examples/jbeam_idle.rs` runs them on the CPU reference (drag off), lowest node 2 cm
above the ground.

**At rest, 30 s.**

| vehicle | path | broken | bent | kinetic energy at 30 s |
|---|---|---|---|---|
| BX body | `.truck` (before) | 18 at step 16 (both doors) | 51 | 110 J at 3 s |
| BX body | JBeam mode | 0 | 0 | 2.7 J |
| legocar | `.truck` (before) | 98 in 5 s | 5 | 1,158 J at 5 s |
| legocar | JBeam mode | 0 | 4 | 71 J |

The BX's remaining motion is its body-only configuration: without wheels it rests on
its exhaust and sill skirts (0.3–1 kg nodes), which chatter on the ground at up to
1.8 m/s. The legocar's 4 bent records are four engine-bay supports with
`beamDeform: 0` and `beamSpring: 1` that yield under any load (as the file says). Its
remaining energy is in its rear wheelhouse fillers: 13 node–triangle pairs of
different parts (fenders over the fillers, hubs inside their axles) start inside the
0.03 m self-collision range, and the solver's self-collision contact pushes them apart
every few steps; with self-collision off the legocar settles to 1.2 J within 6 s.

**Into a rigid wall at 100 km/h** (27.78 m/s, 2 s, CPU):

| vehicle | path | broken | bent | KE at 2 s / at spawn | peak node speed | NaN |
|---|---|---|---|---|---|---|
| BX body | `.truck` | 294 | 840 | 921 J / 320 kJ | 2,056 m/s | no |
| BX body | JBeam mode | 249 | 705 | 665 J / 320 kJ | 339 m/s (one front bumper node in the first 8 ms at the wall) | no |
| legocar | `.truck` | 1,238 | 8 | 24.5 kJ / 562 kJ | 742 m/s | no |
| legocar | JBeam mode | 1,289 | 11 | 130 kJ / 562 kJ | 1,557 m/s (a soft-tyre node) | no |

The BX folds its front (bumper, bonnet, wings, doors) and keeps its cabin; its energy
falls monotonically. The legocar comes apart into its bricks, which is what its file
asks (`beamDeform` equal to `beamStrength`: its beams break without bending). Its
front soft tyres, the `.truck` path's stand-ins for pressure wheels, go unstable when
crushed against the wall: their spokes pass their bounds, where the RoR tyre model's
limit damping (12,000 N·s/m) on 0.64 kg tyre nodes exceeds what the 0.5 ms step can
integrate, and a detached tyre node leaves at up to 1.5 km/s (94 kJ of the 130 kJ).
That is the wheel gap of §12, not a JBeam rule.

**GPU against the CPU** (`tests/explicit_jbeam_gpu.rs`): the fixture at rest agrees to
0.14 mm over 2 s with equal counts, and its crash has equal broken and bent counts;
without self-collision (fixed point on the GPU) the BX and the legocar agree to under
a micrometre over the first 100 steps (bit for bit without torsion elements, whose
`atan2` differs in the last bits); over seconds the parked cars' chatter takes runs
apart as any two float-level different runs; and the 100 km/h crash's GPU counts lie
within the spread of five nudged CPU runs (BX: GPU 256 broken, 693 bent; CPU 185–235,
701–714; legocar: GPU 1213, 11; CPU 1100–1237, 9–11).

**RoR content unchanged**: `explicit_beams_bitwise` (11 tests: the repository's
sedans, Viper, E36, Mustang, Agora S, fleets, wide passes) bit for bit against the
frozen reference after every kernel change; the E36 drive accuracy ensemble
(`nudges=24 ror_reuse=1`, 120 fps, 100 km/h): median node RMS Rubble–RoR 91.34 mm
against RoR–RoR 90.44 mm, ratio 1.01 [0.74, 1.33], broken 0 / 0–0, bent 11 / 10–13:
PASS, the same numbers as before this work. Chrome compiles all 23 shipped modules
(`explicit-wgsl-check.mjs`).

## 15. Capacity

Not measured yet. The `impact` run (`explicit_beams_scaling jbeam=<dir>:bx:rubble_body_only.pc
scenarios=impact counts= rubble_counts=1,…,2048 fps=120 frames=360 speed=27.78 drag=0
trials=3 nudges=0 ceiling_ms=33.3 deadline_s=780 max_load=8 max_gpu=30`) measured
trial 0 at 1 to 8 vehicles (GPU time per frame 1.15, 2.29, 3.00, 3.89 ms), then
waited for a quiet GPU: macOS `mediaanalysisd` held it at 100 % and the run was
stopped at its 15-minute limit. `mixed` was not started. Body-only means no tyres:
the cars slide on the ground.

# Part II: the remaining JBeam features (2026-11-05)

Same rules as Part I: semantics from BeamNG's public documentation only, JBeam files
read as data, choices marked **Choice**. These sections supersede the "not simulated"
entries of §8, §11 and §12 that they cover. Further sources (retrieved 2026-11-05):

* [P] Pressured beams: https://documentation.beamng.com/modding/vehicle/sections/beams/pressured/
* [WH] Pressure wheels: https://documentation.beamng.com/modding/vehicle/sections/wheels/
* [R] Slidenodes and rails: https://documentation.beamng.com/modding/vehicle/sections/rails/
* and [B], [A], [BD], [S], [L], [T] of Part I, re-read for the fields below.

**One rule for all of Part II: unused features cost nothing and change nothing.** A
beam that uses none of them is spawned, stored and stepped exactly as before (the
same kind, the same records, the same code path), so every vehicle without them runs
bit for bit as before. A beam that uses one carries an *extension record* (§16.1) and
takes a separate function behind an early return, on the CPU and the GPU alike.

## 16. Extension records

16.1. A JBeam-mode beam with any Part II field gets an extension: on the CPU
`ExplicitBeams::beam_ext` (by `ExplicitBeams::ext_of[j] − 1`), on the GPU a block of
four `gStatic` records (its index in `gLists` at `extIndex + j`), plus one mutable
`gBeams` record of state. The beam's type byte carries `TYPE_EXT` (bit 2): only such
beams read their index. Fields:

* E0: feature mask, deform groups 1 and 2 (slot + 1, 0 none), `deformationTriggerRatio`;
* E1, E2: the kind's parameters (§19–§22);
* E3: the precompression ramp (§23) and the extra break groups 2 and 3 (§17).

## 17. Several break groups per element [B], [T]

"A table assigns the beam to several groups. It then breaks when any of those groups
breaks, and it triggers breaking in all of them."

* **Broken by any.** A beam in groups {g₁, g₂, g₃} is broken when any of them is set
  off, exactly as a single-group member (it drops the force it computed in a step
  whose group master precedes it in beam order).
* **Sets off all.** An active member (`breakGroupType` 0) that breaks on its own
  sets off all its groups. **Choice (deterministic order):** it does so one step
  later, at its own place in that step's beam order, and only when none of its groups
  had been set off before it in the step it broke (as the beam-order rule already
  decides whether it was alive). The CPU does this with a pending flag; the GPU in the
  next beam phase, reading the last step's group slots. Breaking groups through a
  group (a member broken by g₁ does not set off g₂) does **not** propagate: the
  docs give a beam's own break as the trigger.
* A passive member (type 1) is broken by any of its groups and sets off none.
* At most three groups per beam (the first three; a warning names more). Neither the
  BX nor the legocar uses tables; the Astral R uses 75.
* Triangles with several groups: §18 lists up to one group per triangle (the first;
  warning).

## 18. Triangle breaking [B], [T]

"By default, a triangle breaks when any beam forming one of its sides breaks", and
when its `breakGroup` breaks; `disableTriangleBreaking` on a beam keeps that beam's
break from breaking triangles ("when the beam only serves as structural support").
A broken triangle switches "to a non-collision, non-aero state".

* In JBeam mode each collision triangle gets a break list: for each side, the first
  beam (desc order) between its two nodes without `disableTriangleBreaking`, and a
  representative of its break group: the first beam of the group that sets it off on
  its own (an active, single-group member), when there is one (warning otherwise).
* **Dead** from the step after any listed beam is broken: the state at the start of
  the step's beam pass. A dead triangle takes part in neither self-collision nor
  vehicle contact: it has no candidates (the throttles see "no candidates"). It never
  comes back.
* Vehicles where no triangle can break (every side beam has
  `disableTriangleBreaking`, no group) get no break lists: nothing changes.
* GPU: per triangle four beam indices at `triBreak + t`; the triangle's self-group
  word (`gMisc` .z) carries bit 31 when it has a list. The triangle geometry
  (`triangleGeo`, `vvTriangle`) reads the four beams' broken state (`gBeams` .z set,
  .w not 1: broken before this beam phase) and gives a dead triangle an empty box.

### 18.1 Rubble's `.truck` extension: breakable cab triangles

**Rubble extension, not RoR** (the showcase car's authoring, `truck.rs`): a `cab` line
may carry a fifth field, a break group (a number: the beams' group of the same
number), and a sixth, the edges that break it, a bit mask (bit k: the edge from its
node k to node k + 1; default 7, all three). A triangle with a break group is dead
(§18: no collision, no aero) once that group is set off; its edge mask says which of
its sides' beams count. RoR's own `cab` lines have four fields: RoR files are read as
before, bit for bit.

### 18.2 Rubble's `.truck` extension: a file that is its vehicle at rest (2026-11-05)

**Rubble extension, not RoR** (the showcase car's authoring, `truck.rs`,
`ExplicitBeams::from_truck`). A file may be its vehicle standing at rest on flat
ground, so it spawns with nothing moving: no drop onto its tyres, no settling
transient.

* `rubble_rest_pose <y>`: the node positions are the vehicle at rest on ground at
  height `y`. The beams keep their own rest lengths through the beams' fifth field
  (the rest length over the spawn length). Shocks keep theirs through their
  precompression.
* `rubble_tyre_pose`, a section of two kinds of lines:
  * `wheel, x1, y1, z1, x2, y2, z2`: the rest positions of the wheel's axle nodes.
    The tyre's beams' rest lengths are built round these positions (the
    undeflected tyre).
  * `wheel, node, x, y, z`: where the tyre's generated node `node` spawns (the
    deflected tyre at rest). `wheel` is the file's wheel index; `node` runs from 0
    to 2 × rays − 1, in build order.

  A posed tyre is built from its line's first axle node. The lower-z rule does not
  apply to it, so its nodes are named as they were built.
* `rubble_spawn_contacts`, lines `a, b, c, node`: a collision triangle (by its
  nodes) and a node that collide although the node starts in the triangle's range.
  §10.1's spawn exclusions are the design pose's; these pairs rest in contact.
* `ExplicitBeams::spawn_clearance()` gives the height to spawn the lowest node above
  the ground. It is 2 cm for any other file. For a rest pose it is minus the
  lowest node's depth under its ground, because the tyres' contact patches stand in
  contact. Consumers spawn with it in place of a fixed 2 cm.

Files without these lines load as before, bit for bit.

### 18.3 Rubble's `.truck` extension: JBeam's plastic limits on a beam (2026-11-05)

**Rubble extension, not RoR** (`truck.rs`, `ExplicitBeams::set_jbeam_plastic`). A
section `rubble_jbeam_plastic` of lines `a, b, deformLimitStress, deformLimit,
deformLimitExpansion` turns the file's plain deformable beam between nodes a and b
into a JBeam NORMAL beam (`JbeamPlain`, §3) with its spring, damping, thresholds,
strength and plastic coefficient: its hardened thresholds never pass
`deformLimitStress` (N), and its rest length stays within (1 − d)·L and (1 + dₑ)·L
of its rest length L at load (a negative value: no limit). It loads exactly as
JBeam mode loads those keys; the beam keeps the file's plastic coefficient (RoR's
`set_beam_defaults` field) and has no RoR break exception (§2). The showcase car's
crush members use it (`vehicle-authoring.md`, "Crush members"). The force law is
per beam on the CPU and the GPU, so such a beam in a `.truck` vehicle runs §3's
rule in both (`showcase_car_gpu` wall and side-pole counts match). Files without
the section load as before, bit for bit.

## 19. Deform groups [B]

`deformGroup` (string or table) names the groups a beam triggers; they drive visual
damage (glass, lamps; flexbody material swaps) and some powertrain damage.
`deformationTriggerRatio` r: "a shortening or lengthening greater than that amount
triggers it". Breaking a beam with a deform group triggers it too.

* **Trigger:** |L − L₀| > r·L₀ after a yield (L the rest length, L₀ the spawn rest
  length), or the beam breaks (on its own or by its group). **Choice:** without r only
  breaking triggers. Up to two deform groups per beam (warning past).
* **State:** per vehicle, per deform group, a word that is 0 until triggered, then 1
  forever. `ExplicitBeams::deform_triggered()` and the GPU's
  `ExplicitBeamsGpu::deform_triggered()` return it with the group names
  (`SoftVehicleDesc::deform_groups`). It is physics state only; the visual layer reads
  it. It has no effect on the beam.
* GPU: `gGroups` at `deformBase` + the vehicle's offset; set with an atomic max where
  a beam yields or breaks.

## 20. ANISOTROPIC beams (kind `JbeamAniso`) [A]

`beamSpring`/`beamDamp` "used only while the beam is compressed"; `springExpansion`,
`dampExpansion` "when the beam is stretched past its spawned length"; `transitionZone`
z "a fraction of spawned length" over which the rates move from the compression to the
expansion values (default 0: instant); `beamLongBound` b (default FLT_MAX) breaks it at
(1 + b)·ℓ₀, `beamLongExtent` (m) overrides b.

* The switch is at the spawned length S (2026-11-05; before, at the rest length L,
  so a precompressed beam rested on its expansion branch). s = ℓ − S. s ≤ 0: (k, c).
  s ≥ z·S: (kₑ, cₑ). Between: σ = s/(z·S) and k = k + (kₑ − k)·σ, c likewise
  (**Choice**: linear; the docs say "ramp"). The force stays −k·(ℓ − L): it is
  continuous for z > 0 (σ = 0 at S); with z = 0 and precompression it steps at S,
  as the docs' "instant" switch says. **Choice:** S = L + (ℓ₀ − L_target), so it
  moves with yield as the rest length does, and during the precompression ramp
  (§23) the ramp still to come is added back (S = ℓ₀ throughout). Without
  precompression S is L, bit for bit (E2 .y = ℓ₀ − L_target = 0). Missing
  `springExpansion` or `dampExpansion` take `beamSpring`/`beamDamp` (**Choice**).
* Effect: coast-down C_rr 0.053 → 0.050 (supercar), 0.079 → 0.074 (Astral R):
  not the rolling loss (§31).
* The break length is in metres from the spawn length ℓ₀ (as SUPPORT's, §4).
* Yield, deform limits and strength as NORMAL (§3), with the k of this step's side.

## 21. PRESSURED beams (kind `JbeamPressured`) [P]

A virtual air cylinder whose "gas volume depends only on the beam's length";
"halving the length doubles the pressure"; `volumeCoef` c "scales the virtual
cylinder length relative to the beam length. Larger values make pressure rise
earlier. Values above 1 eventually reach infinite pressure. Values below 1 never
reach infinity. 0 means pressure never changes"; force "from pressure acting on
`surface`" A (m², default 1); "in extension, the total force combines the beamSpring
force with the pressure force".

* **Volume** V(ℓ) ∝ ℓ₀ − c·(ℓ₀ − ℓ) (the only linear law with all four documented
  properties; ℓ₀ the spawn length). Isothermal: p_abs = p_abs,0 · ℓ₀ / (ℓ₀ − c·(ℓ₀ − ℓ)).
* **Pressures. Choice:** `pressure` (Pa) and `pressurePSI` (× 6894.757 Pa) are gauge
  (the docs: "ambient pressure is not included"; the PSI form "adds a fixed 101325
  Pa", i.e. converts to absolute); p_abs,0 = 101325 Pa + gauge. The push is the gauge
  pressure times the area: F_p = (p_abs − 101325)·A pushing the ends apart. A
  denominator at or below 1 % of ℓ₀ is held there (the cylinder bottoms out).
* `pressureLimit` (`pressureLimitPSI`): the gauge pressure never passes it.
  `maxPressure` (`maxPressurePSI`): the beam breaks when the gauge pressure passes it.
* **Spring. Choice:** the beam's spring acts only in extension (e > 0), its damping
  always: F = F_p − k·max(e, 0) − c·u. Strength applies to |F| (no yield: a cylinder is
  not plastic).

## 22. Rebound and velocity-split damping (BOUNDED) [BD]

"Compression uses the base value, and expansion uses the rebound value." Per side,
with u the extension rate (u > 0 expanding):

| | compression (u < 0) | expansion (u > 0) |
|---|---|---|
| slow | `beamDamp` c | `beamDampRebound` (default c) |
| fast | `beamDampFast` (default c) | `beamDampReboundFast` (default the rebound slow value) |
| split | `beamDampVelocitySplit` v_s (default ∞) | `beamDampVelocitySplitRebound` (default v_s) |
| limit | `beamLimitDamp` | `beamLimitDampRebound` (default `beamLimitDamp`) |

The slow/fast law per side is §5's (continuous at the split). A bounded beam that
sets none of the rebound fields is §5's beam unchanged. NORMAL beams: the docs give
rebound only for bounded beams; a NORMAL beam's rebound fields are ignored (warning).

## 23. `beamPrecompressionTime`, `beamLongExtent`

* **Precompression time** [B]: "time taken for precompressed beams to reach their
  requested length on spawn". The rest length starts at the spawn distance ℓ₀ and
  moves by (L_target − ℓ₀)/N each step for N = round(T / 0.5 ms) steps (**Choice**:
  linear), then stays; a yield in the ramp moves it as usual on top. L₀ (the spawn rest
  length the deform limits and triggers measure from) is L_target.
* **`beamLongExtent`** [S], [A]: the support's (or anisotropic beam's) absolute break
  length in metres; it overrides `beamLongBound`.

## 24. LBEAMs (kind `JbeamLbeam`) [L]

id1 and id2 are the beam's ends, id3 the vertex; beams id1–id3 and id2–id3 must exist;
"any change in angle between [id1,id3] and [id2,id3] is resisted by the L-Beam from
[id1,id2]", and it "applies no force when only the L-beam's own length changes" (of
an arm).

* **Choice (units N/m, as `beamSpring`):** the energy U = ½k·(ℓ − L)², ℓ = |x₁ − x₂|
  and L its rest length at the arms' current lengths, a = |x₁ − x₃|, b = |x₂ − x₃|,
  L = √(a² + b² − 2ab·cos θ₀). The spring is −∇U on all three nodes: −k(ℓ − L) along
  id1–id2 (the "beam from [id1,id2]") plus k(ℓ − L)·∂L/∂a along the arm x₁ − x₃ on id1,
  k(ℓ − L)·∂L/∂b along x₂ − x₃ on id2, and their negative sum on id3
  (∂L/∂a = (a − b·cos θ₀)/L). A pure stretch of an arm keeps ℓ − L (no force), an
  angle change does not. Momentum, angular momentum and energy are conserved.
* **Before (2026-11-03 to 2026-11-05)** only the id1–id2 pair force was applied. It
  is not the gradient of any energy (a follower force: it does work round a closed
  path). The FMVSS 214 barrier's honeycomb-shear LBEAMs flew its face apart in free
  flight (a bumper beam stretched 1.1 m within 0.25 s) and the barrier passed through
  the car with 8 contact hits: "LBEAMs kill vehicle contact". Contact itself was not
  at fault. Test: `explicit_contact_gpu.rs`, a sedan into the pressure-wheel fixture
  car (LBEAM tyres) pushes it, CPU and GPU; the unit test checks −∇U by central
  differences.
* The damper is a beam's: −c·u along id1–id2, u its extension rate (a pair dashpot,
  always dissipative). `dampCutoffHz` applies ([B] lists L-beams).
* θ₀ at spawn: the spawn angle (precompression scales L: θ₀ from p·ℓ₀).
* **Deform. Choice:** past `beamDeform` the rest angle moves so |k·(ℓ − L)| stays at
  the deform force (perfect plasticity); past `beamStrength` it breaks.
* **Records:** the LBEAM (kind 14; id3 in E1 .x) and a second record x₃–x₃
  (`JbeamLbeam2`, kind 20) whose first slot carries the vertex's force on the GPU; it
  has no force of its own (as §26's and §31's second records).
* Effect on the pressure-wheel fixture car at rest: it now rolls to and fro inside
  one flat of its 12-ray tyres (about 1.3 Hz, up to 44 J at the turns' middle (37 J
  since the L-beam layout of §25, 2026-11-05), the
  body over 0.13 m of a 0.17 m flat), weakly damped; the follower force had damped
  it to about 1 J. **Open**; the test checks the exact bounds (no energy above the
  2 cm drop's, never off its flat). Coast-down C_rr is unchanged (0.052 / 0.071).
* The `.truck` path still drops LBEAMs (RoR has none).

## 25. Pressure wheels (JBeam mode, opt-in) [WH]

The docs give the parts, not the layout: hub nodes on the axle "through fairly rigid
standard beams", tread nodes "on the tire's outer layer" joined to the hub "with
several beams", "a pressure group uses both hub and tire nodes", `numRays` "spokes per
side, must be even" (default 10), `radius` (default 0.5), `hubRadius` (default 0.65 ×
radius), `wheelOffset`, `tireWidth`/`hubWidth` (default the axle-node distance), the
beam sets and their flags, `pressurePSI`, `maxPressurePSI` (default 10). **Choice:**
the layout below, Rubble's own.

* **Frame.** Axle nodes n₁ → n₂, axis a = unit(n₂ − n₁), centre c = midpoint +
  `wheelOffset`·a; reference direction r₀ = the component of the arm node's offset (or
  of +y) normal to a; rays at angles 2πi/N (i = 0 … N − 1) about a.
* **Nodes.** Hub rings k = 0 (toward n₁), 1 at c + (k − ½)·hubWidth·a, radius
  hubRadius; tyre rings k = 0, 1 at ±½ tireWidth, radius `radius` (rings of one
  side at the same angles). Masses `hubNodeWeight` / `nodeWeight` (`hubWeight`,
  `tireWeight` divided over their 2N nodes override them). Tyre nodes collide with the
  ground and statics with their `frictionCoef`; no self-collision; hub nodes collide.
* **Settings across parts.** BeamNG splits a wheel over parts (a part with the
  rows, rim and tyre parts with modifiers only). **Choice:** a row takes the
  pressureWheels modifiers of every part of each slot subtree around its own part,
  from the whole vehicle down to its own part (nearer subtrees override; within one,
  assembly order, later wins), then its own part's and its own options.
* **Beams** (each set's `Spring`, `Damp`, `Deform`, `Strength`, `DampCutoffHz`,
  `Precompression`; **Choice** for missing values: no deform, no break, and for the
  hub sets the stiffest spring the step integrates with margin on the hub nodes,
  k = 0.04·m/h² (ω·h = 0.2 per beam) with damping 0.2·√(k·m); a reinforcement set
  without Spring or Damp takes its base set's (`wheelTreadReinfBeam` the tread beam's,
  `wheelPeripheryReinfBeam` the periphery beam's, `wheelSideReinfBeam` and
  `wheelReinfBeam` the side beam's; 2026-11-05: before, the 4.3 MN/m default, which
  put the forklift's 1.1 kg tread nodes at ω·h = 2.12); other sets 4.3 MN/m and
  580 N·s/m):
  * `hubBeam`: hub(k, i) to the other side's axle node; `hubSideBeam`: to its own
    side's axle node (radial spokes);
  * `hubTreadBeam`: hub(0, i)–hub(1, i) and hub(0, i)–hub(1, i + 1);
  * `hubPeripheryBeam`: hub(k, i)–hub(k, i + 1);
  * `wheelSideBeam` (ANISOTROPIC with `wheelSideBeamSpringExpansion`,
    `…DampExpansion`, `wheelSideTransitionZone`): tyre(k, i)–hub(k, i);
  * `wheelSideReinfBeam` (`enableTireSideReinfBeams`, anisotropic likewise):
    tyre(k, i)–hub(k, i ± 1);
  * `wheelReinfBeam` (`enableTireReinfBeams`, unless L-beams): tyre(k, i)–hub(1 − k, i);
  * L-beams (`enableTireLbeams`; `wheelLbeamSpring`… else `wheelReinfBeam`'s values,
    [WH]: `wheelReinfBeam*` are the "Spring, Damp, Deform, Strength, DampCutoffHz,
    Precompression of diagonal tire reinforcement beams, or L-beams (diagonal)"; L-beams
    are "similar to TireReinfBeams" and "If enabled, TireReinfBeams will be disabled";
    else the side beam's; its `DampCutoffHz` and `Precompression` too) (2026-11-05:
    before, the side beam's: on the forklift's tyres, three L-beams per tread node with
    the side beam's 1,150 N·s/m gave c·h/m = 2.6 on 1.1 kg tread nodes and the tyres
    diverged at the Nyquist rate): one per tread node on the reinforcement diagonal,
    vertex tyre(k, i) with ends hub(k, i) and tyre(1 − k, i) (2026-11-05: before, two
    more per tread node, hub(k, i) with tyre(k, i ± 1), a Rubble Choice the docs do not
    have, and precompression 1; they carried half the supercar's rolling loss, §31);
  * `wheelTreadBeam`: tyre(0, i)–tyre(1, i); `wheelTreadReinfBeam`
    (`enableTreadReinfBeams`): tyre(0, i)–tyre(1, i + 1);
  * `wheelPeripheryBeam`: tyre(k, i)–tyre(k, i + 1); `wheelPeripheryReinfBeam`
    (`enableTirePeripheryReinfBeams`): tyre(k, i)–tyre(k, i + 2);
  * `tireSupportBeam` (`enableTireSupportBeams`): SUPPORT from its own side's axle node
    to tyre(k, i).
* **Pressure group (§26)** over the closed surface: both sidewalls (hub(k, ·)–tyre(k, ·)
  bands), the tread band and the hub band, 8N triangles. Gauge `pressurePSI`; the rise
  above it is capped at `maxPressurePSI` (**Choice**: the docs' "stop internal pressure
  from rising too high"). Tyre triangles are not collision triangles
  (`triangleCollision` default false; when true they are added as collision
  triangles).
* **Rolling.** With `roll` the wheel's nodes spawn rolling at the vehicle's speed
  (as the soft tyres' `roll_tyres`).
* Not simulated: brakes and drive torque (no powertrain), `conicityFactor`, hubcaps,
  `nodeS`, tyre friction models (**Choice:** a tyre node's friction is the model's
  `noLoadCoef` when given, else `frictionCoef`; BeamNG's `frictionCoef` multiplies
  its ground model's, which Rubble's contact model has not),
  `hubWeightGainRatio` (warnings).
* Selected by `JbeamWheels::Pressure` (`ExplicitBeams::from_desc_jbeam_with`); the
  default stays the `.truck` path's soft tyres (§12), and RoR content keeps its tyres.

## 26. Pressure groups (kinds `JbeamPressureGroup`, `JbeamPressureTri`)

A closed triangle surface with a gas inside: V = Σ (x₀ · (x₁ × x₂))/6 over its
triangles (outward winding), isothermal p_abs = p_abs,0·V₀/V (V held at 1 % of V₀ or
more), gauge p = p_abs − 101325 Pa capped as the owner says; each triangle pushes its
three nodes outward with p·(x₁ − x₀) × (x₂ − x₀)/6 each (one third of p·area·n̂ each).
On a closed surface the pushes sum to zero (momentum conserved).

* **Records.** A group record (no force; its nodes a = b = its first triangle's first
  node) then two records per triangle (nodes x₀–x₁ carrying their forces, then x₂–x₂
  carrying x₂'s), in beam order: the node sums are the beam sums of the solver, in
  the same order on CPU and GPU.
* **Order. Choice:** the group record computes V and p in a step's beam pass from that
  step's positions; the triangles use the p of the previous step (state double
  buffered by step parity; at spawn p = p₀). One step (0.5 ms) of lag keeps the
  group's sum and its triangles in one pass on the GPU, with no new barrier.
* Punctures are not simulated: the group keeps its gas when beams of its surface
  break (a GPU beam phase cannot read this step's breaks without a race).

## 27. Rails and slidenodes [R]

The explicit solver's slide nodes (`explicit-beams.md`, the `.truck` path's
`slidenodes`, the GPU's slide pass in the EB_AERO modules) take the JBeam rows:
`links:` (a polyline of nodes; its segments the beams between consecutive link nodes,
a zero-stiffness beam is added where none exists), `looped`, `capped`; per slide node
`railName`, `attached` (default true), `tolerance` (m, default 0), `spring` (N/m,
default 4.3 MN/m), `strength` (N, default ∞: detaches past it), `capStrength`.
**Choice:** the solver's slide-node force law, unchanged (the node pulled toward
its nearest point on its current segment once past the tolerance, the opposite force
on the segment's ends by their weights, one segment hop per step, `strength` its break
force). The rail's ends always hold the node (the solver's segment clamp): `capped`
and `capStrength` are not distinguished and an uncapped end does not let the node run
off (warning when a rail is uncapped or sets `capStrength`). `fixToRail` (default true)
moves the node onto its rail at spawn (to within its tolerance) [R]. (The Astral R's
strut masses sit off their rails in the file: without it they were thrown off at
2.7 m/s in the first step.)

## 28. Two limits of the fixed explicit step

Community content carries values the 0.5 ms explicit step cannot integrate as
written; BeamNG runs them. Both rules below change nothing for a beam or node inside
the step's limits, apply in JBeam mode only, and the spawn warns with their counts.

* **Stiff dampers.** A beam whose damping gives c·h/μ > 1 (μ the reduced mass of its
  two nodes; explicit damping diverges past 2) takes the backward-Euler damping of
  that pair, c/(1 + c·h/μ), for its slow, fast, limit and rebound rates. (The
  Monster supercar has two such beams: 100 kN·s/m on 7 and 10 kg nodes.)
* **Tyre friction.** On a pressure-wheel node (`NODE_FRICTION_LIMIT`) the ground's
  tangential force at most stops the node's slip in one step: the solver's friction
  law's viscous branch, on light tread nodes carrying a tyre's pressure load, would
  otherwise reverse the slip every step and grow.
* **Tyre ground push** (2026-11-05). **Choice** (BeamNG documents no node–ground
  collision response: its ground models [GM] give friction, roughness, depth and
  fluid parameters, nothing on how a node is pushed out; [N] gives only the
  self-collision distance). On a pressure-wheel node the ground's normal
  push lets the node rise at most half its depth p in one step (velocity out of the
  ground ≤ 0.5·p/h). The solver's contact restores 0.2 of the depth per step as a
  velocity the node keeps; a 0.12 kg tread node under the 5 kN of its wheel's load
  then left the ground every 8 steps, fell back 9 mm in one step at 17 m/s, and the
  friction of those landings walked the Monster supercar sideways 0.15 m/s at rest
  (0.5 m in 4 s, with either damping law). At most half keeps the node in contact.
  Rest creep, `tests/explicit_jbeam_ext.rs` `creep` (exact physics: no net
  horizontal force on a car at rest, so a straight-line fit of its centre of mass
  over 8 s after 2 s may move no more than the sway around it, plus 1 mm):

  | | drift / sway before | after |
  |---|---|---|
  | Monster supercar | 214 / 156 mm (**creeps**) | 1.5 / 45 mm |
  | Astral R | — | 1.6 / 24 mm |
  | fixture car (`PART2_CAR`, 4 s) | 1.5 / 9.5 mm | 23 / 60 mm |

  Factors measured (supercar drift over 8 s, Astral R drift, fixture-car KE at
  4 s): none (before) 214 mm (creeps), —, 0.25 J; 0.2 (the solver's own
  fraction) 34 mm, 0.7 mm, 11.5 J; 0.3: 7.7, 2.2 mm, 16.9 J; 0.4: 1.4, 15 mm,
  0.92 J; **0.5: 1.5, 1.6 mm, 1.0 J**; 0.6: 14, 8.3 mm, 2.0 J; 0.7: 1.1, 6.7 mm,
  5.7 J; 0.8: 68, 15 mm, 10.8 J; 0.9: 13, 1.0 mm, 4.3 J; 1.0 (back to the
  surface, where the node then loses its support) 53, 12 mm, 9.9 J. Every factor
  stops the creep test failing on the supercar except 0.8 (marginal); none keeps
  the fixture car's 0.25 J. The fixture car's residual is rocking on its 12-ray
  tyres' flats, ±3 cm fore and aft at about 1 Hz on free-rolling wheels (no brakes
  are simulated); before, the bouncing tread nodes damped it (and walked the
  supercar). Its KE at 4 s swings between 0.9 and 17 J with the factor (the
  rocking's phase), its sway is 4–7 cm for every factor. **Open** (a regression
  from 0.25 J): a rolling-resistance or brake model would damp it; neither is in
  scope (no powertrain).

## 29. Acceptance vehicles (CPU reference, 2026-11-05)

`examples/jbeam_accept.rs` (JBeam mode, pressure wheels, drag off, lowest node 2 cm up;
then a rigid wall at 56 km/h). This first table is with the first §6 law (bilinear);
§6 gives every vehicle's numbers with the RC law:

| vehicle | nodes / beams | at rest, 4 s | 56 km/h wall, 1.5 s |
|---|---|---|---|
| Monster supercar (GPL-3.0) | 423 / 1,838 | 0 broken, KE 315 J falling to 56 J at 10 s (its spoiler and body flex) | 0 broken, 37 bent, KE 237 → 7.5 kJ, no NaN |
| AgentY legocar | 1,013 / 7,940 | 0 broken, 6 bent, KE 79 J | 540 broken (its bricks part), KE 179 → 4.5 kJ, peak 22 m/s, no NaN |
| Astral R (CC BY-NC-SA 4.0) | 820 / 6,342 | **unstable**: its rear subframe node `rx1r` diverges at 50 ms | not run |

**The Astral R** (found 2026-11-05, fixed the same day). Its highest spring mode is
ω·h = 1.77 (the BX's 1.75, the supercar's 1.72: inside the explicit step's limit of 2),
and its stiff rear control arms (12–15 MN/m, 1.2–1.5 kN·s/m) carry `dampCutoffHz` 500.
The first §6 law, a bilinear low-pass, anti-damps every mode above ω·h = √2 at that
cut-off (§6's analysis), so the arms' mode grew from rest. With §6's RC law, and
§10.1 for its rear bumper's centre node lying on a diffuser triangle, it rests and
crashes (wrc17_tarmac_balanced.pc, CPU):

| at rest, 4 s | 56 km/h wall, 1.5 s |
|---|---|
| 0 broken, 0 bent, KE 15 J (12 J at 10 s; tread nodes of its pressure wheels at ≤ 4 m/s), centre of mass 3.3 cm from 1 s to 4 s | 213 broken, 267 bent, 3 of 14 deform groups, 32 of 362 triangles dead; KE 252 kJ → 24.7 kJ, no NaN |

Its JBeam has no meshes (the author's repository leaves the `.dae` files out), so it
does not count for B3's "with flexbodies".

**On the GPU** (`engine-webgpu/tests/explicit_jbeam_community_gpu.rs`, opt-in
`RUBBLE_BEAMNG_MODS`, `RUBBLE_JBEAM_DIR`; RC damping law and §28's ground push;
M3 Max, 2026-11-05). Rest 4 s, then a rigid wall at 56 km/h for 1.5 s; CPU / GPU:

| vehicle | rest: broken, KE | rest: GPU − CPU positions | wall: broken / bent | wall: KE left (of the impact's) |
|---|---|---|---|---|
| Monster supercar | 0 / 0, 34.3 / 34.3 J | 0.00 mm (bit for bit) | 0 / 34, 0 / 34 | 13.8 / 13.8 kJ (237 kJ) |
| Astral R | 0 / 0, 8.3 / 11.1 J | rms 10 mm, max 19 mm | 251 / 266, 218 / 273 | 22.8 / 19.4 kJ (245 kJ) |
| AgentY legocar | 0 / 0, 85 / 58 J | rms 64 mm, max 153 mm | 486 / 11, 527 / 11 | 1.9 / 1.5 kJ (172 kJ) |

The supercar has no self-collision nodes and matches bit for bit; the GPU's
self-collision forces are fixed point (§14), so the other two part ways at rest
(their parked chatter) and in the crash (chaotic: counts within 25 %, the test's
rule). The supercar is a stiff space frame: it bends 34 beams, breaks none, and
rebounds off the wall (its centre of mass rises 0.6 m as it pitches).

**TPU Medium Forklift** (RLS Career Overhaul, AGPL-3.0; self-contained, own
pressure wheels and `.dae` flexbodies; `FL45D Triplex.pc`, 693 nodes, 7.9 t; CPU,
2026-11-05). It first diverged in two ways, both §25 layout choices that differed from
the docs, now fixed (§25): its tyres' reinforcement sets give no Spring, and took
4.3 MN/m (ω·h = 2.12 on 1.1 kg tread nodes; now their base sets', ω·h = 1.75 for the
whole vehicle); and its L-beams took the side beam's 1,150 N·s/m instead of
`wheelReinfBeam`'s 600 (c·h/m = 2.6 on a tread node: a Nyquist-rate divergence in
8 steps). Now: at rest 4 s KE 3.9 J; but its beacon light's break group (12 beams of
about 3.3 kN deform, 3.8 kN strength) breaks at 52 ms. Not the spawn drop (it breaks
spawned touching the ground too), and BeamNG documents no spawn settling beyond
`beamPrecompressionTime`: its four base nodes start 1.0 cm above the roll hoop's top
triangles (other node groups, so §10's exclusion does not apply, nor §10.1's, as the
hoop's triangles have a group), inside the 3 cm self-collision range. [CI]'s
push-out happens, and the solver's contact (a projection that cancels the node's
force and removes 20 % of the depth per step, not a spring) pushes them out through
their mounts, which yield and break. With self-collision off it rests with nothing
broken (KE 30 J at 2 s). Fixed by §10.1's spawn exclusions (12 pairs): it rests
4 s with nothing broken, KE 4.8 J. Into the wall at 56 km/h
it breaks nothing more and bends 5 beams: its forks and mast (85 MN/m between 50 kg
nodes) are nearly elastic, so it pitches up over its front axle (centre of mass
+1.65 m at 0.5 s) and rebounds at about 7 m/s. Energy ledger (KE + m·g·Δy of the
centre of mass): 965 kJ at impact, 420 kJ at 0.5 s, 374 kJ at 1.0 s, 367 kJ at
1.5 s: falling monotonically, so no energy appears; the rebound is the content's
stiffness, with no plastic forks. Whether that is plausible for a forklift is
open.

## 30. GPU layout of Part II and verification

* Params words 64–68: `extIndex` (gLists, per beam its extension + 1), `extBase`
  (gStatic, 4 records per extension, index fields globalised by the host),
  `extState` (gBeams, per extension (ramp steps left, pressure even, pressure odd,
  0)), `triBreak` (gStatic, per triangle its 4 beams), `deformBase` (gGroups after
  the two parities of group slots: the deform group words). The parameter stride is
  512 bytes (80 words).
* A beam with an extension has `TYPE_EXT` in its type byte; `beamUpdate` sends it
  to `jbeamExtUpdate` before `jbeamUpdate` (whose code is unchanged), `groupDrops` to
  `groupDropsExt` (only when a master broke this step), and its first beam phase
  after an own break calls `extPending` (several groups, §17). Breakable triangles
  carry bit 31 of their self-group word; `triangleGeo` and `vvTriangle` give a dead
  one an empty box; `tryContact` masks the bit. Pressure-wheel nodes carry
  `NODE_FRICTION_LIMIT` (32) for the ground contact (§28).
* Verification (2026-11-05, M3 Max): `explicit_beams_bitwise` (11 tests) bit for
  bit against the frozen reference after the change; `tests/explicit_jbeam_ext_gpu.rs`:
  the break-group ladders, the breaking panel (triangles and deform groups), the
  precompression ramp and the rail bit for bit against the CPU (0.0000 mm, equal
  broken counts and deform states); the pressure-wheel car within 2.2 mm over 2 s
  at rest (ground friction's float-level differences) and identical into a wall at
  56 km/h over 1 s. The same Slang compiles for the CUDA and Metal targets
  (`slangc -target cuda|metal`, with and without `EB_AERO`), and the Metal output
  compiles to AIR with `xcrun metal`; no CUDA toolchain or NVIDIA GPU here, so the
  CUDA build is a Slang type-check only.
* Cost (`explicit_beams_scaling jbeam=…:bx:rubble_body_only.pc`, 120 fps, 240 frames,
  3 trials, a busy machine: load 12–22 from other sessions). The BX body now runs
  Part II itself (704 beams with deform groups, breakable triangles), so this is the
  cost with the features in use. GPU ms per frame, before → after: idle 1.06 → 1.02
  (1 car), 0.94 → 1.11 (16), 2.26 → 2.45 (64); impact at 56 km/h 1.39 → 1.59,
  2.22 → 2.34, 5.23 → 5.61 (+7 % at 64 cars).

## 31. Triangle aerodynamics, and what the census leaves out (2026-11-05)

[T]: "The aero system in beamNG works by calculating drag on all triangles based
purely on the speed of the airflow, the surface area and the drag properties of the
triangles. The drag and lift forces are then applied on the adjacent nodes.
Triangles are not affected by other triangles"; lift follows "the angle of attack of
the triangle compared to the airflow" up to `stallAngle`, "at which point the lift
will start going down with added angle". Fields: `dragCoef` (default 100, "a
percentage of a flat plate of the same size"; "around 10 for most exposed body
panels"), `liftCoef` (default `dragCoef`), `stallAngle` (default 0.58 rad),
`skinDragCoef` (default 0; mud-tyre paddles).

* **Rule** (`explicit_jbeam_ext::aero_tri_force`, Slang `KIND_JATRI`): still air,
  the flow u = −(mean of the triangle's node velocities), A its area, n̂ its normal,
  α its angle of attack (sin α = |n̂·û|). Drag ½ρ·C·(dragCoef/100)·A·|u|²·sin α along
  û (the projected area); lift ½ρ·C·(liftCoef/100)·A·|u|²·C_L across û in the plane
  of û and n̂, toward the side the flow pushes the plate, C_L = sin α cos α up to the
  stall, sin α_s cos α past it (continuous, falling to zero at 90°); a negative
  `liftCoef` reverses the lift (the Monster supercar's spoilers use −1.8 and −2).
  Each node takes a third. **Choices:** ρ = 1.225 kg/m³ (ISA sea level); C = 1.28, a flat plate normal
  to the flow (Hoerner), for `dragCoef` 100; the flat plate's normal-force law for
  the angle dependence; the stalled law above.
* **Records:** two per triangle with a drag or lift coefficient, after the beams
  (kinds `JbeamAeroTri`, `JbeamAeroTri2`, extension records as §26's pressure
  triangles: E1 = (x₂, ½ρC·dragCoef/100, ½ρC·liftCoef/100, sin α_s)). Every desc
  triangle (collision or not: BeamNG's NONCOLLIDABLE triangles keep their aero)
  except pressure-wheel triangles (their `dragCoef` is not simulated).
* **On when the solver's drag is on** (`ExplicitBeamsSettings::drag`); in JBeam mode
  it then replaces the per-node air drag and turbulence (the `.truck` path's rule):
  BeamNG's air drag is the triangles'. Drag off, nothing changes (bit for bit).
* Broken triangles keep their aero (**Choice**; [T]'s "non-aero state" of a broken
  triangle would need §18's break lists on every triangle).
* `skinDragCoef` is not simulated (the census lists it when used).
* Checks: `tests/explicit_jbeam_ext.rs` (one plate: drag normal to the flow, drag and
  lift at 30°, the stalled lift at 60°, to 0.1 %); GPU against CPU in
  `tests/explicit_jbeam_ext_gpu.rs`. Coasting at 100 km/h the Astral R's triangles
  add 1.04 kN (C·A ≈ 2.2 m², its diffuser and wings' `dragCoef` 40–46 included).

**Census.** `jbeam::unsupported_features` lists only the physics JBeam mode does not
simulate. `jbeam::out_of_scope_features` lists the rest with why: the powertrain and
its torsion reactors (out of scope); hydros' `inputSource` and thrusters (controller
inputs; with no driver the input is zero, so a hydro holds its spawn length and a
thruster pushes nothing, which is what the solver does); `beamLimitRebound` (a key
BeamNG documents nowhere: its bounded beams' rebound limit is `beamLimitDampRebound`,
§22; unknown keys are ignored). For the B3 vehicles (the Monster supercar, the
Astral R, the legocar) `unsupported_features` is empty.

**The Monster supercar's coasting (found and fixed 2026-11-05).** It slowed at
37 m/s² and reversed because the acceptance harnesses drove it sideways: the file
models the car facing −X (`pack::forward_axis`), and `examples/jbeam_accept.rs` and
`explicit_jbeam_community_gpu.rs` spawned every car moving along +Z, so its wheels
were not spun (their axles lay along the motion) and it slid on its tyres. Both now
turn the car to face +Z first (the crash lab always did). Every supercar "56 km/h
wall" figure above dated before this note was a side impact; facing the wall it
breaks nothing, bends 52 beams and leaves at about 5 m/s (KE 24 kJ at 1.5 s of
237 kJ).

**Coast-down** (`tests/explicit_jbeam_ext.rs`, opt-in; 25 m/s, 0.5–1.5 s): the
momentum never grows, and with drag on the deceleration is at least the triangles'
own drag summed along the run (it is: supercar 11.4 kN against 10.9 kN, Astral R
2.29 kN against 0.66 kN plus rolling). Rolling resistance alone: C_rr = 0.053
(supercar) and 0.079 (Astral R), 4–6 times the documented 0.010–0.015 of passenger
tyres: the pressure tyres' tread and sidewall losses (12–16-ray polygons, the §28
ground rule) are too high. **Open.**

**Where the loss is (2026-11-05, `energy_ledger.rs` coasting, CPU, no air, 0.5–1.5 s;
per beam −Σ f·Δℓ − ΔU with the trapezoid of f over each step).** Supercar at 9 / 25
m/s: beams dissipate 6.0 / 28.7 kJ/s (ledger), of which sidewalls (tread to hub,
704 beams) 5.1 / 25.5, periphery 0.4 / 1.9, tread 0.05 / 0.2, hub 0.2 / 0.5, the
rest of the car 0.2 / 0.3 (the parts sum to the ledger within 1 %). Astral R:
sidewalls 6.7 / 33.9 of 7.5 / 38.0 (its "car" part does not close: its torsion and
pressure records' scalar force is not their work). Measured causes, each a
diagnostic (`COAST_DIAG`, not a model change):
* the sidewalls' dampers (`wheelSideBeamDamp`, `…DampExpansion` set to 0): the
  loss falls by 12–15 % only (supercar 25.3 kJ/s at 25 m/s);
* yield: no rest length moves over the window;
* the polygon (`numRays` doubled, 24/32 rays): the loss is unchanged (33.0 kJ/s);
* the sidewalls' stiff expansion branch (expansion rates set to the compression
  ones): the loss doubles (61.2 kJ/s), so it follows the sidewalls' deformation.
So the rolling loss is neither the damping law, nor yield, nor the flats: with
their dampers off, the conservative sidewall springs still lose energy each
revolution in proportion to their deflection. **Open (next):** the tread nodes'
landing through the §28 ground rule (it removes 80 % of a landing node's normal
speed, an inelastic impact whose energy the sidewalls' rebound then carries), and
§20's switch point (the docs switch to the expansion rates "when the beam is
stretched past its spawned length", ℓ₀; Rubble switches at its precompressed rest
length L = p·ℓ₀, so a precompressed sidewall rests on its expansion branch).

**The two suspects tested, and the sink found (2026-11-05, same harness; loss =
start − end mechanical energy over the 1 s window, kJ/s, supercar / Astral R at 25 m/s).**
* *The conclusion above was wrong.* `nodamp=wheelSideBeam` zeroes the sidewall
  beams' dampers only; the "sidewall" part (tread to hub) also holds the L-beams
  (`enableTireLbeams`, both cars), which take `wheelReinfBeamDamp`, 135–227 N·s/m
  against the sidewalls' 65–120. Zeroing each damper set in turn: L-beams 25.1 →
  11.3 / 36.4 → 19.4, sidewall beams → 22.4 / 30.9, periphery → 24.1 / 35.0, tread →
  25.1 / 36.4. All the sidewall-part dampers off: the Astral R loses 9.5 kJ/s (the
  supercar's tyres go unstable without them). The springs are not the sink: the
  dampers are, the L-beams' first.
* *§20's switch point* (fixed to the docs' spawned length): C_rr 0.053 → 0.050 /
  0.079 → 0.074.
* *The L-beam layout* (fixed to the docs, §25): Rubble had three L-beams per tread
  node, two of them on hub(k, i)–tyre(k, i ± 1) chords that the docs do not have
  ([WH]: L-beams are "(diagonal)", in place of the diagonal reinforcement beams), and
  no precompression. With one diagonal L-beam per tread node and
  `wheelReinfBeamPrecompression`: loss 13.5 / 30.9 kJ/s; **C_rr 0.024 (supercar,
  was 0.050) and 0.063 (Astral R, was 0.074)**.
* *Touchdowns through the ground rule* (a node below the ground that was above it
  the step before; its normal kinetic energy ½m·v_y², all of which the 80 %-per-step
  rule removes): supercar 470 a second (12.8 per tyre revolution), 3.3 kJ/s, 7 J
  each; Astral R 1,444 (29.5), 8.1 kJ/s, 5.6 J each: a quarter of the loss at 25 m/s,
  3–6 % at 9 m/s (0.17 / 0.33 kJ/s). Not the sink. It does not shrink with the
  polygon (`rays=2`: unchanged; half the mass lands as often again): the tread lands
  at about v·sin θ of its patch-entry angle. BeamNG documents no node–ground normal
  response ([GM], [N]), so the rule stays; a restitution would be a modelling Choice
  with nothing measured behind it.
* What is left of the Astral R's: its sidewall beams' dampers carry 13 kJ/s of
  30.9 (`nodamp=wheelSideBeam` 17.9), the L-beams' 3 (27.9), the touchdowns 8.
  **Open:** 4–6 times the documented C_rr for the Astral R, 2 times for the supercar.
* The fixture car's rest rocking (§24, §28) is the same mechanism with less loss:
  peak kinetic energy 1–4 s 37.3 J (was 44 J), 17.7 J at 4 s; the polygon flats'
  well and free-rolling wheels, damped only by the tyres' rolling loss.

**The tread's landing, measured against geometry (2026-11-05, `energy_ledger.rs`
coasting, which now also reports each tyre's loaded deflection and tread mass).**
Astral R: 2,028 kg, tread 5.09 kg per tyre (32 nodes of 0.159 kg), loaded deflection
δ = 8–11 mm at 9 m/s and 10–12 mm at 25 m/s, R = 0.325 m. A point-mass tread node
rotating at radius R reaches the flattened patch at angle θ, cos θ = 1 − δ/R
(13.5° at 9 m/s, 14.9° at 25 m/s), moving towards the ground at v·sin θ (2.1 and
6.4 m/s). It lands with ½m(v·sin θ)² = 0.35 J and 3.3 J; measured 0.40 J and 5.6 J
(1,444 landings a second at 25 m/s, 29.5 per tyre revolution: almost every tread node,
every revolution). Per tyre that is a force of about M_tread·v²·sin²θ / (4πR): 205 N
for the car at 25 m/s (5.1 kJ/s of the 8.1 measured; the rest is the node's radial
ringing as it leaves the patch, at 196 g centripetal). It does not depend on the
polygon (the rays set the node mass, not the total tread mass or θ), as `rays=2`
showed. The landing's impulse is what the sidewall and L-beam dampers then take
(13 and 3 kJ/s). The loss scales with the total tread mass and the deflection, and
roughly with v² (C_rr 0.025 at 9 m/s, 0.063 at 25 m/s; a real tyre's is close to
constant). **So the remaining sink is the lumped tread's inelastic landing at its
patch-entry angle** (§28's node–ground rule on a belt with no bending stiffness ahead
of the patch, unlike a real tyre's belt), not a mis-set beam. Every input is the
file's own (node weights, deflection from its pressure and springs). **Not fixed:**
removing it needs a modelling Choice with nothing measured behind it: a ground
restitution, or tread bending ahead of the patch the file does not author. This is the
user's decision (AGENTS.md: no lossy or invented model on our own initiative).

**Drag area check** (`jbeam_accept … aero=`: Σ 1.28·(dragCoef/100)·A·|n̂·f̂|
over the desc's triangles, f̂ the forward axis; one-sided areas, each triangle once,
SI units, the default 100 only where a row sets none). The reading is the docs':
the values are the files'. The supercar's 61.6 m² is 96 % two groups whose
`dragCoef` comes from its own variables, `$dragCoeffc` (front body, default 1000,
range 0–10,000, "Front Drag coeficient") and `$dragCoef` on the spoiler (10,000;
its comment says 120); its 68 body triangles at `dragCoef` 10 give 0.45 m², a real
supercar's order (C_d·A ≈ 0.6–0.8 m²). The Astral R: 1.76 m², of which 0.98 m² its
wings and aero devices (`dragCoef` 100–160), 0.5 m² its body panels (6–12): a rally
car with its aero kit at C_d·A ≈ 1.0–1.2 m², so ours is about 1.5× (both faces of
each triangle drag, as [T] says, "Triangles are not affected by other triangles").
No triangle takes the default 100 in either car except where the author left it.
## 32. The contact-force log and `rubble_contact_breaks` (2026-11-05)

**Contact force (Rubble, both solvers).** A node's contact force in a step is the
sum of the magnitudes of every contact force on it that step: the node pass's
(ground, boxes, static shapes, rigid bodies: the latter's sum per node),
self-collision's and vehicle contact's, as the contacter and as a struck
triangle's node (its share w·|F|). **Choice:** magnitudes summed (a pinched node
counts both pushes); fixed point at 1/32 N, as the kernel's accumulators. Off by
default (`ExplicitBeams::contact_force`, `enable_contact_force`): nothing changes for
a vehicle without it. A step's sum is complete after the vehicle-contact pass; the
next node pass moves it to `last` (and a running peak) and restarts it.
Readback: CPU `contact_forces()` (last step, N) and `take_contact_force_peak()`;
GPU `ExplicitBeamsGpu::contact_forces()` (per vehicle: last, peak, contact-break
states) and `clear_contact_force_peaks()`. GPU: per-vehicle flag bit 8
(`FLAG_CFORCE`, its nodes take the shapes path of the node phase); params words
`cfAcc` (gAcc, per node sum/last/peak), `cfNode` (gLists, per node break group and
force), `cfPend` (gAcc, per group slot its state).

**`rubble_contact_breaks`** (a `.truck` section, **Rubble extension, not RoR**): lines
`group, force_N, nodes…`. Break group `group` goes off when the contact force on any
listed node exceeds `force_N` in a step. Forward only: the step's force is tested at
the next node pass and the group breaks in that step's beam pass, as if set off by
its first beam (beam 0 keeps that step's force; its other beams do not), once. A
group must be some beam's own break group and a node may be in one line only; the
loader fails with a clear error otherwise. Use: windscreen and glass direct-impact
breaking (a pane's nodes and its break group), and the visual layer's paint-wear
pressure (the log alone). Tests: `contact_breaks.rs` (CPU: a resting node's mean
contact force is its weight to 0.05 N; a 0.5 m drop breaks the group the step after
its force passes 50 N, never at 10 MN; the lines are read and checked),
`explicit_contact_gpu.rs` `contact_breaks_step_like_the_cpu` (the same step on the
GPU, logs within 0.05 N + 0.1 %).

## 33. `rubble_exact_lengths`: cost, parity, effect (2026-11-05)

`rubble_exact_lengths` (a `.truck` line, **Rubble extension**) computes beam lengths
through an exact square root in place of RoR's fast inverse square root (worst
length error 0.175 %, `energy_ledger.rs`). Measured on the showcase sedan:
* **Cost:** none. 32 low-resolution sedans at rest on the GPU (M3 Max), 1.816 ms a
  frame fast, 1.814 ms exact (`showcase_car_gpu.rs` `exact_lengths_cost`, ignored).
* **CPU–GPU parity: not bit for bit.** At rest for 2 s the fast path's GPU matches
  the CPU to 0.000 mm; the exact path differs by 1.4 mm rms (7.8 mm at most, a
  tyre node chattering on the ground). The kernel's `sdiv(1, ssqrt(q))` (an
  approximate root and reciprocal, each with one Newton correction) is not always
  the CPU's correctly rounded 1/√q, and the rest contact amplifies the last bit.
  Both stay within the showcase GPU tests' tolerances (`SC_EXACT=1`).
* **References** (`showcase_car.rs`, `SC_EXACT=1`, CPU): all 26 tests pass. Changes
  are within crash-to-crash scatter: 55 km/h wall residual crush max 497 → 503 mm,
  Campbell b1 29.4 → 29.9 s⁻¹ (120 km/h 27.0 → 28.3); FMVSS 214 peak 210 → 217 kN;
  FMVSS 216a 4.78 → 4.77 × weight; side pole intrusion 0.024 → 0.024 m; T-bone
  inside 9 → 5 of 1,141; rest energy 8.30 → 8.20 kJ.
* **Recommendation (before 2026-11-05's fix):** adopt it for the showcase car only
  once the kernel's exact inverse length is correctly rounded (bit parity with the
  CPU, as the fast path); the physics gain at today's references is small.

**One sequence on both sides (2026-11-05).** The exact path's length and inverse
length are now an explicitly specified sequence the CPU
(`explicit_beams::exact_length`) and the kernel (`sExactLength`) run identically:
y = the fast inverse square root's magic estimate and four Newton steps
y ← y·(1.5 − (½q·y)·y) (IEEE single products and differences in that order;
`exact_inv_sqrt`, within 1.65 ulp of 1/√q), and the length ℓ = q·y with one Newton
correction through its exact residual, ℓ ← fma(fma(−ℓ, ℓ, q), ½y, ℓ) (Rust's
`mul_add`, the kernel's `fma`: fused on both). No hardware square root or division:
the GPU backends promise neither correctly rounded. **Choice** over a correctly
rounded 1/√q. Measured: ℓ is √q correctly rounded for every 7th float of q from
10⁻⁶ to 10⁴ (40 million, `energy_ledger.rs`; the old ℓ = q·(1/√q) was within an
ulp), so a beam spawned at its length has e = 0 exactly; finite at q = 0 (the old
path gave a NaN length there). The inverse length (the damping rate and the push
direction) is y. Checked:
* y on wgpu (Metal backend) against the CPU, every float of q from 10⁻⁴ to 16
  (147 million): all bit for bit;
* the procedural sedan at rest, exact and fast, every node's position and velocity
  bit for bit after each of 2,400 steps, wgpu (`showcase_car_gpu.rs`
  `exact_lengths_step_bit_for_bit`) and Metal (`explicit_parity.rs`
  `metal_exact_lengths_step_bit_for_bit`);
* the showcase sedan at rest for 2 s: rms 1.4 → 0.25 mm (max 8.0 mm), **not yet bit
  for bit:** one node differs by one ulp of velocity at step 5, and the rest
  follow. Not the length (proved identical above), not self-collision (no contacts),
  not a subnormal; the car's rest lengths are the fast path's (`authoring`
  settles with it, §18.2), so with exact lengths its beams are strained at spawn
  (one yields at step 2) and it moves at rest where the fast path keeps it still.
  **Open:** which of the showcase car's paths (ropes, supports, limited springs, the
  yield it reaches) differs. For the showcase to adopt the exact path it should also
  author its rest lengths with it.
* Unchanged: the fast path (RoR's), and JBeam mode's exact lengths
  (`jbeam_force_filtered`, `jbeam_ext_force`; the kernel's `jbeamUpdate` and
  `jbeamExtUpdate`, which keep 1/√q and `sdiv(1, ssqrt(q))`, not bit for bit:
  a follow-up can move them to `exact_length`).
