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

**Yield follows the elastic force only (2026-11-06).** The yield test reads the
whole force F = −k·e − D(u), damping included, and moves the rest length by
δ = e ± (T/k)(1 − p). When damping alone carries |F| past T while the spring's own
force is below it, δ had the wrong sign: in compression the rest length grew to
ℓ + T/k (a spring storing T²/2k the step had not been given). On the Gavril Vertex's
engine mounts (k 100 N/m, deform about 2 kN) that put 20 kJ in each of two beams in
one step at the 56 km/h wall: the world ledger rose 37.6 kJ. **Fix:** a beam yields
only when δ moves its rest length toward its current length (δ < 0 in compression,
δ > 0 in tension), so plastic flow never stores energy; the same δ, gated (CPU
`jbeam_force_filtered`, `jbeam_yield`; kernel `jbeamUpdate`, `jbeamExtUpdate`), bit for
bit on the GPU. Every yield whose δ had the right sign is unchanged. The RoR profile's
`Plain` rule is RoR's and unchanged. Measured: the Vertex's wall ledger rise
37.6 kJ → 0.03 kJ (`jbeam_accept ledger=1`); `tension_yield_gpu` now judges each
beam's largest yield over the run (a beam flung apart yields in tension, then, its
Tc being low, back in compression on the rebound, which it no longer did by damping
alone).

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

### 7.1 Node couplers (`couplerTag` → `tag`, 2026-11-06) [C]

BeamNG's documented coupler system: a *coupler node* (`couplerTag` t) couples to a
*tag node* (`tag` t). "Active" nodes move toward each other and merge; a coupler node
"connects to one tag node at a time", and a tag node to one coupler node.
`couplerStartRadius` "also sets the coupler to Active after spawn, so nodes in range
connect immediately"; `couplerRadius` alone needs the coupler toggle (a driver
input). `couplerStrength` (N, default FLT_MAX): "if the two nodes have different
values, the lower one applies". `breakGroup` / `breakGroupType` are common node
arguments: the connection breaks when the group breaks, and a type-0 coupler breaking
breaks the group. `couplerLock` and `couplerWeld` only protect a coupler from the
toggle (no physics). `couplerLockRadius` (default 0.025 m) is where connected nodes
"are treated as one and snap together"; `couplerLatchSpeed` (default 0.3 m/s) the
approach speed while Active. This is how BeamNG content latches doors, bonnets, boot
lids, soft tops (the CCF, the Gavril Vertex and the ETK K-series use it; their
`couplerNodes` controller tables carry no `couplingStartRadius`).

* **Pairing (Choice for order):** coupler nodes with `couplerStartRadius` > 0 in node
  order, each to the nearest free tag node of its tag within that radius (lowest
  index on a tie). Others: none coupled (warnings count them: no start radius, no
  free tag node).
* **The coupling:** §7's zero-length spring–damper (the nodes "considered one"),
  the same `JbeamCoupler` record, strength the lower of the two nodes'. A pair an
  advanced controller already coupled stays one coupling (the lower strength).
* **Break groups:** the coupler node's `breakGroup` (a name or a table) then the tag
  node's, at most three (§17), of the coupler node's `breakGroupType` (the tag
  node's when the coupler node names none). A coupler in more than one group gets an
  extension record (`EXT_GROUPS`, §17) and its own breaks set its groups off by §17's
  pending rule; with one group, as §7.
* **Latching approach not simulated:** every pair in the three cars spawns at 0 m
  (within `couplerLockRadius`); a pair farther apart is coupled at spawn with a
  warning naming the largest gap.
* **Effect:** CCF `f4_sport_M` 4 node couplers (+2 advanced), Vertex 7, ETK `kr6` 7,
  the hood latches passive in both hinge groups.
* Tests: `explicit_jbeam.rs` `node_couplers_couple_at_spawn_and_break_by_strength_and_group`
  (`jbeam_fixture::NODE_COUPLERS`); GPU `explicit_jbeam_ext_gpu.rs`
  `node_couplers_step_bit_for_bit` (800 steps bit for bit through both kinds of
  break). Kernel: `jcouplerUpdate`, and a `KIND_JCOUPLER` arm in `jbeamExtUpdate`.

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
range (0.03 m; 0.025 m since §36's band) and contact rules are otherwise the solver's.

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
range (inside it, within the contact range of its plane (2.5 cm since §36; 3 cm
when measured below), and not excluded by §10's
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

**Leak, after anti-clip (2026-11-05, §36).** With anti-clip on, sedan1's T-bone leak
is unchanged within noise (12.0 vs 13.6 mean leaked nodes over five offsets). The
leaked nodes' crossings start at hits in front of the skin, with the side held: the
contact force per step is too weak against the closing skin's momentum, and a node
driven past CLIP_DEPTH (10 cm) is forgotten. Next: the contact force against the
closing skin (with the showcase car's heavier-node build), after the profile's
plasticity.

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

**Since §36:** a line may name a rope (`r`) beam, which becomes a tension-only JBeam
beam (`JbeamRope`), and may take a sixth field, `breakAtExpansion` (nonzero): the
beam tears the step its rest length would pass (1 + deformLimitExpansion)·L (stored
as a negative expansion bound), in tension only; a compressed beam is untouched.
Without the field the beam holds at the limit, elastic about it (as before).

### 18.4 Rubble's `.truck` extension: a tension-side yield force (2026-11-05)

**Rubble extension, not RoR or JBeam** (`truck.rs`, `ExplicitBeams::set_tension_yield`).
JBeam has one `beamDeform` for both directions; its documentation has no expansion-side
deform force (only `deformLimit` / `deformLimitExpansion`, which limit the plastic
change, and the anisotropic and bounded types, which split stiffness and damping, not
yield). A thin sheet buckles in compression (its edge's yield is Winter's effective
width, `vehicle-authoring.md`), but in tension it carries its material's yield. A
section `rubble_tension_yield` of lines `a, b, Tt` sets the deformable plain beam a–b's
(`Plain` or `JbeamPlain`) tension threshold to Tt newtons; its compression threshold
stays the file's deform threshold, and the gate is min(Tc, Tt, S). Nothing else
changes: the yield rule (§3, `explicit-beams.md` §2) already keeps Tc and Tt apart and
hardens each on its own side; spawn set them equal. No kernel change. A line naming no
deformable plain beam, or Tt ≤ 0, fails the load. Files without the section load as
before, bit for bit. Tests: `tension_yield.rs` (one line sets only its beam's Tt and
gate; a missing beam fails), engine-webgpu `tension_yield_gpu.rs` (beams with Tt ≠ Tc
stretched and squeezed past both, in the `.truck` and the JBeam rule: every node and
beam the same float as the CPU's for 400 steps). The showcase car's sheet edges use it.

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
* **The third node in the row's options** (2026-11-06): [L]'s examples write it
  `["a", "b", {"id3:": "c"}]`, with the header's node-column colon. Inline option
  keys now drop a trailing `:` (the same field as the header's). Before, the
  boathauler trailer's 20 suspension LBEAMs were skipped as "LBEAM without id3".

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
* **Parking brake (§25.1).** Not simulated otherwise: the service brake
  (`brakeTorque`) and drive torque (no powertrain), `conicityFactor`, hubcaps,
  `nodeS`, tyre friction models (**Choice:** a tyre node's friction is the model's
  `noLoadCoef` when given, else `frictionCoef`; BeamNG's `frictionCoef` multiplies
  its ground model's, which Rubble's contact model has not),
  `hubWeightGainRatio` (warnings).
* Selected by `JbeamWheels::Pressure` (`ExplicitBeams::from_desc_jbeam_with`); the
  default stays the `.truck` path's soft tyres (§12), and RoR content keeps its tyres.

### 25.1 Parking brake (`parkingTorque`; Rubble's own model)

A pressure wheel row whose `parkingTorque` T is positive and which names a
`nodeArm:` gets a brake (`WheelBrake`): a Coulomb torque between the hub's ring
nodes (the disc) and the arm node (the caliper's mount) about the axle. The
input `ExplicitBeams::parking_brake` u ∈ [0, 1] (0, released, by default) scales
T; with u = 0 or no brakes no pass runs, so content without them is unchanged
bit for bit.

Each step, after the node pass and the aero and command passes (forces from this
step's integrated state), per brake in file order (`brake_apply`):

* axis a = (node2 − node1)/|node2 − node1|, point c and velocity v_c the axle nodes'
  means; r⊥(p) = (p − c) − a((p − c)·a);
* the disc: I_w = Σ m |r⊥|², ω_w = Σ m a·(r⊥ × (v − v_c)) / I_w over the hub rings'
  2N nodes; the caliper: I_a = m_a |r⊥a|², ω_a = a·(r⊥a × (v_a − v_c)) / |r⊥a|²;
* τ = −sgn(ω_w − ω_a) min(uT, |ω_w − ω_a| I_eff / h), I_eff = I_w I_a / (I_w + I_a):
  stick removes at most the two bodies' relative spin in one step (a damper at
  c·h/I_eff = 1, inside the explicit bound of 2), slip is the torque T;
* forces: each disc node m (a × r⊥) τ/I_w, the arm (a × r⊥a)(−τ/|r⊥a|²), each axle
  node half the negative of their sum. The net force is zero, the torques about
  the axle are τ and −τ.

The tyre turns with the disc through its side beams. The service brake
(`brakeTorque`) has no input in Rubble and stays unsimulated. On raycast wheels
the drive's handbrake is the parking brake (`explicit_drive`).

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

### 29.1 Community cars: CCF, Gavril Vertex, ETK K-series (2026-11-06)

Three full community cars with flexbodies (B3), local only (`RUBBLE_JBEAM_COMMUNITY`,
the folder holding the mods; never committed): the 2020 Hirochi CCF (`thw_ccf2`,
`f4_sport_M.pc`), the Gavril Vertex NA2 (`innocent_gavril_vertex`), the 1994 ETK
K-series (`etkk_lucasbe`, `kr6.pc`). Their tyres, brake pads and bulbs are BeamNG's
shared `common` parts, not local: **wheel, brake and bulb slots are ignored**, and the
cars run on Rubble's raycast wheels (`JbeamWheels::Raycast`), or on their hubs
(pressure wheels with `hasTire` false) for the crash scenes. Census: no unsupported
physics feature (couplers §7.1 were the last); out of scope: the powertrain and
hydro inputs (§31).

`jbeam_accept` (CPU, raycast wheels, drag off), rest 4 s then a rigid wall at 56 km/h
for 1.5 s:

| car | nodes / beams | at rest 4 s | wall: broken / bent, deform groups, KE left (of the impact's) |
|---|---|---|---|
| CCF | 635 / 4,101 | 0 broken, 0 bent, KE 0.1 J, 0.13 m/s | 144 / 493, 11 of 33, 5.8 kJ (177 kJ) |
| Vertex | 519 / 3,407 | 0, 0, 0.0 J | 26 / 472, 13 of 41, 5.8 kJ (160 kJ) |
| ETK | 467 / 2,873 | 0, 0, 0.5 J, 0.15 m/s | 2 / 359, 6 of 18, 7.1 kJ (172 kJ) |

The CCF tears its front fascia off (its bumper break groups), and a piece lands well
to the side in the browser crash lab: **open** whether that is the content's light
fascia or too much energy in the tear.

**Browser (GPU) against the CPU reference** (2026-11-06, JBeam mode on raycast wheels
both; `community-check.mjs` against `jbeam_accept wheels=raycast`), the wall at
56 km/h after 1.5 s, broken / bent: ETK 2 / 369 against 2 / 359, CCF 142 / 494
against 144 / 493, Vertex 26 / 466 against 26 / 472. The lab's impact detection fires
at 570 ms (the run-up), not before. A GPU ETK that sheds fragments and ends as a heap
was the `.truck` path's (before the JBeam-mode default).

**Wheels coming off after the wall (2026-11-06): the visual, not the physics.** The
ETK breaks two beams at 56 km/h (a bounded transmission-to-differential beam and the
bonnet latch), the SUV 44 (headlights, its turbo intake); no hub, suspension or
break-group beam. Its hub nodes stay within 3 cm of their place against the centre
of mass. The wheels drawn away from the car were the wheel meshes placed at their
`.dae` model origin, fixed by moving each wheel piece onto its wheel
(`flexmesh`, fd87ec32 on).

**Through the `.truck` export the same content does not rest.** The ETK on raycast
wheels through `desc_to_truck` (RoR's rules): 11 broken, 99 bent and a node at
9.8 m/s within 1 s (JBeam mode: 0, 0, 0.15 m/s). That is what tore the browser drive
yard's parked ETKs and sent cars past their contact frame bounds; the page now runs
JBeam content in JBeam mode by default (`?jbeamMode=truck` opts back).

**The full-size SUV and the boathauler trailer still do not rest (open, content past
the step's limit).** On raycast wheels and on their hubs alike:
* `fsw_fullsizesuv` (`mid-sport.pc`): nodes at 5–7 m/s for good, 6 beams bent at
  spawn. Its rear axle node `rx2l` (4.7 kg) carries eight 9.4 MN/m beams to the
  frame and an 18.8 MN/m axle beam: a Nyquist-rate oscillation (the force flips sign
  every step and grows about 5 % a step from 50 ms). The spring modes' power
  iteration gives ω·h = 2.07 (node `b1rr`), past the explicit step's limit of 2.
* `boathauler` (`gooseneck.pc`): its leaf-spring axle nodes (`x35*`, `x45*`, 6.2 kg,
  about 45 MN/m of plain beams each, plus the LBEAMs of §24) oscillate at the Nyquist
  rate at 20 m/s (position ±5 mm), KE 23 kJ steady; the plain springs alone give
  ω·h = 1.45, the LBEAMs' arm terms take it past 2.
Both run in BeamNG at the same 2 kHz, so its integrator or node handling differs from
the explicit solver's in a way its documentation does not give. Not fixed: every rule
that would rest them (mass scaling, softening the springs, an implicit pass on the
stiff subgraph) changes the content's physics, which needs the user's decision
(AGENTS.md).

*Not the missing wheel mass (2026-11-06).* The hypothesis: BeamNG's shared wheel,
tyre, brake and hub parts (not local) add mass the hub and suspension nodes lack. The
raycast spawn already adds each wheel's mass (20 kg) to its hub nodes, and the slots
the SUV and trailer miss are brake pads and bulbs only (no nodes). Measured
(`jbeam_accept`, raycast wheels, 2 s at rest; measurement options):
* **SUV.** 30 kg more per wheel on its hub nodes (`hubmass=30`): 6.8 m/s, KE 241 J
  (without 6.2 m/s, 207 J). The oscillating node `rx2l` is not a hub node: it is the
  rear trackbar's frame mount (4.7 kg, `nodeWeight` 4.7 in the file), tied to both
  frame rails by eight 9.4 MN/m beams and to the axle by the 18.8 MN/m trackbar; its
  lateral mode is at ω·h ≈ 2.0 by hand ((4·9.4 MN/m)/4.7 kg + 18.8 MN/m/2.35 kg).
  Without the torsion bars, the couplers or the damping filters it still runs
  (5.8 / 6.7 / 15.7 m/s).
* **Trailer.** Its axle torsion bars: without them (`droptorsion=` the ten bars
  `x15·`/`x14r`…, spring 810,100 N·m/rad, two per axle about the axle tube and about
  its top beam) it rests: 0.27 m/s, KE 3.3 J (with them 20.7 m/s, 23 kJ). Without
  the LBEAMs, the frame's torsion bars or the leaf bars about `x14`…: still unstable.
  Its axle tube's twist mode from these bars alone: the four 6.2 kg corner nodes 0.13 m
  off the twist axis, I ≈ 0.22 kg·m² per end, ω·h ≈ 1.9 with both bars; with the tube's
  plain beams past 2. The node weights are the file's (6.2 kg inline on each corner;
  the leaf nodes 3 kg from a modifier row, which the importer carries). BeamNG's
  torsion bar page gives the roles (levers 1 and 4, axis 2–3, Nm/rad) and warns that
  levers far from 90° to the axis are "less stable"; here they are at 90°. How BeamNG
  applies the torque to the four nodes is not documented (§9's dihedral gradient is a
  Choice): **open**, the next suspect.

*The torsion law and the damped limit (2026-11-06).*
* **§9's law is the exact one.** τ·∂φ/∂xᵢ (the Blondel–Karplus gradient) puts τ/r⊥
  on each lever end along its plane's normal (r⊥ its distance from the axis line,
  not the arm's length), and the axis nodes' reactions leave no net force and no net
  torque; it is −∇ of ½k(φ − φ₀)², so momentum, angular momentum and energy are
  conserved (`explicit_jbeam.rs` `torsion_forces_are_the_exact_lever_law`, and
  `dihedral_gradient_matches_the_f64_reference`'s central differences). No lever-arm
  error, no double application. Nothing to fix.
* **With it the trailer's highest mode is past the limit.** The spring modes' power
  iteration with the torsion elements' k·g·gᵀ (`jbeam_accept modes=1 torsion=1`):
  ω·h = 2.013 at the axle corner `x34l` (6.2 kg); the plain beams alone 1.448; without
  the ten axle bars 1.715. The 810,100 N·m/rad bars on 0.267 m levers are
  11.4 MN/m at each lever end.
* **SUV, with damping.** Symplectic Euler with a dashpot is stable to
  ω·h ≤ 2(√(1 + ζ²) − ζ). The beams on `rx2l`: to the frame 9.4 MN/m, 141 N·s/m,
  μ ≈ 3.4 kg, ζ = c/(2√(kμ)) = 0.012 (limit 1.98); the trackbar 18.8 MN/m,
  940 N·s/m, μ = 2.35 kg, ζ = 0.071 (limit 1.86). Its lateral mode at ≈ 2.0 is past
  both; the spring modes' iteration gives 2.07 for the whole car (node `b1rr`, its
  supports at full k). The damping is the documented one: −c·u with u the extension
  rate from the step's start velocities (the force pass reads the velocities before
  the node pass updates them), the trackbar's through its 500 Hz `dampCutoffHz`
  filter (§6; without the filters the SUV breaks 12 beams at rest), and the
  stiff-damper rule (§28) does not apply (c·h/μ = 0.2). No deviation found.
* So both vehicles are past the 0.5 ms explicit step's stability limit as written,
  with the exact laws. BeamNG runs them at 2 kHz; how it does (its integrator, or a
  stabilisation it does not document) is not public. Left as is (content physics
  unchanged); the options are the user's: a smaller step for such vehicles, or an
  implicit treatment of their stiffest elements.
* **Resolved (2026-11-06, §39, the BeamNG profile's default):** the explicit damping's half-step lag
  lowers the limit below 2 (to 1.87–1.98 at their ζ); centred damping
  (`centred_damping`) rests every SUV configuration, the trailer's axles and the
  CCF's broken ones at the same 2 kHz step, content physics unchanged.

### 29.2 Parts a configuration names that are not local (2026-11-06)

A community mod's configurations name parts from BeamNG's stock content (tyres,
wheels, brake pads, bulbs, occupants) and from other mods (a "godmode" suspension),
none of them local. **Choice** (BeamNG documents none; the data's own slots give the
fallback):
* **A missing part takes its slot's default** when that is local (the slot row's
  `default`), with a warning (`the local … stands in`).
* **A wheel slot** (`wheel_*`) whose part and default are both missing takes the
  first local part by name of its slot type, else of its family (`wheel_F_6` →
  `wheel_F_5`): the wheel part carries the hub nodes (`fw1rr`, `fw1r`, …) the
  suspension's beams and the `pressureWheels` rows use, so without it a car has no
  hubs and no wheels. Its rim and hub geometry is the stand-in's; its tyre (BeamNG's
  shared tyres) stays missing, so a raycast wheel's radius is the row's own or its
  hub's.
* Else the slot stays empty (warning), as before.
* **No wheels at all** (a boat; a bare body with its suspension slots empty, CCF
  `skull`): raycast spawns give no drive and a warning, never an error
  (`from_desc_jbeam_raycast`); the page shows a notice and the picker sends the
  vehicle to the 15 m drop.
* Beams the file gives `beamStrength` 0 (e.g. the Vertex's hood-open switch beams
  in `beater`, `vanilla`) break at once by design; the sweep lists them apart.

Sweep of every configuration (`community_sweep`, CPU, JBeam mode on raycast wheels,
1 s at rest): see the report of 2026-11-06 (scratchpad); the classes left are content
at the explicit step's edge (§29.1: the SUV, the trailer, the CCF's `country` and
`f6_offroadster` subframes, its `f4_cup` window nets).

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

**A compliant, hysteretic tread contact, measured (2026-11-05; CPU, reverted, the
patch is in the session scratchpad).** The coordinator's decision: replace §28's
inelastic ground rule on pressure-wheel tread nodes only with a compliant contact whose
loss comes from measured tread-rubber hysteresis.
* *Source of tan δ.* Mogi, Hase, Mizone and Nishitani, "Research on Optimization of
  Viscoelastic Properties of Tire Tread Rubber to Reduce Rolling Resistance and
  Decrease Braking Distance", Honda R&D Technical Review 19(1), April 2007: tan δ at
  60 °C and about 10 Hz is the index of rolling resistance, and the tread is about
  50 % of it. That paper gives no number. The value used, **tan δ = 0.12**, is a
  representative tread compound's (0.117 for a TDAE-oil reference compound, an ACS
  Rubber Division meeting abstract seen in a search index but not retrieved).
  **Unverified at the source:** a cited table (Clark, *Mechanics of Pneumatic
  Tires*; NRC Special Report 286) is still owed.
* *Hysteretic, rate-independent* (**Choice**: k = m/h², ω·h = 1, the stiffest the step
  integrates with margin; real tread rubber is about 15 times stiffer, about 10 MN/m
  per node, and cannot be integrated at 2 kHz; loading k·p, unloading k·p·(1 − ψ)
  below the episode's peak, ψ = (π/2)·tan δ, so a 0 → p₀ → 0 cycle loses a
  sinusoidal cycle's π·tan δ·k·a²): C_rr **0.016** (supercar, was 0.024) and
  **0.051** (Astral R, was 0.063). The landing energy now mostly comes back (ground
  vertical work −2.2 kJ/s), but the sidewall and L-beam dampers still take 17 kJ/s
  of the Astral R's 25. Fails the rest creep: the supercar drifts 105 mm against a
  sway of 103 mm. Below the peak the loop is elastic, so small oscillations are not
  damped and the tread walks.
* *Kelvin–Voigt* (c = k·tan δ/(2π·10 Hz), implicit, c·h/m = 3.8; stateless): C_rr
  0.035 and 0.069, worse. The viscous law over-damps the tread's fast modes,
  which are far above 10 Hz. Fails `pressure_wheel_car_stands_on_its_tyres` (the
  fixture car rolls 0.195 m, off its flats).
* **Not adopted.** Neither passes the rest tests. A hysteresis that damps small
  cycles too (reversal-point, Masing-type) needs per-node state on both solvers. The
  one that reaches the band at all, the supercar's, leaves the Astral R at 0.051,
  because its remaining sink is its own sidewall dampers. **Open, for the coordinator.**

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

## 34. The tread contact: hysteretic, with per-node reversal state (2026-11-05)

Pressure-wheel tread nodes (`NODE_TREAD`, both tyre rings; hubs and every other
node keep §28's rule, bit for bit) touch the ground through a compliant contact
whose only loss is the tread compound's measured hysteresis. It replaces §28's
inelastic rule for them, which removed 80 % of a landing node's normal speed (§31:
the lumped tread's patch-entry landings).

* **Source of tan δ.** Um, Kwon, Lee, Kim, Kim, Kim and Lee, "The Influence of
  Styrene Content in Solution Styrene Butadiene Rubber on Silica-Filled Tire Tread
  Compounds", *Polymers* 15(21) 4288 (2023), doi:10.3390/polym15214288, Table 9:
  tan δ at 60 °C (DMA, ARES-G2, 10 Hz, 0.2 % strain) 0.0823, 0.0818 and 0.0805 for
  three silica-filled S-SBR passenger tread compounds (60 phr silica, 6 phr TESPT).
  **TREAD_TAN_DELTA = 0.0815**, their mean. (A natural-rubber truck tread is higher:
  0.123–0.158 in Chu et al., *Polymers* 17(6) 714 (2025), Table 8, same conditions.)
  The 0.12 of §31's experiment is replaced: it was never retrieved at its source.
* **Law** (`explicit_beams::tread_hysteresis`, kernel `treadGround`): at depth p > 0
  the normal force is k·(p + z), k = m·`TREAD_K`. The hysteretic deflection z (per
  node, with the depth the step before: `ExplicitBeams::tread`, gBeams at
  `treadState`) follows the depth with slope μ while |z| grows and λ after a
  reversal until z passes zero (the rest of that step's travel at μ, so the law
  does not depend on the step); z ≥ −p (no pull); off the ground z and the depth
  reset to 0. Every steady cycle closes on itself centred on the elastic line k·p,
  and its loop is a parallelogram whose loss is tan δ of its secant stiffness for
  any amplitude and mean depth (`tread_contact_loses_tan_delta_per_cycle_at_any_amplitude`:
  0.0815 at 20 µm to 1 mm). Friction: the solver's law on that normal force, and
  §28's slip limit.
* **Choices.** The loop's shape: r = μ/λ = 4/π − 1, which gives the parallelogram
  the area and the mid-depth width of the linear viscoelastic ellipse of the same
  tan δ; λ = 0.6736 then follows from tan δ (μ = 0.1841). The stiffness: the
  stiffest tangent, k·(1 + λ), is m/h² (ω·h = 1, the stiffest the step integrates
  with margin; real tread rubber is about 15 times stiffer and cannot be integrated
  at 2 kHz). The loss per landing scales as F²/k, so C_rr depends on this Choice.
* **Results** (CPU; `community_cars_coast_down_by_aero_and_rolling_resistance`,
  25 m/s, 0.5–1.5 s): **C_rr 0.016 supercar** (was 0.024; passenger tyres
  0.010–0.015), **0.050 Astral R** (was 0.063). The Astral R's remainder is its
  file's own tyre dampers (`energy_ledger.rs` coasting at 25 m/s: sidewall beams
  16.1 kJ/s and periphery 4.1 of 24.5; the tread contact 2.75): content, not the
  solver. Supercar: tread contact 2.5 of 8.1 kJ/s, sidewalls 3.4.
* **Rest.** The fixture car stands (peak kinetic energy 1–4 s 40.0 J, 27.2 J at
  4 s; was 37.3 and 17.7) and does not creep (drift 11.7 mm, sway 53 mm). The
  supercar's spawn sag (about 1 kJ: it spawns with its springs unloaded) now bounces
  on its file's shock absorbers (`beamDamp` 600–1,100 N·s/m; its kinetic energy decays as a damping
  ratio of about 0.03) for about 5 s, where the inelastic ground absorbed it in 2 s; the
  community rest-creep test settles longer before its 8 s window (was 2 s; at 2 s
  the supercar still has 800 J and is not at rest). With 6 s it passed (supercar
  drift 2.1 mm, sway 53 mm) until §35's last-bit changes, after which the bounce
  leaves the supercar rolling at 0.22 m/s at 4 s (energy ledger: its mechanical
  energy only falls; the ground's horizontal work is −25 J over 10 s), and the
  tyres' low-speed rolling loss (about C_rr 0.003 at 0.2 m/s: the patch-entry
  landings that dominate at speed scale as v²) stops it only at about 12 s. The
  test settles 12 s: supercar drift 4.4 mm, sway 39 mm; Astral R 0.7 mm, 2.0 mm.
  **Open:** the low-speed rolling loss is about a fifth of a real tyre's, whose
  C_rr is nearly constant with speed.
* **GPU.** The same law in the kernel (strict arithmetic in the CPU's order; the
  constants as bit patterns); a vehicle with tread nodes (vehicle flag 512,
  `FLAG_TREAD`) takes the shapes path of the node phase, and the coupled path has
  the branch too. The fixture car at rest steps bit for bit with the CPU until the
  JBeam beam path's own last-bit difference (§33, step 2,688); §35 removes it.

## 35. JBeam mode at CPU↔GPU bit parity: exact lengths, atan2, slide sums (2026-11-05)

§33's follow-up. JBeam mode's beam lengths took 1/√q (CPU `1.0 / q.sqrt()`, kernel
`sdiv(1, ssqrt(q))`), which differ in the last bit now and then; at rest the
contacts amplify it (the fixture car parted at step 2,688).
* **Lengths.** `jbeam_force_filtered`, `jbeam_ext_force` and `lbeam_force` (kernel
  `jbeamUpdate`, `jbeamExtUpdate`) take §33's `exact_length` / `sExactLength`: the
  length √q correctly rounded and the inverse length within 1.65 ulp, one specified
  sequence on both. The `.truck` (RoR) profile is unchanged.
* **atan2.** The torsion elements' dihedral angle used the library atan2 (Rust's
  libm, the GPU's hardware one), not the same function: the Astral R parted at
  step 1. Both now run `exact_atan2` / `sAtan2`: a = min/max of |x|, |y|; past
  tan(π/8), π/4 + atan((a − 1)/(a + 1)); the series of atan to t²³ (truncation below
  4·10⁻⁹), Horner in t²; then the octant. Within 1.1 ulp of the exact angle
  (`exact_atan2_is_the_angle`). **Choice** (a specified sequence over a correctly
  rounded atan2, as §33's).
* **Slide forces.** The kernel sums each node's slide forces from zero in slide
  order and adds the sum to its force; the CPU added them one by one, which rounds
  differently where a rail end takes several. The CPU now sums them as the kernel
  does (`slide_pass`).
* **Checked** (`explicit_jbeam_ext_gpu.rs` `jbeam_mode_steps_bit_for_bit`, every
  node's position and velocity after each step, wgpu on Metal): the fixture car at
  rest 4,000 steps and at 10 m/s 2,000 steps; the supercar at rest 2,000 steps; the
  Astral R 426 steps. **Open:** at step 427 an Astral R node's first self-collision
  contact differs in its last bit (the triangle contact pass, common to every mode:
  its force reaches the node through the 1/32 N accumulator, and the beam forces
  were checked equal bit for bit). Not JBeam mode's; a follow-up. **Resolved in
  §36** (`fixed_contact`).

## 36. The BeamNG profile (2026-11-05)

`ExplicitBeamsSettings::beamng()` and `ExplicitBeams::two_body_contact`: switches
that JBeam content runs (`from_desc_jbeam_with`) and a `.truck` runs when it has a
`rubble_beamng_profile` line (**Rubble extension**). The default `.truck` (Rigs of
Rods) profile is unchanged, bit for bit, for RoR comparison runs. Each switch is a
field and can be turned back alone (`s.settings.<switch> = …`, `s.two_body_contact`).
The research behind the list is `ror-vs-beamng-research.md` §5.

| Switch | RoR profile | BeamNG profile | Field |
|---|---|---|---|
| Beam lengths | fast inverse square root | exact (§33, §35) | `settings.fast_inv_sqrt` |
| Idle-triangle throttle | skip an unhit triangle up to 12 steps | every triangle every step | `settings.triangle_throttle` |
| Contact rule (vehicle and self) | static surface: cancel the contacter's whole force | two-body (§10.3) with mass splitting | `two_body_contact` |
| Anti-clip (vehicle and self) | the band test only | side memory and a swept crossing test | `anti_clip` |
| Contact band | 2 cm (`set_collision_range`) | 2.5 cm, unless the file sets one | `collision_range` |
| `NONCOLLIDABLE` triangles (JBeam) | not collision triangles | anti-clip only | `anti_only_tris` |
| Contact force sums (CPU) | floats into the node's force | the GPU's fixed point (1/32 N, ties to even) | `fixed_contact` |
| Self- and vehicle-contact work | RoR's penetration push (0.2·p/h) | never positive (§37) | `dissipative_contact` |
| `.truck` deformable plain beams | RoR's rule (`Plain`) | JBeam's NORMAL rule (§3, `JbeamPlain`) | `use_jbeam_plasticity` |
| `.truck` ropes | RoR's rope | JBeam's NORMAL rule in tension, none in compression (`JbeamRope`) | `use_jbeam_plasticity` |
| Strength falls with plastic stretch | yes (k × the stretch) | no | `plastic_strength_drop` |
| Friction law's exp and pow | bit tricks (Schraudolph) | `exact_exp` / `exact_pow` | `exact_friction` |
| Air drag (`.truck`) | per node, DRAG·|v|·v (mass-independent), turbulence | the triangles' (§31), dragCoef 100 or `rubble_drag_coef` | `settings.drag` → aero triangles |

**No throttle.** RoR's throttle skips work whose result can differ (AGENTS.md: not
outside the RoR profile). A contacter arriving at 10 m/s at a triangle idle for a
second falls through with it (y −1.40 m after 0.2 s) and is caught without it
(0.020 m, the band's edge): `explicit_beams.rs`
`the_throttle_lets_a_fast_node_through_and_the_beamng_profile_does_not`. GPU:
vehicle flag 1024 (`FLAG_NO_THROTTLE`), the throttles capped at 0
(`throttleCap`).

**Two-body self-collision.** Self-collision used RoR's static-surface rule: it
cancels the contacter's whole force (its beams' load) and moves it onto the struck
triangle's nodes, with the weighted mean mass and every hit sized as if alone. In
the one-tier showcase car's 120 km/h wall that injects energy: the self-contact
work rises to +131 J over the first second (cumulative), the largest rise of
mechanical energy between frames is 119 J, and its size follows topology with no
physical link (a lid's rigidifier node wired to every lid skin node sits in the
contact band of the lid's own triangles). With `two_body_contact` the self hits
take vehicle contact's two-body rule (§10.3) within one vehicle: the step's hits are
found first (positions only), every node of a hit counts it (k sub-bodies), the
reduced mass is 1/(k_h/m_h + Σ w²·k/m), the cancelled force is the pair's relative
acceleration from the forces at the pass's start times that mass; then the hits
apply in the order found. Measured (CPU, `energy_ledger.rs`
`self_contact_never_adds_energy_in_the_120_kmh_wall`): cumulative self-contact work
never above 0 (−6.9 kJ over 1 s), mechanical energy never rises; the largest rise
between frames 6.2 J (showcase `f_wall`, sedan1), 0.00 J between tenths. The ledger
tests assert it for every two-body crash (`check`). GPU: the counts per node in gAcc
at `selfSplit` (cleared by its node phase; a two-body vehicle, flag 128, takes the
shapes path), every hit computed in the first round (no hit reads another's force),
the overflow path (`selfSerial`) likewise; `explicit_contact_gpu.rs`
`self_collision_two_body_steps_like_the_cpu` (the authored sedan into a wall: 0.0000
mm when self-contact starts; broken 127 / 134, bent 529 / 523 after 0.6 s). JBeam
mode's bit-for-bit runs (§35) are unchanged.

**Anti-clip.** [BN-nodes]: anti-clip forces near the surface "decide which side of
the triangle the node ends up on"; `NONCOLLIDABLE` triangles "still have anti-clip";
a triangle is 2.5 cm thick ("a node starts feeling a triangle at around 2.5cm").
RoR's band test is stateless: a node faster than r/h (40 m/s at 2 cm and 0.5 ms)
can step over the band, and a node pushed past the plane is pushed out of the far
side. With `anti_clip`, per (node, triangle) pair, in vehicle contact (on when the
owner or the partner has it) and self-collision
(`explicit_contact::clip_test`):
* **The side.** The side the node held for the triangle last step (it was in
  contact); else the side it came from when it crossed the plane over the step
  inside the triangle (the swept test: the node and the triangle a step back,
  x − v·h, the crossing linear in the step; its weights α, β interpolated, γ =
  (1 − α) − β); else, new and over the triangle within the band, the side it is on
  (self-collision: the sign of δ; vehicle contact: §5.3's vote).
* **The contact.** Weights inside (the crossing's, or now) and −`CLIP_DEPTH` ≤ d ≤ r
  with d = side·δ: a hit with normal side·n, depth min(r − d, r), so a node up to
  `CLIP_DEPTH` behind is pushed back to its side as one at the plane
  (**Choice**, `clip_depth`: the contact rule's push grows with depth, 0.2·p/h, and
  with the whole depth a node 10 cm through was kicked out at 40 m/s; BeamNG's BX
  coupe's side-pole ensemble went non-finite in one of 8 runs, and with the cap all
  8 are finite, residual crush 332–365 mm, anti-clip off 357–385). **Choice:** `CLIP_DEPTH` 0.1 m
  (BeamNG documents that a triangle with no node within 10 cm goes inactive).
* **Memory.** Each contact records owner << 21 | triangle << 1 | (side > 0) for the
  next step; the node pass makes the recorded words the held ones. Two slots: the
  two smallest words a step records (**Choice**: an order-free rule, so the GPU's
  lanes, recording in any order, keep the CPU's; a node touching three triangles at
  once keeps the two of smallest key). At most 1,022 vehicles and 2²⁰ − 1 collision
  triangles per vehicle in one upload, checked.
* **Reach.** The box tests reach the nodes that can cross: self-collision's
  triangle box grows by (its fastest node's speed + the vehicle's fastest) · h +
  `CLIP_DEPTH`; vehicle contact's candidate box per point by (the triangle's fastest
  travel + the point's) + `CLIP_DEPTH`.
* **`NONCOLLIDABLE`.** JBeam's NONCOLLIDABLE triangles become collision triangles
  after the others (`anti_only_tris`), part of self-collision, vehicle contact,
  breaking (§18) and the spawn exclusions (§10.1), only while `anti_clip` is on: no
  force in front of the plane (d ≥ 0: the side recorded only), behind it pushed back
  to the plane (depth min(−d, r)). `.truck` content has no such type.
* **Band.** JBeam content's range is 2.5 cm (was 3 cm, a Choice), a profile `.truck`'s
  2.5 cm unless it has `set_collision_range`.
* **Spawn exclusions** (§10.1) count a node on a triangle's edge to rounding as
  inside (**Choice**: 10⁻⁴ in barycentric weight). Without it the Astral R's nodes
  477 and 478, at T-junctions on the edges of NONCOLLIDABLE triangles 376 and 377,
  were caught crossing them as the car settled (step 2).

Measured (CPU, `explicit_contact.rs`
`anti_clip_catches_a_point_faster_than_the_band_per_step`): a point through another
vehicle's floor at 70 / 100 / 150 m/s ends 4.1 / 9.7 / 14.7 m below it with the band
test and above it with anti-clip; `an_anti_clip_only_triangle_holds_a_node_at_its_plane`:
a node falling at 1 m/s rests at 0.0200 m on a collision triangle, at 0.0000 m on an
anti-clip-only one, and falls through it with anti-clip off. GPU (flag 2048
`FLAG_ANTI_CLIP`; 4096 `FLAG_CLIP_MEM` on every vehicle of an upload with any: the
node phase rolls the memory and takes the vehicle's fastest speed (gAcc at
`clipVmax`, two parities), the shapes path; memory at `clipMem`, 4 words a node,
recorded by atomic min; triangle bit 30 anti-clip only): `explicit_contact_gpu.rs`
`anti_clip_steps_like_the_cpu`, points at 30–150 m/s through a floor, a
self-collision node at 90 m/s and an anti-clip-only triangle, 3 and 6 vehicles (the
small group and the owner-by-owner pass), one-step then 17-step frames: the side
memories equal word for word after every frame, positions within 1.1 µm (the
accumulator's rounding, §35).

**Fixed-point contact sums (CPU ↔ GPU bit parity).** The GPU adds contact forces
per node in a fixed-point atomic sum (units of 1/32 N), order free; the CPU added
floats, so the first contact step differed in a last bit (§35's open point). With
`fixed_contact` the CPU sums every self-collision and vehicle-contact force (each
hit's share to each node) the same way: each component ×32, clamped to ±2·10⁹,
rounded to nearest with ties to even (`to_fixed`), summed as integers; a hit reads
its node's force as force + sum/32 (+ the step's slide forces for vehicle contact);
the node pass takes force + sum/32, then the slide forces (`force_now`). This is the
GPU's arithmetic exactly, so no information is lost against the GPU result. The
kernel's `toFixed` now spells out ties to even (WGSL's `round` is ties to even,
Metal's `round` away from zero, so the native Metal host differed from wgpu at
ties). The RoR profile keeps float sums. Checked: `jbeam_mode_steps_bit_for_bit`,
the Astral R 1,200 steps bit for bit (its first self-contact at step 354);
`anti_clip_steps_like_the_cpu`, positions equal after 162 steps of crossings.

*The quantisation does no work (2026-11-06).* sedan1's speed sweep failed its
ledger in one run (51 km/h, drag on: +9.4 J between tenths); with `rubble_drag_coef
0` another run failed (+31.6 J), and with f32 sums none, so the fixed point was
suspected. Measured (`ExplicitBeams::contact_exact`, a measurement: beside the
fixed sums, each node's contact forces summed in f64 and counted; the
quantisation is the fixed sum less that, its work taken at the node's mean
velocity over the step, as the ledger's): over the 24 sweep runs (47–59 km/h, drag
on and off, 2 s) the quantisation's net work is −0.0033…+0.0036 J per run, at
most +0.0036 J over any tenth, the largest error on a node 0.065 N (the rounding's
worst case, adds·√3/64 N over the node's path, 0.3–1.2 J per run); over the
crushed 51 km/h car at rest to 12 s, 0.0000 J. It is unbiased (ties to even, no
correlation with velocity), no add saturates (|F| < 2²⁴/32 N keeps the
conversion exact), and the order does not enter (integer sums).
What rises is **self-contact**: the world ledger (`showcase_car.rs` `n_wall_runs`,
`N_TERMS=1`) puts +102 J of self-contact work in the failing tenth, nearly all
on one 0.55 kg node (352, beside cowl node 320) rattling at 1.2–1.4 m/s against
a triangle it is a vertex of (weight 0.88) with node 320 inside its band. The
source is the contact rule's penetration push (`contact_force`: an approaching
pair inside the band is given 0.2·p/h of separating speed, 3 m/s at 7.5 mm): a
node held inside a band by its beams is kicked out whenever it approaches and
pulled back by them, a limit cycle the beams' damping then absorbs. It runs with
f32 sums too (the most self-contact work over a tenth across the sweep: fixed
0.2–216 J, f32 0.1–152 J; at rest after the crash +5 J per tenth for good), and
with the push zeroed in a trial the late self-contact work fell to 0–7 J per
tenth. Which run's tenth crosses the ledger's limit is chaotic: any change to the
arithmetic moves it. Checked: `showcase_car.rs`
`fixed_contact_quantisation_does_no_work` (the sweep and the long rest: at most
+0.01 J over a run and over any tenth, and within the worst case). Resolved
in §37.

**Plasticity.** [B]: a beam deforms past `beamDeform` and "will no longer return back
to its original shape"; `deformLimit` / `deformLimitExpansion` cap the plastic change
and `deformLimitStress` the hardening; nothing documents a strength that falls as the
beam stretches (RoR's rule: S ← S − k·ΔL in tension). The profile's `.truck` content
now runs JBeam's NORMAL rule on every deformable plain beam (no limits, no cap: JBeam's
defaults; a `rubble_jbeam_plastic` line sets them), and no profile beam (`.truck` or
JBeam) loses strength with stretch. GPU: vehicle flag 8192 (`FLAG_KEEP_STRENGTH`).
* **Spring-back stays.** RoR's "spring-back" with plastic coefficient 0 is the
  elastic recovery of an elastic–plastic spring: the yield rule moves the rest length
  so the elastic stretch stays T/k and the force stays at the threshold while the
  beam yields; unloading then recovers T/k. That is the physics of any such spring.
  Coefficient 1 (the rest length jumps to the current length) drops the force to about
  zero on the next step, so a yielding beam carries no load between yield events.
  **Decision:** keep the file's coefficient (sedan1's: 0). The recovery is large only
  where a part's yield strain T/(k·L) is far above its material's (0.1–0.2 % for
  steel); that is content (`vehicle-authoring.md`, springback), not this rule.
* **Rupture.** Without the strength drop, a beam's only failures are its strength and
  its deform limits, as in BeamNG. A `.truck` file that wants steel to tear at its
  elongation at break sets `deformLimitExpansion` with `breakAtExpansion`
  (`rubble_jbeam_plastic`'s sixth field, §18.3): it tears at that stretch in tension,
  never in compression. (A strength set at a hardening cap, the alternative, tears
  crushed beams: compression hardens Tc toward the cap and the halfway rule's
  overshoot passes S; the showcase car measured 136 / 86 / 19 broken at margins 1.05
  / 1.5 / 3 in its 56 km/h wall, against 4 without.)
* **Ropes.** A profile rope (`r`) is `JbeamRope` (kind 21): in compression no spring
  and a tenth of its damping (RoR's rope), in tension JBeam's NORMAL rule, its
  strength kept as it yields. RoR's rope lowered its strength by k·δ each step, which
  at a slow pull (δ ≈ 0.1 mm a step) is below half an ulp of a 110 kN strength and
  never accumulated (the showcase car's door ropes held at 80–114 kN). Profile
  hardware ropes tear by `breakAtExpansion`. `tension_yield_gpu.rs`
  `ropes_and_tearing_step_like_the_cpu`: ropes and plain beams, torn and held, bit for
  bit on the GPU; `explicit_beams.rs` `profile_truck_ropes_and_break_at_expansion`.
* **Measured** (`crash_invariants.rs`, sedan1's 56 km/h wall, CPU; `CI_DROP=1` puts
  the drop back, `CI_PSET=1` sets coefficient 1, measurement only), lid vs rear body
  peak/end; rear body RMS; rear third plastic/broken; ripple body/bootlid/bonnet/door;
  whole car broken/plastic:

| Rule | Lid (mm) | Rear RMS (mm) | Rear third | Ripple (mm) | Car |
|---|---|---|---|---|---|
| RoR (before) | 425.5 / 167.5 | 42.5 | 79 / 9 | 4.0 / 12.9 / 7.6 / 4.0 | 114 / 947 |
| JBeam rule, with the drop | 443.8 / 156.6 | 45.0 | 85 / 8 | 4.1 / 7.8 / 6.7 / 3.5 | 104 / 979 |
| JBeam rule, no drop (**adopted**) | 442.1 / 221.6 | 45.4 | 90 / 0 | 2.9 / 8.9 / 8.1 / 4.5 | 4 / 1,078 |
| Coefficient 1, no drop (reference) | 423.3 / 236.2 | 45.4 | 102 / 0 | 2.9 / 6.8 / 7.4 / 4.8 | 4 / 1,135 |

  The drop is what broke beams (114 → 4; the rear third's 9 → 0); the rule change
  alone moves little. The lid's travel and the rear third's plastic beams do not
  improve: the rear damage is not plasticity's. The ripple falls at the boot lid
  (12.9 → 8.9 mm) and the body (4.0 → 2.9), not to 1 mm. The authored `sedan` (no
  profile) is bit for bit unchanged.

**Air drag.** A profile `.truck` with drag on takes §31's triangle aerodynamics on
every cab triangle (one per node set: a cab line of both groups is one aero triangle)
at BeamNG's defaults (dragCoef 100, liftCoef = dragCoef, stall 0.58 rad), and its
per-node drag and turbulence go off, as JBeam mode's. `rubble_drag_coef <d> [<l>]`
(**Rubble extension**) sets every triangle's coefficients. Measured
(`crash_invariants.rs` `profile_drag_is_the_triangles`, sedan1 coasting at 100 km/h
clear of the ground): 3,417 N with the default 100; the per-node rule would give
38,310 N (0.05·|v|² on each of its nodes, whatever their mass); a real sedan's
½ρ·Cd·A at Cd·A ≈ 0.65 m² is 307 N. BeamNG's default is 11 times a real car's here
(every panel a flat plate, inside ones too); `rubble_drag_coef 9` gives 310 N. The
GPU runs the same aero records (`showcase_car_gpu`, sedan1: at rest, the side pole and
the wall equal to the CPU's).

**Exact exp and pow.** The friction law (ground, static shapes, the tread, self-
and vehicle contact) took RoR's bit-trick eˣ (Schraudolph: up to 4 % off) and xʸ
(off by tens of percent between powers of two). The profile takes `exact_exp` (k =
⌊x·log₂e + ½⌋, Cody–Waite reduction, Taylor to r⁷, 2ᵏ by bits; within 0.8 ulp over
[−87, 88]) and `exact_pow` = exact_exp(y·`exact_ln` x) (ln by the atanh series to s¹⁰;
within 1.4 ulp; pow within 10⁻⁶ on the friction law's range): one specified
sequence on the CPU and in the kernel (`sExactExp`, `sExactLn`, `sExactPow`), so the
two agree bit for bit (`explicit_beams.rs` `exact_exp_is_the_exponential`;
`jbeam_mode_steps_bit_for_bit` with the profile's friction). Vehicle flag 16384
(`FLAG_EXACT_FRICTION`); the kernel's `Surface` carries it (`surfaceV`). The rigid
coupling's contacts keep RoR's law (its CPU reference has no vehicle). JBeam's
`dampCutoffHz` filter coefficient 1 − e^(−2π·f·h) also takes `exact_exp` (the host's
libm differed between machines).

**Damping-only beams and `dampCutoffHz` for `.truck`.** A profile `.truck` beam with
spring 0 (`set_beam_defaults 0, c, …`) is a damper: force −c·u, no yield (BeamNG's
damping-only beams). `rubble_damp_cutoff` (**Rubble extension**), lines `a, b, hz`:
the profile's JBeam beam a–b damps its extension rate filtered at hz (§2's
`dampCutoffHz`); a line naming no such beam fails the load. `rubble_aero_tri`
(**Rubble extension**), lines `a, b, c, dragCoef, liftCoef`: per cab triangle (nodes
in any order) its aerodynamics; with the section, unlisted cab triangles have none;
a line naming no cab triangle fails the load. Tests: `explicit_beams.rs`
`profile_truck_damping_only_beams_and_damp_cutoff`, `profile_truck_aero_triangles`.

**Effect on the showcase car (sedan1, CPU, the profile's 2.5 cm band, anti-clip off
and on).** T-bone at 80 km/h (`showcase_car.rs` `t_car_to_car_t_bone_at_80_kmh`,
live-skin leak metric, limit 10 of 993), striker offset −0.03 / 0 / +0.03 / 0.40 /
0.43 m: off 19 / 13 / 15 / 14 / 7, on 16 / 16 / 16 / 10 / 2 (means 13.6 and 12.0,
within the run-to-run chaos; deepest leak 0.19–0.24 m either way). The leaked nodes'
last inward crossings start at hits in front of the skin (on: 93 of 97 crossings,
"in front & out" 69, "behind & pushed in" 0): anti-clip keeps every node's side, but
the contact rule's push per step cannot hold a node the struck skin closes on, which
then passes CLIP_DEPTH and is forgotten. Not anti-clip's to fix. The frontal wall
(`crash_invariants.rs`, `CI_ANTI=0|1`): every row identical; its 3,408 self-contacts
are band hits on the side of δ (held sides 3,394), none crossed.

**Still to come in this profile:** `externalCollisionBias`; flexbody 4-node binding.

## 37. Self- and vehicle contact do no positive work (BeamNG profile, 2026-11-06)

§36's note found sedan1's ledger rise in self-contact: RoR's contact rule
(`contact_force`) cancels the pair's normal force, removes 80 % of an approach and
adds 0.2·p/h of separating speed at depth p in the band (3 m/s at 7.5 mm, 10 m/s at
the plane). A node its beams hold inside a band is kicked out whenever it
approaches and pulled back: a limit cycle, +5 J per tenth of a second at rest for
good, up to +216 J in a tenth of the speed sweep. Friction's stick term (−F_t below
3 m/s of slip) does the same in the plane. **Decision** (coordinator, BeamNG
first): in the BeamNG profile a contact never does positive work, and it still
resists penetration. The RoR profile is unchanged, bit for bit.

**Rule** (`ExplicitBeams::dissipative_contact`, self-collision, and vehicle contact
when the owner or the partner has it; CPU `contact_response`, `no_positive_work`,
`shift_depth`; GPU vehicle flag 32768 `FLAG_DISSIPATIVE`, `contactResponse`,
`noPositiveWork`, `shiftDepth`, `shiftAdd`/`shiftTake`). With vn, fn the pair's
normal speed and force and m the reduced mass (the two-body rule's, §10.3):
* **Normal force.** r = −fn − (vn + max(vn, 0))·m/h: an approach is stopped
  (vn′ = 0), a separation never made (separating, vn′ ≥ −vn). Its work over the
  step, r·(vn + vn′)/2·h, is −(½·m·vn² + ½·h·fn·vn) approaching and 0 otherwise.
* **Friction.** The tangential part t of the force is scaled by the largest
  s ∈ [0, 1] that keeps the whole force's work over the step at the pair's mean
  velocity ≤ 0 (the positive root of a0 + a1·s + b·s², a0 the normal part's).
* **Penetration: a position step.** RoR's push moved the pair 0.2·p apart per step
  as speed that stays (the pump). Here the pair moves δ = 0.2·p apart with no speed:
  a position impulse P = δ·m shared like the hit's force, each node by its
  sub-body's inverse mass (k/m), summed per node in fixed point (2⁻²⁴ m, ties to
  even, order free) and added by the next node pass after x + h·v
  (`contact_shift`; GPU gAcc at `selfSplit + totalNodes`, 3 words a node, the
  shapes path, every vehicle of an upload with one, flag 65536). Against an inward
  force (fn < 0) the step stores |fn|·δ (first order); it is capped to the energy
  the step's normal force took out of the pair: δ ≤ (½·m·vn² + ½·h·fn·vn)/|fn|.
  With fn ≥ 0 it releases energy and is not capped. A pair not approaching gets no
  step against a load: a node its beams hold in a band stays where it is.
* Only in the two-body rule (the profile's). Deterministic, strict single precision
  in one order on both.

**Choices.**
* *The cap is per hit and first order.* A hit's step is bounded by its own
  predicted dissipation; with mass splitting and several hits on a node the
  realised sums differ a little. Measured below: over any tenth, contacts plus
  steps at most 2 % of the steps' stored energy above zero (the strict "≤ 0 per
  hit" bound, a k-fold cap with each hit's largest sub-body count, let the T-bone
  leak 35 of 983); after the crash settles, their sum is negative.
* *Not tried further:* a step only for nodes behind the plane (leak 16), a budget
  of the approach's kinetic energy alone (leak 26), half the budget (leak 17).

Measured (CPU, sedan1, medium; `showcase_car.rs` `DISSIPATIVE_CONTACT=0` gives RoR's
push on both cars for comparison; the first measurement of §37, 0eb963d2, set it on
the struck car only, so its T-bone "RoR" column was wrong):
* **T-bone at 80 km/h** (`t_car_to_car_t_bone_at_80_kmh`, striker offset −0.03 / 0 /
  +0.03 / 0.40 / 0.43 m), leaked (net crossings) / inside (winding): this rule 9 /
  6 / 4 / 7 / 5 and 20 / 6 / 4 / 14 / 19; RoR's push 9 / 8 / 14 / 1 / 1 and 23 / 8 /
  14 / 1 / 1 (limit 10 leaked: RoR's fails at +0.03 m); 0eb963d2's rule without the
  step 46 leaked at 0 m.
* **Speed sweep** (n's 24 runs, 47–59 km/h, drag on and off, 2 s): self-contact work
  over any tenth at most −0.0008 J (was up to +216 J with the push); n's 32 ledgers
  pass (largest rise between tenths 0.00 J; was 9.4 J at 51 km/h); bands as before
  (max crush 0.729 m, peak 30.0 g, b1 19.3, t_m 87 ms, restitution 0.116; the car's
  own calibration fails them either way).
* **At rest after the 51 km/h crash to 12 s:** contacts plus steps −118 J over
  2–12 s; self-contact work per tenth at most −0.24 J.
* **BX side pole** (`beamng_o`, the hub-only BX, 8 runs at 32 km/h): residual crush
  17–41 mm (mean 30), RoR's push 15–43 (mean 29); all finite.

Checked: `showcase_car.rs` `contacts_do_no_positive_work` (the sweep, the long rest,
the T-bone: self- and vehicle-contact work over any tenth at most +0.01 J; with the
steps' stored energy, at most +0.01 J beyond 2 % of it per tenth, and at most
+0.01 J in all after the crash). GPU: `explicit_contact_gpu.rs`
`anti_clip_steps_like_the_cpu` with the rule on the floor and the self-colliding
vehicle: side memories word for word, positions bit for bit (the 150 m/s point now
ends 4.6 cm behind the floor, the 90 m/s self-collision node 1.1 cm behind its
triangle: the step is capped by the hit's budget; RoR's push threw them 14 and 42
cm clear).

### 37.1 At rest: the ground, the residual stop, friction's slip (2026-11-06)

sedan1 crushed at 51 km/h and at rest to 12 s still moved: kinetic energy about
15 J, nodes at 0.2–4 m/s, 800 J a second sloshing between self-contact and the
position steps. Three causes, three changes (BeamNG profile only; the RoR profile
is unchanged, bit for bit).

* **Ground, box and shape contact are dissipative too.** The node pass's static
  contacts take §37's rule (`ContactSurface::dissipative_if`, GPU `surfaceV` with
  `FLAG_DISSIPATIVE`). RoR's 0.2·p/h push on the ground did +8 to +10 J per tenth on
  wheel nodes at rest. Its penetration becomes a position step up, the node's own
  (`ExplicitBeams::ground_lift`, GPU `groundContact` and `takeSteps`), capped by
  the contact's budget as every §37 step is; boxes and shapes have no depth term
  (p = 0) and get none. Ledger: `EnergyLedger::w_ground_shift`.
* **The residual stop.** A two-body hit's stop uses the split reduced mass, so a hit
  whose nodes are in other hits stops only m_split·inv of its pair's approach
  alone. The rest, (vn + h·a)(1 − m_split·inv), stayed as a ghost speed that the
  steps cancelled every step. After the hits, one residual pass
  (`residual_self_pass`; GPU `selfResidualForces`, `selfResidualMerge`, and
  `selfSerialResidual` past the hit list's room) stops what is left. Each hit's pair,
  at the forces now (the first pass's contact sums included), approaching
  (vn < 0) and still approaching over the step (a·h + vn < 0), takes
  r = −(a·m + vn·m/h) > 0 along its normal. (Separating pairs have no ghost
  speed. Including them, the residual over-stopped where hits share nodes:
  +0.4 J of self-contact work in a tenth of the 48 km/h wall.) m is the first pass's reduced mass (its counts).
  No friction and no step. All the forces are found before any is added, as fixed
  point, so the pass is order free and momentum-exact. **Choices (measured, the
  wall sweep's self-contact work over any tenth):**
  * one pass: at most −0.6 J;
  * two or four passes, with the first pass's counts or with the active hits'
    counts only: up to +0.25 to +2 J (Jacobi over-stops where hits share nodes
    with conflicting normals);
  * the node-level removal of the speed along a node's own step (tried first):
    settles, but breaks momentum (the struck pressure-wheel car is not pushed).
  * Vehicle contact's residual pass exists on the CPU as a measurement
    (`VEHICLE_RESIDUAL=1`). It is not on the GPU yet, so it is off.
* **Friction never reverses the slip.** At zero work a sliding pair could swap its
  slip vt for −vt every step, for good (a node held ±4 m/s and 5 J that way).
  `no_positive_work` now also caps the tangential scale s so that the slip after the
  step, along vt, stays ≥ 0.
* **Self contact's position-step shares are each node's own inverse mass.** The step is
  still P = δ·m with the split m, shared as the hit's force is, w·P over each
  node's own mass. Over the sub-body's k/m, a node of k hits moved k times its
  share, and the steps stored more than their budget. Vehicle contact keeps k/m
  (**Choice**, measured): own masses there let the 80 km/h T-bone leak 15–17 of
  1,026 striker nodes at offsets 0 and −0.03 m; k/m leaked 6–7.

Measured (CPU, sedan1, medium, at rest after the wall, the 8th–12th second):

| | self-contact work | relative kinetic energy | relative rms |
|---|---|---|---|
| before (§37 as committed) | about 800 J/s | about 12 J | 130 mm/s |
| now, 48 / 51 / 54 / 57 km/h | 25–50 J/s | 0.5–1.7 J | 28–52 mm/s |

Relative means relative to the body's rigid motion, wheels left out.

The sweep's ledgers over any tenth now have their contacts and steps between
−0.4 and −4 J beyond the 2 % allowance (they were up to +0.09 J).

**Floor (open).** The motion left is a tangle of two or three nodes of the crushed
bonnet and body: mutual hits whose normals conflict, so no single pass stops them.
Over a second at rest the beams do +48 J against −50 J of contact, while the steps'
first-order estimate shows +7 J. The quantisation floor (single-precision positions
at the car's coordinates, ulp/h; contact steps of 2⁻²⁴ m) is about 1 mm/s, about
60× below.

`showcase_car.rs` `crushed_car_settles` checks it (51 km/h, the solver stepped
alone): kinetic energy per second monotone from 3 s (1,233 J to 5.3 J in the
12th second); relative rms below 80 mm/s (56) and every body node below 2.5 m/s
(1.49) over the 12th second. These are guards on the floor, not targets. The car has no brake to set (beam tyres; `brakeTorque` is not
simulated outside raycast drive). Probes: `settle_probe`, `contact_work_probe`
(`--ignored`).

**Verified (2026-11-06, after merging showcase-main).**
* wgpu: `explicit_contact_gpu`, `explicit_beams_bitwise`, `explicit_jbeam_gpu`,
  `explicit_jbeam_ext_gpu`, `explicit_drive_gpu`, `explicit_loop_cap`.
* Metal: `vehicle_parity`, `explicit_parity`, `explicit_drive_parity`.
* All 23 WGSL modules compile in Chrome.
* world-wasm `explicit_scenes`.
* CPU: `energy_ledger`; `crushed_car_settles`; T-bone leaked 7 / 5 / 6 / 2 / 4 of
  1,026 at the five offsets (≤ 0.7 %).
* n's 32 ledgers: no rise between tenths over 0.04 J. Its bands (max crush,
  b1, t_m) fail, as before: the car's calibration.

**The work guard (2026-11-06, later).** Tracing the failing tenth (56 km/h, drag
off, 1.3–1.6 s; `contact_parts_probe`) split self-contact work by phase:
first-pass normal −2.3 J, friction **+1.4 to +1.7 J**, residual −0.3 J.

The cause is friction in the bonnet tangle. Each hit's friction cap
(`no_positive_work`) judges it at its own predicted velocity with split masses,
and hits sharing nodes then realise other velocities. The T-bone showed the same
for vehicle contact's normal force: +32 J in one tenth when the cars met again
at 0.6 s.

**Rule** (`friction_guard_pass`, `guard_vehicle_pass`): after the residual pass,
each staged hit's force t (its first-pass force with its residual) is judged at
the pair's realised mean velocity over the step. That is the forces now (every
staged sum of the step or turn) on the nodes' own masses. Its work is
W(s) = s·A + s²·B, with
* A = t·(2v + h·a′), where a′ is the pair's relative acceleration without t;
* B = h·|t|²·inv.

Where W(1) > 0 the hit takes −(1 − s)·t, with s = −A/B (0 if A ≥ 0).

A force that does positive work moves the pair the way it pushes, so scaling its
normal part lets nothing in. Every guard is found from the same sums (order
free, momentum-exact).

**Quantisation-neutral sums.** The first pass and the residual add to the
accumulators as a staging the later passes read. At the end each hit's total
(first pass, plus residual, plus guard) replaces its staged adds: the
accumulators as they were, plus one add per share of the total. The fixed-point
rounding is the first pass's alone.

**Vehicle contact's residual pass is on.** It uses the same staging, first-pass
counts and approaching pairs only. The T-bone leaked 8 / 9 / 5 / 2 / 2 of 1,026
with it off and 3 / 4 / 6 / 0 / 2 with it on; the contact-work invariant holds
either way.

Measured (CPU, after the change):
* `contacts_do_no_positive_work` passes (all 26 runs, the 12 s rest, the T-bone).
* `fixed_contact_quantisation_does_no_work` passes (largest run +0.0016 J).
* `crushed_car_settles`: relative rms 36 mm/s, fastest 0.85 m/s, kinetic energy
  per second 213 → 3.4 J from the 3rd to the 12th second. A crushed part settling
  lifted one second (13.7 → 18.7 J in the 7th), so the check is per two seconds.

**GPU.** Per node, 8 words of gAcc at `resAt` (selfSplit + 4·totalNodes): the
residual's staged sums B and the totals' deltas D, fix(total share) −
fix(first-pass share), merged into the accumulators. That gives the CPU's restore
and re-add, integer for integer.
* **Self contact.** `selfResidualPhase` (residual, guard and totals, merge) after
  `selfApply`'s rounds; `selfSerialResidual` past the hit list's room, which finds
  the hits again and recomputes the first pass without its position step.
* **Vehicle contact.** `vvResidualPhase` per owner after its listed hits:
  `vvPass` in phase S and at the end; `vvPassSmall` after each level. A turn whose
  list overflowed (more than 2 × points hits) applies its hits at once, out of
  the specification's order as before, and stages none.

Verified:
* wgpu: `explicit_contact_gpu`, including `anti_clip_steps_like_the_cpu`
  bit-exact with dissipative contact in a small group and owner by owner;
  `explicit_beams_bitwise`, `explicit_jbeam_gpu`, `explicit_jbeam_ext_gpu`,
  `explicit_drive_gpu`, `explicit_loop_cap`.
* Metal: `vehicle_parity`, `explicit_parity`, `explicit_drive_parity`.

**Stated floor.** Single-precision positions move by whole ulps (about 1 mm/s
at the car's farthest coordinate) and contact steps by 2⁻²⁴ m (0.12 mm/s). The
< 1 mm/s target is withdrawn; the tangle keeps about 35× the floor.

The bounds below are superseded.

**Open: `fixed_contact_quantisation_does_no_work`.** The residual pass's extra
contact sums put 48 km/h with drag at +0.0102 J over the run, against the 0.01 J
bound (worst case 5.9 J); the others are within ±0.015 J.

**Open: the per-tenth contact-work bound.** `contacts_do_no_positive_work` fails
on a run or two of its 26, which ones depending on the build. Before the merge it
was 58 and 59 km/h with drag; after it, 56 km/h without drag (+0.38 J). At 58 km/h,
one tenth at rest (about 1.5 s) had contacts plus steps +0.60 J beyond the 2 %
allowance: the tangle's
steps store 2–3 J in a tenth whose contact dissipates 2 J. Over the run after the
crash, the same sum is −19 J. Every other run passes, as do the 12 s rest and the
T-bone. Tried without a fix:
* the step's cap scaled by m_split·inv;
* no residual pass (54 km/h, drag off: +29 J).

## 38. Surface contact: collision triangles against rigid shapes (opt-in, 2026-11-06)

**Problem.** Static and rigid contact was node only (a node strictly inside a shape is
pushed out), so a shape narrower than the node spacing passes between nodes with beams
and skin across it: the imported ETK (nodes about 0.5 m apart) let a 20 cm side pole
2.0 m into it at 32 km/h. **Decision** (user, 2026-11-06): rigid colliders stay
shapes; the soft body's collision triangles meet them by their surface, opt-in, with
node-only staying the default bit for bit.

**Switch.** `ExplicitBeams::surface_contact` (CPU, `explicit_surface.rs`); vehicle flag
131072 `FLAG_SURFACE` (GPU, `explicit_beams_surface.slang`); spec field
`surfaceContact` (`ExplicitSim`); page `?surfaceContact=1` (softbody.html, the crash
lab keeps it). Off: nothing runs (a uniform early return on the GPU).

**Features (no double counting).** Node contact keeps every node; a triangle takes the
contacts whose nearest feature on the triangle is an edge or its face. Per (live,
collidable, not anti-clip-only) triangle and shape whose boxes come within the band r:
* **X candidates**, the shape's features against the triangle as closest-point pairs: a
  box's 12 edges against the triangle (each end, the edge against each triangle edge,
  the edge piercing the face) and each triangle edge's run inside the box (distance 0
  at its middle); a capsule's or vertical cylinder's axis segment the same way (a
  cylinder's pairs taken to the solid cylinder's nearest point); a sphere's centre
  against the triangle; a hull's edges and the triangle edges inside its planes (CPU
  reference: proxy hulls, vertices and edges from their planes). Gap = distance less
  the shape's radius.
* **V**: each triangle vertex outside the shape, its exact distance to the shape. If a
  vertex is no farther than the nearest X candidate (+`TIE`, 0.1 mm) the contact is
  the node's and the triangle takes none. A vertex inside the shape does not count (the
  triangle still meets what crosses it beside that node).
* The X candidates within `TIE` of the nearest are **one contact at their mean**
  (weights, triangle point, shape point): a pole's face parallel to a panel is nearest
  along a strip whose ends the candidates are, so the contact is the strip's middle.
  Weights on one vertex only: the node's.
* **Normal**: from the triangle point toward the shape point; touching or crossing
  (gap or their distance ≤ 10 µm), the separating axis of least overlap (triangle
  normal, the shape's face axes, then edge × edge; a later axis must beat the best by
  10 µm), oriented from the triangle toward the shape. **Choice**: the minimum
  translation keeps a triangle crossing a wall edge-on pushed along the wall's normal,
  not sideways along its own.

**Response.** Vehicle contact's two-body rule (§10.3) with the shape as the contacter:
infinite mass when static or kinematic; a dynamic proxy k·(1/M + (r×n)·I⁻¹(r×n)), k its
surface hits this step × its share. The triangle point's inverse mass Σ w²·k/m (k the
node's surface hits this step: mass splitting); the relative velocity the shape's at
the point less Σ w·v; the pair's force −Σ w·F/m times the reduced mass, F each node's
force at the pass's start (its force plus its contact sum, a snapshot). Then §37's rule,
always (**Choice**: a new mode, no RoR legacy): dissipative, with the position step, in
the vehicle's band r (2.5 cm in the BeamNG profile); gap g ≤ r gives depth
min(r − g, r) (anti-clip's: a node behind is pushed as at the plane). The triangle's
nodes take −w·f through the fixed-point sums (the vehicle's `fixed_contact`) and the
position step their shares; a dynamic body takes +f at the contact point (fixed-point
surface sums per shape, integrated with the next step's reaction); the rigid body takes
no position step. Node contact in the next node pass reads the node's force with these
sums in it and only tops up to its own target: no double counting where both act.

**Order.** After self-collision, before the slide nodes, every step: find (positions
only) and count, then apply in triangle and shape order. GPU: three barrier-separated
phases (the snapshot; the find, counts by atomics; the apply, hits recomputed:
positions do not change), the same strict arithmetic in the CPU's order. The frame
kernel and the contact groups run it; the wide passes are off for a fleet with it
(`wide_ok`); with rigid proxies (the World's coupling) it is the CPU reference's only
for now and the plan refuses it (a clear error, not a silent fallback). Params:
`surfSnap` (gBeams, a float4 per node), `surfSplit` (gAcc, a word per node).

**Checked.**
* CPU, `tests/explicit_surface.rs`: triangle against box (face, turned edge, crossing),
  capsule, sphere, cylinder and hull, each with the expected gap, normal and point; a
  shape nearest a vertex left to the node; a plate with node lines 0.5 m apart at
  10 m/s against a 20 cm pole between them: node-only passes (mean x −2.4 m), surface
  contact catches it, its front edge's middle never more than 2 cm into the pole; a
  free 20 kg block takes the equal and opposite force: momentum closes to 0.9 mN·s of
  2,000 N·s (fixed-point rounding); a plate resting on a 4 cm bar under its middle (its
  nodes off the bar) is held, never below the bar's top, and gains no energy (0.1 s
  means never above their earlier peak; kinetic energy at rest < 0.01 J); off, or
  nothing near: bit for bit.
* GPU, `engine-world-wasm/tests/surface_contact_gpu.rs` (wgpu on Metal): sedan1 in
  the crash lab's side pole at 32 km/h, every node's position and velocity bit for bit
  with the CPU after each frame, off and on (2,941 hits); the 75° and frontal poles
  with it on (2,342 and 4,352 hits).

**Measured** (CPU reference, `examples/surface_pole.rs`, the crash lab's own spawn,
32 km/h, 0.6 s; off → on). Depth: how far any node, beam point or triangle point gets
inside the pole box (beams and triangles sampled); p99 over every inside element and
frame; crush: the struck side's skin in the pole's lane against the far side.

| Car, test | node / beam / triangle depth (mm) | p99 (mm), inside samples | peak (kN) | crush max / end (mm) | µs/step |
|---|---|---|---|---|---|
| ETK, side 90° | 31.6 / 99.6 / 98.1 → 47.6 / 95.7 / 9.8 | 97.3 (2,315) → 95.7 (96) | 148 → 401 | 2,030 / 1,916 → 353 / 160 | 90 → 255 |
| ETK, oblique 75° | 73.8 / 100 / 98.2 → 28.1 / 92.6 / 33.3 | 97.6 (1,167) → 91.1 (158) | 291 → 379 | 778 / 778 → 366 / 49 | 112 → 151 |
| ETK, frontal | 55.1 / 99.5 / 97.8 → 65.0 / 98.4 / 7.2 | 97.3 (1,376) → 96.1 (184) | 200 → 291 | 402 / 361 → 442 / 391 | 110 → 127 |
| sedan1, side 90° | 6.2 / 100 / 99.1 → 1.2 / 98.3 / 0.0 | 98.2 (7,385) → 97.9 (202) | 251 → 235 | 412 → 699 | 1,379 → 1,383 |
| sedan1, oblique 75° | 4.1 / 100 / 99.8 → 1.2 / 99.8 / 0.0 | 98.0 (8,874) → 93.7 (154) | 295 → 239 | 635 → 1,081 | 1,285 → 1,654 |
| sedan1, frontal | 4.7 / 99.8 / 99.2 → 6.3 / 99.7 / 17.8 | 95.4 (2,037) → 97.5 (755) | 231 → 229 | 537 / 491 → 674 / 534 | 1,856 → 1,468 |

* The ETK no longer lets the pole through: side crush 2.0 m (the pole through the car)
  → 353 mm peak, 160 mm residual; the peak force rises to 401 kN as the skin and
  structure take the load.
* The skin wraps the pole: the deepest triangle point inside it falls from 98–100 mm
  (the pole across the panel) to 0–33 mm; the elements inside fall 7–50 fold.
* What remains inside are beams that are on no collision triangle (interior structure,
  cross-car members behind the skin). Beam-edge contact behind the same switch is the
  next step (not built).
* sedan1's skin-based crush rises (412 → 699 mm side) because the skin now holds
  instead of passing through the pole uncrushed while the structure behind held (node
  depth 6 mm). The NCAP residual crush calibration (§36, `o_side_pole_matches_the_nhtsa_sedans`)
  must be re-run with it on before any default change.
* Cost (CPU reference: every triangle against every slot whose box is near; no
  broad-phase beyond the per-vehicle slot list): the ETK side pole 2.8× per step,
  sedan1 1.0–1.3×.
* GPU (`--gpu`, wgpu on Metal, M3 Max, wall clock per step with the frame's wait):
  sedan1 alone at the side pole 195 → 616 µs/step (3.2×: the triangles at the pole
  each run their whole narrow phase in one lane, twice: find, then apply); the
  `arena` of 64 sedan1 into walls and poles 8.23 → 9.76 ms per fleet step (+19 %:
  most triangles cull out; the three barriers and the snapshot per step remain). The
  ETK (RoR profile: float contact sums, not bit for bit, so its GPU run is its own
  chaotic path): side pole, triangle depth 98.9 → 25.9 mm, inside samples 2,101 → 281,
  crush max 456 → 339 mm, end 302 → 61 mm, peak 153 → 398 kN, 121 → 188 µs/step.
* Lossless speed-ups not yet taken: keep each triangle's find result for its apply
  (most hit one slot), spread a triangle's candidates over lanes, and a per-vehicle
  slot grid for large static sets.

**Open:** beam-edge contact; GPU proxies (the World's coupling); the single-vehicle GPU
cost; the NCAP pole calibration with it on. The default stays node only until those
hold.

### 38.1 Round 2: beam edges, point band, deepest point (opt-in, 2026-11-06)

Round 1 left interior beams on no collision triangle crossing the pole (p99 ≈ 95 mm).
Three additions, all behind the same switch (off: bit for bit as before, CPU and GPU):
* **Beam-edge contact** (`segment_contact`, GPU `surfSegment`). The segments are
  `surface_segments`: every beam of a physical kind (`segment_kind`: plain, limited,
  support, rope, shocks2, JBeam plain/support/bounded/aniso/pressured/rope; not
  triggers, couplers, torsion, LBEAM, pressure or aero records) between two distinct
  collidable nodes (not `NODE_NO_GROUND`) that is no edge of a collision triangle the
  pass meets (a triangle owns its edges, a node its ends). Fixed at spawn. A broken
  beam is skipped (GPU: its flag or this step's detacher-group break, as the CPU sees
  it after the beam pass). X candidates per shape edge (box's twelve, a capsule's or
  cylinder's axis, a hull's edges): the edge–beam closest points unless parallel, and
  each edge end against the beam (a parallel run's ends); a sphere's centre against
  the beam; the beam's run inside a box or hull (both its ends, distance 0). V: the
  beam's ends, as a triangle's vertices. The same mean, band, law and splitting;
  weights (w₀, w₁, 0) on nodes (a, b, a).
* **Deepest point when touching or crossing.** With the normal from the separating
  axis (gap or point distance ≤ 10 µm), the contact is no longer the mean of the
  candidates (a run's middle the element turns about: measured, a sedan1 bumper beam
  held at its middle rotated 4 cm into the pole while the middle separated) but the
  deepest of the candidates with shared weights along the normal (ties within 0.1 mm
  their mean). The runs inside a box are now listed by both ends (triangles too). A
  beam whose deepest candidates spread along it (the least-overlap axis is usually
  across the beam, so a run's two ends tie) gets **two contacts**, at the least and
  the most parameter, so it is held at both.
* **Point contact in the band** (`point_contact`, GPU `surfPoint`). A collidable node
  outside a shape and within the band of it (gap > 10 µm) gets the same two-body law
  toward the shape's nearest point (a hull's: back along its farthest plane). A node
  strictly inside stays node contact's, which tops up to its own target (no double
  count). Without it, a vertex-nearest contact was the node's but node contact has no
  band, so an ETK front node reached 36 mm inside.

Checked: `tests/explicit_surface.rs` (a beam under a box face by its edges, its end
the node's, capsule, sphere, cylinder; a tilted beam through a box held at both run
ends; a node in the band; the plate's one segment, its non-triangle diagonal), and
`surface_contact_gpu.rs` bit for bit on the GPU: side 2,923, oblique 2,404, frontal
9,040 hits.

**Measured** (CPU, `surface_pole.rs`, 32 km/h, 0.6 s, flag on; round 1 → round 2).
The harness now counts colliding matter only: nodes not `NODE_NO_GROUND` and beams of
`segment_kind` between two of them (round 1's 30–65 mm ETK node depths were
non-colliding nodes; flag off, the same filter: ETK node depth 4–10 mm).

| Car, test | node / beam / tri depth (mm) | p99 (mm), inside samples | peak (kN) | crush max / end (mm) |
|---|---|---|---|---|
| ETK, side 90° | 0.0 / 0.5 / 0.0 | 0.5 (1) | 373 | 336 / 0 |
| ETK, oblique 75° | 0 / 0 / 0 | 0 (0) | 338 | 735 / 518 |
| ETK, frontal | 0 / 0 / 0 | 0 (0) | 216 | 548 / 497 |
| sedan1, side 90° | 0 / 0 / 0 | 0 (0) | 365 | 459 |
| sedan1, oblique 75° | 0 / 1.2 / 0 | 1.2 (2) | 349 | 874 |
| sedan1, frontal | 0 / 24.7 / 17.3 | 24.7 (31) | 212 | 592 / 480 |

Flag off (same harness): p99 96–98 mm on every case. Every case is now within the
2.5 cm band. Crush moves (sedan1 oblique 635 → 874 mm, ETK oblique 778 → 735 mm):
the pole now meets the structure where it is, see §38.2.

Cost (GPU, wgpu on Metal, M3 Max, shared with other work): sedan1 at the side pole,
1 s, 118 → 644 µs/step (round 1 measured 195 → 616 on a quieter GPU); the 64-car
`arena` 7.41 → 8.30 ms per fleet step (+12 %). CPU reference: sedan1 425 → 490
µs/step, the ETK 85 → 105.

GPU, lossless: the find stores per triangle, segment and node the mask of the slots
it hit (gAcc at `surfMark`; bit b slot b, bit 31 any slot from 31 on) and the apply
recomputes only those: sedan1 at the side pole 107.9 → 608.3 µs/step (118.2 → 643.8
without the masks in the same session); the 64-car arena 6.90 → 7.52 ms per fleet step
(+9 %, was +12 %). Bit for bit as before.

### 38.2 Calibration with surface contact on (sedan1, CPU, 2026-11-06)

`showcase_car.rs` with `SURFACE_CONTACT=1` (a measurement switch). The momentum
ledger now counts the surface pass's force on the nodes (`ExplicitBeams::surface_force`,
as the fixed-point sums take it) as the obstacle's impulse; before, it showed as
"internal" (c, d, n failed their ledgers on that alone).

| Test | off | on | band |
|---|---|---|---|
| o, NCAP side pole 32 km/h 75°, ensemble mean residual crush | 299 mm (145–424), pass | **68 mm (42–115), fails** | 297–445 mm |
| c, side pole at the B-pillar 32 km/h | door intrusion 11 mm, B-pillar 14, passes | 93 mm, 9, passes; ledgers close | not through |
| d, centre pole 50 km/h | CoM 1.66 m behind, passes | 3.52 m behind, passes; ledgers close | not through |
| n, wall 56 km/h ensemble | max crush 474, b1 31.1, dynamic 668, restitution **0.2018 fails** (pre-existing) | 520, 28.0, 657, 0.1994: **every band passes** | 380–580, 22.9–39.2, 620–750, 0.13–0.20 |
| q, FMVSS 216a roof | 69.9 kN = 4.91 × weight | the same (the platen is not a static shape) | ≥ 3.0 |
| MDB (`explicit_scenes.rs`) | not run: vehicle against vehicle, no static shape meets the surface pass | | |

The side pole moves: the pole used to slip between the door's nodes and the measured
crush (the struck skin's nodes against the far side) included that pass-through; met by
the skin and the beams, the door holds and the residual crush falls to a quarter. That
is sedan1's side structure calibrated with node-only contact, not a contact error: with
surface contact on, its door and sill strengths need re-tuning against §4.11 before the
flag can be the default. The ETK has no band tests in this branch (its harness,
`vehicle_compare`, is on `feat/vehicle-compare`, which does not carry surface contact).

**Recommendation: not the default yet.** It removes pass-through (p99 penetration
91–98 → ≤ 25 mm on every pole case) and brings the wall back inside every band, but it
takes sedan1's NCAP side pole from inside its band (299 mm) to far below it (68 mm), and
costs 5.6× per step for one car at a pole on the GPU (+9 % for the 64-car arena).
Default it once sedan1's side is re-tuned with it on and the World-broadphase work below
lands.

### 38.3 One collision system with the World (built, 2026-11-06)

Decision (user, 2026-11-06): surface contact reuses the rigid engine's collision
machinery. Branch `feat/softbody-world-broadphase`; measurements in
`engine/docs/measurements/softbody-world-broadphase.md`.

**Shared narrow phase.** The strict closest-feature queries moved out of the explicit
kernel into `slang/physics/feature_query.slang` (helpers `slang/physics/strict.slang`,
the explicit kernel's `nf`/`sdot`/... unchanged in arithmetic), with one Rust twin,
`engine_schema::feature_query` (`explicit_surface` re-exports it). Point, segment and
triangle against box, convex hull, capsule, vertical cylinder, sphere and a single
triangle (a trimesh's, or a heightfield cell's two, `heightfield_cell`), weights,
normal, gap, point; strict poses (`shape_pose`, `axes`), bounds, and casts (below). A
hull is read in the World's own record format (the hull arena's vertices, edges and
planes, the collider's scale applied): `vehicle_coupling.slang` copies them after the
proxy's planes when any vehicle has surface contact (`hullGeom`), and the kernel's
accessors (`fqHullVertex`, `fqHullEdge`, `fqHullPlane`) read them there. The rigid
engine's SAT manifolds (`convex.slang`, `box_collide.slang`) are untouched: rigid
results are unchanged by construction. Planes and heightfields stay out of the coupling
(the vehicles keep their own ground, as §1 of `explicit-vehicles-in-world.md`): their
queries exist and are tested, but no proxy carries them.

**Proxies in the surface pass.** The GPU surface pass meets the coupling's proxy shapes
after the static slots (the CPU's `surface_shapes` order: boxes, shapes, proxies), static
and dynamic, every kind (box, sphere, capsule, hull). A dynamic body is split over its
surface hits this step and its vehicles (`share`), the inertia term strict (the CPU uses
the same `fq::rot`), and takes +f at the contact point into the coupling's step sums (the
node contact's), integrated at the next step's start. The CPU reference now integrates
its proxies at the step's end, after the surface pass (`ProxySet::end_step` moved out of
the node pass; nothing between reads them, so with surface contact off every result is
bit for bit the same); a proxy's surface reactions join that step's sums
(`surface_sum` removed). The plan's refusal of surface contact with proxies is gone.
The proxy loops are their own functions beside the unchanged static-slot loops
(`surfTriangleProxies`, `surfSegmentProxies`, `surfPointProxies`): with them inline an
unrelated edit changed the turned-proxy results on Apple's compiler (the sensitivity of
`explicit-vehicles-in-world.md` §8.3).

Checked: `engine-webgpu/tests/explicit_surface_proxy_gpu.rs` (wgpu on Metal): the repo
sedan, surface contact on, flying at 10 m/s into a 20 cm pole that is a proxy: a static
turned box, a static turned hull (vertices and edges from the arena) and a free 60 kg
box: first contact frames within 2.4e-7 m of the CPU, the vehicle's speed within 1 %, the
free pole's velocity within 1 % and horizontal momentum closed to 0.03 of 37,628 N s; with
coupling room but no proxies, bit for bit the static build. `surface_contact_gpu.rs` (the
static slots against the CPU) still bit for bit. Native Metal
(`engine-metal/tests/explicit_surface_proxy_parity.rs`): unturned poles match (1.8e-7 m);
**open**: a turned proxy left the CPU by 3.9 cm at first contact there before the proxy
loops were split out; the ignored test `metal_surface_contact_meets_a_turned_box_proxy_like_the_cpu`
reruns it.

**Broad phase.** See `explicit-vehicles-in-world.md` §11: the coupling plan's
candidates come from the World broadphase's persistent resting table and the step's
moving and moved colliders, not a scan of every collider; the plan is the scan's word
for word.

### 38.4 Fast proxies: casts (2026-11-06)

Problem: contact with a proxy is discrete at 2 kHz. A body of least extent e moving v·h
past e in a step can cross a panel or a node between steps; nodes it carries past its
centre leave by the far side and push it on.

Rule (with surface contact on; `RUBBLE_SURFACE_CAST=0` turns it off for measurement):
* A proxy body is **fast** when (|v| + |ω| R) h > e/2 (R its shape's bound radius; e a
  box's least half extent, a sphere's or capsule's radius, a hull's least plane offset,
  at least 1 cm). Slow bodies take exactly the path they took before.
* For each element (triangle, beam segment, node) with no discrete contact against a
  fast shape: its time of impact over the step's relative motion d = (v_body − v_element)
  h, by conservative advancement on `fq::distance` (`fq::toi`, at most 24 advances, each
  the distance past the band over |d|: the band is never jumped), in the swept cull box.
* A gathering pass takes each fast body's earliest time of impact t_min over every element
  (GPU: in the snapshot phase, an atomic max of 0x7FFFFFFF − t's bits, by step parity).
  Only elements met at t ≤ t_min + TIE/|d| get a contact: at their own time of impact,
  against the shape moved there (`fq::translated`), so on the side it came from; the body
  takes it at its own point now (`Found::shift`). An element it would reach only after
  passing through another is not met this step.
* Node contact (the node pass) with a fast body that overtakes the node (relative motion
  along the body's own): the node leaves along that motion, not by the nearest face.
* The response, mass splitting and reactions are the surface pass's own.

Checked: `engine-softbody/tests/explicit_surface_cast.rs`: a steel ball (3 and 6 cm
radius) at 300 and 600 m/s into a 0.6 m plate is met (without the casts the 3 cm ball
passes untouched), momentum closes, kinetic energy falls; a 5 m/s ball runs the same
path. `ccd_cannonball` (sedan1 and the repo sedan, CPU, wgpu and the World): the table
is in the measurements file. No projectile tunnels at 30 to 600 m/s with the casts, and
none gains speed. Before them the repo sedan let a 6 cm ball through at 600 m/s with 47 %
of its momentum; an intermediate version (contact point at the moved pose) spun and sped
the ball up: the body's lever arm is its point now. CPU and GPU agree on the momentum
given to within 2 points; the World's run differs as its time coupling does (the ball
integrates in the rigid step between windows).

### 38.5 Cost: lossless speed-ups against the proxies (2026-11-06)

Branch `feat/surface-contact-perf`; measurements in
`engine/docs/measurements/softbody-world-broadphase.md` §3.

**Where the time went** (`vehicle_broadphase_bench --probe`, EB_ABLATE bits 2^20–2^30 in
`explicit_beams_surface.slang`, time only; one repo sedan among 11 proxy shapes, frame of
17 steps, GPU timestamps): the frame 16.7 ms, of which the surface pass 15.9 ms (its three
barriers 0.13 ms). The static slots and the snapshot were 0.5 ms; the proxies 15.4 ms: the
per-element cull (each element read every shape's fast test and grown sphere) 1.8 ms, the
shapes' strict poses and bounds rebuilt per element 0.8 ms, the narrow phase 12.9 ms
(triangles 11.0, segments 3.1, points 0.9 ms; the apply recomputing the find's hits 6 ms).
Of 134 pairs whose boxes met, 9 hit: the narrow phase ran mostly to reject, a box's twelve
edges or a hull's eighteen against each triangle, four candidate passes per contact, on
the few lanes whose elements sit by a shape while the rest of the workgroup waited.

**The pair path** (`surfaceStep`; every decision uniform, each heavy function called once
per phase, as the Metal and Chrome compilers inline every call): per step each proxy's
bounds once
(`surfPairBounds`); every live element against them, each meeting pair listed
(`surfPairCull`: exactly fqNear's test, so the same pairs the per-element loops reach; their
grown-sphere cull only ever skips pairs fqNear rejects); a barrier; the pairs spread over
the lanes for the narrow phase (`surfPairFind`), each hit counted as the loops count it and
kept (`SURF_HIT_CAP` 256 records of 19 words per vehicle); the apply from the kept hits.
Every sum it feeds is fixed point or an integer count, so order is free. Fallbacks, both
exact: more pairs than `SURF_PAIR_CAP` (2048) and the step takes the per-element loops;
more hits than the records hold and the apply recomputes every pair's. A fast proxy
(casts, §38.4) stays with the per-element loops (its bounds in the listing empty, the
loops told to meet the fast proxies only), beside the pairs for the rest. Memory per vehicle with surface contact
and the coupling: 2048 + 256·19 + 6·S words of gNext (≈ 30 KB with S = 128), two gAcc
words. Switch (tests, measurement): coupling header bit 16, `RUBBLE_SURFACE_PAIRS=0`.

**Edge culls in the shared queries** (`feature_query.slang` `fqFar`, `fqPrune`): a pass keeps
candidates of gap at most its threshold (pass 0 the least so far, capped at the contact's
band: a least gap past it is no contact either way; later passes gx + TIE). Every candidate
of a shape edge pairs a point of the edge with a point of the element, so its gap is at
least their boxes' gap, less the rounding of the computed points; an edge farther than
the threshold by 1e-4 m + 1e-5 of the largest coordinate is skipped. The Rust twin keeps
the full walk (the same candidates count). Switch: coupling header bit 32,
`RUBBLE_SURFACE_CULLS=0`.

Result (frame GPU time, the switches timed from one saved state; measurements file §3):
one sedan among 11 proxies 21.9 → 4.2 ms per 17 steps (the pass 1.24 → 0.20 ms a step);
the crash town, 4 cars among 29–77 house chunks each, 55.7 → 17.1 ms; 32 cars 213 → 37 ms
(minima on a shared GPU; 16.7 ms with the pass off).

Checked bit for bit: `engine-world/tests/surface_pairs.rs` (a sedan between rows of boxes,
turned boxes, hulls, turned hulls, into a capsule, a sphere, a free box thrown at it and a
fast steel ball (casts beside the pairs):
pairs, culls and both against neither, every vehicle state word, proxy record and coupling
sum, through the strike and at rest); `surface_contact_gpu.rs` (static slots, against the
CPU twin, with the culls on) and `explicit_surface_proxy_gpu.rs` unchanged; the World
coupling tests (`explicit_vehicles.rs`, including the workgroup-copy and cull bit-exact
checks) pass.

### 38.6 Friction against a turning body (cast regression, 2026-11-06)

**Problem.** After the casts, a 6 cm steel ball at 100 m/s into the repo sedan gave 76 %
of its momentum and went into the car (before: it bounced). Traced on the CPU: its first
hit (a cast contact) left it at 3,545 rad/s, more kinetic energy than it had. The surface
contact's friction against a dynamic proxy was capped (§37's no positive work) with the
pair's reduced mass along the normal; a turning body answers a tangential impulse through
its inertia as well (a sphere 3.5 times as much), so the cap let the friction drive the
contact point's slip far past zero and spin the body up. Without friction the ball bounced
(105 %).

**Rule** (`no_positive_work_body`, GPU `surfBodyCap`; dynamic proxies only, after
contact_response's own cap, so the friction is the lesser of both): the pair's velocity
change per unit impulse x is k(x) = imt·x + split·(x/M + (I⁻¹(a × x)) × a) (a the arm from
the body's centre, I⁻¹ in world, split its mass splitting); n·k(n) = 1/m keeps the normal
rule; the friction t is scaled by the largest s ∈ [0, 1] for which
(r·n + s·t)·(u + h·k(r·n + s·t)) ≤ 0, u = 2v + h·f/m. Static bodies and static slots take
the old path bit for bit.

Checked: `explicit_surface_cast.rs` `a_glancing_ball_is_not_spun_up_by_friction` (100 m/s
at 60° and 45°, 30 m/s at 70°: energy, rotation and the plate's height under gravity
included, does not rise; without the cap it rose 6.5-fold); the cannonball table (the
measurements file §2.1): the repo sedan's 6 cm ball at 100 m/s 75.9 → 103.0 % (bounces),
its 300 m/s 122.4 → 101.5 %, the spins that exceeded the projectile's speed gone
(12,682 → 596 rad/s at 600 m/s); no projectile tunnels and none gains energy.

### 38.7 Default or not (2026-11-06)

**Not yet.** The cost is now small and flat where it was the blocker (one car: 0.20 ms a
step against 1.24; flat in the map's bodies), but:
* the crash town still pays for it: 32 cars among house chunks 37 ms a frame against 17
  with it off (the per-element static-slot and fast-proxy loops, a contact group's
  members stepping the pass one after another, and the hits' narrow phase remain); 64
  and 256 cars were not measured (§3 of the measurements file: the merged branch's
  64-car town stalls with the pass off too);
* sedan1's NCAP side pole falls out of its band with it on (§38.2), a calibration the
  default must wait for;
* native Metal (Slang to MSL): the turned-proxy check
  (`metal_surface_contact_meets_a_turned_box_proxy_like_the_cpu`) was not re-run: the
  four parity tests' kernel compile ran past the 20-minute bound on a shared machine and
  was stopped. wgpu on Metal matches the CPU for turned proxies (2.4e-7 m).

Next: the static slots and fast proxies on the pair path too, a group's members side by
side in the pass, and the Metal turned-proxy run on a quiet machine.

### 38.8 Friction: a true stick (2026-11-06)

**Problem** (reported from `feat/vehicle-gen3-r4`'s `pole_slide_mm`): in the 75° oblique pole
the car slid along the pole with surface contact on (gen3 0.7–1.3 m, the ETK 1.7 m; node
contact 0.13 m) and the pole crush collapsed. Checked against the list:
* coefficient: `ContactSurface::BOX` (μs 1.2, kinetic curve from 1.2 to 0.75 plus 0.01·σ)
  times the nodes' friction, weighted over the element, as node contact's;
* the cap (§37 `no_positive_work`, and §38.6's for dynamic bodies): clipped 3–8 % of the
  friction in the ETK's oblique pole, not more;
* the tangent: per hit, from that hit's normal; no state, so nothing is lost when a
  contact moves to an edge or the deepest point (a round shape's crossing normal from
  radial axes was tried as well and changed nothing measured: not taken);
* **the static/kinetic split: the defect.** The surface pass took RoR's law
  (`contact_response`), whose static branch cancels the tangential *force* but meets an
  existing *slip* with only 1 − e^(−σ/3 m/s) of the limit. A node inside a shape hides
  it (the shape holds the node by its depth); a skin held at the band does not: a panel
  on two bars between its node lines crept down a 25° slope (9 cm in 1 s, 0.7 m at 40°)
  where node contact holds to 48°.

**Rule** (`surface_response`, GPU `surfResponse`; surface contact only, node contact
unchanged): the tangential force that stops the slip within the step, −(f_t + v_t·m/h),
is taken whole while within μs·r; past it the pair slides on RoR's kinetic curve against
its slip. Then §37's cap. Measurement switch (CPU): `RUBBLE_SURFACE_FRICTION=ror`.

Checked: `engine-softbody/tests/explicit_surface_friction.rs`: on two capsules and on two
bars (no node touching), a panel pushed along them holds under 1.1 × its weight and slides
at 1.5 (node contact the same, to the millimetre); set sliding it stops within the
law's distance (1 m/s: 4.3 cm; node contact on RoR's law 23.6 cm); on an incline it holds
to 48° and slides at 55° (as node contact). `surface_contact_gpu` (static slots, GPU
against the CPU) and `explicit_surface_proxy_gpu` pass; flag off bit for bit.

Measured (`surface_pole --scene oblique --cyl`: the crash lab's oblique pole as a 254 mm
cylinder, 32 km/h, 1 s, CPU; pole slide = the pole's shift along the car's long axis):

| car | node contact | surface, RoR friction | surface, stick |
|---|---|---|---|
| ETK slide / residual crush | 51 / 468 mm | 624 / 205 mm | 324 / 307 mm |
| sedan1 slide / residual crush | −124 / 569 mm | 2,637 / 1,515 mm | 2,726 / 1,613 mm |

The ETK's slide halves and its residual crush rises toward node contact's. sedan1's does
not move, and it is not friction: with every node's friction ×3 it slides the same
2.7 m, frictionless 3.4 m the other way; its side gives 1.3–1.6 m to the pole with
surface contact on (0.6 m node only), so the pole runs along the collapsed door. Open:
sedan1's side structure under surface contact (with §38.2's calibration).

## 39. Centred damping (2026-11-06)

`ExplicitBeamsSettings::centred_damping`: **on in the BeamNG profile** (`beamng()`,
since 2026-11-06), off in the Rigs of Rods profile, which stays bit for bit. A
vehicle turns it back alone (`s.settings.centred_damping = false`); the crash lab
and the showcase page take `?centredDamping=0/1` (ExplicitSim's `centredDamping`,
BeamNG profile vehicles only). Research branch `research/explicit-stability`.

**Why.** §29.1 left the full-size SUV, the boathauler trailer and the CCF's
`country`, `f4_cup` and `f6_offroadster` configurations vibrating or breaking at
rest: content BeamNG runs at the same 2 kHz. The step's own limit is not the
whole story: with a dashpot, symplectic Euler (velocity first, then position,
the damping read from the step's start velocity) is stable only to
ω·h ≤ 2(√(1 + ζ²) − ζ), so damping *lowers* the limit (1.94 at ζ = 0.03, 1.87 at
0.07). The damping reads v_{n−½}, half a step behind the position it acts at
(the θ/2 lag of §6), and near the Nyquist rate that lag turns the dashpot into
an anti-dashpot. BeamNG's staff say its content sits at that edge: "Most of the
vehicles have the spring values very close to the max value for a given mass"
(Goosah, BeamNG team, forum 2018-07-25 [F]); the census measures BeamNG parts
at 1.1–4× a conservative per-node bound with ζ 0.03–0.05
(`beamng-authoring-census.md`). Modes between 1.87 and 2 then run in an engine
whose damping is centred in time and diverge in ours.

**What BeamNG publishes (searched 2026-11-06; nothing on its integrator).**
* The beam law is Hooke's law with a dashpot, F = −kx − cẋ, run "up to 2KHz" for
  stability (Gambi, BeamNG, Panichella, ICSE-SEIP [G]). The BeamNG.tech
  technical paper [BT] says nothing on the integrator.
* Instability is "a beamSpring and/or beamDamp values that are too high for the
  nodeWeight", fixed by "lowering beamSpring/Damp or raising nodeWeight"; the
  worst case shows an "instability detected" message [CI]. So BeamNG has a
  stability limit of its own and no automatic stabilisation, mass scaling or
  minimum node weight is documented ([N]: default `nodeWeight` 25 kg as of
  0.40.2, no minimum). Too much beamDamp also explodes ([I], [P]): its damping is
  explicit with a damping limit, as the plain law here.
* `dampCutoffHz` exists to "run higher damping before instability" (§6, [PN]).
* No public source gives the time level of the damping velocity, a
  `beamDampVelocitySplit` limit beyond §22, or a per-node spring/mass limit.
* [N] documents one mass rule the importer drops: a fuel tank's mass is spread
  over the nodes of its `fuel` engine group. The SUV's tank nodes are not the
  ones that vibrate (`rx2l`), so it is not the cause; implemented since (§40).

So the law is documented and ours matches it (§29.1); the time level of the
velocity in the dashpot is not. Centred damping is the standard explicit answer
from molecular dynamics with velocity-dependent forces: the dashpot reads a
velocity predicted half a step ahead, v + ½·h·F/m (Groot and Warren's modified
velocity Verlet for dissipative particle dynamics, λ = ½ [GW]).

**Rule.** After the node update (each node v₁ = v₀ + h·F/m, x₁ = x₀ + h·v₁), with
Δvᵢ = v₁ − v₀ every JBeam record whose last beam pass applied damping of slope s
(∂(damping force)/∂u at that pass's rate, the `dampCutoffHz` gain included: c, the
fast rate past the velocity split, the limit damping in the bound zone, ×0.1 for a
rope in compression, 0 for a support in tension or a broken record; a torsion
bar's c, a coupler's c) adds −½·s·Δu to its ends along its current direction (Δu
= (Δv_a − Δv_b)·d̂; a torsion bar −½·s·Δφ̇ through its dihedral gradient, a
coupler −½·s·(Δv_a − Δv_b)), equal and opposite. Each node sums these in its
incidence order and takes v += h·c/m, x += h·(h·c/m). The damping the step applied
then reads ½(v₀ + v₁): the mean of the step's start and end velocities.
* Linear in its inputs; momentum is exact (every record's correction sums to zero).
* Breaks, yield and the filter state are decided by the beam pass as before (at
  the start velocity); only the motion is corrected.
* RoR (`.truck`) kinds take no correction.
* Cost on the CPU: one pass over the records and one over the nodes, +10 % per
  step on the ETK (237 → 262 µs, 17 configurations) and +13 % on the CCF (289 →
  327 µs, 26), the minimum of four alternating runs at rest (M3 Max). The
  dry-pass form of the same rule (two full beam passes) costs about three times
  that. On the GPU two more phases per step with three barriers (not timed: the
  GPU is shared). Off, nothing runs and nothing changes.

**One mode** (x″ = −ω²x − (c/m)x′, Ω = ω·h, γ = c·h/m = 2ζΩ; the step map's
spectral radius, scratchpad `mode1.py`):

| ζ | current: largest stable Ω | damping on the predicted end velocity | backward-Euler damping (local) | **centred** |
|---|---|---|---|---|
| 0.01 | 1.980 | 2.021 | 2.020 | 2.000 |
| 0.03 | 1.941 | 2.070 | 2.061 | 2.004 |
| 0.07 | 1.865 | 2.213 | 2.145 | 2.023 |
| 0.15 | 1.722 | 3.333 | 2.322 | 2.148 |

| Ω | current: largest stable γ | predicted end velocity | backward Euler | **centred** |
|---|---|---|---|---|
| 0.1 | 1.995 | 1.000 | unbounded | 2.000 |
| 1.0 | 1.500 | 1.000 | unbounded | 2.000 |
| 1.9 | 0.195 | 1.000 | unbounded | 1.998 |

Centred damping never lowers the limit below 2 and keeps the plain law's damping
budget (γ < 2 at every Ω), where the predicted-end-velocity schemes halve it (γ < 1:
nodes whose beams' summed c·h/m lies between 1 and 2, which run now, would diverge).

**Measured** (CPU, `community_sweep`, JBeam mode on raycast wheels, 1 s at rest;
clean: no beam broken or bent, fastest node < 1 m/s):

| vehicle | configs | current: clean / marginal / broken | predicted end velocity (dry pass) | centred (dry pass, 2 beam passes) | **centred** |
|---|---|---|---|---|---|
| Full-size SUV | 22 | 2 / 20 / 0 | (SUV and trailer: 0 clean, arctic diverges) | 22 / 0 / 0 | 22 / 0 / 0 |
| Boat trailer | 8 | 0 / 8 / 0 | | 0 / 8 / 0 | 0 / 8 / 0 |
| Hirochi CCF | 43 | 26 / 9 / 8 | | 39 / 4 / 0 | 39 / 4 / 0 |
| Gavril Vertex | 106 | 97 / 9 / 0 | | 97 / 9 / 0 | 96 / 10 / 0 |
| ETK K-series | 17 | 17 / 0 / 0 | | 17 / 0 / 0 | 17 / 0 / 0 |
| 48 ft yacht | 2 | 2 / 0 / 0 | | 2 / 0 / 0 | 2 / 0 / 0 |
| **all** | 198 | 144 / 46 / 8 | | 177 / 21 / 0 | 176 / 22 / 0 |

* The SUV: 4.7–20.4 m/s and 5–28 bent beams at rest → 0.19–0.67 m/s, none bent.
* The trailer: 20.7 m/s, 23 kJ → 0.16–0.28 m/s, 1.5–13 J; its 14 bent beams are
  a settle of its leaf springs (30 before), not a vibration.
* The CCF: every broken configuration rests (`country_M` 22.6 m/s and 16 broken
  → 0.06 m/s; `f4_cup_*` 27 m/s → 0.20; `f6_offroadster_M` 29 m/s, 18 broken →
  0.07; `f6_tuned_M` 2 broken → 0).
* The Vertex: no configuration changes class except `offroader` (5 bent → clean)
  and `hohoho_its_christmas_time` (0.98 → 1.98 m/s, a loose prop at 1 s; KE 4.6 J);
  `hocus_pocus…` stays at 9.8 m/s both ways (a loose prop).
* The predicted-end-velocity variant is what "damping applied with the end-of-step
  velocity" gives explicitly; it fails as the analysis says (the arctic SUV, clean
  before, diverges).

**Crash scenes** (`vehicle_compare`, CPU, RUBBLE_DAMP_SCHEME=4 against 0): wall56,
1 s after impact:

| car | rest: KE, fastest node | broken / bent | crush dynamic / residual | peak force | restitution | beam dissipation | energy closure |
|---|---|---|---|---|---|---|---|
| ETK kr6 | 0.5 J, 0.148 → 0.0 J, 0.001 m/s | 2 / 338 → 2 / 332 | 489 / 193 → 486 / 188 mm | 712 → 755 kN | 0.35 → 0.37 | 122.9 → 120.9 kJ | 2·10⁻⁵ → −1·10⁻⁵ |
| CCF f4_sport_M | 0.0 J, 0.131 → 0.001 m/s | 141 / 462 → 142 / 459 | 474 / 6 → 473 / 75 mm | 581 → 581 kN | 0.27 → 0.27 | 152.0 → 152.2 kJ | 1·10⁻⁶ → 0 |
| Vertex | 0.0 J, 0.001 m/s both | 31 / 438 → 40 / 436 | 479 / 118 → 479 / 117 mm | 542 → 582 kN | 0.22 → 0.20 | 111.0 → 111.8 kJ | −7·10⁻⁶ → −8·10⁻⁶ |

The crash itself is the same within its run-to-run spread (break counts of torn
fascia are chaotic, §29.1); the residual vibration of the parked cars goes. The CCF's
residual crush (6 → 75 mm) is its springback (0.94 before): the front settles
instead of ringing back. The energy ledger closes as before.

**GPU.** `slang/kernels/explicit_beams.slang` (`FLAG_CENTRED`, `centredPhase`,
`centredBeam`, `centredNode`; the slope in the first incidence entry's `.w`, a
per-node Δv region at `gP.centredDv`) in the vehicle workgroups, the contact
groups and the wide passes (the node loops keep each node's Δv; two more passes per
step, `wide_centred_beams` and `wide_centred_nodes`, only when a vehicle has it).
Metal (`engine-metal/tests/explicit_centred_damping.rs`) as wgpu: the fixture car
at rest and the latched box into a wall bit for bit on the frame and the wide
passes.

GPU cost (`centred_damping_frame_cost`, `RUBBLE_CENTRED_TIMING=256`: 256 fixture
cars parked apart, M3 Max shared with other work, median 60 fps frame of 240,
the four configurations interleaved frame by frame):

| path | off | centred | |
|---|---|---|---|
| wide passes | 36.9 ms | 42.0 ms | +14 % |
| vehicle workgroups | 10.3 ms | 12.2 ms | +19 % |

**Where it goes, and what did not cut it (2026-11-06).** The same measurement,
three rounds of two binaries alternating (each round its own interleaved off/on),
with kernel variants that skip one part (measurement builds, not kept):

| variant | vehicle workgroups | wide passes |
|---|---|---|
| as shipped | +23.0 … +24.9 % | +8.3 … +9.0 % |
| the records' corrections skipped | +6.9 … +7.5 % | +6.8 … +7.0 % |
| the nodes' sums skipped | +20.6 … +21.2 % | +4.9 … +5.3 % |
| both skipped (barriers, Δv) | +2.7 … +3.3 % | +4.8 … +5.0 % |

So on the vehicle workgroups the records' corrections are three quarters of the
cost, the nodes' sums a sixth; on the wide passes the two extra dispatches are
half. (The earlier +14 % of the wide passes did not repeat: +8–9 % in every
later round.) Bit-identical changes tried:
* the step-start velocity in a register in the node loop (no Δv buffer round
  trip), the barrier after the nodes' sums dropped (what follows reads a node only
  after its own barrier), the beam phase's flag read skipped when no vehicle has
  the flag: kept (simpler, no slower), +23.7 … +24.2 % / +7.2 … +8.0 %: within the
  noise;
* each record's correction in its own coalesced entries (no incidence lookups in
  the records' pass) and the nodes gathering them through their incidence lists:
  worse, +39 % / +19 %, not kept.
The rest is the scheme's work itself (one more pass over the records with their
lengths at the new positions, one over the nodes); a cheaper form would change
the results.

(The community cars do not take the wide passes: they carry commands or shocks2.) `tests/explicit_centred_damping_gpu.rs`: bit for bit every
step against the CPU for the pressure-wheel fixture car at rest (4,000 steps) and at
10 m/s (2,000), the latched box into a wall (1,500 steps, breaks), the break-group
fixtures pulled apart, the node couplers; opt-in the full-size SUV on its hubs
(2,000 steps bit for bit, fastest node 0.30 m/s). Off: the existing parity tests
pass unchanged, and `jbeam_accept` gives the same output as before this change
(ETK, CCF, SUV, trailer; raycast and pressure wheels).

Sources:
* [F] BeamNG forum, Goosah (BeamNG team, global moderator), 2018-07-25/26:
  https://www.beamng.com/goto/post?id=898127
* [G] A. Gambi, BeamNG GmbH, S. Panichella, soft-body simulation for CPS testing
  (ICSE-SEIP): https://beamng.tech/research/ (PDF in `inspiration/research-explicit-stability/`)
* [BT] BeamNG.tech technical paper (2021):
  https://beamng.tech/blog/2021-06-21-beamng-tech-whitepaper/bng_technical_paper.pdf
* [P] BeamNG soft-body physics page: https://beamng.com/game/about/physics
* [GW] R. D. Groot, P. B. Warren, "Dissipative particle dynamics: bridging the gap
  between atomistic and mesoscopic simulation", J. Chem. Phys. 107, 4423 (1997):
  the modified velocity Verlet with the dissipative force at a predicted velocity
  (λ = ½).
* [CI], [I], [N], [PN] as in the header.

### 39.1 The BeamNG profile's default (2026-11-06)

`beamng()` turns centred damping on; the Rigs of Rods profile is unchanged, bit for
bit. Re-run with the fuel-tank mass of §40 in both columns (it moves the community
cars' masses, so "off" here is not §39's "current"):

* **Community sweep** (198 configurations, 1 s at rest): off 141 / 49 / 8 clean /
  marginal / broken, **on 176 / 22 / 0** (the same classes as §39's centred column;
  the Vertex's `powerdemonrwd` 1.01 → 0.79 m/s, `taxi_taxi_carrier` 0.96 → 1.00).
  Re-run on `merge/showcase-main` with the complete energy ledger and the yield fix
  (feat/jbeam-community 81283d5e) and surface contact r2: off 144 / 46 / 8, **on
  176 / 22 / 0**.
* **`vehicle_compare`** (sedan1, ETK kr6 and kc6, CCF, Vertex;
  `engine/docs/measurements/vehicle-compare.{md,json}` is the new baseline, the old
  one had sedan1 and the ETKs only). Every reference that moved, off → on:

| car | measure | off | on |
|---|---|---|---|
| sedan1 | wall56 bent beams | 974 | 980 |
| sedan1 | wall56 crush dynamic / residual | 808.5 / 614.5 mm | 809.6 / 630.6 mm |
| sedan1 | wall56 peak force, restitution, Campbell b1 | 935.7 kN, 0.189, 21.7 | 929.1 kN, 0.188, 21.1 |
| sedan1 | pole32 crush dynamic / residual | 398.9 / 318.4 mm | 396.2 / 298.5 mm |
| sedan1 | MDB crush dynamic / residual | 665.1 / 552.3 mm | 656.5 / 562.9 mm |
| sedan1 | roof peak, door peak | 68.5, 24.1 kN | 67.5, 24.7 kN |
| ETK kr6 | rest KE, fastest node | 0.5 J, 0.148 m/s | 0.0 J, 0.001 m/s |
| ETK kr6 | wall56 bent; crush dynamic / residual | 335; 501.0 / 197.5 mm | 337; 497.0 / 193.6 mm |
| ETK kr6 | wall56 peak, restitution, b1 | 714.5 kN, 0.352, 67.5 | 765.5 kN, 0.362, 68.9 |
| ETK kr6 | pole32 dynamic / residual | 166.4 / 62.2 mm | 162.9 / 53.9 mm |
| ETK kr6 | MDB dynamic / residual | 471.0 / 293.5 mm | 467.3 / 228.2 mm |
| ETK kr6 | roof, door peak | 84.3, 38.5 kN | 84.7, 38.3 kN |
| ETK kc6 | rest KE, fastest node | 0.5 J, 0.148 m/s | 0.0 J, 0.001 m/s |
| ETK kc6 | wall56 bent; dynamic / residual | 331; 497.1 / 196.1 mm | 338; 495.4 / 194.5 mm |
| ETK kc6 | wall56 peak, restitution, b1 | 698.5 kN, 0.338, 68.0 | 760.2 kN, 0.360, 68.5 |
| ETK kc6 | pole32 dynamic / residual | 103.5 / 57.3 mm | 98.8 / 55.0 mm |
| ETK kc6 | MDB dynamic / residual | 474.6 / 300.3 mm | 466.6 / 230.8 mm |
| ETK kc6 | roof, door peak | 133.9, 38.2 kN | 133.4, 86.5 kN |
| CCF | rest fastest node | 0.062 m/s | 0.001 m/s |
| CCF | wall56 broken / bent | 140 / 474 | 142 / 476 |
| CCF | wall56 dynamic / residual | 481.4 / 240.7 mm | 479.5 / 19.9 mm |
| CCF | wall56 peak, restitution, b1 | 604.0 kN, 0.264, 55.4 | 585.2 kN, 0.276, 669.9 |
| CCF | pole32 dynamic / residual | 364.0 / 284.7 mm | 364.6 / 286.0 mm |
| CCF | MDB dynamic / residual | 306.2 / 127.3 mm | 300.6 / 127.2 mm |
| CCF | roof, door peak | 103.1, 189.1 kN | 109.2, 375.0 kN |
| Vertex | wall56 broken / bent | 33 / 453 | 45 / 444 |
| Vertex | wall56 dynamic / residual | 488.9 / 121.7 mm | 488.4 / 124.9 mm |
| Vertex | wall56 peak, restitution, b1 | 558.0 kN, 0.214, 109.5 | 576.8 kN, 0.201, 106.7 |
| Vertex | pole32 dynamic / residual | 266.8 / 198.6 mm | 301.1 / 238.7 mm |
| Vertex | MDB dynamic / residual | 331.5 / 241.5 mm | 323.8 / 234.7 mm |
| Vertex | roof, door peak | 17.1, 54.6 kN | 16.8, 60.7 kN |

  Readings: the parked cars stop ringing (0.15 → 0.001 m/s); dynamic crush moves
  by 0–2 % (the Vertex pole 13 %); the residual crushes and the door pulls move
  where a path branches. The ETK kc6's and the CCF's door pulls follow the off
  curves to the hinges (the same forces to the 10 mm sample) and then take
  another break order (the kc6 loses its soft top's front mount too), so their
  peaks are a different event, not a stiffer door. The CCF's wall residual (241 →
  20 mm, so b1 670) is the front springing back to its shape: §39's single run
  read 6 → 75 mm for the same car without the fuel mass. Crush residuals at this
  car's springback are not a stable reference.
* **sedan1's profile suites** (`showcase_car`) and `crash_invariants`: off and on
  run the same 30 and 9 tests. `crash_invariants` passes both ways. `showcase_car`
  fails 6 tests off and 5 on, all of them on the base branch too (after its
  compile fix): `f_wall_at_120_kmh…` (energy rose 13.0 J between tenths, both),
  `g_pulling_a_door_tears_it_off` (the latch at 78.1 kN, not 30.0–64.3, both),
  `t_car_to_car_t_bone_at_80_kmh` (13 of 1,165 striker nodes in the skin, both),
  `n_wall_matches_the_nhtsa_sedans` (restitution 0.2018, not 0.13–0.20, both),
  `contacts_do_no_positive_work` (0.38 J off, 0.23 J on, over a tenth); and
  off only `fixed_contact_quantisation_does_no_work` (0.0102 J), which passes on.
  None is caused by this change; they are follow-ups.

## 40. Fuel-tank mass (2026-11-06)

What BeamNG documents: [N] "`fuel`: These nodes contain the mass of the fuel inside
the vehicle's fuel tank, which will be spread evenly between them and added to
their nodeWeight"; a vehicle can have several tanks, each with its own engine group.
[ES] (https://documentation.beamng.com/modding/vehicle/sections/energystorage/,
retrieved 2026-11-06): a `fuelTank` row of `energyStorage` names a table with
`energyType` ("Presets for energy density and weight density": gasoline, diesel,
kerosene; any other needs `energyDensity` and `fuelLiquidDensity`),
`fuelLiquidDensity` (kg/L), `fuelCapacity` (L, default 0), `startingFuelCapacity`
(L, default `fuelCapacity`, "usually tied to a variable") and `fuel` (the nodes, e.g.
`{"[engineGroup]:": ["fuel"]}`); the nodes "share the fuel's weight equally".

**Rule** (`jbeam.rs` `fuel_mass`): for each `fuelTank`, starting volume ×
density, spread evenly over the nodes whose `engineGroup` (their last definition's)
is one of the tank's `fuel` groups, added to their `nodeWeight` at load. **Choice**:
BeamNG names the presets but not their densities; gasoline 0.74, diesel 0.84,
kerosene 0.80 kg/L (typical values); an `energyType` other than these without
`fuelLiquidDensity`, or a tank whose groups have no node, adds nothing and warns.
Other tank types (N₂O, air) are not simulated. Content without a `fuelTank` loads
as before, bit for bit. The full-size SUV's default 90 L of petrol is 66.6 kg on its
tank's nodes.
