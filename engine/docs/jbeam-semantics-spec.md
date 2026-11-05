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

"Frequency in Hz above which damping is not applied" (NORMAL, BOUNDED and LBEAM). The
BX and the legocar put it on damper beams and engine mounts (250–1000 Hz) and on the
legocar's steering dampers (1000 Hz); without it the legocar's front axle oscillated at
the step's Nyquist rate and blew up. **Choice:** the damping uses y, the extension rate
through a first-order low-pass at the cut-off f_c discretised with the bilinear
transform: K = tan(π·f_c·h), a = K/(1 + K), b = (K − 1)/(K + 1),
y_n = a·(u_n + u_{n−1}) − b·y_{n−1}. It has no gain at the Nyquist rate, so a
period-two oscillation is never damped. f_c·h is capped at 0.45 (the pole reaches the
unit circle at 0.5): cut-offs from 900 Hz up act as 900 Hz. The filter state starts at
zero.

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

* e = ℓ − L. e ≤ 0: (k, c). e ≥ z·L: (kₑ, cₑ). Between: σ = e/(z·L) and
  k = k + (kₑ − k)·σ, c likewise (**Choice**: linear; the docs say "ramp"). The force
  is continuous (it is 0 at e = 0 for every σ). Missing `springExpansion` or
  `dampExpansion` take `beamSpring`/`beamDamp` (**Choice**).
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

* **Choice (units N/m, as `beamSpring`):** a beam between id1 and id2 whose rest
  length follows the arms: with a = |x₁ − x₃|, b = |x₂ − x₃| now and θ₀ the rest
  angle at id3, L(t) = √(a² + b² − 2ab·cos θ₀); F = −k·(ℓ − L) − c·u along id1–id2.
  A pure stretch of an arm moves L with it (no force), an angle change does not. The
  pair force conserves momentum and angular momentum.
* θ₀ at spawn: the spawn angle (precompression scales L: θ₀ from p·ℓ₀).
* **Deform. Choice:** past `beamDeform` the rest angle moves so |k·(ℓ − L)| stays at
  the deform force (perfect plasticity); past `beamStrength` it breaks. `dampCutoffHz`
  applies ([B] lists L-beams).
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
* **Beams** (each set's `Spring`, `Damp`, `Deform`, `Strength`, `DampCutoffHz`,
  `Precompression`, missing values from the row's beam defaults):
  * `hubBeam`: hub(k, i) to the other side's axle node; `hubSideBeam`: to its own
    side's axle node (radial spokes);
  * `hubTreadBeam`: hub(0, i)–hub(1, i) and hub(0, i)–hub(1, i + 1);
  * `hubPeripheryBeam`: hub(k, i)–hub(k, i + 1);
  * `wheelSideBeam` (ANISOTROPIC with `wheelSideBeamSpringExpansion`,
    `…DampExpansion`, `wheelSideTransitionZone`): tyre(k, i)–hub(k, i);
  * `wheelSideReinfBeam` (`enableTireSideReinfBeams`, anisotropic likewise):
    tyre(k, i)–hub(k, i ± 1);
  * `wheelReinfBeam` (`enableTireReinfBeams`, unless L-beams): tyre(k, i)–hub(1 − k, i);
  * L-beams (`enableTireLbeams`; `wheelLbeamSpring`… else the side beam's values):
    vertex tyre(k, i) with ends hub(k, i) and tyre(1 − k, i) (lateral), and hub(k, i)
    and tyre(k, i + 1) (shear: torque between hub and tread);
  * `wheelTreadBeam`: tyre(0, i)–tyre(1, i); `wheelTreadReinfBeam`
    (`enableTreadReinfBeams`): tyre(0, i)–tyre(1, i ± 1);
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
  `nodeS`, tyre friction models (`noLoadCoef`… : node friction only),
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
* The group breaks (pressure lost: p = 0 from then on, the records stay) when any of
  its triangles' sides breaks (a beam between two of a triangle's nodes; a tyre
  puncture). **Choice.**

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
off (warning when a rail is uncapped or sets `capStrength`). `fixToRail` is not
simulated: the node keeps its spawn position (warning when false is asked for; the
files place slide nodes on their rails).
