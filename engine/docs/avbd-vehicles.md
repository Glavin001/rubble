# AVBD vehicles: an opt-in implicit mode for the explicit node–beam cars

Status: prototype and measurement (2026-11-07), branch `feat/avbd-vehicles`. The
explicit 2 kHz solver stays the reference and the default; nothing here changes it.

The question: can the BeamNG-style vehicles (`engine_softbody::explicit_beams`, JBeam
and `.truck` content) run on Rubble's AVBD solver at the World's rate instead of the
explicit 2 kHz step, so vehicles scale across the GPU, and still look right? The bar
(the user, 2026-11-07): it need not match the explicit solver exactly, but it must be
close and must never look like jello; a viewer must not be able to tell something is
off. A visible softness or wobble is a failure whatever the numbers say.

**Answer (§7): not viable for crashing cars; possible only as a parked/cruising LOD,
and even there not cheaper.** At the sweep counts that are cheaper than the explicit
frame on the GPU (120 Hz × 10–20), the cars sit 55–80 mm lower than the explicit car,
keep 20 mm of body deformation after a landing and crush 30–60 % deeper with no
springback: soft, the failure the bar names. At 240 Hz × 50 and above, rest and the
landing match the explicit car (sag, shape, landing deformation within a few mm),
but every crash still differs visibly: no springback, 6–25× the broken beams (doors,
headlights, bumpers and fenders come off a 56 km/h frontal the explicit ETK keeps), the
car slides past or into the pole. Those settings cost several times the explicit
frame on the GPU (§6). §7 lists what would have to change to revisit it.

## 1. What exists

* **The explicit solver** (`explicit_beams.rs`, `slang/kernels/explicit_beams.slang`):
  node–beam vehicles at a fixed 0.5 ms step, the JBeam semantics of
  `jbeam-semantics-spec.md`, coupled two-way to the World's rigid bodies
  (`explicit-vehicles-in-world.md`). The reference.
* **Rubble's AVBD soft-body path** (`slang/kernels/soft_step.slang`, CPU reference
  `engine-softbody/src/cpu.rs`, `softbody.html`'s "soft" mode): the same node–beam
  vehicles as `BODY_PARTICLE` bodies inside the World's AVBD step. Per iteration the
  rigid colours, then a VBD block solve per particle colour (inertia, contacts and
  joints through `primalAccumulate`, beams from the soft arena), Chebyshev, dual
  updates; after the iterations `soft_plastic` applies yield and breaking at the
  converged state and `soft_break_groups` the groups. It covers plain, support, rope,
  bounded, hydro and anisotropic beams, LBEAMs, torsion bars, rails and slidenodes,
  node–triangle self-contact with anti-clip, vehicle-local frames, BDF2 and crash
  substeps (`softbody-beamng-parity.md`). Contact with the World is the World's own:
  the particles are AVBD bodies.
* **What was already known** (`softbody-beamng-parity.md` phases 0, 5, 9): against an
  f64 explicit oracle, the sedan at 240 Hz × 10 over-crushes a 56 km/h wall by about
  60 % and taps at 25 Hz against 46; 960 Hz × 30 matches. The 1,014-node car does not
  converge at real-time settings. That is why the vehicles moved to the explicit
  solver.

This study maps the *explicit* vehicles (their JBeam semantics, as they ship) onto an
AVBD step, on the CPU first, measures it against the explicit reference on the same
cars and scenes, and times the existing GPU AVBD path on the same car.

## 2. The mapping (`engine-softbody/src/avbd_beams.rs`)

Opt-in per vehicle: `ExplicitBeams::avbd = Some(AvbdState)` (`AvbdSettings`: rate,
sweeps). With it `None` (the default) `ExplicitBeams::step` is the explicit step,
unchanged (the mode is one branch at its top). Content with an element the mapping
does not cover fails `AvbdState::new` with an error naming it (raycast wheels, wings,
rigid proxies, the tread contact, LBEAMs, pressured beams, pressure groups, shocks2 and
hydro commands are not mapped in the prototype).

One step of length H (backward Euler: the VBD/AVBD energy
½‖x − y‖²_M/H² + Σ E_e(x)):

| JBeam / explicit semantics | AVBD step |
|---|---|
| Beams with spring and damping (`JbeamPlain`, `Plain`), §3 | A two-node element in both nodes' 3×3 blocks. Its force is the explicit law itself (`jbeam_force_filtered`, `jbeam_ext_force_with` for records with extensions, `beam_force`) evaluated on a copy at the iterate with the velocity (x − x₀)/H, so the damping is implicit. Its stiffness is the law's slope along the beam (a finite difference, which includes c/H) plus the tension's geometric stiffness T/ℓ(I − êêᵀ). |
| Plasticity (`beamDeform`, hardening to `deformLimitStress`), §3 | Inside the solve the law on the copy caps the force at the yield force, so a yielding beam is perfectly plastic in the iteration (slope 0). The rest length commits after the step (below). |
| `deformLimit`, `deformLimitExpansion` | The law's own clamp of the rest length, at the commit. |
| Breaking (`beamStrength`), break groups §8, §17, deform groups §19, torsion watches | Inside the solve a copy past its strength carries no force. After the step the explicit beam pass runs once at the converged state (positions x, velocities (x − x₀)/H) with its forces discarded: yield, limits, breaks, groups and deform groups commit by exactly the explicit rules (the soft path's `soft_plastic` placement). Forward-only. |
| SUPPORT (§4), ROPE, BOUNDED (§5, §22), ANISOTROPIC (§20) | The same element: the law's one-sidedness, limit springs, velocity split and expansion rates come with the law; the slope follows them. |
| Couplers (§7) | A zero-length isotropic element: force from `coupler_force`, stiffness (k + c/H)·I. |
| Torsion bars (§9) | A four-node element: force from `torsion_force` on a copy, per-node diagonal stiffness from finite differences (clamped ≥ 0). |
| Slidenodes and rails (§27) | Segment chosen at the step start by the explicit rail pass (forces discarded), then the rail spring as a three-node implicit element (finite-difference diagonal). Taken explicitly at H it blew a 5 kg node off at 685 m/s within three steps. |
| Triangle aerodynamics (§31), air drag | Explicit forces at the step start (small, velocity-only). |
| Ground, static boxes and shapes | Augmented-Lagrangian inequality constraints, one per node and surface, found at the iterate: penalty 10³·m/H², multiplier updated after each sweep. Friction: the explicit friction law's magnitude (`contact_friction`'s curve, adhesion ms(1 − e^(−σ/va)) under va, the sliding curve above) at the step's slip speed, against the slip, its secant as the stiffness (lagged normal force). |
| Self-contact (§10, §36, §37) | Node–triangle inequality constraints found at the step start (node within the band plus its travel, projection inside the triangle; self-groups, spawn exclusions, anti-clip-only and dead triangles honoured): the node is kept on the side it started, at most at the contact range (no positive work, §37's intent), the reaction spread to the triangle's nodes by its barycentric weights. Frictionless in the prototype. |
| Contact with the World's rigid bodies | Not in the CPU prototype. On the GPU it is free: the AVBD particles are World bodies (`soft_step.slang`), simpler than the explicit coupling's proxies and mass splitting. |
| Vehicle–vehicle contact | Not in the prototype (the soft path has none between bodies' triangles beyond node contacts). |

## 3. Step rate and sweeps

The World steps at 60 or 120 Hz. The mode's own rate H is 60, 120 or 240 Hz (an
integer number of mode steps per World step), with N Gauss–Seidel sweeps over the
node colours per step. Measured: 60, 120 and 240 Hz at 20, 50 and 100 sweeps.

Why the numbers are what they are: a beam of stiffness k between nodes of mass m has
k·H²/m (stiffness over inertia in the node's block). On the ETK the median is 46 at
120 Hz (k ≈ 9.5·10⁵ N/m, m ≈ 1 kg; `avbd_census`), the p90 beam ω = √(k/m) is
1,500 rad/s. With stiffness dominating inertia by 10–50× per node, a sweep moves a
correction about one beam hop, and the car is 20–30 hops long. The low modes the eye
sees (the body's bending and twist, the sag onto the suspension, the front structure's
crush) need on the order of the hop count squared in sweeps to converge with plain
Gauss–Seidel; fewer sweeps leave them soft.

## 4. Against jello: what was tried

* **Material stiffness inside the solve.** Beams are not softened: their law and slope
  are the explicit ones. (The soft path's penalty ramp, k ← min(k*, k + β|C|), starts
  soft and climbs; at few iterations it is softer still.)
* **Hard constraints (augmented Lagrangian) for contacts**, so contact does not add
  compliance; the beams stay elastic as authored. Making stiff structural beams hard
  constraints would change the material (they yield and break by force); it was not
  adopted, and it does not change the convergence argument of §3 (a multiplier needs
  the same sweeps to propagate).
* **More sweeps and a higher rate** (measured, §5): the only lever that works.
* **Chebyshev acceleration** (ρ = 0.9 at 120 Hz × 50): diverged in the wall crash
  (NaN). Not usable with contacts and yielding beams without per-node safeguards.
* **Mass ratios.** The block solve is per node, so a 22 kg engine node beside 0.25 kg
  skin nodes is handled exactly in each block; the ratio does not destabilise, it slows
  the propagation through the light nodes further.
* **Divergence guard.** One node's correction per sweep is capped (0.2 m, counted in
  `AvbdStats::clamped`); without it the sedan's and the ETK's walls diverged in some
  settings.

## 5. Measurements (CPU reference against the explicit solver)

Harness: `examples/avbd_vehicles.rs` (rest from still for 2 s; a landing: every node
given 1 m/s down; the vehicle-compare harness's own `scene_wall` 56 km/h and
`scene_pole` 32 km/h with every metric defined there). Same car template, same scene,
explicit 2 kHz against the AVBD mode. Raw JSON: `engine/docs/measurements/avbd-vehicles/`.

Cars: the ETK K-series kc6 (community JBeam, BeamNG profile, 585 nodes, 4,304 beam
records; local `__mods/etkk_tex`, never committed) and the repo's authored `sedan1`
(1,026 nodes, 8,926 records). Configs `<Hz>x<sweeps>`. Columns: rest from still for 2 s
(COM sag, RMS of the body nodes' displacement about their rigid fit, COM drift on the
ground); a landing (1 m/s down) over 1.5 s (body deformation peak and at the end, COM
dip); the 56 km/h full-width wall (dynamic and residual crush, springback ratio, peak
cabin g at 100 Hz, restitution, broken beams / hardware groups broken, KE left at 1 s);
the 32 km/h 75° pole (dynamic and residual crush at the pole, broken beams). Explicit
first. Generated by `engine/tools/avbd-vehicles-table.py`.

#### ETK kc6

| config | rest sag mm | rest shape mm | rest drift mm | landing deform peak / end mm | landing COM dip mm | wall dyn / residual crush mm | springback | peak g | restitution | wall broken beams / hardware | wall KE end kJ | pole dyn / residual mm | pole broken | CPU s per sim s (wall) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| explicit | 18.4 | 6.1 | 106 | 20.3 / 5.6 | 66.5 | 526 / 232 | 0.56 | 69.4 | 0.36 | 3 / 1 | 14.9 | 99 / 55 | 8 | 0.71 |
| 60x20 | 89.2 | 23.5 | 555 | 32.5 / 23.8 | 101.9 | 1144 / 1114 | 0.03 | 51.0 | 0.06 | 234 / 24 | 0.5 | 49 / 31 (tunnelled) | 57 | 0.64 |
| 60x100 | 65.2 | 17.7 | 314 | 23.9 / 17.7 | 86.7 | 315 / 284 | 0.10 | 111.1 | 0.10 | 128 / 13 | 0.4 | 65 / 0 | 46 | 2.74 |
| 120x20 | 73.7 | 19.5 | 338 | 25.7 / 19.5 | 91.0 | 676 / 793 | -0.17 | 46.5 | 0.11 | 156 / 16 | 0.8 | 2385 / 193 | 81 | 1.23 |
| 120x50 | 42.7 | 12.0 | 259 | 23.1 / 10.9 | 83.8 | 402 / 346 | 0.14 | 71.3 | 0.17 | 125 / 12 | 1.2 | 84 / 61 | 34 | 2.72 |
| 120x100 | 22.7 | 6.0 | 184 | 21.0 / 5.3 | 73.3 | 311 / 221 | 0.29 | 88.0 | 0.16 | 107 / 10 | 0.5 | 102 / 62 | 25 | 10.99 |
| 240x20 | 26.0 | 6.8 | 211 | 22.6 / 6.1 | 80.3 | 485 / 524 | -0.08 | 51.2 | 0.19 | 182 / 17 | 1.9 | 164 / 8 | 87 | 2.36 |
| 240x50 | 13.9 | 3.3 | 94 | 19.7 / 4.5 | 67.3 | 388 / 288 | 0.26 | 61.3 | 0.22 | 99 / 9 | 0.7 | 178 / 132 | 19 | 5.29 |
| 240x100 | 16.3 | 4.4 | 27 | 19.0 / 4.2 | 64.3 | 374 / 374 | 0.00 | 62.6 | 0.12 | 74 / 6 | 2.3 | 144 / 100 | 43 | 10.35 |

#### sedan1

| config | rest sag mm | rest shape mm | rest drift mm | landing deform peak / end mm | landing COM dip mm | wall dyn / residual crush mm | springback | peak g | restitution | wall broken beams / hardware | wall KE end kJ | pole dyn / residual mm | pole broken | CPU s per sim s (wall) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| explicit | -1.3 | 0.4 | 13 | 11.3 / 3.5 | 69.6 | 842 / 651 | 0.23 | 34.3 | 0.18 | 13 / 0 | 0.3 | 393 / 304 | 0 | 1.17 |
| 120x20 | 80.4 | 23.6 | 47 | 27.9 / 24.6 | 113.5 | 1131 / 1124 | 0.01 | 26.8 | 0.09 | 86 / 0 | 0.7 | 699 / 481 | 4 | 2.84 |
| 120x50 | 63.8 | 14.2 | 11 | 17.7 / 15.3 | 85.1 | 730 / 726 | 0.01 | 37.1 | 0.10 | 47 / 0 | 0.1 | 480 / 392 | 4 | 5.80 |
| 120x100 | 23.2 | 6.0 | 3 | 12.8 / 7.4 | 74.6 | 554 / 624 | -0.12 | 45.9 | 0.12 | 28 / 0 | 0.0 | 376 / 254 | 7 | 11.00 |
| 240x50 | 4.6 | 1.8 | 6 | 11.6 / 3.5 | 70.4 | 647 / 645 | 0.00 | 30.2 | 0.09 | 102 / 2 | 0.2 | 438 / 328 | 3 | 10.74 |
| 240x100 | 0.2 | 0.5 | 10 | 11.1 / 2.6 | 68.7 | diverged | – | – | – | 5144 / 22 | – | 407 / 323 | 6 | 254 |

Notes on the table:

* **Closest settings.** At rest and on landing 240 × 50 and up match the explicit car
  within a few mm (sedan: 4.6 mm sag against −1.3, 1.8 mm shape against 0.4). The pole
  crush comes close at 120 × 100 (ETK 102 / 62 mm against 99 / 55; sedan 376 / 254
  against 393 / 304), but with 3–7× the broken beams. No setting is close in the wall:
  springback and restitution are always low and the ETK always loses parts.
* **Sedan 240 × 100 wall diverged** even with the guard (more sweeps, not fewer):
  the crash steps are not near a stable fixed point of the sweep.

* **The AVBD mode holds its state between its steps** while the harness samples at
  0.5 ms, so its peak g (Δv over one step, filtered at 100 Hz) is a lower bound.
* **Rest drift**: from still the explicit ETK rolls 106 mm in 2 s on its own wheels
  (its template's polygonal tyres); the AVBD ETK rolls further (94–555 mm) except at
  240 × 100. The sedan holds still on both.
* **Divergence guard**: without the per-sweep cap the sedan's wall diverged at 120 × 20
  and 120 × 50 and the ETK's at 240 × 50; with it every run but the sedan's 240 × 100
  wall stays finite, and the cap fires in crashes (e.g. 220 node corrections in the sedan's 120 × 50 wall), a sign of
  how far from converged the crash steps are (the last sweep still moves nodes by
  13–15 mm on average in the sedan's wall).
* **60 Hz tunnels**: at 60 Hz the ETK passes through the 254 mm pole (pole reach 7.5 m
  at 60 × 20): 8.9 m/s × 1/60 s is 15 cm a step, more than the pole's radius, and the
  node-only contact finds the node already past the pole's middle.
* **Wall stuck nodes** (fixed in the final runs): a node that ended a step deep inside
  the wall box near its bottom was pushed out of the nearest (bottom) face into the
  ground and stayed; the final runs take the face the node entered through
  (`entry_face`).

**What the eye sees** (videos and stills in `~/Development/rubble-artifacts/avbd-vehicles/`,
`engine/tools/avbd-vehicles-videos.sh`; explicit left, AVBD right, 8× slow motion):

* `etk_kc6-bump-…120x20`: at rest and after the landing the AVBD car sits visibly
  lower at the nose and does not come back up (20 mm of body deformation left, against
  6). Soft: fails the bar.
* `etk_kc6-bump-…240x100`: indistinguishable from the explicit car in the stills
  (0, 60, 150 ms and the end; the numbers agree through the run). Passes.

(Judged from the stills each run writes; the videos were rendered for the user to
watch in motion.)
* `etk_kc6-wall-…120x50`: the front folds about as far, but the car stops dead at the
  wall instead of bouncing back 1.5 m, and corner pieces of the bumper hang off on long
  stretched beams. Fails.
* `etk_kc6-wall-…240x100`: rebounds less, headlights and bumper corners torn off and
  left at the wall on stretched beams. Fails.
* `etk_kc6-pole-…120x50`: the explicit car stops on the pole and rebounds; the AVBD
  car slides along it and ends with the pole inside the cabin's outline. Fails badly.
* `sedan1-bump-…120x20`, `sedan1-wall-…120x50`: as the ETK (low, soft; dead stop).

## 6. GPU cost

No new kernel was written: the GPU realisation of this mode is Rubble's existing AVBD
soft path (`soft_step.slang`, every loop already capped), which the prototype maps onto.
`avbd_soft_bench` spawns N ETKs (the soft lowering of the same kc6: 457 particles,
2,767 beams; rigid core off) parked on a ground box and waits for each World step;
`vehicle_scale_bench fleet N` (`VS_CAR=etkk`, vehicle sleep off) runs the explicit ETK
(kr6, raycast wheels) driving in lanes in the World. Both take the GPU lock; M3 Max,
other agents' load present, so p95/p99 are noisy. `engine/tools/avbd-vehicles-gpu.sh`.
ms per 120 Hz World step, p50 / p95:

| cars | AVBD 120 Hz × 10 | × 20 | × 50 | explicit 2 kHz (17 steps) |
|---|---|---|---|---|
| 1 | 4.2 / 7.1 | 6.6 / 11.5 | 50.1 / 56.3 | 7.3 / 8.1 |
| 16 | 4.3 / 6.6 | 7.2 / 10.8 | 54.4 / 60.0 | 8.0 / 8.7 |
| 64 | 7.4 / 13.6 | 11.5 / 22.5 | 69.1 / 78.4 | 16.0 / 48.7 |

(240 Hz × 20, one car: 6.5 ms per 1/240 s step, so 13 ms per 120 Hz World step.)

* At 10–20 sweeps the AVBD path is 1.4–2× cheaper than the explicit frame at 64 cars,
  but those settings are the soft ones of §5 (55–80 mm sag, 30–60 % over-crush).
* The settings that pass at rest (240 Hz × 50, 120 Hz × 100) cost 2 steps × ≥ 50 sweeps
  or 100 sweeps per World step: past 50 sweeps the soft path's step jumps to 50–70 ms
  (1 to 64 cars), 4–9× the explicit frame. There is no speed-up at matched quality.
* Why: a sweep is a pass over every particle colour (the ETK colours into about 15),
  each a dispatch; the explicit step is a beam pass and a node pass. 17 explicit steps
  are about 35 passes of cheap, fully parallel work; 100 sweeps × 15 colours are
  1,500 dependent passes. Gauss–Seidel's propagation (§3) is paid in serial passes.

## 7. Verdict and recommendation

**Settings that pass the bar:** none for crashes. For a car at rest, cruising or
landing, 240 Hz × 50 or more passes (sedan and ETK within a few mm of the explicit
car's sag, shape and landing; the ETK still rolls on its wheels more than the explicit
one below 240 × 100).

**Speed-up:** none at matched quality: those settings cost 4–9× the explicit frame on
the GPU (§6). The cheap settings (120 Hz × 10–20, 1.4–2× faster than explicit at 64
cars) are visibly soft.

**Where AVBD fails:**

1. *Convergence.* Stiffness dominates inertia 10–50× per node (§3); Gauss–Seidel needs
   about the hop count squared in sweeps to stiffen the low modes. Under-converged, the
   body sags and deforms permanently (rest, landing) and crushes too far.
2. *Fracture.* Breaking reads the converged state of a step that is not converged in a
   crash: local strains are wrong, and break groups fire that the explicit car keeps
   (hinges, headlights, bumpers, fenders: 6–25× the broken beams).
3. *Restitution and springback.* Backward Euler takes the elastic rebound out (KE at
   1 s 0.4–2.3 kJ against 14.9; springback 0–0.29 against 0.56). A car that stops dead
   on a wall reads as wrong.
4. *Thin obstacles.* Node-only contact at 60–120 Hz lets the car slide past or into a
   254 mm pole; the explicit step moves 4 mm per step at the same speed.
5. *Robustness.* Without a per-node correction cap, crashes diverged; Chebyshev
   acceleration diverged.

**Recommendation: not viable as the default for authored cars, nor as a crash mode.**
Keep the explicit solver. The one place an implicit step is defensible is an LOD for
cars that are parked or cruising far from the camera with no contact but the ground,
at 240 Hz × 50 or more, and that is not cheaper than the explicit frame on this GPU,
so it buys nothing today; vehicle sleep (`vehicle-sleep`, approved) already makes
parked cars free. The mode stays in the tree as an opt-in measurement path
(`ExplicitBeams::avbd`), off everywhere, so the question can be re-asked cheaply.

**What would have to change to revisit it:** a global or multilevel solve for the
low modes (a coarse rigid-cluster correction or a preconditioned Newton–Krylov step
per vehicle), so sweeps no longer scale with the hop count squared; fracture decided
from a converged estimate (or substepping crash islands at a high rate, which brings
back the explicit cost); and a restitution-preserving integrator (BDF2 or TR-BDF2
already diverged in the soft path's 100 km/h crash, `softbody-beamng-parity.md` §6).
Each is a research project, not a tuning knob.
