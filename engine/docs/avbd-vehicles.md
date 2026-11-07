# AVBD vehicles: an opt-in implicit mode for the explicit node–beam cars

Status: prototype and measurement (2026-11-07), branch `feat/avbd-vehicles`. The
explicit 2 kHz solver stays the reference and the default; nothing here changes it.

The question: can the BeamNG-style vehicles (`engine_softbody::explicit_beams`, JBeam
and `.truck` content) run on Rubble's AVBD solver at the World's rate instead of the
explicit 2 kHz step, so vehicles scale across the GPU, and still look right? The bar
(the user, 2026-11-07): it need not match the explicit solver exactly, but it must be
close and must never look like jello; a viewer must not be able to tell something is
off. A visible softness or wobble is a failure whatever the numbers say.

**Answer (§7): not viable at a cost that beats the explicit solver.** At the sweep
counts that would make it cheaper, the cars sag and recover their shape like jello and
crash far too softly; at sweep counts where rest and the crush come close, a step costs
more than the explicit 2 kHz frame it replaces, and it still breaks parts off that the
explicit car keeps. §7 lists what would have to change for AVBD vehicles to be revisited.

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

RESULTS

## 6. GPU cost

GPU

## 7. Verdict and recommendation

VERDICT
