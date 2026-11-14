# RoR vs BeamNG: what to keep, where to diverge

Research note, 2026-11-05. Sources are public docs, issue trackers, forum posts and
papers only. No RoR or BeamNG source code was read. **Fact** marks a claim a cited
source makes. **Inference** marks our own reasoning. "Not found" means no public
source turned up.

Main sources (short keys used below):

- [BN-beams] BeamNG JBeam beams: https://documentation.beamng.com/modding/vehicle/sections/beams/
- [BN-nodes] BeamNG JBeam nodes (collision, friction): https://documentation.beamng.com/modding/vehicle/sections/nodes/
- [BN-tips] JBeam tips and tricks: https://documentation.beamng.com/modding/vehicle/intro_jbeam/jbeamtips/
- [BN-issues] JBeam common issues: https://documentation.beamng.com/modding/vehicle/intro_jbeam/common_issues/
- [BN-flex] Flexbodies: https://documentation.beamng.com/modding/vehicle/sections/flexbodies/
- [BN-coupler] Coupler system: https://documentation.beamng.com/modding/vehicle/sections/nodes/couplers/
- [BN-torsion] Torsionbars: https://documentation.beamng.com/modding/vehicle/sections/torsionbars/
- [BN-debug] Debug tools: https://documentation.beamng.com/modding/vehicle/intro_jbeam/debugtools/
- [BN-step] Tight coupling (step rate): https://documentation.beamng.com/beamng_tech/tools/cosimulationeditor/tight_coupling/
- [BN-tyre2] "A look at tire development in BeamNG (Part 2)", devblog, 2015-03-12: https://beamng.com/game/news/blog/a-look-at-tire-development-in-beamng-part-2
- [BN-0.8] BeamNG.drive 0.8 patch notes, 2016-12-21: https://beamng.com/game/news/patch/version-0-8-rolling-in
- [Maul] Maul, Mueller, Stamatogiannakis, Fischer (BeamNG GmbH) and Gambi, "Soft-body Simulation and Procedural Generation for the Development and Testing of Cyber-physical Systems", ICSE-SEIP: https://beamng.tech/research/ (local copy: `inspiration/research-ror-beamng/`)
- [Arya] Arya, Yao, Davy, Fard, "A novel and fast crash simulation method: Revolutionising racetrack safety barrier analysis" (2024/25): https://www.sciencedirect.com/science/article/pii/S2590123024021133 ; abstract at https://www.researchgate.net/publication/387803690 ; summary at https://beamng.tech/blog/racetrack-safety-analysis-beamngtech/
- [RoR-truck] RoR truck file format: https://docs.rigsofrods.org/vehicle-creation/fileformat-truck/
- [RoR-concepts] RoR vehicle concepts: https://docs.rigsofrods.org/vehicle-creation/vehicle-concepts/
- [RoR-#N] GitHub issue N at https://github.com/RigsOfRods/rigs-of-rods/issues/N

The RoR forum (forum.rigsofrods.org) sits behind a bot check, so it was not read.

## 1. Known RoR physics problems

| Problem | What RoR developers and community say | Fixed? |
|---|---|---|
| **Objects get stuck in each other or pass through** | **Fact:** maintainer ohlidalp calls it "a typical RoR behavior - colliding objects get stuck in each other". He says mesh surfaces "penetrate each other and get stuck or fall through each other" and cites a tricycle falling through a bus floor. Thin or dense collision meshes grab cars. [RoR-#227] | Open since 2015. |
| **Collision is node-against-triangle only, and only for listed contacters** | **Fact:** inter-vehicle contact uses `contacters` nodes against `c`-flagged `submesh` triangles. Contacters "concerns only contacts with other trucks or loads". `set_collision_range` defaults to 2 cm, and raising it to 4 cm makes "penetrations … a lot more difficult". [RoR-truck] **Fact:** collision is slow. Broadphase is "one AABB per actor and narrowphase processes all nodes". [RoR-#3375] Rubble's RoR profile also reproduces a per-triangle throttle that skips idle triangles for up to 12 steps (`explicit-beams.md`, §3). | Open (#3375, 2025). |
| **Plastic deformation springs back ("bounce back")** | **Fact:** the plastic-deformation coefficient in `set_beam_defaults` defaults to 0.0. A contributor explains that 0.0 makes the structure "bounce back" after hitting a wall, and that older cars deform wrongly because the beam defaults changed around 0.37–0.38. [RoR-#1506], [RoR-truck] | Closed as won't-fix. Content must set the coefficient itself. |
| **Spawn instability or explosion** | **Fact:** the docs warn "Excessive spring will result in an unstable chassis", that excessive damping can crash the game, and that the defaults are "overspringy"/underdamped. Minimum node mass is 50 kg unless `minimass` is set. [RoR-truck] Some vehicles collapse or deform on spawn [RoR-#171], [RoR-#2562]. A 2025 parser change made a truck pack explode [RoR-#3298]. | Case by case. |
| **Shaking and jitter** | **Fact:** vehicles shake at very high speed, and the shake persists after slowing [RoR-#1313]. The maintainer points to RoR's split relative/absolute node positions. Camera jitter about 6 km from the origin was traced to float precision [RoR-#3448]. | #1313 is open. #3448 had a fix in progress in 2026. |
| **Light wheels unstable; wheel model** | **Fact:** "Wheel stability is a frequent topic around here". Wheel torque is applied as forces on the outer nodes [RoR-#3189]. The maintainer agrees "RoR does need a new wheel model" [RoR-#2145]. | Open. |
| **Inter-actor beams and forces** | **Fact:** some inter-actor links copy positions between nodes, which the maintainer calls a hazard to numeric stability [RoR-#2974]. | Open. |
| **Flexbody visual artefacts** | **Fact:** each vertex binds to 3 nodes (ref, x, y) plus an offset. Vertices the automatic filter cannot bind "default to `Node 0`" [RoR-#3255]. Deformation runs on the CPU every frame, and normal mapping is disabled to save time [RoR-#2861]. RoR's own docs say submesh panels "deform like if they were made of rubber" [RoR-concepts]. | A manual `forvert` override was added [RoR-truck]. #2861 is open. |
| **Energy gain** | Not found. | n/a |
| **Panels rippling or "jello"** | Not found as an RoR issue. **Inference:** two documented RoR traits push toward it: plastic coefficient 0 (spring-back) and one `set_beam_defaults` line covering a whole section (see §5). | n/a |
| **Stiffness limit from the fixed step** | **Fact:** both projects say spring or damping too high for the node mass makes the vehicle unstable [RoR-truck], [BN-issues]. Neither documents a formula. | Inherent to explicit stepping. |

## 2. How BeamNG differs

**Integrator and step.** **Fact:** physics steps "compute exactly 0.5ms of simulation time (by default)", which is 2 kHz. `-physicsfps` can run it at, for example, 4000 Hz, but "2000Hz is the supported standard". Below 20 FPS it falls into slow motion [BN-step]. Beams are "a parallel mass-spring-damper circuit", F = −kx − cẋ. Nodes are rotation-free point masses. Vehicles run as independent actors on separate threads [Maul]. The integrator itself is **not found**.

**Beam types** [BN-beams], [Maul]:

| Type | Why it exists |
|---|---|
| NORMAL | General structure. |
| SUPPORT | Resists compression only, used "as a limiter to stop parts from clipping". It can break at a set length. |
| BOUNDED | Separate spring and damping for compression vs extension and for low vs high speed. It is expensive, so it is used for dampers and bump stops. |
| ANISOTROPIC | Different stiffness and damping in expansion and compression. Used for tyres, ropes and soft tops. |
| LBEAM | Three nodes, resists angle change. Used for tyre sidewalls and leaf springs. |
| PRESSURED | Force inversely proportional to length. Used for air springs and tyres. |
| HYDRO | Changes length on command, for example steering. |
| BROKEN | Carries no force. Cannot be set in JBeam. |
| Torsionbars (a separate section) | Four nodes, rotational stiffness. Used for anti-roll bars "and to stiffen flat structures without rigidifier nodes" [BN-torsion]. |

**Deformation and plasticity** [BN-beams]:

| Property | Default and meaning |
|---|---|
| `beamSpring` | 4,300,000 N/m |
| `beamDamp` | 580 N·s/m |
| `beamDeform` | 220,000 N yield force |
| `beamStrength` | FLT_MAX (never breaks) |
| `deformLimit` / `deformLimitExpansion` | Cap the permanent change in compression and expansion as a ratio of the original length. Docs call `deformLimitExpansion` the most-used limiter. Staff give typical values of 1.05–1.1 (https://www.beamng.com/posts/1313359). |
| `deformLimitStress` | Caps how far `beamDeform` can grow. **Inference:** this implies the yield force rises as a beam deforms (work hardening). |
| `deformationTriggerRatio` | Triggers deform groups. |
| `dampCutoffHz` | Applies damping only above a cutoff frequency (NORMAL, BOUNDED and LBEAM beams). |

**Collision** [BN-nodes], [BN-tips]:

- **Fact:** collision is node against triangle only, with three priorities: heightmap, static, and dynamic (self and external). Dynamic wins, so "with high enough force a node can be slightly pushed through a solid wall", then it pops back out.
- **Anti-clip forces** near the surface "decide which side of the triangle the node ends up on". `NONCOLLIDABLE` triangles still have anti-clip. Every self-colliding node should have at least NONCOLLIDABLE triangles over it.
- **Geometry rules:** triangle "thickness" is 2.5 cm, changeable with `externalCollisionBias`. A triangle with no collidable node within 10 cm goes inactive. Nodes should sit 3–3.5 cm from triangles.
- **Grouping:** `selfCollision` defaults to false. Nodes in the same `group` don't collide with that group's triangles.
- **History:** the 0.8 patch notes list "Vastly improved vehicle collisions" [BN-0.8].

**Friction and ground** [BN-nodes]:

- **Defaults:** `frictionCoef` 1 (usually 0.5 in practice), separate `slidingFrictionCoef`, Stribeck exponent 1.75.
- **Load sensitivity:** friction falls from `noLoadCoef` to `fullLoadCoef` along `loadSensitivitySlope`.
- **Tyre feel:** `softnessCoef` and `treadCoef`.
- `nodeMaterial` "only affects sound and particle properties, not physics".

**Damping practice** [BN-tips], [BN-issues]:

- **Diagnosis:** vibration is "a sign of too high spring and/or damp values per node weight".
- **Damping-only beams:** for wobble, "add beams purely for damping, going to some other part that isn't directly connected". Give them maximum strength. A spring of 0 is allowed so the beam is not flagged as a duplicate [BN-beams].
- **Flat panels:** "add a rigidifier node offset from the surface and connected to every panel node".

**Break and deform groups** [BN-beams], [BN-tips]:

- `breakGroup` breaks every beam in the group together. `breakGroupType` 1 is broken by the group but never breaks it.
- `deformGroup` switches flexbody materials, for example glass, and can affect the powertrain.
- **Leaking groups:** a group carried into the next part is a known authoring bug. Fix it with a reset row [BN-issues].
- `disableMeshBreaking` and `disableTriangleBreaking` decide what visuals and collision do when a beam breaks.

**Flexbodies** [BN-flex]:

- **Binding:** each vertex maps to "3 or 4 nodes - a center node, VX and VY nodes" (optionally VZ), chosen near right angles. **Fact:** BeamNG adds the optional 4th (VZ) node, which RoR does not have.
- **Smoothing:** none is documented. The only advice is to keep nodes "aligned in a grid of sorts, with no outliers" to avoid spikes.
- **Glass and lights:** handled through deform groups.

**Tyres.** **Fact:** pressureWheels "uses air pressure physics laws to apply forces to the surfaces of the tire" in place of the old precompressed beams. The tyre uses unidirectional node friction, so the friction circle emerges from the simulation. Curves come from an in-engine rig compared against Pacejka 96 [BN-tyre2].

**Panel ripple and solid-feeling metal.** No BeamNG source names "ripple". The documented tools are:

- rigidifier nodes, torsionbars on flat structures and damping-only beams;
- `deformLimitExpansion` against over-stretched beams;
- per-part `beamDeform`/`beamStrength` tuning [BN-tips], [BN-beams], [BN-torsion].

Our own memory of BeamNG content ("realism = parts, not density") agrees.

## 3. Specific divergences

| Topic | RoR | BeamNG | More physical? |
|---|---|---|---|
| **Contact response** | **Fact:** listed contacters against `c` triangles within a 2 cm range [RoR-truck]. In Rubble's RoR profile: a stateless reaction (0.8·vₙ plus a 0.2·p/h push), no memory of which side the node came from, and idle triangles throttled for up to 12 steps (`explicit-beams.md`). | Per-node `collision`/`selfCollision`/`staticCollision` flags, 2.5 cm thickness, anti-clip that decides the side, and NONCOLLIDABLE anti-clip triangles [BN-nodes]. | **Inference:** BeamNG's. Knowing the side prevents the "stuck in each other" failure RoR's maintainer describes [RoR-#227]. Skipping idle triangles can miss a fast first hit. |
| **Damping defaults** | 12,000 N·s/m, with the defaults described as "overspringy"/underdamped. Support beams get 1/10 damping in tension. Shocks suggest c = 2√(mk) [RoR-truck]. | 580 N·s/m, plus damping-only beams and `dampCutoffHz` [BN-beams]. | Neither default is "correct". Damping represents material loss and must suit node mass (RoR minimum 50 kg vs BeamNG default 25 kg [RoR-truck], [BN-nodes]). **Inference:** BeamNG's targeted damping (cutoff-frequency, damping-only beams) is the better tool against high-frequency ripple without making the whole body treacle. |
| **Inverse square root and length precision** | Not documented by RoR. Rubble reproduces the Quake method with one Newton step to match RoR. Its peak relative error is about 1.75×10⁻³ (Lomont 2003; Day, arXiv:2307.15600). | Not found. | **Inference:** exact is better. The error changes with the length's mantissa, so it adds a small spurious force of up to about k·L·1.75×10⁻³ that varies as the beam moves. For k = 9×10⁶ N/m and L = 0.5 m that is about 8 kN, small next to yield forces but not zero, and it jumps where q crosses a power of two. An exact `sqrt`/`rsqrt` costs little on GPUs. |
| **Plastic spring-back** | Coefficient 0 by default, so beams spring back by the yield strain [RoR-#1506]. | Permanent set at `beamDeform`, capped by `deformLimit*`, with growing yield capped by `deformLimitStress` [BN-beams]. | BeamNG's, for sheet steel, which keeps most of its plastic set. |
| **Closures (bonnet, boot, doors)** | Only beams, ropes, ties and locks. Not found: a latch primitive. | Hinge beams in a breakGroup, plus a "slightly weaker latch" (https://beamng.com/goto/post?id=91974). Current content uses coupler latches: `couplerTag`/`tag`, `couplerStrength` (12,000 N in the hood example), `couplerWeld`, `couplerStartRadius` 0.01, and a `breakGroup` [BN-coupler]. A broken latch "cannot reconnect". SUPPORT beams keep panels from clipping. The debug view has a "Latches" text mode [BN-debug]. Latch strength sits in "advanced coupler controllers" (BeamNG Team, 2023: https://beamng.com/threads/is-there-a-way-to-globally-scale-the-strength-of-the-new-latches.91662). Staff note that A-pillar deformation can unlatch doors (https://www.beamng.com/goto/post?id=1421480). | BeamNG's. A latch that holds up to a set force and then releases once is how real latches fail. |
| **How BeamNG keeps the bonnet and boot shut** | n/a | Welded coupler latches with a finite strength, hinge beams in their own breakGroup, and SUPPORT beams against the body. Not found: any public number for the deceleration a latch must survive. | — |

## 4. Test coverage and validation

**RoR.** **Fact:** the repository's CI has two workflows, `build-game.yml` and `linux-native.yml`. The repository tree has no test directory. One file, `resources/scripts/example_autotest_scenario.as`, suggests scripted scenarios. Only file names were checked (via `gh api`). No unit-test or physics-regression suite was found. Bugs are reproduced by hand with a named vehicle and map ([RoR-#227] asks reporters for a "testing scenario").

**BeamNG.**

- **Barrier study [Arya]:** a peer-reviewed mass-spring model of F1 tyre barriers in BeamNG.tech, compared with FE and physical crash tests. The abstract reports peak crash-vehicle g "within 1 g" and maximum intrusion "within 8 cm". FE was more accurate. The soft-body model was cheaper. This is a barrier, not a car body.
- **No NHTSA or IIHS whole-vehicle crash validation found.** The Drive (2023) argues BeamNG crash videos are not realistic, because it lacks OEM material and structure data (https://www.thedrive.com/news/no-those-viral-beamng-crash-test-videos-arent-realistic).
- **Research uses** (https://beamng.tech/research/): crash reconstruction from accident sketches (Gambi et al., 2022), checked against reports, not physics. Energy-equivalent speed from BeamNG crash images (Toquica, 2021). Sim-to-real damage images (Fischer et al., 2021).
- **[Maul]:** the 714-node, 4,479-beam reference car and a performance benchmark. Its physics content is qualitative.
- **Tools [BN-debug]:** beam views (Stress, Displacement, Deformation, Break/Deform Groups, Frequency, Support/Bounded bounds), node views (Forces, Node Stability), a Node Triangle Self Collision Detector (≥ 0.035 m advised), the Node/Beam Info app (percentage deformed or broken), and a **Roof Crusher Tester** that replicates the IIHS roof test (force / 9.81 vs the weight class). Tyres were measured on an in-engine rig against Pacejka 96 [BN-tyre2].

## 5. Recommendations for Rubble

Each item becomes an explicit `ExplicitBeamsSettings` switch, a "BeamNG profile" next to the RoR profile, consistent with `explicit-beams.md`. The RoR profile stays for `REF=ror` comparisons.

1. **Exact inverse square root (and exact `exp`/`pow`/`sqrt`) by default for JBeam content.**
   - **Evidence:** the 1.75×10⁻³ error above. BeamNG documents no such approximation.
   - **Cost:** lossless in intent, cheap on the GPU.
   - **Scope:** keep `fast_inv_sqrt` for RoR parity runs only.
2. **Self-collision and vehicle contact following BeamNG's rules.**
   - **Rules:** per-node flags, side-memory anti-clip (already on `feat/softbody-beamng-parity`), NONCOLLIDABLE anti-clip triangles, the 2.5 cm band, `externalCollisionBias`, and same-`group` exclusion.
   - **Drop** RoR's triangle throttle outside the RoR profile. It skips work whose result can differ, which AGENTS.md forbids by default.
   - **Evidence:** [BN-nodes], [BN-tips], [RoR-#227].
3. **Plasticity from JBeam semantics.**
   - **Settings:** full permanent set at `beamDeform`, `deformLimit`/`deformLimitExpansion` caps, and `deformLimitStress` capping the growth of `beamDeform`.
   - **Keep out of the BeamNG profile:** RoR's 0.0 coefficient spring-back and its "strength falls with plastic stretch" rule.
   - **Evidence:** [BN-beams], [RoR-#1506].
   - **Inference:** uncapped stretch plus spring-back is a likely source of panels rippling and of a boot that keeps moving after a frontal hit. Test this before assuming it.
4. **Latches as couplers with a breakGroup.** Import `couplerTag`/`tag`/`couplerStrength`/`couplerWeld`/`couplerStartRadius` as a node-pair latch that releases once and never reconnects [BN-coupler]. This is the documented way BeamNG keeps closures shut.
5. **Damping tools, not a global damping change.** Support damping-only beams (`beamSpring` 0), `dampCutoffHz`, and torsionbars or rigidifiers on panels [BN-beams], [BN-tips], [BN-torsion]. Read damping from the JBeam file, and don't carry RoR's 12,000 N·s/m default into JBeam parts.
6. **Flexbodies.** Support the optional 4th (VZ) node [BN-flex]. Fail loudly on vertices that cannot be bound, rather than RoR's silent fallback to node 0 [RoR-#3255]. Deform on the GPU [RoR-#2861].
7. **Precision.** Keep node positions vehicle-local with origin rebasing, which the spec already has [RoR-#3448], [RoR-#1313].

**Before acting, diagnose the crash-lab ripple.** Run the frontal crash with each switch flipped separately (exact rsqrt, plastic coefficient 1, triangle throttle off, `deformLimitExpansion`) and record which one changes the boot displacement. Only then change defaults.

**Invariants and scenarios the literature suggests:**

- **Rest stability:** a spawned vehicle settles with no broken beams, no deformed beams, and no "Node Stability" or Stress flicker [BN-issues], [BN-debug].
- **Per-node limit:** the spring/damp-to-mass limit is checked at load, with a clear error rather than an explosion.
- **No pass-through:** across a speed sweep, no node ends on the far side of a triangle it started in front of (anti-clip invariant) [BN-nodes].
- **Energy:** with damping and contact off, total energy does not grow over N steps. Not found in public sources, but basic.
- **Plastic set:** load a beam past `beamDeform` and release it. The residual length matches the model and stays within `deformLimit*`.
- **Latches:** they hold below `couplerStrength`, release above it, never re-latch, and their breakGroup removes the hinge beams together.
- **Crash scenarios:** a frontal barrier crash measuring peak deceleration and intrusion (the [Arya] metrics, 1 g / 8 cm as a sanity scale). A roof-crush test following the BeamNG Roof Crusher (IIHS method) [BN-debug]. A rear-structure check: after a frontal crash, the boot lid stays latched and rear beams stay within a small deformation percentage (Node/Beam Info metric).
- **Tyres:** a rig sweep of slip angle and load against a Pacejka-style curve, checking that cornering stiffness falls as load rises [BN-tyre2].
