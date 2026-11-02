# Soft-body vehicle solver survey (2026-11-02)

What seven engines do for deformable bodies, and what Rubble should take from
each for its target: BeamNG-resolution vehicles, dozens to hundreds at once,
deforming like metal rather than jelly. The engines were Rigs of Rods, Jolt,
PhysX GPU deformables, NVIDIA Newton (VBD/XPBD), MuJoCo and MuJoCo Warp flex,
the AVBD codebases, and JGS2 together with an elasticity-solver benchmark.

The notes describe behaviour in our own words. No code was copied or
translated (clean-room rule). The per-engine notes are in [softbody-research/](softbody-research/README.md).

## Findings that hold across engines

1. **Nobody else ships plasticity or fracture for soft bodies.**
   - Jolt, Newton, MuJoCo and the AVBD solvers keep rest shapes immutable.
   - PhysX has a plasticity kernel, but it is switched off.
   - Rigs of Rods is the only reference with yield, permanent set and
     breaking. Rubble's explicit solver already reproduces those rules.
   - That part of the design is ours to own.
2. **Jelly comes from soft stiffness, not from too little damping.**
   - Ramping a material's stiffness as a penalty, AVBD-style, made a
     1e6 N/m spring behave like 7.5e4 N/m.
   - XPBD missed the true implicit step by 9% at k = 1e5 and dt = 1/60, and
     by 65% at dt = 1/8.
   - Unaugmented VBD penalties go soft on stiff meshes with high mass ratios
     when they get too few iterations.
   - Real stiffness at large steps needs either hard constraints with a
     persistent multiplier (an augmented Lagrangian), or small explicit steps,
     which is what RoR does at 2 kHz.
3. **The metal look in RoR comes from three things:**
   - very stiff beams relative to node mass;
   - a hard yield plateau that moves rest length, so the part stays bent;
   - thresholds that rescale with that rest length, a crude work hardening.

   None of these needs a special solver. They survive in any tier that
   resolves the stiffness.
4. **GPU fleets come from batching every body into each dispatch.**
   - MuJoCo Warp uses (world, item) arrays.
   - PhysX indexes all active bodies from one launch.
   - Three-avbd measured about 12 µs per empty dispatch, so below about 100k
     bodies the dispatch count sets the frame time.
   - Merging the small tail colours into one launch gave about 3.4x.
5. **RoR's scaling limits are structural:**
   - it parallelises per vehicle, so one heavy car runs on one core;
   - it has two thread barriers per 0.5 ms step;
   - partner search and sleep wake-up are O(N²);
   - it rebuilds per-vehicle k-d trees every step.

   One global grid and one dispatch per pass over all vehicles removes each
   of these. That is the shape of Rubble's wide passes.

## Recommendations, ranked

### 1. Reference tier: keep the explicit 2 kHz node–beam solver and finish its fleet path

This is the only tier that can meet the RoR-parity goal, because it is RoR's
model. The single-car numbers already beat RoR. The gap is 64 cars: about
11 ms against the 8.33 ms budget.

Next, all lossless:
- Pack per-step work into flat lists across vehicles, as PhysX and MuJoCo
  Warp do, so no grid is sized for the largest vehicle.
- Keep stop and break decisions on the GPU.
- Measure the wall-impact frames, where self-collision grows.

### 2. Fast tier, if wanted: augmented-Lagrangian block descent (AVBD-style) on the same beam graph

- Stiff beams become hard rows with a persistent multiplier and warm start.
  Never ramp a finite material stiffness.
- Yield and breaking go in the dual update. The multiplier is the beam
  force: past yield, move rest length and clamp the multiplier; past
  strength, mask the beam. Masking keeps the colouring valid.
- Colour the static beam graph once. Use Gauss–Seidel within colours with
  no Jacobi fallback; hard rows blew up under Jacobi.
- Fuse the tail colours into one launch. Give lanes to nodes with many
  beams.
- Prefer substeps over iterations: stiffness travels about one link per
  colour sweep.
- If convergence stalls, try Chebyshev or Anderson acceleration. Both need
  no precompute and survive topology change.
- Avoid JGS2's precomputed subspace: yield and breaks make it stale.
- Avoid XPBD or PBD for sheet metal (see the error figures above).
- Validate by plotting residual against iterations on the real JBeam truss,
  using a Newton-solve reference. Prove it does not read as jelly with the
  jello metrics: effective stiffness against target, springback after yield,
  and oscillation decay.
- This tier is a different physics model from RoR. It needs the user's
  decision and must stay an explicit option.

### 3. Contact

- Keep node-against-triangle contact with a margin, a barycentric spread of
  the reaction, and Stribeck friction from a surface material. That matches
  RoR.
- Use one global grid shared by self-collision and vehicle-against-vehicle
  contact, which Rubble already has.
- Accumulate forces with gather or atomics. RoR's cross-vehicle force
  writes are an unsynchronised race.
- Integrate inter-vehicle forces in the same step. RoR applies them one
  step late.
- Lossless culls to add: skip pairs that share a node, and filter by rest
  distance (PhysX).
- Fixed contact buffers with per-vehicle overflow flags must fail loudly,
  never drop contacts silently.
- **Lossy, so explicit option only:**
  - RoR's per-triangle "no hit, check less often" throttle;
  - Jolt's one contact plane per frame;
  - PhysX's stress-based contact skipping;
  - contact budget reduction (MuJoCo Warp).
- For the fast tier at larger steps: Newton's barrier contact with
  per-vertex displacement truncation keeps nodes from passing through.

### 4. Visuals: GPU skinning from node positions, with no readback

- Bind each render vertex to three nodes once and cache it: the nearest
  node, the next nearest, and one near-orthogonal to that pair.
- Rebuild positions on the GPU each frame.
- Recompute normals from the deformed mesh. RoR only rotates them, so its
  creases shade soft.
- Keep a permanent per-vertex damage mask that drives a dent and scratch
  texture.

### 5. Drivetrain on nodes

- Apply drive and brake torque as tangential forces on tyre nodes, with
  the reaction on the mounts.
- Brake with a predictive, clamped opposing torque.
- Split torque through a differential tree.
- Steer by changing a beam's rest length.
- Rubble lacks only RoR's detailed tyre options; RoR itself has no tyre
  model beyond node friction.

### 6. Sleep

- Sleep per vehicle or per island, with wake-up driven by the contact graph
  rather than RoR's O(N²) box scan.
- Read back only the changes, matching the large-maps cost model.

### 7. Sheet-metal bending (fast tier only)

- Wardetzky constant-stencil bending (MuJoCo) adds panel stiffness cheaply.
  When a panel yields, update its curved rest state.
- RoR has no bending term. Adding it to the reference tier would change the
  model, so it belongs to the fast tier only.

## Not adopted

- Generalised-coordinate global solves per vehicle (MuJoCo). A 1000-node car
  has about 3000 degrees of freedom, past where MuJoCo Warp scales.
- Tet FEM simulation meshes (PhysX, Newton): BeamNG content is beams.
- Saving the state and re-solving a step. Destruction is forward-only.
