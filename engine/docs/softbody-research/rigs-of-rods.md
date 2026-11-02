# Rigs of Rods: soft-body vehicle design

Behaviour notes from a read-only review (2026-11-02) of the RigsOfRods source under
`source/main/`. RoR is GPL; nothing here is code. Rubble's explicit solver was built
from its own specification ([explicit-beams.md](../explicit-beams.md)) and is
compared against RoR only as a black box.

## 1. Step structure

- **Frame:** frame time is clamped to 1/20 s and simulated in fixed 0.5 ms steps
  (2 kHz), about 33 steps per frame at 60 fps. Leftover time carries over.
- **Once per frame:** sleep state, input, AI, engine audio, visuals and network.
- **Threading:** physics runs on a one-thread simulation pool and is normally joined
  at once. An optional async mode overlaps it with rendering.
- **Each step, in order:**
  1. Serial preparation: hooks and ropes.
  2. Per-vehicle compute, in parallel, one task per vehicle.
  3. Serial beams between vehicles (ties, hooks).
  4. Per-vehicle collision against other vehicles, in parallel.
  5. Free forces.
- **Inside a vehicle's compute, in order:**
  1. Nodes first: terrain and object collision, symplectic Euler integration,
     then the force is reset to gravity and drag and buoyancy are added.
  2. Then differentials, wheels, shocks, hydros, commands, ties, engine, beams,
     self-collision and slide nodes.
- **One-step lag:** forces gathered in a step, including contacts between
  vehicles, are integrated at the start of the next step.
- **Graphics side:** flexbodies (deforming meshes) and props update once per
  frame.

## 2. Performance design

- **Threads:** the worker pool size is cores − 1, clamped between 1 and 8.
- **Per-vehicle parallelism only:** one heavy vehicle never spreads across
  cores.
- **Barriers:** two thread barriers per step, about 4,000 per second. With few
  vehicles the synchronisation cost dominates.
- **Memory layout:** no SIMD. Nodes are an array of structs that mix position,
  velocity, force and flags. Beams hold raw node pointers, so the beam pass
  reads scattered memory.
- **Speed tricks:** approximate square roots, and a cheap force check before the
  plastic branch.
- **Sleep:**
  - A vehicle sleeps after 10 s nearly at rest. AI vehicles never sleep.
  - Wake-up recursively scans pairs of vehicles with overlapping current or
    predicted boxes, which is O(N²).
  - A sleeping vehicle costs no physics.
- **Fleet scaling:** per-vehicle work grows linearly. Partner discovery and
  sleep wake-up are O(N²). The per-step global barriers at 2 kHz cap the fleet
  size.
- **User knobs:**
  - worker count;
  - async physics;
  - pseudo-collisions for multiplayer;
  - the collision throttle (below);
  - flexbody cache;
  - a global scale for beam defaults.

## 3. Contact between vehicles

- **Tests:** node against triangle only, with no edge–edge tests.
- **Search structure:**
  - Each vehicle builds a k-d tree of other vehicles' contactable nodes, taking
    only vehicles whose box overlaps its own.
  - The tree is rebuilt or refreshed every step, and split lazily while queries
    run.
  - Each collision triangle queries the tree with its box grown by a 2 cm
    margin.
- **Hit test:** the node lies inside the triangle by barycentric coordinates and
  within the margin. A neighbour vote flips the normal for back-face hits.
- **Response:**
  - a reduced-mass penalty with friction from a surface material;
  - the reaction is spread over the triangle's three nodes by barycentric
    weights;
  - symmetric, because each vehicle tests its own triangles against the other's
    nodes.
- **Throttle (lossy):**
  - A triangle with no hits is tested less often, skipping up to 12 steps and
    backing off as misses accumulate.
  - Throttling resets when the vehicles' relative speed exceeds 4 m/s.
- **Tunnelling:** with a 2 cm margin, avoiding pass-through relies on the 2 kHz
  step.
- **Race:** parallel per-vehicle tasks write forces into other vehicles' nodes
  without synchronisation.
- **Self-collision:** works the same way, using a tree over the vehicle's own
  nodes.

## 4. Wheels and drivetrain

- **Wheels are plain nodes and beams.**
  - The basic wheel is two rings of tyre nodes around two axle nodes.
  - Other variants add separate rim and tyre rings with their own stiffness, or
    a visual mesh that deforms with the nodes.
- **Torque:**
  - A tree of wheel, axle and transfer-case differentials splits engine torque.
  - Traction control and ABS act as pulsed factors.
  - Drive and brake torque are tangential forces on the tyre nodes, with a
    reaction on the axle mounts.
- **Brakes:** a predictive torque opposing wheel rotation, corrected by the
  previous step's error and clamped to the brake strength. Wheels stop without
  flipping sign.
- **Wheel speed:** measured from node velocities.
- **Steering:** hydraulic beams change their rest length.
- **Tyre contact:** there is no tyre model. Each tyre node meets the ground like
  any other node:
  - normal force from a blend of approach velocity and penetration;
  - smoothed static friction at low speed, then a Stribeck curve from the
    surface's parameters.

  Slip angle and load sensitivity come from the tyre nodes deforming.

## 5. Deformation and its appearance

- **Plastic beams:**
  - Beams are Hookean springs with damping.
  - Past a compression or tension threshold force, the rest length moves so the
    force sits at yield, so the part stays bent.
  - The thresholds then rescale with the new rest length, a crude work
    hardening.
  - Tension yield lowers the beam's strength.
  - A beam breaks when its force exceeds its strength. Detacher groups cascade a
    break to other beams and to wheels.
- **Defaults:** spring 9e6, damping 1.2e4, deform threshold 4e5 N, break 1e6 N,
  plastic coefficient 0, minimum node mass 50 kg. The threshold is clamped
  unless the advanced-deformation mode is set, and a global scale can be
  applied.
- **Why it reads as metal:**
  - very stiff beams relative to node mass;
  - a hard yield plateau;
  - permanent rest-length drift, so there is no springback.
- **Flexbody skinning:**
  - At spawn each render vertex is bound to three nodes: the nearest, the next
    nearest, and the nearest node roughly orthogonal (within 45°) to that pair.
    Its offset is stored in that non-orthonormal frame. The binding can be
    cached to disk.
  - Each frame the vertex is rebuilt from the deformed frame.
  - Normals are rotated with the frame, not recomputed, so sharp creases shade
    soft.
  - Skinning runs as CPU tasks, and the whole vertex buffer is uploaded every
    frame.
- **Damage look:** a per-vertex bit is set permanently once the vertex's node
  touches something. It blends in a dent and dirt texture.
- **Cab and props:** cab triangles recompute smooth normals. Props are rigid,
  riding a three-node frame.

## 6. Lessons for Rubble

**Match in behaviour (parity):**
- **Contact:** node–triangle contact with a margin, barycentric spread of the
  reaction, reduced-mass penalty, and Stribeck friction from a surface
  material.
- **Plasticity:** a yield plateau, rest-length drift, rescaled thresholds, and
  a global threshold scale.
- **Drivetrain:** torque as node forces, predictive clamped brakes, a
  differential tree, and steering by rest length.

**Improve on RoR:**
- **Dispatches:** one GPU dispatch per pass over all vehicles, instead of two
  CPU barriers per step.
- **Search:** one global grid shared by every triangle, instead of per-vehicle
  k-d trees rebuilt every step and O(N²) partner search.
- **Race:** gather or atomics, instead of unsynchronised writes to other
  vehicles' forces.
- **One-step lag:** forces between vehicles in the same step, not one step
  late.
- **Memory layout:** structure-of-arrays nodes and index-based beams.
- **Skinning:** on the GPU straight from node positions with no readback,
  recomputing normals so creases shade sharp.
- **Sleep:** islands woken by contact rather than a pairwise box scan.

**Lossy, so an explicit option only (AGENTS.md):** the per-triangle collision
throttle.
