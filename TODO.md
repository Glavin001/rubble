# Rubble GPU Physics Engine — Implementation Checklist

Status tracking against the Rubble Software Specification v1.1.0.

## Legend
- [x] Done and working
- [~] Partial — built but not integrated, or CPU-side instead of GPU
- [ ] Not started / missing

---

## Core Architecture

- [x] Workspace: 7 crates (rubble-math, rubble-gpu, rubble-primitives, rubble-shapes2d, rubble-shapes3d, rubble2d, rubble3d)
- [x] wgpu backend with WGSL compute shaders
- [x] `GpuContext` abstraction (device, queue, adapter)
- [x] `GpuBuffer<T>` with upload/download/grow
- [x] `GpuAtomicCounter` for contact/pair counts
- [x] `ComputeKernel` shader wrapper (WGSL + SPIR-V via naga)
- [x] Generational index allocator (`BodyHandle` with index + generation)
- [x] `SimConfig` / `SimConfig2D` with gravity, dt, solver_iterations, max_bodies, beta, k_start, warmstart_decay

## rust-GPU Shader Library (`rubble-shaders`)

- [x] All physics kernels ported to Rust via `spirv-std` for multi-GPU target support
  - [x] 3D Predict kernel (`predict_3d`) — gravity integration, position/quaternion prediction
  - [x] 3D AVBD Solver kernel (`avbd_solve_3d`) — augmented Lagrangian, graph-colored dispatch, friction
  - [x] 3D Extract Velocity kernel (`extract_velocity_3d`) — position recomputation, angular velocity from quaternion delta
  - [x] 2D Predict kernel (`predict_2d`) — gravity, position (x, y, angle) prediction
  - [x] 2D AVBD Solver kernel (`avbd_solve_2d`) — 2D augmented Lagrangian with cross2d rotational terms
  - [x] 2D Extract Velocity kernel (`extract_velocity_2d`) — position recomputation
  - [x] Sphere-Sphere narrowphase kernel (`sphere_sphere_test`)
  - [x] Trivial test kernel (`multiply_by_two`)
- [x] Shared GPU types: Body3D, Body2D, BodyProps3D, Contact3D, Contact2D, SimParams, SolveRange, Aabb, shape data
- [x] Math helpers: qmul, qconj, quat_rotate, mat3_mul, cross2d
- [x] spirv-builder build.rs pipeline (ready for rust-gpu toolchain activation)
- [x] SPIR-V → WGSL transpilation path via `ComputeKernel::from_spirv()` + naga

## Multi-GPU Support (`rubble-gpu`)

- [x] `MultiGpuContext` — enumerate all Vulkan adapters, create device/queue per GPU
- [x] `GpuDevice` — wraps device, queue, adapter info; `as_context()` for API compatibility
- [x] `WorkDistribution` — EvenSplit, RangeBased, SingleDevice strategies
- [x] `MultiGpuBuffer<T>` — per-device buffer mirror with upload_to_all/download_from/gather_results
- [x] `GpuDevicePool` — parallel kernel dispatch across multiple GPUs
- [x] `GpuContext::enumerate_adapters()` — list all available GPU adapters
- [x] `GpuContext::new_with_adapter()` — create context from specific adapter

## GPU Primitives (`rubble-primitives`)

- [x] `GpuPrefixScan` — Blelloch exclusive/inclusive scan, WGSL shader, workgroup size 256
- [x] `GpuRadixSort` — 4-bit-per-pass (8 passes for 32-bit keys), histogram + prefix scan + scatter
- [x] `GpuStreamCompaction` — prefix scan of predicates + scatter
- [x] **Integration**: Radix sort wired into GPU LBVH broadphase for Morton code sorting
  - [x] Wire radix sort into broadphase pair sorting or shape-pair dispatch sorting
  - [x] Wire prefix scan into broadphase (used internally by GpuRadixSort in LBVH)
  - [x] Wire stream compaction into broadphase filtering (used internally by GpuLbvh)

## PingPongBuffer (`rubble-gpu`)

- [x] `PingPongBuffer<T>` — dual `GpuBuffer`, swap(), current()/next(), upload/download
- [x] **Integration**: Used in both 2D and 3D pipelines
  - [x] Use for double-buffered body state in predict→solve→extract loop

## Shapes

### 2D (`rubble-shapes2d`)
- [x] Circle (`CircleData`)
- [x] Rect (`RectData`)
- [x] Capsule (`CapsuleData2D`)
- [x] Convex polygon up to 64 vertices (`ConvexPolygonData`, `ConvexVertex2D`)

### 3D (`rubble-shapes3d`)
- [x] Sphere (`SphereData`)
- [x] Box (`BoxData`)
- [x] Capsule (`CapsuleData`)
- [x] Convex hull up to 64 vertices (`ConvexHullData`, `ConvexVertex3D`, validation)
- [x] Plane (stored as `Vec4(nx, ny, nz, distance)`)
- [x] Compound shape (`CompoundShape`, `CompoundShapeGpu`, `CompoundChildGpu`)
- [x] AABB computation for all shapes (both CPU-side and GPU shader)

## Pipeline — Predict

- [x] 3D predict shader: gravity integration, position prediction, orientation prediction (quaternion)
- [x] 2D predict shader: gravity integration, position prediction, angle prediction
- [x] Saves `old_states` for velocity extraction
- [x] Static bodies (inv_mass=0) skip prediction
- [x] Kinematic bodies (FLAG_KINEMATIC) skip prediction

## Pipeline — Broadphase

### LBVH
- [x] Morton code computation (30-bit 3D, 30-bit 2D)
- [x] Karras 2012 binary radix tree construction
- [x] Overlap pair finding via tree traversal
- [x] **GPU-native**: Morton codes computed on GPU, sorted via GpuRadixSort, parallel BVH traversal
  - [x] GPU-native LBVH build (Morton sort on GPU via GpuRadixSort)
  - [x] GPU-native pair finding (parallel BVH traversal compute shader)

### AABB Compute Shader
- [x] Sphere, Box, Capsule, Convex Hull AABB on GPU
- [x] Plane: huge AABB (1e4) so it pairs with everything
- [x] Compound: conservative 100-unit AABB (CPU expansion handles precision)

## Pipeline — Narrowphase (GPU)

### 3D Shape-Pair Dispatch
- [x] Pairs sorted so `shape_type_a <= shape_type_b` for consistent dispatch
- [x] If-else chain dispatches to correct test function
- [x] Radix sort pairs by `(type_a << 16 | type_b)` key for SIMD-friendly batching

### 3D Collision Tests (all in `narrowphase_wgsl.rs`)
- [x] Sphere-Sphere
- [x] Sphere-Box
- [x] Sphere-Capsule (via sphere-sphere on endpoints)
- [x] Sphere-ConvexHull
- [x] Sphere-Plane
- [x] Box-Box — 15-axis SAT + Sutherland-Hodgman clipping + manifold reduction ≤4
- [x] Box-Capsule
- [x] Box-ConvexHull
- [x] Box-Plane (8-vertex test)
- [x] Capsule-Capsule
- [x] Capsule-ConvexHull
- [x] Capsule-Plane (2 sphere tests)
- [x] ConvexHull-ConvexHull — SAT + face clipping
- [x] ConvexHull-Plane (vertex test)
- [x] Plane-Plane: no collision (correct)

### 2D Collision Tests
- [x] Circle-Circle
- [x] Circle-Rect
- [x] Rect-Rect (SAT)
- [x] Circle-Capsule, Rect-Capsule, Capsule-Capsule
- [x] Convex polygon support

### Sutherland-Hodgman Clipping
- [x] `clip_polygon_against_plane()` — clips up to 8-vertex polygon against half-plane
- [x] `get_box_face_vertices()` / `get_box_face_normal()` — box face extraction
- [x] Used in `box_box_test()` for face contacts
- [x] Used in `hull_hull_test()` for reference face clipping

### Manifold Reduction
- [x] `reduce_manifold()` — reduces to ≤4 contacts via area maximization
  1. Pick deepest point
  2. Pick farthest from deepest
  3. Pick point maximizing triangle area
  4. Pick point on opposite side maximizing quadrilateral area

### Hull-Hull Edge-Edge SAT
- [x] Brute-force O(na*nb) edge-edge axes in hull_hull_test — correct and fast for ≤64-vertex hulls (at most 4096 iterations on GPU)

### Compound Shapes
- [x] CPU-side pair expansion in `run_detection()` — when broadphase pair involves compound, expand on CPU
- [x] `get_compound_children_world()` — transforms children to world space
- [x] `generate_compound_contacts_cpu()` — sphere-based proximity test per child pair
- [x] Local BVH per compound — `CompoundShape` has `bvh_nodes` field, BVH is built, traversal used for culling
  - [x] Use BVH for compound child culling in `generate_compound_contacts_cpu()`

## Pipeline — AVBD Solver

- [x] Graph-colored Gauss-Seidel: `color_contacts()` with greedy coloring
- [x] Contacts sorted by color, `SolveRange` uniform provides (offset, count) per group
- [x] Per-color dispatch prevents data races (no two contacts in same group share a body)
- [x] Augmented Lagrangian: `impulse = (-dC * penalty + lambda_old) / (w_eff * penalty + 1.0)`
- [x] Penalty stiffness ramp: `penalty_k += 10.0 * (-depth)`
- [x] Baumgarte positional stabilization: `bias = 0.2 * depth / dt`
- [x] Normal impulse clamped to ≥0 (non-attractive)
- [x] Tangential friction impulse with Coulomb cone clamping: `min(tang_len / w_eff, mu * impulse_clamped)`
- [x] Per-body friction averaging: `mu = (props_a.friction + props_b.friction) * 0.5`
- [x] Inverse inertia tensor application for angular impulses (3D) / scalar inertia (2D)
- [x] Dual variable (lambda) update for warm starting

## Pipeline — Velocity Extraction

- [x] `pos_new = pos_old + dt * v_solved` — position from solved velocity
- [x] Angular velocity extraction from quaternion delta (3D)
- [x] Static bodies skipped

## Contact Persistence + Warm Starting

- [x] `ContactPersistence3D` / `ContactPersistence2D` — tracks contacts across frames
- [x] `warm_start_contacts_3d()` / `warm_start_contacts_2d()` — matches new contacts to previous frame
- [x] Lambda carry-forward with 0.95 decay
- [x] Matching by spatial proximity (1cm threshold)

## Collision Events

- [x] `CollisionEvent::Started(a, b)` / `CollisionEvent::Ended(a, b)`
- [x] Generated from contact pair diffs between frames
- [x] `drain_collision_events()` API — returns and clears event queue

## World API

### 3D (`rubble3d`)
- [x] `World::new(config)` — creates GPU context and pipeline
- [x] `add_body(desc)` / `remove_body(handle)` — generational handle management
- [x] `get_position()` / `get_velocity()` / `get_rotation()`
- [x] `get_angular_velocity()` / `set_angular_velocity()`
- [x] `set_position()` / `set_velocity()`
- [x] `set_body_kinematic(handle, bool)`
- [x] `step()` — full GPU physics step (predict → AABB → broadphase → narrowphase → solve → extract)
- [x] `raycast(origin, dir, max_t)` — closest hit (handle, t, normal)
- [x] `raycast_batch(rays)` — batch of raycasts
- [x] `overlap_aabb(min, max)` — query bodies overlapping AABB
- [x] `drain_collision_events()` — collision event queue
- [x] `body_count()` / `gpu_pipeline()` diagnostics
- [x] `SimConfig` derives `Clone`

### 2D (`rubble2d`)
- [x] Same API surface as 3D adapted for 2D (Vec2, angle instead of quaternion)
- [x] Per-body friction stored in `_pad0.x` of `RigidBodyState2D`
- [x] `get_angular_velocity()` / `set_angular_velocity()`

## Bug Fixes Applied

- [x] **Critical**: `GpuBuffer::download()` returned empty for GPU-written buffers (narrowphase contacts) because `len` field was 0. Fixed by calling `set_len()` before download in both 2D and 3D pipelines.
- [x] Clippy warnings (map_or, loop indexing)
- [x] Non-exhaustive pattern for compound shapes in `ray_shape_test()`

## Test Coverage

- [x] 327+ tests total (132 unit + 195 integration/scenario), 0 compile failures
- [x] rubble-math: 24 tests (types, flags, state accessors)
- [x] rubble-gpu: 8 unit + 13 multi-GPU integration tests (buffer upload/download, atomic counter, ping-pong, device enumeration, work distribution, parallel compute)
- [x] rubble-primitives: 10 tests (prefix scan, radix sort, stream compaction, GPU LBVH)
- [x] rubble-shapes2d: 7 tests (shape data, AABB)
- [x] rubble-shapes3d: 12 tests (hull validation, AABB, compound BVH)
- [x] rubble2d: 27 unit + 17 integration + 25 AVBD solver tests (momentum conservation, energy, friction, stability, stacking, mixed shapes, stress)
- [x] rubble3d: 29 unit + 17 integration + 27 scenario + 25 AVBD solver + 26 exhaustive physics + 17 GPU performance tests
- [x] **AVBD solver accuracy tests**: momentum conservation (equal/unequal/3-body), energy non-increase, PE→KE conversion, convergence with iterations, penalty stiffness, warm starting, Baumgarte correction, friction behavior, angular dynamics
- [x] **Physics scenario tests**: stacking (2/3 high), domino chains, Newton's cradle, rotational dynamics, all collision pair types (sphere-capsule, box-capsule, capsule-capsule, hull-capsule, hull-plane), kinematic bodies, compound shapes, raycasting, AABB queries
- [x] **GPU performance tests**: scaling (16/32/64/128/256 bodies), sustained 500-1000 step simulation, dynamic add/remove, buffer overflow recovery, high velocity impacts, broadphase edge cases, multiple gravity directions
- [x] **Multi-GPU tests**: device enumeration, work distribution (even/range/single), buffer sync, parallel compute, edge cases (zero items, empty buffers)
- [x] **Numerical stability tests**: extreme mass ratios (1000:0.01), zero/small/large dt, many simultaneous contacts, high velocity

---

## File Map

| File | Purpose |
|---|---|
| `crates/rubble-math/src/lib.rs` | Shared types: BodyHandle, RigidBodyState, Contact, CollisionEvent, flags |
| `crates/rubble-gpu/src/lib.rs` | GpuContext, GpuError |
| `crates/rubble-gpu/src/buffer.rs` | GpuBuffer, GpuAtomicCounter, PingPongBuffer |
| `crates/rubble-gpu/src/kernel.rs` | ComputeKernel shader wrapper (WGSL + SPIR-V) |
| `crates/rubble-gpu/src/context.rs` | GpuContext, enumerate_adapters, new_with_adapter |
| `crates/rubble-gpu/src/multi_gpu.rs` | MultiGpuContext, GpuDevice, MultiGpuBuffer, GpuDevicePool, WorkDistribution |
| `crates/rubble-primitives/src/lib.rs` | Re-exports for GPU primitives |
| `crates/rubble-primitives/src/prefix_scan.rs` | GpuPrefixScan (Blelloch) |
| `crates/rubble-primitives/src/radix_sort.rs` | GpuRadixSort (4-bit passes) |
| `crates/rubble-primitives/src/compaction.rs` | GpuStreamCompaction |
| `crates/rubble-primitives/src/gpu_lbvh.rs` | GpuLbvh: GPU Morton codes + radix sort + CPU Karras tree + GPU pair traversal |
| `crates/rubble-shaders/src/lib.rs` | Rust-GPU shader library: all physics kernels in Rust for multi-GPU target SPIR-V |
| `crates/rubble-shaders/build.rs` | spirv-builder pipeline for SPIR-V compilation |
| `crates/rubble-shapes2d/src/lib.rs` | 2D shape data structs + AABB |
| `crates/rubble-shapes3d/src/lib.rs` | 3D shape data, convex hull validation, Gauss Map, compound BVH |
| `crates/rubble2d/src/lib.rs` | World2D public API |
| `crates/rubble2d/src/gpu/mod.rs` | 2D GPU pipeline (predict, AABB, broadphase, narrowphase, solve, extract) |
| `crates/rubble2d/src/gpu/predict_wgsl.rs` | 2D predict WGSL shader |
| `crates/rubble2d/src/gpu/narrowphase_wgsl.rs` | 2D narrowphase WGSL shader |
| `crates/rubble2d/src/gpu/avbd_solve_wgsl.rs` | 2D AVBD solver WGSL shader |
| `crates/rubble2d/src/gpu/extract_velocity_wgsl.rs` | 2D velocity extraction WGSL shader |
| `crates/rubble3d/src/lib.rs` | World3D public API |
| `crates/rubble3d/src/gpu/mod.rs` | 3D GPU pipeline + compound expansion + graph coloring |
| `crates/rubble3d/src/gpu/predict_wgsl.rs` | 3D predict WGSL shader |
| `crates/rubble3d/src/gpu/narrowphase_wgsl.rs` | 3D narrowphase WGSL (clipping, manifold, all pairs) |
| `crates/rubble3d/src/gpu/avbd_solve_wgsl.rs` | 3D AVBD solver WGSL shader |
| `crates/rubble3d/src/gpu/extract_velocity_wgsl.rs` | 3D velocity extraction WGSL shader |
