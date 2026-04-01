# Rubble GPU Physics Engine — Implementation Checklist

Status tracking against the Ferrophys Software Specification v1.1.0.

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
- [x] `ComputeKernel` shader wrapper
- [x] Generational index allocator (`BodyHandle` with index + generation)
- [x] `SimConfig` / `SimConfig2D` with gravity, dt, solver_iterations, max_bodies, beta, k_start, warmstart_decay

## GPU Primitives (`rubble-primitives`)

- [x] `GpuPrefixScan` — Blelloch exclusive/inclusive scan, WGSL shader, workgroup size 256
- [x] `GpuRadixSort` — 4-bit-per-pass (8 passes for 32-bit keys), histogram + prefix scan + scatter
- [x] `GpuStreamCompaction` — prefix scan of predicates + scatter
- [x] **Integration**: Radix sort wired into GPU LBVH broadphase for Morton code sorting
  - [x] Wire radix sort into broadphase pair sorting or shape-pair dispatch sorting
  - [~] Wire prefix scan into broadphase or contact buffer management
  - [~] Wire stream compaction into broadphase filtering

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
- [x] `GaussMapEntry` struct and `precompute_gauss_map()` function

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

### Gauss Map (Edge-Edge Pruning)
- [x] `precompute_gauss_map()` in rubble-shapes3d — enumerates non-parallel edge pairs
- [x] `gauss_map_offset` / `gauss_map_count` fields in `ConvexHullData`
- [x] **Wired in**: Gauss Map entries computed on body creation, uploaded, used in hull-hull SAT
  - [x] Call `precompute_gauss_map()` when adding convex hull bodies
  - [x] Upload Gauss Map entries to GPU buffer
  - [x] Use in hull-hull SAT to prune edge-edge axes

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
- [x] `set_position()` / `set_velocity()` / `set_angular_velocity()`
- [x] `set_body_kinematic(handle, bool)`
- [x] `step()` — full GPU physics step (predict → AABB → broadphase → narrowphase → solve → extract)
- [x] `raycast(origin, dir, max_t)` — closest hit (handle, t, normal)
- [x] `raycast_batch(rays)` — batch of raycasts
- [x] `overlap_aabb(min, max)` — query bodies overlapping AABB
- [x] `drain_collision_events()` — collision event queue
- [x] `body_count()` / `gpu_pipeline()` diagnostics

### 2D (`rubble2d`)
- [x] Same API surface as 3D adapted for 2D (Vec2, angle instead of quaternion)
- [x] Per-body friction stored in `_pad0.x` of `RigidBodyState2D`

## Bug Fixes Applied

- [x] **Critical**: `GpuBuffer::download()` returned empty for GPU-written buffers (narrowphase contacts) because `len` field was 0. Fixed by calling `set_len()` before download in both 2D and 3D pipelines.
- [x] Clippy warnings (map_or, loop indexing)
- [x] Non-exhaustive pattern for compound shapes in `ray_shape_test()`

## Test Coverage

- [x] 221 tests total, 0 failures
- [x] rubble-math: 24 tests (types, flags, state accessors)
- [x] rubble-gpu: 8 tests (buffer upload/download, atomic counter, ping-pong)
- [x] rubble-primitives: 10 tests (prefix scan, radix sort, stream compaction)
- [x] rubble-shapes2d: 7 tests (shape data, AABB)
- [x] rubble-shapes3d: 15 tests (hull validation, AABB, gauss map, compound BVH)
- [x] rubble2d: 33 tests (world API, shapes, raycast, kinematic, collision events, stress)
- [x] rubble3d: 35 unit + 17 integration + 27 scenario tests
- [x] GPU integration tests (gpu_integration.rs): broadphase, narrowphase, solver, multi-step
- [x] GPU scenario tests (gpu_scenarios.rs): energy conservation, momentum, stacking, stress

## Remaining Work (Priority Order)

### High Priority — Integration of existing modules
1. [x] Wire `precompute_gauss_map()` into convex hull body creation, upload entries, use in hull-hull SAT
2. [x] Wire `GpuRadixSort` into broadphase pair sorting (sort by shape-type key for batched dispatch)
3. [x] Use `PingPongBuffer` for body state double-buffering in predict→solve→extract
4. [x] Use compound BVH for child culling in `generate_compound_contacts_cpu()`

### Medium Priority — GPU-native broadphase
5. [x] GPU-native Morton code sort (use `GpuRadixSort` on Morton-coded AABBs)
6. [~] GPU-native Karras tree construction (CPU-side build, GPU Morton sort + GPU pair traversal)
7. [x] GPU-native overlap pair finding (parallel BVH traversal)

### Low Priority — Robustness & polish
8. [x] Buffer overflow recovery: detect overflow, resize buffers, re-run narrowphase
9. [x] encase (or static_assert) layout validation for all GPU structs
10. [x] GPU-batched raycast dispatch
11. [x] Shape-pair dispatch sorting via radix sort for SIMD-friendly narrowphase

---

## File Map

| File | Purpose |
|---|---|
| `crates/rubble-math/src/lib.rs` | Shared types: BodyHandle, RigidBodyState, Contact, CollisionEvent, flags |
| `crates/rubble-gpu/src/lib.rs` | GpuContext, GpuError |
| `crates/rubble-gpu/src/buffer.rs` | GpuBuffer, GpuAtomicCounter, PingPongBuffer |
| `crates/rubble-gpu/src/kernel.rs` | ComputeKernel shader wrapper |
| `crates/rubble-primitives/src/lib.rs` | Re-exports for GPU primitives |
| `crates/rubble-primitives/src/prefix_scan.rs` | GpuPrefixScan (Blelloch) |
| `crates/rubble-primitives/src/radix_sort.rs` | GpuRadixSort (4-bit passes) |
| `crates/rubble-primitives/src/compaction.rs` | GpuStreamCompaction |
| `crates/rubble-shapes2d/src/lib.rs` | 2D shape data structs + AABB |
| `crates/rubble-shapes3d/src/lib.rs` | 3D shape data, convex hull validation, Gauss Map, compound BVH |
| `crates/rubble2d/src/lib.rs` | World2D public API |
| `crates/rubble2d/src/gpu/mod.rs` | 2D GPU pipeline (predict, AABB, broadphase, narrowphase, solve, extract) |
| `crates/rubble2d/src/gpu/predict_wgsl.rs` | 2D predict WGSL shader |
| `crates/rubble2d/src/gpu/narrowphase_wgsl.rs` | 2D narrowphase WGSL shader |
| `crates/rubble2d/src/gpu/avbd_solve_wgsl.rs` | 2D AVBD solver WGSL shader |
| `crates/rubble2d/src/gpu/extract_velocity_wgsl.rs` | 2D velocity extraction WGSL shader |
| `crates/rubble2d/src/gpu/lbvh.rs` | 2D LBVH (CPU-side) |
| `crates/rubble3d/src/lib.rs` | World3D public API |
| `crates/rubble3d/src/gpu/mod.rs` | 3D GPU pipeline + compound expansion + graph coloring |
| `crates/rubble3d/src/gpu/predict_wgsl.rs` | 3D predict WGSL shader |
| `crates/rubble3d/src/gpu/narrowphase_wgsl.rs` | 3D narrowphase WGSL (clipping, manifold, all pairs) |
| `crates/rubble3d/src/gpu/avbd_solve_wgsl.rs` | 3D AVBD solver WGSL shader |
| `crates/rubble3d/src/gpu/extract_velocity_wgsl.rs` | 3D velocity extraction WGSL shader |
| `crates/rubble3d/src/gpu/broadphase_pairs_wgsl.rs` | Broadphase pair WGSL shader |
| `crates/rubble3d/src/gpu/lbvh.rs` | 3D LBVH (CPU-side, Karras 2012) |
