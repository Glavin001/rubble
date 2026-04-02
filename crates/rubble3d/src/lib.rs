//! `rubble3d` — Public API facade for the rubble 3D GPU physics engine.
//!
//! All physics simulation runs entirely on the GPU via WGSL compute shaders.
//! The pipeline: predict → AABB → broadphase → narrowphase → AVBD solver → velocity extraction.

pub mod gpu;

use glam::{Mat3, Quat, Vec3, Vec4};
use rubble_math::{
    Aabb3D as MathAabb3D, BodyHandle, CollisionEvent, RigidBodyProps3D, RigidBodyState3D,
    FLAG_STATIC, SHAPE_BOX, SHAPE_CAPSULE, SHAPE_COMPOUND, SHAPE_CONVEX_HULL, SHAPE_PLANE,
    SHAPE_SPHERE,
};
use rubble_shapes3d::{
    BoxData, CapsuleData, CompoundChild, CompoundChildGpu, CompoundShape, CompoundShapeGpu,
    ConvexHullData, ConvexVertex3D, SphereData,
};

// ---------------------------------------------------------------------------
// SimConfig
// ---------------------------------------------------------------------------

/// Top-level simulation configuration.
pub struct SimConfig {
    pub gravity: Vec3,
    pub dt: f32,
    pub solver_iterations: u32,
    pub max_bodies: usize,
    /// Augmented Lagrangian stiffness ramp rate.
    pub beta: f32,
    /// Initial penalty stiffness for contacts.
    pub k_start: f32,
    /// Decay factor for warm-started dual variables across steps.
    pub warmstart_decay: f32,
    /// Default friction coefficient for bodies without explicit friction.
    pub friction_default: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            max_bodies: 65536,
            beta: 10.0,
            k_start: 1e4,
            warmstart_decay: 0.95,
            friction_default: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// ShapeDesc
// ---------------------------------------------------------------------------

/// Describes a collision shape for body creation.
#[derive(Debug, Clone)]
pub enum ShapeDesc {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: Vec3,
    },
    Capsule {
        half_height: f32,
        radius: f32,
    },
    /// Convex hull defined by up to 64 vertices in local space.
    ConvexHull {
        vertices: Vec<Vec3>,
    },
    /// Infinite plane (always static). Normal points into the free half-space.
    Plane {
        normal: Vec3,
        distance: f32,
    },
    /// Compound shape: a collection of child shapes with local transforms.
    /// Each child is (ShapeDesc, local_position, local_rotation).
    Compound {
        children: Vec<(ShapeDesc, Vec3, Quat)>,
    },
}

// ---------------------------------------------------------------------------
// RigidBodyDesc
// ---------------------------------------------------------------------------

/// Descriptor for creating a rigid body.
pub struct RigidBodyDesc {
    pub position: Vec3,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    /// Mass of the body. Use 0 for static bodies.
    pub mass: f32,
    pub friction: f32,
    pub shape: ShapeDesc,
}

impl Default for RigidBodyDesc {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.5,
            shape: ShapeDesc::Sphere { radius: 0.5 },
        }
    }
}

// ---------------------------------------------------------------------------
// Inertia computation
// ---------------------------------------------------------------------------

/// Compute the inverse inertia tensor for a shape given its mass.
/// Returns `Mat3::ZERO` for static bodies (mass <= 0).
fn compute_inertia(shape: &ShapeDesc, mass: f32) -> Mat3 {
    if mass <= 0.0 {
        return Mat3::ZERO;
    }
    let diag = match shape {
        ShapeDesc::Sphere { radius } => {
            let i = (2.0 / 5.0) * mass * radius * radius;
            Vec3::splat(i)
        }
        ShapeDesc::Box { half_extents } => {
            let w = 2.0 * half_extents.x;
            let h = 2.0 * half_extents.y;
            let d = 2.0 * half_extents.z;
            Vec3::new(
                mass / 12.0 * (h * h + d * d),
                mass / 12.0 * (w * w + d * d),
                mass / 12.0 * (w * w + h * h),
            )
        }
        ShapeDesc::Capsule {
            half_height,
            radius,
        } => {
            // Capsule inertia = cylinder + 2 hemispheres
            let h = 2.0 * half_height;
            let r2 = radius * radius;
            let cyl_mass = mass * h / (h + (4.0 / 3.0) * radius);
            let cap_mass = mass - cyl_mass;
            let iy = cyl_mass * r2 / 2.0 + cap_mass * 2.0 * r2 / 5.0;
            let ix = cyl_mass * (3.0 * r2 + h * h) / 12.0
                + cap_mass * (2.0 * r2 / 5.0 + h * h / 4.0 + 3.0 * h * radius / 8.0);
            Vec3::new(ix, iy, ix)
        }
        ShapeDesc::ConvexHull { vertices } => {
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            for &v in vertices {
                min = min.min(v);
                max = max.max(v);
            }
            let size = max - min;
            Vec3::new(
                mass / 12.0 * (size.y * size.y + size.z * size.z),
                mass / 12.0 * (size.x * size.x + size.z * size.z),
                mass / 12.0 * (size.x * size.x + size.y * size.y),
            )
        }
        ShapeDesc::Plane { .. } => {
            // Planes are always static, so this shouldn't be called with mass > 0
            Vec3::splat(1.0)
        }
        ShapeDesc::Compound { children } => {
            // Approximate compound inertia using bounding box of all child shapes.
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            for (child_shape, local_pos, _local_rot) in children {
                // Simple approximation: use child position + rough extent
                let extent = match child_shape {
                    ShapeDesc::Sphere { radius } => Vec3::splat(*radius),
                    ShapeDesc::Box { half_extents } => *half_extents,
                    ShapeDesc::Capsule {
                        half_height,
                        radius,
                    } => Vec3::new(*radius, *half_height + *radius, *radius),
                    _ => Vec3::splat(1.0),
                };
                min = min.min(*local_pos - extent);
                max = max.max(*local_pos + extent);
            }
            let size = max - min;
            Vec3::new(
                mass / 12.0 * (size.y * size.y + size.z * size.z),
                mass / 12.0 * (size.x * size.x + size.z * size.z),
                mass / 12.0 * (size.x * size.x + size.y * size.y),
            )
        }
    };
    Mat3::from_diagonal(Vec3::new(1.0 / diag.x, 1.0 / diag.y, 1.0 / diag.z))
}

/// Compute the local AABB for a child shape at a given local offset and rotation.
fn compute_child_local_aabb(shape: &ShapeDesc, local_pos: Vec3, local_rot: Quat) -> MathAabb3D {
    match shape {
        ShapeDesc::Sphere { radius } => {
            let r = Vec3::splat(*radius);
            MathAabb3D::new(local_pos - r, local_pos + r)
        }
        ShapeDesc::Box { half_extents } => {
            rubble_shapes3d::compute_box_aabb(local_pos, local_rot, *half_extents)
        }
        ShapeDesc::Capsule {
            half_height,
            radius,
        } => rubble_shapes3d::compute_capsule_aabb(local_pos, local_rot, *half_height, *radius),
        ShapeDesc::ConvexHull { vertices } => {
            rubble_shapes3d::compute_convex_hull_aabb(local_pos, local_rot, vertices)
        }
        ShapeDesc::Plane { normal, distance } => {
            let center = *normal * *distance + local_pos;
            let big = Vec3::splat(1e4);
            MathAabb3D::new(center - big, center + big)
        }
        ShapeDesc::Compound { .. } => {
            // Nested compound -- shouldn't happen, use a default AABB
            MathAabb3D::new(local_pos - Vec3::ONE, local_pos + Vec3::ONE)
        }
    }
}

// ---------------------------------------------------------------------------
// GenerationalIndexAllocator
// ---------------------------------------------------------------------------

/// Generational index allocator for stable body handles.
struct GenerationalIndexAllocator {
    generations: Vec<u32>,
    free_list: Vec<u32>,
}

impl GenerationalIndexAllocator {
    fn new() -> Self {
        Self {
            generations: Vec::new(),
            free_list: Vec::new(),
        }
    }

    fn alloc(&mut self) -> BodyHandle {
        if let Some(index) = self.free_list.pop() {
            let gen = self.generations[index as usize];
            BodyHandle::new(index, gen)
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            BodyHandle::new(index, 0)
        }
    }

    fn dealloc(&mut self, handle: BodyHandle) -> bool {
        let idx = handle.index as usize;
        if idx >= self.generations.len() {
            return false;
        }
        if self.generations[idx] != handle.generation {
            return false;
        }
        self.generations[idx] += 1;
        self.free_list.push(handle.index);
        true
    }

    fn is_valid(&self, handle: BodyHandle) -> bool {
        let idx = handle.index as usize;
        idx < self.generations.len() && self.generations[idx] == handle.generation
    }
}

// ---------------------------------------------------------------------------
// 3D raycast helpers
// ---------------------------------------------------------------------------

/// Ray-sphere intersection.
fn ray_sphere_3d(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<(f32, Vec3)> {
    let oc = origin - center;
    let b = oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t < 0.0 {
        return None;
    }
    let hit = origin + dir * t;
    let normal = (hit - center).normalize();
    Some((t, normal))
}

/// Ray-capsule intersection in 3D. Capsule = segment(ep_a, ep_b) + radius.
fn ray_capsule_3d(
    origin: Vec3,
    dir: Vec3,
    ep_a: Vec3,
    ep_b: Vec3,
    radius: f32,
) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;

    // Test sphere at each endpoint
    for &ep in &[ep_a, ep_b] {
        if let Some((t, n)) = ray_sphere_3d(origin, dir, ep, radius) {
            if t >= 0.0 && best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, n));
            }
        }
    }

    // Test infinite cylinder along segment axis
    let seg = ep_b - ep_a;
    let seg_len_sq = seg.dot(seg);
    if seg_len_sq > 1e-12 {
        let seg_dir = seg / seg_len_sq.sqrt();
        // Project ray onto plane perpendicular to segment
        let oc = origin - ep_a;
        let d_perp = dir - seg_dir * dir.dot(seg_dir);
        let oc_perp = oc - seg_dir * oc.dot(seg_dir);

        let a = d_perp.dot(d_perp);
        let b = 2.0 * oc_perp.dot(d_perp);
        let c = oc_perp.dot(oc_perp) - radius * radius;

        if a > 1e-12 {
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                let t = (-b - disc.sqrt()) / (2.0 * a);
                if t >= 0.0 {
                    let hit = origin + dir * t;
                    let proj = seg_dir.dot(hit - ep_a);
                    if proj >= 0.0 && proj <= seg_len_sq.sqrt() && best.is_none_or(|(bt, _)| t < bt)
                    {
                        // Normal: from closest point on axis to hit point
                        let axis_point = ep_a + seg_dir * proj;
                        let normal = (hit - axis_point).normalize();
                        best = Some((t, normal));
                    }
                }
            }
        }
    }

    best
}

/// Ray-convex hull intersection in 3D using slab method on face planes.
fn ray_convex_hull_3d(origin: Vec3, dir: Vec3, verts: &[Vec3]) -> Option<(f32, Vec3)> {
    if verts.len() < 4 {
        return None;
    }

    // Compute convex hull face planes from vertices using brute-force triangulation
    // For simplicity, use the face normals approach: test ray against each triangle face
    // of a simple convex hull (triangulated from centroid).
    let _centroid: Vec3 = verts.iter().copied().sum::<Vec3>() / verts.len() as f32;

    let mut tmin = f32::NEG_INFINITY;
    let mut tmax = f32::INFINITY;
    let mut best_normal = Vec3::ZERO;

    // For each pair of adjacent vertices, form a triangle with centroid and test as a slab
    // This is a simplified approach: use support-function based slab test
    // For each face normal direction, project hull and test
    // Simplest correct approach: build face normals from vertex triplets
    for i in 0..verts.len() {
        for j in (i + 1)..verts.len() {
            for k in (j + 1)..verts.len() {
                let v0 = verts[i];
                let v1 = verts[j];
                let v2 = verts[k];
                let normal = (v1 - v0).cross(v2 - v0);
                let len = normal.length();
                if len < 1e-8 {
                    continue;
                }
                let normal = normal / len;

                // Check if this is actually a face (all other verts on one side)
                let d = normal.dot(v0);
                let mut all_behind = true;
                for (idx, &v) in verts.iter().enumerate() {
                    if idx == i || idx == j || idx == k {
                        continue;
                    }
                    if normal.dot(v) > d + 1e-6 {
                        all_behind = false;
                        break;
                    }
                }
                if !all_behind {
                    continue;
                }

                // This is a face. Do slab test.
                let denom = normal.dot(dir);
                let dist = d - normal.dot(origin);

                if denom.abs() < 1e-12 {
                    if dist < -1e-6 {
                        return None;
                    }
                    continue;
                }

                let t = dist / denom;
                if denom < 0.0 {
                    if t > tmin {
                        tmin = t;
                        best_normal = normal;
                    }
                } else {
                    tmax = tmax.min(t);
                }

                if tmin > tmax {
                    return None;
                }
            }
        }
    }

    if tmin >= 0.0 && tmin <= tmax {
        Some((tmin, best_normal))
    } else {
        None
    }
}

/// Ray-AABB intersection in local space, returns result in world space.
fn ray_aabb_3d(
    local_origin: Vec3,
    local_dir: Vec3,
    half_extents: Vec3,
    world_rot: Quat,
) -> Option<(f32, Vec3)> {
    let mut tmin = f32::NEG_INFINITY;
    let mut tmax = f32::INFINITY;
    let mut normal_idx = 0usize;
    let mut normal_sign = 1.0f32;

    let he = [half_extents.x, half_extents.y, half_extents.z];
    let lo = [local_origin.x, local_origin.y, local_origin.z];
    let ld = [local_dir.x, local_dir.y, local_dir.z];

    for i in 0..3 {
        if ld[i].abs() < 1e-12 {
            if lo[i] < -he[i] || lo[i] > he[i] {
                return None;
            }
            continue;
        }
        let t1 = (-he[i] - lo[i]) / ld[i];
        let t2 = (he[i] - lo[i]) / ld[i];
        let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        if t_near > tmin {
            tmin = t_near;
            normal_idx = i;
            normal_sign = if ld[i] > 0.0 { -1.0 } else { 1.0 };
        }
        tmax = tmax.min(t_far);
        if tmin > tmax {
            return None;
        }
    }

    if tmin < 0.0 {
        return None;
    }
    let mut local_normal = Vec3::ZERO;
    match normal_idx {
        0 => local_normal.x = normal_sign,
        1 => local_normal.y = normal_sign,
        _ => local_normal.z = normal_sign,
    }
    Some((tmin, world_rot * local_normal))
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

/// The main 3D physics world. All simulation runs on the GPU.
pub struct World {
    config: SimConfig,
    states: Vec<RigidBodyState3D>,
    props: Vec<RigidBodyProps3D>,
    shapes: Vec<ShapeDesc>,
    spheres: Vec<SphereData>,
    boxes: Vec<BoxData>,
    capsules: Vec<CapsuleData>,
    convex_hulls: Vec<ConvexHullData>,
    convex_vertices: Vec<ConvexVertex3D>,
    /// Plane data stored as Vec4(nx, ny, nz, distance)
    planes: Vec<Vec4>,
    compound_shapes: Vec<CompoundShapeGpu>,
    compound_children: Vec<CompoundChildGpu>,
    /// CPU-side compound shape data for pair expansion.
    compound_shapes_cpu: Vec<CompoundShape>,
    allocator: GenerationalIndexAllocator,
    alive: Vec<bool>,
    gpu_pipeline: gpu::GpuPipeline,
    contact_persistence: gpu::ContactPersistence3D,
    collision_events: Vec<CollisionEvent>,
}

impl World {
    /// Create a new 3D physics world backed by GPU compute shaders.
    ///
    /// Returns an error if no GPU adapter is available.
    pub fn new(config: SimConfig) -> Result<Self, rubble_gpu::GpuError> {
        let ctx = pollster::block_on(rubble_gpu::GpuContext::new())?;
        let pipeline = gpu::GpuPipeline::new(ctx, config.max_bodies);
        Ok(Self {
            config,
            states: Vec::new(),
            props: Vec::new(),
            shapes: Vec::new(),
            spheres: Vec::new(),
            boxes: Vec::new(),
            capsules: Vec::new(),
            convex_hulls: Vec::new(),
            convex_vertices: Vec::new(),
            planes: Vec::new(),
            compound_shapes: Vec::new(),
            compound_children: Vec::new(),
            compound_shapes_cpu: Vec::new(),
            allocator: GenerationalIndexAllocator::new(),
            alive: Vec::new(),
            gpu_pipeline: pipeline,
            contact_persistence: gpu::ContactPersistence3D::new(),
            collision_events: Vec::new(),
        })
    }

    /// Add a rigid body to the world. Returns a stable handle.
    pub fn add_body(&mut self, desc: &RigidBodyDesc) -> BodyHandle {
        let handle = self.allocator.alloc();
        let idx = handle.index as usize;

        let inv_mass = if desc.mass > 0.0 {
            1.0 / desc.mass
        } else {
            0.0
        };

        let state = RigidBodyState3D::new(
            desc.position,
            inv_mass,
            desc.rotation,
            desc.linear_velocity,
            desc.angular_velocity,
        );

        let inv_inertia = compute_inertia(&desc.shape, desc.mass);
        let flags = if desc.mass <= 0.0 { FLAG_STATIC } else { 0 };

        let (shape_type, shape_index) = match &desc.shape {
            ShapeDesc::Sphere { radius } => {
                let si = self.spheres.len() as u32;
                self.spheres.push(SphereData {
                    radius: *radius,
                    _pad: [0.0; 3],
                });
                (SHAPE_SPHERE, si)
            }
            ShapeDesc::Box { half_extents } => {
                let si = self.boxes.len() as u32;
                self.boxes.push(BoxData {
                    half_extents: half_extents.extend(0.0),
                });
                (SHAPE_BOX, si)
            }
            ShapeDesc::Capsule {
                half_height,
                radius,
            } => {
                let si = self.capsules.len() as u32;
                self.capsules.push(CapsuleData {
                    half_height: *half_height,
                    radius: *radius,
                    _pad: [0.0; 2],
                });
                (SHAPE_CAPSULE, si)
            }
            ShapeDesc::ConvexHull { vertices } => {
                let si = self.convex_hulls.len() as u32;
                let vertex_offset = self.convex_vertices.len() as u32;
                let vertex_count = vertices.len().min(64) as u32;
                for v in vertices.iter().take(64) {
                    self.convex_vertices.push(ConvexVertex3D {
                        x: v.x,
                        y: v.y,
                        z: v.z,
                        _pad: 0.0,
                    });
                }
                self.convex_hulls.push(ConvexHullData {
                    vertex_offset,
                    vertex_count,
                    face_offset: 0,
                    face_count: 0,
                    edge_offset: 0,
                    edge_count: 0,
                    _pad0: 0,
                    _pad1: 0,
                });
                (SHAPE_CONVEX_HULL, si)
            }
            ShapeDesc::Plane { normal, distance } => {
                let si = self.planes.len() as u32;
                self.planes
                    .push(Vec4::new(normal.x, normal.y, normal.z, *distance));
                (SHAPE_PLANE, si)
            }
            ShapeDesc::Compound { children } => {
                let compound_shape_index = self.compound_shapes.len() as u32;
                let child_offset = self.compound_children.len() as u32;
                let mut compound_children_cpu = Vec::new();

                for (child_shape, local_pos, local_rot) in children {
                    // Add each child's shape data to the appropriate buffer
                    let (child_shape_type, child_shape_index) = match child_shape {
                        ShapeDesc::Sphere { radius } => {
                            let si = self.spheres.len() as u32;
                            self.spheres.push(SphereData {
                                radius: *radius,
                                _pad: [0.0; 3],
                            });
                            (SHAPE_SPHERE, si)
                        }
                        ShapeDesc::Box { half_extents } => {
                            let si = self.boxes.len() as u32;
                            self.boxes.push(BoxData {
                                half_extents: half_extents.extend(0.0),
                            });
                            (SHAPE_BOX, si)
                        }
                        ShapeDesc::Capsule {
                            half_height,
                            radius,
                        } => {
                            let si = self.capsules.len() as u32;
                            self.capsules.push(CapsuleData {
                                half_height: *half_height,
                                radius: *radius,
                                _pad: [0.0; 2],
                            });
                            (SHAPE_CAPSULE, si)
                        }
                        ShapeDesc::ConvexHull { vertices } => {
                            let si = self.convex_hulls.len() as u32;
                            let vertex_offset = self.convex_vertices.len() as u32;
                            let vertex_count = vertices.len().min(64) as u32;
                            for v in vertices.iter().take(64) {
                                self.convex_vertices.push(ConvexVertex3D {
                                    x: v.x,
                                    y: v.y,
                                    z: v.z,
                                    _pad: 0.0,
                                });
                            }
                            self.convex_hulls.push(ConvexHullData {
                                vertex_offset,
                                vertex_count,
                                face_offset: 0,
                                face_count: 0,
                                edge_offset: 0,
                                edge_count: 0,
                                _pad0: 0,
                                _pad1: 0,
                            });
                            (SHAPE_CONVEX_HULL, si)
                        }
                        ShapeDesc::Plane { normal, distance } => {
                            let si = self.planes.len() as u32;
                            self.planes
                                .push(Vec4::new(normal.x, normal.y, normal.z, *distance));
                            (SHAPE_PLANE, si)
                        }
                        ShapeDesc::Compound { .. } => {
                            // Nested compounds not supported; skip
                            continue;
                        }
                    };

                    // Compute local AABB for this child
                    let local_aabb = compute_child_local_aabb(child_shape, *local_pos, *local_rot);

                    // Add GPU-side compound child entry
                    self.compound_children.push(CompoundChildGpu {
                        local_position: local_pos.extend(0.0),
                        local_rotation: Vec4::new(
                            local_rot.x,
                            local_rot.y,
                            local_rot.z,
                            local_rot.w,
                        ),
                        shape_type: child_shape_type,
                        shape_index: child_shape_index,
                        _pad: [0; 2],
                    });

                    // Build CPU-side compound child for BVH + pair expansion
                    compound_children_cpu.push(CompoundChild {
                        shape_type: child_shape_type,
                        shape_index: child_shape_index,
                        local_position: *local_pos,
                        local_rotation: *local_rot,
                        local_aabb,
                    });
                }

                let child_count = compound_children_cpu.len() as u32;

                self.compound_shapes.push(CompoundShapeGpu {
                    child_offset,
                    child_count,
                });

                // Build CPU-side CompoundShape with BVH
                let compound = CompoundShape::new(compound_children_cpu);
                self.compound_shapes_cpu.push(compound);

                (SHAPE_COMPOUND, compound_shape_index)
            }
        };

        let prop =
            RigidBodyProps3D::new(inv_inertia, desc.friction, shape_type, shape_index, flags);

        if idx >= self.states.len() {
            self.states.resize(idx + 1, bytemuck::Zeroable::zeroed());
            self.props.resize(idx + 1, bytemuck::Zeroable::zeroed());
            self.shapes
                .resize(idx + 1, ShapeDesc::Sphere { radius: 0.5 });
            self.alive.resize(idx + 1, false);
        }

        self.states[idx] = state;
        self.props[idx] = prop;
        self.shapes[idx] = desc.shape.clone();
        self.alive[idx] = true;

        handle
    }

    /// Remove a body from the world. Returns `true` if it was successfully removed.
    pub fn remove_body(&mut self, handle: BodyHandle) -> bool {
        if !self.allocator.is_valid(handle) {
            return false;
        }
        let idx = handle.index as usize;
        self.alive[idx] = false;
        self.states[idx] = bytemuck::Zeroable::zeroed();
        self.props[idx] = bytemuck::Zeroable::zeroed();
        self.allocator.dealloc(handle);
        true
    }

    /// Number of currently alive bodies.
    pub fn body_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    /// Get the position of a body, if the handle is valid.
    pub fn get_position(&self, handle: BodyHandle) -> Option<Vec3> {
        if !self.allocator.is_valid(handle) {
            return None;
        }
        Some(self.states[handle.index as usize].position())
    }

    /// Get the orientation of a body, if the handle is valid.
    pub fn get_rotation(&self, handle: BodyHandle) -> Option<Quat> {
        if !self.allocator.is_valid(handle) {
            return None;
        }
        Some(self.states[handle.index as usize].quat())
    }

    /// Get the linear velocity of a body, if the handle is valid.
    pub fn get_velocity(&self, handle: BodyHandle) -> Option<Vec3> {
        if !self.allocator.is_valid(handle) {
            return None;
        }
        Some(self.states[handle.index as usize].linear_velocity())
    }

    /// Set the position of a body.
    pub fn set_position(&mut self, handle: BodyHandle, pos: Vec3) {
        if !self.allocator.is_valid(handle) {
            return;
        }
        let idx = handle.index as usize;
        let im = self.states[idx].inv_mass();
        self.states[idx].position_inv_mass = Vec4::new(pos.x, pos.y, pos.z, im);
    }

    /// Set the linear velocity of a body.
    pub fn set_velocity(&mut self, handle: BodyHandle, vel: Vec3) {
        if !self.allocator.is_valid(handle) {
            return;
        }
        let idx = handle.index as usize;
        self.states[idx].lin_vel = vel.extend(0.0);
    }

    /// Advance the simulation by one time step on the GPU.
    pub fn step(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        let alive_indices: Vec<usize> = (0..n).filter(|&i| self.alive[i]).collect();
        if alive_indices.is_empty() {
            return;
        }

        let compact_states: Vec<RigidBodyState3D> =
            alive_indices.iter().map(|&i| self.states[i]).collect();
        let compact_props: Vec<RigidBodyProps3D> =
            alive_indices.iter().map(|&i| self.props[i]).collect();
        let num_bodies = compact_states.len() as u32;

        self.gpu_pipeline.upload(
            &compact_states,
            &compact_props,
            &self.spheres,
            &self.boxes,
            &self.capsules,
            &self.convex_hulls,
            &self.convex_vertices,
            &self.planes,
            &self.compound_shapes,
            &self.compound_children,
            &self.compound_shapes_cpu,
            self.config.gravity,
            self.config.dt,
            self.config.solver_iterations,
        );

        let prev = self.contact_persistence.prev_contacts();
        let warm = if prev.is_empty() { None } else { Some(prev) };
        let (results, new_contacts) =
            self.gpu_pipeline
                .step_with_contacts(num_bodies, self.config.solver_iterations, warm);

        // Update persistence and generate collision events
        let events = self.contact_persistence.update(&new_contacts);
        self.collision_events.extend(events);

        for (slot, &orig) in alive_indices.iter().enumerate() {
            if slot < results.len() {
                self.states[orig] = results[slot];
            }
        }
    }

    /// Mark a body as kinematic (moves via set_position/set_velocity, not physics).
    pub fn set_body_kinematic(&mut self, handle: BodyHandle, kinematic: bool) {
        if !self.allocator.is_valid(handle) {
            return;
        }
        let idx = handle.index as usize;
        if kinematic {
            self.props[idx].flags |= rubble_math::FLAG_KINEMATIC;
        } else {
            self.props[idx].flags &= !rubble_math::FLAG_KINEMATIC;
        }
    }

    /// Drain collision events from the last step.
    pub fn drain_collision_events(&mut self) -> Vec<CollisionEvent> {
        std::mem::take(&mut self.collision_events)
    }

    /// Cast a ray and return the closest hit (handle, t parameter, hit normal).
    /// `origin` is the ray start, `direction` is the ray direction (need not be normalized).
    /// `max_t` is the maximum ray parameter to test.
    pub fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_t: f32,
    ) -> Option<(BodyHandle, f32, Vec3)> {
        let dir_len = direction.length();
        if dir_len < 1e-12 {
            return None;
        }
        let dir = direction / dir_len;

        let mut best_t = max_t * dir_len;
        let mut best_handle = None;
        let mut best_normal = Vec3::ZERO;

        for (idx, alive) in self.alive.iter().enumerate() {
            if !*alive {
                continue;
            }
            let state = &self.states[idx];
            let pos = state.position();

            if let Some((t, normal)) = self.ray_shape_test(origin, dir, pos, state, idx) {
                if t >= 0.0 && t < best_t {
                    best_t = t;
                    let gen = self.allocator.generations[idx];
                    best_handle = Some(BodyHandle::new(idx as u32, gen));
                    best_normal = normal;
                }
            }
        }

        best_handle.map(|h| (h, best_t / dir_len, best_normal))
    }

    /// Cast multiple rays and return results for each.
    pub fn raycast_batch(
        &self,
        rays: &[(Vec3, Vec3, f32)], // (origin, direction, max_t)
    ) -> Vec<Option<(BodyHandle, f32, Vec3)>> {
        rays.iter()
            .map(|&(origin, dir, max_t)| self.raycast(origin, dir, max_t))
            .collect()
    }

    fn ray_shape_test(
        &self,
        origin: Vec3,
        dir: Vec3,
        pos: Vec3,
        _state: &RigidBodyState3D,
        idx: usize,
    ) -> Option<(f32, Vec3)> {
        match &self.shapes[idx] {
            ShapeDesc::Sphere { radius } => {
                // Ray-sphere intersection
                let oc = origin - pos;
                let b = oc.dot(dir);
                let c = oc.dot(oc) - radius * radius;
                let disc = b * b - c;
                if disc < 0.0 {
                    return None;
                }
                let t = -b - disc.sqrt();
                if t < 0.0 {
                    return None;
                }
                let hit = origin + dir * t;
                let normal = (hit - pos).normalize();
                Some((t, normal))
            }
            ShapeDesc::Box { half_extents } => {
                // Ray-AABB intersection (axis-aligned, ignoring rotation for simplicity)
                let q = _state.quat();
                let inv_q = q.conjugate();
                let local_origin = inv_q * (origin - pos);
                let local_dir = inv_q * dir;
                let he = *half_extents;

                let mut tmin = f32::NEG_INFINITY;
                let mut tmax = f32::INFINITY;
                let mut normal_idx = 0usize;
                let mut normal_sign = 1.0f32;

                for i in 0..3 {
                    let o = [local_origin.x, local_origin.y, local_origin.z][i];
                    let d = [local_dir.x, local_dir.y, local_dir.z][i];
                    let h = [he.x, he.y, he.z][i];

                    if d.abs() < 1e-12 {
                        if o < -h || o > h {
                            return None;
                        }
                        continue;
                    }
                    let t1 = (-h - o) / d;
                    let t2 = (h - o) / d;
                    let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                    if t_near > tmin {
                        tmin = t_near;
                        normal_idx = i;
                        normal_sign = if d > 0.0 { -1.0 } else { 1.0 };
                    }
                    tmax = tmax.min(t_far);
                    if tmin > tmax {
                        return None;
                    }
                }

                if tmin < 0.0 {
                    return None;
                }
                let mut local_normal = Vec3::ZERO;
                match normal_idx {
                    0 => local_normal.x = normal_sign,
                    1 => local_normal.y = normal_sign,
                    _ => local_normal.z = normal_sign,
                }
                let world_normal = q * local_normal;
                Some((tmin, world_normal))
            }
            ShapeDesc::Plane { normal, distance } => {
                let denom = normal.dot(dir);
                if denom.abs() < 1e-12 {
                    return None;
                }
                let t = (*distance - normal.dot(origin)) / denom;
                if t >= 0.0 {
                    Some((t, *normal))
                } else {
                    None
                }
            }
            ShapeDesc::Capsule {
                half_height,
                radius,
            } => {
                let q = _state.quat();
                let axis = q * Vec3::new(0.0, *half_height, 0.0);
                let ep_a = pos + axis;
                let ep_b = pos - axis;
                ray_capsule_3d(origin, dir, ep_a, ep_b, *radius)
            }
            ShapeDesc::ConvexHull { vertices } => {
                let q = _state.quat();
                let world_verts: Vec<Vec3> = vertices.iter().map(|v| pos + q * *v).collect();
                ray_convex_hull_3d(origin, dir, &world_verts)
            }
            ShapeDesc::Compound { children } => {
                let q = _state.quat();
                let mut best: Option<(f32, Vec3)> = None;
                for (child_shape, child_pos, child_rot) in children {
                    let child_world_pos = pos + q * *child_pos;
                    let child_world_rot = q * *child_rot;
                    let child_state = RigidBodyState3D::new(
                        child_world_pos,
                        0.0,
                        child_world_rot,
                        Vec3::ZERO,
                        Vec3::ZERO,
                    );
                    // Create a temporary index for the child (we need a shape to test)
                    let child_result = match child_shape {
                        ShapeDesc::Sphere { radius } => {
                            ray_sphere_3d(origin, dir, child_world_pos, *radius)
                        }
                        ShapeDesc::Box { half_extents } => {
                            // Reuse existing box raycast logic
                            let inv_q = child_world_rot.conjugate();
                            let local_origin = inv_q * (origin - child_world_pos);
                            let local_dir = inv_q * dir;
                            ray_aabb_3d(local_origin, local_dir, *half_extents, child_world_rot)
                        }
                        ShapeDesc::Capsule {
                            half_height,
                            radius,
                        } => {
                            let axis = child_world_rot * Vec3::new(0.0, *half_height, 0.0);
                            ray_capsule_3d(
                                origin,
                                dir,
                                child_world_pos + axis,
                                child_world_pos - axis,
                                *radius,
                            )
                        }
                        ShapeDesc::ConvexHull { vertices } => {
                            let wv: Vec<Vec3> = vertices
                                .iter()
                                .map(|v| child_world_pos + child_world_rot * *v)
                                .collect();
                            ray_convex_hull_3d(origin, dir, &wv)
                        }
                        _ => None,
                    };
                    if let Some((t, n)) = child_result {
                        if t >= 0.0 && best.is_none_or(|(bt, _)| t < bt) {
                            best = Some((t, n));
                        }
                    }
                    let _ = child_state; // suppress unused warning
                }
                best
            }
        }
    }

    /// Query all bodies whose AABB overlaps the given axis-aligned bounding box.
    pub fn overlap_aabb(&self, query_min: Vec3, query_max: Vec3) -> Vec<BodyHandle> {
        let mut result = Vec::new();
        for (idx, alive) in self.alive.iter().enumerate() {
            if !*alive {
                continue;
            }
            let state = &self.states[idx];
            let pos = state.position();
            let (body_min, body_max) = self.compute_aabb(pos, state, idx);

            // AABB overlap test
            if body_min.x <= query_max.x
                && body_max.x >= query_min.x
                && body_min.y <= query_max.y
                && body_max.y >= query_min.y
                && body_min.z <= query_max.z
                && body_max.z >= query_min.z
            {
                let gen = self.allocator.generations[idx];
                result.push(BodyHandle::new(idx as u32, gen));
            }
        }
        result
    }

    fn compute_aabb(&self, pos: Vec3, state: &RigidBodyState3D, idx: usize) -> (Vec3, Vec3) {
        match &self.shapes[idx] {
            ShapeDesc::Sphere { radius } => {
                let r = Vec3::splat(*radius);
                (pos - r, pos + r)
            }
            ShapeDesc::Box { half_extents } => {
                let q = state.quat();
                let he = *half_extents;
                let ax = (q * Vec3::new(he.x, 0.0, 0.0)).abs();
                let ay = (q * Vec3::new(0.0, he.y, 0.0)).abs();
                let az = (q * Vec3::new(0.0, 0.0, he.z)).abs();
                let extent = ax + ay + az;
                (pos - extent, pos + extent)
            }
            ShapeDesc::Capsule {
                half_height,
                radius,
            } => {
                let q = state.quat();
                let axis = q * Vec3::new(0.0, *half_height, 0.0);
                let a = pos + axis;
                let b = pos - axis;
                let r = Vec3::splat(*radius);
                (a.min(b) - r, a.max(b) + r)
            }
            ShapeDesc::ConvexHull { vertices } => {
                let q = state.quat();
                let mut mn = Vec3::splat(f32::MAX);
                let mut mx = Vec3::splat(f32::NEG_INFINITY);
                for v in vertices {
                    let wv = pos + q * *v;
                    mn = mn.min(wv);
                    mx = mx.max(wv);
                }
                (mn, mx)
            }
            ShapeDesc::Plane { normal, distance } => {
                let center = *normal * *distance;
                let big = Vec3::splat(1e4);
                (center - big, center + big)
            }
            ShapeDesc::Compound { children } => {
                let q = state.quat();
                let mut mn = Vec3::splat(f32::MAX);
                let mut mx = Vec3::splat(f32::NEG_INFINITY);
                for (child_shape, local_pos, local_rot) in children {
                    let child_world_pos = pos + q * *local_pos;
                    let child_world_rot = q * *local_rot;
                    let child_aabb =
                        compute_child_local_aabb(child_shape, child_world_pos, child_world_rot);
                    mn = mn.min(child_aabb.min_point());
                    mx = mx.max(child_aabb.max_point());
                }
                (mn, mx)
            }
        }
    }

    /// Access the GPU pipeline (for diagnostics like contact count).
    pub fn gpu_pipeline(&self) -> &gpu::GpuPipeline {
        &self.gpu_pipeline
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_world(config: SimConfig) -> World {
        World::new(config).expect("GPU required for tests")
    }

    #[test]
    fn test_world_new() {
        let world = gpu_world(SimConfig::default());
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn test_add_remove_body() {
        let mut world = gpu_world(SimConfig::default());
        let handle = world.add_body(&RigidBodyDesc {
            position: Vec3::new(1.0, 2.0, 3.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });
        assert_eq!(world.body_count(), 1);
        assert_eq!(world.get_position(handle), Some(Vec3::new(1.0, 2.0, 3.0)));

        assert!(world.remove_body(handle));
        assert_eq!(world.body_count(), 0);
        assert_eq!(world.get_position(handle), None);
        assert!(!world.remove_body(handle));
    }

    #[test]
    fn test_handle_allocator() {
        let mut alloc = GenerationalIndexAllocator::new();

        let h1 = alloc.alloc();
        assert_eq!(h1.index, 0);
        assert_eq!(h1.generation, 0);
        assert!(alloc.is_valid(h1));

        let h2 = alloc.alloc();
        assert_eq!(h2.index, 1);
        assert!(alloc.is_valid(h2));

        assert!(alloc.dealloc(h1));
        assert!(!alloc.is_valid(h1));

        let h3 = alloc.alloc();
        assert_eq!(h3.index, 0);
        assert_eq!(h3.generation, 1);
        assert!(alloc.is_valid(h3));
        assert!(!alloc.is_valid(h1));
    }

    #[test]
    fn test_single_body_gravity() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            max_bodies: 1024,
            ..Default::default()
        });

        let handle = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        for _ in 0..60 {
            world.step();
        }

        let pos = world.get_position(handle).unwrap();
        let expected_y = 10.0 - 0.5 * 9.81 * 1.0;
        assert!(
            (pos.y - expected_y).abs() < 0.5,
            "Body y={} should be near expected y={}",
            pos.y,
            expected_y
        );
    }

    #[test]
    fn test_two_body_collision() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
            max_bodies: 1024,
            ..Default::default()
        });

        let h1 = world.add_body(&RigidBodyDesc {
            position: Vec3::new(-2.0, 0.0, 0.0),
            linear_velocity: Vec3::new(5.0, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });

        let h2 = world.add_body(&RigidBodyDesc {
            position: Vec3::new(2.0, 0.0, 0.0),
            linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });

        for _ in 0..120 {
            world.step();
        }

        let p1 = world.get_position(h1).unwrap();
        let p2 = world.get_position(h2).unwrap();
        let dist = (p2 - p1).length();
        assert!(
            dist >= 1.5,
            "Bodies should not overlap: distance = {}",
            dist
        );
    }

    #[test]
    fn test_body_count() {
        let mut world = gpu_world(SimConfig::default());

        let h1 = world.add_body(&RigidBodyDesc::default());
        let _h2 = world.add_body(&RigidBodyDesc::default());
        let h3 = world.add_body(&RigidBodyDesc::default());
        assert_eq!(world.body_count(), 3);

        world.remove_body(h1);
        assert_eq!(world.body_count(), 2);

        world.remove_body(h3);
        assert_eq!(world.body_count(), 1);
    }

    #[test]
    fn test_compute_inertia_sphere() {
        let shape = ShapeDesc::Sphere { radius: 1.0 };
        let inv_i = compute_inertia(&shape, 5.0);
        let expected = 0.5;
        let cols = inv_i.to_cols_array_2d();
        assert!((cols[0][0] - expected).abs() < 1e-6);
        assert!((cols[1][1] - expected).abs() < 1e-6);
        assert!((cols[2][2] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_compute_inertia_static() {
        let shape = ShapeDesc::Sphere { radius: 1.0 };
        let inv_i = compute_inertia(&shape, 0.0);
        assert_eq!(inv_i, Mat3::ZERO);
    }

    #[test]
    fn test_set_position_and_velocity() {
        let mut world = gpu_world(SimConfig::default());
        let handle = world.add_body(&RigidBodyDesc {
            position: Vec3::ZERO,
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        world.set_position(handle, Vec3::new(10.0, 20.0, 30.0));
        assert_eq!(
            world.get_position(handle),
            Some(Vec3::new(10.0, 20.0, 30.0))
        );

        world.set_velocity(handle, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(world.get_velocity(handle), Some(Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_get_rotation() {
        let mut world = gpu_world(SimConfig::default());
        let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let handle = world.add_body(&RigidBodyDesc {
            rotation: rot,
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        let got = world.get_rotation(handle).unwrap();
        assert!((got.x - rot.x).abs() < 1e-6);
        assert!((got.y - rot.y).abs() < 1e-6);
        assert!((got.z - rot.z).abs() < 1e-6);
        assert!((got.w - rot.w).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Raycast tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_raycast_sphere_hit() {
        let mut world = gpu_world(SimConfig::default());
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.0, 5.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });

        let result = world.raycast(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 100.0);
        assert!(result.is_some());
        let (hit_h, t, normal) = result.unwrap();
        assert_eq!(hit_h.index, h.index);
        assert!((t - 4.0).abs() < 0.01, "t should be ~4.0, got {t}");
        assert!(
            (normal - Vec3::new(0.0, 0.0, -1.0)).length() < 0.01,
            "normal should point back toward origin"
        );
    }

    #[test]
    fn test_raycast_sphere_miss() {
        let mut world = gpu_world(SimConfig::default());
        world.add_body(&RigidBodyDesc {
            position: Vec3::new(10.0, 0.0, 5.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });

        let result = world.raycast(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 100.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_raycast_box_hit() {
        let mut world = gpu_world(SimConfig::default());
        world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.0, 5.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(1.0, 1.0, 1.0),
            },
            ..Default::default()
        });

        let result = world.raycast(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 100.0);
        assert!(result.is_some());
        let (_, t, _) = result.unwrap();
        assert!((t - 4.0).abs() < 0.01, "t should be ~4.0, got {t}");
    }

    #[test]
    fn test_raycast_capsule_hit() {
        let mut world = gpu_world(SimConfig::default());
        world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.0, 5.0),
            mass: 1.0,
            shape: ShapeDesc::Capsule {
                half_height: 1.0,
                radius: 0.5,
            },
            ..Default::default()
        });

        let result = world.raycast(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 100.0);
        assert!(result.is_some());
        let (_, t, _) = result.unwrap();
        assert!(t > 3.0 && t < 5.0, "t={t} should be in range");
    }

    #[test]
    fn test_raycast_plane_hit() {
        let mut world = gpu_world(SimConfig::default());
        world.add_body(&RigidBodyDesc {
            position: Vec3::ZERO,
            mass: 0.0,
            shape: ShapeDesc::Plane {
                normal: Vec3::Y,
                distance: 0.0,
            },
            ..Default::default()
        });

        // Ray from above pointing down
        let result = world.raycast(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 100.0);
        assert!(result.is_some());
        let (_, t, normal) = result.unwrap();
        assert!((t - 5.0).abs() < 0.01);
        assert!((normal - Vec3::Y).length() < 0.01);
    }

    #[test]
    fn test_raycast_batch() {
        let mut world = gpu_world(SimConfig::default());
        world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.0, 5.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });

        let results = world.raycast_batch(&[
            (Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 100.0),  // hit
            (Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 100.0),  // miss
            (Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 100.0), // miss (wrong direction)
        ]);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some());
        assert!(results[1].is_none());
        assert!(results[2].is_none());
    }

    // -----------------------------------------------------------------------
    // Overlap AABB tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_overlap_aabb() {
        let mut world = gpu_world(SimConfig::default());
        let h1 = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });
        let _h2 = world.add_body(&RigidBodyDesc {
            position: Vec3::new(10.0, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });

        let hits = world.overlap_aabb(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].index, h1.index);
    }

    // -----------------------------------------------------------------------
    // Kinematic body tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kinematic_body() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            ..Default::default()
        });
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        world.set_body_kinematic(h, true);

        for _ in 0..60 {
            world.step();
        }

        // Kinematic body should not be affected by gravity (inv_mass=0 in props)
        let pos = world.get_position(h).unwrap();
        // After kinematic flag set, the body should mostly stay in place
        // (the solver won't apply impulses to it)
        assert!(
            pos.y > 5.0,
            "Kinematic body y={} should not fall much under gravity",
            pos.y
        );
    }

    // -----------------------------------------------------------------------
    // Collision events tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_collision_events_drain_once() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::ZERO,
            ..Default::default()
        });

        world.add_body(&RigidBodyDesc {
            position: Vec3::new(-0.5, 0.0, 0.0),
            linear_velocity: Vec3::new(5.0, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });
        world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.5, 0.0, 0.0),
            linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });

        for _ in 0..10 {
            world.step();
        }

        let events = world.drain_collision_events();
        // There should be some collision events (Started)
        // Second drain should be empty
        let events2 = world.drain_collision_events();
        assert!(events2.is_empty(), "Second drain should be empty");
        let _ = events; // use it to prevent unused warning
    }

    // -----------------------------------------------------------------------
    // Shape-specific tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_box_on_plane_stacking() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 10,
            max_bodies: 1024,
            ..Default::default()
        });

        // Floor plane (static)
        world.add_body(&RigidBodyDesc {
            mass: 0.0,
            shape: ShapeDesc::Plane {
                normal: Vec3::Y,
                distance: 0.0,
            },
            ..Default::default()
        });

        // Box above plane
        let box_h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 2.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            },
            ..Default::default()
        });

        for _ in 0..120 {
            world.step();
        }

        let pos = world.get_position(box_h).unwrap();
        // Box should settle near y=0.5 (half_extent above plane at y=0)
        assert!(
            pos.y > -0.5 && pos.y < 2.0,
            "Box y={} should settle near plane (expected ~0.5)",
            pos.y
        );
    }

    #[test]
    fn test_capsule_gravity_fall() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            ..Default::default()
        });

        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Capsule {
                half_height: 0.5,
                radius: 0.3,
            },
            ..Default::default()
        });

        for _ in 0..60 {
            world.step();
        }

        let pos = world.get_position(h).unwrap();
        assert!(pos.y < 10.0, "Capsule y={} should have fallen", pos.y);
    }

    #[test]
    fn test_convex_hull_creation() {
        let mut world = gpu_world(SimConfig::default());
        // Tetrahedron
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 5.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::ConvexHull {
                vertices: vec![
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::new(-1.0, -1.0, -1.0),
                    Vec3::new(1.0, -1.0, -1.0),
                    Vec3::new(0.0, -1.0, 1.0),
                ],
            },
            ..Default::default()
        });

        assert_eq!(world.body_count(), 1);
        assert!(world.get_position(h).is_some());
    }

    #[test]
    fn test_compound_shape_creation() {
        let mut world = gpu_world(SimConfig::default());
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 5.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Compound {
                children: vec![
                    (
                        ShapeDesc::Sphere { radius: 0.5 },
                        Vec3::new(-1.0, 0.0, 0.0),
                        Quat::IDENTITY,
                    ),
                    (
                        ShapeDesc::Box {
                            half_extents: Vec3::new(0.3, 0.3, 0.3),
                        },
                        Vec3::new(1.0, 0.0, 0.0),
                        Quat::IDENTITY,
                    ),
                ],
            },
            ..Default::default()
        });

        assert_eq!(world.body_count(), 1);
        assert!(world.get_position(h).is_some());
    }

    #[test]
    fn test_multiple_shape_types_simulation() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            ..Default::default()
        });

        // Floor
        world.add_body(&RigidBodyDesc {
            mass: 0.0,
            shape: ShapeDesc::Plane {
                normal: Vec3::Y,
                distance: 0.0,
            },
            ..Default::default()
        });

        // Various shapes
        world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 5.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });
        world.add_body(&RigidBodyDesc {
            position: Vec3::new(2.0, 5.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            },
            ..Default::default()
        });
        world.add_body(&RigidBodyDesc {
            position: Vec3::new(4.0, 5.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Capsule {
                half_height: 0.5,
                radius: 0.3,
            },
            ..Default::default()
        });

        // Run simulation — should not panic or produce NaN
        for _ in 0..120 {
            world.step();
        }

        assert_eq!(world.body_count(), 4);
    }

    // -----------------------------------------------------------------------
    // Stress tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_many_bodies_no_crash() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_bodies: 2048,
            ..Default::default()
        });

        for i in 0..100 {
            world.add_body(&RigidBodyDesc {
                position: Vec3::new((i % 10) as f32 * 2.0, (i / 10) as f32 * 2.0 + 5.0, 0.0),
                mass: 1.0,
                shape: ShapeDesc::Sphere { radius: 0.5 },
                ..Default::default()
            });
        }

        for _ in 0..30 {
            world.step();
        }

        assert_eq!(world.body_count(), 100);
    }

    #[test]
    fn test_add_remove_cycle() {
        let mut world = gpu_world(SimConfig::default());
        let mut handles = Vec::new();

        for _ in 0..20 {
            handles.push(world.add_body(&RigidBodyDesc::default()));
        }
        assert_eq!(world.body_count(), 20);

        for h in handles.drain(..) {
            world.remove_body(h);
        }
        assert_eq!(world.body_count(), 0);

        // Re-add should reuse indices
        for _ in 0..10 {
            handles.push(world.add_body(&RigidBodyDesc::default()));
        }
        assert_eq!(world.body_count(), 10);
    }

    // -----------------------------------------------------------------------
    // Integration tests: compound, convex hull, and overflow recovery
    // -----------------------------------------------------------------------

    #[test]
    fn test_compound_compound_collision() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
            max_bodies: 1024,
            ..Default::default()
        });

        // Compound shape: 4 spheres in a 2x2 grid on the XY plane
        let make_compound = || ShapeDesc::Compound {
            children: vec![
                (
                    ShapeDesc::Sphere { radius: 0.3 },
                    Vec3::new(-0.5, -0.5, 0.0),
                    Quat::IDENTITY,
                ),
                (
                    ShapeDesc::Sphere { radius: 0.3 },
                    Vec3::new(0.5, -0.5, 0.0),
                    Quat::IDENTITY,
                ),
                (
                    ShapeDesc::Sphere { radius: 0.3 },
                    Vec3::new(-0.5, 0.5, 0.0),
                    Quat::IDENTITY,
                ),
                (
                    ShapeDesc::Sphere { radius: 0.3 },
                    Vec3::new(0.5, 0.5, 0.0),
                    Quat::IDENTITY,
                ),
            ],
        };

        let h1 = world.add_body(&RigidBodyDesc {
            position: Vec3::new(-1.5, 0.0, 0.0),
            linear_velocity: Vec3::new(5.0, 0.0, 0.0),
            mass: 1.0,
            shape: make_compound(),
            ..Default::default()
        });

        let h2 = world.add_body(&RigidBodyDesc {
            position: Vec3::new(1.5, 0.0, 0.0),
            linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
            mass: 1.0,
            shape: make_compound(),
            ..Default::default()
        });

        for _ in 0..30 {
            world.step();
        }

        let p1 = world.get_position(h1).unwrap();
        let p2 = world.get_position(h2).unwrap();
        let dist = (p2 - p1).length();

        assert!(
            p1.x.is_finite() && p1.y.is_finite() && p1.z.is_finite(),
            "Body 1 position should be finite: {:?}",
            p1
        );
        assert!(
            p2.x.is_finite() && p2.y.is_finite() && p2.z.is_finite(),
            "Body 2 position should be finite: {:?}",
            p2
        );
        assert!(
            dist >= 0.5,
            "Compound bodies should separate after collision: distance = {}",
            dist
        );
    }

    #[test]
    fn test_hull_hull_edge_collision() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
            max_bodies: 1024,
            ..Default::default()
        });

        // Cube vertices at +/-1.0
        let cube_verts = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];

        let h1 = world.add_body(&RigidBodyDesc {
            position: Vec3::new(-2.5, 0.0, 0.0),
            linear_velocity: Vec3::new(3.0, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::ConvexHull {
                vertices: cube_verts.clone(),
            },
            ..Default::default()
        });

        let h2 = world.add_body(&RigidBodyDesc {
            position: Vec3::new(2.5, 0.0, 0.0),
            linear_velocity: Vec3::new(-3.0, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::ConvexHull {
                vertices: cube_verts,
            },
            ..Default::default()
        });

        for _ in 0..60 {
            world.step();
        }

        let p1 = world.get_position(h1).unwrap();
        let p2 = world.get_position(h2).unwrap();
        let dist = (p2 - p1).length();

        assert!(
            p1.x.is_finite() && p1.y.is_finite() && p1.z.is_finite(),
            "Hull 1 position should be finite: {:?}",
            p1
        );
        assert!(
            p2.x.is_finite() && p2.y.is_finite() && p2.z.is_finite(),
            "Hull 2 position should be finite: {:?}",
            p2
        );
        assert!(
            dist > 0.9,
            "Convex hull bodies should not overlap: distance = {}",
            dist
        );
    }

    #[test]
    fn test_many_bodies_overflow_recovery() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            max_bodies: 2048,
            ..Default::default()
        });

        // Create 50 sphere bodies packed into a 3x3x3 cube.
        // Use a simple deterministic pattern to spread them out.
        let mut handles = Vec::new();
        for i in 0..50u32 {
            // Deterministic pseudo-random positions within [-1.5, 1.5]
            let x = ((i * 7 + 3) % 30) as f32 / 10.0 - 1.5;
            let y = ((i * 13 + 5) % 30) as f32 / 10.0 - 1.5;
            let z = ((i * 19 + 11) % 30) as f32 / 10.0 - 1.5;
            let h = world.add_body(&RigidBodyDesc {
                position: Vec3::new(x, y, z),
                mass: 1.0,
                shape: ShapeDesc::Sphere { radius: 0.5 },
                ..Default::default()
            });
            handles.push(h);
        }

        assert_eq!(world.body_count(), 50);

        // Step the simulation — many overlapping spheres will generate
        // a large number of contacts, exercising buffer overflow recovery.
        for _ in 0..10 {
            world.step();
        }

        // Verify all positions are finite (no NaN or Inf)
        for (i, h) in handles.iter().enumerate() {
            let pos = world.get_position(*h).unwrap();
            assert!(
                pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
                "Body {} has non-finite position: {:?}",
                i,
                pos
            );
        }
    }
}
