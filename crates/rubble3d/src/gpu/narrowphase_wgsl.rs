/// WGSL source for GPU narrowphase contact generation.
///
/// For each broadphase pair, detects collisions based on shape types
/// (sphere-sphere, sphere-box, box-box) and writes contacts to the output
/// buffer using an atomic counter.
pub const NARROWPHASE_WGSL: &str = r#"
// ---------- Types ----------

struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

struct BodyProps {
    inv_inertia_row0: vec4<f32>,
    inv_inertia_row1: vec4<f32>,
    inv_inertia_row2: vec4<f32>,
    friction:         f32,
    shape_type:       u32,
    shape_index:      u32,
    flags:            u32,
};

struct SphereData {
    radius: f32,
    _pad0:  f32,
    _pad1:  f32,
    _pad2:  f32,
};

struct BoxDataGpu {
    half_extents: vec4<f32>,
};

struct Pair {
    a: u32,
    b: u32,
};

struct Contact {
    point:      vec4<f32>,
    normal:     vec4<f32>,
    body_a:     u32,
    body_b:     u32,
    feature_id: u32,
    _pad:       u32,
    lambda_n:   f32,
    lambda_t1:  f32,
    lambda_t2:  f32,
    penalty_k:  f32,
};

struct SimParams {
    gravity:           vec4<f32>,
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
};

const SHAPE_SPHERE: u32 = 0u;
const SHAPE_BOX:    u32 = 1u;

@group(0) @binding(0) var<storage, read>       bodies:        array<Body>;
@group(0) @binding(1) var<storage, read>       props:         array<BodyProps>;
@group(0) @binding(2) var<storage, read>       pairs:         array<Pair>;
@group(0) @binding(3) var<storage, read>       pair_count_in: array<u32>;
@group(0) @binding(4) var<storage, read>       spheres:       array<SphereData>;
@group(0) @binding(5) var<storage, read>       boxes:         array<BoxDataGpu>;
@group(0) @binding(6) var<storage, read_write> contacts:      array<Contact>;
@group(0) @binding(7) var<storage, read_write> contact_count: atomic<u32>;
@group(0) @binding(8) var<uniform>             params:        SimParams;

// Quaternion rotation of a vector: q * v * conj(q)
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

fn emit_contact(
    point: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
    body_a: u32,
    body_b: u32,
    max_contacts: u32,
) {
    let slot = atomicAdd(&contact_count, 1u);
    if slot >= max_contacts {
        return;
    }
    contacts[slot].point  = vec4<f32>(point, depth);
    contacts[slot].normal = vec4<f32>(normal, 0.0);
    contacts[slot].body_a = body_a;
    contacts[slot].body_b = body_b;
    contacts[slot].feature_id = 0u;
    contacts[slot]._pad = 0u;
    contacts[slot].lambda_n = 0.0;
    contacts[slot].lambda_t1 = 0.0;
    contacts[slot].lambda_t2 = 0.0;
    contacts[slot].penalty_k = 1e4;
}

fn sphere_sphere_test(
    pos_a: vec3<f32>, radius_a: f32,
    pos_b: vec3<f32>, radius_b: f32,
    body_a: u32, body_b: u32,
    max_contacts: u32,
) {
    let diff = pos_b - pos_a;
    let dist2 = dot(diff, diff);
    let sum_r = radius_a + radius_b;
    if dist2 >= sum_r * sum_r || dist2 < 1e-12 {
        return;
    }
    let dist = sqrt(dist2);
    let normal = diff / dist;
    let depth = dist - sum_r; // negative when penetrating
    let point = pos_a + normal * (radius_a + depth * 0.5);
    emit_contact(point, normal, depth, body_a, body_b, max_contacts);
}

fn sphere_box_test(
    sphere_pos: vec3<f32>, radius: f32,
    box_pos: vec3<f32>, box_rot: vec4<f32>, half_ext: vec3<f32>,
    body_sphere: u32, body_box: u32,
    max_contacts: u32,
) {
    // Transform sphere center into box local space
    let inv_rot = vec4<f32>(-box_rot.xyz, box_rot.w);
    let local_center = quat_rotate(inv_rot, sphere_pos - box_pos);

    // Clamp to box
    let closest = clamp(local_center, -half_ext, half_ext);
    let diff = local_center - closest;
    let dist2 = dot(diff, diff);

    if dist2 >= radius * radius {
        return;
    }

    var normal_local: vec3<f32>;
    var depth: f32;

    if dist2 > 1e-12 {
        // Sphere center outside box
        let dist = sqrt(dist2);
        normal_local = diff / dist;
        depth = dist - radius;
    } else {
        // Sphere center inside box — find closest face
        let face_dists = half_ext - abs(local_center);
        let min_dist = min(face_dists.x, min(face_dists.y, face_dists.z));
        if face_dists.x <= min_dist + 1e-6 {
            normal_local = vec3<f32>(select(-1.0, 1.0, local_center.x > 0.0), 0.0, 0.0);
        } else if face_dists.y <= min_dist + 1e-6 {
            normal_local = vec3<f32>(0.0, select(-1.0, 1.0, local_center.y > 0.0), 0.0);
        } else {
            normal_local = vec3<f32>(0.0, 0.0, select(-1.0, 1.0, local_center.z > 0.0));
        }
        depth = -(min_dist + radius);
    }

    // Negate normal: geometric normal points box→sphere (B→A), but convention is A→B
    let normal_world = -quat_rotate(box_rot, normal_local);
    let contact_point = sphere_pos + normal_world * (radius + depth * 0.5);
    emit_contact(contact_point, normal_world, depth, body_sphere, body_box, max_contacts);
}

fn box_box_test(
    pos_a: vec3<f32>, rot_a: vec4<f32>, he_a: vec3<f32>,
    pos_b: vec3<f32>, rot_b: vec4<f32>, he_b: vec3<f32>,
    body_a: u32, body_b: u32,
    max_contacts: u32,
) {
    // SAT test with 15 axes (6 face normals + 9 edge-edge)
    // We use a simplified version that checks face axes only for GPU efficiency.
    let axes_a = array<vec3<f32>, 3>(
        quat_rotate(rot_a, vec3<f32>(1.0, 0.0, 0.0)),
        quat_rotate(rot_a, vec3<f32>(0.0, 1.0, 0.0)),
        quat_rotate(rot_a, vec3<f32>(0.0, 0.0, 1.0)),
    );
    let axes_b = array<vec3<f32>, 3>(
        quat_rotate(rot_b, vec3<f32>(1.0, 0.0, 0.0)),
        quat_rotate(rot_b, vec3<f32>(0.0, 1.0, 0.0)),
        quat_rotate(rot_b, vec3<f32>(0.0, 0.0, 1.0)),
    );

    let d = pos_b - pos_a;
    var min_depth = -1e30;
    var best_normal = vec3<f32>(0.0, 1.0, 0.0);

    let he_arr_a = array<f32, 3>(he_a.x, he_a.y, he_a.z);
    let he_arr_b = array<f32, 3>(he_b.x, he_b.y, he_b.z);

    // Test 6 face axes (3 from A, 3 from B)
    for (var i = 0u; i < 3u; i = i + 1u) {
        // Axis from A
        let axis = axes_a[i];
        let proj_a = he_arr_a[i];
        let proj_b = abs(dot(axes_b[0], axis)) * he_arr_b[0]
                   + abs(dot(axes_b[1], axis)) * he_arr_b[1]
                   + abs(dot(axes_b[2], axis)) * he_arr_b[2];
        let center_proj = dot(d, axis);
        let overlap = proj_a + proj_b - abs(center_proj);
        if overlap < 0.0 {
            return; // separated
        }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            best_normal = axis * select(1.0, -1.0, center_proj < 0.0);
        }
    }
    for (var i = 0u; i < 3u; i = i + 1u) {
        // Axis from B
        let axis = axes_b[i];
        let proj_a = abs(dot(axes_a[0], axis)) * he_arr_a[0]
                   + abs(dot(axes_a[1], axis)) * he_arr_a[1]
                   + abs(dot(axes_a[2], axis)) * he_arr_a[2];
        let proj_b = he_arr_b[i];
        let center_proj = dot(d, axis);
        let overlap = proj_a + proj_b - abs(center_proj);
        if overlap < 0.0 {
            return;
        }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            best_normal = axis * select(1.0, -1.0, center_proj < 0.0);
        }
    }

    // Contact point: midpoint along best normal direction
    let contact_point = (pos_a + pos_b) * 0.5 + best_normal * min_depth * 0.5;
    emit_contact(contact_point, best_normal, min_depth, body_a, body_b, max_contacts);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pi = gid.x;
    let num_pairs = pair_count_in[0];
    if pi >= num_pairs {
        return;
    }

    let pair = pairs[pi];
    let a = pair.a;
    let b = pair.b;

    let pos_a = bodies[a].position_inv_mass.xyz;
    let pos_b = bodies[b].position_inv_mass.xyz;
    let rot_a = bodies[a].orientation;
    let rot_b = bodies[b].orientation;
    let st_a = props[a].shape_type;
    let st_b = props[b].shape_type;

    let max_contacts = params.num_bodies * 8u;

    if st_a == SHAPE_SPHERE && st_b == SHAPE_SPHERE {
        let ra = spheres[props[a].shape_index].radius;
        let rb = spheres[props[b].shape_index].radius;
        sphere_sphere_test(pos_a, ra, pos_b, rb, a, b, max_contacts);
    } else if st_a == SHAPE_SPHERE && st_b == SHAPE_BOX {
        let ra = spheres[props[a].shape_index].radius;
        let hb = boxes[props[b].shape_index].half_extents.xyz;
        sphere_box_test(pos_a, ra, pos_b, rot_b, hb, a, b, max_contacts);
    } else if st_a == SHAPE_BOX && st_b == SHAPE_SPHERE {
        let rb = spheres[props[b].shape_index].radius;
        let ha = boxes[props[a].shape_index].half_extents.xyz;
        sphere_box_test(pos_b, rb, pos_a, rot_a, ha, b, a, max_contacts);
    } else if st_a == SHAPE_BOX && st_b == SHAPE_BOX {
        let ha = boxes[props[a].shape_index].half_extents.xyz;
        let hb = boxes[props[b].shape_index].half_extents.xyz;
        box_box_test(pos_a, rot_a, ha, pos_b, rot_b, hb, a, b, max_contacts);
    }
    // Other combinations (capsule, convex hull) are not yet implemented on GPU.
}
"#;
