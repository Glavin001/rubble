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

const SHAPE_SPHERE:      u32 = 0u;
const SHAPE_BOX:         u32 = 1u;
const SHAPE_CONVEX_HULL: u32 = 3u;

struct ConvexHullInfo {
    vertex_offset: u32,
    vertex_count:  u32,
    face_offset:   u32,
    face_count:    u32,
    edge_offset:   u32,
    edge_count:    u32,
    gauss_map_offset: u32,
    gauss_map_count:  u32,
};

struct ConvexVert {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read>       bodies:        array<Body>;
@group(0) @binding(1) var<storage, read>       props:         array<BodyProps>;
@group(0) @binding(2) var<storage, read>       pairs:         array<Pair>;
@group(0) @binding(3) var<storage, read>       pair_count_in: array<u32>;
@group(0) @binding(4) var<storage, read>       spheres:       array<SphereData>;
@group(0) @binding(5) var<storage, read>       boxes:         array<BoxDataGpu>;
@group(0) @binding(6) var<storage, read_write> contacts:      array<Contact>;
@group(0) @binding(7) var<storage, read_write> contact_count: atomic<u32>;
@group(0) @binding(8) var<uniform>             params:        SimParams;
@group(0) @binding(9) var<storage, read>       convex_hulls:  array<ConvexHullInfo>;
@group(0) @binding(10) var<storage, read>      convex_verts:  array<ConvexVert>;

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

// Get a world-space vertex from a convex hull
fn hull_world_vert(hull_si: u32, vi: u32, pos: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
    let hull = convex_hulls[hull_si];
    let cv = convex_verts[hull.vertex_offset + vi];
    return pos + quat_rotate(rot, vec3<f32>(cv.x, cv.y, cv.z));
}

// Project a convex hull onto an axis and return (min, max)
fn hull_project(hull_si: u32, pos: vec3<f32>, rot: vec4<f32>, axis: vec3<f32>) -> vec2<f32> {
    let hull = convex_hulls[hull_si];
    var mn = 1e30;
    var mx = -1e30;
    for (var i = 0u; i < hull.vertex_count; i = i + 1u) {
        let wv = hull_world_vert(hull_si, i, pos, rot);
        let p = dot(wv, axis);
        mn = min(mn, p);
        mx = max(mx, p);
    }
    return vec2<f32>(mn, mx);
}

// SAT test between two convex hulls using face normals derived from edges.
// Returns true if contact was emitted.
fn hull_hull_test(
    pos_a: vec3<f32>, rot_a: vec4<f32>, si_a: u32,
    pos_b: vec3<f32>, rot_b: vec4<f32>, si_b: u32,
    body_a: u32, body_b: u32,
    max_contacts: u32,
) {
    let hull_a = convex_hulls[si_a];
    let hull_b = convex_hulls[si_b];
    let d = pos_b - pos_a;

    var min_depth = -1e30;
    var best_normal = vec3<f32>(0.0, 1.0, 0.0);

    // Test edge normals from hull A
    let na = hull_a.vertex_count;
    for (var i = 0u; i < na; i = i + 1u) {
        let v0 = hull_world_vert(si_a, i, pos_a, rot_a);
        let v1 = hull_world_vert(si_a, (i + 1u) % na, pos_a, rot_a);
        let v2 = hull_world_vert(si_a, (i + 2u) % na, pos_a, rot_a);
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        var axis = cross(edge1, edge2);
        let len2 = dot(axis, axis);
        if len2 < 1e-12 {
            continue;
        }
        axis = axis / sqrt(len2);
        // Ensure axis points from A toward B
        if dot(axis, d) < 0.0 {
            axis = -axis;
        }
        let proj_a = hull_project(si_a, pos_a, rot_a, axis);
        let proj_b = hull_project(si_b, pos_b, rot_b, axis);
        let overlap = min(proj_a.y, proj_b.y) - max(proj_a.x, proj_b.x);
        if overlap < 0.0 {
            return; // separated
        }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            best_normal = axis;
        }
    }

    // Test edge normals from hull B
    let nb = hull_b.vertex_count;
    for (var i = 0u; i < nb; i = i + 1u) {
        let v0 = hull_world_vert(si_b, i, pos_b, rot_b);
        let v1 = hull_world_vert(si_b, (i + 1u) % nb, pos_b, rot_b);
        let v2 = hull_world_vert(si_b, (i + 2u) % nb, pos_b, rot_b);
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        var axis = cross(edge1, edge2);
        let len2 = dot(axis, axis);
        if len2 < 1e-12 {
            continue;
        }
        axis = axis / sqrt(len2);
        if dot(axis, d) < 0.0 {
            axis = -axis;
        }
        let proj_a = hull_project(si_a, pos_a, rot_a, axis);
        let proj_b = hull_project(si_b, pos_b, rot_b, axis);
        let overlap = min(proj_a.y, proj_b.y) - max(proj_a.x, proj_b.x);
        if overlap < 0.0 {
            return;
        }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            best_normal = axis;
        }
    }

    let contact_point = (pos_a + pos_b) * 0.5 + best_normal * min_depth * 0.5;
    emit_contact(contact_point, best_normal, min_depth, body_a, body_b, max_contacts);
}

// SAT test: sphere vs convex hull
fn sphere_hull_test(
    sphere_pos: vec3<f32>, radius: f32,
    hull_pos: vec3<f32>, hull_rot: vec4<f32>, hull_si: u32,
    body_sphere: u32, body_hull: u32,
    max_contacts: u32,
) {
    let hull = convex_hulls[hull_si];
    // Find closest vertex to sphere center
    var closest_dist2 = 1e30;
    var closest_vert = hull_pos;
    for (var i = 0u; i < hull.vertex_count; i = i + 1u) {
        let wv = hull_world_vert(hull_si, i, hull_pos, hull_rot);
        let d2 = dot(wv - sphere_pos, wv - sphere_pos);
        if d2 < closest_dist2 {
            closest_dist2 = d2;
            closest_vert = wv;
        }
    }

    // Test axis from sphere center to closest vertex
    let d = closest_vert - sphere_pos;
    let dist = sqrt(dot(d, d));
    if dist < 1e-12 {
        return;
    }
    let axis = d / dist;

    // Project sphere
    let sp_center = dot(sphere_pos, axis);
    let sp_min = sp_center - radius;
    let sp_max = sp_center + radius;

    // Project hull
    let hp = hull_project(hull_si, hull_pos, hull_rot, axis);

    let overlap = min(sp_max, hp.y) - max(sp_min, hp.x);
    if overlap < 0.0 {
        return;
    }

    // Also test face normals from hull
    let n = hull.vertex_count;
    for (var i = 0u; i < n; i = i + 1u) {
        let v0 = hull_world_vert(hull_si, i, hull_pos, hull_rot);
        let v1 = hull_world_vert(hull_si, (i + 1u) % n, hull_pos, hull_rot);
        let v2 = hull_world_vert(hull_si, (i + 2u) % n, hull_pos, hull_rot);
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        var face_axis = cross(e1, e2);
        let len2 = dot(face_axis, face_axis);
        if len2 < 1e-12 {
            continue;
        }
        face_axis = face_axis / sqrt(len2);
        let spc = dot(sphere_pos, face_axis);
        let s_min = spc - radius;
        let s_max = spc + radius;
        let h_proj = hull_project(hull_si, hull_pos, hull_rot, face_axis);
        let face_overlap = min(s_max, h_proj.y) - max(s_min, h_proj.x);
        if face_overlap < 0.0 {
            return;
        }
    }

    let depth = -overlap;
    let normal = select(axis, -axis, dot(axis, sphere_pos - hull_pos) > 0.0);
    let contact_point = sphere_pos + normal * (radius + depth * 0.5);
    emit_contact(contact_point, normal, depth, body_sphere, body_hull, max_contacts);
}

// SAT test: box vs convex hull
// Treats the box as having 3 face-normal axes and tests hull edge normals too.
fn box_hull_test(
    box_pos: vec3<f32>, box_rot: vec4<f32>, half_ext: vec3<f32>,
    hull_pos: vec3<f32>, hull_rot: vec4<f32>, hull_si: u32,
    body_box: u32, body_hull: u32,
    max_contacts: u32,
) {
    let box_axes = array<vec3<f32>, 3>(
        quat_rotate(box_rot, vec3<f32>(1.0, 0.0, 0.0)),
        quat_rotate(box_rot, vec3<f32>(0.0, 1.0, 0.0)),
        quat_rotate(box_rot, vec3<f32>(0.0, 0.0, 1.0)),
    );
    let he_arr = array<f32, 3>(half_ext.x, half_ext.y, half_ext.z);
    let d = hull_pos - box_pos;

    var min_depth = -1e30;
    var best_normal = vec3<f32>(0.0, 1.0, 0.0);

    // Test 3 box face axes
    for (var i = 0u; i < 3u; i = i + 1u) {
        let axis = box_axes[i];
        // Project box: half-extent along this axis
        let proj_box = he_arr[i];
        // Project hull
        let hp = hull_project(hull_si, hull_pos, hull_rot, axis);
        let box_center = dot(box_pos, axis);
        let box_min = box_center - proj_box;
        let box_max = box_center + proj_box;
        let overlap = min(box_max, hp.y) - max(box_min, hp.x);
        if overlap < 0.0 {
            return;
        }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            let center_proj = dot(d, axis);
            best_normal = axis * select(1.0, -1.0, center_proj < 0.0);
        }
    }

    // Test hull face normals
    let hull = convex_hulls[hull_si];
    let n = hull.vertex_count;
    for (var i = 0u; i < n; i = i + 1u) {
        let v0 = hull_world_vert(hull_si, i, hull_pos, hull_rot);
        let v1 = hull_world_vert(hull_si, (i + 1u) % n, hull_pos, hull_rot);
        let v2 = hull_world_vert(hull_si, (i + 2u) % n, hull_pos, hull_rot);
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        var axis = cross(e1, e2);
        let len2 = dot(axis, axis);
        if len2 < 1e-12 {
            continue;
        }
        axis = axis / sqrt(len2);
        if dot(axis, d) < 0.0 {
            axis = -axis;
        }
        // Project box onto this axis
        let proj_box = abs(dot(box_axes[0], axis)) * he_arr[0]
                     + abs(dot(box_axes[1], axis)) * he_arr[1]
                     + abs(dot(box_axes[2], axis)) * he_arr[2];
        let box_center = dot(box_pos, axis);
        let box_min = box_center - proj_box;
        let box_max = box_center + proj_box;
        let hp = hull_project(hull_si, hull_pos, hull_rot, axis);
        let overlap = min(box_max, hp.y) - max(box_min, hp.x);
        if overlap < 0.0 {
            return;
        }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            best_normal = axis;
        }
    }

    let contact_point = (box_pos + hull_pos) * 0.5 + best_normal * min_depth * 0.5;
    emit_contact(contact_point, best_normal, min_depth, body_box, body_hull, max_contacts);
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
    } else if st_a == SHAPE_CONVEX_HULL && st_b == SHAPE_CONVEX_HULL {
        hull_hull_test(pos_a, rot_a, props[a].shape_index, pos_b, rot_b, props[b].shape_index, a, b, max_contacts);
    } else if st_a == SHAPE_SPHERE && st_b == SHAPE_CONVEX_HULL {
        let ra = spheres[props[a].shape_index].radius;
        sphere_hull_test(pos_a, ra, pos_b, rot_b, props[b].shape_index, a, b, max_contacts);
    } else if st_a == SHAPE_CONVEX_HULL && st_b == SHAPE_SPHERE {
        let rb = spheres[props[b].shape_index].radius;
        sphere_hull_test(pos_b, rb, pos_a, rot_a, props[a].shape_index, b, a, max_contacts);
    } else if st_a == SHAPE_BOX && st_b == SHAPE_CONVEX_HULL {
        let ha = boxes[props[a].shape_index].half_extents.xyz;
        box_hull_test(pos_a, rot_a, ha, pos_b, rot_b, props[b].shape_index, a, b, max_contacts);
    } else if st_a == SHAPE_CONVEX_HULL && st_b == SHAPE_BOX {
        let hb = boxes[props[b].shape_index].half_extents.xyz;
        box_hull_test(pos_b, rot_b, hb, pos_a, rot_a, props[a].shape_index, b, a, max_contacts);
    }
}
"#;
