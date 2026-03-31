/// WGSL source for GPU narrowphase contact generation (3D).
///
/// Handles all collision pairs: sphere, box, capsule, convex hull, plane.
/// Full 15-axis SAT for box-box with edge-edge axes.
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

struct CapsuleDataGpu {
    half_height: f32,
    radius: f32,
    _pad0: f32,
    _pad1: f32,
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
const SHAPE_CAPSULE:     u32 = 2u;
const SHAPE_CONVEX_HULL: u32 = 3u;
const SHAPE_PLANE:       u32 = 4u;

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
@group(0) @binding(11) var<storage, read>      capsules:      array<CapsuleDataGpu>;
@group(0) @binding(12) var<storage, read>      plane_data:    array<vec4<f32>>;

// ---------- Helpers ----------

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

// ---------- Closest point on segment ----------

fn closest_point_on_segment(a: vec3<f32>, b: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let len2 = dot(ab, ab);
    if len2 < 1e-12 {
        return a;
    }
    let t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    return a + ab * t;
}

// Closest points between two segments. Returns (pt_on_seg1, pt_on_seg2).
fn closest_points_segments(p1: vec3<f32>, q1: vec3<f32>, p2: vec3<f32>, q2: vec3<f32>) -> array<vec3<f32>, 2> {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    if a < 1e-12 && e < 1e-12 {
        return array<vec3<f32>, 2>(p1, p2);
    }

    var s: f32;
    var t: f32;

    if a < 1e-12 {
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if e < 1e-12 {
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            if abs(denom) > 1e-12 {
                s = clamp((b * f - c * e) / denom, 0.0, 1.0);
            } else {
                s = 0.0;
            }
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }

    return array<vec3<f32>, 2>(p1 + d1 * s, p2 + d2 * t);
}

// ---------- Sphere-Sphere ----------

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
    let depth = dist - sum_r;
    let point = pos_a + normal * (radius_a + depth * 0.5);
    emit_contact(point, normal, depth, body_a, body_b, max_contacts);
}

// ---------- Sphere-Box ----------

fn sphere_box_test(
    sphere_pos: vec3<f32>, radius: f32,
    box_pos: vec3<f32>, box_rot: vec4<f32>, half_ext: vec3<f32>,
    body_sphere: u32, body_box: u32,
    max_contacts: u32,
) {
    let inv_rot = vec4<f32>(-box_rot.xyz, box_rot.w);
    let local_center = quat_rotate(inv_rot, sphere_pos - box_pos);
    let closest = clamp(local_center, -half_ext, half_ext);
    let diff = local_center - closest;
    let dist2 = dot(diff, diff);

    if dist2 >= radius * radius {
        return;
    }

    var normal_local: vec3<f32>;
    var depth: f32;

    if dist2 > 1e-12 {
        let dist = sqrt(dist2);
        normal_local = diff / dist;
        depth = dist - radius;
    } else {
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

    let normal_world = -quat_rotate(box_rot, normal_local);
    let contact_point = sphere_pos + normal_world * (radius + depth * 0.5);
    emit_contact(contact_point, normal_world, depth, body_sphere, body_box, max_contacts);
}

// ---------- Box-Box (full 15-axis SAT) ----------

fn box_box_test(
    pos_a: vec3<f32>, rot_a: vec4<f32>, he_a: vec3<f32>,
    pos_b: vec3<f32>, rot_b: vec4<f32>, he_b: vec3<f32>,
    body_a: u32, body_b: u32,
    max_contacts: u32,
) {
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

    // 6 face axes (3 from A, 3 from B)
    for (var i = 0u; i < 3u; i = i + 1u) {
        let axis = axes_a[i];
        let proj_a = he_arr_a[i];
        let proj_b = abs(dot(axes_b[0], axis)) * he_arr_b[0]
                   + abs(dot(axes_b[1], axis)) * he_arr_b[1]
                   + abs(dot(axes_b[2], axis)) * he_arr_b[2];
        let center_proj = dot(d, axis);
        let overlap = proj_a + proj_b - abs(center_proj);
        if overlap < 0.0 { return; }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            best_normal = axis * select(1.0, -1.0, center_proj < 0.0);
        }
    }
    for (var i = 0u; i < 3u; i = i + 1u) {
        let axis = axes_b[i];
        let proj_a = abs(dot(axes_a[0], axis)) * he_arr_a[0]
                   + abs(dot(axes_a[1], axis)) * he_arr_a[1]
                   + abs(dot(axes_a[2], axis)) * he_arr_a[2];
        let proj_b = he_arr_b[i];
        let center_proj = dot(d, axis);
        let overlap = proj_a + proj_b - abs(center_proj);
        if overlap < 0.0 { return; }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            best_normal = axis * select(1.0, -1.0, center_proj < 0.0);
        }
    }

    // 9 edge-edge cross product axes
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            var axis = cross(axes_a[i], axes_b[j]);
            let len2 = dot(axis, axis);
            if len2 < 1e-8 { continue; }
            axis = axis / sqrt(len2);
            let proj_a = abs(dot(axes_a[0], axis)) * he_arr_a[0]
                       + abs(dot(axes_a[1], axis)) * he_arr_a[1]
                       + abs(dot(axes_a[2], axis)) * he_arr_a[2];
            let proj_b = abs(dot(axes_b[0], axis)) * he_arr_b[0]
                       + abs(dot(axes_b[1], axis)) * he_arr_b[1]
                       + abs(dot(axes_b[2], axis)) * he_arr_b[2];
            let center_proj = dot(d, axis);
            let overlap = proj_a + proj_b - abs(center_proj);
            if overlap < 0.0 { return; }
            let depth = -overlap;
            if depth > min_depth {
                min_depth = depth;
                best_normal = axis * select(1.0, -1.0, center_proj < 0.0);
            }
        }
    }

    let contact_point = (pos_a + pos_b) * 0.5 + best_normal * min_depth * 0.5;
    emit_contact(contact_point, best_normal, min_depth, body_a, body_b, max_contacts);
}

// ---------- Sphere-Capsule ----------

fn sphere_capsule_test(
    sphere_pos: vec3<f32>, sphere_r: f32,
    cap_pos: vec3<f32>, cap_rot: vec4<f32>, cap_hh: f32, cap_r: f32,
    body_sphere: u32, body_capsule: u32,
    max_contacts: u32,
) {
    let axis = quat_rotate(cap_rot, vec3<f32>(0.0, 1.0, 0.0));
    let seg_a = cap_pos - axis * cap_hh;
    let seg_b = cap_pos + axis * cap_hh;
    let closest = closest_point_on_segment(seg_a, seg_b, sphere_pos);
    sphere_sphere_test(sphere_pos, sphere_r, closest, cap_r, body_sphere, body_capsule, max_contacts);
}

// ---------- Box-Capsule ----------

fn box_capsule_test(
    box_pos: vec3<f32>, box_rot: vec4<f32>, half_ext: vec3<f32>,
    cap_pos: vec3<f32>, cap_rot: vec4<f32>, cap_hh: f32, cap_r: f32,
    body_box: u32, body_capsule: u32,
    max_contacts: u32,
) {
    // Approximate: test capsule endpoints and midpoint as spheres
    let axis = quat_rotate(cap_rot, vec3<f32>(0.0, 1.0, 0.0));
    let seg_a = cap_pos - axis * cap_hh;
    let seg_b = cap_pos + axis * cap_hh;

    // Find closest point on capsule segment to box center
    let closest = closest_point_on_segment(seg_a, seg_b, box_pos);
    sphere_box_test(closest, cap_r, box_pos, box_rot, half_ext, body_capsule, body_box, max_contacts);
}

// ---------- Capsule-Capsule ----------

fn capsule_capsule_test(
    pos_a: vec3<f32>, rot_a: vec4<f32>, hh_a: f32, r_a: f32,
    pos_b: vec3<f32>, rot_b: vec4<f32>, hh_b: f32, r_b: f32,
    body_a: u32, body_b: u32,
    max_contacts: u32,
) {
    let axis_a = quat_rotate(rot_a, vec3<f32>(0.0, 1.0, 0.0));
    let axis_b = quat_rotate(rot_b, vec3<f32>(0.0, 1.0, 0.0));
    let a1 = pos_a - axis_a * hh_a;
    let a2 = pos_a + axis_a * hh_a;
    let b1 = pos_b - axis_b * hh_b;
    let b2 = pos_b + axis_b * hh_b;
    let pts = closest_points_segments(a1, a2, b1, b2);
    sphere_sphere_test(pts[0], r_a, pts[1], r_b, body_a, body_b, max_contacts);
}

// ---------- Capsule-Hull ----------

fn capsule_hull_test(
    cap_pos: vec3<f32>, cap_rot: vec4<f32>, cap_hh: f32, cap_r: f32,
    hull_pos: vec3<f32>, hull_rot: vec4<f32>, hull_si: u32,
    body_capsule: u32, body_hull: u32,
    max_contacts: u32,
) {
    // Approximate: test closest point on segment to hull center as a sphere
    let axis = quat_rotate(cap_rot, vec3<f32>(0.0, 1.0, 0.0));
    let seg_a = cap_pos - axis * cap_hh;
    let seg_b = cap_pos + axis * cap_hh;
    let closest = closest_point_on_segment(seg_a, seg_b, hull_pos);
    sphere_hull_test(closest, cap_r, hull_pos, hull_rot, hull_si, body_capsule, body_hull, max_contacts);
}

// ---------- Plane contacts ----------

fn plane_sphere_test(
    plane_normal: vec3<f32>, plane_dist: f32,
    sphere_pos: vec3<f32>, sphere_r: f32,
    body_plane: u32, body_sphere: u32,
    max_contacts: u32,
) {
    let d = dot(plane_normal, sphere_pos) - plane_dist;
    if d >= sphere_r {
        return;
    }
    let depth = -(sphere_r - d);
    let point = sphere_pos - plane_normal * d;
    emit_contact(point, plane_normal, depth, body_plane, body_sphere, max_contacts);
}

fn plane_box_test(
    plane_normal: vec3<f32>, plane_dist: f32,
    box_pos: vec3<f32>, box_rot: vec4<f32>, half_ext: vec3<f32>,
    body_plane: u32, body_box: u32,
    max_contacts: u32,
) {
    // Test 8 box vertices against plane
    let signs = array<vec3<f32>, 8>(
        vec3<f32>(-1.0, -1.0, -1.0),
        vec3<f32>(-1.0, -1.0,  1.0),
        vec3<f32>(-1.0,  1.0, -1.0),
        vec3<f32>(-1.0,  1.0,  1.0),
        vec3<f32>( 1.0, -1.0, -1.0),
        vec3<f32>( 1.0, -1.0,  1.0),
        vec3<f32>( 1.0,  1.0, -1.0),
        vec3<f32>( 1.0,  1.0,  1.0),
    );
    for (var i = 0u; i < 8u; i = i + 1u) {
        let local_v = signs[i] * half_ext;
        let world_v = box_pos + quat_rotate(box_rot, local_v);
        let d = dot(plane_normal, world_v) - plane_dist;
        if d < 0.0 {
            let point = world_v - plane_normal * d;
            emit_contact(point, plane_normal, d, body_plane, body_box, max_contacts);
        }
    }
}

fn plane_capsule_test(
    plane_normal: vec3<f32>, plane_dist: f32,
    cap_pos: vec3<f32>, cap_rot: vec4<f32>, cap_hh: f32, cap_r: f32,
    body_plane: u32, body_capsule: u32,
    max_contacts: u32,
) {
    let axis = quat_rotate(cap_rot, vec3<f32>(0.0, 1.0, 0.0));
    let seg_a = cap_pos - axis * cap_hh;
    let seg_b = cap_pos + axis * cap_hh;
    // Test both endpoints as spheres
    plane_sphere_test(plane_normal, plane_dist, seg_a, cap_r, body_plane, body_capsule, max_contacts);
    plane_sphere_test(plane_normal, plane_dist, seg_b, cap_r, body_plane, body_capsule, max_contacts);
}

fn plane_hull_test(
    plane_normal: vec3<f32>, plane_dist: f32,
    hull_pos: vec3<f32>, hull_rot: vec4<f32>, hull_si: u32,
    body_plane: u32, body_hull: u32,
    max_contacts: u32,
) {
    let hull = convex_hulls[hull_si];
    for (var i = 0u; i < hull.vertex_count; i = i + 1u) {
        let cv = convex_verts[hull.vertex_offset + i];
        let local_v = vec3<f32>(cv.x, cv.y, cv.z);
        let world_v = hull_pos + quat_rotate(hull_rot, local_v);
        let d = dot(plane_normal, world_v) - plane_dist;
        if d < 0.0 {
            let point = world_v - plane_normal * d;
            emit_contact(point, plane_normal, d, body_plane, body_hull, max_contacts);
        }
    }
}

// ---------- Convex hull helpers ----------

fn hull_world_vert(hull_si: u32, vi: u32, pos: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
    let hull = convex_hulls[hull_si];
    let cv = convex_verts[hull.vertex_offset + vi];
    return pos + quat_rotate(rot, vec3<f32>(cv.x, cv.y, cv.z));
}

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

// ---------- Hull-Hull SAT ----------

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

    let na = hull_a.vertex_count;
    for (var i = 0u; i < na; i = i + 1u) {
        let v0 = hull_world_vert(si_a, i, pos_a, rot_a);
        let v1 = hull_world_vert(si_a, (i + 1u) % na, pos_a, rot_a);
        let v2 = hull_world_vert(si_a, (i + 2u) % na, pos_a, rot_a);
        var axis = cross(v1 - v0, v2 - v0);
        let len2 = dot(axis, axis);
        if len2 < 1e-12 { continue; }
        axis = axis / sqrt(len2);
        if dot(axis, d) < 0.0 { axis = -axis; }
        let proj_a = hull_project(si_a, pos_a, rot_a, axis);
        let proj_b = hull_project(si_b, pos_b, rot_b, axis);
        let overlap = min(proj_a.y, proj_b.y) - max(proj_a.x, proj_b.x);
        if overlap < 0.0 { return; }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            best_normal = axis;
        }
    }

    let nb = hull_b.vertex_count;
    for (var i = 0u; i < nb; i = i + 1u) {
        let v0 = hull_world_vert(si_b, i, pos_b, rot_b);
        let v1 = hull_world_vert(si_b, (i + 1u) % nb, pos_b, rot_b);
        let v2 = hull_world_vert(si_b, (i + 2u) % nb, pos_b, rot_b);
        var axis = cross(v1 - v0, v2 - v0);
        let len2 = dot(axis, axis);
        if len2 < 1e-12 { continue; }
        axis = axis / sqrt(len2);
        if dot(axis, d) < 0.0 { axis = -axis; }
        let proj_a = hull_project(si_a, pos_a, rot_a, axis);
        let proj_b = hull_project(si_b, pos_b, rot_b, axis);
        let overlap = min(proj_a.y, proj_b.y) - max(proj_a.x, proj_b.x);
        if overlap < 0.0 { return; }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            best_normal = axis;
        }
    }

    let contact_point = (pos_a + pos_b) * 0.5 + best_normal * min_depth * 0.5;
    emit_contact(contact_point, best_normal, min_depth, body_a, body_b, max_contacts);
}

// ---------- Sphere-Hull ----------

fn sphere_hull_test(
    sphere_pos: vec3<f32>, radius: f32,
    hull_pos: vec3<f32>, hull_rot: vec4<f32>, hull_si: u32,
    body_sphere: u32, body_hull: u32,
    max_contacts: u32,
) {
    let hull = convex_hulls[hull_si];
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

    let dv = closest_vert - sphere_pos;
    let dist = sqrt(dot(dv, dv));
    if dist < 1e-12 { return; }
    let axis = dv / dist;

    let sp_center = dot(sphere_pos, axis);
    let sp_min = sp_center - radius;
    let sp_max = sp_center + radius;
    let hp = hull_project(hull_si, hull_pos, hull_rot, axis);
    let overlap = min(sp_max, hp.y) - max(sp_min, hp.x);
    if overlap < 0.0 { return; }

    let n = hull.vertex_count;
    for (var i = 0u; i < n; i = i + 1u) {
        let v0 = hull_world_vert(hull_si, i, hull_pos, hull_rot);
        let v1 = hull_world_vert(hull_si, (i + 1u) % n, hull_pos, hull_rot);
        let v2 = hull_world_vert(hull_si, (i + 2u) % n, hull_pos, hull_rot);
        var face_axis = cross(v1 - v0, v2 - v0);
        let len2 = dot(face_axis, face_axis);
        if len2 < 1e-12 { continue; }
        face_axis = face_axis / sqrt(len2);
        let spc = dot(sphere_pos, face_axis);
        let s_min = spc - radius;
        let s_max = spc + radius;
        let h_proj = hull_project(hull_si, hull_pos, hull_rot, face_axis);
        let face_overlap = min(s_max, h_proj.y) - max(s_min, h_proj.x);
        if face_overlap < 0.0 { return; }
    }

    let depth = -overlap;
    let normal = select(axis, -axis, dot(axis, sphere_pos - hull_pos) > 0.0);
    let contact_point = sphere_pos + normal * (radius + depth * 0.5);
    emit_contact(contact_point, normal, depth, body_sphere, body_hull, max_contacts);
}

// ---------- Box-Hull ----------

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

    for (var i = 0u; i < 3u; i = i + 1u) {
        let axis = box_axes[i];
        let proj_box = he_arr[i];
        let hp = hull_project(hull_si, hull_pos, hull_rot, axis);
        let box_center = dot(box_pos, axis);
        let box_min = box_center - proj_box;
        let box_max = box_center + proj_box;
        let overlap = min(box_max, hp.y) - max(box_min, hp.x);
        if overlap < 0.0 { return; }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            let center_proj = dot(d, axis);
            best_normal = axis * select(1.0, -1.0, center_proj < 0.0);
        }
    }

    let hull = convex_hulls[hull_si];
    let n = hull.vertex_count;
    for (var i = 0u; i < n; i = i + 1u) {
        let v0 = hull_world_vert(hull_si, i, hull_pos, hull_rot);
        let v1 = hull_world_vert(hull_si, (i + 1u) % n, hull_pos, hull_rot);
        let v2 = hull_world_vert(hull_si, (i + 2u) % n, hull_pos, hull_rot);
        var axis = cross(v1 - v0, v2 - v0);
        let len2 = dot(axis, axis);
        if len2 < 1e-12 { continue; }
        axis = axis / sqrt(len2);
        if dot(axis, d) < 0.0 { axis = -axis; }
        let proj_box = abs(dot(box_axes[0], axis)) * he_arr[0]
                     + abs(dot(box_axes[1], axis)) * he_arr[1]
                     + abs(dot(box_axes[2], axis)) * he_arr[2];
        let box_center = dot(box_pos, axis);
        let box_min = box_center - proj_box;
        let box_max = box_center + proj_box;
        let hp = hull_project(hull_si, hull_pos, hull_rot, axis);
        let overlap = min(box_max, hp.y) - max(box_min, hp.x);
        if overlap < 0.0 { return; }
        let depth = -overlap;
        if depth > min_depth {
            min_depth = depth;
            best_normal = axis;
        }
    }

    let contact_point = (box_pos + hull_pos) * 0.5 + best_normal * min_depth * 0.5;
    emit_contact(contact_point, best_normal, min_depth, body_box, body_hull, max_contacts);
}

// ---------- Main dispatch ----------

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
    let si_a = props[a].shape_index;
    let si_b = props[b].shape_index;

    let max_contacts = params.num_bodies * 8u;

    // Order: ensure st_a <= st_b for consistent dispatch
    var s1 = st_a; var s2 = st_b;
    var p1 = pos_a; var p2 = pos_b;
    var r1 = rot_a; var r2 = rot_b;
    var i1 = si_a; var i2 = si_b;
    var b1 = a; var b2 = b;
    if s1 > s2 {
        s1 = st_b; s2 = st_a;
        p1 = pos_b; p2 = pos_a;
        r1 = rot_b; r2 = rot_a;
        i1 = si_b; i2 = si_a;
        b1 = b; b2 = a;
    }

    // Dispatch based on sorted shape pair
    if s1 == SHAPE_SPHERE && s2 == SHAPE_SPHERE {
        let ra = spheres[i1].radius;
        let rb = spheres[i2].radius;
        sphere_sphere_test(p1, ra, p2, rb, b1, b2, max_contacts);
    } else if s1 == SHAPE_SPHERE && s2 == SHAPE_BOX {
        let ra = spheres[i1].radius;
        let hb = boxes[i2].half_extents.xyz;
        sphere_box_test(p1, ra, p2, r2, hb, b1, b2, max_contacts);
    } else if s1 == SHAPE_SPHERE && s2 == SHAPE_CAPSULE {
        let ra = spheres[i1].radius;
        let cap = capsules[i2];
        sphere_capsule_test(p1, ra, p2, r2, cap.half_height, cap.radius, b1, b2, max_contacts);
    } else if s1 == SHAPE_SPHERE && s2 == SHAPE_CONVEX_HULL {
        let ra = spheres[i1].radius;
        sphere_hull_test(p1, ra, p2, r2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_SPHERE && s2 == SHAPE_PLANE {
        let ra = spheres[i1].radius;
        let pd = plane_data[i2];
        plane_sphere_test(pd.xyz, pd.w, p1, ra, b2, b1, max_contacts);
    } else if s1 == SHAPE_BOX && s2 == SHAPE_BOX {
        let ha = boxes[i1].half_extents.xyz;
        let hb = boxes[i2].half_extents.xyz;
        box_box_test(p1, r1, ha, p2, r2, hb, b1, b2, max_contacts);
    } else if s1 == SHAPE_BOX && s2 == SHAPE_CAPSULE {
        let ha = boxes[i1].half_extents.xyz;
        let cap = capsules[i2];
        box_capsule_test(p1, r1, ha, p2, r2, cap.half_height, cap.radius, b1, b2, max_contacts);
    } else if s1 == SHAPE_BOX && s2 == SHAPE_CONVEX_HULL {
        let ha = boxes[i1].half_extents.xyz;
        box_hull_test(p1, r1, ha, p2, r2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_BOX && s2 == SHAPE_PLANE {
        let ha = boxes[i1].half_extents.xyz;
        let pd = plane_data[i2];
        plane_box_test(pd.xyz, pd.w, p1, r1, ha, b2, b1, max_contacts);
    } else if s1 == SHAPE_CAPSULE && s2 == SHAPE_CAPSULE {
        let ca = capsules[i1];
        let cb = capsules[i2];
        capsule_capsule_test(p1, r1, ca.half_height, ca.radius, p2, r2, cb.half_height, cb.radius, b1, b2, max_contacts);
    } else if s1 == SHAPE_CAPSULE && s2 == SHAPE_CONVEX_HULL {
        let cap = capsules[i1];
        capsule_hull_test(p1, r1, cap.half_height, cap.radius, p2, r2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_CAPSULE && s2 == SHAPE_PLANE {
        let cap = capsules[i1];
        let pd = plane_data[i2];
        plane_capsule_test(pd.xyz, pd.w, p1, r1, cap.half_height, cap.radius, b2, b1, max_contacts);
    } else if s1 == SHAPE_CONVEX_HULL && s2 == SHAPE_CONVEX_HULL {
        hull_hull_test(p1, r1, i1, p2, r2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_CONVEX_HULL && s2 == SHAPE_PLANE {
        let pd = plane_data[i2];
        plane_hull_test(pd.xyz, pd.w, p1, r1, i1, b2, b1, max_contacts);
    }
    // SHAPE_PLANE vs SHAPE_PLANE: no collision
}
"#;
