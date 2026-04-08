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
    inertia_row0:     vec4<f32>,
    inertia_row1:     vec4<f32>,
    inertia_row2:     vec4<f32>,
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
    point:          vec4<f32>,
    normal:         vec4<f32>,
    tangent:        vec4<f32>,
    local_anchor_a: vec4<f32>,
    local_anchor_b: vec4<f32>,
    lambda:         vec4<f32>,
    penalty:        vec4<f32>,
    body_a:         u32,
    body_b:         u32,
    feature_id:     u32,
    flags:          u32,
};

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
    quality: vec4<f32>,
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
    _pad0: u32,
    _pad1: u32,
};

struct ConvexVert {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
};

struct PlaneParams {
    num_planes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    planes: array<vec4<f32>, 16>,
};

@group(0) @binding(0) var<storage, read>       bodies:        array<Body>;
@group(0) @binding(1) var<storage, read>       props:         array<BodyProps>;
@group(0) @binding(2) var<storage, read>       pairs:         array<Pair>;
@group(0) @binding(3) var<storage, read>       spheres:       array<SphereData>;
@group(0) @binding(4) var<storage, read>       boxes:         array<BoxDataGpu>;
@group(0) @binding(5) var<storage, read_write> contacts:      array<Contact>;
@group(0) @binding(6) var<storage, read_write> contact_count: atomic<u32>;
@group(0) @binding(7) var<uniform>             params:        SimParams;
@group(0) @binding(8) var<storage, read>       convex_hulls:  array<ConvexHullInfo>;
@group(0) @binding(9) var<storage, read>       convex_verts:  array<ConvexVert>;
@group(0) @binding(10) var<storage, read>      capsules:      array<CapsuleDataGpu>;
@group(0) @binding(11) var<uniform>            plane_params:  PlaneParams;

// ---------- Helpers ----------

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn build_tangent(normal: vec3<f32>) -> vec3<f32> {
    let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.z) > 0.707);
    return normalize(cross(axis, normal));
}

fn emit_contact(
    point: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
    body_a: u32,
    body_b: u32,
    feature_id: u32,
    max_contacts: u32,
) {
    let world_a = point + normal * depth * 0.5;
    let world_b = point - normal * depth * 0.5;
    emit_contact_world_points(world_a, world_b, normal, body_a, body_b, feature_id, max_contacts);
}

fn emit_contact_world_points(
    world_a: vec3<f32>,
    world_b: vec3<f32>,
    normal: vec3<f32>,
    body_a: u32,
    body_b: u32,
    feature_id: u32,
    max_contacts: u32,
) {
    let slot = atomicAdd(&contact_count, 1u);
    if slot >= max_contacts {
        return;
    }
    let tangent = build_tangent(normal);
    let pos_a = bodies[body_a].position_inv_mass.xyz;
    let pos_b = bodies[body_b].position_inv_mass.xyz;
    let q_a = bodies[body_a].orientation;
    let q_b = bodies[body_b].orientation;
    let point = (world_a + world_b) * 0.5;
    let depth = dot(normal, world_a - world_b);
    let local_a = quat_rotate(quat_conj(q_a), world_a - pos_a);
    let local_b = quat_rotate(quat_conj(q_b), world_b - pos_b);
    let margin = params.quality.z;  // penetration_slop
    let c_n_initial = depth + margin;
    contacts[slot].point  = vec4<f32>(point, depth);
    contacts[slot].normal = vec4<f32>(normal, c_n_initial);
    contacts[slot].tangent = vec4<f32>(tangent, 0.0);
    contacts[slot].local_anchor_a = vec4<f32>(local_a, 0.0);
    contacts[slot].local_anchor_b = vec4<f32>(local_b, 0.0);
    contacts[slot].lambda = vec4<f32>(0.0);
    let k_start = params.solver.z;
    contacts[slot].penalty = vec4<f32>(k_start, k_start, k_start, 0.0);
    contacts[slot].body_a = body_a;
    contacts[slot].body_b = body_b;
    contacts[slot].feature_id = feature_id;
    contacts[slot].flags = 0u;
}

fn emit_plane_contact(
    plane_point: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
    body_dynamic: u32,
    body_plane: u32,
    feature_id: u32,
    max_contacts: u32,
) {
    let slot = atomicAdd(&contact_count, 1u);
    if slot >= max_contacts {
        return;
    }
    let tangent = build_tangent(normal);
    let pos_a = bodies[body_dynamic].position_inv_mass.xyz;
    let pos_b = bodies[body_plane].position_inv_mass.xyz;
    let q_a = bodies[body_dynamic].orientation;
    let q_b = bodies[body_plane].orientation;
    let world_a = plane_point + normal * depth;
    let world_b = plane_point;
    let local_a = quat_rotate(quat_conj(q_a), world_a - pos_a);
    let local_b = quat_rotate(quat_conj(q_b), world_b - pos_b);
    let margin = params.quality.z;  // penetration_slop
    let c_n_initial = depth + margin;
    contacts[slot].point = vec4<f32>((world_a + world_b) * 0.5, depth);
    contacts[slot].normal = vec4<f32>(normal, c_n_initial);
    contacts[slot].tangent = vec4<f32>(tangent, 0.0);
    contacts[slot].local_anchor_a = vec4<f32>(local_a, 0.0);
    contacts[slot].local_anchor_b = vec4<f32>(local_b, 0.0);
    contacts[slot].lambda = vec4<f32>(0.0);
    let k_start = params.solver.z;
    contacts[slot].penalty = vec4<f32>(k_start, k_start, k_start, 0.0);
    contacts[slot].body_a = body_dynamic;
    contacts[slot].body_b = body_plane;
    contacts[slot].feature_id = feature_id;
    contacts[slot].flags = 0u;
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
    let contact_offset = params.quality.x;
    let diff = pos_b - pos_a;
    let dist2 = dot(diff, diff);
    let sum_r = radius_a + radius_b + contact_offset;
    if dist2 >= sum_r * sum_r || dist2 < 1e-12 {
        return;
    }
    let dist = sqrt(dist2);
    let normal = -diff / dist;
    let depth = dist - (radius_a + radius_b); // depth relative to actual radii, not prediction
    let point = pos_a - normal * (radius_a + depth * 0.5);
    emit_contact(point, normal, depth, body_a, body_b, 1u, max_contacts);
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

    let normal_world = quat_rotate(box_rot, normal_local);
    let contact_point = sphere_pos - normal_world * (radius + depth * 0.5);
    emit_contact(contact_point, normal_world, depth, body_sphere, body_box, 1u, max_contacts);
}

// ---------- Sutherland-Hodgman polygon clipping ----------

// Clip polygon (up to 8 verts) against a single plane (points on positive side kept).
// Returns new vertex count.
fn clip_polygon_against_plane(
    verts: ptr<function, array<vec3<f32>, 8>>,
    count: u32,
    plane_normal: vec3<f32>,
    plane_dist: f32,
) -> u32 {
    var out: array<vec3<f32>, 8>;
    var out_count = 0u;
    if count == 0u { return 0u; }

    for (var i = 0u; i < count; i = i + 1u) {
        let j = (i + 1u) % count;
        let vi = (*verts)[i];
        let vj = (*verts)[j];
        let di = dot(plane_normal, vi) - plane_dist;
        let dj = dot(plane_normal, vj) - plane_dist;

        if di >= 0.0 {
            // Current vertex is inside
            if out_count < 8u {
                out[out_count] = vi;
                out_count = out_count + 1u;
            }
            if dj < 0.0 {
                // Next is outside, emit intersection
                let t = di / (di - dj);
                if out_count < 8u {
                    out[out_count] = vi + (vj - vi) * t;
                    out_count = out_count + 1u;
                }
            }
        } else if dj >= 0.0 {
            // Current outside, next inside, emit intersection
            let t = di / (di - dj);
            if out_count < 8u {
                out[out_count] = vi + (vj - vi) * t;
                out_count = out_count + 1u;
            }
        }
    }

    for (var i = 0u; i < out_count; i = i + 1u) {
        (*verts)[i] = out[i];
    }
    return out_count;
}

// Get 4 vertices of a box face in world space.
// face_idx 0..5: +X, -X, +Y, -Y, +Z, -Z
fn get_box_face_vertices(
    he: vec3<f32>,
    axes: array<vec3<f32>, 3>,
    pos: vec3<f32>,
    face_idx: u32,
) -> array<vec3<f32>, 4> {
    // Determine which axis and sign
    let axis_i = face_idx / 2u;
    let sign = select(1.0, -1.0, (face_idx & 1u) != 0u);

    // Two tangent axes
    let t0_i = (axis_i + 1u) % 3u;
    let t1_i = (axis_i + 2u) % 3u;

    let he_arr = array<f32, 3>(he.x, he.y, he.z);

    let center = pos + axes[axis_i] * (sign * he_arr[axis_i]);
    let u = axes[t0_i] * he_arr[t0_i];
    let v = axes[t1_i] * he_arr[t1_i];

    // Wind CCW when viewed from outside (along +normal direction)
    return array<vec3<f32>, 4>(
        center - u - v,
        center + u - v,
        center + u + v,
        center - u + v,
    );
}

// Get outward normal of a box face
fn get_box_face_normal(axes: array<vec3<f32>, 3>, face_idx: u32) -> vec3<f32> {
    let axis_i = face_idx / 2u;
    let sign = select(1.0, -1.0, (face_idx & 1u) != 0u);
    return axes[axis_i] * sign;
}

fn support_point_box(
    pos: vec3<f32>,
    axes: array<vec3<f32>, 3>,
    he: vec3<f32>,
    dir: vec3<f32>,
) -> vec3<f32> {
    let sx = select(-1.0, 1.0, dot(dir, axes[0]) >= 0.0);
    let sy = select(-1.0, 1.0, dot(dir, axes[1]) >= 0.0);
    let sz = select(-1.0, 1.0, dot(dir, axes[2]) >= 0.0);
    return pos
        + axes[0] * (he.x * sx)
        + axes[1] * (he.y * sy)
        + axes[2] * (he.z * sz);
}

// ---------- Manifold reduction (area maximisation) ----------

// Reduce up to 8 contact points to at most 4, preserving area coverage.
// Returns count of output contacts (up to 4). Results written to out_points/out_depths.
fn reduce_manifold(
    points: ptr<function, array<vec3<f32>, 8>>,
    depths: ptr<function, array<f32, 8>>,
    count: u32,
    normal: vec3<f32>,
    out_points: ptr<function, array<vec3<f32>, 4>>,
    out_depths: ptr<function, array<f32, 4>>,
) -> u32 {
    if count <= 4u {
        let c = min(count, 4u);
        for (var i = 0u; i < c; i = i + 1u) {
            (*out_points)[i] = (*points)[i];
            (*out_depths)[i] = (*depths)[i];
        }
        return c;
    }

    // 1. Pick deepest point (most negative depth = most penetrating)
    var idx0 = 0u;
    var best_d = (*depths)[0];
    for (var i = 1u; i < count; i = i + 1u) {
        if (*depths)[i] < best_d {
            best_d = (*depths)[i];
            idx0 = i;
        }
    }

    // 2. Pick point farthest from idx0
    var idx1 = 0u;
    var best_dist2 = -1.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let dd = (*points)[i] - (*points)[idx0];
        let d2 = dot(dd, dd);
        if d2 > best_dist2 {
            best_dist2 = d2;
            idx1 = i;
        }
    }

    // 3. Pick point maximizing triangle area with (idx0, idx1)
    var idx2 = 0u;
    var best_area = -1.0;
    let e01 = (*points)[idx1] - (*points)[idx0];
    for (var i = 0u; i < count; i = i + 1u) {
        if i == idx0 || i == idx1 { continue; }
        let e0i = (*points)[i] - (*points)[idx0];
        let cr = cross(e01, e0i);
        let area = abs(dot(cr, normal));
        if area > best_area {
            best_area = area;
            idx2 = i;
        }
    }

    // 4. Pick point maximizing quadrilateral area (on opposite side of triangle edge 0->1)
    var idx3 = 0u;
    var best_area2 = -1.0;
    // Signed area direction of triangle 0,1,2
    let tri_sign = dot(cross((*points)[idx1] - (*points)[idx0], (*points)[idx2] - (*points)[idx0]), normal);
    for (var i = 0u; i < count; i = i + 1u) {
        if i == idx0 || i == idx1 || i == idx2 { continue; }
        // We want the point on the opposite side of the line idx0->idx1 from idx2
        let e0i = (*points)[i] - (*points)[idx0];
        let side = dot(cross(e01, e0i), normal);
        // Opposite side means sign differs from tri_sign
        if side * tri_sign >= 0.0 { continue; }
        let area = abs(side);
        if area > best_area2 {
            best_area2 = area;
            idx3 = i;
        }
    }
    // If no point on opposite side, just pick farthest remaining
    if best_area2 < 0.0 {
        var best_fd = -1.0;
        for (var i = 0u; i < count; i = i + 1u) {
            if i == idx0 || i == idx1 || i == idx2 { continue; }
            let dd = (*points)[i] - (*points)[idx0];
            let fd = dot(dd, dd);
            if fd > best_fd {
                best_fd = fd;
                idx3 = i;
            }
        }
    }

    (*out_points)[0] = (*points)[idx0];
    (*out_points)[1] = (*points)[idx1];
    (*out_points)[2] = (*points)[idx2];
    (*out_points)[3] = (*points)[idx3];
    (*out_depths)[0] = (*depths)[idx0];
    (*out_depths)[1] = (*depths)[idx1];
    (*out_depths)[2] = (*depths)[idx2];
    (*out_depths)[3] = (*depths)[idx3];
    return 4u;
}

// ---------- Box-Box (full 15-axis SAT + Sutherland-Hodgman clipping) ----------

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
    var best_face_sep = -1e30;
    var best_face_normal = vec3<f32>(0.0, 1.0, 0.0);
    var best_face_axis_type = 0u; // 0=face_a, 1=face_b
    var best_face_axis = 0u;
    var have_face_axis = false;

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
        let separation = abs(center_proj) - (proj_a + proj_b);
        if separation > 0.0 { return; }
        if !have_face_axis || separation > best_face_sep {
            have_face_axis = true;
            best_face_sep = separation;
            best_face_normal = axis * select(1.0, -1.0, center_proj < 0.0);
            best_face_axis_type = 0u;
            best_face_axis = i;
        }
    }
    for (var i = 0u; i < 3u; i = i + 1u) {
        let axis = axes_b[i];
        let proj_a = abs(dot(axes_a[0], axis)) * he_arr_a[0]
                   + abs(dot(axes_a[1], axis)) * he_arr_a[1]
                   + abs(dot(axes_a[2], axis)) * he_arr_a[2];
        let proj_b = he_arr_b[i];
        let center_proj = dot(d, axis);
        let separation = abs(center_proj) - (proj_a + proj_b);
        if separation > 0.0 { return; }
        if !have_face_axis || separation > best_face_sep {
            have_face_axis = true;
            best_face_sep = separation;
            best_face_normal = axis * select(1.0, -1.0, center_proj < 0.0);
            best_face_axis_type = 1u;
            best_face_axis = i;
        }
    }

    // 9 edge-edge cross product axes
    var best_edge_sep = -1e30;
    var best_edge_normal = vec3<f32>(0.0, 1.0, 0.0);
    var best_edge_a = 0u;
    var best_edge_b = 0u;
    var have_edge_axis = false;
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
            let separation = abs(center_proj) - (proj_a + proj_b);
            if separation > 0.0 { return; }
            if !have_edge_axis || separation > best_edge_sep {
                have_edge_axis = true;
                best_edge_sep = separation;
                best_edge_normal = axis * select(1.0, -1.0, center_proj < 0.0);
                best_edge_a = i;
                best_edge_b = j;
            }
        }
    }

    if !have_face_axis { return; }

    let edge_rel_tol = 0.95;
    let edge_abs_tol = 0.01;
    var min_depth = best_face_sep;
    var best_normal = best_face_normal;
    var best_axis_type = best_face_axis_type; // 0=face_a, 1=face_b, 2=edge
    var edge_is_distinct = have_edge_axis;
    if edge_is_distinct {
        for (var i = 0u; i < 3u; i = i + 1u) {
            if abs(dot(best_edge_normal, axes_a[i])) > 0.98 || abs(dot(best_edge_normal, axes_b[i])) > 0.98 {
                edge_is_distinct = false;
            }
        }
    }
    if edge_is_distinct && edge_rel_tol * best_edge_sep > best_face_sep + edge_abs_tol {
        min_depth = best_edge_sep;
        best_normal = best_edge_normal;
        best_axis_type = 2u;
    }

    // Ensure normal points from A to B
    if dot(best_normal, d) < 0.0 {
        best_normal = -best_normal;
    }
    let contact_normal = -best_normal;

    // ----- Edge-edge: single contact point -----
    if best_axis_type == 2u {
        // Find closest points between the two edges
        let sign_a = select(1.0, -1.0, dot(axes_a[best_edge_a], best_normal) < 0.0);
        let sign_b = select(1.0, -1.0, dot(axes_b[best_edge_b], best_normal) > 0.0);
        // Edge center on each box face
        let t0a = (best_edge_a + 1u) % 3u;
        let t1a = (best_edge_a + 2u) % 3u;
        let t0b = (best_edge_b + 1u) % 3u;
        let t1b = (best_edge_b + 2u) % 3u;
        let he_aa = array<f32, 3>(he_a.x, he_a.y, he_a.z);
        let he_bb = array<f32, 3>(he_b.x, he_b.y, he_b.z);
        // Midpoint of edge on A: project onto the two non-edge axes to get edge center
        let edge_mid_a = pos_a
            + axes_a[t0a] * (he_aa[t0a] * select(1.0, -1.0, dot(axes_a[t0a], best_normal) < 0.0))
            + axes_a[t1a] * (he_aa[t1a] * select(1.0, -1.0, dot(axes_a[t1a], best_normal) < 0.0));
        let edge_mid_b = pos_b
            + axes_b[t0b] * (he_bb[t0b] * select(1.0, -1.0, dot(axes_b[t0b], best_normal) > 0.0))
            + axes_b[t1b] * (he_bb[t1b] * select(1.0, -1.0, dot(axes_b[t1b], best_normal) > 0.0));
        let edge_a_start = edge_mid_a - axes_a[best_edge_a] * he_aa[best_edge_a];
        let edge_a_end   = edge_mid_a + axes_a[best_edge_a] * he_aa[best_edge_a];
        let edge_b_start = edge_mid_b - axes_b[best_edge_b] * he_bb[best_edge_b];
        let edge_b_end   = edge_mid_b + axes_b[best_edge_b] * he_bb[best_edge_b];
        let pts = closest_points_segments(edge_a_start, edge_a_end, edge_b_start, edge_b_end);
        let feature = 0x03000000u | (0x02u << 16u) | ((best_edge_a & 0xFFu) << 8u) | (best_edge_b & 0xFFu);
        emit_contact_world_points(pts[0], pts[1], contact_normal, body_a, body_b, feature, max_contacts);
        return;
    }

    // ----- Face contact: Sutherland-Hodgman clipping -----

    // Determine reference and incident faces
    var ref_axes: array<vec3<f32>, 3>;
    var inc_axes: array<vec3<f32>, 3>;
    var ref_he: vec3<f32>;
    var inc_he: vec3<f32>;
    var ref_pos: vec3<f32>;
    var inc_pos: vec3<f32>;
    let ref_axis = best_face_axis;

    if best_axis_type == 0u {
        // Reference face on A
        ref_axes = axes_a;
        inc_axes = axes_b;
        ref_he = he_a;
        inc_he = he_b;
        ref_pos = pos_a;
        inc_pos = pos_b;
    } else {
        // Reference face on B
        ref_axes = axes_b;
        inc_axes = axes_a;
        ref_he = he_b;
        inc_he = he_a;
        ref_pos = pos_b;
        inc_pos = pos_a;
    }

    // Reference face outward normal must point from the reference box toward
    // the incident box, matching the CPU reference manifold builder.
    let ref_normal = select(-best_normal, best_normal, best_axis_type == 0u);
    let ref_face = ref_axis * 2u + select(0u, 1u, dot(ref_axes[ref_axis], ref_normal) < 0.0);

    var inc_axis = 0u;
    var best_dot_inc = -1.0;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let dd = abs(dot(inc_axes[i], ref_normal));
        if dd > best_dot_inc {
            best_dot_inc = dd;
            inc_axis = i;
        }
    }
    let ref_he_arr = array<f32, 3>(ref_he.x, ref_he.y, ref_he.z);
    let inc_he_arr = array<f32, 3>(inc_he.x, inc_he.y, inc_he.z);

    var ref_u = vec3<f32>(0.0, 0.0, 0.0);
    var ref_v = vec3<f32>(0.0, 0.0, 0.0);
    var ref_extent_u = 0.0;
    var ref_extent_v = 0.0;
    if ref_axis == 0u {
        ref_u = ref_axes[1];
        ref_v = ref_axes[2];
        ref_extent_u = ref_he.y;
        ref_extent_v = ref_he.z;
    } else if ref_axis == 1u {
        ref_u = ref_axes[0];
        ref_v = ref_axes[2];
        ref_extent_u = ref_he.x;
        ref_extent_v = ref_he.z;
    } else {
        ref_u = ref_axes[0];
        ref_v = ref_axes[1];
        ref_extent_u = ref_he.x;
        ref_extent_v = ref_he.y;
    }
    let ref_center = ref_pos + ref_normal * ref_he_arr[ref_axis];

    var inc_u = vec3<f32>(0.0, 0.0, 0.0);
    var inc_v = vec3<f32>(0.0, 0.0, 0.0);
    var inc_extent_u = 0.0;
    var inc_extent_v = 0.0;
    if inc_axis == 0u {
        inc_u = inc_axes[1];
        inc_v = inc_axes[2];
        inc_extent_u = inc_he.y;
        inc_extent_v = inc_he.z;
    } else if inc_axis == 1u {
        inc_u = inc_axes[0];
        inc_v = inc_axes[2];
        inc_extent_u = inc_he.x;
        inc_extent_v = inc_he.z;
    } else {
        inc_u = inc_axes[0];
        inc_v = inc_axes[1];
        inc_extent_u = inc_he.x;
        inc_extent_v = inc_he.y;
    }
    let inc_sign = select(1.0, -1.0, dot(inc_axes[inc_axis], ref_normal) > 0.0);
    let inc_center = inc_pos + inc_axes[inc_axis] * (inc_sign * inc_he_arr[inc_axis]);

    var clip_poly: array<vec3<f32>, 8>;
    clip_poly[0] = inc_center + inc_u * inc_extent_u + inc_v * inc_extent_v;
    clip_poly[1] = inc_center - inc_u * inc_extent_u + inc_v * inc_extent_v;
    clip_poly[2] = inc_center - inc_u * inc_extent_u - inc_v * inc_extent_v;
    clip_poly[3] = inc_center + inc_u * inc_extent_u - inc_v * inc_extent_v;
    var clip_count = 4u;

    // The generic clip helper keeps the positive half-space, so negate the CPU
    // reference planes which keep points on the <= side.
    clip_count = clip_polygon_against_plane(
        &clip_poly,
        clip_count,
        -ref_u,
        -(dot(ref_u, ref_center) + ref_extent_u),
    );
    clip_count = clip_polygon_against_plane(
        &clip_poly,
        clip_count,
        ref_u,
        -(dot(-ref_u, ref_center) + ref_extent_u),
    );
    clip_count = clip_polygon_against_plane(
        &clip_poly,
        clip_count,
        -ref_v,
        -(dot(ref_v, ref_center) + ref_extent_v),
    );
    clip_count = clip_polygon_against_plane(
        &clip_poly,
        clip_count,
        ref_v,
        -(dot(-ref_v, ref_center) + ref_extent_v),
    );
    let face_feature_prefix =
        0x03000000u |
        ((best_axis_type & 0xFFu) << 20u) |
        ((ref_axis & 0xFFu) << 16u) |
        ((inc_axis & 0xFFu) << 8u);
    if clip_count == 0u {
        let world_a = support_point_box(pos_a, axes_a, he_a, -contact_normal);
        let world_b = support_point_box(pos_b, axes_b, he_b, contact_normal);
        emit_contact_world_points(
            world_a,
            world_b,
            contact_normal,
            body_a,
            body_b,
            face_feature_prefix,
            max_contacts,
        );
        return;
    }

    // Keep only penetrating points and store the exact per-body anchors so the
    // solver sees the same lever arms as the CPU reference manifold builder.
    var final_points: array<vec3<f32>, 8>;
    var final_world_a: array<vec3<f32>, 8>;
    var final_world_b: array<vec3<f32>, 8>;
    var final_features: array<u32, 8>;
    var final_count = 0u;

    for (var i = 0u; i < clip_count; i = i + 1u) {
        let sep = dot(clip_poly[i] - ref_center, ref_normal);
        if sep <= 1e-5 {
            let p_incident = clip_poly[i];
            let p_reference = p_incident - ref_normal * sep;
            var world_a = p_reference;
            var world_b = p_incident;
            if best_axis_type == 1u {
                world_a = p_incident;
                world_b = p_reference;
            }
            let midpoint = (world_a + world_b) * 0.5;
            var duplicate = false;
            for (var j = 0u; j < final_count; j = j + 1u) {
                let dp = final_points[j] - midpoint;
                if dot(dp, dp) < 1e-6 {
                    duplicate = true;
                }
            }
            if !duplicate {
                final_points[final_count] = midpoint;
                final_world_a[final_count] = world_a;
                final_world_b[final_count] = world_b;
                final_features[final_count] = i;
                final_count = final_count + 1u;
            }
        }
    }

    if final_count == 0u {
        let world_a = support_point_box(pos_a, axes_a, he_a, -contact_normal);
        let world_b = support_point_box(pos_b, axes_b, he_b, contact_normal);
        emit_contact_world_points(
            world_a,
            world_b,
            contact_normal,
            body_a,
            body_b,
            face_feature_prefix,
            max_contacts,
        );
        return;
    }

    for (var i = 0u; i < final_count; i = i + 1u) {
        let feature =
            face_feature_prefix |
            (final_features[i] & 0xFFu);
        emit_contact_world_points(
            final_world_a[i],
            final_world_b[i],
            contact_normal,
            body_a,
            body_b,
            feature,
            max_contacts,
        );
    }
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
    body_sphere: u32, body_plane: u32,
    max_contacts: u32,
) {
    let d = dot(plane_normal, sphere_pos) - plane_dist;
    if d >= sphere_r {
        return;
    }
    let normal = plane_normal;
    let depth = d - sphere_r;
    let plane_point = sphere_pos - plane_normal * d;
    emit_plane_contact(plane_point, normal, depth, body_sphere, body_plane, 1u, max_contacts);
}

fn plane_box_test(
    plane_normal: vec3<f32>, plane_dist: f32,
    box_pos: vec3<f32>, box_rot: vec4<f32>, half_ext: vec3<f32>,
    body_box: u32, body_plane: u32,
    max_contacts: u32,
) {
    let normal = plane_normal;
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
            let plane_point = world_v - plane_normal * d;
            let feature = 0x05000000u | i;
            emit_plane_contact(plane_point, normal, d, body_box, body_plane, feature, max_contacts);
        }
    }
}

fn static_floor_box_test(
    plane_normal: vec3<f32>, plane_dist: f32,
    box_pos: vec3<f32>, box_rot: vec4<f32>, half_ext: vec3<f32>,
    body_box: u32, body_plane: u32,
    max_contacts: u32,
) {
    let normal = plane_normal;
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
            let plane_point = world_v - plane_normal * d;
            let feature = 0x31000000u | i;
            emit_plane_contact(plane_point, normal, d, body_box, body_plane, feature, max_contacts);
        }
    }
}

fn stacked_box_plane_test(
    plane_normal: vec3<f32>, plane_dist: f32,
    box_pos: vec3<f32>, box_rot: vec4<f32>, half_ext: vec3<f32>,
    body_box: u32, body_plane: u32,
    max_contacts: u32,
) {
    let normal = plane_normal;
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
            let plane_point = world_v - plane_normal * d;
            let feature = 0x32000000u | i;
            emit_plane_contact(plane_point, normal, d, body_box, body_plane, feature, max_contacts);
        }
    }
}

fn is_floor_like_box(half_ext: vec3<f32>) -> bool {
    return half_ext.x >= half_ext.y * 4.0 && half_ext.z >= half_ext.y * 4.0;
}

fn plane_capsule_test(
    plane_normal: vec3<f32>, plane_dist: f32,
    cap_pos: vec3<f32>, cap_rot: vec4<f32>, cap_hh: f32, cap_r: f32,
    body_capsule: u32, body_plane: u32,
    max_contacts: u32,
) {
    let axis = quat_rotate(cap_rot, vec3<f32>(0.0, 1.0, 0.0));
    let seg_a = cap_pos - axis * cap_hh;
    let seg_b = cap_pos + axis * cap_hh;
    // Test both endpoints as spheres
    plane_sphere_test(plane_normal, plane_dist, seg_a, cap_r, body_capsule, body_plane, max_contacts);
    plane_sphere_test(plane_normal, plane_dist, seg_b, cap_r, body_capsule, body_plane, max_contacts);
}

fn plane_hull_test(
    plane_normal: vec3<f32>, plane_dist: f32,
    hull_pos: vec3<f32>, hull_rot: vec4<f32>, hull_si: u32,
    body_hull: u32, body_plane: u32,
    max_contacts: u32,
) {
    let normal = plane_normal;
    let hull = convex_hulls[hull_si];
    for (var i = 0u; i < hull.vertex_count; i = i + 1u) {
        let cv = convex_verts[hull.vertex_offset + i];
        let local_v = vec3<f32>(cv.x, cv.y, cv.z);
        let world_v = hull_pos + quat_rotate(hull_rot, local_v);
        let d = dot(plane_normal, world_v) - plane_dist;
        if d < 0.0 {
            let plane_point = world_v - plane_normal * d;
            let feature = 0x06000000u | i;
            emit_plane_contact(plane_point, normal, d, body_hull, body_plane, feature, max_contacts);
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

// ---------- Hull-Hull SAT + clipping ----------

// Get the support vertex index on a hull along a direction
fn hull_support_idx(si: u32, pos: vec3<f32>, rot: vec4<f32>, dir: vec3<f32>) -> u32 {
    let hull = convex_hulls[si];
    var best_i = 0u;
    var best_d = -1e30;
    for (var i = 0u; i < hull.vertex_count; i = i + 1u) {
        let wv = hull_world_vert(si, i, pos, rot);
        let dd = dot(wv, dir);
        if dd > best_d {
            best_d = dd;
            best_i = i;
        }
    }
    return best_i;
}

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
    var best_from_a = true;
    var best_face_v0 = 0u;
    var best_face_v1 = 0u;
    var best_face_v2 = 0u;
    var best_is_edge = false;
    var best_edge_i = 0u;
    var best_edge_j = 0u;

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
            best_from_a = true;
            best_face_v0 = i;
            best_face_v1 = (i + 1u) % na;
            best_face_v2 = (i + 2u) % na;
            best_is_edge = false;
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
            best_from_a = false;
            best_face_v0 = i;
            best_face_v1 = (i + 1u) % nb;
            best_face_v2 = (i + 2u) % nb;
            best_is_edge = false;
        }
    }

    // Cross-hull edge-edge axes: brute-force O(na*nb).
    // With max 64 vertices per hull, this is at most 4096 iterations —
    // fast enough on GPU and simpler than Minkowski face pruning.
    for (var i = 0u; i < na; i = i + 1u) {
        let ea0 = hull_world_vert(si_a, i, pos_a, rot_a);
        let ea1 = hull_world_vert(si_a, (i + 1u) % na, pos_a, rot_a);
        let dir_a = ea1 - ea0;
        for (var j = 0u; j < nb; j = j + 1u) {
            let eb0 = hull_world_vert(si_b, j, pos_b, rot_b);
            let eb1 = hull_world_vert(si_b, (j + 1u) % nb, pos_b, rot_b);
            let dir_b = eb1 - eb0;
            var axis = cross(dir_a, dir_b);
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
                best_is_edge = true;
                best_edge_i = i;
                best_edge_j = j;
            }
        }
    }

    // Ensure normal points from A to B
    if dot(best_normal, d) < 0.0 {
        best_normal = -best_normal;
    }

    // Edge-edge: emit single contact from closest points on the two edges
    if best_is_edge {
        let ea0 = hull_world_vert(si_a, best_edge_i, pos_a, rot_a);
        let ea1 = hull_world_vert(si_a, (best_edge_i + 1u) % na, pos_a, rot_a);
        let eb0 = hull_world_vert(si_b, best_edge_j, pos_b, rot_b);
        let eb1 = hull_world_vert(si_b, (best_edge_j + 1u) % nb, pos_b, rot_b);
        let pts = closest_points_segments(ea0, ea1, eb0, eb1);
        let contact_point = (pts[0] + pts[1]) * 0.5;
        let feature = 0x07000000u | ((best_edge_i & 0xFFu) << 8u) | (best_edge_j & 0xFFu);
        emit_contact(contact_point, best_normal, min_depth, body_a, body_b, feature, max_contacts);
        return;
    }

    // Build reference face polygon (the 3-vertex triangle from the SAT face)
    // and clip incident hull's closest face against it.

    // Reference face: use the triangle that generated the best_normal
    var ref_si: u32;
    var ref_pos: vec3<f32>;
    var ref_rot: vec4<f32>;
    var inc_si: u32;
    var inc_pos: vec3<f32>;
    var inc_rot: vec4<f32>;

    if best_from_a {
        ref_si = si_a; ref_pos = pos_a; ref_rot = rot_a;
        inc_si = si_b; inc_pos = pos_b; inc_rot = rot_b;
    } else {
        ref_si = si_b; ref_pos = pos_b; ref_rot = rot_b;
        inc_si = si_a; inc_pos = pos_a; inc_rot = rot_a;
    }

    // Reference face: 3 vertices forming the face
    let rv0 = hull_world_vert(ref_si, best_face_v0, ref_pos, ref_rot);
    let rv1 = hull_world_vert(ref_si, best_face_v1, ref_pos, ref_rot);
    let rv2 = hull_world_vert(ref_si, best_face_v2, ref_pos, ref_rot);

    // Reference face normal (recompute to ensure correct orientation)
    var ref_normal = cross(rv1 - rv0, rv2 - rv0);
    let rn_len = length(ref_normal);
    if rn_len < 1e-12 {
        // Degenerate face, fall back to single contact
        let contact_point = (pos_a + pos_b) * 0.5 + best_normal * min_depth * 0.5;
        emit_contact(contact_point, best_normal, min_depth, body_a, body_b, 0x08000000u, max_contacts);
        return;
    }
    ref_normal = ref_normal / rn_len;
    if dot(ref_normal, best_normal) < 0.0 {
        ref_normal = -ref_normal;
    }

    // Build incident face: find the support vertex on the incident hull most
    // anti-aligned with best_normal, then build a small fan around it
    let inc_hull = convex_hulls[inc_si];
    let inc_n = inc_hull.vertex_count;
    let support_i = hull_support_idx(inc_si, inc_pos, inc_rot, -best_normal);

    // Use 3 consecutive vertices around support as incident polygon (triangle)
    var clip_poly: array<vec3<f32>, 8>;
    let ic0 = support_i;
    let ic1 = (support_i + 1u) % inc_n;
    let ic2 = (support_i + inc_n - 1u) % inc_n;
    clip_poly[0] = hull_world_vert(inc_si, ic2, inc_pos, inc_rot);
    clip_poly[1] = hull_world_vert(inc_si, ic0, inc_pos, inc_rot);
    clip_poly[2] = hull_world_vert(inc_si, ic1, inc_pos, inc_rot);
    var clip_count = 3u;

    // Clip against 3 side planes of the reference triangle
    let ref_tri = array<vec3<f32>, 3>(rv0, rv1, rv2);
    for (var i = 0u; i < 3u; i = i + 1u) {
        let j = (i + 1u) % 3u;
        let edge = ref_tri[j] - ref_tri[i];
        let side_normal = cross(ref_normal, edge);
        let side_len = length(side_normal);
        if side_len < 1e-12 { continue; }
        let sn = side_normal / side_len;
        let side_dist = dot(sn, ref_tri[i]);
        clip_count = clip_polygon_against_plane(&clip_poly, clip_count, sn, side_dist);
    }

    if clip_count == 0u { return; }

    // Filter: keep points below reference plane
    let ref_plane_dist = dot(ref_normal, rv0);
    var final_points: array<vec3<f32>, 8>;
    var final_depths: array<f32, 8>;
    var final_count = 0u;

    for (var i = 0u; i < clip_count; i = i + 1u) {
        let sep = dot(ref_normal, clip_poly[i]) - ref_plane_dist;
        if sep <= 0.0 {
            final_points[final_count] = clip_poly[i];
            final_depths[final_count] = sep;
            final_count = final_count + 1u;
        }
    }

    if final_count == 0u {
        // Fallback: single contact at midpoint
        let contact_point = (pos_a + pos_b) * 0.5 + best_normal * min_depth * 0.5;
        emit_contact(contact_point, best_normal, min_depth, body_a, body_b, 0x09000000u, max_contacts);
        return;
    }

    // Reduce to at most 4 contacts
    var out_points: array<vec3<f32>, 4>;
    var out_depths: array<f32, 4>;
    let emit_count = reduce_manifold(&final_points, &final_depths, final_count, best_normal, &out_points, &out_depths);

    for (var i = 0u; i < emit_count; i = i + 1u) {
        let feature = 0x0A000000u | ((i & 0xFFu) << 0u);
        emit_contact(out_points[i], best_normal, out_depths[i], body_a, body_b, feature, max_contacts);
    }
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
    emit_contact(contact_point, normal, depth, body_sphere, body_hull, 1u, max_contacts);
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
    emit_contact(contact_point, best_normal, min_depth, body_box, body_hull, 0x0B000000u, max_contacts);
}

// ---------- Main dispatch ----------

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pi = gid.x;
    let num_pairs = params.counts.z;
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

    let max_contacts = params.counts.x * 8u;

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
        let pd = plane_params.planes[i2];
        plane_sphere_test(pd.xyz, pd.w, p1, ra, b1, b2, max_contacts);
    } else if s1 == SHAPE_BOX && s2 == SHAPE_BOX {
        let ha = boxes[i1].half_extents.xyz;
        let hb = boxes[i2].half_extents.xyz;
        let a_static = bodies[b1].position_inv_mass.w <= 0.0;
        let b_static = bodies[b2].position_inv_mass.w <= 0.0;
        if a_static && !b_static && is_floor_like_box(ha) {
            let plane_normal = quat_rotate(r1, vec3<f32>(0.0, 1.0, 0.0));
            let plane_dist = dot(plane_normal, p1 + plane_normal * ha.y);
            static_floor_box_test(plane_normal, plane_dist, p2, r2, hb, b2, b1, max_contacts);
        } else if b_static && !a_static && is_floor_like_box(hb) {
            let plane_normal = quat_rotate(r2, vec3<f32>(0.0, 1.0, 0.0));
            let plane_dist = dot(plane_normal, p2 + plane_normal * hb.y);
            static_floor_box_test(plane_normal, plane_dist, p1, r1, ha, b1, b2, max_contacts);
        } else {
            box_box_test(p1, r1, ha, p2, r2, hb, b1, b2, max_contacts);
        }
    } else if s1 == SHAPE_BOX && s2 == SHAPE_CAPSULE {
        let ha = boxes[i1].half_extents.xyz;
        let cap = capsules[i2];
        box_capsule_test(p1, r1, ha, p2, r2, cap.half_height, cap.radius, b1, b2, max_contacts);
    } else if s1 == SHAPE_BOX && s2 == SHAPE_CONVEX_HULL {
        let ha = boxes[i1].half_extents.xyz;
        box_hull_test(p1, r1, ha, p2, r2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_BOX && s2 == SHAPE_PLANE {
        let ha = boxes[i1].half_extents.xyz;
        let pd = plane_params.planes[i2];
        plane_box_test(pd.xyz, pd.w, p1, r1, ha, b1, b2, max_contacts);
    } else if s1 == SHAPE_CAPSULE && s2 == SHAPE_CAPSULE {
        let ca = capsules[i1];
        let cb = capsules[i2];
        capsule_capsule_test(p1, r1, ca.half_height, ca.radius, p2, r2, cb.half_height, cb.radius, b1, b2, max_contacts);
    } else if s1 == SHAPE_CAPSULE && s2 == SHAPE_CONVEX_HULL {
        let cap = capsules[i1];
        capsule_hull_test(p1, r1, cap.half_height, cap.radius, p2, r2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_CAPSULE && s2 == SHAPE_PLANE {
        let cap = capsules[i1];
        let pd = plane_params.planes[i2];
        plane_capsule_test(pd.xyz, pd.w, p1, r1, cap.half_height, cap.radius, b1, b2, max_contacts);
    } else if s1 == SHAPE_CONVEX_HULL && s2 == SHAPE_CONVEX_HULL {
        hull_hull_test(p1, r1, i1, p2, r2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_CONVEX_HULL && s2 == SHAPE_PLANE {
        let pd = plane_params.planes[i2];
        plane_hull_test(pd.xyz, pd.w, p1, r1, i1, b1, b2, max_contacts);
    }
    // SHAPE_PLANE vs SHAPE_PLANE: no collision
}
"#;
