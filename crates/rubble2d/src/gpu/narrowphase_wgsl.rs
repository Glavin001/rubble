/// WGSL source for GPU narrowphase contact generation (2D).
///
/// For each broadphase pair, detects collisions based on shape types
/// (circle-circle, circle-rect, capsule, polygon) and writes contacts to the output buffer.
pub const NARROWPHASE_2D_WGSL: &str = r#"
// ---------- Types ----------

struct Body2D {
    position_inv_mass: vec4<f32>, // (x, y, angle, 1/m)
    lin_vel:           vec4<f32>, // (vx, vy, angular_vel, 0)
    _pad0:             vec4<f32>,
    _pad1:             vec4<f32>,
};

struct ShapeInfo {
    shape_type:  u32,
    shape_index: u32,
};

struct CircleData {
    radius: f32,
    _pad0:  f32,
    _pad1:  f32,
    _pad2:  f32,
};

struct RectDataGpu {
    half_extents: vec4<f32>, // (hx, hy, 0, 0)
};

struct Pair {
    a: u32,
    b: u32,
};

struct Contact2D {
    point:           vec4<f32>, // (x, y, depth, 0)
    normal:          vec4<f32>, // (nx, ny, tx, ty)
    local_anchors:   vec4<f32>, // (rA.x, rA.y, rB.x, rB.y)
    lambda_penalty:  vec4<f32>, // (lambda_n, lambda_t, penalty_n, penalty_t)
    body_a:     u32,
    body_b:     u32,
    feature_id: u32,
    flags:      u32,
};

struct SimParams2D {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
};

const SHAPE_CIRCLE:         u32 = 0u;
const SHAPE_RECT:           u32 = 1u;
const SHAPE_CONVEX_POLYGON: u32 = 2u;
const SHAPE_CAPSULE:        u32 = 3u;

struct ConvexPolyInfo {
    vertex_offset: u32,
    vertex_count:  u32,
    _pad0:         u32,
    _pad1:         u32,
};

struct ConvexVert2D {
    x: f32,
    y: f32,
    _pad0: f32,
    _pad1: f32,
};

struct CapsuleData2DGpu {
    half_height: f32,
    radius:      f32,
    _pad0:       f32,
    _pad1:       f32,
};

@group(0) @binding(0) var<storage, read>       bodies:        array<Body2D>;
@group(0) @binding(1) var<storage, read>       shape_infos:   array<ShapeInfo>;
@group(0) @binding(2) var<storage, read>       pairs:         array<Pair>;
@group(0) @binding(3) var<storage, read>       circles:       array<CircleData>;
@group(0) @binding(4) var<storage, read>       rects:         array<RectDataGpu>;
@group(0) @binding(5) var<storage, read_write> contacts:      array<Contact2D>;
@group(0) @binding(6) var<storage, read_write> contact_count: atomic<u32>;
@group(0) @binding(7) var<uniform>             params:        SimParams2D;
@group(0) @binding(8) var<storage, read>       convex_polys:  array<ConvexPolyInfo>;
@group(0) @binding(9) var<storage, read>       convex_verts:  array<ConvexVert2D>;
@group(0) @binding(10) var<storage, read>      capsules_2d:   array<CapsuleData2DGpu>;
@group(0) @binding(11) var<storage, read_write> pair_count:   atomic<u32>;

fn emit_contact_2d(
    point: vec2<f32>,
    normal: vec2<f32>,
    depth: f32,
    body_a: u32,
    body_b: u32,
    feature_id: u32,
    max_contacts: u32,
) {
    let slot = atomicAdd(&contact_count, 1u);
    if slot >= max_contacts {
        return;
    }
    let tangent = vec2<f32>(-normal.y, normal.x);
    let pos_a = bodies[body_a].position_inv_mass.xy;
    let pos_b = bodies[body_b].position_inv_mass.xy;
    let angle_a = bodies[body_a].position_inv_mass.z;
    let angle_b = bodies[body_b].position_inv_mass.z;
    let ca_a = cos(angle_a);
    let sa_a = sin(angle_a);
    let ca_b = cos(angle_b);
    let sa_b = sin(angle_b);
    let world_a = point + normal * depth * 0.5;
    let world_b = point - normal * depth * 0.5;
    let local_a = vec2<f32>(
        ca_a * (world_a.x - pos_a.x) + sa_a * (world_a.y - pos_a.y),
        -sa_a * (world_a.x - pos_a.x) + ca_a * (world_a.y - pos_a.y),
    );
    let local_b = vec2<f32>(
        ca_b * (world_b.x - pos_b.x) + sa_b * (world_b.y - pos_b.y),
        -sa_b * (world_b.x - pos_b.x) + ca_b * (world_b.y - pos_b.y),
    );
    contacts[slot].point  = vec4<f32>(point.x, point.y, depth, 0.0);
    contacts[slot].normal = vec4<f32>(normal.x, normal.y, tangent.x, tangent.y);
    contacts[slot].local_anchors = vec4<f32>(local_a.x, local_a.y, local_b.x, local_b.y);
    let k_start = params.solver.z;
    contacts[slot].lambda_penalty = vec4<f32>(0.0, 0.0, k_start, k_start);
    contacts[slot].body_a = body_a;
    contacts[slot].body_b = body_b;
    contacts[slot].feature_id = feature_id;
    contacts[slot].flags = 0u;
}

// ---------- Helpers ----------

// Closest point on line segment ab to point p
fn closest_point_on_segment_2d(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> vec2<f32> {
    let ab = b - a;
    let len2 = dot(ab, ab);
    if len2 < 1e-12 {
        return a;
    }
    let t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    return a + ab * t;
}

// Get capsule segment endpoints in world space
fn capsule_endpoints_2d(pos: vec2<f32>, angle: f32, half_height: f32) -> array<vec2<f32>, 2> {
    let ca = cos(angle);
    let sa = sin(angle);
    // Local Y axis
    let axis = vec2<f32>(-sa * half_height, ca * half_height);
    return array<vec2<f32>, 2>(pos + axis, pos - axis);
}

struct IncidentEdge2D {
    points: array<vec2<f32>, 2>,
    edge_index: u32,
};

struct ClipResult2D {
    points: array<vec2<f32>, 2>,
    count: u32,
};

fn shape_vertex_count_2d(shape_type: u32, shape_index: u32) -> u32 {
    if shape_type == SHAPE_RECT {
        return 4u;
    }
    return convex_polys[shape_index].vertex_count;
}

fn shape_local_vert_2d(shape_type: u32, shape_index: u32, idx: u32) -> vec2<f32> {
    if shape_type == SHAPE_RECT {
        let he = rects[shape_index].half_extents.xy;
        switch idx & 3u {
            case 0u: { return vec2<f32>(-he.x, -he.y); }
            case 1u: { return vec2<f32>( he.x, -he.y); }
            case 2u: { return vec2<f32>( he.x,  he.y); }
            default: { return vec2<f32>(-he.x,  he.y); }
        }
    }
    let poly = convex_polys[shape_index];
    let cv = convex_verts[poly.vertex_offset + idx];
    return vec2<f32>(cv.x, cv.y);
}

fn shape_world_vert_2d(
    shape_type: u32,
    shape_index: u32,
    pos: vec2<f32>,
    ca: f32,
    sa: f32,
    idx: u32,
) -> vec2<f32> {
    let local_v = shape_local_vert_2d(shape_type, shape_index, idx);
    return pos + rotate2d(local_v, ca, sa);
}

fn shape_face_normal_2d(
    shape_type: u32,
    shape_index: u32,
    pos: vec2<f32>,
    ca: f32,
    sa: f32,
    edge_idx: u32,
) -> vec2<f32> {
    let count = shape_vertex_count_2d(shape_type, shape_index);
    let v0 = shape_world_vert_2d(shape_type, shape_index, pos, ca, sa, edge_idx);
    let v1 = shape_world_vert_2d(shape_type, shape_index, pos, ca, sa, (edge_idx + 1u) % count);
    let edge = v1 - v0;
    let n = vec2<f32>(edge.y, -edge.x);
    let len = length(n);
    if len < 1e-12 {
        return vec2<f32>(0.0, 1.0);
    }
    return n / len;
}

fn project_shape_2d(
    shape_type: u32,
    shape_index: u32,
    pos: vec2<f32>,
    ca: f32,
    sa: f32,
    axis: vec2<f32>,
) -> vec2<f32> {
    let count = shape_vertex_count_2d(shape_type, shape_index);
    var mn = 1e30;
    var mx = -1e30;
    for (var i = 0u; i < count; i = i + 1u) {
        let wv = shape_world_vert_2d(shape_type, shape_index, pos, ca, sa, i);
        let p = dot(wv, axis);
        mn = min(mn, p);
        mx = max(mx, p);
    }
    return vec2<f32>(mn, mx);
}

fn incident_edge_2d(
    shape_type: u32,
    shape_index: u32,
    pos: vec2<f32>,
    ca: f32,
    sa: f32,
    reference_normal: vec2<f32>,
) -> IncidentEdge2D {
    let count = shape_vertex_count_2d(shape_type, shape_index);
    var best_edge = 0u;
    var best_dot = 1e30;
    for (var i = 0u; i < count; i = i + 1u) {
        let n = shape_face_normal_2d(shape_type, shape_index, pos, ca, sa, i);
        let d = dot(n, reference_normal);
        if d < best_dot {
            best_dot = d;
            best_edge = i;
        }
    }
    var result: IncidentEdge2D;
    result.edge_index = best_edge;
    result.points[0] = shape_world_vert_2d(shape_type, shape_index, pos, ca, sa, best_edge);
    result.points[1] =
        shape_world_vert_2d(shape_type, shape_index, pos, ca, sa, (best_edge + 1u) % count);
    return result;
}

fn clip_segment_to_line_2d(
    p0: vec2<f32>,
    p1: vec2<f32>,
    normal: vec2<f32>,
    offset: f32,
) -> ClipResult2D {
    var result: ClipResult2D;
    result.count = 0u;

    let d0 = dot(normal, p0) - offset;
    let d1 = dot(normal, p1) - offset;

    if d0 <= 0.0 {
        result.points[result.count] = p0;
        result.count = result.count + 1u;
    }
    if d1 <= 0.0 {
        result.points[result.count] = p1;
        result.count = result.count + 1u;
    }

    if d0 * d1 < 0.0 && result.count < 2u {
        let t = d0 / (d0 - d1);
        result.points[result.count] = p0 + (p1 - p0) * t;
        result.count = result.count + 1u;
    }
    return result;
}

fn convex_convex_manifold_2d(
    shape_a: u32, si_a: u32, pos_a: vec2<f32>, angle_a: f32,
    shape_b: u32, si_b: u32, pos_b: vec2<f32>, angle_b: f32,
    body_a: u32, body_b: u32,
    max_contacts: u32,
) {
    let ca_a = cos(angle_a);
    let sa_a = sin(angle_a);
    let ca_b = cos(angle_b);
    let sa_b = sin(angle_b);
    let d = pos_b - pos_a;

    var best_sep = -1e30;
    var ref_shape = 0u;
    var ref_edge = 0u;
    var normal_ab = vec2<f32>(0.0, 1.0);
    var found_axis = false;

    let count_a = shape_vertex_count_2d(shape_a, si_a);
    for (var i = 0u; i < count_a; i = i + 1u) {
        let outward = shape_face_normal_2d(shape_a, si_a, pos_a, ca_a, sa_a, i);
        if dot(outward, d) <= 0.0 {
            continue;
        }
        let proj_a = project_shape_2d(shape_a, si_a, pos_a, ca_a, sa_a, outward);
        let proj_b = project_shape_2d(shape_b, si_b, pos_b, ca_b, sa_b, outward);
        let overlap = min(proj_a.y, proj_b.y) - max(proj_a.x, proj_b.x);
        if overlap < 0.0 {
            return;
        }
        let sep = -overlap;
        if !found_axis || sep > best_sep {
            best_sep = sep;
            ref_shape = 0u;
            ref_edge = i;
            normal_ab = outward;
            found_axis = true;
        }
    }

    let count_b = shape_vertex_count_2d(shape_b, si_b);
    for (var i = 0u; i < count_b; i = i + 1u) {
        let outward_b = shape_face_normal_2d(shape_b, si_b, pos_b, ca_b, sa_b, i);
        let axis = -outward_b;
        if dot(axis, d) <= 0.0 {
            continue;
        }
        let proj_a = project_shape_2d(shape_a, si_a, pos_a, ca_a, sa_a, axis);
        let proj_b = project_shape_2d(shape_b, si_b, pos_b, ca_b, sa_b, axis);
        let overlap = min(proj_a.y, proj_b.y) - max(proj_a.x, proj_b.x);
        if overlap < 0.0 {
            return;
        }
        let sep = -overlap;
        if !found_axis || sep > best_sep {
            best_sep = sep;
            ref_shape = 1u;
            ref_edge = i;
            normal_ab = axis;
            found_axis = true;
        }
    }

    if !found_axis {
        // Deep overlap / degenerate case fallback.
        let point = (pos_a + pos_b) * 0.5 + normal_ab * best_sep * 0.5;
        emit_contact_2d(point, normal_ab, best_sep, body_a, body_b, 1u, max_contacts);
        return;
    }

    var ref_shape_type = shape_a;
    var ref_si = si_a;
    var ref_pos = pos_a;
    var ref_ca = ca_a;
    var ref_sa = sa_a;
    var inc_shape_type = shape_b;
    var inc_si = si_b;
    var inc_pos = pos_b;
    var inc_ca = ca_b;
    var inc_sa = sa_b;
    var ref_outward = normal_ab;

    if ref_shape == 1u {
        ref_shape_type = shape_b;
        ref_si = si_b;
        ref_pos = pos_b;
        ref_ca = ca_b;
        ref_sa = sa_b;
        inc_shape_type = shape_a;
        inc_si = si_a;
        inc_pos = pos_a;
        inc_ca = ca_a;
        inc_sa = sa_a;
        ref_outward = -normal_ab;
    }

    let ref_count = shape_vertex_count_2d(ref_shape_type, ref_si);
    let ref_v0 = shape_world_vert_2d(ref_shape_type, ref_si, ref_pos, ref_ca, ref_sa, ref_edge);
    let ref_v1 =
        shape_world_vert_2d(ref_shape_type, ref_si, ref_pos, ref_ca, ref_sa, (ref_edge + 1u) % ref_count);
    let side_normal = normalize(ref_v1 - ref_v0);
    let front = dot(ref_outward, ref_v0);
    let neg_side = -dot(side_normal, ref_v0);
    let pos_side = dot(side_normal, ref_v1);

    let incident = incident_edge_2d(inc_shape_type, inc_si, inc_pos, inc_ca, inc_sa, ref_outward);
    let clip1 = clip_segment_to_line_2d(incident.points[0], incident.points[1], -side_normal, neg_side);
    if clip1.count < 2u {
        return;
    }
    let clip2 = clip_segment_to_line_2d(clip1.points[0], clip1.points[1], side_normal, pos_side);
    if clip2.count == 0u {
        return;
    }

    var emitted = 0u;
    for (var i = 0u; i < clip2.count; i = i + 1u) {
        let separation = dot(ref_outward, clip2.points[i]) - front;
        if separation <= 0.0 {
            let point = clip2.points[i] - ref_outward * separation * 0.5;
            let feature =
                ((ref_shape & 0xFFu) << 24u) |
                ((ref_edge & 0xFFu) << 16u) |
                ((incident.edge_index & 0xFFu) << 8u) |
                (i & 0xFFu);
            emit_contact_2d(point, normal_ab, separation, body_a, body_b, feature, max_contacts);
            emitted = emitted + 1u;
        }
    }

    if emitted == 0u {
        let point = (pos_a + pos_b) * 0.5 + normal_ab * best_sep * 0.5;
        let feature =
            ((ref_shape & 0xFFu) << 24u) |
            ((ref_edge & 0xFFu) << 16u) |
            ((incident.edge_index & 0xFFu) << 8u);
        emit_contact_2d(point, normal_ab, best_sep, body_a, body_b, feature, max_contacts);
    }
}

// Closest points between two segments, returns (point_on_ab, point_on_cd)
fn closest_points_segments_2d(
    a: vec2<f32>, b: vec2<f32>,
    c: vec2<f32>, d: vec2<f32>,
) -> array<vec2<f32>, 2> {
    let ab = b - a;
    let cd = d - c;
    let ac = c - a;
    let d1 = dot(ab, ab);
    let d2 = dot(cd, cd);
    let d3 = dot(ab, cd);
    let d4 = dot(ab, ac);
    let d5 = dot(cd, ac);

    var s: f32;
    var t: f32;
    let denom = d1 * d2 - d3 * d3;

    if denom < 1e-12 {
        s = 0.0;
        t = d5 / max(d2, 1e-12);
    } else {
        s = (d4 * d2 - d3 * d5) / denom;
        t = (d4 * d3 - d1 * d5) / denom;
    }

    s = clamp(s, 0.0, 1.0);
    t = clamp((d3 * s + d5) / max(d2, 1e-12), 0.0, 1.0);
    s = clamp((d3 * t - d4) / max(d1, 1e-12), 0.0, 1.0);

    return array<vec2<f32>, 2>(a + ab * s, c + cd * t);
}

// ---------- Collision tests ----------

fn circle_circle_test(
    pos_a: vec2<f32>, radius_a: f32,
    pos_b: vec2<f32>, radius_b: f32,
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
    let normal = -diff / dist;
    let depth = dist - sum_r; // negative when penetrating
    let point = pos_a - normal * (radius_a + depth * 0.5);
    emit_contact_2d(point, normal, depth, body_a, body_b, 1u, max_contacts);
}

fn circle_rect_test(
    circle_pos: vec2<f32>, radius: f32,
    rect_pos: vec2<f32>, rect_angle: f32, half_ext: vec2<f32>,
    body_circle: u32, body_rect: u32,
    max_contacts: u32,
) {
    // Transform circle center into rect local space
    let ca = cos(-rect_angle);
    let sa = sin(-rect_angle);
    let d = circle_pos - rect_pos;
    let local_center = vec2<f32>(ca * d.x - sa * d.y, sa * d.x + ca * d.y);

    // Clamp to rect
    let closest = clamp(local_center, -half_ext, half_ext);
    let diff = local_center - closest;
    let dist2 = dot(diff, diff);

    if dist2 >= radius * radius {
        return;
    }

    var normal_local: vec2<f32>;
    var depth: f32;

    if dist2 > 1e-12 {
        // Circle center outside rect
        let dist = sqrt(dist2);
        normal_local = diff / dist;
        depth = dist - radius;
    } else {
        // Circle center inside rect -- find closest edge
        let face_dists = half_ext - abs(local_center);
        if face_dists.x <= face_dists.y {
            normal_local = vec2<f32>(select(-1.0, 1.0, local_center.x > 0.0), 0.0);
            depth = -(face_dists.x + radius);
        } else {
            normal_local = vec2<f32>(0.0, select(-1.0, 1.0, local_center.y > 0.0));
            depth = -(face_dists.y + radius);
        }
    }

    // Rotate normal back to world space
    let ca_fwd = cos(rect_angle);
    let sa_fwd = sin(rect_angle);
    let normal_world = vec2<f32>(
        ca_fwd * normal_local.x - sa_fwd * normal_local.y,
        sa_fwd * normal_local.x + ca_fwd * normal_local.y,
    );
    // Solver convention expects normals from body_b to body_a.
    let normal_ab = normal_world;
    let contact_point = circle_pos - normal_ab * (radius + depth * 0.5);
    emit_contact_2d(contact_point, normal_ab, depth, body_circle, body_rect, 1u, max_contacts);
}

fn rect_rect_test(
    pos_a: vec2<f32>, angle_a: f32, si_a: u32,
    pos_b: vec2<f32>, angle_b: f32, si_b: u32,
    body_a: u32, body_b: u32,
    max_contacts: u32,
) {
    convex_convex_manifold_2d(
        SHAPE_RECT,
        si_a,
        pos_a,
        angle_a,
        SHAPE_RECT,
        si_b,
        pos_b,
        angle_b,
        body_a,
        body_b,
        max_contacts,
    );
}

// Rotate a 2D vector by angle
fn rotate2d(v: vec2<f32>, ca: f32, sa: f32) -> vec2<f32> {
    return vec2<f32>(ca * v.x - sa * v.y, sa * v.x + ca * v.y);
}

// Get a world-space vertex from a convex polygon
fn poly_world_vert(poly_si: u32, vi: u32, pos: vec2<f32>, ca: f32, sa: f32) -> vec2<f32> {
    let poly = convex_polys[poly_si];
    let cv = convex_verts[poly.vertex_offset + vi];
    let local_v = vec2<f32>(cv.x, cv.y);
    return pos + rotate2d(local_v, ca, sa);
}

// Project a convex polygon onto an axis and return (min, max)
fn poly_project(poly_si: u32, pos: vec2<f32>, ca: f32, sa: f32, axis: vec2<f32>) -> vec2<f32> {
    let poly = convex_polys[poly_si];
    var mn = 1e30;
    var mx = -1e30;
    for (var i = 0u; i < poly.vertex_count; i = i + 1u) {
        let wv = poly_world_vert(poly_si, i, pos, ca, sa);
        let p = dot(wv, axis);
        mn = min(mn, p);
        mx = max(mx, p);
    }
    return vec2<f32>(mn, mx);
}

// SAT test between two convex polygons
fn poly_poly_test(
    pos_a: vec2<f32>, angle_a: f32, si_a: u32,
    pos_b: vec2<f32>, angle_b: f32, si_b: u32,
    body_a: u32, body_b: u32,
    max_contacts: u32,
) {
    convex_convex_manifold_2d(
        SHAPE_CONVEX_POLYGON,
        si_a,
        pos_a,
        angle_a,
        SHAPE_CONVEX_POLYGON,
        si_b,
        pos_b,
        angle_b,
        body_a,
        body_b,
        max_contacts,
    );
}

// SAT test: circle vs convex polygon
fn circle_poly_test(
    circle_pos: vec2<f32>, radius: f32,
    poly_pos: vec2<f32>, poly_angle: f32, poly_si: u32,
    body_circle: u32, body_poly: u32,
    max_contacts: u32,
) {
    let ca = cos(poly_angle);
    let sa = sin(poly_angle);
    let poly = convex_polys[poly_si];
    let d = poly_pos - circle_pos;

    var min_overlap = 1e30;
    var best_normal = vec2<f32>(0.0, 1.0);

    // Find closest vertex to circle center
    var closest_dist2 = 1e30;
    var closest_vert = poly_pos;
    for (var i = 0u; i < poly.vertex_count; i = i + 1u) {
        let wv = poly_world_vert(poly_si, i, poly_pos, ca, sa);
        let dv = wv - circle_pos;
        let d2 = dot(dv, dv);
        if d2 < closest_dist2 {
            closest_dist2 = d2;
            closest_vert = wv;
        }
    }

    // Test axis from circle center to closest vertex
    let cv_diff = closest_vert - circle_pos;
    let cv_dist = sqrt(dot(cv_diff, cv_diff));
    if cv_dist > 1e-12 {
        let axis = cv_diff / cv_dist;
        let sp_c = dot(circle_pos, axis);
        let sp_min = sp_c - radius;
        let sp_max = sp_c + radius;
        let pp = poly_project(poly_si, poly_pos, ca, sa, axis);
        let overlap = min(sp_max, pp.y) - max(sp_min, pp.x);
        if overlap < 0.0 {
            return;
        }
        if overlap < min_overlap {
            min_overlap = overlap;
            best_normal = select(axis, -axis, dot(axis, d) > 0.0);
        }
    }

    // Test edge normals
    let n = poly.vertex_count;
    for (var i = 0u; i < n; i = i + 1u) {
        let v0 = poly_world_vert(poly_si, i, poly_pos, ca, sa);
        let v1 = poly_world_vert(poly_si, (i + 1u) % n, poly_pos, ca, sa);
        let edge = v1 - v0;
        var axis = vec2<f32>(-edge.y, edge.x);
        let len2 = dot(axis, axis);
        if len2 < 1e-12 {
            continue;
        }
        axis = axis / sqrt(len2);
        let sp_c = dot(circle_pos, axis);
        let sp_min = sp_c - radius;
        let sp_max = sp_c + radius;
        let pp = poly_project(poly_si, poly_pos, ca, sa, axis);
        let overlap = min(sp_max, pp.y) - max(sp_min, pp.x);
        if overlap < 0.0 {
            return;
        }
        if overlap < min_overlap {
            min_overlap = overlap;
            best_normal = select(axis, -axis, dot(axis, d) > 0.0);
        }
    }

    let depth = -min_overlap;
    // Normal points A→B (circle→poly)
    let normal = best_normal;
    let contact_point = circle_pos + normal * (radius + depth * 0.5);
    emit_contact_2d(contact_point, normal, depth, body_circle, body_poly, 1u, max_contacts);
}

// SAT test: rect vs convex polygon
// Tests 2 rect face axes + polygon edge normals.
fn rect_poly_test(
    rect_pos: vec2<f32>, rect_angle: f32, rect_si: u32,
    poly_pos: vec2<f32>, poly_angle: f32, poly_si: u32,
    body_rect: u32, body_poly: u32,
    max_contacts: u32,
) {
    convex_convex_manifold_2d(
        SHAPE_RECT,
        rect_si,
        rect_pos,
        rect_angle,
        SHAPE_CONVEX_POLYGON,
        poly_si,
        poly_pos,
        poly_angle,
        body_rect,
        body_poly,
        max_contacts,
    );
}

// ---------- Capsule collision tests ----------

// Circle vs Capsule: find closest point on capsule segment to circle center,
// then reduce to circle-circle test.
fn circle_capsule_test(
    circle_pos: vec2<f32>, circle_radius: f32,
    cap_pos: vec2<f32>, cap_angle: f32, cap_si: u32,
    body_circle: u32, body_cap: u32,
    max_contacts: u32,
) {
    let cap = capsules_2d[cap_si];
    let eps = capsule_endpoints_2d(cap_pos, cap_angle, cap.half_height);
    let closest = closest_point_on_segment_2d(eps[0], eps[1], circle_pos);
    // Reduce to circle-circle
    circle_circle_test(circle_pos, circle_radius, closest, cap.radius, body_circle, body_cap, max_contacts);
}

// Capsule vs Capsule: closest points between the two segments, then circle-circle.
fn capsule_capsule_test(
    pos_a: vec2<f32>, angle_a: f32, si_a: u32,
    pos_b: vec2<f32>, angle_b: f32, si_b: u32,
    body_a: u32, body_b: u32,
    max_contacts: u32,
) {
    let cap_a = capsules_2d[si_a];
    let cap_b = capsules_2d[si_b];
    let eps_a = capsule_endpoints_2d(pos_a, angle_a, cap_a.half_height);
    let eps_b = capsule_endpoints_2d(pos_b, angle_b, cap_b.half_height);
    let cp = closest_points_segments_2d(eps_a[0], eps_a[1], eps_b[0], eps_b[1]);
    circle_circle_test(cp[0], cap_a.radius, cp[1], cap_b.radius, body_a, body_b, max_contacts);
}

// Rect vs Capsule: find closest point on capsule segment to rect center,
// then test that sphere against the rect. Also test the two capsule endpoints.
fn rect_capsule_test(
    rect_pos: vec2<f32>, rect_angle: f32, half_ext: vec2<f32>,
    cap_pos: vec2<f32>, cap_angle: f32, cap_si: u32,
    body_rect: u32, body_cap: u32,
    max_contacts: u32,
) {
    let cap = capsules_2d[cap_si];
    let eps = capsule_endpoints_2d(cap_pos, cap_angle, cap.half_height);

    // Test each capsule endpoint sphere against rect
    circle_rect_test(eps[0], cap.radius, rect_pos, rect_angle, half_ext, body_cap, body_rect, max_contacts);
    circle_rect_test(eps[1], cap.radius, rect_pos, rect_angle, half_ext, body_cap, body_rect, max_contacts);

    // Also test the segment midpoint (capsule center) for better coverage
    let mid = closest_point_on_segment_2d(eps[0], eps[1], rect_pos);
    circle_rect_test(mid, cap.radius, rect_pos, rect_angle, half_ext, body_cap, body_rect, max_contacts);
}

// Capsule vs Polygon: find closest point on capsule segment to each polygon edge,
// take the closest, reduce to circle-poly test.
fn capsule_poly_test(
    cap_pos: vec2<f32>, cap_angle: f32, cap_si: u32,
    poly_pos: vec2<f32>, poly_angle: f32, poly_si: u32,
    body_cap: u32, body_poly: u32,
    max_contacts: u32,
) {
    let cap = capsules_2d[cap_si];
    let eps = capsule_endpoints_2d(cap_pos, cap_angle, cap.half_height);

    // Find closest point on capsule segment to polygon center
    let closest = closest_point_on_segment_2d(eps[0], eps[1], poly_pos);
    // Reduce to circle vs polygon
    circle_poly_test(closest, cap.radius, poly_pos, poly_angle, poly_si, body_cap, body_poly, max_contacts);
}

// ---------- Main dispatch ----------

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pi = gid.x;
    let num_pairs = atomicLoad(&pair_count);
    if pi >= num_pairs {
        return;
    }

    let pair = pairs[pi];
    let a = pair.a;
    let b = pair.b;

    let pos_a = bodies[a].position_inv_mass.xy;
    let pos_b = bodies[b].position_inv_mass.xy;
    let angle_a = bodies[a].position_inv_mass.z;
    let angle_b = bodies[b].position_inv_mass.z;
    let st_a = shape_infos[a].shape_type;
    let st_b = shape_infos[b].shape_type;
    let si_a = shape_infos[a].shape_index;
    let si_b = shape_infos[b].shape_index;

    let max_contacts = params.counts.x * 8u;

    // Sort shape types so s1 <= s2 for consistent dispatch
    var s1 = st_a;
    var s2 = st_b;
    var p1 = pos_a;
    var p2 = pos_b;
    var a1 = angle_a;
    var a2 = angle_b;
    var i1 = si_a;
    var i2 = si_b;
    var b1 = a;
    var b2 = b;

    if s1 > s2 {
        s1 = st_b;
        s2 = st_a;
        p1 = pos_b;
        p2 = pos_a;
        a1 = angle_b;
        a2 = angle_a;
        i1 = si_b;
        i2 = si_a;
        b1 = b;
        b2 = a;
    }

    // CIRCLE(0) pairs
    if s1 == SHAPE_CIRCLE && s2 == SHAPE_CIRCLE {
        let ra = circles[i1].radius;
        let rb = circles[i2].radius;
        circle_circle_test(p1, ra, p2, rb, b1, b2, max_contacts);
    } else if s1 == SHAPE_CIRCLE && s2 == SHAPE_RECT {
        let ra = circles[i1].radius;
        let hb = rects[i2].half_extents.xy;
        circle_rect_test(p1, ra, p2, a2, hb, b1, b2, max_contacts);
    } else if s1 == SHAPE_CIRCLE && s2 == SHAPE_CONVEX_POLYGON {
        let ra = circles[i1].radius;
        circle_poly_test(p1, ra, p2, a2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_CIRCLE && s2 == SHAPE_CAPSULE {
        let ra = circles[i1].radius;
        circle_capsule_test(p1, ra, p2, a2, i2, b1, b2, max_contacts);
    }
    // RECT(1) pairs
    else if s1 == SHAPE_RECT && s2 == SHAPE_RECT {
        rect_rect_test(p1, a1, i1, p2, a2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_RECT && s2 == SHAPE_CONVEX_POLYGON {
        rect_poly_test(p1, a1, i1, p2, a2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_RECT && s2 == SHAPE_CAPSULE {
        let ha = rects[i1].half_extents.xy;
        rect_capsule_test(p1, a1, ha, p2, a2, i2, b1, b2, max_contacts);
    }
    // POLYGON(2) pairs
    else if s1 == SHAPE_CONVEX_POLYGON && s2 == SHAPE_CONVEX_POLYGON {
        poly_poly_test(p1, a1, i1, p2, a2, i2, b1, b2, max_contacts);
    } else if s1 == SHAPE_CONVEX_POLYGON && s2 == SHAPE_CAPSULE {
        capsule_poly_test(p2, a2, i2, p1, a1, i1, b2, b1, max_contacts);
    }
    // CAPSULE(3) pairs
    else if s1 == SHAPE_CAPSULE && s2 == SHAPE_CAPSULE {
        capsule_capsule_test(p1, a1, i1, p2, a2, i2, b1, b2, max_contacts);
    }
}
"#;
