/// WGSL source for GPU narrowphase contact generation (2D).
///
/// For each broadphase pair, detects collisions based on shape types
/// (circle-circle, circle-rect) and writes contacts to the output buffer.
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
    point:      vec4<f32>, // (x, y, depth, 0)
    normal:     vec4<f32>, // (nx, ny, 0, 0)
    body_a:     u32,
    body_b:     u32,
    feature_id: u32,
    _pad:       u32,
    lambda_n:   f32,
    lambda_t:   f32,
    penalty_k:  f32,
    _pad2:      f32,
};

struct SimParams2D {
    gravity:           vec4<f32>,
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
};

const SHAPE_CIRCLE: u32 = 0u;
const SHAPE_RECT:   u32 = 1u;

@group(0) @binding(0) var<storage, read>       bodies:        array<Body2D>;
@group(0) @binding(1) var<storage, read>       shape_infos:   array<ShapeInfo>;
@group(0) @binding(2) var<storage, read>       pairs:         array<Pair>;
@group(0) @binding(3) var<storage, read>       pair_count_in: array<u32>;
@group(0) @binding(4) var<storage, read>       circles:       array<CircleData>;
@group(0) @binding(5) var<storage, read>       rects:         array<RectDataGpu>;
@group(0) @binding(6) var<storage, read_write> contacts:      array<Contact2D>;
@group(0) @binding(7) var<storage, read_write> contact_count: atomic<u32>;
@group(0) @binding(8) var<uniform>             params:        SimParams2D;

fn emit_contact_2d(
    point: vec2<f32>,
    normal: vec2<f32>,
    depth: f32,
    body_a: u32,
    body_b: u32,
    max_contacts: u32,
) {
    let slot = atomicAdd(&contact_count, 1u);
    if slot >= max_contacts {
        return;
    }
    contacts[slot].point  = vec4<f32>(point.x, point.y, depth, 0.0);
    contacts[slot].normal = vec4<f32>(normal.x, normal.y, 0.0, 0.0);
    contacts[slot].body_a = body_a;
    contacts[slot].body_b = body_b;
    contacts[slot].feature_id = 0u;
    contacts[slot]._pad = 0u;
    contacts[slot].lambda_n = 0.0;
    contacts[slot].lambda_t = 0.0;
    contacts[slot].penalty_k = 1e4;
    contacts[slot]._pad2 = 0.0;
}

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
    let normal = diff / dist;
    let depth = dist - sum_r; // negative when penetrating
    let point = pos_a + normal * (radius_a + depth * 0.5);
    emit_contact_2d(point, normal, depth, body_a, body_b, max_contacts);
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
    // Negate normal: geometric normal points rect→circle (B→A), but convention is A→B
    let normal_ab = -normal_world;
    let contact_point = circle_pos + normal_ab * (radius + depth * 0.5);
    emit_contact_2d(contact_point, normal_ab, depth, body_circle, body_rect, max_contacts);
}

fn rect_rect_test(
    pos_a: vec2<f32>, angle_a: f32, he_a: vec2<f32>,
    pos_b: vec2<f32>, angle_b: f32, he_b: vec2<f32>,
    body_a: u32, body_b: u32,
    max_contacts: u32,
) {
    // SAT with 4 face-normal axes (2 per rect)
    let ca_a = cos(angle_a);
    let sa_a = sin(angle_a);
    let ca_b = cos(angle_b);
    let sa_b = sin(angle_b);

    let axis_a0 = vec2<f32>(ca_a, sa_a);  // local X of A
    let axis_a1 = vec2<f32>(-sa_a, ca_a); // local Y of A
    let axis_b0 = vec2<f32>(ca_b, sa_b);  // local X of B
    let axis_b1 = vec2<f32>(-sa_b, ca_b); // local Y of B

    let d = pos_b - pos_a;
    var min_depth = -1e30;
    var best_normal = vec2<f32>(0.0, 1.0);

    // Helper arrays for iteration
    let axes = array<vec2<f32>, 4>(axis_a0, axis_a1, axis_b0, axis_b1);
    let he_a_arr = array<f32, 2>(he_a.x, he_a.y);
    let he_b_arr = array<f32, 2>(he_b.x, he_b.y);

    // Test axes from A (indices 0, 1)
    for (var i = 0u; i < 2u; i = i + 1u) {
        let axis = axes[i];
        let proj_a = he_a_arr[i];
        let proj_b = abs(dot(axis_b0, axis)) * he_b_arr[0]
                   + abs(dot(axis_b1, axis)) * he_b_arr[1];
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

    // Test axes from B (indices 2, 3 -> mapped to he_b index 0, 1)
    for (var i = 0u; i < 2u; i = i + 1u) {
        let axis = axes[i + 2u];
        let proj_a = abs(dot(axis_a0, axis)) * he_a_arr[0]
                   + abs(dot(axis_a1, axis)) * he_a_arr[1];
        let proj_b = he_b_arr[i];
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

    // Contact point: midpoint adjusted by normal and depth
    let contact_point = (pos_a + pos_b) * 0.5 + best_normal * min_depth * 0.5;
    emit_contact_2d(contact_point, best_normal, min_depth, body_a, body_b, max_contacts);
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

    let pos_a = bodies[a].position_inv_mass.xy;
    let pos_b = bodies[b].position_inv_mass.xy;
    let angle_a = bodies[a].position_inv_mass.z;
    let angle_b = bodies[b].position_inv_mass.z;
    let st_a = shape_infos[a].shape_type;
    let st_b = shape_infos[b].shape_type;

    let max_contacts = params.num_bodies * 8u;

    if st_a == SHAPE_CIRCLE && st_b == SHAPE_CIRCLE {
        let ra = circles[shape_infos[a].shape_index].radius;
        let rb = circles[shape_infos[b].shape_index].radius;
        circle_circle_test(pos_a, ra, pos_b, rb, a, b, max_contacts);
    } else if st_a == SHAPE_CIRCLE && st_b == SHAPE_RECT {
        let ra = circles[shape_infos[a].shape_index].radius;
        let hb = rects[shape_infos[b].shape_index].half_extents.xy;
        circle_rect_test(pos_a, ra, pos_b, angle_b, hb, a, b, max_contacts);
    } else if st_a == SHAPE_RECT && st_b == SHAPE_CIRCLE {
        let rb = circles[shape_infos[b].shape_index].radius;
        let ha = rects[shape_infos[a].shape_index].half_extents.xy;
        circle_rect_test(pos_b, rb, pos_a, angle_a, ha, b, a, max_contacts);
    } else if st_a == SHAPE_RECT && st_b == SHAPE_RECT {
        let ha = rects[shape_infos[a].shape_index].half_extents.xy;
        let hb = rects[shape_infos[b].shape_index].half_extents.xy;
        rect_rect_test(pos_a, angle_a, ha, pos_b, angle_b, hb, a, b, max_contacts);
    }
}
"#;
