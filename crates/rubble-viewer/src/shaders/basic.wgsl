struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    _pad0: f32,
    camera_pos: vec3<f32>,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct InstanceInput {
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) color: vec4<f32>,
};

fn normal_matrix(model_3x3: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = model_3x3[0];
    let c1 = model_3x3[1];
    let c2 = model_3x3[2];
    let inv_c0 = cross(c1, c2);
    let inv_c1 = cross(c2, c0);
    let inv_c2 = cross(c0, c1);
    let det = dot(c0, inv_c0);
    let safe_det = select(1.0, det, abs(det) > 1e-8);
    return transpose(mat3x3<f32>(inv_c0 / safe_det, inv_c1 / safe_det, inv_c2 / safe_det));
}

@vertex
fn vs_main(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(inst.model_0, inst.model_1, inst.model_2, inst.model_3);
    let world_pos = model * vec4<f32>(vert.position, 1.0);
    let normal_mat = normal_matrix(mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz));
    let world_normal = normalize(normal_mat * vert.normal);

    var out: VertexOutput;
    out.clip_position = u.view_proj * world_pos;
    out.world_normal = world_normal;
    out.world_pos = world_pos.xyz;
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(u.light_dir);
    let v = normalize(u.camera_pos - in.world_pos);
    let h = normalize(l + v);

    let ambient = 0.15;
    let diffuse = max(dot(n, l), 0.0) * 0.7;
    let spec = pow(max(dot(n, h), 0.0), 32.0) * 0.3;

    let lighting = ambient + diffuse + spec;
    let rgb = in.color.rgb * lighting;
    return vec4<f32>(rgb, in.color.a);
}

// Grid ground plane -- separate entry points so we can draw it as
// a full-screen quad without instance data.

struct GridVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_grid(@builtin(vertex_index) idx: u32) -> GridVertexOutput {
    // Two-triangle full-screen quad mapped to a large ground plane
    let size = 50.0;
    var positions = array<vec3<f32>, 6>(
        vec3<f32>(-size, 0.0, -size),
        vec3<f32>( size, 0.0, -size),
        vec3<f32>( size, 0.0,  size),
        vec3<f32>(-size, 0.0, -size),
        vec3<f32>( size, 0.0,  size),
        vec3<f32>(-size, 0.0,  size),
    );
    let pos = positions[idx];
    var out: GridVertexOutput;
    out.clip_position = u.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    return out;
}

@fragment
fn fs_grid(in: GridVertexOutput) -> @location(0) vec4<f32> {
    let grid_spacing = 1.0;
    let gx = abs(fract(in.world_pos.x / grid_spacing + 0.5) - 0.5);
    let gz = abs(fract(in.world_pos.z / grid_spacing + 0.5) - 0.5);
    let line = min(gx, gz);
    let thickness = 0.02;
    let edge = smoothstep(0.0, thickness, line);
    let grid_color = vec3<f32>(0.3, 0.3, 0.35);
    let bg_color = vec3<f32>(0.12, 0.12, 0.14);
    let color = mix(grid_color, bg_color, edge);
    let dist = length(in.world_pos.xz);
    let fade = 1.0 - smoothstep(20.0, 50.0, dist);
    return vec4<f32>(color, fade);
}
