pub const EXTRACT_RENDER_TRANSFORMS_WGSL: &str = r#"
struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

struct RenderTransform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
    quality: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       bodies:     array<Body>;
@group(0) @binding(1) var<storage, read_write> transforms: array<RenderTransform>;
@group(0) @binding(2) var<uniform>             params:     SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.counts.x {
        return;
    }

    let body = bodies[idx];
    transforms[idx].position = body.position_inv_mass;
    transforms[idx].rotation = body.orientation;
}
"#;
