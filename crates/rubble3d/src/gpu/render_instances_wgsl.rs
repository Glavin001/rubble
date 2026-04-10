pub const BUILD_RENDER_INSTANCES_WGSL: &str = r#"
struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

struct RenderBody {
    scale: vec4<f32>,
    color: vec4<f32>,
    shape_data: vec4<u32>,
};

struct InstanceData {
    model: mat4x4<f32>,
    color: vec4<f32>,
};

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
    quality: vec4<f32>,
};

const RENDER_SHAPE_SPHERE: u32 = 0u;
const RENDER_SHAPE_BOX: u32 = 1u;
const RENDER_SHAPE_CAPSULE: u32 = 2u;
const RENDER_SHAPE_HIDDEN: u32 = 0xffffffffu;

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read> render_bodies: array<RenderBody>;
@group(0) @binding(2) var<storage, read_write> sphere_instances: array<InstanceData>;
@group(0) @binding(3) var<storage, read_write> cube_instances: array<InstanceData>;
@group(0) @binding(4) var<storage, read_write> capsule_instances: array<InstanceData>;
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

@group(0) @binding(5) var<storage, read_write> draw_args: array<DrawIndexedIndirectArgs, 3>;
@group(0) @binding(6) var<uniform> params: SimParams;

fn model_matrix(position: vec3<f32>, q: vec4<f32>, scale: vec3<f32>) -> mat4x4<f32> {
    let xx = q.x * q.x;
    let yy = q.y * q.y;
    let zz = q.z * q.z;
    let xy = q.x * q.y;
    let xz = q.x * q.z;
    let yz = q.y * q.z;
    let wx = q.w * q.x;
    let wy = q.w * q.y;
    let wz = q.w * q.z;

    let c0 = vec4<f32>(
        (1.0 - 2.0 * (yy + zz)) * scale.x,
        (2.0 * (xy + wz)) * scale.x,
        (2.0 * (xz - wy)) * scale.x,
        0.0,
    );
    let c1 = vec4<f32>(
        (2.0 * (xy - wz)) * scale.y,
        (1.0 - 2.0 * (xx + zz)) * scale.y,
        (2.0 * (yz + wx)) * scale.y,
        0.0,
    );
    let c2 = vec4<f32>(
        (2.0 * (xz + wy)) * scale.z,
        (2.0 * (yz - wx)) * scale.z,
        (1.0 - 2.0 * (xx + yy)) * scale.z,
        0.0,
    );
    let c3 = vec4<f32>(position, 1.0);
    return mat4x4<f32>(c0, c1, c2, c3);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.counts.x {
        return;
    }

    let render_meta = render_bodies[idx];
    let shape = render_meta.shape_data.x;
    if shape == RENDER_SHAPE_HIDDEN {
        return;
    }

    let body = bodies[idx];
    let instance = InstanceData(
        model_matrix(body.position_inv_mass.xyz, body.orientation, render_meta.scale.xyz),
        render_meta.color,
    );

    switch shape {
        case RENDER_SHAPE_SPHERE: {
            let out_idx = atomicAdd(&draw_args[0].instance_count, 1u);
            sphere_instances[out_idx] = instance;
        }
        case RENDER_SHAPE_BOX: {
            let out_idx = atomicAdd(&draw_args[1].instance_count, 1u);
            cube_instances[out_idx] = instance;
        }
        case RENDER_SHAPE_CAPSULE: {
            let out_idx = atomicAdd(&draw_args[2].instance_count, 1u);
            capsule_instances[out_idx] = instance;
        }
        default: {}
    }
}
"#;
