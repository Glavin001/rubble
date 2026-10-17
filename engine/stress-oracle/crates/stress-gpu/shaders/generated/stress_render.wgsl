struct RenderChunk_std430_0
{
    @align(16) center_0 : vec4<f32>,
    @align(16) half_0 : vec4<f32>,
    @align(16) rot0_0 : vec4<f32>,
    @align(16) rot1_0 : vec4<f32>,
    @align(16) rot2_0 : vec4<f32>,
    @align(16) pose_pos_0 : vec4<f32>,
    @align(16) pose_rot0_0 : vec4<f32>,
    @align(16) pose_rot1_0 : vec4<f32>,
    @align(16) pose_rot2_0 : vec4<f32>,
};

@binding(1) @group(0) var<storage, read> render_chunks_0 : array<RenderChunk_std430_0>;

@binding(2) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

struct Camera_std140_0
{
    @align(16) view_proj_0 : array<vec4<f32>, i32(4)>,
    @align(16) light_0 : vec4<f32>,
    @align(16) stress_max_0 : f32,
    @align(4) exaggerate_0 : f32,
    @align(8) pad0_0 : f32,
    @align(4) pad1_0 : f32,
};

@binding(0) @group(0) var<uniform> camera_0 : Camera_std140_0;
fn apply_rows_0( r0_0 : vec4<f32>,  r1_0 : vec4<f32>,  r2_0 : vec4<f32>,  x_0 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(r0_0.xyz, x_0), dot(r1_0.xyz, x_0), dot(r2_0.xyz, x_0));
}

fn rotate_by_0( theta_0 : vec3<f32>,  x_1 : vec3<f32>) -> vec3<f32>
{
    var angle_0 : f32 = length(theta_0);
    if(angle_0 < 9.999999960041972e-13f)
    {
        return x_1 + cross(theta_0, x_1);
    }
    var k_0 : vec3<f32> = theta_0 / vec3<f32>(angle_0);
    var _S1 : f32 = cos(angle_0);
    return x_1 * vec3<f32>(_S1) + cross(k_0, x_1) * vec3<f32>(sin(angle_0)) + k_0 * vec3<f32>(dot(k_0, x_1)) * vec3<f32>((1.0f - _S1));
}

fn ramp_0( t_0 : f32) -> vec3<f32>
{
    const c0_0 : vec3<f32> = vec3<f32>(0.15000000596046448f, 0.25f, 0.85000002384185791f);
    const c1_0 : vec3<f32> = vec3<f32>(0.10000000149011612f, 0.75f, 0.85000002384185791f);
    const c2_0 : vec3<f32> = vec3<f32>(0.25f, 0.80000001192092896f, 0.30000001192092896f);
    const c3_0 : vec3<f32> = vec3<f32>(0.94999998807907104f, 0.85000002384185791f, 0.20000000298023224f);
    const c4_0 : vec3<f32> = vec3<f32>(0.89999997615814209f, 0.20000000298023224f, 0.15000000596046448f);
    var s_0 : f32 = saturate(t_0) * 4.0f;
    if(s_0 < 1.0f)
    {
        return mix(c0_0, c1_0, vec3<f32>(s_0));
    }
    if(s_0 < 2.0f)
    {
        return mix(c1_0, c2_0, vec3<f32>((s_0 - 1.0f)));
    }
    if(s_0 < 3.0f)
    {
        return mix(c2_0, c3_0, vec3<f32>((s_0 - 2.0f)));
    }
    return mix(c3_0, c4_0, vec3<f32>((s_0 - 3.0f)));
}

struct VertexOut_0
{
    @builtin(position) position_0 : vec4<f32>,
    @location(0) normal_0 : vec3<f32>,
    @location(1) color_0 : vec3<f32>,
};

@vertex
fn stress_vs(@builtin(vertex_index) vertex_0 : u32, @builtin(instance_index) instance_0 : u32) -> VertexOut_0
{
    var face_0 : u32 = vertex_0 / u32(6);
    var corner_0 : u32 = vertex_0 % u32(6);
    var axis_0 : u32 = face_0 / u32(2);
    var sign_0 : f32;
    if((face_0 % u32(2)) == u32(0))
    {
        sign_0 = 1.0f;
    }
    else
    {
        sign_0 = -1.0f;
    }
    const _S2 : vec2<f32> = vec2<f32>(-1.0f, -1.0f);
    const _S3 : vec2<f32> = vec2<f32>(1.0f, 1.0f);
    var quad_0 : array<vec2<f32>, i32(6)> = array<vec2<f32>, i32(6)>( _S2, vec2<f32>(1.0f, -1.0f), _S3, _S2, _S3, vec2<f32>(-1.0f, 1.0f) );
    var st_0 : vec2<f32> = quad_0[corner_0];
    if(sign_0 < 0.0f)
    {
        st_0[i32(0)] = - st_0.x;
    }
    var p_0 : vec3<f32>;
    var n_0 : vec3<f32>;
    if(axis_0 == u32(0))
    {
        var _S4 : vec3<f32> = vec3<f32>(sign_0, 0.0f, 0.0f);
        p_0 = vec3<f32>(sign_0, st_0.x, st_0.y);
        n_0 = _S4;
    }
    else
    {
        if(axis_0 == u32(1))
        {
            var _S5 : vec3<f32> = vec3<f32>(0.0f, sign_0, 0.0f);
            p_0 = vec3<f32>(st_0.y, sign_0, st_0.x);
            n_0 = _S5;
        }
        else
        {
            var _S6 : vec3<f32> = vec3<f32>(0.0f, 0.0f, sign_0);
            p_0 = vec3<f32>(st_0.x, st_0.y, sign_0);
            n_0 = _S6;
        }
    }
    var _S7 : u32 = u32(4) * instance_0;
    var u_0 : vec4<f32> = state_0[_S7];
    var theta_1 : vec3<f32> = state_0[_S7 + u32(1)].xyz * vec3<f32>(camera_0.exaggerate_0);
    var normal_body_0 : vec3<f32> = rotate_by_0(theta_1, apply_rows_0(render_chunks_0[instance_0].rot0_0, render_chunks_0[instance_0].rot1_0, render_chunks_0[instance_0].rot2_0, n_0));
    var world_0 : vec3<f32> = render_chunks_0[instance_0].pose_pos_0.xyz + apply_rows_0(render_chunks_0[instance_0].pose_rot0_0, render_chunks_0[instance_0].pose_rot1_0, render_chunks_0[instance_0].pose_rot2_0, render_chunks_0[instance_0].center_0.xyz + u_0.xyz * vec3<f32>(camera_0.exaggerate_0) + rotate_by_0(theta_1, apply_rows_0(render_chunks_0[instance_0].rot0_0, render_chunks_0[instance_0].rot1_0, render_chunks_0[instance_0].rot2_0, p_0 * render_chunks_0[instance_0].half_0.xyz)));
    var o_0 : VertexOut_0;
    o_0.position_0 = camera_0.view_proj_0[i32(0)] * vec4<f32>(world_0.x) + camera_0.view_proj_0[i32(1)] * vec4<f32>(world_0.y) + camera_0.view_proj_0[i32(2)] * vec4<f32>(world_0.z) + camera_0.view_proj_0[i32(3)];
    o_0.normal_0 = apply_rows_0(render_chunks_0[instance_0].pose_rot0_0, render_chunks_0[instance_0].pose_rot1_0, render_chunks_0[instance_0].pose_rot2_0, normal_body_0);
    o_0.color_0 = ramp_0(u_0.w / camera_0.stress_max_0);
    return o_0;
}

struct pixelOutput_0
{
    @location(0) output_0 : vec4<f32>,
};

struct pixelInput_0
{
    @location(0) normal_1 : vec3<f32>,
    @location(1) color_1 : vec3<f32>,
};

@fragment
fn stress_fs( _S8 : pixelInput_0, @builtin(position) position_1 : vec4<f32>) -> pixelOutput_0
{
    var _S9 : pixelOutput_0 = pixelOutput_0( vec4<f32>(_S8.color_1 * vec3<f32>((0.34999999403953552f + 0.64999997615814209f * max(dot(normalize(_S8.normal_1), normalize(camera_0.light_0.xyz)), 0.0f))), 1.0f) );
    return _S9;
}

