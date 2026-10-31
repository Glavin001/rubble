struct ShapeParams_std140_0
{
    @align(16) collider_count_0 : u32,
    @align(4) pad0_0 : u32,
    @align(8) pad1_0 : u32,
    @align(4) pad2_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : ShapeParams_std140_0;
@binding(1) @group(0) var<storage, read> rest_0 : array<vec4<f32>>;

@binding(2) @group(0) var<storage, read> collider_info_0 : array<vec4<u32>>;

@binding(3) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read_write> colliders_0 : array<vec4<f32>>;

struct Quat_0
{
     w_0 : f32,
     x_0 : f32,
     y_0 : f32,
     z_0 : f32,
};

fn quat_of_0( q_0 : vec4<f32>) -> Quat_0
{
    var r_0 : Quat_0;
    r_0.x_0 = q_0.x;
    r_0.y_0 = q_0.y;
    r_0.z_0 = q_0.z;
    r_0.w_0 = q_0.w;
    return r_0;
}

fn safe_normalize_0( v_0 : vec3<f32>) -> vec3<f32>
{
    var n_0 : f32 = length(v_0);
    var _S1 : vec3<f32>;
    if(n_0 > 1.00000000317107685e-30f)
    {
        _S1 = v_0 / vec3<f32>(n_0);
    }
    else
    {
        _S1 = vec3<f32>(0.0f);
    }
    return _S1;
}

fn from_axis_angle_0( axis_0 : vec3<f32>,  angle_0 : f32) -> Quat_0
{
    var a_0 : vec3<f32> = safe_normalize_0(axis_0);
    var _S2 : f32 = 0.5f * angle_0;
    var s_0 : f32 = sin(_S2);
    var q_1 : Quat_0;
    q_1.w_0 = cos(_S2);
    q_1.x_0 = a_0.x * s_0;
    q_1.y_0 = a_0.y * s_0;
    q_1.z_0 = a_0.z * s_0;
    return q_1;
}

fn quat_mul_0( a_1 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_1 : Quat_0;
    r_1.w_0 = a_1.w_0 * o_0.w_0 - a_1.x_0 * o_0.x_0 - a_1.y_0 * o_0.y_0 - a_1.z_0 * o_0.z_0;
    r_1.x_0 = a_1.w_0 * o_0.x_0 + a_1.x_0 * o_0.w_0 + a_1.y_0 * o_0.z_0 - a_1.z_0 * o_0.y_0;
    r_1.y_0 = a_1.w_0 * o_0.y_0 - a_1.x_0 * o_0.z_0 + a_1.y_0 * o_0.w_0 + a_1.z_0 * o_0.x_0;
    r_1.z_0 = a_1.w_0 * o_0.z_0 + a_1.x_0 * o_0.y_0 - a_1.y_0 * o_0.x_0 + a_1.z_0 * o_0.w_0;
    return r_1;
}

fn conj_of_0( q_2 : Quat_0) -> Quat_0
{
    var c_0 : Quat_0;
    c_0.w_0 = q_2.w_0;
    c_0.x_0 = - q_2.x_0;
    c_0.y_0 = - q_2.y_0;
    c_0.z_0 = - q_2.z_0;
    return c_0;
}

fn rotate_0( q_3 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_3.x_0, q_3.y_0, q_3.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_1) * vec3<f32>(2.0f);
    return v_1 + t_0 * vec3<f32>(q_3.w_0) + cross(qv_0, t_0);
}

fn normalized_0( q_4 : Quat_0) -> Quat_0
{
    var _S3 : f32 = q_4.w_0;
    var _S4 : f32 = q_4.x_0;
    var _S5 : f32 = q_4.y_0;
    var _S6 : f32 = q_4.z_0;
    var n_1 : f32 = sqrt(_S3 * _S3 + _S4 * _S4 + _S5 * _S5 + _S6 * _S6);
    var r_2 : Quat_0;
    r_2.w_0 = q_4.w_0 / n_1;
    r_2.x_0 = q_4.x_0 / n_1;
    r_2.y_0 = q_4.y_0 / n_1;
    r_2.z_0 = q_4.z_0 / n_1;
    return r_2;
}

fn quat_vec_0( q_5 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_5.x_0, q_5.y_0, q_5.z_0, q_5.w_0);
}

@compute
@workgroup_size(64, 1, 1)
fn shapes_deform(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var i_0 : u32 = id_0.x;
    if(i_0 >= (params_0.collider_count_0))
    {
        return;
    }
    var _S7 : u32 = u32(4) * i_0;
    var cb_0 : vec4<f32> = rest_0[_S7];
    var info_0 : vec4<u32> = collider_info_0[u32(2) * i_0];
    var _S8 : bool;
    if((cb_0.w) == 0.0f)
    {
        _S8 = true;
    }
    else
    {
        _S8 = (info_0.z) != u32(2);
    }
    if(_S8)
    {
        _S8 = true;
    }
    else
    {
        _S8 = (info_0.x) == u32(4294967295);
    }
    if(_S8)
    {
        return;
    }
    var _S9 : u32 = u32(4) * info_0.x;
    var th_0 : vec3<f32> = state_0[_S9 + u32(1)].xyz;
    var _S10 : u32 = _S7 + u32(1);
    var to_body_0 : Quat_0 = quat_of_0(rest_0[_S10]);
    var turn_0 : Quat_0 = quat_mul_0(quat_mul_0(to_body_0, from_axis_angle_0(th_0, length(th_0))), conj_of_0(to_body_0));
    var _S11 : u32 = _S7 + u32(2);
    var r0_0 : Quat_0 = quat_of_0(rest_0[_S7 + u32(3)]);
    var _S12 : vec3<f32> = cb_0.xyz;
    colliders_0[_S10] = vec4<f32>(_S12 + rotate_0(to_body_0, state_0[_S9].xyz) + rotate_0(turn_0, rest_0[_S11].xyz - _S12), colliders_0[_S10].w);
    colliders_0[_S11] = quat_vec_0(normalized_0(quat_mul_0(turn_0, r0_0)));
    return;
}

