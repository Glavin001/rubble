struct RouteParams_std140_0
{
    @align(16) island_count_0 : u32,
    @align(4) impactor_count_0 : u32,
    @align(8) engine_base_0 : u32,
    @align(4) pad0_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) speed_0 : f32,
    @align(8) margin_0 : f32,
    @align(4) pad1_0 : f32,
};

@binding(0) @group(0) var<uniform> params_0 : RouteParams_std140_0;
struct Island_std430_0
{
    @align(16) range_0 : vec4<u32>,
    @align(16) info_0 : vec4<u32>,
    @align(16) com_0 : vec4<f32>,
    @align(16) inertia0_0 : vec4<f32>,
    @align(16) inertia1_0 : vec4<f32>,
    @align(16) inertia2_0 : vec4<f32>,
    @align(16) inv0_0 : vec4<f32>,
    @align(16) inv1_0 : vec4<f32>,
    @align(16) inv2_0 : vec4<f32>,
    @align(16) wcom_0 : vec4<f32>,
    @align(16) winv0_0 : vec4<f32>,
    @align(16) winv1_0 : vec4<f32>,
    @align(16) winv2_0 : vec4<f32>,
    @align(16) rotation_0 : vec4<f32>,
    @align(16) position_0 : vec4<f32>,
    @align(16) position_err_0 : vec4<f32>,
    @align(16) velocity_0 : vec4<f32>,
    @align(16) velocity_err_0 : vec4<f32>,
    @align(16) angular_velocity_0 : vec4<f32>,
    @align(16) done_0 : u32,
    @align(4) coverage_0 : u32,
    @align(8) wide_groups_0 : u32,
    @align(4) done_pad_0 : u32,
    @align(16) probes_0 : vec4<u32>,
    @align(16) energy_0 : vec4<f32>,
    @align(16) rotation_err_0 : vec4<f32>,
    @align(16) momentum_0 : vec4<f32>,
    @align(16) momentum_err_0 : vec4<f32>,
};

@binding(1) @group(0) var<storage, read> islands_0 : array<Island_std430_0>;

@binding(4) @group(0) var<storage, read> ctab_0 : array<vec4<f32>>;

struct EngineBody_std430_0
{
    @align(16) pos_inv_mass_0 : vec4<f32>,
    @align(16) rot_0 : vec4<f32>,
    @align(16) inv_inertia_flags_0 : vec4<f32>,
    @align(16) vel_sleep_0 : vec4<f32>,
    @align(16) ang_vel_gen_0 : vec4<f32>,
    @align(16) initial_pos_0 : vec4<f32>,
    @align(16) initial_rot_0 : vec4<f32>,
};

@binding(5) @group(0) var<storage, read> bodies_0 : array<EngineBody_std430_0>;

struct ChunkStatic_std430_0
{
    @align(16) center_0 : vec4<f32>,
    @align(16) inertia0_1 : vec4<f32>,
    @align(16) inertia1_1 : vec4<f32>,
    @align(16) inertia2_1 : vec4<f32>,
    @align(16) inv0_1 : vec4<f32>,
    @align(16) inv1_1 : vec4<f32>,
    @align(16) inv2_1 : vec4<f32>,
    @align(16) scale_0 : vec4<f32>,
    @align(16) info_1 : vec4<u32>,
    @align(16) load_range_0 : vec4<u32>,
    @align(16) half_0 : vec4<f32>,
    @align(16) crot0_0 : vec4<f32>,
    @align(16) crot1_0 : vec4<f32>,
    @align(16) crot2_0 : vec4<f32>,
    @align(16) cmat_0 : vec4<f32>,
    @align(16) cinfo_0 : vec4<u32>,
};

@binding(2) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

@binding(3) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read> impactors_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read_write> route_0 : array<u32>;

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

fn rotate_0( q_1 : Quat_0,  v_0 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_1.x_0, q_1.y_0, q_1.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_0) * vec3<f32>(2.0f);
    return v_0 + t_0 * vec3<f32>(q_1.w_0) + cross(qv_0, t_0);
}

fn quat_mul_0( a_0 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_1 : Quat_0;
    r_1.w_0 = a_0.w_0 * o_0.w_0 - a_0.x_0 * o_0.x_0 - a_0.y_0 * o_0.y_0 - a_0.z_0 * o_0.z_0;
    r_1.x_0 = a_0.w_0 * o_0.x_0 + a_0.x_0 * o_0.w_0 + a_0.y_0 * o_0.z_0 - a_0.z_0 * o_0.y_0;
    r_1.y_0 = a_0.w_0 * o_0.y_0 - a_0.x_0 * o_0.z_0 + a_0.y_0 * o_0.w_0 + a_0.z_0 * o_0.x_0;
    r_1.z_0 = a_0.w_0 * o_0.z_0 + a_0.x_0 * o_0.y_0 - a_0.y_0 * o_0.x_0 + a_0.z_0 * o_0.w_0;
    return r_1;
}

fn normalized_0( q_2 : Quat_0) -> Quat_0
{
    var _S1 : f32 = q_2.w_0;
    var _S2 : f32 = q_2.x_0;
    var _S3 : f32 = q_2.y_0;
    var _S4 : f32 = q_2.z_0;
    var n_0 : f32 = sqrt(_S1 * _S1 + _S2 * _S2 + _S3 * _S3 + _S4 * _S4);
    var r_2 : Quat_0;
    r_2.w_0 = q_2.w_0 / n_0;
    r_2.x_0 = q_2.x_0 / n_0;
    r_2.y_0 = q_2.y_0 / n_0;
    r_2.z_0 = q_2.z_0 / n_0;
    return r_2;
}

fn to_zup_0( v_1 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_1.x, - v_1.z, v_1.y);
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

var<workgroup> g_lo_0 : array<vec3<f32>, i32(64)>;

var<workgroup> g_hi_0 : array<vec3<f32>, i32(64)>;

@compute
@workgroup_size(64, 1, 1)
fn route_islands(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var i_0 : u32 = group_0.x;
    var t_1 : u32 = thread_0.x;
    var overlap_0 : bool;
    if(i_0 >= (params_0.island_count_0))
    {
        overlap_0 = true;
    }
    else
    {
        overlap_0 = (params_0.impactor_count_0) == u32(0);
    }
    if(overlap_0)
    {
        return;
    }
    var _S5 : vec4<u32> = islands_0[i_0].range_0;
    if((islands_0[i_0].range_0.z) >= (islands_0[i_0].range_0.w))
    {
        return;
    }
    var _S6 : u32 = u32(3) * i_0;
    var body_0 : u32 = (bitcast<u32>((ctab_0[params_0.engine_base_0 + _S6].x)));
    var q_3 : Quat_0 = quat_of_0(islands_0[i_0].rotation_0);
    var origin_0 : vec3<f32> = islands_0[i_0].position_0.xyz;
    var _S7 : vec3<f32> = origin_0 + rotate_0(q_3, islands_0[i_0].com_0.xyz);
    var _S8 : vec3<f32> = islands_0[i_0].velocity_0.xyz;
    var _S9 : vec3<f32> = islands_0[i_0].angular_velocity_0.xyz;
    var origin_1 : vec3<f32>;
    var centre_0 : vec3<f32>;
    var v_2 : vec3<f32>;
    var w_1 : vec3<f32>;
    var q_4 : Quat_0;
    if(body_0 != u32(4294967295))
    {
        var p_inv_0 : Quat_0 = quat_of_0(ctab_0[params_0.engine_base_0 + _S6 + u32(1)]);
        var com_1 : vec3<f32> = ctab_0[params_0.engine_base_0 + _S6 + u32(2)].xyz;
        var b_0 : EngineBody_std430_0 = bodies_0[body_0];
        var a_inv_0 : Quat_0;
        a_inv_0.w_0 = 0.70710676908493042f;
        a_inv_0.x_0 = 0.70710676908493042f;
        a_inv_0.y_0 = 0.0f;
        a_inv_0.z_0 = 0.0f;
        var q_5 : Quat_0 = normalized_0(quat_mul_0(quat_mul_0(a_inv_0, quat_of_0(b_0.rot_0)), p_inv_0));
        var centre_1 : vec3<f32> = to_zup_0(b_0.pos_inv_mass_0.xyz);
        var _S10 : vec3<f32> = to_zup_0(b_0.vel_sleep_0.xyz);
        var _S11 : vec3<f32> = to_zup_0(b_0.ang_vel_gen_0.xyz);
        origin_1 = centre_1 - rotate_0(q_5, com_1);
        q_4 = q_5;
        centre_0 = centre_1;
        v_2 = _S10;
        w_1 = _S11;
    }
    else
    {
        origin_1 = origin_0;
        q_4 = q_3;
        centre_0 = _S7;
        v_2 = _S8;
        w_1 = _S9;
    }
    var inf_0 : f32 = float_bits_0(u32(2139095040));
    var _S12 : vec3<f32> = vec3<f32>(- inf_0);
    var _S13 : u32 = _S5.x + t_1;
    var lo_0 : vec3<f32> = vec3<f32>(inf_0);
    var hi_0 : vec3<f32> = _S12;
    var c_0 : u32 = _S13;
    loop
    {
        if(c_0 < (_S5.y))
        {
        }
        else
        {
            break;
        }
        var p_0 : vec3<f32> = origin_1 + rotate_0(q_4, chunks_0[c_0].center_0.xyz + state_0[u32(4) * c_0].xyz);
        var _S14 : vec3<f32> = vec3<f32>(chunks_0[c_0].half_0.w);
        var _S15 : vec3<f32> = min(lo_0, p_0 - _S14);
        var _S16 : vec3<f32> = max(hi_0, p_0 + _S14);
        var c_1 : u32 = c_0 + u32(64);
        lo_0 = _S15;
        hi_0 = _S16;
        c_0 = c_1;
    }
    g_lo_0[t_1] = lo_0;
    g_hi_0[t_1] = hi_0;
    workgroupBarrier();
    var s_0 : u32 = u32(32);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_1 < s_0)
        {
            var _S17 : u32 = t_1 + s_0;
            g_lo_0[t_1] = min(g_lo_0[t_1], g_lo_0[_S17]);
            g_hi_0[t_1] = max(g_hi_0[t_1], g_hi_0[_S17]);
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    if(t_1 != u32(0))
    {
        return;
    }
    var dt_1 : f32 = params_0.dt_0;
    var _S18 : vec3<f32> = vec3<f32>(params_0.dt_0);
    var _S19 : f32 = length(w_1) * 2.0f * length(max(abs(g_hi_0[i32(0)] - centre_0), abs(centre_0 - g_lo_0[i32(0)])));
    var _S20 : vec3<f32> = vec3<f32>(params_0.margin_0);
    var sweep_0 : vec3<f32> = abs(v_2) * _S18 + vec3<f32>((_S19 * params_0.dt_0)) + _S20;
    var _S21 : vec3<f32> = g_lo_0[i32(0)] - sweep_0;
    var _S22 : vec3<f32> = g_hi_0[i32(0)] + sweep_0;
    var k_0 : u32 = u32(0);
    var found_0 : u32 = u32(0);
    loop
    {
        if(k_0 < (params_0.impactor_count_0))
        {
        }
        else
        {
            break;
        }
        var imp_0 : vec4<f32> = impactors_0[k_0];
        var b_1 : EngineBody_std430_0 = bodies_0[(bitcast<u32>((imp_0.x)))];
        var c_2 : vec3<f32> = to_zup_0(b_1.pos_inv_mass_0.xyz);
        var vk_0 : vec3<f32> = to_zup_0(b_1.vel_sleep_0.xyz);
        var r_3 : f32 = imp_0.y;
        var _S23 : f32 = length(to_zup_0(b_1.ang_vel_gen_0.xyz)) * r_3;
        var ext_0 : vec3<f32> = vec3<f32>(r_3) + abs(vk_0) * _S18 + vec3<f32>((_S23 * dt_1)) + _S20;
        if((all(((c_2 - ext_0) <= _S22))))
        {
            overlap_0 = (all((_S21 <= (c_2 + ext_0))));
        }
        else
        {
            overlap_0 = false;
        }
        var closing_0 : f32 = length(vk_0 - v_2) + _S23 + _S19;
        var _S24 : bool;
        if(overlap_0)
        {
            _S24 = closing_0 > (params_0.speed_0);
        }
        else
        {
            _S24 = false;
        }
        if(_S24)
        {
            found_0 = found_0 + u32(1);
        }
        k_0 = k_0 + u32(1);
    }
    if(found_0 > u32(0))
    {
        route_0[i32(0)] = u32(1);
    }
    return;
}

