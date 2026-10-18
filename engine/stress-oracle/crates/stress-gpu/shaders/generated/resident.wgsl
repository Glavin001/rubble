struct ResidentParams_std140_0
{
    @align(16) chunk_count_0 : u32,
    @align(4) island_count_0 : u32,
    @align(8) sload_base_0 : u32,
    @align(4) rml_0 : u32,
    @align(16) gravity_0 : vec4<f32>,
    @align(16) dt_0 : f32,
    @align(4) props_base_0 : u32,
    @align(8) engine_base_0 : u32,
    @align(4) eload_base_0 : u32,
    @align(16) pulse_base_0 : u32,
    @align(4) pulse_count_base_0 : u32,
    @align(8) pulse_cap_0 : u32,
    @align(4) status_base_0 : u32,
    @align(16) impact_peak_0 : u32,
    @align(4) pad0_0 : u32,
    @align(8) pad1_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : ResidentParams_std140_0;
@binding(7) @group(0) var<storage, read_write> ctab_0 : array<vec4<f32>>;

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

@binding(8) @group(0) var<storage, read> start_bodies_0 : array<EngineBody_std430_0>;

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

@binding(3) @group(0) var<storage, read_write> islands_0 : array<Island_std430_0>;

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

@binding(4) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

@binding(5) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

@binding(2) @group(0) var<storage, read> sort_0 : array<u32>;

struct Record_std430_0
{
    @align(16) ids_0 : vec4<u32>,
    @align(16) point_0 : vec4<f32>,
    @align(16) impulse_0 : vec4<f32>,
    @align(16) other_0 : vec4<f32>,
    @align(16) key_0 : vec4<u32>,
};

@binding(1) @group(0) var<storage, read> manifolds_0 : array<Record_std430_0>;

@binding(6) @group(0) var<storage, read_write> scratch_0 : array<vec4<f32>>;

fn isinf_0( x_0 : f32) -> bool
{
    var _S1 : u32 = (bitcast<u32>((x_0)));
    var _S2 : u32 = (_S1 & (u32(8388607)));
    var _S3 : bool;
    if(((((_S1 >> (u32(23)))) & (u32(255)))) == u32(255))
    {
        _S3 = _S2 == u32(0);
    }
    else
    {
        _S3 = false;
    }
    return _S3;
}

struct Quat_0
{
     w_0 : f32,
     x_1 : f32,
     y_0 : f32,
     z_0 : f32,
};

fn quat_of_0( q_0 : vec4<f32>) -> Quat_0
{
    var r_0 : Quat_0;
    r_0.x_1 = q_0.x;
    r_0.y_0 = q_0.y;
    r_0.z_0 = q_0.z;
    r_0.w_0 = q_0.w;
    return r_0;
}

fn quat_mul_0( a_0 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_1 : Quat_0;
    r_1.w_0 = a_0.w_0 * o_0.w_0 - a_0.x_1 * o_0.x_1 - a_0.y_0 * o_0.y_0 - a_0.z_0 * o_0.z_0;
    r_1.x_1 = a_0.w_0 * o_0.x_1 + a_0.x_1 * o_0.w_0 + a_0.y_0 * o_0.z_0 - a_0.z_0 * o_0.y_0;
    r_1.y_0 = a_0.w_0 * o_0.y_0 - a_0.x_1 * o_0.z_0 + a_0.y_0 * o_0.w_0 + a_0.z_0 * o_0.x_1;
    r_1.z_0 = a_0.w_0 * o_0.z_0 + a_0.x_1 * o_0.y_0 - a_0.y_0 * o_0.x_1 + a_0.z_0 * o_0.w_0;
    return r_1;
}

fn normalized_0( q_1 : Quat_0) -> Quat_0
{
    var _S4 : f32 = q_1.w_0;
    var _S5 : f32 = q_1.x_1;
    var _S6 : f32 = q_1.y_0;
    var _S7 : f32 = q_1.z_0;
    var n_0 : f32 = sqrt(_S4 * _S4 + _S5 * _S5 + _S6 * _S6 + _S7 * _S7);
    var r_2 : Quat_0;
    r_2.w_0 = q_1.w_0 / n_0;
    r_2.x_1 = q_1.x_1 / n_0;
    r_2.y_0 = q_1.y_0 / n_0;
    r_2.z_0 = q_1.z_0 / n_0;
    return r_2;
}

fn to_zup_0( v_0 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_0.x, - v_0.z, v_0.y);
}

fn quat_vec_0( q_2 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_2.x_1, q_2.y_0, q_2.z_0, q_2.w_0);
}

fn rotate_0( q_3 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_3.x_1, q_3.y_0, q_3.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_1) * vec3<f32>(2.0f);
    return v_1 + t_0 * vec3<f32>(q_3.w_0) + cross(qv_0, t_0);
}

fn inverse_rotate_0( q_4 : Quat_0,  v_2 : vec3<f32>) -> vec3<f32>
{
    var c_0 : Quat_0;
    c_0.w_0 = q_4.w_0;
    c_0.x_1 = - q_4.x_1;
    c_0.y_0 = - q_4.y_0;
    c_0.z_0 = - q_4.z_0;
    return rotate_0(c_0, v_2);
}

fn mat_mul_0( r0_0 : vec4<f32>,  r1_0 : vec4<f32>,  r2_0 : vec4<f32>,  v_3 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(r0_0.xyz, v_3), dot(r1_0.xyz, v_3), dot(r2_0.xyz, v_3));
}

fn world_mul_0( q_5 : Quat_0,  r0_1 : vec4<f32>,  r1_1 : vec4<f32>,  r2_1 : vec4<f32>,  v_4 : vec3<f32>) -> vec3<f32>
{
    return rotate_0(q_5, mat_mul_0(r0_1, r1_1, r2_1, inverse_rotate_0(q_5, v_4)));
}

struct Island_0
{
     range_0 : vec4<u32>,
     info_0 : vec4<u32>,
     com_0 : vec4<f32>,
     inertia0_0 : vec4<f32>,
     inertia1_0 : vec4<f32>,
     inertia2_0 : vec4<f32>,
     inv0_0 : vec4<f32>,
     inv1_0 : vec4<f32>,
     inv2_0 : vec4<f32>,
     wcom_0 : vec4<f32>,
     winv0_0 : vec4<f32>,
     winv1_0 : vec4<f32>,
     winv2_0 : vec4<f32>,
     rotation_0 : vec4<f32>,
     position_0 : vec4<f32>,
     position_err_0 : vec4<f32>,
     velocity_0 : vec4<f32>,
     velocity_err_0 : vec4<f32>,
     angular_velocity_0 : vec4<f32>,
     done_0 : u32,
     coverage_0 : u32,
     wide_groups_0 : u32,
     done_pad_0 : u32,
     probes_0 : vec4<u32>,
     energy_0 : vec4<f32>,
     rotation_err_0 : vec4<f32>,
     momentum_0 : vec4<f32>,
     momentum_err_0 : vec4<f32>,
};

@compute
@workgroup_size(64, 1, 1)
fn resident_poses(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var i_0 : u32 = id_0.x;
    if(i_0 >= (params_0.island_count_0))
    {
        return;
    }
    var _S8 : u32 = u32(3) * i_0;
    var body_0 : u32 = (bitcast<u32>((ctab_0[params_0.engine_base_0 + _S8].x)));
    if(body_0 == u32(4294967295))
    {
        return;
    }
    var p_inv_0 : Quat_0 = quat_of_0(ctab_0[params_0.engine_base_0 + _S8 + u32(1)]);
    var com_1 : vec3<f32> = ctab_0[params_0.engine_base_0 + _S8 + u32(2)].xyz;
    var b_0 : EngineBody_std430_0 = start_bodies_0[body_0];
    var a_inv_0 : Quat_0;
    a_inv_0.w_0 = 0.70710676908493042f;
    a_inv_0.x_1 = 0.70710676908493042f;
    a_inv_0.y_0 = 0.0f;
    a_inv_0.z_0 = 0.0f;
    var q_6 : Quat_0 = normalized_0(quat_mul_0(quat_mul_0(a_inv_0, quat_of_0(b_0.rot_0)), p_inv_0));
    var com_world_0 : vec3<f32> = to_zup_0(b_0.pos_inv_mass_0.xyz);
    var isl_0 : Island_0;
    isl_0.range_0 = islands_0[i_0].range_0;
    isl_0.info_0 = islands_0[i_0].info_0;
    isl_0.com_0 = islands_0[i_0].com_0;
    isl_0.inertia0_0 = islands_0[i_0].inertia0_0;
    isl_0.inertia1_0 = islands_0[i_0].inertia1_0;
    isl_0.inertia2_0 = islands_0[i_0].inertia2_0;
    isl_0.inv0_0 = islands_0[i_0].inv0_0;
    isl_0.inv1_0 = islands_0[i_0].inv1_0;
    isl_0.inv2_0 = islands_0[i_0].inv2_0;
    isl_0.wcom_0 = islands_0[i_0].wcom_0;
    isl_0.winv0_0 = islands_0[i_0].winv0_0;
    isl_0.winv1_0 = islands_0[i_0].winv1_0;
    isl_0.winv2_0 = islands_0[i_0].winv2_0;
    isl_0.rotation_0 = islands_0[i_0].rotation_0;
    isl_0.position_0 = islands_0[i_0].position_0;
    isl_0.position_err_0 = islands_0[i_0].position_err_0;
    isl_0.velocity_0 = islands_0[i_0].velocity_0;
    isl_0.velocity_err_0 = islands_0[i_0].velocity_err_0;
    isl_0.angular_velocity_0 = islands_0[i_0].angular_velocity_0;
    isl_0.done_0 = islands_0[i_0].done_0;
    isl_0.coverage_0 = islands_0[i_0].coverage_0;
    isl_0.wide_groups_0 = islands_0[i_0].wide_groups_0;
    isl_0.done_pad_0 = islands_0[i_0].done_pad_0;
    isl_0.probes_0 = islands_0[i_0].probes_0;
    isl_0.energy_0 = islands_0[i_0].energy_0;
    isl_0.rotation_err_0 = islands_0[i_0].rotation_err_0;
    isl_0.momentum_0 = islands_0[i_0].momentum_0;
    isl_0.momentum_err_0 = islands_0[i_0].momentum_err_0;
    isl_0.rotation_0 = quat_vec_0(q_6);
    var _S9 : vec4<f32> = vec4<f32>(0.0f);
    isl_0.rotation_err_0 = _S9;
    isl_0.position_0 = vec4<f32>(com_world_0 - rotate_0(q_6, com_1), 0.0f);
    isl_0.position_err_0 = _S9;
    isl_0.velocity_0 = vec4<f32>(to_zup_0(b_0.vel_sleep_0.xyz), 0.0f);
    isl_0.velocity_err_0 = _S9;
    var w_1 : vec3<f32> = to_zup_0(b_0.ang_vel_gen_0.xyz);
    isl_0.angular_velocity_0 = vec4<f32>(w_1, 0.0f);
    isl_0.momentum_0 = vec4<f32>(world_mul_0(q_6, isl_0.inertia0_0, isl_0.inertia1_0, isl_0.inertia2_0, w_1), 0.0f);
    isl_0.momentum_err_0 = _S9;
    islands_0[i_0].range_0 = isl_0.range_0;
    islands_0[i_0].info_0 = isl_0.info_0;
    islands_0[i_0].com_0 = isl_0.com_0;
    islands_0[i_0].inertia0_0 = isl_0.inertia0_0;
    islands_0[i_0].inertia1_0 = isl_0.inertia1_0;
    islands_0[i_0].inertia2_0 = isl_0.inertia2_0;
    islands_0[i_0].inv0_0 = isl_0.inv0_0;
    islands_0[i_0].inv1_0 = isl_0.inv1_0;
    islands_0[i_0].inv2_0 = isl_0.inv2_0;
    islands_0[i_0].wcom_0 = isl_0.wcom_0;
    islands_0[i_0].winv0_0 = isl_0.winv0_0;
    islands_0[i_0].winv1_0 = isl_0.winv1_0;
    islands_0[i_0].winv2_0 = isl_0.winv2_0;
    islands_0[i_0].rotation_0 = isl_0.rotation_0;
    islands_0[i_0].position_0 = isl_0.position_0;
    islands_0[i_0].position_err_0 = isl_0.position_err_0;
    islands_0[i_0].velocity_0 = isl_0.velocity_0;
    islands_0[i_0].velocity_err_0 = isl_0.velocity_err_0;
    islands_0[i_0].angular_velocity_0 = isl_0.angular_velocity_0;
    islands_0[i_0].done_0 = isl_0.done_0;
    islands_0[i_0].coverage_0 = isl_0.coverage_0;
    islands_0[i_0].wide_groups_0 = isl_0.wide_groups_0;
    islands_0[i_0].done_pad_0 = isl_0.done_pad_0;
    islands_0[i_0].probes_0 = isl_0.probes_0;
    islands_0[i_0].energy_0 = isl_0.energy_0;
    islands_0[i_0].rotation_err_0 = isl_0.rotation_err_0;
    islands_0[i_0].momentum_0 = isl_0.momentum_0;
    islands_0[i_0].momentum_err_0 = isl_0.momentum_err_0;
    return;
}

fn pulse_mean_0( start_0 : f32,  plateau_0 : f32,  plateau_time_0 : f32,  peak_0 : f32,  duration_0 : f32,  dt_1 : f32) -> f32
{
    var _S10 : f32 = start_0 + plateau_time_0;
    var sum_0 : f32 = plateau_0 * max(min(_S10, dt_1) - max(start_0, 0.0f), 0.0f);
    var _S11 : f32 = max(0.0f, _S10);
    var _S12 : f32 = min(dt_1, _S10 + duration_0);
    var _S13 : bool;
    if(_S12 > _S11)
    {
        _S13 = duration_0 > 0.0f;
    }
    else
    {
        _S13 = false;
    }
    var sum_1 : f32;
    if(_S13)
    {
        var w_2 : f32 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S11 - _S10)) - cos(w_2 * (_S12 - _S10)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

fn hertz_duration_0( m_0 : f32,  r_3 : f32,  e_0 : f32,  v_5 : f32) -> f32
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_3 * e_0 * e_0 * max(abs(v_5), 0.00100000004749745f)), 0.20000000298023224f);
}

fn island_rotation_0( _S14 : u32) -> Quat_0
{
    return quat_of_0(islands_0[_S14].rotation_0);
}

@compute
@workgroup_size(64, 1, 1)
fn resident_filter(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var m_1 : f32;
    var overflow_0 : bool;
    var overflow_1 : bool;
    var p_0 : u32;
    var kept_0 : u32;
    var q3_0 : u32;
    var f_total_0 : vec3<f32>;
    var t_total_0 : vec3<f32>;
    var k_0 : u32 = id_1.x;
    var n_1 : u32 = params_0.chunk_count_0;
    if(k_0 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S15 : u32 = chunks_0[k_0].info_1.y;
    var isl_1 : Island_std430_0 = islands_0[_S15];
    var _S16 : vec4<f32> = chunks_0[k_0].center_0;
    var _S17 : vec3<f32> = islands_0[_S15].position_0.xyz + rotate_0(island_rotation_0(_S15), chunks_0[k_0].center_0.xyz + state_0[u32(4) * k_0].xyz);
    var _S18 : vec4<f32> = ctab_0[params_0.props_base_0 + k_0];
    var _S19 : f32;
    if((((islands_0[_S15].info_0.x) & (u32(1)))) != u32(0))
    {
        _S19 = (bitcast<f32>((u32(2139095040))));
    }
    else
    {
        _S19 = isl_1.com_0.w;
    }
    var _S20 : f32 = params_0.dt_0;
    var _S21 : vec3<f32> = vec3<f32>(0.0f);
    var _S22 : u32 = sort_0[k_0];
    var _S23 : u32 = sort_0[n_1 + u32(1) + k_0];
    var first_0 : bool = true;
    var last_0 : u32 = u32(0);
    var step_0 : u32 = u32(0);
    var f_total_1 : vec3<f32> = _S21;
    var t_total_1 : vec3<f32> = _S21;
    loop
    {
        if(step_0 < _S23)
        {
        }
        else
        {
            break;
        }
        p_0 = u32(4294967295);
        kept_0 = u32(0);
        q3_0 = u32(0);
        loop
        {
            if(q3_0 < _S23)
            {
            }
            else
            {
                break;
            }
            var key_1 : u32 = manifolds_0[_S22 + q3_0].ids_0.y + u32(1);
            if(first_0)
            {
                overflow_0 = true;
            }
            else
            {
                overflow_0 = key_1 > last_0;
            }
            if(overflow_0)
            {
                if(p_0 == u32(4294967295))
                {
                    overflow_1 = true;
                }
                else
                {
                    overflow_1 = key_1 < kept_0;
                }
            }
            else
            {
                overflow_1 = false;
            }
            if(overflow_1)
            {
                p_0 = q3_0;
                kept_0 = key_1;
            }
            q3_0 = q3_0 + u32(1);
        }
        if(p_0 == u32(4294967295))
        {
            break;
        }
        var r_4 : Record_std430_0 = manifolds_0[_S22 + p_0];
        var _S24 : vec3<f32> = r_4.impulse_0.xyz;
        var j_0 : f32 = length(_S24);
        if(j_0 == 0.0f)
        {
            var _S25 : u32 = step_0 + u32(1);
            first_0 = false;
            last_0 = kept_0;
            step_0 = _S25;
            continue;
        }
        var m_other_0 : f32 = r_4.other_0.x;
        if(isinf_0(_S19))
        {
            if(isinf_0(m_other_0))
            {
                m_1 = _S16.w;
            }
            else
            {
                m_1 = m_other_0;
            }
        }
        else
        {
            if(isinf_0(m_other_0))
            {
                m_1 = _S19;
            }
            else
            {
                m_1 = _S19 * m_other_0 / (_S19 + m_other_0);
            }
        }
        var _S26 : f32 = min(j_0, 2.0f * m_1 * max(r_4.impulse_0.w - params_0.gravity_0.w, 0.0f));
        if(_S26 < j_0)
        {
            var f_0 : vec3<f32> = _S24 / vec3<f32>(j_0) * vec3<f32>(((j_0 - _S26) / _S20));
            var t_total_2 : vec3<f32> = t_total_1 + cross(r_4.point_0.xyz - _S17, f_0);
            f_total_0 = f_total_1 + f_0;
            t_total_0 = t_total_2;
        }
        else
        {
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
        }
        f_total_1 = f_total_0;
        t_total_1 = t_total_0;
        var _S25 : u32 = step_0 + u32(1);
        first_0 = false;
        last_0 = kept_0;
        step_0 = _S25;
    }
    var _S27 : u32 = params_0.pulse_base_0 + u32(3) * params_0.pulse_cap_0 * k_0;
    var _S28 : u32 = (bitcast<u32>((ctab_0[params_0.pulse_count_base_0 + k_0].x)));
    p_0 = u32(0);
    kept_0 = u32(0);
    loop
    {
        if(p_0 < _S28)
        {
        }
        else
        {
            break;
        }
        var _S29 : u32 = _S27 + u32(3) * p_0;
        var a_1 : vec4<f32> = ctab_0[_S29];
        var b_1 : vec4<f32> = ctab_0[_S29 + u32(1)];
        var c_1 : vec4<f32> = ctab_0[_S29 + u32(2)];
        var _S30 : f32 = ctab_0[_S29 + u32(1)].w;
        var mean_0 : f32 = pulse_mean_0(ctab_0[_S29].x, ctab_0[_S29].y, ctab_0[_S29].z, ctab_0[_S29].w, _S30, _S20);
        if(mean_0 != 0.0f)
        {
            var f_1 : vec3<f32> = c_1.xyz * vec3<f32>(mean_0);
            var t_total_3 : vec3<f32> = t_total_1 + cross(b_1.xyz - _S17, f_1);
            f_total_1 = f_total_1 + f_1;
            t_total_1 = t_total_3;
        }
        if((a_1.x + a_1.z + _S30) > _S20)
        {
            a_1[i32(0)] = a_1[i32(0)] - _S20;
            var _S31 : u32 = _S27 + u32(3) * kept_0;
            ctab_0[_S31] = a_1;
            ctab_0[_S31 + u32(1)] = b_1;
            ctab_0[_S31 + u32(2)] = c_1;
            kept_0 = kept_0 + u32(1);
        }
        p_0 = p_0 + u32(1);
    }
    overflow_0 = false;
    q3_0 = u32(0);
    loop
    {
        if(q3_0 < _S23)
        {
        }
        else
        {
            break;
        }
        var r_5 : Record_std430_0 = manifolds_0[_S22 + q3_0];
        var _S32 : vec3<f32> = r_5.impulse_0.xyz;
        var j_1 : f32 = length(_S32);
        if(j_1 == 0.0f)
        {
            overflow_1 = overflow_0;
            var _S33 : u32 = q3_0 + u32(1);
            overflow_0 = overflow_1;
            q3_0 = _S33;
            continue;
        }
        var m_other_1 : f32 = r_5.other_0.x;
        if(isinf_0(_S19))
        {
            if(isinf_0(m_other_1))
            {
                m_1 = _S16.w;
            }
            else
            {
                m_1 = m_other_1;
            }
        }
        else
        {
            if(isinf_0(m_other_1))
            {
                m_1 = _S19;
            }
            else
            {
                m_1 = _S19 * m_other_1 / (_S19 + m_other_1);
            }
        }
        var _S34 : f32 = r_5.impulse_0.w;
        var _S35 : f32 = min(j_1, 2.0f * m_1 * max(_S34 - params_0.gravity_0.w, 0.0f));
        if(_S35 <= 0.0f)
        {
            overflow_1 = overflow_0;
            var _S33 : u32 = q3_0 + u32(1);
            overflow_0 = overflow_1;
            q3_0 = _S33;
            continue;
        }
        var _S36 : f32 = _S18.y;
        var duration_1 : f32 = hertz_duration_0(m_1, min(r_5.other_0.z, _S18.z), 1.0f / ((1.0f - _S36 * _S36) / _S18.x + 1.0f / r_5.other_0.y), _S34);
        var peak_1 : f32 = 3.14159274101257324f * _S35 / (2.0f * duration_1);
        var dir_0 : vec3<f32> = _S32 / vec3<f32>(j_1);
        var mean_1 : f32;
        if((params_0.impact_peak_0) != u32(0))
        {
            mean_1 = peak_1;
        }
        else
        {
            mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S20);
        }
        if(mean_1 != 0.0f)
        {
            var f_2 : vec3<f32> = dir_0 * vec3<f32>(mean_1);
            var t_total_4 : vec3<f32> = t_total_1 + cross(r_5.point_0.xyz - _S17, f_2);
            f_total_0 = f_total_1 + f_2;
            t_total_0 = t_total_4;
        }
        else
        {
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
        }
        var kept_1 : u32;
        if(duration_1 > _S20)
        {
            if(kept_0 < (params_0.pulse_cap_0))
            {
                var _S37 : u32 = _S27 + u32(3) * kept_0;
                ctab_0[_S37] = vec4<f32>(- _S20, 0.0f, 0.0f, peak_1);
                ctab_0[_S37 + u32(1)] = vec4<f32>(r_5.point_0.xyz, duration_1);
                ctab_0[_S37 + u32(2)] = vec4<f32>(dir_0, 0.0f);
                var _S38 : u32 = kept_0 + u32(1);
                overflow_1 = overflow_0;
                kept_1 = _S38;
            }
            else
            {
                overflow_1 = true;
                kept_1 = kept_0;
            }
        }
        else
        {
            overflow_1 = overflow_0;
            kept_1 = kept_0;
        }
        f_total_1 = f_total_0;
        t_total_1 = t_total_0;
        kept_0 = kept_1;
        var _S33 : u32 = q3_0 + u32(1);
        overflow_0 = overflow_1;
        q3_0 = _S33;
    }
    ctab_0[params_0.pulse_count_base_0 + k_0] = vec4<f32>((bitcast<f32>((kept_0))), 0.0f, 0.0f, 0.0f);
    if(overflow_0)
    {
        ctab_0[params_0.status_base_0] = vec4<f32>((bitcast<f32>((u32(1)))), 0.0f, 0.0f, 0.0f);
    }
    var _S39 : u32 = u32(2) * k_0;
    ctab_0[params_0.eload_base_0 + _S39] = vec4<f32>(f_total_1, 0.0f);
    ctab_0[params_0.eload_base_0 + _S39 + u32(1)] = vec4<f32>(t_total_1, 0.0f);
    return;
}

var<workgroup> g_f_0 : array<vec3<f32>, i32(256)>;

var<workgroup> g_t_0 : array<vec3<f32>, i32(256)>;

@compute
@workgroup_size(256, 1, 1)
fn resident_static_loads(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var i_1 : u32 = group_0.x;
    var tid_0 : u32 = thread_0.x;
    if(i_1 >= (params_0.island_count_0))
    {
        return;
    }
    var isl_2 : Island_std430_0 = islands_0[i_1];
    if((((islands_0[i_1].info_0.z) & (u32(8)))) == u32(0))
    {
        return;
    }
    var _S40 : Quat_0 = island_rotation_0(i_1);
    var w_3 : vec3<f32> = isl_2.angular_velocity_0.xyz;
    var _S41 : vec3<f32> = params_0.gravity_0.xyz;
    var _S42 : vec3<f32> = isl_2.position_0.xyz;
    var _S43 : vec3<f32> = isl_2.com_0.xyz;
    var _S44 : vec3<f32> = _S42 + rotate_0(_S40, _S43);
    var _S45 : vec3<f32> = vec3<f32>(0.0f);
    var _S46 : bool = !((((isl_2.info_0.x) & (u32(1)))) != u32(0));
    var s_0 : u32;
    var f_3 : vec3<f32>;
    var t_1 : vec3<f32>;
    if(_S46)
    {
        s_0 = isl_2.range_0.x + tid_0;
        f_3 = _S45;
        t_1 = _S45;
        loop
        {
            if(s_0 < (isl_2.range_0.y))
            {
            }
            else
            {
                break;
            }
            var _S47 : u32 = u32(2) * s_0;
            var fc_0 : vec3<f32> = ctab_0[params_0.eload_base_0 + _S47].xyz + _S41 * vec3<f32>(chunks_0[s_0].center_0.w);
            var f_4 : vec3<f32> = f_3 + fc_0;
            var t_2 : vec3<f32> = t_1 + (cross(_S42 + rotate_0(_S40, chunks_0[s_0].center_0.xyz + state_0[u32(4) * s_0].xyz) - _S44, fc_0) + ctab_0[params_0.eload_base_0 + _S47 + u32(1)].xyz);
            s_0 = s_0 + u32(256);
            f_3 = f_4;
            t_1 = t_2;
        }
    }
    else
    {
        f_3 = _S45;
        t_1 = _S45;
    }
    g_f_0[tid_0] = f_3;
    g_t_0[tid_0] = t_1;
    workgroupBarrier();
    s_0 = u32(128);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < s_0)
        {
            var _S48 : u32 = tid_0 + s_0;
            g_f_0[tid_0] = g_f_0[tid_0] + g_f_0[_S48];
            g_t_0[tid_0] = g_t_0[tid_0] + g_t_0[_S48];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var a_2 : vec3<f32>;
    var alpha_0 : vec3<f32>;
    if(_S46)
    {
        var _S49 : vec3<f32> = world_mul_0(_S40, isl_2.inv0_0, isl_2.inv1_0, isl_2.inv2_0, g_t_0[i32(0)] - cross(w_3, world_mul_0(_S40, isl_2.inertia0_0, isl_2.inertia1_0, isl_2.inertia2_0, w_3)));
        a_2 = g_f_0[i32(0)] / vec3<f32>(isl_2.com_0.w);
        alpha_0 = _S49;
    }
    else
    {
        a_2 = _S45;
        alpha_0 = _S45;
    }
    var _S50 : bool = (params_0.rml_0) != u32(0);
    var _S51 : vec3<f32> = inverse_rotate_0(_S40, w_3);
    var c_2 : u32 = isl_2.range_0.x + tid_0;
    loop
    {
        if(c_2 < (isl_2.range_0.y))
        {
        }
        else
        {
            break;
        }
        var mass_0 : f32 = chunks_0[c_2].center_0.w;
        var _S52 : u32 = u32(4) * c_2;
        var r_world_0 : vec3<f32> = rotate_0(_S40, chunks_0[c_2].center_0.xyz + state_0[_S52].xyz - _S43);
        var _S53 : u32 = u32(2) * c_2;
        var _S54 : vec3<f32> = vec3<f32>(mass_0);
        var f_world_0 : vec3<f32> = ctab_0[params_0.eload_base_0 + _S53].xyz + _S41 * _S54;
        var t_world_0 : vec3<f32> = ctab_0[params_0.eload_base_0 + _S53 + u32(1)].xyz;
        var f_world_1 : vec3<f32>;
        var t_world_1 : vec3<f32>;
        if(_S50)
        {
            var t_world_2 : vec3<f32> = t_world_0 - (world_mul_0(_S40, chunks_0[c_2].inertia0_1, chunks_0[c_2].inertia1_1, chunks_0[c_2].inertia2_1, alpha_0) + cross(w_3, world_mul_0(_S40, chunks_0[c_2].inertia0_1, chunks_0[c_2].inertia1_1, chunks_0[c_2].inertia2_1, w_3)));
            f_world_1 = f_world_0 - (a_2 + cross(alpha_0, r_world_0) + cross(w_3, cross(w_3, r_world_0))) * _S54;
            t_world_1 = t_world_2;
        }
        else
        {
            f_world_1 = f_world_0;
            t_world_1 = t_world_0;
        }
        var fb_0 : vec3<f32> = inverse_rotate_0(_S40, f_world_1);
        var mo_0 : vec3<f32> = inverse_rotate_0(_S40, t_world_1);
        var fb_1 : vec3<f32>;
        var mo_1 : vec3<f32>;
        if(_S50)
        {
            var sw_0 : vec3<f32> = state_0[_S52 + u32(3)].xyz;
            var _S55 : vec3<f32> = mat_mul_0(chunks_0[c_2].inertia0_1, chunks_0[c_2].inertia1_1, chunks_0[c_2].inertia2_1, sw_0);
            var mo_2 : vec3<f32> = mo_0 - (cross(_S51, _S55) + cross(sw_0, mat_mul_0(chunks_0[c_2].inertia0_1, chunks_0[c_2].inertia1_1, chunks_0[c_2].inertia2_1, _S51)) + cross(sw_0, _S55));
            fb_1 = fb_0 - cross(_S51, state_0[_S52 + u32(2)].xyz) * vec3<f32>((2.0f * mass_0));
            mo_1 = mo_2;
        }
        else
        {
            fb_1 = fb_0;
            mo_1 = mo_0;
        }
        scratch_0[params_0.sload_base_0 + _S53] = vec4<f32>(fb_1, 0.0f);
        scratch_0[params_0.sload_base_0 + _S53 + u32(1)] = vec4<f32>(mo_1, 0.0f);
        c_2 = c_2 + u32(256);
    }
    return;
}

