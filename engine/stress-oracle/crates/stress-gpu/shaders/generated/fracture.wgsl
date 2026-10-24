struct FractureParams_std140_0
{
    @align(16) chunk_count_0 : u32,
    @align(4) bond_count_0 : u32,
    @align(8) body_count_0 : u32,
    @align(4) max_bodies_0 : u32,
    @align(16) step_0 : u32,
    @align(4) pool_cap_0 : u32,
    @align(8) impact_peak_0 : u32,
    @align(4) parity_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) band_0 : f32,
    @align(8) scale_m_0 : f32,
    @align(4) scale_s_0 : f32,
    @align(16) scale_j_0 : f32,
    @align(4) strike_speed_0 : f32,
    @align(8) shot_count_0 : u32,
    @align(4) pad3_0 : f32,
    @align(16) load_model_0 : u32,
    @align(4) strike_model_0 : u32,
    @align(8) fracture_work_0 : u32,
    @align(4) sweeps_0 : u32,
    @align(16) gx_0 : f32,
    @align(4) gy_0 : f32,
    @align(8) gz_0 : f32,
    @align(4) lp_c1_0 : f32,
    @align(16) lp_c2_0 : f32,
    @align(4) ext_chunks_0 : u32,
    @align(8) ext_adj_0 : u32,
    @align(4) ext_mat_0 : u32,
    @align(16) il_passes_0 : u32,
    @align(4) lp_scale_0 : f32,
    @align(8) pad4_0 : f32,
    @align(4) pad5_0 : f32,
};

@binding(0) @group(0) var<uniform> params_0 : FractureParams_std140_0;
@binding(4) @group(0) var<storage, read_write> work_0 : array<u32>;

@binding(6) @group(0) var<storage, read_write> frames_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read_write> bodies_0 : array<vec4<f32>>;

@binding(1) @group(0) var<storage, read> chunks_0 : array<vec4<f32>>;

@binding(11) @group(0) var<storage, read> sort_0 : array<u32>;

@binding(5) @group(0) var<storage, read_write> atom_0 : array<atomic<u32>>;

struct Record_std430_0
{
    @align(16) ids_0 : vec4<u32>,
    @align(16) point_0 : vec4<f32>,
    @align(16) impulse_0 : vec4<f32>,
    @align(16) other_0 : vec4<f32>,
    @align(16) key_0 : vec4<u32>,
};

@binding(10) @group(0) var<storage, read> manifolds_0 : array<Record_std430_0>;

@binding(12) @group(0) var<storage, read_write> loads_0 : array<vec4<f32>>;

@binding(2) @group(0) var<storage, read> bonds_0 : array<vec4<u32>>;

@binding(13) @group(0) var<storage, read_write> args_0 : array<u32>;

@binding(15) @group(0) var<storage, read> info_0 : array<u32>;

struct ScanParams_std140_0
{
    @align(16) n_0 : u32,
    @align(4) in_base_0 : u32,
    @align(8) out_base_0 : u32,
    @align(4) block_base_0 : u32,
};

@binding(14) @group(0) var<uniform> scan_0 : ScanParams_std140_0;
@binding(3) @group(0) var<storage, read> parts_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read_write> colliders_0 : array<vec4<f32>>;

@binding(9) @group(0) var<storage, read_write> collider_info_0 : array<vec4<u32>>;

@binding(17) @group(0) var<storage, read> view_chunks_0 : array<vec4<f32>>;

struct ViewParams_std140_0
{
    @align(16) mode_0 : u32,
    @align(4) shot_count_1 : u32,
    @align(8) impactor_base_0 : u32,
    @align(4) pad0_0 : u32,
    @align(16) rest_speed_0 : f32,
    @align(4) pad1_0 : f32,
    @align(8) pad2_0 : f32,
    @align(4) pad3_1 : f32,
};

@binding(16) @group(0) var<uniform> view_0 : ViewParams_std140_0;
@binding(18) @group(0) var<storage, read_write> objects_0 : array<vec4<f32>>;

@binding(20) @group(0) var<storage, read> shot_bodies_0 : array<u32>;

@binding(19) @group(0) var<storage, read_write> lines_0 : array<f32>;

@binding(21) @group(0) var<storage, read> geometry_0 : array<vec4<f32>>;

@binding(22) @group(0) var<storage, read> shots_0 : array<vec4<f32>>;

@binding(24) @group(0) var<storage, read_write> lp_0 : array<vec4<f32>>;

@binding(23) @group(0) var<storage, read> ext_0 : array<vec4<f32>>;

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

fn w_body_0() -> u32
{
    return u32(0);
}

fn body_of_0( c_0 : u32) -> u32
{
    return work_0[w_body_0() + c_0];
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

fn conj_0( q_1 : Quat_0) -> Quat_0
{
    var c_1 : Quat_0;
    c_1.w_0 = q_1.w_0;
    c_1.x_1 = - q_1.x_1;
    c_1.y_0 = - q_1.y_0;
    c_1.z_0 = - q_1.z_0;
    return c_1;
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

fn normalized_0( q_2 : Quat_0) -> Quat_0
{
    var _S4 : f32 = q_2.w_0;
    var _S5 : f32 = q_2.x_1;
    var _S6 : f32 = q_2.y_0;
    var _S7 : f32 = q_2.z_0;
    var n_1 : f32 = sqrt(_S4 * _S4 + _S5 * _S5 + _S6 * _S6 + _S7 * _S7);
    var r_2 : Quat_0;
    r_2.w_0 = q_2.w_0 / n_1;
    r_2.x_1 = q_2.x_1 / n_1;
    r_2.y_0 = q_2.y_0 / n_1;
    r_2.z_0 = q_2.z_0 / n_1;
    return r_2;
}

fn rot_from_rubble_0( q_3 : Quat_0) -> Quat_0
{
    var axes_0 : Quat_0;
    axes_0.w_0 = 0.70710676908493042f;
    axes_0.x_1 = 0.70710676908493042f;
    axes_0.y_0 = 0.0f;
    axes_0.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_0, q_3));
}

fn from_rubble_0( v_0 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_0.x, - v_0.z, v_0.y);
}

fn rotate_0( q_4 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_4.x_1, q_4.y_0, q_4.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_1) * vec3<f32>(2.0f);
    return v_1 + t_0 * vec3<f32>(q_4.w_0) + cross(qv_0, t_0);
}

fn structure_pose_0( slot_0 : u32,  b_0 : u32,  q_5 : ptr<function, Quat_0>,  origin_0 : ptr<function, vec3<f32>>)
{
    var _S8 : u32 = u32(2) * slot_0;
    var com_0 : vec3<f32> = frames_0[_S8].xyz;
    var _S9 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_0 + u32(1)])), conj_0(quat_of_0(frames_0[_S8 + u32(1)]))));
    (*q_5) = _S9;
    (*origin_0) = from_rubble_0(bodies_0[b_0].xyz) - rotate_0(_S9, com_0);
    return;
}

fn structure_pose_1( slot_1 : u32,  b_1 : u32,  q_6 : ptr<function, Quat_0>,  origin_1 : ptr<function, vec3<f32>>)
{
    var _S10 : u32 = u32(2) * slot_1;
    var com_1 : vec3<f32> = frames_0[_S10].xyz;
    var _S11 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_1 + u32(1)])), conj_0(quat_of_0(frames_0[_S10 + u32(1)]))));
    (*q_6) = _S11;
    (*origin_1) = from_rubble_0(bodies_0[b_1].xyz) - rotate_0(_S11, com_1);
    return;
}

fn structure_pose_2( slot_2 : u32,  b_2 : u32,  q_7 : ptr<function, Quat_0>,  origin_2 : ptr<function, vec3<f32>>)
{
    var _S12 : u32 = u32(2) * slot_2;
    var com_2 : vec3<f32> = frames_0[_S12].xyz;
    var _S13 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_2 + u32(1)])), conj_0(quat_of_0(frames_0[_S12 + u32(1)]))));
    (*q_7) = _S13;
    (*origin_2) = from_rubble_0(bodies_0[b_2].xyz) - rotate_0(_S13, com_2);
    return;
}

fn structure_pose_3( slot_3 : u32,  b_3 : u32,  q_8 : ptr<function, Quat_0>,  origin_3 : ptr<function, vec3<f32>>)
{
    var _S14 : u32 = u32(2) * slot_3;
    var com_3 : vec3<f32> = frames_0[_S14].xyz;
    var _S15 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_3 + u32(1)])), conj_0(quat_of_0(frames_0[_S14 + u32(1)]))));
    (*q_8) = _S15;
    (*origin_3) = from_rubble_0(bodies_0[b_3].xyz) - rotate_0(_S15, com_3);
    return;
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

fn w_alive_0() -> u32
{
    return params_0.chunk_count_0;
}

fn w_label_0() -> u32
{
    return w_alive_0() + params_0.bond_count_0;
}

fn w_slot_0() -> u32
{
    return w_label_0() + params_0.chunk_count_0;
}

fn w_new_0() -> u32
{
    return w_slot_0() + params_0.chunk_count_0;
}

fn w_rank_0() -> u32
{
    return w_new_0() + params_0.chunk_count_0;
}

fn w_free_0() -> u32
{
    return w_rank_0() + params_0.chunk_count_0;
}

fn w_free_rank_0() -> u32
{
    return w_free_0() + params_0.body_count_0;
}

fn w_free_list_0() -> u32
{
    return w_free_rank_0() + params_0.body_count_0;
}

fn w_pulses_0() -> u32
{
    return w_free_list_0() + params_0.body_count_0 + u32(2050);
}

fn a_sums_0() -> u32
{
    return u32(3) * params_0.chunk_count_0;
}

fn a_dirty_0() -> u32
{
    return a_sums_0() + u32(20) * params_0.chunk_count_0;
}

fn a_best_count_0() -> u32
{
    return a_dirty_0() + params_0.body_count_0;
}

fn a_best_root_0() -> u32
{
    return a_best_count_0() + params_0.body_count_0;
}

fn a_counters_0() -> u32
{
    return a_best_root_0() + params_0.body_count_0;
}

fn l_pool_0( half_0 : u32) -> u32
{
    return params_0.chunk_count_0 + u32(3) * params_0.pool_cap_0 * half_0;
}

fn pulse_mean_0( start_0 : f32,  plateau_0 : f32,  plateau_time_0 : f32,  peak_0 : f32,  duration_0 : f32,  dt_1 : f32) -> f32
{
    var _S16 : f32 = start_0 + plateau_time_0;
    var sum_0 : f32 = plateau_0 * max(min(_S16, dt_1) - max(start_0, 0.0f), 0.0f);
    var _S17 : f32 = max(0.0f, _S16);
    var _S18 : f32 = min(dt_1, _S16 + duration_0);
    var _S19 : bool;
    if(_S18 > _S17)
    {
        _S19 = duration_0 > 0.0f;
    }
    else
    {
        _S19 = false;
    }
    var sum_1 : f32;
    if(_S19)
    {
        var w_1 : f32 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_1 * (cos(w_1 * (_S17 - _S16)) - cos(w_1 * (_S18 - _S16)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

fn hertz_duration_0( m_0 : f32,  r_3 : f32,  e_0 : f32,  v_2 : f32) -> f32
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_3 * e_0 * e_0 * max(abs(v_2), 0.00100000004749745f)), 0.20000000298023224f);
}

fn l_vec_0() -> u32
{
    return params_0.chunk_count_0 + u32(6) * params_0.pool_cap_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_loads(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var f_total_0 : vec3<f32>;
    var t_total_0 : vec3<f32>;
    var m_1 : f32;
    var p_0 : u32;
    var kept_0 : u32;
    var q3_0 : u32;
    var overflow_0 : bool;
    var overflow_1 : bool;
    var f_total_1 : vec3<f32>;
    var t_total_1 : vec3<f32>;
    var k_0 : u32 = id_0.x;
    if(k_0 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_4 : u32 = body_of_0(k_0);
    var b_4 : u32 = u32(7) * slot_4;
    var q_9 : Quat_0;
    var origin_4 : vec3<f32>;
    structure_pose_3(slot_4, b_4, &(q_9), &(origin_4));
    var _S20 : u32 = u32(4) * k_0;
    var c0_0 : vec4<f32> = chunks_0[_S20];
    var centre_0 : vec3<f32> = origin_4 + rotate_0(q_9, c0_0.xyz);
    var inv_mass_0 : f32 = bodies_0[b_4].w;
    var m_body_0 : f32;
    if(inv_mass_0 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_0;
    }
    else
    {
        m_body_0 = float_bits_0(u32(2139095040));
    }
    var from_0 : u32 = params_0.parity_0;
    var to_0 : u32 = u32(1) - params_0.parity_0;
    var _S21 : u32 = w_pulses_0() + _S20;
    var _S22 : u32 = _S21 + u32(2) * params_0.parity_0;
    var off_0 : u32 = work_0[_S22];
    var carried_0 : u32 = work_0[_S22 + u32(1)];
    var room_0 : u32 = work_0[_S22 + u32(1)] + sort_0[params_0.chunk_count_0 + u32(1) + k_0];
    var base_0 : u32;
    var room_1 : u32;
    if(room_0 != u32(0))
    {
        var _S23 : u32 = a_counters_0();
        var base_1 : u32 = atomicAdd(&(atom_0[_S23 + u32(5) + to_0]), room_0);
        if((base_1 + room_0) > (params_0.pool_cap_0))
        {
            var _S24 : u32 = atomicOr(&(atom_0[_S23 + u32(3)]), u32(8));
            if(base_1 < (params_0.pool_cap_0))
            {
                base_0 = params_0.pool_cap_0 - base_1;
            }
            else
            {
                base_0 = u32(0);
            }
        }
        else
        {
            base_0 = room_0;
        }
        var _S25 : u32 = base_0;
        base_0 = base_1;
        room_1 = _S25;
    }
    else
    {
        base_0 = u32(0);
        room_1 = room_0;
    }
    var _S26 : u32 = params_0.chunk_count_0;
    var _S27 : f32 = c0_0.w;
    var _S28 : vec4<f32> = chunks_0[_S20 + u32(3)];
    var _S29 : f32 = params_0.band_0;
    var _S30 : f32 = params_0.dt_0;
    var _S31 : bool = (params_0.impact_peak_0) != u32(0);
    var _S32 : u32 = l_pool_0(from_0) + u32(3) * off_0;
    var _S33 : u32 = l_pool_0(to_0) + u32(3) * base_0;
    loop
    {
        var _S34 : vec3<f32> = vec3<f32>(0.0f);
        var _S35 : u32 = sort_0[k_0];
        var _S36 : u32 = sort_0[_S26 + u32(1) + k_0];
        var first_0 : bool = true;
        var last_0 : u32 = u32(0);
        var step_1 : u32 = u32(0);
        f_total_0 = _S34;
        t_total_0 = _S34;
        loop
        {
            if(step_1 < _S36)
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
                if(q3_0 < _S36)
                {
                }
                else
                {
                    break;
                }
                var key_1 : u32 = manifolds_0[_S35 + q3_0].ids_0.y + u32(1);
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
            var r_4 : Record_std430_0 = manifolds_0[_S35 + p_0];
            var _S37 : vec3<f32> = r_4.impulse_0.xyz;
            var j_0 : f32 = length(_S37);
            if(j_0 == 0.0f)
            {
                var _S38 : u32 = step_1 + u32(1);
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S38;
                continue;
            }
            var m_other_0 : f32 = r_4.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_0))
                {
                    m_1 = _S27;
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
                    m_1 = m_body_0;
                }
                else
                {
                    m_1 = m_body_0 * m_other_0 / (m_body_0 + m_other_0);
                }
            }
            var _S39 : f32 = min(j_0, 2.0f * m_1 * max(r_4.impulse_0.w - _S29, 0.0f));
            if(_S39 < j_0)
            {
                var f_0 : vec3<f32> = _S37 / vec3<f32>(j_0) * vec3<f32>(((j_0 - _S39) / _S30));
                var _S40 : vec3<f32> = t_total_0 + cross(r_4.point_0.xyz - centre_0, f_0);
                f_total_1 = f_total_0 + f_0;
                t_total_1 = _S40;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            var _S38 : u32 = step_1 + u32(1);
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S38;
        }
        p_0 = u32(0);
        kept_0 = u32(0);
        loop
        {
            if(p_0 < carried_0)
            {
            }
            else
            {
                break;
            }
            var _S41 : u32 = _S32 + u32(3) * p_0;
            var a_1 : vec4<f32> = loads_0[_S41];
            var b_5 : vec4<f32> = loads_0[_S41 + u32(1)];
            var c_2 : vec4<f32> = loads_0[_S41 + u32(2)];
            var _S42 : f32 = loads_0[_S41 + u32(1)].w;
            var mean_0 : f32 = pulse_mean_0(loads_0[_S41].x, loads_0[_S41].y, loads_0[_S41].z, loads_0[_S41].w, _S42, _S30);
            if(mean_0 != 0.0f)
            {
                var f_1 : vec3<f32> = c_2.xyz * vec3<f32>(mean_0);
                var _S43 : vec3<f32> = t_total_0 + cross(b_5.xyz - centre_0, f_1);
                f_total_0 = f_total_0 + f_1;
                t_total_0 = _S43;
            }
            if((a_1.x + a_1.z + _S42) > _S30)
            {
                a_1[i32(0)] = a_1[i32(0)] - _S30;
                var _S44 : u32 = _S33 + u32(3) * kept_0;
                loads_0[_S44] = a_1;
                loads_0[_S44 + u32(1)] = b_5;
                loads_0[_S44 + u32(2)] = c_2;
                kept_0 = kept_0 + u32(1);
            }
            p_0 = p_0 + u32(1);
        }
        q3_0 = u32(0);
        overflow_0 = false;
        loop
        {
            if(q3_0 < _S36)
            {
            }
            else
            {
                break;
            }
            var r_5 : Record_std430_0 = manifolds_0[_S35 + q3_0];
            var _S45 : vec3<f32> = r_5.impulse_0.xyz;
            var j_1 : f32 = length(_S45);
            if(j_1 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var m_other_1 : f32 = r_5.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_1))
                {
                    m_1 = _S27;
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
                    m_1 = m_body_0;
                }
                else
                {
                    m_1 = m_body_0 * m_other_1 / (m_body_0 + m_other_1);
                }
            }
            var _S46 : f32 = r_5.impulse_0.w;
            var _S47 : f32 = min(j_1, 2.0f * m_1 * max(_S46 - _S29, 0.0f));
            if(_S47 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var _S48 : f32 = _S28.y;
            var duration_1 : f32 = hertz_duration_0(m_1, min(r_5.other_0.z, _S28.z), 1.0f / ((1.0f - _S48 * _S48) / _S28.x + 1.0f / r_5.other_0.y), _S46);
            var peak_1 : f32 = 3.14159274101257324f * _S47 / (2.0f * duration_1);
            var dir_0 : vec3<f32> = _S45 / vec3<f32>(j_1);
            var mean_1 : f32;
            if(_S31)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S30);
            }
            if(mean_1 != 0.0f)
            {
                var f_2 : vec3<f32> = dir_0 * vec3<f32>(mean_1);
                var _S49 : vec3<f32> = t_total_0 + cross(r_5.point_0.xyz - centre_0, f_2);
                f_total_1 = f_total_0 + f_2;
                t_total_1 = _S49;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            var kept_1 : u32;
            if(duration_1 > _S30)
            {
                if(kept_0 < room_1)
                {
                    var _S50 : u32 = _S33 + u32(3) * kept_0;
                    loads_0[_S50] = vec4<f32>(- _S30, 0.0f, 0.0f, peak_1);
                    loads_0[_S50 + u32(1)] = vec4<f32>(r_5.point_0.xyz, duration_1);
                    loads_0[_S50 + u32(2)] = vec4<f32>(dir_0, 0.0f);
                    kept_1 = kept_0 + u32(1);
                    overflow_1 = overflow_0;
                }
                else
                {
                    kept_1 = kept_0;
                    overflow_1 = true;
                }
            }
            else
            {
                kept_1 = kept_0;
                overflow_1 = overflow_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            kept_0 = kept_1;
            q3_0 = q3_0 + u32(1);
            overflow_0 = overflow_1;
        }
        break;
    }
    if(overflow_0)
    {
        var _S51 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(8));
    }
    var _S52 : u32 = _S21 + u32(2) * to_0;
    work_0[_S52] = base_0;
    work_0[_S52 + u32(1)] = kept_0;
    loads_0[k_0] = vec4<f32>(length(f_total_0), 0.0f, 0.0f, 0.0f);
    if((params_0.load_model_0) == u32(1))
    {
        var _S53 : u32 = l_vec_0() + u32(2) * k_0;
        loads_0[_S53] = vec4<f32>(f_total_0, 0.0f);
        loads_0[_S53 + u32(1)] = vec4<f32>(t_total_0, 0.0f);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_breaks(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var i_0 : u32 = id_1.x;
    var _S54 : bool;
    if(i_0 >= (params_0.bond_count_0))
    {
        _S54 = true;
    }
    else
    {
        _S54 = work_0[w_alive_0() + i_0] == u32(0);
    }
    if(_S54)
    {
        return;
    }
    var bond_0 : vec4<u32> = bonds_0[i_0];
    var _S55 : u32 = bond_0.x;
    if((max(loads_0[_S55].x, loads_0[bond_0.y].x)) > (bitcast<f32>((bond_0.z))))
    {
        work_0[w_alive_0() + i_0] = u32(0);
        var _S56 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S55)]), u32(1));
        var _S57 : u32 = a_counters_0();
        var _S58 : u32 = atomicAdd(&(atom_0[_S57]), u32(1));
        var _S59 : u32 = atomicAdd(&(atom_0[_S57 + u32(1)]), u32(1));
    }
    return;
}

fn groups_of_0( n_2 : u32,  size_0 : u32) -> u32
{
    var _S60 : u32 = (n_2 + size_0 - u32(1)) / size_0;
    return _S60;
}

fn set_args_0( at_0 : u32,  x_2 : u32)
{
    var _S61 : u32 = u32(3) * at_0;
    args_0[_S61] = x_2;
    args_0[_S61 + u32(1)] = u32(1);
    args_0[_S61 + u32(2)] = u32(1);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_gate(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var _S62 : u32 = a_counters_0();
    var _S63 : u32 = atomicLoad(&(atom_0[_S62]));
    var run_0 : bool = _S63 != u32(0);
    atomicStore(&(atom_0[_S62]), u32(0));
    atomicStore(&(atom_0[_S62 + u32(5) + params_0.parity_0]), u32(0));
    if(run_0)
    {
        var _S64 : u32 = atomicAdd(&(atom_0[_S62 + u32(4)]), u32(1));
    }
    var _S65 : u32;
    if(run_0)
    {
        var _S66 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S65 = _S66;
    }
    else
    {
        _S65 = u32(0);
    }
    set_args_0(u32(0), _S65);
    if(run_0)
    {
        var _S67 : u32 = groups_of_0(params_0.bond_count_0, u32(64));
        _S65 = _S67;
    }
    else
    {
        _S65 = u32(0);
    }
    set_args_0(u32(1), _S65);
    if(run_0)
    {
        var _S68 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S65 = _S68;
    }
    else
    {
        _S65 = u32(0);
    }
    set_args_0(u32(2), _S65);
    if(run_0)
    {
        var _S69 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S65 = _S69;
    }
    else
    {
        _S65 = u32(0);
    }
    set_args_0(u32(3), _S65);
    if(run_0)
    {
        var _S70 : u32 = groups_of_0(params_0.body_count_0, u32(1024));
        _S65 = _S70;
    }
    else
    {
        _S65 = u32(0);
    }
    set_args_0(u32(4), _S65);
    var _S71 : i32;
    if(run_0)
    {
        _S71 = i32(1);
    }
    else
    {
        _S71 = i32(0);
    }
    set_args_0(u32(5), u32(_S71));
    return;
}

fn dirty_0( body_0 : u32) -> bool
{
    var _S72 : u32 = atomicLoad(&(atom_0[a_dirty_0() + body_0]));
    return _S72 != u32(0);
}

fn a_parent_0() -> u32
{
    return u32(0);
}

fn a_count_0() -> u32
{
    return params_0.chunk_count_0;
}

fn a_anchor_0() -> u32
{
    return u32(2) * params_0.chunk_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_init(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var c_3 : u32 = id_3.x;
    var _S73 : bool;
    if(c_3 >= (params_0.chunk_count_0))
    {
        _S73 = true;
    }
    else
    {
        var _S74 : bool = dirty_0(body_of_0(c_3));
        _S73 = !_S74;
    }
    if(_S73)
    {
        return;
    }
    atomicStore(&(atom_0[a_parent_0() + c_3]), c_3);
    atomicStore(&(atom_0[a_count_0() + c_3]), u32(0));
    atomicStore(&(atom_0[a_anchor_0() + c_3]), u32(0));
    var w_2 : u32 = u32(0);
    loop
    {
        if(w_2 < u32(20))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[a_sums_0() + u32(20) * c_3 + w_2]), u32(0));
        w_2 = w_2 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_body_init(@builtin(global_invocation_id) id_4 : vec3<u32>)
{
    var s_0 : u32 = id_4.x;
    var _S75 : bool;
    if(s_0 >= (params_0.body_count_0))
    {
        _S75 = true;
    }
    else
    {
        var _S76 : bool = dirty_0(s_0);
        _S75 = !_S76;
    }
    if(_S75)
    {
        return;
    }
    atomicStore(&(atom_0[a_best_count_0() + s_0]), u32(0));
    atomicStore(&(atom_0[a_best_root_0() + s_0]), u32(4294967295));
    return;
}

fn find_0( x_3 : u32) -> u32
{
    var _S77 : u32 = a_parent_0();
    var _S78 : u32 = atomicLoad(&(atom_0[_S77 + x_3]));
    var p_1 : u32 = _S78;
    var _S79 : u32 = x_3;
    var guard_0 : u32 = u32(0);
    loop
    {
        var _S80 : bool;
        if(guard_0 < u32(1048576))
        {
            _S80 = p_1 != _S79;
        }
        else
        {
            _S80 = false;
        }
        if(_S80)
        {
        }
        else
        {
            break;
        }
        var gp_0 : u32 = atomicLoad(&(atom_0[_S77 + p_1]));
        if(gp_0 != p_1)
        {
            var _S81 : u32 = atomicCompareExchangeWeak(&(atom_0[_S77 + _S79]), p_1, gp_0).old_value;
        }
        var _S82 : u32 = guard_0 + u32(1);
        _S79 = p_1;
        p_1 = gp_0;
        guard_0 = _S82;
    }
    return _S79;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_union(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var i_1 : u32 = id_5.x;
    var _S83 : bool;
    if(i_1 >= (params_0.bond_count_0))
    {
        _S83 = true;
    }
    else
    {
        _S83 = work_0[w_alive_0() + i_1] == u32(0);
    }
    if(_S83)
    {
        return;
    }
    var bond_1 : vec4<u32> = bonds_0[i_1];
    var _S84 : u32 = bond_1.x;
    var _S85 : bool = dirty_0(body_of_0(_S84));
    if(!_S85)
    {
        return;
    }
    var _S86 : u32 = bond_1.y;
    var a_2 : u32 = _S84;
    var b_6 : u32 = _S86;
    var guard_1 : u32 = u32(0);
    loop
    {
        if(guard_1 < u32(4096))
        {
        }
        else
        {
            break;
        }
        var ra_0 : u32 = find_0(a_2);
        var rb_0 : u32 = find_0(b_6);
        if(ra_0 == rb_0)
        {
            return;
        }
        var _S87 : u32 = max(ra_0, rb_0);
        var _S88 : u32 = min(ra_0, rb_0);
        var was_0 : u32 = atomicCompareExchangeWeak(&(atom_0[a_parent_0() + _S87]), _S87, _S88).old_value;
        if(was_0 == _S87)
        {
            return;
        }
        var _S89 : u32 = guard_1 + u32(1);
        a_2 = _S87;
        b_6 = _S88;
        guard_1 = _S89;
    }
    var _S90 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(4));
    return;
}

fn add64_0( at_1 : u32,  value_0 : f32,  scale_0 : f32)
{
    var x_4 : f32 = round(value_0 * scale_0);
    var x_5 : f32;
    if(!((abs(x_4)) < 1.073741824e+09f))
    {
        var _S91 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
        x_5 = 0.0f;
    }
    else
    {
        x_5 = x_4;
    }
    var v_3 : i32 = i32(x_5);
    var lo_0 : u32 = (bitcast<u32>((v_3)));
    var old_0 : u32 = atomicAdd(&(atom_0[at_1]), lo_0);
    var carry_0 : u32;
    if((old_0 + lo_0) < old_0)
    {
        carry_0 = u32(1);
    }
    else
    {
        carry_0 = u32(0);
    }
    var _S92 : u32;
    if(v_3 < i32(0))
    {
        _S92 = u32(4294967295);
    }
    else
    {
        _S92 = u32(0);
    }
    var hi_0 : u32 = _S92 + carry_0;
    if(hi_0 != u32(0))
    {
        var _S93 : u32 = atomicAdd(&(atom_0[at_1 + u32(1)]), hi_0);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_flatten(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var c_4 : u32 = id_6.x;
    var _S94 : bool;
    if(c_4 >= (params_0.chunk_count_0))
    {
        _S94 = true;
    }
    else
    {
        var _S95 : bool = dirty_0(body_of_0(c_4));
        _S94 = !_S95;
    }
    if(_S94)
    {
        return;
    }
    var r_6 : u32 = find_0(c_4);
    work_0[w_label_0() + c_4] = r_6;
    var _S96 : u32 = atomicAdd(&(atom_0[a_count_0() + r_6]), u32(1));
    if(((info_0[c_4] & (u32(1)))) != u32(0))
    {
        var _S97 : u32 = atomicOr(&(atom_0[a_anchor_0() + r_6]), u32(1));
    }
    var _S98 : u32 = u32(4) * c_4;
    var c0_1 : vec4<f32> = chunks_0[_S98];
    var c1_0 : vec4<f32> = chunks_0[_S98 + u32(1)];
    var c2_0 : vec4<f32> = chunks_0[_S98 + u32(2)];
    var m_2 : f32 = c0_1.w;
    var d_0 : vec3<f32> = c0_1.xyz - chunks_0[u32(4) * r_6].xyz;
    var dd_0 : f32 = dot(d_0, d_0);
    var at_2 : u32 = a_sums_0() + u32(20) * r_6;
    add64_0(at_2, m_2, params_0.scale_m_0);
    var _S99 : f32 = d_0.x;
    var _S100 : f32 = m_2 * _S99;
    add64_0(at_2 + u32(2), _S100, params_0.scale_s_0);
    var _S101 : f32 = d_0.y;
    var _S102 : f32 = m_2 * _S101;
    add64_0(at_2 + u32(4), _S102, params_0.scale_s_0);
    var _S103 : f32 = d_0.z;
    add64_0(at_2 + u32(6), m_2 * _S103, params_0.scale_s_0);
    add64_0(at_2 + u32(8), c1_0.x + m_2 * (dd_0 - _S99 * _S99), params_0.scale_j_0);
    add64_0(at_2 + u32(10), c1_0.y + m_2 * (dd_0 - _S101 * _S101), params_0.scale_j_0);
    add64_0(at_2 + u32(12), c1_0.z + m_2 * (dd_0 - _S103 * _S103), params_0.scale_j_0);
    add64_0(at_2 + u32(14), c1_0.w - _S100 * _S101, params_0.scale_j_0);
    add64_0(at_2 + u32(16), c2_0.x - _S100 * _S103, params_0.scale_j_0);
    add64_0(at_2 + u32(18), c2_0.y - _S102 * _S103, params_0.scale_j_0);
    return;
}

fn is_root_0( c_5 : u32) -> bool
{
    var _S104 : bool;
    if(c_5 < (params_0.chunk_count_0))
    {
        var _S105 : bool = dirty_0(body_of_0(c_5));
        _S104 = _S105;
    }
    else
    {
        _S104 = false;
    }
    if(_S104)
    {
        _S104 = work_0[w_label_0() + c_5] == c_5;
    }
    else
    {
        _S104 = false;
    }
    return _S104;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_count(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var c_6 : u32 = id_7.x;
    var _S106 : bool = is_root_0(c_6);
    if(!_S106)
    {
        return;
    }
    var _S107 : u32 = atomicLoad(&(atom_0[a_count_0() + c_6]));
    var _S108 : u32 = atomicMax(&(atom_0[a_best_count_0() + body_of_0(c_6)]), _S107);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_root(@builtin(global_invocation_id) id_8 : vec3<u32>)
{
    var c_7 : u32 = id_8.x;
    var _S109 : bool = is_root_0(c_7);
    if(!_S109)
    {
        return;
    }
    var s_1 : u32 = body_of_0(c_7);
    var _S110 : u32 = atomicLoad(&(atom_0[a_count_0() + c_7]));
    var _S111 : u32 = atomicLoad(&(atom_0[a_best_count_0() + s_1]));
    if(_S110 == _S111)
    {
        var _S112 : u32 = atomicMin(&(atom_0[a_best_root_0() + s_1]), c_7);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_mark(@builtin(global_invocation_id) id_9 : vec3<u32>)
{
    var c_8 : u32 = id_9.x;
    if(c_8 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S113 : bool = is_root_0(c_8);
    var _S114 : bool;
    if(_S113)
    {
        var _S115 : u32 = atomicLoad(&(atom_0[a_best_root_0() + body_of_0(c_8)]));
        _S114 = _S115 != c_8;
    }
    else
    {
        _S114 = false;
    }
    var _S116 : u32;
    if(_S114)
    {
        _S116 = u32(1);
    }
    else
    {
        _S116 = u32(0);
    }
    work_0[w_new_0() + c_8] = _S116;
    return;
}

var<workgroup> g_scan_0 : array<u32, i32(1024)>;

@compute
@workgroup_size(256, 1, 1)
fn fr_scan_blocks(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var step_2 : u32;
    var add_0 : u32;
    var t_1 : u32 = thread_0.x;
    var _S117 : u32 = group_0.x;
    var _S118 : u32 = u32(1024) * _S117 + u32(4) * t_1;
    var v_4 : array<u32, i32(4)>;
    var j_2 : u32 = u32(0);
    var sum_2 : u32 = u32(0);
    loop
    {
        if(j_2 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S118 + j_2) < (scan_0.n_0))
        {
            step_2 = work_0[scan_0.in_base_0 + _S118 + j_2];
        }
        else
        {
            step_2 = u32(0);
        }
        v_4[j_2] = step_2;
        var sum_3 : u32 = sum_2 + step_2;
        j_2 = j_2 + u32(1);
        sum_2 = sum_3;
    }
    g_scan_0[t_1] = sum_2;
    workgroupBarrier();
    step_2 = u32(1);
    loop
    {
        if(step_2 < u32(256))
        {
        }
        else
        {
            break;
        }
        if(t_1 >= step_2)
        {
            add_0 = g_scan_0[t_1 - step_2];
        }
        else
        {
            add_0 = u32(0);
        }
        workgroupBarrier();
        g_scan_0[t_1] = g_scan_0[t_1] + add_0;
        workgroupBarrier();
        step_2 = (step_2 << (u32(1)));
    }
    var _S119 : u32 = g_scan_0[t_1] - sum_2;
    j_2 = u32(0);
    add_0 = _S119;
    loop
    {
        if(j_2 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S118 + j_2) < (scan_0.n_0))
        {
            work_0[scan_0.out_base_0 + _S118 + j_2] = add_0;
        }
        var run_1 : u32 = add_0 + v_4[j_2];
        j_2 = j_2 + u32(1);
        add_0 = run_1;
    }
    if(t_1 == u32(255))
    {
        work_0[scan_0.block_base_0 + _S117] = g_scan_0[i32(255)];
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn fr_scan_top(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var t_2 : u32 = thread_1.x;
    var blocks_0 : u32 = (scan_0.n_0 + u32(1023)) / u32(1024);
    var _S120 : bool = t_2 < blocks_0;
    var v_5 : u32;
    if(_S120)
    {
        v_5 = work_0[scan_0.block_base_0 + t_2];
    }
    else
    {
        v_5 = u32(0);
    }
    g_scan_0[t_2] = v_5;
    workgroupBarrier();
    var step_3 : u32 = u32(1);
    loop
    {
        if(step_3 < u32(1024))
        {
        }
        else
        {
            break;
        }
        var add_1 : u32;
        if(t_2 >= step_3)
        {
            add_1 = g_scan_0[t_2 - step_3];
        }
        else
        {
            add_1 = u32(0);
        }
        workgroupBarrier();
        g_scan_0[t_2] = g_scan_0[t_2] + add_1;
        workgroupBarrier();
        step_3 = (step_3 << (u32(1)));
    }
    if(_S120)
    {
        work_0[scan_0.block_base_0 + t_2] = g_scan_0[t_2] - v_5;
    }
    if(t_2 == u32(1023))
    {
        work_0[scan_0.block_base_0 + blocks_0] = g_scan_0[i32(1023)];
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_scan_add(@builtin(global_invocation_id) id_10 : vec3<u32>)
{
    var i_2 : u32 = id_10.x;
    if(i_2 >= (scan_0.n_0))
    {
        return;
    }
    work_0[scan_0.out_base_0 + i_2] = work_0[scan_0.out_base_0 + i_2] + work_0[scan_0.block_base_0 + i_2 / u32(1024)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_freelist(@builtin(global_invocation_id) id_11 : vec3<u32>)
{
    var s_2 : u32 = id_11.x;
    var _S121 : bool;
    if(s_2 >= (params_0.body_count_0))
    {
        _S121 = true;
    }
    else
    {
        _S121 = work_0[w_free_0() + s_2] == u32(0);
    }
    if(_S121)
    {
        return;
    }
    work_0[w_free_list_0() + work_0[w_free_rank_0() + s_2]] = s_2;
    return;
}

fn scan_new_blocks_0() -> u32
{
    return w_free_list_0() + params_0.body_count_0;
}

fn scan_free_blocks_0() -> u32
{
    return scan_new_blocks_0() + u32(1025);
}

fn free_total_0() -> u32
{
    return work_0[scan_free_blocks_0() + (params_0.body_count_0 + u32(1023)) / u32(1024)];
}

@compute
@workgroup_size(64, 1, 1)
fn fr_assign(@builtin(global_invocation_id) id_12 : vec3<u32>)
{
    var c_9 : u32 = id_12.x;
    var _S122 : bool = is_root_0(c_9);
    if(!_S122)
    {
        return;
    }
    var s_3 : u32 = body_of_0(c_9);
    var _S123 : u32 = atomicLoad(&(atom_0[a_best_root_0() + s_3]));
    var slot_5 : u32;
    if(_S123 != c_9)
    {
        var k_1 : u32 = work_0[w_rank_0() + c_9];
        if(work_0[w_rank_0() + c_9] < (free_total_0()))
        {
            var slot_6 : u32 = work_0[w_free_list_0() + k_1];
            work_0[w_free_0() + work_0[w_free_list_0() + k_1]] = u32(0);
            var _S124 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(2)]), u32(1));
            slot_5 = slot_6;
        }
        else
        {
            var _S125 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(1));
            slot_5 = s_3;
        }
    }
    else
    {
        slot_5 = s_3;
    }
    work_0[w_slot_0() + c_9] = slot_5;
    return;
}

fn read64_0( at_3 : u32,  scale_1 : f32) -> f32
{
    var _S126 : u32 = atomicLoad(&(atom_0[at_3 + u32(1)]));
    var hi_1 : i32 = (bitcast<i32>((_S126)));
    var lo_1 : u32 = atomicLoad(&(atom_0[at_3]));
    return (f32(hi_1) * 4.294967296e+09f + f32(lo_1)) / scale_1;
}

fn quat_from_columns_0( c0_2 : vec3<f32>,  c1_1 : vec3<f32>,  c2_1 : vec3<f32>) -> Quat_0
{
    var m00_0 : f32 = c0_2.x;
    var m01_0 : f32 = c0_2.y;
    var m02_0 : f32 = c0_2.z;
    var m10_0 : f32 = c1_1.x;
    var m11_0 : f32 = c1_1.y;
    var m12_0 : f32 = c1_1.z;
    var m20_0 : f32 = c2_1.x;
    var m21_0 : f32 = c2_1.y;
    var m22_0 : f32 = c2_1.z;
    var q_10 : Quat_0;
    if(m22_0 <= 0.0f)
    {
        var dif10_0 : f32 = m11_0 - m00_0;
        var omm22_0 : f32 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            var f_3 : f32 = omm22_0 - dif10_0;
            var inv_0 : f32 = 0.5f / sqrt(f_3);
            q_10.x_1 = f_3 * inv_0;
            q_10.y_0 = (m01_0 + m10_0) * inv_0;
            q_10.z_0 = (m02_0 + m20_0) * inv_0;
            q_10.w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            var f_4 : f32 = omm22_0 + dif10_0;
            var inv_1 : f32 = 0.5f / sqrt(f_4);
            q_10.x_1 = (m01_0 + m10_0) * inv_1;
            q_10.y_0 = f_4 * inv_1;
            q_10.z_0 = (m12_0 + m21_0) * inv_1;
            q_10.w_0 = (m20_0 - m02_0) * inv_1;
        }
    }
    else
    {
        var sum10_0 : f32 = m11_0 + m00_0;
        var opm22_0 : f32 = 1.0f + m22_0;
        if(sum10_0 <= 0.0f)
        {
            var f_5 : f32 = opm22_0 - sum10_0;
            var inv_2 : f32 = 0.5f / sqrt(f_5);
            q_10.x_1 = (m02_0 + m20_0) * inv_2;
            q_10.y_0 = (m12_0 + m21_0) * inv_2;
            q_10.z_0 = f_5 * inv_2;
            q_10.w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            var f_6 : f32 = opm22_0 + sum10_0;
            var inv_3 : f32 = 0.5f / sqrt(f_6);
            q_10.x_1 = (m12_0 - m21_0) * inv_3;
            q_10.y_0 = (m20_0 - m02_0) * inv_3;
            q_10.z_0 = (m01_0 - m10_0) * inv_3;
            q_10.w_0 = f_6 * inv_3;
        }
    }
    return q_10;
}

fn principal_axes_0( r0_0 : vec3<f32>,  r1_0 : vec3<f32>,  r2_0 : vec3<f32>,  rot_0 : ptr<function, Quat_0>,  moments_0 : ptr<function, vec3<f32>>)
{
    var _S127 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S128 : array<f32, i32(3)> = array<f32, i32(3)>( r1_0.x, r1_0.y, r1_0.z );
    var _S129 : array<f32, i32(3)> = array<f32, i32(3)>( r2_0.x, r2_0.y, r2_0.z );
    var a_3 : array<array<f32, i32(3)>, i32(3)>;
    a_3[i32(0)] = _S127;
    a_3[i32(1)] = _S128;
    a_3[i32(2)] = _S129;
    var _S130 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S131 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S132 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_6 : array<array<f32, i32(3)>, i32(3)>;
    v_6[i32(0)] = _S130;
    v_6[i32(1)] = _S131;
    v_6[i32(2)] = _S132;
    var it_0 : u32 = u32(0);
    loop
    {
        if(it_0 < u32(32))
        {
        }
        else
        {
            break;
        }
        var q_11 : u32;
        if((abs(a_3[i32(0)][i32(2)])) > (abs(a_3[u32(0)][u32(1)])))
        {
            q_11 = u32(2);
        }
        else
        {
            q_11 = u32(1);
        }
        var p_2 : u32;
        var q_12 : u32;
        if((abs(a_3[i32(1)][i32(2)])) > (abs(a_3[u32(0)][q_11])))
        {
            p_2 = u32(1);
            q_12 = u32(2);
        }
        else
        {
            p_2 = u32(0);
            q_12 = q_11;
        }
        if((abs(a_3[p_2][q_12])) <= (9.99999982451670044e-15f * max(max(abs(a_3[i32(0)][i32(0)]), abs(a_3[i32(1)][i32(1)])), max(abs(a_3[i32(2)][i32(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        var theta_0 : f32 = 0.5f * atan2(2.0f * a_3[p_2][q_12], a_3[q_12][q_12] - a_3[p_2][p_2]);
        var s_4 : f32 = sin(theta_0);
        var c_10 : f32 = cos(theta_0);
        var app_0 : f32 = a_3[p_2][p_2];
        var aqq_0 : f32 = a_3[q_12][q_12];
        var apq_0 : f32 = a_3[p_2][q_12];
        var k_2 : u32 = u32(0);
        loop
        {
            if(k_2 < u32(3))
            {
            }
            else
            {
                break;
            }
            var _S133 : bool;
            if(k_2 != p_2)
            {
                _S133 = k_2 != q_12;
            }
            else
            {
                _S133 = false;
            }
            if(_S133)
            {
                var kp_0 : f32 = a_3[k_2][p_2];
                var kq_0 : f32 = a_3[k_2][q_12];
                var _S134 : f32 = c_10 * a_3[k_2][p_2] - s_4 * a_3[k_2][q_12];
                a_3[k_2][p_2] = _S134;
                a_3[p_2][k_2] = _S134;
                var _S135 : f32 = s_4 * kp_0 + c_10 * kq_0;
                a_3[k_2][q_12] = _S135;
                a_3[q_12][k_2] = _S135;
            }
            k_2 = k_2 + u32(1);
        }
        var _S136 : f32 = c_10 * c_10;
        var _S137 : f32 = 2.0f * s_4 * c_10 * apq_0;
        var _S138 : f32 = s_4 * s_4;
        a_3[p_2][p_2] = _S136 * app_0 - _S137 + _S138 * aqq_0;
        a_3[q_12][q_12] = _S138 * app_0 + _S137 + _S136 * aqq_0;
        a_3[p_2][q_12] = 0.0f;
        a_3[q_12][p_2] = 0.0f;
        var row_0 : u32 = u32(0);
        loop
        {
            if(row_0 < u32(3))
            {
            }
            else
            {
                break;
            }
            var kp_1 : f32 = v_6[row_0][p_2];
            var kq_1 : f32 = v_6[row_0][q_12];
            v_6[row_0][p_2] = c_10 * v_6[row_0][p_2] - s_4 * v_6[row_0][q_12];
            v_6[row_0][q_12] = s_4 * kp_1 + c_10 * kq_1;
            row_0 = row_0 + u32(1);
        }
        it_0 = it_0 + u32(1);
    }
    var x_6 : vec3<f32> = vec3<f32>(v_6[i32(0)][i32(0)], v_6[i32(1)][i32(0)], v_6[i32(2)][i32(0)]);
    var y_1 : vec3<f32> = vec3<f32>(v_6[i32(0)][i32(1)], v_6[i32(1)][i32(1)], v_6[i32(2)][i32(1)]);
    var z_1 : vec3<f32> = vec3<f32>(v_6[i32(0)][i32(2)], v_6[i32(1)][i32(2)], v_6[i32(2)][i32(2)]);
    var z_2 : vec3<f32>;
    if((dot(cross(x_6, y_1), z_1)) < 0.0f)
    {
        z_2 = (vec3<f32>(0) - z_1);
    }
    else
    {
        z_2 = z_1;
    }
    (*rot_0) = normalized_0(quat_from_columns_0(x_6, y_1, z_2));
    (*moments_0) = vec3<f32>(a_3[i32(0)][i32(0)], a_3[i32(1)][i32(1)], a_3[i32(2)][i32(2)]);
    return;
}

fn to_rubble_0( v_7 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_7.x, v_7.z, - v_7.y);
}

fn rot_to_rubble_0( q_13 : Quat_0) -> Quat_0
{
    var axes_1 : Quat_0;
    axes_1.w_0 = 0.70710676908493042f;
    axes_1.x_1 = -0.70710676908493042f;
    axes_1.y_0 = 0.0f;
    axes_1.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_1, q_13));
}

fn stage_0( c_11 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * c_11;
}

fn quat_vec_0( q_14 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_14.x_1, q_14.y_0, q_14.z_0, q_14.w_0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_piece(@builtin(global_invocation_id) id_13 : vec3<u32>)
{
    var c_12 : u32 = id_13.x;
    var _S139 : bool = is_root_0(c_12);
    if(!_S139)
    {
        return;
    }
    var parent_0 : u32 = body_of_0(c_12);
    var pb_0 : u32 = u32(7) * parent_0;
    var at_4 : u32 = a_sums_0() + u32(20) * c_12;
    var mass_0 : f32 = read64_0(at_4, params_0.scale_m_0);
    var _S140 : f32 = read64_0(at_4 + u32(2), params_0.scale_s_0);
    var _S141 : f32 = read64_0(at_4 + u32(4), params_0.scale_s_0);
    var _S142 : f32 = read64_0(at_4 + u32(6), params_0.scale_s_0);
    var d_1 : vec3<f32> = vec3<f32>(_S140, _S141, _S142) / vec3<f32>(max(mass_0, 1.00000000317107685e-30f));
    var jxx_0 : f32 = read64_0(at_4 + u32(8), params_0.scale_j_0);
    var jyy_0 : f32 = read64_0(at_4 + u32(10), params_0.scale_j_0);
    var jzz_0 : f32 = read64_0(at_4 + u32(12), params_0.scale_j_0);
    var jxy_0 : f32 = read64_0(at_4 + u32(14), params_0.scale_j_0);
    var jxz_0 : f32 = read64_0(at_4 + u32(16), params_0.scale_j_0);
    var jyz_0 : f32 = read64_0(at_4 + u32(18), params_0.scale_j_0);
    var dd_1 : f32 = dot(d_1, d_1);
    var _S143 : f32 = d_1.x;
    var _S144 : f32 = mass_0 * _S143;
    var _S145 : f32 = d_1.y;
    var _S146 : f32 = jxy_0 + _S144 * _S145;
    var _S147 : f32 = d_1.z;
    var _S148 : f32 = jxz_0 + _S144 * _S147;
    var _S149 : f32 = jyz_0 + mass_0 * _S145 * _S147;
    var com_4 : vec3<f32> = chunks_0[u32(4) * c_12].xyz + d_1;
    var principal_0 : Quat_0;
    var moments_1 : vec3<f32>;
    principal_axes_0(vec3<f32>(jxx_0 - mass_0 * (dd_1 - _S143 * _S143), _S146, _S148), vec3<f32>(_S146, jyy_0 - mass_0 * (dd_1 - _S145 * _S145), _S149), vec3<f32>(_S148, _S149, jzz_0 - mass_0 * (dd_1 - _S147 * _S147)), &(principal_0), &(moments_1));
    var _S150 : u32 = atomicLoad(&(atom_0[a_anchor_0() + c_12]));
    var anchored_0 : bool = _S150 != u32(0);
    var q_15 : Quat_0;
    var origin_5 : vec3<f32>;
    structure_pose_3(parent_0, pb_0, &(q_15), &(origin_5));
    var p_parent_0 : vec3<f32> = bodies_0[pb_0].xyz;
    var v_parent_0 : vec3<f32> = bodies_0[pb_0 + u32(3)].xyz;
    var w_parent_0 : vec3<f32> = bodies_0[pb_0 + u32(4)].xyz;
    var pos_0 : vec3<f32> = to_rubble_0(origin_5 + rotate_0(q_15, com_4));
    var orientation_0 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(q_15, principal_0)));
    var _S151 : bool;
    if(anchored_0)
    {
        _S151 = true;
    }
    else
    {
        _S151 = !(mass_0 > 0.0f);
    }
    var inv_mass_1 : f32;
    if(_S151)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / mass_0;
    }
    var _S152 : vec3<f32> = vec3<f32>(0.0f);
    var inv_inertia_0 : vec3<f32>;
    if(!anchored_0)
    {
        var _S153 : f32;
        if((moments_1.x) > 0.0f)
        {
            _S153 = 1.0f / moments_1.x;
        }
        else
        {
            _S153 = 0.0f;
        }
        var _S154 : f32;
        if((moments_1.y) > 0.0f)
        {
            _S154 = 1.0f / moments_1.y;
        }
        else
        {
            _S154 = 0.0f;
        }
        var _S155 : f32;
        if((moments_1.z) > 0.0f)
        {
            _S155 = 1.0f / moments_1.z;
        }
        else
        {
            _S155 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S153, _S154, _S155);
    }
    else
    {
        inv_inertia_0 = _S152;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_0)
    {
        velocity_0 = _S152;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    var spin_0 : vec3<f32>;
    if(anchored_0)
    {
        spin_0 = _S152;
    }
    else
    {
        spin_0 = w_parent_0;
    }
    var flags_0 : u32;
    if(anchored_0)
    {
        flags_0 = u32(0);
    }
    else
    {
        flags_0 = u32(9);
    }
    var st_0 : u32 = stage_0(c_12);
    frames_0[st_0] = vec4<f32>(pos_0, inv_mass_1);
    frames_0[st_0 + u32(1)] = quat_vec_0(orientation_0);
    frames_0[st_0 + u32(2)] = vec4<f32>(inv_inertia_0, (bitcast<f32>((flags_0))));
    frames_0[st_0 + u32(3)] = vec4<f32>(velocity_0, 0.0f);
    frames_0[st_0 + u32(4)] = vec4<f32>(spin_0, (bitcast<f32>((params_0.step_0))));
    frames_0[st_0 + u32(5)] = vec4<f32>(0.0f);
    frames_0[st_0 + u32(6)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    frames_0[st_0 + u32(7)] = vec4<f32>(com_4, frames_0[u32(2) * parent_0].w);
    frames_0[st_0 + u32(8)] = quat_vec_0(principal_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_commit(@builtin(global_invocation_id) id_14 : vec3<u32>)
{
    var c_13 : u32 = id_14.x;
    var _S156 : bool = is_root_0(c_13);
    if(!_S156)
    {
        return;
    }
    var slot_7 : u32 = work_0[w_slot_0() + c_13];
    var st_1 : u32 = stage_0(c_13);
    var region_0 : u32 = u32(0);
    loop
    {
        if(region_0 < u32(2))
        {
        }
        else
        {
            break;
        }
        var w_3 : u32 = u32(0);
        loop
        {
            if(w_3 < u32(7))
            {
            }
            else
            {
                break;
            }
            bodies_0[u32(7) * (slot_7 + region_0 * params_0.max_bodies_0) + w_3] = frames_0[st_1 + w_3];
            w_3 = w_3 + u32(1);
        }
        region_0 = region_0 + u32(1);
    }
    var _S157 : u32 = u32(2) * slot_7;
    frames_0[_S157] = frames_0[st_1 + u32(7)];
    frames_0[_S157 + u32(1)] = frames_0[st_1 + u32(8)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_reparent(@builtin(global_invocation_id) id_15 : vec3<u32>)
{
    var c_14 : u32 = id_15.x;
    var _S158 : bool;
    if(c_14 >= (params_0.chunk_count_0))
    {
        _S158 = true;
    }
    else
    {
        var _S159 : bool = dirty_0(body_of_0(c_14));
        _S158 = !_S159;
    }
    if(_S158)
    {
        return;
    }
    var r_7 : u32 = work_0[w_label_0() + c_14];
    var slot_8 : u32 = work_0[w_slot_0() + work_0[w_label_0() + c_14]];
    work_0[w_body_0() + c_14] = work_0[w_slot_0() + work_0[w_label_0() + c_14]];
    var _S160 : u32 = u32(2) * slot_8;
    var _S161 : vec3<f32> = frames_0[_S160].xyz;
    var _S162 : Quat_0 = conj_0(quat_of_0(frames_0[_S160 + u32(1)]));
    var inv_mass_2 : f32 = bodies_0[u32(7) * slot_8].w;
    var _S163 : f32;
    if(inv_mass_2 > 0.0f)
    {
        _S163 = 1.0f / inv_mass_2;
    }
    else
    {
        _S163 = float_bits_0(u32(2139095040));
    }
    var c2_2 : vec4<f32> = chunks_0[u32(4) * c_14 + u32(2)];
    var first_1 : u32 = (bitcast<u32>((c2_2.z)));
    var _S164 : u32 = (bitcast<u32>((c2_2.w)));
    var _S165 : vec4<f32> = chunks_0[u32(4) * r_7 + u32(3)];
    var p_3 : u32 = first_1;
    loop
    {
        if(p_3 < (first_1 + _S164))
        {
        }
        else
        {
            break;
        }
        var _S166 : u32 = u32(2) * p_3;
        var o_1 : vec4<f32> = parts_0[_S166];
        var col_0 : u32 = (bitcast<u32>((o_1.w)));
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S166 + u32(1)]);
        var _S167 : u32 = u32(4) * col_0;
        colliders_0[_S167] = vec4<f32>((bitcast<f32>((slot_8))), colliders_0[_S167].y, colliders_0[_S167].z, colliders_0[_S167].w);
        var _S168 : u32 = _S167 + u32(1);
        colliders_0[_S168] = vec4<f32>(rotate_0(_S162, o_1.xyz - _S161), colliders_0[_S168].w);
        colliders_0[_S167 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S162, rot_1)));
        var _S169 : u32 = u32(2) * col_0;
        collider_info_0[_S169] = vec4<u32>(collider_info_0[_S169].x, slot_8, collider_info_0[_S169].z, collider_info_0[_S169].w);
        collider_info_0[_S169 + u32(1)] = vec4<u32>((bitcast<u32>((_S163))), (bitcast<u32>((_S165.x))), (bitcast<u32>((_S165.w))), u32(0));
        p_3 = p_3 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_body_clear(@builtin(global_invocation_id) id_16 : vec3<u32>)
{
    var s_5 : u32 = id_16.x;
    if(s_5 >= (params_0.body_count_0))
    {
        return;
    }
    atomicStore(&(atom_0[a_dirty_0() + s_5]), u32(0));
    return;
}

fn slot_color_0( slot_9 : u32) -> vec3<f32>
{
    var h_0 : f32 = fract(f32(slot_9) * 0.6180340051651001f) * 6.0f;
    var f_7 : f32 = fract(h_0);
    var q_16 : f32 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_7);
    var t_3 : f32 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_7));
    var k_3 : u32 = u32(h_0);
    var _S170 : vec3<f32>;
    if(k_3 == u32(0))
    {
        _S170 = vec3<f32>(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_3 == u32(1))
        {
            _S170 = vec3<f32>(q_16, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_3 == u32(2))
            {
                _S170 = vec3<f32>(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_3 == u32(3))
                {
                    _S170 = vec3<f32>(0.34959998726844788f, q_16, 0.92000001668930054f);
                }
                else
                {
                    if(k_3 == u32(4))
                    {
                        _S170 = vec3<f32>(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S170 = vec3<f32>(0.92000001668930054f, 0.34959998726844788f, q_16);
                    }
                }
            }
        }
    }
    return _S170;
}

fn write_object_0( o_2 : u32,  q_17 : Quat_0,  t_4 : vec3<f32>,  color_0 : vec4<f32>)
{
    var x_7 : vec3<f32> = rotate_0(q_17, vec3<f32>(1.0f, 0.0f, 0.0f));
    var y_2 : vec3<f32> = rotate_0(q_17, vec3<f32>(0.0f, 1.0f, 0.0f));
    var z_3 : vec3<f32> = rotate_0(q_17, vec3<f32>(0.0f, 0.0f, 1.0f));
    var _S171 : u32 = u32(4) * o_2;
    objects_0[_S171] = vec4<f32>(x_7.x, y_2.x, z_3.x, t_4.x);
    objects_0[_S171 + u32(1)] = vec4<f32>(x_7.y, y_2.y, z_3.y, t_4.y);
    objects_0[_S171 + u32(2)] = vec4<f32>(x_7.z, y_2.z, z_3.z, t_4.z);
    objects_0[_S171 + u32(3)] = color_0;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_chunks(@builtin(global_invocation_id) id_17 : vec3<u32>)
{
    var k_4 : u32 = id_17.x;
    if(k_4 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_10 : u32 = body_of_0(k_4);
    var b_7 : u32 = u32(7) * slot_10;
    var q_18 : Quat_0;
    var origin_6 : vec3<f32>;
    structure_pose_3(slot_10, b_7, &(q_18), &(origin_6));
    var _S172 : u32 = u32(3) * k_4;
    var v0_0 : vec4<f32> = view_chunks_0[_S172];
    var base_2 : vec3<f32> = view_chunks_0[_S172 + u32(2)].xyz;
    var fixed_0 : bool = (bodies_0[b_7].w) == 0.0f;
    var color_1 : vec3<f32>;
    if((view_0.mode_0) == u32(1))
    {
        var _S173 : f32 = max(length(bodies_0[b_7 + u32(3)].xyz), length(bodies_0[b_7 + u32(4)].xyz));
        if(fixed_0)
        {
            color_1 = vec3<f32>(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S173 > (view_0.rest_speed_0))
            {
                color_1 = vec3<f32>(1.0f, 0.44999998807907104f, 0.11999999731779099f);
            }
            else
            {
                color_1 = vec3<f32>(0.25f, 0.44999998807907104f, 0.85000002384185791f);
            }
        }
    }
    else
    {
        if((view_0.mode_0) == u32(2))
        {
            color_1 = slot_color_0(slot_10);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = vec3<f32>(0.44999998807907104f) * base_2 + vec3<f32>(0.55000001192092896f) * slot_color_0(slot_10);
            }
        }
    }
    write_object_0((bitcast<u32>((v0_0.w))), normalized_0(quat_mul_0(q_18, quat_of_0(view_chunks_0[_S172 + u32(1)]))), origin_6 + rotate_0(q_18, v0_0.xyz), vec4<f32>(color_1, 1.0f));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_shots(@builtin(global_invocation_id) id_18 : vec3<u32>)
{
    var i_3 : u32 = id_18.x;
    if(i_3 >= (view_0.shot_count_1))
    {
        return;
    }
    var b_8 : u32 = u32(7) * shot_bodies_0[i_3];
    var o_3 : u32 = view_0.impactor_base_0 + i_3;
    if((bodies_0[b_8].w) == 0.0f)
    {
        var w_4 : u32 = u32(0);
        loop
        {
            if(w_4 < u32(4))
            {
            }
            else
            {
                break;
            }
            objects_0[u32(4) * o_3 + w_4] = vec4<f32>(0.0f);
            w_4 = w_4 + u32(1);
        }
        return;
    }
    write_object_0(o_3, rot_from_rubble_0(quat_of_0(bodies_0[b_8 + u32(1)])), from_rubble_0(bodies_0[b_8].xyz), vec4<f32>(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f));
    return;
}

fn chunk_world_0( k_5 : u32) -> vec3<f32>
{
    var slot_11 : u32 = body_of_0(k_5);
    var q_19 : Quat_0;
    var origin_7 : vec3<f32>;
    structure_pose_1(slot_11, u32(7) * slot_11, &(q_19), &(origin_7));
    return origin_7 + rotate_0(q_19, view_chunks_0[u32(3) * k_5].xyz);
}

fn write_line_vertex_0( v_8 : u32,  p_4 : vec3<f32>)
{
    var _S174 : u32 = u32(7) * v_8;
    lines_0[_S174] = p_4.x;
    lines_0[_S174 + u32(1)] = p_4.y;
    lines_0[_S174 + u32(2)] = p_4.z;
    lines_0[_S174 + u32(3)] = 1.0f;
    lines_0[_S174 + u32(4)] = 0.86000001430511475f;
    lines_0[_S174 + u32(5)] = 0.25f;
    lines_0[_S174 + u32(6)] = 1.0f;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_bonds(@builtin(global_invocation_id) id_19 : vec3<u32>)
{
    var i_4 : u32 = id_19.x;
    if(i_4 >= (params_0.bond_count_0))
    {
        return;
    }
    var bond_2 : vec4<u32> = bonds_0[i_4];
    var a_4 : vec3<f32> = chunk_world_0(bond_2.x);
    var b_9 : vec3<f32>;
    if(work_0[w_alive_0() + i_4] != u32(0))
    {
        b_9 = chunk_world_0(bond_2.y);
    }
    else
    {
        b_9 = a_4;
    }
    var _S175 : u32 = u32(2) * i_4;
    write_line_vertex_0(_S175, a_4);
    write_line_vertex_0(_S175 + u32(1), b_9);
    return;
}

fn a_strike_0() -> u32
{
    return a_counters_0() + u32(16);
}

fn strike_at_0( b_10 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * params_0.chunk_count_0 + u32(3) * b_10;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_reset(@builtin(global_invocation_id) id_20 : vec3<u32>)
{
    var b_11 : u32 = id_20.x;
    if(b_11 >= (params_0.body_count_0))
    {
        return;
    }
    var at_5 : u32 = a_strike_0() + u32(9) * b_11;
    var w_5 : u32 = u32(0);
    loop
    {
        if(w_5 < u32(3))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[at_5 + w_5]), u32(4294967295));
        atomicStore(&(atom_0[at_5 + u32(3) + w_5]), u32(0));
        w_5 = w_5 + u32(1);
    }
    atomicStore(&(atom_0[at_5 + u32(6)]), u32(2139095040));
    atomicStore(&(atom_0[at_5 + u32(7)]), u32(2139095040));
    atomicStore(&(atom_0[at_5 + u32(8)]), u32(4294967295));
    frames_0[strike_at_0(b_11)][i32(3)] = 0.0f;
    return;
}

fn chunk_box_world_0( k_6 : u32,  q_20 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_12 : u32 = body_of_0(k_6);
    var origin_8 : vec3<f32>;
    structure_pose_2(slot_12, u32(7) * slot_12, &((*q_20)), &(origin_8));
    return origin_8 + rotate_0((*q_20), geometry_0[u32(3) * k_6].xyz);
}

fn chunk_box_world_1( k_7 : u32,  q_21 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_13 : u32 = body_of_0(k_7);
    var origin_9 : vec3<f32>;
    structure_pose_3(slot_13, u32(7) * slot_13, &((*q_21)), &(origin_9));
    return origin_9 + rotate_0((*q_21), geometry_0[u32(3) * k_7].xyz);
}

fn ordered_0( f_8 : f32) -> u32
{
    var u_0 : u32 = (bitcast<u32>((f_8)));
    var _S176 : u32;
    if(((u_0 & (u32(2147483648)))) != u32(0))
    {
        _S176 = ~u_0;
    }
    else
    {
        _S176 = (u_0 | (u32(2147483648)));
    }
    return _S176;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_bounds(@builtin(global_invocation_id) id_21 : vec3<u32>)
{
    var k_8 : u32 = id_21.x;
    var _S177 : bool;
    if(k_8 >= (params_0.chunk_count_0))
    {
        _S177 = true;
    }
    else
    {
        _S177 = (params_0.shot_count_0) == u32(0);
    }
    if(_S177)
    {
        return;
    }
    var q_22 : Quat_0;
    var _S178 : vec3<f32> = chunk_box_world_1(k_8, &(q_22));
    var _S179 : f32 = geometry_0[u32(3) * k_8].w;
    var _S180 : u32 = a_strike_0() + u32(9) * body_of_0(k_8);
    var ax_0 : u32 = u32(0);
    loop
    {
        if(ax_0 < u32(3))
        {
        }
        else
        {
            break;
        }
        var _S181 : u32 = ax_0;
        var _S182 : u32 = atomicMin(&(atom_0[_S180 + ax_0]), ordered_0(_S178[ax_0] - _S179));
        var _S183 : u32 = atomicMax(&(atom_0[_S180 + u32(3) + ax_0]), ordered_0(_S178[_S181] + _S179));
        ax_0 = ax_0 + u32(1);
    }
    return;
}

fn unordered_0( u_1 : u32) -> f32
{
    var _S184 : u32;
    if(((u_1 & (u32(2147483648)))) != u32(0))
    {
        _S184 = (u_1 & (u32(2147483647)));
    }
    else
    {
        _S184 = ~u_1;
    }
    return (bitcast<f32>((_S184)));
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_find(@builtin(global_invocation_id) id_22 : vec3<u32>)
{
    var b_12 : u32 = id_22.x;
    var _S185 : bool;
    if(b_12 >= (params_0.body_count_0))
    {
        _S185 = true;
    }
    else
    {
        _S185 = (params_0.shot_count_0) == u32(0);
    }
    if(_S185)
    {
        return;
    }
    var at_6 : u32 = a_strike_0() + u32(9) * b_12;
    var _S186 : u32 = atomicLoad(&(atom_0[at_6]));
    if(_S186 == u32(4294967295))
    {
        return;
    }
    var _S187 : u32 = atomicLoad(&(atom_0[at_6]));
    var _S188 : f32 = unordered_0(_S187);
    var _S189 : u32 = atomicLoad(&(atom_0[at_6 + u32(1)]));
    var _S190 : f32 = unordered_0(_S189);
    var _S191 : u32 = atomicLoad(&(atom_0[at_6 + u32(2)]));
    var _S192 : vec3<f32> = vec3<f32>(_S188, _S190, unordered_0(_S191));
    var _S193 : u32 = atomicLoad(&(atom_0[at_6 + u32(3)]));
    var _S194 : f32 = unordered_0(_S193);
    var _S195 : u32 = atomicLoad(&(atom_0[at_6 + u32(4)]));
    var _S196 : f32 = unordered_0(_S195);
    var _S197 : u32 = atomicLoad(&(atom_0[at_6 + u32(5)]));
    var _S198 : vec3<f32> = vec3<f32>(_S194, _S196, unordered_0(_S197));
    var _S199 : vec3<f32> = from_rubble_0(bodies_0[u32(7) * b_12 + u32(3)].xyz);
    var _S200 : f32 = float_bits_0(u32(2139095040));
    var k_9 : u32 = u32(0);
    loop
    {
        if(k_9 < (params_0.shot_count_0))
        {
        }
        else
        {
            break;
        }
        var info_1 : vec4<f32> = shots_0[u32(2) * k_9];
        var sb_0 : u32 = u32(7) * (bitcast<u32>((info_1.x)));
        if((bodies_0[sb_0].w) == 0.0f)
        {
            k_9 = k_9 + u32(1);
            continue;
        }
        var c_15 : vec3<f32> = from_rubble_0(bodies_0[sb_0].xyz);
        var rel_0 : vec3<f32> = from_rubble_0(bodies_0[sb_0 + u32(3)].xyz) - _S199;
        if(!((length(rel_0)) > (params_0.strike_speed_0)))
        {
            k_9 = k_9 + u32(1);
            continue;
        }
        var _S201 : vec3<f32> = vec3<f32>(info_1.y);
        var _S202 : vec3<f32> = _S192 - _S201;
        var _S203 : vec3<f32> = _S198 + _S201;
        var t0_0 : f32 = 0.0f;
        var t1_0 : f32 = params_0.dt_0;
        var ax_1 : u32 = u32(0);
        loop
        {
            if(ax_1 < u32(3))
            {
            }
            else
            {
                break;
            }
            var _S204 : u32 = ax_1;
            if((abs(rel_0[ax_1])) < 9.999999960041972e-13f)
            {
                var _S205 : u32 = ax_1;
                if((c_15[ax_1]) < (_S202[ax_1]))
                {
                    _S185 = true;
                }
                else
                {
                    _S185 = (c_15[_S205]) > (_S203[ax_1]);
                }
                var t0_1 : f32;
                if(_S185)
                {
                    t0_1 = _S200;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_1 = ax_1 + u32(1);
                continue;
            }
            var ta_0 : f32 = (_S202[ax_1] - c_15[ax_1]) / rel_0[_S204];
            var tb_0 : f32 = (_S203[ax_1] - c_15[ax_1]) / rel_0[_S204];
            var _S206 : f32 = min(t1_0, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_0 = _S206;
            ax_1 = ax_1 + u32(1);
        }
        if(t0_0 <= t1_0)
        {
            var st_2 : u32 = strike_at_0(b_12);
            frames_0[st_2] = vec4<f32>(c_15, f32(k_9 + u32(1)));
            frames_0[st_2 + u32(1)] = vec4<f32>(rel_0, t0_0);
            return;
        }
        k_9 = k_9 + u32(1);
    }
    return;
}

fn struck_0( b_13 : u32,  shot_0 : ptr<function, u32>) -> bool
{
    var k_10 : u32 = u32(frames_0[strike_at_0(b_13)].w);
    (*shot_0) = k_10 - u32(1);
    return k_10 != u32(0);
}

fn struck_1( b_14 : u32,  shot_1 : ptr<function, u32>) -> bool
{
    var k_11 : u32 = u32(frames_0[strike_at_0(b_14)].w);
    (*shot_1) = k_11 - u32(1);
    return k_11 != u32(0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_entry(@builtin(global_invocation_id) id_23 : vec3<u32>)
{
    var k_12 : u32 = id_23.x;
    var _S207 : bool;
    if(k_12 >= (params_0.chunk_count_0))
    {
        _S207 = true;
    }
    else
    {
        _S207 = (params_0.shot_count_0) == u32(0);
    }
    if(_S207)
    {
        return;
    }
    var b_15 : u32 = body_of_0(k_12);
    var shot_2 : u32;
    var _S208 : bool = struck_1(b_15, &(shot_2));
    if(!_S208)
    {
        return;
    }
    var _S209 : u32 = strike_at_0(b_15);
    var s1_0 : vec4<f32> = frames_0[_S209 + u32(1)];
    var from_1 : vec3<f32> = frames_0[_S209].xyz;
    var rel_1 : vec3<f32> = frames_0[_S209 + u32(1)].xyz;
    var d_2 : vec3<f32> = rel_1 / vec3<f32>(max(length(rel_1), 9.999999960041972e-13f));
    var _S210 : f32 = shots_0[u32(2) * shot_2].y;
    var q_23 : Quat_0;
    var cc_0 : vec3<f32> = chunk_box_world_1(k_12, &(q_23));
    var _S211 : u32 = u32(3) * k_12;
    var _S212 : Quat_0 = normalized_0(quat_mul_0(q_23, quat_of_0(geometry_0[_S211 + u32(2)])));
    var _S213 : vec3<f32> = geometry_0[_S211 + u32(1)].xyz;
    var _S214 : f32 = float_bits_0(u32(2139095040));
    var t_in_0 : f32 = 0.0f;
    var t_out_0 : f32 = params_0.dt_0;
    var ax_2 : u32 = u32(0);
    loop
    {
        if(ax_2 < u32(3))
        {
        }
        else
        {
            break;
        }
        var e_1 : vec3<f32> = vec3<f32>(0.0f);
        e_1[ax_2] = 1.0f;
        var axis_0 : vec3<f32> = rotate_0(_S212, e_1);
        var o_4 : f32 = dot(from_1 - cc_0, axis_0);
        var dv_0 : f32 = dot(rel_1, axis_0);
        var ext_1 : f32 = _S213[ax_2] + _S210;
        if((abs(dv_0)) < 9.999999960041972e-13f)
        {
            var t_in_1 : f32;
            if((abs(o_4)) > ext_1)
            {
                t_in_1 = _S214;
            }
            else
            {
                t_in_1 = t_in_0;
            }
            t_in_0 = t_in_1;
            ax_2 = ax_2 + u32(1);
            continue;
        }
        var ta_1 : f32 = (- ext_1 - o_4) / dv_0;
        var tb_1 : f32 = (ext_1 - o_4) / dv_0;
        var _S215 : f32 = min(t_out_0, max(ta_1, tb_1));
        t_in_0 = max(t_in_0, min(ta_1, tb_1));
        t_out_0 = _S215;
        ax_2 = ax_2 + u32(1);
    }
    var at_7 : u32 = a_strike_0() + u32(9) * b_15;
    if(t_in_0 <= t_out_0)
    {
        var _S216 : u32 = atomicMin(&(atom_0[at_7 + u32(6)]), (bitcast<u32>((t_in_0))));
    }
    var r_8 : vec3<f32> = cc_0 - (from_1 + rel_1 * vec3<f32>(s1_0.w));
    var along_0 : f32 = dot(r_8, d_2);
    var _S217 : u32 = atomicMin(&(atom_0[at_7 + u32(7)]), (bitcast<u32>((length(r_8 - d_2 * vec3<f32>(along_0)) + abs(along_0) * 0.00100000004749745f))));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_nearest(@builtin(global_invocation_id) id_24 : vec3<u32>)
{
    var k_13 : u32 = id_24.x;
    var _S218 : bool;
    if(k_13 >= (params_0.chunk_count_0))
    {
        _S218 = true;
    }
    else
    {
        _S218 = (params_0.shot_count_0) == u32(0);
    }
    if(_S218)
    {
        return;
    }
    var b_16 : u32 = body_of_0(k_13);
    var shot_3 : u32;
    if(!struck_1(b_16, &(shot_3)))
    {
        return;
    }
    var _S219 : u32 = strike_at_0(b_16);
    var _S220 : vec3<f32> = frames_0[_S219 + u32(1)].xyz;
    var d_3 : vec3<f32> = _S220 / vec3<f32>(max(length(_S220), 9.999999960041972e-13f));
    var q_24 : Quat_0;
    var r_9 : vec3<f32> = chunk_box_world_1(k_13, &(q_24)) - (frames_0[_S219].xyz + _S220 * vec3<f32>(frames_0[_S219 + u32(1)].w));
    var along_1 : f32 = dot(r_9, d_3);
    var at_8 : u32 = a_strike_0() + u32(9) * b_16;
    var _S221 : u32 = (bitcast<u32>((length(r_9 - d_3 * vec3<f32>(along_1)) + abs(along_1) * 0.00100000004749745f)));
    var _S222 : u32 = atomicLoad(&(atom_0[at_8 + u32(7)]));
    if(_S221 == _S222)
    {
        var _S223 : u32 = atomicMin(&(atom_0[at_8 + u32(8)]), k_13);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_force(@builtin(global_invocation_id) id_25 : vec3<u32>)
{
    var b_17 : u32 = id_25.x;
    var _S224 : bool;
    if(b_17 >= (params_0.body_count_0))
    {
        _S224 = true;
    }
    else
    {
        _S224 = (params_0.shot_count_0) == u32(0);
    }
    if(_S224)
    {
        return;
    }
    var shot_4 : u32;
    var _S225 : bool = struck_1(b_17, &(shot_4));
    if(!_S225)
    {
        return;
    }
    var at_9 : u32 = a_strike_0() + u32(9) * b_17;
    var _S226 : u32 = atomicLoad(&(atom_0[at_9 + u32(6)]));
    var t_in_2 : f32 = (bitcast<f32>((_S226)));
    var entry_0 : u32 = atomicLoad(&(atom_0[at_9 + u32(8)]));
    var st_3 : u32 = strike_at_0(b_17);
    if(!(t_in_2 <= (params_0.dt_0)))
    {
        _S224 = true;
    }
    else
    {
        _S224 = entry_0 == u32(4294967295);
    }
    if(_S224)
    {
        frames_0[st_3][i32(3)] = 0.0f;
        return;
    }
    var rel_2 : vec3<f32> = frames_0[st_3 + u32(1)].xyz;
    var speed_0 : f32 = length(rel_2);
    var info_2 : vec4<f32> = shots_0[u32(2) * shot_4];
    var _S227 : f32 = max(info_2.z, 0.00100000004749745f);
    var m_imp_0 : f32 = info_2.w;
    var e_imp_0 : f32 = shots_0[u32(2) * shot_4 + u32(1)].x;
    var contact_0 : vec3<f32> = frames_0[st_3].xyz + rel_2 * vec3<f32>(t_in_2) + rel_2 / vec3<f32>(max(speed_0, 9.999999960041972e-13f)) * vec3<f32>(info_2.y);
    var inv_mass_3 : f32 = bodies_0[u32(7) * b_17].w;
    var anchored_1 : bool = inv_mass_3 == 0.0f;
    var m_body_1 : f32;
    if(anchored_1)
    {
        m_body_1 = 0.0f;
    }
    else
    {
        m_body_1 = 1.0f / inv_mass_3;
    }
    var m_eff_0 : f32;
    if(anchored_1)
    {
        m_eff_0 = m_imp_0;
    }
    else
    {
        m_eff_0 = m_imp_0 * m_body_1 / (m_imp_0 + m_body_1);
    }
    var modulus_0 : f32 = 1.0f / (1.0f / e_imp_0 + 1.0f / geometry_0[u32(3) * entry_0 + u32(1)].w);
    frames_0[st_3 + u32(2)] = vec4<f32>(contact_0, m_eff_0 * speed_0 / (2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (_S227 * modulus_0 * modulus_0 * max(speed_0, 0.00100000004749745f)), 0.20000000298023224f)));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_breaks(@builtin(global_invocation_id) id_26 : vec3<u32>)
{
    var i_5 : u32 = id_26.x;
    var _S228 : bool;
    if(i_5 >= (params_0.bond_count_0))
    {
        _S228 = true;
    }
    else
    {
        _S228 = (params_0.shot_count_0) == u32(0);
    }
    if(_S228)
    {
        _S228 = true;
    }
    else
    {
        _S228 = work_0[w_alive_0() + i_5] == u32(0);
    }
    if(_S228)
    {
        return;
    }
    var bond_3 : vec4<u32> = bonds_0[i_5];
    var _S229 : u32 = bond_3.x;
    var b_18 : u32 = body_of_0(_S229);
    var shot_5 : u32;
    if(!struck_1(b_18, &(shot_5)))
    {
        return;
    }
    var q_25 : Quat_0;
    var pa_0 : vec3<f32> = chunk_box_world_1(_S229, &(q_25));
    var pb_1 : vec3<f32> = chunk_box_world_1(bond_3.y, &(q_25));
    var dist_0 : f32 = length(vec3<f32>(0.5f) * (pa_0 + pb_1) - frames_0[strike_at_0(b_18) + u32(2)].xyz);
    if((frames_0[strike_at_0(b_18) + u32(2)].w * min(1.0f, (bitcast<f32>((bond_3.w))) / (6.28318548202514648f * max(dist_0 * dist_0, 9.999999960041972e-13f)))) > (bitcast<f32>((bond_3.z))))
    {
        work_0[w_alive_0() + i_5] = u32(0);
        var _S230 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_18]), u32(1));
        var _S231 : u32 = a_counters_0();
        var _S232 : u32 = atomicAdd(&(atom_0[_S231]), u32(1));
        var _S233 : u32 = atomicAdd(&(atom_0[_S231 + u32(1)]), u32(1));
    }
    return;
}

fn a_lp_0( b_19 : u32) -> u32
{
    return a_strike_0() + u32(9) * params_0.body_count_0 + u32(14) * b_19;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_clear(@builtin(global_invocation_id) id_27 : vec3<u32>)
{
    var b_20 : u32 = id_27.x;
    if(b_20 >= (params_0.body_count_0))
    {
        return;
    }
    var at_10 : u32 = a_lp_0(b_20);
    var w_6 : u32 = u32(0);
    loop
    {
        if(w_6 < u32(12))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[at_10 + w_6]), u32(0));
        w_6 = w_6 + u32(1);
    }
    atomicStore(&(atom_0[at_10 + u32(12)]), u32(4294967295));
    atomicStore(&(atom_0[at_10 + u32(13)]), u32(0));
    return;
}

fn lp_f_0( k_14 : u32) -> u32
{
    return u32(14) * k_14 + u32(6);
}

fn add64w_0( at_11 : u32,  value_1 : f32,  scale_2 : f32)
{
    var x_8 : f32 = round(value_1 * scale_2);
    var x_9 : f32;
    if(!((abs(x_8)) < 3.99999993722699776e+18f))
    {
        var _S234 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
        x_9 = 0.0f;
    }
    else
    {
        x_9 = x_8;
    }
    var u_2 : f32 = abs(x_9);
    var h_1 : f32 = floor(u_2 * 2.32830643653869629e-10f);
    var lo_2 : u32 = u32(u_2 - h_1 * 4.294967296e+09f);
    var hi_2 : u32 = u32(h_1);
    var lo_3 : u32;
    var hi_3 : u32;
    if(x_9 < 0.0f)
    {
        var lo_4 : u32 = ~lo_2 + u32(1);
        var _S235 : u32 = ~hi_2;
        if(lo_4 == u32(0))
        {
            lo_3 = u32(1);
        }
        else
        {
            lo_3 = u32(0);
        }
        var _S236 : u32 = _S235 + lo_3;
        lo_3 = lo_4;
        hi_3 = _S236;
    }
    else
    {
        lo_3 = lo_2;
        hi_3 = hi_2;
    }
    var old_1 : u32 = atomicAdd(&(atom_0[at_11]), lo_3);
    var carry_1 : u32;
    if((old_1 + lo_3) < old_1)
    {
        carry_1 = u32(1);
    }
    else
    {
        carry_1 = u32(0);
    }
    var hi_4 : u32 = hi_3 + carry_1;
    if(hi_4 != u32(0))
    {
        var _S237 : u32 = atomicAdd(&(atom_0[at_11 + u32(1)]), hi_4);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sum(@builtin(global_invocation_id) id_28 : vec3<u32>)
{
    var k_15 : u32 = id_28.x;
    if(k_15 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_14 : u32 = body_of_0(k_15);
    var q_26 : Quat_0;
    var origin_10 : vec3<f32>;
    structure_pose_3(slot_14, u32(7) * slot_14, &(q_26), &(origin_10));
    var qi_0 : Quat_0 = conj_0(q_26);
    var c0_3 : vec4<f32> = chunks_0[u32(4) * k_15];
    var _S238 : u32 = l_vec_0() + u32(2) * k_15;
    var f_9 : vec3<f32> = rotate_0(qi_0, loads_0[_S238].xyz) + vec3<f32>(c0_3.w) * rotate_0(qi_0, vec3<f32>(params_0.gx_0, params_0.gy_0, params_0.gz_0));
    var t_5 : vec3<f32> = rotate_0(qi_0, loads_0[_S238 + u32(1)].xyz);
    var _S239 : u32 = lp_f_0(k_15);
    lp_0[_S239] = vec4<f32>(f_9, 0.0f);
    lp_0[_S239 + u32(1)] = vec4<f32>(t_5, 0.0f);
    var _S240 : vec3<f32> = t_5 + cross(c0_3.xyz - frames_0[u32(2) * slot_14].xyz, f_9);
    var at_12 : u32 = a_lp_0(slot_14);
    var a_5 : u32 = u32(0);
    loop
    {
        if(a_5 < u32(3))
        {
        }
        else
        {
            break;
        }
        var _S241 : u32 = u32(2) * a_5;
        add64w_0(at_12 + _S241, f_9[a_5], params_0.lp_scale_0);
        add64w_0(at_12 + u32(6) + _S241, _S240[a_5], params_0.lp_scale_0);
        a_5 = a_5 + u32(1);
    }
    var _S242 : u32 = atomicMin(&(atom_0[at_12 + u32(12)]), k_15);
    if(((info_0[k_15] & (u32(1)))) != u32(0))
    {
        atomicStore(&(atom_0[at_12 + u32(13)]), u32(1));
    }
    return;
}

fn read64w3_0( at_13 : u32,  scale_3 : f32) -> vec3<f32>
{
    var _S243 : f32 = read64_0(at_13, scale_3);
    var _S244 : f32 = read64_0(at_13 + u32(2), scale_3);
    var _S245 : f32 = read64_0(at_13 + u32(4), scale_3);
    return vec3<f32>(_S243, _S244, _S245);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_relief(@builtin(global_invocation_id) id_29 : vec3<u32>)
{
    var k_16 : u32 = id_29.x;
    if(k_16 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_15 : u32 = body_of_0(k_16);
    var at_14 : u32 = a_lp_0(slot_15);
    var _S246 : u32 = atomicLoad(&(atom_0[at_14 + u32(13)]));
    var supported_0 : bool = _S246 != u32(0);
    var fixed_1 : bool;
    if(((info_0[k_16] & (u32(1)))) != u32(0))
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            var _S247 : u32 = atomicLoad(&(atom_0[at_14 + u32(12)]));
            fixed_1 = _S247 == k_16;
        }
        else
        {
            fixed_1 = false;
        }
    }
    var _S248 : u32 = lp_f_0(k_16);
    var f_10 : vec4<f32> = lp_0[_S248];
    var _S249 : u32 = _S248 + u32(1);
    var t_6 : vec4<f32> = lp_0[_S249];
    var b_21 : u32 = u32(7) * slot_15;
    var inv_mass_4 : f32 = bodies_0[b_21].w;
    var _S250 : bool;
    if(!supported_0)
    {
        _S250 = inv_mass_4 > 0.0f;
    }
    else
    {
        _S250 = false;
    }
    if(_S250)
    {
        var force_0 : vec3<f32> = read64w3_0(at_14, params_0.lp_scale_0);
        var moment_0 : vec3<f32> = read64w3_0(at_14 + u32(6), params_0.lp_scale_0);
        var _S251 : u32 = u32(2) * slot_15;
        var p_5 : Quat_0 = quat_of_0(frames_0[_S251 + u32(1)]);
        var alpha_0 : vec3<f32> = rotate_0(p_5, bodies_0[b_21 + u32(2)].xyz * rotate_0(conj_0(p_5), moment_0));
        var _S252 : u32 = u32(4) * k_16;
        var c0_4 : vec4<f32> = chunks_0[_S252];
        var c1_2 : vec4<f32> = chunks_0[_S252 + u32(1)];
        var c2_3 : vec4<f32> = chunks_0[_S252 + u32(2)];
        var _S253 : vec3<f32> = f_10.xyz - vec3<f32>(c0_4.w) * (force_0 * vec3<f32>(inv_mass_4) + cross(alpha_0, c0_4.xyz - frames_0[_S251].xyz));
        f_10.x = _S253.x;
        f_10.y = _S253.y;
        f_10.z = _S253.z;
        var _S254 : f32 = alpha_0.x;
        var _S255 : f32 = c1_2.w;
        var _S256 : f32 = alpha_0.y;
        var _S257 : f32 = c2_3.x;
        var _S258 : f32 = alpha_0.z;
        var _S259 : f32 = c2_3.y;
        var _S260 : vec3<f32> = t_6.xyz - vec3<f32>(c1_2.x * _S254 + _S255 * _S256 + _S257 * _S258, _S255 * _S254 + c1_2.y * _S256 + _S259 * _S258, _S257 * _S254 + _S259 * _S256 + c1_2.z * _S258);
        t_6.x = _S260.x;
        t_6.y = _S260.y;
        t_6.z = _S260.z;
    }
    var _S261 : f32;
    if(fixed_1)
    {
        _S261 = 1.0f;
    }
    else
    {
        _S261 = 0.0f;
    }
    f_10[i32(3)] = _S261;
    lp_0[_S248] = f_10;
    lp_0[_S249] = t_6;
    return;
}

fn adj_bond_0( at_15 : u32) -> u32
{
    return (bitcast<u32>((ext_0[params_0.ext_adj_0 + at_15 / u32(4)][at_15 % u32(4)])));
}

struct JointBond_0
{
     geom0_0 : vec4<f32>,
     geom1_0 : vec4<f32>,
     stiff0_0 : vec4<f32>,
     stiff1_0 : vec4<f32>,
     rebar0_0 : vec4<f32>,
     rebar1_0 : vec4<f32>,
     ids_1 : vec4<u32>,
};

struct LpBond_0
{
     centroid_0 : vec3<f32>,
     n_3 : vec3<f32>,
     t1_1 : vec3<f32>,
     t2_0 : vec3<f32>,
     law_0 : JointBond_0,
};

fn lp_bond_0( i_6 : u32) -> LpBond_0
{
    var o_5 : u32 = u32(10) * i_6;
    var b_22 : LpBond_0;
    b_22.centroid_0 = ext_0[o_5].xyz;
    var _S262 : vec3<f32> = ext_0[o_5 + u32(1)].xyz;
    b_22.n_3 = _S262;
    var _S263 : vec3<f32> = ext_0[o_5 + u32(2)].xyz;
    b_22.t1_1 = _S263;
    b_22.t2_0 = cross(_S262, _S263);
    b_22.law_0.geom0_0 = ext_0[o_5 + u32(3)];
    b_22.law_0.geom1_0 = ext_0[o_5 + u32(4)];
    b_22.law_0.stiff0_0 = ext_0[o_5 + u32(5)];
    b_22.law_0.stiff1_0 = ext_0[o_5 + u32(6)];
    b_22.law_0.rebar0_0 = ext_0[o_5 + u32(7)];
    b_22.law_0.rebar1_0 = ext_0[o_5 + u32(8)];
    b_22.law_0.ids_1 = (bitcast<vec4<u32>>((ext_0[o_5 + u32(9)])));
    return b_22;
}

fn tri_0( i_7 : u32,  j_3 : u32) -> u32
{
    return i_7 * (i_7 + u32(1)) / u32(2) + j_3;
}

fn lp_l_0( k_17 : u32) -> u32
{
    return u32(14) * k_17 + u32(8);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_block(@builtin(global_invocation_id) id_30 : vec3<u32>)
{
    var q_27 : u32;
    var a_6 : u32;
    var p2_0 : u32;
    var q2_0 : u32;
    var a2_0 : u32;
    var s_6 : f32;
    var k_18 : u32 = id_30.x;
    var ok_0 : bool;
    if(k_18 >= (params_0.chunk_count_0))
    {
        ok_0 = true;
    }
    else
    {
        ok_0 = (lp_0[lp_f_0(k_18)].w) != 0.0f;
    }
    if(ok_0)
    {
        return;
    }
    var d_4 : array<array<f32, i32(6)>, i32(6)>;
    var p_6 : u32 = u32(0);
    loop
    {
        if(p_6 < u32(6))
        {
        }
        else
        {
            break;
        }
        q_27 = u32(0);
        loop
        {
            if(q_27 < u32(6))
            {
            }
            else
            {
                break;
            }
            d_4[p_6][q_27] = 0.0f;
            q_27 = q_27 + u32(1);
        }
        p_6 = p_6 + u32(1);
    }
    var _S264 : u32 = u32(4) * k_18;
    var range_0 : vec4<f32> = ext_0[params_0.ext_chunks_0 + _S264];
    var _S265 : u32 = (bitcast<u32>((range_0.x)));
    var _S266 : u32 = (bitcast<u32>((range_0.y)));
    q_27 = u32(0);
    loop
    {
        if(q_27 < _S266)
        {
        }
        else
        {
            break;
        }
        var i_8 : u32 = adj_bond_0(_S265 + q_27);
        if(work_0[w_alive_0() + i_8] == u32(0))
        {
            q_27 = q_27 + u32(1);
            continue;
        }
        var b_23 : LpBond_0 = lp_bond_0(i_8);
        var _S267 : vec3<f32> = b_23.centroid_0 - chunks_0[_S264].xyz;
        var _S268 : f32 = b_23.law_0.stiff0_0.y;
        var _S269 : array<f32, i32(6)> = array<f32, i32(6)>( _S268, _S268, b_23.law_0.stiff0_0.x, b_23.law_0.stiff0_0.z, b_23.law_0.stiff0_0.w, b_23.law_0.stiff1_0.x );
        var _S270 : array<vec3<f32>, i32(3)> = array<vec3<f32>, i32(3)>( b_23.t1_1, b_23.t2_0, b_23.n_3 );
        var j_4 : array<array<f32, i32(6)>, i32(6)>;
        a_6 = u32(0);
        loop
        {
            if(a_6 < u32(3))
            {
            }
            else
            {
                break;
            }
            var c_16 : vec3<f32> = cross(_S267, _S270[a_6]);
            var _S271 : f32 = _S270[a_6].x;
            j_4[a_6][i32(0)] = _S271;
            var _S272 : f32 = _S270[a_6].y;
            j_4[a_6][i32(1)] = _S272;
            var _S273 : f32 = _S270[a_6].z;
            j_4[a_6][i32(2)] = _S273;
            j_4[a_6][i32(3)] = c_16.x;
            j_4[a_6][i32(4)] = c_16.y;
            j_4[a_6][i32(5)] = c_16.z;
            j_4[u32(3) + a_6][i32(0)] = 0.0f;
            j_4[u32(3) + a_6][i32(1)] = 0.0f;
            j_4[u32(3) + a_6][i32(2)] = 0.0f;
            j_4[u32(3) + a_6][i32(3)] = _S271;
            j_4[u32(3) + a_6][i32(4)] = _S272;
            j_4[u32(3) + a_6][i32(5)] = _S273;
            a_6 = a_6 + u32(1);
        }
        p2_0 = u32(0);
        loop
        {
            if(p2_0 < u32(6))
            {
            }
            else
            {
                break;
            }
            q2_0 = u32(0);
            loop
            {
                if(q2_0 < u32(6))
                {
                }
                else
                {
                    break;
                }
                a2_0 = u32(0);
                s_6 = 0.0f;
                loop
                {
                    if(a2_0 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var s_7 : f32 = s_6 + j_4[a2_0][p2_0] * _S269[a2_0] * j_4[a2_0][q2_0];
                    a2_0 = a2_0 + u32(1);
                    s_6 = s_7;
                }
                d_4[p2_0][q2_0] = d_4[p2_0][q2_0] + s_6;
                q2_0 = q2_0 + u32(1);
            }
            p2_0 = p2_0 + u32(1);
        }
        q_27 = q_27 + u32(1);
    }
    var l_0 : array<f32, i32(21)>;
    a_6 = u32(0);
    loop
    {
        if(a_6 < u32(21))
        {
        }
        else
        {
            break;
        }
        l_0[a_6] = 0.0f;
        a_6 = a_6 + u32(1);
    }
    ok_0 = true;
    p2_0 = u32(0);
    loop
    {
        if(p2_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        var ok_1 : bool = ok_0;
        q2_0 = u32(0);
        loop
        {
            if(q2_0 <= p2_0)
            {
            }
            else
            {
                break;
            }
            a2_0 = u32(0);
            s_6 = d_4[p2_0][q2_0];
            loop
            {
                if(a2_0 < q2_0)
                {
                }
                else
                {
                    break;
                }
                var s_8 : f32 = s_6 - l_0[tri_0(p2_0, a2_0)] * l_0[tri_0(q2_0, a2_0)];
                a2_0 = a2_0 + u32(1);
                s_6 = s_8;
            }
            if(p2_0 == q2_0)
            {
                var _S274 : bool;
                if(ok_1)
                {
                    _S274 = s_6 > 0.0f;
                }
                else
                {
                    _S274 = false;
                }
                l_0[tri_0(p2_0, p2_0)] = sqrt(max(s_6, 1.00000000317107685e-30f));
                ok_1 = _S274;
            }
            else
            {
                l_0[tri_0(p2_0, q2_0)] = s_6 / l_0[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + u32(1);
        }
        var _S275 : u32 = p2_0 + u32(1);
        ok_0 = ok_1;
        p2_0 = _S275;
    }
    if(!ok_0)
    {
        lp_0[lp_f_0(k_18)][i32(3)] = 1.0f;
        var _S276 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(32));
        return;
    }
    q2_0 = u32(0);
    loop
    {
        if(q2_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S277 : u32 = u32(4) * q2_0;
        var _S278 : f32 = l_0[_S277];
        var _S279 : u32 = _S277 + u32(1);
        if(_S279 < u32(21))
        {
            s_6 = l_0[_S279];
        }
        else
        {
            s_6 = 0.0f;
        }
        var _S280 : u32 = _S277 + u32(2);
        var _S281 : f32;
        if(_S280 < u32(21))
        {
            _S281 = l_0[_S280];
        }
        else
        {
            _S281 = 0.0f;
        }
        var _S282 : u32 = _S277 + u32(3);
        var _S283 : f32;
        if(_S282 < u32(21))
        {
            _S283 = l_0[_S282];
        }
        else
        {
            _S283 = 0.0f;
        }
        lp_0[lp_l_0(k_18) + q2_0] = vec4<f32>(_S278, s_6, _S281, _S283);
        q2_0 = q2_0 + u32(1);
    }
    return;
}

fn lp_x_0( k_19 : u32,  copy_0 : u32) -> u32
{
    return u32(14) * k_19 + u32(2) * copy_0;
}

fn lp_d_0( k_20 : u32) -> u32
{
    return u32(14) * k_20 + u32(4);
}

fn lp_force_0( b_24 : LpBond_0,  ka_0 : u32,  kb_0 : u32,  c_17 : u32,  q_lin_0 : ptr<function, vec3<f32>>,  q_ang_0 : ptr<function, vec3<f32>>)
{
    var _S284 : u32 = lp_x_0(ka_0, c_17);
    var _S285 : u32 = lp_x_0(kb_0, c_17);
    var _S286 : vec3<f32> = lp_0[_S285 + u32(1)].xyz;
    var _S287 : vec3<f32> = lp_0[_S284 + u32(1)].xyz;
    var delta_0 : vec3<f32> = lp_0[_S285].xyz + cross(_S286, b_24.centroid_0 - chunks_0[u32(4) * kb_0].xyz) - (lp_0[_S284].xyz + cross(_S287, b_24.centroid_0 - chunks_0[u32(4) * ka_0].xyz));
    var phi_0 : vec3<f32> = _S286 - _S287;
    var kt_0 : f32 = b_24.law_0.stiff1_0.x;
    var _S288 : f32 = b_24.law_0.stiff0_0.y;
    (*q_lin_0) = vec3<f32>(_S288 * dot(delta_0, b_24.t1_1), _S288 * dot(delta_0, b_24.t2_0), b_24.law_0.stiff0_0.x * dot(delta_0, b_24.n_3));
    (*q_ang_0) = vec3<f32>(b_24.law_0.stiff0_0.z * dot(phi_0, b_24.t1_1), b_24.law_0.stiff0_0.w * dot(phi_0, b_24.t2_0), kt_0 * dot(phi_0, b_24.n_3));
    return;
}

fn lp_force_1( b_25 : LpBond_0,  ka_1 : u32,  kb_1 : u32,  c_18 : u32,  q_lin_1 : ptr<function, vec3<f32>>,  q_ang_1 : ptr<function, vec3<f32>>)
{
    var _S289 : u32 = lp_x_0(ka_1, c_18);
    var _S290 : u32 = lp_x_0(kb_1, c_18);
    var _S291 : vec3<f32> = lp_0[_S290 + u32(1)].xyz;
    var _S292 : vec3<f32> = lp_0[_S289 + u32(1)].xyz;
    var delta_1 : vec3<f32> = lp_0[_S290].xyz + cross(_S291, b_25.centroid_0 - chunks_0[u32(4) * kb_1].xyz) - (lp_0[_S289].xyz + cross(_S292, b_25.centroid_0 - chunks_0[u32(4) * ka_1].xyz));
    var phi_1 : vec3<f32> = _S291 - _S292;
    var kt_1 : f32 = b_25.law_0.stiff1_0.x;
    var _S293 : f32 = b_25.law_0.stiff0_0.y;
    (*q_lin_1) = vec3<f32>(_S293 * dot(delta_1, b_25.t1_1), _S293 * dot(delta_1, b_25.t2_0), b_25.law_0.stiff0_0.x * dot(delta_1, b_25.n_3));
    (*q_ang_1) = vec3<f32>(b_25.law_0.stiff0_0.z * dot(phi_1, b_25.t1_1), b_25.law_0.stiff0_0.w * dot(phi_1, b_25.t2_0), kt_1 * dot(phi_1, b_25.n_3));
    return;
}

fn lp_sweep_0( k_21 : u32,  from_2 : u32)
{
    var s_9 : f32;
    var to_1 : u32 = u32(1) - from_2;
    var _S294 : u32 = lp_x_0(k_21, from_2);
    var x0_0 : vec4<f32> = lp_0[_S294];
    var x1_0 : vec4<f32> = lp_0[_S294 + u32(1)];
    var _S295 : u32 = lp_f_0(k_21);
    var f_11 : vec4<f32> = lp_0[_S295];
    if((lp_0[_S295].w) != 0.0f)
    {
        var _S296 : u32 = lp_x_0(k_21, to_1);
        lp_0[_S296] = x0_0;
        lp_0[_S296 + u32(1)] = x1_0;
        var _S297 : u32 = lp_d_0(k_21);
        var _S298 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S297] = _S298;
        lp_0[_S297 + u32(1)] = _S298;
        return;
    }
    var _S299 : f32 = f_11.y;
    var _S300 : f32 = f_11.z;
    var r_10 : array<f32, i32(6)>;
    r_10[i32(0)] = f_11.x;
    r_10[i32(1)] = _S299;
    r_10[i32(2)] = _S300;
    r_10[i32(3)] = 0.0f;
    r_10[i32(4)] = 0.0f;
    r_10[i32(5)] = 0.0f;
    var t_7 : vec3<f32> = lp_0[_S295 + u32(1)].xyz;
    r_10[i32(3)] = t_7.x;
    r_10[i32(4)] = t_7.y;
    r_10[i32(5)] = t_7.z;
    var _S301 : u32 = u32(4) * k_21;
    var range_1 : vec4<f32> = ext_0[params_0.ext_chunks_0 + _S301];
    var _S302 : u32 = (bitcast<u32>((range_1.x)));
    var _S303 : u32 = (bitcast<u32>((range_1.y)));
    var e_2 : u32 = u32(0);
    loop
    {
        if(e_2 < _S303)
        {
        }
        else
        {
            break;
        }
        var i_9 : u32 = adj_bond_0(_S302 + e_2);
        if(work_0[w_alive_0() + i_9] == u32(0))
        {
            e_2 = e_2 + u32(1);
            continue;
        }
        var bond_4 : vec4<u32> = bonds_0[i_9];
        var b_26 : LpBond_0 = lp_bond_0(i_9);
        var _S304 : u32 = bond_4.x;
        var q_lin_2 : vec3<f32>;
        var q_ang_2 : vec3<f32>;
        lp_force_0(b_26, _S304, bond_4.y, from_2, &(q_lin_2), &(q_ang_2));
        var fb_0 : vec3<f32> = b_26.t1_1 * vec3<f32>(q_lin_2.x) + b_26.t2_0 * vec3<f32>(q_lin_2.y) + b_26.n_3 * vec3<f32>(q_lin_2.z);
        var mb_0 : vec3<f32> = b_26.t1_1 * vec3<f32>(q_ang_2.x) + b_26.t2_0 * vec3<f32>(q_ang_2.y) + b_26.n_3 * vec3<f32>(q_ang_2.z);
        var arm_0 : vec3<f32> = b_26.centroid_0 - chunks_0[_S301].xyz;
        var _S305 : bool = k_21 == _S304;
        var lf_0 : vec3<f32>;
        if(_S305)
        {
            lf_0 = fb_0;
        }
        else
        {
            lf_0 = (vec3<f32>(0) - fb_0);
        }
        var _S306 : vec3<f32>;
        if(_S305)
        {
            _S306 = mb_0;
        }
        else
        {
            _S306 = (vec3<f32>(0) - mb_0);
        }
        var lm_0 : vec3<f32> = _S306 + cross(arm_0, lf_0);
        r_10[i32(0)] = r_10[i32(0)] + lf_0.x;
        r_10[i32(1)] = r_10[i32(1)] + lf_0.y;
        r_10[i32(2)] = r_10[i32(2)] + lf_0.z;
        r_10[i32(3)] = r_10[i32(3)] + lm_0.x;
        r_10[i32(4)] = r_10[i32(4)] + lm_0.y;
        r_10[i32(5)] = r_10[i32(5)] + lm_0.z;
        e_2 = e_2 + u32(1);
    }
    var l_1 : array<f32, i32(21)>;
    var z0_0 : u32 = u32(0);
    loop
    {
        if(z0_0 < u32(21))
        {
        }
        else
        {
            break;
        }
        l_1[z0_0] = 0.0f;
        z0_0 = z0_0 + u32(1);
    }
    var w_7 : u32 = u32(0);
    loop
    {
        if(w_7 < u32(6))
        {
        }
        else
        {
            break;
        }
        var v_9 : vec4<f32> = lp_0[lp_l_0(k_21) + w_7];
        var _S307 : u32 = u32(4) * w_7;
        l_1[_S307] = lp_0[lp_l_0(k_21) + w_7].x;
        var _S308 : u32 = _S307 + u32(1);
        if(_S308 < u32(21))
        {
            l_1[_S308] = v_9.y;
        }
        var _S309 : u32 = _S307 + u32(2);
        if(_S309 < u32(21))
        {
            l_1[_S309] = v_9.z;
        }
        var _S310 : u32 = _S307 + u32(3);
        if(_S310 < u32(21))
        {
            l_1[_S310] = v_9.w;
        }
        w_7 = w_7 + u32(1);
    }
    var y_3 : array<f32, i32(6)>;
    y_3[i32(0)] = 0.0f;
    y_3[i32(1)] = 0.0f;
    y_3[i32(2)] = 0.0f;
    y_3[i32(3)] = 0.0f;
    y_3[i32(4)] = 0.0f;
    y_3[i32(5)] = 0.0f;
    var i2_0 : u32 = u32(0);
    loop
    {
        if(i2_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        var p_7 : u32 = u32(0);
        s_9 = r_10[i2_0];
        loop
        {
            if(p_7 < i2_0)
            {
            }
            else
            {
                break;
            }
            var s_10 : f32 = s_9 - l_1[tri_0(i2_0, p_7)] * y_3[p_7];
            p_7 = p_7 + u32(1);
            s_9 = s_10;
        }
        y_3[i2_0] = s_9 / l_1[tri_0(i2_0, i2_0)];
        i2_0 = i2_0 + u32(1);
    }
    var z_4 : array<f32, i32(6)>;
    z_4[i32(0)] = 0.0f;
    z_4[i32(1)] = 0.0f;
    z_4[i32(2)] = 0.0f;
    z_4[i32(3)] = 0.0f;
    z_4[i32(4)] = 0.0f;
    z_4[i32(5)] = 0.0f;
    var i3_0 : i32 = i32(5);
    loop
    {
        if(i3_0 >= i32(0))
        {
        }
        else
        {
            break;
        }
        var p2_1 : i32 = i3_0 + i32(1);
        s_9 = y_3[i3_0];
        loop
        {
            if(p2_1 < i32(6))
            {
            }
            else
            {
                break;
            }
            var s_11 : f32 = s_9 - l_1[tri_0(u32(p2_1), u32(i3_0))] * z_4[p2_1];
            p2_1 = p2_1 + i32(1);
            s_9 = s_11;
        }
        var _S311 : u32 = u32(i3_0);
        z_4[i3_0] = s_9 / l_1[tri_0(_S311, _S311)];
        i3_0 = i3_0 - i32(1);
    }
    var _S312 : u32 = lp_d_0(k_21);
    var d0_0 : vec4<f32> = lp_0[_S312];
    var _S313 : u32 = _S312 + u32(1);
    var d1_0 : vec4<f32> = lp_0[_S313];
    var _S314 : vec3<f32> = vec3<f32>(params_0.lp_c1_0) * lp_0[_S312].xyz + vec3<f32>(params_0.lp_c2_0) * vec3<f32>(z_4[i32(0)], z_4[i32(1)], z_4[i32(2)]);
    d0_0.x = _S314.x;
    d0_0.y = _S314.y;
    d0_0.z = _S314.z;
    var _S315 : vec3<f32> = vec3<f32>(params_0.lp_c1_0) * lp_0[_S313].xyz + vec3<f32>(params_0.lp_c2_0) * vec3<f32>(z_4[i32(3)], z_4[i32(4)], z_4[i32(5)]);
    d1_0.x = _S315.x;
    d1_0.y = _S315.y;
    d1_0.z = _S315.z;
    lp_0[_S312] = d0_0;
    lp_0[_S313] = d1_0;
    var _S316 : u32 = lp_x_0(k_21, to_1);
    lp_0[_S316] = vec4<f32>(x0_0.xyz + d0_0.xyz, 0.0f);
    lp_0[_S316 + u32(1)] = vec4<f32>(x1_0.xyz + d1_0.xyz, 0.0f);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sweep_even(@builtin(global_invocation_id) id_31 : vec3<u32>)
{
    var _S317 : u32 = id_31.x;
    if(_S317 < (params_0.chunk_count_0))
    {
        lp_sweep_0(_S317, u32(0));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sweep_odd(@builtin(global_invocation_id) id_32 : vec3<u32>)
{
    var _S318 : u32 = id_32.x;
    if(_S318 < (params_0.chunk_count_0))
    {
        lp_sweep_0(_S318, u32(1));
    }
    return;
}

fn fdiv_0( a_7 : f32,  b_27 : f32) -> f32
{
    return a_7 / b_27;
}

fn fsqrt_0( a_8 : f32) -> f32
{
    return sqrt(a_8);
}

struct Measures_0
{
     tension_0 : f32,
     shear_0 : f32,
     normal_compression_0 : f32,
     compression_0 : f32,
     compressive_force_0 : f32,
};

fn stress_measures_0( b_28 : JointBond_0,  q_lin_3 : vec3<f32>,  q_ang_3 : vec3<f32>) -> Measures_0
{
    var area_0 : f32 = b_28.geom0_0.x;
    var _S319 : f32 = q_lin_3.z;
    var axial_0 : f32 = fdiv_0(_S319, area_0);
    var bending_0 : f32 = fdiv_0(abs(q_ang_3.x), b_28.geom1_0.x) + fdiv_0(abs(q_ang_3.y), b_28.geom1_0.y);
    var _S320 : f32 = q_lin_3.x;
    var _S321 : f32 = q_lin_3.y;
    var shear_1 : f32 = fdiv_0(fsqrt_0(_S320 * _S320 + _S321 * _S321), area_0) + fdiv_0(abs(q_ang_3.z), b_28.geom0_0.w);
    var m_3 : Measures_0;
    m_3.tension_0 = axial_0 + bending_0;
    m_3.shear_0 = shear_1;
    var _S322 : f32 = - axial_0;
    m_3.normal_compression_0 = max(_S322, 0.0f);
    m_3.compression_0 = _S322 + bending_0;
    m_3.compressive_force_0 = max(- _S319, 0.0f);
    return m_3;
}

struct JointMaterial_0
{
     strength_0 : vec4<f32>,
     energy_0 : vec4<f32>,
     dif_0 : vec4<f32>,
     misc_0 : vec4<f32>,
     kind_flags_0 : vec4<u32>,
};

fn lp_material_0( m_4 : u32) -> JointMaterial_0
{
    var o_6 : u32 = params_0.ext_mat_0 + u32(5) * m_4;
    var j_5 : JointMaterial_0;
    j_5.strength_0 = ext_0[o_6];
    j_5.energy_0 = ext_0[o_6 + u32(1)];
    j_5.dif_0 = ext_0[o_6 + u32(2)];
    j_5.misc_0 = ext_0[o_6 + u32(3)];
    j_5.kind_flags_0 = (bitcast<vec4<u32>>((ext_0[o_6 + u32(4)])));
    return j_5;
}

fn infinity_0() -> f32
{
    return float_bits_0(u32(2139095040));
}

fn failure_indices_0( mat_0 : JointMaterial_0,  b_29 : JointBond_0,  m_5 : Measures_0,  multiplier_0 : f32) -> vec4<f32>
{
    var fc_0 : f32 = mat_0.strength_0.y * multiplier_0;
    var _S323 : f32 = min(mat_0.strength_0.z * multiplier_0 + mat_0.strength_0.w * m_5.normal_compression_0, mat_0.energy_0.x * multiplier_0);
    var idx_0 : vec4<f32>;
    idx_0[i32(0)] = max(fdiv_0(m_5.tension_0, mat_0.strength_0.x * multiplier_0), 0.0f);
    var _S324 : f32;
    if(_S323 > 0.0f)
    {
        _S324 = fdiv_0(m_5.shear_0, _S323);
    }
    else
    {
        _S324 = infinity_0();
    }
    idx_0[i32(1)] = _S324;
    idx_0[i32(2)] = max(fdiv_0(m_5.compression_0, fc_0), 0.0f);
    var _S325 : f32 = b_29.stiff1_0.y;
    if(_S325 > 0.0f)
    {
        _S324 = fdiv_0(m_5.compressive_force_0, _S325);
    }
    else
    {
        _S324 = 0.0f;
    }
    idx_0[i32(3)] = _S324;
    return idx_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_breaks(@builtin(global_invocation_id) id_33 : vec3<u32>)
{
    var i_10 : u32 = id_33.x;
    var _S326 : bool;
    if(i_10 >= (params_0.bond_count_0))
    {
        _S326 = true;
    }
    else
    {
        _S326 = work_0[w_alive_0() + i_10] == u32(0);
    }
    if(_S326)
    {
        return;
    }
    var bond_5 : vec4<u32> = bonds_0[i_10];
    var b_30 : LpBond_0 = lp_bond_0(i_10);
    var _S327 : u32 = bond_5.x;
    var q_lin_4 : vec3<f32>;
    var q_ang_4 : vec3<f32>;
    lp_force_1(b_30, _S327, bond_5.y, u32(0), &(q_lin_4), &(q_ang_4));
    var idx_1 : vec4<f32> = failure_indices_0(lp_material_0(b_30.law_0.ids_1.x), b_30.law_0, stress_measures_0(b_30.law_0, q_lin_4, q_ang_4), b_30.law_0.geom1_0.w);
    var _S328 : f32 = max(max(idx_1.x, idx_1.y), max(idx_1.z, idx_1.w));
    var _S329 : u32 = a_counters_0();
    var _S330 : u32 = atomicMax(&(atom_0[_S329 + u32(11)]), (bitcast<u32>((max(_S328, 0.0f)))));
    if(_S328 >= 1.0f)
    {
        work_0[w_alive_0() + i_10] = u32(0);
        var _S331 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S327)]), u32(1));
        var _S332 : u32 = atomicAdd(&(atom_0[_S329]), u32(1));
        var _S333 : u32 = atomicAdd(&(atom_0[_S329 + u32(1)]), u32(1));
    }
    return;
}

fn a_first_0( shot_6 : u32) -> u32
{
    return a_strike_0() + u32(27) * params_0.body_count_0 + shot_6;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_first_reset(@builtin(global_invocation_id) id_34 : vec3<u32>)
{
    var _S334 : u32 = id_34.x;
    if(_S334 < (params_0.shot_count_0))
    {
        atomicStore(&(atom_0[a_first_0(_S334)]), u32(2139095040));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_first(@builtin(global_invocation_id) id_35 : vec3<u32>)
{
    var b_31 : u32 = id_35.x;
    var _S335 : bool;
    if(b_31 >= (params_0.body_count_0))
    {
        _S335 = true;
    }
    else
    {
        _S335 = (params_0.shot_count_0) == u32(0);
    }
    if(_S335)
    {
        return;
    }
    var shot_7 : u32;
    var _S336 : bool = struck_1(b_31, &(shot_7));
    if(!_S336)
    {
        return;
    }
    var _S337 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_31 + u32(6)]));
    var _S338 : u32 = atomicMin(&(atom_0[a_first_0(shot_7)]), _S337);
    return;
}

fn il_path_0( b_32 : u32,  contact_1 : ptr<function, vec3<f32>>,  d_5 : ptr<function, vec3<f32>>,  speed_1 : ptr<function, f32>)
{
    var st_4 : u32 = strike_at_0(b_32);
    var rel_3 : vec3<f32> = frames_0[st_4 + u32(1)].xyz;
    var _S339 : f32 = length(rel_3);
    (*speed_1) = _S339;
    (*d_5) = rel_3 / vec3<f32>(max(_S339, 9.999999960041972e-13f));
    (*contact_1) = frames_0[st_4 + u32(2)].xyz;
    return;
}

fn il_path_1( b_33 : u32,  contact_2 : ptr<function, vec3<f32>>,  d_6 : ptr<function, vec3<f32>>,  speed_2 : ptr<function, f32>)
{
    var st_5 : u32 = strike_at_0(b_33);
    var rel_4 : vec3<f32> = frames_0[st_5 + u32(1)].xyz;
    var _S340 : f32 = length(rel_4);
    (*speed_2) = _S340;
    (*d_6) = rel_4 / vec3<f32>(max(_S340, 9.999999960041972e-13f));
    (*contact_2) = frames_0[st_5 + u32(2)].xyz;
    return;
}

fn il_interval_0( k_22 : u32,  p_8 : vec3<f32>,  d_7 : vec3<f32>,  pad_0 : f32) -> vec2<f32>
{
    var q_28 : Quat_0;
    var _S341 : vec3<f32> = chunk_box_world_0(k_22, &(q_28));
    var _S342 : u32 = u32(3) * k_22;
    var _S343 : Quat_0 = normalized_0(quat_mul_0(q_28, quat_of_0(geometry_0[_S342 + u32(2)])));
    var _S344 : vec3<f32> = geometry_0[_S342 + u32(1)].xyz;
    var inf_0 : f32 = float_bits_0(u32(2139095040));
    var t0_2 : f32 = - inf_0;
    var t1_2 : f32 = inf_0;
    var ax_3 : u32 = u32(0);
    loop
    {
        if(ax_3 < u32(3))
        {
        }
        else
        {
            break;
        }
        var e_3 : vec3<f32> = vec3<f32>(0.0f);
        e_3[ax_3] = 1.0f;
        var axis_1 : vec3<f32> = rotate_0(_S343, e_3);
        var o_7 : f32 = dot(p_8 - _S341, axis_1);
        var dv_1 : f32 = dot(d_7, axis_1);
        var ext_2 : f32 = _S344[ax_3] + pad_0;
        if((abs(dv_1)) < 9.99999971718068537e-10f)
        {
            if((abs(o_7)) > ext_2)
            {
                return vec2<f32>(1.0f, 0.0f);
            }
            ax_3 = ax_3 + u32(1);
            continue;
        }
        var ta_2 : f32 = (- ext_2 - o_7) / dv_1;
        var tb_2 : f32 = (ext_2 - o_7) / dv_1;
        var _S345 : f32 = min(t1_2, max(ta_2, tb_2));
        t0_2 = max(t0_2, min(ta_2, tb_2));
        t1_2 = _S345;
        ax_3 = ax_3 + u32(1);
    }
    return vec2<f32>(t0_2, t1_2);
}

fn a_il_0( b_34 : u32) -> u32
{
    return a_strike_0() + u32(23) * params_0.body_count_0 + u32(4) * b_34;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_begin(@builtin(global_invocation_id) id_36 : vec3<u32>)
{
    var b_35 : u32 = id_36.x;
    var _S346 : bool;
    if(b_35 >= (params_0.body_count_0))
    {
        _S346 = true;
    }
    else
    {
        _S346 = (params_0.shot_count_0) == u32(0);
    }
    if(_S346)
    {
        return;
    }
    var shot_8 : u32;
    var _S347 : bool = struck_1(b_35, &(shot_8));
    if(!_S347)
    {
        return;
    }
    var _S348 : u32 = a_strike_0() + u32(9) * b_35;
    var _S349 : u32 = atomicLoad(&(atom_0[_S348 + u32(6)]));
    var _S350 : u32 = atomicLoad(&(atom_0[a_first_0(shot_8)]));
    if(_S349 != _S350)
    {
        frames_0[strike_at_0(b_35)][i32(3)] = 0.0f;
        return;
    }
    var contact_3 : vec3<f32>;
    var d_8 : vec3<f32>;
    var speed_3 : f32;
    il_path_1(b_35, &(contact_3), &(d_8), &(speed_3));
    var entry_1 : u32 = atomicLoad(&(atom_0[_S348 + u32(8)]));
    var end_0 : u32 = (bitcast<u32>((max(il_interval_0(entry_1, contact_3, d_8, shots_0[u32(2) * shot_8].z).y, 0.0f))));
    var _S351 : u32 = a_il_0(b_35);
    atomicStore(&(atom_0[_S351]), end_0);
    atomicStore(&(atom_0[_S351 + u32(1)]), end_0);
    atomicStore(&(atom_0[_S351 + u32(2)]), u32(0));
    atomicStore(&(atom_0[_S351 + u32(3)]), u32(0));
    return;
}

fn il_extend_0( k_23 : u32,  from_3 : u32)
{
    var _S352 : bool;
    if(k_23 >= (params_0.chunk_count_0))
    {
        _S352 = true;
    }
    else
    {
        _S352 = (params_0.shot_count_0) == u32(0);
    }
    if(_S352)
    {
        return;
    }
    var b_36 : u32 = body_of_0(k_23);
    var shot_9 : u32;
    var _S353 : bool = struck_0(b_36, &(shot_9));
    if(!_S353)
    {
        return;
    }
    var contact_4 : vec3<f32>;
    var d_9 : vec3<f32>;
    var speed_4 : f32;
    il_path_0(b_36, &(contact_4), &(d_9), &(speed_4));
    var pad_1 : f32 = shots_0[u32(2) * shot_9].z;
    var _S354 : u32 = a_il_0(b_36);
    var _S355 : u32 = atomicLoad(&(atom_0[_S354 + from_3]));
    var end_1 : f32 = (bitcast<f32>((_S355)));
    var to_2 : u32 = _S354 + u32(1) - from_3;
    var _S356 : u32 = atomicMax(&(atom_0[to_2]), (bitcast<u32>((end_1))));
    var t_8 : vec2<f32> = il_interval_0(k_23, contact_4, d_9, pad_1);
    var _S357 : f32 = t_8.x;
    var _S358 : f32 = t_8.y;
    if(_S357 <= _S358)
    {
        _S352 = _S357 <= (end_1 + 0.00100000004749745f + 0.01999999955296516f * pad_1);
    }
    else
    {
        _S352 = false;
    }
    if(_S352)
    {
        _S352 = _S358 > end_1;
    }
    else
    {
        _S352 = false;
    }
    if(_S352)
    {
        var _S359 : u32 = atomicMax(&(atom_0[to_2]), (bitcast<u32>((_S358))));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_extend_even(@builtin(global_invocation_id) id_37 : vec3<u32>)
{
    il_extend_0(id_37.x, u32(0));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_extend_odd(@builtin(global_invocation_id) id_38 : vec3<u32>)
{
    il_extend_0(id_38.x, u32(1));
    return;
}

fn w_marks_0() -> u32
{
    return w_pulses_0() + u32(4) * params_0.chunk_count_0;
}

fn lp_body_0( b_37 : u32) -> u32
{
    return u32(14) * params_0.chunk_count_0 + u32(2) * b_37;
}

fn fpow_0( a_9 : f32,  b_38 : f32) -> f32
{
    return pow(a_9, b_38);
}

fn dif_factor_0( mat_1 : JointMaterial_0,  strain_rate_0 : f32) -> f32
{
    var r_11 : f32 = abs(strain_rate_0);
    var ref_0 : f32 = mat_1.dif_0.x;
    if(r_11 <= ref_0)
    {
        return 1.0f;
    }
    var _S360 : f32 = mat_1.dif_0.z;
    var f_12 : f32;
    if(r_11 <= _S360)
    {
        f_12 = fpow_0(fdiv_0(r_11, ref_0), mat_1.dif_0.y);
    }
    else
    {
        f_12 = fpow_0(fdiv_0(_S360, ref_0), mat_1.dif_0.y) * fpow_0(fdiv_0(r_11, _S360), mat_1.dif_0.w);
    }
    return clamp(f_12, 1.0f, mat_1.misc_0.x);
}

fn il_impulse_0( b_39 : u32,  sb_1 : u32,  m_shot_0 : f32,  contact_5 : vec3<f32>,  d_10 : vec3<f32>,  j_6 : f32)
{
    var jr_0 : vec3<f32> = to_rubble_0(d_10) * vec3<f32>(j_6);
    var at_16 : u32 = u32(7) * b_39;
    var inv_mass_5 : f32 = bodies_0[at_16].w;
    var _S361 : vec3<f32> = jr_0 * vec3<f32>(inv_mass_5);
    var _S362 : vec3<f32> = vec3<f32>(0.0f);
    var dw_0 : vec3<f32>;
    if(inv_mass_5 > 0.0f)
    {
        var q_29 : Quat_0 = quat_of_0(bodies_0[at_16 + u32(1)]);
        dw_0 = rotate_0(q_29, bodies_0[at_16 + u32(2)].xyz * rotate_0(conj_0(q_29), cross(to_rubble_0(contact_5) - bodies_0[at_16].xyz, jr_0)));
    }
    else
    {
        dw_0 = _S362;
    }
    var region_1 : u32 = u32(0);
    loop
    {
        if(region_1 < u32(2))
        {
        }
        else
        {
            break;
        }
        var s_at_0 : u32 = u32(7) * (sb_1 + region_1 * params_0.max_bodies_0) + u32(3);
        var _S363 : vec3<f32> = bodies_0[s_at_0].xyz - jr_0 / vec3<f32>(m_shot_0);
        bodies_0[s_at_0].x = _S363.x;
        bodies_0[s_at_0].y = _S363.y;
        bodies_0[s_at_0].z = _S363.z;
        var w_at_0 : u32 = u32(7) * (b_39 + region_1 * params_0.max_bodies_0);
        var _S364 : u32 = w_at_0 + u32(3);
        var _S365 : vec3<f32> = bodies_0[_S364].xyz + _S361;
        bodies_0[_S364].x = _S365.x;
        bodies_0[_S364].y = _S365.y;
        bodies_0[_S364].z = _S365.z;
        var _S366 : u32 = w_at_0 + u32(4);
        var _S367 : vec3<f32> = bodies_0[_S366].xyz + dw_0;
        bodies_0[_S366].x = _S367.x;
        bodies_0[_S366].y = _S367.y;
        bodies_0[_S366].z = _S367.z;
        region_1 = region_1 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_strike(@builtin(global_invocation_id) id_39 : vec3<u32>)
{
    var b_40 : u32 = id_39.x;
    var damage_0 : bool;
    if(b_40 >= (params_0.body_count_0))
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = (params_0.shot_count_0) == u32(0);
    }
    if(damage_0)
    {
        return;
    }
    var shot_10 : u32;
    var _S368 : bool = struck_1(b_40, &(shot_10));
    if(!_S368)
    {
        return;
    }
    var at_17 : u32 = a_strike_0() + u32(9) * b_40;
    var _S369 : u32 = atomicLoad(&(atom_0[at_17 + u32(6)]));
    var t_in_3 : f32 = (bitcast<f32>((_S369)));
    var entry_2 : u32 = atomicLoad(&(atom_0[at_17 + u32(8)]));
    var contact_6 : vec3<f32>;
    var d_11 : vec3<f32>;
    var speed_5 : f32;
    il_path_1(b_40, &(contact_6), &(d_11), &(speed_5));
    var info_3 : vec4<f32> = shots_0[u32(2) * shot_10];
    var _S370 : f32 = max(info_3.z, 0.00100000004749745f);
    var m_imp_1 : f32 = info_3.w;
    var e_imp_1 : f32 = shots_0[u32(2) * shot_10 + u32(1)].x;
    var inv_mass_6 : f32 = bodies_0[u32(7) * b_40].w;
    var m_6 : f32;
    if(inv_mass_6 == 0.0f)
    {
        m_6 = m_imp_1;
    }
    else
    {
        m_6 = m_imp_1 / (1.0f + m_imp_1 * inv_mass_6);
    }
    var launch_0 : u32 = (bitcast<u32>((shots_0[u32(2) * shot_10 + u32(1)].y)));
    if(launch_0 != u32(0))
    {
        damage_0 = work_0[w_marks_0() + entry_2] == launch_0;
    }
    else
    {
        damage_0 = false;
    }
    if(damage_0)
    {
        var _S371 : u32 = lp_body_0(b_40);
        var _S372 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S371] = _S372;
        lp_0[_S371 + u32(1)] = _S372;
        return;
    }
    var _S373 : u32 = a_il_0(b_40);
    var _S374 : u32 = atomicLoad(&(atom_0[_S373]));
    var _S375 : u32 = atomicLoad(&(atom_0[_S373 + u32(1)]));
    if(_S374 != _S375)
    {
        var _S376 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(16));
    }
    var _S377 : f32 = max((bitcast<f32>((_S374))) - _S370, 9.99999997475242708e-07f);
    if(!(t_in_3 > 0.0f))
    {
        var _S378 : u32 = lp_body_0(b_40);
        lp_0[_S378] = vec4<f32>(_S377, 0.0f, 0.0f, 0.0f);
        lp_0[_S378 + u32(1)] = vec4<f32>(0.0f);
        return;
    }
    var o_8 : u32 = params_0.ext_chunks_0 + u32(4) * entry_2;
    var mat0_0 : vec4<f32> = ext_0[o_8 + u32(1)];
    var dm_0 : JointMaterial_0;
    var _S379 : vec4<f32> = vec4<f32>(0.0f);
    dm_0.strength_0 = _S379;
    dm_0.energy_0 = _S379;
    dm_0.dif_0 = ext_0[o_8 + u32(2)];
    var mat3_0 : vec4<f32> = ext_0[o_8 + u32(3)];
    var _S380 : f32 = mat3_0.x;
    dm_0.misc_0 = vec4<f32>(_S380, 0.0f, 0.0f, 0.0f);
    dm_0.kind_flags_0 = vec4<u32>(u32(0));
    var nu_0 : f32 = mat3_0.y;
    var e_target_0 : f32 = mat3_0.z / (1.0f - nu_0 * nu_0);
    var e_star_0 : f32 = 1.0f / (1.0f / e_imp_1 + 1.0f / e_target_0);
    var _S381 : f32 = max(speed_5, 0.00100000004749745f);
    var _S382 : f32 = sqrt(_S370);
    var h0_0 : f32 = pow(15.0f * m_6 * _S381 * _S381 / (16.0f * e_star_0 * _S382), 0.40000000596046448f);
    var t_c_0 : f32 = 2.94000005722045898f * h0_0 / _S381;
    var force_1 : f32 = 1.33333337306976318f * e_star_0 * _S382 * pow(h0_0, 1.5f);
    var a_10 : f32 = sqrt(_S370 * h0_0);
    var p0_0 : f32 = 3.0f * force_1 / (6.28318548202514648f * a_10 * a_10);
    var ring_0 : f32 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    var k_24 : f32;
    if(_S380 > 0.0f)
    {
        k_24 = dif_factor_0(dm_0, _S381 / _S377);
    }
    else
    {
        k_24 = 1.0f;
    }
    var ft_0 : f32 = k_24 * mat0_0.y;
    var fc_1 : f32 = k_24 * mat0_0.z;
    var tau_0 : f32 = k_24 * mat0_0.w;
    if(p0_0 > fc_1)
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ring_0 > ft_0;
    }
    var _S383 : f32 = fc_1 * (3.14159274101257324f * _S370 * _S370);
    var _S384 : f32 = min(_S383 * _S377, 1.57079637050628662f * (2.0f * _S370 + _S377) * _S377 * _S377 * tau_0);
    var _S385 : f32 = 2.0f * _S384;
    var v_bl_0 : f32 = sqrt(_S385 / m_6);
    var perforate_0 : bool;
    if(damage_0)
    {
        perforate_0 = _S381 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    var r2_1 : f32 = _S370 + _S377;
    var _S386 : f32 = mat0_0.x;
    var plug_0 : f32 = _S386 * 3.14159274101257324f * _S377 / 3.0f * (_S370 * _S370 + _S370 * r2_1 + r2_1 * r2_1);
    var v_res_0 : f32;
    if(perforate_0)
    {
        v_res_0 = m_imp_1 / (m_imp_1 + plug_0) * sqrt(max(_S381 * _S381 - v_bl_0 * v_bl_0, 0.0f));
    }
    else
    {
        v_res_0 = 0.0f;
    }
    var spall_0 : bool;
    if(damage_0)
    {
        spall_0 = !perforate_0;
    }
    else
    {
        spall_0 = false;
    }
    var depth_0 : f32;
    if(spall_0)
    {
        depth_0 = min(0.5f * m_6 * _S381 * _S381 / _S383, _S377);
    }
    else
    {
        depth_0 = 0.0f;
    }
    var crater_0 : bool = depth_0 > a_10;
    var c_19 : f32 = sqrt(e_target_0 / _S386);
    var z_t_0 : f32 = _S386 * c_19;
    var rho_p_0 : f32 = m_imp_1 / (4.18879032135009766f * _S370 * _S370 * _S370);
    var z_p_0 : f32 = rho_p_0 * sqrt(e_imp_1 / rho_p_0);
    var mean_pressure_0 : f32 = force_1 / (3.14159274101257324f * a_10 * a_10);
    var _S387 : f32 = min(min(z_t_0 * _S381 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_1);
    var source_radius_0 : f32;
    if(mean_pressure_0 > fc_1)
    {
        source_radius_0 = max(a_10, _S370);
    }
    else
    {
        source_radius_0 = a_10;
    }
    var pulse_0 : f32 = c_19 * t_c_0;
    var back_0 : f32 = _S387 * min(source_radius_0 / _S377, 1.0f) * min(2.0f * _S377 / pulse_0, 1.0f);
    if(damage_0)
    {
        spall_0 = !perforate_0;
    }
    else
    {
        spall_0 = false;
    }
    if(spall_0)
    {
        spall_0 = back_0 > ft_0;
    }
    else
    {
        spall_0 = false;
    }
    var scab_0 : f32;
    if(spall_0)
    {
        scab_0 = 0.5f * min(pulse_0, _S377);
    }
    else
    {
        scab_0 = 0.0f;
    }
    var region_2 : u32;
    if(damage_0)
    {
        region_2 = u32(1);
    }
    else
    {
        region_2 = u32(0);
    }
    var _S388 : u32;
    if(crater_0)
    {
        _S388 = u32(2);
    }
    else
    {
        _S388 = u32(0);
    }
    var _S389 : u32 = (region_2 | (_S388));
    if(spall_0)
    {
        region_2 = u32(4);
    }
    else
    {
        region_2 = u32(0);
    }
    var _S390 : u32 = (_S389 | (region_2));
    if(perforate_0)
    {
        region_2 = u32(8);
    }
    else
    {
        region_2 = u32(0);
    }
    var _S391 : u32 = lp_body_0(b_40);
    lp_0[_S391] = vec4<f32>(_S377, v_bl_0, v_res_0, (bitcast<f32>(((_S390 | (region_2))))));
    if(crater_0)
    {
        k_24 = depth_0;
    }
    else
    {
        k_24 = 0.0f;
    }
    lp_0[_S391 + u32(1)] = vec4<f32>(k_24, scab_0, a_10, _S384);
    if(perforate_0)
    {
        var _S392 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(10)]), u32(1));
    }
    else
    {
        if(spall_0)
        {
            var _S393 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(9)]), u32(1));
        }
        else
        {
            if(crater_0)
            {
                var _S394 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(8)]), u32(1));
            }
            else
            {
                var _S395 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(7)]), u32(1));
            }
        }
    }
    var sb_2 : u32 = (bitcast<u32>((info_3.x)));
    if(perforate_0)
    {
        if((params_0.fracture_work_0) != u32(0))
        {
            il_impulse_0(b_40, sb_2, m_imp_1, contact_6, d_11, m_6 * (_S381 - sqrt(max(_S381 * _S381 - _S385 / m_6, 0.0f))));
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_40, sb_2, m_imp_1, contact_6, d_11, m_6 * _S381);
            var _S396 : vec3<f32> = to_rubble_0(d_11) * vec3<f32>((_S381 * clamp(t_in_3, 0.0f, params_0.dt_0)));
            region_2 = u32(0);
            loop
            {
                if(region_2 < u32(2))
                {
                }
                else
                {
                    break;
                }
                var _S397 : u32 = u32(7) * (sb_2 + region_2 * params_0.max_bodies_0);
                var _S398 : vec3<f32> = bodies_0[_S397].xyz + _S396;
                bodies_0[_S397].x = _S398.x;
                bodies_0[_S397].y = _S398.y;
                bodies_0[_S397].z = _S398.z;
                region_2 = region_2 + u32(1);
            }
        }
    }
    return;
}

struct IlPlace_0
{
     along_2 : f32,
     radial_0 : f32,
     entry_3 : bool,
};

fn il_place_0( k_25 : u32,  entry_4 : u32,  contact_7 : vec3<f32>,  d_12 : vec3<f32>) -> IlPlace_0
{
    var slot_16 : u32 = body_of_0(k_25);
    var q_30 : Quat_0;
    var origin_11 : vec3<f32>;
    structure_pose_0(slot_16, u32(7) * slot_16, &(q_30), &(origin_11));
    var rel_5 : vec3<f32> = origin_11 + rotate_0(q_30, chunks_0[u32(4) * k_25].xyz) - contact_7;
    var p_9 : IlPlace_0;
    var _S399 : f32 = dot(rel_5, d_12);
    p_9.along_2 = _S399;
    p_9.radial_0 = length(rel_5 - d_12 * vec3<f32>(_S399));
    p_9.entry_3 = k_25 == entry_4;
    return p_9;
}

fn il_plug_0( outcome_0 : u32,  r_12 : f32,  h_2 : f32,  p_10 : IlPlace_0) -> bool
{
    var _S400 : bool;
    if(((outcome_0 & (u32(8)))) != u32(0))
    {
        if(p_10.entry_3)
        {
            _S400 = true;
        }
        else
        {
            if((p_10.along_2) >= 0.0f)
            {
                _S400 = (p_10.along_2) <= h_2;
            }
            else
            {
                _S400 = false;
            }
            if(_S400)
            {
                _S400 = (p_10.radial_0) <= (r_12 + p_10.along_2);
            }
            else
            {
                _S400 = false;
            }
        }
    }
    else
    {
        _S400 = false;
    }
    return _S400;
}

fn il_scab_0( outcome_1 : u32,  r_13 : f32,  h_3 : f32,  scab_1 : f32,  p_11 : IlPlace_0) -> bool
{
    var _S401 : bool;
    if(((outcome_1 & (u32(4)))) != u32(0))
    {
        _S401 = (p_11.along_2) >= (h_3 - scab_1);
    }
    else
    {
        _S401 = false;
    }
    if(_S401)
    {
        _S401 = (p_11.along_2) <= h_3;
    }
    else
    {
        _S401 = false;
    }
    if(_S401)
    {
        _S401 = (p_11.radial_0) <= (r_13 + h_3);
    }
    else
    {
        _S401 = false;
    }
    return _S401;
}

fn il_crater_0( outcome_2 : u32,  r_14 : f32,  depth_1 : f32,  a_11 : f32,  p_12 : IlPlace_0) -> bool
{
    var _S402 : bool;
    if(((outcome_2 & (u32(2)))) != u32(0))
    {
        _S402 = (p_12.along_2) >= 0.0f;
    }
    else
    {
        _S402 = false;
    }
    if(_S402)
    {
        _S402 = (p_12.along_2) <= depth_1;
    }
    else
    {
        _S402 = false;
    }
    if(_S402)
    {
        _S402 = (p_12.radial_0) <= (max(r_14, a_11));
    }
    else
    {
        _S402 = false;
    }
    return _S402;
}

fn w_carry_0() -> u32
{
    return w_marks_0() + params_0.chunk_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_mark(@builtin(global_invocation_id) id_40 : vec3<u32>)
{
    var k_26 : u32 = id_40.x;
    var path_0 : bool;
    if(k_26 >= (params_0.chunk_count_0))
    {
        path_0 = true;
    }
    else
    {
        path_0 = (params_0.shot_count_0) == u32(0);
    }
    if(path_0)
    {
        return;
    }
    var b_41 : u32 = body_of_0(k_26);
    var shot_11 : u32;
    var _S403 : bool = struck_1(b_41, &(shot_11));
    if(!_S403)
    {
        return;
    }
    var _S404 : u32 = lp_body_0(b_41);
    var s0_0 : vec4<f32> = lp_0[_S404];
    var s1_1 : vec4<f32> = lp_0[_S404 + u32(1)];
    var outcome_3 : u32 = (bitcast<u32>((lp_0[_S404].w)));
    if(((outcome_3 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_8 : vec3<f32>;
    var d_13 : vec3<f32>;
    var speed_6 : f32;
    il_path_1(b_41, &(contact_8), &(d_13), &(speed_6));
    var entry_5 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_41 + u32(8)]));
    var _S405 : f32 = max(shots_0[u32(2) * shot_11].z, 0.00100000004749745f);
    var p_13 : IlPlace_0 = il_place_0(k_26, entry_5, contact_8, d_13);
    var _S406 : f32 = s0_0.x;
    var plug_1 : bool = il_plug_0(outcome_3, _S405, _S406, p_13);
    var bound_0 : f32 = shots_0[u32(2) * shot_11].y;
    var t_9 : vec2<f32> = il_interval_0(k_26, contact_8 - d_13 * vec3<f32>(bound_0), d_13, bound_0);
    var _S407 : f32 = t_9.x;
    var _S408 : f32 = t_9.y;
    if(_S407 <= _S408)
    {
        path_0 = _S407 <= (_S406 + 2.0f * bound_0);
    }
    else
    {
        path_0 = false;
    }
    if(path_0)
    {
        path_0 = _S408 >= 0.0f;
    }
    else
    {
        path_0 = false;
    }
    if(plug_1)
    {
        path_0 = true;
    }
    if(path_0)
    {
        path_0 = true;
    }
    else
    {
        path_0 = il_scab_0(outcome_3, _S405, _S406, s1_1.y, p_13);
    }
    if(path_0)
    {
        path_0 = true;
    }
    else
    {
        path_0 = il_crater_0(outcome_3, _S405, s1_1.x, s1_1.z, p_13);
    }
    if(path_0)
    {
        work_0[w_marks_0() + k_26] = (bitcast<u32>((shots_0[u32(2) * shot_11 + u32(1)].y)));
    }
    if(plug_1)
    {
        work_0[w_carry_0() + k_26] = b_41 + u32(1);
        add64_0(a_il_0(b_41) + u32(2), chunks_0[u32(4) * k_26].w, params_0.scale_m_0);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_carry(@builtin(global_invocation_id) id_41 : vec3<u32>)
{
    var k_27 : u32 = id_41.x;
    var _S409 : bool;
    if(k_27 >= (params_0.chunk_count_0))
    {
        _S409 = true;
    }
    else
    {
        _S409 = (params_0.shot_count_0) == u32(0);
    }
    if(_S409)
    {
        return;
    }
    var _S410 : u32 = w_carry_0() + k_27;
    var c_20 : u32 = work_0[_S410];
    if(work_0[_S410] == u32(0))
    {
        return;
    }
    work_0[_S410] = u32(0);
    var b0_0 : u32 = c_20 - u32(1);
    var shot_12 : u32;
    var _S411 : bool = struck_1(b0_0, &(shot_12));
    if(!_S411)
    {
        return;
    }
    var contact_9 : vec3<f32>;
    var d_14 : vec3<f32>;
    var speed_7 : f32;
    il_path_1(b0_0, &(contact_9), &(d_14), &(speed_7));
    var dr_0 : vec3<f32> = to_rubble_0(d_14);
    var sb_3 : u32 = (bitcast<u32>((shots_0[u32(2) * shot_12].x)));
    var m_s_0 : f32 = shots_0[u32(2) * shot_12].w;
    var plug_mass_0 : f32 = read64_0(a_il_0(b0_0) + u32(2), params_0.scale_m_0);
    var v_wall_0 : vec3<f32> = bodies_0[u32(7) * b0_0 + u32(3)].xyz;
    var _S412 : u32 = u32(7) * sb_3;
    var v_shot_0 : vec3<f32> = bodies_0[_S412 + u32(3)].xyz;
    var closing_0 : f32 = dot(v_shot_0 - v_wall_0, dr_0);
    var v_r_0 : f32 = m_s_0 * closing_0 / (m_s_0 + plug_mass_0);
    var _S413 : vec3<f32> = v_shot_0 + dr_0 * vec3<f32>((v_r_0 - closing_0));
    var _S414 : vec3<f32> = v_wall_0 + dr_0 * vec3<f32>(v_r_0);
    var _S415 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b0_0 + u32(6)]));
    var _S416 : vec3<f32> = bodies_0[_S412].xyz + dr_0 * vec3<f32>(((closing_0 - v_r_0) * clamp((bitcast<f32>((_S415))), 0.0f, params_0.dt_0)));
    var _S417 : u32 = body_of_0(k_27);
    var region_3 : u32 = u32(0);
    loop
    {
        if(region_3 < u32(2))
        {
        }
        else
        {
            break;
        }
        var s_at_1 : u32 = u32(7) * (sb_3 + region_3 * params_0.max_bodies_0);
        bodies_0[s_at_1].x = _S416.x;
        bodies_0[s_at_1].y = _S416.y;
        bodies_0[s_at_1].z = _S416.z;
        bodies_0[s_at_1 + u32(3)].x = _S413.x;
        bodies_0[s_at_1 + u32(3)].y = _S413.y;
        bodies_0[s_at_1 + u32(3)].z = _S413.z;
        bodies_0[u32(7) * (_S417 + region_3 * params_0.max_bodies_0) + u32(3)].x = _S414.x;
        bodies_0[u32(7) * (_S417 + region_3 * params_0.max_bodies_0) + u32(3)].y = _S414.y;
        bodies_0[u32(7) * (_S417 + region_3 * params_0.max_bodies_0) + u32(3)].z = _S414.z;
        region_3 = region_3 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_breaks(@builtin(global_invocation_id) id_42 : vec3<u32>)
{
    var i_11 : u32 = id_42.x;
    var cut_0 : bool;
    if(i_11 >= (params_0.bond_count_0))
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = (params_0.shot_count_0) == u32(0);
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = work_0[w_alive_0() + i_11] == u32(0);
    }
    if(cut_0)
    {
        return;
    }
    var bond_6 : vec4<u32> = bonds_0[i_11];
    var _S418 : u32 = bond_6.x;
    var b_42 : u32 = body_of_0(_S418);
    var shot_13 : u32;
    var _S419 : bool = struck_1(b_42, &(shot_13));
    if(!_S419)
    {
        return;
    }
    var _S420 : u32 = lp_body_0(b_42);
    var s0_1 : vec4<f32> = lp_0[_S420];
    var s1_2 : vec4<f32> = lp_0[_S420 + u32(1)];
    var outcome_4 : u32 = (bitcast<u32>((lp_0[_S420].w)));
    if(((outcome_4 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_10 : vec3<f32>;
    var d_15 : vec3<f32>;
    var speed_8 : f32;
    il_path_1(b_42, &(contact_10), &(d_15), &(speed_8));
    var entry_6 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_42 + u32(8)]));
    var _S421 : f32 = max(shots_0[u32(2) * shot_13].z, 0.00100000004749745f);
    var h_4 : f32 = s0_1.x;
    var pa_1 : IlPlace_0 = il_place_0(_S418, entry_6, contact_10, d_15);
    var pb_2 : IlPlace_0 = il_place_0(bond_6.y, entry_6, contact_10, d_15);
    if((il_plug_0(outcome_4, _S421, h_4, pa_1)) != (il_plug_0(outcome_4, _S421, h_4, pb_2)))
    {
        cut_0 = true;
    }
    else
    {
        var _S422 : f32 = s1_2.y;
        cut_0 = (il_scab_0(outcome_4, _S421, h_4, _S422, pa_1)) != (il_scab_0(outcome_4, _S421, h_4, _S422, pb_2));
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = il_crater_0(outcome_4, _S421, s1_2.x, s1_2.z, pa_1);
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = il_crater_0(outcome_4, _S421, s1_2.x, s1_2.z, pb_2);
    }
    if(cut_0)
    {
        work_0[w_alive_0() + i_11] = u32(0);
        var _S423 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_42]), u32(1));
        var _S424 : u32 = a_counters_0();
        var _S425 : u32 = atomicAdd(&(atom_0[_S424]), u32(1));
        var _S426 : u32 = atomicAdd(&(atom_0[_S424 + u32(1)]), u32(1));
    }
    return;
}

