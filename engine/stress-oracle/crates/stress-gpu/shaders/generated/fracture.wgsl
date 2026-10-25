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

struct CrushParams_std140_0
{
    @align(16) enabled_0 : u32,
    @align(4) mode_1 : u32,
    @align(8) fragments_0 : u32,
    @align(4) pool_size_0 : u32,
    @align(16) max_events_0 : u32,
    @align(4) t_bonds_0 : u32,
    @align(8) t_adj_0 : u32,
    @align(4) t_list_0 : u32,
    @align(16) t_pool_0 : u32,
    @align(4) r_bill_0 : u32,
    @align(8) r_shot_0 : u32,
    @align(4) r_events_0 : u32,
    @align(16) r_dist_0 : u32,
    @align(4) park_y_0 : f32,
    @align(8) park_step_0 : f32,
    @align(4) pad0_1 : f32,
};

@binding(25) @group(0) var<uniform> crush_0 : CrushParams_std140_0;
@binding(23) @group(0) var<storage, read> crush_tab_0 : array<vec4<f32>>;

@binding(24) @group(0) var<storage, read_write> crush_rw_0 : array<vec4<f32>>;

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
    var _S12 : f32 = start_0 + plateau_time_0;
    var sum_0 : f32 = plateau_0 * max(min(_S12, dt_1) - max(start_0, 0.0f), 0.0f);
    var _S13 : f32 = max(0.0f, _S12);
    var _S14 : f32 = min(dt_1, _S12 + duration_0);
    var _S15 : bool;
    if(_S14 > _S13)
    {
        _S15 = duration_0 > 0.0f;
    }
    else
    {
        _S15 = false;
    }
    var sum_1 : f32;
    if(_S15)
    {
        var w_1 : f32 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_1 * (cos(w_1 * (_S13 - _S12)) - cos(w_1 * (_S14 - _S12)));
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

@compute
@workgroup_size(64, 1, 1)
fn fr_loads(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var f_total_0 : vec3<f32>;
    var m_1 : f32;
    var p_0 : u32;
    var kept_0 : u32;
    var q3_0 : u32;
    var overflow_0 : bool;
    var overflow_1 : bool;
    var f_total_1 : vec3<f32>;
    var t_total_0 : vec3<f32>;
    var k_0 : u32 = id_0.x;
    if(k_0 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_2 : u32 = body_of_0(k_0);
    var b_2 : u32 = u32(7) * slot_2;
    var q_7 : Quat_0;
    var origin_2 : vec3<f32>;
    structure_pose_1(slot_2, b_2, &(q_7), &(origin_2));
    var _S16 : u32 = u32(4) * k_0;
    var c0_0 : vec4<f32> = chunks_0[_S16];
    var centre_0 : vec3<f32> = origin_2 + rotate_0(q_7, c0_0.xyz);
    var inv_mass_0 : f32 = bodies_0[b_2].w;
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
    var _S17 : u32 = w_pulses_0() + _S16;
    var _S18 : u32 = _S17 + u32(2) * params_0.parity_0;
    var off_0 : u32 = work_0[_S18];
    var carried_0 : u32 = work_0[_S18 + u32(1)];
    var room_0 : u32 = work_0[_S18 + u32(1)] + sort_0[params_0.chunk_count_0 + u32(1) + k_0];
    var base_0 : u32;
    var room_1 : u32;
    if(room_0 != u32(0))
    {
        var _S19 : u32 = a_counters_0();
        var base_1 : u32 = atomicAdd(&(atom_0[_S19 + u32(5) + to_0]), room_0);
        if((base_1 + room_0) > (params_0.pool_cap_0))
        {
            var _S20 : u32 = atomicOr(&(atom_0[_S19 + u32(3)]), u32(8));
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
        var _S21 : u32 = base_0;
        base_0 = base_1;
        room_1 = _S21;
    }
    else
    {
        base_0 = u32(0);
        room_1 = room_0;
    }
    var _S22 : u32 = params_0.chunk_count_0;
    var _S23 : f32 = c0_0.w;
    var _S24 : vec4<f32> = chunks_0[_S16 + u32(3)];
    var _S25 : f32 = params_0.band_0;
    var _S26 : f32 = params_0.dt_0;
    var _S27 : bool = (params_0.impact_peak_0) != u32(0);
    var _S28 : u32 = l_pool_0(from_0) + u32(3) * off_0;
    var _S29 : u32 = l_pool_0(to_0) + u32(3) * base_0;
    loop
    {
        var _S30 : vec3<f32> = vec3<f32>(0.0f);
        var _S31 : u32 = sort_0[k_0];
        var _S32 : u32 = sort_0[_S22 + u32(1) + k_0];
        var first_0 : bool = true;
        var last_0 : u32 = u32(0);
        var step_1 : u32 = u32(0);
        f_total_0 = _S30;
        var t_total_1 : vec3<f32> = _S30;
        loop
        {
            if(step_1 < _S32)
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
                if(q3_0 < _S32)
                {
                }
                else
                {
                    break;
                }
                var key_1 : u32 = manifolds_0[_S31 + q3_0].ids_0.y + u32(1);
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
            var r_4 : Record_std430_0 = manifolds_0[_S31 + p_0];
            var _S33 : vec3<f32> = r_4.impulse_0.xyz;
            var j_0 : f32 = length(_S33);
            if(j_0 == 0.0f)
            {
                var _S34 : u32 = step_1 + u32(1);
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S34;
                continue;
            }
            var m_other_0 : f32 = r_4.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_0))
                {
                    m_1 = _S23;
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
            var _S35 : f32 = min(j_0, 2.0f * m_1 * max(r_4.impulse_0.w - _S25, 0.0f));
            if(_S35 < j_0)
            {
                var f_0 : vec3<f32> = _S33 / vec3<f32>(j_0) * vec3<f32>(((j_0 - _S35) / _S26));
                var _S36 : vec3<f32> = t_total_1 + cross(r_4.point_0.xyz - centre_0, f_0);
                f_total_1 = f_total_0 + f_0;
                t_total_0 = _S36;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            var _S34 : u32 = step_1 + u32(1);
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S34;
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
            var _S37 : u32 = _S28 + u32(3) * p_0;
            var a_1 : vec4<f32> = loads_0[_S37];
            var b_3 : vec4<f32> = loads_0[_S37 + u32(1)];
            var c_2 : vec4<f32> = loads_0[_S37 + u32(2)];
            var _S38 : f32 = loads_0[_S37 + u32(1)].w;
            var mean_0 : f32 = pulse_mean_0(loads_0[_S37].x, loads_0[_S37].y, loads_0[_S37].z, loads_0[_S37].w, _S38, _S26);
            if(mean_0 != 0.0f)
            {
                var f_1 : vec3<f32> = c_2.xyz * vec3<f32>(mean_0);
                var _S39 : vec3<f32> = t_total_1 + cross(b_3.xyz - centre_0, f_1);
                f_total_0 = f_total_0 + f_1;
                t_total_1 = _S39;
            }
            if((a_1.x + a_1.z + _S38) > _S26)
            {
                a_1[i32(0)] = a_1[i32(0)] - _S26;
                var _S40 : u32 = _S29 + u32(3) * kept_0;
                loads_0[_S40] = a_1;
                loads_0[_S40 + u32(1)] = b_3;
                loads_0[_S40 + u32(2)] = c_2;
                kept_0 = kept_0 + u32(1);
            }
            p_0 = p_0 + u32(1);
        }
        q3_0 = u32(0);
        overflow_0 = false;
        loop
        {
            if(q3_0 < _S32)
            {
            }
            else
            {
                break;
            }
            var r_5 : Record_std430_0 = manifolds_0[_S31 + q3_0];
            var _S41 : vec3<f32> = r_5.impulse_0.xyz;
            var j_1 : f32 = length(_S41);
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
                    m_1 = _S23;
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
            var _S42 : f32 = r_5.impulse_0.w;
            var _S43 : f32 = min(j_1, 2.0f * m_1 * max(_S42 - _S25, 0.0f));
            if(_S43 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var _S44 : f32 = _S24.y;
            var duration_1 : f32 = hertz_duration_0(m_1, min(r_5.other_0.z, _S24.z), 1.0f / ((1.0f - _S44 * _S44) / _S24.x + 1.0f / r_5.other_0.y), _S42);
            var peak_1 : f32 = 3.14159274101257324f * _S43 / (2.0f * duration_1);
            var dir_0 : vec3<f32> = _S41 / vec3<f32>(j_1);
            var mean_1 : f32;
            if(_S27)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S26);
            }
            if(mean_1 != 0.0f)
            {
                var f_2 : vec3<f32> = dir_0 * vec3<f32>(mean_1);
                var _S45 : vec3<f32> = t_total_1 + cross(r_5.point_0.xyz - centre_0, f_2);
                f_total_1 = f_total_0 + f_2;
                t_total_0 = _S45;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            var kept_1 : u32;
            if(duration_1 > _S26)
            {
                if(kept_0 < room_1)
                {
                    var _S46 : u32 = _S29 + u32(3) * kept_0;
                    loads_0[_S46] = vec4<f32>(- _S26, 0.0f, 0.0f, peak_1);
                    loads_0[_S46 + u32(1)] = vec4<f32>(r_5.point_0.xyz, duration_1);
                    loads_0[_S46 + u32(2)] = vec4<f32>(dir_0, 0.0f);
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
            t_total_1 = t_total_0;
            kept_0 = kept_1;
            q3_0 = q3_0 + u32(1);
            overflow_0 = overflow_1;
        }
        break;
    }
    if(overflow_0)
    {
        var _S47 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(8));
    }
    var _S48 : u32 = _S17 + u32(2) * to_0;
    work_0[_S48] = base_0;
    work_0[_S48 + u32(1)] = kept_0;
    loads_0[k_0] = vec4<f32>(length(f_total_0), 0.0f, 0.0f, 0.0f);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_breaks(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var i_0 : u32 = id_1.x;
    var _S49 : bool;
    if(i_0 >= (params_0.bond_count_0))
    {
        _S49 = true;
    }
    else
    {
        _S49 = work_0[w_alive_0() + i_0] == u32(0);
    }
    if(_S49)
    {
        return;
    }
    var bond_0 : vec4<u32> = bonds_0[i_0];
    var _S50 : u32 = bond_0.x;
    if((max(loads_0[_S50].x, loads_0[bond_0.y].x)) > (bitcast<f32>((bond_0.z))))
    {
        work_0[w_alive_0() + i_0] = u32(0);
        var _S51 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S50)]), u32(1));
        var _S52 : u32 = a_counters_0();
        var _S53 : u32 = atomicAdd(&(atom_0[_S52]), u32(1));
        var _S54 : u32 = atomicAdd(&(atom_0[_S52 + u32(1)]), u32(1));
    }
    return;
}

fn groups_of_0( n_2 : u32,  size_0 : u32) -> u32
{
    var _S55 : u32 = (n_2 + size_0 - u32(1)) / size_0;
    return _S55;
}

fn set_args_0( at_0 : u32,  x_2 : u32)
{
    var _S56 : u32 = u32(3) * at_0;
    args_0[_S56] = x_2;
    args_0[_S56 + u32(1)] = u32(1);
    args_0[_S56 + u32(2)] = u32(1);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_gate(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var _S57 : u32 = a_counters_0();
    var _S58 : u32 = atomicLoad(&(atom_0[_S57]));
    var run_0 : bool = _S58 != u32(0);
    atomicStore(&(atom_0[_S57]), u32(0));
    atomicStore(&(atom_0[_S57 + u32(5) + params_0.parity_0]), u32(0));
    if(run_0)
    {
        var _S59 : u32 = atomicAdd(&(atom_0[_S57 + u32(4)]), u32(1));
    }
    var _S60 : u32;
    if(run_0)
    {
        var _S61 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S60 = _S61;
    }
    else
    {
        _S60 = u32(0);
    }
    set_args_0(u32(0), _S60);
    if(run_0)
    {
        var _S62 : u32 = groups_of_0(params_0.bond_count_0, u32(64));
        _S60 = _S62;
    }
    else
    {
        _S60 = u32(0);
    }
    set_args_0(u32(1), _S60);
    if(run_0)
    {
        var _S63 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S60 = _S63;
    }
    else
    {
        _S60 = u32(0);
    }
    set_args_0(u32(2), _S60);
    if(run_0)
    {
        var _S64 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S60 = _S64;
    }
    else
    {
        _S60 = u32(0);
    }
    set_args_0(u32(3), _S60);
    if(run_0)
    {
        var _S65 : u32 = groups_of_0(params_0.body_count_0, u32(1024));
        _S60 = _S65;
    }
    else
    {
        _S60 = u32(0);
    }
    set_args_0(u32(4), _S60);
    var _S66 : i32;
    if(run_0)
    {
        _S66 = i32(1);
    }
    else
    {
        _S66 = i32(0);
    }
    set_args_0(u32(5), u32(_S66));
    return;
}

fn dirty_0( body_0 : u32) -> bool
{
    var _S67 : u32 = atomicLoad(&(atom_0[a_dirty_0() + body_0]));
    return _S67 != u32(0);
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
    var _S68 : bool;
    if(c_3 >= (params_0.chunk_count_0))
    {
        _S68 = true;
    }
    else
    {
        var _S69 : bool = dirty_0(body_of_0(c_3));
        _S68 = !_S69;
    }
    if(_S68)
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
    var _S70 : bool;
    if(s_0 >= (params_0.body_count_0))
    {
        _S70 = true;
    }
    else
    {
        var _S71 : bool = dirty_0(s_0);
        _S70 = !_S71;
    }
    if(_S70)
    {
        return;
    }
    atomicStore(&(atom_0[a_best_count_0() + s_0]), u32(0));
    atomicStore(&(atom_0[a_best_root_0() + s_0]), u32(4294967295));
    return;
}

fn find_0( x_3 : u32) -> u32
{
    var _S72 : u32 = a_parent_0();
    var _S73 : u32 = atomicLoad(&(atom_0[_S72 + x_3]));
    var p_1 : u32 = _S73;
    var _S74 : u32 = x_3;
    var guard_0 : u32 = u32(0);
    loop
    {
        var _S75 : bool;
        if(guard_0 < u32(1048576))
        {
            _S75 = p_1 != _S74;
        }
        else
        {
            _S75 = false;
        }
        if(_S75)
        {
        }
        else
        {
            break;
        }
        var gp_0 : u32 = atomicLoad(&(atom_0[_S72 + p_1]));
        if(gp_0 != p_1)
        {
            var _S76 : u32 = atomicCompareExchangeWeak(&(atom_0[_S72 + _S74]), p_1, gp_0).old_value;
        }
        var _S77 : u32 = guard_0 + u32(1);
        _S74 = p_1;
        p_1 = gp_0;
        guard_0 = _S77;
    }
    return _S74;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_union(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var i_1 : u32 = id_5.x;
    var _S78 : bool;
    if(i_1 >= (params_0.bond_count_0))
    {
        _S78 = true;
    }
    else
    {
        _S78 = work_0[w_alive_0() + i_1] == u32(0);
    }
    if(_S78)
    {
        return;
    }
    var bond_1 : vec4<u32> = bonds_0[i_1];
    var _S79 : u32 = bond_1.x;
    var _S80 : bool = dirty_0(body_of_0(_S79));
    if(!_S80)
    {
        return;
    }
    var _S81 : u32 = bond_1.y;
    var a_2 : u32 = _S79;
    var b_4 : u32 = _S81;
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
        var rb_0 : u32 = find_0(b_4);
        if(ra_0 == rb_0)
        {
            return;
        }
        var _S82 : u32 = max(ra_0, rb_0);
        var _S83 : u32 = min(ra_0, rb_0);
        var was_0 : u32 = atomicCompareExchangeWeak(&(atom_0[a_parent_0() + _S82]), _S82, _S83).old_value;
        if(was_0 == _S82)
        {
            return;
        }
        var _S84 : u32 = guard_1 + u32(1);
        a_2 = _S82;
        b_4 = _S83;
        guard_1 = _S84;
    }
    var _S85 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(4));
    return;
}

fn add64_0( at_1 : u32,  value_0 : f32,  scale_0 : f32)
{
    var x_4 : f32 = round(value_0 * scale_0);
    var x_5 : f32;
    if(!((abs(x_4)) < 1.073741824e+09f))
    {
        var _S86 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
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
    var _S87 : u32;
    if(v_3 < i32(0))
    {
        _S87 = u32(4294967295);
    }
    else
    {
        _S87 = u32(0);
    }
    var hi_0 : u32 = _S87 + carry_0;
    if(hi_0 != u32(0))
    {
        var _S88 : u32 = atomicAdd(&(atom_0[at_1 + u32(1)]), hi_0);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_flatten(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var c_4 : u32 = id_6.x;
    var _S89 : bool;
    if(c_4 >= (params_0.chunk_count_0))
    {
        _S89 = true;
    }
    else
    {
        var _S90 : bool = dirty_0(body_of_0(c_4));
        _S89 = !_S90;
    }
    if(_S89)
    {
        return;
    }
    var r_6 : u32 = find_0(c_4);
    work_0[w_label_0() + c_4] = r_6;
    var _S91 : u32 = atomicAdd(&(atom_0[a_count_0() + r_6]), u32(1));
    if(((info_0[c_4] & (u32(1)))) != u32(0))
    {
        var _S92 : u32 = atomicOr(&(atom_0[a_anchor_0() + r_6]), u32(1));
    }
    var _S93 : u32 = u32(4) * c_4;
    var c0_1 : vec4<f32> = chunks_0[_S93];
    var c1_0 : vec4<f32> = chunks_0[_S93 + u32(1)];
    var c2_0 : vec4<f32> = chunks_0[_S93 + u32(2)];
    var m_2 : f32 = c0_1.w;
    var d_0 : vec3<f32> = c0_1.xyz - chunks_0[u32(4) * r_6].xyz;
    var dd_0 : f32 = dot(d_0, d_0);
    var at_2 : u32 = a_sums_0() + u32(20) * r_6;
    add64_0(at_2, m_2, params_0.scale_m_0);
    var _S94 : f32 = d_0.x;
    var _S95 : f32 = m_2 * _S94;
    add64_0(at_2 + u32(2), _S95, params_0.scale_s_0);
    var _S96 : f32 = d_0.y;
    var _S97 : f32 = m_2 * _S96;
    add64_0(at_2 + u32(4), _S97, params_0.scale_s_0);
    var _S98 : f32 = d_0.z;
    add64_0(at_2 + u32(6), m_2 * _S98, params_0.scale_s_0);
    add64_0(at_2 + u32(8), c1_0.x + m_2 * (dd_0 - _S94 * _S94), params_0.scale_j_0);
    add64_0(at_2 + u32(10), c1_0.y + m_2 * (dd_0 - _S96 * _S96), params_0.scale_j_0);
    add64_0(at_2 + u32(12), c1_0.z + m_2 * (dd_0 - _S98 * _S98), params_0.scale_j_0);
    add64_0(at_2 + u32(14), c1_0.w - _S95 * _S96, params_0.scale_j_0);
    add64_0(at_2 + u32(16), c2_0.x - _S95 * _S98, params_0.scale_j_0);
    add64_0(at_2 + u32(18), c2_0.y - _S97 * _S98, params_0.scale_j_0);
    return;
}

fn is_root_0( c_5 : u32) -> bool
{
    var _S99 : bool;
    if(c_5 < (params_0.chunk_count_0))
    {
        var _S100 : bool = dirty_0(body_of_0(c_5));
        _S99 = _S100;
    }
    else
    {
        _S99 = false;
    }
    if(_S99)
    {
        _S99 = work_0[w_label_0() + c_5] == c_5;
    }
    else
    {
        _S99 = false;
    }
    return _S99;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_count(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var c_6 : u32 = id_7.x;
    var _S101 : bool = is_root_0(c_6);
    if(!_S101)
    {
        return;
    }
    var _S102 : u32 = atomicLoad(&(atom_0[a_count_0() + c_6]));
    var _S103 : u32 = atomicMax(&(atom_0[a_best_count_0() + body_of_0(c_6)]), _S102);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_root(@builtin(global_invocation_id) id_8 : vec3<u32>)
{
    var c_7 : u32 = id_8.x;
    var _S104 : bool = is_root_0(c_7);
    if(!_S104)
    {
        return;
    }
    var s_1 : u32 = body_of_0(c_7);
    var _S105 : u32 = atomicLoad(&(atom_0[a_count_0() + c_7]));
    var _S106 : u32 = atomicLoad(&(atom_0[a_best_count_0() + s_1]));
    if(_S105 == _S106)
    {
        var _S107 : u32 = atomicMin(&(atom_0[a_best_root_0() + s_1]), c_7);
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
    var _S108 : bool = is_root_0(c_8);
    var _S109 : bool;
    if(_S108)
    {
        var _S110 : u32 = atomicLoad(&(atom_0[a_best_root_0() + body_of_0(c_8)]));
        _S109 = _S110 != c_8;
    }
    else
    {
        _S109 = false;
    }
    var _S111 : u32;
    if(_S109)
    {
        _S111 = u32(1);
    }
    else
    {
        _S111 = u32(0);
    }
    work_0[w_new_0() + c_8] = _S111;
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
    var _S112 : u32 = group_0.x;
    var _S113 : u32 = u32(1024) * _S112 + u32(4) * t_1;
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
        if((_S113 + j_2) < (scan_0.n_0))
        {
            step_2 = work_0[scan_0.in_base_0 + _S113 + j_2];
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
    var _S114 : u32 = g_scan_0[t_1] - sum_2;
    j_2 = u32(0);
    add_0 = _S114;
    loop
    {
        if(j_2 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S113 + j_2) < (scan_0.n_0))
        {
            work_0[scan_0.out_base_0 + _S113 + j_2] = add_0;
        }
        var run_1 : u32 = add_0 + v_4[j_2];
        j_2 = j_2 + u32(1);
        add_0 = run_1;
    }
    if(t_1 == u32(255))
    {
        work_0[scan_0.block_base_0 + _S112] = g_scan_0[i32(255)];
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn fr_scan_top(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var t_2 : u32 = thread_1.x;
    var blocks_0 : u32 = (scan_0.n_0 + u32(1023)) / u32(1024);
    var _S115 : bool = t_2 < blocks_0;
    var v_5 : u32;
    if(_S115)
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
    if(_S115)
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
    var _S116 : bool;
    if(s_2 >= (params_0.body_count_0))
    {
        _S116 = true;
    }
    else
    {
        _S116 = work_0[w_free_0() + s_2] == u32(0);
    }
    if(_S116)
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
    var _S117 : bool = is_root_0(c_9);
    if(!_S117)
    {
        return;
    }
    var s_3 : u32 = body_of_0(c_9);
    var _S118 : u32 = atomicLoad(&(atom_0[a_best_root_0() + s_3]));
    var slot_3 : u32;
    if(_S118 != c_9)
    {
        var k_1 : u32 = work_0[w_rank_0() + c_9];
        if(work_0[w_rank_0() + c_9] < (free_total_0()))
        {
            var slot_4 : u32 = work_0[w_free_list_0() + k_1];
            work_0[w_free_0() + work_0[w_free_list_0() + k_1]] = u32(0);
            var _S119 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(2)]), u32(1));
            slot_3 = slot_4;
        }
        else
        {
            var _S120 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(1));
            slot_3 = s_3;
        }
    }
    else
    {
        slot_3 = s_3;
    }
    work_0[w_slot_0() + c_9] = slot_3;
    return;
}

fn read64_0( at_3 : u32,  scale_1 : f32) -> f32
{
    var _S121 : u32 = atomicLoad(&(atom_0[at_3 + u32(1)]));
    var hi_1 : i32 = (bitcast<i32>((_S121)));
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
    var q_8 : Quat_0;
    if(m22_0 <= 0.0f)
    {
        var dif10_0 : f32 = m11_0 - m00_0;
        var omm22_0 : f32 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            var f_3 : f32 = omm22_0 - dif10_0;
            var inv_0 : f32 = 0.5f / sqrt(f_3);
            q_8.x_1 = f_3 * inv_0;
            q_8.y_0 = (m01_0 + m10_0) * inv_0;
            q_8.z_0 = (m02_0 + m20_0) * inv_0;
            q_8.w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            var f_4 : f32 = omm22_0 + dif10_0;
            var inv_1 : f32 = 0.5f / sqrt(f_4);
            q_8.x_1 = (m01_0 + m10_0) * inv_1;
            q_8.y_0 = f_4 * inv_1;
            q_8.z_0 = (m12_0 + m21_0) * inv_1;
            q_8.w_0 = (m20_0 - m02_0) * inv_1;
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
            q_8.x_1 = (m02_0 + m20_0) * inv_2;
            q_8.y_0 = (m12_0 + m21_0) * inv_2;
            q_8.z_0 = f_5 * inv_2;
            q_8.w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            var f_6 : f32 = opm22_0 + sum10_0;
            var inv_3 : f32 = 0.5f / sqrt(f_6);
            q_8.x_1 = (m12_0 - m21_0) * inv_3;
            q_8.y_0 = (m20_0 - m02_0) * inv_3;
            q_8.z_0 = (m01_0 - m10_0) * inv_3;
            q_8.w_0 = f_6 * inv_3;
        }
    }
    return q_8;
}

fn principal_axes_0( r0_0 : vec3<f32>,  r1_0 : vec3<f32>,  r2_0 : vec3<f32>,  rot_0 : ptr<function, Quat_0>,  moments_0 : ptr<function, vec3<f32>>)
{
    var _S122 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S123 : array<f32, i32(3)> = array<f32, i32(3)>( r1_0.x, r1_0.y, r1_0.z );
    var _S124 : array<f32, i32(3)> = array<f32, i32(3)>( r2_0.x, r2_0.y, r2_0.z );
    var a_3 : array<array<f32, i32(3)>, i32(3)>;
    a_3[i32(0)] = _S122;
    a_3[i32(1)] = _S123;
    a_3[i32(2)] = _S124;
    var _S125 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S126 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S127 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_6 : array<array<f32, i32(3)>, i32(3)>;
    v_6[i32(0)] = _S125;
    v_6[i32(1)] = _S126;
    v_6[i32(2)] = _S127;
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
        var q_9 : u32;
        if((abs(a_3[i32(0)][i32(2)])) > (abs(a_3[u32(0)][u32(1)])))
        {
            q_9 = u32(2);
        }
        else
        {
            q_9 = u32(1);
        }
        var p_2 : u32;
        var q_10 : u32;
        if((abs(a_3[i32(1)][i32(2)])) > (abs(a_3[u32(0)][q_9])))
        {
            p_2 = u32(1);
            q_10 = u32(2);
        }
        else
        {
            p_2 = u32(0);
            q_10 = q_9;
        }
        if((abs(a_3[p_2][q_10])) <= (9.99999982451670044e-15f * max(max(abs(a_3[i32(0)][i32(0)]), abs(a_3[i32(1)][i32(1)])), max(abs(a_3[i32(2)][i32(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        var theta_0 : f32 = 0.5f * atan2(2.0f * a_3[p_2][q_10], a_3[q_10][q_10] - a_3[p_2][p_2]);
        var s_4 : f32 = sin(theta_0);
        var c_10 : f32 = cos(theta_0);
        var app_0 : f32 = a_3[p_2][p_2];
        var aqq_0 : f32 = a_3[q_10][q_10];
        var apq_0 : f32 = a_3[p_2][q_10];
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
            var _S128 : bool;
            if(k_2 != p_2)
            {
                _S128 = k_2 != q_10;
            }
            else
            {
                _S128 = false;
            }
            if(_S128)
            {
                var kp_0 : f32 = a_3[k_2][p_2];
                var kq_0 : f32 = a_3[k_2][q_10];
                var _S129 : f32 = c_10 * a_3[k_2][p_2] - s_4 * a_3[k_2][q_10];
                a_3[k_2][p_2] = _S129;
                a_3[p_2][k_2] = _S129;
                var _S130 : f32 = s_4 * kp_0 + c_10 * kq_0;
                a_3[k_2][q_10] = _S130;
                a_3[q_10][k_2] = _S130;
            }
            k_2 = k_2 + u32(1);
        }
        var _S131 : f32 = c_10 * c_10;
        var _S132 : f32 = 2.0f * s_4 * c_10 * apq_0;
        var _S133 : f32 = s_4 * s_4;
        a_3[p_2][p_2] = _S131 * app_0 - _S132 + _S133 * aqq_0;
        a_3[q_10][q_10] = _S133 * app_0 + _S132 + _S131 * aqq_0;
        a_3[p_2][q_10] = 0.0f;
        a_3[q_10][p_2] = 0.0f;
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
            var kq_1 : f32 = v_6[row_0][q_10];
            v_6[row_0][p_2] = c_10 * v_6[row_0][p_2] - s_4 * v_6[row_0][q_10];
            v_6[row_0][q_10] = s_4 * kp_1 + c_10 * kq_1;
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

fn rot_to_rubble_0( q_11 : Quat_0) -> Quat_0
{
    var axes_1 : Quat_0;
    axes_1.w_0 = 0.70710676908493042f;
    axes_1.x_1 = -0.70710676908493042f;
    axes_1.y_0 = 0.0f;
    axes_1.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_1, q_11));
}

fn stage_0( c_11 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * c_11;
}

fn quat_vec_0( q_12 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_12.x_1, q_12.y_0, q_12.z_0, q_12.w_0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_piece(@builtin(global_invocation_id) id_13 : vec3<u32>)
{
    var c_12 : u32 = id_13.x;
    var _S134 : bool = is_root_0(c_12);
    if(!_S134)
    {
        return;
    }
    var parent_0 : u32 = body_of_0(c_12);
    var pb_0 : u32 = u32(7) * parent_0;
    var at_4 : u32 = a_sums_0() + u32(20) * c_12;
    var mass_0 : f32 = read64_0(at_4, params_0.scale_m_0);
    var _S135 : f32 = read64_0(at_4 + u32(2), params_0.scale_s_0);
    var _S136 : f32 = read64_0(at_4 + u32(4), params_0.scale_s_0);
    var _S137 : f32 = read64_0(at_4 + u32(6), params_0.scale_s_0);
    var d_1 : vec3<f32> = vec3<f32>(_S135, _S136, _S137) / vec3<f32>(max(mass_0, 1.00000000317107685e-30f));
    var jxx_0 : f32 = read64_0(at_4 + u32(8), params_0.scale_j_0);
    var jyy_0 : f32 = read64_0(at_4 + u32(10), params_0.scale_j_0);
    var jzz_0 : f32 = read64_0(at_4 + u32(12), params_0.scale_j_0);
    var jxy_0 : f32 = read64_0(at_4 + u32(14), params_0.scale_j_0);
    var jxz_0 : f32 = read64_0(at_4 + u32(16), params_0.scale_j_0);
    var jyz_0 : f32 = read64_0(at_4 + u32(18), params_0.scale_j_0);
    var dd_1 : f32 = dot(d_1, d_1);
    var _S138 : f32 = d_1.x;
    var _S139 : f32 = mass_0 * _S138;
    var _S140 : f32 = d_1.y;
    var _S141 : f32 = jxy_0 + _S139 * _S140;
    var _S142 : f32 = d_1.z;
    var _S143 : f32 = jxz_0 + _S139 * _S142;
    var _S144 : f32 = jyz_0 + mass_0 * _S140 * _S142;
    var com_2 : vec3<f32> = chunks_0[u32(4) * c_12].xyz + d_1;
    var principal_0 : Quat_0;
    var moments_1 : vec3<f32>;
    principal_axes_0(vec3<f32>(jxx_0 - mass_0 * (dd_1 - _S138 * _S138), _S141, _S143), vec3<f32>(_S141, jyy_0 - mass_0 * (dd_1 - _S140 * _S140), _S144), vec3<f32>(_S143, _S144, jzz_0 - mass_0 * (dd_1 - _S142 * _S142)), &(principal_0), &(moments_1));
    var _S145 : u32 = atomicLoad(&(atom_0[a_anchor_0() + c_12]));
    var anchored_0 : bool = _S145 != u32(0);
    var q_13 : Quat_0;
    var origin_3 : vec3<f32>;
    structure_pose_1(parent_0, pb_0, &(q_13), &(origin_3));
    var p_parent_0 : vec3<f32> = bodies_0[pb_0].xyz;
    var v_parent_0 : vec3<f32> = bodies_0[pb_0 + u32(3)].xyz;
    var w_parent_0 : vec3<f32> = bodies_0[pb_0 + u32(4)].xyz;
    var pos_0 : vec3<f32> = to_rubble_0(origin_3 + rotate_0(q_13, com_2));
    var orientation_0 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(q_13, principal_0)));
    var _S146 : bool;
    if(anchored_0)
    {
        _S146 = true;
    }
    else
    {
        _S146 = !(mass_0 > 0.0f);
    }
    var inv_mass_1 : f32;
    if(_S146)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / mass_0;
    }
    var _S147 : vec3<f32> = vec3<f32>(0.0f);
    var inv_inertia_0 : vec3<f32>;
    if(!anchored_0)
    {
        var _S148 : f32;
        if((moments_1.x) > 0.0f)
        {
            _S148 = 1.0f / moments_1.x;
        }
        else
        {
            _S148 = 0.0f;
        }
        var _S149 : f32;
        if((moments_1.y) > 0.0f)
        {
            _S149 = 1.0f / moments_1.y;
        }
        else
        {
            _S149 = 0.0f;
        }
        var _S150 : f32;
        if((moments_1.z) > 0.0f)
        {
            _S150 = 1.0f / moments_1.z;
        }
        else
        {
            _S150 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S148, _S149, _S150);
    }
    else
    {
        inv_inertia_0 = _S147;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_0)
    {
        velocity_0 = _S147;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    var spin_0 : vec3<f32>;
    if(anchored_0)
    {
        spin_0 = _S147;
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
    frames_0[st_0 + u32(7)] = vec4<f32>(com_2, frames_0[u32(2) * parent_0].w);
    frames_0[st_0 + u32(8)] = quat_vec_0(principal_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_commit(@builtin(global_invocation_id) id_14 : vec3<u32>)
{
    var c_13 : u32 = id_14.x;
    var _S151 : bool = is_root_0(c_13);
    if(!_S151)
    {
        return;
    }
    var slot_5 : u32 = work_0[w_slot_0() + c_13];
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
            bodies_0[u32(7) * (slot_5 + region_0 * params_0.max_bodies_0) + w_3] = frames_0[st_1 + w_3];
            w_3 = w_3 + u32(1);
        }
        region_0 = region_0 + u32(1);
    }
    var _S152 : u32 = u32(2) * slot_5;
    frames_0[_S152] = frames_0[st_1 + u32(7)];
    frames_0[_S152 + u32(1)] = frames_0[st_1 + u32(8)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_reparent(@builtin(global_invocation_id) id_15 : vec3<u32>)
{
    var c_14 : u32 = id_15.x;
    var _S153 : bool;
    if(c_14 >= (params_0.chunk_count_0))
    {
        _S153 = true;
    }
    else
    {
        var _S154 : bool = dirty_0(body_of_0(c_14));
        _S153 = !_S154;
    }
    if(_S153)
    {
        return;
    }
    var r_7 : u32 = work_0[w_label_0() + c_14];
    var slot_6 : u32 = work_0[w_slot_0() + work_0[w_label_0() + c_14]];
    work_0[w_body_0() + c_14] = work_0[w_slot_0() + work_0[w_label_0() + c_14]];
    var _S155 : u32 = u32(2) * slot_6;
    var _S156 : vec3<f32> = frames_0[_S155].xyz;
    var _S157 : Quat_0 = conj_0(quat_of_0(frames_0[_S155 + u32(1)]));
    var inv_mass_2 : f32 = bodies_0[u32(7) * slot_6].w;
    var _S158 : f32;
    if(inv_mass_2 > 0.0f)
    {
        _S158 = 1.0f / inv_mass_2;
    }
    else
    {
        _S158 = float_bits_0(u32(2139095040));
    }
    var c2_2 : vec4<f32> = chunks_0[u32(4) * c_14 + u32(2)];
    var first_1 : u32 = (bitcast<u32>((c2_2.z)));
    var _S159 : u32 = (bitcast<u32>((c2_2.w)));
    var _S160 : vec4<f32> = chunks_0[u32(4) * r_7 + u32(3)];
    var p_3 : u32 = first_1;
    loop
    {
        if(p_3 < (first_1 + _S159))
        {
        }
        else
        {
            break;
        }
        var _S161 : u32 = u32(2) * p_3;
        var o_1 : vec4<f32> = parts_0[_S161];
        var col_0 : u32 = (bitcast<u32>((o_1.w)));
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S161 + u32(1)]);
        var _S162 : u32 = u32(4) * col_0;
        colliders_0[_S162] = vec4<f32>((bitcast<f32>((slot_6))), colliders_0[_S162].y, colliders_0[_S162].z, colliders_0[_S162].w);
        var _S163 : u32 = _S162 + u32(1);
        colliders_0[_S163] = vec4<f32>(rotate_0(_S157, o_1.xyz - _S156), colliders_0[_S163].w);
        colliders_0[_S162 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S157, rot_1)));
        var _S164 : u32 = u32(2) * col_0;
        collider_info_0[_S164] = vec4<u32>(collider_info_0[_S164].x, slot_6, collider_info_0[_S164].z, collider_info_0[_S164].w);
        collider_info_0[_S164 + u32(1)] = vec4<u32>((bitcast<u32>((_S158))), (bitcast<u32>((_S160.x))), (bitcast<u32>((_S160.w))), u32(0));
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

fn slot_color_0( slot_7 : u32) -> vec3<f32>
{
    var h_0 : f32 = fract(f32(slot_7) * 0.6180340051651001f) * 6.0f;
    var f_7 : f32 = fract(h_0);
    var q_14 : f32 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_7);
    var t_3 : f32 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_7));
    var k_3 : u32 = u32(h_0);
    var _S165 : vec3<f32>;
    if(k_3 == u32(0))
    {
        _S165 = vec3<f32>(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_3 == u32(1))
        {
            _S165 = vec3<f32>(q_14, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_3 == u32(2))
            {
                _S165 = vec3<f32>(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_3 == u32(3))
                {
                    _S165 = vec3<f32>(0.34959998726844788f, q_14, 0.92000001668930054f);
                }
                else
                {
                    if(k_3 == u32(4))
                    {
                        _S165 = vec3<f32>(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S165 = vec3<f32>(0.92000001668930054f, 0.34959998726844788f, q_14);
                    }
                }
            }
        }
    }
    return _S165;
}

fn write_object_0( o_2 : u32,  q_15 : Quat_0,  t_4 : vec3<f32>,  color_0 : vec4<f32>)
{
    var x_7 : vec3<f32> = rotate_0(q_15, vec3<f32>(1.0f, 0.0f, 0.0f));
    var y_2 : vec3<f32> = rotate_0(q_15, vec3<f32>(0.0f, 1.0f, 0.0f));
    var z_3 : vec3<f32> = rotate_0(q_15, vec3<f32>(0.0f, 0.0f, 1.0f));
    var _S166 : u32 = u32(4) * o_2;
    objects_0[_S166] = vec4<f32>(x_7.x, y_2.x, z_3.x, t_4.x);
    objects_0[_S166 + u32(1)] = vec4<f32>(x_7.y, y_2.y, z_3.y, t_4.y);
    objects_0[_S166 + u32(2)] = vec4<f32>(x_7.z, y_2.z, z_3.z, t_4.z);
    objects_0[_S166 + u32(3)] = color_0;
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
    var slot_8 : u32 = body_of_0(k_4);
    var b_5 : u32 = u32(7) * slot_8;
    var q_16 : Quat_0;
    var origin_4 : vec3<f32>;
    structure_pose_1(slot_8, b_5, &(q_16), &(origin_4));
    var _S167 : u32 = u32(3) * k_4;
    var v0_0 : vec4<f32> = view_chunks_0[_S167];
    var base_2 : vec3<f32> = view_chunks_0[_S167 + u32(2)].xyz;
    var fixed_0 : bool = (bodies_0[b_5].w) == 0.0f;
    var color_1 : vec3<f32>;
    if((view_0.mode_0) == u32(1))
    {
        var _S168 : f32 = max(length(bodies_0[b_5 + u32(3)].xyz), length(bodies_0[b_5 + u32(4)].xyz));
        if(fixed_0)
        {
            color_1 = vec3<f32>(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S168 > (view_0.rest_speed_0))
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
            color_1 = slot_color_0(slot_8);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = vec3<f32>(0.44999998807907104f) * base_2 + vec3<f32>(0.55000001192092896f) * slot_color_0(slot_8);
            }
        }
    }
    write_object_0((bitcast<u32>((v0_0.w))), normalized_0(quat_mul_0(q_16, quat_of_0(view_chunks_0[_S167 + u32(1)]))), origin_4 + rotate_0(q_16, v0_0.xyz), vec4<f32>(color_1, 1.0f));
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
    var b_6 : u32 = u32(7) * shot_bodies_0[i_3];
    var o_3 : u32 = view_0.impactor_base_0 + i_3;
    if((bodies_0[b_6].w) == 0.0f)
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
    write_object_0(o_3, rot_from_rubble_0(quat_of_0(bodies_0[b_6 + u32(1)])), from_rubble_0(bodies_0[b_6].xyz), vec4<f32>(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f));
    return;
}

fn chunk_world_0( k_5 : u32) -> vec3<f32>
{
    var slot_9 : u32 = body_of_0(k_5);
    var q_17 : Quat_0;
    var origin_5 : vec3<f32>;
    structure_pose_0(slot_9, u32(7) * slot_9, &(q_17), &(origin_5));
    return origin_5 + rotate_0(q_17, view_chunks_0[u32(3) * k_5].xyz);
}

fn write_line_vertex_0( v_8 : u32,  p_4 : vec3<f32>)
{
    var _S169 : u32 = u32(7) * v_8;
    lines_0[_S169] = p_4.x;
    lines_0[_S169 + u32(1)] = p_4.y;
    lines_0[_S169 + u32(2)] = p_4.z;
    lines_0[_S169 + u32(3)] = 1.0f;
    lines_0[_S169 + u32(4)] = 0.86000001430511475f;
    lines_0[_S169 + u32(5)] = 0.25f;
    lines_0[_S169 + u32(6)] = 1.0f;
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
    var b_7 : vec3<f32>;
    if(work_0[w_alive_0() + i_4] != u32(0))
    {
        b_7 = chunk_world_0(bond_2.y);
    }
    else
    {
        b_7 = a_4;
    }
    var _S170 : u32 = u32(2) * i_4;
    write_line_vertex_0(_S170, a_4);
    write_line_vertex_0(_S170 + u32(1), b_7);
    return;
}

fn a_strike_0() -> u32
{
    return a_counters_0() + u32(16);
}

fn strike_at_0( b_8 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * params_0.chunk_count_0 + u32(3) * b_8;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_reset(@builtin(global_invocation_id) id_20 : vec3<u32>)
{
    var b_9 : u32 = id_20.x;
    if(b_9 >= (params_0.body_count_0))
    {
        return;
    }
    var at_5 : u32 = a_strike_0() + u32(9) * b_9;
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
    frames_0[strike_at_0(b_9)][i32(3)] = 0.0f;
    return;
}

fn chunk_box_world_0( k_6 : u32,  q_18 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_10 : u32 = body_of_0(k_6);
    var origin_6 : vec3<f32>;
    structure_pose_1(slot_10, u32(7) * slot_10, &((*q_18)), &(origin_6));
    return origin_6 + rotate_0((*q_18), geometry_0[u32(3) * k_6].xyz);
}

fn ordered_0( f_8 : f32) -> u32
{
    var u_0 : u32 = (bitcast<u32>((f_8)));
    var _S171 : u32;
    if(((u_0 & (u32(2147483648)))) != u32(0))
    {
        _S171 = ~u_0;
    }
    else
    {
        _S171 = (u_0 | (u32(2147483648)));
    }
    return _S171;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_bounds(@builtin(global_invocation_id) id_21 : vec3<u32>)
{
    var k_7 : u32 = id_21.x;
    var _S172 : bool;
    if(k_7 >= (params_0.chunk_count_0))
    {
        _S172 = true;
    }
    else
    {
        _S172 = (params_0.shot_count_0) == u32(0);
    }
    if(_S172)
    {
        return;
    }
    var q_19 : Quat_0;
    var _S173 : vec3<f32> = chunk_box_world_0(k_7, &(q_19));
    var _S174 : f32 = geometry_0[u32(3) * k_7].w;
    var _S175 : u32 = a_strike_0() + u32(9) * body_of_0(k_7);
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
        var _S176 : u32 = ax_0;
        var _S177 : u32 = atomicMin(&(atom_0[_S175 + ax_0]), ordered_0(_S173[ax_0] - _S174));
        var _S178 : u32 = atomicMax(&(atom_0[_S175 + u32(3) + ax_0]), ordered_0(_S173[_S176] + _S174));
        ax_0 = ax_0 + u32(1);
    }
    return;
}

fn unordered_0( u_1 : u32) -> f32
{
    var _S179 : u32;
    if(((u_1 & (u32(2147483648)))) != u32(0))
    {
        _S179 = (u_1 & (u32(2147483647)));
    }
    else
    {
        _S179 = ~u_1;
    }
    return (bitcast<f32>((_S179)));
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_find(@builtin(global_invocation_id) id_22 : vec3<u32>)
{
    var b_10 : u32 = id_22.x;
    var _S180 : bool;
    if(b_10 >= (params_0.body_count_0))
    {
        _S180 = true;
    }
    else
    {
        _S180 = (params_0.shot_count_0) == u32(0);
    }
    if(_S180)
    {
        return;
    }
    var at_6 : u32 = a_strike_0() + u32(9) * b_10;
    var _S181 : u32 = atomicLoad(&(atom_0[at_6]));
    if(_S181 == u32(4294967295))
    {
        return;
    }
    var _S182 : u32 = atomicLoad(&(atom_0[at_6]));
    var _S183 : f32 = unordered_0(_S182);
    var _S184 : u32 = atomicLoad(&(atom_0[at_6 + u32(1)]));
    var _S185 : f32 = unordered_0(_S184);
    var _S186 : u32 = atomicLoad(&(atom_0[at_6 + u32(2)]));
    var _S187 : vec3<f32> = vec3<f32>(_S183, _S185, unordered_0(_S186));
    var _S188 : u32 = atomicLoad(&(atom_0[at_6 + u32(3)]));
    var _S189 : f32 = unordered_0(_S188);
    var _S190 : u32 = atomicLoad(&(atom_0[at_6 + u32(4)]));
    var _S191 : f32 = unordered_0(_S190);
    var _S192 : u32 = atomicLoad(&(atom_0[at_6 + u32(5)]));
    var _S193 : vec3<f32> = vec3<f32>(_S189, _S191, unordered_0(_S192));
    var _S194 : vec3<f32> = from_rubble_0(bodies_0[u32(7) * b_10 + u32(3)].xyz);
    var _S195 : f32 = float_bits_0(u32(2139095040));
    var k_8 : u32 = u32(0);
    loop
    {
        if(k_8 < (params_0.shot_count_0))
        {
        }
        else
        {
            break;
        }
        var info_1 : vec4<f32> = shots_0[u32(2) * k_8];
        var sb_0 : u32 = u32(7) * (bitcast<u32>((info_1.x)));
        if((bodies_0[sb_0].w) == 0.0f)
        {
            k_8 = k_8 + u32(1);
            continue;
        }
        var c_15 : vec3<f32> = from_rubble_0(bodies_0[sb_0].xyz);
        var rel_0 : vec3<f32> = from_rubble_0(bodies_0[sb_0 + u32(3)].xyz) - _S194;
        if(!((length(rel_0)) > (params_0.strike_speed_0)))
        {
            k_8 = k_8 + u32(1);
            continue;
        }
        var _S196 : vec3<f32> = vec3<f32>(info_1.y);
        var _S197 : vec3<f32> = _S187 - _S196;
        var _S198 : vec3<f32> = _S193 + _S196;
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
            var _S199 : u32 = ax_1;
            if((abs(rel_0[ax_1])) < 9.999999960041972e-13f)
            {
                var _S200 : u32 = ax_1;
                if((c_15[ax_1]) < (_S197[ax_1]))
                {
                    _S180 = true;
                }
                else
                {
                    _S180 = (c_15[_S200]) > (_S198[ax_1]);
                }
                var t0_1 : f32;
                if(_S180)
                {
                    t0_1 = _S195;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_1 = ax_1 + u32(1);
                continue;
            }
            var ta_0 : f32 = (_S197[ax_1] - c_15[ax_1]) / rel_0[_S199];
            var tb_0 : f32 = (_S198[ax_1] - c_15[ax_1]) / rel_0[_S199];
            var _S201 : f32 = min(t1_0, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_0 = _S201;
            ax_1 = ax_1 + u32(1);
        }
        if(t0_0 <= t1_0)
        {
            var st_2 : u32 = strike_at_0(b_10);
            frames_0[st_2] = vec4<f32>(c_15, f32(k_8 + u32(1)));
            frames_0[st_2 + u32(1)] = vec4<f32>(rel_0, t0_0);
            return;
        }
        k_8 = k_8 + u32(1);
    }
    return;
}

fn struck_0( b_11 : u32,  shot_0 : ptr<function, u32>) -> bool
{
    var k_9 : u32 = u32(frames_0[strike_at_0(b_11)].w);
    (*shot_0) = k_9 - u32(1);
    return k_9 != u32(0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_entry(@builtin(global_invocation_id) id_23 : vec3<u32>)
{
    var k_10 : u32 = id_23.x;
    var _S202 : bool;
    if(k_10 >= (params_0.chunk_count_0))
    {
        _S202 = true;
    }
    else
    {
        _S202 = (params_0.shot_count_0) == u32(0);
    }
    if(_S202)
    {
        return;
    }
    var b_12 : u32 = body_of_0(k_10);
    var shot_1 : u32;
    var _S203 : bool = struck_0(b_12, &(shot_1));
    if(!_S203)
    {
        return;
    }
    var _S204 : u32 = strike_at_0(b_12);
    var s1_0 : vec4<f32> = frames_0[_S204 + u32(1)];
    var from_1 : vec3<f32> = frames_0[_S204].xyz;
    var rel_1 : vec3<f32> = frames_0[_S204 + u32(1)].xyz;
    var d_2 : vec3<f32> = rel_1 / vec3<f32>(max(length(rel_1), 9.999999960041972e-13f));
    var _S205 : f32 = shots_0[u32(2) * shot_1].y;
    var q_20 : Quat_0;
    var cc_0 : vec3<f32> = chunk_box_world_0(k_10, &(q_20));
    var _S206 : u32 = u32(3) * k_10;
    var _S207 : Quat_0 = normalized_0(quat_mul_0(q_20, quat_of_0(geometry_0[_S206 + u32(2)])));
    var _S208 : vec3<f32> = geometry_0[_S206 + u32(1)].xyz;
    var _S209 : f32 = float_bits_0(u32(2139095040));
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
        var axis_0 : vec3<f32> = rotate_0(_S207, e_1);
        var o_4 : f32 = dot(from_1 - cc_0, axis_0);
        var dv_0 : f32 = dot(rel_1, axis_0);
        var ext_0 : f32 = _S208[ax_2] + _S205;
        if((abs(dv_0)) < 9.999999960041972e-13f)
        {
            var t_in_1 : f32;
            if((abs(o_4)) > ext_0)
            {
                t_in_1 = _S209;
            }
            else
            {
                t_in_1 = t_in_0;
            }
            t_in_0 = t_in_1;
            ax_2 = ax_2 + u32(1);
            continue;
        }
        var ta_1 : f32 = (- ext_0 - o_4) / dv_0;
        var tb_1 : f32 = (ext_0 - o_4) / dv_0;
        var _S210 : f32 = min(t_out_0, max(ta_1, tb_1));
        t_in_0 = max(t_in_0, min(ta_1, tb_1));
        t_out_0 = _S210;
        ax_2 = ax_2 + u32(1);
    }
    var at_7 : u32 = a_strike_0() + u32(9) * b_12;
    if(t_in_0 <= t_out_0)
    {
        var _S211 : u32 = atomicMin(&(atom_0[at_7 + u32(6)]), (bitcast<u32>((t_in_0))));
    }
    var r_8 : vec3<f32> = cc_0 - (from_1 + rel_1 * vec3<f32>(s1_0.w));
    var along_0 : f32 = dot(r_8, d_2);
    var _S212 : u32 = atomicMin(&(atom_0[at_7 + u32(7)]), (bitcast<u32>((length(r_8 - d_2 * vec3<f32>(along_0)) + abs(along_0) * 0.00100000004749745f))));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_nearest(@builtin(global_invocation_id) id_24 : vec3<u32>)
{
    var k_11 : u32 = id_24.x;
    var _S213 : bool;
    if(k_11 >= (params_0.chunk_count_0))
    {
        _S213 = true;
    }
    else
    {
        _S213 = (params_0.shot_count_0) == u32(0);
    }
    if(_S213)
    {
        return;
    }
    var b_13 : u32 = body_of_0(k_11);
    var shot_2 : u32;
    if(!struck_0(b_13, &(shot_2)))
    {
        return;
    }
    var _S214 : u32 = strike_at_0(b_13);
    var _S215 : vec3<f32> = frames_0[_S214 + u32(1)].xyz;
    var d_3 : vec3<f32> = _S215 / vec3<f32>(max(length(_S215), 9.999999960041972e-13f));
    var q_21 : Quat_0;
    var r_9 : vec3<f32> = chunk_box_world_0(k_11, &(q_21)) - (frames_0[_S214].xyz + _S215 * vec3<f32>(frames_0[_S214 + u32(1)].w));
    var along_1 : f32 = dot(r_9, d_3);
    var at_8 : u32 = a_strike_0() + u32(9) * b_13;
    var _S216 : u32 = (bitcast<u32>((length(r_9 - d_3 * vec3<f32>(along_1)) + abs(along_1) * 0.00100000004749745f)));
    var _S217 : u32 = atomicLoad(&(atom_0[at_8 + u32(7)]));
    if(_S216 == _S217)
    {
        var _S218 : u32 = atomicMin(&(atom_0[at_8 + u32(8)]), k_11);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_force(@builtin(global_invocation_id) id_25 : vec3<u32>)
{
    var b_14 : u32 = id_25.x;
    var _S219 : bool;
    if(b_14 >= (params_0.body_count_0))
    {
        _S219 = true;
    }
    else
    {
        _S219 = (params_0.shot_count_0) == u32(0);
    }
    if(_S219)
    {
        return;
    }
    var shot_3 : u32;
    var _S220 : bool = struck_0(b_14, &(shot_3));
    if(!_S220)
    {
        return;
    }
    var at_9 : u32 = a_strike_0() + u32(9) * b_14;
    var _S221 : u32 = atomicLoad(&(atom_0[at_9 + u32(6)]));
    var t_in_2 : f32 = (bitcast<f32>((_S221)));
    var entry_0 : u32 = atomicLoad(&(atom_0[at_9 + u32(8)]));
    var st_3 : u32 = strike_at_0(b_14);
    if(!(t_in_2 <= (params_0.dt_0)))
    {
        _S219 = true;
    }
    else
    {
        _S219 = entry_0 == u32(4294967295);
    }
    if(_S219)
    {
        frames_0[st_3][i32(3)] = 0.0f;
        return;
    }
    var rel_2 : vec3<f32> = frames_0[st_3 + u32(1)].xyz;
    var speed_0 : f32 = length(rel_2);
    var info_2 : vec4<f32> = shots_0[u32(2) * shot_3];
    var _S222 : f32 = max(info_2.z, 0.00100000004749745f);
    var m_imp_0 : f32 = info_2.w;
    var e_imp_0 : f32 = shots_0[u32(2) * shot_3 + u32(1)].x;
    var contact_0 : vec3<f32> = frames_0[st_3].xyz + rel_2 * vec3<f32>(t_in_2) + rel_2 / vec3<f32>(max(speed_0, 9.999999960041972e-13f)) * vec3<f32>(info_2.y);
    var inv_mass_3 : f32 = bodies_0[u32(7) * b_14].w;
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
    frames_0[st_3 + u32(2)] = vec4<f32>(contact_0, m_eff_0 * speed_0 / (2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (_S222 * modulus_0 * modulus_0 * max(speed_0, 0.00100000004749745f)), 0.20000000298023224f)));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_breaks(@builtin(global_invocation_id) id_26 : vec3<u32>)
{
    var i_5 : u32 = id_26.x;
    var _S223 : bool;
    if(i_5 >= (params_0.bond_count_0))
    {
        _S223 = true;
    }
    else
    {
        _S223 = (params_0.shot_count_0) == u32(0);
    }
    if(_S223)
    {
        _S223 = true;
    }
    else
    {
        _S223 = work_0[w_alive_0() + i_5] == u32(0);
    }
    if(_S223)
    {
        return;
    }
    var bond_3 : vec4<u32> = bonds_0[i_5];
    var _S224 : u32 = bond_3.x;
    var b_15 : u32 = body_of_0(_S224);
    var shot_4 : u32;
    if(!struck_0(b_15, &(shot_4)))
    {
        return;
    }
    var q_22 : Quat_0;
    var pa_0 : vec3<f32> = chunk_box_world_0(_S224, &(q_22));
    var pb_1 : vec3<f32> = chunk_box_world_0(bond_3.y, &(q_22));
    var dist_0 : f32 = length(vec3<f32>(0.5f) * (pa_0 + pb_1) - frames_0[strike_at_0(b_15) + u32(2)].xyz);
    if((frames_0[strike_at_0(b_15) + u32(2)].w * min(1.0f, (bitcast<f32>((bond_3.w))) / (6.28318548202514648f * max(dist_0 * dist_0, 9.999999960041972e-13f)))) > (bitcast<f32>((bond_3.z))))
    {
        work_0[w_alive_0() + i_5] = u32(0);
        var _S225 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_15]), u32(1));
        var _S226 : u32 = a_counters_0();
        var _S227 : u32 = atomicAdd(&(atom_0[_S226]), u32(1));
        var _S228 : u32 = atomicAdd(&(atom_0[_S226 + u32(1)]), u32(1));
    }
    return;
}

fn crushable_0( k_12 : u32) -> bool
{
    return (((bitcast<u32>((crush_tab_0[u32(2) * k_12 + u32(1)].z))) & (u32(1)))) != u32(0);
}

fn crush_flags_0( k_13 : u32) -> u32
{
    return (bitcast<u32>((crush_rw_0[k_13].w)));
}

struct Sym_0
{
     d_4 : vec3<f32>,
     o_5 : vec3<f32>,
};

fn sym_zero_0() -> Sym_0
{
    var s_6 : Sym_0;
    var _S229 : vec3<f32> = vec3<f32>(0.0f);
    s_6.d_4 = _S229;
    s_6.o_5 = _S229;
    return s_6;
}

fn sym_outer_0( r_10 : vec3<f32>,  f_9 : vec3<f32>) -> Sym_0
{
    var s_7 : Sym_0;
    s_7.d_4 = r_10 * f_9;
    var _S230 : f32 = r_10.x;
    var _S231 : f32 = f_9.y;
    var _S232 : f32 = r_10.y;
    var _S233 : f32 = f_9.x;
    var _S234 : f32 = f_9.z;
    var _S235 : f32 = r_10.z;
    s_7.o_5 = vec3<f32>(0.5f) * vec3<f32>(_S230 * _S231 + _S232 * _S233, _S230 * _S234 + _S235 * _S233, _S232 * _S234 + _S235 * _S231);
    return s_7;
}

fn sym_add_0( a_5 : Sym_0,  b_16 : Sym_0) -> Sym_0
{
    var s_8 : Sym_0;
    s_8.d_4 = a_5.d_4 + b_16.d_4;
    s_8.o_5 = a_5.o_5 + b_16.o_5;
    return s_8;
}

fn add_load_0( w_6 : ptr<function, Sym_0>,  f_sum_0 : ptr<function, vec3<f32>>,  spread_0 : ptr<function, vec4<f32>>,  r_11 : vec3<f32>,  f_10 : vec3<f32>)
{
    (*w_6) = sym_add_0((*w_6), sym_outer_0(r_11, f_10));
    (*f_sum_0) = (*f_sum_0) + f_10;
    var m_3 : f32 = length(f_10);
    (*spread_0) = (*spread_0) + vec4<f32>(vec3<f32>(m_3) * r_11, m_3);
    return;
}

fn bond_centroid_0( k_14 : u32,  q_23 : Quat_0,  origin_7 : vec3<f32>,  centre_1 : vec3<f32>) -> vec4<f32>
{
    var adj_0 : vec4<f32> = crush_tab_0[crush_0.t_adj_0 + k_14];
    var first_2 : u32 = (bitcast<u32>((adj_0.x)));
    var _S236 : u32 = (bitcast<u32>((adj_0.y)));
    var _S237 : vec3<f32> = vec3<f32>(0.0f);
    var j_3 : u32 = first_2;
    var sum_4 : vec3<f32> = _S237;
    var area_0 : f32 = 0.0f;
    loop
    {
        if(j_3 < (first_2 + _S236))
        {
        }
        else
        {
            break;
        }
        var bi_0 : u32 = (bitcast<u32>((crush_tab_0[crush_0.t_list_0 + j_3 / u32(4)][j_3 % u32(4)])));
        if(work_0[w_alive_0() + bi_0] == u32(0))
        {
            j_3 = j_3 + u32(1);
            continue;
        }
        var bc_0 : vec4<f32> = crush_tab_0[crush_0.t_bonds_0 + bi_0];
        var _S238 : f32 = bc_0.w;
        var area_1 : f32 = area_0 + _S238;
        sum_4 = sum_4 + vec3<f32>(_S238) * (origin_7 + rotate_0(q_23, bc_0.xyz) - centre_1);
        area_0 = area_1;
        j_3 = j_3 + u32(1);
    }
    if(area_0 > 0.0f)
    {
        sum_4 = sum_4 / vec3<f32>(area_0);
    }
    else
    {
        sum_4 = _S237;
    }
    return vec4<f32>(sum_4, area_0);
}

fn balanced_0( w_7 : Sym_0,  f_11 : vec3<f32>,  spread_1 : vec4<f32>,  bonds_1 : vec4<f32>) -> Sym_0
{
    var at_10 : vec3<f32>;
    if((bonds_1.w) > 0.0f)
    {
        at_10 = bonds_1.xyz;
    }
    else
    {
        var _S239 : f32 = spread_1.w;
        if(_S239 > 0.0f)
        {
            at_10 = spread_1.xyz / vec3<f32>(_S239);
        }
        else
        {
            at_10 = vec3<f32>(0.0f);
        }
    }
    return sym_add_0(w_7, sym_outer_0(at_10, (vec3<f32>(0) - f_11)));
}

fn crush_invariants_0( v_9 : Sym_0,  volume_0 : f32) -> vec2<f32>
{
    var inv_4 : f32 = 1.0f / volume_0;
    var xx_0 : f32 = v_9.d_4.x * inv_4;
    var yy_0 : f32 = v_9.d_4.y * inv_4;
    var zz_0 : f32 = v_9.d_4.z * inv_4;
    var xy_0 : f32 = v_9.o_5.x * inv_4;
    var xz_0 : f32 = v_9.o_5.y * inv_4;
    var yz_0 : f32 = v_9.o_5.z * inv_4;
    var p_5 : f32 = - (xx_0 + yy_0 + zz_0) / 3.0f;
    var dx_0 : f32 = xx_0 + p_5;
    var dy_0 : f32 = yy_0 + p_5;
    var dz_0 : f32 = zz_0 + p_5;
    return vec2<f32>(p_5, sqrt(1.5f * max(dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0 + 2.0f * (xy_0 * xy_0 + xz_0 * xz_0 + yz_0 * yz_0), 0.0f)));
}

fn crush_excess_0( pq_0 : vec2<f32>,  m0_0 : vec4<f32>) -> f32
{
    var _S240 : f32 = pq_0.x;
    if(!(_S240 > 0.0f))
    {
        return 0.0f;
    }
    var _S241 : f32 = max(pq_0.y - (m0_0.y + m0_0.z * _S240), _S240 - m0_0.x);
    var _S242 : f32;
    if(_S241 > 0.0f)
    {
        _S242 = _S241;
    }
    else
    {
        _S242 = 0.0f;
    }
    return _S242;
}

fn crush_two_0( a_6 : Sym_0,  dt_a_0 : f32,  b_17 : Sym_0,  dt_b_0 : f32,  k_15 : u32,  damage_0 : f32) -> vec4<f32>
{
    var _S243 : u32 = u32(2) * k_15;
    var m0_1 : vec4<f32> = crush_tab_0[_S243];
    var m1_0 : vec4<f32> = crush_tab_0[_S243 + u32(1)];
    var _S244 : f32 = m1_0.y;
    var pa_1 : vec2<f32> = crush_invariants_0(a_6, _S244);
    var pb_2 : vec2<f32> = crush_invariants_0(b_17, _S244);
    var ea_0 : f32 = crush_excess_0(pa_1, m0_1);
    var _S245 : bool = dt_b_0 > 0.0f;
    var eb_0 : f32;
    if(_S245)
    {
        eb_0 = crush_excess_0(pb_2, m0_1);
    }
    else
    {
        eb_0 = 0.0f;
    }
    var d_5 : f32 = damage_0 + (ea_0 * ea_0 * dt_a_0 + eb_0 * eb_0 * dt_b_0) / (m1_0.x * m0_1.w);
    var peak_2 : vec2<f32>;
    if(_S245)
    {
        peak_2 = max(pa_1, pb_2);
    }
    else
    {
        peak_2 = pa_1;
    }
    if(d_5 >= 1.0f)
    {
        return vec4<f32>(1.0f, peak_2, 1.0f);
    }
    return vec4<f32>(d_5, peak_2, 0.0f);
}

fn crush_record_0( k_16 : u32,  step_4 : vec4<f32>,  cause_0 : u32)
{
    var st_4 : vec4<f32> = crush_rw_0[k_16];
    var flags_1 : u32 = (bitcast<u32>((crush_rw_0[k_16].w)));
    var flags_2 : u32;
    if((step_4.w) != 0.0f)
    {
        flags_2 = (((((flags_1 | (u32(1)))) | (u32(2)))) | (((cause_0 << (u32(4))))));
    }
    else
    {
        flags_2 = flags_1;
    }
    crush_rw_0[k_16] = vec4<f32>(step_4.x, max(st_4.y, step_4.y), max(st_4.z, step_4.z), (bitcast<f32>((flags_2))));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_loads(@builtin(global_invocation_id) id_27 : vec3<u32>)
{
    var k_17 : u32 = id_27.x;
    var _S246 : bool;
    if(k_17 >= (params_0.chunk_count_0))
    {
        _S246 = true;
    }
    else
    {
        _S246 = (crush_0.enabled_0) == u32(0);
    }
    if(_S246)
    {
        _S246 = true;
    }
    else
    {
        _S246 = !crushable_0(k_17);
    }
    if(_S246)
    {
        _S246 = true;
    }
    else
    {
        _S246 = (((crush_flags_0(k_17)) & (u32(1)))) != u32(0);
    }
    if(_S246)
    {
        return;
    }
    var slot_11 : u32 = body_of_0(k_17);
    var b_18 : u32 = u32(7) * slot_11;
    var q_24 : Quat_0;
    var origin_8 : vec3<f32>;
    structure_pose_1(slot_11, b_18, &(q_24), &(origin_8));
    var _S247 : u32 = u32(4) * k_17;
    var c0_3 : vec4<f32> = chunks_0[_S247];
    var _S248 : vec4<f32> = chunks_0[_S247 + u32(3)];
    var centre_2 : vec3<f32> = origin_8 + rotate_0(q_24, c0_3.xyz);
    var inv_mass_4 : f32 = bodies_0[b_18].w;
    var _S249 : f32;
    if(inv_mass_4 > 0.0f)
    {
        _S249 = 1.0f / inv_mass_4;
    }
    else
    {
        _S249 = float_bits_0(u32(2139095040));
    }
    var dt_2 : f32 = params_0.dt_0;
    var _S250 : Sym_0 = sym_zero_0();
    var wa_0 : Sym_0 = _S250;
    var _S251 : vec3<f32> = vec3<f32>(0.0f);
    var fa_0 : vec3<f32> = _S251;
    var _S252 : vec4<f32> = vec4<f32>(0.0f);
    var sa_0 : vec4<f32> = _S252;
    var wb_0 : Sym_0 = _S250;
    var fb_0 : vec3<f32> = _S251;
    var sb_1 : vec4<f32> = _S252;
    var _S253 : u32 = sort_0[k_17];
    var _S254 : u32 = sort_0[params_0.chunk_count_0 + u32(1) + k_17];
    var tau_0 : f32 = 0.0f;
    var i_6 : u32 = u32(0);
    loop
    {
        if(i_6 < _S254)
        {
        }
        else
        {
            break;
        }
        var r_12 : Record_std430_0 = manifolds_0[_S253 + i_6];
        var _S255 : vec3<f32> = r_12.impulse_0.xyz;
        var j_4 : f32 = length(_S255);
        if(j_4 == 0.0f)
        {
            i_6 = i_6 + u32(1);
            continue;
        }
        var m_other_2 : f32 = r_12.other_0.x;
        var m_4 : f32;
        if(isinf_0(_S249))
        {
            if(isinf_0(m_other_2))
            {
                m_4 = c0_3.w;
            }
            else
            {
                m_4 = m_other_2;
            }
        }
        else
        {
            if(isinf_0(m_other_2))
            {
                m_4 = _S249;
            }
            else
            {
                m_4 = _S249 * m_other_2 / (_S249 + m_other_2);
            }
        }
        var _S256 : f32 = r_12.impulse_0.w;
        var _S257 : f32 = min(j_4, 2.0f * m_4 * max(_S256 - params_0.band_0, 0.0f));
        var rr_0 : vec3<f32> = r_12.point_0.xyz - centre_2;
        var dir_1 : vec3<f32> = _S255 / vec3<f32>(j_4);
        if(_S257 < j_4)
        {
            add_load_0(&(wa_0), &(fa_0), &(sa_0), rr_0, dir_1 * vec3<f32>(((j_4 - _S257) / dt_2)));
        }
        var tau_1 : f32;
        if(_S257 > 0.0f)
        {
            var _S258 : f32 = _S248.y;
            var duration_2 : f32 = hertz_duration_0(m_4, min(r_12.other_0.z, _S248.z), 1.0f / ((1.0f - _S258 * _S258) / _S248.x + 1.0f / r_12.other_0.y), _S256);
            add_load_0(&(wb_0), &(fb_0), &(sb_1), rr_0, dir_1 * vec3<f32>((3.14159274101257324f * _S257 / (2.0f * duration_2))));
            tau_1 = max(tau_0, min(duration_2, dt_2));
        }
        else
        {
            tau_1 = tau_0;
        }
        tau_0 = tau_1;
        i_6 = i_6 + u32(1);
    }
    var _S259 : u32 = w_pulses_0() + _S247 + u32(2) * params_0.parity_0;
    var _S260 : u32 = work_0[_S259 + u32(1)];
    var _S261 : u32 = l_pool_0(params_0.parity_0) + u32(3) * work_0[_S259];
    var p_6 : u32 = u32(0);
    loop
    {
        if(p_6 < _S260)
        {
        }
        else
        {
            break;
        }
        var _S262 : u32 = _S261 + u32(3) * p_6;
        var bb_0 : vec4<f32> = loads_0[_S262 + u32(1)];
        var c_16 : vec4<f32> = loads_0[_S262 + u32(2)];
        var mean_2 : f32 = pulse_mean_0(loads_0[_S262].x, loads_0[_S262].y, loads_0[_S262].z, loads_0[_S262].w, loads_0[_S262 + u32(1)].w, dt_2);
        if(mean_2 != 0.0f)
        {
            add_load_0(&(wa_0), &(fa_0), &(sa_0), bb_0.xyz - centre_2, c_16.xyz * vec3<f32>(mean_2));
        }
        p_6 = p_6 + u32(1);
    }
    if(!((sa_0.w) > 0.0f))
    {
        _S246 = !((sb_1.w) > 0.0f);
    }
    else
    {
        _S246 = false;
    }
    if(_S246)
    {
        return;
    }
    var bonds_2 : vec4<f32> = bond_centroid_0(k_17, q_24, origin_8, centre_2);
    crush_record_0(k_17, crush_two_0(balanced_0(wa_0, fa_0, sa_0, bonds_2), dt_2 - tau_0, balanced_0(sym_add_0(wa_0, wb_0), fa_0 + fb_0, sa_0 + sb_1, bonds_2), tau_0, k_17, crush_rw_0[k_17].x), u32(1));
    return;
}

fn a_crush_0() -> u32
{
    return a_strike_0() + u32(9) * params_0.body_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_entry(@builtin(global_invocation_id) id_28 : vec3<u32>)
{
    var k_18 : u32 = id_28.x;
    var _S263 : bool;
    if(k_18 >= (params_0.chunk_count_0))
    {
        _S263 = true;
    }
    else
    {
        _S263 = (params_0.shot_count_0) == u32(0);
    }
    if(_S263)
    {
        _S263 = true;
    }
    else
    {
        _S263 = (crush_0.enabled_0) == u32(0);
    }
    if(_S263)
    {
        return;
    }
    var b_19 : u32 = body_of_0(k_18);
    var shot_5 : u32;
    if(!struck_0(b_19, &(shot_5)))
    {
        return;
    }
    var _S264 : u32 = strike_at_0(b_19);
    var _S265 : vec3<f32> = frames_0[_S264 + u32(1)].xyz;
    var d_6 : vec3<f32> = _S265 / vec3<f32>(max(length(_S265), 9.999999960041972e-13f));
    var q_25 : Quat_0;
    var r_13 : vec3<f32> = chunk_box_world_0(k_18, &(q_25)) - (frames_0[_S264].xyz + _S265 * vec3<f32>(frames_0[_S264 + u32(1)].w));
    var along_2 : f32 = dot(r_13, d_6);
    var dist_1 : f32 = length(r_13 - d_6 * vec3<f32>(along_2)) + abs(along_2) * 0.00100000004749745f;
    crush_rw_0[crush_0.r_dist_0 + k_18] = vec4<f32>(dist_1, 0.0f, 0.0f, 0.0f);
    var _S266 : u32 = atomicMax(&(atom_0[a_crush_0() + u32(2) * b_19]), ~(bitcast<u32>((dist_1))));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_nearest(@builtin(global_invocation_id) id_29 : vec3<u32>)
{
    var k_19 : u32 = id_29.x;
    var _S267 : bool;
    if(k_19 >= (params_0.chunk_count_0))
    {
        _S267 = true;
    }
    else
    {
        _S267 = (params_0.shot_count_0) == u32(0);
    }
    if(_S267)
    {
        _S267 = true;
    }
    else
    {
        _S267 = (crush_0.enabled_0) == u32(0);
    }
    if(_S267)
    {
        return;
    }
    var b_20 : u32 = body_of_0(k_19);
    var shot_6 : u32;
    if(!struck_0(b_20, &(shot_6)))
    {
        return;
    }
    var _S268 : u32 = ~(bitcast<u32>((crush_rw_0[crush_0.r_dist_0 + k_19].x)));
    var _S269 : u32 = a_crush_0() + u32(2) * b_20;
    var _S270 : u32 = atomicLoad(&(atom_0[_S269]));
    if(_S268 == _S270)
    {
        var _S271 : u32 = atomicMax(&(atom_0[_S269 + u32(1)]), ~k_19);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_crush(@builtin(global_invocation_id) id_30 : vec3<u32>)
{
    var b_21 : u32 = id_30.x;
    var _S272 : bool;
    if(b_21 >= (params_0.body_count_0))
    {
        _S272 = true;
    }
    else
    {
        _S272 = (params_0.shot_count_0) == u32(0);
    }
    if(_S272)
    {
        _S272 = true;
    }
    else
    {
        _S272 = (crush_0.enabled_0) == u32(0);
    }
    if(_S272)
    {
        return;
    }
    var _S273 : u32 = u32(2) * b_21;
    var _S274 : u32 = a_crush_0() + _S273;
    var _S275 : u32 = _S274 + u32(1);
    var _S276 : u32 = atomicLoad(&(atom_0[_S275]));
    var k_20 : u32 = ~_S276;
    atomicStore(&(atom_0[_S274]), u32(0));
    atomicStore(&(atom_0[_S275]), u32(0));
    crush_rw_0[crush_0.r_bill_0 + _S273] = vec4<f32>(0.0f);
    var shot_7 : u32;
    var _S277 : bool = struck_0(b_21, &(shot_7));
    if(!_S277)
    {
        return;
    }
    var _S278 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_21 + u32(6)]));
    var t_in_3 : f32 = (bitcast<f32>((_S278)));
    if(!(t_in_3 <= (params_0.dt_0)))
    {
        _S272 = true;
    }
    else
    {
        _S272 = k_20 == u32(4294967295);
    }
    if(_S272)
    {
        _S272 = true;
    }
    else
    {
        _S272 = !crushable_0(k_20);
    }
    if(_S272)
    {
        _S272 = true;
    }
    else
    {
        _S272 = (((crush_flags_0(k_20)) & (u32(1)))) != u32(0);
    }
    if(_S272)
    {
        return;
    }
    var st_5 : u32 = strike_at_0(b_21);
    var s0_0 : vec4<f32> = frames_0[st_5];
    var _S279 : vec3<f32> = frames_0[st_5 + u32(1)].xyz;
    var speed_1 : f32 = length(_S279);
    var d_7 : vec3<f32> = _S279 / vec3<f32>(max(speed_1, 9.999999960041972e-13f));
    var info_3 : vec4<f32> = shots_0[u32(2) * shot_7];
    var radius_0 : f32 = info_3.y;
    var _S280 : f32 = max(info_3.z, 0.00100000004749745f);
    var m_imp_1 : f32 = info_3.w;
    var _S281 : u32 = u32(7) * b_21;
    var inv_mass_5 : f32 = bodies_0[_S281].w;
    var _S282 : bool = inv_mass_5 == 0.0f;
    var m_body_2 : f32;
    if(_S282)
    {
        m_body_2 = 0.0f;
    }
    else
    {
        m_body_2 = 1.0f / inv_mass_5;
    }
    var m_eff_1 : f32;
    if(_S282)
    {
        m_eff_1 = m_imp_1;
    }
    else
    {
        m_eff_1 = m_imp_1 * m_body_2 / (m_imp_1 + m_body_2);
    }
    var modulus_1 : f32 = 1.0f / (1.0f / shots_0[u32(2) * shot_7 + u32(1)].x + 1.0f / geometry_0[u32(3) * k_20 + u32(1)].w);
    var tau_2 : f32 = 2.86999988555908203f * pow(m_eff_1 * m_eff_1 / (_S280 * modulus_1 * modulus_1 * max(speed_1, 0.00100000004749745f)), 0.20000000298023224f);
    var contact_1 : vec3<f32> = s0_0.xyz + _S279 * vec3<f32>(t_in_3) + d_7 * vec3<f32>(radius_0);
    var f_12 : vec3<f32> = d_7 * vec3<f32>((m_eff_1 * speed_1 / tau_2));
    var q_26 : Quat_0;
    var origin_9 : vec3<f32>;
    structure_pose_1(b_21, _S281, &(q_26), &(origin_9));
    var centre_3 : vec3<f32> = origin_9 + rotate_0(q_26, chunks_0[u32(4) * k_20].xyz);
    var bonds_3 : vec4<f32> = bond_centroid_0(k_20, q_26, origin_9, centre_3);
    if(!((bonds_3.w) > 0.0f))
    {
        return;
    }
    var w_8 : Sym_0 = sym_add_0(sym_outer_0(contact_1 - centre_3, f_12), sym_outer_0(bonds_3.xyz, (vec3<f32>(0) - f_12)));
    var before_0 : f32 = crush_rw_0[k_20].x;
    var step_5 : vec4<f32> = crush_two_0(w_8, min(tau_2, params_0.dt_0), w_8, 0.0f, k_20, before_0);
    crush_record_0(k_20, step_5, u32(2));
    var _S283 : u32 = u32(2) * k_20;
    crush_rw_0[crush_0.r_bill_0 + _S273] = vec4<f32>((step_5.x - before_0) * crush_tab_0[_S283].w * crush_tab_0[_S283 + u32(1)].y, f32(shot_7 + u32(1)), 0.0f, 0.0f);
    crush_rw_0[crush_0.r_bill_0 + _S273 + u32(1)] = vec4<f32>(contact_1, 0.0f);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_bill(@builtin(global_invocation_id) id_31 : vec3<u32>)
{
    var s_9 : u32 = id_31.x;
    var _S284 : bool;
    if(s_9 >= (params_0.shot_count_0))
    {
        _S284 = true;
    }
    else
    {
        _S284 = (crush_0.enabled_0) == u32(0);
    }
    if(_S284)
    {
        return;
    }
    var b_22 : u32 = u32(0);
    var energy_0 : f32 = 0.0f;
    loop
    {
        if(b_22 < (params_0.body_count_0))
        {
        }
        else
        {
            break;
        }
        var bill_0 : vec4<f32> = crush_rw_0[crush_0.r_bill_0 + u32(2) * b_22];
        if((crush_rw_0[crush_0.r_bill_0 + u32(2) * b_22].y) == f32(s_9 + u32(1)))
        {
            energy_0 = energy_0 + bill_0.x;
        }
        b_22 = b_22 + u32(1);
    }
    crush_rw_0[crush_0.r_shot_0 + s_9] = vec4<f32>(0.0f);
    if(!(energy_0 > 0.0f))
    {
        return;
    }
    var sb_2 : u32 = u32(7) * (bitcast<u32>((shots_0[u32(2) * s_9].x)));
    var inv_mass_6 : f32 = bodies_0[sb_2].w;
    if(!(inv_mass_6 > 0.0f))
    {
        return;
    }
    var m_5 : f32 = 1.0f / inv_mass_6;
    var v_10 : vec3<f32> = bodies_0[sb_2 + u32(3)].xyz;
    var speed_2 : f32 = length(v_10);
    var j_5 : f32 = m_5 * (speed_2 - sqrt(max(speed_2 * speed_2 - 2.0f * energy_0 / m_5, 0.0f)));
    var _S285 : bool = speed_2 > 0.0f;
    var _S286 : vec3<f32>;
    if(_S285)
    {
        _S286 = (vec3<f32>(0) - v_10) * vec3<f32>((j_5 / (m_5 * speed_2)));
    }
    else
    {
        _S286 = vec3<f32>(0.0f);
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
        var w_9 : u32 = sb_2 + region_1 * u32(7) * params_0.max_bodies_0 + u32(3);
        bodies_0[w_9] = vec4<f32>(bodies_0[w_9].xyz + _S286, bodies_0[w_9].w);
        region_1 = region_1 + u32(1);
    }
    if(_S285)
    {
        _S286 = v_10 * vec3<f32>((j_5 / speed_2));
    }
    else
    {
        _S286 = vec3<f32>(0.0f);
    }
    crush_rw_0[crush_0.r_shot_0 + s_9] = vec4<f32>(_S286, energy_0);
    return;
}

fn inverse_rotate_0( q_27 : Quat_0,  v_11 : vec3<f32>) -> vec3<f32>
{
    var c_17 : Quat_0;
    c_17.w_0 = q_27.w_0;
    c_17.x_1 = - q_27.x_1;
    c_17.y_0 = - q_27.y_0;
    c_17.z_0 = - q_27.z_0;
    return rotate_0(c_17, v_11);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_push(@builtin(global_invocation_id) id_32 : vec3<u32>)
{
    var b_23 : u32 = id_32.x;
    var _S287 : bool;
    if(b_23 >= (params_0.body_count_0))
    {
        _S287 = true;
    }
    else
    {
        _S287 = (params_0.shot_count_0) == u32(0);
    }
    if(_S287)
    {
        _S287 = true;
    }
    else
    {
        _S287 = (crush_0.enabled_0) == u32(0);
    }
    if(_S287)
    {
        return;
    }
    var _S288 : u32 = u32(2) * b_23;
    var bill_1 : vec4<f32> = crush_rw_0[crush_0.r_bill_0 + _S288];
    var _S289 : f32 = crush_rw_0[crush_0.r_bill_0 + _S288].x;
    if(!(_S289 > 0.0f))
    {
        return;
    }
    var paid_0 : vec4<f32> = crush_rw_0[crush_0.r_shot_0 + (u32(bill_1.y) - u32(1))];
    var bw_0 : u32 = u32(7) * b_23;
    var inv_mass_7 : f32 = bodies_0[bw_0].w;
    var _S290 : f32 = crush_rw_0[crush_0.r_shot_0 + (u32(bill_1.y) - u32(1))].w;
    if(!(_S290 > 0.0f))
    {
        _S287 = true;
    }
    else
    {
        _S287 = inv_mass_7 == 0.0f;
    }
    if(_S287)
    {
        return;
    }
    var j_6 : vec3<f32> = paid_0.xyz * vec3<f32>((_S289 / _S290));
    var o_6 : Quat_0 = quat_of_0(bodies_0[bw_0 + u32(1)]);
    var _S291 : vec3<f32> = rotate_0(o_6, bodies_0[bw_0 + u32(2)].xyz * inverse_rotate_0(o_6, cross(to_rubble_0(crush_rw_0[crush_0.r_bill_0 + _S288 + u32(1)].xyz) - bodies_0[bw_0].xyz, j_6)));
    var region_2 : u32 = u32(0);
    loop
    {
        if(region_2 < u32(2))
        {
        }
        else
        {
            break;
        }
        var w_10 : u32 = bw_0 + region_2 * u32(7) * params_0.max_bodies_0;
        var _S292 : u32 = w_10 + u32(3);
        bodies_0[_S292] = vec4<f32>(bodies_0[_S292].xyz + j_6 * vec3<f32>(inv_mass_7), bodies_0[_S292].w);
        var _S293 : u32 = w_10 + u32(4);
        bodies_0[_S293] = vec4<f32>(bodies_0[_S293].xyz + _S291, bodies_0[_S293].w);
        region_2 = region_2 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_bonds(@builtin(global_invocation_id) id_33 : vec3<u32>)
{
    var i_7 : u32 = id_33.x;
    var _S294 : bool;
    if(i_7 >= (params_0.bond_count_0))
    {
        _S294 = true;
    }
    else
    {
        _S294 = (crush_0.enabled_0) == u32(0);
    }
    if(_S294)
    {
        _S294 = true;
    }
    else
    {
        _S294 = work_0[w_alive_0() + i_7] == u32(0);
    }
    if(_S294)
    {
        return;
    }
    var bond_4 : vec4<u32> = bonds_0[i_7];
    var _S295 : u32 = bond_4.x;
    if((((((bitcast<u32>((crush_rw_0[_S295].w))) | ((bitcast<u32>((crush_rw_0[bond_4.y].w)))))) & (u32(2)))) == u32(0))
    {
        return;
    }
    work_0[w_alive_0() + i_7] = u32(0);
    var _S296 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S295)]), u32(1));
    var _S297 : u32 = a_counters_0();
    var _S298 : u32 = atomicAdd(&(atom_0[_S297]), u32(1));
    var _S299 : u32 = atomicAdd(&(atom_0[_S297 + u32(1)]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_mark(@builtin(global_invocation_id) id_34 : vec3<u32>)
{
    var c_18 : u32 = id_34.x;
    var _S300 : bool;
    if(c_18 >= (params_0.chunk_count_0))
    {
        _S300 = true;
    }
    else
    {
        _S300 = (crush_0.enabled_0) == u32(0);
    }
    if(_S300)
    {
        return;
    }
    var now_0 : u32;
    if((((crush_flags_0(c_18)) & (u32(2)))) != u32(0))
    {
        now_0 = u32(1);
    }
    else
    {
        now_0 = u32(0);
    }
    work_0[w_new_0() + c_18] = now_0;
    if(now_0 != u32(0))
    {
        var _S301 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(10)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_crush_gate(@builtin(global_invocation_id) id_35 : vec3<u32>)
{
    var run_2 : bool;
    if((crush_0.enabled_0) != u32(0))
    {
        var _S302 : u32 = atomicLoad(&(atom_0[a_counters_0() + u32(10)]));
        run_2 = _S302 != u32(0);
    }
    else
    {
        run_2 = false;
    }
    var _S303 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S303 + u32(10)]), u32(0));
    atomicStore(&(atom_0[_S303 + u32(7)]), u32(0));
    var _S304 : u32;
    if(run_2)
    {
        var _S305 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S304 = _S305;
    }
    else
    {
        _S304 = u32(0);
    }
    set_args_0(u32(6), _S304);
    if(run_2)
    {
        var _S306 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S304 = _S306;
    }
    else
    {
        _S304 = u32(0);
    }
    set_args_0(u32(7), _S304);
    var _S307 : i32;
    if(run_2)
    {
        _S307 = i32(1);
    }
    else
    {
        _S307 = i32(0);
    }
    set_args_0(u32(8), u32(_S307));
    return;
}

fn write_body_0( slot_12 : u32,  pos_1 : vec3<f32>,  inv_mass_8 : f32,  orientation_1 : Quat_0,  inv_inertia_1 : vec3<f32>,  flags_3 : u32,  v_12 : vec3<f32>,  w_11 : vec3<f32>)
{
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
        var b_24 : u32 = u32(7) * (slot_12 + region_3 * params_0.max_bodies_0);
        bodies_0[b_24] = vec4<f32>(pos_1, inv_mass_8);
        bodies_0[b_24 + u32(1)] = quat_vec_0(orientation_1);
        bodies_0[b_24 + u32(2)] = vec4<f32>(inv_inertia_1, (bitcast<f32>((flags_3))));
        bodies_0[b_24 + u32(3)] = vec4<f32>(v_12, 0.0f);
        bodies_0[b_24 + u32(4)] = vec4<f32>(w_11, (bitcast<f32>((params_0.step_0))));
        bodies_0[b_24 + u32(5)] = vec4<f32>(0.0f);
        bodies_0[b_24 + u32(6)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
        region_3 = region_3 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_apply(@builtin(global_invocation_id) id_36 : vec3<u32>)
{
    var ax_3 : u32;
    var k_21 : u32 = id_36.x;
    var _S308 : bool;
    if(k_21 >= (params_0.chunk_count_0))
    {
        _S308 = true;
    }
    else
    {
        _S308 = (crush_0.enabled_0) == u32(0);
    }
    if(_S308)
    {
        return;
    }
    var cs_0 : vec4<f32> = crush_rw_0[k_21];
    var flags_4 : u32 = (bitcast<u32>((crush_rw_0[k_21].w)));
    if(((flags_4 & (u32(2)))) == u32(0))
    {
        return;
    }
    crush_rw_0[k_21] = vec4<f32>(cs_0.xyz, (bitcast<f32>(((flags_4 & (u32(4294967293)))))));
    var slot_13 : u32 = body_of_0(k_21);
    var bw_1 : u32 = u32(7) * slot_13;
    var b0_0 : vec4<f32> = bodies_0[bw_1];
    var orientation_2 : Quat_0 = quat_of_0(bodies_0[bw_1 + u32(1)]);
    var v_13 : vec3<f32> = bodies_0[bw_1 + u32(3)].xyz;
    var w_12 : vec3<f32> = bodies_0[bw_1 + u32(4)].xyz;
    var _S309 : u32 = u32(4) * k_21;
    var mass_1 : f32 = chunks_0[_S309].w;
    var volume_1 : f32 = crush_tab_0[u32(2) * k_21 + u32(1)].y;
    var rank_0 : u32 = work_0[w_rank_0() + k_21];
    if(work_0[w_rank_0() + k_21] >= (crush_0.max_events_0))
    {
        var _S310 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(32));
    }
    var _S311 : vec3<f32> = vec3<f32>(0.0f);
    if((crush_0.mode_1) == u32(1))
    {
        _S308 = (crush_0.fragments_0) > u32(0);
    }
    else
    {
        _S308 = false;
    }
    var first_3 : u32;
    var half_1 : vec3<f32>;
    if(_S308)
    {
        var _S312 : u32 = a_counters_0();
        var _S313 : u32 = atomicLoad(&(atom_0[_S312 + u32(8)]));
        var base_3 : u32 = _S313 + rank_0 * crush_0.fragments_0;
        if((base_3 + crush_0.fragments_0) > (crush_0.pool_size_0))
        {
            var _S314 : u32 = atomicOr(&(atom_0[_S312 + u32(3)]), u32(16));
            first_3 = u32(4294967295);
        }
        else
        {
            first_3 = base_3;
        }
        var q_28 : Quat_0;
        var origin_10 : vec3<f32>;
        structure_pose_1(slot_13, bw_1, &(q_28), &(origin_10));
        var _S315 : vec3<f32> = origin_10 + rotate_0(q_28, chunks_0[_S309].xyz);
        var _S316 : u32 = u32(3) * k_21;
        var box_0 : Quat_0 = normalized_0(quat_mul_0(q_28, quat_of_0(geometry_0[_S316 + u32(2)])));
        var h_1 : vec3<f32> = geometry_0[_S316 + u32(1)].xyz;
        var g_0 : vec3<u32> = vec3<u32>(vec3<i32>(i32(1)));
        var cell_0 : vec3<f32> = h_1;
        var n_3 : u32 = crush_0.fragments_0;
        loop
        {
            if(n_3 > u32(1))
            {
            }
            else
            {
                break;
            }
            if((cell_0.x) >= (cell_0.y))
            {
                _S308 = (cell_0.x) >= (cell_0.z);
            }
            else
            {
                _S308 = false;
            }
            if(_S308)
            {
                ax_3 = u32(0);
            }
            else
            {
                if((cell_0.y) >= (cell_0.z))
                {
                    ax_3 = u32(1);
                }
                else
                {
                    ax_3 = u32(2);
                }
            }
            g_0[ax_3] = g_0[ax_3] * u32(2);
            cell_0[ax_3] = cell_0[ax_3] * 0.5f;
            n_3 = (n_3 >> (u32(1)));
        }
        var box_volume_0 : f32 = 8.0f * h_1.x * h_1.y * h_1.z;
        var shrink_0 : f32;
        if(box_volume_0 > 0.0f)
        {
            shrink_0 = min(0.94999998807907104f, pow(volume_1 / box_volume_0, 0.3333333432674408f));
        }
        else
        {
            shrink_0 = 0.94999998807907104f;
        }
        var half_2 : vec3<f32> = cell_0 * vec3<f32>(shrink_0);
        var mf_0 : f32 = mass_1 / f32(crush_0.fragments_0);
        var _S317 : f32 = half_2.y;
        var _S318 : f32 = _S317 * _S317;
        var _S319 : f32 = half_2.z;
        var _S320 : f32 = _S319 * _S319;
        var _S321 : f32 = half_2.x;
        var _S322 : f32 = _S321 * _S321;
        var inertia_0 : vec3<f32> = vec3<f32>((mf_0 / 3.0f)) * vec3<f32>(_S318 + _S320, _S322 + _S320, _S322 + _S318);
        var _S323 : vec3<f32> = vec3<f32>(1.0f / inertia_0.x, 1.0f / inertia_0.y, 1.0f / inertia_0.z);
        var _S324 : Quat_0 = rot_to_rubble_0(box_0);
        var _S325 : f32 = chunks_0[_S309 + u32(3)].x;
        var _S326 : f32 = min(_S321, min(_S317, _S319));
        ax_3 = u32(0);
        loop
        {
            if(ax_3 < (crush_0.fragments_0))
            {
                _S308 = first_3 != u32(4294967295);
            }
            else
            {
                _S308 = false;
            }
            if(_S308)
            {
            }
            else
            {
                break;
            }
            var _S327 : u32 = ax_3 % g_0.x;
            var _S328 : u32 = ax_3 / g_0.x;
            var _S329 : u32 = _S328 % g_0.y;
            var _S330 : u32 = ax_3 / (g_0.x * g_0.y);
            var x_8 : vec3<f32> = to_rubble_0(_S315 + rotate_0(box_0, (vec3<f32>(0) - h_1) + cell_0 * (vec3<f32>(2.0f) * vec3<f32>(vec3<u32>(_S327, _S329, _S330)) + vec3<f32>(1.0f))));
            var entry_1 : vec4<f32> = crush_tab_0[crush_0.t_pool_0 + first_3 + ax_3];
            var fs_0 : u32 = (bitcast<u32>((entry_1.x)));
            var fc_0 : u32 = (bitcast<u32>((entry_1.y)));
            write_body_0(fs_0, x_8, 1.0f / mf_0, _S324, _S323, u32(9), v_13 + cross(w_12, x_8 - b0_0.xyz), w_12);
            var _S331 : u32 = u32(4) * fc_0;
            var _S332 : u32 = _S331 + u32(1);
            colliders_0[_S332] = vec4<f32>(0.0f, 0.0f, 0.0f, colliders_0[_S332].w);
            colliders_0[_S331 + u32(2)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
            colliders_0[_S331 + u32(3)] = vec4<f32>(half_2, length(half_2));
            var _S333 : u32 = u32(2) * fc_0;
            collider_info_0[_S333] = vec4<u32>(u32(4294967295), fs_0, u32(2), collider_info_0[_S333].w);
            collider_info_0[_S333 + u32(1)] = vec4<u32>((bitcast<u32>((mf_0))), (bitcast<u32>((_S325))), (bitcast<u32>((_S326))), u32(0));
            ax_3 = ax_3 + u32(1);
        }
        half_1 = half_2;
    }
    else
    {
        first_3 = u32(4294967295);
        half_1 = _S311;
    }
    if(rank_0 < (crush_0.max_events_0))
    {
        var ev_0 : u32 = crush_0.r_events_0 + u32(6) * rank_0;
        crush_rw_0[ev_0] = vec4<f32>((bitcast<f32>((k_21))), (bitcast<f32>((first_3))), (bitcast<f32>(((flags_4 >> (u32(4)))))), cs_0.x);
        crush_rw_0[ev_0 + u32(1)] = vec4<f32>(b0_0.xyz, mass_1);
        crush_rw_0[ev_0 + u32(2)] = quat_vec_0(orientation_2);
        crush_rw_0[ev_0 + u32(3)] = vec4<f32>(v_13, cs_0.y);
        crush_rw_0[ev_0 + u32(4)] = vec4<f32>(w_12, cs_0.z);
        crush_rw_0[ev_0 + u32(5)] = vec4<f32>(half_1, volume_1);
    }
    write_body_0(slot_13, vec3<f32>(f32(slot_13 % u32(1024)) * crush_0.park_step_0, crush_0.park_y_0, f32(slot_13 / u32(1024)) * crush_0.park_step_0), 0.0f, orientation_2, _S311, u32(0), _S311, _S311);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_crush_end(@builtin(global_invocation_id) id_37 : vec3<u32>)
{
    var total_0 : u32 = work_0[scan_new_blocks_0() + (params_0.chunk_count_0 + u32(1023)) / u32(1024)];
    var _S334 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S334 + u32(7)]), work_0[scan_new_blocks_0() + (params_0.chunk_count_0 + u32(1023)) / u32(1024)]);
    var _S335 : u32 = atomicAdd(&(atom_0[_S334 + u32(9)]), total_0);
    if((crush_0.mode_1) == u32(1))
    {
        var _S336 : u32 = atomicAdd(&(atom_0[_S334 + u32(8)]), total_0 * crush_0.fragments_0);
    }
    return;
}

