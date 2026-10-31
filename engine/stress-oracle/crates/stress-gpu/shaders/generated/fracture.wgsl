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
    @align(16) park_y_0 : f32,
    @align(4) park_step_0 : f32,
    @align(8) flow_0 : u32,
    @align(4) t_normal_0 : u32,
    @align(16) r_flow_0 : u32,
    @align(4) r_blast_0 : u32,
    @align(8) pad0_1 : f32,
    @align(4) pad1_1 : f32,
};

@binding(25) @group(0) var<uniform> crush_0 : CrushParams_std140_0;
@binding(24) @group(0) var<storage, read_write> crush_rw_0 : array<vec4<f32>>;

@binding(23) @group(0) var<storage, read> crush_tab_0 : array<vec4<f32>>;

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

fn structure_pose_4( slot_4 : u32,  b_4 : u32,  q_9 : ptr<function, Quat_0>,  origin_4 : ptr<function, vec3<f32>>)
{
    var _S16 : u32 = u32(2) * slot_4;
    var com_4 : vec3<f32> = frames_0[_S16].xyz;
    var _S17 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_4 + u32(1)])), conj_0(quat_of_0(frames_0[_S16 + u32(1)]))));
    (*q_9) = _S17;
    (*origin_4) = from_rubble_0(bodies_0[b_4].xyz) - rotate_0(_S17, com_4);
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
    var _S18 : f32 = start_0 + plateau_time_0;
    var sum_0 : f32 = plateau_0 * max(min(_S18, dt_1) - max(start_0, 0.0f), 0.0f);
    var _S19 : f32 = max(0.0f, _S18);
    var _S20 : f32 = min(dt_1, _S18 + duration_0);
    var _S21 : bool;
    if(_S20 > _S19)
    {
        _S21 = duration_0 > 0.0f;
    }
    else
    {
        _S21 = false;
    }
    var sum_1 : f32;
    if(_S21)
    {
        var w_1 : f32 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_1 * (cos(w_1 * (_S19 - _S18)) - cos(w_1 * (_S20 - _S18)));
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
    var slot_5 : u32 = body_of_0(k_0);
    var b_5 : u32 = u32(7) * slot_5;
    var q_10 : Quat_0;
    var origin_5 : vec3<f32>;
    structure_pose_4(slot_5, b_5, &(q_10), &(origin_5));
    var _S22 : u32 = u32(4) * k_0;
    var c0_0 : vec4<f32> = chunks_0[_S22];
    var centre_0 : vec3<f32> = origin_5 + rotate_0(q_10, c0_0.xyz);
    var inv_mass_0 : f32 = bodies_0[b_5].w;
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
    var _S23 : u32 = w_pulses_0() + _S22;
    var _S24 : u32 = _S23 + u32(2) * params_0.parity_0;
    var off_0 : u32 = work_0[_S24];
    var carried_0 : u32 = work_0[_S24 + u32(1)];
    var room_0 : u32 = work_0[_S24 + u32(1)] + sort_0[params_0.chunk_count_0 + u32(1) + k_0];
    var base_0 : u32;
    var room_1 : u32;
    if(room_0 != u32(0))
    {
        var _S25 : u32 = a_counters_0();
        var base_1 : u32 = atomicAdd(&(atom_0[_S25 + u32(5) + to_0]), room_0);
        if((base_1 + room_0) > (params_0.pool_cap_0))
        {
            var _S26 : u32 = atomicOr(&(atom_0[_S25 + u32(3)]), u32(8));
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
        var _S27 : u32 = base_0;
        base_0 = base_1;
        room_1 = _S27;
    }
    else
    {
        base_0 = u32(0);
        room_1 = room_0;
    }
    var _S28 : u32 = params_0.chunk_count_0;
    var _S29 : f32 = c0_0.w;
    var _S30 : vec4<f32> = chunks_0[_S22 + u32(3)];
    var _S31 : f32 = params_0.band_0;
    var _S32 : f32 = params_0.dt_0;
    var _S33 : bool = (params_0.impact_peak_0) != u32(0);
    var _S34 : u32 = l_pool_0(from_0) + u32(3) * off_0;
    var _S35 : u32 = l_pool_0(to_0) + u32(3) * base_0;
    loop
    {
        var _S36 : vec3<f32> = vec3<f32>(0.0f);
        var _S37 : u32 = sort_0[k_0];
        var _S38 : u32 = sort_0[_S28 + u32(1) + k_0];
        var first_0 : bool = true;
        var last_0 : u32 = u32(0);
        var step_1 : u32 = u32(0);
        f_total_0 = _S36;
        var t_total_1 : vec3<f32> = _S36;
        loop
        {
            if(step_1 < _S38)
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
                if(q3_0 < _S38)
                {
                }
                else
                {
                    break;
                }
                var key_1 : u32 = manifolds_0[_S37 + q3_0].ids_0.y + u32(1);
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
            var r_4 : Record_std430_0 = manifolds_0[_S37 + p_0];
            var _S39 : vec3<f32> = r_4.impulse_0.xyz;
            var j_0 : f32 = length(_S39);
            if(j_0 == 0.0f)
            {
                var _S40 : u32 = step_1 + u32(1);
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S40;
                continue;
            }
            var m_other_0 : f32 = r_4.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_0))
                {
                    m_1 = _S29;
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
            var _S41 : f32 = min(j_0, 2.0f * m_1 * max(r_4.impulse_0.w - _S31, 0.0f));
            if(_S41 < j_0)
            {
                var f_0 : vec3<f32> = _S39 / vec3<f32>(j_0) * vec3<f32>(((j_0 - _S41) / _S32));
                var _S42 : vec3<f32> = t_total_1 + cross(r_4.point_0.xyz - centre_0, f_0);
                f_total_1 = f_total_0 + f_0;
                t_total_0 = _S42;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            var _S40 : u32 = step_1 + u32(1);
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S40;
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
            var _S43 : u32 = _S34 + u32(3) * p_0;
            var a_1 : vec4<f32> = loads_0[_S43];
            var b_6 : vec4<f32> = loads_0[_S43 + u32(1)];
            var c_2 : vec4<f32> = loads_0[_S43 + u32(2)];
            var _S44 : f32 = loads_0[_S43 + u32(1)].w;
            var mean_0 : f32 = pulse_mean_0(loads_0[_S43].x, loads_0[_S43].y, loads_0[_S43].z, loads_0[_S43].w, _S44, _S32);
            if(mean_0 != 0.0f)
            {
                var f_1 : vec3<f32> = c_2.xyz * vec3<f32>(mean_0);
                var _S45 : vec3<f32> = t_total_1 + cross(b_6.xyz - centre_0, f_1);
                f_total_0 = f_total_0 + f_1;
                t_total_1 = _S45;
            }
            if((a_1.x + a_1.z + _S44) > _S32)
            {
                a_1[i32(0)] = a_1[i32(0)] - _S32;
                var _S46 : u32 = _S35 + u32(3) * kept_0;
                loads_0[_S46] = a_1;
                loads_0[_S46 + u32(1)] = b_6;
                loads_0[_S46 + u32(2)] = c_2;
                kept_0 = kept_0 + u32(1);
            }
            p_0 = p_0 + u32(1);
        }
        q3_0 = u32(0);
        overflow_0 = false;
        loop
        {
            if(q3_0 < _S38)
            {
            }
            else
            {
                break;
            }
            var r_5 : Record_std430_0 = manifolds_0[_S37 + q3_0];
            var _S47 : vec3<f32> = r_5.impulse_0.xyz;
            var j_1 : f32 = length(_S47);
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
                    m_1 = _S29;
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
            var _S48 : f32 = r_5.impulse_0.w;
            var _S49 : f32 = min(j_1, 2.0f * m_1 * max(_S48 - _S31, 0.0f));
            if(_S49 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var _S50 : f32 = _S30.y;
            var duration_1 : f32 = hertz_duration_0(m_1, min(r_5.other_0.z, _S30.z), 1.0f / ((1.0f - _S50 * _S50) / _S30.x + 1.0f / r_5.other_0.y), _S48);
            var peak_1 : f32 = 3.14159274101257324f * _S49 / (2.0f * duration_1);
            var dir_0 : vec3<f32> = _S47 / vec3<f32>(j_1);
            var mean_1 : f32;
            if(_S33)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S32);
            }
            if(mean_1 != 0.0f)
            {
                var f_2 : vec3<f32> = dir_0 * vec3<f32>(mean_1);
                var _S51 : vec3<f32> = t_total_1 + cross(r_5.point_0.xyz - centre_0, f_2);
                f_total_1 = f_total_0 + f_2;
                t_total_0 = _S51;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            var kept_1 : u32;
            if(duration_1 > _S32)
            {
                if(kept_0 < room_1)
                {
                    var _S52 : u32 = _S35 + u32(3) * kept_0;
                    loads_0[_S52] = vec4<f32>(- _S32, 0.0f, 0.0f, peak_1);
                    loads_0[_S52 + u32(1)] = vec4<f32>(r_5.point_0.xyz, duration_1);
                    loads_0[_S52 + u32(2)] = vec4<f32>(dir_0, 0.0f);
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
        var _S53 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(8));
    }
    var _S54 : u32 = _S23 + u32(2) * to_0;
    work_0[_S54] = base_0;
    work_0[_S54 + u32(1)] = kept_0;
    loads_0[k_0] = vec4<f32>(length(f_total_0), 0.0f, 0.0f, 0.0f);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_breaks(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var i_0 : u32 = id_1.x;
    var _S55 : bool;
    if(i_0 >= (params_0.bond_count_0))
    {
        _S55 = true;
    }
    else
    {
        _S55 = work_0[w_alive_0() + i_0] == u32(0);
    }
    if(_S55)
    {
        return;
    }
    var bond_0 : vec4<u32> = bonds_0[i_0];
    var _S56 : u32 = bond_0.x;
    if((max(loads_0[_S56].x, loads_0[bond_0.y].x)) > (bitcast<f32>((bond_0.z))))
    {
        work_0[w_alive_0() + i_0] = u32(0);
        var _S57 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S56)]), u32(1));
        var _S58 : u32 = a_counters_0();
        var _S59 : u32 = atomicAdd(&(atom_0[_S58]), u32(1));
        var _S60 : u32 = atomicAdd(&(atom_0[_S58 + u32(1)]), u32(1));
    }
    return;
}

fn w_host_breaks_0() -> u32
{
    return w_pulses_0() + u32(4) * params_0.chunk_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_host_breaks(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var i_1 : u32 = id_2.x;
    var _S61 : u32 = w_host_breaks_0();
    if(i_1 >= work_0[_S61])
    {
        return;
    }
    var b_7 : u32 = work_0[_S61 + u32(1) + i_1];
    var _S62 : bool;
    if(work_0[_S61 + u32(1) + i_1] >= (params_0.bond_count_0))
    {
        _S62 = true;
    }
    else
    {
        _S62 = work_0[w_alive_0() + b_7] == u32(0);
    }
    if(_S62)
    {
        return;
    }
    work_0[w_alive_0() + b_7] = u32(0);
    var _S63 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(bonds_0[b_7].x)]), u32(1));
    var _S64 : u32 = a_counters_0();
    var _S65 : u32 = atomicAdd(&(atom_0[_S64]), u32(1));
    var _S66 : u32 = atomicAdd(&(atom_0[_S64 + u32(1)]), u32(1));
    return;
}

fn groups_of_0( n_2 : u32,  size_0 : u32) -> u32
{
    var _S67 : u32 = (n_2 + size_0 - u32(1)) / size_0;
    return _S67;
}

fn set_args_0( at_0 : u32,  x_2 : u32)
{
    var _S68 : u32 = u32(3) * at_0;
    args_0[_S68] = x_2;
    args_0[_S68 + u32(1)] = u32(1);
    args_0[_S68 + u32(2)] = u32(1);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_gate(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var _S69 : u32 = a_counters_0();
    var _S70 : u32 = atomicLoad(&(atom_0[_S69]));
    var run_0 : bool = _S70 != u32(0);
    atomicStore(&(atom_0[_S69]), u32(0));
    atomicStore(&(atom_0[_S69 + u32(5) + params_0.parity_0]), u32(0));
    if(run_0)
    {
        var _S71 : u32 = atomicAdd(&(atom_0[_S69 + u32(4)]), u32(1));
    }
    var _S72 : u32;
    if(run_0)
    {
        var _S73 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S72 = _S73;
    }
    else
    {
        _S72 = u32(0);
    }
    set_args_0(u32(0), _S72);
    if(run_0)
    {
        var _S74 : u32 = groups_of_0(params_0.bond_count_0, u32(64));
        _S72 = _S74;
    }
    else
    {
        _S72 = u32(0);
    }
    set_args_0(u32(1), _S72);
    if(run_0)
    {
        var _S75 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S72 = _S75;
    }
    else
    {
        _S72 = u32(0);
    }
    set_args_0(u32(2), _S72);
    if(run_0)
    {
        var _S76 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S72 = _S76;
    }
    else
    {
        _S72 = u32(0);
    }
    set_args_0(u32(3), _S72);
    if(run_0)
    {
        var _S77 : u32 = groups_of_0(params_0.body_count_0, u32(1024));
        _S72 = _S77;
    }
    else
    {
        _S72 = u32(0);
    }
    set_args_0(u32(4), _S72);
    var _S78 : i32;
    if(run_0)
    {
        _S78 = i32(1);
    }
    else
    {
        _S78 = i32(0);
    }
    set_args_0(u32(5), u32(_S78));
    return;
}

fn dirty_0( body_0 : u32) -> bool
{
    var _S79 : u32 = atomicLoad(&(atom_0[a_dirty_0() + body_0]));
    return _S79 != u32(0);
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
fn fr_uf_init(@builtin(global_invocation_id) id_4 : vec3<u32>)
{
    var c_3 : u32 = id_4.x;
    var _S80 : bool;
    if(c_3 >= (params_0.chunk_count_0))
    {
        _S80 = true;
    }
    else
    {
        var _S81 : bool = dirty_0(body_of_0(c_3));
        _S80 = !_S81;
    }
    if(_S80)
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
fn fr_body_init(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var s_0 : u32 = id_5.x;
    var _S82 : bool;
    if(s_0 >= (params_0.body_count_0))
    {
        _S82 = true;
    }
    else
    {
        var _S83 : bool = dirty_0(s_0);
        _S82 = !_S83;
    }
    if(_S82)
    {
        return;
    }
    atomicStore(&(atom_0[a_best_count_0() + s_0]), u32(0));
    atomicStore(&(atom_0[a_best_root_0() + s_0]), u32(4294967295));
    return;
}

fn find_0( x_3 : u32) -> u32
{
    var _S84 : u32 = a_parent_0();
    var _S85 : u32 = atomicLoad(&(atom_0[_S84 + x_3]));
    var p_1 : u32 = _S85;
    var _S86 : u32 = x_3;
    var guard_0 : u32 = u32(0);
    loop
    {
        var _S87 : bool;
        if(guard_0 < u32(1048576))
        {
            _S87 = p_1 != _S86;
        }
        else
        {
            _S87 = false;
        }
        if(_S87)
        {
        }
        else
        {
            break;
        }
        var gp_0 : u32 = atomicLoad(&(atom_0[_S84 + p_1]));
        if(gp_0 != p_1)
        {
            var _S88 : u32 = atomicCompareExchangeWeak(&(atom_0[_S84 + _S86]), p_1, gp_0).old_value;
        }
        var _S89 : u32 = guard_0 + u32(1);
        _S86 = p_1;
        p_1 = gp_0;
        guard_0 = _S89;
    }
    return _S86;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_union(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var i_2 : u32 = id_6.x;
    var _S90 : bool;
    if(i_2 >= (params_0.bond_count_0))
    {
        _S90 = true;
    }
    else
    {
        _S90 = work_0[w_alive_0() + i_2] == u32(0);
    }
    if(_S90)
    {
        return;
    }
    var bond_1 : vec4<u32> = bonds_0[i_2];
    var _S91 : u32 = bond_1.x;
    var _S92 : bool = dirty_0(body_of_0(_S91));
    if(!_S92)
    {
        return;
    }
    var _S93 : u32 = bond_1.y;
    var a_2 : u32 = _S91;
    var b_8 : u32 = _S93;
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
        var rb_0 : u32 = find_0(b_8);
        if(ra_0 == rb_0)
        {
            return;
        }
        var _S94 : u32 = max(ra_0, rb_0);
        var _S95 : u32 = min(ra_0, rb_0);
        var was_0 : u32 = atomicCompareExchangeWeak(&(atom_0[a_parent_0() + _S94]), _S94, _S95).old_value;
        if(was_0 == _S94)
        {
            return;
        }
        var _S96 : u32 = guard_1 + u32(1);
        a_2 = _S94;
        b_8 = _S95;
        guard_1 = _S96;
    }
    var _S97 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(4));
    return;
}

fn add64_0( at_1 : u32,  value_0 : f32,  scale_0 : f32)
{
    var x_4 : f32 = round(value_0 * scale_0);
    var x_5 : f32;
    if(!((abs(x_4)) < 1.073741824e+09f))
    {
        var _S98 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
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
    var _S99 : u32;
    if(v_3 < i32(0))
    {
        _S99 = u32(4294967295);
    }
    else
    {
        _S99 = u32(0);
    }
    var hi_0 : u32 = _S99 + carry_0;
    if(hi_0 != u32(0))
    {
        var _S100 : u32 = atomicAdd(&(atom_0[at_1 + u32(1)]), hi_0);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_flatten(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var c_4 : u32 = id_7.x;
    var _S101 : bool;
    if(c_4 >= (params_0.chunk_count_0))
    {
        _S101 = true;
    }
    else
    {
        var _S102 : bool = dirty_0(body_of_0(c_4));
        _S101 = !_S102;
    }
    if(_S101)
    {
        return;
    }
    var r_6 : u32 = find_0(c_4);
    work_0[w_label_0() + c_4] = r_6;
    var _S103 : u32 = atomicAdd(&(atom_0[a_count_0() + r_6]), u32(1));
    if(((info_0[c_4] & (u32(1)))) != u32(0))
    {
        var _S104 : u32 = atomicOr(&(atom_0[a_anchor_0() + r_6]), u32(1));
    }
    var _S105 : u32 = u32(4) * c_4;
    var c0_1 : vec4<f32> = chunks_0[_S105];
    var c1_0 : vec4<f32> = chunks_0[_S105 + u32(1)];
    var c2_0 : vec4<f32> = chunks_0[_S105 + u32(2)];
    var m_2 : f32 = c0_1.w;
    var d_0 : vec3<f32> = c0_1.xyz - chunks_0[u32(4) * r_6].xyz;
    var dd_0 : f32 = dot(d_0, d_0);
    var at_2 : u32 = a_sums_0() + u32(20) * r_6;
    add64_0(at_2, m_2, params_0.scale_m_0);
    var _S106 : f32 = d_0.x;
    var _S107 : f32 = m_2 * _S106;
    add64_0(at_2 + u32(2), _S107, params_0.scale_s_0);
    var _S108 : f32 = d_0.y;
    var _S109 : f32 = m_2 * _S108;
    add64_0(at_2 + u32(4), _S109, params_0.scale_s_0);
    var _S110 : f32 = d_0.z;
    add64_0(at_2 + u32(6), m_2 * _S110, params_0.scale_s_0);
    add64_0(at_2 + u32(8), c1_0.x + m_2 * (dd_0 - _S106 * _S106), params_0.scale_j_0);
    add64_0(at_2 + u32(10), c1_0.y + m_2 * (dd_0 - _S108 * _S108), params_0.scale_j_0);
    add64_0(at_2 + u32(12), c1_0.z + m_2 * (dd_0 - _S110 * _S110), params_0.scale_j_0);
    add64_0(at_2 + u32(14), c1_0.w - _S107 * _S108, params_0.scale_j_0);
    add64_0(at_2 + u32(16), c2_0.x - _S107 * _S110, params_0.scale_j_0);
    add64_0(at_2 + u32(18), c2_0.y - _S109 * _S110, params_0.scale_j_0);
    return;
}

fn is_root_0( c_5 : u32) -> bool
{
    var _S111 : bool;
    if(c_5 < (params_0.chunk_count_0))
    {
        var _S112 : bool = dirty_0(body_of_0(c_5));
        _S111 = _S112;
    }
    else
    {
        _S111 = false;
    }
    if(_S111)
    {
        _S111 = work_0[w_label_0() + c_5] == c_5;
    }
    else
    {
        _S111 = false;
    }
    return _S111;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_count(@builtin(global_invocation_id) id_8 : vec3<u32>)
{
    var c_6 : u32 = id_8.x;
    var _S113 : bool = is_root_0(c_6);
    if(!_S113)
    {
        return;
    }
    var _S114 : u32 = atomicLoad(&(atom_0[a_count_0() + c_6]));
    var _S115 : u32 = atomicMax(&(atom_0[a_best_count_0() + body_of_0(c_6)]), _S114);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_root(@builtin(global_invocation_id) id_9 : vec3<u32>)
{
    var c_7 : u32 = id_9.x;
    var _S116 : bool = is_root_0(c_7);
    if(!_S116)
    {
        return;
    }
    var s_1 : u32 = body_of_0(c_7);
    var _S117 : u32 = atomicLoad(&(atom_0[a_count_0() + c_7]));
    var _S118 : u32 = atomicLoad(&(atom_0[a_best_count_0() + s_1]));
    if(_S117 == _S118)
    {
        var _S119 : u32 = atomicMin(&(atom_0[a_best_root_0() + s_1]), c_7);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_mark(@builtin(global_invocation_id) id_10 : vec3<u32>)
{
    var c_8 : u32 = id_10.x;
    if(c_8 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S120 : bool = is_root_0(c_8);
    var _S121 : bool;
    if(_S120)
    {
        var _S122 : u32 = atomicLoad(&(atom_0[a_best_root_0() + body_of_0(c_8)]));
        _S121 = _S122 != c_8;
    }
    else
    {
        _S121 = false;
    }
    var _S123 : u32;
    if(_S121)
    {
        _S123 = u32(1);
    }
    else
    {
        _S123 = u32(0);
    }
    work_0[w_new_0() + c_8] = _S123;
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
    var _S124 : u32 = group_0.x;
    var _S125 : u32 = u32(1024) * _S124 + u32(4) * t_1;
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
        if((_S125 + j_2) < (scan_0.n_0))
        {
            step_2 = work_0[scan_0.in_base_0 + _S125 + j_2];
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
    var _S126 : u32 = g_scan_0[t_1] - sum_2;
    j_2 = u32(0);
    add_0 = _S126;
    loop
    {
        if(j_2 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S125 + j_2) < (scan_0.n_0))
        {
            work_0[scan_0.out_base_0 + _S125 + j_2] = add_0;
        }
        var run_1 : u32 = add_0 + v_4[j_2];
        j_2 = j_2 + u32(1);
        add_0 = run_1;
    }
    if(t_1 == u32(255))
    {
        work_0[scan_0.block_base_0 + _S124] = g_scan_0[i32(255)];
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn fr_scan_top(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var t_2 : u32 = thread_1.x;
    var blocks_0 : u32 = (scan_0.n_0 + u32(1023)) / u32(1024);
    var _S127 : bool = t_2 < blocks_0;
    var v_5 : u32;
    if(_S127)
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
    if(_S127)
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
fn fr_scan_add(@builtin(global_invocation_id) id_11 : vec3<u32>)
{
    var i_3 : u32 = id_11.x;
    if(i_3 >= (scan_0.n_0))
    {
        return;
    }
    work_0[scan_0.out_base_0 + i_3] = work_0[scan_0.out_base_0 + i_3] + work_0[scan_0.block_base_0 + i_3 / u32(1024)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_freelist(@builtin(global_invocation_id) id_12 : vec3<u32>)
{
    var s_2 : u32 = id_12.x;
    var _S128 : bool;
    if(s_2 >= (params_0.body_count_0))
    {
        _S128 = true;
    }
    else
    {
        _S128 = work_0[w_free_0() + s_2] == u32(0);
    }
    if(_S128)
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
fn fr_assign(@builtin(global_invocation_id) id_13 : vec3<u32>)
{
    var c_9 : u32 = id_13.x;
    var _S129 : bool = is_root_0(c_9);
    if(!_S129)
    {
        return;
    }
    var s_3 : u32 = body_of_0(c_9);
    var _S130 : u32 = atomicLoad(&(atom_0[a_best_root_0() + s_3]));
    var slot_6 : u32;
    if(_S130 != c_9)
    {
        var k_1 : u32 = work_0[w_rank_0() + c_9];
        if(work_0[w_rank_0() + c_9] < (free_total_0()))
        {
            var slot_7 : u32 = work_0[w_free_list_0() + k_1];
            work_0[w_free_0() + work_0[w_free_list_0() + k_1]] = u32(0);
            var _S131 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(2)]), u32(1));
            slot_6 = slot_7;
        }
        else
        {
            var _S132 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(1));
            slot_6 = s_3;
        }
    }
    else
    {
        slot_6 = s_3;
    }
    work_0[w_slot_0() + c_9] = slot_6;
    return;
}

fn read64_0( at_3 : u32,  scale_1 : f32) -> f32
{
    var _S133 : u32 = atomicLoad(&(atom_0[at_3 + u32(1)]));
    var hi_1 : i32 = (bitcast<i32>((_S133)));
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
    var q_11 : Quat_0;
    if(m22_0 <= 0.0f)
    {
        var dif10_0 : f32 = m11_0 - m00_0;
        var omm22_0 : f32 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            var f_3 : f32 = omm22_0 - dif10_0;
            var inv_0 : f32 = 0.5f / sqrt(f_3);
            q_11.x_1 = f_3 * inv_0;
            q_11.y_0 = (m01_0 + m10_0) * inv_0;
            q_11.z_0 = (m02_0 + m20_0) * inv_0;
            q_11.w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            var f_4 : f32 = omm22_0 + dif10_0;
            var inv_1 : f32 = 0.5f / sqrt(f_4);
            q_11.x_1 = (m01_0 + m10_0) * inv_1;
            q_11.y_0 = f_4 * inv_1;
            q_11.z_0 = (m12_0 + m21_0) * inv_1;
            q_11.w_0 = (m20_0 - m02_0) * inv_1;
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
            q_11.x_1 = (m02_0 + m20_0) * inv_2;
            q_11.y_0 = (m12_0 + m21_0) * inv_2;
            q_11.z_0 = f_5 * inv_2;
            q_11.w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            var f_6 : f32 = opm22_0 + sum10_0;
            var inv_3 : f32 = 0.5f / sqrt(f_6);
            q_11.x_1 = (m12_0 - m21_0) * inv_3;
            q_11.y_0 = (m20_0 - m02_0) * inv_3;
            q_11.z_0 = (m01_0 - m10_0) * inv_3;
            q_11.w_0 = f_6 * inv_3;
        }
    }
    return q_11;
}

fn principal_axes_0( r0_0 : vec3<f32>,  r1_0 : vec3<f32>,  r2_0 : vec3<f32>,  rot_0 : ptr<function, Quat_0>,  moments_0 : ptr<function, vec3<f32>>)
{
    var _S134 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S135 : array<f32, i32(3)> = array<f32, i32(3)>( r1_0.x, r1_0.y, r1_0.z );
    var _S136 : array<f32, i32(3)> = array<f32, i32(3)>( r2_0.x, r2_0.y, r2_0.z );
    var a_3 : array<array<f32, i32(3)>, i32(3)>;
    a_3[i32(0)] = _S134;
    a_3[i32(1)] = _S135;
    a_3[i32(2)] = _S136;
    var _S137 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S138 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S139 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_6 : array<array<f32, i32(3)>, i32(3)>;
    v_6[i32(0)] = _S137;
    v_6[i32(1)] = _S138;
    v_6[i32(2)] = _S139;
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
        var q_12 : u32;
        if((abs(a_3[i32(0)][i32(2)])) > (abs(a_3[u32(0)][u32(1)])))
        {
            q_12 = u32(2);
        }
        else
        {
            q_12 = u32(1);
        }
        var p_2 : u32;
        var q_13 : u32;
        if((abs(a_3[i32(1)][i32(2)])) > (abs(a_3[u32(0)][q_12])))
        {
            p_2 = u32(1);
            q_13 = u32(2);
        }
        else
        {
            p_2 = u32(0);
            q_13 = q_12;
        }
        if((abs(a_3[p_2][q_13])) <= (9.99999982451670044e-15f * max(max(abs(a_3[i32(0)][i32(0)]), abs(a_3[i32(1)][i32(1)])), max(abs(a_3[i32(2)][i32(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        var theta_0 : f32 = 0.5f * atan2(2.0f * a_3[p_2][q_13], a_3[q_13][q_13] - a_3[p_2][p_2]);
        var s_4 : f32 = sin(theta_0);
        var c_10 : f32 = cos(theta_0);
        var app_0 : f32 = a_3[p_2][p_2];
        var aqq_0 : f32 = a_3[q_13][q_13];
        var apq_0 : f32 = a_3[p_2][q_13];
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
            var _S140 : bool;
            if(k_2 != p_2)
            {
                _S140 = k_2 != q_13;
            }
            else
            {
                _S140 = false;
            }
            if(_S140)
            {
                var kp_0 : f32 = a_3[k_2][p_2];
                var kq_0 : f32 = a_3[k_2][q_13];
                var _S141 : f32 = c_10 * a_3[k_2][p_2] - s_4 * a_3[k_2][q_13];
                a_3[k_2][p_2] = _S141;
                a_3[p_2][k_2] = _S141;
                var _S142 : f32 = s_4 * kp_0 + c_10 * kq_0;
                a_3[k_2][q_13] = _S142;
                a_3[q_13][k_2] = _S142;
            }
            k_2 = k_2 + u32(1);
        }
        var _S143 : f32 = c_10 * c_10;
        var _S144 : f32 = 2.0f * s_4 * c_10 * apq_0;
        var _S145 : f32 = s_4 * s_4;
        a_3[p_2][p_2] = _S143 * app_0 - _S144 + _S145 * aqq_0;
        a_3[q_13][q_13] = _S145 * app_0 + _S144 + _S143 * aqq_0;
        a_3[p_2][q_13] = 0.0f;
        a_3[q_13][p_2] = 0.0f;
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
            var kq_1 : f32 = v_6[row_0][q_13];
            v_6[row_0][p_2] = c_10 * v_6[row_0][p_2] - s_4 * v_6[row_0][q_13];
            v_6[row_0][q_13] = s_4 * kp_1 + c_10 * kq_1;
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

fn rot_to_rubble_0( q_14 : Quat_0) -> Quat_0
{
    var axes_1 : Quat_0;
    axes_1.w_0 = 0.70710676908493042f;
    axes_1.x_1 = -0.70710676908493042f;
    axes_1.y_0 = 0.0f;
    axes_1.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_1, q_14));
}

fn stage_0( c_11 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * c_11;
}

fn quat_vec_0( q_15 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_15.x_1, q_15.y_0, q_15.z_0, q_15.w_0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_piece(@builtin(global_invocation_id) id_14 : vec3<u32>)
{
    var c_12 : u32 = id_14.x;
    var _S146 : bool = is_root_0(c_12);
    if(!_S146)
    {
        return;
    }
    var parent_0 : u32 = body_of_0(c_12);
    var pb_0 : u32 = u32(7) * parent_0;
    var at_4 : u32 = a_sums_0() + u32(20) * c_12;
    var mass_0 : f32 = read64_0(at_4, params_0.scale_m_0);
    var _S147 : f32 = read64_0(at_4 + u32(2), params_0.scale_s_0);
    var _S148 : f32 = read64_0(at_4 + u32(4), params_0.scale_s_0);
    var _S149 : f32 = read64_0(at_4 + u32(6), params_0.scale_s_0);
    var d_1 : vec3<f32> = vec3<f32>(_S147, _S148, _S149) / vec3<f32>(max(mass_0, 1.00000000317107685e-30f));
    var jxx_0 : f32 = read64_0(at_4 + u32(8), params_0.scale_j_0);
    var jyy_0 : f32 = read64_0(at_4 + u32(10), params_0.scale_j_0);
    var jzz_0 : f32 = read64_0(at_4 + u32(12), params_0.scale_j_0);
    var jxy_0 : f32 = read64_0(at_4 + u32(14), params_0.scale_j_0);
    var jxz_0 : f32 = read64_0(at_4 + u32(16), params_0.scale_j_0);
    var jyz_0 : f32 = read64_0(at_4 + u32(18), params_0.scale_j_0);
    var dd_1 : f32 = dot(d_1, d_1);
    var _S150 : f32 = d_1.x;
    var _S151 : f32 = mass_0 * _S150;
    var _S152 : f32 = d_1.y;
    var _S153 : f32 = jxy_0 + _S151 * _S152;
    var _S154 : f32 = d_1.z;
    var _S155 : f32 = jxz_0 + _S151 * _S154;
    var _S156 : f32 = jyz_0 + mass_0 * _S152 * _S154;
    var com_5 : vec3<f32> = chunks_0[u32(4) * c_12].xyz + d_1;
    var principal_0 : Quat_0;
    var moments_1 : vec3<f32>;
    principal_axes_0(vec3<f32>(jxx_0 - mass_0 * (dd_1 - _S150 * _S150), _S153, _S155), vec3<f32>(_S153, jyy_0 - mass_0 * (dd_1 - _S152 * _S152), _S156), vec3<f32>(_S155, _S156, jzz_0 - mass_0 * (dd_1 - _S154 * _S154)), &(principal_0), &(moments_1));
    var _S157 : u32 = atomicLoad(&(atom_0[a_anchor_0() + c_12]));
    var anchored_0 : bool = _S157 != u32(0);
    var q_16 : Quat_0;
    var origin_6 : vec3<f32>;
    structure_pose_4(parent_0, pb_0, &(q_16), &(origin_6));
    var p_parent_0 : vec3<f32> = bodies_0[pb_0].xyz;
    var v_parent_0 : vec3<f32> = bodies_0[pb_0 + u32(3)].xyz;
    var w_parent_0 : vec3<f32> = bodies_0[pb_0 + u32(4)].xyz;
    var pos_0 : vec3<f32> = to_rubble_0(origin_6 + rotate_0(q_16, com_5));
    var orientation_0 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(q_16, principal_0)));
    var _S158 : bool;
    if(anchored_0)
    {
        _S158 = true;
    }
    else
    {
        _S158 = !(mass_0 > 0.0f);
    }
    var inv_mass_1 : f32;
    if(_S158)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / mass_0;
    }
    var _S159 : vec3<f32> = vec3<f32>(0.0f);
    var inv_inertia_0 : vec3<f32>;
    if(!anchored_0)
    {
        var _S160 : f32;
        if((moments_1.x) > 0.0f)
        {
            _S160 = 1.0f / moments_1.x;
        }
        else
        {
            _S160 = 0.0f;
        }
        var _S161 : f32;
        if((moments_1.y) > 0.0f)
        {
            _S161 = 1.0f / moments_1.y;
        }
        else
        {
            _S161 = 0.0f;
        }
        var _S162 : f32;
        if((moments_1.z) > 0.0f)
        {
            _S162 = 1.0f / moments_1.z;
        }
        else
        {
            _S162 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S160, _S161, _S162);
    }
    else
    {
        inv_inertia_0 = _S159;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_0)
    {
        velocity_0 = _S159;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    var spin_0 : vec3<f32>;
    if(anchored_0)
    {
        spin_0 = _S159;
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
    frames_0[st_0 + u32(7)] = vec4<f32>(com_5, frames_0[u32(2) * parent_0].w);
    frames_0[st_0 + u32(8)] = quat_vec_0(principal_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_commit(@builtin(global_invocation_id) id_15 : vec3<u32>)
{
    var c_13 : u32 = id_15.x;
    var _S163 : bool = is_root_0(c_13);
    if(!_S163)
    {
        return;
    }
    var slot_8 : u32 = work_0[w_slot_0() + c_13];
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
            bodies_0[u32(7) * (slot_8 + region_0 * params_0.max_bodies_0) + w_3] = frames_0[st_1 + w_3];
            w_3 = w_3 + u32(1);
        }
        region_0 = region_0 + u32(1);
    }
    var _S164 : u32 = u32(2) * slot_8;
    frames_0[_S164] = frames_0[st_1 + u32(7)];
    frames_0[_S164 + u32(1)] = frames_0[st_1 + u32(8)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_reparent(@builtin(global_invocation_id) id_16 : vec3<u32>)
{
    var c_14 : u32 = id_16.x;
    var _S165 : bool;
    if(c_14 >= (params_0.chunk_count_0))
    {
        _S165 = true;
    }
    else
    {
        var _S166 : bool = dirty_0(body_of_0(c_14));
        _S165 = !_S166;
    }
    if(_S165)
    {
        return;
    }
    var r_7 : u32 = work_0[w_label_0() + c_14];
    var slot_9 : u32 = work_0[w_slot_0() + work_0[w_label_0() + c_14]];
    work_0[w_body_0() + c_14] = work_0[w_slot_0() + work_0[w_label_0() + c_14]];
    var _S167 : u32 = u32(2) * slot_9;
    var _S168 : vec3<f32> = frames_0[_S167].xyz;
    var _S169 : Quat_0 = conj_0(quat_of_0(frames_0[_S167 + u32(1)]));
    var inv_mass_2 : f32 = bodies_0[u32(7) * slot_9].w;
    var _S170 : f32;
    if(inv_mass_2 > 0.0f)
    {
        _S170 = 1.0f / inv_mass_2;
    }
    else
    {
        _S170 = float_bits_0(u32(2139095040));
    }
    var c2_2 : vec4<f32> = chunks_0[u32(4) * c_14 + u32(2)];
    var first_1 : u32 = (bitcast<u32>((c2_2.z)));
    var _S171 : u32 = (bitcast<u32>((c2_2.w)));
    var _S172 : vec4<f32> = chunks_0[u32(4) * r_7 + u32(3)];
    var p_3 : u32 = first_1;
    loop
    {
        if(p_3 < (first_1 + _S171))
        {
        }
        else
        {
            break;
        }
        var _S173 : u32 = u32(2) * p_3;
        var o_1 : vec4<f32> = parts_0[_S173];
        var col_0 : u32 = (bitcast<u32>((o_1.w)));
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S173 + u32(1)]);
        var _S174 : u32 = u32(4) * col_0;
        colliders_0[_S174] = vec4<f32>((bitcast<f32>((slot_9))), colliders_0[_S174].y, colliders_0[_S174].z, colliders_0[_S174].w);
        var _S175 : u32 = _S174 + u32(1);
        colliders_0[_S175] = vec4<f32>(rotate_0(_S169, o_1.xyz - _S168), colliders_0[_S175].w);
        colliders_0[_S174 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S169, rot_1)));
        var _S176 : u32 = u32(2) * col_0;
        collider_info_0[_S176] = vec4<u32>(collider_info_0[_S176].x, slot_9, collider_info_0[_S176].z, collider_info_0[_S176].w);
        collider_info_0[_S176 + u32(1)] = vec4<u32>((bitcast<u32>((_S170))), (bitcast<u32>((_S172.x))), (bitcast<u32>((_S172.w))), u32(0));
        p_3 = p_3 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_body_clear(@builtin(global_invocation_id) id_17 : vec3<u32>)
{
    var s_5 : u32 = id_17.x;
    if(s_5 >= (params_0.body_count_0))
    {
        return;
    }
    atomicStore(&(atom_0[a_dirty_0() + s_5]), u32(0));
    return;
}

fn slot_color_0( slot_10 : u32) -> vec3<f32>
{
    var h_0 : f32 = fract(f32(slot_10) * 0.6180340051651001f) * 6.0f;
    var f_7 : f32 = fract(h_0);
    var q_17 : f32 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_7);
    var t_3 : f32 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_7));
    var k_3 : u32 = u32(h_0);
    var _S177 : vec3<f32>;
    if(k_3 == u32(0))
    {
        _S177 = vec3<f32>(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_3 == u32(1))
        {
            _S177 = vec3<f32>(q_17, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_3 == u32(2))
            {
                _S177 = vec3<f32>(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_3 == u32(3))
                {
                    _S177 = vec3<f32>(0.34959998726844788f, q_17, 0.92000001668930054f);
                }
                else
                {
                    if(k_3 == u32(4))
                    {
                        _S177 = vec3<f32>(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S177 = vec3<f32>(0.92000001668930054f, 0.34959998726844788f, q_17);
                    }
                }
            }
        }
    }
    return _S177;
}

fn write_object_0( o_2 : u32,  q_18 : Quat_0,  t_4 : vec3<f32>,  color_0 : vec4<f32>)
{
    var x_7 : vec3<f32> = rotate_0(q_18, vec3<f32>(1.0f, 0.0f, 0.0f));
    var y_2 : vec3<f32> = rotate_0(q_18, vec3<f32>(0.0f, 1.0f, 0.0f));
    var z_3 : vec3<f32> = rotate_0(q_18, vec3<f32>(0.0f, 0.0f, 1.0f));
    var _S178 : u32 = u32(4) * o_2;
    objects_0[_S178] = vec4<f32>(x_7.x, y_2.x, z_3.x, t_4.x);
    objects_0[_S178 + u32(1)] = vec4<f32>(x_7.y, y_2.y, z_3.y, t_4.y);
    objects_0[_S178 + u32(2)] = vec4<f32>(x_7.z, y_2.z, z_3.z, t_4.z);
    objects_0[_S178 + u32(3)] = color_0;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_chunks(@builtin(global_invocation_id) id_18 : vec3<u32>)
{
    var k_4 : u32 = id_18.x;
    if(k_4 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_11 : u32 = body_of_0(k_4);
    var b_9 : u32 = u32(7) * slot_11;
    var q_19 : Quat_0;
    var origin_7 : vec3<f32>;
    structure_pose_4(slot_11, b_9, &(q_19), &(origin_7));
    var _S179 : u32 = u32(3) * k_4;
    var v0_0 : vec4<f32> = view_chunks_0[_S179];
    var base_2 : vec3<f32> = view_chunks_0[_S179 + u32(2)].xyz;
    var fixed_0 : bool = (bodies_0[b_9].w) == 0.0f;
    var color_1 : vec3<f32>;
    if((view_0.mode_0) == u32(1))
    {
        var _S180 : f32 = max(length(bodies_0[b_9 + u32(3)].xyz), length(bodies_0[b_9 + u32(4)].xyz));
        if(fixed_0)
        {
            color_1 = vec3<f32>(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S180 > (view_0.rest_speed_0))
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
            color_1 = slot_color_0(slot_11);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = vec3<f32>(0.44999998807907104f) * base_2 + vec3<f32>(0.55000001192092896f) * slot_color_0(slot_11);
            }
        }
    }
    write_object_0((bitcast<u32>((v0_0.w))), normalized_0(quat_mul_0(q_19, quat_of_0(view_chunks_0[_S179 + u32(1)]))), origin_7 + rotate_0(q_19, v0_0.xyz), vec4<f32>(color_1, 1.0f));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_shots(@builtin(global_invocation_id) id_19 : vec3<u32>)
{
    var i_4 : u32 = id_19.x;
    if(i_4 >= (view_0.shot_count_1))
    {
        return;
    }
    var b_10 : u32 = u32(7) * shot_bodies_0[i_4];
    var o_3 : u32 = view_0.impactor_base_0 + i_4;
    if((bodies_0[b_10].w) == 0.0f)
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
    write_object_0(o_3, rot_from_rubble_0(quat_of_0(bodies_0[b_10 + u32(1)])), from_rubble_0(bodies_0[b_10].xyz), vec4<f32>(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f));
    return;
}

fn chunk_world_0( k_5 : u32) -> vec3<f32>
{
    var slot_12 : u32 = body_of_0(k_5);
    var q_20 : Quat_0;
    var origin_8 : vec3<f32>;
    structure_pose_1(slot_12, u32(7) * slot_12, &(q_20), &(origin_8));
    return origin_8 + rotate_0(q_20, view_chunks_0[u32(3) * k_5].xyz);
}

fn write_line_vertex_0( v_8 : u32,  p_4 : vec3<f32>)
{
    var _S181 : u32 = u32(7) * v_8;
    lines_0[_S181] = p_4.x;
    lines_0[_S181 + u32(1)] = p_4.y;
    lines_0[_S181 + u32(2)] = p_4.z;
    lines_0[_S181 + u32(3)] = 1.0f;
    lines_0[_S181 + u32(4)] = 0.86000001430511475f;
    lines_0[_S181 + u32(5)] = 0.25f;
    lines_0[_S181 + u32(6)] = 1.0f;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_bonds(@builtin(global_invocation_id) id_20 : vec3<u32>)
{
    var i_5 : u32 = id_20.x;
    if(i_5 >= (params_0.bond_count_0))
    {
        return;
    }
    var bond_2 : vec4<u32> = bonds_0[i_5];
    var a_4 : vec3<f32> = chunk_world_0(bond_2.x);
    var b_11 : vec3<f32>;
    if(work_0[w_alive_0() + i_5] != u32(0))
    {
        b_11 = chunk_world_0(bond_2.y);
    }
    else
    {
        b_11 = a_4;
    }
    var _S182 : u32 = u32(2) * i_5;
    write_line_vertex_0(_S182, a_4);
    write_line_vertex_0(_S182 + u32(1), b_11);
    return;
}

fn a_strike_0() -> u32
{
    return a_counters_0() + u32(16);
}

fn strike_at_0( b_12 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * params_0.chunk_count_0 + u32(3) * b_12;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_reset(@builtin(global_invocation_id) id_21 : vec3<u32>)
{
    var b_13 : u32 = id_21.x;
    if(b_13 >= (params_0.body_count_0))
    {
        return;
    }
    var at_5 : u32 = a_strike_0() + u32(9) * b_13;
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
    frames_0[strike_at_0(b_13)][i32(3)] = 0.0f;
    return;
}

fn chunk_box_world_0( k_6 : u32,  q_21 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_13 : u32 = body_of_0(k_6);
    var origin_9 : vec3<f32>;
    structure_pose_2(slot_13, u32(7) * slot_13, &((*q_21)), &(origin_9));
    return origin_9 + rotate_0((*q_21), geometry_0[u32(3) * k_6].xyz);
}

fn chunk_box_world_1( k_7 : u32,  q_22 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_14 : u32 = body_of_0(k_7);
    var origin_10 : vec3<f32>;
    structure_pose_4(slot_14, u32(7) * slot_14, &((*q_22)), &(origin_10));
    return origin_10 + rotate_0((*q_22), geometry_0[u32(3) * k_7].xyz);
}

fn ordered_0( f_8 : f32) -> u32
{
    var u_0 : u32 = (bitcast<u32>((f_8)));
    var _S183 : u32;
    if(((u_0 & (u32(2147483648)))) != u32(0))
    {
        _S183 = ~u_0;
    }
    else
    {
        _S183 = (u_0 | (u32(2147483648)));
    }
    return _S183;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_bounds(@builtin(global_invocation_id) id_22 : vec3<u32>)
{
    var k_8 : u32 = id_22.x;
    var _S184 : bool;
    if(k_8 >= (params_0.chunk_count_0))
    {
        _S184 = true;
    }
    else
    {
        _S184 = (params_0.shot_count_0) == u32(0);
    }
    if(_S184)
    {
        return;
    }
    var q_23 : Quat_0;
    var _S185 : vec3<f32> = chunk_box_world_1(k_8, &(q_23));
    var _S186 : f32 = geometry_0[u32(3) * k_8].w;
    var _S187 : u32 = a_strike_0() + u32(9) * body_of_0(k_8);
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
        var _S188 : u32 = ax_0;
        var _S189 : u32 = atomicMin(&(atom_0[_S187 + ax_0]), ordered_0(_S185[ax_0] - _S186));
        var _S190 : u32 = atomicMax(&(atom_0[_S187 + u32(3) + ax_0]), ordered_0(_S185[_S188] + _S186));
        ax_0 = ax_0 + u32(1);
    }
    return;
}

fn unordered_0( u_1 : u32) -> f32
{
    var _S191 : u32;
    if(((u_1 & (u32(2147483648)))) != u32(0))
    {
        _S191 = (u_1 & (u32(2147483647)));
    }
    else
    {
        _S191 = ~u_1;
    }
    return (bitcast<f32>((_S191)));
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_find(@builtin(global_invocation_id) id_23 : vec3<u32>)
{
    var b_14 : u32 = id_23.x;
    var _S192 : bool;
    if(b_14 >= (params_0.body_count_0))
    {
        _S192 = true;
    }
    else
    {
        _S192 = (params_0.shot_count_0) == u32(0);
    }
    if(_S192)
    {
        return;
    }
    var at_6 : u32 = a_strike_0() + u32(9) * b_14;
    var _S193 : u32 = atomicLoad(&(atom_0[at_6]));
    if(_S193 == u32(4294967295))
    {
        return;
    }
    var _S194 : u32 = atomicLoad(&(atom_0[at_6]));
    var _S195 : f32 = unordered_0(_S194);
    var _S196 : u32 = atomicLoad(&(atom_0[at_6 + u32(1)]));
    var _S197 : f32 = unordered_0(_S196);
    var _S198 : u32 = atomicLoad(&(atom_0[at_6 + u32(2)]));
    var _S199 : vec3<f32> = vec3<f32>(_S195, _S197, unordered_0(_S198));
    var _S200 : u32 = atomicLoad(&(atom_0[at_6 + u32(3)]));
    var _S201 : f32 = unordered_0(_S200);
    var _S202 : u32 = atomicLoad(&(atom_0[at_6 + u32(4)]));
    var _S203 : f32 = unordered_0(_S202);
    var _S204 : u32 = atomicLoad(&(atom_0[at_6 + u32(5)]));
    var _S205 : vec3<f32> = vec3<f32>(_S201, _S203, unordered_0(_S204));
    var _S206 : vec3<f32> = from_rubble_0(bodies_0[u32(7) * b_14 + u32(3)].xyz);
    var _S207 : f32 = float_bits_0(u32(2139095040));
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
        var rel_0 : vec3<f32> = from_rubble_0(bodies_0[sb_0 + u32(3)].xyz) - _S206;
        if(!((length(rel_0)) > (params_0.strike_speed_0)))
        {
            k_9 = k_9 + u32(1);
            continue;
        }
        var _S208 : vec3<f32> = vec3<f32>(info_1.y);
        var _S209 : vec3<f32> = _S199 - _S208;
        var _S210 : vec3<f32> = _S205 + _S208;
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
            var _S211 : u32 = ax_1;
            if((abs(rel_0[ax_1])) < 9.999999960041972e-13f)
            {
                var _S212 : u32 = ax_1;
                if((c_15[ax_1]) < (_S209[ax_1]))
                {
                    _S192 = true;
                }
                else
                {
                    _S192 = (c_15[_S212]) > (_S210[ax_1]);
                }
                var t0_1 : f32;
                if(_S192)
                {
                    t0_1 = _S207;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_1 = ax_1 + u32(1);
                continue;
            }
            var ta_0 : f32 = (_S209[ax_1] - c_15[ax_1]) / rel_0[_S211];
            var tb_0 : f32 = (_S210[ax_1] - c_15[ax_1]) / rel_0[_S211];
            var _S213 : f32 = min(t1_0, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_0 = _S213;
            ax_1 = ax_1 + u32(1);
        }
        if(t0_0 <= t1_0)
        {
            var st_2 : u32 = strike_at_0(b_14);
            frames_0[st_2] = vec4<f32>(c_15, f32(k_9 + u32(1)));
            frames_0[st_2 + u32(1)] = vec4<f32>(rel_0, t0_0);
            return;
        }
        k_9 = k_9 + u32(1);
    }
    return;
}

fn struck_0( b_15 : u32,  shot_0 : ptr<function, u32>) -> bool
{
    var k_10 : u32 = u32(frames_0[strike_at_0(b_15)].w);
    (*shot_0) = k_10 - u32(1);
    return k_10 != u32(0);
}

fn a_strike_dist_0() -> u32
{
    return a_strike_0() + u32(9) * params_0.body_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_entry(@builtin(global_invocation_id) id_24 : vec3<u32>)
{
    var k_11 : u32 = id_24.x;
    var _S214 : bool;
    if(k_11 >= (params_0.chunk_count_0))
    {
        _S214 = true;
    }
    else
    {
        _S214 = (params_0.shot_count_0) == u32(0);
    }
    if(_S214)
    {
        return;
    }
    var b_16 : u32 = body_of_0(k_11);
    var shot_1 : u32;
    var _S215 : bool = struck_0(b_16, &(shot_1));
    if(!_S215)
    {
        return;
    }
    var _S216 : u32 = strike_at_0(b_16);
    var s1_0 : vec4<f32> = frames_0[_S216 + u32(1)];
    var from_1 : vec3<f32> = frames_0[_S216].xyz;
    var rel_1 : vec3<f32> = frames_0[_S216 + u32(1)].xyz;
    var d_2 : vec3<f32> = rel_1 / vec3<f32>(max(length(rel_1), 9.999999960041972e-13f));
    var _S217 : f32 = shots_0[u32(2) * shot_1].y;
    var q_24 : Quat_0;
    var cc_0 : vec3<f32> = chunk_box_world_1(k_11, &(q_24));
    var _S218 : u32 = u32(3) * k_11;
    var _S219 : Quat_0 = normalized_0(quat_mul_0(q_24, quat_of_0(geometry_0[_S218 + u32(2)])));
    var _S220 : vec3<f32> = geometry_0[_S218 + u32(1)].xyz;
    var _S221 : f32 = float_bits_0(u32(2139095040));
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
        var axis_0 : vec3<f32> = rotate_0(_S219, e_1);
        var o_4 : f32 = dot(from_1 - cc_0, axis_0);
        var dv_0 : f32 = dot(rel_1, axis_0);
        var ext_0 : f32 = _S220[ax_2] + _S217;
        if((abs(dv_0)) < 9.999999960041972e-13f)
        {
            var t_in_1 : f32;
            if((abs(o_4)) > ext_0)
            {
                t_in_1 = _S221;
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
        var _S222 : f32 = min(t_out_0, max(ta_1, tb_1));
        t_in_0 = max(t_in_0, min(ta_1, tb_1));
        t_out_0 = _S222;
        ax_2 = ax_2 + u32(1);
    }
    var at_7 : u32 = a_strike_0() + u32(9) * b_16;
    if(t_in_0 <= t_out_0)
    {
        var _S223 : u32 = atomicMin(&(atom_0[at_7 + u32(6)]), (bitcast<u32>((t_in_0))));
    }
    var r_8 : vec3<f32> = cc_0 - (from_1 + rel_1 * vec3<f32>(s1_0.w));
    var along_0 : f32 = dot(r_8, d_2);
    var _S224 : u32 = (bitcast<u32>((length(r_8 - d_2 * vec3<f32>(along_0)) + abs(along_0) * 0.00100000004749745f)));
    atomicStore(&(atom_0[a_strike_dist_0() + k_11]), _S224);
    var _S225 : u32 = atomicMin(&(atom_0[at_7 + u32(7)]), _S224);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_nearest(@builtin(global_invocation_id) id_25 : vec3<u32>)
{
    var k_12 : u32 = id_25.x;
    var _S226 : bool;
    if(k_12 >= (params_0.chunk_count_0))
    {
        _S226 = true;
    }
    else
    {
        _S226 = (params_0.shot_count_0) == u32(0);
    }
    if(_S226)
    {
        return;
    }
    var b_17 : u32 = body_of_0(k_12);
    var shot_2 : u32;
    if(!struck_0(b_17, &(shot_2)))
    {
        return;
    }
    var at_8 : u32 = a_strike_0() + u32(9) * b_17;
    var _S227 : u32 = atomicLoad(&(atom_0[a_strike_dist_0() + k_12]));
    var _S228 : u32 = atomicLoad(&(atom_0[at_8 + u32(7)]));
    if(_S227 == _S228)
    {
        var _S229 : u32 = atomicMin(&(atom_0[at_8 + u32(8)]), k_12);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_force(@builtin(global_invocation_id) id_26 : vec3<u32>)
{
    var b_18 : u32 = id_26.x;
    var _S230 : bool;
    if(b_18 >= (params_0.body_count_0))
    {
        _S230 = true;
    }
    else
    {
        _S230 = (params_0.shot_count_0) == u32(0);
    }
    if(_S230)
    {
        return;
    }
    var shot_3 : u32;
    var _S231 : bool = struck_0(b_18, &(shot_3));
    if(!_S231)
    {
        return;
    }
    var at_9 : u32 = a_strike_0() + u32(9) * b_18;
    var _S232 : u32 = atomicLoad(&(atom_0[at_9 + u32(6)]));
    var t_in_2 : f32 = (bitcast<f32>((_S232)));
    var entry_0 : u32 = atomicLoad(&(atom_0[at_9 + u32(8)]));
    var st_3 : u32 = strike_at_0(b_18);
    if(!(t_in_2 <= (params_0.dt_0)))
    {
        _S230 = true;
    }
    else
    {
        _S230 = entry_0 == u32(4294967295);
    }
    if(_S230)
    {
        frames_0[st_3][i32(3)] = 0.0f;
        return;
    }
    var rel_2 : vec3<f32> = frames_0[st_3 + u32(1)].xyz;
    var speed_0 : f32 = length(rel_2);
    var info_2 : vec4<f32> = shots_0[u32(2) * shot_3];
    var _S233 : f32 = max(info_2.z, 0.00100000004749745f);
    var m_imp_0 : f32 = info_2.w;
    var e_imp_0 : f32 = shots_0[u32(2) * shot_3 + u32(1)].x;
    var contact_0 : vec3<f32> = frames_0[st_3].xyz + rel_2 * vec3<f32>(t_in_2) + rel_2 / vec3<f32>(max(speed_0, 9.999999960041972e-13f)) * vec3<f32>(info_2.y);
    var inv_mass_3 : f32 = bodies_0[u32(7) * b_18].w;
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
    frames_0[st_3 + u32(2)] = vec4<f32>(contact_0, m_eff_0 * speed_0 / (2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (_S233 * modulus_0 * modulus_0 * max(speed_0, 0.00100000004749745f)), 0.20000000298023224f)));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_breaks(@builtin(global_invocation_id) id_27 : vec3<u32>)
{
    var i_6 : u32 = id_27.x;
    var _S234 : bool;
    if(i_6 >= (params_0.bond_count_0))
    {
        _S234 = true;
    }
    else
    {
        _S234 = (params_0.shot_count_0) == u32(0);
    }
    if(_S234)
    {
        _S234 = true;
    }
    else
    {
        _S234 = work_0[w_alive_0() + i_6] == u32(0);
    }
    if(_S234)
    {
        return;
    }
    var bond_3 : vec4<u32> = bonds_0[i_6];
    var _S235 : u32 = bond_3.x;
    var b_19 : u32 = body_of_0(_S235);
    var shot_4 : u32;
    if(!struck_0(b_19, &(shot_4)))
    {
        return;
    }
    var q_25 : Quat_0;
    var pa_0 : vec3<f32> = chunk_box_world_1(_S235, &(q_25));
    var pb_1 : vec3<f32> = chunk_box_world_1(bond_3.y, &(q_25));
    var dist_0 : f32 = length(vec3<f32>(0.5f) * (pa_0 + pb_1) - frames_0[strike_at_0(b_19) + u32(2)].xyz);
    if((frames_0[strike_at_0(b_19) + u32(2)].w * min(1.0f, (bitcast<f32>((bond_3.w))) / (6.28318548202514648f * max(dist_0 * dist_0, 9.999999960041972e-13f)))) > (bitcast<f32>((bond_3.z))))
    {
        work_0[w_alive_0() + i_6] = u32(0);
        var _S236 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_19]), u32(1));
        var _S237 : u32 = a_counters_0();
        var _S238 : u32 = atomicAdd(&(atom_0[_S237]), u32(1));
        var _S239 : u32 = atomicAdd(&(atom_0[_S237 + u32(1)]), u32(1));
    }
    return;
}

fn crushable_0( k_13 : u32) -> bool
{
    return (((bitcast<u32>((crush_tab_0[u32(2) * k_13 + u32(1)].z))) & (u32(1)))) != u32(0);
}

fn crush_flags_0( k_14 : u32) -> u32
{
    return (bitcast<u32>((crush_rw_0[k_14].w)));
}

fn crush_record_0( k_15 : u32,  step_4 : vec4<f32>,  cause_0 : u32)
{
    var st_4 : vec4<f32> = crush_rw_0[k_15];
    var flags_1 : u32 = (bitcast<u32>((crush_rw_0[k_15].w)));
    var flags_2 : u32;
    if((step_4.w) != 0.0f)
    {
        flags_2 = (((((flags_1 | (u32(1)))) | (u32(2)))) | (((cause_0 << (u32(4))))));
    }
    else
    {
        flags_2 = flags_1;
    }
    crush_rw_0[k_15] = vec4<f32>(step_4.x, max(st_4.y, step_4.y), max(st_4.z, step_4.z), (bitcast<f32>((flags_2))));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_blast(@builtin(global_invocation_id) id_28 : vec3<u32>)
{
    var k_16 : u32 = id_28.x;
    var _S240 : bool;
    if(k_16 >= (params_0.chunk_count_0))
    {
        _S240 = true;
    }
    else
    {
        _S240 = (crush_0.enabled_0) == u32(0);
    }
    if(_S240)
    {
        return;
    }
    var w_6 : vec4<f32> = crush_rw_0[crush_0.r_blast_0 + k_16];
    var _S241 : f32 = crush_rw_0[crush_0.r_blast_0 + k_16].x;
    if(!(_S241 > 0.0f))
    {
        _S240 = !((w_6.y) > 0.0f);
    }
    else
    {
        _S240 = false;
    }
    if(_S240)
    {
        _S240 = !((w_6.z) > 0.0f);
    }
    else
    {
        _S240 = false;
    }
    if(_S240)
    {
        return;
    }
    crush_rw_0[crush_0.r_blast_0 + k_16] = vec4<f32>(0.0f);
    if(!crushable_0(k_16))
    {
        _S240 = true;
    }
    else
    {
        _S240 = (((crush_flags_0(k_16)) & (u32(1)))) != u32(0);
    }
    if(_S240)
    {
        return;
    }
    var d_3 : f32 = crush_rw_0[k_16].x + _S241;
    var step_5 : vec4<f32>;
    if(d_3 >= 1.0f)
    {
        step_5 = vec4<f32>(1.0f, w_6.y, w_6.z, 1.0f);
    }
    else
    {
        step_5 = vec4<f32>(d_3, w_6.y, w_6.z, 0.0f);
    }
    crush_record_0(k_16, step_5, u32(3));
    return;
}

struct Sym_0
{
     d_4 : vec3<f32>,
     o_5 : vec3<f32>,
};

fn sym_zero_0() -> Sym_0
{
    var s_6 : Sym_0;
    var _S242 : vec3<f32> = vec3<f32>(0.0f);
    s_6.d_4 = _S242;
    s_6.o_5 = _S242;
    return s_6;
}

fn sym_outer_0( r_9 : vec3<f32>,  f_9 : vec3<f32>) -> Sym_0
{
    var s_7 : Sym_0;
    s_7.d_4 = r_9 * f_9;
    var _S243 : f32 = r_9.x;
    var _S244 : f32 = f_9.y;
    var _S245 : f32 = r_9.y;
    var _S246 : f32 = f_9.x;
    var _S247 : f32 = f_9.z;
    var _S248 : f32 = r_9.z;
    s_7.o_5 = vec3<f32>(0.5f) * vec3<f32>(_S243 * _S244 + _S245 * _S246, _S243 * _S247 + _S248 * _S246, _S245 * _S247 + _S248 * _S244);
    return s_7;
}

fn sym_add_0( a_5 : Sym_0,  b_20 : Sym_0) -> Sym_0
{
    var s_8 : Sym_0;
    s_8.d_4 = a_5.d_4 + b_20.d_4;
    s_8.o_5 = a_5.o_5 + b_20.o_5;
    return s_8;
}

fn add_load_0( w_7 : ptr<function, Sym_0>,  f_sum_0 : ptr<function, vec3<f32>>,  spread_0 : ptr<function, vec4<f32>>,  r_10 : vec3<f32>,  f_10 : vec3<f32>)
{
    (*w_7) = sym_add_0((*w_7), sym_outer_0(r_10, f_10));
    (*f_sum_0) = (*f_sum_0) + f_10;
    var m_3 : f32 = length(f_10);
    (*spread_0) = (*spread_0) + vec4<f32>(vec3<f32>(m_3) * r_10, m_3);
    return;
}

fn add_load_1( w_8 : ptr<function, Sym_0>,  f_sum_1 : ptr<function, vec3<f32>>,  spread_1 : ptr<function, vec4<f32>>,  r_11 : vec3<f32>,  f_11 : vec3<f32>)
{
    (*w_8) = sym_add_0((*w_8), sym_outer_0(r_11, f_11));
    (*f_sum_1) = (*f_sum_1) + f_11;
    var m_4 : f32 = length(f_11);
    (*spread_1) = (*spread_1) + vec4<f32>(vec3<f32>(m_4) * r_11, m_4);
    return;
}

fn flow_virial_0( k_17 : u32,  q_26 : Quat_0,  origin_11 : vec3<f32>,  centre_1 : vec3<f32>,  wf_0 : ptr<function, Sym_0>,  total_0 : ptr<function, f32>)
{
    (*wf_0) = sym_zero_0();
    (*total_0) = 0.0f;
    var adj_0 : vec4<f32> = crush_tab_0[crush_0.t_adj_0 + k_17];
    var first_2 : u32 = (bitcast<u32>((adj_0.x)));
    var _S249 : u32 = (bitcast<u32>((adj_0.y)));
    var j_3 : u32 = first_2;
    loop
    {
        if(j_3 < (first_2 + _S249))
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
        var l_0 : f32 = crush_rw_0[crush_0.r_flow_0 + bi_0 / u32(4)][bi_0 % u32(4)];
        if(!((crush_rw_0[crush_0.r_flow_0 + bi_0 / u32(4)][bi_0 % u32(4)]) > 0.0f))
        {
            j_3 = j_3 + u32(1);
            continue;
        }
        var r_12 : vec3<f32> = origin_11 + rotate_0(q_26, crush_tab_0[crush_0.t_bonds_0 + bi_0].xyz) - centre_1;
        var n_3 : vec3<f32> = rotate_0(q_26, crush_tab_0[crush_0.t_normal_0 + bi_0].xyz);
        var out_n_0 : vec3<f32>;
        if((dot(n_3, r_12)) >= 0.0f)
        {
            out_n_0 = n_3;
        }
        else
        {
            out_n_0 = (vec3<f32>(0) - n_3);
        }
        (*wf_0) = sym_add_0((*wf_0), sym_outer_0(r_12, vec3<f32>(- l_0) * out_n_0));
        (*total_0) = (*total_0) + l_0;
        j_3 = j_3 + u32(1);
    }
    return;
}

fn bond_centroid_0( k_18 : u32,  q_27 : Quat_0,  origin_12 : vec3<f32>,  centre_2 : vec3<f32>) -> vec4<f32>
{
    var adj_1 : vec4<f32> = crush_tab_0[crush_0.t_adj_0 + k_18];
    var first_3 : u32 = (bitcast<u32>((adj_1.x)));
    var _S250 : u32 = (bitcast<u32>((adj_1.y)));
    var _S251 : vec3<f32> = vec3<f32>(0.0f);
    var j_4 : u32 = first_3;
    var sum_4 : vec3<f32> = _S251;
    var area_0 : f32 = 0.0f;
    loop
    {
        if(j_4 < (first_3 + _S250))
        {
        }
        else
        {
            break;
        }
        var bi_1 : u32 = (bitcast<u32>((crush_tab_0[crush_0.t_list_0 + j_4 / u32(4)][j_4 % u32(4)])));
        if(work_0[w_alive_0() + bi_1] == u32(0))
        {
            j_4 = j_4 + u32(1);
            continue;
        }
        var bc_0 : vec4<f32> = crush_tab_0[crush_0.t_bonds_0 + bi_1];
        var _S252 : f32 = bc_0.w;
        var area_1 : f32 = area_0 + _S252;
        sum_4 = sum_4 + vec3<f32>(_S252) * (origin_12 + rotate_0(q_27, bc_0.xyz) - centre_2);
        area_0 = area_1;
        j_4 = j_4 + u32(1);
    }
    if(area_0 > 0.0f)
    {
        sum_4 = sum_4 / vec3<f32>(area_0);
    }
    else
    {
        sum_4 = _S251;
    }
    return vec4<f32>(sum_4, area_0);
}

fn balanced_0( w_9 : Sym_0,  f_12 : vec3<f32>,  spread_2 : vec4<f32>,  bonds_1 : vec4<f32>) -> Sym_0
{
    var at_10 : vec3<f32>;
    if((bonds_1.w) > 0.0f)
    {
        at_10 = bonds_1.xyz;
    }
    else
    {
        var _S253 : f32 = spread_2.w;
        if(_S253 > 0.0f)
        {
            at_10 = spread_2.xyz / vec3<f32>(_S253);
        }
        else
        {
            at_10 = vec3<f32>(0.0f);
        }
    }
    return sym_add_0(w_9, sym_outer_0(at_10, (vec3<f32>(0) - f_12)));
}

fn sym_norm2_0( s_9 : Sym_0) -> f32
{
    var _S254 : vec3<f32> = s_9.d_4;
    var _S255 : vec3<f32> = s_9.o_5;
    return dot(_S254, _S254) + 2.0f * dot(_S255, _S255);
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
    var _S256 : f32 = pq_0.x;
    if(!(_S256 > 0.0f))
    {
        return 0.0f;
    }
    var _S257 : f32 = max(pq_0.y - (m0_0.y + m0_0.z * _S256), _S256 - m0_0.x);
    var _S258 : f32;
    if(_S257 > 0.0f)
    {
        _S258 = _S257;
    }
    else
    {
        _S258 = 0.0f;
    }
    return _S258;
}

fn crush_two_0( a_6 : Sym_0,  dt_a_0 : f32,  b_21 : Sym_0,  dt_b_0 : f32,  k_19 : u32,  damage_0 : f32) -> vec4<f32>
{
    var _S259 : u32 = u32(2) * k_19;
    var m0_1 : vec4<f32> = crush_tab_0[_S259];
    var m1_0 : vec4<f32> = crush_tab_0[_S259 + u32(1)];
    var _S260 : f32 = m1_0.y;
    var pa_1 : vec2<f32> = crush_invariants_0(a_6, _S260);
    var pb_2 : vec2<f32> = crush_invariants_0(b_21, _S260);
    var ea_0 : f32 = crush_excess_0(pa_1, m0_1);
    var _S261 : bool = dt_b_0 > 0.0f;
    var eb_0 : f32;
    if(_S261)
    {
        eb_0 = crush_excess_0(pb_2, m0_1);
    }
    else
    {
        eb_0 = 0.0f;
    }
    var d_5 : f32 = damage_0 + (ea_0 * ea_0 * dt_a_0 + eb_0 * eb_0 * dt_b_0) / (m1_0.x * m0_1.w);
    var peak_2 : vec2<f32>;
    if(_S261)
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

fn crush_loads_flow_0( k_20 : u32)
{
    var slot_15 : u32 = body_of_0(k_20);
    var b_22 : u32 = u32(7) * slot_15;
    var q_28 : Quat_0;
    var origin_13 : vec3<f32>;
    structure_pose_3(slot_15, b_22, &(q_28), &(origin_13));
    var _S262 : u32 = u32(4) * k_20;
    var c0_3 : vec4<f32> = chunks_0[_S262];
    var _S263 : vec4<f32> = chunks_0[_S262 + u32(3)];
    var centre_3 : vec3<f32> = origin_13 + rotate_0(q_28, c0_3.xyz);
    var inv_mass_4 : f32 = bodies_0[b_22].w;
    var _S264 : f32;
    if(inv_mass_4 > 0.0f)
    {
        _S264 = 1.0f / inv_mass_4;
    }
    else
    {
        _S264 = float_bits_0(u32(2139095040));
    }
    var dt_2 : f32 = params_0.dt_0;
    var _S265 : Sym_0 = sym_zero_0();
    var wa_0 : Sym_0 = _S265;
    var _S266 : vec3<f32> = vec3<f32>(0.0f);
    var fa_0 : vec3<f32> = _S266;
    var _S267 : vec4<f32> = vec4<f32>(0.0f);
    var sa_0 : vec4<f32> = _S267;
    var wb_0 : Sym_0 = _S265;
    var fb_0 : vec3<f32> = _S266;
    var sb_1 : vec4<f32> = _S267;
    var _S268 : u32 = sort_0[k_20];
    var _S269 : u32 = sort_0[params_0.chunk_count_0 + u32(1) + k_20];
    var tau_0 : f32 = 0.0f;
    var i_7 : u32 = u32(0);
    loop
    {
        if(i_7 < _S269)
        {
        }
        else
        {
            break;
        }
        var r_13 : Record_std430_0 = manifolds_0[_S268 + i_7];
        var _S270 : vec3<f32> = r_13.impulse_0.xyz;
        var j_5 : f32 = length(_S270);
        if(j_5 == 0.0f)
        {
            i_7 = i_7 + u32(1);
            continue;
        }
        var m_other_2 : f32 = r_13.other_0.x;
        var m_5 : f32;
        if(isinf_0(_S264))
        {
            if(isinf_0(m_other_2))
            {
                m_5 = c0_3.w;
            }
            else
            {
                m_5 = m_other_2;
            }
        }
        else
        {
            if(isinf_0(m_other_2))
            {
                m_5 = _S264;
            }
            else
            {
                m_5 = _S264 * m_other_2 / (_S264 + m_other_2);
            }
        }
        var _S271 : f32 = r_13.impulse_0.w;
        var _S272 : f32 = min(j_5, 2.0f * m_5 * max(_S271 - params_0.band_0, 0.0f));
        var rr_0 : vec3<f32> = r_13.point_0.xyz - centre_3;
        var dir_1 : vec3<f32> = _S270 / vec3<f32>(j_5);
        if(_S272 < j_5)
        {
            add_load_0(&(wa_0), &(fa_0), &(sa_0), rr_0, dir_1 * vec3<f32>(((j_5 - _S272) / dt_2)));
        }
        var tau_1 : f32;
        if(_S272 > 0.0f)
        {
            var _S273 : f32 = _S263.y;
            var duration_2 : f32 = hertz_duration_0(m_5, min(r_13.other_0.z, _S263.z), 1.0f / ((1.0f - _S273 * _S273) / _S263.x + 1.0f / r_13.other_0.y), _S271);
            add_load_0(&(wb_0), &(fb_0), &(sb_1), rr_0, dir_1 * vec3<f32>((3.14159274101257324f * _S272 / (2.0f * duration_2))));
            tau_1 = max(tau_0, min(duration_2, dt_2));
        }
        else
        {
            tau_1 = tau_0;
        }
        tau_0 = tau_1;
        i_7 = i_7 + u32(1);
    }
    var _S274 : u32 = w_pulses_0() + _S262 + u32(2) * params_0.parity_0;
    var _S275 : u32 = work_0[_S274 + u32(1)];
    var _S276 : u32 = l_pool_0(params_0.parity_0) + u32(3) * work_0[_S274];
    var p_6 : u32 = u32(0);
    loop
    {
        if(p_6 < _S275)
        {
        }
        else
        {
            break;
        }
        var _S277 : u32 = _S276 + u32(3) * p_6;
        var bb_0 : vec4<f32> = loads_0[_S277 + u32(1)];
        var c_16 : vec4<f32> = loads_0[_S277 + u32(2)];
        var mean_2 : f32 = pulse_mean_0(loads_0[_S277].x, loads_0[_S277].y, loads_0[_S277].z, loads_0[_S277].w, loads_0[_S277 + u32(1)].w, dt_2);
        if(mean_2 != 0.0f)
        {
            add_load_0(&(wa_0), &(fa_0), &(sa_0), bb_0.xyz - centre_3, c_16.xyz * vec3<f32>(mean_2));
        }
        p_6 = p_6 + u32(1);
    }
    var wf_1 : Sym_0;
    var flow_total_0 : f32;
    flow_virial_0(k_20, q_28, origin_13, centre_3, &(wf_1), &(flow_total_0));
    var contact_1 : bool;
    if((sa_0.w) > 0.0f)
    {
        contact_1 = true;
    }
    else
    {
        contact_1 = (sb_1.w) > 0.0f;
    }
    var _S278 : bool;
    if(!contact_1)
    {
        _S278 = !(flow_total_0 > 0.0f);
    }
    else
    {
        _S278 = false;
    }
    if(_S278)
    {
        return;
    }
    var _S279 : Sym_0 = wf_1;
    var _S280 : Sym_0 = wf_1;
    var peak_contact_0 : Sym_0;
    var base_3 : Sym_0;
    var peak_3 : Sym_0;
    if(contact_1)
    {
        var bonds_2 : vec4<f32> = bond_centroid_0(k_20, q_28, origin_13, centre_3);
        var _S281 : Sym_0 = sym_add_0(balanced_0(wa_0, fa_0, sa_0, bonds_2), wf_1);
        var peak_contact_1 : Sym_0 = balanced_0(sym_add_0(wa_0, wb_0), fa_0 + fb_0, sa_0 + sb_1, bonds_2);
        var _S282 : Sym_0 = sym_add_0(peak_contact_1, wf_1);
        peak_contact_0 = peak_contact_1;
        base_3 = _S281;
        peak_3 = _S282;
    }
    else
    {
        peak_contact_0 = _S265;
        base_3 = _S279;
        peak_3 = _S280;
    }
    var cause_1 : u32;
    if((sym_norm2_0(wf_1)) >= (sym_norm2_0(peak_contact_0)))
    {
        cause_1 = u32(4);
    }
    else
    {
        cause_1 = u32(1);
    }
    crush_record_0(k_20, crush_two_0(base_3, dt_2 - tau_0, peak_3, tau_0, k_20, crush_rw_0[k_20].x), cause_1);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_loads(@builtin(global_invocation_id) id_29 : vec3<u32>)
{
    var k_21 : u32 = id_29.x;
    var _S283 : bool;
    if(k_21 >= (params_0.chunk_count_0))
    {
        _S283 = true;
    }
    else
    {
        _S283 = (crush_0.enabled_0) == u32(0);
    }
    if(_S283)
    {
        _S283 = true;
    }
    else
    {
        _S283 = !crushable_0(k_21);
    }
    if(_S283)
    {
        _S283 = true;
    }
    else
    {
        _S283 = (((crush_flags_0(k_21)) & (u32(1)))) != u32(0);
    }
    if(_S283)
    {
        return;
    }
    if((crush_0.flow_0) != u32(0))
    {
        crush_loads_flow_0(k_21);
        return;
    }
    var slot_16 : u32 = body_of_0(k_21);
    var b_23 : u32 = u32(7) * slot_16;
    var q_29 : Quat_0;
    var origin_14 : vec3<f32>;
    structure_pose_4(slot_16, b_23, &(q_29), &(origin_14));
    var _S284 : u32 = u32(4) * k_21;
    var c0_4 : vec4<f32> = chunks_0[_S284];
    var _S285 : vec4<f32> = chunks_0[_S284 + u32(3)];
    var centre_4 : vec3<f32> = origin_14 + rotate_0(q_29, c0_4.xyz);
    var inv_mass_5 : f32 = bodies_0[b_23].w;
    var _S286 : f32;
    if(inv_mass_5 > 0.0f)
    {
        _S286 = 1.0f / inv_mass_5;
    }
    else
    {
        _S286 = float_bits_0(u32(2139095040));
    }
    var dt_3 : f32 = params_0.dt_0;
    var _S287 : Sym_0 = sym_zero_0();
    var wa_1 : Sym_0 = _S287;
    var _S288 : vec3<f32> = vec3<f32>(0.0f);
    var fa_1 : vec3<f32> = _S288;
    var _S289 : vec4<f32> = vec4<f32>(0.0f);
    var sa_1 : vec4<f32> = _S289;
    var wb_1 : Sym_0 = _S287;
    var fb_1 : vec3<f32> = _S288;
    var sb_2 : vec4<f32> = _S289;
    var _S290 : u32 = sort_0[k_21];
    var _S291 : u32 = sort_0[params_0.chunk_count_0 + u32(1) + k_21];
    var tau_2 : f32 = 0.0f;
    var i_8 : u32 = u32(0);
    loop
    {
        if(i_8 < _S291)
        {
        }
        else
        {
            break;
        }
        var r_14 : Record_std430_0 = manifolds_0[_S290 + i_8];
        var _S292 : vec3<f32> = r_14.impulse_0.xyz;
        var j_6 : f32 = length(_S292);
        if(j_6 == 0.0f)
        {
            i_8 = i_8 + u32(1);
            continue;
        }
        var m_other_3 : f32 = r_14.other_0.x;
        var m_6 : f32;
        if(isinf_0(_S286))
        {
            if(isinf_0(m_other_3))
            {
                m_6 = c0_4.w;
            }
            else
            {
                m_6 = m_other_3;
            }
        }
        else
        {
            if(isinf_0(m_other_3))
            {
                m_6 = _S286;
            }
            else
            {
                m_6 = _S286 * m_other_3 / (_S286 + m_other_3);
            }
        }
        var _S293 : f32 = r_14.impulse_0.w;
        var _S294 : f32 = min(j_6, 2.0f * m_6 * max(_S293 - params_0.band_0, 0.0f));
        var rr_1 : vec3<f32> = r_14.point_0.xyz - centre_4;
        var dir_2 : vec3<f32> = _S292 / vec3<f32>(j_6);
        if(_S294 < j_6)
        {
            add_load_1(&(wa_1), &(fa_1), &(sa_1), rr_1, dir_2 * vec3<f32>(((j_6 - _S294) / dt_3)));
        }
        var tau_3 : f32;
        if(_S294 > 0.0f)
        {
            var _S295 : f32 = _S285.y;
            var duration_3 : f32 = hertz_duration_0(m_6, min(r_14.other_0.z, _S285.z), 1.0f / ((1.0f - _S295 * _S295) / _S285.x + 1.0f / r_14.other_0.y), _S293);
            add_load_1(&(wb_1), &(fb_1), &(sb_2), rr_1, dir_2 * vec3<f32>((3.14159274101257324f * _S294 / (2.0f * duration_3))));
            tau_3 = max(tau_2, min(duration_3, dt_3));
        }
        else
        {
            tau_3 = tau_2;
        }
        tau_2 = tau_3;
        i_8 = i_8 + u32(1);
    }
    var _S296 : u32 = w_pulses_0() + _S284 + u32(2) * params_0.parity_0;
    var _S297 : u32 = work_0[_S296 + u32(1)];
    var _S298 : u32 = l_pool_0(params_0.parity_0) + u32(3) * work_0[_S296];
    var p_7 : u32 = u32(0);
    loop
    {
        if(p_7 < _S297)
        {
        }
        else
        {
            break;
        }
        var _S299 : u32 = _S298 + u32(3) * p_7;
        var bb_1 : vec4<f32> = loads_0[_S299 + u32(1)];
        var c_17 : vec4<f32> = loads_0[_S299 + u32(2)];
        var mean_3 : f32 = pulse_mean_0(loads_0[_S299].x, loads_0[_S299].y, loads_0[_S299].z, loads_0[_S299].w, loads_0[_S299 + u32(1)].w, dt_3);
        if(mean_3 != 0.0f)
        {
            add_load_1(&(wa_1), &(fa_1), &(sa_1), bb_1.xyz - centre_4, c_17.xyz * vec3<f32>(mean_3));
        }
        p_7 = p_7 + u32(1);
    }
    if(!((sa_1.w) > 0.0f))
    {
        _S283 = !((sb_2.w) > 0.0f);
    }
    else
    {
        _S283 = false;
    }
    if(_S283)
    {
        return;
    }
    var bonds_3 : vec4<f32> = bond_centroid_0(k_21, q_29, origin_14, centre_4);
    crush_record_0(k_21, crush_two_0(balanced_0(wa_1, fa_1, sa_1, bonds_3), dt_3 - tau_2, balanced_0(sym_add_0(wa_1, wb_1), fa_1 + fb_1, sa_1 + sb_2, bonds_3), tau_2, k_21, crush_rw_0[k_21].x), u32(1));
    return;
}

fn a_cover_0() -> u32
{
    return a_strike_dist_0() + params_0.chunk_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_patch(@builtin(global_invocation_id) id_30 : vec3<u32>)
{
    var b_24 : u32 = id_30.x;
    var _S300 : bool;
    if(b_24 >= (params_0.body_count_0))
    {
        _S300 = true;
    }
    else
    {
        _S300 = (params_0.shot_count_0) == u32(0);
    }
    if(_S300)
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
    var bill_0 : u32 = crush_0.r_bill_0 + u32(4) * b_24;
    var _S301 : vec4<f32> = vec4<f32>(0.0f);
    crush_rw_0[bill_0] = _S301;
    var _S302 : u32 = bill_0 + u32(3);
    crush_rw_0[_S302] = _S301;
    var shot_5 : u32;
    var _S303 : bool = struck_0(b_24, &(shot_5));
    if(!_S303)
    {
        return;
    }
    var _S304 : u32 = a_strike_0() + u32(9) * b_24;
    var k_22 : u32 = atomicLoad(&(atom_0[_S304 + u32(8)]));
    var _S305 : u32 = atomicLoad(&(atom_0[_S304 + u32(6)]));
    var t_in_3 : f32 = (bitcast<f32>((_S305)));
    if(!(t_in_3 <= (params_0.dt_0)))
    {
        _S300 = true;
    }
    else
    {
        _S300 = k_22 == u32(4294967295);
    }
    if(_S300)
    {
        return;
    }
    var st_5 : u32 = strike_at_0(b_24);
    var s0_0 : vec4<f32> = frames_0[st_5];
    var _S306 : vec3<f32> = frames_0[st_5 + u32(1)].xyz;
    var speed_1 : f32 = length(_S306);
    var d_6 : vec3<f32> = _S306 / vec3<f32>(max(speed_1, 9.999999960041972e-13f));
    var info_3 : vec4<f32> = shots_0[u32(2) * shot_5];
    var radius_0 : f32 = info_3.y;
    var _S307 : f32 = max(info_3.z, 0.00100000004749745f);
    var m_imp_1 : f32 = info_3.w;
    var inv_mass_6 : f32 = bodies_0[u32(7) * b_24].w;
    var _S308 : bool = inv_mass_6 == 0.0f;
    var m_body_2 : f32;
    if(_S308)
    {
        m_body_2 = 0.0f;
    }
    else
    {
        m_body_2 = 1.0f / inv_mass_6;
    }
    var m_eff_1 : f32;
    if(_S308)
    {
        m_eff_1 = m_imp_1;
    }
    else
    {
        m_eff_1 = m_imp_1 * m_body_2 / (m_imp_1 + m_body_2);
    }
    var modulus_1 : f32 = 1.0f / (1.0f / shots_0[u32(2) * shot_5 + u32(1)].x + 1.0f / geometry_0[u32(3) * k_22 + u32(1)].w);
    var _S309 : f32 = max(speed_1, 0.00100000004749745f);
    var tau_4 : f32 = 2.86999988555908203f * pow(m_eff_1 * m_eff_1 / (_S307 * modulus_1 * modulus_1 * _S309), 0.20000000298023224f);
    var force_0 : f32 = m_eff_1 * speed_1 / tau_4;
    var contact_2 : vec3<f32> = s0_0.xyz + _S306 * vec3<f32>(t_in_3) + d_6 * vec3<f32>(radius_0);
    var delta_0 : f32 = pow(15.0f * m_eff_1 * _S309 * _S309 / (16.0f * modulus_1 * sqrt(_S307)), 0.40000000596046448f);
    var a_h_0 : f32 = sqrt(radius_0 * delta_0);
    var p_cap_0 : f32;
    if(crushable_0(k_22))
    {
        p_cap_0 = crush_tab_0[u32(2) * k_22].x;
    }
    else
    {
        p_cap_0 = 0.0f;
    }
    var a_p_0 : f32;
    if(p_cap_0 > 0.0f)
    {
        a_p_0 = sqrt(force_0 / (3.14159274101257324f * p_cap_0));
    }
    else
    {
        a_p_0 = 0.0f;
    }
    var _S310 : f32 = min(radius_0, max(a_h_0, a_p_0));
    var _S311 : f32 = max(delta_0, radius_0 - sqrt(max(radius_0 * radius_0 - _S310 * _S310, 0.0f)));
    crush_rw_0[bill_0 + u32(1)] = vec4<f32>(contact_2, 0.0f);
    crush_rw_0[bill_0 + u32(2)] = vec4<f32>(d_6, _S310);
    crush_rw_0[_S302] = vec4<f32>(force_0, tau_4, _S311, f32(k_22 + u32(1)));
    var _S312 : u32 = a_cover_0() + u32(128) * b_24;
    var s_10 : u32 = u32(0);
    loop
    {
        if(s_10 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S313 : u32 = _S312 + u32(2) * s_10;
        atomicStore(&(atom_0[_S313]), u32(4294967295));
        atomicStore(&(atom_0[_S313 + u32(1)]), u32(4294967295));
        s_10 = s_10 + u32(1);
    }
    return;
}

fn cover_patch_0( k_23 : u32,  b_25 : ptr<function, u32>,  p1_0 : ptr<function, vec4<f32>>,  p2_0 : ptr<function, vec4<f32>>) -> bool
{
    (*b_25) = body_of_0(k_23);
    var _S314 : vec4<f32> = vec4<f32>(0.0f);
    (*p1_0) = _S314;
    (*p2_0) = _S314;
    var shot_6 : u32;
    var _S315 : bool = struck_0((*b_25), &(shot_6));
    var _S316 : bool;
    if(!_S315)
    {
        _S316 = true;
    }
    else
    {
        _S316 = (((crush_flags_0(k_23)) & (u32(1)))) != u32(0);
    }
    if(_S316)
    {
        return false;
    }
    var bill_1 : u32 = crush_0.r_bill_0 + u32(4) * (*b_25);
    if((crush_rw_0[bill_1 + u32(3)].w) == 0.0f)
    {
        return false;
    }
    (*p1_0) = vec4<f32>(crush_rw_0[bill_1 + u32(1)].xyz, shots_0[u32(2) * shot_6].y);
    (*p2_0) = crush_rw_0[bill_1 + u32(2)];
    return true;
}

fn cover_offset_0( s_11 : u32,  d_7 : vec3<f32>,  a_7 : f32,  rho_0 : ptr<function, f32>) -> vec3<f32>
{
    var ring_0 : u32 = s_11 / u32(8);
    var sector_0 : u32 = s_11 % u32(8);
    (*rho_0) = a_7 * sqrt((f32(ring_0) + 0.5f) / 8.0f);
    var theta_1 : f32 = 6.28318548202514648f * (f32(sector_0) + 0.5f * f32((ring_0 & (u32(1))))) / 8.0f;
    var e_2 : vec3<f32>;
    if((abs(d_7.x)) < 0.57735002040863037f)
    {
        e_2 = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_7.y)) < 0.57735002040863037f)
        {
            e_2 = vec3<f32>(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_2 = vec3<f32>(0.0f, 0.0f, 1.0f);
        }
    }
    var u_2 : vec3<f32> = normalize(cross(d_7, e_2));
    return vec3<f32>((*rho_0)) * (vec3<f32>(cos(theta_1)) * u_2 + vec3<f32>(sin(theta_1)) * cross(d_7, u_2));
}

fn cover_offset_1( s_12 : u32,  d_8 : vec3<f32>,  a_8 : f32,  rho_1 : ptr<function, f32>) -> vec3<f32>
{
    var ring_1 : u32 = s_12 / u32(8);
    var sector_1 : u32 = s_12 % u32(8);
    (*rho_1) = a_8 * sqrt((f32(ring_1) + 0.5f) / 8.0f);
    var theta_2 : f32 = 6.28318548202514648f * (f32(sector_1) + 0.5f * f32((ring_1 & (u32(1))))) / 8.0f;
    var e_3 : vec3<f32>;
    if((abs(d_8.x)) < 0.57735002040863037f)
    {
        e_3 = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_8.y)) < 0.57735002040863037f)
        {
            e_3 = vec3<f32>(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_3 = vec3<f32>(0.0f, 0.0f, 1.0f);
        }
    }
    var u_3 : vec3<f32> = normalize(cross(d_8, e_3));
    return vec3<f32>((*rho_1)) * (vec3<f32>(cos(theta_2)) * u_3 + vec3<f32>(sin(theta_2)) * cross(d_8, u_3));
}

fn cover_depth_0( k_24 : u32,  s_13 : u32,  contact_3 : vec3<f32>,  d_9 : vec3<f32>,  a_9 : f32,  radius_1 : f32) -> f32
{
    var rho_2 : f32;
    var _S317 : f32 = 2.0f * radius_1;
    var _S318 : vec3<f32> = contact_3 + cover_offset_0(s_13, d_9, a_9, &(rho_2)) - d_9 * vec3<f32>(_S317);
    var q_30 : Quat_0;
    var _S319 : vec3<f32> = chunk_box_world_0(k_24, &(q_30));
    var _S320 : u32 = u32(3) * k_24;
    var _S321 : Quat_0 = normalized_0(quat_mul_0(q_30, quat_of_0(geometry_0[_S320 + u32(2)])));
    var _S322 : vec3<f32> = geometry_0[_S320 + u32(1)].xyz;
    var t0_2 : f32 = -1.00000001504746622e+30f;
    var t1_1 : f32 = 1.00000001504746622e+30f;
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
        var e_4 : vec3<f32> = vec3<f32>(0.0f);
        e_4[ax_3] = 1.0f;
        var axis_1 : vec3<f32> = rotate_0(_S321, e_4);
        var o_6 : f32 = dot(_S318 - _S319, axis_1);
        var dv_1 : f32 = dot(d_9, axis_1);
        if((abs(dv_1)) < 9.99999971718068537e-10f)
        {
            if((abs(o_6)) > (_S322[ax_3]))
            {
                return 1.00000001504746622e+30f;
            }
            ax_3 = ax_3 + u32(1);
            continue;
        }
        var ta_2 : f32 = (- _S322[ax_3] - o_6) / dv_1;
        var tb_2 : f32 = (_S322[ax_3] - o_6) / dv_1;
        var _S323 : f32 = min(t1_1, max(ta_2, tb_2));
        t0_2 = max(t0_2, min(ta_2, tb_2));
        t1_1 = _S323;
        ax_3 = ax_3 + u32(1);
    }
    var _S324 : bool;
    if(!(t0_2 <= t1_1))
    {
        _S324 = true;
    }
    else
    {
        _S324 = t1_1 < 0.0f;
    }
    if(_S324)
    {
        return 1.00000001504746622e+30f;
    }
    return t0_2 - _S317;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_cover(@builtin(global_invocation_id) id_31 : vec3<u32>)
{
    var k_25 : u32 = id_31.x;
    var _S325 : bool;
    if(k_25 >= (params_0.chunk_count_0))
    {
        _S325 = true;
    }
    else
    {
        _S325 = (params_0.shot_count_0) == u32(0);
    }
    if(_S325)
    {
        _S325 = true;
    }
    else
    {
        _S325 = (crush_0.enabled_0) == u32(0);
    }
    if(_S325)
    {
        return;
    }
    var b_26 : u32;
    var p1_1 : vec4<f32>;
    var p2_1 : vec4<f32>;
    var _S326 : bool = cover_patch_0(k_25, &(b_26), &(p1_1), &(p2_1));
    if(!_S326)
    {
        return;
    }
    var _S327 : u32 = a_cover_0() + u32(128) * b_26;
    var s_14 : u32 = u32(0);
    loop
    {
        if(s_14 < u32(64))
        {
        }
        else
        {
            break;
        }
        var t_5 : f32 = cover_depth_0(k_25, s_14, p1_1.xyz, p2_1.xyz, p2_1.w, p1_1.w);
        if(t_5 < 5.0000000752373311e+29f)
        {
            var _S328 : u32 = atomicMin(&(atom_0[_S327 + u32(2) * s_14]), ordered_0(t_5));
        }
        s_14 = s_14 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_claim(@builtin(global_invocation_id) id_32 : vec3<u32>)
{
    var k_26 : u32 = id_32.x;
    var _S329 : bool;
    if(k_26 >= (params_0.chunk_count_0))
    {
        _S329 = true;
    }
    else
    {
        _S329 = (params_0.shot_count_0) == u32(0);
    }
    if(_S329)
    {
        _S329 = true;
    }
    else
    {
        _S329 = (crush_0.enabled_0) == u32(0);
    }
    if(_S329)
    {
        return;
    }
    var b_27 : u32;
    var p1_2 : vec4<f32>;
    var p2_2 : vec4<f32>;
    var _S330 : bool = cover_patch_0(k_26, &(b_27), &(p1_2), &(p2_2));
    if(!_S330)
    {
        return;
    }
    var _S331 : u32 = a_cover_0() + u32(128) * b_27;
    var s_15 : u32 = u32(0);
    loop
    {
        if(s_15 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S332 : u32 = _S331 + u32(2) * s_15;
        var first_4 : u32 = atomicLoad(&(atom_0[_S332]));
        if(first_4 == u32(4294967295))
        {
            s_15 = s_15 + u32(1);
            continue;
        }
        var t_6 : f32 = cover_depth_0(k_26, s_15, p1_2.xyz, p2_2.xyz, p2_2.w, p1_2.w);
        if(t_6 < 5.0000000752373311e+29f)
        {
            _S329 = t_6 <= (unordered_0(first_4) + 0.00009999999747379f);
        }
        else
        {
            _S329 = false;
        }
        if(_S329)
        {
            var _S333 : u32 = atomicMin(&(atom_0[_S332 + u32(1)]), k_26);
        }
        s_15 = s_15 + u32(1);
    }
    return;
}

fn strike_load_0( k_27 : u32,  slot_17 : u32,  f_13 : vec3<f32>,  at_11 : vec3<f32>,  tau_5 : f32,  budget_0 : f32) -> f32
{
    var _S334 : bool;
    if(!crushable_0(k_27))
    {
        _S334 = true;
    }
    else
    {
        _S334 = (((crush_flags_0(k_27)) & (u32(1)))) != u32(0);
    }
    if(_S334)
    {
        _S334 = true;
    }
    else
    {
        _S334 = !(budget_0 > 0.0f);
    }
    if(_S334)
    {
        return 0.0f;
    }
    var q_31 : Quat_0;
    var origin_15 : vec3<f32>;
    structure_pose_0(slot_17, u32(7) * slot_17, &(q_31), &(origin_15));
    var centre_5 : vec3<f32> = origin_15 + rotate_0(q_31, chunks_0[u32(4) * k_27].xyz);
    var bonds_4 : vec4<f32> = bond_centroid_0(k_27, q_31, origin_15, centre_5);
    if(!((bonds_4.w) > 0.0f))
    {
        return 0.0f;
    }
    var w_10 : Sym_0 = sym_add_0(sym_outer_0(at_11 - centre_5, f_13), sym_outer_0(bonds_4.xyz, (vec3<f32>(0) - f_13)));
    var before_0 : f32 = crush_rw_0[k_27].x;
    var step_6 : vec4<f32> = crush_two_0(w_10, min(tau_5, params_0.dt_0), w_10, 0.0f, k_27, before_0);
    var _S335 : u32 = u32(2) * k_27;
    var per_0 : f32 = crush_tab_0[_S335].w * crush_tab_0[_S335 + u32(1)].y;
    var energy_0 : f32 = (step_6.x - before_0) * per_0;
    var energy_1 : f32;
    var step_7 : vec4<f32>;
    if(energy_0 > budget_0)
    {
        var d_10 : f32 = before_0 + budget_0 / per_0;
        var _S336 : f32 = min(d_10, 1.0f);
        var _S337 : f32 = step_6.y;
        var _S338 : f32 = step_6.z;
        if(d_10 >= 1.0f)
        {
            energy_1 = 1.0f;
        }
        else
        {
            energy_1 = 0.0f;
        }
        step_7 = vec4<f32>(_S336, _S337, _S338, energy_1);
        energy_1 = budget_0;
    }
    else
    {
        step_7 = step_6;
        energy_1 = energy_0;
    }
    crush_record_0(k_27, step_7, u32(2));
    return energy_1;
}

fn cell_in_0( mask_0 : vec2<u32>,  c_18 : u32) -> bool
{
    return (((((mask_0[c_18 / u32(32)]) >> ((c_18 % u32(32))))) & (u32(1)))) != u32(0);
}

fn strike_crush_body_0( b_28 : u32,  s_16 : u32,  budget_1 : f32) -> f32
{
    var total_1 : u32;
    var _S339 : bool;
    var bill_2 : u32 = crush_0.r_bill_0 + u32(4) * b_28;
    var _S340 : u32 = bill_2 + u32(1);
    var p3_0 : vec4<f32> = crush_rw_0[bill_2 + u32(3)];
    var contact_4 : vec3<f32> = crush_rw_0[_S340].xyz;
    var d_11 : vec3<f32> = crush_rw_0[bill_2 + u32(2)].xyz;
    var _S341 : f32 = crush_rw_0[bill_2 + u32(2)].w;
    var _S342 : f32 = shots_0[u32(2) * s_16].y;
    var _S343 : u32 = a_cover_0() + u32(128) * b_28;
    var first_5 : f32 = 1.00000001504746622e+30f;
    var c_19 : u32 = u32(0);
    loop
    {
        if(c_19 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S344 : u32 = _S343 + u32(2) * c_19;
        var _S345 : u32 = atomicLoad(&(atom_0[_S344 + u32(1)]));
        if(_S345 == u32(4294967295))
        {
            c_19 = c_19 + u32(1);
            continue;
        }
        var rho_3 : f32;
        var _S346 : vec3<f32> = cover_offset_1(c_19, d_11, _S341, &(rho_3));
        var _S347 : u32 = atomicLoad(&(atom_0[_S344]));
        first_5 = min(first_5, unordered_0(_S347) + _S342 - sqrt(max(_S342 * _S342 - rho_3 * rho_3, 0.0f)));
        c_19 = c_19 + u32(1);
    }
    var counted_0 : vec2<u32> = vec2<u32>(u32(0));
    c_19 = u32(0);
    var total_2 : u32 = u32(0);
    loop
    {
        if(c_19 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S348 : u32 = _S343 + u32(2) * c_19;
        var _S349 : u32 = atomicLoad(&(atom_0[_S348 + u32(1)]));
        if(_S349 == u32(4294967295))
        {
            c_19 = c_19 + u32(1);
            continue;
        }
        var rho_4 : f32;
        var _S350 : vec3<f32> = cover_offset_1(c_19, d_11, _S341, &(rho_4));
        var _S351 : u32 = atomicLoad(&(atom_0[_S348]));
        if((unordered_0(_S351) + _S342 - sqrt(max(_S342 * _S342 - rho_4 * rho_4, 0.0f)) - first_5) <= (p3_0.z))
        {
            counted_0[c_19 / u32(32)] = ((counted_0[c_19 / u32(32)]) | (((u32(1) << ((c_19 % u32(32)))))));
            total_1 = total_2 + u32(1);
        }
        else
        {
            total_1 = total_2;
        }
        total_2 = total_1;
        c_19 = c_19 + u32(1);
    }
    var _S352 : vec3<f32> = vec3<f32>(0.0f);
    var energy_2 : f32;
    var point_1 : vec3<f32>;
    if(total_2 == u32(0))
    {
        var energy_3 : f32 = strike_load_0(u32(p3_0.w) - u32(1), b_28, d_11 * vec3<f32>(p3_0.x), contact_4, p3_0.y, budget_1);
        var _S353 : vec3<f32> = vec3<f32>(energy_3) * contact_4;
        energy_2 = energy_3;
        point_1 = _S353;
    }
    else
    {
        energy_2 = 0.0f;
        point_1 = _S352;
    }
    c_19 = u32(0);
    loop
    {
        if(c_19 < u32(64))
        {
        }
        else
        {
            break;
        }
        if(!cell_in_0(counted_0, c_19))
        {
            c_19 = c_19 + u32(1);
            continue;
        }
        var k_28 : u32 = atomicLoad(&(atom_0[_S343 + u32(2) * c_19 + u32(1)]));
        var seen_0 : bool = false;
        total_1 = u32(0);
        loop
        {
            if(total_1 < c_19)
            {
            }
            else
            {
                break;
            }
            if(seen_0)
            {
                seen_0 = true;
            }
            else
            {
                if(cell_in_0(counted_0, total_1))
                {
                    var _S354 : u32 = atomicLoad(&(atom_0[_S343 + u32(2) * total_1 + u32(1)]));
                    _S339 = _S354 == k_28;
                }
                else
                {
                    _S339 = false;
                }
                seen_0 = _S339;
            }
            total_1 = total_1 + u32(1);
        }
        if(seen_0)
        {
            c_19 = c_19 + u32(1);
            continue;
        }
        var e_5 : u32 = c_19;
        var at_12 : vec3<f32> = _S352;
        var n_4 : u32 = u32(0);
        loop
        {
            if(e_5 < u32(64))
            {
            }
            else
            {
                break;
            }
            if(!cell_in_0(counted_0, e_5))
            {
                _S339 = true;
            }
            else
            {
                var _S355 : u32 = atomicLoad(&(atom_0[_S343 + u32(2) * e_5 + u32(1)]));
                _S339 = _S355 != k_28;
            }
            if(_S339)
            {
                e_5 = e_5 + u32(1);
                continue;
            }
            var rho_5 : f32;
            var _S356 : vec3<f32> = contact_4 + cover_offset_1(e_5, d_11, _S341, &(rho_5));
            var _S357 : u32 = atomicLoad(&(atom_0[_S343 + u32(2) * e_5]));
            var _S358 : u32 = n_4 + u32(1);
            at_12 = at_12 + (_S356 + d_11 * vec3<f32>(unordered_0(_S357)));
            n_4 = _S358;
            e_5 = e_5 + u32(1);
        }
        var _S359 : f32 = f32(n_4);
        var at_13 : vec3<f32> = at_12 / vec3<f32>(_S359);
        var took_0 : f32 = strike_load_0(k_28, b_28, d_11 * vec3<f32>((p3_0.x * _S359 / f32(total_2))), at_13, p3_0.y, budget_1 - energy_2);
        var point_2 : vec3<f32> = point_1 + vec3<f32>(took_0) * at_13;
        energy_2 = energy_2 + took_0;
        point_1 = point_2;
        c_19 = c_19 + u32(1);
    }
    crush_rw_0[bill_2] = vec4<f32>(energy_2, f32(s_16 + u32(1)), 0.0f, 0.0f);
    if(energy_2 > 0.0f)
    {
        crush_rw_0[_S340] = vec4<f32>(point_1 / vec3<f32>(energy_2), 0.0f);
    }
    return energy_2;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_crush(@builtin(global_invocation_id) id_33 : vec3<u32>)
{
    var s_17 : u32 = id_33.x;
    var _S360 : bool;
    if(s_17 >= (params_0.shot_count_0))
    {
        _S360 = true;
    }
    else
    {
        _S360 = (crush_0.enabled_0) == u32(0);
    }
    if(_S360)
    {
        return;
    }
    var sb_3 : u32 = u32(7) * (bitcast<u32>((shots_0[u32(2) * s_17].x)));
    var inv_mass_7 : f32 = bodies_0[sb_3].w;
    var v_10 : vec3<f32> = bodies_0[sb_3 + u32(3)].xyz;
    var budget_2 : f32;
    if(inv_mass_7 > 0.0f)
    {
        budget_2 = 0.5f * dot(v_10, v_10) / inv_mass_7;
    }
    else
    {
        budget_2 = 1.00000001504746622e+30f;
    }
    var b_29 : u32 = u32(0);
    loop
    {
        if(b_29 < (params_0.body_count_0))
        {
        }
        else
        {
            break;
        }
        var shot_7 : u32;
        var _S361 : bool = struck_0(b_29, &(shot_7));
        if(!_S361)
        {
            _S360 = true;
        }
        else
        {
            _S360 = shot_7 != s_17;
        }
        var _S362 : bool;
        if(_S360)
        {
            _S362 = true;
        }
        else
        {
            _S362 = (crush_rw_0[crush_0.r_bill_0 + u32(4) * b_29 + u32(3)].w) == 0.0f;
        }
        if(_S362)
        {
            b_29 = b_29 + u32(1);
            continue;
        }
        var _S363 : f32 = strike_crush_body_0(b_29, s_17, budget_2);
        budget_2 = budget_2 - _S363;
        b_29 = b_29 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_bill(@builtin(global_invocation_id) id_34 : vec3<u32>)
{
    var s_18 : u32 = id_34.x;
    var _S364 : bool;
    if(s_18 >= (params_0.shot_count_0))
    {
        _S364 = true;
    }
    else
    {
        _S364 = (crush_0.enabled_0) == u32(0);
    }
    if(_S364)
    {
        return;
    }
    var b_30 : u32 = u32(0);
    var energy_4 : f32 = 0.0f;
    loop
    {
        if(b_30 < (params_0.body_count_0))
        {
        }
        else
        {
            break;
        }
        var bill_3 : vec4<f32> = crush_rw_0[crush_0.r_bill_0 + u32(4) * b_30];
        if((crush_rw_0[crush_0.r_bill_0 + u32(4) * b_30].y) == f32(s_18 + u32(1)))
        {
            energy_4 = energy_4 + bill_3.x;
        }
        b_30 = b_30 + u32(1);
    }
    crush_rw_0[crush_0.r_shot_0 + s_18] = vec4<f32>(0.0f);
    if(!(energy_4 > 0.0f))
    {
        return;
    }
    var sb_4 : u32 = u32(7) * (bitcast<u32>((shots_0[u32(2) * s_18].x)));
    var inv_mass_8 : f32 = bodies_0[sb_4].w;
    if(!(inv_mass_8 > 0.0f))
    {
        return;
    }
    var m_7 : f32 = 1.0f / inv_mass_8;
    var v_11 : vec3<f32> = bodies_0[sb_4 + u32(3)].xyz;
    var speed_2 : f32 = length(v_11);
    var j_7 : f32 = m_7 * (speed_2 - sqrt(max(speed_2 * speed_2 - 2.0f * energy_4 / m_7, 0.0f)));
    var _S365 : bool = speed_2 > 0.0f;
    var _S366 : vec3<f32>;
    if(_S365)
    {
        _S366 = (vec3<f32>(0) - v_11) * vec3<f32>((j_7 / (m_7 * speed_2)));
    }
    else
    {
        _S366 = vec3<f32>(0.0f);
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
        var w_11 : u32 = sb_4 + region_1 * u32(7) * params_0.max_bodies_0 + u32(3);
        bodies_0[w_11] = vec4<f32>(bodies_0[w_11].xyz + _S366, bodies_0[w_11].w);
        region_1 = region_1 + u32(1);
    }
    if(_S365)
    {
        _S366 = v_11 * vec3<f32>((j_7 / speed_2));
    }
    else
    {
        _S366 = vec3<f32>(0.0f);
    }
    crush_rw_0[crush_0.r_shot_0 + s_18] = vec4<f32>(_S366, energy_4);
    return;
}

fn inverse_rotate_0( q_32 : Quat_0,  v_12 : vec3<f32>) -> vec3<f32>
{
    var c_20 : Quat_0;
    c_20.w_0 = q_32.w_0;
    c_20.x_1 = - q_32.x_1;
    c_20.y_0 = - q_32.y_0;
    c_20.z_0 = - q_32.z_0;
    return rotate_0(c_20, v_12);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_push(@builtin(global_invocation_id) id_35 : vec3<u32>)
{
    var b_31 : u32 = id_35.x;
    var _S367 : bool;
    if(b_31 >= (params_0.body_count_0))
    {
        _S367 = true;
    }
    else
    {
        _S367 = (params_0.shot_count_0) == u32(0);
    }
    if(_S367)
    {
        _S367 = true;
    }
    else
    {
        _S367 = (crush_0.enabled_0) == u32(0);
    }
    if(_S367)
    {
        return;
    }
    var _S368 : u32 = u32(4) * b_31;
    var bill_4 : vec4<f32> = crush_rw_0[crush_0.r_bill_0 + _S368];
    var _S369 : f32 = crush_rw_0[crush_0.r_bill_0 + _S368].x;
    if(!(_S369 > 0.0f))
    {
        return;
    }
    var paid_0 : vec4<f32> = crush_rw_0[crush_0.r_shot_0 + (u32(bill_4.y) - u32(1))];
    var bw_0 : u32 = u32(7) * b_31;
    var inv_mass_9 : f32 = bodies_0[bw_0].w;
    var _S370 : f32 = crush_rw_0[crush_0.r_shot_0 + (u32(bill_4.y) - u32(1))].w;
    if(!(_S370 > 0.0f))
    {
        _S367 = true;
    }
    else
    {
        _S367 = inv_mass_9 == 0.0f;
    }
    if(_S367)
    {
        return;
    }
    var j_8 : vec3<f32> = paid_0.xyz * vec3<f32>((_S369 / _S370));
    var o_7 : Quat_0 = quat_of_0(bodies_0[bw_0 + u32(1)]);
    var _S371 : vec3<f32> = rotate_0(o_7, bodies_0[bw_0 + u32(2)].xyz * inverse_rotate_0(o_7, cross(to_rubble_0(crush_rw_0[crush_0.r_bill_0 + _S368 + u32(1)].xyz) - bodies_0[bw_0].xyz, j_8)));
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
        var w_12 : u32 = bw_0 + region_2 * u32(7) * params_0.max_bodies_0;
        var _S372 : u32 = w_12 + u32(3);
        bodies_0[_S372] = vec4<f32>(bodies_0[_S372].xyz + j_8 * vec3<f32>(inv_mass_9), bodies_0[_S372].w);
        var _S373 : u32 = w_12 + u32(4);
        bodies_0[_S373] = vec4<f32>(bodies_0[_S373].xyz + _S371, bodies_0[_S373].w);
        region_2 = region_2 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_bonds(@builtin(global_invocation_id) id_36 : vec3<u32>)
{
    var i_9 : u32 = id_36.x;
    var _S374 : bool;
    if(i_9 >= (params_0.bond_count_0))
    {
        _S374 = true;
    }
    else
    {
        _S374 = (crush_0.enabled_0) == u32(0);
    }
    if(_S374)
    {
        _S374 = true;
    }
    else
    {
        _S374 = work_0[w_alive_0() + i_9] == u32(0);
    }
    if(_S374)
    {
        return;
    }
    var bond_4 : vec4<u32> = bonds_0[i_9];
    var _S375 : u32 = bond_4.x;
    if((((((bitcast<u32>((crush_rw_0[_S375].w))) | ((bitcast<u32>((crush_rw_0[bond_4.y].w)))))) & (u32(2)))) == u32(0))
    {
        return;
    }
    work_0[w_alive_0() + i_9] = u32(0);
    var _S376 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S375)]), u32(1));
    var _S377 : u32 = a_counters_0();
    var _S378 : u32 = atomicAdd(&(atom_0[_S377]), u32(1));
    var _S379 : u32 = atomicAdd(&(atom_0[_S377 + u32(1)]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_mark(@builtin(global_invocation_id) id_37 : vec3<u32>)
{
    var c_21 : u32 = id_37.x;
    var _S380 : bool;
    if(c_21 >= (params_0.chunk_count_0))
    {
        _S380 = true;
    }
    else
    {
        _S380 = (crush_0.enabled_0) == u32(0);
    }
    if(_S380)
    {
        return;
    }
    var now_0 : u32;
    if((((crush_flags_0(c_21)) & (u32(2)))) != u32(0))
    {
        now_0 = u32(1);
    }
    else
    {
        now_0 = u32(0);
    }
    work_0[w_new_0() + c_21] = now_0;
    if(now_0 != u32(0))
    {
        var _S381 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(10)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_crush_gate(@builtin(global_invocation_id) id_38 : vec3<u32>)
{
    var run_2 : bool;
    if((crush_0.enabled_0) != u32(0))
    {
        var _S382 : u32 = atomicLoad(&(atom_0[a_counters_0() + u32(10)]));
        run_2 = _S382 != u32(0);
    }
    else
    {
        run_2 = false;
    }
    var _S383 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S383 + u32(10)]), u32(0));
    atomicStore(&(atom_0[_S383 + u32(7)]), u32(0));
    var _S384 : u32;
    if(run_2)
    {
        var _S385 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S384 = _S385;
    }
    else
    {
        _S384 = u32(0);
    }
    set_args_0(u32(6), _S384);
    if(run_2)
    {
        var _S386 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S384 = _S386;
    }
    else
    {
        _S384 = u32(0);
    }
    set_args_0(u32(7), _S384);
    var _S387 : i32;
    if(run_2)
    {
        _S387 = i32(1);
    }
    else
    {
        _S387 = i32(0);
    }
    set_args_0(u32(8), u32(_S387));
    return;
}

fn write_body_0( slot_18 : u32,  pos_1 : vec3<f32>,  inv_mass_10 : f32,  orientation_1 : Quat_0,  inv_inertia_1 : vec3<f32>,  flags_3 : u32,  v_13 : vec3<f32>,  w_13 : vec3<f32>)
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
        var b_32 : u32 = u32(7) * (slot_18 + region_3 * params_0.max_bodies_0);
        bodies_0[b_32] = vec4<f32>(pos_1, inv_mass_10);
        bodies_0[b_32 + u32(1)] = quat_vec_0(orientation_1);
        bodies_0[b_32 + u32(2)] = vec4<f32>(inv_inertia_1, (bitcast<f32>((flags_3))));
        bodies_0[b_32 + u32(3)] = vec4<f32>(v_13, 0.0f);
        bodies_0[b_32 + u32(4)] = vec4<f32>(w_13, (bitcast<f32>((params_0.step_0))));
        bodies_0[b_32 + u32(5)] = vec4<f32>(0.0f);
        bodies_0[b_32 + u32(6)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
        region_3 = region_3 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_apply(@builtin(global_invocation_id) id_39 : vec3<u32>)
{
    var ax_4 : u32;
    var k_29 : u32 = id_39.x;
    var _S388 : bool;
    if(k_29 >= (params_0.chunk_count_0))
    {
        _S388 = true;
    }
    else
    {
        _S388 = (crush_0.enabled_0) == u32(0);
    }
    if(_S388)
    {
        return;
    }
    var cs_0 : vec4<f32> = crush_rw_0[k_29];
    var flags_4 : u32 = (bitcast<u32>((crush_rw_0[k_29].w)));
    if(((flags_4 & (u32(2)))) == u32(0))
    {
        return;
    }
    crush_rw_0[k_29] = vec4<f32>(cs_0.xyz, (bitcast<f32>(((flags_4 & (u32(4294967293)))))));
    var slot_19 : u32 = body_of_0(k_29);
    var bw_1 : u32 = u32(7) * slot_19;
    var b0_0 : vec4<f32> = bodies_0[bw_1];
    var orientation_2 : Quat_0 = quat_of_0(bodies_0[bw_1 + u32(1)]);
    var v_14 : vec3<f32> = bodies_0[bw_1 + u32(3)].xyz;
    var w_14 : vec3<f32> = bodies_0[bw_1 + u32(4)].xyz;
    var _S389 : u32 = u32(4) * k_29;
    var mass_1 : f32 = chunks_0[_S389].w;
    var volume_1 : f32 = crush_tab_0[u32(2) * k_29 + u32(1)].y;
    var rank_0 : u32 = work_0[w_rank_0() + k_29];
    if(work_0[w_rank_0() + k_29] >= (crush_0.max_events_0))
    {
        var _S390 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(32));
    }
    var _S391 : vec3<f32> = vec3<f32>(0.0f);
    if((crush_0.mode_1) == u32(1))
    {
        _S388 = (crush_0.fragments_0) > u32(0);
    }
    else
    {
        _S388 = false;
    }
    var first_6 : u32;
    var half_1 : vec3<f32>;
    if(_S388)
    {
        var _S392 : u32 = a_counters_0();
        var _S393 : u32 = atomicLoad(&(atom_0[_S392 + u32(8)]));
        var base_4 : u32 = _S393 + rank_0 * crush_0.fragments_0;
        if((base_4 + crush_0.fragments_0) > (crush_0.pool_size_0))
        {
            var _S394 : u32 = atomicOr(&(atom_0[_S392 + u32(3)]), u32(16));
            first_6 = u32(4294967295);
        }
        else
        {
            first_6 = base_4;
        }
        var q_33 : Quat_0;
        var origin_16 : vec3<f32>;
        structure_pose_4(slot_19, bw_1, &(q_33), &(origin_16));
        var _S395 : vec3<f32> = origin_16 + rotate_0(q_33, chunks_0[_S389].xyz);
        var _S396 : u32 = u32(3) * k_29;
        var box_0 : Quat_0 = normalized_0(quat_mul_0(q_33, quat_of_0(geometry_0[_S396 + u32(2)])));
        var h_1 : vec3<f32> = geometry_0[_S396 + u32(1)].xyz;
        var g_0 : vec3<u32> = vec3<u32>(vec3<i32>(i32(1)));
        var cell_0 : vec3<f32> = h_1;
        var n_5 : u32 = crush_0.fragments_0;
        loop
        {
            if(n_5 > u32(1))
            {
            }
            else
            {
                break;
            }
            if((cell_0.x) >= (cell_0.y))
            {
                _S388 = (cell_0.x) >= (cell_0.z);
            }
            else
            {
                _S388 = false;
            }
            if(_S388)
            {
                ax_4 = u32(0);
            }
            else
            {
                if((cell_0.y) >= (cell_0.z))
                {
                    ax_4 = u32(1);
                }
                else
                {
                    ax_4 = u32(2);
                }
            }
            g_0[ax_4] = g_0[ax_4] * u32(2);
            cell_0[ax_4] = cell_0[ax_4] * 0.5f;
            n_5 = (n_5 >> (u32(1)));
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
        var _S397 : f32 = half_2.y;
        var _S398 : f32 = _S397 * _S397;
        var _S399 : f32 = half_2.z;
        var _S400 : f32 = _S399 * _S399;
        var _S401 : f32 = half_2.x;
        var _S402 : f32 = _S401 * _S401;
        var inertia_0 : vec3<f32> = vec3<f32>((mf_0 / 3.0f)) * vec3<f32>(_S398 + _S400, _S402 + _S400, _S402 + _S398);
        var _S403 : vec3<f32> = vec3<f32>(1.0f / inertia_0.x, 1.0f / inertia_0.y, 1.0f / inertia_0.z);
        var _S404 : Quat_0 = rot_to_rubble_0(box_0);
        var _S405 : f32 = chunks_0[_S389 + u32(3)].x;
        var _S406 : f32 = min(_S401, min(_S397, _S399));
        ax_4 = u32(0);
        loop
        {
            if(ax_4 < (crush_0.fragments_0))
            {
                _S388 = first_6 != u32(4294967295);
            }
            else
            {
                _S388 = false;
            }
            if(_S388)
            {
            }
            else
            {
                break;
            }
            var _S407 : u32 = ax_4 % g_0.x;
            var _S408 : u32 = ax_4 / g_0.x;
            var _S409 : u32 = _S408 % g_0.y;
            var _S410 : u32 = ax_4 / (g_0.x * g_0.y);
            var x_8 : vec3<f32> = to_rubble_0(_S395 + rotate_0(box_0, (vec3<f32>(0) - h_1) + cell_0 * (vec3<f32>(2.0f) * vec3<f32>(vec3<u32>(_S407, _S409, _S410)) + vec3<f32>(1.0f))));
            var entry_1 : vec4<f32> = crush_tab_0[crush_0.t_pool_0 + first_6 + ax_4];
            var fs_0 : u32 = (bitcast<u32>((entry_1.x)));
            var fc_0 : u32 = (bitcast<u32>((entry_1.y)));
            write_body_0(fs_0, x_8, 1.0f / mf_0, _S404, _S403, u32(9), v_14 + cross(w_14, x_8 - b0_0.xyz), w_14);
            var _S411 : u32 = u32(4) * fc_0;
            var _S412 : u32 = _S411 + u32(1);
            colliders_0[_S412] = vec4<f32>(0.0f, 0.0f, 0.0f, colliders_0[_S412].w);
            colliders_0[_S411 + u32(2)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
            colliders_0[_S411 + u32(3)] = vec4<f32>(half_2, length(half_2));
            var _S413 : u32 = u32(2) * fc_0;
            collider_info_0[_S413] = vec4<u32>(u32(4294967295), fs_0, u32(2), collider_info_0[_S413].w);
            collider_info_0[_S413 + u32(1)] = vec4<u32>((bitcast<u32>((mf_0))), (bitcast<u32>((_S405))), (bitcast<u32>((_S406))), u32(0));
            ax_4 = ax_4 + u32(1);
        }
        half_1 = half_2;
    }
    else
    {
        first_6 = u32(4294967295);
        half_1 = _S391;
    }
    if(rank_0 < (crush_0.max_events_0))
    {
        var ev_0 : u32 = crush_0.r_events_0 + u32(6) * rank_0;
        crush_rw_0[ev_0] = vec4<f32>((bitcast<f32>((k_29))), (bitcast<f32>((first_6))), (bitcast<f32>(((flags_4 >> (u32(4)))))), cs_0.x);
        crush_rw_0[ev_0 + u32(1)] = vec4<f32>(b0_0.xyz, mass_1);
        crush_rw_0[ev_0 + u32(2)] = quat_vec_0(orientation_2);
        crush_rw_0[ev_0 + u32(3)] = vec4<f32>(v_14, cs_0.y);
        crush_rw_0[ev_0 + u32(4)] = vec4<f32>(w_14, cs_0.z);
        crush_rw_0[ev_0 + u32(5)] = vec4<f32>(half_1, volume_1);
    }
    write_body_0(slot_19, vec3<f32>(f32(slot_19 % u32(1024)) * crush_0.park_step_0, crush_0.park_y_0, f32(slot_19 / u32(1024)) * crush_0.park_step_0), 0.0f, orientation_2, _S391, u32(0), _S391, _S391);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_crush_end(@builtin(global_invocation_id) id_40 : vec3<u32>)
{
    var total_3 : u32 = work_0[scan_new_blocks_0() + (params_0.chunk_count_0 + u32(1023)) / u32(1024)];
    var _S414 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S414 + u32(7)]), work_0[scan_new_blocks_0() + (params_0.chunk_count_0 + u32(1023)) / u32(1024)]);
    var _S415 : u32 = atomicAdd(&(atom_0[_S414 + u32(9)]), total_3);
    if((crush_0.mode_1) == u32(1))
    {
        var _S416 : u32 = atomicAdd(&(atom_0[_S414 + u32(8)]), total_3 * crush_0.fragments_0);
    }
    return;
}

