struct FractureParams_std140_0
{
    @align(16) chunk_count_0 : u32,
    @align(4) bond_count_0 : u32,
    @align(8) body_count_0 : u32,
    @align(4) max_bodies_0 : u32,
    @align(16) step_0 : u32,
    @align(4) pulse_cap_0 : u32,
    @align(8) impact_peak_0 : u32,
    @align(4) pad0_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) band_0 : f32,
    @align(8) scale_m_0 : f32,
    @align(4) scale_s_0 : f32,
    @align(16) scale_j_0 : f32,
    @align(4) pad1_0 : f32,
    @align(8) pad2_0 : f32,
    @align(4) pad3_0 : f32,
};

@binding(0) @group(0) var<uniform> params_0 : FractureParams_std140_0;
@binding(4) @group(0) var<storage, read_write> work_0 : array<u32>;

@binding(6) @group(0) var<storage, read_write> frames_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read_write> bodies_0 : array<vec4<f32>>;

@binding(1) @group(0) var<storage, read> chunks_0 : array<vec4<f32>>;

struct Record_std430_0
{
    @align(16) ids_0 : vec4<u32>,
    @align(16) point_0 : vec4<f32>,
    @align(16) impulse_0 : vec4<f32>,
    @align(16) other_0 : vec4<f32>,
    @align(16) key_0 : vec4<u32>,
};

@binding(10) @group(0) var<storage, read> manifolds_0 : array<Record_std430_0>;

@binding(11) @group(0) var<storage, read> sort_0 : array<u32>;

@binding(12) @group(0) var<storage, read_write> loads_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read_write> atom_0 : array<atomic<u32>>;

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
    @align(4) shot_count_0 : u32,
    @align(8) impactor_base_0 : u32,
    @align(4) pad0_1 : u32,
    @align(16) rest_speed_0 : f32,
    @align(4) pad1_1 : f32,
    @align(8) pad2_1 : f32,
    @align(4) pad3_1 : f32,
};

@binding(16) @group(0) var<uniform> view_0 : ViewParams_std140_0;
@binding(18) @group(0) var<storage, read_write> objects_0 : array<vec4<f32>>;

@binding(20) @group(0) var<storage, read> shot_bodies_0 : array<u32>;

@binding(19) @group(0) var<storage, read_write> lines_0 : array<f32>;

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

fn l_pulse_base_0() -> u32
{
    return params_0.chunk_count_0;
}

fn l_pulse_count_base_0() -> u32
{
    return params_0.chunk_count_0 + u32(3) * params_0.pulse_cap_0 * params_0.chunk_count_0;
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

@compute
@workgroup_size(64, 1, 1)
fn fr_loads(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var f_total_0 : vec3<f32>;
    var m_1 : f32;
    var overflow_0 : bool;
    var overflow_1 : bool;
    var p_0 : u32;
    var kept_0 : u32;
    var q3_0 : u32;
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
    var _S17 : u32 = params_0.chunk_count_0;
    var _S18 : f32 = c0_0.w;
    var _S19 : vec4<f32> = chunks_0[_S16 + u32(3)];
    var _S20 : f32 = params_0.band_0;
    var _S21 : f32 = params_0.dt_0;
    var _S22 : bool = (params_0.impact_peak_0) != u32(0);
    var _S23 : u32 = l_pulse_base_0();
    var _S24 : u32 = l_pulse_count_base_0();
    var _S25 : u32 = params_0.pulse_cap_0;
    loop
    {
        var _S26 : vec3<f32> = vec3<f32>(0.0f);
        var _S27 : u32 = sort_0[k_0];
        var _S28 : u32 = sort_0[_S17 + u32(1) + k_0];
        var first_0 : bool = true;
        var last_0 : u32 = u32(0);
        var step_1 : u32 = u32(0);
        f_total_0 = _S26;
        var t_total_1 : vec3<f32> = _S26;
        loop
        {
            if(step_1 < _S28)
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
                if(q3_0 < _S28)
                {
                }
                else
                {
                    break;
                }
                var key_1 : u32 = manifolds_0[_S27 + q3_0].ids_0.y + u32(1);
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
            var r_4 : Record_std430_0 = manifolds_0[_S27 + p_0];
            var _S29 : vec3<f32> = r_4.impulse_0.xyz;
            var j_0 : f32 = length(_S29);
            if(j_0 == 0.0f)
            {
                var _S30 : u32 = step_1 + u32(1);
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S30;
                continue;
            }
            var m_other_0 : f32 = r_4.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_0))
                {
                    m_1 = _S18;
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
            var _S31 : f32 = min(j_0, 2.0f * m_1 * max(r_4.impulse_0.w - _S20, 0.0f));
            if(_S31 < j_0)
            {
                var f_0 : vec3<f32> = _S29 / vec3<f32>(j_0) * vec3<f32>(((j_0 - _S31) / _S21));
                var _S32 : vec3<f32> = t_total_1 + cross(r_4.point_0.xyz - centre_0, f_0);
                f_total_1 = f_total_0 + f_0;
                t_total_0 = _S32;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            var _S30 : u32 = step_1 + u32(1);
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S30;
        }
        var _S33 : u32 = _S23 + u32(3) * _S25 * k_0;
        var _S34 : u32 = _S24 + k_0;
        var _S35 : u32 = (bitcast<u32>((loads_0[_S34].x)));
        p_0 = u32(0);
        kept_0 = u32(0);
        loop
        {
            if(p_0 < _S35)
            {
            }
            else
            {
                break;
            }
            var _S36 : u32 = _S33 + u32(3) * p_0;
            var a_1 : vec4<f32> = loads_0[_S36];
            var b_3 : vec4<f32> = loads_0[_S36 + u32(1)];
            var c_2 : vec4<f32> = loads_0[_S36 + u32(2)];
            var _S37 : f32 = loads_0[_S36 + u32(1)].w;
            var mean_0 : f32 = pulse_mean_0(loads_0[_S36].x, loads_0[_S36].y, loads_0[_S36].z, loads_0[_S36].w, _S37, _S21);
            if(mean_0 != 0.0f)
            {
                var f_1 : vec3<f32> = c_2.xyz * vec3<f32>(mean_0);
                var _S38 : vec3<f32> = t_total_1 + cross(b_3.xyz - centre_0, f_1);
                f_total_0 = f_total_0 + f_1;
                t_total_1 = _S38;
            }
            if((a_1.x + a_1.z + _S37) > _S21)
            {
                a_1[i32(0)] = a_1[i32(0)] - _S21;
                var _S39 : u32 = _S33 + u32(3) * kept_0;
                loads_0[_S39] = a_1;
                loads_0[_S39 + u32(1)] = b_3;
                loads_0[_S39 + u32(2)] = c_2;
                kept_0 = kept_0 + u32(1);
            }
            p_0 = p_0 + u32(1);
        }
        q3_0 = u32(0);
        overflow_0 = false;
        loop
        {
            if(q3_0 < _S28)
            {
            }
            else
            {
                break;
            }
            var r_5 : Record_std430_0 = manifolds_0[_S27 + q3_0];
            var _S40 : vec3<f32> = r_5.impulse_0.xyz;
            var j_1 : f32 = length(_S40);
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
                    m_1 = _S18;
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
            var _S41 : f32 = r_5.impulse_0.w;
            var _S42 : f32 = min(j_1, 2.0f * m_1 * max(_S41 - _S20, 0.0f));
            if(_S42 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var _S43 : f32 = _S19.y;
            var duration_1 : f32 = hertz_duration_0(m_1, min(r_5.other_0.z, _S19.z), 1.0f / ((1.0f - _S43 * _S43) / _S19.x + 1.0f / r_5.other_0.y), _S41);
            var peak_1 : f32 = 3.14159274101257324f * _S42 / (2.0f * duration_1);
            var dir_0 : vec3<f32> = _S40 / vec3<f32>(j_1);
            var mean_1 : f32;
            if(_S22)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S21);
            }
            if(mean_1 != 0.0f)
            {
                var f_2 : vec3<f32> = dir_0 * vec3<f32>(mean_1);
                var _S44 : vec3<f32> = t_total_1 + cross(r_5.point_0.xyz - centre_0, f_2);
                f_total_1 = f_total_0 + f_2;
                t_total_0 = _S44;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            var kept_1 : u32;
            if(duration_1 > _S21)
            {
                if(kept_0 < _S25)
                {
                    var _S45 : u32 = _S33 + u32(3) * kept_0;
                    loads_0[_S45] = vec4<f32>(- _S21, 0.0f, 0.0f, peak_1);
                    loads_0[_S45 + u32(1)] = vec4<f32>(r_5.point_0.xyz, duration_1);
                    loads_0[_S45 + u32(2)] = vec4<f32>(dir_0, 0.0f);
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
            kept_0 = kept_1;
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            q3_0 = q3_0 + u32(1);
            overflow_0 = overflow_1;
        }
        loads_0[_S34] = vec4<f32>((bitcast<f32>((kept_0))), 0.0f, 0.0f, 0.0f);
        break;
    }
    if(overflow_0)
    {
        var _S46 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(8));
    }
    loads_0[k_0] = vec4<f32>(length(f_total_0), 0.0f, 0.0f, 0.0f);
    return;
}

fn w_alive_0() -> u32
{
    return params_0.chunk_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_breaks(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var i_0 : u32 = id_1.x;
    var _S47 : bool;
    if(i_0 >= (params_0.bond_count_0))
    {
        _S47 = true;
    }
    else
    {
        _S47 = work_0[w_alive_0() + i_0] == u32(0);
    }
    if(_S47)
    {
        return;
    }
    var bond_0 : vec4<u32> = bonds_0[i_0];
    var _S48 : u32 = bond_0.x;
    if((max(loads_0[_S48].x, loads_0[bond_0.y].x)) > (bitcast<f32>((bond_0.z))))
    {
        work_0[w_alive_0() + i_0] = u32(0);
        var _S49 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S48)]), u32(1));
        var _S50 : u32 = a_counters_0();
        var _S51 : u32 = atomicAdd(&(atom_0[_S50]), u32(1));
        var _S52 : u32 = atomicAdd(&(atom_0[_S50 + u32(1)]), u32(1));
    }
    return;
}

fn groups_of_0( n_2 : u32,  size_0 : u32) -> u32
{
    var _S53 : u32 = (n_2 + size_0 - u32(1)) / size_0;
    return _S53;
}

fn set_args_0( at_0 : u32,  x_2 : u32)
{
    var _S54 : u32 = u32(3) * at_0;
    args_0[_S54] = x_2;
    args_0[_S54 + u32(1)] = u32(1);
    args_0[_S54 + u32(2)] = u32(1);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_gate(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var _S55 : u32 = a_counters_0();
    var _S56 : u32 = atomicLoad(&(atom_0[_S55]));
    var run_0 : bool = _S56 != u32(0);
    atomicStore(&(atom_0[_S55]), u32(0));
    if(run_0)
    {
        var _S57 : u32 = atomicAdd(&(atom_0[_S55 + u32(4)]), u32(1));
    }
    var _S58 : u32;
    if(run_0)
    {
        var _S59 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S58 = _S59;
    }
    else
    {
        _S58 = u32(0);
    }
    set_args_0(u32(0), _S58);
    if(run_0)
    {
        var _S60 : u32 = groups_of_0(params_0.bond_count_0, u32(64));
        _S58 = _S60;
    }
    else
    {
        _S58 = u32(0);
    }
    set_args_0(u32(1), _S58);
    if(run_0)
    {
        var _S61 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S58 = _S61;
    }
    else
    {
        _S58 = u32(0);
    }
    set_args_0(u32(2), _S58);
    if(run_0)
    {
        var _S62 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S58 = _S62;
    }
    else
    {
        _S58 = u32(0);
    }
    set_args_0(u32(3), _S58);
    if(run_0)
    {
        var _S63 : u32 = groups_of_0(params_0.body_count_0, u32(1024));
        _S58 = _S63;
    }
    else
    {
        _S58 = u32(0);
    }
    set_args_0(u32(4), _S58);
    var _S64 : i32;
    if(run_0)
    {
        _S64 = i32(1);
    }
    else
    {
        _S64 = i32(0);
    }
    set_args_0(u32(5), u32(_S64));
    return;
}

fn dirty_0( body_0 : u32) -> bool
{
    var _S65 : u32 = atomicLoad(&(atom_0[a_dirty_0() + body_0]));
    return _S65 != u32(0);
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
    var _S66 : bool;
    if(c_3 >= (params_0.chunk_count_0))
    {
        _S66 = true;
    }
    else
    {
        var _S67 : bool = dirty_0(body_of_0(c_3));
        _S66 = !_S67;
    }
    if(_S66)
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
    var _S68 : bool;
    if(s_0 >= (params_0.body_count_0))
    {
        _S68 = true;
    }
    else
    {
        var _S69 : bool = dirty_0(s_0);
        _S68 = !_S69;
    }
    if(_S68)
    {
        return;
    }
    atomicStore(&(atom_0[a_best_count_0() + s_0]), u32(0));
    atomicStore(&(atom_0[a_best_root_0() + s_0]), u32(4294967295));
    return;
}

fn find_0( x_3 : u32) -> u32
{
    var _S70 : u32 = a_parent_0();
    var _S71 : u32 = atomicLoad(&(atom_0[_S70 + x_3]));
    var p_1 : u32 = _S71;
    var _S72 : u32 = x_3;
    var guard_0 : u32 = u32(0);
    loop
    {
        var _S73 : bool;
        if(guard_0 < u32(1048576))
        {
            _S73 = p_1 != _S72;
        }
        else
        {
            _S73 = false;
        }
        if(_S73)
        {
        }
        else
        {
            break;
        }
        var gp_0 : u32 = atomicLoad(&(atom_0[_S70 + p_1]));
        if(gp_0 != p_1)
        {
            var _S74 : u32 = atomicCompareExchangeWeak(&(atom_0[_S70 + _S72]), p_1, gp_0).old_value;
        }
        var _S75 : u32 = guard_0 + u32(1);
        _S72 = p_1;
        p_1 = gp_0;
        guard_0 = _S75;
    }
    return _S72;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_union(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var i_1 : u32 = id_5.x;
    var _S76 : bool;
    if(i_1 >= (params_0.bond_count_0))
    {
        _S76 = true;
    }
    else
    {
        _S76 = work_0[w_alive_0() + i_1] == u32(0);
    }
    if(_S76)
    {
        return;
    }
    var bond_1 : vec4<u32> = bonds_0[i_1];
    var _S77 : u32 = bond_1.x;
    var _S78 : bool = dirty_0(body_of_0(_S77));
    if(!_S78)
    {
        return;
    }
    var _S79 : u32 = bond_1.y;
    var a_2 : u32 = _S77;
    var b_4 : u32 = _S79;
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
        var _S80 : u32 = max(ra_0, rb_0);
        var _S81 : u32 = min(ra_0, rb_0);
        var was_0 : u32 = atomicCompareExchangeWeak(&(atom_0[a_parent_0() + _S80]), _S80, _S81).old_value;
        if(was_0 == _S80)
        {
            return;
        }
        var _S82 : u32 = guard_1 + u32(1);
        a_2 = _S80;
        b_4 = _S81;
        guard_1 = _S82;
    }
    var _S83 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(4));
    return;
}

fn w_label_0() -> u32
{
    return w_alive_0() + params_0.bond_count_0;
}

fn add64_0( at_1 : u32,  value_0 : f32,  scale_0 : f32)
{
    var x_4 : f32 = round(value_0 * scale_0);
    var x_5 : f32;
    if(!((abs(x_4)) < 1.073741824e+09f))
    {
        var _S84 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
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
    var _S85 : u32;
    if(v_3 < i32(0))
    {
        _S85 = u32(4294967295);
    }
    else
    {
        _S85 = u32(0);
    }
    var hi_0 : u32 = _S85 + carry_0;
    if(hi_0 != u32(0))
    {
        var _S86 : u32 = atomicAdd(&(atom_0[at_1 + u32(1)]), hi_0);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_flatten(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var c_4 : u32 = id_6.x;
    var _S87 : bool;
    if(c_4 >= (params_0.chunk_count_0))
    {
        _S87 = true;
    }
    else
    {
        var _S88 : bool = dirty_0(body_of_0(c_4));
        _S87 = !_S88;
    }
    if(_S87)
    {
        return;
    }
    var r_6 : u32 = find_0(c_4);
    work_0[w_label_0() + c_4] = r_6;
    var _S89 : u32 = atomicAdd(&(atom_0[a_count_0() + r_6]), u32(1));
    if(((info_0[c_4] & (u32(1)))) != u32(0))
    {
        var _S90 : u32 = atomicOr(&(atom_0[a_anchor_0() + r_6]), u32(1));
    }
    var _S91 : u32 = u32(4) * c_4;
    var c0_1 : vec4<f32> = chunks_0[_S91];
    var c1_0 : vec4<f32> = chunks_0[_S91 + u32(1)];
    var c2_0 : vec4<f32> = chunks_0[_S91 + u32(2)];
    var m_2 : f32 = c0_1.w;
    var d_0 : vec3<f32> = c0_1.xyz - chunks_0[u32(4) * r_6].xyz;
    var dd_0 : f32 = dot(d_0, d_0);
    var at_2 : u32 = a_sums_0() + u32(20) * r_6;
    add64_0(at_2, m_2, params_0.scale_m_0);
    var _S92 : f32 = d_0.x;
    var _S93 : f32 = m_2 * _S92;
    add64_0(at_2 + u32(2), _S93, params_0.scale_s_0);
    var _S94 : f32 = d_0.y;
    var _S95 : f32 = m_2 * _S94;
    add64_0(at_2 + u32(4), _S95, params_0.scale_s_0);
    var _S96 : f32 = d_0.z;
    add64_0(at_2 + u32(6), m_2 * _S96, params_0.scale_s_0);
    add64_0(at_2 + u32(8), c1_0.x + m_2 * (dd_0 - _S92 * _S92), params_0.scale_j_0);
    add64_0(at_2 + u32(10), c1_0.y + m_2 * (dd_0 - _S94 * _S94), params_0.scale_j_0);
    add64_0(at_2 + u32(12), c1_0.z + m_2 * (dd_0 - _S96 * _S96), params_0.scale_j_0);
    add64_0(at_2 + u32(14), c1_0.w - _S93 * _S94, params_0.scale_j_0);
    add64_0(at_2 + u32(16), c2_0.x - _S93 * _S96, params_0.scale_j_0);
    add64_0(at_2 + u32(18), c2_0.y - _S95 * _S96, params_0.scale_j_0);
    return;
}

fn is_root_0( c_5 : u32) -> bool
{
    var _S97 : bool;
    if(c_5 < (params_0.chunk_count_0))
    {
        var _S98 : bool = dirty_0(body_of_0(c_5));
        _S97 = _S98;
    }
    else
    {
        _S97 = false;
    }
    if(_S97)
    {
        _S97 = work_0[w_label_0() + c_5] == c_5;
    }
    else
    {
        _S97 = false;
    }
    return _S97;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_count(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var c_6 : u32 = id_7.x;
    var _S99 : bool = is_root_0(c_6);
    if(!_S99)
    {
        return;
    }
    var _S100 : u32 = atomicLoad(&(atom_0[a_count_0() + c_6]));
    var _S101 : u32 = atomicMax(&(atom_0[a_best_count_0() + body_of_0(c_6)]), _S100);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_root(@builtin(global_invocation_id) id_8 : vec3<u32>)
{
    var c_7 : u32 = id_8.x;
    var _S102 : bool = is_root_0(c_7);
    if(!_S102)
    {
        return;
    }
    var s_1 : u32 = body_of_0(c_7);
    var _S103 : u32 = atomicLoad(&(atom_0[a_count_0() + c_7]));
    var _S104 : u32 = atomicLoad(&(atom_0[a_best_count_0() + s_1]));
    if(_S103 == _S104)
    {
        var _S105 : u32 = atomicMin(&(atom_0[a_best_root_0() + s_1]), c_7);
    }
    return;
}

fn w_slot_0() -> u32
{
    return w_label_0() + params_0.chunk_count_0;
}

fn w_new_0() -> u32
{
    return w_slot_0() + params_0.chunk_count_0;
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
    var _S106 : bool = is_root_0(c_8);
    var _S107 : bool;
    if(_S106)
    {
        var _S108 : u32 = atomicLoad(&(atom_0[a_best_root_0() + body_of_0(c_8)]));
        _S107 = _S108 != c_8;
    }
    else
    {
        _S107 = false;
    }
    var _S109 : u32;
    if(_S107)
    {
        _S109 = u32(1);
    }
    else
    {
        _S109 = u32(0);
    }
    work_0[w_new_0() + c_8] = _S109;
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
    var _S110 : u32 = group_0.x;
    var _S111 : u32 = u32(1024) * _S110 + u32(4) * t_1;
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
        if((_S111 + j_2) < (scan_0.n_0))
        {
            step_2 = work_0[scan_0.in_base_0 + _S111 + j_2];
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
    var _S112 : u32 = g_scan_0[t_1] - sum_2;
    j_2 = u32(0);
    add_0 = _S112;
    loop
    {
        if(j_2 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S111 + j_2) < (scan_0.n_0))
        {
            work_0[scan_0.out_base_0 + _S111 + j_2] = add_0;
        }
        var run_1 : u32 = add_0 + v_4[j_2];
        j_2 = j_2 + u32(1);
        add_0 = run_1;
    }
    if(t_1 == u32(255))
    {
        work_0[scan_0.block_base_0 + _S110] = g_scan_0[i32(255)];
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn fr_scan_top(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var t_2 : u32 = thread_1.x;
    var blocks_0 : u32 = (scan_0.n_0 + u32(1023)) / u32(1024);
    var _S113 : bool = t_2 < blocks_0;
    var v_5 : u32;
    if(_S113)
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
    if(_S113)
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

@compute
@workgroup_size(64, 1, 1)
fn fr_freelist(@builtin(global_invocation_id) id_11 : vec3<u32>)
{
    var s_2 : u32 = id_11.x;
    var _S114 : bool;
    if(s_2 >= (params_0.body_count_0))
    {
        _S114 = true;
    }
    else
    {
        _S114 = work_0[w_free_0() + s_2] == u32(0);
    }
    if(_S114)
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
    var _S115 : bool = is_root_0(c_9);
    if(!_S115)
    {
        return;
    }
    var s_3 : u32 = body_of_0(c_9);
    var _S116 : u32 = atomicLoad(&(atom_0[a_best_root_0() + s_3]));
    var slot_3 : u32;
    if(_S116 != c_9)
    {
        var k_1 : u32 = work_0[w_rank_0() + c_9];
        if(work_0[w_rank_0() + c_9] < (free_total_0()))
        {
            var slot_4 : u32 = work_0[w_free_list_0() + k_1];
            work_0[w_free_0() + work_0[w_free_list_0() + k_1]] = u32(0);
            var _S117 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(2)]), u32(1));
            slot_3 = slot_4;
        }
        else
        {
            var _S118 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(1));
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
    var _S119 : u32 = atomicLoad(&(atom_0[at_3 + u32(1)]));
    var hi_1 : i32 = (bitcast<i32>((_S119)));
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
    var _S120 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S121 : array<f32, i32(3)> = array<f32, i32(3)>( r1_0.x, r1_0.y, r1_0.z );
    var _S122 : array<f32, i32(3)> = array<f32, i32(3)>( r2_0.x, r2_0.y, r2_0.z );
    var a_3 : array<array<f32, i32(3)>, i32(3)>;
    a_3[i32(0)] = _S120;
    a_3[i32(1)] = _S121;
    a_3[i32(2)] = _S122;
    var _S123 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S124 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S125 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_6 : array<array<f32, i32(3)>, i32(3)>;
    v_6[i32(0)] = _S123;
    v_6[i32(1)] = _S124;
    v_6[i32(2)] = _S125;
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
            var _S126 : bool;
            if(k_2 != p_2)
            {
                _S126 = k_2 != q_10;
            }
            else
            {
                _S126 = false;
            }
            if(_S126)
            {
                var kp_0 : f32 = a_3[k_2][p_2];
                var kq_0 : f32 = a_3[k_2][q_10];
                var _S127 : f32 = c_10 * a_3[k_2][p_2] - s_4 * a_3[k_2][q_10];
                a_3[k_2][p_2] = _S127;
                a_3[p_2][k_2] = _S127;
                var _S128 : f32 = s_4 * kp_0 + c_10 * kq_0;
                a_3[k_2][q_10] = _S128;
                a_3[q_10][k_2] = _S128;
            }
            k_2 = k_2 + u32(1);
        }
        var _S129 : f32 = c_10 * c_10;
        var _S130 : f32 = 2.0f * s_4 * c_10 * apq_0;
        var _S131 : f32 = s_4 * s_4;
        a_3[p_2][p_2] = _S129 * app_0 - _S130 + _S131 * aqq_0;
        a_3[q_10][q_10] = _S131 * app_0 + _S130 + _S129 * aqq_0;
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
    var _S132 : bool = is_root_0(c_12);
    if(!_S132)
    {
        return;
    }
    var parent_0 : u32 = body_of_0(c_12);
    var pb_0 : u32 = u32(7) * parent_0;
    var at_4 : u32 = a_sums_0() + u32(20) * c_12;
    var mass_0 : f32 = read64_0(at_4, params_0.scale_m_0);
    var _S133 : f32 = read64_0(at_4 + u32(2), params_0.scale_s_0);
    var _S134 : f32 = read64_0(at_4 + u32(4), params_0.scale_s_0);
    var _S135 : f32 = read64_0(at_4 + u32(6), params_0.scale_s_0);
    var d_1 : vec3<f32> = vec3<f32>(_S133, _S134, _S135) / vec3<f32>(max(mass_0, 1.00000000317107685e-30f));
    var jxx_0 : f32 = read64_0(at_4 + u32(8), params_0.scale_j_0);
    var jyy_0 : f32 = read64_0(at_4 + u32(10), params_0.scale_j_0);
    var jzz_0 : f32 = read64_0(at_4 + u32(12), params_0.scale_j_0);
    var jxy_0 : f32 = read64_0(at_4 + u32(14), params_0.scale_j_0);
    var jxz_0 : f32 = read64_0(at_4 + u32(16), params_0.scale_j_0);
    var jyz_0 : f32 = read64_0(at_4 + u32(18), params_0.scale_j_0);
    var dd_1 : f32 = dot(d_1, d_1);
    var _S136 : f32 = d_1.x;
    var _S137 : f32 = mass_0 * _S136;
    var _S138 : f32 = d_1.y;
    var _S139 : f32 = jxy_0 + _S137 * _S138;
    var _S140 : f32 = d_1.z;
    var _S141 : f32 = jxz_0 + _S137 * _S140;
    var _S142 : f32 = jyz_0 + mass_0 * _S138 * _S140;
    var com_2 : vec3<f32> = chunks_0[u32(4) * c_12].xyz + d_1;
    var principal_0 : Quat_0;
    var moments_1 : vec3<f32>;
    principal_axes_0(vec3<f32>(jxx_0 - mass_0 * (dd_1 - _S136 * _S136), _S139, _S141), vec3<f32>(_S139, jyy_0 - mass_0 * (dd_1 - _S138 * _S138), _S142), vec3<f32>(_S141, _S142, jzz_0 - mass_0 * (dd_1 - _S140 * _S140)), &(principal_0), &(moments_1));
    var _S143 : u32 = atomicLoad(&(atom_0[a_anchor_0() + c_12]));
    var anchored_0 : bool = _S143 != u32(0);
    var q_13 : Quat_0;
    var origin_3 : vec3<f32>;
    structure_pose_1(parent_0, pb_0, &(q_13), &(origin_3));
    var p_parent_0 : vec3<f32> = bodies_0[pb_0].xyz;
    var v_parent_0 : vec3<f32> = bodies_0[pb_0 + u32(3)].xyz;
    var w_parent_0 : vec3<f32> = bodies_0[pb_0 + u32(4)].xyz;
    var pos_0 : vec3<f32> = to_rubble_0(origin_3 + rotate_0(q_13, com_2));
    var orientation_0 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(q_13, principal_0)));
    var _S144 : bool;
    if(anchored_0)
    {
        _S144 = true;
    }
    else
    {
        _S144 = !(mass_0 > 0.0f);
    }
    var inv_mass_1 : f32;
    if(_S144)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / mass_0;
    }
    var _S145 : vec3<f32> = vec3<f32>(0.0f);
    var inv_inertia_0 : vec3<f32>;
    if(!anchored_0)
    {
        var _S146 : f32;
        if((moments_1.x) > 0.0f)
        {
            _S146 = 1.0f / moments_1.x;
        }
        else
        {
            _S146 = 0.0f;
        }
        var _S147 : f32;
        if((moments_1.y) > 0.0f)
        {
            _S147 = 1.0f / moments_1.y;
        }
        else
        {
            _S147 = 0.0f;
        }
        var _S148 : f32;
        if((moments_1.z) > 0.0f)
        {
            _S148 = 1.0f / moments_1.z;
        }
        else
        {
            _S148 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S146, _S147, _S148);
    }
    else
    {
        inv_inertia_0 = _S145;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_0)
    {
        velocity_0 = _S145;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    var spin_0 : vec3<f32>;
    if(anchored_0)
    {
        spin_0 = _S145;
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
    var _S149 : bool = is_root_0(c_13);
    if(!_S149)
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
    var _S150 : u32 = u32(2) * slot_5;
    frames_0[_S150] = frames_0[st_1 + u32(7)];
    frames_0[_S150 + u32(1)] = frames_0[st_1 + u32(8)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_reparent(@builtin(global_invocation_id) id_15 : vec3<u32>)
{
    var c_14 : u32 = id_15.x;
    var _S151 : bool;
    if(c_14 >= (params_0.chunk_count_0))
    {
        _S151 = true;
    }
    else
    {
        var _S152 : bool = dirty_0(body_of_0(c_14));
        _S151 = !_S152;
    }
    if(_S151)
    {
        return;
    }
    var r_7 : u32 = work_0[w_label_0() + c_14];
    var slot_6 : u32 = work_0[w_slot_0() + work_0[w_label_0() + c_14]];
    work_0[w_body_0() + c_14] = work_0[w_slot_0() + work_0[w_label_0() + c_14]];
    var _S153 : u32 = u32(2) * slot_6;
    var _S154 : vec3<f32> = frames_0[_S153].xyz;
    var _S155 : Quat_0 = conj_0(quat_of_0(frames_0[_S153 + u32(1)]));
    var inv_mass_2 : f32 = bodies_0[u32(7) * slot_6].w;
    var _S156 : f32;
    if(inv_mass_2 > 0.0f)
    {
        _S156 = 1.0f / inv_mass_2;
    }
    else
    {
        _S156 = float_bits_0(u32(2139095040));
    }
    var c2_2 : vec4<f32> = chunks_0[u32(4) * c_14 + u32(2)];
    var first_1 : u32 = (bitcast<u32>((c2_2.z)));
    var _S157 : u32 = (bitcast<u32>((c2_2.w)));
    var _S158 : vec4<f32> = chunks_0[u32(4) * r_7 + u32(3)];
    var p_3 : u32 = first_1;
    loop
    {
        if(p_3 < (first_1 + _S157))
        {
        }
        else
        {
            break;
        }
        var _S159 : u32 = u32(2) * p_3;
        var o_1 : vec4<f32> = parts_0[_S159];
        var col_0 : u32 = (bitcast<u32>((o_1.w)));
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S159 + u32(1)]);
        var _S160 : u32 = u32(4) * col_0;
        colliders_0[_S160] = vec4<f32>((bitcast<f32>((slot_6))), colliders_0[_S160].y, colliders_0[_S160].z, colliders_0[_S160].w);
        var _S161 : u32 = _S160 + u32(1);
        colliders_0[_S161] = vec4<f32>(rotate_0(_S155, o_1.xyz - _S154), colliders_0[_S161].w);
        colliders_0[_S160 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S155, rot_1)));
        var _S162 : u32 = u32(2) * col_0;
        collider_info_0[_S162] = vec4<u32>(collider_info_0[_S162].x, slot_6, collider_info_0[_S162].z, collider_info_0[_S162].w);
        collider_info_0[_S162 + u32(1)] = vec4<u32>((bitcast<u32>((_S156))), (bitcast<u32>((_S158.x))), (bitcast<u32>((_S158.w))), u32(0));
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
    var _S163 : vec3<f32>;
    if(k_3 == u32(0))
    {
        _S163 = vec3<f32>(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_3 == u32(1))
        {
            _S163 = vec3<f32>(q_14, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_3 == u32(2))
            {
                _S163 = vec3<f32>(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_3 == u32(3))
                {
                    _S163 = vec3<f32>(0.34959998726844788f, q_14, 0.92000001668930054f);
                }
                else
                {
                    if(k_3 == u32(4))
                    {
                        _S163 = vec3<f32>(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S163 = vec3<f32>(0.92000001668930054f, 0.34959998726844788f, q_14);
                    }
                }
            }
        }
    }
    return _S163;
}

fn write_object_0( o_2 : u32,  q_15 : Quat_0,  t_4 : vec3<f32>,  color_0 : vec4<f32>)
{
    var x_7 : vec3<f32> = rotate_0(q_15, vec3<f32>(1.0f, 0.0f, 0.0f));
    var y_2 : vec3<f32> = rotate_0(q_15, vec3<f32>(0.0f, 1.0f, 0.0f));
    var z_3 : vec3<f32> = rotate_0(q_15, vec3<f32>(0.0f, 0.0f, 1.0f));
    var _S164 : u32 = u32(4) * o_2;
    objects_0[_S164] = vec4<f32>(x_7.x, y_2.x, z_3.x, t_4.x);
    objects_0[_S164 + u32(1)] = vec4<f32>(x_7.y, y_2.y, z_3.y, t_4.y);
    objects_0[_S164 + u32(2)] = vec4<f32>(x_7.z, y_2.z, z_3.z, t_4.z);
    objects_0[_S164 + u32(3)] = color_0;
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
    var _S165 : u32 = u32(3) * k_4;
    var v0_0 : vec4<f32> = view_chunks_0[_S165];
    var base_0 : vec3<f32> = view_chunks_0[_S165 + u32(2)].xyz;
    var fixed_0 : bool = (bodies_0[b_5].w) == 0.0f;
    var color_1 : vec3<f32>;
    if((view_0.mode_0) == u32(1))
    {
        var _S166 : f32 = max(length(bodies_0[b_5 + u32(3)].xyz), length(bodies_0[b_5 + u32(4)].xyz));
        if(fixed_0)
        {
            color_1 = vec3<f32>(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S166 > (view_0.rest_speed_0))
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
                color_1 = base_0;
            }
            else
            {
                color_1 = vec3<f32>(0.44999998807907104f) * base_0 + vec3<f32>(0.55000001192092896f) * slot_color_0(slot_8);
            }
        }
    }
    write_object_0((bitcast<u32>((v0_0.w))), normalized_0(quat_mul_0(q_16, quat_of_0(view_chunks_0[_S165 + u32(1)]))), origin_4 + rotate_0(q_16, v0_0.xyz), vec4<f32>(color_1, 1.0f));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_shots(@builtin(global_invocation_id) id_18 : vec3<u32>)
{
    var i_3 : u32 = id_18.x;
    if(i_3 >= (view_0.shot_count_0))
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
    var _S167 : u32 = u32(7) * v_8;
    lines_0[_S167] = p_4.x;
    lines_0[_S167 + u32(1)] = p_4.y;
    lines_0[_S167 + u32(2)] = p_4.z;
    lines_0[_S167 + u32(3)] = 1.0f;
    lines_0[_S167 + u32(4)] = 0.86000001430511475f;
    lines_0[_S167 + u32(5)] = 0.25f;
    lines_0[_S167 + u32(6)] = 1.0f;
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
    var _S168 : u32 = u32(2) * i_4;
    write_line_vertex_0(_S168, a_4);
    write_line_vertex_0(_S168 + u32(1), b_7);
    return;
}

