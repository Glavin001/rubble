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
        var w_1 : f32 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_1 * (cos(w_1 * (_S11 - _S10)) - cos(w_1 * (_S12 - _S10)));
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
    var slot_1 : u32 = body_of_0(k_0);
    var b_1 : u32 = u32(7) * slot_1;
    var q_6 : Quat_0;
    var origin_1 : vec3<f32>;
    structure_pose_0(slot_1, b_1, &(q_6), &(origin_1));
    var _S14 : u32 = u32(4) * k_0;
    var c0_0 : vec4<f32> = chunks_0[_S14];
    var centre_0 : vec3<f32> = origin_1 + rotate_0(q_6, c0_0.xyz);
    var inv_mass_0 : f32 = bodies_0[b_1].w;
    var m_body_0 : f32;
    if(inv_mass_0 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_0;
    }
    else
    {
        m_body_0 = float_bits_0(u32(2139095040));
    }
    var _S15 : u32 = params_0.chunk_count_0;
    var _S16 : f32 = c0_0.w;
    var _S17 : vec4<f32> = chunks_0[_S14 + u32(3)];
    var _S18 : f32 = params_0.band_0;
    var _S19 : f32 = params_0.dt_0;
    var _S20 : bool = (params_0.impact_peak_0) != u32(0);
    var _S21 : u32 = l_pulse_base_0();
    var _S22 : u32 = l_pulse_count_base_0();
    var _S23 : u32 = params_0.pulse_cap_0;
    loop
    {
        var _S24 : vec3<f32> = vec3<f32>(0.0f);
        var _S25 : u32 = sort_0[k_0];
        var _S26 : u32 = sort_0[_S15 + u32(1) + k_0];
        var first_0 : bool = true;
        var last_0 : u32 = u32(0);
        var step_1 : u32 = u32(0);
        f_total_0 = _S24;
        var t_total_1 : vec3<f32> = _S24;
        loop
        {
            if(step_1 < _S26)
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
                if(q3_0 < _S26)
                {
                }
                else
                {
                    break;
                }
                var key_1 : u32 = manifolds_0[_S25 + q3_0].ids_0.y + u32(1);
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
            var r_4 : Record_std430_0 = manifolds_0[_S25 + p_0];
            var _S27 : vec3<f32> = r_4.impulse_0.xyz;
            var j_0 : f32 = length(_S27);
            if(j_0 == 0.0f)
            {
                var _S28 : u32 = step_1 + u32(1);
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S28;
                continue;
            }
            var m_other_0 : f32 = r_4.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_0))
                {
                    m_1 = _S16;
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
            var _S29 : f32 = min(j_0, 2.0f * m_1 * max(r_4.impulse_0.w - _S18, 0.0f));
            if(_S29 < j_0)
            {
                var f_0 : vec3<f32> = _S27 / vec3<f32>(j_0) * vec3<f32>(((j_0 - _S29) / _S19));
                var _S30 : vec3<f32> = t_total_1 + cross(r_4.point_0.xyz - centre_0, f_0);
                f_total_1 = f_total_0 + f_0;
                t_total_0 = _S30;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            var _S28 : u32 = step_1 + u32(1);
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S28;
        }
        var _S31 : u32 = _S21 + u32(3) * _S23 * k_0;
        var _S32 : u32 = _S22 + k_0;
        var _S33 : u32 = (bitcast<u32>((loads_0[_S32].x)));
        p_0 = u32(0);
        kept_0 = u32(0);
        loop
        {
            if(p_0 < _S33)
            {
            }
            else
            {
                break;
            }
            var _S34 : u32 = _S31 + u32(3) * p_0;
            var a_1 : vec4<f32> = loads_0[_S34];
            var b_2 : vec4<f32> = loads_0[_S34 + u32(1)];
            var c_2 : vec4<f32> = loads_0[_S34 + u32(2)];
            var _S35 : f32 = loads_0[_S34 + u32(1)].w;
            var mean_0 : f32 = pulse_mean_0(loads_0[_S34].x, loads_0[_S34].y, loads_0[_S34].z, loads_0[_S34].w, _S35, _S19);
            if(mean_0 != 0.0f)
            {
                var f_1 : vec3<f32> = c_2.xyz * vec3<f32>(mean_0);
                var _S36 : vec3<f32> = t_total_1 + cross(b_2.xyz - centre_0, f_1);
                f_total_0 = f_total_0 + f_1;
                t_total_1 = _S36;
            }
            if((a_1.x + a_1.z + _S35) > _S19)
            {
                a_1[i32(0)] = a_1[i32(0)] - _S19;
                var _S37 : u32 = _S31 + u32(3) * kept_0;
                loads_0[_S37] = a_1;
                loads_0[_S37 + u32(1)] = b_2;
                loads_0[_S37 + u32(2)] = c_2;
                kept_0 = kept_0 + u32(1);
            }
            p_0 = p_0 + u32(1);
        }
        q3_0 = u32(0);
        overflow_0 = false;
        loop
        {
            if(q3_0 < _S26)
            {
            }
            else
            {
                break;
            }
            var r_5 : Record_std430_0 = manifolds_0[_S25 + q3_0];
            var _S38 : vec3<f32> = r_5.impulse_0.xyz;
            var j_1 : f32 = length(_S38);
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
                    m_1 = _S16;
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
            var _S39 : f32 = r_5.impulse_0.w;
            var _S40 : f32 = min(j_1, 2.0f * m_1 * max(_S39 - _S18, 0.0f));
            if(_S40 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var _S41 : f32 = _S17.y;
            var duration_1 : f32 = hertz_duration_0(m_1, min(r_5.other_0.z, _S17.z), 1.0f / ((1.0f - _S41 * _S41) / _S17.x + 1.0f / r_5.other_0.y), _S39);
            var peak_1 : f32 = 3.14159274101257324f * _S40 / (2.0f * duration_1);
            var dir_0 : vec3<f32> = _S38 / vec3<f32>(j_1);
            var mean_1 : f32;
            if(_S20)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S19);
            }
            if(mean_1 != 0.0f)
            {
                var f_2 : vec3<f32> = dir_0 * vec3<f32>(mean_1);
                var _S42 : vec3<f32> = t_total_1 + cross(r_5.point_0.xyz - centre_0, f_2);
                f_total_1 = f_total_0 + f_2;
                t_total_0 = _S42;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            var kept_1 : u32;
            if(duration_1 > _S19)
            {
                if(kept_0 < _S23)
                {
                    var _S43 : u32 = _S31 + u32(3) * kept_0;
                    loads_0[_S43] = vec4<f32>(- _S19, 0.0f, 0.0f, peak_1);
                    loads_0[_S43 + u32(1)] = vec4<f32>(r_5.point_0.xyz, duration_1);
                    loads_0[_S43 + u32(2)] = vec4<f32>(dir_0, 0.0f);
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
        loads_0[_S32] = vec4<f32>((bitcast<f32>((kept_0))), 0.0f, 0.0f, 0.0f);
        break;
    }
    if(overflow_0)
    {
        var _S44 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(8));
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
    var _S45 : bool;
    if(i_0 >= (params_0.bond_count_0))
    {
        _S45 = true;
    }
    else
    {
        _S45 = work_0[w_alive_0() + i_0] == u32(0);
    }
    if(_S45)
    {
        return;
    }
    var bond_0 : vec4<u32> = bonds_0[i_0];
    var _S46 : u32 = bond_0.x;
    if((max(loads_0[_S46].x, loads_0[bond_0.y].x)) > (bitcast<f32>((bond_0.z))))
    {
        work_0[w_alive_0() + i_0] = u32(0);
        var _S47 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S46)]), u32(1));
        var _S48 : u32 = a_counters_0();
        var _S49 : u32 = atomicAdd(&(atom_0[_S48]), u32(1));
        var _S50 : u32 = atomicAdd(&(atom_0[_S48 + u32(1)]), u32(1));
    }
    return;
}

fn groups_of_0( n_2 : u32,  size_0 : u32) -> u32
{
    var _S51 : u32 = (n_2 + size_0 - u32(1)) / size_0;
    return _S51;
}

fn set_args_0( at_0 : u32,  x_2 : u32)
{
    var _S52 : u32 = u32(3) * at_0;
    args_0[_S52] = x_2;
    args_0[_S52 + u32(1)] = u32(1);
    args_0[_S52 + u32(2)] = u32(1);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_gate(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var _S53 : u32 = a_counters_0();
    var _S54 : u32 = atomicLoad(&(atom_0[_S53]));
    var run_0 : bool = _S54 != u32(0);
    atomicStore(&(atom_0[_S53]), u32(0));
    if(run_0)
    {
        var _S55 : u32 = atomicAdd(&(atom_0[_S53 + u32(4)]), u32(1));
    }
    var _S56 : u32;
    if(run_0)
    {
        var _S57 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S56 = _S57;
    }
    else
    {
        _S56 = u32(0);
    }
    set_args_0(u32(0), _S56);
    if(run_0)
    {
        var _S58 : u32 = groups_of_0(params_0.bond_count_0, u32(64));
        _S56 = _S58;
    }
    else
    {
        _S56 = u32(0);
    }
    set_args_0(u32(1), _S56);
    if(run_0)
    {
        var _S59 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S56 = _S59;
    }
    else
    {
        _S56 = u32(0);
    }
    set_args_0(u32(2), _S56);
    if(run_0)
    {
        var _S60 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S56 = _S60;
    }
    else
    {
        _S56 = u32(0);
    }
    set_args_0(u32(3), _S56);
    if(run_0)
    {
        var _S61 : u32 = groups_of_0(params_0.body_count_0, u32(1024));
        _S56 = _S61;
    }
    else
    {
        _S56 = u32(0);
    }
    set_args_0(u32(4), _S56);
    var _S62 : i32;
    if(run_0)
    {
        _S62 = i32(1);
    }
    else
    {
        _S62 = i32(0);
    }
    set_args_0(u32(5), u32(_S62));
    return;
}

fn dirty_0( body_0 : u32) -> bool
{
    var _S63 : u32 = atomicLoad(&(atom_0[a_dirty_0() + body_0]));
    return _S63 != u32(0);
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
    var _S64 : bool;
    if(c_3 >= (params_0.chunk_count_0))
    {
        _S64 = true;
    }
    else
    {
        var _S65 : bool = dirty_0(body_of_0(c_3));
        _S64 = !_S65;
    }
    if(_S64)
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
    var _S66 : bool;
    if(s_0 >= (params_0.body_count_0))
    {
        _S66 = true;
    }
    else
    {
        var _S67 : bool = dirty_0(s_0);
        _S66 = !_S67;
    }
    if(_S66)
    {
        return;
    }
    atomicStore(&(atom_0[a_best_count_0() + s_0]), u32(0));
    atomicStore(&(atom_0[a_best_root_0() + s_0]), u32(4294967295));
    return;
}

fn find_0( x_3 : u32) -> u32
{
    var _S68 : u32 = a_parent_0();
    var _S69 : u32 = atomicLoad(&(atom_0[_S68 + x_3]));
    var p_1 : u32 = _S69;
    var _S70 : u32 = x_3;
    var guard_0 : u32 = u32(0);
    loop
    {
        var _S71 : bool;
        if(guard_0 < u32(1048576))
        {
            _S71 = p_1 != _S70;
        }
        else
        {
            _S71 = false;
        }
        if(_S71)
        {
        }
        else
        {
            break;
        }
        var gp_0 : u32 = atomicLoad(&(atom_0[_S68 + p_1]));
        if(gp_0 != p_1)
        {
            var _S72 : u32 = atomicCompareExchangeWeak(&(atom_0[_S68 + _S70]), p_1, gp_0).old_value;
        }
        var _S73 : u32 = guard_0 + u32(1);
        _S70 = p_1;
        p_1 = gp_0;
        guard_0 = _S73;
    }
    return _S70;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_union(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var i_1 : u32 = id_5.x;
    var _S74 : bool;
    if(i_1 >= (params_0.bond_count_0))
    {
        _S74 = true;
    }
    else
    {
        _S74 = work_0[w_alive_0() + i_1] == u32(0);
    }
    if(_S74)
    {
        return;
    }
    var bond_1 : vec4<u32> = bonds_0[i_1];
    var _S75 : u32 = bond_1.x;
    var _S76 : bool = dirty_0(body_of_0(_S75));
    if(!_S76)
    {
        return;
    }
    var _S77 : u32 = bond_1.y;
    var a_2 : u32 = _S75;
    var b_3 : u32 = _S77;
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
        var rb_0 : u32 = find_0(b_3);
        if(ra_0 == rb_0)
        {
            return;
        }
        var _S78 : u32 = max(ra_0, rb_0);
        var _S79 : u32 = min(ra_0, rb_0);
        var was_0 : u32 = atomicCompareExchangeWeak(&(atom_0[a_parent_0() + _S78]), _S78, _S79).old_value;
        if(was_0 == _S78)
        {
            return;
        }
        var _S80 : u32 = guard_1 + u32(1);
        a_2 = _S78;
        b_3 = _S79;
        guard_1 = _S80;
    }
    var _S81 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(4));
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
        var _S82 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
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
    var _S83 : u32;
    if(v_3 < i32(0))
    {
        _S83 = u32(4294967295);
    }
    else
    {
        _S83 = u32(0);
    }
    var hi_0 : u32 = _S83 + carry_0;
    if(hi_0 != u32(0))
    {
        var _S84 : u32 = atomicAdd(&(atom_0[at_1 + u32(1)]), hi_0);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_flatten(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var c_4 : u32 = id_6.x;
    var _S85 : bool;
    if(c_4 >= (params_0.chunk_count_0))
    {
        _S85 = true;
    }
    else
    {
        var _S86 : bool = dirty_0(body_of_0(c_4));
        _S85 = !_S86;
    }
    if(_S85)
    {
        return;
    }
    var r_6 : u32 = find_0(c_4);
    work_0[w_label_0() + c_4] = r_6;
    var _S87 : u32 = atomicAdd(&(atom_0[a_count_0() + r_6]), u32(1));
    if(((info_0[c_4] & (u32(1)))) != u32(0))
    {
        var _S88 : u32 = atomicOr(&(atom_0[a_anchor_0() + r_6]), u32(1));
    }
    var _S89 : u32 = u32(4) * c_4;
    var c0_1 : vec4<f32> = chunks_0[_S89];
    var c1_0 : vec4<f32> = chunks_0[_S89 + u32(1)];
    var c2_0 : vec4<f32> = chunks_0[_S89 + u32(2)];
    var m_2 : f32 = c0_1.w;
    var d_0 : vec3<f32> = c0_1.xyz - chunks_0[u32(4) * r_6].xyz;
    var dd_0 : f32 = dot(d_0, d_0);
    var at_2 : u32 = a_sums_0() + u32(20) * r_6;
    add64_0(at_2, m_2, params_0.scale_m_0);
    var _S90 : f32 = d_0.x;
    var _S91 : f32 = m_2 * _S90;
    add64_0(at_2 + u32(2), _S91, params_0.scale_s_0);
    var _S92 : f32 = d_0.y;
    var _S93 : f32 = m_2 * _S92;
    add64_0(at_2 + u32(4), _S93, params_0.scale_s_0);
    var _S94 : f32 = d_0.z;
    add64_0(at_2 + u32(6), m_2 * _S94, params_0.scale_s_0);
    add64_0(at_2 + u32(8), c1_0.x + m_2 * (dd_0 - _S90 * _S90), params_0.scale_j_0);
    add64_0(at_2 + u32(10), c1_0.y + m_2 * (dd_0 - _S92 * _S92), params_0.scale_j_0);
    add64_0(at_2 + u32(12), c1_0.z + m_2 * (dd_0 - _S94 * _S94), params_0.scale_j_0);
    add64_0(at_2 + u32(14), c1_0.w - _S91 * _S92, params_0.scale_j_0);
    add64_0(at_2 + u32(16), c2_0.x - _S91 * _S94, params_0.scale_j_0);
    add64_0(at_2 + u32(18), c2_0.y - _S93 * _S94, params_0.scale_j_0);
    return;
}

fn is_root_0( c_5 : u32) -> bool
{
    var _S95 : bool;
    if(c_5 < (params_0.chunk_count_0))
    {
        var _S96 : bool = dirty_0(body_of_0(c_5));
        _S95 = _S96;
    }
    else
    {
        _S95 = false;
    }
    if(_S95)
    {
        _S95 = work_0[w_label_0() + c_5] == c_5;
    }
    else
    {
        _S95 = false;
    }
    return _S95;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_count(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var c_6 : u32 = id_7.x;
    var _S97 : bool = is_root_0(c_6);
    if(!_S97)
    {
        return;
    }
    var _S98 : u32 = atomicLoad(&(atom_0[a_count_0() + c_6]));
    var _S99 : u32 = atomicMax(&(atom_0[a_best_count_0() + body_of_0(c_6)]), _S98);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_root(@builtin(global_invocation_id) id_8 : vec3<u32>)
{
    var c_7 : u32 = id_8.x;
    var _S100 : bool = is_root_0(c_7);
    if(!_S100)
    {
        return;
    }
    var s_1 : u32 = body_of_0(c_7);
    var _S101 : u32 = atomicLoad(&(atom_0[a_count_0() + c_7]));
    var _S102 : u32 = atomicLoad(&(atom_0[a_best_count_0() + s_1]));
    if(_S101 == _S102)
    {
        var _S103 : u32 = atomicMin(&(atom_0[a_best_root_0() + s_1]), c_7);
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
    var _S104 : bool = is_root_0(c_8);
    var _S105 : bool;
    if(_S104)
    {
        var _S106 : u32 = atomicLoad(&(atom_0[a_best_root_0() + body_of_0(c_8)]));
        _S105 = _S106 != c_8;
    }
    else
    {
        _S105 = false;
    }
    var _S107 : u32;
    if(_S105)
    {
        _S107 = u32(1);
    }
    else
    {
        _S107 = u32(0);
    }
    work_0[w_new_0() + c_8] = _S107;
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
    var _S108 : u32 = group_0.x;
    var _S109 : u32 = u32(1024) * _S108 + u32(4) * t_1;
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
        if((_S109 + j_2) < (scan_0.n_0))
        {
            step_2 = work_0[scan_0.in_base_0 + _S109 + j_2];
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
    var _S110 : u32 = g_scan_0[t_1] - sum_2;
    j_2 = u32(0);
    add_0 = _S110;
    loop
    {
        if(j_2 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S109 + j_2) < (scan_0.n_0))
        {
            work_0[scan_0.out_base_0 + _S109 + j_2] = add_0;
        }
        var run_1 : u32 = add_0 + v_4[j_2];
        j_2 = j_2 + u32(1);
        add_0 = run_1;
    }
    if(t_1 == u32(255))
    {
        work_0[scan_0.block_base_0 + _S108] = g_scan_0[i32(255)];
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn fr_scan_top(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var t_2 : u32 = thread_1.x;
    var blocks_0 : u32 = (scan_0.n_0 + u32(1023)) / u32(1024);
    var _S111 : bool = t_2 < blocks_0;
    var v_5 : u32;
    if(_S111)
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
    if(_S111)
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
    var _S112 : bool;
    if(s_2 >= (params_0.body_count_0))
    {
        _S112 = true;
    }
    else
    {
        _S112 = work_0[w_free_0() + s_2] == u32(0);
    }
    if(_S112)
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
    var _S113 : bool = is_root_0(c_9);
    if(!_S113)
    {
        return;
    }
    var s_3 : u32 = body_of_0(c_9);
    var _S114 : u32 = atomicLoad(&(atom_0[a_best_root_0() + s_3]));
    var slot_2 : u32;
    if(_S114 != c_9)
    {
        var k_1 : u32 = work_0[w_rank_0() + c_9];
        if(work_0[w_rank_0() + c_9] < (free_total_0()))
        {
            var slot_3 : u32 = work_0[w_free_list_0() + k_1];
            work_0[w_free_0() + work_0[w_free_list_0() + k_1]] = u32(0);
            var _S115 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(2)]), u32(1));
            slot_2 = slot_3;
        }
        else
        {
            var _S116 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(1));
            slot_2 = s_3;
        }
    }
    else
    {
        slot_2 = s_3;
    }
    work_0[w_slot_0() + c_9] = slot_2;
    return;
}

fn read64_0( at_3 : u32,  scale_1 : f32) -> f32
{
    var _S117 : u32 = atomicLoad(&(atom_0[at_3 + u32(1)]));
    var hi_1 : i32 = (bitcast<i32>((_S117)));
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
    var q_7 : Quat_0;
    if(m22_0 <= 0.0f)
    {
        var dif10_0 : f32 = m11_0 - m00_0;
        var omm22_0 : f32 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            var f_3 : f32 = omm22_0 - dif10_0;
            var inv_0 : f32 = 0.5f / sqrt(f_3);
            q_7.x_1 = f_3 * inv_0;
            q_7.y_0 = (m01_0 + m10_0) * inv_0;
            q_7.z_0 = (m02_0 + m20_0) * inv_0;
            q_7.w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            var f_4 : f32 = omm22_0 + dif10_0;
            var inv_1 : f32 = 0.5f / sqrt(f_4);
            q_7.x_1 = (m01_0 + m10_0) * inv_1;
            q_7.y_0 = f_4 * inv_1;
            q_7.z_0 = (m12_0 + m21_0) * inv_1;
            q_7.w_0 = (m20_0 - m02_0) * inv_1;
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
            q_7.x_1 = (m02_0 + m20_0) * inv_2;
            q_7.y_0 = (m12_0 + m21_0) * inv_2;
            q_7.z_0 = f_5 * inv_2;
            q_7.w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            var f_6 : f32 = opm22_0 + sum10_0;
            var inv_3 : f32 = 0.5f / sqrt(f_6);
            q_7.x_1 = (m12_0 - m21_0) * inv_3;
            q_7.y_0 = (m20_0 - m02_0) * inv_3;
            q_7.z_0 = (m01_0 - m10_0) * inv_3;
            q_7.w_0 = f_6 * inv_3;
        }
    }
    return q_7;
}

fn principal_axes_0( r0_0 : vec3<f32>,  r1_0 : vec3<f32>,  r2_0 : vec3<f32>,  rot_0 : ptr<function, Quat_0>,  moments_0 : ptr<function, vec3<f32>>)
{
    var _S118 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S119 : array<f32, i32(3)> = array<f32, i32(3)>( r1_0.x, r1_0.y, r1_0.z );
    var _S120 : array<f32, i32(3)> = array<f32, i32(3)>( r2_0.x, r2_0.y, r2_0.z );
    var a_3 : array<array<f32, i32(3)>, i32(3)>;
    a_3[i32(0)] = _S118;
    a_3[i32(1)] = _S119;
    a_3[i32(2)] = _S120;
    var _S121 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S122 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S123 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_6 : array<array<f32, i32(3)>, i32(3)>;
    v_6[i32(0)] = _S121;
    v_6[i32(1)] = _S122;
    v_6[i32(2)] = _S123;
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
        var q_8 : u32;
        if((abs(a_3[i32(0)][i32(2)])) > (abs(a_3[u32(0)][u32(1)])))
        {
            q_8 = u32(2);
        }
        else
        {
            q_8 = u32(1);
        }
        var p_2 : u32;
        var q_9 : u32;
        if((abs(a_3[i32(1)][i32(2)])) > (abs(a_3[u32(0)][q_8])))
        {
            p_2 = u32(1);
            q_9 = u32(2);
        }
        else
        {
            p_2 = u32(0);
            q_9 = q_8;
        }
        if((abs(a_3[p_2][q_9])) <= (9.99999982451670044e-15f * max(max(abs(a_3[i32(0)][i32(0)]), abs(a_3[i32(1)][i32(1)])), max(abs(a_3[i32(2)][i32(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        var theta_0 : f32 = 0.5f * atan2(2.0f * a_3[p_2][q_9], a_3[q_9][q_9] - a_3[p_2][p_2]);
        var s_4 : f32 = sin(theta_0);
        var c_10 : f32 = cos(theta_0);
        var app_0 : f32 = a_3[p_2][p_2];
        var aqq_0 : f32 = a_3[q_9][q_9];
        var apq_0 : f32 = a_3[p_2][q_9];
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
            var _S124 : bool;
            if(k_2 != p_2)
            {
                _S124 = k_2 != q_9;
            }
            else
            {
                _S124 = false;
            }
            if(_S124)
            {
                var kp_0 : f32 = a_3[k_2][p_2];
                var kq_0 : f32 = a_3[k_2][q_9];
                var _S125 : f32 = c_10 * a_3[k_2][p_2] - s_4 * a_3[k_2][q_9];
                a_3[k_2][p_2] = _S125;
                a_3[p_2][k_2] = _S125;
                var _S126 : f32 = s_4 * kp_0 + c_10 * kq_0;
                a_3[k_2][q_9] = _S126;
                a_3[q_9][k_2] = _S126;
            }
            k_2 = k_2 + u32(1);
        }
        var _S127 : f32 = c_10 * c_10;
        var _S128 : f32 = 2.0f * s_4 * c_10 * apq_0;
        var _S129 : f32 = s_4 * s_4;
        a_3[p_2][p_2] = _S127 * app_0 - _S128 + _S129 * aqq_0;
        a_3[q_9][q_9] = _S129 * app_0 + _S128 + _S127 * aqq_0;
        a_3[p_2][q_9] = 0.0f;
        a_3[q_9][p_2] = 0.0f;
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
            var kq_1 : f32 = v_6[row_0][q_9];
            v_6[row_0][p_2] = c_10 * v_6[row_0][p_2] - s_4 * v_6[row_0][q_9];
            v_6[row_0][q_9] = s_4 * kp_1 + c_10 * kq_1;
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

fn rot_to_rubble_0( q_10 : Quat_0) -> Quat_0
{
    var axes_1 : Quat_0;
    axes_1.w_0 = 0.70710676908493042f;
    axes_1.x_1 = -0.70710676908493042f;
    axes_1.y_0 = 0.0f;
    axes_1.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_1, q_10));
}

fn stage_0( c_11 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * c_11;
}

fn quat_vec_0( q_11 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_11.x_1, q_11.y_0, q_11.z_0, q_11.w_0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_piece(@builtin(global_invocation_id) id_13 : vec3<u32>)
{
    var c_12 : u32 = id_13.x;
    var _S130 : bool = is_root_0(c_12);
    if(!_S130)
    {
        return;
    }
    var parent_0 : u32 = body_of_0(c_12);
    var pb_0 : u32 = u32(7) * parent_0;
    var at_4 : u32 = a_sums_0() + u32(20) * c_12;
    var mass_0 : f32 = read64_0(at_4, params_0.scale_m_0);
    var _S131 : f32 = read64_0(at_4 + u32(2), params_0.scale_s_0);
    var _S132 : f32 = read64_0(at_4 + u32(4), params_0.scale_s_0);
    var _S133 : f32 = read64_0(at_4 + u32(6), params_0.scale_s_0);
    var d_1 : vec3<f32> = vec3<f32>(_S131, _S132, _S133) / vec3<f32>(max(mass_0, 1.00000000317107685e-30f));
    var jxx_0 : f32 = read64_0(at_4 + u32(8), params_0.scale_j_0);
    var jyy_0 : f32 = read64_0(at_4 + u32(10), params_0.scale_j_0);
    var jzz_0 : f32 = read64_0(at_4 + u32(12), params_0.scale_j_0);
    var jxy_0 : f32 = read64_0(at_4 + u32(14), params_0.scale_j_0);
    var jxz_0 : f32 = read64_0(at_4 + u32(16), params_0.scale_j_0);
    var jyz_0 : f32 = read64_0(at_4 + u32(18), params_0.scale_j_0);
    var dd_1 : f32 = dot(d_1, d_1);
    var _S134 : f32 = d_1.x;
    var _S135 : f32 = mass_0 * _S134;
    var _S136 : f32 = d_1.y;
    var _S137 : f32 = jxy_0 + _S135 * _S136;
    var _S138 : f32 = d_1.z;
    var _S139 : f32 = jxz_0 + _S135 * _S138;
    var _S140 : f32 = jyz_0 + mass_0 * _S136 * _S138;
    var com_1 : vec3<f32> = chunks_0[u32(4) * c_12].xyz + d_1;
    var principal_0 : Quat_0;
    var moments_1 : vec3<f32>;
    principal_axes_0(vec3<f32>(jxx_0 - mass_0 * (dd_1 - _S134 * _S134), _S137, _S139), vec3<f32>(_S137, jyy_0 - mass_0 * (dd_1 - _S136 * _S136), _S140), vec3<f32>(_S139, _S140, jzz_0 - mass_0 * (dd_1 - _S138 * _S138)), &(principal_0), &(moments_1));
    var _S141 : u32 = atomicLoad(&(atom_0[a_anchor_0() + c_12]));
    var anchored_0 : bool = _S141 != u32(0);
    var q_12 : Quat_0;
    var origin_2 : vec3<f32>;
    structure_pose_0(parent_0, pb_0, &(q_12), &(origin_2));
    var p_parent_0 : vec3<f32> = bodies_0[pb_0].xyz;
    var v_parent_0 : vec3<f32> = bodies_0[pb_0 + u32(3)].xyz;
    var w_parent_0 : vec3<f32> = bodies_0[pb_0 + u32(4)].xyz;
    var pos_0 : vec3<f32> = to_rubble_0(origin_2 + rotate_0(q_12, com_1));
    var orientation_0 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(q_12, principal_0)));
    var _S142 : bool;
    if(anchored_0)
    {
        _S142 = true;
    }
    else
    {
        _S142 = !(mass_0 > 0.0f);
    }
    var inv_mass_1 : f32;
    if(_S142)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / mass_0;
    }
    var _S143 : vec3<f32> = vec3<f32>(0.0f);
    var inv_inertia_0 : vec3<f32>;
    if(!anchored_0)
    {
        var _S144 : f32;
        if((moments_1.x) > 0.0f)
        {
            _S144 = 1.0f / moments_1.x;
        }
        else
        {
            _S144 = 0.0f;
        }
        var _S145 : f32;
        if((moments_1.y) > 0.0f)
        {
            _S145 = 1.0f / moments_1.y;
        }
        else
        {
            _S145 = 0.0f;
        }
        var _S146 : f32;
        if((moments_1.z) > 0.0f)
        {
            _S146 = 1.0f / moments_1.z;
        }
        else
        {
            _S146 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S144, _S145, _S146);
    }
    else
    {
        inv_inertia_0 = _S143;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_0)
    {
        velocity_0 = _S143;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    var spin_0 : vec3<f32>;
    if(anchored_0)
    {
        spin_0 = _S143;
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
    frames_0[st_0 + u32(7)] = vec4<f32>(com_1, frames_0[u32(2) * parent_0].w);
    frames_0[st_0 + u32(8)] = quat_vec_0(principal_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_commit(@builtin(global_invocation_id) id_14 : vec3<u32>)
{
    var c_13 : u32 = id_14.x;
    var _S147 : bool = is_root_0(c_13);
    if(!_S147)
    {
        return;
    }
    var slot_4 : u32 = work_0[w_slot_0() + c_13];
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
            bodies_0[u32(7) * (slot_4 + region_0 * params_0.max_bodies_0) + w_3] = frames_0[st_1 + w_3];
            w_3 = w_3 + u32(1);
        }
        region_0 = region_0 + u32(1);
    }
    var _S148 : u32 = u32(2) * slot_4;
    frames_0[_S148] = frames_0[st_1 + u32(7)];
    frames_0[_S148 + u32(1)] = frames_0[st_1 + u32(8)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_reparent(@builtin(global_invocation_id) id_15 : vec3<u32>)
{
    var c_14 : u32 = id_15.x;
    var _S149 : bool;
    if(c_14 >= (params_0.chunk_count_0))
    {
        _S149 = true;
    }
    else
    {
        var _S150 : bool = dirty_0(body_of_0(c_14));
        _S149 = !_S150;
    }
    if(_S149)
    {
        return;
    }
    var r_7 : u32 = work_0[w_label_0() + c_14];
    var slot_5 : u32 = work_0[w_slot_0() + work_0[w_label_0() + c_14]];
    work_0[w_body_0() + c_14] = work_0[w_slot_0() + work_0[w_label_0() + c_14]];
    var _S151 : u32 = u32(2) * slot_5;
    var _S152 : vec3<f32> = frames_0[_S151].xyz;
    var _S153 : Quat_0 = conj_0(quat_of_0(frames_0[_S151 + u32(1)]));
    var inv_mass_2 : f32 = bodies_0[u32(7) * slot_5].w;
    var _S154 : f32;
    if(inv_mass_2 > 0.0f)
    {
        _S154 = 1.0f / inv_mass_2;
    }
    else
    {
        _S154 = float_bits_0(u32(2139095040));
    }
    var c2_2 : vec4<f32> = chunks_0[u32(4) * c_14 + u32(2)];
    var first_1 : u32 = (bitcast<u32>((c2_2.z)));
    var _S155 : u32 = (bitcast<u32>((c2_2.w)));
    var _S156 : vec4<f32> = chunks_0[u32(4) * r_7 + u32(3)];
    var p_3 : u32 = first_1;
    loop
    {
        if(p_3 < (first_1 + _S155))
        {
        }
        else
        {
            break;
        }
        var _S157 : u32 = u32(2) * p_3;
        var o_1 : vec4<f32> = parts_0[_S157];
        var col_0 : u32 = (bitcast<u32>((o_1.w)));
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S157 + u32(1)]);
        var _S158 : u32 = u32(4) * col_0;
        colliders_0[_S158] = vec4<f32>((bitcast<f32>((slot_5))), colliders_0[_S158].y, colliders_0[_S158].z, colliders_0[_S158].w);
        var _S159 : u32 = _S158 + u32(1);
        colliders_0[_S159] = vec4<f32>(rotate_0(_S153, o_1.xyz - _S152), colliders_0[_S159].w);
        colliders_0[_S158 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S153, rot_1)));
        var _S160 : u32 = u32(2) * col_0;
        collider_info_0[_S160] = vec4<u32>(collider_info_0[_S160].x, slot_5, collider_info_0[_S160].z, collider_info_0[_S160].w);
        collider_info_0[_S160 + u32(1)] = vec4<u32>((bitcast<u32>((_S154))), (bitcast<u32>((_S156.x))), (bitcast<u32>((_S156.w))), u32(0));
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

