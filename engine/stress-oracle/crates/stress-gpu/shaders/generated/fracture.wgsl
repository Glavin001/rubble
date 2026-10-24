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
    @align(8) cs_sweeps_0 : u32,
    @align(4) cs_row_cap_0 : u32,
    @align(16) cs_c1_0 : f32,
    @align(4) cs_c2_0 : f32,
    @align(8) cs_row_offset_0 : u32,
    @align(4) pad7_0 : f32,
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

@binding(28) @group(0) var<storage, read> lp_sides_0 : array<vec4<f32>>;

@binding(26) @group(0) var<storage, read> engine_counters_0 : array<u32>;

@binding(25) @group(0) var<storage, read_write> engine_contacts_0 : array<vec4<f32>>;

@binding(27) @group(0) var<storage, read_write> cs_anchors_0 : array<vec4<f32>>;

fn lp_body_0( b_0 : u32) -> u32
{
    return u32(22) * params_0.chunk_count_0 + u32(6) * b_0;
}

fn il_t_in_0( b_1 : u32) -> f32
{
    return lp_0[lp_body_0(b_1) + u32(5)].w;
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

fn a_strike_0() -> u32
{
    return a_counters_0() + u32(16);
}

fn il_t_in_1( b_2 : u32) -> f32
{
    var _S1 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_2 + u32(6)]));
    return (bitcast<f32>((_S1)));
}

struct IlStriker_0
{
     body_0 : u32,
     bound_0 : f32,
     radius_0 : f32,
     mass_0 : f32,
     m_eff_0 : f32,
     modulus_0 : f32,
     launch_0 : u32,
};

fn il_striker_0( b_3 : u32) -> IlStriker_0
{
    var _S2 : u32 = lp_body_0(b_3);
    var s_0 : IlStriker_0;
    s_0.body_0 = (bitcast<u32>((lp_0[_S2 + u32(4)].w)));
    var _S3 : f32 = lp_0[_S2 + u32(4)].y;
    s_0.bound_0 = _S3;
    s_0.radius_0 = _S3;
    s_0.mass_0 = lp_0[_S2 + u32(5)].z;
    s_0.m_eff_0 = lp_0[_S2 + u32(4)].x;
    s_0.modulus_0 = lp_0[_S2 + u32(4)].z;
    s_0.launch_0 = u32(0);
    return s_0;
}

fn a_first_0( shot_0 : u32) -> u32
{
    return a_strike_0() + u32(27) * params_0.body_count_0 + shot_0;
}

fn a_cs_0( b_4 : u32) -> u32
{
    return a_first_0(u32(0)) + u32(64) + u32(12) * b_4;
}

fn il_entry_0( b_5 : u32) -> u32
{
    var _S4 : u32 = atomicLoad(&(atom_0[a_cs_0(b_5) + u32(1)]));
    return _S4 - u32(1);
}

fn il_path_0( b_6 : u32,  contact_0 : ptr<function, vec3<f32>>,  d_0 : ptr<function, vec3<f32>>,  speed_0 : ptr<function, f32>)
{
    var _S5 : u32 = lp_body_0(b_6);
    (*contact_0) = lp_0[_S5 + u32(2)].xyz;
    var r1_0 : vec4<f32> = lp_0[_S5 + u32(3)];
    (*d_0) = lp_0[_S5 + u32(3)].xyz;
    (*speed_0) = r1_0.w;
    return;
}

fn il_path_1( b_7 : u32,  contact_1 : ptr<function, vec3<f32>>,  d_1 : ptr<function, vec3<f32>>,  speed_1 : ptr<function, f32>)
{
    var _S6 : u32 = lp_body_0(b_7);
    (*contact_1) = lp_0[_S6 + u32(2)].xyz;
    var r1_1 : vec4<f32> = lp_0[_S6 + u32(3)];
    (*d_1) = lp_0[_S6 + u32(3)].xyz;
    (*speed_1) = r1_1.w;
    return;
}

fn cs_struck_0( b_8 : u32) -> bool
{
    var _S7 : bool;
    if(b_8 < (params_0.body_count_0))
    {
        var _S8 : u32 = atomicLoad(&(atom_0[a_cs_0(b_8) + u32(1)]));
        _S7 = _S8 != u32(0);
    }
    else
    {
        _S7 = false;
    }
    return _S7;
}

fn il_struck_0( b_9 : u32) -> bool
{
    var _S9 : bool = cs_struck_0(b_9);
    return _S9;
}

fn strike_at_0( b_10 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * params_0.chunk_count_0 + u32(3) * b_10;
}

fn il_striker_1( b_11 : u32) -> IlStriker_0
{
    var _S10 : u32 = u32(2) * (u32(frames_0[strike_at_0(b_11)].w) - u32(1));
    var info_1 : vec4<f32> = shots_0[_S10];
    var s_1 : IlStriker_0;
    s_1.body_0 = (bitcast<u32>((info_1.x)));
    s_1.bound_0 = info_1.y;
    s_1.radius_0 = info_1.z;
    s_1.mass_0 = info_1.w;
    var inv_mass_0 : f32 = bodies_0[u32(7) * b_11].w;
    var _S11 : f32;
    if(inv_mass_0 == 0.0f)
    {
        _S11 = s_1.mass_0;
    }
    else
    {
        _S11 = s_1.mass_0 / (1.0f + s_1.mass_0 * inv_mass_0);
    }
    s_1.m_eff_0 = _S11;
    var _S12 : u32 = _S10 + u32(1);
    s_1.modulus_0 = shots_0[_S12].x;
    s_1.launch_0 = (bitcast<u32>((shots_0[_S12].y)));
    return s_1;
}

fn il_entry_1( b_12 : u32) -> u32
{
    var _S13 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_12 + u32(8)]));
    return _S13;
}

fn il_path_2( b_13 : u32,  contact_2 : ptr<function, vec3<f32>>,  d_2 : ptr<function, vec3<f32>>,  speed_2 : ptr<function, f32>)
{
    var st_0 : u32 = strike_at_0(b_13);
    var rel_0 : vec3<f32> = frames_0[st_0 + u32(1)].xyz;
    var _S14 : f32 = length(rel_0);
    (*speed_2) = _S14;
    (*d_2) = rel_0 / vec3<f32>(max(_S14, 9.999999960041972e-13f));
    (*contact_2) = frames_0[st_0 + u32(2)].xyz;
    return;
}

fn il_path_3( b_14 : u32,  contact_3 : ptr<function, vec3<f32>>,  d_3 : ptr<function, vec3<f32>>,  speed_3 : ptr<function, f32>)
{
    var st_1 : u32 = strike_at_0(b_14);
    var rel_1 : vec3<f32> = frames_0[st_1 + u32(1)].xyz;
    var _S15 : f32 = length(rel_1);
    (*speed_3) = _S15;
    (*d_3) = rel_1 / vec3<f32>(max(_S15, 9.999999960041972e-13f));
    (*contact_3) = frames_0[st_1 + u32(2)].xyz;
    return;
}

fn struck_0( b_15 : u32,  shot_1 : ptr<function, u32>) -> bool
{
    var k_0 : u32 = u32(frames_0[strike_at_0(b_15)].w);
    (*shot_1) = k_0 - u32(1);
    return k_0 != u32(0);
}

fn il_struck_1( b_16 : u32) -> bool
{
    var shot_2 : u32;
    var _S16 : bool;
    if((params_0.shot_count_0) > u32(0))
    {
        _S16 = b_16 < (params_0.body_count_0);
    }
    else
    {
        _S16 = false;
    }
    if(_S16)
    {
        _S16 = struck_0(b_16, &(shot_2));
    }
    else
    {
        _S16 = false;
    }
    return _S16;
}

fn cs_rows_0() -> u32
{
    return min(engine_counters_0[i32(2)], params_0.cs_row_cap_0);
}

fn cs_row_0( r_0 : u32) -> u32
{
    return u32(8) * (params_0.cs_row_offset_0 + r_0);
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
     x_0 : f32,
     y_0 : f32,
     z_0 : f32,
};

fn quat_of_0( q_0 : vec4<f32>) -> Quat_0
{
    var r_1 : Quat_0;
    r_1.x_0 = q_0.x;
    r_1.y_0 = q_0.y;
    r_1.z_0 = q_0.z;
    r_1.w_0 = q_0.w;
    return r_1;
}

fn rotate_0( q_1 : Quat_0,  v_0 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_1.x_0, q_1.y_0, q_1.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_0) * vec3<f32>(2.0f);
    return v_0 + t_0 * vec3<f32>(q_1.w_0) + cross(qv_0, t_0);
}

fn conj_0( q_2 : Quat_0) -> Quat_0
{
    var c_1 : Quat_0;
    c_1.w_0 = q_2.w_0;
    c_1.x_0 = - q_2.x_0;
    c_1.y_0 = - q_2.y_0;
    c_1.z_0 = - q_2.z_0;
    return c_1;
}

fn cs_angular_0( b_17 : u32,  r_2 : vec3<f32>,  u_0 : vec3<f32>) -> f32
{
    var at_0 : u32 = u32(7) * b_17;
    var l_0 : vec3<f32> = rotate_0(conj_0(quat_of_0(bodies_0[at_0 + u32(1)])), cross(r_2, u_0));
    return dot(l_0 * bodies_0[at_0 + u32(2)].xyz, l_0);
}

fn cs_side_0( r_3 : u32,  side_0 : u32,  k_1 : ptr<function, u32>,  s_2 : ptr<function, u32>,  ob_0 : ptr<function, u32>,  co_0 : ptr<function, u32>,  p_0 : ptr<function, vec3<f32>>,  u_1 : ptr<function, vec3<f32>>,  v_1 : ptr<function, f32>,  m_eff_1 : ptr<function, f32>,  t_in_0 : ptr<function, f32>) -> bool
{
    (*t_in_0) = 0.0f;
    (*k_1) = u32(4294967295);
    (*s_2) = u32(4294967295);
    (*ob_0) = u32(4294967295);
    (*co_0) = u32(4294967295);
    var _S17 : vec3<f32> = vec3<f32>(0.0f);
    (*p_0) = _S17;
    (*u_1) = _S17;
    (*v_1) = 0.0f;
    (*m_eff_1) = 0.0f;
    var row_0 : u32 = cs_row_0(r_3);
    var ids_1 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_0])));
    if((ids_1.w) == u32(0))
    {
        return false;
    }
    var _S18 : bool = side_0 == u32(0);
    var cm_0 : u32;
    if(_S18)
    {
        cm_0 = ids_1.x;
    }
    else
    {
        cm_0 = ids_1.y;
    }
    var _S19 : u32;
    if(_S18)
    {
        _S19 = ids_1.y;
    }
    else
    {
        _S19 = ids_1.x;
    }
    (*co_0) = _S19;
    var _S20 : u32 = collider_info_0[u32(2) * cm_0].x;
    (*k_1) = _S20;
    var _S21 : bool;
    if(_S20 == u32(4294967295))
    {
        _S21 = true;
    }
    else
    {
        _S21 = (*k_1) >= (params_0.chunk_count_0);
    }
    if(_S21)
    {
        _S21 = true;
    }
    else
    {
        _S21 = (collider_info_0[u32(2) * (*co_0)].z) != u32(2);
    }
    if(_S21)
    {
        return false;
    }
    (*s_2) = body_of_0((*k_1));
    var _S22 : u32 = (bitcast<u32>((colliders_0[u32(4) * (*co_0)].x)));
    (*ob_0) = _S22;
    if(_S22 == (*s_2))
    {
        _S21 = true;
    }
    else
    {
        _S21 = (*ob_0) >= (params_0.max_bodies_0);
    }
    if(_S21)
    {
        return false;
    }
    var bs_0 : u32 = u32(7) * (*s_2);
    var bo_0 : u32 = u32(7) * (*ob_0);
    var inv_o_0 : f32 = bodies_0[bo_0].w;
    if(!(inv_o_0 > 0.0f))
    {
        return false;
    }
    var n_1 : vec3<f32> = engine_contacts_0[row_0 + u32(1)].xyz;
    var _S23 : vec3<f32>;
    if(_S18)
    {
        _S23 = (vec3<f32>(0) - n_1);
    }
    else
    {
        _S23 = n_1;
    }
    (*u_1) = _S23;
    var _S24 : Quat_0 = quat_of_0(bodies_0[bs_0 + u32(1)]);
    if(_S18)
    {
        _S23 = engine_contacts_0[row_0 + u32(2)].xyz;
    }
    else
    {
        _S23 = engine_contacts_0[row_0 + u32(3)].xyz;
    }
    var rs_0 : vec3<f32> = rotate_0(_S24, _S23);
    var _S25 : Quat_0 = quat_of_0(bodies_0[bo_0 + u32(1)]);
    if(_S18)
    {
        _S23 = engine_contacts_0[row_0 + u32(3)].xyz;
    }
    else
    {
        _S23 = engine_contacts_0[row_0 + u32(2)].xyz;
    }
    var ro_0 : vec3<f32> = rotate_0(_S25, _S23);
    var _S26 : vec3<f32> = bodies_0[bs_0].xyz + rs_0;
    (*p_0) = _S26;
    var gap_0 : f32 = dot(_S26 - (bodies_0[bo_0].xyz + ro_0), (*u_1));
    var _S27 : f32 = dot(bodies_0[bo_0 + u32(3)].xyz + cross(bodies_0[bo_0 + u32(4)].xyz, ro_0) - (bodies_0[bs_0 + u32(3)].xyz + cross(bodies_0[bs_0 + u32(4)].xyz, rs_0)), (*u_1));
    (*v_1) = _S27;
    if(!(_S27 > (params_0.strike_speed_0)))
    {
        return false;
    }
    var w_1 : f32 = bodies_0[bs_0].w + cs_angular_0((*s_2), rs_0, (*u_1)) + inv_o_0 + cs_angular_0((*ob_0), ro_0, (*u_1));
    if(!(w_1 > 0.0f))
    {
        return false;
    }
    (*m_eff_1) = 1.0f / w_1;
    (*t_in_0) = clamp(gap_0 / (*v_1), 0.0f, params_0.dt_0);
    return true;
}

fn cs_identity_0( r_4 : u32,  side_1 : u32) -> vec3<u32>
{
    var ids_2 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[cs_row_0(r_4)])));
    var _S28 : vec3<u32>;
    if(side_1 == u32(0))
    {
        _S28 = vec3<u32>(ids_2.x, ids_2.y, u32(2) * ids_2.z);
    }
    else
    {
        _S28 = vec3<u32>(ids_2.y, ids_2.x, u32(2) * ids_2.z + u32(1));
    }
    return _S28;
}

fn cs_pick_0( r_5 : u32)
{
    if(r_5 >= (cs_rows_0()))
    {
        return;
    }
    var side_2 : u32 = u32(0);
    loop
    {
        if(side_2 < u32(2))
        {
        }
        else
        {
            break;
        }
        var k_2 : u32;
        var s_3 : u32;
        var ob_1 : u32;
        var co_1 : u32;
        var p_1 : vec3<f32>;
        var u_2 : vec3<f32>;
        var v_2 : f32;
        var m_eff_2 : f32;
        var t_in_1 : f32;
        var _S29 : bool = cs_side_0(r_5, side_2, &(k_2), &(s_3), &(ob_1), &(co_1), &(p_1), &(u_2), &(v_2), &(m_eff_2), &(t_in_1));
        var _S30 : bool;
        if(!_S29)
        {
            _S30 = true;
        }
        else
        {
            var _S31 : u32 = (bitcast<u32>((m_eff_2 * v_2)));
            var _S32 : u32 = atomicLoad(&(atom_0[a_cs_0(s_3)]));
            _S30 = _S31 != _S32;
        }
        if(_S30)
        {
            side_2 = side_2 + u32(1);
            continue;
        }
        var id_0 : vec3<u32> = cs_identity_0(r_5, side_2);
        var at_1 : u32 = a_cs_0(s_3);
        var _S33 : u32 = atomicLoad(&(atom_0[at_1 + u32(2)]));
        if((~_S33) != (id_0.x))
        {
            side_2 = side_2 + u32(1);
            continue;
        }
        var _S34 : u32 = atomicLoad(&(atom_0[at_1 + u32(9)]));
        if((~_S34) != (id_0.y))
        {
            side_2 = side_2 + u32(1);
            continue;
        }
        var _S35 : u32 = atomicLoad(&(atom_0[at_1 + u32(10)]));
        if((~_S35) != (id_0.z))
        {
            side_2 = side_2 + u32(1);
            continue;
        }
        atomicStore(&(atom_0[at_1 + u32(11)]), u32(2) * r_5 + side_2 + u32(1));
        side_2 = side_2 + u32(1);
    }
    return;
}

fn cs_pick_1( r_6 : u32)
{
    if(r_6 >= (cs_rows_0()))
    {
        return;
    }
    var side_3 : u32 = u32(0);
    loop
    {
        if(side_3 < u32(2))
        {
        }
        else
        {
            break;
        }
        var k_3 : u32;
        var s_4 : u32;
        var ob_2 : u32;
        var co_2 : u32;
        var p_2 : vec3<f32>;
        var u_3 : vec3<f32>;
        var v_3 : f32;
        var m_eff_3 : f32;
        var t_in_2 : f32;
        var _S36 : bool = cs_side_0(r_6, side_3, &(k_3), &(s_4), &(ob_2), &(co_2), &(p_2), &(u_3), &(v_3), &(m_eff_3), &(t_in_2));
        var _S37 : bool;
        if(!_S36)
        {
            _S37 = true;
        }
        else
        {
            var _S38 : u32 = (bitcast<u32>((m_eff_3 * v_3)));
            var _S39 : u32 = atomicLoad(&(atom_0[a_cs_0(s_4)]));
            _S37 = _S38 != _S39;
        }
        if(_S37)
        {
            side_3 = side_3 + u32(1);
            continue;
        }
        var id_1 : vec3<u32> = cs_identity_0(r_6, side_3);
        var at_2 : u32 = a_cs_0(s_4);
        var _S40 : u32 = atomicLoad(&(atom_0[at_2 + u32(2)]));
        if((~_S40) != (id_1.x))
        {
            side_3 = side_3 + u32(1);
            continue;
        }
        var _S41 : u32 = atomicLoad(&(atom_0[at_2 + u32(9)]));
        if((~_S41) != (id_1.y))
        {
            side_3 = side_3 + u32(1);
            continue;
        }
        var _S42 : u32 = atomicMax(&(atom_0[at_2 + u32(10)]), ~id_1.z);
        side_3 = side_3 + u32(1);
    }
    return;
}

fn cs_pick_2( r_7 : u32)
{
    if(r_7 >= (cs_rows_0()))
    {
        return;
    }
    var side_4 : u32 = u32(0);
    loop
    {
        if(side_4 < u32(2))
        {
        }
        else
        {
            break;
        }
        var k_4 : u32;
        var s_5 : u32;
        var ob_3 : u32;
        var co_3 : u32;
        var p_3 : vec3<f32>;
        var u_4 : vec3<f32>;
        var v_4 : f32;
        var m_eff_4 : f32;
        var t_in_3 : f32;
        var _S43 : bool = cs_side_0(r_7, side_4, &(k_4), &(s_5), &(ob_3), &(co_3), &(p_3), &(u_4), &(v_4), &(m_eff_4), &(t_in_3));
        var _S44 : bool;
        if(!_S43)
        {
            _S44 = true;
        }
        else
        {
            var _S45 : u32 = (bitcast<u32>((m_eff_4 * v_4)));
            var _S46 : u32 = atomicLoad(&(atom_0[a_cs_0(s_5)]));
            _S44 = _S45 != _S46;
        }
        if(_S44)
        {
            side_4 = side_4 + u32(1);
            continue;
        }
        var id_2 : vec3<u32> = cs_identity_0(r_7, side_4);
        var at_3 : u32 = a_cs_0(s_5);
        var _S47 : u32 = atomicLoad(&(atom_0[at_3 + u32(2)]));
        if((~_S47) != (id_2.x))
        {
            side_4 = side_4 + u32(1);
            continue;
        }
        var _S48 : u32 = atomicMax(&(atom_0[at_3 + u32(9)]), ~id_2.y);
        side_4 = side_4 + u32(1);
    }
    return;
}

fn cs_pick_3( r_8 : u32)
{
    if(r_8 >= (cs_rows_0()))
    {
        return;
    }
    var side_5 : u32 = u32(0);
    loop
    {
        if(side_5 < u32(2))
        {
        }
        else
        {
            break;
        }
        var k_5 : u32;
        var s_6 : u32;
        var ob_4 : u32;
        var co_4 : u32;
        var p_4 : vec3<f32>;
        var u_5 : vec3<f32>;
        var v_5 : f32;
        var m_eff_5 : f32;
        var t_in_4 : f32;
        var _S49 : bool = cs_side_0(r_8, side_5, &(k_5), &(s_6), &(ob_4), &(co_4), &(p_4), &(u_5), &(v_5), &(m_eff_5), &(t_in_4));
        var _S50 : bool;
        if(!_S49)
        {
            _S50 = true;
        }
        else
        {
            var _S51 : u32 = (bitcast<u32>((m_eff_5 * v_5)));
            var _S52 : u32 = atomicLoad(&(atom_0[a_cs_0(s_6)]));
            _S50 = _S51 != _S52;
        }
        if(_S50)
        {
            side_5 = side_5 + u32(1);
            continue;
        }
        var id_3 : vec3<u32> = cs_identity_0(r_8, side_5);
        var at_4 : u32 = a_cs_0(s_6);
        var _S53 : u32 = atomicMax(&(atom_0[at_4 + u32(2)]), ~id_3.x);
        side_5 = side_5 + u32(1);
    }
    return;
}

fn w_alive_0() -> u32
{
    return params_0.chunk_count_0;
}

struct IlPlace_0
{
     along_0 : f32,
     radial_0 : f32,
     entry_0 : bool,
};

fn quat_mul_0( a_0 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_9 : Quat_0;
    r_9.w_0 = a_0.w_0 * o_0.w_0 - a_0.x_0 * o_0.x_0 - a_0.y_0 * o_0.y_0 - a_0.z_0 * o_0.z_0;
    r_9.x_0 = a_0.w_0 * o_0.x_0 + a_0.x_0 * o_0.w_0 + a_0.y_0 * o_0.z_0 - a_0.z_0 * o_0.y_0;
    r_9.y_0 = a_0.w_0 * o_0.y_0 - a_0.x_0 * o_0.z_0 + a_0.y_0 * o_0.w_0 + a_0.z_0 * o_0.x_0;
    r_9.z_0 = a_0.w_0 * o_0.z_0 + a_0.x_0 * o_0.y_0 - a_0.y_0 * o_0.x_0 + a_0.z_0 * o_0.w_0;
    return r_9;
}

fn normalized_0( q_3 : Quat_0) -> Quat_0
{
    var _S54 : f32 = q_3.w_0;
    var _S55 : f32 = q_3.x_0;
    var _S56 : f32 = q_3.y_0;
    var _S57 : f32 = q_3.z_0;
    var n_2 : f32 = sqrt(_S54 * _S54 + _S55 * _S55 + _S56 * _S56 + _S57 * _S57);
    var r_10 : Quat_0;
    r_10.w_0 = q_3.w_0 / n_2;
    r_10.x_0 = q_3.x_0 / n_2;
    r_10.y_0 = q_3.y_0 / n_2;
    r_10.z_0 = q_3.z_0 / n_2;
    return r_10;
}

fn rot_from_rubble_0( q_4 : Quat_0) -> Quat_0
{
    var axes_0 : Quat_0;
    axes_0.w_0 = 0.70710676908493042f;
    axes_0.x_0 = 0.70710676908493042f;
    axes_0.y_0 = 0.0f;
    axes_0.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_0, q_4));
}

fn from_rubble_0( v_6 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_6.x, - v_6.z, v_6.y);
}

fn structure_pose_0( slot_0 : u32,  b_18 : u32,  q_5 : ptr<function, Quat_0>,  origin_0 : ptr<function, vec3<f32>>)
{
    var _S58 : u32 = u32(2) * slot_0;
    var com_0 : vec3<f32> = frames_0[_S58].xyz;
    var _S59 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_18 + u32(1)])), conj_0(quat_of_0(frames_0[_S58 + u32(1)]))));
    (*q_5) = _S59;
    (*origin_0) = from_rubble_0(bodies_0[b_18].xyz) - rotate_0(_S59, com_0);
    return;
}

fn il_place_0( k_6 : u32,  entry_1 : u32,  contact_4 : vec3<f32>,  d_4 : vec3<f32>) -> IlPlace_0
{
    var slot_1 : u32 = body_of_0(k_6);
    var q_6 : Quat_0;
    var origin_1 : vec3<f32>;
    structure_pose_0(slot_1, u32(7) * slot_1, &(q_6), &(origin_1));
    var rel_2 : vec3<f32> = origin_1 + rotate_0(q_6, chunks_0[u32(4) * k_6].xyz) - contact_4;
    var p_5 : IlPlace_0;
    var _S60 : f32 = dot(rel_2, d_4);
    p_5.along_0 = _S60;
    p_5.radial_0 = length(rel_2 - d_4 * vec3<f32>(_S60));
    p_5.entry_0 = k_6 == entry_1;
    return p_5;
}

fn il_plug_0( outcome_0 : u32,  r_11 : f32,  h_0 : f32,  p_6 : IlPlace_0) -> bool
{
    var _S61 : bool;
    if(((outcome_0 & (u32(8)))) != u32(0))
    {
        if(p_6.entry_0)
        {
            _S61 = true;
        }
        else
        {
            if((p_6.along_0) >= 0.0f)
            {
                _S61 = (p_6.along_0) <= h_0;
            }
            else
            {
                _S61 = false;
            }
            if(_S61)
            {
                _S61 = (p_6.radial_0) <= (r_11 + p_6.along_0);
            }
            else
            {
                _S61 = false;
            }
        }
    }
    else
    {
        _S61 = false;
    }
    return _S61;
}

fn il_scab_0( outcome_1 : u32,  r_12 : f32,  h_1 : f32,  scab_0 : f32,  p_7 : IlPlace_0) -> bool
{
    var _S62 : bool;
    if(((outcome_1 & (u32(4)))) != u32(0))
    {
        _S62 = (p_7.along_0) >= (h_1 - scab_0);
    }
    else
    {
        _S62 = false;
    }
    if(_S62)
    {
        _S62 = (p_7.along_0) <= h_1;
    }
    else
    {
        _S62 = false;
    }
    if(_S62)
    {
        _S62 = (p_7.radial_0) <= (r_12 + h_1);
    }
    else
    {
        _S62 = false;
    }
    return _S62;
}

fn il_crater_0( outcome_2 : u32,  r_13 : f32,  depth_0 : f32,  a_1 : f32,  p_8 : IlPlace_0) -> bool
{
    var _S63 : bool;
    if(((outcome_2 & (u32(2)))) != u32(0))
    {
        _S63 = (p_8.along_0) >= 0.0f;
    }
    else
    {
        _S63 = false;
    }
    if(_S63)
    {
        _S63 = (p_8.along_0) <= depth_0;
    }
    else
    {
        _S63 = false;
    }
    if(_S63)
    {
        _S63 = (p_8.radial_0) <= (max(r_13, a_1));
    }
    else
    {
        _S63 = false;
    }
    return _S63;
}

fn il_breaks_0( i_0 : u32)
{
    var cut_0 : bool;
    if(i_0 >= (params_0.bond_count_0))
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = work_0[w_alive_0() + i_0] == u32(0);
    }
    if(cut_0)
    {
        return;
    }
    var bond_0 : vec4<u32> = bonds_0[i_0];
    var _S64 : u32 = bond_0.x;
    var b_19 : u32 = body_of_0(_S64);
    var _S65 : bool = il_struck_0(b_19);
    if(!_S65)
    {
        return;
    }
    var _S66 : u32 = lp_body_0(b_19);
    var s0_0 : vec4<f32> = lp_0[_S66];
    var s1_0 : vec4<f32> = lp_0[_S66 + u32(1)];
    var outcome_3 : u32 = (bitcast<u32>((lp_0[_S66].w)));
    if(((outcome_3 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_5 : vec3<f32>;
    var d_5 : vec3<f32>;
    var speed_4 : f32;
    il_path_1(b_19, &(contact_5), &(d_5), &(speed_4));
    var entry_2 : u32 = il_entry_0(b_19);
    var _S67 : f32 = max(il_striker_0(b_19).radius_0, 0.00100000004749745f);
    var h_2 : f32 = s0_0.x;
    var pa_0 : IlPlace_0 = il_place_0(_S64, entry_2, contact_5, d_5);
    var pb_0 : IlPlace_0 = il_place_0(bond_0.y, entry_2, contact_5, d_5);
    if((il_plug_0(outcome_3, _S67, h_2, pa_0)) != (il_plug_0(outcome_3, _S67, h_2, pb_0)))
    {
        cut_0 = true;
    }
    else
    {
        var _S68 : f32 = s1_0.y;
        cut_0 = (il_scab_0(outcome_3, _S67, h_2, _S68, pa_0)) != (il_scab_0(outcome_3, _S67, h_2, _S68, pb_0));
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = il_crater_0(outcome_3, _S67, s1_0.x, s1_0.z, pa_0);
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = il_crater_0(outcome_3, _S67, s1_0.x, s1_0.z, pb_0);
    }
    if(cut_0)
    {
        work_0[w_alive_0() + i_0] = u32(0);
        var _S69 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_19]), u32(1));
        var _S70 : u32 = a_counters_0();
        var _S71 : u32 = atomicAdd(&(atom_0[_S70]), u32(1));
        var _S72 : u32 = atomicAdd(&(atom_0[_S70 + u32(1)]), u32(1));
        var _S73 : u32 = atomicAdd(&(atom_0[_S70 + u32(13)]), u32(1));
    }
    return;
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

fn w_marks_0() -> u32
{
    return w_pulses_0() + u32(4) * params_0.chunk_count_0;
}

fn w_carry_0() -> u32
{
    return w_marks_0() + params_0.chunk_count_0;
}

fn to_rubble_0( v_7 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_7.x, v_7.z, - v_7.y);
}

fn a_il_0( b_20 : u32) -> u32
{
    return a_strike_0() + u32(23) * params_0.body_count_0 + u32(4) * b_20;
}

fn read64_0( at_5 : u32,  scale_0 : f32) -> f32
{
    var _S74 : u32 = atomicLoad(&(atom_0[at_5 + u32(1)]));
    var hi_0 : i32 = (bitcast<i32>((_S74)));
    var lo_0 : u32 = atomicLoad(&(atom_0[at_5]));
    return (f32(hi_0) * 4.294967296e+09f + f32(lo_0)) / scale_0;
}

fn il_carry_0( k_7 : u32)
{
    if(k_7 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S75 : u32 = w_carry_0() + k_7;
    var c_2 : u32 = work_0[_S75];
    if(work_0[_S75] == u32(0))
    {
        return;
    }
    work_0[_S75] = u32(0);
    var b0_0 : u32 = c_2 - u32(1);
    var _S76 : bool = il_struck_0(b0_0);
    if(!_S76)
    {
        return;
    }
    var contact_6 : vec3<f32>;
    var d_6 : vec3<f32>;
    var speed_5 : f32;
    il_path_1(b0_0, &(contact_6), &(d_6), &(speed_5));
    var dr_0 : vec3<f32> = to_rubble_0(d_6);
    var sk_0 : IlStriker_0 = il_striker_0(b0_0);
    var plug_mass_0 : f32 = read64_0(a_il_0(b0_0) + u32(2), params_0.scale_m_0);
    var v_wall_0 : vec3<f32> = bodies_0[u32(7) * b0_0 + u32(3)].xyz;
    var _S77 : u32 = u32(7) * sk_0.body_0;
    var v_shot_0 : vec3<f32> = bodies_0[_S77 + u32(3)].xyz;
    var closing_0 : f32 = dot(v_shot_0 - v_wall_0, dr_0);
    var v_r_0 : f32 = sk_0.mass_0 * closing_0 / (sk_0.mass_0 + plug_mass_0);
    var _S78 : vec3<f32> = v_shot_0 + dr_0 * vec3<f32>((v_r_0 - closing_0));
    var _S79 : vec3<f32> = v_wall_0 + dr_0 * vec3<f32>(v_r_0);
    var _S80 : vec3<f32> = bodies_0[_S77].xyz + dr_0 * vec3<f32>(((closing_0 - v_r_0) * clamp(il_t_in_0(b0_0), 0.0f, params_0.dt_0)));
    var _S81 : u32 = body_of_0(k_7);
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
        var s_at_0 : u32 = u32(7) * (sk_0.body_0 + region_0 * params_0.max_bodies_0);
        bodies_0[s_at_0].x = _S80.x;
        bodies_0[s_at_0].y = _S80.y;
        bodies_0[s_at_0].z = _S80.z;
        bodies_0[s_at_0 + u32(3)].x = _S78.x;
        bodies_0[s_at_0 + u32(3)].y = _S78.y;
        bodies_0[s_at_0 + u32(3)].z = _S78.z;
        bodies_0[u32(7) * (_S81 + region_0 * params_0.max_bodies_0) + u32(3)].x = _S79.x;
        bodies_0[u32(7) * (_S81 + region_0 * params_0.max_bodies_0) + u32(3)].y = _S79.y;
        bodies_0[u32(7) * (_S81 + region_0 * params_0.max_bodies_0) + u32(3)].z = _S79.z;
        region_0 = region_0 + u32(1);
    }
    return;
}

fn add64_0( at_6 : u32,  value_0 : f32,  scale_1 : f32)
{
    var x_1 : f32 = round(value_0 * scale_1);
    var x_2 : f32;
    if(!((abs(x_1)) < 1.073741824e+09f))
    {
        var _S82 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
        x_2 = 0.0f;
    }
    else
    {
        x_2 = x_1;
    }
    var v_8 : i32 = i32(x_2);
    var lo_1 : u32 = (bitcast<u32>((v_8)));
    var old_0 : u32 = atomicAdd(&(atom_0[at_6]), lo_1);
    var carry_0 : u32;
    if((old_0 + lo_1) < old_0)
    {
        carry_0 = u32(1);
    }
    else
    {
        carry_0 = u32(0);
    }
    var _S83 : u32;
    if(v_8 < i32(0))
    {
        _S83 = u32(4294967295);
    }
    else
    {
        _S83 = u32(0);
    }
    var hi_1 : u32 = _S83 + carry_0;
    if(hi_1 != u32(0))
    {
        var _S84 : u32 = atomicAdd(&(atom_0[at_6 + u32(1)]), hi_1);
    }
    return;
}

fn il_mark_0( k_8 : u32)
{
    if(k_8 >= (params_0.chunk_count_0))
    {
        return;
    }
    var b_21 : u32 = body_of_0(k_8);
    var _S85 : bool = il_struck_0(b_21);
    if(!_S85)
    {
        return;
    }
    var s0_1 : vec4<f32> = lp_0[lp_body_0(b_21)];
    var outcome_4 : u32 = (bitcast<u32>((lp_0[lp_body_0(b_21)].w)));
    if(((outcome_4 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_7 : vec3<f32>;
    var d_7 : vec3<f32>;
    var speed_6 : f32;
    il_path_1(b_21, &(contact_7), &(d_7), &(speed_6));
    var entry_3 : u32 = il_entry_0(b_21);
    var plug_0 : bool = il_plug_0(outcome_4, max(il_striker_0(b_21).radius_0, 0.00100000004749745f), s0_1.x, il_place_0(k_8, entry_3, contact_7, d_7));
    if(plug_0)
    {
        work_0[w_carry_0() + k_8] = b_21 + u32(1);
        add64_0(a_il_0(b_21) + u32(2), chunks_0[u32(4) * k_8].w, params_0.scale_m_0);
    }
    return;
}

struct JointMaterial_0
{
     strength_0 : vec4<f32>,
     energy_0 : vec4<f32>,
     dif_0 : vec4<f32>,
     misc_0 : vec4<f32>,
     kind_flags_0 : vec4<u32>,
};

fn fdiv_0( a_2 : f32,  b_22 : f32) -> f32
{
    return a_2 / b_22;
}

fn fpow_0( a_3 : f32,  b_23 : f32) -> f32
{
    return pow(a_3, b_23);
}

fn dif_factor_0( mat_0 : JointMaterial_0,  strain_rate_0 : f32) -> f32
{
    var r_14 : f32 = abs(strain_rate_0);
    var ref_0 : f32 = mat_0.dif_0.x;
    if(r_14 <= ref_0)
    {
        return 1.0f;
    }
    var _S86 : f32 = mat_0.dif_0.z;
    var f_0 : f32;
    if(r_14 <= _S86)
    {
        f_0 = fpow_0(fdiv_0(r_14, ref_0), mat_0.dif_0.y);
    }
    else
    {
        f_0 = fpow_0(fdiv_0(_S86, ref_0), mat_0.dif_0.y) * fpow_0(fdiv_0(r_14, _S86), mat_0.dif_0.w);
    }
    return clamp(f_0, 1.0f, mat_0.misc_0.x);
}

fn il_impulse_0( b_24 : u32,  sb_0 : u32,  m_shot_0 : f32,  contact_8 : vec3<f32>,  d_8 : vec3<f32>,  j_0 : f32)
{
    var jr_0 : vec3<f32> = to_rubble_0(d_8) * vec3<f32>(j_0);
    var at_7 : u32 = u32(7) * b_24;
    var inv_mass_1 : f32 = bodies_0[at_7].w;
    var _S87 : vec3<f32> = jr_0 * vec3<f32>(inv_mass_1);
    var _S88 : vec3<f32> = vec3<f32>(0.0f);
    var dw_0 : vec3<f32>;
    if(inv_mass_1 > 0.0f)
    {
        var q_7 : Quat_0 = quat_of_0(bodies_0[at_7 + u32(1)]);
        dw_0 = rotate_0(q_7, bodies_0[at_7 + u32(2)].xyz * rotate_0(conj_0(q_7), cross(to_rubble_0(contact_8) - bodies_0[at_7].xyz, jr_0)));
    }
    else
    {
        dw_0 = _S88;
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
        var s_at_1 : u32 = u32(7) * (sb_0 + region_1 * params_0.max_bodies_0) + u32(3);
        var _S89 : vec3<f32> = bodies_0[s_at_1].xyz - jr_0 / vec3<f32>(m_shot_0);
        bodies_0[s_at_1].x = _S89.x;
        bodies_0[s_at_1].y = _S89.y;
        bodies_0[s_at_1].z = _S89.z;
        var w_at_0 : u32 = u32(7) * (b_24 + region_1 * params_0.max_bodies_0);
        var _S90 : u32 = w_at_0 + u32(3);
        var _S91 : vec3<f32> = bodies_0[_S90].xyz + _S87;
        bodies_0[_S90].x = _S91.x;
        bodies_0[_S90].y = _S91.y;
        bodies_0[_S90].z = _S91.z;
        var _S92 : u32 = w_at_0 + u32(4);
        var _S93 : vec3<f32> = bodies_0[_S92].xyz + dw_0;
        bodies_0[_S92].x = _S93.x;
        bodies_0[_S92].y = _S93.y;
        bodies_0[_S92].z = _S93.z;
        region_1 = region_1 + u32(1);
    }
    return;
}

fn il_strike_0( b_25 : u32)
{
    var damage_0 : bool;
    if(b_25 >= (params_0.body_count_0))
    {
        damage_0 = true;
    }
    else
    {
        var _S94 : bool = il_struck_0(b_25);
        damage_0 = !_S94;
    }
    if(damage_0)
    {
        return;
    }
    var t_in_5 : f32 = il_t_in_0(b_25);
    var entry_4 : u32 = il_entry_0(b_25);
    var contact_9 : vec3<f32>;
    var d_9 : vec3<f32>;
    var speed_7 : f32;
    il_path_1(b_25, &(contact_9), &(d_9), &(speed_7));
    var sk_1 : IlStriker_0 = il_striker_0(b_25);
    var _S95 : f32 = max(sk_1.radius_0, 0.00100000004749745f);
    if((sk_1.launch_0) != u32(0))
    {
        damage_0 = work_0[w_marks_0() + entry_4] == (sk_1.launch_0);
    }
    else
    {
        damage_0 = false;
    }
    if(damage_0)
    {
        var _S96 : u32 = lp_body_0(b_25);
        var _S97 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S96] = _S97;
        lp_0[_S96 + u32(1)] = _S97;
        return;
    }
    var _S98 : u32 = a_il_0(b_25);
    var _S99 : u32 = atomicLoad(&(atom_0[_S98]));
    var _S100 : u32 = atomicLoad(&(atom_0[_S98 + u32(1)]));
    if(_S99 != _S100)
    {
        var _S101 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(16));
    }
    var _S102 : f32 = max((bitcast<f32>((_S99))) - _S95, 9.99999997475242708e-07f);
    var o_1 : u32 = params_0.ext_chunks_0 + u32(4) * entry_4;
    var mat0_0 : vec4<f32> = ext_0[o_1 + u32(1)];
    var dm_0 : JointMaterial_0;
    var _S103 : vec4<f32> = vec4<f32>(0.0f);
    dm_0.strength_0 = _S103;
    dm_0.energy_0 = _S103;
    dm_0.dif_0 = ext_0[o_1 + u32(2)];
    var mat3_0 : vec4<f32> = ext_0[o_1 + u32(3)];
    var _S104 : f32 = mat3_0.x;
    dm_0.misc_0 = vec4<f32>(_S104, 0.0f, 0.0f, 0.0f);
    dm_0.kind_flags_0 = vec4<u32>(u32(0));
    var nu_0 : f32 = mat3_0.y;
    var e_target_0 : f32 = mat3_0.z / (1.0f - nu_0 * nu_0);
    var e_star_0 : f32 = 1.0f / (1.0f / sk_1.modulus_0 + 1.0f / e_target_0);
    var _S105 : f32 = max(speed_7, 0.00100000004749745f);
    var _S106 : f32 = sqrt(_S95);
    var h0_0 : f32 = pow(15.0f * sk_1.m_eff_0 * _S105 * _S105 / (16.0f * e_star_0 * _S106), 0.40000000596046448f);
    var t_c_0 : f32 = 2.94000005722045898f * h0_0 / _S105;
    var force_0 : f32 = 1.33333337306976318f * e_star_0 * _S106 * pow(h0_0, 1.5f);
    var a_4 : f32 = sqrt(_S95 * h0_0);
    var p0_0 : f32 = 3.0f * force_0 / (6.28318548202514648f * a_4 * a_4);
    var ring_0 : f32 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    var k_9 : f32;
    if(_S104 > 0.0f)
    {
        k_9 = dif_factor_0(dm_0, _S105 / _S102);
    }
    else
    {
        k_9 = 1.0f;
    }
    var ft_0 : f32 = k_9 * mat0_0.y;
    var fc_0 : f32 = k_9 * mat0_0.z;
    var tau_0 : f32 = k_9 * mat0_0.w;
    if(p0_0 > fc_0)
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ring_0 > ft_0;
    }
    var _S107 : f32 = fc_0 * (3.14159274101257324f * _S95 * _S95);
    var _S108 : f32 = min(_S107 * _S102, 1.57079637050628662f * (2.0f * _S95 + _S102) * _S102 * _S102 * tau_0);
    var _S109 : f32 = 2.0f * _S108;
    var v_bl_0 : f32 = sqrt(_S109 / sk_1.m_eff_0);
    var perforate_0 : bool;
    if(damage_0)
    {
        perforate_0 = _S105 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    var r2_0 : f32 = _S95 + _S102;
    var _S110 : f32 = mat0_0.x;
    var plug_1 : f32 = _S110 * 3.14159274101257324f * _S102 / 3.0f * (_S95 * _S95 + _S95 * r2_0 + r2_0 * r2_0);
    var v_res_0 : f32;
    if(perforate_0)
    {
        v_res_0 = sk_1.mass_0 / (sk_1.mass_0 + plug_1) * sqrt(max(_S105 * _S105 - v_bl_0 * v_bl_0, 0.0f));
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
    var depth_1 : f32;
    if(spall_0)
    {
        depth_1 = min(0.5f * sk_1.m_eff_0 * _S105 * _S105 / _S107, _S102);
    }
    else
    {
        depth_1 = 0.0f;
    }
    var crater_0 : bool = depth_1 > a_4;
    var c_3 : f32 = sqrt(e_target_0 / _S110);
    var z_t_0 : f32 = _S110 * c_3;
    var rho_p_0 : f32 = sk_1.mass_0 / (4.18879032135009766f * _S95 * _S95 * _S95);
    var z_p_0 : f32 = rho_p_0 * sqrt(sk_1.modulus_0 / rho_p_0);
    var mean_pressure_0 : f32 = force_0 / (3.14159274101257324f * a_4 * a_4);
    var _S111 : f32 = min(min(z_t_0 * _S105 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_0);
    var source_radius_0 : f32;
    if(mean_pressure_0 > fc_0)
    {
        source_radius_0 = max(a_4, _S95);
    }
    else
    {
        source_radius_0 = a_4;
    }
    var pulse_0 : f32 = c_3 * t_c_0;
    var back_0 : f32 = _S111 * min(source_radius_0 / _S102, 1.0f) * min(2.0f * _S102 / pulse_0, 1.0f);
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
    var scab_1 : f32;
    if(spall_0)
    {
        scab_1 = 0.5f * min(pulse_0, _S102);
    }
    else
    {
        scab_1 = 0.0f;
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
    var _S112 : u32;
    if(crater_0)
    {
        _S112 = u32(2);
    }
    else
    {
        _S112 = u32(0);
    }
    var _S113 : u32 = (region_2 | (_S112));
    if(spall_0)
    {
        region_2 = u32(4);
    }
    else
    {
        region_2 = u32(0);
    }
    var _S114 : u32 = (_S113 | (region_2));
    if(perforate_0)
    {
        region_2 = u32(8);
    }
    else
    {
        region_2 = u32(0);
    }
    var _S115 : u32 = lp_body_0(b_25);
    lp_0[_S115] = vec4<f32>(_S102, v_bl_0, v_res_0, (bitcast<f32>(((_S114 | (region_2))))));
    if(crater_0)
    {
        k_9 = depth_1;
    }
    else
    {
        k_9 = 0.0f;
    }
    lp_0[_S115 + u32(1)] = vec4<f32>(k_9, scab_1, a_4, _S108);
    if(perforate_0)
    {
        var _S116 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(10)]), u32(1));
    }
    else
    {
        if(spall_0)
        {
            var _S117 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(9)]), u32(1));
        }
        else
        {
            if(crater_0)
            {
                var _S118 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(8)]), u32(1));
            }
            else
            {
                var _S119 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(7)]), u32(1));
            }
        }
    }
    if(perforate_0)
    {
        if((params_0.fracture_work_0) != u32(0))
        {
            il_impulse_0(b_25, sk_1.body_0, sk_1.mass_0, contact_9, d_9, sk_1.m_eff_0 * (_S105 - sqrt(max(_S105 * _S105 - _S109 / sk_1.m_eff_0, 0.0f))));
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_25, sk_1.body_0, sk_1.mass_0, contact_9, d_9, sk_1.m_eff_0 * _S105);
            var _S120 : vec3<f32> = to_rubble_0(d_9) * vec3<f32>((_S105 * clamp(t_in_5, 0.0f, params_0.dt_0)));
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
                var _S121 : u32 = u32(7) * (sk_1.body_0 + region_2 * params_0.max_bodies_0);
                var _S122 : vec3<f32> = bodies_0[_S121].xyz + _S120;
                bodies_0[_S121].x = _S122.x;
                bodies_0[_S121].y = _S122.y;
                bodies_0[_S121].z = _S122.z;
                region_2 = region_2 + u32(1);
            }
        }
    }
    return;
}

fn structure_pose_1( slot_2 : u32,  b_26 : u32,  q_8 : ptr<function, Quat_0>,  origin_2 : ptr<function, vec3<f32>>)
{
    var _S123 : u32 = u32(2) * slot_2;
    var com_1 : vec3<f32> = frames_0[_S123].xyz;
    var _S124 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_26 + u32(1)])), conj_0(quat_of_0(frames_0[_S123 + u32(1)]))));
    (*q_8) = _S124;
    (*origin_2) = from_rubble_0(bodies_0[b_26].xyz) - rotate_0(_S124, com_1);
    return;
}

fn chunk_box_world_0( k_10 : u32,  q_9 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_3 : u32 = body_of_0(k_10);
    var origin_3 : vec3<f32>;
    structure_pose_1(slot_3, u32(7) * slot_3, &((*q_9)), &(origin_3));
    return origin_3 + rotate_0((*q_9), geometry_0[u32(3) * k_10].xyz);
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

fn il_interval_0( k_11 : u32,  p_9 : vec3<f32>,  d_10 : vec3<f32>,  pad_0 : f32) -> vec2<f32>
{
    var q_10 : Quat_0;
    var _S125 : vec3<f32> = chunk_box_world_0(k_11, &(q_10));
    var _S126 : u32 = u32(3) * k_11;
    var _S127 : Quat_0 = normalized_0(quat_mul_0(q_10, quat_of_0(geometry_0[_S126 + u32(2)])));
    var _S128 : vec3<f32> = geometry_0[_S126 + u32(1)].xyz;
    var inf_0 : f32 = float_bits_0(u32(2139095040));
    var t0_0 : f32 = - inf_0;
    var t1_0 : f32 = inf_0;
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
        var e_0 : vec3<f32> = vec3<f32>(0.0f);
        e_0[ax_0] = 1.0f;
        var axis_0 : vec3<f32> = rotate_0(_S127, e_0);
        var o_2 : f32 = dot(p_9 - _S125, axis_0);
        var dv_0 : f32 = dot(d_10, axis_0);
        var ext_1 : f32 = _S128[ax_0] + pad_0;
        if((abs(dv_0)) < 9.99999971718068537e-10f)
        {
            if((abs(o_2)) > ext_1)
            {
                return vec2<f32>(1.0f, 0.0f);
            }
            ax_0 = ax_0 + u32(1);
            continue;
        }
        var ta_0 : f32 = (- ext_1 - o_2) / dv_0;
        var tb_0 : f32 = (ext_1 - o_2) / dv_0;
        var _S129 : f32 = min(t1_0, max(ta_0, tb_0));
        t0_0 = max(t0_0, min(ta_0, tb_0));
        t1_0 = _S129;
        ax_0 = ax_0 + u32(1);
    }
    return vec2<f32>(t0_0, t1_0);
}

fn il_extend_0( k_12 : u32,  from_0 : u32)
{
    if(k_12 >= (params_0.chunk_count_0))
    {
        return;
    }
    var b_27 : u32 = body_of_0(k_12);
    var _S130 : bool = il_struck_0(b_27);
    if(!_S130)
    {
        return;
    }
    var contact_10 : vec3<f32>;
    var d_11 : vec3<f32>;
    var speed_8 : f32;
    il_path_0(b_27, &(contact_10), &(d_11), &(speed_8));
    var _S131 : IlStriker_0 = il_striker_0(b_27);
    var _S132 : u32 = a_il_0(b_27);
    var _S133 : u32 = atomicLoad(&(atom_0[_S132 + from_0]));
    var end_0 : f32 = (bitcast<f32>((_S133)));
    var to_0 : u32 = _S132 + u32(1) - from_0;
    var _S134 : u32 = atomicMax(&(atom_0[to_0]), (bitcast<u32>((end_0))));
    var t_1 : vec2<f32> = il_interval_0(k_12, contact_10, d_11, _S131.radius_0);
    var _S135 : f32 = t_1.x;
    var _S136 : f32 = t_1.y;
    var _S137 : bool;
    if(_S135 <= _S136)
    {
        _S137 = _S135 <= (end_0 + 0.00100000004749745f + 0.01999999955296516f * _S131.radius_0);
    }
    else
    {
        _S137 = false;
    }
    if(_S137)
    {
        _S137 = _S136 > end_0;
    }
    else
    {
        _S137 = false;
    }
    if(_S137)
    {
        var _S138 : u32 = atomicMax(&(atom_0[to_0]), (bitcast<u32>((_S136))));
    }
    return;
}

fn il_begin_0( b_28 : u32)
{
    var _S139 : bool;
    if(b_28 >= (params_0.body_count_0))
    {
        _S139 = true;
    }
    else
    {
        var _S140 : bool = il_struck_0(b_28);
        _S139 = !_S140;
    }
    if(_S139)
    {
        return;
    }
    var contact_11 : vec3<f32>;
    var d_12 : vec3<f32>;
    var speed_9 : f32;
    il_path_1(b_28, &(contact_11), &(d_12), &(speed_9));
    var _S141 : u32 = il_entry_0(b_28);
    var end_1 : u32 = (bitcast<u32>((max(il_interval_0(_S141, contact_11, d_12, il_striker_0(b_28).radius_0).y, 0.0f))));
    var _S142 : u32 = a_il_0(b_28);
    atomicStore(&(atom_0[_S142]), end_1);
    atomicStore(&(atom_0[_S142 + u32(1)]), end_1);
    atomicStore(&(atom_0[_S142 + u32(2)]), u32(0));
    atomicStore(&(atom_0[_S142 + u32(3)]), u32(0));
    return;
}

fn il_breaks_1( i_1 : u32)
{
    var cut_1 : bool;
    if(i_1 >= (params_0.bond_count_0))
    {
        cut_1 = true;
    }
    else
    {
        cut_1 = work_0[w_alive_0() + i_1] == u32(0);
    }
    if(cut_1)
    {
        return;
    }
    var bond_1 : vec4<u32> = bonds_0[i_1];
    var _S143 : u32 = bond_1.x;
    var b_29 : u32 = body_of_0(_S143);
    if(!il_struck_1(b_29))
    {
        return;
    }
    var _S144 : u32 = lp_body_0(b_29);
    var s0_2 : vec4<f32> = lp_0[_S144];
    var s1_1 : vec4<f32> = lp_0[_S144 + u32(1)];
    var outcome_5 : u32 = (bitcast<u32>((lp_0[_S144].w)));
    if(((outcome_5 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_12 : vec3<f32>;
    var d_13 : vec3<f32>;
    var speed_10 : f32;
    il_path_3(b_29, &(contact_12), &(d_13), &(speed_10));
    var entry_5 : u32 = il_entry_1(b_29);
    var _S145 : f32 = max(il_striker_1(b_29).radius_0, 0.00100000004749745f);
    var h_3 : f32 = s0_2.x;
    var pa_1 : IlPlace_0 = il_place_0(_S143, entry_5, contact_12, d_13);
    var pb_1 : IlPlace_0 = il_place_0(bond_1.y, entry_5, contact_12, d_13);
    if((il_plug_0(outcome_5, _S145, h_3, pa_1)) != (il_plug_0(outcome_5, _S145, h_3, pb_1)))
    {
        cut_1 = true;
    }
    else
    {
        var _S146 : f32 = s1_1.y;
        cut_1 = (il_scab_0(outcome_5, _S145, h_3, _S146, pa_1)) != (il_scab_0(outcome_5, _S145, h_3, _S146, pb_1));
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        cut_1 = il_crater_0(outcome_5, _S145, s1_1.x, s1_1.z, pa_1);
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        cut_1 = il_crater_0(outcome_5, _S145, s1_1.x, s1_1.z, pb_1);
    }
    if(cut_1)
    {
        work_0[w_alive_0() + i_1] = u32(0);
        var _S147 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_29]), u32(1));
        var _S148 : u32 = a_counters_0();
        var _S149 : u32 = atomicAdd(&(atom_0[_S148]), u32(1));
        var _S150 : u32 = atomicAdd(&(atom_0[_S148 + u32(1)]), u32(1));
    }
    return;
}

fn il_carry_1( k_13 : u32)
{
    if(k_13 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S151 : u32 = w_carry_0() + k_13;
    var c_4 : u32 = work_0[_S151];
    if(work_0[_S151] == u32(0))
    {
        return;
    }
    work_0[_S151] = u32(0);
    var b0_1 : u32 = c_4 - u32(1);
    if(!il_struck_1(b0_1))
    {
        return;
    }
    var contact_13 : vec3<f32>;
    var d_14 : vec3<f32>;
    var speed_11 : f32;
    il_path_3(b0_1, &(contact_13), &(d_14), &(speed_11));
    var dr_1 : vec3<f32> = to_rubble_0(d_14);
    var sk_2 : IlStriker_0 = il_striker_1(b0_1);
    var plug_mass_1 : f32 = read64_0(a_il_0(b0_1) + u32(2), params_0.scale_m_0);
    var v_wall_1 : vec3<f32> = bodies_0[u32(7) * b0_1 + u32(3)].xyz;
    var _S152 : u32 = u32(7) * sk_2.body_0;
    var v_shot_1 : vec3<f32> = bodies_0[_S152 + u32(3)].xyz;
    var closing_1 : f32 = dot(v_shot_1 - v_wall_1, dr_1);
    var v_r_1 : f32 = sk_2.mass_0 * closing_1 / (sk_2.mass_0 + plug_mass_1);
    var _S153 : vec3<f32> = v_shot_1 + dr_1 * vec3<f32>((v_r_1 - closing_1));
    var _S154 : vec3<f32> = v_wall_1 + dr_1 * vec3<f32>(v_r_1);
    var _S155 : f32 = il_t_in_1(b0_1);
    var _S156 : vec3<f32> = bodies_0[_S152].xyz + dr_1 * vec3<f32>(((closing_1 - v_r_1) * clamp(_S155, 0.0f, params_0.dt_0)));
    var _S157 : u32 = body_of_0(k_13);
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
        var s_at_2 : u32 = u32(7) * (sk_2.body_0 + region_3 * params_0.max_bodies_0);
        bodies_0[s_at_2].x = _S156.x;
        bodies_0[s_at_2].y = _S156.y;
        bodies_0[s_at_2].z = _S156.z;
        bodies_0[s_at_2 + u32(3)].x = _S153.x;
        bodies_0[s_at_2 + u32(3)].y = _S153.y;
        bodies_0[s_at_2 + u32(3)].z = _S153.z;
        bodies_0[u32(7) * (_S157 + region_3 * params_0.max_bodies_0) + u32(3)].x = _S154.x;
        bodies_0[u32(7) * (_S157 + region_3 * params_0.max_bodies_0) + u32(3)].y = _S154.y;
        bodies_0[u32(7) * (_S157 + region_3 * params_0.max_bodies_0) + u32(3)].z = _S154.z;
        region_3 = region_3 + u32(1);
    }
    return;
}

fn il_mark_1( k_14 : u32)
{
    if(k_14 >= (params_0.chunk_count_0))
    {
        return;
    }
    var b_30 : u32 = body_of_0(k_14);
    if(!il_struck_1(b_30))
    {
        return;
    }
    var _S158 : u32 = lp_body_0(b_30);
    var s0_3 : vec4<f32> = lp_0[_S158];
    var s1_2 : vec4<f32> = lp_0[_S158 + u32(1)];
    var outcome_6 : u32 = (bitcast<u32>((lp_0[_S158].w)));
    if(((outcome_6 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_14 : vec3<f32>;
    var d_15 : vec3<f32>;
    var speed_12 : f32;
    il_path_3(b_30, &(contact_14), &(d_15), &(speed_12));
    var entry_6 : u32 = il_entry_1(b_30);
    var sk_3 : IlStriker_0 = il_striker_1(b_30);
    var _S159 : f32 = max(sk_3.radius_0, 0.00100000004749745f);
    var p_10 : IlPlace_0 = il_place_0(k_14, entry_6, contact_14, d_15);
    var _S160 : f32 = s0_3.x;
    var plug_2 : bool = il_plug_0(outcome_6, _S159, _S160, p_10);
    var t_2 : vec2<f32> = il_interval_0(k_14, contact_14 - d_15 * vec3<f32>(sk_3.bound_0), d_15, sk_3.bound_0);
    var _S161 : f32 = t_2.x;
    var _S162 : f32 = t_2.y;
    var path_0 : bool;
    if(_S161 <= _S162)
    {
        path_0 = _S161 <= (_S160 + 2.0f * sk_3.bound_0);
    }
    else
    {
        path_0 = false;
    }
    if(path_0)
    {
        path_0 = _S162 >= 0.0f;
    }
    else
    {
        path_0 = false;
    }
    if(plug_2)
    {
        path_0 = true;
    }
    if(path_0)
    {
        path_0 = true;
    }
    else
    {
        path_0 = il_scab_0(outcome_6, _S159, _S160, s1_2.y, p_10);
    }
    if(path_0)
    {
        path_0 = true;
    }
    else
    {
        path_0 = il_crater_0(outcome_6, _S159, s1_2.x, s1_2.z, p_10);
    }
    if(path_0)
    {
        work_0[w_marks_0() + k_14] = sk_3.launch_0;
    }
    if(plug_2)
    {
        work_0[w_carry_0() + k_14] = b_30 + u32(1);
        add64_0(a_il_0(b_30) + u32(2), chunks_0[u32(4) * k_14].w, params_0.scale_m_0);
    }
    return;
}

fn il_strike_1( b_31 : u32)
{
    var damage_1 : bool;
    if(b_31 >= (params_0.body_count_0))
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = !il_struck_1(b_31);
    }
    if(damage_1)
    {
        return;
    }
    var t_in_6 : f32 = il_t_in_1(b_31);
    var entry_7 : u32 = il_entry_1(b_31);
    var contact_15 : vec3<f32>;
    var d_16 : vec3<f32>;
    var speed_13 : f32;
    il_path_3(b_31, &(contact_15), &(d_16), &(speed_13));
    var sk_4 : IlStriker_0 = il_striker_1(b_31);
    var _S163 : f32 = max(sk_4.radius_0, 0.00100000004749745f);
    if((sk_4.launch_0) != u32(0))
    {
        damage_1 = work_0[w_marks_0() + entry_7] == (sk_4.launch_0);
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        var _S164 : u32 = lp_body_0(b_31);
        var _S165 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S164] = _S165;
        lp_0[_S164 + u32(1)] = _S165;
        return;
    }
    var _S166 : u32 = a_il_0(b_31);
    var _S167 : u32 = atomicLoad(&(atom_0[_S166]));
    var _S168 : u32 = atomicLoad(&(atom_0[_S166 + u32(1)]));
    if(_S167 != _S168)
    {
        var _S169 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(16));
    }
    var _S170 : f32 = max((bitcast<f32>((_S167))) - _S163, 9.99999997475242708e-07f);
    if(!(t_in_6 > 0.0f))
    {
        var _S171 : u32 = lp_body_0(b_31);
        lp_0[_S171] = vec4<f32>(_S170, 0.0f, 0.0f, 0.0f);
        lp_0[_S171 + u32(1)] = vec4<f32>(0.0f);
        return;
    }
    var o_3 : u32 = params_0.ext_chunks_0 + u32(4) * entry_7;
    var mat0_1 : vec4<f32> = ext_0[o_3 + u32(1)];
    var dm_1 : JointMaterial_0;
    var _S172 : vec4<f32> = vec4<f32>(0.0f);
    dm_1.strength_0 = _S172;
    dm_1.energy_0 = _S172;
    dm_1.dif_0 = ext_0[o_3 + u32(2)];
    var mat3_1 : vec4<f32> = ext_0[o_3 + u32(3)];
    var _S173 : f32 = mat3_1.x;
    dm_1.misc_0 = vec4<f32>(_S173, 0.0f, 0.0f, 0.0f);
    dm_1.kind_flags_0 = vec4<u32>(u32(0));
    var nu_1 : f32 = mat3_1.y;
    var e_target_1 : f32 = mat3_1.z / (1.0f - nu_1 * nu_1);
    var e_star_1 : f32 = 1.0f / (1.0f / sk_4.modulus_0 + 1.0f / e_target_1);
    var _S174 : f32 = max(speed_13, 0.00100000004749745f);
    var _S175 : f32 = sqrt(_S163);
    var h0_1 : f32 = pow(15.0f * sk_4.m_eff_0 * _S174 * _S174 / (16.0f * e_star_1 * _S175), 0.40000000596046448f);
    var t_c_1 : f32 = 2.94000005722045898f * h0_1 / _S174;
    var force_1 : f32 = 1.33333337306976318f * e_star_1 * _S175 * pow(h0_1, 1.5f);
    var a_5 : f32 = sqrt(_S163 * h0_1);
    var p0_1 : f32 = 3.0f * force_1 / (6.28318548202514648f * a_5 * a_5);
    var ring_1 : f32 = (1.0f - 2.0f * nu_1) * p0_1 / 3.0f;
    var k_15 : f32;
    if(_S173 > 0.0f)
    {
        k_15 = dif_factor_0(dm_1, _S174 / _S170);
    }
    else
    {
        k_15 = 1.0f;
    }
    var ft_1 : f32 = k_15 * mat0_1.y;
    var fc_1 : f32 = k_15 * mat0_1.z;
    var tau_1 : f32 = k_15 * mat0_1.w;
    if(p0_1 > fc_1)
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = ring_1 > ft_1;
    }
    var _S176 : f32 = fc_1 * (3.14159274101257324f * _S163 * _S163);
    var _S177 : f32 = min(_S176 * _S170, 1.57079637050628662f * (2.0f * _S163 + _S170) * _S170 * _S170 * tau_1);
    var _S178 : f32 = 2.0f * _S177;
    var v_bl_1 : f32 = sqrt(_S178 / sk_4.m_eff_0);
    var perforate_1 : bool;
    if(damage_1)
    {
        perforate_1 = _S174 >= v_bl_1;
    }
    else
    {
        perforate_1 = false;
    }
    var r2_1 : f32 = _S163 + _S170;
    var _S179 : f32 = mat0_1.x;
    var plug_3 : f32 = _S179 * 3.14159274101257324f * _S170 / 3.0f * (_S163 * _S163 + _S163 * r2_1 + r2_1 * r2_1);
    var v_res_1 : f32;
    if(perforate_1)
    {
        v_res_1 = sk_4.mass_0 / (sk_4.mass_0 + plug_3) * sqrt(max(_S174 * _S174 - v_bl_1 * v_bl_1, 0.0f));
    }
    else
    {
        v_res_1 = 0.0f;
    }
    var spall_1 : bool;
    if(damage_1)
    {
        spall_1 = !perforate_1;
    }
    else
    {
        spall_1 = false;
    }
    var depth_2 : f32;
    if(spall_1)
    {
        depth_2 = min(0.5f * sk_4.m_eff_0 * _S174 * _S174 / _S176, _S170);
    }
    else
    {
        depth_2 = 0.0f;
    }
    var crater_1 : bool = depth_2 > a_5;
    var c_5 : f32 = sqrt(e_target_1 / _S179);
    var z_t_1 : f32 = _S179 * c_5;
    var rho_p_1 : f32 = sk_4.mass_0 / (4.18879032135009766f * _S163 * _S163 * _S163);
    var z_p_1 : f32 = rho_p_1 * sqrt(sk_4.modulus_0 / rho_p_1);
    var mean_pressure_1 : f32 = force_1 / (3.14159274101257324f * a_5 * a_5);
    var _S180 : f32 = min(min(z_t_1 * _S174 * z_p_1 / (z_p_1 + z_t_1), mean_pressure_1), fc_1);
    var source_radius_1 : f32;
    if(mean_pressure_1 > fc_1)
    {
        source_radius_1 = max(a_5, _S163);
    }
    else
    {
        source_radius_1 = a_5;
    }
    var pulse_1 : f32 = c_5 * t_c_1;
    var back_1 : f32 = _S180 * min(source_radius_1 / _S170, 1.0f) * min(2.0f * _S170 / pulse_1, 1.0f);
    if(damage_1)
    {
        spall_1 = !perforate_1;
    }
    else
    {
        spall_1 = false;
    }
    if(spall_1)
    {
        spall_1 = back_1 > ft_1;
    }
    else
    {
        spall_1 = false;
    }
    var scab_2 : f32;
    if(spall_1)
    {
        scab_2 = 0.5f * min(pulse_1, _S170);
    }
    else
    {
        scab_2 = 0.0f;
    }
    var region_4 : u32;
    if(damage_1)
    {
        region_4 = u32(1);
    }
    else
    {
        region_4 = u32(0);
    }
    var _S181 : u32;
    if(crater_1)
    {
        _S181 = u32(2);
    }
    else
    {
        _S181 = u32(0);
    }
    var _S182 : u32 = (region_4 | (_S181));
    if(spall_1)
    {
        region_4 = u32(4);
    }
    else
    {
        region_4 = u32(0);
    }
    var _S183 : u32 = (_S182 | (region_4));
    if(perforate_1)
    {
        region_4 = u32(8);
    }
    else
    {
        region_4 = u32(0);
    }
    var _S184 : u32 = lp_body_0(b_31);
    lp_0[_S184] = vec4<f32>(_S170, v_bl_1, v_res_1, (bitcast<f32>(((_S183 | (region_4))))));
    if(crater_1)
    {
        k_15 = depth_2;
    }
    else
    {
        k_15 = 0.0f;
    }
    lp_0[_S184 + u32(1)] = vec4<f32>(k_15, scab_2, a_5, _S177);
    if(perforate_1)
    {
        var _S185 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(10)]), u32(1));
    }
    else
    {
        if(spall_1)
        {
            var _S186 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(9)]), u32(1));
        }
        else
        {
            if(crater_1)
            {
                var _S187 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(8)]), u32(1));
            }
            else
            {
                var _S188 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(7)]), u32(1));
            }
        }
    }
    if(perforate_1)
    {
        if((params_0.fracture_work_0) != u32(0))
        {
            il_impulse_0(b_31, sk_4.body_0, sk_4.mass_0, contact_15, d_16, sk_4.m_eff_0 * (_S174 - sqrt(max(_S174 * _S174 - _S178 / sk_4.m_eff_0, 0.0f))));
        }
    }
    else
    {
        if(damage_1)
        {
            il_impulse_0(b_31, sk_4.body_0, sk_4.mass_0, contact_15, d_16, sk_4.m_eff_0 * _S174);
            var _S189 : vec3<f32> = to_rubble_0(d_16) * vec3<f32>((_S174 * clamp(t_in_6, 0.0f, params_0.dt_0)));
            region_4 = u32(0);
            loop
            {
                if(region_4 < u32(2))
                {
                }
                else
                {
                    break;
                }
                var _S190 : u32 = u32(7) * (sk_4.body_0 + region_4 * params_0.max_bodies_0);
                var _S191 : vec3<f32> = bodies_0[_S190].xyz + _S189;
                bodies_0[_S190].x = _S191.x;
                bodies_0[_S190].y = _S191.y;
                bodies_0[_S190].z = _S191.z;
                region_4 = region_4 + u32(1);
            }
        }
    }
    return;
}

fn il_extend_1( k_16 : u32,  from_1 : u32)
{
    if(k_16 >= (params_0.chunk_count_0))
    {
        return;
    }
    var b_32 : u32 = body_of_0(k_16);
    if(!il_struck_1(b_32))
    {
        return;
    }
    var contact_16 : vec3<f32>;
    var d_17 : vec3<f32>;
    var speed_14 : f32;
    il_path_2(b_32, &(contact_16), &(d_17), &(speed_14));
    var _S192 : IlStriker_0 = il_striker_1(b_32);
    var _S193 : u32 = a_il_0(b_32);
    var _S194 : u32 = atomicLoad(&(atom_0[_S193 + from_1]));
    var end_2 : f32 = (bitcast<f32>((_S194)));
    var to_1 : u32 = _S193 + u32(1) - from_1;
    var _S195 : u32 = atomicMax(&(atom_0[to_1]), (bitcast<u32>((end_2))));
    var t_3 : vec2<f32> = il_interval_0(k_16, contact_16, d_17, _S192.radius_0);
    var _S196 : f32 = t_3.x;
    var _S197 : f32 = t_3.y;
    var _S198 : bool;
    if(_S196 <= _S197)
    {
        _S198 = _S196 <= (end_2 + 0.00100000004749745f + 0.01999999955296516f * _S192.radius_0);
    }
    else
    {
        _S198 = false;
    }
    if(_S198)
    {
        _S198 = _S197 > end_2;
    }
    else
    {
        _S198 = false;
    }
    if(_S198)
    {
        var _S199 : u32 = atomicMax(&(atom_0[to_1]), (bitcast<u32>((_S197))));
    }
    return;
}

fn struck_1( b_33 : u32,  shot_3 : ptr<function, u32>) -> bool
{
    var k_17 : u32 = u32(frames_0[strike_at_0(b_33)].w);
    (*shot_3) = k_17 - u32(1);
    return k_17 != u32(0);
}

fn il_begin_1( b_34 : u32)
{
    var _S200 : bool;
    if(b_34 >= (params_0.body_count_0))
    {
        _S200 = true;
    }
    else
    {
        _S200 = !il_struck_1(b_34);
    }
    if(_S200)
    {
        return;
    }
    var shot_4 : u32;
    var _S201 : bool = struck_1(b_34, &(shot_4));
    var _S202 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_34 + u32(6)]));
    var _S203 : u32 = atomicLoad(&(atom_0[a_first_0(shot_4)]));
    if(_S202 != _S203)
    {
        frames_0[strike_at_0(b_34)][i32(3)] = 0.0f;
        return;
    }
    var contact_17 : vec3<f32>;
    var d_18 : vec3<f32>;
    var speed_15 : f32;
    il_path_3(b_34, &(contact_17), &(d_18), &(speed_15));
    var _S204 : u32 = il_entry_1(b_34);
    var end_3 : u32 = (bitcast<u32>((max(il_interval_0(_S204, contact_17, d_18, il_striker_1(b_34).radius_0).y, 0.0f))));
    var _S205 : u32 = a_il_0(b_34);
    atomicStore(&(atom_0[_S205]), end_3);
    atomicStore(&(atom_0[_S205 + u32(1)]), end_3);
    atomicStore(&(atom_0[_S205 + u32(2)]), u32(0));
    atomicStore(&(atom_0[_S205 + u32(3)]), u32(0));
    return;
}

fn isinf_0( x_3 : f32) -> bool
{
    var _S206 : u32 = (bitcast<u32>((x_3)));
    var _S207 : u32 = (_S206 & (u32(8388607)));
    var _S208 : bool;
    if(((((_S206 >> (u32(23)))) & (u32(255)))) == u32(255))
    {
        _S208 = _S207 == u32(0);
    }
    else
    {
        _S208 = false;
    }
    return _S208;
}

fn structure_pose_2( slot_4 : u32,  b_35 : u32,  q_11 : ptr<function, Quat_0>,  origin_4 : ptr<function, vec3<f32>>)
{
    var _S209 : u32 = u32(2) * slot_4;
    var com_2 : vec3<f32> = frames_0[_S209].xyz;
    var _S210 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_35 + u32(1)])), conj_0(quat_of_0(frames_0[_S209 + u32(1)]))));
    (*q_11) = _S210;
    (*origin_4) = from_rubble_0(bodies_0[b_35].xyz) - rotate_0(_S210, com_2);
    return;
}

fn structure_pose_3( slot_5 : u32,  b_36 : u32,  q_12 : ptr<function, Quat_0>,  origin_5 : ptr<function, vec3<f32>>)
{
    var _S211 : u32 = u32(2) * slot_5;
    var com_3 : vec3<f32> = frames_0[_S211].xyz;
    var _S212 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_36 + u32(1)])), conj_0(quat_of_0(frames_0[_S211 + u32(1)]))));
    (*q_12) = _S212;
    (*origin_5) = from_rubble_0(bodies_0[b_36].xyz) - rotate_0(_S212, com_3);
    return;
}

fn structure_pose_4( slot_6 : u32,  b_37 : u32,  q_13 : ptr<function, Quat_0>,  origin_6 : ptr<function, vec3<f32>>)
{
    var _S213 : u32 = u32(2) * slot_6;
    var com_4 : vec3<f32> = frames_0[_S213].xyz;
    var _S214 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_37 + u32(1)])), conj_0(quat_of_0(frames_0[_S213 + u32(1)]))));
    (*q_13) = _S214;
    (*origin_6) = from_rubble_0(bodies_0[b_37].xyz) - rotate_0(_S214, com_4);
    return;
}

fn l_pool_0( half_0 : u32) -> u32
{
    return params_0.chunk_count_0 + u32(3) * params_0.pool_cap_0 * half_0;
}

fn pulse_mean_0( start_0 : f32,  plateau_0 : f32,  plateau_time_0 : f32,  peak_0 : f32,  duration_0 : f32,  dt_1 : f32) -> f32
{
    var _S215 : f32 = start_0 + plateau_time_0;
    var sum_0 : f32 = plateau_0 * max(min(_S215, dt_1) - max(start_0, 0.0f), 0.0f);
    var _S216 : f32 = max(0.0f, _S215);
    var _S217 : f32 = min(dt_1, _S215 + duration_0);
    var _S218 : bool;
    if(_S217 > _S216)
    {
        _S218 = duration_0 > 0.0f;
    }
    else
    {
        _S218 = false;
    }
    var sum_1 : f32;
    if(_S218)
    {
        var w_2 : f32 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S216 - _S215)) - cos(w_2 * (_S217 - _S215)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

fn hertz_duration_0( m_0 : f32,  r_15 : f32,  e_1 : f32,  v_9 : f32) -> f32
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_15 * e_1 * e_1 * max(abs(v_9), 0.00100000004749745f)), 0.20000000298023224f);
}

fn l_vec_0() -> u32
{
    return params_0.chunk_count_0 + u32(6) * params_0.pool_cap_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_loads(@builtin(global_invocation_id) id_4 : vec3<u32>)
{
    var f_total_0 : vec3<f32>;
    var t_total_0 : vec3<f32>;
    var m_1 : f32;
    var p_11 : u32;
    var kept_0 : u32;
    var q3_0 : u32;
    var overflow_0 : bool;
    var overflow_1 : bool;
    var f_total_1 : vec3<f32>;
    var t_total_1 : vec3<f32>;
    var k_18 : u32 = id_4.x;
    if(k_18 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_7 : u32 = body_of_0(k_18);
    var b_38 : u32 = u32(7) * slot_7;
    var q_14 : Quat_0;
    var origin_7 : vec3<f32>;
    structure_pose_4(slot_7, b_38, &(q_14), &(origin_7));
    var _S219 : u32 = u32(4) * k_18;
    var c0_0 : vec4<f32> = chunks_0[_S219];
    var centre_0 : vec3<f32> = origin_7 + rotate_0(q_14, c0_0.xyz);
    var inv_mass_2 : f32 = bodies_0[b_38].w;
    var m_body_0 : f32;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(u32(2139095040));
    }
    var from_2 : u32 = params_0.parity_0;
    var to_2 : u32 = u32(1) - params_0.parity_0;
    var _S220 : u32 = w_pulses_0() + _S219;
    var _S221 : u32 = _S220 + u32(2) * params_0.parity_0;
    var off_0 : u32 = work_0[_S221];
    var carried_0 : u32 = work_0[_S221 + u32(1)];
    var room_0 : u32 = work_0[_S221 + u32(1)] + sort_0[params_0.chunk_count_0 + u32(1) + k_18];
    var base_0 : u32;
    var room_1 : u32;
    if(room_0 != u32(0))
    {
        var _S222 : u32 = a_counters_0();
        var base_1 : u32 = atomicAdd(&(atom_0[_S222 + u32(5) + to_2]), room_0);
        if((base_1 + room_0) > (params_0.pool_cap_0))
        {
            var _S223 : u32 = atomicOr(&(atom_0[_S222 + u32(3)]), u32(8));
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
        var _S224 : u32 = base_0;
        base_0 = base_1;
        room_1 = _S224;
    }
    else
    {
        base_0 = u32(0);
        room_1 = room_0;
    }
    var _S225 : u32 = params_0.chunk_count_0;
    var _S226 : f32 = c0_0.w;
    var _S227 : vec4<f32> = chunks_0[_S219 + u32(3)];
    var _S228 : f32 = params_0.band_0;
    var _S229 : f32 = params_0.dt_0;
    var _S230 : bool = (params_0.impact_peak_0) != u32(0);
    var _S231 : u32 = l_pool_0(from_2) + u32(3) * off_0;
    var _S232 : u32 = l_pool_0(to_2) + u32(3) * base_0;
    loop
    {
        var _S233 : vec3<f32> = vec3<f32>(0.0f);
        var _S234 : u32 = sort_0[k_18];
        var _S235 : u32 = sort_0[_S225 + u32(1) + k_18];
        var first_0 : bool = true;
        var last_0 : u32 = u32(0);
        var step_1 : u32 = u32(0);
        f_total_0 = _S233;
        t_total_0 = _S233;
        loop
        {
            if(step_1 < _S235)
            {
            }
            else
            {
                break;
            }
            p_11 = u32(4294967295);
            kept_0 = u32(0);
            q3_0 = u32(0);
            loop
            {
                if(q3_0 < _S235)
                {
                }
                else
                {
                    break;
                }
                var key_1 : u32 = manifolds_0[_S234 + q3_0].ids_0.y + u32(1);
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
                    if(p_11 == u32(4294967295))
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
                    p_11 = q3_0;
                    kept_0 = key_1;
                }
                q3_0 = q3_0 + u32(1);
            }
            if(p_11 == u32(4294967295))
            {
                break;
            }
            var r_16 : Record_std430_0 = manifolds_0[_S234 + p_11];
            var _S236 : vec3<f32> = r_16.impulse_0.xyz;
            var j_1 : f32 = length(_S236);
            if(j_1 == 0.0f)
            {
                var _S237 : u32 = step_1 + u32(1);
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S237;
                continue;
            }
            var m_other_0 : f32 = r_16.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_0))
                {
                    m_1 = _S226;
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
            var _S238 : f32 = min(j_1, 2.0f * m_1 * max(r_16.impulse_0.w - _S228, 0.0f));
            if(_S238 < j_1)
            {
                var f_1 : vec3<f32> = _S236 / vec3<f32>(j_1) * vec3<f32>(((j_1 - _S238) / _S229));
                var _S239 : vec3<f32> = t_total_0 + cross(r_16.point_0.xyz - centre_0, f_1);
                f_total_1 = f_total_0 + f_1;
                t_total_1 = _S239;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            var _S237 : u32 = step_1 + u32(1);
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S237;
        }
        p_11 = u32(0);
        kept_0 = u32(0);
        loop
        {
            if(p_11 < carried_0)
            {
            }
            else
            {
                break;
            }
            var _S240 : u32 = _S231 + u32(3) * p_11;
            var a_6 : vec4<f32> = loads_0[_S240];
            var b_39 : vec4<f32> = loads_0[_S240 + u32(1)];
            var c_6 : vec4<f32> = loads_0[_S240 + u32(2)];
            var _S241 : f32 = loads_0[_S240 + u32(1)].w;
            var mean_0 : f32 = pulse_mean_0(loads_0[_S240].x, loads_0[_S240].y, loads_0[_S240].z, loads_0[_S240].w, _S241, _S229);
            if(mean_0 != 0.0f)
            {
                var f_2 : vec3<f32> = c_6.xyz * vec3<f32>(mean_0);
                var _S242 : vec3<f32> = t_total_0 + cross(b_39.xyz - centre_0, f_2);
                f_total_0 = f_total_0 + f_2;
                t_total_0 = _S242;
            }
            if((a_6.x + a_6.z + _S241) > _S229)
            {
                a_6[i32(0)] = a_6[i32(0)] - _S229;
                var _S243 : u32 = _S232 + u32(3) * kept_0;
                loads_0[_S243] = a_6;
                loads_0[_S243 + u32(1)] = b_39;
                loads_0[_S243 + u32(2)] = c_6;
                kept_0 = kept_0 + u32(1);
            }
            p_11 = p_11 + u32(1);
        }
        q3_0 = u32(0);
        overflow_0 = false;
        loop
        {
            if(q3_0 < _S235)
            {
            }
            else
            {
                break;
            }
            var r_17 : Record_std430_0 = manifolds_0[_S234 + q3_0];
            var _S244 : vec3<f32> = r_17.impulse_0.xyz;
            var j_2 : f32 = length(_S244);
            if(j_2 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var m_other_1 : f32 = r_17.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_1))
                {
                    m_1 = _S226;
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
            var _S245 : f32 = r_17.impulse_0.w;
            var _S246 : f32 = min(j_2, 2.0f * m_1 * max(_S245 - _S228, 0.0f));
            if(_S246 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var _S247 : f32 = _S227.y;
            var duration_1 : f32 = hertz_duration_0(m_1, min(r_17.other_0.z, _S227.z), 1.0f / ((1.0f - _S247 * _S247) / _S227.x + 1.0f / r_17.other_0.y), _S245);
            var peak_1 : f32 = 3.14159274101257324f * _S246 / (2.0f * duration_1);
            var dir_0 : vec3<f32> = _S244 / vec3<f32>(j_2);
            var mean_1 : f32;
            if(_S230)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S229);
            }
            if(mean_1 != 0.0f)
            {
                var f_3 : vec3<f32> = dir_0 * vec3<f32>(mean_1);
                var _S248 : vec3<f32> = t_total_0 + cross(r_17.point_0.xyz - centre_0, f_3);
                f_total_1 = f_total_0 + f_3;
                t_total_1 = _S248;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            var kept_1 : u32;
            if(duration_1 > _S229)
            {
                if(kept_0 < room_1)
                {
                    var _S249 : u32 = _S232 + u32(3) * kept_0;
                    loads_0[_S249] = vec4<f32>(- _S229, 0.0f, 0.0f, peak_1);
                    loads_0[_S249 + u32(1)] = vec4<f32>(r_17.point_0.xyz, duration_1);
                    loads_0[_S249 + u32(2)] = vec4<f32>(dir_0, 0.0f);
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
        var _S250 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(8));
    }
    var _S251 : u32 = _S220 + u32(2) * to_2;
    work_0[_S251] = base_0;
    work_0[_S251 + u32(1)] = kept_0;
    loads_0[k_18] = vec4<f32>(length(f_total_0), 0.0f, 0.0f, 0.0f);
    if((params_0.load_model_0) == u32(1))
    {
        var _S252 : u32 = l_vec_0() + u32(2) * k_18;
        loads_0[_S252] = vec4<f32>(f_total_0, 0.0f);
        loads_0[_S252 + u32(1)] = vec4<f32>(t_total_0, 0.0f);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_breaks(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var i_2 : u32 = id_5.x;
    var _S253 : bool;
    if(i_2 >= (params_0.bond_count_0))
    {
        _S253 = true;
    }
    else
    {
        _S253 = work_0[w_alive_0() + i_2] == u32(0);
    }
    if(_S253)
    {
        return;
    }
    var bond_2 : vec4<u32> = bonds_0[i_2];
    var _S254 : u32 = bond_2.x;
    if((max(loads_0[_S254].x, loads_0[bond_2.y].x)) > (bitcast<f32>((bond_2.z))))
    {
        work_0[w_alive_0() + i_2] = u32(0);
        var _S255 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S254)]), u32(1));
        var _S256 : u32 = a_counters_0();
        var _S257 : u32 = atomicAdd(&(atom_0[_S256]), u32(1));
        var _S258 : u32 = atomicAdd(&(atom_0[_S256 + u32(1)]), u32(1));
    }
    return;
}

fn groups_of_0( n_3 : u32,  size_0 : u32) -> u32
{
    var _S259 : u32 = (n_3 + size_0 - u32(1)) / size_0;
    return _S259;
}

fn set_args_0( at_8 : u32,  x_4 : u32)
{
    var _S260 : u32 = u32(3) * at_8;
    args_0[_S260] = x_4;
    args_0[_S260 + u32(1)] = u32(1);
    args_0[_S260 + u32(2)] = u32(1);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_gate(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var _S261 : u32 = a_counters_0();
    var _S262 : u32 = atomicLoad(&(atom_0[_S261]));
    var run_0 : bool = _S262 != u32(0);
    atomicStore(&(atom_0[_S261]), u32(0));
    atomicStore(&(atom_0[_S261 + u32(5) + params_0.parity_0]), u32(0));
    if(run_0)
    {
        var _S263 : u32 = atomicAdd(&(atom_0[_S261 + u32(4)]), u32(1));
    }
    var _S264 : u32;
    if(run_0)
    {
        var _S265 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S264 = _S265;
    }
    else
    {
        _S264 = u32(0);
    }
    set_args_0(u32(0), _S264);
    if(run_0)
    {
        var _S266 : u32 = groups_of_0(params_0.bond_count_0, u32(64));
        _S264 = _S266;
    }
    else
    {
        _S264 = u32(0);
    }
    set_args_0(u32(1), _S264);
    if(run_0)
    {
        var _S267 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S264 = _S267;
    }
    else
    {
        _S264 = u32(0);
    }
    set_args_0(u32(2), _S264);
    if(run_0)
    {
        var _S268 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S264 = _S268;
    }
    else
    {
        _S264 = u32(0);
    }
    set_args_0(u32(3), _S264);
    if(run_0)
    {
        var _S269 : u32 = groups_of_0(params_0.body_count_0, u32(1024));
        _S264 = _S269;
    }
    else
    {
        _S264 = u32(0);
    }
    set_args_0(u32(4), _S264);
    var _S270 : i32;
    if(run_0)
    {
        _S270 = i32(1);
    }
    else
    {
        _S270 = i32(0);
    }
    set_args_0(u32(5), u32(_S270));
    return;
}

fn dirty_0( body_1 : u32) -> bool
{
    var _S271 : u32 = atomicLoad(&(atom_0[a_dirty_0() + body_1]));
    return _S271 != u32(0);
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
fn fr_uf_init(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var c_7 : u32 = id_7.x;
    var _S272 : bool;
    if(c_7 >= (params_0.chunk_count_0))
    {
        _S272 = true;
    }
    else
    {
        var _S273 : bool = dirty_0(body_of_0(c_7));
        _S272 = !_S273;
    }
    if(_S272)
    {
        return;
    }
    atomicStore(&(atom_0[a_parent_0() + c_7]), c_7);
    atomicStore(&(atom_0[a_count_0() + c_7]), u32(0));
    atomicStore(&(atom_0[a_anchor_0() + c_7]), u32(0));
    var w_3 : u32 = u32(0);
    loop
    {
        if(w_3 < u32(20))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[a_sums_0() + u32(20) * c_7 + w_3]), u32(0));
        w_3 = w_3 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_body_init(@builtin(global_invocation_id) id_8 : vec3<u32>)
{
    var s_7 : u32 = id_8.x;
    var _S274 : bool;
    if(s_7 >= (params_0.body_count_0))
    {
        _S274 = true;
    }
    else
    {
        var _S275 : bool = dirty_0(s_7);
        _S274 = !_S275;
    }
    if(_S274)
    {
        return;
    }
    atomicStore(&(atom_0[a_best_count_0() + s_7]), u32(0));
    atomicStore(&(atom_0[a_best_root_0() + s_7]), u32(4294967295));
    return;
}

fn find_0( x_5 : u32) -> u32
{
    var _S276 : u32 = a_parent_0();
    var _S277 : u32 = atomicLoad(&(atom_0[_S276 + x_5]));
    var p_12 : u32 = _S277;
    var _S278 : u32 = x_5;
    var guard_0 : u32 = u32(0);
    loop
    {
        var _S279 : bool;
        if(guard_0 < u32(1048576))
        {
            _S279 = p_12 != _S278;
        }
        else
        {
            _S279 = false;
        }
        if(_S279)
        {
        }
        else
        {
            break;
        }
        var gp_0 : u32 = atomicLoad(&(atom_0[_S276 + p_12]));
        if(gp_0 != p_12)
        {
            var _S280 : u32 = atomicCompareExchangeWeak(&(atom_0[_S276 + _S278]), p_12, gp_0).old_value;
        }
        var _S281 : u32 = guard_0 + u32(1);
        _S278 = p_12;
        p_12 = gp_0;
        guard_0 = _S281;
    }
    return _S278;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_union(@builtin(global_invocation_id) id_9 : vec3<u32>)
{
    var i_3 : u32 = id_9.x;
    var _S282 : bool;
    if(i_3 >= (params_0.bond_count_0))
    {
        _S282 = true;
    }
    else
    {
        _S282 = work_0[w_alive_0() + i_3] == u32(0);
    }
    if(_S282)
    {
        return;
    }
    var bond_3 : vec4<u32> = bonds_0[i_3];
    var _S283 : u32 = bond_3.x;
    var _S284 : bool = dirty_0(body_of_0(_S283));
    if(!_S284)
    {
        return;
    }
    var _S285 : u32 = bond_3.y;
    var a_7 : u32 = _S283;
    var b_40 : u32 = _S285;
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
        var ra_0 : u32 = find_0(a_7);
        var rb_0 : u32 = find_0(b_40);
        if(ra_0 == rb_0)
        {
            return;
        }
        var _S286 : u32 = max(ra_0, rb_0);
        var _S287 : u32 = min(ra_0, rb_0);
        var was_0 : u32 = atomicCompareExchangeWeak(&(atom_0[a_parent_0() + _S286]), _S286, _S287).old_value;
        if(was_0 == _S286)
        {
            return;
        }
        var _S288 : u32 = guard_1 + u32(1);
        a_7 = _S286;
        b_40 = _S287;
        guard_1 = _S288;
    }
    var _S289 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(4));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_flatten(@builtin(global_invocation_id) id_10 : vec3<u32>)
{
    var c_8 : u32 = id_10.x;
    var _S290 : bool;
    if(c_8 >= (params_0.chunk_count_0))
    {
        _S290 = true;
    }
    else
    {
        var _S291 : bool = dirty_0(body_of_0(c_8));
        _S290 = !_S291;
    }
    if(_S290)
    {
        return;
    }
    var r_18 : u32 = find_0(c_8);
    work_0[w_label_0() + c_8] = r_18;
    var _S292 : u32 = atomicAdd(&(atom_0[a_count_0() + r_18]), u32(1));
    if(((info_0[c_8] & (u32(1)))) != u32(0))
    {
        var _S293 : u32 = atomicOr(&(atom_0[a_anchor_0() + r_18]), u32(1));
    }
    var _S294 : u32 = u32(4) * c_8;
    var c0_1 : vec4<f32> = chunks_0[_S294];
    var c1_0 : vec4<f32> = chunks_0[_S294 + u32(1)];
    var c2_0 : vec4<f32> = chunks_0[_S294 + u32(2)];
    var m_2 : f32 = c0_1.w;
    var d_19 : vec3<f32> = c0_1.xyz - chunks_0[u32(4) * r_18].xyz;
    var dd_0 : f32 = dot(d_19, d_19);
    var at_9 : u32 = a_sums_0() + u32(20) * r_18;
    add64_0(at_9, m_2, params_0.scale_m_0);
    var _S295 : f32 = d_19.x;
    var _S296 : f32 = m_2 * _S295;
    add64_0(at_9 + u32(2), _S296, params_0.scale_s_0);
    var _S297 : f32 = d_19.y;
    var _S298 : f32 = m_2 * _S297;
    add64_0(at_9 + u32(4), _S298, params_0.scale_s_0);
    var _S299 : f32 = d_19.z;
    add64_0(at_9 + u32(6), m_2 * _S299, params_0.scale_s_0);
    add64_0(at_9 + u32(8), c1_0.x + m_2 * (dd_0 - _S295 * _S295), params_0.scale_j_0);
    add64_0(at_9 + u32(10), c1_0.y + m_2 * (dd_0 - _S297 * _S297), params_0.scale_j_0);
    add64_0(at_9 + u32(12), c1_0.z + m_2 * (dd_0 - _S299 * _S299), params_0.scale_j_0);
    add64_0(at_9 + u32(14), c1_0.w - _S296 * _S297, params_0.scale_j_0);
    add64_0(at_9 + u32(16), c2_0.x - _S296 * _S299, params_0.scale_j_0);
    add64_0(at_9 + u32(18), c2_0.y - _S298 * _S299, params_0.scale_j_0);
    return;
}

fn is_root_0( c_9 : u32) -> bool
{
    var _S300 : bool;
    if(c_9 < (params_0.chunk_count_0))
    {
        var _S301 : bool = dirty_0(body_of_0(c_9));
        _S300 = _S301;
    }
    else
    {
        _S300 = false;
    }
    if(_S300)
    {
        _S300 = work_0[w_label_0() + c_9] == c_9;
    }
    else
    {
        _S300 = false;
    }
    return _S300;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_count(@builtin(global_invocation_id) id_11 : vec3<u32>)
{
    var c_10 : u32 = id_11.x;
    var _S302 : bool = is_root_0(c_10);
    if(!_S302)
    {
        return;
    }
    var _S303 : u32 = atomicLoad(&(atom_0[a_count_0() + c_10]));
    var _S304 : u32 = atomicMax(&(atom_0[a_best_count_0() + body_of_0(c_10)]), _S303);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_root(@builtin(global_invocation_id) id_12 : vec3<u32>)
{
    var c_11 : u32 = id_12.x;
    var _S305 : bool = is_root_0(c_11);
    if(!_S305)
    {
        return;
    }
    var s_8 : u32 = body_of_0(c_11);
    var _S306 : u32 = atomicLoad(&(atom_0[a_count_0() + c_11]));
    var _S307 : u32 = atomicLoad(&(atom_0[a_best_count_0() + s_8]));
    if(_S306 == _S307)
    {
        var _S308 : u32 = atomicMin(&(atom_0[a_best_root_0() + s_8]), c_11);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_mark(@builtin(global_invocation_id) id_13 : vec3<u32>)
{
    var c_12 : u32 = id_13.x;
    if(c_12 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S309 : bool = is_root_0(c_12);
    var _S310 : bool;
    if(_S309)
    {
        var _S311 : u32 = atomicLoad(&(atom_0[a_best_root_0() + body_of_0(c_12)]));
        _S310 = _S311 != c_12;
    }
    else
    {
        _S310 = false;
    }
    var _S312 : u32;
    if(_S310)
    {
        _S312 = u32(1);
    }
    else
    {
        _S312 = u32(0);
    }
    work_0[w_new_0() + c_12] = _S312;
    return;
}

var<workgroup> g_scan_0 : array<u32, i32(1024)>;

@compute
@workgroup_size(256, 1, 1)
fn fr_scan_blocks(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var step_2 : u32;
    var add_0 : u32;
    var t_4 : u32 = thread_0.x;
    var _S313 : u32 = group_0.x;
    var _S314 : u32 = u32(1024) * _S313 + u32(4) * t_4;
    var v_10 : array<u32, i32(4)>;
    var j_3 : u32 = u32(0);
    var sum_2 : u32 = u32(0);
    loop
    {
        if(j_3 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S314 + j_3) < (scan_0.n_0))
        {
            step_2 = work_0[scan_0.in_base_0 + _S314 + j_3];
        }
        else
        {
            step_2 = u32(0);
        }
        v_10[j_3] = step_2;
        var sum_3 : u32 = sum_2 + step_2;
        j_3 = j_3 + u32(1);
        sum_2 = sum_3;
    }
    g_scan_0[t_4] = sum_2;
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
        if(t_4 >= step_2)
        {
            add_0 = g_scan_0[t_4 - step_2];
        }
        else
        {
            add_0 = u32(0);
        }
        workgroupBarrier();
        g_scan_0[t_4] = g_scan_0[t_4] + add_0;
        workgroupBarrier();
        step_2 = (step_2 << (u32(1)));
    }
    var _S315 : u32 = g_scan_0[t_4] - sum_2;
    j_3 = u32(0);
    add_0 = _S315;
    loop
    {
        if(j_3 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S314 + j_3) < (scan_0.n_0))
        {
            work_0[scan_0.out_base_0 + _S314 + j_3] = add_0;
        }
        var run_1 : u32 = add_0 + v_10[j_3];
        j_3 = j_3 + u32(1);
        add_0 = run_1;
    }
    if(t_4 == u32(255))
    {
        work_0[scan_0.block_base_0 + _S313] = g_scan_0[i32(255)];
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn fr_scan_top(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var t_5 : u32 = thread_1.x;
    var blocks_0 : u32 = (scan_0.n_0 + u32(1023)) / u32(1024);
    var _S316 : bool = t_5 < blocks_0;
    var v_11 : u32;
    if(_S316)
    {
        v_11 = work_0[scan_0.block_base_0 + t_5];
    }
    else
    {
        v_11 = u32(0);
    }
    g_scan_0[t_5] = v_11;
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
        if(t_5 >= step_3)
        {
            add_1 = g_scan_0[t_5 - step_3];
        }
        else
        {
            add_1 = u32(0);
        }
        workgroupBarrier();
        g_scan_0[t_5] = g_scan_0[t_5] + add_1;
        workgroupBarrier();
        step_3 = (step_3 << (u32(1)));
    }
    if(_S316)
    {
        work_0[scan_0.block_base_0 + t_5] = g_scan_0[t_5] - v_11;
    }
    if(t_5 == u32(1023))
    {
        work_0[scan_0.block_base_0 + blocks_0] = g_scan_0[i32(1023)];
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_scan_add(@builtin(global_invocation_id) id_14 : vec3<u32>)
{
    var i_4 : u32 = id_14.x;
    if(i_4 >= (scan_0.n_0))
    {
        return;
    }
    work_0[scan_0.out_base_0 + i_4] = work_0[scan_0.out_base_0 + i_4] + work_0[scan_0.block_base_0 + i_4 / u32(1024)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_freelist(@builtin(global_invocation_id) id_15 : vec3<u32>)
{
    var s_9 : u32 = id_15.x;
    var _S317 : bool;
    if(s_9 >= (params_0.body_count_0))
    {
        _S317 = true;
    }
    else
    {
        _S317 = work_0[w_free_0() + s_9] == u32(0);
    }
    if(_S317)
    {
        return;
    }
    work_0[w_free_list_0() + work_0[w_free_rank_0() + s_9]] = s_9;
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
fn fr_assign(@builtin(global_invocation_id) id_16 : vec3<u32>)
{
    var c_13 : u32 = id_16.x;
    var _S318 : bool = is_root_0(c_13);
    if(!_S318)
    {
        return;
    }
    var s_10 : u32 = body_of_0(c_13);
    var _S319 : u32 = atomicLoad(&(atom_0[a_best_root_0() + s_10]));
    var slot_8 : u32;
    if(_S319 != c_13)
    {
        var k_19 : u32 = work_0[w_rank_0() + c_13];
        if(work_0[w_rank_0() + c_13] < (free_total_0()))
        {
            var slot_9 : u32 = work_0[w_free_list_0() + k_19];
            work_0[w_free_0() + work_0[w_free_list_0() + k_19]] = u32(0);
            var _S320 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(2)]), u32(1));
            slot_8 = slot_9;
        }
        else
        {
            var _S321 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(1));
            slot_8 = s_10;
        }
    }
    else
    {
        slot_8 = s_10;
    }
    work_0[w_slot_0() + c_13] = slot_8;
    return;
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
    var q_15 : Quat_0;
    if(m22_0 <= 0.0f)
    {
        var dif10_0 : f32 = m11_0 - m00_0;
        var omm22_0 : f32 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            var f_4 : f32 = omm22_0 - dif10_0;
            var inv_0 : f32 = 0.5f / sqrt(f_4);
            q_15.x_0 = f_4 * inv_0;
            q_15.y_0 = (m01_0 + m10_0) * inv_0;
            q_15.z_0 = (m02_0 + m20_0) * inv_0;
            q_15.w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            var f_5 : f32 = omm22_0 + dif10_0;
            var inv_1 : f32 = 0.5f / sqrt(f_5);
            q_15.x_0 = (m01_0 + m10_0) * inv_1;
            q_15.y_0 = f_5 * inv_1;
            q_15.z_0 = (m12_0 + m21_0) * inv_1;
            q_15.w_0 = (m20_0 - m02_0) * inv_1;
        }
    }
    else
    {
        var sum10_0 : f32 = m11_0 + m00_0;
        var opm22_0 : f32 = 1.0f + m22_0;
        if(sum10_0 <= 0.0f)
        {
            var f_6 : f32 = opm22_0 - sum10_0;
            var inv_2 : f32 = 0.5f / sqrt(f_6);
            q_15.x_0 = (m02_0 + m20_0) * inv_2;
            q_15.y_0 = (m12_0 + m21_0) * inv_2;
            q_15.z_0 = f_6 * inv_2;
            q_15.w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            var f_7 : f32 = opm22_0 + sum10_0;
            var inv_3 : f32 = 0.5f / sqrt(f_7);
            q_15.x_0 = (m12_0 - m21_0) * inv_3;
            q_15.y_0 = (m20_0 - m02_0) * inv_3;
            q_15.z_0 = (m01_0 - m10_0) * inv_3;
            q_15.w_0 = f_7 * inv_3;
        }
    }
    return q_15;
}

fn principal_axes_0( r0_0 : vec3<f32>,  r1_2 : vec3<f32>,  r2_2 : vec3<f32>,  rot_0 : ptr<function, Quat_0>,  moments_0 : ptr<function, vec3<f32>>)
{
    var _S322 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S323 : array<f32, i32(3)> = array<f32, i32(3)>( r1_2.x, r1_2.y, r1_2.z );
    var _S324 : array<f32, i32(3)> = array<f32, i32(3)>( r2_2.x, r2_2.y, r2_2.z );
    var a_8 : array<array<f32, i32(3)>, i32(3)>;
    a_8[i32(0)] = _S322;
    a_8[i32(1)] = _S323;
    a_8[i32(2)] = _S324;
    var _S325 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S326 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S327 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_12 : array<array<f32, i32(3)>, i32(3)>;
    v_12[i32(0)] = _S325;
    v_12[i32(1)] = _S326;
    v_12[i32(2)] = _S327;
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
        var q_16 : u32;
        if((abs(a_8[i32(0)][i32(2)])) > (abs(a_8[u32(0)][u32(1)])))
        {
            q_16 = u32(2);
        }
        else
        {
            q_16 = u32(1);
        }
        var p_13 : u32;
        var q_17 : u32;
        if((abs(a_8[i32(1)][i32(2)])) > (abs(a_8[u32(0)][q_16])))
        {
            p_13 = u32(1);
            q_17 = u32(2);
        }
        else
        {
            p_13 = u32(0);
            q_17 = q_16;
        }
        if((abs(a_8[p_13][q_17])) <= (9.99999982451670044e-15f * max(max(abs(a_8[i32(0)][i32(0)]), abs(a_8[i32(1)][i32(1)])), max(abs(a_8[i32(2)][i32(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        var theta_0 : f32 = 0.5f * atan2(2.0f * a_8[p_13][q_17], a_8[q_17][q_17] - a_8[p_13][p_13]);
        var s_11 : f32 = sin(theta_0);
        var c_14 : f32 = cos(theta_0);
        var app_0 : f32 = a_8[p_13][p_13];
        var aqq_0 : f32 = a_8[q_17][q_17];
        var apq_0 : f32 = a_8[p_13][q_17];
        var k_20 : u32 = u32(0);
        loop
        {
            if(k_20 < u32(3))
            {
            }
            else
            {
                break;
            }
            var _S328 : bool;
            if(k_20 != p_13)
            {
                _S328 = k_20 != q_17;
            }
            else
            {
                _S328 = false;
            }
            if(_S328)
            {
                var kp_0 : f32 = a_8[k_20][p_13];
                var kq_0 : f32 = a_8[k_20][q_17];
                var _S329 : f32 = c_14 * a_8[k_20][p_13] - s_11 * a_8[k_20][q_17];
                a_8[k_20][p_13] = _S329;
                a_8[p_13][k_20] = _S329;
                var _S330 : f32 = s_11 * kp_0 + c_14 * kq_0;
                a_8[k_20][q_17] = _S330;
                a_8[q_17][k_20] = _S330;
            }
            k_20 = k_20 + u32(1);
        }
        var _S331 : f32 = c_14 * c_14;
        var _S332 : f32 = 2.0f * s_11 * c_14 * apq_0;
        var _S333 : f32 = s_11 * s_11;
        a_8[p_13][p_13] = _S331 * app_0 - _S332 + _S333 * aqq_0;
        a_8[q_17][q_17] = _S333 * app_0 + _S332 + _S331 * aqq_0;
        a_8[p_13][q_17] = 0.0f;
        a_8[q_17][p_13] = 0.0f;
        var row_1 : u32 = u32(0);
        loop
        {
            if(row_1 < u32(3))
            {
            }
            else
            {
                break;
            }
            var kp_1 : f32 = v_12[row_1][p_13];
            var kq_1 : f32 = v_12[row_1][q_17];
            v_12[row_1][p_13] = c_14 * v_12[row_1][p_13] - s_11 * v_12[row_1][q_17];
            v_12[row_1][q_17] = s_11 * kp_1 + c_14 * kq_1;
            row_1 = row_1 + u32(1);
        }
        it_0 = it_0 + u32(1);
    }
    var x_6 : vec3<f32> = vec3<f32>(v_12[i32(0)][i32(0)], v_12[i32(1)][i32(0)], v_12[i32(2)][i32(0)]);
    var y_1 : vec3<f32> = vec3<f32>(v_12[i32(0)][i32(1)], v_12[i32(1)][i32(1)], v_12[i32(2)][i32(1)]);
    var z_1 : vec3<f32> = vec3<f32>(v_12[i32(0)][i32(2)], v_12[i32(1)][i32(2)], v_12[i32(2)][i32(2)]);
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
    (*moments_0) = vec3<f32>(a_8[i32(0)][i32(0)], a_8[i32(1)][i32(1)], a_8[i32(2)][i32(2)]);
    return;
}

fn rot_to_rubble_0( q_18 : Quat_0) -> Quat_0
{
    var axes_1 : Quat_0;
    axes_1.w_0 = 0.70710676908493042f;
    axes_1.x_0 = -0.70710676908493042f;
    axes_1.y_0 = 0.0f;
    axes_1.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_1, q_18));
}

fn stage_0( c_15 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * c_15;
}

fn quat_vec_0( q_19 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_19.x_0, q_19.y_0, q_19.z_0, q_19.w_0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_piece(@builtin(global_invocation_id) id_17 : vec3<u32>)
{
    var c_16 : u32 = id_17.x;
    var _S334 : bool = is_root_0(c_16);
    if(!_S334)
    {
        return;
    }
    var parent_0 : u32 = body_of_0(c_16);
    var pb_2 : u32 = u32(7) * parent_0;
    var at_10 : u32 = a_sums_0() + u32(20) * c_16;
    var mass_1 : f32 = read64_0(at_10, params_0.scale_m_0);
    var _S335 : f32 = read64_0(at_10 + u32(2), params_0.scale_s_0);
    var _S336 : f32 = read64_0(at_10 + u32(4), params_0.scale_s_0);
    var _S337 : f32 = read64_0(at_10 + u32(6), params_0.scale_s_0);
    var d_20 : vec3<f32> = vec3<f32>(_S335, _S336, _S337) / vec3<f32>(max(mass_1, 1.00000000317107685e-30f));
    var jxx_0 : f32 = read64_0(at_10 + u32(8), params_0.scale_j_0);
    var jyy_0 : f32 = read64_0(at_10 + u32(10), params_0.scale_j_0);
    var jzz_0 : f32 = read64_0(at_10 + u32(12), params_0.scale_j_0);
    var jxy_0 : f32 = read64_0(at_10 + u32(14), params_0.scale_j_0);
    var jxz_0 : f32 = read64_0(at_10 + u32(16), params_0.scale_j_0);
    var jyz_0 : f32 = read64_0(at_10 + u32(18), params_0.scale_j_0);
    var dd_1 : f32 = dot(d_20, d_20);
    var _S338 : f32 = d_20.x;
    var _S339 : f32 = mass_1 * _S338;
    var _S340 : f32 = d_20.y;
    var _S341 : f32 = jxy_0 + _S339 * _S340;
    var _S342 : f32 = d_20.z;
    var _S343 : f32 = jxz_0 + _S339 * _S342;
    var _S344 : f32 = jyz_0 + mass_1 * _S340 * _S342;
    var com_5 : vec3<f32> = chunks_0[u32(4) * c_16].xyz + d_20;
    var principal_0 : Quat_0;
    var moments_1 : vec3<f32>;
    principal_axes_0(vec3<f32>(jxx_0 - mass_1 * (dd_1 - _S338 * _S338), _S341, _S343), vec3<f32>(_S341, jyy_0 - mass_1 * (dd_1 - _S340 * _S340), _S344), vec3<f32>(_S343, _S344, jzz_0 - mass_1 * (dd_1 - _S342 * _S342)), &(principal_0), &(moments_1));
    var _S345 : u32 = atomicLoad(&(atom_0[a_anchor_0() + c_16]));
    var anchored_0 : bool = _S345 != u32(0);
    var q_20 : Quat_0;
    var origin_8 : vec3<f32>;
    structure_pose_4(parent_0, pb_2, &(q_20), &(origin_8));
    var p_parent_0 : vec3<f32> = bodies_0[pb_2].xyz;
    var v_parent_0 : vec3<f32> = bodies_0[pb_2 + u32(3)].xyz;
    var w_parent_0 : vec3<f32> = bodies_0[pb_2 + u32(4)].xyz;
    var pos_0 : vec3<f32> = to_rubble_0(origin_8 + rotate_0(q_20, com_5));
    var orientation_0 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(q_20, principal_0)));
    var _S346 : bool;
    if(anchored_0)
    {
        _S346 = true;
    }
    else
    {
        _S346 = !(mass_1 > 0.0f);
    }
    var inv_mass_3 : f32;
    if(_S346)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / mass_1;
    }
    var _S347 : vec3<f32> = vec3<f32>(0.0f);
    var inv_inertia_0 : vec3<f32>;
    if(!anchored_0)
    {
        var _S348 : f32;
        if((moments_1.x) > 0.0f)
        {
            _S348 = 1.0f / moments_1.x;
        }
        else
        {
            _S348 = 0.0f;
        }
        var _S349 : f32;
        if((moments_1.y) > 0.0f)
        {
            _S349 = 1.0f / moments_1.y;
        }
        else
        {
            _S349 = 0.0f;
        }
        var _S350 : f32;
        if((moments_1.z) > 0.0f)
        {
            _S350 = 1.0f / moments_1.z;
        }
        else
        {
            _S350 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S348, _S349, _S350);
    }
    else
    {
        inv_inertia_0 = _S347;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_0)
    {
        velocity_0 = _S347;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    var spin_0 : vec3<f32>;
    if(anchored_0)
    {
        spin_0 = _S347;
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
    var st_2 : u32 = stage_0(c_16);
    frames_0[st_2] = vec4<f32>(pos_0, inv_mass_3);
    frames_0[st_2 + u32(1)] = quat_vec_0(orientation_0);
    frames_0[st_2 + u32(2)] = vec4<f32>(inv_inertia_0, (bitcast<f32>((flags_0))));
    frames_0[st_2 + u32(3)] = vec4<f32>(velocity_0, 0.0f);
    frames_0[st_2 + u32(4)] = vec4<f32>(spin_0, (bitcast<f32>((params_0.step_0))));
    frames_0[st_2 + u32(5)] = vec4<f32>(0.0f);
    frames_0[st_2 + u32(6)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    frames_0[st_2 + u32(7)] = vec4<f32>(com_5, frames_0[u32(2) * parent_0].w);
    frames_0[st_2 + u32(8)] = quat_vec_0(principal_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_commit(@builtin(global_invocation_id) id_18 : vec3<u32>)
{
    var c_17 : u32 = id_18.x;
    var _S351 : bool = is_root_0(c_17);
    if(!_S351)
    {
        return;
    }
    var slot_10 : u32 = work_0[w_slot_0() + c_17];
    var st_3 : u32 = stage_0(c_17);
    var region_5 : u32 = u32(0);
    loop
    {
        if(region_5 < u32(2))
        {
        }
        else
        {
            break;
        }
        var w_4 : u32 = u32(0);
        loop
        {
            if(w_4 < u32(7))
            {
            }
            else
            {
                break;
            }
            bodies_0[u32(7) * (slot_10 + region_5 * params_0.max_bodies_0) + w_4] = frames_0[st_3 + w_4];
            w_4 = w_4 + u32(1);
        }
        region_5 = region_5 + u32(1);
    }
    var _S352 : u32 = u32(2) * slot_10;
    frames_0[_S352] = frames_0[st_3 + u32(7)];
    frames_0[_S352 + u32(1)] = frames_0[st_3 + u32(8)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_reparent(@builtin(global_invocation_id) id_19 : vec3<u32>)
{
    var c_18 : u32 = id_19.x;
    var _S353 : bool;
    if(c_18 >= (params_0.chunk_count_0))
    {
        _S353 = true;
    }
    else
    {
        var _S354 : bool = dirty_0(body_of_0(c_18));
        _S353 = !_S354;
    }
    if(_S353)
    {
        return;
    }
    var r_19 : u32 = work_0[w_label_0() + c_18];
    var slot_11 : u32 = work_0[w_slot_0() + work_0[w_label_0() + c_18]];
    work_0[w_body_0() + c_18] = work_0[w_slot_0() + work_0[w_label_0() + c_18]];
    var _S355 : u32 = u32(2) * slot_11;
    var _S356 : vec3<f32> = frames_0[_S355].xyz;
    var _S357 : Quat_0 = conj_0(quat_of_0(frames_0[_S355 + u32(1)]));
    var inv_mass_4 : f32 = bodies_0[u32(7) * slot_11].w;
    var _S358 : f32;
    if(inv_mass_4 > 0.0f)
    {
        _S358 = 1.0f / inv_mass_4;
    }
    else
    {
        _S358 = float_bits_0(u32(2139095040));
    }
    var c2_2 : vec4<f32> = chunks_0[u32(4) * c_18 + u32(2)];
    var first_1 : u32 = (bitcast<u32>((c2_2.z)));
    var _S359 : u32 = (bitcast<u32>((c2_2.w)));
    var _S360 : vec4<f32> = chunks_0[u32(4) * r_19 + u32(3)];
    var p_14 : u32 = first_1;
    loop
    {
        if(p_14 < (first_1 + _S359))
        {
        }
        else
        {
            break;
        }
        var _S361 : u32 = u32(2) * p_14;
        var o_4 : vec4<f32> = parts_0[_S361];
        var col_0 : u32 = (bitcast<u32>((o_4.w)));
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S361 + u32(1)]);
        var _S362 : u32 = u32(4) * col_0;
        colliders_0[_S362] = vec4<f32>((bitcast<f32>((slot_11))), colliders_0[_S362].y, colliders_0[_S362].z, colliders_0[_S362].w);
        var _S363 : u32 = _S362 + u32(1);
        colliders_0[_S363] = vec4<f32>(rotate_0(_S357, o_4.xyz - _S356), colliders_0[_S363].w);
        colliders_0[_S362 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S357, rot_1)));
        var _S364 : u32 = u32(2) * col_0;
        collider_info_0[_S364] = vec4<u32>(collider_info_0[_S364].x, slot_11, collider_info_0[_S364].z, collider_info_0[_S364].w);
        collider_info_0[_S364 + u32(1)] = vec4<u32>((bitcast<u32>((_S358))), (bitcast<u32>((_S360.x))), (bitcast<u32>((_S360.w))), u32(0));
        p_14 = p_14 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_body_clear(@builtin(global_invocation_id) id_20 : vec3<u32>)
{
    var s_12 : u32 = id_20.x;
    if(s_12 >= (params_0.body_count_0))
    {
        return;
    }
    atomicStore(&(atom_0[a_dirty_0() + s_12]), u32(0));
    return;
}

fn slot_color_0( slot_12 : u32) -> vec3<f32>
{
    var h_4 : f32 = fract(f32(slot_12) * 0.6180340051651001f) * 6.0f;
    var f_8 : f32 = fract(h_4);
    var q_21 : f32 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_8);
    var t_6 : f32 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_8));
    var k_21 : u32 = u32(h_4);
    var _S365 : vec3<f32>;
    if(k_21 == u32(0))
    {
        _S365 = vec3<f32>(0.92000001668930054f, t_6, 0.34959998726844788f);
    }
    else
    {
        if(k_21 == u32(1))
        {
            _S365 = vec3<f32>(q_21, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_21 == u32(2))
            {
                _S365 = vec3<f32>(0.34959998726844788f, 0.92000001668930054f, t_6);
            }
            else
            {
                if(k_21 == u32(3))
                {
                    _S365 = vec3<f32>(0.34959998726844788f, q_21, 0.92000001668930054f);
                }
                else
                {
                    if(k_21 == u32(4))
                    {
                        _S365 = vec3<f32>(t_6, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S365 = vec3<f32>(0.92000001668930054f, 0.34959998726844788f, q_21);
                    }
                }
            }
        }
    }
    return _S365;
}

fn write_object_0( o_5 : u32,  q_22 : Quat_0,  t_7 : vec3<f32>,  color_0 : vec4<f32>)
{
    var x_7 : vec3<f32> = rotate_0(q_22, vec3<f32>(1.0f, 0.0f, 0.0f));
    var y_2 : vec3<f32> = rotate_0(q_22, vec3<f32>(0.0f, 1.0f, 0.0f));
    var z_3 : vec3<f32> = rotate_0(q_22, vec3<f32>(0.0f, 0.0f, 1.0f));
    var _S366 : u32 = u32(4) * o_5;
    objects_0[_S366] = vec4<f32>(x_7.x, y_2.x, z_3.x, t_7.x);
    objects_0[_S366 + u32(1)] = vec4<f32>(x_7.y, y_2.y, z_3.y, t_7.y);
    objects_0[_S366 + u32(2)] = vec4<f32>(x_7.z, y_2.z, z_3.z, t_7.z);
    objects_0[_S366 + u32(3)] = color_0;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_chunks(@builtin(global_invocation_id) id_21 : vec3<u32>)
{
    var k_22 : u32 = id_21.x;
    if(k_22 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_13 : u32 = body_of_0(k_22);
    var b_41 : u32 = u32(7) * slot_13;
    var q_23 : Quat_0;
    var origin_9 : vec3<f32>;
    structure_pose_4(slot_13, b_41, &(q_23), &(origin_9));
    var _S367 : u32 = u32(3) * k_22;
    var v0_0 : vec4<f32> = view_chunks_0[_S367];
    var base_2 : vec3<f32> = view_chunks_0[_S367 + u32(2)].xyz;
    var fixed_0 : bool = (bodies_0[b_41].w) == 0.0f;
    var color_1 : vec3<f32>;
    if((view_0.mode_0) == u32(1))
    {
        var _S368 : f32 = max(length(bodies_0[b_41 + u32(3)].xyz), length(bodies_0[b_41 + u32(4)].xyz));
        if(fixed_0)
        {
            color_1 = vec3<f32>(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S368 > (view_0.rest_speed_0))
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
            color_1 = slot_color_0(slot_13);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = vec3<f32>(0.44999998807907104f) * base_2 + vec3<f32>(0.55000001192092896f) * slot_color_0(slot_13);
            }
        }
    }
    write_object_0((bitcast<u32>((v0_0.w))), normalized_0(quat_mul_0(q_23, quat_of_0(view_chunks_0[_S367 + u32(1)]))), origin_9 + rotate_0(q_23, v0_0.xyz), vec4<f32>(color_1, 1.0f));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_shots(@builtin(global_invocation_id) id_22 : vec3<u32>)
{
    var i_5 : u32 = id_22.x;
    if(i_5 >= (view_0.shot_count_1))
    {
        return;
    }
    var b_42 : u32 = u32(7) * shot_bodies_0[i_5];
    var o_6 : u32 = view_0.impactor_base_0 + i_5;
    if((bodies_0[b_42].w) == 0.0f)
    {
        var w_5 : u32 = u32(0);
        loop
        {
            if(w_5 < u32(4))
            {
            }
            else
            {
                break;
            }
            objects_0[u32(4) * o_6 + w_5] = vec4<f32>(0.0f);
            w_5 = w_5 + u32(1);
        }
        return;
    }
    write_object_0(o_6, rot_from_rubble_0(quat_of_0(bodies_0[b_42 + u32(1)])), from_rubble_0(bodies_0[b_42].xyz), vec4<f32>(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f));
    return;
}

fn chunk_world_0( k_23 : u32) -> vec3<f32>
{
    var slot_14 : u32 = body_of_0(k_23);
    var q_24 : Quat_0;
    var origin_10 : vec3<f32>;
    structure_pose_2(slot_14, u32(7) * slot_14, &(q_24), &(origin_10));
    return origin_10 + rotate_0(q_24, view_chunks_0[u32(3) * k_23].xyz);
}

fn write_line_vertex_0( v_13 : u32,  p_15 : vec3<f32>)
{
    var _S369 : u32 = u32(7) * v_13;
    lines_0[_S369] = p_15.x;
    lines_0[_S369 + u32(1)] = p_15.y;
    lines_0[_S369 + u32(2)] = p_15.z;
    lines_0[_S369 + u32(3)] = 1.0f;
    lines_0[_S369 + u32(4)] = 0.86000001430511475f;
    lines_0[_S369 + u32(5)] = 0.25f;
    lines_0[_S369 + u32(6)] = 1.0f;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_bonds(@builtin(global_invocation_id) id_23 : vec3<u32>)
{
    var i_6 : u32 = id_23.x;
    if(i_6 >= (params_0.bond_count_0))
    {
        return;
    }
    var bond_4 : vec4<u32> = bonds_0[i_6];
    var a_9 : vec3<f32> = chunk_world_0(bond_4.x);
    var b_43 : vec3<f32>;
    if(work_0[w_alive_0() + i_6] != u32(0))
    {
        b_43 = chunk_world_0(bond_4.y);
    }
    else
    {
        b_43 = a_9;
    }
    var _S370 : u32 = u32(2) * i_6;
    write_line_vertex_0(_S370, a_9);
    write_line_vertex_0(_S370 + u32(1), b_43);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_reset(@builtin(global_invocation_id) id_24 : vec3<u32>)
{
    var b_44 : u32 = id_24.x;
    if(b_44 >= (params_0.body_count_0))
    {
        return;
    }
    var at_11 : u32 = a_strike_0() + u32(9) * b_44;
    var w_6 : u32 = u32(0);
    loop
    {
        if(w_6 < u32(3))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[at_11 + w_6]), u32(4294967295));
        atomicStore(&(atom_0[at_11 + u32(3) + w_6]), u32(0));
        w_6 = w_6 + u32(1);
    }
    atomicStore(&(atom_0[at_11 + u32(6)]), u32(2139095040));
    atomicStore(&(atom_0[at_11 + u32(7)]), u32(2139095040));
    atomicStore(&(atom_0[at_11 + u32(8)]), u32(4294967295));
    frames_0[strike_at_0(b_44)][i32(3)] = 0.0f;
    return;
}

fn chunk_box_world_1( k_24 : u32,  q_25 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_15 : u32 = body_of_0(k_24);
    var origin_11 : vec3<f32>;
    structure_pose_1(slot_15, u32(7) * slot_15, &((*q_25)), &(origin_11));
    return origin_11 + rotate_0((*q_25), geometry_0[u32(3) * k_24].xyz);
}

fn chunk_box_world_2( k_25 : u32,  q_26 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_16 : u32 = body_of_0(k_25);
    var origin_12 : vec3<f32>;
    structure_pose_4(slot_16, u32(7) * slot_16, &((*q_26)), &(origin_12));
    return origin_12 + rotate_0((*q_26), geometry_0[u32(3) * k_25].xyz);
}

fn ordered_0( f_9 : f32) -> u32
{
    var u_6 : u32 = (bitcast<u32>((f_9)));
    var _S371 : u32;
    if(((u_6 & (u32(2147483648)))) != u32(0))
    {
        _S371 = ~u_6;
    }
    else
    {
        _S371 = (u_6 | (u32(2147483648)));
    }
    return _S371;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_bounds(@builtin(global_invocation_id) id_25 : vec3<u32>)
{
    var k_26 : u32 = id_25.x;
    var _S372 : bool;
    if(k_26 >= (params_0.chunk_count_0))
    {
        _S372 = true;
    }
    else
    {
        _S372 = (params_0.shot_count_0) == u32(0);
    }
    if(_S372)
    {
        return;
    }
    var q_27 : Quat_0;
    var _S373 : vec3<f32> = chunk_box_world_2(k_26, &(q_27));
    var _S374 : f32 = geometry_0[u32(3) * k_26].w;
    var _S375 : u32 = a_strike_0() + u32(9) * body_of_0(k_26);
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
        var _S376 : u32 = ax_1;
        var _S377 : u32 = atomicMin(&(atom_0[_S375 + ax_1]), ordered_0(_S373[ax_1] - _S374));
        var _S378 : u32 = atomicMax(&(atom_0[_S375 + u32(3) + ax_1]), ordered_0(_S373[_S376] + _S374));
        ax_1 = ax_1 + u32(1);
    }
    return;
}

fn unordered_0( u_7 : u32) -> f32
{
    var _S379 : u32;
    if(((u_7 & (u32(2147483648)))) != u32(0))
    {
        _S379 = (u_7 & (u32(2147483647)));
    }
    else
    {
        _S379 = ~u_7;
    }
    return (bitcast<f32>((_S379)));
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_find(@builtin(global_invocation_id) id_26 : vec3<u32>)
{
    var b_45 : u32 = id_26.x;
    var _S380 : bool;
    if(b_45 >= (params_0.body_count_0))
    {
        _S380 = true;
    }
    else
    {
        _S380 = (params_0.shot_count_0) == u32(0);
    }
    if(_S380)
    {
        return;
    }
    var at_12 : u32 = a_strike_0() + u32(9) * b_45;
    var _S381 : u32 = atomicLoad(&(atom_0[at_12]));
    if(_S381 == u32(4294967295))
    {
        return;
    }
    var _S382 : u32 = atomicLoad(&(atom_0[at_12]));
    var _S383 : f32 = unordered_0(_S382);
    var _S384 : u32 = atomicLoad(&(atom_0[at_12 + u32(1)]));
    var _S385 : f32 = unordered_0(_S384);
    var _S386 : u32 = atomicLoad(&(atom_0[at_12 + u32(2)]));
    var _S387 : vec3<f32> = vec3<f32>(_S383, _S385, unordered_0(_S386));
    var _S388 : u32 = atomicLoad(&(atom_0[at_12 + u32(3)]));
    var _S389 : f32 = unordered_0(_S388);
    var _S390 : u32 = atomicLoad(&(atom_0[at_12 + u32(4)]));
    var _S391 : f32 = unordered_0(_S390);
    var _S392 : u32 = atomicLoad(&(atom_0[at_12 + u32(5)]));
    var _S393 : vec3<f32> = vec3<f32>(_S389, _S391, unordered_0(_S392));
    var _S394 : vec3<f32> = from_rubble_0(bodies_0[u32(7) * b_45 + u32(3)].xyz);
    var _S395 : f32 = float_bits_0(u32(2139095040));
    var k_27 : u32 = u32(0);
    loop
    {
        if(k_27 < (params_0.shot_count_0))
        {
        }
        else
        {
            break;
        }
        var info_2 : vec4<f32> = shots_0[u32(2) * k_27];
        var sb_1 : u32 = u32(7) * (bitcast<u32>((info_2.x)));
        if((bodies_0[sb_1].w) == 0.0f)
        {
            k_27 = k_27 + u32(1);
            continue;
        }
        var c_19 : vec3<f32> = from_rubble_0(bodies_0[sb_1].xyz);
        var rel_3 : vec3<f32> = from_rubble_0(bodies_0[sb_1 + u32(3)].xyz) - _S394;
        if(!((length(rel_3)) > (params_0.strike_speed_0)))
        {
            k_27 = k_27 + u32(1);
            continue;
        }
        var _S396 : vec3<f32> = vec3<f32>(info_2.y);
        var _S397 : vec3<f32> = _S387 - _S396;
        var _S398 : vec3<f32> = _S393 + _S396;
        var t0_1 : f32 = 0.0f;
        var t1_1 : f32 = params_0.dt_0;
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
            var _S399 : u32 = ax_2;
            if((abs(rel_3[ax_2])) < 9.999999960041972e-13f)
            {
                var _S400 : u32 = ax_2;
                if((c_19[ax_2]) < (_S397[ax_2]))
                {
                    _S380 = true;
                }
                else
                {
                    _S380 = (c_19[_S400]) > (_S398[ax_2]);
                }
                var t0_2 : f32;
                if(_S380)
                {
                    t0_2 = _S395;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + u32(1);
                continue;
            }
            var ta_1 : f32 = (_S397[ax_2] - c_19[ax_2]) / rel_3[_S399];
            var tb_1 : f32 = (_S398[ax_2] - c_19[ax_2]) / rel_3[_S399];
            var _S401 : f32 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S401;
            ax_2 = ax_2 + u32(1);
        }
        if(t0_1 <= t1_1)
        {
            var st_4 : u32 = strike_at_0(b_45);
            frames_0[st_4] = vec4<f32>(c_19, f32(k_27 + u32(1)));
            frames_0[st_4 + u32(1)] = vec4<f32>(rel_3, t0_1);
            return;
        }
        k_27 = k_27 + u32(1);
    }
    return;
}

fn struck_2( b_46 : u32,  shot_5 : ptr<function, u32>) -> bool
{
    var k_28 : u32 = u32(frames_0[strike_at_0(b_46)].w);
    (*shot_5) = k_28 - u32(1);
    return k_28 != u32(0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_entry(@builtin(global_invocation_id) id_27 : vec3<u32>)
{
    var k_29 : u32 = id_27.x;
    var _S402 : bool;
    if(k_29 >= (params_0.chunk_count_0))
    {
        _S402 = true;
    }
    else
    {
        _S402 = (params_0.shot_count_0) == u32(0);
    }
    if(_S402)
    {
        return;
    }
    var b_47 : u32 = body_of_0(k_29);
    var shot_6 : u32;
    var _S403 : bool = struck_2(b_47, &(shot_6));
    if(!_S403)
    {
        return;
    }
    var _S404 : u32 = strike_at_0(b_47);
    var s1_3 : vec4<f32> = frames_0[_S404 + u32(1)];
    var from_3 : vec3<f32> = frames_0[_S404].xyz;
    var rel_4 : vec3<f32> = frames_0[_S404 + u32(1)].xyz;
    var d_21 : vec3<f32> = rel_4 / vec3<f32>(max(length(rel_4), 9.999999960041972e-13f));
    var _S405 : f32 = shots_0[u32(2) * shot_6].y;
    var q_28 : Quat_0;
    var cc_0 : vec3<f32> = chunk_box_world_2(k_29, &(q_28));
    var _S406 : u32 = u32(3) * k_29;
    var _S407 : Quat_0 = normalized_0(quat_mul_0(q_28, quat_of_0(geometry_0[_S406 + u32(2)])));
    var _S408 : vec3<f32> = geometry_0[_S406 + u32(1)].xyz;
    var _S409 : f32 = float_bits_0(u32(2139095040));
    var t_in_7 : f32 = 0.0f;
    var t_out_0 : f32 = params_0.dt_0;
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
        var e_2 : vec3<f32> = vec3<f32>(0.0f);
        e_2[ax_3] = 1.0f;
        var axis_1 : vec3<f32> = rotate_0(_S407, e_2);
        var o_7 : f32 = dot(from_3 - cc_0, axis_1);
        var dv_1 : f32 = dot(rel_4, axis_1);
        var ext_2 : f32 = _S408[ax_3] + _S405;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            var t_in_8 : f32;
            if((abs(o_7)) > ext_2)
            {
                t_in_8 = _S409;
            }
            else
            {
                t_in_8 = t_in_7;
            }
            t_in_7 = t_in_8;
            ax_3 = ax_3 + u32(1);
            continue;
        }
        var ta_2 : f32 = (- ext_2 - o_7) / dv_1;
        var tb_2 : f32 = (ext_2 - o_7) / dv_1;
        var _S410 : f32 = min(t_out_0, max(ta_2, tb_2));
        t_in_7 = max(t_in_7, min(ta_2, tb_2));
        t_out_0 = _S410;
        ax_3 = ax_3 + u32(1);
    }
    var at_13 : u32 = a_strike_0() + u32(9) * b_47;
    if(t_in_7 <= t_out_0)
    {
        var _S411 : u32 = atomicMin(&(atom_0[at_13 + u32(6)]), (bitcast<u32>((t_in_7))));
    }
    var r_20 : vec3<f32> = cc_0 - (from_3 + rel_4 * vec3<f32>(s1_3.w));
    var along_1 : f32 = dot(r_20, d_21);
    var _S412 : u32 = atomicMin(&(atom_0[at_13 + u32(7)]), (bitcast<u32>((length(r_20 - d_21 * vec3<f32>(along_1)) + abs(along_1) * 0.00100000004749745f))));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_nearest(@builtin(global_invocation_id) id_28 : vec3<u32>)
{
    var k_30 : u32 = id_28.x;
    var _S413 : bool;
    if(k_30 >= (params_0.chunk_count_0))
    {
        _S413 = true;
    }
    else
    {
        _S413 = (params_0.shot_count_0) == u32(0);
    }
    if(_S413)
    {
        return;
    }
    var b_48 : u32 = body_of_0(k_30);
    var shot_7 : u32;
    if(!struck_2(b_48, &(shot_7)))
    {
        return;
    }
    var _S414 : u32 = strike_at_0(b_48);
    var _S415 : vec3<f32> = frames_0[_S414 + u32(1)].xyz;
    var d_22 : vec3<f32> = _S415 / vec3<f32>(max(length(_S415), 9.999999960041972e-13f));
    var q_29 : Quat_0;
    var r_21 : vec3<f32> = chunk_box_world_2(k_30, &(q_29)) - (frames_0[_S414].xyz + _S415 * vec3<f32>(frames_0[_S414 + u32(1)].w));
    var along_2 : f32 = dot(r_21, d_22);
    var at_14 : u32 = a_strike_0() + u32(9) * b_48;
    var _S416 : u32 = (bitcast<u32>((length(r_21 - d_22 * vec3<f32>(along_2)) + abs(along_2) * 0.00100000004749745f)));
    var _S417 : u32 = atomicLoad(&(atom_0[at_14 + u32(7)]));
    if(_S416 == _S417)
    {
        var _S418 : u32 = atomicMin(&(atom_0[at_14 + u32(8)]), k_30);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_force(@builtin(global_invocation_id) id_29 : vec3<u32>)
{
    var b_49 : u32 = id_29.x;
    var _S419 : bool;
    if(b_49 >= (params_0.body_count_0))
    {
        _S419 = true;
    }
    else
    {
        _S419 = (params_0.shot_count_0) == u32(0);
    }
    if(_S419)
    {
        return;
    }
    var shot_8 : u32;
    var _S420 : bool = struck_2(b_49, &(shot_8));
    if(!_S420)
    {
        return;
    }
    var at_15 : u32 = a_strike_0() + u32(9) * b_49;
    var _S421 : u32 = atomicLoad(&(atom_0[at_15 + u32(6)]));
    var t_in_9 : f32 = (bitcast<f32>((_S421)));
    var entry_8 : u32 = atomicLoad(&(atom_0[at_15 + u32(8)]));
    var st_5 : u32 = strike_at_0(b_49);
    if(!(t_in_9 <= (params_0.dt_0)))
    {
        _S419 = true;
    }
    else
    {
        _S419 = entry_8 == u32(4294967295);
    }
    if(_S419)
    {
        frames_0[st_5][i32(3)] = 0.0f;
        return;
    }
    var rel_5 : vec3<f32> = frames_0[st_5 + u32(1)].xyz;
    var speed_16 : f32 = length(rel_5);
    var info_3 : vec4<f32> = shots_0[u32(2) * shot_8];
    var _S422 : f32 = max(info_3.z, 0.00100000004749745f);
    var m_imp_0 : f32 = info_3.w;
    var e_imp_0 : f32 = shots_0[u32(2) * shot_8 + u32(1)].x;
    var contact_18 : vec3<f32> = frames_0[st_5].xyz + rel_5 * vec3<f32>(t_in_9) + rel_5 / vec3<f32>(max(speed_16, 9.999999960041972e-13f)) * vec3<f32>(info_3.y);
    var inv_mass_5 : f32 = bodies_0[u32(7) * b_49].w;
    var anchored_1 : bool = inv_mass_5 == 0.0f;
    var m_body_1 : f32;
    if(anchored_1)
    {
        m_body_1 = 0.0f;
    }
    else
    {
        m_body_1 = 1.0f / inv_mass_5;
    }
    var m_eff_6 : f32;
    if(anchored_1)
    {
        m_eff_6 = m_imp_0;
    }
    else
    {
        m_eff_6 = m_imp_0 * m_body_1 / (m_imp_0 + m_body_1);
    }
    var modulus_1 : f32 = 1.0f / (1.0f / e_imp_0 + 1.0f / geometry_0[u32(3) * entry_8 + u32(1)].w);
    frames_0[st_5 + u32(2)] = vec4<f32>(contact_18, m_eff_6 * speed_16 / (2.86999988555908203f * pow(m_eff_6 * m_eff_6 / (_S422 * modulus_1 * modulus_1 * max(speed_16, 0.00100000004749745f)), 0.20000000298023224f)));
    return;
}

fn strike_share_breaks_0( bond_5 : vec4<u32>,  contact_19 : vec3<f32>,  force_2 : f32) -> bool
{
    var q_30 : Quat_0;
    var pa_2 : vec3<f32> = chunk_box_world_1(bond_5.x, &(q_30));
    var pb_3 : vec3<f32> = chunk_box_world_1(bond_5.y, &(q_30));
    var dist_0 : f32 = length(vec3<f32>(0.5f) * (pa_2 + pb_3) - contact_19);
    return (force_2 * min(1.0f, (bitcast<f32>((bond_5.w))) / (6.28318548202514648f * max(dist_0 * dist_0, 9.999999960041972e-13f)))) > (bitcast<f32>((bond_5.z)));
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_breaks(@builtin(global_invocation_id) id_30 : vec3<u32>)
{
    var i_7 : u32 = id_30.x;
    var _S423 : bool;
    if(i_7 >= (params_0.bond_count_0))
    {
        _S423 = true;
    }
    else
    {
        _S423 = (params_0.shot_count_0) == u32(0);
    }
    if(_S423)
    {
        _S423 = true;
    }
    else
    {
        _S423 = work_0[w_alive_0() + i_7] == u32(0);
    }
    if(_S423)
    {
        return;
    }
    var bond_6 : vec4<u32> = bonds_0[i_7];
    var b_50 : u32 = body_of_0(bond_6.x);
    var shot_9 : u32;
    if(!struck_2(b_50, &(shot_9)))
    {
        return;
    }
    if(strike_share_breaks_0(bond_6, frames_0[strike_at_0(b_50) + u32(2)].xyz, frames_0[strike_at_0(b_50) + u32(2)].w))
    {
        work_0[w_alive_0() + i_7] = u32(0);
        var _S424 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_50]), u32(1));
        var _S425 : u32 = a_counters_0();
        var _S426 : u32 = atomicAdd(&(atom_0[_S425]), u32(1));
        var _S427 : u32 = atomicAdd(&(atom_0[_S425 + u32(1)]), u32(1));
    }
    return;
}

fn a_lp_0( b_51 : u32) -> u32
{
    return a_strike_0() + u32(9) * params_0.body_count_0 + u32(14) * b_51;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_clear(@builtin(global_invocation_id) id_31 : vec3<u32>)
{
    var b_52 : u32 = id_31.x;
    if(b_52 >= (params_0.body_count_0))
    {
        return;
    }
    var at_16 : u32 = a_lp_0(b_52);
    var w_7 : u32 = u32(0);
    loop
    {
        if(w_7 < u32(12))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[at_16 + w_7]), u32(0));
        w_7 = w_7 + u32(1);
    }
    atomicStore(&(atom_0[at_16 + u32(12)]), u32(4294967295));
    atomicStore(&(atom_0[at_16 + u32(13)]), u32(0));
    return;
}

fn st_f_0( k_31 : u32,  state_0 : u32) -> u32
{
    return u32(22) * k_31 + state_0 + u32(6);
}

fn lp_f_0( k_32 : u32) -> u32
{
    return st_f_0(k_32, u32(0));
}

fn add64w_0( at_17 : u32,  value_1 : f32,  scale_2 : f32)
{
    var x_8 : f32 = round(value_1 * scale_2);
    var x_9 : f32;
    if(!((abs(x_8)) < 3.99999993722699776e+18f))
    {
        var _S428 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
        x_9 = 0.0f;
    }
    else
    {
        x_9 = x_8;
    }
    var u_8 : f32 = abs(x_9);
    var h_5 : f32 = floor(u_8 * 2.32830643653869629e-10f);
    var lo_2 : u32 = u32(u_8 - h_5 * 4.294967296e+09f);
    var hi_2 : u32 = u32(h_5);
    var lo_3 : u32;
    var hi_3 : u32;
    if(x_9 < 0.0f)
    {
        var lo_4 : u32 = ~lo_2 + u32(1);
        var _S429 : u32 = ~hi_2;
        if(lo_4 == u32(0))
        {
            lo_3 = u32(1);
        }
        else
        {
            lo_3 = u32(0);
        }
        var _S430 : u32 = _S429 + lo_3;
        lo_3 = lo_4;
        hi_3 = _S430;
    }
    else
    {
        lo_3 = lo_2;
        hi_3 = hi_2;
    }
    var old_1 : u32 = atomicAdd(&(atom_0[at_17]), lo_3);
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
        var _S431 : u32 = atomicAdd(&(atom_0[at_17 + u32(1)]), hi_4);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sum(@builtin(global_invocation_id) id_32 : vec3<u32>)
{
    var k_33 : u32 = id_32.x;
    if(k_33 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_17 : u32 = body_of_0(k_33);
    var q_31 : Quat_0;
    var origin_13 : vec3<f32>;
    structure_pose_4(slot_17, u32(7) * slot_17, &(q_31), &(origin_13));
    var qi_0 : Quat_0 = conj_0(q_31);
    var c0_3 : vec4<f32> = chunks_0[u32(4) * k_33];
    var _S432 : u32 = l_vec_0() + u32(2) * k_33;
    var f_10 : vec3<f32> = rotate_0(qi_0, loads_0[_S432].xyz) + vec3<f32>(c0_3.w) * rotate_0(qi_0, vec3<f32>(params_0.gx_0, params_0.gy_0, params_0.gz_0));
    var t_8 : vec3<f32> = rotate_0(qi_0, loads_0[_S432 + u32(1)].xyz);
    var _S433 : u32 = lp_f_0(k_33);
    lp_0[_S433] = vec4<f32>(f_10, 0.0f);
    lp_0[_S433 + u32(1)] = vec4<f32>(t_8, 0.0f);
    var _S434 : vec3<f32> = t_8 + cross(c0_3.xyz - frames_0[u32(2) * slot_17].xyz, f_10);
    var at_18 : u32 = a_lp_0(slot_17);
    var a_10 : u32 = u32(0);
    loop
    {
        if(a_10 < u32(3))
        {
        }
        else
        {
            break;
        }
        var _S435 : u32 = u32(2) * a_10;
        add64w_0(at_18 + _S435, f_10[a_10], params_0.lp_scale_0);
        add64w_0(at_18 + u32(6) + _S435, _S434[a_10], params_0.lp_scale_0);
        a_10 = a_10 + u32(1);
    }
    var _S436 : u32 = atomicMin(&(atom_0[at_18 + u32(12)]), k_33);
    if(((info_0[k_33] & (u32(1)))) != u32(0))
    {
        atomicStore(&(atom_0[at_18 + u32(13)]), u32(1));
    }
    return;
}

fn read64w3_0( at_19 : u32,  scale_3 : f32) -> vec3<f32>
{
    var _S437 : f32 = read64_0(at_19, scale_3);
    var _S438 : f32 = read64_0(at_19 + u32(2), scale_3);
    var _S439 : f32 = read64_0(at_19 + u32(4), scale_3);
    return vec3<f32>(_S437, _S438, _S439);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_relief(@builtin(global_invocation_id) id_33 : vec3<u32>)
{
    var k_34 : u32 = id_33.x;
    if(k_34 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_18 : u32 = body_of_0(k_34);
    var at_20 : u32 = a_lp_0(slot_18);
    var _S440 : u32 = atomicLoad(&(atom_0[at_20 + u32(13)]));
    var supported_0 : bool = _S440 != u32(0);
    var fixed_1 : bool;
    if(((info_0[k_34] & (u32(1)))) != u32(0))
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            var _S441 : u32 = atomicLoad(&(atom_0[at_20 + u32(12)]));
            fixed_1 = _S441 == k_34;
        }
        else
        {
            fixed_1 = false;
        }
    }
    var _S442 : u32 = lp_f_0(k_34);
    var f_11 : vec4<f32> = lp_0[_S442];
    var _S443 : u32 = _S442 + u32(1);
    var t_9 : vec4<f32> = lp_0[_S443];
    var b_53 : u32 = u32(7) * slot_18;
    var inv_mass_6 : f32 = bodies_0[b_53].w;
    var _S444 : bool;
    if(!supported_0)
    {
        _S444 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S444 = false;
    }
    if(_S444)
    {
        var force_3 : vec3<f32> = read64w3_0(at_20, params_0.lp_scale_0);
        var moment_0 : vec3<f32> = read64w3_0(at_20 + u32(6), params_0.lp_scale_0);
        var _S445 : u32 = u32(2) * slot_18;
        var p_16 : Quat_0 = quat_of_0(frames_0[_S445 + u32(1)]);
        var alpha_0 : vec3<f32> = rotate_0(p_16, bodies_0[b_53 + u32(2)].xyz * rotate_0(conj_0(p_16), moment_0));
        var _S446 : u32 = u32(4) * k_34;
        var c0_4 : vec4<f32> = chunks_0[_S446];
        var c1_2 : vec4<f32> = chunks_0[_S446 + u32(1)];
        var c2_3 : vec4<f32> = chunks_0[_S446 + u32(2)];
        var _S447 : vec3<f32> = f_11.xyz - vec3<f32>(c0_4.w) * (force_3 * vec3<f32>(inv_mass_6) + cross(alpha_0, c0_4.xyz - frames_0[_S445].xyz));
        f_11.x = _S447.x;
        f_11.y = _S447.y;
        f_11.z = _S447.z;
        var _S448 : f32 = alpha_0.x;
        var _S449 : f32 = c1_2.w;
        var _S450 : f32 = alpha_0.y;
        var _S451 : f32 = c2_3.x;
        var _S452 : f32 = alpha_0.z;
        var _S453 : f32 = c2_3.y;
        var _S454 : vec3<f32> = t_9.xyz - vec3<f32>(c1_2.x * _S448 + _S449 * _S450 + _S451 * _S452, _S449 * _S448 + c1_2.y * _S450 + _S453 * _S452, _S451 * _S448 + _S453 * _S450 + c1_2.z * _S452);
        t_9.x = _S454.x;
        t_9.y = _S454.y;
        t_9.z = _S454.z;
    }
    var _S455 : f32;
    if(fixed_1)
    {
        _S455 = 1.0f;
    }
    else
    {
        _S455 = 0.0f;
    }
    f_11[i32(3)] = _S455;
    lp_0[_S442] = f_11;
    lp_0[_S443] = t_9;
    return;
}

fn adj_bond_0( at_21 : u32) -> u32
{
    return (bitcast<u32>((ext_0[params_0.ext_adj_0 + at_21 / u32(4)][at_21 % u32(4)])));
}

fn lp_intact_0( k_35 : u32) -> u32
{
    var range_0 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * k_35];
    var _S456 : u32 = (bitcast<u32>((range_0.x)));
    var _S457 : u32 = (bitcast<u32>((range_0.y)));
    var e_3 : u32 = u32(0);
    var intact_0 : u32 = u32(0);
    loop
    {
        if(e_3 < _S457)
        {
        }
        else
        {
            break;
        }
        var _S458 : u32;
        if(work_0[w_alive_0() + adj_bond_0(_S456 + e_3)] != u32(0))
        {
            _S458 = u32(1);
        }
        else
        {
            _S458 = u32(0);
        }
        var intact_1 : u32 = intact_0 + _S458;
        e_3 = e_3 + u32(1);
        intact_0 = intact_1;
    }
    return intact_0;
}

fn lp_l_0( k_36 : u32) -> u32
{
    return u32(22) * k_36 + u32(8);
}

struct JointBond_0
{
     geom0_0 : vec4<f32>,
     geom1_0 : vec4<f32>,
     stiff0_0 : vec4<f32>,
     stiff1_0 : vec4<f32>,
     rebar0_0 : vec4<f32>,
     rebar1_0 : vec4<f32>,
     ids_3 : vec4<u32>,
};

struct LpBond_0
{
     centroid_0 : vec3<f32>,
     n_4 : vec3<f32>,
     t1_2 : vec3<f32>,
     t2_0 : vec3<f32>,
     law_0 : JointBond_0,
};

fn lp_bond_0( i_8 : u32) -> LpBond_0
{
    var o_8 : u32 = u32(10) * i_8;
    var b_54 : LpBond_0;
    b_54.centroid_0 = ext_0[o_8].xyz;
    var _S459 : vec3<f32> = ext_0[o_8 + u32(1)].xyz;
    b_54.n_4 = _S459;
    var _S460 : vec3<f32> = ext_0[o_8 + u32(2)].xyz;
    b_54.t1_2 = _S460;
    b_54.t2_0 = cross(_S459, _S460);
    b_54.law_0.geom0_0 = ext_0[o_8 + u32(3)];
    b_54.law_0.geom1_0 = ext_0[o_8 + u32(4)];
    b_54.law_0.stiff0_0 = ext_0[o_8 + u32(5)];
    b_54.law_0.stiff1_0 = ext_0[o_8 + u32(6)];
    b_54.law_0.rebar0_0 = ext_0[o_8 + u32(7)];
    b_54.law_0.rebar1_0 = ext_0[o_8 + u32(8)];
    b_54.law_0.ids_3 = (bitcast<vec4<u32>>((ext_0[o_8 + u32(9)])));
    return b_54;
}

fn tri_0( i_9 : u32,  j_4 : u32) -> u32
{
    return i_9 * (i_9 + u32(1)) / u32(2) + j_4;
}

fn lp_factor_0( k_37 : u32) -> bool
{
    var q_32 : u32;
    var a_11 : u32;
    var p2_0 : u32;
    var q2_0 : u32;
    var a2_0 : u32;
    var s_13 : f32;
    var _S461 : u32 = lp_l_0(k_37);
    var _S462 : u32 = lp_intact_0(k_37) + u32(1);
    if((bitcast<u32>((lp_0[_S461 + u32(5)].w))) == _S462)
    {
        return true;
    }
    var d_23 : array<array<f32, i32(6)>, i32(6)>;
    var p_17 : u32 = u32(0);
    loop
    {
        if(p_17 < u32(6))
        {
        }
        else
        {
            break;
        }
        q_32 = u32(0);
        loop
        {
            if(q_32 < u32(6))
            {
            }
            else
            {
                break;
            }
            d_23[p_17][q_32] = 0.0f;
            q_32 = q_32 + u32(1);
        }
        p_17 = p_17 + u32(1);
    }
    var _S463 : u32 = u32(4) * k_37;
    var range_1 : vec4<f32> = ext_0[params_0.ext_chunks_0 + _S463];
    var _S464 : u32 = (bitcast<u32>((range_1.x)));
    var _S465 : u32 = (bitcast<u32>((range_1.y)));
    q_32 = u32(0);
    loop
    {
        if(q_32 < _S465)
        {
        }
        else
        {
            break;
        }
        var i_10 : u32 = adj_bond_0(_S464 + q_32);
        if(work_0[w_alive_0() + i_10] == u32(0))
        {
            q_32 = q_32 + u32(1);
            continue;
        }
        var b_55 : LpBond_0 = lp_bond_0(i_10);
        var _S466 : vec3<f32> = b_55.centroid_0 - chunks_0[_S463].xyz;
        var _S467 : f32 = b_55.law_0.stiff0_0.y;
        var _S468 : array<f32, i32(6)> = array<f32, i32(6)>( _S467, _S467, b_55.law_0.stiff0_0.x, b_55.law_0.stiff0_0.z, b_55.law_0.stiff0_0.w, b_55.law_0.stiff1_0.x );
        var _S469 : array<vec3<f32>, i32(3)> = array<vec3<f32>, i32(3)>( b_55.t1_2, b_55.t2_0, b_55.n_4 );
        var j_5 : array<array<f32, i32(6)>, i32(6)>;
        a_11 = u32(0);
        loop
        {
            if(a_11 < u32(3))
            {
            }
            else
            {
                break;
            }
            var c_20 : vec3<f32> = cross(_S466, _S469[a_11]);
            var _S470 : f32 = _S469[a_11].x;
            j_5[a_11][i32(0)] = _S470;
            var _S471 : f32 = _S469[a_11].y;
            j_5[a_11][i32(1)] = _S471;
            var _S472 : f32 = _S469[a_11].z;
            j_5[a_11][i32(2)] = _S472;
            j_5[a_11][i32(3)] = c_20.x;
            j_5[a_11][i32(4)] = c_20.y;
            j_5[a_11][i32(5)] = c_20.z;
            j_5[u32(3) + a_11][i32(0)] = 0.0f;
            j_5[u32(3) + a_11][i32(1)] = 0.0f;
            j_5[u32(3) + a_11][i32(2)] = 0.0f;
            j_5[u32(3) + a_11][i32(3)] = _S470;
            j_5[u32(3) + a_11][i32(4)] = _S471;
            j_5[u32(3) + a_11][i32(5)] = _S472;
            a_11 = a_11 + u32(1);
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
                s_13 = 0.0f;
                loop
                {
                    if(a2_0 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var s_14 : f32 = s_13 + j_5[a2_0][p2_0] * _S468[a2_0] * j_5[a2_0][q2_0];
                    a2_0 = a2_0 + u32(1);
                    s_13 = s_14;
                }
                d_23[p2_0][q2_0] = d_23[p2_0][q2_0] + s_13;
                q2_0 = q2_0 + u32(1);
            }
            p2_0 = p2_0 + u32(1);
        }
        q_32 = q_32 + u32(1);
    }
    var l_1 : array<f32, i32(21)>;
    a_11 = u32(0);
    loop
    {
        if(a_11 < u32(21))
        {
        }
        else
        {
            break;
        }
        l_1[a_11] = 0.0f;
        a_11 = a_11 + u32(1);
    }
    var ok_0 : bool = true;
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
            s_13 = d_23[p2_0][q2_0];
            loop
            {
                if(a2_0 < q2_0)
                {
                }
                else
                {
                    break;
                }
                var s_15 : f32 = s_13 - l_1[tri_0(p2_0, a2_0)] * l_1[tri_0(q2_0, a2_0)];
                a2_0 = a2_0 + u32(1);
                s_13 = s_15;
            }
            if(p2_0 == q2_0)
            {
                var _S473 : bool;
                if(ok_1)
                {
                    _S473 = s_13 > 0.0f;
                }
                else
                {
                    _S473 = false;
                }
                l_1[tri_0(p2_0, p2_0)] = sqrt(max(s_13, 1.00000000317107685e-30f));
                ok_1 = _S473;
            }
            else
            {
                l_1[tri_0(p2_0, q2_0)] = s_13 / l_1[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + u32(1);
        }
        var _S474 : u32 = p2_0 + u32(1);
        ok_0 = ok_1;
        p2_0 = _S474;
    }
    if(!ok_0)
    {
        return false;
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
        var _S475 : u32 = u32(4) * q2_0;
        var _S476 : f32 = l_1[_S475];
        var _S477 : u32 = _S475 + u32(1);
        if(_S477 < u32(21))
        {
            s_13 = l_1[_S477];
        }
        else
        {
            s_13 = 0.0f;
        }
        var _S478 : u32 = _S475 + u32(2);
        var _S479 : f32;
        if(_S478 < u32(21))
        {
            _S479 = l_1[_S478];
        }
        else
        {
            _S479 = 0.0f;
        }
        var _S480 : f32;
        if(q2_0 == u32(5))
        {
            _S480 = (bitcast<f32>((_S462)));
        }
        else
        {
            _S480 = l_1[_S475 + u32(3)];
        }
        lp_0[_S461 + q2_0] = vec4<f32>(_S476, s_13, _S479, _S480);
        q2_0 = q2_0 + u32(1);
    }
    return true;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_block(@builtin(global_invocation_id) id_34 : vec3<u32>)
{
    var k_38 : u32 = id_34.x;
    var _S481 : bool;
    if(k_38 >= (params_0.chunk_count_0))
    {
        _S481 = true;
    }
    else
    {
        _S481 = (lp_0[lp_f_0(k_38)].w) != 0.0f;
    }
    if(_S481)
    {
        return;
    }
    var _S482 : bool = lp_factor_0(k_38);
    if(!_S482)
    {
        lp_0[lp_f_0(k_38)][i32(3)] = 1.0f;
        var _S483 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(32));
    }
    return;
}

fn st_x_0( k_39 : u32,  state_1 : u32,  copy_0 : u32) -> u32
{
    return u32(22) * k_39 + state_1 + u32(2) * copy_0;
}

fn st_d_0( k_40 : u32,  state_2 : u32) -> u32
{
    return u32(22) * k_40 + state_2 + u32(4);
}

fn st_sweep_0( k_41 : u32,  state_3 : u32,  from_4 : u32,  c1_3 : f32,  c2_4 : f32)
{
    var ka_0 : u32;
    var kb_0 : u32;
    var s_16 : f32;
    var to_3 : u32 = u32(1) - from_4;
    var _S484 : u32 = st_x_0(k_41, state_3, from_4);
    var x0_0 : vec4<f32> = lp_0[_S484];
    var x1_0 : vec4<f32> = lp_0[_S484 + u32(1)];
    var _S485 : u32 = st_f_0(k_41, state_3);
    var f_12 : vec4<f32> = lp_0[_S485];
    if((lp_0[_S485].w) != 0.0f)
    {
        var _S486 : u32 = st_x_0(k_41, state_3, to_3);
        lp_0[_S486] = x0_0;
        lp_0[_S486 + u32(1)] = x1_0;
        var _S487 : u32 = st_d_0(k_41, state_3);
        var _S488 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S487] = _S488;
        lp_0[_S487 + u32(1)] = _S488;
        return;
    }
    var _S489 : f32 = f_12.y;
    var _S490 : f32 = f_12.z;
    var r_22 : array<f32, i32(6)>;
    r_22[i32(0)] = f_12.x;
    r_22[i32(1)] = _S489;
    r_22[i32(2)] = _S490;
    r_22[i32(3)] = 0.0f;
    r_22[i32(4)] = 0.0f;
    r_22[i32(5)] = 0.0f;
    var t_10 : vec3<f32> = lp_0[_S485 + u32(1)].xyz;
    r_22[i32(3)] = t_10.x;
    r_22[i32(4)] = t_10.y;
    r_22[i32(5)] = t_10.z;
    var range_2 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * k_41];
    var _S491 : u32 = (bitcast<u32>((range_2.x)));
    var _S492 : u32 = (bitcast<u32>((range_2.y)));
    var e_4 : u32 = u32(0);
    loop
    {
        if(e_4 < _S492)
        {
        }
        else
        {
            break;
        }
        var _S493 : u32 = u32(5) * (_S491 + e_4);
        var s0_4 : vec4<f32> = lp_sides_0[_S493];
        if(work_0[w_alive_0() + (bitcast<u32>((s0_4.w)))] == u32(0))
        {
            e_4 = e_4 + u32(1);
            continue;
        }
        var s1_4 : vec4<f32> = lp_sides_0[_S493 + u32(1)];
        var s2_0 : vec4<f32> = lp_sides_0[_S493 + u32(2)];
        var s3_0 : vec4<f32> = lp_sides_0[_S493 + u32(3)];
        var s4_0 : vec4<f32> = lp_sides_0[_S493 + u32(4)];
        var first_side_0 : bool = (s4_0.w) != 0.0f;
        if(first_side_0)
        {
            ka_0 = k_41;
        }
        else
        {
            ka_0 = (bitcast<u32>((s1_4.w)));
        }
        if(first_side_0)
        {
            kb_0 = (bitcast<u32>((s1_4.w)));
        }
        else
        {
            kb_0 = k_41;
        }
        var ra_1 : vec3<f32> = s0_4.xyz;
        var rb_1 : vec3<f32> = s1_4.xyz;
        var t1_3 : vec3<f32> = s2_0.xyz;
        var n_5 : vec3<f32> = s3_0.xyz;
        var t2_1 : vec3<f32> = cross(n_5, t1_3);
        var _S494 : u32 = st_x_0(ka_0, state_3, from_4);
        var _S495 : u32 = st_x_0(kb_0, state_3, from_4);
        var _S496 : vec3<f32> = lp_0[_S495 + u32(1)].xyz;
        var _S497 : vec3<f32> = lp_0[_S494 + u32(1)].xyz;
        var delta_0 : vec3<f32> = lp_0[_S495].xyz + cross(_S496, rb_1) - (lp_0[_S494].xyz + cross(_S497, ra_1));
        var phi_0 : vec3<f32> = _S496 - _S497;
        var _S498 : f32 = s2_0.w;
        var fb_0 : vec3<f32> = t1_3 * vec3<f32>((_S498 * dot(delta_0, t1_3))) + t2_1 * vec3<f32>((_S498 * dot(delta_0, t2_1))) + n_5 * vec3<f32>((s3_0.w * dot(delta_0, n_5)));
        var mb_0 : vec3<f32> = t1_3 * vec3<f32>((s4_0.x * dot(phi_0, t1_3))) + t2_1 * vec3<f32>((s4_0.y * dot(phi_0, t2_1))) + n_5 * vec3<f32>((s4_0.z * dot(phi_0, n_5)));
        var arm_0 : vec3<f32>;
        if(first_side_0)
        {
            arm_0 = ra_1;
        }
        else
        {
            arm_0 = rb_1;
        }
        var lf_0 : vec3<f32>;
        if(first_side_0)
        {
            lf_0 = fb_0;
        }
        else
        {
            lf_0 = (vec3<f32>(0) - fb_0);
        }
        var _S499 : vec3<f32>;
        if(first_side_0)
        {
            _S499 = mb_0;
        }
        else
        {
            _S499 = (vec3<f32>(0) - mb_0);
        }
        var lm_0 : vec3<f32> = _S499 + cross(arm_0, lf_0);
        r_22[i32(0)] = r_22[i32(0)] + lf_0.x;
        r_22[i32(1)] = r_22[i32(1)] + lf_0.y;
        r_22[i32(2)] = r_22[i32(2)] + lf_0.z;
        r_22[i32(3)] = r_22[i32(3)] + lm_0.x;
        r_22[i32(4)] = r_22[i32(4)] + lm_0.y;
        r_22[i32(5)] = r_22[i32(5)] + lm_0.z;
        e_4 = e_4 + u32(1);
    }
    var l_2 : array<f32, i32(21)>;
    ka_0 = u32(0);
    loop
    {
        if(ka_0 < u32(21))
        {
        }
        else
        {
            break;
        }
        l_2[ka_0] = 0.0f;
        ka_0 = ka_0 + u32(1);
    }
    kb_0 = u32(0);
    loop
    {
        if(kb_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        var v_14 : vec4<f32> = lp_0[lp_l_0(k_41) + kb_0];
        var _S500 : u32 = u32(4) * kb_0;
        l_2[_S500] = lp_0[lp_l_0(k_41) + kb_0].x;
        var _S501 : u32 = _S500 + u32(1);
        if(_S501 < u32(21))
        {
            l_2[_S501] = v_14.y;
        }
        var _S502 : u32 = _S500 + u32(2);
        if(_S502 < u32(21))
        {
            l_2[_S502] = v_14.z;
        }
        var _S503 : u32 = _S500 + u32(3);
        if(_S503 < u32(21))
        {
            l_2[_S503] = v_14.w;
        }
        kb_0 = kb_0 + u32(1);
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
        var p_18 : u32 = u32(0);
        s_16 = r_22[i2_0];
        loop
        {
            if(p_18 < i2_0)
            {
            }
            else
            {
                break;
            }
            var s_17 : f32 = s_16 - l_2[tri_0(i2_0, p_18)] * y_3[p_18];
            p_18 = p_18 + u32(1);
            s_16 = s_17;
        }
        y_3[i2_0] = s_16 / l_2[tri_0(i2_0, i2_0)];
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
        s_16 = y_3[i3_0];
        loop
        {
            if(p2_1 < i32(6))
            {
            }
            else
            {
                break;
            }
            var s_18 : f32 = s_16 - l_2[tri_0(u32(p2_1), u32(i3_0))] * z_4[p2_1];
            p2_1 = p2_1 + i32(1);
            s_16 = s_18;
        }
        var _S504 : u32 = u32(i3_0);
        z_4[i3_0] = s_16 / l_2[tri_0(_S504, _S504)];
        i3_0 = i3_0 - i32(1);
    }
    var _S505 : u32 = st_d_0(k_41, state_3);
    var d0_0 : vec4<f32> = lp_0[_S505];
    var _S506 : u32 = _S505 + u32(1);
    var d1_0 : vec4<f32> = lp_0[_S506];
    var _S507 : vec3<f32> = vec3<f32>(c1_3);
    var _S508 : vec3<f32> = vec3<f32>(c2_4);
    var _S509 : vec3<f32> = _S507 * lp_0[_S505].xyz + _S508 * vec3<f32>(z_4[i32(0)], z_4[i32(1)], z_4[i32(2)]);
    d0_0.x = _S509.x;
    d0_0.y = _S509.y;
    d0_0.z = _S509.z;
    var _S510 : vec3<f32> = _S507 * lp_0[_S506].xyz + _S508 * vec3<f32>(z_4[i32(3)], z_4[i32(4)], z_4[i32(5)]);
    d1_0.x = _S510.x;
    d1_0.y = _S510.y;
    d1_0.z = _S510.z;
    lp_0[_S505] = d0_0;
    lp_0[_S506] = d1_0;
    var _S511 : u32 = st_x_0(k_41, state_3, to_3);
    lp_0[_S511] = vec4<f32>(x0_0.xyz + d0_0.xyz, 0.0f);
    lp_0[_S511 + u32(1)] = vec4<f32>(x1_0.xyz + d1_0.xyz, 0.0f);
    return;
}

fn lp_sweep_0( k_42 : u32,  from_5 : u32)
{
    st_sweep_0(k_42, u32(0), from_5, params_0.lp_c1_0, params_0.lp_c2_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sweep_even(@builtin(global_invocation_id) id_35 : vec3<u32>)
{
    var _S512 : u32 = id_35.x;
    if(_S512 < (params_0.chunk_count_0))
    {
        lp_sweep_0(_S512, u32(0));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sweep_odd(@builtin(global_invocation_id) id_36 : vec3<u32>)
{
    var _S513 : u32 = id_36.x;
    if(_S513 < (params_0.chunk_count_0))
    {
        lp_sweep_0(_S513, u32(1));
    }
    return;
}

fn st_force_0( b_56 : LpBond_0,  ka_1 : u32,  kb_1 : u32,  state_4 : u32,  c_21 : u32,  q_lin_0 : ptr<function, vec3<f32>>,  q_ang_0 : ptr<function, vec3<f32>>)
{
    var _S514 : u32 = st_x_0(ka_1, state_4, c_21);
    var _S515 : u32 = st_x_0(kb_1, state_4, c_21);
    var _S516 : vec3<f32> = lp_0[_S515 + u32(1)].xyz;
    var _S517 : vec3<f32> = lp_0[_S514 + u32(1)].xyz;
    var delta_1 : vec3<f32> = lp_0[_S515].xyz + cross(_S516, b_56.centroid_0 - chunks_0[u32(4) * kb_1].xyz) - (lp_0[_S514].xyz + cross(_S517, b_56.centroid_0 - chunks_0[u32(4) * ka_1].xyz));
    var phi_1 : vec3<f32> = _S516 - _S517;
    var kt_0 : f32 = b_56.law_0.stiff1_0.x;
    var _S518 : f32 = b_56.law_0.stiff0_0.y;
    (*q_lin_0) = vec3<f32>(_S518 * dot(delta_1, b_56.t1_2), _S518 * dot(delta_1, b_56.t2_0), b_56.law_0.stiff0_0.x * dot(delta_1, b_56.n_4));
    (*q_ang_0) = vec3<f32>(b_56.law_0.stiff0_0.z * dot(phi_1, b_56.t1_2), b_56.law_0.stiff0_0.w * dot(phi_1, b_56.t2_0), kt_0 * dot(phi_1, b_56.n_4));
    return;
}

fn st_force_1( b_57 : LpBond_0,  ka_2 : u32,  kb_2 : u32,  state_5 : u32,  c_22 : u32,  q_lin_1 : ptr<function, vec3<f32>>,  q_ang_1 : ptr<function, vec3<f32>>)
{
    var _S519 : u32 = st_x_0(ka_2, state_5, c_22);
    var _S520 : u32 = st_x_0(kb_2, state_5, c_22);
    var _S521 : vec3<f32> = lp_0[_S520 + u32(1)].xyz;
    var _S522 : vec3<f32> = lp_0[_S519 + u32(1)].xyz;
    var delta_2 : vec3<f32> = lp_0[_S520].xyz + cross(_S521, b_57.centroid_0 - chunks_0[u32(4) * kb_2].xyz) - (lp_0[_S519].xyz + cross(_S522, b_57.centroid_0 - chunks_0[u32(4) * ka_2].xyz));
    var phi_2 : vec3<f32> = _S521 - _S522;
    var kt_1 : f32 = b_57.law_0.stiff1_0.x;
    var _S523 : f32 = b_57.law_0.stiff0_0.y;
    (*q_lin_1) = vec3<f32>(_S523 * dot(delta_2, b_57.t1_2), _S523 * dot(delta_2, b_57.t2_0), b_57.law_0.stiff0_0.x * dot(delta_2, b_57.n_4));
    (*q_ang_1) = vec3<f32>(b_57.law_0.stiff0_0.z * dot(phi_2, b_57.t1_2), b_57.law_0.stiff0_0.w * dot(phi_2, b_57.t2_0), kt_1 * dot(phi_2, b_57.n_4));
    return;
}

fn st_force_2( b_58 : LpBond_0,  ka_3 : u32,  kb_3 : u32,  state_6 : u32,  c_23 : u32,  q_lin_2 : ptr<function, vec3<f32>>,  q_ang_2 : ptr<function, vec3<f32>>)
{
    var _S524 : u32 = st_x_0(ka_3, state_6, c_23);
    var _S525 : u32 = st_x_0(kb_3, state_6, c_23);
    var _S526 : vec3<f32> = lp_0[_S525 + u32(1)].xyz;
    var _S527 : vec3<f32> = lp_0[_S524 + u32(1)].xyz;
    var delta_3 : vec3<f32> = lp_0[_S525].xyz + cross(_S526, b_58.centroid_0 - chunks_0[u32(4) * kb_3].xyz) - (lp_0[_S524].xyz + cross(_S527, b_58.centroid_0 - chunks_0[u32(4) * ka_3].xyz));
    var phi_3 : vec3<f32> = _S526 - _S527;
    var kt_2 : f32 = b_58.law_0.stiff1_0.x;
    var _S528 : f32 = b_58.law_0.stiff0_0.y;
    (*q_lin_2) = vec3<f32>(_S528 * dot(delta_3, b_58.t1_2), _S528 * dot(delta_3, b_58.t2_0), b_58.law_0.stiff0_0.x * dot(delta_3, b_58.n_4));
    (*q_ang_2) = vec3<f32>(b_58.law_0.stiff0_0.z * dot(phi_3, b_58.t1_2), b_58.law_0.stiff0_0.w * dot(phi_3, b_58.t2_0), kt_2 * dot(phi_3, b_58.n_4));
    return;
}

fn lp_force_0( b_59 : LpBond_0,  ka_4 : u32,  kb_4 : u32,  c_24 : u32,  q_lin_3 : ptr<function, vec3<f32>>,  q_ang_3 : ptr<function, vec3<f32>>)
{
    st_force_1(b_59, ka_4, kb_4, u32(0), c_24, &((*q_lin_3)), &((*q_ang_3)));
    return;
}

fn lp_force_1( b_60 : LpBond_0,  ka_5 : u32,  kb_5 : u32,  c_25 : u32,  q_lin_4 : ptr<function, vec3<f32>>,  q_ang_4 : ptr<function, vec3<f32>>)
{
    st_force_2(b_60, ka_5, kb_5, u32(0), c_25, &((*q_lin_4)), &((*q_ang_4)));
    return;
}

fn fsqrt_0( a_12 : f32) -> f32
{
    return sqrt(a_12);
}

struct Measures_0
{
     tension_0 : f32,
     shear_0 : f32,
     normal_compression_0 : f32,
     compression_0 : f32,
     compressive_force_0 : f32,
};

fn stress_measures_0( b_61 : JointBond_0,  q_lin_5 : vec3<f32>,  q_ang_5 : vec3<f32>) -> Measures_0
{
    var area_0 : f32 = b_61.geom0_0.x;
    var _S529 : f32 = q_lin_5.z;
    var axial_0 : f32 = fdiv_0(_S529, area_0);
    var bending_0 : f32 = fdiv_0(abs(q_ang_5.x), b_61.geom1_0.x) + fdiv_0(abs(q_ang_5.y), b_61.geom1_0.y);
    var _S530 : f32 = q_lin_5.x;
    var _S531 : f32 = q_lin_5.y;
    var shear_1 : f32 = fdiv_0(fsqrt_0(_S530 * _S530 + _S531 * _S531), area_0) + fdiv_0(abs(q_ang_5.z), b_61.geom0_0.w);
    var m_3 : Measures_0;
    m_3.tension_0 = axial_0 + bending_0;
    m_3.shear_0 = shear_1;
    var _S532 : f32 = - axial_0;
    m_3.normal_compression_0 = max(_S532, 0.0f);
    m_3.compression_0 = _S532 + bending_0;
    m_3.compressive_force_0 = max(- _S529, 0.0f);
    return m_3;
}

fn lp_material_0( m_4 : u32) -> JointMaterial_0
{
    var o_9 : u32 = params_0.ext_mat_0 + u32(5) * m_4;
    var j_6 : JointMaterial_0;
    j_6.strength_0 = ext_0[o_9];
    j_6.energy_0 = ext_0[o_9 + u32(1)];
    j_6.dif_0 = ext_0[o_9 + u32(2)];
    j_6.misc_0 = ext_0[o_9 + u32(3)];
    j_6.kind_flags_0 = (bitcast<vec4<u32>>((ext_0[o_9 + u32(4)])));
    return j_6;
}

fn infinity_0() -> f32
{
    return float_bits_0(u32(2139095040));
}

fn failure_indices_0( mat_1 : JointMaterial_0,  b_62 : JointBond_0,  m_5 : Measures_0,  multiplier_0 : f32) -> vec4<f32>
{
    var fc_2 : f32 = mat_1.strength_0.y * multiplier_0;
    var _S533 : f32 = min(mat_1.strength_0.z * multiplier_0 + mat_1.strength_0.w * m_5.normal_compression_0, mat_1.energy_0.x * multiplier_0);
    var idx_0 : vec4<f32>;
    idx_0[i32(0)] = max(fdiv_0(m_5.tension_0, mat_1.strength_0.x * multiplier_0), 0.0f);
    var _S534 : f32;
    if(_S533 > 0.0f)
    {
        _S534 = fdiv_0(m_5.shear_0, _S533);
    }
    else
    {
        _S534 = infinity_0();
    }
    idx_0[i32(1)] = _S534;
    idx_0[i32(2)] = max(fdiv_0(m_5.compression_0, fc_2), 0.0f);
    var _S535 : f32 = b_62.stiff1_0.y;
    if(_S535 > 0.0f)
    {
        _S534 = fdiv_0(m_5.compressive_force_0, _S535);
    }
    else
    {
        _S534 = 0.0f;
    }
    idx_0[i32(3)] = _S534;
    return idx_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_breaks(@builtin(global_invocation_id) id_37 : vec3<u32>)
{
    var i_11 : u32 = id_37.x;
    var _S536 : bool;
    if(i_11 >= (params_0.bond_count_0))
    {
        _S536 = true;
    }
    else
    {
        _S536 = work_0[w_alive_0() + i_11] == u32(0);
    }
    if(_S536)
    {
        return;
    }
    var bond_7 : vec4<u32> = bonds_0[i_11];
    var b_63 : LpBond_0 = lp_bond_0(i_11);
    var _S537 : u32 = bond_7.x;
    var q_lin_6 : vec3<f32>;
    var q_ang_6 : vec3<f32>;
    lp_force_1(b_63, _S537, bond_7.y, u32(0), &(q_lin_6), &(q_ang_6));
    var idx_1 : vec4<f32> = failure_indices_0(lp_material_0(b_63.law_0.ids_3.x), b_63.law_0, stress_measures_0(b_63.law_0, q_lin_6, q_ang_6), b_63.law_0.geom1_0.w);
    var _S538 : f32 = max(max(idx_1.x, idx_1.y), max(idx_1.z, idx_1.w));
    var _S539 : u32 = a_counters_0();
    var _S540 : u32 = atomicMax(&(atom_0[_S539 + u32(11)]), (bitcast<u32>((max(_S538, 0.0f)))));
    if(_S538 >= 1.0f)
    {
        work_0[w_alive_0() + i_11] = u32(0);
        var _S541 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S537)]), u32(1));
        var _S542 : u32 = atomicAdd(&(atom_0[_S539]), u32(1));
        var _S543 : u32 = atomicAdd(&(atom_0[_S539 + u32(1)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_first_reset(@builtin(global_invocation_id) id_38 : vec3<u32>)
{
    var _S544 : u32 = id_38.x;
    if(_S544 < (params_0.shot_count_0))
    {
        atomicStore(&(atom_0[a_first_0(_S544)]), u32(2139095040));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_first(@builtin(global_invocation_id) id_39 : vec3<u32>)
{
    var b_64 : u32 = id_39.x;
    var _S545 : bool;
    if(b_64 >= (params_0.body_count_0))
    {
        _S545 = true;
    }
    else
    {
        _S545 = (params_0.shot_count_0) == u32(0);
    }
    if(_S545)
    {
        return;
    }
    var shot_10 : u32;
    var _S546 : bool = struck_2(b_64, &(shot_10));
    if(!_S546)
    {
        return;
    }
    var _S547 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_64 + u32(6)]));
    var _S548 : u32 = atomicMin(&(atom_0[a_first_0(shot_10)]), _S547);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_begin(@builtin(global_invocation_id) id_40 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_begin_1(id_40.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_extend_even(@builtin(global_invocation_id) id_41 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_extend_1(id_41.x, u32(0));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_extend_odd(@builtin(global_invocation_id) id_42 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_extend_1(id_42.x, u32(1));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_strike(@builtin(global_invocation_id) id_43 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_strike_1(id_43.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_mark(@builtin(global_invocation_id) id_44 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_mark_1(id_44.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_carry(@builtin(global_invocation_id) id_45 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_carry_1(id_45.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_breaks(@builtin(global_invocation_id) id_46 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_breaks_1(id_46.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_begin(@builtin(global_invocation_id) id_47 : vec3<u32>)
{
    il_begin_0(id_47.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_extend_even(@builtin(global_invocation_id) id_48 : vec3<u32>)
{
    il_extend_0(id_48.x, u32(0));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_extend_odd(@builtin(global_invocation_id) id_49 : vec3<u32>)
{
    il_extend_0(id_49.x, u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_strike(@builtin(global_invocation_id) id_50 : vec3<u32>)
{
    il_strike_0(id_50.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_mark(@builtin(global_invocation_id) id_51 : vec3<u32>)
{
    il_mark_0(id_51.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_carry(@builtin(global_invocation_id) id_52 : vec3<u32>)
{
    il_carry_0(id_52.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_breaks(@builtin(global_invocation_id) id_53 : vec3<u32>)
{
    il_breaks_0(id_53.x);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_cs_begin(@builtin(global_invocation_id) id_54 : vec3<u32>)
{
    var _S549 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S549 + u32(14)]), u32(0));
    atomicStore(&(atom_0[_S549 + u32(15)]), u32(0));
    var _S550 : u32 = groups_of_0(cs_rows_0(), u32(64));
    set_args_0(u32(7), _S550);
    return;
}

fn cs_side_1( r_23 : u32,  side_6 : u32,  k_43 : ptr<function, u32>,  s_19 : ptr<function, u32>,  ob_5 : ptr<function, u32>,  co_5 : ptr<function, u32>,  p_19 : ptr<function, vec3<f32>>,  u_9 : ptr<function, vec3<f32>>,  v_15 : ptr<function, f32>,  m_eff_7 : ptr<function, f32>,  t_in_10 : ptr<function, f32>) -> bool
{
    (*t_in_10) = 0.0f;
    (*k_43) = u32(4294967295);
    (*s_19) = u32(4294967295);
    (*ob_5) = u32(4294967295);
    (*co_5) = u32(4294967295);
    var _S551 : vec3<f32> = vec3<f32>(0.0f);
    (*p_19) = _S551;
    (*u_9) = _S551;
    (*v_15) = 0.0f;
    (*m_eff_7) = 0.0f;
    var row_2 : u32 = cs_row_0(r_23);
    var ids_4 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_2])));
    if((ids_4.w) == u32(0))
    {
        return false;
    }
    var _S552 : bool = side_6 == u32(0);
    var cm_1 : u32;
    if(_S552)
    {
        cm_1 = ids_4.x;
    }
    else
    {
        cm_1 = ids_4.y;
    }
    var _S553 : u32;
    if(_S552)
    {
        _S553 = ids_4.y;
    }
    else
    {
        _S553 = ids_4.x;
    }
    (*co_5) = _S553;
    var _S554 : u32 = collider_info_0[u32(2) * cm_1].x;
    (*k_43) = _S554;
    var _S555 : bool;
    if(_S554 == u32(4294967295))
    {
        _S555 = true;
    }
    else
    {
        _S555 = (*k_43) >= (params_0.chunk_count_0);
    }
    if(_S555)
    {
        _S555 = true;
    }
    else
    {
        _S555 = (collider_info_0[u32(2) * (*co_5)].z) != u32(2);
    }
    if(_S555)
    {
        return false;
    }
    (*s_19) = body_of_0((*k_43));
    var _S556 : u32 = (bitcast<u32>((colliders_0[u32(4) * (*co_5)].x)));
    (*ob_5) = _S556;
    if(_S556 == (*s_19))
    {
        _S555 = true;
    }
    else
    {
        _S555 = (*ob_5) >= (params_0.max_bodies_0);
    }
    if(_S555)
    {
        return false;
    }
    var bs_1 : u32 = u32(7) * (*s_19);
    var bo_1 : u32 = u32(7) * (*ob_5);
    var inv_o_1 : f32 = bodies_0[bo_1].w;
    if(!(inv_o_1 > 0.0f))
    {
        return false;
    }
    var n_6 : vec3<f32> = engine_contacts_0[row_2 + u32(1)].xyz;
    var _S557 : vec3<f32>;
    if(_S552)
    {
        _S557 = (vec3<f32>(0) - n_6);
    }
    else
    {
        _S557 = n_6;
    }
    (*u_9) = _S557;
    var _S558 : Quat_0 = quat_of_0(bodies_0[bs_1 + u32(1)]);
    if(_S552)
    {
        _S557 = engine_contacts_0[row_2 + u32(2)].xyz;
    }
    else
    {
        _S557 = engine_contacts_0[row_2 + u32(3)].xyz;
    }
    var rs_1 : vec3<f32> = rotate_0(_S558, _S557);
    var _S559 : Quat_0 = quat_of_0(bodies_0[bo_1 + u32(1)]);
    if(_S552)
    {
        _S557 = engine_contacts_0[row_2 + u32(3)].xyz;
    }
    else
    {
        _S557 = engine_contacts_0[row_2 + u32(2)].xyz;
    }
    var ro_1 : vec3<f32> = rotate_0(_S559, _S557);
    var _S560 : vec3<f32> = bodies_0[bs_1].xyz + rs_1;
    (*p_19) = _S560;
    var gap_1 : f32 = dot(_S560 - (bodies_0[bo_1].xyz + ro_1), (*u_9));
    var _S561 : f32 = dot(bodies_0[bo_1 + u32(3)].xyz + cross(bodies_0[bo_1 + u32(4)].xyz, ro_1) - (bodies_0[bs_1 + u32(3)].xyz + cross(bodies_0[bs_1 + u32(4)].xyz, rs_1)), (*u_9));
    (*v_15) = _S561;
    if(!(_S561 > (params_0.strike_speed_0)))
    {
        return false;
    }
    var w_8 : f32 = bodies_0[bs_1].w + cs_angular_0((*s_19), rs_1, (*u_9)) + inv_o_1 + cs_angular_0((*ob_5), ro_1, (*u_9));
    if(!(w_8 > 0.0f))
    {
        return false;
    }
    (*m_eff_7) = 1.0f / w_8;
    (*t_in_10) = clamp(gap_1 / (*v_15), 0.0f, params_0.dt_0);
    return true;
}

fn w_cs_list_0() -> u32
{
    return w_carry_0() + params_0.chunk_count_0;
}

fn w_cs_bodies_0() -> u32
{
    return w_cs_list_0() + params_0.chunk_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_contacts(@builtin(global_invocation_id) id_55 : vec3<u32>)
{
    var r_24 : u32 = id_55.x;
    if(r_24 >= (cs_rows_0()))
    {
        return;
    }
    var side_7 : u32 = u32(0);
    loop
    {
        if(side_7 < u32(2))
        {
        }
        else
        {
            break;
        }
        var k_44 : u32;
        var s_20 : u32;
        var ob_6 : u32;
        var co_6 : u32;
        var p_20 : vec3<f32>;
        var u_10 : vec3<f32>;
        var v_16 : f32;
        var m_eff_8 : f32;
        var t_in_11 : f32;
        var _S562 : bool = cs_side_1(r_24, side_7, &(k_44), &(s_20), &(ob_6), &(co_6), &(p_20), &(u_10), &(v_16), &(m_eff_8), &(t_in_11));
        if(_S562)
        {
            var _S563 : u32 = atomicMax(&(atom_0[a_cs_0(s_20)]), (bitcast<u32>((m_eff_8 * v_16))));
            var _S564 : u32 = atomicExchange(&(atom_0[a_cs_0(s_20) + u32(8)]), u32(1));
            if(_S564 == u32(0))
            {
                var _S565 : u32 = w_cs_bodies_0();
                var _S566 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(14)]), u32(1));
                work_0[_S565 + _S566] = s_20;
            }
        }
        side_7 = side_7 + u32(1);
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_cs_gate(@builtin(global_invocation_id) id_56 : vec3<u32>)
{
    var struck_3 : u32 = atomicLoad(&(atom_0[a_counters_0() + u32(14)]));
    var any_0 : bool = struck_3 > u32(0);
    var _S567 : u32;
    if(any_0)
    {
        var _S568 : u32 = groups_of_0(cs_rows_0(), u32(64));
        _S567 = _S568;
    }
    else
    {
        _S567 = u32(0);
    }
    set_args_0(u32(8), _S567);
    var _S569 : u32 = groups_of_0(struck_3, u32(64));
    set_args_0(u32(9), _S569);
    set_args_0(u32(6), struck_3);
    if(any_0)
    {
        var _S570 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S567 = _S570;
    }
    else
    {
        _S567 = u32(0);
    }
    set_args_0(u32(10), _S567);
    if(any_0)
    {
        var _S571 : u32 = groups_of_0(params_0.bond_count_0, u32(64));
        _S567 = _S571;
    }
    else
    {
        _S567 = u32(0);
    }
    set_args_0(u32(11), _S567);
    if(any_0)
    {
        var _S572 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S567 = _S572;
    }
    else
    {
        _S567 = u32(0);
    }
    set_args_0(u32(12), _S567);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_a(@builtin(global_invocation_id) id_57 : vec3<u32>)
{
    cs_pick_3(id_57.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_b(@builtin(global_invocation_id) id_58 : vec3<u32>)
{
    cs_pick_2(id_58.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_c(@builtin(global_invocation_id) id_59 : vec3<u32>)
{
    cs_pick_1(id_59.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_d(@builtin(global_invocation_id) id_60 : vec3<u32>)
{
    cs_pick_0(id_60.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_record(@builtin(global_invocation_id) id_61 : vec3<u32>)
{
    var _S573 : u32 = id_61.x;
    var _S574 : u32 = a_counters_0();
    var _S575 : u32 = atomicLoad(&(atom_0[_S574 + u32(14)]));
    if(_S573 >= _S575)
    {
        return;
    }
    var s_21 : u32 = work_0[w_cs_bodies_0() + _S573];
    var _S576 : u32 = a_cs_0(work_0[w_cs_bodies_0() + _S573]);
    var _S577 : u32 = atomicLoad(&(atom_0[_S576 + u32(11)]));
    var key_2 : u32 = _S577 - u32(1);
    var k_45 : u32;
    var s2_1 : u32;
    var ob_7 : u32;
    var co_7 : u32;
    var p_21 : vec3<f32>;
    var u_11 : vec3<f32>;
    var v_17 : f32;
    var m_eff_9 : f32;
    var t_in_12 : f32;
    var _S578 : bool = cs_side_1(key_2 / u32(2), key_2 % u32(2), &(k_45), &(s2_1), &(ob_7), &(co_7), &(p_21), &(u_11), &(v_17), &(m_eff_9), &(t_in_12));
    if(!_S578)
    {
        return;
    }
    var e_o_0 : f32 = (bitcast<f32>((collider_info_0[u32(2) * co_7 + u32(1)].y)));
    var _S579 : f32 = max((bitcast<f32>((collider_info_0[u32(2) * co_7 + u32(1)].z))), 0.00100000004749745f);
    var j_7 : f32 = m_eff_9 * v_17;
    var tau_2 : f32 = hertz_duration_0(m_eff_9, _S579, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / geometry_0[u32(3) * k_45 + u32(1)].w), v_17);
    var _S580 : u32 = lp_body_0(s_21);
    lp_0[_S580 + u32(2)] = vec4<f32>(from_rubble_0(p_21), 3.14159274101257324f * j_7 / (2.0f * tau_2));
    lp_0[_S580 + u32(3)] = vec4<f32>(from_rubble_0(u_11), v_17);
    lp_0[_S580 + u32(4)] = vec4<f32>(m_eff_9, _S579, e_o_0, (bitcast<f32>((ob_7))));
    lp_0[_S580 + u32(5)] = vec4<f32>((bitcast<f32>((k_45))), j_7 / tau_2, 1.0f / bodies_0[u32(7) * ob_7].w, t_in_12);
    atomicStore(&(atom_0[_S576 + u32(1)]), k_45 + u32(1));
    var _S581 : u32 = atomicAdd(&(atom_0[_S574 + u32(12)]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_hertz(@builtin(global_invocation_id) id_62 : vec3<u32>)
{
    var i_12 : u32 = id_62.x;
    var _S582 : bool;
    if(i_12 >= (params_0.bond_count_0))
    {
        _S582 = true;
    }
    else
    {
        _S582 = work_0[w_alive_0() + i_12] == u32(0);
    }
    if(_S582)
    {
        return;
    }
    var bond_8 : vec4<u32> = bonds_0[i_12];
    var b_65 : u32 = body_of_0(bond_8.x);
    var _S583 : bool = cs_struck_0(b_65);
    if(!_S583)
    {
        return;
    }
    var _S584 : u32 = lp_body_0(b_65);
    if(strike_share_breaks_0(bond_8, lp_0[_S584 + u32(2)].xyz, lp_0[_S584 + u32(5)].y))
    {
        work_0[w_alive_0() + i_12] = u32(0);
        var _S585 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_65]), u32(1));
        var _S586 : u32 = a_counters_0();
        var _S587 : u32 = atomicAdd(&(atom_0[_S586]), u32(1));
        var _S588 : u32 = atomicAdd(&(atom_0[_S586 + u32(1)]), u32(1));
        var _S589 : u32 = atomicAdd(&(atom_0[_S586 + u32(13)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_cs_gate2(@builtin(global_invocation_id) id_63 : vec3<u32>)
{
    var _S590 : u32 = atomicLoad(&(atom_0[a_counters_0()]));
    var _S591 : u32;
    if(_S590 > u32(0))
    {
        var _S592 : u32 = groups_of_0(cs_rows_0(), u32(64));
        _S591 = _S592;
    }
    else
    {
        _S591 = u32(0);
    }
    set_args_0(u32(13), _S591);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_save(@builtin(global_invocation_id) id_64 : vec3<u32>)
{
    var r_25 : u32 = id_64.x;
    if(r_25 >= (cs_rows_0()))
    {
        return;
    }
    var row_3 : u32 = cs_row_0(r_25);
    var _S593 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_3])));
    var side_8 : u32 = u32(0);
    loop
    {
        if(side_8 < u32(2))
        {
        }
        else
        {
            break;
        }
        var _S594 : vec4<f32> = vec4<f32>(0.0f);
        var _S595 : bool = side_8 == u32(0);
        var col_1 : u32;
        if(_S595)
        {
            col_1 = _S593.x;
        }
        else
        {
            col_1 = _S593.y;
        }
        var k_46 : u32;
        if((_S593.w) != u32(0))
        {
            k_46 = collider_info_0[u32(2) * col_1].x;
        }
        else
        {
            k_46 = u32(4294967295);
        }
        var _S596 : bool;
        if(k_46 != u32(4294967295))
        {
            _S596 = k_46 < (params_0.chunk_count_0);
        }
        else
        {
            _S596 = false;
        }
        var _S597 : bool;
        if(_S596)
        {
            var _S598 : bool = dirty_0(body_of_0(k_46));
            _S597 = _S598;
        }
        else
        {
            _S597 = false;
        }
        var saved_0 : vec4<f32>;
        if(_S597)
        {
            var at_22 : u32 = u32(7) * body_of_0(k_46);
            var anchor_0 : vec3<f32>;
            if(_S595)
            {
                anchor_0 = engine_contacts_0[row_3 + u32(2)].xyz;
            }
            else
            {
                anchor_0 = engine_contacts_0[row_3 + u32(3)].xyz;
            }
            saved_0 = vec4<f32>(bodies_0[at_22].xyz + rotate_0(quat_of_0(bodies_0[at_22 + u32(1)]), anchor_0), 1.0f);
        }
        else
        {
            saved_0 = _S594;
        }
        cs_anchors_0[u32(2) * r_25 + side_8] = saved_0;
        side_8 = side_8 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_clean(@builtin(global_invocation_id) id_65 : vec3<u32>)
{
    var _S599 : u32 = id_65.x;
    var _S600 : u32 = atomicLoad(&(atom_0[a_counters_0() + u32(14)]));
    if(_S599 >= _S600)
    {
        return;
    }
    var _S601 : u32 = a_cs_0(work_0[w_cs_bodies_0() + _S599]);
    var w_9 : u32 = u32(0);
    loop
    {
        if(w_9 < u32(12))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[_S601 + w_9]), u32(0));
        w_9 = w_9 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_fix(@builtin(global_invocation_id) id_66 : vec3<u32>)
{
    var r_26 : u32 = id_66.x;
    if(r_26 >= (cs_rows_0()))
    {
        return;
    }
    var row_4 : u32 = cs_row_0(r_26);
    var _S602 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_4])));
    var moved_0 : bool = false;
    var side_9 : u32 = u32(0);
    loop
    {
        if(side_9 < u32(2))
        {
        }
        else
        {
            break;
        }
        var saved_1 : vec4<f32> = cs_anchors_0[u32(2) * r_26 + side_9];
        if((cs_anchors_0[u32(2) * r_26 + side_9].w) == 0.0f)
        {
            side_9 = side_9 + u32(1);
            continue;
        }
        var col_2 : u32;
        if(side_9 == u32(0))
        {
            col_2 = _S602.x;
        }
        else
        {
            col_2 = _S602.y;
        }
        var at_23 : u32 = u32(7) * (bitcast<u32>((colliders_0[u32(4) * col_2].x)));
        var anchor_1 : vec3<f32> = rotate_0(conj_0(quat_of_0(bodies_0[at_23 + u32(1)])), saved_1.xyz - bodies_0[at_23].xyz);
        engine_contacts_0[row_4 + u32(2) + side_9].x = anchor_1.x;
        engine_contacts_0[row_4 + u32(2) + side_9].y = anchor_1.y;
        engine_contacts_0[row_4 + u32(2) + side_9].z = anchor_1.z;
        moved_0 = true;
        side_9 = side_9 + u32(1);
    }
    if(moved_0)
    {
        engine_contacts_0[row_4 + u32(4)][i32(3)] = 0.0f;
        engine_contacts_0[row_4 + u32(5)][i32(3)] = 0.0f;
        var _S603 : u32 = row_4 + u32(6);
        engine_contacts_0[_S603] = vec4<f32>(1.0f, 1.0f, 0.0f, engine_contacts_0[_S603].w);
        engine_contacts_0[row_4 + u32(7)][i32(1)] = 0.0f;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_count(@builtin(global_invocation_id) id_67 : vec3<u32>)
{
    var k_47 : u32 = id_67.x;
    if(k_47 >= (params_0.chunk_count_0))
    {
        return;
    }
    var s_22 : u32 = body_of_0(k_47);
    var _S604 : bool = cs_struck_0(s_22);
    if(!_S604)
    {
        return;
    }
    var _S605 : u32 = a_cs_0(s_22);
    var _S606 : u32 = atomicMax(&(atom_0[_S605 + u32(3)]), ~k_47);
    if(((info_0[k_47] & (u32(1)))) != u32(0))
    {
        atomicStore(&(atom_0[_S605 + u32(4)]), u32(1));
    }
    var _S607 : u32 = atomicAdd(&(atom_0[_S605 + u32(5)]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_base(@builtin(global_invocation_id) id_68 : vec3<u32>)
{
    var _S608 : u32 = id_68.x;
    var _S609 : u32 = a_counters_0();
    var _S610 : u32 = atomicLoad(&(atom_0[_S609 + u32(14)]));
    if(_S608 >= _S610)
    {
        return;
    }
    var _S611 : u32 = a_cs_0(work_0[w_cs_bodies_0() + _S608]);
    var _S612 : u32 = atomicLoad(&(atom_0[_S611 + u32(5)]));
    var _S613 : u32 = atomicAdd(&(atom_0[_S609 + u32(15)]), _S612);
    atomicStore(&(atom_0[_S611 + u32(6)]), _S613);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_collect(@builtin(global_invocation_id) id_69 : vec3<u32>)
{
    var k_48 : u32 = id_69.x;
    if(k_48 >= (params_0.chunk_count_0))
    {
        return;
    }
    var s_23 : u32 = body_of_0(k_48);
    var _S614 : bool = cs_struck_0(s_23);
    if(!_S614)
    {
        return;
    }
    var _S615 : u32 = w_cs_list_0();
    var _S616 : u32 = a_cs_0(s_23);
    var _S617 : u32 = atomicLoad(&(atom_0[_S616 + u32(6)]));
    var _S618 : u32 = _S615 + _S617;
    var _S619 : u32 = atomicAdd(&(atom_0[_S616 + u32(7)]), u32(1));
    work_0[_S618 + _S619] = k_48;
    return;
}

fn cs_load_0( k_49 : u32,  s_24 : u32)
{
    var _S620 : u32 = a_cs_0(s_24);
    var _S621 : u32 = atomicLoad(&(atom_0[_S620 + u32(4)]));
    var supported_1 : bool = _S621 != u32(0);
    var fixed_2 : bool;
    if(((info_0[k_49] & (u32(1)))) != u32(0))
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            var _S622 : u32 = atomicLoad(&(atom_0[_S620 + u32(3)]));
            fixed_2 = (~_S622) == k_49;
        }
        else
        {
            fixed_2 = false;
        }
    }
    var _S623 : u32 = lp_body_0(s_24);
    var _S624 : u32 = u32(7) * s_24;
    var q_33 : Quat_0;
    var origin_14 : vec3<f32>;
    structure_pose_3(s_24, _S624, &(q_33), &(origin_14));
    var qi_1 : Quat_0 = conj_0(q_33);
    var force_4 : vec3<f32> = rotate_0(qi_1, lp_0[_S623 + u32(3)].xyz * vec3<f32>(lp_0[_S623 + u32(2)].w));
    var contact_20 : vec3<f32> = rotate_0(qi_1, lp_0[_S623 + u32(2)].xyz - origin_14);
    var _S625 : u32 = u32(4) * k_49;
    var c0_5 : vec4<f32> = chunks_0[_S625];
    var _S626 : vec3<f32> = vec3<f32>(0.0f);
    var _S627 : u32 = atomicLoad(&(atom_0[_S620 + u32(1)]));
    var f_13 : vec3<f32>;
    var t_11 : vec3<f32>;
    if(_S627 == (k_49 + u32(1)))
    {
        var _S628 : vec3<f32> = cross(contact_20 - c0_5.xyz, force_4);
        f_13 = force_4;
        t_11 = _S628;
    }
    else
    {
        f_13 = _S626;
        t_11 = _S626;
    }
    var inv_mass_7 : f32 = bodies_0[_S624].w;
    var _S629 : bool;
    if(!supported_1)
    {
        _S629 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S629 = false;
    }
    if(_S629)
    {
        var _S630 : u32 = u32(2) * s_24;
        var com_6 : vec3<f32> = frames_0[_S630].xyz;
        var p_22 : Quat_0 = quat_of_0(frames_0[_S630 + u32(1)]);
        var alpha_1 : vec3<f32> = rotate_0(p_22, bodies_0[_S624 + u32(2)].xyz * rotate_0(conj_0(p_22), cross(contact_20 - com_6, force_4)));
        var c1_4 : vec4<f32> = chunks_0[_S625 + u32(1)];
        var c2_5 : vec4<f32> = chunks_0[_S625 + u32(2)];
        var _S631 : f32 = alpha_1.x;
        var _S632 : f32 = c1_4.w;
        var _S633 : f32 = alpha_1.y;
        var _S634 : f32 = c2_5.x;
        var _S635 : f32 = alpha_1.z;
        var _S636 : f32 = c2_5.y;
        var t_12 : vec3<f32> = t_11 - vec3<f32>(c1_4.x * _S631 + _S632 * _S633 + _S634 * _S635, _S632 * _S631 + c1_4.y * _S633 + _S636 * _S635, _S634 * _S631 + _S636 * _S633 + c1_4.z * _S635);
        f_13 = f_13 - vec3<f32>(c0_5.w) * (force_4 * vec3<f32>(inv_mass_7) + cross(alpha_1, c0_5.xyz - com_6));
        t_11 = t_12;
    }
    var _S637 : u32 = st_f_0(k_49, u32(14));
    var _S638 : f32;
    if(fixed_2)
    {
        _S638 = 1.0f;
    }
    else
    {
        _S638 = 0.0f;
    }
    lp_0[_S637] = vec4<f32>(f_13, _S638);
    lp_0[_S637 + u32(1)] = vec4<f32>(t_11, 0.0f);
    var c_26 : u32 = u32(0);
    loop
    {
        if(c_26 < u32(2))
        {
        }
        else
        {
            break;
        }
        var _S639 : u32 = st_x_0(k_49, u32(14), c_26);
        var _S640 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S639] = _S640;
        lp_0[_S639 + u32(1)] = _S640;
        c_26 = c_26 + u32(1);
    }
    var _S641 : u32 = st_d_0(k_49, u32(14));
    var _S642 : vec4<f32> = vec4<f32>(0.0f);
    lp_0[_S641] = _S642;
    lp_0[_S641 + u32(1)] = _S642;
    return;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn cs_break_0( i_13 : u32,  s_25 : u32)
{
    var bond_9 : vec4<u32> = bonds_0[i_13];
    var b_66 : LpBond_0 = lp_bond_0(i_13);
    var _S643 : u32 = bond_9.x;
    var _S644 : u32 = bond_9.y;
    var q_lin_7 : vec3<f32>;
    var q_ang_7 : vec3<f32>;
    st_force_0(b_66, _S643, _S644, u32(14), u32(0), &(q_lin_7), &(q_ang_7));
    if((params_0.load_model_0) == u32(1))
    {
        var b_lin_0 : vec3<f32>;
        var b_ang_0 : vec3<f32>;
        lp_force_0(b_66, _S643, _S644, u32(0), &(b_lin_0), &(b_ang_0));
        q_lin_7 = q_lin_7 + b_lin_0;
        q_ang_7 = q_ang_7 + b_ang_0;
    }
    var idx_2 : vec4<f32> = failure_indices_0(lp_material_0(b_66.law_0.ids_3.x), b_66.law_0, stress_measures_0(b_66.law_0, q_lin_7, q_ang_7), b_66.law_0.geom1_0.w);
    if((max(max(idx_2.x, idx_2.y), max(idx_2.z, idx_2.w))) >= 1.0f)
    {
        work_0[w_alive_0() + i_13] = u32(0);
        var _S645 : u32 = atomicOr(&(atom_0[a_dirty_0() + s_25]), u32(1));
        var _S646 : u32 = a_counters_0();
        var _S647 : u32 = atomicAdd(&(atom_0[_S646]), u32(1));
        var _S648 : u32 = atomicAdd(&(atom_0[_S646 + u32(1)]), u32(1));
        var _S649 : u32 = atomicAdd(&(atom_0[_S646 + u32(13)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn fr_cs_solve(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var _S650 : bool;
    var j3_0 : u32;
    var s_26 : u32 = work_0[w_cs_bodies_0() + group_1.x];
    var t_13 : u32 = thread_2.x;
    var _S651 : u32 = a_cs_0(work_0[w_cs_bodies_0() + group_1.x]);
    var _S652 : u32 = atomicLoad(&(atom_0[_S651 + u32(6)]));
    var _S653 : u32 = atomicLoad(&(atom_0[_S651 + u32(5)]));
    var j_8 : u32 = t_13;
    loop
    {
        if(j_8 < _S653)
        {
        }
        else
        {
            break;
        }
        cs_load_0(work_0[w_cs_list_0() + _S652 + j_8], s_26);
        j_8 = j_8 + u32(256);
    }
    device_group_barrier_0();
    var j2_0 : u32 = t_13;
    loop
    {
        if(j2_0 < _S653)
        {
        }
        else
        {
            break;
        }
        var k_50 : u32 = work_0[w_cs_list_0() + _S652 + j2_0];
        var _S654 : u32 = st_f_0(work_0[w_cs_list_0() + _S652 + j2_0], u32(14));
        if((lp_0[_S654].w) == 0.0f)
        {
            var _S655 : bool = lp_factor_0(k_50);
            _S650 = !_S655;
        }
        else
        {
            _S650 = false;
        }
        if(_S650)
        {
            lp_0[_S654][i32(3)] = 1.0f;
            var _S656 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(32));
        }
        j2_0 = j2_0 + u32(256);
    }
    device_group_barrier_0();
    var sweep_0 : u32 = u32(0);
    loop
    {
        if(sweep_0 < (params_0.cs_sweeps_0))
        {
        }
        else
        {
            break;
        }
        j3_0 = t_13;
        loop
        {
            if(j3_0 < _S653)
            {
            }
            else
            {
                break;
            }
            st_sweep_0(work_0[w_cs_list_0() + _S652 + j3_0], u32(14), sweep_0 % u32(2), params_0.cs_c1_0, params_0.cs_c2_0);
            j3_0 = j3_0 + u32(256);
        }
        device_group_barrier_0();
        sweep_0 = sweep_0 + u32(1);
    }
    j3_0 = t_13;
    loop
    {
        if(j3_0 < _S653)
        {
        }
        else
        {
            break;
        }
        var k_51 : u32 = work_0[w_cs_list_0() + _S652 + j3_0];
        var range_3 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * work_0[w_cs_list_0() + _S652 + j3_0]];
        var _S657 : u32 = (bitcast<u32>((range_3.x)));
        var _S658 : u32 = (bitcast<u32>((range_3.y)));
        var e_5 : u32 = u32(0);
        loop
        {
            if(e_5 < _S658)
            {
            }
            else
            {
                break;
            }
            var i_14 : u32 = adj_bond_0(_S657 + e_5);
            if((bonds_0[i_14].x) == k_51)
            {
                _S650 = work_0[w_alive_0() + i_14] != u32(0);
            }
            else
            {
                _S650 = false;
            }
            if(_S650)
            {
                cs_break_0(i_14, s_26);
            }
            e_5 = e_5 + u32(1);
        }
        j3_0 = j3_0 + u32(256);
    }
    return;
}

