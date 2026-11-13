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
    @align(4) chunk_used_0 : u32,
    @align(16) bond_used_0 : u32,
    @align(4) ev_broken_cap_0 : u32,
    @align(8) ev_moved_cap_0 : u32,
    @align(4) ev_frames_cap_0 : u32,
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
    @align(16) lp_scale_0 : f32,
    @align(4) cs_sweeps_0 : u32,
    @align(8) cs_row_cap_0 : u32,
    @align(4) cs_c1_0 : f32,
    @align(16) cs_c2_0 : f32,
    @align(4) cs_row_offset_0 : u32,
    @align(8) ext_sides_0 : u32,
    @align(4) strike_radius_0 : f32,
    @align(16) gi_mode_0 : u32,
    @align(4) gi_scale_0 : f32,
    @align(8) gi_sweeps_0 : u32,
    @align(4) gi_pad_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : FractureParams_std140_0;
@binding(5) @group(0) var<storage, read_write> atom_0 : array<atomic<u32>>;

@binding(4) @group(0) var<storage, read_write> work_0 : array<u32>;

@binding(6) @group(0) var<storage, read_write> frames_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read_write> bodies_0 : array<vec4<f32>>;

@binding(1) @group(0) var<storage, read> chunks_0 : array<vec4<f32>>;

@binding(11) @group(0) var<storage, read> sort_0 : array<u32>;

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
    @align(4) pad3_0 : f32,
};

@binding(16) @group(0) var<uniform> view_0 : ViewParams_std140_0;
@binding(18) @group(0) var<storage, read_write> objects_0 : array<vec4<f32>>;

@binding(20) @group(0) var<storage, read> shot_bodies_0 : array<u32>;

@binding(19) @group(0) var<storage, read_write> lines_0 : array<f32>;

@binding(21) @group(0) var<storage, read> geometry_0 : array<vec4<f32>>;

@binding(22) @group(0) var<storage, read> shots_0 : array<vec4<f32>>;

@binding(24) @group(0) var<storage, read_write> lp_0 : array<vec4<f32>>;

@binding(23) @group(0) var<storage, read> ext_0 : array<vec4<f32>>;

@binding(25) @group(0) var<storage, read_write> engine_contacts_0 : array<vec4<f32>>;

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
    @align(8) particle_ccd_0 : u32,
    @align(4) r_virial_0 : u32,
    @align(16) bond_virial_0 : u32,
    @align(4) pad1_1 : f32,
};

@binding(28) @group(0) var<uniform> crush_0 : CrushParams_std140_0;
@binding(27) @group(0) var<storage, read_write> crush_rw_0 : array<vec4<f32>>;

@binding(26) @group(0) var<storage, read> crush_tab_0 : array<vec4<f32>>;

fn lp_body_0( b_0 : u32) -> u32
{
    return u32(22) * params_0.chunk_count_0 + u32(7) * b_0;
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
    return a_counters_0() + u32(24);
}

fn il_t_in_1( b_2 : u32) -> f32
{
    var _S1 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_2 + u32(6)]));
    return (bitcast<f32>((_S1)));
}

fn w_body_0() -> u32
{
    return u32(0);
}

fn body_of_0( c_0 : u32) -> u32
{
    return work_0[w_body_0() + c_0];
}

fn a_first_0( shot_0 : u32) -> u32
{
    return a_strike_0() + u32(29) * params_0.body_count_0 + shot_0;
}

fn a_il_cursor_0() -> u32
{
    return a_first_0(u32(0)) + u32(64);
}

fn a_cs_0( b_3 : u32) -> u32
{
    return a_il_cursor_0() + u32(1) + u32(12) * b_3;
}

fn cs_struck_0( b_4 : u32) -> bool
{
    var _S2 : bool;
    if(b_4 < (params_0.body_count_0))
    {
        var _S3 : u32 = atomicLoad(&(atom_0[a_cs_0(b_4) + u32(1)]));
        _S2 = _S3 != u32(0);
    }
    else
    {
        _S2 = false;
    }
    return _S2;
}

fn il_struck_0( b_5 : u32) -> bool
{
    var _S4 : bool = cs_struck_0(b_5);
    return _S4;
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

fn il_striker_0( b_7 : u32) -> IlStriker_0
{
    var _S6 : u32 = lp_body_0(b_7);
    var s_0 : IlStriker_0;
    s_0.body_0 = (bitcast<u32>((lp_0[_S6 + u32(4)].w)));
    var _S7 : f32 = lp_0[_S6 + u32(4)].y;
    s_0.bound_0 = 1.00999999046325684f * _S7 + 0.00100000004749745f;
    s_0.radius_0 = _S7;
    s_0.mass_0 = lp_0[_S6 + u32(5)].z;
    s_0.m_eff_0 = lp_0[_S6 + u32(4)].x;
    s_0.modulus_0 = lp_0[_S6 + u32(4)].z;
    s_0.launch_0 = (bitcast<u32>((lp_0[_S6 + u32(6)].y)));
    return s_0;
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
    var r_0 : Quat_0;
    r_0.x_0 = q_0.x;
    r_0.y_0 = q_0.y;
    r_0.z_0 = q_0.z;
    r_0.w_0 = q_0.w;
    return r_0;
}

fn conj_0( q_1 : Quat_0) -> Quat_0
{
    var c_1 : Quat_0;
    c_1.w_0 = q_1.w_0;
    c_1.x_0 = - q_1.x_0;
    c_1.y_0 = - q_1.y_0;
    c_1.z_0 = - q_1.z_0;
    return c_1;
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
    var _S8 : f32 = q_2.w_0;
    var _S9 : f32 = q_2.x_0;
    var _S10 : f32 = q_2.y_0;
    var _S11 : f32 = q_2.z_0;
    var n_1 : f32 = sqrt(_S8 * _S8 + _S9 * _S9 + _S10 * _S10 + _S11 * _S11);
    var r_2 : Quat_0;
    r_2.w_0 = q_2.w_0 / n_1;
    r_2.x_0 = q_2.x_0 / n_1;
    r_2.y_0 = q_2.y_0 / n_1;
    r_2.z_0 = q_2.z_0 / n_1;
    return r_2;
}

fn rot_from_rubble_0( q_3 : Quat_0) -> Quat_0
{
    var axes_0 : Quat_0;
    axes_0.w_0 = 0.70710676908493042f;
    axes_0.x_0 = 0.70710676908493042f;
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
    var qv_0 : vec3<f32> = vec3<f32>(q_4.x_0, q_4.y_0, q_4.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_1) * vec3<f32>(2.0f);
    return v_1 + t_0 * vec3<f32>(q_4.w_0) + cross(qv_0, t_0);
}

fn structure_pose_0( slot_0 : u32,  b_8 : u32,  q_5 : ptr<function, Quat_0>,  origin_0 : ptr<function, vec3<f32>>)
{
    var _S12 : u32 = u32(2) * slot_0;
    var com_0 : vec3<f32> = frames_0[_S12].xyz;
    var _S13 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_8 + u32(1)])), conj_0(quat_of_0(frames_0[_S12 + u32(1)]))));
    (*q_5) = _S13;
    (*origin_0) = from_rubble_0(bodies_0[b_8].xyz) - rotate_0(_S13, com_0);
    return;
}

fn chunk_box_world_0( k_0 : u32,  q_6 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_1 : u32 = body_of_0(k_0);
    var origin_1 : vec3<f32>;
    structure_pose_0(slot_1, u32(7) * slot_1, &((*q_6)), &(origin_1));
    return origin_1 + rotate_0((*q_6), geometry_0[u32(3) * k_0].xyz);
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

fn il_interval_0( k_1 : u32,  p_0 : vec3<f32>,  d_1 : vec3<f32>,  pad_0 : f32) -> vec2<f32>
{
    var q_7 : Quat_0;
    var _S14 : vec3<f32> = chunk_box_world_0(k_1, &(q_7));
    var _S15 : u32 = u32(3) * k_1;
    var _S16 : Quat_0 = normalized_0(quat_mul_0(q_7, quat_of_0(geometry_0[_S15 + u32(2)])));
    var _S17 : vec3<f32> = geometry_0[_S15 + u32(1)].xyz;
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
        var axis_0 : vec3<f32> = rotate_0(_S16, e_0);
        var o_1 : f32 = dot(p_0 - _S14, axis_0);
        var dv_0 : f32 = dot(d_1, axis_0);
        var ext_1 : f32 = _S17[ax_0] + pad_0;
        if((abs(dv_0)) < 9.99999971718068537e-10f)
        {
            if((abs(o_1)) > ext_1)
            {
                return vec2<f32>(1.0f, 0.0f);
            }
            ax_0 = ax_0 + u32(1);
            continue;
        }
        var ta_0 : f32 = (- ext_1 - o_1) / dv_0;
        var tb_0 : f32 = (ext_1 - o_1) / dv_0;
        var _S18 : f32 = min(t1_0, max(ta_0, tb_0));
        t0_0 = max(t0_0, min(ta_0, tb_0));
        t1_0 = _S18;
        ax_0 = ax_0 + u32(1);
    }
    return vec2<f32>(t0_0, t1_0);
}

fn a_il_0( b_9 : u32) -> u32
{
    return a_strike_0() + u32(23) * params_0.body_count_0 + u32(6) * b_9;
}

fn il_tube_0( k_2 : u32,  b_10 : ptr<function, u32>,  t_1 : ptr<function, vec2<f32>>) -> bool
{
    (*b_10) = u32(0);
    (*t_1) = vec2<f32>(1.0f, 0.0f);
    if(k_2 >= (params_0.chunk_count_0))
    {
        return false;
    }
    var _S19 : u32 = body_of_0(k_2);
    (*b_10) = _S19;
    var _S20 : bool = il_struck_0(_S19);
    if(!_S20)
    {
        return false;
    }
    var contact_1 : vec3<f32>;
    var d_2 : vec3<f32>;
    var speed_1 : f32;
    il_path_0((*b_10), &(contact_1), &(d_2), &(speed_1));
    var _S21 : vec2<f32> = il_interval_0(k_2, contact_1, d_2, il_striker_0((*b_10)).radius_0);
    (*t_1) = _S21;
    var _S22 : bool;
    if((_S21.x) <= (_S21.y))
    {
        var _S23 : f32 = (*t_1).y;
        var _S24 : u32 = atomicLoad(&(atom_0[a_il_0((*b_10))]));
        _S22 = _S23 > (bitcast<f32>((_S24)));
    }
    else
    {
        _S22 = false;
    }
    return _S22;
}

fn strike_at_0( b_11 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * params_0.chunk_count_0 + u32(3) * b_11;
}

fn struck_0( b_12 : u32,  shot_1 : ptr<function, u32>) -> bool
{
    var k_3 : u32 = u32(frames_0[strike_at_0(b_12)].w);
    (*shot_1) = k_3 - u32(1);
    return k_3 != u32(0);
}

fn il_struck_1( b_13 : u32) -> bool
{
    var shot_2 : u32;
    var _S25 : bool;
    if((params_0.shot_count_0) > u32(0))
    {
        _S25 = b_13 < (params_0.body_count_0);
    }
    else
    {
        _S25 = false;
    }
    if(_S25)
    {
        _S25 = struck_0(b_13, &(shot_2));
    }
    else
    {
        _S25 = false;
    }
    return _S25;
}

fn il_path_1( b_14 : u32,  contact_2 : ptr<function, vec3<f32>>,  d_3 : ptr<function, vec3<f32>>,  speed_2 : ptr<function, f32>)
{
    var st_0 : u32 = strike_at_0(b_14);
    var rel_0 : vec3<f32> = frames_0[st_0 + u32(1)].xyz;
    var _S26 : f32 = length(rel_0);
    (*speed_2) = _S26;
    (*d_3) = rel_0 / vec3<f32>(max(_S26, 9.999999960041972e-13f));
    (*contact_2) = frames_0[st_0 + u32(2)].xyz;
    return;
}

fn il_striker_1( b_15 : u32) -> IlStriker_0
{
    var _S27 : u32 = u32(2) * (u32(frames_0[strike_at_0(b_15)].w) - u32(1));
    var info_1 : vec4<f32> = shots_0[_S27];
    var s_1 : IlStriker_0;
    s_1.body_0 = (bitcast<u32>((info_1.x)));
    s_1.bound_0 = info_1.y;
    s_1.radius_0 = info_1.z;
    s_1.mass_0 = info_1.w;
    var inv_mass_0 : f32 = bodies_0[u32(7) * b_15].w;
    var _S28 : f32;
    if(inv_mass_0 == 0.0f)
    {
        _S28 = s_1.mass_0;
    }
    else
    {
        _S28 = s_1.mass_0 / (1.0f + s_1.mass_0 * inv_mass_0);
    }
    s_1.m_eff_0 = _S28;
    var _S29 : u32 = _S27 + u32(1);
    s_1.modulus_0 = shots_0[_S29].x;
    s_1.launch_0 = (bitcast<u32>((shots_0[_S29].y)));
    return s_1;
}

fn il_tube_1( k_4 : u32,  b_16 : ptr<function, u32>,  t_2 : ptr<function, vec2<f32>>) -> bool
{
    (*b_16) = u32(0);
    (*t_2) = vec2<f32>(1.0f, 0.0f);
    if(k_4 >= (params_0.chunk_count_0))
    {
        return false;
    }
    var _S30 : u32 = body_of_0(k_4);
    (*b_16) = _S30;
    if(!il_struck_1(_S30))
    {
        return false;
    }
    var contact_3 : vec3<f32>;
    var d_4 : vec3<f32>;
    var speed_3 : f32;
    il_path_1((*b_16), &(contact_3), &(d_4), &(speed_3));
    var _S31 : vec2<f32> = il_interval_0(k_4, contact_3, d_4, il_striker_1((*b_16)).radius_0);
    (*t_2) = _S31;
    var _S32 : bool;
    if((_S31.x) <= (_S31.y))
    {
        var _S33 : f32 = (*t_2).y;
        var _S34 : u32 = atomicLoad(&(atom_0[a_il_0((*b_16))]));
        _S32 = _S33 > (bitcast<f32>((_S34)));
    }
    else
    {
        _S32 = false;
    }
    return _S32;
}

fn il_entry_0( b_17 : u32) -> u32
{
    var _S35 : u32 = atomicLoad(&(atom_0[a_cs_0(b_17) + u32(1)]));
    return _S35 - u32(1);
}

fn il_path_2( b_18 : u32,  contact_4 : ptr<function, vec3<f32>>,  d_5 : ptr<function, vec3<f32>>,  speed_4 : ptr<function, f32>)
{
    var _S36 : u32 = lp_body_0(b_18);
    (*contact_4) = lp_0[_S36 + u32(2)].xyz;
    var r1_1 : vec4<f32> = lp_0[_S36 + u32(3)];
    (*d_5) = lp_0[_S36 + u32(3)].xyz;
    (*speed_4) = r1_1.w;
    return;
}

fn il_entry_1( b_19 : u32) -> u32
{
    var _S37 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_19 + u32(8)]));
    return _S37;
}

fn il_path_3( b_20 : u32,  contact_5 : ptr<function, vec3<f32>>,  d_6 : ptr<function, vec3<f32>>,  speed_5 : ptr<function, f32>)
{
    var st_1 : u32 = strike_at_0(b_20);
    var rel_1 : vec3<f32> = frames_0[st_1 + u32(1)].xyz;
    var _S38 : f32 = length(rel_1);
    (*speed_5) = _S38;
    (*d_6) = rel_1 / vec3<f32>(max(_S38, 9.999999960041972e-13f));
    (*contact_5) = frames_0[st_1 + u32(2)].xyz;
    return;
}

fn isinf_0( x_1 : f32) -> bool
{
    var _S39 : u32 = (bitcast<u32>((x_1)));
    var _S40 : u32 = (_S39 & (u32(8388607)));
    var _S41 : bool;
    if(((((_S39 >> (u32(23)))) & (u32(255)))) == u32(255))
    {
        _S41 = _S40 == u32(0);
    }
    else
    {
        _S41 = false;
    }
    return _S41;
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

fn w_marks_0() -> u32
{
    return w_pulses_0() + u32(4) * params_0.chunk_count_0;
}

fn w_carry_0() -> u32
{
    return w_marks_0() + params_0.chunk_count_0;
}

fn w_il_tube_0() -> u32
{
    return w_carry_0() + u32(2) * params_0.chunk_count_0 + params_0.body_count_0;
}

fn w_host_breaks_0() -> u32
{
    return w_il_tube_0() + u32(2) * params_0.chunk_count_0;
}

fn w_cs_rows_0() -> u32
{
    return w_host_breaks_0() + u32(1) + params_0.bond_count_0;
}

fn cs_rows_0() -> u32
{
    return min(work_0[w_cs_rows_0()], params_0.cs_row_cap_0);
}

fn cs_row_0( r_3 : u32) -> u32
{
    return u32(8) * (params_0.cs_row_offset_0 + r_3);
}

fn cs_angular_0( b_21 : u32,  r_4 : vec3<f32>,  u_0 : vec3<f32>) -> f32
{
    var at_0 : u32 = u32(7) * b_21;
    var l_0 : vec3<f32> = rotate_0(conj_0(quat_of_0(bodies_0[at_0 + u32(1)])), cross(r_4, u_0));
    return dot(l_0 * bodies_0[at_0 + u32(2)].xyz, l_0);
}

fn cs_side_0( r_5 : u32,  side_0 : u32,  k_5 : ptr<function, u32>,  s_2 : ptr<function, u32>,  ob_0 : ptr<function, u32>,  co_0 : ptr<function, u32>,  p_1 : ptr<function, vec3<f32>>,  u_1 : ptr<function, vec3<f32>>,  v_2 : ptr<function, f32>,  m_eff_1 : ptr<function, f32>,  t_in_0 : ptr<function, f32>) -> bool
{
    (*t_in_0) = 0.0f;
    (*k_5) = u32(4294967295);
    (*s_2) = u32(4294967295);
    (*ob_0) = u32(4294967295);
    (*co_0) = u32(4294967295);
    var _S42 : vec3<f32> = vec3<f32>(0.0f);
    (*p_1) = _S42;
    (*u_1) = _S42;
    (*v_2) = 0.0f;
    (*m_eff_1) = 0.0f;
    var row_0 : u32 = cs_row_0(r_5);
    var ids_1 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_0])));
    if((ids_1.w) == u32(0))
    {
        return false;
    }
    var _S43 : bool = side_0 == u32(0);
    var cm_0 : u32;
    if(_S43)
    {
        cm_0 = ids_1.x;
    }
    else
    {
        cm_0 = ids_1.y;
    }
    var _S44 : u32;
    if(_S43)
    {
        _S44 = ids_1.y;
    }
    else
    {
        _S44 = ids_1.x;
    }
    (*co_0) = _S44;
    var _S45 : u32 = collider_info_0[u32(2) * cm_0].x;
    (*k_5) = _S45;
    var _S46 : bool;
    if(_S45 == u32(4294967295))
    {
        _S46 = true;
    }
    else
    {
        _S46 = (*k_5) >= (params_0.chunk_count_0);
    }
    if(_S46)
    {
        _S46 = true;
    }
    else
    {
        _S46 = (collider_info_0[u32(2) * (*co_0)].z) != u32(2);
    }
    if(_S46)
    {
        return false;
    }
    (*s_2) = body_of_0((*k_5));
    var _S47 : u32 = (bitcast<u32>((colliders_0[u32(4) * (*co_0)].x)));
    (*ob_0) = _S47;
    if(_S47 == (*s_2))
    {
        _S46 = true;
    }
    else
    {
        _S46 = (*ob_0) >= (params_0.max_bodies_0);
    }
    if(_S46)
    {
        return false;
    }
    var launch_1 : u32 = collider_info_0[u32(2) * (*co_0) + u32(1)].w;
    if(launch_1 != u32(0))
    {
        _S46 = work_0[w_marks_0() + (*k_5)] == launch_1;
    }
    else
    {
        _S46 = false;
    }
    if(_S46)
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
    var n_2 : vec3<f32> = engine_contacts_0[row_0 + u32(1)].xyz;
    var _S48 : vec3<f32>;
    if(_S43)
    {
        _S48 = (vec3<f32>(0) - n_2);
    }
    else
    {
        _S48 = n_2;
    }
    (*u_1) = _S48;
    var _S49 : Quat_0 = quat_of_0(bodies_0[bs_0 + u32(1)]);
    if(_S43)
    {
        _S48 = engine_contacts_0[row_0 + u32(2)].xyz;
    }
    else
    {
        _S48 = engine_contacts_0[row_0 + u32(3)].xyz;
    }
    var rs_0 : vec3<f32> = rotate_0(_S49, _S48);
    var _S50 : Quat_0 = quat_of_0(bodies_0[bo_0 + u32(1)]);
    if(_S43)
    {
        _S48 = engine_contacts_0[row_0 + u32(3)].xyz;
    }
    else
    {
        _S48 = engine_contacts_0[row_0 + u32(2)].xyz;
    }
    var ro_0 : vec3<f32> = rotate_0(_S50, _S48);
    var _S51 : vec3<f32> = bodies_0[bs_0].xyz + rs_0;
    (*p_1) = _S51;
    var gap_0 : f32 = dot(_S51 - (bodies_0[bo_0].xyz + ro_0), (*u_1));
    var _S52 : f32 = dot(bodies_0[bo_0 + u32(3)].xyz + cross(bodies_0[bo_0 + u32(4)].xyz, ro_0) - (bodies_0[bs_0 + u32(3)].xyz + cross(bodies_0[bs_0 + u32(4)].xyz, rs_0)), (*u_1));
    (*v_2) = _S52;
    if(!(_S52 > (params_0.strike_speed_0)))
    {
        return false;
    }
    var w_1 : f32 = bodies_0[bs_0].w + cs_angular_0((*s_2), rs_0, (*u_1)) + inv_o_0 + cs_angular_0((*ob_0), ro_0, (*u_1));
    if(!(w_1 > 0.0f))
    {
        return false;
    }
    (*m_eff_1) = 1.0f / w_1;
    (*t_in_0) = clamp(gap_0 / (*v_2), 0.0f, params_0.dt_0);
    return true;
}

fn cs_identity_0( r_6 : u32,  side_1 : u32) -> vec3<u32>
{
    var ids_2 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[cs_row_0(r_6)])));
    var _S53 : vec3<u32>;
    if(side_1 == u32(0))
    {
        _S53 = vec3<u32>(ids_2.x, ids_2.y, u32(2) * ids_2.z);
    }
    else
    {
        _S53 = vec3<u32>(ids_2.y, ids_2.x, u32(2) * ids_2.z + u32(1));
    }
    return _S53;
}

fn cs_pick_0( r_7 : u32)
{
    if(r_7 >= (cs_rows_0()))
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
        var k_6 : u32;
        var s_3 : u32;
        var ob_1 : u32;
        var co_1 : u32;
        var p_2 : vec3<f32>;
        var u_2 : vec3<f32>;
        var v_3 : f32;
        var m_eff_2 : f32;
        var t_in_1 : f32;
        var _S54 : bool = cs_side_0(r_7, side_2, &(k_6), &(s_3), &(ob_1), &(co_1), &(p_2), &(u_2), &(v_3), &(m_eff_2), &(t_in_1));
        var _S55 : bool;
        if(!_S54)
        {
            _S55 = true;
        }
        else
        {
            var _S56 : u32 = (bitcast<u32>((m_eff_2 * v_3)));
            var _S57 : u32 = atomicLoad(&(atom_0[a_cs_0(s_3)]));
            _S55 = _S56 != _S57;
        }
        if(_S55)
        {
            side_2 = side_2 + u32(1);
            continue;
        }
        var id_0 : vec3<u32> = cs_identity_0(r_7, side_2);
        var at_1 : u32 = a_cs_0(s_3);
        var _S58 : u32 = atomicLoad(&(atom_0[at_1 + u32(2)]));
        if((~_S58) != (id_0.x))
        {
            side_2 = side_2 + u32(1);
            continue;
        }
        var _S59 : u32 = atomicLoad(&(atom_0[at_1 + u32(9)]));
        if((~_S59) != (id_0.y))
        {
            side_2 = side_2 + u32(1);
            continue;
        }
        var _S60 : u32 = atomicLoad(&(atom_0[at_1 + u32(10)]));
        if((~_S60) != (id_0.z))
        {
            side_2 = side_2 + u32(1);
            continue;
        }
        atomicStore(&(atom_0[at_1 + u32(11)]), u32(2) * r_7 + side_2 + u32(1));
        side_2 = side_2 + u32(1);
    }
    return;
}

fn cs_pick_1( r_8 : u32)
{
    if(r_8 >= (cs_rows_0()))
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
        var k_7 : u32;
        var s_4 : u32;
        var ob_2 : u32;
        var co_2 : u32;
        var p_3 : vec3<f32>;
        var u_3 : vec3<f32>;
        var v_4 : f32;
        var m_eff_3 : f32;
        var t_in_2 : f32;
        var _S61 : bool = cs_side_0(r_8, side_3, &(k_7), &(s_4), &(ob_2), &(co_2), &(p_3), &(u_3), &(v_4), &(m_eff_3), &(t_in_2));
        var _S62 : bool;
        if(!_S61)
        {
            _S62 = true;
        }
        else
        {
            var _S63 : u32 = (bitcast<u32>((m_eff_3 * v_4)));
            var _S64 : u32 = atomicLoad(&(atom_0[a_cs_0(s_4)]));
            _S62 = _S63 != _S64;
        }
        if(_S62)
        {
            side_3 = side_3 + u32(1);
            continue;
        }
        var id_1 : vec3<u32> = cs_identity_0(r_8, side_3);
        var at_2 : u32 = a_cs_0(s_4);
        var _S65 : u32 = atomicLoad(&(atom_0[at_2 + u32(2)]));
        if((~_S65) != (id_1.x))
        {
            side_3 = side_3 + u32(1);
            continue;
        }
        var _S66 : u32 = atomicLoad(&(atom_0[at_2 + u32(9)]));
        if((~_S66) != (id_1.y))
        {
            side_3 = side_3 + u32(1);
            continue;
        }
        var _S67 : u32 = atomicMax(&(atom_0[at_2 + u32(10)]), ~id_1.z);
        side_3 = side_3 + u32(1);
    }
    return;
}

fn cs_pick_2( r_9 : u32)
{
    if(r_9 >= (cs_rows_0()))
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
        var k_8 : u32;
        var s_5 : u32;
        var ob_3 : u32;
        var co_3 : u32;
        var p_4 : vec3<f32>;
        var u_4 : vec3<f32>;
        var v_5 : f32;
        var m_eff_4 : f32;
        var t_in_3 : f32;
        var _S68 : bool = cs_side_0(r_9, side_4, &(k_8), &(s_5), &(ob_3), &(co_3), &(p_4), &(u_4), &(v_5), &(m_eff_4), &(t_in_3));
        var _S69 : bool;
        if(!_S68)
        {
            _S69 = true;
        }
        else
        {
            var _S70 : u32 = (bitcast<u32>((m_eff_4 * v_5)));
            var _S71 : u32 = atomicLoad(&(atom_0[a_cs_0(s_5)]));
            _S69 = _S70 != _S71;
        }
        if(_S69)
        {
            side_4 = side_4 + u32(1);
            continue;
        }
        var id_2 : vec3<u32> = cs_identity_0(r_9, side_4);
        var at_3 : u32 = a_cs_0(s_5);
        var _S72 : u32 = atomicLoad(&(atom_0[at_3 + u32(2)]));
        if((~_S72) != (id_2.x))
        {
            side_4 = side_4 + u32(1);
            continue;
        }
        var _S73 : u32 = atomicMax(&(atom_0[at_3 + u32(9)]), ~id_2.y);
        side_4 = side_4 + u32(1);
    }
    return;
}

fn cs_pick_3( r_10 : u32)
{
    if(r_10 >= (cs_rows_0()))
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
        var k_9 : u32;
        var s_6 : u32;
        var ob_4 : u32;
        var co_4 : u32;
        var p_5 : vec3<f32>;
        var u_5 : vec3<f32>;
        var v_6 : f32;
        var m_eff_5 : f32;
        var t_in_4 : f32;
        var _S74 : bool = cs_side_0(r_10, side_5, &(k_9), &(s_6), &(ob_4), &(co_4), &(p_5), &(u_5), &(v_6), &(m_eff_5), &(t_in_4));
        var _S75 : bool;
        if(!_S74)
        {
            _S75 = true;
        }
        else
        {
            var _S76 : u32 = (bitcast<u32>((m_eff_5 * v_6)));
            var _S77 : u32 = atomicLoad(&(atom_0[a_cs_0(s_6)]));
            _S75 = _S76 != _S77;
        }
        if(_S75)
        {
            side_5 = side_5 + u32(1);
            continue;
        }
        var id_3 : vec3<u32> = cs_identity_0(r_10, side_5);
        var at_4 : u32 = a_cs_0(s_6);
        var _S78 : u32 = atomicMax(&(atom_0[at_4 + u32(2)]), ~id_3.x);
        side_5 = side_5 + u32(1);
    }
    return;
}

struct IlPlace_0
{
     along_0 : f32,
     radial_0 : f32,
     entry_0 : bool,
     crossed_0 : bool,
};

fn structure_pose_1( slot_2 : u32,  b_22 : u32,  q_8 : ptr<function, Quat_0>,  origin_2 : ptr<function, vec3<f32>>)
{
    var _S79 : u32 = u32(2) * slot_2;
    var com_1 : vec3<f32> = frames_0[_S79].xyz;
    var _S80 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_22 + u32(1)])), conj_0(quat_of_0(frames_0[_S79 + u32(1)]))));
    (*q_8) = _S80;
    (*origin_2) = from_rubble_0(bodies_0[b_22].xyz) - rotate_0(_S80, com_1);
    return;
}

fn chunk_box_world_1( k_10 : u32,  q_9 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_3 : u32 = body_of_0(k_10);
    var origin_3 : vec3<f32>;
    structure_pose_0(slot_3, u32(7) * slot_3, &((*q_9)), &(origin_3));
    return origin_3 + rotate_0((*q_9), geometry_0[u32(3) * k_10].xyz);
}

fn il_crossed_0( k_11 : u32,  a_1 : vec3<f32>,  d_7 : vec3<f32>,  length_0 : f32,  r_11 : f32) -> bool
{
    var q_10 : Quat_0;
    var cc_0 : vec3<f32> = chunk_box_world_1(k_11, &(q_10));
    var _S81 : u32 = u32(3) * k_11;
    var half_0 : vec3<f32> = geometry_0[_S81 + u32(1)].xyz;
    var _S82 : Quat_0 = conj_0(normalized_0(quat_mul_0(q_10, quat_of_0(geometry_0[_S81 + u32(2)]))));
    var al_0 : vec3<f32> = rotate_0(_S82, a_1 - cc_0);
    var ul_0 : vec3<f32> = rotate_0(_S82, d_7);
    var lo_0 : f32 = 0.0f;
    var hi_0 : f32 = length_0;
    var pass_0 : u32 = u32(0);
    loop
    {
        if(pass_0 < u32(40))
        {
        }
        else
        {
            break;
        }
        var _S83 : f32 = hi_0 - lo_0;
        var m1_0 : f32 = lo_0 + _S83 / 3.0f;
        var m2_0 : f32 = hi_0 - _S83 / 3.0f;
        var _S84 : vec3<f32> = vec3<f32>(0.0f);
        var e1_0 : vec3<f32> = max(abs(al_0 + ul_0 * vec3<f32>(m1_0)) - half_0, _S84);
        var e2_0 : vec3<f32> = max(abs(al_0 + ul_0 * vec3<f32>(m2_0)) - half_0, _S84);
        if((dot(e1_0, e1_0)) <= (dot(e2_0, e2_0)))
        {
            hi_0 = m2_0;
        }
        else
        {
            lo_0 = m1_0;
        }
        pass_0 = pass_0 + u32(1);
    }
    var e_1 : vec3<f32> = max(abs(al_0 + ul_0 * vec3<f32>((0.5f * (lo_0 + hi_0)))) - half_0, vec3<f32>(0.0f));
    return (dot(e_1, e_1)) < (0.99980002641677856f * r_11 * r_11);
}

fn il_place_0( k_12 : u32,  entry_1 : u32,  contact_6 : vec3<f32>,  d_8 : vec3<f32>,  r_12 : f32,  h_0 : f32) -> IlPlace_0
{
    var slot_4 : u32 = body_of_0(k_12);
    var q_11 : Quat_0;
    var origin_4 : vec3<f32>;
    structure_pose_1(slot_4, u32(7) * slot_4, &(q_11), &(origin_4));
    var rel_2 : vec3<f32> = origin_4 + rotate_0(q_11, chunks_0[u32(4) * k_12].xyz) - contact_6;
    var p_6 : IlPlace_0;
    var _S85 : f32 = dot(rel_2, d_8);
    p_6.along_0 = _S85;
    p_6.radial_0 = length(rel_2 - d_8 * vec3<f32>(_S85));
    p_6.entry_0 = k_12 == entry_1;
    p_6.crossed_0 = il_crossed_0(k_12, contact_6 - d_8 * vec3<f32>(r_12), d_8, h_0 + 2.0f * r_12, r_12);
    return p_6;
}

fn il_plug_0( outcome_0 : u32,  r_13 : f32,  h_1 : f32,  p_7 : IlPlace_0) -> bool
{
    var _S86 : bool;
    if(((outcome_0 & (u32(8)))) != u32(0))
    {
        if(p_7.entry_0)
        {
            _S86 = true;
        }
        else
        {
            _S86 = p_7.crossed_0;
        }
        if(_S86)
        {
            _S86 = true;
        }
        else
        {
            if((p_7.along_0) >= 0.0f)
            {
                _S86 = (p_7.along_0) <= h_1;
            }
            else
            {
                _S86 = false;
            }
            if(_S86)
            {
                _S86 = (p_7.radial_0) <= (r_13 + p_7.along_0);
            }
            else
            {
                _S86 = false;
            }
        }
    }
    else
    {
        _S86 = false;
    }
    return _S86;
}

fn il_scab_0( outcome_1 : u32,  r_14 : f32,  h_2 : f32,  scab_0 : f32,  p_8 : IlPlace_0) -> bool
{
    var _S87 : bool;
    if(((outcome_1 & (u32(4)))) != u32(0))
    {
        _S87 = (p_8.along_0) >= (h_2 - scab_0);
    }
    else
    {
        _S87 = false;
    }
    if(_S87)
    {
        _S87 = (p_8.along_0) <= h_2;
    }
    else
    {
        _S87 = false;
    }
    if(_S87)
    {
        _S87 = (p_8.radial_0) <= (r_14 + h_2);
    }
    else
    {
        _S87 = false;
    }
    return _S87;
}

fn il_crater_0( outcome_2 : u32,  r_15 : f32,  depth_0 : f32,  a_2 : f32,  p_9 : IlPlace_0) -> bool
{
    var _S88 : bool;
    if(((outcome_2 & (u32(2)))) != u32(0))
    {
        _S88 = (p_9.along_0) >= 0.0f;
    }
    else
    {
        _S88 = false;
    }
    if(_S88)
    {
        _S88 = (p_9.along_0) <= depth_0;
    }
    else
    {
        _S88 = false;
    }
    if(_S88)
    {
        _S88 = (p_9.radial_0) <= (max(r_15, a_2));
    }
    else
    {
        _S88 = false;
    }
    return _S88;
}

fn a_strike_dist_0() -> u32
{
    return a_cs_0(u32(0)) + u32(12) * params_0.body_count_0;
}

fn a_events_0() -> u32
{
    return a_strike_dist_0() + params_0.chunk_count_0;
}

fn event_broken_0( bond_0 : u32)
{
    if((params_0.ev_broken_cap_0) == u32(0))
    {
        return;
    }
    var _S89 : u32 = a_counters_0();
    var i_0 : u32 = atomicAdd(&(atom_0[_S89 + u32(20)]), u32(1));
    if(i_0 < (params_0.ev_broken_cap_0))
    {
        atomicStore(&(atom_0[a_events_0() + i_0]), bond_0);
    }
    else
    {
        var _S90 : u32 = atomicOr(&(atom_0[_S89 + u32(3)]), u32(128));
    }
    return;
}

fn il_breaks_0( i_1 : u32)
{
    var cut_0 : bool;
    if(i_1 >= (params_0.bond_count_0))
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = work_0[w_alive_0() + i_1] == u32(0);
    }
    if(cut_0)
    {
        return;
    }
    var bond_1 : vec4<u32> = bonds_0[i_1];
    var _S91 : u32 = bond_1.x;
    var b_23 : u32 = body_of_0(_S91);
    var _S92 : bool = il_struck_0(b_23);
    if(!_S92)
    {
        return;
    }
    var _S93 : u32 = lp_body_0(b_23);
    var s0_0 : vec4<f32> = lp_0[_S93];
    var s1_0 : vec4<f32> = lp_0[_S93 + u32(1)];
    var outcome_3 : u32 = (bitcast<u32>((lp_0[_S93].w)));
    if(((outcome_3 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_7 : vec3<f32>;
    var d_9 : vec3<f32>;
    var speed_6 : f32;
    il_path_2(b_23, &(contact_7), &(d_9), &(speed_6));
    var entry_2 : u32 = il_entry_0(b_23);
    var _S94 : f32 = max(il_striker_0(b_23).radius_0, 0.00100000004749745f);
    var h_3 : f32 = s0_0.x;
    var pa_0 : IlPlace_0 = il_place_0(_S91, entry_2, contact_7, d_9, _S94, h_3);
    var pb_0 : IlPlace_0 = il_place_0(bond_1.y, entry_2, contact_7, d_9, _S94, h_3);
    if((il_plug_0(outcome_3, _S94, h_3, pa_0)) != (il_plug_0(outcome_3, _S94, h_3, pb_0)))
    {
        cut_0 = true;
    }
    else
    {
        var _S95 : f32 = s1_0.y;
        cut_0 = (il_scab_0(outcome_3, _S94, h_3, _S95, pa_0)) != (il_scab_0(outcome_3, _S94, h_3, _S95, pb_0));
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = il_crater_0(outcome_3, _S94, s1_0.x, s1_0.z, pa_0);
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = il_crater_0(outcome_3, _S94, s1_0.x, s1_0.z, pb_0);
    }
    if(cut_0)
    {
        work_0[w_alive_0() + i_1] = u32(0);
        var _S96 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_23]), u32(1));
        var _S97 : u32 = a_counters_0();
        var _S98 : u32 = atomicAdd(&(atom_0[_S97]), u32(1));
        var _S99 : u32 = atomicAdd(&(atom_0[_S97 + u32(1)]), u32(1));
        event_broken_0(i_1);
        var _S100 : u32 = atomicAdd(&(atom_0[_S97 + u32(17)]), u32(1));
    }
    return;
}

fn to_rubble_0( v_7 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_7.x, v_7.z, - v_7.y);
}

fn read64_0( at_5 : u32,  scale_0 : f32) -> f32
{
    var _S101 : u32 = atomicLoad(&(atom_0[at_5 + u32(1)]));
    var hi_1 : i32 = (bitcast<i32>((_S101)));
    var lo_1 : u32 = atomicLoad(&(atom_0[at_5]));
    return (f32(hi_1) * 4.294967296e+09f + f32(lo_1)) / scale_0;
}

fn il_carry_0( k_13 : u32)
{
    if(k_13 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S102 : u32 = w_carry_0() + k_13;
    var c_2 : u32 = work_0[_S102];
    if(work_0[_S102] == u32(0))
    {
        return;
    }
    work_0[_S102] = u32(0);
    var b0_0 : u32 = c_2 - u32(1);
    var _S103 : bool = il_struck_0(b0_0);
    if(!_S103)
    {
        return;
    }
    var contact_8 : vec3<f32>;
    var d_10 : vec3<f32>;
    var speed_7 : f32;
    il_path_2(b0_0, &(contact_8), &(d_10), &(speed_7));
    var dr_0 : vec3<f32> = to_rubble_0(d_10);
    var sk_0 : IlStriker_0 = il_striker_0(b0_0);
    var plug_mass_0 : f32 = read64_0(a_il_0(b0_0) + u32(2), params_0.scale_m_0);
    var v_wall_0 : vec3<f32> = bodies_0[u32(7) * b0_0 + u32(3)].xyz;
    var _S104 : u32 = u32(7) * sk_0.body_0;
    var v_shot_0 : vec3<f32> = bodies_0[_S104 + u32(3)].xyz;
    var closing_0 : f32 = dot(v_shot_0 - v_wall_0, dr_0);
    var v_r_0 : f32 = sk_0.mass_0 * closing_0 / (sk_0.mass_0 + plug_mass_0);
    var _S105 : vec3<f32> = v_shot_0 + dr_0 * vec3<f32>((v_r_0 - closing_0));
    var _S106 : vec3<f32> = v_wall_0 + dr_0 * vec3<f32>(v_r_0);
    var _S107 : vec3<f32> = bodies_0[_S104].xyz + dr_0 * vec3<f32>(((speed_7 - v_r_0) * clamp(il_t_in_0(b0_0), 0.0f, params_0.dt_0)));
    var _S108 : u32 = body_of_0(k_13);
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
        bodies_0[s_at_0].x = _S107.x;
        bodies_0[s_at_0].y = _S107.y;
        bodies_0[s_at_0].z = _S107.z;
        bodies_0[s_at_0 + u32(3)].x = _S105.x;
        bodies_0[s_at_0 + u32(3)].y = _S105.y;
        bodies_0[s_at_0 + u32(3)].z = _S105.z;
        bodies_0[u32(7) * (_S108 + region_0 * params_0.max_bodies_0) + u32(3)].x = _S106.x;
        bodies_0[u32(7) * (_S108 + region_0 * params_0.max_bodies_0) + u32(3)].y = _S106.y;
        bodies_0[u32(7) * (_S108 + region_0 * params_0.max_bodies_0) + u32(3)].z = _S106.z;
        region_0 = region_0 + u32(1);
    }
    return;
}

fn add64_0( at_6 : u32,  value_0 : f32,  scale_1 : f32)
{
    var x_2 : f32 = round(value_0 * scale_1);
    var x_3 : f32;
    if(!((abs(x_2)) < 1.073741824e+09f))
    {
        var _S109 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
        x_3 = 0.0f;
    }
    else
    {
        x_3 = x_2;
    }
    var v_8 : i32 = i32(x_3);
    var lo_2 : u32 = (bitcast<u32>((v_8)));
    var old_0 : u32 = atomicAdd(&(atom_0[at_6]), lo_2);
    var carry_0 : u32;
    if((old_0 + lo_2) < old_0)
    {
        carry_0 = u32(1);
    }
    else
    {
        carry_0 = u32(0);
    }
    var _S110 : u32;
    if(v_8 < i32(0))
    {
        _S110 = u32(4294967295);
    }
    else
    {
        _S110 = u32(0);
    }
    var hi_2 : u32 = _S110 + carry_0;
    if(hi_2 != u32(0))
    {
        var _S111 : u32 = atomicAdd(&(atom_0[at_6 + u32(1)]), hi_2);
    }
    return;
}

fn il_mark_0( k_14 : u32)
{
    if(k_14 >= (params_0.chunk_count_0))
    {
        return;
    }
    var b_24 : u32 = body_of_0(k_14);
    var _S112 : bool = il_struck_0(b_24);
    if(!_S112)
    {
        return;
    }
    var _S113 : u32 = lp_body_0(b_24);
    var s0_1 : vec4<f32> = lp_0[_S113];
    var s1_1 : vec4<f32> = lp_0[_S113 + u32(1)];
    var outcome_4 : u32 = (bitcast<u32>((lp_0[_S113].w)));
    if(((outcome_4 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_9 : vec3<f32>;
    var d_11 : vec3<f32>;
    var speed_8 : f32;
    il_path_2(b_24, &(contact_9), &(d_11), &(speed_8));
    var entry_3 : u32 = il_entry_0(b_24);
    var sk_1 : IlStriker_0 = il_striker_0(b_24);
    var _S114 : f32 = max(sk_1.radius_0, 0.00100000004749745f);
    var _S115 : f32 = s0_1.x;
    var p_10 : IlPlace_0 = il_place_0(k_14, entry_3, contact_9, d_11, _S114, _S115);
    var plug_0 : bool = il_plug_0(outcome_4, _S114, _S115, p_10);
    if((sk_1.launch_0) != u32(0))
    {
        var t_3 : vec2<f32> = il_interval_0(k_14, contact_9 - d_11 * vec3<f32>(sk_1.bound_0), d_11, sk_1.bound_0);
        var _S116 : f32 = t_3.x;
        var _S117 : f32 = t_3.y;
        var path_0 : bool;
        if(_S116 <= _S117)
        {
            path_0 = _S116 <= (_S115 + 2.0f * sk_1.bound_0);
        }
        else
        {
            path_0 = false;
        }
        if(path_0)
        {
            path_0 = _S117 >= 0.0f;
        }
        else
        {
            path_0 = false;
        }
        if(plug_0)
        {
            path_0 = true;
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            path_0 = il_scab_0(outcome_4, _S114, _S115, s1_1.y, p_10);
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            path_0 = il_crater_0(outcome_4, _S114, s1_1.x, s1_1.z, p_10);
        }
        if(path_0)
        {
            work_0[w_marks_0() + k_14] = sk_1.launch_0;
        }
    }
    if(plug_0)
    {
        work_0[w_carry_0() + k_14] = b_24 + u32(1);
        add64_0(a_il_0(b_24) + u32(2), chunks_0[u32(4) * k_14].w, params_0.scale_m_0);
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

fn fdiv_0( a_3 : f32,  b_25 : f32) -> f32
{
    return a_3 / b_25;
}

fn fpow_0( a_4 : f32,  b_26 : f32) -> f32
{
    return pow(a_4, b_26);
}

fn dif_factor_0( mat_0 : JointMaterial_0,  strain_rate_0 : f32) -> f32
{
    var r_16 : f32 = abs(strain_rate_0);
    var ref_0 : f32 = mat_0.dif_0.x;
    if(r_16 <= ref_0)
    {
        return 1.0f;
    }
    var _S118 : f32 = mat_0.dif_0.z;
    var f_0 : f32;
    if(r_16 <= _S118)
    {
        f_0 = fpow_0(fdiv_0(r_16, ref_0), mat_0.dif_0.y);
    }
    else
    {
        f_0 = fpow_0(fdiv_0(_S118, ref_0), mat_0.dif_0.y) * fpow_0(fdiv_0(r_16, _S118), mat_0.dif_0.w);
    }
    return clamp(f_0, 1.0f, mat_0.misc_0.x);
}

fn il_impulse_0( b_27 : u32,  sb_0 : u32,  m_shot_0 : f32,  contact_10 : vec3<f32>,  d_12 : vec3<f32>,  j_0 : f32)
{
    var jr_0 : vec3<f32> = to_rubble_0(d_12) * vec3<f32>(j_0);
    var at_7 : u32 = u32(7) * b_27;
    var inv_mass_1 : f32 = bodies_0[at_7].w;
    var _S119 : vec3<f32> = jr_0 * vec3<f32>(inv_mass_1);
    var _S120 : vec3<f32> = vec3<f32>(0.0f);
    var dw_0 : vec3<f32>;
    if(inv_mass_1 > 0.0f)
    {
        var q_12 : Quat_0 = quat_of_0(bodies_0[at_7 + u32(1)]);
        dw_0 = rotate_0(q_12, bodies_0[at_7 + u32(2)].xyz * rotate_0(conj_0(q_12), cross(to_rubble_0(contact_10) - bodies_0[at_7].xyz, jr_0)));
    }
    else
    {
        dw_0 = _S120;
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
        var _S121 : vec3<f32> = bodies_0[s_at_1].xyz - jr_0 / vec3<f32>(m_shot_0);
        bodies_0[s_at_1].x = _S121.x;
        bodies_0[s_at_1].y = _S121.y;
        bodies_0[s_at_1].z = _S121.z;
        var w_at_0 : u32 = u32(7) * (b_27 + region_1 * params_0.max_bodies_0);
        var _S122 : u32 = w_at_0 + u32(3);
        var _S123 : vec3<f32> = bodies_0[_S122].xyz + _S119;
        bodies_0[_S122].x = _S123.x;
        bodies_0[_S122].y = _S123.y;
        bodies_0[_S122].z = _S123.z;
        var _S124 : u32 = w_at_0 + u32(4);
        var _S125 : vec3<f32> = bodies_0[_S124].xyz + dw_0;
        bodies_0[_S124].x = _S125.x;
        bodies_0[_S124].y = _S125.y;
        bodies_0[_S124].z = _S125.z;
        region_1 = region_1 + u32(1);
    }
    return;
}

fn il_strike_0( b_28 : u32)
{
    var damage_0 : bool;
    if(b_28 >= (params_0.body_count_0))
    {
        damage_0 = true;
    }
    else
    {
        var _S126 : bool = il_struck_0(b_28);
        damage_0 = !_S126;
    }
    if(damage_0)
    {
        return;
    }
    var t_in_5 : f32 = il_t_in_0(b_28);
    var entry_4 : u32 = il_entry_0(b_28);
    var contact_11 : vec3<f32>;
    var d_13 : vec3<f32>;
    var speed_9 : f32;
    il_path_2(b_28, &(contact_11), &(d_13), &(speed_9));
    var sk_2 : IlStriker_0 = il_striker_0(b_28);
    var _S127 : f32 = max(sk_2.radius_0, 0.00100000004749745f);
    if((sk_2.launch_0) != u32(0))
    {
        damage_0 = work_0[w_marks_0() + entry_4] == (sk_2.launch_0);
    }
    else
    {
        damage_0 = false;
    }
    if(damage_0)
    {
        var _S128 : u32 = lp_body_0(b_28);
        var _S129 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S128] = _S129;
        lp_0[_S128 + u32(1)] = _S129;
        return;
    }
    var _S130 : u32 = atomicLoad(&(atom_0[a_il_0(b_28)]));
    var _S131 : f32 = max((bitcast<f32>((_S130))) - _S127, 9.99999997475242708e-07f);
    var o_2 : u32 = params_0.ext_chunks_0 + u32(4) * entry_4;
    var mat0_0 : vec4<f32> = ext_0[o_2 + u32(1)];
    var dm_0 : JointMaterial_0;
    var _S132 : vec4<f32> = vec4<f32>(0.0f);
    dm_0.strength_0 = _S132;
    dm_0.energy_0 = _S132;
    dm_0.dif_0 = ext_0[o_2 + u32(2)];
    var mat3_0 : vec4<f32> = ext_0[o_2 + u32(3)];
    var _S133 : f32 = mat3_0.x;
    dm_0.misc_0 = vec4<f32>(_S133, 0.0f, 0.0f, 0.0f);
    dm_0.kind_flags_0 = vec4<u32>(u32(0));
    var nu_0 : f32 = mat3_0.y;
    var e_target_0 : f32 = mat3_0.z / (1.0f - nu_0 * nu_0);
    var e_star_0 : f32 = 1.0f / (1.0f / sk_2.modulus_0 + 1.0f / e_target_0);
    var _S134 : f32 = max(speed_9, 0.00100000004749745f);
    var _S135 : f32 = sqrt(_S127);
    var h0_0 : f32 = pow(15.0f * sk_2.m_eff_0 * _S134 * _S134 / (16.0f * e_star_0 * _S135), 0.40000000596046448f);
    var t_c_0 : f32 = 2.94000005722045898f * h0_0 / _S134;
    var force_0 : f32 = 1.33333337306976318f * e_star_0 * _S135 * pow(h0_0, 1.5f);
    var a_5 : f32 = sqrt(_S127 * h0_0);
    var p0_0 : f32 = 3.0f * force_0 / (6.28318548202514648f * a_5 * a_5);
    var ring_0 : f32 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    var k_15 : f32;
    if(_S133 > 0.0f)
    {
        k_15 = dif_factor_0(dm_0, _S134 / _S131);
    }
    else
    {
        k_15 = 1.0f;
    }
    var ft_0 : f32 = k_15 * mat0_0.y;
    var fc_0 : f32 = k_15 * mat0_0.z;
    var tau_0 : f32 = k_15 * mat0_0.w;
    if(p0_0 > fc_0)
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ring_0 > ft_0;
    }
    var _S136 : f32 = fc_0 * (3.14159274101257324f * _S127 * _S127);
    var _S137 : f32 = min(_S136 * _S131, 1.57079637050628662f * (2.0f * _S127 + _S131) * _S131 * _S131 * tau_0);
    var _S138 : f32 = 2.0f * _S137;
    var v_bl_0 : f32 = sqrt(_S138 / sk_2.m_eff_0);
    var perforate_0 : bool;
    if(damage_0)
    {
        perforate_0 = _S134 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    var r2_0 : f32 = _S127 + _S131;
    var _S139 : f32 = mat0_0.x;
    var plug_1 : f32 = _S139 * 3.14159274101257324f * _S131 / 3.0f * (_S127 * _S127 + _S127 * r2_0 + r2_0 * r2_0);
    var v_res_0 : f32;
    if(perforate_0)
    {
        v_res_0 = sk_2.mass_0 / (sk_2.mass_0 + plug_1) * sqrt(max(_S134 * _S134 - v_bl_0 * v_bl_0, 0.0f));
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
        depth_1 = min(0.5f * sk_2.m_eff_0 * _S134 * _S134 / _S136, _S131);
    }
    else
    {
        depth_1 = 0.0f;
    }
    var crater_0 : bool = depth_1 > a_5;
    var c_3 : f32 = sqrt(e_target_0 / _S139);
    var z_t_0 : f32 = _S139 * c_3;
    var rho_p_0 : f32 = sk_2.mass_0 / (4.18879032135009766f * _S127 * _S127 * _S127);
    var z_p_0 : f32 = rho_p_0 * sqrt(sk_2.modulus_0 / rho_p_0);
    var mean_pressure_0 : f32 = force_0 / (3.14159274101257324f * a_5 * a_5);
    var _S140 : f32 = min(min(z_t_0 * _S134 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_0);
    var source_radius_0 : f32;
    if(mean_pressure_0 > fc_0)
    {
        source_radius_0 = max(a_5, _S127);
    }
    else
    {
        source_radius_0 = a_5;
    }
    var pulse_0 : f32 = c_3 * t_c_0;
    var back_0 : f32 = _S140 * min(source_radius_0 / _S131, 1.0f) * min(2.0f * _S131 / pulse_0, 1.0f);
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
        scab_1 = 0.5f * min(pulse_0, _S131);
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
    var _S141 : u32;
    if(crater_0)
    {
        _S141 = u32(2);
    }
    else
    {
        _S141 = u32(0);
    }
    var _S142 : u32 = (region_2 | (_S141));
    if(spall_0)
    {
        region_2 = u32(4);
    }
    else
    {
        region_2 = u32(0);
    }
    var _S143 : u32 = (_S142 | (region_2));
    if(perforate_0)
    {
        region_2 = u32(8);
    }
    else
    {
        region_2 = u32(0);
    }
    var _S144 : u32 = lp_body_0(b_28);
    lp_0[_S144] = vec4<f32>(_S131, v_bl_0, v_res_0, (bitcast<f32>(((_S143 | (region_2))))));
    if(crater_0)
    {
        k_15 = depth_1;
    }
    else
    {
        k_15 = 0.0f;
    }
    lp_0[_S144 + u32(1)] = vec4<f32>(k_15, scab_1, a_5, _S137);
    if(perforate_0)
    {
        var _S145 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(14)]), u32(1));
    }
    else
    {
        if(spall_0)
        {
            var _S146 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(13)]), u32(1));
        }
        else
        {
            if(crater_0)
            {
                var _S147 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(12)]), u32(1));
            }
            else
            {
                var _S148 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(11)]), u32(1));
            }
        }
    }
    if(perforate_0)
    {
        if((params_0.fracture_work_0) != u32(0))
        {
            il_impulse_0(b_28, sk_2.body_0, sk_2.mass_0, contact_11, d_13, sk_2.m_eff_0 * (_S134 - sqrt(max(_S134 * _S134 - _S138 / sk_2.m_eff_0, 0.0f))));
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_28, sk_2.body_0, sk_2.mass_0, contact_11, d_13, sk_2.m_eff_0 * _S134);
            var _S149 : vec3<f32> = to_rubble_0(d_13) * vec3<f32>((_S134 * clamp(t_in_5, 0.0f, params_0.dt_0)));
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
                var _S150 : u32 = u32(7) * (sk_2.body_0 + region_2 * params_0.max_bodies_0);
                var _S151 : vec3<f32> = bodies_0[_S150].xyz + _S149;
                bodies_0[_S150].x = _S151.x;
                bodies_0[_S150].y = _S151.y;
                bodies_0[_S150].z = _S151.z;
                region_2 = region_2 + u32(1);
            }
        }
    }
    return;
}

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_1 : u32,  thread_0 : u32) -> u32
{
    if(thread_0 == u32(0))
    {
        g_group_uniform_0 = value_1;
    }
    var _S152 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S152;
}

fn group_uniform_float_0( value_2 : f32,  thread_1 : u32) -> f32
{
    var _S153 : u32 = group_uniform_0((bitcast<u32>((value_2))), thread_1);
    return (bitcast<f32>((_S153)));
}

var<workgroup> g_il_best_0 : array<f32, i32(64)>;

fn il_run_0( b_29 : u32,  i_2 : u32)
{
    var _S154 : bool = il_struck_0(b_29);
    var j_1 : u32;
    if(_S154)
    {
        j_1 = u32(1);
    }
    else
    {
        j_1 = u32(0);
    }
    var _S155 : u32 = group_uniform_0(j_1, i_2);
    if(_S155 == u32(0))
    {
        return;
    }
    var _S156 : f32 = 0.00100000004749745f + 0.01999999955296516f * il_striker_0(b_29).radius_0;
    var _S157 : u32 = a_il_0(b_29);
    var _S158 : u32 = atomicLoad(&(atom_0[_S157 + u32(4)]));
    var _S159 : u32 = atomicLoad(&(atom_0[_S157 + u32(1)]));
    var _S160 : u32 = atomicLoad(&(atom_0[_S157]));
    var _S161 : f32 = group_uniform_float_0((bitcast<f32>((_S160))), i_2);
    var end_0 : f32 = _S161;
    loop
    {
        var best_0 : f32 = end_0;
        j_1 = i_2;
        loop
        {
            if(j_1 < _S159)
            {
            }
            else
            {
                break;
            }
            var at_8 : u32 = w_il_tube_0() + u32(2) * (_S158 + j_1);
            if((bitcast<f32>((work_0[at_8]))) <= (end_0 + _S156))
            {
                best_0 = max(best_0, (bitcast<f32>((work_0[at_8 + u32(1)]))));
            }
            j_1 = j_1 + u32(64);
        }
        g_il_best_0[i_2] = best_0;
        workgroupBarrier();
        var step_1 : u32 = u32(32);
        loop
        {
            if(step_1 > u32(0))
            {
            }
            else
            {
                break;
            }
            if(i_2 < step_1)
            {
                g_il_best_0[i_2] = max(g_il_best_0[i_2], g_il_best_0[i_2 + step_1]);
            }
            workgroupBarrier();
            step_1 = (step_1 >> (u32(1)));
        }
        var next_0 : f32 = group_uniform_float_0(g_il_best_0[i32(0)], i_2);
        workgroupBarrier();
        if(!(next_0 > end_0))
        {
            break;
        }
        end_0 = next_0;
    }
    if(i_2 == u32(0))
    {
        atomicStore(&(atom_0[_S157]), (bitcast<u32>((end_0))));
    }
    return;
}

fn il_tube_collect_0( k_16 : u32)
{
    var b_30 : u32;
    var t_4 : vec2<f32>;
    var _S162 : bool = il_tube_0(k_16, &(b_30), &(t_4));
    if(!_S162)
    {
        return;
    }
    var _S163 : u32 = w_il_tube_0();
    var _S164 : u32 = atomicLoad(&(atom_0[a_il_0(b_30) + u32(4)]));
    var _S165 : u32 = atomicAdd(&(atom_0[a_il_0(b_30) + u32(5)]), u32(1));
    var at_9 : u32 = _S163 + u32(2) * (_S164 + _S165);
    work_0[at_9] = (bitcast<u32>((t_4.x)));
    work_0[at_9 + u32(1)] = (bitcast<u32>((t_4.y)));
    return;
}

fn il_tube_base_0( b_31 : u32)
{
    var _S166 : bool;
    if(b_31 < (params_0.body_count_0))
    {
        var _S167 : bool = il_struck_0(b_31);
        _S166 = _S167;
    }
    else
    {
        _S166 = false;
    }
    if(_S166)
    {
        var _S168 : u32 = a_il_0(b_31);
        var _S169 : u32 = atomicLoad(&(atom_0[_S168 + u32(1)]));
        var _S170 : u32 = atomicAdd(&(atom_0[a_il_cursor_0()]), _S169);
        atomicStore(&(atom_0[_S168 + u32(4)]), _S170);
    }
    return;
}

fn il_tube_count_0( k_17 : u32)
{
    var b_32 : u32;
    var t_5 : vec2<f32>;
    var _S171 : bool = il_tube_0(k_17, &(b_32), &(t_5));
    if(_S171)
    {
        var _S172 : u32 = atomicAdd(&(atom_0[a_il_0(b_32) + u32(1)]), u32(1));
    }
    return;
}

fn il_begin_0( b_33 : u32)
{
    var _S173 : bool;
    if(b_33 >= (params_0.body_count_0))
    {
        _S173 = true;
    }
    else
    {
        var _S174 : bool = il_struck_0(b_33);
        _S173 = !_S174;
    }
    if(_S173)
    {
        return;
    }
    var contact_12 : vec3<f32>;
    var d_14 : vec3<f32>;
    var speed_10 : f32;
    il_path_2(b_33, &(contact_12), &(d_14), &(speed_10));
    var _S175 : u32 = il_entry_0(b_33);
    var _S176 : u32 = a_il_0(b_33);
    atomicStore(&(atom_0[_S176]), (bitcast<u32>((max(il_interval_0(_S175, contact_12, d_14, il_striker_0(b_33).radius_0).y, 0.0f)))));
    var i_3 : u32 = u32(1);
    loop
    {
        if(i_3 < u32(6))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[_S176 + i_3]), u32(0));
        i_3 = i_3 + u32(1);
    }
    return;
}

fn il_breaks_1( i_4 : u32)
{
    var cut_1 : bool;
    if(i_4 >= (params_0.bond_count_0))
    {
        cut_1 = true;
    }
    else
    {
        cut_1 = work_0[w_alive_0() + i_4] == u32(0);
    }
    if(cut_1)
    {
        return;
    }
    var bond_2 : vec4<u32> = bonds_0[i_4];
    var _S177 : u32 = bond_2.x;
    var b_34 : u32 = body_of_0(_S177);
    if(!il_struck_1(b_34))
    {
        return;
    }
    var _S178 : u32 = lp_body_0(b_34);
    var s0_2 : vec4<f32> = lp_0[_S178];
    var s1_2 : vec4<f32> = lp_0[_S178 + u32(1)];
    var outcome_5 : u32 = (bitcast<u32>((lp_0[_S178].w)));
    if(((outcome_5 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_13 : vec3<f32>;
    var d_15 : vec3<f32>;
    var speed_11 : f32;
    il_path_3(b_34, &(contact_13), &(d_15), &(speed_11));
    var entry_5 : u32 = il_entry_1(b_34);
    var _S179 : f32 = max(il_striker_1(b_34).radius_0, 0.00100000004749745f);
    var h_4 : f32 = s0_2.x;
    var pa_1 : IlPlace_0 = il_place_0(_S177, entry_5, contact_13, d_15, _S179, h_4);
    var pb_1 : IlPlace_0 = il_place_0(bond_2.y, entry_5, contact_13, d_15, _S179, h_4);
    if((il_plug_0(outcome_5, _S179, h_4, pa_1)) != (il_plug_0(outcome_5, _S179, h_4, pb_1)))
    {
        cut_1 = true;
    }
    else
    {
        var _S180 : f32 = s1_2.y;
        cut_1 = (il_scab_0(outcome_5, _S179, h_4, _S180, pa_1)) != (il_scab_0(outcome_5, _S179, h_4, _S180, pb_1));
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        cut_1 = il_crater_0(outcome_5, _S179, s1_2.x, s1_2.z, pa_1);
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        cut_1 = il_crater_0(outcome_5, _S179, s1_2.x, s1_2.z, pb_1);
    }
    if(cut_1)
    {
        work_0[w_alive_0() + i_4] = u32(0);
        var _S181 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_34]), u32(1));
        var _S182 : u32 = a_counters_0();
        var _S183 : u32 = atomicAdd(&(atom_0[_S182]), u32(1));
        var _S184 : u32 = atomicAdd(&(atom_0[_S182 + u32(1)]), u32(1));
        event_broken_0(i_4);
    }
    return;
}

fn il_carry_1( k_18 : u32)
{
    if(k_18 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S185 : u32 = w_carry_0() + k_18;
    var c_4 : u32 = work_0[_S185];
    if(work_0[_S185] == u32(0))
    {
        return;
    }
    work_0[_S185] = u32(0);
    var b0_1 : u32 = c_4 - u32(1);
    if(!il_struck_1(b0_1))
    {
        return;
    }
    var contact_14 : vec3<f32>;
    var d_16 : vec3<f32>;
    var speed_12 : f32;
    il_path_3(b0_1, &(contact_14), &(d_16), &(speed_12));
    var dr_1 : vec3<f32> = to_rubble_0(d_16);
    var sk_3 : IlStriker_0 = il_striker_1(b0_1);
    var plug_mass_1 : f32 = read64_0(a_il_0(b0_1) + u32(2), params_0.scale_m_0);
    var v_wall_1 : vec3<f32> = bodies_0[u32(7) * b0_1 + u32(3)].xyz;
    var _S186 : u32 = u32(7) * sk_3.body_0;
    var v_shot_1 : vec3<f32> = bodies_0[_S186 + u32(3)].xyz;
    var closing_1 : f32 = dot(v_shot_1 - v_wall_1, dr_1);
    var v_r_1 : f32 = sk_3.mass_0 * closing_1 / (sk_3.mass_0 + plug_mass_1);
    var _S187 : vec3<f32> = v_shot_1 + dr_1 * vec3<f32>((v_r_1 - closing_1));
    var _S188 : vec3<f32> = v_wall_1 + dr_1 * vec3<f32>(v_r_1);
    var _S189 : f32 = il_t_in_1(b0_1);
    var _S190 : vec3<f32> = bodies_0[_S186].xyz + dr_1 * vec3<f32>(((speed_12 - v_r_1) * clamp(_S189, 0.0f, params_0.dt_0)));
    var _S191 : u32 = body_of_0(k_18);
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
        var s_at_2 : u32 = u32(7) * (sk_3.body_0 + region_3 * params_0.max_bodies_0);
        bodies_0[s_at_2].x = _S190.x;
        bodies_0[s_at_2].y = _S190.y;
        bodies_0[s_at_2].z = _S190.z;
        bodies_0[s_at_2 + u32(3)].x = _S187.x;
        bodies_0[s_at_2 + u32(3)].y = _S187.y;
        bodies_0[s_at_2 + u32(3)].z = _S187.z;
        bodies_0[u32(7) * (_S191 + region_3 * params_0.max_bodies_0) + u32(3)].x = _S188.x;
        bodies_0[u32(7) * (_S191 + region_3 * params_0.max_bodies_0) + u32(3)].y = _S188.y;
        bodies_0[u32(7) * (_S191 + region_3 * params_0.max_bodies_0) + u32(3)].z = _S188.z;
        region_3 = region_3 + u32(1);
    }
    return;
}

fn il_mark_1( k_19 : u32)
{
    if(k_19 >= (params_0.chunk_count_0))
    {
        return;
    }
    var b_35 : u32 = body_of_0(k_19);
    if(!il_struck_1(b_35))
    {
        return;
    }
    var _S192 : u32 = lp_body_0(b_35);
    var s0_3 : vec4<f32> = lp_0[_S192];
    var s1_3 : vec4<f32> = lp_0[_S192 + u32(1)];
    var outcome_6 : u32 = (bitcast<u32>((lp_0[_S192].w)));
    if(((outcome_6 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_15 : vec3<f32>;
    var d_17 : vec3<f32>;
    var speed_13 : f32;
    il_path_3(b_35, &(contact_15), &(d_17), &(speed_13));
    var entry_6 : u32 = il_entry_1(b_35);
    var sk_4 : IlStriker_0 = il_striker_1(b_35);
    var _S193 : f32 = max(sk_4.radius_0, 0.00100000004749745f);
    var _S194 : f32 = s0_3.x;
    var p_11 : IlPlace_0 = il_place_0(k_19, entry_6, contact_15, d_17, _S193, _S194);
    var plug_2 : bool = il_plug_0(outcome_6, _S193, _S194, p_11);
    if((sk_4.launch_0) != u32(0))
    {
        var t_6 : vec2<f32> = il_interval_0(k_19, contact_15 - d_17 * vec3<f32>(sk_4.bound_0), d_17, sk_4.bound_0);
        var _S195 : f32 = t_6.x;
        var _S196 : f32 = t_6.y;
        var path_1 : bool;
        if(_S195 <= _S196)
        {
            path_1 = _S195 <= (_S194 + 2.0f * sk_4.bound_0);
        }
        else
        {
            path_1 = false;
        }
        if(path_1)
        {
            path_1 = _S196 >= 0.0f;
        }
        else
        {
            path_1 = false;
        }
        if(plug_2)
        {
            path_1 = true;
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            path_1 = il_scab_0(outcome_6, _S193, _S194, s1_3.y, p_11);
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            path_1 = il_crater_0(outcome_6, _S193, s1_3.x, s1_3.z, p_11);
        }
        if(path_1)
        {
            work_0[w_marks_0() + k_19] = sk_4.launch_0;
        }
    }
    if(plug_2)
    {
        work_0[w_carry_0() + k_19] = b_35 + u32(1);
        add64_0(a_il_0(b_35) + u32(2), chunks_0[u32(4) * k_19].w, params_0.scale_m_0);
    }
    return;
}

fn dirty_0( body_1 : u32) -> bool
{
    var _S197 : u32 = atomicLoad(&(atom_0[a_dirty_0() + body_1]));
    return _S197 != u32(0);
}

fn w_cs_list_0() -> u32
{
    return w_carry_0() + params_0.chunk_count_0;
}

fn w_cs_bodies_0() -> u32
{
    return w_cs_list_0() + params_0.chunk_count_0;
}

fn gi_record_0( b_36 : u32,  entry_7 : u32,  contact_16 : vec3<f32>,  d_18 : vec3<f32>,  v_9 : f32,  j_2 : f32,  tau_1 : f32,  local_0 : f32)
{
    var _S198 : bool = dirty_0(b_36);
    if(_S198)
    {
        return;
    }
    var _S199 : u32 = lp_body_0(b_36);
    lp_0[_S199 + u32(2)] = vec4<f32>(contact_16, 3.14159274101257324f * j_2 / (2.0f * max(tau_1, 9.99999971718068537e-10f)));
    lp_0[_S199 + u32(3)] = vec4<f32>(d_18, v_9);
    lp_0[_S199 + u32(4)] = vec4<f32>(local_0, 0.0f, 0.0f, 0.0f);
    lp_0[_S199 + u32(6)] = vec4<f32>(0.0f, 0.0f, tau_1, 0.0f);
    var _S200 : u32 = a_cs_0(b_36);
    atomicStore(&(atom_0[_S200 + u32(1)]), entry_7 + u32(1));
    var _S201 : u32 = atomicExchange(&(atom_0[_S200 + u32(8)]), u32(1));
    if(_S201 == u32(0))
    {
        var _S202 : u32 = w_cs_bodies_0();
        var _S203 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(18)]), u32(1));
        work_0[_S202 + _S203] = b_36;
    }
    return;
}

fn il_strike_1( b_37 : u32)
{
    var damage_1 : bool;
    if(b_37 >= (params_0.body_count_0))
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = !il_struck_1(b_37);
    }
    if(damage_1)
    {
        return;
    }
    var t_in_6 : f32 = il_t_in_1(b_37);
    var entry_8 : u32 = il_entry_1(b_37);
    var contact_17 : vec3<f32>;
    var d_19 : vec3<f32>;
    var speed_14 : f32;
    il_path_3(b_37, &(contact_17), &(d_19), &(speed_14));
    var sk_5 : IlStriker_0 = il_striker_1(b_37);
    var _S204 : f32 = max(sk_5.radius_0, 0.00100000004749745f);
    if((sk_5.launch_0) != u32(0))
    {
        damage_1 = work_0[w_marks_0() + entry_8] == (sk_5.launch_0);
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        var _S205 : u32 = lp_body_0(b_37);
        var _S206 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S205] = _S206;
        lp_0[_S205 + u32(1)] = _S206;
        return;
    }
    var _S207 : u32 = atomicLoad(&(atom_0[a_il_0(b_37)]));
    var _S208 : f32 = max((bitcast<f32>((_S207))) - _S204, 9.99999997475242708e-07f);
    if(!(t_in_6 > 0.0f))
    {
        var _S209 : u32 = lp_body_0(b_37);
        lp_0[_S209] = vec4<f32>(_S208, 0.0f, 0.0f, 0.0f);
        lp_0[_S209 + u32(1)] = vec4<f32>(0.0f);
        return;
    }
    var o_3 : u32 = params_0.ext_chunks_0 + u32(4) * entry_8;
    var mat0_1 : vec4<f32> = ext_0[o_3 + u32(1)];
    var dm_1 : JointMaterial_0;
    var _S210 : vec4<f32> = vec4<f32>(0.0f);
    dm_1.strength_0 = _S210;
    dm_1.energy_0 = _S210;
    dm_1.dif_0 = ext_0[o_3 + u32(2)];
    var mat3_1 : vec4<f32> = ext_0[o_3 + u32(3)];
    var _S211 : f32 = mat3_1.x;
    dm_1.misc_0 = vec4<f32>(_S211, 0.0f, 0.0f, 0.0f);
    dm_1.kind_flags_0 = vec4<u32>(u32(0));
    var nu_1 : f32 = mat3_1.y;
    var e_target_1 : f32 = mat3_1.z / (1.0f - nu_1 * nu_1);
    var e_star_1 : f32 = 1.0f / (1.0f / sk_5.modulus_0 + 1.0f / e_target_1);
    var _S212 : f32 = max(speed_14, 0.00100000004749745f);
    var _S213 : f32 = sqrt(_S204);
    var h0_1 : f32 = pow(15.0f * sk_5.m_eff_0 * _S212 * _S212 / (16.0f * e_star_1 * _S213), 0.40000000596046448f);
    var t_c_1 : f32 = 2.94000005722045898f * h0_1 / _S212;
    var force_1 : f32 = 1.33333337306976318f * e_star_1 * _S213 * pow(h0_1, 1.5f);
    var a_6 : f32 = sqrt(_S204 * h0_1);
    var p0_1 : f32 = 3.0f * force_1 / (6.28318548202514648f * a_6 * a_6);
    var ring_1 : f32 = (1.0f - 2.0f * nu_1) * p0_1 / 3.0f;
    var k_20 : f32;
    if(_S211 > 0.0f)
    {
        k_20 = dif_factor_0(dm_1, _S212 / _S208);
    }
    else
    {
        k_20 = 1.0f;
    }
    var ft_1 : f32 = k_20 * mat0_1.y;
    var fc_1 : f32 = k_20 * mat0_1.z;
    var tau_2 : f32 = k_20 * mat0_1.w;
    if(p0_1 > fc_1)
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = ring_1 > ft_1;
    }
    var _S214 : f32 = fc_1 * (3.14159274101257324f * _S204 * _S204);
    var _S215 : f32 = min(_S214 * _S208, 1.57079637050628662f * (2.0f * _S204 + _S208) * _S208 * _S208 * tau_2);
    var _S216 : f32 = 2.0f * _S215;
    var v_bl_1 : f32 = sqrt(_S216 / sk_5.m_eff_0);
    var perforate_1 : bool;
    if(damage_1)
    {
        perforate_1 = _S212 >= v_bl_1;
    }
    else
    {
        perforate_1 = false;
    }
    var r2_1 : f32 = _S204 + _S208;
    var _S217 : f32 = mat0_1.x;
    var plug_3 : f32 = _S217 * 3.14159274101257324f * _S208 / 3.0f * (_S204 * _S204 + _S204 * r2_1 + r2_1 * r2_1);
    var v_res_1 : f32;
    if(perforate_1)
    {
        v_res_1 = sk_5.mass_0 / (sk_5.mass_0 + plug_3) * sqrt(max(_S212 * _S212 - v_bl_1 * v_bl_1, 0.0f));
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
        depth_2 = min(0.5f * sk_5.m_eff_0 * _S212 * _S212 / _S214, _S208);
    }
    else
    {
        depth_2 = 0.0f;
    }
    var crater_1 : bool = depth_2 > a_6;
    var c_5 : f32 = sqrt(e_target_1 / _S217);
    var z_t_1 : f32 = _S217 * c_5;
    var rho_p_1 : f32 = sk_5.mass_0 / (4.18879032135009766f * _S204 * _S204 * _S204);
    var z_p_1 : f32 = rho_p_1 * sqrt(sk_5.modulus_0 / rho_p_1);
    var mean_pressure_1 : f32 = force_1 / (3.14159274101257324f * a_6 * a_6);
    var _S218 : f32 = min(min(z_t_1 * _S212 * z_p_1 / (z_p_1 + z_t_1), mean_pressure_1), fc_1);
    var source_radius_1 : f32;
    if(mean_pressure_1 > fc_1)
    {
        source_radius_1 = max(a_6, _S204);
    }
    else
    {
        source_radius_1 = a_6;
    }
    var pulse_1 : f32 = c_5 * t_c_1;
    var back_1 : f32 = _S218 * min(source_radius_1 / _S208, 1.0f) * min(2.0f * _S208 / pulse_1, 1.0f);
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
        scab_2 = 0.5f * min(pulse_1, _S208);
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
    var _S219 : u32;
    if(crater_1)
    {
        _S219 = u32(2);
    }
    else
    {
        _S219 = u32(0);
    }
    var _S220 : u32 = (region_4 | (_S219));
    if(spall_1)
    {
        region_4 = u32(4);
    }
    else
    {
        region_4 = u32(0);
    }
    var _S221 : u32 = (_S220 | (region_4));
    if(perforate_1)
    {
        region_4 = u32(8);
    }
    else
    {
        region_4 = u32(0);
    }
    var _S222 : u32 = lp_body_0(b_37);
    lp_0[_S222] = vec4<f32>(_S208, v_bl_1, v_res_1, (bitcast<f32>(((_S221 | (region_4))))));
    if(crater_1)
    {
        k_20 = depth_2;
    }
    else
    {
        k_20 = 0.0f;
    }
    lp_0[_S222 + u32(1)] = vec4<f32>(k_20, scab_2, a_6, _S215);
    if(perforate_1)
    {
        var _S223 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(14)]), u32(1));
    }
    else
    {
        if(spall_1)
        {
            var _S224 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(13)]), u32(1));
        }
        else
        {
            if(crater_1)
            {
                var _S225 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(12)]), u32(1));
            }
            else
            {
                var _S226 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(11)]), u32(1));
            }
        }
    }
    if(perforate_1)
    {
        if((params_0.fracture_work_0) != u32(0))
        {
            il_impulse_0(b_37, sk_5.body_0, sk_5.mass_0, contact_17, d_19, sk_5.m_eff_0 * (_S212 - sqrt(max(_S212 * _S212 - _S216 / sk_5.m_eff_0, 0.0f))));
        }
    }
    else
    {
        if(damage_1)
        {
            il_impulse_0(b_37, sk_5.body_0, sk_5.mass_0, contact_17, d_19, sk_5.m_eff_0 * _S212);
            var _S227 : vec3<f32> = to_rubble_0(d_19) * vec3<f32>((_S212 * clamp(t_in_6, 0.0f, params_0.dt_0)));
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
                var _S228 : u32 = u32(7) * (sk_5.body_0 + region_4 * params_0.max_bodies_0);
                var _S229 : vec3<f32> = bodies_0[_S228].xyz + _S227;
                bodies_0[_S228].x = _S229.x;
                bodies_0[_S228].y = _S229.y;
                bodies_0[_S228].z = _S229.z;
                region_4 = region_4 + u32(1);
            }
        }
    }
    if((params_0.gi_mode_0) != u32(0))
    {
        if(damage_1)
        {
            damage_1 = true;
        }
        else
        {
            damage_1 = (params_0.cs_sweeps_0) == u32(0);
        }
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        var _S230 : vec3<f32> = contact_17;
        var _S231 : vec3<f32> = d_19;
        if(perforate_1)
        {
            k_20 = sqrt(max(_S212 * _S212 - v_bl_1 * v_bl_1, 0.0f));
        }
        else
        {
            k_20 = 0.0f;
        }
        var _S232 : f32 = sk_5.m_eff_0 * (_S212 - k_20);
        if(perforate_1)
        {
            k_20 = _S208;
        }
        else
        {
            k_20 = depth_2;
        }
        gi_record_0(b_37, entry_8, _S230, _S231, _S212, _S232, max(t_c_1, 2.0f * k_20 / max(_S212 + v_res_1, 9.99999971718068537e-10f)), _S208);
    }
    return;
}

fn il_run_1( b_38 : u32,  i_5 : u32)
{
    var j_3 : u32;
    if(il_struck_1(b_38))
    {
        j_3 = u32(1);
    }
    else
    {
        j_3 = u32(0);
    }
    var _S233 : u32 = group_uniform_0(j_3, i_5);
    if(_S233 == u32(0))
    {
        return;
    }
    var _S234 : f32 = 0.00100000004749745f + 0.01999999955296516f * il_striker_1(b_38).radius_0;
    var _S235 : u32 = a_il_0(b_38);
    var _S236 : u32 = atomicLoad(&(atom_0[_S235 + u32(4)]));
    var _S237 : u32 = atomicLoad(&(atom_0[_S235 + u32(1)]));
    var _S238 : u32 = atomicLoad(&(atom_0[_S235]));
    var _S239 : f32 = group_uniform_float_0((bitcast<f32>((_S238))), i_5);
    var end_1 : f32 = _S239;
    loop
    {
        var best_1 : f32 = end_1;
        j_3 = i_5;
        loop
        {
            if(j_3 < _S237)
            {
            }
            else
            {
                break;
            }
            var at_10 : u32 = w_il_tube_0() + u32(2) * (_S236 + j_3);
            if((bitcast<f32>((work_0[at_10]))) <= (end_1 + _S234))
            {
                best_1 = max(best_1, (bitcast<f32>((work_0[at_10 + u32(1)]))));
            }
            j_3 = j_3 + u32(64);
        }
        g_il_best_0[i_5] = best_1;
        workgroupBarrier();
        var step_2 : u32 = u32(32);
        loop
        {
            if(step_2 > u32(0))
            {
            }
            else
            {
                break;
            }
            if(i_5 < step_2)
            {
                g_il_best_0[i_5] = max(g_il_best_0[i_5], g_il_best_0[i_5 + step_2]);
            }
            workgroupBarrier();
            step_2 = (step_2 >> (u32(1)));
        }
        var next_1 : f32 = group_uniform_float_0(g_il_best_0[i32(0)], i_5);
        workgroupBarrier();
        if(!(next_1 > end_1))
        {
            break;
        }
        end_1 = next_1;
    }
    if(i_5 == u32(0))
    {
        atomicStore(&(atom_0[_S235]), (bitcast<u32>((end_1))));
    }
    return;
}

fn il_tube_collect_1( k_21 : u32)
{
    var b_39 : u32;
    var t_7 : vec2<f32>;
    var _S240 : bool = il_tube_1(k_21, &(b_39), &(t_7));
    if(!_S240)
    {
        return;
    }
    var _S241 : u32 = w_il_tube_0();
    var _S242 : u32 = atomicLoad(&(atom_0[a_il_0(b_39) + u32(4)]));
    var _S243 : u32 = atomicAdd(&(atom_0[a_il_0(b_39) + u32(5)]), u32(1));
    var at_11 : u32 = _S241 + u32(2) * (_S242 + _S243);
    work_0[at_11] = (bitcast<u32>((t_7.x)));
    work_0[at_11 + u32(1)] = (bitcast<u32>((t_7.y)));
    return;
}

fn il_tube_base_1( b_40 : u32)
{
    var _S244 : bool;
    if(b_40 < (params_0.body_count_0))
    {
        _S244 = il_struck_1(b_40);
    }
    else
    {
        _S244 = false;
    }
    if(_S244)
    {
        var _S245 : u32 = a_il_0(b_40);
        var _S246 : u32 = atomicLoad(&(atom_0[_S245 + u32(1)]));
        var _S247 : u32 = atomicAdd(&(atom_0[a_il_cursor_0()]), _S246);
        atomicStore(&(atom_0[_S245 + u32(4)]), _S247);
    }
    return;
}

fn il_tube_count_1( k_22 : u32)
{
    var b_41 : u32;
    var t_8 : vec2<f32>;
    var _S248 : bool = il_tube_1(k_22, &(b_41), &(t_8));
    if(_S248)
    {
        var _S249 : u32 = atomicAdd(&(atom_0[a_il_0(b_41) + u32(1)]), u32(1));
    }
    return;
}

fn struck_1( b_42 : u32,  shot_3 : ptr<function, u32>) -> bool
{
    var k_23 : u32 = u32(frames_0[strike_at_0(b_42)].w);
    (*shot_3) = k_23 - u32(1);
    return k_23 != u32(0);
}

fn il_begin_1( b_43 : u32)
{
    var _S250 : bool;
    if(b_43 >= (params_0.body_count_0))
    {
        _S250 = true;
    }
    else
    {
        _S250 = !il_struck_1(b_43);
    }
    if(_S250)
    {
        return;
    }
    var shot_4 : u32;
    var _S251 : bool = struck_1(b_43, &(shot_4));
    var _S252 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_43 + u32(6)]));
    var _S253 : u32 = atomicLoad(&(atom_0[a_first_0(shot_4)]));
    if(_S252 != _S253)
    {
        frames_0[strike_at_0(b_43)][i32(3)] = 0.0f;
        return;
    }
    var contact_18 : vec3<f32>;
    var d_20 : vec3<f32>;
    var speed_15 : f32;
    il_path_3(b_43, &(contact_18), &(d_20), &(speed_15));
    var _S254 : u32 = il_entry_1(b_43);
    var _S255 : u32 = a_il_0(b_43);
    atomicStore(&(atom_0[_S255]), (bitcast<u32>((max(il_interval_0(_S254, contact_18, d_20, il_striker_1(b_43).radius_0).y, 0.0f)))));
    var i_6 : u32 = u32(1);
    loop
    {
        if(i_6 < u32(6))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[_S255 + i_6]), u32(0));
        i_6 = i_6 + u32(1);
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_events_reset(@builtin(global_invocation_id) id_4 : vec3<u32>)
{
    var _S256 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S256 + u32(20)]), u32(0));
    atomicStore(&(atom_0[_S256 + u32(21)]), u32(0));
    atomicStore(&(atom_0[_S256 + u32(22)]), u32(0));
    return;
}

fn live_0( c_6 : u32) -> bool
{
    var _S257 : bool;
    if(c_6 < (params_0.chunk_used_0))
    {
        _S257 = (body_of_0(c_6)) != u32(4294967295);
    }
    else
    {
        _S257 = false;
    }
    return _S257;
}

fn structure_pose_2( slot_5 : u32,  b_44 : u32,  q_13 : ptr<function, Quat_0>,  origin_5 : ptr<function, vec3<f32>>)
{
    var _S258 : u32 = u32(2) * slot_5;
    var com_2 : vec3<f32> = frames_0[_S258].xyz;
    var _S259 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_44 + u32(1)])), conj_0(quat_of_0(frames_0[_S258 + u32(1)]))));
    (*q_13) = _S259;
    (*origin_5) = from_rubble_0(bodies_0[b_44].xyz) - rotate_0(_S259, com_2);
    return;
}

fn structure_pose_3( slot_6 : u32,  b_45 : u32,  q_14 : ptr<function, Quat_0>,  origin_6 : ptr<function, vec3<f32>>)
{
    var _S260 : u32 = u32(2) * slot_6;
    var com_3 : vec3<f32> = frames_0[_S260].xyz;
    var _S261 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_45 + u32(1)])), conj_0(quat_of_0(frames_0[_S260 + u32(1)]))));
    (*q_14) = _S261;
    (*origin_6) = from_rubble_0(bodies_0[b_45].xyz) - rotate_0(_S261, com_3);
    return;
}

fn structure_pose_4( slot_7 : u32,  b_46 : u32,  q_15 : ptr<function, Quat_0>,  origin_7 : ptr<function, vec3<f32>>)
{
    var _S262 : u32 = u32(2) * slot_7;
    var com_4 : vec3<f32> = frames_0[_S262].xyz;
    var _S263 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_46 + u32(1)])), conj_0(quat_of_0(frames_0[_S262 + u32(1)]))));
    (*q_15) = _S263;
    (*origin_7) = from_rubble_0(bodies_0[b_46].xyz) - rotate_0(_S263, com_4);
    return;
}

fn structure_pose_5( slot_8 : u32,  b_47 : u32,  q_16 : ptr<function, Quat_0>,  origin_8 : ptr<function, vec3<f32>>)
{
    var _S264 : u32 = u32(2) * slot_8;
    var com_5 : vec3<f32> = frames_0[_S264].xyz;
    var _S265 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_47 + u32(1)])), conj_0(quat_of_0(frames_0[_S264 + u32(1)]))));
    (*q_16) = _S265;
    (*origin_8) = from_rubble_0(bodies_0[b_47].xyz) - rotate_0(_S265, com_5);
    return;
}

fn w_frame_0() -> u32
{
    return w_cs_rows_0() + u32(1);
}

fn frame_parity_0() -> u32
{
    return work_0[w_frame_0()];
}

fn l_pool_0( half_1 : u32) -> u32
{
    return params_0.chunk_count_0 + u32(3) * params_0.pool_cap_0 * half_1;
}

fn pulse_mean_0( start_0 : f32,  plateau_0 : f32,  plateau_time_0 : f32,  peak_0 : f32,  duration_0 : f32,  dt_1 : f32) -> f32
{
    var _S266 : f32 = start_0 + plateau_time_0;
    var sum_0 : f32 = plateau_0 * max(min(_S266, dt_1) - max(start_0, 0.0f), 0.0f);
    var _S267 : f32 = max(0.0f, _S266);
    var _S268 : f32 = min(dt_1, _S266 + duration_0);
    var _S269 : bool;
    if(_S268 > _S267)
    {
        _S269 = duration_0 > 0.0f;
    }
    else
    {
        _S269 = false;
    }
    var sum_1 : f32;
    if(_S269)
    {
        var w_2 : f32 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S267 - _S266)) - cos(w_2 * (_S268 - _S266)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

fn hertz_duration_0( m_0 : f32,  r_17 : f32,  e_2 : f32,  v_10 : f32) -> f32
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_17 * e_2 * e_2 * max(abs(v_10), 0.00100000004749745f)), 0.20000000298023224f);
}

fn l_vec_0() -> u32
{
    return params_0.chunk_count_0 + u32(6) * params_0.pool_cap_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_loads(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var f_total_0 : vec3<f32>;
    var t_total_0 : vec3<f32>;
    var m_1 : f32;
    var p_12 : u32;
    var kept_0 : u32;
    var q3_0 : u32;
    var overflow_0 : bool;
    var overflow_1 : bool;
    var f_total_1 : vec3<f32>;
    var t_total_1 : vec3<f32>;
    var k_24 : u32 = id_5.x;
    if(!live_0(k_24))
    {
        return;
    }
    var slot_9 : u32 = body_of_0(k_24);
    var b_48 : u32 = u32(7) * slot_9;
    var q_17 : Quat_0;
    var origin_9 : vec3<f32>;
    structure_pose_5(slot_9, b_48, &(q_17), &(origin_9));
    var _S270 : u32 = u32(4) * k_24;
    var c0_0 : vec4<f32> = chunks_0[_S270];
    var centre_0 : vec3<f32> = origin_9 + rotate_0(q_17, c0_0.xyz);
    var inv_mass_2 : f32 = bodies_0[b_48].w;
    var m_body_0 : f32;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(u32(2139095040));
    }
    var from_0 : u32 = frame_parity_0();
    var to_0 : u32 = u32(1) - from_0;
    var _S271 : u32 = w_pulses_0() + _S270;
    var _S272 : u32 = _S271 + u32(2) * from_0;
    var off_0 : u32 = work_0[_S272];
    var carried_0 : u32 = work_0[_S272 + u32(1)];
    var room_0 : u32 = work_0[_S272 + u32(1)] + sort_0[params_0.chunk_count_0 + u32(1) + k_24];
    var base_0 : u32;
    var room_1 : u32;
    if(room_0 != u32(0))
    {
        var _S273 : u32 = a_counters_0();
        var base_1 : u32 = atomicAdd(&(atom_0[_S273 + u32(5) + to_0]), room_0);
        if((base_1 + room_0) > (params_0.pool_cap_0))
        {
            var _S274 : u32 = atomicOr(&(atom_0[_S273 + u32(3)]), u32(8));
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
        var _S275 : u32 = base_0;
        base_0 = base_1;
        room_1 = _S275;
    }
    else
    {
        base_0 = u32(0);
        room_1 = room_0;
    }
    var _S276 : u32 = params_0.chunk_count_0;
    var _S277 : f32 = c0_0.w;
    var _S278 : vec4<f32> = chunks_0[_S270 + u32(3)];
    var _S279 : f32 = params_0.band_0;
    var _S280 : f32 = params_0.dt_0;
    var _S281 : bool = (params_0.impact_peak_0) != u32(0);
    var _S282 : u32 = l_pool_0(from_0) + u32(3) * off_0;
    var _S283 : u32 = l_pool_0(to_0) + u32(3) * base_0;
    loop
    {
        var _S284 : vec3<f32> = vec3<f32>(0.0f);
        var _S285 : u32 = sort_0[k_24];
        var _S286 : u32 = sort_0[_S276 + u32(1) + k_24];
        var first_0 : bool = true;
        var last_0 : u32 = u32(0);
        var step_3 : u32 = u32(0);
        f_total_0 = _S284;
        t_total_0 = _S284;
        loop
        {
            if(step_3 < _S286)
            {
            }
            else
            {
                break;
            }
            p_12 = u32(4294967295);
            kept_0 = u32(0);
            q3_0 = u32(0);
            loop
            {
                if(q3_0 < _S286)
                {
                }
                else
                {
                    break;
                }
                var key_1 : u32 = manifolds_0[_S285 + q3_0].ids_0.y + u32(1);
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
                    if(p_12 == u32(4294967295))
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
                    p_12 = q3_0;
                    kept_0 = key_1;
                }
                q3_0 = q3_0 + u32(1);
            }
            if(p_12 == u32(4294967295))
            {
                break;
            }
            var r_18 : Record_std430_0 = manifolds_0[_S285 + p_12];
            var _S287 : vec3<f32> = r_18.impulse_0.xyz;
            var j_4 : f32 = length(_S287);
            if(j_4 == 0.0f)
            {
                var _S288 : u32 = step_3 + u32(1);
                first_0 = false;
                last_0 = kept_0;
                step_3 = _S288;
                continue;
            }
            var m_other_0 : f32 = r_18.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_0))
                {
                    m_1 = _S277;
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
            var _S289 : f32 = min(j_4, 2.0f * m_1 * max(r_18.impulse_0.w - _S279, 0.0f));
            if(_S289 < j_4)
            {
                var f_1 : vec3<f32> = _S287 / vec3<f32>(j_4) * vec3<f32>(((j_4 - _S289) / _S280));
                var _S290 : vec3<f32> = t_total_0 + cross(r_18.point_0.xyz - centre_0, f_1);
                f_total_1 = f_total_0 + f_1;
                t_total_1 = _S290;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            var _S288 : u32 = step_3 + u32(1);
            first_0 = false;
            last_0 = kept_0;
            step_3 = _S288;
        }
        p_12 = u32(0);
        kept_0 = u32(0);
        loop
        {
            if(p_12 < carried_0)
            {
            }
            else
            {
                break;
            }
            var _S291 : u32 = _S282 + u32(3) * p_12;
            var a_7 : vec4<f32> = loads_0[_S291];
            var b_49 : vec4<f32> = loads_0[_S291 + u32(1)];
            var c_7 : vec4<f32> = loads_0[_S291 + u32(2)];
            var _S292 : f32 = loads_0[_S291 + u32(1)].w;
            var mean_0 : f32 = pulse_mean_0(loads_0[_S291].x, loads_0[_S291].y, loads_0[_S291].z, loads_0[_S291].w, _S292, _S280);
            if(mean_0 != 0.0f)
            {
                var f_2 : vec3<f32> = c_7.xyz * vec3<f32>(mean_0);
                var _S293 : vec3<f32> = t_total_0 + cross(b_49.xyz - centre_0, f_2);
                f_total_0 = f_total_0 + f_2;
                t_total_0 = _S293;
            }
            if((a_7.x + a_7.z + _S292) > _S280)
            {
                a_7[i32(0)] = a_7[i32(0)] - _S280;
                var _S294 : u32 = _S283 + u32(3) * kept_0;
                loads_0[_S294] = a_7;
                loads_0[_S294 + u32(1)] = b_49;
                loads_0[_S294 + u32(2)] = c_7;
                kept_0 = kept_0 + u32(1);
            }
            p_12 = p_12 + u32(1);
        }
        q3_0 = u32(0);
        overflow_0 = false;
        loop
        {
            if(q3_0 < _S286)
            {
            }
            else
            {
                break;
            }
            var r_19 : Record_std430_0 = manifolds_0[_S285 + q3_0];
            var _S295 : vec3<f32> = r_19.impulse_0.xyz;
            var j_5 : f32 = length(_S295);
            if(j_5 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var m_other_1 : f32 = r_19.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_1))
                {
                    m_1 = _S277;
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
            var _S296 : f32 = r_19.impulse_0.w;
            var _S297 : f32 = min(j_5, 2.0f * m_1 * max(_S296 - _S279, 0.0f));
            if(_S297 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var _S298 : f32 = _S278.y;
            var duration_1 : f32 = hertz_duration_0(m_1, min(r_19.other_0.z, _S278.z), 1.0f / ((1.0f - _S298 * _S298) / _S278.x + 1.0f / r_19.other_0.y), _S296);
            var peak_1 : f32 = 3.14159274101257324f * _S297 / (2.0f * duration_1);
            var dir_0 : vec3<f32> = _S295 / vec3<f32>(j_5);
            var mean_1 : f32;
            if(_S281)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S280);
            }
            if(mean_1 != 0.0f)
            {
                var f_3 : vec3<f32> = dir_0 * vec3<f32>(mean_1);
                var _S299 : vec3<f32> = t_total_0 + cross(r_19.point_0.xyz - centre_0, f_3);
                f_total_1 = f_total_0 + f_3;
                t_total_1 = _S299;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            var kept_1 : u32;
            if(duration_1 > _S280)
            {
                if(kept_0 < room_1)
                {
                    var _S300 : u32 = _S283 + u32(3) * kept_0;
                    loads_0[_S300] = vec4<f32>(- _S280, 0.0f, 0.0f, peak_1);
                    loads_0[_S300 + u32(1)] = vec4<f32>(r_19.point_0.xyz, duration_1);
                    loads_0[_S300 + u32(2)] = vec4<f32>(dir_0, 0.0f);
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
        var _S301 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(8));
    }
    var _S302 : u32 = _S271 + u32(2) * to_0;
    work_0[_S302] = base_0;
    work_0[_S302 + u32(1)] = kept_0;
    loads_0[k_24] = vec4<f32>(length(f_total_0), f_total_0);
    if((params_0.load_model_0) == u32(1))
    {
        var _S303 : u32 = l_vec_0() + u32(2) * k_24;
        loads_0[_S303] = vec4<f32>(f_total_0, 0.0f);
        loads_0[_S303 + u32(1)] = vec4<f32>(t_total_0, 0.0f);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_breaks(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var i_7 : u32 = id_6.x;
    var _S304 : bool;
    if(i_7 >= (params_0.bond_used_0))
    {
        _S304 = true;
    }
    else
    {
        _S304 = work_0[w_alive_0() + i_7] == u32(0);
    }
    if(_S304)
    {
        return;
    }
    var bond_3 : vec4<u32> = bonds_0[i_7];
    var _S305 : u32 = bond_3.x;
    if((max(loads_0[_S305].x, loads_0[bond_3.y].x)) > (bitcast<f32>((bond_3.z))))
    {
        work_0[w_alive_0() + i_7] = u32(0);
        var _S306 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S305)]), u32(1));
        var _S307 : u32 = a_counters_0();
        var _S308 : u32 = atomicAdd(&(atom_0[_S307]), u32(1));
        var _S309 : u32 = atomicAdd(&(atom_0[_S307 + u32(1)]), u32(1));
        event_broken_0(i_7);
    }
    return;
}

fn a_moved_0() -> u32
{
    return a_events_0() + params_0.ev_broken_cap_0;
}

fn a_view_0() -> u32
{
    return a_moved_0() + u32(2) * params_0.ev_moved_cap_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_stress_clear(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var _S310 : u32 = id_7.x;
    if(_S310 < (params_0.chunk_count_0))
    {
        atomicStore(&(atom_0[a_view_0() + _S310]), u32(0));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_stress_breaks(@builtin(global_invocation_id) id_8 : vec3<u32>)
{
    var i_8 : u32 = id_8.x;
    var _S311 : bool;
    if(i_8 >= (params_0.bond_used_0))
    {
        _S311 = true;
    }
    else
    {
        _S311 = work_0[w_alive_0() + i_8] == u32(0);
    }
    if(_S311)
    {
        return;
    }
    var bond_4 : vec4<u32> = bonds_0[i_8];
    var capacity_0 : f32 = (bitcast<f32>((bond_4.z)));
    if(capacity_0 <= 0.0f)
    {
        return;
    }
    var _S312 : u32 = bond_4.x;
    var _S313 : u32 = bond_4.y;
    var u_6 : u32 = (bitcast<u32>((max(max(loads_0[_S312].x, loads_0[_S313].x), 0.0f) / capacity_0)));
    var _S314 : u32 = a_view_0();
    var _S315 : u32 = atomicMax(&(atom_0[_S314 + _S312]), u_6);
    var _S316 : u32 = atomicMax(&(atom_0[_S314 + _S313]), u_6);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_host_breaks(@builtin(global_invocation_id) id_9 : vec3<u32>)
{
    var i_9 : u32 = id_9.x;
    var _S317 : u32 = w_host_breaks_0();
    if(i_9 >= work_0[_S317])
    {
        return;
    }
    var b_50 : u32 = work_0[_S317 + u32(1) + i_9];
    var _S318 : bool;
    if(work_0[_S317 + u32(1) + i_9] >= (params_0.bond_count_0))
    {
        _S318 = true;
    }
    else
    {
        _S318 = work_0[w_alive_0() + b_50] == u32(0);
    }
    if(_S318)
    {
        return;
    }
    work_0[w_alive_0() + b_50] = u32(0);
    var _S319 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(bonds_0[b_50].x)]), u32(1));
    var _S320 : u32 = a_counters_0();
    var _S321 : u32 = atomicAdd(&(atom_0[_S320]), u32(1));
    var _S322 : u32 = atomicAdd(&(atom_0[_S320 + u32(1)]), u32(1));
    event_broken_0(b_50);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_frame_begin(@builtin(global_invocation_id) id_10 : vec3<u32>)
{
    var _S323 : u32 = w_frame_0();
    work_0[_S323] = u32(1) - work_0[_S323];
    var _S324 : u32 = _S323 + u32(1);
    work_0[_S324] = work_0[_S324] + u32(1);
    var _S325 : u32 = _S323 + u32(2);
    work_0[_S325] = work_0[_S325] + u32(1);
    return;
}

fn groups_of_0( n_3 : u32,  size_0 : u32) -> u32
{
    var _S326 : u32 = (n_3 + size_0 - u32(1)) / size_0;
    return _S326;
}

fn set_args_0( at_12 : u32,  x_4 : u32)
{
    var _S327 : u32 = u32(3) * at_12;
    args_0[_S327] = x_4;
    args_0[_S327 + u32(1)] = u32(1);
    args_0[_S327 + u32(2)] = u32(1);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_gate(@builtin(global_invocation_id) id_11 : vec3<u32>)
{
    var _S328 : u32 = a_counters_0();
    var _S329 : u32 = atomicLoad(&(atom_0[_S328]));
    var run_0 : bool = _S329 != u32(0);
    atomicStore(&(atom_0[_S328]), u32(0));
    atomicStore(&(atom_0[_S328 + u32(5) + frame_parity_0()]), u32(0));
    if(run_0)
    {
        var _S330 : u32 = atomicAdd(&(atom_0[_S328 + u32(4)]), u32(1));
    }
    var _S331 : u32;
    if(run_0)
    {
        var _S332 : u32 = groups_of_0(params_0.chunk_used_0, u32(64));
        _S331 = _S332;
    }
    else
    {
        _S331 = u32(0);
    }
    set_args_0(u32(0), _S331);
    if(run_0)
    {
        var _S333 : u32 = groups_of_0(params_0.bond_used_0, u32(64));
        _S331 = _S333;
    }
    else
    {
        _S331 = u32(0);
    }
    set_args_0(u32(1), _S331);
    if(run_0)
    {
        var _S334 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S331 = _S334;
    }
    else
    {
        _S331 = u32(0);
    }
    set_args_0(u32(2), _S331);
    if(run_0)
    {
        var _S335 : u32 = groups_of_0(params_0.chunk_used_0, u32(1024));
        _S331 = _S335;
    }
    else
    {
        _S331 = u32(0);
    }
    set_args_0(u32(3), _S331);
    if(run_0)
    {
        var _S336 : u32 = groups_of_0(params_0.body_count_0, u32(1024));
        _S331 = _S336;
    }
    else
    {
        _S331 = u32(0);
    }
    set_args_0(u32(4), _S331);
    var _S337 : i32;
    if(run_0)
    {
        _S337 = i32(1);
    }
    else
    {
        _S337 = i32(0);
    }
    set_args_0(u32(5), u32(_S337));
    return;
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
fn fr_uf_init(@builtin(global_invocation_id) id_12 : vec3<u32>)
{
    var c_8 : u32 = id_12.x;
    var _S338 : bool;
    if(!live_0(c_8))
    {
        _S338 = true;
    }
    else
    {
        var _S339 : bool = dirty_0(body_of_0(c_8));
        _S338 = !_S339;
    }
    if(_S338)
    {
        return;
    }
    atomicStore(&(atom_0[a_parent_0() + c_8]), c_8);
    atomicStore(&(atom_0[a_count_0() + c_8]), u32(0));
    atomicStore(&(atom_0[a_anchor_0() + c_8]), u32(0));
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
        atomicStore(&(atom_0[a_sums_0() + u32(20) * c_8 + w_3]), u32(0));
        w_3 = w_3 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_body_init(@builtin(global_invocation_id) id_13 : vec3<u32>)
{
    var s_7 : u32 = id_13.x;
    var _S340 : bool;
    if(s_7 >= (params_0.body_count_0))
    {
        _S340 = true;
    }
    else
    {
        var _S341 : bool = dirty_0(s_7);
        _S340 = !_S341;
    }
    if(_S340)
    {
        return;
    }
    atomicStore(&(atom_0[a_best_count_0() + s_7]), u32(0));
    atomicStore(&(atom_0[a_best_root_0() + s_7]), u32(4294967295));
    return;
}

fn find_0( x_5 : u32) -> u32
{
    var _S342 : u32 = a_parent_0();
    var _S343 : u32 = atomicLoad(&(atom_0[_S342 + x_5]));
    var p_13 : u32 = _S343;
    var _S344 : u32 = x_5;
    var guard_0 : u32 = u32(0);
    loop
    {
        var _S345 : bool;
        if(guard_0 < u32(1048576))
        {
            _S345 = p_13 != _S344;
        }
        else
        {
            _S345 = false;
        }
        if(_S345)
        {
        }
        else
        {
            break;
        }
        var gp_0 : u32 = atomicLoad(&(atom_0[_S342 + p_13]));
        if(gp_0 != p_13)
        {
            var _S346 : u32 = atomicCompareExchangeWeak(&(atom_0[_S342 + _S344]), p_13, gp_0).old_value;
        }
        var _S347 : u32 = guard_0 + u32(1);
        _S344 = p_13;
        p_13 = gp_0;
        guard_0 = _S347;
    }
    return _S344;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_union(@builtin(global_invocation_id) id_14 : vec3<u32>)
{
    var i_10 : u32 = id_14.x;
    var _S348 : bool;
    if(i_10 >= (params_0.bond_used_0))
    {
        _S348 = true;
    }
    else
    {
        _S348 = work_0[w_alive_0() + i_10] == u32(0);
    }
    if(_S348)
    {
        return;
    }
    var bond_5 : vec4<u32> = bonds_0[i_10];
    var _S349 : u32 = bond_5.x;
    var _S350 : bool = dirty_0(body_of_0(_S349));
    if(!_S350)
    {
        return;
    }
    var _S351 : u32 = bond_5.y;
    var a_8 : u32 = _S349;
    var b_51 : u32 = _S351;
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
        var ra_0 : u32 = find_0(a_8);
        var rb_0 : u32 = find_0(b_51);
        if(ra_0 == rb_0)
        {
            return;
        }
        var _S352 : u32 = max(ra_0, rb_0);
        var _S353 : u32 = min(ra_0, rb_0);
        var was_0 : u32 = atomicCompareExchangeWeak(&(atom_0[a_parent_0() + _S352]), _S352, _S353).old_value;
        if(was_0 == _S352)
        {
            return;
        }
        var _S354 : u32 = guard_1 + u32(1);
        a_8 = _S352;
        b_51 = _S353;
        guard_1 = _S354;
    }
    var _S355 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(4));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_flatten(@builtin(global_invocation_id) id_15 : vec3<u32>)
{
    var c_9 : u32 = id_15.x;
    var _S356 : bool;
    if(!live_0(c_9))
    {
        _S356 = true;
    }
    else
    {
        var _S357 : bool = dirty_0(body_of_0(c_9));
        _S356 = !_S357;
    }
    if(_S356)
    {
        return;
    }
    var r_20 : u32 = find_0(c_9);
    work_0[w_label_0() + c_9] = r_20;
    var _S358 : u32 = atomicAdd(&(atom_0[a_count_0() + r_20]), u32(1));
    if(((info_0[c_9] & (u32(1)))) != u32(0))
    {
        var _S359 : u32 = atomicOr(&(atom_0[a_anchor_0() + r_20]), u32(1));
    }
    var _S360 : u32 = u32(4) * c_9;
    var c0_1 : vec4<f32> = chunks_0[_S360];
    var c1_0 : vec4<f32> = chunks_0[_S360 + u32(1)];
    var c2_0 : vec4<f32> = chunks_0[_S360 + u32(2)];
    var m_2 : f32 = c0_1.w;
    var d_21 : vec3<f32> = c0_1.xyz - chunks_0[u32(4) * r_20].xyz;
    var dd_0 : f32 = dot(d_21, d_21);
    var at_13 : u32 = a_sums_0() + u32(20) * r_20;
    add64_0(at_13, m_2, params_0.scale_m_0);
    var _S361 : f32 = d_21.x;
    var _S362 : f32 = m_2 * _S361;
    add64_0(at_13 + u32(2), _S362, params_0.scale_s_0);
    var _S363 : f32 = d_21.y;
    var _S364 : f32 = m_2 * _S363;
    add64_0(at_13 + u32(4), _S364, params_0.scale_s_0);
    var _S365 : f32 = d_21.z;
    add64_0(at_13 + u32(6), m_2 * _S365, params_0.scale_s_0);
    add64_0(at_13 + u32(8), c1_0.x + m_2 * (dd_0 - _S361 * _S361), params_0.scale_j_0);
    add64_0(at_13 + u32(10), c1_0.y + m_2 * (dd_0 - _S363 * _S363), params_0.scale_j_0);
    add64_0(at_13 + u32(12), c1_0.z + m_2 * (dd_0 - _S365 * _S365), params_0.scale_j_0);
    add64_0(at_13 + u32(14), c1_0.w - _S362 * _S363, params_0.scale_j_0);
    add64_0(at_13 + u32(16), c2_0.x - _S362 * _S365, params_0.scale_j_0);
    add64_0(at_13 + u32(18), c2_0.y - _S364 * _S365, params_0.scale_j_0);
    return;
}

fn is_root_0( c_10 : u32) -> bool
{
    var _S366 : bool;
    if(live_0(c_10))
    {
        var _S367 : bool = dirty_0(body_of_0(c_10));
        _S366 = _S367;
    }
    else
    {
        _S366 = false;
    }
    if(_S366)
    {
        _S366 = work_0[w_label_0() + c_10] == c_10;
    }
    else
    {
        _S366 = false;
    }
    return _S366;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_count(@builtin(global_invocation_id) id_16 : vec3<u32>)
{
    var c_11 : u32 = id_16.x;
    var _S368 : bool = is_root_0(c_11);
    if(!_S368)
    {
        return;
    }
    var _S369 : u32 = atomicLoad(&(atom_0[a_count_0() + c_11]));
    var _S370 : u32 = atomicMax(&(atom_0[a_best_count_0() + body_of_0(c_11)]), _S369);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_root(@builtin(global_invocation_id) id_17 : vec3<u32>)
{
    var c_12 : u32 = id_17.x;
    var _S371 : bool = is_root_0(c_12);
    if(!_S371)
    {
        return;
    }
    var s_8 : u32 = body_of_0(c_12);
    var _S372 : u32 = atomicLoad(&(atom_0[a_count_0() + c_12]));
    var _S373 : u32 = atomicLoad(&(atom_0[a_best_count_0() + s_8]));
    if(_S372 == _S373)
    {
        var _S374 : u32 = atomicMin(&(atom_0[a_best_root_0() + s_8]), c_12);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_mark(@builtin(global_invocation_id) id_18 : vec3<u32>)
{
    var c_13 : u32 = id_18.x;
    if(c_13 >= (params_0.chunk_used_0))
    {
        return;
    }
    var _S375 : bool = is_root_0(c_13);
    var _S376 : bool;
    if(_S375)
    {
        var _S377 : u32 = atomicLoad(&(atom_0[a_best_root_0() + body_of_0(c_13)]));
        _S376 = _S377 != c_13;
    }
    else
    {
        _S376 = false;
    }
    var _S378 : u32;
    if(_S376)
    {
        _S378 = u32(1);
    }
    else
    {
        _S378 = u32(0);
    }
    work_0[w_new_0() + c_13] = _S378;
    return;
}

var<workgroup> g_scan_0 : array<u32, i32(1024)>;

@compute
@workgroup_size(256, 1, 1)
fn fr_scan_blocks(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var step_4 : u32;
    var add_0 : u32;
    var t_9 : u32 = thread_2.x;
    var _S379 : u32 = group_0.x;
    var _S380 : u32 = u32(1024) * _S379 + u32(4) * t_9;
    var v_11 : array<u32, i32(4)>;
    var j_6 : u32 = u32(0);
    var sum_2 : u32 = u32(0);
    loop
    {
        if(j_6 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S380 + j_6) < (scan_0.n_0))
        {
            step_4 = work_0[scan_0.in_base_0 + _S380 + j_6];
        }
        else
        {
            step_4 = u32(0);
        }
        v_11[j_6] = step_4;
        var sum_3 : u32 = sum_2 + step_4;
        j_6 = j_6 + u32(1);
        sum_2 = sum_3;
    }
    g_scan_0[t_9] = sum_2;
    workgroupBarrier();
    step_4 = u32(1);
    loop
    {
        if(step_4 < u32(256))
        {
        }
        else
        {
            break;
        }
        if(t_9 >= step_4)
        {
            add_0 = g_scan_0[t_9 - step_4];
        }
        else
        {
            add_0 = u32(0);
        }
        workgroupBarrier();
        g_scan_0[t_9] = g_scan_0[t_9] + add_0;
        workgroupBarrier();
        step_4 = (step_4 << (u32(1)));
    }
    var _S381 : u32 = g_scan_0[t_9] - sum_2;
    j_6 = u32(0);
    add_0 = _S381;
    loop
    {
        if(j_6 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S380 + j_6) < (scan_0.n_0))
        {
            work_0[scan_0.out_base_0 + _S380 + j_6] = add_0;
        }
        var run_1 : u32 = add_0 + v_11[j_6];
        j_6 = j_6 + u32(1);
        add_0 = run_1;
    }
    if(t_9 == u32(255))
    {
        work_0[scan_0.block_base_0 + _S379] = g_scan_0[i32(255)];
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn fr_scan_top(@builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var t_10 : u32 = thread_3.x;
    var blocks_0 : u32 = (scan_0.n_0 + u32(1023)) / u32(1024);
    var _S382 : bool = t_10 < blocks_0;
    var v_12 : u32;
    if(_S382)
    {
        v_12 = work_0[scan_0.block_base_0 + t_10];
    }
    else
    {
        v_12 = u32(0);
    }
    g_scan_0[t_10] = v_12;
    workgroupBarrier();
    var step_5 : u32 = u32(1);
    loop
    {
        if(step_5 < u32(1024))
        {
        }
        else
        {
            break;
        }
        var add_1 : u32;
        if(t_10 >= step_5)
        {
            add_1 = g_scan_0[t_10 - step_5];
        }
        else
        {
            add_1 = u32(0);
        }
        workgroupBarrier();
        g_scan_0[t_10] = g_scan_0[t_10] + add_1;
        workgroupBarrier();
        step_5 = (step_5 << (u32(1)));
    }
    if(_S382)
    {
        work_0[scan_0.block_base_0 + t_10] = g_scan_0[t_10] - v_12;
    }
    if(t_10 == u32(1023))
    {
        work_0[scan_0.block_base_0 + blocks_0] = g_scan_0[i32(1023)];
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_scan_add(@builtin(global_invocation_id) id_19 : vec3<u32>)
{
    var i_11 : u32 = id_19.x;
    if(i_11 >= (scan_0.n_0))
    {
        return;
    }
    work_0[scan_0.out_base_0 + i_11] = work_0[scan_0.out_base_0 + i_11] + work_0[scan_0.block_base_0 + i_11 / u32(1024)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_freelist(@builtin(global_invocation_id) id_20 : vec3<u32>)
{
    var s_9 : u32 = id_20.x;
    var _S383 : bool;
    if(s_9 >= (params_0.body_count_0))
    {
        _S383 = true;
    }
    else
    {
        _S383 = work_0[w_free_0() + s_9] == u32(0);
    }
    if(_S383)
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
fn fr_assign(@builtin(global_invocation_id) id_21 : vec3<u32>)
{
    var c_14 : u32 = id_21.x;
    var _S384 : bool = is_root_0(c_14);
    if(!_S384)
    {
        return;
    }
    var s_10 : u32 = body_of_0(c_14);
    var _S385 : u32 = atomicLoad(&(atom_0[a_best_root_0() + s_10]));
    var slot_10 : u32;
    if(_S385 != c_14)
    {
        var k_25 : u32 = work_0[w_rank_0() + c_14];
        if(work_0[w_rank_0() + c_14] < (free_total_0()))
        {
            var slot_11 : u32 = work_0[w_free_list_0() + k_25];
            work_0[w_free_0() + work_0[w_free_list_0() + k_25]] = u32(0);
            var _S386 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(2)]), u32(1));
            slot_10 = slot_11;
        }
        else
        {
            var _S387 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(1));
            slot_10 = s_10;
        }
    }
    else
    {
        slot_10 = s_10;
    }
    work_0[w_slot_0() + c_14] = slot_10;
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
    var q_18 : Quat_0;
    if(m22_0 <= 0.0f)
    {
        var dif10_0 : f32 = m11_0 - m00_0;
        var omm22_0 : f32 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            var f_4 : f32 = omm22_0 - dif10_0;
            var inv_0 : f32 = 0.5f / sqrt(f_4);
            q_18.x_0 = f_4 * inv_0;
            q_18.y_0 = (m01_0 + m10_0) * inv_0;
            q_18.z_0 = (m02_0 + m20_0) * inv_0;
            q_18.w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            var f_5 : f32 = omm22_0 + dif10_0;
            var inv_1 : f32 = 0.5f / sqrt(f_5);
            q_18.x_0 = (m01_0 + m10_0) * inv_1;
            q_18.y_0 = f_5 * inv_1;
            q_18.z_0 = (m12_0 + m21_0) * inv_1;
            q_18.w_0 = (m20_0 - m02_0) * inv_1;
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
            q_18.x_0 = (m02_0 + m20_0) * inv_2;
            q_18.y_0 = (m12_0 + m21_0) * inv_2;
            q_18.z_0 = f_6 * inv_2;
            q_18.w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            var f_7 : f32 = opm22_0 + sum10_0;
            var inv_3 : f32 = 0.5f / sqrt(f_7);
            q_18.x_0 = (m12_0 - m21_0) * inv_3;
            q_18.y_0 = (m20_0 - m02_0) * inv_3;
            q_18.z_0 = (m01_0 - m10_0) * inv_3;
            q_18.w_0 = f_7 * inv_3;
        }
    }
    return q_18;
}

fn principal_axes_0( r0_0 : vec3<f32>,  r1_2 : vec3<f32>,  r2_2 : vec3<f32>,  rot_0 : ptr<function, Quat_0>,  moments_0 : ptr<function, vec3<f32>>)
{
    var _S388 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S389 : array<f32, i32(3)> = array<f32, i32(3)>( r1_2.x, r1_2.y, r1_2.z );
    var _S390 : array<f32, i32(3)> = array<f32, i32(3)>( r2_2.x, r2_2.y, r2_2.z );
    var a_9 : array<array<f32, i32(3)>, i32(3)>;
    a_9[i32(0)] = _S388;
    a_9[i32(1)] = _S389;
    a_9[i32(2)] = _S390;
    var _S391 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S392 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S393 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_13 : array<array<f32, i32(3)>, i32(3)>;
    v_13[i32(0)] = _S391;
    v_13[i32(1)] = _S392;
    v_13[i32(2)] = _S393;
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
        var q_19 : u32;
        if((abs(a_9[i32(0)][i32(2)])) > (abs(a_9[u32(0)][u32(1)])))
        {
            q_19 = u32(2);
        }
        else
        {
            q_19 = u32(1);
        }
        var p_14 : u32;
        var q_20 : u32;
        if((abs(a_9[i32(1)][i32(2)])) > (abs(a_9[u32(0)][q_19])))
        {
            p_14 = u32(1);
            q_20 = u32(2);
        }
        else
        {
            p_14 = u32(0);
            q_20 = q_19;
        }
        if((abs(a_9[p_14][q_20])) <= (9.99999982451670044e-15f * max(max(abs(a_9[i32(0)][i32(0)]), abs(a_9[i32(1)][i32(1)])), max(abs(a_9[i32(2)][i32(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        var theta_0 : f32 = 0.5f * atan2(2.0f * a_9[p_14][q_20], a_9[q_20][q_20] - a_9[p_14][p_14]);
        var s_11 : f32 = sin(theta_0);
        var c_15 : f32 = cos(theta_0);
        var app_0 : f32 = a_9[p_14][p_14];
        var aqq_0 : f32 = a_9[q_20][q_20];
        var apq_0 : f32 = a_9[p_14][q_20];
        var k_26 : u32 = u32(0);
        loop
        {
            if(k_26 < u32(3))
            {
            }
            else
            {
                break;
            }
            var _S394 : bool;
            if(k_26 != p_14)
            {
                _S394 = k_26 != q_20;
            }
            else
            {
                _S394 = false;
            }
            if(_S394)
            {
                var kp_0 : f32 = a_9[k_26][p_14];
                var kq_0 : f32 = a_9[k_26][q_20];
                var _S395 : f32 = c_15 * a_9[k_26][p_14] - s_11 * a_9[k_26][q_20];
                a_9[k_26][p_14] = _S395;
                a_9[p_14][k_26] = _S395;
                var _S396 : f32 = s_11 * kp_0 + c_15 * kq_0;
                a_9[k_26][q_20] = _S396;
                a_9[q_20][k_26] = _S396;
            }
            k_26 = k_26 + u32(1);
        }
        var _S397 : f32 = c_15 * c_15;
        var _S398 : f32 = 2.0f * s_11 * c_15 * apq_0;
        var _S399 : f32 = s_11 * s_11;
        a_9[p_14][p_14] = _S397 * app_0 - _S398 + _S399 * aqq_0;
        a_9[q_20][q_20] = _S399 * app_0 + _S398 + _S397 * aqq_0;
        a_9[p_14][q_20] = 0.0f;
        a_9[q_20][p_14] = 0.0f;
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
            var kp_1 : f32 = v_13[row_1][p_14];
            var kq_1 : f32 = v_13[row_1][q_20];
            v_13[row_1][p_14] = c_15 * v_13[row_1][p_14] - s_11 * v_13[row_1][q_20];
            v_13[row_1][q_20] = s_11 * kp_1 + c_15 * kq_1;
            row_1 = row_1 + u32(1);
        }
        it_0 = it_0 + u32(1);
    }
    var x_6 : vec3<f32> = vec3<f32>(v_13[i32(0)][i32(0)], v_13[i32(1)][i32(0)], v_13[i32(2)][i32(0)]);
    var y_1 : vec3<f32> = vec3<f32>(v_13[i32(0)][i32(1)], v_13[i32(1)][i32(1)], v_13[i32(2)][i32(1)]);
    var z_1 : vec3<f32> = vec3<f32>(v_13[i32(0)][i32(2)], v_13[i32(1)][i32(2)], v_13[i32(2)][i32(2)]);
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
    (*moments_0) = vec3<f32>(a_9[i32(0)][i32(0)], a_9[i32(1)][i32(1)], a_9[i32(2)][i32(2)]);
    return;
}

fn rot_to_rubble_0( q_21 : Quat_0) -> Quat_0
{
    var axes_1 : Quat_0;
    axes_1.w_0 = 0.70710676908493042f;
    axes_1.x_0 = -0.70710676908493042f;
    axes_1.y_0 = 0.0f;
    axes_1.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_1, q_21));
}

fn stage_0( c_16 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * c_16;
}

fn quat_vec_0( q_22 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_22.x_0, q_22.y_0, q_22.z_0, q_22.w_0);
}

fn frame_step_0() -> u32
{
    return work_0[w_frame_0() + u32(1)];
}

@compute
@workgroup_size(64, 1, 1)
fn fr_piece(@builtin(global_invocation_id) id_22 : vec3<u32>)
{
    var c_17 : u32 = id_22.x;
    var _S400 : bool = is_root_0(c_17);
    if(!_S400)
    {
        return;
    }
    var parent_0 : u32 = body_of_0(c_17);
    var pb_2 : u32 = u32(7) * parent_0;
    var at_14 : u32 = a_sums_0() + u32(20) * c_17;
    var mass_1 : f32 = read64_0(at_14, params_0.scale_m_0);
    var _S401 : f32 = read64_0(at_14 + u32(2), params_0.scale_s_0);
    var _S402 : f32 = read64_0(at_14 + u32(4), params_0.scale_s_0);
    var _S403 : f32 = read64_0(at_14 + u32(6), params_0.scale_s_0);
    var d_22 : vec3<f32> = vec3<f32>(_S401, _S402, _S403) / vec3<f32>(max(mass_1, 1.00000000317107685e-30f));
    var jxx_0 : f32 = read64_0(at_14 + u32(8), params_0.scale_j_0);
    var jyy_0 : f32 = read64_0(at_14 + u32(10), params_0.scale_j_0);
    var jzz_0 : f32 = read64_0(at_14 + u32(12), params_0.scale_j_0);
    var jxy_0 : f32 = read64_0(at_14 + u32(14), params_0.scale_j_0);
    var jxz_0 : f32 = read64_0(at_14 + u32(16), params_0.scale_j_0);
    var jyz_0 : f32 = read64_0(at_14 + u32(18), params_0.scale_j_0);
    var dd_1 : f32 = dot(d_22, d_22);
    var _S404 : f32 = d_22.x;
    var _S405 : f32 = mass_1 * _S404;
    var _S406 : f32 = d_22.y;
    var _S407 : f32 = jxy_0 + _S405 * _S406;
    var _S408 : f32 = d_22.z;
    var _S409 : f32 = jxz_0 + _S405 * _S408;
    var _S410 : f32 = jyz_0 + mass_1 * _S406 * _S408;
    var com_6 : vec3<f32> = chunks_0[u32(4) * c_17].xyz + d_22;
    var principal_0 : Quat_0;
    var moments_1 : vec3<f32>;
    principal_axes_0(vec3<f32>(jxx_0 - mass_1 * (dd_1 - _S404 * _S404), _S407, _S409), vec3<f32>(_S407, jyy_0 - mass_1 * (dd_1 - _S406 * _S406), _S410), vec3<f32>(_S409, _S410, jzz_0 - mass_1 * (dd_1 - _S408 * _S408)), &(principal_0), &(moments_1));
    var _S411 : u32 = atomicLoad(&(atom_0[a_anchor_0() + c_17]));
    var anchored_0 : bool = _S411 != u32(0);
    var q_23 : Quat_0;
    var origin_10 : vec3<f32>;
    structure_pose_5(parent_0, pb_2, &(q_23), &(origin_10));
    var p_parent_0 : vec3<f32> = bodies_0[pb_2].xyz;
    var v_parent_0 : vec3<f32> = bodies_0[pb_2 + u32(3)].xyz;
    var w_parent_0 : vec3<f32> = bodies_0[pb_2 + u32(4)].xyz;
    var pos_0 : vec3<f32> = to_rubble_0(origin_10 + rotate_0(q_23, com_6));
    var orientation_0 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(q_23, principal_0)));
    var _S412 : bool;
    if(anchored_0)
    {
        _S412 = true;
    }
    else
    {
        _S412 = !(mass_1 > 0.0f);
    }
    var inv_mass_3 : f32;
    if(_S412)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / mass_1;
    }
    var _S413 : vec3<f32> = vec3<f32>(0.0f);
    var inv_inertia_0 : vec3<f32>;
    if(!anchored_0)
    {
        var _S414 : f32;
        if((moments_1.x) > 0.0f)
        {
            _S414 = 1.0f / moments_1.x;
        }
        else
        {
            _S414 = 0.0f;
        }
        var _S415 : f32;
        if((moments_1.y) > 0.0f)
        {
            _S415 = 1.0f / moments_1.y;
        }
        else
        {
            _S415 = 0.0f;
        }
        var _S416 : f32;
        if((moments_1.z) > 0.0f)
        {
            _S416 = 1.0f / moments_1.z;
        }
        else
        {
            _S416 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S414, _S415, _S416);
    }
    else
    {
        inv_inertia_0 = _S413;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_0)
    {
        velocity_0 = _S413;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    var spin_0 : vec3<f32>;
    if(anchored_0)
    {
        spin_0 = _S413;
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
    var st_2 : u32 = stage_0(c_17);
    frames_0[st_2] = vec4<f32>(pos_0, inv_mass_3);
    frames_0[st_2 + u32(1)] = quat_vec_0(orientation_0);
    frames_0[st_2 + u32(2)] = vec4<f32>(inv_inertia_0, (bitcast<f32>((flags_0))));
    frames_0[st_2 + u32(3)] = vec4<f32>(velocity_0, 0.0f);
    frames_0[st_2 + u32(4)] = vec4<f32>(spin_0, (bitcast<f32>((frame_step_0()))));
    frames_0[st_2 + u32(5)] = vec4<f32>(0.0f);
    frames_0[st_2 + u32(6)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    frames_0[st_2 + u32(7)] = vec4<f32>(com_6, frames_0[u32(2) * parent_0].w);
    frames_0[st_2 + u32(8)] = quat_vec_0(principal_0);
    return;
}

fn f_events_0() -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * params_0.chunk_count_0 + u32(3) * params_0.body_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_commit(@builtin(global_invocation_id) id_23 : vec3<u32>)
{
    var c_18 : u32 = id_23.x;
    var _S417 : bool = is_root_0(c_18);
    if(!_S417)
    {
        return;
    }
    var slot_12 : u32 = work_0[w_slot_0() + c_18];
    var st_3 : u32 = stage_0(c_18);
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
            bodies_0[u32(7) * (slot_12 + region_5 * params_0.max_bodies_0) + w_4] = frames_0[st_3 + w_4];
            w_4 = w_4 + u32(1);
        }
        region_5 = region_5 + u32(1);
    }
    var _S418 : u32 = u32(2) * slot_12;
    var _S419 : u32 = st_3 + u32(7);
    frames_0[_S418] = frames_0[_S419];
    var _S420 : u32 = st_3 + u32(8);
    frames_0[_S418 + u32(1)] = frames_0[_S420];
    if((params_0.ev_frames_cap_0) != u32(0))
    {
        var _S421 : u32 = a_counters_0();
        var i_12 : u32 = atomicAdd(&(atom_0[_S421 + u32(22)]), u32(1));
        if(i_12 < (params_0.ev_frames_cap_0))
        {
            var at_15 : u32 = f_events_0() + u32(3) * i_12;
            frames_0[at_15] = vec4<f32>((bitcast<f32>((slot_12))), 0.0f, 0.0f, 0.0f);
            frames_0[at_15 + u32(1)] = frames_0[_S419];
            frames_0[at_15 + u32(2)] = frames_0[_S420];
        }
        else
        {
            var _S422 : u32 = atomicOr(&(atom_0[_S421 + u32(3)]), u32(128));
        }
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_reparent(@builtin(global_invocation_id) id_24 : vec3<u32>)
{
    var c_19 : u32 = id_24.x;
    var _S423 : bool;
    if(!live_0(c_19))
    {
        _S423 = true;
    }
    else
    {
        var _S424 : bool = dirty_0(body_of_0(c_19));
        _S423 = !_S424;
    }
    if(_S423)
    {
        return;
    }
    var r_21 : u32 = work_0[w_label_0() + c_19];
    var slot_13 : u32 = work_0[w_slot_0() + work_0[w_label_0() + c_19]];
    if(work_0[w_slot_0() + work_0[w_label_0() + c_19]] != (body_of_0(c_19)))
    {
        _S423 = (params_0.ev_moved_cap_0) != u32(0);
    }
    else
    {
        _S423 = false;
    }
    if(_S423)
    {
        var _S425 : u32 = a_counters_0();
        var i_13 : u32 = atomicAdd(&(atom_0[_S425 + u32(21)]), u32(1));
        if(i_13 < (params_0.ev_moved_cap_0))
        {
            var _S426 : u32 = a_moved_0() + u32(2) * i_13;
            atomicStore(&(atom_0[_S426]), c_19);
            atomicStore(&(atom_0[_S426 + u32(1)]), slot_13);
        }
        else
        {
            var _S427 : u32 = atomicOr(&(atom_0[_S425 + u32(3)]), u32(128));
        }
    }
    work_0[w_body_0() + c_19] = slot_13;
    var _S428 : u32 = u32(2) * slot_13;
    var _S429 : vec3<f32> = frames_0[_S428].xyz;
    var _S430 : Quat_0 = conj_0(quat_of_0(frames_0[_S428 + u32(1)]));
    var inv_mass_4 : f32 = bodies_0[u32(7) * slot_13].w;
    var _S431 : f32;
    if(inv_mass_4 > 0.0f)
    {
        _S431 = 1.0f / inv_mass_4;
    }
    else
    {
        _S431 = float_bits_0(u32(2139095040));
    }
    var c2_2 : vec4<f32> = chunks_0[u32(4) * c_19 + u32(2)];
    var first_1 : u32 = (bitcast<u32>((c2_2.z)));
    var _S432 : u32 = (bitcast<u32>((c2_2.w)));
    var _S433 : vec4<f32> = chunks_0[u32(4) * r_21 + u32(3)];
    var p_15 : u32 = first_1;
    loop
    {
        if(p_15 < (first_1 + _S432))
        {
        }
        else
        {
            break;
        }
        var _S434 : u32 = u32(2) * p_15;
        var o_4 : vec4<f32> = parts_0[_S434];
        var col_0 : u32 = (bitcast<u32>((o_4.w)));
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S434 + u32(1)]);
        var _S435 : u32 = u32(4) * col_0;
        colliders_0[_S435] = vec4<f32>((bitcast<f32>((slot_13))), colliders_0[_S435].y, colliders_0[_S435].z, colliders_0[_S435].w);
        var _S436 : u32 = _S435 + u32(1);
        colliders_0[_S436] = vec4<f32>(rotate_0(_S430, o_4.xyz - _S429), colliders_0[_S436].w);
        colliders_0[_S435 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S430, rot_1)));
        var _S437 : u32 = u32(2) * col_0;
        collider_info_0[_S437] = vec4<u32>(collider_info_0[_S437].x, slot_13, collider_info_0[_S437].z, collider_info_0[_S437].w);
        collider_info_0[_S437 + u32(1)] = vec4<u32>((bitcast<u32>((_S431))), (bitcast<u32>((_S433.x))), (bitcast<u32>((_S433.w))), u32(0));
        p_15 = p_15 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_body_clear(@builtin(global_invocation_id) id_25 : vec3<u32>)
{
    var s_12 : u32 = id_25.x;
    if(s_12 >= (params_0.body_count_0))
    {
        return;
    }
    atomicStore(&(atom_0[a_dirty_0() + s_12]), u32(0));
    return;
}

fn slot_color_0( slot_14 : u32) -> vec3<f32>
{
    var h_5 : f32 = fract(f32(slot_14) * 0.6180340051651001f) * 6.0f;
    var f_8 : f32 = fract(h_5);
    var q_24 : f32 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_8);
    var t_11 : f32 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_8));
    var k_27 : u32 = u32(h_5);
    var _S438 : vec3<f32>;
    if(k_27 == u32(0))
    {
        _S438 = vec3<f32>(0.92000001668930054f, t_11, 0.34959998726844788f);
    }
    else
    {
        if(k_27 == u32(1))
        {
            _S438 = vec3<f32>(q_24, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_27 == u32(2))
            {
                _S438 = vec3<f32>(0.34959998726844788f, 0.92000001668930054f, t_11);
            }
            else
            {
                if(k_27 == u32(3))
                {
                    _S438 = vec3<f32>(0.34959998726844788f, q_24, 0.92000001668930054f);
                }
                else
                {
                    if(k_27 == u32(4))
                    {
                        _S438 = vec3<f32>(t_11, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S438 = vec3<f32>(0.92000001668930054f, 0.34959998726844788f, q_24);
                    }
                }
            }
        }
    }
    return _S438;
}

fn write_object_0( o_5 : u32,  q_25 : Quat_0,  t_12 : vec3<f32>,  color_0 : vec4<f32>)
{
    var x_7 : vec3<f32> = rotate_0(q_25, vec3<f32>(1.0f, 0.0f, 0.0f));
    var y_2 : vec3<f32> = rotate_0(q_25, vec3<f32>(0.0f, 1.0f, 0.0f));
    var z_3 : vec3<f32> = rotate_0(q_25, vec3<f32>(0.0f, 0.0f, 1.0f));
    var _S439 : u32 = u32(4) * o_5;
    objects_0[_S439] = vec4<f32>(x_7.x, y_2.x, z_3.x, t_12.x);
    objects_0[_S439 + u32(1)] = vec4<f32>(x_7.y, y_2.y, z_3.y, t_12.y);
    objects_0[_S439 + u32(2)] = vec4<f32>(x_7.z, y_2.z, z_3.z, t_12.z);
    objects_0[_S439 + u32(3)] = color_0;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_chunks(@builtin(global_invocation_id) id_26 : vec3<u32>)
{
    var k_28 : u32 = id_26.x;
    if(!live_0(k_28))
    {
        return;
    }
    var slot_15 : u32 = body_of_0(k_28);
    var b_52 : u32 = u32(7) * slot_15;
    var q_26 : Quat_0;
    var origin_11 : vec3<f32>;
    structure_pose_5(slot_15, b_52, &(q_26), &(origin_11));
    var _S440 : u32 = u32(3) * k_28;
    var v0_0 : vec4<f32> = view_chunks_0[_S440];
    var base_2 : vec3<f32> = view_chunks_0[_S440 + u32(2)].xyz;
    var fixed_0 : bool = (bodies_0[b_52].w) == 0.0f;
    var color_1 : vec3<f32>;
    if((view_0.mode_0) == u32(1))
    {
        var _S441 : f32 = max(length(bodies_0[b_52 + u32(3)].xyz), length(bodies_0[b_52 + u32(4)].xyz));
        if(fixed_0)
        {
            color_1 = vec3<f32>(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S441 > (view_0.rest_speed_0))
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
            color_1 = slot_color_0(slot_15);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = vec3<f32>(0.44999998807907104f) * base_2 + vec3<f32>(0.55000001192092896f) * slot_color_0(slot_15);
            }
        }
    }
    write_object_0((bitcast<u32>((v0_0.w))), normalized_0(quat_mul_0(q_26, quat_of_0(view_chunks_0[_S440 + u32(1)]))), origin_11 + rotate_0(q_26, v0_0.xyz), vec4<f32>(color_1, 1.0f));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_shots(@builtin(global_invocation_id) id_27 : vec3<u32>)
{
    var i_14 : u32 = id_27.x;
    if(i_14 >= (view_0.shot_count_1))
    {
        return;
    }
    var b_53 : u32 = u32(7) * shot_bodies_0[i_14];
    var o_6 : u32 = view_0.impactor_base_0 + i_14;
    if((bodies_0[b_53].w) == 0.0f)
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
    write_object_0(o_6, rot_from_rubble_0(quat_of_0(bodies_0[b_53 + u32(1)])), from_rubble_0(bodies_0[b_53].xyz), vec4<f32>(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f));
    return;
}

fn chunk_world_0( k_29 : u32) -> vec3<f32>
{
    var slot_16 : u32 = body_of_0(k_29);
    var q_27 : Quat_0;
    var origin_12 : vec3<f32>;
    structure_pose_3(slot_16, u32(7) * slot_16, &(q_27), &(origin_12));
    return origin_12 + rotate_0(q_27, view_chunks_0[u32(3) * k_29].xyz);
}

fn write_line_vertex_0( v_14 : u32,  p_16 : vec3<f32>)
{
    var _S442 : u32 = u32(7) * v_14;
    lines_0[_S442] = p_16.x;
    lines_0[_S442 + u32(1)] = p_16.y;
    lines_0[_S442 + u32(2)] = p_16.z;
    lines_0[_S442 + u32(3)] = 1.0f;
    lines_0[_S442 + u32(4)] = 0.86000001430511475f;
    lines_0[_S442 + u32(5)] = 0.25f;
    lines_0[_S442 + u32(6)] = 1.0f;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_bonds(@builtin(global_invocation_id) id_28 : vec3<u32>)
{
    var i_15 : u32 = id_28.x;
    if(i_15 >= (params_0.bond_used_0))
    {
        return;
    }
    var bond_6 : vec4<u32> = bonds_0[i_15];
    var a_10 : vec3<f32> = chunk_world_0(bond_6.x);
    var b_54 : vec3<f32>;
    if(work_0[w_alive_0() + i_15] != u32(0))
    {
        b_54 = chunk_world_0(bond_6.y);
    }
    else
    {
        b_54 = a_10;
    }
    var _S443 : u32 = u32(2) * i_15;
    write_line_vertex_0(_S443, a_10);
    write_line_vertex_0(_S443 + u32(1), b_54);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_reset(@builtin(global_invocation_id) id_29 : vec3<u32>)
{
    var b_55 : u32 = id_29.x;
    if(b_55 >= (params_0.body_count_0))
    {
        return;
    }
    var at_16 : u32 = a_strike_0() + u32(9) * b_55;
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
        atomicStore(&(atom_0[at_16 + w_6]), u32(4294967295));
        atomicStore(&(atom_0[at_16 + u32(3) + w_6]), u32(0));
        w_6 = w_6 + u32(1);
    }
    atomicStore(&(atom_0[at_16 + u32(6)]), u32(2139095040));
    atomicStore(&(atom_0[at_16 + u32(7)]), u32(2139095040));
    atomicStore(&(atom_0[at_16 + u32(8)]), u32(4294967295));
    frames_0[strike_at_0(b_55)][i32(3)] = 0.0f;
    return;
}

fn chunk_box_world_2( k_30 : u32,  q_28 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_17 : u32 = body_of_0(k_30);
    var origin_13 : vec3<f32>;
    structure_pose_0(slot_17, u32(7) * slot_17, &((*q_28)), &(origin_13));
    return origin_13 + rotate_0((*q_28), geometry_0[u32(3) * k_30].xyz);
}

fn chunk_box_world_3( k_31 : u32,  q_29 : ptr<function, Quat_0>) -> vec3<f32>
{
    var slot_18 : u32 = body_of_0(k_31);
    var origin_14 : vec3<f32>;
    structure_pose_5(slot_18, u32(7) * slot_18, &((*q_29)), &(origin_14));
    return origin_14 + rotate_0((*q_29), geometry_0[u32(3) * k_31].xyz);
}

fn ordered_0( f_9 : f32) -> u32
{
    var u_7 : u32 = (bitcast<u32>((f_9)));
    var _S444 : u32;
    if(((u_7 & (u32(2147483648)))) != u32(0))
    {
        _S444 = ~u_7;
    }
    else
    {
        _S444 = (u_7 | (u32(2147483648)));
    }
    return _S444;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_bounds(@builtin(global_invocation_id) id_30 : vec3<u32>)
{
    var k_32 : u32 = id_30.x;
    var _S445 : bool;
    if(!live_0(k_32))
    {
        _S445 = true;
    }
    else
    {
        _S445 = (params_0.shot_count_0) == u32(0);
    }
    if(_S445)
    {
        return;
    }
    var q_30 : Quat_0;
    var _S446 : vec3<f32> = chunk_box_world_3(k_32, &(q_30));
    var _S447 : f32 = geometry_0[u32(3) * k_32].w;
    var _S448 : u32 = a_strike_0() + u32(9) * body_of_0(k_32);
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
        var _S449 : u32 = ax_1;
        var _S450 : u32 = atomicMin(&(atom_0[_S448 + ax_1]), ordered_0(_S446[ax_1] - _S447));
        var _S451 : u32 = atomicMax(&(atom_0[_S448 + u32(3) + ax_1]), ordered_0(_S446[_S449] + _S447));
        ax_1 = ax_1 + u32(1);
    }
    return;
}

fn unordered_0( u_8 : u32) -> f32
{
    var _S452 : u32;
    if(((u_8 & (u32(2147483648)))) != u32(0))
    {
        _S452 = (u_8 & (u32(2147483647)));
    }
    else
    {
        _S452 = ~u_8;
    }
    return (bitcast<f32>((_S452)));
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_find(@builtin(global_invocation_id) id_31 : vec3<u32>)
{
    var b_56 : u32 = id_31.x;
    var _S453 : bool;
    if(b_56 >= (params_0.body_count_0))
    {
        _S453 = true;
    }
    else
    {
        _S453 = (params_0.shot_count_0) == u32(0);
    }
    if(_S453)
    {
        return;
    }
    var at_17 : u32 = a_strike_0() + u32(9) * b_56;
    var _S454 : u32 = atomicLoad(&(atom_0[at_17]));
    if(_S454 == u32(4294967295))
    {
        return;
    }
    var _S455 : u32 = atomicLoad(&(atom_0[at_17]));
    var _S456 : f32 = unordered_0(_S455);
    var _S457 : u32 = atomicLoad(&(atom_0[at_17 + u32(1)]));
    var _S458 : f32 = unordered_0(_S457);
    var _S459 : u32 = atomicLoad(&(atom_0[at_17 + u32(2)]));
    var _S460 : vec3<f32> = vec3<f32>(_S456, _S458, unordered_0(_S459));
    var _S461 : u32 = atomicLoad(&(atom_0[at_17 + u32(3)]));
    var _S462 : f32 = unordered_0(_S461);
    var _S463 : u32 = atomicLoad(&(atom_0[at_17 + u32(4)]));
    var _S464 : f32 = unordered_0(_S463);
    var _S465 : u32 = atomicLoad(&(atom_0[at_17 + u32(5)]));
    var _S466 : vec3<f32> = vec3<f32>(_S462, _S464, unordered_0(_S465));
    var _S467 : vec3<f32> = from_rubble_0(bodies_0[u32(7) * b_56 + u32(3)].xyz);
    var _S468 : f32 = float_bits_0(u32(2139095040));
    var k_33 : u32 = u32(0);
    loop
    {
        if(k_33 < (params_0.shot_count_0))
        {
        }
        else
        {
            break;
        }
        var info_2 : vec4<f32> = shots_0[u32(2) * k_33];
        var sb_1 : u32 = u32(7) * (bitcast<u32>((info_2.x)));
        if((bodies_0[sb_1].w) == 0.0f)
        {
            k_33 = k_33 + u32(1);
            continue;
        }
        var c_20 : vec3<f32> = from_rubble_0(bodies_0[sb_1].xyz);
        var rel_3 : vec3<f32> = from_rubble_0(bodies_0[sb_1 + u32(3)].xyz) - _S467;
        if(!((length(rel_3)) > (params_0.strike_speed_0)))
        {
            k_33 = k_33 + u32(1);
            continue;
        }
        var _S469 : vec3<f32> = vec3<f32>(info_2.y);
        var _S470 : vec3<f32> = _S460 - _S469;
        var _S471 : vec3<f32> = _S466 + _S469;
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
            var _S472 : u32 = ax_2;
            if((abs(rel_3[ax_2])) < 9.999999960041972e-13f)
            {
                var _S473 : u32 = ax_2;
                if((c_20[ax_2]) < (_S470[ax_2]))
                {
                    _S453 = true;
                }
                else
                {
                    _S453 = (c_20[_S473]) > (_S471[ax_2]);
                }
                var t0_2 : f32;
                if(_S453)
                {
                    t0_2 = _S468;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + u32(1);
                continue;
            }
            var ta_1 : f32 = (_S470[ax_2] - c_20[ax_2]) / rel_3[_S472];
            var tb_1 : f32 = (_S471[ax_2] - c_20[ax_2]) / rel_3[_S472];
            var _S474 : f32 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S474;
            ax_2 = ax_2 + u32(1);
        }
        if(t0_1 <= t1_1)
        {
            var st_4 : u32 = strike_at_0(b_56);
            frames_0[st_4] = vec4<f32>(c_20, f32(k_33 + u32(1)));
            frames_0[st_4 + u32(1)] = vec4<f32>(rel_3, t0_1);
            return;
        }
        k_33 = k_33 + u32(1);
    }
    return;
}

fn struck_2( b_57 : u32,  shot_5 : ptr<function, u32>) -> bool
{
    var k_34 : u32 = u32(frames_0[strike_at_0(b_57)].w);
    (*shot_5) = k_34 - u32(1);
    return k_34 != u32(0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_entry(@builtin(global_invocation_id) id_32 : vec3<u32>)
{
    var k_35 : u32 = id_32.x;
    var _S475 : bool;
    if(!live_0(k_35))
    {
        _S475 = true;
    }
    else
    {
        _S475 = (params_0.shot_count_0) == u32(0);
    }
    if(_S475)
    {
        return;
    }
    var b_58 : u32 = body_of_0(k_35);
    var shot_6 : u32;
    var _S476 : bool = struck_2(b_58, &(shot_6));
    if(!_S476)
    {
        return;
    }
    var _S477 : u32 = strike_at_0(b_58);
    var s1_4 : vec4<f32> = frames_0[_S477 + u32(1)];
    var from_1 : vec3<f32> = frames_0[_S477].xyz;
    var rel_4 : vec3<f32> = frames_0[_S477 + u32(1)].xyz;
    var d_23 : vec3<f32> = rel_4 / vec3<f32>(max(length(rel_4), 9.999999960041972e-13f));
    var _S478 : f32 = shots_0[u32(2) * shot_6].y;
    var q_31 : Quat_0;
    var cc_1 : vec3<f32> = chunk_box_world_3(k_35, &(q_31));
    var _S479 : u32 = u32(3) * k_35;
    var _S480 : Quat_0 = normalized_0(quat_mul_0(q_31, quat_of_0(geometry_0[_S479 + u32(2)])));
    var _S481 : vec3<f32> = geometry_0[_S479 + u32(1)].xyz;
    var _S482 : f32 = float_bits_0(u32(2139095040));
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
        var e_3 : vec3<f32> = vec3<f32>(0.0f);
        e_3[ax_3] = 1.0f;
        var axis_1 : vec3<f32> = rotate_0(_S480, e_3);
        var o_7 : f32 = dot(from_1 - cc_1, axis_1);
        var dv_1 : f32 = dot(rel_4, axis_1);
        var ext_2 : f32 = _S481[ax_3] + _S478;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            var t_in_8 : f32;
            if((abs(o_7)) > ext_2)
            {
                t_in_8 = _S482;
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
        var _S483 : f32 = min(t_out_0, max(ta_2, tb_2));
        t_in_7 = max(t_in_7, min(ta_2, tb_2));
        t_out_0 = _S483;
        ax_3 = ax_3 + u32(1);
    }
    var at_18 : u32 = a_strike_0() + u32(9) * b_58;
    if(t_in_7 <= t_out_0)
    {
        var _S484 : u32 = atomicMin(&(atom_0[at_18 + u32(6)]), (bitcast<u32>((t_in_7))));
    }
    var r_22 : vec3<f32> = cc_1 - (from_1 + rel_4 * vec3<f32>(s1_4.w));
    var along_1 : f32 = dot(r_22, d_23);
    var _S485 : u32 = (bitcast<u32>((length(r_22 - d_23 * vec3<f32>(along_1)) + abs(along_1) * 0.00100000004749745f)));
    atomicStore(&(atom_0[a_strike_dist_0() + k_35]), _S485);
    var _S486 : u32 = atomicMin(&(atom_0[at_18 + u32(7)]), _S485);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_nearest(@builtin(global_invocation_id) id_33 : vec3<u32>)
{
    var k_36 : u32 = id_33.x;
    var _S487 : bool;
    if(!live_0(k_36))
    {
        _S487 = true;
    }
    else
    {
        _S487 = (params_0.shot_count_0) == u32(0);
    }
    if(_S487)
    {
        return;
    }
    var b_59 : u32 = body_of_0(k_36);
    var shot_7 : u32;
    if(!struck_2(b_59, &(shot_7)))
    {
        return;
    }
    var at_19 : u32 = a_strike_0() + u32(9) * b_59;
    var _S488 : u32 = atomicLoad(&(atom_0[a_strike_dist_0() + k_36]));
    var _S489 : u32 = atomicLoad(&(atom_0[at_19 + u32(7)]));
    if(_S488 == _S489)
    {
        var _S490 : u32 = atomicMin(&(atom_0[at_19 + u32(8)]), k_36);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_force(@builtin(global_invocation_id) id_34 : vec3<u32>)
{
    var b_60 : u32 = id_34.x;
    var _S491 : bool;
    if(b_60 >= (params_0.body_count_0))
    {
        _S491 = true;
    }
    else
    {
        _S491 = (params_0.shot_count_0) == u32(0);
    }
    if(_S491)
    {
        return;
    }
    var shot_8 : u32;
    var _S492 : bool = struck_2(b_60, &(shot_8));
    if(!_S492)
    {
        return;
    }
    var at_20 : u32 = a_strike_0() + u32(9) * b_60;
    var _S493 : u32 = atomicLoad(&(atom_0[at_20 + u32(6)]));
    var t_in_9 : f32 = (bitcast<f32>((_S493)));
    var entry_9 : u32 = atomicLoad(&(atom_0[at_20 + u32(8)]));
    var st_5 : u32 = strike_at_0(b_60);
    if(!(t_in_9 <= (params_0.dt_0)))
    {
        _S491 = true;
    }
    else
    {
        _S491 = entry_9 == u32(4294967295);
    }
    if(_S491)
    {
        frames_0[st_5][i32(3)] = 0.0f;
        return;
    }
    var rel_5 : vec3<f32> = frames_0[st_5 + u32(1)].xyz;
    var speed_16 : f32 = length(rel_5);
    var info_3 : vec4<f32> = shots_0[u32(2) * shot_8];
    var _S494 : f32 = max(info_3.z, 0.00100000004749745f);
    var m_imp_0 : f32 = info_3.w;
    var e_imp_0 : f32 = shots_0[u32(2) * shot_8 + u32(1)].x;
    var contact_19 : vec3<f32> = frames_0[st_5].xyz + rel_5 * vec3<f32>(t_in_9) + rel_5 / vec3<f32>(max(speed_16, 9.999999960041972e-13f)) * vec3<f32>(info_3.y);
    var inv_mass_5 : f32 = bodies_0[u32(7) * b_60].w;
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
    var modulus_1 : f32 = 1.0f / (1.0f / e_imp_0 + 1.0f / geometry_0[u32(3) * entry_9 + u32(1)].w);
    frames_0[st_5 + u32(2)] = vec4<f32>(contact_19, m_eff_6 * speed_16 / (2.86999988555908203f * pow(m_eff_6 * m_eff_6 / (_S494 * modulus_1 * modulus_1 * max(speed_16, 0.00100000004749745f)), 0.20000000298023224f)));
    return;
}

fn strike_share_breaks_0( bond_7 : vec4<u32>,  contact_20 : vec3<f32>,  force_2 : f32) -> bool
{
    var q_32 : Quat_0;
    var pa_2 : vec3<f32> = chunk_box_world_2(bond_7.x, &(q_32));
    var pb_3 : vec3<f32> = chunk_box_world_2(bond_7.y, &(q_32));
    var dist_0 : f32 = length(vec3<f32>(0.5f) * (pa_2 + pb_3) - contact_20);
    return (force_2 * min(1.0f, (bitcast<f32>((bond_7.w))) / (6.28318548202514648f * max(dist_0 * dist_0, 9.999999960041972e-13f)))) > (bitcast<f32>((bond_7.z)));
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_breaks(@builtin(global_invocation_id) id_35 : vec3<u32>)
{
    var i_16 : u32 = id_35.x;
    var _S495 : bool;
    if(i_16 >= (params_0.bond_used_0))
    {
        _S495 = true;
    }
    else
    {
        _S495 = (params_0.shot_count_0) == u32(0);
    }
    if(_S495)
    {
        _S495 = true;
    }
    else
    {
        _S495 = work_0[w_alive_0() + i_16] == u32(0);
    }
    if(_S495)
    {
        return;
    }
    var bond_8 : vec4<u32> = bonds_0[i_16];
    var b_61 : u32 = body_of_0(bond_8.x);
    var shot_9 : u32;
    if(!struck_2(b_61, &(shot_9)))
    {
        return;
    }
    if(strike_share_breaks_0(bond_8, frames_0[strike_at_0(b_61) + u32(2)].xyz, frames_0[strike_at_0(b_61) + u32(2)].w))
    {
        work_0[w_alive_0() + i_16] = u32(0);
        var _S496 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_61]), u32(1));
        var _S497 : u32 = a_counters_0();
        var _S498 : u32 = atomicAdd(&(atom_0[_S497]), u32(1));
        var _S499 : u32 = atomicAdd(&(atom_0[_S497 + u32(1)]), u32(1));
        event_broken_0(i_16);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_radial(@builtin(global_invocation_id) id_36 : vec3<u32>)
{
    var i_17 : u32 = id_36.x;
    var _S500 : bool;
    if(i_17 >= (params_0.bond_used_0))
    {
        _S500 = true;
    }
    else
    {
        _S500 = (params_0.shot_count_0) == u32(0);
    }
    if(_S500)
    {
        _S500 = true;
    }
    else
    {
        _S500 = work_0[w_alive_0() + i_17] == u32(0);
    }
    if(_S500)
    {
        return;
    }
    var bond_9 : vec4<u32> = bonds_0[i_17];
    var _S501 : u32 = bond_9.x;
    var b_62 : u32 = body_of_0(_S501);
    var shot_10 : u32;
    if(!struck_2(b_62, &(shot_10)))
    {
        return;
    }
    var contact_21 : vec3<f32> = frames_0[strike_at_0(b_62) + u32(2)].xyz;
    var q_33 : Quat_0;
    var pa_3 : vec3<f32> = chunk_box_world_3(_S501, &(q_33));
    var pb_4 : vec3<f32> = chunk_box_world_3(bond_9.y, &(q_33));
    if((length(vec3<f32>(0.5f) * (pa_3 + pb_4) - contact_21)) <= (params_0.strike_radius_0))
    {
        work_0[w_alive_0() + i_17] = u32(0);
        var _S502 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_62]), u32(1));
        var _S503 : u32 = a_counters_0();
        var _S504 : u32 = atomicAdd(&(atom_0[_S503]), u32(1));
        var _S505 : u32 = atomicAdd(&(atom_0[_S503 + u32(1)]), u32(1));
        event_broken_0(i_17);
    }
    return;
}

fn a_lp_0( b_63 : u32) -> u32
{
    return a_strike_0() + u32(9) * params_0.body_count_0 + u32(14) * b_63;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_clear(@builtin(global_invocation_id) id_37 : vec3<u32>)
{
    var b_64 : u32 = id_37.x;
    if(b_64 >= (params_0.body_count_0))
    {
        return;
    }
    var at_21 : u32 = a_lp_0(b_64);
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
        atomicStore(&(atom_0[at_21 + w_7]), u32(0));
        w_7 = w_7 + u32(1);
    }
    atomicStore(&(atom_0[at_21 + u32(12)]), u32(4294967295));
    atomicStore(&(atom_0[at_21 + u32(13)]), u32(0));
    return;
}

fn st_f_0( k_37 : u32,  state_0 : u32) -> u32
{
    return u32(22) * k_37 + state_0 + u32(6);
}

fn lp_f_0( k_38 : u32) -> u32
{
    return st_f_0(k_38, u32(0));
}

fn add64w_0( at_22 : u32,  value_3 : f32,  scale_2 : f32)
{
    var x_8 : f32 = round(value_3 * scale_2);
    var x_9 : f32;
    if(!((abs(x_8)) < 3.99999993722699776e+18f))
    {
        var _S506 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
        x_9 = 0.0f;
    }
    else
    {
        x_9 = x_8;
    }
    var u_9 : f32 = abs(x_9);
    var h_6 : f32 = floor(u_9 * 2.32830643653869629e-10f);
    var lo_3 : u32 = u32(u_9 - h_6 * 4.294967296e+09f);
    var hi_3 : u32 = u32(h_6);
    var lo_4 : u32;
    var hi_4 : u32;
    if(x_9 < 0.0f)
    {
        var lo_5 : u32 = ~lo_3 + u32(1);
        var _S507 : u32 = ~hi_3;
        if(lo_5 == u32(0))
        {
            lo_4 = u32(1);
        }
        else
        {
            lo_4 = u32(0);
        }
        var _S508 : u32 = _S507 + lo_4;
        lo_4 = lo_5;
        hi_4 = _S508;
    }
    else
    {
        lo_4 = lo_3;
        hi_4 = hi_3;
    }
    var old_1 : u32 = atomicAdd(&(atom_0[at_22]), lo_4);
    var carry_1 : u32;
    if((old_1 + lo_4) < old_1)
    {
        carry_1 = u32(1);
    }
    else
    {
        carry_1 = u32(0);
    }
    var hi_5 : u32 = hi_4 + carry_1;
    if(hi_5 != u32(0))
    {
        var _S509 : u32 = atomicAdd(&(atom_0[at_22 + u32(1)]), hi_5);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sum(@builtin(global_invocation_id) id_38 : vec3<u32>)
{
    var k_39 : u32 = id_38.x;
    if(k_39 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_19 : u32 = body_of_0(k_39);
    var q_34 : Quat_0;
    var origin_15 : vec3<f32>;
    structure_pose_5(slot_19, u32(7) * slot_19, &(q_34), &(origin_15));
    var qi_0 : Quat_0 = conj_0(q_34);
    var c0_3 : vec4<f32> = chunks_0[u32(4) * k_39];
    var _S510 : u32 = l_vec_0() + u32(2) * k_39;
    var f_10 : vec3<f32> = rotate_0(qi_0, loads_0[_S510].xyz) + vec3<f32>(c0_3.w) * rotate_0(qi_0, vec3<f32>(params_0.gx_0, params_0.gy_0, params_0.gz_0));
    var t_13 : vec3<f32> = rotate_0(qi_0, loads_0[_S510 + u32(1)].xyz);
    var _S511 : u32 = lp_f_0(k_39);
    lp_0[_S511] = vec4<f32>(f_10, 0.0f);
    lp_0[_S511 + u32(1)] = vec4<f32>(t_13, 0.0f);
    var _S512 : vec3<f32> = t_13 + cross(c0_3.xyz - frames_0[u32(2) * slot_19].xyz, f_10);
    var at_23 : u32 = a_lp_0(slot_19);
    var a_11 : u32 = u32(0);
    loop
    {
        if(a_11 < u32(3))
        {
        }
        else
        {
            break;
        }
        var _S513 : u32 = u32(2) * a_11;
        add64w_0(at_23 + _S513, f_10[a_11], params_0.lp_scale_0);
        add64w_0(at_23 + u32(6) + _S513, _S512[a_11], params_0.lp_scale_0);
        a_11 = a_11 + u32(1);
    }
    var _S514 : u32 = atomicMin(&(atom_0[at_23 + u32(12)]), k_39);
    if(((info_0[k_39] & (u32(1)))) != u32(0))
    {
        atomicStore(&(atom_0[at_23 + u32(13)]), u32(1));
    }
    return;
}

fn read64w3_0( at_24 : u32,  scale_3 : f32) -> vec3<f32>
{
    var _S515 : f32 = read64_0(at_24, scale_3);
    var _S516 : f32 = read64_0(at_24 + u32(2), scale_3);
    var _S517 : f32 = read64_0(at_24 + u32(4), scale_3);
    return vec3<f32>(_S515, _S516, _S517);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_relief(@builtin(global_invocation_id) id_39 : vec3<u32>)
{
    var k_40 : u32 = id_39.x;
    if(k_40 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_20 : u32 = body_of_0(k_40);
    var at_25 : u32 = a_lp_0(slot_20);
    var _S518 : u32 = atomicLoad(&(atom_0[at_25 + u32(13)]));
    var supported_0 : bool = _S518 != u32(0);
    var fixed_1 : bool;
    if(((info_0[k_40] & (u32(1)))) != u32(0))
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            var _S519 : u32 = atomicLoad(&(atom_0[at_25 + u32(12)]));
            fixed_1 = _S519 == k_40;
        }
        else
        {
            fixed_1 = false;
        }
    }
    var _S520 : u32 = lp_f_0(k_40);
    var f_11 : vec4<f32> = lp_0[_S520];
    var _S521 : u32 = _S520 + u32(1);
    var t_14 : vec4<f32> = lp_0[_S521];
    var b_65 : u32 = u32(7) * slot_20;
    var inv_mass_6 : f32 = bodies_0[b_65].w;
    var _S522 : bool;
    if(!supported_0)
    {
        _S522 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S522 = false;
    }
    if(_S522)
    {
        var force_3 : vec3<f32> = read64w3_0(at_25, params_0.lp_scale_0);
        var moment_0 : vec3<f32> = read64w3_0(at_25 + u32(6), params_0.lp_scale_0);
        var _S523 : u32 = u32(2) * slot_20;
        var p_17 : Quat_0 = quat_of_0(frames_0[_S523 + u32(1)]);
        var alpha_0 : vec3<f32> = rotate_0(p_17, bodies_0[b_65 + u32(2)].xyz * rotate_0(conj_0(p_17), moment_0));
        var _S524 : u32 = u32(4) * k_40;
        var c0_4 : vec4<f32> = chunks_0[_S524];
        var c1_2 : vec4<f32> = chunks_0[_S524 + u32(1)];
        var c2_3 : vec4<f32> = chunks_0[_S524 + u32(2)];
        var _S525 : vec3<f32> = f_11.xyz - vec3<f32>(c0_4.w) * (force_3 * vec3<f32>(inv_mass_6) + cross(alpha_0, c0_4.xyz - frames_0[_S523].xyz));
        f_11.x = _S525.x;
        f_11.y = _S525.y;
        f_11.z = _S525.z;
        var _S526 : f32 = alpha_0.x;
        var _S527 : f32 = c1_2.w;
        var _S528 : f32 = alpha_0.y;
        var _S529 : f32 = c2_3.x;
        var _S530 : f32 = alpha_0.z;
        var _S531 : f32 = c2_3.y;
        var _S532 : vec3<f32> = t_14.xyz - vec3<f32>(c1_2.x * _S526 + _S527 * _S528 + _S529 * _S530, _S527 * _S526 + c1_2.y * _S528 + _S531 * _S530, _S529 * _S526 + _S531 * _S528 + c1_2.z * _S530);
        t_14.x = _S532.x;
        t_14.y = _S532.y;
        t_14.z = _S532.z;
    }
    var _S533 : f32;
    if(fixed_1)
    {
        _S533 = 1.0f;
    }
    else
    {
        _S533 = 0.0f;
    }
    f_11[i32(3)] = _S533;
    lp_0[_S520] = f_11;
    lp_0[_S521] = t_14;
    return;
}

fn adj_bond_0( at_26 : u32) -> u32
{
    return (bitcast<u32>((ext_0[params_0.ext_adj_0 + at_26 / u32(4)][at_26 % u32(4)])));
}

fn lp_intact_0( k_41 : u32) -> u32
{
    var range_0 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * k_41];
    var _S534 : u32 = (bitcast<u32>((range_0.x)));
    var _S535 : u32 = (bitcast<u32>((range_0.y)));
    var e_4 : u32 = u32(0);
    var intact_0 : u32 = u32(0);
    loop
    {
        if(e_4 < _S535)
        {
        }
        else
        {
            break;
        }
        var _S536 : u32;
        if(work_0[w_alive_0() + adj_bond_0(_S534 + e_4)] != u32(0))
        {
            _S536 = u32(1);
        }
        else
        {
            _S536 = u32(0);
        }
        var intact_1 : u32 = intact_0 + _S536;
        e_4 = e_4 + u32(1);
        intact_0 = intact_1;
    }
    return intact_0;
}

fn lp_l_0( k_42 : u32) -> u32
{
    return u32(22) * k_42 + u32(8);
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

fn lp_bond_0( i_18 : u32) -> LpBond_0
{
    var o_8 : u32 = u32(10) * i_18;
    var b_66 : LpBond_0;
    b_66.centroid_0 = ext_0[o_8].xyz;
    var _S537 : vec3<f32> = ext_0[o_8 + u32(1)].xyz;
    b_66.n_4 = _S537;
    var _S538 : vec3<f32> = ext_0[o_8 + u32(2)].xyz;
    b_66.t1_2 = _S538;
    b_66.t2_0 = cross(_S537, _S538);
    b_66.law_0.geom0_0 = ext_0[o_8 + u32(3)];
    b_66.law_0.geom1_0 = ext_0[o_8 + u32(4)];
    b_66.law_0.stiff0_0 = ext_0[o_8 + u32(5)];
    b_66.law_0.stiff1_0 = ext_0[o_8 + u32(6)];
    b_66.law_0.rebar0_0 = ext_0[o_8 + u32(7)];
    b_66.law_0.rebar1_0 = ext_0[o_8 + u32(8)];
    b_66.law_0.ids_3 = (bitcast<vec4<u32>>((ext_0[o_8 + u32(9)])));
    return b_66;
}

fn tri_0( i_19 : u32,  j_7 : u32) -> u32
{
    return i_19 * (i_19 + u32(1)) / u32(2) + j_7;
}

fn lp_factor_shifted_0( k_43 : u32,  d_24 : array<array<f32, i32(6)>, i32(6)>,  intact_2 : u32) -> bool
{
    var j2_0 : u32;
    var s_13 : f32;
    var s_14 : f32;
    var _S539 : f32 = max(d_24[i32(0)][i32(0)], max(d_24[i32(1)][i32(1)], d_24[i32(2)][i32(2)]));
    var _S540 : f32 = max(d_24[i32(3)][i32(3)], max(d_24[i32(4)][i32(4)], d_24[i32(5)][i32(5)]));
    var ok_0 : bool;
    if(!(_S539 > 0.0f))
    {
        ok_0 = true;
    }
    else
    {
        ok_0 = !(_S540 > 0.0f);
    }
    if(ok_0)
    {
        return false;
    }
    var attempt_0 : u32 = u32(0);
    var shift_0 : f32 = 9.99999997475242708e-07f;
    loop
    {
        if(attempt_0 < u32(4))
        {
        }
        else
        {
            break;
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
        ok_0 = true;
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
            var ok_1 : bool = ok_0;
            j2_0 = u32(0);
            loop
            {
                if(j2_0 <= i2_0)
                {
                }
                else
                {
                    break;
                }
                var _S541 : u32 = i2_0;
                var _S542 : u32 = j2_0;
                var _S543 : bool = i2_0 == j2_0;
                if(_S543)
                {
                    if(i2_0 < u32(3))
                    {
                        s_13 = _S539;
                    }
                    else
                    {
                        s_13 = _S540;
                    }
                    s_13 = d_24[_S541][_S542] + shift_0 * s_13;
                }
                else
                {
                    s_13 = d_24[_S541][_S542];
                }
                var p3_0 : u32 = u32(0);
                s_14 = s_13;
                loop
                {
                    if(p3_0 < j2_0)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var s_15 : f32 = s_14 - l_1[tri_0(i2_0, p3_0)] * l_1[tri_0(j2_0, p3_0)];
                    p3_0 = p3_0 + u32(1);
                    s_14 = s_15;
                }
                if(_S543)
                {
                    var _S544 : bool;
                    if(ok_1)
                    {
                        _S544 = s_14 > 0.0f;
                    }
                    else
                    {
                        _S544 = false;
                    }
                    l_1[tri_0(i2_0, i2_0)] = sqrt(max(s_14, 1.00000000317107685e-30f));
                    ok_1 = _S544;
                }
                else
                {
                    l_1[tri_0(i2_0, j2_0)] = s_14 / l_1[tri_0(j2_0, j2_0)];
                }
                j2_0 = j2_0 + u32(1);
            }
            var _S545 : u32 = i2_0 + u32(1);
            ok_0 = ok_1;
            i2_0 = _S545;
        }
        if(ok_0)
        {
            j2_0 = u32(0);
            loop
            {
                if(j2_0 < u32(6))
                {
                }
                else
                {
                    break;
                }
                var _S546 : u32 = u32(4) * j2_0;
                var _S547 : f32 = l_1[_S546];
                var _S548 : u32 = _S546 + u32(1);
                if(_S548 < u32(21))
                {
                    shift_0 = l_1[_S548];
                }
                else
                {
                    shift_0 = 0.0f;
                }
                var _S549 : u32 = _S546 + u32(2);
                if(_S549 < u32(21))
                {
                    s_13 = l_1[_S549];
                }
                else
                {
                    s_13 = 0.0f;
                }
                if(j2_0 == u32(5))
                {
                    s_14 = (bitcast<f32>((intact_2 + u32(1))));
                }
                else
                {
                    s_14 = l_1[_S546 + u32(3)];
                }
                lp_0[lp_l_0(k_43) + j2_0] = vec4<f32>(_S547, shift_0, s_13, s_14);
                j2_0 = j2_0 + u32(1);
            }
            return true;
        }
        var shift_1 : f32 = shift_0 * 10.0f;
        attempt_0 = attempt_0 + u32(1);
        shift_0 = shift_1;
    }
    return false;
}

fn lp_factor_0( k_44 : u32) -> bool
{
    var q_35 : u32;
    var a_12 : u32;
    var p2_0 : u32;
    var q2_0 : u32;
    var a2_0 : u32;
    var s_16 : f32;
    var intact_3 : u32 = lp_intact_0(k_44);
    var _S550 : u32 = lp_l_0(k_44);
    var _S551 : u32 = intact_3 + u32(1);
    if((bitcast<u32>((lp_0[_S550 + u32(5)].w))) == _S551)
    {
        return true;
    }
    var d_25 : array<array<f32, i32(6)>, i32(6)>;
    var p_18 : u32 = u32(0);
    loop
    {
        if(p_18 < u32(6))
        {
        }
        else
        {
            break;
        }
        q_35 = u32(0);
        loop
        {
            if(q_35 < u32(6))
            {
            }
            else
            {
                break;
            }
            d_25[p_18][q_35] = 0.0f;
            q_35 = q_35 + u32(1);
        }
        p_18 = p_18 + u32(1);
    }
    var _S552 : u32 = u32(4) * k_44;
    var range_1 : vec4<f32> = ext_0[params_0.ext_chunks_0 + _S552];
    var _S553 : u32 = (bitcast<u32>((range_1.x)));
    var _S554 : u32 = (bitcast<u32>((range_1.y)));
    q_35 = u32(0);
    loop
    {
        if(q_35 < _S554)
        {
        }
        else
        {
            break;
        }
        var i_20 : u32 = adj_bond_0(_S553 + q_35);
        if(work_0[w_alive_0() + i_20] == u32(0))
        {
            q_35 = q_35 + u32(1);
            continue;
        }
        var b_67 : LpBond_0 = lp_bond_0(i_20);
        var _S555 : vec3<f32> = b_67.centroid_0 - chunks_0[_S552].xyz;
        var _S556 : f32 = b_67.law_0.stiff0_0.y;
        var _S557 : array<f32, i32(6)> = array<f32, i32(6)>( _S556, _S556, b_67.law_0.stiff0_0.x, b_67.law_0.stiff0_0.z, b_67.law_0.stiff0_0.w, b_67.law_0.stiff1_0.x );
        var _S558 : array<vec3<f32>, i32(3)> = array<vec3<f32>, i32(3)>( b_67.t1_2, b_67.t2_0, b_67.n_4 );
        var j_8 : array<array<f32, i32(6)>, i32(6)>;
        a_12 = u32(0);
        loop
        {
            if(a_12 < u32(3))
            {
            }
            else
            {
                break;
            }
            var c_21 : vec3<f32> = cross(_S555, _S558[a_12]);
            var _S559 : f32 = _S558[a_12].x;
            j_8[a_12][i32(0)] = _S559;
            var _S560 : f32 = _S558[a_12].y;
            j_8[a_12][i32(1)] = _S560;
            var _S561 : f32 = _S558[a_12].z;
            j_8[a_12][i32(2)] = _S561;
            j_8[a_12][i32(3)] = c_21.x;
            j_8[a_12][i32(4)] = c_21.y;
            j_8[a_12][i32(5)] = c_21.z;
            j_8[u32(3) + a_12][i32(0)] = 0.0f;
            j_8[u32(3) + a_12][i32(1)] = 0.0f;
            j_8[u32(3) + a_12][i32(2)] = 0.0f;
            j_8[u32(3) + a_12][i32(3)] = _S559;
            j_8[u32(3) + a_12][i32(4)] = _S560;
            j_8[u32(3) + a_12][i32(5)] = _S561;
            a_12 = a_12 + u32(1);
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
                s_16 = 0.0f;
                loop
                {
                    if(a2_0 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var s_17 : f32 = s_16 + j_8[a2_0][p2_0] * _S557[a2_0] * j_8[a2_0][q2_0];
                    a2_0 = a2_0 + u32(1);
                    s_16 = s_17;
                }
                d_25[p2_0][q2_0] = d_25[p2_0][q2_0] + s_16;
                q2_0 = q2_0 + u32(1);
            }
            p2_0 = p2_0 + u32(1);
        }
        q_35 = q_35 + u32(1);
    }
    var l_2 : array<f32, i32(21)>;
    a_12 = u32(0);
    loop
    {
        if(a_12 < u32(21))
        {
        }
        else
        {
            break;
        }
        l_2[a_12] = 0.0f;
        a_12 = a_12 + u32(1);
    }
    var ok_2 : bool = true;
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
        var ok_3 : bool = ok_2;
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
            s_16 = d_25[p2_0][q2_0];
            loop
            {
                if(a2_0 < q2_0)
                {
                }
                else
                {
                    break;
                }
                var s_18 : f32 = s_16 - l_2[tri_0(p2_0, a2_0)] * l_2[tri_0(q2_0, a2_0)];
                a2_0 = a2_0 + u32(1);
                s_16 = s_18;
            }
            if(p2_0 == q2_0)
            {
                var _S562 : bool;
                if(ok_3)
                {
                    _S562 = s_16 > 0.0f;
                }
                else
                {
                    _S562 = false;
                }
                l_2[tri_0(p2_0, p2_0)] = sqrt(max(s_16, 1.00000000317107685e-30f));
                ok_3 = _S562;
            }
            else
            {
                l_2[tri_0(p2_0, q2_0)] = s_16 / l_2[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + u32(1);
        }
        var _S563 : u32 = p2_0 + u32(1);
        ok_2 = ok_3;
        p2_0 = _S563;
    }
    if(!ok_2)
    {
        var _S564 : bool = lp_factor_shifted_0(k_44, d_25, intact_3);
        return _S564;
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
        var _S565 : u32 = u32(4) * q2_0;
        var _S566 : f32 = l_2[_S565];
        var _S567 : u32 = _S565 + u32(1);
        if(_S567 < u32(21))
        {
            s_16 = l_2[_S567];
        }
        else
        {
            s_16 = 0.0f;
        }
        var _S568 : u32 = _S565 + u32(2);
        var _S569 : f32;
        if(_S568 < u32(21))
        {
            _S569 = l_2[_S568];
        }
        else
        {
            _S569 = 0.0f;
        }
        var _S570 : f32;
        if(q2_0 == u32(5))
        {
            _S570 = (bitcast<f32>((_S551)));
        }
        else
        {
            _S570 = l_2[_S565 + u32(3)];
        }
        lp_0[_S550 + q2_0] = vec4<f32>(_S566, s_16, _S569, _S570);
        q2_0 = q2_0 + u32(1);
    }
    return true;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_block(@builtin(global_invocation_id) id_40 : vec3<u32>)
{
    var k_45 : u32 = id_40.x;
    var _S571 : bool;
    if(k_45 >= (params_0.chunk_count_0))
    {
        _S571 = true;
    }
    else
    {
        _S571 = (lp_0[lp_f_0(k_45)].w) != 0.0f;
    }
    if(_S571)
    {
        return;
    }
    var _S572 : bool = lp_factor_0(k_45);
    if(!_S572)
    {
        lp_0[lp_f_0(k_45)][i32(3)] = 1.0f;
        var _S573 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(64));
    }
    return;
}

fn st_x_0( k_46 : u32,  state_1 : u32,  copy_0 : u32) -> u32
{
    return u32(22) * k_46 + state_1 + u32(2) * copy_0;
}

fn st_d_0( k_47 : u32,  state_2 : u32) -> u32
{
    return u32(22) * k_47 + state_2 + u32(4);
}

fn lp_side_0( i_21 : u32) -> vec4<f32>
{
    return ext_0[params_0.ext_sides_0 + i_21];
}

fn st_sweep_0( k_48 : u32,  state_3 : u32,  from_2 : u32,  c1_3 : f32,  c2_4 : f32)
{
    var ka_0 : u32;
    var kb_0 : u32;
    var s_19 : f32;
    var to_1 : u32 = u32(1) - from_2;
    var _S574 : u32 = st_x_0(k_48, state_3, from_2);
    var x0_0 : vec4<f32> = lp_0[_S574];
    var x1_0 : vec4<f32> = lp_0[_S574 + u32(1)];
    var _S575 : u32 = st_f_0(k_48, state_3);
    var f_12 : vec4<f32> = lp_0[_S575];
    if((lp_0[_S575].w) != 0.0f)
    {
        var _S576 : u32 = st_x_0(k_48, state_3, to_1);
        lp_0[_S576] = x0_0;
        lp_0[_S576 + u32(1)] = x1_0;
        var _S577 : u32 = st_d_0(k_48, state_3);
        var _S578 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S577] = _S578;
        lp_0[_S577 + u32(1)] = _S578;
        return;
    }
    var _S579 : f32 = f_12.y;
    var _S580 : f32 = f_12.z;
    var r_23 : array<f32, i32(6)>;
    r_23[i32(0)] = f_12.x;
    r_23[i32(1)] = _S579;
    r_23[i32(2)] = _S580;
    r_23[i32(3)] = 0.0f;
    r_23[i32(4)] = 0.0f;
    r_23[i32(5)] = 0.0f;
    var t_15 : vec3<f32> = lp_0[_S575 + u32(1)].xyz;
    r_23[i32(3)] = t_15.x;
    r_23[i32(4)] = t_15.y;
    r_23[i32(5)] = t_15.z;
    var range_2 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * k_48];
    var _S581 : u32 = (bitcast<u32>((range_2.x)));
    var _S582 : u32 = (bitcast<u32>((range_2.y)));
    var e_5 : u32 = u32(0);
    loop
    {
        if(e_5 < _S582)
        {
        }
        else
        {
            break;
        }
        var _S583 : u32 = u32(5) * (_S581 + e_5);
        var s0_4 : vec4<f32> = lp_side_0(_S583);
        if(work_0[w_alive_0() + (bitcast<u32>((s0_4.w)))] == u32(0))
        {
            e_5 = e_5 + u32(1);
            continue;
        }
        var s1_5 : vec4<f32> = lp_side_0(_S583 + u32(1));
        var s2_0 : vec4<f32> = lp_side_0(_S583 + u32(2));
        var s3_0 : vec4<f32> = lp_side_0(_S583 + u32(3));
        var s4_0 : vec4<f32> = lp_side_0(_S583 + u32(4));
        var first_side_0 : bool = (s4_0.w) != 0.0f;
        if(first_side_0)
        {
            ka_0 = k_48;
        }
        else
        {
            ka_0 = (bitcast<u32>((s1_5.w)));
        }
        if(first_side_0)
        {
            kb_0 = (bitcast<u32>((s1_5.w)));
        }
        else
        {
            kb_0 = k_48;
        }
        var ra_1 : vec3<f32> = s0_4.xyz;
        var rb_1 : vec3<f32> = s1_5.xyz;
        var t1_3 : vec3<f32> = s2_0.xyz;
        var n_5 : vec3<f32> = s3_0.xyz;
        var t2_1 : vec3<f32> = cross(n_5, t1_3);
        var _S584 : u32 = st_x_0(ka_0, state_3, from_2);
        var _S585 : u32 = st_x_0(kb_0, state_3, from_2);
        var _S586 : vec3<f32> = lp_0[_S585 + u32(1)].xyz;
        var _S587 : vec3<f32> = lp_0[_S584 + u32(1)].xyz;
        var delta_0 : vec3<f32> = lp_0[_S585].xyz + cross(_S586, rb_1) - (lp_0[_S584].xyz + cross(_S587, ra_1));
        var phi_0 : vec3<f32> = _S586 - _S587;
        var _S588 : f32 = s2_0.w;
        var fb_0 : vec3<f32> = t1_3 * vec3<f32>((_S588 * dot(delta_0, t1_3))) + t2_1 * vec3<f32>((_S588 * dot(delta_0, t2_1))) + n_5 * vec3<f32>((s3_0.w * dot(delta_0, n_5)));
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
        var _S589 : vec3<f32>;
        if(first_side_0)
        {
            _S589 = mb_0;
        }
        else
        {
            _S589 = (vec3<f32>(0) - mb_0);
        }
        var lm_0 : vec3<f32> = _S589 + cross(arm_0, lf_0);
        r_23[i32(0)] = r_23[i32(0)] + lf_0.x;
        r_23[i32(1)] = r_23[i32(1)] + lf_0.y;
        r_23[i32(2)] = r_23[i32(2)] + lf_0.z;
        r_23[i32(3)] = r_23[i32(3)] + lm_0.x;
        r_23[i32(4)] = r_23[i32(4)] + lm_0.y;
        r_23[i32(5)] = r_23[i32(5)] + lm_0.z;
        e_5 = e_5 + u32(1);
    }
    var l_3 : array<f32, i32(21)>;
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
        l_3[ka_0] = 0.0f;
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
        var v_15 : vec4<f32> = lp_0[lp_l_0(k_48) + kb_0];
        var _S590 : u32 = u32(4) * kb_0;
        l_3[_S590] = lp_0[lp_l_0(k_48) + kb_0].x;
        var _S591 : u32 = _S590 + u32(1);
        if(_S591 < u32(21))
        {
            l_3[_S591] = v_15.y;
        }
        var _S592 : u32 = _S590 + u32(2);
        if(_S592 < u32(21))
        {
            l_3[_S592] = v_15.z;
        }
        var _S593 : u32 = _S590 + u32(3);
        if(_S593 < u32(21))
        {
            l_3[_S593] = v_15.w;
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
    var i2_1 : u32 = u32(0);
    loop
    {
        if(i2_1 < u32(6))
        {
        }
        else
        {
            break;
        }
        var p_19 : u32 = u32(0);
        s_19 = r_23[i2_1];
        loop
        {
            if(p_19 < i2_1)
            {
            }
            else
            {
                break;
            }
            var s_20 : f32 = s_19 - l_3[tri_0(i2_1, p_19)] * y_3[p_19];
            p_19 = p_19 + u32(1);
            s_19 = s_20;
        }
        y_3[i2_1] = s_19 / l_3[tri_0(i2_1, i2_1)];
        i2_1 = i2_1 + u32(1);
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
        s_19 = y_3[i3_0];
        loop
        {
            if(p2_1 < i32(6))
            {
            }
            else
            {
                break;
            }
            var s_21 : f32 = s_19 - l_3[tri_0(u32(p2_1), u32(i3_0))] * z_4[p2_1];
            p2_1 = p2_1 + i32(1);
            s_19 = s_21;
        }
        var _S594 : u32 = u32(i3_0);
        z_4[i3_0] = s_19 / l_3[tri_0(_S594, _S594)];
        i3_0 = i3_0 - i32(1);
    }
    var _S595 : u32 = st_d_0(k_48, state_3);
    var d0_0 : vec4<f32> = lp_0[_S595];
    var _S596 : u32 = _S595 + u32(1);
    var d1_0 : vec4<f32> = lp_0[_S596];
    var _S597 : vec3<f32> = vec3<f32>(c1_3);
    var _S598 : vec3<f32> = vec3<f32>(c2_4);
    var _S599 : vec3<f32> = _S597 * lp_0[_S595].xyz + _S598 * vec3<f32>(z_4[i32(0)], z_4[i32(1)], z_4[i32(2)]);
    d0_0.x = _S599.x;
    d0_0.y = _S599.y;
    d0_0.z = _S599.z;
    var _S600 : vec3<f32> = _S597 * lp_0[_S596].xyz + _S598 * vec3<f32>(z_4[i32(3)], z_4[i32(4)], z_4[i32(5)]);
    d1_0.x = _S600.x;
    d1_0.y = _S600.y;
    d1_0.z = _S600.z;
    lp_0[_S595] = d0_0;
    lp_0[_S596] = d1_0;
    var _S601 : u32 = st_x_0(k_48, state_3, to_1);
    lp_0[_S601] = vec4<f32>(x0_0.xyz + d0_0.xyz, 0.0f);
    lp_0[_S601 + u32(1)] = vec4<f32>(x1_0.xyz + d1_0.xyz, 0.0f);
    return;
}

fn lp_sweep_0( k_49 : u32,  from_3 : u32)
{
    st_sweep_0(k_49, u32(0), from_3, params_0.lp_c1_0, params_0.lp_c2_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sweep_even(@builtin(global_invocation_id) id_41 : vec3<u32>)
{
    var _S602 : u32 = id_41.x;
    if(_S602 < (params_0.chunk_count_0))
    {
        lp_sweep_0(_S602, u32(0));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sweep_odd(@builtin(global_invocation_id) id_42 : vec3<u32>)
{
    var _S603 : u32 = id_42.x;
    if(_S603 < (params_0.chunk_count_0))
    {
        lp_sweep_0(_S603, u32(1));
    }
    return;
}

fn st_force_0( b_68 : LpBond_0,  ka_1 : u32,  kb_1 : u32,  state_4 : u32,  c_22 : u32,  q_lin_0 : ptr<function, vec3<f32>>,  q_ang_0 : ptr<function, vec3<f32>>)
{
    var _S604 : u32 = st_x_0(ka_1, state_4, c_22);
    var _S605 : u32 = st_x_0(kb_1, state_4, c_22);
    var _S606 : vec3<f32> = lp_0[_S605 + u32(1)].xyz;
    var _S607 : vec3<f32> = lp_0[_S604 + u32(1)].xyz;
    var delta_1 : vec3<f32> = lp_0[_S605].xyz + cross(_S606, b_68.centroid_0 - chunks_0[u32(4) * kb_1].xyz) - (lp_0[_S604].xyz + cross(_S607, b_68.centroid_0 - chunks_0[u32(4) * ka_1].xyz));
    var phi_1 : vec3<f32> = _S606 - _S607;
    var kt_0 : f32 = b_68.law_0.stiff1_0.x;
    var _S608 : f32 = b_68.law_0.stiff0_0.y;
    (*q_lin_0) = vec3<f32>(_S608 * dot(delta_1, b_68.t1_2), _S608 * dot(delta_1, b_68.t2_0), b_68.law_0.stiff0_0.x * dot(delta_1, b_68.n_4));
    (*q_ang_0) = vec3<f32>(b_68.law_0.stiff0_0.z * dot(phi_1, b_68.t1_2), b_68.law_0.stiff0_0.w * dot(phi_1, b_68.t2_0), kt_0 * dot(phi_1, b_68.n_4));
    return;
}

fn st_force_1( b_69 : LpBond_0,  ka_2 : u32,  kb_2 : u32,  state_5 : u32,  c_23 : u32,  q_lin_1 : ptr<function, vec3<f32>>,  q_ang_1 : ptr<function, vec3<f32>>)
{
    var _S609 : u32 = st_x_0(ka_2, state_5, c_23);
    var _S610 : u32 = st_x_0(kb_2, state_5, c_23);
    var _S611 : vec3<f32> = lp_0[_S610 + u32(1)].xyz;
    var _S612 : vec3<f32> = lp_0[_S609 + u32(1)].xyz;
    var delta_2 : vec3<f32> = lp_0[_S610].xyz + cross(_S611, b_69.centroid_0 - chunks_0[u32(4) * kb_2].xyz) - (lp_0[_S609].xyz + cross(_S612, b_69.centroid_0 - chunks_0[u32(4) * ka_2].xyz));
    var phi_2 : vec3<f32> = _S611 - _S612;
    var kt_1 : f32 = b_69.law_0.stiff1_0.x;
    var _S613 : f32 = b_69.law_0.stiff0_0.y;
    (*q_lin_1) = vec3<f32>(_S613 * dot(delta_2, b_69.t1_2), _S613 * dot(delta_2, b_69.t2_0), b_69.law_0.stiff0_0.x * dot(delta_2, b_69.n_4));
    (*q_ang_1) = vec3<f32>(b_69.law_0.stiff0_0.z * dot(phi_2, b_69.t1_2), b_69.law_0.stiff0_0.w * dot(phi_2, b_69.t2_0), kt_1 * dot(phi_2, b_69.n_4));
    return;
}

fn st_force_2( b_70 : LpBond_0,  ka_3 : u32,  kb_3 : u32,  state_6 : u32,  c_24 : u32,  q_lin_2 : ptr<function, vec3<f32>>,  q_ang_2 : ptr<function, vec3<f32>>)
{
    var _S614 : u32 = st_x_0(ka_3, state_6, c_24);
    var _S615 : u32 = st_x_0(kb_3, state_6, c_24);
    var _S616 : vec3<f32> = lp_0[_S615 + u32(1)].xyz;
    var _S617 : vec3<f32> = lp_0[_S614 + u32(1)].xyz;
    var delta_3 : vec3<f32> = lp_0[_S615].xyz + cross(_S616, b_70.centroid_0 - chunks_0[u32(4) * kb_3].xyz) - (lp_0[_S614].xyz + cross(_S617, b_70.centroid_0 - chunks_0[u32(4) * ka_3].xyz));
    var phi_3 : vec3<f32> = _S616 - _S617;
    var kt_2 : f32 = b_70.law_0.stiff1_0.x;
    var _S618 : f32 = b_70.law_0.stiff0_0.y;
    (*q_lin_2) = vec3<f32>(_S618 * dot(delta_3, b_70.t1_2), _S618 * dot(delta_3, b_70.t2_0), b_70.law_0.stiff0_0.x * dot(delta_3, b_70.n_4));
    (*q_ang_2) = vec3<f32>(b_70.law_0.stiff0_0.z * dot(phi_3, b_70.t1_2), b_70.law_0.stiff0_0.w * dot(phi_3, b_70.t2_0), kt_2 * dot(phi_3, b_70.n_4));
    return;
}

fn lp_force_0( b_71 : LpBond_0,  ka_4 : u32,  kb_4 : u32,  c_25 : u32,  q_lin_3 : ptr<function, vec3<f32>>,  q_ang_3 : ptr<function, vec3<f32>>)
{
    st_force_1(b_71, ka_4, kb_4, u32(0), c_25, &((*q_lin_3)), &((*q_ang_3)));
    return;
}

fn lp_force_1( b_72 : LpBond_0,  ka_5 : u32,  kb_5 : u32,  c_26 : u32,  q_lin_4 : ptr<function, vec3<f32>>,  q_ang_4 : ptr<function, vec3<f32>>)
{
    st_force_2(b_72, ka_5, kb_5, u32(0), c_26, &((*q_lin_4)), &((*q_ang_4)));
    return;
}

fn fsqrt_0( a_13 : f32) -> f32
{
    return sqrt(a_13);
}

struct Measures_0
{
     tension_0 : f32,
     shear_0 : f32,
     normal_compression_0 : f32,
     compression_0 : f32,
     compressive_force_0 : f32,
};

fn stress_measures_0( b_73 : JointBond_0,  q_lin_5 : vec3<f32>,  q_ang_5 : vec3<f32>) -> Measures_0
{
    var area_0 : f32 = b_73.geom0_0.x;
    var _S619 : f32 = q_lin_5.z;
    var axial_0 : f32 = fdiv_0(_S619, area_0);
    var bending_0 : f32 = fdiv_0(abs(q_ang_5.x), b_73.geom1_0.x) + fdiv_0(abs(q_ang_5.y), b_73.geom1_0.y);
    var _S620 : f32 = q_lin_5.x;
    var _S621 : f32 = q_lin_5.y;
    var shear_1 : f32 = fdiv_0(fsqrt_0(_S620 * _S620 + _S621 * _S621), area_0) + fdiv_0(abs(q_ang_5.z), b_73.geom0_0.w);
    var m_3 : Measures_0;
    m_3.tension_0 = axial_0 + bending_0;
    m_3.shear_0 = shear_1;
    var _S622 : f32 = - axial_0;
    m_3.normal_compression_0 = max(_S622, 0.0f);
    m_3.compression_0 = _S622 + bending_0;
    m_3.compressive_force_0 = max(- _S619, 0.0f);
    return m_3;
}

fn lp_material_0( m_4 : u32) -> JointMaterial_0
{
    var o_9 : u32 = params_0.ext_mat_0 + u32(5) * m_4;
    var j_9 : JointMaterial_0;
    j_9.strength_0 = ext_0[o_9];
    j_9.energy_0 = ext_0[o_9 + u32(1)];
    j_9.dif_0 = ext_0[o_9 + u32(2)];
    j_9.misc_0 = ext_0[o_9 + u32(3)];
    j_9.kind_flags_0 = (bitcast<vec4<u32>>((ext_0[o_9 + u32(4)])));
    return j_9;
}

fn infinity_0() -> f32
{
    return float_bits_0(u32(2139095040));
}

fn failure_indices_0( mat_1 : JointMaterial_0,  b_74 : JointBond_0,  m_5 : Measures_0,  multiplier_0 : f32) -> vec4<f32>
{
    var fc_2 : f32 = mat_1.strength_0.y * multiplier_0;
    var _S623 : f32 = min(mat_1.strength_0.z * multiplier_0 + mat_1.strength_0.w * m_5.normal_compression_0, mat_1.energy_0.x * multiplier_0);
    var idx_0 : vec4<f32>;
    idx_0[i32(0)] = max(fdiv_0(m_5.tension_0, mat_1.strength_0.x * multiplier_0), 0.0f);
    var _S624 : f32;
    if(_S623 > 0.0f)
    {
        _S624 = fdiv_0(m_5.shear_0, _S623);
    }
    else
    {
        _S624 = infinity_0();
    }
    idx_0[i32(1)] = _S624;
    idx_0[i32(2)] = max(fdiv_0(m_5.compression_0, fc_2), 0.0f);
    var _S625 : f32 = b_74.stiff1_0.y;
    if(_S625 > 0.0f)
    {
        _S624 = fdiv_0(m_5.compressive_force_0, _S625);
    }
    else
    {
        _S624 = 0.0f;
    }
    idx_0[i32(3)] = _S624;
    return idx_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_breaks(@builtin(global_invocation_id) id_43 : vec3<u32>)
{
    var i_22 : u32 = id_43.x;
    var _S626 : bool;
    if(i_22 >= (params_0.bond_count_0))
    {
        _S626 = true;
    }
    else
    {
        _S626 = work_0[w_alive_0() + i_22] == u32(0);
    }
    if(_S626)
    {
        return;
    }
    var bond_10 : vec4<u32> = bonds_0[i_22];
    var b_75 : LpBond_0 = lp_bond_0(i_22);
    var _S627 : u32 = bond_10.x;
    var q_lin_6 : vec3<f32>;
    var q_ang_6 : vec3<f32>;
    lp_force_1(b_75, _S627, bond_10.y, u32(0), &(q_lin_6), &(q_ang_6));
    var idx_1 : vec4<f32> = failure_indices_0(lp_material_0(b_75.law_0.ids_3.x), b_75.law_0, stress_measures_0(b_75.law_0, q_lin_6, q_ang_6), b_75.law_0.geom1_0.w);
    var _S628 : f32 = max(max(idx_1.x, idx_1.y), max(idx_1.z, idx_1.w));
    var _S629 : u32 = a_counters_0();
    var _S630 : u32 = atomicMax(&(atom_0[_S629 + u32(15)]), (bitcast<u32>((max(_S628, 0.0f)))));
    if(_S628 >= 1.0f)
    {
        work_0[w_alive_0() + i_22] = u32(0);
        var _S631 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S627)]), u32(1));
        var _S632 : u32 = atomicAdd(&(atom_0[_S629]), u32(1));
        var _S633 : u32 = atomicAdd(&(atom_0[_S629 + u32(1)]), u32(1));
        event_broken_0(i_22);
    }
    return;
}

fn lp_view_0( k_50 : u32) -> u32
{
    return u32(22) * params_0.chunk_count_0 + u32(7) * params_0.body_count_0 + k_50;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_view(@builtin(global_invocation_id) id_44 : vec3<u32>)
{
    var k_51 : u32 = id_44.x;
    if(k_51 >= (params_0.chunk_count_0))
    {
        return;
    }
    var range_3 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * k_51];
    var _S634 : u32 = (bitcast<u32>((range_3.x)));
    var _S635 : u32 = (bitcast<u32>((range_3.y)));
    var worst_0 : vec4<f32> = vec4<f32>(0.0f);
    var e_6 : u32 = u32(0);
    loop
    {
        if(e_6 < _S635)
        {
        }
        else
        {
            break;
        }
        var i_23 : u32 = (bitcast<u32>((lp_side_0(u32(5) * (_S634 + e_6)).w)));
        if(work_0[w_alive_0() + i_23] == u32(0))
        {
            e_6 = e_6 + u32(1);
            continue;
        }
        var bond_11 : vec4<u32> = bonds_0[i_23];
        var b_76 : LpBond_0 = lp_bond_0(i_23);
        var q_lin_7 : vec3<f32>;
        var q_ang_7 : vec3<f32>;
        lp_force_1(b_76, bond_11.x, bond_11.y, u32(0), &(q_lin_7), &(q_ang_7));
        worst_0 = max(worst_0, failure_indices_0(lp_material_0(b_76.law_0.ids_3.x), b_76.law_0, stress_measures_0(b_76.law_0, q_lin_7, q_ang_7), b_76.law_0.geom1_0.w));
        e_6 = e_6 + u32(1);
    }
    lp_0[lp_view_0(k_51)] = worst_0;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_first_reset(@builtin(global_invocation_id) id_45 : vec3<u32>)
{
    var _S636 : u32 = id_45.x;
    if(_S636 == u32(0))
    {
        atomicStore(&(atom_0[a_il_cursor_0()]), u32(0));
    }
    if(_S636 < (params_0.shot_count_0))
    {
        atomicStore(&(atom_0[a_first_0(_S636)]), u32(2139095040));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_first(@builtin(global_invocation_id) id_46 : vec3<u32>)
{
    var b_77 : u32 = id_46.x;
    var _S637 : bool;
    if(b_77 >= (params_0.body_count_0))
    {
        _S637 = true;
    }
    else
    {
        _S637 = (params_0.shot_count_0) == u32(0);
    }
    if(_S637)
    {
        return;
    }
    var shot_11 : u32;
    var _S638 : bool = struck_2(b_77, &(shot_11));
    if(!_S638)
    {
        return;
    }
    var _S639 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_77 + u32(6)]));
    var _S640 : u32 = atomicMin(&(atom_0[a_first_0(shot_11)]), _S639);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_begin(@builtin(global_invocation_id) id_47 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_begin_1(id_47.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_tube_count(@builtin(global_invocation_id) id_48 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_tube_count_1(id_48.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_tube_base(@builtin(global_invocation_id) id_49 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_tube_base_1(id_49.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_tube_collect(@builtin(global_invocation_id) id_50 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_tube_collect_1(id_50.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_run(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var b_78 : u32 = group_1.y * u32(65535) + group_1.x;
    var _S641 : bool;
    if((params_0.shot_count_0) > u32(0))
    {
        _S641 = b_78 < (params_0.body_count_0);
    }
    else
    {
        _S641 = false;
    }
    if(_S641)
    {
        il_run_1(b_78, thread_4.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_strike(@builtin(global_invocation_id) id_51 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_strike_1(id_51.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_mark(@builtin(global_invocation_id) id_52 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_mark_1(id_52.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_carry(@builtin(global_invocation_id) id_53 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_carry_1(id_53.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_breaks(@builtin(global_invocation_id) id_54 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_breaks_1(id_54.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_begin(@builtin(global_invocation_id) id_55 : vec3<u32>)
{
    il_begin_0(id_55.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_tube_count(@builtin(global_invocation_id) id_56 : vec3<u32>)
{
    il_tube_count_0(id_56.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_tube_base(@builtin(global_invocation_id) id_57 : vec3<u32>)
{
    il_tube_base_0(id_57.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_tube_collect(@builtin(global_invocation_id) id_58 : vec3<u32>)
{
    il_tube_collect_0(id_58.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_run(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    il_run_0(work_0[w_cs_bodies_0() + group_2.x], thread_5.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_strike(@builtin(global_invocation_id) id_59 : vec3<u32>)
{
    il_strike_0(id_59.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_mark(@builtin(global_invocation_id) id_60 : vec3<u32>)
{
    il_mark_0(id_60.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_carry(@builtin(global_invocation_id) id_61 : vec3<u32>)
{
    il_carry_0(id_61.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_breaks(@builtin(global_invocation_id) id_62 : vec3<u32>)
{
    il_breaks_0(id_62.x);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_cs_begin(@builtin(global_invocation_id) id_63 : vec3<u32>)
{
    var _S642 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S642 + u32(18)]), u32(0));
    atomicStore(&(atom_0[_S642 + u32(19)]), u32(0));
    atomicStore(&(atom_0[a_il_cursor_0()]), u32(0));
    var _S643 : u32 = groups_of_0(cs_rows_0(), u32(64));
    set_args_0(u32(7), _S643);
    return;
}

fn cs_side_1( r_24 : u32,  side_6 : u32,  k_52 : ptr<function, u32>,  s_22 : ptr<function, u32>,  ob_5 : ptr<function, u32>,  co_5 : ptr<function, u32>,  p_20 : ptr<function, vec3<f32>>,  u_10 : ptr<function, vec3<f32>>,  v_16 : ptr<function, f32>,  m_eff_7 : ptr<function, f32>,  t_in_10 : ptr<function, f32>) -> bool
{
    (*t_in_10) = 0.0f;
    (*k_52) = u32(4294967295);
    (*s_22) = u32(4294967295);
    (*ob_5) = u32(4294967295);
    (*co_5) = u32(4294967295);
    var _S644 : vec3<f32> = vec3<f32>(0.0f);
    (*p_20) = _S644;
    (*u_10) = _S644;
    (*v_16) = 0.0f;
    (*m_eff_7) = 0.0f;
    var row_2 : u32 = cs_row_0(r_24);
    var ids_4 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_2])));
    if((ids_4.w) == u32(0))
    {
        return false;
    }
    var _S645 : bool = side_6 == u32(0);
    var cm_1 : u32;
    if(_S645)
    {
        cm_1 = ids_4.x;
    }
    else
    {
        cm_1 = ids_4.y;
    }
    var _S646 : u32;
    if(_S645)
    {
        _S646 = ids_4.y;
    }
    else
    {
        _S646 = ids_4.x;
    }
    (*co_5) = _S646;
    var _S647 : u32 = collider_info_0[u32(2) * cm_1].x;
    (*k_52) = _S647;
    var _S648 : bool;
    if(_S647 == u32(4294967295))
    {
        _S648 = true;
    }
    else
    {
        _S648 = (*k_52) >= (params_0.chunk_count_0);
    }
    if(_S648)
    {
        _S648 = true;
    }
    else
    {
        _S648 = (collider_info_0[u32(2) * (*co_5)].z) != u32(2);
    }
    if(_S648)
    {
        return false;
    }
    (*s_22) = body_of_0((*k_52));
    var _S649 : u32 = (bitcast<u32>((colliders_0[u32(4) * (*co_5)].x)));
    (*ob_5) = _S649;
    if(_S649 == (*s_22))
    {
        _S648 = true;
    }
    else
    {
        _S648 = (*ob_5) >= (params_0.max_bodies_0);
    }
    if(_S648)
    {
        return false;
    }
    var launch_2 : u32 = collider_info_0[u32(2) * (*co_5) + u32(1)].w;
    if(launch_2 != u32(0))
    {
        _S648 = work_0[w_marks_0() + (*k_52)] == launch_2;
    }
    else
    {
        _S648 = false;
    }
    if(_S648)
    {
        return false;
    }
    var bs_1 : u32 = u32(7) * (*s_22);
    var bo_1 : u32 = u32(7) * (*ob_5);
    var inv_o_1 : f32 = bodies_0[bo_1].w;
    if(!(inv_o_1 > 0.0f))
    {
        return false;
    }
    var n_6 : vec3<f32> = engine_contacts_0[row_2 + u32(1)].xyz;
    var _S650 : vec3<f32>;
    if(_S645)
    {
        _S650 = (vec3<f32>(0) - n_6);
    }
    else
    {
        _S650 = n_6;
    }
    (*u_10) = _S650;
    var _S651 : Quat_0 = quat_of_0(bodies_0[bs_1 + u32(1)]);
    if(_S645)
    {
        _S650 = engine_contacts_0[row_2 + u32(2)].xyz;
    }
    else
    {
        _S650 = engine_contacts_0[row_2 + u32(3)].xyz;
    }
    var rs_1 : vec3<f32> = rotate_0(_S651, _S650);
    var _S652 : Quat_0 = quat_of_0(bodies_0[bo_1 + u32(1)]);
    if(_S645)
    {
        _S650 = engine_contacts_0[row_2 + u32(3)].xyz;
    }
    else
    {
        _S650 = engine_contacts_0[row_2 + u32(2)].xyz;
    }
    var ro_1 : vec3<f32> = rotate_0(_S652, _S650);
    var _S653 : vec3<f32> = bodies_0[bs_1].xyz + rs_1;
    (*p_20) = _S653;
    var gap_1 : f32 = dot(_S653 - (bodies_0[bo_1].xyz + ro_1), (*u_10));
    var _S654 : f32 = dot(bodies_0[bo_1 + u32(3)].xyz + cross(bodies_0[bo_1 + u32(4)].xyz, ro_1) - (bodies_0[bs_1 + u32(3)].xyz + cross(bodies_0[bs_1 + u32(4)].xyz, rs_1)), (*u_10));
    (*v_16) = _S654;
    if(!(_S654 > (params_0.strike_speed_0)))
    {
        return false;
    }
    var w_8 : f32 = bodies_0[bs_1].w + cs_angular_0((*s_22), rs_1, (*u_10)) + inv_o_1 + cs_angular_0((*ob_5), ro_1, (*u_10));
    if(!(w_8 > 0.0f))
    {
        return false;
    }
    (*m_eff_7) = 1.0f / w_8;
    (*t_in_10) = clamp(gap_1 / (*v_16), 0.0f, params_0.dt_0);
    return true;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_contacts(@builtin(global_invocation_id) id_64 : vec3<u32>)
{
    var r_25 : u32 = id_64.x;
    if(r_25 >= (cs_rows_0()))
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
        var k_53 : u32;
        var s_23 : u32;
        var ob_6 : u32;
        var co_6 : u32;
        var p_21 : vec3<f32>;
        var u_11 : vec3<f32>;
        var v_17 : f32;
        var m_eff_8 : f32;
        var t_in_11 : f32;
        var _S655 : bool = cs_side_1(r_25, side_7, &(k_53), &(s_23), &(ob_6), &(co_6), &(p_21), &(u_11), &(v_17), &(m_eff_8), &(t_in_11));
        if(_S655)
        {
            var _S656 : u32 = atomicMax(&(atom_0[a_cs_0(s_23)]), (bitcast<u32>((m_eff_8 * v_17))));
            var _S657 : u32 = atomicExchange(&(atom_0[a_cs_0(s_23) + u32(8)]), u32(1));
            if(_S657 == u32(0))
            {
                var _S658 : u32 = w_cs_bodies_0();
                var _S659 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(18)]), u32(1));
                work_0[_S658 + _S659] = s_23;
            }
        }
        side_7 = side_7 + u32(1);
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_cs_gate(@builtin(global_invocation_id) id_65 : vec3<u32>)
{
    var struck_3 : u32 = atomicLoad(&(atom_0[a_counters_0() + u32(18)]));
    var any_0 : bool = struck_3 > u32(0);
    var _S660 : u32;
    if(any_0)
    {
        var _S661 : u32 = groups_of_0(cs_rows_0(), u32(64));
        _S660 = _S661;
    }
    else
    {
        _S660 = u32(0);
    }
    set_args_0(u32(8), _S660);
    var _S662 : u32 = groups_of_0(struck_3, u32(64));
    set_args_0(u32(9), _S662);
    set_args_0(u32(6), struck_3);
    if(any_0)
    {
        var _S663 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S660 = _S663;
    }
    else
    {
        _S660 = u32(0);
    }
    set_args_0(u32(10), _S660);
    if(any_0)
    {
        var _S664 : u32 = groups_of_0(params_0.bond_count_0, u32(64));
        _S660 = _S664;
    }
    else
    {
        _S660 = u32(0);
    }
    set_args_0(u32(11), _S660);
    if(any_0)
    {
        var _S665 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S660 = _S665;
    }
    else
    {
        _S660 = u32(0);
    }
    set_args_0(u32(12), _S660);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_a(@builtin(global_invocation_id) id_66 : vec3<u32>)
{
    cs_pick_3(id_66.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_b(@builtin(global_invocation_id) id_67 : vec3<u32>)
{
    cs_pick_2(id_67.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_c(@builtin(global_invocation_id) id_68 : vec3<u32>)
{
    cs_pick_1(id_68.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_d(@builtin(global_invocation_id) id_69 : vec3<u32>)
{
    cs_pick_0(id_69.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_record(@builtin(global_invocation_id) id_70 : vec3<u32>)
{
    var _S666 : u32 = id_70.x;
    var _S667 : u32 = a_counters_0();
    var _S668 : u32 = atomicLoad(&(atom_0[_S667 + u32(18)]));
    if(_S666 >= _S668)
    {
        return;
    }
    var s_24 : u32 = work_0[w_cs_bodies_0() + _S666];
    var _S669 : u32 = a_cs_0(work_0[w_cs_bodies_0() + _S666]);
    var _S670 : u32 = atomicLoad(&(atom_0[_S669 + u32(11)]));
    var key_2 : u32 = _S670 - u32(1);
    var k_54 : u32;
    var s2_1 : u32;
    var ob_7 : u32;
    var co_7 : u32;
    var p_22 : vec3<f32>;
    var u_12 : vec3<f32>;
    var v_18 : f32;
    var m_eff_9 : f32;
    var t_in_12 : f32;
    var _S671 : bool = cs_side_1(key_2 / u32(2), key_2 % u32(2), &(k_54), &(s2_1), &(ob_7), &(co_7), &(p_22), &(u_12), &(v_18), &(m_eff_9), &(t_in_12));
    if(!_S671)
    {
        return;
    }
    var other_1 : vec4<u32> = collider_info_0[u32(2) * co_7 + u32(1)];
    var e_o_0 : f32 = (bitcast<f32>((collider_info_0[u32(2) * co_7 + u32(1)].y)));
    var _S672 : f32 = max((bitcast<f32>((collider_info_0[u32(2) * co_7 + u32(1)].z))), 0.00100000004749745f);
    var j_10 : f32 = m_eff_9 * v_18;
    var tau_3 : f32 = hertz_duration_0(m_eff_9, _S672, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / geometry_0[u32(3) * k_54 + u32(1)].w), v_18);
    var _S673 : u32 = lp_body_0(s_24);
    lp_0[_S673 + u32(2)] = vec4<f32>(from_rubble_0(p_22), 3.14159274101257324f * j_10 / (2.0f * tau_3));
    lp_0[_S673 + u32(3)] = vec4<f32>(from_rubble_0(u_12), v_18);
    lp_0[_S673 + u32(4)] = vec4<f32>(m_eff_9, _S672, e_o_0, (bitcast<f32>((ob_7))));
    lp_0[_S673 + u32(5)] = vec4<f32>((bitcast<f32>((k_54))), j_10 / tau_3, 1.0f / bodies_0[u32(7) * ob_7].w, t_in_12);
    lp_0[_S673 + u32(6)] = vec4<f32>((bitcast<f32>((co_7))), (bitcast<f32>((other_1.w))), tau_3, 0.0f);
    atomicStore(&(atom_0[_S669 + u32(1)]), k_54 + u32(1));
    var _S674 : u32 = atomicAdd(&(atom_0[_S667 + u32(16)]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_hertz(@builtin(global_invocation_id) id_71 : vec3<u32>)
{
    var i_24 : u32 = id_71.x;
    var _S675 : bool;
    if(i_24 >= (params_0.bond_count_0))
    {
        _S675 = true;
    }
    else
    {
        _S675 = work_0[w_alive_0() + i_24] == u32(0);
    }
    if(_S675)
    {
        return;
    }
    var bond_12 : vec4<u32> = bonds_0[i_24];
    var b_79 : u32 = body_of_0(bond_12.x);
    var _S676 : bool = cs_struck_0(b_79);
    if(!_S676)
    {
        return;
    }
    var _S677 : u32 = lp_body_0(b_79);
    if(strike_share_breaks_0(bond_12, lp_0[_S677 + u32(2)].xyz, lp_0[_S677 + u32(5)].y))
    {
        work_0[w_alive_0() + i_24] = u32(0);
        var _S678 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_79]), u32(1));
        var _S679 : u32 = a_counters_0();
        var _S680 : u32 = atomicAdd(&(atom_0[_S679]), u32(1));
        var _S681 : u32 = atomicAdd(&(atom_0[_S679 + u32(1)]), u32(1));
        event_broken_0(i_24);
        var _S682 : u32 = atomicAdd(&(atom_0[_S679 + u32(17)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_cs_gate2(@builtin(global_invocation_id) id_72 : vec3<u32>)
{
    var _S683 : u32 = atomicLoad(&(atom_0[a_counters_0()]));
    var _S684 : u32;
    if(_S683 > u32(0))
    {
        var _S685 : u32 = groups_of_0(cs_rows_0(), u32(64));
        _S684 = _S685;
    }
    else
    {
        _S684 = u32(0);
    }
    set_args_0(u32(13), _S684);
    return;
}

fn lp_anchor_0( r_26 : u32,  side_8 : u32) -> u32
{
    return u32(22) * params_0.chunk_count_0 + u32(7) * params_0.body_count_0 + params_0.chunk_count_0 + u32(2) * r_26 + side_8;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_save(@builtin(global_invocation_id) id_73 : vec3<u32>)
{
    var r_27 : u32 = id_73.x;
    if(r_27 >= (cs_rows_0()))
    {
        return;
    }
    var row_3 : u32 = cs_row_0(r_27);
    var _S686 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_3])));
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
        var _S687 : vec4<f32> = vec4<f32>(0.0f);
        var _S688 : bool = side_9 == u32(0);
        var col_1 : u32;
        if(_S688)
        {
            col_1 = _S686.x;
        }
        else
        {
            col_1 = _S686.y;
        }
        var k_55 : u32;
        if((_S686.w) != u32(0))
        {
            k_55 = collider_info_0[u32(2) * col_1].x;
        }
        else
        {
            k_55 = u32(4294967295);
        }
        var _S689 : bool;
        if(k_55 != u32(4294967295))
        {
            _S689 = k_55 < (params_0.chunk_count_0);
        }
        else
        {
            _S689 = false;
        }
        var _S690 : bool;
        if(_S689)
        {
            var _S691 : bool = dirty_0(body_of_0(k_55));
            _S690 = _S691;
        }
        else
        {
            _S690 = false;
        }
        var saved_0 : vec4<f32>;
        if(_S690)
        {
            var at_27 : u32 = u32(7) * body_of_0(k_55);
            var anchor_0 : vec3<f32>;
            if(_S688)
            {
                anchor_0 = engine_contacts_0[row_3 + u32(2)].xyz;
            }
            else
            {
                anchor_0 = engine_contacts_0[row_3 + u32(3)].xyz;
            }
            saved_0 = vec4<f32>(bodies_0[at_27].xyz + rotate_0(quat_of_0(bodies_0[at_27 + u32(1)]), anchor_0), 1.0f);
        }
        else
        {
            saved_0 = _S687;
        }
        lp_0[lp_anchor_0(r_27, side_9)] = saved_0;
        side_9 = side_9 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_clean(@builtin(global_invocation_id) id_74 : vec3<u32>)
{
    var _S692 : u32 = id_74.x;
    var _S693 : u32 = atomicLoad(&(atom_0[a_counters_0() + u32(18)]));
    if(_S692 >= _S693)
    {
        return;
    }
    var _S694 : u32 = a_cs_0(work_0[w_cs_bodies_0() + _S692]);
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
        atomicStore(&(atom_0[_S694 + w_9]), u32(0));
        w_9 = w_9 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_fix(@builtin(global_invocation_id) id_75 : vec3<u32>)
{
    var r_28 : u32 = id_75.x;
    if(r_28 >= (cs_rows_0()))
    {
        return;
    }
    var row_4 : u32 = cs_row_0(r_28);
    var _S695 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_4])));
    var moved_0 : bool = false;
    var side_10 : u32 = u32(0);
    loop
    {
        if(side_10 < u32(2))
        {
        }
        else
        {
            break;
        }
        var saved_1 : vec4<f32> = lp_0[lp_anchor_0(r_28, side_10)];
        if((lp_0[lp_anchor_0(r_28, side_10)].w) == 0.0f)
        {
            side_10 = side_10 + u32(1);
            continue;
        }
        var col_2 : u32;
        if(side_10 == u32(0))
        {
            col_2 = _S695.x;
        }
        else
        {
            col_2 = _S695.y;
        }
        var at_28 : u32 = u32(7) * (bitcast<u32>((colliders_0[u32(4) * col_2].x)));
        var anchor_1 : vec3<f32> = rotate_0(conj_0(quat_of_0(bodies_0[at_28 + u32(1)])), saved_1.xyz - bodies_0[at_28].xyz);
        engine_contacts_0[row_4 + u32(2) + side_10].x = anchor_1.x;
        engine_contacts_0[row_4 + u32(2) + side_10].y = anchor_1.y;
        engine_contacts_0[row_4 + u32(2) + side_10].z = anchor_1.z;
        moved_0 = true;
        side_10 = side_10 + u32(1);
    }
    if(moved_0)
    {
        engine_contacts_0[row_4 + u32(4)][i32(3)] = 0.0f;
        engine_contacts_0[row_4 + u32(5)][i32(3)] = 0.0f;
        var _S696 : u32 = row_4 + u32(6);
        engine_contacts_0[_S696] = vec4<f32>(1.0f, 1.0f, 0.0f, engine_contacts_0[_S696].w);
        engine_contacts_0[row_4 + u32(7)][i32(1)] = 0.0f;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_count(@builtin(global_invocation_id) id_76 : vec3<u32>)
{
    var k_56 : u32 = id_76.x;
    if(k_56 >= (params_0.chunk_count_0))
    {
        return;
    }
    var s_25 : u32 = body_of_0(k_56);
    var _S697 : bool = cs_struck_0(s_25);
    if(!_S697)
    {
        return;
    }
    var _S698 : u32 = a_cs_0(s_25);
    var _S699 : u32 = atomicMax(&(atom_0[_S698 + u32(3)]), ~k_56);
    if(((info_0[k_56] & (u32(1)))) != u32(0))
    {
        atomicStore(&(atom_0[_S698 + u32(4)]), u32(1));
    }
    var _S700 : u32 = atomicAdd(&(atom_0[_S698 + u32(5)]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_base(@builtin(global_invocation_id) id_77 : vec3<u32>)
{
    var _S701 : u32 = id_77.x;
    var _S702 : u32 = a_counters_0();
    var _S703 : u32 = atomicLoad(&(atom_0[_S702 + u32(18)]));
    if(_S701 >= _S703)
    {
        return;
    }
    var _S704 : u32 = a_cs_0(work_0[w_cs_bodies_0() + _S701]);
    var _S705 : u32 = atomicLoad(&(atom_0[_S704 + u32(5)]));
    var _S706 : u32 = atomicAdd(&(atom_0[_S702 + u32(19)]), _S705);
    atomicStore(&(atom_0[_S704 + u32(6)]), _S706);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_collect(@builtin(global_invocation_id) id_78 : vec3<u32>)
{
    var k_57 : u32 = id_78.x;
    if(k_57 >= (params_0.chunk_count_0))
    {
        return;
    }
    var s_26 : u32 = body_of_0(k_57);
    var _S707 : bool = cs_struck_0(s_26);
    if(!_S707)
    {
        return;
    }
    var _S708 : u32 = w_cs_list_0();
    var _S709 : u32 = a_cs_0(s_26);
    var _S710 : u32 = atomicLoad(&(atom_0[_S709 + u32(6)]));
    var _S711 : u32 = _S708 + _S710;
    var _S712 : u32 = atomicAdd(&(atom_0[_S709 + u32(7)]), u32(1));
    work_0[_S711 + _S712] = k_57;
    return;
}

fn cs_load_0( k_58 : u32,  s_27 : u32)
{
    var _S713 : u32 = a_cs_0(s_27);
    var _S714 : u32 = atomicLoad(&(atom_0[_S713 + u32(4)]));
    var supported_1 : bool = _S714 != u32(0);
    var fixed_2 : bool;
    if(((info_0[k_58] & (u32(1)))) != u32(0))
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            var _S715 : u32 = atomicLoad(&(atom_0[_S713 + u32(3)]));
            fixed_2 = (~_S715) == k_58;
        }
        else
        {
            fixed_2 = false;
        }
    }
    var _S716 : u32 = lp_body_0(s_27);
    var _S717 : u32 = u32(7) * s_27;
    var q_36 : Quat_0;
    var origin_16 : vec3<f32>;
    structure_pose_3(s_27, _S717, &(q_36), &(origin_16));
    var qi_1 : Quat_0 = conj_0(q_36);
    var force_4 : vec3<f32> = rotate_0(qi_1, lp_0[_S716 + u32(3)].xyz * vec3<f32>(lp_0[_S716 + u32(2)].w));
    var contact_22 : vec3<f32> = rotate_0(qi_1, lp_0[_S716 + u32(2)].xyz - origin_16);
    var _S718 : u32 = u32(4) * k_58;
    var c0_5 : vec4<f32> = chunks_0[_S718];
    var _S719 : vec3<f32> = vec3<f32>(0.0f);
    var _S720 : u32 = atomicLoad(&(atom_0[_S713 + u32(1)]));
    var f_13 : vec3<f32>;
    var t_16 : vec3<f32>;
    if(_S720 == (k_58 + u32(1)))
    {
        var _S721 : vec3<f32> = cross(contact_22 - c0_5.xyz, force_4);
        f_13 = force_4;
        t_16 = _S721;
    }
    else
    {
        f_13 = _S719;
        t_16 = _S719;
    }
    var inv_mass_7 : f32 = bodies_0[_S717].w;
    var _S722 : bool;
    if(!supported_1)
    {
        _S722 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S722 = false;
    }
    if(_S722)
    {
        var _S723 : u32 = u32(2) * s_27;
        var com_7 : vec3<f32> = frames_0[_S723].xyz;
        var p_23 : Quat_0 = quat_of_0(frames_0[_S723 + u32(1)]);
        var alpha_1 : vec3<f32> = rotate_0(p_23, bodies_0[_S717 + u32(2)].xyz * rotate_0(conj_0(p_23), cross(contact_22 - com_7, force_4)));
        var c1_4 : vec4<f32> = chunks_0[_S718 + u32(1)];
        var c2_5 : vec4<f32> = chunks_0[_S718 + u32(2)];
        var _S724 : f32 = alpha_1.x;
        var _S725 : f32 = c1_4.w;
        var _S726 : f32 = alpha_1.y;
        var _S727 : f32 = c2_5.x;
        var _S728 : f32 = alpha_1.z;
        var _S729 : f32 = c2_5.y;
        var t_17 : vec3<f32> = t_16 - vec3<f32>(c1_4.x * _S724 + _S725 * _S726 + _S727 * _S728, _S725 * _S724 + c1_4.y * _S726 + _S729 * _S728, _S727 * _S724 + _S729 * _S726 + c1_4.z * _S728);
        f_13 = f_13 - vec3<f32>(c0_5.w) * (force_4 * vec3<f32>(inv_mass_7) + cross(alpha_1, c0_5.xyz - com_7));
        t_16 = t_17;
    }
    var _S730 : u32 = st_f_0(k_58, u32(14));
    var _S731 : f32;
    if(fixed_2)
    {
        _S731 = 1.0f;
    }
    else
    {
        _S731 = 0.0f;
    }
    lp_0[_S730] = vec4<f32>(f_13, _S731);
    lp_0[_S730 + u32(1)] = vec4<f32>(t_16, 0.0f);
    var c_27 : u32 = u32(0);
    loop
    {
        if(c_27 < u32(2))
        {
        }
        else
        {
            break;
        }
        var _S732 : u32 = st_x_0(k_58, u32(14), c_27);
        var _S733 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S732] = _S733;
        lp_0[_S732 + u32(1)] = _S733;
        c_27 = c_27 + u32(1);
    }
    var _S734 : u32 = st_d_0(k_58, u32(14));
    var _S735 : vec4<f32> = vec4<f32>(0.0f);
    lp_0[_S734] = _S735;
    lp_0[_S734 + u32(1)] = _S735;
    return;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn cs_break_0( i_25 : u32,  s_28 : u32)
{
    var bond_13 : vec4<u32> = bonds_0[i_25];
    var b_80 : LpBond_0 = lp_bond_0(i_25);
    var _S736 : u32 = bond_13.x;
    var _S737 : u32 = bond_13.y;
    var q_lin_8 : vec3<f32>;
    var q_ang_8 : vec3<f32>;
    st_force_0(b_80, _S736, _S737, u32(14), u32(0), &(q_lin_8), &(q_ang_8));
    if((params_0.load_model_0) == u32(1))
    {
        var b_lin_0 : vec3<f32>;
        var b_ang_0 : vec3<f32>;
        lp_force_0(b_80, _S736, _S737, u32(0), &(b_lin_0), &(b_ang_0));
        q_lin_8 = q_lin_8 + b_lin_0;
        q_ang_8 = q_ang_8 + b_ang_0;
    }
    var idx_2 : vec4<f32> = failure_indices_0(lp_material_0(b_80.law_0.ids_3.x), b_80.law_0, stress_measures_0(b_80.law_0, q_lin_8, q_ang_8), b_80.law_0.geom1_0.w);
    if((max(max(idx_2.x, idx_2.y), max(idx_2.z, idx_2.w))) >= 1.0f)
    {
        work_0[w_alive_0() + i_25] = u32(0);
        var _S738 : u32 = atomicOr(&(atom_0[a_dirty_0() + s_28]), u32(1));
        var _S739 : u32 = a_counters_0();
        var _S740 : u32 = atomicAdd(&(atom_0[_S739]), u32(1));
        var _S741 : u32 = atomicAdd(&(atom_0[_S739 + u32(1)]), u32(1));
        event_broken_0(i_25);
        var _S742 : u32 = atomicAdd(&(atom_0[_S739 + u32(17)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn fr_cs_solve(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var _S743 : bool;
    var j3_0 : u32;
    var s_29 : u32 = work_0[w_cs_bodies_0() + group_3.x];
    var t_18 : u32 = thread_6.x;
    var _S744 : u32 = a_cs_0(work_0[w_cs_bodies_0() + group_3.x]);
    var _S745 : u32 = atomicLoad(&(atom_0[_S744 + u32(6)]));
    var _S746 : u32 = atomicLoad(&(atom_0[_S744 + u32(5)]));
    var j_11 : u32 = t_18;
    loop
    {
        if(j_11 < _S746)
        {
        }
        else
        {
            break;
        }
        cs_load_0(work_0[w_cs_list_0() + _S745 + j_11], s_29);
        j_11 = j_11 + u32(256);
    }
    device_group_barrier_0();
    var j2_1 : u32 = t_18;
    loop
    {
        if(j2_1 < _S746)
        {
        }
        else
        {
            break;
        }
        var k_59 : u32 = work_0[w_cs_list_0() + _S745 + j2_1];
        var _S747 : u32 = st_f_0(work_0[w_cs_list_0() + _S745 + j2_1], u32(14));
        if((lp_0[_S747].w) == 0.0f)
        {
            var _S748 : bool = lp_factor_0(k_59);
            _S743 = !_S748;
        }
        else
        {
            _S743 = false;
        }
        if(_S743)
        {
            lp_0[_S747][i32(3)] = 1.0f;
            var _S749 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(64));
        }
        j2_1 = j2_1 + u32(256);
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
        j3_0 = t_18;
        loop
        {
            if(j3_0 < _S746)
            {
            }
            else
            {
                break;
            }
            st_sweep_0(work_0[w_cs_list_0() + _S745 + j3_0], u32(14), sweep_0 % u32(2), params_0.cs_c1_0, params_0.cs_c2_0);
            j3_0 = j3_0 + u32(256);
        }
        device_group_barrier_0();
        sweep_0 = sweep_0 + u32(1);
    }
    j3_0 = t_18;
    loop
    {
        if(j3_0 < _S746)
        {
        }
        else
        {
            break;
        }
        var k_60 : u32 = work_0[w_cs_list_0() + _S745 + j3_0];
        var range_4 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * work_0[w_cs_list_0() + _S745 + j3_0]];
        var _S750 : u32 = (bitcast<u32>((range_4.x)));
        var _S751 : u32 = (bitcast<u32>((range_4.y)));
        var e_7 : u32 = u32(0);
        loop
        {
            if(e_7 < _S751)
            {
            }
            else
            {
                break;
            }
            var i_26 : u32 = adj_bond_0(_S750 + e_7);
            if((bonds_0[i_26].x) == k_60)
            {
                _S743 = work_0[w_alive_0() + i_26] != u32(0);
            }
            else
            {
                _S743 = false;
            }
            if(_S743)
            {
                cs_break_0(i_26, s_29);
            }
            e_7 = e_7 + u32(1);
        }
        j3_0 = j3_0 + u32(256);
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_gi_begin(@builtin(global_invocation_id) id_79 : vec3<u32>)
{
    var _S752 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S752 + u32(18)]), u32(0));
    atomicStore(&(atom_0[_S752 + u32(19)]), u32(0));
    return;
}

var<workgroup> g_gi_x_0 : array<f32, i32(256)>;

var<workgroup> g_gi_m_0 : array<f32, i32(256)>;

fn gi_acc_0( lo_6 : ptr<function, u32>,  hi_6 : ptr<function, u32>,  value_4 : f32)
{
    var x_10 : f32 = round(value_4 * 1.073741824e+09f);
    var x_11 : f32;
    if(!((abs(x_10)) < 3.99999993722699776e+18f))
    {
        x_11 = 0.0f;
    }
    else
    {
        x_11 = x_10;
    }
    var u_13 : f32 = abs(x_11);
    var h_7 : f32 = floor(u_13 * 2.32830643653869629e-10f);
    var l_4 : u32 = u32(u_13 - h_7 * 4.294967296e+09f);
    var g_0 : u32 = u32(h_7);
    var l_5 : u32;
    var g_1 : u32;
    if(x_11 < 0.0f)
    {
        var l_6 : u32 = ~l_4 + u32(1);
        var _S753 : u32 = ~g_0;
        if(l_6 == u32(0))
        {
            l_5 = u32(1);
        }
        else
        {
            l_5 = u32(0);
        }
        var _S754 : u32 = _S753 + l_5;
        l_5 = l_6;
        g_1 = _S754;
    }
    else
    {
        l_5 = l_4;
        g_1 = g_0;
    }
    var sum_4 : u32 = (*lo_6) + l_5;
    if(sum_4 < (*lo_6))
    {
        l_5 = u32(1);
    }
    else
    {
        l_5 = u32(0);
    }
    (*hi_6) = (*hi_6) + (g_1 + l_5);
    (*lo_6) = sum_4;
    return;
}

var<workgroup> g_gi_lo_0 : array<u32, i32(1280)>;

var<workgroup> g_gi_hi_0 : array<u32, i32(1280)>;

fn gi_value_0( lo_7 : u32,  hi_7 : u32) -> f32
{
    var negative_0 : bool = ((hi_7 & (u32(2147483648)))) != u32(0);
    var _S755 : u32;
    var _S756 : u32;
    if(negative_0)
    {
        var _S757 : u32 = ~lo_7 + u32(1);
        var _S758 : u32 = ~hi_7;
        if(_S757 == u32(0))
        {
            _S755 = u32(1);
        }
        else
        {
            _S755 = u32(0);
        }
        _S755 = _S758 + _S755;
        _S756 = _S757;
    }
    else
    {
        _S755 = hi_7;
        _S756 = lo_7;
    }
    var v_19 : f32 = (f32(_S755) * 4.294967296e+09f + f32(_S756)) / 1.073741824e+09f;
    var _S759 : f32;
    if(negative_0)
    {
        _S759 = - v_19;
    }
    else
    {
        _S759 = v_19;
    }
    return _S759;
}

fn gi_dlf_0( r_29 : f32) -> f32
{
    if(!(r_29 > 0.0f))
    {
        return 0.0f;
    }
    if(r_29 > 33.0f)
    {
        return r_29 / (r_29 - 1.0f);
    }
    if((abs(r_29 - 1.0f)) < 0.00009999999747379f)
    {
        return 1.57079637050628662f;
    }
    var _S760 : f32 = r_29 * r_29;
    var _S761 : f32 = r_29 + 1.0f;
    var _S762 : u32 = u32(floor(0.5f * _S761));
    var best_2 : f32 = abs(2.0f * r_29 * cos(1.57079637050628662f * r_29) / (1.0f - _S760));
    var l_7 : u32 = u32(1);
    loop
    {
        if(l_7 <= _S762)
        {
        }
        else
        {
            break;
        }
        var a_14 : f32 = 6.28318548202514648f * f32(l_7) / _S761;
        var _S763 : f32 = max(best_2, abs((sin(a_14) - sin(r_29 * a_14) / r_29) / (1.0f - 1.0f / _S760)));
        var _S764 : u32 = l_7 + u32(1);
        best_2 = _S763;
        l_7 = _S764;
    }
    return best_2;
}

fn gi_factor_0( s_30 : u32,  t_19 : u32,  base_3 : u32,  count_0 : u32) -> f32
{
    var _S765 : bool;
    var x2_0 : f32 = 0.0f;
    var mm_0 : f32 = 0.0f;
    var j_12 : u32 = t_19;
    loop
    {
        if(j_12 < count_0)
        {
        }
        else
        {
            break;
        }
        var u_14 : vec3<f32> = lp_0[st_x_0(work_0[w_cs_list_0() + base_3 + j_12], u32(14), u32(0))].xyz;
        var _S766 : f32 = max(x2_0, dot(u_14, u_14));
        var _S767 : f32 = max(mm_0, chunks_0[u32(4) * work_0[w_cs_list_0() + base_3 + j_12]].w);
        var j_13 : u32 = j_12 + u32(256);
        x2_0 = _S766;
        mm_0 = _S767;
        j_12 = j_13;
    }
    g_gi_x_0[t_19] = x2_0;
    g_gi_m_0[t_19] = mm_0;
    workgroupBarrier();
    var step_6 : u32 = u32(128);
    loop
    {
        if(step_6 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_19 < step_6)
        {
            var _S768 : u32 = t_19 + step_6;
            g_gi_x_0[t_19] = max(g_gi_x_0[t_19], g_gi_x_0[_S768]);
            g_gi_m_0[t_19] = max(g_gi_m_0[t_19], g_gi_m_0[_S768]);
        }
        workgroupBarrier();
        step_6 = (step_6 >> (u32(1)));
    }
    var x2_1 : f32 = g_gi_x_0[i32(0)];
    var mm_1 : f32 = g_gi_m_0[i32(0)];
    var xn_0 : f32 = sqrt(g_gi_x_0[i32(0)]);
    var _S769 : u32 = lp_body_0(s_30);
    var f0_0 : f32 = lp_0[_S769 + u32(2)].w;
    var any_1 : bool;
    if((g_gi_x_0[i32(0)]) > 0.0f)
    {
        any_1 = mm_1 > 0.0f;
    }
    else
    {
        any_1 = false;
    }
    if(any_1)
    {
        any_1 = f0_0 > 0.0f;
    }
    else
    {
        any_1 = false;
    }
    var lo_8 : array<u32, i32(5)>;
    var hi_8 : array<u32, i32(5)>;
    var i_27 : u32 = u32(0);
    loop
    {
        if(i_27 < u32(5))
        {
        }
        else
        {
            break;
        }
        lo_8[i_27] = u32(0);
        hi_8[i_27] = u32(0);
        i_27 = i_27 + u32(1);
    }
    var j2_2 : u32 = t_19;
    loop
    {
        if(any_1)
        {
            _S765 = j2_2 < count_0;
        }
        else
        {
            _S765 = false;
        }
        if(_S765)
        {
        }
        else
        {
            break;
        }
        var _S770 : u32 = st_x_0(work_0[w_cs_list_0() + base_3 + j2_2], u32(14), u32(0));
        var u_15 : vec3<f32> = lp_0[_S770].xyz;
        var th_0 : vec3<f32> = lp_0[_S770 + u32(1)].xyz;
        var _S771 : u32 = st_f_0(work_0[w_cs_list_0() + base_3 + j2_2], u32(14));
        var _S772 : u32 = u32(4) * work_0[w_cs_list_0() + base_3 + j2_2];
        var c0_6 : vec4<f32> = chunks_0[_S772];
        var c1_5 : vec4<f32> = chunks_0[_S772 + u32(1)];
        var c2_6 : vec4<f32> = chunks_0[_S772 + u32(2)];
        var _S773 : f32 = th_0.x;
        var _S774 : f32 = c1_5.w;
        var _S775 : f32 = th_0.y;
        var _S776 : f32 = c2_6.x;
        var _S777 : f32 = th_0.z;
        var _S778 : f32 = c2_6.y;
        var ith_0 : vec3<f32> = vec3<f32>(c1_5.x * _S773 + _S774 * _S775 + _S776 * _S777, _S774 * _S773 + c1_5.y * _S775 + _S778 * _S777, _S776 * _S773 + _S778 * _S775 + c1_5.z * _S777);
        var _S779 : f32 = (dot(lp_0[_S771].xyz, u_15) + dot(lp_0[_S771 + u32(1)].xyz, th_0)) / (f0_0 * xn_0);
        var _S780 : u32 = lo_8[i32(0)];
        var _S781 : u32 = hi_8[i32(0)];
        gi_acc_0(&(_S780), &(_S781), _S779);
        lo_8[i32(0)] = _S780;
        hi_8[i32(0)] = _S781;
        var _S782 : f32 = c0_6.w;
        var _S783 : f32 = (_S782 * dot(u_15, u_15) + dot(th_0, ith_0)) / (mm_1 * x2_1);
        var _S784 : u32 = lo_8[i32(1)];
        var _S785 : u32 = hi_8[i32(1)];
        gi_acc_0(&(_S784), &(_S785), _S783);
        lo_8[i32(1)] = _S784;
        hi_8[i32(1)] = _S785;
        var p_24 : vec3<f32> = u_15 * vec3<f32>((_S782 / (mm_1 * xn_0)));
        var _S786 : f32 = p_24.x;
        var _S787 : u32 = lo_8[i32(2)];
        var _S788 : u32 = hi_8[i32(2)];
        gi_acc_0(&(_S787), &(_S788), _S786);
        lo_8[i32(2)] = _S787;
        hi_8[i32(2)] = _S788;
        var _S789 : f32 = p_24.y;
        var _S790 : u32 = lo_8[i32(3)];
        var _S791 : u32 = hi_8[i32(3)];
        gi_acc_0(&(_S790), &(_S791), _S789);
        lo_8[i32(3)] = _S790;
        hi_8[i32(3)] = _S791;
        var _S792 : f32 = p_24.z;
        var _S793 : u32 = lo_8[i32(4)];
        var _S794 : u32 = hi_8[i32(4)];
        gi_acc_0(&(_S793), &(_S794), _S792);
        lo_8[i32(4)] = _S793;
        hi_8[i32(4)] = _S794;
        j2_2 = j2_2 + u32(256);
    }
    var i2_2 : u32 = u32(0);
    loop
    {
        if(i2_2 < u32(5))
        {
        }
        else
        {
            break;
        }
        var _S795 : u32 = u32(5) * t_19 + i2_2;
        g_gi_lo_0[_S795] = lo_8[i2_2];
        g_gi_hi_0[_S795] = hi_8[i2_2];
        i2_2 = i2_2 + u32(1);
    }
    workgroupBarrier();
    var step2_0 : u32 = u32(128);
    loop
    {
        if(step2_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_19 < step2_0)
        {
            var i3_1 : u32 = u32(0);
            loop
            {
                if(i3_1 < u32(5))
                {
                }
                else
                {
                    break;
                }
                var a_15 : u32 = u32(5) * t_19 + i3_1;
                var o_10 : u32 = u32(5) * (t_19 + step2_0) + i3_1;
                var sum_5 : u32 = g_gi_lo_0[a_15] + g_gi_lo_0[o_10];
                var _S796 : u32 = g_gi_hi_0[o_10];
                if(sum_5 < (g_gi_lo_0[a_15]))
                {
                    j_12 = u32(1);
                }
                else
                {
                    j_12 = u32(0);
                }
                g_gi_hi_0[a_15] = g_gi_hi_0[a_15] + (_S796 + j_12);
                g_gi_lo_0[a_15] = sum_5;
                i3_1 = i3_1 + u32(1);
            }
        }
        workgroupBarrier();
        step2_0 = (step2_0 >> (u32(1)));
    }
    var work_sum_0 : f32 = gi_value_0(g_gi_lo_0[i32(0)], g_gi_hi_0[i32(0)]);
    var inertia_0 : f32 = gi_value_0(g_gi_lo_0[i32(1)], g_gi_hi_0[i32(1)]);
    var p_sum_0 : vec3<f32> = vec3<f32>(gi_value_0(g_gi_lo_0[i32(2)], g_gi_hi_0[i32(2)]), gi_value_0(g_gi_lo_0[i32(3)], g_gi_hi_0[i32(3)]), gi_value_0(g_gi_lo_0[i32(4)], g_gi_hi_0[i32(4)]));
    var inv_mass_8 : f32 = bodies_0[u32(7) * s_30].w;
    var _S797 : u32 = atomicLoad(&(atom_0[a_cs_0(s_30) + u32(4)]));
    if(_S797 == u32(0))
    {
        _S765 = inv_mass_8 > 0.0f;
    }
    else
    {
        _S765 = false;
    }
    var inertia_1 : f32;
    if(_S765)
    {
        inertia_1 = inertia_0 - mm_1 * inv_mass_8 * dot(p_sum_0, p_sum_0);
    }
    else
    {
        inertia_1 = inertia_0;
    }
    var w_10 : f32 = f0_0 * xn_0 * work_sum_0;
    var m_6 : f32 = mm_1 * x2_1 * inertia_1;
    if(any_1)
    {
        any_1 = w_10 > 0.0f;
    }
    else
    {
        any_1 = false;
    }
    if(any_1)
    {
        any_1 = m_6 > 0.0f;
    }
    else
    {
        any_1 = false;
    }
    var omega_0 : f32;
    if(any_1)
    {
        omega_0 = sqrt(w_10 / m_6);
    }
    else
    {
        omega_0 = 0.0f;
    }
    var _S798 : u32 = _S769 + u32(6);
    var tau_4 : f32 = lp_0[_S798].z;
    if(t_19 == u32(0))
    {
        any_1 = count_0 > u32(0);
    }
    else
    {
        any_1 = false;
    }
    if(any_1)
    {
        lp_0[_S798][i32(3)] = omega_0;
    }
    if(omega_0 > 0.0f)
    {
        any_1 = tau_4 > 0.0f;
    }
    else
    {
        any_1 = false;
    }
    if(any_1)
    {
        x2_0 = gi_dlf_0(omega_0 * tau_4 / 3.14159274101257324f);
    }
    else
    {
        x2_0 = 1.0f;
    }
    return x2_0;
}

fn gi_break_0( i_28 : u32,  s_31 : u32,  scale_4 : f32,  contact_23 : vec3<f32>,  local_1 : f32)
{
    var bond_14 : vec4<u32> = bonds_0[i_28];
    var b_81 : LpBond_0 = lp_bond_0(i_28);
    if((length(b_81.centroid_0 - contact_23)) <= local_1)
    {
        return;
    }
    var _S799 : u32 = bond_14.x;
    var _S800 : u32 = bond_14.y;
    var q_lin_9 : vec3<f32>;
    var q_ang_9 : vec3<f32>;
    st_force_0(b_81, _S799, _S800, u32(14), u32(0), &(q_lin_9), &(q_ang_9));
    var _S801 : vec3<f32> = vec3<f32>(scale_4);
    q_lin_9 = q_lin_9 * _S801;
    q_ang_9 = q_ang_9 * _S801;
    if((params_0.load_model_0) == u32(1))
    {
        var b_lin_1 : vec3<f32>;
        var b_ang_1 : vec3<f32>;
        lp_force_0(b_81, _S799, _S800, u32(0), &(b_lin_1), &(b_ang_1));
        q_lin_9 = q_lin_9 + b_lin_1;
        q_ang_9 = q_ang_9 + b_ang_1;
    }
    var idx_3 : vec4<f32> = failure_indices_0(lp_material_0(b_81.law_0.ids_3.x), b_81.law_0, stress_measures_0(b_81.law_0, q_lin_9, q_ang_9), b_81.law_0.geom1_0.w);
    if((max(max(idx_3.x, idx_3.y), max(idx_3.z, idx_3.w))) >= 1.0f)
    {
        work_0[w_alive_0() + i_28] = u32(0);
        var _S802 : u32 = atomicOr(&(atom_0[a_dirty_0() + s_31]), u32(1));
        var _S803 : u32 = a_counters_0();
        var _S804 : u32 = atomicAdd(&(atom_0[_S803]), u32(1));
        var _S805 : u32 = atomicAdd(&(atom_0[_S803 + u32(1)]), u32(1));
        event_broken_0(i_28);
        var _S806 : u32 = atomicAdd(&(atom_0[_S803 + u32(17)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn fr_gi_solve(@builtin(workgroup_id) group_4 : vec3<u32>, @builtin(local_invocation_id) thread_7 : vec3<u32>)
{
    var _S807 : bool;
    var j3_1 : u32;
    var t_20 : u32 = thread_7.x;
    var _S808 : u32 = group_4.x;
    var _S809 : u32 = a_counters_0();
    var _S810 : u32 = atomicLoad(&(atom_0[_S809 + u32(18)]));
    var j_14 : u32;
    if(_S808 < _S810)
    {
        j_14 = u32(1);
    }
    else
    {
        j_14 = u32(0);
    }
    var _S811 : u32 = group_uniform_0(j_14, t_20);
    if(_S811 == u32(0))
    {
        return;
    }
    var s_32 : u32 = work_0[w_cs_bodies_0() + _S808];
    var _S812 : u32 = a_cs_0(work_0[w_cs_bodies_0() + _S808]);
    var base_4 : u32 = atomicLoad(&(atom_0[_S812 + u32(6)]));
    var count_1 : u32 = atomicLoad(&(atom_0[_S812 + u32(5)]));
    j_14 = t_20;
    loop
    {
        if(j_14 < count_1)
        {
        }
        else
        {
            break;
        }
        cs_load_0(work_0[w_cs_list_0() + base_4 + j_14], s_32);
        j_14 = j_14 + u32(256);
    }
    device_group_barrier_0();
    var j2_3 : u32 = t_20;
    loop
    {
        if(j2_3 < count_1)
        {
        }
        else
        {
            break;
        }
        var k_61 : u32 = work_0[w_cs_list_0() + base_4 + j2_3];
        var _S813 : u32 = st_f_0(work_0[w_cs_list_0() + base_4 + j2_3], u32(14));
        if((lp_0[_S813].w) == 0.0f)
        {
            var _S814 : bool = lp_factor_0(k_61);
            _S807 = !_S814;
        }
        else
        {
            _S807 = false;
        }
        if(_S807)
        {
            lp_0[_S813][i32(3)] = 1.0f;
            if((lp_intact_0(k_61)) != u32(0))
            {
                var _S815 : u32 = atomicOr(&(atom_0[_S809 + u32(3)]), u32(64));
            }
        }
        j2_3 = j2_3 + u32(256);
    }
    device_group_barrier_0();
    var _S816 : u32 = group_uniform_0(clamp(u32(2) * u32(ceil(32.0f * pow(f32(max(count_1, u32(1))), 0.3333333432674408f))), u32(256), params_0.gi_sweeps_0), t_20);
    var sweep_1 : u32 = u32(0);
    loop
    {
        if(sweep_1 < _S816)
        {
        }
        else
        {
            break;
        }
        j3_1 = t_20;
        loop
        {
            if(j3_1 < count_1)
            {
            }
            else
            {
                break;
            }
            st_sweep_0(work_0[w_cs_list_0() + base_4 + j3_1], u32(14), sweep_1 % u32(2), params_0.cs_c1_0, params_0.cs_c2_0);
            j3_1 = j3_1 + u32(256);
        }
        device_group_barrier_0();
        sweep_1 = sweep_1 + u32(1);
    }
    var scale_5 : f32 = params_0.gi_scale_0;
    var scale_6 : f32;
    if((params_0.gi_mode_0) == u32(2))
    {
        var _S817 : f32 = gi_factor_0(s_32, t_20, base_4, count_1);
        var scale_7 : f32 = scale_5 * _S817;
        device_group_barrier_0();
        scale_6 = scale_7;
    }
    else
    {
        scale_6 = scale_5;
    }
    var _S818 : u32 = atomicLoad(&(atom_0[_S812 + u32(11)]));
    var _S819 : f32;
    if(_S818 == u32(0))
    {
        _S819 = lp_0[lp_body_0(s_32) + u32(4)].x;
    }
    else
    {
        _S819 = 0.0f;
    }
    var sq_0 : Quat_0;
    var s_origin_0 : vec3<f32>;
    structure_pose_5(s_32, u32(7) * s_32, &(sq_0), &(s_origin_0));
    var _S820 : vec3<f32> = rotate_0(conj_0(sq_0), lp_0[lp_body_0(s_32) + u32(2)].xyz - s_origin_0);
    j3_1 = t_20;
    loop
    {
        if(j3_1 < count_1)
        {
        }
        else
        {
            break;
        }
        var k_62 : u32 = work_0[w_cs_list_0() + base_4 + j3_1];
        var range_5 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * work_0[w_cs_list_0() + base_4 + j3_1]];
        var _S821 : u32 = (bitcast<u32>((range_5.x)));
        var _S822 : u32 = (bitcast<u32>((range_5.y)));
        var e_8 : u32 = u32(0);
        loop
        {
            if(e_8 < _S822)
            {
            }
            else
            {
                break;
            }
            var i_29 : u32 = adj_bond_0(_S821 + e_8);
            if((bonds_0[i_29].x) == k_62)
            {
                _S807 = work_0[w_alive_0() + i_29] != u32(0);
            }
            else
            {
                _S807 = false;
            }
            if(_S807)
            {
                gi_break_0(i_29, s_32, scale_6, _S820, _S819);
            }
            e_8 = e_8 + u32(1);
        }
        j3_1 = j3_1 + u32(256);
    }
    return;
}

fn crushable_0( k_63 : u32) -> bool
{
    return (((bitcast<u32>((crush_tab_0[u32(2) * k_63 + u32(1)].z))) & (u32(1)))) != u32(0);
}

fn crush_flags_0( k_64 : u32) -> u32
{
    return (bitcast<u32>((crush_rw_0[k_64].w)));
}

fn crush_record_0( k_65 : u32,  step_7 : vec4<f32>,  cause_0 : u32)
{
    var st_6 : vec4<f32> = crush_rw_0[k_65];
    var flags_1 : u32 = (bitcast<u32>((crush_rw_0[k_65].w)));
    var flags_2 : u32;
    if((step_7.w) != 0.0f)
    {
        flags_2 = (((((flags_1 | (u32(1)))) | (u32(2)))) | (((cause_0 << (u32(4))))));
    }
    else
    {
        flags_2 = flags_1;
    }
    crush_rw_0[k_65] = vec4<f32>(step_7.x, max(st_6.y, step_7.y), max(st_6.z, step_7.z), (bitcast<f32>((flags_2))));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_blast(@builtin(global_invocation_id) id_80 : vec3<u32>)
{
    var k_66 : u32 = id_80.x;
    var _S823 : bool;
    if(k_66 >= (params_0.chunk_count_0))
    {
        _S823 = true;
    }
    else
    {
        _S823 = (crush_0.enabled_0) == u32(0);
    }
    if(_S823)
    {
        return;
    }
    var w_11 : vec4<f32> = crush_rw_0[crush_0.r_blast_0 + k_66];
    var _S824 : f32 = crush_rw_0[crush_0.r_blast_0 + k_66].x;
    if(!(_S824 > 0.0f))
    {
        _S823 = !((w_11.y) > 0.0f);
    }
    else
    {
        _S823 = false;
    }
    if(_S823)
    {
        _S823 = !((w_11.z) > 0.0f);
    }
    else
    {
        _S823 = false;
    }
    if(_S823)
    {
        return;
    }
    crush_rw_0[crush_0.r_blast_0 + k_66] = vec4<f32>(0.0f);
    if(!crushable_0(k_66))
    {
        _S823 = true;
    }
    else
    {
        _S823 = (((crush_flags_0(k_66)) & (u32(1)))) != u32(0);
    }
    if(_S823)
    {
        return;
    }
    var d_26 : f32 = crush_rw_0[k_66].x + _S824;
    var step_8 : vec4<f32>;
    if(d_26 >= 1.0f)
    {
        step_8 = vec4<f32>(1.0f, w_11.y, w_11.z, 1.0f);
    }
    else
    {
        step_8 = vec4<f32>(d_26, w_11.y, w_11.z, 0.0f);
    }
    crush_record_0(k_66, step_8, u32(3));
    return;
}

struct Sym_0
{
     d_27 : vec3<f32>,
     o_11 : vec3<f32>,
};

fn sym_zero_0() -> Sym_0
{
    var s_33 : Sym_0;
    var _S825 : vec3<f32> = vec3<f32>(0.0f);
    s_33.d_27 = _S825;
    s_33.o_11 = _S825;
    return s_33;
}

fn sym_outer_0( r_30 : vec3<f32>,  f_14 : vec3<f32>) -> Sym_0
{
    var s_34 : Sym_0;
    s_34.d_27 = r_30 * f_14;
    var _S826 : f32 = r_30.x;
    var _S827 : f32 = f_14.y;
    var _S828 : f32 = r_30.y;
    var _S829 : f32 = f_14.x;
    var _S830 : f32 = f_14.z;
    var _S831 : f32 = r_30.z;
    s_34.o_11 = vec3<f32>(0.5f) * vec3<f32>(_S826 * _S827 + _S828 * _S829, _S826 * _S830 + _S831 * _S829, _S828 * _S830 + _S831 * _S827);
    return s_34;
}

fn sym_add_0( a_16 : Sym_0,  b_82 : Sym_0) -> Sym_0
{
    var s_35 : Sym_0;
    s_35.d_27 = a_16.d_27 + b_82.d_27;
    s_35.o_11 = a_16.o_11 + b_82.o_11;
    return s_35;
}

fn add_load_0( w_12 : ptr<function, Sym_0>,  f_sum_0 : ptr<function, vec3<f32>>,  spread_0 : ptr<function, vec4<f32>>,  r_31 : vec3<f32>,  f_15 : vec3<f32>)
{
    (*w_12) = sym_add_0((*w_12), sym_outer_0(r_31, f_15));
    (*f_sum_0) = (*f_sum_0) + f_15;
    var m_7 : f32 = length(f_15);
    (*spread_0) = (*spread_0) + vec4<f32>(vec3<f32>(m_7) * r_31, m_7);
    return;
}

fn add_load_1( w_13 : ptr<function, Sym_0>,  f_sum_1 : ptr<function, vec3<f32>>,  spread_1 : ptr<function, vec4<f32>>,  r_32 : vec3<f32>,  f_16 : vec3<f32>)
{
    (*w_13) = sym_add_0((*w_13), sym_outer_0(r_32, f_16));
    (*f_sum_1) = (*f_sum_1) + f_16;
    var m_8 : f32 = length(f_16);
    (*spread_1) = (*spread_1) + vec4<f32>(vec3<f32>(m_8) * r_32, m_8);
    return;
}

fn flow_virial_0( k_67 : u32,  q_37 : Quat_0,  origin_17 : vec3<f32>,  centre_1 : vec3<f32>,  wf_0 : ptr<function, Sym_0>,  total_0 : ptr<function, f32>)
{
    (*wf_0) = sym_zero_0();
    (*total_0) = 0.0f;
    var adj_0 : vec4<f32> = crush_tab_0[crush_0.t_adj_0 + k_67];
    var first_2 : u32 = (bitcast<u32>((adj_0.x)));
    var _S832 : u32 = (bitcast<u32>((adj_0.y)));
    var j_15 : u32 = first_2;
    loop
    {
        if(j_15 < (first_2 + _S832))
        {
        }
        else
        {
            break;
        }
        var bi_0 : u32 = (bitcast<u32>((crush_tab_0[crush_0.t_list_0 + j_15 / u32(4)][j_15 % u32(4)])));
        if(work_0[w_alive_0() + bi_0] == u32(0))
        {
            j_15 = j_15 + u32(1);
            continue;
        }
        var l_8 : f32 = crush_rw_0[crush_0.r_flow_0 + bi_0 / u32(4)][bi_0 % u32(4)];
        if(!((crush_rw_0[crush_0.r_flow_0 + bi_0 / u32(4)][bi_0 % u32(4)]) > 0.0f))
        {
            j_15 = j_15 + u32(1);
            continue;
        }
        var r_33 : vec3<f32> = origin_17 + rotate_0(q_37, crush_tab_0[crush_0.t_bonds_0 + bi_0].xyz) - centre_1;
        var n_7 : vec3<f32> = rotate_0(q_37, crush_tab_0[crush_0.t_normal_0 + bi_0].xyz);
        var out_n_0 : vec3<f32>;
        if((dot(n_7, r_33)) >= 0.0f)
        {
            out_n_0 = n_7;
        }
        else
        {
            out_n_0 = (vec3<f32>(0) - n_7);
        }
        (*wf_0) = sym_add_0((*wf_0), sym_outer_0(r_33, vec3<f32>(- l_8) * out_n_0));
        (*total_0) = (*total_0) + l_8;
        j_15 = j_15 + u32(1);
    }
    return;
}

fn bond_centroid_0( k_68 : u32,  q_38 : Quat_0,  origin_18 : vec3<f32>,  centre_2 : vec3<f32>) -> vec4<f32>
{
    var adj_1 : vec4<f32> = crush_tab_0[crush_0.t_adj_0 + k_68];
    var first_3 : u32 = (bitcast<u32>((adj_1.x)));
    var _S833 : u32 = (bitcast<u32>((adj_1.y)));
    var _S834 : vec3<f32> = vec3<f32>(0.0f);
    var j_16 : u32 = first_3;
    var sum_6 : vec3<f32> = _S834;
    var area_1 : f32 = 0.0f;
    loop
    {
        if(j_16 < (first_3 + _S833))
        {
        }
        else
        {
            break;
        }
        var bi_1 : u32 = (bitcast<u32>((crush_tab_0[crush_0.t_list_0 + j_16 / u32(4)][j_16 % u32(4)])));
        if(work_0[w_alive_0() + bi_1] == u32(0))
        {
            j_16 = j_16 + u32(1);
            continue;
        }
        var bc_0 : vec4<f32> = crush_tab_0[crush_0.t_bonds_0 + bi_1];
        var _S835 : f32 = bc_0.w;
        var area_2 : f32 = area_1 + _S835;
        sum_6 = sum_6 + vec3<f32>(_S835) * (origin_18 + rotate_0(q_38, bc_0.xyz) - centre_2);
        area_1 = area_2;
        j_16 = j_16 + u32(1);
    }
    if(area_1 > 0.0f)
    {
        sum_6 = sum_6 / vec3<f32>(area_1);
    }
    else
    {
        sum_6 = _S834;
    }
    return vec4<f32>(sum_6, area_1);
}

fn balanced_0( w_14 : Sym_0,  f_17 : vec3<f32>,  spread_2 : vec4<f32>,  bonds_1 : vec4<f32>) -> Sym_0
{
    var at_29 : vec3<f32>;
    if((bonds_1.w) > 0.0f)
    {
        at_29 = bonds_1.xyz;
    }
    else
    {
        var _S836 : f32 = spread_2.w;
        if(_S836 > 0.0f)
        {
            at_29 = spread_2.xyz / vec3<f32>(_S836);
        }
        else
        {
            at_29 = vec3<f32>(0.0f);
        }
    }
    return sym_add_0(w_14, sym_outer_0(at_29, (vec3<f32>(0) - f_17)));
}

fn sym_norm2_0( s_36 : Sym_0) -> f32
{
    var _S837 : vec3<f32> = s_36.d_27;
    var _S838 : vec3<f32> = s_36.o_11;
    return dot(_S837, _S837) + 2.0f * dot(_S838, _S838);
}

fn crush_invariants_0( v_20 : Sym_0,  volume_0 : f32) -> vec2<f32>
{
    var inv_4 : f32 = 1.0f / volume_0;
    var xx_0 : f32 = v_20.d_27.x * inv_4;
    var yy_0 : f32 = v_20.d_27.y * inv_4;
    var zz_0 : f32 = v_20.d_27.z * inv_4;
    var xy_0 : f32 = v_20.o_11.x * inv_4;
    var xz_0 : f32 = v_20.o_11.y * inv_4;
    var yz_0 : f32 = v_20.o_11.z * inv_4;
    var p_25 : f32 = - (xx_0 + yy_0 + zz_0) / 3.0f;
    var dx_0 : f32 = xx_0 + p_25;
    var dy_0 : f32 = yy_0 + p_25;
    var dz_0 : f32 = zz_0 + p_25;
    return vec2<f32>(p_25, sqrt(1.5f * max(dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0 + 2.0f * (xy_0 * xy_0 + xz_0 * xz_0 + yz_0 * yz_0), 0.0f)));
}

fn crush_excess_0( pq_0 : vec2<f32>,  m0_0 : vec4<f32>) -> f32
{
    var _S839 : f32 = pq_0.x;
    if(!(_S839 > 0.0f))
    {
        return 0.0f;
    }
    var _S840 : f32 = max(pq_0.y - (m0_0.y + m0_0.z * _S839), _S839 - m0_0.x);
    var _S841 : f32;
    if(_S840 > 0.0f)
    {
        _S841 = _S840;
    }
    else
    {
        _S841 = 0.0f;
    }
    return _S841;
}

fn crush_two_0( a_17 : Sym_0,  dt_a_0 : f32,  b_83 : Sym_0,  dt_b_0 : f32,  k_69 : u32,  damage_2 : f32) -> vec4<f32>
{
    var _S842 : u32 = u32(2) * k_69;
    var m0_1 : vec4<f32> = crush_tab_0[_S842];
    var m1_1 : vec4<f32> = crush_tab_0[_S842 + u32(1)];
    var _S843 : f32 = m1_1.y;
    var pa_4 : vec2<f32> = crush_invariants_0(a_17, _S843);
    var pb_5 : vec2<f32> = crush_invariants_0(b_83, _S843);
    var ea_0 : f32 = crush_excess_0(pa_4, m0_1);
    var _S844 : bool = dt_b_0 > 0.0f;
    var eb_0 : f32;
    if(_S844)
    {
        eb_0 = crush_excess_0(pb_5, m0_1);
    }
    else
    {
        eb_0 = 0.0f;
    }
    var d_28 : f32 = damage_2 + (ea_0 * ea_0 * dt_a_0 + eb_0 * eb_0 * dt_b_0) / (m1_1.x * m0_1.w);
    var peak_2 : vec2<f32>;
    if(_S844)
    {
        peak_2 = max(pa_4, pb_5);
    }
    else
    {
        peak_2 = pa_4;
    }
    if(d_28 >= 1.0f)
    {
        return vec4<f32>(1.0f, peak_2, 1.0f);
    }
    return vec4<f32>(d_28, peak_2, 0.0f);
}

fn crush_loads_flow_0( k_70 : u32)
{
    var slot_21 : u32 = body_of_0(k_70);
    var b_84 : u32 = u32(7) * slot_21;
    var q_39 : Quat_0;
    var origin_19 : vec3<f32>;
    structure_pose_4(slot_21, b_84, &(q_39), &(origin_19));
    var _S845 : u32 = u32(4) * k_70;
    var c0_7 : vec4<f32> = chunks_0[_S845];
    var _S846 : vec4<f32> = chunks_0[_S845 + u32(3)];
    var centre_3 : vec3<f32> = origin_19 + rotate_0(q_39, c0_7.xyz);
    var inv_mass_9 : f32 = bodies_0[b_84].w;
    var _S847 : f32;
    if(inv_mass_9 > 0.0f)
    {
        _S847 = 1.0f / inv_mass_9;
    }
    else
    {
        _S847 = float_bits_0(u32(2139095040));
    }
    var dt_2 : f32 = params_0.dt_0;
    var _S848 : Sym_0 = sym_zero_0();
    var wa_0 : Sym_0 = _S848;
    var _S849 : vec3<f32> = vec3<f32>(0.0f);
    var fa_0 : vec3<f32> = _S849;
    var _S850 : vec4<f32> = vec4<f32>(0.0f);
    var sa_0 : vec4<f32> = _S850;
    var wb_0 : Sym_0 = _S848;
    var fb_1 : vec3<f32> = _S849;
    var sb_2 : vec4<f32> = _S850;
    var _S851 : u32 = sort_0[k_70];
    var _S852 : u32 = sort_0[params_0.chunk_count_0 + u32(1) + k_70];
    var tau_5 : f32 = 0.0f;
    var i_30 : u32 = u32(0);
    loop
    {
        if(i_30 < _S852)
        {
        }
        else
        {
            break;
        }
        var r_34 : Record_std430_0 = manifolds_0[_S851 + i_30];
        var _S853 : vec3<f32> = r_34.impulse_0.xyz;
        var j_17 : f32 = length(_S853);
        if(j_17 == 0.0f)
        {
            i_30 = i_30 + u32(1);
            continue;
        }
        var m_other_2 : f32 = r_34.other_0.x;
        var m_9 : f32;
        if(isinf_0(_S847))
        {
            if(isinf_0(m_other_2))
            {
                m_9 = c0_7.w;
            }
            else
            {
                m_9 = m_other_2;
            }
        }
        else
        {
            if(isinf_0(m_other_2))
            {
                m_9 = _S847;
            }
            else
            {
                m_9 = _S847 * m_other_2 / (_S847 + m_other_2);
            }
        }
        var _S854 : f32 = r_34.impulse_0.w;
        var _S855 : f32 = min(j_17, 2.0f * m_9 * max(_S854 - params_0.band_0, 0.0f));
        var rr_0 : vec3<f32> = r_34.point_0.xyz - centre_3;
        var dir_1 : vec3<f32> = _S853 / vec3<f32>(j_17);
        if(_S855 < j_17)
        {
            add_load_0(&(wa_0), &(fa_0), &(sa_0), rr_0, dir_1 * vec3<f32>(((j_17 - _S855) / dt_2)));
        }
        var tau_6 : f32;
        if(_S855 > 0.0f)
        {
            var _S856 : f32 = _S846.y;
            var duration_2 : f32 = hertz_duration_0(m_9, min(r_34.other_0.z, _S846.z), 1.0f / ((1.0f - _S856 * _S856) / _S846.x + 1.0f / r_34.other_0.y), _S854);
            add_load_0(&(wb_0), &(fb_1), &(sb_2), rr_0, dir_1 * vec3<f32>((3.14159274101257324f * _S855 / (2.0f * duration_2))));
            tau_6 = max(tau_5, min(duration_2, dt_2));
        }
        else
        {
            tau_6 = tau_5;
        }
        tau_5 = tau_6;
        i_30 = i_30 + u32(1);
    }
    var from_4 : u32 = frame_parity_0();
    var _S857 : u32 = w_pulses_0() + _S845 + u32(2) * from_4;
    var _S858 : u32 = work_0[_S857 + u32(1)];
    var _S859 : u32 = l_pool_0(from_4) + u32(3) * work_0[_S857];
    var p_26 : u32 = u32(0);
    loop
    {
        if(p_26 < _S858)
        {
        }
        else
        {
            break;
        }
        var _S860 : u32 = _S859 + u32(3) * p_26;
        var bb_0 : vec4<f32> = loads_0[_S860 + u32(1)];
        var c_28 : vec4<f32> = loads_0[_S860 + u32(2)];
        var mean_2 : f32 = pulse_mean_0(loads_0[_S860].x, loads_0[_S860].y, loads_0[_S860].z, loads_0[_S860].w, loads_0[_S860 + u32(1)].w, dt_2);
        if(mean_2 != 0.0f)
        {
            add_load_0(&(wa_0), &(fa_0), &(sa_0), bb_0.xyz - centre_3, c_28.xyz * vec3<f32>(mean_2));
        }
        p_26 = p_26 + u32(1);
    }
    var wf_1 : Sym_0;
    var flow_total_0 : f32;
    flow_virial_0(k_70, q_39, origin_19, centre_3, &(wf_1), &(flow_total_0));
    var contact_24 : bool;
    if((sa_0.w) > 0.0f)
    {
        contact_24 = true;
    }
    else
    {
        contact_24 = (sb_2.w) > 0.0f;
    }
    var _S861 : bool;
    if(!contact_24)
    {
        _S861 = !(flow_total_0 > 0.0f);
    }
    else
    {
        _S861 = false;
    }
    if(_S861)
    {
        return;
    }
    var _S862 : Sym_0 = wf_1;
    var _S863 : Sym_0 = wf_1;
    var peak_contact_0 : Sym_0;
    var base_5 : Sym_0;
    var peak_3 : Sym_0;
    if(contact_24)
    {
        var bonds_2 : vec4<f32> = bond_centroid_0(k_70, q_39, origin_19, centre_3);
        var _S864 : Sym_0 = sym_add_0(balanced_0(wa_0, fa_0, sa_0, bonds_2), wf_1);
        var peak_contact_1 : Sym_0 = balanced_0(sym_add_0(wa_0, wb_0), fa_0 + fb_1, sa_0 + sb_2, bonds_2);
        var _S865 : Sym_0 = sym_add_0(peak_contact_1, wf_1);
        peak_contact_0 = peak_contact_1;
        base_5 = _S864;
        peak_3 = _S865;
    }
    else
    {
        peak_contact_0 = _S848;
        base_5 = _S862;
        peak_3 = _S863;
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
    crush_record_0(k_70, crush_two_0(base_5, dt_2 - tau_5, peak_3, tau_5, k_70, crush_rw_0[k_70].x), cause_1);
    return;
}

fn sym_rotated_0( q_40 : Quat_0,  d_29 : vec3<f32>,  o_12 : vec3<f32>) -> Sym_0
{
    var _S866 : f32 = o_12.x;
    var _S867 : f32 = o_12.y;
    var _S868 : f32 = o_12.z;
    var c0_8 : vec3<f32> = rotate_0(q_40, vec3<f32>(1.0f, 0.0f, 0.0f));
    var c1_6 : vec3<f32> = rotate_0(q_40, vec3<f32>(0.0f, 1.0f, 0.0f));
    var c2_7 : vec3<f32> = rotate_0(q_40, vec3<f32>(0.0f, 0.0f, 1.0f));
    var r_35 : mat3x3<f32> = mat3x3<f32>(c0_8.x, c1_6.x, c2_7.x, c0_8.y, c1_6.y, c2_7.y, c0_8.z, c1_6.z, c2_7.z);
    var w_15 : mat3x3<f32> = (((transpose(r_35)) * ((((mat3x3<f32>(d_29.x, _S866, _S867, _S866, d_29.y, _S868, _S867, _S868, d_29.z)) * (r_35))))));
    var out_s_0 : Sym_0;
    out_s_0.d_27 = vec3<f32>(w_15[i32(0)][i32(0)], w_15[i32(1)][i32(1)], w_15[i32(2)][i32(2)]);
    out_s_0.o_11 = vec3<f32>(w_15[i32(0)][i32(1)], w_15[i32(0)][i32(2)], w_15[i32(1)][i32(2)]);
    return out_s_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_loads(@builtin(global_invocation_id) id_81 : vec3<u32>)
{
    var k_71 : u32 = id_81.x;
    var _S869 : bool;
    if(k_71 >= (params_0.chunk_count_0))
    {
        _S869 = true;
    }
    else
    {
        _S869 = (crush_0.enabled_0) == u32(0);
    }
    if(_S869)
    {
        _S869 = true;
    }
    else
    {
        _S869 = !crushable_0(k_71);
    }
    if(_S869)
    {
        _S869 = true;
    }
    else
    {
        _S869 = (((crush_flags_0(k_71)) & (u32(1)))) != u32(0);
    }
    if(_S869)
    {
        return;
    }
    if((crush_0.flow_0) != u32(0))
    {
        crush_loads_flow_0(k_71);
        return;
    }
    var slot_22 : u32 = body_of_0(k_71);
    var b_85 : u32 = u32(7) * slot_22;
    var q_41 : Quat_0;
    var origin_20 : vec3<f32>;
    structure_pose_5(slot_22, b_85, &(q_41), &(origin_20));
    var _S870 : u32 = u32(4) * k_71;
    var c0_9 : vec4<f32> = chunks_0[_S870];
    var _S871 : vec4<f32> = chunks_0[_S870 + u32(3)];
    var centre_4 : vec3<f32> = origin_20 + rotate_0(q_41, c0_9.xyz);
    var inv_mass_10 : f32 = bodies_0[b_85].w;
    var _S872 : f32;
    if(inv_mass_10 > 0.0f)
    {
        _S872 = 1.0f / inv_mass_10;
    }
    else
    {
        _S872 = float_bits_0(u32(2139095040));
    }
    var dt_3 : f32 = params_0.dt_0;
    var _S873 : Sym_0 = sym_zero_0();
    var wa_1 : Sym_0 = _S873;
    var _S874 : vec3<f32> = vec3<f32>(0.0f);
    var fa_1 : vec3<f32> = _S874;
    var _S875 : vec4<f32> = vec4<f32>(0.0f);
    var sa_1 : vec4<f32> = _S875;
    var wb_1 : Sym_0 = _S873;
    var fb_2 : vec3<f32> = _S874;
    var sb_3 : vec4<f32> = _S875;
    var _S876 : u32 = sort_0[k_71];
    var _S877 : u32 = sort_0[params_0.chunk_count_0 + u32(1) + k_71];
    var tau_7 : f32 = 0.0f;
    var i_31 : u32 = u32(0);
    loop
    {
        if(i_31 < _S877)
        {
        }
        else
        {
            break;
        }
        var r_36 : Record_std430_0 = manifolds_0[_S876 + i_31];
        var _S878 : vec3<f32> = r_36.impulse_0.xyz;
        var j_18 : f32 = length(_S878);
        if(j_18 == 0.0f)
        {
            i_31 = i_31 + u32(1);
            continue;
        }
        var m_other_3 : f32 = r_36.other_0.x;
        var m_10 : f32;
        if(isinf_0(_S872))
        {
            if(isinf_0(m_other_3))
            {
                m_10 = c0_9.w;
            }
            else
            {
                m_10 = m_other_3;
            }
        }
        else
        {
            if(isinf_0(m_other_3))
            {
                m_10 = _S872;
            }
            else
            {
                m_10 = _S872 * m_other_3 / (_S872 + m_other_3);
            }
        }
        var _S879 : f32 = r_36.impulse_0.w;
        var _S880 : f32 = min(j_18, 2.0f * m_10 * max(_S879 - params_0.band_0, 0.0f));
        var rr_1 : vec3<f32> = r_36.point_0.xyz - centre_4;
        var dir_2 : vec3<f32> = _S878 / vec3<f32>(j_18);
        if(_S880 < j_18)
        {
            add_load_1(&(wa_1), &(fa_1), &(sa_1), rr_1, dir_2 * vec3<f32>(((j_18 - _S880) / dt_3)));
        }
        var tau_8 : f32;
        if(_S880 > 0.0f)
        {
            var _S881 : f32 = _S871.y;
            var duration_3 : f32 = hertz_duration_0(m_10, min(r_36.other_0.z, _S871.z), 1.0f / ((1.0f - _S881 * _S881) / _S871.x + 1.0f / r_36.other_0.y), _S879);
            add_load_1(&(wb_1), &(fb_2), &(sb_3), rr_1, dir_2 * vec3<f32>((3.14159274101257324f * _S880 / (2.0f * duration_3))));
            tau_8 = max(tau_7, min(duration_3, dt_3));
        }
        else
        {
            tau_8 = tau_7;
        }
        tau_7 = tau_8;
        i_31 = i_31 + u32(1);
    }
    var from_5 : u32 = frame_parity_0();
    var _S882 : u32 = w_pulses_0() + _S870 + u32(2) * from_5;
    var _S883 : u32 = work_0[_S882 + u32(1)];
    var _S884 : u32 = l_pool_0(from_5) + u32(3) * work_0[_S882];
    var p_27 : u32 = u32(0);
    loop
    {
        if(p_27 < _S883)
        {
        }
        else
        {
            break;
        }
        var _S885 : u32 = _S884 + u32(3) * p_27;
        var bb_1 : vec4<f32> = loads_0[_S885 + u32(1)];
        var c_29 : vec4<f32> = loads_0[_S885 + u32(2)];
        var mean_3 : f32 = pulse_mean_0(loads_0[_S885].x, loads_0[_S885].y, loads_0[_S885].z, loads_0[_S885].w, loads_0[_S885 + u32(1)].w, dt_3);
        if(mean_3 != 0.0f)
        {
            add_load_1(&(wa_1), &(fa_1), &(sa_1), bb_1.xyz - centre_4, c_29.xyz * vec3<f32>(mean_3));
        }
        p_27 = p_27 + u32(1);
    }
    if((crush_0.bond_virial_0) != u32(0))
    {
        var _S886 : u32 = u32(2) * k_71;
        var v_21 : Sym_0 = sym_rotated_0(q_41, crush_rw_0[crush_0.r_virial_0 + _S886].xyz, crush_rw_0[crush_0.r_virial_0 + _S886 + u32(1)].xyz);
        crush_record_0(k_71, crush_two_0(sym_add_0(wa_1, v_21), dt_3 - tau_7, sym_add_0(sym_add_0(wa_1, wb_1), v_21), tau_7, k_71, crush_rw_0[k_71].x), u32(1));
        return;
    }
    if(!((sa_1.w) > 0.0f))
    {
        _S869 = !((sb_3.w) > 0.0f);
    }
    else
    {
        _S869 = false;
    }
    if(_S869)
    {
        return;
    }
    var bonds_3 : vec4<f32> = bond_centroid_0(k_71, q_41, origin_20, centre_4);
    crush_record_0(k_71, crush_two_0(balanced_0(wa_1, fa_1, sa_1, bonds_3), dt_3 - tau_7, balanced_0(sym_add_0(wa_1, wb_1), fa_1 + fb_2, sa_1 + sb_3, bonds_3), tau_7, k_71, crush_rw_0[k_71].x), u32(1));
    return;
}

fn a_cover_0() -> u32
{
    return a_moved_0() + u32(2) * params_0.ev_moved_cap_0 + params_0.chunk_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_patch(@builtin(global_invocation_id) id_82 : vec3<u32>)
{
    var b_86 : u32 = id_82.x;
    var _S887 : bool;
    if(b_86 >= (params_0.body_count_0))
    {
        _S887 = true;
    }
    else
    {
        _S887 = (params_0.shot_count_0) == u32(0);
    }
    if(_S887)
    {
        _S887 = true;
    }
    else
    {
        _S887 = (crush_0.enabled_0) == u32(0);
    }
    if(_S887)
    {
        return;
    }
    var bill_0 : u32 = crush_0.r_bill_0 + u32(4) * b_86;
    var _S888 : vec4<f32> = vec4<f32>(0.0f);
    crush_rw_0[bill_0] = _S888;
    var _S889 : u32 = bill_0 + u32(3);
    crush_rw_0[_S889] = _S888;
    var shot_12 : u32;
    var _S890 : bool = struck_2(b_86, &(shot_12));
    if(!_S890)
    {
        return;
    }
    var _S891 : u32 = a_strike_0() + u32(9) * b_86;
    var k_72 : u32 = atomicLoad(&(atom_0[_S891 + u32(8)]));
    var _S892 : u32 = atomicLoad(&(atom_0[_S891 + u32(6)]));
    var t_in_13 : f32 = (bitcast<f32>((_S892)));
    if(!(t_in_13 <= (params_0.dt_0)))
    {
        _S887 = true;
    }
    else
    {
        _S887 = k_72 == u32(4294967295);
    }
    if(_S887)
    {
        return;
    }
    var st_7 : u32 = strike_at_0(b_86);
    var s0_5 : vec4<f32> = frames_0[st_7];
    var _S893 : vec3<f32> = frames_0[st_7 + u32(1)].xyz;
    var speed_17 : f32 = length(_S893);
    var d_30 : vec3<f32> = _S893 / vec3<f32>(max(speed_17, 9.999999960041972e-13f));
    var info_4 : vec4<f32> = shots_0[u32(2) * shot_12];
    var radius_1 : f32 = info_4.y;
    var _S894 : f32 = max(info_4.z, 0.00100000004749745f);
    var m_imp_1 : f32 = info_4.w;
    var inv_mass_11 : f32 = bodies_0[u32(7) * b_86].w;
    var _S895 : bool = inv_mass_11 == 0.0f;
    var m_body_2 : f32;
    if(_S895)
    {
        m_body_2 = 0.0f;
    }
    else
    {
        m_body_2 = 1.0f / inv_mass_11;
    }
    var m_eff_10 : f32;
    if(_S895)
    {
        m_eff_10 = m_imp_1;
    }
    else
    {
        m_eff_10 = m_imp_1 * m_body_2 / (m_imp_1 + m_body_2);
    }
    var modulus_2 : f32 = 1.0f / (1.0f / shots_0[u32(2) * shot_12 + u32(1)].x + 1.0f / geometry_0[u32(3) * k_72 + u32(1)].w);
    var _S896 : f32 = max(speed_17, 0.00100000004749745f);
    var tau_9 : f32 = 2.86999988555908203f * pow(m_eff_10 * m_eff_10 / (_S894 * modulus_2 * modulus_2 * _S896), 0.20000000298023224f);
    var force_5 : f32 = m_eff_10 * speed_17 / tau_9;
    var contact_25 : vec3<f32> = s0_5.xyz + _S893 * vec3<f32>(t_in_13) + d_30 * vec3<f32>(radius_1);
    var delta_4 : f32 = pow(15.0f * m_eff_10 * _S896 * _S896 / (16.0f * modulus_2 * sqrt(_S894)), 0.40000000596046448f);
    var a_h_0 : f32 = sqrt(radius_1 * delta_4);
    var p_cap_0 : f32;
    if(crushable_0(k_72))
    {
        p_cap_0 = crush_tab_0[u32(2) * k_72].x;
    }
    else
    {
        p_cap_0 = 0.0f;
    }
    var a_p_0 : f32;
    if(p_cap_0 > 0.0f)
    {
        a_p_0 = sqrt(force_5 / (3.14159274101257324f * p_cap_0));
    }
    else
    {
        a_p_0 = 0.0f;
    }
    var _S897 : f32 = min(radius_1, max(a_h_0, a_p_0));
    var _S898 : f32 = max(delta_4, radius_1 - sqrt(max(radius_1 * radius_1 - _S897 * _S897, 0.0f)));
    crush_rw_0[bill_0 + u32(1)] = vec4<f32>(contact_25, 0.0f);
    crush_rw_0[bill_0 + u32(2)] = vec4<f32>(d_30, _S897);
    crush_rw_0[_S889] = vec4<f32>(force_5, tau_9, _S898, f32(k_72 + u32(1)));
    var _S899 : u32 = a_cover_0() + u32(128) * b_86;
    var s_37 : u32 = u32(0);
    loop
    {
        if(s_37 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S900 : u32 = _S899 + u32(2) * s_37;
        atomicStore(&(atom_0[_S900]), u32(4294967295));
        atomicStore(&(atom_0[_S900 + u32(1)]), u32(4294967295));
        s_37 = s_37 + u32(1);
    }
    return;
}

fn cover_patch_0( k_73 : u32,  b_87 : ptr<function, u32>,  p1_0 : ptr<function, vec4<f32>>,  p2_2 : ptr<function, vec4<f32>>) -> bool
{
    (*b_87) = body_of_0(k_73);
    var _S901 : vec4<f32> = vec4<f32>(0.0f);
    (*p1_0) = _S901;
    (*p2_2) = _S901;
    var shot_13 : u32;
    var _S902 : bool = struck_2((*b_87), &(shot_13));
    var _S903 : bool;
    if(!_S902)
    {
        _S903 = true;
    }
    else
    {
        _S903 = (((crush_flags_0(k_73)) & (u32(1)))) != u32(0);
    }
    if(_S903)
    {
        return false;
    }
    var bill_1 : u32 = crush_0.r_bill_0 + u32(4) * (*b_87);
    if((crush_rw_0[bill_1 + u32(3)].w) == 0.0f)
    {
        return false;
    }
    (*p1_0) = vec4<f32>(crush_rw_0[bill_1 + u32(1)].xyz, shots_0[u32(2) * shot_13].y);
    (*p2_2) = crush_rw_0[bill_1 + u32(2)];
    return true;
}

fn cover_offset_0( s_38 : u32,  d_31 : vec3<f32>,  a_18 : f32,  rho_0 : ptr<function, f32>) -> vec3<f32>
{
    var ring_2 : u32 = s_38 / u32(8);
    var sector_0 : u32 = s_38 % u32(8);
    (*rho_0) = a_18 * sqrt((f32(ring_2) + 0.5f) / 8.0f);
    var theta_1 : f32 = 6.28318548202514648f * (f32(sector_0) + 0.5f * f32((ring_2 & (u32(1))))) / 8.0f;
    var e_9 : vec3<f32>;
    if((abs(d_31.x)) < 0.57735002040863037f)
    {
        e_9 = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_31.y)) < 0.57735002040863037f)
        {
            e_9 = vec3<f32>(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_9 = vec3<f32>(0.0f, 0.0f, 1.0f);
        }
    }
    var u_16 : vec3<f32> = normalize(cross(d_31, e_9));
    return vec3<f32>((*rho_0)) * (vec3<f32>(cos(theta_1)) * u_16 + vec3<f32>(sin(theta_1)) * cross(d_31, u_16));
}

fn cover_offset_1( s_39 : u32,  d_32 : vec3<f32>,  a_19 : f32,  rho_1 : ptr<function, f32>) -> vec3<f32>
{
    var ring_3 : u32 = s_39 / u32(8);
    var sector_1 : u32 = s_39 % u32(8);
    (*rho_1) = a_19 * sqrt((f32(ring_3) + 0.5f) / 8.0f);
    var theta_2 : f32 = 6.28318548202514648f * (f32(sector_1) + 0.5f * f32((ring_3 & (u32(1))))) / 8.0f;
    var e_10 : vec3<f32>;
    if((abs(d_32.x)) < 0.57735002040863037f)
    {
        e_10 = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_32.y)) < 0.57735002040863037f)
        {
            e_10 = vec3<f32>(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_10 = vec3<f32>(0.0f, 0.0f, 1.0f);
        }
    }
    var u_17 : vec3<f32> = normalize(cross(d_32, e_10));
    return vec3<f32>((*rho_1)) * (vec3<f32>(cos(theta_2)) * u_17 + vec3<f32>(sin(theta_2)) * cross(d_32, u_17));
}

fn cover_depth_0( k_74 : u32,  s_40 : u32,  contact_26 : vec3<f32>,  d_33 : vec3<f32>,  a_20 : f32,  radius_2 : f32) -> f32
{
    var rho_2 : f32;
    var _S904 : f32 = 2.0f * radius_2;
    var _S905 : vec3<f32> = contact_26 + cover_offset_0(s_40, d_33, a_20, &(rho_2)) - d_33 * vec3<f32>(_S904);
    var q_42 : Quat_0;
    var _S906 : vec3<f32> = chunk_box_world_2(k_74, &(q_42));
    var _S907 : u32 = u32(3) * k_74;
    var _S908 : Quat_0 = normalized_0(quat_mul_0(q_42, quat_of_0(geometry_0[_S907 + u32(2)])));
    var _S909 : vec3<f32> = geometry_0[_S907 + u32(1)].xyz;
    var t0_3 : f32 = -1.00000001504746622e+30f;
    var t1_4 : f32 = 1.00000001504746622e+30f;
    var ax_4 : u32 = u32(0);
    loop
    {
        if(ax_4 < u32(3))
        {
        }
        else
        {
            break;
        }
        var e_11 : vec3<f32> = vec3<f32>(0.0f);
        e_11[ax_4] = 1.0f;
        var axis_2 : vec3<f32> = rotate_0(_S908, e_11);
        var o_13 : f32 = dot(_S905 - _S906, axis_2);
        var dv_2 : f32 = dot(d_33, axis_2);
        if((abs(dv_2)) < 9.99999971718068537e-10f)
        {
            if((abs(o_13)) > (_S909[ax_4]))
            {
                return 1.00000001504746622e+30f;
            }
            ax_4 = ax_4 + u32(1);
            continue;
        }
        var ta_3 : f32 = (- _S909[ax_4] - o_13) / dv_2;
        var tb_3 : f32 = (_S909[ax_4] - o_13) / dv_2;
        var _S910 : f32 = min(t1_4, max(ta_3, tb_3));
        t0_3 = max(t0_3, min(ta_3, tb_3));
        t1_4 = _S910;
        ax_4 = ax_4 + u32(1);
    }
    var _S911 : bool;
    if(!(t0_3 <= t1_4))
    {
        _S911 = true;
    }
    else
    {
        _S911 = t1_4 < 0.0f;
    }
    if(_S911)
    {
        return 1.00000001504746622e+30f;
    }
    return t0_3 - _S904;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_cover(@builtin(global_invocation_id) id_83 : vec3<u32>)
{
    var k_75 : u32 = id_83.x;
    var _S912 : bool;
    if(k_75 >= (params_0.chunk_count_0))
    {
        _S912 = true;
    }
    else
    {
        _S912 = (params_0.shot_count_0) == u32(0);
    }
    if(_S912)
    {
        _S912 = true;
    }
    else
    {
        _S912 = (crush_0.enabled_0) == u32(0);
    }
    if(_S912)
    {
        return;
    }
    var b_88 : u32;
    var p1_1 : vec4<f32>;
    var p2_3 : vec4<f32>;
    var _S913 : bool = cover_patch_0(k_75, &(b_88), &(p1_1), &(p2_3));
    if(!_S913)
    {
        return;
    }
    var _S914 : u32 = a_cover_0() + u32(128) * b_88;
    var s_41 : u32 = u32(0);
    loop
    {
        if(s_41 < u32(64))
        {
        }
        else
        {
            break;
        }
        var t_21 : f32 = cover_depth_0(k_75, s_41, p1_1.xyz, p2_3.xyz, p2_3.w, p1_1.w);
        if(t_21 < 5.0000000752373311e+29f)
        {
            var _S915 : u32 = atomicMin(&(atom_0[_S914 + u32(2) * s_41]), ordered_0(t_21));
        }
        s_41 = s_41 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_claim(@builtin(global_invocation_id) id_84 : vec3<u32>)
{
    var k_76 : u32 = id_84.x;
    var _S916 : bool;
    if(k_76 >= (params_0.chunk_count_0))
    {
        _S916 = true;
    }
    else
    {
        _S916 = (params_0.shot_count_0) == u32(0);
    }
    if(_S916)
    {
        _S916 = true;
    }
    else
    {
        _S916 = (crush_0.enabled_0) == u32(0);
    }
    if(_S916)
    {
        return;
    }
    var b_89 : u32;
    var p1_2 : vec4<f32>;
    var p2_4 : vec4<f32>;
    var _S917 : bool = cover_patch_0(k_76, &(b_89), &(p1_2), &(p2_4));
    if(!_S917)
    {
        return;
    }
    var _S918 : u32 = a_cover_0() + u32(128) * b_89;
    var s_42 : u32 = u32(0);
    loop
    {
        if(s_42 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S919 : u32 = _S918 + u32(2) * s_42;
        var first_4 : u32 = atomicLoad(&(atom_0[_S919]));
        if(first_4 == u32(4294967295))
        {
            s_42 = s_42 + u32(1);
            continue;
        }
        var t_22 : f32 = cover_depth_0(k_76, s_42, p1_2.xyz, p2_4.xyz, p2_4.w, p1_2.w);
        if(t_22 < 5.0000000752373311e+29f)
        {
            _S916 = t_22 <= (unordered_0(first_4) + 0.00009999999747379f);
        }
        else
        {
            _S916 = false;
        }
        if(_S916)
        {
            var _S920 : u32 = atomicMin(&(atom_0[_S919 + u32(1)]), k_76);
        }
        s_42 = s_42 + u32(1);
    }
    return;
}

fn strike_load_0( k_77 : u32,  slot_23 : u32,  f_18 : vec3<f32>,  at_30 : vec3<f32>,  tau_10 : f32,  budget_0 : f32) -> f32
{
    var _S921 : bool;
    if(!crushable_0(k_77))
    {
        _S921 = true;
    }
    else
    {
        _S921 = (((crush_flags_0(k_77)) & (u32(1)))) != u32(0);
    }
    if(_S921)
    {
        _S921 = true;
    }
    else
    {
        _S921 = !(budget_0 > 0.0f);
    }
    if(_S921)
    {
        return 0.0f;
    }
    var q_43 : Quat_0;
    var origin_21 : vec3<f32>;
    structure_pose_2(slot_23, u32(7) * slot_23, &(q_43), &(origin_21));
    var centre_5 : vec3<f32> = origin_21 + rotate_0(q_43, chunks_0[u32(4) * k_77].xyz);
    var bonds_4 : vec4<f32> = bond_centroid_0(k_77, q_43, origin_21, centre_5);
    if(!((bonds_4.w) > 0.0f))
    {
        return 0.0f;
    }
    var w_16 : Sym_0 = sym_add_0(sym_outer_0(at_30 - centre_5, f_18), sym_outer_0(bonds_4.xyz, (vec3<f32>(0) - f_18)));
    var before_0 : f32 = crush_rw_0[k_77].x;
    var step_9 : vec4<f32> = crush_two_0(w_16, min(tau_10, params_0.dt_0), w_16, 0.0f, k_77, before_0);
    var _S922 : u32 = u32(2) * k_77;
    var per_0 : f32 = crush_tab_0[_S922].w * crush_tab_0[_S922 + u32(1)].y;
    var energy_1 : f32 = (step_9.x - before_0) * per_0;
    var energy_2 : f32;
    var step_10 : vec4<f32>;
    if(energy_1 > budget_0)
    {
        var d_34 : f32 = before_0 + budget_0 / per_0;
        var _S923 : f32 = min(d_34, 1.0f);
        var _S924 : f32 = step_9.y;
        var _S925 : f32 = step_9.z;
        if(d_34 >= 1.0f)
        {
            energy_2 = 1.0f;
        }
        else
        {
            energy_2 = 0.0f;
        }
        step_10 = vec4<f32>(_S923, _S924, _S925, energy_2);
        energy_2 = budget_0;
    }
    else
    {
        step_10 = step_9;
        energy_2 = energy_1;
    }
    crush_record_0(k_77, step_10, u32(2));
    return energy_2;
}

fn cell_in_0( mask_0 : vec2<u32>,  c_30 : u32) -> bool
{
    return (((((mask_0[c_30 / u32(32)]) >> ((c_30 % u32(32))))) & (u32(1)))) != u32(0);
}

fn strike_crush_body_0( b_90 : u32,  s_43 : u32,  budget_1 : f32) -> f32
{
    var total_1 : u32;
    var _S926 : bool;
    var bill_2 : u32 = crush_0.r_bill_0 + u32(4) * b_90;
    var _S927 : u32 = bill_2 + u32(1);
    var p3_1 : vec4<f32> = crush_rw_0[bill_2 + u32(3)];
    var contact_27 : vec3<f32> = crush_rw_0[_S927].xyz;
    var d_35 : vec3<f32> = crush_rw_0[bill_2 + u32(2)].xyz;
    var _S928 : f32 = crush_rw_0[bill_2 + u32(2)].w;
    var _S929 : f32 = shots_0[u32(2) * s_43].y;
    var _S930 : u32 = a_cover_0() + u32(128) * b_90;
    var first_5 : f32 = 1.00000001504746622e+30f;
    var c_31 : u32 = u32(0);
    loop
    {
        if(c_31 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S931 : u32 = _S930 + u32(2) * c_31;
        var _S932 : u32 = atomicLoad(&(atom_0[_S931 + u32(1)]));
        if(_S932 == u32(4294967295))
        {
            c_31 = c_31 + u32(1);
            continue;
        }
        var rho_3 : f32;
        var _S933 : vec3<f32> = cover_offset_1(c_31, d_35, _S928, &(rho_3));
        var _S934 : u32 = atomicLoad(&(atom_0[_S931]));
        first_5 = min(first_5, unordered_0(_S934) + _S929 - sqrt(max(_S929 * _S929 - rho_3 * rho_3, 0.0f)));
        c_31 = c_31 + u32(1);
    }
    var counted_0 : vec2<u32> = vec2<u32>(u32(0));
    c_31 = u32(0);
    var total_2 : u32 = u32(0);
    loop
    {
        if(c_31 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S935 : u32 = _S930 + u32(2) * c_31;
        var _S936 : u32 = atomicLoad(&(atom_0[_S935 + u32(1)]));
        if(_S936 == u32(4294967295))
        {
            c_31 = c_31 + u32(1);
            continue;
        }
        var rho_4 : f32;
        var _S937 : vec3<f32> = cover_offset_1(c_31, d_35, _S928, &(rho_4));
        var _S938 : u32 = atomicLoad(&(atom_0[_S935]));
        if((unordered_0(_S938) + _S929 - sqrt(max(_S929 * _S929 - rho_4 * rho_4, 0.0f)) - first_5) <= (p3_1.z))
        {
            counted_0[c_31 / u32(32)] = ((counted_0[c_31 / u32(32)]) | (((u32(1) << ((c_31 % u32(32)))))));
            total_1 = total_2 + u32(1);
        }
        else
        {
            total_1 = total_2;
        }
        total_2 = total_1;
        c_31 = c_31 + u32(1);
    }
    var _S939 : vec3<f32> = vec3<f32>(0.0f);
    var energy_3 : f32;
    var point_1 : vec3<f32>;
    if(total_2 == u32(0))
    {
        var energy_4 : f32 = strike_load_0(u32(p3_1.w) - u32(1), b_90, d_35 * vec3<f32>(p3_1.x), contact_27, p3_1.y, budget_1);
        var _S940 : vec3<f32> = vec3<f32>(energy_4) * contact_27;
        energy_3 = energy_4;
        point_1 = _S940;
    }
    else
    {
        energy_3 = 0.0f;
        point_1 = _S939;
    }
    c_31 = u32(0);
    loop
    {
        if(c_31 < u32(64))
        {
        }
        else
        {
            break;
        }
        if(!cell_in_0(counted_0, c_31))
        {
            c_31 = c_31 + u32(1);
            continue;
        }
        var k_78 : u32 = atomicLoad(&(atom_0[_S930 + u32(2) * c_31 + u32(1)]));
        var seen_0 : bool = false;
        total_1 = u32(0);
        loop
        {
            if(total_1 < c_31)
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
                    var _S941 : u32 = atomicLoad(&(atom_0[_S930 + u32(2) * total_1 + u32(1)]));
                    _S926 = _S941 == k_78;
                }
                else
                {
                    _S926 = false;
                }
                seen_0 = _S926;
            }
            total_1 = total_1 + u32(1);
        }
        if(seen_0)
        {
            c_31 = c_31 + u32(1);
            continue;
        }
        var e_12 : u32 = c_31;
        var at_31 : vec3<f32> = _S939;
        var n_8 : u32 = u32(0);
        loop
        {
            if(e_12 < u32(64))
            {
            }
            else
            {
                break;
            }
            if(!cell_in_0(counted_0, e_12))
            {
                _S926 = true;
            }
            else
            {
                var _S942 : u32 = atomicLoad(&(atom_0[_S930 + u32(2) * e_12 + u32(1)]));
                _S926 = _S942 != k_78;
            }
            if(_S926)
            {
                e_12 = e_12 + u32(1);
                continue;
            }
            var rho_5 : f32;
            var _S943 : vec3<f32> = contact_27 + cover_offset_1(e_12, d_35, _S928, &(rho_5));
            var _S944 : u32 = atomicLoad(&(atom_0[_S930 + u32(2) * e_12]));
            var _S945 : u32 = n_8 + u32(1);
            at_31 = at_31 + (_S943 + d_35 * vec3<f32>(unordered_0(_S944)));
            n_8 = _S945;
            e_12 = e_12 + u32(1);
        }
        var _S946 : f32 = f32(n_8);
        var at_32 : vec3<f32> = at_31 / vec3<f32>(_S946);
        var took_0 : f32 = strike_load_0(k_78, b_90, d_35 * vec3<f32>((p3_1.x * _S946 / f32(total_2))), at_32, p3_1.y, budget_1 - energy_3);
        var point_2 : vec3<f32> = point_1 + vec3<f32>(took_0) * at_32;
        energy_3 = energy_3 + took_0;
        point_1 = point_2;
        c_31 = c_31 + u32(1);
    }
    crush_rw_0[bill_2] = vec4<f32>(energy_3, f32(s_43 + u32(1)), 0.0f, 0.0f);
    if(energy_3 > 0.0f)
    {
        crush_rw_0[_S927] = vec4<f32>(point_1 / vec3<f32>(energy_3), 0.0f);
    }
    return energy_3;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_crush(@builtin(global_invocation_id) id_85 : vec3<u32>)
{
    var s_44 : u32 = id_85.x;
    var _S947 : bool;
    if(s_44 >= (params_0.shot_count_0))
    {
        _S947 = true;
    }
    else
    {
        _S947 = (crush_0.enabled_0) == u32(0);
    }
    if(_S947)
    {
        return;
    }
    var sb_4 : u32 = u32(7) * (bitcast<u32>((shots_0[u32(2) * s_44].x)));
    var inv_mass_12 : f32 = bodies_0[sb_4].w;
    var v_22 : vec3<f32> = bodies_0[sb_4 + u32(3)].xyz;
    var budget_2 : f32;
    if(inv_mass_12 > 0.0f)
    {
        budget_2 = 0.5f * dot(v_22, v_22) / inv_mass_12;
    }
    else
    {
        budget_2 = 1.00000001504746622e+30f;
    }
    var b_91 : u32 = u32(0);
    loop
    {
        if(b_91 < (params_0.body_count_0))
        {
        }
        else
        {
            break;
        }
        var shot_14 : u32;
        var _S948 : bool = struck_2(b_91, &(shot_14));
        if(!_S948)
        {
            _S947 = true;
        }
        else
        {
            _S947 = shot_14 != s_44;
        }
        var _S949 : bool;
        if(_S947)
        {
            _S949 = true;
        }
        else
        {
            _S949 = (crush_rw_0[crush_0.r_bill_0 + u32(4) * b_91 + u32(3)].w) == 0.0f;
        }
        if(_S949)
        {
            b_91 = b_91 + u32(1);
            continue;
        }
        var _S950 : f32 = strike_crush_body_0(b_91, s_44, budget_2);
        budget_2 = budget_2 - _S950;
        b_91 = b_91 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_bill(@builtin(global_invocation_id) id_86 : vec3<u32>)
{
    var s_45 : u32 = id_86.x;
    var _S951 : bool;
    if(s_45 >= (params_0.shot_count_0))
    {
        _S951 = true;
    }
    else
    {
        _S951 = (crush_0.enabled_0) == u32(0);
    }
    if(_S951)
    {
        return;
    }
    var b_92 : u32 = u32(0);
    var energy_5 : f32 = 0.0f;
    loop
    {
        if(b_92 < (params_0.body_count_0))
        {
        }
        else
        {
            break;
        }
        var bill_3 : vec4<f32> = crush_rw_0[crush_0.r_bill_0 + u32(4) * b_92];
        if((crush_rw_0[crush_0.r_bill_0 + u32(4) * b_92].y) == f32(s_45 + u32(1)))
        {
            energy_5 = energy_5 + bill_3.x;
        }
        b_92 = b_92 + u32(1);
    }
    crush_rw_0[crush_0.r_shot_0 + s_45] = vec4<f32>(0.0f);
    if(!(energy_5 > 0.0f))
    {
        return;
    }
    var sb_5 : u32 = u32(7) * (bitcast<u32>((shots_0[u32(2) * s_45].x)));
    var inv_mass_13 : f32 = bodies_0[sb_5].w;
    if(!(inv_mass_13 > 0.0f))
    {
        return;
    }
    var m_11 : f32 = 1.0f / inv_mass_13;
    var v_23 : vec3<f32> = bodies_0[sb_5 + u32(3)].xyz;
    var speed_18 : f32 = length(v_23);
    var j_19 : f32 = m_11 * (speed_18 - sqrt(max(speed_18 * speed_18 - 2.0f * energy_5 / m_11, 0.0f)));
    var _S952 : bool = speed_18 > 0.0f;
    var _S953 : vec3<f32>;
    if(_S952)
    {
        _S953 = (vec3<f32>(0) - v_23) * vec3<f32>((j_19 / (m_11 * speed_18)));
    }
    else
    {
        _S953 = vec3<f32>(0.0f);
    }
    var region_6 : u32 = u32(0);
    loop
    {
        if(region_6 < u32(2))
        {
        }
        else
        {
            break;
        }
        var w_17 : u32 = sb_5 + region_6 * u32(7) * params_0.max_bodies_0 + u32(3);
        bodies_0[w_17] = vec4<f32>(bodies_0[w_17].xyz + _S953, bodies_0[w_17].w);
        region_6 = region_6 + u32(1);
    }
    if(_S952)
    {
        _S953 = v_23 * vec3<f32>((j_19 / speed_18));
    }
    else
    {
        _S953 = vec3<f32>(0.0f);
    }
    crush_rw_0[crush_0.r_shot_0 + s_45] = vec4<f32>(_S953, energy_5);
    return;
}

fn inverse_rotate_0( q_44 : Quat_0,  v_24 : vec3<f32>) -> vec3<f32>
{
    var c_32 : Quat_0;
    c_32.w_0 = q_44.w_0;
    c_32.x_0 = - q_44.x_0;
    c_32.y_0 = - q_44.y_0;
    c_32.z_0 = - q_44.z_0;
    return rotate_0(c_32, v_24);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_push(@builtin(global_invocation_id) id_87 : vec3<u32>)
{
    var b_93 : u32 = id_87.x;
    var _S954 : bool;
    if(b_93 >= (params_0.body_count_0))
    {
        _S954 = true;
    }
    else
    {
        _S954 = (params_0.shot_count_0) == u32(0);
    }
    if(_S954)
    {
        _S954 = true;
    }
    else
    {
        _S954 = (crush_0.enabled_0) == u32(0);
    }
    if(_S954)
    {
        return;
    }
    var _S955 : u32 = u32(4) * b_93;
    var bill_4 : vec4<f32> = crush_rw_0[crush_0.r_bill_0 + _S955];
    var _S956 : f32 = crush_rw_0[crush_0.r_bill_0 + _S955].x;
    if(!(_S956 > 0.0f))
    {
        return;
    }
    var paid_0 : vec4<f32> = crush_rw_0[crush_0.r_shot_0 + (u32(bill_4.y) - u32(1))];
    var bw_0 : u32 = u32(7) * b_93;
    var inv_mass_14 : f32 = bodies_0[bw_0].w;
    var _S957 : f32 = crush_rw_0[crush_0.r_shot_0 + (u32(bill_4.y) - u32(1))].w;
    if(!(_S957 > 0.0f))
    {
        _S954 = true;
    }
    else
    {
        _S954 = inv_mass_14 == 0.0f;
    }
    if(_S954)
    {
        return;
    }
    var j_20 : vec3<f32> = paid_0.xyz * vec3<f32>((_S956 / _S957));
    var o_14 : Quat_0 = quat_of_0(bodies_0[bw_0 + u32(1)]);
    var _S958 : vec3<f32> = rotate_0(o_14, bodies_0[bw_0 + u32(2)].xyz * inverse_rotate_0(o_14, cross(to_rubble_0(crush_rw_0[crush_0.r_bill_0 + _S955 + u32(1)].xyz) - bodies_0[bw_0].xyz, j_20)));
    var region_7 : u32 = u32(0);
    loop
    {
        if(region_7 < u32(2))
        {
        }
        else
        {
            break;
        }
        var w_18 : u32 = bw_0 + region_7 * u32(7) * params_0.max_bodies_0;
        var _S959 : u32 = w_18 + u32(3);
        bodies_0[_S959] = vec4<f32>(bodies_0[_S959].xyz + j_20 * vec3<f32>(inv_mass_14), bodies_0[_S959].w);
        var _S960 : u32 = w_18 + u32(4);
        bodies_0[_S960] = vec4<f32>(bodies_0[_S960].xyz + _S958, bodies_0[_S960].w);
        region_7 = region_7 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_bonds(@builtin(global_invocation_id) id_88 : vec3<u32>)
{
    var i_32 : u32 = id_88.x;
    var _S961 : bool;
    if(i_32 >= (params_0.bond_count_0))
    {
        _S961 = true;
    }
    else
    {
        _S961 = (crush_0.enabled_0) == u32(0);
    }
    if(_S961)
    {
        _S961 = true;
    }
    else
    {
        _S961 = work_0[w_alive_0() + i_32] == u32(0);
    }
    if(_S961)
    {
        return;
    }
    var bond_15 : vec4<u32> = bonds_0[i_32];
    var _S962 : u32 = bond_15.x;
    if((((((bitcast<u32>((crush_rw_0[_S962].w))) | ((bitcast<u32>((crush_rw_0[bond_15.y].w)))))) & (u32(2)))) == u32(0))
    {
        return;
    }
    work_0[w_alive_0() + i_32] = u32(0);
    var _S963 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S962)]), u32(1));
    var _S964 : u32 = a_counters_0();
    var _S965 : u32 = atomicAdd(&(atom_0[_S964]), u32(1));
    var _S966 : u32 = atomicAdd(&(atom_0[_S964 + u32(1)]), u32(1));
    event_broken_0(i_32);
    return;
}

fn crush_pieces_0( k_79 : u32) -> u32
{
    if((crush_0.mode_1) == u32(1))
    {
        return crush_0.fragments_0;
    }
    if((crush_0.mode_1) == u32(2))
    {
        return u32(crush_tab_0[u32(2) * k_79 + u32(1)].w + 0.5f);
    }
    return u32(0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_mark(@builtin(global_invocation_id) id_89 : vec3<u32>)
{
    var c_33 : u32 = id_89.x;
    var _S967 : bool;
    if(c_33 >= (params_0.chunk_count_0))
    {
        _S967 = true;
    }
    else
    {
        _S967 = (crush_0.enabled_0) == u32(0);
    }
    if(_S967)
    {
        return;
    }
    var now_0 : u32;
    if((((crush_flags_0(c_33)) & (u32(2)))) != u32(0))
    {
        now_0 = u32(1);
    }
    else
    {
        now_0 = u32(0);
    }
    var _S968 : bool = now_0 != u32(0);
    if(_S968)
    {
        now_0 = (u32(1048576) | ((crush_pieces_0(c_33))));
    }
    else
    {
        now_0 = u32(0);
    }
    work_0[w_new_0() + c_33] = now_0;
    if(_S968)
    {
        var _S969 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(10)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_crush_gate(@builtin(global_invocation_id) id_90 : vec3<u32>)
{
    var run_2 : bool;
    if((crush_0.enabled_0) != u32(0))
    {
        var _S970 : u32 = atomicLoad(&(atom_0[a_counters_0() + u32(10)]));
        run_2 = _S970 != u32(0);
    }
    else
    {
        run_2 = false;
    }
    var _S971 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S971 + u32(10)]), u32(0));
    atomicStore(&(atom_0[_S971 + u32(7)]), u32(0));
    var _S972 : u32;
    if(run_2)
    {
        var _S973 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S972 = _S973;
    }
    else
    {
        _S972 = u32(0);
    }
    set_args_0(u32(14), _S972);
    if(run_2)
    {
        var _S974 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S972 = _S974;
    }
    else
    {
        _S972 = u32(0);
    }
    set_args_0(u32(15), _S972);
    var _S975 : i32;
    if(run_2)
    {
        _S975 = i32(1);
    }
    else
    {
        _S975 = i32(0);
    }
    set_args_0(u32(16), u32(_S975));
    return;
}

fn write_body_0( slot_24 : u32,  pos_1 : vec3<f32>,  inv_mass_15 : f32,  orientation_1 : Quat_0,  inv_inertia_1 : vec3<f32>,  flags_3 : u32,  v_25 : vec3<f32>,  w_19 : vec3<f32>)
{
    var region_8 : u32 = u32(0);
    loop
    {
        if(region_8 < u32(2))
        {
        }
        else
        {
            break;
        }
        var b_94 : u32 = u32(7) * (slot_24 + region_8 * params_0.max_bodies_0);
        bodies_0[b_94] = vec4<f32>(pos_1, inv_mass_15);
        bodies_0[b_94 + u32(1)] = quat_vec_0(orientation_1);
        bodies_0[b_94 + u32(2)] = vec4<f32>(inv_inertia_1, (bitcast<f32>((flags_3))));
        bodies_0[b_94 + u32(3)] = vec4<f32>(v_25, 0.0f);
        bodies_0[b_94 + u32(4)] = vec4<f32>(w_19, (bitcast<f32>((frame_step_0()))));
        bodies_0[b_94 + u32(5)] = vec4<f32>(0.0f);
        bodies_0[b_94 + u32(6)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
        region_8 = region_8 + u32(1);
    }
    return;
}

fn crush_particle_spot_0( k_80 : u32,  i_33 : u32) -> vec3<f32>
{
    var h_8 : u32 = ((k_80 * u32(2654435761)) ^ ((i_33 * u32(2246822519))));
    var h_9 : u32 = ((h_8 ^ (((h_8 >> (u32(15))))))) * u32(739982445);
    var h_10 : u32 = ((h_9 ^ (((h_9 >> (u32(12))))))) * u32(695872825);
    var h_11 : u32 = (h_10 ^ (((h_10 >> (u32(15))))));
    return vec3<f32>(f32((h_11 & (u32(1023)))), f32((((h_11 >> (u32(10)))) & (u32(1023)))), f32(((h_11 * u32(2654435761) + u32(2135587861)) & (u32(1023))))) / vec3<f32>(511.5f) - vec3<f32>(1.0f);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_apply(@builtin(global_invocation_id) id_91 : vec3<u32>)
{
    var k_81 : u32 = id_91.x;
    var _S976 : bool;
    if(k_81 >= (params_0.chunk_count_0))
    {
        _S976 = true;
    }
    else
    {
        _S976 = (crush_0.enabled_0) == u32(0);
    }
    if(_S976)
    {
        return;
    }
    var cs_0 : vec4<f32> = crush_rw_0[k_81];
    var flags_4 : u32 = (bitcast<u32>((crush_rw_0[k_81].w)));
    if(((flags_4 & (u32(2)))) == u32(0))
    {
        return;
    }
    crush_rw_0[k_81] = vec4<f32>(cs_0.xyz, (bitcast<f32>(((flags_4 & (u32(4294967293)))))));
    var slot_25 : u32 = body_of_0(k_81);
    var bw_1 : u32 = u32(7) * slot_25;
    var b0_2 : vec4<f32> = bodies_0[bw_1];
    var orientation_2 : Quat_0 = quat_of_0(bodies_0[bw_1 + u32(1)]);
    var v_26 : vec3<f32> = bodies_0[bw_1 + u32(3)].xyz;
    var w_20 : vec3<f32> = bodies_0[bw_1 + u32(4)].xyz;
    var _S977 : u32 = u32(4) * k_81;
    var mass_2 : f32 = chunks_0[_S977].w;
    var volume_1 : f32 = crush_tab_0[u32(2) * k_81 + u32(1)].y;
    var scanned_0 : u32 = work_0[w_rank_0() + k_81];
    var rank_0 : u32 = work_0[w_rank_0() + k_81] / u32(1048576);
    var pieces_0 : u32 = crush_pieces_0(k_81);
    if(rank_0 >= (crush_0.max_events_0))
    {
        var _S978 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(32));
    }
    var _S979 : vec3<f32> = vec3<f32>(0.0f);
    if((crush_0.mode_1) == u32(1))
    {
        _S976 = (crush_0.fragments_0) > u32(0);
    }
    else
    {
        _S976 = false;
    }
    var first_6 : u32;
    var n_9 : u32;
    var i_34 : u32;
    var half_2 : vec3<f32>;
    if(_S976)
    {
        var _S980 : u32 = a_counters_0();
        var _S981 : u32 = atomicLoad(&(atom_0[_S980 + u32(8)]));
        var base_6 : u32 = _S981 + ((scanned_0 & (u32(1048575))));
        if((base_6 + crush_0.fragments_0) > (crush_0.pool_size_0))
        {
            var _S982 : u32 = atomicOr(&(atom_0[_S980 + u32(3)]), u32(16));
            first_6 = u32(4294967295);
        }
        else
        {
            first_6 = base_6;
        }
        var q_45 : Quat_0;
        var origin_22 : vec3<f32>;
        structure_pose_5(slot_25, bw_1, &(q_45), &(origin_22));
        var _S983 : vec3<f32> = origin_22 + rotate_0(q_45, chunks_0[_S977].xyz);
        var _S984 : u32 = u32(3) * k_81;
        var box_0 : Quat_0 = normalized_0(quat_mul_0(q_45, quat_of_0(geometry_0[_S984 + u32(2)])));
        var h_12 : vec3<f32> = geometry_0[_S984 + u32(1)].xyz;
        var g_2 : vec3<u32> = vec3<u32>(vec3<i32>(i32(1)));
        var cell_0 : vec3<f32> = h_12;
        n_9 = crush_0.fragments_0;
        loop
        {
            if(n_9 > u32(1))
            {
            }
            else
            {
                break;
            }
            if((cell_0.x) >= (cell_0.y))
            {
                _S976 = (cell_0.x) >= (cell_0.z);
            }
            else
            {
                _S976 = false;
            }
            if(_S976)
            {
                i_34 = u32(0);
            }
            else
            {
                if((cell_0.y) >= (cell_0.z))
                {
                    i_34 = u32(1);
                }
                else
                {
                    i_34 = u32(2);
                }
            }
            g_2[i_34] = g_2[i_34] * u32(2);
            cell_0[i_34] = cell_0[i_34] * 0.5f;
            n_9 = (n_9 >> (u32(1)));
        }
        var box_volume_0 : f32 = 8.0f * h_12.x * h_12.y * h_12.z;
        var shrink_0 : f32;
        if(box_volume_0 > 0.0f)
        {
            shrink_0 = min(0.94999998807907104f, pow(volume_1 / box_volume_0, 0.3333333432674408f));
        }
        else
        {
            shrink_0 = 0.94999998807907104f;
        }
        var half_3 : vec3<f32> = cell_0 * vec3<f32>(shrink_0);
        var mf_0 : f32 = mass_2 / f32(crush_0.fragments_0);
        var _S985 : f32 = half_3.y;
        var _S986 : f32 = _S985 * _S985;
        var _S987 : f32 = half_3.z;
        var _S988 : f32 = _S987 * _S987;
        var _S989 : f32 = half_3.x;
        var _S990 : f32 = _S989 * _S989;
        var inertia_2 : vec3<f32> = vec3<f32>((mf_0 / 3.0f)) * vec3<f32>(_S986 + _S988, _S990 + _S988, _S990 + _S986);
        var _S991 : vec3<f32> = vec3<f32>(1.0f / inertia_2.x, 1.0f / inertia_2.y, 1.0f / inertia_2.z);
        var _S992 : Quat_0 = rot_to_rubble_0(box_0);
        var _S993 : f32 = chunks_0[_S977 + u32(3)].x;
        var _S994 : f32 = min(_S989, min(_S985, _S987));
        i_34 = u32(0);
        loop
        {
            if(i_34 < (crush_0.fragments_0))
            {
                _S976 = first_6 != u32(4294967295);
            }
            else
            {
                _S976 = false;
            }
            if(_S976)
            {
            }
            else
            {
                break;
            }
            var _S995 : u32 = i_34 % g_2.x;
            var _S996 : u32 = i_34 / g_2.x;
            var _S997 : u32 = _S996 % g_2.y;
            var _S998 : u32 = i_34 / (g_2.x * g_2.y);
            var x_12 : vec3<f32> = to_rubble_0(_S983 + rotate_0(box_0, (vec3<f32>(0) - h_12) + cell_0 * (vec3<f32>(2.0f) * vec3<f32>(vec3<u32>(_S995, _S997, _S998)) + vec3<f32>(1.0f))));
            var entry_10 : vec4<f32> = crush_tab_0[crush_0.t_pool_0 + first_6 + i_34];
            var fs_0 : u32 = (bitcast<u32>((entry_10.x)));
            var fc_3 : u32 = (bitcast<u32>((entry_10.y)));
            write_body_0(fs_0, x_12, 1.0f / mf_0, _S992, _S991, u32(9), v_26 + cross(w_20, x_12 - b0_2.xyz), w_20);
            var _S999 : u32 = u32(4) * fc_3;
            var _S1000 : u32 = _S999 + u32(1);
            colliders_0[_S1000] = vec4<f32>(0.0f, 0.0f, 0.0f, colliders_0[_S1000].w);
            colliders_0[_S999 + u32(2)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
            colliders_0[_S999 + u32(3)] = vec4<f32>(half_3, length(half_3));
            var _S1001 : u32 = u32(2) * fc_3;
            collider_info_0[_S1001] = vec4<u32>(u32(4294967295), fs_0, u32(2), collider_info_0[_S1001].w);
            collider_info_0[_S1001 + u32(1)] = vec4<u32>((bitcast<u32>((mf_0))), (bitcast<u32>((_S993))), (bitcast<u32>((_S994))), u32(0));
            i_34 = i_34 + u32(1);
        }
        half_2 = half_3;
    }
    else
    {
        first_6 = u32(4294967295);
        half_2 = _S979;
    }
    if((crush_0.mode_1) == u32(2))
    {
        _S976 = pieces_0 > u32(0);
    }
    else
    {
        _S976 = false;
    }
    if(_S976)
    {
        var _S1002 : u32 = a_counters_0();
        var _S1003 : u32 = atomicLoad(&(atom_0[_S1002 + u32(8)]));
        var base_7 : u32 = _S1003 + ((scanned_0 & (u32(1048575))));
        if((base_7 + pieces_0) > (crush_0.pool_size_0))
        {
            var _S1004 : u32 = atomicOr(&(atom_0[_S1002 + u32(3)]), u32(16));
        }
        else
        {
            first_6 = base_7;
        }
        var q_46 : Quat_0;
        var origin_23 : vec3<f32>;
        structure_pose_5(slot_25, bw_1, &(q_46), &(origin_23));
        var _S1005 : vec3<f32> = origin_23 + rotate_0(q_46, chunks_0[_S977].xyz);
        var _S1006 : u32 = u32(3) * k_81;
        var _S1007 : Quat_0 = normalized_0(quat_mul_0(q_46, quat_of_0(geometry_0[_S1006 + u32(2)])));
        var h_13 : vec3<f32> = geometry_0[_S1006 + u32(1)].xyz;
        var _S1008 : f32 = f32(pieces_0);
        var _S1009 : f32 = max(pow(0.75f * volume_1 / (3.14159274101257324f * _S1008), 0.3333333432674408f), 0.0020000000949949f);
        var _S1010 : vec3<f32> = vec3<f32>(_S1009, _S1009, _S1009);
        var _S1011 : f32 = mass_2 / _S1008;
        var _S1012 : f32 = chunks_0[_S977 + u32(3)].x;
        var _S1013 : vec3<f32> = max(h_13 - vec3<f32>(_S1009), vec3<f32>(0.25f) * h_13);
        i_34 = u32(0);
        var mean_4 : vec3<f32> = _S979;
        loop
        {
            if(i_34 < pieces_0)
            {
            }
            else
            {
                break;
            }
            var mean_5 : vec3<f32> = mean_4 + crush_particle_spot_0(k_81, i_34);
            i_34 = i_34 + u32(1);
            mean_4 = mean_5;
        }
        var mean_6 : vec3<f32> = mean_4 / vec3<f32>(_S1008);
        i_34 = u32(0);
        loop
        {
            if(i_34 < pieces_0)
            {
                _S976 = first_6 != u32(4294967295);
            }
            else
            {
                _S976 = false;
            }
            if(_S976)
            {
            }
            else
            {
                break;
            }
            var x_13 : vec3<f32> = to_rubble_0(_S1005 + rotate_0(_S1007, _S1013 * (crush_particle_spot_0(k_81, i_34) - mean_6)));
            var entry_11 : vec4<f32> = crush_tab_0[crush_0.t_pool_0 + first_6 + i_34];
            var fs_1 : u32 = (bitcast<u32>((entry_11.x)));
            var fc_4 : u32 = (bitcast<u32>((entry_11.y)));
            var _S1014 : f32 = 1.0f / _S1011;
            const _S1015 : vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
            var _S1016 : Quat_0 = quat_of_0(_S1015);
            if((crush_0.particle_ccd_0) != u32(0))
            {
                n_9 = u32(33024);
            }
            else
            {
                n_9 = u32(0);
            }
            write_body_0(fs_1, x_13, _S1014, _S1016, _S979, (u32(9) | (n_9)), v_26 + cross(w_20, x_13 - b0_2.xyz), _S979);
            var _S1017 : u32 = u32(4) * fc_4;
            var _S1018 : u32 = _S1017 + u32(1);
            colliders_0[_S1018] = vec4<f32>(0.0f, 0.0f, 0.0f, colliders_0[_S1018].w);
            colliders_0[_S1017 + u32(2)] = _S1015;
            colliders_0[_S1017 + u32(3)] = vec4<f32>(_S1009, 0.0f, 0.0f, _S1009);
            var _S1019 : u32 = u32(2) * fc_4;
            collider_info_0[_S1019] = vec4<u32>(u32(4294967295), fs_1, u32(2), collider_info_0[_S1019].w);
            collider_info_0[_S1019 + u32(1)] = vec4<u32>((bitcast<u32>((_S1011))), (bitcast<u32>((_S1012))), (bitcast<u32>((_S1009))), u32(0));
            i_34 = i_34 + u32(1);
        }
        half_2 = _S1010;
    }
    if(rank_0 < (crush_0.max_events_0))
    {
        var ev_0 : u32 = crush_0.r_events_0 + u32(6) * rank_0;
        crush_rw_0[ev_0] = vec4<f32>((bitcast<f32>((k_81))), (bitcast<f32>((first_6))), (bitcast<f32>(((flags_4 >> (u32(4)))))), cs_0.x);
        crush_rw_0[ev_0 + u32(1)] = vec4<f32>(b0_2.xyz, mass_2);
        crush_rw_0[ev_0 + u32(2)] = quat_vec_0(orientation_2);
        crush_rw_0[ev_0 + u32(3)] = vec4<f32>(v_26, cs_0.y);
        crush_rw_0[ev_0 + u32(4)] = vec4<f32>(w_20, cs_0.z);
        crush_rw_0[ev_0 + u32(5)] = vec4<f32>(half_2, volume_1);
    }
    write_body_0(slot_25, vec3<f32>(f32(slot_25 % u32(1024)) * crush_0.park_step_0, crush_0.park_y_0, f32(slot_25 / u32(1024)) * crush_0.park_step_0), 0.0f, orientation_2, _S979, u32(0), _S979, _S979);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_crush_end(@builtin(global_invocation_id) id_92 : vec3<u32>)
{
    var total_3 : u32 = work_0[scan_new_blocks_0() + (params_0.chunk_count_0 + u32(1023)) / u32(1024)];
    var events_0 : u32 = work_0[scan_new_blocks_0() + (params_0.chunk_count_0 + u32(1023)) / u32(1024)] / u32(1048576);
    var _S1020 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S1020 + u32(7)]), events_0);
    var _S1021 : u32 = atomicAdd(&(atom_0[_S1020 + u32(9)]), events_0);
    var _S1022 : u32 = atomicAdd(&(atom_0[_S1020 + u32(8)]), (total_3 & (u32(1048575))));
    return;
}

