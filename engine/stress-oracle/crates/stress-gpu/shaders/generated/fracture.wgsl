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
    @align(16) lp_scale_0 : f32,
    @align(4) cs_sweeps_0 : u32,
    @align(8) cs_row_cap_0 : u32,
    @align(4) cs_c1_0 : f32,
    @align(16) cs_c2_0 : f32,
    @align(4) cs_row_offset_0 : u32,
    @align(8) ext_sides_0 : u32,
    @align(4) pad8_0 : f32,
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
    @align(8) pad0_1 : f32,
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
    return a_counters_0() + u32(20);
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
    var _S89 : u32 = bond_0.x;
    var b_23 : u32 = body_of_0(_S89);
    var _S90 : bool = il_struck_0(b_23);
    if(!_S90)
    {
        return;
    }
    var _S91 : u32 = lp_body_0(b_23);
    var s0_0 : vec4<f32> = lp_0[_S91];
    var s1_0 : vec4<f32> = lp_0[_S91 + u32(1)];
    var outcome_3 : u32 = (bitcast<u32>((lp_0[_S91].w)));
    if(((outcome_3 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_7 : vec3<f32>;
    var d_9 : vec3<f32>;
    var speed_6 : f32;
    il_path_2(b_23, &(contact_7), &(d_9), &(speed_6));
    var entry_2 : u32 = il_entry_0(b_23);
    var _S92 : f32 = max(il_striker_0(b_23).radius_0, 0.00100000004749745f);
    var h_3 : f32 = s0_0.x;
    var pa_0 : IlPlace_0 = il_place_0(_S89, entry_2, contact_7, d_9, _S92, h_3);
    var pb_0 : IlPlace_0 = il_place_0(bond_0.y, entry_2, contact_7, d_9, _S92, h_3);
    if((il_plug_0(outcome_3, _S92, h_3, pa_0)) != (il_plug_0(outcome_3, _S92, h_3, pb_0)))
    {
        cut_0 = true;
    }
    else
    {
        var _S93 : f32 = s1_0.y;
        cut_0 = (il_scab_0(outcome_3, _S92, h_3, _S93, pa_0)) != (il_scab_0(outcome_3, _S92, h_3, _S93, pb_0));
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = il_crater_0(outcome_3, _S92, s1_0.x, s1_0.z, pa_0);
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = il_crater_0(outcome_3, _S92, s1_0.x, s1_0.z, pb_0);
    }
    if(cut_0)
    {
        work_0[w_alive_0() + i_0] = u32(0);
        var _S94 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_23]), u32(1));
        var _S95 : u32 = a_counters_0();
        var _S96 : u32 = atomicAdd(&(atom_0[_S95]), u32(1));
        var _S97 : u32 = atomicAdd(&(atom_0[_S95 + u32(1)]), u32(1));
        var _S98 : u32 = atomicAdd(&(atom_0[_S95 + u32(17)]), u32(1));
    }
    return;
}

fn to_rubble_0( v_7 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_7.x, v_7.z, - v_7.y);
}

fn read64_0( at_5 : u32,  scale_0 : f32) -> f32
{
    var _S99 : u32 = atomicLoad(&(atom_0[at_5 + u32(1)]));
    var hi_1 : i32 = (bitcast<i32>((_S99)));
    var lo_1 : u32 = atomicLoad(&(atom_0[at_5]));
    return (f32(hi_1) * 4.294967296e+09f + f32(lo_1)) / scale_0;
}

fn il_carry_0( k_13 : u32)
{
    if(k_13 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S100 : u32 = w_carry_0() + k_13;
    var c_2 : u32 = work_0[_S100];
    if(work_0[_S100] == u32(0))
    {
        return;
    }
    work_0[_S100] = u32(0);
    var b0_0 : u32 = c_2 - u32(1);
    var _S101 : bool = il_struck_0(b0_0);
    if(!_S101)
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
    var _S102 : u32 = u32(7) * sk_0.body_0;
    var v_shot_0 : vec3<f32> = bodies_0[_S102 + u32(3)].xyz;
    var closing_0 : f32 = dot(v_shot_0 - v_wall_0, dr_0);
    var v_r_0 : f32 = sk_0.mass_0 * closing_0 / (sk_0.mass_0 + plug_mass_0);
    var _S103 : vec3<f32> = v_shot_0 + dr_0 * vec3<f32>((v_r_0 - closing_0));
    var _S104 : vec3<f32> = v_wall_0 + dr_0 * vec3<f32>(v_r_0);
    var _S105 : vec3<f32> = bodies_0[_S102].xyz + dr_0 * vec3<f32>(((speed_7 - v_r_0) * clamp(il_t_in_0(b0_0), 0.0f, params_0.dt_0)));
    var _S106 : u32 = body_of_0(k_13);
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
        bodies_0[s_at_0].x = _S105.x;
        bodies_0[s_at_0].y = _S105.y;
        bodies_0[s_at_0].z = _S105.z;
        bodies_0[s_at_0 + u32(3)].x = _S103.x;
        bodies_0[s_at_0 + u32(3)].y = _S103.y;
        bodies_0[s_at_0 + u32(3)].z = _S103.z;
        bodies_0[u32(7) * (_S106 + region_0 * params_0.max_bodies_0) + u32(3)].x = _S104.x;
        bodies_0[u32(7) * (_S106 + region_0 * params_0.max_bodies_0) + u32(3)].y = _S104.y;
        bodies_0[u32(7) * (_S106 + region_0 * params_0.max_bodies_0) + u32(3)].z = _S104.z;
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
        var _S107 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
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
    var _S108 : u32;
    if(v_8 < i32(0))
    {
        _S108 = u32(4294967295);
    }
    else
    {
        _S108 = u32(0);
    }
    var hi_2 : u32 = _S108 + carry_0;
    if(hi_2 != u32(0))
    {
        var _S109 : u32 = atomicAdd(&(atom_0[at_6 + u32(1)]), hi_2);
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
    var _S110 : bool = il_struck_0(b_24);
    if(!_S110)
    {
        return;
    }
    var _S111 : u32 = lp_body_0(b_24);
    var s0_1 : vec4<f32> = lp_0[_S111];
    var s1_1 : vec4<f32> = lp_0[_S111 + u32(1)];
    var outcome_4 : u32 = (bitcast<u32>((lp_0[_S111].w)));
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
    var _S112 : f32 = max(sk_1.radius_0, 0.00100000004749745f);
    var _S113 : f32 = s0_1.x;
    var p_10 : IlPlace_0 = il_place_0(k_14, entry_3, contact_9, d_11, _S112, _S113);
    var plug_0 : bool = il_plug_0(outcome_4, _S112, _S113, p_10);
    if((sk_1.launch_0) != u32(0))
    {
        var t_3 : vec2<f32> = il_interval_0(k_14, contact_9 - d_11 * vec3<f32>(sk_1.bound_0), d_11, sk_1.bound_0);
        var _S114 : f32 = t_3.x;
        var _S115 : f32 = t_3.y;
        var path_0 : bool;
        if(_S114 <= _S115)
        {
            path_0 = _S114 <= (_S113 + 2.0f * sk_1.bound_0);
        }
        else
        {
            path_0 = false;
        }
        if(path_0)
        {
            path_0 = _S115 >= 0.0f;
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
            path_0 = il_scab_0(outcome_4, _S112, _S113, s1_1.y, p_10);
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            path_0 = il_crater_0(outcome_4, _S112, s1_1.x, s1_1.z, p_10);
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
    var _S116 : f32 = mat_0.dif_0.z;
    var f_0 : f32;
    if(r_16 <= _S116)
    {
        f_0 = fpow_0(fdiv_0(r_16, ref_0), mat_0.dif_0.y);
    }
    else
    {
        f_0 = fpow_0(fdiv_0(_S116, ref_0), mat_0.dif_0.y) * fpow_0(fdiv_0(r_16, _S116), mat_0.dif_0.w);
    }
    return clamp(f_0, 1.0f, mat_0.misc_0.x);
}

fn il_impulse_0( b_27 : u32,  sb_0 : u32,  m_shot_0 : f32,  contact_10 : vec3<f32>,  d_12 : vec3<f32>,  j_0 : f32)
{
    var jr_0 : vec3<f32> = to_rubble_0(d_12) * vec3<f32>(j_0);
    var at_7 : u32 = u32(7) * b_27;
    var inv_mass_1 : f32 = bodies_0[at_7].w;
    var _S117 : vec3<f32> = jr_0 * vec3<f32>(inv_mass_1);
    var _S118 : vec3<f32> = vec3<f32>(0.0f);
    var dw_0 : vec3<f32>;
    if(inv_mass_1 > 0.0f)
    {
        var q_12 : Quat_0 = quat_of_0(bodies_0[at_7 + u32(1)]);
        dw_0 = rotate_0(q_12, bodies_0[at_7 + u32(2)].xyz * rotate_0(conj_0(q_12), cross(to_rubble_0(contact_10) - bodies_0[at_7].xyz, jr_0)));
    }
    else
    {
        dw_0 = _S118;
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
        var _S119 : vec3<f32> = bodies_0[s_at_1].xyz - jr_0 / vec3<f32>(m_shot_0);
        bodies_0[s_at_1].x = _S119.x;
        bodies_0[s_at_1].y = _S119.y;
        bodies_0[s_at_1].z = _S119.z;
        var w_at_0 : u32 = u32(7) * (b_27 + region_1 * params_0.max_bodies_0);
        var _S120 : u32 = w_at_0 + u32(3);
        var _S121 : vec3<f32> = bodies_0[_S120].xyz + _S117;
        bodies_0[_S120].x = _S121.x;
        bodies_0[_S120].y = _S121.y;
        bodies_0[_S120].z = _S121.z;
        var _S122 : u32 = w_at_0 + u32(4);
        var _S123 : vec3<f32> = bodies_0[_S122].xyz + dw_0;
        bodies_0[_S122].x = _S123.x;
        bodies_0[_S122].y = _S123.y;
        bodies_0[_S122].z = _S123.z;
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
        var _S124 : bool = il_struck_0(b_28);
        damage_0 = !_S124;
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
    var _S125 : f32 = max(sk_2.radius_0, 0.00100000004749745f);
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
        var _S126 : u32 = lp_body_0(b_28);
        var _S127 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S126] = _S127;
        lp_0[_S126 + u32(1)] = _S127;
        return;
    }
    var _S128 : u32 = atomicLoad(&(atom_0[a_il_0(b_28)]));
    var _S129 : f32 = max((bitcast<f32>((_S128))) - _S125, 9.99999997475242708e-07f);
    var o_2 : u32 = params_0.ext_chunks_0 + u32(4) * entry_4;
    var mat0_0 : vec4<f32> = ext_0[o_2 + u32(1)];
    var dm_0 : JointMaterial_0;
    var _S130 : vec4<f32> = vec4<f32>(0.0f);
    dm_0.strength_0 = _S130;
    dm_0.energy_0 = _S130;
    dm_0.dif_0 = ext_0[o_2 + u32(2)];
    var mat3_0 : vec4<f32> = ext_0[o_2 + u32(3)];
    var _S131 : f32 = mat3_0.x;
    dm_0.misc_0 = vec4<f32>(_S131, 0.0f, 0.0f, 0.0f);
    dm_0.kind_flags_0 = vec4<u32>(u32(0));
    var nu_0 : f32 = mat3_0.y;
    var e_target_0 : f32 = mat3_0.z / (1.0f - nu_0 * nu_0);
    var e_star_0 : f32 = 1.0f / (1.0f / sk_2.modulus_0 + 1.0f / e_target_0);
    var _S132 : f32 = max(speed_9, 0.00100000004749745f);
    var _S133 : f32 = sqrt(_S125);
    var h0_0 : f32 = pow(15.0f * sk_2.m_eff_0 * _S132 * _S132 / (16.0f * e_star_0 * _S133), 0.40000000596046448f);
    var t_c_0 : f32 = 2.94000005722045898f * h0_0 / _S132;
    var force_0 : f32 = 1.33333337306976318f * e_star_0 * _S133 * pow(h0_0, 1.5f);
    var a_5 : f32 = sqrt(_S125 * h0_0);
    var p0_0 : f32 = 3.0f * force_0 / (6.28318548202514648f * a_5 * a_5);
    var ring_0 : f32 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    var k_15 : f32;
    if(_S131 > 0.0f)
    {
        k_15 = dif_factor_0(dm_0, _S132 / _S129);
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
    var _S134 : f32 = fc_0 * (3.14159274101257324f * _S125 * _S125);
    var _S135 : f32 = min(_S134 * _S129, 1.57079637050628662f * (2.0f * _S125 + _S129) * _S129 * _S129 * tau_0);
    var _S136 : f32 = 2.0f * _S135;
    var v_bl_0 : f32 = sqrt(_S136 / sk_2.m_eff_0);
    var perforate_0 : bool;
    if(damage_0)
    {
        perforate_0 = _S132 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    var r2_0 : f32 = _S125 + _S129;
    var _S137 : f32 = mat0_0.x;
    var plug_1 : f32 = _S137 * 3.14159274101257324f * _S129 / 3.0f * (_S125 * _S125 + _S125 * r2_0 + r2_0 * r2_0);
    var v_res_0 : f32;
    if(perforate_0)
    {
        v_res_0 = sk_2.mass_0 / (sk_2.mass_0 + plug_1) * sqrt(max(_S132 * _S132 - v_bl_0 * v_bl_0, 0.0f));
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
        depth_1 = min(0.5f * sk_2.m_eff_0 * _S132 * _S132 / _S134, _S129);
    }
    else
    {
        depth_1 = 0.0f;
    }
    var crater_0 : bool = depth_1 > a_5;
    var c_3 : f32 = sqrt(e_target_0 / _S137);
    var z_t_0 : f32 = _S137 * c_3;
    var rho_p_0 : f32 = sk_2.mass_0 / (4.18879032135009766f * _S125 * _S125 * _S125);
    var z_p_0 : f32 = rho_p_0 * sqrt(sk_2.modulus_0 / rho_p_0);
    var mean_pressure_0 : f32 = force_0 / (3.14159274101257324f * a_5 * a_5);
    var _S138 : f32 = min(min(z_t_0 * _S132 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_0);
    var source_radius_0 : f32;
    if(mean_pressure_0 > fc_0)
    {
        source_radius_0 = max(a_5, _S125);
    }
    else
    {
        source_radius_0 = a_5;
    }
    var pulse_0 : f32 = c_3 * t_c_0;
    var back_0 : f32 = _S138 * min(source_radius_0 / _S129, 1.0f) * min(2.0f * _S129 / pulse_0, 1.0f);
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
        scab_1 = 0.5f * min(pulse_0, _S129);
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
    var _S139 : u32;
    if(crater_0)
    {
        _S139 = u32(2);
    }
    else
    {
        _S139 = u32(0);
    }
    var _S140 : u32 = (region_2 | (_S139));
    if(spall_0)
    {
        region_2 = u32(4);
    }
    else
    {
        region_2 = u32(0);
    }
    var _S141 : u32 = (_S140 | (region_2));
    if(perforate_0)
    {
        region_2 = u32(8);
    }
    else
    {
        region_2 = u32(0);
    }
    var _S142 : u32 = lp_body_0(b_28);
    lp_0[_S142] = vec4<f32>(_S129, v_bl_0, v_res_0, (bitcast<f32>(((_S141 | (region_2))))));
    if(crater_0)
    {
        k_15 = depth_1;
    }
    else
    {
        k_15 = 0.0f;
    }
    lp_0[_S142 + u32(1)] = vec4<f32>(k_15, scab_1, a_5, _S135);
    if(perforate_0)
    {
        var _S143 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(14)]), u32(1));
    }
    else
    {
        if(spall_0)
        {
            var _S144 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(13)]), u32(1));
        }
        else
        {
            if(crater_0)
            {
                var _S145 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(12)]), u32(1));
            }
            else
            {
                var _S146 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(11)]), u32(1));
            }
        }
    }
    if(perforate_0)
    {
        if((params_0.fracture_work_0) != u32(0))
        {
            il_impulse_0(b_28, sk_2.body_0, sk_2.mass_0, contact_11, d_13, sk_2.m_eff_0 * (_S132 - sqrt(max(_S132 * _S132 - _S136 / sk_2.m_eff_0, 0.0f))));
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_28, sk_2.body_0, sk_2.mass_0, contact_11, d_13, sk_2.m_eff_0 * _S132);
            var _S147 : vec3<f32> = to_rubble_0(d_13) * vec3<f32>((_S132 * clamp(t_in_5, 0.0f, params_0.dt_0)));
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
                var _S148 : u32 = u32(7) * (sk_2.body_0 + region_2 * params_0.max_bodies_0);
                var _S149 : vec3<f32> = bodies_0[_S148].xyz + _S147;
                bodies_0[_S148].x = _S149.x;
                bodies_0[_S148].y = _S149.y;
                bodies_0[_S148].z = _S149.z;
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
    var _S150 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S150;
}

fn group_uniform_float_0( value_2 : f32,  thread_1 : u32) -> f32
{
    var _S151 : u32 = group_uniform_0((bitcast<u32>((value_2))), thread_1);
    return (bitcast<f32>((_S151)));
}

var<workgroup> g_il_best_0 : array<f32, i32(64)>;

fn il_run_0( b_29 : u32,  i_1 : u32)
{
    var _S152 : bool = il_struck_0(b_29);
    var j_1 : u32;
    if(_S152)
    {
        j_1 = u32(1);
    }
    else
    {
        j_1 = u32(0);
    }
    var _S153 : u32 = group_uniform_0(j_1, i_1);
    if(_S153 == u32(0))
    {
        return;
    }
    var _S154 : f32 = 0.00100000004749745f + 0.01999999955296516f * il_striker_0(b_29).radius_0;
    var _S155 : u32 = a_il_0(b_29);
    var _S156 : u32 = atomicLoad(&(atom_0[_S155 + u32(4)]));
    var _S157 : u32 = atomicLoad(&(atom_0[_S155 + u32(1)]));
    var _S158 : u32 = atomicLoad(&(atom_0[_S155]));
    var _S159 : f32 = group_uniform_float_0((bitcast<f32>((_S158))), i_1);
    var end_0 : f32 = _S159;
    loop
    {
        var best_0 : f32 = end_0;
        j_1 = i_1;
        loop
        {
            if(j_1 < _S157)
            {
            }
            else
            {
                break;
            }
            var at_8 : u32 = w_il_tube_0() + u32(2) * (_S156 + j_1);
            if((bitcast<f32>((work_0[at_8]))) <= (end_0 + _S154))
            {
                best_0 = max(best_0, (bitcast<f32>((work_0[at_8 + u32(1)]))));
            }
            j_1 = j_1 + u32(64);
        }
        g_il_best_0[i_1] = best_0;
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
            if(i_1 < step_1)
            {
                g_il_best_0[i_1] = max(g_il_best_0[i_1], g_il_best_0[i_1 + step_1]);
            }
            workgroupBarrier();
            step_1 = (step_1 >> (u32(1)));
        }
        var next_0 : f32 = group_uniform_float_0(g_il_best_0[i32(0)], i_1);
        workgroupBarrier();
        if(!(next_0 > end_0))
        {
            break;
        }
        end_0 = next_0;
    }
    if(i_1 == u32(0))
    {
        atomicStore(&(atom_0[_S155]), (bitcast<u32>((end_0))));
    }
    return;
}

fn il_tube_collect_0( k_16 : u32)
{
    var b_30 : u32;
    var t_4 : vec2<f32>;
    var _S160 : bool = il_tube_0(k_16, &(b_30), &(t_4));
    if(!_S160)
    {
        return;
    }
    var _S161 : u32 = w_il_tube_0();
    var _S162 : u32 = atomicLoad(&(atom_0[a_il_0(b_30) + u32(4)]));
    var _S163 : u32 = atomicAdd(&(atom_0[a_il_0(b_30) + u32(5)]), u32(1));
    var at_9 : u32 = _S161 + u32(2) * (_S162 + _S163);
    work_0[at_9] = (bitcast<u32>((t_4.x)));
    work_0[at_9 + u32(1)] = (bitcast<u32>((t_4.y)));
    return;
}

fn il_tube_base_0( b_31 : u32)
{
    var _S164 : bool;
    if(b_31 < (params_0.body_count_0))
    {
        var _S165 : bool = il_struck_0(b_31);
        _S164 = _S165;
    }
    else
    {
        _S164 = false;
    }
    if(_S164)
    {
        var _S166 : u32 = a_il_0(b_31);
        var _S167 : u32 = atomicLoad(&(atom_0[_S166 + u32(1)]));
        var _S168 : u32 = atomicAdd(&(atom_0[a_il_cursor_0()]), _S167);
        atomicStore(&(atom_0[_S166 + u32(4)]), _S168);
    }
    return;
}

fn il_tube_count_0( k_17 : u32)
{
    var b_32 : u32;
    var t_5 : vec2<f32>;
    var _S169 : bool = il_tube_0(k_17, &(b_32), &(t_5));
    if(_S169)
    {
        var _S170 : u32 = atomicAdd(&(atom_0[a_il_0(b_32) + u32(1)]), u32(1));
    }
    return;
}

fn il_begin_0( b_33 : u32)
{
    var _S171 : bool;
    if(b_33 >= (params_0.body_count_0))
    {
        _S171 = true;
    }
    else
    {
        var _S172 : bool = il_struck_0(b_33);
        _S171 = !_S172;
    }
    if(_S171)
    {
        return;
    }
    var contact_12 : vec3<f32>;
    var d_14 : vec3<f32>;
    var speed_10 : f32;
    il_path_2(b_33, &(contact_12), &(d_14), &(speed_10));
    var _S173 : u32 = il_entry_0(b_33);
    var _S174 : u32 = a_il_0(b_33);
    atomicStore(&(atom_0[_S174]), (bitcast<u32>((max(il_interval_0(_S173, contact_12, d_14, il_striker_0(b_33).radius_0).y, 0.0f)))));
    var i_2 : u32 = u32(1);
    loop
    {
        if(i_2 < u32(6))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[_S174 + i_2]), u32(0));
        i_2 = i_2 + u32(1);
    }
    return;
}

fn il_breaks_1( i_3 : u32)
{
    var cut_1 : bool;
    if(i_3 >= (params_0.bond_count_0))
    {
        cut_1 = true;
    }
    else
    {
        cut_1 = work_0[w_alive_0() + i_3] == u32(0);
    }
    if(cut_1)
    {
        return;
    }
    var bond_1 : vec4<u32> = bonds_0[i_3];
    var _S175 : u32 = bond_1.x;
    var b_34 : u32 = body_of_0(_S175);
    if(!il_struck_1(b_34))
    {
        return;
    }
    var _S176 : u32 = lp_body_0(b_34);
    var s0_2 : vec4<f32> = lp_0[_S176];
    var s1_2 : vec4<f32> = lp_0[_S176 + u32(1)];
    var outcome_5 : u32 = (bitcast<u32>((lp_0[_S176].w)));
    if(((outcome_5 & (u32(14)))) == u32(0))
    {
        return;
    }
    var contact_13 : vec3<f32>;
    var d_15 : vec3<f32>;
    var speed_11 : f32;
    il_path_3(b_34, &(contact_13), &(d_15), &(speed_11));
    var entry_5 : u32 = il_entry_1(b_34);
    var _S177 : f32 = max(il_striker_1(b_34).radius_0, 0.00100000004749745f);
    var h_4 : f32 = s0_2.x;
    var pa_1 : IlPlace_0 = il_place_0(_S175, entry_5, contact_13, d_15, _S177, h_4);
    var pb_1 : IlPlace_0 = il_place_0(bond_1.y, entry_5, contact_13, d_15, _S177, h_4);
    if((il_plug_0(outcome_5, _S177, h_4, pa_1)) != (il_plug_0(outcome_5, _S177, h_4, pb_1)))
    {
        cut_1 = true;
    }
    else
    {
        var _S178 : f32 = s1_2.y;
        cut_1 = (il_scab_0(outcome_5, _S177, h_4, _S178, pa_1)) != (il_scab_0(outcome_5, _S177, h_4, _S178, pb_1));
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        cut_1 = il_crater_0(outcome_5, _S177, s1_2.x, s1_2.z, pa_1);
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        cut_1 = il_crater_0(outcome_5, _S177, s1_2.x, s1_2.z, pb_1);
    }
    if(cut_1)
    {
        work_0[w_alive_0() + i_3] = u32(0);
        var _S179 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_34]), u32(1));
        var _S180 : u32 = a_counters_0();
        var _S181 : u32 = atomicAdd(&(atom_0[_S180]), u32(1));
        var _S182 : u32 = atomicAdd(&(atom_0[_S180 + u32(1)]), u32(1));
    }
    return;
}

fn il_carry_1( k_18 : u32)
{
    if(k_18 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S183 : u32 = w_carry_0() + k_18;
    var c_4 : u32 = work_0[_S183];
    if(work_0[_S183] == u32(0))
    {
        return;
    }
    work_0[_S183] = u32(0);
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
    var _S184 : u32 = u32(7) * sk_3.body_0;
    var v_shot_1 : vec3<f32> = bodies_0[_S184 + u32(3)].xyz;
    var closing_1 : f32 = dot(v_shot_1 - v_wall_1, dr_1);
    var v_r_1 : f32 = sk_3.mass_0 * closing_1 / (sk_3.mass_0 + plug_mass_1);
    var _S185 : vec3<f32> = v_shot_1 + dr_1 * vec3<f32>((v_r_1 - closing_1));
    var _S186 : vec3<f32> = v_wall_1 + dr_1 * vec3<f32>(v_r_1);
    var _S187 : f32 = il_t_in_1(b0_1);
    var _S188 : vec3<f32> = bodies_0[_S184].xyz + dr_1 * vec3<f32>(((speed_12 - v_r_1) * clamp(_S187, 0.0f, params_0.dt_0)));
    var _S189 : u32 = body_of_0(k_18);
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
        bodies_0[s_at_2].x = _S188.x;
        bodies_0[s_at_2].y = _S188.y;
        bodies_0[s_at_2].z = _S188.z;
        bodies_0[s_at_2 + u32(3)].x = _S185.x;
        bodies_0[s_at_2 + u32(3)].y = _S185.y;
        bodies_0[s_at_2 + u32(3)].z = _S185.z;
        bodies_0[u32(7) * (_S189 + region_3 * params_0.max_bodies_0) + u32(3)].x = _S186.x;
        bodies_0[u32(7) * (_S189 + region_3 * params_0.max_bodies_0) + u32(3)].y = _S186.y;
        bodies_0[u32(7) * (_S189 + region_3 * params_0.max_bodies_0) + u32(3)].z = _S186.z;
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
    var _S190 : u32 = lp_body_0(b_35);
    var s0_3 : vec4<f32> = lp_0[_S190];
    var s1_3 : vec4<f32> = lp_0[_S190 + u32(1)];
    var outcome_6 : u32 = (bitcast<u32>((lp_0[_S190].w)));
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
    var _S191 : f32 = max(sk_4.radius_0, 0.00100000004749745f);
    var _S192 : f32 = s0_3.x;
    var p_11 : IlPlace_0 = il_place_0(k_19, entry_6, contact_15, d_17, _S191, _S192);
    var plug_2 : bool = il_plug_0(outcome_6, _S191, _S192, p_11);
    if((sk_4.launch_0) != u32(0))
    {
        var t_6 : vec2<f32> = il_interval_0(k_19, contact_15 - d_17 * vec3<f32>(sk_4.bound_0), d_17, sk_4.bound_0);
        var _S193 : f32 = t_6.x;
        var _S194 : f32 = t_6.y;
        var path_1 : bool;
        if(_S193 <= _S194)
        {
            path_1 = _S193 <= (_S192 + 2.0f * sk_4.bound_0);
        }
        else
        {
            path_1 = false;
        }
        if(path_1)
        {
            path_1 = _S194 >= 0.0f;
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
            path_1 = il_scab_0(outcome_6, _S191, _S192, s1_3.y, p_11);
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            path_1 = il_crater_0(outcome_6, _S191, s1_3.x, s1_3.z, p_11);
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

fn il_strike_1( b_36 : u32)
{
    var damage_1 : bool;
    if(b_36 >= (params_0.body_count_0))
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = !il_struck_1(b_36);
    }
    if(damage_1)
    {
        return;
    }
    var t_in_6 : f32 = il_t_in_1(b_36);
    var entry_7 : u32 = il_entry_1(b_36);
    var contact_16 : vec3<f32>;
    var d_18 : vec3<f32>;
    var speed_14 : f32;
    il_path_3(b_36, &(contact_16), &(d_18), &(speed_14));
    var sk_5 : IlStriker_0 = il_striker_1(b_36);
    var _S195 : f32 = max(sk_5.radius_0, 0.00100000004749745f);
    if((sk_5.launch_0) != u32(0))
    {
        damage_1 = work_0[w_marks_0() + entry_7] == (sk_5.launch_0);
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        var _S196 : u32 = lp_body_0(b_36);
        var _S197 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S196] = _S197;
        lp_0[_S196 + u32(1)] = _S197;
        return;
    }
    var _S198 : u32 = atomicLoad(&(atom_0[a_il_0(b_36)]));
    var _S199 : f32 = max((bitcast<f32>((_S198))) - _S195, 9.99999997475242708e-07f);
    if(!(t_in_6 > 0.0f))
    {
        var _S200 : u32 = lp_body_0(b_36);
        lp_0[_S200] = vec4<f32>(_S199, 0.0f, 0.0f, 0.0f);
        lp_0[_S200 + u32(1)] = vec4<f32>(0.0f);
        return;
    }
    var o_3 : u32 = params_0.ext_chunks_0 + u32(4) * entry_7;
    var mat0_1 : vec4<f32> = ext_0[o_3 + u32(1)];
    var dm_1 : JointMaterial_0;
    var _S201 : vec4<f32> = vec4<f32>(0.0f);
    dm_1.strength_0 = _S201;
    dm_1.energy_0 = _S201;
    dm_1.dif_0 = ext_0[o_3 + u32(2)];
    var mat3_1 : vec4<f32> = ext_0[o_3 + u32(3)];
    var _S202 : f32 = mat3_1.x;
    dm_1.misc_0 = vec4<f32>(_S202, 0.0f, 0.0f, 0.0f);
    dm_1.kind_flags_0 = vec4<u32>(u32(0));
    var nu_1 : f32 = mat3_1.y;
    var e_target_1 : f32 = mat3_1.z / (1.0f - nu_1 * nu_1);
    var e_star_1 : f32 = 1.0f / (1.0f / sk_5.modulus_0 + 1.0f / e_target_1);
    var _S203 : f32 = max(speed_14, 0.00100000004749745f);
    var _S204 : f32 = sqrt(_S195);
    var h0_1 : f32 = pow(15.0f * sk_5.m_eff_0 * _S203 * _S203 / (16.0f * e_star_1 * _S204), 0.40000000596046448f);
    var t_c_1 : f32 = 2.94000005722045898f * h0_1 / _S203;
    var force_1 : f32 = 1.33333337306976318f * e_star_1 * _S204 * pow(h0_1, 1.5f);
    var a_6 : f32 = sqrt(_S195 * h0_1);
    var p0_1 : f32 = 3.0f * force_1 / (6.28318548202514648f * a_6 * a_6);
    var ring_1 : f32 = (1.0f - 2.0f * nu_1) * p0_1 / 3.0f;
    var k_20 : f32;
    if(_S202 > 0.0f)
    {
        k_20 = dif_factor_0(dm_1, _S203 / _S199);
    }
    else
    {
        k_20 = 1.0f;
    }
    var ft_1 : f32 = k_20 * mat0_1.y;
    var fc_1 : f32 = k_20 * mat0_1.z;
    var tau_1 : f32 = k_20 * mat0_1.w;
    if(p0_1 > fc_1)
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = ring_1 > ft_1;
    }
    var _S205 : f32 = fc_1 * (3.14159274101257324f * _S195 * _S195);
    var _S206 : f32 = min(_S205 * _S199, 1.57079637050628662f * (2.0f * _S195 + _S199) * _S199 * _S199 * tau_1);
    var _S207 : f32 = 2.0f * _S206;
    var v_bl_1 : f32 = sqrt(_S207 / sk_5.m_eff_0);
    var perforate_1 : bool;
    if(damage_1)
    {
        perforate_1 = _S203 >= v_bl_1;
    }
    else
    {
        perforate_1 = false;
    }
    var r2_1 : f32 = _S195 + _S199;
    var _S208 : f32 = mat0_1.x;
    var plug_3 : f32 = _S208 * 3.14159274101257324f * _S199 / 3.0f * (_S195 * _S195 + _S195 * r2_1 + r2_1 * r2_1);
    var v_res_1 : f32;
    if(perforate_1)
    {
        v_res_1 = sk_5.mass_0 / (sk_5.mass_0 + plug_3) * sqrt(max(_S203 * _S203 - v_bl_1 * v_bl_1, 0.0f));
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
        depth_2 = min(0.5f * sk_5.m_eff_0 * _S203 * _S203 / _S205, _S199);
    }
    else
    {
        depth_2 = 0.0f;
    }
    var crater_1 : bool = depth_2 > a_6;
    var c_5 : f32 = sqrt(e_target_1 / _S208);
    var z_t_1 : f32 = _S208 * c_5;
    var rho_p_1 : f32 = sk_5.mass_0 / (4.18879032135009766f * _S195 * _S195 * _S195);
    var z_p_1 : f32 = rho_p_1 * sqrt(sk_5.modulus_0 / rho_p_1);
    var mean_pressure_1 : f32 = force_1 / (3.14159274101257324f * a_6 * a_6);
    var _S209 : f32 = min(min(z_t_1 * _S203 * z_p_1 / (z_p_1 + z_t_1), mean_pressure_1), fc_1);
    var source_radius_1 : f32;
    if(mean_pressure_1 > fc_1)
    {
        source_radius_1 = max(a_6, _S195);
    }
    else
    {
        source_radius_1 = a_6;
    }
    var pulse_1 : f32 = c_5 * t_c_1;
    var back_1 : f32 = _S209 * min(source_radius_1 / _S199, 1.0f) * min(2.0f * _S199 / pulse_1, 1.0f);
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
        scab_2 = 0.5f * min(pulse_1, _S199);
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
    var _S210 : u32;
    if(crater_1)
    {
        _S210 = u32(2);
    }
    else
    {
        _S210 = u32(0);
    }
    var _S211 : u32 = (region_4 | (_S210));
    if(spall_1)
    {
        region_4 = u32(4);
    }
    else
    {
        region_4 = u32(0);
    }
    var _S212 : u32 = (_S211 | (region_4));
    if(perforate_1)
    {
        region_4 = u32(8);
    }
    else
    {
        region_4 = u32(0);
    }
    var _S213 : u32 = lp_body_0(b_36);
    lp_0[_S213] = vec4<f32>(_S199, v_bl_1, v_res_1, (bitcast<f32>(((_S212 | (region_4))))));
    if(crater_1)
    {
        k_20 = depth_2;
    }
    else
    {
        k_20 = 0.0f;
    }
    lp_0[_S213 + u32(1)] = vec4<f32>(k_20, scab_2, a_6, _S206);
    if(perforate_1)
    {
        var _S214 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(14)]), u32(1));
    }
    else
    {
        if(spall_1)
        {
            var _S215 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(13)]), u32(1));
        }
        else
        {
            if(crater_1)
            {
                var _S216 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(12)]), u32(1));
            }
            else
            {
                var _S217 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(11)]), u32(1));
            }
        }
    }
    if(perforate_1)
    {
        if((params_0.fracture_work_0) != u32(0))
        {
            il_impulse_0(b_36, sk_5.body_0, sk_5.mass_0, contact_16, d_18, sk_5.m_eff_0 * (_S203 - sqrt(max(_S203 * _S203 - _S207 / sk_5.m_eff_0, 0.0f))));
        }
    }
    else
    {
        if(damage_1)
        {
            il_impulse_0(b_36, sk_5.body_0, sk_5.mass_0, contact_16, d_18, sk_5.m_eff_0 * _S203);
            var _S218 : vec3<f32> = to_rubble_0(d_18) * vec3<f32>((_S203 * clamp(t_in_6, 0.0f, params_0.dt_0)));
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
                var _S219 : u32 = u32(7) * (sk_5.body_0 + region_4 * params_0.max_bodies_0);
                var _S220 : vec3<f32> = bodies_0[_S219].xyz + _S218;
                bodies_0[_S219].x = _S220.x;
                bodies_0[_S219].y = _S220.y;
                bodies_0[_S219].z = _S220.z;
                region_4 = region_4 + u32(1);
            }
        }
    }
    return;
}

fn il_run_1( b_37 : u32,  i_4 : u32)
{
    var j_2 : u32;
    if(il_struck_1(b_37))
    {
        j_2 = u32(1);
    }
    else
    {
        j_2 = u32(0);
    }
    var _S221 : u32 = group_uniform_0(j_2, i_4);
    if(_S221 == u32(0))
    {
        return;
    }
    var _S222 : f32 = 0.00100000004749745f + 0.01999999955296516f * il_striker_1(b_37).radius_0;
    var _S223 : u32 = a_il_0(b_37);
    var _S224 : u32 = atomicLoad(&(atom_0[_S223 + u32(4)]));
    var _S225 : u32 = atomicLoad(&(atom_0[_S223 + u32(1)]));
    var _S226 : u32 = atomicLoad(&(atom_0[_S223]));
    var _S227 : f32 = group_uniform_float_0((bitcast<f32>((_S226))), i_4);
    var end_1 : f32 = _S227;
    loop
    {
        var best_1 : f32 = end_1;
        j_2 = i_4;
        loop
        {
            if(j_2 < _S225)
            {
            }
            else
            {
                break;
            }
            var at_10 : u32 = w_il_tube_0() + u32(2) * (_S224 + j_2);
            if((bitcast<f32>((work_0[at_10]))) <= (end_1 + _S222))
            {
                best_1 = max(best_1, (bitcast<f32>((work_0[at_10 + u32(1)]))));
            }
            j_2 = j_2 + u32(64);
        }
        g_il_best_0[i_4] = best_1;
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
            if(i_4 < step_2)
            {
                g_il_best_0[i_4] = max(g_il_best_0[i_4], g_il_best_0[i_4 + step_2]);
            }
            workgroupBarrier();
            step_2 = (step_2 >> (u32(1)));
        }
        var next_1 : f32 = group_uniform_float_0(g_il_best_0[i32(0)], i_4);
        workgroupBarrier();
        if(!(next_1 > end_1))
        {
            break;
        }
        end_1 = next_1;
    }
    if(i_4 == u32(0))
    {
        atomicStore(&(atom_0[_S223]), (bitcast<u32>((end_1))));
    }
    return;
}

fn il_tube_collect_1( k_21 : u32)
{
    var b_38 : u32;
    var t_7 : vec2<f32>;
    var _S228 : bool = il_tube_1(k_21, &(b_38), &(t_7));
    if(!_S228)
    {
        return;
    }
    var _S229 : u32 = w_il_tube_0();
    var _S230 : u32 = atomicLoad(&(atom_0[a_il_0(b_38) + u32(4)]));
    var _S231 : u32 = atomicAdd(&(atom_0[a_il_0(b_38) + u32(5)]), u32(1));
    var at_11 : u32 = _S229 + u32(2) * (_S230 + _S231);
    work_0[at_11] = (bitcast<u32>((t_7.x)));
    work_0[at_11 + u32(1)] = (bitcast<u32>((t_7.y)));
    return;
}

fn il_tube_base_1( b_39 : u32)
{
    var _S232 : bool;
    if(b_39 < (params_0.body_count_0))
    {
        _S232 = il_struck_1(b_39);
    }
    else
    {
        _S232 = false;
    }
    if(_S232)
    {
        var _S233 : u32 = a_il_0(b_39);
        var _S234 : u32 = atomicLoad(&(atom_0[_S233 + u32(1)]));
        var _S235 : u32 = atomicAdd(&(atom_0[a_il_cursor_0()]), _S234);
        atomicStore(&(atom_0[_S233 + u32(4)]), _S235);
    }
    return;
}

fn il_tube_count_1( k_22 : u32)
{
    var b_40 : u32;
    var t_8 : vec2<f32>;
    var _S236 : bool = il_tube_1(k_22, &(b_40), &(t_8));
    if(_S236)
    {
        var _S237 : u32 = atomicAdd(&(atom_0[a_il_0(b_40) + u32(1)]), u32(1));
    }
    return;
}

fn struck_1( b_41 : u32,  shot_3 : ptr<function, u32>) -> bool
{
    var k_23 : u32 = u32(frames_0[strike_at_0(b_41)].w);
    (*shot_3) = k_23 - u32(1);
    return k_23 != u32(0);
}

fn il_begin_1( b_42 : u32)
{
    var _S238 : bool;
    if(b_42 >= (params_0.body_count_0))
    {
        _S238 = true;
    }
    else
    {
        _S238 = !il_struck_1(b_42);
    }
    if(_S238)
    {
        return;
    }
    var shot_4 : u32;
    var _S239 : bool = struck_1(b_42, &(shot_4));
    var _S240 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_42 + u32(6)]));
    var _S241 : u32 = atomicLoad(&(atom_0[a_first_0(shot_4)]));
    if(_S240 != _S241)
    {
        frames_0[strike_at_0(b_42)][i32(3)] = 0.0f;
        return;
    }
    var contact_17 : vec3<f32>;
    var d_19 : vec3<f32>;
    var speed_15 : f32;
    il_path_3(b_42, &(contact_17), &(d_19), &(speed_15));
    var _S242 : u32 = il_entry_1(b_42);
    var _S243 : u32 = a_il_0(b_42);
    atomicStore(&(atom_0[_S243]), (bitcast<u32>((max(il_interval_0(_S242, contact_17, d_19, il_striker_1(b_42).radius_0).y, 0.0f)))));
    var i_5 : u32 = u32(1);
    loop
    {
        if(i_5 < u32(6))
        {
        }
        else
        {
            break;
        }
        atomicStore(&(atom_0[_S243 + i_5]), u32(0));
        i_5 = i_5 + u32(1);
    }
    return;
}

fn structure_pose_2( slot_5 : u32,  b_43 : u32,  q_13 : ptr<function, Quat_0>,  origin_5 : ptr<function, vec3<f32>>)
{
    var _S244 : u32 = u32(2) * slot_5;
    var com_2 : vec3<f32> = frames_0[_S244].xyz;
    var _S245 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_43 + u32(1)])), conj_0(quat_of_0(frames_0[_S244 + u32(1)]))));
    (*q_13) = _S245;
    (*origin_5) = from_rubble_0(bodies_0[b_43].xyz) - rotate_0(_S245, com_2);
    return;
}

fn structure_pose_3( slot_6 : u32,  b_44 : u32,  q_14 : ptr<function, Quat_0>,  origin_6 : ptr<function, vec3<f32>>)
{
    var _S246 : u32 = u32(2) * slot_6;
    var com_3 : vec3<f32> = frames_0[_S246].xyz;
    var _S247 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_44 + u32(1)])), conj_0(quat_of_0(frames_0[_S246 + u32(1)]))));
    (*q_14) = _S247;
    (*origin_6) = from_rubble_0(bodies_0[b_44].xyz) - rotate_0(_S247, com_3);
    return;
}

fn structure_pose_4( slot_7 : u32,  b_45 : u32,  q_15 : ptr<function, Quat_0>,  origin_7 : ptr<function, vec3<f32>>)
{
    var _S248 : u32 = u32(2) * slot_7;
    var com_4 : vec3<f32> = frames_0[_S248].xyz;
    var _S249 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_45 + u32(1)])), conj_0(quat_of_0(frames_0[_S248 + u32(1)]))));
    (*q_15) = _S249;
    (*origin_7) = from_rubble_0(bodies_0[b_45].xyz) - rotate_0(_S249, com_4);
    return;
}

fn structure_pose_5( slot_8 : u32,  b_46 : u32,  q_16 : ptr<function, Quat_0>,  origin_8 : ptr<function, vec3<f32>>)
{
    var _S250 : u32 = u32(2) * slot_8;
    var com_5 : vec3<f32> = frames_0[_S250].xyz;
    var _S251 : Quat_0 = normalized_0(quat_mul_0(rot_from_rubble_0(quat_of_0(bodies_0[b_46 + u32(1)])), conj_0(quat_of_0(frames_0[_S250 + u32(1)]))));
    (*q_16) = _S251;
    (*origin_8) = from_rubble_0(bodies_0[b_46].xyz) - rotate_0(_S251, com_5);
    return;
}

fn l_pool_0( half_1 : u32) -> u32
{
    return params_0.chunk_count_0 + u32(3) * params_0.pool_cap_0 * half_1;
}

fn pulse_mean_0( start_0 : f32,  plateau_0 : f32,  plateau_time_0 : f32,  peak_0 : f32,  duration_0 : f32,  dt_1 : f32) -> f32
{
    var _S252 : f32 = start_0 + plateau_time_0;
    var sum_0 : f32 = plateau_0 * max(min(_S252, dt_1) - max(start_0, 0.0f), 0.0f);
    var _S253 : f32 = max(0.0f, _S252);
    var _S254 : f32 = min(dt_1, _S252 + duration_0);
    var _S255 : bool;
    if(_S254 > _S253)
    {
        _S255 = duration_0 > 0.0f;
    }
    else
    {
        _S255 = false;
    }
    var sum_1 : f32;
    if(_S255)
    {
        var w_2 : f32 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S253 - _S252)) - cos(w_2 * (_S254 - _S252)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

fn hertz_duration_0( m_0 : f32,  r_17 : f32,  e_2 : f32,  v_9 : f32) -> f32
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_17 * e_2 * e_2 * max(abs(v_9), 0.00100000004749745f)), 0.20000000298023224f);
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
    var p_12 : u32;
    var kept_0 : u32;
    var q3_0 : u32;
    var overflow_0 : bool;
    var overflow_1 : bool;
    var f_total_1 : vec3<f32>;
    var t_total_1 : vec3<f32>;
    var k_24 : u32 = id_4.x;
    if(k_24 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_9 : u32 = body_of_0(k_24);
    var b_47 : u32 = u32(7) * slot_9;
    var q_17 : Quat_0;
    var origin_9 : vec3<f32>;
    structure_pose_5(slot_9, b_47, &(q_17), &(origin_9));
    var _S256 : u32 = u32(4) * k_24;
    var c0_0 : vec4<f32> = chunks_0[_S256];
    var centre_0 : vec3<f32> = origin_9 + rotate_0(q_17, c0_0.xyz);
    var inv_mass_2 : f32 = bodies_0[b_47].w;
    var m_body_0 : f32;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(u32(2139095040));
    }
    var from_0 : u32 = params_0.parity_0;
    var to_0 : u32 = u32(1) - params_0.parity_0;
    var _S257 : u32 = w_pulses_0() + _S256;
    var _S258 : u32 = _S257 + u32(2) * params_0.parity_0;
    var off_0 : u32 = work_0[_S258];
    var carried_0 : u32 = work_0[_S258 + u32(1)];
    var room_0 : u32 = work_0[_S258 + u32(1)] + sort_0[params_0.chunk_count_0 + u32(1) + k_24];
    var base_0 : u32;
    var room_1 : u32;
    if(room_0 != u32(0))
    {
        var _S259 : u32 = a_counters_0();
        var base_1 : u32 = atomicAdd(&(atom_0[_S259 + u32(5) + to_0]), room_0);
        if((base_1 + room_0) > (params_0.pool_cap_0))
        {
            var _S260 : u32 = atomicOr(&(atom_0[_S259 + u32(3)]), u32(8));
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
        var _S261 : u32 = base_0;
        base_0 = base_1;
        room_1 = _S261;
    }
    else
    {
        base_0 = u32(0);
        room_1 = room_0;
    }
    var _S262 : u32 = params_0.chunk_count_0;
    var _S263 : f32 = c0_0.w;
    var _S264 : vec4<f32> = chunks_0[_S256 + u32(3)];
    var _S265 : f32 = params_0.band_0;
    var _S266 : f32 = params_0.dt_0;
    var _S267 : bool = (params_0.impact_peak_0) != u32(0);
    var _S268 : u32 = l_pool_0(from_0) + u32(3) * off_0;
    var _S269 : u32 = l_pool_0(to_0) + u32(3) * base_0;
    loop
    {
        var _S270 : vec3<f32> = vec3<f32>(0.0f);
        var _S271 : u32 = sort_0[k_24];
        var _S272 : u32 = sort_0[_S262 + u32(1) + k_24];
        var first_0 : bool = true;
        var last_0 : u32 = u32(0);
        var step_3 : u32 = u32(0);
        f_total_0 = _S270;
        t_total_0 = _S270;
        loop
        {
            if(step_3 < _S272)
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
                if(q3_0 < _S272)
                {
                }
                else
                {
                    break;
                }
                var key_1 : u32 = manifolds_0[_S271 + q3_0].ids_0.y + u32(1);
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
            var r_18 : Record_std430_0 = manifolds_0[_S271 + p_12];
            var _S273 : vec3<f32> = r_18.impulse_0.xyz;
            var j_3 : f32 = length(_S273);
            if(j_3 == 0.0f)
            {
                var _S274 : u32 = step_3 + u32(1);
                first_0 = false;
                last_0 = kept_0;
                step_3 = _S274;
                continue;
            }
            var m_other_0 : f32 = r_18.other_0.x;
            if(isinf_0(m_body_0))
            {
                if(isinf_0(m_other_0))
                {
                    m_1 = _S263;
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
            var _S275 : f32 = min(j_3, 2.0f * m_1 * max(r_18.impulse_0.w - _S265, 0.0f));
            if(_S275 < j_3)
            {
                var f_1 : vec3<f32> = _S273 / vec3<f32>(j_3) * vec3<f32>(((j_3 - _S275) / _S266));
                var _S276 : vec3<f32> = t_total_0 + cross(r_18.point_0.xyz - centre_0, f_1);
                f_total_1 = f_total_0 + f_1;
                t_total_1 = _S276;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            var _S274 : u32 = step_3 + u32(1);
            first_0 = false;
            last_0 = kept_0;
            step_3 = _S274;
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
            var _S277 : u32 = _S268 + u32(3) * p_12;
            var a_7 : vec4<f32> = loads_0[_S277];
            var b_48 : vec4<f32> = loads_0[_S277 + u32(1)];
            var c_6 : vec4<f32> = loads_0[_S277 + u32(2)];
            var _S278 : f32 = loads_0[_S277 + u32(1)].w;
            var mean_0 : f32 = pulse_mean_0(loads_0[_S277].x, loads_0[_S277].y, loads_0[_S277].z, loads_0[_S277].w, _S278, _S266);
            if(mean_0 != 0.0f)
            {
                var f_2 : vec3<f32> = c_6.xyz * vec3<f32>(mean_0);
                var _S279 : vec3<f32> = t_total_0 + cross(b_48.xyz - centre_0, f_2);
                f_total_0 = f_total_0 + f_2;
                t_total_0 = _S279;
            }
            if((a_7.x + a_7.z + _S278) > _S266)
            {
                a_7[i32(0)] = a_7[i32(0)] - _S266;
                var _S280 : u32 = _S269 + u32(3) * kept_0;
                loads_0[_S280] = a_7;
                loads_0[_S280 + u32(1)] = b_48;
                loads_0[_S280 + u32(2)] = c_6;
                kept_0 = kept_0 + u32(1);
            }
            p_12 = p_12 + u32(1);
        }
        q3_0 = u32(0);
        overflow_0 = false;
        loop
        {
            if(q3_0 < _S272)
            {
            }
            else
            {
                break;
            }
            var r_19 : Record_std430_0 = manifolds_0[_S271 + q3_0];
            var _S281 : vec3<f32> = r_19.impulse_0.xyz;
            var j_4 : f32 = length(_S281);
            if(j_4 == 0.0f)
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
                    m_1 = _S263;
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
            var _S282 : f32 = r_19.impulse_0.w;
            var _S283 : f32 = min(j_4, 2.0f * m_1 * max(_S282 - _S265, 0.0f));
            if(_S283 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + u32(1);
                overflow_0 = overflow_1;
                continue;
            }
            var _S284 : f32 = _S264.y;
            var duration_1 : f32 = hertz_duration_0(m_1, min(r_19.other_0.z, _S264.z), 1.0f / ((1.0f - _S284 * _S284) / _S264.x + 1.0f / r_19.other_0.y), _S282);
            var peak_1 : f32 = 3.14159274101257324f * _S283 / (2.0f * duration_1);
            var dir_0 : vec3<f32> = _S281 / vec3<f32>(j_4);
            var mean_1 : f32;
            if(_S267)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S266);
            }
            if(mean_1 != 0.0f)
            {
                var f_3 : vec3<f32> = dir_0 * vec3<f32>(mean_1);
                var _S285 : vec3<f32> = t_total_0 + cross(r_19.point_0.xyz - centre_0, f_3);
                f_total_1 = f_total_0 + f_3;
                t_total_1 = _S285;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            var kept_1 : u32;
            if(duration_1 > _S266)
            {
                if(kept_0 < room_1)
                {
                    var _S286 : u32 = _S269 + u32(3) * kept_0;
                    loads_0[_S286] = vec4<f32>(- _S266, 0.0f, 0.0f, peak_1);
                    loads_0[_S286 + u32(1)] = vec4<f32>(r_19.point_0.xyz, duration_1);
                    loads_0[_S286 + u32(2)] = vec4<f32>(dir_0, 0.0f);
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
        var _S287 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(8));
    }
    var _S288 : u32 = _S257 + u32(2) * to_0;
    work_0[_S288] = base_0;
    work_0[_S288 + u32(1)] = kept_0;
    loads_0[k_24] = vec4<f32>(length(f_total_0), 0.0f, 0.0f, 0.0f);
    if((params_0.load_model_0) == u32(1))
    {
        var _S289 : u32 = l_vec_0() + u32(2) * k_24;
        loads_0[_S289] = vec4<f32>(f_total_0, 0.0f);
        loads_0[_S289 + u32(1)] = vec4<f32>(t_total_0, 0.0f);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_breaks(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var i_6 : u32 = id_5.x;
    var _S290 : bool;
    if(i_6 >= (params_0.bond_count_0))
    {
        _S290 = true;
    }
    else
    {
        _S290 = work_0[w_alive_0() + i_6] == u32(0);
    }
    if(_S290)
    {
        return;
    }
    var bond_2 : vec4<u32> = bonds_0[i_6];
    var _S291 : u32 = bond_2.x;
    if((max(loads_0[_S291].x, loads_0[bond_2.y].x)) > (bitcast<f32>((bond_2.z))))
    {
        work_0[w_alive_0() + i_6] = u32(0);
        var _S292 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S291)]), u32(1));
        var _S293 : u32 = a_counters_0();
        var _S294 : u32 = atomicAdd(&(atom_0[_S293]), u32(1));
        var _S295 : u32 = atomicAdd(&(atom_0[_S293 + u32(1)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_host_breaks(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var i_7 : u32 = id_6.x;
    var _S296 : u32 = w_host_breaks_0();
    if(i_7 >= work_0[_S296])
    {
        return;
    }
    var b_49 : u32 = work_0[_S296 + u32(1) + i_7];
    var _S297 : bool;
    if(work_0[_S296 + u32(1) + i_7] >= (params_0.bond_count_0))
    {
        _S297 = true;
    }
    else
    {
        _S297 = work_0[w_alive_0() + b_49] == u32(0);
    }
    if(_S297)
    {
        return;
    }
    work_0[w_alive_0() + b_49] = u32(0);
    var _S298 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(bonds_0[b_49].x)]), u32(1));
    var _S299 : u32 = a_counters_0();
    var _S300 : u32 = atomicAdd(&(atom_0[_S299]), u32(1));
    var _S301 : u32 = atomicAdd(&(atom_0[_S299 + u32(1)]), u32(1));
    return;
}

fn groups_of_0( n_3 : u32,  size_0 : u32) -> u32
{
    var _S302 : u32 = (n_3 + size_0 - u32(1)) / size_0;
    return _S302;
}

fn set_args_0( at_12 : u32,  x_4 : u32)
{
    var _S303 : u32 = u32(3) * at_12;
    args_0[_S303] = x_4;
    args_0[_S303 + u32(1)] = u32(1);
    args_0[_S303 + u32(2)] = u32(1);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_gate(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var _S304 : u32 = a_counters_0();
    var _S305 : u32 = atomicLoad(&(atom_0[_S304]));
    var run_0 : bool = _S305 != u32(0);
    atomicStore(&(atom_0[_S304]), u32(0));
    atomicStore(&(atom_0[_S304 + u32(5) + params_0.parity_0]), u32(0));
    if(run_0)
    {
        var _S306 : u32 = atomicAdd(&(atom_0[_S304 + u32(4)]), u32(1));
    }
    var _S307 : u32;
    if(run_0)
    {
        var _S308 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S307 = _S308;
    }
    else
    {
        _S307 = u32(0);
    }
    set_args_0(u32(0), _S307);
    if(run_0)
    {
        var _S309 : u32 = groups_of_0(params_0.bond_count_0, u32(64));
        _S307 = _S309;
    }
    else
    {
        _S307 = u32(0);
    }
    set_args_0(u32(1), _S307);
    if(run_0)
    {
        var _S310 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S307 = _S310;
    }
    else
    {
        _S307 = u32(0);
    }
    set_args_0(u32(2), _S307);
    if(run_0)
    {
        var _S311 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S307 = _S311;
    }
    else
    {
        _S307 = u32(0);
    }
    set_args_0(u32(3), _S307);
    if(run_0)
    {
        var _S312 : u32 = groups_of_0(params_0.body_count_0, u32(1024));
        _S307 = _S312;
    }
    else
    {
        _S307 = u32(0);
    }
    set_args_0(u32(4), _S307);
    var _S313 : i32;
    if(run_0)
    {
        _S313 = i32(1);
    }
    else
    {
        _S313 = i32(0);
    }
    set_args_0(u32(5), u32(_S313));
    return;
}

fn dirty_0( body_1 : u32) -> bool
{
    var _S314 : u32 = atomicLoad(&(atom_0[a_dirty_0() + body_1]));
    return _S314 != u32(0);
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
fn fr_uf_init(@builtin(global_invocation_id) id_8 : vec3<u32>)
{
    var c_7 : u32 = id_8.x;
    var _S315 : bool;
    if(c_7 >= (params_0.chunk_count_0))
    {
        _S315 = true;
    }
    else
    {
        var _S316 : bool = dirty_0(body_of_0(c_7));
        _S315 = !_S316;
    }
    if(_S315)
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
fn fr_body_init(@builtin(global_invocation_id) id_9 : vec3<u32>)
{
    var s_7 : u32 = id_9.x;
    var _S317 : bool;
    if(s_7 >= (params_0.body_count_0))
    {
        _S317 = true;
    }
    else
    {
        var _S318 : bool = dirty_0(s_7);
        _S317 = !_S318;
    }
    if(_S317)
    {
        return;
    }
    atomicStore(&(atom_0[a_best_count_0() + s_7]), u32(0));
    atomicStore(&(atom_0[a_best_root_0() + s_7]), u32(4294967295));
    return;
}

fn find_0( x_5 : u32) -> u32
{
    var _S319 : u32 = a_parent_0();
    var _S320 : u32 = atomicLoad(&(atom_0[_S319 + x_5]));
    var p_13 : u32 = _S320;
    var _S321 : u32 = x_5;
    var guard_0 : u32 = u32(0);
    loop
    {
        var _S322 : bool;
        if(guard_0 < u32(1048576))
        {
            _S322 = p_13 != _S321;
        }
        else
        {
            _S322 = false;
        }
        if(_S322)
        {
        }
        else
        {
            break;
        }
        var gp_0 : u32 = atomicLoad(&(atom_0[_S319 + p_13]));
        if(gp_0 != p_13)
        {
            var _S323 : u32 = atomicCompareExchangeWeak(&(atom_0[_S319 + _S321]), p_13, gp_0).old_value;
        }
        var _S324 : u32 = guard_0 + u32(1);
        _S321 = p_13;
        p_13 = gp_0;
        guard_0 = _S324;
    }
    return _S321;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_union(@builtin(global_invocation_id) id_10 : vec3<u32>)
{
    var i_8 : u32 = id_10.x;
    var _S325 : bool;
    if(i_8 >= (params_0.bond_count_0))
    {
        _S325 = true;
    }
    else
    {
        _S325 = work_0[w_alive_0() + i_8] == u32(0);
    }
    if(_S325)
    {
        return;
    }
    var bond_3 : vec4<u32> = bonds_0[i_8];
    var _S326 : u32 = bond_3.x;
    var _S327 : bool = dirty_0(body_of_0(_S326));
    if(!_S327)
    {
        return;
    }
    var _S328 : u32 = bond_3.y;
    var a_8 : u32 = _S326;
    var b_50 : u32 = _S328;
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
        var rb_0 : u32 = find_0(b_50);
        if(ra_0 == rb_0)
        {
            return;
        }
        var _S329 : u32 = max(ra_0, rb_0);
        var _S330 : u32 = min(ra_0, rb_0);
        var was_0 : u32 = atomicCompareExchangeWeak(&(atom_0[a_parent_0() + _S329]), _S329, _S330).old_value;
        if(was_0 == _S329)
        {
            return;
        }
        var _S331 : u32 = guard_1 + u32(1);
        a_8 = _S329;
        b_50 = _S330;
        guard_1 = _S331;
    }
    var _S332 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(4));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_uf_flatten(@builtin(global_invocation_id) id_11 : vec3<u32>)
{
    var c_8 : u32 = id_11.x;
    var _S333 : bool;
    if(c_8 >= (params_0.chunk_count_0))
    {
        _S333 = true;
    }
    else
    {
        var _S334 : bool = dirty_0(body_of_0(c_8));
        _S333 = !_S334;
    }
    if(_S333)
    {
        return;
    }
    var r_20 : u32 = find_0(c_8);
    work_0[w_label_0() + c_8] = r_20;
    var _S335 : u32 = atomicAdd(&(atom_0[a_count_0() + r_20]), u32(1));
    if(((info_0[c_8] & (u32(1)))) != u32(0))
    {
        var _S336 : u32 = atomicOr(&(atom_0[a_anchor_0() + r_20]), u32(1));
    }
    var _S337 : u32 = u32(4) * c_8;
    var c0_1 : vec4<f32> = chunks_0[_S337];
    var c1_0 : vec4<f32> = chunks_0[_S337 + u32(1)];
    var c2_0 : vec4<f32> = chunks_0[_S337 + u32(2)];
    var m_2 : f32 = c0_1.w;
    var d_20 : vec3<f32> = c0_1.xyz - chunks_0[u32(4) * r_20].xyz;
    var dd_0 : f32 = dot(d_20, d_20);
    var at_13 : u32 = a_sums_0() + u32(20) * r_20;
    add64_0(at_13, m_2, params_0.scale_m_0);
    var _S338 : f32 = d_20.x;
    var _S339 : f32 = m_2 * _S338;
    add64_0(at_13 + u32(2), _S339, params_0.scale_s_0);
    var _S340 : f32 = d_20.y;
    var _S341 : f32 = m_2 * _S340;
    add64_0(at_13 + u32(4), _S341, params_0.scale_s_0);
    var _S342 : f32 = d_20.z;
    add64_0(at_13 + u32(6), m_2 * _S342, params_0.scale_s_0);
    add64_0(at_13 + u32(8), c1_0.x + m_2 * (dd_0 - _S338 * _S338), params_0.scale_j_0);
    add64_0(at_13 + u32(10), c1_0.y + m_2 * (dd_0 - _S340 * _S340), params_0.scale_j_0);
    add64_0(at_13 + u32(12), c1_0.z + m_2 * (dd_0 - _S342 * _S342), params_0.scale_j_0);
    add64_0(at_13 + u32(14), c1_0.w - _S339 * _S340, params_0.scale_j_0);
    add64_0(at_13 + u32(16), c2_0.x - _S339 * _S342, params_0.scale_j_0);
    add64_0(at_13 + u32(18), c2_0.y - _S341 * _S342, params_0.scale_j_0);
    return;
}

fn is_root_0( c_9 : u32) -> bool
{
    var _S343 : bool;
    if(c_9 < (params_0.chunk_count_0))
    {
        var _S344 : bool = dirty_0(body_of_0(c_9));
        _S343 = _S344;
    }
    else
    {
        _S343 = false;
    }
    if(_S343)
    {
        _S343 = work_0[w_label_0() + c_9] == c_9;
    }
    else
    {
        _S343 = false;
    }
    return _S343;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_count(@builtin(global_invocation_id) id_12 : vec3<u32>)
{
    var c_10 : u32 = id_12.x;
    var _S345 : bool = is_root_0(c_10);
    if(!_S345)
    {
        return;
    }
    var _S346 : u32 = atomicLoad(&(atom_0[a_count_0() + c_10]));
    var _S347 : u32 = atomicMax(&(atom_0[a_best_count_0() + body_of_0(c_10)]), _S346);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_best_root(@builtin(global_invocation_id) id_13 : vec3<u32>)
{
    var c_11 : u32 = id_13.x;
    var _S348 : bool = is_root_0(c_11);
    if(!_S348)
    {
        return;
    }
    var s_8 : u32 = body_of_0(c_11);
    var _S349 : u32 = atomicLoad(&(atom_0[a_count_0() + c_11]));
    var _S350 : u32 = atomicLoad(&(atom_0[a_best_count_0() + s_8]));
    if(_S349 == _S350)
    {
        var _S351 : u32 = atomicMin(&(atom_0[a_best_root_0() + s_8]), c_11);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_mark(@builtin(global_invocation_id) id_14 : vec3<u32>)
{
    var c_12 : u32 = id_14.x;
    if(c_12 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S352 : bool = is_root_0(c_12);
    var _S353 : bool;
    if(_S352)
    {
        var _S354 : u32 = atomicLoad(&(atom_0[a_best_root_0() + body_of_0(c_12)]));
        _S353 = _S354 != c_12;
    }
    else
    {
        _S353 = false;
    }
    var _S355 : u32;
    if(_S353)
    {
        _S355 = u32(1);
    }
    else
    {
        _S355 = u32(0);
    }
    work_0[w_new_0() + c_12] = _S355;
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
    var _S356 : u32 = group_0.x;
    var _S357 : u32 = u32(1024) * _S356 + u32(4) * t_9;
    var v_10 : array<u32, i32(4)>;
    var j_5 : u32 = u32(0);
    var sum_2 : u32 = u32(0);
    loop
    {
        if(j_5 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S357 + j_5) < (scan_0.n_0))
        {
            step_4 = work_0[scan_0.in_base_0 + _S357 + j_5];
        }
        else
        {
            step_4 = u32(0);
        }
        v_10[j_5] = step_4;
        var sum_3 : u32 = sum_2 + step_4;
        j_5 = j_5 + u32(1);
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
    var _S358 : u32 = g_scan_0[t_9] - sum_2;
    j_5 = u32(0);
    add_0 = _S358;
    loop
    {
        if(j_5 < u32(4))
        {
        }
        else
        {
            break;
        }
        if((_S357 + j_5) < (scan_0.n_0))
        {
            work_0[scan_0.out_base_0 + _S357 + j_5] = add_0;
        }
        var run_1 : u32 = add_0 + v_10[j_5];
        j_5 = j_5 + u32(1);
        add_0 = run_1;
    }
    if(t_9 == u32(255))
    {
        work_0[scan_0.block_base_0 + _S356] = g_scan_0[i32(255)];
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn fr_scan_top(@builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var t_10 : u32 = thread_3.x;
    var blocks_0 : u32 = (scan_0.n_0 + u32(1023)) / u32(1024);
    var _S359 : bool = t_10 < blocks_0;
    var v_11 : u32;
    if(_S359)
    {
        v_11 = work_0[scan_0.block_base_0 + t_10];
    }
    else
    {
        v_11 = u32(0);
    }
    g_scan_0[t_10] = v_11;
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
    if(_S359)
    {
        work_0[scan_0.block_base_0 + t_10] = g_scan_0[t_10] - v_11;
    }
    if(t_10 == u32(1023))
    {
        work_0[scan_0.block_base_0 + blocks_0] = g_scan_0[i32(1023)];
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_scan_add(@builtin(global_invocation_id) id_15 : vec3<u32>)
{
    var i_9 : u32 = id_15.x;
    if(i_9 >= (scan_0.n_0))
    {
        return;
    }
    work_0[scan_0.out_base_0 + i_9] = work_0[scan_0.out_base_0 + i_9] + work_0[scan_0.block_base_0 + i_9 / u32(1024)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_freelist(@builtin(global_invocation_id) id_16 : vec3<u32>)
{
    var s_9 : u32 = id_16.x;
    var _S360 : bool;
    if(s_9 >= (params_0.body_count_0))
    {
        _S360 = true;
    }
    else
    {
        _S360 = work_0[w_free_0() + s_9] == u32(0);
    }
    if(_S360)
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
fn fr_assign(@builtin(global_invocation_id) id_17 : vec3<u32>)
{
    var c_13 : u32 = id_17.x;
    var _S361 : bool = is_root_0(c_13);
    if(!_S361)
    {
        return;
    }
    var s_10 : u32 = body_of_0(c_13);
    var _S362 : u32 = atomicLoad(&(atom_0[a_best_root_0() + s_10]));
    var slot_10 : u32;
    if(_S362 != c_13)
    {
        var k_25 : u32 = work_0[w_rank_0() + c_13];
        if(work_0[w_rank_0() + c_13] < (free_total_0()))
        {
            var slot_11 : u32 = work_0[w_free_list_0() + k_25];
            work_0[w_free_0() + work_0[w_free_list_0() + k_25]] = u32(0);
            var _S363 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(2)]), u32(1));
            slot_10 = slot_11;
        }
        else
        {
            var _S364 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(1));
            slot_10 = s_10;
        }
    }
    else
    {
        slot_10 = s_10;
    }
    work_0[w_slot_0() + c_13] = slot_10;
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
    var _S365 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S366 : array<f32, i32(3)> = array<f32, i32(3)>( r1_2.x, r1_2.y, r1_2.z );
    var _S367 : array<f32, i32(3)> = array<f32, i32(3)>( r2_2.x, r2_2.y, r2_2.z );
    var a_9 : array<array<f32, i32(3)>, i32(3)>;
    a_9[i32(0)] = _S365;
    a_9[i32(1)] = _S366;
    a_9[i32(2)] = _S367;
    var _S368 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S369 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S370 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_12 : array<array<f32, i32(3)>, i32(3)>;
    v_12[i32(0)] = _S368;
    v_12[i32(1)] = _S369;
    v_12[i32(2)] = _S370;
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
        var c_14 : f32 = cos(theta_0);
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
            var _S371 : bool;
            if(k_26 != p_14)
            {
                _S371 = k_26 != q_20;
            }
            else
            {
                _S371 = false;
            }
            if(_S371)
            {
                var kp_0 : f32 = a_9[k_26][p_14];
                var kq_0 : f32 = a_9[k_26][q_20];
                var _S372 : f32 = c_14 * a_9[k_26][p_14] - s_11 * a_9[k_26][q_20];
                a_9[k_26][p_14] = _S372;
                a_9[p_14][k_26] = _S372;
                var _S373 : f32 = s_11 * kp_0 + c_14 * kq_0;
                a_9[k_26][q_20] = _S373;
                a_9[q_20][k_26] = _S373;
            }
            k_26 = k_26 + u32(1);
        }
        var _S374 : f32 = c_14 * c_14;
        var _S375 : f32 = 2.0f * s_11 * c_14 * apq_0;
        var _S376 : f32 = s_11 * s_11;
        a_9[p_14][p_14] = _S374 * app_0 - _S375 + _S376 * aqq_0;
        a_9[q_20][q_20] = _S376 * app_0 + _S375 + _S374 * aqq_0;
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
            var kp_1 : f32 = v_12[row_1][p_14];
            var kq_1 : f32 = v_12[row_1][q_20];
            v_12[row_1][p_14] = c_14 * v_12[row_1][p_14] - s_11 * v_12[row_1][q_20];
            v_12[row_1][q_20] = s_11 * kp_1 + c_14 * kq_1;
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

fn stage_0( c_15 : u32) -> u32
{
    return u32(2) * params_0.body_count_0 + u32(9) * c_15;
}

fn quat_vec_0( q_22 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_22.x_0, q_22.y_0, q_22.z_0, q_22.w_0);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_piece(@builtin(global_invocation_id) id_18 : vec3<u32>)
{
    var c_16 : u32 = id_18.x;
    var _S377 : bool = is_root_0(c_16);
    if(!_S377)
    {
        return;
    }
    var parent_0 : u32 = body_of_0(c_16);
    var pb_2 : u32 = u32(7) * parent_0;
    var at_14 : u32 = a_sums_0() + u32(20) * c_16;
    var mass_1 : f32 = read64_0(at_14, params_0.scale_m_0);
    var _S378 : f32 = read64_0(at_14 + u32(2), params_0.scale_s_0);
    var _S379 : f32 = read64_0(at_14 + u32(4), params_0.scale_s_0);
    var _S380 : f32 = read64_0(at_14 + u32(6), params_0.scale_s_0);
    var d_21 : vec3<f32> = vec3<f32>(_S378, _S379, _S380) / vec3<f32>(max(mass_1, 1.00000000317107685e-30f));
    var jxx_0 : f32 = read64_0(at_14 + u32(8), params_0.scale_j_0);
    var jyy_0 : f32 = read64_0(at_14 + u32(10), params_0.scale_j_0);
    var jzz_0 : f32 = read64_0(at_14 + u32(12), params_0.scale_j_0);
    var jxy_0 : f32 = read64_0(at_14 + u32(14), params_0.scale_j_0);
    var jxz_0 : f32 = read64_0(at_14 + u32(16), params_0.scale_j_0);
    var jyz_0 : f32 = read64_0(at_14 + u32(18), params_0.scale_j_0);
    var dd_1 : f32 = dot(d_21, d_21);
    var _S381 : f32 = d_21.x;
    var _S382 : f32 = mass_1 * _S381;
    var _S383 : f32 = d_21.y;
    var _S384 : f32 = jxy_0 + _S382 * _S383;
    var _S385 : f32 = d_21.z;
    var _S386 : f32 = jxz_0 + _S382 * _S385;
    var _S387 : f32 = jyz_0 + mass_1 * _S383 * _S385;
    var com_6 : vec3<f32> = chunks_0[u32(4) * c_16].xyz + d_21;
    var principal_0 : Quat_0;
    var moments_1 : vec3<f32>;
    principal_axes_0(vec3<f32>(jxx_0 - mass_1 * (dd_1 - _S381 * _S381), _S384, _S386), vec3<f32>(_S384, jyy_0 - mass_1 * (dd_1 - _S383 * _S383), _S387), vec3<f32>(_S386, _S387, jzz_0 - mass_1 * (dd_1 - _S385 * _S385)), &(principal_0), &(moments_1));
    var _S388 : u32 = atomicLoad(&(atom_0[a_anchor_0() + c_16]));
    var anchored_0 : bool = _S388 != u32(0);
    var q_23 : Quat_0;
    var origin_10 : vec3<f32>;
    structure_pose_5(parent_0, pb_2, &(q_23), &(origin_10));
    var p_parent_0 : vec3<f32> = bodies_0[pb_2].xyz;
    var v_parent_0 : vec3<f32> = bodies_0[pb_2 + u32(3)].xyz;
    var w_parent_0 : vec3<f32> = bodies_0[pb_2 + u32(4)].xyz;
    var pos_0 : vec3<f32> = to_rubble_0(origin_10 + rotate_0(q_23, com_6));
    var orientation_0 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(q_23, principal_0)));
    var _S389 : bool;
    if(anchored_0)
    {
        _S389 = true;
    }
    else
    {
        _S389 = !(mass_1 > 0.0f);
    }
    var inv_mass_3 : f32;
    if(_S389)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / mass_1;
    }
    var _S390 : vec3<f32> = vec3<f32>(0.0f);
    var inv_inertia_0 : vec3<f32>;
    if(!anchored_0)
    {
        var _S391 : f32;
        if((moments_1.x) > 0.0f)
        {
            _S391 = 1.0f / moments_1.x;
        }
        else
        {
            _S391 = 0.0f;
        }
        var _S392 : f32;
        if((moments_1.y) > 0.0f)
        {
            _S392 = 1.0f / moments_1.y;
        }
        else
        {
            _S392 = 0.0f;
        }
        var _S393 : f32;
        if((moments_1.z) > 0.0f)
        {
            _S393 = 1.0f / moments_1.z;
        }
        else
        {
            _S393 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S391, _S392, _S393);
    }
    else
    {
        inv_inertia_0 = _S390;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_0)
    {
        velocity_0 = _S390;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    var spin_0 : vec3<f32>;
    if(anchored_0)
    {
        spin_0 = _S390;
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
    frames_0[st_2 + u32(7)] = vec4<f32>(com_6, frames_0[u32(2) * parent_0].w);
    frames_0[st_2 + u32(8)] = quat_vec_0(principal_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_commit(@builtin(global_invocation_id) id_19 : vec3<u32>)
{
    var c_17 : u32 = id_19.x;
    var _S394 : bool = is_root_0(c_17);
    if(!_S394)
    {
        return;
    }
    var slot_12 : u32 = work_0[w_slot_0() + c_17];
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
            bodies_0[u32(7) * (slot_12 + region_5 * params_0.max_bodies_0) + w_4] = frames_0[st_3 + w_4];
            w_4 = w_4 + u32(1);
        }
        region_5 = region_5 + u32(1);
    }
    var _S395 : u32 = u32(2) * slot_12;
    frames_0[_S395] = frames_0[st_3 + u32(7)];
    frames_0[_S395 + u32(1)] = frames_0[st_3 + u32(8)];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_reparent(@builtin(global_invocation_id) id_20 : vec3<u32>)
{
    var c_18 : u32 = id_20.x;
    var _S396 : bool;
    if(c_18 >= (params_0.chunk_count_0))
    {
        _S396 = true;
    }
    else
    {
        var _S397 : bool = dirty_0(body_of_0(c_18));
        _S396 = !_S397;
    }
    if(_S396)
    {
        return;
    }
    var r_21 : u32 = work_0[w_label_0() + c_18];
    var slot_13 : u32 = work_0[w_slot_0() + work_0[w_label_0() + c_18]];
    work_0[w_body_0() + c_18] = work_0[w_slot_0() + work_0[w_label_0() + c_18]];
    var _S398 : u32 = u32(2) * slot_13;
    var _S399 : vec3<f32> = frames_0[_S398].xyz;
    var _S400 : Quat_0 = conj_0(quat_of_0(frames_0[_S398 + u32(1)]));
    var inv_mass_4 : f32 = bodies_0[u32(7) * slot_13].w;
    var _S401 : f32;
    if(inv_mass_4 > 0.0f)
    {
        _S401 = 1.0f / inv_mass_4;
    }
    else
    {
        _S401 = float_bits_0(u32(2139095040));
    }
    var c2_2 : vec4<f32> = chunks_0[u32(4) * c_18 + u32(2)];
    var first_1 : u32 = (bitcast<u32>((c2_2.z)));
    var _S402 : u32 = (bitcast<u32>((c2_2.w)));
    var _S403 : vec4<f32> = chunks_0[u32(4) * r_21 + u32(3)];
    var p_15 : u32 = first_1;
    loop
    {
        if(p_15 < (first_1 + _S402))
        {
        }
        else
        {
            break;
        }
        var _S404 : u32 = u32(2) * p_15;
        var o_4 : vec4<f32> = parts_0[_S404];
        var col_0 : u32 = (bitcast<u32>((o_4.w)));
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S404 + u32(1)]);
        var _S405 : u32 = u32(4) * col_0;
        colliders_0[_S405] = vec4<f32>((bitcast<f32>((slot_13))), colliders_0[_S405].y, colliders_0[_S405].z, colliders_0[_S405].w);
        var _S406 : u32 = _S405 + u32(1);
        colliders_0[_S406] = vec4<f32>(rotate_0(_S400, o_4.xyz - _S399), colliders_0[_S406].w);
        colliders_0[_S405 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S400, rot_1)));
        var _S407 : u32 = u32(2) * col_0;
        collider_info_0[_S407] = vec4<u32>(collider_info_0[_S407].x, slot_13, collider_info_0[_S407].z, collider_info_0[_S407].w);
        collider_info_0[_S407 + u32(1)] = vec4<u32>((bitcast<u32>((_S401))), (bitcast<u32>((_S403.x))), (bitcast<u32>((_S403.w))), u32(0));
        p_15 = p_15 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_body_clear(@builtin(global_invocation_id) id_21 : vec3<u32>)
{
    var s_12 : u32 = id_21.x;
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
    var _S408 : vec3<f32>;
    if(k_27 == u32(0))
    {
        _S408 = vec3<f32>(0.92000001668930054f, t_11, 0.34959998726844788f);
    }
    else
    {
        if(k_27 == u32(1))
        {
            _S408 = vec3<f32>(q_24, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_27 == u32(2))
            {
                _S408 = vec3<f32>(0.34959998726844788f, 0.92000001668930054f, t_11);
            }
            else
            {
                if(k_27 == u32(3))
                {
                    _S408 = vec3<f32>(0.34959998726844788f, q_24, 0.92000001668930054f);
                }
                else
                {
                    if(k_27 == u32(4))
                    {
                        _S408 = vec3<f32>(t_11, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S408 = vec3<f32>(0.92000001668930054f, 0.34959998726844788f, q_24);
                    }
                }
            }
        }
    }
    return _S408;
}

fn write_object_0( o_5 : u32,  q_25 : Quat_0,  t_12 : vec3<f32>,  color_0 : vec4<f32>)
{
    var x_7 : vec3<f32> = rotate_0(q_25, vec3<f32>(1.0f, 0.0f, 0.0f));
    var y_2 : vec3<f32> = rotate_0(q_25, vec3<f32>(0.0f, 1.0f, 0.0f));
    var z_3 : vec3<f32> = rotate_0(q_25, vec3<f32>(0.0f, 0.0f, 1.0f));
    var _S409 : u32 = u32(4) * o_5;
    objects_0[_S409] = vec4<f32>(x_7.x, y_2.x, z_3.x, t_12.x);
    objects_0[_S409 + u32(1)] = vec4<f32>(x_7.y, y_2.y, z_3.y, t_12.y);
    objects_0[_S409 + u32(2)] = vec4<f32>(x_7.z, y_2.z, z_3.z, t_12.z);
    objects_0[_S409 + u32(3)] = color_0;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_chunks(@builtin(global_invocation_id) id_22 : vec3<u32>)
{
    var k_28 : u32 = id_22.x;
    if(k_28 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_15 : u32 = body_of_0(k_28);
    var b_51 : u32 = u32(7) * slot_15;
    var q_26 : Quat_0;
    var origin_11 : vec3<f32>;
    structure_pose_5(slot_15, b_51, &(q_26), &(origin_11));
    var _S410 : u32 = u32(3) * k_28;
    var v0_0 : vec4<f32> = view_chunks_0[_S410];
    var base_2 : vec3<f32> = view_chunks_0[_S410 + u32(2)].xyz;
    var fixed_0 : bool = (bodies_0[b_51].w) == 0.0f;
    var color_1 : vec3<f32>;
    if((view_0.mode_0) == u32(1))
    {
        var _S411 : f32 = max(length(bodies_0[b_51 + u32(3)].xyz), length(bodies_0[b_51 + u32(4)].xyz));
        if(fixed_0)
        {
            color_1 = vec3<f32>(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S411 > (view_0.rest_speed_0))
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
    write_object_0((bitcast<u32>((v0_0.w))), normalized_0(quat_mul_0(q_26, quat_of_0(view_chunks_0[_S410 + u32(1)]))), origin_11 + rotate_0(q_26, v0_0.xyz), vec4<f32>(color_1, 1.0f));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_shots(@builtin(global_invocation_id) id_23 : vec3<u32>)
{
    var i_10 : u32 = id_23.x;
    if(i_10 >= (view_0.shot_count_1))
    {
        return;
    }
    var b_52 : u32 = u32(7) * shot_bodies_0[i_10];
    var o_6 : u32 = view_0.impactor_base_0 + i_10;
    if((bodies_0[b_52].w) == 0.0f)
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
    write_object_0(o_6, rot_from_rubble_0(quat_of_0(bodies_0[b_52 + u32(1)])), from_rubble_0(bodies_0[b_52].xyz), vec4<f32>(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f));
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

fn write_line_vertex_0( v_13 : u32,  p_16 : vec3<f32>)
{
    var _S412 : u32 = u32(7) * v_13;
    lines_0[_S412] = p_16.x;
    lines_0[_S412 + u32(1)] = p_16.y;
    lines_0[_S412 + u32(2)] = p_16.z;
    lines_0[_S412 + u32(3)] = 1.0f;
    lines_0[_S412 + u32(4)] = 0.86000001430511475f;
    lines_0[_S412 + u32(5)] = 0.25f;
    lines_0[_S412 + u32(6)] = 1.0f;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_view_bonds(@builtin(global_invocation_id) id_24 : vec3<u32>)
{
    var i_11 : u32 = id_24.x;
    if(i_11 >= (params_0.bond_count_0))
    {
        return;
    }
    var bond_4 : vec4<u32> = bonds_0[i_11];
    var a_10 : vec3<f32> = chunk_world_0(bond_4.x);
    var b_53 : vec3<f32>;
    if(work_0[w_alive_0() + i_11] != u32(0))
    {
        b_53 = chunk_world_0(bond_4.y);
    }
    else
    {
        b_53 = a_10;
    }
    var _S413 : u32 = u32(2) * i_11;
    write_line_vertex_0(_S413, a_10);
    write_line_vertex_0(_S413 + u32(1), b_53);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_reset(@builtin(global_invocation_id) id_25 : vec3<u32>)
{
    var b_54 : u32 = id_25.x;
    if(b_54 >= (params_0.body_count_0))
    {
        return;
    }
    var at_15 : u32 = a_strike_0() + u32(9) * b_54;
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
        atomicStore(&(atom_0[at_15 + w_6]), u32(4294967295));
        atomicStore(&(atom_0[at_15 + u32(3) + w_6]), u32(0));
        w_6 = w_6 + u32(1);
    }
    atomicStore(&(atom_0[at_15 + u32(6)]), u32(2139095040));
    atomicStore(&(atom_0[at_15 + u32(7)]), u32(2139095040));
    atomicStore(&(atom_0[at_15 + u32(8)]), u32(4294967295));
    frames_0[strike_at_0(b_54)][i32(3)] = 0.0f;
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
    var u_6 : u32 = (bitcast<u32>((f_9)));
    var _S414 : u32;
    if(((u_6 & (u32(2147483648)))) != u32(0))
    {
        _S414 = ~u_6;
    }
    else
    {
        _S414 = (u_6 | (u32(2147483648)));
    }
    return _S414;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_bounds(@builtin(global_invocation_id) id_26 : vec3<u32>)
{
    var k_32 : u32 = id_26.x;
    var _S415 : bool;
    if(k_32 >= (params_0.chunk_count_0))
    {
        _S415 = true;
    }
    else
    {
        _S415 = (params_0.shot_count_0) == u32(0);
    }
    if(_S415)
    {
        return;
    }
    var q_30 : Quat_0;
    var _S416 : vec3<f32> = chunk_box_world_3(k_32, &(q_30));
    var _S417 : f32 = geometry_0[u32(3) * k_32].w;
    var _S418 : u32 = a_strike_0() + u32(9) * body_of_0(k_32);
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
        var _S419 : u32 = ax_1;
        var _S420 : u32 = atomicMin(&(atom_0[_S418 + ax_1]), ordered_0(_S416[ax_1] - _S417));
        var _S421 : u32 = atomicMax(&(atom_0[_S418 + u32(3) + ax_1]), ordered_0(_S416[_S419] + _S417));
        ax_1 = ax_1 + u32(1);
    }
    return;
}

fn unordered_0( u_7 : u32) -> f32
{
    var _S422 : u32;
    if(((u_7 & (u32(2147483648)))) != u32(0))
    {
        _S422 = (u_7 & (u32(2147483647)));
    }
    else
    {
        _S422 = ~u_7;
    }
    return (bitcast<f32>((_S422)));
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_find(@builtin(global_invocation_id) id_27 : vec3<u32>)
{
    var b_55 : u32 = id_27.x;
    var _S423 : bool;
    if(b_55 >= (params_0.body_count_0))
    {
        _S423 = true;
    }
    else
    {
        _S423 = (params_0.shot_count_0) == u32(0);
    }
    if(_S423)
    {
        return;
    }
    var at_16 : u32 = a_strike_0() + u32(9) * b_55;
    var _S424 : u32 = atomicLoad(&(atom_0[at_16]));
    if(_S424 == u32(4294967295))
    {
        return;
    }
    var _S425 : u32 = atomicLoad(&(atom_0[at_16]));
    var _S426 : f32 = unordered_0(_S425);
    var _S427 : u32 = atomicLoad(&(atom_0[at_16 + u32(1)]));
    var _S428 : f32 = unordered_0(_S427);
    var _S429 : u32 = atomicLoad(&(atom_0[at_16 + u32(2)]));
    var _S430 : vec3<f32> = vec3<f32>(_S426, _S428, unordered_0(_S429));
    var _S431 : u32 = atomicLoad(&(atom_0[at_16 + u32(3)]));
    var _S432 : f32 = unordered_0(_S431);
    var _S433 : u32 = atomicLoad(&(atom_0[at_16 + u32(4)]));
    var _S434 : f32 = unordered_0(_S433);
    var _S435 : u32 = atomicLoad(&(atom_0[at_16 + u32(5)]));
    var _S436 : vec3<f32> = vec3<f32>(_S432, _S434, unordered_0(_S435));
    var _S437 : vec3<f32> = from_rubble_0(bodies_0[u32(7) * b_55 + u32(3)].xyz);
    var _S438 : f32 = float_bits_0(u32(2139095040));
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
        var c_19 : vec3<f32> = from_rubble_0(bodies_0[sb_1].xyz);
        var rel_3 : vec3<f32> = from_rubble_0(bodies_0[sb_1 + u32(3)].xyz) - _S437;
        if(!((length(rel_3)) > (params_0.strike_speed_0)))
        {
            k_33 = k_33 + u32(1);
            continue;
        }
        var _S439 : vec3<f32> = vec3<f32>(info_2.y);
        var _S440 : vec3<f32> = _S430 - _S439;
        var _S441 : vec3<f32> = _S436 + _S439;
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
            var _S442 : u32 = ax_2;
            if((abs(rel_3[ax_2])) < 9.999999960041972e-13f)
            {
                var _S443 : u32 = ax_2;
                if((c_19[ax_2]) < (_S440[ax_2]))
                {
                    _S423 = true;
                }
                else
                {
                    _S423 = (c_19[_S443]) > (_S441[ax_2]);
                }
                var t0_2 : f32;
                if(_S423)
                {
                    t0_2 = _S438;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + u32(1);
                continue;
            }
            var ta_1 : f32 = (_S440[ax_2] - c_19[ax_2]) / rel_3[_S442];
            var tb_1 : f32 = (_S441[ax_2] - c_19[ax_2]) / rel_3[_S442];
            var _S444 : f32 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S444;
            ax_2 = ax_2 + u32(1);
        }
        if(t0_1 <= t1_1)
        {
            var st_4 : u32 = strike_at_0(b_55);
            frames_0[st_4] = vec4<f32>(c_19, f32(k_33 + u32(1)));
            frames_0[st_4 + u32(1)] = vec4<f32>(rel_3, t0_1);
            return;
        }
        k_33 = k_33 + u32(1);
    }
    return;
}

fn struck_2( b_56 : u32,  shot_5 : ptr<function, u32>) -> bool
{
    var k_34 : u32 = u32(frames_0[strike_at_0(b_56)].w);
    (*shot_5) = k_34 - u32(1);
    return k_34 != u32(0);
}

fn a_strike_dist_0() -> u32
{
    return a_cs_0(u32(0)) + u32(12) * params_0.body_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_entry(@builtin(global_invocation_id) id_28 : vec3<u32>)
{
    var k_35 : u32 = id_28.x;
    var _S445 : bool;
    if(k_35 >= (params_0.chunk_count_0))
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
    var b_57 : u32 = body_of_0(k_35);
    var shot_6 : u32;
    var _S446 : bool = struck_2(b_57, &(shot_6));
    if(!_S446)
    {
        return;
    }
    var _S447 : u32 = strike_at_0(b_57);
    var s1_4 : vec4<f32> = frames_0[_S447 + u32(1)];
    var from_1 : vec3<f32> = frames_0[_S447].xyz;
    var rel_4 : vec3<f32> = frames_0[_S447 + u32(1)].xyz;
    var d_22 : vec3<f32> = rel_4 / vec3<f32>(max(length(rel_4), 9.999999960041972e-13f));
    var _S448 : f32 = shots_0[u32(2) * shot_6].y;
    var q_31 : Quat_0;
    var cc_1 : vec3<f32> = chunk_box_world_3(k_35, &(q_31));
    var _S449 : u32 = u32(3) * k_35;
    var _S450 : Quat_0 = normalized_0(quat_mul_0(q_31, quat_of_0(geometry_0[_S449 + u32(2)])));
    var _S451 : vec3<f32> = geometry_0[_S449 + u32(1)].xyz;
    var _S452 : f32 = float_bits_0(u32(2139095040));
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
        var axis_1 : vec3<f32> = rotate_0(_S450, e_3);
        var o_7 : f32 = dot(from_1 - cc_1, axis_1);
        var dv_1 : f32 = dot(rel_4, axis_1);
        var ext_2 : f32 = _S451[ax_3] + _S448;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            var t_in_8 : f32;
            if((abs(o_7)) > ext_2)
            {
                t_in_8 = _S452;
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
        var _S453 : f32 = min(t_out_0, max(ta_2, tb_2));
        t_in_7 = max(t_in_7, min(ta_2, tb_2));
        t_out_0 = _S453;
        ax_3 = ax_3 + u32(1);
    }
    var at_17 : u32 = a_strike_0() + u32(9) * b_57;
    if(t_in_7 <= t_out_0)
    {
        var _S454 : u32 = atomicMin(&(atom_0[at_17 + u32(6)]), (bitcast<u32>((t_in_7))));
    }
    var r_22 : vec3<f32> = cc_1 - (from_1 + rel_4 * vec3<f32>(s1_4.w));
    var along_1 : f32 = dot(r_22, d_22);
    var _S455 : u32 = (bitcast<u32>((length(r_22 - d_22 * vec3<f32>(along_1)) + abs(along_1) * 0.00100000004749745f)));
    atomicStore(&(atom_0[a_strike_dist_0() + k_35]), _S455);
    var _S456 : u32 = atomicMin(&(atom_0[at_17 + u32(7)]), _S455);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_nearest(@builtin(global_invocation_id) id_29 : vec3<u32>)
{
    var k_36 : u32 = id_29.x;
    var _S457 : bool;
    if(k_36 >= (params_0.chunk_count_0))
    {
        _S457 = true;
    }
    else
    {
        _S457 = (params_0.shot_count_0) == u32(0);
    }
    if(_S457)
    {
        return;
    }
    var b_58 : u32 = body_of_0(k_36);
    var shot_7 : u32;
    if(!struck_2(b_58, &(shot_7)))
    {
        return;
    }
    var at_18 : u32 = a_strike_0() + u32(9) * b_58;
    var _S458 : u32 = atomicLoad(&(atom_0[a_strike_dist_0() + k_36]));
    var _S459 : u32 = atomicLoad(&(atom_0[at_18 + u32(7)]));
    if(_S458 == _S459)
    {
        var _S460 : u32 = atomicMin(&(atom_0[at_18 + u32(8)]), k_36);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_force(@builtin(global_invocation_id) id_30 : vec3<u32>)
{
    var b_59 : u32 = id_30.x;
    var _S461 : bool;
    if(b_59 >= (params_0.body_count_0))
    {
        _S461 = true;
    }
    else
    {
        _S461 = (params_0.shot_count_0) == u32(0);
    }
    if(_S461)
    {
        return;
    }
    var shot_8 : u32;
    var _S462 : bool = struck_2(b_59, &(shot_8));
    if(!_S462)
    {
        return;
    }
    var at_19 : u32 = a_strike_0() + u32(9) * b_59;
    var _S463 : u32 = atomicLoad(&(atom_0[at_19 + u32(6)]));
    var t_in_9 : f32 = (bitcast<f32>((_S463)));
    var entry_8 : u32 = atomicLoad(&(atom_0[at_19 + u32(8)]));
    var st_5 : u32 = strike_at_0(b_59);
    if(!(t_in_9 <= (params_0.dt_0)))
    {
        _S461 = true;
    }
    else
    {
        _S461 = entry_8 == u32(4294967295);
    }
    if(_S461)
    {
        frames_0[st_5][i32(3)] = 0.0f;
        return;
    }
    var rel_5 : vec3<f32> = frames_0[st_5 + u32(1)].xyz;
    var speed_16 : f32 = length(rel_5);
    var info_3 : vec4<f32> = shots_0[u32(2) * shot_8];
    var _S464 : f32 = max(info_3.z, 0.00100000004749745f);
    var m_imp_0 : f32 = info_3.w;
    var e_imp_0 : f32 = shots_0[u32(2) * shot_8 + u32(1)].x;
    var contact_18 : vec3<f32> = frames_0[st_5].xyz + rel_5 * vec3<f32>(t_in_9) + rel_5 / vec3<f32>(max(speed_16, 9.999999960041972e-13f)) * vec3<f32>(info_3.y);
    var inv_mass_5 : f32 = bodies_0[u32(7) * b_59].w;
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
    frames_0[st_5 + u32(2)] = vec4<f32>(contact_18, m_eff_6 * speed_16 / (2.86999988555908203f * pow(m_eff_6 * m_eff_6 / (_S464 * modulus_1 * modulus_1 * max(speed_16, 0.00100000004749745f)), 0.20000000298023224f)));
    return;
}

fn strike_share_breaks_0( bond_5 : vec4<u32>,  contact_19 : vec3<f32>,  force_2 : f32) -> bool
{
    var q_32 : Quat_0;
    var pa_2 : vec3<f32> = chunk_box_world_2(bond_5.x, &(q_32));
    var pb_3 : vec3<f32> = chunk_box_world_2(bond_5.y, &(q_32));
    var dist_0 : f32 = length(vec3<f32>(0.5f) * (pa_2 + pb_3) - contact_19);
    return (force_2 * min(1.0f, (bitcast<f32>((bond_5.w))) / (6.28318548202514648f * max(dist_0 * dist_0, 9.999999960041972e-13f)))) > (bitcast<f32>((bond_5.z)));
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_breaks(@builtin(global_invocation_id) id_31 : vec3<u32>)
{
    var i_12 : u32 = id_31.x;
    var _S465 : bool;
    if(i_12 >= (params_0.bond_count_0))
    {
        _S465 = true;
    }
    else
    {
        _S465 = (params_0.shot_count_0) == u32(0);
    }
    if(_S465)
    {
        _S465 = true;
    }
    else
    {
        _S465 = work_0[w_alive_0() + i_12] == u32(0);
    }
    if(_S465)
    {
        return;
    }
    var bond_6 : vec4<u32> = bonds_0[i_12];
    var b_60 : u32 = body_of_0(bond_6.x);
    var shot_9 : u32;
    if(!struck_2(b_60, &(shot_9)))
    {
        return;
    }
    if(strike_share_breaks_0(bond_6, frames_0[strike_at_0(b_60) + u32(2)].xyz, frames_0[strike_at_0(b_60) + u32(2)].w))
    {
        work_0[w_alive_0() + i_12] = u32(0);
        var _S466 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_60]), u32(1));
        var _S467 : u32 = a_counters_0();
        var _S468 : u32 = atomicAdd(&(atom_0[_S467]), u32(1));
        var _S469 : u32 = atomicAdd(&(atom_0[_S467 + u32(1)]), u32(1));
    }
    return;
}

fn a_lp_0( b_61 : u32) -> u32
{
    return a_strike_0() + u32(9) * params_0.body_count_0 + u32(14) * b_61;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_clear(@builtin(global_invocation_id) id_32 : vec3<u32>)
{
    var b_62 : u32 = id_32.x;
    if(b_62 >= (params_0.body_count_0))
    {
        return;
    }
    var at_20 : u32 = a_lp_0(b_62);
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
        atomicStore(&(atom_0[at_20 + w_7]), u32(0));
        w_7 = w_7 + u32(1);
    }
    atomicStore(&(atom_0[at_20 + u32(12)]), u32(4294967295));
    atomicStore(&(atom_0[at_20 + u32(13)]), u32(0));
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

fn add64w_0( at_21 : u32,  value_3 : f32,  scale_2 : f32)
{
    var x_8 : f32 = round(value_3 * scale_2);
    var x_9 : f32;
    if(!((abs(x_8)) < 3.99999993722699776e+18f))
    {
        var _S470 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(2));
        x_9 = 0.0f;
    }
    else
    {
        x_9 = x_8;
    }
    var u_8 : f32 = abs(x_9);
    var h_6 : f32 = floor(u_8 * 2.32830643653869629e-10f);
    var lo_3 : u32 = u32(u_8 - h_6 * 4.294967296e+09f);
    var hi_3 : u32 = u32(h_6);
    var lo_4 : u32;
    var hi_4 : u32;
    if(x_9 < 0.0f)
    {
        var lo_5 : u32 = ~lo_3 + u32(1);
        var _S471 : u32 = ~hi_3;
        if(lo_5 == u32(0))
        {
            lo_4 = u32(1);
        }
        else
        {
            lo_4 = u32(0);
        }
        var _S472 : u32 = _S471 + lo_4;
        lo_4 = lo_5;
        hi_4 = _S472;
    }
    else
    {
        lo_4 = lo_3;
        hi_4 = hi_3;
    }
    var old_1 : u32 = atomicAdd(&(atom_0[at_21]), lo_4);
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
        var _S473 : u32 = atomicAdd(&(atom_0[at_21 + u32(1)]), hi_5);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sum(@builtin(global_invocation_id) id_33 : vec3<u32>)
{
    var k_39 : u32 = id_33.x;
    if(k_39 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_19 : u32 = body_of_0(k_39);
    var q_33 : Quat_0;
    var origin_15 : vec3<f32>;
    structure_pose_5(slot_19, u32(7) * slot_19, &(q_33), &(origin_15));
    var qi_0 : Quat_0 = conj_0(q_33);
    var c0_3 : vec4<f32> = chunks_0[u32(4) * k_39];
    var _S474 : u32 = l_vec_0() + u32(2) * k_39;
    var f_10 : vec3<f32> = rotate_0(qi_0, loads_0[_S474].xyz) + vec3<f32>(c0_3.w) * rotate_0(qi_0, vec3<f32>(params_0.gx_0, params_0.gy_0, params_0.gz_0));
    var t_13 : vec3<f32> = rotate_0(qi_0, loads_0[_S474 + u32(1)].xyz);
    var _S475 : u32 = lp_f_0(k_39);
    lp_0[_S475] = vec4<f32>(f_10, 0.0f);
    lp_0[_S475 + u32(1)] = vec4<f32>(t_13, 0.0f);
    var _S476 : vec3<f32> = t_13 + cross(c0_3.xyz - frames_0[u32(2) * slot_19].xyz, f_10);
    var at_22 : u32 = a_lp_0(slot_19);
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
        var _S477 : u32 = u32(2) * a_11;
        add64w_0(at_22 + _S477, f_10[a_11], params_0.lp_scale_0);
        add64w_0(at_22 + u32(6) + _S477, _S476[a_11], params_0.lp_scale_0);
        a_11 = a_11 + u32(1);
    }
    var _S478 : u32 = atomicMin(&(atom_0[at_22 + u32(12)]), k_39);
    if(((info_0[k_39] & (u32(1)))) != u32(0))
    {
        atomicStore(&(atom_0[at_22 + u32(13)]), u32(1));
    }
    return;
}

fn read64w3_0( at_23 : u32,  scale_3 : f32) -> vec3<f32>
{
    var _S479 : f32 = read64_0(at_23, scale_3);
    var _S480 : f32 = read64_0(at_23 + u32(2), scale_3);
    var _S481 : f32 = read64_0(at_23 + u32(4), scale_3);
    return vec3<f32>(_S479, _S480, _S481);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_relief(@builtin(global_invocation_id) id_34 : vec3<u32>)
{
    var k_40 : u32 = id_34.x;
    if(k_40 >= (params_0.chunk_count_0))
    {
        return;
    }
    var slot_20 : u32 = body_of_0(k_40);
    var at_24 : u32 = a_lp_0(slot_20);
    var _S482 : u32 = atomicLoad(&(atom_0[at_24 + u32(13)]));
    var supported_0 : bool = _S482 != u32(0);
    var fixed_1 : bool;
    if(((info_0[k_40] & (u32(1)))) != u32(0))
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            var _S483 : u32 = atomicLoad(&(atom_0[at_24 + u32(12)]));
            fixed_1 = _S483 == k_40;
        }
        else
        {
            fixed_1 = false;
        }
    }
    var _S484 : u32 = lp_f_0(k_40);
    var f_11 : vec4<f32> = lp_0[_S484];
    var _S485 : u32 = _S484 + u32(1);
    var t_14 : vec4<f32> = lp_0[_S485];
    var b_63 : u32 = u32(7) * slot_20;
    var inv_mass_6 : f32 = bodies_0[b_63].w;
    var _S486 : bool;
    if(!supported_0)
    {
        _S486 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S486 = false;
    }
    if(_S486)
    {
        var force_3 : vec3<f32> = read64w3_0(at_24, params_0.lp_scale_0);
        var moment_0 : vec3<f32> = read64w3_0(at_24 + u32(6), params_0.lp_scale_0);
        var _S487 : u32 = u32(2) * slot_20;
        var p_17 : Quat_0 = quat_of_0(frames_0[_S487 + u32(1)]);
        var alpha_0 : vec3<f32> = rotate_0(p_17, bodies_0[b_63 + u32(2)].xyz * rotate_0(conj_0(p_17), moment_0));
        var _S488 : u32 = u32(4) * k_40;
        var c0_4 : vec4<f32> = chunks_0[_S488];
        var c1_2 : vec4<f32> = chunks_0[_S488 + u32(1)];
        var c2_3 : vec4<f32> = chunks_0[_S488 + u32(2)];
        var _S489 : vec3<f32> = f_11.xyz - vec3<f32>(c0_4.w) * (force_3 * vec3<f32>(inv_mass_6) + cross(alpha_0, c0_4.xyz - frames_0[_S487].xyz));
        f_11.x = _S489.x;
        f_11.y = _S489.y;
        f_11.z = _S489.z;
        var _S490 : f32 = alpha_0.x;
        var _S491 : f32 = c1_2.w;
        var _S492 : f32 = alpha_0.y;
        var _S493 : f32 = c2_3.x;
        var _S494 : f32 = alpha_0.z;
        var _S495 : f32 = c2_3.y;
        var _S496 : vec3<f32> = t_14.xyz - vec3<f32>(c1_2.x * _S490 + _S491 * _S492 + _S493 * _S494, _S491 * _S490 + c1_2.y * _S492 + _S495 * _S494, _S493 * _S490 + _S495 * _S492 + c1_2.z * _S494);
        t_14.x = _S496.x;
        t_14.y = _S496.y;
        t_14.z = _S496.z;
    }
    var _S497 : f32;
    if(fixed_1)
    {
        _S497 = 1.0f;
    }
    else
    {
        _S497 = 0.0f;
    }
    f_11[i32(3)] = _S497;
    lp_0[_S484] = f_11;
    lp_0[_S485] = t_14;
    return;
}

fn adj_bond_0( at_25 : u32) -> u32
{
    return (bitcast<u32>((ext_0[params_0.ext_adj_0 + at_25 / u32(4)][at_25 % u32(4)])));
}

fn lp_intact_0( k_41 : u32) -> u32
{
    var range_0 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * k_41];
    var _S498 : u32 = (bitcast<u32>((range_0.x)));
    var _S499 : u32 = (bitcast<u32>((range_0.y)));
    var e_4 : u32 = u32(0);
    var intact_0 : u32 = u32(0);
    loop
    {
        if(e_4 < _S499)
        {
        }
        else
        {
            break;
        }
        var _S500 : u32;
        if(work_0[w_alive_0() + adj_bond_0(_S498 + e_4)] != u32(0))
        {
            _S500 = u32(1);
        }
        else
        {
            _S500 = u32(0);
        }
        var intact_1 : u32 = intact_0 + _S500;
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

fn lp_bond_0( i_13 : u32) -> LpBond_0
{
    var o_8 : u32 = u32(10) * i_13;
    var b_64 : LpBond_0;
    b_64.centroid_0 = ext_0[o_8].xyz;
    var _S501 : vec3<f32> = ext_0[o_8 + u32(1)].xyz;
    b_64.n_4 = _S501;
    var _S502 : vec3<f32> = ext_0[o_8 + u32(2)].xyz;
    b_64.t1_2 = _S502;
    b_64.t2_0 = cross(_S501, _S502);
    b_64.law_0.geom0_0 = ext_0[o_8 + u32(3)];
    b_64.law_0.geom1_0 = ext_0[o_8 + u32(4)];
    b_64.law_0.stiff0_0 = ext_0[o_8 + u32(5)];
    b_64.law_0.stiff1_0 = ext_0[o_8 + u32(6)];
    b_64.law_0.rebar0_0 = ext_0[o_8 + u32(7)];
    b_64.law_0.rebar1_0 = ext_0[o_8 + u32(8)];
    b_64.law_0.ids_3 = (bitcast<vec4<u32>>((ext_0[o_8 + u32(9)])));
    return b_64;
}

fn tri_0( i_14 : u32,  j_6 : u32) -> u32
{
    return i_14 * (i_14 + u32(1)) / u32(2) + j_6;
}

fn lp_factor_0( k_43 : u32) -> bool
{
    var q_34 : u32;
    var a_12 : u32;
    var p2_0 : u32;
    var q2_0 : u32;
    var a2_0 : u32;
    var s_13 : f32;
    var _S503 : u32 = lp_l_0(k_43);
    var _S504 : u32 = lp_intact_0(k_43) + u32(1);
    if((bitcast<u32>((lp_0[_S503 + u32(5)].w))) == _S504)
    {
        return true;
    }
    var d_23 : array<array<f32, i32(6)>, i32(6)>;
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
        q_34 = u32(0);
        loop
        {
            if(q_34 < u32(6))
            {
            }
            else
            {
                break;
            }
            d_23[p_18][q_34] = 0.0f;
            q_34 = q_34 + u32(1);
        }
        p_18 = p_18 + u32(1);
    }
    var _S505 : u32 = u32(4) * k_43;
    var range_1 : vec4<f32> = ext_0[params_0.ext_chunks_0 + _S505];
    var _S506 : u32 = (bitcast<u32>((range_1.x)));
    var _S507 : u32 = (bitcast<u32>((range_1.y)));
    q_34 = u32(0);
    loop
    {
        if(q_34 < _S507)
        {
        }
        else
        {
            break;
        }
        var i_15 : u32 = adj_bond_0(_S506 + q_34);
        if(work_0[w_alive_0() + i_15] == u32(0))
        {
            q_34 = q_34 + u32(1);
            continue;
        }
        var b_65 : LpBond_0 = lp_bond_0(i_15);
        var _S508 : vec3<f32> = b_65.centroid_0 - chunks_0[_S505].xyz;
        var _S509 : f32 = b_65.law_0.stiff0_0.y;
        var _S510 : array<f32, i32(6)> = array<f32, i32(6)>( _S509, _S509, b_65.law_0.stiff0_0.x, b_65.law_0.stiff0_0.z, b_65.law_0.stiff0_0.w, b_65.law_0.stiff1_0.x );
        var _S511 : array<vec3<f32>, i32(3)> = array<vec3<f32>, i32(3)>( b_65.t1_2, b_65.t2_0, b_65.n_4 );
        var j_7 : array<array<f32, i32(6)>, i32(6)>;
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
            var c_20 : vec3<f32> = cross(_S508, _S511[a_12]);
            var _S512 : f32 = _S511[a_12].x;
            j_7[a_12][i32(0)] = _S512;
            var _S513 : f32 = _S511[a_12].y;
            j_7[a_12][i32(1)] = _S513;
            var _S514 : f32 = _S511[a_12].z;
            j_7[a_12][i32(2)] = _S514;
            j_7[a_12][i32(3)] = c_20.x;
            j_7[a_12][i32(4)] = c_20.y;
            j_7[a_12][i32(5)] = c_20.z;
            j_7[u32(3) + a_12][i32(0)] = 0.0f;
            j_7[u32(3) + a_12][i32(1)] = 0.0f;
            j_7[u32(3) + a_12][i32(2)] = 0.0f;
            j_7[u32(3) + a_12][i32(3)] = _S512;
            j_7[u32(3) + a_12][i32(4)] = _S513;
            j_7[u32(3) + a_12][i32(5)] = _S514;
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
                    var s_14 : f32 = s_13 + j_7[a2_0][p2_0] * _S510[a2_0] * j_7[a2_0][q2_0];
                    a2_0 = a2_0 + u32(1);
                    s_13 = s_14;
                }
                d_23[p2_0][q2_0] = d_23[p2_0][q2_0] + s_13;
                q2_0 = q2_0 + u32(1);
            }
            p2_0 = p2_0 + u32(1);
        }
        q_34 = q_34 + u32(1);
    }
    var l_1 : array<f32, i32(21)>;
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
        l_1[a_12] = 0.0f;
        a_12 = a_12 + u32(1);
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
                var _S515 : bool;
                if(ok_1)
                {
                    _S515 = s_13 > 0.0f;
                }
                else
                {
                    _S515 = false;
                }
                l_1[tri_0(p2_0, p2_0)] = sqrt(max(s_13, 1.00000000317107685e-30f));
                ok_1 = _S515;
            }
            else
            {
                l_1[tri_0(p2_0, q2_0)] = s_13 / l_1[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + u32(1);
        }
        var _S516 : u32 = p2_0 + u32(1);
        ok_0 = ok_1;
        p2_0 = _S516;
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
        var _S517 : u32 = u32(4) * q2_0;
        var _S518 : f32 = l_1[_S517];
        var _S519 : u32 = _S517 + u32(1);
        if(_S519 < u32(21))
        {
            s_13 = l_1[_S519];
        }
        else
        {
            s_13 = 0.0f;
        }
        var _S520 : u32 = _S517 + u32(2);
        var _S521 : f32;
        if(_S520 < u32(21))
        {
            _S521 = l_1[_S520];
        }
        else
        {
            _S521 = 0.0f;
        }
        var _S522 : f32;
        if(q2_0 == u32(5))
        {
            _S522 = (bitcast<f32>((_S504)));
        }
        else
        {
            _S522 = l_1[_S517 + u32(3)];
        }
        lp_0[_S503 + q2_0] = vec4<f32>(_S518, s_13, _S521, _S522);
        q2_0 = q2_0 + u32(1);
    }
    return true;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_block(@builtin(global_invocation_id) id_35 : vec3<u32>)
{
    var k_44 : u32 = id_35.x;
    var _S523 : bool;
    if(k_44 >= (params_0.chunk_count_0))
    {
        _S523 = true;
    }
    else
    {
        _S523 = (lp_0[lp_f_0(k_44)].w) != 0.0f;
    }
    if(_S523)
    {
        return;
    }
    var _S524 : bool = lp_factor_0(k_44);
    if(!_S524)
    {
        lp_0[lp_f_0(k_44)][i32(3)] = 1.0f;
        var _S525 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(64));
    }
    return;
}

fn st_x_0( k_45 : u32,  state_1 : u32,  copy_0 : u32) -> u32
{
    return u32(22) * k_45 + state_1 + u32(2) * copy_0;
}

fn st_d_0( k_46 : u32,  state_2 : u32) -> u32
{
    return u32(22) * k_46 + state_2 + u32(4);
}

fn lp_side_0( i_16 : u32) -> vec4<f32>
{
    return ext_0[params_0.ext_sides_0 + i_16];
}

fn st_sweep_0( k_47 : u32,  state_3 : u32,  from_2 : u32,  c1_3 : f32,  c2_4 : f32)
{
    var ka_0 : u32;
    var kb_0 : u32;
    var s_16 : f32;
    var to_1 : u32 = u32(1) - from_2;
    var _S526 : u32 = st_x_0(k_47, state_3, from_2);
    var x0_0 : vec4<f32> = lp_0[_S526];
    var x1_0 : vec4<f32> = lp_0[_S526 + u32(1)];
    var _S527 : u32 = st_f_0(k_47, state_3);
    var f_12 : vec4<f32> = lp_0[_S527];
    if((lp_0[_S527].w) != 0.0f)
    {
        var _S528 : u32 = st_x_0(k_47, state_3, to_1);
        lp_0[_S528] = x0_0;
        lp_0[_S528 + u32(1)] = x1_0;
        var _S529 : u32 = st_d_0(k_47, state_3);
        var _S530 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S529] = _S530;
        lp_0[_S529 + u32(1)] = _S530;
        return;
    }
    var _S531 : f32 = f_12.y;
    var _S532 : f32 = f_12.z;
    var r_23 : array<f32, i32(6)>;
    r_23[i32(0)] = f_12.x;
    r_23[i32(1)] = _S531;
    r_23[i32(2)] = _S532;
    r_23[i32(3)] = 0.0f;
    r_23[i32(4)] = 0.0f;
    r_23[i32(5)] = 0.0f;
    var t_15 : vec3<f32> = lp_0[_S527 + u32(1)].xyz;
    r_23[i32(3)] = t_15.x;
    r_23[i32(4)] = t_15.y;
    r_23[i32(5)] = t_15.z;
    var range_2 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * k_47];
    var _S533 : u32 = (bitcast<u32>((range_2.x)));
    var _S534 : u32 = (bitcast<u32>((range_2.y)));
    var e_5 : u32 = u32(0);
    loop
    {
        if(e_5 < _S534)
        {
        }
        else
        {
            break;
        }
        var _S535 : u32 = u32(5) * (_S533 + e_5);
        var s0_4 : vec4<f32> = lp_side_0(_S535);
        if(work_0[w_alive_0() + (bitcast<u32>((s0_4.w)))] == u32(0))
        {
            e_5 = e_5 + u32(1);
            continue;
        }
        var s1_5 : vec4<f32> = lp_side_0(_S535 + u32(1));
        var s2_0 : vec4<f32> = lp_side_0(_S535 + u32(2));
        var s3_0 : vec4<f32> = lp_side_0(_S535 + u32(3));
        var s4_0 : vec4<f32> = lp_side_0(_S535 + u32(4));
        var first_side_0 : bool = (s4_0.w) != 0.0f;
        if(first_side_0)
        {
            ka_0 = k_47;
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
            kb_0 = k_47;
        }
        var ra_1 : vec3<f32> = s0_4.xyz;
        var rb_1 : vec3<f32> = s1_5.xyz;
        var t1_3 : vec3<f32> = s2_0.xyz;
        var n_5 : vec3<f32> = s3_0.xyz;
        var t2_1 : vec3<f32> = cross(n_5, t1_3);
        var _S536 : u32 = st_x_0(ka_0, state_3, from_2);
        var _S537 : u32 = st_x_0(kb_0, state_3, from_2);
        var _S538 : vec3<f32> = lp_0[_S537 + u32(1)].xyz;
        var _S539 : vec3<f32> = lp_0[_S536 + u32(1)].xyz;
        var delta_0 : vec3<f32> = lp_0[_S537].xyz + cross(_S538, rb_1) - (lp_0[_S536].xyz + cross(_S539, ra_1));
        var phi_0 : vec3<f32> = _S538 - _S539;
        var _S540 : f32 = s2_0.w;
        var fb_0 : vec3<f32> = t1_3 * vec3<f32>((_S540 * dot(delta_0, t1_3))) + t2_1 * vec3<f32>((_S540 * dot(delta_0, t2_1))) + n_5 * vec3<f32>((s3_0.w * dot(delta_0, n_5)));
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
        var _S541 : vec3<f32>;
        if(first_side_0)
        {
            _S541 = mb_0;
        }
        else
        {
            _S541 = (vec3<f32>(0) - mb_0);
        }
        var lm_0 : vec3<f32> = _S541 + cross(arm_0, lf_0);
        r_23[i32(0)] = r_23[i32(0)] + lf_0.x;
        r_23[i32(1)] = r_23[i32(1)] + lf_0.y;
        r_23[i32(2)] = r_23[i32(2)] + lf_0.z;
        r_23[i32(3)] = r_23[i32(3)] + lm_0.x;
        r_23[i32(4)] = r_23[i32(4)] + lm_0.y;
        r_23[i32(5)] = r_23[i32(5)] + lm_0.z;
        e_5 = e_5 + u32(1);
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
        var v_14 : vec4<f32> = lp_0[lp_l_0(k_47) + kb_0];
        var _S542 : u32 = u32(4) * kb_0;
        l_2[_S542] = lp_0[lp_l_0(k_47) + kb_0].x;
        var _S543 : u32 = _S542 + u32(1);
        if(_S543 < u32(21))
        {
            l_2[_S543] = v_14.y;
        }
        var _S544 : u32 = _S542 + u32(2);
        if(_S544 < u32(21))
        {
            l_2[_S544] = v_14.z;
        }
        var _S545 : u32 = _S542 + u32(3);
        if(_S545 < u32(21))
        {
            l_2[_S545] = v_14.w;
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
        var p_19 : u32 = u32(0);
        s_16 = r_23[i2_0];
        loop
        {
            if(p_19 < i2_0)
            {
            }
            else
            {
                break;
            }
            var s_17 : f32 = s_16 - l_2[tri_0(i2_0, p_19)] * y_3[p_19];
            p_19 = p_19 + u32(1);
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
        var _S546 : u32 = u32(i3_0);
        z_4[i3_0] = s_16 / l_2[tri_0(_S546, _S546)];
        i3_0 = i3_0 - i32(1);
    }
    var _S547 : u32 = st_d_0(k_47, state_3);
    var d0_0 : vec4<f32> = lp_0[_S547];
    var _S548 : u32 = _S547 + u32(1);
    var d1_0 : vec4<f32> = lp_0[_S548];
    var _S549 : vec3<f32> = vec3<f32>(c1_3);
    var _S550 : vec3<f32> = vec3<f32>(c2_4);
    var _S551 : vec3<f32> = _S549 * lp_0[_S547].xyz + _S550 * vec3<f32>(z_4[i32(0)], z_4[i32(1)], z_4[i32(2)]);
    d0_0.x = _S551.x;
    d0_0.y = _S551.y;
    d0_0.z = _S551.z;
    var _S552 : vec3<f32> = _S549 * lp_0[_S548].xyz + _S550 * vec3<f32>(z_4[i32(3)], z_4[i32(4)], z_4[i32(5)]);
    d1_0.x = _S552.x;
    d1_0.y = _S552.y;
    d1_0.z = _S552.z;
    lp_0[_S547] = d0_0;
    lp_0[_S548] = d1_0;
    var _S553 : u32 = st_x_0(k_47, state_3, to_1);
    lp_0[_S553] = vec4<f32>(x0_0.xyz + d0_0.xyz, 0.0f);
    lp_0[_S553 + u32(1)] = vec4<f32>(x1_0.xyz + d1_0.xyz, 0.0f);
    return;
}

fn lp_sweep_0( k_48 : u32,  from_3 : u32)
{
    st_sweep_0(k_48, u32(0), from_3, params_0.lp_c1_0, params_0.lp_c2_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sweep_even(@builtin(global_invocation_id) id_36 : vec3<u32>)
{
    var _S554 : u32 = id_36.x;
    if(_S554 < (params_0.chunk_count_0))
    {
        lp_sweep_0(_S554, u32(0));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_sweep_odd(@builtin(global_invocation_id) id_37 : vec3<u32>)
{
    var _S555 : u32 = id_37.x;
    if(_S555 < (params_0.chunk_count_0))
    {
        lp_sweep_0(_S555, u32(1));
    }
    return;
}

fn st_force_0( b_66 : LpBond_0,  ka_1 : u32,  kb_1 : u32,  state_4 : u32,  c_21 : u32,  q_lin_0 : ptr<function, vec3<f32>>,  q_ang_0 : ptr<function, vec3<f32>>)
{
    var _S556 : u32 = st_x_0(ka_1, state_4, c_21);
    var _S557 : u32 = st_x_0(kb_1, state_4, c_21);
    var _S558 : vec3<f32> = lp_0[_S557 + u32(1)].xyz;
    var _S559 : vec3<f32> = lp_0[_S556 + u32(1)].xyz;
    var delta_1 : vec3<f32> = lp_0[_S557].xyz + cross(_S558, b_66.centroid_0 - chunks_0[u32(4) * kb_1].xyz) - (lp_0[_S556].xyz + cross(_S559, b_66.centroid_0 - chunks_0[u32(4) * ka_1].xyz));
    var phi_1 : vec3<f32> = _S558 - _S559;
    var kt_0 : f32 = b_66.law_0.stiff1_0.x;
    var _S560 : f32 = b_66.law_0.stiff0_0.y;
    (*q_lin_0) = vec3<f32>(_S560 * dot(delta_1, b_66.t1_2), _S560 * dot(delta_1, b_66.t2_0), b_66.law_0.stiff0_0.x * dot(delta_1, b_66.n_4));
    (*q_ang_0) = vec3<f32>(b_66.law_0.stiff0_0.z * dot(phi_1, b_66.t1_2), b_66.law_0.stiff0_0.w * dot(phi_1, b_66.t2_0), kt_0 * dot(phi_1, b_66.n_4));
    return;
}

fn st_force_1( b_67 : LpBond_0,  ka_2 : u32,  kb_2 : u32,  state_5 : u32,  c_22 : u32,  q_lin_1 : ptr<function, vec3<f32>>,  q_ang_1 : ptr<function, vec3<f32>>)
{
    var _S561 : u32 = st_x_0(ka_2, state_5, c_22);
    var _S562 : u32 = st_x_0(kb_2, state_5, c_22);
    var _S563 : vec3<f32> = lp_0[_S562 + u32(1)].xyz;
    var _S564 : vec3<f32> = lp_0[_S561 + u32(1)].xyz;
    var delta_2 : vec3<f32> = lp_0[_S562].xyz + cross(_S563, b_67.centroid_0 - chunks_0[u32(4) * kb_2].xyz) - (lp_0[_S561].xyz + cross(_S564, b_67.centroid_0 - chunks_0[u32(4) * ka_2].xyz));
    var phi_2 : vec3<f32> = _S563 - _S564;
    var kt_1 : f32 = b_67.law_0.stiff1_0.x;
    var _S565 : f32 = b_67.law_0.stiff0_0.y;
    (*q_lin_1) = vec3<f32>(_S565 * dot(delta_2, b_67.t1_2), _S565 * dot(delta_2, b_67.t2_0), b_67.law_0.stiff0_0.x * dot(delta_2, b_67.n_4));
    (*q_ang_1) = vec3<f32>(b_67.law_0.stiff0_0.z * dot(phi_2, b_67.t1_2), b_67.law_0.stiff0_0.w * dot(phi_2, b_67.t2_0), kt_1 * dot(phi_2, b_67.n_4));
    return;
}

fn st_force_2( b_68 : LpBond_0,  ka_3 : u32,  kb_3 : u32,  state_6 : u32,  c_23 : u32,  q_lin_2 : ptr<function, vec3<f32>>,  q_ang_2 : ptr<function, vec3<f32>>)
{
    var _S566 : u32 = st_x_0(ka_3, state_6, c_23);
    var _S567 : u32 = st_x_0(kb_3, state_6, c_23);
    var _S568 : vec3<f32> = lp_0[_S567 + u32(1)].xyz;
    var _S569 : vec3<f32> = lp_0[_S566 + u32(1)].xyz;
    var delta_3 : vec3<f32> = lp_0[_S567].xyz + cross(_S568, b_68.centroid_0 - chunks_0[u32(4) * kb_3].xyz) - (lp_0[_S566].xyz + cross(_S569, b_68.centroid_0 - chunks_0[u32(4) * ka_3].xyz));
    var phi_3 : vec3<f32> = _S568 - _S569;
    var kt_2 : f32 = b_68.law_0.stiff1_0.x;
    var _S570 : f32 = b_68.law_0.stiff0_0.y;
    (*q_lin_2) = vec3<f32>(_S570 * dot(delta_3, b_68.t1_2), _S570 * dot(delta_3, b_68.t2_0), b_68.law_0.stiff0_0.x * dot(delta_3, b_68.n_4));
    (*q_ang_2) = vec3<f32>(b_68.law_0.stiff0_0.z * dot(phi_3, b_68.t1_2), b_68.law_0.stiff0_0.w * dot(phi_3, b_68.t2_0), kt_2 * dot(phi_3, b_68.n_4));
    return;
}

fn lp_force_0( b_69 : LpBond_0,  ka_4 : u32,  kb_4 : u32,  c_24 : u32,  q_lin_3 : ptr<function, vec3<f32>>,  q_ang_3 : ptr<function, vec3<f32>>)
{
    st_force_1(b_69, ka_4, kb_4, u32(0), c_24, &((*q_lin_3)), &((*q_ang_3)));
    return;
}

fn lp_force_1( b_70 : LpBond_0,  ka_5 : u32,  kb_5 : u32,  c_25 : u32,  q_lin_4 : ptr<function, vec3<f32>>,  q_ang_4 : ptr<function, vec3<f32>>)
{
    st_force_2(b_70, ka_5, kb_5, u32(0), c_25, &((*q_lin_4)), &((*q_ang_4)));
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

fn stress_measures_0( b_71 : JointBond_0,  q_lin_5 : vec3<f32>,  q_ang_5 : vec3<f32>) -> Measures_0
{
    var area_0 : f32 = b_71.geom0_0.x;
    var _S571 : f32 = q_lin_5.z;
    var axial_0 : f32 = fdiv_0(_S571, area_0);
    var bending_0 : f32 = fdiv_0(abs(q_ang_5.x), b_71.geom1_0.x) + fdiv_0(abs(q_ang_5.y), b_71.geom1_0.y);
    var _S572 : f32 = q_lin_5.x;
    var _S573 : f32 = q_lin_5.y;
    var shear_1 : f32 = fdiv_0(fsqrt_0(_S572 * _S572 + _S573 * _S573), area_0) + fdiv_0(abs(q_ang_5.z), b_71.geom0_0.w);
    var m_3 : Measures_0;
    m_3.tension_0 = axial_0 + bending_0;
    m_3.shear_0 = shear_1;
    var _S574 : f32 = - axial_0;
    m_3.normal_compression_0 = max(_S574, 0.0f);
    m_3.compression_0 = _S574 + bending_0;
    m_3.compressive_force_0 = max(- _S571, 0.0f);
    return m_3;
}

fn lp_material_0( m_4 : u32) -> JointMaterial_0
{
    var o_9 : u32 = params_0.ext_mat_0 + u32(5) * m_4;
    var j_8 : JointMaterial_0;
    j_8.strength_0 = ext_0[o_9];
    j_8.energy_0 = ext_0[o_9 + u32(1)];
    j_8.dif_0 = ext_0[o_9 + u32(2)];
    j_8.misc_0 = ext_0[o_9 + u32(3)];
    j_8.kind_flags_0 = (bitcast<vec4<u32>>((ext_0[o_9 + u32(4)])));
    return j_8;
}

fn infinity_0() -> f32
{
    return float_bits_0(u32(2139095040));
}

fn failure_indices_0( mat_1 : JointMaterial_0,  b_72 : JointBond_0,  m_5 : Measures_0,  multiplier_0 : f32) -> vec4<f32>
{
    var fc_2 : f32 = mat_1.strength_0.y * multiplier_0;
    var _S575 : f32 = min(mat_1.strength_0.z * multiplier_0 + mat_1.strength_0.w * m_5.normal_compression_0, mat_1.energy_0.x * multiplier_0);
    var idx_0 : vec4<f32>;
    idx_0[i32(0)] = max(fdiv_0(m_5.tension_0, mat_1.strength_0.x * multiplier_0), 0.0f);
    var _S576 : f32;
    if(_S575 > 0.0f)
    {
        _S576 = fdiv_0(m_5.shear_0, _S575);
    }
    else
    {
        _S576 = infinity_0();
    }
    idx_0[i32(1)] = _S576;
    idx_0[i32(2)] = max(fdiv_0(m_5.compression_0, fc_2), 0.0f);
    var _S577 : f32 = b_72.stiff1_0.y;
    if(_S577 > 0.0f)
    {
        _S576 = fdiv_0(m_5.compressive_force_0, _S577);
    }
    else
    {
        _S576 = 0.0f;
    }
    idx_0[i32(3)] = _S576;
    return idx_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_lp_breaks(@builtin(global_invocation_id) id_38 : vec3<u32>)
{
    var i_17 : u32 = id_38.x;
    var _S578 : bool;
    if(i_17 >= (params_0.bond_count_0))
    {
        _S578 = true;
    }
    else
    {
        _S578 = work_0[w_alive_0() + i_17] == u32(0);
    }
    if(_S578)
    {
        return;
    }
    var bond_7 : vec4<u32> = bonds_0[i_17];
    var b_73 : LpBond_0 = lp_bond_0(i_17);
    var _S579 : u32 = bond_7.x;
    var q_lin_6 : vec3<f32>;
    var q_ang_6 : vec3<f32>;
    lp_force_1(b_73, _S579, bond_7.y, u32(0), &(q_lin_6), &(q_ang_6));
    var idx_1 : vec4<f32> = failure_indices_0(lp_material_0(b_73.law_0.ids_3.x), b_73.law_0, stress_measures_0(b_73.law_0, q_lin_6, q_ang_6), b_73.law_0.geom1_0.w);
    var _S580 : f32 = max(max(idx_1.x, idx_1.y), max(idx_1.z, idx_1.w));
    var _S581 : u32 = a_counters_0();
    var _S582 : u32 = atomicMax(&(atom_0[_S581 + u32(15)]), (bitcast<u32>((max(_S580, 0.0f)))));
    if(_S580 >= 1.0f)
    {
        work_0[w_alive_0() + i_17] = u32(0);
        var _S583 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S579)]), u32(1));
        var _S584 : u32 = atomicAdd(&(atom_0[_S581]), u32(1));
        var _S585 : u32 = atomicAdd(&(atom_0[_S581 + u32(1)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_first_reset(@builtin(global_invocation_id) id_39 : vec3<u32>)
{
    var _S586 : u32 = id_39.x;
    if(_S586 == u32(0))
    {
        atomicStore(&(atom_0[a_il_cursor_0()]), u32(0));
    }
    if(_S586 < (params_0.shot_count_0))
    {
        atomicStore(&(atom_0[a_first_0(_S586)]), u32(2139095040));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_first(@builtin(global_invocation_id) id_40 : vec3<u32>)
{
    var b_74 : u32 = id_40.x;
    var _S587 : bool;
    if(b_74 >= (params_0.body_count_0))
    {
        _S587 = true;
    }
    else
    {
        _S587 = (params_0.shot_count_0) == u32(0);
    }
    if(_S587)
    {
        return;
    }
    var shot_10 : u32;
    var _S588 : bool = struck_2(b_74, &(shot_10));
    if(!_S588)
    {
        return;
    }
    var _S589 : u32 = atomicLoad(&(atom_0[a_strike_0() + u32(9) * b_74 + u32(6)]));
    var _S590 : u32 = atomicMin(&(atom_0[a_first_0(shot_10)]), _S589);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_begin(@builtin(global_invocation_id) id_41 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_begin_1(id_41.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_tube_count(@builtin(global_invocation_id) id_42 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_tube_count_1(id_42.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_tube_base(@builtin(global_invocation_id) id_43 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_tube_base_1(id_43.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_tube_collect(@builtin(global_invocation_id) id_44 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_tube_collect_1(id_44.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_run(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_run_1(group_1.x, thread_4.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_strike(@builtin(global_invocation_id) id_45 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_strike_1(id_45.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_mark(@builtin(global_invocation_id) id_46 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_mark_1(id_46.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_carry(@builtin(global_invocation_id) id_47 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_carry_1(id_47.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_il_breaks(@builtin(global_invocation_id) id_48 : vec3<u32>)
{
    if((params_0.shot_count_0) > u32(0))
    {
        il_breaks_1(id_48.x);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_begin(@builtin(global_invocation_id) id_49 : vec3<u32>)
{
    il_begin_0(id_49.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_tube_count(@builtin(global_invocation_id) id_50 : vec3<u32>)
{
    il_tube_count_0(id_50.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_tube_base(@builtin(global_invocation_id) id_51 : vec3<u32>)
{
    il_tube_base_0(id_51.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_tube_collect(@builtin(global_invocation_id) id_52 : vec3<u32>)
{
    il_tube_collect_0(id_52.x);
    return;
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
fn fr_csl_run(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    il_run_0(work_0[w_cs_bodies_0() + group_2.x], thread_5.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_strike(@builtin(global_invocation_id) id_53 : vec3<u32>)
{
    il_strike_0(id_53.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_mark(@builtin(global_invocation_id) id_54 : vec3<u32>)
{
    il_mark_0(id_54.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_carry(@builtin(global_invocation_id) id_55 : vec3<u32>)
{
    il_carry_0(id_55.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_csl_breaks(@builtin(global_invocation_id) id_56 : vec3<u32>)
{
    il_breaks_0(id_56.x);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_cs_begin(@builtin(global_invocation_id) id_57 : vec3<u32>)
{
    var _S591 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S591 + u32(18)]), u32(0));
    atomicStore(&(atom_0[_S591 + u32(19)]), u32(0));
    atomicStore(&(atom_0[a_il_cursor_0()]), u32(0));
    var _S592 : u32 = groups_of_0(cs_rows_0(), u32(64));
    set_args_0(u32(7), _S592);
    return;
}

fn cs_side_1( r_24 : u32,  side_6 : u32,  k_49 : ptr<function, u32>,  s_19 : ptr<function, u32>,  ob_5 : ptr<function, u32>,  co_5 : ptr<function, u32>,  p_20 : ptr<function, vec3<f32>>,  u_9 : ptr<function, vec3<f32>>,  v_15 : ptr<function, f32>,  m_eff_7 : ptr<function, f32>,  t_in_10 : ptr<function, f32>) -> bool
{
    (*t_in_10) = 0.0f;
    (*k_49) = u32(4294967295);
    (*s_19) = u32(4294967295);
    (*ob_5) = u32(4294967295);
    (*co_5) = u32(4294967295);
    var _S593 : vec3<f32> = vec3<f32>(0.0f);
    (*p_20) = _S593;
    (*u_9) = _S593;
    (*v_15) = 0.0f;
    (*m_eff_7) = 0.0f;
    var row_2 : u32 = cs_row_0(r_24);
    var ids_4 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_2])));
    if((ids_4.w) == u32(0))
    {
        return false;
    }
    var _S594 : bool = side_6 == u32(0);
    var cm_1 : u32;
    if(_S594)
    {
        cm_1 = ids_4.x;
    }
    else
    {
        cm_1 = ids_4.y;
    }
    var _S595 : u32;
    if(_S594)
    {
        _S595 = ids_4.y;
    }
    else
    {
        _S595 = ids_4.x;
    }
    (*co_5) = _S595;
    var _S596 : u32 = collider_info_0[u32(2) * cm_1].x;
    (*k_49) = _S596;
    var _S597 : bool;
    if(_S596 == u32(4294967295))
    {
        _S597 = true;
    }
    else
    {
        _S597 = (*k_49) >= (params_0.chunk_count_0);
    }
    if(_S597)
    {
        _S597 = true;
    }
    else
    {
        _S597 = (collider_info_0[u32(2) * (*co_5)].z) != u32(2);
    }
    if(_S597)
    {
        return false;
    }
    (*s_19) = body_of_0((*k_49));
    var _S598 : u32 = (bitcast<u32>((colliders_0[u32(4) * (*co_5)].x)));
    (*ob_5) = _S598;
    if(_S598 == (*s_19))
    {
        _S597 = true;
    }
    else
    {
        _S597 = (*ob_5) >= (params_0.max_bodies_0);
    }
    if(_S597)
    {
        return false;
    }
    var launch_2 : u32 = collider_info_0[u32(2) * (*co_5) + u32(1)].w;
    if(launch_2 != u32(0))
    {
        _S597 = work_0[w_marks_0() + (*k_49)] == launch_2;
    }
    else
    {
        _S597 = false;
    }
    if(_S597)
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
    var _S599 : vec3<f32>;
    if(_S594)
    {
        _S599 = (vec3<f32>(0) - n_6);
    }
    else
    {
        _S599 = n_6;
    }
    (*u_9) = _S599;
    var _S600 : Quat_0 = quat_of_0(bodies_0[bs_1 + u32(1)]);
    if(_S594)
    {
        _S599 = engine_contacts_0[row_2 + u32(2)].xyz;
    }
    else
    {
        _S599 = engine_contacts_0[row_2 + u32(3)].xyz;
    }
    var rs_1 : vec3<f32> = rotate_0(_S600, _S599);
    var _S601 : Quat_0 = quat_of_0(bodies_0[bo_1 + u32(1)]);
    if(_S594)
    {
        _S599 = engine_contacts_0[row_2 + u32(3)].xyz;
    }
    else
    {
        _S599 = engine_contacts_0[row_2 + u32(2)].xyz;
    }
    var ro_1 : vec3<f32> = rotate_0(_S601, _S599);
    var _S602 : vec3<f32> = bodies_0[bs_1].xyz + rs_1;
    (*p_20) = _S602;
    var gap_1 : f32 = dot(_S602 - (bodies_0[bo_1].xyz + ro_1), (*u_9));
    var _S603 : f32 = dot(bodies_0[bo_1 + u32(3)].xyz + cross(bodies_0[bo_1 + u32(4)].xyz, ro_1) - (bodies_0[bs_1 + u32(3)].xyz + cross(bodies_0[bs_1 + u32(4)].xyz, rs_1)), (*u_9));
    (*v_15) = _S603;
    if(!(_S603 > (params_0.strike_speed_0)))
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

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_contacts(@builtin(global_invocation_id) id_58 : vec3<u32>)
{
    var r_25 : u32 = id_58.x;
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
        var k_50 : u32;
        var s_20 : u32;
        var ob_6 : u32;
        var co_6 : u32;
        var p_21 : vec3<f32>;
        var u_10 : vec3<f32>;
        var v_16 : f32;
        var m_eff_8 : f32;
        var t_in_11 : f32;
        var _S604 : bool = cs_side_1(r_25, side_7, &(k_50), &(s_20), &(ob_6), &(co_6), &(p_21), &(u_10), &(v_16), &(m_eff_8), &(t_in_11));
        if(_S604)
        {
            var _S605 : u32 = atomicMax(&(atom_0[a_cs_0(s_20)]), (bitcast<u32>((m_eff_8 * v_16))));
            var _S606 : u32 = atomicExchange(&(atom_0[a_cs_0(s_20) + u32(8)]), u32(1));
            if(_S606 == u32(0))
            {
                var _S607 : u32 = w_cs_bodies_0();
                var _S608 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(18)]), u32(1));
                work_0[_S607 + _S608] = s_20;
            }
        }
        side_7 = side_7 + u32(1);
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_cs_gate(@builtin(global_invocation_id) id_59 : vec3<u32>)
{
    var struck_3 : u32 = atomicLoad(&(atom_0[a_counters_0() + u32(18)]));
    var any_0 : bool = struck_3 > u32(0);
    var _S609 : u32;
    if(any_0)
    {
        var _S610 : u32 = groups_of_0(cs_rows_0(), u32(64));
        _S609 = _S610;
    }
    else
    {
        _S609 = u32(0);
    }
    set_args_0(u32(8), _S609);
    var _S611 : u32 = groups_of_0(struck_3, u32(64));
    set_args_0(u32(9), _S611);
    set_args_0(u32(6), struck_3);
    if(any_0)
    {
        var _S612 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S609 = _S612;
    }
    else
    {
        _S609 = u32(0);
    }
    set_args_0(u32(10), _S609);
    if(any_0)
    {
        var _S613 : u32 = groups_of_0(params_0.bond_count_0, u32(64));
        _S609 = _S613;
    }
    else
    {
        _S609 = u32(0);
    }
    set_args_0(u32(11), _S609);
    if(any_0)
    {
        var _S614 : u32 = groups_of_0(params_0.body_count_0, u32(64));
        _S609 = _S614;
    }
    else
    {
        _S609 = u32(0);
    }
    set_args_0(u32(12), _S609);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_a(@builtin(global_invocation_id) id_60 : vec3<u32>)
{
    cs_pick_3(id_60.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_b(@builtin(global_invocation_id) id_61 : vec3<u32>)
{
    cs_pick_2(id_61.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_c(@builtin(global_invocation_id) id_62 : vec3<u32>)
{
    cs_pick_1(id_62.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_pick_d(@builtin(global_invocation_id) id_63 : vec3<u32>)
{
    cs_pick_0(id_63.x);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_record(@builtin(global_invocation_id) id_64 : vec3<u32>)
{
    var _S615 : u32 = id_64.x;
    var _S616 : u32 = a_counters_0();
    var _S617 : u32 = atomicLoad(&(atom_0[_S616 + u32(18)]));
    if(_S615 >= _S617)
    {
        return;
    }
    var s_21 : u32 = work_0[w_cs_bodies_0() + _S615];
    var _S618 : u32 = a_cs_0(work_0[w_cs_bodies_0() + _S615]);
    var _S619 : u32 = atomicLoad(&(atom_0[_S618 + u32(11)]));
    var key_2 : u32 = _S619 - u32(1);
    var k_51 : u32;
    var s2_1 : u32;
    var ob_7 : u32;
    var co_7 : u32;
    var p_22 : vec3<f32>;
    var u_11 : vec3<f32>;
    var v_17 : f32;
    var m_eff_9 : f32;
    var t_in_12 : f32;
    var _S620 : bool = cs_side_1(key_2 / u32(2), key_2 % u32(2), &(k_51), &(s2_1), &(ob_7), &(co_7), &(p_22), &(u_11), &(v_17), &(m_eff_9), &(t_in_12));
    if(!_S620)
    {
        return;
    }
    var other_1 : vec4<u32> = collider_info_0[u32(2) * co_7 + u32(1)];
    var e_o_0 : f32 = (bitcast<f32>((collider_info_0[u32(2) * co_7 + u32(1)].y)));
    var _S621 : f32 = max((bitcast<f32>((collider_info_0[u32(2) * co_7 + u32(1)].z))), 0.00100000004749745f);
    var j_9 : f32 = m_eff_9 * v_17;
    var tau_2 : f32 = hertz_duration_0(m_eff_9, _S621, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / geometry_0[u32(3) * k_51 + u32(1)].w), v_17);
    var _S622 : u32 = lp_body_0(s_21);
    lp_0[_S622 + u32(2)] = vec4<f32>(from_rubble_0(p_22), 3.14159274101257324f * j_9 / (2.0f * tau_2));
    lp_0[_S622 + u32(3)] = vec4<f32>(from_rubble_0(u_11), v_17);
    lp_0[_S622 + u32(4)] = vec4<f32>(m_eff_9, _S621, e_o_0, (bitcast<f32>((ob_7))));
    lp_0[_S622 + u32(5)] = vec4<f32>((bitcast<f32>((k_51))), j_9 / tau_2, 1.0f / bodies_0[u32(7) * ob_7].w, t_in_12);
    lp_0[_S622 + u32(6)] = vec4<f32>((bitcast<f32>((co_7))), (bitcast<f32>((other_1.w))), 0.0f, 0.0f);
    atomicStore(&(atom_0[_S618 + u32(1)]), k_51 + u32(1));
    var _S623 : u32 = atomicAdd(&(atom_0[_S616 + u32(16)]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_hertz(@builtin(global_invocation_id) id_65 : vec3<u32>)
{
    var i_18 : u32 = id_65.x;
    var _S624 : bool;
    if(i_18 >= (params_0.bond_count_0))
    {
        _S624 = true;
    }
    else
    {
        _S624 = work_0[w_alive_0() + i_18] == u32(0);
    }
    if(_S624)
    {
        return;
    }
    var bond_8 : vec4<u32> = bonds_0[i_18];
    var b_75 : u32 = body_of_0(bond_8.x);
    var _S625 : bool = cs_struck_0(b_75);
    if(!_S625)
    {
        return;
    }
    var _S626 : u32 = lp_body_0(b_75);
    if(strike_share_breaks_0(bond_8, lp_0[_S626 + u32(2)].xyz, lp_0[_S626 + u32(5)].y))
    {
        work_0[w_alive_0() + i_18] = u32(0);
        var _S627 : u32 = atomicOr(&(atom_0[a_dirty_0() + b_75]), u32(1));
        var _S628 : u32 = a_counters_0();
        var _S629 : u32 = atomicAdd(&(atom_0[_S628]), u32(1));
        var _S630 : u32 = atomicAdd(&(atom_0[_S628 + u32(1)]), u32(1));
        var _S631 : u32 = atomicAdd(&(atom_0[_S628 + u32(17)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_cs_gate2(@builtin(global_invocation_id) id_66 : vec3<u32>)
{
    var _S632 : u32 = atomicLoad(&(atom_0[a_counters_0()]));
    var _S633 : u32;
    if(_S632 > u32(0))
    {
        var _S634 : u32 = groups_of_0(cs_rows_0(), u32(64));
        _S633 = _S634;
    }
    else
    {
        _S633 = u32(0);
    }
    set_args_0(u32(13), _S633);
    return;
}

fn lp_anchor_0( r_26 : u32,  side_8 : u32) -> u32
{
    return u32(22) * params_0.chunk_count_0 + u32(7) * params_0.body_count_0 + u32(2) * r_26 + side_8;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_save(@builtin(global_invocation_id) id_67 : vec3<u32>)
{
    var r_27 : u32 = id_67.x;
    if(r_27 >= (cs_rows_0()))
    {
        return;
    }
    var row_3 : u32 = cs_row_0(r_27);
    var _S635 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_3])));
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
        var _S636 : vec4<f32> = vec4<f32>(0.0f);
        var _S637 : bool = side_9 == u32(0);
        var col_1 : u32;
        if(_S637)
        {
            col_1 = _S635.x;
        }
        else
        {
            col_1 = _S635.y;
        }
        var k_52 : u32;
        if((_S635.w) != u32(0))
        {
            k_52 = collider_info_0[u32(2) * col_1].x;
        }
        else
        {
            k_52 = u32(4294967295);
        }
        var _S638 : bool;
        if(k_52 != u32(4294967295))
        {
            _S638 = k_52 < (params_0.chunk_count_0);
        }
        else
        {
            _S638 = false;
        }
        var _S639 : bool;
        if(_S638)
        {
            var _S640 : bool = dirty_0(body_of_0(k_52));
            _S639 = _S640;
        }
        else
        {
            _S639 = false;
        }
        var saved_0 : vec4<f32>;
        if(_S639)
        {
            var at_26 : u32 = u32(7) * body_of_0(k_52);
            var anchor_0 : vec3<f32>;
            if(_S637)
            {
                anchor_0 = engine_contacts_0[row_3 + u32(2)].xyz;
            }
            else
            {
                anchor_0 = engine_contacts_0[row_3 + u32(3)].xyz;
            }
            saved_0 = vec4<f32>(bodies_0[at_26].xyz + rotate_0(quat_of_0(bodies_0[at_26 + u32(1)]), anchor_0), 1.0f);
        }
        else
        {
            saved_0 = _S636;
        }
        lp_0[lp_anchor_0(r_27, side_9)] = saved_0;
        side_9 = side_9 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_clean(@builtin(global_invocation_id) id_68 : vec3<u32>)
{
    var _S641 : u32 = id_68.x;
    var _S642 : u32 = atomicLoad(&(atom_0[a_counters_0() + u32(18)]));
    if(_S641 >= _S642)
    {
        return;
    }
    var _S643 : u32 = a_cs_0(work_0[w_cs_bodies_0() + _S641]);
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
        atomicStore(&(atom_0[_S643 + w_9]), u32(0));
        w_9 = w_9 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_fix(@builtin(global_invocation_id) id_69 : vec3<u32>)
{
    var r_28 : u32 = id_69.x;
    if(r_28 >= (cs_rows_0()))
    {
        return;
    }
    var row_4 : u32 = cs_row_0(r_28);
    var _S644 : vec4<u32> = (bitcast<vec4<u32>>((engine_contacts_0[row_4])));
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
            col_2 = _S644.x;
        }
        else
        {
            col_2 = _S644.y;
        }
        var at_27 : u32 = u32(7) * (bitcast<u32>((colliders_0[u32(4) * col_2].x)));
        var anchor_1 : vec3<f32> = rotate_0(conj_0(quat_of_0(bodies_0[at_27 + u32(1)])), saved_1.xyz - bodies_0[at_27].xyz);
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
        var _S645 : u32 = row_4 + u32(6);
        engine_contacts_0[_S645] = vec4<f32>(1.0f, 1.0f, 0.0f, engine_contacts_0[_S645].w);
        engine_contacts_0[row_4 + u32(7)][i32(1)] = 0.0f;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_count(@builtin(global_invocation_id) id_70 : vec3<u32>)
{
    var k_53 : u32 = id_70.x;
    if(k_53 >= (params_0.chunk_count_0))
    {
        return;
    }
    var s_22 : u32 = body_of_0(k_53);
    var _S646 : bool = cs_struck_0(s_22);
    if(!_S646)
    {
        return;
    }
    var _S647 : u32 = a_cs_0(s_22);
    var _S648 : u32 = atomicMax(&(atom_0[_S647 + u32(3)]), ~k_53);
    if(((info_0[k_53] & (u32(1)))) != u32(0))
    {
        atomicStore(&(atom_0[_S647 + u32(4)]), u32(1));
    }
    var _S649 : u32 = atomicAdd(&(atom_0[_S647 + u32(5)]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_base(@builtin(global_invocation_id) id_71 : vec3<u32>)
{
    var _S650 : u32 = id_71.x;
    var _S651 : u32 = a_counters_0();
    var _S652 : u32 = atomicLoad(&(atom_0[_S651 + u32(18)]));
    if(_S650 >= _S652)
    {
        return;
    }
    var _S653 : u32 = a_cs_0(work_0[w_cs_bodies_0() + _S650]);
    var _S654 : u32 = atomicLoad(&(atom_0[_S653 + u32(5)]));
    var _S655 : u32 = atomicAdd(&(atom_0[_S651 + u32(19)]), _S654);
    atomicStore(&(atom_0[_S653 + u32(6)]), _S655);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_cs_collect(@builtin(global_invocation_id) id_72 : vec3<u32>)
{
    var k_54 : u32 = id_72.x;
    if(k_54 >= (params_0.chunk_count_0))
    {
        return;
    }
    var s_23 : u32 = body_of_0(k_54);
    var _S656 : bool = cs_struck_0(s_23);
    if(!_S656)
    {
        return;
    }
    var _S657 : u32 = w_cs_list_0();
    var _S658 : u32 = a_cs_0(s_23);
    var _S659 : u32 = atomicLoad(&(atom_0[_S658 + u32(6)]));
    var _S660 : u32 = _S657 + _S659;
    var _S661 : u32 = atomicAdd(&(atom_0[_S658 + u32(7)]), u32(1));
    work_0[_S660 + _S661] = k_54;
    return;
}

fn cs_load_0( k_55 : u32,  s_24 : u32)
{
    var _S662 : u32 = a_cs_0(s_24);
    var _S663 : u32 = atomicLoad(&(atom_0[_S662 + u32(4)]));
    var supported_1 : bool = _S663 != u32(0);
    var fixed_2 : bool;
    if(((info_0[k_55] & (u32(1)))) != u32(0))
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            var _S664 : u32 = atomicLoad(&(atom_0[_S662 + u32(3)]));
            fixed_2 = (~_S664) == k_55;
        }
        else
        {
            fixed_2 = false;
        }
    }
    var _S665 : u32 = lp_body_0(s_24);
    var _S666 : u32 = u32(7) * s_24;
    var q_35 : Quat_0;
    var origin_16 : vec3<f32>;
    structure_pose_4(s_24, _S666, &(q_35), &(origin_16));
    var qi_1 : Quat_0 = conj_0(q_35);
    var force_4 : vec3<f32> = rotate_0(qi_1, lp_0[_S665 + u32(3)].xyz * vec3<f32>(lp_0[_S665 + u32(2)].w));
    var contact_20 : vec3<f32> = rotate_0(qi_1, lp_0[_S665 + u32(2)].xyz - origin_16);
    var _S667 : u32 = u32(4) * k_55;
    var c0_5 : vec4<f32> = chunks_0[_S667];
    var _S668 : vec3<f32> = vec3<f32>(0.0f);
    var _S669 : u32 = atomicLoad(&(atom_0[_S662 + u32(1)]));
    var f_13 : vec3<f32>;
    var t_16 : vec3<f32>;
    if(_S669 == (k_55 + u32(1)))
    {
        var _S670 : vec3<f32> = cross(contact_20 - c0_5.xyz, force_4);
        f_13 = force_4;
        t_16 = _S670;
    }
    else
    {
        f_13 = _S668;
        t_16 = _S668;
    }
    var inv_mass_7 : f32 = bodies_0[_S666].w;
    var _S671 : bool;
    if(!supported_1)
    {
        _S671 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S671 = false;
    }
    if(_S671)
    {
        var _S672 : u32 = u32(2) * s_24;
        var com_7 : vec3<f32> = frames_0[_S672].xyz;
        var p_23 : Quat_0 = quat_of_0(frames_0[_S672 + u32(1)]);
        var alpha_1 : vec3<f32> = rotate_0(p_23, bodies_0[_S666 + u32(2)].xyz * rotate_0(conj_0(p_23), cross(contact_20 - com_7, force_4)));
        var c1_4 : vec4<f32> = chunks_0[_S667 + u32(1)];
        var c2_5 : vec4<f32> = chunks_0[_S667 + u32(2)];
        var _S673 : f32 = alpha_1.x;
        var _S674 : f32 = c1_4.w;
        var _S675 : f32 = alpha_1.y;
        var _S676 : f32 = c2_5.x;
        var _S677 : f32 = alpha_1.z;
        var _S678 : f32 = c2_5.y;
        var t_17 : vec3<f32> = t_16 - vec3<f32>(c1_4.x * _S673 + _S674 * _S675 + _S676 * _S677, _S674 * _S673 + c1_4.y * _S675 + _S678 * _S677, _S676 * _S673 + _S678 * _S675 + c1_4.z * _S677);
        f_13 = f_13 - vec3<f32>(c0_5.w) * (force_4 * vec3<f32>(inv_mass_7) + cross(alpha_1, c0_5.xyz - com_7));
        t_16 = t_17;
    }
    var _S679 : u32 = st_f_0(k_55, u32(14));
    var _S680 : f32;
    if(fixed_2)
    {
        _S680 = 1.0f;
    }
    else
    {
        _S680 = 0.0f;
    }
    lp_0[_S679] = vec4<f32>(f_13, _S680);
    lp_0[_S679 + u32(1)] = vec4<f32>(t_16, 0.0f);
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
        var _S681 : u32 = st_x_0(k_55, u32(14), c_26);
        var _S682 : vec4<f32> = vec4<f32>(0.0f);
        lp_0[_S681] = _S682;
        lp_0[_S681 + u32(1)] = _S682;
        c_26 = c_26 + u32(1);
    }
    var _S683 : u32 = st_d_0(k_55, u32(14));
    var _S684 : vec4<f32> = vec4<f32>(0.0f);
    lp_0[_S683] = _S684;
    lp_0[_S683 + u32(1)] = _S684;
    return;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn cs_break_0( i_19 : u32,  s_25 : u32)
{
    var bond_9 : vec4<u32> = bonds_0[i_19];
    var b_76 : LpBond_0 = lp_bond_0(i_19);
    var _S685 : u32 = bond_9.x;
    var _S686 : u32 = bond_9.y;
    var q_lin_7 : vec3<f32>;
    var q_ang_7 : vec3<f32>;
    st_force_0(b_76, _S685, _S686, u32(14), u32(0), &(q_lin_7), &(q_ang_7));
    if((params_0.load_model_0) == u32(1))
    {
        var b_lin_0 : vec3<f32>;
        var b_ang_0 : vec3<f32>;
        lp_force_0(b_76, _S685, _S686, u32(0), &(b_lin_0), &(b_ang_0));
        q_lin_7 = q_lin_7 + b_lin_0;
        q_ang_7 = q_ang_7 + b_ang_0;
    }
    var idx_2 : vec4<f32> = failure_indices_0(lp_material_0(b_76.law_0.ids_3.x), b_76.law_0, stress_measures_0(b_76.law_0, q_lin_7, q_ang_7), b_76.law_0.geom1_0.w);
    if((max(max(idx_2.x, idx_2.y), max(idx_2.z, idx_2.w))) >= 1.0f)
    {
        work_0[w_alive_0() + i_19] = u32(0);
        var _S687 : u32 = atomicOr(&(atom_0[a_dirty_0() + s_25]), u32(1));
        var _S688 : u32 = a_counters_0();
        var _S689 : u32 = atomicAdd(&(atom_0[_S688]), u32(1));
        var _S690 : u32 = atomicAdd(&(atom_0[_S688 + u32(1)]), u32(1));
        var _S691 : u32 = atomicAdd(&(atom_0[_S688 + u32(17)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn fr_cs_solve(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var _S692 : bool;
    var j3_0 : u32;
    var s_26 : u32 = work_0[w_cs_bodies_0() + group_3.x];
    var t_18 : u32 = thread_6.x;
    var _S693 : u32 = a_cs_0(work_0[w_cs_bodies_0() + group_3.x]);
    var _S694 : u32 = atomicLoad(&(atom_0[_S693 + u32(6)]));
    var _S695 : u32 = atomicLoad(&(atom_0[_S693 + u32(5)]));
    var j_10 : u32 = t_18;
    loop
    {
        if(j_10 < _S695)
        {
        }
        else
        {
            break;
        }
        cs_load_0(work_0[w_cs_list_0() + _S694 + j_10], s_26);
        j_10 = j_10 + u32(256);
    }
    device_group_barrier_0();
    var j2_0 : u32 = t_18;
    loop
    {
        if(j2_0 < _S695)
        {
        }
        else
        {
            break;
        }
        var k_56 : u32 = work_0[w_cs_list_0() + _S694 + j2_0];
        var _S696 : u32 = st_f_0(work_0[w_cs_list_0() + _S694 + j2_0], u32(14));
        if((lp_0[_S696].w) == 0.0f)
        {
            var _S697 : bool = lp_factor_0(k_56);
            _S692 = !_S697;
        }
        else
        {
            _S692 = false;
        }
        if(_S692)
        {
            lp_0[_S696][i32(3)] = 1.0f;
            var _S698 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(64));
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
        j3_0 = t_18;
        loop
        {
            if(j3_0 < _S695)
            {
            }
            else
            {
                break;
            }
            st_sweep_0(work_0[w_cs_list_0() + _S694 + j3_0], u32(14), sweep_0 % u32(2), params_0.cs_c1_0, params_0.cs_c2_0);
            j3_0 = j3_0 + u32(256);
        }
        device_group_barrier_0();
        sweep_0 = sweep_0 + u32(1);
    }
    j3_0 = t_18;
    loop
    {
        if(j3_0 < _S695)
        {
        }
        else
        {
            break;
        }
        var k_57 : u32 = work_0[w_cs_list_0() + _S694 + j3_0];
        var range_3 : vec4<f32> = ext_0[params_0.ext_chunks_0 + u32(4) * work_0[w_cs_list_0() + _S694 + j3_0]];
        var _S699 : u32 = (bitcast<u32>((range_3.x)));
        var _S700 : u32 = (bitcast<u32>((range_3.y)));
        var e_6 : u32 = u32(0);
        loop
        {
            if(e_6 < _S700)
            {
            }
            else
            {
                break;
            }
            var i_20 : u32 = adj_bond_0(_S699 + e_6);
            if((bonds_0[i_20].x) == k_57)
            {
                _S692 = work_0[w_alive_0() + i_20] != u32(0);
            }
            else
            {
                _S692 = false;
            }
            if(_S692)
            {
                cs_break_0(i_20, s_26);
            }
            e_6 = e_6 + u32(1);
        }
        j3_0 = j3_0 + u32(256);
    }
    return;
}

fn crushable_0( k_58 : u32) -> bool
{
    return (((bitcast<u32>((crush_tab_0[u32(2) * k_58 + u32(1)].z))) & (u32(1)))) != u32(0);
}

fn crush_flags_0( k_59 : u32) -> u32
{
    return (bitcast<u32>((crush_rw_0[k_59].w)));
}

fn crush_record_0( k_60 : u32,  step_6 : vec4<f32>,  cause_0 : u32)
{
    var st_6 : vec4<f32> = crush_rw_0[k_60];
    var flags_1 : u32 = (bitcast<u32>((crush_rw_0[k_60].w)));
    var flags_2 : u32;
    if((step_6.w) != 0.0f)
    {
        flags_2 = (((((flags_1 | (u32(1)))) | (u32(2)))) | (((cause_0 << (u32(4))))));
    }
    else
    {
        flags_2 = flags_1;
    }
    crush_rw_0[k_60] = vec4<f32>(step_6.x, max(st_6.y, step_6.y), max(st_6.z, step_6.z), (bitcast<f32>((flags_2))));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_blast(@builtin(global_invocation_id) id_73 : vec3<u32>)
{
    var k_61 : u32 = id_73.x;
    var _S701 : bool;
    if(k_61 >= (params_0.chunk_count_0))
    {
        _S701 = true;
    }
    else
    {
        _S701 = (crush_0.enabled_0) == u32(0);
    }
    if(_S701)
    {
        return;
    }
    var w_10 : vec4<f32> = crush_rw_0[crush_0.r_blast_0 + k_61];
    var _S702 : f32 = crush_rw_0[crush_0.r_blast_0 + k_61].x;
    if(!(_S702 > 0.0f))
    {
        _S701 = !((w_10.y) > 0.0f);
    }
    else
    {
        _S701 = false;
    }
    if(_S701)
    {
        _S701 = !((w_10.z) > 0.0f);
    }
    else
    {
        _S701 = false;
    }
    if(_S701)
    {
        return;
    }
    crush_rw_0[crush_0.r_blast_0 + k_61] = vec4<f32>(0.0f);
    if(!crushable_0(k_61))
    {
        _S701 = true;
    }
    else
    {
        _S701 = (((crush_flags_0(k_61)) & (u32(1)))) != u32(0);
    }
    if(_S701)
    {
        return;
    }
    var d_24 : f32 = crush_rw_0[k_61].x + _S702;
    var step_7 : vec4<f32>;
    if(d_24 >= 1.0f)
    {
        step_7 = vec4<f32>(1.0f, w_10.y, w_10.z, 1.0f);
    }
    else
    {
        step_7 = vec4<f32>(d_24, w_10.y, w_10.z, 0.0f);
    }
    crush_record_0(k_61, step_7, u32(3));
    return;
}

struct Sym_0
{
     d_25 : vec3<f32>,
     o_10 : vec3<f32>,
};

fn sym_zero_0() -> Sym_0
{
    var s_27 : Sym_0;
    var _S703 : vec3<f32> = vec3<f32>(0.0f);
    s_27.d_25 = _S703;
    s_27.o_10 = _S703;
    return s_27;
}

fn sym_outer_0( r_29 : vec3<f32>,  f_14 : vec3<f32>) -> Sym_0
{
    var s_28 : Sym_0;
    s_28.d_25 = r_29 * f_14;
    var _S704 : f32 = r_29.x;
    var _S705 : f32 = f_14.y;
    var _S706 : f32 = r_29.y;
    var _S707 : f32 = f_14.x;
    var _S708 : f32 = f_14.z;
    var _S709 : f32 = r_29.z;
    s_28.o_10 = vec3<f32>(0.5f) * vec3<f32>(_S704 * _S705 + _S706 * _S707, _S704 * _S708 + _S709 * _S707, _S706 * _S708 + _S709 * _S705);
    return s_28;
}

fn sym_add_0( a_14 : Sym_0,  b_77 : Sym_0) -> Sym_0
{
    var s_29 : Sym_0;
    s_29.d_25 = a_14.d_25 + b_77.d_25;
    s_29.o_10 = a_14.o_10 + b_77.o_10;
    return s_29;
}

fn add_load_0( w_11 : ptr<function, Sym_0>,  f_sum_0 : ptr<function, vec3<f32>>,  spread_0 : ptr<function, vec4<f32>>,  r_30 : vec3<f32>,  f_15 : vec3<f32>)
{
    (*w_11) = sym_add_0((*w_11), sym_outer_0(r_30, f_15));
    (*f_sum_0) = (*f_sum_0) + f_15;
    var m_6 : f32 = length(f_15);
    (*spread_0) = (*spread_0) + vec4<f32>(vec3<f32>(m_6) * r_30, m_6);
    return;
}

fn add_load_1( w_12 : ptr<function, Sym_0>,  f_sum_1 : ptr<function, vec3<f32>>,  spread_1 : ptr<function, vec4<f32>>,  r_31 : vec3<f32>,  f_16 : vec3<f32>)
{
    (*w_12) = sym_add_0((*w_12), sym_outer_0(r_31, f_16));
    (*f_sum_1) = (*f_sum_1) + f_16;
    var m_7 : f32 = length(f_16);
    (*spread_1) = (*spread_1) + vec4<f32>(vec3<f32>(m_7) * r_31, m_7);
    return;
}

fn flow_virial_0( k_62 : u32,  q_36 : Quat_0,  origin_17 : vec3<f32>,  centre_1 : vec3<f32>,  wf_0 : ptr<function, Sym_0>,  total_0 : ptr<function, f32>)
{
    (*wf_0) = sym_zero_0();
    (*total_0) = 0.0f;
    var adj_0 : vec4<f32> = crush_tab_0[crush_0.t_adj_0 + k_62];
    var first_2 : u32 = (bitcast<u32>((adj_0.x)));
    var _S710 : u32 = (bitcast<u32>((adj_0.y)));
    var j_11 : u32 = first_2;
    loop
    {
        if(j_11 < (first_2 + _S710))
        {
        }
        else
        {
            break;
        }
        var bi_0 : u32 = (bitcast<u32>((crush_tab_0[crush_0.t_list_0 + j_11 / u32(4)][j_11 % u32(4)])));
        if(work_0[w_alive_0() + bi_0] == u32(0))
        {
            j_11 = j_11 + u32(1);
            continue;
        }
        var l_3 : f32 = crush_rw_0[crush_0.r_flow_0 + bi_0 / u32(4)][bi_0 % u32(4)];
        if(!((crush_rw_0[crush_0.r_flow_0 + bi_0 / u32(4)][bi_0 % u32(4)]) > 0.0f))
        {
            j_11 = j_11 + u32(1);
            continue;
        }
        var r_32 : vec3<f32> = origin_17 + rotate_0(q_36, crush_tab_0[crush_0.t_bonds_0 + bi_0].xyz) - centre_1;
        var n_7 : vec3<f32> = rotate_0(q_36, crush_tab_0[crush_0.t_normal_0 + bi_0].xyz);
        var out_n_0 : vec3<f32>;
        if((dot(n_7, r_32)) >= 0.0f)
        {
            out_n_0 = n_7;
        }
        else
        {
            out_n_0 = (vec3<f32>(0) - n_7);
        }
        (*wf_0) = sym_add_0((*wf_0), sym_outer_0(r_32, vec3<f32>(- l_3) * out_n_0));
        (*total_0) = (*total_0) + l_3;
        j_11 = j_11 + u32(1);
    }
    return;
}

fn bond_centroid_0( k_63 : u32,  q_37 : Quat_0,  origin_18 : vec3<f32>,  centre_2 : vec3<f32>) -> vec4<f32>
{
    var adj_1 : vec4<f32> = crush_tab_0[crush_0.t_adj_0 + k_63];
    var first_3 : u32 = (bitcast<u32>((adj_1.x)));
    var _S711 : u32 = (bitcast<u32>((adj_1.y)));
    var _S712 : vec3<f32> = vec3<f32>(0.0f);
    var j_12 : u32 = first_3;
    var sum_4 : vec3<f32> = _S712;
    var area_1 : f32 = 0.0f;
    loop
    {
        if(j_12 < (first_3 + _S711))
        {
        }
        else
        {
            break;
        }
        var bi_1 : u32 = (bitcast<u32>((crush_tab_0[crush_0.t_list_0 + j_12 / u32(4)][j_12 % u32(4)])));
        if(work_0[w_alive_0() + bi_1] == u32(0))
        {
            j_12 = j_12 + u32(1);
            continue;
        }
        var bc_0 : vec4<f32> = crush_tab_0[crush_0.t_bonds_0 + bi_1];
        var _S713 : f32 = bc_0.w;
        var area_2 : f32 = area_1 + _S713;
        sum_4 = sum_4 + vec3<f32>(_S713) * (origin_18 + rotate_0(q_37, bc_0.xyz) - centre_2);
        area_1 = area_2;
        j_12 = j_12 + u32(1);
    }
    if(area_1 > 0.0f)
    {
        sum_4 = sum_4 / vec3<f32>(area_1);
    }
    else
    {
        sum_4 = _S712;
    }
    return vec4<f32>(sum_4, area_1);
}

fn balanced_0( w_13 : Sym_0,  f_17 : vec3<f32>,  spread_2 : vec4<f32>,  bonds_1 : vec4<f32>) -> Sym_0
{
    var at_28 : vec3<f32>;
    if((bonds_1.w) > 0.0f)
    {
        at_28 = bonds_1.xyz;
    }
    else
    {
        var _S714 : f32 = spread_2.w;
        if(_S714 > 0.0f)
        {
            at_28 = spread_2.xyz / vec3<f32>(_S714);
        }
        else
        {
            at_28 = vec3<f32>(0.0f);
        }
    }
    return sym_add_0(w_13, sym_outer_0(at_28, (vec3<f32>(0) - f_17)));
}

fn sym_norm2_0( s_30 : Sym_0) -> f32
{
    var _S715 : vec3<f32> = s_30.d_25;
    var _S716 : vec3<f32> = s_30.o_10;
    return dot(_S715, _S715) + 2.0f * dot(_S716, _S716);
}

fn crush_invariants_0( v_18 : Sym_0,  volume_0 : f32) -> vec2<f32>
{
    var inv_4 : f32 = 1.0f / volume_0;
    var xx_0 : f32 = v_18.d_25.x * inv_4;
    var yy_0 : f32 = v_18.d_25.y * inv_4;
    var zz_0 : f32 = v_18.d_25.z * inv_4;
    var xy_0 : f32 = v_18.o_10.x * inv_4;
    var xz_0 : f32 = v_18.o_10.y * inv_4;
    var yz_0 : f32 = v_18.o_10.z * inv_4;
    var p_24 : f32 = - (xx_0 + yy_0 + zz_0) / 3.0f;
    var dx_0 : f32 = xx_0 + p_24;
    var dy_0 : f32 = yy_0 + p_24;
    var dz_0 : f32 = zz_0 + p_24;
    return vec2<f32>(p_24, sqrt(1.5f * max(dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0 + 2.0f * (xy_0 * xy_0 + xz_0 * xz_0 + yz_0 * yz_0), 0.0f)));
}

fn crush_excess_0( pq_0 : vec2<f32>,  m0_0 : vec4<f32>) -> f32
{
    var _S717 : f32 = pq_0.x;
    if(!(_S717 > 0.0f))
    {
        return 0.0f;
    }
    var _S718 : f32 = max(pq_0.y - (m0_0.y + m0_0.z * _S717), _S717 - m0_0.x);
    var _S719 : f32;
    if(_S718 > 0.0f)
    {
        _S719 = _S718;
    }
    else
    {
        _S719 = 0.0f;
    }
    return _S719;
}

fn crush_two_0( a_15 : Sym_0,  dt_a_0 : f32,  b_78 : Sym_0,  dt_b_0 : f32,  k_64 : u32,  damage_2 : f32) -> vec4<f32>
{
    var _S720 : u32 = u32(2) * k_64;
    var m0_1 : vec4<f32> = crush_tab_0[_S720];
    var m1_1 : vec4<f32> = crush_tab_0[_S720 + u32(1)];
    var _S721 : f32 = m1_1.y;
    var pa_3 : vec2<f32> = crush_invariants_0(a_15, _S721);
    var pb_4 : vec2<f32> = crush_invariants_0(b_78, _S721);
    var ea_0 : f32 = crush_excess_0(pa_3, m0_1);
    var _S722 : bool = dt_b_0 > 0.0f;
    var eb_0 : f32;
    if(_S722)
    {
        eb_0 = crush_excess_0(pb_4, m0_1);
    }
    else
    {
        eb_0 = 0.0f;
    }
    var d_26 : f32 = damage_2 + (ea_0 * ea_0 * dt_a_0 + eb_0 * eb_0 * dt_b_0) / (m1_1.x * m0_1.w);
    var peak_2 : vec2<f32>;
    if(_S722)
    {
        peak_2 = max(pa_3, pb_4);
    }
    else
    {
        peak_2 = pa_3;
    }
    if(d_26 >= 1.0f)
    {
        return vec4<f32>(1.0f, peak_2, 1.0f);
    }
    return vec4<f32>(d_26, peak_2, 0.0f);
}

fn crush_loads_flow_0( k_65 : u32)
{
    var slot_21 : u32 = body_of_0(k_65);
    var b_79 : u32 = u32(7) * slot_21;
    var q_38 : Quat_0;
    var origin_19 : vec3<f32>;
    structure_pose_4(slot_21, b_79, &(q_38), &(origin_19));
    var _S723 : u32 = u32(4) * k_65;
    var c0_6 : vec4<f32> = chunks_0[_S723];
    var _S724 : vec4<f32> = chunks_0[_S723 + u32(3)];
    var centre_3 : vec3<f32> = origin_19 + rotate_0(q_38, c0_6.xyz);
    var inv_mass_8 : f32 = bodies_0[b_79].w;
    var _S725 : f32;
    if(inv_mass_8 > 0.0f)
    {
        _S725 = 1.0f / inv_mass_8;
    }
    else
    {
        _S725 = float_bits_0(u32(2139095040));
    }
    var dt_2 : f32 = params_0.dt_0;
    var _S726 : Sym_0 = sym_zero_0();
    var wa_0 : Sym_0 = _S726;
    var _S727 : vec3<f32> = vec3<f32>(0.0f);
    var fa_0 : vec3<f32> = _S727;
    var _S728 : vec4<f32> = vec4<f32>(0.0f);
    var sa_0 : vec4<f32> = _S728;
    var wb_0 : Sym_0 = _S726;
    var fb_1 : vec3<f32> = _S727;
    var sb_2 : vec4<f32> = _S728;
    var _S729 : u32 = sort_0[k_65];
    var _S730 : u32 = sort_0[params_0.chunk_count_0 + u32(1) + k_65];
    var tau_3 : f32 = 0.0f;
    var i_21 : u32 = u32(0);
    loop
    {
        if(i_21 < _S730)
        {
        }
        else
        {
            break;
        }
        var r_33 : Record_std430_0 = manifolds_0[_S729 + i_21];
        var _S731 : vec3<f32> = r_33.impulse_0.xyz;
        var j_13 : f32 = length(_S731);
        if(j_13 == 0.0f)
        {
            i_21 = i_21 + u32(1);
            continue;
        }
        var m_other_2 : f32 = r_33.other_0.x;
        var m_8 : f32;
        if(isinf_0(_S725))
        {
            if(isinf_0(m_other_2))
            {
                m_8 = c0_6.w;
            }
            else
            {
                m_8 = m_other_2;
            }
        }
        else
        {
            if(isinf_0(m_other_2))
            {
                m_8 = _S725;
            }
            else
            {
                m_8 = _S725 * m_other_2 / (_S725 + m_other_2);
            }
        }
        var _S732 : f32 = r_33.impulse_0.w;
        var _S733 : f32 = min(j_13, 2.0f * m_8 * max(_S732 - params_0.band_0, 0.0f));
        var rr_0 : vec3<f32> = r_33.point_0.xyz - centre_3;
        var dir_1 : vec3<f32> = _S731 / vec3<f32>(j_13);
        if(_S733 < j_13)
        {
            add_load_0(&(wa_0), &(fa_0), &(sa_0), rr_0, dir_1 * vec3<f32>(((j_13 - _S733) / dt_2)));
        }
        var tau_4 : f32;
        if(_S733 > 0.0f)
        {
            var _S734 : f32 = _S724.y;
            var duration_2 : f32 = hertz_duration_0(m_8, min(r_33.other_0.z, _S724.z), 1.0f / ((1.0f - _S734 * _S734) / _S724.x + 1.0f / r_33.other_0.y), _S732);
            add_load_0(&(wb_0), &(fb_1), &(sb_2), rr_0, dir_1 * vec3<f32>((3.14159274101257324f * _S733 / (2.0f * duration_2))));
            tau_4 = max(tau_3, min(duration_2, dt_2));
        }
        else
        {
            tau_4 = tau_3;
        }
        tau_3 = tau_4;
        i_21 = i_21 + u32(1);
    }
    var _S735 : u32 = w_pulses_0() + _S723 + u32(2) * params_0.parity_0;
    var _S736 : u32 = work_0[_S735 + u32(1)];
    var _S737 : u32 = l_pool_0(params_0.parity_0) + u32(3) * work_0[_S735];
    var p_25 : u32 = u32(0);
    loop
    {
        if(p_25 < _S736)
        {
        }
        else
        {
            break;
        }
        var _S738 : u32 = _S737 + u32(3) * p_25;
        var bb_0 : vec4<f32> = loads_0[_S738 + u32(1)];
        var c_27 : vec4<f32> = loads_0[_S738 + u32(2)];
        var mean_2 : f32 = pulse_mean_0(loads_0[_S738].x, loads_0[_S738].y, loads_0[_S738].z, loads_0[_S738].w, loads_0[_S738 + u32(1)].w, dt_2);
        if(mean_2 != 0.0f)
        {
            add_load_0(&(wa_0), &(fa_0), &(sa_0), bb_0.xyz - centre_3, c_27.xyz * vec3<f32>(mean_2));
        }
        p_25 = p_25 + u32(1);
    }
    var wf_1 : Sym_0;
    var flow_total_0 : f32;
    flow_virial_0(k_65, q_38, origin_19, centre_3, &(wf_1), &(flow_total_0));
    var contact_21 : bool;
    if((sa_0.w) > 0.0f)
    {
        contact_21 = true;
    }
    else
    {
        contact_21 = (sb_2.w) > 0.0f;
    }
    var _S739 : bool;
    if(!contact_21)
    {
        _S739 = !(flow_total_0 > 0.0f);
    }
    else
    {
        _S739 = false;
    }
    if(_S739)
    {
        return;
    }
    var _S740 : Sym_0 = wf_1;
    var _S741 : Sym_0 = wf_1;
    var peak_contact_0 : Sym_0;
    var base_3 : Sym_0;
    var peak_3 : Sym_0;
    if(contact_21)
    {
        var bonds_2 : vec4<f32> = bond_centroid_0(k_65, q_38, origin_19, centre_3);
        var _S742 : Sym_0 = sym_add_0(balanced_0(wa_0, fa_0, sa_0, bonds_2), wf_1);
        var peak_contact_1 : Sym_0 = balanced_0(sym_add_0(wa_0, wb_0), fa_0 + fb_1, sa_0 + sb_2, bonds_2);
        var _S743 : Sym_0 = sym_add_0(peak_contact_1, wf_1);
        peak_contact_0 = peak_contact_1;
        base_3 = _S742;
        peak_3 = _S743;
    }
    else
    {
        peak_contact_0 = _S726;
        base_3 = _S740;
        peak_3 = _S741;
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
    crush_record_0(k_65, crush_two_0(base_3, dt_2 - tau_3, peak_3, tau_3, k_65, crush_rw_0[k_65].x), cause_1);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_loads(@builtin(global_invocation_id) id_74 : vec3<u32>)
{
    var k_66 : u32 = id_74.x;
    var _S744 : bool;
    if(k_66 >= (params_0.chunk_count_0))
    {
        _S744 = true;
    }
    else
    {
        _S744 = (crush_0.enabled_0) == u32(0);
    }
    if(_S744)
    {
        _S744 = true;
    }
    else
    {
        _S744 = !crushable_0(k_66);
    }
    if(_S744)
    {
        _S744 = true;
    }
    else
    {
        _S744 = (((crush_flags_0(k_66)) & (u32(1)))) != u32(0);
    }
    if(_S744)
    {
        return;
    }
    if((crush_0.flow_0) != u32(0))
    {
        crush_loads_flow_0(k_66);
        return;
    }
    var slot_22 : u32 = body_of_0(k_66);
    var b_80 : u32 = u32(7) * slot_22;
    var q_39 : Quat_0;
    var origin_20 : vec3<f32>;
    structure_pose_5(slot_22, b_80, &(q_39), &(origin_20));
    var _S745 : u32 = u32(4) * k_66;
    var c0_7 : vec4<f32> = chunks_0[_S745];
    var _S746 : vec4<f32> = chunks_0[_S745 + u32(3)];
    var centre_4 : vec3<f32> = origin_20 + rotate_0(q_39, c0_7.xyz);
    var inv_mass_9 : f32 = bodies_0[b_80].w;
    var _S747 : f32;
    if(inv_mass_9 > 0.0f)
    {
        _S747 = 1.0f / inv_mass_9;
    }
    else
    {
        _S747 = float_bits_0(u32(2139095040));
    }
    var dt_3 : f32 = params_0.dt_0;
    var _S748 : Sym_0 = sym_zero_0();
    var wa_1 : Sym_0 = _S748;
    var _S749 : vec3<f32> = vec3<f32>(0.0f);
    var fa_1 : vec3<f32> = _S749;
    var _S750 : vec4<f32> = vec4<f32>(0.0f);
    var sa_1 : vec4<f32> = _S750;
    var wb_1 : Sym_0 = _S748;
    var fb_2 : vec3<f32> = _S749;
    var sb_3 : vec4<f32> = _S750;
    var _S751 : u32 = sort_0[k_66];
    var _S752 : u32 = sort_0[params_0.chunk_count_0 + u32(1) + k_66];
    var tau_5 : f32 = 0.0f;
    var i_22 : u32 = u32(0);
    loop
    {
        if(i_22 < _S752)
        {
        }
        else
        {
            break;
        }
        var r_34 : Record_std430_0 = manifolds_0[_S751 + i_22];
        var _S753 : vec3<f32> = r_34.impulse_0.xyz;
        var j_14 : f32 = length(_S753);
        if(j_14 == 0.0f)
        {
            i_22 = i_22 + u32(1);
            continue;
        }
        var m_other_3 : f32 = r_34.other_0.x;
        var m_9 : f32;
        if(isinf_0(_S747))
        {
            if(isinf_0(m_other_3))
            {
                m_9 = c0_7.w;
            }
            else
            {
                m_9 = m_other_3;
            }
        }
        else
        {
            if(isinf_0(m_other_3))
            {
                m_9 = _S747;
            }
            else
            {
                m_9 = _S747 * m_other_3 / (_S747 + m_other_3);
            }
        }
        var _S754 : f32 = r_34.impulse_0.w;
        var _S755 : f32 = min(j_14, 2.0f * m_9 * max(_S754 - params_0.band_0, 0.0f));
        var rr_1 : vec3<f32> = r_34.point_0.xyz - centre_4;
        var dir_2 : vec3<f32> = _S753 / vec3<f32>(j_14);
        if(_S755 < j_14)
        {
            add_load_1(&(wa_1), &(fa_1), &(sa_1), rr_1, dir_2 * vec3<f32>(((j_14 - _S755) / dt_3)));
        }
        var tau_6 : f32;
        if(_S755 > 0.0f)
        {
            var _S756 : f32 = _S746.y;
            var duration_3 : f32 = hertz_duration_0(m_9, min(r_34.other_0.z, _S746.z), 1.0f / ((1.0f - _S756 * _S756) / _S746.x + 1.0f / r_34.other_0.y), _S754);
            add_load_1(&(wb_1), &(fb_2), &(sb_3), rr_1, dir_2 * vec3<f32>((3.14159274101257324f * _S755 / (2.0f * duration_3))));
            tau_6 = max(tau_5, min(duration_3, dt_3));
        }
        else
        {
            tau_6 = tau_5;
        }
        tau_5 = tau_6;
        i_22 = i_22 + u32(1);
    }
    var _S757 : u32 = w_pulses_0() + _S745 + u32(2) * params_0.parity_0;
    var _S758 : u32 = work_0[_S757 + u32(1)];
    var _S759 : u32 = l_pool_0(params_0.parity_0) + u32(3) * work_0[_S757];
    var p_26 : u32 = u32(0);
    loop
    {
        if(p_26 < _S758)
        {
        }
        else
        {
            break;
        }
        var _S760 : u32 = _S759 + u32(3) * p_26;
        var bb_1 : vec4<f32> = loads_0[_S760 + u32(1)];
        var c_28 : vec4<f32> = loads_0[_S760 + u32(2)];
        var mean_3 : f32 = pulse_mean_0(loads_0[_S760].x, loads_0[_S760].y, loads_0[_S760].z, loads_0[_S760].w, loads_0[_S760 + u32(1)].w, dt_3);
        if(mean_3 != 0.0f)
        {
            add_load_1(&(wa_1), &(fa_1), &(sa_1), bb_1.xyz - centre_4, c_28.xyz * vec3<f32>(mean_3));
        }
        p_26 = p_26 + u32(1);
    }
    if(!((sa_1.w) > 0.0f))
    {
        _S744 = !((sb_3.w) > 0.0f);
    }
    else
    {
        _S744 = false;
    }
    if(_S744)
    {
        return;
    }
    var bonds_3 : vec4<f32> = bond_centroid_0(k_66, q_39, origin_20, centre_4);
    crush_record_0(k_66, crush_two_0(balanced_0(wa_1, fa_1, sa_1, bonds_3), dt_3 - tau_5, balanced_0(sym_add_0(wa_1, wb_1), fa_1 + fb_2, sa_1 + sb_3, bonds_3), tau_5, k_66, crush_rw_0[k_66].x), u32(1));
    return;
}

fn a_cover_0() -> u32
{
    return a_strike_dist_0() + params_0.chunk_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_patch(@builtin(global_invocation_id) id_75 : vec3<u32>)
{
    var b_81 : u32 = id_75.x;
    var _S761 : bool;
    if(b_81 >= (params_0.body_count_0))
    {
        _S761 = true;
    }
    else
    {
        _S761 = (params_0.shot_count_0) == u32(0);
    }
    if(_S761)
    {
        _S761 = true;
    }
    else
    {
        _S761 = (crush_0.enabled_0) == u32(0);
    }
    if(_S761)
    {
        return;
    }
    var bill_0 : u32 = crush_0.r_bill_0 + u32(4) * b_81;
    var _S762 : vec4<f32> = vec4<f32>(0.0f);
    crush_rw_0[bill_0] = _S762;
    var _S763 : u32 = bill_0 + u32(3);
    crush_rw_0[_S763] = _S762;
    var shot_11 : u32;
    var _S764 : bool = struck_2(b_81, &(shot_11));
    if(!_S764)
    {
        return;
    }
    var _S765 : u32 = a_strike_0() + u32(9) * b_81;
    var k_67 : u32 = atomicLoad(&(atom_0[_S765 + u32(8)]));
    var _S766 : u32 = atomicLoad(&(atom_0[_S765 + u32(6)]));
    var t_in_13 : f32 = (bitcast<f32>((_S766)));
    if(!(t_in_13 <= (params_0.dt_0)))
    {
        _S761 = true;
    }
    else
    {
        _S761 = k_67 == u32(4294967295);
    }
    if(_S761)
    {
        return;
    }
    var st_7 : u32 = strike_at_0(b_81);
    var s0_5 : vec4<f32> = frames_0[st_7];
    var _S767 : vec3<f32> = frames_0[st_7 + u32(1)].xyz;
    var speed_17 : f32 = length(_S767);
    var d_27 : vec3<f32> = _S767 / vec3<f32>(max(speed_17, 9.999999960041972e-13f));
    var info_4 : vec4<f32> = shots_0[u32(2) * shot_11];
    var radius_1 : f32 = info_4.y;
    var _S768 : f32 = max(info_4.z, 0.00100000004749745f);
    var m_imp_1 : f32 = info_4.w;
    var inv_mass_10 : f32 = bodies_0[u32(7) * b_81].w;
    var _S769 : bool = inv_mass_10 == 0.0f;
    var m_body_2 : f32;
    if(_S769)
    {
        m_body_2 = 0.0f;
    }
    else
    {
        m_body_2 = 1.0f / inv_mass_10;
    }
    var m_eff_10 : f32;
    if(_S769)
    {
        m_eff_10 = m_imp_1;
    }
    else
    {
        m_eff_10 = m_imp_1 * m_body_2 / (m_imp_1 + m_body_2);
    }
    var modulus_2 : f32 = 1.0f / (1.0f / shots_0[u32(2) * shot_11 + u32(1)].x + 1.0f / geometry_0[u32(3) * k_67 + u32(1)].w);
    var _S770 : f32 = max(speed_17, 0.00100000004749745f);
    var tau_7 : f32 = 2.86999988555908203f * pow(m_eff_10 * m_eff_10 / (_S768 * modulus_2 * modulus_2 * _S770), 0.20000000298023224f);
    var force_5 : f32 = m_eff_10 * speed_17 / tau_7;
    var contact_22 : vec3<f32> = s0_5.xyz + _S767 * vec3<f32>(t_in_13) + d_27 * vec3<f32>(radius_1);
    var delta_4 : f32 = pow(15.0f * m_eff_10 * _S770 * _S770 / (16.0f * modulus_2 * sqrt(_S768)), 0.40000000596046448f);
    var a_h_0 : f32 = sqrt(radius_1 * delta_4);
    var p_cap_0 : f32;
    if(crushable_0(k_67))
    {
        p_cap_0 = crush_tab_0[u32(2) * k_67].x;
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
    var _S771 : f32 = min(radius_1, max(a_h_0, a_p_0));
    var _S772 : f32 = max(delta_4, radius_1 - sqrt(max(radius_1 * radius_1 - _S771 * _S771, 0.0f)));
    crush_rw_0[bill_0 + u32(1)] = vec4<f32>(contact_22, 0.0f);
    crush_rw_0[bill_0 + u32(2)] = vec4<f32>(d_27, _S771);
    crush_rw_0[_S763] = vec4<f32>(force_5, tau_7, _S772, f32(k_67 + u32(1)));
    var _S773 : u32 = a_cover_0() + u32(128) * b_81;
    var s_31 : u32 = u32(0);
    loop
    {
        if(s_31 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S774 : u32 = _S773 + u32(2) * s_31;
        atomicStore(&(atom_0[_S774]), u32(4294967295));
        atomicStore(&(atom_0[_S774 + u32(1)]), u32(4294967295));
        s_31 = s_31 + u32(1);
    }
    return;
}

fn cover_patch_0( k_68 : u32,  b_82 : ptr<function, u32>,  p1_0 : ptr<function, vec4<f32>>,  p2_2 : ptr<function, vec4<f32>>) -> bool
{
    (*b_82) = body_of_0(k_68);
    var _S775 : vec4<f32> = vec4<f32>(0.0f);
    (*p1_0) = _S775;
    (*p2_2) = _S775;
    var shot_12 : u32;
    var _S776 : bool = struck_2((*b_82), &(shot_12));
    var _S777 : bool;
    if(!_S776)
    {
        _S777 = true;
    }
    else
    {
        _S777 = (((crush_flags_0(k_68)) & (u32(1)))) != u32(0);
    }
    if(_S777)
    {
        return false;
    }
    var bill_1 : u32 = crush_0.r_bill_0 + u32(4) * (*b_82);
    if((crush_rw_0[bill_1 + u32(3)].w) == 0.0f)
    {
        return false;
    }
    (*p1_0) = vec4<f32>(crush_rw_0[bill_1 + u32(1)].xyz, shots_0[u32(2) * shot_12].y);
    (*p2_2) = crush_rw_0[bill_1 + u32(2)];
    return true;
}

fn cover_offset_0( s_32 : u32,  d_28 : vec3<f32>,  a_16 : f32,  rho_0 : ptr<function, f32>) -> vec3<f32>
{
    var ring_2 : u32 = s_32 / u32(8);
    var sector_0 : u32 = s_32 % u32(8);
    (*rho_0) = a_16 * sqrt((f32(ring_2) + 0.5f) / 8.0f);
    var theta_1 : f32 = 6.28318548202514648f * (f32(sector_0) + 0.5f * f32((ring_2 & (u32(1))))) / 8.0f;
    var e_7 : vec3<f32>;
    if((abs(d_28.x)) < 0.57735002040863037f)
    {
        e_7 = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_28.y)) < 0.57735002040863037f)
        {
            e_7 = vec3<f32>(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_7 = vec3<f32>(0.0f, 0.0f, 1.0f);
        }
    }
    var u_12 : vec3<f32> = normalize(cross(d_28, e_7));
    return vec3<f32>((*rho_0)) * (vec3<f32>(cos(theta_1)) * u_12 + vec3<f32>(sin(theta_1)) * cross(d_28, u_12));
}

fn cover_offset_1( s_33 : u32,  d_29 : vec3<f32>,  a_17 : f32,  rho_1 : ptr<function, f32>) -> vec3<f32>
{
    var ring_3 : u32 = s_33 / u32(8);
    var sector_1 : u32 = s_33 % u32(8);
    (*rho_1) = a_17 * sqrt((f32(ring_3) + 0.5f) / 8.0f);
    var theta_2 : f32 = 6.28318548202514648f * (f32(sector_1) + 0.5f * f32((ring_3 & (u32(1))))) / 8.0f;
    var e_8 : vec3<f32>;
    if((abs(d_29.x)) < 0.57735002040863037f)
    {
        e_8 = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_29.y)) < 0.57735002040863037f)
        {
            e_8 = vec3<f32>(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_8 = vec3<f32>(0.0f, 0.0f, 1.0f);
        }
    }
    var u_13 : vec3<f32> = normalize(cross(d_29, e_8));
    return vec3<f32>((*rho_1)) * (vec3<f32>(cos(theta_2)) * u_13 + vec3<f32>(sin(theta_2)) * cross(d_29, u_13));
}

fn cover_depth_0( k_69 : u32,  s_34 : u32,  contact_23 : vec3<f32>,  d_30 : vec3<f32>,  a_18 : f32,  radius_2 : f32) -> f32
{
    var rho_2 : f32;
    var _S778 : f32 = 2.0f * radius_2;
    var _S779 : vec3<f32> = contact_23 + cover_offset_0(s_34, d_30, a_18, &(rho_2)) - d_30 * vec3<f32>(_S778);
    var q_40 : Quat_0;
    var _S780 : vec3<f32> = chunk_box_world_2(k_69, &(q_40));
    var _S781 : u32 = u32(3) * k_69;
    var _S782 : Quat_0 = normalized_0(quat_mul_0(q_40, quat_of_0(geometry_0[_S781 + u32(2)])));
    var _S783 : vec3<f32> = geometry_0[_S781 + u32(1)].xyz;
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
        var e_9 : vec3<f32> = vec3<f32>(0.0f);
        e_9[ax_4] = 1.0f;
        var axis_2 : vec3<f32> = rotate_0(_S782, e_9);
        var o_11 : f32 = dot(_S779 - _S780, axis_2);
        var dv_2 : f32 = dot(d_30, axis_2);
        if((abs(dv_2)) < 9.99999971718068537e-10f)
        {
            if((abs(o_11)) > (_S783[ax_4]))
            {
                return 1.00000001504746622e+30f;
            }
            ax_4 = ax_4 + u32(1);
            continue;
        }
        var ta_3 : f32 = (- _S783[ax_4] - o_11) / dv_2;
        var tb_3 : f32 = (_S783[ax_4] - o_11) / dv_2;
        var _S784 : f32 = min(t1_4, max(ta_3, tb_3));
        t0_3 = max(t0_3, min(ta_3, tb_3));
        t1_4 = _S784;
        ax_4 = ax_4 + u32(1);
    }
    var _S785 : bool;
    if(!(t0_3 <= t1_4))
    {
        _S785 = true;
    }
    else
    {
        _S785 = t1_4 < 0.0f;
    }
    if(_S785)
    {
        return 1.00000001504746622e+30f;
    }
    return t0_3 - _S778;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_cover(@builtin(global_invocation_id) id_76 : vec3<u32>)
{
    var k_70 : u32 = id_76.x;
    var _S786 : bool;
    if(k_70 >= (params_0.chunk_count_0))
    {
        _S786 = true;
    }
    else
    {
        _S786 = (params_0.shot_count_0) == u32(0);
    }
    if(_S786)
    {
        _S786 = true;
    }
    else
    {
        _S786 = (crush_0.enabled_0) == u32(0);
    }
    if(_S786)
    {
        return;
    }
    var b_83 : u32;
    var p1_1 : vec4<f32>;
    var p2_3 : vec4<f32>;
    var _S787 : bool = cover_patch_0(k_70, &(b_83), &(p1_1), &(p2_3));
    if(!_S787)
    {
        return;
    }
    var _S788 : u32 = a_cover_0() + u32(128) * b_83;
    var s_35 : u32 = u32(0);
    loop
    {
        if(s_35 < u32(64))
        {
        }
        else
        {
            break;
        }
        var t_19 : f32 = cover_depth_0(k_70, s_35, p1_1.xyz, p2_3.xyz, p2_3.w, p1_1.w);
        if(t_19 < 5.0000000752373311e+29f)
        {
            var _S789 : u32 = atomicMin(&(atom_0[_S788 + u32(2) * s_35]), ordered_0(t_19));
        }
        s_35 = s_35 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_claim(@builtin(global_invocation_id) id_77 : vec3<u32>)
{
    var k_71 : u32 = id_77.x;
    var _S790 : bool;
    if(k_71 >= (params_0.chunk_count_0))
    {
        _S790 = true;
    }
    else
    {
        _S790 = (params_0.shot_count_0) == u32(0);
    }
    if(_S790)
    {
        _S790 = true;
    }
    else
    {
        _S790 = (crush_0.enabled_0) == u32(0);
    }
    if(_S790)
    {
        return;
    }
    var b_84 : u32;
    var p1_2 : vec4<f32>;
    var p2_4 : vec4<f32>;
    var _S791 : bool = cover_patch_0(k_71, &(b_84), &(p1_2), &(p2_4));
    if(!_S791)
    {
        return;
    }
    var _S792 : u32 = a_cover_0() + u32(128) * b_84;
    var s_36 : u32 = u32(0);
    loop
    {
        if(s_36 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S793 : u32 = _S792 + u32(2) * s_36;
        var first_4 : u32 = atomicLoad(&(atom_0[_S793]));
        if(first_4 == u32(4294967295))
        {
            s_36 = s_36 + u32(1);
            continue;
        }
        var t_20 : f32 = cover_depth_0(k_71, s_36, p1_2.xyz, p2_4.xyz, p2_4.w, p1_2.w);
        if(t_20 < 5.0000000752373311e+29f)
        {
            _S790 = t_20 <= (unordered_0(first_4) + 0.00009999999747379f);
        }
        else
        {
            _S790 = false;
        }
        if(_S790)
        {
            var _S794 : u32 = atomicMin(&(atom_0[_S793 + u32(1)]), k_71);
        }
        s_36 = s_36 + u32(1);
    }
    return;
}

fn strike_load_0( k_72 : u32,  slot_23 : u32,  f_18 : vec3<f32>,  at_29 : vec3<f32>,  tau_8 : f32,  budget_0 : f32) -> f32
{
    var _S795 : bool;
    if(!crushable_0(k_72))
    {
        _S795 = true;
    }
    else
    {
        _S795 = (((crush_flags_0(k_72)) & (u32(1)))) != u32(0);
    }
    if(_S795)
    {
        _S795 = true;
    }
    else
    {
        _S795 = !(budget_0 > 0.0f);
    }
    if(_S795)
    {
        return 0.0f;
    }
    var q_41 : Quat_0;
    var origin_21 : vec3<f32>;
    structure_pose_2(slot_23, u32(7) * slot_23, &(q_41), &(origin_21));
    var centre_5 : vec3<f32> = origin_21 + rotate_0(q_41, chunks_0[u32(4) * k_72].xyz);
    var bonds_4 : vec4<f32> = bond_centroid_0(k_72, q_41, origin_21, centre_5);
    if(!((bonds_4.w) > 0.0f))
    {
        return 0.0f;
    }
    var w_14 : Sym_0 = sym_add_0(sym_outer_0(at_29 - centre_5, f_18), sym_outer_0(bonds_4.xyz, (vec3<f32>(0) - f_18)));
    var before_0 : f32 = crush_rw_0[k_72].x;
    var step_8 : vec4<f32> = crush_two_0(w_14, min(tau_8, params_0.dt_0), w_14, 0.0f, k_72, before_0);
    var _S796 : u32 = u32(2) * k_72;
    var per_0 : f32 = crush_tab_0[_S796].w * crush_tab_0[_S796 + u32(1)].y;
    var energy_1 : f32 = (step_8.x - before_0) * per_0;
    var energy_2 : f32;
    var step_9 : vec4<f32>;
    if(energy_1 > budget_0)
    {
        var d_31 : f32 = before_0 + budget_0 / per_0;
        var _S797 : f32 = min(d_31, 1.0f);
        var _S798 : f32 = step_8.y;
        var _S799 : f32 = step_8.z;
        if(d_31 >= 1.0f)
        {
            energy_2 = 1.0f;
        }
        else
        {
            energy_2 = 0.0f;
        }
        step_9 = vec4<f32>(_S797, _S798, _S799, energy_2);
        energy_2 = budget_0;
    }
    else
    {
        step_9 = step_8;
        energy_2 = energy_1;
    }
    crush_record_0(k_72, step_9, u32(2));
    return energy_2;
}

fn cell_in_0( mask_0 : vec2<u32>,  c_29 : u32) -> bool
{
    return (((((mask_0[c_29 / u32(32)]) >> ((c_29 % u32(32))))) & (u32(1)))) != u32(0);
}

fn strike_crush_body_0( b_85 : u32,  s_37 : u32,  budget_1 : f32) -> f32
{
    var total_1 : u32;
    var _S800 : bool;
    var bill_2 : u32 = crush_0.r_bill_0 + u32(4) * b_85;
    var _S801 : u32 = bill_2 + u32(1);
    var p3_0 : vec4<f32> = crush_rw_0[bill_2 + u32(3)];
    var contact_24 : vec3<f32> = crush_rw_0[_S801].xyz;
    var d_32 : vec3<f32> = crush_rw_0[bill_2 + u32(2)].xyz;
    var _S802 : f32 = crush_rw_0[bill_2 + u32(2)].w;
    var _S803 : f32 = shots_0[u32(2) * s_37].y;
    var _S804 : u32 = a_cover_0() + u32(128) * b_85;
    var first_5 : f32 = 1.00000001504746622e+30f;
    var c_30 : u32 = u32(0);
    loop
    {
        if(c_30 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S805 : u32 = _S804 + u32(2) * c_30;
        var _S806 : u32 = atomicLoad(&(atom_0[_S805 + u32(1)]));
        if(_S806 == u32(4294967295))
        {
            c_30 = c_30 + u32(1);
            continue;
        }
        var rho_3 : f32;
        var _S807 : vec3<f32> = cover_offset_1(c_30, d_32, _S802, &(rho_3));
        var _S808 : u32 = atomicLoad(&(atom_0[_S805]));
        first_5 = min(first_5, unordered_0(_S808) + _S803 - sqrt(max(_S803 * _S803 - rho_3 * rho_3, 0.0f)));
        c_30 = c_30 + u32(1);
    }
    var counted_0 : vec2<u32> = vec2<u32>(u32(0));
    c_30 = u32(0);
    var total_2 : u32 = u32(0);
    loop
    {
        if(c_30 < u32(64))
        {
        }
        else
        {
            break;
        }
        var _S809 : u32 = _S804 + u32(2) * c_30;
        var _S810 : u32 = atomicLoad(&(atom_0[_S809 + u32(1)]));
        if(_S810 == u32(4294967295))
        {
            c_30 = c_30 + u32(1);
            continue;
        }
        var rho_4 : f32;
        var _S811 : vec3<f32> = cover_offset_1(c_30, d_32, _S802, &(rho_4));
        var _S812 : u32 = atomicLoad(&(atom_0[_S809]));
        if((unordered_0(_S812) + _S803 - sqrt(max(_S803 * _S803 - rho_4 * rho_4, 0.0f)) - first_5) <= (p3_0.z))
        {
            counted_0[c_30 / u32(32)] = ((counted_0[c_30 / u32(32)]) | (((u32(1) << ((c_30 % u32(32)))))));
            total_1 = total_2 + u32(1);
        }
        else
        {
            total_1 = total_2;
        }
        total_2 = total_1;
        c_30 = c_30 + u32(1);
    }
    var _S813 : vec3<f32> = vec3<f32>(0.0f);
    var energy_3 : f32;
    var point_1 : vec3<f32>;
    if(total_2 == u32(0))
    {
        var energy_4 : f32 = strike_load_0(u32(p3_0.w) - u32(1), b_85, d_32 * vec3<f32>(p3_0.x), contact_24, p3_0.y, budget_1);
        var _S814 : vec3<f32> = vec3<f32>(energy_4) * contact_24;
        energy_3 = energy_4;
        point_1 = _S814;
    }
    else
    {
        energy_3 = 0.0f;
        point_1 = _S813;
    }
    c_30 = u32(0);
    loop
    {
        if(c_30 < u32(64))
        {
        }
        else
        {
            break;
        }
        if(!cell_in_0(counted_0, c_30))
        {
            c_30 = c_30 + u32(1);
            continue;
        }
        var k_73 : u32 = atomicLoad(&(atom_0[_S804 + u32(2) * c_30 + u32(1)]));
        var seen_0 : bool = false;
        total_1 = u32(0);
        loop
        {
            if(total_1 < c_30)
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
                    var _S815 : u32 = atomicLoad(&(atom_0[_S804 + u32(2) * total_1 + u32(1)]));
                    _S800 = _S815 == k_73;
                }
                else
                {
                    _S800 = false;
                }
                seen_0 = _S800;
            }
            total_1 = total_1 + u32(1);
        }
        if(seen_0)
        {
            c_30 = c_30 + u32(1);
            continue;
        }
        var e_10 : u32 = c_30;
        var at_30 : vec3<f32> = _S813;
        var n_8 : u32 = u32(0);
        loop
        {
            if(e_10 < u32(64))
            {
            }
            else
            {
                break;
            }
            if(!cell_in_0(counted_0, e_10))
            {
                _S800 = true;
            }
            else
            {
                var _S816 : u32 = atomicLoad(&(atom_0[_S804 + u32(2) * e_10 + u32(1)]));
                _S800 = _S816 != k_73;
            }
            if(_S800)
            {
                e_10 = e_10 + u32(1);
                continue;
            }
            var rho_5 : f32;
            var _S817 : vec3<f32> = contact_24 + cover_offset_1(e_10, d_32, _S802, &(rho_5));
            var _S818 : u32 = atomicLoad(&(atom_0[_S804 + u32(2) * e_10]));
            var _S819 : u32 = n_8 + u32(1);
            at_30 = at_30 + (_S817 + d_32 * vec3<f32>(unordered_0(_S818)));
            n_8 = _S819;
            e_10 = e_10 + u32(1);
        }
        var _S820 : f32 = f32(n_8);
        var at_31 : vec3<f32> = at_30 / vec3<f32>(_S820);
        var took_0 : f32 = strike_load_0(k_73, b_85, d_32 * vec3<f32>((p3_0.x * _S820 / f32(total_2))), at_31, p3_0.y, budget_1 - energy_3);
        var point_2 : vec3<f32> = point_1 + vec3<f32>(took_0) * at_31;
        energy_3 = energy_3 + took_0;
        point_1 = point_2;
        c_30 = c_30 + u32(1);
    }
    crush_rw_0[bill_2] = vec4<f32>(energy_3, f32(s_37 + u32(1)), 0.0f, 0.0f);
    if(energy_3 > 0.0f)
    {
        crush_rw_0[_S801] = vec4<f32>(point_1 / vec3<f32>(energy_3), 0.0f);
    }
    return energy_3;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_strike_crush(@builtin(global_invocation_id) id_78 : vec3<u32>)
{
    var s_38 : u32 = id_78.x;
    var _S821 : bool;
    if(s_38 >= (params_0.shot_count_0))
    {
        _S821 = true;
    }
    else
    {
        _S821 = (crush_0.enabled_0) == u32(0);
    }
    if(_S821)
    {
        return;
    }
    var sb_4 : u32 = u32(7) * (bitcast<u32>((shots_0[u32(2) * s_38].x)));
    var inv_mass_11 : f32 = bodies_0[sb_4].w;
    var v_19 : vec3<f32> = bodies_0[sb_4 + u32(3)].xyz;
    var budget_2 : f32;
    if(inv_mass_11 > 0.0f)
    {
        budget_2 = 0.5f * dot(v_19, v_19) / inv_mass_11;
    }
    else
    {
        budget_2 = 1.00000001504746622e+30f;
    }
    var b_86 : u32 = u32(0);
    loop
    {
        if(b_86 < (params_0.body_count_0))
        {
        }
        else
        {
            break;
        }
        var shot_13 : u32;
        var _S822 : bool = struck_2(b_86, &(shot_13));
        if(!_S822)
        {
            _S821 = true;
        }
        else
        {
            _S821 = shot_13 != s_38;
        }
        var _S823 : bool;
        if(_S821)
        {
            _S823 = true;
        }
        else
        {
            _S823 = (crush_rw_0[crush_0.r_bill_0 + u32(4) * b_86 + u32(3)].w) == 0.0f;
        }
        if(_S823)
        {
            b_86 = b_86 + u32(1);
            continue;
        }
        var _S824 : f32 = strike_crush_body_0(b_86, s_38, budget_2);
        budget_2 = budget_2 - _S824;
        b_86 = b_86 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_bill(@builtin(global_invocation_id) id_79 : vec3<u32>)
{
    var s_39 : u32 = id_79.x;
    var _S825 : bool;
    if(s_39 >= (params_0.shot_count_0))
    {
        _S825 = true;
    }
    else
    {
        _S825 = (crush_0.enabled_0) == u32(0);
    }
    if(_S825)
    {
        return;
    }
    var b_87 : u32 = u32(0);
    var energy_5 : f32 = 0.0f;
    loop
    {
        if(b_87 < (params_0.body_count_0))
        {
        }
        else
        {
            break;
        }
        var bill_3 : vec4<f32> = crush_rw_0[crush_0.r_bill_0 + u32(4) * b_87];
        if((crush_rw_0[crush_0.r_bill_0 + u32(4) * b_87].y) == f32(s_39 + u32(1)))
        {
            energy_5 = energy_5 + bill_3.x;
        }
        b_87 = b_87 + u32(1);
    }
    crush_rw_0[crush_0.r_shot_0 + s_39] = vec4<f32>(0.0f);
    if(!(energy_5 > 0.0f))
    {
        return;
    }
    var sb_5 : u32 = u32(7) * (bitcast<u32>((shots_0[u32(2) * s_39].x)));
    var inv_mass_12 : f32 = bodies_0[sb_5].w;
    if(!(inv_mass_12 > 0.0f))
    {
        return;
    }
    var m_10 : f32 = 1.0f / inv_mass_12;
    var v_20 : vec3<f32> = bodies_0[sb_5 + u32(3)].xyz;
    var speed_18 : f32 = length(v_20);
    var j_15 : f32 = m_10 * (speed_18 - sqrt(max(speed_18 * speed_18 - 2.0f * energy_5 / m_10, 0.0f)));
    var _S826 : bool = speed_18 > 0.0f;
    var _S827 : vec3<f32>;
    if(_S826)
    {
        _S827 = (vec3<f32>(0) - v_20) * vec3<f32>((j_15 / (m_10 * speed_18)));
    }
    else
    {
        _S827 = vec3<f32>(0.0f);
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
        var w_15 : u32 = sb_5 + region_6 * u32(7) * params_0.max_bodies_0 + u32(3);
        bodies_0[w_15] = vec4<f32>(bodies_0[w_15].xyz + _S827, bodies_0[w_15].w);
        region_6 = region_6 + u32(1);
    }
    if(_S826)
    {
        _S827 = v_20 * vec3<f32>((j_15 / speed_18));
    }
    else
    {
        _S827 = vec3<f32>(0.0f);
    }
    crush_rw_0[crush_0.r_shot_0 + s_39] = vec4<f32>(_S827, energy_5);
    return;
}

fn inverse_rotate_0( q_42 : Quat_0,  v_21 : vec3<f32>) -> vec3<f32>
{
    var c_31 : Quat_0;
    c_31.w_0 = q_42.w_0;
    c_31.x_0 = - q_42.x_0;
    c_31.y_0 = - q_42.y_0;
    c_31.z_0 = - q_42.z_0;
    return rotate_0(c_31, v_21);
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_push(@builtin(global_invocation_id) id_80 : vec3<u32>)
{
    var b_88 : u32 = id_80.x;
    var _S828 : bool;
    if(b_88 >= (params_0.body_count_0))
    {
        _S828 = true;
    }
    else
    {
        _S828 = (params_0.shot_count_0) == u32(0);
    }
    if(_S828)
    {
        _S828 = true;
    }
    else
    {
        _S828 = (crush_0.enabled_0) == u32(0);
    }
    if(_S828)
    {
        return;
    }
    var _S829 : u32 = u32(4) * b_88;
    var bill_4 : vec4<f32> = crush_rw_0[crush_0.r_bill_0 + _S829];
    var _S830 : f32 = crush_rw_0[crush_0.r_bill_0 + _S829].x;
    if(!(_S830 > 0.0f))
    {
        return;
    }
    var paid_0 : vec4<f32> = crush_rw_0[crush_0.r_shot_0 + (u32(bill_4.y) - u32(1))];
    var bw_0 : u32 = u32(7) * b_88;
    var inv_mass_13 : f32 = bodies_0[bw_0].w;
    var _S831 : f32 = crush_rw_0[crush_0.r_shot_0 + (u32(bill_4.y) - u32(1))].w;
    if(!(_S831 > 0.0f))
    {
        _S828 = true;
    }
    else
    {
        _S828 = inv_mass_13 == 0.0f;
    }
    if(_S828)
    {
        return;
    }
    var j_16 : vec3<f32> = paid_0.xyz * vec3<f32>((_S830 / _S831));
    var o_12 : Quat_0 = quat_of_0(bodies_0[bw_0 + u32(1)]);
    var _S832 : vec3<f32> = rotate_0(o_12, bodies_0[bw_0 + u32(2)].xyz * inverse_rotate_0(o_12, cross(to_rubble_0(crush_rw_0[crush_0.r_bill_0 + _S829 + u32(1)].xyz) - bodies_0[bw_0].xyz, j_16)));
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
        var w_16 : u32 = bw_0 + region_7 * u32(7) * params_0.max_bodies_0;
        var _S833 : u32 = w_16 + u32(3);
        bodies_0[_S833] = vec4<f32>(bodies_0[_S833].xyz + j_16 * vec3<f32>(inv_mass_13), bodies_0[_S833].w);
        var _S834 : u32 = w_16 + u32(4);
        bodies_0[_S834] = vec4<f32>(bodies_0[_S834].xyz + _S832, bodies_0[_S834].w);
        region_7 = region_7 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_bonds(@builtin(global_invocation_id) id_81 : vec3<u32>)
{
    var i_23 : u32 = id_81.x;
    var _S835 : bool;
    if(i_23 >= (params_0.bond_count_0))
    {
        _S835 = true;
    }
    else
    {
        _S835 = (crush_0.enabled_0) == u32(0);
    }
    if(_S835)
    {
        _S835 = true;
    }
    else
    {
        _S835 = work_0[w_alive_0() + i_23] == u32(0);
    }
    if(_S835)
    {
        return;
    }
    var bond_10 : vec4<u32> = bonds_0[i_23];
    var _S836 : u32 = bond_10.x;
    if((((((bitcast<u32>((crush_rw_0[_S836].w))) | ((bitcast<u32>((crush_rw_0[bond_10.y].w)))))) & (u32(2)))) == u32(0))
    {
        return;
    }
    work_0[w_alive_0() + i_23] = u32(0);
    var _S837 : u32 = atomicOr(&(atom_0[a_dirty_0() + body_of_0(_S836)]), u32(1));
    var _S838 : u32 = a_counters_0();
    var _S839 : u32 = atomicAdd(&(atom_0[_S838]), u32(1));
    var _S840 : u32 = atomicAdd(&(atom_0[_S838 + u32(1)]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_mark(@builtin(global_invocation_id) id_82 : vec3<u32>)
{
    var c_32 : u32 = id_82.x;
    var _S841 : bool;
    if(c_32 >= (params_0.chunk_count_0))
    {
        _S841 = true;
    }
    else
    {
        _S841 = (crush_0.enabled_0) == u32(0);
    }
    if(_S841)
    {
        return;
    }
    var now_0 : u32;
    if((((crush_flags_0(c_32)) & (u32(2)))) != u32(0))
    {
        now_0 = u32(1);
    }
    else
    {
        now_0 = u32(0);
    }
    work_0[w_new_0() + c_32] = now_0;
    if(now_0 != u32(0))
    {
        var _S842 : u32 = atomicAdd(&(atom_0[a_counters_0() + u32(10)]), u32(1));
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_crush_gate(@builtin(global_invocation_id) id_83 : vec3<u32>)
{
    var run_2 : bool;
    if((crush_0.enabled_0) != u32(0))
    {
        var _S843 : u32 = atomicLoad(&(atom_0[a_counters_0() + u32(10)]));
        run_2 = _S843 != u32(0);
    }
    else
    {
        run_2 = false;
    }
    var _S844 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S844 + u32(10)]), u32(0));
    atomicStore(&(atom_0[_S844 + u32(7)]), u32(0));
    var _S845 : u32;
    if(run_2)
    {
        var _S846 : u32 = groups_of_0(params_0.chunk_count_0, u32(64));
        _S845 = _S846;
    }
    else
    {
        _S845 = u32(0);
    }
    set_args_0(u32(14), _S845);
    if(run_2)
    {
        var _S847 : u32 = groups_of_0(params_0.chunk_count_0, u32(1024));
        _S845 = _S847;
    }
    else
    {
        _S845 = u32(0);
    }
    set_args_0(u32(15), _S845);
    var _S848 : i32;
    if(run_2)
    {
        _S848 = i32(1);
    }
    else
    {
        _S848 = i32(0);
    }
    set_args_0(u32(16), u32(_S848));
    return;
}

fn write_body_0( slot_24 : u32,  pos_1 : vec3<f32>,  inv_mass_14 : f32,  orientation_1 : Quat_0,  inv_inertia_1 : vec3<f32>,  flags_3 : u32,  v_22 : vec3<f32>,  w_17 : vec3<f32>)
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
        var b_89 : u32 = u32(7) * (slot_24 + region_8 * params_0.max_bodies_0);
        bodies_0[b_89] = vec4<f32>(pos_1, inv_mass_14);
        bodies_0[b_89 + u32(1)] = quat_vec_0(orientation_1);
        bodies_0[b_89 + u32(2)] = vec4<f32>(inv_inertia_1, (bitcast<f32>((flags_3))));
        bodies_0[b_89 + u32(3)] = vec4<f32>(v_22, 0.0f);
        bodies_0[b_89 + u32(4)] = vec4<f32>(w_17, (bitcast<f32>((params_0.step_0))));
        bodies_0[b_89 + u32(5)] = vec4<f32>(0.0f);
        bodies_0[b_89 + u32(6)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
        region_8 = region_8 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn fr_crush_apply(@builtin(global_invocation_id) id_84 : vec3<u32>)
{
    var ax_5 : u32;
    var k_74 : u32 = id_84.x;
    var _S849 : bool;
    if(k_74 >= (params_0.chunk_count_0))
    {
        _S849 = true;
    }
    else
    {
        _S849 = (crush_0.enabled_0) == u32(0);
    }
    if(_S849)
    {
        return;
    }
    var cs_0 : vec4<f32> = crush_rw_0[k_74];
    var flags_4 : u32 = (bitcast<u32>((crush_rw_0[k_74].w)));
    if(((flags_4 & (u32(2)))) == u32(0))
    {
        return;
    }
    crush_rw_0[k_74] = vec4<f32>(cs_0.xyz, (bitcast<f32>(((flags_4 & (u32(4294967293)))))));
    var slot_25 : u32 = body_of_0(k_74);
    var bw_1 : u32 = u32(7) * slot_25;
    var b0_2 : vec4<f32> = bodies_0[bw_1];
    var orientation_2 : Quat_0 = quat_of_0(bodies_0[bw_1 + u32(1)]);
    var v_23 : vec3<f32> = bodies_0[bw_1 + u32(3)].xyz;
    var w_18 : vec3<f32> = bodies_0[bw_1 + u32(4)].xyz;
    var _S850 : u32 = u32(4) * k_74;
    var mass_2 : f32 = chunks_0[_S850].w;
    var volume_1 : f32 = crush_tab_0[u32(2) * k_74 + u32(1)].y;
    var rank_0 : u32 = work_0[w_rank_0() + k_74];
    if(work_0[w_rank_0() + k_74] >= (crush_0.max_events_0))
    {
        var _S851 : u32 = atomicOr(&(atom_0[a_counters_0() + u32(3)]), u32(32));
    }
    var _S852 : vec3<f32> = vec3<f32>(0.0f);
    if((crush_0.mode_1) == u32(1))
    {
        _S849 = (crush_0.fragments_0) > u32(0);
    }
    else
    {
        _S849 = false;
    }
    var first_6 : u32;
    var half_2 : vec3<f32>;
    if(_S849)
    {
        var _S853 : u32 = a_counters_0();
        var _S854 : u32 = atomicLoad(&(atom_0[_S853 + u32(8)]));
        var base_4 : u32 = _S854 + rank_0 * crush_0.fragments_0;
        if((base_4 + crush_0.fragments_0) > (crush_0.pool_size_0))
        {
            var _S855 : u32 = atomicOr(&(atom_0[_S853 + u32(3)]), u32(16));
            first_6 = u32(4294967295);
        }
        else
        {
            first_6 = base_4;
        }
        var q_43 : Quat_0;
        var origin_22 : vec3<f32>;
        structure_pose_5(slot_25, bw_1, &(q_43), &(origin_22));
        var _S856 : vec3<f32> = origin_22 + rotate_0(q_43, chunks_0[_S850].xyz);
        var _S857 : u32 = u32(3) * k_74;
        var box_0 : Quat_0 = normalized_0(quat_mul_0(q_43, quat_of_0(geometry_0[_S857 + u32(2)])));
        var h_7 : vec3<f32> = geometry_0[_S857 + u32(1)].xyz;
        var g_0 : vec3<u32> = vec3<u32>(vec3<i32>(i32(1)));
        var cell_0 : vec3<f32> = h_7;
        var n_9 : u32 = crush_0.fragments_0;
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
                _S849 = (cell_0.x) >= (cell_0.z);
            }
            else
            {
                _S849 = false;
            }
            if(_S849)
            {
                ax_5 = u32(0);
            }
            else
            {
                if((cell_0.y) >= (cell_0.z))
                {
                    ax_5 = u32(1);
                }
                else
                {
                    ax_5 = u32(2);
                }
            }
            g_0[ax_5] = g_0[ax_5] * u32(2);
            cell_0[ax_5] = cell_0[ax_5] * 0.5f;
            n_9 = (n_9 >> (u32(1)));
        }
        var box_volume_0 : f32 = 8.0f * h_7.x * h_7.y * h_7.z;
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
        var _S858 : f32 = half_3.y;
        var _S859 : f32 = _S858 * _S858;
        var _S860 : f32 = half_3.z;
        var _S861 : f32 = _S860 * _S860;
        var _S862 : f32 = half_3.x;
        var _S863 : f32 = _S862 * _S862;
        var inertia_0 : vec3<f32> = vec3<f32>((mf_0 / 3.0f)) * vec3<f32>(_S859 + _S861, _S863 + _S861, _S863 + _S859);
        var _S864 : vec3<f32> = vec3<f32>(1.0f / inertia_0.x, 1.0f / inertia_0.y, 1.0f / inertia_0.z);
        var _S865 : Quat_0 = rot_to_rubble_0(box_0);
        var _S866 : f32 = chunks_0[_S850 + u32(3)].x;
        var _S867 : f32 = min(_S862, min(_S858, _S860));
        ax_5 = u32(0);
        loop
        {
            if(ax_5 < (crush_0.fragments_0))
            {
                _S849 = first_6 != u32(4294967295);
            }
            else
            {
                _S849 = false;
            }
            if(_S849)
            {
            }
            else
            {
                break;
            }
            var _S868 : u32 = ax_5 % g_0.x;
            var _S869 : u32 = ax_5 / g_0.x;
            var _S870 : u32 = _S869 % g_0.y;
            var _S871 : u32 = ax_5 / (g_0.x * g_0.y);
            var x_10 : vec3<f32> = to_rubble_0(_S856 + rotate_0(box_0, (vec3<f32>(0) - h_7) + cell_0 * (vec3<f32>(2.0f) * vec3<f32>(vec3<u32>(_S868, _S870, _S871)) + vec3<f32>(1.0f))));
            var entry_9 : vec4<f32> = crush_tab_0[crush_0.t_pool_0 + first_6 + ax_5];
            var fs_0 : u32 = (bitcast<u32>((entry_9.x)));
            var fc_3 : u32 = (bitcast<u32>((entry_9.y)));
            write_body_0(fs_0, x_10, 1.0f / mf_0, _S865, _S864, u32(9), v_23 + cross(w_18, x_10 - b0_2.xyz), w_18);
            var _S872 : u32 = u32(4) * fc_3;
            var _S873 : u32 = _S872 + u32(1);
            colliders_0[_S873] = vec4<f32>(0.0f, 0.0f, 0.0f, colliders_0[_S873].w);
            colliders_0[_S872 + u32(2)] = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
            colliders_0[_S872 + u32(3)] = vec4<f32>(half_3, length(half_3));
            var _S874 : u32 = u32(2) * fc_3;
            collider_info_0[_S874] = vec4<u32>(u32(4294967295), fs_0, u32(2), collider_info_0[_S874].w);
            collider_info_0[_S874 + u32(1)] = vec4<u32>((bitcast<u32>((mf_0))), (bitcast<u32>((_S866))), (bitcast<u32>((_S867))), u32(0));
            ax_5 = ax_5 + u32(1);
        }
        half_2 = half_3;
    }
    else
    {
        first_6 = u32(4294967295);
        half_2 = _S852;
    }
    if(rank_0 < (crush_0.max_events_0))
    {
        var ev_0 : u32 = crush_0.r_events_0 + u32(6) * rank_0;
        crush_rw_0[ev_0] = vec4<f32>((bitcast<f32>((k_74))), (bitcast<f32>((first_6))), (bitcast<f32>(((flags_4 >> (u32(4)))))), cs_0.x);
        crush_rw_0[ev_0 + u32(1)] = vec4<f32>(b0_2.xyz, mass_2);
        crush_rw_0[ev_0 + u32(2)] = quat_vec_0(orientation_2);
        crush_rw_0[ev_0 + u32(3)] = vec4<f32>(v_23, cs_0.y);
        crush_rw_0[ev_0 + u32(4)] = vec4<f32>(w_18, cs_0.z);
        crush_rw_0[ev_0 + u32(5)] = vec4<f32>(half_2, volume_1);
    }
    write_body_0(slot_25, vec3<f32>(f32(slot_25 % u32(1024)) * crush_0.park_step_0, crush_0.park_y_0, f32(slot_25 / u32(1024)) * crush_0.park_step_0), 0.0f, orientation_2, _S852, u32(0), _S852, _S852);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn fr_crush_end(@builtin(global_invocation_id) id_85 : vec3<u32>)
{
    var total_3 : u32 = work_0[scan_new_blocks_0() + (params_0.chunk_count_0 + u32(1023)) / u32(1024)];
    var _S875 : u32 = a_counters_0();
    atomicStore(&(atom_0[_S875 + u32(7)]), work_0[scan_new_blocks_0() + (params_0.chunk_count_0 + u32(1023)) / u32(1024)]);
    var _S876 : u32 = atomicAdd(&(atom_0[_S875 + u32(9)]), total_3);
    if((crush_0.mode_1) == u32(1))
    {
        var _S877 : u32 = atomicAdd(&(atom_0[_S875 + u32(8)]), total_3 * crush_0.fragments_0);
    }
    return;
}

