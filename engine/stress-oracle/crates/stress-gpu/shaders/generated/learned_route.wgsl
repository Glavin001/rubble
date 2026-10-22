@binding(10) @group(0) var<storage, read> counters_0 : array<u32>;

struct RouteParams_std140_0
{
    @align(16) island_cap_0 : u32,
    @align(4) impactor_count_0 : u32,
    @align(8) engine_base_0 : u32,
    @align(4) n_pad_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) speed_0 : f32,
    @align(8) tiered_0 : u32,
    @align(4) fast_joins_0 : u32,
    @align(16) debris_0 : u32,
    @align(4) ground_0 : u32,
    @align(8) ground_height_0 : f32,
    @align(4) ground_modulus_0 : f32,
    @align(16) edge_cap_0 : u32,
    @align(4) always_0 : u32,
    @align(8) pad0_0 : u32,
    @align(4) pad1_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : RouteParams_std140_0;
@binding(11) @group(0) var<storage, read> route_0 : array<u32>;

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

@binding(5) @group(0) var<storage, read> bodies_0 : array<vec4<f32>>;

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

@binding(8) @group(0) var<storage, read> rtab_0 : array<u32>;

struct JointBond_std430_0
{
    @align(16) geom0_0 : vec4<f32>,
    @align(16) geom1_0 : vec4<f32>,
    @align(16) stiff0_0 : vec4<f32>,
    @align(16) stiff1_0 : vec4<f32>,
    @align(16) rebar0_0 : vec4<f32>,
    @align(16) rebar1_0 : vec4<f32>,
    @align(16) ids_0 : vec4<u32>,
};

struct BondStatic_std430_0
{
    @align(16) t1_0 : vec4<f32>,
    @align(16) t2_0 : vec4<f32>,
    @align(16) normal_0 : vec4<f32>,
    @align(16) ra_0 : vec4<f32>,
    @align(16) rb_0 : vec4<f32>,
    @align(16) centroid_0 : vec4<f32>,
    @align(16) c_lin_0 : vec4<f32>,
    @align(16) c_ang_0 : vec4<f32>,
    @align(16) law_0 : JointBond_std430_0,
};

@binding(6) @group(0) var<storage, read> bond_static_0 : array<BondStatic_std430_0>;

struct JointMaterial_std140_0
{
    @align(16) strength_0 : vec4<f32>,
    @align(16) energy_1 : vec4<f32>,
    @align(16) dif_0 : vec4<f32>,
    @align(16) misc_0 : vec4<f32>,
    @align(16) kind_flags_0 : vec4<u32>,
};

struct _Array_std140_JointMaterial64_0
{
    @align(16) data_0 : array<JointMaterial_std140_0, i32(64)>,
};

struct MaterialTable_std140_0
{
    @align(16) m_0 : _Array_std140_JointMaterial64_0,
};

@binding(7) @group(0) var<uniform> materials_0 : MaterialTable_std140_0;
@binding(12) @group(0) var<storage, read_write> parts_0 : array<vec4<f32>>;

@binding(9) @group(0) var<storage, read> imps_0 : array<vec4<f32>>;

@binding(16) @group(0) var<storage, read_write> actl_0 : array<atomic<u32>>;

@binding(13) @group(0) var<storage, read_write> sorted_0 : array<vec2<u32>>;

@binding(15) @group(0) var<storage, read_write> lab_0 : array<atomic<u32>>;

@binding(17) @group(0) var<storage, read_write> seed_0 : array<u32>;

@binding(19) @group(0) var<storage, read> pass_0 : array<vec4<u32>>;

@binding(14) @group(0) var<storage, read_write> edges_0 : array<vec2<u32>>;

@binding(18) @group(0) var<storage, read_write> isl_0 : array<vec4<u32>>;

@binding(20) @group(0) var<storage, read_write> ctl_0 : array<u32>;

fn live_islands_0() -> u32
{
    return counters_0[i32(0)];
}

fn gate_0() -> bool
{
    var _S1 : bool;
    if((params_0.always_0) != u32(0))
    {
        _S1 = true;
    }
    else
    {
        _S1 = route_0[i32(0)] != u32(0);
    }
    return _S1;
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

fn quat_mul_0( a_0 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_1 : Quat_0;
    r_1.w_0 = a_0.w_0 * o_0.w_0 - a_0.x_0 * o_0.x_0 - a_0.y_0 * o_0.y_0 - a_0.z_0 * o_0.z_0;
    r_1.x_0 = a_0.w_0 * o_0.x_0 + a_0.x_0 * o_0.w_0 + a_0.y_0 * o_0.z_0 - a_0.z_0 * o_0.y_0;
    r_1.y_0 = a_0.w_0 * o_0.y_0 - a_0.x_0 * o_0.z_0 + a_0.y_0 * o_0.w_0 + a_0.z_0 * o_0.x_0;
    r_1.z_0 = a_0.w_0 * o_0.z_0 + a_0.x_0 * o_0.y_0 - a_0.y_0 * o_0.x_0 + a_0.z_0 * o_0.w_0;
    return r_1;
}

fn quat_vec_0( q_1 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_1.x_0, q_1.y_0, q_1.z_0, q_1.w_0);
}

fn qmul_0( a_1 : vec4<f32>,  b_0 : vec4<f32>) -> vec4<f32>
{
    return quat_vec_0(quat_mul_0(quat_of_0(a_1), quat_of_0(b_0)));
}

fn to_zup_0( v_0 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_0.x, - v_0.z, v_0.y);
}

fn rotate_0( q_2 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_2.x_0, q_2.y_0, q_2.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_1) * vec3<f32>(2.0f);
    return v_1 + t_0 * vec3<f32>(q_2.w_0) + cross(qv_0, t_0);
}

struct Frame_0
{
     q_3 : vec4<f32>,
     origin_0 : vec3<f32>,
     com_1 : vec3<f32>,
     v_2 : vec3<f32>,
     w_1 : vec3<f32>,
};

fn island_frame_0( i_0 : u32) -> Frame_0
{
    var _S2 : u32 = u32(3) * i_0;
    var body_0 : u32 = (bitcast<u32>((ctab_0[params_0.engine_base_0 + _S2].x)));
    var f_0 : Frame_0;
    f_0.q_3 = islands_0[i_0].rotation_0;
    f_0.origin_0 = islands_0[i_0].position_0.xyz;
    f_0.v_2 = islands_0[i_0].velocity_0.xyz;
    f_0.w_1 = islands_0[i_0].angular_velocity_0.xyz;
    if(body_0 != u32(4294967295))
    {
        var com_2 : vec3<f32> = ctab_0[params_0.engine_base_0 + _S2 + u32(2)].xyz;
        var _S3 : u32 = u32(7) * body_0;
        var _S4 : vec4<f32> = normalize(qmul_0(qmul_0(vec4<f32>(0.70710676908493042f, 0.0f, 0.0f, 0.70710676908493042f), bodies_0[_S3 + u32(1)]), ctab_0[params_0.engine_base_0 + _S2 + u32(1)]));
        f_0.q_3 = _S4;
        var _S5 : vec3<f32> = to_zup_0(bodies_0[_S3].xyz);
        f_0.com_1 = _S5;
        f_0.origin_0 = _S5 - rotate_0(quat_of_0(_S4), com_2);
        f_0.v_2 = to_zup_0(bodies_0[_S3 + u32(3)].xyz);
        f_0.w_1 = to_zup_0(bodies_0[_S3 + u32(4)].xyz);
    }
    else
    {
        f_0.com_1 = f_0.origin_0 + rotate_0(quat_of_0(f_0.q_3), islands_0[i_0].com_0.xyz);
    }
    return f_0;
}

fn float_inf_0() -> f32
{
    return (bitcast<f32>((u32(2139095040))));
}

var<workgroup> g_lo_0 : array<vec3<f32>, i32(64)>;

var<workgroup> g_hi_0 : array<vec3<f32>, i32(64)>;

var<workgroup> g_a_0 : array<vec4<f32>, i32(64)>;

var<workgroup> g_cap_0 : array<f32, i32(64)>;

struct Part_0
{
     lo_0 : vec3<f32>,
     hi_0 : vec3<f32>,
     box_lo_0 : vec3<f32>,
     box_hi_0 : vec3<f32>,
     center_1 : vec3<f32>,
     v_3 : vec3<f32>,
     w_2 : vec3<f32>,
     reach_0 : f32,
     mass_0 : f32,
     modulus_0 : f32,
     radius_0 : f32,
     round_trip_0 : f32,
     capacity_0 : f32,
     flags_0 : u32,
};

fn store_part_0( i_1 : u32,  p_0 : Part_0)
{
    var _S6 : u32 = u32(8) * i_1;
    parts_0[_S6] = vec4<f32>(p_0.lo_0, (bitcast<f32>((p_0.flags_0))));
    parts_0[_S6 + u32(1)] = vec4<f32>(p_0.hi_0, p_0.mass_0);
    parts_0[_S6 + u32(2)] = vec4<f32>(p_0.box_lo_0, p_0.reach_0);
    parts_0[_S6 + u32(3)] = vec4<f32>(p_0.box_hi_0, p_0.modulus_0);
    parts_0[_S6 + u32(4)] = vec4<f32>(p_0.center_1, p_0.radius_0);
    parts_0[_S6 + u32(5)] = vec4<f32>(p_0.v_3, p_0.round_trip_0);
    parts_0[_S6 + u32(6)] = vec4<f32>(p_0.w_2, p_0.capacity_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn lr_islands(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var _S7 : u32;
    var i_2 : u32 = group_0.x;
    var t_1 : u32 = thread_0.x;
    var _S8 : bool;
    if(i_2 >= (live_islands_0()))
    {
        _S8 = true;
    }
    else
    {
        _S8 = !gate_0();
    }
    if(_S8)
    {
        return;
    }
    var f_1 : Frame_0 = island_frame_0(i_2);
    var _S9 : vec4<u32> = islands_0[i_2].range_0;
    var _S10 : u32 = islands_0[i_2].range_0.x;
    var n_0 : u32 = islands_0[i_2].range_0.y - _S10;
    var inf_0 : f32 = float_inf_0();
    var _S11 : vec3<f32> = vec3<f32>(inf_0);
    var _S12 : vec3<f32> = vec3<f32>(- inf_0);
    var a_2 : vec4<f32> = vec4<f32>(0.0f, inf_0, inf_0, inf_0);
    var lo_1 : vec3<f32> = _S11;
    var hi_1 : vec3<f32> = _S12;
    var c_0 : u32 = t_1;
    loop
    {
        if(c_0 < n_0)
        {
        }
        else
        {
            break;
        }
        var k_0 : u32 = _S10 + c_0;
        var p_1 : vec3<f32> = f_1.origin_0 + rotate_0(quat_of_0(f_1.q_3), chunks_0[k_0].center_0.xyz + state_0[u32(4) * k_0].xyz);
        var r_2 : f32 = chunks_0[k_0].half_0.w;
        var _S13 : vec3<f32> = vec3<f32>(r_2);
        var _S14 : vec3<f32> = min(lo_1, p_1 - _S13);
        var _S15 : vec3<f32> = max(hi_1, p_1 + _S13);
        a_2[i32(0)] = max(a_2.x, length(p_1 - f_1.com_1) + r_2);
        a_2[i32(1)] = min(a_2.y, chunks_0[k_0].cmat_0.x);
        a_2[i32(2)] = min(a_2.z, (bitcast<f32>((rtab_0[u32(2) + rtab_0[i32(0)] + rtab_0[u32(2) + chunks_0[k_0].info_1.z] + chunks_0[k_0].info_1.w]))));
        a_2[i32(3)] = min(a_2.w, min(chunks_0[k_0].half_0.x, min(chunks_0[k_0].half_0.y, chunks_0[k_0].half_0.z)));
        var c_1 : u32 = c_0 + u32(64);
        lo_1 = _S14;
        hi_1 = _S15;
        c_0 = c_1;
    }
    var _S16 : u32 = _S9.z;
    var _S17 : u32 = _S16 + t_1;
    var cap_0 : f32 = inf_0;
    var b_1 : u32 = _S17;
    loop
    {
        var _S18 : u32 = _S9.w;
        _S7 = _S18;
        if(b_1 < _S18)
        {
        }
        else
        {
            break;
        }
        var _S19 : f32 = min(cap_0, materials_0.m_0.data_0[bond_static_0[b_1].law_0.ids_0.x].strength_0.x * bond_static_0[b_1].law_0.geom0_0.x * bond_static_0[b_1].law_0.geom1_0.w);
        var b_2 : u32 = b_1 + u32(64);
        cap_0 = _S19;
        b_1 = b_2;
    }
    workgroupBarrier();
    g_lo_0[t_1] = lo_1;
    g_hi_0[t_1] = hi_1;
    g_a_0[t_1] = a_2;
    g_cap_0[t_1] = cap_0;
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
            var _S20 : u32 = t_1 + s_0;
            g_lo_0[t_1] = min(g_lo_0[t_1], g_lo_0[_S20]);
            g_hi_0[t_1] = max(g_hi_0[t_1], g_hi_0[_S20]);
            g_a_0[t_1] = vec4<f32>(max(g_a_0[t_1].x, g_a_0[_S20].x), min(g_a_0[t_1].yzw, g_a_0[_S20].yzw));
            g_cap_0[t_1] = min(g_cap_0[t_1], g_cap_0[_S20]);
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    if(t_1 != u32(0))
    {
        return;
    }
    var p_2 : Part_0;
    p_2.flags_0 = u32(0);
    if(n_0 == u32(0))
    {
        store_part_0(i_2, p_2);
        return;
    }
    var lo_2 : vec3<f32> = g_lo_0[i32(0)];
    var hi_2 : vec3<f32> = g_hi_0[i32(0)];
    a_2 = g_a_0[i32(0)];
    var anchored_0 : bool = (((islands_0[i_2].info_0.x) & (u32(1)))) != u32(0);
    var _S21 : f32 = max(g_hi_0[i32(0)].x - g_lo_0[i32(0)].x, max(g_hi_0[i32(0)].y - g_lo_0[i32(0)].y, g_hi_0[i32(0)].z - g_lo_0[i32(0)].z));
    var sweep_0 : vec3<f32> = abs(f_1.v_2) * vec3<f32>(params_0.dt_0) + vec3<f32>((length(f_1.w_1) * g_a_0[i32(0)].x * params_0.dt_0));
    if(_S7 > _S16)
    {
        c_0 = u32(4);
    }
    else
    {
        c_0 = u32(0);
    }
    var _S22 : u32 = (u32(1) | (c_0));
    if(anchored_0)
    {
        c_0 = u32(16);
    }
    else
    {
        c_0 = u32(0);
    }
    p_2.flags_0 = (_S22 | (c_0));
    p_2.lo_0 = lo_2 - sweep_0;
    p_2.hi_0 = hi_2 + sweep_0;
    p_2.box_lo_0 = lo_2;
    p_2.box_hi_0 = hi_2;
    p_2.center_1 = f_1.com_1;
    p_2.v_3 = f_1.v_2;
    p_2.w_2 = f_1.w_1;
    p_2.reach_0 = a_2.x;
    if(anchored_0)
    {
        cap_0 = inf_0;
    }
    else
    {
        cap_0 = islands_0[i_2].com_0.w;
    }
    p_2.mass_0 = cap_0;
    p_2.modulus_0 = a_2.y;
    p_2.radius_0 = a_2.w;
    p_2.round_trip_0 = 2.0f * _S21 / a_2.z;
    p_2.capacity_0 = g_cap_0[i32(0)];
    store_part_0(i_2, p_2);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn lr_impactors(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var k_1 : u32 = id_0.x;
    var _S23 : bool;
    if(k_1 >= (min(params_0.impactor_count_0, u32(32))))
    {
        _S23 = true;
    }
    else
    {
        _S23 = !gate_0();
    }
    if(_S23)
    {
        return;
    }
    var _S24 : u32 = u32(4) * k_1;
    var a_3 : vec4<f32> = imps_0[_S24];
    var b_3 : vec4<f32> = imps_0[_S24 + u32(1)];
    var r_3 : f32 = a_3.y;
    var _S25 : u32 = u32(7) * (bitcast<u32>((a_3.x)));
    var c_2 : vec3<f32> = to_zup_0(bodies_0[_S25].xyz);
    var v_4 : vec3<f32> = to_zup_0(bodies_0[_S25 + u32(3)].xyz);
    var w_3 : vec3<f32> = to_zup_0(bodies_0[_S25 + u32(4)].xyz);
    var sweep_1 : vec3<f32> = abs(v_4) * vec3<f32>(params_0.dt_0) + vec3<f32>((length(w_3) * r_3 * params_0.dt_0));
    var p_3 : Part_0;
    p_3.flags_0 = u32(3);
    var _S26 : vec3<f32> = vec3<f32>(r_3);
    var _S27 : vec3<f32> = c_2 - _S26;
    p_3.lo_0 = _S27 - sweep_1;
    var _S28 : vec3<f32> = c_2 + _S26;
    p_3.hi_0 = _S28 + sweep_1;
    p_3.box_lo_0 = _S27;
    p_3.box_hi_0 = _S28;
    p_3.center_1 = c_2;
    p_3.v_3 = v_4;
    p_3.w_2 = w_3;
    p_3.reach_0 = r_3;
    p_3.mass_0 = a_3.w;
    p_3.modulus_0 = b_3.w;
    p_3.radius_0 = a_3.z;
    p_3.round_trip_0 = 0.0f;
    p_3.capacity_0 = 0.0f;
    store_part_0(live_islands_0() + k_1, p_3);
    return;
}

fn load_part_0( i_3 : u32) -> Part_0
{
    var _S29 : u32 = u32(8) * i_3;
    var p_4 : Part_0;
    p_4.lo_0 = parts_0[_S29].xyz;
    p_4.flags_0 = (bitcast<u32>((parts_0[_S29].w)));
    p_4.hi_0 = parts_0[_S29 + u32(1)].xyz;
    p_4.mass_0 = parts_0[_S29 + u32(1)].w;
    p_4.box_lo_0 = parts_0[_S29 + u32(2)].xyz;
    p_4.reach_0 = parts_0[_S29 + u32(2)].w;
    p_4.box_hi_0 = parts_0[_S29 + u32(3)].xyz;
    p_4.modulus_0 = parts_0[_S29 + u32(3)].w;
    p_4.center_1 = parts_0[_S29 + u32(4)].xyz;
    p_4.radius_0 = parts_0[_S29 + u32(4)].w;
    p_4.v_3 = parts_0[_S29 + u32(5)].xyz;
    p_4.round_trip_0 = parts_0[_S29 + u32(5)].w;
    p_4.w_2 = parts_0[_S29 + u32(6)].xyz;
    p_4.capacity_0 = parts_0[_S29 + u32(6)].w;
    return p_4;
}

fn overlaps_0( a_4 : Part_0,  b_4 : Part_0) -> bool
{
    var _S30 : bool;
    if((all(((a_4.lo_0) <= (b_4.hi_0)))))
    {
        _S30 = (all(((b_4.lo_0) <= (a_4.hi_0))));
    }
    else
    {
        _S30 = false;
    }
    return _S30;
}

fn closing_speed_0( a_5 : Part_0,  b_5 : Part_0) -> f32
{
    var spin_0 : f32 = length(a_5.w_2) * a_5.reach_0 + length(b_5.w_2) * b_5.reach_0;
    var rel_0 : vec3<f32> = a_5.v_3 - b_5.v_3;
    var n0_0 : vec3<f32> = clamp(a_5.center_1, b_5.box_lo_0, b_5.box_hi_0) - a_5.center_1;
    var n1_0 : vec3<f32> = clamp(b_5.center_1, a_5.box_lo_0, a_5.box_hi_0) - b_5.center_1;
    var l0_0 : f32 = length(n0_0);
    var l1_0 : f32 = length(n1_0);
    var _S31 : bool;
    if(!(l0_0 > 0.0f))
    {
        _S31 = true;
    }
    else
    {
        _S31 = !(l1_0 > 0.0f);
    }
    if(_S31)
    {
        return length(rel_0) + spin_0;
    }
    return max(dot(rel_0, n0_0 / vec3<f32>(l0_0)), dot(rel_0, (vec3<f32>(0) - n1_0) / vec3<f32>(l1_0))) + spin_0;
}

@compute
@workgroup_size(64, 1, 1)
fn lr_debris(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var c_3 : u32 = id_1.x;
    var _S32 : bool;
    if((params_0.debris_0) == u32(0))
    {
        _S32 = true;
    }
    else
    {
        _S32 = c_3 >= (live_islands_0());
    }
    if(_S32)
    {
        _S32 = true;
    }
    else
    {
        _S32 = !gate_0();
    }
    if(_S32)
    {
        return;
    }
    var pc_0 : Part_0 = load_part_0(c_3);
    if((((pc_0.flags_0) & (u32(1)))) == u32(0))
    {
        _S32 = true;
    }
    else
    {
        _S32 = (((pc_0.flags_0) & (u32(4)))) == u32(0);
    }
    if(_S32)
    {
        return;
    }
    var _S33 : vec3<f32> = vec3<f32>(0.0f);
    var _S34 : u32 = live_islands_0();
    var g0_0 : vec3<f32> = _S33;
    var g1_0 : vec3<f32> = _S33;
    var k_2 : u32 = u32(0);
    loop
    {
        if(k_2 < (min(params_0.impactor_count_0, u32(32))))
        {
        }
        else
        {
            break;
        }
        var pk_0 : Part_0 = load_part_0(_S34 + k_2);
        if(!overlaps_0(pk_0, pc_0))
        {
            _S32 = true;
        }
        else
        {
            _S32 = !((closing_speed_0(pk_0, pc_0)) > (params_0.speed_0));
        }
        if(_S32)
        {
            k_2 = k_2 + u32(1);
            continue;
        }
        var d_0 : vec3<f32> = (pk_0.v_3 - pc_0.v_3) * vec3<f32>((2.0f * params_0.dt_0));
        var _S35 : vec3<f32> = max(g1_0, d_0);
        g0_0 = min(g0_0, d_0);
        g1_0 = _S35;
        k_2 = k_2 + u32(1);
    }
    if((any((g0_0 != _S33))))
    {
        _S32 = true;
    }
    else
    {
        _S32 = (any((g1_0 != _S33)));
    }
    if(_S32)
    {
        var _S36 : u32 = u32(8) * c_3;
        parts_0[_S36] = vec4<f32>(pc_0.lo_0 + g0_0, (bitcast<f32>((((pc_0.flags_0) | (u32(8)))))));
        parts_0[_S36 + u32(1)] = vec4<f32>(pc_0.hi_0 + g1_0, pc_0.mass_0);
    }
    return;
}

fn part_count_0() -> u32
{
    return live_islands_0() + min(params_0.impactor_count_0, u32(32));
}

fn sortable_0( x_1 : f32) -> u32
{
    var u_0 : u32 = (bitcast<u32>((x_1)));
    var _S37 : u32;
    if(((u_0 & (u32(2147483648)))) != u32(0))
    {
        _S37 = ~u_0;
    }
    else
    {
        _S37 = (u_0 | (u32(2147483648)));
    }
    return _S37;
}

@compute
@workgroup_size(64, 1, 1)
fn lr_keys(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var s_1 : u32 = id_2.x;
    if(s_1 >= (params_0.n_pad_0))
    {
        return;
    }
    if(s_1 < u32(56))
    {
        atomicStore(&(actl_0[s_1]), u32(0));
    }
    var _S38 : bool = s_1 < (part_count_0());
    var used_0 : bool;
    if(_S38)
    {
        used_0 = gate_0();
    }
    else
    {
        used_0 = false;
    }
    if(used_0)
    {
        used_0 = (((bitcast<u32>((parts_0[u32(8) * s_1].w))) & (u32(1)))) != u32(0);
    }
    else
    {
        used_0 = false;
    }
    var _S39 : u32;
    if(used_0)
    {
        _S39 = sortable_0(parts_0[u32(8) * s_1].x);
    }
    else
    {
        _S39 = u32(4294967295);
    }
    var _S40 : u32;
    if(used_0)
    {
        _S40 = s_1;
    }
    else
    {
        _S40 = u32(4294967295);
    }
    sorted_0[s_1] = vec2<u32>(_S39, _S40);
    if(_S38)
    {
        atomicStore(&(lab_0[s_1]), s_1);
        seed_0[s_1] = u32(0);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn lr_bitonic(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var kj_0 : vec4<u32> = pass_0[i32(0)];
    var k_3 : u32 = kj_0.x;
    var j_0 : u32 = kj_0.y;
    var t_2 : u32 = id_3.x;
    if(t_2 >= (params_0.n_pad_0 / u32(2)))
    {
        return;
    }
    var _S41 : u32 = j_0 - u32(1);
    var lo_3 : u32 = (((((t_2 & ((~_S41)))) << (u32(1)))) | (((t_2 & (_S41)))));
    var hi_3 : u32 = (lo_3 | (j_0));
    var a_6 : vec2<u32> = sorted_0[lo_3];
    var b_6 : vec2<u32> = sorted_0[hi_3];
    var swap_0 : bool;
    if(((lo_3 & (k_3))) == u32(0))
    {
        var _S42 : u32 = a_6.x;
        var _S43 : u32 = b_6.x;
        if(_S42 > _S43)
        {
            swap_0 = true;
        }
        else
        {
            if(_S42 == _S43)
            {
                swap_0 = (a_6.y) > (b_6.y);
            }
            else
            {
                swap_0 = false;
            }
        }
    }
    else
    {
        var _S44 : u32 = a_6.x;
        var _S45 : u32 = b_6.x;
        if(_S44 < _S45)
        {
            swap_0 = true;
        }
        else
        {
            if(_S44 == _S45)
            {
                swap_0 = (a_6.y) < (b_6.y);
            }
            else
            {
                swap_0 = false;
            }
        }
    }
    if(swap_0)
    {
        sorted_0[lo_3] = b_6;
        sorted_0[hi_3] = a_6;
    }
    return;
}

fn hertz_duration_0( m_1 : f32,  r_4 : f32,  e_0 : f32,  v_5 : f32) -> f32
{
    return 2.86999988555908203f * pow(m_1 * m_1 / (r_4 * e_0 * e_0 * max(abs(v_5), 0.00100000004749745f)), 0.20000000298023224f);
}

fn rigid_fails_0( a_7 : Part_0,  b_7 : Part_0,  speed_1 : f32) -> bool
{
    if((((b_7.flags_0) & (u32(4)))) == u32(0))
    {
        return false;
    }
    var fa_0 : bool = (((a_7.flags_0) & (u32(16)))) == u32(0);
    var fb_0 : bool = (((b_7.flags_0) & (u32(16)))) == u32(0);
    var _S46 : bool;
    if(fa_0)
    {
        _S46 = fb_0;
    }
    else
    {
        _S46 = false;
    }
    var m_2 : f32;
    if(_S46)
    {
        m_2 = a_7.mass_0 * b_7.mass_0 / (a_7.mass_0 + b_7.mass_0);
    }
    else
    {
        if(fa_0)
        {
            m_2 = a_7.mass_0;
        }
        else
        {
            if(fb_0)
            {
            }
            else
            {
                return false;
            }
            m_2 = b_7.mass_0;
        }
    }
    if(!(speed_1 > 0.0f))
    {
        return false;
    }
    var duration_0 : f32 = hertz_duration_0(m_2, min(a_7.radius_0, b_7.radius_0), 1.0f / (1.0f / a_7.modulus_0 + 1.0f / b_7.modulus_0), speed_1);
    var peak_0 : f32 = 3.14159274101257324f * (2.0f * m_2 * speed_1) / (2.0f * duration_0);
    if(duration_0 < (b_7.round_trip_0))
    {
        _S46 = true;
    }
    else
    {
        _S46 = peak_0 > (b_7.capacity_0);
    }
    return _S46;
}

fn ground_owned_0( p_5 : Part_0) -> bool
{
    var speed_2 : f32 = p_5.v_3.z * -1.0f + length(p_5.w_2) * p_5.reach_0;
    var g_0 : Part_0;
    g_0.flags_0 = u32(17);
    var _S47 : f32 = float_inf_0();
    g_0.mass_0 = _S47;
    g_0.modulus_0 = params_0.ground_modulus_0;
    g_0.radius_0 = _S47;
    return rigid_fails_0(g_0, p_5, speed_2);
}

@compute
@workgroup_size(64, 1, 1)
fn lr_sweep(@builtin(global_invocation_id) id_4 : vec3<u32>)
{
    var s_2 : u32 = id_4.x;
    var fast_0 : bool;
    if(s_2 >= (params_0.n_pad_0))
    {
        fast_0 = true;
    }
    else
    {
        fast_0 = !gate_0();
    }
    if(fast_0)
    {
        return;
    }
    var ia_0 : u32 = sorted_0[s_2].y;
    if(ia_0 == u32(4294967295))
    {
        return;
    }
    var a_8 : Part_0 = load_part_0(ia_0);
    var tiered_1 : bool = (params_0.tiered_0) != u32(0);
    if((params_0.ground_0) != u32(0))
    {
        fast_0 = (a_8.lo_0.z) <= (params_0.ground_height_0);
    }
    else
    {
        fast_0 = false;
    }
    if(fast_0)
    {
        if(!tiered_1)
        {
            fast_0 = ground_owned_0(a_8);
        }
        else
        {
            fast_0 = false;
        }
        if(fast_0)
        {
            seed_0[ia_0] = u32(1);
        }
    }
    var q_4 : u32 = s_2 + u32(1);
    loop
    {
        if(q_4 < (params_0.n_pad_0))
        {
        }
        else
        {
            break;
        }
        var ib_0 : u32 = sorted_0[q_4].y;
        if(ib_0 == u32(4294967295))
        {
            break;
        }
        var b_8 : Part_0 = load_part_0(ib_0);
        if((b_8.lo_0.x) > (a_8.hi_0.x))
        {
            break;
        }
        if(!overlaps_0(a_8, b_8))
        {
            q_4 = q_4 + u32(1);
            continue;
        }
        var _S48 : u32 = ((a_8.flags_0) | ((b_8.flags_0)));
        var impactor_0 : bool = ((_S48 & (u32(2)))) != u32(0);
        var sab_0 : f32 = closing_speed_0(a_8, b_8);
        var sba_0 : f32 = closing_speed_0(b_8, a_8);
        var _S49 : bool = !tiered_1;
        if(_S49)
        {
            fast_0 = true;
        }
        else
        {
            fast_0 = (max(sab_0, sba_0)) > (params_0.speed_0);
        }
        var _S50 : bool;
        if(_S49)
        {
            _S50 = true;
        }
        else
        {
            _S50 = impactor_0;
        }
        var _S51 : bool;
        if(_S50)
        {
            _S51 = fast_0;
        }
        else
        {
            _S51 = false;
        }
        var owned_0 : bool;
        if(_S51)
        {
            if(rigid_fails_0(a_8, b_8, sab_0))
            {
                owned_0 = true;
            }
            else
            {
                owned_0 = rigid_fails_0(b_8, a_8, sba_0);
            }
        }
        else
        {
            owned_0 = false;
        }
        var both_immovable_0 : bool;
        if((((a_8.flags_0) & (u32(16)))) != u32(0))
        {
            both_immovable_0 = (((b_8.flags_0) & (u32(16)))) != u32(0);
        }
        else
        {
            both_immovable_0 = false;
        }
        if(both_immovable_0)
        {
            q_4 = q_4 + u32(1);
            continue;
        }
        var _S52 : bool;
        if(tiered_1)
        {
            _S52 = (params_0.fast_joins_0) != u32(0);
        }
        else
        {
            _S52 = false;
        }
        var _S53 : bool;
        if(_S52)
        {
            _S53 = !impactor_0;
        }
        else
        {
            _S53 = false;
        }
        var _S54 : bool;
        if(_S53)
        {
            _S54 = !fast_0;
        }
        else
        {
            _S54 = false;
        }
        var _S55 : bool;
        if(_S54)
        {
            _S55 = ((_S48 & (u32(8)))) == u32(0);
        }
        else
        {
            _S55 = false;
        }
        if(_S55)
        {
            q_4 = q_4 + u32(1);
            continue;
        }
        var e_1 : u32 = atomicAdd(&(actl_0[u32(0)]), u32(1));
        if(e_1 < (params_0.edge_cap_0))
        {
            edges_0[e_1] = vec2<u32>(ia_0, ib_0);
        }
        else
        {
            atomicStore(&(actl_0[u32(1)]), u32(1));
        }
        if(owned_0)
        {
            seed_0[ia_0] = u32(1);
            seed_0[ib_0] = u32(1);
        }
        q_4 = q_4 + u32(1);
    }
    return;
}

fn edge_count_0() -> u32
{
    var _S56 : u32 = atomicLoad(&(actl_0[u32(0)]));
    return min(_S56, params_0.edge_cap_0);
}

@compute
@workgroup_size(64, 1, 1)
fn lr_hook(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var r_5 : u32 = pass_0[i32(0)].z;
    var e_2 : u32 = id_5.x;
    var _S57 : bool;
    if(!gate_0())
    {
        _S57 = true;
    }
    else
    {
        if(r_5 > u32(0))
        {
            var _S58 : u32 = atomicLoad(&(actl_0[u32(8) + r_5 - u32(1)]));
            _S57 = _S58 == u32(0);
        }
        else
        {
            _S57 = false;
        }
    }
    if(_S57)
    {
        _S57 = true;
    }
    else
    {
        var _S59 : u32 = edge_count_0();
        _S57 = e_2 >= _S59;
    }
    if(_S57)
    {
        return;
    }
    var ab_0 : vec2<u32> = edges_0[e_2];
    var _S60 : u32 = edges_0[e_2].x;
    var la_0 : u32 = atomicLoad(&(lab_0[_S60]));
    var _S61 : u32 = ab_0.y;
    var lb_0 : u32 = atomicLoad(&(lab_0[_S61]));
    if(la_0 == lb_0)
    {
        return;
    }
    var _S62 : u32 = min(la_0, lb_0);
    var _S63 : u32 = atomicMin(&(lab_0[_S60]), _S62);
    var _S64 : u32 = atomicMin(&(lab_0[_S61]), _S62);
    var _S65 : u32 = atomicMin(&(lab_0[max(la_0, lb_0)]), _S62);
    atomicStore(&(actl_0[u32(8) + r_5]), u32(1));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn lr_jump(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var r_6 : u32 = pass_0[i32(0)].z;
    var i_4 : u32 = id_6.x;
    var _S66 : bool;
    if(!gate_0())
    {
        _S66 = true;
    }
    else
    {
        if(r_6 > u32(0))
        {
            var _S67 : u32 = atomicLoad(&(actl_0[u32(8) + r_6 - u32(1)]));
            _S66 = _S67 == u32(0);
        }
        else
        {
            _S66 = false;
        }
    }
    if(_S66)
    {
        _S66 = true;
    }
    else
    {
        _S66 = i_4 >= (part_count_0());
    }
    if(_S66)
    {
        return;
    }
    var l_0 : u32 = atomicLoad(&(lab_0[i_4]));
    var ll_0 : u32 = atomicLoad(&(lab_0[l_0]));
    if(ll_0 < l_0)
    {
        var _S68 : u32 = atomicMin(&(lab_0[i_4]), ll_0);
        atomicStore(&(actl_0[u32(8) + r_6]), u32(1));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn lr_roots(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var i_5 : u32 = id_7.x;
    var _S69 : bool;
    if(!gate_0())
    {
        _S69 = true;
    }
    else
    {
        _S69 = i_5 >= (part_count_0());
    }
    if(_S69)
    {
        return;
    }
    if(((seed_0[i_5] & (u32(1)))) != u32(0))
    {
        var _S70 : u32 = atomicLoad(&(lab_0[i_5]));
        seed_0[_S70] = (seed_0[_S70] | (u32(2)));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn lr_finish(@builtin(global_invocation_id) id_8 : vec3<u32>)
{
    var i_6 : u32 = id_8.x;
    var live_0 : u32 = live_islands_0();
    if(i_6 >= (part_count_0()))
    {
        return;
    }
    var valid_0 : bool;
    if(gate_0())
    {
        valid_0 = (((bitcast<u32>((parts_0[u32(8) * i_6].w))) & (u32(1)))) != u32(0);
    }
    else
    {
        valid_0 = false;
    }
    var routed_0 : bool;
    if(valid_0)
    {
        var _S71 : u32 = atomicLoad(&(lab_0[i_6]));
        routed_0 = ((seed_0[_S71] & (u32(2)))) != u32(0);
    }
    else
    {
        routed_0 = false;
    }
    var _S72 : u32;
    if(i_6 < live_0)
    {
        var _S73 : u32 = u32(5) * i_6;
        if(routed_0)
        {
            _S72 = u32(2147483648);
        }
        else
        {
            _S72 = u32(0);
        }
        isl_0[_S73] = vec4<u32>(_S72, u32(4294967295), u32(0), u32(0));
        var _S74 : vec4<u32> = vec4<u32>(vec4<i32>(i32(0)));
        isl_0[_S73 + u32(3)] = _S74;
        isl_0[_S73 + u32(4)] = _S74;
    }
    else
    {
        if(routed_0)
        {
            var _S75 : u32 = atomicOr(&(actl_0[u32(2)]), (u32(1) << ((i_6 - live_0))));
        }
    }
    if(i_6 == u32(0))
    {
        var _S76 : u32 = atomicLoad(&(actl_0[u32(1)]));
        var _S77 : u32 = atomicLoad(&(actl_0[u32(55)]));
        if(_S77 != u32(0))
        {
            _S72 = u32(2);
        }
        else
        {
            _S72 = u32(0);
        }
        var err_0 : u32 = (_S76 | (_S72));
        if(gate_0())
        {
            _S72 = err_0;
        }
        else
        {
            _S72 = u32(0);
        }
        ctl_0[u32(20)] = _S72;
        if(gate_0())
        {
            var _S78 : u32 = atomicLoad(&(actl_0[u32(0)]));
            _S72 = _S78;
        }
        else
        {
            _S72 = u32(0);
        }
        ctl_0[u32(21)] = _S72;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn lr_mark(@builtin(global_invocation_id) id_9 : vec3<u32>)
{
    var i_7 : u32 = id_9.x;
    var _S79 : bool;
    if(i_7 >= (live_islands_0()))
    {
        _S79 = true;
    }
    else
    {
        _S79 = !gate_0();
    }
    if(_S79)
    {
        return;
    }
    var _S80 : u32 = u32(5) * i_7;
    var e_3 : vec4<u32> = isl_0[_S80];
    if((isl_0[_S80].x) != u32(0))
    {
        var _S81 : u32 = atomicLoad(&(actl_0[u32(2)]));
        isl_0[_S80] = vec4<u32>((u32(2147483648) | (((_S81 & (u32(2147483647)))))), e_3.yzw);
    }
    return;
}

