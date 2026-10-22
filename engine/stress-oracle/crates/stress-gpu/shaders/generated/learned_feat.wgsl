struct FeatParams_std140_0
{
    @align(16) impactor_count_0 : u32,
    @align(4) frame_0 : u32,
    @align(8) layout_bonds_0 : u32,
    @align(4) rows_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) speed_0 : f32,
    @align(8) restitution_0 : f32,
    @align(4) contact_friction_0 : f32,
    @align(16) gravity_0 : vec4<f32>,
    @align(16) bases_0 : vec4<u32>,
    @align(16) bases2_0 : vec4<u32>,
    @align(16) caps_0 : vec4<u32>,
    @align(16) count_only_0 : u32,
    @align(4) max_nodes_0 : u32,
    @align(8) layout_cap_0 : u32,
    @align(4) max_bonds_cap_0 : u32,
    @align(16) prev_frame_0 : u32,
    @align(4) engine_base_0 : u32,
    @align(8) chunk_count_0 : u32,
    @align(4) pad1_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : FeatParams_std140_0;
@binding(11) @group(0) var<storage, read_write> bmap_0 : array<u32>;

@binding(28) @group(0) var<storage, read> counters_0 : array<u32>;

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

@binding(24) @group(0) var<storage, read> islands_0 : array<Island_std430_0>;

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

@binding(14) @group(0) var<storage, read> bond_static_0 : array<BondStatic_std430_0>;

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

@binding(13) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

@binding(9) @group(0) var<storage, read> sinc_0 : array<u32>;

@binding(10) @group(0) var<storage, read> slook_0 : array<vec4<u32>>;

struct JointState_std430_0
{
    @align(4) damage_0 : f32,
    @align(4) crush_0 : f32,
    @align(4) kappa_0 : f32,
    @align(4) kappa_c_0 : f32,
    @align(4) ductility_0 : f32,
    @align(4) ductility_c_0 : f32,
    @align(4) life_0 : f32,
    @align(4) plastic_x_0 : f32,
    @align(4) plastic_y_0 : f32,
    @align(4) plastic_t_0 : f32,
    @align(4) rebar_plastic_0 : f32,
    @align(4) rebar_slip0_0 : f32,
    @align(4) rebar_slip1_0 : f32,
    @align(4) rebar_work_0 : f32,
    @align(4) rebar_broken_0 : f32,
    @align(4) strain_rate_0 : f32,
    @align(4) governing_stress_0 : f32,
    @align(4) dissipated_0 : f32,
    @align(4) utilization_0 : f32,
    @align(4) mode_0 : u32,
};

struct BondDyn_std430_0
{
    @align(16) js_0 : JointState_std430_0,
    @align(16) force_lin_0 : vec4<f32>,
    @align(16) force_ang_0 : vec4<f32>,
    @align(16) sums_0 : vec4<f32>,
    @align(16) comps_0 : vec4<f32>,
    @align(16) events_0 : vec4<u32>,
};

@binding(15) @group(0) var<storage, read> bond_dyn_0 : array<BondDyn_std430_0>;

@binding(1) @group(0) var<storage, read> ctl_0 : array<u32>;

@binding(18) @group(0) var<storage, read_write> work_0 : array<f32>;

@binding(3) @group(0) var<storage, read_write> nodes_0 : array<vec4<f32>>;

@binding(12) @group(0) var<storage, read_write> key_node_0 : array<vec2<u32>>;

@binding(7) @group(0) var<storage, read> sc_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> bonds_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read> sb_0 : array<vec4<f32>>;

@binding(22) @group(0) var<storage, read_write> bedge_0 : array<vec4<u32>>;

@binding(2) @group(0) var<storage, read_write> calls_0 : array<vec4<f32>>;

@binding(16) @group(0) var<storage, read> imps_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read> cisl_0 : array<vec4<u32>>;

@binding(6) @group(0) var<storage, read> cislf_0 : array<vec4<f32>>;

@binding(17) @group(0) var<storage, read_write> contacts_0 : array<vec4<f32>>;

@binding(23) @group(0) var<storage, read_write> fctl_0 : array<u32>;

@binding(20) @group(0) var<storage, read_write> edges_0 : array<u32>;

@binding(21) @group(0) var<storage, read> cols_0 : array<u32>;

@binding(19) @group(0) var<storage, read_write> feat_0 : array<f32>;

@binding(25) @group(0) var<storage, read> ctab_0 : array<vec4<f32>>;

@binding(26) @group(0) var<storage, read> bodies_0 : array<vec4<f32>>;

@binding(27) @group(0) var<storage, read_write> state_0 : array<vec4<f32>>;

@compute
@workgroup_size(64, 1, 1)
fn feat_unmap(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : u32 = id_0.x;
    if(_S1 < (params_0.rows_0))
    {
        bmap_0[params_0.layout_cap_0 + _S1] = u32(4294967295);
    }
    return;
}

fn chunk_row_of_0( layout_chunk_0 : u32) -> u32
{
    return sinc_0[u32(2) + chunks_0[layout_chunk_0].info_1.z] + chunks_0[layout_chunk_0].info_1.w;
}

fn bond_row_lookup_0( ga_0 : u32,  gb_0 : u32) -> u32
{
    var _S2 : bool;
    var _S3 : u32 = min(ga_0, gb_0);
    var _S4 : u32 = max(ga_0, gb_0);
    var lo_0 : u32 = u32(0);
    var hi_0 : u32 = params_0.rows_0;
    loop
    {
        if(lo_0 < hi_0)
        {
        }
        else
        {
            break;
        }
        var mid_0 : u32 = (lo_0 + hi_0) / u32(2);
        var e_0 : vec4<u32> = slook_0[mid_0];
        var _S5 : u32 = e_0.x;
        if(_S5 < _S3)
        {
            _S2 = true;
        }
        else
        {
            if(_S5 == _S3)
            {
                _S2 = (e_0.y) < _S4;
            }
            else
            {
                _S2 = false;
            }
        }
        if(_S2)
        {
            lo_0 = mid_0 + u32(1);
        }
        else
        {
            hi_0 = mid_0;
        }
    }
    if(lo_0 < (params_0.rows_0))
    {
        _S2 = (slook_0[lo_0].x) == _S3;
    }
    else
    {
        _S2 = false;
    }
    if(_S2)
    {
        _S2 = (slook_0[lo_0].y) == _S4;
    }
    else
    {
        _S2 = false;
    }
    if(_S2)
    {
        return slook_0[lo_0].z;
    }
    return u32(4294967295);
}

fn connected_0( st_0 : ptr<function, JointState_std430_0>,  has_rebar_0 : bool) -> bool
{
    var _S6 : bool;
    if(((*st_0).damage_0) < 1.0f)
    {
        _S6 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S6 = ((*st_0).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S6 = false;
        }
    }
    return _S6;
}

@compute
@workgroup_size(64, 1, 1)
fn feat_map(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var i_0 : u32 = group_0.x;
    if(i_0 >= counters_0[i32(0)])
    {
        return;
    }
    var range_1 : vec4<u32> = islands_0[i_0].range_0;
    var l_0 : u32 = islands_0[i_0].range_0.z + thread_0.x;
    loop
    {
        if(l_0 < (range_1.w))
        {
        }
        else
        {
            break;
        }
        var law_1 : JointBond_std430_0 = bond_static_0[l_0].law_0;
        var row_0 : u32 = bond_row_lookup_0(chunk_row_of_0(bond_static_0[l_0].law_0.ids_0.y), chunk_row_of_0(bond_static_0[l_0].law_0.ids_0.z));
        bmap_0[l_0] = row_0;
        if(row_0 == u32(4294967295))
        {
            l_0 = l_0 + u32(64);
            continue;
        }
        bmap_0[params_0.layout_cap_0 + row_0] = l_0;
        var _S7 : JointState_std430_0 = bond_dyn_0[l_0].js_0;
        var _S8 : bool = connected_0(&(_S7), (law_1.stiff1_0.w) != 0.0f);
        var at_0 : u32 = params_0.layout_cap_0 + params_0.rows_0 + row_0;
        var _S9 : bool;
        if(!_S8)
        {
            _S9 = bmap_0[at_0] == u32(4294967295);
        }
        else
        {
            _S9 = false;
        }
        if(_S9)
        {
            bmap_0[at_0] = params_0.frame_0;
        }
        l_0 = l_0 + u32(64);
    }
    return;
}

fn node_0( n_0 : u32,  k_0 : u32) -> vec4<f32>
{
    return nodes_0[u32(18) * n_0 + k_0];
}

@compute
@workgroup_size(64, 1, 1)
fn feat_keys(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var n_1 : u32 = id_1.x;
    if(n_1 >= ctl_0[i32(2)])
    {
        return;
    }
    var k_1 : u32 = u32(0);
    loop
    {
        if(k_1 < u32(24))
        {
        }
        else
        {
            break;
        }
        work_0[u32(24) * n_1 + k_1] = 0.0f;
        k_1 = k_1 + u32(1);
    }
    if((bitcast<u32>((node_0(n_1, u32(4)).w))) != u32(0))
    {
        return;
    }
    var w12_0 : vec4<f32> = node_0(n_1, u32(12));
    var row_1 : u32 = sinc_0[u32(2) + (bitcast<u32>((w12_0.z)))] + (bitcast<u32>((w12_0.w)));
    key_node_0[row_1] = vec2<u32>(n_1 + u32(1), params_0.frame_0);
    var _S10 : u32 = u32(6) * row_1;
    var s1_0 : vec4<f32> = sc_0[_S10 + u32(1)];
    var s2_0 : vec4<f32> = sc_0[_S10 + u32(2)];
    var s3_0 : vec4<f32> = sc_0[_S10 + u32(3)];
    var _S11 : u32 = u32(24) * n_1;
    var _S12 : u32 = _S11 + u32(9);
    work_0[_S12] = s2_0.x;
    work_0[_S12 + u32(1)] = s2_0.y;
    work_0[_S12 + u32(2)] = s2_0.z;
    work_0[_S11 + u32(12)] = s2_0.w;
    work_0[_S11 + u32(13)] = s1_0.y;
    var _S13 : f32;
    if((s1_0.z) != 0.0f)
    {
        _S13 = 1.0f;
    }
    else
    {
        if((s1_0.w) != 0.0f)
        {
            _S13 = 2.0f;
        }
        else
        {
            _S13 = 0.0f;
        }
    }
    work_0[_S11 + u32(14)] = _S13;
    var _S14 : vec4<f32> = sc_0[_S10 + u32(4)];
    var _S15 : vec4<f32> = sc_0[_S10 + u32(5)];
    work_0[_S11 + u32(15)] = sc_0[_S10].y;
    work_0[_S11 + u32(16)] = (bitcast<f32>((row_1)));
    k_1 = u32(0);
    loop
    {
        if(k_1 < u32(4))
        {
        }
        else
        {
            break;
        }
        work_0[_S11 + u32(17) + k_1] = _S15[k_1];
        k_1 = k_1 + u32(1);
    }
    k_1 = u32(0);
    loop
    {
        if(k_1 < u32(3))
        {
        }
        else
        {
            break;
        }
        work_0[_S11 + u32(21) + k_1] = _S14[k_1];
        k_1 = k_1 + u32(1);
    }
    var _S16 : u32 = u32(18) * n_1;
    nodes_0[_S16 + u32(13)] = vec4<f32>(node_0(n_1, u32(13)).xyz, s3_0.w);
    nodes_0[_S16 + u32(2)] = vec4<f32>(s3_0.xyz, node_0(n_1, u32(2)).w);
    return;
}

fn node_of_key_0( row_2 : u32) -> u32
{
    var k_2 : vec2<u32> = key_node_0[row_2];
    var _S17 : u32 = key_node_0[row_2].x;
    var _S18 : bool;
    if(_S17 != u32(0))
    {
        _S18 = (k_2.y) == (params_0.frame_0);
    }
    else
    {
        _S18 = false;
    }
    var _S19 : u32;
    if(_S18)
    {
        _S19 = _S17 - u32(1);
    }
    else
    {
        _S19 = u32(4294967295);
    }
    return _S19;
}

@compute
@workgroup_size(64, 1, 1)
fn feat_brow(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var b_0 : u32 = id_2.x;
    if(b_0 >= ctl_0[i32(3)])
    {
        return;
    }
    var l_1 : u32 = (bitcast<u32>((bonds_0[u32(2) * b_0 + u32(1)].z)));
    var ends_0 : vec4<u32> = (bitcast<vec4<u32>>((sb_0[u32(9) * bmap_0[l_1] + u32(7)])));
    bedge_0[b_0] = vec4<u32>(node_of_key_0(ends_0.x), node_of_key_0(ends_0.y), bmap_0[l_1], l_1);
    return;
}

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_0 : u32,  thread_1 : u32) -> u32
{
    if(thread_1 == u32(0))
    {
        g_group_uniform_0 = value_0;
    }
    var _S20 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S20;
}

fn call_mask_0( g_0 : u32) -> u32
{
    return (bitcast<u32>((calls_0[u32(10) * g_0 + u32(5)].w)));
}

fn call_e0_0( g_1 : u32) -> u32
{
    return (bitcast<u32>((calls_0[u32(10) * g_1 + u32(6)].x)));
}

fn call_ne_0( g_2 : u32) -> u32
{
    return (bitcast<u32>((calls_0[u32(10) * g_2 + u32(6)].y)));
}

fn call_first_0( g_3 : u32) -> u32
{
    return (bitcast<u32>((calls_0[u32(10) * g_3 + u32(2)].z)));
}

fn call_chunks_0( g_4 : u32) -> u32
{
    return (bitcast<u32>((calls_0[u32(10) * g_4 + u32(3)].w)));
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

var<workgroup> g_red_0 : array<array<vec4<f32>, i32(64)>, i32(2)>;

fn group_min_0( t_0 : u32,  a_0 : vec4<f32>) -> vec4<f32>
{
    workgroupBarrier();
    g_red_0[i32(0)][t_0] = a_0;
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
        if(t_0 < s_0)
        {
            g_red_0[i32(0)][t_0] = min(g_red_0[i32(0)][t_0], g_red_0[i32(0)][t_0 + s_0]);
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var r_0 : vec4<f32> = g_red_0[i32(0)][i32(0)];
    workgroupBarrier();
    return r_0;
}

fn group_sum_0( t_1 : u32,  a_1 : vec4<f32>) -> vec4<f32>
{
    workgroupBarrier();
    g_red_0[i32(0)][t_1] = a_1;
    workgroupBarrier();
    var s_1 : u32 = u32(32);
    loop
    {
        if(s_1 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_1 < s_1)
        {
            g_red_0[i32(0)][t_1] = g_red_0[i32(0)][t_1] + g_red_0[i32(0)][t_1 + s_1];
        }
        workgroupBarrier();
        s_1 = (s_1 >> (u32(1)));
    }
    var r_1 : vec4<f32> = g_red_0[i32(0)][i32(0)];
    workgroupBarrier();
    return r_1;
}

var<workgroup> g_mv_0 : array<f32, i32(64)>;

var<workgroup> g_mi_0 : array<u32, i32(64)>;

fn group_argmin_0( t_2 : u32,  v_0 : f32,  i_1 : u32) -> u32
{
    workgroupBarrier();
    g_mv_0[t_2] = v_0;
    g_mi_0[t_2] = i_1;
    workgroupBarrier();
    var s_2 : u32 = u32(32);
    loop
    {
        if(s_2 > u32(0))
        {
        }
        else
        {
            break;
        }
        var _S21 : bool;
        if(t_2 < s_2)
        {
            var _S22 : u32 = t_2 + s_2;
            if((g_mv_0[_S22]) < (g_mv_0[t_2]))
            {
                _S21 = true;
            }
            else
            {
                if((g_mv_0[_S22]) == (g_mv_0[t_2]))
                {
                    _S21 = (g_mi_0[_S22]) < (g_mi_0[t_2]);
                }
                else
                {
                    _S21 = false;
                }
            }
        }
        else
        {
            _S21 = false;
        }
        if(_S21)
        {
            var _S23 : u32 = t_2 + s_2;
            g_mv_0[t_2] = g_mv_0[_S23];
            g_mi_0[t_2] = g_mi_0[_S23];
        }
        workgroupBarrier();
        s_2 = (s_2 >> (u32(1)));
    }
    var r_2 : u32 = g_mi_0[i32(0)];
    workgroupBarrier();
    return r_2;
}

var<workgroup> g_plug_0 : array<u32, i32(4096)>;

var<workgroup> g_flag_0 : u32;

fn hertz_duration_0( m_0 : f32,  r_3 : f32,  e_1 : f32,  v_1 : f32) -> f32
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_3 * e_1 * e_1 * max(abs(v_1), 0.00100000004749745f)), 0.20000000298023224f);
}

@compute
@workgroup_size(64, 1, 1)
fn feat_contacts(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var t0_0 : f32;
    var _S24 : bool;
    var iter_0 : u32;
    var j_0 : u32;
    var bj_0 : u32;
    var marked_0 : bool;
    var _S25 : bool;
    var _S26 : bool;
    var g_5 : u32 = group_1.x;
    var t_3 : u32 = thread_2.x;
    var k_3 : u32;
    if(g_5 >= ctl_0[i32(1)])
    {
        k_3 = u32(1);
    }
    else
    {
        k_3 = u32(0);
    }
    var _S27 : u32 = group_uniform_0(k_3, t_3);
    if(_S27 != u32(0))
    {
        return;
    }
    var _S28 : u32 = group_uniform_0(call_mask_0(g_5), t_3);
    var _S29 : u32 = group_uniform_0(call_e0_0(g_5), t_3);
    var _S30 : u32 = group_uniform_0(call_ne_0(g_5), t_3);
    var _S31 : u32 = call_first_0(g_5);
    var _S32 : u32 = call_chunks_0(g_5);
    var _S33 : u32 = u32(10) * g_5 + u32(8);
    var _S34 : u32 = (bitcast<u32>((calls_0[_S33].x)));
    var vmax_0 : f32 = 0.0f;
    k_3 = u32(0);
    var rows_1 : u32 = u32(0);
    loop
    {
        if(k_3 < u32(32))
        {
        }
        else
        {
            break;
        }
        var _S35 : u32 = (u32(1) << (k_3));
        var vmax_1 : f32;
        if(((_S28 & (_S35))) == u32(0))
        {
            vmax_1 = vmax_0;
            var _S36 : u32 = k_3 + u32(1);
            vmax_0 = vmax_1;
            k_3 = _S36;
            continue;
        }
        var inode_0 : u32 = _S31 + _S32 + countOneBits((_S28 & ((_S35 - u32(1)))));
        var _S37 : vec3<f32> = node_0(inode_0, u32(0)).xyz;
        var _S38 : vec3<f32> = node_0(inode_0, u32(3)).xyz;
        var _S39 : u32 = u32(4) * k_3;
        var _S40 : f32 = imps_0[_S39 + u32(2)].x;
        var _S41 : f32 = imps_0[_S39 + u32(3)].z;
        var _S42 : f32 = imps_0[_S39].w;
        var _S43 : f32 = imps_0[_S39 + u32(1)].w;
        vmax_1 = vmax_0;
        var e_2 : u32 = _S29;
        var rows_2 : u32 = rows_1;
        loop
        {
            if(e_2 < (_S29 + _S30))
            {
            }
            else
            {
                break;
            }
            var _S44 : u32 = u32(2) * e_2;
            var c0_0 : vec4<u32> = cisl_0[_S44];
            var _S45 : u32 = _S44 + u32(1);
            var c1_0 : vec4<u32> = cisl_0[_S45];
            var f1_0 : vec4<f32> = cislf_0[_S45];
            var at_1 : u32 = _S34 + rows_2;
            var _S46 : u32 = rows_2 + u32(1);
            var _S47 : bool = t_3 == u32(0);
            if(_S47)
            {
                contacts_0[u32(6) * at_1 + u32(5)] = vec4<f32>(0.0f);
            }
            var n_isl_0 : u32 = group_uniform_0(c0_0.z, t_3);
            var _S48 : bool;
            if(n_isl_0 == u32(0))
            {
                _S48 = true;
            }
            else
            {
                _S48 = (c1_0.z) == u32(0);
            }
            if(_S48)
            {
                e_2 = e_2 + u32(1);
                rows_2 = _S46;
                continue;
            }
            var v_2 : vec3<f32> = _S38 - f1_0.xyz;
            var speed_1 : f32 = length(v_2);
            var _S49 : u32;
            if(speed_1 > 0.0f)
            {
                _S49 = u32(1);
            }
            else
            {
                _S49 = u32(0);
            }
            var _S50 : u32 = group_uniform_0(_S49, t_3);
            if(_S50 == u32(0))
            {
                e_2 = e_2 + u32(1);
                rows_2 = _S46;
                continue;
            }
            var inf_0 : f32 = float_bits_0(u32(2139095040));
            var _S51 : vec4<f32> = vec4<f32>(- inf_0);
            var lo_1 : vec4<f32> = vec4<f32>(inf_0);
            var hi_1 : vec4<f32> = _S51;
            var j_1 : u32 = t_3;
            var msum_0 : f32 = 0.0f;
            loop
            {
                if(j_1 < n_isl_0)
                {
                }
                else
                {
                    break;
                }
                var n_2 : u32 = c0_0.y + j_1;
                var p_0 : vec3<f32> = node_0(n_2, u32(0)).xyz;
                var _S52 : vec3<f32> = vec3<f32>((work_0[u32(24) * n_2 + u32(12)] + _S40));
                var _S53 : vec4<f32> = min(lo_1, vec4<f32>(p_0 - _S52, 0.0f));
                var _S54 : vec4<f32> = max(hi_1, vec4<f32>(p_0 + _S52, 0.0f));
                var msum_1 : f32 = msum_0 + node_0(n_2, u32(0)).w;
                var j_2 : u32 = j_1 + u32(64);
                lo_1 = _S53;
                hi_1 = _S54;
                j_1 = j_2;
                msum_0 = msum_1;
            }
            var _S55 : vec4<f32> = group_min_0(t_3, lo_1);
            var _S56 : vec4<f32> = group_min_0(t_3, (vec4<f32>(0) - hi_1));
            var _S57 : vec4<f32> = (vec4<f32>(0) - _S56);
            var _S58 : vec4<f32> = group_sum_0(t_3, vec4<f32>(msum_0, 0.0f, 0.0f, 0.0f));
            var cl_mass_0 : f32 = _S58.x;
            var t0_1 : f32 = 0.0f;
            var t1_1 : f32 = params_0.dt_0;
            var a_2 : u32 = u32(0);
            loop
            {
                if(a_2 < u32(3))
                {
                }
                else
                {
                    break;
                }
                var _S59 : u32 = a_2;
                if((abs(v_2[a_2])) < 9.999999960041972e-13f)
                {
                    var _S60 : u32 = a_2;
                    if((_S37[a_2]) < (_S55[a_2]))
                    {
                        _S24 = true;
                    }
                    else
                    {
                        _S24 = (_S37[_S60]) > (_S57[a_2]);
                    }
                    if(_S24)
                    {
                        t0_0 = inf_0;
                    }
                    else
                    {
                        t0_0 = t0_1;
                    }
                    t0_1 = t0_0;
                    a_2 = a_2 + u32(1);
                    continue;
                }
                var ta_0 : f32 = (_S55[a_2] - _S37[a_2]) / v_2[_S59];
                var tb_0 : f32 = (_S57[a_2] - _S37[a_2]) / v_2[_S59];
                var _S61 : f32 = min(t1_1, max(ta_0, tb_0));
                t0_1 = max(t0_1, min(ta_0, tb_0));
                t1_1 = _S61;
                a_2 = a_2 + u32(1);
            }
            var _S62 : u32;
            if(t0_1 > t1_1)
            {
                _S62 = u32(1);
            }
            else
            {
                _S62 = u32(0);
            }
            var _S63 : u32 = group_uniform_0(_S62, t_3);
            if(_S63 != u32(0))
            {
                e_2 = e_2 + u32(1);
                rows_2 = _S46;
                continue;
            }
            var d_0 : vec3<f32> = v_2 / vec3<f32>(speed_1);
            var hit_0 : vec3<f32> = _S37 + v_2 * vec3<f32>(t0_1);
            t0_0 = inf_0;
            var best_j_0 : u32 = u32(4294967295);
            var j_3 : u32 = t_3;
            loop
            {
                if(j_3 < n_isl_0)
                {
                }
                else
                {
                    break;
                }
                var q_0 : vec3<f32> = node_0(c0_0.y + j_3, u32(0)).xyz - hit_0;
                var along_0 : f32 = dot(q_0, d_0);
                var dist_0 : f32 = length(q_0 - d_0 * vec3<f32>(along_0)) + abs(along_0) * 0.00100000004749745f;
                if(dist_0 < t0_0)
                {
                    t0_0 = dist_0;
                    best_j_0 = j_3;
                }
                j_3 = j_3 + u32(64);
            }
            var entry_0 : u32 = group_argmin_0(t_3, t0_0, best_j_0);
            var _S64 : u32 = c0_0.y;
            var entry_n_0 : u32 = _S64 + entry_0;
            var _S65 : f32 = dot(node_0(entry_n_0, u32(0)).xyz - hit_0, d_0);
            var _S66 : bool = n_isl_0 <= u32(4096);
            var j_4 : u32 = t_3;
            loop
            {
                if(j_4 < (min(n_isl_0, u32(4096))))
                {
                }
                else
                {
                    break;
                }
                if(j_4 == entry_0)
                {
                    iter_0 = u32(1);
                }
                else
                {
                    iter_0 = u32(0);
                }
                g_plug_0[j_4] = iter_0;
                j_4 = j_4 + u32(64);
            }
            workgroupBarrier();
            iter_0 = u32(0);
            loop
            {
                if(iter_0 < n_isl_0)
                {
                    _S24 = _S66;
                }
                else
                {
                    _S24 = false;
                }
                if(_S24)
                {
                }
                else
                {
                    break;
                }
                if(_S47)
                {
                    g_flag_0 = u32(0);
                }
                workgroupBarrier();
                j_0 = t_3;
                loop
                {
                    if(j_0 < (c1_0.z))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var be_0 : vec4<u32> = bedge_0[c1_0.y + j_0];
                    var _S67 : u32 = bedge_0[c1_0.y + j_0].x;
                    if(_S67 == u32(4294967295))
                    {
                        marked_0 = true;
                    }
                    else
                    {
                        marked_0 = (be_0.y) == u32(4294967295);
                    }
                    if(marked_0)
                    {
                        j_0 = j_0 + u32(64);
                        continue;
                    }
                    var la_0 : u32 = _S67 - _S64;
                    var lb_0 : u32 = be_0.y - _S64;
                    if(la_0 >= n_isl_0)
                    {
                        _S25 = true;
                    }
                    else
                    {
                        _S25 = lb_0 >= n_isl_0;
                    }
                    if(_S25)
                    {
                        _S26 = true;
                    }
                    else
                    {
                        _S26 = (g_plug_0[la_0]) == (g_plug_0[lb_0]);
                    }
                    if(_S26)
                    {
                        j_0 = j_0 + u32(64);
                        continue;
                    }
                    if((g_plug_0[la_0]) != u32(0))
                    {
                        bj_0 = lb_0;
                    }
                    else
                    {
                        bj_0 = la_0;
                    }
                    var _S68 : u32 = _S64 + bj_0;
                    var q_1 : vec3<f32> = node_0(_S68, u32(0)).xyz - hit_0;
                    var along_1 : f32 = dot(q_1, d_0);
                    var lateral_0 : f32 = length(q_1 - d_0 * vec3<f32>(along_1));
                    var _S69 : bool;
                    if(along_1 >= (_S65 - work_0[u32(24) * _S68 + u32(12)]))
                    {
                        _S69 = lateral_0 <= (_S41 + max(along_1 - _S65, 0.0f));
                    }
                    else
                    {
                        _S69 = false;
                    }
                    if(_S69)
                    {
                        g_plug_0[bj_0] = u32(1);
                        g_flag_0 = u32(1);
                    }
                    j_0 = j_0 + u32(64);
                }
                workgroupBarrier();
                var _S70 : u32 = group_uniform_0(g_flag_0, t_3);
                if(_S70 == u32(0))
                {
                    break;
                }
                workgroupBarrier();
                iter_0 = iter_0 + u32(1);
            }
            var acc_0 : vec4<f32> = vec4<f32>(0.0f);
            j_0 = t_3;
            loop
            {
                if(j_0 < n_isl_0)
                {
                }
                else
                {
                    break;
                }
                if((g_plug_0[j_0]) != u32(0))
                {
                    acc_0 = acc_0 + vec4<f32>(node_0(_S64 + j_0, u32(0)).w, 1.0f, 0.0f, 0.0f);
                }
                j_0 = j_0 + u32(64);
            }
            bj_0 = t_3;
            loop
            {
                if(bj_0 < (c1_0.z))
                {
                }
                else
                {
                    break;
                }
                var be_1 : vec4<u32> = bedge_0[c1_0.y + bj_0];
                var _S71 : u32 = bedge_0[c1_0.y + bj_0].x;
                if(_S71 == u32(4294967295))
                {
                    marked_0 = true;
                }
                else
                {
                    marked_0 = (be_1.y) == u32(4294967295);
                }
                if(marked_0)
                {
                    bj_0 = bj_0 + u32(64);
                    continue;
                }
                var la_1 : u32 = _S71 - _S64;
                var lb_1 : u32 = be_1.y - _S64;
                if(la_1 < n_isl_0)
                {
                    _S25 = lb_1 < n_isl_0;
                }
                else
                {
                    _S25 = false;
                }
                if(_S25)
                {
                    _S26 = ((g_plug_0[la_1]) != u32(0)) != ((g_plug_0[lb_1]) != u32(0));
                }
                else
                {
                    _S26 = false;
                }
                if(_S26)
                {
                    var _S72 : u32 = u32(9) * be_1.z;
                    acc_0[i32(2)] = acc_0[i32(2)] + sb_0[_S72 + u32(5)].x * sb_0[_S72].w * sb_0[_S72 + u32(4)].z;
                }
                bj_0 = bj_0 + u32(64);
            }
            var _S73 : vec4<f32> = group_sum_0(t_3, acc_0);
            acc_0 = _S73;
            var m_plug_0 : f32 = _S73.x;
            var _S74 : f32 = f32(n_isl_0);
            var has_plug_0 : bool = (_S73.y) != _S74;
            var capacity_0 : f32 = _S73.z;
            var modulus_0 : f32 = 1.0f / (1.0f / _S43 + 1.0f / node_0(entry_n_0, u32(13)).w);
            var m_red_0 : f32;
            if((f1_0.w) != 0.0f)
            {
                m_red_0 = _S42;
            }
            else
            {
                m_red_0 = cl_mass_0 * _S42 / (cl_mass_0 + _S42);
            }
            var hertz_0 : f32 = hertz_duration_0(m_red_0, _S41, modulus_0, speed_1);
            var peak_0 : f32 = 3.14159274101257324f * (2.0f * m_red_0 * speed_1) / (2.0f * hertz_0);
            var ratio_0 : f32;
            if(has_plug_0)
            {
                var m_eff_0 : f32 = m_plug_0 * _S42 / (m_plug_0 + _S42);
                ratio_0 = m_eff_0 * speed_1 / hertz_duration_0(m_eff_0, _S41, modulus_0, speed_1) / max(capacity_0, 1.00000000317107685e-30f);
            }
            else
            {
                ratio_0 = 0.0f;
            }
            if(has_plug_0)
            {
                marked_0 = speed_1 > (params_0.speed_0);
            }
            else
            {
                marked_0 = false;
            }
            if(marked_0)
            {
                var j_5 : u32 = t_3;
                loop
                {
                    if(j_5 < n_isl_0)
                    {
                    }
                    else
                    {
                        break;
                    }
                    if((g_plug_0[j_5]) != u32(0))
                    {
                        work_0[u32(24) * (_S64 + j_5) + u32(3)] = f32(at_1 - _S34 + u32(1));
                    }
                    j_5 = j_5 + u32(64);
                }
            }
            if(_S47)
            {
                var _S75 : u32 = u32(6) * at_1;
                contacts_0[_S75] = vec4<f32>((bitcast<f32>((k_3))), (bitcast<f32>((entry_n_0))), (bitcast<f32>((e_2))), t0_1);
                contacts_0[_S75 + u32(1)] = vec4<f32>(hit_0, speed_1);
                contacts_0[_S75 + u32(2)] = vec4<f32>(d_0, m_red_0);
                contacts_0[_S75 + u32(3)] = vec4<f32>(modulus_0, _S41, hertz_0, peak_0);
                var _S76 : f32;
                if(has_plug_0)
                {
                    _S76 = m_plug_0;
                }
                else
                {
                    _S76 = 0.0f;
                }
                var _S77 : f32;
                if(marked_0)
                {
                    _S77 = 1.0f;
                }
                else
                {
                    _S77 = 0.0f;
                }
                contacts_0[_S75 + u32(4)] = vec4<f32>(ratio_0, _S76, _S77, _S74);
                contacts_0[_S75 + u32(5)] = vec4<f32>(1.0f, 0.0f, 0.0f, 0.0f);
            }
            var _S78 : f32 = max(vmax_1, speed_1);
            workgroupBarrier();
            vmax_1 = _S78;
            e_2 = e_2 + u32(1);
            rows_2 = _S46;
        }
        rows_1 = rows_2;
        var _S36 : u32 = k_3 + u32(1);
        vmax_0 = vmax_1;
        k_3 = _S36;
    }
    if(t_3 == u32(0))
    {
        calls_0[_S33] = vec4<f32>(calls_0[_S33].x, (bitcast<f32>((rows_1))), 0.0f, vmax_0);
    }
    return;
}

fn call_bfirst_0( g_6 : u32) -> u32
{
    return (bitcast<u32>((calls_0[u32(10) * g_6 + u32(4)].x)));
}

fn call_nbonds_0( g_7 : u32) -> u32
{
    return (bitcast<u32>((calls_0[u32(10) * g_7 + u32(4)].y)));
}

var<workgroup> g_scan_0 : array<u32, i32(64)>;

fn group_scan_0( t_4 : u32,  v_3 : u32) -> u32
{
    workgroupBarrier();
    g_scan_0[t_4] = v_3;
    workgroupBarrier();
    var step_0 : u32 = u32(1);
    loop
    {
        if(step_0 < u32(64))
        {
        }
        else
        {
            break;
        }
        var add_0 : u32;
        if(t_4 >= step_0)
        {
            add_0 = g_scan_0[t_4 - step_0];
        }
        else
        {
            add_0 = u32(0);
        }
        workgroupBarrier();
        g_scan_0[t_4] = g_scan_0[t_4] + add_0;
        workgroupBarrier();
        step_0 = (step_0 << (u32(1)));
    }
    return g_scan_0[t_4];
}

fn group_total_0() -> u32
{
    var r_4 : u32 = g_scan_0[u32(63)];
    workgroupBarrier();
    return r_4;
}

fn sort_work_0( t_5 : u32,  base_0 : u32,  n_3 : u32)
{
    var m_1 : u32 = u32(1);
    loop
    {
        if(m_1 < n_3)
        {
        }
        else
        {
            break;
        }
        m_1 = (m_1 << (u32(1)));
    }
    var _S79 : f32 = float_bits_0(u32(2139095040));
    var j_6 : u32 = n_3 + t_5;
    loop
    {
        if(j_6 < m_1)
        {
        }
        else
        {
            break;
        }
        work_0[base_0 + j_6] = _S79;
        j_6 = j_6 + u32(64);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    var size_0 : u32 = u32(2);
    loop
    {
        if(size_0 <= m_1)
        {
        }
        else
        {
            break;
        }
        var stride_0 : u32 = (size_0 >> (u32(1)));
        loop
        {
            if(stride_0 > u32(0))
            {
            }
            else
            {
                break;
            }
            j_6 = t_5;
            loop
            {
                if(j_6 < m_1)
                {
                }
                else
                {
                    break;
                }
                var p_1 : u32 = (j_6 ^ (stride_0));
                if(p_1 > j_6)
                {
                    var _S80 : u32 = base_0 + j_6;
                    var a_3 : f32 = work_0[_S80];
                    var _S81 : u32 = base_0 + p_1;
                    var b_1 : f32 = work_0[_S81];
                    if((work_0[_S80] > work_0[_S81]) == (((j_6 & (size_0))) == u32(0)))
                    {
                        work_0[_S80] = b_1;
                        work_0[_S81] = a_3;
                    }
                }
                j_6 = j_6 + u32(64);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            stride_0 = (stride_0 >> (u32(1)));
        }
        size_0 = (size_0 << (u32(1)));
    }
    return;
}

fn median_work_0( base_1 : u32,  n_4 : u32) -> f32
{
    var k_4 : u32 = n_4 / u32(2);
    var _S82 : f32;
    if((n_4 % u32(2)) == u32(1))
    {
        _S82 = work_0[base_1 + k_4];
    }
    else
    {
        var _S83 : u32 = base_1 + k_4;
        _S82 = 0.5f * (work_0[_S83 - u32(1)] + work_0[_S83]);
    }
    return _S82;
}

var<workgroup> g_scalars_0 : vec4<f32>;

@compute
@workgroup_size(64, 1, 1)
fn feat_scales(@builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var _S84 : bool;
    var ok_0 : u32;
    var t_6 : u32 = thread_3.x;
    var _S85 : u32 = group_uniform_0(ctl_0[i32(1)], t_6);
    var _S86 : u32 = u32(24) * params_0.max_nodes_0;
    var g_8 : u32 = u32(0);
    loop
    {
        if(g_8 < _S85)
        {
        }
        else
        {
            break;
        }
        var first_0 : u32 = call_first_0(g_8);
        var nc_0 : u32 = group_uniform_0(call_chunks_0(g_8), t_6);
        var _S87 : u32 = call_bfirst_0(g_8);
        var _S88 : u32 = group_uniform_0(call_nbonds_0(g_8), t_6);
        var _S89 : u32 = u32(10) * g_8;
        var L_0 : f32 = calls_0[_S89].x;
        var best_0 : f32 = -1.0f;
        var best_j_1 : u32 = u32(0);
        var j_7 : u32 = t_6;
        loop
        {
            if(j_7 < nc_0)
            {
            }
            else
            {
                break;
            }
            var m_2 : f32 = node_0(first_0 + j_7, u32(0)).w;
            if(m_2 >= best_0)
            {
                best_0 = m_2;
                best_j_1 = j_7;
            }
            j_7 = j_7 + u32(64);
        }
        workgroupBarrier();
        g_mv_0[t_6] = best_0;
        g_mi_0[t_6] = best_j_1;
        workgroupBarrier();
        var s_3 : u32 = u32(32);
        loop
        {
            if(s_3 > u32(0))
            {
            }
            else
            {
                break;
            }
            if(t_6 < s_3)
            {
                var _S90 : u32 = t_6 + s_3;
                if((g_mv_0[_S90]) > (g_mv_0[t_6]))
                {
                    _S84 = true;
                }
                else
                {
                    if((g_mv_0[_S90]) == (g_mv_0[t_6]))
                    {
                        _S84 = (g_mi_0[_S90]) > (g_mi_0[t_6]);
                    }
                    else
                    {
                        _S84 = false;
                    }
                }
            }
            else
            {
                _S84 = false;
            }
            if(_S84)
            {
                var _S91 : u32 = t_6 + s_3;
                g_mv_0[t_6] = g_mv_0[_S91];
                g_mi_0[t_6] = g_mi_0[_S91];
            }
            workgroupBarrier();
            s_3 = (s_3 >> (u32(1)));
        }
        var dom_0 : u32 = first_0 + g_mi_0[i32(0)];
        workgroupBarrier();
        var w12_1 : vec4<f32> = node_0(dom_0, u32(12));
        var _S92 : u32 = u32(6) * (sinc_0[u32(2) + (bitcast<u32>((w12_1.z)))] + (bitcast<u32>((w12_1.w))));
        var d0_0 : vec4<f32> = sc_0[_S92];
        var d1_0 : vec4<f32> = sc_0[_S92 + u32(1)];
        var c_wave_0 : f32 = sqrt(d0_0.y / d0_0.z);
        var ga_1 : vec4<f32> = vec4<f32>(0.0f);
        var b0_0 : u32 = u32(0);
        var carry_0 : u32 = u32(0);
        loop
        {
            if(b0_0 < _S88)
            {
            }
            else
            {
                break;
            }
            var j_8 : u32 = b0_0 + t_6;
            var _S93 : bool = j_8 < _S88;
            var be_2 : vec4<u32>;
            if(_S93)
            {
                be_2 = bedge_0[_S87 + j_8];
            }
            else
            {
                be_2 = vec4<u32>(u32(4294967295));
            }
            if(_S93)
            {
                _S84 = (be_2.x) != u32(4294967295);
            }
            else
            {
                _S84 = false;
            }
            var _S94 : bool;
            if(_S84)
            {
                _S94 = (be_2.y) != u32(4294967295);
            }
            else
            {
                _S94 = false;
            }
            if(_S94)
            {
                ok_0 = u32(1);
            }
            else
            {
                ok_0 = u32(0);
            }
            var _S95 : u32 = group_scan_0(t_6, ok_0);
            var at_2 : u32 = carry_0 + _S95 - ok_0;
            if(ok_0 != u32(0))
            {
                var _S96 : u32 = u32(9) * be_2.z;
                var area_0 : f32 = sb_0[_S96].w;
                var gab_0 : f32 = sb_0[_S96 + u32(6)].y * area_0;
                work_0[_S86 + at_2] = sb_0[_S96 + u32(5)].x;
                work_0[_S86 + u32(131072) + at_2] = gab_0;
                work_0[_S86 + u32(262144) + at_2] = area_0;
                ga_1[i32(0)] = ga_1[i32(0)] + gab_0;
                if((work_0[u32(24) * be_2.x + u32(3)] != 0.0f) != (work_0[u32(24) * be_2.y + u32(3)] != 0.0f))
                {
                    ga_1[i32(1)] = ga_1[i32(1)] + gab_0;
                }
            }
            var _S97 : u32 = group_total_0();
            var carry_1 : u32 = carry_0 + _S97;
            b0_0 = b0_0 + u32(64);
            carry_0 = carry_1;
        }
        var nlive_0 : u32 = group_uniform_0(carry_0, t_6);
        var _S98 : vec4<f32> = group_sum_0(t_6, ga_1);
        ga_1 = _S98;
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var _S99 : f32 = d0_0.w;
        var _S100 : f32 = d1_0.x * L_0 * L_0;
        var _S101 : f32 = L_0 * L_0;
        var f_t_0 : f32;
        var GA_0 : f32;
        var med_area_0 : f32;
        if(nlive_0 > u32(0))
        {
            sort_work_0(t_6, _S86, nlive_0);
            var _S102 : u32 = _S86 + u32(131072);
            sort_work_0(t_6, _S102, nlive_0);
            var _S103 : u32 = _S86 + u32(262144);
            sort_work_0(t_6, _S103, nlive_0);
            if(t_6 == u32(0))
            {
                g_scalars_0 = vec4<f32>(median_work_0(_S86, nlive_0), median_work_0(_S102, nlive_0), median_work_0(_S103, nlive_0), 0.0f);
            }
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            var _S104 : f32 = g_scalars_0.x;
            var _S105 : f32 = g_scalars_0.y;
            var _S106 : f32 = g_scalars_0.z;
            storageBarrier(); textureBarrier(); workgroupBarrier();;
            f_t_0 = _S104;
            GA_0 = _S105;
            med_area_0 = _S106;
        }
        else
        {
            f_t_0 = _S99;
            GA_0 = _S100;
            med_area_0 = _S101;
        }
        ok_0 = t_6;
        loop
        {
            if(ok_0 < nc_0)
            {
            }
            else
            {
                break;
            }
            var w_0 : vec4<f32> = node_0(first_0 + ok_0, u32(12));
            work_0[_S86 + ok_0] = sc_0[u32(6) * (sinc_0[u32(2) + (bitcast<u32>((w_0.z)))] + (bitcast<u32>((w_0.w))))].y;
            ok_0 = ok_0 + u32(64);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        sort_work_0(t_6, _S86, nc_0);
        var _S107 : bool = t_6 == u32(0);
        if(_S107)
        {
            g_scalars_0[i32(3)] = median_work_0(_S86, nc_0);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var e_med_0 : f32 = g_scalars_0.w;
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        if(_S107)
        {
            var _S108 : u32 = _S89 + u32(1);
            var s1_1 : vec4<f32> = calls_0[_S108];
            calls_0[_S89] = vec4<f32>(calls_0[_S89].x, calls_0[_S89].y, calls_0[_S89].z, c_wave_0);
            calls_0[_S108] = vec4<f32>(f_t_0, e_med_0, GA_0, s1_1.w);
            calls_0[_S89 + u32(9)] = vec4<f32>(med_area_0, ga_1.x, ga_1.y, 0.0f);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        g_8 = g_8 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn feat_summ(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var n_5 : u32 = id_3.x;
    var _S109 : bool;
    if(n_5 >= ctl_0[i32(2)])
    {
        _S109 = true;
    }
    else
    {
        _S109 = (bitcast<u32>((node_0(n_5, u32(4)).w))) != u32(0);
    }
    if(_S109)
    {
        return;
    }
    var w12_2 : vec4<f32> = node_0(n_5, u32(12));
    var _S110 : u32 = sinc_0[i32(1)];
    var off_0 : u32 = u32(2) + sinc_0[i32(0)];
    var _S111 : u32 = off_0 + (sinc_0[u32(2) + (bitcast<u32>((w12_2.z)))] + (bitcast<u32>((w12_2.w))));
    var a_4 : u32 = sinc_0[_S111];
    var _S112 : u32 = sinc_0[_S111 + u32(1)];
    var dmax_0 : f32 = 0.0f;
    var i_2 : u32 = a_4;
    var nlive_1 : f32 = 0.0f;
    var dsum_0 : f32 = 0.0f;
    var nbroken_0 : f32 = 0.0f;
    loop
    {
        if(i_2 < _S112)
        {
        }
        else
        {
            break;
        }
        var l_2 : u32 = bmap_0[params_0.layout_cap_0 + sinc_0[off_0 + _S110 + u32(1) + i_2]];
        if(bmap_0[params_0.layout_cap_0 + sinc_0[off_0 + _S110 + u32(1) + i_2]] != u32(4294967295))
        {
            var _S113 : JointState_std430_0 = bond_dyn_0[l_2].js_0;
            var _S114 : bool = connected_0(&(_S113), (bond_static_0[l_2].law_0.stiff1_0.w) != 0.0f);
            _S109 = _S114;
        }
        else
        {
            _S109 = false;
        }
        if(_S109)
        {
            var nlive_2 : f32 = nlive_1 + 1.0f;
            var dsum_1 : f32 = dsum_0 + bond_dyn_0[l_2].js_0.damage_0;
            dmax_0 = max(dmax_0, bond_dyn_0[l_2].js_0.damage_0);
            nlive_1 = nlive_2;
            dsum_0 = dsum_1;
        }
        else
        {
            nbroken_0 = nbroken_0 + 1.0f;
        }
        i_2 = i_2 + u32(1);
    }
    var _S115 : u32 = u32(24) * n_5;
    work_0[_S115 + u32(5)] = dsum_0;
    work_0[_S115 + u32(6)] = dmax_0;
    work_0[_S115 + u32(7)] = nlive_1;
    work_0[_S115 + u32(8)] = nbroken_0;
    return;
}

fn sphere_contact_time_0( dx_0 : vec3<f32>,  dv_0 : vec3<f32>,  r_5 : f32,  dt_1 : f32) -> f32
{
    var c_0 : f32 = dot(dx_0, dx_0) - r_5 * r_5;
    if(c_0 <= 0.0f)
    {
        return 0.0f;
    }
    var a_5 : f32 = dot(dv_0, dv_0);
    var b_2 : f32 = dot(dx_0, dv_0);
    var _S116 : bool;
    if(a_5 <= 0.0f)
    {
        _S116 = true;
    }
    else
    {
        _S116 = b_2 >= 0.0f;
    }
    if(_S116)
    {
        return -1.0f;
    }
    var disc_0 : f32 = b_2 * b_2 - a_5 * c_0;
    if(disc_0 < 0.0f)
    {
        return -1.0f;
    }
    var t_7 : f32 = (- b_2 - sqrt(disc_0)) / a_5;
    var _S117 : f32;
    if(t_7 <= dt_1)
    {
        _S117 = max(t_7, 0.0f);
    }
    else
    {
        _S117 = -1.0f;
    }
    return _S117;
}

@compute
@workgroup_size(64, 1, 1)
fn feat_graph(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var g_9 : u32 = group_2.x;
    var t_8 : u32 = thread_4.x;
    var j_9 : u32;
    if(g_9 >= ctl_0[i32(1)])
    {
        j_9 = u32(1);
    }
    else
    {
        j_9 = u32(0);
    }
    var _S118 : u32 = group_uniform_0(j_9, t_8);
    if(_S118 != u32(0))
    {
        return;
    }
    var _S119 : u32 = call_first_0(g_9);
    var _S120 : u32 = group_uniform_0(call_chunks_0(g_9), t_8);
    var _S121 : u32 = call_bfirst_0(g_9);
    var _S122 : u32 = group_uniform_0(call_nbonds_0(g_9), t_8);
    var _S123 : u32 = u32(10) * g_9;
    var _S124 : u32 = _S123 + u32(8);
    var _S125 : u32 = (bitcast<u32>((calls_0[_S124].x)));
    var _S126 : u32 = group_uniform_0((bitcast<u32>((calls_0[_S124].y))), t_8);
    var _S127 : f32 = calls_0[_S123].w;
    var _S128 : f32 = float_bits_0(u32(2139095040));
    var _S129 : f32 = params_0.dt_0;
    j_9 = t_8;
    loop
    {
        if(j_9 < _S120)
        {
        }
        else
        {
            break;
        }
        var _S130 : u32 = u32(24) * (_S119 + j_9);
        work_0[_S130] = _S128;
        work_0[_S130 + u32(1)] = _S128;
        var _S131 : f32;
        if(work_0[_S130 + u32(14)] != 0.0f)
        {
            _S131 = 0.0f;
        }
        else
        {
            _S131 = _S128;
        }
        work_0[_S130 + u32(2)] = _S131;
        j_9 = j_9 + u32(64);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    var _S132 : bool = t_8 == u32(0);
    var iter_1 : u32;
    if(_S132)
    {
        iter_1 = u32(0);
        loop
        {
            if(iter_1 < _S126)
            {
            }
            else
            {
                break;
            }
            var _S133 : u32 = u32(6) * (_S125 + iter_1);
            var c0_1 : vec4<f32> = contacts_0[_S133];
            if((contacts_0[_S133 + u32(5)].x) == 0.0f)
            {
                iter_1 = iter_1 + u32(1);
                continue;
            }
            var en_0 : u32 = (bitcast<u32>((c0_1.y)));
            var k_5 : u32 = (bitcast<u32>((c0_1.x)));
            var _S134 : u32 = u32(24) * en_0;
            work_0[_S134] = 0.0f;
            var inode_1 : u32 = _S119 + _S120 + countOneBits(((call_mask_0(g_9)) & ((((u32(1) << (k_5))) - u32(1)))));
            var tc_0 : f32 = sphere_contact_time_0(node_0(en_0, u32(13)).xyz - node_0(inode_1, u32(13)).xyz, node_0(en_0, u32(3)).xyz - node_0(inode_1, u32(3)).xyz, imps_0[u32(4) * k_5 + u32(2)].x + work_0[_S134 + u32(12)], _S129);
            if(tc_0 >= 0.0f)
            {
                var _S135 : u32 = _S134 + u32(1);
                work_0[_S135] = min(work_0[_S135], tc_0);
            }
            iter_1 = iter_1 + u32(1);
        }
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    iter_1 = u32(0);
    loop
    {
        if(iter_1 < (_S120 + u32(1)))
        {
        }
        else
        {
            break;
        }
        if(_S132)
        {
            g_flag_0 = u32(0);
        }
        workgroupBarrier();
        j_9 = t_8;
        loop
        {
            if(j_9 < _S122)
            {
            }
            else
            {
                break;
            }
            var be_3 : vec4<u32> = bedge_0[_S121 + j_9];
            var _S136 : u32 = bedge_0[_S121 + j_9].x;
            var _S137 : bool;
            if(_S136 == u32(4294967295))
            {
                _S137 = true;
            }
            else
            {
                _S137 = (be_3.y) == u32(4294967295);
            }
            if(_S137)
            {
                j_9 = j_9 + u32(64);
                continue;
            }
            var side_0 : u32 = u32(0);
            loop
            {
                if(side_0 < u32(2))
                {
                }
                else
                {
                    break;
                }
                var _S138 : bool = side_0 == u32(0);
                var u_0 : u32;
                if(_S138)
                {
                    u_0 = _S136;
                }
                else
                {
                    u_0 = be_3.y;
                }
                var w_1 : u32;
                if(_S138)
                {
                    w_1 = be_3.y;
                }
                else
                {
                    w_1 = _S136;
                }
                var _S139 : u32 = u32(24) * u_0;
                var hu_0 : f32 = work_0[_S139] + 1.0f;
                var _S140 : u32 = u32(24) * w_1;
                if(hu_0 < work_0[_S140])
                {
                    work_0[_S140] = hu_0;
                    g_flag_0 = u32(1);
                }
                var au_0 : f32 = work_0[_S139 + u32(2)] + 1.0f;
                var _S141 : u32 = _S140 + u32(2);
                if(au_0 < work_0[_S141])
                {
                    work_0[_S141] = au_0;
                    g_flag_0 = u32(1);
                }
                var tu_0 : f32 = work_0[_S139 + u32(1)] + length(node_0(u_0, u32(13)).xyz - node_0(w_1, u32(13)).xyz) / _S127;
                var _S142 : u32 = _S140 + u32(1);
                if(tu_0 < work_0[_S142])
                {
                    work_0[_S142] = tu_0;
                    g_flag_0 = u32(1);
                }
                side_0 = side_0 + u32(1);
            }
            j_9 = j_9 + u32(64);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
        var _S143 : u32 = group_uniform_0(g_flag_0, t_8);
        if(_S143 == u32(0))
        {
            break;
        }
        workgroupBarrier();
        iter_1 = iter_1 + u32(1);
    }
    return;
}

fn edge_base_0( g_10 : u32,  s_4 : u32) -> u32
{
    return fctl_0[u32(8) + u32(6) * g_10 + u32(3) + s_4];
}

fn quat_matrix_0( q_2 : vec4<f32>) -> mat3x3<f32>
{
    var w_2 : f32 = q_2.w;
    var x_0 : f32 = q_2.x;
    var y_0 : f32 = q_2.y;
    var z_0 : f32 = q_2.z;
    var _S144 : f32 = y_0 * y_0;
    var _S145 : f32 = z_0 * z_0;
    var _S146 : f32 = x_0 * y_0;
    var _S147 : f32 = w_2 * z_0;
    var _S148 : f32 = x_0 * z_0;
    var _S149 : f32 = w_2 * y_0;
    var _S150 : f32 = x_0 * x_0;
    var _S151 : f32 = y_0 * z_0;
    var _S152 : f32 = w_2 * x_0;
    return mat3x3<f32>(vec3<f32>(1.0f - 2.0f * (_S144 + _S145), 2.0f * (_S146 - _S147), 2.0f * (_S148 + _S149)), vec3<f32>(2.0f * (_S146 + _S147), 1.0f - 2.0f * (_S150 + _S145), 2.0f * (_S151 - _S152)), vec3<f32>(2.0f * (_S148 - _S149), 2.0f * (_S151 + _S152), 1.0f - 2.0f * (_S150 + _S144)));
}

fn col3_0( m_3 : mat3x3<f32>,  c_1 : u32) -> vec3<f32>
{
    return vec3<f32>(m_3[i32(0)][c_1], m_3[i32(1)][c_1], m_3[i32(2)][c_1]);
}

fn tangent_of_0( n_6 : vec3<f32>,  body_0 : mat3x3<f32>) -> vec3<f32>
{
    var a_6 : u32 = u32(0);
    loop
    {
        if(a_6 < u32(2))
        {
        }
        else
        {
            break;
        }
        var axis_0 : vec3<f32> = col3_0(body_0, a_6);
        var t_9 : vec3<f32> = axis_0 - n_6 * vec3<f32>(dot(axis_0, n_6));
        if((length(t_9)) > 9.99999997475242708e-07f)
        {
            return normalize(t_9);
        }
        a_6 = a_6 + u32(1);
    }
    var axis_1 : vec3<f32> = col3_0(body_0, u32(2));
    return normalize(axis_1 - n_6 * vec3<f32>(dot(axis_1, n_6)));
}

fn support_of_0( r_6 : mat3x3<f32>,  half_1 : vec3<f32>,  n_7 : vec3<f32>) -> f32
{
    var l_3 : vec3<f32> = (((n_7) * (transpose(r_6))));
    return abs(l_3.x) * half_1.x + abs(l_3.y) * half_1.y + abs(l_3.z) * half_1.z;
}

fn ln_ratio_0( a_7 : f32,  b_3 : f32) -> f32
{
    return log(max(a_7, 9.99999997475242708e-07f) / max(b_3, 9.99999997475242708e-07f));
}

fn call_cbase_0( g_11 : u32) -> u32
{
    return (bitcast<u32>((calls_0[u32(10) * g_11 + u32(6)].z)));
}

fn call_ibase_0( g_12 : u32) -> u32
{
    return (bitcast<u32>((calls_0[u32(10) * g_12 + u32(6)].w)));
}

fn global_node_0( g_13 : u32,  j_10 : u32) -> u32
{
    var nc_1 : u32 = call_chunks_0(g_13);
    if(j_10 < nc_1)
    {
        return call_cbase_0(g_13) + j_10;
    }
    return fctl_0[i32(4)] + call_ibase_0(g_13) + (j_10 - nc_1);
}

fn put_edge_0( section_0 : u32,  row_3 : u32,  src_0 : u32,  dst_0 : u32)
{
    var base_2 : u32;
    if(section_0 == u32(4))
    {
        base_2 = u32(0);
    }
    else
    {
        if(section_0 == u32(5))
        {
            base_2 = u32(2) * params_0.caps_0.x;
        }
        else
        {
            base_2 = u32(2) * (params_0.caps_0.x + params_0.caps_0.y);
        }
    }
    var _S153 : u32 = base_2 + u32(2) * row_3;
    edges_0[_S153] = src_0;
    edges_0[_S153 + u32(1)] = dst_0;
    return;
}

fn col_of_0( section_1 : u32,  k_6 : u32) -> u32
{
    return cols_0[section_1 * u32(65) + u32(1) + k_6];
}

fn stride_of_0( section_2 : u32) -> u32
{
    return cols_0[section_2 * u32(65)];
}

fn put_0( base_3 : u32,  section_3 : u32,  row_4 : u32,  k_7 : u32,  v_4 : f32)
{
    var c_2 : u32 = col_of_0(section_3, k_7);
    if(c_2 != u32(4294967295))
    {
        feat_0[base_3 + row_4 * stride_of_0(section_3) + c_2] = v_4;
    }
    return;
}

fn put_rot6_0( base_4 : u32,  section_4 : u32,  row_5 : u32,  k0_0 : u32,  t1_2 : vec3<f32>,  t2_1 : vec3<f32>,  body_1 : mat3x3<f32>)
{
    var _S154 : mat3x3<f32> = transpose(body_1);
    var a_8 : vec3<f32> = (((t1_2) * (_S154)));
    var b_4 : vec3<f32> = (((t2_1) * (_S154)));
    put_0(base_4, section_4, row_5, k0_0, a_8.x);
    put_0(base_4, section_4, row_5, k0_0 + u32(1), a_8.y);
    put_0(base_4, section_4, row_5, k0_0 + u32(2), a_8.z);
    put_0(base_4, section_4, row_5, k0_0 + u32(3), b_4.x);
    put_0(base_4, section_4, row_5, k0_0 + u32(4), b_4.y);
    put_0(base_4, section_4, row_5, k0_0 + u32(5), b_4.z);
    return;
}

fn put_key_0( section_5 : u32,  k_8 : u32,  a_9 : u32,  b_5 : u32)
{
    var _S155 : u32 = u32(4) * (params_0.caps_0.x + params_0.caps_0.y + params_0.caps_0.z);
    var _S156 : u32;
    if(section_5 == u32(4))
    {
        _S156 = u32(0);
    }
    else
    {
        if(section_5 == u32(5))
        {
            _S156 = params_0.caps_0.x;
        }
        else
        {
            _S156 = params_0.caps_0.x + params_0.caps_0.y;
        }
    }
    var _S157 : u32 = _S155 + _S156 + u32(2) * k_8;
    edges_0[_S157] = a_9;
    edges_0[_S157 + u32(1)] = b_5;
    return;
}

fn edge_count_0( g_14 : u32,  s_5 : u32,  n_8 : u32)
{
    fctl_0[u32(8) + u32(6) * g_14 + s_5] = n_8;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn feat_ic(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    var g_15 : u32 = group_3.x;
    var t_10 : u32 = thread_5.x;
    var _S158 : bool;
    if(g_15 >= ctl_0[i32(1)])
    {
        _S158 = true;
    }
    else
    {
        _S158 = t_10 != u32(0);
    }
    if(_S158)
    {
        return;
    }
    var _S159 : u32 = call_first_0(g_15);
    var _S160 : u32 = call_chunks_0(g_15);
    var _S161 : u32 = u32(10) * g_15;
    var _S162 : u32 = _S161 + u32(8);
    var _S163 : u32 = (bitcast<u32>((calls_0[_S162].x)));
    var _S164 : u32 = (bitcast<u32>((calls_0[_S162].y)));
    var _S165 : f32 = calls_0[_S161].x;
    var _S166 : f32 = calls_0[_S161].z;
    var _S167 : f32 = calls_0[_S161].w;
    var _S168 : f32 = calls_0[_S161 + u32(1)].x;
    var _S169 : u32;
    if((params_0.count_only_0) != u32(0))
    {
        _S169 = u32(0);
    }
    else
    {
        _S169 = edge_base_0(g_15, u32(1));
    }
    var _S170 : u32 = params_0.bases2_0.y;
    var _S171 : u32 = _S161 + u32(9);
    var _S172 : f32 = calls_0[_S171].x;
    var r_7 : u32 = u32(0);
    var count_0 : u32 = u32(0);
    loop
    {
        if(r_7 < _S164)
        {
        }
        else
        {
            break;
        }
        var _S173 : u32 = u32(6) * (_S163 + r_7);
        if((contacts_0[_S173 + u32(5)].x) == 0.0f)
        {
            r_7 = r_7 + u32(1);
            continue;
        }
        if((params_0.count_only_0) == u32(0))
        {
            var _S174 : vec4<f32> = contacts_0[_S173 + u32(3)];
            var k_9 : u32 = (bitcast<u32>((contacts_0[_S173].x)));
            var en_1 : u32 = (bitcast<u32>((contacts_0[_S173].y)));
            var inode_2 : u32 = _S159 + _S160 + countOneBits(((call_mask_0(g_15)) & ((((u32(1) << (k_9))) - u32(1)))));
            var ri_0 : mat3x3<f32> = quat_matrix_0(node_0(inode_2, u32(1)));
            var nrm_0 : vec3<f32> = normalize(contacts_0[_S173 + u32(2)].xyz);
            var t1_3 : vec3<f32> = tangent_of_0(nrm_0, ri_0);
            var _S175 : vec3<f32> = cross(nrm_0, t1_3);
            var _S176 : u32 = u32(24) * en_1;
            var _S177 : u32 = _S176 + u32(9);
            var _S178 : u32 = u32(4) * k_9;
            var _S179 : u32 = _S178 + u32(2);
            var hn_0 : f32 = imps_0[_S179].x;
            var _S180 : f32 = dot(node_0(en_1, u32(13)).xyz - node_0(inode_2, u32(13)).xyz, nrm_0) - support_of_0(ri_0, vec3<f32>(imps_0[_S179].y, imps_0[_S179].z, imps_0[_S179].w), nrm_0) - support_of_0(quat_matrix_0(node_0(en_1, u32(1))), vec3<f32>(work_0[_S177], work_0[_S177 + u32(1)], work_0[_S177 + u32(2)]), nrm_0);
            var _S181 : f32 = contacts_0[_S173 + u32(1)].w;
            var ek_0 : f32 = 0.5f * contacts_0[_S173 + u32(2)].w * _S181 * _S181;
            var plug_row_0 : u32 = u32(work_0[_S176 + u32(3)]);
            var ga_plug_0 : f32 = calls_0[_S171].z;
            if(ga_plug_0 > 0.0f)
            {
                _S158 = plug_row_0 != u32(0);
            }
            else
            {
                _S158 = false;
            }
            var _S182 : f32;
            if(_S158)
            {
                _S182 = ln_ratio_0(ek_0, ga_plug_0);
            }
            else
            {
                _S182 = 0.0f;
            }
            var _S183 : u32 = _S178 + u32(3);
            var fr_imp_0 : f32 = imps_0[_S183].y;
            var fr_ch_0 : f32 = work_0[_S176 + u32(13)];
            var _S184 : f32;
            if((params_0.contact_friction_0) >= 0.0f)
            {
                _S184 = params_0.contact_friction_0;
            }
            else
            {
                _S184 = min(fr_imp_0, fr_ch_0);
            }
            var _S185 : f32 = imps_0[_S183].z;
            var tc_1 : f32 = sphere_contact_time_0(node_0(en_1, u32(13)).xyz - node_0(inode_2, u32(13)).xyz, node_0(en_1, u32(3)).xyz - node_0(inode_2, u32(3)).xyz, hn_0 + work_0[_S176 + u32(12)], params_0.dt_0);
            var _S186 : f32;
            if(tc_1 < 0.0f)
            {
                _S186 = 1.0f;
            }
            else
            {
                if(tc_1 <= 0.0f)
                {
                    _S186 = -1.0f;
                }
                else
                {
                    _S186 = tc_1 / params_0.dt_0;
                }
            }
            var _S187 : f32 = imps_0[_S183].x;
            var _S188 : f32 = work_0[_S176 + u32(15)];
            var side_1 : u32 = u32(0);
            loop
            {
                if(side_1 < u32(2))
                {
                }
                else
                {
                    break;
                }
                var row_6 : u32 = _S169 + count_0 + side_1;
                var _S189 : bool = side_1 == u32(0);
                var sn_0 : u32;
                if(_S189)
                {
                    sn_0 = inode_2;
                }
                else
                {
                    sn_0 = en_1;
                }
                var dn_0 : u32;
                if(_S189)
                {
                    dn_0 = en_1;
                }
                else
                {
                    dn_0 = inode_2;
                }
                var ft1_0 : vec3<f32>;
                if(_S189)
                {
                    ft1_0 = t1_3;
                }
                else
                {
                    ft1_0 = (vec3<f32>(0) - t1_3);
                }
                var fn_0 : vec3<f32>;
                if(_S189)
                {
                    fn_0 = nrm_0;
                }
                else
                {
                    fn_0 = (vec3<f32>(0) - nrm_0);
                }
                var rel_0 : vec3<f32> = node_0(sn_0, u32(3)).xyz - node_0(dn_0, u32(3)).xyz;
                var _S190 : f32 = dot(ft1_0, rel_0);
                var _S191 : f32 = dot(_S175, rel_0);
                var _S192 : f32 = dot(fn_0, rel_0);
                var wi_0 : vec3<f32> = node_0(inode_2, u32(4)).xyz * vec3<f32>((_S185 / _S166));
                var _S193 : f32 = dot(ft1_0, wi_0);
                var _S194 : f32 = dot(_S175, wi_0);
                var _S195 : f32 = dot(fn_0, wi_0);
                put_edge_0(u32(5), row_6, global_node_0(g_15, sn_0 - _S159), global_node_0(g_15, dn_0 - _S159));
                put_0(_S170, u32(5), row_6, u32(0), _S192 / _S166);
                put_0(_S170, u32(5), row_6, u32(1), _S190 / _S166);
                put_0(_S170, u32(5), row_6, u32(2), _S191 / _S166);
                put_0(_S170, u32(5), row_6, u32(3), _S192 / _S167);
                put_0(_S170, u32(5), row_6, u32(4), ln_ratio_0(node_0(dn_0, u32(0)).w, node_0(sn_0, u32(0)).w));
                put_0(_S170, u32(5), row_6, u32(5), ln_ratio_0(_S185, _S165));
                put_0(_S170, u32(5), row_6, u32(6), ln_ratio_0(_S187, _S188));
                put_0(_S170, u32(5), row_6, u32(7), _S193);
                put_0(_S170, u32(5), row_6, u32(8), _S194);
                put_0(_S170, u32(5), row_6, u32(9), _S195);
                put_0(_S170, u32(5), row_6, u32(10), _S186);
                put_0(_S170, u32(5), row_6, u32(11), _S182);
                put_0(_S170, u32(5), row_6, u32(12), ln_ratio_0(_S174.w, _S168 * _S172));
                put_0(_S170, u32(5), row_6, u32(13), _S174.z / params_0.dt_0);
                var _S196 : f32;
                if(_S189)
                {
                    _S196 = 1.0f;
                }
                else
                {
                    _S196 = 0.0f;
                }
                put_0(_S170, u32(5), row_6, u32(14), _S196);
                put_0(_S170, u32(5), row_6, u32(15), max(- _S180, 0.0f) / _S165);
                put_0(_S170, u32(5), row_6, u32(16), _S184);
                put_0(_S170, u32(5), row_6, u32(17), params_0.restitution_0);
                put_rot6_0(_S170, u32(5), row_6, u32(18), ft1_0, _S175, quat_matrix_0(node_0(sn_0, u32(1))));
                put_rot6_0(_S170, u32(5), row_6, u32(24), ft1_0, _S175, quat_matrix_0(node_0(dn_0, u32(1))));
                work_0[u32(24) * sn_0 + u32(4)] = 1.0f;
                side_1 = side_1 + u32(1);
            }
            put_key_0(u32(5), (_S169 + count_0) / u32(2), (bitcast<u32>((node_0(en_1, u32(12)).z))), ((bitcast<u32>((node_0(en_1, u32(12)).w))) | (((k_9 << (u32(24)))))));
        }
        count_0 = count_0 + u32(2);
        r_7 = r_7 + u32(1);
    }
    if((params_0.count_only_0) != u32(0))
    {
        edge_count_0(g_15, u32(1), count_0);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn feat_cracked_find(@builtin(workgroup_id) group_4 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var g_16 : u32 = group_4.x;
    var t_11 : u32 = thread_6.x;
    var _S197 : bool;
    if(g_16 >= ctl_0[i32(1)])
    {
        _S197 = true;
    }
    else
    {
        _S197 = t_11 != u32(0);
    }
    if(_S197)
    {
        return;
    }
    var first_1 : u32 = call_first_0(g_16);
    var nc_2 : u32 = call_chunks_0(g_16);
    var _S198 : u32 = sinc_0[i32(1)];
    var _S199 : u32 = u32(2) + sinc_0[i32(0)];
    var slots_0 : u32 = params_0.max_bonds_cap_0 + u32(8) * first_1;
    var cap_0 : u32 = u32(8) * nc_2 - u32(1);
    var j_11 : u32 = u32(0);
    var count_1 : u32 = u32(0);
    loop
    {
        if(j_11 < nc_2)
        {
        }
        else
        {
            break;
        }
        var na_0 : u32 = first_1 + j_11;
        var rowa_0 : u32 = sinc_0[u32(2) + (bitcast<u32>((node_0(na_0, u32(12)).z)))] + (bitcast<u32>((node_0(na_0, u32(12)).w)));
        var _S200 : u32 = _S199 + rowa_0;
        var i_3 : u32 = sinc_0[_S200];
        var count_2 : u32 = count_1;
        loop
        {
            if(i_3 < sinc_0[_S200 + u32(1)])
            {
            }
            else
            {
                break;
            }
            var br_0 : u32 = sinc_0[_S199 + _S198 + u32(1) + i_3];
            var ends_1 : vec4<u32> = (bitcast<vec4<u32>>((sb_0[u32(9) * br_0 + u32(7)])));
            if((ends_1.x) != rowa_0)
            {
                i_3 = i_3 + u32(1);
                continue;
            }
            var nb_0 : u32 = node_of_key_0(ends_1.y);
            if(nb_0 == u32(4294967295))
            {
                _S197 = true;
            }
            else
            {
                _S197 = nb_0 < first_1;
            }
            var _S201 : bool;
            if(_S197)
            {
                _S201 = true;
            }
            else
            {
                _S201 = nb_0 >= (first_1 + nc_2);
            }
            if(_S201)
            {
                i_3 = i_3 + u32(1);
                continue;
            }
            var l_4 : u32 = bmap_0[params_0.layout_cap_0 + br_0];
            var _S202 : bool;
            if(bmap_0[params_0.layout_cap_0 + br_0] != u32(4294967295))
            {
                var _S203 : JointState_std430_0 = bond_dyn_0[l_4].js_0;
                var _S204 : bool = connected_0(&(_S203), (bond_static_0[l_4].law_0.stiff1_0.w) != 0.0f);
                _S202 = _S204;
            }
            else
            {
                _S202 = false;
            }
            if(_S202)
            {
                i_3 = i_3 + u32(1);
                continue;
            }
            if(count_2 < cap_0)
            {
                bedge_0[slots_0 + u32(1) + count_2] = vec4<u32>(br_0, na_0, nb_0, bmap_0[params_0.layout_cap_0 + params_0.rows_0 + br_0]);
            }
            count_2 = count_2 + u32(1);
            i_3 = i_3 + u32(1);
        }
        j_11 = j_11 + u32(1);
        count_1 = count_2;
    }
    var _S205 : u32 = min(count_1, cap_0);
    if(count_1 > cap_0)
    {
        j_11 = u32(1);
    }
    else
    {
        j_11 = u32(0);
    }
    bedge_0[slots_0] = vec4<u32>(_S205, j_11, u32(0), u32(0));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn feat_cracked(@builtin(workgroup_id) group_5 : vec3<u32>, @builtin(local_invocation_id) thread_7 : vec3<u32>)
{
    var g_17 : u32 = group_5.x;
    var t_12 : u32 = thread_7.x;
    var _S206 : bool;
    if(g_17 >= ctl_0[i32(1)])
    {
        _S206 = true;
    }
    else
    {
        _S206 = t_12 != u32(0);
    }
    if(_S206)
    {
        return;
    }
    var first_2 : u32 = call_first_0(g_17);
    var _S207 : f32 = calls_0[u32(10) * g_17].x;
    var _S208 : f32 = calls_0[u32(10) * g_17].z;
    var _S209 : u32;
    if((params_0.count_only_0) != u32(0))
    {
        _S209 = u32(0);
    }
    else
    {
        _S209 = edge_base_0(g_17, u32(0));
    }
    var _S210 : u32 = params_0.bases2_0.x;
    var slots_1 : u32 = params_0.max_bonds_cap_0 + u32(8) * first_2;
    var _S211 : u32 = bedge_0[slots_1].x;
    var ci_0 : u32 = u32(0);
    var count_3 : u32 = u32(0);
    loop
    {
        if(ci_0 < _S211)
        {
        }
        else
        {
            break;
        }
        var na_1 : u32 = bedge_0[slots_1 + u32(1) + ci_0].y;
        var nb_1 : u32 = bedge_0[slots_1 + u32(1) + ci_0].z;
        var stamp_0 : u32 = bedge_0[slots_1 + u32(1) + ci_0].w;
        var ra_1 : mat3x3<f32> = quat_matrix_0(node_0(na_1, u32(1)));
        var rbm_0 : mat3x3<f32> = quat_matrix_0(node_0(nb_1, u32(1)));
        var _S212 : u32 = u32(24) * na_1;
        var _S213 : u32 = _S212 + u32(17);
        var rs_0 : mat3x3<f32> = (((transpose(quat_matrix_0(vec4<f32>(work_0[_S213], work_0[_S213 + u32(1)], work_0[_S213 + u32(2)], work_0[_S213 + u32(3)])))) * (ra_1)));
        var _S214 : u32 = u32(9) * bedge_0[slots_1 + u32(1) + ci_0].x;
        var g0_0 : vec4<f32> = sb_0[_S214];
        var ends_2 : vec4<u32> = (bitcast<vec4<u32>>((sb_0[_S214 + u32(7)])));
        var nrm_1 : vec3<f32> = (((sb_0[_S214 + u32(1)].xyz) * (rs_0)));
        var t1_4 : vec3<f32> = (((sb_0[_S214 + u32(2)].xyz) * (rs_0)));
        var xa_0 : vec3<f32> = node_0(na_1, u32(13)).xyz;
        var _S215 : vec3<f32> = node_0(nb_1, u32(13)).xyz - xa_0;
        var nrm_2 : vec3<f32>;
        var t1_5 : vec3<f32>;
        if((dot(nrm_1, _S215)) < 0.0f)
        {
            var _S216 : vec3<f32> = (vec3<f32>(0) - t1_4);
            nrm_2 = (vec3<f32>(0) - nrm_1);
            t1_5 = _S216;
        }
        else
        {
            nrm_2 = nrm_1;
            t1_5 = t1_4;
        }
        var _S217 : u32 = _S212 + u32(9);
        var ha_0 : vec3<f32> = vec3<f32>(work_0[_S217], work_0[_S217 + u32(1)], work_0[_S217 + u32(2)]);
        var _S218 : u32 = u32(24) * nb_1;
        var _S219 : u32 = _S218 + u32(9);
        var hb_0 : vec3<f32> = vec3<f32>(work_0[_S219], work_0[_S219 + u32(1)], work_0[_S219 + u32(2)]);
        var gap_0 : f32 = dot(_S215, nrm_2) - support_of_0(ra_1, ha_0, nrm_2) - support_of_0(rbm_0, hb_0, nrm_2);
        var t2_2 : vec3<f32> = cross(nrm_2, t1_5);
        var across1_0 : bool = (abs(dot(_S215, t1_5))) <= (support_of_0(ra_1, ha_0, t1_5) + support_of_0(rbm_0, hb_0, t1_5));
        var across2_0 : bool = (abs(dot(_S215, t2_2))) <= (support_of_0(ra_1, ha_0, t2_2) + support_of_0(rbm_0, hb_0, t2_2));
        if(gap_0 > (0.25f * _S207))
        {
            _S206 = true;
        }
        else
        {
            _S206 = !across1_0;
        }
        var _S220 : bool;
        if(_S206)
        {
            _S220 = true;
        }
        else
        {
            _S220 = !across2_0;
        }
        if(_S220)
        {
            ci_0 = ci_0 + u32(1);
            continue;
        }
        if((params_0.count_only_0) == u32(0))
        {
            var _S221 : u32 = _S212 + u32(21);
            var _S222 : vec3<f32> = xa_0 + (((g0_0.xyz - vec3<f32>(work_0[_S221], work_0[_S221 + u32(1)], work_0[_S221 + u32(2)])) * (rs_0)));
            var _S223 : f32;
            if(stamp_0 == u32(4294967295))
            {
                _S223 = 1.0f;
            }
            else
            {
                _S223 = min(f32(params_0.frame_0 - stamp_0) / 8.0f, 1.0f);
            }
            var fa_0 : f32 = work_0[_S212 + u32(13)];
            var fbr_0 : f32 = work_0[_S218 + u32(13)];
            var _S224 : f32;
            if((params_0.contact_friction_0) >= 0.0f)
            {
                _S224 = params_0.contact_friction_0;
            }
            else
            {
                _S224 = min(fa_0, fbr_0);
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
                var row_7 : u32 = _S209 + count_3 + side_2;
                var _S225 : bool = side_2 == u32(0);
                var sn_1 : u32;
                if(_S225)
                {
                    sn_1 = na_1;
                }
                else
                {
                    sn_1 = nb_1;
                }
                var dn_1 : u32;
                if(_S225)
                {
                    dn_1 = nb_1;
                }
                else
                {
                    dn_1 = na_1;
                }
                var ft1_1 : vec3<f32>;
                if(_S225)
                {
                    ft1_1 = t1_5;
                }
                else
                {
                    ft1_1 = (vec3<f32>(0) - t1_5);
                }
                var fn_1 : vec3<f32>;
                if(_S225)
                {
                    fn_1 = nrm_2;
                }
                else
                {
                    fn_1 = (vec3<f32>(0) - nrm_2);
                }
                var q_3 : vec3<f32> = _S222 - node_0(sn_1, u32(13)).xyz;
                var dv_1 : vec3<f32> = node_0(dn_1, u32(3)).xyz + cross(node_0(dn_1, u32(4)).xyz, _S222 - node_0(dn_1, u32(13)).xyz) - (node_0(sn_1, u32(3)).xyz + cross(node_0(sn_1, u32(4)).xyz, _S222 - node_0(sn_1, u32(13)).xyz));
                put_edge_0(u32(4), row_7, global_node_0(g_17, sn_1 - first_2), global_node_0(g_17, dn_1 - first_2));
                put_0(_S210, u32(4), row_7, u32(0), ln_ratio_0(g0_0.w, _S207 * _S207));
                put_0(_S210, u32(4), row_7, u32(1), dot(ft1_1, q_3) / _S207);
                put_0(_S210, u32(4), row_7, u32(2), dot(t2_2, q_3) / _S207);
                put_0(_S210, u32(4), row_7, u32(3), dot(fn_1, q_3) / _S207);
                put_0(_S210, u32(4), row_7, u32(4), gap_0 / _S207);
                put_0(_S210, u32(4), row_7, u32(5), _S224);
                put_0(_S210, u32(4), row_7, u32(6), dot(ft1_1, dv_1) / _S208);
                put_0(_S210, u32(4), row_7, u32(7), dot(t2_2, dv_1) / _S208);
                put_0(_S210, u32(4), row_7, u32(8), dot(fn_1, dv_1) / _S208);
                put_0(_S210, u32(4), row_7, u32(9), _S223);
                put_rot6_0(_S210, u32(4), row_7, u32(10), ft1_1, t2_2, quat_matrix_0(node_0(sn_1, u32(1))));
                put_rot6_0(_S210, u32(4), row_7, u32(16), ft1_1, t2_2, quat_matrix_0(node_0(dn_1, u32(1))));
                work_0[u32(24) * sn_1 + u32(4)] = 1.0f;
                side_2 = side_2 + u32(1);
            }
            put_key_0(u32(4), (_S209 + count_3) / u32(2), ends_2.z, ends_2.w);
        }
        count_3 = count_3 + u32(2);
        ci_0 = ci_0 + u32(1);
    }
    if((params_0.count_only_0) != u32(0))
    {
        edge_count_0(g_17, u32(0), count_3);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn feat_sc(@builtin(workgroup_id) group_6 : vec3<u32>, @builtin(local_invocation_id) thread_8 : vec3<u32>)
{
    var g_18 : u32 = group_6.x;
    var t_13 : u32 = thread_8.x;
    var _S226 : bool;
    if(g_18 >= ctl_0[i32(1)])
    {
        _S226 = true;
    }
    else
    {
        _S226 = t_13 != u32(0);
    }
    if(_S226)
    {
        return;
    }
    var _S227 : u32 = call_first_0(g_18);
    var e0_0 : u32 = call_e0_0(g_18);
    var _S228 : u32 = call_ne_0(g_18);
    var _S229 : f32 = calls_0[u32(10) * g_18].x;
    var _S230 : f32 = calls_0[u32(10) * g_18].z;
    var _S231 : f32 = calls_0[u32(10) * g_18].w;
    var _S232 : u32;
    if((params_0.count_only_0) != u32(0))
    {
        _S232 = u32(0);
    }
    else
    {
        _S232 = edge_base_0(g_18, u32(2));
    }
    var _S233 : u32 = params_0.bases2_0.z;
    var ea_0 : u32 = e0_0;
    var count_4 : u32 = u32(0);
    loop
    {
        var _S234 : u32 = e0_0 + _S228;
        if(ea_0 < _S234)
        {
        }
        else
        {
            break;
        }
        var _S235 : vec4<u32> = cisl_0[u32(2) * ea_0];
        var ja_0 : u32 = u32(0);
        loop
        {
            if(ja_0 < (_S235.z))
            {
            }
            else
            {
                break;
            }
            var na_2 : u32 = _S235.y + ja_0;
            var _S236 : u32 = u32(24) * na_2;
            var _S237 : u32 = (bitcast<u32>((work_0[_S236 + u32(16)])));
            var _S238 : vec3<f32> = node_0(na_2, u32(13)).xyz;
            var _S239 : f32 = work_0[_S236 + u32(12)];
            var e_3 : u32 = e0_0;
            var count_5 : u32 = count_4;
            loop
            {
                if(e_3 < _S234)
                {
                }
                else
                {
                    break;
                }
                var count_6 : u32;
                if(e_3 == ea_0)
                {
                    count_6 = count_5;
                    e_3 = e_3 + u32(1);
                    count_5 = count_6;
                    continue;
                }
                var _S240 : vec4<u32> = cisl_0[u32(2) * e_3];
                var j_12 : u32 = u32(0);
                count_6 = count_5;
                loop
                {
                    if(j_12 < (_S240.z))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var nb_2 : u32 = _S240.y + j_12;
                    var _S241 : u32 = u32(24) * nb_2;
                    var rb_1 : u32 = (bitcast<u32>((work_0[_S241 + u32(16)])));
                    if(rb_1 <= _S237)
                    {
                        j_12 = j_12 + u32(1);
                        continue;
                    }
                    var d_1 : vec3<f32> = node_0(nb_2, u32(13)).xyz - _S238;
                    var _S242 : f32 = length(d_1);
                    if(!(_S242 < (_S239 + work_0[_S241 + u32(12)])))
                    {
                        j_12 = j_12 + u32(1);
                        continue;
                    }
                    if((bond_row_lookup_0(_S237, rb_1)) != u32(4294967295))
                    {
                        j_12 = j_12 + u32(1);
                        continue;
                    }
                    if(_S242 < 9.999999960041972e-13f)
                    {
                        j_12 = j_12 + u32(1);
                        continue;
                    }
                    var nrm_3 : vec3<f32> = normalize(d_1);
                    var rma_0 : mat3x3<f32> = quat_matrix_0(node_0(na_2, u32(1)));
                    var _S243 : u32 = _S236 + u32(9);
                    var _S244 : u32 = _S241 + u32(9);
                    var gap_1 : f32 = _S242 - support_of_0(rma_0, vec3<f32>(work_0[_S243], work_0[_S243 + u32(1)], work_0[_S243 + u32(2)]), nrm_3) - support_of_0(quat_matrix_0(node_0(nb_2, u32(1))), vec3<f32>(work_0[_S244], work_0[_S244 + u32(1)], work_0[_S244 + u32(2)]), nrm_3);
                    if(gap_1 > (0.25f * _S229))
                    {
                        j_12 = j_12 + u32(1);
                        continue;
                    }
                    if((params_0.count_only_0) == u32(0))
                    {
                        var t1_6 : vec3<f32> = tangent_of_0(nrm_3, rma_0);
                        var _S245 : vec3<f32> = cross(nrm_3, t1_6);
                        var fa_1 : f32 = work_0[_S236 + u32(13)];
                        var fbr_1 : f32 = work_0[_S241 + u32(13)];
                        var _S246 : f32;
                        if((params_0.contact_friction_0) >= 0.0f)
                        {
                            _S246 = params_0.contact_friction_0;
                        }
                        else
                        {
                            _S246 = min(fa_1, fbr_1);
                        }
                        var _S247 : u32 = (bitcast<u32>((node_0(na_2, u32(12)).z)));
                        var _S248 : u32 = (bitcast<u32>((node_0(nb_2, u32(12)).z)));
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
                            var row_8 : u32 = _S232 + count_6 + side_3;
                            var _S249 : bool = side_3 == u32(0);
                            var sn_2 : u32;
                            if(_S249)
                            {
                                sn_2 = na_2;
                            }
                            else
                            {
                                sn_2 = nb_2;
                            }
                            var dn_2 : u32;
                            if(_S249)
                            {
                                dn_2 = nb_2;
                            }
                            else
                            {
                                dn_2 = na_2;
                            }
                            var ft1_2 : vec3<f32>;
                            if(_S249)
                            {
                                ft1_2 = t1_6;
                            }
                            else
                            {
                                ft1_2 = (vec3<f32>(0) - t1_6);
                            }
                            var fn_2 : vec3<f32>;
                            if(_S249)
                            {
                                fn_2 = nrm_3;
                            }
                            else
                            {
                                fn_2 = (vec3<f32>(0) - nrm_3);
                            }
                            var rel_1 : vec3<f32> = node_0(sn_2, u32(3)).xyz - node_0(dn_2, u32(3)).xyz;
                            var _S250 : f32 = dot(ft1_2, rel_1);
                            var _S251 : f32 = dot(_S245, rel_1);
                            var _S252 : f32 = dot(fn_2, rel_1);
                            var t_con_0 : f32;
                            if(gap_1 <= 0.0f)
                            {
                                t_con_0 = -1.0f;
                            }
                            else
                            {
                                if(_S252 > 0.0f)
                                {
                                    _S226 = (gap_1 / _S252) < (params_0.dt_0);
                                }
                                else
                                {
                                    _S226 = false;
                                }
                                if(_S226)
                                {
                                    t_con_0 = gap_1 / _S252 / params_0.dt_0;
                                }
                                else
                                {
                                    t_con_0 = 1.0f;
                                }
                            }
                            put_edge_0(u32(6), row_8, global_node_0(g_18, sn_2 - _S227), global_node_0(g_18, dn_2 - _S227));
                            put_0(_S233, u32(6), row_8, u32(0), _S252 / _S230);
                            put_0(_S233, u32(6), row_8, u32(1), _S250 / _S230);
                            put_0(_S233, u32(6), row_8, u32(2), _S251 / _S230);
                            put_0(_S233, u32(6), row_8, u32(3), _S252 / _S231);
                            put_0(_S233, u32(6), row_8, u32(4), ln_ratio_0(node_0(dn_2, u32(0)).w, node_0(sn_2, u32(0)).w));
                            put_0(_S233, u32(6), row_8, u32(5), t_con_0);
                            put_0(_S233, u32(6), row_8, u32(6), max(- gap_1, 0.0f) / _S229);
                            put_0(_S233, u32(6), row_8, u32(7), _S246);
                            put_0(_S233, u32(6), row_8, u32(8), 0.0f);
                            var _S253 : f32;
                            if(_S247 == _S248)
                            {
                                _S253 = 1.0f;
                            }
                            else
                            {
                                _S253 = 0.0f;
                            }
                            put_0(_S233, u32(6), row_8, u32(9), _S253);
                            put_rot6_0(_S233, u32(6), row_8, u32(10), ft1_2, _S245, quat_matrix_0(node_0(sn_2, u32(1))));
                            put_rot6_0(_S233, u32(6), row_8, u32(16), ft1_2, _S245, quat_matrix_0(node_0(dn_2, u32(1))));
                            work_0[u32(24) * sn_2 + u32(4)] = 1.0f;
                            side_3 = side_3 + u32(1);
                        }
                        put_key_0(u32(6), (_S232 + count_6) / u32(2), _S237, rb_1);
                    }
                    count_6 = count_6 + u32(2);
                    j_12 = j_12 + u32(1);
                }
                e_3 = e_3 + u32(1);
                count_5 = count_6;
            }
            ja_0 = ja_0 + u32(1);
            count_4 = count_5;
        }
        ea_0 = ea_0 + u32(1);
    }
    if((params_0.count_only_0) != u32(0))
    {
        edge_count_0(g_18, u32(2), count_4);
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn feat_prefix()
{
    var _S254 : u32 = ctl_0[i32(1)];
    var g_19 : u32 = u32(0);
    var tot0_0 : u32 = u32(0);
    var tot1_0 : u32 = u32(0);
    var tot2_0 : u32 = u32(0);
    loop
    {
        if(g_19 < _S254)
        {
        }
        else
        {
            break;
        }
        var _S255 : u32 = u32(8) + u32(6) * g_19;
        fctl_0[_S255 + u32(3)] = tot0_0;
        fctl_0[_S255 + u32(4)] = tot1_0;
        fctl_0[_S255 + u32(5)] = tot2_0;
        var tot0_1 : u32 = tot0_0 + fctl_0[_S255];
        var tot1_1 : u32 = tot1_0 + fctl_0[_S255 + u32(1)];
        var tot2_1 : u32 = tot2_0 + fctl_0[_S255 + u32(2)];
        g_19 = g_19 + u32(1);
        tot0_0 = tot0_1;
        tot1_0 = tot1_1;
        tot2_0 = tot2_1;
    }
    var over_0 : u32;
    if(tot0_0 > (u32(2) * params_0.caps_0.x))
    {
        over_0 = u32(1);
    }
    else
    {
        over_0 = u32(0);
    }
    if(tot1_0 > (u32(2) * params_0.caps_0.y))
    {
        over_0 = (over_0 | (u32(2)));
    }
    if(tot2_0 > (u32(2) * params_0.caps_0.z))
    {
        over_0 = (over_0 | (u32(4)));
    }
    g_19 = u32(0);
    var cand_over_0 : u32 = u32(0);
    loop
    {
        if(g_19 < _S254)
        {
        }
        else
        {
            break;
        }
        var cand_over_1 : u32 = (cand_over_0 | ((bedge_0[params_0.max_bonds_cap_0 + u32(8) * call_first_0(g_19)].y)));
        g_19 = g_19 + u32(1);
        cand_over_0 = cand_over_1;
    }
    fctl_0[i32(5)] = cand_over_0;
    fctl_0[i32(0)] = tot0_0;
    fctl_0[i32(1)] = tot1_0;
    fctl_0[i32(2)] = tot2_0;
    fctl_0[i32(3)] = over_0;
    g_19 = u32(0);
    var chunks_n_0 : u32 = u32(0);
    loop
    {
        if(g_19 < _S254)
        {
        }
        else
        {
            break;
        }
        var chunks_n_1 : u32 = chunks_n_0 + call_chunks_0(g_19);
        g_19 = g_19 + u32(1);
        chunks_n_0 = chunks_n_1;
    }
    fctl_0[i32(4)] = chunks_n_0;
    return;
}

fn call_nodes_0( g_20 : u32) -> u32
{
    return (bitcast<u32>((calls_0[u32(10) * g_20 + u32(2)].w)));
}

fn cap4_0( x_1 : f32) -> f32
{
    var _S256 : f32;
    if(x_1 < 3.00000000549775576e+38f)
    {
        _S256 = min(x_1, 4.0f);
    }
    else
    {
        _S256 = 4.0f;
    }
    return _S256;
}

@compute
@workgroup_size(64, 1, 1)
fn feat_nodes(@builtin(workgroup_id) group_7 : vec3<u32>, @builtin(local_invocation_id) thread_9 : vec3<u32>)
{
    var ke_imp_0 : f32;
    var _S257 : f32;
    var _S258 : bool;
    var g_21 : u32 = group_7.x;
    var t_14 : u32 = thread_9.x;
    var j_13 : u32;
    if(g_21 >= ctl_0[i32(1)])
    {
        j_13 = u32(1);
    }
    else
    {
        j_13 = u32(0);
    }
    var _S259 : u32 = group_uniform_0(j_13, t_14);
    if(_S259 != u32(0))
    {
        return;
    }
    var _S260 : u32 = call_first_0(g_21);
    var n_9 : u32 = group_uniform_0(call_nodes_0(g_21), t_14);
    var nc_3 : u32 = group_uniform_0(call_chunks_0(g_21), t_14);
    var _S261 : u32 = group_uniform_0(call_nbonds_0(g_21), t_14);
    var _S262 : u32 = u32(10) * g_21;
    var L_1 : f32 = calls_0[_S262].x;
    var _S263 : f32 = calls_0[_S262].y;
    var vref_0 : f32 = calls_0[_S262].z;
    var c_wave_1 : f32 = calls_0[_S262].w;
    var _S264 : f32 = calls_0[_S262 + u32(1)].y;
    var dt_2 : f32 = params_0.dt_0;
    var grav_0 : vec3<f32> = params_0.gravity_0.xyz;
    var _S265 : vec4<f32> = vec4<f32>(0.0f);
    j_13 = t_14;
    var a_10 : vec4<f32> = _S265;
    var b_6 : vec4<f32> = _S265;
    loop
    {
        if(j_13 < n_9)
        {
        }
        else
        {
            break;
        }
        var _S266 : u32 = _S260 + j_13;
        var m_4 : f32 = node_0(_S266, u32(0)).w;
        var a_11 : vec4<f32> = a_10 + vec4<f32>(node_0(_S266, u32(3)).xyz * vec3<f32>(m_4), m_4);
        var _S267 : bool = j_13 < nc_3;
        if(_S267)
        {
            ke_imp_0 = m_4;
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        if(_S267)
        {
            _S258 = (node_0(_S266, u32(2)).w) != 0.0f;
        }
        else
        {
            _S258 = false;
        }
        if(_S258)
        {
            _S257 = 1.0f;
        }
        else
        {
            _S257 = 0.0f;
        }
        var b_7 : vec4<f32> = b_6 + vec4<f32>(ke_imp_0, _S257, 0.0f, 0.0f);
        j_13 = j_13 + u32(64);
        a_10 = a_11;
        b_6 = b_7;
    }
    var a_12 : vec4<f32> = group_sum_0(t_14, a_10);
    var b_8 : vec4<f32> = group_sum_0(t_14, b_6);
    var _S268 : vec3<f32> = a_12.xyz / vec3<f32>(a_12.w);
    var _S269 : f32 = b_8.x;
    var anchored_0 : bool = (b_8.y) > 0.0f;
    var _S270 : u32 = _S262 + u32(9);
    var _S271 : f32 = calls_0[_S270].y;
    var _S272 : f32 = calls_0[_S270].z;
    var _S273 : u32 = params_0.bases_0.x;
    var _S274 : u32 = params_0.bases_0.y;
    var fbg_0 : u32 = params_0.bases_0.z;
    var _S275 : u32 = params_0.bases2_0.w;
    j_13 = t_14;
    ke_imp_0 = 0.0f;
    loop
    {
        if(j_13 < n_9)
        {
        }
        else
        {
            break;
        }
        var k_10 : u32 = _S260 + j_13;
        var rb_2 : mat3x3<f32> = quat_matrix_0(node_0(k_10, u32(1)));
        var bt_0 : mat3x3<f32> = transpose(rb_2);
        var m_5 : f32 = node_0(k_10, u32(0)).w;
        var gb_1 : vec3<f32> = (((grav_0) * (bt_0))) / vec3<f32>(9.81000041961669922f);
        var q_4 : vec4<f32> = node_0(k_10, u32(1));
        var _S276 : u32 = _S275 + u32(7) * global_node_0(g_21, j_13);
        feat_0[_S276] = node_0(k_10, u32(0)).x;
        feat_0[_S276 + u32(1)] = node_0(k_10, u32(0)).y;
        feat_0[_S276 + u32(2)] = node_0(k_10, u32(0)).z;
        feat_0[_S276 + u32(3)] = q_4.w;
        feat_0[_S276 + u32(4)] = q_4.x;
        feat_0[_S276 + u32(5)] = q_4.y;
        feat_0[_S276 + u32(6)] = q_4.z;
        var c_3 : u32;
        if(j_13 < nc_3)
        {
            var row_9 : u32 = call_cbase_0(g_21) + j_13;
            var inr_0 : vec3<f32> = node_0(k_10, u32(2)).xyz;
            var _S277 : u32 = u32(24) * k_10;
            var _S278 : u32 = _S277 + u32(9);
            var fixed_0 : f32 = work_0[_S277 + u32(14)];
            var nlive_3 : f32 = work_0[_S277 + u32(7)];
            var vr_0 : vec3<f32> = (((node_0(k_10, u32(3)).xyz - _S268) * (bt_0))) / vec3<f32>(vref_0);
            var wb_0 : vec3<f32> = (((node_0(k_10, u32(4)).xyz) * (bt_0))) * vec3<f32>((L_1 / vref_0));
            put_0(_S273, u32(0), row_9, u32(0), ln_ratio_0(m_5, _S263));
            var _S279 : f32 = m_5 * L_1 * L_1;
            put_0(_S273, u32(0), row_9, u32(1), inr_0.x / _S279);
            put_0(_S273, u32(0), row_9, u32(2), inr_0.y / _S279);
            put_0(_S273, u32(0), row_9, u32(3), inr_0.z / _S279);
            put_0(_S273, u32(0), row_9, u32(4), work_0[_S278] / L_1);
            put_0(_S273, u32(0), row_9, u32(5), work_0[_S278 + u32(1)] / L_1);
            put_0(_S273, u32(0), row_9, u32(6), work_0[_S278 + u32(2)] / L_1);
            if(work_0[_S277 + u32(14)] == 1.0f)
            {
                _S257 = 1.0f;
            }
            else
            {
                _S257 = 0.0f;
            }
            put_0(_S273, u32(0), row_9, u32(7), _S257);
            var _S280 : f32;
            if(fixed_0 == 2.0f)
            {
                _S280 = 1.0f;
            }
            else
            {
                _S280 = 0.0f;
            }
            put_0(_S273, u32(0), row_9, u32(8), _S280);
            var _S281 : f32;
            if(nlive_3 > 0.0f)
            {
                _S281 = work_0[_S277 + u32(5)] / nlive_3;
            }
            else
            {
                _S281 = 0.0f;
            }
            put_0(_S273, u32(0), row_9, u32(9), _S281);
            put_0(_S273, u32(0), row_9, u32(10), work_0[_S277 + u32(6)]);
            put_0(_S273, u32(0), row_9, u32(11), nlive_3 / 6.0f);
            put_0(_S273, u32(0), row_9, u32(12), work_0[_S277 + u32(8)] / 6.0f);
            put_0(_S273, u32(0), row_9, u32(13), cap4_0(work_0[_S277] / 16.0f));
            put_0(_S273, u32(0), row_9, u32(14), cap4_0(work_0[_S277 + u32(1)] / dt_2));
            put_0(_S273, u32(0), row_9, u32(15), cap4_0(work_0[_S277 + u32(2)] / 16.0f));
            put_0(_S273, u32(0), row_9, u32(16), gb_1.x);
            put_0(_S273, u32(0), row_9, u32(17), gb_1.y);
            put_0(_S273, u32(0), row_9, u32(18), gb_1.z);
            put_0(_S273, u32(0), row_9, u32(19), vr_0.x);
            put_0(_S273, u32(0), row_9, u32(20), vr_0.y);
            put_0(_S273, u32(0), row_9, u32(21), vr_0.z);
            put_0(_S273, u32(0), row_9, u32(22), wb_0.x);
            put_0(_S273, u32(0), row_9, u32(23), wb_0.y);
            put_0(_S273, u32(0), row_9, u32(24), wb_0.z);
            var _S282 : f32;
            if(work_0[_S277 + u32(3)] != 0.0f)
            {
                _S282 = 1.0f;
            }
            else
            {
                _S282 = 0.0f;
            }
            put_0(_S273, u32(0), row_9, u32(25), _S282);
            c_3 = u32(26);
            loop
            {
                if(c_3 < u32(32))
                {
                }
                else
                {
                    break;
                }
                put_0(_S273, u32(0), row_9, c_3, 0.0f);
                c_3 = c_3 + u32(1);
            }
            put_0(_S273, u32(0), row_9, u32(32), work_0[_S277 + u32(4)]);
            var c_4 : u32 = u32(33);
            loop
            {
                if(c_4 < u32(39))
                {
                }
                else
                {
                    break;
                }
                put_0(_S273, u32(0), row_9, c_4, 0.0f);
                c_4 = c_4 + u32(1);
            }
        }
        else
        {
            var row_10 : u32 = call_ibase_0(g_21) + (j_13 - nc_3);
            var _S283 : u32 = u32(4) * (bitcast<u32>((node_0(k_10, u32(12)).x))) + u32(3);
            var r_face_0 : f32 = imps_0[_S283].z;
            var inr_1 : vec3<f32> = node_0(k_10, u32(2)).xyz;
            var r2_0 : f32 = r_face_0 * r_face_0;
            var vr_1 : vec3<f32> = (((node_0(k_10, u32(3)).xyz - _S268) * (bt_0)));
            var v_5 : vec3<f32> = node_0(k_10, u32(3)).xyz;
            var w_3 : vec3<f32> = node_0(k_10, u32(4)).xyz;
            var _S284 : f32 = inr_1.x;
            var _S285 : f32 = inr_1.y;
            var _S286 : f32 = inr_1.z;
            var ke_imp_1 : f32 = ke_imp_0 + (0.5f * m_5 * dot(v_5, v_5) + 0.5f * dot(w_3, (((w_3) * ((((bt_0) * ((((mat3x3<f32>(vec3<f32>(_S284, 0.0f, 0.0f), vec3<f32>(0.0f, _S285, 0.0f), vec3<f32>(0.0f, 0.0f, _S286))) * (rb_2)))))))))));
            put_0(_S274, u32(1), row_10, u32(0), ln_ratio_0(m_5, _S269));
            put_0(_S274, u32(1), row_10, u32(1), ln_ratio_0(m_5, _S263));
            var _S287 : f32 = m_5 * r2_0;
            put_0(_S274, u32(1), row_10, u32(2), _S284 / _S287);
            put_0(_S274, u32(1), row_10, u32(3), _S285 / _S287);
            put_0(_S274, u32(1), row_10, u32(4), _S286 / _S287);
            put_0(_S274, u32(1), row_10, u32(5), ln_ratio_0(r_face_0, L_1));
            put_0(_S274, u32(1), row_10, u32(6), ln_ratio_0(imps_0[_S283].x, _S264));
            put_0(_S274, u32(1), row_10, u32(7), imps_0[_S283].w);
            put_0(_S274, u32(1), row_10, u32(8), gb_1.x);
            put_0(_S274, u32(1), row_10, u32(9), gb_1.y);
            put_0(_S274, u32(1), row_10, u32(10), gb_1.z);
            put_0(_S274, u32(1), row_10, u32(11), vr_1.x / vref_0);
            put_0(_S274, u32(1), row_10, u32(12), vr_1.y / vref_0);
            put_0(_S274, u32(1), row_10, u32(13), vr_1.z / vref_0);
            put_0(_S274, u32(1), row_10, u32(14), length(node_0(k_10, u32(3)).xyz - _S268) / c_wave_1);
            put_0(_S274, u32(1), row_10, u32(15), work_0[u32(24) * k_10 + u32(4)]);
            c_3 = u32(16);
            loop
            {
                if(c_3 < u32(22))
                {
                }
                else
                {
                    break;
                }
                put_0(_S274, u32(1), row_10, c_3, 0.0f);
                c_3 = c_3 + u32(1);
            }
            ke_imp_0 = ke_imp_1;
        }
        j_13 = j_13 + u32(64);
    }
    var _S288 : vec4<f32> = group_sum_0(t_14, vec4<f32>(ke_imp_0, 0.0f, 0.0f, 0.0f));
    var ke_imp_2 : f32 = _S288.x;
    if(t_14 == u32(0))
    {
        var ni_0 : u32 = n_9 - nc_3;
        var _S289 : bool = ke_imp_2 > 0.0f;
        if(_S289)
        {
            _S258 = _S271 > 0.0f;
        }
        else
        {
            _S258 = false;
        }
        if(_S258)
        {
            ke_imp_0 = ln_ratio_0(ke_imp_2, _S271);
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        put_0(fbg_0, u32(2), g_21, u32(0), ke_imp_0);
        if(_S289)
        {
            _S258 = _S272 > 0.0f;
        }
        else
        {
            _S258 = false;
        }
        if(_S258)
        {
            ke_imp_0 = ln_ratio_0(ke_imp_2, _S272);
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        put_0(fbg_0, u32(2), g_21, u32(1), ke_imp_0);
        put_0(fbg_0, u32(2), g_21, u32(2), log(dt_2 * c_wave_1 / L_1));
        put_0(fbg_0, u32(2), g_21, u32(3), dt_2 * vref_0 / L_1);
        put_0(fbg_0, u32(2), g_21, u32(4), f32(ni_0) / 4.0f);
        put_0(fbg_0, u32(2), g_21, u32(5), log(f32(nc_3)) / 10.0f);
        put_0(fbg_0, u32(2), g_21, u32(6), length(grav_0) / 9.81000041961669922f);
        if(anchored_0)
        {
            ke_imp_0 = 1.0f;
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        put_0(fbg_0, u32(2), g_21, u32(7), ke_imp_0);
        put_0(fbg_0, u32(2), g_21, u32(8), 0.0f);
        put_0(fbg_0, u32(2), g_21, u32(9), 0.0f);
        put_0(fbg_0, u32(2), g_21, u32(10), 0.0f);
        put_0(fbg_0, u32(2), g_21, u32(11), vref_0 / c_wave_1);
    }
    return;
}

fn fail_margin_of_0( js_1 : ptr<function, JointState_std430_0>,  s8_0 : vec4<f32>,  rebar_0 : bool,  live_0 : bool) -> f32
{
    var ductile_0 : bool = (s8_0.x) != 0.0f;
    var _S290 : bool = !((s8_0.y) != 0.0f);
    var kf_ts_0 : f32;
    if(_S290)
    {
        kf_ts_0 = 1.0f;
    }
    else
    {
        if(ductile_0)
        {
            kf_ts_0 = 0.5f * ((*js_1).ductility_0 + 1.0f);
        }
        else
        {
            kf_ts_0 = max((*js_1).ductility_0, 1.0f);
        }
    }
    var kf_c_0 : f32;
    if(_S290)
    {
        kf_c_0 = 1.0f;
    }
    else
    {
        if(ductile_0)
        {
            kf_c_0 = 0.5f * ((*js_1).ductility_c_0 + 1.0f);
        }
        else
        {
            kf_c_0 = max((*js_1).ductility_c_0, 1.0f);
        }
    }
    var _S291 : f32 = max((*js_1).kappa_0 / kf_ts_0, (*js_1).kappa_c_0 / kf_c_0);
    var _S292 : bool;
    if(rebar_0)
    {
        _S292 = (s8_0.z) > 0.0f;
    }
    else
    {
        _S292 = false;
    }
    var q_5 : f32;
    if(_S292)
    {
        q_5 = min(_S291, (*js_1).rebar_work_0 / s8_0.z);
    }
    else
    {
        q_5 = _S291;
    }
    if(live_0)
    {
        _S292 = q_5 >= 1.0f;
    }
    else
    {
        _S292 = false;
    }
    if(_S292)
    {
        q_5 = 0.99999994039535522f;
    }
    return q_5;
}

@compute
@workgroup_size(64, 1, 1)
fn feat_bonds(@builtin(global_invocation_id) id_4 : vec3<u32>)
{
    var b_9 : u32 = id_4.x;
    if(b_9 >= ctl_0[i32(3)])
    {
        return;
    }
    var be_4 : vec4<u32> = bedge_0[b_9];
    var _S293 : u32 = bedge_0[b_9].x;
    var _S294 : bool;
    if(_S293 == u32(4294967295))
    {
        _S294 = true;
    }
    else
    {
        _S294 = (be_4.y) == u32(4294967295);
    }
    if(_S294)
    {
        return;
    }
    var _S295 : u32 = u32(10) * (bitcast<u32>((node_0(_S293, u32(3)).w)));
    var L_2 : f32 = calls_0[_S295].x;
    var _S296 : f32 = calls_0[_S295].z;
    var _S297 : f32 = calls_0[_S295 + u32(1)].x;
    var e_med_1 : f32 = calls_0[_S295 + u32(1)].y;
    var _S298 : f32 = calls_0[_S295 + u32(1)].z;
    var _S299 : u32 = u32(9) * be_4.z;
    var g0_1 : vec4<f32> = sb_0[_S299];
    var _S300 : vec4<f32> = sb_0[_S299 + u32(3)];
    var k4_0 : vec4<f32> = sb_0[_S299 + u32(4)];
    var st0_0 : vec4<f32> = sb_0[_S299 + u32(5)];
    var st1_0 : vec4<f32> = sb_0[_S299 + u32(6)];
    var s8_1 : vec4<f32> = sb_0[_S299 + u32(8)];
    var nb_3 : u32 = be_4.y;
    var _S301 : u32 = u32(24) * _S293;
    var _S302 : u32 = _S301 + u32(17);
    var rs_1 : mat3x3<f32> = (((transpose(quat_matrix_0(vec4<f32>(work_0[_S302], work_0[_S302 + u32(1)], work_0[_S302 + u32(2)], work_0[_S302 + u32(3)])))) * (quat_matrix_0(node_0(_S293, u32(1))))));
    var xa_1 : vec3<f32> = node_0(_S293, u32(13)).xyz;
    var nrm_4 : vec3<f32> = (((sb_0[_S299 + u32(1)].xyz) * (rs_1)));
    var t1_7 : vec3<f32> = (((sb_0[_S299 + u32(2)].xyz) * (rs_1)));
    var nrm_5 : vec3<f32>;
    var t1_8 : vec3<f32>;
    if((dot(nrm_4, node_0(nb_3, u32(13)).xyz - xa_1)) < 0.0f)
    {
        var _S303 : vec3<f32> = (vec3<f32>(0) - t1_7);
        nrm_5 = (vec3<f32>(0) - nrm_4);
        t1_8 = _S303;
    }
    else
    {
        nrm_5 = nrm_4;
        t1_8 = t1_7;
    }
    var _S304 : vec3<f32> = cross(nrm_5, t1_8);
    var _S305 : u32 = _S301 + u32(21);
    var _S306 : vec3<f32> = xa_1 + (((g0_1.xyz - vec3<f32>(work_0[_S305], work_0[_S305 + u32(1)], work_0[_S305 + u32(2)])) * (rs_1)));
    var area_1 : f32 = g0_1.w;
    var _S307 : f32 = e_med_1 * area_1;
    var _S308 : f32 = _S307 / L_2;
    var _S309 : f32 = _S307 * L_2;
    var rebar_1 : bool = (k4_0.w) != 0.0f;
    var _S310 : JointState_std430_0 = bond_dyn_0[be_4.w].js_0;
    var _S311 : f32 = fail_margin_of_0(&(_S310), s8_1, rebar_1, true);
    var pa_0 : u32 = u32(work_0[_S301 + u32(3)]);
    var pb_0 : u32 = u32(work_0[u32(24) * nb_3 + u32(3)]);
    var in_plug_0 : bool = (pa_0 != u32(0)) != (pb_0 != u32(0));
    var ratio_1 : f32;
    if(in_plug_0)
    {
        ratio_1 = ln_ratio_0(contacts_0[u32(6) * ((bitcast<u32>((calls_0[_S295 + u32(8)].x))) + max(pa_0, pb_0) - u32(1)) + u32(4)].x, 1.0f);
    }
    else
    {
        ratio_1 = 0.0f;
    }
    var _S312 : f32 = st0_0.x;
    var _S313 : f32 = st0_0.y;
    var _S314 : f32 = st0_0.z;
    var _S315 : f32 = st1_0.y;
    var _S316 : f32 = st1_0.z;
    var _S317 : f32 = st1_0.w;
    var _S318 : u32 = params_0.bases_0.w;
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
        var r_8 : u32 = u32(2) * b_9 + side_4;
        var _S319 : bool = side_4 == u32(0);
        var sn_3 : u32;
        if(_S319)
        {
            sn_3 = _S293;
        }
        else
        {
            sn_3 = nb_3;
        }
        var dn_3 : u32;
        if(_S319)
        {
            dn_3 = nb_3;
        }
        else
        {
            dn_3 = _S293;
        }
        var ft1_3 : vec3<f32>;
        if(_S319)
        {
            ft1_3 = t1_8;
        }
        else
        {
            ft1_3 = (vec3<f32>(0) - t1_8);
        }
        var fn_3 : vec3<f32>;
        if(_S319)
        {
            fn_3 = nrm_5;
        }
        else
        {
            fn_3 = (vec3<f32>(0) - nrm_5);
        }
        var q_6 : vec3<f32> = _S306 - node_0(sn_3, u32(13)).xyz;
        var dv_2 : vec3<f32> = node_0(dn_3, u32(3)).xyz + cross(node_0(dn_3, u32(4)).xyz, _S306 - node_0(dn_3, u32(13)).xyz) - (node_0(sn_3, u32(3)).xyz + cross(node_0(sn_3, u32(4)).xyz, _S306 - node_0(sn_3, u32(13)).xyz));
        put_0(_S318, u32(3), r_8, u32(0), ln_ratio_0(area_1, L_2 * L_2));
        put_0(_S318, u32(3), r_8, u32(1), dot(ft1_3, q_6) / L_2);
        put_0(_S318, u32(3), r_8, u32(2), dot(_S304, q_6) / L_2);
        put_0(_S318, u32(3), r_8, u32(3), dot(fn_3, q_6) / L_2);
        put_0(_S318, u32(3), r_8, u32(4), ln_ratio_0(_S300.y, _S308));
        var _S320 : f32 = ln_ratio_0(_S300.z, _S308);
        put_0(_S318, u32(3), r_8, u32(5), _S320);
        put_0(_S318, u32(3), r_8, u32(6), _S320);
        put_0(_S318, u32(3), r_8, u32(7), ln_ratio_0(_S300.w, _S309));
        put_0(_S318, u32(3), r_8, u32(8), ln_ratio_0(k4_0.x, _S309));
        put_0(_S318, u32(3), r_8, u32(9), ln_ratio_0(k4_0.y, _S309));
        put_0(_S318, u32(3), r_8, u32(10), ln_ratio_0(_S312, _S297));
        put_0(_S318, u32(3), r_8, u32(11), ln_ratio_0(_S314, _S297));
        put_0(_S318, u32(3), r_8, u32(12), ln_ratio_0(_S313, _S297));
        put_0(_S318, u32(3), r_8, u32(13), ln_ratio_0(_S315 * area_1, _S298));
        put_0(_S318, u32(3), r_8, u32(14), ln_ratio_0(_S316 * area_1, _S298));
        put_0(_S318, u32(3), r_8, u32(15), ln_ratio_0(_S317 * area_1, _S298));
        put_0(_S318, u32(3), r_8, u32(16), k4_0.z);
        put_0(_S318, u32(3), r_8, u32(17), _S310.damage_0);
        put_0(_S318, u32(3), r_8, u32(18), _S310.crush_0);
        put_0(_S318, u32(3), r_8, u32(19), _S311);
        var _S321 : f32 = log(1.0f + max(_S310.utilization_0, 0.0f));
        put_0(_S318, u32(3), r_8, u32(20), _S321);
        put_0(_S318, u32(3), r_8, u32(21), _S321);
        var _S322 : f32;
        if(rebar_1)
        {
            _S322 = 1.0f;
        }
        else
        {
            _S322 = 0.0f;
        }
        put_0(_S318, u32(3), r_8, u32(22), _S322);
        put_0(_S318, u32(3), r_8, u32(23), 0.0f);
        var _S323 : f32;
        if(in_plug_0)
        {
            _S323 = 1.0f;
        }
        else
        {
            _S323 = 0.0f;
        }
        put_0(_S318, u32(3), r_8, u32(24), _S323);
        put_0(_S318, u32(3), r_8, u32(25), ratio_1);
        put_0(_S318, u32(3), r_8, u32(26), dot(ft1_3, dv_2) / _S296);
        put_0(_S318, u32(3), r_8, u32(27), dot(_S304, dv_2) / _S296);
        put_0(_S318, u32(3), r_8, u32(28), dot(fn_3, dv_2) / _S296);
        put_rot6_0(_S318, u32(3), r_8, u32(29), ft1_3, _S304, quat_matrix_0(node_0(sn_3, u32(1))));
        put_rot6_0(_S318, u32(3), r_8, u32(35), ft1_3, _S304, quat_matrix_0(node_0(dn_3, u32(1))));
        put_0(_S318, u32(3), r_8, u32(41), 0.0f);
        put_0(_S318, u32(3), r_8, u32(42), 0.0f);
        put_0(_S318, u32(3), r_8, u32(43), 0.0f);
        side_4 = side_4 + u32(1);
    }
    return;
}

struct Quat_0
{
     w_4 : f32,
     x_2 : f32,
     y_1 : f32,
     z_1 : f32,
};

fn quat_of_0( q_7 : vec4<f32>) -> Quat_0
{
    var r_9 : Quat_0;
    r_9.x_2 = q_7.x;
    r_9.y_1 = q_7.y;
    r_9.z_1 = q_7.z;
    r_9.w_4 = q_7.w;
    return r_9;
}

fn quat_mul_0( a_13 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_10 : Quat_0;
    r_10.w_4 = a_13.w_4 * o_0.w_4 - a_13.x_2 * o_0.x_2 - a_13.y_1 * o_0.y_1 - a_13.z_1 * o_0.z_1;
    r_10.x_2 = a_13.w_4 * o_0.x_2 + a_13.x_2 * o_0.w_4 + a_13.y_1 * o_0.z_1 - a_13.z_1 * o_0.y_1;
    r_10.y_1 = a_13.w_4 * o_0.y_1 - a_13.x_2 * o_0.z_1 + a_13.y_1 * o_0.w_4 + a_13.z_1 * o_0.x_2;
    r_10.z_1 = a_13.w_4 * o_0.z_1 + a_13.x_2 * o_0.y_1 - a_13.y_1 * o_0.x_2 + a_13.z_1 * o_0.w_4;
    return r_10;
}

fn quat_vec_0( q_8 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_8.x_2, q_8.y_1, q_8.z_1, q_8.w_4);
}

fn qmulf_0( a_14 : vec4<f32>,  b_10 : vec4<f32>) -> vec4<f32>
{
    return quat_vec_0(quat_mul_0(quat_of_0(a_14), quat_of_0(b_10)));
}

fn to_zup_f_0( v_6 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_6.x, - v_6.z, v_6.y);
}

fn rotate_0( q_9 : Quat_0,  v_7 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_9.x_2, q_9.y_1, q_9.z_1);
    var t_15 : vec3<f32> = cross(qv_0, v_7) * vec3<f32>(2.0f);
    return v_7 + t_15 * vec3<f32>(q_9.w_4) + cross(qv_0, t_15);
}

fn inverse_rotate_0( q_10 : Quat_0,  v_8 : vec3<f32>) -> vec3<f32>
{
    var c_5 : Quat_0;
    c_5.w_4 = q_10.w_4;
    c_5.x_2 = - q_10.x_2;
    c_5.y_1 = - q_10.y_1;
    c_5.z_1 = - q_10.z_1;
    return rotate_0(c_5, v_8);
}

fn rotvec_quat_f_0( r_11 : vec3<f32>) -> vec4<f32>
{
    var th_0 : f32 = length(r_11);
    if(th_0 < 9.999999960041972e-13f)
    {
        return normalize(vec4<f32>(vec3<f32>(0.5f) * r_11, 1.0f));
    }
    var _S324 : f32 = 0.5f * th_0;
    return vec4<f32>(r_11 / vec3<f32>(th_0) * vec3<f32>(sin(_S324)), cos(_S324));
}

fn qconj_0( q_11 : vec4<f32>) -> vec4<f32>
{
    return vec4<f32>((vec3<f32>(0) - q_11.xyz), q_11.w);
}

fn quat_rotvec_0( q_12 : vec4<f32>) -> vec3<f32>
{
    var p_2 : vec4<f32>;
    if((q_12.w) < 0.0f)
    {
        p_2 = (vec4<f32>(0) - q_12);
    }
    else
    {
        p_2 = q_12;
    }
    var _S325 : vec3<f32> = p_2.xyz;
    var s_6 : f32 = length(_S325);
    if(s_6 < 9.999999960041972e-13f)
    {
        return vec3<f32>(2.0f) * _S325;
    }
    return _S325 / vec3<f32>(s_6) * vec3<f32>((2.0f * atan2(s_6, p_2.w)));
}

@compute
@workgroup_size(64, 1, 1)
fn feat_deform(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var k_11 : u32 = id_5.x;
    if(k_11 >= (params_0.chunk_count_0))
    {
        return;
    }
    var info_2 : vec4<u32> = chunks_0[k_11].info_1;
    var kn_0 : vec2<u32> = key_node_0[sinc_0[u32(2) + chunks_0[k_11].info_1.z] + chunks_0[k_11].info_1.w];
    var _S326 : u32 = key_node_0[sinc_0[u32(2) + chunks_0[k_11].info_1.z] + chunks_0[k_11].info_1.w].x;
    var _S327 : bool;
    if(_S326 == u32(0))
    {
        _S327 = true;
    }
    else
    {
        _S327 = (kn_0.y) != (params_0.prev_frame_0);
    }
    if(_S327)
    {
        return;
    }
    var n_10 : u32 = _S326 - u32(1);
    if((bitcast<u32>((calls_0[u32(10) * (bitcast<u32>((node_0(n_10, u32(3)).w))) + u32(4)].w))) != u32(0))
    {
        return;
    }
    var i_4 : u32 = info_2.y;
    var anchored_1 : bool = (((islands_0[i_4].info_0.x) & (u32(1)))) != u32(0);
    var q_13 : vec4<f32> = islands_0[i_4].rotation_0;
    var com_b_0 : vec3<f32> = islands_0[i_4].com_0.xyz;
    var _S328 : vec3<f32> = vec3<f32>(0.0f);
    var _S329 : u32 = u32(3) * i_4;
    var body_2 : u32 = (bitcast<u32>((ctab_0[params_0.engine_base_0 + _S329].x)));
    var com_w_0 : vec3<f32>;
    var com_b_1 : vec3<f32>;
    var wb_1 : vec3<f32>;
    var vb_0 : vec3<f32>;
    var q_14 : vec4<f32>;
    if(body_2 != u32(4294967295))
    {
        var _S330 : u32 = u32(7) * body_2;
        var _S331 : vec4<f32> = normalize(qmulf_0(qmulf_0(vec4<f32>(0.70710676908493042f, 0.0f, 0.0f, 0.70710676908493042f), bodies_0[_S330 + u32(1)]), ctab_0[params_0.engine_base_0 + _S329 + u32(1)]));
        var _S332 : vec3<f32> = ctab_0[params_0.engine_base_0 + _S329 + u32(2)].xyz;
        var _S333 : vec3<f32> = to_zup_f_0(bodies_0[_S330 + u32(3)].xyz);
        var _S334 : vec3<f32> = to_zup_f_0(bodies_0[_S330 + u32(4)].xyz);
        com_w_0 = to_zup_f_0(bodies_0[_S330].xyz);
        q_14 = _S331;
        com_b_1 = _S332;
        wb_1 = _S334;
        vb_0 = _S333;
    }
    else
    {
        com_w_0 = islands_0[i_4].position_0.xyz + rotate_0(quat_of_0(q_13), com_b_0);
        q_14 = q_13;
        com_b_1 = com_b_0;
        wb_1 = _S328;
        vb_0 = _S328;
    }
    var _S335 : Quat_0 = quat_of_0(q_14);
    var center_1 : vec3<f32> = chunks_0[k_11].center_0.xyz;
    var u_1 : vec3<f32> = inverse_rotate_0(_S335, node_0(n_10, u32(13)).xyz + node_0(n_10, u32(10)).xyz - (com_w_0 - rotate_0(_S335, com_b_1))) - center_1;
    var _S336 : u32 = u32(24) * n_10 + u32(17);
    var th_1 : vec3<f32> = quat_rotvec_0(normalize(qmulf_0(qconj_0(q_14), normalize(qmulf_0(rotvec_quat_f_0(node_0(n_10, u32(11)).xyz), qmulf_0(node_0(n_10, u32(1)), qconj_0(vec4<f32>(work_0[_S336], work_0[_S336 + u32(1)], work_0[_S336 + u32(2)], work_0[_S336 + u32(3)]))))))));
    var r_12 : vec3<f32> = rotate_0(_S335, center_1 + u_1 - com_b_1);
    var v1_0 : vec3<f32> = node_0(n_10, u32(8)).xyz;
    var w1_0 : vec3<f32> = node_0(n_10, u32(9)).xyz;
    var vh_0 : vec3<f32>;
    if(anchored_1)
    {
        vh_0 = inverse_rotate_0(_S335, v1_0 - cross(wb_1, r_12));
    }
    else
    {
        vh_0 = inverse_rotate_0(_S335, v1_0 - vb_0 - cross(wb_1, r_12));
    }
    var wh_0 : vec3<f32>;
    if(anchored_1)
    {
        wh_0 = inverse_rotate_0(_S335, w1_0);
    }
    else
    {
        wh_0 = inverse_rotate_0(_S335, w1_0 - wb_1);
    }
    var _S337 : u32 = u32(4) * k_11;
    state_0[_S337] = vec4<f32>(u_1, state_0[_S337].w);
    var _S338 : u32 = _S337 + u32(1);
    state_0[_S338] = vec4<f32>(th_1, state_0[_S338].w);
    var _S339 : u32 = _S337 + u32(2);
    state_0[_S339] = vec4<f32>(vh_0, state_0[_S339].w);
    var _S340 : u32 = _S337 + u32(3);
    state_0[_S340] = vec4<f32>(wh_0, state_0[_S340].w);
    return;
}

