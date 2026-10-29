@binding(11) @group(0) var<storage, read_write> meta_0 : array<atomic<u32>>;

@binding(12) @group(0) var<storage, read> topo_args_0 : array<u32>;

@binding(1) @group(0) var<storage, read> islands_0 : array<vec4<u32>>;

struct BlastParams_std140_0
{
    @align(16) island_count_0 : u32,
    @align(4) max_iterations_0 : u32,
    @align(8) warm_0 : u32,
    @align(4) skip_settled_0 : u32,
    @align(16) tolerance_0 : f32,
    @align(4) recip_length_0 : f32,
    @align(8) node_count_0 : u32,
    @align(4) bond_count_0 : u32,
    @align(16) linear_scale_0 : f32,
    @align(4) angular_scale_0 : f32,
    @align(8) bend_gain_max_0 : f32,
    @align(4) fibres_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) damage_rate_0 : f32,
    @align(8) apply_damage_0 : u32,
    @align(4) force_tolerance_0 : f32,
    @align(16) gravity_0 : vec4<f32>,
    @align(16) w_alive_0 : u32,
    @align(4) a_dirty_0 : u32,
    @align(8) a_counters_0 : u32,
    @align(4) wide_min_0 : u32,
    @align(16) wide_rows_0 : u32,
    @align(4) wide_blocks_0 : u32,
    @align(8) pad2_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : BlastParams_std140_0;
@binding(10) @group(0) var<storage, read_write> wide_0 : array<vec4<f32>>;

@binding(9) @group(0) var<storage, read> lists_0 : array<u32>;

struct BlastNode_std430_0
{
    @align(16) w_ang_0 : f32,
    @align(4) w_lin_0 : f32,
    @align(8) ref_begin_0 : u32,
    @align(4) ref_end_0 : u32,
    @align(16) position_0 : vec4<f32>,
};

@binding(2) @group(0) var<storage, read> nodes_0 : array<BlastNode_std430_0>;

@binding(5) @group(0) var<storage, read> accel_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read_write> scratch_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read> refs_0 : array<u32>;

struct BlastBond_std430_0
{
    @align(16) n0_0 : u32,
    @align(4) n1_0 : u32,
    @align(8) s_0 : f32,
    @align(4) health_0 : f32,
    @align(16) o0_0 : vec4<f32>,
    @align(16) o1_0 : vec4<f32>,
};

@binding(4) @group(0) var<storage, read_write> bonds_0 : array<BlastBond_std430_0>;

@binding(6) @group(0) var<storage, read_write> x_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read_write> report_0 : array<vec4<u32>>;

fn isnan_0( x_1 : f32) -> bool
{
    var _S1 : u32 = (bitcast<u32>((x_1)));
    var _S2 : u32 = (_S1 & (u32(8388607)));
    var _S3 : bool;
    if(((((_S1 >> (u32(23)))) & (u32(255)))) == u32(255))
    {
        _S3 = _S2 != u32(0);
    }
    else
    {
        _S3 = false;
    }
    return _S3;
}

fn isinf_0( x_2 : f32) -> bool
{
    var _S4 : u32 = (bitcast<u32>((x_2)));
    var _S5 : u32 = (_S4 & (u32(8388607)));
    var _S6 : bool;
    if(((((_S4 >> (u32(23)))) & (u32(255)))) == u32(255))
    {
        _S6 = _S5 == u32(0);
    }
    else
    {
        _S6 = false;
    }
    return _S6;
}

fn isfinite_0( x_3 : f32) -> bool
{
    var _S7 : bool;
    if(isinf_0(x_3))
    {
        _S7 = true;
    }
    else
    {
        _S7 = isnan_0(x_3);
    }
    return !_S7;
}

@compute
@workgroup_size(1, 1, 1)
fn w_reset(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    atomicStore(&(meta_0[u32(0)]), u32(0));
    atomicStore(&(meta_0[u32(1)]), u32(0));
    atomicStore(&(meta_0[u32(2)]), u32(0));
    return;
}

fn blocks_0( n_0 : u32) -> u32
{
    return (n_0 + u32(1024) - u32(1)) / u32(1024);
}

@compute
@workgroup_size(64, 1, 1)
fn w_classify(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var k_0 : u32 = id_1.x;
    if(k_0 >= topo_args_0[i32(3)])
    {
        return;
    }
    var r_0 : vec4<u32> = islands_0[k_0];
    var _S8 : u32 = r_0.y - r_0.x;
    if(_S8 <= (params_0.wide_min_0))
    {
        return;
    }
    var row_0 : u32 = atomicAdd(&(meta_0[u32(0)]), u32(1));
    if(row_0 < (params_0.wide_rows_0))
    {
        atomicStore(&(meta_0[u32(16) + row_0]), k_0);
    }
    var _S9 : u32 = atomicMax(&(meta_0[u32(1)]), blocks_0(_S8));
    var _S10 : u32 = atomicMax(&(meta_0[u32(2)]), blocks_0(r_0.w - r_0.z));
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn w_args(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var _S11 : u32 = atomicLoad(&(meta_0[u32(0)]));
    var _S12 : u32 = min(_S11, params_0.wide_rows_0);
    var _S13 : bool = _S12 > u32(0);
    var nb_0 : u32;
    if(_S13)
    {
        var _S14 : u32 = atomicLoad(&(meta_0[u32(1)]));
        nb_0 = _S14;
    }
    else
    {
        nb_0 = u32(0);
    }
    var bb_0 : u32;
    if(_S13)
    {
        var _S15 : u32 = atomicLoad(&(meta_0[u32(2)]));
        bb_0 = _S15;
    }
    else
    {
        bb_0 = u32(0);
    }
    atomicStore(&(meta_0[u32(4)]), nb_0);
    atomicStore(&(meta_0[u32(5)]), _S12);
    atomicStore(&(meta_0[u32(6)]), u32(1));
    atomicStore(&(meta_0[u32(8)]), bb_0);
    atomicStore(&(meta_0[u32(9)]), _S12);
    atomicStore(&(meta_0[u32(10)]), u32(1));
    atomicStore(&(meta_0[u32(12)]), _S12);
    atomicStore(&(meta_0[u32(13)]), u32(1));
    atomicStore(&(meta_0[u32(14)]), u32(1));
    return;
}

fn row_island_0( row_1 : u32) -> u32
{
    var _S16 : u32 = atomicLoad(&(meta_0[u32(16) + row_1]));
    return _S16;
}

fn st_0( row_2 : u32,  slot_0 : u32) -> u32
{
    return row_2 * u32(16) + slot_0;
}

fn flags_of_0( row_3 : u32) -> u32
{
    return (bitcast<u32>((wide_0[st_0(row_3, u32(2))].w)));
}

var<workgroup> g_group_uniform4_0 : vec4<u32>;

fn group_uniform4_0( value_0 : vec4<u32>,  thread_0 : u32) -> vec4<u32>
{
    if(thread_0 == u32(0))
    {
        g_group_uniform4_0 = value_0;
    }
    var _S17 : vec4<u32> = (workgroupUniformLoad(&((g_group_uniform4_0))));
    return _S17;
}

struct Block_0
{
     row_4 : u32,
     island_0 : u32,
     begin_0 : u32,
     end_0 : u32,
     flags_0 : u32,
};

fn node_block_0( group_0 : vec3<u32>,  tid_0 : u32) -> Block_0
{
    var b_0 : Block_0;
    var _S18 : u32 = group_0.y;
    b_0.row_4 = _S18;
    var _S19 : u32 = row_island_0(_S18);
    var u_0 : vec4<u32> = group_uniform4_0(vec4<u32>(_S19, u32(0), u32(0), flags_of_0(_S18)), tid_0);
    var _S20 : u32 = u_0.x;
    b_0.island_0 = _S20;
    b_0.flags_0 = u_0.w;
    var g_0 : vec4<u32> = group_uniform4_0(islands_0[_S20], tid_0);
    var _S21 : u32 = g_0.x + group_0.x * u32(1024);
    b_0.begin_0 = _S21;
    b_0.end_0 = min(_S21 + u32(1024), g_0.y);
    return b_0;
}

fn ns_0( n_1 : u32,  v_0 : u32) -> u32
{
    return n_1 * u32(24) + v_0;
}

fn norm2_0( a_0 : vec3<f32>,  b_1 : vec3<f32>) -> f32
{
    return dot(a_0, a_0) + dot(b_1, b_1);
}

fn set_v_0( n_2 : u32,  v_1 : u32,  a_1 : vec3<f32>,  l_0 : vec3<f32>)
{
    var _S22 : u32 = ns_0(n_2, v_1);
    scratch_0[_S22] = vec4<f32>(a_1, 0.0f);
    scratch_0[_S22 + u32(1)] = vec4<f32>(l_0, 0.0f);
    return;
}

fn build_block_0( node_0 : u32) -> bool
{
    var n_3 : BlastNode_std430_0 = nodes_0[node_0];
    var _S23 : vec3<f32> = vec3<f32>(0.0f);
    var r_1 : u32 = n_3.ref_begin_0;
    var diag_0 : vec3<f32> = _S23;
    var offd_0 : vec3<f32> = _S23;
    var ks_0 : vec3<f32> = _S23;
    var cs_0 : f32 = 0.0f;
    loop
    {
        if(r_1 < (n_3.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_0 : u32 = refs_0[r_1];
        var b_2 : BlastBond_std430_0 = bonds_0[(ref_0 & (u32(2147483647)))];
        if(!((bonds_0[(ref_0 & (u32(2147483647)))].health_0) > 0.0f))
        {
            r_1 = r_1 + u32(1);
            continue;
        }
        var o_0 : vec3<f32>;
        if(((ref_0 & (u32(2147483648)))) != u32(0))
        {
            o_0 = b_2.o1_0.xyz;
        }
        else
        {
            o_0 = b_2.o0_0.xyz;
        }
        var _S24 : f32 = b_2.s_0;
        var s2_0 : f32 = _S24 * _S24;
        var _S25 : vec3<f32> = vec3<f32>(s2_0);
        var _S26 : f32 = o_0.x;
        var _S27 : f32 = o_0.y;
        var _S28 : f32 = o_0.z;
        var offd_1 : vec3<f32> = offd_0 + _S25 * (vec3<f32>(0) - vec3<f32>(_S26 * _S27, _S26 * _S28, _S27 * _S28));
        var ks_1 : vec3<f32> = ks_0 + _S25 * o_0;
        var cs_1 : f32 = cs_0 + s2_0;
        diag_0 = diag_0 + _S25 * (vec3<f32>((1.0f + dot(o_0, o_0))) - o_0 * o_0);
        offd_0 = offd_1;
        ks_0 = ks_1;
        cs_0 = cs_1;
        r_1 = r_1 + u32(1);
    }
    var _S29 : f32 = n_3.w_ang_0;
    var da_0 : f32 = _S29 * _S29;
    var c_0 : f32 = n_3.w_lin_0 * n_3.w_lin_0 * cs_0;
    var k_1 : vec3<f32> = ks_0 * vec3<f32>((n_3.w_ang_0 * n_3.w_lin_0));
    var a0_0 : vec4<f32> = vec4<f32>(diag_0 * vec3<f32>(da_0), offd_0.x * da_0);
    var _S30 : f32 = offd_0.y * da_0;
    var _S31 : f32 = offd_0.z * da_0;
    var a1_0 : vec4<f32> = vec4<f32>(_S30, _S31, c_0, 0.0f);
    var _S32 : u32 = ns_0(node_0, u32(18));
    scratch_0[_S32] = a0_0;
    scratch_0[_S32 + u32(1)] = a1_0;
    scratch_0[_S32 + u32(2)] = vec4<f32>(k_1, 0.0f);
    if(!(c_0 > 0.0f))
    {
        return false;
    }
    var ic_0 : f32 = 1.0f / c_0;
    var kk_0 : f32 = dot(k_1, k_1);
    var _S33 : f32 = k_1.x;
    var sxx_0 : f32 = a0_0.x + (_S33 * _S33 - kk_0) * ic_0;
    var _S34 : f32 = k_1.y;
    var syy_0 : f32 = a0_0.y + (_S34 * _S34 - kk_0) * ic_0;
    var _S35 : f32 = k_1.z;
    var szz_0 : f32 = a0_0.z + (_S35 * _S35 - kk_0) * ic_0;
    var sxy_0 : f32 = a0_0.w + _S33 * _S34 * ic_0;
    var sxz_0 : f32 = _S30 + _S33 * _S35 * ic_0;
    var syz_0 : f32 = _S31 + _S34 * _S35 * ic_0;
    var cxx_0 : f32 = syy_0 * szz_0 - syz_0 * syz_0;
    var cxy_0 : f32 = sxz_0 * syz_0 - sxy_0 * szz_0;
    var cxz_0 : f32 = sxy_0 * syz_0 - sxz_0 * syy_0;
    var det_0 : f32 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    var _S36 : bool;
    if(!(det_0 > 0.0f))
    {
        _S36 = true;
    }
    else
    {
        _S36 = !isfinite_0(det_0);
    }
    if(_S36)
    {
        return false;
    }
    var id_3 : f32 = 1.0f / det_0;
    var cyz_0 : f32 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    var _S37 : u32 = ns_0(node_0, u32(21));
    scratch_0[_S37] = vec4<f32>(cxx_0 * id_3, (sxx_0 * szz_0 - sxz_0 * sxz_0) * id_3, (sxx_0 * syy_0 - sxy_0 * sxy_0) * id_3, cxy_0 * id_3);
    scratch_0[_S37 + u32(1)] = vec4<f32>(cxz_0 * id_3, cyz_0 * id_3, ic_0, 0.0f);
    scratch_0[_S37 + u32(2)] = vec4<f32>(k_1 * vec3<f32>(ic_0), 0.0f);
    return true;
}

var<workgroup> g_blast_red_0 : array<vec2<f32>, i32(1024)>;

var<workgroup> g_blast_sum_0 : vec2<f32>;

fn blast_group_sum_0( tid_1 : u32,  v_2 : vec2<f32>) -> vec2<f32>
{
    g_blast_red_0[tid_1] = v_2;
    workgroupBarrier();
    var k_2 : u32 = u32(512);
    loop
    {
        if(k_2 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_1 < k_2)
        {
            g_blast_red_0[tid_1] = g_blast_red_0[tid_1] + g_blast_red_0[tid_1 + k_2];
        }
        workgroupBarrier();
        k_2 = (k_2 >> (u32(1)));
    }
    if(tid_1 == u32(0))
    {
        g_blast_sum_0 = g_blast_red_0[i32(0)];
    }
    var s_1 : vec2<f32> = (workgroupUniformLoad(&((g_blast_sum_0))));
    workgroupBarrier();
    return s_1;
}

fn pt_0( row_5 : u32,  block_0 : u32,  k_3 : u32) -> u32
{
    return params_0.wide_rows_0 * u32(16) + (row_5 * params_0.wide_blocks_0 + block_0) * u32(6) + k_3;
}

fn write_partial_0( tid_2 : u32,  row_6 : u32,  block_1 : u32,  k_4 : u32,  v_3 : vec4<f32>)
{
    var a_2 : vec2<f32> = blast_group_sum_0(tid_2, v_3.xy);
    var b_3 : vec2<f32> = blast_group_sum_0(tid_2, v_3.zw);
    if(tid_2 == u32(0))
    {
        wide_0[pt_0(row_6, block_1, k_4)] = vec4<f32>(a_2, b_3);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_init_nodes(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var tid_3 : u32 = thread_1.x;
    var b_4 : Block_0 = node_block_0(group_1, tid_3);
    if((b_4.begin_0) >= (b_4.end_0))
    {
        return;
    }
    var _S38 : vec4<f32> = vec4<f32>(0.0f);
    var p0_0 : vec4<f32> = _S38;
    var p_0 : u32 = b_4.begin_0 + tid_3;
    var p1_0 : vec4<f32>;
    if(p_0 < (b_4.end_0))
    {
        var i_0 : u32 = lists_0[p_0];
        var n_4 : BlastNode_std430_0 = nodes_0[i_0];
        var _S39 : u32 = i_0 * u32(2);
        var a_ang_0 : vec4<f32> = accel_0[_S39];
        var a_lin_0 : vec4<f32> = accel_0[_S39 + u32(1)];
        var _S40 : u32 = ns_0(i_0, u32(16));
        var _S41 : bool;
        if((any((a_ang_0 != scratch_0[_S40]))))
        {
            _S41 = true;
        }
        else
        {
            _S41 = (any((a_lin_0 != scratch_0[_S40 + u32(1)])));
        }
        if(_S41)
        {
            p0_0[i32(2)] = 1.0f;
        }
        var _S42 : vec3<f32> = (vec3<f32>(0) - a_ang_0.xyz);
        var _S43 : f32;
        if((n_4.w_ang_0) > 0.0f)
        {
            _S43 = 1.0f / n_4.w_ang_0;
        }
        else
        {
            _S43 = 1.0f;
        }
        var ba_0 : vec3<f32> = _S42 * vec3<f32>(_S43);
        var _S44 : vec3<f32> = (vec3<f32>(0) - a_lin_0.xyz);
        if((n_4.w_lin_0) > 0.0f)
        {
            _S43 = params_0.recip_length_0 / n_4.w_lin_0;
        }
        else
        {
            _S43 = params_0.recip_length_0;
        }
        var bl_0 : vec3<f32> = _S44 * vec3<f32>(_S43);
        p0_0[i32(0)] = norm2_0(ba_0, bl_0);
        set_v_0(i_0, u32(14), ba_0, bl_0);
        set_v_0(i_0, u32(0), ba_0, bl_0);
        scratch_0[_S40] = a_ang_0;
        scratch_0[_S40 + u32(1)] = a_lin_0;
        var _S45 : vec3<f32> = vec3<f32>(0.0f);
        set_v_0(i_0, u32(6), _S45, _S45);
        set_v_0(i_0, u32(2), _S45, _S45);
        var _S46 : bool = build_block_0(i_0);
        if(!_S46)
        {
            p0_0[i32(1)] = 1.0f;
        }
        var w_0 : f32 = 1.0f / (n_4.w_lin_0 * n_4.w_lin_0);
        p1_0 = vec4<f32>(n_4.position_0.xyz * vec3<f32>(w_0), w_0);
    }
    else
    {
        p1_0 = _S38;
    }
    var _S47 : u32 = group_1.x;
    write_partial_0(tid_3, b_4.row_4, _S47, u32(0), p0_0);
    write_partial_0(tid_3, b_4.row_4, _S47, u32(1), p1_0);
    return;
}

fn bond_block_0( group_2 : vec3<u32>,  tid_4 : u32) -> Block_0
{
    var _S48 : Block_0 = node_block_0(group_2, tid_4);
    var b_5 : Block_0 = _S48;
    var g_1 : vec4<u32> = group_uniform4_0(islands_0[b_5.island_0], tid_4);
    var _S49 : u32 = g_1.z + group_2.x * u32(1024);
    b_5.begin_0 = _S49;
    b_5.end_0 = min(_S49 + u32(1024), g_1.w);
    return b_5;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_init_bonds(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var tid_5 : u32 = thread_2.x;
    var b_6 : Block_0 = bond_block_0(group_3, tid_5);
    if((b_6.begin_0) >= (b_6.end_0))
    {
        return;
    }
    var p_1 : u32 = b_6.begin_0 + tid_5;
    var anchor_0 : f32;
    if(p_1 < (b_6.end_0))
    {
        var j_0 : u32 = lists_0[p_1];
        var bd_0 : BlastBond_std430_0 = bonds_0[j_0];
        var _S50 : bool;
        if((bonds_0[j_0].health_0) > 0.0f)
        {
            if(!((nodes_0[bd_0.n0_0].w_lin_0) > 0.0f))
            {
                _S50 = true;
            }
            else
            {
                _S50 = !((nodes_0[bd_0.n1_0].w_lin_0) > 0.0f);
            }
        }
        else
        {
            _S50 = false;
        }
        if(_S50)
        {
            anchor_0 = 1.0f;
        }
        else
        {
            anchor_0 = 0.0f;
        }
        if((params_0.warm_0) == u32(0))
        {
            var _S51 : u32 = j_0 * u32(2);
            var _S52 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S51] = _S52;
            x_0[_S51 + u32(1)] = _S52;
        }
    }
    else
    {
        anchor_0 = 0.0f;
    }
    write_partial_0(tid_5, b_6.row_4, group_3.x, u32(2), vec4<f32>(anchor_0, 0.0f, 0.0f, 0.0f));
    return;
}

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_1 : u32,  thread_3 : u32) -> u32
{
    if(thread_3 == u32(0))
    {
        g_group_uniform_0 = value_1;
    }
    var _S53 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S53;
}

fn row_sum_0( tid_6 : u32,  row_7 : u32,  count_0 : u32,  k_5 : u32) -> vec4<f32>
{
    var _S54 : vec4<f32> = vec4<f32>(0.0f);
    var b_7 : u32 = tid_6;
    var s_2 : vec4<f32> = _S54;
    loop
    {
        if(b_7 < count_0)
        {
        }
        else
        {
            break;
        }
        var s_3 : vec4<f32> = s_2 + wide_0[pt_0(row_7, b_7, k_5)];
        b_7 = b_7 + u32(1024);
        s_2 = s_3;
    }
    var x_4 : vec2<f32> = blast_group_sum_0(tid_6, s_2.xy);
    var y_0 : vec2<f32> = blast_group_sum_0(tid_6, s_2.zw);
    return vec4<f32>(x_4, y_0);
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

@compute
@workgroup_size(1024, 1, 1)
fn w_setup(@builtin(workgroup_id) group_4 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var tid_7 : u32 = thread_4.x;
    var row_8 : u32 = group_4.x;
    var _S55 : u32 = row_island_0(row_8);
    var island_1 : u32 = group_uniform_0(_S55, tid_7);
    var r_2 : vec4<u32> = group_uniform4_0(islands_0[island_1], tid_7);
    var nb_1 : u32 = blocks_0(r_2.y - r_2.x);
    var bb_1 : u32 = blocks_0(r_2.w - r_2.z);
    var s0_0 : vec4<f32> = row_sum_0(tid_7, row_8, nb_1, u32(0));
    var s1_0 : vec4<f32> = row_sum_0(tid_7, row_8, nb_1, u32(1));
    var s2_1 : vec4<f32> = row_sum_0(tid_7, row_8, bb_1, u32(2));
    if(tid_7 != u32(0))
    {
        return;
    }
    var last_0 : vec4<u32> = report_0[island_1];
    var _S56 : bool;
    if((params_0.warm_0) != u32(0))
    {
        _S56 = (((last_0.y) & (u32(4)))) != u32(0);
    }
    else
    {
        _S56 = false;
    }
    if(_S56)
    {
        _S56 = (s0_0.z) == 0.0f;
    }
    else
    {
        _S56 = false;
    }
    var f_0 : u32;
    if(_S56)
    {
        f_0 = u32(128);
    }
    else
    {
        f_0 = u32(1);
    }
    var delta2_0 : f32 = s0_0.x * params_0.tolerance_0 * params_0.tolerance_0;
    if(f_0 == u32(1))
    {
        if((s0_0.y) > 0.0f)
        {
            f_0 = u32(4);
        }
        else
        {
            if(!(delta2_0 > 0.0f))
            {
                f_0 = u32(18);
            }
        }
    }
    if((s2_1.x) == 0.0f)
    {
        f_0 = (f_0 | (u32(8)));
    }
    var _S57 : f32 = float_bits_0(u32(2139095040));
    wide_0[st_0(row_8, u32(0))] = vec4<f32>(delta2_0, 0.0f, 0.0f, _S57);
    wide_0[st_0(row_8, u32(1))] = vec4<f32>(0.0f, 0.0f, 0.0f, _S57);
    var _S58 : f32 = (bitcast<f32>((u32(0))));
    wide_0[st_0(row_8, u32(2))] = vec4<f32>(0.0f, 0.0f, _S58, (bitcast<f32>((f_0))));
    var _S59 : vec3<f32>;
    if((s1_0.y) > 0.0f)
    {
        _S59 = s1_0.xyz / vec3<f32>(s1_0.w);
    }
    else
    {
        _S59 = vec3<f32>(0.0f);
    }
    wide_0[st_0(row_8, u32(3))] = vec4<f32>(_S59, _S58);
    return;
}

fn wide_rel_0( row_9 : u32,  n_5 : u32) -> vec3<f32>
{
    return nodes_0[n_5].position_0.xyz - wide_0[st_0(row_9, u32(3))].xyz;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_gram_nodes(@builtin(workgroup_id) group_5 : vec3<u32>, @builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    var tid_8 : u32 = thread_5.x;
    var b_8 : Block_0 = node_block_0(group_5, tid_8);
    var _S60 : bool;
    if((b_8.begin_0) >= (b_8.end_0))
    {
        _S60 = true;
    }
    else
    {
        _S60 = (((b_8.flags_0) & (u32(8)))) == u32(0);
    }
    if(_S60)
    {
        _S60 = true;
    }
    else
    {
        _S60 = (((b_8.flags_0) & (u32(1)))) == u32(0);
    }
    if(_S60)
    {
        return;
    }
    var rr_0 : array<f32, i32(6)>;
    var rt_0 : array<vec3<f32>, i32(3)>;
    var z_0 : u32 = u32(0);
    loop
    {
        if(z_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        rr_0[z_0] = 0.0f;
        z_0 = z_0 + u32(1);
    }
    z_0 = u32(0);
    loop
    {
        if(z_0 < u32(3))
        {
        }
        else
        {
            break;
        }
        rt_0[z_0] = vec3<f32>(0.0f);
        z_0 = z_0 + u32(1);
    }
    var p_2 : u32 = b_8.begin_0 + tid_8;
    var tt_0 : f32;
    if(p_2 < (b_8.end_0))
    {
        var i_1 : u32 = lists_0[p_2];
        var n_6 : BlastNode_std430_0 = nodes_0[i_1];
        var _S61 : f32 = n_6.w_ang_0;
        var wa_0 : f32 = 1.0f / (_S61 * _S61);
        var _S62 : f32 = n_6.w_lin_0;
        var wl_0 : f32 = 1.0f / (_S62 * _S62);
        var q_0 : vec3<f32> = wide_rel_0(b_8.row_4, i_1);
        var _S63 : vec3<f32> = cross(q_0, vec3<f32>(1.0f, 0.0f, 0.0f));
        var _S64 : vec3<f32> = cross(q_0, vec3<f32>(0.0f, 1.0f, 0.0f));
        var _S65 : vec3<f32> = cross(q_0, vec3<f32>(0.0f, 0.0f, 1.0f));
        var c_1 : array<vec3<f32>, i32(3)> = array<vec3<f32>, i32(3)>( _S63, _S64, _S65 );
        rr_0[i32(0)] = wa_0 + wl_0 * dot(_S63, _S63);
        rr_0[i32(1)] = wa_0 + wl_0 * dot(_S64, _S64);
        rr_0[i32(2)] = wa_0 + wl_0 * dot(_S65, _S65);
        rr_0[i32(3)] = wl_0 * dot(_S63, _S64);
        rr_0[i32(4)] = wl_0 * dot(_S63, _S65);
        rr_0[i32(5)] = wl_0 * dot(_S64, _S65);
        var kk_1 : u32 = u32(0);
        loop
        {
            if(kk_1 < u32(3))
            {
            }
            else
            {
                break;
            }
            rt_0[kk_1] = vec3<f32>(wl_0) * c_1[kk_1];
            kk_1 = kk_1 + u32(1);
        }
        tt_0 = wl_0;
    }
    else
    {
        tt_0 = 0.0f;
    }
    var _S66 : u32 = group_5.x;
    write_partial_0(tid_8, b_8.row_4, _S66, u32(0), vec4<f32>(rr_0[i32(0)], rr_0[i32(1)], rr_0[i32(2)], rr_0[i32(3)]));
    write_partial_0(tid_8, b_8.row_4, _S66, u32(1), vec4<f32>(rr_0[i32(4)], rr_0[i32(5)], rt_0[i32(0)].x, rt_0[i32(0)].y));
    write_partial_0(tid_8, b_8.row_4, _S66, u32(2), vec4<f32>(rt_0[i32(0)].z, rt_0[i32(1)].x, rt_0[i32(1)].y, rt_0[i32(1)].z));
    write_partial_0(tid_8, b_8.row_4, _S66, u32(3), vec4<f32>(rt_0[i32(2)], tt_0));
    return;
}

fn row_range_0( tid_9 : u32,  row_10 : u32) -> vec4<u32>
{
    var _S67 : u32 = row_island_0(row_10);
    var _S68 : u32 = group_uniform_0(_S67, tid_9);
    var _S69 : vec4<u32> = group_uniform4_0(islands_0[_S68], tid_9);
    return _S69;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_gram(@builtin(workgroup_id) group_6 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var tid_10 : u32 = thread_6.x;
    var row_11 : u32 = group_6.x;
    var f_1 : u32 = group_uniform_0(flags_of_0(row_11), tid_10);
    var _S70 : bool;
    if(((f_1 & (u32(8)))) == u32(0))
    {
        _S70 = true;
    }
    else
    {
        _S70 = ((f_1 & (u32(1)))) == u32(0);
    }
    if(_S70)
    {
        return;
    }
    var r_3 : vec4<u32> = row_range_0(tid_10, row_11);
    var nb_2 : u32 = blocks_0(r_3.y - r_3.x);
    var a_3 : vec4<f32> = row_sum_0(tid_10, row_11, nb_2, u32(0));
    var b_9 : vec4<f32> = row_sum_0(tid_10, row_11, nb_2, u32(1));
    var c_2 : vec4<f32> = row_sum_0(tid_10, row_11, nb_2, u32(2));
    var d_0 : vec4<f32> = row_sum_0(tid_10, row_11, nb_2, u32(3));
    if(tid_10 != u32(0))
    {
        return;
    }
    var m_0 : array<f32, i32(36)>;
    var z_1 : u32 = u32(0);
    loop
    {
        if(z_1 < u32(36))
        {
        }
        else
        {
            break;
        }
        m_0[z_1] = 0.0f;
        z_1 = z_1 + u32(1);
    }
    m_0[i32(0)] = a_3.x;
    m_0[i32(7)] = a_3.y;
    m_0[i32(14)] = a_3.z;
    var _S71 : f32 = a_3.w;
    m_0[i32(1)] = _S71;
    m_0[i32(6)] = _S71;
    var _S72 : f32 = b_9.x;
    m_0[i32(2)] = _S72;
    m_0[i32(12)] = _S72;
    var _S73 : f32 = b_9.y;
    m_0[i32(8)] = _S73;
    m_0[i32(13)] = _S73;
    var _S74 : vec3<f32> = vec3<f32>(b_9.z, b_9.w, c_2.x);
    var _S75 : vec3<f32> = vec3<f32>(c_2.y, c_2.z, c_2.w);
    var _S76 : vec3<f32> = d_0.xyz;
    var j_1 : u32 = u32(0);
    loop
    {
        if(j_1 < u32(3))
        {
        }
        else
        {
            break;
        }
        var _S77 : u32 = u32(3) + j_1;
        m_0[_S77] = _S74[j_1];
        m_0[u32(9) + j_1] = _S75[j_1];
        m_0[u32(15) + j_1] = _S76[j_1];
        var _S78 : u32 = _S77 * u32(6);
        m_0[_S78] = _S74[j_1];
        m_0[_S78 + u32(1)] = _S75[j_1];
        m_0[_S78 + u32(2)] = _S76[j_1];
        m_0[_S78 + u32(3) + j_1] = d_0.w;
        j_1 = j_1 + u32(1);
    }
    z_1 = u32(0);
    loop
    {
        if(z_1 < u32(9))
        {
        }
        else
        {
            break;
        }
        var _S79 : u32 = u32(4) * z_1;
        wide_0[st_0(row_11, u32(6) + z_1)] = vec4<f32>(m_0[_S79], m_0[_S79 + u32(1)], m_0[_S79 + u32(2)], m_0[_S79 + u32(3)]);
        z_1 = z_1 + u32(1);
    }
    return;
}

fn bs_0( j_2 : u32) -> u32
{
    return params_0.node_count_0 * u32(24) + u32(2) * j_2;
}

fn va_0( n_7 : u32,  v_4 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_7, v_4)].xyz;
}

fn vl_0( n_8 : u32,  v_5 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_8, v_5) + u32(1)].xyz;
}

fn bond_term_0( b_10 : ptr<function, BlastBond_std430_0>,  v_6 : u32,  ta_0 : ptr<function, vec3<f32>>,  tl_0 : ptr<function, vec3<f32>>)
{
    var m0_0 : BlastNode_std430_0 = nodes_0[(*b_10).n0_0];
    var m1_0 : BlastNode_std430_0 = nodes_0[(*b_10).n1_0];
    var a0_1 : vec3<f32> = va_0((*b_10).n0_0, v_6) * vec3<f32>(m0_0.w_ang_0);
    var l0_0 : vec3<f32> = vl_0((*b_10).n0_0, v_6) * vec3<f32>(m0_0.w_lin_0);
    var a1_1 : vec3<f32> = va_0((*b_10).n1_0, v_6) * vec3<f32>(m1_0.w_ang_0);
    var l1_0 : vec3<f32> = vl_0((*b_10).n1_0, v_6) * vec3<f32>(m1_0.w_lin_0);
    var _S80 : vec3<f32> = vec3<f32>((*b_10).s_0);
    (*ta_0) = (a0_1 - a1_1) * _S80;
    (*tl_0) = (l0_0 - l1_0 + (cross((*b_10).o0_0.xyz, a0_1) - cross((*b_10).o1_0.xyz, a1_1))) * _S80;
    return;
}

fn bond_term_1( b_11 : ptr<function, BlastBond_std430_0>,  v_7 : u32,  ta_1 : ptr<function, vec3<f32>>,  tl_1 : ptr<function, vec3<f32>>)
{
    var m0_1 : BlastNode_std430_0 = nodes_0[(*b_11).n0_0];
    var m1_1 : BlastNode_std430_0 = nodes_0[(*b_11).n1_0];
    var a0_2 : vec3<f32> = va_0((*b_11).n0_0, v_7) * vec3<f32>(m0_1.w_ang_0);
    var l0_1 : vec3<f32> = vl_0((*b_11).n0_0, v_7) * vec3<f32>(m0_1.w_lin_0);
    var a1_2 : vec3<f32> = va_0((*b_11).n1_0, v_7) * vec3<f32>(m1_1.w_ang_0);
    var l1_1 : vec3<f32> = vl_0((*b_11).n1_0, v_7) * vec3<f32>(m1_1.w_lin_0);
    var _S81 : vec3<f32> = vec3<f32>((*b_11).s_0);
    (*ta_1) = (a0_2 - a1_2) * _S81;
    (*tl_1) = (l0_1 - l1_1 + (cross((*b_11).o0_0.xyz, a0_2) - cross((*b_11).o1_0.xyz, a1_2))) * _S81;
    return;
}

fn gather_0( node_1 : u32,  from_t_0 : bool,  with_u_0 : bool,  ang_0 : ptr<function, vec3<f32>>,  lin_0 : ptr<function, vec3<f32>>)
{
    var n_9 : BlastNode_std430_0 = nodes_0[node_1];
    var _S82 : vec3<f32> = vec3<f32>(0.0f);
    var r_4 : u32 = n_9.ref_begin_0;
    var acc_a_0 : vec3<f32> = _S82;
    var acc_l_0 : vec3<f32> = _S82;
    loop
    {
        if(r_4 < (n_9.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_1 : u32 = refs_0[r_4];
        var j_3 : u32 = (ref_1 & (u32(2147483647)));
        var _S83 : BlastBond_std430_0 = bonds_0[j_3];
        if(!((_S83.health_0) > 0.0f))
        {
            r_4 = r_4 + u32(1);
            continue;
        }
        var ya_0 : vec3<f32>;
        var yl_0 : vec3<f32>;
        if(from_t_0)
        {
            var _S84 : u32 = bs_0(j_3);
            var _S85 : vec3<f32> = scratch_0[_S84 + u32(1)].xyz;
            ya_0 = scratch_0[_S84].xyz;
            yl_0 = _S85;
        }
        else
        {
            var _S86 : u32 = j_3 * u32(2);
            var ya_1 : vec3<f32> = x_0[_S86].xyz;
            var yl_1 : vec3<f32> = x_0[_S86 + u32(1)].xyz;
            if(with_u_0)
            {
                var ta_2 : vec3<f32>;
                var tl_2 : vec3<f32>;
                bond_term_1(&(_S83), u32(6), &(ta_2), &(tl_2));
                var yl_2 : vec3<f32> = yl_1 + tl_2;
                ya_0 = ya_1 + ta_2;
                yl_0 = yl_2;
            }
            else
            {
                ya_0 = ya_1;
                yl_0 = yl_1;
            }
        }
        var _S87 : vec3<f32> = vec3<f32>(_S83.s_0);
        var ya_2 : vec3<f32> = ya_0 * _S87;
        var yl_3 : vec3<f32> = yl_0 * _S87;
        var acc_a_1 : vec3<f32>;
        var acc_l_1 : vec3<f32>;
        if(((ref_1 & (u32(2147483648)))) != u32(0))
        {
            var acc_l_2 : vec3<f32> = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross(_S83.o1_0.xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            var acc_l_3 : vec3<f32> = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross(_S83.o0_0.xyz, yl_3));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_4 = r_4 + u32(1);
    }
    (*ang_0) = acc_a_0 * vec3<f32>(n_9.w_ang_0);
    (*lin_0) = acc_l_0 * vec3<f32>(n_9.w_lin_0);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_warm(@builtin(workgroup_id) group_7 : vec3<u32>, @builtin(local_invocation_id) thread_7 : vec3<u32>)
{
    var tid_11 : u32 = thread_7.x;
    var b_12 : Block_0 = node_block_0(group_7, tid_11);
    var _S88 : bool;
    if((b_12.begin_0) >= (b_12.end_0))
    {
        _S88 = true;
    }
    else
    {
        _S88 = (params_0.warm_0) == u32(0);
    }
    if(_S88)
    {
        _S88 = true;
    }
    else
    {
        _S88 = (((b_12.flags_0) & (u32(1)))) == u32(0);
    }
    if(_S88)
    {
        return;
    }
    var p_3 : u32 = b_12.begin_0 + tid_11;
    if(p_3 < (b_12.end_0))
    {
        var i_2 : u32 = lists_0[p_3];
        var a_4 : vec3<f32>;
        var l_1 : vec3<f32>;
        gather_0(i_2, false, false, &(a_4), &(l_1));
        set_v_0(i_2, u32(0), va_0(i_2, u32(14)) - a_4, vl_0(i_2, u32(14)) - l_1);
    }
    return;
}

fn runs_0( flags_1 : u32,  only_rebuild_0 : bool) -> bool
{
    var _S89 : bool;
    if(((flags_1 & (u32(1)))) != u32(0))
    {
        if(!only_rebuild_0)
        {
            _S89 = true;
        }
        else
        {
            _S89 = ((flags_1 & (u32(64)))) != u32(0);
        }
    }
    else
    {
        _S89 = false;
    }
    return _S89;
}

fn proj_part_0( group_8 : vec3<u32>,  tid_12 : u32,  v_8 : u32,  only_rebuild_1 : bool)
{
    var b_13 : Block_0 = node_block_0(group_8, tid_12);
    var _S90 : bool;
    if((b_13.begin_0) >= (b_13.end_0))
    {
        _S90 = true;
    }
    else
    {
        _S90 = !runs_0(b_13.flags_0, only_rebuild_1);
    }
    if(_S90)
    {
        _S90 = true;
    }
    else
    {
        _S90 = (((b_13.flags_0) & (u32(8)))) == u32(0);
    }
    if(_S90)
    {
        return;
    }
    var _S91 : vec3<f32> = vec3<f32>(0.0f);
    var p_4 : u32 = b_13.begin_0 + tid_12;
    var sr_0 : vec3<f32>;
    var stt_0 : vec3<f32>;
    if(p_4 < (b_13.end_0))
    {
        var i_3 : u32 = lists_0[p_4];
        var n_10 : BlastNode_std430_0 = nodes_0[i_3];
        var lin_1 : vec3<f32> = vl_0(i_3, v_8) / vec3<f32>(n_10.w_lin_0);
        sr_0 = va_0(i_3, v_8) / vec3<f32>(n_10.w_ang_0) - cross(wide_rel_0(b_13.row_4, i_3), lin_1);
        stt_0 = lin_1;
    }
    else
    {
        sr_0 = _S91;
        stt_0 = _S91;
    }
    var _S92 : u32 = group_8.x;
    write_partial_0(tid_12, b_13.row_4, _S92, u32(0), vec4<f32>(sr_0, stt_0.x));
    write_partial_0(tid_12, b_13.row_4, _S92, u32(1), vec4<f32>(stt_0.yz, 0.0f, 0.0f));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_part_r(@builtin(workgroup_id) g_2 : vec3<u32>, @builtin(local_invocation_id) t_0 : vec3<u32>)
{
    proj_part_0(g_2, t_0.x, u32(0), false);
    return;
}

fn proj_solve_0( group_9 : vec3<u32>,  tid_13 : u32,  only_rebuild_2 : bool)
{
    var k_6 : u32;
    var row_12 : u32 = group_9.x;
    var f_2 : u32 = group_uniform_0(flags_of_0(row_12), tid_13);
    var ok_0 : bool;
    if(!runs_0(f_2, only_rebuild_2))
    {
        ok_0 = true;
    }
    else
    {
        ok_0 = ((f_2 & (u32(8)))) == u32(0);
    }
    if(ok_0)
    {
        return;
    }
    var r_5 : vec4<u32> = row_range_0(tid_13, row_12);
    var nb_3 : u32 = blocks_0(r_5.y - r_5.x);
    var a_5 : vec4<f32> = row_sum_0(tid_13, row_12, nb_3, u32(0));
    var b_14 : vec4<f32> = row_sum_0(tid_13, row_12, nb_3, u32(1));
    if(tid_13 != u32(0))
    {
        return;
    }
    var g_3 : array<array<f32, i32(6)>, i32(6)>;
    var _S93 : f32 = a_5.y;
    var _S94 : f32 = a_5.z;
    var _S95 : f32 = a_5.w;
    var _S96 : f32 = b_14.x;
    var _S97 : f32 = b_14.y;
    var rhs_0 : array<f32, i32(6)>;
    rhs_0[i32(0)] = a_5.x;
    rhs_0[i32(1)] = _S93;
    rhs_0[i32(2)] = _S94;
    rhs_0[i32(3)] = _S95;
    rhs_0[i32(4)] = _S96;
    rhs_0[i32(5)] = _S97;
    var z_2 : u32 = u32(0);
    loop
    {
        if(z_2 < u32(36))
        {
        }
        else
        {
            break;
        }
        g_3[z_2 / u32(6)][z_2 % u32(6)] = wide_0[st_0(row_12, u32(6) + z_2 / u32(4))][z_2 % u32(4)];
        z_2 = z_2 + u32(1);
    }
    var c_3 : array<f32, i32(6)>;
    var col_0 : u32 = u32(0);
    loop
    {
        if(col_0 < u32(6))
        {
        }
        else
        {
            ok_0 = true;
            break;
        }
        var _S98 : u32 = col_0 + u32(1);
        var piv_0 : u32 = col_0;
        var rr_1 : u32 = _S98;
        loop
        {
            if(rr_1 < u32(6))
            {
            }
            else
            {
                break;
            }
            if((abs(g_3[rr_1][col_0])) > (abs(g_3[piv_0][col_0])))
            {
                piv_0 = rr_1;
            }
            rr_1 = rr_1 + u32(1);
        }
        k_6 = u32(0);
        loop
        {
            if(k_6 < u32(6))
            {
            }
            else
            {
                break;
            }
            var t_1 : f32 = g_3[col_0][k_6];
            g_3[col_0][k_6] = g_3[piv_0][k_6];
            g_3[piv_0][k_6] = t_1;
            k_6 = k_6 + u32(1);
        }
        var tb_0 : f32 = rhs_0[col_0];
        rhs_0[col_0] = rhs_0[piv_0];
        rhs_0[piv_0] = tb_0;
        if(!((abs(g_3[col_0][col_0])) > 0.0f))
        {
            ok_0 = false;
            break;
        }
        var rr_2 : u32 = _S98;
        loop
        {
            if(rr_2 < u32(6))
            {
            }
            else
            {
                break;
            }
            var fct_0 : f32 = g_3[rr_2][col_0] / g_3[col_0][col_0];
            var k_7 : u32 = col_0;
            loop
            {
                if(k_7 < u32(6))
                {
                }
                else
                {
                    break;
                }
                g_3[rr_2][k_7] = g_3[rr_2][k_7] - fct_0 * g_3[col_0][k_7];
                k_7 = k_7 + u32(1);
            }
            rhs_0[rr_2] = rhs_0[rr_2] - fct_0 * rhs_0[col_0];
            rr_2 = rr_2 + u32(1);
        }
        col_0 = _S98;
    }
    z_2 = u32(0);
    loop
    {
        if(z_2 < u32(6))
        {
        }
        else
        {
            break;
        }
        c_3[z_2] = 0.0f;
        z_2 = z_2 + u32(1);
    }
    if(ok_0)
    {
        var rr_3 : i32 = i32(5);
        loop
        {
            if(rr_3 >= i32(0))
            {
            }
            else
            {
                break;
            }
            k_6 = u32(rr_3) + u32(1);
            var s_4 : f32 = rhs_0[rr_3];
            loop
            {
                if(k_6 < u32(6))
                {
                }
                else
                {
                    break;
                }
                var s_5 : f32 = s_4 - g_3[rr_3][k_6] * c_3[k_6];
                k_6 = k_6 + u32(1);
                s_4 = s_5;
            }
            c_3[rr_3] = s_4 / g_3[rr_3][rr_3];
            rr_3 = rr_3 - i32(1);
        }
    }
    wide_0[st_0(row_12, u32(4))] = vec4<f32>(c_3[i32(0)], c_3[i32(1)], c_3[i32(2)], 0.0f);
    wide_0[st_0(row_12, u32(5))] = vec4<f32>(c_3[i32(3)], c_3[i32(4)], c_3[i32(5)], 0.0f);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_solve(@builtin(workgroup_id) g_4 : vec3<u32>, @builtin(local_invocation_id) t_2 : vec3<u32>)
{
    proj_solve_0(g_4, t_2.x, false);
    return;
}

fn proj_apply_0( group_10 : vec3<u32>,  tid_14 : u32,  v_9 : u32,  only_rebuild_3 : bool)
{
    var b_15 : Block_0 = node_block_0(group_10, tid_14);
    var _S99 : bool;
    if((b_15.begin_0) >= (b_15.end_0))
    {
        _S99 = true;
    }
    else
    {
        _S99 = !runs_0(b_15.flags_0, only_rebuild_3);
    }
    if(_S99)
    {
        _S99 = true;
    }
    else
    {
        _S99 = (((b_15.flags_0) & (u32(8)))) == u32(0);
    }
    if(_S99)
    {
        return;
    }
    var p_5 : u32 = b_15.begin_0 + tid_14;
    if(p_5 >= (b_15.end_0))
    {
        return;
    }
    var i_4 : u32 = lists_0[p_5];
    var n_11 : BlastNode_std430_0 = nodes_0[i_4];
    var w_1 : vec3<f32> = wide_0[st_0(b_15.row_4, u32(4))].xyz;
    set_v_0(i_4, v_9, va_0(i_4, v_9) - w_1 / vec3<f32>(n_11.w_ang_0), vl_0(i_4, v_9) - (wide_0[st_0(b_15.row_4, u32(5))].xyz + cross(wide_rel_0(b_15.row_4, i_4), w_1)) / vec3<f32>(n_11.w_lin_0));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_apply_r(@builtin(workgroup_id) g_5 : vec3<u32>, @builtin(local_invocation_id) t_3 : vec3<u32>)
{
    proj_apply_0(g_5, t_3.x, u32(0), false);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_part_rb(@builtin(workgroup_id) g_6 : vec3<u32>, @builtin(local_invocation_id) t_4 : vec3<u32>)
{
    proj_part_0(g_6, t_4.x, u32(0), true);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_solve_rb(@builtin(workgroup_id) g_7 : vec3<u32>, @builtin(local_invocation_id) t_5 : vec3<u32>)
{
    proj_solve_0(g_7, t_5.x, true);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_apply_rb(@builtin(workgroup_id) g_8 : vec3<u32>, @builtin(local_invocation_id) t_6 : vec3<u32>)
{
    proj_apply_0(g_8, t_6.x, u32(0), true);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_part_g(@builtin(workgroup_id) g_9 : vec3<u32>, @builtin(local_invocation_id) t_7 : vec3<u32>)
{
    proj_part_0(g_9, t_7.x, u32(8), false);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_apply_g(@builtin(workgroup_id) g_10 : vec3<u32>, @builtin(local_invocation_id) t_8 : vec3<u32>)
{
    proj_apply_0(g_10, t_8.x, u32(8), false);
    return;
}

fn bond_r_0( group_11 : vec3<u32>,  tid_15 : u32,  only_rebuild_4 : bool)
{
    var b_16 : Block_0 = bond_block_0(group_11, tid_15);
    var _S100 : bool;
    if((b_16.begin_0) >= (b_16.end_0))
    {
        _S100 = true;
    }
    else
    {
        _S100 = !runs_0(b_16.flags_0, only_rebuild_4);
    }
    if(_S100)
    {
        return;
    }
    var p_6 : u32 = b_16.begin_0 + tid_15;
    var z_3 : f32;
    var f_3 : f32;
    if(p_6 < (b_16.end_0))
    {
        var j_4 : u32 = lists_0[p_6];
        var _S101 : BlastBond_std430_0 = bonds_0[j_4];
        var _S102 : vec3<f32> = vec3<f32>(0.0f);
        var ta_3 : vec3<f32> = _S102;
        var tl_3 : vec3<f32> = _S102;
        if((_S101.health_0) > 0.0f)
        {
            bond_term_0(&(_S101), u32(0), &(ta_3), &(tl_3));
            if((params_0.force_tolerance_0) > 0.0f)
            {
                var ua_0 : vec3<f32>;
                var ul_0 : vec3<f32>;
                bond_term_0(&(_S101), u32(6), &(ua_0), &(ul_0));
                var _S103 : u32 = j_4 * u32(2);
                z_3 = norm2_0(x_0[_S103].xyz + ua_0, x_0[_S103 + u32(1)].xyz + ul_0);
            }
            else
            {
                z_3 = 0.0f;
            }
        }
        else
        {
            z_3 = 0.0f;
        }
        var _S104 : f32 = norm2_0(ta_3, tl_3);
        var _S105 : u32 = bs_0(j_4);
        scratch_0[_S105] = vec4<f32>(ta_3, 0.0f);
        scratch_0[_S105 + u32(1)] = vec4<f32>(tl_3, 0.0f);
        var _S106 : f32 = z_3;
        z_3 = _S104;
        f_3 = _S106;
    }
    else
    {
        z_3 = 0.0f;
        f_3 = 0.0f;
    }
    write_partial_0(tid_15, b_16.row_4, group_11.x, u32(2), vec4<f32>(z_3, f_3, 0.0f, 0.0f));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_bond_r(@builtin(workgroup_id) g_11 : vec3<u32>, @builtin(local_invocation_id) t_9 : vec3<u32>)
{
    bond_r_0(g_11, t_9.x, false);
    return;
}

fn node_rn_0( group_12 : vec3<u32>,  tid_16 : u32,  only_rebuild_5 : bool)
{
    var b_17 : Block_0 = node_block_0(group_12, tid_16);
    var _S107 : bool;
    if((b_17.begin_0) >= (b_17.end_0))
    {
        _S107 = true;
    }
    else
    {
        _S107 = !runs_0(b_17.flags_0, only_rebuild_5);
    }
    if(_S107)
    {
        return;
    }
    var p_7 : u32 = b_17.begin_0 + tid_16;
    var s_6 : f32;
    if(p_7 < (b_17.end_0))
    {
        var i_5 : u32 = lists_0[p_7];
        s_6 = norm2_0(va_0(i_5, u32(0)), vl_0(i_5, u32(0)));
    }
    else
    {
        s_6 = 0.0f;
    }
    write_partial_0(tid_16, b_17.row_4, group_12.x, u32(3), vec4<f32>(s_6, 0.0f, 0.0f, 0.0f));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_node_rn(@builtin(workgroup_id) g_12 : vec3<u32>, @builtin(local_invocation_id) t_10 : vec3<u32>)
{
    node_rn_0(g_12, t_10.x, false);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_bond_rb(@builtin(workgroup_id) g_13 : vec3<u32>, @builtin(local_invocation_id) t_11 : vec3<u32>)
{
    bond_r_0(g_13, t_11.x, true);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_node_rnb(@builtin(workgroup_id) g_14 : vec3<u32>, @builtin(local_invocation_id) t_12 : vec3<u32>)
{
    node_rn_0(g_14, t_12.x, true);
    return;
}

fn iteration_of_0( row_13 : u32) -> u32
{
    return (bitcast<u32>((wide_0[st_0(row_13, u32(2))].z)));
}

fn set_flags_0( row_14 : u32,  f_4 : u32)
{
    var _S108 : u32 = st_0(row_14, u32(2));
    var c_4 : vec4<f32> = wide_0[_S108];
    c_4[i32(3)] = (bitcast<f32>((f_4)));
    wide_0[_S108] = c_4;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_test(@builtin(workgroup_id) group_13 : vec3<u32>, @builtin(local_invocation_id) thread_8 : vec3<u32>)
{
    var tid_17 : u32 = thread_8.x;
    var row_15 : u32 = group_13.x;
    var f_5 : u32 = group_uniform_0(flags_of_0(row_15), tid_17);
    if(((f_5 & (u32(1)))) == u32(0))
    {
        return;
    }
    var r_6 : vec4<u32> = row_range_0(tid_17, row_15);
    var z_4 : vec4<f32> = row_sum_0(tid_17, row_15, blocks_0(r_6.w - r_6.z), u32(2));
    var rn_0 : vec4<f32> = row_sum_0(tid_17, row_15, blocks_0(r_6.y - r_6.x), u32(3));
    if(tid_17 != u32(0))
    {
        return;
    }
    var _S109 : u32 = st_0(row_15, u32(0));
    var s0_1 : vec4<f32> = wide_0[_S109];
    var delta2_1 : f32 = wide_0[_S109].x;
    var _S110 : f32 = z_4.x;
    var _S111 : bool = _S110 <= delta2_1;
    var hold_0 : bool;
    if(_S111)
    {
        hold_0 = !((rn_0.x) <= delta2_1);
    }
    else
    {
        hold_0 = false;
    }
    var it_0 : u32 = iteration_of_0(row_15);
    var f_6 : u32 = (f_5 & (u32(4294967199)));
    var f_7 : u32;
    if(hold_0)
    {
        f_7 = (f_6 | (u32(32)));
    }
    else
    {
        f_7 = f_6;
    }
    var _S112 : bool;
    if(it_0 > u32(0))
    {
        _S112 = true;
    }
    else
    {
        _S112 = (params_0.warm_0) != u32(0);
    }
    if(_S112)
    {
        _S112 = _S111;
    }
    else
    {
        _S112 = false;
    }
    if(_S112)
    {
        hold_0 = !hold_0;
    }
    else
    {
        hold_0 = false;
    }
    if(hold_0)
    {
        f_7 = (f_7 | (u32(64)));
    }
    s0_1[i32(1)] = _S110;
    wide_0[_S109] = s0_1;
    var _S113 : u32 = st_0(row_15, u32(1));
    wide_0[_S113] = wide_0[_S113];
    set_flags_0(row_15, f_7);
    var _S114 : u32 = st_0(row_15, u32(2));
    var cnt_0 : vec4<f32> = wide_0[_S114];
    var _S115 : u32 = st_0(row_15, u32(3));
    wide_0[_S115] = vec4<f32>(wide_0[_S115].xyz, wide_0[_S115].w);
    wide_0[pt_0(row_15, u32(0), u32(5))] = vec4<f32>(z_4.y, 0.0f, 0.0f, 0.0f);
    wide_0[_S114] = cnt_0;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_rebuild(@builtin(workgroup_id) group_14 : vec3<u32>, @builtin(local_invocation_id) thread_9 : vec3<u32>)
{
    var tid_18 : u32 = thread_9.x;
    var b_18 : Block_0 = node_block_0(group_14, tid_18);
    var _S116 : bool;
    if((b_18.begin_0) >= (b_18.end_0))
    {
        _S116 = true;
    }
    else
    {
        _S116 = !runs_0(b_18.flags_0, true);
    }
    if(_S116)
    {
        return;
    }
    var p_8 : u32 = b_18.begin_0 + tid_18;
    if(p_8 < (b_18.end_0))
    {
        var i_6 : u32 = lists_0[p_8];
        var a_6 : vec3<f32>;
        var l_2 : vec3<f32>;
        gather_0(i_6, false, true, &(a_6), &(l_2));
        set_v_0(i_6, u32(0), va_0(i_6, u32(14)) - a_6, vl_0(i_6, u32(14)) - l_2);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_decide(@builtin(workgroup_id) group_15 : vec3<u32>, @builtin(local_invocation_id) thread_10 : vec3<u32>)
{
    var tid_19 : u32 = thread_10.x;
    var row_16 : u32 = group_15.x;
    var f_8 : u32 = group_uniform_0(flags_of_0(row_16), tid_19);
    if(((f_8 & (u32(1)))) == u32(0))
    {
        return;
    }
    var r_7 : vec4<u32> = row_range_0(tid_19, row_16);
    var _S117 : vec4<f32> = vec4<f32>(0.0f);
    var _S118 : bool = ((f_8 & (u32(64)))) != u32(0);
    var z_5 : vec4<f32>;
    var rn_1 : vec4<f32>;
    if(_S118)
    {
        var _S119 : vec4<f32> = row_sum_0(tid_19, row_16, blocks_0(r_7.w - r_7.z), u32(2));
        var _S120 : vec4<f32> = row_sum_0(tid_19, row_16, blocks_0(r_7.y - r_7.x), u32(3));
        z_5 = _S119;
        rn_1 = _S120;
    }
    else
    {
        z_5 = _S117;
        rn_1 = _S117;
    }
    if(tid_19 != u32(0))
    {
        return;
    }
    var _S121 : u32 = st_0(row_16, u32(0));
    var s0_2 : vec4<f32> = wide_0[_S121];
    var s1_1 : vec4<f32> = wide_0[st_0(row_16, u32(1))];
    var cnt_1 : vec4<f32> = wide_0[st_0(row_16, u32(2))];
    var _S122 : u32 = st_0(row_16, u32(3));
    var cen_0 : vec4<f32> = wide_0[_S122];
    var delta2_2 : f32 = wide_0[_S121].x;
    var _S123 : f32 = wide_0[_S121].y;
    var _S124 : bool = ((f_8 & (u32(32)))) != u32(0);
    var hold_1 : bool;
    var num_0 : f32;
    if(_S118)
    {
        var num_1 : f32 = z_5.x;
        if(num_1 <= delta2_2)
        {
            hold_1 = !((rn_1.x) <= delta2_2);
        }
        else
        {
            hold_1 = false;
        }
        s0_2[i32(2)] = 0.0f;
        num_0 = num_1;
    }
    else
    {
        num_0 = _S123;
        hold_1 = _S124;
    }
    var f_9 : u32 = (f_8 & (u32(4294967199)));
    var it_1 : u32 = (bitcast<u32>((cnt_1.z)));
    var best_it_0 : u32 = (bitcast<u32>((cen_0.w)));
    var stop_0 : bool;
    var best_it_1 : u32;
    if(num_0 < (s0_2.w * 0.99000000953674316f))
    {
        s0_2[i32(3)] = num_0;
        stop_0 = false;
        best_it_1 = it_1;
    }
    else
    {
        stop_0 = (it_1 - best_it_0) >= u32(512);
        best_it_1 = best_it_0;
    }
    var _S125 : bool;
    if(!stop_0)
    {
        _S125 = num_0 <= delta2_2;
    }
    else
    {
        _S125 = false;
    }
    if(_S125)
    {
        hold_1 = !hold_1;
    }
    else
    {
        hold_1 = false;
    }
    var f_10 : u32;
    if(hold_1)
    {
        var f_11 : u32 = (f_9 | (u32(2)));
        stop_0 = true;
        f_10 = f_11;
    }
    else
    {
        f_10 = f_9;
    }
    if(!stop_0)
    {
        hold_1 = (params_0.force_tolerance_0) > 0.0f;
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        hold_1 = it_1 > u32(0);
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        hold_1 = (s1_1.w) <= (params_0.force_tolerance_0 * (cnt_1.x + cnt_1.y));
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        if((s1_1.w) <= (params_0.force_tolerance_0 * sqrt(wide_0[pt_0(row_16, u32(0), u32(5))].x)))
        {
            var f_12 : u32 = (f_10 | (u32(2)));
            stop_0 = true;
            f_10 = f_12;
        }
    }
    if(stop_0)
    {
        hold_1 = true;
    }
    else
    {
        hold_1 = it_1 >= (params_0.max_iterations_0);
    }
    if(hold_1)
    {
        f_10 = (f_10 & (u32(4294967294)));
    }
    s0_2[i32(1)] = num_0;
    wide_0[_S121] = s0_2;
    wide_0[_S122] = vec4<f32>(cen_0.xyz, (bitcast<f32>((best_it_1))));
    set_flags_0(row_16, f_10);
    return;
}

fn sym_mul_0( s0_3 : vec4<f32>,  s1_2 : vec4<f32>,  v_10 : vec3<f32>) -> vec3<f32>
{
    var _S126 : f32 = v_10.x;
    var _S127 : f32 = s0_3.w;
    var _S128 : f32 = v_10.y;
    var _S129 : f32 = s1_2.x;
    var _S130 : f32 = v_10.z;
    var _S131 : f32 = s1_2.y;
    return vec3<f32>(s0_3.x * _S126 + _S127 * _S128 + _S129 * _S130, _S127 * _S126 + s0_3.y * _S128 + _S131 * _S130, _S129 * _S126 + _S131 * _S128 + s0_3.z * _S130);
}

fn inverse_apply_0( n_12 : u32,  a_7 : vec3<f32>,  l_3 : vec3<f32>,  oa_0 : ptr<function, vec3<f32>>,  ol_0 : ptr<function, vec3<f32>>)
{
    var _S132 : u32 = ns_0(n_12, u32(21));
    var s1_3 : vec4<f32> = scratch_0[_S132 + u32(1)];
    var kc_0 : vec3<f32> = scratch_0[_S132 + u32(2)].xyz;
    var _S133 : vec3<f32> = sym_mul_0(scratch_0[_S132], scratch_0[_S132 + u32(1)], a_7 + cross(kc_0, l_3));
    (*oa_0) = _S133;
    (*ol_0) = l_3 * vec3<f32>(s1_3.z) - cross(kc_0, _S133);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_pre1(@builtin(workgroup_id) group_16 : vec3<u32>, @builtin(local_invocation_id) thread_11 : vec3<u32>)
{
    var tid_20 : u32 = thread_11.x;
    var b_19 : Block_0 = node_block_0(group_16, tid_20);
    var _S134 : bool;
    if((b_19.begin_0) >= (b_19.end_0))
    {
        _S134 = true;
    }
    else
    {
        _S134 = !runs_0(b_19.flags_0, false);
    }
    if(_S134)
    {
        return;
    }
    var p_9 : u32 = b_19.begin_0 + tid_20;
    if(p_9 >= (b_19.end_0))
    {
        return;
    }
    var i_7 : u32 = lists_0[p_9];
    if((iteration_of_0(b_19.row_4)) == u32(0))
    {
        set_v_0(i_7, u32(8), va_0(i_7, u32(0)), vl_0(i_7, u32(0)));
        return;
    }
    var a_8 : vec3<f32>;
    var l_4 : vec3<f32>;
    inverse_apply_0(i_7, va_0(i_7, u32(0)), vl_0(i_7, u32(0)), &(a_8), &(l_4));
    set_v_0(i_7, u32(10), a_8, l_4);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_bond_loc(@builtin(workgroup_id) group_17 : vec3<u32>, @builtin(local_invocation_id) thread_12 : vec3<u32>)
{
    var tid_21 : u32 = thread_12.x;
    var b_20 : Block_0 = bond_block_0(group_17, tid_21);
    var _S135 : bool;
    if((b_20.begin_0) >= (b_20.end_0))
    {
        _S135 = true;
    }
    else
    {
        _S135 = !runs_0(b_20.flags_0, false);
    }
    if(_S135)
    {
        _S135 = true;
    }
    else
    {
        _S135 = (iteration_of_0(b_20.row_4)) == u32(0);
    }
    if(_S135)
    {
        return;
    }
    var p_10 : u32 = b_20.begin_0 + tid_21;
    if(p_10 >= (b_20.end_0))
    {
        return;
    }
    var j_5 : u32 = lists_0[p_10];
    var _S136 : BlastBond_std430_0 = bonds_0[j_5];
    var _S137 : vec3<f32> = vec3<f32>(0.0f);
    var ta_4 : vec3<f32> = _S137;
    var tl_4 : vec3<f32> = _S137;
    if((_S136.health_0) > 0.0f)
    {
        bond_term_1(&(_S136), u32(10), &(ta_4), &(tl_4));
    }
    var _S138 : u32 = bs_0(j_5);
    scratch_0[_S138] = vec4<f32>(ta_4, 0.0f);
    scratch_0[_S138 + u32(1)] = vec4<f32>(tl_4, 0.0f);
    return;
}

fn block_apply_0( n_13 : u32,  a_9 : vec3<f32>,  l_5 : vec3<f32>,  oa_1 : ptr<function, vec3<f32>>,  ol_1 : ptr<function, vec3<f32>>)
{
    var _S139 : u32 = ns_0(n_13, u32(18));
    var b1_0 : vec4<f32> = scratch_0[_S139 + u32(1)];
    var k_8 : vec3<f32> = scratch_0[_S139 + u32(2)].xyz;
    (*oa_1) = sym_mul_0(scratch_0[_S139], scratch_0[_S139 + u32(1)], a_9) - cross(k_8, l_5);
    (*ol_1) = cross(k_8, a_9) + vec3<f32>(b1_0.z) * l_5;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_pre2(@builtin(workgroup_id) group_18 : vec3<u32>, @builtin(local_invocation_id) thread_13 : vec3<u32>)
{
    var tid_22 : u32 = thread_13.x;
    var b_21 : Block_0 = node_block_0(group_18, tid_22);
    var _S140 : bool;
    if((b_21.begin_0) >= (b_21.end_0))
    {
        _S140 = true;
    }
    else
    {
        _S140 = !runs_0(b_21.flags_0, false);
    }
    if(_S140)
    {
        _S140 = true;
    }
    else
    {
        _S140 = (iteration_of_0(b_21.row_4)) == u32(0);
    }
    if(_S140)
    {
        return;
    }
    var p_11 : u32 = b_21.begin_0 + tid_22;
    if(p_11 >= (b_21.end_0))
    {
        return;
    }
    var i_8 : u32 = lists_0[p_11];
    var oa_2 : vec3<f32>;
    var ol_2 : vec3<f32>;
    gather_0(i_8, true, false, &(oa_2), &(ol_2));
    var sa_0 : vec3<f32>;
    var sl_0 : vec3<f32>;
    block_apply_0(i_8, va_0(i_8, u32(10)), vl_0(i_8, u32(10)), &(sa_0), &(sl_0));
    var ma_0 : vec3<f32>;
    var ml_0 : vec3<f32>;
    inverse_apply_0(i_8, oa_2 - sa_0, ol_2 - sl_0, &(ma_0), &(ml_0));
    var _S141 : vec3<f32> = vec3<f32>(1.68946564197540283f);
    var _S142 : vec3<f32> = vec3<f32>(1.5220409631729126f);
    set_v_0(i_8, u32(8), _S141 * va_0(i_8, u32(10)) - _S142 * ma_0, _S141 * vl_0(i_8, u32(10)) - _S142 * ml_0);
    return;
}

var<workgroup> g_wide_max_0 : array<f32, i32(1024)>;

var<workgroup> g_wide_max_out_0 : f32;

fn wide_group_max_0( tid_23 : u32,  v_11 : f32) -> f32
{
    g_wide_max_0[tid_23] = v_11;
    workgroupBarrier();
    var k_9 : u32 = u32(512);
    loop
    {
        if(k_9 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_23 < k_9)
        {
            g_wide_max_0[tid_23] = max(g_wide_max_0[tid_23], g_wide_max_0[tid_23 + k_9]);
        }
        workgroupBarrier();
        k_9 = (k_9 >> (u32(1)));
    }
    if(tid_23 == u32(0))
    {
        g_wide_max_out_0 = g_wide_max_0[i32(0)];
    }
    var _S143 : f32 = (workgroupUniformLoad(&((g_wide_max_out_0))));
    workgroupBarrier();
    return _S143;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_max(@builtin(workgroup_id) group_19 : vec3<u32>, @builtin(local_invocation_id) thread_14 : vec3<u32>)
{
    var tid_24 : u32 = thread_14.x;
    var b_22 : Block_0 = node_block_0(group_19, tid_24);
    var _S144 : bool;
    if((b_22.begin_0) >= (b_22.end_0))
    {
        _S144 = true;
    }
    else
    {
        _S144 = !runs_0(b_22.flags_0, false);
    }
    if(_S144)
    {
        return;
    }
    var p_12 : u32 = b_22.begin_0 + tid_24;
    var m_1 : f32;
    if(p_12 < (b_22.end_0))
    {
        var i_9 : u32 = lists_0[p_12];
        var a_10 : vec3<f32> = abs(va_0(i_9, u32(8)));
        var l_6 : vec3<f32> = abs(vl_0(i_9, u32(8)));
        m_1 = max(max(max(a_10.x, a_10.y), max(a_10.z, l_6.x)), max(l_6.y, l_6.z));
    }
    else
    {
        m_1 = 0.0f;
    }
    var mm_0 : f32 = wide_group_max_0(tid_24, m_1);
    if(tid_24 == u32(0))
    {
        wide_0[pt_0(b_22.row_4, group_19.x, u32(4))] = vec4<f32>(mm_0, 0.0f, 0.0f, 0.0f);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_mag(@builtin(workgroup_id) group_20 : vec3<u32>, @builtin(local_invocation_id) thread_15 : vec3<u32>)
{
    var tid_25 : u32 = thread_15.x;
    var row_17 : u32 = group_20.x;
    var f_13 : u32 = group_uniform_0(flags_of_0(row_17), tid_25);
    if(((f_13 & (u32(1)))) == u32(0))
    {
        return;
    }
    var r_8 : vec4<u32> = row_range_0(tid_25, row_17);
    var _S145 : u32 = blocks_0(r_8.y - r_8.x);
    var m_2 : f32 = 0.0f;
    var blk_0 : u32 = tid_25;
    loop
    {
        if(blk_0 < _S145)
        {
        }
        else
        {
            break;
        }
        var _S146 : f32 = max(m_2, wide_0[pt_0(row_17, blk_0, u32(4))].x);
        var blk_1 : u32 = blk_0 + u32(1024);
        m_2 = _S146;
        blk_0 = blk_1;
    }
    var mag_0 : f32 = wide_group_max_0(tid_25, m_2);
    if(tid_25 != u32(0))
    {
        return;
    }
    var _S147 : bool;
    if(!(mag_0 > 0.0f))
    {
        _S147 = true;
    }
    else
    {
        _S147 = !isfinite_0(mag_0);
    }
    if(_S147)
    {
        set_flags_0(row_17, (((f_13 & (u32(4294967294)))) | (u32(4))));
        return;
    }
    var _S148 : u32 = st_0(row_17, u32(1));
    var s1_4 : vec4<f32> = wide_0[_S148];
    s1_4[i32(1)] = 1.0f / mag_0;
    wide_0[_S148] = s1_4;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_scale(@builtin(workgroup_id) group_21 : vec3<u32>, @builtin(local_invocation_id) thread_16 : vec3<u32>)
{
    var tid_26 : u32 = thread_16.x;
    var b_23 : Block_0 = node_block_0(group_21, tid_26);
    var _S149 : bool;
    if((b_23.begin_0) >= (b_23.end_0))
    {
        _S149 = true;
    }
    else
    {
        _S149 = !runs_0(b_23.flags_0, false);
    }
    if(_S149)
    {
        return;
    }
    var p_13 : u32 = b_23.begin_0 + tid_26;
    var gl_0 : f32;
    if(p_13 < (b_23.end_0))
    {
        var i_10 : u32 = lists_0[p_13];
        var _S150 : vec3<f32> = vec3<f32>(wide_0[st_0(b_23.row_4, u32(1))].y);
        var ga_0 : vec3<f32> = va_0(i_10, u32(8)) * _S150;
        var gn_0 : vec3<f32> = vl_0(i_10, u32(8)) * _S150;
        set_v_0(i_10, u32(8), ga_0, gn_0);
        gl_0 = dot(va_0(i_10, u32(0)), ga_0) + dot(vl_0(i_10, u32(0)), gn_0);
    }
    else
    {
        gl_0 = 0.0f;
    }
    write_partial_0(tid_26, b_23.row_4, group_21.x, u32(0), vec4<f32>(gl_0, 0.0f, 0.0f, 0.0f));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_gamma(@builtin(workgroup_id) group_22 : vec3<u32>, @builtin(local_invocation_id) thread_17 : vec3<u32>)
{
    var tid_27 : u32 = thread_17.x;
    var row_18 : u32 = group_22.x;
    var f_14 : u32 = group_uniform_0(flags_of_0(row_18), tid_27);
    if(((f_14 & (u32(1)))) == u32(0))
    {
        return;
    }
    var r_9 : vec4<u32> = row_range_0(tid_27, row_18);
    var _S151 : vec4<f32> = row_sum_0(tid_27, row_18, blocks_0(r_9.y - r_9.x), u32(0));
    var gamma_0 : f32 = _S151.x;
    if(tid_27 != u32(0))
    {
        return;
    }
    var _S152 : bool;
    if(!(gamma_0 > 0.0f))
    {
        _S152 = true;
    }
    else
    {
        _S152 = !isfinite_0(gamma_0);
    }
    if(_S152)
    {
        set_flags_0(row_18, (((f_14 & (u32(4294967294)))) | (u32(4))));
        return;
    }
    var s0_4 : vec4<f32> = wide_0[st_0(row_18, u32(0))];
    if((iteration_of_0(row_18)) > u32(1))
    {
        _S152 = (s0_4.z) > 0.0f;
    }
    else
    {
        _S152 = false;
    }
    var beta_0 : f32;
    if(_S152)
    {
        beta_0 = gamma_0 / s0_4.z;
    }
    else
    {
        beta_0 = 0.0f;
    }
    var _S153 : u32 = st_0(row_18, u32(1));
    var s1_5 : vec4<f32> = wide_0[_S153];
    s1_5[i32(0)] = gamma_0;
    s1_5[i32(2)] = beta_0;
    wide_0[_S153] = s1_5;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_dir(@builtin(workgroup_id) group_23 : vec3<u32>, @builtin(local_invocation_id) thread_18 : vec3<u32>)
{
    var tid_28 : u32 = thread_18.x;
    var b_24 : Block_0 = node_block_0(group_23, tid_28);
    var _S154 : bool;
    if((b_24.begin_0) >= (b_24.end_0))
    {
        _S154 = true;
    }
    else
    {
        _S154 = !runs_0(b_24.flags_0, false);
    }
    if(_S154)
    {
        return;
    }
    var p_14 : u32 = b_24.begin_0 + tid_28;
    if(p_14 >= (b_24.end_0))
    {
        return;
    }
    var i_11 : u32 = lists_0[p_14];
    var beta_1 : f32 = wide_0[st_0(b_24.row_4, u32(1))].z;
    if(beta_1 == 0.0f)
    {
        set_v_0(i_11, u32(2), va_0(i_11, u32(8)), vl_0(i_11, u32(8)));
    }
    else
    {
        var _S155 : vec3<f32> = vec3<f32>(beta_1);
        set_v_0(i_11, u32(2), va_0(i_11, u32(8)) + va_0(i_11, u32(2)) * _S155, vl_0(i_11, u32(8)) + vl_0(i_11, u32(2)) * _S155);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_bond_p(@builtin(workgroup_id) group_24 : vec3<u32>, @builtin(local_invocation_id) thread_19 : vec3<u32>)
{
    var tid_29 : u32 = thread_19.x;
    var b_25 : Block_0 = bond_block_0(group_24, tid_29);
    var _S156 : bool;
    if((b_25.begin_0) >= (b_25.end_0))
    {
        _S156 = true;
    }
    else
    {
        _S156 = !runs_0(b_25.flags_0, false);
    }
    if(_S156)
    {
        return;
    }
    var p_15 : u32 = b_25.begin_0 + tid_29;
    var z_6 : f32;
    if(p_15 < (b_25.end_0))
    {
        var j_6 : u32 = lists_0[p_15];
        var _S157 : BlastBond_std430_0 = bonds_0[j_6];
        var _S158 : vec3<f32> = vec3<f32>(0.0f);
        var ta_5 : vec3<f32> = _S158;
        var tl_5 : vec3<f32> = _S158;
        if((_S157.health_0) > 0.0f)
        {
            bond_term_1(&(_S157), u32(2), &(ta_5), &(tl_5));
        }
        var _S159 : f32 = norm2_0(ta_5, tl_5);
        var _S160 : u32 = bs_0(j_6);
        scratch_0[_S160] = vec4<f32>(ta_5, 0.0f);
        scratch_0[_S160 + u32(1)] = vec4<f32>(tl_5, 0.0f);
        z_6 = _S159;
    }
    else
    {
        z_6 = 0.0f;
    }
    write_partial_0(tid_29, b_25.row_4, group_24.x, u32(1), vec4<f32>(z_6, 0.0f, 0.0f, 0.0f));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_node_q(@builtin(workgroup_id) group_25 : vec3<u32>, @builtin(local_invocation_id) thread_20 : vec3<u32>)
{
    var tid_30 : u32 = thread_20.x;
    var b_26 : Block_0 = node_block_0(group_25, tid_30);
    var _S161 : bool;
    if((b_26.begin_0) >= (b_26.end_0))
    {
        _S161 = true;
    }
    else
    {
        _S161 = !runs_0(b_26.flags_0, false);
    }
    if(_S161)
    {
        return;
    }
    var p_16 : u32 = b_26.begin_0 + tid_30;
    if(p_16 >= (b_26.end_0))
    {
        return;
    }
    var i_12 : u32 = lists_0[p_16];
    var a_11 : vec3<f32>;
    var l_7 : vec3<f32>;
    gather_0(i_12, true, false, &(a_11), &(l_7));
    set_v_0(i_12, u32(4), a_11, l_7);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_den(@builtin(workgroup_id) group_26 : vec3<u32>, @builtin(local_invocation_id) thread_21 : vec3<u32>)
{
    var tid_31 : u32 = thread_21.x;
    var row_19 : u32 = group_26.x;
    var f_15 : u32 = group_uniform_0(flags_of_0(row_19), tid_31);
    if(((f_15 & (u32(1)))) == u32(0))
    {
        return;
    }
    var r_10 : vec4<u32> = row_range_0(tid_31, row_19);
    var _S162 : vec4<f32> = row_sum_0(tid_31, row_19, blocks_0(r_10.w - r_10.z), u32(1));
    var den_0 : f32 = _S162.x;
    if(tid_31 != u32(0))
    {
        return;
    }
    var _S163 : u32 = st_0(row_19, u32(0));
    var s0_5 : vec4<f32> = wide_0[_S163];
    var _S164 : u32 = st_0(row_19, u32(1));
    var s1_6 : vec4<f32> = wide_0[_S164];
    var _S165 : u32 = st_0(row_19, u32(2));
    var cnt_2 : vec4<f32> = wide_0[_S165];
    var gamma_1 : f32 = wide_0[_S164].x;
    var _S166 : bool;
    if((params_0.force_tolerance_0) > 0.0f)
    {
        if(den_0 > 0.0f)
        {
            _S166 = gamma_1 > 0.0f;
        }
        else
        {
            _S166 = false;
        }
        var _S167 : f32;
        if(_S166)
        {
            _S167 = gamma_1 / sqrt(den_0);
        }
        else
        {
            _S167 = float_bits_0(u32(2139095040));
        }
        s1_6[i32(3)] = _S167;
        if(isfinite_0(s1_6.w))
        {
            cnt_2[i32(1)] = cnt_2[i32(1)] + s1_6.w;
        }
    }
    s0_5[i32(2)] = gamma_1;
    cnt_2[i32(2)] = (bitcast<f32>(((bitcast<u32>((cnt_2.z))) + u32(1))));
    if(!(den_0 > 0.0f))
    {
        _S166 = true;
    }
    else
    {
        _S166 = !isfinite_0(den_0);
    }
    var f_16 : u32;
    if(_S166)
    {
        var f_17 : u32 = (f_15 & (u32(4294967294)));
        s1_6[i32(1)] = 0.0f;
        f_16 = f_17;
    }
    else
    {
        s1_6[i32(1)] = gamma_1 / den_0;
        f_16 = f_15;
    }
    wide_0[_S163] = s0_5;
    wide_0[_S164] = s1_6;
    cnt_2[i32(3)] = (bitcast<f32>((f_16)));
    wide_0[_S165] = cnt_2;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_update(@builtin(workgroup_id) group_27 : vec3<u32>, @builtin(local_invocation_id) thread_22 : vec3<u32>)
{
    var tid_32 : u32 = thread_22.x;
    var b_27 : Block_0 = node_block_0(group_27, tid_32);
    var _S168 : bool;
    if((b_27.begin_0) >= (b_27.end_0))
    {
        _S168 = true;
    }
    else
    {
        _S168 = !runs_0(b_27.flags_0, false);
    }
    if(_S168)
    {
        return;
    }
    var p_17 : u32 = b_27.begin_0 + tid_32;
    if(p_17 >= (b_27.end_0))
    {
        return;
    }
    var i_13 : u32 = lists_0[p_17];
    var _S169 : vec3<f32> = vec3<f32>(wide_0[st_0(b_27.row_4, u32(1))].y);
    set_v_0(i_13, u32(6), va_0(i_13, u32(6)) + va_0(i_13, u32(2)) * _S169, vl_0(i_13, u32(6)) + vl_0(i_13, u32(2)) * _S169);
    set_v_0(i_13, u32(0), va_0(i_13, u32(0)) - va_0(i_13, u32(4)) * _S169, vl_0(i_13, u32(0)) - vl_0(i_13, u32(4)) * _S169);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_final(@builtin(workgroup_id) group_28 : vec3<u32>, @builtin(local_invocation_id) thread_23 : vec3<u32>)
{
    var tid_33 : u32 = thread_23.x;
    var b_28 : Block_0 = bond_block_0(group_28, tid_33);
    var _S170 : bool;
    if((b_28.begin_0) >= (b_28.end_0))
    {
        _S170 = true;
    }
    else
    {
        _S170 = (((b_28.flags_0) & (u32(128)))) != u32(0);
    }
    if(_S170)
    {
        return;
    }
    var p_18 : u32 = b_28.begin_0 + tid_33;
    if(p_18 >= (b_28.end_0))
    {
        return;
    }
    var j_7 : u32 = lists_0[p_18];
    var _S171 : BlastBond_std430_0 = bonds_0[j_7];
    if(!((_S171.health_0) > 0.0f))
    {
        _S170 = true;
    }
    else
    {
        _S170 = (((b_28.flags_0) & (u32(16)))) != u32(0);
    }
    if(_S170)
    {
        var _S172 : u32 = j_7 * u32(2);
        var _S173 : vec4<f32> = vec4<f32>(0.0f);
        x_0[_S172] = _S173;
        x_0[_S172 + u32(1)] = _S173;
        return;
    }
    var ta_6 : vec3<f32>;
    var tl_6 : vec3<f32>;
    bond_term_1(&(_S171), u32(6), &(ta_6), &(tl_6));
    var _S174 : u32 = j_7 * u32(2);
    x_0[_S174] = vec4<f32>(x_0[_S174].xyz + ta_6, 0.0f);
    var _S175 : u32 = _S174 + u32(1);
    x_0[_S175] = vec4<f32>(x_0[_S175].xyz + tl_6, 0.0f);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn w_report(@builtin(workgroup_id) group_29 : vec3<u32>)
{
    var row_20 : u32 = group_29.x;
    var island_2 : u32 = row_island_0(row_20);
    var f_18 : u32 = flags_of_0(row_20);
    var last_1 : vec4<u32> = report_0[island_2];
    if(((f_18 & (u32(128)))) != u32(0))
    {
        report_0[island_2] = vec4<u32>(u32(0), u32(7), last_1.z, last_1.w);
        return;
    }
    var ok_1 : bool;
    if(((f_18 & (u32(2)))) != u32(0))
    {
        ok_1 = ((f_18 & (u32(4)))) == u32(0);
    }
    else
    {
        ok_1 = false;
    }
    var it_2 : u32 = iteration_of_0(row_20);
    var _S176 : u32;
    if(ok_1)
    {
        _S176 = u32(1);
    }
    else
    {
        _S176 = u32(0);
    }
    if(ok_1)
    {
        ok_1 = (params_0.warm_0) != u32(0);
    }
    else
    {
        ok_1 = false;
    }
    if(ok_1)
    {
        ok_1 = it_2 == u32(0);
    }
    else
    {
        ok_1 = false;
    }
    var _S177 : u32;
    if(ok_1)
    {
        _S177 = u32(4);
    }
    else
    {
        _S177 = u32(0);
    }
    report_0[island_2] = vec4<u32>(it_2, (_S176 | (_S177)), (bitcast<u32>((wide_0[st_0(row_20, u32(0))].y))), (bitcast<u32>((wide_0[st_0(row_20, u32(0))].x))));
    return;
}

