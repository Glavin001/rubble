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
    @align(4) pad_0 : u32,
    @align(16) gravity_0 : vec4<f32>,
    @align(16) w_alive_0 : u32,
    @align(4) a_dirty_0 : u32,
    @align(8) a_counters_0 : u32,
    @align(4) pad1_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : BlastParams_std140_0;
@binding(3) @group(0) var<storage, read_write> topo_0 : array<atomic<u32>>;

@binding(7) @group(0) var<storage, read_write> args_0 : array<u32>;

struct BlastBond_std430_0
{
    @align(16) n0_0 : u32,
    @align(4) n1_0 : u32,
    @align(8) s_0 : f32,
    @align(4) health_0 : f32,
    @align(16) o0_0 : vec4<f32>,
    @align(16) o1_0 : vec4<f32>,
};

@binding(2) @group(0) var<storage, read> bonds_0 : array<BlastBond_std430_0>;

struct BlastNode_std430_0
{
    @align(4) w_ang_0 : f32,
    @align(4) w_lin_0 : f32,
    @align(4) ref_begin_0 : u32,
    @align(4) ref_end_0 : u32,
};

@binding(1) @group(0) var<storage, read> nodes_0 : array<BlastNode_std430_0>;

@binding(4) @group(0) var<storage, read_write> islands_0 : array<vec4<u32>>;

@binding(6) @group(0) var<storage, read_write> report_0 : array<vec4<u32>>;

@binding(5) @group(0) var<storage, read_write> lists_0 : array<u32>;

fn topo_total_0( node_count_1 : u32) -> u32
{
    return u32(10) * node_count_1 + u32(3) * ((node_count_1 + u32(1023)) / u32(1024));
}

fn topo_dirty_0( node_count_2 : u32) -> u32
{
    return topo_total_0(node_count_2) + u32(4);
}

fn set_args_0( at_0 : u32,  x_0 : u32)
{
    args_0[at_0] = x_0;
    args_0[at_0 + u32(1)] = u32(1);
    args_0[at_0 + u32(2)] = u32(1);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn bt_gate(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : u32 = atomicExchange(&(topo_0[topo_dirty_0(params_0.node_count_0)]), u32(0));
    var run_0 : bool = _S1 != u32(0);
    var _S2 : u32;
    if(run_0)
    {
        _S2 = (params_0.node_count_0 + u32(63)) / u32(64);
    }
    else
    {
        _S2 = u32(0);
    }
    set_args_0(u32(4), _S2);
    if(run_0)
    {
        _S2 = (params_0.bond_count_0 + u32(63)) / u32(64);
    }
    else
    {
        _S2 = u32(0);
    }
    set_args_0(u32(7), _S2);
    if(run_0)
    {
        _S2 = (params_0.node_count_0 + u32(1024) - u32(1)) / u32(1024);
    }
    else
    {
        _S2 = u32(0);
    }
    set_args_0(u32(10), _S2);
    var _S3 : i32;
    if(run_0)
    {
        _S3 = i32(1);
    }
    else
    {
        _S3 = i32(0);
    }
    set_args_0(u32(13), u32(_S3));
    set_args_0(u32(16), u32(0));
    return;
}

fn t_parent_0() -> u32
{
    return u32(0);
}

fn t_nodes_0() -> u32
{
    return u32(2) * params_0.node_count_0;
}

fn t_bonds_0() -> u32
{
    return u32(3) * params_0.node_count_0;
}

fn t_node_cursor_0() -> u32
{
    return u32(4) * params_0.node_count_0;
}

fn t_bond_cursor_0() -> u32
{
    return u32(5) * params_0.node_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_reset(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var i_0 : u32 = id_1.x;
    if(i_0 >= (params_0.node_count_0))
    {
        return;
    }
    atomicStore(&(topo_0[t_parent_0() + i_0]), i_0);
    atomicStore(&(topo_0[t_nodes_0() + i_0]), u32(0));
    atomicStore(&(topo_0[t_bonds_0() + i_0]), u32(0));
    atomicStore(&(topo_0[t_node_cursor_0() + i_0]), u32(0));
    atomicStore(&(topo_0[t_bond_cursor_0() + i_0]), u32(0));
    return;
}

fn is_dynamic_0( i_1 : u32) -> bool
{
    return (nodes_0[i_1].w_lin_0) > 0.0f;
}

fn find_0( i_2 : u32) -> u32
{
    var _S4 : u32 = t_parent_0();
    var _S5 : u32 = atomicLoad(&(topo_0[_S4 + i_2]));
    var p_0 : u32 = _S5;
    var _S6 : u32 = i_2;
    loop
    {
        if(p_0 != _S6)
        {
        }
        else
        {
            break;
        }
        var _S7 : u32 = _S4 + p_0;
        var g_0 : u32 = atomicLoad(&(topo_0[_S7]));
        if(g_0 != p_0)
        {
            var _S8 : u32 = atomicCompareExchangeWeak(&(topo_0[_S4 + _S6]), p_0, g_0).old_value;
        }
        var _S9 : u32 = atomicLoad(&(topo_0[_S7]));
        _S6 = p_0;
        p_0 = _S9;
    }
    return _S6;
}

fn t_total_0() -> u32
{
    return topo_total_0(params_0.node_count_0);
}

@compute
@workgroup_size(64, 1, 1)
fn bt_union(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var j_0 : u32 = id_2.x;
    if(j_0 >= (params_0.bond_count_0))
    {
        return;
    }
    var b_0 : BlastBond_std430_0 = bonds_0[j_0];
    var _S10 : bool;
    if(!((b_0.health_0) > 0.0f))
    {
        _S10 = true;
    }
    else
    {
        _S10 = !is_dynamic_0(b_0.n0_0);
    }
    if(_S10)
    {
        _S10 = true;
    }
    else
    {
        _S10 = !is_dynamic_0(b_0.n1_0);
    }
    if(_S10)
    {
        return;
    }
    var a_0 : u32 = b_0.n0_0;
    var c_0 : u32 = b_0.n1_0;
    var guard_0 : u32 = u32(0);
    loop
    {
        if(guard_0 < u32(4096))
        {
        }
        else
        {
            break;
        }
        var a_1 : u32 = find_0(a_0);
        var c_1 : u32 = find_0(c_0);
        if(a_1 == c_1)
        {
            return;
        }
        var _S11 : u32 = max(a_1, c_1);
        var _S12 : u32 = min(a_1, c_1);
        var _S13 : u32 = atomicCompareExchangeWeak(&(topo_0[t_parent_0() + _S11]), _S11, _S12).old_value;
        if(_S13 == _S11)
        {
            return;
        }
        var _S14 : u32 = guard_0 + u32(1);
        a_0 = _S11;
        c_0 = _S12;
        guard_0 = _S14;
    }
    atomicStore(&(topo_0[t_total_0() + u32(3)]), u32(1));
    return;
}

fn t_label_0() -> u32
{
    return params_0.node_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_flatten(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var i_3 : u32 = id_3.x;
    if(i_3 >= (params_0.node_count_0))
    {
        return;
    }
    var _S15 : u32;
    if(is_dynamic_0(i_3))
    {
        var _S16 : u32 = find_0(i_3);
        _S15 = _S16;
    }
    else
    {
        _S15 = u32(4294967295);
    }
    atomicStore(&(topo_0[t_label_0() + i_3]), _S15);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_count_nodes(@builtin(global_invocation_id) id_4 : vec3<u32>)
{
    var i_4 : u32 = id_4.x;
    var _S17 : bool;
    if(i_4 >= (params_0.node_count_0))
    {
        _S17 = true;
    }
    else
    {
        _S17 = !is_dynamic_0(i_4);
    }
    if(_S17)
    {
        return;
    }
    var _S18 : u32 = t_nodes_0();
    var _S19 : u32 = atomicLoad(&(topo_0[t_label_0() + i_4]));
    var _S20 : u32 = atomicAdd(&(topo_0[_S18 + _S19]), u32(1));
    return;
}

fn couples_0( b_1 : ptr<function, BlastBond_std430_0>) -> bool
{
    var _S21 : bool;
    if(((*b_1).health_0) > 0.0f)
    {
        if(is_dynamic_0((*b_1).n0_0))
        {
            _S21 = true;
        }
        else
        {
            _S21 = is_dynamic_0((*b_1).n1_0);
        }
    }
    else
    {
        _S21 = false;
    }
    return _S21;
}

fn bond_root_0( b_2 : ptr<function, BlastBond_std430_0>) -> u32
{
    var _S22 : u32 = t_label_0();
    var _S23 : u32 = (*b_2).n0_0;
    var _S24 : u32;
    if(is_dynamic_0((*b_2).n0_0))
    {
        _S24 = _S23;
    }
    else
    {
        _S24 = (*b_2).n1_0;
    }
    var _S25 : u32 = atomicLoad(&(topo_0[_S22 + _S24]));
    return _S25;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_count_bonds(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var j_1 : u32 = id_5.x;
    if(j_1 >= (params_0.bond_count_0))
    {
        return;
    }
    var _S26 : BlastBond_std430_0 = bonds_0[j_1];
    var _S27 : bool = couples_0(&(_S26));
    if(_S27)
    {
        var _S28 : u32 = t_bonds_0();
        var _S29 : u32 = bond_root_0(&(_S26));
        var _S30 : u32 = atomicAdd(&(topo_0[_S28 + _S29]), u32(1));
    }
    return;
}

fn scan_input_0( e_0 : u32) -> vec3<u32>
{
    var _S31 : bool;
    if(e_0 >= (params_0.node_count_0))
    {
        _S31 = true;
    }
    else
    {
        var _S32 : u32 = atomicLoad(&(topo_0[t_label_0() + e_0]));
        _S31 = _S32 != e_0;
    }
    if(_S31)
    {
        return vec3<u32>(vec3<i32>(i32(0)));
    }
    var nb_0 : u32 = atomicLoad(&(topo_0[t_bonds_0() + e_0]));
    var _S33 : vec3<u32>;
    if(nb_0 > u32(0))
    {
        var _S34 : u32 = atomicLoad(&(topo_0[t_nodes_0() + e_0]));
        _S33 = vec3<u32>(u32(1), _S34, nb_0);
    }
    else
    {
        _S33 = vec3<u32>(vec3<i32>(i32(0)));
    }
    return _S33;
}

var<workgroup> g_scan_0 : array<vec3<u32>, i32(1024)>;

fn group_exclusive_scan_0( tid_0 : u32,  v_0 : vec3<u32>,  total_0 : ptr<function, vec3<u32>>) -> vec3<u32>
{
    g_scan_0[tid_0] = v_0;
    workgroupBarrier();
    var d_0 : u32 = u32(1);
    loop
    {
        if(d_0 < u32(1024))
        {
        }
        else
        {
            break;
        }
        var add_0 : vec3<u32>;
        if(tid_0 >= d_0)
        {
            add_0 = g_scan_0[tid_0 - d_0];
        }
        else
        {
            add_0 = vec3<u32>(vec3<i32>(i32(0)));
        }
        workgroupBarrier();
        g_scan_0[tid_0] = g_scan_0[tid_0] + add_0;
        workgroupBarrier();
        d_0 = (d_0 << (u32(1)));
    }
    (*total_0) = g_scan_0[u32(1023)];
    var inclusive_0 : vec3<u32> = g_scan_0[tid_0];
    workgroupBarrier();
    return inclusive_0 - v_0;
}

fn t_scan_0() -> u32
{
    return u32(6) * params_0.node_count_0;
}

fn t_blocks_0() -> u32
{
    return u32(10) * params_0.node_count_0;
}

@compute
@workgroup_size(1024, 1, 1)
fn bt_scan_blocks(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var tid_1 : u32 = thread_0.x;
    var _S35 : u32 = group_0.x;
    var e_1 : u32 = _S35 * u32(1024) + tid_1;
    var v_1 : vec3<u32> = scan_input_0(e_1);
    var total_1 : vec3<u32>;
    var ex_0 : vec3<u32> = group_exclusive_scan_0(tid_1, v_1, &(total_1));
    if(e_1 < (params_0.node_count_0))
    {
        var _S36 : u32 = t_scan_0() + u32(3) * e_1;
        atomicStore(&(topo_0[_S36]), ex_0.x);
        atomicStore(&(topo_0[_S36 + u32(1)]), ex_0.y);
        atomicStore(&(topo_0[_S36 + u32(2)]), ex_0.z);
    }
    if(tid_1 == u32(0))
    {
        var _S37 : u32 = t_blocks_0() + u32(3) * _S35;
        atomicStore(&(topo_0[_S37]), total_1.x);
        atomicStore(&(topo_0[_S37 + u32(1)]), total_1.y);
        atomicStore(&(topo_0[_S37 + u32(2)]), total_1.z);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn bt_scan_top(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var tid_2 : u32 = thread_1.x;
    var _S38 : bool = tid_2 < ((params_0.node_count_0 + u32(1024) - u32(1)) / u32(1024));
    var v_2 : vec3<u32>;
    if(_S38)
    {
        var _S39 : u32 = t_blocks_0() + u32(3) * tid_2;
        var _S40 : u32 = atomicLoad(&(topo_0[_S39]));
        var _S41 : u32 = atomicLoad(&(topo_0[_S39 + u32(1)]));
        var _S42 : u32 = atomicLoad(&(topo_0[_S39 + u32(2)]));
        v_2 = vec3<u32>(_S40, _S41, _S42);
    }
    else
    {
        v_2 = vec3<u32>(vec3<i32>(i32(0)));
    }
    var total_2 : vec3<u32>;
    var ex_1 : vec3<u32> = group_exclusive_scan_0(tid_2, v_2, &(total_2));
    if(_S38)
    {
        var _S43 : u32 = t_blocks_0() + u32(3) * tid_2;
        atomicStore(&(topo_0[_S43]), ex_1.x);
        atomicStore(&(topo_0[_S43 + u32(1)]), ex_1.y);
        atomicStore(&(topo_0[_S43 + u32(2)]), ex_1.z);
    }
    if(tid_2 == u32(0))
    {
        var _S44 : u32 = t_total_0();
        atomicStore(&(topo_0[_S44]), total_2.x);
        atomicStore(&(topo_0[_S44 + u32(1)]), total_2.y);
        atomicStore(&(topo_0[_S44 + u32(2)]), total_2.z);
        set_args_0(u32(0), total_2.x);
        args_0[u32(3)] = total_2.x;
        set_args_0(u32(16), total_2.x);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn bt_scan_add(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var _S45 : u32 = group_1.x;
    var e_2 : u32 = _S45 * u32(1024) + thread_2.x;
    if(e_2 >= (params_0.node_count_0))
    {
        return;
    }
    var k_0 : u32 = u32(0);
    loop
    {
        if(k_0 < u32(3))
        {
        }
        else
        {
            break;
        }
        var _S46 : u32 = atomicLoad(&(topo_0[t_blocks_0() + u32(3) * _S45 + k_0]));
        var _S47 : u32 = atomicAdd(&(topo_0[t_scan_0() + u32(3) * e_2 + k_0]), _S46);
        k_0 = k_0 + u32(1);
    }
    return;
}

fn t_island_of_0() -> u32
{
    return u32(9) * params_0.node_count_0;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_islands(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var e_3 : u32 = id_6.x;
    if(e_3 >= (params_0.node_count_0))
    {
        return;
    }
    var v_3 : vec3<u32> = scan_input_0(e_3);
    if((v_3.x) == u32(0))
    {
        return;
    }
    var _S48 : u32 = t_scan_0() + u32(3) * e_3;
    var k_1 : u32 = atomicLoad(&(topo_0[_S48]));
    var n0_1 : u32 = atomicLoad(&(topo_0[_S48 + u32(1)]));
    var _S49 : u32 = atomicLoad(&(topo_0[t_total_0() + u32(1)]));
    var _S50 : u32 = atomicLoad(&(topo_0[_S48 + u32(2)]));
    var b0_0 : u32 = _S49 + _S50;
    islands_0[k_1] = vec4<u32>(n0_1, n0_1 + v_3.y, b0_0, b0_0 + v_3.z);
    atomicStore(&(topo_0[t_island_of_0() + e_3]), k_1);
    report_0[k_1] = vec4<u32>(vec4<i32>(i32(0)));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_scatter_nodes(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var i_5 : u32 = id_7.x;
    var _S51 : bool;
    if(i_5 >= (params_0.node_count_0))
    {
        _S51 = true;
    }
    else
    {
        _S51 = !is_dynamic_0(i_5);
    }
    if(_S51)
    {
        return;
    }
    var r_0 : u32 = atomicLoad(&(topo_0[t_label_0() + i_5]));
    var _S52 : u32 = atomicLoad(&(topo_0[t_bonds_0() + r_0]));
    if(_S52 == u32(0))
    {
        return;
    }
    var k_2 : u32 = atomicLoad(&(topo_0[t_island_of_0() + r_0]));
    var _S53 : u32 = islands_0[k_2].x;
    var _S54 : u32 = atomicAdd(&(topo_0[t_node_cursor_0() + r_0]), u32(1));
    lists_0[_S53 + _S54] = i_5;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_scatter_bonds(@builtin(global_invocation_id) id_8 : vec3<u32>)
{
    var j_2 : u32 = id_8.x;
    if(j_2 >= (params_0.bond_count_0))
    {
        return;
    }
    var _S55 : BlastBond_std430_0 = bonds_0[j_2];
    var _S56 : bool = couples_0(&(_S55));
    if(!_S56)
    {
        return;
    }
    var _S57 : u32 = bond_root_0(&(_S55));
    var k_3 : u32 = atomicLoad(&(topo_0[t_island_of_0() + _S57]));
    var _S58 : u32 = islands_0[k_3].z;
    var _S59 : u32 = atomicAdd(&(topo_0[t_bond_cursor_0() + _S57]), u32(1));
    lists_0[_S58 + _S59] = j_2;
    return;
}

var<workgroup> g_group_uniform4_0 : vec4<u32>;

fn group_uniform4_0( value_0 : vec4<u32>,  thread_3 : u32) -> vec4<u32>
{
    if(thread_3 == u32(0))
    {
        g_group_uniform4_0 = value_0;
    }
    var _S60 : vec4<u32> = (workgroupUniformLoad(&((g_group_uniform4_0))));
    return _S60;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn sort_list_0( tid_3 : u32,  begin_0 : u32,  n_0 : u32)
{
    var span_0 : u32 = u32(1);
    loop
    {
        if(span_0 < n_0)
        {
        }
        else
        {
            break;
        }
        span_0 = (span_0 << (u32(1)));
    }
    var k_4 : u32 = u32(2);
    loop
    {
        if(k_4 <= span_0)
        {
        }
        else
        {
            break;
        }
        var _S61 : u32 = (k_4 >> (u32(1)));
        var j_3 : u32 = _S61;
        loop
        {
            if(j_3 > u32(0))
            {
            }
            else
            {
                break;
            }
            var t_0 : u32 = tid_3;
            loop
            {
                if(t_0 < span_0)
                {
                }
                else
                {
                    break;
                }
                var l_0 : u32;
                if(j_3 == _S61)
                {
                    l_0 = (t_0 ^ ((k_4 - u32(1))));
                }
                else
                {
                    l_0 = (t_0 ^ (j_3));
                }
                var _S62 : bool;
                if(l_0 > t_0)
                {
                    _S62 = l_0 < n_0;
                }
                else
                {
                    _S62 = false;
                }
                if(_S62)
                {
                    var _S63 : u32 = begin_0 + t_0;
                    var a_2 : u32 = lists_0[_S63];
                    var _S64 : u32 = begin_0 + l_0;
                    var b_3 : u32 = lists_0[_S64];
                    if(lists_0[_S63] > lists_0[_S64])
                    {
                        lists_0[_S63] = b_3;
                        lists_0[_S64] = a_2;
                    }
                }
                t_0 = t_0 + u32(1024);
            }
            device_group_barrier_0();
            j_3 = (j_3 >> (u32(1)));
        }
        k_4 = (k_4 << (u32(1)));
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn bt_sort(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var tid_4 : u32 = thread_4.x;
    var _S65 : u32 = islands_0[group_2.x].x;
    var _S66 : u32 = islands_0[group_2.x].z;
    var u_0 : vec4<u32> = group_uniform4_0(vec4<u32>(_S65, islands_0[group_2.x].y - _S65, _S66, islands_0[group_2.x].w - _S66), tid_4);
    sort_list_0(tid_4, u_0.x, u_0.y);
    sort_list_0(tid_4, u_0.z, u_0.w);
    return;
}

