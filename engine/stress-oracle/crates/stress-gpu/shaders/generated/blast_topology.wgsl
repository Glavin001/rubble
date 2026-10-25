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
};

@binding(0) @group(0) var<uniform> params_0 : BlastParams_std140_0;
@binding(3) @group(0) var<storage, read_write> topo_0 : array<atomic<u32>>;

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

@binding(7) @group(0) var<storage, read_write> args_0 : array<u32>;

@binding(4) @group(0) var<storage, read_write> islands_0 : array<vec4<u32>>;

@binding(6) @group(0) var<storage, read_write> report_0 : array<vec4<u32>>;

@binding(5) @group(0) var<storage, read_write> lists_0 : array<u32>;

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
fn bt_reset(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var i_0 : u32 = id_0.x;
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
    var _S1 : u32 = t_parent_0();
    var _S2 : u32 = atomicLoad(&(topo_0[_S1 + i_2]));
    var p_0 : u32 = _S2;
    var _S3 : u32 = i_2;
    loop
    {
        if(p_0 != _S3)
        {
        }
        else
        {
            break;
        }
        var _S4 : u32 = _S1 + p_0;
        var g_0 : u32 = atomicLoad(&(topo_0[_S4]));
        if(g_0 != p_0)
        {
            var _S5 : u32 = atomicCompareExchangeWeak(&(topo_0[_S1 + _S3]), p_0, g_0).old_value;
        }
        var _S6 : u32 = atomicLoad(&(topo_0[_S4]));
        _S3 = p_0;
        p_0 = _S6;
    }
    return _S3;
}

fn t_blocks_0() -> u32
{
    return u32(10) * params_0.node_count_0;
}

fn t_total_0() -> u32
{
    return t_blocks_0() + u32(3) * ((params_0.node_count_0 + u32(1024) - u32(1)) / u32(1024));
}

@compute
@workgroup_size(64, 1, 1)
fn bt_union(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var j_0 : u32 = id_1.x;
    if(j_0 >= (params_0.bond_count_0))
    {
        return;
    }
    var b_0 : BlastBond_std430_0 = bonds_0[j_0];
    var _S7 : bool;
    if(!((b_0.health_0) > 0.0f))
    {
        _S7 = true;
    }
    else
    {
        _S7 = !is_dynamic_0(b_0.n0_0);
    }
    if(_S7)
    {
        _S7 = true;
    }
    else
    {
        _S7 = !is_dynamic_0(b_0.n1_0);
    }
    if(_S7)
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
        var _S8 : u32 = max(a_1, c_1);
        var _S9 : u32 = min(a_1, c_1);
        var _S10 : u32 = atomicCompareExchangeWeak(&(topo_0[t_parent_0() + _S8]), _S8, _S9).old_value;
        if(_S10 == _S8)
        {
            return;
        }
        var _S11 : u32 = guard_0 + u32(1);
        a_0 = _S8;
        c_0 = _S9;
        guard_0 = _S11;
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
fn bt_flatten(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var i_3 : u32 = id_2.x;
    if(i_3 >= (params_0.node_count_0))
    {
        return;
    }
    var _S12 : u32;
    if(is_dynamic_0(i_3))
    {
        var _S13 : u32 = find_0(i_3);
        _S12 = _S13;
    }
    else
    {
        _S12 = u32(4294967295);
    }
    atomicStore(&(topo_0[t_label_0() + i_3]), _S12);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_count_nodes(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var i_4 : u32 = id_3.x;
    var _S14 : bool;
    if(i_4 >= (params_0.node_count_0))
    {
        _S14 = true;
    }
    else
    {
        _S14 = !is_dynamic_0(i_4);
    }
    if(_S14)
    {
        return;
    }
    var _S15 : u32 = t_nodes_0();
    var _S16 : u32 = atomicLoad(&(topo_0[t_label_0() + i_4]));
    var _S17 : u32 = atomicAdd(&(topo_0[_S15 + _S16]), u32(1));
    return;
}

fn couples_0( b_1 : ptr<function, BlastBond_std430_0>) -> bool
{
    var _S18 : bool;
    if(((*b_1).health_0) > 0.0f)
    {
        if(is_dynamic_0((*b_1).n0_0))
        {
            _S18 = true;
        }
        else
        {
            _S18 = is_dynamic_0((*b_1).n1_0);
        }
    }
    else
    {
        _S18 = false;
    }
    return _S18;
}

fn bond_root_0( b_2 : ptr<function, BlastBond_std430_0>) -> u32
{
    var _S19 : u32 = t_label_0();
    var _S20 : u32 = (*b_2).n0_0;
    var _S21 : u32;
    if(is_dynamic_0((*b_2).n0_0))
    {
        _S21 = _S20;
    }
    else
    {
        _S21 = (*b_2).n1_0;
    }
    var _S22 : u32 = atomicLoad(&(topo_0[_S19 + _S21]));
    return _S22;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_count_bonds(@builtin(global_invocation_id) id_4 : vec3<u32>)
{
    var j_1 : u32 = id_4.x;
    if(j_1 >= (params_0.bond_count_0))
    {
        return;
    }
    var _S23 : BlastBond_std430_0 = bonds_0[j_1];
    var _S24 : bool = couples_0(&(_S23));
    if(_S24)
    {
        var _S25 : u32 = t_bonds_0();
        var _S26 : u32 = bond_root_0(&(_S23));
        var _S27 : u32 = atomicAdd(&(topo_0[_S25 + _S26]), u32(1));
    }
    return;
}

fn scan_input_0( e_0 : u32) -> vec3<u32>
{
    var _S28 : bool;
    if(e_0 >= (params_0.node_count_0))
    {
        _S28 = true;
    }
    else
    {
        var _S29 : u32 = atomicLoad(&(topo_0[t_label_0() + e_0]));
        _S28 = _S29 != e_0;
    }
    if(_S28)
    {
        return vec3<u32>(vec3<i32>(i32(0)));
    }
    var nb_0 : u32 = atomicLoad(&(topo_0[t_bonds_0() + e_0]));
    var _S30 : vec3<u32>;
    if(nb_0 > u32(0))
    {
        var _S31 : u32 = atomicLoad(&(topo_0[t_nodes_0() + e_0]));
        _S30 = vec3<u32>(u32(1), _S31, nb_0);
    }
    else
    {
        _S30 = vec3<u32>(vec3<i32>(i32(0)));
    }
    return _S30;
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

@compute
@workgroup_size(1024, 1, 1)
fn bt_scan_blocks(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var tid_1 : u32 = thread_0.x;
    var _S32 : u32 = group_0.x;
    var e_1 : u32 = _S32 * u32(1024) + tid_1;
    var v_1 : vec3<u32> = scan_input_0(e_1);
    var total_1 : vec3<u32>;
    var ex_0 : vec3<u32> = group_exclusive_scan_0(tid_1, v_1, &(total_1));
    if(e_1 < (params_0.node_count_0))
    {
        var _S33 : u32 = t_scan_0() + u32(3) * e_1;
        atomicStore(&(topo_0[_S33]), ex_0.x);
        atomicStore(&(topo_0[_S33 + u32(1)]), ex_0.y);
        atomicStore(&(topo_0[_S33 + u32(2)]), ex_0.z);
    }
    if(tid_1 == u32(0))
    {
        var _S34 : u32 = t_blocks_0() + u32(3) * _S32;
        atomicStore(&(topo_0[_S34]), total_1.x);
        atomicStore(&(topo_0[_S34 + u32(1)]), total_1.y);
        atomicStore(&(topo_0[_S34 + u32(2)]), total_1.z);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn bt_scan_top(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var tid_2 : u32 = thread_1.x;
    var _S35 : bool = tid_2 < ((params_0.node_count_0 + u32(1024) - u32(1)) / u32(1024));
    var v_2 : vec3<u32>;
    if(_S35)
    {
        var _S36 : u32 = t_blocks_0() + u32(3) * tid_2;
        var _S37 : u32 = atomicLoad(&(topo_0[_S36]));
        var _S38 : u32 = atomicLoad(&(topo_0[_S36 + u32(1)]));
        var _S39 : u32 = atomicLoad(&(topo_0[_S36 + u32(2)]));
        v_2 = vec3<u32>(_S37, _S38, _S39);
    }
    else
    {
        v_2 = vec3<u32>(vec3<i32>(i32(0)));
    }
    var total_2 : vec3<u32>;
    var ex_1 : vec3<u32> = group_exclusive_scan_0(tid_2, v_2, &(total_2));
    if(_S35)
    {
        var _S40 : u32 = t_blocks_0() + u32(3) * tid_2;
        atomicStore(&(topo_0[_S40]), ex_1.x);
        atomicStore(&(topo_0[_S40 + u32(1)]), ex_1.y);
        atomicStore(&(topo_0[_S40 + u32(2)]), ex_1.z);
    }
    if(tid_2 == u32(0))
    {
        var _S41 : u32 = t_total_0();
        atomicStore(&(topo_0[_S41]), total_2.x);
        atomicStore(&(topo_0[_S41 + u32(1)]), total_2.y);
        atomicStore(&(topo_0[_S41 + u32(2)]), total_2.z);
        args_0[i32(0)] = total_2.x;
        args_0[i32(1)] = u32(1);
        args_0[i32(2)] = u32(1);
        args_0[i32(3)] = total_2.x;
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn bt_scan_add(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var _S42 : u32 = group_1.x;
    var e_2 : u32 = _S42 * u32(1024) + thread_2.x;
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
        var _S43 : u32 = atomicLoad(&(topo_0[t_blocks_0() + u32(3) * _S42 + k_0]));
        var _S44 : u32 = atomicAdd(&(topo_0[t_scan_0() + u32(3) * e_2 + k_0]), _S43);
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
fn bt_islands(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var e_3 : u32 = id_5.x;
    if(e_3 >= (params_0.node_count_0))
    {
        return;
    }
    var v_3 : vec3<u32> = scan_input_0(e_3);
    if((v_3.x) == u32(0))
    {
        return;
    }
    var _S45 : u32 = t_scan_0() + u32(3) * e_3;
    var k_1 : u32 = atomicLoad(&(topo_0[_S45]));
    var n0_1 : u32 = atomicLoad(&(topo_0[_S45 + u32(1)]));
    var _S46 : u32 = atomicLoad(&(topo_0[t_total_0() + u32(1)]));
    var _S47 : u32 = atomicLoad(&(topo_0[_S45 + u32(2)]));
    var b0_0 : u32 = _S46 + _S47;
    islands_0[k_1] = vec4<u32>(n0_1, n0_1 + v_3.y, b0_0, b0_0 + v_3.z);
    atomicStore(&(topo_0[t_island_of_0() + e_3]), k_1);
    report_0[k_1] = vec4<u32>(vec4<i32>(i32(0)));
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_scatter_nodes(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var i_5 : u32 = id_6.x;
    var _S48 : bool;
    if(i_5 >= (params_0.node_count_0))
    {
        _S48 = true;
    }
    else
    {
        _S48 = !is_dynamic_0(i_5);
    }
    if(_S48)
    {
        return;
    }
    var r_0 : u32 = atomicLoad(&(topo_0[t_label_0() + i_5]));
    var _S49 : u32 = atomicLoad(&(topo_0[t_bonds_0() + r_0]));
    if(_S49 == u32(0))
    {
        return;
    }
    var k_2 : u32 = atomicLoad(&(topo_0[t_island_of_0() + r_0]));
    var _S50 : u32 = islands_0[k_2].x;
    var _S51 : u32 = atomicAdd(&(topo_0[t_node_cursor_0() + r_0]), u32(1));
    lists_0[_S50 + _S51] = i_5;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn bt_scatter_bonds(@builtin(global_invocation_id) id_7 : vec3<u32>)
{
    var j_2 : u32 = id_7.x;
    if(j_2 >= (params_0.bond_count_0))
    {
        return;
    }
    var _S52 : BlastBond_std430_0 = bonds_0[j_2];
    var _S53 : bool = couples_0(&(_S52));
    if(!_S53)
    {
        return;
    }
    var _S54 : u32 = bond_root_0(&(_S52));
    var k_3 : u32 = atomicLoad(&(topo_0[t_island_of_0() + _S54]));
    var _S55 : u32 = islands_0[k_3].z;
    var _S56 : u32 = atomicAdd(&(topo_0[t_bond_cursor_0() + _S54]), u32(1));
    lists_0[_S55 + _S56] = j_2;
    return;
}

var<workgroup> g_group_uniform4_0 : vec4<u32>;

fn group_uniform4_0( value_0 : vec4<u32>,  thread_3 : u32) -> vec4<u32>
{
    if(thread_3 == u32(0))
    {
        g_group_uniform4_0 = value_0;
    }
    var _S57 : vec4<u32> = (workgroupUniformLoad(&((g_group_uniform4_0))));
    return _S57;
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
        var _S58 : u32 = (k_4 >> (u32(1)));
        var j_3 : u32 = _S58;
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
                if(j_3 == _S58)
                {
                    l_0 = (t_0 ^ ((k_4 - u32(1))));
                }
                else
                {
                    l_0 = (t_0 ^ (j_3));
                }
                var _S59 : bool;
                if(l_0 > t_0)
                {
                    _S59 = l_0 < n_0;
                }
                else
                {
                    _S59 = false;
                }
                if(_S59)
                {
                    var _S60 : u32 = begin_0 + t_0;
                    var a_2 : u32 = lists_0[_S60];
                    var _S61 : u32 = begin_0 + l_0;
                    var b_3 : u32 = lists_0[_S61];
                    if(lists_0[_S60] > lists_0[_S61])
                    {
                        lists_0[_S60] = b_3;
                        lists_0[_S61] = a_2;
                    }
                }
                t_0 = t_0 + u32(256);
            }
            device_group_barrier_0();
            j_3 = (j_3 >> (u32(1)));
        }
        k_4 = (k_4 << (u32(1)));
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn bt_sort(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var tid_4 : u32 = thread_4.x;
    var _S62 : u32 = islands_0[group_2.x].x;
    var _S63 : u32 = islands_0[group_2.x].z;
    var u_0 : vec4<u32> = group_uniform4_0(vec4<u32>(_S62, islands_0[group_2.x].y - _S62, _S63, islands_0[group_2.x].w - _S63), tid_4);
    sort_list_0(tid_4, u_0.x, u_0.y);
    sort_list_0(tid_4, u_0.z, u_0.w);
    return;
}

