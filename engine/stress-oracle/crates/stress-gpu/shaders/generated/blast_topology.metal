#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint t_parent_0()
{
    return 0U;
}

struct BlastParams_0
{
    uint island_count_0;
    uint max_iterations_0;
    uint warm_0;
    uint skip_settled_0;
    float tolerance_0;
    float recip_length_0;
    uint node_count_0;
    uint bond_count_0;
    float linear_scale_0;
    float angular_scale_0;
    float bend_gain_max_0;
    uint fibres_0;
    float dt_0;
    float damage_rate_0;
    uint apply_damage_0;
    uint pad_0;
};

struct BlastBond_natural_0
{
    uint n0_0;
    uint n1_0;
    float s_0;
    float health_0;
    packed_float4 o0_0;
    packed_float4 o1_0;
};

struct BlastNode_0
{
    float w_ang_0;
    float w_lin_0;
    uint ref_begin_0;
    uint ref_end_0;
};

struct KernelContext_0
{
    BlastParams_0 constant* params_0;
    atomic<uint> device* topo_0;
    BlastBond_natural_0 device* bonds_0;
    BlastNode_0 device* nodes_0;
    uint device* args_0;
    packed_uint4 device* islands_0;
    packed_uint4 device* report_0;
    uint device* lists_0;
    array<uint3, int(1024)> threadgroup* g_scan_0;
};

uint t_nodes_0(KernelContext_0 thread* kernelContext_0)
{
    return 2U * kernelContext_0->params_0->node_count_0;
}

uint t_bonds_0(KernelContext_0 thread* kernelContext_1)
{
    return 3U * kernelContext_1->params_0->node_count_0;
}

uint t_node_cursor_0(KernelContext_0 thread* kernelContext_2)
{
    return 4U * kernelContext_2->params_0->node_count_0;
}

uint t_bond_cursor_0(KernelContext_0 thread* kernelContext_3)
{
    return 5U * kernelContext_3->params_0->node_count_0;
}

[[kernel]] void bt_reset(uint3 id_0 [[thread_position_in_grid]], BlastParams_0 constant* params_1 [[buffer(0)]], atomic<uint> device* topo_1 [[buffer(3)]], BlastBond_natural_0 device* bonds_1 [[buffer(2)]], BlastNode_0 device* nodes_1 [[buffer(1)]], uint device* args_1 [[buffer(7)]], packed_uint4 device* islands_1 [[buffer(4)]], packed_uint4 device* report_1 [[buffer(6)]], uint device* lists_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->params_0 = params_1;
    (&kernelContext_4)->topo_0 = topo_1;
    (&kernelContext_4)->bonds_0 = bonds_1;
    (&kernelContext_4)->nodes_0 = nodes_1;
    (&kernelContext_4)->args_0 = args_1;
    (&kernelContext_4)->islands_0 = islands_1;
    (&kernelContext_4)->report_0 = report_1;
    (&kernelContext_4)->lists_0 = lists_1;
    threadgroup array<uint3, int(1024)> g_scan_1;
    (&kernelContext_4)->g_scan_0 = &g_scan_1;
    uint i_0 = id_0.x;
    if(i_0 >= (params_1->node_count_0))
    {
        return;
    }
    atomic_store_explicit((&kernelContext_4)->topo_0+(t_parent_0() + i_0), i_0, memory_order_relaxed);
    uint _S1 = t_nodes_0(&kernelContext_4);
    atomic_store_explicit((&kernelContext_4)->topo_0+(_S1 + i_0), 0U, memory_order_relaxed);
    uint _S2 = t_bonds_0(&kernelContext_4);
    atomic_store_explicit((&kernelContext_4)->topo_0+(_S2 + i_0), 0U, memory_order_relaxed);
    uint _S3 = t_node_cursor_0(&kernelContext_4);
    atomic_store_explicit((&kernelContext_4)->topo_0+(_S3 + i_0), 0U, memory_order_relaxed);
    uint _S4 = t_bond_cursor_0(&kernelContext_4);
    atomic_store_explicit((&kernelContext_4)->topo_0+(_S4 + i_0), 0U, memory_order_relaxed);
    return;
}

bool is_dynamic_0(uint i_1, KernelContext_0 thread* kernelContext_5)
{
    return (kernelContext_5->nodes_0[i_1].w_lin_0) > 0.0f;
}

uint find_0(uint i_2, KernelContext_0 thread* kernelContext_6)
{
    uint _S5 = t_parent_0();
    uint _S6 = atomic_load_explicit(kernelContext_6->topo_0+(_S5 + i_2), memory_order_relaxed);
    uint p_0 = _S6;
    uint _S7 = i_2;
    for(;;)
    {
        if(p_0 != _S7)
        {
        }
        else
        {
            break;
        }
        uint _S8 = _S5 + p_0;
        uint g_0 = atomic_load_explicit(kernelContext_6->topo_0+_S8, memory_order_relaxed);
        if(g_0 != p_0)
        {
            uint _S9;
            {
            uint _metal_cas_comparand = p_0;
            atomic_compare_exchange_weak_explicit(kernelContext_6->topo_0+(_S5 + _S7), &_metal_cas_comparand, g_0, memory_order_relaxed, memory_order_relaxed);
            _S9 = _metal_cas_comparand;
            }
        }
        uint _S10 = atomic_load_explicit(kernelContext_6->topo_0+_S8, memory_order_relaxed);
        _S7 = p_0;
        p_0 = _S10;
    }
    return _S7;
}

uint t_blocks_0(KernelContext_0 thread* kernelContext_7)
{
    return 10U * kernelContext_7->params_0->node_count_0;
}

uint t_total_0(KernelContext_0 thread* kernelContext_8)
{
    uint _S11 = t_blocks_0(kernelContext_8);
    return _S11 + 3U * ((kernelContext_8->params_0->node_count_0 + 1024U - 1U) / 1024U);
}

[[kernel]] void bt_union(uint3 id_1 [[thread_position_in_grid]], BlastParams_0 constant* params_2 [[buffer(0)]], atomic<uint> device* topo_2 [[buffer(3)]], BlastBond_natural_0 device* bonds_2 [[buffer(2)]], BlastNode_0 device* nodes_2 [[buffer(1)]], uint device* args_2 [[buffer(7)]], packed_uint4 device* islands_2 [[buffer(4)]], packed_uint4 device* report_2 [[buffer(6)]], uint device* lists_2 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->params_0 = params_2;
    (&kernelContext_9)->topo_0 = topo_2;
    (&kernelContext_9)->bonds_0 = bonds_2;
    (&kernelContext_9)->nodes_0 = nodes_2;
    (&kernelContext_9)->args_0 = args_2;
    (&kernelContext_9)->islands_0 = islands_2;
    (&kernelContext_9)->report_0 = report_2;
    (&kernelContext_9)->lists_0 = lists_2;
    threadgroup array<uint3, int(1024)> g_scan_2;
    (&kernelContext_9)->g_scan_0 = &g_scan_2;
    uint j_0 = id_1.x;
    if(j_0 >= (params_2->bond_count_0))
    {
        return;
    }
    BlastBond_natural_0 b_0 = (&kernelContext_9)->bonds_0[j_0];
    bool _S12;
    if(!((b_0.health_0) > 0.0f))
    {
        _S12 = true;
    }
    else
    {
        bool _S13 = is_dynamic_0(b_0.n0_0, &kernelContext_9);
        _S12 = !_S13;
    }
    if(_S12)
    {
        _S12 = true;
    }
    else
    {
        bool _S14 = is_dynamic_0(b_0.n1_0, &kernelContext_9);
        _S12 = !_S14;
    }
    if(_S12)
    {
        return;
    }
    uint a_0 = b_0.n0_0;
    uint c_0 = b_0.n1_0;
    uint guard_0 = 0U;
    for(;;)
    {
        if(guard_0 < 4096U)
        {
        }
        else
        {
            break;
        }
        uint _S15 = find_0(a_0, &kernelContext_9);
        uint _S16 = find_0(c_0, &kernelContext_9);
        if(_S15 == _S16)
        {
            return;
        }
        uint _S17 = max(_S15, _S16);
        uint _S18 = min(_S15, _S16);
        uint _S19;
        {
        uint _metal_cas_comparand = _S17;
        atomic_compare_exchange_weak_explicit((&kernelContext_9)->topo_0+(t_parent_0() + _S17), &_metal_cas_comparand, _S18, memory_order_relaxed, memory_order_relaxed);
        _S19 = _metal_cas_comparand;
        }
        if(_S19 == _S17)
        {
            return;
        }
        uint _S20 = guard_0 + 1U;
        a_0 = _S17;
        c_0 = _S18;
        guard_0 = _S20;
    }
    uint _S21 = t_total_0(&kernelContext_9);
    atomic_store_explicit((&kernelContext_9)->topo_0+(_S21 + 3U), 1U, memory_order_relaxed);
    return;
}

uint t_label_0(KernelContext_0 thread* kernelContext_10)
{
    return kernelContext_10->params_0->node_count_0;
}

[[kernel]] void bt_flatten(uint3 id_2 [[thread_position_in_grid]], BlastParams_0 constant* params_3 [[buffer(0)]], atomic<uint> device* topo_3 [[buffer(3)]], BlastBond_natural_0 device* bonds_3 [[buffer(2)]], BlastNode_0 device* nodes_3 [[buffer(1)]], uint device* args_3 [[buffer(7)]], packed_uint4 device* islands_3 [[buffer(4)]], packed_uint4 device* report_3 [[buffer(6)]], uint device* lists_3 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_11;
    (&kernelContext_11)->params_0 = params_3;
    (&kernelContext_11)->topo_0 = topo_3;
    (&kernelContext_11)->bonds_0 = bonds_3;
    (&kernelContext_11)->nodes_0 = nodes_3;
    (&kernelContext_11)->args_0 = args_3;
    (&kernelContext_11)->islands_0 = islands_3;
    (&kernelContext_11)->report_0 = report_3;
    (&kernelContext_11)->lists_0 = lists_3;
    threadgroup array<uint3, int(1024)> g_scan_3;
    (&kernelContext_11)->g_scan_0 = &g_scan_3;
    uint i_3 = id_2.x;
    if(i_3 >= (params_3->node_count_0))
    {
        return;
    }
    uint _S22 = t_label_0(&kernelContext_11);
    atomic<uint> device* _S23 = (&kernelContext_11)->topo_0+(_S22 + i_3);
    bool _S24 = is_dynamic_0(i_3, &kernelContext_11);
    uint _S25;
    if(_S24)
    {
        uint _S26 = find_0(i_3, &kernelContext_11);
        _S25 = _S26;
    }
    else
    {
        _S25 = 4294967295U;
    }
    atomic_store_explicit(_S23, _S25, memory_order_relaxed);
    return;
}

[[kernel]] void bt_count_nodes(uint3 id_3 [[thread_position_in_grid]], BlastParams_0 constant* params_4 [[buffer(0)]], atomic<uint> device* topo_4 [[buffer(3)]], BlastBond_natural_0 device* bonds_4 [[buffer(2)]], BlastNode_0 device* nodes_4 [[buffer(1)]], uint device* args_4 [[buffer(7)]], packed_uint4 device* islands_4 [[buffer(4)]], packed_uint4 device* report_4 [[buffer(6)]], uint device* lists_4 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_12;
    (&kernelContext_12)->params_0 = params_4;
    (&kernelContext_12)->topo_0 = topo_4;
    (&kernelContext_12)->bonds_0 = bonds_4;
    (&kernelContext_12)->nodes_0 = nodes_4;
    (&kernelContext_12)->args_0 = args_4;
    (&kernelContext_12)->islands_0 = islands_4;
    (&kernelContext_12)->report_0 = report_4;
    (&kernelContext_12)->lists_0 = lists_4;
    threadgroup array<uint3, int(1024)> g_scan_4;
    (&kernelContext_12)->g_scan_0 = &g_scan_4;
    uint i_4 = id_3.x;
    bool _S27;
    if(i_4 >= (params_4->node_count_0))
    {
        _S27 = true;
    }
    else
    {
        bool _S28 = is_dynamic_0(i_4, &kernelContext_12);
        _S27 = !_S28;
    }
    if(_S27)
    {
        return;
    }
    uint _S29 = t_nodes_0(&kernelContext_12);
    uint _S30 = t_label_0(&kernelContext_12);
    uint _S31 = atomic_load_explicit((&kernelContext_12)->topo_0+(_S30 + i_4), memory_order_relaxed);
    uint _S32 = atomic_fetch_add_explicit((&kernelContext_12)->topo_0+(_S29 + _S31), 1U, memory_order_relaxed);
    return;
}

bool couples_0(const BlastBond_natural_0 thread* b_1, KernelContext_0 thread* kernelContext_13)
{
    bool _S33;
    if((b_1->health_0) > 0.0f)
    {
        bool _S34 = is_dynamic_0(b_1->n0_0, kernelContext_13);
        if(_S34)
        {
            _S33 = true;
        }
        else
        {
            bool _S35 = is_dynamic_0(b_1->n1_0, kernelContext_13);
            _S33 = _S35;
        }
    }
    else
    {
        _S33 = false;
    }
    return _S33;
}

uint bond_root_0(const BlastBond_natural_0 thread* b_2, KernelContext_0 thread* kernelContext_14)
{
    uint _S36 = t_label_0(kernelContext_14);
    uint _S37 = b_2->n0_0;
    bool _S38 = is_dynamic_0(b_2->n0_0, kernelContext_14);
    uint _S39;
    if(_S38)
    {
        _S39 = _S37;
    }
    else
    {
        _S39 = b_2->n1_0;
    }
    uint _S40 = atomic_load_explicit(kernelContext_14->topo_0+(_S36 + _S39), memory_order_relaxed);
    return _S40;
}

[[kernel]] void bt_count_bonds(uint3 id_4 [[thread_position_in_grid]], BlastParams_0 constant* params_5 [[buffer(0)]], atomic<uint> device* topo_5 [[buffer(3)]], BlastBond_natural_0 device* bonds_5 [[buffer(2)]], BlastNode_0 device* nodes_5 [[buffer(1)]], uint device* args_5 [[buffer(7)]], packed_uint4 device* islands_5 [[buffer(4)]], packed_uint4 device* report_5 [[buffer(6)]], uint device* lists_5 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_15;
    (&kernelContext_15)->params_0 = params_5;
    (&kernelContext_15)->topo_0 = topo_5;
    (&kernelContext_15)->bonds_0 = bonds_5;
    (&kernelContext_15)->nodes_0 = nodes_5;
    (&kernelContext_15)->args_0 = args_5;
    (&kernelContext_15)->islands_0 = islands_5;
    (&kernelContext_15)->report_0 = report_5;
    (&kernelContext_15)->lists_0 = lists_5;
    threadgroup array<uint3, int(1024)> g_scan_5;
    (&kernelContext_15)->g_scan_0 = &g_scan_5;
    uint j_1 = id_4.x;
    if(j_1 >= (params_5->bond_count_0))
    {
        return;
    }
    thread BlastBond_natural_0 _S41 = (&kernelContext_15)->bonds_0[j_1];
    bool _S42 = couples_0(&_S41, &kernelContext_15);
    if(_S42)
    {
        uint _S43 = t_bonds_0(&kernelContext_15);
        uint _S44 = bond_root_0(&_S41, &kernelContext_15);
        uint _S45 = atomic_fetch_add_explicit((&kernelContext_15)->topo_0+(_S43 + _S44), 1U, memory_order_relaxed);
    }
    return;
}

uint3 scan_input_0(uint e_0, KernelContext_0 thread* kernelContext_16)
{
    bool _S46;
    if(e_0 >= (kernelContext_16->params_0->node_count_0))
    {
        _S46 = true;
    }
    else
    {
        uint _S47 = t_label_0(kernelContext_16);
        uint _S48 = atomic_load_explicit(kernelContext_16->topo_0+(_S47 + e_0), memory_order_relaxed);
        _S46 = _S48 != e_0;
    }
    if(_S46)
    {
        return uint3(int3(int(0)) );
    }
    uint _S49 = t_bonds_0(kernelContext_16);
    uint nb_0 = atomic_load_explicit(kernelContext_16->topo_0+(_S49 + e_0), memory_order_relaxed);
    uint3 _S50;
    if(nb_0 > 0U)
    {
        uint _S51 = t_nodes_0(kernelContext_16);
        uint _S52 = atomic_load_explicit(kernelContext_16->topo_0+(_S51 + e_0), memory_order_relaxed);
        _S50 = uint3(1U, _S52, nb_0);
    }
    else
    {
        _S50 = uint3(int3(int(0)) );
    }
    return _S50;
}

uint3 group_exclusive_scan_0(uint tid_0, uint3 v_0, uint3 thread* total_0, KernelContext_0 thread* kernelContext_17)
{
    (*kernelContext_17->g_scan_0)[tid_0] = v_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint d_0 = 1U;
    for(;;)
    {
        if(d_0 < 1024U)
        {
        }
        else
        {
            break;
        }
        uint3 add_0;
        if(tid_0 >= d_0)
        {
            add_0 = (*kernelContext_17->g_scan_0)[tid_0 - d_0];
        }
        else
        {
            add_0 = uint3(int3(int(0)) );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*kernelContext_17->g_scan_0)[tid_0] = (*kernelContext_17->g_scan_0)[tid_0] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        d_0 = d_0 << 1U;
    }
    *total_0 = (*kernelContext_17->g_scan_0)[1023U];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_17->g_scan_0)[tid_0] - v_0;
}

uint t_scan_0(KernelContext_0 thread* kernelContext_18)
{
    return 6U * kernelContext_18->params_0->node_count_0;
}

[[kernel]] void bt_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_6 [[buffer(0)]], atomic<uint> device* topo_6 [[buffer(3)]], BlastBond_natural_0 device* bonds_6 [[buffer(2)]], BlastNode_0 device* nodes_6 [[buffer(1)]], uint device* args_6 [[buffer(7)]], packed_uint4 device* islands_6 [[buffer(4)]], packed_uint4 device* report_6 [[buffer(6)]], uint device* lists_6 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_19;
    (&kernelContext_19)->params_0 = params_6;
    (&kernelContext_19)->topo_0 = topo_6;
    (&kernelContext_19)->bonds_0 = bonds_6;
    (&kernelContext_19)->nodes_0 = nodes_6;
    (&kernelContext_19)->args_0 = args_6;
    (&kernelContext_19)->islands_0 = islands_6;
    (&kernelContext_19)->report_0 = report_6;
    (&kernelContext_19)->lists_0 = lists_6;
    threadgroup array<uint3, int(1024)> g_scan_6;
    (&kernelContext_19)->g_scan_0 = &g_scan_6;
    uint tid_1 = thread_0.x;
    uint _S53 = group_0.x;
    uint e_1 = _S53 * 1024U + tid_1;
    uint3 _S54 = scan_input_0(e_1, &kernelContext_19);
    thread uint3 total_1;
    uint3 _S55 = group_exclusive_scan_0(tid_1, _S54, &total_1, &kernelContext_19);
    if(e_1 < ((&kernelContext_19)->params_0->node_count_0))
    {
        uint _S56 = t_scan_0(&kernelContext_19);
        uint _S57 = _S56 + 3U * e_1;
        atomic_store_explicit((&kernelContext_19)->topo_0+_S57, _S55.x, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_19)->topo_0+(_S57 + 1U), _S55.y, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_19)->topo_0+(_S57 + 2U), _S55.z, memory_order_relaxed);
    }
    if(tid_1 == 0U)
    {
        uint _S58 = t_blocks_0(&kernelContext_19);
        uint _S59 = _S58 + 3U * _S53;
        atomic_store_explicit((&kernelContext_19)->topo_0+_S59, total_1.x, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_19)->topo_0+(_S59 + 1U), total_1.y, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_19)->topo_0+(_S59 + 2U), total_1.z, memory_order_relaxed);
    }
    return;
}

[[kernel]] void bt_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_7 [[buffer(0)]], atomic<uint> device* topo_7 [[buffer(3)]], BlastBond_natural_0 device* bonds_7 [[buffer(2)]], BlastNode_0 device* nodes_7 [[buffer(1)]], uint device* args_7 [[buffer(7)]], packed_uint4 device* islands_7 [[buffer(4)]], packed_uint4 device* report_7 [[buffer(6)]], uint device* lists_7 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_20;
    (&kernelContext_20)->params_0 = params_7;
    (&kernelContext_20)->topo_0 = topo_7;
    (&kernelContext_20)->bonds_0 = bonds_7;
    (&kernelContext_20)->nodes_0 = nodes_7;
    (&kernelContext_20)->args_0 = args_7;
    (&kernelContext_20)->islands_0 = islands_7;
    (&kernelContext_20)->report_0 = report_7;
    (&kernelContext_20)->lists_0 = lists_7;
    threadgroup array<uint3, int(1024)> g_scan_7;
    (&kernelContext_20)->g_scan_0 = &g_scan_7;
    uint tid_2 = thread_1.x;
    bool _S60 = tid_2 < ((params_7->node_count_0 + 1024U - 1U) / 1024U);
    uint3 v_1;
    if(_S60)
    {
        uint _S61 = t_blocks_0(&kernelContext_20);
        uint _S62 = _S61 + 3U * tid_2;
        uint _S63 = atomic_load_explicit((&kernelContext_20)->topo_0+_S62, memory_order_relaxed);
        uint _S64 = atomic_load_explicit((&kernelContext_20)->topo_0+(_S62 + 1U), memory_order_relaxed);
        uint _S65 = atomic_load_explicit((&kernelContext_20)->topo_0+(_S62 + 2U), memory_order_relaxed);
        v_1 = uint3(_S63, _S64, _S65);
    }
    else
    {
        v_1 = uint3(int3(int(0)) );
    }
    thread uint3 total_2;
    uint3 _S66 = group_exclusive_scan_0(tid_2, v_1, &total_2, &kernelContext_20);
    if(_S60)
    {
        uint _S67 = t_blocks_0(&kernelContext_20);
        uint _S68 = _S67 + 3U * tid_2;
        atomic_store_explicit((&kernelContext_20)->topo_0+_S68, _S66.x, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_20)->topo_0+(_S68 + 1U), _S66.y, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_20)->topo_0+(_S68 + 2U), _S66.z, memory_order_relaxed);
    }
    if(tid_2 == 0U)
    {
        uint _S69 = t_total_0(&kernelContext_20);
        atomic_store_explicit((&kernelContext_20)->topo_0+_S69, total_2.x, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_20)->topo_0+(_S69 + 1U), total_2.y, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_20)->topo_0+(_S69 + 2U), total_2.z, memory_order_relaxed);
        *((&kernelContext_20)->args_0+int(0)) = total_2.x;
        *((&kernelContext_20)->args_0+int(1)) = 1U;
        *((&kernelContext_20)->args_0+int(2)) = 1U;
        *((&kernelContext_20)->args_0+int(3)) = total_2.x;
    }
    return;
}

[[kernel]] void bt_scan_add(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_8 [[buffer(0)]], atomic<uint> device* topo_8 [[buffer(3)]], BlastBond_natural_0 device* bonds_8 [[buffer(2)]], BlastNode_0 device* nodes_8 [[buffer(1)]], uint device* args_8 [[buffer(7)]], packed_uint4 device* islands_8 [[buffer(4)]], packed_uint4 device* report_8 [[buffer(6)]], uint device* lists_8 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_21;
    (&kernelContext_21)->params_0 = params_8;
    (&kernelContext_21)->topo_0 = topo_8;
    (&kernelContext_21)->bonds_0 = bonds_8;
    (&kernelContext_21)->nodes_0 = nodes_8;
    (&kernelContext_21)->args_0 = args_8;
    (&kernelContext_21)->islands_0 = islands_8;
    (&kernelContext_21)->report_0 = report_8;
    (&kernelContext_21)->lists_0 = lists_8;
    threadgroup array<uint3, int(1024)> g_scan_8;
    (&kernelContext_21)->g_scan_0 = &g_scan_8;
    uint _S70 = group_1.x;
    uint e_2 = _S70 * 1024U + thread_2.x;
    if(e_2 >= (params_8->node_count_0))
    {
        return;
    }
    uint k_0 = 0U;
    for(;;)
    {
        if(k_0 < 3U)
        {
        }
        else
        {
            break;
        }
        uint _S71 = t_scan_0(&kernelContext_21);
        atomic<uint> device* _S72 = (&kernelContext_21)->topo_0+(_S71 + 3U * e_2 + k_0);
        uint _S73 = t_blocks_0(&kernelContext_21);
        uint _S74 = atomic_load_explicit((&kernelContext_21)->topo_0+(_S73 + 3U * _S70 + k_0), memory_order_relaxed);
        uint _S75 = atomic_fetch_add_explicit(_S72, _S74, memory_order_relaxed);
        k_0 = k_0 + 1U;
    }
    return;
}

uint t_island_of_0(KernelContext_0 thread* kernelContext_22)
{
    return 9U * kernelContext_22->params_0->node_count_0;
}

[[kernel]] void bt_islands(uint3 id_5 [[thread_position_in_grid]], BlastParams_0 constant* params_9 [[buffer(0)]], atomic<uint> device* topo_9 [[buffer(3)]], BlastBond_natural_0 device* bonds_9 [[buffer(2)]], BlastNode_0 device* nodes_9 [[buffer(1)]], uint device* args_9 [[buffer(7)]], packed_uint4 device* islands_9 [[buffer(4)]], packed_uint4 device* report_9 [[buffer(6)]], uint device* lists_9 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_23;
    (&kernelContext_23)->params_0 = params_9;
    (&kernelContext_23)->topo_0 = topo_9;
    (&kernelContext_23)->bonds_0 = bonds_9;
    (&kernelContext_23)->nodes_0 = nodes_9;
    (&kernelContext_23)->args_0 = args_9;
    (&kernelContext_23)->islands_0 = islands_9;
    (&kernelContext_23)->report_0 = report_9;
    (&kernelContext_23)->lists_0 = lists_9;
    threadgroup array<uint3, int(1024)> g_scan_9;
    (&kernelContext_23)->g_scan_0 = &g_scan_9;
    uint e_3 = id_5.x;
    if(e_3 >= (params_9->node_count_0))
    {
        return;
    }
    uint3 _S76 = scan_input_0(e_3, &kernelContext_23);
    if((_S76.x) == 0U)
    {
        return;
    }
    uint _S77 = t_scan_0(&kernelContext_23);
    uint _S78 = _S77 + 3U * e_3;
    uint k_1 = atomic_load_explicit((&kernelContext_23)->topo_0+_S78, memory_order_relaxed);
    uint n0_1 = atomic_load_explicit((&kernelContext_23)->topo_0+(_S78 + 1U), memory_order_relaxed);
    uint _S79 = t_total_0(&kernelContext_23);
    uint _S80 = atomic_load_explicit((&kernelContext_23)->topo_0+(_S79 + 1U), memory_order_relaxed);
    uint _S81 = atomic_load_explicit((&kernelContext_23)->topo_0+(_S78 + 2U), memory_order_relaxed);
    uint b0_0 = _S80 + _S81;
    *((&kernelContext_23)->islands_0+k_1) = packed_uint4(uint4(n0_1, n0_1 + _S76.y, b0_0, b0_0 + _S76.z)) ;
    uint _S82 = t_island_of_0(&kernelContext_23);
    atomic_store_explicit((&kernelContext_23)->topo_0+(_S82 + e_3), k_1, memory_order_relaxed);
    *((&kernelContext_23)->report_0+k_1) = packed_uint4(uint4(int4(int(0)) )) ;
    return;
}

[[kernel]] void bt_scatter_nodes(uint3 id_6 [[thread_position_in_grid]], BlastParams_0 constant* params_10 [[buffer(0)]], atomic<uint> device* topo_10 [[buffer(3)]], BlastBond_natural_0 device* bonds_10 [[buffer(2)]], BlastNode_0 device* nodes_10 [[buffer(1)]], uint device* args_10 [[buffer(7)]], packed_uint4 device* islands_10 [[buffer(4)]], packed_uint4 device* report_10 [[buffer(6)]], uint device* lists_10 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_24;
    (&kernelContext_24)->params_0 = params_10;
    (&kernelContext_24)->topo_0 = topo_10;
    (&kernelContext_24)->bonds_0 = bonds_10;
    (&kernelContext_24)->nodes_0 = nodes_10;
    (&kernelContext_24)->args_0 = args_10;
    (&kernelContext_24)->islands_0 = islands_10;
    (&kernelContext_24)->report_0 = report_10;
    (&kernelContext_24)->lists_0 = lists_10;
    threadgroup array<uint3, int(1024)> g_scan_10;
    (&kernelContext_24)->g_scan_0 = &g_scan_10;
    uint i_5 = id_6.x;
    bool _S83;
    if(i_5 >= (params_10->node_count_0))
    {
        _S83 = true;
    }
    else
    {
        bool _S84 = is_dynamic_0(i_5, &kernelContext_24);
        _S83 = !_S84;
    }
    if(_S83)
    {
        return;
    }
    uint _S85 = t_label_0(&kernelContext_24);
    uint r_0 = atomic_load_explicit((&kernelContext_24)->topo_0+(_S85 + i_5), memory_order_relaxed);
    uint _S86 = t_bonds_0(&kernelContext_24);
    uint _S87 = atomic_load_explicit((&kernelContext_24)->topo_0+(_S86 + r_0), memory_order_relaxed);
    if(_S87 == 0U)
    {
        return;
    }
    uint _S88 = t_island_of_0(&kernelContext_24);
    uint k_2 = atomic_load_explicit((&kernelContext_24)->topo_0+(_S88 + r_0), memory_order_relaxed);
    uint _S89 = (uint4(*((&kernelContext_24)->islands_0+k_2)) ).x;
    uint _S90 = t_node_cursor_0(&kernelContext_24);
    uint _S91 = atomic_fetch_add_explicit((&kernelContext_24)->topo_0+(_S90 + r_0), 1U, memory_order_relaxed);
    *((&kernelContext_24)->lists_0+(_S89 + _S91)) = i_5;
    return;
}

[[kernel]] void bt_scatter_bonds(uint3 id_7 [[thread_position_in_grid]], BlastParams_0 constant* params_11 [[buffer(0)]], atomic<uint> device* topo_11 [[buffer(3)]], BlastBond_natural_0 device* bonds_11 [[buffer(2)]], BlastNode_0 device* nodes_11 [[buffer(1)]], uint device* args_11 [[buffer(7)]], packed_uint4 device* islands_11 [[buffer(4)]], packed_uint4 device* report_11 [[buffer(6)]], uint device* lists_11 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_25;
    (&kernelContext_25)->params_0 = params_11;
    (&kernelContext_25)->topo_0 = topo_11;
    (&kernelContext_25)->bonds_0 = bonds_11;
    (&kernelContext_25)->nodes_0 = nodes_11;
    (&kernelContext_25)->args_0 = args_11;
    (&kernelContext_25)->islands_0 = islands_11;
    (&kernelContext_25)->report_0 = report_11;
    (&kernelContext_25)->lists_0 = lists_11;
    threadgroup array<uint3, int(1024)> g_scan_11;
    (&kernelContext_25)->g_scan_0 = &g_scan_11;
    uint j_2 = id_7.x;
    if(j_2 >= (params_11->bond_count_0))
    {
        return;
    }
    thread BlastBond_natural_0 _S92 = (&kernelContext_25)->bonds_0[j_2];
    bool _S93 = couples_0(&_S92, &kernelContext_25);
    if(!_S93)
    {
        return;
    }
    uint _S94 = bond_root_0(&_S92, &kernelContext_25);
    uint _S95 = t_island_of_0(&kernelContext_25);
    uint k_3 = atomic_load_explicit((&kernelContext_25)->topo_0+(_S95 + _S94), memory_order_relaxed);
    uint _S96 = (uint4(*((&kernelContext_25)->islands_0+k_3)) ).z;
    uint _S97 = t_bond_cursor_0(&kernelContext_25);
    uint _S98 = atomic_fetch_add_explicit((&kernelContext_25)->topo_0+(_S97 + _S94), 1U, memory_order_relaxed);
    *((&kernelContext_25)->lists_0+(_S96 + _S98)) = j_2;
    return;
}

uint4 group_uniform4_0(uint4 value_0, uint thread_3)
{
    return value_0;
}

void sort_list_0(uint tid_3, uint begin_0, uint n_0, KernelContext_0 thread* kernelContext_26)
{
    uint span_0 = 1U;
    for(;;)
    {
        if(span_0 < n_0)
        {
        }
        else
        {
            break;
        }
        span_0 = span_0 << 1U;
    }
    uint k_4 = 2U;
    for(;;)
    {
        if(k_4 <= span_0)
        {
        }
        else
        {
            break;
        }
        uint _S99 = k_4 >> 1U;
        uint j_3 = _S99;
        for(;;)
        {
            if(j_3 > 0U)
            {
            }
            else
            {
                break;
            }
            uint t_0 = tid_3;
            for(;;)
            {
                if(t_0 < span_0)
                {
                }
                else
                {
                    break;
                }
                uint l_0;
                if(j_3 == _S99)
                {
                    l_0 = t_0 ^ (k_4 - 1U);
                }
                else
                {
                    l_0 = t_0 ^ j_3;
                }
                bool _S100;
                if(l_0 > t_0)
                {
                    _S100 = l_0 < n_0;
                }
                else
                {
                    _S100 = false;
                }
                if(_S100)
                {
                    uint _S101 = begin_0 + t_0;
                    uint device* _S102 = kernelContext_26->lists_0+_S101;
                    uint a_1 = *_S102;
                    uint _S103 = begin_0 + l_0;
                    uint device* _S104 = kernelContext_26->lists_0+_S103;
                    uint b_3 = *_S104;
                    if((*_S102) > (*_S104))
                    {
                        *(kernelContext_26->lists_0+_S101) = b_3;
                        *(kernelContext_26->lists_0+_S103) = a_1;
                    }
                }
                t_0 = t_0 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            j_3 = j_3 >> 1U;
        }
        k_4 = k_4 << 1U;
    }
    return;
}

[[kernel]] void bt_sort(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_12 [[buffer(0)]], atomic<uint> device* topo_12 [[buffer(3)]], BlastBond_natural_0 device* bonds_12 [[buffer(2)]], BlastNode_0 device* nodes_12 [[buffer(1)]], uint device* args_12 [[buffer(7)]], packed_uint4 device* islands_12 [[buffer(4)]], packed_uint4 device* report_12 [[buffer(6)]], uint device* lists_12 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_27;
    (&kernelContext_27)->params_0 = params_12;
    (&kernelContext_27)->topo_0 = topo_12;
    (&kernelContext_27)->bonds_0 = bonds_12;
    (&kernelContext_27)->nodes_0 = nodes_12;
    (&kernelContext_27)->args_0 = args_12;
    (&kernelContext_27)->islands_0 = islands_12;
    (&kernelContext_27)->report_0 = report_12;
    (&kernelContext_27)->lists_0 = lists_12;
    threadgroup array<uint3, int(1024)> g_scan_12;
    (&kernelContext_27)->g_scan_0 = &g_scan_12;
    uint tid_4 = thread_4.x;
    uint4 _S105 = uint4(*(islands_12+group_2.x)) ;
    uint _S106 = _S105.x;
    uint _S107 = _S105.z;
    uint4 u_0 = group_uniform4_0(uint4(_S106, _S105.y - _S106, _S107, _S105.w - _S107), tid_4);
    sort_list_0(tid_4, u_0.x, u_0.y, &kernelContext_27);
    sort_list_0(tid_4, u_0.z, u_0.w, &kernelContext_27);
    return;
}

