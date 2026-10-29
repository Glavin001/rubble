#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint topo_total_0(uint node_count_0)
{
    return 10U * node_count_0 + 3U * ((node_count_0 + 1023U) / 1024U);
}

uint topo_dirty_0(uint node_count_1)
{
    return topo_total_0(node_count_1) + 4U;
}

struct BlastParams_0
{
    uint island_count_0;
    uint max_iterations_0;
    uint warm_0;
    uint skip_settled_0;
    float tolerance_0;
    float recip_length_0;
    uint node_count_2;
    uint bond_count_0;
    float linear_scale_0;
    float angular_scale_0;
    float bend_gain_max_0;
    uint fibres_0;
    float dt_0;
    float damage_rate_0;
    uint apply_damage_0;
    float force_tolerance_0;
    float4 gravity_0;
    uint w_alive_0;
    uint a_dirty_0;
    uint a_counters_0;
    uint pad1_0;
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

struct BlastNode_natural_0
{
    float w_ang_0;
    float w_lin_0;
    uint ref_begin_0;
    uint ref_end_0;
    packed_float4 position_0;
};

struct KernelContext_0
{
    BlastParams_0 constant* params_0;
    atomic<uint> device* topo_0;
    uint device* args_0;
    BlastBond_natural_0 device* bonds_0;
    BlastNode_natural_0 device* nodes_0;
    packed_uint4 device* islands_0;
    packed_uint4 device* report_0;
    uint device* lists_0;
    array<uint3, int(1024)> threadgroup* g_scan_0;
};

void set_args_0(uint at_0, uint x_0, KernelContext_0 thread* kernelContext_0)
{
    *(kernelContext_0->args_0+at_0) = x_0;
    *(kernelContext_0->args_0+(at_0 + 1U)) = 1U;
    *(kernelContext_0->args_0+(at_0 + 2U)) = 1U;
    return;
}

[[kernel]] void bt_gate(uint3 id_0 [[thread_position_in_grid]], BlastParams_0 constant* params_1 [[buffer(0)]], atomic<uint> device* topo_1 [[buffer(3)]], uint device* args_1 [[buffer(7)]], BlastBond_natural_0 device* bonds_1 [[buffer(2)]], BlastNode_natural_0 device* nodes_1 [[buffer(1)]], packed_uint4 device* islands_1 [[buffer(4)]], packed_uint4 device* report_1 [[buffer(6)]], uint device* lists_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_1;
    (&kernelContext_1)->topo_0 = topo_1;
    (&kernelContext_1)->args_0 = args_1;
    (&kernelContext_1)->bonds_0 = bonds_1;
    (&kernelContext_1)->nodes_0 = nodes_1;
    (&kernelContext_1)->islands_0 = islands_1;
    (&kernelContext_1)->report_0 = report_1;
    (&kernelContext_1)->lists_0 = lists_1;
    threadgroup array<uint3, int(1024)> g_scan_1;
    (&kernelContext_1)->g_scan_0 = &g_scan_1;
    uint _S1 = atomic_exchange_explicit(topo_1+topo_dirty_0(params_1->node_count_2), 0U, memory_order_relaxed);
    bool run_0 = _S1 != 0U;
    uint _S2;
    if(run_0)
    {
        _S2 = (params_1->node_count_2 + 63U) / 64U;
    }
    else
    {
        _S2 = 0U;
    }
    set_args_0(4U, _S2, &kernelContext_1);
    if(run_0)
    {
        _S2 = ((&kernelContext_1)->params_0->bond_count_0 + 63U) / 64U;
    }
    else
    {
        _S2 = 0U;
    }
    set_args_0(7U, _S2, &kernelContext_1);
    if(run_0)
    {
        _S2 = (params_1->node_count_2 + 1024U - 1U) / 1024U;
    }
    else
    {
        _S2 = 0U;
    }
    set_args_0(10U, _S2, &kernelContext_1);
    int _S3;
    if(run_0)
    {
        _S3 = int(1);
    }
    else
    {
        _S3 = int(0);
    }
    set_args_0(13U, uint(_S3), &kernelContext_1);
    set_args_0(16U, 0U, &kernelContext_1);
    return;
}

uint t_parent_0()
{
    return 0U;
}

uint t_nodes_0(KernelContext_0 thread* kernelContext_2)
{
    return 2U * kernelContext_2->params_0->node_count_2;
}

uint t_bonds_0(KernelContext_0 thread* kernelContext_3)
{
    return 3U * kernelContext_3->params_0->node_count_2;
}

uint t_node_cursor_0(KernelContext_0 thread* kernelContext_4)
{
    return 4U * kernelContext_4->params_0->node_count_2;
}

uint t_bond_cursor_0(KernelContext_0 thread* kernelContext_5)
{
    return 5U * kernelContext_5->params_0->node_count_2;
}

[[kernel]] void bt_reset(uint3 id_1 [[thread_position_in_grid]], BlastParams_0 constant* params_2 [[buffer(0)]], atomic<uint> device* topo_2 [[buffer(3)]], uint device* args_2 [[buffer(7)]], BlastBond_natural_0 device* bonds_2 [[buffer(2)]], BlastNode_natural_0 device* nodes_2 [[buffer(1)]], packed_uint4 device* islands_2 [[buffer(4)]], packed_uint4 device* report_2 [[buffer(6)]], uint device* lists_2 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->params_0 = params_2;
    (&kernelContext_6)->topo_0 = topo_2;
    (&kernelContext_6)->args_0 = args_2;
    (&kernelContext_6)->bonds_0 = bonds_2;
    (&kernelContext_6)->nodes_0 = nodes_2;
    (&kernelContext_6)->islands_0 = islands_2;
    (&kernelContext_6)->report_0 = report_2;
    (&kernelContext_6)->lists_0 = lists_2;
    threadgroup array<uint3, int(1024)> g_scan_2;
    (&kernelContext_6)->g_scan_0 = &g_scan_2;
    uint i_0 = id_1.x;
    if(i_0 >= (params_2->node_count_2))
    {
        return;
    }
    atomic_store_explicit((&kernelContext_6)->topo_0+(t_parent_0() + i_0), i_0, memory_order_relaxed);
    uint _S4 = t_nodes_0(&kernelContext_6);
    atomic_store_explicit((&kernelContext_6)->topo_0+(_S4 + i_0), 0U, memory_order_relaxed);
    uint _S5 = t_bonds_0(&kernelContext_6);
    atomic_store_explicit((&kernelContext_6)->topo_0+(_S5 + i_0), 0U, memory_order_relaxed);
    uint _S6 = t_node_cursor_0(&kernelContext_6);
    atomic_store_explicit((&kernelContext_6)->topo_0+(_S6 + i_0), 0U, memory_order_relaxed);
    uint _S7 = t_bond_cursor_0(&kernelContext_6);
    atomic_store_explicit((&kernelContext_6)->topo_0+(_S7 + i_0), 0U, memory_order_relaxed);
    return;
}

bool is_dynamic_0(uint i_1, KernelContext_0 thread* kernelContext_7)
{
    return (kernelContext_7->nodes_0[i_1].w_lin_0) > 0.0f;
}

uint find_0(uint i_2, KernelContext_0 thread* kernelContext_8)
{
    uint _S8 = t_parent_0();
    uint _S9 = atomic_load_explicit(kernelContext_8->topo_0+(_S8 + i_2), memory_order_relaxed);
    uint p_0 = _S9;
    uint _S10 = i_2;
    for(;;)
    {
        if(p_0 != _S10)
        {
        }
        else
        {
            break;
        }
        uint _S11 = _S8 + p_0;
        uint g_0 = atomic_load_explicit(kernelContext_8->topo_0+_S11, memory_order_relaxed);
        if(g_0 != p_0)
        {
            uint _S12;
            {
            uint _metal_cas_comparand = p_0;
            atomic_compare_exchange_weak_explicit(kernelContext_8->topo_0+(_S8 + _S10), &_metal_cas_comparand, g_0, memory_order_relaxed, memory_order_relaxed);
            _S12 = _metal_cas_comparand;
            }
        }
        uint _S13 = atomic_load_explicit(kernelContext_8->topo_0+_S11, memory_order_relaxed);
        _S10 = p_0;
        p_0 = _S13;
    }
    return _S10;
}

uint t_total_0(KernelContext_0 thread* kernelContext_9)
{
    return topo_total_0(kernelContext_9->params_0->node_count_2);
}

[[kernel]] void bt_union(uint3 id_2 [[thread_position_in_grid]], BlastParams_0 constant* params_3 [[buffer(0)]], atomic<uint> device* topo_3 [[buffer(3)]], uint device* args_3 [[buffer(7)]], BlastBond_natural_0 device* bonds_3 [[buffer(2)]], BlastNode_natural_0 device* nodes_3 [[buffer(1)]], packed_uint4 device* islands_3 [[buffer(4)]], packed_uint4 device* report_3 [[buffer(6)]], uint device* lists_3 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_10;
    (&kernelContext_10)->params_0 = params_3;
    (&kernelContext_10)->topo_0 = topo_3;
    (&kernelContext_10)->args_0 = args_3;
    (&kernelContext_10)->bonds_0 = bonds_3;
    (&kernelContext_10)->nodes_0 = nodes_3;
    (&kernelContext_10)->islands_0 = islands_3;
    (&kernelContext_10)->report_0 = report_3;
    (&kernelContext_10)->lists_0 = lists_3;
    threadgroup array<uint3, int(1024)> g_scan_3;
    (&kernelContext_10)->g_scan_0 = &g_scan_3;
    uint j_0 = id_2.x;
    if(j_0 >= (params_3->bond_count_0))
    {
        return;
    }
    BlastBond_natural_0 b_0 = (&kernelContext_10)->bonds_0[j_0];
    bool _S14;
    if(!((b_0.health_0) > 0.0f))
    {
        _S14 = true;
    }
    else
    {
        bool _S15 = is_dynamic_0(b_0.n0_0, &kernelContext_10);
        _S14 = !_S15;
    }
    if(_S14)
    {
        _S14 = true;
    }
    else
    {
        bool _S16 = is_dynamic_0(b_0.n1_0, &kernelContext_10);
        _S14 = !_S16;
    }
    if(_S14)
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
        uint _S17 = find_0(a_0, &kernelContext_10);
        uint _S18 = find_0(c_0, &kernelContext_10);
        if(_S17 == _S18)
        {
            return;
        }
        uint _S19 = max(_S17, _S18);
        uint _S20 = min(_S17, _S18);
        uint _S21;
        {
        uint _metal_cas_comparand = _S19;
        atomic_compare_exchange_weak_explicit((&kernelContext_10)->topo_0+(t_parent_0() + _S19), &_metal_cas_comparand, _S20, memory_order_relaxed, memory_order_relaxed);
        _S21 = _metal_cas_comparand;
        }
        if(_S21 == _S19)
        {
            return;
        }
        uint _S22 = guard_0 + 1U;
        a_0 = _S19;
        c_0 = _S20;
        guard_0 = _S22;
    }
    uint _S23 = t_total_0(&kernelContext_10);
    atomic_store_explicit((&kernelContext_10)->topo_0+(_S23 + 3U), 1U, memory_order_relaxed);
    return;
}

uint t_label_0(KernelContext_0 thread* kernelContext_11)
{
    return kernelContext_11->params_0->node_count_2;
}

[[kernel]] void bt_flatten(uint3 id_3 [[thread_position_in_grid]], BlastParams_0 constant* params_4 [[buffer(0)]], atomic<uint> device* topo_4 [[buffer(3)]], uint device* args_4 [[buffer(7)]], BlastBond_natural_0 device* bonds_4 [[buffer(2)]], BlastNode_natural_0 device* nodes_4 [[buffer(1)]], packed_uint4 device* islands_4 [[buffer(4)]], packed_uint4 device* report_4 [[buffer(6)]], uint device* lists_4 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_12;
    (&kernelContext_12)->params_0 = params_4;
    (&kernelContext_12)->topo_0 = topo_4;
    (&kernelContext_12)->args_0 = args_4;
    (&kernelContext_12)->bonds_0 = bonds_4;
    (&kernelContext_12)->nodes_0 = nodes_4;
    (&kernelContext_12)->islands_0 = islands_4;
    (&kernelContext_12)->report_0 = report_4;
    (&kernelContext_12)->lists_0 = lists_4;
    threadgroup array<uint3, int(1024)> g_scan_4;
    (&kernelContext_12)->g_scan_0 = &g_scan_4;
    uint i_3 = id_3.x;
    if(i_3 >= (params_4->node_count_2))
    {
        return;
    }
    uint _S24 = t_label_0(&kernelContext_12);
    atomic<uint> device* _S25 = (&kernelContext_12)->topo_0+(_S24 + i_3);
    bool _S26 = is_dynamic_0(i_3, &kernelContext_12);
    uint _S27;
    if(_S26)
    {
        uint _S28 = find_0(i_3, &kernelContext_12);
        _S27 = _S28;
    }
    else
    {
        _S27 = 4294967295U;
    }
    atomic_store_explicit(_S25, _S27, memory_order_relaxed);
    return;
}

[[kernel]] void bt_count_nodes(uint3 id_4 [[thread_position_in_grid]], BlastParams_0 constant* params_5 [[buffer(0)]], atomic<uint> device* topo_5 [[buffer(3)]], uint device* args_5 [[buffer(7)]], BlastBond_natural_0 device* bonds_5 [[buffer(2)]], BlastNode_natural_0 device* nodes_5 [[buffer(1)]], packed_uint4 device* islands_5 [[buffer(4)]], packed_uint4 device* report_5 [[buffer(6)]], uint device* lists_5 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_13;
    (&kernelContext_13)->params_0 = params_5;
    (&kernelContext_13)->topo_0 = topo_5;
    (&kernelContext_13)->args_0 = args_5;
    (&kernelContext_13)->bonds_0 = bonds_5;
    (&kernelContext_13)->nodes_0 = nodes_5;
    (&kernelContext_13)->islands_0 = islands_5;
    (&kernelContext_13)->report_0 = report_5;
    (&kernelContext_13)->lists_0 = lists_5;
    threadgroup array<uint3, int(1024)> g_scan_5;
    (&kernelContext_13)->g_scan_0 = &g_scan_5;
    uint i_4 = id_4.x;
    bool _S29;
    if(i_4 >= (params_5->node_count_2))
    {
        _S29 = true;
    }
    else
    {
        bool _S30 = is_dynamic_0(i_4, &kernelContext_13);
        _S29 = !_S30;
    }
    if(_S29)
    {
        return;
    }
    uint _S31 = t_nodes_0(&kernelContext_13);
    uint _S32 = t_label_0(&kernelContext_13);
    uint _S33 = atomic_load_explicit((&kernelContext_13)->topo_0+(_S32 + i_4), memory_order_relaxed);
    uint _S34 = atomic_fetch_add_explicit((&kernelContext_13)->topo_0+(_S31 + _S33), 1U, memory_order_relaxed);
    return;
}

bool couples_0(const BlastBond_natural_0 thread* b_1, KernelContext_0 thread* kernelContext_14)
{
    bool _S35;
    if((b_1->health_0) > 0.0f)
    {
        bool _S36 = is_dynamic_0(b_1->n0_0, kernelContext_14);
        if(_S36)
        {
            _S35 = true;
        }
        else
        {
            bool _S37 = is_dynamic_0(b_1->n1_0, kernelContext_14);
            _S35 = _S37;
        }
    }
    else
    {
        _S35 = false;
    }
    return _S35;
}

uint bond_root_0(const BlastBond_natural_0 thread* b_2, KernelContext_0 thread* kernelContext_15)
{
    uint _S38 = t_label_0(kernelContext_15);
    uint _S39 = b_2->n0_0;
    bool _S40 = is_dynamic_0(b_2->n0_0, kernelContext_15);
    uint _S41;
    if(_S40)
    {
        _S41 = _S39;
    }
    else
    {
        _S41 = b_2->n1_0;
    }
    uint _S42 = atomic_load_explicit(kernelContext_15->topo_0+(_S38 + _S41), memory_order_relaxed);
    return _S42;
}

[[kernel]] void bt_count_bonds(uint3 id_5 [[thread_position_in_grid]], BlastParams_0 constant* params_6 [[buffer(0)]], atomic<uint> device* topo_6 [[buffer(3)]], uint device* args_6 [[buffer(7)]], BlastBond_natural_0 device* bonds_6 [[buffer(2)]], BlastNode_natural_0 device* nodes_6 [[buffer(1)]], packed_uint4 device* islands_6 [[buffer(4)]], packed_uint4 device* report_6 [[buffer(6)]], uint device* lists_6 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_16;
    (&kernelContext_16)->params_0 = params_6;
    (&kernelContext_16)->topo_0 = topo_6;
    (&kernelContext_16)->args_0 = args_6;
    (&kernelContext_16)->bonds_0 = bonds_6;
    (&kernelContext_16)->nodes_0 = nodes_6;
    (&kernelContext_16)->islands_0 = islands_6;
    (&kernelContext_16)->report_0 = report_6;
    (&kernelContext_16)->lists_0 = lists_6;
    threadgroup array<uint3, int(1024)> g_scan_6;
    (&kernelContext_16)->g_scan_0 = &g_scan_6;
    uint j_1 = id_5.x;
    if(j_1 >= (params_6->bond_count_0))
    {
        return;
    }
    thread BlastBond_natural_0 _S43 = (&kernelContext_16)->bonds_0[j_1];
    bool _S44 = couples_0(&_S43, &kernelContext_16);
    if(_S44)
    {
        uint _S45 = t_bonds_0(&kernelContext_16);
        uint _S46 = bond_root_0(&_S43, &kernelContext_16);
        uint _S47 = atomic_fetch_add_explicit((&kernelContext_16)->topo_0+(_S45 + _S46), 1U, memory_order_relaxed);
    }
    return;
}

uint3 scan_input_0(uint e_0, KernelContext_0 thread* kernelContext_17)
{
    bool _S48;
    if(e_0 >= (kernelContext_17->params_0->node_count_2))
    {
        _S48 = true;
    }
    else
    {
        uint _S49 = t_label_0(kernelContext_17);
        uint _S50 = atomic_load_explicit(kernelContext_17->topo_0+(_S49 + e_0), memory_order_relaxed);
        _S48 = _S50 != e_0;
    }
    if(_S48)
    {
        return uint3(int3(int(0)) );
    }
    uint _S51 = t_bonds_0(kernelContext_17);
    uint nb_0 = atomic_load_explicit(kernelContext_17->topo_0+(_S51 + e_0), memory_order_relaxed);
    uint3 _S52;
    if(nb_0 > 0U)
    {
        uint _S53 = t_nodes_0(kernelContext_17);
        uint _S54 = atomic_load_explicit(kernelContext_17->topo_0+(_S53 + e_0), memory_order_relaxed);
        _S52 = uint3(1U, _S54, nb_0);
    }
    else
    {
        _S52 = uint3(int3(int(0)) );
    }
    return _S52;
}

uint3 group_exclusive_scan_0(uint tid_0, uint3 v_0, uint3 thread* total_0, KernelContext_0 thread* kernelContext_18)
{
    (*kernelContext_18->g_scan_0)[tid_0] = v_0;
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
            add_0 = (*kernelContext_18->g_scan_0)[tid_0 - d_0];
        }
        else
        {
            add_0 = uint3(int3(int(0)) );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*kernelContext_18->g_scan_0)[tid_0] = (*kernelContext_18->g_scan_0)[tid_0] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        d_0 = d_0 << 1U;
    }
    *total_0 = (*kernelContext_18->g_scan_0)[1023U];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_18->g_scan_0)[tid_0] - v_0;
}

uint t_scan_0(KernelContext_0 thread* kernelContext_19)
{
    return 6U * kernelContext_19->params_0->node_count_2;
}

uint t_blocks_0(KernelContext_0 thread* kernelContext_20)
{
    return 10U * kernelContext_20->params_0->node_count_2;
}

[[kernel]] void bt_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_7 [[buffer(0)]], atomic<uint> device* topo_7 [[buffer(3)]], uint device* args_7 [[buffer(7)]], BlastBond_natural_0 device* bonds_7 [[buffer(2)]], BlastNode_natural_0 device* nodes_7 [[buffer(1)]], packed_uint4 device* islands_7 [[buffer(4)]], packed_uint4 device* report_7 [[buffer(6)]], uint device* lists_7 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_21;
    (&kernelContext_21)->params_0 = params_7;
    (&kernelContext_21)->topo_0 = topo_7;
    (&kernelContext_21)->args_0 = args_7;
    (&kernelContext_21)->bonds_0 = bonds_7;
    (&kernelContext_21)->nodes_0 = nodes_7;
    (&kernelContext_21)->islands_0 = islands_7;
    (&kernelContext_21)->report_0 = report_7;
    (&kernelContext_21)->lists_0 = lists_7;
    threadgroup array<uint3, int(1024)> g_scan_7;
    (&kernelContext_21)->g_scan_0 = &g_scan_7;
    uint tid_1 = thread_0.x;
    uint _S55 = group_0.x;
    uint e_1 = _S55 * 1024U + tid_1;
    uint3 _S56 = scan_input_0(e_1, &kernelContext_21);
    thread uint3 total_1;
    uint3 _S57 = group_exclusive_scan_0(tid_1, _S56, &total_1, &kernelContext_21);
    if(e_1 < ((&kernelContext_21)->params_0->node_count_2))
    {
        uint _S58 = t_scan_0(&kernelContext_21);
        uint _S59 = _S58 + 3U * e_1;
        atomic_store_explicit((&kernelContext_21)->topo_0+_S59, _S57.x, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_21)->topo_0+(_S59 + 1U), _S57.y, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_21)->topo_0+(_S59 + 2U), _S57.z, memory_order_relaxed);
    }
    if(tid_1 == 0U)
    {
        uint _S60 = t_blocks_0(&kernelContext_21);
        uint _S61 = _S60 + 3U * _S55;
        atomic_store_explicit((&kernelContext_21)->topo_0+_S61, total_1.x, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_21)->topo_0+(_S61 + 1U), total_1.y, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_21)->topo_0+(_S61 + 2U), total_1.z, memory_order_relaxed);
    }
    return;
}

[[kernel]] void bt_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_8 [[buffer(0)]], atomic<uint> device* topo_8 [[buffer(3)]], uint device* args_8 [[buffer(7)]], BlastBond_natural_0 device* bonds_8 [[buffer(2)]], BlastNode_natural_0 device* nodes_8 [[buffer(1)]], packed_uint4 device* islands_8 [[buffer(4)]], packed_uint4 device* report_8 [[buffer(6)]], uint device* lists_8 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_22;
    (&kernelContext_22)->params_0 = params_8;
    (&kernelContext_22)->topo_0 = topo_8;
    (&kernelContext_22)->args_0 = args_8;
    (&kernelContext_22)->bonds_0 = bonds_8;
    (&kernelContext_22)->nodes_0 = nodes_8;
    (&kernelContext_22)->islands_0 = islands_8;
    (&kernelContext_22)->report_0 = report_8;
    (&kernelContext_22)->lists_0 = lists_8;
    threadgroup array<uint3, int(1024)> g_scan_8;
    (&kernelContext_22)->g_scan_0 = &g_scan_8;
    uint tid_2 = thread_1.x;
    bool _S62 = tid_2 < ((params_8->node_count_2 + 1024U - 1U) / 1024U);
    uint3 v_1;
    if(_S62)
    {
        uint _S63 = t_blocks_0(&kernelContext_22);
        uint _S64 = _S63 + 3U * tid_2;
        uint _S65 = atomic_load_explicit((&kernelContext_22)->topo_0+_S64, memory_order_relaxed);
        uint _S66 = atomic_load_explicit((&kernelContext_22)->topo_0+(_S64 + 1U), memory_order_relaxed);
        uint _S67 = atomic_load_explicit((&kernelContext_22)->topo_0+(_S64 + 2U), memory_order_relaxed);
        v_1 = uint3(_S65, _S66, _S67);
    }
    else
    {
        v_1 = uint3(int3(int(0)) );
    }
    thread uint3 total_2;
    uint3 _S68 = group_exclusive_scan_0(tid_2, v_1, &total_2, &kernelContext_22);
    if(_S62)
    {
        uint _S69 = t_blocks_0(&kernelContext_22);
        uint _S70 = _S69 + 3U * tid_2;
        atomic_store_explicit((&kernelContext_22)->topo_0+_S70, _S68.x, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_22)->topo_0+(_S70 + 1U), _S68.y, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_22)->topo_0+(_S70 + 2U), _S68.z, memory_order_relaxed);
    }
    if(tid_2 == 0U)
    {
        uint _S71 = t_total_0(&kernelContext_22);
        atomic_store_explicit((&kernelContext_22)->topo_0+_S71, total_2.x, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_22)->topo_0+(_S71 + 1U), total_2.y, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_22)->topo_0+(_S71 + 2U), total_2.z, memory_order_relaxed);
        set_args_0(0U, total_2.x, &kernelContext_22);
        *((&kernelContext_22)->args_0+3U) = total_2.x;
        set_args_0(16U, total_2.x, &kernelContext_22);
    }
    return;
}

[[kernel]] void bt_scan_add(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_9 [[buffer(0)]], atomic<uint> device* topo_9 [[buffer(3)]], uint device* args_9 [[buffer(7)]], BlastBond_natural_0 device* bonds_9 [[buffer(2)]], BlastNode_natural_0 device* nodes_9 [[buffer(1)]], packed_uint4 device* islands_9 [[buffer(4)]], packed_uint4 device* report_9 [[buffer(6)]], uint device* lists_9 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_23;
    (&kernelContext_23)->params_0 = params_9;
    (&kernelContext_23)->topo_0 = topo_9;
    (&kernelContext_23)->args_0 = args_9;
    (&kernelContext_23)->bonds_0 = bonds_9;
    (&kernelContext_23)->nodes_0 = nodes_9;
    (&kernelContext_23)->islands_0 = islands_9;
    (&kernelContext_23)->report_0 = report_9;
    (&kernelContext_23)->lists_0 = lists_9;
    threadgroup array<uint3, int(1024)> g_scan_9;
    (&kernelContext_23)->g_scan_0 = &g_scan_9;
    uint _S72 = group_1.x;
    uint e_2 = _S72 * 1024U + thread_2.x;
    if(e_2 >= (params_9->node_count_2))
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
        uint _S73 = t_scan_0(&kernelContext_23);
        atomic<uint> device* _S74 = (&kernelContext_23)->topo_0+(_S73 + 3U * e_2 + k_0);
        uint _S75 = t_blocks_0(&kernelContext_23);
        uint _S76 = atomic_load_explicit((&kernelContext_23)->topo_0+(_S75 + 3U * _S72 + k_0), memory_order_relaxed);
        uint _S77 = atomic_fetch_add_explicit(_S74, _S76, memory_order_relaxed);
        k_0 = k_0 + 1U;
    }
    return;
}

uint t_island_of_0(KernelContext_0 thread* kernelContext_24)
{
    return 9U * kernelContext_24->params_0->node_count_2;
}

[[kernel]] void bt_islands(uint3 id_6 [[thread_position_in_grid]], BlastParams_0 constant* params_10 [[buffer(0)]], atomic<uint> device* topo_10 [[buffer(3)]], uint device* args_10 [[buffer(7)]], BlastBond_natural_0 device* bonds_10 [[buffer(2)]], BlastNode_natural_0 device* nodes_10 [[buffer(1)]], packed_uint4 device* islands_10 [[buffer(4)]], packed_uint4 device* report_10 [[buffer(6)]], uint device* lists_10 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_25;
    (&kernelContext_25)->params_0 = params_10;
    (&kernelContext_25)->topo_0 = topo_10;
    (&kernelContext_25)->args_0 = args_10;
    (&kernelContext_25)->bonds_0 = bonds_10;
    (&kernelContext_25)->nodes_0 = nodes_10;
    (&kernelContext_25)->islands_0 = islands_10;
    (&kernelContext_25)->report_0 = report_10;
    (&kernelContext_25)->lists_0 = lists_10;
    threadgroup array<uint3, int(1024)> g_scan_10;
    (&kernelContext_25)->g_scan_0 = &g_scan_10;
    uint e_3 = id_6.x;
    if(e_3 >= (params_10->node_count_2))
    {
        return;
    }
    uint3 _S78 = scan_input_0(e_3, &kernelContext_25);
    if((_S78.x) == 0U)
    {
        return;
    }
    uint _S79 = t_scan_0(&kernelContext_25);
    uint _S80 = _S79 + 3U * e_3;
    uint k_1 = atomic_load_explicit((&kernelContext_25)->topo_0+_S80, memory_order_relaxed);
    uint n0_1 = atomic_load_explicit((&kernelContext_25)->topo_0+(_S80 + 1U), memory_order_relaxed);
    uint _S81 = t_total_0(&kernelContext_25);
    uint _S82 = atomic_load_explicit((&kernelContext_25)->topo_0+(_S81 + 1U), memory_order_relaxed);
    uint _S83 = atomic_load_explicit((&kernelContext_25)->topo_0+(_S80 + 2U), memory_order_relaxed);
    uint b0_0 = _S82 + _S83;
    *((&kernelContext_25)->islands_0+k_1) = packed_uint4(uint4(n0_1, n0_1 + _S78.y, b0_0, b0_0 + _S78.z)) ;
    uint _S84 = t_island_of_0(&kernelContext_25);
    atomic_store_explicit((&kernelContext_25)->topo_0+(_S84 + e_3), k_1, memory_order_relaxed);
    *((&kernelContext_25)->report_0+k_1) = packed_uint4(uint4(int4(int(0)) )) ;
    return;
}

[[kernel]] void bt_scatter_nodes(uint3 id_7 [[thread_position_in_grid]], BlastParams_0 constant* params_11 [[buffer(0)]], atomic<uint> device* topo_11 [[buffer(3)]], uint device* args_11 [[buffer(7)]], BlastBond_natural_0 device* bonds_11 [[buffer(2)]], BlastNode_natural_0 device* nodes_11 [[buffer(1)]], packed_uint4 device* islands_11 [[buffer(4)]], packed_uint4 device* report_11 [[buffer(6)]], uint device* lists_11 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_26;
    (&kernelContext_26)->params_0 = params_11;
    (&kernelContext_26)->topo_0 = topo_11;
    (&kernelContext_26)->args_0 = args_11;
    (&kernelContext_26)->bonds_0 = bonds_11;
    (&kernelContext_26)->nodes_0 = nodes_11;
    (&kernelContext_26)->islands_0 = islands_11;
    (&kernelContext_26)->report_0 = report_11;
    (&kernelContext_26)->lists_0 = lists_11;
    threadgroup array<uint3, int(1024)> g_scan_11;
    (&kernelContext_26)->g_scan_0 = &g_scan_11;
    uint i_5 = id_7.x;
    bool _S85;
    if(i_5 >= (params_11->node_count_2))
    {
        _S85 = true;
    }
    else
    {
        bool _S86 = is_dynamic_0(i_5, &kernelContext_26);
        _S85 = !_S86;
    }
    if(_S85)
    {
        return;
    }
    uint _S87 = t_label_0(&kernelContext_26);
    uint r_0 = atomic_load_explicit((&kernelContext_26)->topo_0+(_S87 + i_5), memory_order_relaxed);
    uint _S88 = t_bonds_0(&kernelContext_26);
    uint _S89 = atomic_load_explicit((&kernelContext_26)->topo_0+(_S88 + r_0), memory_order_relaxed);
    if(_S89 == 0U)
    {
        return;
    }
    uint _S90 = t_island_of_0(&kernelContext_26);
    uint k_2 = atomic_load_explicit((&kernelContext_26)->topo_0+(_S90 + r_0), memory_order_relaxed);
    uint _S91 = (uint4(*((&kernelContext_26)->islands_0+k_2)) ).x;
    uint _S92 = t_node_cursor_0(&kernelContext_26);
    uint _S93 = atomic_fetch_add_explicit((&kernelContext_26)->topo_0+(_S92 + r_0), 1U, memory_order_relaxed);
    *((&kernelContext_26)->lists_0+(_S91 + _S93)) = i_5;
    return;
}

[[kernel]] void bt_scatter_bonds(uint3 id_8 [[thread_position_in_grid]], BlastParams_0 constant* params_12 [[buffer(0)]], atomic<uint> device* topo_12 [[buffer(3)]], uint device* args_12 [[buffer(7)]], BlastBond_natural_0 device* bonds_12 [[buffer(2)]], BlastNode_natural_0 device* nodes_12 [[buffer(1)]], packed_uint4 device* islands_12 [[buffer(4)]], packed_uint4 device* report_12 [[buffer(6)]], uint device* lists_12 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_27;
    (&kernelContext_27)->params_0 = params_12;
    (&kernelContext_27)->topo_0 = topo_12;
    (&kernelContext_27)->args_0 = args_12;
    (&kernelContext_27)->bonds_0 = bonds_12;
    (&kernelContext_27)->nodes_0 = nodes_12;
    (&kernelContext_27)->islands_0 = islands_12;
    (&kernelContext_27)->report_0 = report_12;
    (&kernelContext_27)->lists_0 = lists_12;
    threadgroup array<uint3, int(1024)> g_scan_12;
    (&kernelContext_27)->g_scan_0 = &g_scan_12;
    uint j_2 = id_8.x;
    if(j_2 >= (params_12->bond_count_0))
    {
        return;
    }
    thread BlastBond_natural_0 _S94 = (&kernelContext_27)->bonds_0[j_2];
    bool _S95 = couples_0(&_S94, &kernelContext_27);
    if(!_S95)
    {
        return;
    }
    uint _S96 = bond_root_0(&_S94, &kernelContext_27);
    uint _S97 = t_island_of_0(&kernelContext_27);
    uint k_3 = atomic_load_explicit((&kernelContext_27)->topo_0+(_S97 + _S96), memory_order_relaxed);
    uint _S98 = (uint4(*((&kernelContext_27)->islands_0+k_3)) ).z;
    uint _S99 = t_bond_cursor_0(&kernelContext_27);
    uint _S100 = atomic_fetch_add_explicit((&kernelContext_27)->topo_0+(_S99 + _S96), 1U, memory_order_relaxed);
    *((&kernelContext_27)->lists_0+(_S98 + _S100)) = j_2;
    return;
}

uint4 group_uniform4_0(uint4 value_0, uint thread_3)
{
    return value_0;
}

void sort_list_0(uint tid_3, uint begin_0, uint n_0, KernelContext_0 thread* kernelContext_28)
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
        uint _S101 = k_4 >> 1U;
        uint j_3 = _S101;
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
                if(j_3 == _S101)
                {
                    l_0 = t_0 ^ (k_4 - 1U);
                }
                else
                {
                    l_0 = t_0 ^ j_3;
                }
                bool _S102;
                if(l_0 > t_0)
                {
                    _S102 = l_0 < n_0;
                }
                else
                {
                    _S102 = false;
                }
                if(_S102)
                {
                    uint _S103 = begin_0 + t_0;
                    uint device* _S104 = kernelContext_28->lists_0+_S103;
                    uint a_1 = *_S104;
                    uint _S105 = begin_0 + l_0;
                    uint device* _S106 = kernelContext_28->lists_0+_S105;
                    uint b_3 = *_S106;
                    if((*_S104) > (*_S106))
                    {
                        *(kernelContext_28->lists_0+_S103) = b_3;
                        *(kernelContext_28->lists_0+_S105) = a_1;
                    }
                }
                t_0 = t_0 + 1024U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            j_3 = j_3 >> 1U;
        }
        k_4 = k_4 << 1U;
    }
    return;
}

[[kernel]] void bt_sort(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_13 [[buffer(0)]], atomic<uint> device* topo_13 [[buffer(3)]], uint device* args_13 [[buffer(7)]], BlastBond_natural_0 device* bonds_13 [[buffer(2)]], BlastNode_natural_0 device* nodes_13 [[buffer(1)]], packed_uint4 device* islands_13 [[buffer(4)]], packed_uint4 device* report_13 [[buffer(6)]], uint device* lists_13 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_29;
    (&kernelContext_29)->params_0 = params_13;
    (&kernelContext_29)->topo_0 = topo_13;
    (&kernelContext_29)->args_0 = args_13;
    (&kernelContext_29)->bonds_0 = bonds_13;
    (&kernelContext_29)->nodes_0 = nodes_13;
    (&kernelContext_29)->islands_0 = islands_13;
    (&kernelContext_29)->report_0 = report_13;
    (&kernelContext_29)->lists_0 = lists_13;
    threadgroup array<uint3, int(1024)> g_scan_13;
    (&kernelContext_29)->g_scan_0 = &g_scan_13;
    uint tid_4 = thread_4.x;
    uint4 _S107 = uint4(*(islands_13+group_2.x)) ;
    uint _S108 = _S107.x;
    uint _S109 = _S107.z;
    uint4 u_0 = group_uniform4_0(uint4(_S108, _S107.y - _S108, _S109, _S107.w - _S109), tid_4);
    sort_list_0(tid_4, u_0.x, u_0.y, &kernelContext_29);
    sort_list_0(tid_4, u_0.z, u_0.w, &kernelContext_29);
    return;
}

