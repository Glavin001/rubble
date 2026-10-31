#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
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
    float force_tolerance_0;
    float4 gravity_0;
    uint w_alive_0;
    uint a_dirty_0;
    uint a_counters_0;
    uint wide_min_0;
    uint wide_rows_0;
    uint wide_blocks_0;
    uint pad2_0;
};

struct BlastNode_natural_0
{
    float w_ang_0;
    float w_lin_0;
    uint ref_begin_0;
    uint ref_end_0;
    packed_float4 position_0;
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

struct KernelContext_0
{
    atomic<uint> device* meta_0;
    uint device* topo_args_0;
    packed_uint4 device* islands_0;
    BlastParams_0 constant* params_0;
    packed_float4 device* wide_0;
    uint device* lists_0;
    BlastNode_natural_0 device* nodes_0;
    packed_float4 device* accel_0;
    packed_float4 device* scratch_0;
    uint device* refs_0;
    BlastBond_natural_0 device* bonds_0;
    packed_float4 device* x_0;
    packed_uint4 device* report_0;
    array<float2, int(32)> threadgroup* g_blast_part_0;
    array<float, int(1024)> threadgroup* g_wide_max_0;
    float threadgroup* g_wide_max_out_0;
};

[[kernel]] void w_reset(uint3 id_0 [[thread_position_in_grid]], atomic<uint> device* meta_1 [[buffer(11)]], uint device* topo_args_1 [[buffer(12)]], packed_uint4 device* islands_1 [[buffer(1)]], BlastParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* wide_1 [[buffer(10)]], uint device* lists_1 [[buffer(9)]], BlastNode_natural_0 device* nodes_1 [[buffer(2)]], packed_float4 device* accel_1 [[buffer(5)]], packed_float4 device* scratch_1 [[buffer(7)]], uint device* refs_1 [[buffer(3)]], BlastBond_natural_0 device* bonds_1 [[buffer(4)]], packed_float4 device* x_1 [[buffer(6)]], packed_uint4 device* report_1 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->meta_0 = meta_1;
    (&kernelContext_0)->topo_args_0 = topo_args_1;
    (&kernelContext_0)->islands_0 = islands_1;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->wide_0 = wide_1;
    (&kernelContext_0)->lists_0 = lists_1;
    (&kernelContext_0)->nodes_0 = nodes_1;
    (&kernelContext_0)->accel_0 = accel_1;
    (&kernelContext_0)->scratch_0 = scratch_1;
    (&kernelContext_0)->refs_0 = refs_1;
    (&kernelContext_0)->bonds_0 = bonds_1;
    (&kernelContext_0)->x_0 = x_1;
    (&kernelContext_0)->report_0 = report_1;
    threadgroup array<float2, int(32)> g_blast_part_1;
    (&kernelContext_0)->g_blast_part_0 = &g_blast_part_1;
    threadgroup array<float, int(1024)> g_wide_max_1;
    (&kernelContext_0)->g_wide_max_0 = &g_wide_max_1;
    threadgroup float g_wide_max_out_1;
    (&kernelContext_0)->g_wide_max_out_0 = &g_wide_max_out_1;
    atomic_store_explicit(meta_1+0U, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_0)->meta_0+1U, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_0)->meta_0+2U, 0U, memory_order_relaxed);
    return;
}

uint blocks_0(uint n_0)
{
    return (n_0 + 1024U - 1U) / 1024U;
}

[[kernel]] void w_classify(uint3 id_1 [[thread_position_in_grid]], atomic<uint> device* meta_2 [[buffer(11)]], uint device* topo_args_2 [[buffer(12)]], packed_uint4 device* islands_2 [[buffer(1)]], BlastParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* wide_2 [[buffer(10)]], uint device* lists_2 [[buffer(9)]], BlastNode_natural_0 device* nodes_2 [[buffer(2)]], packed_float4 device* accel_2 [[buffer(5)]], packed_float4 device* scratch_2 [[buffer(7)]], uint device* refs_2 [[buffer(3)]], BlastBond_natural_0 device* bonds_2 [[buffer(4)]], packed_float4 device* x_2 [[buffer(6)]], packed_uint4 device* report_2 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->meta_0 = meta_2;
    (&kernelContext_1)->topo_args_0 = topo_args_2;
    (&kernelContext_1)->islands_0 = islands_2;
    (&kernelContext_1)->params_0 = params_2;
    (&kernelContext_1)->wide_0 = wide_2;
    (&kernelContext_1)->lists_0 = lists_2;
    (&kernelContext_1)->nodes_0 = nodes_2;
    (&kernelContext_1)->accel_0 = accel_2;
    (&kernelContext_1)->scratch_0 = scratch_2;
    (&kernelContext_1)->refs_0 = refs_2;
    (&kernelContext_1)->bonds_0 = bonds_2;
    (&kernelContext_1)->x_0 = x_2;
    (&kernelContext_1)->report_0 = report_2;
    threadgroup array<float2, int(32)> g_blast_part_2;
    (&kernelContext_1)->g_blast_part_0 = &g_blast_part_2;
    threadgroup array<float, int(1024)> g_wide_max_2;
    (&kernelContext_1)->g_wide_max_0 = &g_wide_max_2;
    threadgroup float g_wide_max_out_2;
    (&kernelContext_1)->g_wide_max_out_0 = &g_wide_max_out_2;
    uint k_0 = id_1.x;
    if(k_0 >= topo_args_2[int(3)])
    {
        return;
    }
    uint4 _S1 = uint4(*((&kernelContext_1)->islands_0+k_0)) ;
    uint _S2 = _S1.y - _S1.x;
    if(_S2 <= ((&kernelContext_1)->params_0->wide_min_0))
    {
        return;
    }
    uint row_0 = atomic_fetch_add_explicit((&kernelContext_1)->meta_0+0U, 1U, memory_order_relaxed);
    if(row_0 < ((&kernelContext_1)->params_0->wide_rows_0))
    {
        atomic_store_explicit((&kernelContext_1)->meta_0+(16U + row_0), k_0, memory_order_relaxed);
    }
    uint _S3 = atomic_fetch_max_explicit((&kernelContext_1)->meta_0+1U, blocks_0(_S2), memory_order_relaxed);
    uint _S4 = atomic_fetch_max_explicit((&kernelContext_1)->meta_0+2U, blocks_0(_S1.w - _S1.z), memory_order_relaxed);
    return;
}

[[kernel]] void w_args(uint3 id_2 [[thread_position_in_grid]], atomic<uint> device* meta_3 [[buffer(11)]], uint device* topo_args_3 [[buffer(12)]], packed_uint4 device* islands_3 [[buffer(1)]], BlastParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* wide_3 [[buffer(10)]], uint device* lists_3 [[buffer(9)]], BlastNode_natural_0 device* nodes_3 [[buffer(2)]], packed_float4 device* accel_3 [[buffer(5)]], packed_float4 device* scratch_3 [[buffer(7)]], uint device* refs_3 [[buffer(3)]], BlastBond_natural_0 device* bonds_3 [[buffer(4)]], packed_float4 device* x_3 [[buffer(6)]], packed_uint4 device* report_3 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->meta_0 = meta_3;
    (&kernelContext_2)->topo_args_0 = topo_args_3;
    (&kernelContext_2)->islands_0 = islands_3;
    (&kernelContext_2)->params_0 = params_3;
    (&kernelContext_2)->wide_0 = wide_3;
    (&kernelContext_2)->lists_0 = lists_3;
    (&kernelContext_2)->nodes_0 = nodes_3;
    (&kernelContext_2)->accel_0 = accel_3;
    (&kernelContext_2)->scratch_0 = scratch_3;
    (&kernelContext_2)->refs_0 = refs_3;
    (&kernelContext_2)->bonds_0 = bonds_3;
    (&kernelContext_2)->x_0 = x_3;
    (&kernelContext_2)->report_0 = report_3;
    threadgroup array<float2, int(32)> g_blast_part_3;
    (&kernelContext_2)->g_blast_part_0 = &g_blast_part_3;
    threadgroup array<float, int(1024)> g_wide_max_3;
    (&kernelContext_2)->g_wide_max_0 = &g_wide_max_3;
    threadgroup float g_wide_max_out_3;
    (&kernelContext_2)->g_wide_max_out_0 = &g_wide_max_out_3;
    uint _S5 = atomic_load_explicit(meta_3+0U, memory_order_relaxed);
    uint _S6 = min(_S5, (&kernelContext_2)->params_0->wide_rows_0);
    bool _S7 = _S6 > 0U;
    uint nb_0;
    if(_S7)
    {
        uint _S8 = atomic_load_explicit((&kernelContext_2)->meta_0+1U, memory_order_relaxed);
        nb_0 = _S8;
    }
    else
    {
        nb_0 = 0U;
    }
    uint bb_0;
    if(_S7)
    {
        uint _S9 = atomic_load_explicit((&kernelContext_2)->meta_0+2U, memory_order_relaxed);
        bb_0 = _S9;
    }
    else
    {
        bb_0 = 0U;
    }
    atomic_store_explicit((&kernelContext_2)->meta_0+4U, nb_0, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_2)->meta_0+5U, _S6, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_2)->meta_0+6U, 1U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_2)->meta_0+8U, bb_0, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_2)->meta_0+9U, _S6, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_2)->meta_0+10U, 1U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_2)->meta_0+12U, _S6, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_2)->meta_0+13U, 1U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_2)->meta_0+14U, 1U, memory_order_relaxed);
    return;
}

uint row_island_0(uint row_1, KernelContext_0 thread* kernelContext_3)
{
    uint _S10 = atomic_load_explicit(kernelContext_3->meta_0+(16U + row_1), memory_order_relaxed);
    return _S10;
}

uint st_0(uint row_2, uint slot_0)
{
    return row_2 * 16U + slot_0;
}

uint flags_of_0(uint row_3, KernelContext_0 thread* kernelContext_4)
{
    return (as_type<uint>(((float4(*(kernelContext_4->wide_0+st_0(row_3, 2U))) ).w)));
}

uint4 group_uniform4_0(uint4 value_0, uint thread_0)
{
    return value_0;
}

struct Block_0
{
    uint row_4;
    uint island_0;
    uint begin_0;
    uint end_0;
    uint flags_0;
};

Block_0 node_block_0(uint3 group_0, uint tid_0, KernelContext_0 thread* kernelContext_5)
{
    thread Block_0 b_0;
    uint _S11 = group_0.y;
    (&b_0)->row_4 = _S11;
    uint _S12 = row_island_0(_S11, kernelContext_5);
    uint _S13 = flags_of_0(_S11, kernelContext_5);
    uint4 u_0 = group_uniform4_0(uint4(_S12, 0U, 0U, _S13), tid_0);
    uint _S14 = u_0.x;
    (&b_0)->island_0 = _S14;
    (&b_0)->flags_0 = u_0.w;
    uint4 g_0 = group_uniform4_0(uint4(*(kernelContext_5->islands_0+_S14)) , tid_0);
    uint _S15 = g_0.x + group_0.x * 1024U;
    (&b_0)->begin_0 = _S15;
    (&b_0)->end_0 = min(_S15 + 1024U, g_0.y);
    return b_0;
}

uint ns_0(uint n_1, uint v_0)
{
    return n_1 * 24U + v_0;
}

float norm2_0(float3 a_0, float3 b_1)
{
    return dot(a_0, a_0) + dot(b_1, b_1);
}

void set_v_0(uint n_2, uint v_1, float3 a_1, float3 l_0, KernelContext_0 thread* kernelContext_6)
{
    uint _S16 = ns_0(n_2, v_1);
    *(kernelContext_6->scratch_0+_S16) = packed_float4(float4(a_1, 0.0f)) ;
    *(kernelContext_6->scratch_0+(_S16 + 1U)) = packed_float4(float4(l_0, 0.0f)) ;
    return;
}

bool build_block_0(uint node_0, KernelContext_0 thread* kernelContext_7)
{
    BlastNode_natural_0 n_3 = kernelContext_7->nodes_0[node_0];
    float3 _S17 = float3(0.0f) ;
    uint r_0 = n_3.ref_begin_0;
    float3 diag_0 = _S17;
    float3 offd_0 = _S17;
    float3 ks_0 = _S17;
    float cs_0 = 0.0f;
    for(;;)
    {
        if(r_0 < (n_3.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_0 = kernelContext_7->refs_0[r_0];
        BlastBond_natural_0 device* _S18 = kernelContext_7->bonds_0+(ref_0 & 2147483647U);
        BlastBond_natural_0 b_2 = *_S18;
        if(!(((*_S18).health_0) > 0.0f))
        {
            r_0 = r_0 + 1U;
            continue;
        }
        float3 o_0;
        if((ref_0 & 2147483648U) != 0U)
        {
            o_0 = (float4(b_2.o1_0) ).xyz;
        }
        else
        {
            o_0 = (float4(b_2.o0_0) ).xyz;
        }
        float _S19 = b_2.s_0;
        float s2_0 = _S19 * _S19;
        float3 _S20 = float3(s2_0) ;
        float _S21 = o_0.x;
        float _S22 = o_0.y;
        float _S23 = o_0.z;
        float3 offd_1 = offd_0 + _S20 * - float3(_S21 * _S22, _S21 * _S23, _S22 * _S23);
        float3 ks_1 = ks_0 + _S20 * o_0;
        float cs_1 = cs_0 + s2_0;
        diag_0 = diag_0 + _S20 * (float3((1.0f + dot(o_0, o_0)))  - o_0 * o_0);
        offd_0 = offd_1;
        ks_0 = ks_1;
        cs_0 = cs_1;
        r_0 = r_0 + 1U;
    }
    float _S24 = n_3.w_ang_0;
    float da_0 = _S24 * _S24;
    float c_0 = n_3.w_lin_0 * n_3.w_lin_0 * cs_0;
    float3 k_1 = ks_0 * float3((n_3.w_ang_0 * n_3.w_lin_0)) ;
    float4 a0_0 = float4(diag_0 * float3(da_0) , offd_0.x * da_0);
    float _S25 = offd_0.y * da_0;
    float _S26 = offd_0.z * da_0;
    float4 a1_0 = float4(_S25, _S26, c_0, 0.0f);
    uint _S27 = ns_0(node_0, 18U);
    *(kernelContext_7->scratch_0+_S27) = packed_float4(a0_0) ;
    *(kernelContext_7->scratch_0+(_S27 + 1U)) = packed_float4(a1_0) ;
    *(kernelContext_7->scratch_0+(_S27 + 2U)) = packed_float4(float4(k_1, 0.0f)) ;
    if(!(c_0 > 0.0f))
    {
        return false;
    }
    float ic_0 = 1.0f / c_0;
    float kk_0 = dot(k_1, k_1);
    float _S28 = k_1.x;
    float sxx_0 = a0_0.x + (_S28 * _S28 - kk_0) * ic_0;
    float _S29 = k_1.y;
    float syy_0 = a0_0.y + (_S29 * _S29 - kk_0) * ic_0;
    float _S30 = k_1.z;
    float szz_0 = a0_0.z + (_S30 * _S30 - kk_0) * ic_0;
    float sxy_0 = a0_0.w + _S28 * _S29 * ic_0;
    float sxz_0 = _S25 + _S28 * _S30 * ic_0;
    float syz_0 = _S26 + _S29 * _S30 * ic_0;
    float cxx_0 = syy_0 * szz_0 - syz_0 * syz_0;
    float cxy_0 = sxz_0 * syz_0 - sxy_0 * szz_0;
    float cxz_0 = sxy_0 * syz_0 - sxz_0 * syy_0;
    float det_0 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    bool _S31;
    if(!(det_0 > 0.0f))
    {
        _S31 = true;
    }
    else
    {
        _S31 = !isfinite(det_0);
    }
    if(_S31)
    {
        return false;
    }
    float id_3 = 1.0f / det_0;
    float cyz_0 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    uint _S32 = ns_0(node_0, 21U);
    *(kernelContext_7->scratch_0+_S32) = packed_float4(float4(cxx_0 * id_3, (sxx_0 * szz_0 - sxz_0 * sxz_0) * id_3, (sxx_0 * syy_0 - sxy_0 * sxy_0) * id_3, cxy_0 * id_3)) ;
    *(kernelContext_7->scratch_0+(_S32 + 1U)) = packed_float4(float4(cxz_0 * id_3, cyz_0 * id_3, ic_0, 0.0f)) ;
    *(kernelContext_7->scratch_0+(_S32 + 2U)) = packed_float4(float4(k_1 * float3(ic_0) , 0.0f)) ;
    return true;
}

float2 blast_group_sum_0(uint tid_1, float2 v_2, KernelContext_0 thread* kernelContext_8)
{
    float2 w_0 = simd_sum(v_2);
    if((tid_1 % 32U) == 0U)
    {
        (*kernelContext_8->g_blast_part_0)[tid_1 / 32U] = w_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 _S33 = (*kernelContext_8->g_blast_part_0)[int(0)];
    uint k_2 = 1U;
    float2 s_1 = _S33;
    for(;;)
    {
        if(k_2 < 32U)
        {
        }
        else
        {
            break;
        }
        float2 s_2 = s_1 + (*kernelContext_8->g_blast_part_0)[k_2];
        k_2 = k_2 + 1U;
        s_1 = s_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s_1;
}

uint pt_0(uint row_5, uint block_0, uint k_3, KernelContext_0 thread* kernelContext_9)
{
    return kernelContext_9->params_0->wide_rows_0 * 16U + (row_5 * kernelContext_9->params_0->wide_blocks_0 + block_0) * 6U + k_3;
}

void write_partial_0(uint tid_2, uint row_6, uint block_1, uint k_4, float4 v_3, KernelContext_0 thread* kernelContext_10)
{
    float2 _S34 = blast_group_sum_0(tid_2, v_3.xy, kernelContext_10);
    float2 _S35 = blast_group_sum_0(tid_2, v_3.zw, kernelContext_10);
    if(tid_2 == 0U)
    {
        uint _S36 = pt_0(row_6, block_1, k_4, kernelContext_10);
        *(kernelContext_10->wide_0+_S36) = packed_float4(float4(_S34, _S35)) ;
    }
    return;
}

[[kernel]] void w_init_nodes(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], atomic<uint> device* meta_4 [[buffer(11)]], uint device* topo_args_4 [[buffer(12)]], packed_uint4 device* islands_4 [[buffer(1)]], BlastParams_0 constant* params_4 [[buffer(0)]], packed_float4 device* wide_4 [[buffer(10)]], uint device* lists_4 [[buffer(9)]], BlastNode_natural_0 device* nodes_4 [[buffer(2)]], packed_float4 device* accel_4 [[buffer(5)]], packed_float4 device* scratch_4 [[buffer(7)]], uint device* refs_4 [[buffer(3)]], BlastBond_natural_0 device* bonds_4 [[buffer(4)]], packed_float4 device* x_4 [[buffer(6)]], packed_uint4 device* report_4 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_11;
    (&kernelContext_11)->meta_0 = meta_4;
    (&kernelContext_11)->topo_args_0 = topo_args_4;
    (&kernelContext_11)->islands_0 = islands_4;
    (&kernelContext_11)->params_0 = params_4;
    (&kernelContext_11)->wide_0 = wide_4;
    (&kernelContext_11)->lists_0 = lists_4;
    (&kernelContext_11)->nodes_0 = nodes_4;
    (&kernelContext_11)->accel_0 = accel_4;
    (&kernelContext_11)->scratch_0 = scratch_4;
    (&kernelContext_11)->refs_0 = refs_4;
    (&kernelContext_11)->bonds_0 = bonds_4;
    (&kernelContext_11)->x_0 = x_4;
    (&kernelContext_11)->report_0 = report_4;
    threadgroup array<float2, int(32)> g_blast_part_4;
    (&kernelContext_11)->g_blast_part_0 = &g_blast_part_4;
    threadgroup array<float, int(1024)> g_wide_max_4;
    (&kernelContext_11)->g_wide_max_0 = &g_wide_max_4;
    threadgroup float g_wide_max_out_4;
    (&kernelContext_11)->g_wide_max_out_0 = &g_wide_max_out_4;
    uint tid_3 = thread_1.x;
    Block_0 _S37 = node_block_0(group_1, tid_3, &kernelContext_11);
    if((_S37.begin_0) >= (_S37.end_0))
    {
        return;
    }
    float4 _S38 = float4(0.0f) ;
    thread float4 p0_0 = _S38;
    uint p_0 = _S37.begin_0 + tid_3;
    float4 p1_0;
    if(p_0 < (_S37.end_0))
    {
        uint i_0 = (&kernelContext_11)->lists_0[p_0];
        BlastNode_natural_0 n_4 = (&kernelContext_11)->nodes_0[i_0];
        uint _S39 = i_0 * 2U;
        float4 _S40 = float4(*((&kernelContext_11)->accel_0+_S39)) ;
        float4 _S41 = float4(*((&kernelContext_11)->accel_0+(_S39 + 1U))) ;
        uint _S42 = ns_0(i_0, 16U);
        bool _S43;
        if(any(_S40 != (float4(*((&kernelContext_11)->scratch_0+_S42)) )))
        {
            _S43 = true;
        }
        else
        {
            _S43 = any(_S41 != (float4(*((&kernelContext_11)->scratch_0+(_S42 + 1U))) ));
        }
        if(_S43)
        {
            p0_0.z = 1.0f;
        }
        float3 _S44 = - _S40.xyz;
        float _S45;
        if((n_4.w_ang_0) > 0.0f)
        {
            _S45 = 1.0f / n_4.w_ang_0;
        }
        else
        {
            _S45 = 1.0f;
        }
        float3 ba_0 = _S44 * float3(_S45) ;
        float3 _S46 = - _S41.xyz;
        if((n_4.w_lin_0) > 0.0f)
        {
            _S45 = (&kernelContext_11)->params_0->recip_length_0 / n_4.w_lin_0;
        }
        else
        {
            _S45 = (&kernelContext_11)->params_0->recip_length_0;
        }
        float3 bl_0 = _S46 * float3(_S45) ;
        p0_0.x = norm2_0(ba_0, bl_0);
        set_v_0(i_0, 14U, ba_0, bl_0, &kernelContext_11);
        set_v_0(i_0, 0U, ba_0, bl_0, &kernelContext_11);
        *((&kernelContext_11)->scratch_0+_S42) = packed_float4(_S40) ;
        *((&kernelContext_11)->scratch_0+(_S42 + 1U)) = packed_float4(_S41) ;
        float3 _S47 = float3(0.0f) ;
        set_v_0(i_0, 6U, _S47, _S47, &kernelContext_11);
        set_v_0(i_0, 2U, _S47, _S47, &kernelContext_11);
        bool _S48 = build_block_0(i_0, &kernelContext_11);
        if(!_S48)
        {
            p0_0.y = 1.0f;
        }
        float w_1 = 1.0f / (n_4.w_lin_0 * n_4.w_lin_0);
        p1_0 = float4((float4(n_4.position_0) ).xyz * float3(w_1) , w_1);
    }
    else
    {
        p1_0 = _S38;
    }
    uint _S49 = group_1.x;
    write_partial_0(tid_3, _S37.row_4, _S49, 0U, p0_0, &kernelContext_11);
    write_partial_0(tid_3, _S37.row_4, _S49, 1U, p1_0, &kernelContext_11);
    return;
}

Block_0 bond_block_0(uint3 group_2, uint tid_4, KernelContext_0 thread* kernelContext_12)
{
    Block_0 _S50 = node_block_0(group_2, tid_4, kernelContext_12);
    thread Block_0 b_3 = _S50;
    uint4 g_1 = group_uniform4_0(uint4(*(kernelContext_12->islands_0+(&b_3)->island_0)) , tid_4);
    uint _S51 = g_1.z + group_2.x * 1024U;
    (&b_3)->begin_0 = _S51;
    (&b_3)->end_0 = min(_S51 + 1024U, g_1.w);
    return b_3;
}

[[kernel]] void w_init_bonds(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], atomic<uint> device* meta_5 [[buffer(11)]], uint device* topo_args_5 [[buffer(12)]], packed_uint4 device* islands_5 [[buffer(1)]], BlastParams_0 constant* params_5 [[buffer(0)]], packed_float4 device* wide_5 [[buffer(10)]], uint device* lists_5 [[buffer(9)]], BlastNode_natural_0 device* nodes_5 [[buffer(2)]], packed_float4 device* accel_5 [[buffer(5)]], packed_float4 device* scratch_5 [[buffer(7)]], uint device* refs_5 [[buffer(3)]], BlastBond_natural_0 device* bonds_5 [[buffer(4)]], packed_float4 device* x_5 [[buffer(6)]], packed_uint4 device* report_5 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_13;
    (&kernelContext_13)->meta_0 = meta_5;
    (&kernelContext_13)->topo_args_0 = topo_args_5;
    (&kernelContext_13)->islands_0 = islands_5;
    (&kernelContext_13)->params_0 = params_5;
    (&kernelContext_13)->wide_0 = wide_5;
    (&kernelContext_13)->lists_0 = lists_5;
    (&kernelContext_13)->nodes_0 = nodes_5;
    (&kernelContext_13)->accel_0 = accel_5;
    (&kernelContext_13)->scratch_0 = scratch_5;
    (&kernelContext_13)->refs_0 = refs_5;
    (&kernelContext_13)->bonds_0 = bonds_5;
    (&kernelContext_13)->x_0 = x_5;
    (&kernelContext_13)->report_0 = report_5;
    threadgroup array<float2, int(32)> g_blast_part_5;
    (&kernelContext_13)->g_blast_part_0 = &g_blast_part_5;
    threadgroup array<float, int(1024)> g_wide_max_5;
    (&kernelContext_13)->g_wide_max_0 = &g_wide_max_5;
    threadgroup float g_wide_max_out_5;
    (&kernelContext_13)->g_wide_max_out_0 = &g_wide_max_out_5;
    uint tid_5 = thread_2.x;
    Block_0 _S52 = bond_block_0(group_3, tid_5, &kernelContext_13);
    if((_S52.begin_0) >= (_S52.end_0))
    {
        return;
    }
    uint p_1 = _S52.begin_0 + tid_5;
    float anchor_0;
    if(p_1 < (_S52.end_0))
    {
        uint j_0 = (&kernelContext_13)->lists_0[p_1];
        BlastBond_natural_0 device* _S53 = (&kernelContext_13)->bonds_0+j_0;
        BlastBond_natural_0 bd_0 = *_S53;
        bool _S54;
        if(((*_S53).health_0) > 0.0f)
        {
            if(!(((&kernelContext_13)->nodes_0[bd_0.n0_0].w_lin_0) > 0.0f))
            {
                _S54 = true;
            }
            else
            {
                _S54 = !(((&kernelContext_13)->nodes_0[bd_0.n1_0].w_lin_0) > 0.0f);
            }
        }
        else
        {
            _S54 = false;
        }
        if(_S54)
        {
            anchor_0 = 1.0f;
        }
        else
        {
            anchor_0 = 0.0f;
        }
        if(((&kernelContext_13)->params_0->warm_0) == 0U)
        {
            uint _S55 = j_0 * 2U;
            packed_float4 _S56 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_13)->x_0+_S55) = _S56;
            *((&kernelContext_13)->x_0+(_S55 + 1U)) = _S56;
        }
    }
    else
    {
        anchor_0 = 0.0f;
    }
    write_partial_0(tid_5, _S52.row_4, group_3.x, 2U, float4(anchor_0, 0.0f, 0.0f, 0.0f), &kernelContext_13);
    return;
}

uint group_uniform_0(uint value_1, uint thread_3)
{
    return value_1;
}

float4 row_sum_0(uint tid_6, uint row_7, uint count_0, uint k_5, KernelContext_0 thread* kernelContext_14)
{
    float4 _S57 = float4(0.0f) ;
    uint b_4 = tid_6;
    float4 s_3 = _S57;
    for(;;)
    {
        if(b_4 < count_0)
        {
        }
        else
        {
            break;
        }
        uint _S58 = pt_0(row_7, b_4, k_5, kernelContext_14);
        float4 s_4 = s_3 + float4(*(kernelContext_14->wide_0+_S58)) ;
        b_4 = b_4 + 1024U;
        s_3 = s_4;
    }
    float2 _S59 = blast_group_sum_0(tid_6, s_3.xy, kernelContext_14);
    float2 _S60 = blast_group_sum_0(tid_6, s_3.zw, kernelContext_14);
    return float4(_S59, _S60);
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

[[kernel]] void w_setup(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], atomic<uint> device* meta_6 [[buffer(11)]], uint device* topo_args_6 [[buffer(12)]], packed_uint4 device* islands_6 [[buffer(1)]], BlastParams_0 constant* params_6 [[buffer(0)]], packed_float4 device* wide_6 [[buffer(10)]], uint device* lists_6 [[buffer(9)]], BlastNode_natural_0 device* nodes_6 [[buffer(2)]], packed_float4 device* accel_6 [[buffer(5)]], packed_float4 device* scratch_6 [[buffer(7)]], uint device* refs_6 [[buffer(3)]], BlastBond_natural_0 device* bonds_6 [[buffer(4)]], packed_float4 device* x_6 [[buffer(6)]], packed_uint4 device* report_6 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_15;
    (&kernelContext_15)->meta_0 = meta_6;
    (&kernelContext_15)->topo_args_0 = topo_args_6;
    (&kernelContext_15)->islands_0 = islands_6;
    (&kernelContext_15)->params_0 = params_6;
    (&kernelContext_15)->wide_0 = wide_6;
    (&kernelContext_15)->lists_0 = lists_6;
    (&kernelContext_15)->nodes_0 = nodes_6;
    (&kernelContext_15)->accel_0 = accel_6;
    (&kernelContext_15)->scratch_0 = scratch_6;
    (&kernelContext_15)->refs_0 = refs_6;
    (&kernelContext_15)->bonds_0 = bonds_6;
    (&kernelContext_15)->x_0 = x_6;
    (&kernelContext_15)->report_0 = report_6;
    threadgroup array<float2, int(32)> g_blast_part_6;
    (&kernelContext_15)->g_blast_part_0 = &g_blast_part_6;
    threadgroup array<float, int(1024)> g_wide_max_6;
    (&kernelContext_15)->g_wide_max_0 = &g_wide_max_6;
    threadgroup float g_wide_max_out_6;
    (&kernelContext_15)->g_wide_max_out_0 = &g_wide_max_out_6;
    uint tid_7 = thread_4.x;
    uint row_8 = group_4.x;
    uint _S61 = row_island_0(row_8, &kernelContext_15);
    uint island_1 = group_uniform_0(_S61, tid_7);
    uint4 r_1 = group_uniform4_0(uint4(*((&kernelContext_15)->islands_0+island_1)) , tid_7);
    uint nb_1 = blocks_0(r_1.y - r_1.x);
    uint bb_1 = blocks_0(r_1.w - r_1.z);
    float4 _S62 = row_sum_0(tid_7, row_8, nb_1, 0U, &kernelContext_15);
    float4 _S63 = row_sum_0(tid_7, row_8, nb_1, 1U, &kernelContext_15);
    float4 _S64 = row_sum_0(tid_7, row_8, bb_1, 2U, &kernelContext_15);
    if(tid_7 != 0U)
    {
        return;
    }
    uint4 _S65 = uint4(*((&kernelContext_15)->report_0+island_1)) ;
    bool _S66;
    if(((&kernelContext_15)->params_0->warm_0) != 0U)
    {
        _S66 = ((_S65.y) & 4U) != 0U;
    }
    else
    {
        _S66 = false;
    }
    if(_S66)
    {
        _S66 = (_S62.z) == 0.0f;
    }
    else
    {
        _S66 = false;
    }
    uint f_0;
    if(_S66)
    {
        f_0 = 128U;
    }
    else
    {
        f_0 = 1U;
    }
    float delta2_0 = _S62.x * (&kernelContext_15)->params_0->tolerance_0 * (&kernelContext_15)->params_0->tolerance_0;
    if(f_0 == 1U)
    {
        if((_S62.y) > 0.0f)
        {
            f_0 = 4U;
        }
        else
        {
            if(!(delta2_0 > 0.0f))
            {
                f_0 = 18U;
            }
        }
    }
    if((_S64.x) == 0.0f)
    {
        f_0 = f_0 | 8U;
    }
    float _S67 = float_bits_0(2139095040U);
    *((&kernelContext_15)->wide_0+st_0(row_8, 0U)) = packed_float4(float4(delta2_0, 0.0f, 0.0f, _S67)) ;
    *((&kernelContext_15)->wide_0+st_0(row_8, 1U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, _S67)) ;
    float _S68 = (as_type<float>((0U)));
    *((&kernelContext_15)->wide_0+st_0(row_8, 2U)) = packed_float4(float4(0.0f, 0.0f, _S68, (as_type<float>((f_0))))) ;
    packed_float4 device* _S69 = (&kernelContext_15)->wide_0+st_0(row_8, 3U);
    float3 _S70;
    if((_S63.y) > 0.0f)
    {
        _S70 = _S63.xyz / float3(_S63.w) ;
    }
    else
    {
        _S70 = float3(0.0f) ;
    }
    *_S69 = packed_float4(float4(_S70, _S68)) ;
    return;
}

float3 wide_rel_0(uint row_9, uint n_5, KernelContext_0 thread* kernelContext_16)
{
    return (float4(kernelContext_16->nodes_0[n_5].position_0) ).xyz - (float4(*(kernelContext_16->wide_0+st_0(row_9, 3U))) ).xyz;
}

[[kernel]] void w_gram_nodes(uint3 group_5 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], atomic<uint> device* meta_7 [[buffer(11)]], uint device* topo_args_7 [[buffer(12)]], packed_uint4 device* islands_7 [[buffer(1)]], BlastParams_0 constant* params_7 [[buffer(0)]], packed_float4 device* wide_7 [[buffer(10)]], uint device* lists_7 [[buffer(9)]], BlastNode_natural_0 device* nodes_7 [[buffer(2)]], packed_float4 device* accel_7 [[buffer(5)]], packed_float4 device* scratch_7 [[buffer(7)]], uint device* refs_7 [[buffer(3)]], BlastBond_natural_0 device* bonds_7 [[buffer(4)]], packed_float4 device* x_7 [[buffer(6)]], packed_uint4 device* report_7 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_17;
    (&kernelContext_17)->meta_0 = meta_7;
    (&kernelContext_17)->topo_args_0 = topo_args_7;
    (&kernelContext_17)->islands_0 = islands_7;
    (&kernelContext_17)->params_0 = params_7;
    (&kernelContext_17)->wide_0 = wide_7;
    (&kernelContext_17)->lists_0 = lists_7;
    (&kernelContext_17)->nodes_0 = nodes_7;
    (&kernelContext_17)->accel_0 = accel_7;
    (&kernelContext_17)->scratch_0 = scratch_7;
    (&kernelContext_17)->refs_0 = refs_7;
    (&kernelContext_17)->bonds_0 = bonds_7;
    (&kernelContext_17)->x_0 = x_7;
    (&kernelContext_17)->report_0 = report_7;
    threadgroup array<float2, int(32)> g_blast_part_7;
    (&kernelContext_17)->g_blast_part_0 = &g_blast_part_7;
    threadgroup array<float, int(1024)> g_wide_max_7;
    (&kernelContext_17)->g_wide_max_0 = &g_wide_max_7;
    threadgroup float g_wide_max_out_7;
    (&kernelContext_17)->g_wide_max_out_0 = &g_wide_max_out_7;
    uint tid_8 = thread_5.x;
    Block_0 _S71 = node_block_0(group_5, tid_8, &kernelContext_17);
    bool _S72;
    if((_S71.begin_0) >= (_S71.end_0))
    {
        _S72 = true;
    }
    else
    {
        _S72 = ((_S71.flags_0) & 8U) == 0U;
    }
    if(_S72)
    {
        _S72 = true;
    }
    else
    {
        _S72 = ((_S71.flags_0) & 1U) == 0U;
    }
    if(_S72)
    {
        return;
    }
    thread array<float, int(6)> rr_0;
    thread array<float3, int(3)> rt_0;
    uint z_0 = 0U;
    for(;;)
    {
        if(z_0 < 6U)
        {
        }
        else
        {
            break;
        }
        rr_0[z_0] = 0.0f;
        z_0 = z_0 + 1U;
    }
    z_0 = 0U;
    for(;;)
    {
        if(z_0 < 3U)
        {
        }
        else
        {
            break;
        }
        rt_0[z_0] = float3(0.0f) ;
        z_0 = z_0 + 1U;
    }
    uint p_2 = _S71.begin_0 + tid_8;
    float tt_0;
    if(p_2 < (_S71.end_0))
    {
        uint i_1 = (&kernelContext_17)->lists_0[p_2];
        BlastNode_natural_0 n_6 = (&kernelContext_17)->nodes_0[i_1];
        float _S73 = n_6.w_ang_0;
        float wa_0 = 1.0f / (_S73 * _S73);
        float _S74 = n_6.w_lin_0;
        float wl_0 = 1.0f / (_S74 * _S74);
        float3 _S75 = wide_rel_0(_S71.row_4, i_1, &kernelContext_17);
        float3 _S76 = cross(_S75, float3(1.0f, 0.0f, 0.0f));
        float3 _S77 = cross(_S75, float3(0.0f, 1.0f, 0.0f));
        float3 _S78 = cross(_S75, float3(0.0f, 0.0f, 1.0f));
        array<float3, int(3)> c_1 = { { _S76, _S77, _S78 } };
        rr_0[int(0)] = wa_0 + wl_0 * dot(_S76, _S76);
        rr_0[int(1)] = wa_0 + wl_0 * dot(_S77, _S77);
        rr_0[int(2)] = wa_0 + wl_0 * dot(_S78, _S78);
        rr_0[int(3)] = wl_0 * dot(_S76, _S77);
        rr_0[int(4)] = wl_0 * dot(_S76, _S78);
        rr_0[int(5)] = wl_0 * dot(_S77, _S78);
        uint kk_1 = 0U;
        for(;;)
        {
            if(kk_1 < 3U)
            {
            }
            else
            {
                break;
            }
            rt_0[kk_1] = float3(wl_0)  * c_1[kk_1];
            kk_1 = kk_1 + 1U;
        }
        tt_0 = wl_0;
    }
    else
    {
        tt_0 = 0.0f;
    }
    uint _S79 = group_5.x;
    write_partial_0(tid_8, _S71.row_4, _S79, 0U, float4(rr_0[int(0)], rr_0[int(1)], rr_0[int(2)], rr_0[int(3)]), &kernelContext_17);
    write_partial_0(tid_8, _S71.row_4, _S79, 1U, float4(rr_0[int(4)], rr_0[int(5)], rt_0[int(0)].x, rt_0[int(0)].y), &kernelContext_17);
    write_partial_0(tid_8, _S71.row_4, _S79, 2U, float4(rt_0[int(0)].z, rt_0[int(1)].x, rt_0[int(1)].y, rt_0[int(1)].z), &kernelContext_17);
    write_partial_0(tid_8, _S71.row_4, _S79, 3U, float4(rt_0[int(2)], tt_0), &kernelContext_17);
    return;
}

uint4 row_range_0(uint tid_9, uint row_10, KernelContext_0 thread* kernelContext_18)
{
    uint _S80 = row_island_0(row_10, kernelContext_18);
    return group_uniform4_0(uint4(*(kernelContext_18->islands_0+group_uniform_0(_S80, tid_9))) , tid_9);
}

[[kernel]] void w_gram(uint3 group_6 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], atomic<uint> device* meta_8 [[buffer(11)]], uint device* topo_args_8 [[buffer(12)]], packed_uint4 device* islands_8 [[buffer(1)]], BlastParams_0 constant* params_8 [[buffer(0)]], packed_float4 device* wide_8 [[buffer(10)]], uint device* lists_8 [[buffer(9)]], BlastNode_natural_0 device* nodes_8 [[buffer(2)]], packed_float4 device* accel_8 [[buffer(5)]], packed_float4 device* scratch_8 [[buffer(7)]], uint device* refs_8 [[buffer(3)]], BlastBond_natural_0 device* bonds_8 [[buffer(4)]], packed_float4 device* x_8 [[buffer(6)]], packed_uint4 device* report_8 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_19;
    (&kernelContext_19)->meta_0 = meta_8;
    (&kernelContext_19)->topo_args_0 = topo_args_8;
    (&kernelContext_19)->islands_0 = islands_8;
    (&kernelContext_19)->params_0 = params_8;
    (&kernelContext_19)->wide_0 = wide_8;
    (&kernelContext_19)->lists_0 = lists_8;
    (&kernelContext_19)->nodes_0 = nodes_8;
    (&kernelContext_19)->accel_0 = accel_8;
    (&kernelContext_19)->scratch_0 = scratch_8;
    (&kernelContext_19)->refs_0 = refs_8;
    (&kernelContext_19)->bonds_0 = bonds_8;
    (&kernelContext_19)->x_0 = x_8;
    (&kernelContext_19)->report_0 = report_8;
    threadgroup array<float2, int(32)> g_blast_part_8;
    (&kernelContext_19)->g_blast_part_0 = &g_blast_part_8;
    threadgroup array<float, int(1024)> g_wide_max_8;
    (&kernelContext_19)->g_wide_max_0 = &g_wide_max_8;
    threadgroup float g_wide_max_out_8;
    (&kernelContext_19)->g_wide_max_out_0 = &g_wide_max_out_8;
    uint tid_10 = thread_6.x;
    uint row_11 = group_6.x;
    uint _S81 = flags_of_0(row_11, &kernelContext_19);
    uint f_1 = group_uniform_0(_S81, tid_10);
    bool _S82;
    if((f_1 & 8U) == 0U)
    {
        _S82 = true;
    }
    else
    {
        _S82 = (f_1 & 1U) == 0U;
    }
    if(_S82)
    {
        return;
    }
    uint4 _S83 = row_range_0(tid_10, row_11, &kernelContext_19);
    uint nb_2 = blocks_0(_S83.y - _S83.x);
    float4 _S84 = row_sum_0(tid_10, row_11, nb_2, 0U, &kernelContext_19);
    float4 _S85 = row_sum_0(tid_10, row_11, nb_2, 1U, &kernelContext_19);
    float4 _S86 = row_sum_0(tid_10, row_11, nb_2, 2U, &kernelContext_19);
    float4 _S87 = row_sum_0(tid_10, row_11, nb_2, 3U, &kernelContext_19);
    if(tid_10 != 0U)
    {
        return;
    }
    thread array<float, int(36)> m_0;
    uint z_1 = 0U;
    for(;;)
    {
        if(z_1 < 36U)
        {
        }
        else
        {
            break;
        }
        m_0[z_1] = 0.0f;
        z_1 = z_1 + 1U;
    }
    m_0[int(0)] = _S84.x;
    m_0[int(7)] = _S84.y;
    m_0[int(14)] = _S84.z;
    float _S88 = _S84.w;
    m_0[int(1)] = _S88;
    m_0[int(6)] = _S88;
    float _S89 = _S85.x;
    m_0[int(2)] = _S89;
    m_0[int(12)] = _S89;
    float _S90 = _S85.y;
    m_0[int(8)] = _S90;
    m_0[int(13)] = _S90;
    float3 _S91 = float3(_S85.z, _S85.w, _S86.x);
    float3 _S92 = float3(_S86.y, _S86.z, _S86.w);
    float3 _S93 = _S87.xyz;
    uint j_1 = 0U;
    for(;;)
    {
        if(j_1 < 3U)
        {
        }
        else
        {
            break;
        }
        uint _S94 = 3U + j_1;
        m_0[_S94] = _S91[j_1];
        m_0[9U + j_1] = _S92[j_1];
        m_0[15U + j_1] = _S93[j_1];
        uint _S95 = _S94 * 6U;
        m_0[_S95] = _S91[j_1];
        m_0[_S95 + 1U] = _S92[j_1];
        m_0[_S95 + 2U] = _S93[j_1];
        m_0[_S95 + 3U + j_1] = _S87.w;
        j_1 = j_1 + 1U;
    }
    z_1 = 0U;
    for(;;)
    {
        if(z_1 < 9U)
        {
        }
        else
        {
            break;
        }
        uint _S96 = 4U * z_1;
        *((&kernelContext_19)->wide_0+st_0(row_11, 6U + z_1)) = packed_float4(float4(m_0[_S96], m_0[_S96 + 1U], m_0[_S96 + 2U], m_0[_S96 + 3U])) ;
        z_1 = z_1 + 1U;
    }
    return;
}

uint bs_0(uint j_2, KernelContext_0 thread* kernelContext_20)
{
    return kernelContext_20->params_0->node_count_0 * 24U + 2U * j_2;
}

float3 va_0(uint n_7, uint v_4, KernelContext_0 thread* kernelContext_21)
{
    return (float4(*(kernelContext_21->scratch_0+ns_0(n_7, v_4))) ).xyz;
}

float3 vl_0(uint n_8, uint v_5, KernelContext_0 thread* kernelContext_22)
{
    return (float4(*(kernelContext_22->scratch_0+(ns_0(n_8, v_5) + 1U))) ).xyz;
}

void bond_term_0(const BlastBond_natural_0 thread* b_5, uint v_6, float3 thread* ta_0, float3 thread* tl_0, KernelContext_0 thread* kernelContext_23)
{
    uint _S97 = b_5->n0_0;
    BlastNode_natural_0 m0_0 = kernelContext_23->nodes_0[b_5->n0_0];
    uint _S98 = b_5->n1_0;
    BlastNode_natural_0 m1_0 = kernelContext_23->nodes_0[b_5->n1_0];
    float3 _S99 = va_0(b_5->n0_0, v_6, kernelContext_23);
    float3 a0_1 = _S99 * float3(m0_0.w_ang_0) ;
    float3 _S100 = vl_0(_S97, v_6, kernelContext_23);
    float3 l0_0 = _S100 * float3(m0_0.w_lin_0) ;
    float3 _S101 = va_0(_S98, v_6, kernelContext_23);
    float3 a1_1 = _S101 * float3(m1_0.w_ang_0) ;
    float3 _S102 = vl_0(_S98, v_6, kernelContext_23);
    float3 l1_0 = _S102 * float3(m1_0.w_lin_0) ;
    float3 _S103 = float3(b_5->s_0) ;
    *ta_0 = (a0_1 - a1_1) * _S103;
    *tl_0 = (l0_0 - l1_0 + (cross((float4(b_5->o0_0) ).xyz, a0_1) - cross((float4(b_5->o1_0) ).xyz, a1_1))) * _S103;
    return;
}

void bond_term_1(const BlastBond_natural_0 thread* b_6, uint v_7, float3 thread* ta_1, float3 thread* tl_1, KernelContext_0 thread* kernelContext_24)
{
    uint _S104 = b_6->n0_0;
    BlastNode_natural_0 m0_1 = kernelContext_24->nodes_0[b_6->n0_0];
    uint _S105 = b_6->n1_0;
    BlastNode_natural_0 m1_1 = kernelContext_24->nodes_0[b_6->n1_0];
    float3 _S106 = va_0(b_6->n0_0, v_7, kernelContext_24);
    float3 a0_2 = _S106 * float3(m0_1.w_ang_0) ;
    float3 _S107 = vl_0(_S104, v_7, kernelContext_24);
    float3 l0_1 = _S107 * float3(m0_1.w_lin_0) ;
    float3 _S108 = va_0(_S105, v_7, kernelContext_24);
    float3 a1_2 = _S108 * float3(m1_1.w_ang_0) ;
    float3 _S109 = vl_0(_S105, v_7, kernelContext_24);
    float3 l1_1 = _S109 * float3(m1_1.w_lin_0) ;
    float3 _S110 = float3(b_6->s_0) ;
    *ta_1 = (a0_2 - a1_2) * _S110;
    *tl_1 = (l0_1 - l1_1 + (cross((float4(b_6->o0_0) ).xyz, a0_2) - cross((float4(b_6->o1_0) ).xyz, a1_2))) * _S110;
    return;
}

void gather_0(uint node_1, bool from_t_0, bool with_u_0, float3 thread* ang_0, float3 thread* lin_0, KernelContext_0 thread* kernelContext_25)
{
    BlastNode_natural_0 n_9 = kernelContext_25->nodes_0[node_1];
    float3 _S111 = float3(0.0f) ;
    uint r_2 = n_9.ref_begin_0;
    float3 acc_a_0 = _S111;
    float3 acc_l_0 = _S111;
    for(;;)
    {
        if(r_2 < (n_9.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_1 = kernelContext_25->refs_0[r_2];
        uint j_3 = ref_1 & 2147483647U;
        thread BlastBond_natural_0 _S112 = *(kernelContext_25->bonds_0+j_3);
        if(!(((&_S112)->health_0) > 0.0f))
        {
            r_2 = r_2 + 1U;
            continue;
        }
        float3 ya_0;
        float3 yl_0;
        if(from_t_0)
        {
            uint _S113 = bs_0(j_3, kernelContext_25);
            float3 _S114 = (float4(*(kernelContext_25->scratch_0+(_S113 + 1U))) ).xyz;
            ya_0 = (float4(*(kernelContext_25->scratch_0+_S113)) ).xyz;
            yl_0 = _S114;
        }
        else
        {
            uint _S115 = j_3 * 2U;
            float3 ya_1 = (float4(*(kernelContext_25->x_0+_S115)) ).xyz;
            float3 yl_1 = (float4(*(kernelContext_25->x_0+(_S115 + 1U))) ).xyz;
            if(with_u_0)
            {
                thread float3 ta_2;
                thread float3 tl_2;
                bond_term_0(&_S112, 6U, &ta_2, &tl_2, kernelContext_25);
                float3 yl_2 = yl_1 + tl_2;
                ya_0 = ya_1 + ta_2;
                yl_0 = yl_2;
            }
            else
            {
                ya_0 = ya_1;
                yl_0 = yl_1;
            }
        }
        float3 _S116 = float3((&_S112)->s_0) ;
        float3 ya_2 = ya_0 * _S116;
        float3 yl_3 = yl_0 * _S116;
        float3 acc_a_1;
        float3 acc_l_1;
        if((ref_1 & 2147483648U) != 0U)
        {
            float3 acc_l_2 = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross((float4((&_S112)->o1_0) ).xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            float3 acc_l_3 = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross((float4((&_S112)->o0_0) ).xyz, yl_3));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_2 = r_2 + 1U;
    }
    *ang_0 = acc_a_0 * float3(n_9.w_ang_0) ;
    *lin_0 = acc_l_0 * float3(n_9.w_lin_0) ;
    return;
}

[[kernel]] void w_warm(uint3 group_7 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], atomic<uint> device* meta_9 [[buffer(11)]], uint device* topo_args_9 [[buffer(12)]], packed_uint4 device* islands_9 [[buffer(1)]], BlastParams_0 constant* params_9 [[buffer(0)]], packed_float4 device* wide_9 [[buffer(10)]], uint device* lists_9 [[buffer(9)]], BlastNode_natural_0 device* nodes_9 [[buffer(2)]], packed_float4 device* accel_9 [[buffer(5)]], packed_float4 device* scratch_9 [[buffer(7)]], uint device* refs_9 [[buffer(3)]], BlastBond_natural_0 device* bonds_9 [[buffer(4)]], packed_float4 device* x_9 [[buffer(6)]], packed_uint4 device* report_9 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_26;
    (&kernelContext_26)->meta_0 = meta_9;
    (&kernelContext_26)->topo_args_0 = topo_args_9;
    (&kernelContext_26)->islands_0 = islands_9;
    (&kernelContext_26)->params_0 = params_9;
    (&kernelContext_26)->wide_0 = wide_9;
    (&kernelContext_26)->lists_0 = lists_9;
    (&kernelContext_26)->nodes_0 = nodes_9;
    (&kernelContext_26)->accel_0 = accel_9;
    (&kernelContext_26)->scratch_0 = scratch_9;
    (&kernelContext_26)->refs_0 = refs_9;
    (&kernelContext_26)->bonds_0 = bonds_9;
    (&kernelContext_26)->x_0 = x_9;
    (&kernelContext_26)->report_0 = report_9;
    threadgroup array<float2, int(32)> g_blast_part_9;
    (&kernelContext_26)->g_blast_part_0 = &g_blast_part_9;
    threadgroup array<float, int(1024)> g_wide_max_9;
    (&kernelContext_26)->g_wide_max_0 = &g_wide_max_9;
    threadgroup float g_wide_max_out_9;
    (&kernelContext_26)->g_wide_max_out_0 = &g_wide_max_out_9;
    uint tid_11 = thread_7.x;
    Block_0 _S117 = node_block_0(group_7, tid_11, &kernelContext_26);
    bool _S118;
    if((_S117.begin_0) >= (_S117.end_0))
    {
        _S118 = true;
    }
    else
    {
        _S118 = ((&kernelContext_26)->params_0->warm_0) == 0U;
    }
    if(_S118)
    {
        _S118 = true;
    }
    else
    {
        _S118 = ((_S117.flags_0) & 1U) == 0U;
    }
    if(_S118)
    {
        return;
    }
    uint p_3 = _S117.begin_0 + tid_11;
    if(p_3 < (_S117.end_0))
    {
        uint i_2 = (&kernelContext_26)->lists_0[p_3];
        thread float3 a_2;
        thread float3 l_1;
        gather_0(i_2, false, false, &a_2, &l_1, &kernelContext_26);
        float3 _S119 = va_0(i_2, 14U, &kernelContext_26);
        float3 _S120 = _S119 - a_2;
        float3 _S121 = vl_0(i_2, 14U, &kernelContext_26);
        set_v_0(i_2, 0U, _S120, _S121 - l_1, &kernelContext_26);
    }
    return;
}

bool runs_0(uint flags_1, bool only_rebuild_0)
{
    bool _S122;
    if((flags_1 & 1U) != 0U)
    {
        if(!only_rebuild_0)
        {
            _S122 = true;
        }
        else
        {
            _S122 = (flags_1 & 64U) != 0U;
        }
    }
    else
    {
        _S122 = false;
    }
    return _S122;
}

void proj_part_0(uint3 group_8, uint tid_12, uint v_8, bool only_rebuild_1, KernelContext_0 thread* kernelContext_27)
{
    Block_0 _S123 = node_block_0(group_8, tid_12, kernelContext_27);
    bool _S124;
    if((_S123.begin_0) >= (_S123.end_0))
    {
        _S124 = true;
    }
    else
    {
        _S124 = !runs_0(_S123.flags_0, only_rebuild_1);
    }
    if(_S124)
    {
        _S124 = true;
    }
    else
    {
        _S124 = ((_S123.flags_0) & 8U) == 0U;
    }
    if(_S124)
    {
        return;
    }
    float3 _S125 = float3(0.0f) ;
    uint p_4 = _S123.begin_0 + tid_12;
    float3 sr_0;
    float3 stt_0;
    if(p_4 < (_S123.end_0))
    {
        uint i_3 = kernelContext_27->lists_0[p_4];
        BlastNode_natural_0 n_10 = kernelContext_27->nodes_0[i_3];
        float3 _S126 = vl_0(i_3, v_8, kernelContext_27);
        float3 lin_1 = _S126 / float3(n_10.w_lin_0) ;
        float3 _S127 = va_0(i_3, v_8, kernelContext_27);
        float3 _S128 = _S127 / float3(n_10.w_ang_0) ;
        float3 _S129 = wide_rel_0(_S123.row_4, i_3, kernelContext_27);
        sr_0 = _S128 - cross(_S129, lin_1);
        stt_0 = lin_1;
    }
    else
    {
        sr_0 = _S125;
        stt_0 = _S125;
    }
    uint _S130 = group_8.x;
    write_partial_0(tid_12, _S123.row_4, _S130, 0U, float4(sr_0, stt_0.x), kernelContext_27);
    write_partial_0(tid_12, _S123.row_4, _S130, 1U, float4(stt_0.yz, 0.0f, 0.0f), kernelContext_27);
    return;
}

[[kernel]] void w_proj_part_r(uint3 g_2 [[threadgroup_position_in_grid]], uint3 t_0 [[thread_position_in_threadgroup]], atomic<uint> device* meta_10 [[buffer(11)]], uint device* topo_args_10 [[buffer(12)]], packed_uint4 device* islands_10 [[buffer(1)]], BlastParams_0 constant* params_10 [[buffer(0)]], packed_float4 device* wide_10 [[buffer(10)]], uint device* lists_10 [[buffer(9)]], BlastNode_natural_0 device* nodes_10 [[buffer(2)]], packed_float4 device* accel_10 [[buffer(5)]], packed_float4 device* scratch_10 [[buffer(7)]], uint device* refs_10 [[buffer(3)]], BlastBond_natural_0 device* bonds_10 [[buffer(4)]], packed_float4 device* x_10 [[buffer(6)]], packed_uint4 device* report_10 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_28;
    (&kernelContext_28)->meta_0 = meta_10;
    (&kernelContext_28)->topo_args_0 = topo_args_10;
    (&kernelContext_28)->islands_0 = islands_10;
    (&kernelContext_28)->params_0 = params_10;
    (&kernelContext_28)->wide_0 = wide_10;
    (&kernelContext_28)->lists_0 = lists_10;
    (&kernelContext_28)->nodes_0 = nodes_10;
    (&kernelContext_28)->accel_0 = accel_10;
    (&kernelContext_28)->scratch_0 = scratch_10;
    (&kernelContext_28)->refs_0 = refs_10;
    (&kernelContext_28)->bonds_0 = bonds_10;
    (&kernelContext_28)->x_0 = x_10;
    (&kernelContext_28)->report_0 = report_10;
    threadgroup array<float2, int(32)> g_blast_part_10;
    (&kernelContext_28)->g_blast_part_0 = &g_blast_part_10;
    threadgroup array<float, int(1024)> g_wide_max_10;
    (&kernelContext_28)->g_wide_max_0 = &g_wide_max_10;
    threadgroup float g_wide_max_out_10;
    (&kernelContext_28)->g_wide_max_out_0 = &g_wide_max_out_10;
    proj_part_0(g_2, t_0.x, 0U, false, &kernelContext_28);
    return;
}

void proj_solve_0(uint3 group_9, uint tid_13, bool only_rebuild_2, KernelContext_0 thread* kernelContext_29)
{
    uint k_6;
    uint row_12 = group_9.x;
    uint _S131 = flags_of_0(row_12, kernelContext_29);
    uint f_2 = group_uniform_0(_S131, tid_13);
    bool ok_0;
    if(!runs_0(f_2, only_rebuild_2))
    {
        ok_0 = true;
    }
    else
    {
        ok_0 = (f_2 & 8U) == 0U;
    }
    if(ok_0)
    {
        return;
    }
    uint4 _S132 = row_range_0(tid_13, row_12, kernelContext_29);
    uint nb_3 = blocks_0(_S132.y - _S132.x);
    float4 _S133 = row_sum_0(tid_13, row_12, nb_3, 0U, kernelContext_29);
    float4 _S134 = row_sum_0(tid_13, row_12, nb_3, 1U, kernelContext_29);
    if(tid_13 != 0U)
    {
        return;
    }
    thread array<array<float, int(6)>, int(6)> g_3;
    float _S135 = _S133.y;
    float _S136 = _S133.z;
    float _S137 = _S133.w;
    float _S138 = _S134.x;
    float _S139 = _S134.y;
    thread array<float, int(6)> rhs_0;
    rhs_0[int(0)] = _S133.x;
    rhs_0[int(1)] = _S135;
    rhs_0[int(2)] = _S136;
    rhs_0[int(3)] = _S137;
    rhs_0[int(4)] = _S138;
    rhs_0[int(5)] = _S139;
    uint z_2 = 0U;
    for(;;)
    {
        if(z_2 < 36U)
        {
        }
        else
        {
            break;
        }
        g_3[z_2 / 6U][z_2 % 6U] = (float4(*(kernelContext_29->wide_0+st_0(row_12, 6U + z_2 / 4U))) )[z_2 % 4U];
        z_2 = z_2 + 1U;
    }
    thread array<float, int(6)> c_2;
    uint col_0 = 0U;
    for(;;)
    {
        if(col_0 < 6U)
        {
        }
        else
        {
            ok_0 = true;
            break;
        }
        uint _S140 = col_0 + 1U;
        uint piv_0 = col_0;
        uint rr_1 = _S140;
        for(;;)
        {
            if(rr_1 < 6U)
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
            rr_1 = rr_1 + 1U;
        }
        k_6 = 0U;
        for(;;)
        {
            if(k_6 < 6U)
            {
            }
            else
            {
                break;
            }
            float t_1 = g_3[col_0][k_6];
            g_3[col_0][k_6] = g_3[piv_0][k_6];
            g_3[piv_0][k_6] = t_1;
            k_6 = k_6 + 1U;
        }
        float tb_0 = rhs_0[col_0];
        rhs_0[col_0] = rhs_0[piv_0];
        rhs_0[piv_0] = tb_0;
        if(!((abs(g_3[col_0][col_0])) > 0.0f))
        {
            ok_0 = false;
            break;
        }
        uint rr_2 = _S140;
        for(;;)
        {
            if(rr_2 < 6U)
            {
            }
            else
            {
                break;
            }
            float fct_0 = g_3[rr_2][col_0] / g_3[col_0][col_0];
            uint k_7 = col_0;
            for(;;)
            {
                if(k_7 < 6U)
                {
                }
                else
                {
                    break;
                }
                g_3[rr_2][k_7] = g_3[rr_2][k_7] - fct_0 * g_3[col_0][k_7];
                k_7 = k_7 + 1U;
            }
            rhs_0[rr_2] = rhs_0[rr_2] - fct_0 * rhs_0[col_0];
            rr_2 = rr_2 + 1U;
        }
        col_0 = _S140;
    }
    z_2 = 0U;
    for(;;)
    {
        if(z_2 < 6U)
        {
        }
        else
        {
            break;
        }
        c_2[z_2] = 0.0f;
        z_2 = z_2 + 1U;
    }
    if(ok_0)
    {
        int rr_3 = int(5);
        for(;;)
        {
            if(rr_3 >= int(0))
            {
            }
            else
            {
                break;
            }
            k_6 = uint(rr_3) + 1U;
            float s_5 = rhs_0[rr_3];
            for(;;)
            {
                if(k_6 < 6U)
                {
                }
                else
                {
                    break;
                }
                float s_6 = s_5 - g_3[rr_3][k_6] * c_2[k_6];
                k_6 = k_6 + 1U;
                s_5 = s_6;
            }
            c_2[rr_3] = s_5 / g_3[rr_3][rr_3];
            rr_3 = rr_3 - int(1);
        }
    }
    *(kernelContext_29->wide_0+st_0(row_12, 4U)) = packed_float4(float4(c_2[int(0)], c_2[int(1)], c_2[int(2)], 0.0f)) ;
    *(kernelContext_29->wide_0+st_0(row_12, 5U)) = packed_float4(float4(c_2[int(3)], c_2[int(4)], c_2[int(5)], 0.0f)) ;
    return;
}

[[kernel]] void w_proj_solve(uint3 g_4 [[threadgroup_position_in_grid]], uint3 t_2 [[thread_position_in_threadgroup]], atomic<uint> device* meta_11 [[buffer(11)]], uint device* topo_args_11 [[buffer(12)]], packed_uint4 device* islands_11 [[buffer(1)]], BlastParams_0 constant* params_11 [[buffer(0)]], packed_float4 device* wide_11 [[buffer(10)]], uint device* lists_11 [[buffer(9)]], BlastNode_natural_0 device* nodes_11 [[buffer(2)]], packed_float4 device* accel_11 [[buffer(5)]], packed_float4 device* scratch_11 [[buffer(7)]], uint device* refs_11 [[buffer(3)]], BlastBond_natural_0 device* bonds_11 [[buffer(4)]], packed_float4 device* x_11 [[buffer(6)]], packed_uint4 device* report_11 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_30;
    (&kernelContext_30)->meta_0 = meta_11;
    (&kernelContext_30)->topo_args_0 = topo_args_11;
    (&kernelContext_30)->islands_0 = islands_11;
    (&kernelContext_30)->params_0 = params_11;
    (&kernelContext_30)->wide_0 = wide_11;
    (&kernelContext_30)->lists_0 = lists_11;
    (&kernelContext_30)->nodes_0 = nodes_11;
    (&kernelContext_30)->accel_0 = accel_11;
    (&kernelContext_30)->scratch_0 = scratch_11;
    (&kernelContext_30)->refs_0 = refs_11;
    (&kernelContext_30)->bonds_0 = bonds_11;
    (&kernelContext_30)->x_0 = x_11;
    (&kernelContext_30)->report_0 = report_11;
    threadgroup array<float2, int(32)> g_blast_part_11;
    (&kernelContext_30)->g_blast_part_0 = &g_blast_part_11;
    threadgroup array<float, int(1024)> g_wide_max_11;
    (&kernelContext_30)->g_wide_max_0 = &g_wide_max_11;
    threadgroup float g_wide_max_out_11;
    (&kernelContext_30)->g_wide_max_out_0 = &g_wide_max_out_11;
    proj_solve_0(g_4, t_2.x, false, &kernelContext_30);
    return;
}

void proj_apply_0(uint3 group_10, uint tid_14, uint v_9, bool only_rebuild_3, KernelContext_0 thread* kernelContext_31)
{
    Block_0 _S141 = node_block_0(group_10, tid_14, kernelContext_31);
    bool _S142;
    if((_S141.begin_0) >= (_S141.end_0))
    {
        _S142 = true;
    }
    else
    {
        _S142 = !runs_0(_S141.flags_0, only_rebuild_3);
    }
    if(_S142)
    {
        _S142 = true;
    }
    else
    {
        _S142 = ((_S141.flags_0) & 8U) == 0U;
    }
    if(_S142)
    {
        return;
    }
    uint p_5 = _S141.begin_0 + tid_14;
    if(p_5 >= (_S141.end_0))
    {
        return;
    }
    uint i_4 = kernelContext_31->lists_0[p_5];
    BlastNode_natural_0 n_11 = kernelContext_31->nodes_0[i_4];
    float3 w_2 = (float4(*(kernelContext_31->wide_0+st_0(_S141.row_4, 4U))) ).xyz;
    float3 t_3 = (float4(*(kernelContext_31->wide_0+st_0(_S141.row_4, 5U))) ).xyz;
    float3 _S143 = va_0(i_4, v_9, kernelContext_31);
    float3 _S144 = _S143 - w_2 / float3(n_11.w_ang_0) ;
    float3 _S145 = vl_0(i_4, v_9, kernelContext_31);
    float3 _S146 = wide_rel_0(_S141.row_4, i_4, kernelContext_31);
    set_v_0(i_4, v_9, _S144, _S145 - (t_3 + cross(_S146, w_2)) / float3(n_11.w_lin_0) , kernelContext_31);
    return;
}

[[kernel]] void w_proj_apply_r(uint3 g_5 [[threadgroup_position_in_grid]], uint3 t_4 [[thread_position_in_threadgroup]], atomic<uint> device* meta_12 [[buffer(11)]], uint device* topo_args_12 [[buffer(12)]], packed_uint4 device* islands_12 [[buffer(1)]], BlastParams_0 constant* params_12 [[buffer(0)]], packed_float4 device* wide_12 [[buffer(10)]], uint device* lists_12 [[buffer(9)]], BlastNode_natural_0 device* nodes_12 [[buffer(2)]], packed_float4 device* accel_12 [[buffer(5)]], packed_float4 device* scratch_12 [[buffer(7)]], uint device* refs_12 [[buffer(3)]], BlastBond_natural_0 device* bonds_12 [[buffer(4)]], packed_float4 device* x_12 [[buffer(6)]], packed_uint4 device* report_12 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_32;
    (&kernelContext_32)->meta_0 = meta_12;
    (&kernelContext_32)->topo_args_0 = topo_args_12;
    (&kernelContext_32)->islands_0 = islands_12;
    (&kernelContext_32)->params_0 = params_12;
    (&kernelContext_32)->wide_0 = wide_12;
    (&kernelContext_32)->lists_0 = lists_12;
    (&kernelContext_32)->nodes_0 = nodes_12;
    (&kernelContext_32)->accel_0 = accel_12;
    (&kernelContext_32)->scratch_0 = scratch_12;
    (&kernelContext_32)->refs_0 = refs_12;
    (&kernelContext_32)->bonds_0 = bonds_12;
    (&kernelContext_32)->x_0 = x_12;
    (&kernelContext_32)->report_0 = report_12;
    threadgroup array<float2, int(32)> g_blast_part_12;
    (&kernelContext_32)->g_blast_part_0 = &g_blast_part_12;
    threadgroup array<float, int(1024)> g_wide_max_12;
    (&kernelContext_32)->g_wide_max_0 = &g_wide_max_12;
    threadgroup float g_wide_max_out_12;
    (&kernelContext_32)->g_wide_max_out_0 = &g_wide_max_out_12;
    proj_apply_0(g_5, t_4.x, 0U, false, &kernelContext_32);
    return;
}

[[kernel]] void w_proj_part_rb(uint3 g_6 [[threadgroup_position_in_grid]], uint3 t_5 [[thread_position_in_threadgroup]], atomic<uint> device* meta_13 [[buffer(11)]], uint device* topo_args_13 [[buffer(12)]], packed_uint4 device* islands_13 [[buffer(1)]], BlastParams_0 constant* params_13 [[buffer(0)]], packed_float4 device* wide_13 [[buffer(10)]], uint device* lists_13 [[buffer(9)]], BlastNode_natural_0 device* nodes_13 [[buffer(2)]], packed_float4 device* accel_13 [[buffer(5)]], packed_float4 device* scratch_13 [[buffer(7)]], uint device* refs_13 [[buffer(3)]], BlastBond_natural_0 device* bonds_13 [[buffer(4)]], packed_float4 device* x_13 [[buffer(6)]], packed_uint4 device* report_13 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_33;
    (&kernelContext_33)->meta_0 = meta_13;
    (&kernelContext_33)->topo_args_0 = topo_args_13;
    (&kernelContext_33)->islands_0 = islands_13;
    (&kernelContext_33)->params_0 = params_13;
    (&kernelContext_33)->wide_0 = wide_13;
    (&kernelContext_33)->lists_0 = lists_13;
    (&kernelContext_33)->nodes_0 = nodes_13;
    (&kernelContext_33)->accel_0 = accel_13;
    (&kernelContext_33)->scratch_0 = scratch_13;
    (&kernelContext_33)->refs_0 = refs_13;
    (&kernelContext_33)->bonds_0 = bonds_13;
    (&kernelContext_33)->x_0 = x_13;
    (&kernelContext_33)->report_0 = report_13;
    threadgroup array<float2, int(32)> g_blast_part_13;
    (&kernelContext_33)->g_blast_part_0 = &g_blast_part_13;
    threadgroup array<float, int(1024)> g_wide_max_13;
    (&kernelContext_33)->g_wide_max_0 = &g_wide_max_13;
    threadgroup float g_wide_max_out_13;
    (&kernelContext_33)->g_wide_max_out_0 = &g_wide_max_out_13;
    proj_part_0(g_6, t_5.x, 0U, true, &kernelContext_33);
    return;
}

[[kernel]] void w_proj_solve_rb(uint3 g_7 [[threadgroup_position_in_grid]], uint3 t_6 [[thread_position_in_threadgroup]], atomic<uint> device* meta_14 [[buffer(11)]], uint device* topo_args_14 [[buffer(12)]], packed_uint4 device* islands_14 [[buffer(1)]], BlastParams_0 constant* params_14 [[buffer(0)]], packed_float4 device* wide_14 [[buffer(10)]], uint device* lists_14 [[buffer(9)]], BlastNode_natural_0 device* nodes_14 [[buffer(2)]], packed_float4 device* accel_14 [[buffer(5)]], packed_float4 device* scratch_14 [[buffer(7)]], uint device* refs_14 [[buffer(3)]], BlastBond_natural_0 device* bonds_14 [[buffer(4)]], packed_float4 device* x_14 [[buffer(6)]], packed_uint4 device* report_14 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_34;
    (&kernelContext_34)->meta_0 = meta_14;
    (&kernelContext_34)->topo_args_0 = topo_args_14;
    (&kernelContext_34)->islands_0 = islands_14;
    (&kernelContext_34)->params_0 = params_14;
    (&kernelContext_34)->wide_0 = wide_14;
    (&kernelContext_34)->lists_0 = lists_14;
    (&kernelContext_34)->nodes_0 = nodes_14;
    (&kernelContext_34)->accel_0 = accel_14;
    (&kernelContext_34)->scratch_0 = scratch_14;
    (&kernelContext_34)->refs_0 = refs_14;
    (&kernelContext_34)->bonds_0 = bonds_14;
    (&kernelContext_34)->x_0 = x_14;
    (&kernelContext_34)->report_0 = report_14;
    threadgroup array<float2, int(32)> g_blast_part_14;
    (&kernelContext_34)->g_blast_part_0 = &g_blast_part_14;
    threadgroup array<float, int(1024)> g_wide_max_14;
    (&kernelContext_34)->g_wide_max_0 = &g_wide_max_14;
    threadgroup float g_wide_max_out_14;
    (&kernelContext_34)->g_wide_max_out_0 = &g_wide_max_out_14;
    proj_solve_0(g_7, t_6.x, true, &kernelContext_34);
    return;
}

[[kernel]] void w_proj_apply_rb(uint3 g_8 [[threadgroup_position_in_grid]], uint3 t_7 [[thread_position_in_threadgroup]], atomic<uint> device* meta_15 [[buffer(11)]], uint device* topo_args_15 [[buffer(12)]], packed_uint4 device* islands_15 [[buffer(1)]], BlastParams_0 constant* params_15 [[buffer(0)]], packed_float4 device* wide_15 [[buffer(10)]], uint device* lists_15 [[buffer(9)]], BlastNode_natural_0 device* nodes_15 [[buffer(2)]], packed_float4 device* accel_15 [[buffer(5)]], packed_float4 device* scratch_15 [[buffer(7)]], uint device* refs_15 [[buffer(3)]], BlastBond_natural_0 device* bonds_15 [[buffer(4)]], packed_float4 device* x_15 [[buffer(6)]], packed_uint4 device* report_15 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_35;
    (&kernelContext_35)->meta_0 = meta_15;
    (&kernelContext_35)->topo_args_0 = topo_args_15;
    (&kernelContext_35)->islands_0 = islands_15;
    (&kernelContext_35)->params_0 = params_15;
    (&kernelContext_35)->wide_0 = wide_15;
    (&kernelContext_35)->lists_0 = lists_15;
    (&kernelContext_35)->nodes_0 = nodes_15;
    (&kernelContext_35)->accel_0 = accel_15;
    (&kernelContext_35)->scratch_0 = scratch_15;
    (&kernelContext_35)->refs_0 = refs_15;
    (&kernelContext_35)->bonds_0 = bonds_15;
    (&kernelContext_35)->x_0 = x_15;
    (&kernelContext_35)->report_0 = report_15;
    threadgroup array<float2, int(32)> g_blast_part_15;
    (&kernelContext_35)->g_blast_part_0 = &g_blast_part_15;
    threadgroup array<float, int(1024)> g_wide_max_15;
    (&kernelContext_35)->g_wide_max_0 = &g_wide_max_15;
    threadgroup float g_wide_max_out_15;
    (&kernelContext_35)->g_wide_max_out_0 = &g_wide_max_out_15;
    proj_apply_0(g_8, t_7.x, 0U, true, &kernelContext_35);
    return;
}

[[kernel]] void w_proj_part_g(uint3 g_9 [[threadgroup_position_in_grid]], uint3 t_8 [[thread_position_in_threadgroup]], atomic<uint> device* meta_16 [[buffer(11)]], uint device* topo_args_16 [[buffer(12)]], packed_uint4 device* islands_16 [[buffer(1)]], BlastParams_0 constant* params_16 [[buffer(0)]], packed_float4 device* wide_16 [[buffer(10)]], uint device* lists_16 [[buffer(9)]], BlastNode_natural_0 device* nodes_16 [[buffer(2)]], packed_float4 device* accel_16 [[buffer(5)]], packed_float4 device* scratch_16 [[buffer(7)]], uint device* refs_16 [[buffer(3)]], BlastBond_natural_0 device* bonds_16 [[buffer(4)]], packed_float4 device* x_16 [[buffer(6)]], packed_uint4 device* report_16 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_36;
    (&kernelContext_36)->meta_0 = meta_16;
    (&kernelContext_36)->topo_args_0 = topo_args_16;
    (&kernelContext_36)->islands_0 = islands_16;
    (&kernelContext_36)->params_0 = params_16;
    (&kernelContext_36)->wide_0 = wide_16;
    (&kernelContext_36)->lists_0 = lists_16;
    (&kernelContext_36)->nodes_0 = nodes_16;
    (&kernelContext_36)->accel_0 = accel_16;
    (&kernelContext_36)->scratch_0 = scratch_16;
    (&kernelContext_36)->refs_0 = refs_16;
    (&kernelContext_36)->bonds_0 = bonds_16;
    (&kernelContext_36)->x_0 = x_16;
    (&kernelContext_36)->report_0 = report_16;
    threadgroup array<float2, int(32)> g_blast_part_16;
    (&kernelContext_36)->g_blast_part_0 = &g_blast_part_16;
    threadgroup array<float, int(1024)> g_wide_max_16;
    (&kernelContext_36)->g_wide_max_0 = &g_wide_max_16;
    threadgroup float g_wide_max_out_16;
    (&kernelContext_36)->g_wide_max_out_0 = &g_wide_max_out_16;
    proj_part_0(g_9, t_8.x, 8U, false, &kernelContext_36);
    return;
}

[[kernel]] void w_proj_apply_g(uint3 g_10 [[threadgroup_position_in_grid]], uint3 t_9 [[thread_position_in_threadgroup]], atomic<uint> device* meta_17 [[buffer(11)]], uint device* topo_args_17 [[buffer(12)]], packed_uint4 device* islands_17 [[buffer(1)]], BlastParams_0 constant* params_17 [[buffer(0)]], packed_float4 device* wide_17 [[buffer(10)]], uint device* lists_17 [[buffer(9)]], BlastNode_natural_0 device* nodes_17 [[buffer(2)]], packed_float4 device* accel_17 [[buffer(5)]], packed_float4 device* scratch_17 [[buffer(7)]], uint device* refs_17 [[buffer(3)]], BlastBond_natural_0 device* bonds_17 [[buffer(4)]], packed_float4 device* x_17 [[buffer(6)]], packed_uint4 device* report_17 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_37;
    (&kernelContext_37)->meta_0 = meta_17;
    (&kernelContext_37)->topo_args_0 = topo_args_17;
    (&kernelContext_37)->islands_0 = islands_17;
    (&kernelContext_37)->params_0 = params_17;
    (&kernelContext_37)->wide_0 = wide_17;
    (&kernelContext_37)->lists_0 = lists_17;
    (&kernelContext_37)->nodes_0 = nodes_17;
    (&kernelContext_37)->accel_0 = accel_17;
    (&kernelContext_37)->scratch_0 = scratch_17;
    (&kernelContext_37)->refs_0 = refs_17;
    (&kernelContext_37)->bonds_0 = bonds_17;
    (&kernelContext_37)->x_0 = x_17;
    (&kernelContext_37)->report_0 = report_17;
    threadgroup array<float2, int(32)> g_blast_part_17;
    (&kernelContext_37)->g_blast_part_0 = &g_blast_part_17;
    threadgroup array<float, int(1024)> g_wide_max_17;
    (&kernelContext_37)->g_wide_max_0 = &g_wide_max_17;
    threadgroup float g_wide_max_out_17;
    (&kernelContext_37)->g_wide_max_out_0 = &g_wide_max_out_17;
    proj_apply_0(g_10, t_9.x, 8U, false, &kernelContext_37);
    return;
}

void bond_r_0(uint3 group_11, uint tid_15, bool only_rebuild_4, KernelContext_0 thread* kernelContext_38)
{
    Block_0 _S147 = bond_block_0(group_11, tid_15, kernelContext_38);
    bool _S148;
    if((_S147.begin_0) >= (_S147.end_0))
    {
        _S148 = true;
    }
    else
    {
        _S148 = !runs_0(_S147.flags_0, only_rebuild_4);
    }
    if(_S148)
    {
        return;
    }
    uint p_6 = _S147.begin_0 + tid_15;
    float z_3;
    float f_3;
    if(p_6 < (_S147.end_0))
    {
        uint j_4 = kernelContext_38->lists_0[p_6];
        thread BlastBond_natural_0 _S149 = *(kernelContext_38->bonds_0+j_4);
        float3 _S150 = float3(0.0f) ;
        thread float3 ta_3 = _S150;
        thread float3 tl_3 = _S150;
        if(((&_S149)->health_0) > 0.0f)
        {
            bond_term_0(&_S149, 0U, &ta_3, &tl_3, kernelContext_38);
            if((kernelContext_38->params_0->force_tolerance_0) > 0.0f)
            {
                thread float3 ua_0;
                thread float3 ul_0;
                bond_term_0(&_S149, 6U, &ua_0, &ul_0, kernelContext_38);
                uint _S151 = j_4 * 2U;
                z_3 = norm2_0((float4(*(kernelContext_38->x_0+_S151)) ).xyz + ua_0, (float4(*(kernelContext_38->x_0+(_S151 + 1U))) ).xyz + ul_0);
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
        float _S152 = norm2_0(ta_3, tl_3);
        uint _S153 = bs_0(j_4, kernelContext_38);
        *(kernelContext_38->scratch_0+_S153) = packed_float4(float4(ta_3, 0.0f)) ;
        *(kernelContext_38->scratch_0+(_S153 + 1U)) = packed_float4(float4(tl_3, 0.0f)) ;
        float _S154 = z_3;
        z_3 = _S152;
        f_3 = _S154;
    }
    else
    {
        z_3 = 0.0f;
        f_3 = 0.0f;
    }
    write_partial_0(tid_15, _S147.row_4, group_11.x, 2U, float4(z_3, f_3, 0.0f, 0.0f), kernelContext_38);
    return;
}

[[kernel]] void w_bond_r(uint3 g_11 [[threadgroup_position_in_grid]], uint3 t_10 [[thread_position_in_threadgroup]], atomic<uint> device* meta_18 [[buffer(11)]], uint device* topo_args_18 [[buffer(12)]], packed_uint4 device* islands_18 [[buffer(1)]], BlastParams_0 constant* params_18 [[buffer(0)]], packed_float4 device* wide_18 [[buffer(10)]], uint device* lists_18 [[buffer(9)]], BlastNode_natural_0 device* nodes_18 [[buffer(2)]], packed_float4 device* accel_18 [[buffer(5)]], packed_float4 device* scratch_18 [[buffer(7)]], uint device* refs_18 [[buffer(3)]], BlastBond_natural_0 device* bonds_18 [[buffer(4)]], packed_float4 device* x_18 [[buffer(6)]], packed_uint4 device* report_18 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_39;
    (&kernelContext_39)->meta_0 = meta_18;
    (&kernelContext_39)->topo_args_0 = topo_args_18;
    (&kernelContext_39)->islands_0 = islands_18;
    (&kernelContext_39)->params_0 = params_18;
    (&kernelContext_39)->wide_0 = wide_18;
    (&kernelContext_39)->lists_0 = lists_18;
    (&kernelContext_39)->nodes_0 = nodes_18;
    (&kernelContext_39)->accel_0 = accel_18;
    (&kernelContext_39)->scratch_0 = scratch_18;
    (&kernelContext_39)->refs_0 = refs_18;
    (&kernelContext_39)->bonds_0 = bonds_18;
    (&kernelContext_39)->x_0 = x_18;
    (&kernelContext_39)->report_0 = report_18;
    threadgroup array<float2, int(32)> g_blast_part_18;
    (&kernelContext_39)->g_blast_part_0 = &g_blast_part_18;
    threadgroup array<float, int(1024)> g_wide_max_18;
    (&kernelContext_39)->g_wide_max_0 = &g_wide_max_18;
    threadgroup float g_wide_max_out_18;
    (&kernelContext_39)->g_wide_max_out_0 = &g_wide_max_out_18;
    bond_r_0(g_11, t_10.x, false, &kernelContext_39);
    return;
}

void node_rn_0(uint3 group_12, uint tid_16, bool only_rebuild_5, KernelContext_0 thread* kernelContext_40)
{
    Block_0 _S155 = node_block_0(group_12, tid_16, kernelContext_40);
    bool _S156;
    if((_S155.begin_0) >= (_S155.end_0))
    {
        _S156 = true;
    }
    else
    {
        _S156 = !runs_0(_S155.flags_0, only_rebuild_5);
    }
    if(_S156)
    {
        return;
    }
    uint p_7 = _S155.begin_0 + tid_16;
    float s_7;
    if(p_7 < (_S155.end_0))
    {
        uint i_5 = kernelContext_40->lists_0[p_7];
        float3 _S157 = va_0(i_5, 0U, kernelContext_40);
        float3 _S158 = vl_0(i_5, 0U, kernelContext_40);
        s_7 = norm2_0(_S157, _S158);
    }
    else
    {
        s_7 = 0.0f;
    }
    write_partial_0(tid_16, _S155.row_4, group_12.x, 3U, float4(s_7, 0.0f, 0.0f, 0.0f), kernelContext_40);
    return;
}

[[kernel]] void w_node_rn(uint3 g_12 [[threadgroup_position_in_grid]], uint3 t_11 [[thread_position_in_threadgroup]], atomic<uint> device* meta_19 [[buffer(11)]], uint device* topo_args_19 [[buffer(12)]], packed_uint4 device* islands_19 [[buffer(1)]], BlastParams_0 constant* params_19 [[buffer(0)]], packed_float4 device* wide_19 [[buffer(10)]], uint device* lists_19 [[buffer(9)]], BlastNode_natural_0 device* nodes_19 [[buffer(2)]], packed_float4 device* accel_19 [[buffer(5)]], packed_float4 device* scratch_19 [[buffer(7)]], uint device* refs_19 [[buffer(3)]], BlastBond_natural_0 device* bonds_19 [[buffer(4)]], packed_float4 device* x_19 [[buffer(6)]], packed_uint4 device* report_19 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_41;
    (&kernelContext_41)->meta_0 = meta_19;
    (&kernelContext_41)->topo_args_0 = topo_args_19;
    (&kernelContext_41)->islands_0 = islands_19;
    (&kernelContext_41)->params_0 = params_19;
    (&kernelContext_41)->wide_0 = wide_19;
    (&kernelContext_41)->lists_0 = lists_19;
    (&kernelContext_41)->nodes_0 = nodes_19;
    (&kernelContext_41)->accel_0 = accel_19;
    (&kernelContext_41)->scratch_0 = scratch_19;
    (&kernelContext_41)->refs_0 = refs_19;
    (&kernelContext_41)->bonds_0 = bonds_19;
    (&kernelContext_41)->x_0 = x_19;
    (&kernelContext_41)->report_0 = report_19;
    threadgroup array<float2, int(32)> g_blast_part_19;
    (&kernelContext_41)->g_blast_part_0 = &g_blast_part_19;
    threadgroup array<float, int(1024)> g_wide_max_19;
    (&kernelContext_41)->g_wide_max_0 = &g_wide_max_19;
    threadgroup float g_wide_max_out_19;
    (&kernelContext_41)->g_wide_max_out_0 = &g_wide_max_out_19;
    node_rn_0(g_12, t_11.x, false, &kernelContext_41);
    return;
}

[[kernel]] void w_bond_rb(uint3 g_13 [[threadgroup_position_in_grid]], uint3 t_12 [[thread_position_in_threadgroup]], atomic<uint> device* meta_20 [[buffer(11)]], uint device* topo_args_20 [[buffer(12)]], packed_uint4 device* islands_20 [[buffer(1)]], BlastParams_0 constant* params_20 [[buffer(0)]], packed_float4 device* wide_20 [[buffer(10)]], uint device* lists_20 [[buffer(9)]], BlastNode_natural_0 device* nodes_20 [[buffer(2)]], packed_float4 device* accel_20 [[buffer(5)]], packed_float4 device* scratch_20 [[buffer(7)]], uint device* refs_20 [[buffer(3)]], BlastBond_natural_0 device* bonds_20 [[buffer(4)]], packed_float4 device* x_20 [[buffer(6)]], packed_uint4 device* report_20 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_42;
    (&kernelContext_42)->meta_0 = meta_20;
    (&kernelContext_42)->topo_args_0 = topo_args_20;
    (&kernelContext_42)->islands_0 = islands_20;
    (&kernelContext_42)->params_0 = params_20;
    (&kernelContext_42)->wide_0 = wide_20;
    (&kernelContext_42)->lists_0 = lists_20;
    (&kernelContext_42)->nodes_0 = nodes_20;
    (&kernelContext_42)->accel_0 = accel_20;
    (&kernelContext_42)->scratch_0 = scratch_20;
    (&kernelContext_42)->refs_0 = refs_20;
    (&kernelContext_42)->bonds_0 = bonds_20;
    (&kernelContext_42)->x_0 = x_20;
    (&kernelContext_42)->report_0 = report_20;
    threadgroup array<float2, int(32)> g_blast_part_20;
    (&kernelContext_42)->g_blast_part_0 = &g_blast_part_20;
    threadgroup array<float, int(1024)> g_wide_max_20;
    (&kernelContext_42)->g_wide_max_0 = &g_wide_max_20;
    threadgroup float g_wide_max_out_20;
    (&kernelContext_42)->g_wide_max_out_0 = &g_wide_max_out_20;
    bond_r_0(g_13, t_12.x, true, &kernelContext_42);
    return;
}

[[kernel]] void w_node_rnb(uint3 g_14 [[threadgroup_position_in_grid]], uint3 t_13 [[thread_position_in_threadgroup]], atomic<uint> device* meta_21 [[buffer(11)]], uint device* topo_args_21 [[buffer(12)]], packed_uint4 device* islands_21 [[buffer(1)]], BlastParams_0 constant* params_21 [[buffer(0)]], packed_float4 device* wide_21 [[buffer(10)]], uint device* lists_21 [[buffer(9)]], BlastNode_natural_0 device* nodes_21 [[buffer(2)]], packed_float4 device* accel_21 [[buffer(5)]], packed_float4 device* scratch_21 [[buffer(7)]], uint device* refs_21 [[buffer(3)]], BlastBond_natural_0 device* bonds_21 [[buffer(4)]], packed_float4 device* x_21 [[buffer(6)]], packed_uint4 device* report_21 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_43;
    (&kernelContext_43)->meta_0 = meta_21;
    (&kernelContext_43)->topo_args_0 = topo_args_21;
    (&kernelContext_43)->islands_0 = islands_21;
    (&kernelContext_43)->params_0 = params_21;
    (&kernelContext_43)->wide_0 = wide_21;
    (&kernelContext_43)->lists_0 = lists_21;
    (&kernelContext_43)->nodes_0 = nodes_21;
    (&kernelContext_43)->accel_0 = accel_21;
    (&kernelContext_43)->scratch_0 = scratch_21;
    (&kernelContext_43)->refs_0 = refs_21;
    (&kernelContext_43)->bonds_0 = bonds_21;
    (&kernelContext_43)->x_0 = x_21;
    (&kernelContext_43)->report_0 = report_21;
    threadgroup array<float2, int(32)> g_blast_part_21;
    (&kernelContext_43)->g_blast_part_0 = &g_blast_part_21;
    threadgroup array<float, int(1024)> g_wide_max_21;
    (&kernelContext_43)->g_wide_max_0 = &g_wide_max_21;
    threadgroup float g_wide_max_out_21;
    (&kernelContext_43)->g_wide_max_out_0 = &g_wide_max_out_21;
    node_rn_0(g_14, t_13.x, true, &kernelContext_43);
    return;
}

uint iteration_of_0(uint row_13, KernelContext_0 thread* kernelContext_44)
{
    return (as_type<uint>(((float4(*(kernelContext_44->wide_0+st_0(row_13, 2U))) ).z)));
}

void set_flags_0(uint row_14, uint f_4, KernelContext_0 thread* kernelContext_45)
{
    uint _S159 = st_0(row_14, 2U);
    thread float4 c_3 = float4(*(kernelContext_45->wide_0+_S159)) ;
    c_3.w = (as_type<float>((f_4)));
    *(kernelContext_45->wide_0+_S159) = packed_float4(c_3) ;
    return;
}

[[kernel]] void w_test(uint3 group_13 [[threadgroup_position_in_grid]], uint3 thread_8 [[thread_position_in_threadgroup]], atomic<uint> device* meta_22 [[buffer(11)]], uint device* topo_args_22 [[buffer(12)]], packed_uint4 device* islands_22 [[buffer(1)]], BlastParams_0 constant* params_22 [[buffer(0)]], packed_float4 device* wide_22 [[buffer(10)]], uint device* lists_22 [[buffer(9)]], BlastNode_natural_0 device* nodes_22 [[buffer(2)]], packed_float4 device* accel_22 [[buffer(5)]], packed_float4 device* scratch_22 [[buffer(7)]], uint device* refs_22 [[buffer(3)]], BlastBond_natural_0 device* bonds_22 [[buffer(4)]], packed_float4 device* x_22 [[buffer(6)]], packed_uint4 device* report_22 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_46;
    (&kernelContext_46)->meta_0 = meta_22;
    (&kernelContext_46)->topo_args_0 = topo_args_22;
    (&kernelContext_46)->islands_0 = islands_22;
    (&kernelContext_46)->params_0 = params_22;
    (&kernelContext_46)->wide_0 = wide_22;
    (&kernelContext_46)->lists_0 = lists_22;
    (&kernelContext_46)->nodes_0 = nodes_22;
    (&kernelContext_46)->accel_0 = accel_22;
    (&kernelContext_46)->scratch_0 = scratch_22;
    (&kernelContext_46)->refs_0 = refs_22;
    (&kernelContext_46)->bonds_0 = bonds_22;
    (&kernelContext_46)->x_0 = x_22;
    (&kernelContext_46)->report_0 = report_22;
    threadgroup array<float2, int(32)> g_blast_part_22;
    (&kernelContext_46)->g_blast_part_0 = &g_blast_part_22;
    threadgroup array<float, int(1024)> g_wide_max_22;
    (&kernelContext_46)->g_wide_max_0 = &g_wide_max_22;
    threadgroup float g_wide_max_out_22;
    (&kernelContext_46)->g_wide_max_out_0 = &g_wide_max_out_22;
    uint tid_17 = thread_8.x;
    uint row_15 = group_13.x;
    uint _S160 = flags_of_0(row_15, &kernelContext_46);
    uint f_5 = group_uniform_0(_S160, tid_17);
    if((f_5 & 1U) == 0U)
    {
        return;
    }
    uint4 _S161 = row_range_0(tid_17, row_15, &kernelContext_46);
    float4 _S162 = row_sum_0(tid_17, row_15, blocks_0(_S161.w - _S161.z), 2U, &kernelContext_46);
    float4 _S163 = row_sum_0(tid_17, row_15, blocks_0(_S161.y - _S161.x), 3U, &kernelContext_46);
    if(tid_17 != 0U)
    {
        return;
    }
    uint _S164 = st_0(row_15, 0U);
    float4 _S165 = float4(*((&kernelContext_46)->wide_0+_S164)) ;
    thread float4 s0_0 = _S165;
    float delta2_1 = _S165.x;
    float _S166 = _S162.x;
    bool _S167 = _S166 <= delta2_1;
    bool hold_0;
    if(_S167)
    {
        hold_0 = !((_S163.x) <= delta2_1);
    }
    else
    {
        hold_0 = false;
    }
    uint _S168 = iteration_of_0(row_15, &kernelContext_46);
    uint f_6 = f_5 & 4294967199U;
    uint f_7;
    if(hold_0)
    {
        f_7 = f_6 | 32U;
    }
    else
    {
        f_7 = f_6;
    }
    bool _S169;
    if(_S168 > 0U)
    {
        _S169 = true;
    }
    else
    {
        _S169 = ((&kernelContext_46)->params_0->warm_0) != 0U;
    }
    if(_S169)
    {
        _S169 = _S167;
    }
    else
    {
        _S169 = false;
    }
    if(_S169)
    {
        hold_0 = !hold_0;
    }
    else
    {
        hold_0 = false;
    }
    if(hold_0)
    {
        f_7 = f_7 | 64U;
    }
    s0_0.y = _S166;
    *((&kernelContext_46)->wide_0+_S164) = packed_float4(s0_0) ;
    uint _S170 = st_0(row_15, 1U);
    *((&kernelContext_46)->wide_0+_S170) = packed_float4(float4(*((&kernelContext_46)->wide_0+_S170)) ) ;
    set_flags_0(row_15, f_7, &kernelContext_46);
    uint _S171 = st_0(row_15, 2U);
    float4 _S172 = float4(*((&kernelContext_46)->wide_0+_S171)) ;
    uint _S173 = st_0(row_15, 3U);
    *((&kernelContext_46)->wide_0+_S173) = packed_float4(float4((float4(*((&kernelContext_46)->wide_0+_S173)) ).xyz, (float4(*((&kernelContext_46)->wide_0+_S173)) ).w)) ;
    uint _S174 = pt_0(row_15, 0U, 5U, &kernelContext_46);
    *((&kernelContext_46)->wide_0+_S174) = packed_float4(float4(_S162.y, 0.0f, 0.0f, 0.0f)) ;
    *((&kernelContext_46)->wide_0+_S171) = packed_float4(_S172) ;
    return;
}

[[kernel]] void w_rebuild(uint3 group_14 [[threadgroup_position_in_grid]], uint3 thread_9 [[thread_position_in_threadgroup]], atomic<uint> device* meta_23 [[buffer(11)]], uint device* topo_args_23 [[buffer(12)]], packed_uint4 device* islands_23 [[buffer(1)]], BlastParams_0 constant* params_23 [[buffer(0)]], packed_float4 device* wide_23 [[buffer(10)]], uint device* lists_23 [[buffer(9)]], BlastNode_natural_0 device* nodes_23 [[buffer(2)]], packed_float4 device* accel_23 [[buffer(5)]], packed_float4 device* scratch_23 [[buffer(7)]], uint device* refs_23 [[buffer(3)]], BlastBond_natural_0 device* bonds_23 [[buffer(4)]], packed_float4 device* x_23 [[buffer(6)]], packed_uint4 device* report_23 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_47;
    (&kernelContext_47)->meta_0 = meta_23;
    (&kernelContext_47)->topo_args_0 = topo_args_23;
    (&kernelContext_47)->islands_0 = islands_23;
    (&kernelContext_47)->params_0 = params_23;
    (&kernelContext_47)->wide_0 = wide_23;
    (&kernelContext_47)->lists_0 = lists_23;
    (&kernelContext_47)->nodes_0 = nodes_23;
    (&kernelContext_47)->accel_0 = accel_23;
    (&kernelContext_47)->scratch_0 = scratch_23;
    (&kernelContext_47)->refs_0 = refs_23;
    (&kernelContext_47)->bonds_0 = bonds_23;
    (&kernelContext_47)->x_0 = x_23;
    (&kernelContext_47)->report_0 = report_23;
    threadgroup array<float2, int(32)> g_blast_part_23;
    (&kernelContext_47)->g_blast_part_0 = &g_blast_part_23;
    threadgroup array<float, int(1024)> g_wide_max_23;
    (&kernelContext_47)->g_wide_max_0 = &g_wide_max_23;
    threadgroup float g_wide_max_out_23;
    (&kernelContext_47)->g_wide_max_out_0 = &g_wide_max_out_23;
    uint tid_18 = thread_9.x;
    Block_0 _S175 = node_block_0(group_14, tid_18, &kernelContext_47);
    bool _S176;
    if((_S175.begin_0) >= (_S175.end_0))
    {
        _S176 = true;
    }
    else
    {
        _S176 = !runs_0(_S175.flags_0, true);
    }
    if(_S176)
    {
        return;
    }
    uint p_8 = _S175.begin_0 + tid_18;
    if(p_8 < (_S175.end_0))
    {
        uint i_6 = (&kernelContext_47)->lists_0[p_8];
        thread float3 a_3;
        thread float3 l_2;
        gather_0(i_6, false, true, &a_3, &l_2, &kernelContext_47);
        float3 _S177 = va_0(i_6, 14U, &kernelContext_47);
        float3 _S178 = _S177 - a_3;
        float3 _S179 = vl_0(i_6, 14U, &kernelContext_47);
        set_v_0(i_6, 0U, _S178, _S179 - l_2, &kernelContext_47);
    }
    return;
}

[[kernel]] void w_decide(uint3 group_15 [[threadgroup_position_in_grid]], uint3 thread_10 [[thread_position_in_threadgroup]], atomic<uint> device* meta_24 [[buffer(11)]], uint device* topo_args_24 [[buffer(12)]], packed_uint4 device* islands_24 [[buffer(1)]], BlastParams_0 constant* params_24 [[buffer(0)]], packed_float4 device* wide_24 [[buffer(10)]], uint device* lists_24 [[buffer(9)]], BlastNode_natural_0 device* nodes_24 [[buffer(2)]], packed_float4 device* accel_24 [[buffer(5)]], packed_float4 device* scratch_24 [[buffer(7)]], uint device* refs_24 [[buffer(3)]], BlastBond_natural_0 device* bonds_24 [[buffer(4)]], packed_float4 device* x_24 [[buffer(6)]], packed_uint4 device* report_24 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_48;
    (&kernelContext_48)->meta_0 = meta_24;
    (&kernelContext_48)->topo_args_0 = topo_args_24;
    (&kernelContext_48)->islands_0 = islands_24;
    (&kernelContext_48)->params_0 = params_24;
    (&kernelContext_48)->wide_0 = wide_24;
    (&kernelContext_48)->lists_0 = lists_24;
    (&kernelContext_48)->nodes_0 = nodes_24;
    (&kernelContext_48)->accel_0 = accel_24;
    (&kernelContext_48)->scratch_0 = scratch_24;
    (&kernelContext_48)->refs_0 = refs_24;
    (&kernelContext_48)->bonds_0 = bonds_24;
    (&kernelContext_48)->x_0 = x_24;
    (&kernelContext_48)->report_0 = report_24;
    threadgroup array<float2, int(32)> g_blast_part_24;
    (&kernelContext_48)->g_blast_part_0 = &g_blast_part_24;
    threadgroup array<float, int(1024)> g_wide_max_24;
    (&kernelContext_48)->g_wide_max_0 = &g_wide_max_24;
    threadgroup float g_wide_max_out_24;
    (&kernelContext_48)->g_wide_max_out_0 = &g_wide_max_out_24;
    uint tid_19 = thread_10.x;
    uint row_16 = group_15.x;
    uint _S180 = flags_of_0(row_16, &kernelContext_48);
    uint f_8 = group_uniform_0(_S180, tid_19);
    if((f_8 & 1U) == 0U)
    {
        return;
    }
    uint4 _S181 = row_range_0(tid_19, row_16, &kernelContext_48);
    float4 _S182 = float4(0.0f) ;
    bool _S183 = (f_8 & 64U) != 0U;
    float4 z_4;
    float4 rn_0;
    if(_S183)
    {
        float4 _S184 = row_sum_0(tid_19, row_16, blocks_0(_S181.w - _S181.z), 2U, &kernelContext_48);
        float4 _S185 = row_sum_0(tid_19, row_16, blocks_0(_S181.y - _S181.x), 3U, &kernelContext_48);
        z_4 = _S184;
        rn_0 = _S185;
    }
    else
    {
        z_4 = _S182;
        rn_0 = _S182;
    }
    if(tid_19 != 0U)
    {
        return;
    }
    uint _S186 = st_0(row_16, 0U);
    float4 _S187 = float4(*((&kernelContext_48)->wide_0+_S186)) ;
    thread float4 s0_1 = _S187;
    float4 _S188 = float4(*((&kernelContext_48)->wide_0+st_0(row_16, 1U))) ;
    float4 _S189 = float4(*((&kernelContext_48)->wide_0+st_0(row_16, 2U))) ;
    uint _S190 = st_0(row_16, 3U);
    float4 _S191 = float4(*((&kernelContext_48)->wide_0+_S190)) ;
    float delta2_2 = _S187.x;
    float _S192 = _S187.y;
    bool _S193 = (f_8 & 32U) != 0U;
    bool hold_1;
    float num_0;
    if(_S183)
    {
        float num_1 = z_4.x;
        if(num_1 <= delta2_2)
        {
            hold_1 = !((rn_0.x) <= delta2_2);
        }
        else
        {
            hold_1 = false;
        }
        s0_1.z = 0.0f;
        num_0 = num_1;
    }
    else
    {
        num_0 = _S192;
        hold_1 = _S193;
    }
    uint f_9 = f_8 & 4294967199U;
    uint it_0 = (as_type<uint>((_S189.z)));
    uint best_it_0 = (as_type<uint>((_S191.w)));
    bool stop_0;
    uint best_it_1;
    if(num_0 < (s0_1.w * 0.99000000953674316f))
    {
        s0_1.w = num_0;
        stop_0 = false;
        best_it_1 = it_0;
    }
    else
    {
        stop_0 = (it_0 - best_it_0) >= 512U;
        best_it_1 = best_it_0;
    }
    bool _S194;
    if(!stop_0)
    {
        _S194 = num_0 <= delta2_2;
    }
    else
    {
        _S194 = false;
    }
    if(_S194)
    {
        hold_1 = !hold_1;
    }
    else
    {
        hold_1 = false;
    }
    uint f_10;
    if(hold_1)
    {
        uint f_11 = f_9 | 2U;
        stop_0 = true;
        f_10 = f_11;
    }
    else
    {
        f_10 = f_9;
    }
    if(!stop_0)
    {
        hold_1 = ((&kernelContext_48)->params_0->force_tolerance_0) > 0.0f;
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        hold_1 = it_0 > 0U;
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        hold_1 = (_S188.w) <= ((&kernelContext_48)->params_0->force_tolerance_0 * (_S189.x + _S189.y));
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        float _S195 = _S188.w;
        float _S196 = (&kernelContext_48)->params_0->force_tolerance_0;
        uint _S197 = pt_0(row_16, 0U, 5U, &kernelContext_48);
        if(_S195 <= (_S196 * sqrt((float4(*((&kernelContext_48)->wide_0+_S197)) ).x)))
        {
            uint f_12 = f_10 | 2U;
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
        hold_1 = it_0 >= ((&kernelContext_48)->params_0->max_iterations_0);
    }
    if(hold_1)
    {
        f_10 = f_10 & 4294967294U;
    }
    s0_1.y = num_0;
    *((&kernelContext_48)->wide_0+_S186) = packed_float4(s0_1) ;
    *((&kernelContext_48)->wide_0+_S190) = packed_float4(float4(_S191.xyz, (as_type<float>((best_it_1))))) ;
    set_flags_0(row_16, f_10, &kernelContext_48);
    return;
}

float3 sym_mul_0(float4 s0_2, float4 s1_0, float3 v_10)
{
    float _S198 = v_10.x;
    float _S199 = s0_2.w;
    float _S200 = v_10.y;
    float _S201 = s1_0.x;
    float _S202 = v_10.z;
    float _S203 = s1_0.y;
    return float3(s0_2.x * _S198 + _S199 * _S200 + _S201 * _S202, _S199 * _S198 + s0_2.y * _S200 + _S203 * _S202, _S201 * _S198 + _S203 * _S200 + s0_2.z * _S202);
}

void inverse_apply_0(uint n_12, float3 a_4, float3 l_3, float3 thread* oa_0, float3 thread* ol_0, KernelContext_0 thread* kernelContext_49)
{
    uint _S204 = ns_0(n_12, 21U);
    float4 _S205 = float4(*(kernelContext_49->scratch_0+(_S204 + 1U))) ;
    float3 kc_0 = (float4(*(kernelContext_49->scratch_0+(_S204 + 2U))) ).xyz;
    float3 _S206 = sym_mul_0(float4(*(kernelContext_49->scratch_0+_S204)) , _S205, a_4 + cross(kc_0, l_3));
    *oa_0 = _S206;
    *ol_0 = l_3 * float3(_S205.z)  - cross(kc_0, _S206);
    return;
}

[[kernel]] void w_pre1(uint3 group_16 [[threadgroup_position_in_grid]], uint3 thread_11 [[thread_position_in_threadgroup]], atomic<uint> device* meta_25 [[buffer(11)]], uint device* topo_args_25 [[buffer(12)]], packed_uint4 device* islands_25 [[buffer(1)]], BlastParams_0 constant* params_25 [[buffer(0)]], packed_float4 device* wide_25 [[buffer(10)]], uint device* lists_25 [[buffer(9)]], BlastNode_natural_0 device* nodes_25 [[buffer(2)]], packed_float4 device* accel_25 [[buffer(5)]], packed_float4 device* scratch_25 [[buffer(7)]], uint device* refs_25 [[buffer(3)]], BlastBond_natural_0 device* bonds_25 [[buffer(4)]], packed_float4 device* x_25 [[buffer(6)]], packed_uint4 device* report_25 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_50;
    (&kernelContext_50)->meta_0 = meta_25;
    (&kernelContext_50)->topo_args_0 = topo_args_25;
    (&kernelContext_50)->islands_0 = islands_25;
    (&kernelContext_50)->params_0 = params_25;
    (&kernelContext_50)->wide_0 = wide_25;
    (&kernelContext_50)->lists_0 = lists_25;
    (&kernelContext_50)->nodes_0 = nodes_25;
    (&kernelContext_50)->accel_0 = accel_25;
    (&kernelContext_50)->scratch_0 = scratch_25;
    (&kernelContext_50)->refs_0 = refs_25;
    (&kernelContext_50)->bonds_0 = bonds_25;
    (&kernelContext_50)->x_0 = x_25;
    (&kernelContext_50)->report_0 = report_25;
    threadgroup array<float2, int(32)> g_blast_part_25;
    (&kernelContext_50)->g_blast_part_0 = &g_blast_part_25;
    threadgroup array<float, int(1024)> g_wide_max_25;
    (&kernelContext_50)->g_wide_max_0 = &g_wide_max_25;
    threadgroup float g_wide_max_out_25;
    (&kernelContext_50)->g_wide_max_out_0 = &g_wide_max_out_25;
    uint tid_20 = thread_11.x;
    Block_0 _S207 = node_block_0(group_16, tid_20, &kernelContext_50);
    bool _S208;
    if((_S207.begin_0) >= (_S207.end_0))
    {
        _S208 = true;
    }
    else
    {
        _S208 = !runs_0(_S207.flags_0, false);
    }
    if(_S208)
    {
        return;
    }
    uint p_9 = _S207.begin_0 + tid_20;
    if(p_9 >= (_S207.end_0))
    {
        return;
    }
    uint i_7 = (&kernelContext_50)->lists_0[p_9];
    uint _S209 = iteration_of_0(_S207.row_4, &kernelContext_50);
    if(_S209 == 0U)
    {
        float3 _S210 = va_0(i_7, 0U, &kernelContext_50);
        float3 _S211 = vl_0(i_7, 0U, &kernelContext_50);
        set_v_0(i_7, 8U, _S210, _S211, &kernelContext_50);
        return;
    }
    float3 _S212 = va_0(i_7, 0U, &kernelContext_50);
    float3 _S213 = vl_0(i_7, 0U, &kernelContext_50);
    thread float3 a_5;
    thread float3 l_4;
    inverse_apply_0(i_7, _S212, _S213, &a_5, &l_4, &kernelContext_50);
    set_v_0(i_7, 10U, a_5, l_4, &kernelContext_50);
    return;
}

[[kernel]] void w_bond_loc(uint3 group_17 [[threadgroup_position_in_grid]], uint3 thread_12 [[thread_position_in_threadgroup]], atomic<uint> device* meta_26 [[buffer(11)]], uint device* topo_args_26 [[buffer(12)]], packed_uint4 device* islands_26 [[buffer(1)]], BlastParams_0 constant* params_26 [[buffer(0)]], packed_float4 device* wide_26 [[buffer(10)]], uint device* lists_26 [[buffer(9)]], BlastNode_natural_0 device* nodes_26 [[buffer(2)]], packed_float4 device* accel_26 [[buffer(5)]], packed_float4 device* scratch_26 [[buffer(7)]], uint device* refs_26 [[buffer(3)]], BlastBond_natural_0 device* bonds_26 [[buffer(4)]], packed_float4 device* x_26 [[buffer(6)]], packed_uint4 device* report_26 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_51;
    (&kernelContext_51)->meta_0 = meta_26;
    (&kernelContext_51)->topo_args_0 = topo_args_26;
    (&kernelContext_51)->islands_0 = islands_26;
    (&kernelContext_51)->params_0 = params_26;
    (&kernelContext_51)->wide_0 = wide_26;
    (&kernelContext_51)->lists_0 = lists_26;
    (&kernelContext_51)->nodes_0 = nodes_26;
    (&kernelContext_51)->accel_0 = accel_26;
    (&kernelContext_51)->scratch_0 = scratch_26;
    (&kernelContext_51)->refs_0 = refs_26;
    (&kernelContext_51)->bonds_0 = bonds_26;
    (&kernelContext_51)->x_0 = x_26;
    (&kernelContext_51)->report_0 = report_26;
    threadgroup array<float2, int(32)> g_blast_part_26;
    (&kernelContext_51)->g_blast_part_0 = &g_blast_part_26;
    threadgroup array<float, int(1024)> g_wide_max_26;
    (&kernelContext_51)->g_wide_max_0 = &g_wide_max_26;
    threadgroup float g_wide_max_out_26;
    (&kernelContext_51)->g_wide_max_out_0 = &g_wide_max_out_26;
    uint tid_21 = thread_12.x;
    Block_0 _S214 = bond_block_0(group_17, tid_21, &kernelContext_51);
    bool _S215;
    if((_S214.begin_0) >= (_S214.end_0))
    {
        _S215 = true;
    }
    else
    {
        _S215 = !runs_0(_S214.flags_0, false);
    }
    if(_S215)
    {
        _S215 = true;
    }
    else
    {
        uint _S216 = iteration_of_0(_S214.row_4, &kernelContext_51);
        _S215 = _S216 == 0U;
    }
    if(_S215)
    {
        return;
    }
    uint p_10 = _S214.begin_0 + tid_21;
    if(p_10 >= (_S214.end_0))
    {
        return;
    }
    uint j_5 = (&kernelContext_51)->lists_0[p_10];
    thread BlastBond_natural_0 _S217 = *((&kernelContext_51)->bonds_0+j_5);
    float3 _S218 = float3(0.0f) ;
    thread float3 ta_4 = _S218;
    thread float3 tl_4 = _S218;
    if(((&_S217)->health_0) > 0.0f)
    {
        bond_term_1(&_S217, 10U, &ta_4, &tl_4, &kernelContext_51);
    }
    uint _S219 = bs_0(j_5, &kernelContext_51);
    *((&kernelContext_51)->scratch_0+_S219) = packed_float4(float4(ta_4, 0.0f)) ;
    *((&kernelContext_51)->scratch_0+(_S219 + 1U)) = packed_float4(float4(tl_4, 0.0f)) ;
    return;
}

void block_apply_0(uint n_13, float3 a_6, float3 l_5, float3 thread* oa_1, float3 thread* ol_1, KernelContext_0 thread* kernelContext_52)
{
    uint _S220 = ns_0(n_13, 18U);
    float4 _S221 = float4(*(kernelContext_52->scratch_0+(_S220 + 1U))) ;
    float3 k_8 = (float4(*(kernelContext_52->scratch_0+(_S220 + 2U))) ).xyz;
    *oa_1 = sym_mul_0(float4(*(kernelContext_52->scratch_0+_S220)) , _S221, a_6) - cross(k_8, l_5);
    *ol_1 = cross(k_8, a_6) + float3(_S221.z)  * l_5;
    return;
}

[[kernel]] void w_pre2(uint3 group_18 [[threadgroup_position_in_grid]], uint3 thread_13 [[thread_position_in_threadgroup]], atomic<uint> device* meta_27 [[buffer(11)]], uint device* topo_args_27 [[buffer(12)]], packed_uint4 device* islands_27 [[buffer(1)]], BlastParams_0 constant* params_27 [[buffer(0)]], packed_float4 device* wide_27 [[buffer(10)]], uint device* lists_27 [[buffer(9)]], BlastNode_natural_0 device* nodes_27 [[buffer(2)]], packed_float4 device* accel_27 [[buffer(5)]], packed_float4 device* scratch_27 [[buffer(7)]], uint device* refs_27 [[buffer(3)]], BlastBond_natural_0 device* bonds_27 [[buffer(4)]], packed_float4 device* x_27 [[buffer(6)]], packed_uint4 device* report_27 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_53;
    (&kernelContext_53)->meta_0 = meta_27;
    (&kernelContext_53)->topo_args_0 = topo_args_27;
    (&kernelContext_53)->islands_0 = islands_27;
    (&kernelContext_53)->params_0 = params_27;
    (&kernelContext_53)->wide_0 = wide_27;
    (&kernelContext_53)->lists_0 = lists_27;
    (&kernelContext_53)->nodes_0 = nodes_27;
    (&kernelContext_53)->accel_0 = accel_27;
    (&kernelContext_53)->scratch_0 = scratch_27;
    (&kernelContext_53)->refs_0 = refs_27;
    (&kernelContext_53)->bonds_0 = bonds_27;
    (&kernelContext_53)->x_0 = x_27;
    (&kernelContext_53)->report_0 = report_27;
    threadgroup array<float2, int(32)> g_blast_part_27;
    (&kernelContext_53)->g_blast_part_0 = &g_blast_part_27;
    threadgroup array<float, int(1024)> g_wide_max_27;
    (&kernelContext_53)->g_wide_max_0 = &g_wide_max_27;
    threadgroup float g_wide_max_out_27;
    (&kernelContext_53)->g_wide_max_out_0 = &g_wide_max_out_27;
    uint tid_22 = thread_13.x;
    Block_0 _S222 = node_block_0(group_18, tid_22, &kernelContext_53);
    bool _S223;
    if((_S222.begin_0) >= (_S222.end_0))
    {
        _S223 = true;
    }
    else
    {
        _S223 = !runs_0(_S222.flags_0, false);
    }
    if(_S223)
    {
        _S223 = true;
    }
    else
    {
        uint _S224 = iteration_of_0(_S222.row_4, &kernelContext_53);
        _S223 = _S224 == 0U;
    }
    if(_S223)
    {
        return;
    }
    uint p_11 = _S222.begin_0 + tid_22;
    if(p_11 >= (_S222.end_0))
    {
        return;
    }
    uint i_8 = (&kernelContext_53)->lists_0[p_11];
    thread float3 oa_2;
    thread float3 ol_2;
    gather_0(i_8, true, false, &oa_2, &ol_2, &kernelContext_53);
    float3 _S225 = va_0(i_8, 10U, &kernelContext_53);
    float3 _S226 = vl_0(i_8, 10U, &kernelContext_53);
    thread float3 sa_0;
    thread float3 sl_0;
    block_apply_0(i_8, _S225, _S226, &sa_0, &sl_0, &kernelContext_53);
    thread float3 ma_0;
    thread float3 ml_0;
    inverse_apply_0(i_8, oa_2 - sa_0, ol_2 - sl_0, &ma_0, &ml_0, &kernelContext_53);
    float3 _S227 = va_0(i_8, 10U, &kernelContext_53);
    float3 _S228 = float3(1.68946564197540283f) ;
    float3 _S229 = float3(1.5220409631729126f) ;
    float3 _S230 = _S228 * _S227 - _S229 * ma_0;
    float3 _S231 = vl_0(i_8, 10U, &kernelContext_53);
    set_v_0(i_8, 8U, _S230, _S228 * _S231 - _S229 * ml_0, &kernelContext_53);
    return;
}

float wide_group_max_0(uint tid_23, float v_11, KernelContext_0 thread* kernelContext_54)
{
    (*kernelContext_54->g_wide_max_0)[tid_23] = v_11;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint k_9 = 512U;
    for(;;)
    {
        if(k_9 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_23 < k_9)
        {
            (*kernelContext_54->g_wide_max_0)[tid_23] = max((*kernelContext_54->g_wide_max_0)[tid_23], (*kernelContext_54->g_wide_max_0)[tid_23 + k_9]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_9 = k_9 >> 1U;
    }
    if(tid_23 == 0U)
    {
        *kernelContext_54->g_wide_max_out_0 = (*kernelContext_54->g_wide_max_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S232 = kernelContext_54->g_wide_max_out_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *_S232;
}

[[kernel]] void w_max(uint3 group_19 [[threadgroup_position_in_grid]], uint3 thread_14 [[thread_position_in_threadgroup]], atomic<uint> device* meta_28 [[buffer(11)]], uint device* topo_args_28 [[buffer(12)]], packed_uint4 device* islands_28 [[buffer(1)]], BlastParams_0 constant* params_28 [[buffer(0)]], packed_float4 device* wide_28 [[buffer(10)]], uint device* lists_28 [[buffer(9)]], BlastNode_natural_0 device* nodes_28 [[buffer(2)]], packed_float4 device* accel_28 [[buffer(5)]], packed_float4 device* scratch_28 [[buffer(7)]], uint device* refs_28 [[buffer(3)]], BlastBond_natural_0 device* bonds_28 [[buffer(4)]], packed_float4 device* x_28 [[buffer(6)]], packed_uint4 device* report_28 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_55;
    (&kernelContext_55)->meta_0 = meta_28;
    (&kernelContext_55)->topo_args_0 = topo_args_28;
    (&kernelContext_55)->islands_0 = islands_28;
    (&kernelContext_55)->params_0 = params_28;
    (&kernelContext_55)->wide_0 = wide_28;
    (&kernelContext_55)->lists_0 = lists_28;
    (&kernelContext_55)->nodes_0 = nodes_28;
    (&kernelContext_55)->accel_0 = accel_28;
    (&kernelContext_55)->scratch_0 = scratch_28;
    (&kernelContext_55)->refs_0 = refs_28;
    (&kernelContext_55)->bonds_0 = bonds_28;
    (&kernelContext_55)->x_0 = x_28;
    (&kernelContext_55)->report_0 = report_28;
    threadgroup array<float2, int(32)> g_blast_part_28;
    (&kernelContext_55)->g_blast_part_0 = &g_blast_part_28;
    threadgroup array<float, int(1024)> g_wide_max_28;
    (&kernelContext_55)->g_wide_max_0 = &g_wide_max_28;
    threadgroup float g_wide_max_out_28;
    (&kernelContext_55)->g_wide_max_out_0 = &g_wide_max_out_28;
    uint tid_24 = thread_14.x;
    Block_0 _S233 = node_block_0(group_19, tid_24, &kernelContext_55);
    bool _S234;
    if((_S233.begin_0) >= (_S233.end_0))
    {
        _S234 = true;
    }
    else
    {
        _S234 = !runs_0(_S233.flags_0, false);
    }
    if(_S234)
    {
        return;
    }
    uint p_12 = _S233.begin_0 + tid_24;
    float m_1;
    if(p_12 < (_S233.end_0))
    {
        uint i_9 = (&kernelContext_55)->lists_0[p_12];
        float3 _S235 = va_0(i_9, 8U, &kernelContext_55);
        float3 a_7 = abs(_S235);
        float3 _S236 = vl_0(i_9, 8U, &kernelContext_55);
        float3 l_6 = abs(_S236);
        m_1 = max(max(max(a_7.x, a_7.y), max(a_7.z, l_6.x)), max(l_6.y, l_6.z));
    }
    else
    {
        m_1 = 0.0f;
    }
    float _S237 = wide_group_max_0(tid_24, m_1, &kernelContext_55);
    if(tid_24 == 0U)
    {
        uint _S238 = pt_0(_S233.row_4, group_19.x, 4U, &kernelContext_55);
        *((&kernelContext_55)->wide_0+_S238) = packed_float4(float4(_S237, 0.0f, 0.0f, 0.0f)) ;
    }
    return;
}

[[kernel]] void w_mag(uint3 group_20 [[threadgroup_position_in_grid]], uint3 thread_15 [[thread_position_in_threadgroup]], atomic<uint> device* meta_29 [[buffer(11)]], uint device* topo_args_29 [[buffer(12)]], packed_uint4 device* islands_29 [[buffer(1)]], BlastParams_0 constant* params_29 [[buffer(0)]], packed_float4 device* wide_29 [[buffer(10)]], uint device* lists_29 [[buffer(9)]], BlastNode_natural_0 device* nodes_29 [[buffer(2)]], packed_float4 device* accel_29 [[buffer(5)]], packed_float4 device* scratch_29 [[buffer(7)]], uint device* refs_29 [[buffer(3)]], BlastBond_natural_0 device* bonds_29 [[buffer(4)]], packed_float4 device* x_29 [[buffer(6)]], packed_uint4 device* report_29 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_56;
    (&kernelContext_56)->meta_0 = meta_29;
    (&kernelContext_56)->topo_args_0 = topo_args_29;
    (&kernelContext_56)->islands_0 = islands_29;
    (&kernelContext_56)->params_0 = params_29;
    (&kernelContext_56)->wide_0 = wide_29;
    (&kernelContext_56)->lists_0 = lists_29;
    (&kernelContext_56)->nodes_0 = nodes_29;
    (&kernelContext_56)->accel_0 = accel_29;
    (&kernelContext_56)->scratch_0 = scratch_29;
    (&kernelContext_56)->refs_0 = refs_29;
    (&kernelContext_56)->bonds_0 = bonds_29;
    (&kernelContext_56)->x_0 = x_29;
    (&kernelContext_56)->report_0 = report_29;
    threadgroup array<float2, int(32)> g_blast_part_29;
    (&kernelContext_56)->g_blast_part_0 = &g_blast_part_29;
    threadgroup array<float, int(1024)> g_wide_max_29;
    (&kernelContext_56)->g_wide_max_0 = &g_wide_max_29;
    threadgroup float g_wide_max_out_29;
    (&kernelContext_56)->g_wide_max_out_0 = &g_wide_max_out_29;
    uint tid_25 = thread_15.x;
    uint row_17 = group_20.x;
    uint _S239 = flags_of_0(row_17, &kernelContext_56);
    uint f_13 = group_uniform_0(_S239, tid_25);
    if((f_13 & 1U) == 0U)
    {
        return;
    }
    uint4 _S240 = row_range_0(tid_25, row_17, &kernelContext_56);
    uint _S241 = blocks_0(_S240.y - _S240.x);
    float m_2 = 0.0f;
    uint blk_0 = tid_25;
    for(;;)
    {
        if(blk_0 < _S241)
        {
        }
        else
        {
            break;
        }
        uint _S242 = pt_0(row_17, blk_0, 4U, &kernelContext_56);
        float _S243 = max(m_2, (float4(*((&kernelContext_56)->wide_0+_S242)) ).x);
        uint blk_1 = blk_0 + 1024U;
        m_2 = _S243;
        blk_0 = blk_1;
    }
    float _S244 = wide_group_max_0(tid_25, m_2, &kernelContext_56);
    if(tid_25 != 0U)
    {
        return;
    }
    bool _S245;
    if(!(_S244 > 0.0f))
    {
        _S245 = true;
    }
    else
    {
        _S245 = !isfinite(_S244);
    }
    if(_S245)
    {
        set_flags_0(row_17, (f_13 & 4294967294U) | 4U, &kernelContext_56);
        return;
    }
    uint _S246 = st_0(row_17, 1U);
    thread float4 s1_1 = float4(*((&kernelContext_56)->wide_0+_S246)) ;
    s1_1.y = 1.0f / _S244;
    *((&kernelContext_56)->wide_0+_S246) = packed_float4(s1_1) ;
    return;
}

[[kernel]] void w_scale(uint3 group_21 [[threadgroup_position_in_grid]], uint3 thread_16 [[thread_position_in_threadgroup]], atomic<uint> device* meta_30 [[buffer(11)]], uint device* topo_args_30 [[buffer(12)]], packed_uint4 device* islands_30 [[buffer(1)]], BlastParams_0 constant* params_30 [[buffer(0)]], packed_float4 device* wide_30 [[buffer(10)]], uint device* lists_30 [[buffer(9)]], BlastNode_natural_0 device* nodes_30 [[buffer(2)]], packed_float4 device* accel_30 [[buffer(5)]], packed_float4 device* scratch_30 [[buffer(7)]], uint device* refs_30 [[buffer(3)]], BlastBond_natural_0 device* bonds_30 [[buffer(4)]], packed_float4 device* x_30 [[buffer(6)]], packed_uint4 device* report_30 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_57;
    (&kernelContext_57)->meta_0 = meta_30;
    (&kernelContext_57)->topo_args_0 = topo_args_30;
    (&kernelContext_57)->islands_0 = islands_30;
    (&kernelContext_57)->params_0 = params_30;
    (&kernelContext_57)->wide_0 = wide_30;
    (&kernelContext_57)->lists_0 = lists_30;
    (&kernelContext_57)->nodes_0 = nodes_30;
    (&kernelContext_57)->accel_0 = accel_30;
    (&kernelContext_57)->scratch_0 = scratch_30;
    (&kernelContext_57)->refs_0 = refs_30;
    (&kernelContext_57)->bonds_0 = bonds_30;
    (&kernelContext_57)->x_0 = x_30;
    (&kernelContext_57)->report_0 = report_30;
    threadgroup array<float2, int(32)> g_blast_part_30;
    (&kernelContext_57)->g_blast_part_0 = &g_blast_part_30;
    threadgroup array<float, int(1024)> g_wide_max_30;
    (&kernelContext_57)->g_wide_max_0 = &g_wide_max_30;
    threadgroup float g_wide_max_out_30;
    (&kernelContext_57)->g_wide_max_out_0 = &g_wide_max_out_30;
    uint tid_26 = thread_16.x;
    Block_0 _S247 = node_block_0(group_21, tid_26, &kernelContext_57);
    bool _S248;
    if((_S247.begin_0) >= (_S247.end_0))
    {
        _S248 = true;
    }
    else
    {
        _S248 = !runs_0(_S247.flags_0, false);
    }
    if(_S248)
    {
        return;
    }
    uint p_13 = _S247.begin_0 + tid_26;
    float gl_0;
    if(p_13 < (_S247.end_0))
    {
        uint i_10 = (&kernelContext_57)->lists_0[p_13];
        float inv_0 = (float4(*((&kernelContext_57)->wide_0+st_0(_S247.row_4, 1U))) ).y;
        float3 _S249 = va_0(i_10, 8U, &kernelContext_57);
        float3 _S250 = float3(inv_0) ;
        float3 ga_0 = _S249 * _S250;
        float3 _S251 = vl_0(i_10, 8U, &kernelContext_57);
        float3 gn_0 = _S251 * _S250;
        set_v_0(i_10, 8U, ga_0, gn_0, &kernelContext_57);
        float3 _S252 = va_0(i_10, 0U, &kernelContext_57);
        float _S253 = dot(_S252, ga_0);
        float3 _S254 = vl_0(i_10, 0U, &kernelContext_57);
        gl_0 = _S253 + dot(_S254, gn_0);
    }
    else
    {
        gl_0 = 0.0f;
    }
    write_partial_0(tid_26, _S247.row_4, group_21.x, 0U, float4(gl_0, 0.0f, 0.0f, 0.0f), &kernelContext_57);
    return;
}

[[kernel]] void w_gamma(uint3 group_22 [[threadgroup_position_in_grid]], uint3 thread_17 [[thread_position_in_threadgroup]], atomic<uint> device* meta_31 [[buffer(11)]], uint device* topo_args_31 [[buffer(12)]], packed_uint4 device* islands_31 [[buffer(1)]], BlastParams_0 constant* params_31 [[buffer(0)]], packed_float4 device* wide_31 [[buffer(10)]], uint device* lists_31 [[buffer(9)]], BlastNode_natural_0 device* nodes_31 [[buffer(2)]], packed_float4 device* accel_31 [[buffer(5)]], packed_float4 device* scratch_31 [[buffer(7)]], uint device* refs_31 [[buffer(3)]], BlastBond_natural_0 device* bonds_31 [[buffer(4)]], packed_float4 device* x_31 [[buffer(6)]], packed_uint4 device* report_31 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_58;
    (&kernelContext_58)->meta_0 = meta_31;
    (&kernelContext_58)->topo_args_0 = topo_args_31;
    (&kernelContext_58)->islands_0 = islands_31;
    (&kernelContext_58)->params_0 = params_31;
    (&kernelContext_58)->wide_0 = wide_31;
    (&kernelContext_58)->lists_0 = lists_31;
    (&kernelContext_58)->nodes_0 = nodes_31;
    (&kernelContext_58)->accel_0 = accel_31;
    (&kernelContext_58)->scratch_0 = scratch_31;
    (&kernelContext_58)->refs_0 = refs_31;
    (&kernelContext_58)->bonds_0 = bonds_31;
    (&kernelContext_58)->x_0 = x_31;
    (&kernelContext_58)->report_0 = report_31;
    threadgroup array<float2, int(32)> g_blast_part_31;
    (&kernelContext_58)->g_blast_part_0 = &g_blast_part_31;
    threadgroup array<float, int(1024)> g_wide_max_31;
    (&kernelContext_58)->g_wide_max_0 = &g_wide_max_31;
    threadgroup float g_wide_max_out_31;
    (&kernelContext_58)->g_wide_max_out_0 = &g_wide_max_out_31;
    uint tid_27 = thread_17.x;
    uint row_18 = group_22.x;
    uint _S255 = flags_of_0(row_18, &kernelContext_58);
    uint f_14 = group_uniform_0(_S255, tid_27);
    if((f_14 & 1U) == 0U)
    {
        return;
    }
    uint4 _S256 = row_range_0(tid_27, row_18, &kernelContext_58);
    float4 _S257 = row_sum_0(tid_27, row_18, blocks_0(_S256.y - _S256.x), 0U, &kernelContext_58);
    float gamma_0 = _S257.x;
    if(tid_27 != 0U)
    {
        return;
    }
    bool _S258;
    if(!(gamma_0 > 0.0f))
    {
        _S258 = true;
    }
    else
    {
        _S258 = !isfinite(gamma_0);
    }
    if(_S258)
    {
        set_flags_0(row_18, (f_14 & 4294967294U) | 4U, &kernelContext_58);
        return;
    }
    float4 _S259 = float4(*((&kernelContext_58)->wide_0+st_0(row_18, 0U))) ;
    uint _S260 = iteration_of_0(row_18, &kernelContext_58);
    if(_S260 > 1U)
    {
        _S258 = (_S259.z) > 0.0f;
    }
    else
    {
        _S258 = false;
    }
    float beta_0;
    if(_S258)
    {
        beta_0 = gamma_0 / _S259.z;
    }
    else
    {
        beta_0 = 0.0f;
    }
    uint _S261 = st_0(row_18, 1U);
    thread float4 s1_2 = float4(*((&kernelContext_58)->wide_0+_S261)) ;
    s1_2.x = gamma_0;
    s1_2.z = beta_0;
    *((&kernelContext_58)->wide_0+_S261) = packed_float4(s1_2) ;
    return;
}

[[kernel]] void w_dir(uint3 group_23 [[threadgroup_position_in_grid]], uint3 thread_18 [[thread_position_in_threadgroup]], atomic<uint> device* meta_32 [[buffer(11)]], uint device* topo_args_32 [[buffer(12)]], packed_uint4 device* islands_32 [[buffer(1)]], BlastParams_0 constant* params_32 [[buffer(0)]], packed_float4 device* wide_32 [[buffer(10)]], uint device* lists_32 [[buffer(9)]], BlastNode_natural_0 device* nodes_32 [[buffer(2)]], packed_float4 device* accel_32 [[buffer(5)]], packed_float4 device* scratch_32 [[buffer(7)]], uint device* refs_32 [[buffer(3)]], BlastBond_natural_0 device* bonds_32 [[buffer(4)]], packed_float4 device* x_32 [[buffer(6)]], packed_uint4 device* report_32 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_59;
    (&kernelContext_59)->meta_0 = meta_32;
    (&kernelContext_59)->topo_args_0 = topo_args_32;
    (&kernelContext_59)->islands_0 = islands_32;
    (&kernelContext_59)->params_0 = params_32;
    (&kernelContext_59)->wide_0 = wide_32;
    (&kernelContext_59)->lists_0 = lists_32;
    (&kernelContext_59)->nodes_0 = nodes_32;
    (&kernelContext_59)->accel_0 = accel_32;
    (&kernelContext_59)->scratch_0 = scratch_32;
    (&kernelContext_59)->refs_0 = refs_32;
    (&kernelContext_59)->bonds_0 = bonds_32;
    (&kernelContext_59)->x_0 = x_32;
    (&kernelContext_59)->report_0 = report_32;
    threadgroup array<float2, int(32)> g_blast_part_32;
    (&kernelContext_59)->g_blast_part_0 = &g_blast_part_32;
    threadgroup array<float, int(1024)> g_wide_max_32;
    (&kernelContext_59)->g_wide_max_0 = &g_wide_max_32;
    threadgroup float g_wide_max_out_32;
    (&kernelContext_59)->g_wide_max_out_0 = &g_wide_max_out_32;
    uint tid_28 = thread_18.x;
    Block_0 _S262 = node_block_0(group_23, tid_28, &kernelContext_59);
    bool _S263;
    if((_S262.begin_0) >= (_S262.end_0))
    {
        _S263 = true;
    }
    else
    {
        _S263 = !runs_0(_S262.flags_0, false);
    }
    if(_S263)
    {
        return;
    }
    uint p_14 = _S262.begin_0 + tid_28;
    if(p_14 >= (_S262.end_0))
    {
        return;
    }
    uint i_11 = (&kernelContext_59)->lists_0[p_14];
    float beta_1 = (float4(*((&kernelContext_59)->wide_0+st_0(_S262.row_4, 1U))) ).z;
    if(beta_1 == 0.0f)
    {
        float3 _S264 = va_0(i_11, 8U, &kernelContext_59);
        float3 _S265 = vl_0(i_11, 8U, &kernelContext_59);
        set_v_0(i_11, 2U, _S264, _S265, &kernelContext_59);
    }
    else
    {
        float3 _S266 = va_0(i_11, 8U, &kernelContext_59);
        float3 _S267 = va_0(i_11, 2U, &kernelContext_59);
        float3 _S268 = float3(beta_1) ;
        float3 _S269 = _S266 + _S267 * _S268;
        float3 _S270 = vl_0(i_11, 8U, &kernelContext_59);
        float3 _S271 = vl_0(i_11, 2U, &kernelContext_59);
        set_v_0(i_11, 2U, _S269, _S270 + _S271 * _S268, &kernelContext_59);
    }
    return;
}

[[kernel]] void w_bond_p(uint3 group_24 [[threadgroup_position_in_grid]], uint3 thread_19 [[thread_position_in_threadgroup]], atomic<uint> device* meta_33 [[buffer(11)]], uint device* topo_args_33 [[buffer(12)]], packed_uint4 device* islands_33 [[buffer(1)]], BlastParams_0 constant* params_33 [[buffer(0)]], packed_float4 device* wide_33 [[buffer(10)]], uint device* lists_33 [[buffer(9)]], BlastNode_natural_0 device* nodes_33 [[buffer(2)]], packed_float4 device* accel_33 [[buffer(5)]], packed_float4 device* scratch_33 [[buffer(7)]], uint device* refs_33 [[buffer(3)]], BlastBond_natural_0 device* bonds_33 [[buffer(4)]], packed_float4 device* x_33 [[buffer(6)]], packed_uint4 device* report_33 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_60;
    (&kernelContext_60)->meta_0 = meta_33;
    (&kernelContext_60)->topo_args_0 = topo_args_33;
    (&kernelContext_60)->islands_0 = islands_33;
    (&kernelContext_60)->params_0 = params_33;
    (&kernelContext_60)->wide_0 = wide_33;
    (&kernelContext_60)->lists_0 = lists_33;
    (&kernelContext_60)->nodes_0 = nodes_33;
    (&kernelContext_60)->accel_0 = accel_33;
    (&kernelContext_60)->scratch_0 = scratch_33;
    (&kernelContext_60)->refs_0 = refs_33;
    (&kernelContext_60)->bonds_0 = bonds_33;
    (&kernelContext_60)->x_0 = x_33;
    (&kernelContext_60)->report_0 = report_33;
    threadgroup array<float2, int(32)> g_blast_part_33;
    (&kernelContext_60)->g_blast_part_0 = &g_blast_part_33;
    threadgroup array<float, int(1024)> g_wide_max_33;
    (&kernelContext_60)->g_wide_max_0 = &g_wide_max_33;
    threadgroup float g_wide_max_out_33;
    (&kernelContext_60)->g_wide_max_out_0 = &g_wide_max_out_33;
    uint tid_29 = thread_19.x;
    Block_0 _S272 = bond_block_0(group_24, tid_29, &kernelContext_60);
    bool _S273;
    if((_S272.begin_0) >= (_S272.end_0))
    {
        _S273 = true;
    }
    else
    {
        _S273 = !runs_0(_S272.flags_0, false);
    }
    if(_S273)
    {
        return;
    }
    uint p_15 = _S272.begin_0 + tid_29;
    float z_5;
    if(p_15 < (_S272.end_0))
    {
        uint j_6 = (&kernelContext_60)->lists_0[p_15];
        thread BlastBond_natural_0 _S274 = *((&kernelContext_60)->bonds_0+j_6);
        float3 _S275 = float3(0.0f) ;
        thread float3 ta_5 = _S275;
        thread float3 tl_5 = _S275;
        if(((&_S274)->health_0) > 0.0f)
        {
            bond_term_1(&_S274, 2U, &ta_5, &tl_5, &kernelContext_60);
        }
        float _S276 = norm2_0(ta_5, tl_5);
        uint _S277 = bs_0(j_6, &kernelContext_60);
        *((&kernelContext_60)->scratch_0+_S277) = packed_float4(float4(ta_5, 0.0f)) ;
        *((&kernelContext_60)->scratch_0+(_S277 + 1U)) = packed_float4(float4(tl_5, 0.0f)) ;
        z_5 = _S276;
    }
    else
    {
        z_5 = 0.0f;
    }
    write_partial_0(tid_29, _S272.row_4, group_24.x, 1U, float4(z_5, 0.0f, 0.0f, 0.0f), &kernelContext_60);
    return;
}

[[kernel]] void w_node_q(uint3 group_25 [[threadgroup_position_in_grid]], uint3 thread_20 [[thread_position_in_threadgroup]], atomic<uint> device* meta_34 [[buffer(11)]], uint device* topo_args_34 [[buffer(12)]], packed_uint4 device* islands_34 [[buffer(1)]], BlastParams_0 constant* params_34 [[buffer(0)]], packed_float4 device* wide_34 [[buffer(10)]], uint device* lists_34 [[buffer(9)]], BlastNode_natural_0 device* nodes_34 [[buffer(2)]], packed_float4 device* accel_34 [[buffer(5)]], packed_float4 device* scratch_34 [[buffer(7)]], uint device* refs_34 [[buffer(3)]], BlastBond_natural_0 device* bonds_34 [[buffer(4)]], packed_float4 device* x_34 [[buffer(6)]], packed_uint4 device* report_34 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_61;
    (&kernelContext_61)->meta_0 = meta_34;
    (&kernelContext_61)->topo_args_0 = topo_args_34;
    (&kernelContext_61)->islands_0 = islands_34;
    (&kernelContext_61)->params_0 = params_34;
    (&kernelContext_61)->wide_0 = wide_34;
    (&kernelContext_61)->lists_0 = lists_34;
    (&kernelContext_61)->nodes_0 = nodes_34;
    (&kernelContext_61)->accel_0 = accel_34;
    (&kernelContext_61)->scratch_0 = scratch_34;
    (&kernelContext_61)->refs_0 = refs_34;
    (&kernelContext_61)->bonds_0 = bonds_34;
    (&kernelContext_61)->x_0 = x_34;
    (&kernelContext_61)->report_0 = report_34;
    threadgroup array<float2, int(32)> g_blast_part_34;
    (&kernelContext_61)->g_blast_part_0 = &g_blast_part_34;
    threadgroup array<float, int(1024)> g_wide_max_34;
    (&kernelContext_61)->g_wide_max_0 = &g_wide_max_34;
    threadgroup float g_wide_max_out_34;
    (&kernelContext_61)->g_wide_max_out_0 = &g_wide_max_out_34;
    uint tid_30 = thread_20.x;
    Block_0 _S278 = node_block_0(group_25, tid_30, &kernelContext_61);
    bool _S279;
    if((_S278.begin_0) >= (_S278.end_0))
    {
        _S279 = true;
    }
    else
    {
        _S279 = !runs_0(_S278.flags_0, false);
    }
    if(_S279)
    {
        return;
    }
    uint p_16 = _S278.begin_0 + tid_30;
    if(p_16 >= (_S278.end_0))
    {
        return;
    }
    uint i_12 = (&kernelContext_61)->lists_0[p_16];
    thread float3 a_8;
    thread float3 l_7;
    gather_0(i_12, true, false, &a_8, &l_7, &kernelContext_61);
    set_v_0(i_12, 4U, a_8, l_7, &kernelContext_61);
    return;
}

[[kernel]] void w_den(uint3 group_26 [[threadgroup_position_in_grid]], uint3 thread_21 [[thread_position_in_threadgroup]], atomic<uint> device* meta_35 [[buffer(11)]], uint device* topo_args_35 [[buffer(12)]], packed_uint4 device* islands_35 [[buffer(1)]], BlastParams_0 constant* params_35 [[buffer(0)]], packed_float4 device* wide_35 [[buffer(10)]], uint device* lists_35 [[buffer(9)]], BlastNode_natural_0 device* nodes_35 [[buffer(2)]], packed_float4 device* accel_35 [[buffer(5)]], packed_float4 device* scratch_35 [[buffer(7)]], uint device* refs_35 [[buffer(3)]], BlastBond_natural_0 device* bonds_35 [[buffer(4)]], packed_float4 device* x_35 [[buffer(6)]], packed_uint4 device* report_35 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_62;
    (&kernelContext_62)->meta_0 = meta_35;
    (&kernelContext_62)->topo_args_0 = topo_args_35;
    (&kernelContext_62)->islands_0 = islands_35;
    (&kernelContext_62)->params_0 = params_35;
    (&kernelContext_62)->wide_0 = wide_35;
    (&kernelContext_62)->lists_0 = lists_35;
    (&kernelContext_62)->nodes_0 = nodes_35;
    (&kernelContext_62)->accel_0 = accel_35;
    (&kernelContext_62)->scratch_0 = scratch_35;
    (&kernelContext_62)->refs_0 = refs_35;
    (&kernelContext_62)->bonds_0 = bonds_35;
    (&kernelContext_62)->x_0 = x_35;
    (&kernelContext_62)->report_0 = report_35;
    threadgroup array<float2, int(32)> g_blast_part_35;
    (&kernelContext_62)->g_blast_part_0 = &g_blast_part_35;
    threadgroup array<float, int(1024)> g_wide_max_35;
    (&kernelContext_62)->g_wide_max_0 = &g_wide_max_35;
    threadgroup float g_wide_max_out_35;
    (&kernelContext_62)->g_wide_max_out_0 = &g_wide_max_out_35;
    uint tid_31 = thread_21.x;
    uint row_19 = group_26.x;
    uint _S280 = flags_of_0(row_19, &kernelContext_62);
    uint f_15 = group_uniform_0(_S280, tid_31);
    if((f_15 & 1U) == 0U)
    {
        return;
    }
    uint4 _S281 = row_range_0(tid_31, row_19, &kernelContext_62);
    float4 _S282 = row_sum_0(tid_31, row_19, blocks_0(_S281.w - _S281.z), 1U, &kernelContext_62);
    float den_0 = _S282.x;
    if(tid_31 != 0U)
    {
        return;
    }
    uint _S283 = st_0(row_19, 0U);
    thread float4 s0_3 = float4(*((&kernelContext_62)->wide_0+_S283)) ;
    uint _S284 = st_0(row_19, 1U);
    float4 _S285 = float4(*((&kernelContext_62)->wide_0+_S284)) ;
    thread float4 s1_3 = _S285;
    uint _S286 = st_0(row_19, 2U);
    thread float4 cnt_0 = float4(*((&kernelContext_62)->wide_0+_S286)) ;
    float gamma_1 = _S285.x;
    bool _S287;
    if(((&kernelContext_62)->params_0->force_tolerance_0) > 0.0f)
    {
        if(den_0 > 0.0f)
        {
            _S287 = gamma_1 > 0.0f;
        }
        else
        {
            _S287 = false;
        }
        float _S288;
        if(_S287)
        {
            _S288 = gamma_1 / sqrt(den_0);
        }
        else
        {
            _S288 = float_bits_0(2139095040U);
        }
        s1_3.w = _S288;
        if(isfinite(s1_3.w))
        {
            cnt_0.y = cnt_0.y + s1_3.w;
        }
    }
    s0_3.z = gamma_1;
    cnt_0.z = (as_type<float>(((as_type<uint>((cnt_0.z))) + 1U)));
    if(!(den_0 > 0.0f))
    {
        _S287 = true;
    }
    else
    {
        _S287 = !isfinite(den_0);
    }
    uint f_16;
    if(_S287)
    {
        uint f_17 = f_15 & 4294967294U;
        s1_3.y = 0.0f;
        f_16 = f_17;
    }
    else
    {
        s1_3.y = gamma_1 / den_0;
        f_16 = f_15;
    }
    *((&kernelContext_62)->wide_0+_S283) = packed_float4(s0_3) ;
    *((&kernelContext_62)->wide_0+_S284) = packed_float4(s1_3) ;
    cnt_0.w = (as_type<float>((f_16)));
    *((&kernelContext_62)->wide_0+_S286) = packed_float4(cnt_0) ;
    return;
}

[[kernel]] void w_update(uint3 group_27 [[threadgroup_position_in_grid]], uint3 thread_22 [[thread_position_in_threadgroup]], atomic<uint> device* meta_36 [[buffer(11)]], uint device* topo_args_36 [[buffer(12)]], packed_uint4 device* islands_36 [[buffer(1)]], BlastParams_0 constant* params_36 [[buffer(0)]], packed_float4 device* wide_36 [[buffer(10)]], uint device* lists_36 [[buffer(9)]], BlastNode_natural_0 device* nodes_36 [[buffer(2)]], packed_float4 device* accel_36 [[buffer(5)]], packed_float4 device* scratch_36 [[buffer(7)]], uint device* refs_36 [[buffer(3)]], BlastBond_natural_0 device* bonds_36 [[buffer(4)]], packed_float4 device* x_36 [[buffer(6)]], packed_uint4 device* report_36 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_63;
    (&kernelContext_63)->meta_0 = meta_36;
    (&kernelContext_63)->topo_args_0 = topo_args_36;
    (&kernelContext_63)->islands_0 = islands_36;
    (&kernelContext_63)->params_0 = params_36;
    (&kernelContext_63)->wide_0 = wide_36;
    (&kernelContext_63)->lists_0 = lists_36;
    (&kernelContext_63)->nodes_0 = nodes_36;
    (&kernelContext_63)->accel_0 = accel_36;
    (&kernelContext_63)->scratch_0 = scratch_36;
    (&kernelContext_63)->refs_0 = refs_36;
    (&kernelContext_63)->bonds_0 = bonds_36;
    (&kernelContext_63)->x_0 = x_36;
    (&kernelContext_63)->report_0 = report_36;
    threadgroup array<float2, int(32)> g_blast_part_36;
    (&kernelContext_63)->g_blast_part_0 = &g_blast_part_36;
    threadgroup array<float, int(1024)> g_wide_max_36;
    (&kernelContext_63)->g_wide_max_0 = &g_wide_max_36;
    threadgroup float g_wide_max_out_36;
    (&kernelContext_63)->g_wide_max_out_0 = &g_wide_max_out_36;
    uint tid_32 = thread_22.x;
    Block_0 _S289 = node_block_0(group_27, tid_32, &kernelContext_63);
    bool _S290;
    if((_S289.begin_0) >= (_S289.end_0))
    {
        _S290 = true;
    }
    else
    {
        _S290 = !runs_0(_S289.flags_0, false);
    }
    if(_S290)
    {
        return;
    }
    uint p_17 = _S289.begin_0 + tid_32;
    if(p_17 >= (_S289.end_0))
    {
        return;
    }
    uint i_13 = (&kernelContext_63)->lists_0[p_17];
    float alpha_0 = (float4(*((&kernelContext_63)->wide_0+st_0(_S289.row_4, 1U))) ).y;
    float3 _S291 = va_0(i_13, 6U, &kernelContext_63);
    float3 _S292 = va_0(i_13, 2U, &kernelContext_63);
    float3 _S293 = float3(alpha_0) ;
    float3 _S294 = _S291 + _S292 * _S293;
    float3 _S295 = vl_0(i_13, 6U, &kernelContext_63);
    float3 _S296 = vl_0(i_13, 2U, &kernelContext_63);
    set_v_0(i_13, 6U, _S294, _S295 + _S296 * _S293, &kernelContext_63);
    float3 _S297 = va_0(i_13, 0U, &kernelContext_63);
    float3 _S298 = va_0(i_13, 4U, &kernelContext_63);
    float3 _S299 = _S297 - _S298 * _S293;
    float3 _S300 = vl_0(i_13, 0U, &kernelContext_63);
    float3 _S301 = vl_0(i_13, 4U, &kernelContext_63);
    set_v_0(i_13, 0U, _S299, _S300 - _S301 * _S293, &kernelContext_63);
    return;
}

[[kernel]] void w_final(uint3 group_28 [[threadgroup_position_in_grid]], uint3 thread_23 [[thread_position_in_threadgroup]], atomic<uint> device* meta_37 [[buffer(11)]], uint device* topo_args_37 [[buffer(12)]], packed_uint4 device* islands_37 [[buffer(1)]], BlastParams_0 constant* params_37 [[buffer(0)]], packed_float4 device* wide_37 [[buffer(10)]], uint device* lists_37 [[buffer(9)]], BlastNode_natural_0 device* nodes_37 [[buffer(2)]], packed_float4 device* accel_37 [[buffer(5)]], packed_float4 device* scratch_37 [[buffer(7)]], uint device* refs_37 [[buffer(3)]], BlastBond_natural_0 device* bonds_37 [[buffer(4)]], packed_float4 device* x_37 [[buffer(6)]], packed_uint4 device* report_37 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_64;
    (&kernelContext_64)->meta_0 = meta_37;
    (&kernelContext_64)->topo_args_0 = topo_args_37;
    (&kernelContext_64)->islands_0 = islands_37;
    (&kernelContext_64)->params_0 = params_37;
    (&kernelContext_64)->wide_0 = wide_37;
    (&kernelContext_64)->lists_0 = lists_37;
    (&kernelContext_64)->nodes_0 = nodes_37;
    (&kernelContext_64)->accel_0 = accel_37;
    (&kernelContext_64)->scratch_0 = scratch_37;
    (&kernelContext_64)->refs_0 = refs_37;
    (&kernelContext_64)->bonds_0 = bonds_37;
    (&kernelContext_64)->x_0 = x_37;
    (&kernelContext_64)->report_0 = report_37;
    threadgroup array<float2, int(32)> g_blast_part_37;
    (&kernelContext_64)->g_blast_part_0 = &g_blast_part_37;
    threadgroup array<float, int(1024)> g_wide_max_37;
    (&kernelContext_64)->g_wide_max_0 = &g_wide_max_37;
    threadgroup float g_wide_max_out_37;
    (&kernelContext_64)->g_wide_max_out_0 = &g_wide_max_out_37;
    uint tid_33 = thread_23.x;
    Block_0 _S302 = bond_block_0(group_28, tid_33, &kernelContext_64);
    bool _S303;
    if((_S302.begin_0) >= (_S302.end_0))
    {
        _S303 = true;
    }
    else
    {
        _S303 = ((_S302.flags_0) & 128U) != 0U;
    }
    if(_S303)
    {
        return;
    }
    uint p_18 = _S302.begin_0 + tid_33;
    if(p_18 >= (_S302.end_0))
    {
        return;
    }
    uint j_7 = (&kernelContext_64)->lists_0[p_18];
    thread BlastBond_natural_0 _S304 = *((&kernelContext_64)->bonds_0+j_7);
    if(!(((&_S304)->health_0) > 0.0f))
    {
        _S303 = true;
    }
    else
    {
        _S303 = ((_S302.flags_0) & 16U) != 0U;
    }
    if(_S303)
    {
        uint _S305 = j_7 * 2U;
        packed_float4 _S306 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_64)->x_0+_S305) = _S306;
        *((&kernelContext_64)->x_0+(_S305 + 1U)) = _S306;
        return;
    }
    thread float3 ta_6;
    thread float3 tl_6;
    bond_term_1(&_S304, 6U, &ta_6, &tl_6, &kernelContext_64);
    uint _S307 = j_7 * 2U;
    *((&kernelContext_64)->x_0+_S307) = packed_float4(float4((float4(*((&kernelContext_64)->x_0+_S307)) ).xyz + ta_6, 0.0f)) ;
    uint _S308 = _S307 + 1U;
    *((&kernelContext_64)->x_0+_S308) = packed_float4(float4((float4(*((&kernelContext_64)->x_0+_S308)) ).xyz + tl_6, 0.0f)) ;
    return;
}

[[kernel]] void w_report(uint3 group_29 [[threadgroup_position_in_grid]], atomic<uint> device* meta_38 [[buffer(11)]], uint device* topo_args_38 [[buffer(12)]], packed_uint4 device* islands_38 [[buffer(1)]], BlastParams_0 constant* params_38 [[buffer(0)]], packed_float4 device* wide_38 [[buffer(10)]], uint device* lists_38 [[buffer(9)]], BlastNode_natural_0 device* nodes_38 [[buffer(2)]], packed_float4 device* accel_38 [[buffer(5)]], packed_float4 device* scratch_38 [[buffer(7)]], uint device* refs_38 [[buffer(3)]], BlastBond_natural_0 device* bonds_38 [[buffer(4)]], packed_float4 device* x_38 [[buffer(6)]], packed_uint4 device* report_38 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_65;
    (&kernelContext_65)->meta_0 = meta_38;
    (&kernelContext_65)->topo_args_0 = topo_args_38;
    (&kernelContext_65)->islands_0 = islands_38;
    (&kernelContext_65)->params_0 = params_38;
    (&kernelContext_65)->wide_0 = wide_38;
    (&kernelContext_65)->lists_0 = lists_38;
    (&kernelContext_65)->nodes_0 = nodes_38;
    (&kernelContext_65)->accel_0 = accel_38;
    (&kernelContext_65)->scratch_0 = scratch_38;
    (&kernelContext_65)->refs_0 = refs_38;
    (&kernelContext_65)->bonds_0 = bonds_38;
    (&kernelContext_65)->x_0 = x_38;
    (&kernelContext_65)->report_0 = report_38;
    threadgroup array<float2, int(32)> g_blast_part_38;
    (&kernelContext_65)->g_blast_part_0 = &g_blast_part_38;
    threadgroup array<float, int(1024)> g_wide_max_38;
    (&kernelContext_65)->g_wide_max_0 = &g_wide_max_38;
    threadgroup float g_wide_max_out_38;
    (&kernelContext_65)->g_wide_max_out_0 = &g_wide_max_out_38;
    uint row_20 = group_29.x;
    uint _S309 = row_island_0(row_20, &kernelContext_65);
    uint _S310 = flags_of_0(row_20, &kernelContext_65);
    uint4 _S311 = uint4(*((&kernelContext_65)->report_0+_S309)) ;
    if((_S310 & 128U) != 0U)
    {
        *((&kernelContext_65)->report_0+_S309) = packed_uint4(uint4(0U, 7U, _S311.z, _S311.w)) ;
        return;
    }
    bool ok_1;
    if((_S310 & 2U) != 0U)
    {
        ok_1 = (_S310 & 4U) == 0U;
    }
    else
    {
        ok_1 = false;
    }
    uint _S312 = iteration_of_0(row_20, &kernelContext_65);
    uint _S313;
    if(ok_1)
    {
        _S313 = 1U;
    }
    else
    {
        _S313 = 0U;
    }
    if(ok_1)
    {
        ok_1 = ((&kernelContext_65)->params_0->warm_0) != 0U;
    }
    else
    {
        ok_1 = false;
    }
    if(ok_1)
    {
        ok_1 = _S312 == 0U;
    }
    else
    {
        ok_1 = false;
    }
    uint _S314;
    if(ok_1)
    {
        _S314 = 4U;
    }
    else
    {
        _S314 = 0U;
    }
    float4 _S315 = float4(*((&kernelContext_65)->wide_0+st_0(row_20, 0U))) ;
    *((&kernelContext_65)->report_0+_S309) = packed_uint4(uint4(_S312, _S313 | _S314, (as_type<uint>((_S315.y))), (as_type<uint>((_S315.x))))) ;
    return;
}

