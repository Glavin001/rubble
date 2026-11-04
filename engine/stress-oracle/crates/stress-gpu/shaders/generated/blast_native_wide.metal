#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

template<typename T, int A, int B>
matrix<T,A,B> _slang_matrixCompMult(matrix<T,A,B> m1, matrix<T,A,B> m2)
{
    matrix<T,A,B> result;
    for (int i = 0; i < A; i++)
        result[i] = m1[i] * m2[i];
    return result;
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
    float force_tolerance_0;
    float4 gravity_0;
    uint w_alive_0;
    uint a_dirty_0;
    uint a_counters_0;
    uint wide_min_0;
    uint wide_rows_0;
    uint wide_blocks_0;
    uint opaque_zero_0;
    uint first_polynomial_0;
    uint rotation_0;
    uint node_vectors_0;
    uint carry_0;
    uint section_bending_0;
    uint chunk_loads_0;
    uint r_virial_0;
    uint a_events_0;
    uint ev_broken_cap_0;
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
    packed_float4 device* stiffness_0;
    packed_float4 device* x_0;
    packed_uint4 device* report_0;
    array<float2, int(32)> threadgroup* g_blast_part_0;
    array<float, int(1024)> threadgroup* g_wide_max_0;
    float threadgroup* g_wide_max_out_0;
};

[[kernel]] void w_reset(uint3 id_0 [[thread_position_in_grid]], atomic<uint> device* meta_1 [[buffer(12)]], uint device* topo_args_1 [[buffer(13)]], packed_uint4 device* islands_1 [[buffer(1)]], BlastParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* wide_1 [[buffer(11)]], uint device* lists_1 [[buffer(9)]], BlastNode_natural_0 device* nodes_1 [[buffer(2)]], packed_float4 device* accel_1 [[buffer(5)]], packed_float4 device* scratch_1 [[buffer(7)]], uint device* refs_1 [[buffer(3)]], BlastBond_natural_0 device* bonds_1 [[buffer(4)]], packed_float4 device* stiffness_1 [[buffer(10)]], packed_float4 device* x_1 [[buffer(6)]], packed_uint4 device* report_1 [[buffer(8)]])
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
    (&kernelContext_0)->stiffness_0 = stiffness_1;
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

[[kernel]] void w_classify(uint3 id_1 [[thread_position_in_grid]], atomic<uint> device* meta_2 [[buffer(12)]], uint device* topo_args_2 [[buffer(13)]], packed_uint4 device* islands_2 [[buffer(1)]], BlastParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* wide_2 [[buffer(11)]], uint device* lists_2 [[buffer(9)]], BlastNode_natural_0 device* nodes_2 [[buffer(2)]], packed_float4 device* accel_2 [[buffer(5)]], packed_float4 device* scratch_2 [[buffer(7)]], uint device* refs_2 [[buffer(3)]], BlastBond_natural_0 device* bonds_2 [[buffer(4)]], packed_float4 device* stiffness_2 [[buffer(10)]], packed_float4 device* x_2 [[buffer(6)]], packed_uint4 device* report_2 [[buffer(8)]])
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
    (&kernelContext_1)->stiffness_0 = stiffness_2;
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

[[kernel]] void w_args(uint3 id_2 [[thread_position_in_grid]], atomic<uint> device* meta_3 [[buffer(12)]], uint device* topo_args_3 [[buffer(13)]], packed_uint4 device* islands_3 [[buffer(1)]], BlastParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* wide_3 [[buffer(11)]], uint device* lists_3 [[buffer(9)]], BlastNode_natural_0 device* nodes_3 [[buffer(2)]], packed_float4 device* accel_3 [[buffer(5)]], packed_float4 device* scratch_3 [[buffer(7)]], uint device* refs_3 [[buffer(3)]], BlastBond_natural_0 device* bonds_3 [[buffer(4)]], packed_float4 device* stiffness_3 [[buffer(10)]], packed_float4 device* x_3 [[buffer(6)]], packed_uint4 device* report_3 [[buffer(8)]])
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
    (&kernelContext_2)->stiffness_0 = stiffness_3;
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

uint ns_0(uint n_1, uint v_0, KernelContext_0 thread* kernelContext_6)
{
    return n_1 * kernelContext_6->params_0->node_vectors_0 + v_0;
}

float norm2_0(float3 a_0, float3 b_1)
{
    return dot(a_0, a_0) + dot(b_1, b_1);
}

float3 va_0(uint n_2, uint v_1, KernelContext_0 thread* kernelContext_7)
{
    uint _S16 = ns_0(n_2, v_1, kernelContext_7);
    return (float4(*(kernelContext_7->scratch_0+_S16)) ).xyz;
}

float3 vl_0(uint n_3, uint v_2, KernelContext_0 thread* kernelContext_8)
{
    uint _S17 = ns_0(n_3, v_2, kernelContext_8);
    return (float4(*(kernelContext_8->scratch_0+(_S17 + 1U))) ).xyz;
}

void set_v_0(uint n_4, uint v_3, float3 a_1, float3 l_0, KernelContext_0 thread* kernelContext_9)
{
    uint _S18 = ns_0(n_4, v_3, kernelContext_9);
    *(kernelContext_9->scratch_0+_S18) = packed_float4(float4(a_1, 0.0f)) ;
    *(kernelContext_9->scratch_0+(_S18 + 1U)) = packed_float4(float4(l_0, 0.0f)) ;
    return;
}

float3 stiff_0(uint j_0, uint which_0, float3 v_4, KernelContext_0 thread* kernelContext_10)
{
    uint _S19 = 6U * which_0;
    thread array<float, int(6)> m_0;
    uint k_1 = 0U;
    for(;;)
    {
        if(k_1 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S20 = _S19 + k_1;
        m_0[k_1] = (float4(*(kernelContext_10->stiffness_0+(j_0 * 6U + _S20 / 4U))) )[_S20 % 4U];
        k_1 = k_1 + 1U;
    }
    float _S21 = v_4.x;
    float _S22 = v_4.y;
    float _S23 = v_4.z;
    return float3(m_0[int(0)] * _S21 + m_0[int(3)] * _S22 + m_0[int(4)] * _S23, m_0[int(3)] * _S21 + m_0[int(1)] * _S22 + m_0[int(5)] * _S23, m_0[int(4)] * _S21 + m_0[int(5)] * _S22 + m_0[int(2)] * _S23);
}

float3 stiff_1(uint j_1, uint which_1, float3 v_5, KernelContext_0 thread* kernelContext_11)
{
    uint _S24 = 6U * which_1;
    thread array<float, int(6)> m_1;
    uint k_2 = 0U;
    for(;;)
    {
        if(k_2 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S25 = _S24 + k_2;
        m_1[k_2] = (float4(*(kernelContext_11->stiffness_0+(j_1 * 6U + _S25 / 4U))) )[_S25 % 4U];
        k_2 = k_2 + 1U;
    }
    float _S26 = v_5.x;
    float _S27 = v_5.y;
    float _S28 = v_5.z;
    return float3(m_1[int(0)] * _S26 + m_1[int(3)] * _S27 + m_1[int(4)] * _S28, m_1[int(3)] * _S26 + m_1[int(1)] * _S27 + m_1[int(5)] * _S28, m_1[int(4)] * _S26 + m_1[int(5)] * _S27 + m_1[int(2)] * _S28);
}

matrix<float,int(3),int(3)>  sym3_0(uint j_2, uint which_2, KernelContext_0 thread* kernelContext_12)
{
    float3 _S29 = stiff_1(j_2, which_2, float3(1.0f, 0.0f, 0.0f), kernelContext_12);
    float3 _S30 = stiff_1(j_2, which_2, float3(0.0f, 1.0f, 0.0f), kernelContext_12);
    float3 _S31 = stiff_1(j_2, which_2, float3(0.0f, 0.0f, 1.0f), kernelContext_12);
    return matrix<float,int(3),int(3)> (_S29.x, _S30.x, _S31.x, _S29.y, _S30.y, _S31.y, _S29.z, _S30.z, _S31.z);
}

matrix<float,int(3),int(3)>  skew_0(float3 o_0)
{
    float _S32 = o_0.z;
    float _S33 = o_0.y;
    float _S34 = o_0.x;
    return matrix<float,int(3),int(3)> (0.0f, - _S32, _S33, _S32, 0.0f, - _S34, - _S33, _S34, 0.0f);
}

bool build_full_block_0(uint node_0, KernelContext_0 thread* kernelContext_13)
{
    uint c_0;
    uint k_3;
    float s_1;
    BlastNode_natural_0 n_5 = kernelContext_13->nodes_0[node_0];
    matrix<float,int(3),int(3)>  aa_0 = matrix<float,int(3),int(3)> (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    uint r_0 = n_5.ref_begin_0;
    matrix<float,int(3),int(3)>  aa_1 = aa_0;
    matrix<float,int(3),int(3)>  al_0 = aa_0;
    matrix<float,int(3),int(3)>  ll_0 = aa_0;
    for(;;)
    {
        if(r_0 < (n_5.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_0 = kernelContext_13->refs_0[r_0];
        uint j_3 = ref_0 & 2147483647U;
        BlastBond_natural_0 device* _S35 = kernelContext_13->bonds_0+j_3;
        BlastBond_natural_0 b_2 = *_S35;
        if(!(((*_S35).health_0) > 0.0f))
        {
            r_0 = r_0 + 1U;
            continue;
        }
        float3 o_1;
        if((ref_0 & 2147483648U) != 0U)
        {
            o_1 = (float4(b_2.o1_0) ).xyz;
        }
        else
        {
            o_1 = (float4(b_2.o0_0) ).xyz;
        }
        float _S36 = b_2.s_0;
        float s2_0 = _S36 * _S36;
        matrix<float,int(3),int(3)>  _S37 = sym3_0(j_3, 3U, kernelContext_13);
        matrix<float,int(3),int(3)>  k_4 = skew_0(o_1);
        matrix<float,int(3),int(3)>  _S38 = sym3_0(j_3, 1U, kernelContext_13);
        matrix<float,int(3),int(3)>  _S39 = (((_S37) * (k_4)));
        matrix<float,int(3),int(3)>  _S40 = matrix<float,int(3),int(3)> (s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0);
        matrix<float,int(3),int(3)>  _S41 = al_0 + _slang_matrixCompMult(matrix<float,int(3),int(3)> (- s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0), _S39);
        matrix<float,int(3),int(3)>  _S42 = ll_0 + _slang_matrixCompMult(_S40, _S37);
        aa_1 = aa_1 + _slang_matrixCompMult(_S40, _S38 + (((transpose(k_4)) * (_S39))));
        al_0 = _S41;
        ll_0 = _S42;
        r_0 = r_0 + 1U;
    }
    thread array<array<float, int(6)>, int(6)> m_2;
    r_0 = 0U;
    for(;;)
    {
        if(r_0 < 3U)
        {
        }
        else
        {
            break;
        }
        c_0 = 0U;
        for(;;)
        {
            if(c_0 < 3U)
            {
            }
            else
            {
                break;
            }
            m_2[r_0][c_0] = aa_1[r_0][c_0] * n_5.w_ang_0 * n_5.w_ang_0;
            uint _S43 = c_0 + 3U;
            float _S44 = al_0[r_0][c_0] * n_5.w_ang_0 * n_5.w_lin_0;
            m_2[r_0][_S43] = _S44;
            m_2[_S43][r_0] = _S44;
            m_2[r_0 + 3U][_S43] = ll_0[r_0][c_0] * n_5.w_lin_0 * n_5.w_lin_0;
            c_0 = c_0 + 1U;
        }
        r_0 = r_0 + 1U;
    }
    thread array<float, int(24)> packed_0;
    uint z_0 = 0U;
    for(;;)
    {
        if(z_0 < 24U)
        {
        }
        else
        {
            break;
        }
        packed_0[z_0] = 0.0f;
        z_0 = z_0 + 1U;
    }
    r_0 = 0U;
    for(;;)
    {
        if(r_0 < 6U)
        {
        }
        else
        {
            break;
        }
        c_0 = 0U;
        for(;;)
        {
            if(c_0 <= r_0)
            {
            }
            else
            {
                break;
            }
            packed_0[r_0 * (r_0 + 1U) / 2U + c_0] = m_2[r_0][c_0];
            c_0 = c_0 + 1U;
        }
        r_0 = r_0 + 1U;
    }
    uint q_0 = 0U;
    for(;;)
    {
        if(q_0 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S45 = ns_0(node_0, 18U + q_0, kernelContext_13);
        uint _S46 = 4U * q_0;
        *(kernelContext_13->scratch_0+_S45) = packed_float4(float4(packed_0[_S46], packed_0[_S46 + 1U], packed_0[_S46 + 2U], packed_0[_S46 + 3U])) ;
        q_0 = q_0 + 1U;
    }
    thread array<array<float, int(6)>, int(6)> l_1;
    r_0 = 0U;
    for(;;)
    {
        if(r_0 < 6U)
        {
        }
        else
        {
            break;
        }
        c_0 = 0U;
        for(;;)
        {
            if(c_0 < 6U)
            {
            }
            else
            {
                break;
            }
            l_1[r_0][c_0] = 0.0f;
            c_0 = c_0 + 1U;
        }
        r_0 = r_0 + 1U;
    }
    r_0 = 0U;
    for(;;)
    {
        if(r_0 < 6U)
        {
        }
        else
        {
            break;
        }
        c_0 = 0U;
        for(;;)
        {
            if(c_0 <= r_0)
            {
            }
            else
            {
                break;
            }
            k_3 = 0U;
            s_1 = m_2[r_0][c_0];
            for(;;)
            {
                if(k_3 < c_0)
                {
                }
                else
                {
                    break;
                }
                float s_2 = s_1 - l_1[r_0][k_3] * l_1[c_0][k_3];
                k_3 = k_3 + 1U;
                s_1 = s_2;
            }
            if(r_0 == c_0)
            {
                bool _S47;
                if(!(s_1 > 0.0f))
                {
                    _S47 = true;
                }
                else
                {
                    _S47 = !isfinite(s_1);
                }
                if(_S47)
                {
                    return false;
                }
                l_1[r_0][c_0] = sqrt(s_1);
            }
            else
            {
                l_1[r_0][c_0] = s_1 / l_1[c_0][c_0];
            }
            c_0 = c_0 + 1U;
        }
        r_0 = r_0 + 1U;
    }
    thread array<array<float, int(6)>, int(6)> inv_0;
    c_0 = 0U;
    for(;;)
    {
        if(c_0 < 6U)
        {
        }
        else
        {
            break;
        }
        thread array<float, int(6)> y_0;
        r_0 = 0U;
        for(;;)
        {
            if(r_0 < 6U)
            {
            }
            else
            {
                break;
            }
            if(r_0 == c_0)
            {
                s_1 = 1.0f;
            }
            else
            {
                s_1 = 0.0f;
            }
            k_3 = 0U;
            float s_3 = s_1;
            for(;;)
            {
                if(k_3 < r_0)
                {
                }
                else
                {
                    break;
                }
                float s_4 = s_3 - l_1[r_0][k_3] * y_0[k_3];
                k_3 = k_3 + 1U;
                s_3 = s_4;
            }
            y_0[r_0] = s_3 / l_1[r_0][r_0];
            r_0 = r_0 + 1U;
        }
        int r_1 = int(5);
        for(;;)
        {
            if(r_1 >= int(0))
            {
            }
            else
            {
                break;
            }
            k_3 = uint(r_1) + 1U;
            s_1 = y_0[r_1];
            for(;;)
            {
                if(k_3 < 6U)
                {
                }
                else
                {
                    break;
                }
                float s_5 = s_1 - l_1[k_3][r_1] * inv_0[k_3][c_0];
                k_3 = k_3 + 1U;
                s_1 = s_5;
            }
            inv_0[r_1][c_0] = s_1 / l_1[r_1][r_1];
            r_1 = r_1 - int(1);
        }
        c_0 = c_0 + 1U;
    }
    r_0 = 0U;
    for(;;)
    {
        if(r_0 < 6U)
        {
        }
        else
        {
            break;
        }
        c_0 = 0U;
        for(;;)
        {
            if(c_0 <= r_0)
            {
            }
            else
            {
                break;
            }
            packed_0[r_0 * (r_0 + 1U) / 2U + c_0] = inv_0[r_0][c_0];
            c_0 = c_0 + 1U;
        }
        r_0 = r_0 + 1U;
    }
    q_0 = 0U;
    for(;;)
    {
        if(q_0 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S48 = ns_0(node_0, 24U + q_0, kernelContext_13);
        uint _S49 = 4U * q_0;
        *(kernelContext_13->scratch_0+_S48) = packed_float4(float4(packed_0[_S49], packed_0[_S49 + 1U], packed_0[_S49 + 2U], packed_0[_S49 + 3U])) ;
        q_0 = q_0 + 1U;
    }
    return true;
}

bool build_block_0(uint node_1, KernelContext_0 thread* kernelContext_14)
{
    if((kernelContext_14->params_0->rotation_0) == 2U)
    {
        bool _S50 = build_full_block_0(node_1, kernelContext_14);
        return _S50;
    }
    BlastNode_natural_0 n_6 = kernelContext_14->nodes_0[node_1];
    float3 _S51 = float3(0.0f) ;
    uint r_2 = n_6.ref_begin_0;
    float3 diag_0 = _S51;
    float3 offd_0 = _S51;
    float3 ks_0 = _S51;
    float cs_0 = 0.0f;
    for(;;)
    {
        if(r_2 < (n_6.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_1 = kernelContext_14->refs_0[r_2];
        uint j_4 = ref_1 & 2147483647U;
        BlastBond_natural_0 device* _S52 = kernelContext_14->bonds_0+j_4;
        BlastBond_natural_0 b_3 = *_S52;
        if(!(((*_S52).health_0) > 0.0f))
        {
            r_2 = r_2 + 1U;
            continue;
        }
        float3 o_2;
        if((ref_1 & 2147483648U) != 0U)
        {
            o_2 = (float4(b_3.o1_0) ).xyz;
        }
        else
        {
            o_2 = (float4(b_3.o0_0) ).xyz;
        }
        float _S53 = b_3.s_0;
        float s2_1 = _S53 * _S53;
        float3 diag_1;
        float3 offd_1;
        if((kernelContext_14->params_0->rotation_0) != 0U)
        {
            float3 _S54 = stiff_0(j_4, 1U, float3(1.0f, 0.0f, 0.0f), kernelContext_14);
            float3 _S55 = stiff_0(j_4, 1U, float3(0.0f, 1.0f, 0.0f), kernelContext_14);
            float3 _S56 = stiff_0(j_4, 1U, float3(0.0f, 0.0f, 1.0f), kernelContext_14);
            float3 _S57 = float3(s2_1) ;
            float _S58 = o_2.x;
            float _S59 = o_2.y;
            float _S60 = o_2.z;
            float3 offd_2 = offd_0 + _S57 * (float3(_S55.x, _S56.x, _S56.y) - float3(_S58 * _S59, _S58 * _S60, _S59 * _S60));
            diag_1 = diag_0 + _S57 * (float3(_S54.x, _S55.y, _S56.z) + float3(dot(o_2, o_2))  - o_2 * o_2);
            offd_1 = offd_2;
        }
        else
        {
            float3 _S61 = float3(s2_1) ;
            float _S62 = o_2.x;
            float _S63 = o_2.y;
            float _S64 = o_2.z;
            float3 offd_3 = offd_0 + _S61 * - float3(_S62 * _S63, _S62 * _S64, _S63 * _S64);
            diag_1 = diag_0 + _S61 * (float3((1.0f + dot(o_2, o_2)))  - o_2 * o_2);
            offd_1 = offd_3;
        }
        float3 ks_1 = ks_0 + float3(s2_1)  * o_2;
        float cs_1 = cs_0 + s2_1;
        diag_0 = diag_1;
        offd_0 = offd_1;
        ks_0 = ks_1;
        cs_0 = cs_1;
        r_2 = r_2 + 1U;
    }
    float _S65 = n_6.w_ang_0;
    float da_0 = _S65 * _S65;
    float c_1 = n_6.w_lin_0 * n_6.w_lin_0 * cs_0;
    float3 k_5 = ks_0 * float3((n_6.w_ang_0 * n_6.w_lin_0)) ;
    float4 a0_0 = float4(diag_0 * float3(da_0) , offd_0.x * da_0);
    float _S66 = offd_0.y * da_0;
    float _S67 = offd_0.z * da_0;
    float4 a1_0 = float4(_S66, _S67, c_1, 0.0f);
    uint _S68 = ns_0(node_1, 18U, kernelContext_14);
    *(kernelContext_14->scratch_0+_S68) = packed_float4(a0_0) ;
    *(kernelContext_14->scratch_0+(_S68 + 1U)) = packed_float4(a1_0) ;
    *(kernelContext_14->scratch_0+(_S68 + 2U)) = packed_float4(float4(k_5, 0.0f)) ;
    if(!(c_1 > 0.0f))
    {
        return false;
    }
    float ic_0 = 1.0f / c_1;
    float kk_0 = dot(k_5, k_5);
    float _S69 = k_5.x;
    float sxx_0 = a0_0.x + (_S69 * _S69 - kk_0) * ic_0;
    float _S70 = k_5.y;
    float syy_0 = a0_0.y + (_S70 * _S70 - kk_0) * ic_0;
    float _S71 = k_5.z;
    float szz_0 = a0_0.z + (_S71 * _S71 - kk_0) * ic_0;
    float sxy_0 = a0_0.w + _S69 * _S70 * ic_0;
    float sxz_0 = _S66 + _S69 * _S71 * ic_0;
    float syz_0 = _S67 + _S70 * _S71 * ic_0;
    float cxx_0 = syy_0 * szz_0 - syz_0 * syz_0;
    float cxy_0 = sxz_0 * syz_0 - sxy_0 * szz_0;
    float cxz_0 = sxy_0 * syz_0 - sxz_0 * syy_0;
    float det_0 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    bool _S72;
    if(!(det_0 > 0.0f))
    {
        _S72 = true;
    }
    else
    {
        _S72 = !isfinite(det_0);
    }
    if(_S72)
    {
        return false;
    }
    float id_3 = 1.0f / det_0;
    float cyy_0 = sxx_0 * szz_0 - sxz_0 * sxz_0;
    float cyz_0 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    float czz_0 = sxx_0 * syy_0 - sxy_0 * sxy_0;
    uint _S73 = ns_0(node_1, 21U, kernelContext_14);
    *(kernelContext_14->scratch_0+_S73) = packed_float4(float4(cxx_0 * id_3, cyy_0 * id_3, czz_0 * id_3, cxy_0 * id_3)) ;
    *(kernelContext_14->scratch_0+(_S73 + 1U)) = packed_float4(float4(cxz_0 * id_3, cyz_0 * id_3, ic_0, 0.0f)) ;
    *(kernelContext_14->scratch_0+(_S73 + 2U)) = packed_float4(float4(k_5 * float3(ic_0) , 0.0f)) ;
    return true;
}

float2 blast_group_sum_0(uint tid_1, float2 v_6, KernelContext_0 thread* kernelContext_15)
{
    float2 w_0 = simd_sum(v_6);
    if((tid_1 % 32U) == 0U)
    {
        (*kernelContext_15->g_blast_part_0)[tid_1 / 32U] = w_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 _S74 = (*kernelContext_15->g_blast_part_0)[int(0)];
    uint k_6 = 1U;
    float2 s_6 = _S74;
    for(;;)
    {
        if(k_6 < 32U)
        {
        }
        else
        {
            break;
        }
        float2 s_7 = s_6 + (*kernelContext_15->g_blast_part_0)[k_6];
        k_6 = k_6 + 1U;
        s_6 = s_7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s_6;
}

uint pt_0(uint row_5, uint block_0, uint k_7, KernelContext_0 thread* kernelContext_16)
{
    return kernelContext_16->params_0->wide_rows_0 * 16U + (row_5 * kernelContext_16->params_0->wide_blocks_0 + block_0) * 6U + k_7;
}

void write_partial_0(uint tid_2, uint row_6, uint block_1, uint k_8, float4 v_7, KernelContext_0 thread* kernelContext_17)
{
    float2 _S75 = blast_group_sum_0(tid_2, v_7.xy, kernelContext_17);
    float2 _S76 = blast_group_sum_0(tid_2, v_7.zw, kernelContext_17);
    if(tid_2 == 0U)
    {
        uint _S77 = pt_0(row_6, block_1, k_8, kernelContext_17);
        *(kernelContext_17->wide_0+_S77) = packed_float4(float4(_S75, _S76)) ;
    }
    return;
}

[[kernel]] void w_init_nodes(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], atomic<uint> device* meta_4 [[buffer(12)]], uint device* topo_args_4 [[buffer(13)]], packed_uint4 device* islands_4 [[buffer(1)]], BlastParams_0 constant* params_4 [[buffer(0)]], packed_float4 device* wide_4 [[buffer(11)]], uint device* lists_4 [[buffer(9)]], BlastNode_natural_0 device* nodes_4 [[buffer(2)]], packed_float4 device* accel_4 [[buffer(5)]], packed_float4 device* scratch_4 [[buffer(7)]], uint device* refs_4 [[buffer(3)]], BlastBond_natural_0 device* bonds_4 [[buffer(4)]], packed_float4 device* stiffness_4 [[buffer(10)]], packed_float4 device* x_4 [[buffer(6)]], packed_uint4 device* report_4 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_18;
    (&kernelContext_18)->meta_0 = meta_4;
    (&kernelContext_18)->topo_args_0 = topo_args_4;
    (&kernelContext_18)->islands_0 = islands_4;
    (&kernelContext_18)->params_0 = params_4;
    (&kernelContext_18)->wide_0 = wide_4;
    (&kernelContext_18)->lists_0 = lists_4;
    (&kernelContext_18)->nodes_0 = nodes_4;
    (&kernelContext_18)->accel_0 = accel_4;
    (&kernelContext_18)->scratch_0 = scratch_4;
    (&kernelContext_18)->refs_0 = refs_4;
    (&kernelContext_18)->bonds_0 = bonds_4;
    (&kernelContext_18)->stiffness_0 = stiffness_4;
    (&kernelContext_18)->x_0 = x_4;
    (&kernelContext_18)->report_0 = report_4;
    threadgroup array<float2, int(32)> g_blast_part_4;
    (&kernelContext_18)->g_blast_part_0 = &g_blast_part_4;
    threadgroup array<float, int(1024)> g_wide_max_4;
    (&kernelContext_18)->g_wide_max_0 = &g_wide_max_4;
    threadgroup float g_wide_max_out_4;
    (&kernelContext_18)->g_wide_max_out_0 = &g_wide_max_out_4;
    uint tid_3 = thread_1.x;
    Block_0 _S78 = node_block_0(group_1, tid_3, &kernelContext_18);
    if((_S78.begin_0) >= (_S78.end_0))
    {
        return;
    }
    thread float4 p0_0 = float4(0.0f) ;
    uint p_0 = _S78.begin_0 + tid_3;
    if(p_0 < (_S78.end_0))
    {
        uint i_0 = (&kernelContext_18)->lists_0[p_0];
        BlastNode_natural_0 n_7 = (&kernelContext_18)->nodes_0[i_0];
        uint _S79 = i_0 * 2U;
        float4 _S80 = float4(*((&kernelContext_18)->accel_0+_S79)) ;
        float4 _S81 = float4(*((&kernelContext_18)->accel_0+(_S79 + 1U))) ;
        uint _S82 = ns_0(i_0, 16U, &kernelContext_18);
        bool _S83;
        if(any(_S80 != (float4(*((&kernelContext_18)->scratch_0+_S82)) )))
        {
            _S83 = true;
        }
        else
        {
            _S83 = any(_S81 != (float4(*((&kernelContext_18)->scratch_0+(_S82 + 1U))) ));
        }
        if(_S83)
        {
            p0_0.z = 1.0f;
        }
        float3 _S84 = - _S80.xyz;
        float _S85;
        if((n_7.w_ang_0) > 0.0f)
        {
            _S85 = 1.0f / n_7.w_ang_0;
        }
        else
        {
            _S85 = 1.0f;
        }
        float3 ba_0 = _S84 * float3(_S85) ;
        float3 _S86 = - _S81.xyz;
        if((n_7.w_lin_0) > 0.0f)
        {
            _S85 = (&kernelContext_18)->params_0->recip_length_0 / n_7.w_lin_0;
        }
        else
        {
            _S85 = (&kernelContext_18)->params_0->recip_length_0;
        }
        float3 bl_0 = _S86 * float3(_S85) ;
        p0_0.x = norm2_0(ba_0, bl_0);
        float3 _S87 = va_0(i_0, 14U, &kernelContext_18);
        float3 _S88 = ba_0 - _S87;
        float3 _S89 = vl_0(i_0, 14U, &kernelContext_18);
        p0_0.w = norm2_0(_S88, bl_0 - _S89);
        set_v_0(i_0, 14U, ba_0, bl_0, &kernelContext_18);
        set_v_0(i_0, 0U, ba_0, bl_0, &kernelContext_18);
        *((&kernelContext_18)->scratch_0+_S82) = packed_float4(_S80) ;
        *((&kernelContext_18)->scratch_0+(_S82 + 1U)) = packed_float4(_S81) ;
        float3 _S90 = float3(0.0f) ;
        set_v_0(i_0, 6U, _S90, _S90, &kernelContext_18);
        if(((&kernelContext_18)->params_0->carry_0) == 0U)
        {
            set_v_0(i_0, 2U, _S90, _S90, &kernelContext_18);
        }
        bool _S91 = build_block_0(i_0, &kernelContext_18);
        if(!_S91)
        {
            p0_0.y = 1.0f;
        }
    }
    write_partial_0(tid_3, _S78.row_4, group_1.x, 0U, p0_0, &kernelContext_18);
    return;
}

Block_0 bond_block_0(uint3 group_2, uint tid_4, KernelContext_0 thread* kernelContext_19)
{
    Block_0 _S92 = node_block_0(group_2, tid_4, kernelContext_19);
    thread Block_0 b_4 = _S92;
    uint4 g_1 = group_uniform4_0(uint4(*(kernelContext_19->islands_0+(&b_4)->island_0)) , tid_4);
    uint _S93 = g_1.z + group_2.x * 1024U;
    (&b_4)->begin_0 = _S93;
    (&b_4)->end_0 = min(_S93 + 1024U, g_1.w);
    return b_4;
}

[[kernel]] void w_init_bonds(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], atomic<uint> device* meta_5 [[buffer(12)]], uint device* topo_args_5 [[buffer(13)]], packed_uint4 device* islands_5 [[buffer(1)]], BlastParams_0 constant* params_5 [[buffer(0)]], packed_float4 device* wide_5 [[buffer(11)]], uint device* lists_5 [[buffer(9)]], BlastNode_natural_0 device* nodes_5 [[buffer(2)]], packed_float4 device* accel_5 [[buffer(5)]], packed_float4 device* scratch_5 [[buffer(7)]], uint device* refs_5 [[buffer(3)]], BlastBond_natural_0 device* bonds_5 [[buffer(4)]], packed_float4 device* stiffness_5 [[buffer(10)]], packed_float4 device* x_5 [[buffer(6)]], packed_uint4 device* report_5 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_20;
    (&kernelContext_20)->meta_0 = meta_5;
    (&kernelContext_20)->topo_args_0 = topo_args_5;
    (&kernelContext_20)->islands_0 = islands_5;
    (&kernelContext_20)->params_0 = params_5;
    (&kernelContext_20)->wide_0 = wide_5;
    (&kernelContext_20)->lists_0 = lists_5;
    (&kernelContext_20)->nodes_0 = nodes_5;
    (&kernelContext_20)->accel_0 = accel_5;
    (&kernelContext_20)->scratch_0 = scratch_5;
    (&kernelContext_20)->refs_0 = refs_5;
    (&kernelContext_20)->bonds_0 = bonds_5;
    (&kernelContext_20)->stiffness_0 = stiffness_5;
    (&kernelContext_20)->x_0 = x_5;
    (&kernelContext_20)->report_0 = report_5;
    threadgroup array<float2, int(32)> g_blast_part_5;
    (&kernelContext_20)->g_blast_part_0 = &g_blast_part_5;
    threadgroup array<float, int(1024)> g_wide_max_5;
    (&kernelContext_20)->g_wide_max_0 = &g_wide_max_5;
    threadgroup float g_wide_max_out_5;
    (&kernelContext_20)->g_wide_max_out_0 = &g_wide_max_out_5;
    uint tid_5 = thread_2.x;
    Block_0 _S94 = bond_block_0(group_3, tid_5, &kernelContext_20);
    if((_S94.begin_0) >= (_S94.end_0))
    {
        return;
    }
    uint p_1 = _S94.begin_0 + tid_5;
    float f_0;
    if(p_1 < (_S94.end_0))
    {
        uint j_5 = (&kernelContext_20)->lists_0[p_1];
        if(((&kernelContext_20)->params_0->warm_0) == 0U)
        {
            uint _S95 = j_5 * 2U;
            packed_float4 _S96 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_20)->x_0+_S95) = _S96;
            *((&kernelContext_20)->x_0+(_S95 + 1U)) = _S96;
        }
        uint _S97 = j_5 * 2U;
        f_0 = norm2_0((float4(*((&kernelContext_20)->x_0+_S97)) ).xyz, (float4(*((&kernelContext_20)->x_0+(_S97 + 1U))) ).xyz);
    }
    else
    {
        f_0 = 0.0f;
    }
    write_partial_0(tid_5, _S94.row_4, group_3.x, 2U, float4(f_0, 0.0f, 0.0f, 0.0f), &kernelContext_20);
    return;
}

uint group_uniform_0(uint value_1, uint thread_3)
{
    return value_1;
}

float4 row_sum_0(uint tid_6, uint row_7, uint count_0, uint k_9, KernelContext_0 thread* kernelContext_21)
{
    float4 _S98 = float4(0.0f) ;
    uint b_5 = tid_6;
    float4 s_8 = _S98;
    for(;;)
    {
        if(b_5 < count_0)
        {
        }
        else
        {
            break;
        }
        uint _S99 = pt_0(row_7, b_5, k_9, kernelContext_21);
        float4 s_9 = s_8 + float4(*(kernelContext_21->wide_0+_S99)) ;
        b_5 = b_5 + 1024U;
        s_8 = s_9;
    }
    float2 _S100 = blast_group_sum_0(tid_6, s_8.xy, kernelContext_21);
    float2 _S101 = blast_group_sum_0(tid_6, s_8.zw, kernelContext_21);
    return float4(_S100, _S101);
}

uint blast_frame_0(uint node_count_1, uint k_10)
{
    return node_count_1 + 10U * k_10;
}

uint frame_word_0(uint k_11, uint w_1, KernelContext_0 thread* kernelContext_22)
{
    return (*(kernelContext_22->islands_0+(blast_frame_0(kernelContext_22->params_0->node_count_0, k_11) + w_1 / 4U)))[w_1 % 4U];
}

uint frame_word_1(uint k_12, uint w_2, KernelContext_0 thread* kernelContext_23)
{
    return (*(kernelContext_23->islands_0+(blast_frame_0(kernelContext_23->params_0->node_count_0, k_12) + w_2 / 4U)))[w_2 % 4U];
}

uint3 carry_of_0(uint k_13, KernelContext_0 thread* kernelContext_24)
{
    uint _S102 = frame_word_1(k_13, 36U, kernelContext_24);
    uint _S103 = frame_word_1(k_13, 37U, kernelContext_24);
    uint _S104 = frame_word_1(k_13, 38U, kernelContext_24);
    return uint3(_S102, _S103, _S104);
}

bool carry_kept_0(uint3 c_2, uint count_1, KernelContext_0 thread* kernelContext_25)
{
    float gamma_0 = (as_type<float>((c_2.z)));
    bool _S105;
    if((kernelContext_25->params_0->carry_0) != 0U)
    {
        _S105 = (c_2.x) == count_1;
    }
    else
    {
        _S105 = false;
    }
    if(_S105)
    {
        _S105 = (c_2.y) < (6U * count_1);
    }
    else
    {
        _S105 = false;
    }
    if(_S105)
    {
        _S105 = gamma_0 > 0.0f;
    }
    else
    {
        _S105 = false;
    }
    if(_S105)
    {
        _S105 = isfinite(gamma_0);
    }
    else
    {
        _S105 = false;
    }
    return _S105;
}

void set_carry_0(uint k_14, uint count_2, uint iterations_0, float gamma_1, KernelContext_0 thread* kernelContext_26)
{
    uint at_0 = blast_frame_0(kernelContext_26->params_0->node_count_0, k_14) + 9U;
    thread uint4 v_8 = uint4(*(kernelContext_26->islands_0+at_0)) ;
    v_8.x = count_2;
    v_8.y = iterations_0;
    v_8.z = (as_type<uint>((gamma_1)));
    *(kernelContext_26->islands_0+at_0) = packed_uint4(v_8) ;
    return;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

[[kernel]] void w_setup(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], atomic<uint> device* meta_6 [[buffer(12)]], uint device* topo_args_6 [[buffer(13)]], packed_uint4 device* islands_6 [[buffer(1)]], BlastParams_0 constant* params_6 [[buffer(0)]], packed_float4 device* wide_6 [[buffer(11)]], uint device* lists_6 [[buffer(9)]], BlastNode_natural_0 device* nodes_6 [[buffer(2)]], packed_float4 device* accel_6 [[buffer(5)]], packed_float4 device* scratch_6 [[buffer(7)]], uint device* refs_6 [[buffer(3)]], BlastBond_natural_0 device* bonds_6 [[buffer(4)]], packed_float4 device* stiffness_6 [[buffer(10)]], packed_float4 device* x_6 [[buffer(6)]], packed_uint4 device* report_6 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_27;
    (&kernelContext_27)->meta_0 = meta_6;
    (&kernelContext_27)->topo_args_0 = topo_args_6;
    (&kernelContext_27)->islands_0 = islands_6;
    (&kernelContext_27)->params_0 = params_6;
    (&kernelContext_27)->wide_0 = wide_6;
    (&kernelContext_27)->lists_0 = lists_6;
    (&kernelContext_27)->nodes_0 = nodes_6;
    (&kernelContext_27)->accel_0 = accel_6;
    (&kernelContext_27)->scratch_0 = scratch_6;
    (&kernelContext_27)->refs_0 = refs_6;
    (&kernelContext_27)->bonds_0 = bonds_6;
    (&kernelContext_27)->stiffness_0 = stiffness_6;
    (&kernelContext_27)->x_0 = x_6;
    (&kernelContext_27)->report_0 = report_6;
    threadgroup array<float2, int(32)> g_blast_part_6;
    (&kernelContext_27)->g_blast_part_0 = &g_blast_part_6;
    threadgroup array<float, int(1024)> g_wide_max_6;
    (&kernelContext_27)->g_wide_max_0 = &g_wide_max_6;
    threadgroup float g_wide_max_out_6;
    (&kernelContext_27)->g_wide_max_out_0 = &g_wide_max_out_6;
    uint tid_7 = thread_4.x;
    uint row_8 = group_4.x;
    uint _S106 = row_island_0(row_8, &kernelContext_27);
    uint island_1 = group_uniform_0(_S106, tid_7);
    uint4 r_3 = group_uniform4_0(uint4(*((&kernelContext_27)->islands_0+island_1)) , tid_7);
    uint _S107 = r_3.y - r_3.x;
    float4 _S108 = row_sum_0(tid_7, row_8, blocks_0(_S107), 0U, &kernelContext_27);
    float4 _S109 = row_sum_0(tid_7, row_8, blocks_0(r_3.w - r_3.z), 2U, &kernelContext_27);
    if(tid_7 != 0U)
    {
        return;
    }
    uint4 _S110 = uint4(*((&kernelContext_27)->report_0+island_1)) ;
    bool _S111;
    if(((&kernelContext_27)->params_0->warm_0) != 0U)
    {
        _S111 = ((_S110.y) & 4U) != 0U;
    }
    else
    {
        _S111 = false;
    }
    if(_S111)
    {
        _S111 = (_S108.z) == 0.0f;
    }
    else
    {
        _S111 = false;
    }
    uint f_1;
    if(_S111)
    {
        f_1 = 128U;
    }
    else
    {
        f_1 = 1U;
    }
    float _S112 = _S108.x;
    float delta2_0 = _S112 * (&kernelContext_27)->params_0->tolerance_0 * (&kernelContext_27)->params_0->tolerance_0;
    if(f_1 == 1U)
    {
        if((_S108.y) > 0.0f)
        {
            _S111 = true;
        }
        else
        {
            uint _S113 = frame_word_0(island_1, 2U, &kernelContext_27);
            _S111 = _S113 != 0U;
        }
        if(_S111)
        {
            f_1 = 4U;
        }
        else
        {
            if(!(delta2_0 > 0.0f))
            {
                f_1 = 18U;
            }
        }
    }
    uint _S114 = frame_word_0(island_1, 0U, &kernelContext_27);
    if(_S114 == 0U)
    {
        f_1 = f_1 | 8U;
    }
    uint3 _S115 = carry_of_0(island_1, &kernelContext_27);
    bool _S116 = carry_kept_0(_S115, _S107, &kernelContext_27);
    if(_S116)
    {
        _S111 = (_S108.w) <= ((&kernelContext_27)->params_0->tolerance_0 * (&kernelContext_27)->params_0->tolerance_0 * _S112);
    }
    else
    {
        _S111 = false;
    }
    float previous_0;
    if(_S111)
    {
        uint f_2 = f_1 | 256U;
        previous_0 = (as_type<float>((_S115.z)));
        f_1 = f_2;
    }
    else
    {
        previous_0 = 0.0f;
    }
    if(((&kernelContext_27)->params_0->carry_0) != 0U)
    {
        set_carry_0(island_1, 0U, 0U, 0.0f, &kernelContext_27);
    }
    float _S117 = float_bits_0(2139095040U);
    *((&kernelContext_27)->wide_0+st_0(row_8, 0U)) = packed_float4(float4(delta2_0, 0.0f, previous_0, _S117)) ;
    *((&kernelContext_27)->wide_0+st_0(row_8, 1U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, _S117)) ;
    float _S118 = (as_type<float>((0U)));
    *((&kernelContext_27)->wide_0+st_0(row_8, 2U)) = packed_float4(float4(sqrt(_S109.x), 0.0f, _S118, (as_type<float>((f_1))))) ;
    packed_float4 device* _S119 = (&kernelContext_27)->wide_0+st_0(row_8, 3U);
    if((f_1 & 256U) != 0U)
    {
        f_1 = _S115.y;
    }
    else
    {
        f_1 = 0U;
    }
    *_S119 = packed_float4(float4((as_type<float>((f_1))), 0.0f, 0.0f, _S118)) ;
    return;
}

uint bs_0(uint j_6, KernelContext_0 thread* kernelContext_28)
{
    return kernelContext_28->params_0->node_count_0 * kernelContext_28->params_0->node_vectors_0 + 2U * j_6;
}

float3 rot_a_0(uint j_7, float3 v_9, KernelContext_0 thread* kernelContext_29)
{
    float3 _S120;
    if((kernelContext_29->params_0->rotation_0) != 0U)
    {
        float3 _S121 = stiff_0(j_7, 0U, v_9, kernelContext_29);
        _S120 = _S121;
    }
    else
    {
        _S120 = v_9;
    }
    return _S120;
}

float3 rot_l_0(uint j_8, float3 v_10, KernelContext_0 thread* kernelContext_30)
{
    float3 _S122;
    if((kernelContext_30->params_0->rotation_0) == 2U)
    {
        float3 _S123 = stiff_0(j_8, 2U, v_10, kernelContext_30);
        _S122 = _S123;
    }
    else
    {
        _S122 = v_10;
    }
    return _S122;
}

void bond_term_0(const BlastBond_natural_0 thread* b_6, uint j_9, uint v_11, float3 thread* ta_0, float3 thread* tl_0, KernelContext_0 thread* kernelContext_31)
{
    uint _S124 = b_6->n0_0;
    BlastNode_natural_0 m0_0 = kernelContext_31->nodes_0[b_6->n0_0];
    uint _S125 = b_6->n1_0;
    BlastNode_natural_0 m1_0 = kernelContext_31->nodes_0[b_6->n1_0];
    float3 _S126 = va_0(b_6->n0_0, v_11, kernelContext_31);
    float3 a0_1 = _S126 * float3(m0_0.w_ang_0) ;
    float3 _S127 = vl_0(_S124, v_11, kernelContext_31);
    float3 l0_0 = _S127 * float3(m0_0.w_lin_0) ;
    float3 _S128 = va_0(_S125, v_11, kernelContext_31);
    float3 a1_1 = _S128 * float3(m1_0.w_ang_0) ;
    float3 _S129 = vl_0(_S125, v_11, kernelContext_31);
    float3 l1_0 = _S129 * float3(m1_0.w_lin_0) ;
    if((kernelContext_31->params_0->rotation_0) == 0U)
    {
        float3 _S130 = float3(b_6->s_0) ;
        *ta_0 = (a0_1 - a1_1) * _S130;
        *tl_0 = (l0_0 - l1_0 + (cross((float4(b_6->o0_0) ).xyz, a0_1) - cross((float4(b_6->o1_0) ).xyz, a1_1))) * _S130;
        return;
    }
    float3 _S131 = rot_a_0(j_9, a0_1 - a1_1, kernelContext_31);
    float3 _S132 = float3(b_6->s_0) ;
    *ta_0 = _S131 * _S132;
    float3 _S133 = rot_l_0(j_9, l0_0 - l1_0 + (cross((float4(b_6->o0_0) ).xyz, a0_1) - cross((float4(b_6->o1_0) ).xyz, a1_1)), kernelContext_31);
    *tl_0 = _S133 * _S132;
    return;
}

void bond_term_1(const BlastBond_natural_0 thread* b_7, uint j_10, uint v_12, float3 thread* ta_1, float3 thread* tl_1, KernelContext_0 thread* kernelContext_32)
{
    uint _S134 = b_7->n0_0;
    BlastNode_natural_0 m0_1 = kernelContext_32->nodes_0[b_7->n0_0];
    uint _S135 = b_7->n1_0;
    BlastNode_natural_0 m1_1 = kernelContext_32->nodes_0[b_7->n1_0];
    float3 _S136 = va_0(b_7->n0_0, v_12, kernelContext_32);
    float3 a0_2 = _S136 * float3(m0_1.w_ang_0) ;
    float3 _S137 = vl_0(_S134, v_12, kernelContext_32);
    float3 l0_1 = _S137 * float3(m0_1.w_lin_0) ;
    float3 _S138 = va_0(_S135, v_12, kernelContext_32);
    float3 a1_2 = _S138 * float3(m1_1.w_ang_0) ;
    float3 _S139 = vl_0(_S135, v_12, kernelContext_32);
    float3 l1_1 = _S139 * float3(m1_1.w_lin_0) ;
    if((kernelContext_32->params_0->rotation_0) == 0U)
    {
        float3 _S140 = float3(b_7->s_0) ;
        *ta_1 = (a0_2 - a1_2) * _S140;
        *tl_1 = (l0_1 - l1_1 + (cross((float4(b_7->o0_0) ).xyz, a0_2) - cross((float4(b_7->o1_0) ).xyz, a1_2))) * _S140;
        return;
    }
    float3 _S141 = rot_a_0(j_10, a0_2 - a1_2, kernelContext_32);
    float3 _S142 = float3(b_7->s_0) ;
    *ta_1 = _S141 * _S142;
    float3 _S143 = rot_l_0(j_10, l0_1 - l1_1 + (cross((float4(b_7->o0_0) ).xyz, a0_2) - cross((float4(b_7->o1_0) ).xyz, a1_2)), kernelContext_32);
    *tl_1 = _S143 * _S142;
    return;
}

void gather_0(uint node_2, bool from_t_0, bool with_u_0, float3 thread* ang_0, float3 thread* lin_0, KernelContext_0 thread* kernelContext_33)
{
    BlastNode_natural_0 n_8 = kernelContext_33->nodes_0[node_2];
    float3 _S144 = float3(0.0f) ;
    uint r_4 = n_8.ref_begin_0;
    float3 acc_a_0 = _S144;
    float3 acc_l_0 = _S144;
    for(;;)
    {
        if(r_4 < (n_8.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_2 = kernelContext_33->refs_0[r_4];
        uint j_11 = ref_2 & 2147483647U;
        thread BlastBond_natural_0 _S145 = *(kernelContext_33->bonds_0+j_11);
        if(!(((&_S145)->health_0) > 0.0f))
        {
            r_4 = r_4 + 1U;
            continue;
        }
        float3 ya_0;
        float3 yl_0;
        if(from_t_0)
        {
            uint _S146 = bs_0(j_11, kernelContext_33);
            float3 _S147 = (float4(*(kernelContext_33->scratch_0+(_S146 + 1U))) ).xyz;
            ya_0 = (float4(*(kernelContext_33->scratch_0+_S146)) ).xyz;
            yl_0 = _S147;
        }
        else
        {
            uint _S148 = j_11 * 2U;
            float3 ya_1 = (float4(*(kernelContext_33->x_0+_S148)) ).xyz;
            float3 yl_1 = (float4(*(kernelContext_33->x_0+(_S148 + 1U))) ).xyz;
            if(with_u_0)
            {
                thread float3 ta_2;
                thread float3 tl_2;
                bond_term_0(&_S145, j_11, 6U, &ta_2, &tl_2, kernelContext_33);
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
        float3 yl_3;
        float3 ya_2;
        if((kernelContext_33->params_0->rotation_0) == 0U)
        {
            float3 _S149 = float3((&_S145)->s_0) ;
            float3 ya_3 = ya_0 * _S149;
            yl_3 = yl_0 * _S149;
            ya_2 = ya_3;
        }
        else
        {
            float3 _S150 = rot_a_0(j_11, ya_0, kernelContext_33);
            float3 _S151 = float3((&_S145)->s_0) ;
            float3 _S152 = _S150 * _S151;
            float3 _S153 = rot_l_0(j_11, yl_0, kernelContext_33);
            yl_3 = _S153 * _S151;
            ya_2 = _S152;
        }
        float3 acc_a_1;
        float3 acc_l_1;
        if((ref_2 & 2147483648U) != 0U)
        {
            float3 acc_l_2 = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross((float4((&_S145)->o1_0) ).xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            float3 acc_l_3 = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross((float4((&_S145)->o0_0) ).xyz, yl_3));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_4 = r_4 + 1U;
    }
    *ang_0 = acc_a_0 * float3(n_8.w_ang_0) ;
    *lin_0 = acc_l_0 * float3(n_8.w_lin_0) ;
    return;
}

[[kernel]] void w_warm(uint3 group_5 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], atomic<uint> device* meta_7 [[buffer(12)]], uint device* topo_args_7 [[buffer(13)]], packed_uint4 device* islands_7 [[buffer(1)]], BlastParams_0 constant* params_7 [[buffer(0)]], packed_float4 device* wide_7 [[buffer(11)]], uint device* lists_7 [[buffer(9)]], BlastNode_natural_0 device* nodes_7 [[buffer(2)]], packed_float4 device* accel_7 [[buffer(5)]], packed_float4 device* scratch_7 [[buffer(7)]], uint device* refs_7 [[buffer(3)]], BlastBond_natural_0 device* bonds_7 [[buffer(4)]], packed_float4 device* stiffness_7 [[buffer(10)]], packed_float4 device* x_7 [[buffer(6)]], packed_uint4 device* report_7 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_34;
    (&kernelContext_34)->meta_0 = meta_7;
    (&kernelContext_34)->topo_args_0 = topo_args_7;
    (&kernelContext_34)->islands_0 = islands_7;
    (&kernelContext_34)->params_0 = params_7;
    (&kernelContext_34)->wide_0 = wide_7;
    (&kernelContext_34)->lists_0 = lists_7;
    (&kernelContext_34)->nodes_0 = nodes_7;
    (&kernelContext_34)->accel_0 = accel_7;
    (&kernelContext_34)->scratch_0 = scratch_7;
    (&kernelContext_34)->refs_0 = refs_7;
    (&kernelContext_34)->bonds_0 = bonds_7;
    (&kernelContext_34)->stiffness_0 = stiffness_7;
    (&kernelContext_34)->x_0 = x_7;
    (&kernelContext_34)->report_0 = report_7;
    threadgroup array<float2, int(32)> g_blast_part_7;
    (&kernelContext_34)->g_blast_part_0 = &g_blast_part_7;
    threadgroup array<float, int(1024)> g_wide_max_7;
    (&kernelContext_34)->g_wide_max_0 = &g_wide_max_7;
    threadgroup float g_wide_max_out_7;
    (&kernelContext_34)->g_wide_max_out_0 = &g_wide_max_out_7;
    uint tid_8 = thread_5.x;
    Block_0 _S154 = node_block_0(group_5, tid_8, &kernelContext_34);
    bool _S155;
    if((_S154.begin_0) >= (_S154.end_0))
    {
        _S155 = true;
    }
    else
    {
        _S155 = ((&kernelContext_34)->params_0->warm_0) == 0U;
    }
    if(_S155)
    {
        _S155 = true;
    }
    else
    {
        _S155 = ((_S154.flags_0) & 1U) == 0U;
    }
    if(_S155)
    {
        return;
    }
    uint p_2 = _S154.begin_0 + tid_8;
    if(p_2 < (_S154.end_0))
    {
        uint i_1 = (&kernelContext_34)->lists_0[p_2];
        thread float3 a_2;
        thread float3 l_2;
        gather_0(i_1, false, false, &a_2, &l_2, &kernelContext_34);
        float3 _S156 = va_0(i_1, 14U, &kernelContext_34);
        float3 _S157 = _S156 - a_2;
        float3 _S158 = vl_0(i_1, 14U, &kernelContext_34);
        set_v_0(i_1, 0U, _S157, _S158 - l_2, &kernelContext_34);
    }
    return;
}

bool runs_0(uint flags_1, bool only_rebuild_0)
{
    bool _S159;
    if((flags_1 & 1U) != 0U)
    {
        if(!only_rebuild_0)
        {
            _S159 = true;
        }
        else
        {
            _S159 = (flags_1 & 64U) != 0U;
        }
    }
    else
    {
        _S159 = false;
    }
    return _S159;
}

float frame_float_0(uint k_15, uint w_3, KernelContext_0 thread* kernelContext_35)
{
    uint _S160 = frame_word_1(k_15, w_3, kernelContext_35);
    return (as_type<float>((_S160)));
}

float frame_float_1(uint k_16, uint w_4, KernelContext_0 thread* kernelContext_36)
{
    uint _S161 = frame_word_1(k_16, w_4, kernelContext_36);
    return (as_type<float>((_S161)));
}

float3 frame_axis_0(uint k_17, uint rotations_0, uint kk_1, KernelContext_0 thread* kernelContext_37)
{
    if(rotations_0 == 1U)
    {
        float _S162 = frame_float_1(k_17, 4U, kernelContext_37);
        float _S163 = frame_float_1(k_17, 5U, kernelContext_37);
        float _S164 = frame_float_1(k_17, 6U, kernelContext_37);
        return float3(_S162, _S163, _S164);
    }
    float _S165;
    if(kk_1 == 0U)
    {
        _S165 = 1.0f;
    }
    else
    {
        _S165 = 0.0f;
    }
    float _S166;
    if(kk_1 == 1U)
    {
        _S166 = 1.0f;
    }
    else
    {
        _S166 = 0.0f;
    }
    float _S167;
    if(kk_1 == 2U)
    {
        _S167 = 1.0f;
    }
    else
    {
        _S167 = 0.0f;
    }
    return float3(_S165, _S166, _S167);
}

void frame_terms_0(uint k_18, uint rotations_1, uint i_2, uint v_13, array<float, int(6)> thread* s_10, KernelContext_0 thread* kernelContext_38)
{
    BlastNode_natural_0 n_9 = kernelContext_38->nodes_0[i_2];
    float3 _S168 = vl_0(i_2, v_13, kernelContext_38);
    float3 lin_1 = _S168 / float3(n_9.w_lin_0) ;
    float3 _S169 = va_0(i_2, v_13, kernelContext_38);
    float3 _S170 = _S169 / float3(n_9.w_ang_0)  - cross((float4(n_9.position_0) ).xyz, lin_1);
    uint kk_2 = 0U;
    for(;;)
    {
        if(kk_2 < rotations_1)
        {
        }
        else
        {
            break;
        }
        float3 _S171 = frame_axis_0(k_18, rotations_1, kk_2, kernelContext_38);
        (*s_10)[kk_2] = (*s_10)[kk_2] + dot(_S171, _S170);
        kk_2 = kk_2 + 1U;
    }
    uint j_12 = 0U;
    for(;;)
    {
        if(j_12 < 3U)
        {
        }
        else
        {
            break;
        }
        (*s_10)[rotations_1 + j_12] = (*s_10)[rotations_1 + j_12] + lin_1[j_12];
        j_12 = j_12 + 1U;
    }
    return;
}

void proj_part_0(uint3 group_6, uint tid_9, uint v_14, bool only_rebuild_1, KernelContext_0 thread* kernelContext_39)
{
    Block_0 _S172 = node_block_0(group_6, tid_9, kernelContext_39);
    bool _S173;
    if((_S172.begin_0) >= (_S172.end_0))
    {
        _S173 = true;
    }
    else
    {
        _S173 = !runs_0(_S172.flags_0, only_rebuild_1);
    }
    if(_S173)
    {
        _S173 = true;
    }
    else
    {
        _S173 = ((_S172.flags_0) & 8U) == 0U;
    }
    if(_S173)
    {
        return;
    }
    thread array<float, int(6)> s_11;
    uint z_1 = 0U;
    for(;;)
    {
        if(z_1 < 6U)
        {
        }
        else
        {
            break;
        }
        s_11[z_1] = 0.0f;
        z_1 = z_1 + 1U;
    }
    uint p_3 = _S172.begin_0 + tid_9;
    if(p_3 < (_S172.end_0))
    {
        uint _S174 = row_island_0(_S172.row_4, kernelContext_39);
        uint _S175 = frame_word_0(_S174, 1U, kernelContext_39);
        frame_terms_0(_S174, _S175, kernelContext_39->lists_0[p_3], v_14, &s_11, kernelContext_39);
    }
    uint _S176 = group_6.x;
    write_partial_0(tid_9, _S172.row_4, _S176, 0U, float4(s_11[int(0)], s_11[int(1)], s_11[int(2)], s_11[int(3)]), kernelContext_39);
    write_partial_0(tid_9, _S172.row_4, _S176, 1U, float4(s_11[int(4)], s_11[int(5)], 0.0f, 0.0f), kernelContext_39);
    return;
}

[[kernel]] void w_proj_part_r(uint3 g_2 [[threadgroup_position_in_grid]], uint3 t_0 [[thread_position_in_threadgroup]], atomic<uint> device* meta_8 [[buffer(12)]], uint device* topo_args_8 [[buffer(13)]], packed_uint4 device* islands_8 [[buffer(1)]], BlastParams_0 constant* params_8 [[buffer(0)]], packed_float4 device* wide_8 [[buffer(11)]], uint device* lists_8 [[buffer(9)]], BlastNode_natural_0 device* nodes_8 [[buffer(2)]], packed_float4 device* accel_8 [[buffer(5)]], packed_float4 device* scratch_8 [[buffer(7)]], uint device* refs_8 [[buffer(3)]], BlastBond_natural_0 device* bonds_8 [[buffer(4)]], packed_float4 device* stiffness_8 [[buffer(10)]], packed_float4 device* x_8 [[buffer(6)]], packed_uint4 device* report_8 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_40;
    (&kernelContext_40)->meta_0 = meta_8;
    (&kernelContext_40)->topo_args_0 = topo_args_8;
    (&kernelContext_40)->islands_0 = islands_8;
    (&kernelContext_40)->params_0 = params_8;
    (&kernelContext_40)->wide_0 = wide_8;
    (&kernelContext_40)->lists_0 = lists_8;
    (&kernelContext_40)->nodes_0 = nodes_8;
    (&kernelContext_40)->accel_0 = accel_8;
    (&kernelContext_40)->scratch_0 = scratch_8;
    (&kernelContext_40)->refs_0 = refs_8;
    (&kernelContext_40)->bonds_0 = bonds_8;
    (&kernelContext_40)->stiffness_0 = stiffness_8;
    (&kernelContext_40)->x_0 = x_8;
    (&kernelContext_40)->report_0 = report_8;
    threadgroup array<float2, int(32)> g_blast_part_8;
    (&kernelContext_40)->g_blast_part_0 = &g_blast_part_8;
    threadgroup array<float, int(1024)> g_wide_max_8;
    (&kernelContext_40)->g_wide_max_0 = &g_wide_max_8;
    threadgroup float g_wide_max_out_8;
    (&kernelContext_40)->g_wide_max_out_0 = &g_wide_max_out_8;
    proj_part_0(g_2, t_0.x, 0U, false, &kernelContext_40);
    return;
}

uint4 row_range_0(uint tid_10, uint row_9, KernelContext_0 thread* kernelContext_41)
{
    uint _S177 = row_island_0(row_9, kernelContext_41);
    return group_uniform4_0(uint4(*(kernelContext_41->islands_0+group_uniform_0(_S177, tid_10))) , tid_10);
}

void frame_solve_0(uint k_19, uint rotations_2, const array<float, int(6)> thread* sum_0, float3 thread* omega_0, float3 thread* t_1, KernelContext_0 thread* kernelContext_42)
{
    uint j_13;
    uint d_0 = rotations_2 + 3U;
    thread array<float, int(6)> c_3;
    uint z_2 = 0U;
    for(;;)
    {
        if(z_2 < 6U)
        {
        }
        else
        {
            break;
        }
        c_3[z_2] = 0.0f;
        z_2 = z_2 + 1U;
    }
    uint row_10 = 0U;
    for(;;)
    {
        if(row_10 < d_0)
        {
        }
        else
        {
            break;
        }
        float _S178 = (*sum_0)[row_10];
        float _S179 = frame_float_0(k_19, 8U + row_10, kernelContext_42);
        float _S180 = _S178 * _S179;
        j_13 = 0U;
        float value_2 = _S180;
        for(;;)
        {
            if(j_13 < row_10)
            {
            }
            else
            {
                break;
            }
            float _S181 = frame_float_0(k_19, 14U + row_10 * (row_10 + 1U) / 2U + j_13, kernelContext_42);
            float value_3 = value_2 - _S181 * c_3[j_13];
            j_13 = j_13 + 1U;
            value_2 = value_3;
        }
        uint _S182 = row_10 + 1U;
        float _S183 = frame_float_0(k_19, 14U + row_10 * _S182 / 2U + row_10, kernelContext_42);
        c_3[row_10] = value_2 / _S183;
        row_10 = _S182;
    }
    int row_11 = int(d_0) - int(1);
    for(;;)
    {
        if(row_11 >= int(0))
        {
        }
        else
        {
            break;
        }
        uint r_5 = uint(row_11);
        uint _S184 = r_5 + 1U;
        j_13 = _S184;
        for(;;)
        {
            if(j_13 < d_0)
            {
            }
            else
            {
                break;
            }
            uint _S185 = j_13 + 1U;
            float _S186 = frame_float_0(k_19, 14U + j_13 * _S185 / 2U + r_5, kernelContext_42);
            c_3[r_5] = c_3[r_5] - _S186 * c_3[j_13];
            j_13 = _S185;
        }
        float _S187 = frame_float_0(k_19, 14U + r_5 * _S184 / 2U + r_5, kernelContext_42);
        c_3[r_5] = c_3[r_5] / _S187;
        row_11 = row_11 - int(1);
    }
    z_2 = 0U;
    for(;;)
    {
        if(z_2 < d_0)
        {
        }
        else
        {
            break;
        }
        float _S188 = frame_float_0(k_19, 8U + z_2, kernelContext_42);
        c_3[z_2] = c_3[z_2] * _S188;
        z_2 = z_2 + 1U;
    }
    *omega_0 = float3(0.0f) ;
    j_13 = 0U;
    for(;;)
    {
        if(j_13 < rotations_2)
        {
        }
        else
        {
            break;
        }
        float3 _S189 = frame_axis_0(k_19, rotations_2, j_13, kernelContext_42);
        *omega_0 = *omega_0 + _S189 * float3(c_3[j_13]) ;
        j_13 = j_13 + 1U;
    }
    *t_1 = float3(c_3[rotations_2], c_3[rotations_2 + 1U], c_3[rotations_2 + 2U]);
    return;
}

void proj_solve_0(uint3 group_7, uint tid_11, bool only_rebuild_2, KernelContext_0 thread* kernelContext_43)
{
    uint row_12 = group_7.x;
    uint _S190 = flags_of_0(row_12, kernelContext_43);
    uint f_3 = group_uniform_0(_S190, tid_11);
    bool _S191;
    if(!runs_0(f_3, only_rebuild_2))
    {
        _S191 = true;
    }
    else
    {
        _S191 = (f_3 & 8U) == 0U;
    }
    if(_S191)
    {
        return;
    }
    uint4 _S192 = row_range_0(tid_11, row_12, kernelContext_43);
    uint nb_1 = blocks_0(_S192.y - _S192.x);
    float4 _S193 = row_sum_0(tid_11, row_12, nb_1, 0U, kernelContext_43);
    float4 _S194 = row_sum_0(tid_11, row_12, nb_1, 1U, kernelContext_43);
    if(tid_11 != 0U)
    {
        return;
    }
    uint _S195 = row_island_0(row_12, kernelContext_43);
    float _S196 = _S193.x;
    float _S197 = _S193.y;
    float _S198 = _S193.z;
    float _S199 = _S193.w;
    float _S200 = _S194.x;
    float _S201 = _S194.y;
    uint _S202 = frame_word_0(_S195, 1U, kernelContext_43);
    thread array<float, int(6)> _S203;
    _S203[int(0)] = _S196;
    _S203[int(1)] = _S197;
    _S203[int(2)] = _S198;
    _S203[int(3)] = _S199;
    _S203[int(4)] = _S200;
    _S203[int(5)] = _S201;
    thread float3 omega_1;
    thread float3 t_2;
    frame_solve_0(_S195, _S202, &_S203, &omega_1, &t_2, kernelContext_43);
    *(kernelContext_43->wide_0+st_0(row_12, 4U)) = packed_float4(float4(omega_1, 0.0f)) ;
    *(kernelContext_43->wide_0+st_0(row_12, 5U)) = packed_float4(float4(t_2, 0.0f)) ;
    return;
}

[[kernel]] void w_proj_solve(uint3 g_3 [[threadgroup_position_in_grid]], uint3 t_3 [[thread_position_in_threadgroup]], atomic<uint> device* meta_9 [[buffer(12)]], uint device* topo_args_9 [[buffer(13)]], packed_uint4 device* islands_9 [[buffer(1)]], BlastParams_0 constant* params_9 [[buffer(0)]], packed_float4 device* wide_9 [[buffer(11)]], uint device* lists_9 [[buffer(9)]], BlastNode_natural_0 device* nodes_9 [[buffer(2)]], packed_float4 device* accel_9 [[buffer(5)]], packed_float4 device* scratch_9 [[buffer(7)]], uint device* refs_9 [[buffer(3)]], BlastBond_natural_0 device* bonds_9 [[buffer(4)]], packed_float4 device* stiffness_9 [[buffer(10)]], packed_float4 device* x_9 [[buffer(6)]], packed_uint4 device* report_9 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_44;
    (&kernelContext_44)->meta_0 = meta_9;
    (&kernelContext_44)->topo_args_0 = topo_args_9;
    (&kernelContext_44)->islands_0 = islands_9;
    (&kernelContext_44)->params_0 = params_9;
    (&kernelContext_44)->wide_0 = wide_9;
    (&kernelContext_44)->lists_0 = lists_9;
    (&kernelContext_44)->nodes_0 = nodes_9;
    (&kernelContext_44)->accel_0 = accel_9;
    (&kernelContext_44)->scratch_0 = scratch_9;
    (&kernelContext_44)->refs_0 = refs_9;
    (&kernelContext_44)->bonds_0 = bonds_9;
    (&kernelContext_44)->stiffness_0 = stiffness_9;
    (&kernelContext_44)->x_0 = x_9;
    (&kernelContext_44)->report_0 = report_9;
    threadgroup array<float2, int(32)> g_blast_part_9;
    (&kernelContext_44)->g_blast_part_0 = &g_blast_part_9;
    threadgroup array<float, int(1024)> g_wide_max_9;
    (&kernelContext_44)->g_wide_max_0 = &g_wide_max_9;
    threadgroup float g_wide_max_out_9;
    (&kernelContext_44)->g_wide_max_out_0 = &g_wide_max_out_9;
    proj_solve_0(g_3, t_3.x, false, &kernelContext_44);
    return;
}

void frame_apply_0(uint i_3, uint v_15, float3 omega_2, float3 t_4, KernelContext_0 thread* kernelContext_45)
{
    BlastNode_natural_0 n_10 = kernelContext_45->nodes_0[i_3];
    float3 _S204 = va_0(i_3, v_15, kernelContext_45);
    float3 _S205 = _S204 - omega_2 / float3(n_10.w_ang_0) ;
    float3 _S206 = vl_0(i_3, v_15, kernelContext_45);
    set_v_0(i_3, v_15, _S205, _S206 - (t_4 + cross((float4(n_10.position_0) ).xyz, omega_2)) / float3(n_10.w_lin_0) , kernelContext_45);
    return;
}

void proj_apply_0(uint3 group_8, uint tid_12, uint v_16, bool only_rebuild_3, KernelContext_0 thread* kernelContext_46)
{
    Block_0 _S207 = node_block_0(group_8, tid_12, kernelContext_46);
    bool _S208;
    if((_S207.begin_0) >= (_S207.end_0))
    {
        _S208 = true;
    }
    else
    {
        _S208 = !runs_0(_S207.flags_0, only_rebuild_3);
    }
    if(_S208)
    {
        _S208 = true;
    }
    else
    {
        _S208 = ((_S207.flags_0) & 8U) == 0U;
    }
    if(_S208)
    {
        return;
    }
    uint p_4 = _S207.begin_0 + tid_12;
    if(p_4 >= (_S207.end_0))
    {
        return;
    }
    frame_apply_0(kernelContext_46->lists_0[p_4], v_16, (float4(*(kernelContext_46->wide_0+st_0(_S207.row_4, 4U))) ).xyz, (float4(*(kernelContext_46->wide_0+st_0(_S207.row_4, 5U))) ).xyz, kernelContext_46);
    return;
}

[[kernel]] void w_proj_apply_r(uint3 g_4 [[threadgroup_position_in_grid]], uint3 t_5 [[thread_position_in_threadgroup]], atomic<uint> device* meta_10 [[buffer(12)]], uint device* topo_args_10 [[buffer(13)]], packed_uint4 device* islands_10 [[buffer(1)]], BlastParams_0 constant* params_10 [[buffer(0)]], packed_float4 device* wide_10 [[buffer(11)]], uint device* lists_10 [[buffer(9)]], BlastNode_natural_0 device* nodes_10 [[buffer(2)]], packed_float4 device* accel_10 [[buffer(5)]], packed_float4 device* scratch_10 [[buffer(7)]], uint device* refs_10 [[buffer(3)]], BlastBond_natural_0 device* bonds_10 [[buffer(4)]], packed_float4 device* stiffness_10 [[buffer(10)]], packed_float4 device* x_10 [[buffer(6)]], packed_uint4 device* report_10 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_47;
    (&kernelContext_47)->meta_0 = meta_10;
    (&kernelContext_47)->topo_args_0 = topo_args_10;
    (&kernelContext_47)->islands_0 = islands_10;
    (&kernelContext_47)->params_0 = params_10;
    (&kernelContext_47)->wide_0 = wide_10;
    (&kernelContext_47)->lists_0 = lists_10;
    (&kernelContext_47)->nodes_0 = nodes_10;
    (&kernelContext_47)->accel_0 = accel_10;
    (&kernelContext_47)->scratch_0 = scratch_10;
    (&kernelContext_47)->refs_0 = refs_10;
    (&kernelContext_47)->bonds_0 = bonds_10;
    (&kernelContext_47)->stiffness_0 = stiffness_10;
    (&kernelContext_47)->x_0 = x_10;
    (&kernelContext_47)->report_0 = report_10;
    threadgroup array<float2, int(32)> g_blast_part_10;
    (&kernelContext_47)->g_blast_part_0 = &g_blast_part_10;
    threadgroup array<float, int(1024)> g_wide_max_10;
    (&kernelContext_47)->g_wide_max_0 = &g_wide_max_10;
    threadgroup float g_wide_max_out_10;
    (&kernelContext_47)->g_wide_max_out_0 = &g_wide_max_out_10;
    proj_apply_0(g_4, t_5.x, 0U, false, &kernelContext_47);
    return;
}

[[kernel]] void w_proj_part_rb(uint3 g_5 [[threadgroup_position_in_grid]], uint3 t_6 [[thread_position_in_threadgroup]], atomic<uint> device* meta_11 [[buffer(12)]], uint device* topo_args_11 [[buffer(13)]], packed_uint4 device* islands_11 [[buffer(1)]], BlastParams_0 constant* params_11 [[buffer(0)]], packed_float4 device* wide_11 [[buffer(11)]], uint device* lists_11 [[buffer(9)]], BlastNode_natural_0 device* nodes_11 [[buffer(2)]], packed_float4 device* accel_11 [[buffer(5)]], packed_float4 device* scratch_11 [[buffer(7)]], uint device* refs_11 [[buffer(3)]], BlastBond_natural_0 device* bonds_11 [[buffer(4)]], packed_float4 device* stiffness_11 [[buffer(10)]], packed_float4 device* x_11 [[buffer(6)]], packed_uint4 device* report_11 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_48;
    (&kernelContext_48)->meta_0 = meta_11;
    (&kernelContext_48)->topo_args_0 = topo_args_11;
    (&kernelContext_48)->islands_0 = islands_11;
    (&kernelContext_48)->params_0 = params_11;
    (&kernelContext_48)->wide_0 = wide_11;
    (&kernelContext_48)->lists_0 = lists_11;
    (&kernelContext_48)->nodes_0 = nodes_11;
    (&kernelContext_48)->accel_0 = accel_11;
    (&kernelContext_48)->scratch_0 = scratch_11;
    (&kernelContext_48)->refs_0 = refs_11;
    (&kernelContext_48)->bonds_0 = bonds_11;
    (&kernelContext_48)->stiffness_0 = stiffness_11;
    (&kernelContext_48)->x_0 = x_11;
    (&kernelContext_48)->report_0 = report_11;
    threadgroup array<float2, int(32)> g_blast_part_11;
    (&kernelContext_48)->g_blast_part_0 = &g_blast_part_11;
    threadgroup array<float, int(1024)> g_wide_max_11;
    (&kernelContext_48)->g_wide_max_0 = &g_wide_max_11;
    threadgroup float g_wide_max_out_11;
    (&kernelContext_48)->g_wide_max_out_0 = &g_wide_max_out_11;
    proj_part_0(g_5, t_6.x, 0U, true, &kernelContext_48);
    return;
}

[[kernel]] void w_proj_solve_rb(uint3 g_6 [[threadgroup_position_in_grid]], uint3 t_7 [[thread_position_in_threadgroup]], atomic<uint> device* meta_12 [[buffer(12)]], uint device* topo_args_12 [[buffer(13)]], packed_uint4 device* islands_12 [[buffer(1)]], BlastParams_0 constant* params_12 [[buffer(0)]], packed_float4 device* wide_12 [[buffer(11)]], uint device* lists_12 [[buffer(9)]], BlastNode_natural_0 device* nodes_12 [[buffer(2)]], packed_float4 device* accel_12 [[buffer(5)]], packed_float4 device* scratch_12 [[buffer(7)]], uint device* refs_12 [[buffer(3)]], BlastBond_natural_0 device* bonds_12 [[buffer(4)]], packed_float4 device* stiffness_12 [[buffer(10)]], packed_float4 device* x_12 [[buffer(6)]], packed_uint4 device* report_12 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_49;
    (&kernelContext_49)->meta_0 = meta_12;
    (&kernelContext_49)->topo_args_0 = topo_args_12;
    (&kernelContext_49)->islands_0 = islands_12;
    (&kernelContext_49)->params_0 = params_12;
    (&kernelContext_49)->wide_0 = wide_12;
    (&kernelContext_49)->lists_0 = lists_12;
    (&kernelContext_49)->nodes_0 = nodes_12;
    (&kernelContext_49)->accel_0 = accel_12;
    (&kernelContext_49)->scratch_0 = scratch_12;
    (&kernelContext_49)->refs_0 = refs_12;
    (&kernelContext_49)->bonds_0 = bonds_12;
    (&kernelContext_49)->stiffness_0 = stiffness_12;
    (&kernelContext_49)->x_0 = x_12;
    (&kernelContext_49)->report_0 = report_12;
    threadgroup array<float2, int(32)> g_blast_part_12;
    (&kernelContext_49)->g_blast_part_0 = &g_blast_part_12;
    threadgroup array<float, int(1024)> g_wide_max_12;
    (&kernelContext_49)->g_wide_max_0 = &g_wide_max_12;
    threadgroup float g_wide_max_out_12;
    (&kernelContext_49)->g_wide_max_out_0 = &g_wide_max_out_12;
    proj_solve_0(g_6, t_7.x, true, &kernelContext_49);
    return;
}

[[kernel]] void w_proj_apply_rb(uint3 g_7 [[threadgroup_position_in_grid]], uint3 t_8 [[thread_position_in_threadgroup]], atomic<uint> device* meta_13 [[buffer(12)]], uint device* topo_args_13 [[buffer(13)]], packed_uint4 device* islands_13 [[buffer(1)]], BlastParams_0 constant* params_13 [[buffer(0)]], packed_float4 device* wide_13 [[buffer(11)]], uint device* lists_13 [[buffer(9)]], BlastNode_natural_0 device* nodes_13 [[buffer(2)]], packed_float4 device* accel_13 [[buffer(5)]], packed_float4 device* scratch_13 [[buffer(7)]], uint device* refs_13 [[buffer(3)]], BlastBond_natural_0 device* bonds_13 [[buffer(4)]], packed_float4 device* stiffness_13 [[buffer(10)]], packed_float4 device* x_13 [[buffer(6)]], packed_uint4 device* report_13 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_50;
    (&kernelContext_50)->meta_0 = meta_13;
    (&kernelContext_50)->topo_args_0 = topo_args_13;
    (&kernelContext_50)->islands_0 = islands_13;
    (&kernelContext_50)->params_0 = params_13;
    (&kernelContext_50)->wide_0 = wide_13;
    (&kernelContext_50)->lists_0 = lists_13;
    (&kernelContext_50)->nodes_0 = nodes_13;
    (&kernelContext_50)->accel_0 = accel_13;
    (&kernelContext_50)->scratch_0 = scratch_13;
    (&kernelContext_50)->refs_0 = refs_13;
    (&kernelContext_50)->bonds_0 = bonds_13;
    (&kernelContext_50)->stiffness_0 = stiffness_13;
    (&kernelContext_50)->x_0 = x_13;
    (&kernelContext_50)->report_0 = report_13;
    threadgroup array<float2, int(32)> g_blast_part_13;
    (&kernelContext_50)->g_blast_part_0 = &g_blast_part_13;
    threadgroup array<float, int(1024)> g_wide_max_13;
    (&kernelContext_50)->g_wide_max_0 = &g_wide_max_13;
    threadgroup float g_wide_max_out_13;
    (&kernelContext_50)->g_wide_max_out_0 = &g_wide_max_out_13;
    proj_apply_0(g_7, t_8.x, 0U, true, &kernelContext_50);
    return;
}

[[kernel]] void w_proj_part_g(uint3 g_8 [[threadgroup_position_in_grid]], uint3 t_9 [[thread_position_in_threadgroup]], atomic<uint> device* meta_14 [[buffer(12)]], uint device* topo_args_14 [[buffer(13)]], packed_uint4 device* islands_14 [[buffer(1)]], BlastParams_0 constant* params_14 [[buffer(0)]], packed_float4 device* wide_14 [[buffer(11)]], uint device* lists_14 [[buffer(9)]], BlastNode_natural_0 device* nodes_14 [[buffer(2)]], packed_float4 device* accel_14 [[buffer(5)]], packed_float4 device* scratch_14 [[buffer(7)]], uint device* refs_14 [[buffer(3)]], BlastBond_natural_0 device* bonds_14 [[buffer(4)]], packed_float4 device* stiffness_14 [[buffer(10)]], packed_float4 device* x_14 [[buffer(6)]], packed_uint4 device* report_14 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_51;
    (&kernelContext_51)->meta_0 = meta_14;
    (&kernelContext_51)->topo_args_0 = topo_args_14;
    (&kernelContext_51)->islands_0 = islands_14;
    (&kernelContext_51)->params_0 = params_14;
    (&kernelContext_51)->wide_0 = wide_14;
    (&kernelContext_51)->lists_0 = lists_14;
    (&kernelContext_51)->nodes_0 = nodes_14;
    (&kernelContext_51)->accel_0 = accel_14;
    (&kernelContext_51)->scratch_0 = scratch_14;
    (&kernelContext_51)->refs_0 = refs_14;
    (&kernelContext_51)->bonds_0 = bonds_14;
    (&kernelContext_51)->stiffness_0 = stiffness_14;
    (&kernelContext_51)->x_0 = x_14;
    (&kernelContext_51)->report_0 = report_14;
    threadgroup array<float2, int(32)> g_blast_part_14;
    (&kernelContext_51)->g_blast_part_0 = &g_blast_part_14;
    threadgroup array<float, int(1024)> g_wide_max_14;
    (&kernelContext_51)->g_wide_max_0 = &g_wide_max_14;
    threadgroup float g_wide_max_out_14;
    (&kernelContext_51)->g_wide_max_out_0 = &g_wide_max_out_14;
    proj_part_0(g_8, t_9.x, 8U, false, &kernelContext_51);
    return;
}

[[kernel]] void w_proj_apply_g(uint3 g_9 [[threadgroup_position_in_grid]], uint3 t_10 [[thread_position_in_threadgroup]], atomic<uint> device* meta_15 [[buffer(12)]], uint device* topo_args_15 [[buffer(13)]], packed_uint4 device* islands_15 [[buffer(1)]], BlastParams_0 constant* params_15 [[buffer(0)]], packed_float4 device* wide_15 [[buffer(11)]], uint device* lists_15 [[buffer(9)]], BlastNode_natural_0 device* nodes_15 [[buffer(2)]], packed_float4 device* accel_15 [[buffer(5)]], packed_float4 device* scratch_15 [[buffer(7)]], uint device* refs_15 [[buffer(3)]], BlastBond_natural_0 device* bonds_15 [[buffer(4)]], packed_float4 device* stiffness_15 [[buffer(10)]], packed_float4 device* x_15 [[buffer(6)]], packed_uint4 device* report_15 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_52;
    (&kernelContext_52)->meta_0 = meta_15;
    (&kernelContext_52)->topo_args_0 = topo_args_15;
    (&kernelContext_52)->islands_0 = islands_15;
    (&kernelContext_52)->params_0 = params_15;
    (&kernelContext_52)->wide_0 = wide_15;
    (&kernelContext_52)->lists_0 = lists_15;
    (&kernelContext_52)->nodes_0 = nodes_15;
    (&kernelContext_52)->accel_0 = accel_15;
    (&kernelContext_52)->scratch_0 = scratch_15;
    (&kernelContext_52)->refs_0 = refs_15;
    (&kernelContext_52)->bonds_0 = bonds_15;
    (&kernelContext_52)->stiffness_0 = stiffness_15;
    (&kernelContext_52)->x_0 = x_15;
    (&kernelContext_52)->report_0 = report_15;
    threadgroup array<float2, int(32)> g_blast_part_15;
    (&kernelContext_52)->g_blast_part_0 = &g_blast_part_15;
    threadgroup array<float, int(1024)> g_wide_max_15;
    (&kernelContext_52)->g_wide_max_0 = &g_wide_max_15;
    threadgroup float g_wide_max_out_15;
    (&kernelContext_52)->g_wide_max_out_0 = &g_wide_max_out_15;
    proj_apply_0(g_9, t_10.x, 8U, false, &kernelContext_52);
    return;
}

void bond_r_0(uint3 group_9, uint tid_13, bool only_rebuild_4, KernelContext_0 thread* kernelContext_53)
{
    Block_0 _S209 = bond_block_0(group_9, tid_13, kernelContext_53);
    bool _S210;
    if((_S209.begin_0) >= (_S209.end_0))
    {
        _S210 = true;
    }
    else
    {
        _S210 = !runs_0(_S209.flags_0, only_rebuild_4);
    }
    if(_S210)
    {
        return;
    }
    uint p_5 = _S209.begin_0 + tid_13;
    float z_3;
    float f_4;
    if(p_5 < (_S209.end_0))
    {
        uint j_14 = kernelContext_53->lists_0[p_5];
        thread BlastBond_natural_0 _S211 = *(kernelContext_53->bonds_0+j_14);
        float3 _S212 = float3(0.0f) ;
        thread float3 ta_3 = _S212;
        thread float3 tl_3 = _S212;
        if(((&_S211)->health_0) > 0.0f)
        {
            bond_term_0(&_S211, j_14, 0U, &ta_3, &tl_3, kernelContext_53);
            if((kernelContext_53->params_0->force_tolerance_0) > 0.0f)
            {
                thread float3 ua_0;
                thread float3 ul_0;
                bond_term_0(&_S211, j_14, 6U, &ua_0, &ul_0, kernelContext_53);
                uint _S213 = j_14 * 2U;
                z_3 = norm2_0((float4(*(kernelContext_53->x_0+_S213)) ).xyz + ua_0, (float4(*(kernelContext_53->x_0+(_S213 + 1U))) ).xyz + ul_0);
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
        float _S214 = norm2_0(ta_3, tl_3);
        uint _S215 = bs_0(j_14, kernelContext_53);
        *(kernelContext_53->scratch_0+_S215) = packed_float4(float4(ta_3, 0.0f)) ;
        *(kernelContext_53->scratch_0+(_S215 + 1U)) = packed_float4(float4(tl_3, 0.0f)) ;
        float _S216 = z_3;
        z_3 = _S214;
        f_4 = _S216;
    }
    else
    {
        z_3 = 0.0f;
        f_4 = 0.0f;
    }
    write_partial_0(tid_13, _S209.row_4, group_9.x, 2U, float4(z_3, f_4, 0.0f, 0.0f), kernelContext_53);
    return;
}

[[kernel]] void w_bond_r(uint3 g_10 [[threadgroup_position_in_grid]], uint3 t_11 [[thread_position_in_threadgroup]], atomic<uint> device* meta_16 [[buffer(12)]], uint device* topo_args_16 [[buffer(13)]], packed_uint4 device* islands_16 [[buffer(1)]], BlastParams_0 constant* params_16 [[buffer(0)]], packed_float4 device* wide_16 [[buffer(11)]], uint device* lists_16 [[buffer(9)]], BlastNode_natural_0 device* nodes_16 [[buffer(2)]], packed_float4 device* accel_16 [[buffer(5)]], packed_float4 device* scratch_16 [[buffer(7)]], uint device* refs_16 [[buffer(3)]], BlastBond_natural_0 device* bonds_16 [[buffer(4)]], packed_float4 device* stiffness_16 [[buffer(10)]], packed_float4 device* x_16 [[buffer(6)]], packed_uint4 device* report_16 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_54;
    (&kernelContext_54)->meta_0 = meta_16;
    (&kernelContext_54)->topo_args_0 = topo_args_16;
    (&kernelContext_54)->islands_0 = islands_16;
    (&kernelContext_54)->params_0 = params_16;
    (&kernelContext_54)->wide_0 = wide_16;
    (&kernelContext_54)->lists_0 = lists_16;
    (&kernelContext_54)->nodes_0 = nodes_16;
    (&kernelContext_54)->accel_0 = accel_16;
    (&kernelContext_54)->scratch_0 = scratch_16;
    (&kernelContext_54)->refs_0 = refs_16;
    (&kernelContext_54)->bonds_0 = bonds_16;
    (&kernelContext_54)->stiffness_0 = stiffness_16;
    (&kernelContext_54)->x_0 = x_16;
    (&kernelContext_54)->report_0 = report_16;
    threadgroup array<float2, int(32)> g_blast_part_16;
    (&kernelContext_54)->g_blast_part_0 = &g_blast_part_16;
    threadgroup array<float, int(1024)> g_wide_max_16;
    (&kernelContext_54)->g_wide_max_0 = &g_wide_max_16;
    threadgroup float g_wide_max_out_16;
    (&kernelContext_54)->g_wide_max_out_0 = &g_wide_max_out_16;
    bond_r_0(g_10, t_11.x, false, &kernelContext_54);
    return;
}

void node_rn_0(uint3 group_10, uint tid_14, bool only_rebuild_5, KernelContext_0 thread* kernelContext_55)
{
    Block_0 _S217 = node_block_0(group_10, tid_14, kernelContext_55);
    bool _S218;
    if((_S217.begin_0) >= (_S217.end_0))
    {
        _S218 = true;
    }
    else
    {
        _S218 = !runs_0(_S217.flags_0, only_rebuild_5);
    }
    if(_S218)
    {
        return;
    }
    uint p_6 = _S217.begin_0 + tid_14;
    float s_12;
    if(p_6 < (_S217.end_0))
    {
        uint i_4 = kernelContext_55->lists_0[p_6];
        float3 _S219 = va_0(i_4, 0U, kernelContext_55);
        float3 _S220 = vl_0(i_4, 0U, kernelContext_55);
        s_12 = norm2_0(_S219, _S220);
    }
    else
    {
        s_12 = 0.0f;
    }
    write_partial_0(tid_14, _S217.row_4, group_10.x, 3U, float4(s_12, 0.0f, 0.0f, 0.0f), kernelContext_55);
    return;
}

[[kernel]] void w_node_rn(uint3 g_11 [[threadgroup_position_in_grid]], uint3 t_12 [[thread_position_in_threadgroup]], atomic<uint> device* meta_17 [[buffer(12)]], uint device* topo_args_17 [[buffer(13)]], packed_uint4 device* islands_17 [[buffer(1)]], BlastParams_0 constant* params_17 [[buffer(0)]], packed_float4 device* wide_17 [[buffer(11)]], uint device* lists_17 [[buffer(9)]], BlastNode_natural_0 device* nodes_17 [[buffer(2)]], packed_float4 device* accel_17 [[buffer(5)]], packed_float4 device* scratch_17 [[buffer(7)]], uint device* refs_17 [[buffer(3)]], BlastBond_natural_0 device* bonds_17 [[buffer(4)]], packed_float4 device* stiffness_17 [[buffer(10)]], packed_float4 device* x_17 [[buffer(6)]], packed_uint4 device* report_17 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_56;
    (&kernelContext_56)->meta_0 = meta_17;
    (&kernelContext_56)->topo_args_0 = topo_args_17;
    (&kernelContext_56)->islands_0 = islands_17;
    (&kernelContext_56)->params_0 = params_17;
    (&kernelContext_56)->wide_0 = wide_17;
    (&kernelContext_56)->lists_0 = lists_17;
    (&kernelContext_56)->nodes_0 = nodes_17;
    (&kernelContext_56)->accel_0 = accel_17;
    (&kernelContext_56)->scratch_0 = scratch_17;
    (&kernelContext_56)->refs_0 = refs_17;
    (&kernelContext_56)->bonds_0 = bonds_17;
    (&kernelContext_56)->stiffness_0 = stiffness_17;
    (&kernelContext_56)->x_0 = x_17;
    (&kernelContext_56)->report_0 = report_17;
    threadgroup array<float2, int(32)> g_blast_part_17;
    (&kernelContext_56)->g_blast_part_0 = &g_blast_part_17;
    threadgroup array<float, int(1024)> g_wide_max_17;
    (&kernelContext_56)->g_wide_max_0 = &g_wide_max_17;
    threadgroup float g_wide_max_out_17;
    (&kernelContext_56)->g_wide_max_out_0 = &g_wide_max_out_17;
    node_rn_0(g_11, t_12.x, false, &kernelContext_56);
    return;
}

[[kernel]] void w_bond_rb(uint3 g_12 [[threadgroup_position_in_grid]], uint3 t_13 [[thread_position_in_threadgroup]], atomic<uint> device* meta_18 [[buffer(12)]], uint device* topo_args_18 [[buffer(13)]], packed_uint4 device* islands_18 [[buffer(1)]], BlastParams_0 constant* params_18 [[buffer(0)]], packed_float4 device* wide_18 [[buffer(11)]], uint device* lists_18 [[buffer(9)]], BlastNode_natural_0 device* nodes_18 [[buffer(2)]], packed_float4 device* accel_18 [[buffer(5)]], packed_float4 device* scratch_18 [[buffer(7)]], uint device* refs_18 [[buffer(3)]], BlastBond_natural_0 device* bonds_18 [[buffer(4)]], packed_float4 device* stiffness_18 [[buffer(10)]], packed_float4 device* x_18 [[buffer(6)]], packed_uint4 device* report_18 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_57;
    (&kernelContext_57)->meta_0 = meta_18;
    (&kernelContext_57)->topo_args_0 = topo_args_18;
    (&kernelContext_57)->islands_0 = islands_18;
    (&kernelContext_57)->params_0 = params_18;
    (&kernelContext_57)->wide_0 = wide_18;
    (&kernelContext_57)->lists_0 = lists_18;
    (&kernelContext_57)->nodes_0 = nodes_18;
    (&kernelContext_57)->accel_0 = accel_18;
    (&kernelContext_57)->scratch_0 = scratch_18;
    (&kernelContext_57)->refs_0 = refs_18;
    (&kernelContext_57)->bonds_0 = bonds_18;
    (&kernelContext_57)->stiffness_0 = stiffness_18;
    (&kernelContext_57)->x_0 = x_18;
    (&kernelContext_57)->report_0 = report_18;
    threadgroup array<float2, int(32)> g_blast_part_18;
    (&kernelContext_57)->g_blast_part_0 = &g_blast_part_18;
    threadgroup array<float, int(1024)> g_wide_max_18;
    (&kernelContext_57)->g_wide_max_0 = &g_wide_max_18;
    threadgroup float g_wide_max_out_18;
    (&kernelContext_57)->g_wide_max_out_0 = &g_wide_max_out_18;
    bond_r_0(g_12, t_13.x, true, &kernelContext_57);
    return;
}

[[kernel]] void w_node_rnb(uint3 g_13 [[threadgroup_position_in_grid]], uint3 t_14 [[thread_position_in_threadgroup]], atomic<uint> device* meta_19 [[buffer(12)]], uint device* topo_args_19 [[buffer(13)]], packed_uint4 device* islands_19 [[buffer(1)]], BlastParams_0 constant* params_19 [[buffer(0)]], packed_float4 device* wide_19 [[buffer(11)]], uint device* lists_19 [[buffer(9)]], BlastNode_natural_0 device* nodes_19 [[buffer(2)]], packed_float4 device* accel_19 [[buffer(5)]], packed_float4 device* scratch_19 [[buffer(7)]], uint device* refs_19 [[buffer(3)]], BlastBond_natural_0 device* bonds_19 [[buffer(4)]], packed_float4 device* stiffness_19 [[buffer(10)]], packed_float4 device* x_19 [[buffer(6)]], packed_uint4 device* report_19 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_58;
    (&kernelContext_58)->meta_0 = meta_19;
    (&kernelContext_58)->topo_args_0 = topo_args_19;
    (&kernelContext_58)->islands_0 = islands_19;
    (&kernelContext_58)->params_0 = params_19;
    (&kernelContext_58)->wide_0 = wide_19;
    (&kernelContext_58)->lists_0 = lists_19;
    (&kernelContext_58)->nodes_0 = nodes_19;
    (&kernelContext_58)->accel_0 = accel_19;
    (&kernelContext_58)->scratch_0 = scratch_19;
    (&kernelContext_58)->refs_0 = refs_19;
    (&kernelContext_58)->bonds_0 = bonds_19;
    (&kernelContext_58)->stiffness_0 = stiffness_19;
    (&kernelContext_58)->x_0 = x_19;
    (&kernelContext_58)->report_0 = report_19;
    threadgroup array<float2, int(32)> g_blast_part_19;
    (&kernelContext_58)->g_blast_part_0 = &g_blast_part_19;
    threadgroup array<float, int(1024)> g_wide_max_19;
    (&kernelContext_58)->g_wide_max_0 = &g_wide_max_19;
    threadgroup float g_wide_max_out_19;
    (&kernelContext_58)->g_wide_max_out_0 = &g_wide_max_out_19;
    node_rn_0(g_13, t_14.x, true, &kernelContext_58);
    return;
}

uint iteration_of_0(uint row_13, KernelContext_0 thread* kernelContext_59)
{
    return (as_type<uint>(((float4(*(kernelContext_59->wide_0+st_0(row_13, 2U))) ).z)));
}

void set_flags_0(uint row_14, uint f_5, KernelContext_0 thread* kernelContext_60)
{
    uint _S221 = st_0(row_14, 2U);
    thread float4 c_4 = float4(*(kernelContext_60->wide_0+_S221)) ;
    c_4.w = (as_type<float>((f_5)));
    *(kernelContext_60->wide_0+_S221) = packed_float4(c_4) ;
    return;
}

[[kernel]] void w_test(uint3 group_11 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], atomic<uint> device* meta_20 [[buffer(12)]], uint device* topo_args_20 [[buffer(13)]], packed_uint4 device* islands_20 [[buffer(1)]], BlastParams_0 constant* params_20 [[buffer(0)]], packed_float4 device* wide_20 [[buffer(11)]], uint device* lists_20 [[buffer(9)]], BlastNode_natural_0 device* nodes_20 [[buffer(2)]], packed_float4 device* accel_20 [[buffer(5)]], packed_float4 device* scratch_20 [[buffer(7)]], uint device* refs_20 [[buffer(3)]], BlastBond_natural_0 device* bonds_20 [[buffer(4)]], packed_float4 device* stiffness_20 [[buffer(10)]], packed_float4 device* x_20 [[buffer(6)]], packed_uint4 device* report_20 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_61;
    (&kernelContext_61)->meta_0 = meta_20;
    (&kernelContext_61)->topo_args_0 = topo_args_20;
    (&kernelContext_61)->islands_0 = islands_20;
    (&kernelContext_61)->params_0 = params_20;
    (&kernelContext_61)->wide_0 = wide_20;
    (&kernelContext_61)->lists_0 = lists_20;
    (&kernelContext_61)->nodes_0 = nodes_20;
    (&kernelContext_61)->accel_0 = accel_20;
    (&kernelContext_61)->scratch_0 = scratch_20;
    (&kernelContext_61)->refs_0 = refs_20;
    (&kernelContext_61)->bonds_0 = bonds_20;
    (&kernelContext_61)->stiffness_0 = stiffness_20;
    (&kernelContext_61)->x_0 = x_20;
    (&kernelContext_61)->report_0 = report_20;
    threadgroup array<float2, int(32)> g_blast_part_20;
    (&kernelContext_61)->g_blast_part_0 = &g_blast_part_20;
    threadgroup array<float, int(1024)> g_wide_max_20;
    (&kernelContext_61)->g_wide_max_0 = &g_wide_max_20;
    threadgroup float g_wide_max_out_20;
    (&kernelContext_61)->g_wide_max_out_0 = &g_wide_max_out_20;
    uint tid_15 = thread_6.x;
    uint row_15 = group_11.x;
    uint _S222 = flags_of_0(row_15, &kernelContext_61);
    uint f_6 = group_uniform_0(_S222, tid_15);
    if((f_6 & 1U) == 0U)
    {
        return;
    }
    uint4 _S223 = row_range_0(tid_15, row_15, &kernelContext_61);
    float4 _S224 = row_sum_0(tid_15, row_15, blocks_0(_S223.w - _S223.z), 2U, &kernelContext_61);
    float4 _S225 = row_sum_0(tid_15, row_15, blocks_0(_S223.y - _S223.x), 3U, &kernelContext_61);
    if(tid_15 != 0U)
    {
        return;
    }
    uint _S226 = st_0(row_15, 0U);
    float4 _S227 = float4(*((&kernelContext_61)->wide_0+_S226)) ;
    thread float4 s0_0 = _S227;
    float delta2_1 = _S227.x;
    float _S228 = _S224.x;
    bool _S229 = _S228 <= delta2_1;
    bool hold_0;
    if(_S229)
    {
        hold_0 = !((_S225.x) <= delta2_1);
    }
    else
    {
        hold_0 = false;
    }
    uint _S230 = iteration_of_0(row_15, &kernelContext_61);
    uint f_7 = f_6 & 4294967199U;
    uint f_8;
    if(hold_0)
    {
        f_8 = f_7 | 32U;
    }
    else
    {
        f_8 = f_7;
    }
    bool _S231;
    if(_S230 > 0U)
    {
        _S231 = true;
    }
    else
    {
        _S231 = ((&kernelContext_61)->params_0->warm_0) != 0U;
    }
    if(_S231)
    {
        _S231 = _S229;
    }
    else
    {
        _S231 = false;
    }
    if(_S231)
    {
        hold_0 = !hold_0;
    }
    else
    {
        hold_0 = false;
    }
    if(hold_0)
    {
        f_8 = f_8 | 64U;
    }
    s0_0.y = _S228;
    *((&kernelContext_61)->wide_0+_S226) = packed_float4(s0_0) ;
    uint _S232 = st_0(row_15, 1U);
    *((&kernelContext_61)->wide_0+_S232) = packed_float4(float4(*((&kernelContext_61)->wide_0+_S232)) ) ;
    set_flags_0(row_15, f_8, &kernelContext_61);
    uint _S233 = st_0(row_15, 2U);
    float4 _S234 = float4(*((&kernelContext_61)->wide_0+_S233)) ;
    uint _S235 = st_0(row_15, 3U);
    *((&kernelContext_61)->wide_0+_S235) = packed_float4(float4((float4(*((&kernelContext_61)->wide_0+_S235)) ).xyz, (float4(*((&kernelContext_61)->wide_0+_S235)) ).w)) ;
    uint _S236 = pt_0(row_15, 0U, 5U, &kernelContext_61);
    *((&kernelContext_61)->wide_0+_S236) = packed_float4(float4(_S224.y, 0.0f, 0.0f, 0.0f)) ;
    *((&kernelContext_61)->wide_0+_S233) = packed_float4(_S234) ;
    return;
}

[[kernel]] void w_rebuild(uint3 group_12 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], atomic<uint> device* meta_21 [[buffer(12)]], uint device* topo_args_21 [[buffer(13)]], packed_uint4 device* islands_21 [[buffer(1)]], BlastParams_0 constant* params_21 [[buffer(0)]], packed_float4 device* wide_21 [[buffer(11)]], uint device* lists_21 [[buffer(9)]], BlastNode_natural_0 device* nodes_21 [[buffer(2)]], packed_float4 device* accel_21 [[buffer(5)]], packed_float4 device* scratch_21 [[buffer(7)]], uint device* refs_21 [[buffer(3)]], BlastBond_natural_0 device* bonds_21 [[buffer(4)]], packed_float4 device* stiffness_21 [[buffer(10)]], packed_float4 device* x_21 [[buffer(6)]], packed_uint4 device* report_21 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_62;
    (&kernelContext_62)->meta_0 = meta_21;
    (&kernelContext_62)->topo_args_0 = topo_args_21;
    (&kernelContext_62)->islands_0 = islands_21;
    (&kernelContext_62)->params_0 = params_21;
    (&kernelContext_62)->wide_0 = wide_21;
    (&kernelContext_62)->lists_0 = lists_21;
    (&kernelContext_62)->nodes_0 = nodes_21;
    (&kernelContext_62)->accel_0 = accel_21;
    (&kernelContext_62)->scratch_0 = scratch_21;
    (&kernelContext_62)->refs_0 = refs_21;
    (&kernelContext_62)->bonds_0 = bonds_21;
    (&kernelContext_62)->stiffness_0 = stiffness_21;
    (&kernelContext_62)->x_0 = x_21;
    (&kernelContext_62)->report_0 = report_21;
    threadgroup array<float2, int(32)> g_blast_part_21;
    (&kernelContext_62)->g_blast_part_0 = &g_blast_part_21;
    threadgroup array<float, int(1024)> g_wide_max_21;
    (&kernelContext_62)->g_wide_max_0 = &g_wide_max_21;
    threadgroup float g_wide_max_out_21;
    (&kernelContext_62)->g_wide_max_out_0 = &g_wide_max_out_21;
    uint tid_16 = thread_7.x;
    Block_0 _S237 = node_block_0(group_12, tid_16, &kernelContext_62);
    bool _S238;
    if((_S237.begin_0) >= (_S237.end_0))
    {
        _S238 = true;
    }
    else
    {
        _S238 = !runs_0(_S237.flags_0, true);
    }
    if(_S238)
    {
        return;
    }
    uint p_7 = _S237.begin_0 + tid_16;
    if(p_7 < (_S237.end_0))
    {
        uint i_5 = (&kernelContext_62)->lists_0[p_7];
        thread float3 a_3;
        thread float3 l_3;
        gather_0(i_5, false, true, &a_3, &l_3, &kernelContext_62);
        float3 _S239 = va_0(i_5, 14U, &kernelContext_62);
        float3 _S240 = _S239 - a_3;
        float3 _S241 = vl_0(i_5, 14U, &kernelContext_62);
        set_v_0(i_5, 0U, _S240, _S241 - l_3, &kernelContext_62);
    }
    return;
}

[[kernel]] void w_decide(uint3 group_13 [[threadgroup_position_in_grid]], uint3 thread_8 [[thread_position_in_threadgroup]], atomic<uint> device* meta_22 [[buffer(12)]], uint device* topo_args_22 [[buffer(13)]], packed_uint4 device* islands_22 [[buffer(1)]], BlastParams_0 constant* params_22 [[buffer(0)]], packed_float4 device* wide_22 [[buffer(11)]], uint device* lists_22 [[buffer(9)]], BlastNode_natural_0 device* nodes_22 [[buffer(2)]], packed_float4 device* accel_22 [[buffer(5)]], packed_float4 device* scratch_22 [[buffer(7)]], uint device* refs_22 [[buffer(3)]], BlastBond_natural_0 device* bonds_22 [[buffer(4)]], packed_float4 device* stiffness_22 [[buffer(10)]], packed_float4 device* x_22 [[buffer(6)]], packed_uint4 device* report_22 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_63;
    (&kernelContext_63)->meta_0 = meta_22;
    (&kernelContext_63)->topo_args_0 = topo_args_22;
    (&kernelContext_63)->islands_0 = islands_22;
    (&kernelContext_63)->params_0 = params_22;
    (&kernelContext_63)->wide_0 = wide_22;
    (&kernelContext_63)->lists_0 = lists_22;
    (&kernelContext_63)->nodes_0 = nodes_22;
    (&kernelContext_63)->accel_0 = accel_22;
    (&kernelContext_63)->scratch_0 = scratch_22;
    (&kernelContext_63)->refs_0 = refs_22;
    (&kernelContext_63)->bonds_0 = bonds_22;
    (&kernelContext_63)->stiffness_0 = stiffness_22;
    (&kernelContext_63)->x_0 = x_22;
    (&kernelContext_63)->report_0 = report_22;
    threadgroup array<float2, int(32)> g_blast_part_22;
    (&kernelContext_63)->g_blast_part_0 = &g_blast_part_22;
    threadgroup array<float, int(1024)> g_wide_max_22;
    (&kernelContext_63)->g_wide_max_0 = &g_wide_max_22;
    threadgroup float g_wide_max_out_22;
    (&kernelContext_63)->g_wide_max_out_0 = &g_wide_max_out_22;
    uint tid_17 = thread_8.x;
    uint row_16 = group_13.x;
    uint _S242 = flags_of_0(row_16, &kernelContext_63);
    uint f_9 = group_uniform_0(_S242, tid_17);
    if((f_9 & 1U) == 0U)
    {
        return;
    }
    uint4 _S243 = row_range_0(tid_17, row_16, &kernelContext_63);
    float4 _S244 = float4(0.0f) ;
    bool _S245 = (f_9 & 64U) != 0U;
    float4 z_4;
    float4 rn_0;
    if(_S245)
    {
        float4 _S246 = row_sum_0(tid_17, row_16, blocks_0(_S243.w - _S243.z), 2U, &kernelContext_63);
        float4 _S247 = row_sum_0(tid_17, row_16, blocks_0(_S243.y - _S243.x), 3U, &kernelContext_63);
        z_4 = _S246;
        rn_0 = _S247;
    }
    else
    {
        z_4 = _S244;
        rn_0 = _S244;
    }
    if(tid_17 != 0U)
    {
        return;
    }
    uint _S248 = st_0(row_16, 0U);
    float4 _S249 = float4(*((&kernelContext_63)->wide_0+_S248)) ;
    thread float4 s0_1 = _S249;
    float4 _S250 = float4(*((&kernelContext_63)->wide_0+st_0(row_16, 1U))) ;
    float4 _S251 = float4(*((&kernelContext_63)->wide_0+st_0(row_16, 2U))) ;
    uint _S252 = st_0(row_16, 3U);
    float4 _S253 = float4(*((&kernelContext_63)->wide_0+_S252)) ;
    float delta2_2 = _S249.x;
    float _S254 = _S249.y;
    bool _S255 = (f_9 & 32U) != 0U;
    bool hold_1;
    float num_0;
    if(_S245)
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
        num_0 = _S254;
        hold_1 = _S255;
    }
    uint f_10 = f_9 & 4294967199U;
    uint it_0 = (as_type<uint>((_S251.z)));
    uint best_it_0 = (as_type<uint>((_S253.w)));
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
    bool _S256;
    if(!stop_0)
    {
        _S256 = num_0 <= delta2_2;
    }
    else
    {
        _S256 = false;
    }
    if(_S256)
    {
        hold_1 = !hold_1;
    }
    else
    {
        hold_1 = false;
    }
    uint f_11;
    if(hold_1)
    {
        uint f_12 = f_10 | 2U;
        stop_0 = true;
        f_11 = f_12;
    }
    else
    {
        f_11 = f_10;
    }
    if(!stop_0)
    {
        hold_1 = ((&kernelContext_63)->params_0->force_tolerance_0) > 0.0f;
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
        hold_1 = (_S250.w) <= ((&kernelContext_63)->params_0->force_tolerance_0 * (_S251.x + _S251.y));
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        float _S257 = _S250.w;
        float _S258 = (&kernelContext_63)->params_0->force_tolerance_0;
        uint _S259 = pt_0(row_16, 0U, 5U, &kernelContext_63);
        if(_S257 <= (_S258 * sqrt((float4(*((&kernelContext_63)->wide_0+_S259)) ).x)))
        {
            uint f_13 = f_11 | 2U;
            stop_0 = true;
            f_11 = f_13;
        }
    }
    if(stop_0)
    {
        hold_1 = true;
    }
    else
    {
        hold_1 = it_0 >= ((&kernelContext_63)->params_0->max_iterations_0);
    }
    if(hold_1)
    {
        f_11 = f_11 & 4294967294U;
    }
    s0_1.y = num_0;
    *((&kernelContext_63)->wide_0+_S248) = packed_float4(s0_1) ;
    *((&kernelContext_63)->wide_0+_S252) = packed_float4(float4(_S253.xyz, (as_type<float>((best_it_1))))) ;
    set_flags_0(row_16, f_11, &kernelContext_63);
    return;
}

void full_apply_0(uint n_11, uint at_1, float3 a_4, float3 l_4, float3 thread* oa_0, float3 thread* ol_0, KernelContext_0 thread* kernelContext_64)
{
    thread array<float, int(24)> m_3;
    uint q_1 = 0U;
    for(;;)
    {
        if(q_1 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S260 = ns_0(n_11, at_1 + q_1, kernelContext_64);
        float4 _S261 = float4(*(kernelContext_64->scratch_0+_S260)) ;
        uint _S262 = 4U * q_1;
        m_3[_S262] = _S261.x;
        m_3[_S262 + 1U] = _S261.y;
        m_3[_S262 + 2U] = _S261.z;
        m_3[_S262 + 3U] = _S261.w;
        q_1 = q_1 + 1U;
    }
    array<float, int(6)> _S263 = { { a_4.x, a_4.y, a_4.z, l_4.x, l_4.y, l_4.z } };
    thread array<float, int(6)> o_3;
    uint r_6 = 0U;
    for(;;)
    {
        if(r_6 < 6U)
        {
        }
        else
        {
            break;
        }
        uint c_5 = 0U;
        float s_13 = 0.0f;
        for(;;)
        {
            if(c_5 < 6U)
            {
            }
            else
            {
                break;
            }
            if(r_6 >= c_5)
            {
                q_1 = r_6 * (r_6 + 1U) / 2U + c_5;
            }
            else
            {
                q_1 = c_5 * (c_5 + 1U) / 2U + r_6;
            }
            float s_14 = s_13 + m_3[q_1] * _S263[c_5];
            c_5 = c_5 + 1U;
            s_13 = s_14;
        }
        o_3[r_6] = s_13;
        r_6 = r_6 + 1U;
    }
    *oa_0 = float3(o_3[int(0)], o_3[int(1)], o_3[int(2)]);
    *ol_0 = float3(o_3[int(3)], o_3[int(4)], o_3[int(5)]);
    return;
}

float3 sym_mul_0(float4 s0_2, float4 s1_0, float3 v_17)
{
    float _S264 = v_17.x;
    float _S265 = s0_2.w;
    float _S266 = v_17.y;
    float _S267 = s1_0.x;
    float _S268 = v_17.z;
    float _S269 = s1_0.y;
    return float3(s0_2.x * _S264 + _S265 * _S266 + _S267 * _S268, _S265 * _S264 + s0_2.y * _S266 + _S269 * _S268, _S267 * _S264 + _S269 * _S266 + s0_2.z * _S268);
}

void inverse_apply_0(uint n_12, float3 a_5, float3 l_5, float3 thread* oa_1, float3 thread* ol_1, KernelContext_0 thread* kernelContext_65)
{
    if((kernelContext_65->params_0->rotation_0) == 2U)
    {
        full_apply_0(n_12, 24U, a_5, l_5, oa_1, ol_1, kernelContext_65);
        return;
    }
    uint _S270 = ns_0(n_12, 21U, kernelContext_65);
    float4 _S271 = float4(*(kernelContext_65->scratch_0+(_S270 + 1U))) ;
    float3 kc_0 = (float4(*(kernelContext_65->scratch_0+(_S270 + 2U))) ).xyz;
    float3 _S272 = sym_mul_0(float4(*(kernelContext_65->scratch_0+_S270)) , _S271, a_5 + cross(kc_0, l_5));
    *oa_1 = _S272;
    *ol_1 = l_5 * float3(_S271.z)  - cross(kc_0, _S272);
    return;
}

[[kernel]] void w_pre1(uint3 group_14 [[threadgroup_position_in_grid]], uint3 thread_9 [[thread_position_in_threadgroup]], atomic<uint> device* meta_23 [[buffer(12)]], uint device* topo_args_23 [[buffer(13)]], packed_uint4 device* islands_23 [[buffer(1)]], BlastParams_0 constant* params_23 [[buffer(0)]], packed_float4 device* wide_23 [[buffer(11)]], uint device* lists_23 [[buffer(9)]], BlastNode_natural_0 device* nodes_23 [[buffer(2)]], packed_float4 device* accel_23 [[buffer(5)]], packed_float4 device* scratch_23 [[buffer(7)]], uint device* refs_23 [[buffer(3)]], BlastBond_natural_0 device* bonds_23 [[buffer(4)]], packed_float4 device* stiffness_23 [[buffer(10)]], packed_float4 device* x_23 [[buffer(6)]], packed_uint4 device* report_23 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_66;
    (&kernelContext_66)->meta_0 = meta_23;
    (&kernelContext_66)->topo_args_0 = topo_args_23;
    (&kernelContext_66)->islands_0 = islands_23;
    (&kernelContext_66)->params_0 = params_23;
    (&kernelContext_66)->wide_0 = wide_23;
    (&kernelContext_66)->lists_0 = lists_23;
    (&kernelContext_66)->nodes_0 = nodes_23;
    (&kernelContext_66)->accel_0 = accel_23;
    (&kernelContext_66)->scratch_0 = scratch_23;
    (&kernelContext_66)->refs_0 = refs_23;
    (&kernelContext_66)->bonds_0 = bonds_23;
    (&kernelContext_66)->stiffness_0 = stiffness_23;
    (&kernelContext_66)->x_0 = x_23;
    (&kernelContext_66)->report_0 = report_23;
    threadgroup array<float2, int(32)> g_blast_part_23;
    (&kernelContext_66)->g_blast_part_0 = &g_blast_part_23;
    threadgroup array<float, int(1024)> g_wide_max_23;
    (&kernelContext_66)->g_wide_max_0 = &g_wide_max_23;
    threadgroup float g_wide_max_out_23;
    (&kernelContext_66)->g_wide_max_out_0 = &g_wide_max_out_23;
    uint tid_18 = thread_9.x;
    Block_0 _S273 = node_block_0(group_14, tid_18, &kernelContext_66);
    bool _S274;
    if((_S273.begin_0) >= (_S273.end_0))
    {
        _S274 = true;
    }
    else
    {
        _S274 = !runs_0(_S273.flags_0, false);
    }
    if(_S274)
    {
        return;
    }
    uint p_8 = _S273.begin_0 + tid_18;
    if(p_8 >= (_S273.end_0))
    {
        return;
    }
    uint i_6 = (&kernelContext_66)->lists_0[p_8];
    uint _S275 = iteration_of_0(_S273.row_4, &kernelContext_66);
    if(_S275 == 0U)
    {
        _S274 = ((&kernelContext_66)->params_0->first_polynomial_0) == 0U;
    }
    else
    {
        _S274 = false;
    }
    if(_S274)
    {
        _S274 = ((_S273.flags_0) & 256U) == 0U;
    }
    else
    {
        _S274 = false;
    }
    if(_S274)
    {
        float3 _S276 = va_0(i_6, 0U, &kernelContext_66);
        float3 _S277 = vl_0(i_6, 0U, &kernelContext_66);
        set_v_0(i_6, 8U, _S276, _S277, &kernelContext_66);
        return;
    }
    float3 _S278 = va_0(i_6, 0U, &kernelContext_66);
    float3 _S279 = vl_0(i_6, 0U, &kernelContext_66);
    thread float3 a_6;
    thread float3 l_6;
    inverse_apply_0(i_6, _S278, _S279, &a_6, &l_6, &kernelContext_66);
    set_v_0(i_6, 10U, a_6, l_6, &kernelContext_66);
    return;
}

[[kernel]] void w_bond_loc(uint3 group_15 [[threadgroup_position_in_grid]], uint3 thread_10 [[thread_position_in_threadgroup]], atomic<uint> device* meta_24 [[buffer(12)]], uint device* topo_args_24 [[buffer(13)]], packed_uint4 device* islands_24 [[buffer(1)]], BlastParams_0 constant* params_24 [[buffer(0)]], packed_float4 device* wide_24 [[buffer(11)]], uint device* lists_24 [[buffer(9)]], BlastNode_natural_0 device* nodes_24 [[buffer(2)]], packed_float4 device* accel_24 [[buffer(5)]], packed_float4 device* scratch_24 [[buffer(7)]], uint device* refs_24 [[buffer(3)]], BlastBond_natural_0 device* bonds_24 [[buffer(4)]], packed_float4 device* stiffness_24 [[buffer(10)]], packed_float4 device* x_24 [[buffer(6)]], packed_uint4 device* report_24 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_67;
    (&kernelContext_67)->meta_0 = meta_24;
    (&kernelContext_67)->topo_args_0 = topo_args_24;
    (&kernelContext_67)->islands_0 = islands_24;
    (&kernelContext_67)->params_0 = params_24;
    (&kernelContext_67)->wide_0 = wide_24;
    (&kernelContext_67)->lists_0 = lists_24;
    (&kernelContext_67)->nodes_0 = nodes_24;
    (&kernelContext_67)->accel_0 = accel_24;
    (&kernelContext_67)->scratch_0 = scratch_24;
    (&kernelContext_67)->refs_0 = refs_24;
    (&kernelContext_67)->bonds_0 = bonds_24;
    (&kernelContext_67)->stiffness_0 = stiffness_24;
    (&kernelContext_67)->x_0 = x_24;
    (&kernelContext_67)->report_0 = report_24;
    threadgroup array<float2, int(32)> g_blast_part_24;
    (&kernelContext_67)->g_blast_part_0 = &g_blast_part_24;
    threadgroup array<float, int(1024)> g_wide_max_24;
    (&kernelContext_67)->g_wide_max_0 = &g_wide_max_24;
    threadgroup float g_wide_max_out_24;
    (&kernelContext_67)->g_wide_max_out_0 = &g_wide_max_out_24;
    uint tid_19 = thread_10.x;
    Block_0 _S280 = bond_block_0(group_15, tid_19, &kernelContext_67);
    bool _S281;
    if((_S280.begin_0) >= (_S280.end_0))
    {
        _S281 = true;
    }
    else
    {
        _S281 = !runs_0(_S280.flags_0, false);
    }
    if(_S281)
    {
        _S281 = true;
    }
    else
    {
        uint _S282 = iteration_of_0(_S280.row_4, &kernelContext_67);
        if(_S282 == 0U)
        {
            _S281 = ((&kernelContext_67)->params_0->first_polynomial_0) == 0U;
        }
        else
        {
            _S281 = false;
        }
        if(_S281)
        {
            _S281 = ((_S280.flags_0) & 256U) == 0U;
        }
        else
        {
            _S281 = false;
        }
    }
    if(_S281)
    {
        return;
    }
    uint p_9 = _S280.begin_0 + tid_19;
    if(p_9 >= (_S280.end_0))
    {
        return;
    }
    uint j_15 = (&kernelContext_67)->lists_0[p_9];
    thread BlastBond_natural_0 _S283 = *((&kernelContext_67)->bonds_0+j_15);
    float3 _S284 = float3(0.0f) ;
    thread float3 ta_4 = _S284;
    thread float3 tl_4 = _S284;
    if(((&_S283)->health_0) > 0.0f)
    {
        bond_term_1(&_S283, j_15, 10U, &ta_4, &tl_4, &kernelContext_67);
    }
    uint _S285 = bs_0(j_15, &kernelContext_67);
    *((&kernelContext_67)->scratch_0+_S285) = packed_float4(float4(ta_4, 0.0f)) ;
    *((&kernelContext_67)->scratch_0+(_S285 + 1U)) = packed_float4(float4(tl_4, 0.0f)) ;
    return;
}

void block_apply_0(uint n_13, float3 a_7, float3 l_7, float3 thread* oa_2, float3 thread* ol_2, KernelContext_0 thread* kernelContext_68)
{
    if((kernelContext_68->params_0->rotation_0) == 2U)
    {
        full_apply_0(n_13, 18U, a_7, l_7, oa_2, ol_2, kernelContext_68);
        return;
    }
    uint _S286 = ns_0(n_13, 18U, kernelContext_68);
    float4 _S287 = float4(*(kernelContext_68->scratch_0+(_S286 + 1U))) ;
    float3 k_20 = (float4(*(kernelContext_68->scratch_0+(_S286 + 2U))) ).xyz;
    *oa_2 = sym_mul_0(float4(*(kernelContext_68->scratch_0+_S286)) , _S287, a_7) - cross(k_20, l_7);
    *ol_2 = cross(k_20, a_7) + float3(_S287.z)  * l_7;
    return;
}

[[kernel]] void w_pre2(uint3 group_16 [[threadgroup_position_in_grid]], uint3 thread_11 [[thread_position_in_threadgroup]], atomic<uint> device* meta_25 [[buffer(12)]], uint device* topo_args_25 [[buffer(13)]], packed_uint4 device* islands_25 [[buffer(1)]], BlastParams_0 constant* params_25 [[buffer(0)]], packed_float4 device* wide_25 [[buffer(11)]], uint device* lists_25 [[buffer(9)]], BlastNode_natural_0 device* nodes_25 [[buffer(2)]], packed_float4 device* accel_25 [[buffer(5)]], packed_float4 device* scratch_25 [[buffer(7)]], uint device* refs_25 [[buffer(3)]], BlastBond_natural_0 device* bonds_25 [[buffer(4)]], packed_float4 device* stiffness_25 [[buffer(10)]], packed_float4 device* x_25 [[buffer(6)]], packed_uint4 device* report_25 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_69;
    (&kernelContext_69)->meta_0 = meta_25;
    (&kernelContext_69)->topo_args_0 = topo_args_25;
    (&kernelContext_69)->islands_0 = islands_25;
    (&kernelContext_69)->params_0 = params_25;
    (&kernelContext_69)->wide_0 = wide_25;
    (&kernelContext_69)->lists_0 = lists_25;
    (&kernelContext_69)->nodes_0 = nodes_25;
    (&kernelContext_69)->accel_0 = accel_25;
    (&kernelContext_69)->scratch_0 = scratch_25;
    (&kernelContext_69)->refs_0 = refs_25;
    (&kernelContext_69)->bonds_0 = bonds_25;
    (&kernelContext_69)->stiffness_0 = stiffness_25;
    (&kernelContext_69)->x_0 = x_25;
    (&kernelContext_69)->report_0 = report_25;
    threadgroup array<float2, int(32)> g_blast_part_25;
    (&kernelContext_69)->g_blast_part_0 = &g_blast_part_25;
    threadgroup array<float, int(1024)> g_wide_max_25;
    (&kernelContext_69)->g_wide_max_0 = &g_wide_max_25;
    threadgroup float g_wide_max_out_25;
    (&kernelContext_69)->g_wide_max_out_0 = &g_wide_max_out_25;
    uint tid_20 = thread_11.x;
    Block_0 _S288 = node_block_0(group_16, tid_20, &kernelContext_69);
    bool _S289;
    if((_S288.begin_0) >= (_S288.end_0))
    {
        _S289 = true;
    }
    else
    {
        _S289 = !runs_0(_S288.flags_0, false);
    }
    if(_S289)
    {
        _S289 = true;
    }
    else
    {
        uint _S290 = iteration_of_0(_S288.row_4, &kernelContext_69);
        if(_S290 == 0U)
        {
            _S289 = ((&kernelContext_69)->params_0->first_polynomial_0) == 0U;
        }
        else
        {
            _S289 = false;
        }
        if(_S289)
        {
            _S289 = ((_S288.flags_0) & 256U) == 0U;
        }
        else
        {
            _S289 = false;
        }
    }
    if(_S289)
    {
        return;
    }
    uint p_10 = _S288.begin_0 + tid_20;
    if(p_10 >= (_S288.end_0))
    {
        return;
    }
    uint i_7 = (&kernelContext_69)->lists_0[p_10];
    thread float3 oa_3;
    thread float3 ol_3;
    gather_0(i_7, true, false, &oa_3, &ol_3, &kernelContext_69);
    float3 _S291 = va_0(i_7, 10U, &kernelContext_69);
    float3 _S292 = vl_0(i_7, 10U, &kernelContext_69);
    thread float3 sa_0;
    thread float3 sl_0;
    block_apply_0(i_7, _S291, _S292, &sa_0, &sl_0, &kernelContext_69);
    thread float3 ma_0;
    thread float3 ml_0;
    inverse_apply_0(i_7, oa_3 - sa_0, ol_3 - sl_0, &ma_0, &ml_0, &kernelContext_69);
    float3 _S293 = va_0(i_7, 10U, &kernelContext_69);
    float3 _S294 = float3(1.68946564197540283f) ;
    float3 _S295 = float3(1.5220409631729126f) ;
    float3 _S296 = _S294 * _S293 - _S295 * ma_0;
    float3 _S297 = vl_0(i_7, 10U, &kernelContext_69);
    set_v_0(i_7, 8U, _S296, _S294 * _S297 - _S295 * ml_0, &kernelContext_69);
    return;
}

float wide_group_max_0(uint tid_21, float v_18, KernelContext_0 thread* kernelContext_70)
{
    (*kernelContext_70->g_wide_max_0)[tid_21] = v_18;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint k_21 = 512U;
    for(;;)
    {
        if(k_21 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_21 < k_21)
        {
            (*kernelContext_70->g_wide_max_0)[tid_21] = max((*kernelContext_70->g_wide_max_0)[tid_21], (*kernelContext_70->g_wide_max_0)[tid_21 + k_21]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_21 = k_21 >> 1U;
    }
    if(tid_21 == 0U)
    {
        *kernelContext_70->g_wide_max_out_0 = (*kernelContext_70->g_wide_max_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S298 = kernelContext_70->g_wide_max_out_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *_S298;
}

[[kernel]] void w_max(uint3 group_17 [[threadgroup_position_in_grid]], uint3 thread_12 [[thread_position_in_threadgroup]], atomic<uint> device* meta_26 [[buffer(12)]], uint device* topo_args_26 [[buffer(13)]], packed_uint4 device* islands_26 [[buffer(1)]], BlastParams_0 constant* params_26 [[buffer(0)]], packed_float4 device* wide_26 [[buffer(11)]], uint device* lists_26 [[buffer(9)]], BlastNode_natural_0 device* nodes_26 [[buffer(2)]], packed_float4 device* accel_26 [[buffer(5)]], packed_float4 device* scratch_26 [[buffer(7)]], uint device* refs_26 [[buffer(3)]], BlastBond_natural_0 device* bonds_26 [[buffer(4)]], packed_float4 device* stiffness_26 [[buffer(10)]], packed_float4 device* x_26 [[buffer(6)]], packed_uint4 device* report_26 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_71;
    (&kernelContext_71)->meta_0 = meta_26;
    (&kernelContext_71)->topo_args_0 = topo_args_26;
    (&kernelContext_71)->islands_0 = islands_26;
    (&kernelContext_71)->params_0 = params_26;
    (&kernelContext_71)->wide_0 = wide_26;
    (&kernelContext_71)->lists_0 = lists_26;
    (&kernelContext_71)->nodes_0 = nodes_26;
    (&kernelContext_71)->accel_0 = accel_26;
    (&kernelContext_71)->scratch_0 = scratch_26;
    (&kernelContext_71)->refs_0 = refs_26;
    (&kernelContext_71)->bonds_0 = bonds_26;
    (&kernelContext_71)->stiffness_0 = stiffness_26;
    (&kernelContext_71)->x_0 = x_26;
    (&kernelContext_71)->report_0 = report_26;
    threadgroup array<float2, int(32)> g_blast_part_26;
    (&kernelContext_71)->g_blast_part_0 = &g_blast_part_26;
    threadgroup array<float, int(1024)> g_wide_max_26;
    (&kernelContext_71)->g_wide_max_0 = &g_wide_max_26;
    threadgroup float g_wide_max_out_26;
    (&kernelContext_71)->g_wide_max_out_0 = &g_wide_max_out_26;
    uint tid_22 = thread_12.x;
    Block_0 _S299 = node_block_0(group_17, tid_22, &kernelContext_71);
    bool _S300;
    if((_S299.begin_0) >= (_S299.end_0))
    {
        _S300 = true;
    }
    else
    {
        _S300 = !runs_0(_S299.flags_0, false);
    }
    if(_S300)
    {
        return;
    }
    uint p_11 = _S299.begin_0 + tid_22;
    float m_4;
    if(p_11 < (_S299.end_0))
    {
        uint i_8 = (&kernelContext_71)->lists_0[p_11];
        float3 _S301 = va_0(i_8, 8U, &kernelContext_71);
        float3 a_8 = abs(_S301);
        float3 _S302 = vl_0(i_8, 8U, &kernelContext_71);
        float3 l_8 = abs(_S302);
        m_4 = max(max(max(a_8.x, a_8.y), max(a_8.z, l_8.x)), max(l_8.y, l_8.z));
    }
    else
    {
        m_4 = 0.0f;
    }
    float _S303 = wide_group_max_0(tid_22, m_4, &kernelContext_71);
    if(tid_22 == 0U)
    {
        uint _S304 = pt_0(_S299.row_4, group_17.x, 4U, &kernelContext_71);
        *((&kernelContext_71)->wide_0+_S304) = packed_float4(float4(_S303, 0.0f, 0.0f, 0.0f)) ;
    }
    return;
}

[[kernel]] void w_mag(uint3 group_18 [[threadgroup_position_in_grid]], uint3 thread_13 [[thread_position_in_threadgroup]], atomic<uint> device* meta_27 [[buffer(12)]], uint device* topo_args_27 [[buffer(13)]], packed_uint4 device* islands_27 [[buffer(1)]], BlastParams_0 constant* params_27 [[buffer(0)]], packed_float4 device* wide_27 [[buffer(11)]], uint device* lists_27 [[buffer(9)]], BlastNode_natural_0 device* nodes_27 [[buffer(2)]], packed_float4 device* accel_27 [[buffer(5)]], packed_float4 device* scratch_27 [[buffer(7)]], uint device* refs_27 [[buffer(3)]], BlastBond_natural_0 device* bonds_27 [[buffer(4)]], packed_float4 device* stiffness_27 [[buffer(10)]], packed_float4 device* x_27 [[buffer(6)]], packed_uint4 device* report_27 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_72;
    (&kernelContext_72)->meta_0 = meta_27;
    (&kernelContext_72)->topo_args_0 = topo_args_27;
    (&kernelContext_72)->islands_0 = islands_27;
    (&kernelContext_72)->params_0 = params_27;
    (&kernelContext_72)->wide_0 = wide_27;
    (&kernelContext_72)->lists_0 = lists_27;
    (&kernelContext_72)->nodes_0 = nodes_27;
    (&kernelContext_72)->accel_0 = accel_27;
    (&kernelContext_72)->scratch_0 = scratch_27;
    (&kernelContext_72)->refs_0 = refs_27;
    (&kernelContext_72)->bonds_0 = bonds_27;
    (&kernelContext_72)->stiffness_0 = stiffness_27;
    (&kernelContext_72)->x_0 = x_27;
    (&kernelContext_72)->report_0 = report_27;
    threadgroup array<float2, int(32)> g_blast_part_27;
    (&kernelContext_72)->g_blast_part_0 = &g_blast_part_27;
    threadgroup array<float, int(1024)> g_wide_max_27;
    (&kernelContext_72)->g_wide_max_0 = &g_wide_max_27;
    threadgroup float g_wide_max_out_27;
    (&kernelContext_72)->g_wide_max_out_0 = &g_wide_max_out_27;
    uint tid_23 = thread_13.x;
    uint row_17 = group_18.x;
    uint _S305 = flags_of_0(row_17, &kernelContext_72);
    uint f_14 = group_uniform_0(_S305, tid_23);
    if((f_14 & 1U) == 0U)
    {
        return;
    }
    uint4 _S306 = row_range_0(tid_23, row_17, &kernelContext_72);
    uint _S307 = blocks_0(_S306.y - _S306.x);
    float m_5 = 0.0f;
    uint blk_0 = tid_23;
    for(;;)
    {
        if(blk_0 < _S307)
        {
        }
        else
        {
            break;
        }
        uint _S308 = pt_0(row_17, blk_0, 4U, &kernelContext_72);
        float _S309 = max(m_5, (float4(*((&kernelContext_72)->wide_0+_S308)) ).x);
        uint blk_1 = blk_0 + 1024U;
        m_5 = _S309;
        blk_0 = blk_1;
    }
    float _S310 = wide_group_max_0(tid_23, m_5, &kernelContext_72);
    if(tid_23 != 0U)
    {
        return;
    }
    bool _S311;
    if(!(_S310 > 0.0f))
    {
        _S311 = true;
    }
    else
    {
        _S311 = !isfinite(_S310);
    }
    if(_S311)
    {
        set_flags_0(row_17, (f_14 & 4294967294U) | 4U, &kernelContext_72);
        return;
    }
    uint _S312 = st_0(row_17, 1U);
    thread float4 s1_1 = float4(*((&kernelContext_72)->wide_0+_S312)) ;
    s1_1.y = 1.0f / _S310;
    *((&kernelContext_72)->wide_0+_S312) = packed_float4(s1_1) ;
    return;
}

[[kernel]] void w_scale(uint3 group_19 [[threadgroup_position_in_grid]], uint3 thread_14 [[thread_position_in_threadgroup]], atomic<uint> device* meta_28 [[buffer(12)]], uint device* topo_args_28 [[buffer(13)]], packed_uint4 device* islands_28 [[buffer(1)]], BlastParams_0 constant* params_28 [[buffer(0)]], packed_float4 device* wide_28 [[buffer(11)]], uint device* lists_28 [[buffer(9)]], BlastNode_natural_0 device* nodes_28 [[buffer(2)]], packed_float4 device* accel_28 [[buffer(5)]], packed_float4 device* scratch_28 [[buffer(7)]], uint device* refs_28 [[buffer(3)]], BlastBond_natural_0 device* bonds_28 [[buffer(4)]], packed_float4 device* stiffness_28 [[buffer(10)]], packed_float4 device* x_28 [[buffer(6)]], packed_uint4 device* report_28 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_73;
    (&kernelContext_73)->meta_0 = meta_28;
    (&kernelContext_73)->topo_args_0 = topo_args_28;
    (&kernelContext_73)->islands_0 = islands_28;
    (&kernelContext_73)->params_0 = params_28;
    (&kernelContext_73)->wide_0 = wide_28;
    (&kernelContext_73)->lists_0 = lists_28;
    (&kernelContext_73)->nodes_0 = nodes_28;
    (&kernelContext_73)->accel_0 = accel_28;
    (&kernelContext_73)->scratch_0 = scratch_28;
    (&kernelContext_73)->refs_0 = refs_28;
    (&kernelContext_73)->bonds_0 = bonds_28;
    (&kernelContext_73)->stiffness_0 = stiffness_28;
    (&kernelContext_73)->x_0 = x_28;
    (&kernelContext_73)->report_0 = report_28;
    threadgroup array<float2, int(32)> g_blast_part_28;
    (&kernelContext_73)->g_blast_part_0 = &g_blast_part_28;
    threadgroup array<float, int(1024)> g_wide_max_28;
    (&kernelContext_73)->g_wide_max_0 = &g_wide_max_28;
    threadgroup float g_wide_max_out_28;
    (&kernelContext_73)->g_wide_max_out_0 = &g_wide_max_out_28;
    uint tid_24 = thread_14.x;
    Block_0 _S313 = node_block_0(group_19, tid_24, &kernelContext_73);
    bool _S314;
    if((_S313.begin_0) >= (_S313.end_0))
    {
        _S314 = true;
    }
    else
    {
        _S314 = !runs_0(_S313.flags_0, false);
    }
    if(_S314)
    {
        return;
    }
    uint p_12 = _S313.begin_0 + tid_24;
    float gl_0;
    if(p_12 < (_S313.end_0))
    {
        uint i_9 = (&kernelContext_73)->lists_0[p_12];
        float inv_1 = (float4(*((&kernelContext_73)->wide_0+st_0(_S313.row_4, 1U))) ).y;
        float3 _S315 = va_0(i_9, 8U, &kernelContext_73);
        float3 _S316 = float3(inv_1) ;
        float3 ga_0 = _S315 * _S316;
        float3 _S317 = vl_0(i_9, 8U, &kernelContext_73);
        float3 gn_0 = _S317 * _S316;
        set_v_0(i_9, 8U, ga_0, gn_0, &kernelContext_73);
        float3 _S318 = va_0(i_9, 0U, &kernelContext_73);
        float _S319 = dot(_S318, ga_0);
        float3 _S320 = vl_0(i_9, 0U, &kernelContext_73);
        gl_0 = _S319 + dot(_S320, gn_0);
    }
    else
    {
        gl_0 = 0.0f;
    }
    write_partial_0(tid_24, _S313.row_4, group_19.x, 0U, float4(gl_0, 0.0f, 0.0f, 0.0f), &kernelContext_73);
    return;
}

[[kernel]] void w_gamma(uint3 group_20 [[threadgroup_position_in_grid]], uint3 thread_15 [[thread_position_in_threadgroup]], atomic<uint> device* meta_29 [[buffer(12)]], uint device* topo_args_29 [[buffer(13)]], packed_uint4 device* islands_29 [[buffer(1)]], BlastParams_0 constant* params_29 [[buffer(0)]], packed_float4 device* wide_29 [[buffer(11)]], uint device* lists_29 [[buffer(9)]], BlastNode_natural_0 device* nodes_29 [[buffer(2)]], packed_float4 device* accel_29 [[buffer(5)]], packed_float4 device* scratch_29 [[buffer(7)]], uint device* refs_29 [[buffer(3)]], BlastBond_natural_0 device* bonds_29 [[buffer(4)]], packed_float4 device* stiffness_29 [[buffer(10)]], packed_float4 device* x_29 [[buffer(6)]], packed_uint4 device* report_29 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_74;
    (&kernelContext_74)->meta_0 = meta_29;
    (&kernelContext_74)->topo_args_0 = topo_args_29;
    (&kernelContext_74)->islands_0 = islands_29;
    (&kernelContext_74)->params_0 = params_29;
    (&kernelContext_74)->wide_0 = wide_29;
    (&kernelContext_74)->lists_0 = lists_29;
    (&kernelContext_74)->nodes_0 = nodes_29;
    (&kernelContext_74)->accel_0 = accel_29;
    (&kernelContext_74)->scratch_0 = scratch_29;
    (&kernelContext_74)->refs_0 = refs_29;
    (&kernelContext_74)->bonds_0 = bonds_29;
    (&kernelContext_74)->stiffness_0 = stiffness_29;
    (&kernelContext_74)->x_0 = x_29;
    (&kernelContext_74)->report_0 = report_29;
    threadgroup array<float2, int(32)> g_blast_part_29;
    (&kernelContext_74)->g_blast_part_0 = &g_blast_part_29;
    threadgroup array<float, int(1024)> g_wide_max_29;
    (&kernelContext_74)->g_wide_max_0 = &g_wide_max_29;
    threadgroup float g_wide_max_out_29;
    (&kernelContext_74)->g_wide_max_out_0 = &g_wide_max_out_29;
    uint tid_25 = thread_15.x;
    uint row_18 = group_20.x;
    uint _S321 = flags_of_0(row_18, &kernelContext_74);
    uint f_15 = group_uniform_0(_S321, tid_25);
    if((f_15 & 1U) == 0U)
    {
        return;
    }
    uint4 _S322 = row_range_0(tid_25, row_18, &kernelContext_74);
    float4 _S323 = row_sum_0(tid_25, row_18, blocks_0(_S322.y - _S322.x), 0U, &kernelContext_74);
    float gamma_2 = _S323.x;
    if(tid_25 != 0U)
    {
        return;
    }
    bool _S324;
    if(!(gamma_2 > 0.0f))
    {
        _S324 = true;
    }
    else
    {
        _S324 = !isfinite(gamma_2);
    }
    if(_S324)
    {
        set_flags_0(row_18, (f_15 & 4294967294U) | 4U, &kernelContext_74);
        return;
    }
    float4 _S325 = float4(*((&kernelContext_74)->wide_0+st_0(row_18, 0U))) ;
    uint _S326 = iteration_of_0(row_18, &kernelContext_74);
    bool carried_0 = (f_15 & 256U) != 0U;
    if(_S326 > 1U)
    {
        _S324 = true;
    }
    else
    {
        if(((&kernelContext_74)->params_0->first_polynomial_0) != 0U)
        {
            _S324 = _S326 > 0U;
        }
        else
        {
            _S324 = false;
        }
    }
    if(_S324)
    {
        _S324 = true;
    }
    else
    {
        _S324 = carried_0;
    }
    if(_S324)
    {
        _S324 = (_S325.z) > 0.0f;
    }
    else
    {
        _S324 = false;
    }
    float beta_0;
    if(_S324)
    {
        beta_0 = gamma_2 / _S325.z;
    }
    else
    {
        beta_0 = 0.0f;
    }
    if(carried_0)
    {
        _S324 = _S326 == 0U;
    }
    else
    {
        _S324 = false;
    }
    if(_S324)
    {
        _S324 = !(beta_0 <= 16.0f);
    }
    else
    {
        _S324 = false;
    }
    if(_S324)
    {
        beta_0 = 0.0f;
    }
    uint _S327 = st_0(row_18, 1U);
    thread float4 s1_2 = float4(*((&kernelContext_74)->wide_0+_S327)) ;
    s1_2.x = gamma_2;
    s1_2.z = beta_0;
    *((&kernelContext_74)->wide_0+_S327) = packed_float4(s1_2) ;
    return;
}

[[kernel]] void w_dir(uint3 group_21 [[threadgroup_position_in_grid]], uint3 thread_16 [[thread_position_in_threadgroup]], atomic<uint> device* meta_30 [[buffer(12)]], uint device* topo_args_30 [[buffer(13)]], packed_uint4 device* islands_30 [[buffer(1)]], BlastParams_0 constant* params_30 [[buffer(0)]], packed_float4 device* wide_30 [[buffer(11)]], uint device* lists_30 [[buffer(9)]], BlastNode_natural_0 device* nodes_30 [[buffer(2)]], packed_float4 device* accel_30 [[buffer(5)]], packed_float4 device* scratch_30 [[buffer(7)]], uint device* refs_30 [[buffer(3)]], BlastBond_natural_0 device* bonds_30 [[buffer(4)]], packed_float4 device* stiffness_30 [[buffer(10)]], packed_float4 device* x_30 [[buffer(6)]], packed_uint4 device* report_30 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_75;
    (&kernelContext_75)->meta_0 = meta_30;
    (&kernelContext_75)->topo_args_0 = topo_args_30;
    (&kernelContext_75)->islands_0 = islands_30;
    (&kernelContext_75)->params_0 = params_30;
    (&kernelContext_75)->wide_0 = wide_30;
    (&kernelContext_75)->lists_0 = lists_30;
    (&kernelContext_75)->nodes_0 = nodes_30;
    (&kernelContext_75)->accel_0 = accel_30;
    (&kernelContext_75)->scratch_0 = scratch_30;
    (&kernelContext_75)->refs_0 = refs_30;
    (&kernelContext_75)->bonds_0 = bonds_30;
    (&kernelContext_75)->stiffness_0 = stiffness_30;
    (&kernelContext_75)->x_0 = x_30;
    (&kernelContext_75)->report_0 = report_30;
    threadgroup array<float2, int(32)> g_blast_part_30;
    (&kernelContext_75)->g_blast_part_0 = &g_blast_part_30;
    threadgroup array<float, int(1024)> g_wide_max_30;
    (&kernelContext_75)->g_wide_max_0 = &g_wide_max_30;
    threadgroup float g_wide_max_out_30;
    (&kernelContext_75)->g_wide_max_out_0 = &g_wide_max_out_30;
    uint tid_26 = thread_16.x;
    Block_0 _S328 = node_block_0(group_21, tid_26, &kernelContext_75);
    bool _S329;
    if((_S328.begin_0) >= (_S328.end_0))
    {
        _S329 = true;
    }
    else
    {
        _S329 = !runs_0(_S328.flags_0, false);
    }
    if(_S329)
    {
        return;
    }
    uint p_13 = _S328.begin_0 + tid_26;
    if(p_13 >= (_S328.end_0))
    {
        return;
    }
    uint i_10 = (&kernelContext_75)->lists_0[p_13];
    float beta_1 = (float4(*((&kernelContext_75)->wide_0+st_0(_S328.row_4, 1U))) ).z;
    if(beta_1 == 0.0f)
    {
        float3 _S330 = va_0(i_10, 8U, &kernelContext_75);
        float3 _S331 = vl_0(i_10, 8U, &kernelContext_75);
        set_v_0(i_10, 2U, _S330, _S331, &kernelContext_75);
    }
    else
    {
        float3 _S332 = va_0(i_10, 8U, &kernelContext_75);
        float3 _S333 = va_0(i_10, 2U, &kernelContext_75);
        float3 _S334 = float3(beta_1) ;
        float3 _S335 = _S332 + _S333 * _S334;
        float3 _S336 = vl_0(i_10, 8U, &kernelContext_75);
        float3 _S337 = vl_0(i_10, 2U, &kernelContext_75);
        set_v_0(i_10, 2U, _S335, _S336 + _S337 * _S334, &kernelContext_75);
    }
    return;
}

[[kernel]] void w_bond_p(uint3 group_22 [[threadgroup_position_in_grid]], uint3 thread_17 [[thread_position_in_threadgroup]], atomic<uint> device* meta_31 [[buffer(12)]], uint device* topo_args_31 [[buffer(13)]], packed_uint4 device* islands_31 [[buffer(1)]], BlastParams_0 constant* params_31 [[buffer(0)]], packed_float4 device* wide_31 [[buffer(11)]], uint device* lists_31 [[buffer(9)]], BlastNode_natural_0 device* nodes_31 [[buffer(2)]], packed_float4 device* accel_31 [[buffer(5)]], packed_float4 device* scratch_31 [[buffer(7)]], uint device* refs_31 [[buffer(3)]], BlastBond_natural_0 device* bonds_31 [[buffer(4)]], packed_float4 device* stiffness_31 [[buffer(10)]], packed_float4 device* x_31 [[buffer(6)]], packed_uint4 device* report_31 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_76;
    (&kernelContext_76)->meta_0 = meta_31;
    (&kernelContext_76)->topo_args_0 = topo_args_31;
    (&kernelContext_76)->islands_0 = islands_31;
    (&kernelContext_76)->params_0 = params_31;
    (&kernelContext_76)->wide_0 = wide_31;
    (&kernelContext_76)->lists_0 = lists_31;
    (&kernelContext_76)->nodes_0 = nodes_31;
    (&kernelContext_76)->accel_0 = accel_31;
    (&kernelContext_76)->scratch_0 = scratch_31;
    (&kernelContext_76)->refs_0 = refs_31;
    (&kernelContext_76)->bonds_0 = bonds_31;
    (&kernelContext_76)->stiffness_0 = stiffness_31;
    (&kernelContext_76)->x_0 = x_31;
    (&kernelContext_76)->report_0 = report_31;
    threadgroup array<float2, int(32)> g_blast_part_31;
    (&kernelContext_76)->g_blast_part_0 = &g_blast_part_31;
    threadgroup array<float, int(1024)> g_wide_max_31;
    (&kernelContext_76)->g_wide_max_0 = &g_wide_max_31;
    threadgroup float g_wide_max_out_31;
    (&kernelContext_76)->g_wide_max_out_0 = &g_wide_max_out_31;
    uint tid_27 = thread_17.x;
    Block_0 _S338 = bond_block_0(group_22, tid_27, &kernelContext_76);
    bool _S339;
    if((_S338.begin_0) >= (_S338.end_0))
    {
        _S339 = true;
    }
    else
    {
        _S339 = !runs_0(_S338.flags_0, false);
    }
    if(_S339)
    {
        return;
    }
    uint p_14 = _S338.begin_0 + tid_27;
    float z_5;
    if(p_14 < (_S338.end_0))
    {
        uint j_16 = (&kernelContext_76)->lists_0[p_14];
        thread BlastBond_natural_0 _S340 = *((&kernelContext_76)->bonds_0+j_16);
        float3 _S341 = float3(0.0f) ;
        thread float3 ta_5 = _S341;
        thread float3 tl_5 = _S341;
        if(((&_S340)->health_0) > 0.0f)
        {
            bond_term_1(&_S340, j_16, 2U, &ta_5, &tl_5, &kernelContext_76);
        }
        float _S342 = norm2_0(ta_5, tl_5);
        uint _S343 = bs_0(j_16, &kernelContext_76);
        *((&kernelContext_76)->scratch_0+_S343) = packed_float4(float4(ta_5, 0.0f)) ;
        *((&kernelContext_76)->scratch_0+(_S343 + 1U)) = packed_float4(float4(tl_5, 0.0f)) ;
        z_5 = _S342;
    }
    else
    {
        z_5 = 0.0f;
    }
    write_partial_0(tid_27, _S338.row_4, group_22.x, 1U, float4(z_5, 0.0f, 0.0f, 0.0f), &kernelContext_76);
    return;
}

[[kernel]] void w_node_q(uint3 group_23 [[threadgroup_position_in_grid]], uint3 thread_18 [[thread_position_in_threadgroup]], atomic<uint> device* meta_32 [[buffer(12)]], uint device* topo_args_32 [[buffer(13)]], packed_uint4 device* islands_32 [[buffer(1)]], BlastParams_0 constant* params_32 [[buffer(0)]], packed_float4 device* wide_32 [[buffer(11)]], uint device* lists_32 [[buffer(9)]], BlastNode_natural_0 device* nodes_32 [[buffer(2)]], packed_float4 device* accel_32 [[buffer(5)]], packed_float4 device* scratch_32 [[buffer(7)]], uint device* refs_32 [[buffer(3)]], BlastBond_natural_0 device* bonds_32 [[buffer(4)]], packed_float4 device* stiffness_32 [[buffer(10)]], packed_float4 device* x_32 [[buffer(6)]], packed_uint4 device* report_32 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_77;
    (&kernelContext_77)->meta_0 = meta_32;
    (&kernelContext_77)->topo_args_0 = topo_args_32;
    (&kernelContext_77)->islands_0 = islands_32;
    (&kernelContext_77)->params_0 = params_32;
    (&kernelContext_77)->wide_0 = wide_32;
    (&kernelContext_77)->lists_0 = lists_32;
    (&kernelContext_77)->nodes_0 = nodes_32;
    (&kernelContext_77)->accel_0 = accel_32;
    (&kernelContext_77)->scratch_0 = scratch_32;
    (&kernelContext_77)->refs_0 = refs_32;
    (&kernelContext_77)->bonds_0 = bonds_32;
    (&kernelContext_77)->stiffness_0 = stiffness_32;
    (&kernelContext_77)->x_0 = x_32;
    (&kernelContext_77)->report_0 = report_32;
    threadgroup array<float2, int(32)> g_blast_part_32;
    (&kernelContext_77)->g_blast_part_0 = &g_blast_part_32;
    threadgroup array<float, int(1024)> g_wide_max_32;
    (&kernelContext_77)->g_wide_max_0 = &g_wide_max_32;
    threadgroup float g_wide_max_out_32;
    (&kernelContext_77)->g_wide_max_out_0 = &g_wide_max_out_32;
    uint tid_28 = thread_18.x;
    Block_0 _S344 = node_block_0(group_23, tid_28, &kernelContext_77);
    bool _S345;
    if((_S344.begin_0) >= (_S344.end_0))
    {
        _S345 = true;
    }
    else
    {
        _S345 = !runs_0(_S344.flags_0, false);
    }
    if(_S345)
    {
        return;
    }
    uint p_15 = _S344.begin_0 + tid_28;
    if(p_15 >= (_S344.end_0))
    {
        return;
    }
    uint i_11 = (&kernelContext_77)->lists_0[p_15];
    thread float3 a_9;
    thread float3 l_9;
    gather_0(i_11, true, false, &a_9, &l_9, &kernelContext_77);
    set_v_0(i_11, 4U, a_9, l_9, &kernelContext_77);
    return;
}

[[kernel]] void w_den(uint3 group_24 [[threadgroup_position_in_grid]], uint3 thread_19 [[thread_position_in_threadgroup]], atomic<uint> device* meta_33 [[buffer(12)]], uint device* topo_args_33 [[buffer(13)]], packed_uint4 device* islands_33 [[buffer(1)]], BlastParams_0 constant* params_33 [[buffer(0)]], packed_float4 device* wide_33 [[buffer(11)]], uint device* lists_33 [[buffer(9)]], BlastNode_natural_0 device* nodes_33 [[buffer(2)]], packed_float4 device* accel_33 [[buffer(5)]], packed_float4 device* scratch_33 [[buffer(7)]], uint device* refs_33 [[buffer(3)]], BlastBond_natural_0 device* bonds_33 [[buffer(4)]], packed_float4 device* stiffness_33 [[buffer(10)]], packed_float4 device* x_33 [[buffer(6)]], packed_uint4 device* report_33 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_78;
    (&kernelContext_78)->meta_0 = meta_33;
    (&kernelContext_78)->topo_args_0 = topo_args_33;
    (&kernelContext_78)->islands_0 = islands_33;
    (&kernelContext_78)->params_0 = params_33;
    (&kernelContext_78)->wide_0 = wide_33;
    (&kernelContext_78)->lists_0 = lists_33;
    (&kernelContext_78)->nodes_0 = nodes_33;
    (&kernelContext_78)->accel_0 = accel_33;
    (&kernelContext_78)->scratch_0 = scratch_33;
    (&kernelContext_78)->refs_0 = refs_33;
    (&kernelContext_78)->bonds_0 = bonds_33;
    (&kernelContext_78)->stiffness_0 = stiffness_33;
    (&kernelContext_78)->x_0 = x_33;
    (&kernelContext_78)->report_0 = report_33;
    threadgroup array<float2, int(32)> g_blast_part_33;
    (&kernelContext_78)->g_blast_part_0 = &g_blast_part_33;
    threadgroup array<float, int(1024)> g_wide_max_33;
    (&kernelContext_78)->g_wide_max_0 = &g_wide_max_33;
    threadgroup float g_wide_max_out_33;
    (&kernelContext_78)->g_wide_max_out_0 = &g_wide_max_out_33;
    uint tid_29 = thread_19.x;
    uint row_19 = group_24.x;
    uint _S346 = flags_of_0(row_19, &kernelContext_78);
    uint f_16 = group_uniform_0(_S346, tid_29);
    if((f_16 & 1U) == 0U)
    {
        return;
    }
    uint4 _S347 = row_range_0(tid_29, row_19, &kernelContext_78);
    float4 _S348 = row_sum_0(tid_29, row_19, blocks_0(_S347.w - _S347.z), 1U, &kernelContext_78);
    float den_0 = _S348.x;
    if(tid_29 != 0U)
    {
        return;
    }
    uint _S349 = st_0(row_19, 0U);
    thread float4 s0_3 = float4(*((&kernelContext_78)->wide_0+_S349)) ;
    uint _S350 = st_0(row_19, 1U);
    float4 _S351 = float4(*((&kernelContext_78)->wide_0+_S350)) ;
    thread float4 s1_3 = _S351;
    uint _S352 = st_0(row_19, 2U);
    thread float4 cnt_0 = float4(*((&kernelContext_78)->wide_0+_S352)) ;
    float gamma_3 = _S351.x;
    bool _S353;
    if(((&kernelContext_78)->params_0->force_tolerance_0) > 0.0f)
    {
        if(den_0 > 0.0f)
        {
            _S353 = gamma_3 > 0.0f;
        }
        else
        {
            _S353 = false;
        }
        float _S354;
        if(_S353)
        {
            _S354 = gamma_3 / sqrt(den_0);
        }
        else
        {
            _S354 = float_bits_0(2139095040U);
        }
        s1_3.w = _S354;
        if(isfinite(s1_3.w))
        {
            cnt_0.y = cnt_0.y + s1_3.w;
        }
        if((as_type<uint>((cnt_0.z))) > 0U)
        {
            _S353 = true;
        }
        else
        {
            _S353 = ((&kernelContext_78)->params_0->first_polynomial_0) != 0U;
        }
        if(!_S353)
        {
            s1_3.w = float_bits_0(2139095040U);
        }
    }
    s0_3.z = gamma_3;
    cnt_0.z = (as_type<float>(((as_type<uint>((cnt_0.z))) + 1U)));
    if(!(den_0 > 0.0f))
    {
        _S353 = true;
    }
    else
    {
        _S353 = !isfinite(den_0);
    }
    uint f_17;
    if(_S353)
    {
        uint f_18 = f_16 & 4294967294U;
        s1_3.y = 0.0f;
        f_17 = f_18;
    }
    else
    {
        s1_3.y = gamma_3 / den_0;
        f_17 = f_16;
    }
    *((&kernelContext_78)->wide_0+_S349) = packed_float4(s0_3) ;
    *((&kernelContext_78)->wide_0+_S350) = packed_float4(s1_3) ;
    cnt_0.w = (as_type<float>((f_17)));
    *((&kernelContext_78)->wide_0+_S352) = packed_float4(cnt_0) ;
    return;
}

[[kernel]] void w_update(uint3 group_25 [[threadgroup_position_in_grid]], uint3 thread_20 [[thread_position_in_threadgroup]], atomic<uint> device* meta_34 [[buffer(12)]], uint device* topo_args_34 [[buffer(13)]], packed_uint4 device* islands_34 [[buffer(1)]], BlastParams_0 constant* params_34 [[buffer(0)]], packed_float4 device* wide_34 [[buffer(11)]], uint device* lists_34 [[buffer(9)]], BlastNode_natural_0 device* nodes_34 [[buffer(2)]], packed_float4 device* accel_34 [[buffer(5)]], packed_float4 device* scratch_34 [[buffer(7)]], uint device* refs_34 [[buffer(3)]], BlastBond_natural_0 device* bonds_34 [[buffer(4)]], packed_float4 device* stiffness_34 [[buffer(10)]], packed_float4 device* x_34 [[buffer(6)]], packed_uint4 device* report_34 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_79;
    (&kernelContext_79)->meta_0 = meta_34;
    (&kernelContext_79)->topo_args_0 = topo_args_34;
    (&kernelContext_79)->islands_0 = islands_34;
    (&kernelContext_79)->params_0 = params_34;
    (&kernelContext_79)->wide_0 = wide_34;
    (&kernelContext_79)->lists_0 = lists_34;
    (&kernelContext_79)->nodes_0 = nodes_34;
    (&kernelContext_79)->accel_0 = accel_34;
    (&kernelContext_79)->scratch_0 = scratch_34;
    (&kernelContext_79)->refs_0 = refs_34;
    (&kernelContext_79)->bonds_0 = bonds_34;
    (&kernelContext_79)->stiffness_0 = stiffness_34;
    (&kernelContext_79)->x_0 = x_34;
    (&kernelContext_79)->report_0 = report_34;
    threadgroup array<float2, int(32)> g_blast_part_34;
    (&kernelContext_79)->g_blast_part_0 = &g_blast_part_34;
    threadgroup array<float, int(1024)> g_wide_max_34;
    (&kernelContext_79)->g_wide_max_0 = &g_wide_max_34;
    threadgroup float g_wide_max_out_34;
    (&kernelContext_79)->g_wide_max_out_0 = &g_wide_max_out_34;
    uint tid_30 = thread_20.x;
    Block_0 _S355 = node_block_0(group_25, tid_30, &kernelContext_79);
    bool _S356;
    if((_S355.begin_0) >= (_S355.end_0))
    {
        _S356 = true;
    }
    else
    {
        _S356 = !runs_0(_S355.flags_0, false);
    }
    if(_S356)
    {
        return;
    }
    uint p_16 = _S355.begin_0 + tid_30;
    if(p_16 >= (_S355.end_0))
    {
        return;
    }
    uint i_12 = (&kernelContext_79)->lists_0[p_16];
    float alpha_0 = (float4(*((&kernelContext_79)->wide_0+st_0(_S355.row_4, 1U))) ).y;
    float3 _S357 = va_0(i_12, 6U, &kernelContext_79);
    float3 _S358 = va_0(i_12, 2U, &kernelContext_79);
    float3 _S359 = float3(alpha_0) ;
    float3 _S360 = _S357 + _S358 * _S359;
    float3 _S361 = vl_0(i_12, 6U, &kernelContext_79);
    float3 _S362 = vl_0(i_12, 2U, &kernelContext_79);
    set_v_0(i_12, 6U, _S360, _S361 + _S362 * _S359, &kernelContext_79);
    float3 _S363 = va_0(i_12, 0U, &kernelContext_79);
    float3 _S364 = va_0(i_12, 4U, &kernelContext_79);
    float3 _S365 = _S363 - _S364 * _S359;
    float3 _S366 = vl_0(i_12, 0U, &kernelContext_79);
    float3 _S367 = vl_0(i_12, 4U, &kernelContext_79);
    set_v_0(i_12, 0U, _S365, _S366 - _S367 * _S359, &kernelContext_79);
    return;
}

[[kernel]] void w_final(uint3 group_26 [[threadgroup_position_in_grid]], uint3 thread_21 [[thread_position_in_threadgroup]], atomic<uint> device* meta_35 [[buffer(12)]], uint device* topo_args_35 [[buffer(13)]], packed_uint4 device* islands_35 [[buffer(1)]], BlastParams_0 constant* params_35 [[buffer(0)]], packed_float4 device* wide_35 [[buffer(11)]], uint device* lists_35 [[buffer(9)]], BlastNode_natural_0 device* nodes_35 [[buffer(2)]], packed_float4 device* accel_35 [[buffer(5)]], packed_float4 device* scratch_35 [[buffer(7)]], uint device* refs_35 [[buffer(3)]], BlastBond_natural_0 device* bonds_35 [[buffer(4)]], packed_float4 device* stiffness_35 [[buffer(10)]], packed_float4 device* x_35 [[buffer(6)]], packed_uint4 device* report_35 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_80;
    (&kernelContext_80)->meta_0 = meta_35;
    (&kernelContext_80)->topo_args_0 = topo_args_35;
    (&kernelContext_80)->islands_0 = islands_35;
    (&kernelContext_80)->params_0 = params_35;
    (&kernelContext_80)->wide_0 = wide_35;
    (&kernelContext_80)->lists_0 = lists_35;
    (&kernelContext_80)->nodes_0 = nodes_35;
    (&kernelContext_80)->accel_0 = accel_35;
    (&kernelContext_80)->scratch_0 = scratch_35;
    (&kernelContext_80)->refs_0 = refs_35;
    (&kernelContext_80)->bonds_0 = bonds_35;
    (&kernelContext_80)->stiffness_0 = stiffness_35;
    (&kernelContext_80)->x_0 = x_35;
    (&kernelContext_80)->report_0 = report_35;
    threadgroup array<float2, int(32)> g_blast_part_35;
    (&kernelContext_80)->g_blast_part_0 = &g_blast_part_35;
    threadgroup array<float, int(1024)> g_wide_max_35;
    (&kernelContext_80)->g_wide_max_0 = &g_wide_max_35;
    threadgroup float g_wide_max_out_35;
    (&kernelContext_80)->g_wide_max_out_0 = &g_wide_max_out_35;
    uint tid_31 = thread_21.x;
    Block_0 _S368 = bond_block_0(group_26, tid_31, &kernelContext_80);
    bool _S369;
    if((_S368.begin_0) >= (_S368.end_0))
    {
        _S369 = true;
    }
    else
    {
        _S369 = ((_S368.flags_0) & 128U) != 0U;
    }
    if(_S369)
    {
        return;
    }
    uint p_17 = _S368.begin_0 + tid_31;
    if(p_17 >= (_S368.end_0))
    {
        return;
    }
    uint j_17 = (&kernelContext_80)->lists_0[p_17];
    thread BlastBond_natural_0 _S370 = *((&kernelContext_80)->bonds_0+j_17);
    if(!(((&_S370)->health_0) > 0.0f))
    {
        _S369 = true;
    }
    else
    {
        _S369 = ((_S368.flags_0) & 16U) != 0U;
    }
    if(_S369)
    {
        uint _S371 = j_17 * 2U;
        packed_float4 _S372 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_80)->x_0+_S371) = _S372;
        *((&kernelContext_80)->x_0+(_S371 + 1U)) = _S372;
        return;
    }
    thread float3 ta_6;
    thread float3 tl_6;
    bond_term_1(&_S370, j_17, 6U, &ta_6, &tl_6, &kernelContext_80);
    uint _S373 = j_17 * 2U;
    *((&kernelContext_80)->x_0+_S373) = packed_float4(float4((float4(*((&kernelContext_80)->x_0+_S373)) ).xyz + ta_6, 0.0f)) ;
    uint _S374 = _S373 + 1U;
    *((&kernelContext_80)->x_0+_S374) = packed_float4(float4((float4(*((&kernelContext_80)->x_0+_S374)) ).xyz + tl_6, 0.0f)) ;
    return;
}

uint carry_kept_iterations_0(uint row_20, KernelContext_0 thread* kernelContext_81)
{
    return (as_type<uint>(((float4(*(kernelContext_81->wide_0+st_0(row_20, 3U))) ).x)));
}

[[kernel]] void w_report(uint3 group_27 [[threadgroup_position_in_grid]], atomic<uint> device* meta_36 [[buffer(12)]], uint device* topo_args_36 [[buffer(13)]], packed_uint4 device* islands_36 [[buffer(1)]], BlastParams_0 constant* params_36 [[buffer(0)]], packed_float4 device* wide_36 [[buffer(11)]], uint device* lists_36 [[buffer(9)]], BlastNode_natural_0 device* nodes_36 [[buffer(2)]], packed_float4 device* accel_36 [[buffer(5)]], packed_float4 device* scratch_36 [[buffer(7)]], uint device* refs_36 [[buffer(3)]], BlastBond_natural_0 device* bonds_36 [[buffer(4)]], packed_float4 device* stiffness_36 [[buffer(10)]], packed_float4 device* x_36 [[buffer(6)]], packed_uint4 device* report_36 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_82;
    (&kernelContext_82)->meta_0 = meta_36;
    (&kernelContext_82)->topo_args_0 = topo_args_36;
    (&kernelContext_82)->islands_0 = islands_36;
    (&kernelContext_82)->params_0 = params_36;
    (&kernelContext_82)->wide_0 = wide_36;
    (&kernelContext_82)->lists_0 = lists_36;
    (&kernelContext_82)->nodes_0 = nodes_36;
    (&kernelContext_82)->accel_0 = accel_36;
    (&kernelContext_82)->scratch_0 = scratch_36;
    (&kernelContext_82)->refs_0 = refs_36;
    (&kernelContext_82)->bonds_0 = bonds_36;
    (&kernelContext_82)->stiffness_0 = stiffness_36;
    (&kernelContext_82)->x_0 = x_36;
    (&kernelContext_82)->report_0 = report_36;
    threadgroup array<float2, int(32)> g_blast_part_36;
    (&kernelContext_82)->g_blast_part_0 = &g_blast_part_36;
    threadgroup array<float, int(1024)> g_wide_max_36;
    (&kernelContext_82)->g_wide_max_0 = &g_wide_max_36;
    threadgroup float g_wide_max_out_36;
    (&kernelContext_82)->g_wide_max_out_0 = &g_wide_max_out_36;
    uint row_21 = group_27.x;
    uint _S375 = row_island_0(row_21, &kernelContext_82);
    uint _S376 = flags_of_0(row_21, &kernelContext_82);
    uint4 _S377 = uint4(*((&kernelContext_82)->report_0+_S375)) ;
    if((_S376 & 128U) != 0U)
    {
        *((&kernelContext_82)->report_0+_S375) = packed_uint4(uint4(0U, 7U, _S377.z, _S377.w)) ;
        return;
    }
    bool ok_0;
    if((_S376 & 2U) != 0U)
    {
        ok_0 = (_S376 & 4U) == 0U;
    }
    else
    {
        ok_0 = false;
    }
    uint _S378 = iteration_of_0(row_21, &kernelContext_82);
    uint _S379;
    if(ok_0)
    {
        _S379 = 1U;
    }
    else
    {
        _S379 = 0U;
    }
    if(ok_0)
    {
        ok_0 = ((&kernelContext_82)->params_0->warm_0) != 0U;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        ok_0 = _S378 == 0U;
    }
    else
    {
        ok_0 = false;
    }
    uint _S380;
    if(ok_0)
    {
        _S380 = 4U;
    }
    else
    {
        _S380 = 0U;
    }
    float4 _S381 = float4(*((&kernelContext_82)->wide_0+st_0(row_21, 0U))) ;
    *((&kernelContext_82)->report_0+_S375) = packed_uint4(uint4(_S378, _S379 | _S380, (as_type<uint>((_S381.y))), (as_type<uint>((_S381.x))))) ;
    uint4 _S382 = uint4(*((&kernelContext_82)->islands_0+_S375)) ;
    if(((&kernelContext_82)->params_0->carry_0) != 0U)
    {
        ok_0 = (_S376 & 6U) == 0U;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        uint _S383 = _S382.y - _S382.x;
        if((_S376 & 256U) != 0U)
        {
            uint _S384 = carry_kept_iterations_0(row_21, &kernelContext_82);
            _S379 = _S384;
        }
        else
        {
            _S379 = 0U;
        }
        set_carry_0(_S375, _S383, _S379 + _S378, _S381.z, &kernelContext_82);
    }
    return;
}

