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

void set_v_0(uint n_2, uint v_1, float3 a_1, float3 l_0, KernelContext_0 thread* kernelContext_7)
{
    uint _S16 = ns_0(n_2, v_1, kernelContext_7);
    *(kernelContext_7->scratch_0+_S16) = packed_float4(float4(a_1, 0.0f)) ;
    *(kernelContext_7->scratch_0+(_S16 + 1U)) = packed_float4(float4(l_0, 0.0f)) ;
    return;
}

float3 stiff_0(uint j_0, uint which_0, float3 v_2, KernelContext_0 thread* kernelContext_8)
{
    uint _S17 = 6U * which_0;
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
        uint _S18 = _S17 + k_1;
        m_0[k_1] = (float4(*(kernelContext_8->stiffness_0+(j_0 * 6U + _S18 / 4U))) )[_S18 % 4U];
        k_1 = k_1 + 1U;
    }
    float _S19 = v_2.x;
    float _S20 = v_2.y;
    float _S21 = v_2.z;
    return float3(m_0[int(0)] * _S19 + m_0[int(3)] * _S20 + m_0[int(4)] * _S21, m_0[int(3)] * _S19 + m_0[int(1)] * _S20 + m_0[int(5)] * _S21, m_0[int(4)] * _S19 + m_0[int(5)] * _S20 + m_0[int(2)] * _S21);
}

float3 stiff_1(uint j_1, uint which_1, float3 v_3, KernelContext_0 thread* kernelContext_9)
{
    uint _S22 = 6U * which_1;
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
        uint _S23 = _S22 + k_2;
        m_1[k_2] = (float4(*(kernelContext_9->stiffness_0+(j_1 * 6U + _S23 / 4U))) )[_S23 % 4U];
        k_2 = k_2 + 1U;
    }
    float _S24 = v_3.x;
    float _S25 = v_3.y;
    float _S26 = v_3.z;
    return float3(m_1[int(0)] * _S24 + m_1[int(3)] * _S25 + m_1[int(4)] * _S26, m_1[int(3)] * _S24 + m_1[int(1)] * _S25 + m_1[int(5)] * _S26, m_1[int(4)] * _S24 + m_1[int(5)] * _S25 + m_1[int(2)] * _S26);
}

matrix<float,int(3),int(3)>  sym3_0(uint j_2, uint which_2, KernelContext_0 thread* kernelContext_10)
{
    float3 _S27 = stiff_1(j_2, which_2, float3(1.0f, 0.0f, 0.0f), kernelContext_10);
    float3 _S28 = stiff_1(j_2, which_2, float3(0.0f, 1.0f, 0.0f), kernelContext_10);
    float3 _S29 = stiff_1(j_2, which_2, float3(0.0f, 0.0f, 1.0f), kernelContext_10);
    return matrix<float,int(3),int(3)> (_S27.x, _S28.x, _S29.x, _S27.y, _S28.y, _S29.y, _S27.z, _S28.z, _S29.z);
}

matrix<float,int(3),int(3)>  skew_0(float3 o_0)
{
    float _S30 = o_0.z;
    float _S31 = o_0.y;
    float _S32 = o_0.x;
    return matrix<float,int(3),int(3)> (0.0f, - _S30, _S31, _S30, 0.0f, - _S32, - _S31, _S32, 0.0f);
}

bool build_full_block_0(uint node_0, KernelContext_0 thread* kernelContext_11)
{
    uint c_0;
    uint k_3;
    float s_1;
    BlastNode_natural_0 n_3 = kernelContext_11->nodes_0[node_0];
    matrix<float,int(3),int(3)>  aa_0 = matrix<float,int(3),int(3)> (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    uint r_0 = n_3.ref_begin_0;
    matrix<float,int(3),int(3)>  aa_1 = aa_0;
    matrix<float,int(3),int(3)>  al_0 = aa_0;
    matrix<float,int(3),int(3)>  ll_0 = aa_0;
    for(;;)
    {
        if(r_0 < (n_3.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_0 = kernelContext_11->refs_0[r_0];
        uint j_3 = ref_0 & 2147483647U;
        BlastBond_natural_0 device* _S33 = kernelContext_11->bonds_0+j_3;
        BlastBond_natural_0 b_2 = *_S33;
        if(!(((*_S33).health_0) > 0.0f))
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
        float _S34 = b_2.s_0;
        float s2_0 = _S34 * _S34;
        matrix<float,int(3),int(3)>  _S35 = sym3_0(j_3, 3U, kernelContext_11);
        matrix<float,int(3),int(3)>  k_4 = skew_0(o_1);
        matrix<float,int(3),int(3)>  _S36 = sym3_0(j_3, 1U, kernelContext_11);
        matrix<float,int(3),int(3)>  _S37 = (((_S35) * (k_4)));
        matrix<float,int(3),int(3)>  _S38 = matrix<float,int(3),int(3)> (s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0);
        matrix<float,int(3),int(3)>  _S39 = al_0 + _slang_matrixCompMult(matrix<float,int(3),int(3)> (- s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0), _S37);
        matrix<float,int(3),int(3)>  _S40 = ll_0 + _slang_matrixCompMult(_S38, _S35);
        aa_1 = aa_1 + _slang_matrixCompMult(_S38, _S36 + (((transpose(k_4)) * (_S37))));
        al_0 = _S39;
        ll_0 = _S40;
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
            m_2[r_0][c_0] = aa_1[r_0][c_0] * n_3.w_ang_0 * n_3.w_ang_0;
            uint _S41 = c_0 + 3U;
            float _S42 = al_0[r_0][c_0] * n_3.w_ang_0 * n_3.w_lin_0;
            m_2[r_0][_S41] = _S42;
            m_2[_S41][r_0] = _S42;
            m_2[r_0 + 3U][_S41] = ll_0[r_0][c_0] * n_3.w_lin_0 * n_3.w_lin_0;
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
        uint _S43 = ns_0(node_0, 18U + q_0, kernelContext_11);
        uint _S44 = 4U * q_0;
        *(kernelContext_11->scratch_0+_S43) = packed_float4(float4(packed_0[_S44], packed_0[_S44 + 1U], packed_0[_S44 + 2U], packed_0[_S44 + 3U])) ;
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
                bool _S45;
                if(!(s_1 > 0.0f))
                {
                    _S45 = true;
                }
                else
                {
                    _S45 = !isfinite(s_1);
                }
                if(_S45)
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
        uint _S46 = ns_0(node_0, 24U + q_0, kernelContext_11);
        uint _S47 = 4U * q_0;
        *(kernelContext_11->scratch_0+_S46) = packed_float4(float4(packed_0[_S47], packed_0[_S47 + 1U], packed_0[_S47 + 2U], packed_0[_S47 + 3U])) ;
        q_0 = q_0 + 1U;
    }
    return true;
}

bool build_block_0(uint node_1, KernelContext_0 thread* kernelContext_12)
{
    if((kernelContext_12->params_0->rotation_0) == 2U)
    {
        bool _S48 = build_full_block_0(node_1, kernelContext_12);
        return _S48;
    }
    BlastNode_natural_0 n_4 = kernelContext_12->nodes_0[node_1];
    float3 _S49 = float3(0.0f) ;
    uint r_2 = n_4.ref_begin_0;
    float3 diag_0 = _S49;
    float3 offd_0 = _S49;
    float3 ks_0 = _S49;
    float cs_0 = 0.0f;
    for(;;)
    {
        if(r_2 < (n_4.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_1 = kernelContext_12->refs_0[r_2];
        uint j_4 = ref_1 & 2147483647U;
        BlastBond_natural_0 device* _S50 = kernelContext_12->bonds_0+j_4;
        BlastBond_natural_0 b_3 = *_S50;
        if(!(((*_S50).health_0) > 0.0f))
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
        float _S51 = b_3.s_0;
        float s2_1 = _S51 * _S51;
        float3 _S52 = float3(1.0f) ;
        float3 wd_0;
        float3 wo_0;
        if((kernelContext_12->params_0->rotation_0) != 0U)
        {
            float3 _S53 = stiff_0(j_4, 1U, float3(1.0f, 0.0f, 0.0f), kernelContext_12);
            float3 _S54 = stiff_0(j_4, 1U, float3(0.0f, 1.0f, 0.0f), kernelContext_12);
            float3 _S55 = stiff_0(j_4, 1U, float3(0.0f, 0.0f, 1.0f), kernelContext_12);
            float3 _S56 = float3(_S54.x, _S55.x, _S55.y);
            wd_0 = float3(_S53.x, _S54.y, _S55.z);
            wo_0 = _S56;
        }
        else
        {
            wd_0 = _S52;
            wo_0 = _S49;
        }
        float3 _S57 = float3(s2_1) ;
        float _S58 = o_2.x;
        float _S59 = o_2.y;
        float _S60 = o_2.z;
        float3 offd_1 = offd_0 + _S57 * (wo_0 - float3(_S58 * _S59, _S58 * _S60, _S59 * _S60));
        float3 ks_1 = ks_0 + _S57 * o_2;
        float cs_1 = cs_0 + s2_1;
        diag_0 = diag_0 + _S57 * (wd_0 + float3(dot(o_2, o_2))  - o_2 * o_2);
        offd_0 = offd_1;
        ks_0 = ks_1;
        cs_0 = cs_1;
        r_2 = r_2 + 1U;
    }
    float _S61 = n_4.w_ang_0;
    float da_0 = _S61 * _S61;
    float c_1 = n_4.w_lin_0 * n_4.w_lin_0 * cs_0;
    float3 k_5 = ks_0 * float3((n_4.w_ang_0 * n_4.w_lin_0)) ;
    float4 a0_0 = float4(diag_0 * float3(da_0) , offd_0.x * da_0);
    float _S62 = offd_0.y * da_0;
    float _S63 = offd_0.z * da_0;
    float4 a1_0 = float4(_S62, _S63, c_1, 0.0f);
    uint _S64 = ns_0(node_1, 18U, kernelContext_12);
    *(kernelContext_12->scratch_0+_S64) = packed_float4(a0_0) ;
    *(kernelContext_12->scratch_0+(_S64 + 1U)) = packed_float4(a1_0) ;
    *(kernelContext_12->scratch_0+(_S64 + 2U)) = packed_float4(float4(k_5, 0.0f)) ;
    if(!(c_1 > 0.0f))
    {
        return false;
    }
    float ic_0 = 1.0f / c_1;
    float kk_0 = dot(k_5, k_5);
    float _S65 = k_5.x;
    float sxx_0 = a0_0.x + (_S65 * _S65 - kk_0) * ic_0;
    float _S66 = k_5.y;
    float syy_0 = a0_0.y + (_S66 * _S66 - kk_0) * ic_0;
    float _S67 = k_5.z;
    float szz_0 = a0_0.z + (_S67 * _S67 - kk_0) * ic_0;
    float sxy_0 = a0_0.w + _S65 * _S66 * ic_0;
    float sxz_0 = _S62 + _S65 * _S67 * ic_0;
    float syz_0 = _S63 + _S66 * _S67 * ic_0;
    float cxx_0 = syy_0 * szz_0 - syz_0 * syz_0;
    float cxy_0 = sxz_0 * syz_0 - sxy_0 * szz_0;
    float cxz_0 = sxy_0 * syz_0 - sxz_0 * syy_0;
    float det_0 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    bool _S68;
    if(!(det_0 > 0.0f))
    {
        _S68 = true;
    }
    else
    {
        _S68 = !isfinite(det_0);
    }
    if(_S68)
    {
        return false;
    }
    float id_3 = 1.0f / det_0;
    float cyy_0 = sxx_0 * szz_0 - sxz_0 * sxz_0;
    float cyz_0 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    float czz_0 = sxx_0 * syy_0 - sxy_0 * sxy_0;
    uint _S69 = ns_0(node_1, 21U, kernelContext_12);
    *(kernelContext_12->scratch_0+_S69) = packed_float4(float4(cxx_0 * id_3, cyy_0 * id_3, czz_0 * id_3, cxy_0 * id_3)) ;
    *(kernelContext_12->scratch_0+(_S69 + 1U)) = packed_float4(float4(cxz_0 * id_3, cyz_0 * id_3, ic_0, 0.0f)) ;
    *(kernelContext_12->scratch_0+(_S69 + 2U)) = packed_float4(float4(k_5 * float3(ic_0) , 0.0f)) ;
    return true;
}

float2 blast_group_sum_0(uint tid_1, float2 v_4, KernelContext_0 thread* kernelContext_13)
{
    float2 w_0 = simd_sum(v_4);
    if((tid_1 % 32U) == 0U)
    {
        (*kernelContext_13->g_blast_part_0)[tid_1 / 32U] = w_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 _S70 = (*kernelContext_13->g_blast_part_0)[int(0)];
    uint k_6 = 1U;
    float2 s_6 = _S70;
    for(;;)
    {
        if(k_6 < 32U)
        {
        }
        else
        {
            break;
        }
        float2 s_7 = s_6 + (*kernelContext_13->g_blast_part_0)[k_6];
        k_6 = k_6 + 1U;
        s_6 = s_7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s_6;
}

uint pt_0(uint row_5, uint block_0, uint k_7, KernelContext_0 thread* kernelContext_14)
{
    return kernelContext_14->params_0->wide_rows_0 * 16U + (row_5 * kernelContext_14->params_0->wide_blocks_0 + block_0) * 6U + k_7;
}

void write_partial_0(uint tid_2, uint row_6, uint block_1, uint k_8, float4 v_5, KernelContext_0 thread* kernelContext_15)
{
    float2 _S71 = blast_group_sum_0(tid_2, v_5.xy, kernelContext_15);
    float2 _S72 = blast_group_sum_0(tid_2, v_5.zw, kernelContext_15);
    if(tid_2 == 0U)
    {
        uint _S73 = pt_0(row_6, block_1, k_8, kernelContext_15);
        *(kernelContext_15->wide_0+_S73) = packed_float4(float4(_S71, _S72)) ;
    }
    return;
}

[[kernel]] void w_init_nodes(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], atomic<uint> device* meta_4 [[buffer(12)]], uint device* topo_args_4 [[buffer(13)]], packed_uint4 device* islands_4 [[buffer(1)]], BlastParams_0 constant* params_4 [[buffer(0)]], packed_float4 device* wide_4 [[buffer(11)]], uint device* lists_4 [[buffer(9)]], BlastNode_natural_0 device* nodes_4 [[buffer(2)]], packed_float4 device* accel_4 [[buffer(5)]], packed_float4 device* scratch_4 [[buffer(7)]], uint device* refs_4 [[buffer(3)]], BlastBond_natural_0 device* bonds_4 [[buffer(4)]], packed_float4 device* stiffness_4 [[buffer(10)]], packed_float4 device* x_4 [[buffer(6)]], packed_uint4 device* report_4 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_16;
    (&kernelContext_16)->meta_0 = meta_4;
    (&kernelContext_16)->topo_args_0 = topo_args_4;
    (&kernelContext_16)->islands_0 = islands_4;
    (&kernelContext_16)->params_0 = params_4;
    (&kernelContext_16)->wide_0 = wide_4;
    (&kernelContext_16)->lists_0 = lists_4;
    (&kernelContext_16)->nodes_0 = nodes_4;
    (&kernelContext_16)->accel_0 = accel_4;
    (&kernelContext_16)->scratch_0 = scratch_4;
    (&kernelContext_16)->refs_0 = refs_4;
    (&kernelContext_16)->bonds_0 = bonds_4;
    (&kernelContext_16)->stiffness_0 = stiffness_4;
    (&kernelContext_16)->x_0 = x_4;
    (&kernelContext_16)->report_0 = report_4;
    threadgroup array<float2, int(32)> g_blast_part_4;
    (&kernelContext_16)->g_blast_part_0 = &g_blast_part_4;
    threadgroup array<float, int(1024)> g_wide_max_4;
    (&kernelContext_16)->g_wide_max_0 = &g_wide_max_4;
    threadgroup float g_wide_max_out_4;
    (&kernelContext_16)->g_wide_max_out_0 = &g_wide_max_out_4;
    uint tid_3 = thread_1.x;
    Block_0 _S74 = node_block_0(group_1, tid_3, &kernelContext_16);
    if((_S74.begin_0) >= (_S74.end_0))
    {
        return;
    }
    thread float4 p0_0 = float4(0.0f) ;
    uint p_0 = _S74.begin_0 + tid_3;
    if(p_0 < (_S74.end_0))
    {
        uint i_0 = (&kernelContext_16)->lists_0[p_0];
        BlastNode_natural_0 n_5 = (&kernelContext_16)->nodes_0[i_0];
        uint _S75 = i_0 * 2U;
        float4 _S76 = float4(*((&kernelContext_16)->accel_0+_S75)) ;
        float4 _S77 = float4(*((&kernelContext_16)->accel_0+(_S75 + 1U))) ;
        uint _S78 = ns_0(i_0, 16U, &kernelContext_16);
        bool _S79;
        if(any(_S76 != (float4(*((&kernelContext_16)->scratch_0+_S78)) )))
        {
            _S79 = true;
        }
        else
        {
            _S79 = any(_S77 != (float4(*((&kernelContext_16)->scratch_0+(_S78 + 1U))) ));
        }
        if(_S79)
        {
            p0_0.z = 1.0f;
        }
        float3 _S80 = - _S76.xyz;
        float _S81;
        if((n_5.w_ang_0) > 0.0f)
        {
            _S81 = 1.0f / n_5.w_ang_0;
        }
        else
        {
            _S81 = 1.0f;
        }
        float3 ba_0 = _S80 * float3(_S81) ;
        float3 _S82 = - _S77.xyz;
        if((n_5.w_lin_0) > 0.0f)
        {
            _S81 = (&kernelContext_16)->params_0->recip_length_0 / n_5.w_lin_0;
        }
        else
        {
            _S81 = (&kernelContext_16)->params_0->recip_length_0;
        }
        float3 bl_0 = _S82 * float3(_S81) ;
        p0_0.x = norm2_0(ba_0, bl_0);
        set_v_0(i_0, 14U, ba_0, bl_0, &kernelContext_16);
        set_v_0(i_0, 0U, ba_0, bl_0, &kernelContext_16);
        *((&kernelContext_16)->scratch_0+_S78) = packed_float4(_S76) ;
        *((&kernelContext_16)->scratch_0+(_S78 + 1U)) = packed_float4(_S77) ;
        float3 _S83 = float3(0.0f) ;
        set_v_0(i_0, 6U, _S83, _S83, &kernelContext_16);
        set_v_0(i_0, 2U, _S83, _S83, &kernelContext_16);
        bool _S84 = build_block_0(i_0, &kernelContext_16);
        if(!_S84)
        {
            p0_0.y = 1.0f;
        }
    }
    write_partial_0(tid_3, _S74.row_4, group_1.x, 0U, p0_0, &kernelContext_16);
    return;
}

Block_0 bond_block_0(uint3 group_2, uint tid_4, KernelContext_0 thread* kernelContext_17)
{
    Block_0 _S85 = node_block_0(group_2, tid_4, kernelContext_17);
    thread Block_0 b_4 = _S85;
    uint4 g_1 = group_uniform4_0(uint4(*(kernelContext_17->islands_0+(&b_4)->island_0)) , tid_4);
    uint _S86 = g_1.z + group_2.x * 1024U;
    (&b_4)->begin_0 = _S86;
    (&b_4)->end_0 = min(_S86 + 1024U, g_1.w);
    return b_4;
}

[[kernel]] void w_init_bonds(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], atomic<uint> device* meta_5 [[buffer(12)]], uint device* topo_args_5 [[buffer(13)]], packed_uint4 device* islands_5 [[buffer(1)]], BlastParams_0 constant* params_5 [[buffer(0)]], packed_float4 device* wide_5 [[buffer(11)]], uint device* lists_5 [[buffer(9)]], BlastNode_natural_0 device* nodes_5 [[buffer(2)]], packed_float4 device* accel_5 [[buffer(5)]], packed_float4 device* scratch_5 [[buffer(7)]], uint device* refs_5 [[buffer(3)]], BlastBond_natural_0 device* bonds_5 [[buffer(4)]], packed_float4 device* stiffness_5 [[buffer(10)]], packed_float4 device* x_5 [[buffer(6)]], packed_uint4 device* report_5 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_18;
    (&kernelContext_18)->meta_0 = meta_5;
    (&kernelContext_18)->topo_args_0 = topo_args_5;
    (&kernelContext_18)->islands_0 = islands_5;
    (&kernelContext_18)->params_0 = params_5;
    (&kernelContext_18)->wide_0 = wide_5;
    (&kernelContext_18)->lists_0 = lists_5;
    (&kernelContext_18)->nodes_0 = nodes_5;
    (&kernelContext_18)->accel_0 = accel_5;
    (&kernelContext_18)->scratch_0 = scratch_5;
    (&kernelContext_18)->refs_0 = refs_5;
    (&kernelContext_18)->bonds_0 = bonds_5;
    (&kernelContext_18)->stiffness_0 = stiffness_5;
    (&kernelContext_18)->x_0 = x_5;
    (&kernelContext_18)->report_0 = report_5;
    threadgroup array<float2, int(32)> g_blast_part_5;
    (&kernelContext_18)->g_blast_part_0 = &g_blast_part_5;
    threadgroup array<float, int(1024)> g_wide_max_5;
    (&kernelContext_18)->g_wide_max_0 = &g_wide_max_5;
    threadgroup float g_wide_max_out_5;
    (&kernelContext_18)->g_wide_max_out_0 = &g_wide_max_out_5;
    uint tid_5 = thread_2.x;
    Block_0 _S87 = bond_block_0(group_3, tid_5, &kernelContext_18);
    if((_S87.begin_0) >= (_S87.end_0))
    {
        return;
    }
    uint p_1 = _S87.begin_0 + tid_5;
    bool _S88;
    if(p_1 < (_S87.end_0))
    {
        _S88 = ((&kernelContext_18)->params_0->warm_0) == 0U;
    }
    else
    {
        _S88 = false;
    }
    if(_S88)
    {
        uint _S89 = (&kernelContext_18)->lists_0[p_1] * 2U;
        packed_float4 _S90 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_18)->x_0+_S89) = _S90;
        *((&kernelContext_18)->x_0+(_S89 + 1U)) = _S90;
    }
    return;
}

uint group_uniform_0(uint value_1, uint thread_3)
{
    return value_1;
}

float4 row_sum_0(uint tid_6, uint row_7, uint count_0, uint k_9, KernelContext_0 thread* kernelContext_19)
{
    float4 _S91 = float4(0.0f) ;
    uint b_5 = tid_6;
    float4 s_8 = _S91;
    for(;;)
    {
        if(b_5 < count_0)
        {
        }
        else
        {
            break;
        }
        uint _S92 = pt_0(row_7, b_5, k_9, kernelContext_19);
        float4 s_9 = s_8 + float4(*(kernelContext_19->wide_0+_S92)) ;
        b_5 = b_5 + 1024U;
        s_8 = s_9;
    }
    float2 _S93 = blast_group_sum_0(tid_6, s_8.xy, kernelContext_19);
    float2 _S94 = blast_group_sum_0(tid_6, s_8.zw, kernelContext_19);
    return float4(_S93, _S94);
}

uint blast_frame_0(uint node_count_1, uint k_10)
{
    return node_count_1 + 9U * k_10;
}

uint frame_word_0(uint k_11, uint w_1, KernelContext_0 thread* kernelContext_20)
{
    return (uint4(*(kernelContext_20->islands_0+(blast_frame_0(kernelContext_20->params_0->node_count_0, k_11) + w_1 / 4U))) )[w_1 % 4U];
}

uint frame_word_1(uint k_12, uint w_2, KernelContext_0 thread* kernelContext_21)
{
    return (uint4(*(kernelContext_21->islands_0+(blast_frame_0(kernelContext_21->params_0->node_count_0, k_12) + w_2 / 4U))) )[w_2 % 4U];
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

[[kernel]] void w_setup(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], atomic<uint> device* meta_6 [[buffer(12)]], uint device* topo_args_6 [[buffer(13)]], packed_uint4 device* islands_6 [[buffer(1)]], BlastParams_0 constant* params_6 [[buffer(0)]], packed_float4 device* wide_6 [[buffer(11)]], uint device* lists_6 [[buffer(9)]], BlastNode_natural_0 device* nodes_6 [[buffer(2)]], packed_float4 device* accel_6 [[buffer(5)]], packed_float4 device* scratch_6 [[buffer(7)]], uint device* refs_6 [[buffer(3)]], BlastBond_natural_0 device* bonds_6 [[buffer(4)]], packed_float4 device* stiffness_6 [[buffer(10)]], packed_float4 device* x_6 [[buffer(6)]], packed_uint4 device* report_6 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_22;
    (&kernelContext_22)->meta_0 = meta_6;
    (&kernelContext_22)->topo_args_0 = topo_args_6;
    (&kernelContext_22)->islands_0 = islands_6;
    (&kernelContext_22)->params_0 = params_6;
    (&kernelContext_22)->wide_0 = wide_6;
    (&kernelContext_22)->lists_0 = lists_6;
    (&kernelContext_22)->nodes_0 = nodes_6;
    (&kernelContext_22)->accel_0 = accel_6;
    (&kernelContext_22)->scratch_0 = scratch_6;
    (&kernelContext_22)->refs_0 = refs_6;
    (&kernelContext_22)->bonds_0 = bonds_6;
    (&kernelContext_22)->stiffness_0 = stiffness_6;
    (&kernelContext_22)->x_0 = x_6;
    (&kernelContext_22)->report_0 = report_6;
    threadgroup array<float2, int(32)> g_blast_part_6;
    (&kernelContext_22)->g_blast_part_0 = &g_blast_part_6;
    threadgroup array<float, int(1024)> g_wide_max_6;
    (&kernelContext_22)->g_wide_max_0 = &g_wide_max_6;
    threadgroup float g_wide_max_out_6;
    (&kernelContext_22)->g_wide_max_out_0 = &g_wide_max_out_6;
    uint tid_7 = thread_4.x;
    uint row_8 = group_4.x;
    uint _S95 = row_island_0(row_8, &kernelContext_22);
    uint island_1 = group_uniform_0(_S95, tid_7);
    uint4 r_3 = group_uniform4_0(uint4(*((&kernelContext_22)->islands_0+island_1)) , tid_7);
    float4 _S96 = row_sum_0(tid_7, row_8, blocks_0(r_3.y - r_3.x), 0U, &kernelContext_22);
    if(tid_7 != 0U)
    {
        return;
    }
    uint4 _S97 = uint4(*((&kernelContext_22)->report_0+island_1)) ;
    bool _S98;
    if(((&kernelContext_22)->params_0->warm_0) != 0U)
    {
        _S98 = ((_S97.y) & 4U) != 0U;
    }
    else
    {
        _S98 = false;
    }
    if(_S98)
    {
        _S98 = (_S96.z) == 0.0f;
    }
    else
    {
        _S98 = false;
    }
    uint f_0;
    if(_S98)
    {
        f_0 = 128U;
    }
    else
    {
        f_0 = 1U;
    }
    float delta2_0 = _S96.x * (&kernelContext_22)->params_0->tolerance_0 * (&kernelContext_22)->params_0->tolerance_0;
    if(f_0 == 1U)
    {
        if((_S96.y) > 0.0f)
        {
            _S98 = true;
        }
        else
        {
            uint _S99 = frame_word_0(island_1, 2U, &kernelContext_22);
            _S98 = _S99 != 0U;
        }
        if(_S98)
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
    uint _S100 = frame_word_0(island_1, 0U, &kernelContext_22);
    if(_S100 == 0U)
    {
        f_0 = f_0 | 8U;
    }
    float _S101 = float_bits_0(2139095040U);
    *((&kernelContext_22)->wide_0+st_0(row_8, 0U)) = packed_float4(float4(delta2_0, 0.0f, 0.0f, _S101)) ;
    *((&kernelContext_22)->wide_0+st_0(row_8, 1U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, _S101)) ;
    float _S102 = (as_type<float>((0U)));
    *((&kernelContext_22)->wide_0+st_0(row_8, 2U)) = packed_float4(float4(0.0f, 0.0f, _S102, (as_type<float>((f_0))))) ;
    *((&kernelContext_22)->wide_0+st_0(row_8, 3U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, _S102)) ;
    return;
}

uint bs_0(uint j_5, KernelContext_0 thread* kernelContext_23)
{
    return kernelContext_23->params_0->node_count_0 * kernelContext_23->params_0->node_vectors_0 + 2U * j_5;
}

float3 va_0(uint n_6, uint v_6, KernelContext_0 thread* kernelContext_24)
{
    uint _S103 = ns_0(n_6, v_6, kernelContext_24);
    return (float4(*(kernelContext_24->scratch_0+_S103)) ).xyz;
}

float3 vl_0(uint n_7, uint v_7, KernelContext_0 thread* kernelContext_25)
{
    uint _S104 = ns_0(n_7, v_7, kernelContext_25);
    return (float4(*(kernelContext_25->scratch_0+(_S104 + 1U))) ).xyz;
}

float3 rot_a_0(uint j_6, float3 v_8, KernelContext_0 thread* kernelContext_26)
{
    float3 _S105;
    if((kernelContext_26->params_0->rotation_0) != 0U)
    {
        float3 _S106 = stiff_0(j_6, 0U, v_8, kernelContext_26);
        _S105 = _S106;
    }
    else
    {
        _S105 = v_8;
    }
    return _S105;
}

float3 rot_l_0(uint j_7, float3 v_9, KernelContext_0 thread* kernelContext_27)
{
    float3 _S107;
    if((kernelContext_27->params_0->rotation_0) == 2U)
    {
        float3 _S108 = stiff_0(j_7, 2U, v_9, kernelContext_27);
        _S107 = _S108;
    }
    else
    {
        _S107 = v_9;
    }
    return _S107;
}

void bond_term_0(const BlastBond_natural_0 thread* b_6, uint j_8, uint v_10, float3 thread* ta_0, float3 thread* tl_0, KernelContext_0 thread* kernelContext_28)
{
    uint _S109 = b_6->n0_0;
    BlastNode_natural_0 m0_0 = kernelContext_28->nodes_0[b_6->n0_0];
    uint _S110 = b_6->n1_0;
    BlastNode_natural_0 m1_0 = kernelContext_28->nodes_0[b_6->n1_0];
    float3 _S111 = va_0(b_6->n0_0, v_10, kernelContext_28);
    float3 a0_1 = _S111 * float3(m0_0.w_ang_0) ;
    float3 _S112 = vl_0(_S109, v_10, kernelContext_28);
    float3 l0_0 = _S112 * float3(m0_0.w_lin_0) ;
    float3 _S113 = va_0(_S110, v_10, kernelContext_28);
    float3 a1_1 = _S113 * float3(m1_0.w_ang_0) ;
    float3 _S114 = vl_0(_S110, v_10, kernelContext_28);
    float3 l1_0 = _S114 * float3(m1_0.w_lin_0) ;
    float3 _S115 = rot_a_0(j_8, a0_1 - a1_1, kernelContext_28);
    float3 _S116 = float3(b_6->s_0) ;
    *ta_0 = _S115 * _S116;
    float3 _S117 = rot_l_0(j_8, l0_0 - l1_0 + (cross((float4(b_6->o0_0) ).xyz, a0_1) - cross((float4(b_6->o1_0) ).xyz, a1_1)), kernelContext_28);
    *tl_0 = _S117 * _S116;
    return;
}

void bond_term_1(const BlastBond_natural_0 thread* b_7, uint j_9, uint v_11, float3 thread* ta_1, float3 thread* tl_1, KernelContext_0 thread* kernelContext_29)
{
    uint _S118 = b_7->n0_0;
    BlastNode_natural_0 m0_1 = kernelContext_29->nodes_0[b_7->n0_0];
    uint _S119 = b_7->n1_0;
    BlastNode_natural_0 m1_1 = kernelContext_29->nodes_0[b_7->n1_0];
    float3 _S120 = va_0(b_7->n0_0, v_11, kernelContext_29);
    float3 a0_2 = _S120 * float3(m0_1.w_ang_0) ;
    float3 _S121 = vl_0(_S118, v_11, kernelContext_29);
    float3 l0_1 = _S121 * float3(m0_1.w_lin_0) ;
    float3 _S122 = va_0(_S119, v_11, kernelContext_29);
    float3 a1_2 = _S122 * float3(m1_1.w_ang_0) ;
    float3 _S123 = vl_0(_S119, v_11, kernelContext_29);
    float3 l1_1 = _S123 * float3(m1_1.w_lin_0) ;
    float3 _S124 = rot_a_0(j_9, a0_2 - a1_2, kernelContext_29);
    float3 _S125 = float3(b_7->s_0) ;
    *ta_1 = _S124 * _S125;
    float3 _S126 = rot_l_0(j_9, l0_1 - l1_1 + (cross((float4(b_7->o0_0) ).xyz, a0_2) - cross((float4(b_7->o1_0) ).xyz, a1_2)), kernelContext_29);
    *tl_1 = _S126 * _S125;
    return;
}

void gather_0(uint node_2, bool from_t_0, bool with_u_0, float3 thread* ang_0, float3 thread* lin_0, KernelContext_0 thread* kernelContext_30)
{
    BlastNode_natural_0 n_8 = kernelContext_30->nodes_0[node_2];
    float3 _S127 = float3(0.0f) ;
    uint r_4 = n_8.ref_begin_0;
    float3 acc_a_0 = _S127;
    float3 acc_l_0 = _S127;
    for(;;)
    {
        if(r_4 < (n_8.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_2 = kernelContext_30->refs_0[r_4];
        uint j_10 = ref_2 & 2147483647U;
        thread BlastBond_natural_0 _S128 = *(kernelContext_30->bonds_0+j_10);
        if(!(((&_S128)->health_0) > 0.0f))
        {
            r_4 = r_4 + 1U;
            continue;
        }
        float3 ya_0;
        float3 yl_0;
        if(from_t_0)
        {
            uint _S129 = bs_0(j_10, kernelContext_30);
            float3 _S130 = (float4(*(kernelContext_30->scratch_0+(_S129 + 1U))) ).xyz;
            ya_0 = (float4(*(kernelContext_30->scratch_0+_S129)) ).xyz;
            yl_0 = _S130;
        }
        else
        {
            uint _S131 = j_10 * 2U;
            float3 ya_1 = (float4(*(kernelContext_30->x_0+_S131)) ).xyz;
            float3 yl_1 = (float4(*(kernelContext_30->x_0+(_S131 + 1U))) ).xyz;
            if(with_u_0)
            {
                thread float3 ta_2;
                thread float3 tl_2;
                bond_term_0(&_S128, j_10, 6U, &ta_2, &tl_2, kernelContext_30);
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
        float3 _S132 = rot_a_0(j_10, ya_0, kernelContext_30);
        float3 _S133 = float3((&_S128)->s_0) ;
        float3 ya_2 = _S132 * _S133;
        float3 _S134 = rot_l_0(j_10, yl_0, kernelContext_30);
        float3 yl_3 = _S134 * _S133;
        float3 acc_a_1;
        float3 acc_l_1;
        if((ref_2 & 2147483648U) != 0U)
        {
            float3 acc_l_2 = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross((float4((&_S128)->o1_0) ).xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            float3 acc_l_3 = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross((float4((&_S128)->o0_0) ).xyz, yl_3));
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
    thread KernelContext_0 kernelContext_31;
    (&kernelContext_31)->meta_0 = meta_7;
    (&kernelContext_31)->topo_args_0 = topo_args_7;
    (&kernelContext_31)->islands_0 = islands_7;
    (&kernelContext_31)->params_0 = params_7;
    (&kernelContext_31)->wide_0 = wide_7;
    (&kernelContext_31)->lists_0 = lists_7;
    (&kernelContext_31)->nodes_0 = nodes_7;
    (&kernelContext_31)->accel_0 = accel_7;
    (&kernelContext_31)->scratch_0 = scratch_7;
    (&kernelContext_31)->refs_0 = refs_7;
    (&kernelContext_31)->bonds_0 = bonds_7;
    (&kernelContext_31)->stiffness_0 = stiffness_7;
    (&kernelContext_31)->x_0 = x_7;
    (&kernelContext_31)->report_0 = report_7;
    threadgroup array<float2, int(32)> g_blast_part_7;
    (&kernelContext_31)->g_blast_part_0 = &g_blast_part_7;
    threadgroup array<float, int(1024)> g_wide_max_7;
    (&kernelContext_31)->g_wide_max_0 = &g_wide_max_7;
    threadgroup float g_wide_max_out_7;
    (&kernelContext_31)->g_wide_max_out_0 = &g_wide_max_out_7;
    uint tid_8 = thread_5.x;
    Block_0 _S135 = node_block_0(group_5, tid_8, &kernelContext_31);
    bool _S136;
    if((_S135.begin_0) >= (_S135.end_0))
    {
        _S136 = true;
    }
    else
    {
        _S136 = ((&kernelContext_31)->params_0->warm_0) == 0U;
    }
    if(_S136)
    {
        _S136 = true;
    }
    else
    {
        _S136 = ((_S135.flags_0) & 1U) == 0U;
    }
    if(_S136)
    {
        return;
    }
    uint p_2 = _S135.begin_0 + tid_8;
    if(p_2 < (_S135.end_0))
    {
        uint i_1 = (&kernelContext_31)->lists_0[p_2];
        thread float3 a_2;
        thread float3 l_2;
        gather_0(i_1, false, false, &a_2, &l_2, &kernelContext_31);
        float3 _S137 = va_0(i_1, 14U, &kernelContext_31);
        float3 _S138 = _S137 - a_2;
        float3 _S139 = vl_0(i_1, 14U, &kernelContext_31);
        set_v_0(i_1, 0U, _S138, _S139 - l_2, &kernelContext_31);
    }
    return;
}

bool runs_0(uint flags_1, bool only_rebuild_0)
{
    bool _S140;
    if((flags_1 & 1U) != 0U)
    {
        if(!only_rebuild_0)
        {
            _S140 = true;
        }
        else
        {
            _S140 = (flags_1 & 64U) != 0U;
        }
    }
    else
    {
        _S140 = false;
    }
    return _S140;
}

float frame_float_0(uint k_13, uint w_3, KernelContext_0 thread* kernelContext_32)
{
    uint _S141 = frame_word_1(k_13, w_3, kernelContext_32);
    return (as_type<float>((_S141)));
}

float frame_float_1(uint k_14, uint w_4, KernelContext_0 thread* kernelContext_33)
{
    uint _S142 = frame_word_1(k_14, w_4, kernelContext_33);
    return (as_type<float>((_S142)));
}

float3 frame_axis_0(uint k_15, uint rotations_0, uint kk_1, KernelContext_0 thread* kernelContext_34)
{
    if(rotations_0 == 1U)
    {
        float _S143 = frame_float_1(k_15, 4U, kernelContext_34);
        float _S144 = frame_float_1(k_15, 5U, kernelContext_34);
        float _S145 = frame_float_1(k_15, 6U, kernelContext_34);
        return float3(_S143, _S144, _S145);
    }
    float _S146;
    if(kk_1 == 0U)
    {
        _S146 = 1.0f;
    }
    else
    {
        _S146 = 0.0f;
    }
    float _S147;
    if(kk_1 == 1U)
    {
        _S147 = 1.0f;
    }
    else
    {
        _S147 = 0.0f;
    }
    float _S148;
    if(kk_1 == 2U)
    {
        _S148 = 1.0f;
    }
    else
    {
        _S148 = 0.0f;
    }
    return float3(_S146, _S147, _S148);
}

void frame_terms_0(uint k_16, uint rotations_1, uint i_2, uint v_12, array<float, int(6)> thread* s_10, KernelContext_0 thread* kernelContext_35)
{
    BlastNode_natural_0 n_9 = kernelContext_35->nodes_0[i_2];
    float3 _S149 = vl_0(i_2, v_12, kernelContext_35);
    float3 lin_1 = _S149 / float3(n_9.w_lin_0) ;
    float3 _S150 = va_0(i_2, v_12, kernelContext_35);
    float3 _S151 = _S150 / float3(n_9.w_ang_0)  - cross((float4(n_9.position_0) ).xyz, lin_1);
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
        float3 _S152 = frame_axis_0(k_16, rotations_1, kk_2, kernelContext_35);
        (*s_10)[kk_2] = (*s_10)[kk_2] + dot(_S152, _S151);
        kk_2 = kk_2 + 1U;
    }
    uint j_11 = 0U;
    for(;;)
    {
        if(j_11 < 3U)
        {
        }
        else
        {
            break;
        }
        (*s_10)[rotations_1 + j_11] = (*s_10)[rotations_1 + j_11] + lin_1[j_11];
        j_11 = j_11 + 1U;
    }
    return;
}

void proj_part_0(uint3 group_6, uint tid_9, uint v_13, bool only_rebuild_1, KernelContext_0 thread* kernelContext_36)
{
    Block_0 _S153 = node_block_0(group_6, tid_9, kernelContext_36);
    bool _S154;
    if((_S153.begin_0) >= (_S153.end_0))
    {
        _S154 = true;
    }
    else
    {
        _S154 = !runs_0(_S153.flags_0, only_rebuild_1);
    }
    if(_S154)
    {
        _S154 = true;
    }
    else
    {
        _S154 = ((_S153.flags_0) & 8U) == 0U;
    }
    if(_S154)
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
    uint p_3 = _S153.begin_0 + tid_9;
    if(p_3 < (_S153.end_0))
    {
        uint _S155 = row_island_0(_S153.row_4, kernelContext_36);
        uint _S156 = frame_word_0(_S155, 1U, kernelContext_36);
        frame_terms_0(_S155, _S156, kernelContext_36->lists_0[p_3], v_13, &s_11, kernelContext_36);
    }
    uint _S157 = group_6.x;
    write_partial_0(tid_9, _S153.row_4, _S157, 0U, float4(s_11[int(0)], s_11[int(1)], s_11[int(2)], s_11[int(3)]), kernelContext_36);
    write_partial_0(tid_9, _S153.row_4, _S157, 1U, float4(s_11[int(4)], s_11[int(5)], 0.0f, 0.0f), kernelContext_36);
    return;
}

[[kernel]] void w_proj_part_r(uint3 g_2 [[threadgroup_position_in_grid]], uint3 t_0 [[thread_position_in_threadgroup]], atomic<uint> device* meta_8 [[buffer(12)]], uint device* topo_args_8 [[buffer(13)]], packed_uint4 device* islands_8 [[buffer(1)]], BlastParams_0 constant* params_8 [[buffer(0)]], packed_float4 device* wide_8 [[buffer(11)]], uint device* lists_8 [[buffer(9)]], BlastNode_natural_0 device* nodes_8 [[buffer(2)]], packed_float4 device* accel_8 [[buffer(5)]], packed_float4 device* scratch_8 [[buffer(7)]], uint device* refs_8 [[buffer(3)]], BlastBond_natural_0 device* bonds_8 [[buffer(4)]], packed_float4 device* stiffness_8 [[buffer(10)]], packed_float4 device* x_8 [[buffer(6)]], packed_uint4 device* report_8 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_37;
    (&kernelContext_37)->meta_0 = meta_8;
    (&kernelContext_37)->topo_args_0 = topo_args_8;
    (&kernelContext_37)->islands_0 = islands_8;
    (&kernelContext_37)->params_0 = params_8;
    (&kernelContext_37)->wide_0 = wide_8;
    (&kernelContext_37)->lists_0 = lists_8;
    (&kernelContext_37)->nodes_0 = nodes_8;
    (&kernelContext_37)->accel_0 = accel_8;
    (&kernelContext_37)->scratch_0 = scratch_8;
    (&kernelContext_37)->refs_0 = refs_8;
    (&kernelContext_37)->bonds_0 = bonds_8;
    (&kernelContext_37)->stiffness_0 = stiffness_8;
    (&kernelContext_37)->x_0 = x_8;
    (&kernelContext_37)->report_0 = report_8;
    threadgroup array<float2, int(32)> g_blast_part_8;
    (&kernelContext_37)->g_blast_part_0 = &g_blast_part_8;
    threadgroup array<float, int(1024)> g_wide_max_8;
    (&kernelContext_37)->g_wide_max_0 = &g_wide_max_8;
    threadgroup float g_wide_max_out_8;
    (&kernelContext_37)->g_wide_max_out_0 = &g_wide_max_out_8;
    proj_part_0(g_2, t_0.x, 0U, false, &kernelContext_37);
    return;
}

uint4 row_range_0(uint tid_10, uint row_9, KernelContext_0 thread* kernelContext_38)
{
    uint _S158 = row_island_0(row_9, kernelContext_38);
    return group_uniform4_0(uint4(*(kernelContext_38->islands_0+group_uniform_0(_S158, tid_10))) , tid_10);
}

void frame_solve_0(uint k_17, uint rotations_2, const array<float, int(6)> thread* sum_0, float3 thread* omega_0, float3 thread* t_1, KernelContext_0 thread* kernelContext_39)
{
    uint j_12;
    uint d_0 = rotations_2 + 3U;
    thread array<float, int(6)> c_2;
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
        c_2[z_2] = 0.0f;
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
        float _S159 = (*sum_0)[row_10];
        float _S160 = frame_float_0(k_17, 8U + row_10, kernelContext_39);
        float _S161 = _S159 * _S160;
        j_12 = 0U;
        float value_2 = _S161;
        for(;;)
        {
            if(j_12 < row_10)
            {
            }
            else
            {
                break;
            }
            float _S162 = frame_float_0(k_17, 14U + row_10 * (row_10 + 1U) / 2U + j_12, kernelContext_39);
            float value_3 = value_2 - _S162 * c_2[j_12];
            j_12 = j_12 + 1U;
            value_2 = value_3;
        }
        uint _S163 = row_10 + 1U;
        float _S164 = frame_float_0(k_17, 14U + row_10 * _S163 / 2U + row_10, kernelContext_39);
        c_2[row_10] = value_2 / _S164;
        row_10 = _S163;
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
        uint _S165 = r_5 + 1U;
        j_12 = _S165;
        for(;;)
        {
            if(j_12 < d_0)
            {
            }
            else
            {
                break;
            }
            uint _S166 = j_12 + 1U;
            float _S167 = frame_float_0(k_17, 14U + j_12 * _S166 / 2U + r_5, kernelContext_39);
            c_2[r_5] = c_2[r_5] - _S167 * c_2[j_12];
            j_12 = _S166;
        }
        float _S168 = frame_float_0(k_17, 14U + r_5 * _S165 / 2U + r_5, kernelContext_39);
        c_2[r_5] = c_2[r_5] / _S168;
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
        float _S169 = frame_float_0(k_17, 8U + z_2, kernelContext_39);
        c_2[z_2] = c_2[z_2] * _S169;
        z_2 = z_2 + 1U;
    }
    *omega_0 = float3(0.0f) ;
    j_12 = 0U;
    for(;;)
    {
        if(j_12 < rotations_2)
        {
        }
        else
        {
            break;
        }
        float3 _S170 = frame_axis_0(k_17, rotations_2, j_12, kernelContext_39);
        *omega_0 = *omega_0 + _S170 * float3(c_2[j_12]) ;
        j_12 = j_12 + 1U;
    }
    *t_1 = float3(c_2[rotations_2], c_2[rotations_2 + 1U], c_2[rotations_2 + 2U]);
    return;
}

void proj_solve_0(uint3 group_7, uint tid_11, bool only_rebuild_2, KernelContext_0 thread* kernelContext_40)
{
    uint row_12 = group_7.x;
    uint _S171 = flags_of_0(row_12, kernelContext_40);
    uint f_1 = group_uniform_0(_S171, tid_11);
    bool _S172;
    if(!runs_0(f_1, only_rebuild_2))
    {
        _S172 = true;
    }
    else
    {
        _S172 = (f_1 & 8U) == 0U;
    }
    if(_S172)
    {
        return;
    }
    uint4 _S173 = row_range_0(tid_11, row_12, kernelContext_40);
    uint nb_1 = blocks_0(_S173.y - _S173.x);
    float4 _S174 = row_sum_0(tid_11, row_12, nb_1, 0U, kernelContext_40);
    float4 _S175 = row_sum_0(tid_11, row_12, nb_1, 1U, kernelContext_40);
    if(tid_11 != 0U)
    {
        return;
    }
    uint _S176 = row_island_0(row_12, kernelContext_40);
    float _S177 = _S174.x;
    float _S178 = _S174.y;
    float _S179 = _S174.z;
    float _S180 = _S174.w;
    float _S181 = _S175.x;
    float _S182 = _S175.y;
    uint _S183 = frame_word_0(_S176, 1U, kernelContext_40);
    thread array<float, int(6)> _S184;
    _S184[int(0)] = _S177;
    _S184[int(1)] = _S178;
    _S184[int(2)] = _S179;
    _S184[int(3)] = _S180;
    _S184[int(4)] = _S181;
    _S184[int(5)] = _S182;
    thread float3 omega_1;
    thread float3 t_2;
    frame_solve_0(_S176, _S183, &_S184, &omega_1, &t_2, kernelContext_40);
    *(kernelContext_40->wide_0+st_0(row_12, 4U)) = packed_float4(float4(omega_1, 0.0f)) ;
    *(kernelContext_40->wide_0+st_0(row_12, 5U)) = packed_float4(float4(t_2, 0.0f)) ;
    return;
}

[[kernel]] void w_proj_solve(uint3 g_3 [[threadgroup_position_in_grid]], uint3 t_3 [[thread_position_in_threadgroup]], atomic<uint> device* meta_9 [[buffer(12)]], uint device* topo_args_9 [[buffer(13)]], packed_uint4 device* islands_9 [[buffer(1)]], BlastParams_0 constant* params_9 [[buffer(0)]], packed_float4 device* wide_9 [[buffer(11)]], uint device* lists_9 [[buffer(9)]], BlastNode_natural_0 device* nodes_9 [[buffer(2)]], packed_float4 device* accel_9 [[buffer(5)]], packed_float4 device* scratch_9 [[buffer(7)]], uint device* refs_9 [[buffer(3)]], BlastBond_natural_0 device* bonds_9 [[buffer(4)]], packed_float4 device* stiffness_9 [[buffer(10)]], packed_float4 device* x_9 [[buffer(6)]], packed_uint4 device* report_9 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_41;
    (&kernelContext_41)->meta_0 = meta_9;
    (&kernelContext_41)->topo_args_0 = topo_args_9;
    (&kernelContext_41)->islands_0 = islands_9;
    (&kernelContext_41)->params_0 = params_9;
    (&kernelContext_41)->wide_0 = wide_9;
    (&kernelContext_41)->lists_0 = lists_9;
    (&kernelContext_41)->nodes_0 = nodes_9;
    (&kernelContext_41)->accel_0 = accel_9;
    (&kernelContext_41)->scratch_0 = scratch_9;
    (&kernelContext_41)->refs_0 = refs_9;
    (&kernelContext_41)->bonds_0 = bonds_9;
    (&kernelContext_41)->stiffness_0 = stiffness_9;
    (&kernelContext_41)->x_0 = x_9;
    (&kernelContext_41)->report_0 = report_9;
    threadgroup array<float2, int(32)> g_blast_part_9;
    (&kernelContext_41)->g_blast_part_0 = &g_blast_part_9;
    threadgroup array<float, int(1024)> g_wide_max_9;
    (&kernelContext_41)->g_wide_max_0 = &g_wide_max_9;
    threadgroup float g_wide_max_out_9;
    (&kernelContext_41)->g_wide_max_out_0 = &g_wide_max_out_9;
    proj_solve_0(g_3, t_3.x, false, &kernelContext_41);
    return;
}

void frame_apply_0(uint i_3, uint v_14, float3 omega_2, float3 t_4, KernelContext_0 thread* kernelContext_42)
{
    BlastNode_natural_0 n_10 = kernelContext_42->nodes_0[i_3];
    float3 _S185 = va_0(i_3, v_14, kernelContext_42);
    float3 _S186 = _S185 - omega_2 / float3(n_10.w_ang_0) ;
    float3 _S187 = vl_0(i_3, v_14, kernelContext_42);
    set_v_0(i_3, v_14, _S186, _S187 - (t_4 + cross((float4(n_10.position_0) ).xyz, omega_2)) / float3(n_10.w_lin_0) , kernelContext_42);
    return;
}

void proj_apply_0(uint3 group_8, uint tid_12, uint v_15, bool only_rebuild_3, KernelContext_0 thread* kernelContext_43)
{
    Block_0 _S188 = node_block_0(group_8, tid_12, kernelContext_43);
    bool _S189;
    if((_S188.begin_0) >= (_S188.end_0))
    {
        _S189 = true;
    }
    else
    {
        _S189 = !runs_0(_S188.flags_0, only_rebuild_3);
    }
    if(_S189)
    {
        _S189 = true;
    }
    else
    {
        _S189 = ((_S188.flags_0) & 8U) == 0U;
    }
    if(_S189)
    {
        return;
    }
    uint p_4 = _S188.begin_0 + tid_12;
    if(p_4 >= (_S188.end_0))
    {
        return;
    }
    frame_apply_0(kernelContext_43->lists_0[p_4], v_15, (float4(*(kernelContext_43->wide_0+st_0(_S188.row_4, 4U))) ).xyz, (float4(*(kernelContext_43->wide_0+st_0(_S188.row_4, 5U))) ).xyz, kernelContext_43);
    return;
}

[[kernel]] void w_proj_apply_r(uint3 g_4 [[threadgroup_position_in_grid]], uint3 t_5 [[thread_position_in_threadgroup]], atomic<uint> device* meta_10 [[buffer(12)]], uint device* topo_args_10 [[buffer(13)]], packed_uint4 device* islands_10 [[buffer(1)]], BlastParams_0 constant* params_10 [[buffer(0)]], packed_float4 device* wide_10 [[buffer(11)]], uint device* lists_10 [[buffer(9)]], BlastNode_natural_0 device* nodes_10 [[buffer(2)]], packed_float4 device* accel_10 [[buffer(5)]], packed_float4 device* scratch_10 [[buffer(7)]], uint device* refs_10 [[buffer(3)]], BlastBond_natural_0 device* bonds_10 [[buffer(4)]], packed_float4 device* stiffness_10 [[buffer(10)]], packed_float4 device* x_10 [[buffer(6)]], packed_uint4 device* report_10 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_44;
    (&kernelContext_44)->meta_0 = meta_10;
    (&kernelContext_44)->topo_args_0 = topo_args_10;
    (&kernelContext_44)->islands_0 = islands_10;
    (&kernelContext_44)->params_0 = params_10;
    (&kernelContext_44)->wide_0 = wide_10;
    (&kernelContext_44)->lists_0 = lists_10;
    (&kernelContext_44)->nodes_0 = nodes_10;
    (&kernelContext_44)->accel_0 = accel_10;
    (&kernelContext_44)->scratch_0 = scratch_10;
    (&kernelContext_44)->refs_0 = refs_10;
    (&kernelContext_44)->bonds_0 = bonds_10;
    (&kernelContext_44)->stiffness_0 = stiffness_10;
    (&kernelContext_44)->x_0 = x_10;
    (&kernelContext_44)->report_0 = report_10;
    threadgroup array<float2, int(32)> g_blast_part_10;
    (&kernelContext_44)->g_blast_part_0 = &g_blast_part_10;
    threadgroup array<float, int(1024)> g_wide_max_10;
    (&kernelContext_44)->g_wide_max_0 = &g_wide_max_10;
    threadgroup float g_wide_max_out_10;
    (&kernelContext_44)->g_wide_max_out_0 = &g_wide_max_out_10;
    proj_apply_0(g_4, t_5.x, 0U, false, &kernelContext_44);
    return;
}

[[kernel]] void w_proj_part_rb(uint3 g_5 [[threadgroup_position_in_grid]], uint3 t_6 [[thread_position_in_threadgroup]], atomic<uint> device* meta_11 [[buffer(12)]], uint device* topo_args_11 [[buffer(13)]], packed_uint4 device* islands_11 [[buffer(1)]], BlastParams_0 constant* params_11 [[buffer(0)]], packed_float4 device* wide_11 [[buffer(11)]], uint device* lists_11 [[buffer(9)]], BlastNode_natural_0 device* nodes_11 [[buffer(2)]], packed_float4 device* accel_11 [[buffer(5)]], packed_float4 device* scratch_11 [[buffer(7)]], uint device* refs_11 [[buffer(3)]], BlastBond_natural_0 device* bonds_11 [[buffer(4)]], packed_float4 device* stiffness_11 [[buffer(10)]], packed_float4 device* x_11 [[buffer(6)]], packed_uint4 device* report_11 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_45;
    (&kernelContext_45)->meta_0 = meta_11;
    (&kernelContext_45)->topo_args_0 = topo_args_11;
    (&kernelContext_45)->islands_0 = islands_11;
    (&kernelContext_45)->params_0 = params_11;
    (&kernelContext_45)->wide_0 = wide_11;
    (&kernelContext_45)->lists_0 = lists_11;
    (&kernelContext_45)->nodes_0 = nodes_11;
    (&kernelContext_45)->accel_0 = accel_11;
    (&kernelContext_45)->scratch_0 = scratch_11;
    (&kernelContext_45)->refs_0 = refs_11;
    (&kernelContext_45)->bonds_0 = bonds_11;
    (&kernelContext_45)->stiffness_0 = stiffness_11;
    (&kernelContext_45)->x_0 = x_11;
    (&kernelContext_45)->report_0 = report_11;
    threadgroup array<float2, int(32)> g_blast_part_11;
    (&kernelContext_45)->g_blast_part_0 = &g_blast_part_11;
    threadgroup array<float, int(1024)> g_wide_max_11;
    (&kernelContext_45)->g_wide_max_0 = &g_wide_max_11;
    threadgroup float g_wide_max_out_11;
    (&kernelContext_45)->g_wide_max_out_0 = &g_wide_max_out_11;
    proj_part_0(g_5, t_6.x, 0U, true, &kernelContext_45);
    return;
}

[[kernel]] void w_proj_solve_rb(uint3 g_6 [[threadgroup_position_in_grid]], uint3 t_7 [[thread_position_in_threadgroup]], atomic<uint> device* meta_12 [[buffer(12)]], uint device* topo_args_12 [[buffer(13)]], packed_uint4 device* islands_12 [[buffer(1)]], BlastParams_0 constant* params_12 [[buffer(0)]], packed_float4 device* wide_12 [[buffer(11)]], uint device* lists_12 [[buffer(9)]], BlastNode_natural_0 device* nodes_12 [[buffer(2)]], packed_float4 device* accel_12 [[buffer(5)]], packed_float4 device* scratch_12 [[buffer(7)]], uint device* refs_12 [[buffer(3)]], BlastBond_natural_0 device* bonds_12 [[buffer(4)]], packed_float4 device* stiffness_12 [[buffer(10)]], packed_float4 device* x_12 [[buffer(6)]], packed_uint4 device* report_12 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_46;
    (&kernelContext_46)->meta_0 = meta_12;
    (&kernelContext_46)->topo_args_0 = topo_args_12;
    (&kernelContext_46)->islands_0 = islands_12;
    (&kernelContext_46)->params_0 = params_12;
    (&kernelContext_46)->wide_0 = wide_12;
    (&kernelContext_46)->lists_0 = lists_12;
    (&kernelContext_46)->nodes_0 = nodes_12;
    (&kernelContext_46)->accel_0 = accel_12;
    (&kernelContext_46)->scratch_0 = scratch_12;
    (&kernelContext_46)->refs_0 = refs_12;
    (&kernelContext_46)->bonds_0 = bonds_12;
    (&kernelContext_46)->stiffness_0 = stiffness_12;
    (&kernelContext_46)->x_0 = x_12;
    (&kernelContext_46)->report_0 = report_12;
    threadgroup array<float2, int(32)> g_blast_part_12;
    (&kernelContext_46)->g_blast_part_0 = &g_blast_part_12;
    threadgroup array<float, int(1024)> g_wide_max_12;
    (&kernelContext_46)->g_wide_max_0 = &g_wide_max_12;
    threadgroup float g_wide_max_out_12;
    (&kernelContext_46)->g_wide_max_out_0 = &g_wide_max_out_12;
    proj_solve_0(g_6, t_7.x, true, &kernelContext_46);
    return;
}

[[kernel]] void w_proj_apply_rb(uint3 g_7 [[threadgroup_position_in_grid]], uint3 t_8 [[thread_position_in_threadgroup]], atomic<uint> device* meta_13 [[buffer(12)]], uint device* topo_args_13 [[buffer(13)]], packed_uint4 device* islands_13 [[buffer(1)]], BlastParams_0 constant* params_13 [[buffer(0)]], packed_float4 device* wide_13 [[buffer(11)]], uint device* lists_13 [[buffer(9)]], BlastNode_natural_0 device* nodes_13 [[buffer(2)]], packed_float4 device* accel_13 [[buffer(5)]], packed_float4 device* scratch_13 [[buffer(7)]], uint device* refs_13 [[buffer(3)]], BlastBond_natural_0 device* bonds_13 [[buffer(4)]], packed_float4 device* stiffness_13 [[buffer(10)]], packed_float4 device* x_13 [[buffer(6)]], packed_uint4 device* report_13 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_47;
    (&kernelContext_47)->meta_0 = meta_13;
    (&kernelContext_47)->topo_args_0 = topo_args_13;
    (&kernelContext_47)->islands_0 = islands_13;
    (&kernelContext_47)->params_0 = params_13;
    (&kernelContext_47)->wide_0 = wide_13;
    (&kernelContext_47)->lists_0 = lists_13;
    (&kernelContext_47)->nodes_0 = nodes_13;
    (&kernelContext_47)->accel_0 = accel_13;
    (&kernelContext_47)->scratch_0 = scratch_13;
    (&kernelContext_47)->refs_0 = refs_13;
    (&kernelContext_47)->bonds_0 = bonds_13;
    (&kernelContext_47)->stiffness_0 = stiffness_13;
    (&kernelContext_47)->x_0 = x_13;
    (&kernelContext_47)->report_0 = report_13;
    threadgroup array<float2, int(32)> g_blast_part_13;
    (&kernelContext_47)->g_blast_part_0 = &g_blast_part_13;
    threadgroup array<float, int(1024)> g_wide_max_13;
    (&kernelContext_47)->g_wide_max_0 = &g_wide_max_13;
    threadgroup float g_wide_max_out_13;
    (&kernelContext_47)->g_wide_max_out_0 = &g_wide_max_out_13;
    proj_apply_0(g_7, t_8.x, 0U, true, &kernelContext_47);
    return;
}

[[kernel]] void w_proj_part_g(uint3 g_8 [[threadgroup_position_in_grid]], uint3 t_9 [[thread_position_in_threadgroup]], atomic<uint> device* meta_14 [[buffer(12)]], uint device* topo_args_14 [[buffer(13)]], packed_uint4 device* islands_14 [[buffer(1)]], BlastParams_0 constant* params_14 [[buffer(0)]], packed_float4 device* wide_14 [[buffer(11)]], uint device* lists_14 [[buffer(9)]], BlastNode_natural_0 device* nodes_14 [[buffer(2)]], packed_float4 device* accel_14 [[buffer(5)]], packed_float4 device* scratch_14 [[buffer(7)]], uint device* refs_14 [[buffer(3)]], BlastBond_natural_0 device* bonds_14 [[buffer(4)]], packed_float4 device* stiffness_14 [[buffer(10)]], packed_float4 device* x_14 [[buffer(6)]], packed_uint4 device* report_14 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_48;
    (&kernelContext_48)->meta_0 = meta_14;
    (&kernelContext_48)->topo_args_0 = topo_args_14;
    (&kernelContext_48)->islands_0 = islands_14;
    (&kernelContext_48)->params_0 = params_14;
    (&kernelContext_48)->wide_0 = wide_14;
    (&kernelContext_48)->lists_0 = lists_14;
    (&kernelContext_48)->nodes_0 = nodes_14;
    (&kernelContext_48)->accel_0 = accel_14;
    (&kernelContext_48)->scratch_0 = scratch_14;
    (&kernelContext_48)->refs_0 = refs_14;
    (&kernelContext_48)->bonds_0 = bonds_14;
    (&kernelContext_48)->stiffness_0 = stiffness_14;
    (&kernelContext_48)->x_0 = x_14;
    (&kernelContext_48)->report_0 = report_14;
    threadgroup array<float2, int(32)> g_blast_part_14;
    (&kernelContext_48)->g_blast_part_0 = &g_blast_part_14;
    threadgroup array<float, int(1024)> g_wide_max_14;
    (&kernelContext_48)->g_wide_max_0 = &g_wide_max_14;
    threadgroup float g_wide_max_out_14;
    (&kernelContext_48)->g_wide_max_out_0 = &g_wide_max_out_14;
    proj_part_0(g_8, t_9.x, 8U, false, &kernelContext_48);
    return;
}

[[kernel]] void w_proj_apply_g(uint3 g_9 [[threadgroup_position_in_grid]], uint3 t_10 [[thread_position_in_threadgroup]], atomic<uint> device* meta_15 [[buffer(12)]], uint device* topo_args_15 [[buffer(13)]], packed_uint4 device* islands_15 [[buffer(1)]], BlastParams_0 constant* params_15 [[buffer(0)]], packed_float4 device* wide_15 [[buffer(11)]], uint device* lists_15 [[buffer(9)]], BlastNode_natural_0 device* nodes_15 [[buffer(2)]], packed_float4 device* accel_15 [[buffer(5)]], packed_float4 device* scratch_15 [[buffer(7)]], uint device* refs_15 [[buffer(3)]], BlastBond_natural_0 device* bonds_15 [[buffer(4)]], packed_float4 device* stiffness_15 [[buffer(10)]], packed_float4 device* x_15 [[buffer(6)]], packed_uint4 device* report_15 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_49;
    (&kernelContext_49)->meta_0 = meta_15;
    (&kernelContext_49)->topo_args_0 = topo_args_15;
    (&kernelContext_49)->islands_0 = islands_15;
    (&kernelContext_49)->params_0 = params_15;
    (&kernelContext_49)->wide_0 = wide_15;
    (&kernelContext_49)->lists_0 = lists_15;
    (&kernelContext_49)->nodes_0 = nodes_15;
    (&kernelContext_49)->accel_0 = accel_15;
    (&kernelContext_49)->scratch_0 = scratch_15;
    (&kernelContext_49)->refs_0 = refs_15;
    (&kernelContext_49)->bonds_0 = bonds_15;
    (&kernelContext_49)->stiffness_0 = stiffness_15;
    (&kernelContext_49)->x_0 = x_15;
    (&kernelContext_49)->report_0 = report_15;
    threadgroup array<float2, int(32)> g_blast_part_15;
    (&kernelContext_49)->g_blast_part_0 = &g_blast_part_15;
    threadgroup array<float, int(1024)> g_wide_max_15;
    (&kernelContext_49)->g_wide_max_0 = &g_wide_max_15;
    threadgroup float g_wide_max_out_15;
    (&kernelContext_49)->g_wide_max_out_0 = &g_wide_max_out_15;
    proj_apply_0(g_9, t_10.x, 8U, false, &kernelContext_49);
    return;
}

void bond_r_0(uint3 group_9, uint tid_13, bool only_rebuild_4, KernelContext_0 thread* kernelContext_50)
{
    Block_0 _S190 = bond_block_0(group_9, tid_13, kernelContext_50);
    bool _S191;
    if((_S190.begin_0) >= (_S190.end_0))
    {
        _S191 = true;
    }
    else
    {
        _S191 = !runs_0(_S190.flags_0, only_rebuild_4);
    }
    if(_S191)
    {
        return;
    }
    uint p_5 = _S190.begin_0 + tid_13;
    float z_3;
    float f_2;
    if(p_5 < (_S190.end_0))
    {
        uint j_13 = kernelContext_50->lists_0[p_5];
        thread BlastBond_natural_0 _S192 = *(kernelContext_50->bonds_0+j_13);
        float3 _S193 = float3(0.0f) ;
        thread float3 ta_3 = _S193;
        thread float3 tl_3 = _S193;
        if(((&_S192)->health_0) > 0.0f)
        {
            bond_term_0(&_S192, j_13, 0U, &ta_3, &tl_3, kernelContext_50);
            if((kernelContext_50->params_0->force_tolerance_0) > 0.0f)
            {
                thread float3 ua_0;
                thread float3 ul_0;
                bond_term_0(&_S192, j_13, 6U, &ua_0, &ul_0, kernelContext_50);
                uint _S194 = j_13 * 2U;
                z_3 = norm2_0((float4(*(kernelContext_50->x_0+_S194)) ).xyz + ua_0, (float4(*(kernelContext_50->x_0+(_S194 + 1U))) ).xyz + ul_0);
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
        float _S195 = norm2_0(ta_3, tl_3);
        uint _S196 = bs_0(j_13, kernelContext_50);
        *(kernelContext_50->scratch_0+_S196) = packed_float4(float4(ta_3, 0.0f)) ;
        *(kernelContext_50->scratch_0+(_S196 + 1U)) = packed_float4(float4(tl_3, 0.0f)) ;
        float _S197 = z_3;
        z_3 = _S195;
        f_2 = _S197;
    }
    else
    {
        z_3 = 0.0f;
        f_2 = 0.0f;
    }
    write_partial_0(tid_13, _S190.row_4, group_9.x, 2U, float4(z_3, f_2, 0.0f, 0.0f), kernelContext_50);
    return;
}

[[kernel]] void w_bond_r(uint3 g_10 [[threadgroup_position_in_grid]], uint3 t_11 [[thread_position_in_threadgroup]], atomic<uint> device* meta_16 [[buffer(12)]], uint device* topo_args_16 [[buffer(13)]], packed_uint4 device* islands_16 [[buffer(1)]], BlastParams_0 constant* params_16 [[buffer(0)]], packed_float4 device* wide_16 [[buffer(11)]], uint device* lists_16 [[buffer(9)]], BlastNode_natural_0 device* nodes_16 [[buffer(2)]], packed_float4 device* accel_16 [[buffer(5)]], packed_float4 device* scratch_16 [[buffer(7)]], uint device* refs_16 [[buffer(3)]], BlastBond_natural_0 device* bonds_16 [[buffer(4)]], packed_float4 device* stiffness_16 [[buffer(10)]], packed_float4 device* x_16 [[buffer(6)]], packed_uint4 device* report_16 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_51;
    (&kernelContext_51)->meta_0 = meta_16;
    (&kernelContext_51)->topo_args_0 = topo_args_16;
    (&kernelContext_51)->islands_0 = islands_16;
    (&kernelContext_51)->params_0 = params_16;
    (&kernelContext_51)->wide_0 = wide_16;
    (&kernelContext_51)->lists_0 = lists_16;
    (&kernelContext_51)->nodes_0 = nodes_16;
    (&kernelContext_51)->accel_0 = accel_16;
    (&kernelContext_51)->scratch_0 = scratch_16;
    (&kernelContext_51)->refs_0 = refs_16;
    (&kernelContext_51)->bonds_0 = bonds_16;
    (&kernelContext_51)->stiffness_0 = stiffness_16;
    (&kernelContext_51)->x_0 = x_16;
    (&kernelContext_51)->report_0 = report_16;
    threadgroup array<float2, int(32)> g_blast_part_16;
    (&kernelContext_51)->g_blast_part_0 = &g_blast_part_16;
    threadgroup array<float, int(1024)> g_wide_max_16;
    (&kernelContext_51)->g_wide_max_0 = &g_wide_max_16;
    threadgroup float g_wide_max_out_16;
    (&kernelContext_51)->g_wide_max_out_0 = &g_wide_max_out_16;
    bond_r_0(g_10, t_11.x, false, &kernelContext_51);
    return;
}

void node_rn_0(uint3 group_10, uint tid_14, bool only_rebuild_5, KernelContext_0 thread* kernelContext_52)
{
    Block_0 _S198 = node_block_0(group_10, tid_14, kernelContext_52);
    bool _S199;
    if((_S198.begin_0) >= (_S198.end_0))
    {
        _S199 = true;
    }
    else
    {
        _S199 = !runs_0(_S198.flags_0, only_rebuild_5);
    }
    if(_S199)
    {
        return;
    }
    uint p_6 = _S198.begin_0 + tid_14;
    float s_12;
    if(p_6 < (_S198.end_0))
    {
        uint i_4 = kernelContext_52->lists_0[p_6];
        float3 _S200 = va_0(i_4, 0U, kernelContext_52);
        float3 _S201 = vl_0(i_4, 0U, kernelContext_52);
        s_12 = norm2_0(_S200, _S201);
    }
    else
    {
        s_12 = 0.0f;
    }
    write_partial_0(tid_14, _S198.row_4, group_10.x, 3U, float4(s_12, 0.0f, 0.0f, 0.0f), kernelContext_52);
    return;
}

[[kernel]] void w_node_rn(uint3 g_11 [[threadgroup_position_in_grid]], uint3 t_12 [[thread_position_in_threadgroup]], atomic<uint> device* meta_17 [[buffer(12)]], uint device* topo_args_17 [[buffer(13)]], packed_uint4 device* islands_17 [[buffer(1)]], BlastParams_0 constant* params_17 [[buffer(0)]], packed_float4 device* wide_17 [[buffer(11)]], uint device* lists_17 [[buffer(9)]], BlastNode_natural_0 device* nodes_17 [[buffer(2)]], packed_float4 device* accel_17 [[buffer(5)]], packed_float4 device* scratch_17 [[buffer(7)]], uint device* refs_17 [[buffer(3)]], BlastBond_natural_0 device* bonds_17 [[buffer(4)]], packed_float4 device* stiffness_17 [[buffer(10)]], packed_float4 device* x_17 [[buffer(6)]], packed_uint4 device* report_17 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_53;
    (&kernelContext_53)->meta_0 = meta_17;
    (&kernelContext_53)->topo_args_0 = topo_args_17;
    (&kernelContext_53)->islands_0 = islands_17;
    (&kernelContext_53)->params_0 = params_17;
    (&kernelContext_53)->wide_0 = wide_17;
    (&kernelContext_53)->lists_0 = lists_17;
    (&kernelContext_53)->nodes_0 = nodes_17;
    (&kernelContext_53)->accel_0 = accel_17;
    (&kernelContext_53)->scratch_0 = scratch_17;
    (&kernelContext_53)->refs_0 = refs_17;
    (&kernelContext_53)->bonds_0 = bonds_17;
    (&kernelContext_53)->stiffness_0 = stiffness_17;
    (&kernelContext_53)->x_0 = x_17;
    (&kernelContext_53)->report_0 = report_17;
    threadgroup array<float2, int(32)> g_blast_part_17;
    (&kernelContext_53)->g_blast_part_0 = &g_blast_part_17;
    threadgroup array<float, int(1024)> g_wide_max_17;
    (&kernelContext_53)->g_wide_max_0 = &g_wide_max_17;
    threadgroup float g_wide_max_out_17;
    (&kernelContext_53)->g_wide_max_out_0 = &g_wide_max_out_17;
    node_rn_0(g_11, t_12.x, false, &kernelContext_53);
    return;
}

[[kernel]] void w_bond_rb(uint3 g_12 [[threadgroup_position_in_grid]], uint3 t_13 [[thread_position_in_threadgroup]], atomic<uint> device* meta_18 [[buffer(12)]], uint device* topo_args_18 [[buffer(13)]], packed_uint4 device* islands_18 [[buffer(1)]], BlastParams_0 constant* params_18 [[buffer(0)]], packed_float4 device* wide_18 [[buffer(11)]], uint device* lists_18 [[buffer(9)]], BlastNode_natural_0 device* nodes_18 [[buffer(2)]], packed_float4 device* accel_18 [[buffer(5)]], packed_float4 device* scratch_18 [[buffer(7)]], uint device* refs_18 [[buffer(3)]], BlastBond_natural_0 device* bonds_18 [[buffer(4)]], packed_float4 device* stiffness_18 [[buffer(10)]], packed_float4 device* x_18 [[buffer(6)]], packed_uint4 device* report_18 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_54;
    (&kernelContext_54)->meta_0 = meta_18;
    (&kernelContext_54)->topo_args_0 = topo_args_18;
    (&kernelContext_54)->islands_0 = islands_18;
    (&kernelContext_54)->params_0 = params_18;
    (&kernelContext_54)->wide_0 = wide_18;
    (&kernelContext_54)->lists_0 = lists_18;
    (&kernelContext_54)->nodes_0 = nodes_18;
    (&kernelContext_54)->accel_0 = accel_18;
    (&kernelContext_54)->scratch_0 = scratch_18;
    (&kernelContext_54)->refs_0 = refs_18;
    (&kernelContext_54)->bonds_0 = bonds_18;
    (&kernelContext_54)->stiffness_0 = stiffness_18;
    (&kernelContext_54)->x_0 = x_18;
    (&kernelContext_54)->report_0 = report_18;
    threadgroup array<float2, int(32)> g_blast_part_18;
    (&kernelContext_54)->g_blast_part_0 = &g_blast_part_18;
    threadgroup array<float, int(1024)> g_wide_max_18;
    (&kernelContext_54)->g_wide_max_0 = &g_wide_max_18;
    threadgroup float g_wide_max_out_18;
    (&kernelContext_54)->g_wide_max_out_0 = &g_wide_max_out_18;
    bond_r_0(g_12, t_13.x, true, &kernelContext_54);
    return;
}

[[kernel]] void w_node_rnb(uint3 g_13 [[threadgroup_position_in_grid]], uint3 t_14 [[thread_position_in_threadgroup]], atomic<uint> device* meta_19 [[buffer(12)]], uint device* topo_args_19 [[buffer(13)]], packed_uint4 device* islands_19 [[buffer(1)]], BlastParams_0 constant* params_19 [[buffer(0)]], packed_float4 device* wide_19 [[buffer(11)]], uint device* lists_19 [[buffer(9)]], BlastNode_natural_0 device* nodes_19 [[buffer(2)]], packed_float4 device* accel_19 [[buffer(5)]], packed_float4 device* scratch_19 [[buffer(7)]], uint device* refs_19 [[buffer(3)]], BlastBond_natural_0 device* bonds_19 [[buffer(4)]], packed_float4 device* stiffness_19 [[buffer(10)]], packed_float4 device* x_19 [[buffer(6)]], packed_uint4 device* report_19 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_55;
    (&kernelContext_55)->meta_0 = meta_19;
    (&kernelContext_55)->topo_args_0 = topo_args_19;
    (&kernelContext_55)->islands_0 = islands_19;
    (&kernelContext_55)->params_0 = params_19;
    (&kernelContext_55)->wide_0 = wide_19;
    (&kernelContext_55)->lists_0 = lists_19;
    (&kernelContext_55)->nodes_0 = nodes_19;
    (&kernelContext_55)->accel_0 = accel_19;
    (&kernelContext_55)->scratch_0 = scratch_19;
    (&kernelContext_55)->refs_0 = refs_19;
    (&kernelContext_55)->bonds_0 = bonds_19;
    (&kernelContext_55)->stiffness_0 = stiffness_19;
    (&kernelContext_55)->x_0 = x_19;
    (&kernelContext_55)->report_0 = report_19;
    threadgroup array<float2, int(32)> g_blast_part_19;
    (&kernelContext_55)->g_blast_part_0 = &g_blast_part_19;
    threadgroup array<float, int(1024)> g_wide_max_19;
    (&kernelContext_55)->g_wide_max_0 = &g_wide_max_19;
    threadgroup float g_wide_max_out_19;
    (&kernelContext_55)->g_wide_max_out_0 = &g_wide_max_out_19;
    node_rn_0(g_13, t_14.x, true, &kernelContext_55);
    return;
}

uint iteration_of_0(uint row_13, KernelContext_0 thread* kernelContext_56)
{
    return (as_type<uint>(((float4(*(kernelContext_56->wide_0+st_0(row_13, 2U))) ).z)));
}

void set_flags_0(uint row_14, uint f_3, KernelContext_0 thread* kernelContext_57)
{
    uint _S202 = st_0(row_14, 2U);
    thread float4 c_3 = float4(*(kernelContext_57->wide_0+_S202)) ;
    c_3.w = (as_type<float>((f_3)));
    *(kernelContext_57->wide_0+_S202) = packed_float4(c_3) ;
    return;
}

[[kernel]] void w_test(uint3 group_11 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], atomic<uint> device* meta_20 [[buffer(12)]], uint device* topo_args_20 [[buffer(13)]], packed_uint4 device* islands_20 [[buffer(1)]], BlastParams_0 constant* params_20 [[buffer(0)]], packed_float4 device* wide_20 [[buffer(11)]], uint device* lists_20 [[buffer(9)]], BlastNode_natural_0 device* nodes_20 [[buffer(2)]], packed_float4 device* accel_20 [[buffer(5)]], packed_float4 device* scratch_20 [[buffer(7)]], uint device* refs_20 [[buffer(3)]], BlastBond_natural_0 device* bonds_20 [[buffer(4)]], packed_float4 device* stiffness_20 [[buffer(10)]], packed_float4 device* x_20 [[buffer(6)]], packed_uint4 device* report_20 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_58;
    (&kernelContext_58)->meta_0 = meta_20;
    (&kernelContext_58)->topo_args_0 = topo_args_20;
    (&kernelContext_58)->islands_0 = islands_20;
    (&kernelContext_58)->params_0 = params_20;
    (&kernelContext_58)->wide_0 = wide_20;
    (&kernelContext_58)->lists_0 = lists_20;
    (&kernelContext_58)->nodes_0 = nodes_20;
    (&kernelContext_58)->accel_0 = accel_20;
    (&kernelContext_58)->scratch_0 = scratch_20;
    (&kernelContext_58)->refs_0 = refs_20;
    (&kernelContext_58)->bonds_0 = bonds_20;
    (&kernelContext_58)->stiffness_0 = stiffness_20;
    (&kernelContext_58)->x_0 = x_20;
    (&kernelContext_58)->report_0 = report_20;
    threadgroup array<float2, int(32)> g_blast_part_20;
    (&kernelContext_58)->g_blast_part_0 = &g_blast_part_20;
    threadgroup array<float, int(1024)> g_wide_max_20;
    (&kernelContext_58)->g_wide_max_0 = &g_wide_max_20;
    threadgroup float g_wide_max_out_20;
    (&kernelContext_58)->g_wide_max_out_0 = &g_wide_max_out_20;
    uint tid_15 = thread_6.x;
    uint row_15 = group_11.x;
    uint _S203 = flags_of_0(row_15, &kernelContext_58);
    uint f_4 = group_uniform_0(_S203, tid_15);
    if((f_4 & 1U) == 0U)
    {
        return;
    }
    uint4 _S204 = row_range_0(tid_15, row_15, &kernelContext_58);
    float4 _S205 = row_sum_0(tid_15, row_15, blocks_0(_S204.w - _S204.z), 2U, &kernelContext_58);
    float4 _S206 = row_sum_0(tid_15, row_15, blocks_0(_S204.y - _S204.x), 3U, &kernelContext_58);
    if(tid_15 != 0U)
    {
        return;
    }
    uint _S207 = st_0(row_15, 0U);
    float4 _S208 = float4(*((&kernelContext_58)->wide_0+_S207)) ;
    thread float4 s0_0 = _S208;
    float delta2_1 = _S208.x;
    float _S209 = _S205.x;
    bool _S210 = _S209 <= delta2_1;
    bool hold_0;
    if(_S210)
    {
        hold_0 = !((_S206.x) <= delta2_1);
    }
    else
    {
        hold_0 = false;
    }
    uint _S211 = iteration_of_0(row_15, &kernelContext_58);
    uint f_5 = f_4 & 4294967199U;
    uint f_6;
    if(hold_0)
    {
        f_6 = f_5 | 32U;
    }
    else
    {
        f_6 = f_5;
    }
    bool _S212;
    if(_S211 > 0U)
    {
        _S212 = true;
    }
    else
    {
        _S212 = ((&kernelContext_58)->params_0->warm_0) != 0U;
    }
    if(_S212)
    {
        _S212 = _S210;
    }
    else
    {
        _S212 = false;
    }
    if(_S212)
    {
        hold_0 = !hold_0;
    }
    else
    {
        hold_0 = false;
    }
    if(hold_0)
    {
        f_6 = f_6 | 64U;
    }
    s0_0.y = _S209;
    *((&kernelContext_58)->wide_0+_S207) = packed_float4(s0_0) ;
    uint _S213 = st_0(row_15, 1U);
    *((&kernelContext_58)->wide_0+_S213) = packed_float4(float4(*((&kernelContext_58)->wide_0+_S213)) ) ;
    set_flags_0(row_15, f_6, &kernelContext_58);
    uint _S214 = st_0(row_15, 2U);
    float4 _S215 = float4(*((&kernelContext_58)->wide_0+_S214)) ;
    uint _S216 = st_0(row_15, 3U);
    *((&kernelContext_58)->wide_0+_S216) = packed_float4(float4((float4(*((&kernelContext_58)->wide_0+_S216)) ).xyz, (float4(*((&kernelContext_58)->wide_0+_S216)) ).w)) ;
    uint _S217 = pt_0(row_15, 0U, 5U, &kernelContext_58);
    *((&kernelContext_58)->wide_0+_S217) = packed_float4(float4(_S205.y, 0.0f, 0.0f, 0.0f)) ;
    *((&kernelContext_58)->wide_0+_S214) = packed_float4(_S215) ;
    return;
}

[[kernel]] void w_rebuild(uint3 group_12 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], atomic<uint> device* meta_21 [[buffer(12)]], uint device* topo_args_21 [[buffer(13)]], packed_uint4 device* islands_21 [[buffer(1)]], BlastParams_0 constant* params_21 [[buffer(0)]], packed_float4 device* wide_21 [[buffer(11)]], uint device* lists_21 [[buffer(9)]], BlastNode_natural_0 device* nodes_21 [[buffer(2)]], packed_float4 device* accel_21 [[buffer(5)]], packed_float4 device* scratch_21 [[buffer(7)]], uint device* refs_21 [[buffer(3)]], BlastBond_natural_0 device* bonds_21 [[buffer(4)]], packed_float4 device* stiffness_21 [[buffer(10)]], packed_float4 device* x_21 [[buffer(6)]], packed_uint4 device* report_21 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_59;
    (&kernelContext_59)->meta_0 = meta_21;
    (&kernelContext_59)->topo_args_0 = topo_args_21;
    (&kernelContext_59)->islands_0 = islands_21;
    (&kernelContext_59)->params_0 = params_21;
    (&kernelContext_59)->wide_0 = wide_21;
    (&kernelContext_59)->lists_0 = lists_21;
    (&kernelContext_59)->nodes_0 = nodes_21;
    (&kernelContext_59)->accel_0 = accel_21;
    (&kernelContext_59)->scratch_0 = scratch_21;
    (&kernelContext_59)->refs_0 = refs_21;
    (&kernelContext_59)->bonds_0 = bonds_21;
    (&kernelContext_59)->stiffness_0 = stiffness_21;
    (&kernelContext_59)->x_0 = x_21;
    (&kernelContext_59)->report_0 = report_21;
    threadgroup array<float2, int(32)> g_blast_part_21;
    (&kernelContext_59)->g_blast_part_0 = &g_blast_part_21;
    threadgroup array<float, int(1024)> g_wide_max_21;
    (&kernelContext_59)->g_wide_max_0 = &g_wide_max_21;
    threadgroup float g_wide_max_out_21;
    (&kernelContext_59)->g_wide_max_out_0 = &g_wide_max_out_21;
    uint tid_16 = thread_7.x;
    Block_0 _S218 = node_block_0(group_12, tid_16, &kernelContext_59);
    bool _S219;
    if((_S218.begin_0) >= (_S218.end_0))
    {
        _S219 = true;
    }
    else
    {
        _S219 = !runs_0(_S218.flags_0, true);
    }
    if(_S219)
    {
        return;
    }
    uint p_7 = _S218.begin_0 + tid_16;
    if(p_7 < (_S218.end_0))
    {
        uint i_5 = (&kernelContext_59)->lists_0[p_7];
        thread float3 a_3;
        thread float3 l_3;
        gather_0(i_5, false, true, &a_3, &l_3, &kernelContext_59);
        float3 _S220 = va_0(i_5, 14U, &kernelContext_59);
        float3 _S221 = _S220 - a_3;
        float3 _S222 = vl_0(i_5, 14U, &kernelContext_59);
        set_v_0(i_5, 0U, _S221, _S222 - l_3, &kernelContext_59);
    }
    return;
}

[[kernel]] void w_decide(uint3 group_13 [[threadgroup_position_in_grid]], uint3 thread_8 [[thread_position_in_threadgroup]], atomic<uint> device* meta_22 [[buffer(12)]], uint device* topo_args_22 [[buffer(13)]], packed_uint4 device* islands_22 [[buffer(1)]], BlastParams_0 constant* params_22 [[buffer(0)]], packed_float4 device* wide_22 [[buffer(11)]], uint device* lists_22 [[buffer(9)]], BlastNode_natural_0 device* nodes_22 [[buffer(2)]], packed_float4 device* accel_22 [[buffer(5)]], packed_float4 device* scratch_22 [[buffer(7)]], uint device* refs_22 [[buffer(3)]], BlastBond_natural_0 device* bonds_22 [[buffer(4)]], packed_float4 device* stiffness_22 [[buffer(10)]], packed_float4 device* x_22 [[buffer(6)]], packed_uint4 device* report_22 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_60;
    (&kernelContext_60)->meta_0 = meta_22;
    (&kernelContext_60)->topo_args_0 = topo_args_22;
    (&kernelContext_60)->islands_0 = islands_22;
    (&kernelContext_60)->params_0 = params_22;
    (&kernelContext_60)->wide_0 = wide_22;
    (&kernelContext_60)->lists_0 = lists_22;
    (&kernelContext_60)->nodes_0 = nodes_22;
    (&kernelContext_60)->accel_0 = accel_22;
    (&kernelContext_60)->scratch_0 = scratch_22;
    (&kernelContext_60)->refs_0 = refs_22;
    (&kernelContext_60)->bonds_0 = bonds_22;
    (&kernelContext_60)->stiffness_0 = stiffness_22;
    (&kernelContext_60)->x_0 = x_22;
    (&kernelContext_60)->report_0 = report_22;
    threadgroup array<float2, int(32)> g_blast_part_22;
    (&kernelContext_60)->g_blast_part_0 = &g_blast_part_22;
    threadgroup array<float, int(1024)> g_wide_max_22;
    (&kernelContext_60)->g_wide_max_0 = &g_wide_max_22;
    threadgroup float g_wide_max_out_22;
    (&kernelContext_60)->g_wide_max_out_0 = &g_wide_max_out_22;
    uint tid_17 = thread_8.x;
    uint row_16 = group_13.x;
    uint _S223 = flags_of_0(row_16, &kernelContext_60);
    uint f_7 = group_uniform_0(_S223, tid_17);
    if((f_7 & 1U) == 0U)
    {
        return;
    }
    uint4 _S224 = row_range_0(tid_17, row_16, &kernelContext_60);
    float4 _S225 = float4(0.0f) ;
    bool _S226 = (f_7 & 64U) != 0U;
    float4 z_4;
    float4 rn_0;
    if(_S226)
    {
        float4 _S227 = row_sum_0(tid_17, row_16, blocks_0(_S224.w - _S224.z), 2U, &kernelContext_60);
        float4 _S228 = row_sum_0(tid_17, row_16, blocks_0(_S224.y - _S224.x), 3U, &kernelContext_60);
        z_4 = _S227;
        rn_0 = _S228;
    }
    else
    {
        z_4 = _S225;
        rn_0 = _S225;
    }
    if(tid_17 != 0U)
    {
        return;
    }
    uint _S229 = st_0(row_16, 0U);
    float4 _S230 = float4(*((&kernelContext_60)->wide_0+_S229)) ;
    thread float4 s0_1 = _S230;
    float4 _S231 = float4(*((&kernelContext_60)->wide_0+st_0(row_16, 1U))) ;
    float4 _S232 = float4(*((&kernelContext_60)->wide_0+st_0(row_16, 2U))) ;
    uint _S233 = st_0(row_16, 3U);
    float4 _S234 = float4(*((&kernelContext_60)->wide_0+_S233)) ;
    float delta2_2 = _S230.x;
    float _S235 = _S230.y;
    bool _S236 = (f_7 & 32U) != 0U;
    bool hold_1;
    float num_0;
    if(_S226)
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
        num_0 = _S235;
        hold_1 = _S236;
    }
    uint f_8 = f_7 & 4294967199U;
    uint it_0 = (as_type<uint>((_S232.z)));
    uint best_it_0 = (as_type<uint>((_S234.w)));
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
    bool _S237;
    if(!stop_0)
    {
        _S237 = num_0 <= delta2_2;
    }
    else
    {
        _S237 = false;
    }
    if(_S237)
    {
        hold_1 = !hold_1;
    }
    else
    {
        hold_1 = false;
    }
    uint f_9;
    if(hold_1)
    {
        uint f_10 = f_8 | 2U;
        stop_0 = true;
        f_9 = f_10;
    }
    else
    {
        f_9 = f_8;
    }
    if(!stop_0)
    {
        hold_1 = ((&kernelContext_60)->params_0->force_tolerance_0) > 0.0f;
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
        hold_1 = (_S231.w) <= ((&kernelContext_60)->params_0->force_tolerance_0 * (_S232.x + _S232.y));
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        float _S238 = _S231.w;
        float _S239 = (&kernelContext_60)->params_0->force_tolerance_0;
        uint _S240 = pt_0(row_16, 0U, 5U, &kernelContext_60);
        if(_S238 <= (_S239 * sqrt((float4(*((&kernelContext_60)->wide_0+_S240)) ).x)))
        {
            uint f_11 = f_9 | 2U;
            stop_0 = true;
            f_9 = f_11;
        }
    }
    if(stop_0)
    {
        hold_1 = true;
    }
    else
    {
        hold_1 = it_0 >= ((&kernelContext_60)->params_0->max_iterations_0);
    }
    if(hold_1)
    {
        f_9 = f_9 & 4294967294U;
    }
    s0_1.y = num_0;
    *((&kernelContext_60)->wide_0+_S229) = packed_float4(s0_1) ;
    *((&kernelContext_60)->wide_0+_S233) = packed_float4(float4(_S234.xyz, (as_type<float>((best_it_1))))) ;
    set_flags_0(row_16, f_9, &kernelContext_60);
    return;
}

void full_apply_0(uint n_11, uint at_0, float3 a_4, float3 l_4, float3 thread* oa_0, float3 thread* ol_0, KernelContext_0 thread* kernelContext_61)
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
        uint _S241 = ns_0(n_11, at_0 + q_1, kernelContext_61);
        float4 _S242 = float4(*(kernelContext_61->scratch_0+_S241)) ;
        uint _S243 = 4U * q_1;
        m_3[_S243] = _S242.x;
        m_3[_S243 + 1U] = _S242.y;
        m_3[_S243 + 2U] = _S242.z;
        m_3[_S243 + 3U] = _S242.w;
        q_1 = q_1 + 1U;
    }
    array<float, int(6)> _S244 = { { a_4.x, a_4.y, a_4.z, l_4.x, l_4.y, l_4.z } };
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
        uint c_4 = 0U;
        float s_13 = 0.0f;
        for(;;)
        {
            if(c_4 < 6U)
            {
            }
            else
            {
                break;
            }
            if(r_6 >= c_4)
            {
                q_1 = r_6 * (r_6 + 1U) / 2U + c_4;
            }
            else
            {
                q_1 = c_4 * (c_4 + 1U) / 2U + r_6;
            }
            float s_14 = s_13 + m_3[q_1] * _S244[c_4];
            c_4 = c_4 + 1U;
            s_13 = s_14;
        }
        o_3[r_6] = s_13;
        r_6 = r_6 + 1U;
    }
    *oa_0 = float3(o_3[int(0)], o_3[int(1)], o_3[int(2)]);
    *ol_0 = float3(o_3[int(3)], o_3[int(4)], o_3[int(5)]);
    return;
}

float3 sym_mul_0(float4 s0_2, float4 s1_0, float3 v_16)
{
    float _S245 = v_16.x;
    float _S246 = s0_2.w;
    float _S247 = v_16.y;
    float _S248 = s1_0.x;
    float _S249 = v_16.z;
    float _S250 = s1_0.y;
    return float3(s0_2.x * _S245 + _S246 * _S247 + _S248 * _S249, _S246 * _S245 + s0_2.y * _S247 + _S250 * _S249, _S248 * _S245 + _S250 * _S247 + s0_2.z * _S249);
}

void inverse_apply_0(uint n_12, float3 a_5, float3 l_5, float3 thread* oa_1, float3 thread* ol_1, KernelContext_0 thread* kernelContext_62)
{
    if((kernelContext_62->params_0->rotation_0) == 2U)
    {
        full_apply_0(n_12, 24U, a_5, l_5, oa_1, ol_1, kernelContext_62);
        return;
    }
    uint _S251 = ns_0(n_12, 21U, kernelContext_62);
    float4 _S252 = float4(*(kernelContext_62->scratch_0+(_S251 + 1U))) ;
    float3 kc_0 = (float4(*(kernelContext_62->scratch_0+(_S251 + 2U))) ).xyz;
    float3 _S253 = sym_mul_0(float4(*(kernelContext_62->scratch_0+_S251)) , _S252, a_5 + cross(kc_0, l_5));
    *oa_1 = _S253;
    *ol_1 = l_5 * float3(_S252.z)  - cross(kc_0, _S253);
    return;
}

[[kernel]] void w_pre1(uint3 group_14 [[threadgroup_position_in_grid]], uint3 thread_9 [[thread_position_in_threadgroup]], atomic<uint> device* meta_23 [[buffer(12)]], uint device* topo_args_23 [[buffer(13)]], packed_uint4 device* islands_23 [[buffer(1)]], BlastParams_0 constant* params_23 [[buffer(0)]], packed_float4 device* wide_23 [[buffer(11)]], uint device* lists_23 [[buffer(9)]], BlastNode_natural_0 device* nodes_23 [[buffer(2)]], packed_float4 device* accel_23 [[buffer(5)]], packed_float4 device* scratch_23 [[buffer(7)]], uint device* refs_23 [[buffer(3)]], BlastBond_natural_0 device* bonds_23 [[buffer(4)]], packed_float4 device* stiffness_23 [[buffer(10)]], packed_float4 device* x_23 [[buffer(6)]], packed_uint4 device* report_23 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_63;
    (&kernelContext_63)->meta_0 = meta_23;
    (&kernelContext_63)->topo_args_0 = topo_args_23;
    (&kernelContext_63)->islands_0 = islands_23;
    (&kernelContext_63)->params_0 = params_23;
    (&kernelContext_63)->wide_0 = wide_23;
    (&kernelContext_63)->lists_0 = lists_23;
    (&kernelContext_63)->nodes_0 = nodes_23;
    (&kernelContext_63)->accel_0 = accel_23;
    (&kernelContext_63)->scratch_0 = scratch_23;
    (&kernelContext_63)->refs_0 = refs_23;
    (&kernelContext_63)->bonds_0 = bonds_23;
    (&kernelContext_63)->stiffness_0 = stiffness_23;
    (&kernelContext_63)->x_0 = x_23;
    (&kernelContext_63)->report_0 = report_23;
    threadgroup array<float2, int(32)> g_blast_part_23;
    (&kernelContext_63)->g_blast_part_0 = &g_blast_part_23;
    threadgroup array<float, int(1024)> g_wide_max_23;
    (&kernelContext_63)->g_wide_max_0 = &g_wide_max_23;
    threadgroup float g_wide_max_out_23;
    (&kernelContext_63)->g_wide_max_out_0 = &g_wide_max_out_23;
    uint tid_18 = thread_9.x;
    Block_0 _S254 = node_block_0(group_14, tid_18, &kernelContext_63);
    bool _S255;
    if((_S254.begin_0) >= (_S254.end_0))
    {
        _S255 = true;
    }
    else
    {
        _S255 = !runs_0(_S254.flags_0, false);
    }
    if(_S255)
    {
        return;
    }
    uint p_8 = _S254.begin_0 + tid_18;
    if(p_8 >= (_S254.end_0))
    {
        return;
    }
    uint i_6 = (&kernelContext_63)->lists_0[p_8];
    uint _S256 = iteration_of_0(_S254.row_4, &kernelContext_63);
    if(_S256 == 0U)
    {
        _S255 = ((&kernelContext_63)->params_0->first_polynomial_0) == 0U;
    }
    else
    {
        _S255 = false;
    }
    if(_S255)
    {
        float3 _S257 = va_0(i_6, 0U, &kernelContext_63);
        float3 _S258 = vl_0(i_6, 0U, &kernelContext_63);
        set_v_0(i_6, 8U, _S257, _S258, &kernelContext_63);
        return;
    }
    float3 _S259 = va_0(i_6, 0U, &kernelContext_63);
    float3 _S260 = vl_0(i_6, 0U, &kernelContext_63);
    thread float3 a_6;
    thread float3 l_6;
    inverse_apply_0(i_6, _S259, _S260, &a_6, &l_6, &kernelContext_63);
    set_v_0(i_6, 10U, a_6, l_6, &kernelContext_63);
    return;
}

[[kernel]] void w_bond_loc(uint3 group_15 [[threadgroup_position_in_grid]], uint3 thread_10 [[thread_position_in_threadgroup]], atomic<uint> device* meta_24 [[buffer(12)]], uint device* topo_args_24 [[buffer(13)]], packed_uint4 device* islands_24 [[buffer(1)]], BlastParams_0 constant* params_24 [[buffer(0)]], packed_float4 device* wide_24 [[buffer(11)]], uint device* lists_24 [[buffer(9)]], BlastNode_natural_0 device* nodes_24 [[buffer(2)]], packed_float4 device* accel_24 [[buffer(5)]], packed_float4 device* scratch_24 [[buffer(7)]], uint device* refs_24 [[buffer(3)]], BlastBond_natural_0 device* bonds_24 [[buffer(4)]], packed_float4 device* stiffness_24 [[buffer(10)]], packed_float4 device* x_24 [[buffer(6)]], packed_uint4 device* report_24 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_64;
    (&kernelContext_64)->meta_0 = meta_24;
    (&kernelContext_64)->topo_args_0 = topo_args_24;
    (&kernelContext_64)->islands_0 = islands_24;
    (&kernelContext_64)->params_0 = params_24;
    (&kernelContext_64)->wide_0 = wide_24;
    (&kernelContext_64)->lists_0 = lists_24;
    (&kernelContext_64)->nodes_0 = nodes_24;
    (&kernelContext_64)->accel_0 = accel_24;
    (&kernelContext_64)->scratch_0 = scratch_24;
    (&kernelContext_64)->refs_0 = refs_24;
    (&kernelContext_64)->bonds_0 = bonds_24;
    (&kernelContext_64)->stiffness_0 = stiffness_24;
    (&kernelContext_64)->x_0 = x_24;
    (&kernelContext_64)->report_0 = report_24;
    threadgroup array<float2, int(32)> g_blast_part_24;
    (&kernelContext_64)->g_blast_part_0 = &g_blast_part_24;
    threadgroup array<float, int(1024)> g_wide_max_24;
    (&kernelContext_64)->g_wide_max_0 = &g_wide_max_24;
    threadgroup float g_wide_max_out_24;
    (&kernelContext_64)->g_wide_max_out_0 = &g_wide_max_out_24;
    uint tid_19 = thread_10.x;
    Block_0 _S261 = bond_block_0(group_15, tid_19, &kernelContext_64);
    bool _S262;
    if((_S261.begin_0) >= (_S261.end_0))
    {
        _S262 = true;
    }
    else
    {
        _S262 = !runs_0(_S261.flags_0, false);
    }
    if(_S262)
    {
        _S262 = true;
    }
    else
    {
        uint _S263 = iteration_of_0(_S261.row_4, &kernelContext_64);
        if(_S263 == 0U)
        {
            _S262 = ((&kernelContext_64)->params_0->first_polynomial_0) == 0U;
        }
        else
        {
            _S262 = false;
        }
    }
    if(_S262)
    {
        return;
    }
    uint p_9 = _S261.begin_0 + tid_19;
    if(p_9 >= (_S261.end_0))
    {
        return;
    }
    uint j_14 = (&kernelContext_64)->lists_0[p_9];
    thread BlastBond_natural_0 _S264 = *((&kernelContext_64)->bonds_0+j_14);
    float3 _S265 = float3(0.0f) ;
    thread float3 ta_4 = _S265;
    thread float3 tl_4 = _S265;
    if(((&_S264)->health_0) > 0.0f)
    {
        bond_term_1(&_S264, j_14, 10U, &ta_4, &tl_4, &kernelContext_64);
    }
    uint _S266 = bs_0(j_14, &kernelContext_64);
    *((&kernelContext_64)->scratch_0+_S266) = packed_float4(float4(ta_4, 0.0f)) ;
    *((&kernelContext_64)->scratch_0+(_S266 + 1U)) = packed_float4(float4(tl_4, 0.0f)) ;
    return;
}

void block_apply_0(uint n_13, float3 a_7, float3 l_7, float3 thread* oa_2, float3 thread* ol_2, KernelContext_0 thread* kernelContext_65)
{
    if((kernelContext_65->params_0->rotation_0) == 2U)
    {
        full_apply_0(n_13, 18U, a_7, l_7, oa_2, ol_2, kernelContext_65);
        return;
    }
    uint _S267 = ns_0(n_13, 18U, kernelContext_65);
    float4 _S268 = float4(*(kernelContext_65->scratch_0+(_S267 + 1U))) ;
    float3 k_18 = (float4(*(kernelContext_65->scratch_0+(_S267 + 2U))) ).xyz;
    *oa_2 = sym_mul_0(float4(*(kernelContext_65->scratch_0+_S267)) , _S268, a_7) - cross(k_18, l_7);
    *ol_2 = cross(k_18, a_7) + float3(_S268.z)  * l_7;
    return;
}

[[kernel]] void w_pre2(uint3 group_16 [[threadgroup_position_in_grid]], uint3 thread_11 [[thread_position_in_threadgroup]], atomic<uint> device* meta_25 [[buffer(12)]], uint device* topo_args_25 [[buffer(13)]], packed_uint4 device* islands_25 [[buffer(1)]], BlastParams_0 constant* params_25 [[buffer(0)]], packed_float4 device* wide_25 [[buffer(11)]], uint device* lists_25 [[buffer(9)]], BlastNode_natural_0 device* nodes_25 [[buffer(2)]], packed_float4 device* accel_25 [[buffer(5)]], packed_float4 device* scratch_25 [[buffer(7)]], uint device* refs_25 [[buffer(3)]], BlastBond_natural_0 device* bonds_25 [[buffer(4)]], packed_float4 device* stiffness_25 [[buffer(10)]], packed_float4 device* x_25 [[buffer(6)]], packed_uint4 device* report_25 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_66;
    (&kernelContext_66)->meta_0 = meta_25;
    (&kernelContext_66)->topo_args_0 = topo_args_25;
    (&kernelContext_66)->islands_0 = islands_25;
    (&kernelContext_66)->params_0 = params_25;
    (&kernelContext_66)->wide_0 = wide_25;
    (&kernelContext_66)->lists_0 = lists_25;
    (&kernelContext_66)->nodes_0 = nodes_25;
    (&kernelContext_66)->accel_0 = accel_25;
    (&kernelContext_66)->scratch_0 = scratch_25;
    (&kernelContext_66)->refs_0 = refs_25;
    (&kernelContext_66)->bonds_0 = bonds_25;
    (&kernelContext_66)->stiffness_0 = stiffness_25;
    (&kernelContext_66)->x_0 = x_25;
    (&kernelContext_66)->report_0 = report_25;
    threadgroup array<float2, int(32)> g_blast_part_25;
    (&kernelContext_66)->g_blast_part_0 = &g_blast_part_25;
    threadgroup array<float, int(1024)> g_wide_max_25;
    (&kernelContext_66)->g_wide_max_0 = &g_wide_max_25;
    threadgroup float g_wide_max_out_25;
    (&kernelContext_66)->g_wide_max_out_0 = &g_wide_max_out_25;
    uint tid_20 = thread_11.x;
    Block_0 _S269 = node_block_0(group_16, tid_20, &kernelContext_66);
    bool _S270;
    if((_S269.begin_0) >= (_S269.end_0))
    {
        _S270 = true;
    }
    else
    {
        _S270 = !runs_0(_S269.flags_0, false);
    }
    if(_S270)
    {
        _S270 = true;
    }
    else
    {
        uint _S271 = iteration_of_0(_S269.row_4, &kernelContext_66);
        if(_S271 == 0U)
        {
            _S270 = ((&kernelContext_66)->params_0->first_polynomial_0) == 0U;
        }
        else
        {
            _S270 = false;
        }
    }
    if(_S270)
    {
        return;
    }
    uint p_10 = _S269.begin_0 + tid_20;
    if(p_10 >= (_S269.end_0))
    {
        return;
    }
    uint i_7 = (&kernelContext_66)->lists_0[p_10];
    thread float3 oa_3;
    thread float3 ol_3;
    gather_0(i_7, true, false, &oa_3, &ol_3, &kernelContext_66);
    float3 _S272 = va_0(i_7, 10U, &kernelContext_66);
    float3 _S273 = vl_0(i_7, 10U, &kernelContext_66);
    thread float3 sa_0;
    thread float3 sl_0;
    block_apply_0(i_7, _S272, _S273, &sa_0, &sl_0, &kernelContext_66);
    thread float3 ma_0;
    thread float3 ml_0;
    inverse_apply_0(i_7, oa_3 - sa_0, ol_3 - sl_0, &ma_0, &ml_0, &kernelContext_66);
    float3 _S274 = va_0(i_7, 10U, &kernelContext_66);
    float3 _S275 = float3(1.68946564197540283f) ;
    float3 _S276 = float3(1.5220409631729126f) ;
    float3 _S277 = _S275 * _S274 - _S276 * ma_0;
    float3 _S278 = vl_0(i_7, 10U, &kernelContext_66);
    set_v_0(i_7, 8U, _S277, _S275 * _S278 - _S276 * ml_0, &kernelContext_66);
    return;
}

float wide_group_max_0(uint tid_21, float v_17, KernelContext_0 thread* kernelContext_67)
{
    (*kernelContext_67->g_wide_max_0)[tid_21] = v_17;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint k_19 = 512U;
    for(;;)
    {
        if(k_19 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_21 < k_19)
        {
            (*kernelContext_67->g_wide_max_0)[tid_21] = max((*kernelContext_67->g_wide_max_0)[tid_21], (*kernelContext_67->g_wide_max_0)[tid_21 + k_19]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_19 = k_19 >> 1U;
    }
    if(tid_21 == 0U)
    {
        *kernelContext_67->g_wide_max_out_0 = (*kernelContext_67->g_wide_max_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S279 = kernelContext_67->g_wide_max_out_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *_S279;
}

[[kernel]] void w_max(uint3 group_17 [[threadgroup_position_in_grid]], uint3 thread_12 [[thread_position_in_threadgroup]], atomic<uint> device* meta_26 [[buffer(12)]], uint device* topo_args_26 [[buffer(13)]], packed_uint4 device* islands_26 [[buffer(1)]], BlastParams_0 constant* params_26 [[buffer(0)]], packed_float4 device* wide_26 [[buffer(11)]], uint device* lists_26 [[buffer(9)]], BlastNode_natural_0 device* nodes_26 [[buffer(2)]], packed_float4 device* accel_26 [[buffer(5)]], packed_float4 device* scratch_26 [[buffer(7)]], uint device* refs_26 [[buffer(3)]], BlastBond_natural_0 device* bonds_26 [[buffer(4)]], packed_float4 device* stiffness_26 [[buffer(10)]], packed_float4 device* x_26 [[buffer(6)]], packed_uint4 device* report_26 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_68;
    (&kernelContext_68)->meta_0 = meta_26;
    (&kernelContext_68)->topo_args_0 = topo_args_26;
    (&kernelContext_68)->islands_0 = islands_26;
    (&kernelContext_68)->params_0 = params_26;
    (&kernelContext_68)->wide_0 = wide_26;
    (&kernelContext_68)->lists_0 = lists_26;
    (&kernelContext_68)->nodes_0 = nodes_26;
    (&kernelContext_68)->accel_0 = accel_26;
    (&kernelContext_68)->scratch_0 = scratch_26;
    (&kernelContext_68)->refs_0 = refs_26;
    (&kernelContext_68)->bonds_0 = bonds_26;
    (&kernelContext_68)->stiffness_0 = stiffness_26;
    (&kernelContext_68)->x_0 = x_26;
    (&kernelContext_68)->report_0 = report_26;
    threadgroup array<float2, int(32)> g_blast_part_26;
    (&kernelContext_68)->g_blast_part_0 = &g_blast_part_26;
    threadgroup array<float, int(1024)> g_wide_max_26;
    (&kernelContext_68)->g_wide_max_0 = &g_wide_max_26;
    threadgroup float g_wide_max_out_26;
    (&kernelContext_68)->g_wide_max_out_0 = &g_wide_max_out_26;
    uint tid_22 = thread_12.x;
    Block_0 _S280 = node_block_0(group_17, tid_22, &kernelContext_68);
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
        return;
    }
    uint p_11 = _S280.begin_0 + tid_22;
    float m_4;
    if(p_11 < (_S280.end_0))
    {
        uint i_8 = (&kernelContext_68)->lists_0[p_11];
        float3 _S282 = va_0(i_8, 8U, &kernelContext_68);
        float3 a_8 = abs(_S282);
        float3 _S283 = vl_0(i_8, 8U, &kernelContext_68);
        float3 l_8 = abs(_S283);
        m_4 = max(max(max(a_8.x, a_8.y), max(a_8.z, l_8.x)), max(l_8.y, l_8.z));
    }
    else
    {
        m_4 = 0.0f;
    }
    float _S284 = wide_group_max_0(tid_22, m_4, &kernelContext_68);
    if(tid_22 == 0U)
    {
        uint _S285 = pt_0(_S280.row_4, group_17.x, 4U, &kernelContext_68);
        *((&kernelContext_68)->wide_0+_S285) = packed_float4(float4(_S284, 0.0f, 0.0f, 0.0f)) ;
    }
    return;
}

[[kernel]] void w_mag(uint3 group_18 [[threadgroup_position_in_grid]], uint3 thread_13 [[thread_position_in_threadgroup]], atomic<uint> device* meta_27 [[buffer(12)]], uint device* topo_args_27 [[buffer(13)]], packed_uint4 device* islands_27 [[buffer(1)]], BlastParams_0 constant* params_27 [[buffer(0)]], packed_float4 device* wide_27 [[buffer(11)]], uint device* lists_27 [[buffer(9)]], BlastNode_natural_0 device* nodes_27 [[buffer(2)]], packed_float4 device* accel_27 [[buffer(5)]], packed_float4 device* scratch_27 [[buffer(7)]], uint device* refs_27 [[buffer(3)]], BlastBond_natural_0 device* bonds_27 [[buffer(4)]], packed_float4 device* stiffness_27 [[buffer(10)]], packed_float4 device* x_27 [[buffer(6)]], packed_uint4 device* report_27 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_69;
    (&kernelContext_69)->meta_0 = meta_27;
    (&kernelContext_69)->topo_args_0 = topo_args_27;
    (&kernelContext_69)->islands_0 = islands_27;
    (&kernelContext_69)->params_0 = params_27;
    (&kernelContext_69)->wide_0 = wide_27;
    (&kernelContext_69)->lists_0 = lists_27;
    (&kernelContext_69)->nodes_0 = nodes_27;
    (&kernelContext_69)->accel_0 = accel_27;
    (&kernelContext_69)->scratch_0 = scratch_27;
    (&kernelContext_69)->refs_0 = refs_27;
    (&kernelContext_69)->bonds_0 = bonds_27;
    (&kernelContext_69)->stiffness_0 = stiffness_27;
    (&kernelContext_69)->x_0 = x_27;
    (&kernelContext_69)->report_0 = report_27;
    threadgroup array<float2, int(32)> g_blast_part_27;
    (&kernelContext_69)->g_blast_part_0 = &g_blast_part_27;
    threadgroup array<float, int(1024)> g_wide_max_27;
    (&kernelContext_69)->g_wide_max_0 = &g_wide_max_27;
    threadgroup float g_wide_max_out_27;
    (&kernelContext_69)->g_wide_max_out_0 = &g_wide_max_out_27;
    uint tid_23 = thread_13.x;
    uint row_17 = group_18.x;
    uint _S286 = flags_of_0(row_17, &kernelContext_69);
    uint f_12 = group_uniform_0(_S286, tid_23);
    if((f_12 & 1U) == 0U)
    {
        return;
    }
    uint4 _S287 = row_range_0(tid_23, row_17, &kernelContext_69);
    uint _S288 = blocks_0(_S287.y - _S287.x);
    float m_5 = 0.0f;
    uint blk_0 = tid_23;
    for(;;)
    {
        if(blk_0 < _S288)
        {
        }
        else
        {
            break;
        }
        uint _S289 = pt_0(row_17, blk_0, 4U, &kernelContext_69);
        float _S290 = max(m_5, (float4(*((&kernelContext_69)->wide_0+_S289)) ).x);
        uint blk_1 = blk_0 + 1024U;
        m_5 = _S290;
        blk_0 = blk_1;
    }
    float _S291 = wide_group_max_0(tid_23, m_5, &kernelContext_69);
    if(tid_23 != 0U)
    {
        return;
    }
    bool _S292;
    if(!(_S291 > 0.0f))
    {
        _S292 = true;
    }
    else
    {
        _S292 = !isfinite(_S291);
    }
    if(_S292)
    {
        set_flags_0(row_17, (f_12 & 4294967294U) | 4U, &kernelContext_69);
        return;
    }
    uint _S293 = st_0(row_17, 1U);
    thread float4 s1_1 = float4(*((&kernelContext_69)->wide_0+_S293)) ;
    s1_1.y = 1.0f / _S291;
    *((&kernelContext_69)->wide_0+_S293) = packed_float4(s1_1) ;
    return;
}

[[kernel]] void w_scale(uint3 group_19 [[threadgroup_position_in_grid]], uint3 thread_14 [[thread_position_in_threadgroup]], atomic<uint> device* meta_28 [[buffer(12)]], uint device* topo_args_28 [[buffer(13)]], packed_uint4 device* islands_28 [[buffer(1)]], BlastParams_0 constant* params_28 [[buffer(0)]], packed_float4 device* wide_28 [[buffer(11)]], uint device* lists_28 [[buffer(9)]], BlastNode_natural_0 device* nodes_28 [[buffer(2)]], packed_float4 device* accel_28 [[buffer(5)]], packed_float4 device* scratch_28 [[buffer(7)]], uint device* refs_28 [[buffer(3)]], BlastBond_natural_0 device* bonds_28 [[buffer(4)]], packed_float4 device* stiffness_28 [[buffer(10)]], packed_float4 device* x_28 [[buffer(6)]], packed_uint4 device* report_28 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_70;
    (&kernelContext_70)->meta_0 = meta_28;
    (&kernelContext_70)->topo_args_0 = topo_args_28;
    (&kernelContext_70)->islands_0 = islands_28;
    (&kernelContext_70)->params_0 = params_28;
    (&kernelContext_70)->wide_0 = wide_28;
    (&kernelContext_70)->lists_0 = lists_28;
    (&kernelContext_70)->nodes_0 = nodes_28;
    (&kernelContext_70)->accel_0 = accel_28;
    (&kernelContext_70)->scratch_0 = scratch_28;
    (&kernelContext_70)->refs_0 = refs_28;
    (&kernelContext_70)->bonds_0 = bonds_28;
    (&kernelContext_70)->stiffness_0 = stiffness_28;
    (&kernelContext_70)->x_0 = x_28;
    (&kernelContext_70)->report_0 = report_28;
    threadgroup array<float2, int(32)> g_blast_part_28;
    (&kernelContext_70)->g_blast_part_0 = &g_blast_part_28;
    threadgroup array<float, int(1024)> g_wide_max_28;
    (&kernelContext_70)->g_wide_max_0 = &g_wide_max_28;
    threadgroup float g_wide_max_out_28;
    (&kernelContext_70)->g_wide_max_out_0 = &g_wide_max_out_28;
    uint tid_24 = thread_14.x;
    Block_0 _S294 = node_block_0(group_19, tid_24, &kernelContext_70);
    bool _S295;
    if((_S294.begin_0) >= (_S294.end_0))
    {
        _S295 = true;
    }
    else
    {
        _S295 = !runs_0(_S294.flags_0, false);
    }
    if(_S295)
    {
        return;
    }
    uint p_12 = _S294.begin_0 + tid_24;
    float gl_0;
    if(p_12 < (_S294.end_0))
    {
        uint i_9 = (&kernelContext_70)->lists_0[p_12];
        float inv_1 = (float4(*((&kernelContext_70)->wide_0+st_0(_S294.row_4, 1U))) ).y;
        float3 _S296 = va_0(i_9, 8U, &kernelContext_70);
        float3 _S297 = float3(inv_1) ;
        float3 ga_0 = _S296 * _S297;
        float3 _S298 = vl_0(i_9, 8U, &kernelContext_70);
        float3 gn_0 = _S298 * _S297;
        set_v_0(i_9, 8U, ga_0, gn_0, &kernelContext_70);
        float3 _S299 = va_0(i_9, 0U, &kernelContext_70);
        float _S300 = dot(_S299, ga_0);
        float3 _S301 = vl_0(i_9, 0U, &kernelContext_70);
        gl_0 = _S300 + dot(_S301, gn_0);
    }
    else
    {
        gl_0 = 0.0f;
    }
    write_partial_0(tid_24, _S294.row_4, group_19.x, 0U, float4(gl_0, 0.0f, 0.0f, 0.0f), &kernelContext_70);
    return;
}

[[kernel]] void w_gamma(uint3 group_20 [[threadgroup_position_in_grid]], uint3 thread_15 [[thread_position_in_threadgroup]], atomic<uint> device* meta_29 [[buffer(12)]], uint device* topo_args_29 [[buffer(13)]], packed_uint4 device* islands_29 [[buffer(1)]], BlastParams_0 constant* params_29 [[buffer(0)]], packed_float4 device* wide_29 [[buffer(11)]], uint device* lists_29 [[buffer(9)]], BlastNode_natural_0 device* nodes_29 [[buffer(2)]], packed_float4 device* accel_29 [[buffer(5)]], packed_float4 device* scratch_29 [[buffer(7)]], uint device* refs_29 [[buffer(3)]], BlastBond_natural_0 device* bonds_29 [[buffer(4)]], packed_float4 device* stiffness_29 [[buffer(10)]], packed_float4 device* x_29 [[buffer(6)]], packed_uint4 device* report_29 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_71;
    (&kernelContext_71)->meta_0 = meta_29;
    (&kernelContext_71)->topo_args_0 = topo_args_29;
    (&kernelContext_71)->islands_0 = islands_29;
    (&kernelContext_71)->params_0 = params_29;
    (&kernelContext_71)->wide_0 = wide_29;
    (&kernelContext_71)->lists_0 = lists_29;
    (&kernelContext_71)->nodes_0 = nodes_29;
    (&kernelContext_71)->accel_0 = accel_29;
    (&kernelContext_71)->scratch_0 = scratch_29;
    (&kernelContext_71)->refs_0 = refs_29;
    (&kernelContext_71)->bonds_0 = bonds_29;
    (&kernelContext_71)->stiffness_0 = stiffness_29;
    (&kernelContext_71)->x_0 = x_29;
    (&kernelContext_71)->report_0 = report_29;
    threadgroup array<float2, int(32)> g_blast_part_29;
    (&kernelContext_71)->g_blast_part_0 = &g_blast_part_29;
    threadgroup array<float, int(1024)> g_wide_max_29;
    (&kernelContext_71)->g_wide_max_0 = &g_wide_max_29;
    threadgroup float g_wide_max_out_29;
    (&kernelContext_71)->g_wide_max_out_0 = &g_wide_max_out_29;
    uint tid_25 = thread_15.x;
    uint row_18 = group_20.x;
    uint _S302 = flags_of_0(row_18, &kernelContext_71);
    uint f_13 = group_uniform_0(_S302, tid_25);
    if((f_13 & 1U) == 0U)
    {
        return;
    }
    uint4 _S303 = row_range_0(tid_25, row_18, &kernelContext_71);
    float4 _S304 = row_sum_0(tid_25, row_18, blocks_0(_S303.y - _S303.x), 0U, &kernelContext_71);
    float gamma_0 = _S304.x;
    if(tid_25 != 0U)
    {
        return;
    }
    bool _S305;
    if(!(gamma_0 > 0.0f))
    {
        _S305 = true;
    }
    else
    {
        _S305 = !isfinite(gamma_0);
    }
    if(_S305)
    {
        set_flags_0(row_18, (f_13 & 4294967294U) | 4U, &kernelContext_71);
        return;
    }
    float4 _S306 = float4(*((&kernelContext_71)->wide_0+st_0(row_18, 0U))) ;
    uint _S307 = iteration_of_0(row_18, &kernelContext_71);
    if(_S307 > 1U)
    {
        _S305 = true;
    }
    else
    {
        if(((&kernelContext_71)->params_0->first_polynomial_0) != 0U)
        {
            _S305 = _S307 > 0U;
        }
        else
        {
            _S305 = false;
        }
    }
    if(_S305)
    {
        _S305 = (_S306.z) > 0.0f;
    }
    else
    {
        _S305 = false;
    }
    float beta_0;
    if(_S305)
    {
        beta_0 = gamma_0 / _S306.z;
    }
    else
    {
        beta_0 = 0.0f;
    }
    uint _S308 = st_0(row_18, 1U);
    thread float4 s1_2 = float4(*((&kernelContext_71)->wide_0+_S308)) ;
    s1_2.x = gamma_0;
    s1_2.z = beta_0;
    *((&kernelContext_71)->wide_0+_S308) = packed_float4(s1_2) ;
    return;
}

[[kernel]] void w_dir(uint3 group_21 [[threadgroup_position_in_grid]], uint3 thread_16 [[thread_position_in_threadgroup]], atomic<uint> device* meta_30 [[buffer(12)]], uint device* topo_args_30 [[buffer(13)]], packed_uint4 device* islands_30 [[buffer(1)]], BlastParams_0 constant* params_30 [[buffer(0)]], packed_float4 device* wide_30 [[buffer(11)]], uint device* lists_30 [[buffer(9)]], BlastNode_natural_0 device* nodes_30 [[buffer(2)]], packed_float4 device* accel_30 [[buffer(5)]], packed_float4 device* scratch_30 [[buffer(7)]], uint device* refs_30 [[buffer(3)]], BlastBond_natural_0 device* bonds_30 [[buffer(4)]], packed_float4 device* stiffness_30 [[buffer(10)]], packed_float4 device* x_30 [[buffer(6)]], packed_uint4 device* report_30 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_72;
    (&kernelContext_72)->meta_0 = meta_30;
    (&kernelContext_72)->topo_args_0 = topo_args_30;
    (&kernelContext_72)->islands_0 = islands_30;
    (&kernelContext_72)->params_0 = params_30;
    (&kernelContext_72)->wide_0 = wide_30;
    (&kernelContext_72)->lists_0 = lists_30;
    (&kernelContext_72)->nodes_0 = nodes_30;
    (&kernelContext_72)->accel_0 = accel_30;
    (&kernelContext_72)->scratch_0 = scratch_30;
    (&kernelContext_72)->refs_0 = refs_30;
    (&kernelContext_72)->bonds_0 = bonds_30;
    (&kernelContext_72)->stiffness_0 = stiffness_30;
    (&kernelContext_72)->x_0 = x_30;
    (&kernelContext_72)->report_0 = report_30;
    threadgroup array<float2, int(32)> g_blast_part_30;
    (&kernelContext_72)->g_blast_part_0 = &g_blast_part_30;
    threadgroup array<float, int(1024)> g_wide_max_30;
    (&kernelContext_72)->g_wide_max_0 = &g_wide_max_30;
    threadgroup float g_wide_max_out_30;
    (&kernelContext_72)->g_wide_max_out_0 = &g_wide_max_out_30;
    uint tid_26 = thread_16.x;
    Block_0 _S309 = node_block_0(group_21, tid_26, &kernelContext_72);
    bool _S310;
    if((_S309.begin_0) >= (_S309.end_0))
    {
        _S310 = true;
    }
    else
    {
        _S310 = !runs_0(_S309.flags_0, false);
    }
    if(_S310)
    {
        return;
    }
    uint p_13 = _S309.begin_0 + tid_26;
    if(p_13 >= (_S309.end_0))
    {
        return;
    }
    uint i_10 = (&kernelContext_72)->lists_0[p_13];
    float beta_1 = (float4(*((&kernelContext_72)->wide_0+st_0(_S309.row_4, 1U))) ).z;
    if(beta_1 == 0.0f)
    {
        float3 _S311 = va_0(i_10, 8U, &kernelContext_72);
        float3 _S312 = vl_0(i_10, 8U, &kernelContext_72);
        set_v_0(i_10, 2U, _S311, _S312, &kernelContext_72);
    }
    else
    {
        float3 _S313 = va_0(i_10, 8U, &kernelContext_72);
        float3 _S314 = va_0(i_10, 2U, &kernelContext_72);
        float3 _S315 = float3(beta_1) ;
        float3 _S316 = _S313 + _S314 * _S315;
        float3 _S317 = vl_0(i_10, 8U, &kernelContext_72);
        float3 _S318 = vl_0(i_10, 2U, &kernelContext_72);
        set_v_0(i_10, 2U, _S316, _S317 + _S318 * _S315, &kernelContext_72);
    }
    return;
}

[[kernel]] void w_bond_p(uint3 group_22 [[threadgroup_position_in_grid]], uint3 thread_17 [[thread_position_in_threadgroup]], atomic<uint> device* meta_31 [[buffer(12)]], uint device* topo_args_31 [[buffer(13)]], packed_uint4 device* islands_31 [[buffer(1)]], BlastParams_0 constant* params_31 [[buffer(0)]], packed_float4 device* wide_31 [[buffer(11)]], uint device* lists_31 [[buffer(9)]], BlastNode_natural_0 device* nodes_31 [[buffer(2)]], packed_float4 device* accel_31 [[buffer(5)]], packed_float4 device* scratch_31 [[buffer(7)]], uint device* refs_31 [[buffer(3)]], BlastBond_natural_0 device* bonds_31 [[buffer(4)]], packed_float4 device* stiffness_31 [[buffer(10)]], packed_float4 device* x_31 [[buffer(6)]], packed_uint4 device* report_31 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_73;
    (&kernelContext_73)->meta_0 = meta_31;
    (&kernelContext_73)->topo_args_0 = topo_args_31;
    (&kernelContext_73)->islands_0 = islands_31;
    (&kernelContext_73)->params_0 = params_31;
    (&kernelContext_73)->wide_0 = wide_31;
    (&kernelContext_73)->lists_0 = lists_31;
    (&kernelContext_73)->nodes_0 = nodes_31;
    (&kernelContext_73)->accel_0 = accel_31;
    (&kernelContext_73)->scratch_0 = scratch_31;
    (&kernelContext_73)->refs_0 = refs_31;
    (&kernelContext_73)->bonds_0 = bonds_31;
    (&kernelContext_73)->stiffness_0 = stiffness_31;
    (&kernelContext_73)->x_0 = x_31;
    (&kernelContext_73)->report_0 = report_31;
    threadgroup array<float2, int(32)> g_blast_part_31;
    (&kernelContext_73)->g_blast_part_0 = &g_blast_part_31;
    threadgroup array<float, int(1024)> g_wide_max_31;
    (&kernelContext_73)->g_wide_max_0 = &g_wide_max_31;
    threadgroup float g_wide_max_out_31;
    (&kernelContext_73)->g_wide_max_out_0 = &g_wide_max_out_31;
    uint tid_27 = thread_17.x;
    Block_0 _S319 = bond_block_0(group_22, tid_27, &kernelContext_73);
    bool _S320;
    if((_S319.begin_0) >= (_S319.end_0))
    {
        _S320 = true;
    }
    else
    {
        _S320 = !runs_0(_S319.flags_0, false);
    }
    if(_S320)
    {
        return;
    }
    uint p_14 = _S319.begin_0 + tid_27;
    float z_5;
    if(p_14 < (_S319.end_0))
    {
        uint j_15 = (&kernelContext_73)->lists_0[p_14];
        thread BlastBond_natural_0 _S321 = *((&kernelContext_73)->bonds_0+j_15);
        float3 _S322 = float3(0.0f) ;
        thread float3 ta_5 = _S322;
        thread float3 tl_5 = _S322;
        if(((&_S321)->health_0) > 0.0f)
        {
            bond_term_1(&_S321, j_15, 2U, &ta_5, &tl_5, &kernelContext_73);
        }
        float _S323 = norm2_0(ta_5, tl_5);
        uint _S324 = bs_0(j_15, &kernelContext_73);
        *((&kernelContext_73)->scratch_0+_S324) = packed_float4(float4(ta_5, 0.0f)) ;
        *((&kernelContext_73)->scratch_0+(_S324 + 1U)) = packed_float4(float4(tl_5, 0.0f)) ;
        z_5 = _S323;
    }
    else
    {
        z_5 = 0.0f;
    }
    write_partial_0(tid_27, _S319.row_4, group_22.x, 1U, float4(z_5, 0.0f, 0.0f, 0.0f), &kernelContext_73);
    return;
}

[[kernel]] void w_node_q(uint3 group_23 [[threadgroup_position_in_grid]], uint3 thread_18 [[thread_position_in_threadgroup]], atomic<uint> device* meta_32 [[buffer(12)]], uint device* topo_args_32 [[buffer(13)]], packed_uint4 device* islands_32 [[buffer(1)]], BlastParams_0 constant* params_32 [[buffer(0)]], packed_float4 device* wide_32 [[buffer(11)]], uint device* lists_32 [[buffer(9)]], BlastNode_natural_0 device* nodes_32 [[buffer(2)]], packed_float4 device* accel_32 [[buffer(5)]], packed_float4 device* scratch_32 [[buffer(7)]], uint device* refs_32 [[buffer(3)]], BlastBond_natural_0 device* bonds_32 [[buffer(4)]], packed_float4 device* stiffness_32 [[buffer(10)]], packed_float4 device* x_32 [[buffer(6)]], packed_uint4 device* report_32 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_74;
    (&kernelContext_74)->meta_0 = meta_32;
    (&kernelContext_74)->topo_args_0 = topo_args_32;
    (&kernelContext_74)->islands_0 = islands_32;
    (&kernelContext_74)->params_0 = params_32;
    (&kernelContext_74)->wide_0 = wide_32;
    (&kernelContext_74)->lists_0 = lists_32;
    (&kernelContext_74)->nodes_0 = nodes_32;
    (&kernelContext_74)->accel_0 = accel_32;
    (&kernelContext_74)->scratch_0 = scratch_32;
    (&kernelContext_74)->refs_0 = refs_32;
    (&kernelContext_74)->bonds_0 = bonds_32;
    (&kernelContext_74)->stiffness_0 = stiffness_32;
    (&kernelContext_74)->x_0 = x_32;
    (&kernelContext_74)->report_0 = report_32;
    threadgroup array<float2, int(32)> g_blast_part_32;
    (&kernelContext_74)->g_blast_part_0 = &g_blast_part_32;
    threadgroup array<float, int(1024)> g_wide_max_32;
    (&kernelContext_74)->g_wide_max_0 = &g_wide_max_32;
    threadgroup float g_wide_max_out_32;
    (&kernelContext_74)->g_wide_max_out_0 = &g_wide_max_out_32;
    uint tid_28 = thread_18.x;
    Block_0 _S325 = node_block_0(group_23, tid_28, &kernelContext_74);
    bool _S326;
    if((_S325.begin_0) >= (_S325.end_0))
    {
        _S326 = true;
    }
    else
    {
        _S326 = !runs_0(_S325.flags_0, false);
    }
    if(_S326)
    {
        return;
    }
    uint p_15 = _S325.begin_0 + tid_28;
    if(p_15 >= (_S325.end_0))
    {
        return;
    }
    uint i_11 = (&kernelContext_74)->lists_0[p_15];
    thread float3 a_9;
    thread float3 l_9;
    gather_0(i_11, true, false, &a_9, &l_9, &kernelContext_74);
    set_v_0(i_11, 4U, a_9, l_9, &kernelContext_74);
    return;
}

[[kernel]] void w_den(uint3 group_24 [[threadgroup_position_in_grid]], uint3 thread_19 [[thread_position_in_threadgroup]], atomic<uint> device* meta_33 [[buffer(12)]], uint device* topo_args_33 [[buffer(13)]], packed_uint4 device* islands_33 [[buffer(1)]], BlastParams_0 constant* params_33 [[buffer(0)]], packed_float4 device* wide_33 [[buffer(11)]], uint device* lists_33 [[buffer(9)]], BlastNode_natural_0 device* nodes_33 [[buffer(2)]], packed_float4 device* accel_33 [[buffer(5)]], packed_float4 device* scratch_33 [[buffer(7)]], uint device* refs_33 [[buffer(3)]], BlastBond_natural_0 device* bonds_33 [[buffer(4)]], packed_float4 device* stiffness_33 [[buffer(10)]], packed_float4 device* x_33 [[buffer(6)]], packed_uint4 device* report_33 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_75;
    (&kernelContext_75)->meta_0 = meta_33;
    (&kernelContext_75)->topo_args_0 = topo_args_33;
    (&kernelContext_75)->islands_0 = islands_33;
    (&kernelContext_75)->params_0 = params_33;
    (&kernelContext_75)->wide_0 = wide_33;
    (&kernelContext_75)->lists_0 = lists_33;
    (&kernelContext_75)->nodes_0 = nodes_33;
    (&kernelContext_75)->accel_0 = accel_33;
    (&kernelContext_75)->scratch_0 = scratch_33;
    (&kernelContext_75)->refs_0 = refs_33;
    (&kernelContext_75)->bonds_0 = bonds_33;
    (&kernelContext_75)->stiffness_0 = stiffness_33;
    (&kernelContext_75)->x_0 = x_33;
    (&kernelContext_75)->report_0 = report_33;
    threadgroup array<float2, int(32)> g_blast_part_33;
    (&kernelContext_75)->g_blast_part_0 = &g_blast_part_33;
    threadgroup array<float, int(1024)> g_wide_max_33;
    (&kernelContext_75)->g_wide_max_0 = &g_wide_max_33;
    threadgroup float g_wide_max_out_33;
    (&kernelContext_75)->g_wide_max_out_0 = &g_wide_max_out_33;
    uint tid_29 = thread_19.x;
    uint row_19 = group_24.x;
    uint _S327 = flags_of_0(row_19, &kernelContext_75);
    uint f_14 = group_uniform_0(_S327, tid_29);
    if((f_14 & 1U) == 0U)
    {
        return;
    }
    uint4 _S328 = row_range_0(tid_29, row_19, &kernelContext_75);
    float4 _S329 = row_sum_0(tid_29, row_19, blocks_0(_S328.w - _S328.z), 1U, &kernelContext_75);
    float den_0 = _S329.x;
    if(tid_29 != 0U)
    {
        return;
    }
    uint _S330 = st_0(row_19, 0U);
    thread float4 s0_3 = float4(*((&kernelContext_75)->wide_0+_S330)) ;
    uint _S331 = st_0(row_19, 1U);
    float4 _S332 = float4(*((&kernelContext_75)->wide_0+_S331)) ;
    thread float4 s1_3 = _S332;
    uint _S333 = st_0(row_19, 2U);
    thread float4 cnt_0 = float4(*((&kernelContext_75)->wide_0+_S333)) ;
    float gamma_1 = _S332.x;
    bool _S334;
    if(((&kernelContext_75)->params_0->force_tolerance_0) > 0.0f)
    {
        if(den_0 > 0.0f)
        {
            _S334 = gamma_1 > 0.0f;
        }
        else
        {
            _S334 = false;
        }
        float _S335;
        if(_S334)
        {
            _S335 = gamma_1 / sqrt(den_0);
        }
        else
        {
            _S335 = float_bits_0(2139095040U);
        }
        s1_3.w = _S335;
        if(isfinite(s1_3.w))
        {
            cnt_0.y = cnt_0.y + s1_3.w;
        }
        if((as_type<uint>((cnt_0.z))) > 0U)
        {
            _S334 = true;
        }
        else
        {
            _S334 = ((&kernelContext_75)->params_0->first_polynomial_0) != 0U;
        }
        if(!_S334)
        {
            s1_3.w = float_bits_0(2139095040U);
        }
    }
    s0_3.z = gamma_1;
    cnt_0.z = (as_type<float>(((as_type<uint>((cnt_0.z))) + 1U)));
    if(!(den_0 > 0.0f))
    {
        _S334 = true;
    }
    else
    {
        _S334 = !isfinite(den_0);
    }
    uint f_15;
    if(_S334)
    {
        uint f_16 = f_14 & 4294967294U;
        s1_3.y = 0.0f;
        f_15 = f_16;
    }
    else
    {
        s1_3.y = gamma_1 / den_0;
        f_15 = f_14;
    }
    *((&kernelContext_75)->wide_0+_S330) = packed_float4(s0_3) ;
    *((&kernelContext_75)->wide_0+_S331) = packed_float4(s1_3) ;
    cnt_0.w = (as_type<float>((f_15)));
    *((&kernelContext_75)->wide_0+_S333) = packed_float4(cnt_0) ;
    return;
}

[[kernel]] void w_update(uint3 group_25 [[threadgroup_position_in_grid]], uint3 thread_20 [[thread_position_in_threadgroup]], atomic<uint> device* meta_34 [[buffer(12)]], uint device* topo_args_34 [[buffer(13)]], packed_uint4 device* islands_34 [[buffer(1)]], BlastParams_0 constant* params_34 [[buffer(0)]], packed_float4 device* wide_34 [[buffer(11)]], uint device* lists_34 [[buffer(9)]], BlastNode_natural_0 device* nodes_34 [[buffer(2)]], packed_float4 device* accel_34 [[buffer(5)]], packed_float4 device* scratch_34 [[buffer(7)]], uint device* refs_34 [[buffer(3)]], BlastBond_natural_0 device* bonds_34 [[buffer(4)]], packed_float4 device* stiffness_34 [[buffer(10)]], packed_float4 device* x_34 [[buffer(6)]], packed_uint4 device* report_34 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_76;
    (&kernelContext_76)->meta_0 = meta_34;
    (&kernelContext_76)->topo_args_0 = topo_args_34;
    (&kernelContext_76)->islands_0 = islands_34;
    (&kernelContext_76)->params_0 = params_34;
    (&kernelContext_76)->wide_0 = wide_34;
    (&kernelContext_76)->lists_0 = lists_34;
    (&kernelContext_76)->nodes_0 = nodes_34;
    (&kernelContext_76)->accel_0 = accel_34;
    (&kernelContext_76)->scratch_0 = scratch_34;
    (&kernelContext_76)->refs_0 = refs_34;
    (&kernelContext_76)->bonds_0 = bonds_34;
    (&kernelContext_76)->stiffness_0 = stiffness_34;
    (&kernelContext_76)->x_0 = x_34;
    (&kernelContext_76)->report_0 = report_34;
    threadgroup array<float2, int(32)> g_blast_part_34;
    (&kernelContext_76)->g_blast_part_0 = &g_blast_part_34;
    threadgroup array<float, int(1024)> g_wide_max_34;
    (&kernelContext_76)->g_wide_max_0 = &g_wide_max_34;
    threadgroup float g_wide_max_out_34;
    (&kernelContext_76)->g_wide_max_out_0 = &g_wide_max_out_34;
    uint tid_30 = thread_20.x;
    Block_0 _S336 = node_block_0(group_25, tid_30, &kernelContext_76);
    bool _S337;
    if((_S336.begin_0) >= (_S336.end_0))
    {
        _S337 = true;
    }
    else
    {
        _S337 = !runs_0(_S336.flags_0, false);
    }
    if(_S337)
    {
        return;
    }
    uint p_16 = _S336.begin_0 + tid_30;
    if(p_16 >= (_S336.end_0))
    {
        return;
    }
    uint i_12 = (&kernelContext_76)->lists_0[p_16];
    float alpha_0 = (float4(*((&kernelContext_76)->wide_0+st_0(_S336.row_4, 1U))) ).y;
    float3 _S338 = va_0(i_12, 6U, &kernelContext_76);
    float3 _S339 = va_0(i_12, 2U, &kernelContext_76);
    float3 _S340 = float3(alpha_0) ;
    float3 _S341 = _S338 + _S339 * _S340;
    float3 _S342 = vl_0(i_12, 6U, &kernelContext_76);
    float3 _S343 = vl_0(i_12, 2U, &kernelContext_76);
    set_v_0(i_12, 6U, _S341, _S342 + _S343 * _S340, &kernelContext_76);
    float3 _S344 = va_0(i_12, 0U, &kernelContext_76);
    float3 _S345 = va_0(i_12, 4U, &kernelContext_76);
    float3 _S346 = _S344 - _S345 * _S340;
    float3 _S347 = vl_0(i_12, 0U, &kernelContext_76);
    float3 _S348 = vl_0(i_12, 4U, &kernelContext_76);
    set_v_0(i_12, 0U, _S346, _S347 - _S348 * _S340, &kernelContext_76);
    return;
}

[[kernel]] void w_final(uint3 group_26 [[threadgroup_position_in_grid]], uint3 thread_21 [[thread_position_in_threadgroup]], atomic<uint> device* meta_35 [[buffer(12)]], uint device* topo_args_35 [[buffer(13)]], packed_uint4 device* islands_35 [[buffer(1)]], BlastParams_0 constant* params_35 [[buffer(0)]], packed_float4 device* wide_35 [[buffer(11)]], uint device* lists_35 [[buffer(9)]], BlastNode_natural_0 device* nodes_35 [[buffer(2)]], packed_float4 device* accel_35 [[buffer(5)]], packed_float4 device* scratch_35 [[buffer(7)]], uint device* refs_35 [[buffer(3)]], BlastBond_natural_0 device* bonds_35 [[buffer(4)]], packed_float4 device* stiffness_35 [[buffer(10)]], packed_float4 device* x_35 [[buffer(6)]], packed_uint4 device* report_35 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_77;
    (&kernelContext_77)->meta_0 = meta_35;
    (&kernelContext_77)->topo_args_0 = topo_args_35;
    (&kernelContext_77)->islands_0 = islands_35;
    (&kernelContext_77)->params_0 = params_35;
    (&kernelContext_77)->wide_0 = wide_35;
    (&kernelContext_77)->lists_0 = lists_35;
    (&kernelContext_77)->nodes_0 = nodes_35;
    (&kernelContext_77)->accel_0 = accel_35;
    (&kernelContext_77)->scratch_0 = scratch_35;
    (&kernelContext_77)->refs_0 = refs_35;
    (&kernelContext_77)->bonds_0 = bonds_35;
    (&kernelContext_77)->stiffness_0 = stiffness_35;
    (&kernelContext_77)->x_0 = x_35;
    (&kernelContext_77)->report_0 = report_35;
    threadgroup array<float2, int(32)> g_blast_part_35;
    (&kernelContext_77)->g_blast_part_0 = &g_blast_part_35;
    threadgroup array<float, int(1024)> g_wide_max_35;
    (&kernelContext_77)->g_wide_max_0 = &g_wide_max_35;
    threadgroup float g_wide_max_out_35;
    (&kernelContext_77)->g_wide_max_out_0 = &g_wide_max_out_35;
    uint tid_31 = thread_21.x;
    Block_0 _S349 = bond_block_0(group_26, tid_31, &kernelContext_77);
    bool _S350;
    if((_S349.begin_0) >= (_S349.end_0))
    {
        _S350 = true;
    }
    else
    {
        _S350 = ((_S349.flags_0) & 128U) != 0U;
    }
    if(_S350)
    {
        return;
    }
    uint p_17 = _S349.begin_0 + tid_31;
    if(p_17 >= (_S349.end_0))
    {
        return;
    }
    uint j_16 = (&kernelContext_77)->lists_0[p_17];
    thread BlastBond_natural_0 _S351 = *((&kernelContext_77)->bonds_0+j_16);
    if(!(((&_S351)->health_0) > 0.0f))
    {
        _S350 = true;
    }
    else
    {
        _S350 = ((_S349.flags_0) & 16U) != 0U;
    }
    if(_S350)
    {
        uint _S352 = j_16 * 2U;
        packed_float4 _S353 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_77)->x_0+_S352) = _S353;
        *((&kernelContext_77)->x_0+(_S352 + 1U)) = _S353;
        return;
    }
    thread float3 ta_6;
    thread float3 tl_6;
    bond_term_1(&_S351, j_16, 6U, &ta_6, &tl_6, &kernelContext_77);
    uint _S354 = j_16 * 2U;
    *((&kernelContext_77)->x_0+_S354) = packed_float4(float4((float4(*((&kernelContext_77)->x_0+_S354)) ).xyz + ta_6, 0.0f)) ;
    uint _S355 = _S354 + 1U;
    *((&kernelContext_77)->x_0+_S355) = packed_float4(float4((float4(*((&kernelContext_77)->x_0+_S355)) ).xyz + tl_6, 0.0f)) ;
    return;
}

[[kernel]] void w_report(uint3 group_27 [[threadgroup_position_in_grid]], atomic<uint> device* meta_36 [[buffer(12)]], uint device* topo_args_36 [[buffer(13)]], packed_uint4 device* islands_36 [[buffer(1)]], BlastParams_0 constant* params_36 [[buffer(0)]], packed_float4 device* wide_36 [[buffer(11)]], uint device* lists_36 [[buffer(9)]], BlastNode_natural_0 device* nodes_36 [[buffer(2)]], packed_float4 device* accel_36 [[buffer(5)]], packed_float4 device* scratch_36 [[buffer(7)]], uint device* refs_36 [[buffer(3)]], BlastBond_natural_0 device* bonds_36 [[buffer(4)]], packed_float4 device* stiffness_36 [[buffer(10)]], packed_float4 device* x_36 [[buffer(6)]], packed_uint4 device* report_36 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_78;
    (&kernelContext_78)->meta_0 = meta_36;
    (&kernelContext_78)->topo_args_0 = topo_args_36;
    (&kernelContext_78)->islands_0 = islands_36;
    (&kernelContext_78)->params_0 = params_36;
    (&kernelContext_78)->wide_0 = wide_36;
    (&kernelContext_78)->lists_0 = lists_36;
    (&kernelContext_78)->nodes_0 = nodes_36;
    (&kernelContext_78)->accel_0 = accel_36;
    (&kernelContext_78)->scratch_0 = scratch_36;
    (&kernelContext_78)->refs_0 = refs_36;
    (&kernelContext_78)->bonds_0 = bonds_36;
    (&kernelContext_78)->stiffness_0 = stiffness_36;
    (&kernelContext_78)->x_0 = x_36;
    (&kernelContext_78)->report_0 = report_36;
    threadgroup array<float2, int(32)> g_blast_part_36;
    (&kernelContext_78)->g_blast_part_0 = &g_blast_part_36;
    threadgroup array<float, int(1024)> g_wide_max_36;
    (&kernelContext_78)->g_wide_max_0 = &g_wide_max_36;
    threadgroup float g_wide_max_out_36;
    (&kernelContext_78)->g_wide_max_out_0 = &g_wide_max_out_36;
    uint row_20 = group_27.x;
    uint _S356 = row_island_0(row_20, &kernelContext_78);
    uint _S357 = flags_of_0(row_20, &kernelContext_78);
    uint4 _S358 = uint4(*((&kernelContext_78)->report_0+_S356)) ;
    if((_S357 & 128U) != 0U)
    {
        *((&kernelContext_78)->report_0+_S356) = packed_uint4(uint4(0U, 7U, _S358.z, _S358.w)) ;
        return;
    }
    bool ok_0;
    if((_S357 & 2U) != 0U)
    {
        ok_0 = (_S357 & 4U) == 0U;
    }
    else
    {
        ok_0 = false;
    }
    uint _S359 = iteration_of_0(row_20, &kernelContext_78);
    uint _S360;
    if(ok_0)
    {
        _S360 = 1U;
    }
    else
    {
        _S360 = 0U;
    }
    if(ok_0)
    {
        ok_0 = ((&kernelContext_78)->params_0->warm_0) != 0U;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        ok_0 = _S359 == 0U;
    }
    else
    {
        ok_0 = false;
    }
    uint _S361;
    if(ok_0)
    {
        _S361 = 4U;
    }
    else
    {
        _S361 = 0U;
    }
    float4 _S362 = float4(*((&kernelContext_78)->wide_0+st_0(row_20, 0U))) ;
    *((&kernelContext_78)->report_0+_S356) = packed_uint4(uint4(_S359, _S360 | _S361, (as_type<uint>((_S362.y))), (as_type<uint>((_S362.x))))) ;
    return;
}

