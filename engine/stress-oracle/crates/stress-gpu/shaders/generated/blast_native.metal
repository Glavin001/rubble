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

uint group_uniform_0(uint value_0, uint thread_0)
{
    return value_0;
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
    BlastParams_0 constant* params_0;
    packed_uint4 device* islands_0;
    packed_uint4 device* report_0;
    uint device* lists_0;
    packed_float4 device* accel_0;
    packed_float4 device* scratch_0;
    BlastNode_natural_0 device* nodes_0;
    uint device* refs_0;
    BlastBond_natural_0 device* bonds_0;
    packed_float4 device* stiffness_0;
    packed_float4 device* x_0;
    array<float2, int(32)> threadgroup* g_blast_part_0;
    float3 threadgroup* g_omega_0;
    float3 threadgroup* g_t_0;
    array<float, int(1024)> threadgroup* g_max_0;
    float threadgroup* g_max_out_0;
};

uint ns_0(uint n_0, uint v_0, KernelContext_0 thread* kernelContext_0)
{
    return n_0 * kernelContext_0->params_0->node_vectors_0 + v_0;
}

float2 blast_group_sum_0(uint tid_0, float2 v_1, KernelContext_0 thread* kernelContext_1)
{
    float2 w_0 = simd_sum(v_1);
    if((tid_0 % 32U) == 0U)
    {
        (*kernelContext_1->g_blast_part_0)[tid_0 / 32U] = w_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 _S1 = (*kernelContext_1->g_blast_part_0)[int(0)];
    uint k_0 = 1U;
    float2 s_1 = _S1;
    for(;;)
    {
        if(k_0 < 32U)
        {
        }
        else
        {
            break;
        }
        float2 s_2 = s_1 + (*kernelContext_1->g_blast_part_0)[k_0];
        k_0 = k_0 + 1U;
        s_1 = s_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s_1;
}

uint blast_frame_0(uint node_count_1, uint k_1)
{
    return node_count_1 + 10U * k_1;
}

uint frame_word_0(uint k_2, uint w_1, KernelContext_0 thread* kernelContext_2)
{
    return (*(kernelContext_2->islands_0+(blast_frame_0(kernelContext_2->params_0->node_count_0, k_2) + w_1 / 4U)))[w_1 % 4U];
}

uint frame_word_1(uint k_3, uint w_2, KernelContext_0 thread* kernelContext_3)
{
    return (*(kernelContext_3->islands_0+(blast_frame_0(kernelContext_3->params_0->node_count_0, k_3) + w_2 / 4U)))[w_2 % 4U];
}

uint3 carry_of_0(uint k_4, KernelContext_0 thread* kernelContext_4)
{
    uint _S2 = frame_word_1(k_4, 36U, kernelContext_4);
    uint _S3 = frame_word_1(k_4, 37U, kernelContext_4);
    uint _S4 = frame_word_1(k_4, 38U, kernelContext_4);
    return uint3(_S2, _S3, _S4);
}

uint4 group_uniform4_0(uint4 value_1, uint thread_1)
{
    return value_1;
}

bool carry_kept_0(uint3 c_0, uint count_0, KernelContext_0 thread* kernelContext_5)
{
    float gamma_0 = (as_type<float>((c_0.z)));
    bool _S5;
    if((kernelContext_5->params_0->carry_0) != 0U)
    {
        _S5 = (c_0.x) == count_0;
    }
    else
    {
        _S5 = false;
    }
    if(_S5)
    {
        _S5 = (c_0.y) < (6U * count_0);
    }
    else
    {
        _S5 = false;
    }
    if(_S5)
    {
        _S5 = gamma_0 > 0.0f;
    }
    else
    {
        _S5 = false;
    }
    if(_S5)
    {
        _S5 = isfinite(gamma_0);
    }
    else
    {
        _S5 = false;
    }
    return _S5;
}

void set_carry_0(uint k_5, uint count_1, uint iterations_0, float gamma_1, KernelContext_0 thread* kernelContext_6)
{
    uint at_0 = blast_frame_0(kernelContext_6->params_0->node_count_0, k_5) + 9U;
    thread uint4 v_2 = uint4(*(kernelContext_6->islands_0+at_0)) ;
    v_2.x = count_1;
    v_2.y = iterations_0;
    v_2.z = (as_type<uint>((gamma_1)));
    *(kernelContext_6->islands_0+at_0) = packed_uint4(v_2) ;
    return;
}

float norm2_0(float3 a_0, float3 b_0)
{
    return dot(a_0, a_0) + dot(b_0, b_0);
}

float3 va_0(uint n_1, uint v_3, KernelContext_0 thread* kernelContext_7)
{
    uint _S6 = ns_0(n_1, v_3, kernelContext_7);
    return (float4(*(kernelContext_7->scratch_0+_S6)) ).xyz;
}

float3 vl_0(uint n_2, uint v_4, KernelContext_0 thread* kernelContext_8)
{
    uint _S7 = ns_0(n_2, v_4, kernelContext_8);
    return (float4(*(kernelContext_8->scratch_0+(_S7 + 1U))) ).xyz;
}

void set_v_0(uint n_3, uint v_5, float3 a_1, float3 l_0, KernelContext_0 thread* kernelContext_9)
{
    uint _S8 = ns_0(n_3, v_5, kernelContext_9);
    *(kernelContext_9->scratch_0+_S8) = packed_float4(float4(a_1, 0.0f)) ;
    *(kernelContext_9->scratch_0+(_S8 + 1U)) = packed_float4(float4(l_0, 0.0f)) ;
    return;
}

float3 stiff_0(uint j_0, uint which_0, float3 v_6, KernelContext_0 thread* kernelContext_10)
{
    uint _S9 = 6U * which_0;
    thread array<float, int(6)> m_0;
    uint k_6 = 0U;
    for(;;)
    {
        if(k_6 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S10 = _S9 + k_6;
        m_0[k_6] = (float4(*(kernelContext_10->stiffness_0+(j_0 * 6U + _S10 / 4U))) )[_S10 % 4U];
        k_6 = k_6 + 1U;
    }
    float _S11 = v_6.x;
    float _S12 = v_6.y;
    float _S13 = v_6.z;
    return float3(m_0[int(0)] * _S11 + m_0[int(3)] * _S12 + m_0[int(4)] * _S13, m_0[int(3)] * _S11 + m_0[int(1)] * _S12 + m_0[int(5)] * _S13, m_0[int(4)] * _S11 + m_0[int(5)] * _S12 + m_0[int(2)] * _S13);
}

float3 stiff_1(uint j_1, uint which_1, float3 v_7, KernelContext_0 thread* kernelContext_11)
{
    uint _S14 = 6U * which_1;
    thread array<float, int(6)> m_1;
    uint k_7 = 0U;
    for(;;)
    {
        if(k_7 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S15 = _S14 + k_7;
        m_1[k_7] = (float4(*(kernelContext_11->stiffness_0+(j_1 * 6U + _S15 / 4U))) )[_S15 % 4U];
        k_7 = k_7 + 1U;
    }
    float _S16 = v_7.x;
    float _S17 = v_7.y;
    float _S18 = v_7.z;
    return float3(m_1[int(0)] * _S16 + m_1[int(3)] * _S17 + m_1[int(4)] * _S18, m_1[int(3)] * _S16 + m_1[int(1)] * _S17 + m_1[int(5)] * _S18, m_1[int(4)] * _S16 + m_1[int(5)] * _S17 + m_1[int(2)] * _S18);
}

matrix<float,int(3),int(3)>  sym3_0(uint j_2, uint which_2, KernelContext_0 thread* kernelContext_12)
{
    float3 _S19 = stiff_1(j_2, which_2, float3(1.0f, 0.0f, 0.0f), kernelContext_12);
    float3 _S20 = stiff_1(j_2, which_2, float3(0.0f, 1.0f, 0.0f), kernelContext_12);
    float3 _S21 = stiff_1(j_2, which_2, float3(0.0f, 0.0f, 1.0f), kernelContext_12);
    return matrix<float,int(3),int(3)> (_S19.x, _S20.x, _S21.x, _S19.y, _S20.y, _S21.y, _S19.z, _S20.z, _S21.z);
}

matrix<float,int(3),int(3)>  skew_0(float3 o_0)
{
    float _S22 = o_0.z;
    float _S23 = o_0.y;
    float _S24 = o_0.x;
    return matrix<float,int(3),int(3)> (0.0f, - _S22, _S23, _S22, 0.0f, - _S24, - _S23, _S24, 0.0f);
}

bool build_full_block_0(uint node_0, KernelContext_0 thread* kernelContext_13)
{
    uint c_1;
    uint k_8;
    float s_3;
    BlastNode_natural_0 n_4 = kernelContext_13->nodes_0[node_0];
    matrix<float,int(3),int(3)>  aa_0 = matrix<float,int(3),int(3)> (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    uint r_0 = n_4.ref_begin_0;
    matrix<float,int(3),int(3)>  aa_1 = aa_0;
    matrix<float,int(3),int(3)>  al_0 = aa_0;
    matrix<float,int(3),int(3)>  ll_0 = aa_0;
    for(;;)
    {
        if(r_0 < (n_4.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_0 = kernelContext_13->refs_0[r_0];
        uint j_3 = ref_0 & 2147483647U;
        BlastBond_natural_0 device* _S25 = kernelContext_13->bonds_0+j_3;
        BlastBond_natural_0 b_1 = *_S25;
        if(!(((*_S25).health_0) > 0.0f))
        {
            r_0 = r_0 + 1U;
            continue;
        }
        float3 o_1;
        if((ref_0 & 2147483648U) != 0U)
        {
            o_1 = (float4(b_1.o1_0) ).xyz;
        }
        else
        {
            o_1 = (float4(b_1.o0_0) ).xyz;
        }
        float _S26 = b_1.s_0;
        float s2_0 = _S26 * _S26;
        matrix<float,int(3),int(3)>  _S27 = sym3_0(j_3, 3U, kernelContext_13);
        matrix<float,int(3),int(3)>  k_9 = skew_0(o_1);
        matrix<float,int(3),int(3)>  _S28 = sym3_0(j_3, 1U, kernelContext_13);
        matrix<float,int(3),int(3)>  _S29 = (((_S27) * (k_9)));
        matrix<float,int(3),int(3)>  _S30 = matrix<float,int(3),int(3)> (s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0);
        matrix<float,int(3),int(3)>  _S31 = al_0 + _slang_matrixCompMult(matrix<float,int(3),int(3)> (- s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0), _S29);
        matrix<float,int(3),int(3)>  _S32 = ll_0 + _slang_matrixCompMult(_S30, _S27);
        aa_1 = aa_1 + _slang_matrixCompMult(_S30, _S28 + (((transpose(k_9)) * (_S29))));
        al_0 = _S31;
        ll_0 = _S32;
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
        c_1 = 0U;
        for(;;)
        {
            if(c_1 < 3U)
            {
            }
            else
            {
                break;
            }
            m_2[r_0][c_1] = aa_1[r_0][c_1] * n_4.w_ang_0 * n_4.w_ang_0;
            uint _S33 = c_1 + 3U;
            float _S34 = al_0[r_0][c_1] * n_4.w_ang_0 * n_4.w_lin_0;
            m_2[r_0][_S33] = _S34;
            m_2[_S33][r_0] = _S34;
            m_2[r_0 + 3U][_S33] = ll_0[r_0][c_1] * n_4.w_lin_0 * n_4.w_lin_0;
            c_1 = c_1 + 1U;
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
        c_1 = 0U;
        for(;;)
        {
            if(c_1 <= r_0)
            {
            }
            else
            {
                break;
            }
            packed_0[r_0 * (r_0 + 1U) / 2U + c_1] = m_2[r_0][c_1];
            c_1 = c_1 + 1U;
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
        uint _S35 = ns_0(node_0, 18U + q_0, kernelContext_13);
        uint _S36 = 4U * q_0;
        *(kernelContext_13->scratch_0+_S35) = packed_float4(float4(packed_0[_S36], packed_0[_S36 + 1U], packed_0[_S36 + 2U], packed_0[_S36 + 3U])) ;
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
        c_1 = 0U;
        for(;;)
        {
            if(c_1 < 6U)
            {
            }
            else
            {
                break;
            }
            l_1[r_0][c_1] = 0.0f;
            c_1 = c_1 + 1U;
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
        c_1 = 0U;
        for(;;)
        {
            if(c_1 <= r_0)
            {
            }
            else
            {
                break;
            }
            k_8 = 0U;
            s_3 = m_2[r_0][c_1];
            for(;;)
            {
                if(k_8 < c_1)
                {
                }
                else
                {
                    break;
                }
                float s_4 = s_3 - l_1[r_0][k_8] * l_1[c_1][k_8];
                k_8 = k_8 + 1U;
                s_3 = s_4;
            }
            if(r_0 == c_1)
            {
                bool _S37;
                if(!(s_3 > 0.0f))
                {
                    _S37 = true;
                }
                else
                {
                    _S37 = !isfinite(s_3);
                }
                if(_S37)
                {
                    return false;
                }
                l_1[r_0][c_1] = sqrt(s_3);
            }
            else
            {
                l_1[r_0][c_1] = s_3 / l_1[c_1][c_1];
            }
            c_1 = c_1 + 1U;
        }
        r_0 = r_0 + 1U;
    }
    thread array<array<float, int(6)>, int(6)> inv_0;
    c_1 = 0U;
    for(;;)
    {
        if(c_1 < 6U)
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
            if(r_0 == c_1)
            {
                s_3 = 1.0f;
            }
            else
            {
                s_3 = 0.0f;
            }
            k_8 = 0U;
            float s_5 = s_3;
            for(;;)
            {
                if(k_8 < r_0)
                {
                }
                else
                {
                    break;
                }
                float s_6 = s_5 - l_1[r_0][k_8] * y_0[k_8];
                k_8 = k_8 + 1U;
                s_5 = s_6;
            }
            y_0[r_0] = s_5 / l_1[r_0][r_0];
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
            k_8 = uint(r_1) + 1U;
            s_3 = y_0[r_1];
            for(;;)
            {
                if(k_8 < 6U)
                {
                }
                else
                {
                    break;
                }
                float s_7 = s_3 - l_1[k_8][r_1] * inv_0[k_8][c_1];
                k_8 = k_8 + 1U;
                s_3 = s_7;
            }
            inv_0[r_1][c_1] = s_3 / l_1[r_1][r_1];
            r_1 = r_1 - int(1);
        }
        c_1 = c_1 + 1U;
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
        c_1 = 0U;
        for(;;)
        {
            if(c_1 <= r_0)
            {
            }
            else
            {
                break;
            }
            packed_0[r_0 * (r_0 + 1U) / 2U + c_1] = inv_0[r_0][c_1];
            c_1 = c_1 + 1U;
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
        uint _S38 = ns_0(node_0, 24U + q_0, kernelContext_13);
        uint _S39 = 4U * q_0;
        *(kernelContext_13->scratch_0+_S38) = packed_float4(float4(packed_0[_S39], packed_0[_S39 + 1U], packed_0[_S39 + 2U], packed_0[_S39 + 3U])) ;
        q_0 = q_0 + 1U;
    }
    return true;
}

bool build_block_0(uint node_1, KernelContext_0 thread* kernelContext_14)
{
    if((kernelContext_14->params_0->rotation_0) == 2U)
    {
        bool _S40 = build_full_block_0(node_1, kernelContext_14);
        return _S40;
    }
    BlastNode_natural_0 n_5 = kernelContext_14->nodes_0[node_1];
    float3 _S41 = float3(0.0f) ;
    uint r_2 = n_5.ref_begin_0;
    float3 diag_0 = _S41;
    float3 offd_0 = _S41;
    float3 ks_0 = _S41;
    float cs_0 = 0.0f;
    for(;;)
    {
        if(r_2 < (n_5.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_1 = kernelContext_14->refs_0[r_2];
        uint j_4 = ref_1 & 2147483647U;
        BlastBond_natural_0 device* _S42 = kernelContext_14->bonds_0+j_4;
        BlastBond_natural_0 b_2 = *_S42;
        if(!(((*_S42).health_0) > 0.0f))
        {
            r_2 = r_2 + 1U;
            continue;
        }
        float3 o_2;
        if((ref_1 & 2147483648U) != 0U)
        {
            o_2 = (float4(b_2.o1_0) ).xyz;
        }
        else
        {
            o_2 = (float4(b_2.o0_0) ).xyz;
        }
        float _S43 = b_2.s_0;
        float s2_1 = _S43 * _S43;
        float3 diag_1;
        float3 offd_1;
        if((kernelContext_14->params_0->rotation_0) != 0U)
        {
            float3 _S44 = stiff_0(j_4, 1U, float3(1.0f, 0.0f, 0.0f), kernelContext_14);
            float3 _S45 = stiff_0(j_4, 1U, float3(0.0f, 1.0f, 0.0f), kernelContext_14);
            float3 _S46 = stiff_0(j_4, 1U, float3(0.0f, 0.0f, 1.0f), kernelContext_14);
            float3 _S47 = float3(s2_1) ;
            float _S48 = o_2.x;
            float _S49 = o_2.y;
            float _S50 = o_2.z;
            float3 offd_2 = offd_0 + _S47 * (float3(_S45.x, _S46.x, _S46.y) - float3(_S48 * _S49, _S48 * _S50, _S49 * _S50));
            diag_1 = diag_0 + _S47 * (float3(_S44.x, _S45.y, _S46.z) + float3(dot(o_2, o_2))  - o_2 * o_2);
            offd_1 = offd_2;
        }
        else
        {
            float3 _S51 = float3(s2_1) ;
            float _S52 = o_2.x;
            float _S53 = o_2.y;
            float _S54 = o_2.z;
            float3 offd_3 = offd_0 + _S51 * - float3(_S52 * _S53, _S52 * _S54, _S53 * _S54);
            diag_1 = diag_0 + _S51 * (float3((1.0f + dot(o_2, o_2)))  - o_2 * o_2);
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
    float _S55 = n_5.w_ang_0;
    float da_0 = _S55 * _S55;
    float c_2 = n_5.w_lin_0 * n_5.w_lin_0 * cs_0;
    float3 k_10 = ks_0 * float3((n_5.w_ang_0 * n_5.w_lin_0)) ;
    float4 a0_0 = float4(diag_0 * float3(da_0) , offd_0.x * da_0);
    float _S56 = offd_0.y * da_0;
    float _S57 = offd_0.z * da_0;
    float4 a1_0 = float4(_S56, _S57, c_2, 0.0f);
    uint _S58 = ns_0(node_1, 18U, kernelContext_14);
    *(kernelContext_14->scratch_0+_S58) = packed_float4(a0_0) ;
    *(kernelContext_14->scratch_0+(_S58 + 1U)) = packed_float4(a1_0) ;
    *(kernelContext_14->scratch_0+(_S58 + 2U)) = packed_float4(float4(k_10, 0.0f)) ;
    if(!(c_2 > 0.0f))
    {
        return false;
    }
    float ic_0 = 1.0f / c_2;
    float kk_0 = dot(k_10, k_10);
    float _S59 = k_10.x;
    float sxx_0 = a0_0.x + (_S59 * _S59 - kk_0) * ic_0;
    float _S60 = k_10.y;
    float syy_0 = a0_0.y + (_S60 * _S60 - kk_0) * ic_0;
    float _S61 = k_10.z;
    float szz_0 = a0_0.z + (_S61 * _S61 - kk_0) * ic_0;
    float sxy_0 = a0_0.w + _S59 * _S60 * ic_0;
    float sxz_0 = _S56 + _S59 * _S61 * ic_0;
    float syz_0 = _S57 + _S60 * _S61 * ic_0;
    float cxx_0 = syy_0 * szz_0 - syz_0 * syz_0;
    float cxy_0 = sxz_0 * syz_0 - sxy_0 * szz_0;
    float cxz_0 = sxy_0 * syz_0 - sxz_0 * syy_0;
    float det_0 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    bool _S62;
    if(!(det_0 > 0.0f))
    {
        _S62 = true;
    }
    else
    {
        _S62 = !isfinite(det_0);
    }
    if(_S62)
    {
        return false;
    }
    float id_0 = 1.0f / det_0;
    float cyy_0 = sxx_0 * szz_0 - sxz_0 * sxz_0;
    float cyz_0 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    float czz_0 = sxx_0 * syy_0 - sxy_0 * sxy_0;
    uint _S63 = ns_0(node_1, 21U, kernelContext_14);
    *(kernelContext_14->scratch_0+_S63) = packed_float4(float4(cxx_0 * id_0, cyy_0 * id_0, czz_0 * id_0, cxy_0 * id_0)) ;
    *(kernelContext_14->scratch_0+(_S63 + 1U)) = packed_float4(float4(cxz_0 * id_0, cyz_0 * id_0, ic_0, 0.0f)) ;
    *(kernelContext_14->scratch_0+(_S63 + 2U)) = packed_float4(float4(k_10 * float3(ic_0) , 0.0f)) ;
    return true;
}

uint bs_0(uint j_5, KernelContext_0 thread* kernelContext_15)
{
    return kernelContext_15->params_0->node_count_0 * kernelContext_15->params_0->node_vectors_0 + 2U * j_5;
}

float3 rot_a_0(uint j_6, float3 v_8, KernelContext_0 thread* kernelContext_16)
{
    float3 _S64;
    if((kernelContext_16->params_0->rotation_0) != 0U)
    {
        float3 _S65 = stiff_0(j_6, 0U, v_8, kernelContext_16);
        _S64 = _S65;
    }
    else
    {
        _S64 = v_8;
    }
    return _S64;
}

float3 rot_l_0(uint j_7, float3 v_9, KernelContext_0 thread* kernelContext_17)
{
    float3 _S66;
    if((kernelContext_17->params_0->rotation_0) == 2U)
    {
        float3 _S67 = stiff_0(j_7, 2U, v_9, kernelContext_17);
        _S66 = _S67;
    }
    else
    {
        _S66 = v_9;
    }
    return _S66;
}

void bond_term_0(const BlastBond_natural_0 thread* b_3, uint j_8, uint v_10, float3 thread* ta_0, float3 thread* tl_0, KernelContext_0 thread* kernelContext_18)
{
    uint _S68 = b_3->n0_0;
    BlastNode_natural_0 m0_0 = kernelContext_18->nodes_0[b_3->n0_0];
    uint _S69 = b_3->n1_0;
    BlastNode_natural_0 m1_0 = kernelContext_18->nodes_0[b_3->n1_0];
    float3 _S70 = va_0(b_3->n0_0, v_10, kernelContext_18);
    float3 a0_1 = _S70 * float3(m0_0.w_ang_0) ;
    float3 _S71 = vl_0(_S68, v_10, kernelContext_18);
    float3 l0_0 = _S71 * float3(m0_0.w_lin_0) ;
    float3 _S72 = va_0(_S69, v_10, kernelContext_18);
    float3 a1_1 = _S72 * float3(m1_0.w_ang_0) ;
    float3 _S73 = vl_0(_S69, v_10, kernelContext_18);
    float3 l1_0 = _S73 * float3(m1_0.w_lin_0) ;
    if((kernelContext_18->params_0->rotation_0) == 0U)
    {
        float3 _S74 = float3(b_3->s_0) ;
        *ta_0 = (a0_1 - a1_1) * _S74;
        *tl_0 = (l0_0 - l1_0 + (cross((float4(b_3->o0_0) ).xyz, a0_1) - cross((float4(b_3->o1_0) ).xyz, a1_1))) * _S74;
        return;
    }
    float3 _S75 = rot_a_0(j_8, a0_1 - a1_1, kernelContext_18);
    float3 _S76 = float3(b_3->s_0) ;
    *ta_0 = _S75 * _S76;
    float3 _S77 = rot_l_0(j_8, l0_0 - l1_0 + (cross((float4(b_3->o0_0) ).xyz, a0_1) - cross((float4(b_3->o1_0) ).xyz, a1_1)), kernelContext_18);
    *tl_0 = _S77 * _S76;
    return;
}

void bond_term_1(const BlastBond_natural_0 thread* b_4, uint j_9, uint v_11, float3 thread* ta_1, float3 thread* tl_1, KernelContext_0 thread* kernelContext_19)
{
    uint _S78 = b_4->n0_0;
    BlastNode_natural_0 m0_1 = kernelContext_19->nodes_0[b_4->n0_0];
    uint _S79 = b_4->n1_0;
    BlastNode_natural_0 m1_1 = kernelContext_19->nodes_0[b_4->n1_0];
    float3 _S80 = va_0(b_4->n0_0, v_11, kernelContext_19);
    float3 a0_2 = _S80 * float3(m0_1.w_ang_0) ;
    float3 _S81 = vl_0(_S78, v_11, kernelContext_19);
    float3 l0_1 = _S81 * float3(m0_1.w_lin_0) ;
    float3 _S82 = va_0(_S79, v_11, kernelContext_19);
    float3 a1_2 = _S82 * float3(m1_1.w_ang_0) ;
    float3 _S83 = vl_0(_S79, v_11, kernelContext_19);
    float3 l1_1 = _S83 * float3(m1_1.w_lin_0) ;
    if((kernelContext_19->params_0->rotation_0) == 0U)
    {
        float3 _S84 = float3(b_4->s_0) ;
        *ta_1 = (a0_2 - a1_2) * _S84;
        *tl_1 = (l0_1 - l1_1 + (cross((float4(b_4->o0_0) ).xyz, a0_2) - cross((float4(b_4->o1_0) ).xyz, a1_2))) * _S84;
        return;
    }
    float3 _S85 = rot_a_0(j_9, a0_2 - a1_2, kernelContext_19);
    float3 _S86 = float3(b_4->s_0) ;
    *ta_1 = _S85 * _S86;
    float3 _S87 = rot_l_0(j_9, l0_1 - l1_1 + (cross((float4(b_4->o0_0) ).xyz, a0_2) - cross((float4(b_4->o1_0) ).xyz, a1_2)), kernelContext_19);
    *tl_1 = _S87 * _S86;
    return;
}

void gather_0(uint node_2, bool from_t_0, bool with_u_0, float3 thread* ang_0, float3 thread* lin_0, KernelContext_0 thread* kernelContext_20)
{
    BlastNode_natural_0 n_6 = kernelContext_20->nodes_0[node_2];
    float3 _S88 = float3(0.0f) ;
    uint r_3 = n_6.ref_begin_0;
    float3 acc_a_0 = _S88;
    float3 acc_l_0 = _S88;
    for(;;)
    {
        if(r_3 < (n_6.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_2 = kernelContext_20->refs_0[r_3];
        uint j_10 = ref_2 & 2147483647U;
        thread BlastBond_natural_0 _S89 = *(kernelContext_20->bonds_0+j_10);
        if(!(((&_S89)->health_0) > 0.0f))
        {
            r_3 = r_3 + 1U;
            continue;
        }
        float3 ya_0;
        float3 yl_0;
        if(from_t_0)
        {
            uint _S90 = bs_0(j_10, kernelContext_20);
            float3 _S91 = (float4(*(kernelContext_20->scratch_0+(_S90 + 1U))) ).xyz;
            ya_0 = (float4(*(kernelContext_20->scratch_0+_S90)) ).xyz;
            yl_0 = _S91;
        }
        else
        {
            uint _S92 = j_10 * 2U;
            float3 ya_1 = (float4(*(kernelContext_20->x_0+_S92)) ).xyz;
            float3 yl_1 = (float4(*(kernelContext_20->x_0+(_S92 + 1U))) ).xyz;
            if(with_u_0)
            {
                thread float3 ta_2;
                thread float3 tl_2;
                bond_term_0(&_S89, j_10, 6U, &ta_2, &tl_2, kernelContext_20);
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
        if((kernelContext_20->params_0->rotation_0) == 0U)
        {
            float3 _S93 = float3((&_S89)->s_0) ;
            float3 ya_3 = ya_0 * _S93;
            yl_3 = yl_0 * _S93;
            ya_2 = ya_3;
        }
        else
        {
            float3 _S94 = rot_a_0(j_10, ya_0, kernelContext_20);
            float3 _S95 = float3((&_S89)->s_0) ;
            float3 _S96 = _S94 * _S95;
            float3 _S97 = rot_l_0(j_10, yl_0, kernelContext_20);
            yl_3 = _S97 * _S95;
            ya_2 = _S96;
        }
        float3 acc_a_1;
        float3 acc_l_1;
        if((ref_2 & 2147483648U) != 0U)
        {
            float3 acc_l_2 = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross((float4((&_S89)->o1_0) ).xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            float3 acc_l_3 = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross((float4((&_S89)->o0_0) ).xyz, yl_3));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_3 = r_3 + 1U;
    }
    *ang_0 = acc_a_0 * float3(n_6.w_ang_0) ;
    *lin_0 = acc_l_0 * float3(n_6.w_lin_0) ;
    return;
}

float rebuild_0(uint tid_1, uint n0_1, uint n1_1, bool with_u_1, KernelContext_0 thread* kernelContext_21)
{
    uint p_0 = n0_1 + tid_1;
    float s_8 = 0.0f;
    for(;;)
    {
        if(p_0 < n1_1)
        {
        }
        else
        {
            break;
        }
        uint i_0 = kernelContext_21->lists_0[p_0];
        thread float3 a_2;
        thread float3 l_2;
        gather_0(i_0, false, with_u_1, &a_2, &l_2, kernelContext_21);
        float3 _S98 = va_0(i_0, 14U, kernelContext_21);
        float3 ra_0 = _S98 - a_2;
        float3 _S99 = vl_0(i_0, 14U, kernelContext_21);
        float3 rl_0 = _S99 - l_2;
        set_v_0(i_0, 0U, ra_0, rl_0, kernelContext_21);
        float s_9 = s_8 + norm2_0(ra_0, rl_0);
        p_0 = p_0 + 1024U;
        s_8 = s_9;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    float2 _S100 = blast_group_sum_0(tid_1, float2(s_8, 0.0f), kernelContext_21);
    return _S100.x;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

float frame_float_0(uint k_11, uint w_3, KernelContext_0 thread* kernelContext_22)
{
    uint _S101 = frame_word_1(k_11, w_3, kernelContext_22);
    return (as_type<float>((_S101)));
}

float frame_float_1(uint k_12, uint w_4, KernelContext_0 thread* kernelContext_23)
{
    uint _S102 = frame_word_1(k_12, w_4, kernelContext_23);
    return (as_type<float>((_S102)));
}

float3 frame_axis_0(uint k_13, uint rotations_0, uint kk_1, KernelContext_0 thread* kernelContext_24)
{
    if(rotations_0 == 1U)
    {
        float _S103 = frame_float_1(k_13, 4U, kernelContext_24);
        float _S104 = frame_float_1(k_13, 5U, kernelContext_24);
        float _S105 = frame_float_1(k_13, 6U, kernelContext_24);
        return float3(_S103, _S104, _S105);
    }
    float _S106;
    if(kk_1 == 0U)
    {
        _S106 = 1.0f;
    }
    else
    {
        _S106 = 0.0f;
    }
    float _S107;
    if(kk_1 == 1U)
    {
        _S107 = 1.0f;
    }
    else
    {
        _S107 = 0.0f;
    }
    float _S108;
    if(kk_1 == 2U)
    {
        _S108 = 1.0f;
    }
    else
    {
        _S108 = 0.0f;
    }
    return float3(_S106, _S107, _S108);
}

void frame_terms_0(uint k_14, uint rotations_1, uint i_1, uint v_12, array<float, int(6)> thread* s_10, KernelContext_0 thread* kernelContext_25)
{
    BlastNode_natural_0 n_7 = kernelContext_25->nodes_0[i_1];
    float3 _S109 = vl_0(i_1, v_12, kernelContext_25);
    float3 lin_1 = _S109 / float3(n_7.w_lin_0) ;
    float3 _S110 = va_0(i_1, v_12, kernelContext_25);
    float3 _S111 = _S110 / float3(n_7.w_ang_0)  - cross((float4(n_7.position_0) ).xyz, lin_1);
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
        float3 _S112 = frame_axis_0(k_14, rotations_1, kk_2, kernelContext_25);
        (*s_10)[kk_2] = (*s_10)[kk_2] + dot(_S112, _S111);
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

void frame_solve_0(uint k_15, uint rotations_2, const array<float, int(6)> thread* sum_0, float3 thread* omega_0, float3 thread* t_0, KernelContext_0 thread* kernelContext_26)
{
    uint j_12;
    uint d_0 = rotations_2 + 3U;
    thread array<float, int(6)> c_3;
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
        c_3[z_1] = 0.0f;
        z_1 = z_1 + 1U;
    }
    uint row_0 = 0U;
    for(;;)
    {
        if(row_0 < d_0)
        {
        }
        else
        {
            break;
        }
        float _S113 = (*sum_0)[row_0];
        float _S114 = frame_float_0(k_15, 8U + row_0, kernelContext_26);
        float _S115 = _S113 * _S114;
        j_12 = 0U;
        float value_2 = _S115;
        for(;;)
        {
            if(j_12 < row_0)
            {
            }
            else
            {
                break;
            }
            float _S116 = frame_float_0(k_15, 14U + row_0 * (row_0 + 1U) / 2U + j_12, kernelContext_26);
            float value_3 = value_2 - _S116 * c_3[j_12];
            j_12 = j_12 + 1U;
            value_2 = value_3;
        }
        uint _S117 = row_0 + 1U;
        float _S118 = frame_float_0(k_15, 14U + row_0 * _S117 / 2U + row_0, kernelContext_26);
        c_3[row_0] = value_2 / _S118;
        row_0 = _S117;
    }
    int row_1 = int(d_0) - int(1);
    for(;;)
    {
        if(row_1 >= int(0))
        {
        }
        else
        {
            break;
        }
        uint r_4 = uint(row_1);
        uint _S119 = r_4 + 1U;
        j_12 = _S119;
        for(;;)
        {
            if(j_12 < d_0)
            {
            }
            else
            {
                break;
            }
            uint _S120 = j_12 + 1U;
            float _S121 = frame_float_0(k_15, 14U + j_12 * _S120 / 2U + r_4, kernelContext_26);
            c_3[r_4] = c_3[r_4] - _S121 * c_3[j_12];
            j_12 = _S120;
        }
        float _S122 = frame_float_0(k_15, 14U + r_4 * _S119 / 2U + r_4, kernelContext_26);
        c_3[r_4] = c_3[r_4] / _S122;
        row_1 = row_1 - int(1);
    }
    z_1 = 0U;
    for(;;)
    {
        if(z_1 < d_0)
        {
        }
        else
        {
            break;
        }
        float _S123 = frame_float_0(k_15, 8U + z_1, kernelContext_26);
        c_3[z_1] = c_3[z_1] * _S123;
        z_1 = z_1 + 1U;
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
        float3 _S124 = frame_axis_0(k_15, rotations_2, j_12, kernelContext_26);
        *omega_0 = *omega_0 + _S124 * float3(c_3[j_12]) ;
        j_12 = j_12 + 1U;
    }
    *t_0 = float3(c_3[rotations_2], c_3[rotations_2 + 1U], c_3[rotations_2 + 2U]);
    return;
}

void frame_apply_0(uint i_2, uint v_13, float3 omega_1, float3 t_1, KernelContext_0 thread* kernelContext_27)
{
    BlastNode_natural_0 n_8 = kernelContext_27->nodes_0[i_2];
    float3 _S125 = va_0(i_2, v_13, kernelContext_27);
    float3 _S126 = _S125 - omega_1 / float3(n_8.w_ang_0) ;
    float3 _S127 = vl_0(i_2, v_13, kernelContext_27);
    set_v_0(i_2, v_13, _S126, _S127 - (t_1 + cross((float4(n_8.position_0) ).xyz, omega_1)) / float3(n_8.w_lin_0) , kernelContext_27);
    return;
}

void project_0(uint tid_2, uint k_16, uint rotations_3, uint n0_2, uint n1_2, uint v_14, bool free_0, KernelContext_0 thread* kernelContext_28)
{
    if(!free_0)
    {
        return;
    }
    thread array<float, int(6)> s_11;
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
        s_11[z_2] = 0.0f;
        z_2 = z_2 + 1U;
    }
    uint _S128 = n0_2 + tid_2;
    uint p_1 = _S128;
    for(;;)
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_terms_0(k_16, rotations_3, kernelContext_28->lists_0[p_1], v_14, &s_11, kernelContext_28);
        p_1 = p_1 + 1024U;
    }
    float2 _S129 = blast_group_sum_0(tid_2, float2(s_11[int(0)], s_11[int(1)]), kernelContext_28);
    float2 _S130 = blast_group_sum_0(tid_2, float2(s_11[int(2)], s_11[int(3)]), kernelContext_28);
    float2 _S131 = blast_group_sum_0(tid_2, float2(s_11[int(4)], s_11[int(5)]), kernelContext_28);
    if(tid_2 == 0U)
    {
        float _S132 = _S129.y;
        float _S133 = _S130.x;
        float _S134 = _S130.y;
        float _S135 = _S131.x;
        float _S136 = _S131.y;
        thread array<float, int(6)> _S137;
        _S137[int(0)] = _S129.x;
        _S137[int(1)] = _S132;
        _S137[int(2)] = _S133;
        _S137[int(3)] = _S134;
        _S137[int(4)] = _S135;
        _S137[int(5)] = _S136;
        thread float3 omega_2;
        thread float3 t_2;
        frame_solve_0(k_16, rotations_3, &_S137, &omega_2, &t_2, kernelContext_28);
        *kernelContext_28->g_omega_0 = omega_2;
        *kernelContext_28->g_t_0 = t_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float3 _S138 = *kernelContext_28->g_omega_0;
    float3 _S139 = *kernelContext_28->g_t_0;
    p_1 = _S128;
    for(;;)
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_apply_0(kernelContext_28->lists_0[p_1], v_14, _S138, _S139, kernelContext_28);
        p_1 = p_1 + 1024U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return;
}

float bond_pass_0(uint tid_3, uint b0_0, uint b1_0, uint v_15, KernelContext_0 thread* kernelContext_29)
{
    uint q_1 = b0_0 + tid_3;
    float z_3 = 0.0f;
    for(;;)
    {
        if(q_1 < b1_0)
        {
        }
        else
        {
            break;
        }
        uint j_13 = kernelContext_29->lists_0[q_1];
        thread BlastBond_natural_0 _S140 = *(kernelContext_29->bonds_0+j_13);
        float3 _S141 = float3(0.0f) ;
        thread float3 ta_3 = _S141;
        thread float3 tl_3 = _S141;
        if(((&_S140)->health_0) > 0.0f)
        {
            bond_term_0(&_S140, j_13, v_15, &ta_3, &tl_3, kernelContext_29);
        }
        float z_4 = z_3 + norm2_0(ta_3, tl_3);
        uint _S142 = bs_0(j_13, kernelContext_29);
        *(kernelContext_29->scratch_0+_S142) = packed_float4(float4(ta_3, 0.0f)) ;
        *(kernelContext_29->scratch_0+(_S142 + 1U)) = packed_float4(float4(tl_3, 0.0f)) ;
        q_1 = q_1 + 1024U;
        z_3 = z_4;
    }
    float2 _S143 = blast_group_sum_0(tid_3, float2(z_3, 0.0f), kernelContext_29);
    return _S143.x;
}

float residual_norm_0(uint tid_4, uint n0_3, uint n1_3, KernelContext_0 thread* kernelContext_30)
{
    uint p_2 = n0_3 + tid_4;
    float s_12 = 0.0f;
    for(;;)
    {
        if(p_2 < n1_3)
        {
        }
        else
        {
            break;
        }
        uint i_3 = kernelContext_30->lists_0[p_2];
        float3 _S144 = va_0(i_3, 0U, kernelContext_30);
        float3 _S145 = vl_0(i_3, 0U, kernelContext_30);
        float s_13 = s_12 + norm2_0(_S144, _S145);
        p_2 = p_2 + 1024U;
        s_12 = s_13;
    }
    float2 _S146 = blast_group_sum_0(tid_4, float2(s_12, 0.0f), kernelContext_30);
    return _S146.x;
}

void full_apply_0(uint n_9, uint at_1, float3 a_3, float3 l_3, float3 thread* oa_0, float3 thread* ol_0, KernelContext_0 thread* kernelContext_31)
{
    thread array<float, int(24)> m_3;
    uint q_2 = 0U;
    for(;;)
    {
        if(q_2 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S147 = ns_0(n_9, at_1 + q_2, kernelContext_31);
        float4 _S148 = float4(*(kernelContext_31->scratch_0+_S147)) ;
        uint _S149 = 4U * q_2;
        m_3[_S149] = _S148.x;
        m_3[_S149 + 1U] = _S148.y;
        m_3[_S149 + 2U] = _S148.z;
        m_3[_S149 + 3U] = _S148.w;
        q_2 = q_2 + 1U;
    }
    array<float, int(6)> _S150 = { { a_3.x, a_3.y, a_3.z, l_3.x, l_3.y, l_3.z } };
    thread array<float, int(6)> o_3;
    uint r_5 = 0U;
    for(;;)
    {
        if(r_5 < 6U)
        {
        }
        else
        {
            break;
        }
        uint c_4 = 0U;
        float s_14 = 0.0f;
        for(;;)
        {
            if(c_4 < 6U)
            {
            }
            else
            {
                break;
            }
            if(r_5 >= c_4)
            {
                q_2 = r_5 * (r_5 + 1U) / 2U + c_4;
            }
            else
            {
                q_2 = c_4 * (c_4 + 1U) / 2U + r_5;
            }
            float s_15 = s_14 + m_3[q_2] * _S150[c_4];
            c_4 = c_4 + 1U;
            s_14 = s_15;
        }
        o_3[r_5] = s_14;
        r_5 = r_5 + 1U;
    }
    *oa_0 = float3(o_3[int(0)], o_3[int(1)], o_3[int(2)]);
    *ol_0 = float3(o_3[int(3)], o_3[int(4)], o_3[int(5)]);
    return;
}

float3 sym_mul_0(float4 s0_0, float4 s1_0, float3 v_16)
{
    float _S151 = v_16.x;
    float _S152 = s0_0.w;
    float _S153 = v_16.y;
    float _S154 = s1_0.x;
    float _S155 = v_16.z;
    float _S156 = s1_0.y;
    return float3(s0_0.x * _S151 + _S152 * _S153 + _S154 * _S155, _S152 * _S151 + s0_0.y * _S153 + _S156 * _S155, _S154 * _S151 + _S156 * _S153 + s0_0.z * _S155);
}

void inverse_apply_0(uint n_10, float3 a_4, float3 l_4, float3 thread* oa_1, float3 thread* ol_1, KernelContext_0 thread* kernelContext_32)
{
    if((kernelContext_32->params_0->rotation_0) == 2U)
    {
        full_apply_0(n_10, 24U, a_4, l_4, oa_1, ol_1, kernelContext_32);
        return;
    }
    uint _S157 = ns_0(n_10, 21U, kernelContext_32);
    float4 _S158 = float4(*(kernelContext_32->scratch_0+(_S157 + 1U))) ;
    float3 kc_0 = (float4(*(kernelContext_32->scratch_0+(_S157 + 2U))) ).xyz;
    float3 _S159 = sym_mul_0(float4(*(kernelContext_32->scratch_0+_S157)) , _S158, a_4 + cross(kc_0, l_4));
    *oa_1 = _S159;
    *ol_1 = l_4 * float3(_S158.z)  - cross(kc_0, _S159);
    return;
}

float node_pass_0(uint tid_5, uint n0_4, uint n1_4, uint out_v_0, KernelContext_0 thread* kernelContext_33)
{
    uint p_3 = n0_4 + tid_5;
    float s_16 = 0.0f;
    for(;;)
    {
        if(p_3 < n1_4)
        {
        }
        else
        {
            break;
        }
        uint i_4 = kernelContext_33->lists_0[p_3];
        thread float3 a_5;
        thread float3 l_5;
        gather_0(i_4, true, false, &a_5, &l_5, kernelContext_33);
        set_v_0(i_4, out_v_0, a_5, l_5, kernelContext_33);
        float s_17 = s_16 + norm2_0(a_5, l_5);
        p_3 = p_3 + 1024U;
        s_16 = s_17;
    }
    return s_16;
}

void block_apply_0(uint n_11, float3 a_6, float3 l_6, float3 thread* oa_2, float3 thread* ol_2, KernelContext_0 thread* kernelContext_34)
{
    if((kernelContext_34->params_0->rotation_0) == 2U)
    {
        full_apply_0(n_11, 18U, a_6, l_6, oa_2, ol_2, kernelContext_34);
        return;
    }
    uint _S160 = ns_0(n_11, 18U, kernelContext_34);
    float4 _S161 = float4(*(kernelContext_34->scratch_0+(_S160 + 1U))) ;
    float3 k_17 = (float4(*(kernelContext_34->scratch_0+(_S160 + 2U))) ).xyz;
    *oa_2 = sym_mul_0(float4(*(kernelContext_34->scratch_0+_S160)) , _S161, a_6) - cross(k_17, l_6);
    *ol_2 = cross(k_17, a_6) + float3(_S161.z)  * l_6;
    return;
}

float group_max_0(uint tid_6, float v_17, KernelContext_0 thread* kernelContext_35)
{
    (*kernelContext_35->g_max_0)[tid_6] = v_17;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint k_18 = 512U;
    for(;;)
    {
        if(k_18 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_6 < k_18)
        {
            (*kernelContext_35->g_max_0)[tid_6] = max((*kernelContext_35->g_max_0)[tid_6], (*kernelContext_35->g_max_0)[tid_6 + k_18]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_18 = k_18 >> 1U;
    }
    if(tid_6 == 0U)
    {
        *kernelContext_35->g_max_out_0 = (*kernelContext_35->g_max_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S162 = kernelContext_35->g_max_out_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *_S162;
}

[[kernel]] void blast_native(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_1 [[buffer(0)]], packed_uint4 device* islands_1 [[buffer(1)]], packed_uint4 device* report_1 [[buffer(8)]], uint device* lists_1 [[buffer(9)]], packed_float4 device* accel_1 [[buffer(5)]], packed_float4 device* scratch_1 [[buffer(7)]], BlastNode_natural_0 device* nodes_1 [[buffer(2)]], uint device* refs_1 [[buffer(3)]], BlastBond_natural_0 device* bonds_1 [[buffer(4)]], packed_float4 device* stiffness_1 [[buffer(10)]], packed_float4 device* x_1 [[buffer(6)]])
{
    float force_start_0;
    float best_0;
    bool failed_0;
    uint q_3;
    bool ok_0;
    thread KernelContext_0 kernelContext_36;
    (&kernelContext_36)->params_0 = params_1;
    (&kernelContext_36)->islands_0 = islands_1;
    (&kernelContext_36)->report_0 = report_1;
    (&kernelContext_36)->lists_0 = lists_1;
    (&kernelContext_36)->accel_0 = accel_1;
    (&kernelContext_36)->scratch_0 = scratch_1;
    (&kernelContext_36)->nodes_0 = nodes_1;
    (&kernelContext_36)->refs_0 = refs_1;
    (&kernelContext_36)->bonds_0 = bonds_1;
    (&kernelContext_36)->stiffness_0 = stiffness_1;
    (&kernelContext_36)->x_0 = x_1;
    threadgroup array<float2, int(32)> g_blast_part_1;
    (&kernelContext_36)->g_blast_part_0 = &g_blast_part_1;
    threadgroup float3 g_omega_1;
    (&kernelContext_36)->g_omega_0 = &g_omega_1;
    threadgroup float3 g_t_1;
    (&kernelContext_36)->g_t_0 = &g_t_1;
    threadgroup array<float, int(1024)> g_max_1;
    (&kernelContext_36)->g_max_0 = &g_max_1;
    threadgroup float g_max_out_1;
    (&kernelContext_36)->g_max_out_0 = &g_max_out_1;
    uint tid_7 = thread_2.x;
    uint k_19 = group_0.x;
    if(k_19 >= (params_1->island_count_0))
    {
        return;
    }
    uint4 _S163 = uint4(*((&kernelContext_36)->islands_0+k_19)) ;
    uint n0_5 = _S163.x;
    uint n1_5 = _S163.y;
    uint b0_1 = _S163.z;
    uint _S164 = _S163.w;
    uint _S165 = n1_5 - n0_5;
    uint p_4;
    if(_S165 > ((&kernelContext_36)->params_0->wide_min_0))
    {
        p_4 = 1U;
    }
    else
    {
        p_4 = 0U;
    }
    if((group_uniform_0(p_4, tid_7)) != 0U)
    {
        return;
    }
    uint4 _S166 = uint4(*((&kernelContext_36)->report_0+k_19)) ;
    uint last_flags_0 = group_uniform_0(_S166.y, tid_7);
    bool warm_1 = ((&kernelContext_36)->params_0->warm_0) != 0U;
    bool carried_0;
    if(warm_1)
    {
        carried_0 = (last_flags_0 & 4U) != 0U;
    }
    else
    {
        carried_0 = false;
    }
    float singular_0;
    if(carried_0)
    {
        uint _S167 = n0_5 + tid_7;
        singular_0 = 0.0f;
        p_4 = _S167;
        for(;;)
        {
            if(p_4 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_5 = (&kernelContext_36)->lists_0[p_4];
            uint _S168 = i_5 * 2U;
            float4 _S169 = float4(*((&kernelContext_36)->accel_0+_S168)) ;
            uint _S170 = ns_0(i_5, 16U, &kernelContext_36);
            if(any(_S169 != (float4(*((&kernelContext_36)->scratch_0+_S170)) )))
            {
                carried_0 = true;
            }
            else
            {
                carried_0 = any((float4(*((&kernelContext_36)->accel_0+(_S168 + 1U))) ) != (float4(*((&kernelContext_36)->scratch_0+(_S170 + 1U))) ));
            }
            if(carried_0)
            {
                singular_0 = 1.0f;
            }
            p_4 = p_4 + 1024U;
        }
        float2 _S171 = blast_group_sum_0(tid_7, float2(singular_0, 0.0f), &kernelContext_36);
        if((_S171.x) == 0.0f)
        {
            if(tid_7 == 0U)
            {
                *((&kernelContext_36)->report_0+k_19) = packed_uint4(uint4(0U, 7U, _S166.z, _S166.w)) ;
            }
            return;
        }
    }
    uint3 _S172 = carry_of_0(k_19, &kernelContext_36);
    uint4 carry_1 = group_uniform4_0(uint4(_S172, _S165), tid_7);
    uint3 kept_0 = carry_1.xyz;
    bool _S173 = carry_kept_0(kept_0, carry_1.w, &kernelContext_36);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bool _S174 = tid_7 == 0U;
    if(_S174)
    {
        carried_0 = ((&kernelContext_36)->params_0->carry_0) != 0U;
    }
    else
    {
        carried_0 = false;
    }
    if(carried_0)
    {
        set_carry_0(k_19, 0U, 0U, 0.0f, &kernelContext_36);
    }
    uint _S175 = n0_5 + tid_7;
    singular_0 = 0.0f;
    p_4 = _S175;
    float target_0 = 0.0f;
    float change_0 = 0.0f;
    for(;;)
    {
        if(p_4 < n1_5)
        {
        }
        else
        {
            break;
        }
        uint i_6 = (&kernelContext_36)->lists_0[p_4];
        BlastNode_natural_0 n_12 = (&kernelContext_36)->nodes_0[i_6];
        uint _S176 = i_6 * 2U;
        float4 _S177 = float4(*((&kernelContext_36)->accel_0+_S176)) ;
        float4 _S178 = float4(*((&kernelContext_36)->accel_0+(_S176 + 1U))) ;
        float3 _S179 = - _S177.xyz;
        if((n_12.w_ang_0) > 0.0f)
        {
            force_start_0 = 1.0f / n_12.w_ang_0;
        }
        else
        {
            force_start_0 = 1.0f;
        }
        float3 ba_0 = _S179 * float3(force_start_0) ;
        float3 _S180 = - _S178.xyz;
        if((n_12.w_lin_0) > 0.0f)
        {
            best_0 = (&kernelContext_36)->params_0->recip_length_0 / n_12.w_lin_0;
        }
        else
        {
            best_0 = (&kernelContext_36)->params_0->recip_length_0;
        }
        float3 bl_0 = _S180 * float3(best_0) ;
        float target_1 = target_0 + norm2_0(ba_0, bl_0);
        if(_S173)
        {
            float3 _S181 = va_0(i_6, 14U, &kernelContext_36);
            float3 _S182 = ba_0 - _S181;
            float3 _S183 = vl_0(i_6, 14U, &kernelContext_36);
            change_0 = change_0 + norm2_0(_S182, bl_0 - _S183);
        }
        set_v_0(i_6, 14U, ba_0, bl_0, &kernelContext_36);
        set_v_0(i_6, 0U, ba_0, bl_0, &kernelContext_36);
        uint _S184 = ns_0(i_6, 16U, &kernelContext_36);
        *((&kernelContext_36)->scratch_0+_S184) = packed_float4(_S177) ;
        *((&kernelContext_36)->scratch_0+(_S184 + 1U)) = packed_float4(_S178) ;
        float3 _S185 = float3(0.0f) ;
        set_v_0(i_6, 6U, _S185, _S185, &kernelContext_36);
        if(((&kernelContext_36)->params_0->carry_0) == 0U)
        {
            set_v_0(i_6, 2U, _S185, _S185, &kernelContext_36);
        }
        bool _S186 = build_block_0(i_6, &kernelContext_36);
        if(!_S186)
        {
            singular_0 = 1.0f;
        }
        p_4 = p_4 + 1024U;
        target_0 = target_1;
    }
    uint q_4;
    if(!warm_1)
    {
        q_4 = b0_1 + tid_7;
        for(;;)
        {
            if(q_4 < _S164)
            {
            }
            else
            {
                break;
            }
            uint _S187 = (&kernelContext_36)->lists_0[q_4] * 2U;
            packed_float4 _S188 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_36)->x_0+_S187) = _S188;
            *((&kernelContext_36)->x_0+(_S187 + 1U)) = _S188;
            q_4 = q_4 + 1024U;
        }
    }
    float2 _S189 = blast_group_sum_0(tid_7, float2(target_0, singular_0), &kernelContext_36);
    float _S190 = _S189.x;
    float delta2_0 = _S190 * (&kernelContext_36)->params_0->tolerance_0 * (&kernelContext_36)->params_0->tolerance_0;
    float2 _S191 = blast_group_sum_0(tid_7, float2(change_0, 0.0f), &kernelContext_36);
    float moved_0 = _S191.x;
    if(_S173)
    {
        carried_0 = moved_0 <= ((&kernelContext_36)->params_0->tolerance_0 * (&kernelContext_36)->params_0->tolerance_0 * _S190);
    }
    else
    {
        carried_0 = false;
    }
    uint _S192 = frame_word_0(k_19, 0U, &kernelContext_36);
    uint _S193 = frame_word_0(k_19, 1U, &kernelContext_36);
    uint _S194 = frame_word_0(k_19, 2U, &kernelContext_36);
    uint3 frame_0 = group_uniform4_0(uint4(_S192, _S193, _S194, 0U), tid_7).xyz;
    bool converged_0;
    if((_S189.y) > 0.0f)
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = (frame_0.z) != 0U;
    }
    bool _S195 = (frame_0.x) == 0U;
    uint _S196 = frame_0.y;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if(!(delta2_0 > 0.0f))
    {
        q_4 = b0_1 + tid_7;
        for(;;)
        {
            if(q_4 < _S164)
            {
            }
            else
            {
                break;
            }
            uint _S197 = (&kernelContext_36)->lists_0[q_4] * 2U;
            packed_float4 _S198 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_36)->x_0+_S197) = _S198;
            *((&kernelContext_36)->x_0+(_S197 + 1U)) = _S198;
            q_4 = q_4 + 1024U;
        }
        if(_S174)
        {
            packed_uint4 device* _S199 = (&kernelContext_36)->report_0+k_19;
            if(warm_1)
            {
                p_4 = 4U;
            }
            else
            {
                p_4 = 0U;
            }
            *_S199 = packed_uint4(uint4(0U, 1U | p_4, (as_type<uint>((0.0f))), (as_type<uint>((delta2_0))))) ;
        }
        return;
    }
    if(warm_1)
    {
        float _S200 = rebuild_0(tid_7, n0_5, n1_5, false, &kernelContext_36);
    }
    if(carried_0)
    {
        singular_0 = (as_type<float>((kept_0.z)));
    }
    else
    {
        singular_0 = 0.0f;
    }
    float _S201 = float_bits_0(2139095040U);
    if(((&kernelContext_36)->params_0->force_tolerance_0) > 0.0f)
    {
        q_4 = b0_1 + tid_7;
        force_start_0 = 0.0f;
        for(;;)
        {
            if(q_4 < _S164)
            {
            }
            else
            {
                break;
            }
            uint _S202 = (&kernelContext_36)->lists_0[q_4] * 2U;
            float f_0 = force_start_0 + norm2_0((float4(*((&kernelContext_36)->x_0+_S202)) ).xyz, (float4(*((&kernelContext_36)->x_0+(_S202 + 1U))) ).xyz);
            q_4 = q_4 + 1024U;
            force_start_0 = f_0;
        }
        float2 _S203 = blast_group_sum_0(tid_7, float2(force_start_0, 0.0f), &kernelContext_36);
        force_start_0 = sqrt(_S203.x);
    }
    else
    {
        force_start_0 = 0.0f;
    }
    best_0 = _S201;
    q_4 = 0U;
    float step_0 = _S201;
    float previous_0 = singular_0;
    float num_0 = 0.0f;
    uint iteration_0 = 0U;
    float travel_0 = 0.0f;
    for(;;)
    {
        if(!converged_0)
        {
            failed_0 = iteration_0 < ((&kernelContext_36)->params_0->max_iterations_0);
        }
        else
        {
            failed_0 = false;
        }
        if(failed_0)
        {
        }
        else
        {
            bool _S204 = converged_0;
            converged_0 = false;
            failed_0 = _S204;
            break;
        }
        project_0(tid_7, k_19, _S196, n0_5, n1_5, 0U, _S195, &kernelContext_36);
        float _S205 = bond_pass_0(tid_7, b0_1, _S164, 0U, &kernelContext_36);
        bool _S206 = _S205 <= delta2_0;
        if(_S206)
        {
            float _S207 = residual_norm_0(tid_7, n0_5, n1_5, &kernelContext_36);
            ok_0 = !(_S207 <= delta2_0);
        }
        else
        {
            ok_0 = false;
        }
        bool _S208 = iteration_0 > 0U;
        bool _S209;
        if(_S208)
        {
            _S209 = true;
        }
        else
        {
            _S209 = warm_1;
        }
        bool _S210;
        if(_S209)
        {
            _S210 = _S206;
        }
        else
        {
            _S210 = false;
        }
        bool _S211;
        if(_S210)
        {
            _S211 = !ok_0;
        }
        else
        {
            _S211 = false;
        }
        bool hold_0;
        float num_1;
        float previous_1;
        if(_S211)
        {
            float _S212 = rebuild_0(tid_7, n0_5, n1_5, true, &kernelContext_36);
            project_0(tid_7, k_19, _S196, n0_5, n1_5, 0U, _S195, &kernelContext_36);
            float _S213 = bond_pass_0(tid_7, b0_1, _S164, 0U, &kernelContext_36);
            if(_S213 <= delta2_0)
            {
                float _S214 = residual_norm_0(tid_7, n0_5, n1_5, &kernelContext_36);
                hold_0 = !(_S214 <= delta2_0);
            }
            else
            {
                hold_0 = false;
            }
            num_1 = _S213;
            previous_1 = 0.0f;
        }
        else
        {
            num_1 = _S205;
            hold_0 = ok_0;
            previous_1 = previous_0;
        }
        if(num_1 < (best_0 * 0.99000000953674316f))
        {
            best_0 = num_1;
            q_4 = iteration_0;
        }
        else
        {
            if((iteration_0 - q_4) >= 512U)
            {
                bool _S215 = converged_0;
                converged_0 = false;
                failed_0 = _S215;
                num_0 = num_1;
                previous_0 = previous_1;
                break;
            }
        }
        bool _S216;
        if(num_1 <= delta2_0)
        {
            _S216 = !hold_0;
        }
        else
        {
            _S216 = false;
        }
        if(_S216)
        {
            bool _S217 = converged_0;
            converged_0 = true;
            failed_0 = _S217;
            num_0 = num_1;
            previous_0 = previous_1;
            break;
        }
        bool _S218;
        if(((&kernelContext_36)->params_0->force_tolerance_0) > 0.0f)
        {
            _S218 = _S208;
        }
        else
        {
            _S218 = false;
        }
        bool _S219;
        if(_S218)
        {
            _S219 = step_0 <= ((&kernelContext_36)->params_0->force_tolerance_0 * (force_start_0 + travel_0));
        }
        else
        {
            _S219 = false;
        }
        float mloc_0;
        if(_S219)
        {
            q_3 = b0_1 + tid_7;
            mloc_0 = 0.0f;
            for(;;)
            {
                if(q_3 < _S164)
                {
                }
                else
                {
                    break;
                }
                uint j_14 = (&kernelContext_36)->lists_0[q_3];
                thread BlastBond_natural_0 _S220 = *((&kernelContext_36)->bonds_0+j_14);
                thread float3 ta_4;
                thread float3 tl_4;
                bond_term_1(&_S220, j_14, 6U, &ta_4, &tl_4, &kernelContext_36);
                uint _S221 = j_14 * 2U;
                float f_1 = mloc_0 + norm2_0((float4(*((&kernelContext_36)->x_0+_S221)) ).xyz + ta_4, (float4(*((&kernelContext_36)->x_0+(_S221 + 1U))) ).xyz + tl_4);
                q_3 = q_3 + 1024U;
                mloc_0 = f_1;
            }
            float _S222 = (&kernelContext_36)->params_0->force_tolerance_0;
            float2 _S223 = blast_group_sum_0(tid_7, float2(mloc_0, 0.0f), &kernelContext_36);
            if(step_0 <= (_S222 * sqrt(_S223.x)))
            {
                bool _S224 = converged_0;
                converged_0 = true;
                failed_0 = _S224;
                num_0 = num_1;
                previous_0 = previous_1;
                break;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        bool _S225 = iteration_0 == 0U;
        bool _S226;
        if(_S225)
        {
            _S226 = ((&kernelContext_36)->params_0->first_polynomial_0) == 0U;
        }
        else
        {
            _S226 = false;
        }
        bool _S227;
        if(_S226)
        {
            _S227 = !carried_0;
        }
        else
        {
            _S227 = false;
        }
        if(_S227)
        {
            p_4 = _S175;
            for(;;)
            {
                if(p_4 < n1_5)
                {
                }
                else
                {
                    break;
                }
                uint i_7 = (&kernelContext_36)->lists_0[p_4];
                float3 _S228 = va_0(i_7, 0U, &kernelContext_36);
                float3 _S229 = vl_0(i_7, 0U, &kernelContext_36);
                set_v_0(i_7, 8U, _S228, _S229, &kernelContext_36);
                p_4 = p_4 + 1024U;
            }
        }
        else
        {
            p_4 = _S175;
            for(;;)
            {
                if(p_4 < n1_5)
                {
                }
                else
                {
                    break;
                }
                uint i_8 = (&kernelContext_36)->lists_0[p_4];
                float3 _S230 = va_0(i_8, 0U, &kernelContext_36);
                float3 _S231 = vl_0(i_8, 0U, &kernelContext_36);
                thread float3 a_7;
                thread float3 l_7;
                inverse_apply_0(i_8, _S230, _S231, &a_7, &l_7, &kernelContext_36);
                set_v_0(i_8, 10U, a_7, l_7, &kernelContext_36);
                p_4 = p_4 + 1024U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            float _S232 = bond_pass_0(tid_7, b0_1, _S164, 10U, &kernelContext_36);
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            float _S233 = node_pass_0(tid_7, n0_5, n1_5, 12U, &kernelContext_36);
            q_3 = _S175;
            for(;;)
            {
                if(q_3 < n1_5)
                {
                }
                else
                {
                    break;
                }
                uint i_9 = (&kernelContext_36)->lists_0[q_3];
                float3 _S234 = va_0(i_9, 10U, &kernelContext_36);
                float3 _S235 = vl_0(i_9, 10U, &kernelContext_36);
                thread float3 sa_0;
                thread float3 sl_0;
                block_apply_0(i_9, _S234, _S235, &sa_0, &sl_0, &kernelContext_36);
                float3 _S236 = va_0(i_9, 12U, &kernelContext_36);
                float3 _S237 = _S236 - sa_0;
                float3 _S238 = vl_0(i_9, 12U, &kernelContext_36);
                thread float3 ma_0;
                thread float3 ml_0;
                inverse_apply_0(i_9, _S237, _S238 - sl_0, &ma_0, &ml_0, &kernelContext_36);
                float3 _S239 = va_0(i_9, 10U, &kernelContext_36);
                float3 _S240 = float3(1.68946564197540283f) ;
                float3 _S241 = float3(1.5220409631729126f) ;
                float3 _S242 = _S240 * _S239 - _S241 * ma_0;
                float3 _S243 = vl_0(i_9, 10U, &kernelContext_36);
                set_v_0(i_9, 8U, _S242, _S240 * _S243 - _S241 * ml_0, &kernelContext_36);
                q_3 = q_3 + 1024U;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        project_0(tid_7, k_19, _S196, n0_5, n1_5, 8U, _S195, &kernelContext_36);
        mloc_0 = 0.0f;
        p_4 = _S175;
        for(;;)
        {
            if(p_4 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_10 = (&kernelContext_36)->lists_0[p_4];
            float3 _S244 = va_0(i_10, 8U, &kernelContext_36);
            float3 a_8 = abs(_S244);
            float3 _S245 = vl_0(i_10, 8U, &kernelContext_36);
            float3 l_8 = abs(_S245);
            float _S246 = max(mloc_0, max(max(max(a_8.x, a_8.y), max(a_8.z, l_8.x)), max(l_8.y, l_8.z)));
            uint p_5 = p_4 + 1024U;
            mloc_0 = _S246;
            p_4 = p_5;
        }
        float _S247 = group_max_0(tid_7, mloc_0, &kernelContext_36);
        bool _S248;
        if(!(_S247 > 0.0f))
        {
            _S248 = true;
        }
        else
        {
            _S248 = !isfinite(_S247);
        }
        if(_S248)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_1;
            previous_0 = previous_1;
            break;
        }
        float _S249 = 1.0f / _S247;
        q_3 = _S175;
        float gl_0 = 0.0f;
        for(;;)
        {
            if(q_3 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_11 = (&kernelContext_36)->lists_0[q_3];
            float3 _S250 = va_0(i_11, 8U, &kernelContext_36);
            float3 _S251 = float3(_S249) ;
            float3 ga_0 = _S250 * _S251;
            float3 _S252 = vl_0(i_11, 8U, &kernelContext_36);
            float3 gn_0 = _S252 * _S251;
            set_v_0(i_11, 8U, ga_0, gn_0, &kernelContext_36);
            float3 _S253 = va_0(i_11, 0U, &kernelContext_36);
            float _S254 = dot(_S253, ga_0);
            float3 _S255 = vl_0(i_11, 0U, &kernelContext_36);
            float gl_1 = gl_0 + (_S254 + dot(_S255, gn_0));
            q_3 = q_3 + 1024U;
            gl_0 = gl_1;
        }
        float2 _S256 = blast_group_sum_0(tid_7, float2(gl_0, 0.0f), &kernelContext_36);
        float gamma_2 = _S256.x;
        bool _S257 = gamma_2 > 0.0f;
        bool _S258;
        if(!_S257)
        {
            _S258 = true;
        }
        else
        {
            _S258 = !isfinite(gamma_2);
        }
        if(_S258)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_1;
            previous_0 = previous_1;
            break;
        }
        bool _S259;
        if(iteration_0 > 1U)
        {
            _S259 = true;
        }
        else
        {
            if(((&kernelContext_36)->params_0->first_polynomial_0) != 0U)
            {
                _S259 = _S208;
            }
            else
            {
                _S259 = false;
            }
        }
        bool _S260;
        if(_S259)
        {
            _S260 = true;
        }
        else
        {
            _S260 = carried_0;
        }
        bool _S261;
        if(_S260)
        {
            _S261 = previous_1 > 0.0f;
        }
        else
        {
            _S261 = false;
        }
        float beta_0;
        if(_S261)
        {
            beta_0 = gamma_2 / previous_1;
        }
        else
        {
            beta_0 = 0.0f;
        }
        bool _S262;
        if(carried_0)
        {
            _S262 = _S225;
        }
        else
        {
            _S262 = false;
        }
        bool _S263;
        if(_S262)
        {
            _S263 = !(beta_0 <= 16.0f);
        }
        else
        {
            _S263 = false;
        }
        float beta_1;
        if(_S263)
        {
            beta_1 = 0.0f;
        }
        else
        {
            beta_1 = beta_0;
        }
        uint p_6 = _S175;
        for(;;)
        {
            if(p_6 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_12 = (&kernelContext_36)->lists_0[p_6];
            if(beta_1 == 0.0f)
            {
                float3 _S264 = va_0(i_12, 8U, &kernelContext_36);
                float3 _S265 = vl_0(i_12, 8U, &kernelContext_36);
                set_v_0(i_12, 2U, _S264, _S265, &kernelContext_36);
            }
            else
            {
                float3 _S266 = va_0(i_12, 8U, &kernelContext_36);
                float3 _S267 = va_0(i_12, 2U, &kernelContext_36);
                float3 _S268 = _S266 + _S267 * float3(beta_1) ;
                float3 _S269 = vl_0(i_12, 8U, &kernelContext_36);
                float3 _S270 = vl_0(i_12, 2U, &kernelContext_36);
                set_v_0(i_12, 2U, _S268, _S269 + _S270 * float3(beta_1) , &kernelContext_36);
            }
            p_6 = p_6 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float _S271 = bond_pass_0(tid_7, b0_1, _S164, 2U, &kernelContext_36);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float _S272 = node_pass_0(tid_7, n0_5, n1_5, 4U, &kernelContext_36);
        bool _S273;
        if(((&kernelContext_36)->params_0->force_tolerance_0) > 0.0f)
        {
            if(_S271 > 0.0f)
            {
                _S273 = _S257;
            }
            else
            {
                _S273 = false;
            }
            float step_1;
            if(_S273)
            {
                step_1 = gamma_2 / sqrt(_S271);
            }
            else
            {
                step_1 = _S201;
            }
            float travel_1;
            if(isfinite(step_1))
            {
                travel_1 = travel_0 + step_1;
            }
            else
            {
                travel_1 = travel_0;
            }
            bool _S274;
            if(_S208)
            {
                _S274 = true;
            }
            else
            {
                _S274 = ((&kernelContext_36)->params_0->first_polynomial_0) != 0U;
            }
            float step_2;
            if(!_S274)
            {
                step_2 = _S201;
            }
            else
            {
                step_2 = step_1;
            }
            step_0 = step_2;
            travel_0 = travel_1;
        }
        uint _S275 = iteration_0 + 1U;
        if(!(_S271 > 0.0f))
        {
            _S273 = true;
        }
        else
        {
            _S273 = !isfinite(_S271);
        }
        if(_S273)
        {
            bool _S276 = converged_0;
            converged_0 = false;
            failed_0 = _S276;
            num_0 = num_1;
            previous_0 = gamma_2;
            iteration_0 = _S275;
            break;
        }
        float _S277 = gamma_2 / _S271;
        uint p_7 = _S175;
        for(;;)
        {
            if(p_7 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_13 = (&kernelContext_36)->lists_0[p_7];
            float3 _S278 = va_0(i_13, 6U, &kernelContext_36);
            float3 _S279 = va_0(i_13, 2U, &kernelContext_36);
            float3 _S280 = float3(_S277) ;
            float3 _S281 = _S278 + _S279 * _S280;
            float3 _S282 = vl_0(i_13, 6U, &kernelContext_36);
            float3 _S283 = vl_0(i_13, 2U, &kernelContext_36);
            set_v_0(i_13, 6U, _S281, _S282 + _S283 * _S280, &kernelContext_36);
            float3 _S284 = va_0(i_13, 0U, &kernelContext_36);
            float3 _S285 = va_0(i_13, 4U, &kernelContext_36);
            float3 _S286 = _S284 - _S285 * _S280;
            float3 _S287 = vl_0(i_13, 0U, &kernelContext_36);
            float3 _S288 = vl_0(i_13, 4U, &kernelContext_36);
            set_v_0(i_13, 0U, _S286, _S287 - _S288 * _S280, &kernelContext_36);
            p_7 = p_7 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        previous_0 = gamma_2;
        num_0 = num_1;
        iteration_0 = _S275;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    q_3 = b0_1 + tid_7;
    for(;;)
    {
        if(q_3 < _S164)
        {
        }
        else
        {
            break;
        }
        uint j_15 = (&kernelContext_36)->lists_0[q_3];
        thread BlastBond_natural_0 _S289 = *((&kernelContext_36)->bonds_0+j_15);
        if(!(((&_S289)->health_0) > 0.0f))
        {
            uint _S290 = j_15 * 2U;
            packed_float4 _S291 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_36)->x_0+_S290) = _S291;
            *((&kernelContext_36)->x_0+(_S290 + 1U)) = _S291;
            q_3 = q_3 + 1024U;
            continue;
        }
        thread float3 ta_5;
        thread float3 tl_5;
        bond_term_1(&_S289, j_15, 6U, &ta_5, &tl_5, &kernelContext_36);
        uint _S292 = j_15 * 2U;
        *((&kernelContext_36)->x_0+_S292) = packed_float4(float4((float4(*((&kernelContext_36)->x_0+_S292)) ).xyz + ta_5, 0.0f)) ;
        uint _S293 = _S292 + 1U;
        *((&kernelContext_36)->x_0+_S293) = packed_float4(float4((float4(*((&kernelContext_36)->x_0+_S293)) ).xyz + tl_5, 0.0f)) ;
        q_3 = q_3 + 1024U;
    }
    if(converged_0)
    {
        ok_0 = !failed_0;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        p_4 = 1U;
    }
    else
    {
        p_4 = 0U;
    }
    if(ok_0)
    {
        ok_0 = warm_1;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        ok_0 = iteration_0 == 0U;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        q_4 = 4U;
    }
    else
    {
        q_4 = 0U;
    }
    uint flags_0 = p_4 | q_4;
    if(_S174)
    {
        *((&kernelContext_36)->report_0+k_19) = packed_uint4(uint4(iteration_0, flags_0, (as_type<uint>((num_0))), (as_type<uint>((delta2_0))))) ;
        if(((&kernelContext_36)->params_0->carry_0) != 0U)
        {
            failed_0 = !failed_0;
        }
        else
        {
            failed_0 = false;
        }
        if(failed_0)
        {
            converged_0 = !converged_0;
        }
        else
        {
            converged_0 = false;
        }
        if(converged_0)
        {
            if(carried_0)
            {
                p_4 = kept_0.y;
            }
            else
            {
                p_4 = 0U;
            }
            set_carry_0(k_19, _S165, p_4 + iteration_0, previous_0, &kernelContext_36);
        }
    }
    return;
}

