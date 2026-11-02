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

float norm2_0(float3 a_0, float3 b_0)
{
    return dot(a_0, a_0) + dot(b_0, b_0);
}

void set_v_0(uint n_1, uint v_2, float3 a_1, float3 l_0, KernelContext_0 thread* kernelContext_2)
{
    uint _S2 = ns_0(n_1, v_2, kernelContext_2);
    *(kernelContext_2->scratch_0+_S2) = packed_float4(float4(a_1, 0.0f)) ;
    *(kernelContext_2->scratch_0+(_S2 + 1U)) = packed_float4(float4(l_0, 0.0f)) ;
    return;
}

float3 stiff_0(uint j_0, uint which_0, float3 v_3, KernelContext_0 thread* kernelContext_3)
{
    uint _S3 = 6U * which_0;
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
        uint _S4 = _S3 + k_1;
        m_0[k_1] = (float4(*(kernelContext_3->stiffness_0+(j_0 * 6U + _S4 / 4U))) )[_S4 % 4U];
        k_1 = k_1 + 1U;
    }
    float _S5 = v_3.x;
    float _S6 = v_3.y;
    float _S7 = v_3.z;
    return float3(m_0[int(0)] * _S5 + m_0[int(3)] * _S6 + m_0[int(4)] * _S7, m_0[int(3)] * _S5 + m_0[int(1)] * _S6 + m_0[int(5)] * _S7, m_0[int(4)] * _S5 + m_0[int(5)] * _S6 + m_0[int(2)] * _S7);
}

float3 stiff_1(uint j_1, uint which_1, float3 v_4, KernelContext_0 thread* kernelContext_4)
{
    uint _S8 = 6U * which_1;
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
        uint _S9 = _S8 + k_2;
        m_1[k_2] = (float4(*(kernelContext_4->stiffness_0+(j_1 * 6U + _S9 / 4U))) )[_S9 % 4U];
        k_2 = k_2 + 1U;
    }
    float _S10 = v_4.x;
    float _S11 = v_4.y;
    float _S12 = v_4.z;
    return float3(m_1[int(0)] * _S10 + m_1[int(3)] * _S11 + m_1[int(4)] * _S12, m_1[int(3)] * _S10 + m_1[int(1)] * _S11 + m_1[int(5)] * _S12, m_1[int(4)] * _S10 + m_1[int(5)] * _S11 + m_1[int(2)] * _S12);
}

matrix<float,int(3),int(3)>  sym3_0(uint j_2, uint which_2, KernelContext_0 thread* kernelContext_5)
{
    float3 _S13 = stiff_1(j_2, which_2, float3(1.0f, 0.0f, 0.0f), kernelContext_5);
    float3 _S14 = stiff_1(j_2, which_2, float3(0.0f, 1.0f, 0.0f), kernelContext_5);
    float3 _S15 = stiff_1(j_2, which_2, float3(0.0f, 0.0f, 1.0f), kernelContext_5);
    return matrix<float,int(3),int(3)> (_S13.x, _S14.x, _S15.x, _S13.y, _S14.y, _S15.y, _S13.z, _S14.z, _S15.z);
}

matrix<float,int(3),int(3)>  skew_0(float3 o_0)
{
    float _S16 = o_0.z;
    float _S17 = o_0.y;
    float _S18 = o_0.x;
    return matrix<float,int(3),int(3)> (0.0f, - _S16, _S17, _S16, 0.0f, - _S18, - _S17, _S18, 0.0f);
}

bool build_full_block_0(uint node_0, KernelContext_0 thread* kernelContext_6)
{
    uint c_0;
    uint k_3;
    float s_3;
    BlastNode_natural_0 n_2 = kernelContext_6->nodes_0[node_0];
    matrix<float,int(3),int(3)>  aa_0 = matrix<float,int(3),int(3)> (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    uint r_0 = n_2.ref_begin_0;
    matrix<float,int(3),int(3)>  aa_1 = aa_0;
    matrix<float,int(3),int(3)>  al_0 = aa_0;
    matrix<float,int(3),int(3)>  ll_0 = aa_0;
    for(;;)
    {
        if(r_0 < (n_2.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_0 = kernelContext_6->refs_0[r_0];
        uint j_3 = ref_0 & 2147483647U;
        BlastBond_natural_0 device* _S19 = kernelContext_6->bonds_0+j_3;
        BlastBond_natural_0 b_1 = *_S19;
        if(!(((*_S19).health_0) > 0.0f))
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
        float _S20 = b_1.s_0;
        float s2_0 = _S20 * _S20;
        matrix<float,int(3),int(3)>  _S21 = sym3_0(j_3, 3U, kernelContext_6);
        matrix<float,int(3),int(3)>  k_4 = skew_0(o_1);
        matrix<float,int(3),int(3)>  _S22 = sym3_0(j_3, 1U, kernelContext_6);
        matrix<float,int(3),int(3)>  _S23 = (((_S21) * (k_4)));
        matrix<float,int(3),int(3)>  _S24 = matrix<float,int(3),int(3)> (s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0);
        matrix<float,int(3),int(3)>  _S25 = al_0 + _slang_matrixCompMult(matrix<float,int(3),int(3)> (- s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0), _S23);
        matrix<float,int(3),int(3)>  _S26 = ll_0 + _slang_matrixCompMult(_S24, _S21);
        aa_1 = aa_1 + _slang_matrixCompMult(_S24, _S22 + (((transpose(k_4)) * (_S23))));
        al_0 = _S25;
        ll_0 = _S26;
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
            m_2[r_0][c_0] = aa_1[r_0][c_0] * n_2.w_ang_0 * n_2.w_ang_0;
            uint _S27 = c_0 + 3U;
            float _S28 = al_0[r_0][c_0] * n_2.w_ang_0 * n_2.w_lin_0;
            m_2[r_0][_S27] = _S28;
            m_2[_S27][r_0] = _S28;
            m_2[r_0 + 3U][_S27] = ll_0[r_0][c_0] * n_2.w_lin_0 * n_2.w_lin_0;
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
        uint _S29 = ns_0(node_0, 18U + q_0, kernelContext_6);
        uint _S30 = 4U * q_0;
        *(kernelContext_6->scratch_0+_S29) = packed_float4(float4(packed_0[_S30], packed_0[_S30 + 1U], packed_0[_S30 + 2U], packed_0[_S30 + 3U])) ;
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
            s_3 = m_2[r_0][c_0];
            for(;;)
            {
                if(k_3 < c_0)
                {
                }
                else
                {
                    break;
                }
                float s_4 = s_3 - l_1[r_0][k_3] * l_1[c_0][k_3];
                k_3 = k_3 + 1U;
                s_3 = s_4;
            }
            if(r_0 == c_0)
            {
                bool _S31;
                if(!(s_3 > 0.0f))
                {
                    _S31 = true;
                }
                else
                {
                    _S31 = !isfinite(s_3);
                }
                if(_S31)
                {
                    return false;
                }
                l_1[r_0][c_0] = sqrt(s_3);
            }
            else
            {
                l_1[r_0][c_0] = s_3 / l_1[c_0][c_0];
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
                s_3 = 1.0f;
            }
            else
            {
                s_3 = 0.0f;
            }
            k_3 = 0U;
            float s_5 = s_3;
            for(;;)
            {
                if(k_3 < r_0)
                {
                }
                else
                {
                    break;
                }
                float s_6 = s_5 - l_1[r_0][k_3] * y_0[k_3];
                k_3 = k_3 + 1U;
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
            k_3 = uint(r_1) + 1U;
            s_3 = y_0[r_1];
            for(;;)
            {
                if(k_3 < 6U)
                {
                }
                else
                {
                    break;
                }
                float s_7 = s_3 - l_1[k_3][r_1] * inv_0[k_3][c_0];
                k_3 = k_3 + 1U;
                s_3 = s_7;
            }
            inv_0[r_1][c_0] = s_3 / l_1[r_1][r_1];
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
        uint _S32 = ns_0(node_0, 24U + q_0, kernelContext_6);
        uint _S33 = 4U * q_0;
        *(kernelContext_6->scratch_0+_S32) = packed_float4(float4(packed_0[_S33], packed_0[_S33 + 1U], packed_0[_S33 + 2U], packed_0[_S33 + 3U])) ;
        q_0 = q_0 + 1U;
    }
    return true;
}

bool build_block_0(uint node_1, KernelContext_0 thread* kernelContext_7)
{
    if((kernelContext_7->params_0->rotation_0) == 2U)
    {
        bool _S34 = build_full_block_0(node_1, kernelContext_7);
        return _S34;
    }
    BlastNode_natural_0 n_3 = kernelContext_7->nodes_0[node_1];
    float3 _S35 = float3(0.0f) ;
    uint r_2 = n_3.ref_begin_0;
    float3 diag_0 = _S35;
    float3 offd_0 = _S35;
    float3 ks_0 = _S35;
    float cs_0 = 0.0f;
    for(;;)
    {
        if(r_2 < (n_3.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_1 = kernelContext_7->refs_0[r_2];
        uint j_4 = ref_1 & 2147483647U;
        BlastBond_natural_0 device* _S36 = kernelContext_7->bonds_0+j_4;
        BlastBond_natural_0 b_2 = *_S36;
        if(!(((*_S36).health_0) > 0.0f))
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
        float _S37 = b_2.s_0;
        float s2_1 = _S37 * _S37;
        float3 _S38 = float3(1.0f) ;
        float3 wd_0;
        float3 wo_0;
        if((kernelContext_7->params_0->rotation_0) != 0U)
        {
            float3 _S39 = stiff_0(j_4, 1U, float3(1.0f, 0.0f, 0.0f), kernelContext_7);
            float3 _S40 = stiff_0(j_4, 1U, float3(0.0f, 1.0f, 0.0f), kernelContext_7);
            float3 _S41 = stiff_0(j_4, 1U, float3(0.0f, 0.0f, 1.0f), kernelContext_7);
            float3 _S42 = float3(_S40.x, _S41.x, _S41.y);
            wd_0 = float3(_S39.x, _S40.y, _S41.z);
            wo_0 = _S42;
        }
        else
        {
            wd_0 = _S38;
            wo_0 = _S35;
        }
        float3 _S43 = float3(s2_1) ;
        float _S44 = o_2.x;
        float _S45 = o_2.y;
        float _S46 = o_2.z;
        float3 offd_1 = offd_0 + _S43 * (wo_0 - float3(_S44 * _S45, _S44 * _S46, _S45 * _S46));
        float3 ks_1 = ks_0 + _S43 * o_2;
        float cs_1 = cs_0 + s2_1;
        diag_0 = diag_0 + _S43 * (wd_0 + float3(dot(o_2, o_2))  - o_2 * o_2);
        offd_0 = offd_1;
        ks_0 = ks_1;
        cs_0 = cs_1;
        r_2 = r_2 + 1U;
    }
    float _S47 = n_3.w_ang_0;
    float da_0 = _S47 * _S47;
    float c_1 = n_3.w_lin_0 * n_3.w_lin_0 * cs_0;
    float3 k_5 = ks_0 * float3((n_3.w_ang_0 * n_3.w_lin_0)) ;
    float4 a0_0 = float4(diag_0 * float3(da_0) , offd_0.x * da_0);
    float _S48 = offd_0.y * da_0;
    float _S49 = offd_0.z * da_0;
    float4 a1_0 = float4(_S48, _S49, c_1, 0.0f);
    uint _S50 = ns_0(node_1, 18U, kernelContext_7);
    *(kernelContext_7->scratch_0+_S50) = packed_float4(a0_0) ;
    *(kernelContext_7->scratch_0+(_S50 + 1U)) = packed_float4(a1_0) ;
    *(kernelContext_7->scratch_0+(_S50 + 2U)) = packed_float4(float4(k_5, 0.0f)) ;
    if(!(c_1 > 0.0f))
    {
        return false;
    }
    float ic_0 = 1.0f / c_1;
    float kk_0 = dot(k_5, k_5);
    float _S51 = k_5.x;
    float sxx_0 = a0_0.x + (_S51 * _S51 - kk_0) * ic_0;
    float _S52 = k_5.y;
    float syy_0 = a0_0.y + (_S52 * _S52 - kk_0) * ic_0;
    float _S53 = k_5.z;
    float szz_0 = a0_0.z + (_S53 * _S53 - kk_0) * ic_0;
    float sxy_0 = a0_0.w + _S51 * _S52 * ic_0;
    float sxz_0 = _S48 + _S51 * _S53 * ic_0;
    float syz_0 = _S49 + _S52 * _S53 * ic_0;
    float cxx_0 = syy_0 * szz_0 - syz_0 * syz_0;
    float cxy_0 = sxz_0 * syz_0 - sxy_0 * szz_0;
    float cxz_0 = sxy_0 * syz_0 - sxz_0 * syy_0;
    float det_0 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    bool _S54;
    if(!(det_0 > 0.0f))
    {
        _S54 = true;
    }
    else
    {
        _S54 = !isfinite(det_0);
    }
    if(_S54)
    {
        return false;
    }
    float id_0 = 1.0f / det_0;
    float cyy_0 = sxx_0 * szz_0 - sxz_0 * sxz_0;
    float cyz_0 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    float czz_0 = sxx_0 * syy_0 - sxy_0 * sxy_0;
    uint _S55 = ns_0(node_1, 21U, kernelContext_7);
    *(kernelContext_7->scratch_0+_S55) = packed_float4(float4(cxx_0 * id_0, cyy_0 * id_0, czz_0 * id_0, cxy_0 * id_0)) ;
    *(kernelContext_7->scratch_0+(_S55 + 1U)) = packed_float4(float4(cxz_0 * id_0, cyz_0 * id_0, ic_0, 0.0f)) ;
    *(kernelContext_7->scratch_0+(_S55 + 2U)) = packed_float4(float4(k_5 * float3(ic_0) , 0.0f)) ;
    return true;
}

uint blast_frame_0(uint node_count_1, uint k_6)
{
    return node_count_1 + 9U * k_6;
}

uint frame_word_0(uint k_7, uint w_1, KernelContext_0 thread* kernelContext_8)
{
    return (uint4(*(kernelContext_8->islands_0+(blast_frame_0(kernelContext_8->params_0->node_count_0, k_7) + w_1 / 4U))) )[w_1 % 4U];
}

uint frame_word_1(uint k_8, uint w_2, KernelContext_0 thread* kernelContext_9)
{
    return (uint4(*(kernelContext_9->islands_0+(blast_frame_0(kernelContext_9->params_0->node_count_0, k_8) + w_2 / 4U))) )[w_2 % 4U];
}

uint4 group_uniform4_0(uint4 value_1, uint thread_1)
{
    return value_1;
}

uint bs_0(uint j_5, KernelContext_0 thread* kernelContext_10)
{
    return kernelContext_10->params_0->node_count_0 * kernelContext_10->params_0->node_vectors_0 + 2U * j_5;
}

float3 va_0(uint n_4, uint v_5, KernelContext_0 thread* kernelContext_11)
{
    uint _S56 = ns_0(n_4, v_5, kernelContext_11);
    return (float4(*(kernelContext_11->scratch_0+_S56)) ).xyz;
}

float3 vl_0(uint n_5, uint v_6, KernelContext_0 thread* kernelContext_12)
{
    uint _S57 = ns_0(n_5, v_6, kernelContext_12);
    return (float4(*(kernelContext_12->scratch_0+(_S57 + 1U))) ).xyz;
}

float3 rot_a_0(uint j_6, float3 v_7, KernelContext_0 thread* kernelContext_13)
{
    float3 _S58;
    if((kernelContext_13->params_0->rotation_0) != 0U)
    {
        float3 _S59 = stiff_0(j_6, 0U, v_7, kernelContext_13);
        _S58 = _S59;
    }
    else
    {
        _S58 = v_7;
    }
    return _S58;
}

float3 rot_l_0(uint j_7, float3 v_8, KernelContext_0 thread* kernelContext_14)
{
    float3 _S60;
    if((kernelContext_14->params_0->rotation_0) == 2U)
    {
        float3 _S61 = stiff_0(j_7, 2U, v_8, kernelContext_14);
        _S60 = _S61;
    }
    else
    {
        _S60 = v_8;
    }
    return _S60;
}

void bond_term_0(const BlastBond_natural_0 thread* b_3, uint j_8, uint v_9, float3 thread* ta_0, float3 thread* tl_0, KernelContext_0 thread* kernelContext_15)
{
    uint _S62 = b_3->n0_0;
    BlastNode_natural_0 m0_0 = kernelContext_15->nodes_0[b_3->n0_0];
    uint _S63 = b_3->n1_0;
    BlastNode_natural_0 m1_0 = kernelContext_15->nodes_0[b_3->n1_0];
    float3 _S64 = va_0(b_3->n0_0, v_9, kernelContext_15);
    float3 a0_1 = _S64 * float3(m0_0.w_ang_0) ;
    float3 _S65 = vl_0(_S62, v_9, kernelContext_15);
    float3 l0_0 = _S65 * float3(m0_0.w_lin_0) ;
    float3 _S66 = va_0(_S63, v_9, kernelContext_15);
    float3 a1_1 = _S66 * float3(m1_0.w_ang_0) ;
    float3 _S67 = vl_0(_S63, v_9, kernelContext_15);
    float3 l1_0 = _S67 * float3(m1_0.w_lin_0) ;
    float3 _S68 = rot_a_0(j_8, a0_1 - a1_1, kernelContext_15);
    float3 _S69 = float3(b_3->s_0) ;
    *ta_0 = _S68 * _S69;
    float3 _S70 = rot_l_0(j_8, l0_0 - l1_0 + (cross((float4(b_3->o0_0) ).xyz, a0_1) - cross((float4(b_3->o1_0) ).xyz, a1_1)), kernelContext_15);
    *tl_0 = _S70 * _S69;
    return;
}

void bond_term_1(const BlastBond_natural_0 thread* b_4, uint j_9, uint v_10, float3 thread* ta_1, float3 thread* tl_1, KernelContext_0 thread* kernelContext_16)
{
    uint _S71 = b_4->n0_0;
    BlastNode_natural_0 m0_1 = kernelContext_16->nodes_0[b_4->n0_0];
    uint _S72 = b_4->n1_0;
    BlastNode_natural_0 m1_1 = kernelContext_16->nodes_0[b_4->n1_0];
    float3 _S73 = va_0(b_4->n0_0, v_10, kernelContext_16);
    float3 a0_2 = _S73 * float3(m0_1.w_ang_0) ;
    float3 _S74 = vl_0(_S71, v_10, kernelContext_16);
    float3 l0_1 = _S74 * float3(m0_1.w_lin_0) ;
    float3 _S75 = va_0(_S72, v_10, kernelContext_16);
    float3 a1_2 = _S75 * float3(m1_1.w_ang_0) ;
    float3 _S76 = vl_0(_S72, v_10, kernelContext_16);
    float3 l1_1 = _S76 * float3(m1_1.w_lin_0) ;
    float3 _S77 = rot_a_0(j_9, a0_2 - a1_2, kernelContext_16);
    float3 _S78 = float3(b_4->s_0) ;
    *ta_1 = _S77 * _S78;
    float3 _S79 = rot_l_0(j_9, l0_1 - l1_1 + (cross((float4(b_4->o0_0) ).xyz, a0_2) - cross((float4(b_4->o1_0) ).xyz, a1_2)), kernelContext_16);
    *tl_1 = _S79 * _S78;
    return;
}

void gather_0(uint node_2, bool from_t_0, bool with_u_0, float3 thread* ang_0, float3 thread* lin_0, KernelContext_0 thread* kernelContext_17)
{
    BlastNode_natural_0 n_6 = kernelContext_17->nodes_0[node_2];
    float3 _S80 = float3(0.0f) ;
    uint r_3 = n_6.ref_begin_0;
    float3 acc_a_0 = _S80;
    float3 acc_l_0 = _S80;
    for(;;)
    {
        if(r_3 < (n_6.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_2 = kernelContext_17->refs_0[r_3];
        uint j_10 = ref_2 & 2147483647U;
        thread BlastBond_natural_0 _S81 = *(kernelContext_17->bonds_0+j_10);
        if(!(((&_S81)->health_0) > 0.0f))
        {
            r_3 = r_3 + 1U;
            continue;
        }
        float3 ya_0;
        float3 yl_0;
        if(from_t_0)
        {
            uint _S82 = bs_0(j_10, kernelContext_17);
            float3 _S83 = (float4(*(kernelContext_17->scratch_0+(_S82 + 1U))) ).xyz;
            ya_0 = (float4(*(kernelContext_17->scratch_0+_S82)) ).xyz;
            yl_0 = _S83;
        }
        else
        {
            uint _S84 = j_10 * 2U;
            float3 ya_1 = (float4(*(kernelContext_17->x_0+_S84)) ).xyz;
            float3 yl_1 = (float4(*(kernelContext_17->x_0+(_S84 + 1U))) ).xyz;
            if(with_u_0)
            {
                thread float3 ta_2;
                thread float3 tl_2;
                bond_term_0(&_S81, j_10, 6U, &ta_2, &tl_2, kernelContext_17);
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
        float3 _S85 = rot_a_0(j_10, ya_0, kernelContext_17);
        float3 _S86 = float3((&_S81)->s_0) ;
        float3 ya_2 = _S85 * _S86;
        float3 _S87 = rot_l_0(j_10, yl_0, kernelContext_17);
        float3 yl_3 = _S87 * _S86;
        float3 acc_a_1;
        float3 acc_l_1;
        if((ref_2 & 2147483648U) != 0U)
        {
            float3 acc_l_2 = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross((float4((&_S81)->o1_0) ).xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            float3 acc_l_3 = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross((float4((&_S81)->o0_0) ).xyz, yl_3));
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

float rebuild_0(uint tid_1, uint n0_1, uint n1_1, bool with_u_1, KernelContext_0 thread* kernelContext_18)
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
        uint i_0 = kernelContext_18->lists_0[p_0];
        thread float3 a_2;
        thread float3 l_2;
        gather_0(i_0, false, with_u_1, &a_2, &l_2, kernelContext_18);
        float3 _S88 = va_0(i_0, 14U, kernelContext_18);
        float3 ra_0 = _S88 - a_2;
        float3 _S89 = vl_0(i_0, 14U, kernelContext_18);
        float3 rl_0 = _S89 - l_2;
        set_v_0(i_0, 0U, ra_0, rl_0, kernelContext_18);
        float s_9 = s_8 + norm2_0(ra_0, rl_0);
        p_0 = p_0 + 1024U;
        s_8 = s_9;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    float2 _S90 = blast_group_sum_0(tid_1, float2(s_8, 0.0f), kernelContext_18);
    return _S90.x;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

float frame_float_0(uint k_9, uint w_3, KernelContext_0 thread* kernelContext_19)
{
    uint _S91 = frame_word_1(k_9, w_3, kernelContext_19);
    return (as_type<float>((_S91)));
}

float frame_float_1(uint k_10, uint w_4, KernelContext_0 thread* kernelContext_20)
{
    uint _S92 = frame_word_1(k_10, w_4, kernelContext_20);
    return (as_type<float>((_S92)));
}

float3 frame_axis_0(uint k_11, uint rotations_0, uint kk_1, KernelContext_0 thread* kernelContext_21)
{
    if(rotations_0 == 1U)
    {
        float _S93 = frame_float_1(k_11, 4U, kernelContext_21);
        float _S94 = frame_float_1(k_11, 5U, kernelContext_21);
        float _S95 = frame_float_1(k_11, 6U, kernelContext_21);
        return float3(_S93, _S94, _S95);
    }
    float _S96;
    if(kk_1 == 0U)
    {
        _S96 = 1.0f;
    }
    else
    {
        _S96 = 0.0f;
    }
    float _S97;
    if(kk_1 == 1U)
    {
        _S97 = 1.0f;
    }
    else
    {
        _S97 = 0.0f;
    }
    float _S98;
    if(kk_1 == 2U)
    {
        _S98 = 1.0f;
    }
    else
    {
        _S98 = 0.0f;
    }
    return float3(_S96, _S97, _S98);
}

void frame_terms_0(uint k_12, uint rotations_1, uint i_1, uint v_11, array<float, int(6)> thread* s_10, KernelContext_0 thread* kernelContext_22)
{
    BlastNode_natural_0 n_7 = kernelContext_22->nodes_0[i_1];
    float3 _S99 = vl_0(i_1, v_11, kernelContext_22);
    float3 lin_1 = _S99 / float3(n_7.w_lin_0) ;
    float3 _S100 = va_0(i_1, v_11, kernelContext_22);
    float3 _S101 = _S100 / float3(n_7.w_ang_0)  - cross((float4(n_7.position_0) ).xyz, lin_1);
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
        float3 _S102 = frame_axis_0(k_12, rotations_1, kk_2, kernelContext_22);
        (*s_10)[kk_2] = (*s_10)[kk_2] + dot(_S102, _S101);
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

void frame_solve_0(uint k_13, uint rotations_2, const array<float, int(6)> thread* sum_0, float3 thread* omega_0, float3 thread* t_0, KernelContext_0 thread* kernelContext_23)
{
    uint j_12;
    uint d_0 = rotations_2 + 3U;
    thread array<float, int(6)> c_2;
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
        c_2[z_1] = 0.0f;
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
        float _S103 = (*sum_0)[row_0];
        float _S104 = frame_float_0(k_13, 8U + row_0, kernelContext_23);
        float _S105 = _S103 * _S104;
        j_12 = 0U;
        float value_2 = _S105;
        for(;;)
        {
            if(j_12 < row_0)
            {
            }
            else
            {
                break;
            }
            float _S106 = frame_float_0(k_13, 14U + row_0 * (row_0 + 1U) / 2U + j_12, kernelContext_23);
            float value_3 = value_2 - _S106 * c_2[j_12];
            j_12 = j_12 + 1U;
            value_2 = value_3;
        }
        uint _S107 = row_0 + 1U;
        float _S108 = frame_float_0(k_13, 14U + row_0 * _S107 / 2U + row_0, kernelContext_23);
        c_2[row_0] = value_2 / _S108;
        row_0 = _S107;
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
        uint _S109 = r_4 + 1U;
        j_12 = _S109;
        for(;;)
        {
            if(j_12 < d_0)
            {
            }
            else
            {
                break;
            }
            uint _S110 = j_12 + 1U;
            float _S111 = frame_float_0(k_13, 14U + j_12 * _S110 / 2U + r_4, kernelContext_23);
            c_2[r_4] = c_2[r_4] - _S111 * c_2[j_12];
            j_12 = _S110;
        }
        float _S112 = frame_float_0(k_13, 14U + r_4 * _S109 / 2U + r_4, kernelContext_23);
        c_2[r_4] = c_2[r_4] / _S112;
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
        float _S113 = frame_float_0(k_13, 8U + z_1, kernelContext_23);
        c_2[z_1] = c_2[z_1] * _S113;
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
        float3 _S114 = frame_axis_0(k_13, rotations_2, j_12, kernelContext_23);
        *omega_0 = *omega_0 + _S114 * float3(c_2[j_12]) ;
        j_12 = j_12 + 1U;
    }
    *t_0 = float3(c_2[rotations_2], c_2[rotations_2 + 1U], c_2[rotations_2 + 2U]);
    return;
}

void frame_apply_0(uint i_2, uint v_12, float3 omega_1, float3 t_1, KernelContext_0 thread* kernelContext_24)
{
    BlastNode_natural_0 n_8 = kernelContext_24->nodes_0[i_2];
    float3 _S115 = va_0(i_2, v_12, kernelContext_24);
    float3 _S116 = _S115 - omega_1 / float3(n_8.w_ang_0) ;
    float3 _S117 = vl_0(i_2, v_12, kernelContext_24);
    set_v_0(i_2, v_12, _S116, _S117 - (t_1 + cross((float4(n_8.position_0) ).xyz, omega_1)) / float3(n_8.w_lin_0) , kernelContext_24);
    return;
}

void project_0(uint tid_2, uint k_14, uint rotations_3, uint n0_2, uint n1_2, uint v_13, bool free_0, KernelContext_0 thread* kernelContext_25)
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
    uint _S118 = n0_2 + tid_2;
    uint p_1 = _S118;
    for(;;)
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_terms_0(k_14, rotations_3, kernelContext_25->lists_0[p_1], v_13, &s_11, kernelContext_25);
        p_1 = p_1 + 1024U;
    }
    float2 _S119 = blast_group_sum_0(tid_2, float2(s_11[int(0)], s_11[int(1)]), kernelContext_25);
    float2 _S120 = blast_group_sum_0(tid_2, float2(s_11[int(2)], s_11[int(3)]), kernelContext_25);
    float2 _S121 = blast_group_sum_0(tid_2, float2(s_11[int(4)], s_11[int(5)]), kernelContext_25);
    if(tid_2 == 0U)
    {
        float _S122 = _S119.y;
        float _S123 = _S120.x;
        float _S124 = _S120.y;
        float _S125 = _S121.x;
        float _S126 = _S121.y;
        thread array<float, int(6)> _S127;
        _S127[int(0)] = _S119.x;
        _S127[int(1)] = _S122;
        _S127[int(2)] = _S123;
        _S127[int(3)] = _S124;
        _S127[int(4)] = _S125;
        _S127[int(5)] = _S126;
        thread float3 omega_2;
        thread float3 t_2;
        frame_solve_0(k_14, rotations_3, &_S127, &omega_2, &t_2, kernelContext_25);
        *kernelContext_25->g_omega_0 = omega_2;
        *kernelContext_25->g_t_0 = t_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float3 _S128 = *kernelContext_25->g_omega_0;
    float3 _S129 = *kernelContext_25->g_t_0;
    p_1 = _S118;
    for(;;)
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_apply_0(kernelContext_25->lists_0[p_1], v_13, _S128, _S129, kernelContext_25);
        p_1 = p_1 + 1024U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return;
}

float bond_pass_0(uint tid_3, uint b0_0, uint b1_0, uint v_14, KernelContext_0 thread* kernelContext_26)
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
        uint j_13 = kernelContext_26->lists_0[q_1];
        thread BlastBond_natural_0 _S130 = *(kernelContext_26->bonds_0+j_13);
        float3 _S131 = float3(0.0f) ;
        thread float3 ta_3 = _S131;
        thread float3 tl_3 = _S131;
        if(((&_S130)->health_0) > 0.0f)
        {
            bond_term_0(&_S130, j_13, v_14, &ta_3, &tl_3, kernelContext_26);
        }
        float z_4 = z_3 + norm2_0(ta_3, tl_3);
        uint _S132 = bs_0(j_13, kernelContext_26);
        *(kernelContext_26->scratch_0+_S132) = packed_float4(float4(ta_3, 0.0f)) ;
        *(kernelContext_26->scratch_0+(_S132 + 1U)) = packed_float4(float4(tl_3, 0.0f)) ;
        q_1 = q_1 + 1024U;
        z_3 = z_4;
    }
    float2 _S133 = blast_group_sum_0(tid_3, float2(z_3, 0.0f), kernelContext_26);
    return _S133.x;
}

float residual_norm_0(uint tid_4, uint n0_3, uint n1_3, KernelContext_0 thread* kernelContext_27)
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
        uint i_3 = kernelContext_27->lists_0[p_2];
        float3 _S134 = va_0(i_3, 0U, kernelContext_27);
        float3 _S135 = vl_0(i_3, 0U, kernelContext_27);
        float s_13 = s_12 + norm2_0(_S134, _S135);
        p_2 = p_2 + 1024U;
        s_12 = s_13;
    }
    float2 _S136 = blast_group_sum_0(tid_4, float2(s_12, 0.0f), kernelContext_27);
    return _S136.x;
}

void full_apply_0(uint n_9, uint at_0, float3 a_3, float3 l_3, float3 thread* oa_0, float3 thread* ol_0, KernelContext_0 thread* kernelContext_28)
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
        uint _S137 = ns_0(n_9, at_0 + q_2, kernelContext_28);
        float4 _S138 = float4(*(kernelContext_28->scratch_0+_S137)) ;
        uint _S139 = 4U * q_2;
        m_3[_S139] = _S138.x;
        m_3[_S139 + 1U] = _S138.y;
        m_3[_S139 + 2U] = _S138.z;
        m_3[_S139 + 3U] = _S138.w;
        q_2 = q_2 + 1U;
    }
    array<float, int(6)> _S140 = { { a_3.x, a_3.y, a_3.z, l_3.x, l_3.y, l_3.z } };
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
        uint c_3 = 0U;
        float s_14 = 0.0f;
        for(;;)
        {
            if(c_3 < 6U)
            {
            }
            else
            {
                break;
            }
            if(r_5 >= c_3)
            {
                q_2 = r_5 * (r_5 + 1U) / 2U + c_3;
            }
            else
            {
                q_2 = c_3 * (c_3 + 1U) / 2U + r_5;
            }
            float s_15 = s_14 + m_3[q_2] * _S140[c_3];
            c_3 = c_3 + 1U;
            s_14 = s_15;
        }
        o_3[r_5] = s_14;
        r_5 = r_5 + 1U;
    }
    *oa_0 = float3(o_3[int(0)], o_3[int(1)], o_3[int(2)]);
    *ol_0 = float3(o_3[int(3)], o_3[int(4)], o_3[int(5)]);
    return;
}

float3 sym_mul_0(float4 s0_0, float4 s1_0, float3 v_15)
{
    float _S141 = v_15.x;
    float _S142 = s0_0.w;
    float _S143 = v_15.y;
    float _S144 = s1_0.x;
    float _S145 = v_15.z;
    float _S146 = s1_0.y;
    return float3(s0_0.x * _S141 + _S142 * _S143 + _S144 * _S145, _S142 * _S141 + s0_0.y * _S143 + _S146 * _S145, _S144 * _S141 + _S146 * _S143 + s0_0.z * _S145);
}

void inverse_apply_0(uint n_10, float3 a_4, float3 l_4, float3 thread* oa_1, float3 thread* ol_1, KernelContext_0 thread* kernelContext_29)
{
    if((kernelContext_29->params_0->rotation_0) == 2U)
    {
        full_apply_0(n_10, 24U, a_4, l_4, oa_1, ol_1, kernelContext_29);
        return;
    }
    uint _S147 = ns_0(n_10, 21U, kernelContext_29);
    float4 _S148 = float4(*(kernelContext_29->scratch_0+(_S147 + 1U))) ;
    float3 kc_0 = (float4(*(kernelContext_29->scratch_0+(_S147 + 2U))) ).xyz;
    float3 _S149 = sym_mul_0(float4(*(kernelContext_29->scratch_0+_S147)) , _S148, a_4 + cross(kc_0, l_4));
    *oa_1 = _S149;
    *ol_1 = l_4 * float3(_S148.z)  - cross(kc_0, _S149);
    return;
}

float node_pass_0(uint tid_5, uint n0_4, uint n1_4, uint out_v_0, KernelContext_0 thread* kernelContext_30)
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
        uint i_4 = kernelContext_30->lists_0[p_3];
        thread float3 a_5;
        thread float3 l_5;
        gather_0(i_4, true, false, &a_5, &l_5, kernelContext_30);
        set_v_0(i_4, out_v_0, a_5, l_5, kernelContext_30);
        float s_17 = s_16 + norm2_0(a_5, l_5);
        p_3 = p_3 + 1024U;
        s_16 = s_17;
    }
    return s_16;
}

void block_apply_0(uint n_11, float3 a_6, float3 l_6, float3 thread* oa_2, float3 thread* ol_2, KernelContext_0 thread* kernelContext_31)
{
    if((kernelContext_31->params_0->rotation_0) == 2U)
    {
        full_apply_0(n_11, 18U, a_6, l_6, oa_2, ol_2, kernelContext_31);
        return;
    }
    uint _S150 = ns_0(n_11, 18U, kernelContext_31);
    float4 _S151 = float4(*(kernelContext_31->scratch_0+(_S150 + 1U))) ;
    float3 k_15 = (float4(*(kernelContext_31->scratch_0+(_S150 + 2U))) ).xyz;
    *oa_2 = sym_mul_0(float4(*(kernelContext_31->scratch_0+_S150)) , _S151, a_6) - cross(k_15, l_6);
    *ol_2 = cross(k_15, a_6) + float3(_S151.z)  * l_6;
    return;
}

float group_max_0(uint tid_6, float v_16, KernelContext_0 thread* kernelContext_32)
{
    (*kernelContext_32->g_max_0)[tid_6] = v_16;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint k_16 = 512U;
    for(;;)
    {
        if(k_16 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_6 < k_16)
        {
            (*kernelContext_32->g_max_0)[tid_6] = max((*kernelContext_32->g_max_0)[tid_6], (*kernelContext_32->g_max_0)[tid_6 + k_16]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_16 = k_16 >> 1U;
    }
    if(tid_6 == 0U)
    {
        *kernelContext_32->g_max_out_0 = (*kernelContext_32->g_max_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S152 = kernelContext_32->g_max_out_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *_S152;
}

[[kernel]] void blast_native(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_1 [[buffer(0)]], packed_uint4 device* islands_1 [[buffer(1)]], packed_uint4 device* report_1 [[buffer(8)]], uint device* lists_1 [[buffer(9)]], packed_float4 device* accel_1 [[buffer(5)]], packed_float4 device* scratch_1 [[buffer(7)]], BlastNode_natural_0 device* nodes_1 [[buffer(2)]], uint device* refs_1 [[buffer(3)]], BlastBond_natural_0 device* bonds_1 [[buffer(4)]], packed_float4 device* stiffness_1 [[buffer(10)]], packed_float4 device* x_1 [[buffer(6)]])
{
    float force_start_0;
    float best_0;
    bool failed_0;
    uint q_3;
    bool ok_0;
    thread KernelContext_0 kernelContext_33;
    (&kernelContext_33)->params_0 = params_1;
    (&kernelContext_33)->islands_0 = islands_1;
    (&kernelContext_33)->report_0 = report_1;
    (&kernelContext_33)->lists_0 = lists_1;
    (&kernelContext_33)->accel_0 = accel_1;
    (&kernelContext_33)->scratch_0 = scratch_1;
    (&kernelContext_33)->nodes_0 = nodes_1;
    (&kernelContext_33)->refs_0 = refs_1;
    (&kernelContext_33)->bonds_0 = bonds_1;
    (&kernelContext_33)->stiffness_0 = stiffness_1;
    (&kernelContext_33)->x_0 = x_1;
    threadgroup array<float2, int(32)> g_blast_part_1;
    (&kernelContext_33)->g_blast_part_0 = &g_blast_part_1;
    threadgroup float3 g_omega_1;
    (&kernelContext_33)->g_omega_0 = &g_omega_1;
    threadgroup float3 g_t_1;
    (&kernelContext_33)->g_t_0 = &g_t_1;
    threadgroup array<float, int(1024)> g_max_1;
    (&kernelContext_33)->g_max_0 = &g_max_1;
    threadgroup float g_max_out_1;
    (&kernelContext_33)->g_max_out_0 = &g_max_out_1;
    uint tid_7 = thread_2.x;
    uint k_17 = group_0.x;
    if(k_17 >= (params_1->island_count_0))
    {
        return;
    }
    uint4 _S153 = uint4(*((&kernelContext_33)->islands_0+k_17)) ;
    uint n0_5 = _S153.x;
    uint n1_5 = _S153.y;
    uint b0_1 = _S153.z;
    uint _S154 = _S153.w;
    uint p_4;
    if((n1_5 - n0_5) > ((&kernelContext_33)->params_0->wide_min_0))
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
    uint4 _S155 = uint4(*((&kernelContext_33)->report_0+k_17)) ;
    uint last_flags_0 = group_uniform_0(_S155.y, tid_7);
    bool warm_1 = ((&kernelContext_33)->params_0->warm_0) != 0U;
    bool converged_0;
    if(warm_1)
    {
        converged_0 = (last_flags_0 & 4U) != 0U;
    }
    else
    {
        converged_0 = false;
    }
    float singular_0;
    if(converged_0)
    {
        uint _S156 = n0_5 + tid_7;
        singular_0 = 0.0f;
        p_4 = _S156;
        for(;;)
        {
            if(p_4 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_5 = (&kernelContext_33)->lists_0[p_4];
            uint _S157 = i_5 * 2U;
            float4 _S158 = float4(*((&kernelContext_33)->accel_0+_S157)) ;
            uint _S159 = ns_0(i_5, 16U, &kernelContext_33);
            if(any(_S158 != (float4(*((&kernelContext_33)->scratch_0+_S159)) )))
            {
                converged_0 = true;
            }
            else
            {
                converged_0 = any((float4(*((&kernelContext_33)->accel_0+(_S157 + 1U))) ) != (float4(*((&kernelContext_33)->scratch_0+(_S159 + 1U))) ));
            }
            if(converged_0)
            {
                singular_0 = 1.0f;
            }
            p_4 = p_4 + 1024U;
        }
        float2 _S160 = blast_group_sum_0(tid_7, float2(singular_0, 0.0f), &kernelContext_33);
        if((_S160.x) == 0.0f)
        {
            if(tid_7 == 0U)
            {
                *((&kernelContext_33)->report_0+k_17) = packed_uint4(uint4(0U, 7U, _S155.z, _S155.w)) ;
            }
            return;
        }
    }
    uint _S161 = n0_5 + tid_7;
    singular_0 = 0.0f;
    p_4 = _S161;
    float target_0 = 0.0f;
    for(;;)
    {
        if(p_4 < n1_5)
        {
        }
        else
        {
            break;
        }
        uint i_6 = (&kernelContext_33)->lists_0[p_4];
        BlastNode_natural_0 n_12 = (&kernelContext_33)->nodes_0[i_6];
        uint _S162 = i_6 * 2U;
        float4 _S163 = float4(*((&kernelContext_33)->accel_0+_S162)) ;
        float4 _S164 = float4(*((&kernelContext_33)->accel_0+(_S162 + 1U))) ;
        float3 _S165 = - _S163.xyz;
        if((n_12.w_ang_0) > 0.0f)
        {
            force_start_0 = 1.0f / n_12.w_ang_0;
        }
        else
        {
            force_start_0 = 1.0f;
        }
        float3 ba_0 = _S165 * float3(force_start_0) ;
        float3 _S166 = - _S164.xyz;
        if((n_12.w_lin_0) > 0.0f)
        {
            best_0 = (&kernelContext_33)->params_0->recip_length_0 / n_12.w_lin_0;
        }
        else
        {
            best_0 = (&kernelContext_33)->params_0->recip_length_0;
        }
        float3 bl_0 = _S166 * float3(best_0) ;
        float target_1 = target_0 + norm2_0(ba_0, bl_0);
        set_v_0(i_6, 14U, ba_0, bl_0, &kernelContext_33);
        set_v_0(i_6, 0U, ba_0, bl_0, &kernelContext_33);
        uint _S167 = ns_0(i_6, 16U, &kernelContext_33);
        *((&kernelContext_33)->scratch_0+_S167) = packed_float4(_S163) ;
        *((&kernelContext_33)->scratch_0+(_S167 + 1U)) = packed_float4(_S164) ;
        float3 _S168 = float3(0.0f) ;
        set_v_0(i_6, 6U, _S168, _S168, &kernelContext_33);
        set_v_0(i_6, 2U, _S168, _S168, &kernelContext_33);
        bool _S169 = build_block_0(i_6, &kernelContext_33);
        if(!_S169)
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
            if(q_4 < _S154)
            {
            }
            else
            {
                break;
            }
            uint _S170 = (&kernelContext_33)->lists_0[q_4] * 2U;
            packed_float4 _S171 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_33)->x_0+_S170) = _S171;
            *((&kernelContext_33)->x_0+(_S170 + 1U)) = _S171;
            q_4 = q_4 + 1024U;
        }
    }
    float2 _S172 = blast_group_sum_0(tid_7, float2(target_0, singular_0), &kernelContext_33);
    float delta2_0 = _S172.x * (&kernelContext_33)->params_0->tolerance_0 * (&kernelContext_33)->params_0->tolerance_0;
    uint _S173 = frame_word_0(k_17, 0U, &kernelContext_33);
    uint _S174 = frame_word_0(k_17, 1U, &kernelContext_33);
    uint _S175 = frame_word_0(k_17, 2U, &kernelContext_33);
    uint3 frame_0 = group_uniform4_0(uint4(_S173, _S174, _S175, 0U), tid_7).xyz;
    if((_S172.y) > 0.0f)
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = (frame_0.z) != 0U;
    }
    bool _S176 = (frame_0.x) == 0U;
    uint _S177 = frame_0.y;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if(!(delta2_0 > 0.0f))
    {
        q_4 = b0_1 + tid_7;
        for(;;)
        {
            if(q_4 < _S154)
            {
            }
            else
            {
                break;
            }
            uint _S178 = (&kernelContext_33)->lists_0[q_4] * 2U;
            packed_float4 _S179 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_33)->x_0+_S178) = _S179;
            *((&kernelContext_33)->x_0+(_S178 + 1U)) = _S179;
            q_4 = q_4 + 1024U;
        }
        if(tid_7 == 0U)
        {
            packed_uint4 device* _S180 = (&kernelContext_33)->report_0+k_17;
            if(warm_1)
            {
                p_4 = 4U;
            }
            else
            {
                p_4 = 0U;
            }
            *_S180 = packed_uint4(uint4(0U, 1U | p_4, (as_type<uint>((0.0f))), (as_type<uint>((delta2_0))))) ;
        }
        return;
    }
    if(warm_1)
    {
        float _S181 = rebuild_0(tid_7, n0_5, n1_5, false, &kernelContext_33);
    }
    float _S182 = float_bits_0(2139095040U);
    if(((&kernelContext_33)->params_0->force_tolerance_0) > 0.0f)
    {
        q_4 = b0_1 + tid_7;
        force_start_0 = 0.0f;
        for(;;)
        {
            if(q_4 < _S154)
            {
            }
            else
            {
                break;
            }
            uint _S183 = (&kernelContext_33)->lists_0[q_4] * 2U;
            float f_0 = force_start_0 + norm2_0((float4(*((&kernelContext_33)->x_0+_S183)) ).xyz, (float4(*((&kernelContext_33)->x_0+(_S183 + 1U))) ).xyz);
            q_4 = q_4 + 1024U;
            force_start_0 = f_0;
        }
        float2 _S184 = blast_group_sum_0(tid_7, float2(force_start_0, 0.0f), &kernelContext_33);
        force_start_0 = sqrt(_S184.x);
    }
    else
    {
        force_start_0 = 0.0f;
    }
    best_0 = _S182;
    q_4 = 0U;
    float step_0 = _S182;
    float previous_0 = 0.0f;
    float num_0 = 0.0f;
    uint iteration_0 = 0U;
    float travel_0 = 0.0f;
    for(;;)
    {
        if(!converged_0)
        {
            failed_0 = iteration_0 < ((&kernelContext_33)->params_0->max_iterations_0);
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
            bool _S185 = converged_0;
            converged_0 = false;
            failed_0 = _S185;
            break;
        }
        project_0(tid_7, k_17, _S177, n0_5, n1_5, 0U, _S176, &kernelContext_33);
        float _S186 = bond_pass_0(tid_7, b0_1, _S154, 0U, &kernelContext_33);
        bool _S187 = _S186 <= delta2_0;
        if(_S187)
        {
            float _S188 = residual_norm_0(tid_7, n0_5, n1_5, &kernelContext_33);
            ok_0 = !(_S188 <= delta2_0);
        }
        else
        {
            ok_0 = false;
        }
        bool _S189 = iteration_0 > 0U;
        bool _S190;
        if(_S189)
        {
            _S190 = true;
        }
        else
        {
            _S190 = warm_1;
        }
        bool _S191;
        if(_S190)
        {
            _S191 = _S187;
        }
        else
        {
            _S191 = false;
        }
        bool _S192;
        if(_S191)
        {
            _S192 = !ok_0;
        }
        else
        {
            _S192 = false;
        }
        bool hold_0;
        float num_1;
        float previous_1;
        if(_S192)
        {
            float _S193 = rebuild_0(tid_7, n0_5, n1_5, true, &kernelContext_33);
            project_0(tid_7, k_17, _S177, n0_5, n1_5, 0U, _S176, &kernelContext_33);
            float _S194 = bond_pass_0(tid_7, b0_1, _S154, 0U, &kernelContext_33);
            if(_S194 <= delta2_0)
            {
                float _S195 = residual_norm_0(tid_7, n0_5, n1_5, &kernelContext_33);
                hold_0 = !(_S195 <= delta2_0);
            }
            else
            {
                hold_0 = false;
            }
            num_1 = _S194;
            previous_1 = 0.0f;
        }
        else
        {
            num_1 = _S186;
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
                bool _S196 = converged_0;
                converged_0 = false;
                failed_0 = _S196;
                num_0 = num_1;
                break;
            }
        }
        bool _S197;
        if(num_1 <= delta2_0)
        {
            _S197 = !hold_0;
        }
        else
        {
            _S197 = false;
        }
        if(_S197)
        {
            bool _S198 = converged_0;
            converged_0 = true;
            failed_0 = _S198;
            num_0 = num_1;
            break;
        }
        bool _S199;
        if(((&kernelContext_33)->params_0->force_tolerance_0) > 0.0f)
        {
            _S199 = _S189;
        }
        else
        {
            _S199 = false;
        }
        bool _S200;
        if(_S199)
        {
            _S200 = step_0 <= ((&kernelContext_33)->params_0->force_tolerance_0 * (force_start_0 + travel_0));
        }
        else
        {
            _S200 = false;
        }
        float mloc_0;
        if(_S200)
        {
            q_3 = b0_1 + tid_7;
            mloc_0 = 0.0f;
            for(;;)
            {
                if(q_3 < _S154)
                {
                }
                else
                {
                    break;
                }
                uint j_14 = (&kernelContext_33)->lists_0[q_3];
                thread BlastBond_natural_0 _S201 = *((&kernelContext_33)->bonds_0+j_14);
                thread float3 ta_4;
                thread float3 tl_4;
                bond_term_1(&_S201, j_14, 6U, &ta_4, &tl_4, &kernelContext_33);
                uint _S202 = j_14 * 2U;
                float f_1 = mloc_0 + norm2_0((float4(*((&kernelContext_33)->x_0+_S202)) ).xyz + ta_4, (float4(*((&kernelContext_33)->x_0+(_S202 + 1U))) ).xyz + tl_4);
                q_3 = q_3 + 1024U;
                mloc_0 = f_1;
            }
            float _S203 = (&kernelContext_33)->params_0->force_tolerance_0;
            float2 _S204 = blast_group_sum_0(tid_7, float2(mloc_0, 0.0f), &kernelContext_33);
            if(step_0 <= (_S203 * sqrt(_S204.x)))
            {
                bool _S205 = converged_0;
                converged_0 = true;
                failed_0 = _S205;
                num_0 = num_1;
                break;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        bool _S206;
        if(iteration_0 == 0U)
        {
            _S206 = ((&kernelContext_33)->params_0->first_polynomial_0) == 0U;
        }
        else
        {
            _S206 = false;
        }
        if(_S206)
        {
            p_4 = _S161;
            for(;;)
            {
                if(p_4 < n1_5)
                {
                }
                else
                {
                    break;
                }
                uint i_7 = (&kernelContext_33)->lists_0[p_4];
                float3 _S207 = va_0(i_7, 0U, &kernelContext_33);
                float3 _S208 = vl_0(i_7, 0U, &kernelContext_33);
                set_v_0(i_7, 8U, _S207, _S208, &kernelContext_33);
                p_4 = p_4 + 1024U;
            }
        }
        else
        {
            p_4 = _S161;
            for(;;)
            {
                if(p_4 < n1_5)
                {
                }
                else
                {
                    break;
                }
                uint i_8 = (&kernelContext_33)->lists_0[p_4];
                float3 _S209 = va_0(i_8, 0U, &kernelContext_33);
                float3 _S210 = vl_0(i_8, 0U, &kernelContext_33);
                thread float3 a_7;
                thread float3 l_7;
                inverse_apply_0(i_8, _S209, _S210, &a_7, &l_7, &kernelContext_33);
                set_v_0(i_8, 10U, a_7, l_7, &kernelContext_33);
                p_4 = p_4 + 1024U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            float _S211 = bond_pass_0(tid_7, b0_1, _S154, 10U, &kernelContext_33);
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            float _S212 = node_pass_0(tid_7, n0_5, n1_5, 12U, &kernelContext_33);
            q_3 = _S161;
            for(;;)
            {
                if(q_3 < n1_5)
                {
                }
                else
                {
                    break;
                }
                uint i_9 = (&kernelContext_33)->lists_0[q_3];
                float3 _S213 = va_0(i_9, 10U, &kernelContext_33);
                float3 _S214 = vl_0(i_9, 10U, &kernelContext_33);
                thread float3 sa_0;
                thread float3 sl_0;
                block_apply_0(i_9, _S213, _S214, &sa_0, &sl_0, &kernelContext_33);
                float3 _S215 = va_0(i_9, 12U, &kernelContext_33);
                float3 _S216 = _S215 - sa_0;
                float3 _S217 = vl_0(i_9, 12U, &kernelContext_33);
                thread float3 ma_0;
                thread float3 ml_0;
                inverse_apply_0(i_9, _S216, _S217 - sl_0, &ma_0, &ml_0, &kernelContext_33);
                float3 _S218 = va_0(i_9, 10U, &kernelContext_33);
                float3 _S219 = float3(1.68946564197540283f) ;
                float3 _S220 = float3(1.5220409631729126f) ;
                float3 _S221 = _S219 * _S218 - _S220 * ma_0;
                float3 _S222 = vl_0(i_9, 10U, &kernelContext_33);
                set_v_0(i_9, 8U, _S221, _S219 * _S222 - _S220 * ml_0, &kernelContext_33);
                q_3 = q_3 + 1024U;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        project_0(tid_7, k_17, _S177, n0_5, n1_5, 8U, _S176, &kernelContext_33);
        mloc_0 = 0.0f;
        p_4 = _S161;
        for(;;)
        {
            if(p_4 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_10 = (&kernelContext_33)->lists_0[p_4];
            float3 _S223 = va_0(i_10, 8U, &kernelContext_33);
            float3 a_8 = abs(_S223);
            float3 _S224 = vl_0(i_10, 8U, &kernelContext_33);
            float3 l_8 = abs(_S224);
            float _S225 = max(mloc_0, max(max(max(a_8.x, a_8.y), max(a_8.z, l_8.x)), max(l_8.y, l_8.z)));
            uint p_5 = p_4 + 1024U;
            mloc_0 = _S225;
            p_4 = p_5;
        }
        float _S226 = group_max_0(tid_7, mloc_0, &kernelContext_33);
        bool _S227;
        if(!(_S226 > 0.0f))
        {
            _S227 = true;
        }
        else
        {
            _S227 = !isfinite(_S226);
        }
        if(_S227)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_1;
            break;
        }
        float _S228 = 1.0f / _S226;
        q_3 = _S161;
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
            uint i_11 = (&kernelContext_33)->lists_0[q_3];
            float3 _S229 = va_0(i_11, 8U, &kernelContext_33);
            float3 _S230 = float3(_S228) ;
            float3 ga_0 = _S229 * _S230;
            float3 _S231 = vl_0(i_11, 8U, &kernelContext_33);
            float3 gn_0 = _S231 * _S230;
            set_v_0(i_11, 8U, ga_0, gn_0, &kernelContext_33);
            float3 _S232 = va_0(i_11, 0U, &kernelContext_33);
            float _S233 = dot(_S232, ga_0);
            float3 _S234 = vl_0(i_11, 0U, &kernelContext_33);
            float gl_1 = gl_0 + (_S233 + dot(_S234, gn_0));
            q_3 = q_3 + 1024U;
            gl_0 = gl_1;
        }
        float2 _S235 = blast_group_sum_0(tid_7, float2(gl_0, 0.0f), &kernelContext_33);
        float gamma_0 = _S235.x;
        bool _S236 = gamma_0 > 0.0f;
        bool _S237;
        if(!_S236)
        {
            _S237 = true;
        }
        else
        {
            _S237 = !isfinite(gamma_0);
        }
        if(_S237)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_1;
            break;
        }
        bool _S238;
        if(iteration_0 > 1U)
        {
            _S238 = true;
        }
        else
        {
            if(((&kernelContext_33)->params_0->first_polynomial_0) != 0U)
            {
                _S238 = _S189;
            }
            else
            {
                _S238 = false;
            }
        }
        bool _S239;
        if(_S238)
        {
            _S239 = previous_1 > 0.0f;
        }
        else
        {
            _S239 = false;
        }
        if(_S239)
        {
            singular_0 = gamma_0 / previous_1;
        }
        else
        {
            singular_0 = 0.0f;
        }
        uint p_6 = _S161;
        for(;;)
        {
            if(p_6 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_12 = (&kernelContext_33)->lists_0[p_6];
            if(singular_0 == 0.0f)
            {
                float3 _S240 = va_0(i_12, 8U, &kernelContext_33);
                float3 _S241 = vl_0(i_12, 8U, &kernelContext_33);
                set_v_0(i_12, 2U, _S240, _S241, &kernelContext_33);
            }
            else
            {
                float3 _S242 = va_0(i_12, 8U, &kernelContext_33);
                float3 _S243 = va_0(i_12, 2U, &kernelContext_33);
                float3 _S244 = _S242 + _S243 * float3(singular_0) ;
                float3 _S245 = vl_0(i_12, 8U, &kernelContext_33);
                float3 _S246 = vl_0(i_12, 2U, &kernelContext_33);
                set_v_0(i_12, 2U, _S244, _S245 + _S246 * float3(singular_0) , &kernelContext_33);
            }
            p_6 = p_6 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float _S247 = bond_pass_0(tid_7, b0_1, _S154, 2U, &kernelContext_33);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float _S248 = node_pass_0(tid_7, n0_5, n1_5, 4U, &kernelContext_33);
        bool _S249;
        if(((&kernelContext_33)->params_0->force_tolerance_0) > 0.0f)
        {
            if(_S247 > 0.0f)
            {
                _S249 = _S236;
            }
            else
            {
                _S249 = false;
            }
            float step_1;
            if(_S249)
            {
                step_1 = gamma_0 / sqrt(_S247);
            }
            else
            {
                step_1 = _S182;
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
            bool _S250;
            if(_S189)
            {
                _S250 = true;
            }
            else
            {
                _S250 = ((&kernelContext_33)->params_0->first_polynomial_0) != 0U;
            }
            float step_2;
            if(!_S250)
            {
                step_2 = _S182;
            }
            else
            {
                step_2 = step_1;
            }
            step_0 = step_2;
            travel_0 = travel_1;
        }
        uint _S251 = iteration_0 + 1U;
        if(!(_S247 > 0.0f))
        {
            _S249 = true;
        }
        else
        {
            _S249 = !isfinite(_S247);
        }
        if(_S249)
        {
            bool _S252 = converged_0;
            converged_0 = false;
            failed_0 = _S252;
            num_0 = num_1;
            iteration_0 = _S251;
            break;
        }
        float _S253 = gamma_0 / _S247;
        uint p_7 = _S161;
        for(;;)
        {
            if(p_7 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_13 = (&kernelContext_33)->lists_0[p_7];
            float3 _S254 = va_0(i_13, 6U, &kernelContext_33);
            float3 _S255 = va_0(i_13, 2U, &kernelContext_33);
            float3 _S256 = float3(_S253) ;
            float3 _S257 = _S254 + _S255 * _S256;
            float3 _S258 = vl_0(i_13, 6U, &kernelContext_33);
            float3 _S259 = vl_0(i_13, 2U, &kernelContext_33);
            set_v_0(i_13, 6U, _S257, _S258 + _S259 * _S256, &kernelContext_33);
            float3 _S260 = va_0(i_13, 0U, &kernelContext_33);
            float3 _S261 = va_0(i_13, 4U, &kernelContext_33);
            float3 _S262 = _S260 - _S261 * _S256;
            float3 _S263 = vl_0(i_13, 0U, &kernelContext_33);
            float3 _S264 = vl_0(i_13, 4U, &kernelContext_33);
            set_v_0(i_13, 0U, _S262, _S263 - _S264 * _S256, &kernelContext_33);
            p_7 = p_7 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        previous_0 = gamma_0;
        num_0 = num_1;
        iteration_0 = _S251;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    q_3 = b0_1 + tid_7;
    for(;;)
    {
        if(q_3 < _S154)
        {
        }
        else
        {
            break;
        }
        uint j_15 = (&kernelContext_33)->lists_0[q_3];
        thread BlastBond_natural_0 _S265 = *((&kernelContext_33)->bonds_0+j_15);
        if(!(((&_S265)->health_0) > 0.0f))
        {
            uint _S266 = j_15 * 2U;
            packed_float4 _S267 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_33)->x_0+_S266) = _S267;
            *((&kernelContext_33)->x_0+(_S266 + 1U)) = _S267;
            q_3 = q_3 + 1024U;
            continue;
        }
        thread float3 ta_5;
        thread float3 tl_5;
        bond_term_1(&_S265, j_15, 6U, &ta_5, &tl_5, &kernelContext_33);
        uint _S268 = j_15 * 2U;
        *((&kernelContext_33)->x_0+_S268) = packed_float4(float4((float4(*((&kernelContext_33)->x_0+_S268)) ).xyz + ta_5, 0.0f)) ;
        uint _S269 = _S268 + 1U;
        *((&kernelContext_33)->x_0+_S269) = packed_float4(float4((float4(*((&kernelContext_33)->x_0+_S269)) ).xyz + tl_5, 0.0f)) ;
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
        converged_0 = warm_1;
    }
    else
    {
        converged_0 = false;
    }
    if(converged_0)
    {
        converged_0 = iteration_0 == 0U;
    }
    else
    {
        converged_0 = false;
    }
    if(converged_0)
    {
        q_4 = 4U;
    }
    else
    {
        q_4 = 0U;
    }
    uint flags_0 = p_4 | q_4;
    if(tid_7 == 0U)
    {
        *((&kernelContext_33)->report_0+k_17) = packed_uint4(uint4(iteration_0, flags_0, (as_type<uint>((num_0))), (as_type<uint>((delta2_0))))) ;
    }
    return;
}

