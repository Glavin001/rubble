#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint group_uniform_0(uint value_0, uint thread_0)
{
    return value_0;
}

uint ns_0(uint n_0, uint v_0)
{
    return n_0 * 24U + v_0;
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
    BlastParams_0 constant* params_0;
    packed_uint4 device* islands_0;
    packed_uint4 device* report_0;
    uint device* lists_0;
    packed_float4 device* accel_0;
    packed_float4 device* scratch_0;
    BlastNode_natural_0 device* nodes_0;
    uint device* refs_0;
    BlastBond_natural_0 device* bonds_0;
    packed_float4 device* x_0;
    array<float2, int(32)> threadgroup* g_blast_part_0;
    float3 threadgroup* g_centre_0;
    array<array<float, int(6)>, int(6)> threadgroup* g_gram_0;
    array<float, int(6)> threadgroup* g_rhs_0;
    array<float, int(6)> threadgroup* g_coef_0;
    array<float, int(1024)> threadgroup* g_max_0;
    float threadgroup* g_max_out_0;
};

float2 blast_group_sum_0(uint tid_0, float2 v_1, KernelContext_0 thread* kernelContext_0)
{
    float2 w_0 = simd_sum(v_1);
    if((tid_0 % 32U) == 0U)
    {
        (*kernelContext_0->g_blast_part_0)[tid_0 / 32U] = w_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 _S1 = (*kernelContext_0->g_blast_part_0)[int(0)];
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
        float2 s_2 = s_1 + (*kernelContext_0->g_blast_part_0)[k_0];
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

void set_v_0(uint n_1, uint v_2, float3 a_1, float3 l_0, KernelContext_0 thread* kernelContext_1)
{
    uint _S2 = ns_0(n_1, v_2);
    *(kernelContext_1->scratch_0+_S2) = packed_float4(float4(a_1, 0.0f)) ;
    *(kernelContext_1->scratch_0+(_S2 + 1U)) = packed_float4(float4(l_0, 0.0f)) ;
    return;
}

bool build_block_0(uint node_0, KernelContext_0 thread* kernelContext_2)
{
    BlastNode_natural_0 n_2 = kernelContext_2->nodes_0[node_0];
    float3 _S3 = float3(0.0f) ;
    uint r_0 = n_2.ref_begin_0;
    float3 diag_0 = _S3;
    float3 offd_0 = _S3;
    float3 ks_0 = _S3;
    float cs_0 = 0.0f;
    for(;;)
    {
        if(r_0 < (n_2.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_0 = kernelContext_2->refs_0[r_0];
        BlastBond_natural_0 device* _S4 = kernelContext_2->bonds_0+(ref_0 & 2147483647U);
        BlastBond_natural_0 b_1 = *_S4;
        if(!(((*_S4).health_0) > 0.0f))
        {
            r_0 = r_0 + 1U;
            continue;
        }
        float3 o_0;
        if((ref_0 & 2147483648U) != 0U)
        {
            o_0 = (float4(b_1.o1_0) ).xyz;
        }
        else
        {
            o_0 = (float4(b_1.o0_0) ).xyz;
        }
        float _S5 = b_1.s_0;
        float s2_0 = _S5 * _S5;
        float3 _S6 = float3(s2_0) ;
        float _S7 = o_0.x;
        float _S8 = o_0.y;
        float _S9 = o_0.z;
        float3 offd_1 = offd_0 + _S6 * - float3(_S7 * _S8, _S7 * _S9, _S8 * _S9);
        float3 ks_1 = ks_0 + _S6 * o_0;
        float cs_1 = cs_0 + s2_0;
        diag_0 = diag_0 + _S6 * (float3((1.0f + dot(o_0, o_0)))  - o_0 * o_0);
        offd_0 = offd_1;
        ks_0 = ks_1;
        cs_0 = cs_1;
        r_0 = r_0 + 1U;
    }
    float _S10 = n_2.w_ang_0;
    float da_0 = _S10 * _S10;
    float c_0 = n_2.w_lin_0 * n_2.w_lin_0 * cs_0;
    float3 k_1 = ks_0 * float3((n_2.w_ang_0 * n_2.w_lin_0)) ;
    float4 a0_0 = float4(diag_0 * float3(da_0) , offd_0.x * da_0);
    float _S11 = offd_0.y * da_0;
    float _S12 = offd_0.z * da_0;
    float4 a1_0 = float4(_S11, _S12, c_0, 0.0f);
    uint _S13 = ns_0(node_0, 18U);
    *(kernelContext_2->scratch_0+_S13) = packed_float4(a0_0) ;
    *(kernelContext_2->scratch_0+(_S13 + 1U)) = packed_float4(a1_0) ;
    *(kernelContext_2->scratch_0+(_S13 + 2U)) = packed_float4(float4(k_1, 0.0f)) ;
    if(!(c_0 > 0.0f))
    {
        return false;
    }
    float ic_0 = 1.0f / c_0;
    float kk_0 = dot(k_1, k_1);
    float _S14 = k_1.x;
    float sxx_0 = a0_0.x + (_S14 * _S14 - kk_0) * ic_0;
    float _S15 = k_1.y;
    float syy_0 = a0_0.y + (_S15 * _S15 - kk_0) * ic_0;
    float _S16 = k_1.z;
    float szz_0 = a0_0.z + (_S16 * _S16 - kk_0) * ic_0;
    float sxy_0 = a0_0.w + _S14 * _S15 * ic_0;
    float sxz_0 = _S11 + _S14 * _S16 * ic_0;
    float syz_0 = _S12 + _S15 * _S16 * ic_0;
    float cxx_0 = syy_0 * szz_0 - syz_0 * syz_0;
    float cxy_0 = sxz_0 * syz_0 - sxy_0 * szz_0;
    float cxz_0 = sxy_0 * syz_0 - sxz_0 * syy_0;
    float det_0 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    bool _S17;
    if(!(det_0 > 0.0f))
    {
        _S17 = true;
    }
    else
    {
        _S17 = !isfinite(det_0);
    }
    if(_S17)
    {
        return false;
    }
    float id_0 = 1.0f / det_0;
    float cyz_0 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    uint _S18 = ns_0(node_0, 21U);
    *(kernelContext_2->scratch_0+_S18) = packed_float4(float4(cxx_0 * id_0, (sxx_0 * szz_0 - sxz_0 * sxz_0) * id_0, (sxx_0 * syy_0 - sxy_0 * sxy_0) * id_0, cxy_0 * id_0)) ;
    *(kernelContext_2->scratch_0+(_S18 + 1U)) = packed_float4(float4(cxz_0 * id_0, cyz_0 * id_0, ic_0, 0.0f)) ;
    *(kernelContext_2->scratch_0+(_S18 + 2U)) = packed_float4(float4(k_1 * float3(ic_0) , 0.0f)) ;
    return true;
}

float3 rel_0(uint n_3, KernelContext_0 thread* kernelContext_3)
{
    return (float4(kernelContext_3->nodes_0[n_3].position_0) ).xyz - *kernelContext_3->g_centre_0;
}

void build_modes_0(uint tid_1, uint n0_1, uint n1_1, KernelContext_0 thread* kernelContext_4)
{
    uint kk_1;
    uint l_1;
    float4 _S19 = float4(0.0f) ;
    uint _S20 = n0_1 + tid_1;
    uint p_0 = _S20;
    float4 cw_0 = _S19;
    for(;;)
    {
        if(p_0 < n1_1)
        {
        }
        else
        {
            break;
        }
        uint i_0 = kernelContext_4->lists_0[p_0];
        float w_1 = 1.0f / (kernelContext_4->nodes_0[i_0].w_lin_0 * kernelContext_4->nodes_0[i_0].w_lin_0);
        float4 cw_1 = cw_0 + float4((float4(kernelContext_4->nodes_0[i_0].position_0) ).xyz * float3(w_1) , w_1);
        p_0 = p_0 + 1024U;
        cw_0 = cw_1;
    }
    float2 _S21 = blast_group_sum_0(tid_1, cw_0.xy, kernelContext_4);
    float2 _S22 = blast_group_sum_0(tid_1, cw_0.zw, kernelContext_4);
    bool _S23 = tid_1 == 0U;
    if(_S23)
    {
        *kernelContext_4->g_centre_0 = float3(_S21.x, _S21.y, _S22.x) / float3(_S22.y) ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
    p_0 = _S20;
    float tt_0 = 0.0f;
    for(;;)
    {
        if(p_0 < n1_1)
        {
        }
        else
        {
            break;
        }
        uint i_1 = kernelContext_4->lists_0[p_0];
        BlastNode_natural_0 n_4 = kernelContext_4->nodes_0[i_1];
        float _S24 = n_4.w_ang_0;
        float wa_0 = 1.0f / (_S24 * _S24);
        float _S25 = n_4.w_lin_0;
        float wl_0 = 1.0f / (_S25 * _S25);
        float3 _S26 = rel_0(i_1, kernelContext_4);
        float3 _S27 = cross(_S26, float3(1.0f, 0.0f, 0.0f));
        float3 _S28 = cross(_S26, float3(0.0f, 1.0f, 0.0f));
        float3 _S29 = cross(_S26, float3(0.0f, 0.0f, 1.0f));
        array<float3, int(3)> c_1 = { { _S27, _S28, _S29 } };
        rr_0[int(0)] = rr_0[int(0)] + (wa_0 + wl_0 * dot(_S27, _S27));
        rr_0[int(1)] = rr_0[int(1)] + (wa_0 + wl_0 * dot(_S28, _S28));
        rr_0[int(2)] = rr_0[int(2)] + (wa_0 + wl_0 * dot(_S29, _S29));
        rr_0[int(3)] = rr_0[int(3)] + wl_0 * dot(_S27, _S28);
        rr_0[int(4)] = rr_0[int(4)] + wl_0 * dot(_S27, _S29);
        rr_0[int(5)] = rr_0[int(5)] + wl_0 * dot(_S28, _S29);
        kk_1 = 0U;
        for(;;)
        {
            if(kk_1 < 3U)
            {
            }
            else
            {
                break;
            }
            rt_0[kk_1] = rt_0[kk_1] + float3(wl_0)  * c_1[kk_1];
            kk_1 = kk_1 + 1U;
        }
        float tt_1 = tt_0 + wl_0;
        p_0 = p_0 + 1024U;
        tt_0 = tt_1;
    }
    float2 _S30 = blast_group_sum_0(tid_1, float2(rr_0[int(0)], rr_0[int(1)]), kernelContext_4);
    float2 _S31 = blast_group_sum_0(tid_1, float2(rr_0[int(2)], rr_0[int(3)]), kernelContext_4);
    float2 _S32 = blast_group_sum_0(tid_1, float2(rr_0[int(4)], rr_0[int(5)]), kernelContext_4);
    float2 _S33 = blast_group_sum_0(tid_1, rt_0[int(0)].xy, kernelContext_4);
    float2 _S34 = blast_group_sum_0(tid_1, float2(rt_0[int(0)].z, rt_0[int(1)].x), kernelContext_4);
    float2 _S35 = blast_group_sum_0(tid_1, rt_0[int(1)].yz, kernelContext_4);
    float2 _S36 = blast_group_sum_0(tid_1, rt_0[int(2)].xy, kernelContext_4);
    float2 _S37 = blast_group_sum_0(tid_1, float2(rt_0[int(2)].z, tt_0), kernelContext_4);
    if(_S23)
    {
        thread array<array<float, int(6)>, int(6)> m_0;
        m_0[int(0)][int(0)] = _S30.x;
        m_0[int(1)][int(1)] = _S30.y;
        m_0[int(2)][int(2)] = _S31.x;
        m_0[int(0)][int(1)] = _S31.y;
        float _S38 = _S32.x;
        m_0[int(0)][int(2)] = _S38;
        m_0[int(1)][int(2)] = _S32.y;
        m_0[int(1)][int(0)] = m_0[int(0)][int(1)];
        m_0[int(2)][int(0)] = _S38;
        m_0[int(2)][int(1)] = m_0[int(1)][int(2)];
        float3 _S39 = float3(_S33.x, _S33.y, _S34.x);
        float3 _S40 = float3(_S34.y, _S35.x, _S35.y);
        float3 _S41 = float3(_S36.x, _S36.y, _S37.x);
        kk_1 = 0U;
        for(;;)
        {
            if(kk_1 < 3U)
            {
            }
            else
            {
                break;
            }
            uint _S42 = 3U + kk_1;
            m_0[int(0)][_S42] = _S39[kk_1];
            m_0[int(1)][_S42] = _S40[kk_1];
            m_0[int(2)][_S42] = _S41[kk_1];
            m_0[_S42][int(0)] = _S39[kk_1];
            m_0[_S42][int(1)] = _S40[kk_1];
            m_0[_S42][int(2)] = _S41[kk_1];
            l_1 = 0U;
            for(;;)
            {
                if(l_1 < 3U)
                {
                }
                else
                {
                    break;
                }
                if(kk_1 == l_1)
                {
                    tt_0 = _S37.y;
                }
                else
                {
                    tt_0 = 0.0f;
                }
                m_0[_S42][3U + l_1] = tt_0;
                l_1 = l_1 + 1U;
            }
            kk_1 = kk_1 + 1U;
        }
        l_1 = 0U;
        for(;;)
        {
            if(l_1 < 6U)
            {
            }
            else
            {
                break;
            }
            uint c2_0 = 0U;
            for(;;)
            {
                if(c2_0 < 6U)
                {
                }
                else
                {
                    break;
                }
                (*kernelContext_4->g_gram_0)[l_1][c2_0] = m_0[l_1][c2_0];
                c2_0 = c2_0 + 1U;
            }
            l_1 = l_1 + 1U;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
}

uint bs_0(uint j_0, KernelContext_0 thread* kernelContext_5)
{
    return kernelContext_5->params_0->node_count_0 * 24U + 2U * j_0;
}

float3 va_0(uint n_5, uint v_3, KernelContext_0 thread* kernelContext_6)
{
    return (float4(*(kernelContext_6->scratch_0+ns_0(n_5, v_3))) ).xyz;
}

float3 vl_0(uint n_6, uint v_4, KernelContext_0 thread* kernelContext_7)
{
    return (float4(*(kernelContext_7->scratch_0+(ns_0(n_6, v_4) + 1U))) ).xyz;
}

void bond_term_0(const BlastBond_natural_0 thread* b_2, uint v_5, float3 thread* ta_0, float3 thread* tl_0, KernelContext_0 thread* kernelContext_8)
{
    uint _S43 = b_2->n0_0;
    BlastNode_natural_0 m0_0 = kernelContext_8->nodes_0[b_2->n0_0];
    uint _S44 = b_2->n1_0;
    BlastNode_natural_0 m1_0 = kernelContext_8->nodes_0[b_2->n1_0];
    float3 _S45 = va_0(b_2->n0_0, v_5, kernelContext_8);
    float3 a0_1 = _S45 * float3(m0_0.w_ang_0) ;
    float3 _S46 = vl_0(_S43, v_5, kernelContext_8);
    float3 l0_0 = _S46 * float3(m0_0.w_lin_0) ;
    float3 _S47 = va_0(_S44, v_5, kernelContext_8);
    float3 a1_1 = _S47 * float3(m1_0.w_ang_0) ;
    float3 _S48 = vl_0(_S44, v_5, kernelContext_8);
    float3 l1_0 = _S48 * float3(m1_0.w_lin_0) ;
    float3 _S49 = float3(b_2->s_0) ;
    *ta_0 = (a0_1 - a1_1) * _S49;
    *tl_0 = (l0_0 - l1_0 + (cross((float4(b_2->o0_0) ).xyz, a0_1) - cross((float4(b_2->o1_0) ).xyz, a1_1))) * _S49;
    return;
}

void bond_term_1(const BlastBond_natural_0 thread* b_3, uint v_6, float3 thread* ta_1, float3 thread* tl_1, KernelContext_0 thread* kernelContext_9)
{
    uint _S50 = b_3->n0_0;
    BlastNode_natural_0 m0_1 = kernelContext_9->nodes_0[b_3->n0_0];
    uint _S51 = b_3->n1_0;
    BlastNode_natural_0 m1_1 = kernelContext_9->nodes_0[b_3->n1_0];
    float3 _S52 = va_0(b_3->n0_0, v_6, kernelContext_9);
    float3 a0_2 = _S52 * float3(m0_1.w_ang_0) ;
    float3 _S53 = vl_0(_S50, v_6, kernelContext_9);
    float3 l0_1 = _S53 * float3(m0_1.w_lin_0) ;
    float3 _S54 = va_0(_S51, v_6, kernelContext_9);
    float3 a1_2 = _S54 * float3(m1_1.w_ang_0) ;
    float3 _S55 = vl_0(_S51, v_6, kernelContext_9);
    float3 l1_1 = _S55 * float3(m1_1.w_lin_0) ;
    float3 _S56 = float3(b_3->s_0) ;
    *ta_1 = (a0_2 - a1_2) * _S56;
    *tl_1 = (l0_1 - l1_1 + (cross((float4(b_3->o0_0) ).xyz, a0_2) - cross((float4(b_3->o1_0) ).xyz, a1_2))) * _S56;
    return;
}

void gather_0(uint node_1, bool from_t_0, bool with_u_0, float3 thread* ang_0, float3 thread* lin_0, KernelContext_0 thread* kernelContext_10)
{
    BlastNode_natural_0 n_7 = kernelContext_10->nodes_0[node_1];
    float3 _S57 = float3(0.0f) ;
    uint r_1 = n_7.ref_begin_0;
    float3 acc_a_0 = _S57;
    float3 acc_l_0 = _S57;
    for(;;)
    {
        if(r_1 < (n_7.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_1 = kernelContext_10->refs_0[r_1];
        uint j_1 = ref_1 & 2147483647U;
        thread BlastBond_natural_0 _S58 = *(kernelContext_10->bonds_0+j_1);
        if(!(((&_S58)->health_0) > 0.0f))
        {
            r_1 = r_1 + 1U;
            continue;
        }
        float3 ya_0;
        float3 yl_0;
        if(from_t_0)
        {
            uint _S59 = bs_0(j_1, kernelContext_10);
            float3 _S60 = (float4(*(kernelContext_10->scratch_0+(_S59 + 1U))) ).xyz;
            ya_0 = (float4(*(kernelContext_10->scratch_0+_S59)) ).xyz;
            yl_0 = _S60;
        }
        else
        {
            uint _S61 = j_1 * 2U;
            float3 ya_1 = (float4(*(kernelContext_10->x_0+_S61)) ).xyz;
            float3 yl_1 = (float4(*(kernelContext_10->x_0+(_S61 + 1U))) ).xyz;
            if(with_u_0)
            {
                thread float3 ta_2;
                thread float3 tl_2;
                bond_term_0(&_S58, 6U, &ta_2, &tl_2, kernelContext_10);
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
        float3 _S62 = float3((&_S58)->s_0) ;
        float3 ya_2 = ya_0 * _S62;
        float3 yl_3 = yl_0 * _S62;
        float3 acc_a_1;
        float3 acc_l_1;
        if((ref_1 & 2147483648U) != 0U)
        {
            float3 acc_l_2 = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross((float4((&_S58)->o1_0) ).xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            float3 acc_l_3 = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross((float4((&_S58)->o0_0) ).xyz, yl_3));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_1 = r_1 + 1U;
    }
    *ang_0 = acc_a_0 * float3(n_7.w_ang_0) ;
    *lin_0 = acc_l_0 * float3(n_7.w_lin_0) ;
    return;
}

float rebuild_0(uint tid_2, uint n0_2, uint n1_2, bool with_u_1, KernelContext_0 thread* kernelContext_11)
{
    uint p_1 = n0_2 + tid_2;
    float s_3 = 0.0f;
    for(;;)
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        uint i_2 = kernelContext_11->lists_0[p_1];
        thread float3 a_2;
        thread float3 l_2;
        gather_0(i_2, false, with_u_1, &a_2, &l_2, kernelContext_11);
        float3 _S63 = va_0(i_2, 14U, kernelContext_11);
        float3 ra_0 = _S63 - a_2;
        float3 _S64 = vl_0(i_2, 14U, kernelContext_11);
        float3 rl_0 = _S64 - l_2;
        set_v_0(i_2, 0U, ra_0, rl_0, kernelContext_11);
        float s_4 = s_3 + norm2_0(ra_0, rl_0);
        p_1 = p_1 + 1024U;
        s_3 = s_4;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    float2 _S65 = blast_group_sum_0(tid_2, float2(s_3, 0.0f), kernelContext_11);
    return _S65.x;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

void solve_gram_0(KernelContext_0 thread* kernelContext_12)
{
    uint c_2;
    uint p_2;
    thread array<array<float, int(6)>, int(6)> a_3;
    thread array<float, int(6)> b_4;
    uint r_2 = 0U;
    for(;;)
    {
        if(r_2 < 6U)
        {
        }
        else
        {
            break;
        }
        b_4[r_2] = (*kernelContext_12->g_rhs_0)[r_2];
        c_2 = 0U;
        for(;;)
        {
            if(c_2 < 6U)
            {
            }
            else
            {
                break;
            }
            a_3[r_2][c_2] = (*kernelContext_12->g_gram_0)[r_2][c_2];
            c_2 = c_2 + 1U;
        }
        r_2 = r_2 + 1U;
    }
    c_2 = 0U;
    for(;;)
    {
        if(c_2 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S66 = c_2 + 1U;
        p_2 = c_2;
        r_2 = _S66;
        for(;;)
        {
            if(r_2 < 6U)
            {
            }
            else
            {
                break;
            }
            if((abs(a_3[r_2][c_2])) > (abs(a_3[p_2][c_2])))
            {
                p_2 = r_2;
            }
            r_2 = r_2 + 1U;
        }
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
            float t_0 = a_3[c_2][k_2];
            a_3[c_2][k_2] = a_3[p_2][k_2];
            a_3[p_2][k_2] = t_0;
            k_2 = k_2 + 1U;
        }
        float tb_0 = b_4[c_2];
        b_4[c_2] = b_4[p_2];
        b_4[p_2] = tb_0;
        float piv_0 = a_3[c_2][c_2];
        uint z_1;
        if(!((abs(a_3[c_2][c_2])) > 0.0f))
        {
            z_1 = 0U;
            for(;;)
            {
                if(z_1 < 6U)
                {
                }
                else
                {
                    break;
                }
                (*kernelContext_12->g_coef_0)[z_1] = 0.0f;
                z_1 = z_1 + 1U;
            }
            return;
        }
        z_1 = _S66;
        for(;;)
        {
            if(z_1 < 6U)
            {
            }
            else
            {
                break;
            }
            float f_0 = a_3[z_1][c_2] / piv_0;
            uint k_3 = c_2;
            for(;;)
            {
                if(k_3 < 6U)
                {
                }
                else
                {
                    break;
                }
                a_3[z_1][k_3] = a_3[z_1][k_3] - f_0 * a_3[c_2][k_3];
                k_3 = k_3 + 1U;
            }
            b_4[z_1] = b_4[z_1] - f_0 * b_4[c_2];
            z_1 = z_1 + 1U;
        }
        c_2 = _S66;
    }
    int r_3 = int(5);
    for(;;)
    {
        if(r_3 >= int(0))
        {
        }
        else
        {
            break;
        }
        p_2 = uint(r_3) + 1U;
        float s_5 = b_4[r_3];
        for(;;)
        {
            if(p_2 < 6U)
            {
            }
            else
            {
                break;
            }
            float s_6 = s_5 - a_3[r_3][p_2] * (*kernelContext_12->g_coef_0)[p_2];
            p_2 = p_2 + 1U;
            s_5 = s_6;
        }
        (*kernelContext_12->g_coef_0)[r_3] = s_5 / a_3[r_3][r_3];
        r_3 = r_3 - int(1);
    }
    return;
}

void project_0(uint tid_3, uint n0_3, uint n1_3, uint v_7, bool free_0, KernelContext_0 thread* kernelContext_13)
{
    if(!free_0)
    {
        return;
    }
    float3 _S67 = float3(0.0f) ;
    uint _S68 = n0_3 + tid_3;
    uint p_3 = _S68;
    float3 sr_0 = _S67;
    float3 st_0 = _S67;
    for(;;)
    {
        if(p_3 < n1_3)
        {
        }
        else
        {
            break;
        }
        uint i_3 = kernelContext_13->lists_0[p_3];
        BlastNode_natural_0 n_8 = kernelContext_13->nodes_0[i_3];
        float3 _S69 = vl_0(i_3, v_7, kernelContext_13);
        float3 lin_1 = _S69 / float3(n_8.w_lin_0) ;
        float3 _S70 = va_0(i_3, v_7, kernelContext_13);
        float3 _S71 = _S70 / float3(n_8.w_ang_0) ;
        float3 _S72 = rel_0(i_3, kernelContext_13);
        float3 sr_1 = sr_0 + (_S71 - cross(_S72, lin_1));
        float3 st_1 = st_0 + lin_1;
        p_3 = p_3 + 1024U;
        sr_0 = sr_1;
        st_0 = st_1;
    }
    float2 _S73 = blast_group_sum_0(tid_3, sr_0.xy, kernelContext_13);
    float2 _S74 = blast_group_sum_0(tid_3, float2(sr_0.z, st_0.x), kernelContext_13);
    float2 _S75 = blast_group_sum_0(tid_3, st_0.yz, kernelContext_13);
    if(tid_3 == 0U)
    {
        (*kernelContext_13->g_rhs_0)[int(0)] = _S73.x;
        (*kernelContext_13->g_rhs_0)[int(1)] = _S73.y;
        (*kernelContext_13->g_rhs_0)[int(2)] = _S74.x;
        (*kernelContext_13->g_rhs_0)[int(3)] = _S74.y;
        (*kernelContext_13->g_rhs_0)[int(4)] = _S75.x;
        (*kernelContext_13->g_rhs_0)[int(5)] = _S75.y;
        solve_gram_0(kernelContext_13);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float3 _S76 = float3((*kernelContext_13->g_coef_0)[int(0)], (*kernelContext_13->g_coef_0)[int(1)], (*kernelContext_13->g_coef_0)[int(2)]);
    float3 _S77 = float3((*kernelContext_13->g_coef_0)[int(3)], (*kernelContext_13->g_coef_0)[int(4)], (*kernelContext_13->g_coef_0)[int(5)]);
    p_3 = _S68;
    for(;;)
    {
        if(p_3 < n1_3)
        {
        }
        else
        {
            break;
        }
        uint i_4 = kernelContext_13->lists_0[p_3];
        BlastNode_natural_0 n_9 = kernelContext_13->nodes_0[i_4];
        float3 _S78 = va_0(i_4, v_7, kernelContext_13);
        float3 _S79 = _S78 - _S76 / float3(n_9.w_ang_0) ;
        float3 _S80 = vl_0(i_4, v_7, kernelContext_13);
        float3 _S81 = rel_0(i_4, kernelContext_13);
        set_v_0(i_4, v_7, _S79, _S80 - (_S77 + cross(_S81, _S76)) / float3(n_9.w_lin_0) , kernelContext_13);
        p_3 = p_3 + 1024U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return;
}

float bond_pass_0(uint tid_4, uint b0_0, uint b1_0, uint v_8, KernelContext_0 thread* kernelContext_14)
{
    uint q_0 = b0_0 + tid_4;
    float z_2 = 0.0f;
    for(;;)
    {
        if(q_0 < b1_0)
        {
        }
        else
        {
            break;
        }
        uint j_2 = kernelContext_14->lists_0[q_0];
        thread BlastBond_natural_0 _S82 = *(kernelContext_14->bonds_0+j_2);
        float3 _S83 = float3(0.0f) ;
        thread float3 ta_3 = _S83;
        thread float3 tl_3 = _S83;
        if(((&_S82)->health_0) > 0.0f)
        {
            bond_term_0(&_S82, v_8, &ta_3, &tl_3, kernelContext_14);
        }
        float z_3 = z_2 + norm2_0(ta_3, tl_3);
        uint _S84 = bs_0(j_2, kernelContext_14);
        *(kernelContext_14->scratch_0+_S84) = packed_float4(float4(ta_3, 0.0f)) ;
        *(kernelContext_14->scratch_0+(_S84 + 1U)) = packed_float4(float4(tl_3, 0.0f)) ;
        q_0 = q_0 + 1024U;
        z_2 = z_3;
    }
    float2 _S85 = blast_group_sum_0(tid_4, float2(z_2, 0.0f), kernelContext_14);
    return _S85.x;
}

float residual_norm_0(uint tid_5, uint n0_4, uint n1_4, KernelContext_0 thread* kernelContext_15)
{
    uint p_4 = n0_4 + tid_5;
    float s_7 = 0.0f;
    for(;;)
    {
        if(p_4 < n1_4)
        {
        }
        else
        {
            break;
        }
        uint i_5 = kernelContext_15->lists_0[p_4];
        float3 _S86 = va_0(i_5, 0U, kernelContext_15);
        float3 _S87 = vl_0(i_5, 0U, kernelContext_15);
        float s_8 = s_7 + norm2_0(_S86, _S87);
        p_4 = p_4 + 1024U;
        s_7 = s_8;
    }
    float2 _S88 = blast_group_sum_0(tid_5, float2(s_7, 0.0f), kernelContext_15);
    return _S88.x;
}

float3 sym_mul_0(float4 s0_0, float4 s1_0, float3 v_9)
{
    float _S89 = v_9.x;
    float _S90 = s0_0.w;
    float _S91 = v_9.y;
    float _S92 = s1_0.x;
    float _S93 = v_9.z;
    float _S94 = s1_0.y;
    return float3(s0_0.x * _S89 + _S90 * _S91 + _S92 * _S93, _S90 * _S89 + s0_0.y * _S91 + _S94 * _S93, _S92 * _S89 + _S94 * _S91 + s0_0.z * _S93);
}

void inverse_apply_0(uint n_10, float3 a_4, float3 l_3, float3 thread* oa_0, float3 thread* ol_0, KernelContext_0 thread* kernelContext_16)
{
    uint _S95 = ns_0(n_10, 21U);
    float4 _S96 = float4(*(kernelContext_16->scratch_0+(_S95 + 1U))) ;
    float3 kc_0 = (float4(*(kernelContext_16->scratch_0+(_S95 + 2U))) ).xyz;
    float3 _S97 = sym_mul_0(float4(*(kernelContext_16->scratch_0+_S95)) , _S96, a_4 + cross(kc_0, l_3));
    *oa_0 = _S97;
    *ol_0 = l_3 * float3(_S96.z)  - cross(kc_0, _S97);
    return;
}

float node_pass_0(uint tid_6, uint n0_5, uint n1_5, uint out_v_0, KernelContext_0 thread* kernelContext_17)
{
    uint p_5 = n0_5 + tid_6;
    float s_9 = 0.0f;
    for(;;)
    {
        if(p_5 < n1_5)
        {
        }
        else
        {
            break;
        }
        uint i_6 = kernelContext_17->lists_0[p_5];
        thread float3 a_5;
        thread float3 l_4;
        gather_0(i_6, true, false, &a_5, &l_4, kernelContext_17);
        set_v_0(i_6, out_v_0, a_5, l_4, kernelContext_17);
        float s_10 = s_9 + norm2_0(a_5, l_4);
        p_5 = p_5 + 1024U;
        s_9 = s_10;
    }
    return s_9;
}

void block_apply_0(uint n_11, float3 a_6, float3 l_5, float3 thread* oa_1, float3 thread* ol_1, KernelContext_0 thread* kernelContext_18)
{
    uint _S98 = ns_0(n_11, 18U);
    float4 _S99 = float4(*(kernelContext_18->scratch_0+(_S98 + 1U))) ;
    float3 k_4 = (float4(*(kernelContext_18->scratch_0+(_S98 + 2U))) ).xyz;
    *oa_1 = sym_mul_0(float4(*(kernelContext_18->scratch_0+_S98)) , _S99, a_6) - cross(k_4, l_5);
    *ol_1 = cross(k_4, a_6) + float3(_S99.z)  * l_5;
    return;
}

float group_max_0(uint tid_7, float v_10, KernelContext_0 thread* kernelContext_19)
{
    (*kernelContext_19->g_max_0)[tid_7] = v_10;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint k_5 = 512U;
    for(;;)
    {
        if(k_5 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_7 < k_5)
        {
            (*kernelContext_19->g_max_0)[tid_7] = max((*kernelContext_19->g_max_0)[tid_7], (*kernelContext_19->g_max_0)[tid_7 + k_5]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_5 = k_5 >> 1U;
    }
    if(tid_7 == 0U)
    {
        *kernelContext_19->g_max_out_0 = (*kernelContext_19->g_max_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S100 = kernelContext_19->g_max_out_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *_S100;
}

[[kernel]] void blast_native(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_1 [[buffer(0)]], packed_uint4 device* islands_1 [[buffer(1)]], packed_uint4 device* report_1 [[buffer(8)]], uint device* lists_1 [[buffer(9)]], packed_float4 device* accel_1 [[buffer(5)]], packed_float4 device* scratch_1 [[buffer(7)]], BlastNode_natural_0 device* nodes_1 [[buffer(2)]], uint device* refs_1 [[buffer(3)]], BlastBond_natural_0 device* bonds_1 [[buffer(4)]], packed_float4 device* x_1 [[buffer(6)]])
{
    float anchor_0;
    float force_start_0;
    bool failed_0;
    bool ok_0;
    thread KernelContext_0 kernelContext_20;
    (&kernelContext_20)->params_0 = params_1;
    (&kernelContext_20)->islands_0 = islands_1;
    (&kernelContext_20)->report_0 = report_1;
    (&kernelContext_20)->lists_0 = lists_1;
    (&kernelContext_20)->accel_0 = accel_1;
    (&kernelContext_20)->scratch_0 = scratch_1;
    (&kernelContext_20)->nodes_0 = nodes_1;
    (&kernelContext_20)->refs_0 = refs_1;
    (&kernelContext_20)->bonds_0 = bonds_1;
    (&kernelContext_20)->x_0 = x_1;
    threadgroup array<float2, int(32)> g_blast_part_1;
    (&kernelContext_20)->g_blast_part_0 = &g_blast_part_1;
    threadgroup float3 g_centre_1;
    (&kernelContext_20)->g_centre_0 = &g_centre_1;
    threadgroup array<array<float, int(6)>, int(6)> g_gram_1;
    (&kernelContext_20)->g_gram_0 = &g_gram_1;
    threadgroup array<float, int(6)> g_rhs_1;
    (&kernelContext_20)->g_rhs_0 = &g_rhs_1;
    threadgroup array<float, int(6)> g_coef_1;
    (&kernelContext_20)->g_coef_0 = &g_coef_1;
    threadgroup array<float, int(1024)> g_max_1;
    (&kernelContext_20)->g_max_0 = &g_max_1;
    threadgroup float g_max_out_1;
    (&kernelContext_20)->g_max_out_0 = &g_max_out_1;
    uint tid_8 = thread_1.x;
    uint k_6 = group_0.x;
    if(k_6 >= (params_1->island_count_0))
    {
        return;
    }
    uint4 _S101 = uint4(*((&kernelContext_20)->islands_0+k_6)) ;
    uint n0_6 = _S101.x;
    uint n1_6 = _S101.y;
    uint b0_1 = _S101.z;
    uint _S102 = _S101.w;
    uint p_6;
    if((n1_6 - n0_6) > ((&kernelContext_20)->params_0->wide_min_0))
    {
        p_6 = 1U;
    }
    else
    {
        p_6 = 0U;
    }
    if((group_uniform_0(p_6, tid_8)) != 0U)
    {
        return;
    }
    uint4 _S103 = uint4(*((&kernelContext_20)->report_0+k_6)) ;
    uint last_flags_0 = group_uniform_0(_S103.y, tid_8);
    bool warm_1 = ((&kernelContext_20)->params_0->warm_0) != 0U;
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
        uint _S104 = n0_6 + tid_8;
        singular_0 = 0.0f;
        p_6 = _S104;
        for(;;)
        {
            if(p_6 < n1_6)
            {
            }
            else
            {
                break;
            }
            uint i_7 = (&kernelContext_20)->lists_0[p_6];
            uint _S105 = i_7 * 2U;
            uint _S106 = ns_0(i_7, 16U);
            if(any((float4(*((&kernelContext_20)->accel_0+_S105)) ) != (float4(*((&kernelContext_20)->scratch_0+_S106)) )))
            {
                converged_0 = true;
            }
            else
            {
                converged_0 = any((float4(*((&kernelContext_20)->accel_0+(_S105 + 1U))) ) != (float4(*((&kernelContext_20)->scratch_0+(_S106 + 1U))) ));
            }
            if(converged_0)
            {
                singular_0 = 1.0f;
            }
            p_6 = p_6 + 1024U;
        }
        float2 _S107 = blast_group_sum_0(tid_8, float2(singular_0, 0.0f), &kernelContext_20);
        if((_S107.x) == 0.0f)
        {
            if(tid_8 == 0U)
            {
                *((&kernelContext_20)->report_0+k_6) = packed_uint4(uint4(0U, 7U, _S103.z, _S103.w)) ;
            }
            return;
        }
    }
    uint _S108 = n0_6 + tid_8;
    singular_0 = 0.0f;
    p_6 = _S108;
    float target_0 = 0.0f;
    for(;;)
    {
        if(p_6 < n1_6)
        {
        }
        else
        {
            break;
        }
        uint i_8 = (&kernelContext_20)->lists_0[p_6];
        BlastNode_natural_0 n_12 = (&kernelContext_20)->nodes_0[i_8];
        uint _S109 = i_8 * 2U;
        float4 _S110 = float4(*((&kernelContext_20)->accel_0+_S109)) ;
        float4 _S111 = float4(*((&kernelContext_20)->accel_0+(_S109 + 1U))) ;
        float3 _S112 = - _S110.xyz;
        if((n_12.w_ang_0) > 0.0f)
        {
            anchor_0 = 1.0f / n_12.w_ang_0;
        }
        else
        {
            anchor_0 = 1.0f;
        }
        float3 ba_0 = _S112 * float3(anchor_0) ;
        float3 _S113 = - _S111.xyz;
        if((n_12.w_lin_0) > 0.0f)
        {
            force_start_0 = (&kernelContext_20)->params_0->recip_length_0 / n_12.w_lin_0;
        }
        else
        {
            force_start_0 = (&kernelContext_20)->params_0->recip_length_0;
        }
        float3 bl_0 = _S113 * float3(force_start_0) ;
        float target_1 = target_0 + norm2_0(ba_0, bl_0);
        set_v_0(i_8, 14U, ba_0, bl_0, &kernelContext_20);
        set_v_0(i_8, 0U, ba_0, bl_0, &kernelContext_20);
        uint _S114 = ns_0(i_8, 16U);
        *((&kernelContext_20)->scratch_0+_S114) = packed_float4(_S110) ;
        *((&kernelContext_20)->scratch_0+(_S114 + 1U)) = packed_float4(_S111) ;
        float3 _S115 = float3(0.0f) ;
        set_v_0(i_8, 6U, _S115, _S115, &kernelContext_20);
        set_v_0(i_8, 2U, _S115, _S115, &kernelContext_20);
        bool _S116 = build_block_0(i_8, &kernelContext_20);
        if(!_S116)
        {
            singular_0 = 1.0f;
        }
        p_6 = p_6 + 1024U;
        target_0 = target_1;
    }
    uint _S117 = b0_1 + tid_8;
    anchor_0 = 0.0f;
    uint q_1 = _S117;
    for(;;)
    {
        if(q_1 < _S102)
        {
        }
        else
        {
            break;
        }
        BlastBond_natural_0 device* _S118 = (&kernelContext_20)->bonds_0+(&kernelContext_20)->lists_0[q_1];
        BlastBond_natural_0 b_5 = *_S118;
        if(((*_S118).health_0) > 0.0f)
        {
            if(!(((&kernelContext_20)->nodes_0[b_5.n0_0].w_lin_0) > 0.0f))
            {
                converged_0 = true;
            }
            else
            {
                converged_0 = !(((&kernelContext_20)->nodes_0[b_5.n1_0].w_lin_0) > 0.0f);
            }
        }
        else
        {
            converged_0 = false;
        }
        if(converged_0)
        {
            anchor_0 = 1.0f;
        }
        q_1 = q_1 + 1024U;
    }
    if(!warm_1)
    {
        q_1 = _S117;
        for(;;)
        {
            if(q_1 < _S102)
            {
            }
            else
            {
                break;
            }
            uint _S119 = (&kernelContext_20)->lists_0[q_1] * 2U;
            packed_float4 _S120 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_20)->x_0+_S119) = _S120;
            *((&kernelContext_20)->x_0+(_S119 + 1U)) = _S120;
            q_1 = q_1 + 1024U;
        }
    }
    float2 _S121 = blast_group_sum_0(tid_8, float2(target_0, singular_0), &kernelContext_20);
    float delta2_0 = _S121.x * (&kernelContext_20)->params_0->tolerance_0 * (&kernelContext_20)->params_0->tolerance_0;
    bool _S122 = (_S121.y) > 0.0f;
    float2 _S123 = blast_group_sum_0(tid_8, float2(anchor_0, 0.0f), &kernelContext_20);
    bool free_1 = (_S123.x) == 0.0f;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if(!(delta2_0 > 0.0f))
    {
        q_1 = _S117;
        for(;;)
        {
            if(q_1 < _S102)
            {
            }
            else
            {
                break;
            }
            uint _S124 = (&kernelContext_20)->lists_0[q_1] * 2U;
            packed_float4 _S125 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_20)->x_0+_S124) = _S125;
            *((&kernelContext_20)->x_0+(_S124 + 1U)) = _S125;
            q_1 = q_1 + 1024U;
        }
        if(tid_8 == 0U)
        {
            packed_uint4 device* _S126 = (&kernelContext_20)->report_0+k_6;
            if(warm_1)
            {
                p_6 = 4U;
            }
            else
            {
                p_6 = 0U;
            }
            *_S126 = packed_uint4(uint4(0U, 1U | p_6, (as_type<uint>((0.0f))), (as_type<uint>((delta2_0))))) ;
        }
        return;
    }
    if(free_1)
    {
        build_modes_0(tid_8, n0_6, n1_6, &kernelContext_20);
    }
    if(warm_1)
    {
        float _S127 = rebuild_0(tid_8, n0_6, n1_6, false, &kernelContext_20);
    }
    float _S128 = float_bits_0(2139095040U);
    if(((&kernelContext_20)->params_0->force_tolerance_0) > 0.0f)
    {
        q_1 = _S117;
        force_start_0 = 0.0f;
        for(;;)
        {
            if(q_1 < _S102)
            {
            }
            else
            {
                break;
            }
            uint _S129 = (&kernelContext_20)->lists_0[q_1] * 2U;
            float f_1 = force_start_0 + norm2_0((float4(*((&kernelContext_20)->x_0+_S129)) ).xyz, (float4(*((&kernelContext_20)->x_0+(_S129 + 1U))) ).xyz);
            q_1 = q_1 + 1024U;
            force_start_0 = f_1;
        }
        float2 _S130 = blast_group_sum_0(tid_8, float2(force_start_0, 0.0f), &kernelContext_20);
        force_start_0 = sqrt(_S130.x);
    }
    else
    {
        force_start_0 = 0.0f;
    }
    float best_0 = _S128;
    uint best_iteration_0 = 0U;
    float step_0 = _S128;
    float previous_0 = 0.0f;
    float num_0 = 0.0f;
    uint iteration_0 = 0U;
    float travel_0 = 0.0f;
    for(;;)
    {
        if(!_S122)
        {
            converged_0 = iteration_0 < ((&kernelContext_20)->params_0->max_iterations_0);
        }
        else
        {
            converged_0 = false;
        }
        if(converged_0)
        {
        }
        else
        {
            converged_0 = false;
            failed_0 = _S122;
            break;
        }
        project_0(tid_8, n0_6, n1_6, 0U, free_1, &kernelContext_20);
        float _S131 = bond_pass_0(tid_8, b0_1, _S102, 0U, &kernelContext_20);
        bool _S132 = _S131 <= delta2_0;
        if(_S132)
        {
            float _S133 = residual_norm_0(tid_8, n0_6, n1_6, &kernelContext_20);
            failed_0 = !(_S133 <= delta2_0);
        }
        else
        {
            failed_0 = false;
        }
        bool _S134 = iteration_0 > 0U;
        if(_S134)
        {
            ok_0 = true;
        }
        else
        {
            ok_0 = warm_1;
        }
        bool _S135;
        if(ok_0)
        {
            _S135 = _S132;
        }
        else
        {
            _S135 = false;
        }
        bool _S136;
        if(_S135)
        {
            _S136 = !failed_0;
        }
        else
        {
            _S136 = false;
        }
        bool hold_0;
        float num_1;
        float previous_1;
        if(_S136)
        {
            float _S137 = rebuild_0(tid_8, n0_6, n1_6, true, &kernelContext_20);
            project_0(tid_8, n0_6, n1_6, 0U, free_1, &kernelContext_20);
            float _S138 = bond_pass_0(tid_8, b0_1, _S102, 0U, &kernelContext_20);
            if(_S138 <= delta2_0)
            {
                float _S139 = residual_norm_0(tid_8, n0_6, n1_6, &kernelContext_20);
                hold_0 = !(_S139 <= delta2_0);
            }
            else
            {
                hold_0 = false;
            }
            num_1 = _S138;
            previous_1 = 0.0f;
        }
        else
        {
            num_1 = _S131;
            hold_0 = failed_0;
            previous_1 = previous_0;
        }
        if(num_1 < (best_0 * 0.99000000953674316f))
        {
            best_0 = num_1;
            best_iteration_0 = iteration_0;
        }
        else
        {
            if((iteration_0 - best_iteration_0) >= 512U)
            {
                converged_0 = false;
                failed_0 = _S122;
                num_0 = num_1;
                break;
            }
        }
        bool _S140;
        if(num_1 <= delta2_0)
        {
            _S140 = !hold_0;
        }
        else
        {
            _S140 = false;
        }
        if(_S140)
        {
            converged_0 = true;
            failed_0 = _S122;
            num_0 = num_1;
            break;
        }
        bool _S141;
        if(((&kernelContext_20)->params_0->force_tolerance_0) > 0.0f)
        {
            _S141 = _S134;
        }
        else
        {
            _S141 = false;
        }
        bool _S142;
        if(_S141)
        {
            _S142 = step_0 <= ((&kernelContext_20)->params_0->force_tolerance_0 * (force_start_0 + travel_0));
        }
        else
        {
            _S142 = false;
        }
        float mloc_0;
        if(_S142)
        {
            q_1 = _S117;
            mloc_0 = 0.0f;
            for(;;)
            {
                if(q_1 < _S102)
                {
                }
                else
                {
                    break;
                }
                uint j_3 = (&kernelContext_20)->lists_0[q_1];
                thread BlastBond_natural_0 _S143 = *((&kernelContext_20)->bonds_0+j_3);
                thread float3 ta_4;
                thread float3 tl_4;
                bond_term_1(&_S143, 6U, &ta_4, &tl_4, &kernelContext_20);
                uint _S144 = j_3 * 2U;
                float f_2 = mloc_0 + norm2_0((float4(*((&kernelContext_20)->x_0+_S144)) ).xyz + ta_4, (float4(*((&kernelContext_20)->x_0+(_S144 + 1U))) ).xyz + tl_4);
                q_1 = q_1 + 1024U;
                mloc_0 = f_2;
            }
            float _S145 = (&kernelContext_20)->params_0->force_tolerance_0;
            float2 _S146 = blast_group_sum_0(tid_8, float2(mloc_0, 0.0f), &kernelContext_20);
            if(step_0 <= (_S145 * sqrt(_S146.x)))
            {
                converged_0 = true;
                failed_0 = _S122;
                num_0 = num_1;
                break;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        uint p_7;
        if(iteration_0 == 0U)
        {
            p_6 = _S108;
            for(;;)
            {
                if(p_6 < n1_6)
                {
                }
                else
                {
                    break;
                }
                uint i_9 = (&kernelContext_20)->lists_0[p_6];
                float3 _S147 = va_0(i_9, 0U, &kernelContext_20);
                float3 _S148 = vl_0(i_9, 0U, &kernelContext_20);
                set_v_0(i_9, 8U, _S147, _S148, &kernelContext_20);
                p_6 = p_6 + 1024U;
            }
        }
        else
        {
            p_6 = _S108;
            for(;;)
            {
                if(p_6 < n1_6)
                {
                }
                else
                {
                    break;
                }
                uint i_10 = (&kernelContext_20)->lists_0[p_6];
                float3 _S149 = va_0(i_10, 0U, &kernelContext_20);
                float3 _S150 = vl_0(i_10, 0U, &kernelContext_20);
                thread float3 a_7;
                thread float3 l_6;
                inverse_apply_0(i_10, _S149, _S150, &a_7, &l_6, &kernelContext_20);
                set_v_0(i_10, 10U, a_7, l_6, &kernelContext_20);
                p_6 = p_6 + 1024U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            float _S151 = bond_pass_0(tid_8, b0_1, _S102, 10U, &kernelContext_20);
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            float _S152 = node_pass_0(tid_8, n0_6, n1_6, 12U, &kernelContext_20);
            p_7 = _S108;
            for(;;)
            {
                if(p_7 < n1_6)
                {
                }
                else
                {
                    break;
                }
                uint i_11 = (&kernelContext_20)->lists_0[p_7];
                float3 _S153 = va_0(i_11, 10U, &kernelContext_20);
                float3 _S154 = vl_0(i_11, 10U, &kernelContext_20);
                thread float3 sa_0;
                thread float3 sl_0;
                block_apply_0(i_11, _S153, _S154, &sa_0, &sl_0, &kernelContext_20);
                float3 _S155 = va_0(i_11, 12U, &kernelContext_20);
                float3 _S156 = _S155 - sa_0;
                float3 _S157 = vl_0(i_11, 12U, &kernelContext_20);
                thread float3 ma_0;
                thread float3 ml_0;
                inverse_apply_0(i_11, _S156, _S157 - sl_0, &ma_0, &ml_0, &kernelContext_20);
                float3 _S158 = va_0(i_11, 10U, &kernelContext_20);
                float3 _S159 = float3(1.68946564197540283f) ;
                float3 _S160 = float3(1.5220409631729126f) ;
                float3 _S161 = _S159 * _S158 - _S160 * ma_0;
                float3 _S162 = vl_0(i_11, 10U, &kernelContext_20);
                set_v_0(i_11, 8U, _S161, _S159 * _S162 - _S160 * ml_0, &kernelContext_20);
                p_7 = p_7 + 1024U;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        project_0(tid_8, n0_6, n1_6, 8U, free_1, &kernelContext_20);
        mloc_0 = 0.0f;
        p_6 = _S108;
        for(;;)
        {
            if(p_6 < n1_6)
            {
            }
            else
            {
                break;
            }
            uint i_12 = (&kernelContext_20)->lists_0[p_6];
            float3 _S163 = va_0(i_12, 8U, &kernelContext_20);
            float3 a_8 = abs(_S163);
            float3 _S164 = vl_0(i_12, 8U, &kernelContext_20);
            float3 l_7 = abs(_S164);
            float _S165 = max(mloc_0, max(max(max(a_8.x, a_8.y), max(a_8.z, l_7.x)), max(l_7.y, l_7.z)));
            uint p_8 = p_6 + 1024U;
            mloc_0 = _S165;
            p_6 = p_8;
        }
        float _S166 = group_max_0(tid_8, mloc_0, &kernelContext_20);
        bool _S167;
        if(!(_S166 > 0.0f))
        {
            _S167 = true;
        }
        else
        {
            _S167 = !isfinite(_S166);
        }
        if(_S167)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_1;
            break;
        }
        float _S168 = 1.0f / _S166;
        p_7 = _S108;
        float gl_0 = 0.0f;
        for(;;)
        {
            if(p_7 < n1_6)
            {
            }
            else
            {
                break;
            }
            uint i_13 = (&kernelContext_20)->lists_0[p_7];
            float3 _S169 = va_0(i_13, 8U, &kernelContext_20);
            float3 _S170 = float3(_S168) ;
            float3 ga_0 = _S169 * _S170;
            float3 _S171 = vl_0(i_13, 8U, &kernelContext_20);
            float3 gn_0 = _S171 * _S170;
            set_v_0(i_13, 8U, ga_0, gn_0, &kernelContext_20);
            float3 _S172 = va_0(i_13, 0U, &kernelContext_20);
            float _S173 = dot(_S172, ga_0);
            float3 _S174 = vl_0(i_13, 0U, &kernelContext_20);
            float gl_1 = gl_0 + (_S173 + dot(_S174, gn_0));
            p_7 = p_7 + 1024U;
            gl_0 = gl_1;
        }
        float2 _S175 = blast_group_sum_0(tid_8, float2(gl_0, 0.0f), &kernelContext_20);
        float gamma_0 = _S175.x;
        bool _S176 = gamma_0 > 0.0f;
        bool _S177;
        if(!_S176)
        {
            _S177 = true;
        }
        else
        {
            _S177 = !isfinite(gamma_0);
        }
        if(_S177)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_1;
            break;
        }
        bool _S178;
        if(iteration_0 > 1U)
        {
            _S178 = previous_1 > 0.0f;
        }
        else
        {
            _S178 = false;
        }
        if(_S178)
        {
            singular_0 = gamma_0 / previous_1;
        }
        else
        {
            singular_0 = 0.0f;
        }
        uint p_9 = _S108;
        for(;;)
        {
            if(p_9 < n1_6)
            {
            }
            else
            {
                break;
            }
            uint i_14 = (&kernelContext_20)->lists_0[p_9];
            if(singular_0 == 0.0f)
            {
                float3 _S179 = va_0(i_14, 8U, &kernelContext_20);
                float3 _S180 = vl_0(i_14, 8U, &kernelContext_20);
                set_v_0(i_14, 2U, _S179, _S180, &kernelContext_20);
            }
            else
            {
                float3 _S181 = va_0(i_14, 8U, &kernelContext_20);
                float3 _S182 = va_0(i_14, 2U, &kernelContext_20);
                float3 _S183 = _S181 + _S182 * float3(singular_0) ;
                float3 _S184 = vl_0(i_14, 8U, &kernelContext_20);
                float3 _S185 = vl_0(i_14, 2U, &kernelContext_20);
                set_v_0(i_14, 2U, _S183, _S184 + _S185 * float3(singular_0) , &kernelContext_20);
            }
            p_9 = p_9 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float _S186 = bond_pass_0(tid_8, b0_1, _S102, 2U, &kernelContext_20);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float _S187 = node_pass_0(tid_8, n0_6, n1_6, 4U, &kernelContext_20);
        bool _S188;
        if(((&kernelContext_20)->params_0->force_tolerance_0) > 0.0f)
        {
            if(_S186 > 0.0f)
            {
                _S188 = _S176;
            }
            else
            {
                _S188 = false;
            }
            float step_1;
            if(_S188)
            {
                step_1 = gamma_0 / sqrt(_S186);
            }
            else
            {
                step_1 = _S128;
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
            step_0 = step_1;
            travel_0 = travel_1;
        }
        uint _S189 = iteration_0 + 1U;
        if(!(_S186 > 0.0f))
        {
            _S188 = true;
        }
        else
        {
            _S188 = !isfinite(_S186);
        }
        if(_S188)
        {
            converged_0 = false;
            failed_0 = _S122;
            num_0 = num_1;
            iteration_0 = _S189;
            break;
        }
        float _S190 = gamma_0 / _S186;
        uint p_10 = _S108;
        for(;;)
        {
            if(p_10 < n1_6)
            {
            }
            else
            {
                break;
            }
            uint i_15 = (&kernelContext_20)->lists_0[p_10];
            float3 _S191 = va_0(i_15, 6U, &kernelContext_20);
            float3 _S192 = va_0(i_15, 2U, &kernelContext_20);
            float3 _S193 = float3(_S190) ;
            float3 _S194 = _S191 + _S192 * _S193;
            float3 _S195 = vl_0(i_15, 6U, &kernelContext_20);
            float3 _S196 = vl_0(i_15, 2U, &kernelContext_20);
            set_v_0(i_15, 6U, _S194, _S195 + _S196 * _S193, &kernelContext_20);
            float3 _S197 = va_0(i_15, 0U, &kernelContext_20);
            float3 _S198 = va_0(i_15, 4U, &kernelContext_20);
            float3 _S199 = _S197 - _S198 * _S193;
            float3 _S200 = vl_0(i_15, 0U, &kernelContext_20);
            float3 _S201 = vl_0(i_15, 4U, &kernelContext_20);
            set_v_0(i_15, 0U, _S199, _S200 - _S201 * _S193, &kernelContext_20);
            p_10 = p_10 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        previous_0 = gamma_0;
        num_0 = num_1;
        iteration_0 = _S189;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    q_1 = _S117;
    for(;;)
    {
        if(q_1 < _S102)
        {
        }
        else
        {
            break;
        }
        uint j_4 = (&kernelContext_20)->lists_0[q_1];
        thread BlastBond_natural_0 _S202 = *((&kernelContext_20)->bonds_0+j_4);
        if(!(((&_S202)->health_0) > 0.0f))
        {
            uint _S203 = j_4 * 2U;
            packed_float4 _S204 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_20)->x_0+_S203) = _S204;
            *((&kernelContext_20)->x_0+(_S203 + 1U)) = _S204;
            q_1 = q_1 + 1024U;
            continue;
        }
        thread float3 ta_5;
        thread float3 tl_5;
        bond_term_1(&_S202, 6U, &ta_5, &tl_5, &kernelContext_20);
        uint _S205 = j_4 * 2U;
        *((&kernelContext_20)->x_0+_S205) = packed_float4(float4((float4(*((&kernelContext_20)->x_0+_S205)) ).xyz + ta_5, 0.0f)) ;
        uint _S206 = _S205 + 1U;
        *((&kernelContext_20)->x_0+_S206) = packed_float4(float4((float4(*((&kernelContext_20)->x_0+_S206)) ).xyz + tl_5, 0.0f)) ;
        q_1 = q_1 + 1024U;
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
        p_6 = 1U;
    }
    else
    {
        p_6 = 0U;
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
        q_1 = 4U;
    }
    else
    {
        q_1 = 0U;
    }
    uint flags_0 = p_6 | q_1;
    if(tid_8 == 0U)
    {
        *((&kernelContext_20)->report_0+k_6) = packed_uint4(uint4(iteration_0, flags_0, (as_type<uint>((num_0))), (as_type<uint>((delta2_0))))) ;
    }
    return;
}

