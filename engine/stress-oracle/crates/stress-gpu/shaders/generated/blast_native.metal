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
    uint opaque_zero_0;
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
    float3 threadgroup* g_omega_0;
    float3 threadgroup* g_t_0;
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

uint blast_frame_0(uint node_count_1, uint k_2)
{
    return node_count_1 + 9U * k_2;
}

uint frame_word_0(uint k_3, uint w_1, KernelContext_0 thread* kernelContext_3)
{
    return (uint4(*(kernelContext_3->islands_0+(blast_frame_0(kernelContext_3->params_0->node_count_0, k_3) + w_1 / 4U))) )[w_1 % 4U];
}

uint frame_word_1(uint k_4, uint w_2, KernelContext_0 thread* kernelContext_4)
{
    return (uint4(*(kernelContext_4->islands_0+(blast_frame_0(kernelContext_4->params_0->node_count_0, k_4) + w_2 / 4U))) )[w_2 % 4U];
}

uint4 group_uniform4_0(uint4 value_1, uint thread_1)
{
    return value_1;
}

uint bs_0(uint j_0, KernelContext_0 thread* kernelContext_5)
{
    return kernelContext_5->params_0->node_count_0 * 24U + 2U * j_0;
}

float3 va_0(uint n_3, uint v_3, KernelContext_0 thread* kernelContext_6)
{
    return (float4(*(kernelContext_6->scratch_0+ns_0(n_3, v_3))) ).xyz;
}

float3 vl_0(uint n_4, uint v_4, KernelContext_0 thread* kernelContext_7)
{
    return (float4(*(kernelContext_7->scratch_0+(ns_0(n_4, v_4) + 1U))) ).xyz;
}

void bond_term_0(const BlastBond_natural_0 thread* b_2, uint v_5, float3 thread* ta_0, float3 thread* tl_0, KernelContext_0 thread* kernelContext_8)
{
    uint _S19 = b_2->n0_0;
    BlastNode_natural_0 m0_0 = kernelContext_8->nodes_0[b_2->n0_0];
    uint _S20 = b_2->n1_0;
    BlastNode_natural_0 m1_0 = kernelContext_8->nodes_0[b_2->n1_0];
    float3 _S21 = va_0(b_2->n0_0, v_5, kernelContext_8);
    float3 a0_1 = _S21 * float3(m0_0.w_ang_0) ;
    float3 _S22 = vl_0(_S19, v_5, kernelContext_8);
    float3 l0_0 = _S22 * float3(m0_0.w_lin_0) ;
    float3 _S23 = va_0(_S20, v_5, kernelContext_8);
    float3 a1_1 = _S23 * float3(m1_0.w_ang_0) ;
    float3 _S24 = vl_0(_S20, v_5, kernelContext_8);
    float3 l1_0 = _S24 * float3(m1_0.w_lin_0) ;
    float3 _S25 = float3(b_2->s_0) ;
    *ta_0 = (a0_1 - a1_1) * _S25;
    *tl_0 = (l0_0 - l1_0 + (cross((float4(b_2->o0_0) ).xyz, a0_1) - cross((float4(b_2->o1_0) ).xyz, a1_1))) * _S25;
    return;
}

void bond_term_1(const BlastBond_natural_0 thread* b_3, uint v_6, float3 thread* ta_1, float3 thread* tl_1, KernelContext_0 thread* kernelContext_9)
{
    uint _S26 = b_3->n0_0;
    BlastNode_natural_0 m0_1 = kernelContext_9->nodes_0[b_3->n0_0];
    uint _S27 = b_3->n1_0;
    BlastNode_natural_0 m1_1 = kernelContext_9->nodes_0[b_3->n1_0];
    float3 _S28 = va_0(b_3->n0_0, v_6, kernelContext_9);
    float3 a0_2 = _S28 * float3(m0_1.w_ang_0) ;
    float3 _S29 = vl_0(_S26, v_6, kernelContext_9);
    float3 l0_1 = _S29 * float3(m0_1.w_lin_0) ;
    float3 _S30 = va_0(_S27, v_6, kernelContext_9);
    float3 a1_2 = _S30 * float3(m1_1.w_ang_0) ;
    float3 _S31 = vl_0(_S27, v_6, kernelContext_9);
    float3 l1_1 = _S31 * float3(m1_1.w_lin_0) ;
    float3 _S32 = float3(b_3->s_0) ;
    *ta_1 = (a0_2 - a1_2) * _S32;
    *tl_1 = (l0_1 - l1_1 + (cross((float4(b_3->o0_0) ).xyz, a0_2) - cross((float4(b_3->o1_0) ).xyz, a1_2))) * _S32;
    return;
}

void gather_0(uint node_1, bool from_t_0, bool with_u_0, float3 thread* ang_0, float3 thread* lin_0, KernelContext_0 thread* kernelContext_10)
{
    BlastNode_natural_0 n_5 = kernelContext_10->nodes_0[node_1];
    float3 _S33 = float3(0.0f) ;
    uint r_1 = n_5.ref_begin_0;
    float3 acc_a_0 = _S33;
    float3 acc_l_0 = _S33;
    for(;;)
    {
        if(r_1 < (n_5.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_1 = kernelContext_10->refs_0[r_1];
        uint j_1 = ref_1 & 2147483647U;
        thread BlastBond_natural_0 _S34 = *(kernelContext_10->bonds_0+j_1);
        if(!(((&_S34)->health_0) > 0.0f))
        {
            r_1 = r_1 + 1U;
            continue;
        }
        float3 ya_0;
        float3 yl_0;
        if(from_t_0)
        {
            uint _S35 = bs_0(j_1, kernelContext_10);
            float3 _S36 = (float4(*(kernelContext_10->scratch_0+(_S35 + 1U))) ).xyz;
            ya_0 = (float4(*(kernelContext_10->scratch_0+_S35)) ).xyz;
            yl_0 = _S36;
        }
        else
        {
            uint _S37 = j_1 * 2U;
            float3 ya_1 = (float4(*(kernelContext_10->x_0+_S37)) ).xyz;
            float3 yl_1 = (float4(*(kernelContext_10->x_0+(_S37 + 1U))) ).xyz;
            if(with_u_0)
            {
                thread float3 ta_2;
                thread float3 tl_2;
                bond_term_0(&_S34, 6U, &ta_2, &tl_2, kernelContext_10);
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
        float3 _S38 = float3((&_S34)->s_0) ;
        float3 ya_2 = ya_0 * _S38;
        float3 yl_3 = yl_0 * _S38;
        float3 acc_a_1;
        float3 acc_l_1;
        if((ref_1 & 2147483648U) != 0U)
        {
            float3 acc_l_2 = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross((float4((&_S34)->o1_0) ).xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            float3 acc_l_3 = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross((float4((&_S34)->o0_0) ).xyz, yl_3));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_1 = r_1 + 1U;
    }
    *ang_0 = acc_a_0 * float3(n_5.w_ang_0) ;
    *lin_0 = acc_l_0 * float3(n_5.w_lin_0) ;
    return;
}

float rebuild_0(uint tid_1, uint n0_1, uint n1_1, bool with_u_1, KernelContext_0 thread* kernelContext_11)
{
    uint p_0 = n0_1 + tid_1;
    float s_3 = 0.0f;
    for(;;)
    {
        if(p_0 < n1_1)
        {
        }
        else
        {
            break;
        }
        uint i_0 = kernelContext_11->lists_0[p_0];
        thread float3 a_2;
        thread float3 l_1;
        gather_0(i_0, false, with_u_1, &a_2, &l_1, kernelContext_11);
        float3 _S39 = va_0(i_0, 14U, kernelContext_11);
        float3 ra_0 = _S39 - a_2;
        float3 _S40 = vl_0(i_0, 14U, kernelContext_11);
        float3 rl_0 = _S40 - l_1;
        set_v_0(i_0, 0U, ra_0, rl_0, kernelContext_11);
        float s_4 = s_3 + norm2_0(ra_0, rl_0);
        p_0 = p_0 + 1024U;
        s_3 = s_4;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    float2 _S41 = blast_group_sum_0(tid_1, float2(s_3, 0.0f), kernelContext_11);
    return _S41.x;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

float frame_float_0(uint k_5, uint w_3, KernelContext_0 thread* kernelContext_12)
{
    uint _S42 = frame_word_1(k_5, w_3, kernelContext_12);
    return (as_type<float>((_S42)));
}

float frame_float_1(uint k_6, uint w_4, KernelContext_0 thread* kernelContext_13)
{
    uint _S43 = frame_word_1(k_6, w_4, kernelContext_13);
    return (as_type<float>((_S43)));
}

float3 frame_axis_0(uint k_7, uint rotations_0, uint kk_1, KernelContext_0 thread* kernelContext_14)
{
    if(rotations_0 == 1U)
    {
        float _S44 = frame_float_1(k_7, 4U, kernelContext_14);
        float _S45 = frame_float_1(k_7, 5U, kernelContext_14);
        float _S46 = frame_float_1(k_7, 6U, kernelContext_14);
        return float3(_S44, _S45, _S46);
    }
    float _S47;
    if(kk_1 == 0U)
    {
        _S47 = 1.0f;
    }
    else
    {
        _S47 = 0.0f;
    }
    float _S48;
    if(kk_1 == 1U)
    {
        _S48 = 1.0f;
    }
    else
    {
        _S48 = 0.0f;
    }
    float _S49;
    if(kk_1 == 2U)
    {
        _S49 = 1.0f;
    }
    else
    {
        _S49 = 0.0f;
    }
    return float3(_S47, _S48, _S49);
}

void frame_terms_0(uint k_8, uint rotations_1, uint i_1, uint v_7, array<float, int(6)> thread* s_5, KernelContext_0 thread* kernelContext_15)
{
    BlastNode_natural_0 n_6 = kernelContext_15->nodes_0[i_1];
    float3 _S50 = vl_0(i_1, v_7, kernelContext_15);
    float3 lin_1 = _S50 / float3(n_6.w_lin_0) ;
    float3 _S51 = va_0(i_1, v_7, kernelContext_15);
    float3 _S52 = _S51 / float3(n_6.w_ang_0)  - cross((float4(n_6.position_0) ).xyz, lin_1);
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
        float3 _S53 = frame_axis_0(k_8, rotations_1, kk_2, kernelContext_15);
        (*s_5)[kk_2] = (*s_5)[kk_2] + dot(_S53, _S52);
        kk_2 = kk_2 + 1U;
    }
    uint j_2 = 0U;
    for(;;)
    {
        if(j_2 < 3U)
        {
        }
        else
        {
            break;
        }
        (*s_5)[rotations_1 + j_2] = (*s_5)[rotations_1 + j_2] + lin_1[j_2];
        j_2 = j_2 + 1U;
    }
    return;
}

void frame_solve_0(uint k_9, uint rotations_2, const array<float, int(6)> thread* sum_0, float3 thread* omega_0, float3 thread* t_0, KernelContext_0 thread* kernelContext_16)
{
    uint j_3;
    uint d_0 = rotations_2 + 3U;
    thread array<float, int(6)> c_1;
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
        c_1[z_0] = 0.0f;
        z_0 = z_0 + 1U;
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
        float _S54 = (*sum_0)[row_0];
        float _S55 = frame_float_0(k_9, 8U + row_0, kernelContext_16);
        float _S56 = _S54 * _S55;
        j_3 = 0U;
        float value_2 = _S56;
        for(;;)
        {
            if(j_3 < row_0)
            {
            }
            else
            {
                break;
            }
            float _S57 = frame_float_0(k_9, 14U + row_0 * (row_0 + 1U) / 2U + j_3, kernelContext_16);
            float value_3 = value_2 - _S57 * c_1[j_3];
            j_3 = j_3 + 1U;
            value_2 = value_3;
        }
        uint _S58 = row_0 + 1U;
        float _S59 = frame_float_0(k_9, 14U + row_0 * _S58 / 2U + row_0, kernelContext_16);
        c_1[row_0] = value_2 / _S59;
        row_0 = _S58;
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
        uint r_2 = uint(row_1);
        uint _S60 = r_2 + 1U;
        j_3 = _S60;
        for(;;)
        {
            if(j_3 < d_0)
            {
            }
            else
            {
                break;
            }
            uint _S61 = j_3 + 1U;
            float _S62 = frame_float_0(k_9, 14U + j_3 * _S61 / 2U + r_2, kernelContext_16);
            c_1[r_2] = c_1[r_2] - _S62 * c_1[j_3];
            j_3 = _S61;
        }
        float _S63 = frame_float_0(k_9, 14U + r_2 * _S60 / 2U + r_2, kernelContext_16);
        c_1[r_2] = c_1[r_2] / _S63;
        row_1 = row_1 - int(1);
    }
    z_0 = 0U;
    for(;;)
    {
        if(z_0 < d_0)
        {
        }
        else
        {
            break;
        }
        float _S64 = frame_float_0(k_9, 8U + z_0, kernelContext_16);
        c_1[z_0] = c_1[z_0] * _S64;
        z_0 = z_0 + 1U;
    }
    *omega_0 = float3(0.0f) ;
    j_3 = 0U;
    for(;;)
    {
        if(j_3 < rotations_2)
        {
        }
        else
        {
            break;
        }
        float3 _S65 = frame_axis_0(k_9, rotations_2, j_3, kernelContext_16);
        *omega_0 = *omega_0 + _S65 * float3(c_1[j_3]) ;
        j_3 = j_3 + 1U;
    }
    *t_0 = float3(c_1[rotations_2], c_1[rotations_2 + 1U], c_1[rotations_2 + 2U]);
    return;
}

void frame_apply_0(uint i_2, uint v_8, float3 omega_1, float3 t_1, KernelContext_0 thread* kernelContext_17)
{
    BlastNode_natural_0 n_7 = kernelContext_17->nodes_0[i_2];
    float3 _S66 = va_0(i_2, v_8, kernelContext_17);
    float3 _S67 = _S66 - omega_1 / float3(n_7.w_ang_0) ;
    float3 _S68 = vl_0(i_2, v_8, kernelContext_17);
    set_v_0(i_2, v_8, _S67, _S68 - (t_1 + cross((float4(n_7.position_0) ).xyz, omega_1)) / float3(n_7.w_lin_0) , kernelContext_17);
    return;
}

void project_0(uint tid_2, uint k_10, uint rotations_3, uint n0_2, uint n1_2, uint v_9, bool free_0, KernelContext_0 thread* kernelContext_18)
{
    if(!free_0)
    {
        return;
    }
    thread array<float, int(6)> s_6;
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
        s_6[z_1] = 0.0f;
        z_1 = z_1 + 1U;
    }
    uint _S69 = n0_2 + tid_2;
    uint p_1 = _S69;
    for(;;)
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_terms_0(k_10, rotations_3, kernelContext_18->lists_0[p_1], v_9, &s_6, kernelContext_18);
        p_1 = p_1 + 1024U;
    }
    float2 _S70 = blast_group_sum_0(tid_2, float2(s_6[int(0)], s_6[int(1)]), kernelContext_18);
    float2 _S71 = blast_group_sum_0(tid_2, float2(s_6[int(2)], s_6[int(3)]), kernelContext_18);
    float2 _S72 = blast_group_sum_0(tid_2, float2(s_6[int(4)], s_6[int(5)]), kernelContext_18);
    if(tid_2 == 0U)
    {
        float _S73 = _S70.y;
        float _S74 = _S71.x;
        float _S75 = _S71.y;
        float _S76 = _S72.x;
        float _S77 = _S72.y;
        thread array<float, int(6)> _S78;
        _S78[int(0)] = _S70.x;
        _S78[int(1)] = _S73;
        _S78[int(2)] = _S74;
        _S78[int(3)] = _S75;
        _S78[int(4)] = _S76;
        _S78[int(5)] = _S77;
        thread float3 omega_2;
        thread float3 t_2;
        frame_solve_0(k_10, rotations_3, &_S78, &omega_2, &t_2, kernelContext_18);
        *kernelContext_18->g_omega_0 = omega_2;
        *kernelContext_18->g_t_0 = t_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float3 _S79 = *kernelContext_18->g_omega_0;
    float3 _S80 = *kernelContext_18->g_t_0;
    p_1 = _S69;
    for(;;)
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_apply_0(kernelContext_18->lists_0[p_1], v_9, _S79, _S80, kernelContext_18);
        p_1 = p_1 + 1024U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return;
}

float bond_pass_0(uint tid_3, uint b0_0, uint b1_0, uint v_10, KernelContext_0 thread* kernelContext_19)
{
    uint q_0 = b0_0 + tid_3;
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
        uint j_4 = kernelContext_19->lists_0[q_0];
        thread BlastBond_natural_0 _S81 = *(kernelContext_19->bonds_0+j_4);
        float3 _S82 = float3(0.0f) ;
        thread float3 ta_3 = _S82;
        thread float3 tl_3 = _S82;
        if(((&_S81)->health_0) > 0.0f)
        {
            bond_term_0(&_S81, v_10, &ta_3, &tl_3, kernelContext_19);
        }
        float z_3 = z_2 + norm2_0(ta_3, tl_3);
        uint _S83 = bs_0(j_4, kernelContext_19);
        *(kernelContext_19->scratch_0+_S83) = packed_float4(float4(ta_3, 0.0f)) ;
        *(kernelContext_19->scratch_0+(_S83 + 1U)) = packed_float4(float4(tl_3, 0.0f)) ;
        q_0 = q_0 + 1024U;
        z_2 = z_3;
    }
    float2 _S84 = blast_group_sum_0(tid_3, float2(z_2, 0.0f), kernelContext_19);
    return _S84.x;
}

float residual_norm_0(uint tid_4, uint n0_3, uint n1_3, KernelContext_0 thread* kernelContext_20)
{
    uint p_2 = n0_3 + tid_4;
    float s_7 = 0.0f;
    for(;;)
    {
        if(p_2 < n1_3)
        {
        }
        else
        {
            break;
        }
        uint i_3 = kernelContext_20->lists_0[p_2];
        float3 _S85 = va_0(i_3, 0U, kernelContext_20);
        float3 _S86 = vl_0(i_3, 0U, kernelContext_20);
        float s_8 = s_7 + norm2_0(_S85, _S86);
        p_2 = p_2 + 1024U;
        s_7 = s_8;
    }
    float2 _S87 = blast_group_sum_0(tid_4, float2(s_7, 0.0f), kernelContext_20);
    return _S87.x;
}

float3 sym_mul_0(float4 s0_0, float4 s1_0, float3 v_11)
{
    float _S88 = v_11.x;
    float _S89 = s0_0.w;
    float _S90 = v_11.y;
    float _S91 = s1_0.x;
    float _S92 = v_11.z;
    float _S93 = s1_0.y;
    return float3(s0_0.x * _S88 + _S89 * _S90 + _S91 * _S92, _S89 * _S88 + s0_0.y * _S90 + _S93 * _S92, _S91 * _S88 + _S93 * _S90 + s0_0.z * _S92);
}

void inverse_apply_0(uint n_8, float3 a_3, float3 l_2, float3 thread* oa_0, float3 thread* ol_0, KernelContext_0 thread* kernelContext_21)
{
    uint _S94 = ns_0(n_8, 21U);
    float4 _S95 = float4(*(kernelContext_21->scratch_0+(_S94 + 1U))) ;
    float3 kc_0 = (float4(*(kernelContext_21->scratch_0+(_S94 + 2U))) ).xyz;
    float3 _S96 = sym_mul_0(float4(*(kernelContext_21->scratch_0+_S94)) , _S95, a_3 + cross(kc_0, l_2));
    *oa_0 = _S96;
    *ol_0 = l_2 * float3(_S95.z)  - cross(kc_0, _S96);
    return;
}

float node_pass_0(uint tid_5, uint n0_4, uint n1_4, uint out_v_0, KernelContext_0 thread* kernelContext_22)
{
    uint p_3 = n0_4 + tid_5;
    float s_9 = 0.0f;
    for(;;)
    {
        if(p_3 < n1_4)
        {
        }
        else
        {
            break;
        }
        uint i_4 = kernelContext_22->lists_0[p_3];
        thread float3 a_4;
        thread float3 l_3;
        gather_0(i_4, true, false, &a_4, &l_3, kernelContext_22);
        set_v_0(i_4, out_v_0, a_4, l_3, kernelContext_22);
        float s_10 = s_9 + norm2_0(a_4, l_3);
        p_3 = p_3 + 1024U;
        s_9 = s_10;
    }
    return s_9;
}

void block_apply_0(uint n_9, float3 a_5, float3 l_4, float3 thread* oa_1, float3 thread* ol_1, KernelContext_0 thread* kernelContext_23)
{
    uint _S97 = ns_0(n_9, 18U);
    float4 _S98 = float4(*(kernelContext_23->scratch_0+(_S97 + 1U))) ;
    float3 k_11 = (float4(*(kernelContext_23->scratch_0+(_S97 + 2U))) ).xyz;
    *oa_1 = sym_mul_0(float4(*(kernelContext_23->scratch_0+_S97)) , _S98, a_5) - cross(k_11, l_4);
    *ol_1 = cross(k_11, a_5) + float3(_S98.z)  * l_4;
    return;
}

float group_max_0(uint tid_6, float v_12, KernelContext_0 thread* kernelContext_24)
{
    (*kernelContext_24->g_max_0)[tid_6] = v_12;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint k_12 = 512U;
    for(;;)
    {
        if(k_12 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_6 < k_12)
        {
            (*kernelContext_24->g_max_0)[tid_6] = max((*kernelContext_24->g_max_0)[tid_6], (*kernelContext_24->g_max_0)[tid_6 + k_12]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_12 = k_12 >> 1U;
    }
    if(tid_6 == 0U)
    {
        *kernelContext_24->g_max_out_0 = (*kernelContext_24->g_max_0)[int(0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threadgroup* _S99 = kernelContext_24->g_max_out_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *_S99;
}

[[kernel]] void blast_native(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_1 [[buffer(0)]], packed_uint4 device* islands_1 [[buffer(1)]], packed_uint4 device* report_1 [[buffer(8)]], uint device* lists_1 [[buffer(9)]], packed_float4 device* accel_1 [[buffer(5)]], packed_float4 device* scratch_1 [[buffer(7)]], BlastNode_natural_0 device* nodes_1 [[buffer(2)]], uint device* refs_1 [[buffer(3)]], BlastBond_natural_0 device* bonds_1 [[buffer(4)]], packed_float4 device* x_1 [[buffer(6)]])
{
    float force_start_0;
    float best_0;
    bool failed_0;
    uint q_1;
    bool ok_0;
    thread KernelContext_0 kernelContext_25;
    (&kernelContext_25)->params_0 = params_1;
    (&kernelContext_25)->islands_0 = islands_1;
    (&kernelContext_25)->report_0 = report_1;
    (&kernelContext_25)->lists_0 = lists_1;
    (&kernelContext_25)->accel_0 = accel_1;
    (&kernelContext_25)->scratch_0 = scratch_1;
    (&kernelContext_25)->nodes_0 = nodes_1;
    (&kernelContext_25)->refs_0 = refs_1;
    (&kernelContext_25)->bonds_0 = bonds_1;
    (&kernelContext_25)->x_0 = x_1;
    threadgroup array<float2, int(32)> g_blast_part_1;
    (&kernelContext_25)->g_blast_part_0 = &g_blast_part_1;
    threadgroup float3 g_omega_1;
    (&kernelContext_25)->g_omega_0 = &g_omega_1;
    threadgroup float3 g_t_1;
    (&kernelContext_25)->g_t_0 = &g_t_1;
    threadgroup array<float, int(1024)> g_max_1;
    (&kernelContext_25)->g_max_0 = &g_max_1;
    threadgroup float g_max_out_1;
    (&kernelContext_25)->g_max_out_0 = &g_max_out_1;
    uint tid_7 = thread_2.x;
    uint k_13 = group_0.x;
    if(k_13 >= (params_1->island_count_0))
    {
        return;
    }
    uint4 _S100 = uint4(*((&kernelContext_25)->islands_0+k_13)) ;
    uint n0_5 = _S100.x;
    uint n1_5 = _S100.y;
    uint b0_1 = _S100.z;
    uint _S101 = _S100.w;
    uint p_4;
    if((n1_5 - n0_5) > ((&kernelContext_25)->params_0->wide_min_0))
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
    uint4 _S102 = uint4(*((&kernelContext_25)->report_0+k_13)) ;
    uint last_flags_0 = group_uniform_0(_S102.y, tid_7);
    bool warm_1 = ((&kernelContext_25)->params_0->warm_0) != 0U;
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
        uint _S103 = n0_5 + tid_7;
        singular_0 = 0.0f;
        p_4 = _S103;
        for(;;)
        {
            if(p_4 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_5 = (&kernelContext_25)->lists_0[p_4];
            uint _S104 = i_5 * 2U;
            uint _S105 = ns_0(i_5, 16U);
            if(any((float4(*((&kernelContext_25)->accel_0+_S104)) ) != (float4(*((&kernelContext_25)->scratch_0+_S105)) )))
            {
                converged_0 = true;
            }
            else
            {
                converged_0 = any((float4(*((&kernelContext_25)->accel_0+(_S104 + 1U))) ) != (float4(*((&kernelContext_25)->scratch_0+(_S105 + 1U))) ));
            }
            if(converged_0)
            {
                singular_0 = 1.0f;
            }
            p_4 = p_4 + 1024U;
        }
        float2 _S106 = blast_group_sum_0(tid_7, float2(singular_0, 0.0f), &kernelContext_25);
        if((_S106.x) == 0.0f)
        {
            if(tid_7 == 0U)
            {
                *((&kernelContext_25)->report_0+k_13) = packed_uint4(uint4(0U, 7U, _S102.z, _S102.w)) ;
            }
            return;
        }
    }
    uint _S107 = n0_5 + tid_7;
    singular_0 = 0.0f;
    p_4 = _S107;
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
        uint i_6 = (&kernelContext_25)->lists_0[p_4];
        BlastNode_natural_0 n_10 = (&kernelContext_25)->nodes_0[i_6];
        uint _S108 = i_6 * 2U;
        float4 _S109 = float4(*((&kernelContext_25)->accel_0+_S108)) ;
        float4 _S110 = float4(*((&kernelContext_25)->accel_0+(_S108 + 1U))) ;
        float3 _S111 = - _S109.xyz;
        if((n_10.w_ang_0) > 0.0f)
        {
            force_start_0 = 1.0f / n_10.w_ang_0;
        }
        else
        {
            force_start_0 = 1.0f;
        }
        float3 ba_0 = _S111 * float3(force_start_0) ;
        float3 _S112 = - _S110.xyz;
        if((n_10.w_lin_0) > 0.0f)
        {
            best_0 = (&kernelContext_25)->params_0->recip_length_0 / n_10.w_lin_0;
        }
        else
        {
            best_0 = (&kernelContext_25)->params_0->recip_length_0;
        }
        float3 bl_0 = _S112 * float3(best_0) ;
        float target_1 = target_0 + norm2_0(ba_0, bl_0);
        set_v_0(i_6, 14U, ba_0, bl_0, &kernelContext_25);
        set_v_0(i_6, 0U, ba_0, bl_0, &kernelContext_25);
        uint _S113 = ns_0(i_6, 16U);
        *((&kernelContext_25)->scratch_0+_S113) = packed_float4(_S109) ;
        *((&kernelContext_25)->scratch_0+(_S113 + 1U)) = packed_float4(_S110) ;
        float3 _S114 = float3(0.0f) ;
        set_v_0(i_6, 6U, _S114, _S114, &kernelContext_25);
        set_v_0(i_6, 2U, _S114, _S114, &kernelContext_25);
        bool _S115 = build_block_0(i_6, &kernelContext_25);
        if(!_S115)
        {
            singular_0 = 1.0f;
        }
        p_4 = p_4 + 1024U;
        target_0 = target_1;
    }
    uint q_2;
    if(!warm_1)
    {
        q_2 = b0_1 + tid_7;
        for(;;)
        {
            if(q_2 < _S101)
            {
            }
            else
            {
                break;
            }
            uint _S116 = (&kernelContext_25)->lists_0[q_2] * 2U;
            packed_float4 _S117 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_25)->x_0+_S116) = _S117;
            *((&kernelContext_25)->x_0+(_S116 + 1U)) = _S117;
            q_2 = q_2 + 1024U;
        }
    }
    float2 _S118 = blast_group_sum_0(tid_7, float2(target_0, singular_0), &kernelContext_25);
    float delta2_0 = _S118.x * (&kernelContext_25)->params_0->tolerance_0 * (&kernelContext_25)->params_0->tolerance_0;
    uint _S119 = frame_word_0(k_13, 0U, &kernelContext_25);
    uint _S120 = frame_word_0(k_13, 1U, &kernelContext_25);
    uint _S121 = frame_word_0(k_13, 2U, &kernelContext_25);
    uint3 frame_0 = group_uniform4_0(uint4(_S119, _S120, _S121, 0U), tid_7).xyz;
    if((_S118.y) > 0.0f)
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = (frame_0.z) != 0U;
    }
    bool _S122 = (frame_0.x) == 0U;
    uint _S123 = frame_0.y;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if(!(delta2_0 > 0.0f))
    {
        q_2 = b0_1 + tid_7;
        for(;;)
        {
            if(q_2 < _S101)
            {
            }
            else
            {
                break;
            }
            uint _S124 = (&kernelContext_25)->lists_0[q_2] * 2U;
            packed_float4 _S125 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_25)->x_0+_S124) = _S125;
            *((&kernelContext_25)->x_0+(_S124 + 1U)) = _S125;
            q_2 = q_2 + 1024U;
        }
        if(tid_7 == 0U)
        {
            packed_uint4 device* _S126 = (&kernelContext_25)->report_0+k_13;
            if(warm_1)
            {
                p_4 = 4U;
            }
            else
            {
                p_4 = 0U;
            }
            *_S126 = packed_uint4(uint4(0U, 1U | p_4, (as_type<uint>((0.0f))), (as_type<uint>((delta2_0))))) ;
        }
        return;
    }
    if(warm_1)
    {
        float _S127 = rebuild_0(tid_7, n0_5, n1_5, false, &kernelContext_25);
    }
    float _S128 = float_bits_0(2139095040U);
    if(((&kernelContext_25)->params_0->force_tolerance_0) > 0.0f)
    {
        q_2 = b0_1 + tid_7;
        force_start_0 = 0.0f;
        for(;;)
        {
            if(q_2 < _S101)
            {
            }
            else
            {
                break;
            }
            uint _S129 = (&kernelContext_25)->lists_0[q_2] * 2U;
            float f_0 = force_start_0 + norm2_0((float4(*((&kernelContext_25)->x_0+_S129)) ).xyz, (float4(*((&kernelContext_25)->x_0+(_S129 + 1U))) ).xyz);
            q_2 = q_2 + 1024U;
            force_start_0 = f_0;
        }
        float2 _S130 = blast_group_sum_0(tid_7, float2(force_start_0, 0.0f), &kernelContext_25);
        force_start_0 = sqrt(_S130.x);
    }
    else
    {
        force_start_0 = 0.0f;
    }
    best_0 = _S128;
    q_2 = 0U;
    float step_0 = _S128;
    float previous_0 = 0.0f;
    float num_0 = 0.0f;
    uint iteration_0 = 0U;
    float travel_0 = 0.0f;
    for(;;)
    {
        if(!converged_0)
        {
            failed_0 = iteration_0 < ((&kernelContext_25)->params_0->max_iterations_0);
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
            bool _S131 = converged_0;
            converged_0 = false;
            failed_0 = _S131;
            break;
        }
        project_0(tid_7, k_13, _S123, n0_5, n1_5, 0U, _S122, &kernelContext_25);
        float _S132 = bond_pass_0(tid_7, b0_1, _S101, 0U, &kernelContext_25);
        bool _S133 = _S132 <= delta2_0;
        if(_S133)
        {
            float _S134 = residual_norm_0(tid_7, n0_5, n1_5, &kernelContext_25);
            ok_0 = !(_S134 <= delta2_0);
        }
        else
        {
            ok_0 = false;
        }
        bool _S135 = iteration_0 > 0U;
        bool _S136;
        if(_S135)
        {
            _S136 = true;
        }
        else
        {
            _S136 = warm_1;
        }
        bool _S137;
        if(_S136)
        {
            _S137 = _S133;
        }
        else
        {
            _S137 = false;
        }
        bool _S138;
        if(_S137)
        {
            _S138 = !ok_0;
        }
        else
        {
            _S138 = false;
        }
        bool hold_0;
        float num_1;
        float previous_1;
        if(_S138)
        {
            float _S139 = rebuild_0(tid_7, n0_5, n1_5, true, &kernelContext_25);
            project_0(tid_7, k_13, _S123, n0_5, n1_5, 0U, _S122, &kernelContext_25);
            float _S140 = bond_pass_0(tid_7, b0_1, _S101, 0U, &kernelContext_25);
            if(_S140 <= delta2_0)
            {
                float _S141 = residual_norm_0(tid_7, n0_5, n1_5, &kernelContext_25);
                hold_0 = !(_S141 <= delta2_0);
            }
            else
            {
                hold_0 = false;
            }
            num_1 = _S140;
            previous_1 = 0.0f;
        }
        else
        {
            num_1 = _S132;
            hold_0 = ok_0;
            previous_1 = previous_0;
        }
        if(num_1 < (best_0 * 0.99000000953674316f))
        {
            best_0 = num_1;
            q_2 = iteration_0;
        }
        else
        {
            if((iteration_0 - q_2) >= 512U)
            {
                bool _S142 = converged_0;
                converged_0 = false;
                failed_0 = _S142;
                num_0 = num_1;
                break;
            }
        }
        bool _S143;
        if(num_1 <= delta2_0)
        {
            _S143 = !hold_0;
        }
        else
        {
            _S143 = false;
        }
        if(_S143)
        {
            bool _S144 = converged_0;
            converged_0 = true;
            failed_0 = _S144;
            num_0 = num_1;
            break;
        }
        bool _S145;
        if(((&kernelContext_25)->params_0->force_tolerance_0) > 0.0f)
        {
            _S145 = _S135;
        }
        else
        {
            _S145 = false;
        }
        bool _S146;
        if(_S145)
        {
            _S146 = step_0 <= ((&kernelContext_25)->params_0->force_tolerance_0 * (force_start_0 + travel_0));
        }
        else
        {
            _S146 = false;
        }
        float mloc_0;
        if(_S146)
        {
            q_1 = b0_1 + tid_7;
            mloc_0 = 0.0f;
            for(;;)
            {
                if(q_1 < _S101)
                {
                }
                else
                {
                    break;
                }
                uint j_5 = (&kernelContext_25)->lists_0[q_1];
                thread BlastBond_natural_0 _S147 = *((&kernelContext_25)->bonds_0+j_5);
                thread float3 ta_4;
                thread float3 tl_4;
                bond_term_1(&_S147, 6U, &ta_4, &tl_4, &kernelContext_25);
                uint _S148 = j_5 * 2U;
                float f_1 = mloc_0 + norm2_0((float4(*((&kernelContext_25)->x_0+_S148)) ).xyz + ta_4, (float4(*((&kernelContext_25)->x_0+(_S148 + 1U))) ).xyz + tl_4);
                q_1 = q_1 + 1024U;
                mloc_0 = f_1;
            }
            float _S149 = (&kernelContext_25)->params_0->force_tolerance_0;
            float2 _S150 = blast_group_sum_0(tid_7, float2(mloc_0, 0.0f), &kernelContext_25);
            if(step_0 <= (_S149 * sqrt(_S150.x)))
            {
                bool _S151 = converged_0;
                converged_0 = true;
                failed_0 = _S151;
                num_0 = num_1;
                break;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if(iteration_0 == 0U)
        {
            p_4 = _S107;
            for(;;)
            {
                if(p_4 < n1_5)
                {
                }
                else
                {
                    break;
                }
                uint i_7 = (&kernelContext_25)->lists_0[p_4];
                float3 _S152 = va_0(i_7, 0U, &kernelContext_25);
                float3 _S153 = vl_0(i_7, 0U, &kernelContext_25);
                set_v_0(i_7, 8U, _S152, _S153, &kernelContext_25);
                p_4 = p_4 + 1024U;
            }
        }
        else
        {
            p_4 = _S107;
            for(;;)
            {
                if(p_4 < n1_5)
                {
                }
                else
                {
                    break;
                }
                uint i_8 = (&kernelContext_25)->lists_0[p_4];
                float3 _S154 = va_0(i_8, 0U, &kernelContext_25);
                float3 _S155 = vl_0(i_8, 0U, &kernelContext_25);
                thread float3 a_6;
                thread float3 l_5;
                inverse_apply_0(i_8, _S154, _S155, &a_6, &l_5, &kernelContext_25);
                set_v_0(i_8, 10U, a_6, l_5, &kernelContext_25);
                p_4 = p_4 + 1024U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            float _S156 = bond_pass_0(tid_7, b0_1, _S101, 10U, &kernelContext_25);
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            float _S157 = node_pass_0(tid_7, n0_5, n1_5, 12U, &kernelContext_25);
            q_1 = _S107;
            for(;;)
            {
                if(q_1 < n1_5)
                {
                }
                else
                {
                    break;
                }
                uint i_9 = (&kernelContext_25)->lists_0[q_1];
                float3 _S158 = va_0(i_9, 10U, &kernelContext_25);
                float3 _S159 = vl_0(i_9, 10U, &kernelContext_25);
                thread float3 sa_0;
                thread float3 sl_0;
                block_apply_0(i_9, _S158, _S159, &sa_0, &sl_0, &kernelContext_25);
                float3 _S160 = va_0(i_9, 12U, &kernelContext_25);
                float3 _S161 = _S160 - sa_0;
                float3 _S162 = vl_0(i_9, 12U, &kernelContext_25);
                thread float3 ma_0;
                thread float3 ml_0;
                inverse_apply_0(i_9, _S161, _S162 - sl_0, &ma_0, &ml_0, &kernelContext_25);
                float3 _S163 = va_0(i_9, 10U, &kernelContext_25);
                float3 _S164 = float3(1.68946564197540283f) ;
                float3 _S165 = float3(1.5220409631729126f) ;
                float3 _S166 = _S164 * _S163 - _S165 * ma_0;
                float3 _S167 = vl_0(i_9, 10U, &kernelContext_25);
                set_v_0(i_9, 8U, _S166, _S164 * _S167 - _S165 * ml_0, &kernelContext_25);
                q_1 = q_1 + 1024U;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        project_0(tid_7, k_13, _S123, n0_5, n1_5, 8U, _S122, &kernelContext_25);
        mloc_0 = 0.0f;
        p_4 = _S107;
        for(;;)
        {
            if(p_4 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_10 = (&kernelContext_25)->lists_0[p_4];
            float3 _S168 = va_0(i_10, 8U, &kernelContext_25);
            float3 a_7 = abs(_S168);
            float3 _S169 = vl_0(i_10, 8U, &kernelContext_25);
            float3 l_6 = abs(_S169);
            float _S170 = max(mloc_0, max(max(max(a_7.x, a_7.y), max(a_7.z, l_6.x)), max(l_6.y, l_6.z)));
            uint p_5 = p_4 + 1024U;
            mloc_0 = _S170;
            p_4 = p_5;
        }
        float _S171 = group_max_0(tid_7, mloc_0, &kernelContext_25);
        bool _S172;
        if(!(_S171 > 0.0f))
        {
            _S172 = true;
        }
        else
        {
            _S172 = !isfinite(_S171);
        }
        if(_S172)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_1;
            break;
        }
        float _S173 = 1.0f / _S171;
        q_1 = _S107;
        float gl_0 = 0.0f;
        for(;;)
        {
            if(q_1 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_11 = (&kernelContext_25)->lists_0[q_1];
            float3 _S174 = va_0(i_11, 8U, &kernelContext_25);
            float3 _S175 = float3(_S173) ;
            float3 ga_0 = _S174 * _S175;
            float3 _S176 = vl_0(i_11, 8U, &kernelContext_25);
            float3 gn_0 = _S176 * _S175;
            set_v_0(i_11, 8U, ga_0, gn_0, &kernelContext_25);
            float3 _S177 = va_0(i_11, 0U, &kernelContext_25);
            float _S178 = dot(_S177, ga_0);
            float3 _S179 = vl_0(i_11, 0U, &kernelContext_25);
            float gl_1 = gl_0 + (_S178 + dot(_S179, gn_0));
            q_1 = q_1 + 1024U;
            gl_0 = gl_1;
        }
        float2 _S180 = blast_group_sum_0(tid_7, float2(gl_0, 0.0f), &kernelContext_25);
        float gamma_0 = _S180.x;
        bool _S181 = gamma_0 > 0.0f;
        bool _S182;
        if(!_S181)
        {
            _S182 = true;
        }
        else
        {
            _S182 = !isfinite(gamma_0);
        }
        if(_S182)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_1;
            break;
        }
        bool _S183;
        if(iteration_0 > 1U)
        {
            _S183 = previous_1 > 0.0f;
        }
        else
        {
            _S183 = false;
        }
        if(_S183)
        {
            singular_0 = gamma_0 / previous_1;
        }
        else
        {
            singular_0 = 0.0f;
        }
        uint p_6 = _S107;
        for(;;)
        {
            if(p_6 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_12 = (&kernelContext_25)->lists_0[p_6];
            if(singular_0 == 0.0f)
            {
                float3 _S184 = va_0(i_12, 8U, &kernelContext_25);
                float3 _S185 = vl_0(i_12, 8U, &kernelContext_25);
                set_v_0(i_12, 2U, _S184, _S185, &kernelContext_25);
            }
            else
            {
                float3 _S186 = va_0(i_12, 8U, &kernelContext_25);
                float3 _S187 = va_0(i_12, 2U, &kernelContext_25);
                float3 _S188 = _S186 + _S187 * float3(singular_0) ;
                float3 _S189 = vl_0(i_12, 8U, &kernelContext_25);
                float3 _S190 = vl_0(i_12, 2U, &kernelContext_25);
                set_v_0(i_12, 2U, _S188, _S189 + _S190 * float3(singular_0) , &kernelContext_25);
            }
            p_6 = p_6 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float _S191 = bond_pass_0(tid_7, b0_1, _S101, 2U, &kernelContext_25);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float _S192 = node_pass_0(tid_7, n0_5, n1_5, 4U, &kernelContext_25);
        bool _S193;
        if(((&kernelContext_25)->params_0->force_tolerance_0) > 0.0f)
        {
            if(_S191 > 0.0f)
            {
                _S193 = _S181;
            }
            else
            {
                _S193 = false;
            }
            float step_1;
            if(_S193)
            {
                step_1 = gamma_0 / sqrt(_S191);
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
        uint _S194 = iteration_0 + 1U;
        if(!(_S191 > 0.0f))
        {
            _S193 = true;
        }
        else
        {
            _S193 = !isfinite(_S191);
        }
        if(_S193)
        {
            bool _S195 = converged_0;
            converged_0 = false;
            failed_0 = _S195;
            num_0 = num_1;
            iteration_0 = _S194;
            break;
        }
        float _S196 = gamma_0 / _S191;
        uint p_7 = _S107;
        for(;;)
        {
            if(p_7 < n1_5)
            {
            }
            else
            {
                break;
            }
            uint i_13 = (&kernelContext_25)->lists_0[p_7];
            float3 _S197 = va_0(i_13, 6U, &kernelContext_25);
            float3 _S198 = va_0(i_13, 2U, &kernelContext_25);
            float3 _S199 = float3(_S196) ;
            float3 _S200 = _S197 + _S198 * _S199;
            float3 _S201 = vl_0(i_13, 6U, &kernelContext_25);
            float3 _S202 = vl_0(i_13, 2U, &kernelContext_25);
            set_v_0(i_13, 6U, _S200, _S201 + _S202 * _S199, &kernelContext_25);
            float3 _S203 = va_0(i_13, 0U, &kernelContext_25);
            float3 _S204 = va_0(i_13, 4U, &kernelContext_25);
            float3 _S205 = _S203 - _S204 * _S199;
            float3 _S206 = vl_0(i_13, 0U, &kernelContext_25);
            float3 _S207 = vl_0(i_13, 4U, &kernelContext_25);
            set_v_0(i_13, 0U, _S205, _S206 - _S207 * _S199, &kernelContext_25);
            p_7 = p_7 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        previous_0 = gamma_0;
        num_0 = num_1;
        iteration_0 = _S194;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    q_1 = b0_1 + tid_7;
    for(;;)
    {
        if(q_1 < _S101)
        {
        }
        else
        {
            break;
        }
        uint j_6 = (&kernelContext_25)->lists_0[q_1];
        thread BlastBond_natural_0 _S208 = *((&kernelContext_25)->bonds_0+j_6);
        if(!(((&_S208)->health_0) > 0.0f))
        {
            uint _S209 = j_6 * 2U;
            packed_float4 _S210 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_25)->x_0+_S209) = _S210;
            *((&kernelContext_25)->x_0+(_S209 + 1U)) = _S210;
            q_1 = q_1 + 1024U;
            continue;
        }
        thread float3 ta_5;
        thread float3 tl_5;
        bond_term_1(&_S208, 6U, &ta_5, &tl_5, &kernelContext_25);
        uint _S211 = j_6 * 2U;
        *((&kernelContext_25)->x_0+_S211) = packed_float4(float4((float4(*((&kernelContext_25)->x_0+_S211)) ).xyz + ta_5, 0.0f)) ;
        uint _S212 = _S211 + 1U;
        *((&kernelContext_25)->x_0+_S212) = packed_float4(float4((float4(*((&kernelContext_25)->x_0+_S212)) ).xyz + tl_5, 0.0f)) ;
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
        q_2 = 4U;
    }
    else
    {
        q_2 = 0U;
    }
    uint flags_0 = p_4 | q_2;
    if(tid_7 == 0U)
    {
        *((&kernelContext_25)->report_0+k_13) = packed_uint4(uint4(iteration_0, flags_0, (as_type<uint>((num_0))), (as_type<uint>((delta2_0))))) ;
    }
    return;
}

