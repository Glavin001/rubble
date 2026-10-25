#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint group_uniform_0(uint value_0, uint thread_0)
{
    return value_0;
}

uint node_slot_0(uint node_0, uint vector_0)
{
    return node_0 * 10U + vector_0;
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

struct BlastNode_0
{
    float w_ang_0;
    float w_lin_0;
    uint ref_begin_0;
    uint ref_end_0;
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
    packed_float4 device* accel_0;
    packed_float4 device* scratch_0;
    BlastNode_0 device* nodes_0;
    packed_float4 device* x_0;
    uint device* refs_0;
    BlastBond_natural_0 device* bonds_0;
    array<float2, int(32)> threadgroup* g_blast_part_0;
};

float2 blast_group_sum_0(uint tid_0, float2 v_0, KernelContext_0 thread* kernelContext_0)
{
    float2 w_0 = simd_sum(v_0);
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
        if(k_0 < 8U)
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

uint bond_slot_0(uint node_count_1, uint bond_0)
{
    return node_count_1 * 10U + bond_0 * 2U;
}

void gather_0(uint node_1, bool from_x_0, float3 thread* ang_0, float3 thread* lin_0, KernelContext_0 thread* kernelContext_1)
{
    BlastNode_0 n_0 = kernelContext_1->nodes_0[node_1];
    float3 _S2 = float3(0.0f) ;
    uint r_0 = n_0.ref_begin_0;
    float3 acc_a_0 = _S2;
    float3 acc_l_0 = _S2;
    for(;;)
    {
        if(r_0 < (n_0.ref_end_0))
        {
        }
        else
        {
            break;
        }
        uint ref_0 = kernelContext_1->refs_0[r_0];
        uint j_0 = ref_0 & 2147483647U;
        BlastBond_natural_0 device* _S3 = kernelContext_1->bonds_0+j_0;
        BlastBond_natural_0 b_1 = *_S3;
        if(!(((*_S3).health_0) > 0.0f))
        {
            r_0 = r_0 + 1U;
            continue;
        }
        float3 yl_0;
        float3 ya_0;
        if(from_x_0)
        {
            uint _S4 = j_0 * 2U;
            float3 _S5 = float3(b_1.s_0) ;
            float3 _S6 = (float4(*(kernelContext_1->x_0+_S4)) ).xyz * _S5;
            yl_0 = (float4(*(kernelContext_1->x_0+(_S4 + 1U))) ).xyz * _S5;
            ya_0 = _S6;
        }
        else
        {
            uint t_0 = bond_slot_0(kernelContext_1->params_0->node_count_0, j_0);
            float3 _S7 = float3(b_1.s_0) ;
            float3 _S8 = (float4(*(kernelContext_1->scratch_0+t_0)) ).xyz * _S7;
            yl_0 = (float4(*(kernelContext_1->scratch_0+(t_0 + 1U))) ).xyz * _S7;
            ya_0 = _S8;
        }
        float3 acc_a_1;
        float3 acc_l_1;
        if((ref_0 & 2147483648U) != 0U)
        {
            float3 acc_l_2 = acc_l_0 - yl_0;
            acc_a_1 = acc_a_0 + (cross((float4(b_1.o1_0) ).xyz, yl_0) - ya_0);
            acc_l_1 = acc_l_2;
        }
        else
        {
            float3 acc_l_3 = acc_l_0 + yl_0;
            acc_a_1 = acc_a_0 + (ya_0 - cross((float4(b_1.o0_0) ).xyz, yl_0));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_0 = r_0 + 1U;
    }
    *ang_0 = acc_a_0 * float3(n_0.w_ang_0) ;
    *lin_0 = acc_l_0 * float3(n_0.w_lin_0) ;
    return;
}

void bond_term_0(const BlastBond_natural_0 thread* b_2, uint vec_0, float3 thread* ang_1, float3 thread* lin_1, KernelContext_0 thread* kernelContext_2)
{
    float3 _S9 = float3(0.0f) ;
    uint _S10 = b_2->n0_0;
    float3 a0_0;
    float3 l0_0;
    if((b_2->n0_0) != 4294967295U)
    {
        BlastNode_0 n_1 = kernelContext_2->nodes_0[_S10];
        uint _S11 = node_slot_0(_S10, vec_0);
        float3 _S12 = (float4(*(kernelContext_2->scratch_0+(_S11 + 1U))) ).xyz * float3(n_1.w_lin_0) ;
        a0_0 = (float4(*(kernelContext_2->scratch_0+_S11)) ).xyz * float3(n_1.w_ang_0) ;
        l0_0 = _S12;
    }
    else
    {
        a0_0 = _S9;
        l0_0 = _S9;
    }
    uint _S13 = b_2->n1_0;
    float3 a1_0;
    float3 l1_0;
    if((b_2->n1_0) != 4294967295U)
    {
        BlastNode_0 n_2 = kernelContext_2->nodes_0[_S13];
        uint _S14 = node_slot_0(_S13, vec_0);
        float3 _S15 = (float4(*(kernelContext_2->scratch_0+(_S14 + 1U))) ).xyz * float3(n_2.w_lin_0) ;
        a1_0 = (float4(*(kernelContext_2->scratch_0+_S14)) ).xyz * float3(n_2.w_ang_0) ;
        l1_0 = _S15;
    }
    else
    {
        a1_0 = _S9;
        l1_0 = _S9;
    }
    float3 _S16 = float3(b_2->s_0) ;
    *ang_1 = (a0_0 - a1_0) * _S16;
    *lin_1 = (l0_0 - l1_0 + (cross((float4(b_2->o0_0) ).xyz, a0_0) - cross((float4(b_2->o1_0) ).xyz, a1_0))) * _S16;
    return;
}

[[kernel]] void blast_solve(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], BlastParams_0 constant* params_1 [[buffer(0)]], packed_uint4 device* islands_1 [[buffer(1)]], packed_uint4 device* report_1 [[buffer(8)]], packed_float4 device* accel_1 [[buffer(5)]], packed_float4 device* scratch_1 [[buffer(7)]], BlastNode_0 device* nodes_1 [[buffer(2)]], packed_float4 device* x_1 [[buffer(6)]], uint device* refs_1 [[buffer(3)]], BlastBond_natural_0 device* bonds_1 [[buffer(4)]])
{
    float ia_0;
    float il_0;
    uint v_1;
    uint j_1;
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->params_0 = params_1;
    (&kernelContext_3)->islands_0 = islands_1;
    (&kernelContext_3)->report_0 = report_1;
    (&kernelContext_3)->accel_0 = accel_1;
    (&kernelContext_3)->scratch_0 = scratch_1;
    (&kernelContext_3)->nodes_0 = nodes_1;
    (&kernelContext_3)->x_0 = x_1;
    (&kernelContext_3)->refs_0 = refs_1;
    (&kernelContext_3)->bonds_0 = bonds_1;
    threadgroup array<float2, int(32)> g_blast_part_1;
    (&kernelContext_3)->g_blast_part_0 = &g_blast_part_1;
    uint tid_1 = thread_1.x;
    uint k_1 = group_0.x;
    if(k_1 >= (params_1->island_count_0))
    {
        return;
    }
    uint4 _S17 = uint4(*((&kernelContext_3)->islands_0+k_1)) ;
    uint n0_1 = _S17.x;
    uint _S18 = _S17.y;
    uint b0_0 = _S17.z;
    uint _S19 = _S17.w;
    uint4 _S20 = uint4(*((&kernelContext_3)->report_0+k_1)) ;
    uint last_flags_0 = group_uniform_0(_S20.y, tid_1);
    bool warm_1 = ((&kernelContext_3)->params_0->warm_0) != 0U;
    bool converged_0;
    if(((&kernelContext_3)->params_0->skip_settled_0) != 0U)
    {
        converged_0 = warm_1;
    }
    else
    {
        converged_0 = false;
    }
    if(converged_0)
    {
        converged_0 = (last_flags_0 & 4U) != 0U;
    }
    else
    {
        converged_0 = false;
    }
    float target_0;
    uint i_0;
    if(converged_0)
    {
        uint _S21 = n0_1 + tid_1;
        target_0 = 0.0f;
        i_0 = _S21;
        for(;;)
        {
            if(i_0 < _S18)
            {
            }
            else
            {
                break;
            }
            uint _S22 = i_0 * 2U;
            uint _S23 = node_slot_0(i_0, 8U);
            if(any((float4(*((&kernelContext_3)->accel_0+_S22)) ) != (float4(*((&kernelContext_3)->scratch_0+_S23)) )))
            {
                converged_0 = true;
            }
            else
            {
                converged_0 = any((float4(*((&kernelContext_3)->accel_0+(_S22 + 1U))) ) != (float4(*((&kernelContext_3)->scratch_0+(_S23 + 1U))) ));
            }
            if(converged_0)
            {
                target_0 = 1.0f;
            }
            i_0 = i_0 + 256U;
        }
        float2 _S24 = blast_group_sum_0(tid_1, float2(target_0, 0.0f), &kernelContext_3);
        if((_S24.x) == 0.0f)
        {
            if(tid_1 == 0U)
            {
                *((&kernelContext_3)->report_0+k_1) = packed_uint4(uint4(0U, 6U, _S20.z, _S20.w)) ;
            }
            return;
        }
    }
    uint _S25 = n0_1 + tid_1;
    i_0 = _S25;
    target_0 = 0.0f;
    for(;;)
    {
        if(i_0 < _S18)
        {
        }
        else
        {
            break;
        }
        BlastNode_0 n_3 = (&kernelContext_3)->nodes_0[i_0];
        uint _S26 = i_0 * 2U;
        float4 _S27 = float4(*((&kernelContext_3)->accel_0+_S26)) ;
        float4 _S28 = float4(*((&kernelContext_3)->accel_0+(_S26 + 1U))) ;
        if((n_3.w_ang_0) > 0.0f)
        {
            ia_0 = 1.0f / n_3.w_ang_0;
        }
        else
        {
            ia_0 = 1.0f;
        }
        if((n_3.w_lin_0) > 0.0f)
        {
            il_0 = (&kernelContext_3)->params_0->recip_length_0 / n_3.w_lin_0;
        }
        else
        {
            il_0 = (&kernelContext_3)->params_0->recip_length_0;
        }
        float3 b_ang_0 = - _S27.xyz * float3(ia_0) ;
        float3 b_lin_0 = - _S28.xyz * float3(il_0) ;
        float target_1 = target_0 + norm2_0(b_ang_0, b_lin_0);
        uint _S29 = node_slot_0(i_0, 0U);
        *((&kernelContext_3)->scratch_0+_S29) = packed_float4(float4(b_ang_0, 0.0f)) ;
        *((&kernelContext_3)->scratch_0+(_S29 + 1U)) = packed_float4(float4(b_lin_0, 0.0f)) ;
        uint _S30 = node_slot_0(i_0, 8U);
        *((&kernelContext_3)->scratch_0+_S30) = packed_float4(_S27) ;
        *((&kernelContext_3)->scratch_0+(_S30 + 1U)) = packed_float4(_S28) ;
        v_1 = 2U;
        for(;;)
        {
            if(v_1 < 8U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_3)->scratch_0+node_slot_0(i_0, v_1)) = packed_float4(float4(0.0f) ) ;
            v_1 = v_1 + 1U;
        }
        i_0 = i_0 + 256U;
        target_0 = target_1;
    }
    if(!warm_1)
    {
        v_1 = b0_0 + tid_1;
        for(;;)
        {
            if(v_1 < _S19)
            {
            }
            else
            {
                break;
            }
            uint _S31 = v_1 * 2U;
            packed_float4 _S32 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_3)->x_0+_S31) = _S32;
            *((&kernelContext_3)->x_0+(_S31 + 1U)) = _S32;
            v_1 = v_1 + 256U;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if(warm_1)
    {
        i_0 = _S25;
        for(;;)
        {
            if(i_0 < _S18)
            {
            }
            else
            {
                break;
            }
            thread float3 ya_1;
            thread float3 yl_1;
            gather_0(i_0, true, &ya_1, &yl_1, &kernelContext_3);
            uint _S33 = node_slot_0(i_0, 0U);
            *((&kernelContext_3)->scratch_0+_S33) = packed_float4(float4((float4(*((&kernelContext_3)->scratch_0+_S33)) ).xyz - ya_1, 0.0f)) ;
            uint _S34 = _S33 + 1U;
            *((&kernelContext_3)->scratch_0+_S34) = packed_float4(float4((float4(*((&kernelContext_3)->scratch_0+_S34)) ).xyz - yl_1, 0.0f)) ;
            i_0 = i_0 + 256U;
        }
    }
    float2 _S35 = blast_group_sum_0(tid_1, float2(target_0, 0.0f), &kernelContext_3);
    float target2_0 = _S35.x * (&kernelContext_3)->params_0->tolerance_0 * (&kernelContext_3)->params_0->tolerance_0;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    ia_0 = 0.0f;
    v_1 = 0U;
    uint iterations_0 = 0U;
    for(;;)
    {
        if(v_1 < ((&kernelContext_3)->params_0->max_iterations_0))
        {
        }
        else
        {
            converged_0 = false;
            il_0 = ia_0;
            break;
        }
        j_1 = b0_0 + tid_1;
        il_0 = 0.0f;
        for(;;)
        {
            if(j_1 < _S19)
            {
            }
            else
            {
                break;
            }
            thread BlastBond_natural_0 _S36 = *((&kernelContext_3)->bonds_0+j_1);
            float3 _S37 = float3(0.0f) ;
            thread float3 ta_0 = _S37;
            thread float3 tl_0 = _S37;
            if(((&_S36)->health_0) > 0.0f)
            {
                bond_term_0(&_S36, 0U, &ta_0, &tl_0, &kernelContext_3);
            }
            float zt_0 = il_0 + norm2_0(ta_0, tl_0);
            uint t_1 = bond_slot_0((&kernelContext_3)->params_0->node_count_0, j_1);
            *((&kernelContext_3)->scratch_0+t_1) = packed_float4(float4(ta_0, 0.0f)) ;
            *((&kernelContext_3)->scratch_0+(t_1 + 1U)) = packed_float4(float4(tl_0, 0.0f)) ;
            j_1 = j_1 + 256U;
            il_0 = zt_0;
        }
        float2 _S38 = blast_group_sum_0(tid_1, float2(il_0, 0.0f), &kernelContext_3);
        float z_0 = _S38.x;
        if(z_0 <= target2_0)
        {
            converged_0 = true;
            il_0 = z_0;
            break;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if(v_1 != 0U)
        {
            converged_0 = ia_0 > 0.0f;
        }
        else
        {
            converged_0 = false;
        }
        if(converged_0)
        {
            target_0 = z_0 / ia_0;
        }
        else
        {
            target_0 = 0.0f;
        }
        i_0 = _S25;
        float qq_0 = 0.0f;
        for(;;)
        {
            if(i_0 < _S18)
            {
            }
            else
            {
                break;
            }
            thread float3 wa_0;
            thread float3 wl_0;
            gather_0(i_0, false, &wa_0, &wl_0, &kernelContext_3);
            uint _S39 = node_slot_0(i_0, 0U);
            uint _S40 = node_slot_0(i_0, 2U);
            uint _S41 = _S40 + 1U;
            float3 pl_0 = (float4(*((&kernelContext_3)->scratch_0+(_S39 + 1U))) ).xyz + (float4(*((&kernelContext_3)->scratch_0+_S41)) ).xyz * float3(target_0) ;
            uint _S42 = node_slot_0(i_0, 4U);
            float3 qa_0 = wa_0 + (float4(*((&kernelContext_3)->scratch_0+_S42)) ).xyz * float3(target_0) ;
            uint _S43 = _S42 + 1U;
            float3 ql_0 = wl_0 + (float4(*((&kernelContext_3)->scratch_0+_S43)) ).xyz * float3(target_0) ;
            *((&kernelContext_3)->scratch_0+_S40) = packed_float4(float4((float4(*((&kernelContext_3)->scratch_0+_S39)) ).xyz + (float4(*((&kernelContext_3)->scratch_0+_S40)) ).xyz * float3(target_0) , 0.0f)) ;
            *((&kernelContext_3)->scratch_0+_S41) = packed_float4(float4(pl_0, 0.0f)) ;
            *((&kernelContext_3)->scratch_0+_S42) = packed_float4(float4(qa_0, 0.0f)) ;
            *((&kernelContext_3)->scratch_0+_S43) = packed_float4(float4(ql_0, 0.0f)) ;
            float qq_1 = qq_0 + norm2_0(qa_0, ql_0);
            i_0 = i_0 + 256U;
            qq_0 = qq_1;
        }
        float2 _S44 = blast_group_sum_0(tid_1, float2(qq_0, 0.0f), &kernelContext_3);
        float q2_0 = _S44.x;
        bool _S45;
        if(!(q2_0 > 0.0f))
        {
            _S45 = true;
        }
        else
        {
            _S45 = !isfinite(q2_0);
        }
        if(_S45)
        {
            converged_0 = false;
            il_0 = z_0;
            break;
        }
        float _S46 = z_0 / q2_0;
        uint i_1 = _S25;
        for(;;)
        {
            if(i_1 < _S18)
            {
            }
            else
            {
                break;
            }
            uint _S47 = node_slot_0(i_1, 6U);
            uint _S48 = node_slot_0(i_1, 2U);
            float3 _S49 = float3(_S46) ;
            *((&kernelContext_3)->scratch_0+_S47) = packed_float4(float4((float4(*((&kernelContext_3)->scratch_0+_S47)) ).xyz + (float4(*((&kernelContext_3)->scratch_0+_S48)) ).xyz * _S49, 0.0f)) ;
            uint _S50 = _S47 + 1U;
            *((&kernelContext_3)->scratch_0+_S50) = packed_float4(float4((float4(*((&kernelContext_3)->scratch_0+_S50)) ).xyz + (float4(*((&kernelContext_3)->scratch_0+(_S48 + 1U))) ).xyz * _S49, 0.0f)) ;
            uint _S51 = node_slot_0(i_1, 0U);
            uint _S52 = node_slot_0(i_1, 4U);
            *((&kernelContext_3)->scratch_0+_S51) = packed_float4(float4((float4(*((&kernelContext_3)->scratch_0+_S51)) ).xyz - (float4(*((&kernelContext_3)->scratch_0+_S52)) ).xyz * _S49, 0.0f)) ;
            uint _S53 = _S51 + 1U;
            *((&kernelContext_3)->scratch_0+_S53) = packed_float4(float4((float4(*((&kernelContext_3)->scratch_0+_S53)) ).xyz - (float4(*((&kernelContext_3)->scratch_0+(_S52 + 1U))) ).xyz * _S49, 0.0f)) ;
            i_1 = i_1 + 256U;
        }
        uint _S54 = iterations_0 + 1U;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        uint _S55 = v_1 + 1U;
        ia_0 = z_0;
        v_1 = _S55;
        iterations_0 = _S54;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    j_1 = b0_0 + tid_1;
    for(;;)
    {
        if(j_1 < _S19)
        {
        }
        else
        {
            break;
        }
        thread BlastBond_natural_0 _S56 = *((&kernelContext_3)->bonds_0+j_1);
        if(!(((&_S56)->health_0) > 0.0f))
        {
            uint _S57 = j_1 * 2U;
            packed_float4 _S58 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_3)->x_0+_S57) = _S58;
            *((&kernelContext_3)->x_0+(_S57 + 1U)) = _S58;
            j_1 = j_1 + 256U;
            continue;
        }
        thread float3 ta_1;
        thread float3 tl_1;
        bond_term_0(&_S56, 6U, &ta_1, &tl_1, &kernelContext_3);
        uint _S59 = j_1 * 2U;
        *((&kernelContext_3)->x_0+_S59) = packed_float4(float4((float4(*((&kernelContext_3)->x_0+_S59)) ).xyz + ta_1, 0.0f)) ;
        uint _S60 = _S59 + 1U;
        *((&kernelContext_3)->x_0+_S60) = packed_float4(float4((float4(*((&kernelContext_3)->x_0+_S60)) ).xyz + tl_1, 0.0f)) ;
        j_1 = j_1 + 256U;
    }
    if(tid_1 == 0U)
    {
        packed_uint4 device* _S61 = (&kernelContext_3)->report_0+k_1;
        if(converged_0)
        {
            i_0 = 5U;
        }
        else
        {
            i_0 = 0U;
        }
        *_S61 = packed_uint4(uint4(iterations_0, i_0, (as_type<uint>((il_0))), (as_type<uint>((target2_0))))) ;
    }
    return;
}

