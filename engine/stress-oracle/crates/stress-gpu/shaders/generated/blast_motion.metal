#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint4 group_uniform4_0(uint4 value_0, uint thread_0)
{
    return value_0;
}

uint blast_frame_0(uint node_count_0, uint k_0)
{
    return node_count_0 + 10U * k_0;
}

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

struct BlastParams_0
{
    uint island_count_0;
    uint max_iterations_0;
    uint warm_0;
    uint skip_settled_0;
    float tolerance_0;
    float recip_length_0;
    uint node_count_1;
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

struct KernelContext_0
{
    packed_uint4 device* islands_0;
    uint device* lists_0;
    BlastBond_natural_0 device* bonds_0;
    BlastNode_natural_0 device* nodes_0;
    BlastParams_0 constant* params_0;
    packed_float4 device* motion_0;
    uint device* refs_0;
    array<float2, int(32)> threadgroup* g_blast_part_0;
    array<uint, int(1024)> threadgroup* g_word_0;
    uint threadgroup* g_rotations_0;
    array<float2, int(24)> threadgroup* g_frame_0;
    array<float2, int(1024)> threadgroup* g_pair_0;
};

void set_frame_0(uint k_1, uint w_0, uint value_1, KernelContext_0 thread* kernelContext_0)
{
    uint at_0 = blast_frame_0(kernelContext_0->params_0->node_count_1, k_1) + w_0 / 4U;
    thread uint4 v_0 = uint4(*(kernelContext_0->islands_0+at_0)) ;
    v_0[w_0 % 4U] = value_1;
    *(kernelContext_0->islands_0+at_0) = packed_uint4(v_0) ;
    return;
}

float2 blast_group_sum_0(uint tid_0, float2 v_1, KernelContext_0 thread* kernelContext_1)
{
    float2 w_1 = simd_sum(v_1);
    if((tid_0 % 32U) == 0U)
    {
        (*kernelContext_1->g_blast_part_0)[tid_0 / 32U] = w_1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 _S1 = (*kernelContext_1->g_blast_part_0)[int(0)];
    uint k_2 = 1U;
    float2 s_1 = _S1;
    for(;;)
    {
        if(k_2 < 32U)
        {
        }
        else
        {
            break;
        }
        float2 s_2 = s_1 + (*kernelContext_1->g_blast_part_0)[k_2];
        k_2 = k_2 + 1U;
        s_1 = s_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s_1;
}

struct Ex_0
{
    array<float, int(8)> t_0;
};

Ex_0 ex_zero_0()
{
    thread Ex_0 r_0;
    uint i_0 = 0U;
    for(;;)
    {
        if(i_0 < 8U)
        {
        }
        else
        {
            break;
        }
        (&r_0)->t_0[i_0] = 0.0f;
        i_0 = i_0 + 1U;
    }
    return r_0;
}

void store_ex_0(uint base_0, const Ex_0 thread* e_0, KernelContext_0 thread* kernelContext_2)
{
    *(kernelContext_2->motion_0+base_0) = packed_float4(float4(e_0->t_0[int(0)], e_0->t_0[int(1)], e_0->t_0[int(2)], e_0->t_0[int(3)])) ;
    *(kernelContext_2->motion_0+(base_0 + 1U)) = packed_float4(float4(e_0->t_0[int(4)], e_0->t_0[int(5)], e_0->t_0[int(6)], e_0->t_0[int(7)])) ;
    return;
}

struct Ex3_0
{
    Ex_0 x_0;
    Ex_0 y_0;
    Ex_0 z_0;
};

void set_position_0(uint n_0, const Ex3_0 thread* p_0, KernelContext_0 thread* kernelContext_3)
{
    uint _S2 = n_0 * 7U;
    store_ex_0(_S2, &p_0->x_0, kernelContext_3);
    store_ex_0(_S2 + 2U, &p_0->y_0, kernelContext_3);
    store_ex_0(_S2 + 4U, &p_0->z_0, kernelContext_3);
    return;
}

void set_level_0(uint n_1, uint l_0, KernelContext_0 thread* kernelContext_4)
{
    *(kernelContext_4->motion_0+(n_1 * 7U + 6U)) = packed_float4(float4((as_type<float>((l_0))), 0.0f, 0.0f, 0.0f)) ;
    return;
}

uint level_0(uint n_2, KernelContext_0 thread* kernelContext_5)
{
    return (as_type<uint>(((float4(*(kernelContext_5->motion_0+(n_2 * 7U + 6U))) ).x)));
}

float opaque_0(float x_1, KernelContext_0 thread* kernelContext_6)
{
    return (as_type<float>(((as_type<uint>((x_1))) ^ (kernelContext_6->params_0->opaque_zero_0))));
}

void two_sum_0(float a_0, float b_0, float thread* s_3, float thread* e_1, KernelContext_0 thread* kernelContext_7)
{
    float _S3 = opaque_0(a_0 + b_0, kernelContext_7);
    *s_3 = _S3;
    float _S4 = opaque_0(_S3 - a_0, kernelContext_7);
    float _S5 = opaque_0(_S3 - _S4, kernelContext_7);
    float _S6 = opaque_0(a_0 - _S5, kernelContext_7);
    float _S7 = opaque_0(b_0 - _S4, kernelContext_7);
    *e_1 = _S6 + _S7;
    return;
}

bool ex_representable_0(const Ex_0 thread* a_1)
{
    uint i_1 = 0U;
    for(;;)
    {
        if(i_1 < 8U)
        {
        }
        else
        {
            break;
        }
        float _S8 = a_1->t_0[i_1];
        if((a_1->t_0[i_1]) != 0.0f)
        {
            float m_0 = abs(_S8);
            bool _S9;
            if(!(m_0 <= 1.15292150460684698e+18f))
            {
                _S9 = true;
            }
            else
            {
                _S9 = m_0 < 7.88860905221011805e-31f;
            }
            if(_S9)
            {
                return false;
            }
        }
        i_1 = i_1 + 1U;
    }
    return true;
}

Ex_0 ex_difference_0(float a_2, float b_1, bool thread* exact_0, KernelContext_0 thread* kernelContext_8)
{
    bool _S10;
    if(isinf(a_2))
    {
        _S10 = true;
    }
    else
    {
        _S10 = isnan(a_2);
    }
    if(_S10)
    {
        _S10 = true;
    }
    else
    {
        _S10 = isinf(b_1);
    }
    if(_S10)
    {
        _S10 = true;
    }
    else
    {
        _S10 = isnan(b_1);
    }
    if(_S10)
    {
        *exact_0 = false;
        return ex_zero_0();
    }
    thread Ex_0 r_1 = ex_zero_0();
    two_sum_0(a_2, - b_1, &(&r_1)->t_0[int(0)], &(&r_1)->t_0[int(1)], kernelContext_8);
    thread Ex_0 _S11 = r_1;
    bool _S12 = ex_representable_0(&_S11);
    if(_S12)
    {
        _S10 = *exact_0;
    }
    else
    {
        _S10 = false;
    }
    *exact_0 = _S10;
    return r_1;
}

Ex3_0 offset_0(const BlastBond_natural_0 thread* b_2, bool thread* exact_1, KernelContext_0 thread* kernelContext_9)
{
    thread Ex3_0 r_2;
    float4 _S13 = float4(b_2->o0_0) ;
    float4 _S14 = float4(b_2->o1_0) ;
    Ex_0 _S15 = ex_difference_0(_S13.x, _S14.x, exact_1, kernelContext_9);
    (&r_2)->x_0 = _S15;
    Ex_0 _S16 = ex_difference_0(_S13.y, _S14.y, exact_1, kernelContext_9);
    (&r_2)->y_0 = _S16;
    Ex_0 _S17 = ex_difference_0(_S13.z, _S14.z, exact_1, kernelContext_9);
    (&r_2)->z_0 = _S17;
    return r_2;
}

Ex3_0 offset_1(const BlastBond_natural_0 thread* b_3, bool thread* exact_2, KernelContext_0 thread* kernelContext_10)
{
    thread Ex3_0 r_3;
    float4 _S18 = float4(b_3->o0_0) ;
    float4 _S19 = float4(b_3->o1_0) ;
    Ex_0 _S20 = ex_difference_0(_S18.x, _S19.x, exact_2, kernelContext_10);
    (&r_3)->x_0 = _S20;
    Ex_0 _S21 = ex_difference_0(_S18.y, _S19.y, exact_2, kernelContext_10);
    (&r_3)->y_0 = _S21;
    Ex_0 _S22 = ex_difference_0(_S18.z, _S19.z, exact_2, kernelContext_10);
    (&r_3)->z_0 = _S22;
    return r_3;
}

Ex_0 load_ex_0(uint base_1, KernelContext_0 thread* kernelContext_11)
{
    float4 _S23 = float4(*(kernelContext_11->motion_0+base_1)) ;
    float4 _S24 = float4(*(kernelContext_11->motion_0+(base_1 + 1U))) ;
    thread Ex_0 r_4;
    (&r_4)->t_0[int(0)] = _S23.x;
    (&r_4)->t_0[int(1)] = _S23.y;
    (&r_4)->t_0[int(2)] = _S23.z;
    (&r_4)->t_0[int(3)] = _S23.w;
    (&r_4)->t_0[int(4)] = _S24.x;
    (&r_4)->t_0[int(5)] = _S24.y;
    (&r_4)->t_0[int(6)] = _S24.z;
    (&r_4)->t_0[int(7)] = _S24.w;
    return r_4;
}

Ex3_0 position_1(uint n_3, KernelContext_0 thread* kernelContext_12)
{
    thread Ex3_0 r_5;
    uint _S25 = n_3 * 7U;
    Ex_0 _S26 = load_ex_0(_S25, kernelContext_12);
    (&r_5)->x_0 = _S26;
    Ex_0 _S27 = load_ex_0(_S25 + 2U, kernelContext_12);
    (&r_5)->y_0 = _S27;
    Ex_0 _S28 = load_ex_0(_S25 + 4U, kernelContext_12);
    (&r_5)->z_0 = _S28;
    return r_5;
}

bool ex_distill_0(array<float, int(16)> thread* v_2, uint count_0, Ex_0 thread* out_ex_0, KernelContext_0 thread* kernelContext_13)
{
    uint pass_0;
    uint i_2;
    *out_ex_0 = ex_zero_0();
    bool exact_3 = true;
    uint term_0 = 0U;
    for(;;)
    {
        if(term_0 < count_0)
        {
        }
        else
        {
            break;
        }
        pass_0 = term_0;
        for(;;)
        {
            uint _S29 = pass_0 + 1U;
            if(_S29 < count_0)
            {
            }
            else
            {
                break;
            }
            i_2 = count_0 - 1U;
            for(;;)
            {
                if(i_2 > term_0)
                {
                }
                else
                {
                    break;
                }
                uint _S30 = i_2 - 1U;
                thread float s_4;
                thread float e_2;
                two_sum_0((*v_2)[_S30], (*v_2)[i_2], &s_4, &e_2, kernelContext_13);
                (*v_2)[_S30] = s_4;
                (*v_2)[i_2] = e_2;
                i_2 = _S30;
            }
            pass_0 = _S29;
        }
        if(term_0 < 8U)
        {
            out_ex_0->t_0[term_0] = (*v_2)[term_0];
        }
        else
        {
            bool exact_4;
            if(((*v_2)[term_0]) != 0.0f)
            {
                exact_4 = false;
            }
            else
            {
                exact_4 = exact_3;
            }
            exact_3 = exact_4;
        }
        term_0 = term_0 + 1U;
    }
    pass_0 = 0U;
    for(;;)
    {
        uint _S31 = pass_0 + 1U;
        if(_S31 < 8U)
        {
        }
        else
        {
            break;
        }
        i_2 = 0U;
        for(;;)
        {
            uint _S32 = i_2 + 1U;
            if(_S32 < 8U)
            {
            }
            else
            {
                break;
            }
            if((out_ex_0->t_0[i_2]) == 0.0f)
            {
                out_ex_0->t_0[i_2] = out_ex_0->t_0[_S32];
                out_ex_0->t_0[_S32] = 0.0f;
            }
            i_2 = _S32;
        }
        pass_0 = _S31;
    }
    if(exact_3)
    {
        thread Ex_0 _S33 = *out_ex_0;
        bool _S34 = ex_representable_0(&_S33);
        exact_3 = _S34;
    }
    else
    {
        exact_3 = false;
    }
    return exact_3;
}

Ex_0 ex_add_0(const Ex_0 thread* a_3, const Ex_0 thread* b_4, bool thread* exact_5, KernelContext_0 thread* kernelContext_14)
{
    thread array<float, int(16)> v_3;
    uint i_3 = 0U;
    for(;;)
    {
        if(i_3 < 16U)
        {
        }
        else
        {
            break;
        }
        v_3[i_3] = 0.0f;
        i_3 = i_3 + 1U;
    }
    bool wide_0 = false;
    i_3 = 3U;
    for(;;)
    {
        if(i_3 < 8U)
        {
        }
        else
        {
            break;
        }
        bool _S35;
        if((a_3->t_0[i_3]) != 0.0f)
        {
            _S35 = true;
        }
        else
        {
            _S35 = (b_4->t_0[i_3]) != 0.0f;
        }
        if(_S35)
        {
            wide_0 = true;
        }
        i_3 = i_3 + 1U;
    }
    uint count_1;
    if(wide_0)
    {
        i_3 = 0U;
        for(;;)
        {
            if(i_3 < 8U)
            {
            }
            else
            {
                break;
            }
            uint _S36 = 2U * i_3;
            v_3[_S36] = a_3->t_0[i_3];
            v_3[_S36 + 1U] = b_4->t_0[i_3];
            i_3 = i_3 + 1U;
        }
        count_1 = 16U;
    }
    else
    {
        float _S37 = a_3->t_0[int(2)];
        if((a_3->t_0[int(2)]) != 0.0f)
        {
            wide_0 = true;
        }
        else
        {
            wide_0 = (b_4->t_0[int(2)]) != 0.0f;
        }
        if(wide_0)
        {
            v_3[int(0)] = a_3->t_0[int(0)];
            v_3[int(1)] = b_4->t_0[int(0)];
            v_3[int(2)] = a_3->t_0[int(1)];
            v_3[int(3)] = b_4->t_0[int(1)];
            v_3[int(4)] = _S37;
            v_3[int(5)] = b_4->t_0[int(2)];
            count_1 = 6U;
        }
        else
        {
            v_3[int(0)] = a_3->t_0[int(0)];
            v_3[int(1)] = b_4->t_0[int(0)];
            v_3[int(2)] = a_3->t_0[int(1)];
            v_3[int(3)] = b_4->t_0[int(1)];
            count_1 = 4U;
        }
    }
    thread Ex_0 r_6;
    bool _S38 = ex_distill_0(&v_3, count_1, &r_6, kernelContext_14);
    if(_S38)
    {
        wide_0 = *exact_5;
    }
    else
    {
        wide_0 = false;
    }
    *exact_5 = wide_0;
    return r_6;
}

Ex3_0 ex3_add_0(const Ex3_0 thread* a_4, const Ex3_0 thread* b_5, bool thread* exact_6, KernelContext_0 thread* kernelContext_15)
{
    thread Ex3_0 r_7;
    Ex_0 _S39 = ex_add_0(&a_4->x_0, &b_5->x_0, exact_6, kernelContext_15);
    (&r_7)->x_0 = _S39;
    Ex_0 _S40 = ex_add_0(&a_4->y_0, &b_5->y_0, exact_6, kernelContext_15);
    (&r_7)->y_0 = _S40;
    Ex_0 _S41 = ex_add_0(&a_4->z_0, &b_5->z_0, exact_6, kernelContext_15);
    (&r_7)->z_0 = _S41;
    return r_7;
}

Ex_0 ex_neg_0(const Ex_0 thread* a_5)
{
    thread Ex_0 _S42 = *a_5;
    uint i_4 = 0U;
    for(;;)
    {
        if(i_4 < 8U)
        {
        }
        else
        {
            break;
        }
        (&_S42)->t_0[i_4] = - (&_S42)->t_0[i_4];
        i_4 = i_4 + 1U;
    }
    return _S42;
}

Ex3_0 ex3_neg_0(const Ex3_0 thread* a_6)
{
    thread Ex3_0 r_8;
    Ex_0 _S43 = ex_neg_0(&a_6->x_0);
    (&r_8)->x_0 = _S43;
    Ex_0 _S44 = ex_neg_0(&a_6->y_0);
    (&r_8)->y_0 = _S44;
    Ex_0 _S45 = ex_neg_0(&a_6->z_0);
    (&r_8)->z_0 = _S45;
    return r_8;
}

Ex3_0 closure_0(const BlastBond_natural_0 thread* b_6, bool thread* exact_7, KernelContext_0 thread* kernelContext_16)
{
    Ex3_0 _S46 = position_1(b_6->n0_0, kernelContext_16);
    Ex3_0 _S47 = offset_1(b_6, exact_7, kernelContext_16);
    thread Ex3_0 _S48 = _S46;
    thread Ex3_0 _S49 = _S47;
    Ex3_0 _S50 = ex3_add_0(&_S48, &_S49, exact_7, kernelContext_16);
    Ex3_0 _S51 = position_1(b_6->n1_0, kernelContext_16);
    thread Ex3_0 _S52 = _S51;
    Ex3_0 _S53 = ex3_neg_0(&_S52);
    thread Ex3_0 _S54 = _S50;
    thread Ex3_0 _S55 = _S53;
    Ex3_0 _S56 = ex3_add_0(&_S54, &_S55, exact_7, kernelContext_16);
    return _S56;
}

bool ex_nonzero_0(const Ex_0 thread* a_7)
{
    return (a_7->t_0[int(0)]) != 0.0f;
}

bool ex3_nonzero_0(const Ex3_0 thread* a_8)
{
    bool _S57 = ex_nonzero_0(&a_8->x_0);
    bool _S58;
    if(_S57)
    {
        _S58 = true;
    }
    else
    {
        bool _S59 = ex_nonzero_0(&a_8->y_0);
        _S58 = _S59;
    }
    if(_S58)
    {
        _S58 = true;
    }
    else
    {
        bool _S60 = ex_nonzero_0(&a_8->z_0);
        _S58 = _S60;
    }
    return _S58;
}

Ex_0 ex_magnitude_0(const Ex_0 thread* a_9)
{
    Ex_0 _S61;
    if((a_9->t_0[int(0)]) < 0.0f)
    {
        Ex_0 _S62 = ex_neg_0(a_9);
        _S61 = _S62;
    }
    else
    {
        _S61 = *a_9;
    }
    return _S61;
}

bool ex_greater_0(const Ex_0 thread* a_10, const Ex_0 thread* b_7, KernelContext_0 thread* kernelContext_17)
{
    thread array<float, int(16)> v_4;
    uint i_5 = 0U;
    for(;;)
    {
        if(i_5 < 8U)
        {
        }
        else
        {
            break;
        }
        uint _S63 = 2U * i_5;
        v_4[_S63] = a_10->t_0[i_5];
        v_4[_S63 + 1U] = - b_7->t_0[i_5];
        i_5 = i_5 + 1U;
    }
    uint pass_1 = 0U;
    for(;;)
    {
        uint _S64 = pass_1 + 1U;
        if(_S64 < 16U)
        {
        }
        else
        {
            break;
        }
        i_5 = 15U;
        for(;;)
        {
            if(i_5 > 0U)
            {
            }
            else
            {
                break;
            }
            uint _S65 = i_5 - 1U;
            thread float s_5;
            thread float e_3;
            two_sum_0(v_4[_S65], v_4[i_5], &s_5, &e_3, kernelContext_17);
            v_4[_S65] = s_5;
            v_4[i_5] = e_3;
            i_5 = _S65;
        }
        pass_1 = _S64;
    }
    i_5 = 0U;
    for(;;)
    {
        if(i_5 < 16U)
        {
        }
        else
        {
            break;
        }
        if((v_4[i_5]) != 0.0f)
        {
            return (v_4[i_5]) > 0.0f;
        }
        i_5 = i_5 + 1U;
    }
    return false;
}

bool ex_greater_1(const Ex_0 thread* a_11, const Ex_0 thread* b_8, KernelContext_0 thread* kernelContext_18)
{
    thread array<float, int(16)> v_5;
    uint i_6 = 0U;
    for(;;)
    {
        if(i_6 < 8U)
        {
        }
        else
        {
            break;
        }
        uint _S66 = 2U * i_6;
        v_5[_S66] = a_11->t_0[i_6];
        v_5[_S66 + 1U] = - b_8->t_0[i_6];
        i_6 = i_6 + 1U;
    }
    uint pass_2 = 0U;
    for(;;)
    {
        uint _S67 = pass_2 + 1U;
        if(_S67 < 16U)
        {
        }
        else
        {
            break;
        }
        i_6 = 15U;
        for(;;)
        {
            if(i_6 > 0U)
            {
            }
            else
            {
                break;
            }
            uint _S68 = i_6 - 1U;
            thread float s_6;
            thread float e_4;
            two_sum_0(v_5[_S68], v_5[i_6], &s_6, &e_4, kernelContext_18);
            v_5[_S68] = s_6;
            v_5[i_6] = e_4;
            i_6 = _S68;
        }
        pass_2 = _S67;
    }
    i_6 = 0U;
    for(;;)
    {
        if(i_6 < 16U)
        {
        }
        else
        {
            break;
        }
        if((v_5[i_6]) != 0.0f)
        {
            return (v_5[i_6]) > 0.0f;
        }
        i_6 = i_6 + 1U;
    }
    return false;
}

Ex_0 ex_max_0(const Ex_0 thread* a_12, const Ex_0 thread* b_9, KernelContext_0 thread* kernelContext_19)
{
    bool _S69 = ex_greater_1(b_9, a_12, kernelContext_19);
    Ex_0 _S70;
    if(_S69)
    {
        _S70 = *b_9;
    }
    else
    {
        _S70 = *a_12;
    }
    return _S70;
}

Ex_0 ex_of_0(float f_0)
{
    thread Ex_0 r_9 = ex_zero_0();
    (&r_9)->t_0[int(0)] = f_0;
    return r_9;
}

Ex_0 ex_scaled_0(const Ex_0 thread* a_13, float power_0)
{
    thread Ex_0 _S71 = *a_13;
    uint i_7 = 0U;
    for(;;)
    {
        if(i_7 < 8U)
        {
        }
        else
        {
            break;
        }
        (&_S71)->t_0[i_7] = (&_S71)->t_0[i_7] * power_0;
        i_7 = i_7 + 1U;
    }
    return _S71;
}

bool significant_0(const BlastBond_natural_0 thread* b_10, const Ex3_0 thread* c_0, KernelContext_0 thread* kernelContext_20)
{
    Ex3_0 _S72 = position_1(b_10->n0_0, kernelContext_20);
    Ex3_0 _S73 = position_1(b_10->n1_0, kernelContext_20);
    float4 _S74 = float4(b_10->o0_0) ;
    float4 _S75 = float4(b_10->o1_0) ;
    float _S76 = max(max(max(abs(_S74.x), abs(_S74.y)), max(abs(_S74.z), abs(_S75.x))), max(abs(_S75.y), abs(_S75.z)));
    float _S77 = max(max(abs((&c_0->x_0)->t_0[int(0)]), abs((&c_0->y_0)->t_0[int(0)])), abs((&c_0->z_0)->t_0[int(0)]));
    float threshold_0 = max(_S76, max(max(max(abs(_S72.x_0.t_0[int(0)]), abs(_S72.y_0.t_0[int(0)])), max(abs(_S72.z_0.t_0[int(0)]), abs(_S73.x_0.t_0[int(0)]))), max(abs(_S73.y_0.t_0[int(0)]), abs(_S73.z_0.t_0[int(0)])))) * 0.00000762939453125f;
    if(_S77 > (threshold_0 * 1.00000095367431641f))
    {
        return true;
    }
    if(_S77 < (threshold_0 * 0.99999904632568359f))
    {
        return false;
    }
    thread Ex_0 _S78 = _S72.x_0;
    Ex_0 _S79 = ex_magnitude_0(&_S78);
    thread Ex_0 _S80 = _S72.y_0;
    Ex_0 _S81 = ex_magnitude_0(&_S80);
    thread Ex_0 _S82 = _S79;
    thread Ex_0 _S83 = _S81;
    Ex_0 _S84 = ex_max_0(&_S82, &_S83, kernelContext_20);
    thread Ex_0 _S85 = _S72.z_0;
    Ex_0 _S86 = ex_magnitude_0(&_S85);
    thread Ex_0 _S87 = _S84;
    thread Ex_0 _S88 = _S86;
    Ex_0 _S89 = ex_max_0(&_S87, &_S88, kernelContext_20);
    thread Ex_0 _S90 = _S73.x_0;
    Ex_0 _S91 = ex_magnitude_0(&_S90);
    thread Ex_0 _S92 = _S73.y_0;
    Ex_0 _S93 = ex_magnitude_0(&_S92);
    thread Ex_0 _S94 = _S91;
    thread Ex_0 _S95 = _S93;
    Ex_0 _S96 = ex_max_0(&_S94, &_S95, kernelContext_20);
    thread Ex_0 _S97 = _S73.z_0;
    Ex_0 _S98 = ex_magnitude_0(&_S97);
    thread Ex_0 _S99 = _S96;
    thread Ex_0 _S100 = _S98;
    Ex_0 _S101 = ex_max_0(&_S99, &_S100, kernelContext_20);
    thread Ex_0 _S102 = _S89;
    thread Ex_0 _S103 = _S101;
    Ex_0 _S104 = ex_max_0(&_S102, &_S103, kernelContext_20);
    Ex_0 _S105 = ex_of_0(_S76);
    thread Ex_0 _S106 = _S104;
    thread Ex_0 _S107 = _S105;
    Ex_0 _S108 = ex_max_0(&_S106, &_S107, kernelContext_20);
    Ex_0 _S109 = ex_magnitude_0(&c_0->x_0);
    Ex_0 _S110 = ex_magnitude_0(&c_0->y_0);
    thread Ex_0 _S111 = _S109;
    thread Ex_0 _S112 = _S110;
    Ex_0 _S113 = ex_max_0(&_S111, &_S112, kernelContext_20);
    Ex_0 _S114 = ex_magnitude_0(&c_0->z_0);
    thread Ex_0 _S115 = _S113;
    thread Ex_0 _S116 = _S114;
    Ex_0 _S117 = ex_max_0(&_S115, &_S116, kernelContext_20);
    thread Ex_0 _S118 = _S108;
    Ex_0 _S119 = ex_scaled_0(&_S118, 0.00000762939453125f);
    thread Ex_0 _S120 = _S117;
    thread Ex_0 _S121 = _S119;
    bool _S122 = ex_greater_0(&_S120, &_S121, kernelContext_20);
    return _S122;
}

uint group_uniform_0(uint value_2, uint thread_1)
{
    return value_2;
}

uint word_min_0(uint tid_1, uint v_6, KernelContext_0 thread* kernelContext_21)
{
    (*kernelContext_21->g_word_0)[tid_1] = v_6;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint k_3 = 512U;
    for(;;)
    {
        if(k_3 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_1 < k_3)
        {
            (*kernelContext_21->g_word_0)[tid_1] = min((*kernelContext_21->g_word_0)[tid_1], (*kernelContext_21->g_word_0)[tid_1 + k_3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_3 = k_3 >> 1U;
    }
    return group_uniform_0((*kernelContext_21->g_word_0)[int(0)], tid_1);
}

float2 ex_pair_0(const Ex_0 thread* a_14, KernelContext_0 thread* kernelContext_22)
{
    uint i_8 = 7U;
    float tail_0 = 0.0f;
    for(;;)
    {
        if(i_8 > 0U)
        {
        }
        else
        {
            break;
        }
        float tail_1 = tail_0 + a_14->t_0[i_8];
        i_8 = i_8 - 1U;
        tail_0 = tail_1;
    }
    thread float s_7;
    thread float e_5;
    two_sum_0(a_14->t_0[int(0)], tail_0, &s_7, &e_5, kernelContext_22);
    return float2(s_7, e_5);
}

struct P3_0
{
    float2 x_2;
    float2 y_1;
    float2 z_1;
};

P3_0 ex3_pair_0(const Ex3_0 thread* a_15, KernelContext_0 thread* kernelContext_23)
{
    thread P3_0 r_10;
    float2 _S123 = ex_pair_0(&a_15->x_0, kernelContext_23);
    (&r_10)->x_2 = _S123;
    float2 _S124 = ex_pair_0(&a_15->y_0, kernelContext_23);
    (&r_10)->y_1 = _S124;
    float2 _S125 = ex_pair_0(&a_15->z_0, kernelContext_23);
    (&r_10)->z_1 = _S125;
    return r_10;
}

float2 p_abs_0(float2 a_16)
{
    float2 _S126;
    if((a_16.x) < 0.0f)
    {
        _S126 = - a_16;
    }
    else
    {
        _S126 = a_16;
    }
    return _S126;
}

bool p_finite_0(float2 a_17)
{
    float _S127 = a_17.x;
    bool _S128;
    if(!isinf(_S127))
    {
        _S128 = !isnan(_S127);
    }
    else
    {
        _S128 = false;
    }
    if(_S128)
    {
        _S128 = !isinf(a_17.y);
    }
    else
    {
        _S128 = false;
    }
    if(_S128)
    {
        _S128 = !isnan(a_17.y);
    }
    else
    {
        _S128 = false;
    }
    return _S128;
}

void two_prod_0(float a_18, float b_11, float thread* p_1, float thread* e_6, KernelContext_0 thread* kernelContext_24)
{
    float _S129 = opaque_0(a_18 * b_11, kernelContext_24);
    *p_1 = _S129;
    *e_6 = fma(a_18, b_11, - _S129);
    return;
}

void fast_two_sum_0(float a_19, float b_12, float thread* s_8, float thread* e_7, KernelContext_0 thread* kernelContext_25)
{
    float _S130 = opaque_0(a_19 + b_12, kernelContext_25);
    *s_8 = _S130;
    float _S131 = opaque_0(_S130 - a_19, kernelContext_25);
    *e_7 = b_12 - _S131;
    return;
}

float2 p_mul_0(float2 a_20, float2 b_13, KernelContext_0 thread* kernelContext_26)
{
    float _S132 = a_20.x;
    float _S133 = b_13.x;
    thread float ch_0;
    thread float cl_0;
    two_prod_0(_S132, _S133, &ch_0, &cl_0, kernelContext_26);
    float _S134 = a_20.y;
    float _S135 = b_13.y;
    thread float zh_0;
    thread float zl_0;
    fast_two_sum_0(ch_0, cl_0 + fma(_S134, _S133, fma(_S132, _S135, _S134 * _S135)), &zh_0, &zl_0, kernelContext_26);
    return float2(zh_0, zl_0);
}

float2 p_div_0(float2 a_21, float2 b_14, KernelContext_0 thread* kernelContext_27)
{
    float _S136 = a_21.x;
    float _S137 = b_14.x;
    float th_0 = _S136 / _S137;
    float2 _S138 = p_mul_0(b_14, float2(th_0, 0.0f), kernelContext_27);
    thread float zh_1;
    thread float zl_1;
    fast_two_sum_0(th_0, (_S136 - _S138.x + (a_21.y - _S138.y)) / _S137, &zh_1, &zl_1, kernelContext_27);
    return float2(zh_1, zl_1);
}

P3_0 p3_scale_0(const P3_0 thread* a_22, float2 k_4, KernelContext_0 thread* kernelContext_28)
{
    thread P3_0 r_11;
    float2 _S139 = p_mul_0(a_22->x_2, k_4, kernelContext_28);
    (&r_11)->x_2 = _S139;
    float2 _S140 = p_mul_0(a_22->y_1, k_4, kernelContext_28);
    (&r_11)->y_1 = _S140;
    float2 _S141 = p_mul_0(a_22->z_1, k_4, kernelContext_28);
    (&r_11)->z_1 = _S141;
    return r_11;
}

float2 p_add_0(float2 a_23, float2 b_15, KernelContext_0 thread* kernelContext_29)
{
    thread float sh_0;
    thread float sl_0;
    two_sum_0(a_23.x, b_15.x, &sh_0, &sl_0, kernelContext_29);
    thread float th_1;
    thread float tl_0;
    two_sum_0(a_23.y, b_15.y, &th_1, &tl_0, kernelContext_29);
    thread float vh_0;
    thread float vl_0;
    fast_two_sum_0(sh_0, sl_0 + th_1, &vh_0, &vl_0, kernelContext_29);
    thread float zh_2;
    thread float zl_2;
    fast_two_sum_0(vh_0, tl_0 + vl_0, &zh_2, &zl_2, kernelContext_29);
    return float2(zh_2, zl_2);
}

float2 p_add_1(float2 a_24, float2 b_16, KernelContext_0 thread* kernelContext_30)
{
    thread float sh_1;
    thread float sl_1;
    two_sum_0(a_24.x, b_16.x, &sh_1, &sl_1, kernelContext_30);
    thread float th_2;
    thread float tl_1;
    two_sum_0(a_24.y, b_16.y, &th_2, &tl_1, kernelContext_30);
    thread float vh_1;
    thread float vl_1;
    fast_two_sum_0(sh_1, sl_1 + th_2, &vh_1, &vl_1, kernelContext_30);
    thread float zh_3;
    thread float zl_3;
    fast_two_sum_0(vh_1, tl_1 + vl_1, &zh_3, &zl_3, kernelContext_30);
    return float2(zh_3, zl_3);
}

float2 p3_dot_0(const P3_0 thread* a_25, const P3_0 thread* b_17, KernelContext_0 thread* kernelContext_31)
{
    float2 _S142 = p_mul_0(a_25->x_2, b_17->x_2, kernelContext_31);
    float2 _S143 = p_mul_0(a_25->y_1, b_17->y_1, kernelContext_31);
    float2 _S144 = p_add_0(_S142, _S143, kernelContext_31);
    float2 _S145 = p_mul_0(a_25->z_1, b_17->z_1, kernelContext_31);
    float2 _S146 = p_add_0(_S144, _S145, kernelContext_31);
    return _S146;
}

float2 p_sqrt_0(float2 a_26, KernelContext_0 thread* kernelContext_32)
{
    float _S147 = a_26.x;
    if(!(_S147 > 0.0f))
    {
        return float2(sqrt(_S147), 0.0f);
    }
    float s_9 = sqrt(_S147);
    thread float p_2;
    thread float e_8;
    two_prod_0(s_9, s_9, &p_2, &e_8, kernelContext_32);
    thread float zh_4;
    thread float zl_4;
    fast_two_sum_0(s_9, (_S147 - p_2 - e_8 + a_26.y) / (2.0f * s_9), &zh_4, &zl_4, kernelContext_32);
    return float2(zh_4, zl_4);
}

bool ex_product_equal_0(const Ex_0 thread* a_27, const Ex_0 thread* b_18, const Ex_0 thread* c_1, const Ex_0 thread* d_0, bool thread* exact_8, KernelContext_0 thread* kernelContext_33)
{
    thread array<float, int(64)> e_9;
    uint z_2 = 0U;
    for(;;)
    {
        if(z_2 < 64U)
        {
        }
        else
        {
            break;
        }
        e_9[z_2] = 0.0f;
        z_2 = z_2 + 1U;
    }
    uint count_2 = 0U;
    uint side_0 = 0U;
    for(;;)
    {
        if(side_0 < 2U)
        {
        }
        else
        {
            break;
        }
        uint i_9 = 0U;
        for(;;)
        {
            if(i_9 < 8U)
            {
            }
            else
            {
                break;
            }
            uint count_3 = count_2;
            uint j_0 = 0U;
            for(;;)
            {
                if(j_0 < 8U)
                {
                }
                else
                {
                    break;
                }
                bool _S148 = side_0 == 0U;
                float x_3;
                if(_S148)
                {
                    x_3 = a_27->t_0[i_9];
                }
                else
                {
                    x_3 = c_1->t_0[i_9];
                }
                float y_2;
                if(_S148)
                {
                    y_2 = b_18->t_0[j_0];
                }
                else
                {
                    y_2 = - d_0->t_0[j_0];
                }
                bool _S149;
                if(x_3 == 0.0f)
                {
                    _S149 = true;
                }
                else
                {
                    _S149 = y_2 == 0.0f;
                }
                uint count_4;
                if(_S149)
                {
                    count_4 = count_3;
                    uint _S150 = j_0 + 1U;
                    count_3 = count_4;
                    j_0 = _S150;
                    continue;
                }
                thread float p_3;
                thread float pe_0;
                two_prod_0(x_3, y_2, &p_3, &pe_0, kernelContext_33);
                count_4 = count_3;
                uint half_0 = 0U;
                for(;;)
                {
                    if(half_0 < 2U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float carry_1;
                    if(half_0 == 0U)
                    {
                        carry_1 = pe_0;
                    }
                    else
                    {
                        carry_1 = p_3;
                    }
                    if(carry_1 == 0.0f)
                    {
                        half_0 = half_0 + 1U;
                        continue;
                    }
                    float carry_2 = carry_1;
                    uint k_5 = 0U;
                    uint next_0 = 0U;
                    for(;;)
                    {
                        if(k_5 < count_4)
                        {
                        }
                        else
                        {
                            break;
                        }
                        thread float s_10;
                        thread float h_0;
                        two_sum_0(carry_2, e_9[k_5], &s_10, &h_0, kernelContext_33);
                        if(h_0 != 0.0f)
                        {
                            uint _S151 = next_0 + 1U;
                            e_9[next_0] = h_0;
                            next_0 = _S151;
                        }
                        float _S152 = s_10;
                        uint _S153 = k_5 + 1U;
                        carry_2 = _S152;
                        k_5 = _S153;
                    }
                    uint next_1;
                    if(carry_2 != 0.0f)
                    {
                        if(next_0 >= 64U)
                        {
                            *exact_8 = false;
                            return false;
                        }
                        uint _S154 = next_0 + 1U;
                        e_9[next_0] = carry_2;
                        next_1 = _S154;
                    }
                    else
                    {
                        next_1 = next_0;
                    }
                    count_4 = next_1;
                    half_0 = half_0 + 1U;
                }
                uint _S150 = j_0 + 1U;
                count_3 = count_4;
                j_0 = _S150;
            }
            uint _S155 = i_9 + 1U;
            count_2 = count_3;
            i_9 = _S155;
        }
        side_0 = side_0 + 1U;
    }
    return count_2 == 0U;
}

bool ex3_collinear_0(const Ex3_0 thread* a_28, const Ex3_0 thread* b_19, bool thread* exact_9, KernelContext_0 thread* kernelContext_34)
{
    bool _S156 = ex_product_equal_0(&a_28->y_0, &b_19->z_0, &a_28->z_0, &b_19->y_0, exact_9, kernelContext_34);
    bool _S157;
    if(_S156)
    {
        bool _S158 = ex_product_equal_0(&a_28->z_0, &b_19->x_0, &a_28->x_0, &b_19->z_0, exact_9, kernelContext_34);
        _S157 = _S158;
    }
    else
    {
        _S157 = false;
    }
    if(_S157)
    {
        bool _S159 = ex_product_equal_0(&a_28->x_0, &b_19->y_0, &a_28->y_0, &b_19->x_0, exact_9, kernelContext_34);
        _S157 = _S159;
    }
    else
    {
        _S157 = false;
    }
    return _S157;
}

float2 p_inverse_square_0(float x_4, KernelContext_0 thread* kernelContext_35)
{
    thread float p_4;
    thread float e_10;
    two_prod_0(x_4, x_4, &p_4, &e_10, kernelContext_35);
    float2 _S160 = p_div_0(float2(1.0f, 0.0f), float2(p_4, e_10), kernelContext_35);
    return _S160;
}

float2 pair_sum_0(uint tid_2, float2 v_7, KernelContext_0 thread* kernelContext_36)
{
    (*kernelContext_36->g_pair_0)[tid_2] = v_7;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint k_6 = 512U;
    for(;;)
    {
        if(k_6 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_2 < k_6)
        {
            float2 _S161 = p_add_0((*kernelContext_36->g_pair_0)[tid_2], (*kernelContext_36->g_pair_0)[tid_2 + k_6], kernelContext_36);
            (*kernelContext_36->g_pair_0)[tid_2] = _S161;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_6 = k_6 >> 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_36->g_pair_0)[int(0)];
}

float2 p_sub_0(float2 a_29, float2 b_20, KernelContext_0 thread* kernelContext_37)
{
    float2 _S162 = p_add_1(a_29, - b_20, kernelContext_37);
    return _S162;
}

P3_0 p3_sub_0(const P3_0 thread* a_30, const P3_0 thread* b_21, KernelContext_0 thread* kernelContext_38)
{
    thread P3_0 r_12;
    float2 _S163 = p_sub_0(a_30->x_2, b_21->x_2, kernelContext_38);
    (&r_12)->x_2 = _S163;
    float2 _S164 = p_sub_0(a_30->y_1, b_21->y_1, kernelContext_38);
    (&r_12)->y_1 = _S164;
    float2 _S165 = p_sub_0(a_30->z_1, b_21->z_1, kernelContext_38);
    (&r_12)->z_1 = _S165;
    return r_12;
}

P3_0 p3_cross_0(const P3_0 thread* a_31, const P3_0 thread* b_22, KernelContext_0 thread* kernelContext_39)
{
    thread P3_0 r_13;
    float2 _S166 = a_31->y_1;
    float2 _S167 = b_22->z_1;
    float2 _S168 = p_mul_0(a_31->y_1, b_22->z_1, kernelContext_39);
    float2 _S169 = a_31->z_1;
    float2 _S170 = b_22->y_1;
    float2 _S171 = p_mul_0(a_31->z_1, b_22->y_1, kernelContext_39);
    float2 _S172 = p_sub_0(_S168, _S171, kernelContext_39);
    (&r_13)->x_2 = _S172;
    float2 _S173 = b_22->x_2;
    float2 _S174 = p_mul_0(_S169, b_22->x_2, kernelContext_39);
    float2 _S175 = a_31->x_2;
    float2 _S176 = p_mul_0(a_31->x_2, _S167, kernelContext_39);
    float2 _S177 = p_sub_0(_S174, _S176, kernelContext_39);
    (&r_13)->y_1 = _S177;
    float2 _S178 = p_mul_0(_S175, _S170, kernelContext_39);
    float2 _S179 = p_mul_0(_S166, _S173, kernelContext_39);
    float2 _S180 = p_sub_0(_S178, _S179, kernelContext_39);
    (&r_13)->z_1 = _S180;
    return r_13;
}

float2 p3_at_0(const P3_0 thread* a_32, uint i_10)
{
    float2 _S181;
    if(i_10 == 0U)
    {
        _S181 = a_32->x_2;
    }
    else
    {
        if(i_10 == 1U)
        {
            _S181 = a_32->y_1;
        }
        else
        {
            _S181 = a_32->z_1;
        }
    }
    return _S181;
}

struct BlastNode_0
{
    float w_ang_0;
    float w_lin_0;
    uint ref_begin_0;
    uint ref_end_0;
    float4 position_0;
};

[[kernel]] void blast_motion(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], packed_uint4 device* islands_1 [[buffer(1)]], uint device* lists_1 [[buffer(5)]], BlastBond_natural_0 device* bonds_1 [[buffer(4)]], BlastNode_natural_0 device* nodes_1 [[buffer(2)]], BlastParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* motion_1 [[buffer(6)]], uint device* refs_1 [[buffer(3)]])
{
    bool second_0;
    float changed_0;
    float changed_1;
    uint best_0;
    uint other_0;
    uint r_14;
    bool _S182;
    bool _S183;
    uint row_0;
    uint col_0;
    float2 value_3;
    thread KernelContext_0 kernelContext_40;
    (&kernelContext_40)->islands_0 = islands_1;
    (&kernelContext_40)->lists_0 = lists_1;
    (&kernelContext_40)->bonds_0 = bonds_1;
    (&kernelContext_40)->nodes_0 = nodes_1;
    (&kernelContext_40)->params_0 = params_1;
    (&kernelContext_40)->motion_0 = motion_1;
    (&kernelContext_40)->refs_0 = refs_1;
    threadgroup array<float2, int(32)> g_blast_part_1;
    (&kernelContext_40)->g_blast_part_0 = &g_blast_part_1;
    threadgroup array<uint, int(1024)> g_word_1;
    (&kernelContext_40)->g_word_0 = &g_word_1;
    threadgroup uint g_rotations_1;
    (&kernelContext_40)->g_rotations_0 = &g_rotations_1;
    threadgroup array<float2, int(24)> g_frame_1;
    (&kernelContext_40)->g_frame_0 = &g_frame_1;
    threadgroup array<float2, int(1024)> g_pair_1;
    (&kernelContext_40)->g_pair_0 = &g_pair_1;
    uint tid_3 = thread_2.x;
    uint k_7 = group_0.x;
    uint4 range_0 = group_uniform4_0(uint4(*(islands_1+k_7)) , tid_3);
    uint n0_1 = range_0.x;
    uint _S184 = range_0.y;
    uint _S185 = range_0.w;
    uint _S186 = range_0.z + tid_3;
    float anchor_0 = 0.0f;
    uint q_0 = _S186;
    for(;;)
    {
        if(q_0 < _S185)
        {
        }
        else
        {
            break;
        }
        BlastBond_natural_0 b_23 = (&kernelContext_40)->bonds_0[(&kernelContext_40)->lists_0[q_0]];
        if((b_23.health_0) > 0.0f)
        {
            if(!((((&kernelContext_40)->nodes_0+b_23.n0_0)->w_lin_0) > 0.0f))
            {
                second_0 = true;
            }
            else
            {
                second_0 = !((((&kernelContext_40)->nodes_0+b_23.n1_0)->w_lin_0) > 0.0f);
            }
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
            anchor_0 = 1.0f;
        }
        q_0 = q_0 + 1024U;
    }
    bool _S187 = tid_3 == 0U;
    uint p_5;
    if(_S187)
    {
        p_5 = 0U;
        for(;;)
        {
            if(p_5 < 40U)
            {
            }
            else
            {
                break;
            }
            set_frame_0(k_7, p_5, 0U, &kernelContext_40);
            p_5 = p_5 + 1U;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    float2 _S188 = blast_group_sum_0(tid_3, float2(anchor_0, 0.0f), &kernelContext_40);
    if((_S188.x) > 0.0f)
    {
        if(_S187)
        {
            set_frame_0(k_7, 0U, 1U, &kernelContext_40);
        }
        return;
    }
    thread bool exact_10 = true;
    uint _S189 = n0_1 + tid_3;
    p_5 = _S189;
    for(;;)
    {
        if(p_5 < _S184)
        {
        }
        else
        {
            break;
        }
        uint i_11 = (&kernelContext_40)->lists_0[p_5];
        thread Ex3_0 zero_0;
        Ex_0 _S190 = ex_zero_0();
        (&zero_0)->x_0 = _S190;
        (&zero_0)->y_0 = _S190;
        (&zero_0)->z_0 = _S190;
        thread Ex3_0 _S191 = zero_0;
        set_position_0(i_11, &_S191, &kernelContext_40);
        if(p_5 == n0_1)
        {
            q_0 = 0U;
        }
        else
        {
            q_0 = 4294967295U;
        }
        set_level_0(i_11, q_0, &kernelContext_40);
        p_5 = p_5 + 1024U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint l_1 = 0U;
    for(;;)
    {
        changed_0 = 0.0f;
        p_5 = _S189;
        for(;;)
        {
            if(p_5 < _S184)
            {
            }
            else
            {
                break;
            }
            uint i_12 = (&kernelContext_40)->lists_0[p_5];
            uint _S192 = level_0(i_12, &kernelContext_40);
            if(_S192 != 4294967295U)
            {
                p_5 = p_5 + 1024U;
                continue;
            }
            BlastNode_natural_0 device* _S193 = (&kernelContext_40)->nodes_0+i_12;
            BlastNode_natural_0 n_4 = *_S193;
            best_0 = 4294967295U;
            second_0 = false;
            r_14 = (*_S193).ref_begin_0;
            for(;;)
            {
                if(r_14 < (n_4.ref_end_0))
                {
                }
                else
                {
                    break;
                }
                uint ref_0 = (&kernelContext_40)->refs_0[r_14];
                uint j_1 = ref_0 & 2147483647U;
                BlastBond_natural_0 b_24 = (&kernelContext_40)->bonds_0[j_1];
                bool _S194 = (ref_0 & 2147483648U) != 0U;
                if(_S194)
                {
                    other_0 = b_24.n0_0;
                }
                else
                {
                    other_0 = b_24.n1_0;
                }
                if((b_24.health_0) > 0.0f)
                {
                    _S182 = (((&kernelContext_40)->nodes_0+other_0)->w_lin_0) > 0.0f;
                }
                else
                {
                    _S182 = false;
                }
                if(_S182)
                {
                    uint _S195 = level_0(other_0, &kernelContext_40);
                    _S183 = _S195 == l_1;
                }
                else
                {
                    _S183 = false;
                }
                bool _S196;
                if(_S183)
                {
                    _S196 = j_1 < best_0;
                }
                else
                {
                    _S196 = false;
                }
                if(_S196)
                {
                    best_0 = j_1;
                    second_0 = _S194;
                }
                r_14 = r_14 + 1U;
            }
            if(best_0 != 4294967295U)
            {
                thread BlastBond_natural_0 _S197 = (&kernelContext_40)->bonds_0[best_0];
                Ex3_0 _S198 = offset_0(&_S197, &exact_10, &kernelContext_40);
                Ex3_0 at_1;
                if(second_0)
                {
                    Ex3_0 _S199 = position_1((&_S197)->n0_0, &kernelContext_40);
                    thread Ex3_0 _S200 = _S199;
                    thread Ex3_0 _S201 = _S198;
                    Ex3_0 _S202 = ex3_add_0(&_S200, &_S201, &exact_10, &kernelContext_40);
                    at_1 = _S202;
                }
                else
                {
                    Ex3_0 _S203 = position_1((&_S197)->n1_0, &kernelContext_40);
                    thread Ex3_0 _S204 = _S198;
                    Ex3_0 _S205 = ex3_neg_0(&_S204);
                    thread Ex3_0 _S206 = _S203;
                    thread Ex3_0 _S207 = _S205;
                    Ex3_0 _S208 = ex3_add_0(&_S206, &_S207, &exact_10, &kernelContext_40);
                    at_1 = _S208;
                }
                thread Ex3_0 _S209 = at_1;
                set_position_0(i_12, &_S209, &kernelContext_40);
                set_level_0(i_12, l_1 + 1U, &kernelContext_40);
                changed_1 = 1.0f;
            }
            else
            {
                changed_1 = changed_0;
            }
            changed_0 = changed_1;
            p_5 = p_5 + 1024U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float2 _S210 = blast_group_sum_0(tid_3, float2(changed_0, 0.0f), &kernelContext_40);
        if((_S210.x) == 0.0f)
        {
            break;
        }
        l_1 = l_1 + 1U;
    }
    changed_1 = 0.0f;
    p_5 = _S189;
    for(;;)
    {
        if(p_5 < _S184)
        {
        }
        else
        {
            break;
        }
        uint _S211 = level_0((&kernelContext_40)->lists_0[p_5], &kernelContext_40);
        if(_S211 == 4294967295U)
        {
            changed_1 = 1.0f;
        }
        p_5 = p_5 + 1024U;
    }
    best_0 = 4294967295U;
    q_0 = _S186;
    for(;;)
    {
        if(q_0 < _S185)
        {
        }
        else
        {
            break;
        }
        uint j_2 = (&kernelContext_40)->lists_0[q_0];
        thread BlastBond_natural_0 _S212 = (&kernelContext_40)->bonds_0[j_2];
        if(!(((&_S212)->health_0) > 0.0f))
        {
            q_0 = q_0 + 1024U;
            continue;
        }
        Ex3_0 _S213 = closure_0(&_S212, &exact_10, &kernelContext_40);
        thread Ex3_0 _S214 = _S213;
        bool _S215 = ex3_nonzero_0(&_S214);
        if(_S215)
        {
            thread Ex3_0 _S216 = _S213;
            bool _S217 = significant_0(&_S212, &_S216, &kernelContext_40);
            second_0 = _S217;
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
            other_0 = min(best_0, j_2);
        }
        else
        {
            other_0 = best_0;
        }
        best_0 = other_0;
        q_0 = q_0 + 1024U;
    }
    uint _S218 = word_min_0(tid_3, best_0, &kernelContext_40);
    float2 value_4;
    if(_S187)
    {
        if(_S218 == 4294967295U)
        {
            q_0 = 3U;
        }
        else
        {
            q_0 = 1U;
        }
        *(&kernelContext_40)->g_rotations_0 = q_0;
        if(_S218 != 4294967295U)
        {
            thread bool e2_0 = true;
            thread BlastBond_natural_0 _S219 = (&kernelContext_40)->bonds_0[_S218];
            Ex3_0 _S220 = closure_0(&_S219, &e2_0, &kernelContext_40);
            thread Ex3_0 _S221 = _S220;
            P3_0 _S222 = ex3_pair_0(&_S221, &kernelContext_40);
            float2 maximum_0 = p_abs_0(_S222.x_2);
            if((abs(_S222.y_1.x)) > (maximum_0.x))
            {
                value_4 = p_abs_0(_S222.y_1);
            }
            else
            {
                value_4 = maximum_0;
            }
            if((abs(_S222.z_1.x)) > (value_4.x))
            {
                value_4 = p_abs_0(_S222.z_1);
            }
            if(!((value_4.x) > 0.0f))
            {
                second_0 = true;
            }
            else
            {
                second_0 = !p_finite_0(value_4);
            }
            if(second_0)
            {
                set_frame_0(k_7, 2U, 4U, &kernelContext_40);
            }
            float2 _S223 = float2(1.0f, 0.0f);
            float2 _S224 = p_div_0(_S223, value_4, &kernelContext_40);
            thread P3_0 _S225 = _S222;
            P3_0 _S226 = p3_scale_0(&_S225, _S224, &kernelContext_40);
            thread P3_0 _S227 = _S226;
            thread P3_0 _S228 = _S226;
            float2 _S229 = p3_dot_0(&_S227, &_S228, &kernelContext_40);
            float2 _S230 = p_sqrt_0(_S229, &kernelContext_40);
            float2 _S231 = p_div_0(_S223, _S230, &kernelContext_40);
            thread P3_0 _S232 = _S226;
            P3_0 _S233 = p3_scale_0(&_S232, _S231, &kernelContext_40);
            (*(&kernelContext_40)->g_frame_0)[int(21)] = _S233.x_2;
            (*(&kernelContext_40)->g_frame_0)[int(22)] = _S233.y_1;
            (*(&kernelContext_40)->g_frame_0)[int(23)] = _S233.z_1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float locked_0;
    if(_S218 != 4294967295U)
    {
        thread bool e2_1 = true;
        thread BlastBond_natural_0 _S234 = (&kernelContext_40)->bonds_0[_S218];
        Ex3_0 _S235 = closure_0(&_S234, &e2_1, &kernelContext_40);
        locked_0 = 0.0f;
        q_0 = _S186;
        for(;;)
        {
            if(q_0 < _S185)
            {
            }
            else
            {
                break;
            }
            thread BlastBond_natural_0 _S236 = (&kernelContext_40)->bonds_0[(&kernelContext_40)->lists_0[q_0]];
            if(!(((&_S236)->health_0) > 0.0f))
            {
                q_0 = q_0 + 1024U;
                continue;
            }
            Ex3_0 _S237 = closure_0(&_S236, &exact_10, &kernelContext_40);
            thread Ex3_0 _S238 = _S237;
            bool _S239 = ex3_nonzero_0(&_S238);
            if(!_S239)
            {
                second_0 = true;
            }
            else
            {
                thread Ex3_0 _S240 = _S237;
                bool _S241 = significant_0(&_S236, &_S240, &kernelContext_40);
                second_0 = !_S241;
            }
            if(second_0)
            {
                q_0 = q_0 + 1024U;
                continue;
            }
            thread Ex3_0 _S242 = _S235;
            thread Ex3_0 _S243 = _S237;
            bool _S244 = ex3_collinear_0(&_S242, &_S243, &exact_10, &kernelContext_40);
            float locked_1;
            if(!_S244)
            {
                locked_1 = 1.0f;
            }
            else
            {
                locked_1 = locked_0;
            }
            locked_0 = locked_1;
            q_0 = q_0 + 1024U;
        }
    }
    else
    {
        locked_0 = 0.0f;
    }
    if(exact_10)
    {
        anchor_0 = 0.0f;
    }
    else
    {
        anchor_0 = 1.0f;
    }
    float2 _S245 = blast_group_sum_0(tid_3, float2(locked_0, anchor_0), &kernelContext_40);
    float2 _S246 = blast_group_sum_0(tid_3, float2(changed_1, 0.0f), &kernelContext_40);
    bool unconnected_0 = (_S246.x) > 0.0f;
    if((_S245.x) > 0.0f)
    {
        q_0 = 0U;
    }
    else
    {
        q_0 = *(&kernelContext_40)->g_rotations_0;
    }
    uint rotations_0 = group_uniform_0(q_0, tid_3);
    thread P3_0 axis_0;
    (&axis_0)->x_2 = (*(&kernelContext_40)->g_frame_0)[int(21)];
    (&axis_0)->y_1 = (*(&kernelContext_40)->g_frame_0)[int(22)];
    (&axis_0)->z_1 = (*(&kernelContext_40)->g_frame_0)[int(23)];
    thread array<float2, int(4)> sums_0;
    other_0 = 0U;
    for(;;)
    {
        if(other_0 < 4U)
        {
        }
        else
        {
            break;
        }
        sums_0[other_0] = float2(0.0f) ;
        other_0 = other_0 + 1U;
    }
    p_5 = _S189;
    for(;;)
    {
        if(p_5 < _S184)
        {
        }
        else
        {
            break;
        }
        uint i_13 = (&kernelContext_40)->lists_0[p_5];
        float2 _S247 = p_inverse_square_0(((&kernelContext_40)->nodes_0+i_13)->w_lin_0, &kernelContext_40);
        Ex3_0 _S248 = position_1(i_13, &kernelContext_40);
        thread Ex3_0 _S249 = _S248;
        P3_0 _S250 = ex3_pair_0(&_S249, &kernelContext_40);
        float2 _S251 = p_add_0(sums_0[int(0)], _S247, &kernelContext_40);
        sums_0[int(0)] = _S251;
        float2 _S252 = p_mul_0(_S250.x_2, _S247, &kernelContext_40);
        float2 _S253 = p_add_0(sums_0[int(1)], _S252, &kernelContext_40);
        sums_0[int(1)] = _S253;
        float2 _S254 = p_mul_0(_S250.y_1, _S247, &kernelContext_40);
        float2 _S255 = p_add_0(sums_0[int(2)], _S254, &kernelContext_40);
        sums_0[int(2)] = _S255;
        float2 _S256 = p_mul_0(_S250.z_1, _S247, &kernelContext_40);
        float2 _S257 = p_add_0(sums_0[int(3)], _S256, &kernelContext_40);
        sums_0[int(3)] = _S257;
        p_5 = p_5 + 1024U;
    }
    other_0 = 0U;
    for(;;)
    {
        if(other_0 < 4U)
        {
        }
        else
        {
            break;
        }
        float2 _S258 = pair_sum_0(tid_3, sums_0[other_0], &kernelContext_40);
        if(_S187)
        {
            (*(&kernelContext_40)->g_frame_0)[other_0] = _S258;
        }
        other_0 = other_0 + 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    thread P3_0 centre_0;
    float2 _S259 = p_div_0((*(&kernelContext_40)->g_frame_0)[int(1)], (*(&kernelContext_40)->g_frame_0)[int(0)], &kernelContext_40);
    (&centre_0)->x_2 = _S259;
    float2 _S260 = p_div_0((*(&kernelContext_40)->g_frame_0)[int(2)], (*(&kernelContext_40)->g_frame_0)[int(0)], &kernelContext_40);
    (&centre_0)->y_1 = _S260;
    float2 _S261 = p_div_0((*(&kernelContext_40)->g_frame_0)[int(3)], (*(&kernelContext_40)->g_frame_0)[int(0)], &kernelContext_40);
    (&centre_0)->z_1 = _S261;
    if(((*(&kernelContext_40)->g_frame_0)[int(0)].x) > 0.0f)
    {
        second_0 = p_finite_0((*(&kernelContext_40)->g_frame_0)[int(0)]);
    }
    else
    {
        second_0 = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S262 = rotations_0 + 3U;
    bool _S263 = rotations_0 == 1U;
    if(_S263)
    {
        thread P3_0 _S264 = axis_0;
        thread P3_0 _S265 = axis_0;
        float2 _S266 = p3_dot_0(&_S264, &_S265, &kernelContext_40);
        value_4 = _S266;
    }
    else
    {
        value_4 = float2(1.0f, 0.0f);
    }
    thread array<float2, int(21)> gram_0;
    other_0 = 0U;
    for(;;)
    {
        if(other_0 < 21U)
        {
        }
        else
        {
            break;
        }
        gram_0[other_0] = float2(0.0f) ;
        other_0 = other_0 + 1U;
    }
    p_5 = _S189;
    for(;;)
    {
        if(p_5 < _S184)
        {
        }
        else
        {
            break;
        }
        uint i_14 = (&kernelContext_40)->lists_0[p_5];
        BlastNode_natural_0 device* _S267 = (&kernelContext_40)->nodes_0+i_14;
        float4 _S268 = float4((*_S267).position_0) ;
        thread BlastNode_0 n_5;
        (&n_5)->w_ang_0 = (*_S267).w_ang_0;
        (&n_5)->w_lin_0 = (*_S267).w_lin_0;
        (&n_5)->ref_begin_0 = (*_S267).ref_begin_0;
        (&n_5)->ref_end_0 = (*_S267).ref_end_0;
        (&n_5)->position_0 = _S268;
        Ex3_0 _S269 = position_1(i_14, &kernelContext_40);
        thread Ex3_0 _S270 = _S269;
        P3_0 _S271 = ex3_pair_0(&_S270, &kernelContext_40);
        thread P3_0 _S272 = _S271;
        thread P3_0 _S273 = centre_0;
        P3_0 _S274 = p3_sub_0(&_S272, &_S273, &kernelContext_40);
        float2 _S275 = p_inverse_square_0((&n_5)->w_lin_0, &kernelContext_40);
        float2 _S276 = p_inverse_square_0((&n_5)->w_ang_0, &kernelContext_40);
        (&n_5)->position_0 = float4(_S274.x_2.x, _S274.y_1.x, _S274.z_1.x, 0.0f);
        BlastNode_natural_0 device* _S277 = (&kernelContext_40)->nodes_0+i_14;
        _S277->w_ang_0 = n_5.w_ang_0;
        _S277->w_lin_0 = n_5.w_lin_0;
        _S277->ref_begin_0 = n_5.ref_begin_0;
        _S277->ref_end_0 = n_5.ref_end_0;
        _S277->position_0 = packed_float4(n_5.position_0) ;
        thread array<P3_0, int(3)> moment_0;
        r_14 = 0U;
        for(;;)
        {
            if(r_14 < 3U)
            {
            }
            else
            {
                break;
            }
            if(_S263)
            {
                thread P3_0 _S278 = _S274;
                thread P3_0 _S279 = axis_0;
                P3_0 _S280 = p3_cross_0(&_S278, &_S279, &kernelContext_40);
                moment_0[r_14] = _S280;
            }
            else
            {
                thread P3_0 e_11;
                if(r_14 == 0U)
                {
                    anchor_0 = 1.0f;
                }
                else
                {
                    anchor_0 = 0.0f;
                }
                (&e_11)->x_2 = float2(anchor_0, 0.0f);
                if(r_14 == 1U)
                {
                    changed_0 = 1.0f;
                }
                else
                {
                    changed_0 = 0.0f;
                }
                (&e_11)->y_1 = float2(changed_0, 0.0f);
                if(r_14 == 2U)
                {
                    changed_1 = 1.0f;
                }
                else
                {
                    changed_1 = 0.0f;
                }
                (&e_11)->z_1 = float2(changed_1, 0.0f);
                thread P3_0 _S281 = _S274;
                thread P3_0 _S282 = e_11;
                P3_0 _S283 = p3_cross_0(&_S281, &_S282, &kernelContext_40);
                moment_0[r_14] = _S283;
            }
            r_14 = r_14 + 1U;
        }
        row_0 = 0U;
        for(;;)
        {
            if(row_0 < _S262)
            {
            }
            else
            {
                break;
            }
            col_0 = 0U;
            for(;;)
            {
                if(col_0 <= row_0)
                {
                }
                else
                {
                    break;
                }
                float2 _S284 = float2(0.0f) ;
                if(row_0 < rotations_0)
                {
                    if(row_0 == col_0)
                    {
                        float2 _S285 = p_mul_0(_S276, value_4, &kernelContext_40);
                        value_3 = _S285;
                    }
                    else
                    {
                        value_3 = _S284;
                    }
                    thread P3_0 _S286 = moment_0[row_0];
                    thread P3_0 _S287 = moment_0[col_0];
                    float2 _S288 = p3_dot_0(&_S286, &_S287, &kernelContext_40);
                    float2 _S289 = p_mul_0(_S275, _S288, &kernelContext_40);
                    float2 _S290 = p_add_0(value_3, _S289, &kernelContext_40);
                    value_3 = _S290;
                }
                else
                {
                    if(col_0 < rotations_0)
                    {
                        uint _S291 = row_0 - rotations_0;
                        thread P3_0 _S292 = moment_0[col_0];
                        float2 _S293 = p3_at_0(&_S292, _S291);
                        float2 _S294 = p_mul_0(_S275, _S293, &kernelContext_40);
                        value_3 = _S294;
                    }
                    else
                    {
                        if(row_0 == col_0)
                        {
                            value_3 = _S275;
                        }
                        else
                        {
                            value_3 = _S284;
                        }
                    }
                }
                float2 _S295 = p_add_0(gram_0[row_0 * (row_0 + 1U) / 2U + col_0], value_3, &kernelContext_40);
                gram_0[row_0 * (row_0 + 1U) / 2U + col_0] = _S295;
                col_0 = col_0 + 1U;
            }
            row_0 = row_0 + 1U;
        }
        p_5 = p_5 + 1024U;
    }
    other_0 = 0U;
    for(;;)
    {
        if(other_0 < 21U)
        {
        }
        else
        {
            break;
        }
        float2 _S296 = pair_sum_0(tid_3, gram_0[other_0], &kernelContext_40);
        if(_S187)
        {
            gram_0[other_0] = _S296;
        }
        other_0 = other_0 + 1U;
    }
    if(tid_3 != 0U)
    {
        return;
    }
    if(unconnected_0)
    {
        q_0 = 2U;
    }
    else
    {
        q_0 = 0U;
    }
    if((_S245.y) > 0.0f)
    {
        p_5 = 16U;
    }
    else
    {
        p_5 = 0U;
    }
    uint error_0 = q_0 | p_5;
    if(!second_0)
    {
        r_14 = error_0 | 4U;
    }
    else
    {
        r_14 = error_0;
    }
    thread array<float2, int(6)> scale_0;
    thread array<float2, int(21)> factor_0;
    other_0 = 0U;
    for(;;)
    {
        if(other_0 < 6U)
        {
        }
        else
        {
            break;
        }
        scale_0[other_0] = float2(0.0f) ;
        other_0 = other_0 + 1U;
    }
    other_0 = 0U;
    for(;;)
    {
        if(other_0 < 21U)
        {
        }
        else
        {
            break;
        }
        factor_0[other_0] = float2(0.0f) ;
        other_0 = other_0 + 1U;
    }
    uint i_15 = 0U;
    for(;;)
    {
        if(i_15 < _S262)
        {
            second_0 = r_14 == 0U;
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
        }
        else
        {
            break;
        }
        uint _S297 = i_15 + 1U;
        float2 diagonal_0 = gram_0[i_15 * _S297 / 2U + i_15];
        if(!((gram_0[i_15 * _S297 / 2U + i_15].x) > 0.0f))
        {
            _S182 = true;
        }
        else
        {
            _S182 = !p_finite_0(diagonal_0);
        }
        if(_S182)
        {
            r_14 = r_14 | 4U;
        }
        else
        {
            float2 _S298 = float2(1.0f, 0.0f);
            float2 _S299 = p_sqrt_0(diagonal_0, &kernelContext_40);
            float2 _S300 = p_div_0(_S298, _S299, &kernelContext_40);
            scale_0[i_15] = _S300;
        }
        i_15 = _S297;
    }
    row_0 = 0U;
    for(;;)
    {
        if(row_0 < _S262)
        {
            second_0 = r_14 == 0U;
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
        }
        else
        {
            break;
        }
        col_0 = 0U;
        uint error_1 = r_14;
        for(;;)
        {
            if(col_0 <= row_0)
            {
                _S182 = error_1 == 0U;
            }
            else
            {
                _S182 = false;
            }
            if(_S182)
            {
            }
            else
            {
                break;
            }
            uint _S301 = row_0 * (row_0 + 1U);
            float2 _S302 = p_mul_0(gram_0[_S301 / 2U + col_0], scale_0[row_0], &kernelContext_40);
            float2 _S303 = p_mul_0(_S302, scale_0[col_0], &kernelContext_40);
            value_4 = _S303;
            uint m_1 = 0U;
            for(;;)
            {
                if(m_1 < col_0)
                {
                }
                else
                {
                    break;
                }
                float2 _S304 = p_mul_0(factor_0[_S301 / 2U + m_1], factor_0[col_0 * (col_0 + 1U) / 2U + m_1], &kernelContext_40);
                float2 _S305 = p_sub_0(value_4, _S304, &kernelContext_40);
                uint _S306 = m_1 + 1U;
                value_4 = _S305;
                m_1 = _S306;
            }
            if(row_0 == col_0)
            {
                if(!((value_4.x) > 0.0f))
                {
                    _S183 = true;
                }
                else
                {
                    _S183 = !p_finite_0(value_4);
                }
                uint error_2;
                if(_S183)
                {
                    error_2 = error_1 | 4U;
                }
                else
                {
                    error_2 = error_1;
                }
                float2 _S307 = p_sqrt_0(value_4, &kernelContext_40);
                value_3 = _S307;
                error_1 = error_2;
            }
            else
            {
                float2 _S308 = p_div_0(value_4, factor_0[col_0 * (col_0 + 1U) / 2U + col_0], &kernelContext_40);
                value_3 = _S308;
            }
            factor_0[_S301 / 2U + col_0] = value_3;
            col_0 = col_0 + 1U;
        }
        row_0 = row_0 + 1U;
        r_14 = error_1;
    }
    set_frame_0(k_7, 1U, rotations_0, &kernelContext_40);
    set_frame_0(k_7, 2U, r_14 | (*((&kernelContext_40)->islands_0+blast_frame_0((&kernelContext_40)->params_0->node_count_1, k_7)))[2U], &kernelContext_40);
    set_frame_0(k_7, 4U, (as_type<uint>(((&axis_0)->x_2.x))), &kernelContext_40);
    set_frame_0(k_7, 5U, (as_type<uint>(((&axis_0)->y_1.x))), &kernelContext_40);
    set_frame_0(k_7, 6U, (as_type<uint>(((&axis_0)->z_1.x))), &kernelContext_40);
    i_15 = 0U;
    for(;;)
    {
        if(i_15 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S309 = 8U + i_15;
        if(i_15 < _S262)
        {
            second_0 = r_14 == 0U;
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
            anchor_0 = scale_0[i_15].x;
        }
        else
        {
            anchor_0 = 0.0f;
        }
        set_frame_0(k_7, _S309, (as_type<uint>((anchor_0))), &kernelContext_40);
        i_15 = i_15 + 1U;
    }
    col_0 = 0U;
    for(;;)
    {
        if(col_0 < 21U)
        {
        }
        else
        {
            break;
        }
        uint _S310 = 14U + col_0;
        if(col_0 < (_S262 * (_S262 + 1U) / 2U))
        {
            second_0 = r_14 == 0U;
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
            anchor_0 = factor_0[col_0].x;
        }
        else
        {
            anchor_0 = 0.0f;
        }
        set_frame_0(k_7, _S310, (as_type<uint>((anchor_0))), &kernelContext_40);
        col_0 = col_0 + 1U;
    }
    return;
}

[[kernel]] void blast_motion_selftest(uint3 id_0 [[thread_position_in_grid]], packed_uint4 device* islands_2 [[buffer(1)]], uint device* lists_2 [[buffer(5)]], BlastBond_natural_0 device* bonds_2 [[buffer(4)]], BlastNode_natural_0 device* nodes_2 [[buffer(2)]], BlastParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* motion_2 [[buffer(6)]], uint device* refs_2 [[buffer(3)]])
{
    thread KernelContext_0 kernelContext_41;
    (&kernelContext_41)->islands_0 = islands_2;
    (&kernelContext_41)->lists_0 = lists_2;
    (&kernelContext_41)->bonds_0 = bonds_2;
    (&kernelContext_41)->nodes_0 = nodes_2;
    (&kernelContext_41)->params_0 = params_2;
    (&kernelContext_41)->motion_0 = motion_2;
    (&kernelContext_41)->refs_0 = refs_2;
    threadgroup array<float2, int(32)> g_blast_part_2;
    (&kernelContext_41)->g_blast_part_0 = &g_blast_part_2;
    threadgroup array<uint, int(1024)> g_word_2;
    (&kernelContext_41)->g_word_0 = &g_word_2;
    threadgroup uint g_rotations_2;
    (&kernelContext_41)->g_rotations_0 = &g_rotations_2;
    threadgroup array<float2, int(24)> g_frame_2;
    (&kernelContext_41)->g_frame_0 = &g_frame_2;
    threadgroup array<float2, int(1024)> g_pair_2;
    (&kernelContext_41)->g_pair_0 = &g_pair_2;
    uint i_16 = id_0.x;
    if(i_16 >= (params_2->node_count_1))
    {
        return;
    }
    uint _S311 = 2U * i_16;
    float4 _S312 = float4(*((&kernelContext_41)->motion_0+_S311)) ;
    thread float s_11;
    thread float e_12;
    two_sum_0(_S312.x, _S312.y, &s_11, &e_12, &kernelContext_41);
    thread float p_6;
    thread float pe_1;
    two_prod_0(_S312.z, _S312.w, &p_6, &pe_1, &kernelContext_41);
    *((&kernelContext_41)->motion_0+(_S311 + 1U)) = packed_float4(float4(s_11, e_12, p_6, pe_1)) ;
    return;
}

