#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint order_at_0(uint i_0)
{
    return 16U + i_0;
}

struct LayoutParams_0
{
    uint parent_count_0;
    uint island_count_0;
    uint next_id_0;
    uint substeps_0;
    uint chunk_words_0;
    uint bond_words_0;
    uint dyn_words_0;
    uint island_words_0;
    uint chunk_count_0;
    uint statics_base_0;
    uint statics_bonds_0;
    uint resident_0;
    uint props_base_0;
    uint eload_base_0;
    uint pulse_base_0;
    uint pulse_count_base_0;
    uint pulse_cap_0;
    uint engine_base_0;
    uint total_0;
    uint gate_flag_base_0;
    uint bond_count_0;
    uint driven_0;
    uint pad2_0;
    uint pad3_0;
};

struct KernelContext_0
{
    LayoutParams_0 constant* params_0;
    uint device* plan_0;
    packed_uint4 device* parents_0;
    packed_float4 device* old_islands_0;
    packed_float4 device* children_0;
    uint device* work_0;
    packed_float4 device* chunks_0;
    packed_float4 device* old_chunks_0;
    packed_float4 device* state_0;
    packed_float4 device* old_state_0;
    packed_float4 device* scratch_0;
    packed_float4 device* ctab_0;
    packed_float4 device* old_ctab_0;
    packed_float4 device* old_bonds_0;
    packed_float4 device* bonds_0;
    packed_float4 device* dyn_0;
    packed_float4 device* old_dyn_0;
    uint device* old_index_0;
    uint device* index_0;
    packed_float4 device* frames_0;
    packed_float4 device* islands_0;
    packed_uint4 device* spawn_slots_0;
    packed_float4 device* spawn_extras_0;
    packed_float4 device* gate_0;
    uint threadgroup* g_cut_first_0;
};

uint cap_0(KernelContext_0 thread* kernelContext_0)
{
    return kernelContext_0->params_0->island_count_0 + kernelContext_0->params_0->total_0;
}

uint cap_1(KernelContext_0 thread* kernelContext_1)
{
    return kernelContext_1->params_0->island_count_0 + kernelContext_1->params_0->total_0;
}

uint child_id_at_0(uint slot_0, KernelContext_0 thread* kernelContext_2)
{
    uint _S1 = cap_0(kernelContext_2);
    return 16U + _S1 + kernelContext_2->params_0->total_0 + slot_0;
}

uint old_new_at_0(uint j_0, KernelContext_0 thread* kernelContext_3)
{
    uint _S2 = cap_0(kernelContext_3);
    return 16U + _S2 + 2U * kernelContext_3->params_0->total_0 + j_0;
}

uint child_island_at_0(uint slot_1, KernelContext_0 thread* kernelContext_4)
{
    uint _S3 = cap_1(kernelContext_4);
    return 16U + _S3 + slot_1;
}

[[kernel]] void layout_plan(uint3 id_0 [[thread_position_in_grid]], LayoutParams_0 constant* params_1 [[buffer(0)]], uint device* plan_1 [[buffer(5)]], packed_uint4 device* parents_1 [[buffer(1)]], packed_float4 device* old_islands_1 [[buffer(16)]], packed_float4 device* children_1 [[buffer(3)]], uint device* work_1 [[buffer(2)]], packed_float4 device* chunks_1 [[buffer(7)]], packed_float4 device* old_chunks_1 [[buffer(6)]], packed_float4 device* state_1 [[buffer(9)]], packed_float4 device* old_state_1 [[buffer(8)]], packed_float4 device* scratch_1 [[buffer(18)]], packed_float4 device* ctab_1 [[buffer(20)]], packed_float4 device* old_ctab_1 [[buffer(19)]], packed_float4 device* old_bonds_1 [[buffer(10)]], packed_float4 device* bonds_1 [[buffer(11)]], packed_float4 device* dyn_1 [[buffer(13)]], packed_float4 device* old_dyn_1 [[buffer(12)]], uint device* old_index_1 [[buffer(14)]], uint device* index_1 [[buffer(15)]], packed_float4 device* frames_1 [[buffer(4)]], packed_float4 device* islands_1 [[buffer(17)]], packed_uint4 device* spawn_slots_1 [[buffer(22)]], packed_float4 device* spawn_extras_1 [[buffer(23)]], packed_float4 device* gate_1 [[buffer(21)]])
{
    uint k_0;
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->params_0 = params_1;
    (&kernelContext_5)->plan_0 = plan_1;
    (&kernelContext_5)->parents_0 = parents_1;
    (&kernelContext_5)->old_islands_0 = old_islands_1;
    (&kernelContext_5)->children_0 = children_1;
    (&kernelContext_5)->work_0 = work_1;
    (&kernelContext_5)->chunks_0 = chunks_1;
    (&kernelContext_5)->old_chunks_0 = old_chunks_1;
    (&kernelContext_5)->state_0 = state_1;
    (&kernelContext_5)->old_state_0 = old_state_1;
    (&kernelContext_5)->scratch_0 = scratch_1;
    (&kernelContext_5)->ctab_0 = ctab_1;
    (&kernelContext_5)->old_ctab_0 = old_ctab_1;
    (&kernelContext_5)->old_bonds_0 = old_bonds_1;
    (&kernelContext_5)->bonds_0 = bonds_1;
    (&kernelContext_5)->dyn_0 = dyn_1;
    (&kernelContext_5)->old_dyn_0 = old_dyn_1;
    (&kernelContext_5)->old_index_0 = old_index_1;
    (&kernelContext_5)->index_0 = index_1;
    (&kernelContext_5)->frames_0 = frames_1;
    (&kernelContext_5)->islands_0 = islands_1;
    (&kernelContext_5)->spawn_slots_0 = spawn_slots_1;
    (&kernelContext_5)->spawn_extras_0 = spawn_extras_1;
    (&kernelContext_5)->gate_0 = gate_1;
    threadgroup uint g_cut_first_1;
    (&kernelContext_5)->g_cut_first_0 = &g_cut_first_1;
    uint n_0 = params_1->island_count_0;
    uint i_1 = 0U;
    for(;;)
    {
        if(i_1 < n_0)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_5)->plan_0+order_at_0(i_1)) = i_1;
        i_1 = i_1 + 1U;
    }
    uint _S4 = (&kernelContext_5)->params_0->next_id_0;
    uint q_0 = (&kernelContext_5)->params_0->parent_count_0;
    uint len_0 = n_0;
    uint next_0 = _S4;
    for(;;)
    {
        if(q_0 > 0U)
        {
        }
        else
        {
            break;
        }
        uint p_0 = q_0 - 1U;
        uint4 _S5 = uint4(*((&kernelContext_5)->parents_0+p_0)) ;
        uint _S6 = _S5.z;
        if(_S6 < 2U)
        {
            q_0 = p_0;
            continue;
        }
        uint len_1 = len_0 - 1U;
        uint _S7 = _S5.x;
        if(_S7 < len_1)
        {
            *((&kernelContext_5)->plan_0+order_at_0(_S7)) = *((&kernelContext_5)->plan_0+order_at_0(len_1));
        }
        k_0 = 0U;
        uint len_2 = len_1;
        uint next_1 = next_0;
        for(;;)
        {
            if(k_0 < _S6)
            {
            }
            else
            {
                break;
            }
            uint _S8 = _S5.y;
            *((&kernelContext_5)->plan_0+order_at_0(len_2)) = n_0 + _S8 + k_0;
            uint _S9 = len_2 + 1U;
            uint _S10 = child_id_at_0(_S8 + k_0, &kernelContext_5);
            *((&kernelContext_5)->plan_0+_S10) = next_1;
            uint _S11 = next_1 + 1U;
            k_0 = k_0 + 1U;
            len_2 = _S9;
            next_1 = _S11;
        }
        len_0 = len_2;
        next_0 = next_1;
        q_0 = p_0;
    }
    *((&kernelContext_5)->plan_0+int(0)) = len_0;
    k_0 = 0U;
    for(;;)
    {
        if(k_0 < n_0)
        {
        }
        else
        {
            break;
        }
        uint _S12 = old_new_at_0(k_0, &kernelContext_5);
        *((&kernelContext_5)->plan_0+_S12) = 4294967295U;
        k_0 = k_0 + 1U;
    }
    i_1 = 0U;
    for(;;)
    {
        if(i_1 < len_0)
        {
        }
        else
        {
            break;
        }
        uint device* _S13 = (&kernelContext_5)->plan_0+order_at_0(i_1);
        uint e_0 = *_S13;
        if((*_S13) < n_0)
        {
            uint _S14 = old_new_at_0(e_0, &kernelContext_5);
            *((&kernelContext_5)->plan_0+_S14) = i_1;
        }
        else
        {
            uint _S15 = child_island_at_0(e_0 - n_0, &kernelContext_5);
            *((&kernelContext_5)->plan_0+_S15) = i_1;
        }
        i_1 = i_1 + 1U;
    }
    return;
}

uint island_word_0(uint i_2, uint w_0, KernelContext_0 thread* kernelContext_6)
{
    return kernelContext_6->params_0->island_words_0 * i_2 + w_0;
}

uint child_by_island_0(uint4 parent_0, uint r_0, KernelContext_0 thread* kernelContext_7)
{
    uint k_1 = 0U;
    for(;;)
    {
        uint _S16 = parent_0.z;
        if(k_1 < _S16)
        {
        }
        else
        {
            break;
        }
        uint _S17 = parent_0.y;
        uint _S18 = child_island_at_0(_S17 + k_1, kernelContext_7);
        uint _S19 = *(kernelContext_7->plan_0+_S18);
        uint j_1 = 0U;
        uint below_0 = 0U;
        for(;;)
        {
            if(j_1 < _S16)
            {
            }
            else
            {
                break;
            }
            uint _S20 = child_island_at_0(_S17 + j_1, kernelContext_7);
            if((*(kernelContext_7->plan_0+_S20)) < _S19)
            {
                below_0 = below_0 + 1U;
            }
            j_1 = j_1 + 1U;
        }
        if(below_0 == r_0)
        {
            return k_1;
        }
        k_1 = k_1 + 1U;
    }
    return 0U;
}

uint new_chunk_at_0(uint slot_2, KernelContext_0 thread* kernelContext_8)
{
    uint _S21 = cap_0(kernelContext_8);
    return 16U + _S21 + 2U * kernelContext_8->params_0->total_0 + kernelContext_8->params_0->island_count_0 + slot_2;
}

uint chunk_word_0(uint c_0, uint w_1, KernelContext_0 thread* kernelContext_9)
{
    return kernelContext_9->params_0->chunk_words_0 * c_0 + w_1;
}

void set_chunk_island_0(uint c_1, uint island_0, KernelContext_0 thread* kernelContext_10)
{
    uint _S22 = chunk_word_0(c_1, 8U, kernelContext_10);
    thread float4 info_0 = float4(*(kernelContext_10->chunks_0+_S22)) ;
    info_0.y = (as_type<float>((island_0)));
    *(kernelContext_10->chunks_0+_S22) = packed_float4(info_0) ;
    return;
}

uint bond_word_0(uint b_0, uint w_2, KernelContext_0 thread* kernelContext_11)
{
    return kernelContext_11->params_0->bond_words_0 * b_0 + w_2;
}

uint bond_new_at_0(uint b_1, KernelContext_0 thread* kernelContext_12)
{
    uint _S23 = cap_0(kernelContext_12);
    return 16U + _S23 + 5U * kernelContext_12->params_0->total_0 + kernelContext_12->params_0->island_count_0 + b_1;
}

uint child_bonds_at_0(uint slot_3, KernelContext_0 thread* kernelContext_13)
{
    uint _S24 = cap_0(kernelContext_13);
    return 16U + _S24 + 3U * kernelContext_13->params_0->total_0 + kernelContext_13->params_0->island_count_0 + 2U * slot_3;
}

void inverse3_0(float3 r0_0, float3 r1_0, float3 r2_0, float3 thread* i0_0, float3 thread* i1_0, float3 thread* i2_0)
{
    float3 c0_0 = cross(r1_0, r2_0);
    float3 c1_0 = cross(r2_0, r0_0);
    float3 c2_0 = cross(r0_0, r1_0);
    float det_0 = dot(r0_0, c0_0);
    float inv_0;
    if(det_0 != 0.0f)
    {
        inv_0 = 1.0f / det_0;
    }
    else
    {
        inv_0 = 0.0f;
    }
    *i0_0 = float3(c0_0.x, c1_0.x, c2_0.x) * float3(inv_0) ;
    *i1_0 = float3(c0_0.y, c1_0.y, c2_0.y) * float3(inv_0) ;
    *i2_0 = float3(c0_0.z, c1_0.z, c2_0.z) * float3(inv_0) ;
    return;
}

struct Quat_0
{
    float w_3;
    float x_0;
    float y_0;
    float z_0;
};

Quat_0 quat_of_0(float4 q_1)
{
    thread Quat_0 r_1;
    (&r_1)->x_0 = q_1.x;
    (&r_1)->y_0 = q_1.y;
    (&r_1)->z_0 = q_1.z;
    (&r_1)->w_3 = q_1.w;
    return r_1;
}

float3 rotate_0(const Quat_0 thread* q_2, float3 v_0)
{
    float3 qv_0 = float3(q_2->x_0, q_2->y_0, q_2->z_0);
    float3 t_0 = cross(qv_0, v_0) * float3(2.0f) ;
    return v_0 + t_0 * float3(q_2->w_3)  + cross(qv_0, t_0);
}

float3 inverse_rotate_0(const Quat_0 thread* q_3, float3 v_1)
{
    thread Quat_0 c_2;
    (&c_2)->w_3 = q_3->w_3;
    (&c_2)->x_0 = - q_3->x_0;
    (&c_2)->y_0 = - q_3->y_0;
    (&c_2)->z_0 = - q_3->z_0;
    thread Quat_0 _S25 = c_2;
    float3 _S26 = rotate_0(&_S25, v_1);
    return _S26;
}

float4 quat_vec_0(const Quat_0 thread* q_4)
{
    return float4(q_4->x_0, q_4->y_0, q_4->z_0, q_4->w_3);
}

[[kernel]] void layout_parent(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], LayoutParams_0 constant* params_2 [[buffer(0)]], uint device* plan_2 [[buffer(5)]], packed_uint4 device* parents_2 [[buffer(1)]], packed_float4 device* old_islands_2 [[buffer(16)]], packed_float4 device* children_2 [[buffer(3)]], uint device* work_2 [[buffer(2)]], packed_float4 device* chunks_2 [[buffer(7)]], packed_float4 device* old_chunks_2 [[buffer(6)]], packed_float4 device* state_2 [[buffer(9)]], packed_float4 device* old_state_2 [[buffer(8)]], packed_float4 device* scratch_2 [[buffer(18)]], packed_float4 device* ctab_2 [[buffer(20)]], packed_float4 device* old_ctab_2 [[buffer(19)]], packed_float4 device* old_bonds_2 [[buffer(10)]], packed_float4 device* bonds_2 [[buffer(11)]], packed_float4 device* dyn_2 [[buffer(13)]], packed_float4 device* old_dyn_2 [[buffer(12)]], uint device* old_index_2 [[buffer(14)]], uint device* index_2 [[buffer(15)]], packed_float4 device* frames_2 [[buffer(4)]], packed_float4 device* islands_2 [[buffer(17)]], packed_uint4 device* spawn_slots_2 [[buffer(22)]], packed_float4 device* spawn_extras_2 [[buffer(23)]], packed_float4 device* gate_2 [[buffer(21)]])
{
    uint w_4;
    uint w_5;
    uint w_6;
    thread KernelContext_0 kernelContext_14;
    (&kernelContext_14)->params_0 = params_2;
    (&kernelContext_14)->plan_0 = plan_2;
    (&kernelContext_14)->parents_0 = parents_2;
    (&kernelContext_14)->old_islands_0 = old_islands_2;
    (&kernelContext_14)->children_0 = children_2;
    (&kernelContext_14)->work_0 = work_2;
    (&kernelContext_14)->chunks_0 = chunks_2;
    (&kernelContext_14)->old_chunks_0 = old_chunks_2;
    (&kernelContext_14)->state_0 = state_2;
    (&kernelContext_14)->old_state_0 = old_state_2;
    (&kernelContext_14)->scratch_0 = scratch_2;
    (&kernelContext_14)->ctab_0 = ctab_2;
    (&kernelContext_14)->old_ctab_0 = old_ctab_2;
    (&kernelContext_14)->old_bonds_0 = old_bonds_2;
    (&kernelContext_14)->bonds_0 = bonds_2;
    (&kernelContext_14)->dyn_0 = dyn_2;
    (&kernelContext_14)->old_dyn_0 = old_dyn_2;
    (&kernelContext_14)->old_index_0 = old_index_2;
    (&kernelContext_14)->index_0 = index_2;
    (&kernelContext_14)->frames_0 = frames_2;
    (&kernelContext_14)->islands_0 = islands_2;
    (&kernelContext_14)->spawn_slots_0 = spawn_slots_2;
    (&kernelContext_14)->spawn_extras_0 = spawn_extras_2;
    (&kernelContext_14)->gate_0 = gate_2;
    threadgroup uint g_cut_first_2;
    (&kernelContext_14)->g_cut_first_0 = &g_cut_first_2;
    uint tid_0 = thread_0.x;
    uint p_1 = group_0.x;
    if(p_1 >= (params_2->parent_count_0))
    {
        return;
    }
    uint4 _S27 = uint4(*((&kernelContext_14)->parents_0+p_1)) ;
    uint _S28 = _S27.z;
    if(_S28 < 2U)
    {
        return;
    }
    uint _S29 = island_word_0(_S27.x, 0U, &kernelContext_14);
    float4 _S30 = float4(*((&kernelContext_14)->old_islands_0+_S29)) ;
    uint c0_1 = (as_type<uint>((_S30.x)));
    uint c1_1 = (as_type<uint>((_S30.y)));
    uint b0_0 = (as_type<uint>((_S30.z)));
    uint b1_0 = (as_type<uint>((_S30.w)));
    uint _S31 = c1_1 - c0_1;
    uint _S32 = _S27.y;
    uint _S33 = (&kernelContext_14)->params_0->total_0;
    bool _S34 = tid_0 == 0U;
    uint m_0;
    uint w_7;
    uint e_1;
    if(_S34)
    {
        m_0 = 0U;
        w_7 = 0U;
        for(;;)
        {
            if(m_0 < _S28)
            {
            }
            else
            {
                break;
            }
            uint _S35 = child_by_island_0(_S27, m_0, &kernelContext_14);
            float4 _S36 = float4(*((&kernelContext_14)->children_0+8U * (_S32 + _S35))) ;
            uint _S37 = (as_type<uint>((_S36.y)));
            uint _S38 = (as_type<uint>((_S36.z)));
            e_1 = 0U;
            for(;;)
            {
                if(e_1 < _S38)
                {
                }
                else
                {
                    break;
                }
                uint _S39 = new_chunk_at_0(_S32 + (&kernelContext_14)->work_0[_S33 + _S32 + _S37 + e_1], &kernelContext_14);
                *((&kernelContext_14)->plan_0+_S39) = w_7;
                uint _S40 = w_7 + 1U;
                e_1 = e_1 + 1U;
                w_7 = _S40;
            }
            m_0 = m_0 + 1U;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    m_0 = tid_0;
    for(;;)
    {
        if(m_0 < _S31)
        {
        }
        else
        {
            break;
        }
        uint _S41 = _S32 + m_0;
        uint _S42 = new_chunk_at_0(_S41, &kernelContext_14);
        uint nk_0 = *((&kernelContext_14)->plan_0+_S42);
        uint _S43 = child_island_at_0(_S32 + (&kernelContext_14)->work_0[_S41], &kernelContext_14);
        uint island_1 = *((&kernelContext_14)->plan_0+_S43);
        w_5 = 0U;
        for(;;)
        {
            if(w_5 < ((&kernelContext_14)->params_0->chunk_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S44 = chunk_word_0(c0_1 + nk_0, w_5, &kernelContext_14);
            packed_float4 device* _S45 = (&kernelContext_14)->chunks_0+_S44;
            uint _S46 = chunk_word_0(c0_1 + m_0, w_5, &kernelContext_14);
            *_S45 = packed_float4(float4(*((&kernelContext_14)->old_chunks_0+_S46)) ) ;
            w_5 = w_5 + 1U;
        }
        uint _S47 = c0_1 + nk_0;
        set_chunk_island_0(_S47, island_1, &kernelContext_14);
        w_6 = 0U;
        for(;;)
        {
            if(w_6 < 4U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_14)->state_0+(4U * _S47 + w_6)) = packed_float4(float4(*((&kernelContext_14)->old_state_0+(4U * (c0_1 + m_0) + w_6))) ) ;
            w_6 = w_6 + 1U;
        }
        w_4 = 0U;
        for(;;)
        {
            if(w_4 < 23U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_14)->scratch_0+((&kernelContext_14)->params_0->statics_base_0 + 23U * _S47 + w_4)) = packed_float4(float4(0.0f) ) ;
            w_4 = w_4 + 1U;
        }
        if(((&kernelContext_14)->params_0->resident_0) != 0U)
        {
            uint o_0 = c0_1 + m_0;
            *((&kernelContext_14)->ctab_0+((&kernelContext_14)->params_0->props_base_0 + _S47)) = packed_float4(float4(*((&kernelContext_14)->old_ctab_0+((&kernelContext_14)->params_0->props_base_0 + o_0))) ) ;
            uint _S48 = 2U * _S47;
            uint _S49 = 2U * o_0;
            *((&kernelContext_14)->ctab_0+((&kernelContext_14)->params_0->eload_base_0 + _S48)) = packed_float4(float4(*((&kernelContext_14)->old_ctab_0+((&kernelContext_14)->params_0->eload_base_0 + _S49))) ) ;
            *((&kernelContext_14)->ctab_0+((&kernelContext_14)->params_0->eload_base_0 + _S48 + 1U)) = packed_float4(float4(*((&kernelContext_14)->old_ctab_0+((&kernelContext_14)->params_0->eload_base_0 + _S49 + 1U))) ) ;
            *((&kernelContext_14)->ctab_0+((&kernelContext_14)->params_0->pulse_count_base_0 + _S47)) = packed_float4(float4(*((&kernelContext_14)->old_ctab_0+((&kernelContext_14)->params_0->pulse_count_base_0 + o_0))) ) ;
            w_7 = 0U;
            for(;;)
            {
                if(w_7 < (3U * (&kernelContext_14)->params_0->pulse_cap_0))
                {
                }
                else
                {
                    break;
                }
                *((&kernelContext_14)->ctab_0+((&kernelContext_14)->params_0->pulse_base_0 + 3U * (&kernelContext_14)->params_0->pulse_cap_0 * _S47 + w_7)) = packed_float4(float4(*((&kernelContext_14)->old_ctab_0+((&kernelContext_14)->params_0->pulse_base_0 + 3U * (&kernelContext_14)->params_0->pulse_cap_0 * o_0 + w_7))) ) ;
                w_7 = w_7 + 1U;
            }
        }
        m_0 = m_0 + 64U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint q_5;
    bool scaled_0;
    if(_S34)
    {
        w_7 = 0U;
        e_1 = 0U;
        for(;;)
        {
            if(w_7 < _S28)
            {
            }
            else
            {
                break;
            }
            uint _S50 = child_by_island_0(_S27, w_7, &kernelContext_14);
            uint _S51 = _S32 + _S50;
            uint _S52 = child_island_at_0(_S51, &kernelContext_14);
            uint _S53 = *((&kernelContext_14)->plan_0+_S52);
            q_5 = b0_0;
            w_4 = e_1;
            for(;;)
            {
                if(q_5 < b1_0)
                {
                }
                else
                {
                    break;
                }
                uint _S54 = bond_word_0(q_5, 14U, &kernelContext_14);
                uint4 ids_0 = (as_type<uint4>((float4(*((&kernelContext_14)->old_bonds_0+_S54)) )));
                uint _S55 = _S32 + ids_0.y - c0_1;
                uint _S56 = _S32 + ids_0.z - c0_1;
                uint kb_0 = (&kernelContext_14)->work_0[_S56];
                if((&kernelContext_14)->work_0[_S55] != _S50)
                {
                    scaled_0 = true;
                }
                else
                {
                    scaled_0 = kb_0 != _S50;
                }
                if(scaled_0)
                {
                    q_5 = q_5 + 1U;
                    continue;
                }
                uint nb_0 = b0_0 + w_4;
                w_5 = 0U;
                for(;;)
                {
                    if(w_5 < ((&kernelContext_14)->params_0->bond_words_0))
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint _S57 = bond_word_0(nb_0, w_5, &kernelContext_14);
                    packed_float4 device* _S58 = (&kernelContext_14)->bonds_0+_S57;
                    uint _S59 = bond_word_0(q_5, w_5, &kernelContext_14);
                    *_S58 = packed_float4(float4(*((&kernelContext_14)->old_bonds_0+_S59)) ) ;
                    w_5 = w_5 + 1U;
                }
                uint _S60 = bond_word_0(nb_0, 14U, &kernelContext_14);
                packed_float4 device* _S61 = (&kernelContext_14)->bonds_0+_S60;
                uint _S62 = ids_0.x;
                uint _S63 = new_chunk_at_0(_S55, &kernelContext_14);
                uint _S64 = c0_1 + *((&kernelContext_14)->plan_0+_S63);
                uint _S65 = new_chunk_at_0(_S56, &kernelContext_14);
                *_S61 = packed_float4((as_type<float4>((uint4(_S62, _S64, c0_1 + *((&kernelContext_14)->plan_0+_S65), _S53))))) ;
                w_6 = 0U;
                for(;;)
                {
                    if(w_6 < ((&kernelContext_14)->params_0->dyn_words_0))
                    {
                    }
                    else
                    {
                        break;
                    }
                    *((&kernelContext_14)->dyn_0+((&kernelContext_14)->params_0->dyn_words_0 * nb_0 + w_6)) = packed_float4(float4(*((&kernelContext_14)->old_dyn_0+((&kernelContext_14)->params_0->dyn_words_0 * q_5 + w_6))) ) ;
                    w_6 = w_6 + 1U;
                }
                uint _S66 = bond_new_at_0(q_5, &kernelContext_14);
                *((&kernelContext_14)->plan_0+_S66) = nb_0;
                w_4 = w_4 + 1U;
                q_5 = q_5 + 1U;
            }
            uint _S67 = child_bonds_at_0(_S51, &kernelContext_14);
            *((&kernelContext_14)->plan_0+_S67) = b0_0 + e_1;
            *((&kernelContext_14)->plan_0+(_S67 + 1U)) = b0_0 + w_4;
            w_7 = w_7 + 1U;
            e_1 = w_4;
        }
        *(&kernelContext_14)->g_cut_first_0 = b0_0 + e_1;
        q_5 = b0_0;
        for(;;)
        {
            if(q_5 < b1_0)
            {
            }
            else
            {
                break;
            }
            uint _S68 = bond_word_0(q_5, 14U, &kernelContext_14);
            uint4 ids_1 = (as_type<uint4>((float4(*((&kernelContext_14)->old_bonds_0+_S68)) )));
            uint _S69 = ids_1.y;
            uint _S70 = ids_1.z;
            if((&kernelContext_14)->work_0[_S32 + _S69 - c0_1] == ((&kernelContext_14)->work_0)[_S32 + _S70 - c0_1])
            {
                q_5 = q_5 + 1U;
                continue;
            }
            uint nb_1 = b0_0 + e_1;
            w_4 = 0U;
            for(;;)
            {
                if(w_4 < ((&kernelContext_14)->params_0->bond_words_0))
                {
                }
                else
                {
                    break;
                }
                uint _S71 = bond_word_0(nb_1, w_4, &kernelContext_14);
                packed_float4 device* _S72 = (&kernelContext_14)->bonds_0+_S71;
                uint _S73 = bond_word_0(q_5, w_4, &kernelContext_14);
                *_S72 = packed_float4(float4(*((&kernelContext_14)->old_bonds_0+_S73)) ) ;
                w_4 = w_4 + 1U;
            }
            uint _S74 = bond_word_0(nb_1, 14U, &kernelContext_14);
            *((&kernelContext_14)->bonds_0+_S74) = packed_float4((as_type<float4>((uint4(ids_1.x, _S69, _S70, 4294967295U))))) ;
            w_5 = 0U;
            for(;;)
            {
                if(w_5 < ((&kernelContext_14)->params_0->dyn_words_0))
                {
                }
                else
                {
                    break;
                }
                *((&kernelContext_14)->dyn_0+((&kernelContext_14)->params_0->dyn_words_0 * nb_1 + w_5)) = packed_float4(float4(*((&kernelContext_14)->old_dyn_0+((&kernelContext_14)->params_0->dyn_words_0 * q_5 + w_5))) ) ;
                w_5 = w_5 + 1U;
            }
            w_6 = 0U;
            for(;;)
            {
                if(w_6 < 3U)
                {
                }
                else
                {
                    break;
                }
                *((&kernelContext_14)->scratch_0+(3U * nb_1 + w_6)) = packed_float4(float4(0.0f) ) ;
                w_6 = w_6 + 1U;
            }
            uint _S75 = 2U * nb_1;
            packed_float4 _S76 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_14)->scratch_0+((&kernelContext_14)->params_0->statics_base_0 + 23U * (&kernelContext_14)->params_0->chunk_count_0 + _S75)) = _S76;
            *((&kernelContext_14)->scratch_0+((&kernelContext_14)->params_0->statics_base_0 + 23U * (&kernelContext_14)->params_0->chunk_count_0 + _S75 + 1U)) = _S76;
            uint _S77 = bond_new_at_0(q_5, &kernelContext_14);
            *((&kernelContext_14)->plan_0+_S77) = 4294967295U;
            e_1 = e_1 + 1U;
            q_5 = q_5 + 1U;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bool _S78;
    if((*(&kernelContext_14)->g_cut_first_0) < b1_0)
    {
        w_7 = 2U * *(&kernelContext_14)->g_cut_first_0;
    }
    else
    {
        uint _S79 = (&kernelContext_14)->old_index_0[c0_1];
        w_7 = 4294967295U;
        e_1 = _S79;
        for(;;)
        {
            if(e_1 < ((&kernelContext_14)->old_index_0)[c1_1])
            {
                scaled_0 = w_7 == 4294967295U;
            }
            else
            {
                scaled_0 = false;
            }
            if(scaled_0)
            {
            }
            else
            {
                break;
            }
            uint entry_0 = (&kernelContext_14)->old_index_0[e_1];
            uint bond_0 = entry_0 >> 1U;
            if(bond_0 < b0_0)
            {
                _S78 = true;
            }
            else
            {
                _S78 = bond_0 >= b1_0;
            }
            if(_S78)
            {
                w_7 = entry_0;
            }
            e_1 = e_1 + 1U;
        }
    }
    if(_S34)
    {
        uint _S80 = (&kernelContext_14)->old_index_0[c0_1];
        e_1 = 0U;
        w_4 = _S80;
        for(;;)
        {
            if(e_1 < _S28)
            {
            }
            else
            {
                break;
            }
            uint _S81 = child_by_island_0(_S27, e_1, &kernelContext_14);
            float4 _S82 = float4(*((&kernelContext_14)->children_0+8U * (_S32 + _S81))) ;
            uint _S83 = (as_type<uint>((_S82.y)));
            uint _S84 = (as_type<uint>((_S82.z)));
            q_5 = 0U;
            for(;;)
            {
                if(q_5 < _S84)
                {
                }
                else
                {
                    break;
                }
                uint m_1 = (&kernelContext_14)->work_0[_S33 + _S32 + _S83 + q_5];
                uint _S85 = new_chunk_at_0(_S32 + m_1, &kernelContext_14);
                *((&kernelContext_14)->index_0+(c0_1 + *((&kernelContext_14)->plan_0+_S85))) = w_4;
                uint _S86 = c0_1 + m_1;
                uint at_0 = w_4 + ((&kernelContext_14)->old_index_0[_S86 + 1U] - (&kernelContext_14)->old_index_0[_S86]);
                q_5 = q_5 + 1U;
                w_4 = at_0;
            }
            e_1 = e_1 + 1U;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    m_0 = tid_0;
    for(;;)
    {
        if(m_0 < _S31)
        {
        }
        else
        {
            break;
        }
        uint _S87 = new_chunk_at_0(_S32 + m_0, &kernelContext_14);
        uint _S88 = *((&kernelContext_14)->index_0+(c0_1 + *((&kernelContext_14)->plan_0+_S87)));
        uint _S89 = c0_1 + m_0;
        uint _S90 = _S89 + 1U;
        uint _S91 = (&kernelContext_14)->old_index_0[_S90] - (&kernelContext_14)->old_index_0[_S89];
        e_1 = (&kernelContext_14)->old_index_0[_S89];
        q_5 = 0U;
        for(;;)
        {
            if(e_1 < ((&kernelContext_14)->old_index_0)[_S90])
            {
            }
            else
            {
                break;
            }
            uint entry_1 = (&kernelContext_14)->old_index_0[e_1];
            uint bond_1 = entry_1 >> 1U;
            if(bond_1 < b0_0)
            {
                scaled_0 = true;
            }
            else
            {
                scaled_0 = bond_1 >= b1_0;
            }
            if(scaled_0)
            {
                e_1 = e_1 + 1U;
                continue;
            }
            uint _S92 = bond_new_at_0(bond_1, &kernelContext_14);
            uint device* _S93 = (&kernelContext_14)->plan_0+_S92;
            uint nb_2 = *_S93;
            if((*_S93) == 4294967295U)
            {
                e_1 = e_1 + 1U;
                continue;
            }
            *((&kernelContext_14)->index_0+(_S88 + q_5)) = 2U * nb_2 + (entry_1 & 1U);
            q_5 = q_5 + 1U;
            e_1 = e_1 + 1U;
        }
        w_4 = q_5;
        for(;;)
        {
            if(w_4 < _S91)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_14)->index_0+(_S88 + w_4)) = w_7;
            w_4 = w_4 + 1U;
        }
        m_0 = m_0 + 64U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    e_1 = tid_0;
    for(;;)
    {
        if(e_1 < _S28)
        {
        }
        else
        {
            break;
        }
        uint slot_4 = _S32 + e_1;
        uint _S94 = child_island_at_0(slot_4, &kernelContext_14);
        uint island_2 = *((&kernelContext_14)->plan_0+_S94);
        uint _S95 = 8U * slot_4;
        float4 _S96 = float4(*((&kernelContext_14)->children_0+_S95)) ;
        uint _S97 = (as_type<uint>((_S96.y)));
        uint count_0 = (as_type<uint>((_S96.z)));
        bool anchored_0 = (as_type<uint>((_S96.w))) != 0U;
        float4 _S98 = float4(*((&kernelContext_14)->children_0+(_S95 + 1U))) ;
        float3 i0_1 = (float4(*((&kernelContext_14)->children_0+(_S95 + 2U))) ).xyz;
        float3 i1_1 = (float4(*((&kernelContext_14)->children_0+(_S95 + 3U))) ).xyz;
        float3 i2_1 = (float4(*((&kernelContext_14)->children_0+(_S95 + 4U))) ).xyz;
        float3 v_2 = (float4(*((&kernelContext_14)->children_0+(_S95 + 5U))) ).xyz;
        float3 w_8 = (float4(*((&kernelContext_14)->children_0+(_S95 + 6U))) ).xyz;
        float3 _S99 = float3(0.0f) ;
        q_5 = 4294967295U;
        scaled_0 = false;
        w_4 = 0U;
        float wmass_0 = 0.0f;
        float3 wmoment_0 = _S99;
        float su_mass_0 = 0.0f;
        float3 su_0 = _S99;
        for(;;)
        {
            if(w_4 < count_0)
            {
            }
            else
            {
                break;
            }
            uint m_2 = (&kernelContext_14)->work_0[_S33 + _S32 + _S97 + w_4];
            uint _S100 = new_chunk_at_0(_S32 + m_2, &kernelContext_14);
            uint _S101 = min(q_5, *((&kernelContext_14)->plan_0+_S100));
            uint _S102 = c0_1 + m_2;
            uint _S103 = chunk_word_0(_S102, 0U, &kernelContext_14);
            float4 _S104 = float4(*((&kernelContext_14)->old_chunks_0+_S103)) ;
            uint _S105 = chunk_word_0(_S102, 7U, &kernelContext_14);
            float mu_0 = (float4(*((&kernelContext_14)->old_chunks_0+_S105)) ).x;
            if(scaled_0)
            {
                scaled_0 = true;
            }
            else
            {
                scaled_0 = mu_0 != 1.0f;
            }
            float _S106 = _S104.w;
            float wmass_1 = wmass_0 + _S106 * mu_0;
            float3 _S107 = float3(_S106) ;
            float3 wmoment_1 = wmoment_0 + _S104.xyz * _S107 * float3(mu_0) ;
            float su_mass_1 = su_mass_0 + _S106;
            float3 su_1 = su_0 + (float4(*((&kernelContext_14)->old_state_0+4U * _S102)) ).xyz * _S107;
            uint _S108 = w_4 + 1U;
            q_5 = _S101;
            w_4 = _S108;
            wmass_0 = wmass_1;
            wmoment_0 = wmoment_1;
            su_mass_0 = su_mass_1;
            su_0 = su_1;
        }
        float3 wcom_0;
        if(scaled_0)
        {
            wcom_0 = wmoment_0 / float3(wmass_0) ;
        }
        else
        {
            wcom_0 = _S98.xyz;
        }
        float3 w0_0;
        float3 w1_0;
        float3 w2_0;
        if(scaled_0)
        {
            w_5 = 0U;
            w0_0 = _S99;
            w1_0 = _S99;
            w2_0 = _S99;
            for(;;)
            {
                if(w_5 < count_0)
                {
                }
                else
                {
                    break;
                }
                uint _S109 = c0_1 + (&kernelContext_14)->work_0[_S33 + _S32 + _S97 + w_5];
                uint _S110 = chunk_word_0(_S109, 0U, &kernelContext_14);
                float4 _S111 = float4(*((&kernelContext_14)->old_chunks_0+_S110)) ;
                uint _S112 = chunk_word_0(_S109, 7U, &kernelContext_14);
                float mu_1 = (float4(*((&kernelContext_14)->old_chunks_0+_S112)) ).x;
                float3 rr_0 = _S111.xyz - wcom_0;
                float dd_0 = dot(rr_0, rr_0);
                uint _S113 = chunk_word_0(_S109, 1U, &kernelContext_14);
                float3 _S114 = float3(_S111.w) ;
                float3 _S115 = float3(mu_1) ;
                float3 w0_1 = w0_0 + ((float4(*((&kernelContext_14)->old_chunks_0+_S113)) ).xyz + (float3(dd_0, 0.0f, 0.0f) - float3(rr_0.x)  * rr_0) * _S114) * _S115;
                uint _S116 = chunk_word_0(_S109, 2U, &kernelContext_14);
                float3 w1_1 = w1_0 + ((float4(*((&kernelContext_14)->old_chunks_0+_S116)) ).xyz + (float3(0.0f, dd_0, 0.0f) - float3(rr_0.y)  * rr_0) * _S114) * _S115;
                uint _S117 = chunk_word_0(_S109, 3U, &kernelContext_14);
                float3 w2_1 = w2_0 + ((float4(*((&kernelContext_14)->old_chunks_0+_S117)) ).xyz + (float3(0.0f, 0.0f, dd_0) - float3(rr_0.z)  * rr_0) * _S114) * _S115;
                w_5 = w_5 + 1U;
                w0_0 = w0_1;
                w1_0 = w1_1;
                w2_0 = w2_1;
            }
        }
        else
        {
            w0_0 = i0_1;
            w1_0 = i1_1;
            w2_0 = i2_1;
        }
        thread float3 v0_0;
        thread float3 v1_0;
        thread float3 v2_0;
        inverse3_0(i0_1, i1_1, i2_1, &v0_0, &v1_0, &v2_0);
        thread float3 x0_0;
        thread float3 x1_0;
        thread float3 x2_0;
        inverse3_0(w0_0, w1_0, w2_0, &x0_0, &x1_0, &x2_0);
        uint _S118 = 5U * p_1;
        Quat_0 q_6 = quat_of_0(float4(*((&kernelContext_14)->frames_0+_S118)) );
        float3 pos_0 = (float4(*((&kernelContext_14)->frames_0+(_S118 + 4U))) ).xyz;
        float3 pos_1;
        if(!anchored_0)
        {
            float3 _S119 = su_0 / float3(su_mass_0) ;
            thread Quat_0 _S120 = q_6;
            float3 _S121 = rotate_0(&_S120, _S119);
            pos_1 = pos_0 + _S121;
        }
        else
        {
            pos_1 = pos_0;
        }
        thread Quat_0 _S122 = q_6;
        float3 _S123 = inverse_rotate_0(&_S122, w_8);
        float3 _S124 = float3(dot(i0_1, _S123), dot(i1_1, _S123), dot(i2_1, _S123));
        thread Quat_0 _S125 = q_6;
        float3 _S126 = rotate_0(&_S125, _S124);
        if(anchored_0)
        {
            m_0 = 1U;
        }
        else
        {
            m_0 = 0U;
        }
        if(((&kernelContext_14)->params_0->driven_0) != 0U)
        {
            w_7 = 2U;
        }
        else
        {
            w_7 = 0U;
        }
        uint flags_0 = (m_0 | w_7) | 16U;
        uint _S127 = child_bonds_at_0(slot_4, &kernelContext_14);
        uint bfirst_0 = *((&kernelContext_14)->plan_0+_S127);
        uint bend_0 = *((&kernelContext_14)->plan_0+(_S127 + 1U));
        thread array<float4, int(25)> rec_0;
        w_5 = 0U;
        for(;;)
        {
            if(w_5 < 25U)
            {
            }
            else
            {
                break;
            }
            rec_0[w_5] = float4(0.0f) ;
            w_5 = w_5 + 1U;
        }
        uint _S128 = c0_1 + q_5;
        rec_0[int(0)] = (as_type<float4>((uint4(_S128, _S128 + count_0, bfirst_0, bend_0))));
        rec_0[int(1)] = (as_type<float4>((uint4(flags_0, 0U, 0U, (&kernelContext_14)->params_0->substeps_0))));
        rec_0[int(2)] = _S98;
        rec_0[int(3)] = float4(i0_1, 0.0f);
        rec_0[int(4)] = float4(i1_1, 0.0f);
        rec_0[int(5)] = float4(i2_1, 0.0f);
        rec_0[int(6)] = float4(v0_0, 0.0f);
        rec_0[int(7)] = float4(v1_0, 0.0f);
        rec_0[int(8)] = float4(v2_0, 0.0f);
        float _S129;
        if(scaled_0)
        {
            _S129 = wmass_0;
        }
        else
        {
            _S129 = _S98.w;
        }
        rec_0[int(9)] = float4(wcom_0, _S129);
        rec_0[int(10)] = float4(x0_0, 0.0f);
        rec_0[int(11)] = float4(x1_0, 0.0f);
        rec_0[int(12)] = float4(x2_0, 0.0f);
        thread Quat_0 _S130 = q_6;
        float4 _S131 = quat_vec_0(&_S130);
        rec_0[int(13)] = _S131;
        rec_0[int(14)] = float4(pos_1, 0.0f);
        float3 _S132;
        if(anchored_0)
        {
            _S132 = _S99;
        }
        else
        {
            _S132 = v_2;
        }
        rec_0[int(16)] = float4(_S132, 0.0f);
        float3 _S133;
        if(anchored_0)
        {
            _S133 = _S99;
        }
        else
        {
            _S133 = w_8;
        }
        rec_0[int(18)] = float4(_S133, 0.0f);
        rec_0[int(21)] = float4(0.0f, 0.0f, 0.0f, (as_type<float>((2139095040U))));
        float3 _S134;
        if(anchored_0)
        {
            _S134 = _S99;
        }
        else
        {
            _S134 = _S126;
        }
        rec_0[int(23)] = float4(_S134, 0.0f);
        w_6 = 0U;
        for(;;)
        {
            if(w_6 < 25U)
            {
            }
            else
            {
                break;
            }
            uint _S135 = island_word_0(island_2, w_6, &kernelContext_14);
            *((&kernelContext_14)->islands_0+_S135) = packed_float4(rec_0[w_6]) ;
            w_6 = w_6 + 1U;
        }
        if(((&kernelContext_14)->params_0->resident_0) != 0U)
        {
            uint4 _S136 = uint4(*((&kernelContext_14)->spawn_slots_0+slot_4)) ;
            uint _S137 = 2U * slot_4;
            float4 _S138 = float4(*((&kernelContext_14)->spawn_extras_0+(_S137 + 1U))) ;
            uint at_1 = (&kernelContext_14)->params_0->engine_base_0 + 3U * island_2;
            if(anchored_0)
            {
                _S78 = true;
            }
            else
            {
                _S78 = (_S136.w) == 0U;
            }
            if(_S78)
            {
                *((&kernelContext_14)->ctab_0+at_1) = packed_float4(float4((as_type<float>((4294967295U))), 0.0f, 0.0f, 0.0f)) ;
                packed_float4 _S139 = packed_float4(float4(0.0f) ) ;
                *((&kernelContext_14)->ctab_0+(at_1 + 1U)) = _S139;
                *((&kernelContext_14)->ctab_0+(at_1 + 2U)) = _S139;
            }
            else
            {
                *((&kernelContext_14)->ctab_0+at_1) = packed_float4(float4((as_type<float>((_S136.x))), 0.0f, 0.0f, 0.0f)) ;
                *((&kernelContext_14)->ctab_0+(at_1 + 1U)) = packed_float4(float4(- _S138.x, - _S138.y, - _S138.z, _S138.w)) ;
                *((&kernelContext_14)->ctab_0+(at_1 + 2U)) = packed_float4(float4((float4(*((&kernelContext_14)->spawn_extras_0+_S137)) ).xyz, 0.0f)) ;
            }
        }
        *((&kernelContext_14)->gate_0+((&kernelContext_14)->params_0->gate_flag_base_0 + island_2)) = packed_float4(float4(0.0f) ) ;
        e_1 = e_1 + 64U;
    }
    return;
}

[[kernel]] void layout_moved(uint3 id_1 [[thread_position_in_grid]], LayoutParams_0 constant* params_3 [[buffer(0)]], uint device* plan_3 [[buffer(5)]], packed_uint4 device* parents_3 [[buffer(1)]], packed_float4 device* old_islands_3 [[buffer(16)]], packed_float4 device* children_3 [[buffer(3)]], uint device* work_3 [[buffer(2)]], packed_float4 device* chunks_3 [[buffer(7)]], packed_float4 device* old_chunks_3 [[buffer(6)]], packed_float4 device* state_3 [[buffer(9)]], packed_float4 device* old_state_3 [[buffer(8)]], packed_float4 device* scratch_3 [[buffer(18)]], packed_float4 device* ctab_3 [[buffer(20)]], packed_float4 device* old_ctab_3 [[buffer(19)]], packed_float4 device* old_bonds_3 [[buffer(10)]], packed_float4 device* bonds_3 [[buffer(11)]], packed_float4 device* dyn_3 [[buffer(13)]], packed_float4 device* old_dyn_3 [[buffer(12)]], uint device* old_index_3 [[buffer(14)]], uint device* index_3 [[buffer(15)]], packed_float4 device* frames_3 [[buffer(4)]], packed_float4 device* islands_3 [[buffer(17)]], packed_uint4 device* spawn_slots_3 [[buffer(22)]], packed_float4 device* spawn_extras_3 [[buffer(23)]], packed_float4 device* gate_3 [[buffer(21)]])
{
    thread KernelContext_0 kernelContext_15;
    (&kernelContext_15)->params_0 = params_3;
    (&kernelContext_15)->plan_0 = plan_3;
    (&kernelContext_15)->parents_0 = parents_3;
    (&kernelContext_15)->old_islands_0 = old_islands_3;
    (&kernelContext_15)->children_0 = children_3;
    (&kernelContext_15)->work_0 = work_3;
    (&kernelContext_15)->chunks_0 = chunks_3;
    (&kernelContext_15)->old_chunks_0 = old_chunks_3;
    (&kernelContext_15)->state_0 = state_3;
    (&kernelContext_15)->old_state_0 = old_state_3;
    (&kernelContext_15)->scratch_0 = scratch_3;
    (&kernelContext_15)->ctab_0 = ctab_3;
    (&kernelContext_15)->old_ctab_0 = old_ctab_3;
    (&kernelContext_15)->old_bonds_0 = old_bonds_3;
    (&kernelContext_15)->bonds_0 = bonds_3;
    (&kernelContext_15)->dyn_0 = dyn_3;
    (&kernelContext_15)->old_dyn_0 = old_dyn_3;
    (&kernelContext_15)->old_index_0 = old_index_3;
    (&kernelContext_15)->index_0 = index_3;
    (&kernelContext_15)->frames_0 = frames_3;
    (&kernelContext_15)->islands_0 = islands_3;
    (&kernelContext_15)->spawn_slots_0 = spawn_slots_3;
    (&kernelContext_15)->spawn_extras_0 = spawn_extras_3;
    (&kernelContext_15)->gate_0 = gate_3;
    threadgroup uint g_cut_first_3;
    (&kernelContext_15)->g_cut_first_0 = &g_cut_first_3;
    uint i_3 = id_1.x;
    uint t_1;
    if(i_3 == (*(plan_3+int(0))))
    {
        t_1 = 0U;
        for(;;)
        {
            if(t_1 < ((&kernelContext_15)->params_0->island_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S140 = island_word_0(i_3, t_1, &kernelContext_15);
            *((&kernelContext_15)->islands_0+_S140) = packed_float4(float4(0.0f) ) ;
            t_1 = t_1 + 1U;
        }
        return;
    }
    if(i_3 >= (*((&kernelContext_15)->plan_0+int(0))))
    {
        return;
    }
    uint device* _S141 = (&kernelContext_15)->plan_0+order_at_0(i_3);
    uint e_2 = *_S141;
    bool _S142;
    if((*_S141) >= ((&kernelContext_15)->params_0->island_count_0))
    {
        _S142 = true;
    }
    else
    {
        _S142 = e_2 == i_3;
    }
    if(_S142)
    {
        return;
    }
    t_1 = 0U;
    for(;;)
    {
        if(t_1 < ((&kernelContext_15)->params_0->island_words_0))
        {
        }
        else
        {
            break;
        }
        uint _S143 = island_word_0(i_3, t_1, &kernelContext_15);
        packed_float4 device* _S144 = (&kernelContext_15)->islands_0+_S143;
        uint _S145 = island_word_0(e_2, t_1, &kernelContext_15);
        *_S144 = packed_float4(float4(*((&kernelContext_15)->old_islands_0+_S145)) ) ;
        t_1 = t_1 + 1U;
    }
    uint _S146 = island_word_0(e_2, 0U, &kernelContext_15);
    float4 _S147 = float4(*((&kernelContext_15)->old_islands_0+_S146)) ;
    uint _S148 = (as_type<uint>((_S147.y)));
    uint b0_1 = (as_type<uint>((_S147.z)));
    uint _S149 = (as_type<uint>((_S147.w)));
    uint c_3 = (as_type<uint>((_S147.x)));
    for(;;)
    {
        if(c_3 < _S148)
        {
        }
        else
        {
            break;
        }
        set_chunk_island_0(c_3, i_3, &kernelContext_15);
        c_3 = c_3 + 1U;
    }
    uint b_2 = b0_1;
    for(;;)
    {
        if(b_2 < _S149)
        {
        }
        else
        {
            break;
        }
        uint _S150 = bond_word_0(b_2, 14U, &kernelContext_15);
        thread float4 ids_2 = float4(*((&kernelContext_15)->bonds_0+_S150)) ;
        ids_2.w = (as_type<float>((i_3)));
        *((&kernelContext_15)->bonds_0+_S150) = packed_float4(ids_2) ;
        b_2 = b_2 + 1U;
    }
    if(((&kernelContext_15)->params_0->resident_0) != 0U)
    {
        t_1 = 0U;
        for(;;)
        {
            if(t_1 < 3U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_15)->ctab_0+((&kernelContext_15)->params_0->engine_base_0 + 3U * i_3 + t_1)) = packed_float4(float4(*((&kernelContext_15)->old_ctab_0+((&kernelContext_15)->params_0->engine_base_0 + 3U * e_2 + t_1))) ) ;
            t_1 = t_1 + 1U;
        }
    }
    *((&kernelContext_15)->gate_0+((&kernelContext_15)->params_0->gate_flag_base_0 + i_3)) = packed_float4(float4(*((&kernelContext_15)->gate_0+((&kernelContext_15)->params_0->gate_flag_base_0 + e_2))) ) ;
    return;
}

