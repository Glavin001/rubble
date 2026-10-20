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
    uint device* control_0;
    uint device* plan_0;
    packed_uint4 device* parents_0;
    LayoutParams_0 constant* params_0;
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
    packed_float4 device* island_copy_0;
    packed_float4 device* old_copy_0;
    array<uint, int(256)> threadgroup* g_class_0;
    uint threadgroup* g_cut_next_0;
    uint threadgroup* g_cut_first_0;
    array<uint, int(256)> threadgroup* g_scan64_0;
};

uint cap_0(KernelContext_0 thread* kernelContext_0)
{
    return *(kernelContext_0->control_0+2U) + *(kernelContext_0->control_0+6U);
}

uint cap_1(KernelContext_0 thread* kernelContext_1)
{
    return *(kernelContext_1->control_0+2U) + *(kernelContext_1->control_0+6U);
}

uint child_id_at_0(uint slot_0, KernelContext_0 thread* kernelContext_2)
{
    uint _S1 = cap_0(kernelContext_2);
    return 16U + _S1 + *(kernelContext_2->control_0+6U) + slot_0;
}

uint old_new_at_0(uint j_0, KernelContext_0 thread* kernelContext_3)
{
    uint _S2 = cap_0(kernelContext_3);
    return 16U + _S2 + 2U * *(kernelContext_3->control_0+6U) + j_0;
}

uint child_island_at_0(uint slot_1, KernelContext_0 thread* kernelContext_4)
{
    uint _S3 = cap_1(kernelContext_4);
    return 16U + _S3 + slot_1;
}

[[kernel]] void layout_plan(uint3 id_0 [[thread_position_in_grid]], uint device* control_1 [[buffer(24)]], uint device* plan_1 [[buffer(5)]], packed_uint4 device* parents_1 [[buffer(1)]], LayoutParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* old_islands_1 [[buffer(16)]], packed_float4 device* children_1 [[buffer(3)]], uint device* work_1 [[buffer(2)]], packed_float4 device* chunks_1 [[buffer(7)]], packed_float4 device* old_chunks_1 [[buffer(6)]], packed_float4 device* state_1 [[buffer(9)]], packed_float4 device* old_state_1 [[buffer(8)]], packed_float4 device* scratch_1 [[buffer(18)]], packed_float4 device* ctab_1 [[buffer(20)]], packed_float4 device* old_ctab_1 [[buffer(19)]], packed_float4 device* old_bonds_1 [[buffer(10)]], packed_float4 device* bonds_1 [[buffer(11)]], packed_float4 device* dyn_1 [[buffer(13)]], packed_float4 device* old_dyn_1 [[buffer(12)]], uint device* old_index_1 [[buffer(14)]], uint device* index_1 [[buffer(15)]], packed_float4 device* frames_1 [[buffer(4)]], packed_float4 device* islands_1 [[buffer(17)]], packed_uint4 device* spawn_slots_1 [[buffer(22)]], packed_float4 device* spawn_extras_1 [[buffer(23)]], packed_float4 device* gate_1 [[buffer(21)]], packed_float4 device* island_copy_1 [[buffer(26)]], packed_float4 device* old_copy_1 [[buffer(25)]])
{
    uint k_0;
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->control_0 = control_1;
    (&kernelContext_5)->plan_0 = plan_1;
    (&kernelContext_5)->parents_0 = parents_1;
    (&kernelContext_5)->params_0 = params_1;
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
    (&kernelContext_5)->island_copy_0 = island_copy_1;
    (&kernelContext_5)->old_copy_0 = old_copy_1;
    threadgroup array<uint, int(256)> g_class_1;
    (&kernelContext_5)->g_class_0 = &g_class_1;
    threadgroup uint g_cut_next_1;
    (&kernelContext_5)->g_cut_next_0 = &g_cut_next_1;
    threadgroup uint g_cut_first_1;
    (&kernelContext_5)->g_cut_first_0 = &g_cut_first_1;
    threadgroup array<uint, int(256)> g_scan64_1;
    (&kernelContext_5)->g_scan64_0 = &g_scan64_1;
    bool _S4;
    if((*(control_1+int(0))) != 0U)
    {
        _S4 = true;
    }
    else
    {
        _S4 = (*((&kernelContext_5)->control_0+1U)) == 0U;
    }
    if(_S4)
    {
        return;
    }
    uint n_0 = *((&kernelContext_5)->control_0+2U);
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
    uint device* _S5 = (&kernelContext_5)->control_0+3U;
    uint q_0 = *((&kernelContext_5)->control_0+1U);
    uint len_0 = n_0;
    uint next_0 = *_S5;
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
        uint4 _S6 = uint4(*((&kernelContext_5)->parents_0+p_0)) ;
        uint _S7 = _S6.z;
        if(_S7 < 2U)
        {
            q_0 = p_0;
            continue;
        }
        uint len_1 = len_0 - 1U;
        uint _S8 = _S6.x;
        if(_S8 < len_1)
        {
            *((&kernelContext_5)->plan_0+order_at_0(_S8)) = *((&kernelContext_5)->plan_0+order_at_0(len_1));
        }
        k_0 = 0U;
        uint len_2 = len_1;
        uint next_1 = next_0;
        for(;;)
        {
            if(k_0 < _S7)
            {
            }
            else
            {
                break;
            }
            uint _S9 = _S6.y;
            *((&kernelContext_5)->plan_0+order_at_0(len_2)) = n_0 + _S9 + k_0;
            uint _S10 = len_2 + 1U;
            uint _S11 = child_id_at_0(_S9 + k_0, &kernelContext_5);
            *((&kernelContext_5)->plan_0+_S11) = next_1;
            uint _S12 = next_1 + 1U;
            k_0 = k_0 + 1U;
            len_2 = _S10;
            next_1 = _S12;
        }
        len_0 = len_2;
        next_0 = next_1;
        q_0 = p_0;
    }
    *((&kernelContext_5)->plan_0+int(0)) = len_0;
    *((&kernelContext_5)->control_0+7U) = len_0;
    *((&kernelContext_5)->control_0+10U) = next_0;
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
        uint _S13 = old_new_at_0(k_0, &kernelContext_5);
        *((&kernelContext_5)->plan_0+_S13) = 4294967295U;
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
        uint device* _S14 = (&kernelContext_5)->plan_0+order_at_0(i_1);
        uint e_0 = *_S14;
        if((*_S14) < n_0)
        {
            uint _S15 = old_new_at_0(e_0, &kernelContext_5);
            *((&kernelContext_5)->plan_0+_S15) = i_1;
        }
        else
        {
            uint _S16 = child_island_at_0(e_0 - n_0, &kernelContext_5);
            *((&kernelContext_5)->plan_0+_S16) = i_1;
        }
        i_1 = i_1 + 1U;
    }
    return;
}

uint group_uniform_0(uint value_0, uint thread_0)
{
    return value_0;
}

uint island_word_0(uint i_2, uint w_0, KernelContext_0 thread* kernelContext_6)
{
    return kernelContext_6->params_0->island_words_0 * i_2 + w_0;
}

bool layout_start_0(uint p_1, uint tid_0, uint4 thread* parent_0, uint thread* c0_0, uint thread* c1_0, uint thread* b0_0, uint thread* b1_0, KernelContext_0 thread* kernelContext_7)
{
    *parent_0 = uint4(int4(int(0)) );
    *c0_0 = 0U;
    *c1_0 = 0U;
    *b0_0 = 0U;
    *b1_0 = 0U;
    bool _S17;
    if(p_1 >= (*(kernelContext_7->control_0+1U)))
    {
        _S17 = true;
    }
    else
    {
        _S17 = (*(kernelContext_7->control_0+int(0))) != 0U;
    }
    uint _S18;
    if(_S17)
    {
        _S18 = 1U;
    }
    else
    {
        _S18 = 0U;
    }
    if((group_uniform_0(_S18, tid_0)) != 0U)
    {
        return false;
    }
    uint4 _S19 = uint4(*(kernelContext_7->parents_0+p_1)) ;
    *parent_0 = _S19;
    if((_S19.z) < 2U)
    {
        _S18 = 1U;
    }
    else
    {
        _S18 = 0U;
    }
    if((group_uniform_0(_S18, tid_0)) != 0U)
    {
        return false;
    }
    uint _S20 = island_word_0((*parent_0).x, 0U, kernelContext_7);
    float4 _S21 = float4(*(kernelContext_7->old_islands_0+_S20)) ;
    *c0_0 = group_uniform_0((as_type<uint>((_S21.x))), tid_0);
    *c1_0 = group_uniform_0((as_type<uint>((_S21.y))), tid_0);
    *b0_0 = group_uniform_0((as_type<uint>((_S21.z))), tid_0);
    *b1_0 = group_uniform_0((as_type<uint>((_S21.w))), tid_0);
    return true;
}

uint child_order_at_0(uint slot_2, KernelContext_0 thread* kernelContext_8)
{
    uint _S22 = cap_0(kernelContext_8);
    return 16U + _S22 + 5U * *(kernelContext_8->control_0+6U) + *(kernelContext_8->control_0+2U) + kernelContext_8->params_0->bond_count_0 + slot_2;
}

void sort_children_0(uint4 parent_1, KernelContext_0 thread* kernelContext_9)
{
    uint k_1 = 0U;
    for(;;)
    {
        if(k_1 < (parent_1.z))
        {
        }
        else
        {
            break;
        }
        uint _S23 = parent_1.y;
        uint _S24 = child_island_at_0(_S23 + k_1, kernelContext_9);
        uint _S25 = *(kernelContext_9->plan_0+_S24);
        uint j_1 = k_1;
        for(;;)
        {
            bool _S26;
            if(j_1 > 0U)
            {
                uint _S27 = child_order_at_0(_S23 + j_1 - 1U, kernelContext_9);
                uint _S28 = child_island_at_0(_S23 + *(kernelContext_9->plan_0+_S27), kernelContext_9);
                _S26 = (*(kernelContext_9->plan_0+_S28)) > _S25;
            }
            else
            {
                _S26 = false;
            }
            if(_S26)
            {
            }
            else
            {
                break;
            }
            uint _S29 = _S23 + j_1;
            uint _S30 = child_order_at_0(_S29, kernelContext_9);
            uint device* _S31 = kernelContext_9->plan_0+_S30;
            uint _S32 = child_order_at_0(_S29 - 1U, kernelContext_9);
            *_S31 = *(kernelContext_9->plan_0+_S32);
            j_1 = j_1 - 1U;
        }
        uint _S33 = child_order_at_0(_S23 + j_1, kernelContext_9);
        *(kernelContext_9->plan_0+_S33) = k_1;
        k_1 = k_1 + 1U;
    }
    return;
}

uint child_by_island_0(uint4 parent_2, uint r_0, KernelContext_0 thread* kernelContext_10)
{
    uint _S34 = child_order_at_0(parent_2.y + r_0, kernelContext_10);
    return *(kernelContext_10->plan_0+_S34);
}

uint child_start_at_0(uint slot_3, KernelContext_0 thread* kernelContext_11)
{
    uint _S35 = cap_0(kernelContext_11);
    return 16U + _S35 + 6U * *(kernelContext_11->control_0+6U) + *(kernelContext_11->control_0+2U) + kernelContext_11->params_0->bond_count_0 + slot_3;
}

uint new_chunk_at_0(uint slot_4, KernelContext_0 thread* kernelContext_12)
{
    uint _S36 = cap_0(kernelContext_12);
    return 16U + _S36 + 2U * *(kernelContext_12->control_0+6U) + *(kernelContext_12->control_0+2U) + slot_4;
}

void lay_positions_0(uint4 parent_3, uint n_1, uint base_0, uint total_1, uint tid_1, KernelContext_0 thread* kernelContext_13)
{
    uint i_3;
    uint lo_0;
    if(tid_1 == 0U)
    {
        sort_children_0(parent_3, kernelContext_13);
        i_3 = 0U;
        lo_0 = 0U;
        for(;;)
        {
            if(i_3 < (parent_3.z))
            {
            }
            else
            {
                break;
            }
            uint _S37 = child_by_island_0(parent_3, i_3, kernelContext_13);
            uint _S38 = base_0 + _S37;
            uint _S39 = child_start_at_0(_S38, kernelContext_13);
            *(kernelContext_13->plan_0+_S39) = lo_0;
            uint at_0 = lo_0 + (as_type<uint>(((float4(*(kernelContext_13->children_0+8U * _S38)) ).z)));
            i_3 = i_3 + 1U;
            lo_0 = at_0;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    i_3 = tid_1;
    for(;;)
    {
        if(i_3 < n_1)
        {
        }
        else
        {
            break;
        }
        uint _S40 = parent_3.z - 1U;
        lo_0 = 0U;
        uint hi_0 = _S40;
        for(;;)
        {
            if(lo_0 < hi_0)
            {
            }
            else
            {
                break;
            }
            uint mid_0 = (lo_0 + hi_0 + 1U) / 2U;
            if((as_type<uint>(((float4(*(kernelContext_13->children_0+8U * (base_0 + mid_0))) ).y))) <= i_3)
            {
                lo_0 = mid_0;
            }
            else
            {
                hi_0 = mid_0 - 1U;
            }
        }
        uint _S41 = base_0 + lo_0;
        uint j_2 = i_3 - (as_type<uint>(((float4(*(kernelContext_13->children_0+8U * _S41)) ).y)));
        uint _S42 = new_chunk_at_0(base_0 + kernelContext_13->work_0[total_1 + base_0 + i_3], kernelContext_13);
        uint device* _S43 = kernelContext_13->plan_0+_S42;
        uint _S44 = child_start_at_0(_S41, kernelContext_13);
        *_S43 = *(kernelContext_13->plan_0+_S44) + j_2;
        i_3 = i_3 + 256U;
    }
    return;
}

uint chunk_word_0(uint c_0, uint w_1, KernelContext_0 thread* kernelContext_14)
{
    return kernelContext_14->params_0->chunk_words_0 * c_0 + w_1;
}

void set_chunk_island_0(uint c_1, uint island_0, KernelContext_0 thread* kernelContext_15)
{
    uint _S45 = chunk_word_0(c_1, 8U, kernelContext_15);
    thread float4 info_0 = float4(*(kernelContext_15->chunks_0+_S45)) ;
    info_0.y = (as_type<float>((island_0)));
    *(kernelContext_15->chunks_0+_S45) = packed_float4(info_0) ;
    return;
}

void lay_chunks_0(uint c0_1, uint n_2, uint base_1, uint tid_2, KernelContext_0 thread* kernelContext_16)
{
    uint m_0 = tid_2;
    for(;;)
    {
        if(m_0 < n_2)
        {
        }
        else
        {
            break;
        }
        uint _S46 = base_1 + m_0;
        uint _S47 = new_chunk_at_0(_S46, kernelContext_16);
        uint nk_0 = *(kernelContext_16->plan_0+_S47);
        uint _S48 = child_island_at_0(base_1 + kernelContext_16->work_0[_S46], kernelContext_16);
        uint island_1 = *(kernelContext_16->plan_0+_S48);
        uint w_2 = 0U;
        for(;;)
        {
            if(w_2 < (kernelContext_16->params_0->chunk_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S49 = chunk_word_0(c0_1 + nk_0, w_2, kernelContext_16);
            packed_float4 device* _S50 = kernelContext_16->chunks_0+_S49;
            uint _S51 = chunk_word_0(c0_1 + m_0, w_2, kernelContext_16);
            *_S50 = packed_float4(float4(*(kernelContext_16->old_chunks_0+_S51)) ) ;
            w_2 = w_2 + 1U;
        }
        uint _S52 = c0_1 + nk_0;
        set_chunk_island_0(_S52, island_1, kernelContext_16);
        uint w_3 = 0U;
        for(;;)
        {
            if(w_3 < 4U)
            {
            }
            else
            {
                break;
            }
            *(kernelContext_16->state_0+(4U * _S52 + w_3)) = packed_float4(float4(*(kernelContext_16->old_state_0+(4U * (c0_1 + m_0) + w_3))) ) ;
            w_3 = w_3 + 1U;
        }
        uint w_4 = 0U;
        for(;;)
        {
            if(w_4 < 23U)
            {
            }
            else
            {
                break;
            }
            *(kernelContext_16->scratch_0+(kernelContext_16->params_0->statics_base_0 + 23U * _S52 + w_4)) = packed_float4(float4(0.0f) ) ;
            w_4 = w_4 + 1U;
        }
        m_0 = m_0 + 256U;
    }
    return;
}

void lay_ctab_chunks_0(uint c0_2, uint n_3, uint base_2, uint tid_3, KernelContext_0 thread* kernelContext_17)
{
    uint m_1 = tid_3;
    for(;;)
    {
        if(m_1 < n_3)
        {
        }
        else
        {
            break;
        }
        uint _S53 = new_chunk_at_0(base_2 + m_1, kernelContext_17);
        uint o_0 = c0_2 + m_1;
        uint d_0 = c0_2 + *(kernelContext_17->plan_0+_S53);
        *(kernelContext_17->ctab_0+(kernelContext_17->params_0->props_base_0 + d_0)) = packed_float4(float4(*(kernelContext_17->old_ctab_0+(kernelContext_17->params_0->props_base_0 + o_0))) ) ;
        uint _S54 = 2U * d_0;
        uint _S55 = 2U * o_0;
        *(kernelContext_17->ctab_0+(kernelContext_17->params_0->eload_base_0 + _S54)) = packed_float4(float4(*(kernelContext_17->old_ctab_0+(kernelContext_17->params_0->eload_base_0 + _S55))) ) ;
        *(kernelContext_17->ctab_0+(kernelContext_17->params_0->eload_base_0 + _S54 + 1U)) = packed_float4(float4(*(kernelContext_17->old_ctab_0+(kernelContext_17->params_0->eload_base_0 + _S55 + 1U))) ) ;
        *(kernelContext_17->ctab_0+(kernelContext_17->params_0->pulse_count_base_0 + d_0)) = packed_float4(float4(*(kernelContext_17->old_ctab_0+(kernelContext_17->params_0->pulse_count_base_0 + o_0))) ) ;
        uint w_5 = 0U;
        for(;;)
        {
            if(w_5 < (3U * kernelContext_17->params_0->pulse_cap_0))
            {
            }
            else
            {
                break;
            }
            *(kernelContext_17->ctab_0+(kernelContext_17->params_0->pulse_base_0 + 3U * kernelContext_17->params_0->pulse_cap_0 * d_0 + w_5)) = packed_float4(float4(*(kernelContext_17->old_ctab_0+(kernelContext_17->params_0->pulse_base_0 + 3U * kernelContext_17->params_0->pulse_cap_0 * o_0 + w_5))) ) ;
            w_5 = w_5 + 1U;
        }
        m_1 = m_1 + 256U;
    }
    return;
}

uint child_bonds_at_0(uint slot_5, KernelContext_0 thread* kernelContext_18)
{
    uint _S56 = cap_0(kernelContext_18);
    return 16U + _S56 + 3U * *(kernelContext_18->control_0+6U) + *(kernelContext_18->control_0+2U) + 2U * slot_5;
}

uint bond_word_0(uint b_0, uint w_6, KernelContext_0 thread* kernelContext_19)
{
    return kernelContext_19->params_0->bond_words_0 * b_0 + w_6;
}

uint bond_new_at_0(uint b_1, KernelContext_0 thread* kernelContext_20)
{
    uint _S57 = cap_0(kernelContext_20);
    return 16U + _S57 + 5U * *(kernelContext_20->control_0+6U) + *(kernelContext_20->control_0+2U) + b_1;
}

void bond_ranks_0(uint4 parent_4, uint base_3, uint c0_3, uint b0_1, uint b1_1, uint tid_4, bool place_0, KernelContext_0 thread* kernelContext_21)
{
    bool _S58;
    uint block_0 = b0_1;
    for(;;)
    {
        if(block_0 < b1_1)
        {
        }
        else
        {
            break;
        }
        uint b_2 = block_0 + tid_4;
        uint cls_0;
        if(b_2 < b1_1)
        {
            uint _S59 = bond_word_0(b_2, 14U, kernelContext_21);
            uint4 ids_0 = (as_type<uint4>((float4(*(kernelContext_21->old_bonds_0+_S59)) )));
            uint ka_0 = kernelContext_21->work_0[base_3 + ids_0.y - c0_3];
            if(ka_0 == (kernelContext_21->work_0)[base_3 + ids_0.z - c0_3])
            {
                cls_0 = ka_0;
            }
            else
            {
                cls_0 = 4294967294U;
            }
        }
        else
        {
            cls_0 = 4294967295U;
        }
        (*kernelContext_21->g_class_0)[tid_4] = cls_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S60 = cls_0 != 4294967295U;
        bool last_0 = _S60;
        uint t_0 = 0U;
        uint rank_0 = 0U;
        for(;;)
        {
            if(t_0 < 256U)
            {
            }
            else
            {
                break;
            }
            if(t_0 < tid_4)
            {
                _S58 = ((*kernelContext_21->g_class_0)[t_0]) == cls_0;
            }
            else
            {
                _S58 = false;
            }
            if(_S58)
            {
                rank_0 = rank_0 + 1U;
            }
            bool _S61;
            if(t_0 > tid_4)
            {
                _S61 = ((*kernelContext_21->g_class_0)[t_0]) == cls_0;
            }
            else
            {
                _S61 = false;
            }
            if(_S61)
            {
                last_0 = false;
            }
            t_0 = t_0 + 1U;
        }
        if(place_0)
        {
            _S58 = _S60;
        }
        else
        {
            _S58 = false;
        }
        if(_S58)
        {
            uint first_0;
            if(cls_0 == 4294967294U)
            {
                first_0 = *kernelContext_21->g_cut_next_0;
            }
            else
            {
                uint _S62 = child_bonds_at_0(base_3 + cls_0, kernelContext_21);
                first_0 = *(kernelContext_21->plan_0+(_S62 + 1U));
            }
            uint _S63 = bond_new_at_0(b_2, kernelContext_21);
            *(kernelContext_21->plan_0+_S63) = first_0 + rank_0;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if(last_0)
        {
            if(place_0)
            {
                if(cls_0 == 4294967294U)
                {
                    *kernelContext_21->g_cut_next_0 = *kernelContext_21->g_cut_next_0 + (rank_0 + 1U);
                }
                else
                {
                    uint _S64 = child_bonds_at_0(base_3 + cls_0, kernelContext_21);
                    uint device* _S65 = kernelContext_21->plan_0+(_S64 + 1U);
                    *_S65 = *_S65 + (rank_0 + 1U);
                }
            }
            else
            {
                if(cls_0 != 4294967294U)
                {
                    uint _S66 = child_bonds_at_0(base_3 + cls_0, kernelContext_21);
                    uint device* _S67 = kernelContext_21->plan_0+_S66;
                    *_S67 = *_S67 + (rank_0 + 1U);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        block_0 = block_0 + 256U;
    }
    return;
}

void lay_bond_ranks_0(uint4 parent_5, uint c0_4, uint b0_2, uint b1_2, uint base_4, uint tid_5, KernelContext_0 thread* kernelContext_22)
{
    uint _S68;
    uint k_2 = tid_5;
    for(;;)
    {
        uint _S69 = parent_5.z;
        _S68 = _S69;
        if(k_2 < _S69)
        {
        }
        else
        {
            break;
        }
        uint _S70 = child_bonds_at_0(base_4 + k_2, kernelContext_22);
        *(kernelContext_22->plan_0+_S70) = 0U;
        k_2 = k_2 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bond_ranks_0(parent_5, base_4, c0_4, b0_2, b1_2, tid_5, false, kernelContext_22);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if(tid_5 == 0U)
    {
        uint r_1 = 0U;
        uint at_1 = 0U;
        for(;;)
        {
            if(r_1 < _S68)
            {
            }
            else
            {
                break;
            }
            uint _S71 = child_by_island_0(parent_5, r_1, kernelContext_22);
            uint _S72 = base_4 + _S71;
            uint _S73 = child_bonds_at_0(_S72, kernelContext_22);
            uint count_0 = *(kernelContext_22->plan_0+_S73);
            uint _S74 = child_bonds_at_0(_S72, kernelContext_22);
            uint _S75 = b0_2 + at_1;
            *(kernelContext_22->plan_0+_S74) = _S75;
            uint _S76 = child_bonds_at_0(_S72, kernelContext_22);
            *(kernelContext_22->plan_0+(_S76 + 1U)) = _S75;
            uint at_2 = at_1 + count_0;
            r_1 = r_1 + 1U;
            at_1 = at_2;
        }
        uint _S77 = b0_2 + at_1;
        *kernelContext_22->g_cut_first_0 = _S77;
        *kernelContext_22->g_cut_next_0 = _S77;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bond_ranks_0(parent_5, base_4, c0_4, b0_2, b1_2, tid_5, true, kernelContext_22);
    return;
}

void lay_bonds_0(uint c0_5, uint b0_3, uint b1_3, uint base_5, uint tid_6, KernelContext_0 thread* kernelContext_23)
{
    uint b_3 = b0_3 + tid_6;
    for(;;)
    {
        if(b_3 < b1_3)
        {
        }
        else
        {
            break;
        }
        uint _S78 = bond_word_0(b_3, 14U, kernelContext_23);
        uint4 ids_1 = (as_type<uint4>((float4(*(kernelContext_23->old_bonds_0+_S78)) )));
        uint _S79 = ids_1.y;
        uint _S80 = base_5 + _S79 - c0_5;
        uint ka_1 = kernelContext_23->work_0[_S80];
        uint _S81 = ids_1.z;
        uint _S82 = base_5 + _S81 - c0_5;
        uint nb_0;
        uint4 new_ids_0;
        if(ka_1 == (kernelContext_23->work_0)[_S82])
        {
            uint _S83 = bond_new_at_0(b_3, kernelContext_23);
            uint _S84 = *(kernelContext_23->plan_0+_S83);
            uint _S85 = ids_1.x;
            uint _S86 = new_chunk_at_0(_S80, kernelContext_23);
            uint _S87 = c0_5 + *(kernelContext_23->plan_0+_S86);
            uint _S88 = new_chunk_at_0(_S82, kernelContext_23);
            uint _S89 = c0_5 + *(kernelContext_23->plan_0+_S88);
            uint _S90 = child_island_at_0(base_5 + ka_1, kernelContext_23);
            uint4 _S91 = uint4(_S85, _S87, _S89, *(kernelContext_23->plan_0+_S90));
            nb_0 = _S84;
            new_ids_0 = _S91;
        }
        else
        {
            uint _S92 = bond_new_at_0(b_3, kernelContext_23);
            uint nb_1 = *(kernelContext_23->plan_0+_S92);
            uint4 _S93 = uint4(ids_1.x, _S79, _S81, 4294967295U);
            nb_0 = 0U;
            for(;;)
            {
                if(nb_0 < 3U)
                {
                }
                else
                {
                    break;
                }
                *(kernelContext_23->scratch_0+(3U * nb_1 + nb_0)) = packed_float4(float4(0.0f) ) ;
                nb_0 = nb_0 + 1U;
            }
            uint _S94 = 2U * nb_1;
            packed_float4 _S95 = packed_float4(float4(0.0f) ) ;
            *(kernelContext_23->scratch_0+(kernelContext_23->params_0->statics_base_0 + 23U * kernelContext_23->params_0->chunk_count_0 + _S94)) = _S95;
            *(kernelContext_23->scratch_0+(kernelContext_23->params_0->statics_base_0 + 23U * kernelContext_23->params_0->chunk_count_0 + _S94 + 1U)) = _S95;
            nb_0 = nb_1;
            new_ids_0 = _S93;
        }
        uint w_7 = 0U;
        for(;;)
        {
            if(w_7 < (kernelContext_23->params_0->bond_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S96 = bond_word_0(nb_0, w_7, kernelContext_23);
            packed_float4 device* _S97 = kernelContext_23->bonds_0+_S96;
            uint _S98 = bond_word_0(b_3, w_7, kernelContext_23);
            *_S97 = packed_float4(float4(*(kernelContext_23->old_bonds_0+_S98)) ) ;
            w_7 = w_7 + 1U;
        }
        uint _S99 = bond_word_0(nb_0, 14U, kernelContext_23);
        *(kernelContext_23->bonds_0+_S99) = packed_float4((as_type<float4>((new_ids_0)))) ;
        uint w_8 = 0U;
        for(;;)
        {
            if(w_8 < (kernelContext_23->params_0->dyn_words_0))
            {
            }
            else
            {
                break;
            }
            *(kernelContext_23->dyn_0+(kernelContext_23->params_0->dyn_words_0 * nb_0 + w_8)) = packed_float4(float4(*(kernelContext_23->old_dyn_0+(kernelContext_23->params_0->dyn_words_0 * b_3 + w_8))) ) ;
            w_8 = w_8 + 1U;
        }
        b_3 = b_3 + 256U;
    }
    return;
}

uint degree_at_0(uint slot_6, KernelContext_0 thread* kernelContext_24)
{
    uint _S100 = cap_0(kernelContext_24);
    return 16U + _S100 + 7U * *(kernelContext_24->control_0+6U) + *(kernelContext_24->control_0+2U) + kernelContext_24->params_0->bond_count_0 + slot_6;
}

void lay_incidence_0(uint c0_6, uint c1_1, uint b0_4, uint b1_4, uint n_4, uint base_6, uint cut_first_0, uint tid_7, KernelContext_0 thread* kernelContext_25)
{
    uint v_0;
    uint step_0;
    uint add_0;
    uint pad_0;
    uint m_2;
    bool _S101;
    if(cut_first_0 < b1_4)
    {
        pad_0 = 2U * cut_first_0;
    }
    else
    {
        uint _S102 = kernelContext_25->old_index_0[c0_6];
        pad_0 = 4294967295U;
        m_2 = _S102;
        for(;;)
        {
            if(m_2 < (kernelContext_25->old_index_0)[c1_1])
            {
                _S101 = pad_0 == 4294967295U;
            }
            else
            {
                _S101 = false;
            }
            if(_S101)
            {
            }
            else
            {
                break;
            }
            uint entry_0 = kernelContext_25->old_index_0[m_2];
            uint bond_0 = entry_0 >> 1U;
            bool _S103;
            if(bond_0 < b0_4)
            {
                _S103 = true;
            }
            else
            {
                _S103 = bond_0 >= b1_4;
            }
            if(_S103)
            {
                pad_0 = entry_0;
            }
            m_2 = m_2 + 1U;
        }
    }
    m_2 = tid_7;
    for(;;)
    {
        if(m_2 < n_4)
        {
        }
        else
        {
            break;
        }
        uint _S104 = new_chunk_at_0(base_6 + m_2, kernelContext_25);
        uint _S105 = degree_at_0(base_6 + *(kernelContext_25->plan_0+_S104), kernelContext_25);
        uint _S106 = c0_6 + m_2;
        *(kernelContext_25->plan_0+_S105) = kernelContext_25->old_index_0[_S106 + 1U] - kernelContext_25->old_index_0[_S106];
        m_2 = m_2 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint _S107 = kernelContext_25->old_index_0[c0_6];
    uint block_1 = 0U;
    uint carry_0 = _S107;
    for(;;)
    {
        if(block_1 < n_4)
        {
        }
        else
        {
            break;
        }
        uint i_4 = block_1 + tid_7;
        bool _S108 = i_4 < n_4;
        if(_S108)
        {
            uint _S109 = degree_at_0(base_6 + i_4, kernelContext_25);
            v_0 = *(kernelContext_25->plan_0+_S109);
        }
        else
        {
            v_0 = 0U;
        }
        (*kernelContext_25->g_scan64_0)[tid_7] = v_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_0 = 1U;
        for(;;)
        {
            if(step_0 < 256U)
            {
            }
            else
            {
                break;
            }
            if(tid_7 >= step_0)
            {
                add_0 = (*kernelContext_25->g_scan64_0)[tid_7 - step_0];
            }
            else
            {
                add_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*kernelContext_25->g_scan64_0)[tid_7] = (*kernelContext_25->g_scan64_0)[tid_7] + add_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_0 = step_0 << 1U;
        }
        if(_S108)
        {
            *(kernelContext_25->index_0+(c0_6 + i_4)) = carry_0 + (*kernelContext_25->g_scan64_0)[tid_7] - v_0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_1 = carry_0 + (*kernelContext_25->g_scan64_0)[255U];
        block_1 = block_1 + 256U;
        carry_0 = carry_1;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    m_2 = tid_7;
    for(;;)
    {
        if(m_2 < n_4)
        {
        }
        else
        {
            break;
        }
        uint _S110 = new_chunk_at_0(base_6 + m_2, kernelContext_25);
        uint _S111 = *(kernelContext_25->index_0+(c0_6 + *(kernelContext_25->plan_0+_S110)));
        uint _S112 = c0_6 + m_2;
        uint _S113 = _S112 + 1U;
        uint _S114 = kernelContext_25->old_index_0[_S113] - kernelContext_25->old_index_0[_S112];
        v_0 = kernelContext_25->old_index_0[_S112];
        step_0 = 0U;
        for(;;)
        {
            if(v_0 < (kernelContext_25->old_index_0)[_S113])
            {
            }
            else
            {
                break;
            }
            uint entry_1 = kernelContext_25->old_index_0[v_0];
            uint bond_1 = entry_1 >> 1U;
            if(bond_1 < b0_4)
            {
                _S101 = true;
            }
            else
            {
                _S101 = bond_1 >= b1_4;
            }
            if(_S101)
            {
                v_0 = v_0 + 1U;
                continue;
            }
            uint _S115 = bond_new_at_0(bond_1, kernelContext_25);
            uint device* _S116 = kernelContext_25->plan_0+_S115;
            uint nb_2 = *_S116;
            if((*_S116) >= cut_first_0)
            {
                v_0 = v_0 + 1U;
                continue;
            }
            *(kernelContext_25->index_0+(_S111 + step_0)) = 2U * nb_2 + (entry_1 & 1U);
            step_0 = step_0 + 1U;
            v_0 = v_0 + 1U;
        }
        add_0 = step_0;
        for(;;)
        {
            if(add_0 < _S114)
            {
            }
            else
            {
                break;
            }
            *(kernelContext_25->index_0+(_S111 + add_0)) = pad_0;
            add_0 = add_0 + 1U;
        }
        m_2 = m_2 + 256U;
    }
    return;
}

void inverse3_0(float3 r0_0, float3 r1_0, float3 r2_0, float3 thread* i0_0, float3 thread* i1_0, float3 thread* i2_0)
{
    float3 c0_7 = cross(r1_0, r2_0);
    float3 c1_2 = cross(r2_0, r0_0);
    float3 c2_0 = cross(r0_0, r1_0);
    float det_0 = dot(r0_0, c0_7);
    float inv_0;
    if(det_0 != 0.0f)
    {
        inv_0 = 1.0f / det_0;
    }
    else
    {
        inv_0 = 0.0f;
    }
    *i0_0 = float3(c0_7.x, c1_2.x, c2_0.x) * float3(inv_0) ;
    *i1_0 = float3(c0_7.y, c1_2.y, c2_0.y) * float3(inv_0) ;
    *i2_0 = float3(c0_7.z, c1_2.z, c2_0.z) * float3(inv_0) ;
    return;
}

struct Quat_0
{
    float w_9;
    float x_0;
    float y_0;
    float z_0;
};

Quat_0 quat_of_0(float4 q_1)
{
    thread Quat_0 r_2;
    (&r_2)->x_0 = q_1.x;
    (&r_2)->y_0 = q_1.y;
    (&r_2)->z_0 = q_1.z;
    (&r_2)->w_9 = q_1.w;
    return r_2;
}

float3 rotate_0(const Quat_0 thread* q_2, float3 v_1)
{
    float3 qv_0 = float3(q_2->x_0, q_2->y_0, q_2->z_0);
    float3 t_1 = cross(qv_0, v_1) * float3(2.0f) ;
    return v_1 + t_1 * float3(q_2->w_9)  + cross(qv_0, t_1);
}

float3 inverse_rotate_0(const Quat_0 thread* q_3, float3 v_2)
{
    thread Quat_0 c_2;
    (&c_2)->w_9 = q_3->w_9;
    (&c_2)->x_0 = - q_3->x_0;
    (&c_2)->y_0 = - q_3->y_0;
    (&c_2)->z_0 = - q_3->z_0;
    thread Quat_0 _S117 = c_2;
    float3 _S118 = rotate_0(&_S117, v_2);
    return _S118;
}

float4 quat_vec_0(const Quat_0 thread* q_4)
{
    return float4(q_4->x_0, q_4->y_0, q_4->z_0, q_4->w_9);
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

void lay_record_0(uint p_2, uint c0_8, uint base_7, uint total_2, uint slot_7, KernelContext_0 thread* kernelContext_26)
{
    uint _S119 = child_island_at_0(slot_7, kernelContext_26);
    uint _S120 = *(kernelContext_26->plan_0+_S119);
    uint _S121 = 8U * slot_7;
    float4 _S122 = float4(*(kernelContext_26->children_0+_S121)) ;
    uint _S123 = (as_type<uint>((_S122.y)));
    uint count_1 = (as_type<uint>((_S122.z)));
    bool anchored_0 = (as_type<uint>((_S122.w))) != 0U;
    float4 _S124 = float4(*(kernelContext_26->children_0+(_S121 + 1U))) ;
    float3 i0_1 = (float4(*(kernelContext_26->children_0+(_S121 + 2U))) ).xyz;
    float3 i1_1 = (float4(*(kernelContext_26->children_0+(_S121 + 3U))) ).xyz;
    float3 i2_1 = (float4(*(kernelContext_26->children_0+(_S121 + 4U))) ).xyz;
    float3 v_3 = (float4(*(kernelContext_26->children_0+(_S121 + 5U))) ).xyz;
    float3 w_10 = (float4(*(kernelContext_26->children_0+(_S121 + 6U))) ).xyz;
    float3 _S125 = float3(0.0f) ;
    uint first_new_0 = 4294967295U;
    bool scaled_0 = false;
    uint j_3 = 0U;
    float wmass_0 = 0.0f;
    float3 wmoment_0 = _S125;
    float su_mass_0 = 0.0f;
    float3 su_0 = _S125;
    for(;;)
    {
        if(j_3 < count_1)
        {
        }
        else
        {
            break;
        }
        uint m_3 = kernelContext_26->work_0[total_2 + base_7 + _S123 + j_3];
        uint _S126 = new_chunk_at_0(base_7 + m_3, kernelContext_26);
        uint _S127 = min(first_new_0, *(kernelContext_26->plan_0+_S126));
        uint _S128 = c0_8 + m_3;
        uint _S129 = chunk_word_0(_S128, 0U, kernelContext_26);
        float4 _S130 = float4(*(kernelContext_26->old_chunks_0+_S129)) ;
        uint _S131 = chunk_word_0(_S128, 7U, kernelContext_26);
        float mu_0 = (float4(*(kernelContext_26->old_chunks_0+_S131)) ).x;
        if(scaled_0)
        {
            scaled_0 = true;
        }
        else
        {
            scaled_0 = mu_0 != 1.0f;
        }
        float _S132 = _S130.w;
        float wmass_1 = wmass_0 + _S132 * mu_0;
        float3 _S133 = float3(_S132) ;
        float3 wmoment_1 = wmoment_0 + _S130.xyz * _S133 * float3(mu_0) ;
        float su_mass_1 = su_mass_0 + _S132;
        float3 su_1 = su_0 + (float4(*(kernelContext_26->old_state_0+4U * _S128)) ).xyz * _S133;
        uint _S134 = j_3 + 1U;
        first_new_0 = _S127;
        j_3 = _S134;
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
        wcom_0 = _S124.xyz;
    }
    float3 w0_0;
    float3 w1_0;
    float3 w2_0;
    if(scaled_0)
    {
        j_3 = 0U;
        w0_0 = _S125;
        w1_0 = _S125;
        w2_0 = _S125;
        for(;;)
        {
            if(j_3 < count_1)
            {
            }
            else
            {
                break;
            }
            uint _S135 = c0_8 + kernelContext_26->work_0[total_2 + base_7 + _S123 + j_3];
            uint _S136 = chunk_word_0(_S135, 0U, kernelContext_26);
            float4 _S137 = float4(*(kernelContext_26->old_chunks_0+_S136)) ;
            uint _S138 = chunk_word_0(_S135, 7U, kernelContext_26);
            float mu_1 = (float4(*(kernelContext_26->old_chunks_0+_S138)) ).x;
            float3 rr_0 = _S137.xyz - wcom_0;
            float dd_0 = dot(rr_0, rr_0);
            uint _S139 = chunk_word_0(_S135, 1U, kernelContext_26);
            float3 _S140 = float3(_S137.w) ;
            float3 _S141 = float3(mu_1) ;
            float3 w0_1 = w0_0 + ((float4(*(kernelContext_26->old_chunks_0+_S139)) ).xyz + (float3(dd_0, 0.0f, 0.0f) - float3(rr_0.x)  * rr_0) * _S140) * _S141;
            uint _S142 = chunk_word_0(_S135, 2U, kernelContext_26);
            float3 w1_1 = w1_0 + ((float4(*(kernelContext_26->old_chunks_0+_S142)) ).xyz + (float3(0.0f, dd_0, 0.0f) - float3(rr_0.y)  * rr_0) * _S140) * _S141;
            uint _S143 = chunk_word_0(_S135, 3U, kernelContext_26);
            float3 w2_1 = w2_0 + ((float4(*(kernelContext_26->old_chunks_0+_S143)) ).xyz + (float3(0.0f, 0.0f, dd_0) - float3(rr_0.z)  * rr_0) * _S140) * _S141;
            j_3 = j_3 + 1U;
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
    uint _S144 = 5U * p_2;
    Quat_0 q_5 = quat_of_0(float4(*(kernelContext_26->frames_0+_S144)) );
    float3 pos_0 = (float4(*(kernelContext_26->frames_0+(_S144 + 4U))) ).xyz;
    float3 pos_1;
    if(!anchored_0)
    {
        float3 _S145 = su_0 / float3(su_mass_0) ;
        thread Quat_0 _S146 = q_5;
        float3 _S147 = rotate_0(&_S146, _S145);
        pos_1 = pos_0 + _S147;
    }
    else
    {
        pos_1 = pos_0;
    }
    thread Quat_0 _S148 = q_5;
    float3 _S149 = inverse_rotate_0(&_S148, w_10);
    float3 _S150 = float3(dot(i0_1, _S149), dot(i1_1, _S149), dot(i2_1, _S149));
    thread Quat_0 _S151 = q_5;
    float3 _S152 = rotate_0(&_S151, _S150);
    if(anchored_0)
    {
        j_3 = 1U;
    }
    else
    {
        j_3 = 0U;
    }
    uint t_2;
    if((kernelContext_26->params_0->driven_0) != 0U)
    {
        t_2 = 2U;
    }
    else
    {
        t_2 = 0U;
    }
    uint flags_0 = (j_3 | t_2) | 16U;
    uint _S153 = child_bonds_at_0(slot_7, kernelContext_26);
    uint bfirst_0 = *(kernelContext_26->plan_0+_S153);
    uint _S154 = child_bonds_at_0(slot_7, kernelContext_26);
    uint bend_0 = *(kernelContext_26->plan_0+(_S154 + 1U));
    thread array<float4, int(25)> rec_0;
    t_2 = 0U;
    for(;;)
    {
        if(t_2 < 25U)
        {
        }
        else
        {
            break;
        }
        rec_0[t_2] = float4(0.0f) ;
        t_2 = t_2 + 1U;
    }
    uint _S155 = c0_8 + first_new_0;
    rec_0[int(0)] = (as_type<float4>((uint4(_S155, _S155 + count_1, bfirst_0, bend_0))));
    rec_0[int(1)] = (as_type<float4>((uint4(flags_0, 0U, 0U, kernelContext_26->params_0->substeps_0))));
    rec_0[int(2)] = _S124;
    rec_0[int(3)] = float4(i0_1, 0.0f);
    rec_0[int(4)] = float4(i1_1, 0.0f);
    rec_0[int(5)] = float4(i2_1, 0.0f);
    rec_0[int(6)] = float4(v0_0, 0.0f);
    rec_0[int(7)] = float4(v1_0, 0.0f);
    rec_0[int(8)] = float4(v2_0, 0.0f);
    if(scaled_0)
    {
    }
    else
    {
        wmass_0 = _S124.w;
    }
    rec_0[int(9)] = float4(wcom_0, wmass_0);
    rec_0[int(10)] = float4(x0_0, 0.0f);
    rec_0[int(11)] = float4(x1_0, 0.0f);
    rec_0[int(12)] = float4(x2_0, 0.0f);
    thread Quat_0 _S156 = q_5;
    float4 _S157 = quat_vec_0(&_S156);
    rec_0[int(13)] = _S157;
    rec_0[int(14)] = float4(pos_1, 0.0f);
    if(anchored_0)
    {
        wmoment_0 = _S125;
    }
    else
    {
        wmoment_0 = v_3;
    }
    rec_0[int(16)] = float4(wmoment_0, 0.0f);
    if(anchored_0)
    {
        wmoment_0 = _S125;
    }
    else
    {
        wmoment_0 = w_10;
    }
    rec_0[int(18)] = float4(wmoment_0, 0.0f);
    uint _S158 = child_id_at_0(slot_7, kernelContext_26);
    rec_0[int(19)] = (as_type<float4>((uint4(0U, 0U, 0U, *(kernelContext_26->plan_0+_S158)))));
    rec_0[int(21)] = float4(0.0f, 0.0f, 0.0f, float_bits_0(2139095040U));
    if(anchored_0)
    {
        wmoment_0 = _S125;
    }
    else
    {
        wmoment_0 = _S152;
    }
    rec_0[int(23)] = float4(wmoment_0, 0.0f);
    t_2 = 0U;
    for(;;)
    {
        if(t_2 < 25U)
        {
        }
        else
        {
            break;
        }
        uint _S159 = island_word_0(_S120, t_2, kernelContext_26);
        *(kernelContext_26->islands_0+_S159) = packed_float4(rec_0[t_2]) ;
        t_2 = t_2 + 1U;
    }
    return;
}

void lay_engine_0(uint slot_8, KernelContext_0 thread* kernelContext_27)
{
    uint _S160 = child_island_at_0(slot_8, kernelContext_27);
    uint island_2 = *(kernelContext_27->plan_0+_S160);
    bool anchored_1 = (as_type<uint>(((float4(*(kernelContext_27->children_0+8U * slot_8)) ).w))) != 0U;
    if((kernelContext_27->params_0->resident_0) != 0U)
    {
        uint4 _S161 = uint4(*(kernelContext_27->spawn_slots_0+slot_8)) ;
        uint _S162 = 2U * slot_8;
        float4 _S163 = float4(*(kernelContext_27->spawn_extras_0+(_S162 + 1U))) ;
        uint at_3 = kernelContext_27->params_0->engine_base_0 + 3U * island_2;
        uint _S164 = _S161.w;
        float3 body_range_0;
        if(_S164 != 0U)
        {
            uint _S165 = _S161.y;
            body_range_0 = (as_type<float3>((uint3(_S161.x, _S165, _S165 + _S161.z))));
        }
        else
        {
            body_range_0 = float3(float_bits_0(4294967295U), 0.0f, 0.0f);
        }
        bool _S166;
        if(anchored_1)
        {
            _S166 = true;
        }
        else
        {
            _S166 = _S164 == 0U;
        }
        if(_S166)
        {
            *(kernelContext_27->ctab_0+at_3) = packed_float4(float4(float_bits_0(4294967295U), body_range_0)) ;
            packed_float4 _S167 = packed_float4(float4(0.0f) ) ;
            *(kernelContext_27->ctab_0+(at_3 + 1U)) = _S167;
            *(kernelContext_27->ctab_0+(at_3 + 2U)) = _S167;
        }
        else
        {
            *(kernelContext_27->ctab_0+at_3) = packed_float4(float4((as_type<float>((_S161.x))), body_range_0)) ;
            *(kernelContext_27->ctab_0+(at_3 + 1U)) = packed_float4(float4(- _S163.x, - _S163.y, - _S163.z, _S163.w)) ;
            *(kernelContext_27->ctab_0+(at_3 + 2U)) = packed_float4(float4((float4(*(kernelContext_27->spawn_extras_0+_S162)) ).xyz, 0.0f)) ;
        }
    }
    *(kernelContext_27->gate_0+(kernelContext_27->params_0->gate_flag_base_0 + island_2)) = packed_float4(float4(0.0f) ) ;
    return;
}

[[kernel]] void layout_parent(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], uint device* control_2 [[buffer(24)]], uint device* plan_2 [[buffer(5)]], packed_uint4 device* parents_2 [[buffer(1)]], LayoutParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* old_islands_2 [[buffer(16)]], packed_float4 device* children_2 [[buffer(3)]], uint device* work_2 [[buffer(2)]], packed_float4 device* chunks_2 [[buffer(7)]], packed_float4 device* old_chunks_2 [[buffer(6)]], packed_float4 device* state_2 [[buffer(9)]], packed_float4 device* old_state_2 [[buffer(8)]], packed_float4 device* scratch_2 [[buffer(18)]], packed_float4 device* ctab_2 [[buffer(20)]], packed_float4 device* old_ctab_2 [[buffer(19)]], packed_float4 device* old_bonds_2 [[buffer(10)]], packed_float4 device* bonds_2 [[buffer(11)]], packed_float4 device* dyn_2 [[buffer(13)]], packed_float4 device* old_dyn_2 [[buffer(12)]], uint device* old_index_2 [[buffer(14)]], uint device* index_2 [[buffer(15)]], packed_float4 device* frames_2 [[buffer(4)]], packed_float4 device* islands_2 [[buffer(17)]], packed_uint4 device* spawn_slots_2 [[buffer(22)]], packed_float4 device* spawn_extras_2 [[buffer(23)]], packed_float4 device* gate_2 [[buffer(21)]], packed_float4 device* island_copy_2 [[buffer(26)]], packed_float4 device* old_copy_2 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_28;
    (&kernelContext_28)->control_0 = control_2;
    (&kernelContext_28)->plan_0 = plan_2;
    (&kernelContext_28)->parents_0 = parents_2;
    (&kernelContext_28)->params_0 = params_2;
    (&kernelContext_28)->old_islands_0 = old_islands_2;
    (&kernelContext_28)->children_0 = children_2;
    (&kernelContext_28)->work_0 = work_2;
    (&kernelContext_28)->chunks_0 = chunks_2;
    (&kernelContext_28)->old_chunks_0 = old_chunks_2;
    (&kernelContext_28)->state_0 = state_2;
    (&kernelContext_28)->old_state_0 = old_state_2;
    (&kernelContext_28)->scratch_0 = scratch_2;
    (&kernelContext_28)->ctab_0 = ctab_2;
    (&kernelContext_28)->old_ctab_0 = old_ctab_2;
    (&kernelContext_28)->old_bonds_0 = old_bonds_2;
    (&kernelContext_28)->bonds_0 = bonds_2;
    (&kernelContext_28)->dyn_0 = dyn_2;
    (&kernelContext_28)->old_dyn_0 = old_dyn_2;
    (&kernelContext_28)->old_index_0 = old_index_2;
    (&kernelContext_28)->index_0 = index_2;
    (&kernelContext_28)->frames_0 = frames_2;
    (&kernelContext_28)->islands_0 = islands_2;
    (&kernelContext_28)->spawn_slots_0 = spawn_slots_2;
    (&kernelContext_28)->spawn_extras_0 = spawn_extras_2;
    (&kernelContext_28)->gate_0 = gate_2;
    (&kernelContext_28)->island_copy_0 = island_copy_2;
    (&kernelContext_28)->old_copy_0 = old_copy_2;
    threadgroup array<uint, int(256)> g_class_2;
    (&kernelContext_28)->g_class_0 = &g_class_2;
    threadgroup uint g_cut_next_2;
    (&kernelContext_28)->g_cut_next_0 = &g_cut_next_2;
    threadgroup uint g_cut_first_2;
    (&kernelContext_28)->g_cut_first_0 = &g_cut_first_2;
    threadgroup array<uint, int(256)> g_scan64_2;
    (&kernelContext_28)->g_scan64_0 = &g_scan64_2;
    uint tid_8 = thread_1.x;
    uint p_3 = group_0.x;
    thread uint4 parent_6;
    thread uint c0_9;
    thread uint c1_3;
    thread uint b0_5;
    thread uint b1_5;
    bool _S168 = layout_start_0(p_3, tid_8, &parent_6, &c0_9, &c1_3, &b0_5, &b1_5, &kernelContext_28);
    if(!_S168)
    {
        return;
    }
    uint n_5 = c1_3 - c0_9;
    uint base_8 = parent_6.y;
    uint device* _S169 = (&kernelContext_28)->control_0+6U;
    uint total_3 = *_S169;
    lay_positions_0(parent_6, n_5, base_8, *_S169, tid_8, &kernelContext_28);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    lay_chunks_0(c0_9, n_5, base_8, tid_8, &kernelContext_28);
    if(((&kernelContext_28)->params_0->resident_0) != 0U)
    {
        lay_ctab_chunks_0(c0_9, n_5, base_8, tid_8, &kernelContext_28);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    lay_bond_ranks_0(parent_6, c0_9, b0_5, b1_5, base_8, tid_8, &kernelContext_28);
    lay_bonds_0(c0_9, b0_5, b1_5, base_8, tid_8, &kernelContext_28);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    lay_incidence_0(c0_9, c1_3, b0_5, b1_5, n_5, base_8, *(&kernelContext_28)->g_cut_first_0, tid_8, &kernelContext_28);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint k_3 = tid_8;
    for(;;)
    {
        if(k_3 < (parent_6.z))
        {
        }
        else
        {
            break;
        }
        uint _S170 = base_8 + k_3;
        lay_record_0(p_3, c0_9, base_8, total_3, _S170, &kernelContext_28);
        lay_engine_0(_S170, &kernelContext_28);
        k_3 = k_3 + 256U;
    }
    return;
}

[[kernel]] void layout_parent_positions(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], uint device* control_3 [[buffer(24)]], uint device* plan_3 [[buffer(5)]], packed_uint4 device* parents_3 [[buffer(1)]], LayoutParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* old_islands_3 [[buffer(16)]], packed_float4 device* children_3 [[buffer(3)]], uint device* work_3 [[buffer(2)]], packed_float4 device* chunks_3 [[buffer(7)]], packed_float4 device* old_chunks_3 [[buffer(6)]], packed_float4 device* state_3 [[buffer(9)]], packed_float4 device* old_state_3 [[buffer(8)]], packed_float4 device* scratch_3 [[buffer(18)]], packed_float4 device* ctab_3 [[buffer(20)]], packed_float4 device* old_ctab_3 [[buffer(19)]], packed_float4 device* old_bonds_3 [[buffer(10)]], packed_float4 device* bonds_3 [[buffer(11)]], packed_float4 device* dyn_3 [[buffer(13)]], packed_float4 device* old_dyn_3 [[buffer(12)]], uint device* old_index_3 [[buffer(14)]], uint device* index_3 [[buffer(15)]], packed_float4 device* frames_3 [[buffer(4)]], packed_float4 device* islands_3 [[buffer(17)]], packed_uint4 device* spawn_slots_3 [[buffer(22)]], packed_float4 device* spawn_extras_3 [[buffer(23)]], packed_float4 device* gate_3 [[buffer(21)]], packed_float4 device* island_copy_3 [[buffer(26)]], packed_float4 device* old_copy_3 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_29;
    (&kernelContext_29)->control_0 = control_3;
    (&kernelContext_29)->plan_0 = plan_3;
    (&kernelContext_29)->parents_0 = parents_3;
    (&kernelContext_29)->params_0 = params_3;
    (&kernelContext_29)->old_islands_0 = old_islands_3;
    (&kernelContext_29)->children_0 = children_3;
    (&kernelContext_29)->work_0 = work_3;
    (&kernelContext_29)->chunks_0 = chunks_3;
    (&kernelContext_29)->old_chunks_0 = old_chunks_3;
    (&kernelContext_29)->state_0 = state_3;
    (&kernelContext_29)->old_state_0 = old_state_3;
    (&kernelContext_29)->scratch_0 = scratch_3;
    (&kernelContext_29)->ctab_0 = ctab_3;
    (&kernelContext_29)->old_ctab_0 = old_ctab_3;
    (&kernelContext_29)->old_bonds_0 = old_bonds_3;
    (&kernelContext_29)->bonds_0 = bonds_3;
    (&kernelContext_29)->dyn_0 = dyn_3;
    (&kernelContext_29)->old_dyn_0 = old_dyn_3;
    (&kernelContext_29)->old_index_0 = old_index_3;
    (&kernelContext_29)->index_0 = index_3;
    (&kernelContext_29)->frames_0 = frames_3;
    (&kernelContext_29)->islands_0 = islands_3;
    (&kernelContext_29)->spawn_slots_0 = spawn_slots_3;
    (&kernelContext_29)->spawn_extras_0 = spawn_extras_3;
    (&kernelContext_29)->gate_0 = gate_3;
    (&kernelContext_29)->island_copy_0 = island_copy_3;
    (&kernelContext_29)->old_copy_0 = old_copy_3;
    threadgroup array<uint, int(256)> g_class_3;
    (&kernelContext_29)->g_class_0 = &g_class_3;
    threadgroup uint g_cut_next_3;
    (&kernelContext_29)->g_cut_next_0 = &g_cut_next_3;
    threadgroup uint g_cut_first_3;
    (&kernelContext_29)->g_cut_first_0 = &g_cut_first_3;
    threadgroup array<uint, int(256)> g_scan64_3;
    (&kernelContext_29)->g_scan64_0 = &g_scan64_3;
    uint _S171 = thread_2.x;
    thread uint4 parent_7;
    thread uint c0_10;
    thread uint c1_4;
    thread uint b0_6;
    thread uint b1_6;
    bool _S172 = layout_start_0(group_1.x, _S171, &parent_7, &c0_10, &c1_4, &b0_6, &b1_6, &kernelContext_29);
    if(!_S172)
    {
        return;
    }
    lay_positions_0(parent_7, c1_4 - c0_10, parent_7.y, *((&kernelContext_29)->control_0+6U), _S171, &kernelContext_29);
    return;
}

[[kernel]] void layout_parent_chunks(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_3 [[thread_position_in_threadgroup]], uint device* control_4 [[buffer(24)]], uint device* plan_4 [[buffer(5)]], packed_uint4 device* parents_4 [[buffer(1)]], LayoutParams_0 constant* params_4 [[buffer(0)]], packed_float4 device* old_islands_4 [[buffer(16)]], packed_float4 device* children_4 [[buffer(3)]], uint device* work_4 [[buffer(2)]], packed_float4 device* chunks_4 [[buffer(7)]], packed_float4 device* old_chunks_4 [[buffer(6)]], packed_float4 device* state_4 [[buffer(9)]], packed_float4 device* old_state_4 [[buffer(8)]], packed_float4 device* scratch_4 [[buffer(18)]], packed_float4 device* ctab_4 [[buffer(20)]], packed_float4 device* old_ctab_4 [[buffer(19)]], packed_float4 device* old_bonds_4 [[buffer(10)]], packed_float4 device* bonds_4 [[buffer(11)]], packed_float4 device* dyn_4 [[buffer(13)]], packed_float4 device* old_dyn_4 [[buffer(12)]], uint device* old_index_4 [[buffer(14)]], uint device* index_4 [[buffer(15)]], packed_float4 device* frames_4 [[buffer(4)]], packed_float4 device* islands_4 [[buffer(17)]], packed_uint4 device* spawn_slots_4 [[buffer(22)]], packed_float4 device* spawn_extras_4 [[buffer(23)]], packed_float4 device* gate_4 [[buffer(21)]], packed_float4 device* island_copy_4 [[buffer(26)]], packed_float4 device* old_copy_4 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_30;
    (&kernelContext_30)->control_0 = control_4;
    (&kernelContext_30)->plan_0 = plan_4;
    (&kernelContext_30)->parents_0 = parents_4;
    (&kernelContext_30)->params_0 = params_4;
    (&kernelContext_30)->old_islands_0 = old_islands_4;
    (&kernelContext_30)->children_0 = children_4;
    (&kernelContext_30)->work_0 = work_4;
    (&kernelContext_30)->chunks_0 = chunks_4;
    (&kernelContext_30)->old_chunks_0 = old_chunks_4;
    (&kernelContext_30)->state_0 = state_4;
    (&kernelContext_30)->old_state_0 = old_state_4;
    (&kernelContext_30)->scratch_0 = scratch_4;
    (&kernelContext_30)->ctab_0 = ctab_4;
    (&kernelContext_30)->old_ctab_0 = old_ctab_4;
    (&kernelContext_30)->old_bonds_0 = old_bonds_4;
    (&kernelContext_30)->bonds_0 = bonds_4;
    (&kernelContext_30)->dyn_0 = dyn_4;
    (&kernelContext_30)->old_dyn_0 = old_dyn_4;
    (&kernelContext_30)->old_index_0 = old_index_4;
    (&kernelContext_30)->index_0 = index_4;
    (&kernelContext_30)->frames_0 = frames_4;
    (&kernelContext_30)->islands_0 = islands_4;
    (&kernelContext_30)->spawn_slots_0 = spawn_slots_4;
    (&kernelContext_30)->spawn_extras_0 = spawn_extras_4;
    (&kernelContext_30)->gate_0 = gate_4;
    (&kernelContext_30)->island_copy_0 = island_copy_4;
    (&kernelContext_30)->old_copy_0 = old_copy_4;
    threadgroup array<uint, int(256)> g_class_4;
    (&kernelContext_30)->g_class_0 = &g_class_4;
    threadgroup uint g_cut_next_4;
    (&kernelContext_30)->g_cut_next_0 = &g_cut_next_4;
    threadgroup uint g_cut_first_4;
    (&kernelContext_30)->g_cut_first_0 = &g_cut_first_4;
    threadgroup array<uint, int(256)> g_scan64_4;
    (&kernelContext_30)->g_scan64_0 = &g_scan64_4;
    uint _S173 = thread_3.x;
    thread uint4 parent_8;
    thread uint c0_11;
    thread uint c1_5;
    thread uint b0_7;
    thread uint b1_7;
    bool _S174 = layout_start_0(group_2.x, _S173, &parent_8, &c0_11, &c1_5, &b0_7, &b1_7, &kernelContext_30);
    if(!_S174)
    {
        return;
    }
    lay_chunks_0(c0_11, c1_5 - c0_11, parent_8.y, _S173, &kernelContext_30);
    return;
}

[[kernel]] void layout_parent_ctab(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], uint device* control_5 [[buffer(24)]], uint device* plan_5 [[buffer(5)]], packed_uint4 device* parents_5 [[buffer(1)]], LayoutParams_0 constant* params_5 [[buffer(0)]], packed_float4 device* old_islands_5 [[buffer(16)]], packed_float4 device* children_5 [[buffer(3)]], uint device* work_5 [[buffer(2)]], packed_float4 device* chunks_5 [[buffer(7)]], packed_float4 device* old_chunks_5 [[buffer(6)]], packed_float4 device* state_5 [[buffer(9)]], packed_float4 device* old_state_5 [[buffer(8)]], packed_float4 device* scratch_5 [[buffer(18)]], packed_float4 device* ctab_5 [[buffer(20)]], packed_float4 device* old_ctab_5 [[buffer(19)]], packed_float4 device* old_bonds_5 [[buffer(10)]], packed_float4 device* bonds_5 [[buffer(11)]], packed_float4 device* dyn_5 [[buffer(13)]], packed_float4 device* old_dyn_5 [[buffer(12)]], uint device* old_index_5 [[buffer(14)]], uint device* index_5 [[buffer(15)]], packed_float4 device* frames_5 [[buffer(4)]], packed_float4 device* islands_5 [[buffer(17)]], packed_uint4 device* spawn_slots_5 [[buffer(22)]], packed_float4 device* spawn_extras_5 [[buffer(23)]], packed_float4 device* gate_5 [[buffer(21)]], packed_float4 device* island_copy_5 [[buffer(26)]], packed_float4 device* old_copy_5 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_31;
    (&kernelContext_31)->control_0 = control_5;
    (&kernelContext_31)->plan_0 = plan_5;
    (&kernelContext_31)->parents_0 = parents_5;
    (&kernelContext_31)->params_0 = params_5;
    (&kernelContext_31)->old_islands_0 = old_islands_5;
    (&kernelContext_31)->children_0 = children_5;
    (&kernelContext_31)->work_0 = work_5;
    (&kernelContext_31)->chunks_0 = chunks_5;
    (&kernelContext_31)->old_chunks_0 = old_chunks_5;
    (&kernelContext_31)->state_0 = state_5;
    (&kernelContext_31)->old_state_0 = old_state_5;
    (&kernelContext_31)->scratch_0 = scratch_5;
    (&kernelContext_31)->ctab_0 = ctab_5;
    (&kernelContext_31)->old_ctab_0 = old_ctab_5;
    (&kernelContext_31)->old_bonds_0 = old_bonds_5;
    (&kernelContext_31)->bonds_0 = bonds_5;
    (&kernelContext_31)->dyn_0 = dyn_5;
    (&kernelContext_31)->old_dyn_0 = old_dyn_5;
    (&kernelContext_31)->old_index_0 = old_index_5;
    (&kernelContext_31)->index_0 = index_5;
    (&kernelContext_31)->frames_0 = frames_5;
    (&kernelContext_31)->islands_0 = islands_5;
    (&kernelContext_31)->spawn_slots_0 = spawn_slots_5;
    (&kernelContext_31)->spawn_extras_0 = spawn_extras_5;
    (&kernelContext_31)->gate_0 = gate_5;
    (&kernelContext_31)->island_copy_0 = island_copy_5;
    (&kernelContext_31)->old_copy_0 = old_copy_5;
    threadgroup array<uint, int(256)> g_class_5;
    (&kernelContext_31)->g_class_0 = &g_class_5;
    threadgroup uint g_cut_next_5;
    (&kernelContext_31)->g_cut_next_0 = &g_cut_next_5;
    threadgroup uint g_cut_first_5;
    (&kernelContext_31)->g_cut_first_0 = &g_cut_first_5;
    threadgroup array<uint, int(256)> g_scan64_5;
    (&kernelContext_31)->g_scan64_0 = &g_scan64_5;
    uint _S175 = thread_4.x;
    thread uint4 parent_9;
    thread uint c0_12;
    thread uint c1_6;
    thread uint b0_8;
    thread uint b1_8;
    bool _S176 = layout_start_0(group_3.x, _S175, &parent_9, &c0_12, &c1_6, &b0_8, &b1_8, &kernelContext_31);
    bool _S177;
    if(!_S176)
    {
        _S177 = true;
    }
    else
    {
        _S177 = ((&kernelContext_31)->params_0->resident_0) == 0U;
    }
    if(_S177)
    {
        return;
    }
    lay_ctab_chunks_0(c0_12, c1_6 - c0_12, parent_9.y, _S175, &kernelContext_31);
    return;
}

[[kernel]] void layout_parent_bond_ranks(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], uint device* control_6 [[buffer(24)]], uint device* plan_6 [[buffer(5)]], packed_uint4 device* parents_6 [[buffer(1)]], LayoutParams_0 constant* params_6 [[buffer(0)]], packed_float4 device* old_islands_6 [[buffer(16)]], packed_float4 device* children_6 [[buffer(3)]], uint device* work_6 [[buffer(2)]], packed_float4 device* chunks_6 [[buffer(7)]], packed_float4 device* old_chunks_6 [[buffer(6)]], packed_float4 device* state_6 [[buffer(9)]], packed_float4 device* old_state_6 [[buffer(8)]], packed_float4 device* scratch_6 [[buffer(18)]], packed_float4 device* ctab_6 [[buffer(20)]], packed_float4 device* old_ctab_6 [[buffer(19)]], packed_float4 device* old_bonds_6 [[buffer(10)]], packed_float4 device* bonds_6 [[buffer(11)]], packed_float4 device* dyn_6 [[buffer(13)]], packed_float4 device* old_dyn_6 [[buffer(12)]], uint device* old_index_6 [[buffer(14)]], uint device* index_6 [[buffer(15)]], packed_float4 device* frames_6 [[buffer(4)]], packed_float4 device* islands_6 [[buffer(17)]], packed_uint4 device* spawn_slots_6 [[buffer(22)]], packed_float4 device* spawn_extras_6 [[buffer(23)]], packed_float4 device* gate_6 [[buffer(21)]], packed_float4 device* island_copy_6 [[buffer(26)]], packed_float4 device* old_copy_6 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_32;
    (&kernelContext_32)->control_0 = control_6;
    (&kernelContext_32)->plan_0 = plan_6;
    (&kernelContext_32)->parents_0 = parents_6;
    (&kernelContext_32)->params_0 = params_6;
    (&kernelContext_32)->old_islands_0 = old_islands_6;
    (&kernelContext_32)->children_0 = children_6;
    (&kernelContext_32)->work_0 = work_6;
    (&kernelContext_32)->chunks_0 = chunks_6;
    (&kernelContext_32)->old_chunks_0 = old_chunks_6;
    (&kernelContext_32)->state_0 = state_6;
    (&kernelContext_32)->old_state_0 = old_state_6;
    (&kernelContext_32)->scratch_0 = scratch_6;
    (&kernelContext_32)->ctab_0 = ctab_6;
    (&kernelContext_32)->old_ctab_0 = old_ctab_6;
    (&kernelContext_32)->old_bonds_0 = old_bonds_6;
    (&kernelContext_32)->bonds_0 = bonds_6;
    (&kernelContext_32)->dyn_0 = dyn_6;
    (&kernelContext_32)->old_dyn_0 = old_dyn_6;
    (&kernelContext_32)->old_index_0 = old_index_6;
    (&kernelContext_32)->index_0 = index_6;
    (&kernelContext_32)->frames_0 = frames_6;
    (&kernelContext_32)->islands_0 = islands_6;
    (&kernelContext_32)->spawn_slots_0 = spawn_slots_6;
    (&kernelContext_32)->spawn_extras_0 = spawn_extras_6;
    (&kernelContext_32)->gate_0 = gate_6;
    (&kernelContext_32)->island_copy_0 = island_copy_6;
    (&kernelContext_32)->old_copy_0 = old_copy_6;
    threadgroup array<uint, int(256)> g_class_6;
    (&kernelContext_32)->g_class_0 = &g_class_6;
    threadgroup uint g_cut_next_6;
    (&kernelContext_32)->g_cut_next_0 = &g_cut_next_6;
    threadgroup uint g_cut_first_6;
    (&kernelContext_32)->g_cut_first_0 = &g_cut_first_6;
    threadgroup array<uint, int(256)> g_scan64_6;
    (&kernelContext_32)->g_scan64_0 = &g_scan64_6;
    uint _S178 = thread_5.x;
    thread uint4 parent_10;
    thread uint c0_13;
    thread uint c1_7;
    thread uint b0_9;
    thread uint b1_9;
    bool _S179 = layout_start_0(group_4.x, _S178, &parent_10, &c0_13, &c1_7, &b0_9, &b1_9, &kernelContext_32);
    if(!_S179)
    {
        return;
    }
    lay_bond_ranks_0(parent_10, c0_13, b0_9, b1_9, parent_10.y, _S178, &kernelContext_32);
    return;
}

[[kernel]] void layout_parent_bonds(uint3 group_5 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], uint device* control_7 [[buffer(24)]], uint device* plan_7 [[buffer(5)]], packed_uint4 device* parents_7 [[buffer(1)]], LayoutParams_0 constant* params_7 [[buffer(0)]], packed_float4 device* old_islands_7 [[buffer(16)]], packed_float4 device* children_7 [[buffer(3)]], uint device* work_7 [[buffer(2)]], packed_float4 device* chunks_7 [[buffer(7)]], packed_float4 device* old_chunks_7 [[buffer(6)]], packed_float4 device* state_7 [[buffer(9)]], packed_float4 device* old_state_7 [[buffer(8)]], packed_float4 device* scratch_7 [[buffer(18)]], packed_float4 device* ctab_7 [[buffer(20)]], packed_float4 device* old_ctab_7 [[buffer(19)]], packed_float4 device* old_bonds_7 [[buffer(10)]], packed_float4 device* bonds_7 [[buffer(11)]], packed_float4 device* dyn_7 [[buffer(13)]], packed_float4 device* old_dyn_7 [[buffer(12)]], uint device* old_index_7 [[buffer(14)]], uint device* index_7 [[buffer(15)]], packed_float4 device* frames_7 [[buffer(4)]], packed_float4 device* islands_7 [[buffer(17)]], packed_uint4 device* spawn_slots_7 [[buffer(22)]], packed_float4 device* spawn_extras_7 [[buffer(23)]], packed_float4 device* gate_7 [[buffer(21)]], packed_float4 device* island_copy_7 [[buffer(26)]], packed_float4 device* old_copy_7 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_33;
    (&kernelContext_33)->control_0 = control_7;
    (&kernelContext_33)->plan_0 = plan_7;
    (&kernelContext_33)->parents_0 = parents_7;
    (&kernelContext_33)->params_0 = params_7;
    (&kernelContext_33)->old_islands_0 = old_islands_7;
    (&kernelContext_33)->children_0 = children_7;
    (&kernelContext_33)->work_0 = work_7;
    (&kernelContext_33)->chunks_0 = chunks_7;
    (&kernelContext_33)->old_chunks_0 = old_chunks_7;
    (&kernelContext_33)->state_0 = state_7;
    (&kernelContext_33)->old_state_0 = old_state_7;
    (&kernelContext_33)->scratch_0 = scratch_7;
    (&kernelContext_33)->ctab_0 = ctab_7;
    (&kernelContext_33)->old_ctab_0 = old_ctab_7;
    (&kernelContext_33)->old_bonds_0 = old_bonds_7;
    (&kernelContext_33)->bonds_0 = bonds_7;
    (&kernelContext_33)->dyn_0 = dyn_7;
    (&kernelContext_33)->old_dyn_0 = old_dyn_7;
    (&kernelContext_33)->old_index_0 = old_index_7;
    (&kernelContext_33)->index_0 = index_7;
    (&kernelContext_33)->frames_0 = frames_7;
    (&kernelContext_33)->islands_0 = islands_7;
    (&kernelContext_33)->spawn_slots_0 = spawn_slots_7;
    (&kernelContext_33)->spawn_extras_0 = spawn_extras_7;
    (&kernelContext_33)->gate_0 = gate_7;
    (&kernelContext_33)->island_copy_0 = island_copy_7;
    (&kernelContext_33)->old_copy_0 = old_copy_7;
    threadgroup array<uint, int(256)> g_class_7;
    (&kernelContext_33)->g_class_0 = &g_class_7;
    threadgroup uint g_cut_next_7;
    (&kernelContext_33)->g_cut_next_0 = &g_cut_next_7;
    threadgroup uint g_cut_first_7;
    (&kernelContext_33)->g_cut_first_0 = &g_cut_first_7;
    threadgroup array<uint, int(256)> g_scan64_7;
    (&kernelContext_33)->g_scan64_0 = &g_scan64_7;
    uint _S180 = thread_6.x;
    thread uint4 parent_11;
    thread uint c0_14;
    thread uint c1_8;
    thread uint b0_10;
    thread uint b1_10;
    bool _S181 = layout_start_0(group_5.x, _S180, &parent_11, &c0_14, &c1_8, &b0_10, &b1_10, &kernelContext_33);
    if(!_S181)
    {
        return;
    }
    lay_bonds_0(c0_14, b0_10, b1_10, parent_11.y, _S180, &kernelContext_33);
    return;
}

[[kernel]] void layout_parent_incidence(uint3 group_6 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], uint device* control_8 [[buffer(24)]], uint device* plan_8 [[buffer(5)]], packed_uint4 device* parents_8 [[buffer(1)]], LayoutParams_0 constant* params_8 [[buffer(0)]], packed_float4 device* old_islands_8 [[buffer(16)]], packed_float4 device* children_8 [[buffer(3)]], uint device* work_8 [[buffer(2)]], packed_float4 device* chunks_8 [[buffer(7)]], packed_float4 device* old_chunks_8 [[buffer(6)]], packed_float4 device* state_8 [[buffer(9)]], packed_float4 device* old_state_8 [[buffer(8)]], packed_float4 device* scratch_8 [[buffer(18)]], packed_float4 device* ctab_8 [[buffer(20)]], packed_float4 device* old_ctab_8 [[buffer(19)]], packed_float4 device* old_bonds_8 [[buffer(10)]], packed_float4 device* bonds_8 [[buffer(11)]], packed_float4 device* dyn_8 [[buffer(13)]], packed_float4 device* old_dyn_8 [[buffer(12)]], uint device* old_index_8 [[buffer(14)]], uint device* index_8 [[buffer(15)]], packed_float4 device* frames_8 [[buffer(4)]], packed_float4 device* islands_8 [[buffer(17)]], packed_uint4 device* spawn_slots_8 [[buffer(22)]], packed_float4 device* spawn_extras_8 [[buffer(23)]], packed_float4 device* gate_8 [[buffer(21)]], packed_float4 device* island_copy_8 [[buffer(26)]], packed_float4 device* old_copy_8 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_34;
    (&kernelContext_34)->control_0 = control_8;
    (&kernelContext_34)->plan_0 = plan_8;
    (&kernelContext_34)->parents_0 = parents_8;
    (&kernelContext_34)->params_0 = params_8;
    (&kernelContext_34)->old_islands_0 = old_islands_8;
    (&kernelContext_34)->children_0 = children_8;
    (&kernelContext_34)->work_0 = work_8;
    (&kernelContext_34)->chunks_0 = chunks_8;
    (&kernelContext_34)->old_chunks_0 = old_chunks_8;
    (&kernelContext_34)->state_0 = state_8;
    (&kernelContext_34)->old_state_0 = old_state_8;
    (&kernelContext_34)->scratch_0 = scratch_8;
    (&kernelContext_34)->ctab_0 = ctab_8;
    (&kernelContext_34)->old_ctab_0 = old_ctab_8;
    (&kernelContext_34)->old_bonds_0 = old_bonds_8;
    (&kernelContext_34)->bonds_0 = bonds_8;
    (&kernelContext_34)->dyn_0 = dyn_8;
    (&kernelContext_34)->old_dyn_0 = old_dyn_8;
    (&kernelContext_34)->old_index_0 = old_index_8;
    (&kernelContext_34)->index_0 = index_8;
    (&kernelContext_34)->frames_0 = frames_8;
    (&kernelContext_34)->islands_0 = islands_8;
    (&kernelContext_34)->spawn_slots_0 = spawn_slots_8;
    (&kernelContext_34)->spawn_extras_0 = spawn_extras_8;
    (&kernelContext_34)->gate_0 = gate_8;
    (&kernelContext_34)->island_copy_0 = island_copy_8;
    (&kernelContext_34)->old_copy_0 = old_copy_8;
    threadgroup array<uint, int(256)> g_class_8;
    (&kernelContext_34)->g_class_0 = &g_class_8;
    threadgroup uint g_cut_next_8;
    (&kernelContext_34)->g_cut_next_0 = &g_cut_next_8;
    threadgroup uint g_cut_first_8;
    (&kernelContext_34)->g_cut_first_0 = &g_cut_first_8;
    threadgroup array<uint, int(256)> g_scan64_8;
    (&kernelContext_34)->g_scan64_0 = &g_scan64_8;
    uint _S182 = thread_7.x;
    thread uint4 parent_12;
    thread uint c0_15;
    thread uint c1_9;
    thread uint b0_11;
    thread uint b1_11;
    bool _S183 = layout_start_0(group_6.x, _S182, &parent_12, &c0_15, &c1_9, &b0_11, &b1_11, &kernelContext_34);
    if(!_S183)
    {
        return;
    }
    uint _S184 = parent_12.y;
    uint _S185 = child_by_island_0(parent_12, parent_12.z - 1U, &kernelContext_34);
    uint _S186 = child_bonds_at_0(_S184 + _S185, &kernelContext_34);
    lay_incidence_0(c0_15, c1_9, b0_11, b1_11, c1_9 - c0_15, parent_12.y, *((&kernelContext_34)->plan_0+(_S186 + 1U)), _S182, &kernelContext_34);
    return;
}

[[kernel]] void layout_parent_records(uint3 group_7 [[threadgroup_position_in_grid]], uint3 thread_8 [[thread_position_in_threadgroup]], uint device* control_9 [[buffer(24)]], uint device* plan_9 [[buffer(5)]], packed_uint4 device* parents_9 [[buffer(1)]], LayoutParams_0 constant* params_9 [[buffer(0)]], packed_float4 device* old_islands_9 [[buffer(16)]], packed_float4 device* children_9 [[buffer(3)]], uint device* work_9 [[buffer(2)]], packed_float4 device* chunks_9 [[buffer(7)]], packed_float4 device* old_chunks_9 [[buffer(6)]], packed_float4 device* state_9 [[buffer(9)]], packed_float4 device* old_state_9 [[buffer(8)]], packed_float4 device* scratch_9 [[buffer(18)]], packed_float4 device* ctab_9 [[buffer(20)]], packed_float4 device* old_ctab_9 [[buffer(19)]], packed_float4 device* old_bonds_9 [[buffer(10)]], packed_float4 device* bonds_9 [[buffer(11)]], packed_float4 device* dyn_9 [[buffer(13)]], packed_float4 device* old_dyn_9 [[buffer(12)]], uint device* old_index_9 [[buffer(14)]], uint device* index_9 [[buffer(15)]], packed_float4 device* frames_9 [[buffer(4)]], packed_float4 device* islands_9 [[buffer(17)]], packed_uint4 device* spawn_slots_9 [[buffer(22)]], packed_float4 device* spawn_extras_9 [[buffer(23)]], packed_float4 device* gate_9 [[buffer(21)]], packed_float4 device* island_copy_9 [[buffer(26)]], packed_float4 device* old_copy_9 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_35;
    (&kernelContext_35)->control_0 = control_9;
    (&kernelContext_35)->plan_0 = plan_9;
    (&kernelContext_35)->parents_0 = parents_9;
    (&kernelContext_35)->params_0 = params_9;
    (&kernelContext_35)->old_islands_0 = old_islands_9;
    (&kernelContext_35)->children_0 = children_9;
    (&kernelContext_35)->work_0 = work_9;
    (&kernelContext_35)->chunks_0 = chunks_9;
    (&kernelContext_35)->old_chunks_0 = old_chunks_9;
    (&kernelContext_35)->state_0 = state_9;
    (&kernelContext_35)->old_state_0 = old_state_9;
    (&kernelContext_35)->scratch_0 = scratch_9;
    (&kernelContext_35)->ctab_0 = ctab_9;
    (&kernelContext_35)->old_ctab_0 = old_ctab_9;
    (&kernelContext_35)->old_bonds_0 = old_bonds_9;
    (&kernelContext_35)->bonds_0 = bonds_9;
    (&kernelContext_35)->dyn_0 = dyn_9;
    (&kernelContext_35)->old_dyn_0 = old_dyn_9;
    (&kernelContext_35)->old_index_0 = old_index_9;
    (&kernelContext_35)->index_0 = index_9;
    (&kernelContext_35)->frames_0 = frames_9;
    (&kernelContext_35)->islands_0 = islands_9;
    (&kernelContext_35)->spawn_slots_0 = spawn_slots_9;
    (&kernelContext_35)->spawn_extras_0 = spawn_extras_9;
    (&kernelContext_35)->gate_0 = gate_9;
    (&kernelContext_35)->island_copy_0 = island_copy_9;
    (&kernelContext_35)->old_copy_0 = old_copy_9;
    threadgroup array<uint, int(256)> g_class_9;
    (&kernelContext_35)->g_class_0 = &g_class_9;
    threadgroup uint g_cut_next_9;
    (&kernelContext_35)->g_cut_next_0 = &g_cut_next_9;
    threadgroup uint g_cut_first_9;
    (&kernelContext_35)->g_cut_first_0 = &g_cut_first_9;
    threadgroup array<uint, int(256)> g_scan64_9;
    (&kernelContext_35)->g_scan64_0 = &g_scan64_9;
    uint _S187 = group_7.x;
    uint _S188 = thread_8.x;
    thread uint4 parent_13;
    thread uint c0_16;
    thread uint c1_10;
    thread uint b0_12;
    thread uint b1_12;
    bool _S189 = layout_start_0(_S187, _S188, &parent_13, &c0_16, &c1_10, &b0_12, &b1_12, &kernelContext_35);
    if(!_S189)
    {
        return;
    }
    uint k_4 = _S188;
    for(;;)
    {
        if(k_4 < (parent_13.z))
        {
        }
        else
        {
            break;
        }
        lay_record_0(_S187, c0_16, parent_13.y, *((&kernelContext_35)->control_0+6U), parent_13.y + k_4, &kernelContext_35);
        k_4 = k_4 + 256U;
    }
    return;
}

[[kernel]] void layout_parent_engine(uint3 group_8 [[threadgroup_position_in_grid]], uint3 thread_9 [[thread_position_in_threadgroup]], uint device* control_10 [[buffer(24)]], uint device* plan_10 [[buffer(5)]], packed_uint4 device* parents_10 [[buffer(1)]], LayoutParams_0 constant* params_10 [[buffer(0)]], packed_float4 device* old_islands_10 [[buffer(16)]], packed_float4 device* children_10 [[buffer(3)]], uint device* work_10 [[buffer(2)]], packed_float4 device* chunks_10 [[buffer(7)]], packed_float4 device* old_chunks_10 [[buffer(6)]], packed_float4 device* state_10 [[buffer(9)]], packed_float4 device* old_state_10 [[buffer(8)]], packed_float4 device* scratch_10 [[buffer(18)]], packed_float4 device* ctab_10 [[buffer(20)]], packed_float4 device* old_ctab_10 [[buffer(19)]], packed_float4 device* old_bonds_10 [[buffer(10)]], packed_float4 device* bonds_10 [[buffer(11)]], packed_float4 device* dyn_10 [[buffer(13)]], packed_float4 device* old_dyn_10 [[buffer(12)]], uint device* old_index_10 [[buffer(14)]], uint device* index_10 [[buffer(15)]], packed_float4 device* frames_10 [[buffer(4)]], packed_float4 device* islands_10 [[buffer(17)]], packed_uint4 device* spawn_slots_10 [[buffer(22)]], packed_float4 device* spawn_extras_10 [[buffer(23)]], packed_float4 device* gate_10 [[buffer(21)]], packed_float4 device* island_copy_10 [[buffer(26)]], packed_float4 device* old_copy_10 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_36;
    (&kernelContext_36)->control_0 = control_10;
    (&kernelContext_36)->plan_0 = plan_10;
    (&kernelContext_36)->parents_0 = parents_10;
    (&kernelContext_36)->params_0 = params_10;
    (&kernelContext_36)->old_islands_0 = old_islands_10;
    (&kernelContext_36)->children_0 = children_10;
    (&kernelContext_36)->work_0 = work_10;
    (&kernelContext_36)->chunks_0 = chunks_10;
    (&kernelContext_36)->old_chunks_0 = old_chunks_10;
    (&kernelContext_36)->state_0 = state_10;
    (&kernelContext_36)->old_state_0 = old_state_10;
    (&kernelContext_36)->scratch_0 = scratch_10;
    (&kernelContext_36)->ctab_0 = ctab_10;
    (&kernelContext_36)->old_ctab_0 = old_ctab_10;
    (&kernelContext_36)->old_bonds_0 = old_bonds_10;
    (&kernelContext_36)->bonds_0 = bonds_10;
    (&kernelContext_36)->dyn_0 = dyn_10;
    (&kernelContext_36)->old_dyn_0 = old_dyn_10;
    (&kernelContext_36)->old_index_0 = old_index_10;
    (&kernelContext_36)->index_0 = index_10;
    (&kernelContext_36)->frames_0 = frames_10;
    (&kernelContext_36)->islands_0 = islands_10;
    (&kernelContext_36)->spawn_slots_0 = spawn_slots_10;
    (&kernelContext_36)->spawn_extras_0 = spawn_extras_10;
    (&kernelContext_36)->gate_0 = gate_10;
    (&kernelContext_36)->island_copy_0 = island_copy_10;
    (&kernelContext_36)->old_copy_0 = old_copy_10;
    threadgroup array<uint, int(256)> g_class_10;
    (&kernelContext_36)->g_class_0 = &g_class_10;
    threadgroup uint g_cut_next_10;
    (&kernelContext_36)->g_cut_next_0 = &g_cut_next_10;
    threadgroup uint g_cut_first_10;
    (&kernelContext_36)->g_cut_first_0 = &g_cut_first_10;
    threadgroup array<uint, int(256)> g_scan64_10;
    (&kernelContext_36)->g_scan64_0 = &g_scan64_10;
    uint _S190 = thread_9.x;
    thread uint4 parent_14;
    thread uint c0_17;
    thread uint c1_11;
    thread uint b0_13;
    thread uint b1_13;
    bool _S191 = layout_start_0(group_8.x, _S190, &parent_14, &c0_17, &c1_11, &b0_13, &b1_13, &kernelContext_36);
    if(!_S191)
    {
        return;
    }
    uint k_5 = _S190;
    for(;;)
    {
        if(k_5 < (parent_14.z))
        {
        }
        else
        {
            break;
        }
        lay_engine_0(parent_14.y + k_5, &kernelContext_36);
        k_5 = k_5 + 256U;
    }
    return;
}

[[kernel]] void layout_moved(uint3 id_1 [[thread_position_in_grid]], uint device* control_11 [[buffer(24)]], uint device* plan_11 [[buffer(5)]], packed_uint4 device* parents_11 [[buffer(1)]], LayoutParams_0 constant* params_11 [[buffer(0)]], packed_float4 device* old_islands_11 [[buffer(16)]], packed_float4 device* children_11 [[buffer(3)]], uint device* work_11 [[buffer(2)]], packed_float4 device* chunks_11 [[buffer(7)]], packed_float4 device* old_chunks_11 [[buffer(6)]], packed_float4 device* state_11 [[buffer(9)]], packed_float4 device* old_state_11 [[buffer(8)]], packed_float4 device* scratch_11 [[buffer(18)]], packed_float4 device* ctab_11 [[buffer(20)]], packed_float4 device* old_ctab_11 [[buffer(19)]], packed_float4 device* old_bonds_11 [[buffer(10)]], packed_float4 device* bonds_11 [[buffer(11)]], packed_float4 device* dyn_11 [[buffer(13)]], packed_float4 device* old_dyn_11 [[buffer(12)]], uint device* old_index_11 [[buffer(14)]], uint device* index_11 [[buffer(15)]], packed_float4 device* frames_11 [[buffer(4)]], packed_float4 device* islands_11 [[buffer(17)]], packed_uint4 device* spawn_slots_11 [[buffer(22)]], packed_float4 device* spawn_extras_11 [[buffer(23)]], packed_float4 device* gate_11 [[buffer(21)]], packed_float4 device* island_copy_11 [[buffer(26)]], packed_float4 device* old_copy_11 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_37;
    (&kernelContext_37)->control_0 = control_11;
    (&kernelContext_37)->plan_0 = plan_11;
    (&kernelContext_37)->parents_0 = parents_11;
    (&kernelContext_37)->params_0 = params_11;
    (&kernelContext_37)->old_islands_0 = old_islands_11;
    (&kernelContext_37)->children_0 = children_11;
    (&kernelContext_37)->work_0 = work_11;
    (&kernelContext_37)->chunks_0 = chunks_11;
    (&kernelContext_37)->old_chunks_0 = old_chunks_11;
    (&kernelContext_37)->state_0 = state_11;
    (&kernelContext_37)->old_state_0 = old_state_11;
    (&kernelContext_37)->scratch_0 = scratch_11;
    (&kernelContext_37)->ctab_0 = ctab_11;
    (&kernelContext_37)->old_ctab_0 = old_ctab_11;
    (&kernelContext_37)->old_bonds_0 = old_bonds_11;
    (&kernelContext_37)->bonds_0 = bonds_11;
    (&kernelContext_37)->dyn_0 = dyn_11;
    (&kernelContext_37)->old_dyn_0 = old_dyn_11;
    (&kernelContext_37)->old_index_0 = old_index_11;
    (&kernelContext_37)->index_0 = index_11;
    (&kernelContext_37)->frames_0 = frames_11;
    (&kernelContext_37)->islands_0 = islands_11;
    (&kernelContext_37)->spawn_slots_0 = spawn_slots_11;
    (&kernelContext_37)->spawn_extras_0 = spawn_extras_11;
    (&kernelContext_37)->gate_0 = gate_11;
    (&kernelContext_37)->island_copy_0 = island_copy_11;
    (&kernelContext_37)->old_copy_0 = old_copy_11;
    threadgroup array<uint, int(256)> g_class_11;
    (&kernelContext_37)->g_class_0 = &g_class_11;
    threadgroup uint g_cut_next_11;
    (&kernelContext_37)->g_cut_next_0 = &g_cut_next_11;
    threadgroup uint g_cut_first_11;
    (&kernelContext_37)->g_cut_first_0 = &g_cut_first_11;
    threadgroup array<uint, int(256)> g_scan64_11;
    (&kernelContext_37)->g_scan64_0 = &g_scan64_11;
    bool _S192;
    if((*(control_11+int(0))) != 0U)
    {
        _S192 = true;
    }
    else
    {
        _S192 = (*((&kernelContext_37)->control_0+1U)) == 0U;
    }
    if(_S192)
    {
        return;
    }
    uint i_5 = id_1.x;
    uint t_3;
    if(i_5 == (*((&kernelContext_37)->plan_0+int(0))))
    {
        t_3 = 0U;
        for(;;)
        {
            if(t_3 < ((&kernelContext_37)->params_0->island_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S193 = island_word_0(i_5, t_3, &kernelContext_37);
            *((&kernelContext_37)->islands_0+_S193) = packed_float4(float4(0.0f) ) ;
            t_3 = t_3 + 1U;
        }
        return;
    }
    if(i_5 >= (*((&kernelContext_37)->plan_0+int(0))))
    {
        return;
    }
    uint device* _S194 = (&kernelContext_37)->plan_0+order_at_0(i_5);
    uint e_1 = *_S194;
    if((*_S194) >= (*((&kernelContext_37)->control_0+2U)))
    {
        _S192 = true;
    }
    else
    {
        _S192 = e_1 == i_5;
    }
    if(_S192)
    {
        return;
    }
    t_3 = 0U;
    for(;;)
    {
        if(t_3 < ((&kernelContext_37)->params_0->island_words_0))
        {
        }
        else
        {
            break;
        }
        uint _S195 = island_word_0(i_5, t_3, &kernelContext_37);
        packed_float4 device* _S196 = (&kernelContext_37)->islands_0+_S195;
        uint _S197 = island_word_0(e_1, t_3, &kernelContext_37);
        *_S196 = packed_float4(float4(*((&kernelContext_37)->old_islands_0+_S197)) ) ;
        t_3 = t_3 + 1U;
    }
    uint _S198 = island_word_0(e_1, 0U, &kernelContext_37);
    float4 _S199 = float4(*((&kernelContext_37)->old_islands_0+_S198)) ;
    uint _S200 = (as_type<uint>((_S199.y)));
    uint b0_14 = (as_type<uint>((_S199.z)));
    uint _S201 = (as_type<uint>((_S199.w)));
    uint c_3 = (as_type<uint>((_S199.x)));
    for(;;)
    {
        if(c_3 < _S200)
        {
        }
        else
        {
            break;
        }
        set_chunk_island_0(c_3, i_5, &kernelContext_37);
        c_3 = c_3 + 1U;
    }
    uint b_4 = b0_14;
    for(;;)
    {
        if(b_4 < _S201)
        {
        }
        else
        {
            break;
        }
        uint _S202 = bond_word_0(b_4, 14U, &kernelContext_37);
        thread float4 ids_2 = float4(*((&kernelContext_37)->bonds_0+_S202)) ;
        ids_2.w = (as_type<float>((i_5)));
        *((&kernelContext_37)->bonds_0+_S202) = packed_float4(ids_2) ;
        b_4 = b_4 + 1U;
    }
    if(((&kernelContext_37)->params_0->resident_0) != 0U)
    {
        t_3 = 0U;
        for(;;)
        {
            if(t_3 < 3U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_37)->ctab_0+((&kernelContext_37)->params_0->engine_base_0 + 3U * i_5 + t_3)) = packed_float4(float4(*((&kernelContext_37)->old_ctab_0+((&kernelContext_37)->params_0->engine_base_0 + 3U * e_1 + t_3))) ) ;
            t_3 = t_3 + 1U;
        }
    }
    *((&kernelContext_37)->gate_0+((&kernelContext_37)->params_0->gate_flag_base_0 + i_5)) = packed_float4(float4(*((&kernelContext_37)->gate_0+((&kernelContext_37)->params_0->gate_flag_base_0 + e_1))) ) ;
    return;
}

[[kernel]] void layout_copy(uint3 id_2 [[thread_position_in_grid]], uint device* control_12 [[buffer(24)]], uint device* plan_12 [[buffer(5)]], packed_uint4 device* parents_12 [[buffer(1)]], LayoutParams_0 constant* params_12 [[buffer(0)]], packed_float4 device* old_islands_12 [[buffer(16)]], packed_float4 device* children_12 [[buffer(3)]], uint device* work_12 [[buffer(2)]], packed_float4 device* chunks_12 [[buffer(7)]], packed_float4 device* old_chunks_12 [[buffer(6)]], packed_float4 device* state_12 [[buffer(9)]], packed_float4 device* old_state_12 [[buffer(8)]], packed_float4 device* scratch_12 [[buffer(18)]], packed_float4 device* ctab_12 [[buffer(20)]], packed_float4 device* old_ctab_12 [[buffer(19)]], packed_float4 device* old_bonds_12 [[buffer(10)]], packed_float4 device* bonds_12 [[buffer(11)]], packed_float4 device* dyn_12 [[buffer(13)]], packed_float4 device* old_dyn_12 [[buffer(12)]], uint device* old_index_12 [[buffer(14)]], uint device* index_12 [[buffer(15)]], packed_float4 device* frames_12 [[buffer(4)]], packed_float4 device* islands_12 [[buffer(17)]], packed_uint4 device* spawn_slots_12 [[buffer(22)]], packed_float4 device* spawn_extras_12 [[buffer(23)]], packed_float4 device* gate_12 [[buffer(21)]], packed_float4 device* island_copy_12 [[buffer(26)]], packed_float4 device* old_copy_12 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_38;
    (&kernelContext_38)->control_0 = control_12;
    (&kernelContext_38)->plan_0 = plan_12;
    (&kernelContext_38)->parents_0 = parents_12;
    (&kernelContext_38)->params_0 = params_12;
    (&kernelContext_38)->old_islands_0 = old_islands_12;
    (&kernelContext_38)->children_0 = children_12;
    (&kernelContext_38)->work_0 = work_12;
    (&kernelContext_38)->chunks_0 = chunks_12;
    (&kernelContext_38)->old_chunks_0 = old_chunks_12;
    (&kernelContext_38)->state_0 = state_12;
    (&kernelContext_38)->old_state_0 = old_state_12;
    (&kernelContext_38)->scratch_0 = scratch_12;
    (&kernelContext_38)->ctab_0 = ctab_12;
    (&kernelContext_38)->old_ctab_0 = old_ctab_12;
    (&kernelContext_38)->old_bonds_0 = old_bonds_12;
    (&kernelContext_38)->bonds_0 = bonds_12;
    (&kernelContext_38)->dyn_0 = dyn_12;
    (&kernelContext_38)->old_dyn_0 = old_dyn_12;
    (&kernelContext_38)->old_index_0 = old_index_12;
    (&kernelContext_38)->index_0 = index_12;
    (&kernelContext_38)->frames_0 = frames_12;
    (&kernelContext_38)->islands_0 = islands_12;
    (&kernelContext_38)->spawn_slots_0 = spawn_slots_12;
    (&kernelContext_38)->spawn_extras_0 = spawn_extras_12;
    (&kernelContext_38)->gate_0 = gate_12;
    (&kernelContext_38)->island_copy_0 = island_copy_12;
    (&kernelContext_38)->old_copy_0 = old_copy_12;
    threadgroup array<uint, int(256)> g_class_12;
    (&kernelContext_38)->g_class_0 = &g_class_12;
    threadgroup uint g_cut_next_12;
    (&kernelContext_38)->g_cut_next_0 = &g_cut_next_12;
    threadgroup uint g_cut_first_12;
    (&kernelContext_38)->g_cut_first_0 = &g_cut_first_12;
    threadgroup array<uint, int(256)> g_scan64_12;
    (&kernelContext_38)->g_scan64_0 = &g_scan64_12;
    bool _S203;
    if((*(control_12+int(0))) != 0U)
    {
        _S203 = true;
    }
    else
    {
        _S203 = (*((&kernelContext_38)->control_0+1U)) == 0U;
    }
    if(_S203)
    {
        return;
    }
    uint i_6 = id_2.x;
    uint device* _S204 = (&kernelContext_38)->plan_0+int(0);
    uint n_6 = *_S204;
    if(i_6 > (*_S204))
    {
        return;
    }
    uint t_4;
    if(i_6 == n_6)
    {
        t_4 = 0U;
        for(;;)
        {
            if(t_4 < ((&kernelContext_38)->params_0->island_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S205 = island_word_0(i_6, t_4, &kernelContext_38);
            *((&kernelContext_38)->island_copy_0+_S205) = packed_float4(float4(0.0f) ) ;
            t_4 = t_4 + 1U;
        }
        return;
    }
    uint device* _S206 = (&kernelContext_38)->plan_0+order_at_0(i_6);
    uint e_2 = *_S206;
    if((*_S206) < (*((&kernelContext_38)->control_0+2U)))
    {
        t_4 = 0U;
        for(;;)
        {
            if(t_4 < ((&kernelContext_38)->params_0->island_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S207 = island_word_0(i_6, t_4, &kernelContext_38);
            packed_float4 device* _S208 = (&kernelContext_38)->island_copy_0+_S207;
            uint _S209 = island_word_0(e_2, t_4, &kernelContext_38);
            *_S208 = packed_float4(float4(*((&kernelContext_38)->old_copy_0+_S209)) ) ;
            t_4 = t_4 + 1U;
        }
    }
    else
    {
        t_4 = 0U;
        for(;;)
        {
            if(t_4 < ((&kernelContext_38)->params_0->island_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S210 = island_word_0(i_6, t_4, &kernelContext_38);
            *((&kernelContext_38)->island_copy_0+_S210) = packed_float4(float4(*((&kernelContext_38)->islands_0+_S210)) ) ;
            t_4 = t_4 + 1U;
        }
    }
    return;
}

