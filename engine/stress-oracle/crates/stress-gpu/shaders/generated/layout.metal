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

uint island_word_0(uint i_2, uint w_0, KernelContext_0 thread* kernelContext_6)
{
    return kernelContext_6->params_0->island_words_0 * i_2 + w_0;
}

uint child_order_at_0(uint slot_2, KernelContext_0 thread* kernelContext_7)
{
    uint _S17 = cap_0(kernelContext_7);
    return 16U + _S17 + 5U * *(kernelContext_7->control_0+6U) + *(kernelContext_7->control_0+2U) + kernelContext_7->params_0->bond_count_0 + slot_2;
}

void sort_children_0(uint4 parent_0, KernelContext_0 thread* kernelContext_8)
{
    uint k_1 = 0U;
    for(;;)
    {
        if(k_1 < (parent_0.z))
        {
        }
        else
        {
            break;
        }
        uint _S18 = parent_0.y;
        uint _S19 = child_island_at_0(_S18 + k_1, kernelContext_8);
        uint _S20 = *(kernelContext_8->plan_0+_S19);
        uint j_1 = k_1;
        for(;;)
        {
            bool _S21;
            if(j_1 > 0U)
            {
                uint _S22 = child_order_at_0(_S18 + j_1 - 1U, kernelContext_8);
                uint _S23 = child_island_at_0(_S18 + *(kernelContext_8->plan_0+_S22), kernelContext_8);
                _S21 = (*(kernelContext_8->plan_0+_S23)) > _S20;
            }
            else
            {
                _S21 = false;
            }
            if(_S21)
            {
            }
            else
            {
                break;
            }
            uint _S24 = _S18 + j_1;
            uint _S25 = child_order_at_0(_S24, kernelContext_8);
            uint device* _S26 = kernelContext_8->plan_0+_S25;
            uint _S27 = child_order_at_0(_S24 - 1U, kernelContext_8);
            *_S26 = *(kernelContext_8->plan_0+_S27);
            j_1 = j_1 - 1U;
        }
        uint _S28 = child_order_at_0(_S18 + j_1, kernelContext_8);
        *(kernelContext_8->plan_0+_S28) = k_1;
        k_1 = k_1 + 1U;
    }
    return;
}

uint child_by_island_0(uint4 parent_1, uint r_0, KernelContext_0 thread* kernelContext_9)
{
    uint _S29 = child_order_at_0(parent_1.y + r_0, kernelContext_9);
    return *(kernelContext_9->plan_0+_S29);
}

uint child_start_at_0(uint slot_3, KernelContext_0 thread* kernelContext_10)
{
    uint _S30 = cap_0(kernelContext_10);
    return 16U + _S30 + 6U * *(kernelContext_10->control_0+6U) + *(kernelContext_10->control_0+2U) + kernelContext_10->params_0->bond_count_0 + slot_3;
}

uint new_chunk_at_0(uint slot_4, KernelContext_0 thread* kernelContext_11)
{
    uint _S31 = cap_0(kernelContext_11);
    return 16U + _S31 + 2U * *(kernelContext_11->control_0+6U) + *(kernelContext_11->control_0+2U) + slot_4;
}

uint chunk_word_0(uint c_0, uint w_1, KernelContext_0 thread* kernelContext_12)
{
    return kernelContext_12->params_0->chunk_words_0 * c_0 + w_1;
}

void set_chunk_island_0(uint c_1, uint island_0, KernelContext_0 thread* kernelContext_13)
{
    uint _S32 = chunk_word_0(c_1, 8U, kernelContext_13);
    thread float4 info_0 = float4(*(kernelContext_13->chunks_0+_S32)) ;
    info_0.y = (as_type<float>((island_0)));
    *(kernelContext_13->chunks_0+_S32) = packed_float4(info_0) ;
    return;
}

uint child_bonds_at_0(uint slot_5, KernelContext_0 thread* kernelContext_14)
{
    uint _S33 = cap_0(kernelContext_14);
    return 16U + _S33 + 3U * *(kernelContext_14->control_0+6U) + *(kernelContext_14->control_0+2U) + 2U * slot_5;
}

uint bond_word_0(uint b_0, uint w_2, KernelContext_0 thread* kernelContext_15)
{
    return kernelContext_15->params_0->bond_words_0 * b_0 + w_2;
}

uint bond_new_at_0(uint b_1, KernelContext_0 thread* kernelContext_16)
{
    uint _S34 = cap_0(kernelContext_16);
    return 16U + _S34 + 5U * *(kernelContext_16->control_0+6U) + *(kernelContext_16->control_0+2U) + b_1;
}

void bond_ranks_0(uint4 parent_2, uint base_0, uint c0_0, uint b0_0, uint b1_0, uint tid_0, bool place_0, KernelContext_0 thread* kernelContext_17)
{
    bool _S35;
    uint block_0 = b0_0;
    for(;;)
    {
        if(block_0 < b1_0)
        {
        }
        else
        {
            break;
        }
        uint b_2 = block_0 + tid_0;
        uint cls_0;
        if(b_2 < b1_0)
        {
            uint _S36 = bond_word_0(b_2, 14U, kernelContext_17);
            uint4 ids_0 = (as_type<uint4>((float4(*(kernelContext_17->old_bonds_0+_S36)) )));
            uint ka_0 = kernelContext_17->work_0[base_0 + ids_0.y - c0_0];
            if(ka_0 == (kernelContext_17->work_0)[base_0 + ids_0.z - c0_0])
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
        (*kernelContext_17->g_class_0)[tid_0] = cls_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool _S37 = cls_0 != 4294967295U;
        bool last_0 = _S37;
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
            if(t_0 < tid_0)
            {
                _S35 = ((*kernelContext_17->g_class_0)[t_0]) == cls_0;
            }
            else
            {
                _S35 = false;
            }
            if(_S35)
            {
                rank_0 = rank_0 + 1U;
            }
            bool _S38;
            if(t_0 > tid_0)
            {
                _S38 = ((*kernelContext_17->g_class_0)[t_0]) == cls_0;
            }
            else
            {
                _S38 = false;
            }
            if(_S38)
            {
                last_0 = false;
            }
            t_0 = t_0 + 1U;
        }
        if(place_0)
        {
            _S35 = _S37;
        }
        else
        {
            _S35 = false;
        }
        if(_S35)
        {
            uint first_0;
            if(cls_0 == 4294967294U)
            {
                first_0 = *kernelContext_17->g_cut_next_0;
            }
            else
            {
                uint _S39 = child_bonds_at_0(base_0 + cls_0, kernelContext_17);
                first_0 = *(kernelContext_17->plan_0+(_S39 + 1U));
            }
            uint _S40 = bond_new_at_0(b_2, kernelContext_17);
            *(kernelContext_17->plan_0+_S40) = first_0 + rank_0;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if(last_0)
        {
            if(place_0)
            {
                if(cls_0 == 4294967294U)
                {
                    *kernelContext_17->g_cut_next_0 = *kernelContext_17->g_cut_next_0 + (rank_0 + 1U);
                }
                else
                {
                    uint _S41 = child_bonds_at_0(base_0 + cls_0, kernelContext_17);
                    uint device* _S42 = kernelContext_17->plan_0+(_S41 + 1U);
                    *_S42 = *_S42 + (rank_0 + 1U);
                }
            }
            else
            {
                if(cls_0 != 4294967294U)
                {
                    uint _S43 = child_bonds_at_0(base_0 + cls_0, kernelContext_17);
                    uint device* _S44 = kernelContext_17->plan_0+_S43;
                    *_S44 = *_S44 + (rank_0 + 1U);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        block_0 = block_0 + 256U;
    }
    return;
}

uint degree_at_0(uint slot_6, KernelContext_0 thread* kernelContext_18)
{
    uint _S45 = cap_0(kernelContext_18);
    return 16U + _S45 + 7U * *(kernelContext_18->control_0+6U) + *(kernelContext_18->control_0+2U) + kernelContext_18->params_0->bond_count_0 + slot_6;
}

void inverse3_0(float3 r0_0, float3 r1_0, float3 r2_0, float3 thread* i0_0, float3 thread* i1_0, float3 thread* i2_0)
{
    float3 c0_1 = cross(r1_0, r2_0);
    float3 c1_0 = cross(r2_0, r0_0);
    float3 c2_0 = cross(r0_0, r1_0);
    float det_0 = dot(r0_0, c0_1);
    float inv_0;
    if(det_0 != 0.0f)
    {
        inv_0 = 1.0f / det_0;
    }
    else
    {
        inv_0 = 0.0f;
    }
    *i0_0 = float3(c0_1.x, c1_0.x, c2_0.x) * float3(inv_0) ;
    *i1_0 = float3(c0_1.y, c1_0.y, c2_0.y) * float3(inv_0) ;
    *i2_0 = float3(c0_1.z, c1_0.z, c2_0.z) * float3(inv_0) ;
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
    float3 t_1 = cross(qv_0, v_0) * float3(2.0f) ;
    return v_0 + t_1 * float3(q_2->w_3)  + cross(qv_0, t_1);
}

float3 inverse_rotate_0(const Quat_0 thread* q_3, float3 v_1)
{
    thread Quat_0 c_2;
    (&c_2)->w_3 = q_3->w_3;
    (&c_2)->x_0 = - q_3->x_0;
    (&c_2)->y_0 = - q_3->y_0;
    (&c_2)->z_0 = - q_3->z_0;
    thread Quat_0 _S46 = c_2;
    float3 _S47 = rotate_0(&_S46, v_1);
    return _S47;
}

float4 quat_vec_0(const Quat_0 thread* q_4)
{
    return float4(q_4->x_0, q_4->y_0, q_4->z_0, q_4->w_3);
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

[[kernel]] void layout_parent(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], uint device* control_2 [[buffer(24)]], uint device* plan_2 [[buffer(5)]], packed_uint4 device* parents_2 [[buffer(1)]], LayoutParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* old_islands_2 [[buffer(16)]], packed_float4 device* children_2 [[buffer(3)]], uint device* work_2 [[buffer(2)]], packed_float4 device* chunks_2 [[buffer(7)]], packed_float4 device* old_chunks_2 [[buffer(6)]], packed_float4 device* state_2 [[buffer(9)]], packed_float4 device* old_state_2 [[buffer(8)]], packed_float4 device* scratch_2 [[buffer(18)]], packed_float4 device* ctab_2 [[buffer(20)]], packed_float4 device* old_ctab_2 [[buffer(19)]], packed_float4 device* old_bonds_2 [[buffer(10)]], packed_float4 device* bonds_2 [[buffer(11)]], packed_float4 device* dyn_2 [[buffer(13)]], packed_float4 device* old_dyn_2 [[buffer(12)]], uint device* old_index_2 [[buffer(14)]], uint device* index_2 [[buffer(15)]], packed_float4 device* frames_2 [[buffer(4)]], packed_float4 device* islands_2 [[buffer(17)]], packed_uint4 device* spawn_slots_2 [[buffer(22)]], packed_float4 device* spawn_extras_2 [[buffer(23)]], packed_float4 device* gate_2 [[buffer(21)]], packed_float4 device* island_copy_2 [[buffer(26)]], packed_float4 device* old_copy_2 [[buffer(25)]])
{
    uint hi_0;
    uint w_4;
    uint w_5;
    uint w_6;
    uint v_2;
    uint step_0;
    uint add_0;
    thread KernelContext_0 kernelContext_19;
    (&kernelContext_19)->control_0 = control_2;
    (&kernelContext_19)->plan_0 = plan_2;
    (&kernelContext_19)->parents_0 = parents_2;
    (&kernelContext_19)->params_0 = params_2;
    (&kernelContext_19)->old_islands_0 = old_islands_2;
    (&kernelContext_19)->children_0 = children_2;
    (&kernelContext_19)->work_0 = work_2;
    (&kernelContext_19)->chunks_0 = chunks_2;
    (&kernelContext_19)->old_chunks_0 = old_chunks_2;
    (&kernelContext_19)->state_0 = state_2;
    (&kernelContext_19)->old_state_0 = old_state_2;
    (&kernelContext_19)->scratch_0 = scratch_2;
    (&kernelContext_19)->ctab_0 = ctab_2;
    (&kernelContext_19)->old_ctab_0 = old_ctab_2;
    (&kernelContext_19)->old_bonds_0 = old_bonds_2;
    (&kernelContext_19)->bonds_0 = bonds_2;
    (&kernelContext_19)->dyn_0 = dyn_2;
    (&kernelContext_19)->old_dyn_0 = old_dyn_2;
    (&kernelContext_19)->old_index_0 = old_index_2;
    (&kernelContext_19)->index_0 = index_2;
    (&kernelContext_19)->frames_0 = frames_2;
    (&kernelContext_19)->islands_0 = islands_2;
    (&kernelContext_19)->spawn_slots_0 = spawn_slots_2;
    (&kernelContext_19)->spawn_extras_0 = spawn_extras_2;
    (&kernelContext_19)->gate_0 = gate_2;
    (&kernelContext_19)->island_copy_0 = island_copy_2;
    (&kernelContext_19)->old_copy_0 = old_copy_2;
    threadgroup array<uint, int(256)> g_class_2;
    (&kernelContext_19)->g_class_0 = &g_class_2;
    threadgroup uint g_cut_next_2;
    (&kernelContext_19)->g_cut_next_0 = &g_cut_next_2;
    threadgroup uint g_cut_first_2;
    (&kernelContext_19)->g_cut_first_0 = &g_cut_first_2;
    threadgroup array<uint, int(256)> g_scan64_2;
    (&kernelContext_19)->g_scan64_0 = &g_scan64_2;
    uint tid_1 = thread_0.x;
    uint p_1 = group_0.x;
    bool scaled_0;
    if(p_1 >= (*(control_2+1U)))
    {
        scaled_0 = true;
    }
    else
    {
        scaled_0 = (*((&kernelContext_19)->control_0+int(0))) != 0U;
    }
    if(scaled_0)
    {
        return;
    }
    uint4 _S48 = uint4(*((&kernelContext_19)->parents_0+p_1)) ;
    uint _S49 = _S48.z;
    if(_S49 < 2U)
    {
        return;
    }
    uint _S50 = island_word_0(_S48.x, 0U, &kernelContext_19);
    float4 _S51 = float4(*((&kernelContext_19)->old_islands_0+_S50)) ;
    uint c0_2 = (as_type<uint>((_S51.x)));
    uint c1_1 = (as_type<uint>((_S51.y)));
    uint b0_1 = (as_type<uint>((_S51.z)));
    uint b1_1 = (as_type<uint>((_S51.w)));
    uint _S52 = c1_1 - c0_2;
    uint base_1 = _S48.y;
    uint _S53 = *((&kernelContext_19)->control_0+6U);
    bool _S54 = tid_1 == 0U;
    uint i_3;
    uint lo_0;
    if(_S54)
    {
        sort_children_0(_S48, &kernelContext_19);
        i_3 = 0U;
        lo_0 = 0U;
        for(;;)
        {
            if(i_3 < _S49)
            {
            }
            else
            {
                break;
            }
            uint _S55 = child_by_island_0(_S48, i_3, &kernelContext_19);
            uint _S56 = base_1 + _S55;
            uint _S57 = child_start_at_0(_S56, &kernelContext_19);
            *((&kernelContext_19)->plan_0+_S57) = lo_0;
            uint at_0 = lo_0 + (as_type<uint>(((float4(*((&kernelContext_19)->children_0+8U * _S56)) ).z)));
            i_3 = i_3 + 1U;
            lo_0 = at_0;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    i_3 = tid_1;
    for(;;)
    {
        if(i_3 < _S52)
        {
        }
        else
        {
            break;
        }
        uint _S58 = _S49 - 1U;
        lo_0 = 0U;
        hi_0 = _S58;
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
            if((as_type<uint>(((float4(*((&kernelContext_19)->children_0+8U * (base_1 + mid_0))) ).y))) <= i_3)
            {
                lo_0 = mid_0;
            }
            else
            {
                hi_0 = mid_0 - 1U;
            }
        }
        uint _S59 = base_1 + lo_0;
        uint j_2 = i_3 - (as_type<uint>(((float4(*((&kernelContext_19)->children_0+8U * _S59)) ).y)));
        uint _S60 = new_chunk_at_0(base_1 + (&kernelContext_19)->work_0[_S53 + base_1 + i_3], &kernelContext_19);
        uint device* _S61 = (&kernelContext_19)->plan_0+_S60;
        uint _S62 = child_start_at_0(_S59, &kernelContext_19);
        *_S61 = *((&kernelContext_19)->plan_0+_S62) + j_2;
        i_3 = i_3 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    lo_0 = tid_1;
    for(;;)
    {
        if(lo_0 < _S52)
        {
        }
        else
        {
            break;
        }
        uint _S63 = base_1 + lo_0;
        uint _S64 = new_chunk_at_0(_S63, &kernelContext_19);
        uint nk_0 = *((&kernelContext_19)->plan_0+_S64);
        uint _S65 = child_island_at_0(base_1 + (&kernelContext_19)->work_0[_S63], &kernelContext_19);
        uint island_1 = *((&kernelContext_19)->plan_0+_S65);
        w_4 = 0U;
        for(;;)
        {
            if(w_4 < ((&kernelContext_19)->params_0->chunk_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S66 = chunk_word_0(c0_2 + nk_0, w_4, &kernelContext_19);
            packed_float4 device* _S67 = (&kernelContext_19)->chunks_0+_S66;
            uint _S68 = chunk_word_0(c0_2 + lo_0, w_4, &kernelContext_19);
            *_S67 = packed_float4(float4(*((&kernelContext_19)->old_chunks_0+_S68)) ) ;
            w_4 = w_4 + 1U;
        }
        uint _S69 = c0_2 + nk_0;
        set_chunk_island_0(_S69, island_1, &kernelContext_19);
        w_5 = 0U;
        for(;;)
        {
            if(w_5 < 4U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_19)->state_0+(4U * _S69 + w_5)) = packed_float4(float4(*((&kernelContext_19)->old_state_0+(4U * (c0_2 + lo_0) + w_5))) ) ;
            w_5 = w_5 + 1U;
        }
        w_6 = 0U;
        for(;;)
        {
            if(w_6 < 23U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_19)->scratch_0+((&kernelContext_19)->params_0->statics_base_0 + 23U * _S69 + w_6)) = packed_float4(float4(0.0f) ) ;
            w_6 = w_6 + 1U;
        }
        if(((&kernelContext_19)->params_0->resident_0) != 0U)
        {
            uint o_0 = c0_2 + lo_0;
            *((&kernelContext_19)->ctab_0+((&kernelContext_19)->params_0->props_base_0 + _S69)) = packed_float4(float4(*((&kernelContext_19)->old_ctab_0+((&kernelContext_19)->params_0->props_base_0 + o_0))) ) ;
            uint _S70 = 2U * _S69;
            uint _S71 = 2U * o_0;
            *((&kernelContext_19)->ctab_0+((&kernelContext_19)->params_0->eload_base_0 + _S70)) = packed_float4(float4(*((&kernelContext_19)->old_ctab_0+((&kernelContext_19)->params_0->eload_base_0 + _S71))) ) ;
            *((&kernelContext_19)->ctab_0+((&kernelContext_19)->params_0->eload_base_0 + _S70 + 1U)) = packed_float4(float4(*((&kernelContext_19)->old_ctab_0+((&kernelContext_19)->params_0->eload_base_0 + _S71 + 1U))) ) ;
            *((&kernelContext_19)->ctab_0+((&kernelContext_19)->params_0->pulse_count_base_0 + _S69)) = packed_float4(float4(*((&kernelContext_19)->old_ctab_0+((&kernelContext_19)->params_0->pulse_count_base_0 + o_0))) ) ;
            hi_0 = 0U;
            for(;;)
            {
                if(hi_0 < (3U * (&kernelContext_19)->params_0->pulse_cap_0))
                {
                }
                else
                {
                    break;
                }
                *((&kernelContext_19)->ctab_0+((&kernelContext_19)->params_0->pulse_base_0 + 3U * (&kernelContext_19)->params_0->pulse_cap_0 * _S69 + hi_0)) = packed_float4(float4(*((&kernelContext_19)->old_ctab_0+((&kernelContext_19)->params_0->pulse_base_0 + 3U * (&kernelContext_19)->params_0->pulse_cap_0 * o_0 + hi_0))) ) ;
                hi_0 = hi_0 + 1U;
            }
        }
        lo_0 = lo_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    hi_0 = tid_1;
    for(;;)
    {
        if(hi_0 < _S49)
        {
        }
        else
        {
            break;
        }
        uint _S72 = child_bonds_at_0(base_1 + hi_0, &kernelContext_19);
        *((&kernelContext_19)->plan_0+_S72) = 0U;
        hi_0 = hi_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bond_ranks_0(_S48, base_1, c0_2, b0_1, b1_1, tid_1, false, &kernelContext_19);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint b_3;
    if(_S54)
    {
        b_3 = 0U;
        w_4 = 0U;
        for(;;)
        {
            if(b_3 < _S49)
            {
            }
            else
            {
                break;
            }
            uint _S73 = child_by_island_0(_S48, b_3, &kernelContext_19);
            uint _S74 = base_1 + _S73;
            uint _S75 = child_bonds_at_0(_S74, &kernelContext_19);
            uint count_0 = *((&kernelContext_19)->plan_0+_S75);
            uint _S76 = child_bonds_at_0(_S74, &kernelContext_19);
            uint _S77 = b0_1 + w_4;
            *((&kernelContext_19)->plan_0+_S76) = _S77;
            uint _S78 = child_bonds_at_0(_S74, &kernelContext_19);
            *((&kernelContext_19)->plan_0+(_S78 + 1U)) = _S77;
            uint at_1 = w_4 + count_0;
            b_3 = b_3 + 1U;
            w_4 = at_1;
        }
        uint _S79 = b0_1 + w_4;
        *(&kernelContext_19)->g_cut_first_0 = _S79;
        *(&kernelContext_19)->g_cut_next_0 = _S79;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bond_ranks_0(_S48, base_1, c0_2, b0_1, b1_1, tid_1, true, &kernelContext_19);
    b_3 = b0_1 + tid_1;
    for(;;)
    {
        if(b_3 < b1_1)
        {
        }
        else
        {
            break;
        }
        uint _S80 = bond_word_0(b_3, 14U, &kernelContext_19);
        uint4 ids_1 = (as_type<uint4>((float4(*((&kernelContext_19)->old_bonds_0+_S80)) )));
        uint _S81 = ids_1.y;
        uint _S82 = base_1 + _S81 - c0_2;
        uint ka_1 = (&kernelContext_19)->work_0[_S82];
        uint _S83 = ids_1.z;
        uint _S84 = base_1 + _S83 - c0_2;
        uint4 new_ids_0;
        if(ka_1 == ((&kernelContext_19)->work_0)[_S84])
        {
            uint _S85 = bond_new_at_0(b_3, &kernelContext_19);
            uint device* _S86 = (&kernelContext_19)->plan_0+_S85;
            uint _S87 = ids_1.x;
            uint _S88 = new_chunk_at_0(_S82, &kernelContext_19);
            uint _S89 = c0_2 + *((&kernelContext_19)->plan_0+_S88);
            uint _S90 = new_chunk_at_0(_S84, &kernelContext_19);
            uint _S91 = c0_2 + *((&kernelContext_19)->plan_0+_S90);
            uint _S92 = child_island_at_0(base_1 + ka_1, &kernelContext_19);
            uint4 _S93 = uint4(_S87, _S89, _S91, *((&kernelContext_19)->plan_0+_S92));
            w_4 = *_S86;
            new_ids_0 = _S93;
        }
        else
        {
            uint _S94 = bond_new_at_0(b_3, &kernelContext_19);
            uint nb_0 = *((&kernelContext_19)->plan_0+_S94);
            uint4 _S95 = uint4(ids_1.x, _S81, _S83, 4294967295U);
            w_4 = 0U;
            for(;;)
            {
                if(w_4 < 3U)
                {
                }
                else
                {
                    break;
                }
                *((&kernelContext_19)->scratch_0+(3U * nb_0 + w_4)) = packed_float4(float4(0.0f) ) ;
                w_4 = w_4 + 1U;
            }
            uint _S96 = 2U * nb_0;
            packed_float4 _S97 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_19)->scratch_0+((&kernelContext_19)->params_0->statics_base_0 + 23U * (&kernelContext_19)->params_0->chunk_count_0 + _S96)) = _S97;
            *((&kernelContext_19)->scratch_0+((&kernelContext_19)->params_0->statics_base_0 + 23U * (&kernelContext_19)->params_0->chunk_count_0 + _S96 + 1U)) = _S97;
            w_4 = nb_0;
            new_ids_0 = _S95;
        }
        w_5 = 0U;
        for(;;)
        {
            if(w_5 < ((&kernelContext_19)->params_0->bond_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S98 = bond_word_0(w_4, w_5, &kernelContext_19);
            packed_float4 device* _S99 = (&kernelContext_19)->bonds_0+_S98;
            uint _S100 = bond_word_0(b_3, w_5, &kernelContext_19);
            *_S99 = packed_float4(float4(*((&kernelContext_19)->old_bonds_0+_S100)) ) ;
            w_5 = w_5 + 1U;
        }
        uint _S101 = bond_word_0(w_4, 14U, &kernelContext_19);
        *((&kernelContext_19)->bonds_0+_S101) = packed_float4((as_type<float4>((new_ids_0)))) ;
        w_6 = 0U;
        for(;;)
        {
            if(w_6 < ((&kernelContext_19)->params_0->dyn_words_0))
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_19)->dyn_0+((&kernelContext_19)->params_0->dyn_words_0 * w_4 + w_6)) = packed_float4(float4(*((&kernelContext_19)->old_dyn_0+((&kernelContext_19)->params_0->dyn_words_0 * b_3 + w_6))) ) ;
            w_6 = w_6 + 1U;
        }
        b_3 = b_3 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bool _S102;
    if((*(&kernelContext_19)->g_cut_first_0) < b1_1)
    {
        w_4 = 2U * *(&kernelContext_19)->g_cut_first_0;
    }
    else
    {
        uint _S103 = (&kernelContext_19)->old_index_0[c0_2];
        w_4 = 4294967295U;
        w_5 = _S103;
        for(;;)
        {
            if(w_5 < ((&kernelContext_19)->old_index_0)[c1_1])
            {
                scaled_0 = w_4 == 4294967295U;
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
            uint entry_0 = (&kernelContext_19)->old_index_0[w_5];
            uint bond_0 = entry_0 >> 1U;
            if(bond_0 < b0_1)
            {
                _S102 = true;
            }
            else
            {
                _S102 = bond_0 >= b1_1;
            }
            if(_S102)
            {
                w_4 = entry_0;
            }
            w_5 = w_5 + 1U;
        }
    }
    lo_0 = tid_1;
    for(;;)
    {
        if(lo_0 < _S52)
        {
        }
        else
        {
            break;
        }
        uint _S104 = new_chunk_at_0(base_1 + lo_0, &kernelContext_19);
        uint _S105 = degree_at_0(base_1 + *((&kernelContext_19)->plan_0+_S104), &kernelContext_19);
        uint _S106 = c0_2 + lo_0;
        *((&kernelContext_19)->plan_0+_S105) = (&kernelContext_19)->old_index_0[_S106 + 1U] - (&kernelContext_19)->old_index_0[_S106];
        lo_0 = lo_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint _S107 = (&kernelContext_19)->old_index_0[c0_2];
    w_5 = 0U;
    w_6 = _S107;
    for(;;)
    {
        if(w_5 < _S52)
        {
        }
        else
        {
            break;
        }
        uint i_4 = w_5 + tid_1;
        bool _S108 = i_4 < _S52;
        if(_S108)
        {
            uint _S109 = degree_at_0(base_1 + i_4, &kernelContext_19);
            v_2 = *((&kernelContext_19)->plan_0+_S109);
        }
        else
        {
            v_2 = 0U;
        }
        (*(&kernelContext_19)->g_scan64_0)[tid_1] = v_2;
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
            if(tid_1 >= step_0)
            {
                add_0 = (*(&kernelContext_19)->g_scan64_0)[tid_1 - step_0];
            }
            else
            {
                add_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*(&kernelContext_19)->g_scan64_0)[tid_1] = (*(&kernelContext_19)->g_scan64_0)[tid_1] + add_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_0 = step_0 << 1U;
        }
        if(_S108)
        {
            *((&kernelContext_19)->index_0+(c0_2 + i_4)) = w_6 + (*(&kernelContext_19)->g_scan64_0)[tid_1] - v_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_0 = w_6 + (*(&kernelContext_19)->g_scan64_0)[255U];
        w_5 = w_5 + 256U;
        w_6 = carry_0;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    lo_0 = tid_1;
    for(;;)
    {
        if(lo_0 < _S52)
        {
        }
        else
        {
            break;
        }
        uint _S110 = new_chunk_at_0(base_1 + lo_0, &kernelContext_19);
        uint _S111 = *((&kernelContext_19)->index_0+(c0_2 + *((&kernelContext_19)->plan_0+_S110)));
        uint _S112 = c0_2 + lo_0;
        uint _S113 = _S112 + 1U;
        uint _S114 = (&kernelContext_19)->old_index_0[_S113] - (&kernelContext_19)->old_index_0[_S112];
        v_2 = (&kernelContext_19)->old_index_0[_S112];
        step_0 = 0U;
        for(;;)
        {
            if(v_2 < ((&kernelContext_19)->old_index_0)[_S113])
            {
            }
            else
            {
                break;
            }
            uint entry_1 = (&kernelContext_19)->old_index_0[v_2];
            uint bond_1 = entry_1 >> 1U;
            if(bond_1 < b0_1)
            {
                scaled_0 = true;
            }
            else
            {
                scaled_0 = bond_1 >= b1_1;
            }
            if(scaled_0)
            {
                v_2 = v_2 + 1U;
                continue;
            }
            uint _S115 = bond_new_at_0(bond_1, &kernelContext_19);
            uint device* _S116 = (&kernelContext_19)->plan_0+_S115;
            uint nb_1 = *_S116;
            if((*_S116) >= (*(&kernelContext_19)->g_cut_first_0))
            {
                v_2 = v_2 + 1U;
                continue;
            }
            *((&kernelContext_19)->index_0+(_S111 + step_0)) = 2U * nb_1 + (entry_1 & 1U);
            step_0 = step_0 + 1U;
            v_2 = v_2 + 1U;
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
            *((&kernelContext_19)->index_0+(_S111 + add_0)) = w_4;
            add_0 = add_0 + 1U;
        }
        lo_0 = lo_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    hi_0 = tid_1;
    for(;;)
    {
        if(hi_0 < _S49)
        {
        }
        else
        {
            break;
        }
        uint slot_7 = base_1 + hi_0;
        uint _S117 = child_island_at_0(slot_7, &kernelContext_19);
        uint island_2 = *((&kernelContext_19)->plan_0+_S117);
        uint _S118 = 8U * slot_7;
        float4 _S119 = float4(*((&kernelContext_19)->children_0+_S118)) ;
        uint _S120 = (as_type<uint>((_S119.y)));
        uint count_1 = (as_type<uint>((_S119.z)));
        bool anchored_0 = (as_type<uint>((_S119.w))) != 0U;
        float4 _S121 = float4(*((&kernelContext_19)->children_0+(_S118 + 1U))) ;
        float3 i0_1 = (float4(*((&kernelContext_19)->children_0+(_S118 + 2U))) ).xyz;
        float3 i1_1 = (float4(*((&kernelContext_19)->children_0+(_S118 + 3U))) ).xyz;
        float3 i2_1 = (float4(*((&kernelContext_19)->children_0+(_S118 + 4U))) ).xyz;
        float3 v_3 = (float4(*((&kernelContext_19)->children_0+(_S118 + 5U))) ).xyz;
        float3 w_7 = (float4(*((&kernelContext_19)->children_0+(_S118 + 6U))) ).xyz;
        float3 _S122 = float3(0.0f) ;
        v_2 = 4294967295U;
        scaled_0 = false;
        step_0 = 0U;
        float wmass_0 = 0.0f;
        float3 wmoment_0 = _S122;
        float su_mass_0 = 0.0f;
        float3 su_0 = _S122;
        for(;;)
        {
            if(step_0 < count_1)
            {
            }
            else
            {
                break;
            }
            uint m_0 = (&kernelContext_19)->work_0[_S53 + base_1 + _S120 + step_0];
            uint _S123 = new_chunk_at_0(base_1 + m_0, &kernelContext_19);
            uint _S124 = min(v_2, *((&kernelContext_19)->plan_0+_S123));
            uint _S125 = c0_2 + m_0;
            uint _S126 = chunk_word_0(_S125, 0U, &kernelContext_19);
            float4 _S127 = float4(*((&kernelContext_19)->old_chunks_0+_S126)) ;
            uint _S128 = chunk_word_0(_S125, 7U, &kernelContext_19);
            float mu_0 = (float4(*((&kernelContext_19)->old_chunks_0+_S128)) ).x;
            if(scaled_0)
            {
                scaled_0 = true;
            }
            else
            {
                scaled_0 = mu_0 != 1.0f;
            }
            float _S129 = _S127.w;
            float wmass_1 = wmass_0 + _S129 * mu_0;
            float3 _S130 = float3(_S129) ;
            float3 wmoment_1 = wmoment_0 + _S127.xyz * _S130 * float3(mu_0) ;
            float su_mass_1 = su_mass_0 + _S129;
            float3 su_1 = su_0 + (float4(*((&kernelContext_19)->old_state_0+4U * _S125)) ).xyz * _S130;
            uint _S131 = step_0 + 1U;
            v_2 = _S124;
            step_0 = _S131;
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
            wcom_0 = _S121.xyz;
        }
        float3 w0_0;
        float3 w1_0;
        float3 w2_0;
        if(scaled_0)
        {
            add_0 = 0U;
            w0_0 = _S122;
            w1_0 = _S122;
            w2_0 = _S122;
            for(;;)
            {
                if(add_0 < count_1)
                {
                }
                else
                {
                    break;
                }
                uint _S132 = c0_2 + (&kernelContext_19)->work_0[_S53 + base_1 + _S120 + add_0];
                uint _S133 = chunk_word_0(_S132, 0U, &kernelContext_19);
                float4 _S134 = float4(*((&kernelContext_19)->old_chunks_0+_S133)) ;
                uint _S135 = chunk_word_0(_S132, 7U, &kernelContext_19);
                float mu_1 = (float4(*((&kernelContext_19)->old_chunks_0+_S135)) ).x;
                float3 rr_0 = _S134.xyz - wcom_0;
                float dd_0 = dot(rr_0, rr_0);
                uint _S136 = chunk_word_0(_S132, 1U, &kernelContext_19);
                float3 _S137 = float3(_S134.w) ;
                float3 _S138 = float3(mu_1) ;
                float3 w0_1 = w0_0 + ((float4(*((&kernelContext_19)->old_chunks_0+_S136)) ).xyz + (float3(dd_0, 0.0f, 0.0f) - float3(rr_0.x)  * rr_0) * _S137) * _S138;
                uint _S139 = chunk_word_0(_S132, 2U, &kernelContext_19);
                float3 w1_1 = w1_0 + ((float4(*((&kernelContext_19)->old_chunks_0+_S139)) ).xyz + (float3(0.0f, dd_0, 0.0f) - float3(rr_0.y)  * rr_0) * _S137) * _S138;
                uint _S140 = chunk_word_0(_S132, 3U, &kernelContext_19);
                float3 w2_1 = w2_0 + ((float4(*((&kernelContext_19)->old_chunks_0+_S140)) ).xyz + (float3(0.0f, 0.0f, dd_0) - float3(rr_0.z)  * rr_0) * _S137) * _S138;
                add_0 = add_0 + 1U;
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
        uint _S141 = 5U * p_1;
        Quat_0 q_5 = quat_of_0(float4(*((&kernelContext_19)->frames_0+_S141)) );
        float3 pos_0 = (float4(*((&kernelContext_19)->frames_0+(_S141 + 4U))) ).xyz;
        float3 pos_1;
        if(!anchored_0)
        {
            float3 _S142 = su_0 / float3(su_mass_0) ;
            thread Quat_0 _S143 = q_5;
            float3 _S144 = rotate_0(&_S143, _S142);
            pos_1 = pos_0 + _S144;
        }
        else
        {
            pos_1 = pos_0;
        }
        thread Quat_0 _S145 = q_5;
        float3 _S146 = inverse_rotate_0(&_S145, w_7);
        float3 _S147 = float3(dot(i0_1, _S146), dot(i1_1, _S146), dot(i2_1, _S146));
        thread Quat_0 _S148 = q_5;
        float3 _S149 = rotate_0(&_S148, _S147);
        if(anchored_0)
        {
            i_3 = 1U;
        }
        else
        {
            i_3 = 0U;
        }
        if(((&kernelContext_19)->params_0->driven_0) != 0U)
        {
            lo_0 = 2U;
        }
        else
        {
            lo_0 = 0U;
        }
        uint flags_0 = (i_3 | lo_0) | 16U;
        uint _S150 = child_bonds_at_0(slot_7, &kernelContext_19);
        uint bfirst_0 = *((&kernelContext_19)->plan_0+_S150);
        uint _S151 = child_bonds_at_0(slot_7, &kernelContext_19);
        uint bend_0 = *((&kernelContext_19)->plan_0+(_S151 + 1U));
        thread array<float4, int(25)> rec_0;
        add_0 = 0U;
        for(;;)
        {
            if(add_0 < 25U)
            {
            }
            else
            {
                break;
            }
            rec_0[add_0] = float4(0.0f) ;
            add_0 = add_0 + 1U;
        }
        uint _S152 = c0_2 + v_2;
        rec_0[int(0)] = (as_type<float4>((uint4(_S152, _S152 + count_1, bfirst_0, bend_0))));
        rec_0[int(1)] = (as_type<float4>((uint4(flags_0, 0U, 0U, (&kernelContext_19)->params_0->substeps_0))));
        rec_0[int(2)] = _S121;
        rec_0[int(3)] = float4(i0_1, 0.0f);
        rec_0[int(4)] = float4(i1_1, 0.0f);
        rec_0[int(5)] = float4(i2_1, 0.0f);
        rec_0[int(6)] = float4(v0_0, 0.0f);
        rec_0[int(7)] = float4(v1_0, 0.0f);
        rec_0[int(8)] = float4(v2_0, 0.0f);
        float _S153;
        if(scaled_0)
        {
            _S153 = wmass_0;
        }
        else
        {
            _S153 = _S121.w;
        }
        rec_0[int(9)] = float4(wcom_0, _S153);
        rec_0[int(10)] = float4(x0_0, 0.0f);
        rec_0[int(11)] = float4(x1_0, 0.0f);
        rec_0[int(12)] = float4(x2_0, 0.0f);
        thread Quat_0 _S154 = q_5;
        float4 _S155 = quat_vec_0(&_S154);
        rec_0[int(13)] = _S155;
        rec_0[int(14)] = float4(pos_1, 0.0f);
        float3 _S156;
        if(anchored_0)
        {
            _S156 = _S122;
        }
        else
        {
            _S156 = v_3;
        }
        rec_0[int(16)] = float4(_S156, 0.0f);
        float3 _S157;
        if(anchored_0)
        {
            _S157 = _S122;
        }
        else
        {
            _S157 = w_7;
        }
        rec_0[int(18)] = float4(_S157, 0.0f);
        uint _S158 = child_id_at_0(slot_7, &kernelContext_19);
        rec_0[int(19)] = (as_type<float4>((uint4(0U, 0U, 0U, *((&kernelContext_19)->plan_0+_S158)))));
        rec_0[int(21)] = float4(0.0f, 0.0f, 0.0f, float_bits_0(2139095040U));
        float3 _S159;
        if(anchored_0)
        {
            _S159 = _S122;
        }
        else
        {
            _S159 = _S149;
        }
        rec_0[int(23)] = float4(_S159, 0.0f);
        uint t_2 = 0U;
        for(;;)
        {
            if(t_2 < 25U)
            {
            }
            else
            {
                break;
            }
            uint _S160 = island_word_0(island_2, t_2, &kernelContext_19);
            *((&kernelContext_19)->islands_0+_S160) = packed_float4(rec_0[t_2]) ;
            t_2 = t_2 + 1U;
        }
        if(((&kernelContext_19)->params_0->resident_0) != 0U)
        {
            uint4 _S161 = uint4(*((&kernelContext_19)->spawn_slots_0+slot_7)) ;
            uint _S162 = 2U * slot_7;
            float4 _S163 = float4(*((&kernelContext_19)->spawn_extras_0+(_S162 + 1U))) ;
            uint at_2 = (&kernelContext_19)->params_0->engine_base_0 + 3U * island_2;
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
            if(anchored_0)
            {
                _S102 = true;
            }
            else
            {
                _S102 = _S164 == 0U;
            }
            if(_S102)
            {
                *((&kernelContext_19)->ctab_0+at_2) = packed_float4(float4(float_bits_0(4294967295U), body_range_0)) ;
                packed_float4 _S166 = packed_float4(float4(0.0f) ) ;
                *((&kernelContext_19)->ctab_0+(at_2 + 1U)) = _S166;
                *((&kernelContext_19)->ctab_0+(at_2 + 2U)) = _S166;
            }
            else
            {
                *((&kernelContext_19)->ctab_0+at_2) = packed_float4(float4((as_type<float>((_S161.x))), body_range_0)) ;
                *((&kernelContext_19)->ctab_0+(at_2 + 1U)) = packed_float4(float4(- _S163.x, - _S163.y, - _S163.z, _S163.w)) ;
                *((&kernelContext_19)->ctab_0+(at_2 + 2U)) = packed_float4(float4((float4(*((&kernelContext_19)->spawn_extras_0+_S162)) ).xyz, 0.0f)) ;
            }
        }
        *((&kernelContext_19)->gate_0+((&kernelContext_19)->params_0->gate_flag_base_0 + island_2)) = packed_float4(float4(0.0f) ) ;
        hi_0 = hi_0 + 256U;
    }
    return;
}

[[kernel]] void layout_moved(uint3 id_1 [[thread_position_in_grid]], uint device* control_3 [[buffer(24)]], uint device* plan_3 [[buffer(5)]], packed_uint4 device* parents_3 [[buffer(1)]], LayoutParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* old_islands_3 [[buffer(16)]], packed_float4 device* children_3 [[buffer(3)]], uint device* work_3 [[buffer(2)]], packed_float4 device* chunks_3 [[buffer(7)]], packed_float4 device* old_chunks_3 [[buffer(6)]], packed_float4 device* state_3 [[buffer(9)]], packed_float4 device* old_state_3 [[buffer(8)]], packed_float4 device* scratch_3 [[buffer(18)]], packed_float4 device* ctab_3 [[buffer(20)]], packed_float4 device* old_ctab_3 [[buffer(19)]], packed_float4 device* old_bonds_3 [[buffer(10)]], packed_float4 device* bonds_3 [[buffer(11)]], packed_float4 device* dyn_3 [[buffer(13)]], packed_float4 device* old_dyn_3 [[buffer(12)]], uint device* old_index_3 [[buffer(14)]], uint device* index_3 [[buffer(15)]], packed_float4 device* frames_3 [[buffer(4)]], packed_float4 device* islands_3 [[buffer(17)]], packed_uint4 device* spawn_slots_3 [[buffer(22)]], packed_float4 device* spawn_extras_3 [[buffer(23)]], packed_float4 device* gate_3 [[buffer(21)]], packed_float4 device* island_copy_3 [[buffer(26)]], packed_float4 device* old_copy_3 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_20;
    (&kernelContext_20)->control_0 = control_3;
    (&kernelContext_20)->plan_0 = plan_3;
    (&kernelContext_20)->parents_0 = parents_3;
    (&kernelContext_20)->params_0 = params_3;
    (&kernelContext_20)->old_islands_0 = old_islands_3;
    (&kernelContext_20)->children_0 = children_3;
    (&kernelContext_20)->work_0 = work_3;
    (&kernelContext_20)->chunks_0 = chunks_3;
    (&kernelContext_20)->old_chunks_0 = old_chunks_3;
    (&kernelContext_20)->state_0 = state_3;
    (&kernelContext_20)->old_state_0 = old_state_3;
    (&kernelContext_20)->scratch_0 = scratch_3;
    (&kernelContext_20)->ctab_0 = ctab_3;
    (&kernelContext_20)->old_ctab_0 = old_ctab_3;
    (&kernelContext_20)->old_bonds_0 = old_bonds_3;
    (&kernelContext_20)->bonds_0 = bonds_3;
    (&kernelContext_20)->dyn_0 = dyn_3;
    (&kernelContext_20)->old_dyn_0 = old_dyn_3;
    (&kernelContext_20)->old_index_0 = old_index_3;
    (&kernelContext_20)->index_0 = index_3;
    (&kernelContext_20)->frames_0 = frames_3;
    (&kernelContext_20)->islands_0 = islands_3;
    (&kernelContext_20)->spawn_slots_0 = spawn_slots_3;
    (&kernelContext_20)->spawn_extras_0 = spawn_extras_3;
    (&kernelContext_20)->gate_0 = gate_3;
    (&kernelContext_20)->island_copy_0 = island_copy_3;
    (&kernelContext_20)->old_copy_0 = old_copy_3;
    threadgroup array<uint, int(256)> g_class_3;
    (&kernelContext_20)->g_class_0 = &g_class_3;
    threadgroup uint g_cut_next_3;
    (&kernelContext_20)->g_cut_next_0 = &g_cut_next_3;
    threadgroup uint g_cut_first_3;
    (&kernelContext_20)->g_cut_first_0 = &g_cut_first_3;
    threadgroup array<uint, int(256)> g_scan64_3;
    (&kernelContext_20)->g_scan64_0 = &g_scan64_3;
    bool _S167;
    if((*(control_3+int(0))) != 0U)
    {
        _S167 = true;
    }
    else
    {
        _S167 = (*((&kernelContext_20)->control_0+1U)) == 0U;
    }
    if(_S167)
    {
        return;
    }
    uint i_5 = id_1.x;
    uint t_3;
    if(i_5 == (*((&kernelContext_20)->plan_0+int(0))))
    {
        t_3 = 0U;
        for(;;)
        {
            if(t_3 < ((&kernelContext_20)->params_0->island_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S168 = island_word_0(i_5, t_3, &kernelContext_20);
            *((&kernelContext_20)->islands_0+_S168) = packed_float4(float4(0.0f) ) ;
            t_3 = t_3 + 1U;
        }
        return;
    }
    if(i_5 >= (*((&kernelContext_20)->plan_0+int(0))))
    {
        return;
    }
    uint device* _S169 = (&kernelContext_20)->plan_0+order_at_0(i_5);
    uint e_1 = *_S169;
    if((*_S169) >= (*((&kernelContext_20)->control_0+2U)))
    {
        _S167 = true;
    }
    else
    {
        _S167 = e_1 == i_5;
    }
    if(_S167)
    {
        return;
    }
    t_3 = 0U;
    for(;;)
    {
        if(t_3 < ((&kernelContext_20)->params_0->island_words_0))
        {
        }
        else
        {
            break;
        }
        uint _S170 = island_word_0(i_5, t_3, &kernelContext_20);
        packed_float4 device* _S171 = (&kernelContext_20)->islands_0+_S170;
        uint _S172 = island_word_0(e_1, t_3, &kernelContext_20);
        *_S171 = packed_float4(float4(*((&kernelContext_20)->old_islands_0+_S172)) ) ;
        t_3 = t_3 + 1U;
    }
    uint _S173 = island_word_0(e_1, 0U, &kernelContext_20);
    float4 _S174 = float4(*((&kernelContext_20)->old_islands_0+_S173)) ;
    uint _S175 = (as_type<uint>((_S174.y)));
    uint b0_2 = (as_type<uint>((_S174.z)));
    uint _S176 = (as_type<uint>((_S174.w)));
    uint c_3 = (as_type<uint>((_S174.x)));
    for(;;)
    {
        if(c_3 < _S175)
        {
        }
        else
        {
            break;
        }
        set_chunk_island_0(c_3, i_5, &kernelContext_20);
        c_3 = c_3 + 1U;
    }
    uint b_4 = b0_2;
    for(;;)
    {
        if(b_4 < _S176)
        {
        }
        else
        {
            break;
        }
        uint _S177 = bond_word_0(b_4, 14U, &kernelContext_20);
        thread float4 ids_2 = float4(*((&kernelContext_20)->bonds_0+_S177)) ;
        ids_2.w = (as_type<float>((i_5)));
        *((&kernelContext_20)->bonds_0+_S177) = packed_float4(ids_2) ;
        b_4 = b_4 + 1U;
    }
    if(((&kernelContext_20)->params_0->resident_0) != 0U)
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
            *((&kernelContext_20)->ctab_0+((&kernelContext_20)->params_0->engine_base_0 + 3U * i_5 + t_3)) = packed_float4(float4(*((&kernelContext_20)->old_ctab_0+((&kernelContext_20)->params_0->engine_base_0 + 3U * e_1 + t_3))) ) ;
            t_3 = t_3 + 1U;
        }
    }
    *((&kernelContext_20)->gate_0+((&kernelContext_20)->params_0->gate_flag_base_0 + i_5)) = packed_float4(float4(*((&kernelContext_20)->gate_0+((&kernelContext_20)->params_0->gate_flag_base_0 + e_1))) ) ;
    return;
}

[[kernel]] void layout_copy(uint3 id_2 [[thread_position_in_grid]], uint device* control_4 [[buffer(24)]], uint device* plan_4 [[buffer(5)]], packed_uint4 device* parents_4 [[buffer(1)]], LayoutParams_0 constant* params_4 [[buffer(0)]], packed_float4 device* old_islands_4 [[buffer(16)]], packed_float4 device* children_4 [[buffer(3)]], uint device* work_4 [[buffer(2)]], packed_float4 device* chunks_4 [[buffer(7)]], packed_float4 device* old_chunks_4 [[buffer(6)]], packed_float4 device* state_4 [[buffer(9)]], packed_float4 device* old_state_4 [[buffer(8)]], packed_float4 device* scratch_4 [[buffer(18)]], packed_float4 device* ctab_4 [[buffer(20)]], packed_float4 device* old_ctab_4 [[buffer(19)]], packed_float4 device* old_bonds_4 [[buffer(10)]], packed_float4 device* bonds_4 [[buffer(11)]], packed_float4 device* dyn_4 [[buffer(13)]], packed_float4 device* old_dyn_4 [[buffer(12)]], uint device* old_index_4 [[buffer(14)]], uint device* index_4 [[buffer(15)]], packed_float4 device* frames_4 [[buffer(4)]], packed_float4 device* islands_4 [[buffer(17)]], packed_uint4 device* spawn_slots_4 [[buffer(22)]], packed_float4 device* spawn_extras_4 [[buffer(23)]], packed_float4 device* gate_4 [[buffer(21)]], packed_float4 device* island_copy_4 [[buffer(26)]], packed_float4 device* old_copy_4 [[buffer(25)]])
{
    thread KernelContext_0 kernelContext_21;
    (&kernelContext_21)->control_0 = control_4;
    (&kernelContext_21)->plan_0 = plan_4;
    (&kernelContext_21)->parents_0 = parents_4;
    (&kernelContext_21)->params_0 = params_4;
    (&kernelContext_21)->old_islands_0 = old_islands_4;
    (&kernelContext_21)->children_0 = children_4;
    (&kernelContext_21)->work_0 = work_4;
    (&kernelContext_21)->chunks_0 = chunks_4;
    (&kernelContext_21)->old_chunks_0 = old_chunks_4;
    (&kernelContext_21)->state_0 = state_4;
    (&kernelContext_21)->old_state_0 = old_state_4;
    (&kernelContext_21)->scratch_0 = scratch_4;
    (&kernelContext_21)->ctab_0 = ctab_4;
    (&kernelContext_21)->old_ctab_0 = old_ctab_4;
    (&kernelContext_21)->old_bonds_0 = old_bonds_4;
    (&kernelContext_21)->bonds_0 = bonds_4;
    (&kernelContext_21)->dyn_0 = dyn_4;
    (&kernelContext_21)->old_dyn_0 = old_dyn_4;
    (&kernelContext_21)->old_index_0 = old_index_4;
    (&kernelContext_21)->index_0 = index_4;
    (&kernelContext_21)->frames_0 = frames_4;
    (&kernelContext_21)->islands_0 = islands_4;
    (&kernelContext_21)->spawn_slots_0 = spawn_slots_4;
    (&kernelContext_21)->spawn_extras_0 = spawn_extras_4;
    (&kernelContext_21)->gate_0 = gate_4;
    (&kernelContext_21)->island_copy_0 = island_copy_4;
    (&kernelContext_21)->old_copy_0 = old_copy_4;
    threadgroup array<uint, int(256)> g_class_4;
    (&kernelContext_21)->g_class_0 = &g_class_4;
    threadgroup uint g_cut_next_4;
    (&kernelContext_21)->g_cut_next_0 = &g_cut_next_4;
    threadgroup uint g_cut_first_4;
    (&kernelContext_21)->g_cut_first_0 = &g_cut_first_4;
    threadgroup array<uint, int(256)> g_scan64_4;
    (&kernelContext_21)->g_scan64_0 = &g_scan64_4;
    bool _S178;
    if((*(control_4+int(0))) != 0U)
    {
        _S178 = true;
    }
    else
    {
        _S178 = (*((&kernelContext_21)->control_0+1U)) == 0U;
    }
    if(_S178)
    {
        return;
    }
    uint i_6 = id_2.x;
    uint device* _S179 = (&kernelContext_21)->plan_0+int(0);
    uint n_1 = *_S179;
    if(i_6 > (*_S179))
    {
        return;
    }
    uint t_4;
    if(i_6 == n_1)
    {
        t_4 = 0U;
        for(;;)
        {
            if(t_4 < ((&kernelContext_21)->params_0->island_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S180 = island_word_0(i_6, t_4, &kernelContext_21);
            *((&kernelContext_21)->island_copy_0+_S180) = packed_float4(float4(0.0f) ) ;
            t_4 = t_4 + 1U;
        }
        return;
    }
    uint device* _S181 = (&kernelContext_21)->plan_0+order_at_0(i_6);
    uint e_2 = *_S181;
    if((*_S181) < (*((&kernelContext_21)->control_0+2U)))
    {
        t_4 = 0U;
        for(;;)
        {
            if(t_4 < ((&kernelContext_21)->params_0->island_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S182 = island_word_0(i_6, t_4, &kernelContext_21);
            packed_float4 device* _S183 = (&kernelContext_21)->island_copy_0+_S182;
            uint _S184 = island_word_0(e_2, t_4, &kernelContext_21);
            *_S183 = packed_float4(float4(*((&kernelContext_21)->old_copy_0+_S184)) ) ;
            t_4 = t_4 + 1U;
        }
    }
    else
    {
        t_4 = 0U;
        for(;;)
        {
            if(t_4 < ((&kernelContext_21)->params_0->island_words_0))
            {
            }
            else
            {
                break;
            }
            uint _S185 = island_word_0(i_6, t_4, &kernelContext_21);
            *((&kernelContext_21)->island_copy_0+_S185) = packed_float4(float4(*((&kernelContext_21)->islands_0+_S185)) ) ;
            t_4 = t_4 + 1U;
        }
    }
    return;
}

