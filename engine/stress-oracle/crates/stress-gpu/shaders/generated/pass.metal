#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint group_uniform_0(uint value_0, uint thread_0)
{
    return value_0;
}

struct PassParams_0
{
    uint chunk_words_0;
    uint bond_words_0;
    uint dyn_words_0;
    uint island_words_0;
    uint props_base_0;
    uint eload_base_0;
    uint pulse_base_0;
    uint pulse_count_base_0;
    uint pulse_cap_0;
    uint engine_base_0;
    uint resident_0;
    uint resume_0;
    uint island_float4s_0;
    uint index_words_0;
    uint engine_float4s_0;
    uint copy_float4s_0;
};

struct KernelContext_0
{
    uint device* counters_0;
    uint device* status_0;
    PassParams_0 constant* params_0;
    packed_float4 device* islands_0;
    packed_uint4 device* parents_0;
    packed_float4 device* ctab_0;
    packed_uint4 device* islands_of_0;
    packed_float4 device* parent_bodies_0;
    uint device* control_0;
    packed_uint4 device* args_0;
    packed_float4 device* old_islands_0;
    uint device* old_index_0;
    uint device* index_0;
    packed_float4 device* old_ctab_0;
    packed_float4 device* old_copy_0;
    packed_float4 device* island_copy_0;
    packed_float4 device* old_chunks_0;
    packed_float4 device* chunks_0;
    packed_float4 device* old_state_0;
    packed_float4 device* state_0;
    packed_float4 device* old_bonds_0;
    packed_float4 device* bonds_0;
    packed_float4 device* old_dyn_0;
    packed_float4 device* bond_dyn_0;
    packed_uint4 device* island_args_0;
    array<uint, int(1024)> threadgroup* g_parents_0;
    array<uint, int(1024)> threadgroup* g_chunks_0;
};

uint4 island_range_0(uint i_0, KernelContext_0 thread* kernelContext_0)
{
    return (as_type<uint4>((float4(*(kernelContext_0->islands_0+kernelContext_0->params_0->island_words_0 * i_0)) )));
}

[[kernel]] void pass_parents(uint3 thread_1 [[thread_position_in_threadgroup]], uint device* counters_1 [[buffer(9)]], uint device* status_1 [[buffer(1)]], PassParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* islands_1 [[buffer(2)]], packed_uint4 device* parents_1 [[buffer(5)]], packed_float4 device* ctab_1 [[buffer(3)]], packed_uint4 device* islands_of_1 [[buffer(6)]], packed_float4 device* parent_bodies_1 [[buffer(7)]], uint device* control_1 [[buffer(4)]], packed_uint4 device* args_1 [[buffer(8)]], packed_float4 device* old_islands_1 [[buffer(22)]], uint device* old_index_1 [[buffer(21)]], uint device* index_1 [[buffer(20)]], packed_float4 device* old_ctab_1 [[buffer(18)]], packed_float4 device* old_copy_1 [[buffer(24)]], packed_float4 device* island_copy_1 [[buffer(23)]], packed_float4 device* old_chunks_1 [[buffer(11)]], packed_float4 device* chunks_1 [[buffer(10)]], packed_float4 device* old_state_1 [[buffer(13)]], packed_float4 device* state_1 [[buffer(12)]], packed_float4 device* old_bonds_1 [[buffer(15)]], packed_float4 device* bonds_1 [[buffer(14)]], packed_float4 device* old_dyn_1 [[buffer(17)]], packed_float4 device* bond_dyn_1 [[buffer(16)]], packed_uint4 device* island_args_1 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->counters_0 = counters_1;
    (&kernelContext_1)->status_0 = status_1;
    (&kernelContext_1)->params_0 = params_1;
    (&kernelContext_1)->islands_0 = islands_1;
    (&kernelContext_1)->parents_0 = parents_1;
    (&kernelContext_1)->ctab_0 = ctab_1;
    (&kernelContext_1)->islands_of_0 = islands_of_1;
    (&kernelContext_1)->parent_bodies_0 = parent_bodies_1;
    (&kernelContext_1)->control_0 = control_1;
    (&kernelContext_1)->args_0 = args_1;
    (&kernelContext_1)->old_islands_0 = old_islands_1;
    (&kernelContext_1)->old_index_0 = old_index_1;
    (&kernelContext_1)->index_0 = index_1;
    (&kernelContext_1)->old_ctab_0 = old_ctab_1;
    (&kernelContext_1)->old_copy_0 = old_copy_1;
    (&kernelContext_1)->island_copy_0 = island_copy_1;
    (&kernelContext_1)->old_chunks_0 = old_chunks_1;
    (&kernelContext_1)->chunks_0 = chunks_1;
    (&kernelContext_1)->old_state_0 = old_state_1;
    (&kernelContext_1)->state_0 = state_1;
    (&kernelContext_1)->old_bonds_0 = old_bonds_1;
    (&kernelContext_1)->bonds_0 = bonds_1;
    (&kernelContext_1)->old_dyn_0 = old_dyn_1;
    (&kernelContext_1)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_1)->island_args_0 = island_args_1;
    threadgroup array<uint, int(1024)> g_parents_1;
    (&kernelContext_1)->g_parents_0 = &g_parents_1;
    threadgroup array<uint, int(1024)> g_chunks_1;
    (&kernelContext_1)->g_chunks_0 = &g_chunks_1;
    uint tid_0 = thread_1.x;
    uint n_0 = group_uniform_0(*(counters_1+int(0)), tid_0);
    bool _S1 = (*(status_1+1U)) != 0U;
    uint base_0 = 0U;
    uint carry_p_0 = 0U;
    uint carry_c_0 = 0U;
    for(;;)
    {
        if(base_0 < n_0)
        {
        }
        else
        {
            break;
        }
        uint i_1 = base_0 + tid_0;
        bool _S2;
        if(_S1)
        {
            _S2 = i_1 < n_0;
        }
        else
        {
            _S2 = false;
        }
        bool parent_0;
        if(_S2)
        {
            parent_0 = ((*((&kernelContext_1)->status_0+(16U + i_1))) & 8U) != 0U;
        }
        else
        {
            parent_0 = false;
        }
        uint4 r_0;
        if(parent_0)
        {
            uint4 _S3 = island_range_0(i_1, &kernelContext_1);
            r_0 = _S3;
        }
        else
        {
            r_0 = uint4(int4(int(0)) );
        }
        uint fp_0;
        if(parent_0)
        {
            fp_0 = 1U;
        }
        else
        {
            fp_0 = 0U;
        }
        uint fc_0;
        if(parent_0)
        {
            fc_0 = r_0.y - r_0.x;
        }
        else
        {
            fc_0 = 0U;
        }
        (*(&kernelContext_1)->g_parents_0)[tid_0] = fp_0;
        (*(&kernelContext_1)->g_chunks_0)[tid_0] = fc_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint step_0 = 1U;
        for(;;)
        {
            if(step_0 < 1024U)
            {
            }
            else
            {
                break;
            }
            bool _S4 = tid_0 >= step_0;
            uint ap_0;
            if(_S4)
            {
                ap_0 = (*(&kernelContext_1)->g_parents_0)[tid_0 - step_0];
            }
            else
            {
                ap_0 = 0U;
            }
            uint ac_0;
            if(_S4)
            {
                ac_0 = (*(&kernelContext_1)->g_chunks_0)[tid_0 - step_0];
            }
            else
            {
                ac_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*(&kernelContext_1)->g_parents_0)[tid_0] = (*(&kernelContext_1)->g_parents_0)[tid_0] + ap_0;
            (*(&kernelContext_1)->g_chunks_0)[tid_0] = (*(&kernelContext_1)->g_chunks_0)[tid_0] + ac_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_0 = step_0 << 1U;
        }
        if(parent_0)
        {
            uint p_0 = carry_p_0 + (*(&kernelContext_1)->g_parents_0)[tid_0] - fp_0;
            *((&kernelContext_1)->parents_0+p_0) = packed_uint4(uint4(i_1, carry_c_0 + (*(&kernelContext_1)->g_chunks_0)[tid_0] - fc_0, 0U, 0U)) ;
            uint _S5 = 3U * i_1;
            float4 _S6 = float4(*((&kernelContext_1)->ctab_0+((&kernelContext_1)->params_0->engine_base_0 + _S5))) ;
            float4 _S7 = float4(*((&kernelContext_1)->ctab_0+((&kernelContext_1)->params_0->engine_base_0 + _S5 + 1U))) ;
            float4 _S8 = float4(*((&kernelContext_1)->ctab_0+((&kernelContext_1)->params_0->engine_base_0 + _S5 + 2U))) ;
            uint4 body_0 = (as_type<uint4>((_S6)));
            *((&kernelContext_1)->islands_of_0+p_0) = packed_uint4(uint4(r_0.x, body_0.y, body_0.z, body_0.w)) ;
            uint _S9 = 2U * p_0;
            *((&kernelContext_1)->parent_bodies_0+_S9) = packed_float4(float4(_S6.x, _S8.xyz)) ;
            *((&kernelContext_1)->parent_bodies_0+(_S9 + 1U)) = packed_float4(float4(- _S7.x, - _S7.y, - _S7.z, _S7.w)) ;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_p_1 = carry_p_0 + (*(&kernelContext_1)->g_parents_0)[int(1023)];
        uint carry_c_1 = carry_c_0 + (*(&kernelContext_1)->g_chunks_0)[int(1023)];
        base_0 = base_0 + 1024U;
        carry_p_0 = carry_p_1;
        carry_c_0 = carry_c_1;
    }
    if(tid_0 != 0U)
    {
        return;
    }
    *((&kernelContext_1)->parents_0+carry_p_0) = packed_uint4(uint4(0U, carry_c_0, 0U, 0U)) ;
    *((&kernelContext_1)->control_0+0U) = 0U;
    *((&kernelContext_1)->control_0+1U) = carry_p_0;
    *((&kernelContext_1)->control_0+2U) = n_0;
    *((&kernelContext_1)->control_0+3U) = *((&kernelContext_1)->counters_0+int(3));
    *((&kernelContext_1)->control_0+4U) = *((&kernelContext_1)->counters_0+int(1));
    *((&kernelContext_1)->control_0+5U) = *((&kernelContext_1)->counters_0+int(2));
    *((&kernelContext_1)->control_0+6U) = carry_c_0;
    *((&kernelContext_1)->args_0+int(0)) = packed_uint4(uint4(carry_p_0, 1U, 1U, 0U)) ;
    *((&kernelContext_1)->args_0+int(1)) = packed_uint4(uint4((carry_c_0 + 63U) / 64U, 1U, 1U, 0U)) ;
    *((&kernelContext_1)->args_0+int(2)) = packed_uint4(uint4((carry_p_0 + 63U) / 64U, 1U, 1U, 0U)) ;
    packed_uint4 device* _S10 = (&kernelContext_1)->args_0+int(3);
    if(carry_p_0 > 0U)
    {
        base_0 = 256U;
    }
    else
    {
        base_0 = 0U;
    }
    *_S10 = packed_uint4(uint4(base_0, 1U, 1U, 0U)) ;
    *((&kernelContext_1)->args_0+int(4)) = packed_uint4(uint4(carry_c_0, 1U, 1U, 0U)) ;
    return;
}

[[kernel]] void pass_whole(uint3 id_0 [[thread_position_in_grid]], uint device* counters_2 [[buffer(9)]], uint device* status_2 [[buffer(1)]], PassParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* islands_2 [[buffer(2)]], packed_uint4 device* parents_2 [[buffer(5)]], packed_float4 device* ctab_2 [[buffer(3)]], packed_uint4 device* islands_of_2 [[buffer(6)]], packed_float4 device* parent_bodies_2 [[buffer(7)]], uint device* control_2 [[buffer(4)]], packed_uint4 device* args_2 [[buffer(8)]], packed_float4 device* old_islands_2 [[buffer(22)]], uint device* old_index_2 [[buffer(21)]], uint device* index_2 [[buffer(20)]], packed_float4 device* old_ctab_2 [[buffer(18)]], packed_float4 device* old_copy_2 [[buffer(24)]], packed_float4 device* island_copy_2 [[buffer(23)]], packed_float4 device* old_chunks_2 [[buffer(11)]], packed_float4 device* chunks_2 [[buffer(10)]], packed_float4 device* old_state_2 [[buffer(13)]], packed_float4 device* state_2 [[buffer(12)]], packed_float4 device* old_bonds_2 [[buffer(15)]], packed_float4 device* bonds_2 [[buffer(14)]], packed_float4 device* old_dyn_2 [[buffer(17)]], packed_float4 device* bond_dyn_2 [[buffer(16)]], packed_uint4 device* island_args_2 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->counters_0 = counters_2;
    (&kernelContext_2)->status_0 = status_2;
    (&kernelContext_2)->params_0 = params_2;
    (&kernelContext_2)->islands_0 = islands_2;
    (&kernelContext_2)->parents_0 = parents_2;
    (&kernelContext_2)->ctab_0 = ctab_2;
    (&kernelContext_2)->islands_of_0 = islands_of_2;
    (&kernelContext_2)->parent_bodies_0 = parent_bodies_2;
    (&kernelContext_2)->control_0 = control_2;
    (&kernelContext_2)->args_0 = args_2;
    (&kernelContext_2)->old_islands_0 = old_islands_2;
    (&kernelContext_2)->old_index_0 = old_index_2;
    (&kernelContext_2)->index_0 = index_2;
    (&kernelContext_2)->old_ctab_0 = old_ctab_2;
    (&kernelContext_2)->old_copy_0 = old_copy_2;
    (&kernelContext_2)->island_copy_0 = island_copy_2;
    (&kernelContext_2)->old_chunks_0 = old_chunks_2;
    (&kernelContext_2)->chunks_0 = chunks_2;
    (&kernelContext_2)->old_state_0 = old_state_2;
    (&kernelContext_2)->state_0 = state_2;
    (&kernelContext_2)->old_bonds_0 = old_bonds_2;
    (&kernelContext_2)->bonds_0 = bonds_2;
    (&kernelContext_2)->old_dyn_0 = old_dyn_2;
    (&kernelContext_2)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_2)->island_args_0 = island_args_2;
    threadgroup array<uint, int(1024)> g_parents_2;
    (&kernelContext_2)->g_parents_0 = &g_parents_2;
    threadgroup array<uint, int(1024)> g_chunks_2;
    (&kernelContext_2)->g_chunks_0 = &g_chunks_2;
    uint _S11 = id_0.x;
    uint k_0 = _S11;
    for(;;)
    {
        if(k_0 < ((&kernelContext_2)->params_0->island_float4s_0))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_2)->old_islands_0+k_0) = packed_float4(float4(*((&kernelContext_2)->islands_0+k_0)) ) ;
        k_0 = k_0 + 65536U;
    }
    k_0 = _S11;
    for(;;)
    {
        if(k_0 < ((&kernelContext_2)->params_0->index_words_0))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_2)->old_index_0+k_0) = (&kernelContext_2)->index_0[k_0];
        k_0 = k_0 + 65536U;
    }
    if(((&kernelContext_2)->params_0->resident_0) != 0U)
    {
        k_0 = _S11;
        for(;;)
        {
            if(k_0 < ((&kernelContext_2)->params_0->engine_float4s_0))
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_2)->old_ctab_0+((&kernelContext_2)->params_0->engine_base_0 + k_0)) = packed_float4(float4(*((&kernelContext_2)->ctab_0+((&kernelContext_2)->params_0->engine_base_0 + k_0))) ) ;
            k_0 = k_0 + 65536U;
        }
    }
    k_0 = _S11;
    for(;;)
    {
        if(k_0 < ((&kernelContext_2)->params_0->copy_float4s_0))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_2)->old_copy_0+k_0) = packed_float4(float4(*((&kernelContext_2)->island_copy_0+k_0)) ) ;
        k_0 = k_0 + 65536U;
    }
    return;
}

void snapshot_chunks_0(uint4 r_1, uint t_0, KernelContext_0 thread* kernelContext_3)
{
    uint _S12;
    uint _S13 = r_1.x;
    uint k_1 = kernelContext_3->params_0->chunk_words_0 * _S13 + t_0;
    for(;;)
    {
        uint _S14 = kernelContext_3->params_0->chunk_words_0;
        uint _S15 = r_1.y;
        _S12 = _S15;
        if(k_1 < (_S14 * _S15))
        {
        }
        else
        {
            break;
        }
        *(kernelContext_3->old_chunks_0+k_1) = packed_float4(float4(*(kernelContext_3->chunks_0+k_1)) ) ;
        k_1 = k_1 + 256U;
    }
    k_1 = 4U * _S13 + t_0;
    for(;;)
    {
        if(k_1 < (4U * _S12))
        {
        }
        else
        {
            break;
        }
        *(kernelContext_3->old_state_0+k_1) = packed_float4(float4(*(kernelContext_3->state_0+k_1)) ) ;
        k_1 = k_1 + 256U;
    }
    return;
}

void snapshot_bonds_0(uint4 r_2, uint t_1, KernelContext_0 thread* kernelContext_4)
{
    uint _S16;
    uint _S17 = r_2.z;
    uint k_2 = kernelContext_4->params_0->bond_words_0 * _S17 + t_1;
    for(;;)
    {
        uint _S18 = kernelContext_4->params_0->bond_words_0;
        uint _S19 = r_2.w;
        _S16 = _S19;
        if(k_2 < (_S18 * _S19))
        {
        }
        else
        {
            break;
        }
        *(kernelContext_4->old_bonds_0+k_2) = packed_float4(float4(*(kernelContext_4->bonds_0+k_2)) ) ;
        k_2 = k_2 + 256U;
    }
    k_2 = kernelContext_4->params_0->dyn_words_0 * _S17 + t_1;
    for(;;)
    {
        if(k_2 < (kernelContext_4->params_0->dyn_words_0 * _S16))
        {
        }
        else
        {
            break;
        }
        *(kernelContext_4->old_dyn_0+k_2) = packed_float4(float4(*(kernelContext_4->bond_dyn_0+k_2)) ) ;
        k_2 = k_2 + 256U;
    }
    return;
}

void snapshot_ctab_0(uint4 r_3, uint t_2, KernelContext_0 thread* kernelContext_5)
{
    uint _S20;
    uint _S21 = r_3.x;
    uint c_0 = _S21 + t_2;
    for(;;)
    {
        uint _S22 = r_3.y;
        _S20 = _S22;
        if(c_0 < _S22)
        {
        }
        else
        {
            break;
        }
        *(kernelContext_5->old_ctab_0+(kernelContext_5->params_0->props_base_0 + c_0)) = packed_float4(float4(*(kernelContext_5->ctab_0+(kernelContext_5->params_0->props_base_0 + c_0))) ) ;
        uint _S23 = 2U * c_0;
        *(kernelContext_5->old_ctab_0+(kernelContext_5->params_0->eload_base_0 + _S23)) = packed_float4(float4(*(kernelContext_5->ctab_0+(kernelContext_5->params_0->eload_base_0 + _S23))) ) ;
        *(kernelContext_5->old_ctab_0+(kernelContext_5->params_0->eload_base_0 + _S23 + 1U)) = packed_float4(float4(*(kernelContext_5->ctab_0+(kernelContext_5->params_0->eload_base_0 + _S23 + 1U))) ) ;
        *(kernelContext_5->old_ctab_0+(kernelContext_5->params_0->pulse_count_base_0 + c_0)) = packed_float4(float4(*(kernelContext_5->ctab_0+(kernelContext_5->params_0->pulse_count_base_0 + c_0))) ) ;
        c_0 = c_0 + 256U;
    }
    uint k_3 = 3U * kernelContext_5->params_0->pulse_cap_0 * _S21 + t_2;
    for(;;)
    {
        if(k_3 < (3U * kernelContext_5->params_0->pulse_cap_0 * _S20))
        {
        }
        else
        {
            break;
        }
        *(kernelContext_5->old_ctab_0+(kernelContext_5->params_0->pulse_base_0 + k_3)) = packed_float4(float4(*(kernelContext_5->ctab_0+(kernelContext_5->params_0->pulse_base_0 + k_3))) ) ;
        k_3 = k_3 + 256U;
    }
    return;
}

[[kernel]] void pass_snapshot(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], uint device* counters_3 [[buffer(9)]], uint device* status_3 [[buffer(1)]], PassParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* islands_3 [[buffer(2)]], packed_uint4 device* parents_3 [[buffer(5)]], packed_float4 device* ctab_3 [[buffer(3)]], packed_uint4 device* islands_of_3 [[buffer(6)]], packed_float4 device* parent_bodies_3 [[buffer(7)]], uint device* control_3 [[buffer(4)]], packed_uint4 device* args_3 [[buffer(8)]], packed_float4 device* old_islands_3 [[buffer(22)]], uint device* old_index_3 [[buffer(21)]], uint device* index_3 [[buffer(20)]], packed_float4 device* old_ctab_3 [[buffer(18)]], packed_float4 device* old_copy_3 [[buffer(24)]], packed_float4 device* island_copy_3 [[buffer(23)]], packed_float4 device* old_chunks_3 [[buffer(11)]], packed_float4 device* chunks_3 [[buffer(10)]], packed_float4 device* old_state_3 [[buffer(13)]], packed_float4 device* state_3 [[buffer(12)]], packed_float4 device* old_bonds_3 [[buffer(15)]], packed_float4 device* bonds_3 [[buffer(14)]], packed_float4 device* old_dyn_3 [[buffer(17)]], packed_float4 device* bond_dyn_3 [[buffer(16)]], packed_uint4 device* island_args_3 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->counters_0 = counters_3;
    (&kernelContext_6)->status_0 = status_3;
    (&kernelContext_6)->params_0 = params_3;
    (&kernelContext_6)->islands_0 = islands_3;
    (&kernelContext_6)->parents_0 = parents_3;
    (&kernelContext_6)->ctab_0 = ctab_3;
    (&kernelContext_6)->islands_of_0 = islands_of_3;
    (&kernelContext_6)->parent_bodies_0 = parent_bodies_3;
    (&kernelContext_6)->control_0 = control_3;
    (&kernelContext_6)->args_0 = args_3;
    (&kernelContext_6)->old_islands_0 = old_islands_3;
    (&kernelContext_6)->old_index_0 = old_index_3;
    (&kernelContext_6)->index_0 = index_3;
    (&kernelContext_6)->old_ctab_0 = old_ctab_3;
    (&kernelContext_6)->old_copy_0 = old_copy_3;
    (&kernelContext_6)->island_copy_0 = island_copy_3;
    (&kernelContext_6)->old_chunks_0 = old_chunks_3;
    (&kernelContext_6)->chunks_0 = chunks_3;
    (&kernelContext_6)->old_state_0 = old_state_3;
    (&kernelContext_6)->state_0 = state_3;
    (&kernelContext_6)->old_bonds_0 = old_bonds_3;
    (&kernelContext_6)->bonds_0 = bonds_3;
    (&kernelContext_6)->old_dyn_0 = old_dyn_3;
    (&kernelContext_6)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_6)->island_args_0 = island_args_3;
    threadgroup array<uint, int(1024)> g_parents_3;
    (&kernelContext_6)->g_parents_0 = &g_parents_3;
    threadgroup array<uint, int(1024)> g_chunks_3;
    (&kernelContext_6)->g_chunks_0 = &g_chunks_3;
    uint p_1 = group_0.x;
    if(p_1 >= (*(control_3+1U)))
    {
        return;
    }
    uint4 _S24 = island_range_0((uint4(*((&kernelContext_6)->parents_0+p_1)) ).x, &kernelContext_6);
    uint t_3 = thread_2.x;
    snapshot_chunks_0(_S24, t_3, &kernelContext_6);
    snapshot_bonds_0(_S24, t_3, &kernelContext_6);
    if(((&kernelContext_6)->params_0->resident_0) == 0U)
    {
        return;
    }
    snapshot_ctab_0(_S24, t_3, &kernelContext_6);
    return;
}

[[kernel]] void pass_snapshot_chunks(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_3 [[thread_position_in_threadgroup]], uint device* counters_4 [[buffer(9)]], uint device* status_4 [[buffer(1)]], PassParams_0 constant* params_4 [[buffer(0)]], packed_float4 device* islands_4 [[buffer(2)]], packed_uint4 device* parents_4 [[buffer(5)]], packed_float4 device* ctab_4 [[buffer(3)]], packed_uint4 device* islands_of_4 [[buffer(6)]], packed_float4 device* parent_bodies_4 [[buffer(7)]], uint device* control_4 [[buffer(4)]], packed_uint4 device* args_4 [[buffer(8)]], packed_float4 device* old_islands_4 [[buffer(22)]], uint device* old_index_4 [[buffer(21)]], uint device* index_4 [[buffer(20)]], packed_float4 device* old_ctab_4 [[buffer(18)]], packed_float4 device* old_copy_4 [[buffer(24)]], packed_float4 device* island_copy_4 [[buffer(23)]], packed_float4 device* old_chunks_4 [[buffer(11)]], packed_float4 device* chunks_4 [[buffer(10)]], packed_float4 device* old_state_4 [[buffer(13)]], packed_float4 device* state_4 [[buffer(12)]], packed_float4 device* old_bonds_4 [[buffer(15)]], packed_float4 device* bonds_4 [[buffer(14)]], packed_float4 device* old_dyn_4 [[buffer(17)]], packed_float4 device* bond_dyn_4 [[buffer(16)]], packed_uint4 device* island_args_4 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_7;
    (&kernelContext_7)->counters_0 = counters_4;
    (&kernelContext_7)->status_0 = status_4;
    (&kernelContext_7)->params_0 = params_4;
    (&kernelContext_7)->islands_0 = islands_4;
    (&kernelContext_7)->parents_0 = parents_4;
    (&kernelContext_7)->ctab_0 = ctab_4;
    (&kernelContext_7)->islands_of_0 = islands_of_4;
    (&kernelContext_7)->parent_bodies_0 = parent_bodies_4;
    (&kernelContext_7)->control_0 = control_4;
    (&kernelContext_7)->args_0 = args_4;
    (&kernelContext_7)->old_islands_0 = old_islands_4;
    (&kernelContext_7)->old_index_0 = old_index_4;
    (&kernelContext_7)->index_0 = index_4;
    (&kernelContext_7)->old_ctab_0 = old_ctab_4;
    (&kernelContext_7)->old_copy_0 = old_copy_4;
    (&kernelContext_7)->island_copy_0 = island_copy_4;
    (&kernelContext_7)->old_chunks_0 = old_chunks_4;
    (&kernelContext_7)->chunks_0 = chunks_4;
    (&kernelContext_7)->old_state_0 = old_state_4;
    (&kernelContext_7)->state_0 = state_4;
    (&kernelContext_7)->old_bonds_0 = old_bonds_4;
    (&kernelContext_7)->bonds_0 = bonds_4;
    (&kernelContext_7)->old_dyn_0 = old_dyn_4;
    (&kernelContext_7)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_7)->island_args_0 = island_args_4;
    threadgroup array<uint, int(1024)> g_parents_4;
    (&kernelContext_7)->g_parents_0 = &g_parents_4;
    threadgroup array<uint, int(1024)> g_chunks_4;
    (&kernelContext_7)->g_chunks_0 = &g_chunks_4;
    uint _S25 = group_1.x;
    if(_S25 >= (*(control_4+1U)))
    {
        return;
    }
    uint4 _S26 = island_range_0((uint4(*((&kernelContext_7)->parents_0+_S25)) ).x, &kernelContext_7);
    snapshot_chunks_0(_S26, thread_3.x, &kernelContext_7);
    return;
}

[[kernel]] void pass_snapshot_bonds(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], uint device* counters_5 [[buffer(9)]], uint device* status_5 [[buffer(1)]], PassParams_0 constant* params_5 [[buffer(0)]], packed_float4 device* islands_5 [[buffer(2)]], packed_uint4 device* parents_5 [[buffer(5)]], packed_float4 device* ctab_5 [[buffer(3)]], packed_uint4 device* islands_of_5 [[buffer(6)]], packed_float4 device* parent_bodies_5 [[buffer(7)]], uint device* control_5 [[buffer(4)]], packed_uint4 device* args_5 [[buffer(8)]], packed_float4 device* old_islands_5 [[buffer(22)]], uint device* old_index_5 [[buffer(21)]], uint device* index_5 [[buffer(20)]], packed_float4 device* old_ctab_5 [[buffer(18)]], packed_float4 device* old_copy_5 [[buffer(24)]], packed_float4 device* island_copy_5 [[buffer(23)]], packed_float4 device* old_chunks_5 [[buffer(11)]], packed_float4 device* chunks_5 [[buffer(10)]], packed_float4 device* old_state_5 [[buffer(13)]], packed_float4 device* state_5 [[buffer(12)]], packed_float4 device* old_bonds_5 [[buffer(15)]], packed_float4 device* bonds_5 [[buffer(14)]], packed_float4 device* old_dyn_5 [[buffer(17)]], packed_float4 device* bond_dyn_5 [[buffer(16)]], packed_uint4 device* island_args_5 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_8;
    (&kernelContext_8)->counters_0 = counters_5;
    (&kernelContext_8)->status_0 = status_5;
    (&kernelContext_8)->params_0 = params_5;
    (&kernelContext_8)->islands_0 = islands_5;
    (&kernelContext_8)->parents_0 = parents_5;
    (&kernelContext_8)->ctab_0 = ctab_5;
    (&kernelContext_8)->islands_of_0 = islands_of_5;
    (&kernelContext_8)->parent_bodies_0 = parent_bodies_5;
    (&kernelContext_8)->control_0 = control_5;
    (&kernelContext_8)->args_0 = args_5;
    (&kernelContext_8)->old_islands_0 = old_islands_5;
    (&kernelContext_8)->old_index_0 = old_index_5;
    (&kernelContext_8)->index_0 = index_5;
    (&kernelContext_8)->old_ctab_0 = old_ctab_5;
    (&kernelContext_8)->old_copy_0 = old_copy_5;
    (&kernelContext_8)->island_copy_0 = island_copy_5;
    (&kernelContext_8)->old_chunks_0 = old_chunks_5;
    (&kernelContext_8)->chunks_0 = chunks_5;
    (&kernelContext_8)->old_state_0 = old_state_5;
    (&kernelContext_8)->state_0 = state_5;
    (&kernelContext_8)->old_bonds_0 = old_bonds_5;
    (&kernelContext_8)->bonds_0 = bonds_5;
    (&kernelContext_8)->old_dyn_0 = old_dyn_5;
    (&kernelContext_8)->bond_dyn_0 = bond_dyn_5;
    (&kernelContext_8)->island_args_0 = island_args_5;
    threadgroup array<uint, int(1024)> g_parents_5;
    (&kernelContext_8)->g_parents_0 = &g_parents_5;
    threadgroup array<uint, int(1024)> g_chunks_5;
    (&kernelContext_8)->g_chunks_0 = &g_chunks_5;
    uint _S27 = group_2.x;
    if(_S27 >= (*(control_5+1U)))
    {
        return;
    }
    uint4 _S28 = island_range_0((uint4(*((&kernelContext_8)->parents_0+_S27)) ).x, &kernelContext_8);
    snapshot_bonds_0(_S28, thread_4.x, &kernelContext_8);
    return;
}

[[kernel]] void pass_snapshot_ctab(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], uint device* counters_6 [[buffer(9)]], uint device* status_6 [[buffer(1)]], PassParams_0 constant* params_6 [[buffer(0)]], packed_float4 device* islands_6 [[buffer(2)]], packed_uint4 device* parents_6 [[buffer(5)]], packed_float4 device* ctab_6 [[buffer(3)]], packed_uint4 device* islands_of_6 [[buffer(6)]], packed_float4 device* parent_bodies_6 [[buffer(7)]], uint device* control_6 [[buffer(4)]], packed_uint4 device* args_6 [[buffer(8)]], packed_float4 device* old_islands_6 [[buffer(22)]], uint device* old_index_6 [[buffer(21)]], uint device* index_6 [[buffer(20)]], packed_float4 device* old_ctab_6 [[buffer(18)]], packed_float4 device* old_copy_6 [[buffer(24)]], packed_float4 device* island_copy_6 [[buffer(23)]], packed_float4 device* old_chunks_6 [[buffer(11)]], packed_float4 device* chunks_6 [[buffer(10)]], packed_float4 device* old_state_6 [[buffer(13)]], packed_float4 device* state_6 [[buffer(12)]], packed_float4 device* old_bonds_6 [[buffer(15)]], packed_float4 device* bonds_6 [[buffer(14)]], packed_float4 device* old_dyn_6 [[buffer(17)]], packed_float4 device* bond_dyn_6 [[buffer(16)]], packed_uint4 device* island_args_6 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->counters_0 = counters_6;
    (&kernelContext_9)->status_0 = status_6;
    (&kernelContext_9)->params_0 = params_6;
    (&kernelContext_9)->islands_0 = islands_6;
    (&kernelContext_9)->parents_0 = parents_6;
    (&kernelContext_9)->ctab_0 = ctab_6;
    (&kernelContext_9)->islands_of_0 = islands_of_6;
    (&kernelContext_9)->parent_bodies_0 = parent_bodies_6;
    (&kernelContext_9)->control_0 = control_6;
    (&kernelContext_9)->args_0 = args_6;
    (&kernelContext_9)->old_islands_0 = old_islands_6;
    (&kernelContext_9)->old_index_0 = old_index_6;
    (&kernelContext_9)->index_0 = index_6;
    (&kernelContext_9)->old_ctab_0 = old_ctab_6;
    (&kernelContext_9)->old_copy_0 = old_copy_6;
    (&kernelContext_9)->island_copy_0 = island_copy_6;
    (&kernelContext_9)->old_chunks_0 = old_chunks_6;
    (&kernelContext_9)->chunks_0 = chunks_6;
    (&kernelContext_9)->old_state_0 = old_state_6;
    (&kernelContext_9)->state_0 = state_6;
    (&kernelContext_9)->old_bonds_0 = old_bonds_6;
    (&kernelContext_9)->bonds_0 = bonds_6;
    (&kernelContext_9)->old_dyn_0 = old_dyn_6;
    (&kernelContext_9)->bond_dyn_0 = bond_dyn_6;
    (&kernelContext_9)->island_args_0 = island_args_6;
    threadgroup array<uint, int(1024)> g_parents_6;
    (&kernelContext_9)->g_parents_0 = &g_parents_6;
    threadgroup array<uint, int(1024)> g_chunks_6;
    (&kernelContext_9)->g_chunks_0 = &g_chunks_6;
    uint _S29 = group_3.x;
    bool _S30;
    if(_S29 >= (*(control_6+1U)))
    {
        _S30 = true;
    }
    else
    {
        _S30 = ((&kernelContext_9)->params_0->resident_0) == 0U;
    }
    if(_S30)
    {
        return;
    }
    uint4 _S31 = island_range_0((uint4(*((&kernelContext_9)->parents_0+_S29)) ).x, &kernelContext_9);
    snapshot_ctab_0(_S31, thread_5.x, &kernelContext_9);
    return;
}

[[kernel]] void pass_finish(uint3 id_1 [[thread_position_in_grid]], uint device* counters_7 [[buffer(9)]], uint device* status_7 [[buffer(1)]], PassParams_0 constant* params_7 [[buffer(0)]], packed_float4 device* islands_7 [[buffer(2)]], packed_uint4 device* parents_7 [[buffer(5)]], packed_float4 device* ctab_7 [[buffer(3)]], packed_uint4 device* islands_of_7 [[buffer(6)]], packed_float4 device* parent_bodies_7 [[buffer(7)]], uint device* control_7 [[buffer(4)]], packed_uint4 device* args_7 [[buffer(8)]], packed_float4 device* old_islands_7 [[buffer(22)]], uint device* old_index_7 [[buffer(21)]], uint device* index_7 [[buffer(20)]], packed_float4 device* old_ctab_7 [[buffer(18)]], packed_float4 device* old_copy_7 [[buffer(24)]], packed_float4 device* island_copy_7 [[buffer(23)]], packed_float4 device* old_chunks_7 [[buffer(11)]], packed_float4 device* chunks_7 [[buffer(10)]], packed_float4 device* old_state_7 [[buffer(13)]], packed_float4 device* state_7 [[buffer(12)]], packed_float4 device* old_bonds_7 [[buffer(15)]], packed_float4 device* bonds_7 [[buffer(14)]], packed_float4 device* old_dyn_7 [[buffer(17)]], packed_float4 device* bond_dyn_7 [[buffer(16)]], packed_uint4 device* island_args_7 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_10;
    (&kernelContext_10)->counters_0 = counters_7;
    (&kernelContext_10)->status_0 = status_7;
    (&kernelContext_10)->params_0 = params_7;
    (&kernelContext_10)->islands_0 = islands_7;
    (&kernelContext_10)->parents_0 = parents_7;
    (&kernelContext_10)->ctab_0 = ctab_7;
    (&kernelContext_10)->islands_of_0 = islands_of_7;
    (&kernelContext_10)->parent_bodies_0 = parent_bodies_7;
    (&kernelContext_10)->control_0 = control_7;
    (&kernelContext_10)->args_0 = args_7;
    (&kernelContext_10)->old_islands_0 = old_islands_7;
    (&kernelContext_10)->old_index_0 = old_index_7;
    (&kernelContext_10)->index_0 = index_7;
    (&kernelContext_10)->old_ctab_0 = old_ctab_7;
    (&kernelContext_10)->old_copy_0 = old_copy_7;
    (&kernelContext_10)->island_copy_0 = island_copy_7;
    (&kernelContext_10)->old_chunks_0 = old_chunks_7;
    (&kernelContext_10)->chunks_0 = chunks_7;
    (&kernelContext_10)->old_state_0 = old_state_7;
    (&kernelContext_10)->state_0 = state_7;
    (&kernelContext_10)->old_bonds_0 = old_bonds_7;
    (&kernelContext_10)->bonds_0 = bonds_7;
    (&kernelContext_10)->old_dyn_0 = old_dyn_7;
    (&kernelContext_10)->bond_dyn_0 = bond_dyn_7;
    (&kernelContext_10)->island_args_0 = island_args_7;
    threadgroup array<uint, int(1024)> g_parents_7;
    (&kernelContext_10)->g_parents_0 = &g_parents_7;
    threadgroup array<uint, int(1024)> g_chunks_7;
    (&kernelContext_10)->g_chunks_0 = &g_chunks_7;
    bool _S32;
    if((*(control_7+1U)) == 0U)
    {
        _S32 = true;
    }
    else
    {
        _S32 = (*((&kernelContext_10)->control_0+0U)) != 0U;
    }
    if(_S32)
    {
        return;
    }
    uint device* _S33 = (&kernelContext_10)->control_0+7U;
    uint live_0 = *_S33;
    *((&kernelContext_10)->counters_0+int(0)) = *_S33;
    *((&kernelContext_10)->counters_0+int(1)) = *((&kernelContext_10)->control_0+8U);
    *((&kernelContext_10)->counters_0+int(2)) = *((&kernelContext_10)->control_0+9U);
    *((&kernelContext_10)->counters_0+int(3)) = *((&kernelContext_10)->control_0+10U);
    uint _S34 = max(live_0, 1U);
    *((&kernelContext_10)->island_args_0+int(0)) = packed_uint4(uint4(_S34, 1U, 1U, 0U)) ;
    *((&kernelContext_10)->island_args_0+int(1)) = packed_uint4(uint4((_S34 + 63U) / 64U, 1U, 1U, 0U)) ;
    if(((&kernelContext_10)->params_0->resume_0) != 0U)
    {
        *((&kernelContext_10)->status_0+0U) = 1U;
        *((&kernelContext_10)->status_0+1U) = 0U;
    }
    return;
}

