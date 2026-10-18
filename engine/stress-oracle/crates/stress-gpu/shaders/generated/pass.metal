#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
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

[[kernel]] void pass_parents(uint3 thread_0 [[thread_position_in_threadgroup]], uint device* counters_1 [[buffer(9)]], uint device* status_1 [[buffer(1)]], PassParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* islands_1 [[buffer(2)]], packed_uint4 device* parents_1 [[buffer(5)]], packed_float4 device* ctab_1 [[buffer(3)]], packed_uint4 device* islands_of_1 [[buffer(6)]], packed_float4 device* parent_bodies_1 [[buffer(7)]], uint device* control_1 [[buffer(4)]], packed_uint4 device* args_1 [[buffer(8)]], packed_float4 device* old_islands_1 [[buffer(22)]], uint device* old_index_1 [[buffer(21)]], uint device* index_1 [[buffer(20)]], packed_float4 device* old_ctab_1 [[buffer(18)]], packed_float4 device* old_copy_1 [[buffer(24)]], packed_float4 device* island_copy_1 [[buffer(23)]], packed_float4 device* old_chunks_1 [[buffer(11)]], packed_float4 device* chunks_1 [[buffer(10)]], packed_float4 device* old_state_1 [[buffer(13)]], packed_float4 device* state_1 [[buffer(12)]], packed_float4 device* old_bonds_1 [[buffer(15)]], packed_float4 device* bonds_1 [[buffer(14)]], packed_float4 device* old_dyn_1 [[buffer(17)]], packed_float4 device* bond_dyn_1 [[buffer(16)]], packed_uint4 device* island_args_1 [[buffer(19)]])
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
    uint tid_0 = thread_0.x;
    uint n_0 = *(counters_1+int(0));
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

[[kernel]] void pass_snapshot(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], uint device* counters_3 [[buffer(9)]], uint device* status_3 [[buffer(1)]], PassParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* islands_3 [[buffer(2)]], packed_uint4 device* parents_3 [[buffer(5)]], packed_float4 device* ctab_3 [[buffer(3)]], packed_uint4 device* islands_of_3 [[buffer(6)]], packed_float4 device* parent_bodies_3 [[buffer(7)]], uint device* control_3 [[buffer(4)]], packed_uint4 device* args_3 [[buffer(8)]], packed_float4 device* old_islands_3 [[buffer(22)]], uint device* old_index_3 [[buffer(21)]], uint device* index_3 [[buffer(20)]], packed_float4 device* old_ctab_3 [[buffer(18)]], packed_float4 device* old_copy_3 [[buffer(24)]], packed_float4 device* island_copy_3 [[buffer(23)]], packed_float4 device* old_chunks_3 [[buffer(11)]], packed_float4 device* chunks_3 [[buffer(10)]], packed_float4 device* old_state_3 [[buffer(13)]], packed_float4 device* state_3 [[buffer(12)]], packed_float4 device* old_bonds_3 [[buffer(15)]], packed_float4 device* bonds_3 [[buffer(14)]], packed_float4 device* old_dyn_3 [[buffer(17)]], packed_float4 device* bond_dyn_3 [[buffer(16)]], packed_uint4 device* island_args_3 [[buffer(19)]])
{
    uint _S12;
    uint _S13;
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->counters_0 = counters_3;
    (&kernelContext_3)->status_0 = status_3;
    (&kernelContext_3)->params_0 = params_3;
    (&kernelContext_3)->islands_0 = islands_3;
    (&kernelContext_3)->parents_0 = parents_3;
    (&kernelContext_3)->ctab_0 = ctab_3;
    (&kernelContext_3)->islands_of_0 = islands_of_3;
    (&kernelContext_3)->parent_bodies_0 = parent_bodies_3;
    (&kernelContext_3)->control_0 = control_3;
    (&kernelContext_3)->args_0 = args_3;
    (&kernelContext_3)->old_islands_0 = old_islands_3;
    (&kernelContext_3)->old_index_0 = old_index_3;
    (&kernelContext_3)->index_0 = index_3;
    (&kernelContext_3)->old_ctab_0 = old_ctab_3;
    (&kernelContext_3)->old_copy_0 = old_copy_3;
    (&kernelContext_3)->island_copy_0 = island_copy_3;
    (&kernelContext_3)->old_chunks_0 = old_chunks_3;
    (&kernelContext_3)->chunks_0 = chunks_3;
    (&kernelContext_3)->old_state_0 = old_state_3;
    (&kernelContext_3)->state_0 = state_3;
    (&kernelContext_3)->old_bonds_0 = old_bonds_3;
    (&kernelContext_3)->bonds_0 = bonds_3;
    (&kernelContext_3)->old_dyn_0 = old_dyn_3;
    (&kernelContext_3)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_3)->island_args_0 = island_args_3;
    threadgroup array<uint, int(1024)> g_parents_3;
    (&kernelContext_3)->g_parents_0 = &g_parents_3;
    threadgroup array<uint, int(1024)> g_chunks_3;
    (&kernelContext_3)->g_chunks_0 = &g_chunks_3;
    uint p_1 = group_0.x;
    if(p_1 >= (*(control_3+1U)))
    {
        return;
    }
    uint4 _S14 = island_range_0((uint4(*((&kernelContext_3)->parents_0+p_1)) ).x, &kernelContext_3);
    uint t_0 = thread_1.x;
    uint _S15 = _S14.x;
    uint k_1 = (&kernelContext_3)->params_0->chunk_words_0 * _S15 + t_0;
    for(;;)
    {
        uint _S16 = (&kernelContext_3)->params_0->chunk_words_0;
        uint _S17 = _S14.y;
        _S12 = _S17;
        if(k_1 < (_S16 * _S17))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_3)->old_chunks_0+k_1) = packed_float4(float4(*((&kernelContext_3)->chunks_0+k_1)) ) ;
        k_1 = k_1 + 256U;
    }
    k_1 = 4U * _S15 + t_0;
    for(;;)
    {
        if(k_1 < (4U * _S12))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_3)->old_state_0+k_1) = packed_float4(float4(*((&kernelContext_3)->state_0+k_1)) ) ;
        k_1 = k_1 + 256U;
    }
    uint _S18 = _S14.z;
    k_1 = (&kernelContext_3)->params_0->bond_words_0 * _S18 + t_0;
    for(;;)
    {
        uint _S19 = (&kernelContext_3)->params_0->bond_words_0;
        uint _S20 = _S14.w;
        _S13 = _S20;
        if(k_1 < (_S19 * _S20))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_3)->old_bonds_0+k_1) = packed_float4(float4(*((&kernelContext_3)->bonds_0+k_1)) ) ;
        k_1 = k_1 + 256U;
    }
    k_1 = (&kernelContext_3)->params_0->dyn_words_0 * _S18 + t_0;
    for(;;)
    {
        if(k_1 < ((&kernelContext_3)->params_0->dyn_words_0 * _S13))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_3)->old_dyn_0+k_1) = packed_float4(float4(*((&kernelContext_3)->bond_dyn_0+k_1)) ) ;
        k_1 = k_1 + 256U;
    }
    if(((&kernelContext_3)->params_0->resident_0) == 0U)
    {
        return;
    }
    uint c_0 = _S15 + t_0;
    for(;;)
    {
        if(c_0 < _S12)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_3)->old_ctab_0+((&kernelContext_3)->params_0->props_base_0 + c_0)) = packed_float4(float4(*((&kernelContext_3)->ctab_0+((&kernelContext_3)->params_0->props_base_0 + c_0))) ) ;
        uint _S21 = 2U * c_0;
        *((&kernelContext_3)->old_ctab_0+((&kernelContext_3)->params_0->eload_base_0 + _S21)) = packed_float4(float4(*((&kernelContext_3)->ctab_0+((&kernelContext_3)->params_0->eload_base_0 + _S21))) ) ;
        *((&kernelContext_3)->old_ctab_0+((&kernelContext_3)->params_0->eload_base_0 + _S21 + 1U)) = packed_float4(float4(*((&kernelContext_3)->ctab_0+((&kernelContext_3)->params_0->eload_base_0 + _S21 + 1U))) ) ;
        *((&kernelContext_3)->old_ctab_0+((&kernelContext_3)->params_0->pulse_count_base_0 + c_0)) = packed_float4(float4(*((&kernelContext_3)->ctab_0+((&kernelContext_3)->params_0->pulse_count_base_0 + c_0))) ) ;
        c_0 = c_0 + 256U;
    }
    k_1 = 3U * (&kernelContext_3)->params_0->pulse_cap_0 * _S15 + t_0;
    for(;;)
    {
        if(k_1 < (3U * (&kernelContext_3)->params_0->pulse_cap_0 * _S12))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_3)->old_ctab_0+((&kernelContext_3)->params_0->pulse_base_0 + k_1)) = packed_float4(float4(*((&kernelContext_3)->ctab_0+((&kernelContext_3)->params_0->pulse_base_0 + k_1))) ) ;
        k_1 = k_1 + 256U;
    }
    return;
}

[[kernel]] void pass_finish(uint3 id_1 [[thread_position_in_grid]], uint device* counters_4 [[buffer(9)]], uint device* status_4 [[buffer(1)]], PassParams_0 constant* params_4 [[buffer(0)]], packed_float4 device* islands_4 [[buffer(2)]], packed_uint4 device* parents_4 [[buffer(5)]], packed_float4 device* ctab_4 [[buffer(3)]], packed_uint4 device* islands_of_4 [[buffer(6)]], packed_float4 device* parent_bodies_4 [[buffer(7)]], uint device* control_4 [[buffer(4)]], packed_uint4 device* args_4 [[buffer(8)]], packed_float4 device* old_islands_4 [[buffer(22)]], uint device* old_index_4 [[buffer(21)]], uint device* index_4 [[buffer(20)]], packed_float4 device* old_ctab_4 [[buffer(18)]], packed_float4 device* old_copy_4 [[buffer(24)]], packed_float4 device* island_copy_4 [[buffer(23)]], packed_float4 device* old_chunks_4 [[buffer(11)]], packed_float4 device* chunks_4 [[buffer(10)]], packed_float4 device* old_state_4 [[buffer(13)]], packed_float4 device* state_4 [[buffer(12)]], packed_float4 device* old_bonds_4 [[buffer(15)]], packed_float4 device* bonds_4 [[buffer(14)]], packed_float4 device* old_dyn_4 [[buffer(17)]], packed_float4 device* bond_dyn_4 [[buffer(16)]], packed_uint4 device* island_args_4 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->counters_0 = counters_4;
    (&kernelContext_4)->status_0 = status_4;
    (&kernelContext_4)->params_0 = params_4;
    (&kernelContext_4)->islands_0 = islands_4;
    (&kernelContext_4)->parents_0 = parents_4;
    (&kernelContext_4)->ctab_0 = ctab_4;
    (&kernelContext_4)->islands_of_0 = islands_of_4;
    (&kernelContext_4)->parent_bodies_0 = parent_bodies_4;
    (&kernelContext_4)->control_0 = control_4;
    (&kernelContext_4)->args_0 = args_4;
    (&kernelContext_4)->old_islands_0 = old_islands_4;
    (&kernelContext_4)->old_index_0 = old_index_4;
    (&kernelContext_4)->index_0 = index_4;
    (&kernelContext_4)->old_ctab_0 = old_ctab_4;
    (&kernelContext_4)->old_copy_0 = old_copy_4;
    (&kernelContext_4)->island_copy_0 = island_copy_4;
    (&kernelContext_4)->old_chunks_0 = old_chunks_4;
    (&kernelContext_4)->chunks_0 = chunks_4;
    (&kernelContext_4)->old_state_0 = old_state_4;
    (&kernelContext_4)->state_0 = state_4;
    (&kernelContext_4)->old_bonds_0 = old_bonds_4;
    (&kernelContext_4)->bonds_0 = bonds_4;
    (&kernelContext_4)->old_dyn_0 = old_dyn_4;
    (&kernelContext_4)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_4)->island_args_0 = island_args_4;
    threadgroup array<uint, int(1024)> g_parents_4;
    (&kernelContext_4)->g_parents_0 = &g_parents_4;
    threadgroup array<uint, int(1024)> g_chunks_4;
    (&kernelContext_4)->g_chunks_0 = &g_chunks_4;
    bool _S22;
    if((*(control_4+1U)) == 0U)
    {
        _S22 = true;
    }
    else
    {
        _S22 = (*((&kernelContext_4)->control_0+0U)) != 0U;
    }
    if(_S22)
    {
        return;
    }
    uint device* _S23 = (&kernelContext_4)->control_0+7U;
    uint live_0 = *_S23;
    *((&kernelContext_4)->counters_0+int(0)) = *_S23;
    *((&kernelContext_4)->counters_0+int(1)) = *((&kernelContext_4)->control_0+8U);
    *((&kernelContext_4)->counters_0+int(2)) = *((&kernelContext_4)->control_0+9U);
    *((&kernelContext_4)->counters_0+int(3)) = *((&kernelContext_4)->control_0+10U);
    uint _S24 = max(live_0, 1U);
    *((&kernelContext_4)->island_args_0+int(0)) = packed_uint4(uint4(_S24, 1U, 1U, 0U)) ;
    *((&kernelContext_4)->island_args_0+int(1)) = packed_uint4(uint4((_S24 + 63U) / 64U, 1U, 1U, 0U)) ;
    if(((&kernelContext_4)->params_0->resume_0) != 0U)
    {
        *((&kernelContext_4)->status_0+0U) = 1U;
        *((&kernelContext_4)->status_0+1U) = 0U;
    }
    return;
}

