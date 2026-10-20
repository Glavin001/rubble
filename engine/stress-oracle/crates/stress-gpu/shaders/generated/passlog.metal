#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint group_uniform_0(uint value_0, uint thread_0)
{
    return value_0;
}

struct LogEntry_0
{
    uint n_0;
    uint total_0;
    bool aborted_0;
    uint first_body_0;
    uint first_collider_0;
    uint nb_0;
    uint nc_0;
};

struct LogParams_0
{
    uint log_words_0;
    uint pad0_0;
    uint pad1_0;
    uint pad2_0;
};

struct KernelContext_0
{
    uint device* control_0;
    packed_uint4 device* parents_0;
    uint device* log_0;
    LogParams_0 constant* params_0;
    uint device* status_0;
    packed_float4 device* old_islands_0;
    packed_float4 device* old_chunks_0;
    packed_float4 device* children_0;
    packed_uint4 device* slots_0;
    packed_float4 device* extras_0;
    uint device* work_0;
    packed_float4 device* bodies_0;
    packed_float4 device* colliders_0;
    uint threadgroup* g_children_0;
    uint threadgroup* g_size_0;
    uint threadgroup* g_at_0;
};

LogEntry_0 log_entry_0(uint n_1, KernelContext_0 thread* kernelContext_0)
{
    thread LogEntry_0 e_0;
    (&e_0)->n_0 = n_1;
    (&e_0)->total_0 = kernelContext_0->control_0[6U];
    bool _S1 = kernelContext_0->control_0[0U] != 0U;
    (&e_0)->aborted_0 = _S1;
    (&e_0)->first_body_0 = kernelContext_0->control_0[4U];
    (&e_0)->first_collider_0 = kernelContext_0->control_0[5U];
    uint _S2;
    if(_S1)
    {
        _S2 = 0U;
    }
    else
    {
        _S2 = kernelContext_0->control_0[8U] - (&e_0)->first_body_0;
    }
    (&e_0)->nb_0 = _S2;
    if((&e_0)->aborted_0)
    {
        _S2 = 0U;
    }
    else
    {
        _S2 = kernelContext_0->control_0[9U] - (&e_0)->first_collider_0;
    }
    (&e_0)->nc_0 = _S2;
    return e_0;
}

void log_place_0(const LogEntry_0 thread* e_1, uint tid_0, bool mark_0, KernelContext_0 thread* kernelContext_1)
{
    uint _S3;
    if(tid_0 == 0U)
    {
        uint p_0 = 0U;
        uint c_0 = 0U;
        for(;;)
        {
            _S3 = e_1->n_0;
            if(p_0 < (e_1->n_0))
            {
            }
            else
            {
                break;
            }
            uint c_1 = c_0 + (uint4(*(kernelContext_1->parents_0+p_0)) ).z;
            p_0 = p_0 + 1U;
            c_0 = c_1;
        }
        uint size_0 = 16U + 4U * _S3 + 14U * c_0 + e_1->total_0 + 28U * e_1->nb_0 + 16U * e_1->nc_0;
        uint _S4 = max(*(kernelContext_1->log_0+int(0)), 16U);
        *kernelContext_1->g_children_0 = c_0;
        if((_S4 + size_0) > (kernelContext_1->params_0->log_words_0))
        {
            if(mark_0)
            {
                *(kernelContext_1->log_0+int(2)) = 1U;
            }
            *kernelContext_1->g_size_0 = 0U;
        }
        else
        {
            *kernelContext_1->g_size_0 = size_0;
        }
        *kernelContext_1->g_at_0 = _S4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
}

void log_head_0(const LogEntry_0 thread* e_2, uint at_0, uint size_1, uint c_2, uint tid_1, KernelContext_0 thread* kernelContext_2)
{
    uint _S5 = e_2->n_0;
    uint _S6 = e_2->total_0;
    bool _S7 = e_2->aborted_0;
    uint _S8 = e_2->first_body_0;
    uint _S9 = e_2->first_collider_0;
    uint _S10 = e_2->nb_0;
    uint _S11 = e_2->nc_0;
    uint j_0;
    uint p_1;
    if(tid_1 == 0U)
    {
        *(kernelContext_2->log_0+at_0) = size_1;
        *(kernelContext_2->log_0+(at_0 + 1U)) = _S5;
        *(kernelContext_2->log_0+(at_0 + 2U)) = c_2;
        uint device* _S12 = kernelContext_2->log_0+(at_0 + 3U);
        if(_S7)
        {
            j_0 = 1U;
        }
        else
        {
            j_0 = 0U;
        }
        *_S12 = j_0;
        *(kernelContext_2->log_0+(at_0 + 4U)) = kernelContext_2->status_0[int(1)] - 1U;
        *(kernelContext_2->log_0+(at_0 + 5U)) = _S8;
        *(kernelContext_2->log_0+(at_0 + 6U)) = _S9;
        *(kernelContext_2->log_0+(at_0 + 7U)) = _S8 + _S10;
        *(kernelContext_2->log_0+(at_0 + 8U)) = _S9 + _S11;
        *(kernelContext_2->log_0+(at_0 + 9U)) = kernelContext_2->control_0[2U];
        *(kernelContext_2->log_0+(at_0 + 10U)) = kernelContext_2->control_0[3U];
        *(kernelContext_2->log_0+(at_0 + 11U)) = _S6;
        j_0 = 12U;
        for(;;)
        {
            if(j_0 < 16U)
            {
            }
            else
            {
                break;
            }
            *(kernelContext_2->log_0+(at_0 + j_0)) = 0U;
            j_0 = j_0 + 1U;
        }
        uint _S13 = at_0 + 16U;
        uint _S14 = _S13 + 4U * _S5;
        p_1 = 0U;
        uint h_0 = _S14;
        for(;;)
        {
            if(p_1 < _S5)
            {
            }
            else
            {
                break;
            }
            uint4 _S15 = uint4(*(kernelContext_2->parents_0+p_1)) ;
            uint _S16 = _S15.x;
            uint _S17 = 25U * _S16;
            uint4 info_0 = (as_type<uint4>((float4(*(kernelContext_2->old_chunks_0+(16U * (as_type<uint4>((float4(*(kernelContext_2->old_islands_0+_S17)) ))).x + 8U))) )));
            uint _S18 = _S13 + 4U * p_1;
            *(kernelContext_2->log_0+_S18) = _S16;
            uint _S19 = _S15.z;
            *(kernelContext_2->log_0+(_S18 + 1U)) = _S19;
            *(kernelContext_2->log_0+(_S18 + 2U)) = (as_type<uint4>((float4(*(kernelContext_2->old_islands_0+(_S17 + 19U))) ))).w;
            *(kernelContext_2->log_0+(_S18 + 3U)) = info_0.z;
            uint k_0 = 0U;
            for(;;)
            {
                if(k_0 < _S19)
                {
                }
                else
                {
                    break;
                }
                uint slot_0 = _S15.y + k_0;
                float4 _S20 = float4(*(kernelContext_2->children_0+8U * slot_0)) ;
                uint4 s_0;
                if(_S7)
                {
                    s_0 = uint4(int4(int(0)) );
                }
                else
                {
                    s_0 = uint4(*(kernelContext_2->slots_0+slot_0)) ;
                }
                uint _S21 = 2U * slot_0;
                float4 _S22 = float4(*(kernelContext_2->extras_0+_S21)) ;
                float4 _S23 = float4(*(kernelContext_2->extras_0+(_S21 + 1U))) ;
                *(kernelContext_2->log_0+h_0) = (as_type<uint>((_S20.z)));
                *(kernelContext_2->log_0+(h_0 + 1U)) = (as_type<uint>((_S20.w)));
                *(kernelContext_2->log_0+(h_0 + 2U)) = s_0.x;
                *(kernelContext_2->log_0+(h_0 + 3U)) = s_0.y;
                *(kernelContext_2->log_0+(h_0 + 4U)) = s_0.z;
                *(kernelContext_2->log_0+(h_0 + 5U)) = s_0.w;
                *(kernelContext_2->log_0+(h_0 + 6U)) = (as_type<uint>((_S22.x)));
                *(kernelContext_2->log_0+(h_0 + 7U)) = (as_type<uint>((_S22.y)));
                *(kernelContext_2->log_0+(h_0 + 8U)) = (as_type<uint>((_S22.z)));
                *(kernelContext_2->log_0+(h_0 + 9U)) = (as_type<uint>((_S22.w)));
                *(kernelContext_2->log_0+(h_0 + 10U)) = (as_type<uint>((_S23.x)));
                *(kernelContext_2->log_0+(h_0 + 11U)) = (as_type<uint>((_S23.y)));
                *(kernelContext_2->log_0+(h_0 + 12U)) = (as_type<uint>((_S23.z)));
                *(kernelContext_2->log_0+(h_0 + 13U)) = (as_type<uint>((_S23.w)));
                uint h_1 = h_0 + 14U;
                k_0 = k_0 + 1U;
                h_0 = h_1;
            }
            p_1 = p_1 + 1U;
        }
    }
    uint _S24 = at_0 + 16U + 4U * _S5 + 14U * c_2;
    j_0 = tid_1;
    for(;;)
    {
        if(j_0 < _S6)
        {
        }
        else
        {
            break;
        }
        p_1 = 0U;
        for(;;)
        {
            uint _S25 = p_1 + 1U;
            bool _S26;
            if(_S25 < _S5)
            {
                _S26 = ((uint4(*(kernelContext_2->parents_0+_S25)) ).y) <= j_0;
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
            p_1 = _S25;
        }
        *(kernelContext_2->log_0+(_S24 + j_0)) = (as_type<uint4>((float4(*(kernelContext_2->old_chunks_0+(16U * ((as_type<uint4>((float4(*(kernelContext_2->old_islands_0+25U * (uint4(*(kernelContext_2->parents_0+p_1)) ).x)) ))).x + kernelContext_2->work_0[_S6 + j_0]) + 8U))) ))).w;
        j_0 = j_0 + 256U;
    }
    return;
}

void log_engine_0(const LogEntry_0 thread* e_3, uint at_1, uint size_2, uint c_3, uint tid_2, KernelContext_0 thread* kernelContext_3)
{
    uint _S27;
    uint b_0 = at_1 + 16U + 4U * e_3->n_0 + 14U * c_3 + e_3->total_0;
    uint j_1 = tid_2;
    for(;;)
    {
        _S27 = e_3->nb_0;
        if(j_1 < (7U * e_3->nb_0))
        {
        }
        else
        {
            break;
        }
        float4 _S28 = float4(*(kernelContext_3->bodies_0+(7U * e_3->first_body_0 + j_1))) ;
        uint _S29 = b_0 + 4U * j_1;
        *(kernelContext_3->log_0+_S29) = (as_type<uint>((_S28.x)));
        *(kernelContext_3->log_0+(_S29 + 1U)) = (as_type<uint>((_S28.y)));
        *(kernelContext_3->log_0+(_S29 + 2U)) = (as_type<uint>((_S28.z)));
        *(kernelContext_3->log_0+(_S29 + 3U)) = (as_type<uint>((_S28.w)));
        j_1 = j_1 + 256U;
    }
    uint _S30 = b_0 + 28U * _S27;
    j_1 = tid_2;
    for(;;)
    {
        if(j_1 < (4U * e_3->nc_0))
        {
        }
        else
        {
            break;
        }
        float4 _S31 = float4(*(kernelContext_3->colliders_0+(4U * e_3->first_collider_0 + j_1))) ;
        uint _S32 = _S30 + 4U * j_1;
        *(kernelContext_3->log_0+_S32) = (as_type<uint>((_S31.x)));
        *(kernelContext_3->log_0+(_S32 + 1U)) = (as_type<uint>((_S31.y)));
        *(kernelContext_3->log_0+(_S32 + 2U)) = (as_type<uint>((_S31.z)));
        *(kernelContext_3->log_0+(_S32 + 3U)) = (as_type<uint>((_S31.w)));
        j_1 = j_1 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid_2 == 0U)
    {
        *(kernelContext_3->log_0+int(0)) = at_1 + size_2;
        *(kernelContext_3->log_0+int(1)) = *(kernelContext_3->log_0+int(1)) + 1U;
    }
    return;
}

[[kernel]] void pass_log(uint3 thread_1 [[thread_position_in_threadgroup]], uint device* control_1 [[buffer(1)]], packed_uint4 device* parents_1 [[buffer(2)]], uint device* log_1 [[buffer(9)]], LogParams_0 constant* params_1 [[buffer(0)]], uint device* status_1 [[buffer(10)]], packed_float4 device* old_islands_1 [[buffer(11)]], packed_float4 device* old_chunks_1 [[buffer(12)]], packed_float4 device* children_1 [[buffer(4)]], packed_uint4 device* slots_1 [[buffer(5)]], packed_float4 device* extras_1 [[buffer(6)]], uint device* work_1 [[buffer(3)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* colliders_1 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->control_0 = control_1;
    (&kernelContext_4)->parents_0 = parents_1;
    (&kernelContext_4)->log_0 = log_1;
    (&kernelContext_4)->params_0 = params_1;
    (&kernelContext_4)->status_0 = status_1;
    (&kernelContext_4)->old_islands_0 = old_islands_1;
    (&kernelContext_4)->old_chunks_0 = old_chunks_1;
    (&kernelContext_4)->children_0 = children_1;
    (&kernelContext_4)->slots_0 = slots_1;
    (&kernelContext_4)->extras_0 = extras_1;
    (&kernelContext_4)->work_0 = work_1;
    (&kernelContext_4)->bodies_0 = bodies_1;
    (&kernelContext_4)->colliders_0 = colliders_1;
    threadgroup uint g_children_1;
    (&kernelContext_4)->g_children_0 = &g_children_1;
    threadgroup uint g_size_1;
    (&kernelContext_4)->g_size_0 = &g_size_1;
    threadgroup uint g_at_1;
    (&kernelContext_4)->g_at_0 = &g_at_1;
    uint tid_3 = thread_1.x;
    uint n_2 = group_uniform_0(control_1[1U], tid_3);
    if(n_2 == 0U)
    {
        return;
    }
    LogEntry_0 _S33 = log_entry_0(n_2, &kernelContext_4);
    thread LogEntry_0 _S34 = _S33;
    log_place_0(&_S34, tid_3, true, &kernelContext_4);
    uint at_2 = *(&kernelContext_4)->g_at_0;
    uint size_3 = group_uniform_0(*(&kernelContext_4)->g_size_0, tid_3);
    uint c_4 = *(&kernelContext_4)->g_children_0;
    if(size_3 == 0U)
    {
        return;
    }
    thread LogEntry_0 _S35 = _S33;
    log_head_0(&_S35, at_2, size_3, c_4, tid_3, &kernelContext_4);
    thread LogEntry_0 _S36 = _S33;
    log_engine_0(&_S36, at_2, size_3, c_4, tid_3, &kernelContext_4);
    return;
}

[[kernel]] void pass_log_head(uint3 thread_2 [[thread_position_in_threadgroup]], uint device* control_2 [[buffer(1)]], packed_uint4 device* parents_2 [[buffer(2)]], uint device* log_2 [[buffer(9)]], LogParams_0 constant* params_2 [[buffer(0)]], uint device* status_2 [[buffer(10)]], packed_float4 device* old_islands_2 [[buffer(11)]], packed_float4 device* old_chunks_2 [[buffer(12)]], packed_float4 device* children_2 [[buffer(4)]], packed_uint4 device* slots_2 [[buffer(5)]], packed_float4 device* extras_2 [[buffer(6)]], uint device* work_2 [[buffer(3)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* colliders_2 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->control_0 = control_2;
    (&kernelContext_5)->parents_0 = parents_2;
    (&kernelContext_5)->log_0 = log_2;
    (&kernelContext_5)->params_0 = params_2;
    (&kernelContext_5)->status_0 = status_2;
    (&kernelContext_5)->old_islands_0 = old_islands_2;
    (&kernelContext_5)->old_chunks_0 = old_chunks_2;
    (&kernelContext_5)->children_0 = children_2;
    (&kernelContext_5)->slots_0 = slots_2;
    (&kernelContext_5)->extras_0 = extras_2;
    (&kernelContext_5)->work_0 = work_2;
    (&kernelContext_5)->bodies_0 = bodies_2;
    (&kernelContext_5)->colliders_0 = colliders_2;
    threadgroup uint g_children_2;
    (&kernelContext_5)->g_children_0 = &g_children_2;
    threadgroup uint g_size_2;
    (&kernelContext_5)->g_size_0 = &g_size_2;
    threadgroup uint g_at_2;
    (&kernelContext_5)->g_at_0 = &g_at_2;
    uint tid_4 = thread_2.x;
    uint n_3 = group_uniform_0(control_2[1U], tid_4);
    if(n_3 == 0U)
    {
        return;
    }
    LogEntry_0 _S37 = log_entry_0(n_3, &kernelContext_5);
    thread LogEntry_0 _S38 = _S37;
    log_place_0(&_S38, tid_4, true, &kernelContext_5);
    uint at_3 = *(&kernelContext_5)->g_at_0;
    uint size_4 = group_uniform_0(*(&kernelContext_5)->g_size_0, tid_4);
    uint c_5 = *(&kernelContext_5)->g_children_0;
    if(size_4 == 0U)
    {
        return;
    }
    thread LogEntry_0 _S39 = _S37;
    log_head_0(&_S39, at_3, size_4, c_5, tid_4, &kernelContext_5);
    return;
}

[[kernel]] void pass_log_tail(uint3 thread_3 [[thread_position_in_threadgroup]], uint device* control_3 [[buffer(1)]], packed_uint4 device* parents_3 [[buffer(2)]], uint device* log_3 [[buffer(9)]], LogParams_0 constant* params_3 [[buffer(0)]], uint device* status_3 [[buffer(10)]], packed_float4 device* old_islands_3 [[buffer(11)]], packed_float4 device* old_chunks_3 [[buffer(12)]], packed_float4 device* children_3 [[buffer(4)]], packed_uint4 device* slots_3 [[buffer(5)]], packed_float4 device* extras_3 [[buffer(6)]], uint device* work_3 [[buffer(3)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* colliders_3 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->control_0 = control_3;
    (&kernelContext_6)->parents_0 = parents_3;
    (&kernelContext_6)->log_0 = log_3;
    (&kernelContext_6)->params_0 = params_3;
    (&kernelContext_6)->status_0 = status_3;
    (&kernelContext_6)->old_islands_0 = old_islands_3;
    (&kernelContext_6)->old_chunks_0 = old_chunks_3;
    (&kernelContext_6)->children_0 = children_3;
    (&kernelContext_6)->slots_0 = slots_3;
    (&kernelContext_6)->extras_0 = extras_3;
    (&kernelContext_6)->work_0 = work_3;
    (&kernelContext_6)->bodies_0 = bodies_3;
    (&kernelContext_6)->colliders_0 = colliders_3;
    threadgroup uint g_children_3;
    (&kernelContext_6)->g_children_0 = &g_children_3;
    threadgroup uint g_size_3;
    (&kernelContext_6)->g_size_0 = &g_size_3;
    threadgroup uint g_at_3;
    (&kernelContext_6)->g_at_0 = &g_at_3;
    uint tid_5 = thread_3.x;
    uint n_4 = group_uniform_0(control_3[1U], tid_5);
    if(n_4 == 0U)
    {
        return;
    }
    LogEntry_0 _S40 = log_entry_0(n_4, &kernelContext_6);
    thread LogEntry_0 _S41 = _S40;
    log_place_0(&_S41, tid_5, false, &kernelContext_6);
    uint at_4 = *(&kernelContext_6)->g_at_0;
    uint size_5 = group_uniform_0(*(&kernelContext_6)->g_size_0, tid_5);
    uint c_6 = *(&kernelContext_6)->g_children_0;
    if(size_5 == 0U)
    {
        return;
    }
    thread LogEntry_0 _S42 = _S40;
    log_engine_0(&_S42, at_4, size_5, c_6, tid_5, &kernelContext_6);
    return;
}

