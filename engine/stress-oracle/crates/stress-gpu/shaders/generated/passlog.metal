#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
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

[[kernel]] void pass_log(uint3 thread_0 [[thread_position_in_threadgroup]], uint device* control_1 [[buffer(1)]], packed_uint4 device* parents_1 [[buffer(2)]], uint device* log_1 [[buffer(9)]], LogParams_0 constant* params_1 [[buffer(0)]], uint device* status_1 [[buffer(10)]], packed_float4 device* children_1 [[buffer(4)]], packed_uint4 device* slots_1 [[buffer(5)]], packed_float4 device* extras_1 [[buffer(6)]], uint device* work_1 [[buffer(3)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* colliders_1 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->control_0 = control_1;
    (&kernelContext_0)->parents_0 = parents_1;
    (&kernelContext_0)->log_0 = log_1;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->status_0 = status_1;
    (&kernelContext_0)->children_0 = children_1;
    (&kernelContext_0)->slots_0 = slots_1;
    (&kernelContext_0)->extras_0 = extras_1;
    (&kernelContext_0)->work_0 = work_1;
    (&kernelContext_0)->bodies_0 = bodies_1;
    (&kernelContext_0)->colliders_0 = colliders_1;
    threadgroup uint g_children_1;
    (&kernelContext_0)->g_children_0 = &g_children_1;
    threadgroup uint g_size_1;
    (&kernelContext_0)->g_size_0 = &g_size_1;
    threadgroup uint g_at_1;
    (&kernelContext_0)->g_at_0 = &g_at_1;
    uint tid_0 = thread_0.x;
    uint n_0 = control_1[1U];
    if(n_0 == 0U)
    {
        return;
    }
    uint total_0 = (&kernelContext_0)->control_0[6U];
    bool aborted_0 = (&kernelContext_0)->control_0[0U] != 0U;
    uint first_body_0 = (&kernelContext_0)->control_0[4U];
    uint first_collider_0 = (&kernelContext_0)->control_0[5U];
    uint nb_0;
    if(aborted_0)
    {
        nb_0 = 0U;
    }
    else
    {
        nb_0 = (&kernelContext_0)->control_0[8U] - first_body_0;
    }
    uint nc_0;
    if(aborted_0)
    {
        nc_0 = 0U;
    }
    else
    {
        nc_0 = (&kernelContext_0)->control_0[9U] - first_collider_0;
    }
    bool _S1 = tid_0 == 0U;
    uint j_0;
    uint p_0;
    if(_S1)
    {
        j_0 = 0U;
        p_0 = 0U;
        for(;;)
        {
            if(j_0 < n_0)
            {
            }
            else
            {
                break;
            }
            uint c_0 = p_0 + (uint4(*((&kernelContext_0)->parents_0+j_0)) ).z;
            j_0 = j_0 + 1U;
            p_0 = c_0;
        }
        uint size_0 = 16U + 2U * n_0 + 14U * p_0 + total_0 + 28U * nb_0 + 16U * nc_0;
        uint _S2 = max(*((&kernelContext_0)->log_0+int(0)), 16U);
        *(&kernelContext_0)->g_children_0 = p_0;
        if((_S2 + size_0) > ((&kernelContext_0)->params_0->log_words_0))
        {
            *((&kernelContext_0)->log_0+int(2)) = 1U;
            *(&kernelContext_0)->g_size_0 = 0U;
        }
        else
        {
            *(&kernelContext_0)->g_size_0 = size_0;
        }
        *(&kernelContext_0)->g_at_0 = _S2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at_0 = *(&kernelContext_0)->g_at_0;
    uint threadgroup* _S3 = (&kernelContext_0)->g_size_0;
    uint size_1 = *_S3;
    uint c_1 = *(&kernelContext_0)->g_children_0;
    if((*_S3) == 0U)
    {
        return;
    }
    if(_S1)
    {
        *((&kernelContext_0)->log_0+at_0) = size_1;
        *((&kernelContext_0)->log_0+(at_0 + 1U)) = n_0;
        *((&kernelContext_0)->log_0+(at_0 + 2U)) = c_1;
        uint device* _S4 = (&kernelContext_0)->log_0+(at_0 + 3U);
        if(aborted_0)
        {
            j_0 = 1U;
        }
        else
        {
            j_0 = 0U;
        }
        *_S4 = j_0;
        *((&kernelContext_0)->log_0+(at_0 + 4U)) = (&kernelContext_0)->status_0[int(1)] - 1U;
        *((&kernelContext_0)->log_0+(at_0 + 5U)) = first_body_0;
        *((&kernelContext_0)->log_0+(at_0 + 6U)) = first_collider_0;
        *((&kernelContext_0)->log_0+(at_0 + 7U)) = first_body_0 + nb_0;
        *((&kernelContext_0)->log_0+(at_0 + 8U)) = first_collider_0 + nc_0;
        *((&kernelContext_0)->log_0+(at_0 + 9U)) = (&kernelContext_0)->control_0[2U];
        *((&kernelContext_0)->log_0+(at_0 + 10U)) = (&kernelContext_0)->control_0[3U];
        *((&kernelContext_0)->log_0+(at_0 + 11U)) = total_0;
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
            *((&kernelContext_0)->log_0+(at_0 + j_0)) = 0U;
            j_0 = j_0 + 1U;
        }
        uint _S5 = at_0 + 16U;
        uint _S6 = _S5 + 2U * n_0;
        p_0 = 0U;
        uint h_0 = _S6;
        for(;;)
        {
            if(p_0 < n_0)
            {
            }
            else
            {
                break;
            }
            uint4 _S7 = uint4(*((&kernelContext_0)->parents_0+p_0)) ;
            uint _S8 = _S5 + 2U * p_0;
            *((&kernelContext_0)->log_0+_S8) = _S7.x;
            uint _S9 = _S7.z;
            *((&kernelContext_0)->log_0+(_S8 + 1U)) = _S9;
            uint k_0 = 0U;
            for(;;)
            {
                if(k_0 < _S9)
                {
                }
                else
                {
                    break;
                }
                uint slot_0 = _S7.y + k_0;
                float4 _S10 = float4(*((&kernelContext_0)->children_0+8U * slot_0)) ;
                uint4 s_0;
                if(aborted_0)
                {
                    s_0 = uint4(int4(int(0)) );
                }
                else
                {
                    s_0 = uint4(*((&kernelContext_0)->slots_0+slot_0)) ;
                }
                uint _S11 = 2U * slot_0;
                float4 _S12 = float4(*((&kernelContext_0)->extras_0+_S11)) ;
                float4 _S13 = float4(*((&kernelContext_0)->extras_0+(_S11 + 1U))) ;
                *((&kernelContext_0)->log_0+h_0) = (as_type<uint>((_S10.z)));
                *((&kernelContext_0)->log_0+(h_0 + 1U)) = (as_type<uint>((_S10.w)));
                *((&kernelContext_0)->log_0+(h_0 + 2U)) = s_0.x;
                *((&kernelContext_0)->log_0+(h_0 + 3U)) = s_0.y;
                *((&kernelContext_0)->log_0+(h_0 + 4U)) = s_0.z;
                *((&kernelContext_0)->log_0+(h_0 + 5U)) = s_0.w;
                *((&kernelContext_0)->log_0+(h_0 + 6U)) = (as_type<uint>((_S12.x)));
                *((&kernelContext_0)->log_0+(h_0 + 7U)) = (as_type<uint>((_S12.y)));
                *((&kernelContext_0)->log_0+(h_0 + 8U)) = (as_type<uint>((_S12.z)));
                *((&kernelContext_0)->log_0+(h_0 + 9U)) = (as_type<uint>((_S12.w)));
                *((&kernelContext_0)->log_0+(h_0 + 10U)) = (as_type<uint>((_S13.x)));
                *((&kernelContext_0)->log_0+(h_0 + 11U)) = (as_type<uint>((_S13.y)));
                *((&kernelContext_0)->log_0+(h_0 + 12U)) = (as_type<uint>((_S13.z)));
                *((&kernelContext_0)->log_0+(h_0 + 13U)) = (as_type<uint>((_S13.w)));
                uint h_1 = h_0 + 14U;
                k_0 = k_0 + 1U;
                h_0 = h_1;
            }
            p_0 = p_0 + 1U;
        }
    }
    uint m_0 = at_0 + 16U + 2U * n_0 + 14U * c_1;
    j_0 = tid_0;
    for(;;)
    {
        if(j_0 < total_0)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_0)->log_0+(m_0 + j_0)) = (&kernelContext_0)->work_0[total_0 + j_0];
        j_0 = j_0 + 256U;
    }
    uint b_0 = m_0 + total_0;
    j_0 = tid_0;
    for(;;)
    {
        if(j_0 < (7U * nb_0))
        {
        }
        else
        {
            break;
        }
        float4 _S14 = float4(*((&kernelContext_0)->bodies_0+(7U * first_body_0 + j_0))) ;
        uint _S15 = b_0 + 4U * j_0;
        *((&kernelContext_0)->log_0+_S15) = (as_type<uint>((_S14.x)));
        *((&kernelContext_0)->log_0+(_S15 + 1U)) = (as_type<uint>((_S14.y)));
        *((&kernelContext_0)->log_0+(_S15 + 2U)) = (as_type<uint>((_S14.z)));
        *((&kernelContext_0)->log_0+(_S15 + 3U)) = (as_type<uint>((_S14.w)));
        j_0 = j_0 + 256U;
    }
    uint _S16 = b_0 + 28U * nb_0;
    j_0 = tid_0;
    for(;;)
    {
        if(j_0 < (4U * nc_0))
        {
        }
        else
        {
            break;
        }
        float4 _S17 = float4(*((&kernelContext_0)->colliders_0+(4U * first_collider_0 + j_0))) ;
        uint _S18 = _S16 + 4U * j_0;
        *((&kernelContext_0)->log_0+_S18) = (as_type<uint>((_S17.x)));
        *((&kernelContext_0)->log_0+(_S18 + 1U)) = (as_type<uint>((_S17.y)));
        *((&kernelContext_0)->log_0+(_S18 + 2U)) = (as_type<uint>((_S17.z)));
        *((&kernelContext_0)->log_0+(_S18 + 3U)) = (as_type<uint>((_S17.w)));
        j_0 = j_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S1)
    {
        *((&kernelContext_0)->log_0+int(0)) = at_0 + size_1;
        *((&kernelContext_0)->log_0+int(1)) = *((&kernelContext_0)->log_0+int(1)) + 1U;
    }
    return;
}

