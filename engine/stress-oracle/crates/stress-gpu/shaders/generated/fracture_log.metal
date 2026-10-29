#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct LogParams_0
{
    uint chunk_count_0;
    uint bond_count_0;
    uint body_count_0;
    uint step_0;
    uint w_alive_0;
    uint cap_bonds_0;
    uint cap_chunks_0;
    uint cap_frames_0;
};

struct KernelContext_0
{
    LogParams_0 constant* params_0;
    uint device* work_0;
    uint device* track_0;
    atomic<uint> device* log_0;
    packed_float4 device* frames_0;
};

bool claim_0(uint s_0, uint cap_0, uint thread* at_0, KernelContext_0 thread* kernelContext_0)
{
    uint i_0 = atomic_fetch_add_explicit(kernelContext_0->log_0+s_0, 1U, memory_order_relaxed);
    uint acked_0 = atomic_load_explicit(kernelContext_0->log_0+(4U + s_0), memory_order_relaxed);
    uint _S1 = i_0 % cap_0;
    *at_0 = _S1;
    if((i_0 - acked_0) >= cap_0)
    {
        atomic_store_explicit(kernelContext_0->log_0+int(3), 1U, memory_order_relaxed);
        return false;
    }
    return true;
}

uint l_bonds_0()
{
    return 8U;
}

[[kernel]] void log_bonds(uint3 id_0 [[thread_position_in_grid]], LogParams_0 constant* params_1 [[buffer(0)]], uint device* work_1 [[buffer(1)]], uint device* track_1 [[buffer(3)]], atomic<uint> device* log_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_1;
    (&kernelContext_1)->work_0 = work_1;
    (&kernelContext_1)->track_0 = track_1;
    (&kernelContext_1)->log_0 = log_1;
    (&kernelContext_1)->frames_0 = frames_1;
    uint j_0 = id_0.x;
    if(j_0 >= (params_1->bond_count_0))
    {
        return;
    }
    uint alive_0;
    if((&kernelContext_1)->work_0[(&kernelContext_1)->params_0->w_alive_0 + j_0] != 0U)
    {
        alive_0 = 1U;
    }
    else
    {
        alive_0 = 0U;
    }
    if((*((&kernelContext_1)->track_0+j_0)) == alive_0)
    {
        return;
    }
    *((&kernelContext_1)->track_0+j_0) = alive_0;
    thread uint at_1;
    bool _S2;
    if(alive_0 == 0U)
    {
        bool _S3 = claim_0(0U, (&kernelContext_1)->params_0->cap_bonds_0, &at_1, &kernelContext_1);
        _S2 = _S3;
    }
    else
    {
        _S2 = false;
    }
    if(_S2)
    {
        uint _S4 = l_bonds_0();
        atomic_store_explicit((&kernelContext_1)->log_0+(_S4 + 2U * at_1), j_0, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_1)->log_0+(_S4 + 2U * at_1 + 1U), (&kernelContext_1)->params_0->step_0, memory_order_relaxed);
    }
    return;
}

uint t_body_0(KernelContext_0 thread* kernelContext_2)
{
    return kernelContext_2->params_0->bond_count_0;
}

uint l_chunks_0(KernelContext_0 thread* kernelContext_3)
{
    return l_bonds_0() + 2U * kernelContext_3->params_0->cap_bonds_0;
}

[[kernel]] void log_chunks(uint3 id_1 [[thread_position_in_grid]], LogParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(1)]], uint device* track_2 [[buffer(3)]], atomic<uint> device* log_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->params_0 = params_2;
    (&kernelContext_4)->work_0 = work_2;
    (&kernelContext_4)->track_0 = track_2;
    (&kernelContext_4)->log_0 = log_2;
    (&kernelContext_4)->frames_0 = frames_2;
    uint c_0 = id_1.x;
    if(c_0 >= (params_2->chunk_count_0))
    {
        return;
    }
    uint body_0 = (&kernelContext_4)->work_0[c_0];
    uint _S5 = t_body_0(&kernelContext_4);
    uint _S6 = _S5 + c_0;
    if((*((&kernelContext_4)->track_0+_S6)) == body_0)
    {
        return;
    }
    *((&kernelContext_4)->track_0+_S6) = body_0;
    thread uint at_2;
    bool _S7 = claim_0(1U, (&kernelContext_4)->params_0->cap_chunks_0, &at_2, &kernelContext_4);
    if(_S7)
    {
        uint _S8 = l_chunks_0(&kernelContext_4);
        atomic_store_explicit((&kernelContext_4)->log_0+(_S8 + 3U * at_2), c_0, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_4)->log_0+(_S8 + 3U * at_2 + 1U), body_0, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_4)->log_0+(_S8 + 3U * at_2 + 2U), (&kernelContext_4)->params_0->step_0, memory_order_relaxed);
    }
    return;
}

uint t_frame_0(KernelContext_0 thread* kernelContext_5)
{
    return kernelContext_5->params_0->bond_count_0 + kernelContext_5->params_0->chunk_count_0;
}

uint l_frames_0(KernelContext_0 thread* kernelContext_6)
{
    uint _S9 = l_chunks_0(kernelContext_6);
    return _S9 + 3U * kernelContext_6->params_0->cap_chunks_0;
}

[[kernel]] void log_frames(uint3 id_2 [[thread_position_in_grid]], LogParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(1)]], uint device* track_3 [[buffer(3)]], atomic<uint> device* log_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_7;
    (&kernelContext_7)->params_0 = params_3;
    (&kernelContext_7)->work_0 = work_3;
    (&kernelContext_7)->track_0 = track_3;
    (&kernelContext_7)->log_0 = log_3;
    (&kernelContext_7)->frames_0 = frames_3;
    uint b_0 = id_2.x;
    if(b_0 >= (params_3->body_count_0))
    {
        return;
    }
    uint _S10 = 2U * b_0;
    uint4 f0_0 = (as_type<uint4>((float4(*((&kernelContext_7)->frames_0+_S10)) )));
    uint4 f1_0 = (as_type<uint4>((float4(*((&kernelContext_7)->frames_0+(_S10 + 1U))) )));
    uint _S11 = t_frame_0(&kernelContext_7);
    uint t_0 = _S11 + 8U * b_0;
    bool same_0;
    if(all(uint4(*((&kernelContext_7)->track_0+t_0), *((&kernelContext_7)->track_0+(t_0 + 1U)), *((&kernelContext_7)->track_0+(t_0 + 2U)), *((&kernelContext_7)->track_0+(t_0 + 3U))) == f0_0))
    {
        same_0 = all(uint4(*((&kernelContext_7)->track_0+(t_0 + 4U)), *((&kernelContext_7)->track_0+(t_0 + 5U)), *((&kernelContext_7)->track_0+(t_0 + 6U)), *((&kernelContext_7)->track_0+(t_0 + 7U))) == f1_0);
    }
    else
    {
        same_0 = false;
    }
    if(same_0)
    {
        return;
    }
    uint w_0 = 0U;
    for(;;)
    {
        if(w_0 < 4U)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_7)->track_0+(t_0 + w_0)) = f0_0[w_0];
        *((&kernelContext_7)->track_0+(t_0 + 4U + w_0)) = f1_0[w_0];
        w_0 = w_0 + 1U;
    }
    thread uint at_3;
    bool _S12 = claim_0(2U, (&kernelContext_7)->params_0->cap_frames_0, &at_3, &kernelContext_7);
    if(_S12)
    {
        uint _S13 = l_frames_0(&kernelContext_7);
        uint e_0 = _S13 + 10U * at_3;
        atomic_store_explicit((&kernelContext_7)->log_0+e_0, b_0, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_7)->log_0+(e_0 + 1U), (&kernelContext_7)->params_0->step_0, memory_order_relaxed);
        w_0 = 0U;
        for(;;)
        {
            if(w_0 < 4U)
            {
            }
            else
            {
                break;
            }
            atomic_store_explicit((&kernelContext_7)->log_0+(e_0 + 2U + w_0), f0_0[w_0], memory_order_relaxed);
            atomic_store_explicit((&kernelContext_7)->log_0+(e_0 + 6U + w_0), f1_0[w_0], memory_order_relaxed);
            w_0 = w_0 + 1U;
        }
    }
    return;
}

