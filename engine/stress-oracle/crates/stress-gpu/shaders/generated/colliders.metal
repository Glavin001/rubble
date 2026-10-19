#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct ColliderParams_0
{
    uint parent_count_0;
    uint chunk_count_0;
    uint collider_count_0;
    uint chunk_words_0;
    uint island_words_0;
    float ground_modulus_0;
    uint threads_0;
    uint pad1_0;
};

struct KernelContext_0
{
    uint device* control_0;
    ColliderParams_0 constant* params_0;
    packed_uint4 device* parents_0;
    packed_uint4 device* islands_of_0;
    uint device* collider_key_0;
    uint device* work_0;
    packed_float4 device* old_chunks_0;
    uint device* structure_base_0;
    packed_uint4 device* part_ranges_0;
    uint device* member_offsets_0;
    packed_float4 device* chunks_0;
    uint device* chunk_slot_0;
    packed_uint4 device* collider_info_0;
    packed_float4 device* islands_0;
    packed_float4 device* chunk_props_0;
};

uint total_slots_0(KernelContext_0 thread* kernelContext_0)
{
    return (uint4(*(kernelContext_0->parents_0+kernelContext_0->control_0[1U])) ).y;
}

uint key_of_0(uint4 info_0, KernelContext_0 thread* kernelContext_1)
{
    return kernelContext_1->structure_base_0[info_0.z] + info_0.w;
}

[[kernel]] void collider_keys(uint3 id_0 [[thread_position_in_grid]], uint device* control_1 [[buffer(11)]], ColliderParams_0 constant* params_1 [[buffer(0)]], packed_uint4 device* parents_1 [[buffer(1)]], packed_uint4 device* islands_of_1 [[buffer(9)]], uint device* collider_key_1 [[buffer(14)]], uint device* work_1 [[buffer(2)]], packed_float4 device* old_chunks_1 [[buffer(6)]], uint device* structure_base_1 [[buffer(10)]], packed_uint4 device* part_ranges_1 [[buffer(5)]], uint device* member_offsets_1 [[buffer(16)]], packed_float4 device* chunks_1 [[buffer(7)]], uint device* chunk_slot_1 [[buffer(13)]], packed_uint4 device* collider_info_1 [[buffer(15)]], packed_float4 device* islands_1 [[buffer(8)]], packed_float4 device* chunk_props_1 [[buffer(12)]])
{
    uint c_0;
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->control_0 = control_1;
    (&kernelContext_2)->params_0 = params_1;
    (&kernelContext_2)->parents_0 = parents_1;
    (&kernelContext_2)->islands_of_0 = islands_of_1;
    (&kernelContext_2)->collider_key_0 = collider_key_1;
    (&kernelContext_2)->work_0 = work_1;
    (&kernelContext_2)->old_chunks_0 = old_chunks_1;
    (&kernelContext_2)->structure_base_0 = structure_base_1;
    (&kernelContext_2)->part_ranges_0 = part_ranges_1;
    (&kernelContext_2)->member_offsets_0 = member_offsets_1;
    (&kernelContext_2)->chunks_0 = chunks_1;
    (&kernelContext_2)->chunk_slot_0 = chunk_slot_1;
    (&kernelContext_2)->collider_info_0 = collider_info_1;
    (&kernelContext_2)->islands_0 = islands_1;
    (&kernelContext_2)->chunk_props_0 = chunk_props_1;
    bool _S1;
    if(control_1[int(0)] != 0U)
    {
        _S1 = true;
    }
    else
    {
        _S1 = (&kernelContext_2)->control_0[1U] == 0U;
    }
    if(_S1)
    {
        return;
    }
    uint t_0 = id_0.x;
    uint _S2 = (&kernelContext_2)->params_0->threads_0;
    uint p_0 = 0U;
    for(;;)
    {
        if(p_0 < ((&kernelContext_2)->control_0)[1U])
        {
        }
        else
        {
            break;
        }
        if(((uint4(*((&kernelContext_2)->parents_0+p_0)) ).z) < 2U)
        {
            p_0 = p_0 + 1U;
            continue;
        }
        uint4 _S3 = uint4(*((&kernelContext_2)->islands_of_0+p_0)) ;
        c_0 = _S3.z + t_0;
        for(;;)
        {
            if(c_0 < (_S3.w))
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_2)->collider_key_0+c_0) = 4294967294U;
            c_0 = c_0 + _S2;
        }
        p_0 = p_0 + 1U;
    }
    uint _S4 = total_slots_0(&kernelContext_2);
    if(t_0 >= _S4)
    {
        return;
    }
    uint _S5 = (&kernelContext_2)->control_0[1U] - 1U;
    p_0 = 0U;
    c_0 = _S5;
    for(;;)
    {
        if(p_0 < c_0)
        {
        }
        else
        {
            break;
        }
        uint mid_0 = (p_0 + c_0 + 1U) / 2U;
        if(((uint4(*((&kernelContext_2)->parents_0+mid_0)) ).y) <= t_0)
        {
            p_0 = mid_0;
        }
        else
        {
            c_0 = mid_0 - 1U;
        }
    }
    if(((uint4(*((&kernelContext_2)->parents_0+p_0)) ).z) < 2U)
    {
        return;
    }
    uint _S6 = key_of_0((as_type<uint4>((float4(*((&kernelContext_2)->old_chunks_0+((&kernelContext_2)->params_0->chunk_words_0 * ((uint4(*((&kernelContext_2)->islands_of_0+p_0)) ).x + (&kernelContext_2)->work_0[_S4 + t_0]) + 8U))) ))), &kernelContext_2);
    uint _S7 = (uint4(*((&kernelContext_2)->part_ranges_0+_S6)) ).y;
    uint _S8 = (&kernelContext_2)->member_offsets_0[t_0];
    uint k_0 = 0U;
    for(;;)
    {
        if(k_0 < _S7)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_2)->collider_key_0+(_S8 + k_0)) = _S6;
        k_0 = k_0 + 1U;
    }
    return;
}

[[kernel]] void collider_slots(uint3 id_1 [[thread_position_in_grid]], uint device* control_2 [[buffer(11)]], ColliderParams_0 constant* params_2 [[buffer(0)]], packed_uint4 device* parents_2 [[buffer(1)]], packed_uint4 device* islands_of_2 [[buffer(9)]], uint device* collider_key_2 [[buffer(14)]], uint device* work_2 [[buffer(2)]], packed_float4 device* old_chunks_2 [[buffer(6)]], uint device* structure_base_2 [[buffer(10)]], packed_uint4 device* part_ranges_2 [[buffer(5)]], uint device* member_offsets_2 [[buffer(16)]], packed_float4 device* chunks_2 [[buffer(7)]], uint device* chunk_slot_2 [[buffer(13)]], packed_uint4 device* collider_info_2 [[buffer(15)]], packed_float4 device* islands_2 [[buffer(8)]], packed_float4 device* chunk_props_2 [[buffer(12)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->control_0 = control_2;
    (&kernelContext_3)->params_0 = params_2;
    (&kernelContext_3)->parents_0 = parents_2;
    (&kernelContext_3)->islands_of_0 = islands_of_2;
    (&kernelContext_3)->collider_key_0 = collider_key_2;
    (&kernelContext_3)->work_0 = work_2;
    (&kernelContext_3)->old_chunks_0 = old_chunks_2;
    (&kernelContext_3)->structure_base_0 = structure_base_2;
    (&kernelContext_3)->part_ranges_0 = part_ranges_2;
    (&kernelContext_3)->member_offsets_0 = member_offsets_2;
    (&kernelContext_3)->chunks_0 = chunks_2;
    (&kernelContext_3)->chunk_slot_0 = chunk_slot_2;
    (&kernelContext_3)->collider_info_0 = collider_info_2;
    (&kernelContext_3)->islands_0 = islands_2;
    (&kernelContext_3)->chunk_props_0 = chunk_props_2;
    bool _S9;
    if(control_2[int(0)] != 0U)
    {
        _S9 = true;
    }
    else
    {
        _S9 = (&kernelContext_3)->control_0[1U] == 0U;
    }
    if(_S9)
    {
        _S9 = true;
    }
    else
    {
        _S9 = (id_1.x) >= ((&kernelContext_3)->params_0->chunk_count_0);
    }
    if(_S9)
    {
        return;
    }
    uint _S10 = id_1.x;
    uint _S11 = key_of_0((as_type<uint4>((float4(*((&kernelContext_3)->chunks_0+((&kernelContext_3)->params_0->chunk_words_0 * _S10 + 8U))) ))), &kernelContext_3);
    *((&kernelContext_3)->chunk_slot_0+_S11) = _S10;
    return;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

[[kernel]] void collider_refresh(uint3 id_2 [[thread_position_in_grid]], uint device* control_3 [[buffer(11)]], ColliderParams_0 constant* params_3 [[buffer(0)]], packed_uint4 device* parents_3 [[buffer(1)]], packed_uint4 device* islands_of_3 [[buffer(9)]], uint device* collider_key_3 [[buffer(14)]], uint device* work_3 [[buffer(2)]], packed_float4 device* old_chunks_3 [[buffer(6)]], uint device* structure_base_3 [[buffer(10)]], packed_uint4 device* part_ranges_3 [[buffer(5)]], uint device* member_offsets_3 [[buffer(16)]], packed_float4 device* chunks_3 [[buffer(7)]], uint device* chunk_slot_3 [[buffer(13)]], packed_uint4 device* collider_info_3 [[buffer(15)]], packed_float4 device* islands_3 [[buffer(8)]], packed_float4 device* chunk_props_3 [[buffer(12)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->control_0 = control_3;
    (&kernelContext_4)->params_0 = params_3;
    (&kernelContext_4)->parents_0 = parents_3;
    (&kernelContext_4)->islands_of_0 = islands_of_3;
    (&kernelContext_4)->collider_key_0 = collider_key_3;
    (&kernelContext_4)->work_0 = work_3;
    (&kernelContext_4)->old_chunks_0 = old_chunks_3;
    (&kernelContext_4)->structure_base_0 = structure_base_3;
    (&kernelContext_4)->part_ranges_0 = part_ranges_3;
    (&kernelContext_4)->member_offsets_0 = member_offsets_3;
    (&kernelContext_4)->chunks_0 = chunks_3;
    (&kernelContext_4)->chunk_slot_0 = chunk_slot_3;
    (&kernelContext_4)->collider_info_0 = collider_info_3;
    (&kernelContext_4)->islands_0 = islands_3;
    (&kernelContext_4)->chunk_props_0 = chunk_props_3;
    uint c_1 = id_2.x;
    bool _S12;
    if(control_3[int(0)] != 0U)
    {
        _S12 = true;
    }
    else
    {
        _S12 = (&kernelContext_4)->control_0[1U] == 0U;
    }
    if(_S12)
    {
        _S12 = true;
    }
    else
    {
        _S12 = c_1 >= ((&kernelContext_4)->params_0->collider_count_0);
    }
    if(_S12)
    {
        return;
    }
    uint device* _S13 = (&kernelContext_4)->collider_key_0+c_1;
    uint key_0 = *_S13;
    if((*_S13) == 4294967295U)
    {
        return;
    }
    float inf_0 = float_bits_0(2139095040U);
    if(key_0 == 4294967294U)
    {
        uint _S14 = 2U * c_1;
        *((&kernelContext_4)->collider_info_0+_S14) = packed_uint4(uint4(4294967295U, 4294967295U, 1U, 0U)) ;
        *((&kernelContext_4)->collider_info_0+(_S14 + 1U)) = packed_uint4((as_type<uint4>((float4(inf_0, (&kernelContext_4)->params_0->ground_modulus_0, 1.0f, 0.0f))))) ;
        *((&kernelContext_4)->collider_key_0+c_1) = 4294967295U;
        return;
    }
    uint device* _S15 = (&kernelContext_4)->chunk_slot_0+key_0;
    uint island_0 = (as_type<uint4>((float4(*((&kernelContext_4)->chunks_0+((&kernelContext_4)->params_0->chunk_words_0 * *_S15 + 8U))) ))).y;
    uint4 info_1 = (as_type<uint4>((float4(*((&kernelContext_4)->islands_0+((&kernelContext_4)->params_0->island_words_0 * island_0 + 1U))) )));
    float mass_0 = (float4(*((&kernelContext_4)->islands_0+((&kernelContext_4)->params_0->island_words_0 * island_0 + 2U))) ).w;
    uint cluster_0 = (as_type<uint4>((float4(*((&kernelContext_4)->islands_0+((&kernelContext_4)->params_0->island_words_0 * island_0 + 19U))) ))).w;
    uint _S16 = key_of_0((as_type<uint4>((float4(*((&kernelContext_4)->chunks_0+((&kernelContext_4)->params_0->chunk_words_0 * (as_type<uint4>((float4(*((&kernelContext_4)->islands_0+(&kernelContext_4)->params_0->island_words_0 * island_0)) ))).x + 8U))) ))), &kernelContext_4);
    float4 _S17 = float4(*((&kernelContext_4)->chunk_props_0+_S16)) ;
    bool anchored_0 = ((info_1.x) & 1U) != 0U;
    uint _S18 = 2U * c_1;
    *((&kernelContext_4)->collider_info_0+_S18) = packed_uint4(uint4(*_S15, cluster_0 << 1U, 2U, 1U)) ;
    packed_uint4 device* _S19 = (&kernelContext_4)->collider_info_0+(_S18 + 1U);
    float _S20;
    if(anchored_0)
    {
        _S20 = inf_0;
    }
    else
    {
        _S20 = mass_0;
    }
    *_S19 = packed_uint4((as_type<uint4>((float4(_S20, _S17.x, _S17.y, 0.0f))))) ;
    return;
}

