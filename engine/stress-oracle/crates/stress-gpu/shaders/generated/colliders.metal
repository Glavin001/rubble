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
    uint pad0_0;
    uint pad1_0;
};

struct KernelContext_0
{
    uint device* control_0;
    packed_uint4 device* parents_0;
    packed_uint4 device* islands_of_0;
    uint device* collider_key_0;
    packed_uint4 device* slots_0;
    packed_float4 device* children_0;
    uint device* work_0;
    ColliderParams_0 constant* params_0;
    packed_float4 device* old_chunks_0;
    uint device* structure_base_0;
    packed_uint4 device* part_ranges_0;
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

uint parent_of_0(uint slot_0, KernelContext_0 thread* kernelContext_1)
{
    uint p_0 = 0U;
    for(;;)
    {
        uint _S1 = p_0 + 1U;
        bool _S2;
        if(_S1 < (kernelContext_1->control_0)[1U])
        {
            _S2 = ((uint4(*(kernelContext_1->parents_0+_S1)) ).y) <= slot_0;
        }
        else
        {
            _S2 = false;
        }
        if(_S2)
        {
        }
        else
        {
            break;
        }
        p_0 = _S1;
    }
    return p_0;
}

uint key_of_0(uint4 info_0, KernelContext_0 thread* kernelContext_2)
{
    return kernelContext_2->structure_base_0[info_0.z] + info_0.w;
}

[[kernel]] void collider_keys(uint3 id_0 [[thread_position_in_grid]], uint device* control_1 [[buffer(11)]], packed_uint4 device* parents_1 [[buffer(1)]], packed_uint4 device* islands_of_1 [[buffer(9)]], uint device* collider_key_1 [[buffer(14)]], packed_uint4 device* slots_1 [[buffer(4)]], packed_float4 device* children_1 [[buffer(3)]], uint device* work_1 [[buffer(2)]], ColliderParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* old_chunks_1 [[buffer(6)]], uint device* structure_base_1 [[buffer(10)]], packed_uint4 device* part_ranges_1 [[buffer(5)]], packed_float4 device* chunks_1 [[buffer(7)]], uint device* chunk_slot_1 [[buffer(13)]], packed_uint4 device* collider_info_1 [[buffer(15)]], packed_float4 device* islands_1 [[buffer(8)]], packed_float4 device* chunk_props_1 [[buffer(12)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->control_0 = control_1;
    (&kernelContext_3)->parents_0 = parents_1;
    (&kernelContext_3)->islands_of_0 = islands_of_1;
    (&kernelContext_3)->collider_key_0 = collider_key_1;
    (&kernelContext_3)->slots_0 = slots_1;
    (&kernelContext_3)->children_0 = children_1;
    (&kernelContext_3)->work_0 = work_1;
    (&kernelContext_3)->params_0 = params_1;
    (&kernelContext_3)->old_chunks_0 = old_chunks_1;
    (&kernelContext_3)->structure_base_0 = structure_base_1;
    (&kernelContext_3)->part_ranges_0 = part_ranges_1;
    (&kernelContext_3)->chunks_0 = chunks_1;
    (&kernelContext_3)->chunk_slot_0 = chunk_slot_1;
    (&kernelContext_3)->collider_info_0 = collider_info_1;
    (&kernelContext_3)->islands_0 = islands_1;
    (&kernelContext_3)->chunk_props_0 = chunk_props_1;
    bool _S3;
    if(control_1[int(0)] != 0U)
    {
        _S3 = true;
    }
    else
    {
        _S3 = (&kernelContext_3)->control_0[1U] == 0U;
    }
    if(_S3)
    {
        return;
    }
    uint t_0 = id_0.x;
    if(t_0 < ((&kernelContext_3)->control_0)[1U])
    {
        _S3 = ((uint4(*((&kernelContext_3)->parents_0+t_0)) ).z) >= 2U;
    }
    else
    {
        _S3 = false;
    }
    uint j_0;
    if(_S3)
    {
        uint4 _S4 = uint4(*((&kernelContext_3)->islands_of_0+t_0)) ;
        j_0 = _S4.z;
        for(;;)
        {
            if(j_0 < (_S4.w))
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_3)->collider_key_0+j_0) = 4294967294U;
            j_0 = j_0 + 1U;
        }
    }
    uint _S5 = total_slots_0(&kernelContext_3);
    if(t_0 >= _S5)
    {
        return;
    }
    uint _S6 = parent_of_0(t_0, &kernelContext_3);
    uint4 _S7 = uint4(*((&kernelContext_3)->parents_0+_S6)) ;
    uint _S8 = _S7.y;
    if((t_0 - _S8) >= (_S7.z))
    {
        _S3 = true;
    }
    else
    {
        _S3 = ((uint4(*((&kernelContext_3)->slots_0+t_0)) ).w) == 0U;
    }
    if(_S3)
    {
        return;
    }
    uint _S9 = (uint4(*((&kernelContext_3)->islands_of_0+_S6)) ).x;
    float4 _S10 = float4(*((&kernelContext_3)->children_0+8U * t_0)) ;
    uint _S11 = (as_type<uint>((_S10.y)));
    uint _S12 = (as_type<uint>((_S10.z)));
    uint _S13 = (uint4(*((&kernelContext_3)->slots_0+t_0)) ).y;
    j_0 = 0U;
    uint col_0 = _S13;
    for(;;)
    {
        if(j_0 < _S12)
        {
        }
        else
        {
            break;
        }
        uint _S14 = key_of_0((as_type<uint4>((float4(*((&kernelContext_3)->old_chunks_0+((&kernelContext_3)->params_0->chunk_words_0 * (_S9 + (&kernelContext_3)->work_0[_S5 + _S8 + _S11 + j_0]) + 8U))) ))), &kernelContext_3);
        uint _S15 = (uint4(*((&kernelContext_3)->part_ranges_0+_S14)) ).y;
        uint k_0 = 0U;
        for(;;)
        {
            if(k_0 < _S15)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_3)->collider_key_0+col_0) = _S14;
            uint _S16 = col_0 + 1U;
            k_0 = k_0 + 1U;
            col_0 = _S16;
        }
        j_0 = j_0 + 1U;
    }
    return;
}

[[kernel]] void collider_slots(uint3 id_1 [[thread_position_in_grid]], uint device* control_2 [[buffer(11)]], packed_uint4 device* parents_2 [[buffer(1)]], packed_uint4 device* islands_of_2 [[buffer(9)]], uint device* collider_key_2 [[buffer(14)]], packed_uint4 device* slots_2 [[buffer(4)]], packed_float4 device* children_2 [[buffer(3)]], uint device* work_2 [[buffer(2)]], ColliderParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* old_chunks_2 [[buffer(6)]], uint device* structure_base_2 [[buffer(10)]], packed_uint4 device* part_ranges_2 [[buffer(5)]], packed_float4 device* chunks_2 [[buffer(7)]], uint device* chunk_slot_2 [[buffer(13)]], packed_uint4 device* collider_info_2 [[buffer(15)]], packed_float4 device* islands_2 [[buffer(8)]], packed_float4 device* chunk_props_2 [[buffer(12)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->control_0 = control_2;
    (&kernelContext_4)->parents_0 = parents_2;
    (&kernelContext_4)->islands_of_0 = islands_of_2;
    (&kernelContext_4)->collider_key_0 = collider_key_2;
    (&kernelContext_4)->slots_0 = slots_2;
    (&kernelContext_4)->children_0 = children_2;
    (&kernelContext_4)->work_0 = work_2;
    (&kernelContext_4)->params_0 = params_2;
    (&kernelContext_4)->old_chunks_0 = old_chunks_2;
    (&kernelContext_4)->structure_base_0 = structure_base_2;
    (&kernelContext_4)->part_ranges_0 = part_ranges_2;
    (&kernelContext_4)->chunks_0 = chunks_2;
    (&kernelContext_4)->chunk_slot_0 = chunk_slot_2;
    (&kernelContext_4)->collider_info_0 = collider_info_2;
    (&kernelContext_4)->islands_0 = islands_2;
    (&kernelContext_4)->chunk_props_0 = chunk_props_2;
    bool _S17;
    if(control_2[int(0)] != 0U)
    {
        _S17 = true;
    }
    else
    {
        _S17 = (&kernelContext_4)->control_0[1U] == 0U;
    }
    if(_S17)
    {
        _S17 = true;
    }
    else
    {
        _S17 = (id_1.x) >= ((&kernelContext_4)->params_0->chunk_count_0);
    }
    if(_S17)
    {
        return;
    }
    uint _S18 = id_1.x;
    uint _S19 = key_of_0((as_type<uint4>((float4(*((&kernelContext_4)->chunks_0+((&kernelContext_4)->params_0->chunk_words_0 * _S18 + 8U))) ))), &kernelContext_4);
    *((&kernelContext_4)->chunk_slot_0+_S19) = _S18;
    return;
}

[[kernel]] void collider_refresh(uint3 id_2 [[thread_position_in_grid]], uint device* control_3 [[buffer(11)]], packed_uint4 device* parents_3 [[buffer(1)]], packed_uint4 device* islands_of_3 [[buffer(9)]], uint device* collider_key_3 [[buffer(14)]], packed_uint4 device* slots_3 [[buffer(4)]], packed_float4 device* children_3 [[buffer(3)]], uint device* work_3 [[buffer(2)]], ColliderParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* old_chunks_3 [[buffer(6)]], uint device* structure_base_3 [[buffer(10)]], packed_uint4 device* part_ranges_3 [[buffer(5)]], packed_float4 device* chunks_3 [[buffer(7)]], uint device* chunk_slot_3 [[buffer(13)]], packed_uint4 device* collider_info_3 [[buffer(15)]], packed_float4 device* islands_3 [[buffer(8)]], packed_float4 device* chunk_props_3 [[buffer(12)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->control_0 = control_3;
    (&kernelContext_5)->parents_0 = parents_3;
    (&kernelContext_5)->islands_of_0 = islands_of_3;
    (&kernelContext_5)->collider_key_0 = collider_key_3;
    (&kernelContext_5)->slots_0 = slots_3;
    (&kernelContext_5)->children_0 = children_3;
    (&kernelContext_5)->work_0 = work_3;
    (&kernelContext_5)->params_0 = params_3;
    (&kernelContext_5)->old_chunks_0 = old_chunks_3;
    (&kernelContext_5)->structure_base_0 = structure_base_3;
    (&kernelContext_5)->part_ranges_0 = part_ranges_3;
    (&kernelContext_5)->chunks_0 = chunks_3;
    (&kernelContext_5)->chunk_slot_0 = chunk_slot_3;
    (&kernelContext_5)->collider_info_0 = collider_info_3;
    (&kernelContext_5)->islands_0 = islands_3;
    (&kernelContext_5)->chunk_props_0 = chunk_props_3;
    uint c_0 = id_2.x;
    bool _S20;
    if(control_3[int(0)] != 0U)
    {
        _S20 = true;
    }
    else
    {
        _S20 = (&kernelContext_5)->control_0[1U] == 0U;
    }
    if(_S20)
    {
        _S20 = true;
    }
    else
    {
        _S20 = c_0 >= ((&kernelContext_5)->params_0->collider_count_0);
    }
    if(_S20)
    {
        return;
    }
    uint device* _S21 = (&kernelContext_5)->collider_key_0+c_0;
    uint key_0 = *_S21;
    if((*_S21) == 4294967295U)
    {
        return;
    }
    float inf_0 = (as_type<float>((2139095040U)));
    if(key_0 == 4294967294U)
    {
        uint _S22 = 2U * c_0;
        *((&kernelContext_5)->collider_info_0+_S22) = packed_uint4(uint4(4294967295U, 4294967295U, 1U, 0U)) ;
        *((&kernelContext_5)->collider_info_0+(_S22 + 1U)) = packed_uint4((as_type<uint4>((float4(inf_0, (&kernelContext_5)->params_0->ground_modulus_0, 1.0f, 0.0f))))) ;
        *((&kernelContext_5)->collider_key_0+c_0) = 4294967295U;
        return;
    }
    uint device* _S23 = (&kernelContext_5)->chunk_slot_0+key_0;
    uint island_0 = (as_type<uint4>((float4(*((&kernelContext_5)->chunks_0+((&kernelContext_5)->params_0->chunk_words_0 * *_S23 + 8U))) ))).y;
    uint4 info_1 = (as_type<uint4>((float4(*((&kernelContext_5)->islands_0+((&kernelContext_5)->params_0->island_words_0 * island_0 + 1U))) )));
    float mass_0 = (float4(*((&kernelContext_5)->islands_0+((&kernelContext_5)->params_0->island_words_0 * island_0 + 2U))) ).w;
    uint cluster_0 = (as_type<uint4>((float4(*((&kernelContext_5)->islands_0+((&kernelContext_5)->params_0->island_words_0 * island_0 + 19U))) ))).w;
    uint _S24 = key_of_0((as_type<uint4>((float4(*((&kernelContext_5)->chunks_0+((&kernelContext_5)->params_0->chunk_words_0 * (as_type<uint4>((float4(*((&kernelContext_5)->islands_0+(&kernelContext_5)->params_0->island_words_0 * island_0)) ))).x + 8U))) ))), &kernelContext_5);
    float4 _S25 = float4(*((&kernelContext_5)->chunk_props_0+_S24)) ;
    bool anchored_0 = ((info_1.x) & 1U) != 0U;
    uint _S26 = 2U * c_0;
    *((&kernelContext_5)->collider_info_0+_S26) = packed_uint4(uint4(*_S23, cluster_0 << 1U, 2U, 1U)) ;
    packed_uint4 device* _S27 = (&kernelContext_5)->collider_info_0+(_S26 + 1U);
    float _S28;
    if(anchored_0)
    {
        _S28 = inf_0;
    }
    else
    {
        _S28 = mass_0;
    }
    *_S27 = packed_uint4((as_type<uint4>((float4(_S28, _S25.x, _S25.y, 0.0f))))) ;
    return;
}

