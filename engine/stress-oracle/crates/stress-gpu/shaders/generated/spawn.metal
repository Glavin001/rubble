#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct ChunkStatic_natural_0
{
    packed_float4 center_0;
    packed_float4 inertia0_0;
    packed_float4 inertia1_0;
    packed_float4 inertia2_0;
    packed_float4 inv0_0;
    packed_float4 inv1_0;
    packed_float4 inv2_0;
    packed_float4 scale_0;
    packed_uint4 info_0;
    packed_uint4 load_range_0;
    packed_float4 half_0;
    packed_float4 crot0_0;
    packed_float4 crot1_0;
    packed_float4 crot2_0;
    packed_float4 cmat_0;
    packed_uint4 cinfo_0;
};

struct SpawnParams_0
{
    uint parent_count_0;
    uint first_body_0;
    uint first_collider_0;
    uint live_0;
    uint max_bodies_0;
    uint max_colliders_0;
    uint island_cap_0;
    uint pad2_0;
};

struct KernelContext_0
{
    uint device* control_0;
    packed_uint4 device* parents_0;
    packed_uint4 device* islands_of_0;
    uint device* work_0;
    ChunkStatic_natural_0 device* chunks_0;
    uint device* structure_base_0;
    packed_uint4 device* part_ranges_0;
    uint device* member_offsets_0;
    SpawnParams_0 constant* params_0;
    packed_float4 device* children_0;
    packed_uint4 device* slots_0;
    packed_float4 device* parts_0;
    packed_float4 device* frames_0;
    packed_float4 device* state_0;
    packed_float4 device* bodies_0;
    packed_float4 device* extras_0;
    packed_float4 device* colliders_0;
    array<uint, int(1024)> threadgroup* g_scan_0;
    uint threadgroup* g_split_parents_0;
};

uint total_slots_0(KernelContext_0 thread* kernelContext_0)
{
    return (uint4(*(kernelContext_0->parents_0+*(kernelContext_0->control_0+1U))) ).y;
}

uint parent_at_0(uint i_0, KernelContext_0 thread* kernelContext_1)
{
    uint _S1 = *(kernelContext_1->control_0+1U) - 1U;
    uint lo_0 = 0U;
    uint hi_0 = _S1;
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
        if(((uint4(*(kernelContext_1->parents_0+mid_0)) ).y) <= i_0)
        {
            lo_0 = mid_0;
        }
        else
        {
            hi_0 = mid_0 - 1U;
        }
    }
    return lo_0;
}

uint4 part_range_0(uint layout_chunk_0, KernelContext_0 thread* kernelContext_2)
{
    uint4 _S2 = uint4((kernelContext_2->chunks_0+layout_chunk_0)->info_0) ;
    return uint4(*(kernelContext_2->part_ranges_0+(kernelContext_2->structure_base_0[_S2.z] + _S2.w))) ;
}

uint member_parts_0(uint i_1, uint total_0, KernelContext_0 thread* kernelContext_3)
{
    uint _S3 = parent_at_0(i_1, kernelContext_3);
    if(((uint4(*(kernelContext_3->parents_0+_S3)) ).z) < 2U)
    {
        return 0U;
    }
    uint4 _S4 = part_range_0((uint4(*(kernelContext_3->islands_of_0+_S3)) ).x + kernelContext_3->work_0[total_0 + i_1], kernelContext_3);
    return _S4.y;
}

uint block_scan_0(uint tid_0, uint v_0, KernelContext_0 thread* kernelContext_4)
{
    (*kernelContext_4->g_scan_0)[tid_0] = v_0;
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
        uint add_0;
        if(tid_0 >= step_0)
        {
            add_0 = (*kernelContext_4->g_scan_0)[tid_0 - step_0];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*kernelContext_4->g_scan_0)[tid_0] = (*kernelContext_4->g_scan_0)[tid_0] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_0 = step_0 << 1U;
    }
    return (*kernelContext_4->g_scan_0)[tid_0] - v_0;
}

bool slot_valid_0(uint s_0, KernelContext_0 thread* kernelContext_5)
{
    uint _S5 = parent_at_0(s_0, kernelContext_5);
    uint4 _S6 = uint4(*(kernelContext_5->parents_0+_S5)) ;
    uint _S7 = _S6.z;
    bool _S8;
    if(_S7 >= 2U)
    {
        _S8 = (s_0 - _S6.y) < _S7;
    }
    else
    {
        _S8 = false;
    }
    return _S8;
}

uint2 child_members_0(uint slot_0, KernelContext_0 thread* kernelContext_6)
{
    float4 _S9 = float4(*(kernelContext_6->children_0+8U * slot_0)) ;
    return uint2((as_type<uint>((_S9.y))), (as_type<uint>((_S9.z))));
}

[[kernel]] void spawn_offsets(uint3 thread_0 [[thread_position_in_threadgroup]], uint device* control_1 [[buffer(15)]], packed_uint4 device* parents_1 [[buffer(1)]], packed_uint4 device* islands_of_1 [[buffer(12)]], uint device* work_1 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(7)]], uint device* structure_base_1 [[buffer(14)]], packed_uint4 device* part_ranges_1 [[buffer(5)]], uint device* member_offsets_1 [[buffer(16)]], SpawnParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* children_1 [[buffer(3)]], packed_uint4 device* slots_1 [[buffer(9)]], packed_float4 device* parts_1 [[buffer(6)]], packed_float4 device* frames_1 [[buffer(4)]], packed_float4 device* state_1 [[buffer(8)]], packed_float4 device* bodies_1 [[buffer(10)]], packed_float4 device* extras_1 [[buffer(13)]], packed_float4 device* colliders_1 [[buffer(11)]])
{
    uint v_1;
    bool fits_0;
    thread KernelContext_0 kernelContext_7;
    (&kernelContext_7)->control_0 = control_1;
    (&kernelContext_7)->parents_0 = parents_1;
    (&kernelContext_7)->islands_of_0 = islands_of_1;
    (&kernelContext_7)->work_0 = work_1;
    (&kernelContext_7)->chunks_0 = chunks_1;
    (&kernelContext_7)->structure_base_0 = structure_base_1;
    (&kernelContext_7)->part_ranges_0 = part_ranges_1;
    (&kernelContext_7)->member_offsets_0 = member_offsets_1;
    (&kernelContext_7)->params_0 = params_1;
    (&kernelContext_7)->children_0 = children_1;
    (&kernelContext_7)->slots_0 = slots_1;
    (&kernelContext_7)->parts_0 = parts_1;
    (&kernelContext_7)->frames_0 = frames_1;
    (&kernelContext_7)->state_0 = state_1;
    (&kernelContext_7)->bodies_0 = bodies_1;
    (&kernelContext_7)->extras_0 = extras_1;
    (&kernelContext_7)->colliders_0 = colliders_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_7)->g_scan_0 = &g_scan_1;
    threadgroup uint g_split_parents_1;
    (&kernelContext_7)->g_split_parents_0 = &g_split_parents_1;
    uint tid_1 = thread_0.x;
    uint body0_0 = *(control_1+4U);
    uint collider0_0 = *(control_1+5U);
    if((*(control_1+1U)) == 0U)
    {
        if(tid_1 == 0U)
        {
            *((&kernelContext_7)->control_0+0U) = 0U;
            *((&kernelContext_7)->control_0+8U) = body0_0;
            *((&kernelContext_7)->control_0+9U) = collider0_0;
        }
        return;
    }
    uint _S10 = total_slots_0(&kernelContext_7);
    uint block_0 = 0U;
    uint carry_0 = 0U;
    for(;;)
    {
        if(block_0 < _S10)
        {
        }
        else
        {
            break;
        }
        uint i_2 = block_0 + tid_1;
        bool _S11 = i_2 < _S10;
        if(_S11)
        {
            uint _S12 = member_parts_0(i_2, _S10, &kernelContext_7);
            v_1 = _S12;
        }
        else
        {
            v_1 = 0U;
        }
        uint _S13 = block_scan_0(tid_1, v_1, &kernelContext_7);
        if(_S11)
        {
            *((&kernelContext_7)->member_offsets_0+i_2) = collider0_0 + carry_0 + _S13;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_1 = carry_0 + (*(&kernelContext_7)->g_scan_0)[int(1023)];
        block_0 = block_0 + 1024U;
        carry_0 = carry_1;
    }
    block_0 = 0U;
    uint bodies_needed_0 = 0U;
    for(;;)
    {
        if(block_0 < _S10)
        {
        }
        else
        {
            break;
        }
        uint s2_0 = block_0 + tid_1;
        if(s2_0 < _S10)
        {
            bool _S14 = slot_valid_0(s2_0, &kernelContext_7);
            fits_0 = _S14;
        }
        else
        {
            fits_0 = false;
        }
        if(fits_0)
        {
            v_1 = 1U;
        }
        else
        {
            v_1 = 0U;
        }
        uint _S15 = block_scan_0(tid_1, v_1, &kernelContext_7);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint bodies_needed_1 = bodies_needed_0 + (*(&kernelContext_7)->g_scan_0)[int(1023)];
        block_0 = block_0 + 1024U;
        bodies_needed_0 = bodies_needed_1;
    }
    bool _S16 = tid_1 == 0U;
    uint c_end_0;
    if(_S16)
    {
        v_1 = 0U;
        c_end_0 = 0U;
        for(;;)
        {
            if(v_1 < (*((&kernelContext_7)->control_0+1U)))
            {
            }
            else
            {
                break;
            }
            if(((uint4(*((&kernelContext_7)->parents_0+v_1)) ).z) >= 2U)
            {
                c_end_0 = c_end_0 + 1U;
            }
            v_1 = v_1 + 1U;
        }
        *(&kernelContext_7)->g_split_parents_0 = c_end_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((bodies_needed_0 - *(&kernelContext_7)->g_split_parents_0) <= ((&kernelContext_7)->params_0->island_cap_0 - min((&kernelContext_7)->params_0->island_cap_0, *((&kernelContext_7)->control_0+2U))))
    {
        if(((&kernelContext_7)->params_0->live_0) == 0U)
        {
            fits_0 = true;
        }
        else
        {
            if((body0_0 + bodies_needed_0) <= ((&kernelContext_7)->params_0->max_bodies_0))
            {
                fits_0 = (collider0_0 + carry_0) <= ((&kernelContext_7)->params_0->max_colliders_0);
            }
            else
            {
                fits_0 = false;
            }
        }
    }
    else
    {
        fits_0 = false;
    }
    if(_S16)
    {
        uint device* _S17 = (&kernelContext_7)->control_0+0U;
        if(fits_0)
        {
            block_0 = 0U;
        }
        else
        {
            block_0 = 1U;
        }
        *_S17 = block_0;
        uint device* _S18 = (&kernelContext_7)->control_0+8U;
        if(fits_0)
        {
            block_0 = body0_0 + bodies_needed_0;
        }
        else
        {
            block_0 = body0_0;
        }
        *_S18 = block_0;
        uint device* _S19 = (&kernelContext_7)->control_0+9U;
        if(fits_0)
        {
            block_0 = collider0_0 + carry_0;
        }
        else
        {
            block_0 = collider0_0;
        }
        *_S19 = block_0;
    }
    if(!fits_0)
    {
        return;
    }
    block_0 = 0U;
    v_1 = 0U;
    for(;;)
    {
        if(block_0 < _S10)
        {
        }
        else
        {
            break;
        }
        uint s2_1 = block_0 + tid_1;
        bool _S20 = s2_1 < _S10;
        bool valid_0;
        if(_S20)
        {
            bool _S21 = slot_valid_0(s2_1, &kernelContext_7);
            valid_0 = _S21;
        }
        else
        {
            valid_0 = false;
        }
        if(valid_0)
        {
            bodies_needed_0 = 1U;
        }
        else
        {
            bodies_needed_0 = 0U;
        }
        uint _S22 = block_scan_0(tid_1, bodies_needed_0, &kernelContext_7);
        if(_S20)
        {
            if(valid_0)
            {
                uint _S23 = parent_at_0(s2_1, &kernelContext_7);
                uint4 _S24 = uint4(*((&kernelContext_7)->parents_0+_S23)) ;
                uint2 _S25 = child_members_0(s2_1, &kernelContext_7);
                uint first_0 = _S24.y + _S25.x;
                uint end_0 = first_0 + _S25.y;
                uint c_first_0 = *((&kernelContext_7)->member_offsets_0+first_0);
                if(end_0 < _S10)
                {
                    c_end_0 = *((&kernelContext_7)->member_offsets_0+end_0);
                }
                else
                {
                    c_end_0 = collider0_0 + carry_0;
                }
                *((&kernelContext_7)->slots_0+s2_1) = packed_uint4(uint4(body0_0 + v_1 + _S22, c_first_0, c_end_0 - c_first_0, 1U)) ;
            }
            else
            {
                *((&kernelContext_7)->slots_0+s2_1) = packed_uint4(uint4(int4(int(0)) )) ;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint bodies_2 = v_1 + (*(&kernelContext_7)->g_scan_0)[int(1023)];
        block_0 = block_0 + 1024U;
        v_1 = bodies_2;
    }
    return;
}

uint parent_of_0(uint slot_1, KernelContext_0 thread* kernelContext_8)
{
    uint p_0 = 0U;
    for(;;)
    {
        uint _S26 = p_0 + 1U;
        bool _S27;
        if(_S26 < (*(kernelContext_8->control_0+1U)))
        {
            _S27 = ((uint4(*(kernelContext_8->parents_0+_S26)) ).y) <= slot_1;
        }
        else
        {
            _S27 = false;
        }
        if(_S27)
        {
        }
        else
        {
            break;
        }
        p_0 = _S26;
    }
    return p_0;
}

struct Quat_0
{
    float w_0;
    float x_0;
    float y_0;
    float z_0;
};

Quat_0 quat_from_columns_0(float3 c0_0, float3 c1_0, float3 c2_0)
{
    float m00_0 = c0_0.x;
    float m01_0 = c0_0.y;
    float m02_0 = c0_0.z;
    float m10_0 = c1_0.x;
    float m11_0 = c1_0.y;
    float m12_0 = c1_0.z;
    float m20_0 = c2_0.x;
    float m21_0 = c2_0.y;
    float m22_0 = c2_0.z;
    thread Quat_0 q_0;
    if(m22_0 <= 0.0f)
    {
        float dif10_0 = m11_0 - m00_0;
        float omm22_0 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            float f_0 = omm22_0 - dif10_0;
            float inv_0 = 0.5f / sqrt(f_0);
            (&q_0)->x_0 = f_0 * inv_0;
            (&q_0)->y_0 = (m01_0 + m10_0) * inv_0;
            (&q_0)->z_0 = (m02_0 + m20_0) * inv_0;
            (&q_0)->w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            float f_1 = omm22_0 + dif10_0;
            float inv_1 = 0.5f / sqrt(f_1);
            (&q_0)->x_0 = (m01_0 + m10_0) * inv_1;
            (&q_0)->y_0 = f_1 * inv_1;
            (&q_0)->z_0 = (m12_0 + m21_0) * inv_1;
            (&q_0)->w_0 = (m20_0 - m02_0) * inv_1;
        }
    }
    else
    {
        float sum10_0 = m11_0 + m00_0;
        float opm22_0 = 1.0f + m22_0;
        if(sum10_0 <= 0.0f)
        {
            float f_2 = opm22_0 - sum10_0;
            float inv_2 = 0.5f / sqrt(f_2);
            (&q_0)->x_0 = (m02_0 + m20_0) * inv_2;
            (&q_0)->y_0 = (m12_0 + m21_0) * inv_2;
            (&q_0)->z_0 = f_2 * inv_2;
            (&q_0)->w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            float f_3 = opm22_0 + sum10_0;
            float inv_3 = 0.5f / sqrt(f_3);
            (&q_0)->x_0 = (m12_0 - m21_0) * inv_3;
            (&q_0)->y_0 = (m20_0 - m02_0) * inv_3;
            (&q_0)->z_0 = (m01_0 - m10_0) * inv_3;
            (&q_0)->w_0 = f_3 * inv_3;
        }
    }
    return q_0;
}

Quat_0 normalized_0(const Quat_0 thread* q_1)
{
    float n_0 = sqrt(q_1->w_0 * q_1->w_0 + q_1->x_0 * q_1->x_0 + q_1->y_0 * q_1->y_0 + q_1->z_0 * q_1->z_0);
    thread Quat_0 r_0;
    (&r_0)->w_0 = q_1->w_0 / n_0;
    (&r_0)->x_0 = q_1->x_0 / n_0;
    (&r_0)->y_0 = q_1->y_0 / n_0;
    (&r_0)->z_0 = q_1->z_0 / n_0;
    return r_0;
}

void principal_axes_0(float3 r0_0, float3 r1_0, float3 r2_0, Quat_0 thread* rot_0, float3 thread* moments_0)
{
    array<float, int(3)> _S28 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S29 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S30 = { { r2_0.x, r2_0.y, r2_0.z } };
    thread array<array<float, int(3)>, int(3)> a_0;
    a_0[int(0)] = _S28;
    a_0[int(1)] = _S29;
    a_0[int(2)] = _S30;
    array<float, int(3)> _S31 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S32 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S33 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_2;
    v_2[int(0)] = _S31;
    v_2[int(1)] = _S32;
    v_2[int(2)] = _S33;
    uint it_0 = 0U;
    for(;;)
    {
        if(it_0 < 32U)
        {
        }
        else
        {
            break;
        }
        uint q_2;
        if((abs(a_0[int(0)][int(2)])) > (abs(a_0[0U][1U])))
        {
            q_2 = 2U;
        }
        else
        {
            q_2 = 1U;
        }
        uint p_1;
        uint q_3;
        if((abs(a_0[int(1)][int(2)])) > (abs(a_0[0U][q_2])))
        {
            p_1 = 1U;
            q_3 = 2U;
        }
        else
        {
            p_1 = 0U;
            q_3 = q_2;
        }
        if((abs(a_0[p_1][q_3])) <= (9.99999982451670044e-15f * max(max(abs(a_0[int(0)][int(0)]), abs(a_0[int(1)][int(1)])), max(abs(a_0[int(2)][int(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        float theta_0 = 0.5f * atan2(2.0f * a_0[p_1][q_3], a_0[q_3][q_3] - a_0[p_1][p_1]);
        float s_1 = sin(theta_0);
        float c_0 = cos(theta_0);
        float app_0 = a_0[p_1][p_1];
        float aqq_0 = a_0[q_3][q_3];
        float apq_0 = a_0[p_1][q_3];
        uint k_0 = 0U;
        for(;;)
        {
            if(k_0 < 3U)
            {
            }
            else
            {
                break;
            }
            bool _S34;
            if(k_0 != p_1)
            {
                _S34 = k_0 != q_3;
            }
            else
            {
                _S34 = false;
            }
            if(_S34)
            {
                float kp_0 = a_0[k_0][p_1];
                float kq_0 = a_0[k_0][q_3];
                float _S35 = c_0 * a_0[k_0][p_1] - s_1 * a_0[k_0][q_3];
                a_0[k_0][p_1] = _S35;
                a_0[p_1][k_0] = _S35;
                float _S36 = s_1 * kp_0 + c_0 * kq_0;
                a_0[k_0][q_3] = _S36;
                a_0[q_3][k_0] = _S36;
            }
            k_0 = k_0 + 1U;
        }
        float _S37 = c_0 * c_0;
        float _S38 = 2.0f * s_1 * c_0 * apq_0;
        float _S39 = s_1 * s_1;
        a_0[p_1][p_1] = _S37 * app_0 - _S38 + _S39 * aqq_0;
        a_0[q_3][q_3] = _S39 * app_0 + _S38 + _S37 * aqq_0;
        a_0[p_1][q_3] = 0.0f;
        a_0[q_3][p_1] = 0.0f;
        uint row_0 = 0U;
        for(;;)
        {
            if(row_0 < 3U)
            {
            }
            else
            {
                break;
            }
            float kp_1 = v_2[row_0][p_1];
            float kq_1 = v_2[row_0][q_3];
            v_2[row_0][p_1] = c_0 * v_2[row_0][p_1] - s_1 * v_2[row_0][q_3];
            v_2[row_0][q_3] = s_1 * kp_1 + c_0 * kq_1;
            row_0 = row_0 + 1U;
        }
        it_0 = it_0 + 1U;
    }
    float3 x_1 = float3(v_2[int(0)][int(0)], v_2[int(1)][int(0)], v_2[int(2)][int(0)]);
    float3 y_1 = float3(v_2[int(0)][int(1)], v_2[int(1)][int(1)], v_2[int(2)][int(1)]);
    float3 z_1 = float3(v_2[int(0)][int(2)], v_2[int(1)][int(2)], v_2[int(2)][int(2)]);
    float3 z_2;
    if((dot(cross(x_1, y_1), z_1)) < 0.0f)
    {
        z_2 = - z_1;
    }
    else
    {
        z_2 = z_1;
    }
    thread Quat_0 _S40 = quat_from_columns_0(x_1, y_1, z_2);
    Quat_0 _S41 = normalized_0(&_S40);
    *rot_0 = _S41;
    *moments_0 = float3(a_0[int(0)][int(0)], a_0[int(1)][int(1)], a_0[int(2)][int(2)]);
    return;
}

Quat_0 conj_0(const Quat_0 thread* q_4)
{
    thread Quat_0 c_1;
    (&c_1)->w_0 = q_4->w_0;
    (&c_1)->x_0 = - q_4->x_0;
    (&c_1)->y_0 = - q_4->y_0;
    (&c_1)->z_0 = - q_4->z_0;
    return c_1;
}

Quat_0 quat_of_0(float4 q_5)
{
    thread Quat_0 r_1;
    (&r_1)->x_0 = q_5.x;
    (&r_1)->y_0 = q_5.y;
    (&r_1)->z_0 = q_5.z;
    (&r_1)->w_0 = q_5.w;
    return r_1;
}

float3 rotate_0(const Quat_0 thread* q_6, float3 v_3)
{
    float3 qv_0 = float3(q_6->x_0, q_6->y_0, q_6->z_0);
    float3 t_0 = cross(qv_0, v_3) * float3(2.0f) ;
    return v_3 + t_0 * float3(q_6->w_0)  + cross(qv_0, t_0);
}

Quat_0 quat_mul_0(const Quat_0 thread* a_1, const Quat_0 thread* o_0)
{
    thread Quat_0 r_2;
    (&r_2)->w_0 = a_1->w_0 * o_0->w_0 - a_1->x_0 * o_0->x_0 - a_1->y_0 * o_0->y_0 - a_1->z_0 * o_0->z_0;
    (&r_2)->x_0 = a_1->w_0 * o_0->x_0 + a_1->x_0 * o_0->w_0 + a_1->y_0 * o_0->z_0 - a_1->z_0 * o_0->y_0;
    (&r_2)->y_0 = a_1->w_0 * o_0->y_0 - a_1->x_0 * o_0->z_0 + a_1->y_0 * o_0->w_0 + a_1->z_0 * o_0->x_0;
    (&r_2)->z_0 = a_1->w_0 * o_0->z_0 + a_1->x_0 * o_0->y_0 - a_1->y_0 * o_0->x_0 + a_1->z_0 * o_0->w_0;
    return r_2;
}

Quat_0 rot_to_rubble_0(const Quat_0 thread* q_7)
{
    thread Quat_0 axes_0;
    (&axes_0)->w_0 = 0.70710676908493042f;
    (&axes_0)->x_0 = -0.70710676908493042f;
    (&axes_0)->y_0 = 0.0f;
    (&axes_0)->z_0 = 0.0f;
    thread Quat_0 _S42 = axes_0;
    Quat_0 _S43 = quat_mul_0(&_S42, q_7);
    thread Quat_0 _S44 = _S43;
    Quat_0 _S45 = normalized_0(&_S44);
    return _S45;
}

float3 to_rubble_0(float3 v_4)
{
    return float3(v_4.x, v_4.z, - v_4.y);
}

float4 quat_vec_0(const Quat_0 thread* q_8)
{
    return float4(q_8->x_0, q_8->y_0, q_8->z_0, q_8->w_0);
}

[[kernel]] void spawn_bodies(uint3 id_0 [[thread_position_in_grid]], uint device* control_2 [[buffer(15)]], packed_uint4 device* parents_2 [[buffer(1)]], packed_uint4 device* islands_of_2 [[buffer(12)]], uint device* work_2 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(7)]], uint device* structure_base_2 [[buffer(14)]], packed_uint4 device* part_ranges_2 [[buffer(5)]], uint device* member_offsets_2 [[buffer(16)]], SpawnParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* children_2 [[buffer(3)]], packed_uint4 device* slots_2 [[buffer(9)]], packed_float4 device* parts_2 [[buffer(6)]], packed_float4 device* frames_2 [[buffer(4)]], packed_float4 device* state_2 [[buffer(8)]], packed_float4 device* bodies_3 [[buffer(10)]], packed_float4 device* extras_2 [[buffer(13)]], packed_float4 device* colliders_2 [[buffer(11)]])
{
    uint _S46;
    uint k_1;
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->control_0 = control_2;
    (&kernelContext_9)->parents_0 = parents_2;
    (&kernelContext_9)->islands_of_0 = islands_of_2;
    (&kernelContext_9)->work_0 = work_2;
    (&kernelContext_9)->chunks_0 = chunks_2;
    (&kernelContext_9)->structure_base_0 = structure_base_2;
    (&kernelContext_9)->part_ranges_0 = part_ranges_2;
    (&kernelContext_9)->member_offsets_0 = member_offsets_2;
    (&kernelContext_9)->params_0 = params_2;
    (&kernelContext_9)->children_0 = children_2;
    (&kernelContext_9)->slots_0 = slots_2;
    (&kernelContext_9)->parts_0 = parts_2;
    (&kernelContext_9)->frames_0 = frames_2;
    (&kernelContext_9)->state_0 = state_2;
    (&kernelContext_9)->bodies_0 = bodies_3;
    (&kernelContext_9)->extras_0 = extras_2;
    (&kernelContext_9)->colliders_0 = colliders_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_9)->g_scan_0 = &g_scan_2;
    threadgroup uint g_split_parents_2;
    (&kernelContext_9)->g_split_parents_0 = &g_split_parents_2;
    uint slot_2 = id_0.x;
    uint _S47 = total_slots_0(&kernelContext_9);
    if(slot_2 >= _S47)
    {
        return;
    }
    uint _S48 = parent_of_0(slot_2, &kernelContext_9);
    uint4 _S49 = uint4(*((&kernelContext_9)->parents_0+_S48)) ;
    uint _S50 = _S49.y;
    bool _S51;
    if((slot_2 - _S50) >= (_S49.z))
    {
        _S51 = true;
    }
    else
    {
        _S51 = ((uint4(*((&kernelContext_9)->slots_0+slot_2)) ).w) == 0U;
    }
    if(_S51)
    {
        _S51 = true;
    }
    else
    {
        _S51 = (*((&kernelContext_9)->control_0+0U)) != 0U;
    }
    if(_S51)
    {
        return;
    }
    uint _S52 = (uint4(*((&kernelContext_9)->islands_of_0+_S48)) ).x;
    uint2 _S53 = child_members_0(slot_2, &kernelContext_9);
    uint _S54 = 8U * slot_2;
    bool anchored_0 = (as_type<uint>(((float4(*((&kernelContext_9)->children_0+_S54)) ).w))) != 0U;
    float3 _S55 = float3(0.0f) ;
    uint j_0 = 0U;
    float mass_0 = 0.0f;
    float3 moment_0 = _S55;
    for(;;)
    {
        uint _S56 = _S53.y;
        _S46 = _S56;
        if(j_0 < _S56)
        {
        }
        else
        {
            break;
        }
        uint4 _S57 = part_range_0(_S52 + (&kernelContext_9)->work_0[_S47 + _S50 + _S53.x + j_0], &kernelContext_9);
        uint _S58 = _S57.x;
        k_1 = _S58;
        for(;;)
        {
            if(k_1 < (_S58 + _S57.y))
            {
            }
            else
            {
                break;
            }
            float4 _S59 = float4(*((&kernelContext_9)->parts_0+8U * k_1)) ;
            float _S60 = _S59.w;
            float mass_1 = mass_0 + _S60;
            float3 moment_1 = moment_0 + _S59.xyz * float3(_S60) ;
            k_1 = k_1 + 1U;
            mass_0 = mass_1;
            moment_0 = moment_1;
        }
        j_0 = j_0 + 1U;
    }
    float3 com_0 = moment_0 / float3(max(mass_0, 1.00000000317107685e-30f)) ;
    j_0 = 0U;
    float3 i0_0 = _S55;
    float3 i1_0 = _S55;
    float3 i2_0 = _S55;
    for(;;)
    {
        if(j_0 < _S46)
        {
        }
        else
        {
            break;
        }
        uint4 _S61 = part_range_0(_S52 + (&kernelContext_9)->work_0[_S47 + _S50 + _S53.x + j_0], &kernelContext_9);
        uint _S62 = _S61.x;
        k_1 = _S62;
        for(;;)
        {
            if(k_1 < (_S62 + _S61.y))
            {
            }
            else
            {
                break;
            }
            uint _S63 = 8U * k_1;
            float4 _S64 = float4(*((&kernelContext_9)->parts_0+_S63)) ;
            float3 d_0 = _S64.xyz - com_0;
            float dd_0 = dot(d_0, d_0);
            float3 _S65 = float3(_S64.w) ;
            float3 i0_1 = i0_0 + ((float4(*((&kernelContext_9)->parts_0+(_S63 + 1U))) ).xyz + (float3(dd_0, 0.0f, 0.0f) - float3(d_0.x)  * d_0) * _S65);
            float3 i1_1 = i1_0 + ((float4(*((&kernelContext_9)->parts_0+(_S63 + 2U))) ).xyz + (float3(0.0f, dd_0, 0.0f) - float3(d_0.y)  * d_0) * _S65);
            float3 i2_1 = i2_0 + ((float4(*((&kernelContext_9)->parts_0+(_S63 + 3U))) ).xyz + (float3(0.0f, 0.0f, dd_0) - float3(d_0.z)  * d_0) * _S65);
            k_1 = k_1 + 1U;
            i0_0 = i0_1;
            i1_0 = i1_1;
            i2_0 = i2_1;
        }
        j_0 = j_0 + 1U;
    }
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(i0_0, i1_0, i2_0, &principal_0, &moments_1);
    uint _S66 = 5U * _S48;
    Quat_0 pose_rot_0 = quat_of_0(float4(*((&kernelContext_9)->frames_0+_S66)) );
    float3 pose_pos_0 = (float4(*((&kernelContext_9)->frames_0+(_S66 + 4U))) ).xyz;
    bool _S67 = !anchored_0;
    float inv_mass_0;
    float3 pose_pos_1;
    if(_S67)
    {
        j_0 = 0U;
        inv_mass_0 = 0.0f;
        pose_pos_1 = _S55;
        for(;;)
        {
            if(j_0 < _S46)
            {
            }
            else
            {
                break;
            }
            uint c_2 = _S52 + (&kernelContext_9)->work_0[_S47 + _S50 + _S53.x + j_0];
            float cm_0 = (float4(((&kernelContext_9)->chunks_0+c_2)->center_0) ).w;
            float sm_0 = inv_mass_0 + cm_0;
            float3 su_0 = pose_pos_1 + (float4(*((&kernelContext_9)->state_0+4U * c_2)) ).xyz * float3(cm_0) ;
            j_0 = j_0 + 1U;
            inv_mass_0 = sm_0;
            pose_pos_1 = su_0;
        }
        float3 _S68 = pose_pos_1 / float3(inv_mass_0) ;
        thread Quat_0 _S69 = pose_rot_0;
        float3 _S70 = rotate_0(&_S69, _S68);
        pose_pos_1 = pose_pos_0 + _S70;
    }
    else
    {
        pose_pos_1 = pose_pos_0;
    }
    uint4 _S71 = uint4(*((&kernelContext_9)->slots_0+slot_2)) ;
    thread Quat_0 _S72 = pose_rot_0;
    thread Quat_0 _S73 = principal_0;
    Quat_0 _S74 = quat_mul_0(&_S72, &_S73);
    thread Quat_0 _S75 = _S74;
    Quat_0 _S76 = normalized_0(&_S75);
    thread Quat_0 _S77 = _S76;
    Quat_0 _S78 = rot_to_rubble_0(&_S77);
    if(anchored_0)
    {
        _S51 = true;
    }
    else
    {
        _S51 = !(mass_0 > 0.0f);
    }
    if(_S51)
    {
        inv_mass_0 = 0.0f;
    }
    else
    {
        inv_mass_0 = 1.0f / mass_0;
    }
    float3 inv_inertia_0;
    if(_S67)
    {
        float _S79;
        if((moments_1.x) > 0.0f)
        {
            _S79 = 1.0f / moments_1.x;
        }
        else
        {
            _S79 = 0.0f;
        }
        float _S80;
        if((moments_1.y) > 0.0f)
        {
            _S80 = 1.0f / moments_1.y;
        }
        else
        {
            _S80 = 0.0f;
        }
        float _S81;
        if((moments_1.z) > 0.0f)
        {
            _S81 = 1.0f / moments_1.z;
        }
        else
        {
            _S81 = 0.0f;
        }
        inv_inertia_0 = float3(_S79, _S80, _S81);
    }
    else
    {
        inv_inertia_0 = _S55;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S55;
    }
    else
    {
        velocity_0 = to_rubble_0((float4(*((&kernelContext_9)->children_0+(_S54 + 5U))) ).xyz);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S55;
    }
    else
    {
        spin_0 = to_rubble_0((float4(*((&kernelContext_9)->children_0+(_S54 + 6U))) ).xyz);
    }
    if(anchored_0)
    {
        k_1 = 0U;
    }
    else
    {
        k_1 = 9U;
    }
    thread Quat_0 _S82 = pose_rot_0;
    float3 _S83 = rotate_0(&_S82, com_0);
    float4 _S84 = float4(to_rubble_0(pose_pos_1 + _S83), inv_mass_0);
    thread Quat_0 _S85 = _S78;
    float4 _S86 = quat_vec_0(&_S85);
    array<float4, int(7)> _S87 = { { _S84, _S86, float4(inv_inertia_0, (as_type<float>((k_1)))), float4(velocity_0, 0.0f), float4(spin_0, 0.0f), float4(0.0f) , float4(0.0f, 0.0f, 0.0f, 1.0f) } };
    if(((&kernelContext_9)->params_0->live_0) != 0U)
    {
        j_0 = _S71.x;
    }
    else
    {
        j_0 = _S71.x - *((&kernelContext_9)->control_0+4U);
    }
    uint _S88 = 7U * j_0;
    uint w_1 = 0U;
    for(;;)
    {
        if(w_1 < 7U)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_9)->bodies_0+(_S88 + w_1)) = packed_float4(_S87[w_1]) ;
        w_1 = w_1 + 1U;
    }
    if(((&kernelContext_9)->params_0->live_0) != 0U)
    {
        w_1 = 0U;
        for(;;)
        {
            if(w_1 < 7U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_9)->bodies_0+(7U * ((&kernelContext_9)->params_0->max_bodies_0 + _S71.x) + w_1)) = packed_float4(_S87[w_1]) ;
            w_1 = w_1 + 1U;
        }
    }
    uint _S89 = 2U * slot_2;
    *((&kernelContext_9)->extras_0+_S89) = packed_float4(float4(com_0, mass_0)) ;
    packed_float4 device* _S90 = (&kernelContext_9)->extras_0+(_S89 + 1U);
    thread Quat_0 _S91 = principal_0;
    float4 _S92 = quat_vec_0(&_S91);
    *_S90 = packed_float4(_S92) ;
    return;
}

[[kernel]] void spawn_colliders(uint3 id_1 [[thread_position_in_grid]], uint device* control_3 [[buffer(15)]], packed_uint4 device* parents_3 [[buffer(1)]], packed_uint4 device* islands_of_3 [[buffer(12)]], uint device* work_3 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(7)]], uint device* structure_base_3 [[buffer(14)]], packed_uint4 device* part_ranges_3 [[buffer(5)]], uint device* member_offsets_3 [[buffer(16)]], SpawnParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* children_3 [[buffer(3)]], packed_uint4 device* slots_3 [[buffer(9)]], packed_float4 device* parts_3 [[buffer(6)]], packed_float4 device* frames_3 [[buffer(4)]], packed_float4 device* state_3 [[buffer(8)]], packed_float4 device* bodies_4 [[buffer(10)]], packed_float4 device* extras_3 [[buffer(13)]], packed_float4 device* colliders_3 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_10;
    (&kernelContext_10)->control_0 = control_3;
    (&kernelContext_10)->parents_0 = parents_3;
    (&kernelContext_10)->islands_of_0 = islands_of_3;
    (&kernelContext_10)->work_0 = work_3;
    (&kernelContext_10)->chunks_0 = chunks_3;
    (&kernelContext_10)->structure_base_0 = structure_base_3;
    (&kernelContext_10)->part_ranges_0 = part_ranges_3;
    (&kernelContext_10)->member_offsets_0 = member_offsets_3;
    (&kernelContext_10)->params_0 = params_3;
    (&kernelContext_10)->children_0 = children_3;
    (&kernelContext_10)->slots_0 = slots_3;
    (&kernelContext_10)->parts_0 = parts_3;
    (&kernelContext_10)->frames_0 = frames_3;
    (&kernelContext_10)->state_0 = state_3;
    (&kernelContext_10)->bodies_0 = bodies_4;
    (&kernelContext_10)->extras_0 = extras_3;
    (&kernelContext_10)->colliders_0 = colliders_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_10)->g_scan_0 = &g_scan_3;
    threadgroup uint g_split_parents_3;
    (&kernelContext_10)->g_split_parents_0 = &g_split_parents_3;
    uint i_3 = id_1.x;
    uint _S93 = total_slots_0(&kernelContext_10);
    bool _S94;
    if(i_3 >= _S93)
    {
        _S94 = true;
    }
    else
    {
        _S94 = (*((&kernelContext_10)->control_0+1U)) == 0U;
    }
    if(_S94)
    {
        _S94 = true;
    }
    else
    {
        _S94 = (*((&kernelContext_10)->control_0+0U)) != 0U;
    }
    if(_S94)
    {
        return;
    }
    uint _S95 = parent_at_0(i_3, &kernelContext_10);
    uint4 _S96 = uint4(*((&kernelContext_10)->parents_0+_S95)) ;
    uint _S97 = _S96.z;
    if(_S97 < 2U)
    {
        return;
    }
    uint _S98 = _S96.y;
    uint _S99 = i_3 - _S98;
    uint _S100 = _S97 - 1U;
    uint lo_1 = 0U;
    uint hi_1 = _S100;
    for(;;)
    {
        if(lo_1 < hi_1)
        {
        }
        else
        {
            break;
        }
        uint mid_1 = (lo_1 + hi_1 + 1U) / 2U;
        uint2 _S101 = child_members_0(_S98 + mid_1, &kernelContext_10);
        if((_S101.x) <= _S99)
        {
            lo_1 = mid_1;
        }
        else
        {
            hi_1 = mid_1 - 1U;
        }
    }
    uint slot_3 = _S98 + lo_1;
    uint4 _S102 = uint4(*((&kernelContext_10)->slots_0+slot_3)) ;
    if((_S102.w) == 0U)
    {
        return;
    }
    uint _S103 = 2U * slot_3;
    float3 _S104 = (float4(*((&kernelContext_10)->extras_0+_S103)) ).xyz;
    thread Quat_0 _S105 = quat_of_0(float4(*((&kernelContext_10)->extras_0+(_S103 + 1U))) );
    Quat_0 _S106 = conj_0(&_S105);
    uint4 _S107 = part_range_0((uint4(*((&kernelContext_10)->islands_of_0+_S95)) ).x + (&kernelContext_10)->work_0[_S93 + i_3], &kernelContext_10);
    uint device* _S108 = (&kernelContext_10)->member_offsets_0+i_3;
    uint _S109 = _S107.x;
    uint k_2 = _S109;
    uint col_0 = *_S108;
    for(;;)
    {
        if(k_2 < (_S109 + _S107.y))
        {
        }
        else
        {
            break;
        }
        uint _S110 = 8U * k_2;
        float4 _S111 = float4(*((&kernelContext_10)->parts_0+(_S110 + 4U))) ;
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_10)->parts_0+(_S110 + 5U))) );
        float4 _S112 = float4(*((&kernelContext_10)->parts_0+(_S110 + 6U))) ;
        float4 _S113 = float4(*((&kernelContext_10)->parts_0+(_S110 + 7U))) ;
        if(((&kernelContext_10)->params_0->live_0) != 0U)
        {
            lo_1 = col_0;
        }
        else
        {
            lo_1 = col_0 - *((&kernelContext_10)->control_0+5U);
        }
        uint cw_0 = 4U * lo_1;
        *((&kernelContext_10)->colliders_0+cw_0) = packed_float4(float4((as_type<float>((_S102.x))), _S112.x, _S112.y, _S112.z)) ;
        packed_float4 device* _S114 = (&kernelContext_10)->colliders_0+(cw_0 + 1U);
        float3 _S115 = _S111.xyz - _S104;
        thread Quat_0 _S116 = _S106;
        float3 _S117 = rotate_0(&_S116, _S115);
        *_S114 = packed_float4(float4(_S117, _S111.w)) ;
        packed_float4 device* _S118 = (&kernelContext_10)->colliders_0+(cw_0 + 2U);
        thread Quat_0 _S119 = _S106;
        thread Quat_0 _S120 = rot_1;
        Quat_0 _S121 = quat_mul_0(&_S119, &_S120);
        thread Quat_0 _S122 = _S121;
        Quat_0 _S123 = normalized_0(&_S122);
        thread Quat_0 _S124 = _S123;
        float4 _S125 = quat_vec_0(&_S124);
        *_S118 = packed_float4(_S125) ;
        *((&kernelContext_10)->colliders_0+(cw_0 + 3U)) = packed_float4(_S113) ;
        uint _S126 = col_0 + 1U;
        k_2 = k_2 + 1U;
        col_0 = _S126;
    }
    return;
}

[[kernel]] void spawn_retire(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], uint device* control_4 [[buffer(15)]], packed_uint4 device* parents_4 [[buffer(1)]], packed_uint4 device* islands_of_4 [[buffer(12)]], uint device* work_4 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_4 [[buffer(7)]], uint device* structure_base_4 [[buffer(14)]], packed_uint4 device* part_ranges_4 [[buffer(5)]], uint device* member_offsets_4 [[buffer(16)]], SpawnParams_0 constant* params_4 [[buffer(0)]], packed_float4 device* children_4 [[buffer(3)]], packed_uint4 device* slots_4 [[buffer(9)]], packed_float4 device* parts_4 [[buffer(6)]], packed_float4 device* frames_4 [[buffer(4)]], packed_float4 device* state_4 [[buffer(8)]], packed_float4 device* bodies_5 [[buffer(10)]], packed_float4 device* extras_4 [[buffer(13)]], packed_float4 device* colliders_4 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_11;
    (&kernelContext_11)->control_0 = control_4;
    (&kernelContext_11)->parents_0 = parents_4;
    (&kernelContext_11)->islands_of_0 = islands_of_4;
    (&kernelContext_11)->work_0 = work_4;
    (&kernelContext_11)->chunks_0 = chunks_4;
    (&kernelContext_11)->structure_base_0 = structure_base_4;
    (&kernelContext_11)->part_ranges_0 = part_ranges_4;
    (&kernelContext_11)->member_offsets_0 = member_offsets_4;
    (&kernelContext_11)->params_0 = params_4;
    (&kernelContext_11)->children_0 = children_4;
    (&kernelContext_11)->slots_0 = slots_4;
    (&kernelContext_11)->parts_0 = parts_4;
    (&kernelContext_11)->frames_0 = frames_4;
    (&kernelContext_11)->state_0 = state_4;
    (&kernelContext_11)->bodies_0 = bodies_5;
    (&kernelContext_11)->extras_0 = extras_4;
    (&kernelContext_11)->colliders_0 = colliders_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_11)->g_scan_0 = &g_scan_4;
    threadgroup uint g_split_parents_4;
    (&kernelContext_11)->g_split_parents_0 = &g_split_parents_4;
    uint p_2 = group_0.x;
    uint tid_2 = thread_1.x;
    bool _S127;
    if(p_2 >= (*(control_4+1U)))
    {
        _S127 = true;
    }
    else
    {
        _S127 = ((&kernelContext_11)->params_0->live_0) == 0U;
    }
    if(_S127)
    {
        _S127 = true;
    }
    else
    {
        _S127 = ((uint4(*((&kernelContext_11)->parents_0+p_2)) ).z) < 2U;
    }
    if(_S127)
    {
        _S127 = true;
    }
    else
    {
        _S127 = (*((&kernelContext_11)->control_0+int(0))) != 0U;
    }
    if(_S127)
    {
        return;
    }
    uint4 _S128 = uint4(*((&kernelContext_11)->islands_of_0+p_2)) ;
    uint c_3;
    if(tid_2 == 0U)
    {
        uint _S129 = _S128.y;
        uint _S130 = 7U * _S129;
        thread array<float4, int(7)> record_0;
        c_3 = 0U;
        for(;;)
        {
            if(c_3 < 7U)
            {
            }
            else
            {
                break;
            }
            record_0[c_3] = float4(*((&kernelContext_11)->bodies_0+(_S130 + c_3))) ;
            c_3 = c_3 + 1U;
        }
        record_0[int(0)].w = 0.0f;
        record_0[int(2)] = float4(0.0f, 0.0f, 0.0f, (as_type<float>((0U))));
        float3 _S131 = float3(0.0f) ;
        record_0[int(3)].xyz = _S131;
        record_0[int(4)].xyz = _S131;
        uint region_0 = 0U;
        for(;;)
        {
            if(region_0 < 2U)
            {
            }
            else
            {
                break;
            }
            c_3 = 0U;
            for(;;)
            {
                if(c_3 < 7U)
                {
                }
                else
                {
                    break;
                }
                *((&kernelContext_11)->bodies_0+(7U * (_S129 + region_0 * (&kernelContext_11)->params_0->max_bodies_0) + c_3)) = packed_float4(record_0[c_3]) ;
                c_3 = c_3 + 1U;
            }
            region_0 = region_0 + 1U;
        }
    }
    c_3 = _S128.z + tid_2;
    for(;;)
    {
        if(c_3 < (_S128.w))
        {
        }
        else
        {
            break;
        }
        uint _S132 = 4U * c_3;
        float4 _S133 = float4(*((&kernelContext_11)->colliders_0+_S132)) ;
        *((&kernelContext_11)->colliders_0+_S132) = packed_float4(float4(_S133.x, _S133.y, (as_type<float>(((as_type<uint>((_S133.z))) & 65535U))), (as_type<float>((32768U))))) ;
        c_3 = c_3 + 64U;
    }
    return;
}

