#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint group_uniform_0(uint value_0, uint thread_0)
{
    return value_0;
}

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
    uint moments_base_0;
    uint info_base_0;
    uint pad0_0;
    uint pad1_0;
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
    array<float4, int(64)> threadgroup* g_s0_0;
    array<float4, int(64)> threadgroup* g_s1_0;
    array<float4, int(64)> threadgroup* g_s2_0;
    array<float4, int(64)> threadgroup* g_s3_0;
    array<float4, int(64)> threadgroup* g_s4_0;
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

[[kernel]] void spawn_offsets(uint3 thread_1 [[thread_position_in_threadgroup]], uint device* control_1 [[buffer(15)]], packed_uint4 device* parents_1 [[buffer(1)]], packed_uint4 device* islands_of_1 [[buffer(12)]], uint device* work_1 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(7)]], uint device* structure_base_1 [[buffer(14)]], packed_uint4 device* part_ranges_1 [[buffer(5)]], uint device* member_offsets_1 [[buffer(16)]], SpawnParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* children_1 [[buffer(3)]], packed_uint4 device* slots_1 [[buffer(9)]], packed_float4 device* parts_1 [[buffer(6)]], packed_float4 device* frames_1 [[buffer(4)]], packed_float4 device* state_1 [[buffer(8)]], packed_float4 device* bodies_1 [[buffer(10)]], packed_float4 device* extras_1 [[buffer(13)]], packed_float4 device* colliders_1 [[buffer(11)]])
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
    threadgroup array<float4, int(64)> g_s0_1;
    (&kernelContext_7)->g_s0_0 = &g_s0_1;
    threadgroup array<float4, int(64)> g_s1_1;
    (&kernelContext_7)->g_s1_0 = &g_s1_1;
    threadgroup array<float4, int(64)> g_s2_1;
    (&kernelContext_7)->g_s2_0 = &g_s2_1;
    threadgroup array<float4, int(64)> g_s3_1;
    (&kernelContext_7)->g_s3_0 = &g_s3_1;
    threadgroup array<float4, int(64)> g_s4_1;
    (&kernelContext_7)->g_s4_0 = &g_s4_1;
    uint tid_1 = thread_1.x;
    uint body0_0 = *(control_1+4U);
    uint collider0_0 = *(control_1+5U);
    if((group_uniform_0(*(control_1+1U), tid_1)) == 0U)
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
    uint _S11 = group_uniform_0(_S10, tid_1);
    uint block_0 = 0U;
    uint carry_0 = 0U;
    for(;;)
    {
        if(block_0 < _S11)
        {
        }
        else
        {
            break;
        }
        uint i_2 = block_0 + tid_1;
        bool _S12 = i_2 < _S11;
        if(_S12)
        {
            uint _S13 = member_parts_0(i_2, _S11, &kernelContext_7);
            v_1 = _S13;
        }
        else
        {
            v_1 = 0U;
        }
        uint _S14 = block_scan_0(tid_1, v_1, &kernelContext_7);
        if(_S12)
        {
            *((&kernelContext_7)->member_offsets_0+i_2) = collider0_0 + carry_0 + _S14;
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
        if(block_0 < _S11)
        {
        }
        else
        {
            break;
        }
        uint s2_0 = block_0 + tid_1;
        if(s2_0 < _S11)
        {
            bool _S15 = slot_valid_0(s2_0, &kernelContext_7);
            fits_0 = _S15;
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
        uint _S16 = block_scan_0(tid_1, v_1, &kernelContext_7);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint bodies_needed_1 = bodies_needed_0 + (*(&kernelContext_7)->g_scan_0)[int(1023)];
        block_0 = block_0 + 1024U;
        bodies_needed_0 = bodies_needed_1;
    }
    bool _S17 = tid_1 == 0U;
    uint c_end_0;
    if(_S17)
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
    if(_S17)
    {
        uint device* _S18 = (&kernelContext_7)->control_0+0U;
        if(fits_0)
        {
            block_0 = 0U;
        }
        else
        {
            block_0 = 1U;
        }
        *_S18 = block_0;
        uint device* _S19 = (&kernelContext_7)->control_0+8U;
        if(fits_0)
        {
            block_0 = body0_0 + bodies_needed_0;
        }
        else
        {
            block_0 = body0_0;
        }
        *_S19 = block_0;
        uint device* _S20 = (&kernelContext_7)->control_0+9U;
        if(fits_0)
        {
            block_0 = collider0_0 + carry_0;
        }
        else
        {
            block_0 = collider0_0;
        }
        *_S20 = block_0;
    }
    if(fits_0)
    {
        block_0 = 1U;
    }
    else
    {
        block_0 = 0U;
    }
    if((group_uniform_0(block_0, tid_1)) == 0U)
    {
        return;
    }
    block_0 = 0U;
    v_1 = 0U;
    for(;;)
    {
        if(block_0 < _S11)
        {
        }
        else
        {
            break;
        }
        uint s2_1 = block_0 + tid_1;
        bool _S21 = s2_1 < _S11;
        bool valid_0;
        if(_S21)
        {
            bool _S22 = slot_valid_0(s2_1, &kernelContext_7);
            valid_0 = _S22;
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
        uint _S23 = block_scan_0(tid_1, bodies_needed_0, &kernelContext_7);
        if(_S21)
        {
            if(valid_0)
            {
                uint _S24 = parent_at_0(s2_1, &kernelContext_7);
                uint4 _S25 = uint4(*((&kernelContext_7)->parents_0+_S24)) ;
                uint2 _S26 = child_members_0(s2_1, &kernelContext_7);
                uint first_0 = _S25.y + _S26.x;
                uint end_0 = first_0 + _S26.y;
                uint c_first_0 = *((&kernelContext_7)->member_offsets_0+first_0);
                if(end_0 < _S11)
                {
                    c_end_0 = *((&kernelContext_7)->member_offsets_0+end_0);
                }
                else
                {
                    c_end_0 = collider0_0 + carry_0;
                }
                *((&kernelContext_7)->slots_0+s2_1) = packed_uint4(uint4(body0_0 + v_1 + _S23, c_first_0, c_end_0 - c_first_0, 1U)) ;
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
        uint _S27 = p_0 + 1U;
        bool _S28;
        if(_S27 < (*(kernelContext_8->control_0+1U)))
        {
            _S28 = ((uint4(*(kernelContext_8->parents_0+_S27)) ).y) <= slot_1;
        }
        else
        {
            _S28 = false;
        }
        if(_S28)
        {
        }
        else
        {
            break;
        }
        p_0 = _S27;
    }
    return p_0;
}

void child_mass_0(uint first_1, uint2 m_0, uint members_0, float thread* mass_0, float3 thread* com_0, float3 thread* i0_0, float3 thread* i1_0, float3 thread* i2_0, KernelContext_0 thread* kernelContext_9)
{
    uint _S29;
    uint k_0;
    *mass_0 = 0.0f;
    float3 _S30 = float3(0.0f) ;
    uint j_0 = 0U;
    float3 moment_0 = _S30;
    for(;;)
    {
        uint _S31 = m_0.y;
        _S29 = _S31;
        if(j_0 < _S31)
        {
        }
        else
        {
            break;
        }
        uint4 _S32 = part_range_0(first_1 + kernelContext_9->work_0[members_0 + j_0], kernelContext_9);
        uint _S33 = _S32.x;
        k_0 = _S33;
        for(;;)
        {
            if(k_0 < (_S33 + _S32.y))
            {
            }
            else
            {
                break;
            }
            float4 _S34 = float4(*(kernelContext_9->parts_0+8U * k_0)) ;
            float _S35 = _S34.w;
            *mass_0 = *mass_0 + _S35;
            float3 moment_1 = moment_0 + _S34.xyz * float3(_S35) ;
            k_0 = k_0 + 1U;
            moment_0 = moment_1;
        }
        j_0 = j_0 + 1U;
    }
    *com_0 = moment_0 / float3(max(*mass_0, 1.00000000317107685e-30f)) ;
    *i0_0 = _S30;
    *i1_0 = _S30;
    *i2_0 = _S30;
    j_0 = 0U;
    for(;;)
    {
        if(j_0 < _S29)
        {
        }
        else
        {
            break;
        }
        uint4 _S36 = part_range_0(first_1 + kernelContext_9->work_0[members_0 + j_0], kernelContext_9);
        uint _S37 = _S36.x;
        k_0 = _S37;
        for(;;)
        {
            if(k_0 < (_S37 + _S36.y))
            {
            }
            else
            {
                break;
            }
            uint _S38 = 8U * k_0;
            float4 _S39 = float4(*(kernelContext_9->parts_0+_S38)) ;
            float3 d_0 = _S39.xyz - *com_0;
            float dd_0 = dot(d_0, d_0);
            float3 _S40 = float3(_S39.w) ;
            *i0_0 = *i0_0 + ((float4(*(kernelContext_9->parts_0+(_S38 + 1U))) ).xyz + (float3(dd_0, 0.0f, 0.0f) - float3(d_0.x)  * d_0) * _S40);
            *i1_0 = *i1_0 + ((float4(*(kernelContext_9->parts_0+(_S38 + 2U))) ).xyz + (float3(0.0f, dd_0, 0.0f) - float3(d_0.y)  * d_0) * _S40);
            *i2_0 = *i2_0 + ((float4(*(kernelContext_9->parts_0+(_S38 + 3U))) ).xyz + (float3(0.0f, 0.0f, dd_0) - float3(d_0.z)  * d_0) * _S40);
            k_0 = k_0 + 1U;
        }
        j_0 = j_0 + 1U;
    }
    return;
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
    array<float, int(3)> _S41 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S42 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S43 = { { r2_0.x, r2_0.y, r2_0.z } };
    thread array<array<float, int(3)>, int(3)> a_0;
    a_0[int(0)] = _S41;
    a_0[int(1)] = _S42;
    a_0[int(2)] = _S43;
    array<float, int(3)> _S44 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S45 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S46 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_2;
    v_2[int(0)] = _S44;
    v_2[int(1)] = _S45;
    v_2[int(2)] = _S46;
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
        uint k_1 = 0U;
        for(;;)
        {
            if(k_1 < 3U)
            {
            }
            else
            {
                break;
            }
            bool _S47;
            if(k_1 != p_1)
            {
                _S47 = k_1 != q_3;
            }
            else
            {
                _S47 = false;
            }
            if(_S47)
            {
                float kp_0 = a_0[k_1][p_1];
                float kq_0 = a_0[k_1][q_3];
                float _S48 = c_0 * a_0[k_1][p_1] - s_1 * a_0[k_1][q_3];
                a_0[k_1][p_1] = _S48;
                a_0[p_1][k_1] = _S48;
                float _S49 = s_1 * kp_0 + c_0 * kq_0;
                a_0[k_1][q_3] = _S49;
                a_0[q_3][k_1] = _S49;
            }
            k_1 = k_1 + 1U;
        }
        float _S50 = c_0 * c_0;
        float _S51 = 2.0f * s_1 * c_0 * apq_0;
        float _S52 = s_1 * s_1;
        a_0[p_1][p_1] = _S50 * app_0 - _S51 + _S52 * aqq_0;
        a_0[q_3][q_3] = _S52 * app_0 + _S51 + _S50 * aqq_0;
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
    thread Quat_0 _S53 = quat_from_columns_0(x_1, y_1, z_2);
    Quat_0 _S54 = normalized_0(&_S53);
    *rot_0 = _S54;
    *moments_0 = float3(a_0[int(0)][int(0)], a_0[int(1)][int(1)], a_0[int(2)][int(2)]);
    return;
}

Quat_0 quat_of_0(float4 q_4)
{
    thread Quat_0 r_1;
    (&r_1)->x_0 = q_4.x;
    (&r_1)->y_0 = q_4.y;
    (&r_1)->z_0 = q_4.z;
    (&r_1)->w_0 = q_4.w;
    return r_1;
}

float3 rotate_0(const Quat_0 thread* q_5, float3 v_3)
{
    float3 qv_0 = float3(q_5->x_0, q_5->y_0, q_5->z_0);
    float3 t_0 = cross(qv_0, v_3) * float3(2.0f) ;
    return v_3 + t_0 * float3(q_5->w_0)  + cross(qv_0, t_0);
}

float3 child_pose_0(uint p_2, uint first_2, uint2 m_1, uint members_1, bool anchored_0, KernelContext_0 thread* kernelContext_10)
{
    uint _S55 = 5U * p_2;
    float3 pose_pos_0 = (float4(*(kernelContext_10->frames_0+(_S55 + 4U))) ).xyz;
    float3 pose_pos_1;
    if(!anchored_0)
    {
        Quat_0 pose_rot_0 = quat_of_0(float4(*(kernelContext_10->frames_0+_S55)) );
        float3 _S56 = float3(0.0f) ;
        uint j_1 = 0U;
        float sm_0 = 0.0f;
        pose_pos_1 = _S56;
        for(;;)
        {
            if(j_1 < (m_1.y))
            {
            }
            else
            {
                break;
            }
            uint c_1 = first_2 + kernelContext_10->work_0[members_1 + j_1];
            float cm_0 = (float4((kernelContext_10->chunks_0+c_1)->center_0) ).w;
            float sm_1 = sm_0 + cm_0;
            float3 su_0 = pose_pos_1 + (float4(*(kernelContext_10->state_0+4U * c_1)) ).xyz * float3(cm_0) ;
            j_1 = j_1 + 1U;
            sm_0 = sm_1;
            pose_pos_1 = su_0;
        }
        float3 _S57 = pose_pos_1 / float3(sm_0) ;
        thread Quat_0 _S58 = pose_rot_0;
        float3 _S59 = rotate_0(&_S58, _S57);
        pose_pos_1 = pose_pos_0 + _S59;
    }
    else
    {
        pose_pos_1 = pose_pos_0;
    }
    return pose_pos_1;
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

Quat_0 rot_to_rubble_0(const Quat_0 thread* q_6)
{
    thread Quat_0 axes_0;
    (&axes_0)->w_0 = 0.70710676908493042f;
    (&axes_0)->x_0 = -0.70710676908493042f;
    (&axes_0)->y_0 = 0.0f;
    (&axes_0)->z_0 = 0.0f;
    thread Quat_0 _S60 = axes_0;
    Quat_0 _S61 = quat_mul_0(&_S60, q_6);
    thread Quat_0 _S62 = _S61;
    Quat_0 _S63 = normalized_0(&_S62);
    return _S63;
}

float3 to_rubble_0(float3 v_4)
{
    return float3(v_4.x, v_4.z, - v_4.y);
}

float4 quat_vec_0(const Quat_0 thread* q_7)
{
    return float4(q_7->x_0, q_7->y_0, q_7->z_0, q_7->w_0);
}

void child_record_0(uint slot_2, uint p_3, bool anchored_1, float mass_1, float3 com_1, const Quat_0 thread* principal_0, float3 moments_1, float3 pose_pos_2, KernelContext_0 thread* kernelContext_11)
{
    Quat_0 pose_rot_1 = quat_of_0(float4(*(kernelContext_11->frames_0+5U * p_3)) );
    uint4 _S64 = uint4(*(kernelContext_11->slots_0+slot_2)) ;
    thread Quat_0 _S65 = pose_rot_1;
    Quat_0 _S66 = quat_mul_0(&_S65, principal_0);
    thread Quat_0 _S67 = _S66;
    Quat_0 _S68 = normalized_0(&_S67);
    thread Quat_0 _S69 = _S68;
    Quat_0 _S70 = rot_to_rubble_0(&_S69);
    bool _S71;
    if(anchored_1)
    {
        _S71 = true;
    }
    else
    {
        _S71 = !(mass_1 > 0.0f);
    }
    float inv_mass_0;
    if(_S71)
    {
        inv_mass_0 = 0.0f;
    }
    else
    {
        inv_mass_0 = 1.0f / mass_1;
    }
    float3 _S72 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_1)
    {
        float _S73 = moments_1.x;
        float _S74;
        if(_S73 > 0.0f)
        {
            _S74 = 1.0f / _S73;
        }
        else
        {
            _S74 = 0.0f;
        }
        float _S75 = moments_1.y;
        float _S76;
        if(_S75 > 0.0f)
        {
            _S76 = 1.0f / _S75;
        }
        else
        {
            _S76 = 0.0f;
        }
        float _S77 = moments_1.z;
        float _S78;
        if(_S77 > 0.0f)
        {
            _S78 = 1.0f / _S77;
        }
        else
        {
            _S78 = 0.0f;
        }
        inv_inertia_0 = float3(_S74, _S76, _S78);
    }
    else
    {
        inv_inertia_0 = _S72;
    }
    float3 velocity_0;
    if(anchored_1)
    {
        velocity_0 = _S72;
    }
    else
    {
        velocity_0 = to_rubble_0((float4(*(kernelContext_11->children_0+(8U * slot_2 + 5U))) ).xyz);
    }
    float3 spin_0;
    if(anchored_1)
    {
        spin_0 = _S72;
    }
    else
    {
        spin_0 = to_rubble_0((float4(*(kernelContext_11->children_0+(8U * slot_2 + 6U))) ).xyz);
    }
    uint flags_0;
    if(anchored_1)
    {
        flags_0 = 0U;
    }
    else
    {
        flags_0 = 9U;
    }
    thread Quat_0 _S79 = pose_rot_1;
    float3 _S80 = rotate_0(&_S79, com_1);
    float4 _S81 = float4(to_rubble_0(pose_pos_2 + _S80), inv_mass_0);
    thread Quat_0 _S82 = _S70;
    float4 _S83 = quat_vec_0(&_S82);
    array<float4, int(7)> _S84 = { { _S81, _S83, float4(inv_inertia_0, (as_type<float>((flags_0)))), float4(velocity_0, 0.0f), float4(spin_0, 0.0f), float4(0.0f) , float4(0.0f, 0.0f, 0.0f, 1.0f) } };
    if((kernelContext_11->params_0->live_0) != 0U)
    {
        flags_0 = _S64.x;
    }
    else
    {
        flags_0 = _S64.x - *(kernelContext_11->control_0+4U);
    }
    uint _S85 = 7U * flags_0;
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
        *(kernelContext_11->bodies_0+(_S85 + w_1)) = packed_float4(_S84[w_1]) ;
        w_1 = w_1 + 1U;
    }
    if((kernelContext_11->params_0->live_0) != 0U)
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
            *(kernelContext_11->bodies_0+(7U * (kernelContext_11->params_0->max_bodies_0 + _S64.x) + w_1)) = packed_float4(_S84[w_1]) ;
            w_1 = w_1 + 1U;
        }
    }
    uint _S86 = 2U * slot_2;
    *(kernelContext_11->extras_0+_S86) = packed_float4(float4(com_1, mass_1)) ;
    packed_float4 device* _S87 = kernelContext_11->extras_0+(_S86 + 1U);
    float4 _S88 = quat_vec_0(principal_0);
    *_S87 = packed_float4(_S88) ;
    return;
}

[[kernel]] void spawn_bodies(uint3 id_0 [[thread_position_in_grid]], uint device* control_2 [[buffer(15)]], packed_uint4 device* parents_2 [[buffer(1)]], packed_uint4 device* islands_of_2 [[buffer(12)]], uint device* work_2 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(7)]], uint device* structure_base_2 [[buffer(14)]], packed_uint4 device* part_ranges_2 [[buffer(5)]], uint device* member_offsets_2 [[buffer(16)]], SpawnParams_0 constant* params_2 [[buffer(0)]], packed_float4 device* children_2 [[buffer(3)]], packed_uint4 device* slots_2 [[buffer(9)]], packed_float4 device* parts_2 [[buffer(6)]], packed_float4 device* frames_2 [[buffer(4)]], packed_float4 device* state_2 [[buffer(8)]], packed_float4 device* bodies_3 [[buffer(10)]], packed_float4 device* extras_2 [[buffer(13)]], packed_float4 device* colliders_2 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_12;
    (&kernelContext_12)->control_0 = control_2;
    (&kernelContext_12)->parents_0 = parents_2;
    (&kernelContext_12)->islands_of_0 = islands_of_2;
    (&kernelContext_12)->work_0 = work_2;
    (&kernelContext_12)->chunks_0 = chunks_2;
    (&kernelContext_12)->structure_base_0 = structure_base_2;
    (&kernelContext_12)->part_ranges_0 = part_ranges_2;
    (&kernelContext_12)->member_offsets_0 = member_offsets_2;
    (&kernelContext_12)->params_0 = params_2;
    (&kernelContext_12)->children_0 = children_2;
    (&kernelContext_12)->slots_0 = slots_2;
    (&kernelContext_12)->parts_0 = parts_2;
    (&kernelContext_12)->frames_0 = frames_2;
    (&kernelContext_12)->state_0 = state_2;
    (&kernelContext_12)->bodies_0 = bodies_3;
    (&kernelContext_12)->extras_0 = extras_2;
    (&kernelContext_12)->colliders_0 = colliders_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_12)->g_scan_0 = &g_scan_2;
    threadgroup uint g_split_parents_2;
    (&kernelContext_12)->g_split_parents_0 = &g_split_parents_2;
    threadgroup array<float4, int(64)> g_s0_2;
    (&kernelContext_12)->g_s0_0 = &g_s0_2;
    threadgroup array<float4, int(64)> g_s1_2;
    (&kernelContext_12)->g_s1_0 = &g_s1_2;
    threadgroup array<float4, int(64)> g_s2_2;
    (&kernelContext_12)->g_s2_0 = &g_s2_2;
    threadgroup array<float4, int(64)> g_s3_2;
    (&kernelContext_12)->g_s3_0 = &g_s3_2;
    threadgroup array<float4, int(64)> g_s4_2;
    (&kernelContext_12)->g_s4_0 = &g_s4_2;
    uint slot_3 = id_0.x;
    uint _S89 = total_slots_0(&kernelContext_12);
    if(slot_3 >= _S89)
    {
        return;
    }
    uint _S90 = parent_of_0(slot_3, &kernelContext_12);
    uint4 _S91 = uint4(*((&kernelContext_12)->parents_0+_S90)) ;
    uint _S92 = _S91.y;
    bool _S93;
    if((slot_3 - _S92) >= (_S91.z))
    {
        _S93 = true;
    }
    else
    {
        _S93 = ((uint4(*((&kernelContext_12)->slots_0+slot_3)) ).w) == 0U;
    }
    if(_S93)
    {
        _S93 = true;
    }
    else
    {
        _S93 = (*((&kernelContext_12)->control_0+0U)) != 0U;
    }
    if(_S93)
    {
        return;
    }
    uint first_3 = (uint4(*((&kernelContext_12)->islands_of_0+_S90)) ).x;
    uint2 _S94 = child_members_0(slot_3, &kernelContext_12);
    bool anchored_2 = (as_type<uint>(((float4(*((&kernelContext_12)->children_0+8U * slot_3)) ).w))) != 0U;
    uint members_2 = _S89 + _S92 + _S94.x;
    thread float mass_2;
    thread float3 com_2;
    thread float3 i0_1;
    thread float3 i1_1;
    thread float3 i2_1;
    child_mass_0(first_3, _S94, members_2, &mass_2, &com_2, &i0_1, &i1_1, &i2_1, &kernelContext_12);
    thread Quat_0 principal_1;
    thread float3 moments_2;
    principal_axes_0(i0_1, i1_1, i2_1, &principal_1, &moments_2);
    float3 _S95 = child_pose_0(_S90, first_3, _S94, members_2, anchored_2, &kernelContext_12);
    thread Quat_0 _S96 = principal_1;
    child_record_0(slot_3, _S90, anchored_2, mass_2, com_2, &_S96, moments_2, _S95, &kernelContext_12);
    return;
}

bool child_spawns_0(uint slot_4, uint total_1, uint thread* p_4, uint4 thread* parent_0, KernelContext_0 thread* kernelContext_13)
{
    *p_4 = 0U;
    *parent_0 = uint4(int4(int(0)) );
    bool _S97;
    if(slot_4 >= total_1)
    {
        _S97 = true;
    }
    else
    {
        _S97 = (*(kernelContext_13->control_0+0U)) != 0U;
    }
    if(_S97)
    {
        return false;
    }
    uint _S98 = parent_of_0(slot_4, kernelContext_13);
    *p_4 = _S98;
    uint4 _S99 = uint4(*(kernelContext_13->parents_0+_S98)) ;
    *parent_0 = _S99;
    if((_S99.z) >= 2U)
    {
        _S97 = (slot_4 - (*parent_0).y) < ((*parent_0).z);
    }
    else
    {
        _S97 = false;
    }
    return _S97;
}

[[kernel]] void spawn_bodies_mass(uint3 id_1 [[thread_position_in_grid]], uint device* control_3 [[buffer(15)]], packed_uint4 device* parents_3 [[buffer(1)]], packed_uint4 device* islands_of_3 [[buffer(12)]], uint device* work_3 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(7)]], uint device* structure_base_3 [[buffer(14)]], packed_uint4 device* part_ranges_3 [[buffer(5)]], uint device* member_offsets_3 [[buffer(16)]], SpawnParams_0 constant* params_3 [[buffer(0)]], packed_float4 device* children_3 [[buffer(3)]], packed_uint4 device* slots_3 [[buffer(9)]], packed_float4 device* parts_3 [[buffer(6)]], packed_float4 device* frames_3 [[buffer(4)]], packed_float4 device* state_3 [[buffer(8)]], packed_float4 device* bodies_4 [[buffer(10)]], packed_float4 device* extras_3 [[buffer(13)]], packed_float4 device* colliders_3 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_14;
    (&kernelContext_14)->control_0 = control_3;
    (&kernelContext_14)->parents_0 = parents_3;
    (&kernelContext_14)->islands_of_0 = islands_of_3;
    (&kernelContext_14)->work_0 = work_3;
    (&kernelContext_14)->chunks_0 = chunks_3;
    (&kernelContext_14)->structure_base_0 = structure_base_3;
    (&kernelContext_14)->part_ranges_0 = part_ranges_3;
    (&kernelContext_14)->member_offsets_0 = member_offsets_3;
    (&kernelContext_14)->params_0 = params_3;
    (&kernelContext_14)->children_0 = children_3;
    (&kernelContext_14)->slots_0 = slots_3;
    (&kernelContext_14)->parts_0 = parts_3;
    (&kernelContext_14)->frames_0 = frames_3;
    (&kernelContext_14)->state_0 = state_3;
    (&kernelContext_14)->bodies_0 = bodies_4;
    (&kernelContext_14)->extras_0 = extras_3;
    (&kernelContext_14)->colliders_0 = colliders_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_14)->g_scan_0 = &g_scan_3;
    threadgroup uint g_split_parents_3;
    (&kernelContext_14)->g_split_parents_0 = &g_split_parents_3;
    threadgroup array<float4, int(64)> g_s0_3;
    (&kernelContext_14)->g_s0_0 = &g_s0_3;
    threadgroup array<float4, int(64)> g_s1_3;
    (&kernelContext_14)->g_s1_0 = &g_s1_3;
    threadgroup array<float4, int(64)> g_s2_3;
    (&kernelContext_14)->g_s2_0 = &g_s2_3;
    threadgroup array<float4, int(64)> g_s3_3;
    (&kernelContext_14)->g_s3_0 = &g_s3_3;
    threadgroup array<float4, int(64)> g_s4_3;
    (&kernelContext_14)->g_s4_0 = &g_s4_3;
    uint slot_5 = id_1.x;
    uint _S100 = total_slots_0(&kernelContext_14);
    thread uint p_5;
    thread uint4 parent_1;
    bool _S101 = child_spawns_0(slot_5, _S100, &p_5, &parent_1, &kernelContext_14);
    if(!_S101)
    {
        return;
    }
    uint2 _S102 = child_members_0(slot_5, &kernelContext_14);
    uint _S103 = (uint4(*((&kernelContext_14)->islands_of_0+p_5)) ).x;
    uint _S104 = total_slots_0(&kernelContext_14);
    thread float mass_3;
    thread float3 com_3;
    thread float3 i0_2;
    thread float3 i1_2;
    thread float3 i2_2;
    child_mass_0(_S103, _S102, _S104 + parent_1.y + _S102.x, &mass_3, &com_3, &i0_2, &i1_2, &i2_2, &kernelContext_14);
    thread Quat_0 principal_2;
    thread float3 moments_3;
    principal_axes_0(i0_2, i1_2, i2_2, &principal_2, &moments_3);
    uint _S105 = 2U * slot_5;
    *((&kernelContext_14)->extras_0+_S105) = packed_float4(float4(com_3, mass_3)) ;
    packed_float4 device* _S106 = (&kernelContext_14)->extras_0+(_S105 + 1U);
    thread Quat_0 _S107 = principal_2;
    float4 _S108 = quat_vec_0(&_S107);
    *_S106 = packed_float4(_S108) ;
    *((&kernelContext_14)->extras_0+((&kernelContext_14)->params_0->moments_base_0 + _S105)) = packed_float4(float4(moments_3, 0.0f)) ;
    return;
}

[[kernel]] void spawn_bodies_pose(uint3 id_2 [[thread_position_in_grid]], uint device* control_4 [[buffer(15)]], packed_uint4 device* parents_4 [[buffer(1)]], packed_uint4 device* islands_of_4 [[buffer(12)]], uint device* work_4 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_4 [[buffer(7)]], uint device* structure_base_4 [[buffer(14)]], packed_uint4 device* part_ranges_4 [[buffer(5)]], uint device* member_offsets_4 [[buffer(16)]], SpawnParams_0 constant* params_4 [[buffer(0)]], packed_float4 device* children_4 [[buffer(3)]], packed_uint4 device* slots_4 [[buffer(9)]], packed_float4 device* parts_4 [[buffer(6)]], packed_float4 device* frames_4 [[buffer(4)]], packed_float4 device* state_4 [[buffer(8)]], packed_float4 device* bodies_5 [[buffer(10)]], packed_float4 device* extras_4 [[buffer(13)]], packed_float4 device* colliders_4 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_15;
    (&kernelContext_15)->control_0 = control_4;
    (&kernelContext_15)->parents_0 = parents_4;
    (&kernelContext_15)->islands_of_0 = islands_of_4;
    (&kernelContext_15)->work_0 = work_4;
    (&kernelContext_15)->chunks_0 = chunks_4;
    (&kernelContext_15)->structure_base_0 = structure_base_4;
    (&kernelContext_15)->part_ranges_0 = part_ranges_4;
    (&kernelContext_15)->member_offsets_0 = member_offsets_4;
    (&kernelContext_15)->params_0 = params_4;
    (&kernelContext_15)->children_0 = children_4;
    (&kernelContext_15)->slots_0 = slots_4;
    (&kernelContext_15)->parts_0 = parts_4;
    (&kernelContext_15)->frames_0 = frames_4;
    (&kernelContext_15)->state_0 = state_4;
    (&kernelContext_15)->bodies_0 = bodies_5;
    (&kernelContext_15)->extras_0 = extras_4;
    (&kernelContext_15)->colliders_0 = colliders_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_15)->g_scan_0 = &g_scan_4;
    threadgroup uint g_split_parents_4;
    (&kernelContext_15)->g_split_parents_0 = &g_split_parents_4;
    threadgroup array<float4, int(64)> g_s0_4;
    (&kernelContext_15)->g_s0_0 = &g_s0_4;
    threadgroup array<float4, int(64)> g_s1_4;
    (&kernelContext_15)->g_s1_0 = &g_s1_4;
    threadgroup array<float4, int(64)> g_s2_4;
    (&kernelContext_15)->g_s2_0 = &g_s2_4;
    threadgroup array<float4, int(64)> g_s3_4;
    (&kernelContext_15)->g_s3_0 = &g_s3_4;
    threadgroup array<float4, int(64)> g_s4_4;
    (&kernelContext_15)->g_s4_0 = &g_s4_4;
    uint slot_6 = id_2.x;
    uint _S109 = total_slots_0(&kernelContext_15);
    thread uint p_6;
    thread uint4 parent_2;
    bool _S110 = child_spawns_0(slot_6, _S109, &p_6, &parent_2, &kernelContext_15);
    if(!_S110)
    {
        return;
    }
    uint2 _S111 = child_members_0(slot_6, &kernelContext_15);
    bool anchored_3 = (as_type<uint>(((float4(*((&kernelContext_15)->children_0+8U * slot_6)) ).w))) != 0U;
    packed_float4 device* _S112 = (&kernelContext_15)->extras_0+((&kernelContext_15)->params_0->moments_base_0 + 2U * slot_6 + 1U);
    uint _S113 = (uint4(*((&kernelContext_15)->islands_of_0+p_6)) ).x;
    uint _S114 = total_slots_0(&kernelContext_15);
    float3 _S115 = child_pose_0(p_6, _S113, _S111, _S114 + parent_2.y + _S111.x, anchored_3, &kernelContext_15);
    *_S112 = packed_float4(float4(_S115, 0.0f)) ;
    return;
}

void phased_record_0(uint slot_7, uint p_7, KernelContext_0 thread* kernelContext_16)
{
    uint _S116 = 2U * slot_7;
    float4 _S117 = float4(*(kernelContext_16->extras_0+_S116)) ;
    bool anchored_4 = (as_type<uint>(((float4(*(kernelContext_16->children_0+8U * slot_7)) ).w))) != 0U;
    float _S118 = _S117.w;
    float3 _S119 = _S117.xyz;
    float3 _S120 = (float4(*(kernelContext_16->extras_0+(kernelContext_16->params_0->moments_base_0 + _S116))) ).xyz;
    float3 _S121 = (float4(*(kernelContext_16->extras_0+(kernelContext_16->params_0->moments_base_0 + _S116 + 1U))) ).xyz;
    thread Quat_0 _S122 = quat_of_0(float4(*(kernelContext_16->extras_0+(_S116 + 1U))) );
    child_record_0(slot_7, p_7, anchored_4, _S118, _S119, &_S122, _S120, _S121, kernelContext_16);
    return;
}

[[kernel]] void spawn_bodies_record(uint3 id_3 [[thread_position_in_grid]], uint device* control_5 [[buffer(15)]], packed_uint4 device* parents_5 [[buffer(1)]], packed_uint4 device* islands_of_5 [[buffer(12)]], uint device* work_5 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_5 [[buffer(7)]], uint device* structure_base_5 [[buffer(14)]], packed_uint4 device* part_ranges_5 [[buffer(5)]], uint device* member_offsets_5 [[buffer(16)]], SpawnParams_0 constant* params_5 [[buffer(0)]], packed_float4 device* children_5 [[buffer(3)]], packed_uint4 device* slots_5 [[buffer(9)]], packed_float4 device* parts_5 [[buffer(6)]], packed_float4 device* frames_5 [[buffer(4)]], packed_float4 device* state_5 [[buffer(8)]], packed_float4 device* bodies_6 [[buffer(10)]], packed_float4 device* extras_5 [[buffer(13)]], packed_float4 device* colliders_5 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_17;
    (&kernelContext_17)->control_0 = control_5;
    (&kernelContext_17)->parents_0 = parents_5;
    (&kernelContext_17)->islands_of_0 = islands_of_5;
    (&kernelContext_17)->work_0 = work_5;
    (&kernelContext_17)->chunks_0 = chunks_5;
    (&kernelContext_17)->structure_base_0 = structure_base_5;
    (&kernelContext_17)->part_ranges_0 = part_ranges_5;
    (&kernelContext_17)->member_offsets_0 = member_offsets_5;
    (&kernelContext_17)->params_0 = params_5;
    (&kernelContext_17)->children_0 = children_5;
    (&kernelContext_17)->slots_0 = slots_5;
    (&kernelContext_17)->parts_0 = parts_5;
    (&kernelContext_17)->frames_0 = frames_5;
    (&kernelContext_17)->state_0 = state_5;
    (&kernelContext_17)->bodies_0 = bodies_6;
    (&kernelContext_17)->extras_0 = extras_5;
    (&kernelContext_17)->colliders_0 = colliders_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_17)->g_scan_0 = &g_scan_5;
    threadgroup uint g_split_parents_5;
    (&kernelContext_17)->g_split_parents_0 = &g_split_parents_5;
    threadgroup array<float4, int(64)> g_s0_5;
    (&kernelContext_17)->g_s0_0 = &g_s0_5;
    threadgroup array<float4, int(64)> g_s1_5;
    (&kernelContext_17)->g_s1_0 = &g_s1_5;
    threadgroup array<float4, int(64)> g_s2_5;
    (&kernelContext_17)->g_s2_0 = &g_s2_5;
    threadgroup array<float4, int(64)> g_s3_5;
    (&kernelContext_17)->g_s3_0 = &g_s3_5;
    threadgroup array<float4, int(64)> g_s4_5;
    (&kernelContext_17)->g_s4_0 = &g_s4_5;
    uint slot_8 = id_3.x;
    uint _S123 = total_slots_0(&kernelContext_17);
    thread uint p_8;
    thread uint4 parent_3;
    bool _S124 = child_spawns_0(slot_8, _S123, &p_8, &parent_3, &kernelContext_17);
    bool _S125;
    if(!_S124)
    {
        _S125 = true;
    }
    else
    {
        _S125 = ((uint4(*((&kernelContext_17)->slots_0+slot_8)) ).w) == 0U;
    }
    if(_S125)
    {
        return;
    }
    phased_record_0(slot_8, p_8, &kernelContext_17);
    return;
}

void spawn_tree_0(uint tid_2, KernelContext_0 thread* kernelContext_18)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_2 = 32U;
    for(;;)
    {
        if(s_2 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_2 < s_2)
        {
            uint _S126 = tid_2 + s_2;
            (*kernelContext_18->g_s0_0)[tid_2] = (*kernelContext_18->g_s0_0)[tid_2] + (*kernelContext_18->g_s0_0)[_S126];
            (*kernelContext_18->g_s1_0)[tid_2] = (*kernelContext_18->g_s1_0)[tid_2] + (*kernelContext_18->g_s1_0)[_S126];
            (*kernelContext_18->g_s2_0)[tid_2] = (*kernelContext_18->g_s2_0)[tid_2] + (*kernelContext_18->g_s2_0)[_S126];
            (*kernelContext_18->g_s3_0)[tid_2] = (*kernelContext_18->g_s3_0)[tid_2] + (*kernelContext_18->g_s3_0)[_S126];
            (*kernelContext_18->g_s4_0)[tid_2] = (*kernelContext_18->g_s4_0)[tid_2] + (*kernelContext_18->g_s4_0)[_S126];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_2 = s_2 >> 1U;
    }
    return;
}

void tree_of_0(uint tid_3, float4 a_2, float4 b_0, float4 c_2, float4 d_1, float4 e_0, KernelContext_0 thread* kernelContext_19)
{
    (*kernelContext_19->g_s0_0)[tid_3] = a_2;
    (*kernelContext_19->g_s1_0)[tid_3] = b_0;
    (*kernelContext_19->g_s2_0)[tid_3] = c_2;
    (*kernelContext_19->g_s3_0)[tid_3] = d_1;
    (*kernelContext_19->g_s4_0)[tid_3] = e_0;
    spawn_tree_0(tid_3, kernelContext_19);
    return;
}

void tree_of_1(uint tid_4, float4 a_3, float4 b_1, float4 c_3, float4 d_2, float4 e_1, KernelContext_0 thread* kernelContext_20)
{
    (*kernelContext_20->g_s0_0)[tid_4] = a_3;
    (*kernelContext_20->g_s1_0)[tid_4] = b_1;
    (*kernelContext_20->g_s2_0)[tid_4] = c_3;
    (*kernelContext_20->g_s3_0)[tid_4] = d_2;
    (*kernelContext_20->g_s4_0)[tid_4] = e_1;
    spawn_tree_0(tid_4, kernelContext_20);
    return;
}

void group_first_tree_0(uint first_4, uint2 m_2, uint members_3, uint tid_5, KernelContext_0 thread* kernelContext_21)
{
    float4 _S127 = float4(0.0f) ;
    uint j_2 = tid_5;
    float4 mm_0 = _S127;
    float4 su_1 = _S127;
    for(;;)
    {
        if(j_2 < (m_2.y))
        {
        }
        else
        {
            break;
        }
        uint c_4 = first_4 + kernelContext_21->work_0[members_3 + j_2];
        uint4 _S128 = part_range_0(c_4, kernelContext_21);
        uint _S129 = _S128.x;
        uint k_2 = _S129;
        for(;;)
        {
            if(k_2 < (_S129 + _S128.y))
            {
            }
            else
            {
                break;
            }
            float4 _S130 = float4(*(kernelContext_21->parts_0+8U * k_2)) ;
            float _S131 = _S130.w;
            float4 mm_1 = mm_0 + float4(_S130.xyz * float3(_S131) , _S131);
            k_2 = k_2 + 1U;
            mm_0 = mm_1;
        }
        float cm_1 = (float4((kernelContext_21->chunks_0+c_4)->center_0) ).w;
        float4 su_2 = su_1 + float4((float4(*(kernelContext_21->state_0+4U * c_4)) ).xyz * float3(cm_1) , cm_1);
        j_2 = j_2 + 64U;
        su_1 = su_2;
    }
    tree_of_0(tid_5, mm_0, _S127, _S127, _S127, su_1, kernelContext_21);
    return;
}

void group_inertia_tree_0(uint first_5, uint2 m_3, uint members_4, uint tid_6, float3 com_4, float4 su_total_0, KernelContext_0 thread* kernelContext_22)
{
    float3 _S132 = float3(0.0f) ;
    uint j_3 = tid_6;
    float3 i0_3 = _S132;
    float3 i1_3 = _S132;
    float3 i2_3 = _S132;
    for(;;)
    {
        if(j_3 < (m_3.y))
        {
        }
        else
        {
            break;
        }
        uint4 _S133 = part_range_0(first_5 + kernelContext_22->work_0[members_4 + j_3], kernelContext_22);
        uint _S134 = _S133.x;
        uint k_3 = _S134;
        for(;;)
        {
            if(k_3 < (_S134 + _S133.y))
            {
            }
            else
            {
                break;
            }
            uint _S135 = 8U * k_3;
            float4 _S136 = float4(*(kernelContext_22->parts_0+_S135)) ;
            float3 d_3 = _S136.xyz - com_4;
            float dd_1 = dot(d_3, d_3);
            float3 _S137 = float3(_S136.w) ;
            float3 i0_4 = i0_3 + ((float4(*(kernelContext_22->parts_0+(_S135 + 1U))) ).xyz + (float3(dd_1, 0.0f, 0.0f) - float3(d_3.x)  * d_3) * _S137);
            float3 i1_4 = i1_3 + ((float4(*(kernelContext_22->parts_0+(_S135 + 2U))) ).xyz + (float3(0.0f, dd_1, 0.0f) - float3(d_3.y)  * d_3) * _S137);
            float3 i2_4 = i2_3 + ((float4(*(kernelContext_22->parts_0+(_S135 + 3U))) ).xyz + (float3(0.0f, 0.0f, dd_1) - float3(d_3.z)  * d_3) * _S137);
            k_3 = k_3 + 1U;
            i0_3 = i0_4;
            i1_3 = i1_4;
            i2_3 = i2_4;
        }
        j_3 = j_3 + 64U;
    }
    float4 _S138 = float4(0.0f) ;
    float4 _S139 = float4(i0_3, 0.0f);
    float4 _S140 = float4(i1_3, 0.0f);
    float4 _S141 = float4(i2_3, 0.0f);
    float4 _S142;
    if(tid_6 == 0U)
    {
        _S142 = su_total_0;
    }
    else
    {
        _S142 = _S138;
    }
    tree_of_0(tid_6, _S138, _S139, _S140, _S141, _S142, kernelContext_22);
    return;
}

[[kernel]] void spawn_bodies_group(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], uint device* control_6 [[buffer(15)]], packed_uint4 device* parents_6 [[buffer(1)]], packed_uint4 device* islands_of_6 [[buffer(12)]], uint device* work_6 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_6 [[buffer(7)]], uint device* structure_base_6 [[buffer(14)]], packed_uint4 device* part_ranges_6 [[buffer(5)]], uint device* member_offsets_6 [[buffer(16)]], SpawnParams_0 constant* params_6 [[buffer(0)]], packed_float4 device* children_6 [[buffer(3)]], packed_uint4 device* slots_6 [[buffer(9)]], packed_float4 device* parts_6 [[buffer(6)]], packed_float4 device* frames_6 [[buffer(4)]], packed_float4 device* state_6 [[buffer(8)]], packed_float4 device* bodies_7 [[buffer(10)]], packed_float4 device* extras_6 [[buffer(13)]], packed_float4 device* colliders_6 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_23;
    (&kernelContext_23)->control_0 = control_6;
    (&kernelContext_23)->parents_0 = parents_6;
    (&kernelContext_23)->islands_of_0 = islands_of_6;
    (&kernelContext_23)->work_0 = work_6;
    (&kernelContext_23)->chunks_0 = chunks_6;
    (&kernelContext_23)->structure_base_0 = structure_base_6;
    (&kernelContext_23)->part_ranges_0 = part_ranges_6;
    (&kernelContext_23)->member_offsets_0 = member_offsets_6;
    (&kernelContext_23)->params_0 = params_6;
    (&kernelContext_23)->children_0 = children_6;
    (&kernelContext_23)->slots_0 = slots_6;
    (&kernelContext_23)->parts_0 = parts_6;
    (&kernelContext_23)->frames_0 = frames_6;
    (&kernelContext_23)->state_0 = state_6;
    (&kernelContext_23)->bodies_0 = bodies_7;
    (&kernelContext_23)->extras_0 = extras_6;
    (&kernelContext_23)->colliders_0 = colliders_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_23)->g_scan_0 = &g_scan_6;
    threadgroup uint g_split_parents_6;
    (&kernelContext_23)->g_split_parents_0 = &g_split_parents_6;
    threadgroup array<float4, int(64)> g_s0_6;
    (&kernelContext_23)->g_s0_0 = &g_s0_6;
    threadgroup array<float4, int(64)> g_s1_6;
    (&kernelContext_23)->g_s1_0 = &g_s1_6;
    threadgroup array<float4, int(64)> g_s2_6;
    (&kernelContext_23)->g_s2_0 = &g_s2_6;
    threadgroup array<float4, int(64)> g_s3_6;
    (&kernelContext_23)->g_s3_0 = &g_s3_6;
    threadgroup array<float4, int(64)> g_s4_6;
    (&kernelContext_23)->g_s4_0 = &g_s4_6;
    uint slot_9 = group_0.x;
    uint tid_7 = thread_2.x;
    uint _S143 = total_slots_0(&kernelContext_23);
    bool _S144;
    if(slot_9 >= _S143)
    {
        _S144 = true;
    }
    else
    {
        _S144 = (*((&kernelContext_23)->control_0+1U)) == 0U;
    }
    if(_S144)
    {
        _S144 = true;
    }
    else
    {
        _S144 = (*((&kernelContext_23)->control_0+0U)) != 0U;
    }
    uint _S145;
    if(_S144)
    {
        _S145 = 1U;
    }
    else
    {
        _S145 = 0U;
    }
    if((group_uniform_0(_S145, tid_7)) != 0U)
    {
        return;
    }
    uint _S146 = parent_at_0(slot_9, &kernelContext_23);
    uint4 _S147 = uint4(*((&kernelContext_23)->parents_0+_S146)) ;
    uint _S148 = _S147.y;
    if((slot_9 - _S148) >= (_S147.z))
    {
        _S144 = true;
    }
    else
    {
        _S144 = ((uint4(*((&kernelContext_23)->slots_0+slot_9)) ).w) == 0U;
    }
    if(_S144)
    {
        _S145 = 1U;
    }
    else
    {
        _S145 = 0U;
    }
    if((group_uniform_0(_S145, tid_7)) != 0U)
    {
        return;
    }
    uint first_6 = (uint4(*((&kernelContext_23)->islands_of_0+_S146)) ).x;
    uint2 _S149 = child_members_0(slot_9, &kernelContext_23);
    bool anchored_5 = (as_type<uint>(((float4(*((&kernelContext_23)->children_0+8U * slot_9)) ).w))) != 0U;
    uint members_5 = _S143 + _S148 + _S149.x;
    group_first_tree_0(first_6, _S149, members_5, tid_7, &kernelContext_23);
    float mass_4 = (*(&kernelContext_23)->g_s0_0)[int(0)].w;
    float3 com_5 = (*(&kernelContext_23)->g_s0_0)[int(0)].xyz / float3(max(mass_4, 1.00000000317107685e-30f)) ;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    group_inertia_tree_0(first_6, _S149, members_5, tid_7, com_5, (*(&kernelContext_23)->g_s4_0)[int(0)], &kernelContext_23);
    if(tid_7 != 0U)
    {
        return;
    }
    thread Quat_0 principal_3;
    thread float3 moments_4;
    principal_axes_0((*(&kernelContext_23)->g_s1_0)[int(0)].xyz, (*(&kernelContext_23)->g_s2_0)[int(0)].xyz, (*(&kernelContext_23)->g_s3_0)[int(0)].xyz, &principal_3, &moments_4);
    uint _S150 = 5U * _S146;
    Quat_0 pose_rot_2 = quat_of_0(float4(*((&kernelContext_23)->frames_0+_S150)) );
    float3 pose_pos_3 = (float4(*((&kernelContext_23)->frames_0+(_S150 + 4U))) ).xyz;
    float3 pose_pos_4;
    if(!anchored_5)
    {
        float3 _S151 = (*(&kernelContext_23)->g_s4_0)[int(0)].xyz / float3((*(&kernelContext_23)->g_s4_0)[int(0)].w) ;
        thread Quat_0 _S152 = pose_rot_2;
        float3 _S153 = rotate_0(&_S152, _S151);
        pose_pos_4 = pose_pos_3 + _S153;
    }
    else
    {
        pose_pos_4 = pose_pos_3;
    }
    thread Quat_0 _S154 = principal_3;
    child_record_0(slot_9, _S146, anchored_5, mass_4, com_5, &_S154, moments_4, pose_pos_4, &kernelContext_23);
    return;
}

bool group_child_0(uint slot_10, uint tid_8, uint thread* p_9, uint4 thread* parent_4, KernelContext_0 thread* kernelContext_24)
{
    *p_9 = 0U;
    *parent_4 = uint4(int4(int(0)) );
    uint _S155 = total_slots_0(kernelContext_24);
    bool _S156;
    if(slot_10 >= _S155)
    {
        _S156 = true;
    }
    else
    {
        _S156 = (*(kernelContext_24->control_0+1U)) == 0U;
    }
    if(_S156)
    {
        _S156 = true;
    }
    else
    {
        _S156 = (*(kernelContext_24->control_0+0U)) != 0U;
    }
    uint _S157;
    if(_S156)
    {
        _S157 = 1U;
    }
    else
    {
        _S157 = 0U;
    }
    if((group_uniform_0(_S157, tid_8)) != 0U)
    {
        return false;
    }
    uint _S158 = parent_at_0(slot_10, kernelContext_24);
    *p_9 = _S158;
    uint4 _S159 = uint4(*(kernelContext_24->parents_0+_S158)) ;
    *parent_4 = _S159;
    if((slot_10 - _S159.y) >= (_S159.z))
    {
        _S156 = true;
    }
    else
    {
        _S156 = ((*parent_4).z) < 2U;
    }
    if(_S156)
    {
        _S157 = 1U;
    }
    else
    {
        _S157 = 0U;
    }
    return (group_uniform_0(_S157, tid_8)) == 0U;
}

void group_mass_tree_0(uint first_7, uint2 m_4, uint members_6, uint tid_9, KernelContext_0 thread* kernelContext_25)
{
    float4 _S160 = float4(0.0f) ;
    uint j_4 = tid_9;
    float4 mm_2 = _S160;
    for(;;)
    {
        if(j_4 < (m_4.y))
        {
        }
        else
        {
            break;
        }
        uint4 _S161 = part_range_0(first_7 + kernelContext_25->work_0[members_6 + j_4], kernelContext_25);
        uint _S162 = _S161.x;
        uint k_4 = _S162;
        for(;;)
        {
            if(k_4 < (_S162 + _S161.y))
            {
            }
            else
            {
                break;
            }
            float4 _S163 = float4(*(kernelContext_25->parts_0+8U * k_4)) ;
            float _S164 = _S163.w;
            float4 mm_3 = mm_2 + float4(_S163.xyz * float3(_S164) , _S164);
            k_4 = k_4 + 1U;
            mm_2 = mm_3;
        }
        j_4 = j_4 + 64U;
    }
    tree_of_0(tid_9, mm_2, _S160, _S160, _S160, _S160, kernelContext_25);
    return;
}

[[kernel]] void spawn_bodies_group_mass(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_3 [[thread_position_in_threadgroup]], uint device* control_7 [[buffer(15)]], packed_uint4 device* parents_7 [[buffer(1)]], packed_uint4 device* islands_of_7 [[buffer(12)]], uint device* work_7 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_7 [[buffer(7)]], uint device* structure_base_7 [[buffer(14)]], packed_uint4 device* part_ranges_7 [[buffer(5)]], uint device* member_offsets_7 [[buffer(16)]], SpawnParams_0 constant* params_7 [[buffer(0)]], packed_float4 device* children_7 [[buffer(3)]], packed_uint4 device* slots_7 [[buffer(9)]], packed_float4 device* parts_7 [[buffer(6)]], packed_float4 device* frames_7 [[buffer(4)]], packed_float4 device* state_7 [[buffer(8)]], packed_float4 device* bodies_8 [[buffer(10)]], packed_float4 device* extras_7 [[buffer(13)]], packed_float4 device* colliders_7 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_26;
    (&kernelContext_26)->control_0 = control_7;
    (&kernelContext_26)->parents_0 = parents_7;
    (&kernelContext_26)->islands_of_0 = islands_of_7;
    (&kernelContext_26)->work_0 = work_7;
    (&kernelContext_26)->chunks_0 = chunks_7;
    (&kernelContext_26)->structure_base_0 = structure_base_7;
    (&kernelContext_26)->part_ranges_0 = part_ranges_7;
    (&kernelContext_26)->member_offsets_0 = member_offsets_7;
    (&kernelContext_26)->params_0 = params_7;
    (&kernelContext_26)->children_0 = children_7;
    (&kernelContext_26)->slots_0 = slots_7;
    (&kernelContext_26)->parts_0 = parts_7;
    (&kernelContext_26)->frames_0 = frames_7;
    (&kernelContext_26)->state_0 = state_7;
    (&kernelContext_26)->bodies_0 = bodies_8;
    (&kernelContext_26)->extras_0 = extras_7;
    (&kernelContext_26)->colliders_0 = colliders_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_26)->g_scan_0 = &g_scan_7;
    threadgroup uint g_split_parents_7;
    (&kernelContext_26)->g_split_parents_0 = &g_split_parents_7;
    threadgroup array<float4, int(64)> g_s0_7;
    (&kernelContext_26)->g_s0_0 = &g_s0_7;
    threadgroup array<float4, int(64)> g_s1_7;
    (&kernelContext_26)->g_s1_0 = &g_s1_7;
    threadgroup array<float4, int(64)> g_s2_7;
    (&kernelContext_26)->g_s2_0 = &g_s2_7;
    threadgroup array<float4, int(64)> g_s3_7;
    (&kernelContext_26)->g_s3_0 = &g_s3_7;
    threadgroup array<float4, int(64)> g_s4_7;
    (&kernelContext_26)->g_s4_0 = &g_s4_7;
    uint slot_11 = group_1.x;
    uint tid_10 = thread_3.x;
    thread uint p_10;
    thread uint4 parent_5;
    bool _S165 = group_child_0(slot_11, tid_10, &p_10, &parent_5, &kernelContext_26);
    if(!_S165)
    {
        return;
    }
    uint first_8 = (uint4(*((&kernelContext_26)->islands_of_0+p_10)) ).x;
    uint2 _S166 = child_members_0(slot_11, &kernelContext_26);
    uint _S167 = total_slots_0(&kernelContext_26);
    uint members_7 = _S167 + parent_5.y + _S166.x;
    group_mass_tree_0(first_8, _S166, members_7, tid_10, &kernelContext_26);
    float mass_5 = (*(&kernelContext_26)->g_s0_0)[int(0)].w;
    float3 com_6 = (*(&kernelContext_26)->g_s0_0)[int(0)].xyz / float3(max(mass_5, 1.00000000317107685e-30f)) ;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    group_inertia_tree_0(first_8, _S166, members_7, tid_10, com_6, float4(0.0f) , &kernelContext_26);
    if(tid_10 != 0U)
    {
        return;
    }
    thread Quat_0 principal_4;
    thread float3 moments_5;
    principal_axes_0((*(&kernelContext_26)->g_s1_0)[int(0)].xyz, (*(&kernelContext_26)->g_s2_0)[int(0)].xyz, (*(&kernelContext_26)->g_s3_0)[int(0)].xyz, &principal_4, &moments_5);
    uint _S168 = 2U * slot_11;
    *((&kernelContext_26)->extras_0+_S168) = packed_float4(float4(com_6, mass_5)) ;
    packed_float4 device* _S169 = (&kernelContext_26)->extras_0+(_S168 + 1U);
    thread Quat_0 _S170 = principal_4;
    float4 _S171 = quat_vec_0(&_S170);
    *_S169 = packed_float4(_S171) ;
    *((&kernelContext_26)->extras_0+((&kernelContext_26)->params_0->moments_base_0 + _S168)) = packed_float4(float4(moments_5, 0.0f)) ;
    return;
}

void group_su_tree_0(uint first_9, uint2 m_5, uint members_8, uint tid_11, KernelContext_0 thread* kernelContext_27)
{
    float4 _S172 = float4(0.0f) ;
    uint j_5 = tid_11;
    float4 su_3 = _S172;
    for(;;)
    {
        if(j_5 < (m_5.y))
        {
        }
        else
        {
            break;
        }
        uint c_5 = first_9 + kernelContext_27->work_0[members_8 + j_5];
        float cm_2 = (float4((kernelContext_27->chunks_0+c_5)->center_0) ).w;
        float4 su_4 = su_3 + float4((float4(*(kernelContext_27->state_0+4U * c_5)) ).xyz * float3(cm_2) , cm_2);
        j_5 = j_5 + 64U;
        su_3 = su_4;
    }
    tree_of_0(tid_11, _S172, _S172, _S172, _S172, su_3, kernelContext_27);
    return;
}

void group_carry_tree_0(uint tid_12, float4 su_total_1, KernelContext_0 thread* kernelContext_28)
{
    float4 _S173 = float4(0.0f) ;
    float4 _S174;
    if(tid_12 == 0U)
    {
        _S174 = su_total_1;
    }
    else
    {
        _S174 = _S173;
    }
    tree_of_1(tid_12, _S173, _S173, _S173, _S173, _S174, kernelContext_28);
    return;
}

[[kernel]] void spawn_bodies_group_pose(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], uint device* control_8 [[buffer(15)]], packed_uint4 device* parents_8 [[buffer(1)]], packed_uint4 device* islands_of_8 [[buffer(12)]], uint device* work_8 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_8 [[buffer(7)]], uint device* structure_base_8 [[buffer(14)]], packed_uint4 device* part_ranges_8 [[buffer(5)]], uint device* member_offsets_8 [[buffer(16)]], SpawnParams_0 constant* params_8 [[buffer(0)]], packed_float4 device* children_8 [[buffer(3)]], packed_uint4 device* slots_8 [[buffer(9)]], packed_float4 device* parts_8 [[buffer(6)]], packed_float4 device* frames_8 [[buffer(4)]], packed_float4 device* state_8 [[buffer(8)]], packed_float4 device* bodies_9 [[buffer(10)]], packed_float4 device* extras_8 [[buffer(13)]], packed_float4 device* colliders_8 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_29;
    (&kernelContext_29)->control_0 = control_8;
    (&kernelContext_29)->parents_0 = parents_8;
    (&kernelContext_29)->islands_of_0 = islands_of_8;
    (&kernelContext_29)->work_0 = work_8;
    (&kernelContext_29)->chunks_0 = chunks_8;
    (&kernelContext_29)->structure_base_0 = structure_base_8;
    (&kernelContext_29)->part_ranges_0 = part_ranges_8;
    (&kernelContext_29)->member_offsets_0 = member_offsets_8;
    (&kernelContext_29)->params_0 = params_8;
    (&kernelContext_29)->children_0 = children_8;
    (&kernelContext_29)->slots_0 = slots_8;
    (&kernelContext_29)->parts_0 = parts_8;
    (&kernelContext_29)->frames_0 = frames_8;
    (&kernelContext_29)->state_0 = state_8;
    (&kernelContext_29)->bodies_0 = bodies_9;
    (&kernelContext_29)->extras_0 = extras_8;
    (&kernelContext_29)->colliders_0 = colliders_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_29)->g_scan_0 = &g_scan_8;
    threadgroup uint g_split_parents_8;
    (&kernelContext_29)->g_split_parents_0 = &g_split_parents_8;
    threadgroup array<float4, int(64)> g_s0_8;
    (&kernelContext_29)->g_s0_0 = &g_s0_8;
    threadgroup array<float4, int(64)> g_s1_8;
    (&kernelContext_29)->g_s1_0 = &g_s1_8;
    threadgroup array<float4, int(64)> g_s2_8;
    (&kernelContext_29)->g_s2_0 = &g_s2_8;
    threadgroup array<float4, int(64)> g_s3_8;
    (&kernelContext_29)->g_s3_0 = &g_s3_8;
    threadgroup array<float4, int(64)> g_s4_8;
    (&kernelContext_29)->g_s4_0 = &g_s4_8;
    uint slot_12 = group_2.x;
    uint tid_13 = thread_4.x;
    thread uint p_11;
    thread uint4 parent_6;
    bool _S175 = group_child_0(slot_12, tid_13, &p_11, &parent_6, &kernelContext_29);
    if(!_S175)
    {
        return;
    }
    uint first_10 = (uint4(*((&kernelContext_29)->islands_of_0+p_11)) ).x;
    uint2 _S176 = child_members_0(slot_12, &kernelContext_29);
    uint _S177 = total_slots_0(&kernelContext_29);
    group_su_tree_0(first_10, _S176, _S177 + parent_6.y + _S176.x, tid_13, &kernelContext_29);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    group_carry_tree_0(tid_13, (*(&kernelContext_29)->g_s4_0)[int(0)], &kernelContext_29);
    if(tid_13 != 0U)
    {
        return;
    }
    Quat_0 pose_rot_3 = quat_of_0(float4(*((&kernelContext_29)->frames_0+5U * p_11)) );
    float3 pose_pos_5 = (float4(*((&kernelContext_29)->frames_0+(5U * p_11 + 4U))) ).xyz;
    float3 pose_pos_6;
    if(!((as_type<uint>(((float4(*((&kernelContext_29)->children_0+8U * slot_12)) ).w))) != 0U))
    {
        float3 _S178 = (*(&kernelContext_29)->g_s4_0)[int(0)].xyz / float3((*(&kernelContext_29)->g_s4_0)[int(0)].w) ;
        thread Quat_0 _S179 = pose_rot_3;
        float3 _S180 = rotate_0(&_S179, _S178);
        pose_pos_6 = pose_pos_5 + _S180;
    }
    else
    {
        pose_pos_6 = pose_pos_5;
    }
    *((&kernelContext_29)->extras_0+((&kernelContext_29)->params_0->moments_base_0 + 2U * slot_12 + 1U)) = packed_float4(float4(pose_pos_6, 0.0f)) ;
    return;
}

[[kernel]] void spawn_bodies_group_record(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], uint device* control_9 [[buffer(15)]], packed_uint4 device* parents_9 [[buffer(1)]], packed_uint4 device* islands_of_9 [[buffer(12)]], uint device* work_9 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_9 [[buffer(7)]], uint device* structure_base_9 [[buffer(14)]], packed_uint4 device* part_ranges_9 [[buffer(5)]], uint device* member_offsets_9 [[buffer(16)]], SpawnParams_0 constant* params_9 [[buffer(0)]], packed_float4 device* children_9 [[buffer(3)]], packed_uint4 device* slots_9 [[buffer(9)]], packed_float4 device* parts_9 [[buffer(6)]], packed_float4 device* frames_9 [[buffer(4)]], packed_float4 device* state_9 [[buffer(8)]], packed_float4 device* bodies_10 [[buffer(10)]], packed_float4 device* extras_9 [[buffer(13)]], packed_float4 device* colliders_9 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_30;
    (&kernelContext_30)->control_0 = control_9;
    (&kernelContext_30)->parents_0 = parents_9;
    (&kernelContext_30)->islands_of_0 = islands_of_9;
    (&kernelContext_30)->work_0 = work_9;
    (&kernelContext_30)->chunks_0 = chunks_9;
    (&kernelContext_30)->structure_base_0 = structure_base_9;
    (&kernelContext_30)->part_ranges_0 = part_ranges_9;
    (&kernelContext_30)->member_offsets_0 = member_offsets_9;
    (&kernelContext_30)->params_0 = params_9;
    (&kernelContext_30)->children_0 = children_9;
    (&kernelContext_30)->slots_0 = slots_9;
    (&kernelContext_30)->parts_0 = parts_9;
    (&kernelContext_30)->frames_0 = frames_9;
    (&kernelContext_30)->state_0 = state_9;
    (&kernelContext_30)->bodies_0 = bodies_10;
    (&kernelContext_30)->extras_0 = extras_9;
    (&kernelContext_30)->colliders_0 = colliders_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_30)->g_scan_0 = &g_scan_9;
    threadgroup uint g_split_parents_9;
    (&kernelContext_30)->g_split_parents_0 = &g_split_parents_9;
    threadgroup array<float4, int(64)> g_s0_9;
    (&kernelContext_30)->g_s0_0 = &g_s0_9;
    threadgroup array<float4, int(64)> g_s1_9;
    (&kernelContext_30)->g_s1_0 = &g_s1_9;
    threadgroup array<float4, int(64)> g_s2_9;
    (&kernelContext_30)->g_s2_0 = &g_s2_9;
    threadgroup array<float4, int(64)> g_s3_9;
    (&kernelContext_30)->g_s3_0 = &g_s3_9;
    threadgroup array<float4, int(64)> g_s4_9;
    (&kernelContext_30)->g_s4_0 = &g_s4_9;
    uint slot_13 = group_3.x;
    thread uint p_12;
    thread uint4 parent_7;
    bool _S181;
    if((thread_5.x) != 0U)
    {
        _S181 = true;
    }
    else
    {
        uint _S182 = total_slots_0(&kernelContext_30);
        bool _S183 = child_spawns_0(slot_13, _S182, &p_12, &parent_7, &kernelContext_30);
        _S181 = !_S183;
    }
    if(_S181)
    {
        _S181 = true;
    }
    else
    {
        _S181 = ((uint4(*((&kernelContext_30)->slots_0+slot_13)) ).w) == 0U;
    }
    if(_S181)
    {
        return;
    }
    phased_record_0(slot_13, p_12, &kernelContext_30);
    return;
}

uint4 member_info_0(uint i_3, uint total_2, uint thread* col_0, KernelContext_0 thread* kernelContext_31)
{
    *col_0 = 0U;
    uint _S184 = parent_at_0(i_3, kernelContext_31);
    uint4 _S185 = uint4(*(kernelContext_31->parents_0+_S184)) ;
    uint _S186 = _S185.z;
    if(_S186 < 2U)
    {
        return uint4(4294967295U, 0U, 0U, 0U);
    }
    uint _S187 = _S185.y;
    uint _S188 = i_3 - _S187;
    uint _S189 = _S186 - 1U;
    uint lo_1 = 0U;
    uint hi_1 = _S189;
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
        uint2 _S190 = child_members_0(_S187 + mid_1, kernelContext_31);
        if((_S190.x) <= _S188)
        {
            lo_1 = mid_1;
        }
        else
        {
            hi_1 = mid_1 - 1U;
        }
    }
    uint slot_14 = _S187 + lo_1;
    uint4 _S191 = uint4(*(kernelContext_31->slots_0+slot_14)) ;
    if((_S191.w) == 0U)
    {
        return uint4(4294967295U, 0U, 0U, 0U);
    }
    uint4 _S192 = part_range_0((uint4(*(kernelContext_31->islands_of_0+_S184)) ).x + kernelContext_31->work_0[total_2 + i_3], kernelContext_31);
    *col_0 = *(kernelContext_31->member_offsets_0+i_3);
    return uint4(slot_14, _S191.x, _S192.x, _S192.y);
}

Quat_0 conj_0(const Quat_0 thread* q_8)
{
    thread Quat_0 c_6;
    (&c_6)->w_0 = q_8->w_0;
    (&c_6)->x_0 = - q_8->x_0;
    (&c_6)->y_0 = - q_8->y_0;
    (&c_6)->z_0 = - q_8->z_0;
    return c_6;
}

void member_colliders_0(uint4 info_1, uint col_1, KernelContext_0 thread* kernelContext_32)
{
    uint _S193 = 2U * info_1.x;
    float3 _S194 = (float4(*(kernelContext_32->extras_0+_S193)) ).xyz;
    thread Quat_0 _S195 = quat_of_0(float4(*(kernelContext_32->extras_0+(_S193 + 1U))) );
    Quat_0 _S196 = conj_0(&_S195);
    uint _S197 = info_1.z;
    uint k_5 = _S197;
    uint _S198 = col_1;
    for(;;)
    {
        if(k_5 < (_S197 + info_1.w))
        {
        }
        else
        {
            break;
        }
        uint _S199 = 8U * k_5;
        float4 _S200 = float4(*(kernelContext_32->parts_0+(_S199 + 4U))) ;
        Quat_0 rot_1 = quat_of_0(float4(*(kernelContext_32->parts_0+(_S199 + 5U))) );
        float4 _S201 = float4(*(kernelContext_32->parts_0+(_S199 + 6U))) ;
        float4 _S202 = float4(*(kernelContext_32->parts_0+(_S199 + 7U))) ;
        uint _S203;
        if((kernelContext_32->params_0->live_0) != 0U)
        {
            _S203 = _S198;
        }
        else
        {
            _S203 = _S198 - *(kernelContext_32->control_0+5U);
        }
        uint cw_0 = 4U * _S203;
        *(kernelContext_32->colliders_0+cw_0) = packed_float4(float4((as_type<float>((info_1.y))), _S201.x, _S201.y, _S201.z)) ;
        packed_float4 device* _S204 = kernelContext_32->colliders_0+(cw_0 + 1U);
        float3 _S205 = _S200.xyz - _S194;
        thread Quat_0 _S206 = _S196;
        float3 _S207 = rotate_0(&_S206, _S205);
        *_S204 = packed_float4(float4(_S207, _S200.w)) ;
        packed_float4 device* _S208 = kernelContext_32->colliders_0+(cw_0 + 2U);
        thread Quat_0 _S209 = _S196;
        thread Quat_0 _S210 = rot_1;
        Quat_0 _S211 = quat_mul_0(&_S209, &_S210);
        thread Quat_0 _S212 = _S211;
        Quat_0 _S213 = normalized_0(&_S212);
        thread Quat_0 _S214 = _S213;
        float4 _S215 = quat_vec_0(&_S214);
        *_S208 = packed_float4(_S215) ;
        *(kernelContext_32->colliders_0+(cw_0 + 3U)) = packed_float4(_S202) ;
        uint _S216 = _S198 + 1U;
        k_5 = k_5 + 1U;
        _S198 = _S216;
    }
    return;
}

[[kernel]] void spawn_colliders(uint3 id_4 [[thread_position_in_grid]], uint device* control_10 [[buffer(15)]], packed_uint4 device* parents_10 [[buffer(1)]], packed_uint4 device* islands_of_10 [[buffer(12)]], uint device* work_10 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_10 [[buffer(7)]], uint device* structure_base_10 [[buffer(14)]], packed_uint4 device* part_ranges_10 [[buffer(5)]], uint device* member_offsets_10 [[buffer(16)]], SpawnParams_0 constant* params_10 [[buffer(0)]], packed_float4 device* children_10 [[buffer(3)]], packed_uint4 device* slots_10 [[buffer(9)]], packed_float4 device* parts_10 [[buffer(6)]], packed_float4 device* frames_10 [[buffer(4)]], packed_float4 device* state_10 [[buffer(8)]], packed_float4 device* bodies_11 [[buffer(10)]], packed_float4 device* extras_10 [[buffer(13)]], packed_float4 device* colliders_10 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_33;
    (&kernelContext_33)->control_0 = control_10;
    (&kernelContext_33)->parents_0 = parents_10;
    (&kernelContext_33)->islands_of_0 = islands_of_10;
    (&kernelContext_33)->work_0 = work_10;
    (&kernelContext_33)->chunks_0 = chunks_10;
    (&kernelContext_33)->structure_base_0 = structure_base_10;
    (&kernelContext_33)->part_ranges_0 = part_ranges_10;
    (&kernelContext_33)->member_offsets_0 = member_offsets_10;
    (&kernelContext_33)->params_0 = params_10;
    (&kernelContext_33)->children_0 = children_10;
    (&kernelContext_33)->slots_0 = slots_10;
    (&kernelContext_33)->parts_0 = parts_10;
    (&kernelContext_33)->frames_0 = frames_10;
    (&kernelContext_33)->state_0 = state_10;
    (&kernelContext_33)->bodies_0 = bodies_11;
    (&kernelContext_33)->extras_0 = extras_10;
    (&kernelContext_33)->colliders_0 = colliders_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_33)->g_scan_0 = &g_scan_10;
    threadgroup uint g_split_parents_10;
    (&kernelContext_33)->g_split_parents_0 = &g_split_parents_10;
    threadgroup array<float4, int(64)> g_s0_10;
    (&kernelContext_33)->g_s0_0 = &g_s0_10;
    threadgroup array<float4, int(64)> g_s1_10;
    (&kernelContext_33)->g_s1_0 = &g_s1_10;
    threadgroup array<float4, int(64)> g_s2_10;
    (&kernelContext_33)->g_s2_0 = &g_s2_10;
    threadgroup array<float4, int(64)> g_s3_10;
    (&kernelContext_33)->g_s3_0 = &g_s3_10;
    threadgroup array<float4, int(64)> g_s4_10;
    (&kernelContext_33)->g_s4_0 = &g_s4_10;
    uint i_4 = id_4.x;
    uint _S217 = total_slots_0(&kernelContext_33);
    bool _S218;
    if(i_4 >= _S217)
    {
        _S218 = true;
    }
    else
    {
        _S218 = (*((&kernelContext_33)->control_0+1U)) == 0U;
    }
    if(_S218)
    {
        _S218 = true;
    }
    else
    {
        _S218 = (*((&kernelContext_33)->control_0+0U)) != 0U;
    }
    if(_S218)
    {
        return;
    }
    thread uint col_2;
    uint4 _S219 = member_info_0(i_4, _S217, &col_2, &kernelContext_33);
    if((_S219.x) != 4294967295U)
    {
        member_colliders_0(_S219, col_2, &kernelContext_33);
    }
    return;
}

[[kernel]] void spawn_collider_members(uint3 id_5 [[thread_position_in_grid]], uint device* control_11 [[buffer(15)]], packed_uint4 device* parents_11 [[buffer(1)]], packed_uint4 device* islands_of_11 [[buffer(12)]], uint device* work_11 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_11 [[buffer(7)]], uint device* structure_base_11 [[buffer(14)]], packed_uint4 device* part_ranges_11 [[buffer(5)]], uint device* member_offsets_11 [[buffer(16)]], SpawnParams_0 constant* params_11 [[buffer(0)]], packed_float4 device* children_11 [[buffer(3)]], packed_uint4 device* slots_11 [[buffer(9)]], packed_float4 device* parts_11 [[buffer(6)]], packed_float4 device* frames_11 [[buffer(4)]], packed_float4 device* state_11 [[buffer(8)]], packed_float4 device* bodies_12 [[buffer(10)]], packed_float4 device* extras_11 [[buffer(13)]], packed_float4 device* colliders_11 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_34;
    (&kernelContext_34)->control_0 = control_11;
    (&kernelContext_34)->parents_0 = parents_11;
    (&kernelContext_34)->islands_of_0 = islands_of_11;
    (&kernelContext_34)->work_0 = work_11;
    (&kernelContext_34)->chunks_0 = chunks_11;
    (&kernelContext_34)->structure_base_0 = structure_base_11;
    (&kernelContext_34)->part_ranges_0 = part_ranges_11;
    (&kernelContext_34)->member_offsets_0 = member_offsets_11;
    (&kernelContext_34)->params_0 = params_11;
    (&kernelContext_34)->children_0 = children_11;
    (&kernelContext_34)->slots_0 = slots_11;
    (&kernelContext_34)->parts_0 = parts_11;
    (&kernelContext_34)->frames_0 = frames_11;
    (&kernelContext_34)->state_0 = state_11;
    (&kernelContext_34)->bodies_0 = bodies_12;
    (&kernelContext_34)->extras_0 = extras_11;
    (&kernelContext_34)->colliders_0 = colliders_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_34)->g_scan_0 = &g_scan_11;
    threadgroup uint g_split_parents_11;
    (&kernelContext_34)->g_split_parents_0 = &g_split_parents_11;
    threadgroup array<float4, int(64)> g_s0_11;
    (&kernelContext_34)->g_s0_0 = &g_s0_11;
    threadgroup array<float4, int(64)> g_s1_11;
    (&kernelContext_34)->g_s1_0 = &g_s1_11;
    threadgroup array<float4, int(64)> g_s2_11;
    (&kernelContext_34)->g_s2_0 = &g_s2_11;
    threadgroup array<float4, int(64)> g_s3_11;
    (&kernelContext_34)->g_s3_0 = &g_s3_11;
    threadgroup array<float4, int(64)> g_s4_11;
    (&kernelContext_34)->g_s4_0 = &g_s4_11;
    uint i_5 = id_5.x;
    uint _S220 = total_slots_0(&kernelContext_34);
    bool _S221;
    if(i_5 >= _S220)
    {
        _S221 = true;
    }
    else
    {
        _S221 = (*((&kernelContext_34)->control_0+1U)) == 0U;
    }
    if(_S221)
    {
        _S221 = true;
    }
    else
    {
        _S221 = (*((&kernelContext_34)->control_0+0U)) != 0U;
    }
    if(_S221)
    {
        return;
    }
    thread uint col_3;
    uint4 _S222 = member_info_0(i_5, _S220, &col_3, &kernelContext_34);
    uint w_2 = 0U;
    for(;;)
    {
        if(w_2 < 4U)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_34)->member_offsets_0+((&kernelContext_34)->params_0->info_base_0 + 4U * i_5 + w_2)) = _S222[w_2];
        w_2 = w_2 + 1U;
    }
    return;
}

[[kernel]] void spawn_colliders_phased(uint3 id_6 [[thread_position_in_grid]], uint device* control_12 [[buffer(15)]], packed_uint4 device* parents_12 [[buffer(1)]], packed_uint4 device* islands_of_12 [[buffer(12)]], uint device* work_12 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_12 [[buffer(7)]], uint device* structure_base_12 [[buffer(14)]], packed_uint4 device* part_ranges_12 [[buffer(5)]], uint device* member_offsets_12 [[buffer(16)]], SpawnParams_0 constant* params_12 [[buffer(0)]], packed_float4 device* children_12 [[buffer(3)]], packed_uint4 device* slots_12 [[buffer(9)]], packed_float4 device* parts_12 [[buffer(6)]], packed_float4 device* frames_12 [[buffer(4)]], packed_float4 device* state_12 [[buffer(8)]], packed_float4 device* bodies_13 [[buffer(10)]], packed_float4 device* extras_12 [[buffer(13)]], packed_float4 device* colliders_12 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_35;
    (&kernelContext_35)->control_0 = control_12;
    (&kernelContext_35)->parents_0 = parents_12;
    (&kernelContext_35)->islands_of_0 = islands_of_12;
    (&kernelContext_35)->work_0 = work_12;
    (&kernelContext_35)->chunks_0 = chunks_12;
    (&kernelContext_35)->structure_base_0 = structure_base_12;
    (&kernelContext_35)->part_ranges_0 = part_ranges_12;
    (&kernelContext_35)->member_offsets_0 = member_offsets_12;
    (&kernelContext_35)->params_0 = params_12;
    (&kernelContext_35)->children_0 = children_12;
    (&kernelContext_35)->slots_0 = slots_12;
    (&kernelContext_35)->parts_0 = parts_12;
    (&kernelContext_35)->frames_0 = frames_12;
    (&kernelContext_35)->state_0 = state_12;
    (&kernelContext_35)->bodies_0 = bodies_13;
    (&kernelContext_35)->extras_0 = extras_12;
    (&kernelContext_35)->colliders_0 = colliders_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_35)->g_scan_0 = &g_scan_12;
    threadgroup uint g_split_parents_12;
    (&kernelContext_35)->g_split_parents_0 = &g_split_parents_12;
    threadgroup array<float4, int(64)> g_s0_12;
    (&kernelContext_35)->g_s0_0 = &g_s0_12;
    threadgroup array<float4, int(64)> g_s1_12;
    (&kernelContext_35)->g_s1_0 = &g_s1_12;
    threadgroup array<float4, int(64)> g_s2_12;
    (&kernelContext_35)->g_s2_0 = &g_s2_12;
    threadgroup array<float4, int(64)> g_s3_12;
    (&kernelContext_35)->g_s3_0 = &g_s3_12;
    threadgroup array<float4, int(64)> g_s4_12;
    (&kernelContext_35)->g_s4_0 = &g_s4_12;
    uint i_6 = id_6.x;
    uint _S223 = total_slots_0(&kernelContext_35);
    bool _S224;
    if(i_6 >= _S223)
    {
        _S224 = true;
    }
    else
    {
        _S224 = (*((&kernelContext_35)->control_0+1U)) == 0U;
    }
    if(_S224)
    {
        _S224 = true;
    }
    else
    {
        _S224 = (*((&kernelContext_35)->control_0+0U)) != 0U;
    }
    if(_S224)
    {
        return;
    }
    uint _S225 = 4U * i_6;
    uint device* _S226 = (&kernelContext_35)->member_offsets_0+((&kernelContext_35)->params_0->info_base_0 + _S225);
    uint4 info_2 = uint4(*_S226, *((&kernelContext_35)->member_offsets_0+((&kernelContext_35)->params_0->info_base_0 + _S225 + 1U)), *((&kernelContext_35)->member_offsets_0+((&kernelContext_35)->params_0->info_base_0 + _S225 + 2U)), *((&kernelContext_35)->member_offsets_0+((&kernelContext_35)->params_0->info_base_0 + _S225 + 3U)));
    if((*_S226) != 4294967295U)
    {
        member_colliders_0(info_2, *((&kernelContext_35)->member_offsets_0+i_6), &kernelContext_35);
    }
    return;
}

[[kernel]] void spawn_retire(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], uint device* control_13 [[buffer(15)]], packed_uint4 device* parents_13 [[buffer(1)]], packed_uint4 device* islands_of_13 [[buffer(12)]], uint device* work_13 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_13 [[buffer(7)]], uint device* structure_base_13 [[buffer(14)]], packed_uint4 device* part_ranges_13 [[buffer(5)]], uint device* member_offsets_13 [[buffer(16)]], SpawnParams_0 constant* params_13 [[buffer(0)]], packed_float4 device* children_13 [[buffer(3)]], packed_uint4 device* slots_13 [[buffer(9)]], packed_float4 device* parts_13 [[buffer(6)]], packed_float4 device* frames_13 [[buffer(4)]], packed_float4 device* state_13 [[buffer(8)]], packed_float4 device* bodies_14 [[buffer(10)]], packed_float4 device* extras_13 [[buffer(13)]], packed_float4 device* colliders_13 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_36;
    (&kernelContext_36)->control_0 = control_13;
    (&kernelContext_36)->parents_0 = parents_13;
    (&kernelContext_36)->islands_of_0 = islands_of_13;
    (&kernelContext_36)->work_0 = work_13;
    (&kernelContext_36)->chunks_0 = chunks_13;
    (&kernelContext_36)->structure_base_0 = structure_base_13;
    (&kernelContext_36)->part_ranges_0 = part_ranges_13;
    (&kernelContext_36)->member_offsets_0 = member_offsets_13;
    (&kernelContext_36)->params_0 = params_13;
    (&kernelContext_36)->children_0 = children_13;
    (&kernelContext_36)->slots_0 = slots_13;
    (&kernelContext_36)->parts_0 = parts_13;
    (&kernelContext_36)->frames_0 = frames_13;
    (&kernelContext_36)->state_0 = state_13;
    (&kernelContext_36)->bodies_0 = bodies_14;
    (&kernelContext_36)->extras_0 = extras_13;
    (&kernelContext_36)->colliders_0 = colliders_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_36)->g_scan_0 = &g_scan_13;
    threadgroup uint g_split_parents_13;
    (&kernelContext_36)->g_split_parents_0 = &g_split_parents_13;
    threadgroup array<float4, int(64)> g_s0_13;
    (&kernelContext_36)->g_s0_0 = &g_s0_13;
    threadgroup array<float4, int(64)> g_s1_13;
    (&kernelContext_36)->g_s1_0 = &g_s1_13;
    threadgroup array<float4, int(64)> g_s2_13;
    (&kernelContext_36)->g_s2_0 = &g_s2_13;
    threadgroup array<float4, int(64)> g_s3_13;
    (&kernelContext_36)->g_s3_0 = &g_s3_13;
    threadgroup array<float4, int(64)> g_s4_13;
    (&kernelContext_36)->g_s4_0 = &g_s4_13;
    uint p_13 = group_4.x;
    uint tid_14 = thread_6.x;
    bool _S227;
    if(p_13 >= (*(control_13+1U)))
    {
        _S227 = true;
    }
    else
    {
        _S227 = ((&kernelContext_36)->params_0->live_0) == 0U;
    }
    if(_S227)
    {
        _S227 = true;
    }
    else
    {
        _S227 = ((uint4(*((&kernelContext_36)->parents_0+p_13)) ).z) < 2U;
    }
    if(_S227)
    {
        _S227 = true;
    }
    else
    {
        _S227 = (*((&kernelContext_36)->control_0+int(0))) != 0U;
    }
    if(_S227)
    {
        return;
    }
    uint4 _S228 = uint4(*((&kernelContext_36)->islands_of_0+p_13)) ;
    uint c_7;
    if(tid_14 == 0U)
    {
        uint _S229 = _S228.y;
        uint _S230 = 7U * _S229;
        thread array<float4, int(7)> record_0;
        c_7 = 0U;
        for(;;)
        {
            if(c_7 < 7U)
            {
            }
            else
            {
                break;
            }
            record_0[c_7] = float4(*((&kernelContext_36)->bodies_0+(_S230 + c_7))) ;
            c_7 = c_7 + 1U;
        }
        record_0[int(0)].w = 0.0f;
        record_0[int(2)] = float4(0.0f, 0.0f, 0.0f, (as_type<float>((0U))));
        float3 _S231 = float3(0.0f) ;
        record_0[int(3)].xyz = _S231;
        record_0[int(4)].xyz = _S231;
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
            c_7 = 0U;
            for(;;)
            {
                if(c_7 < 7U)
                {
                }
                else
                {
                    break;
                }
                *((&kernelContext_36)->bodies_0+(7U * (_S229 + region_0 * (&kernelContext_36)->params_0->max_bodies_0) + c_7)) = packed_float4(record_0[c_7]) ;
                c_7 = c_7 + 1U;
            }
            region_0 = region_0 + 1U;
        }
    }
    c_7 = _S228.z + tid_14;
    for(;;)
    {
        if(c_7 < (_S228.w))
        {
        }
        else
        {
            break;
        }
        uint _S232 = 4U * c_7;
        float4 _S233 = float4(*((&kernelContext_36)->colliders_0+_S232)) ;
        *((&kernelContext_36)->colliders_0+_S232) = packed_float4(float4(_S233.x, _S233.y, (as_type<float>(((as_type<uint>((_S233.z))) & 65535U))), (as_type<float>((32768U))))) ;
        c_7 = c_7 + 64U;
    }
    return;
}

