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
    packed_float4 device* children_0;
    uint device* work_0;
    ChunkStatic_natural_0 device* chunks_0;
    uint device* structure_base_0;
    packed_uint4 device* part_ranges_0;
    SpawnParams_0 constant* params_0;
    packed_uint4 device* slots_0;
    packed_float4 device* parts_0;
    packed_float4 device* frames_0;
    packed_float4 device* state_0;
    packed_float4 device* bodies_0;
    packed_float4 device* extras_0;
    packed_float4 device* colliders_0;
};

uint total_slots_0(KernelContext_0 thread* kernelContext_0)
{
    return (uint4(*(kernelContext_0->parents_0+*(kernelContext_0->control_0+1U))) ).y;
}

uint2 child_members_0(uint slot_0, KernelContext_0 thread* kernelContext_1)
{
    float4 _S1 = float4(*(kernelContext_1->children_0+8U * slot_0)) ;
    return uint2((as_type<uint>((_S1.y))), (as_type<uint>((_S1.z))));
}

uint4 part_range_0(uint layout_chunk_0, KernelContext_0 thread* kernelContext_2)
{
    uint4 _S2 = uint4((kernelContext_2->chunks_0+layout_chunk_0)->info_0) ;
    return uint4(*(kernelContext_2->part_ranges_0+(kernelContext_2->structure_base_0[_S2.z] + _S2.w))) ;
}

[[kernel]] void spawn_offsets(uint3 id_0 [[thread_position_in_grid]], uint device* control_1 [[buffer(15)]], packed_uint4 device* parents_1 [[buffer(1)]], packed_uint4 device* islands_of_1 [[buffer(12)]], packed_float4 device* children_1 [[buffer(3)]], uint device* work_1 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(7)]], uint device* structure_base_1 [[buffer(14)]], packed_uint4 device* part_ranges_1 [[buffer(5)]], SpawnParams_0 constant* params_1 [[buffer(0)]], packed_uint4 device* slots_1 [[buffer(9)]], packed_float4 device* parts_1 [[buffer(6)]], packed_float4 device* frames_1 [[buffer(4)]], packed_float4 device* state_1 [[buffer(8)]], packed_float4 device* bodies_1 [[buffer(10)]], packed_float4 device* extras_1 [[buffer(13)]], packed_float4 device* colliders_1 [[buffer(11)]])
{
    uint colliders_needed_0;
    uint k_0;
    uint j_0;
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->control_0 = control_1;
    (&kernelContext_3)->parents_0 = parents_1;
    (&kernelContext_3)->islands_of_0 = islands_of_1;
    (&kernelContext_3)->children_0 = children_1;
    (&kernelContext_3)->work_0 = work_1;
    (&kernelContext_3)->chunks_0 = chunks_1;
    (&kernelContext_3)->structure_base_0 = structure_base_1;
    (&kernelContext_3)->part_ranges_0 = part_ranges_1;
    (&kernelContext_3)->params_0 = params_1;
    (&kernelContext_3)->slots_0 = slots_1;
    (&kernelContext_3)->parts_0 = parts_1;
    (&kernelContext_3)->frames_0 = frames_1;
    (&kernelContext_3)->state_0 = state_1;
    (&kernelContext_3)->bodies_0 = bodies_1;
    (&kernelContext_3)->extras_0 = extras_1;
    (&kernelContext_3)->colliders_0 = colliders_1;
    uint body_0 = *(control_1+4U);
    uint collider_0 = *(control_1+5U);
    uint _S3 = total_slots_0(&kernelContext_3);
    uint p_0 = 0U;
    uint islands_needed_0 = 0U;
    uint bodies_needed_0 = 0U;
    uint colliders_needed_1 = 0U;
    for(;;)
    {
        if(p_0 < (*((&kernelContext_3)->control_0+1U)))
        {
        }
        else
        {
            break;
        }
        uint4 _S4 = uint4(*((&kernelContext_3)->parents_0+p_0)) ;
        uint _S5 = _S4.z;
        if(_S5 < 2U)
        {
            colliders_needed_0 = colliders_needed_1;
            p_0 = p_0 + 1U;
            colliders_needed_1 = colliders_needed_0;
            continue;
        }
        uint islands_needed_1 = islands_needed_0 + (_S5 - 1U);
        uint bodies_needed_1 = bodies_needed_0 + _S5;
        uint _S6 = (uint4(*((&kernelContext_3)->islands_of_0+p_0)) ).x;
        k_0 = 0U;
        colliders_needed_0 = colliders_needed_1;
        for(;;)
        {
            if(k_0 < _S5)
            {
            }
            else
            {
                break;
            }
            uint _S7 = _S4.y;
            uint2 _S8 = child_members_0(_S7 + k_0, &kernelContext_3);
            j_0 = 0U;
            for(;;)
            {
                if(j_0 < (_S8.y))
                {
                }
                else
                {
                    break;
                }
                uint4 _S9 = part_range_0(_S6 + (&kernelContext_3)->work_0[_S3 + _S7 + _S8.x + j_0], &kernelContext_3);
                uint colliders_needed_2 = colliders_needed_0 + _S9.y;
                j_0 = j_0 + 1U;
                colliders_needed_0 = colliders_needed_2;
            }
            k_0 = k_0 + 1U;
        }
        islands_needed_0 = islands_needed_1;
        bodies_needed_0 = bodies_needed_1;
        p_0 = p_0 + 1U;
        colliders_needed_1 = colliders_needed_0;
    }
    bool fits_0;
    if(islands_needed_0 <= ((&kernelContext_3)->params_0->island_cap_0 - min((&kernelContext_3)->params_0->island_cap_0, *((&kernelContext_3)->control_0+2U))))
    {
        if(((&kernelContext_3)->params_0->live_0) == 0U)
        {
            fits_0 = true;
        }
        else
        {
            if((body_0 + bodies_needed_0) <= ((&kernelContext_3)->params_0->max_bodies_0))
            {
                fits_0 = (collider_0 + colliders_needed_1) <= ((&kernelContext_3)->params_0->max_colliders_0);
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
    uint device* _S10 = (&kernelContext_3)->control_0+0U;
    if(fits_0)
    {
        p_0 = 0U;
    }
    else
    {
        p_0 = 1U;
    }
    *_S10 = p_0;
    *((&kernelContext_3)->control_0+8U) = body_0;
    *((&kernelContext_3)->control_0+9U) = collider_0;
    if(!fits_0)
    {
        return;
    }
    p_0 = 0U;
    colliders_needed_0 = body_0;
    uint collider_1 = collider_0;
    for(;;)
    {
        if(p_0 < (*((&kernelContext_3)->control_0+1U)))
        {
        }
        else
        {
            break;
        }
        uint4 _S11 = uint4(*((&kernelContext_3)->parents_0+p_0)) ;
        uint _S12 = (uint4(*((&kernelContext_3)->islands_of_0+p_0)) ).x;
        uint _S13 = _S11.z;
        if(_S13 < 2U)
        {
            p_0 = p_0 + 1U;
            continue;
        }
        k_0 = 0U;
        uint body_1 = colliders_needed_0;
        uint collider_2 = collider_1;
        for(;;)
        {
            if(k_0 < _S13)
            {
            }
            else
            {
                break;
            }
            uint _S14 = _S11.y;
            uint slot_1 = _S14 + k_0;
            uint2 _S15 = child_members_0(slot_1, &kernelContext_3);
            j_0 = 0U;
            uint count_0 = 0U;
            for(;;)
            {
                if(j_0 < (_S15.y))
                {
                }
                else
                {
                    break;
                }
                uint4 _S16 = part_range_0(_S12 + (&kernelContext_3)->work_0[_S3 + _S14 + _S15.x + j_0], &kernelContext_3);
                uint count_1 = count_0 + _S16.y;
                j_0 = j_0 + 1U;
                count_0 = count_1;
            }
            *((&kernelContext_3)->slots_0+slot_1) = packed_uint4(uint4(body_1, collider_2, count_0, 1U)) ;
            uint _S17 = body_1 + 1U;
            uint collider_3 = collider_2 + count_0;
            k_0 = k_0 + 1U;
            body_1 = _S17;
            collider_2 = collider_3;
        }
        colliders_needed_0 = body_1;
        collider_1 = collider_2;
        p_0 = p_0 + 1U;
    }
    *((&kernelContext_3)->control_0+8U) = colliders_needed_0;
    *((&kernelContext_3)->control_0+9U) = collider_1;
    return;
}

uint parent_of_0(uint slot_2, KernelContext_0 thread* kernelContext_4)
{
    uint p_1 = 0U;
    for(;;)
    {
        uint _S18 = p_1 + 1U;
        bool _S19;
        if(_S18 < (*(kernelContext_4->control_0+1U)))
        {
            _S19 = ((uint4(*(kernelContext_4->parents_0+_S18)) ).y) <= slot_2;
        }
        else
        {
            _S19 = false;
        }
        if(_S19)
        {
        }
        else
        {
            break;
        }
        p_1 = _S18;
    }
    return p_1;
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
    array<float, int(3)> _S20 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S21 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S22 = { { r2_0.x, r2_0.y, r2_0.z } };
    thread array<array<float, int(3)>, int(3)> a_0;
    a_0[int(0)] = _S20;
    a_0[int(1)] = _S21;
    a_0[int(2)] = _S22;
    array<float, int(3)> _S23 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S24 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S25 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_0;
    v_0[int(0)] = _S23;
    v_0[int(1)] = _S24;
    v_0[int(2)] = _S25;
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
        uint p_2;
        uint q_3;
        if((abs(a_0[int(1)][int(2)])) > (abs(a_0[0U][q_2])))
        {
            p_2 = 1U;
            q_3 = 2U;
        }
        else
        {
            p_2 = 0U;
            q_3 = q_2;
        }
        if((abs(a_0[p_2][q_3])) <= (9.99999982451670044e-15f * max(max(abs(a_0[int(0)][int(0)]), abs(a_0[int(1)][int(1)])), max(abs(a_0[int(2)][int(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        float theta_0 = 0.5f * atan2(2.0f * a_0[p_2][q_3], a_0[q_3][q_3] - a_0[p_2][p_2]);
        float s_0 = sin(theta_0);
        float c_0 = cos(theta_0);
        float app_0 = a_0[p_2][p_2];
        float aqq_0 = a_0[q_3][q_3];
        float apq_0 = a_0[p_2][q_3];
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
            bool _S26;
            if(k_1 != p_2)
            {
                _S26 = k_1 != q_3;
            }
            else
            {
                _S26 = false;
            }
            if(_S26)
            {
                float kp_0 = a_0[k_1][p_2];
                float kq_0 = a_0[k_1][q_3];
                float _S27 = c_0 * a_0[k_1][p_2] - s_0 * a_0[k_1][q_3];
                a_0[k_1][p_2] = _S27;
                a_0[p_2][k_1] = _S27;
                float _S28 = s_0 * kp_0 + c_0 * kq_0;
                a_0[k_1][q_3] = _S28;
                a_0[q_3][k_1] = _S28;
            }
            k_1 = k_1 + 1U;
        }
        float _S29 = c_0 * c_0;
        float _S30 = 2.0f * s_0 * c_0 * apq_0;
        float _S31 = s_0 * s_0;
        a_0[p_2][p_2] = _S29 * app_0 - _S30 + _S31 * aqq_0;
        a_0[q_3][q_3] = _S31 * app_0 + _S30 + _S29 * aqq_0;
        a_0[p_2][q_3] = 0.0f;
        a_0[q_3][p_2] = 0.0f;
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
            float kp_1 = v_0[row_0][p_2];
            float kq_1 = v_0[row_0][q_3];
            v_0[row_0][p_2] = c_0 * v_0[row_0][p_2] - s_0 * v_0[row_0][q_3];
            v_0[row_0][q_3] = s_0 * kp_1 + c_0 * kq_1;
            row_0 = row_0 + 1U;
        }
        it_0 = it_0 + 1U;
    }
    float3 x_1 = float3(v_0[int(0)][int(0)], v_0[int(1)][int(0)], v_0[int(2)][int(0)]);
    float3 y_1 = float3(v_0[int(0)][int(1)], v_0[int(1)][int(1)], v_0[int(2)][int(1)]);
    float3 z_1 = float3(v_0[int(0)][int(2)], v_0[int(1)][int(2)], v_0[int(2)][int(2)]);
    float3 z_2;
    if((dot(cross(x_1, y_1), z_1)) < 0.0f)
    {
        z_2 = - z_1;
    }
    else
    {
        z_2 = z_1;
    }
    thread Quat_0 _S32 = quat_from_columns_0(x_1, y_1, z_2);
    Quat_0 _S33 = normalized_0(&_S32);
    *rot_0 = _S33;
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

float3 rotate_0(const Quat_0 thread* q_6, float3 v_1)
{
    float3 qv_0 = float3(q_6->x_0, q_6->y_0, q_6->z_0);
    float3 t_0 = cross(qv_0, v_1) * float3(2.0f) ;
    return v_1 + t_0 * float3(q_6->w_0)  + cross(qv_0, t_0);
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
    thread Quat_0 _S34 = axes_0;
    Quat_0 _S35 = quat_mul_0(&_S34, q_7);
    thread Quat_0 _S36 = _S35;
    Quat_0 _S37 = normalized_0(&_S36);
    return _S37;
}

float3 to_rubble_0(float3 v_2)
{
    return float3(v_2.x, v_2.z, - v_2.y);
}

float4 quat_vec_0(const Quat_0 thread* q_8)
{
    return float4(q_8->x_0, q_8->y_0, q_8->z_0, q_8->w_0);
}

[[kernel]] void spawn_bodies(uint3 id_1 [[thread_position_in_grid]], uint device* control_2 [[buffer(15)]], packed_uint4 device* parents_2 [[buffer(1)]], packed_uint4 device* islands_of_2 [[buffer(12)]], packed_float4 device* children_2 [[buffer(3)]], uint device* work_2 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(7)]], uint device* structure_base_2 [[buffer(14)]], packed_uint4 device* part_ranges_2 [[buffer(5)]], SpawnParams_0 constant* params_2 [[buffer(0)]], packed_uint4 device* slots_2 [[buffer(9)]], packed_float4 device* parts_2 [[buffer(6)]], packed_float4 device* frames_2 [[buffer(4)]], packed_float4 device* state_2 [[buffer(8)]], packed_float4 device* bodies_2 [[buffer(10)]], packed_float4 device* extras_2 [[buffer(13)]], packed_float4 device* colliders_2 [[buffer(11)]])
{
    uint _S38;
    uint k_2;
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->control_0 = control_2;
    (&kernelContext_5)->parents_0 = parents_2;
    (&kernelContext_5)->islands_of_0 = islands_of_2;
    (&kernelContext_5)->children_0 = children_2;
    (&kernelContext_5)->work_0 = work_2;
    (&kernelContext_5)->chunks_0 = chunks_2;
    (&kernelContext_5)->structure_base_0 = structure_base_2;
    (&kernelContext_5)->part_ranges_0 = part_ranges_2;
    (&kernelContext_5)->params_0 = params_2;
    (&kernelContext_5)->slots_0 = slots_2;
    (&kernelContext_5)->parts_0 = parts_2;
    (&kernelContext_5)->frames_0 = frames_2;
    (&kernelContext_5)->state_0 = state_2;
    (&kernelContext_5)->bodies_0 = bodies_2;
    (&kernelContext_5)->extras_0 = extras_2;
    (&kernelContext_5)->colliders_0 = colliders_2;
    uint slot_3 = id_1.x;
    uint _S39 = total_slots_0(&kernelContext_5);
    if(slot_3 >= _S39)
    {
        return;
    }
    uint _S40 = parent_of_0(slot_3, &kernelContext_5);
    uint4 _S41 = uint4(*((&kernelContext_5)->parents_0+_S40)) ;
    uint _S42 = _S41.y;
    bool _S43;
    if((slot_3 - _S42) >= (_S41.z))
    {
        _S43 = true;
    }
    else
    {
        _S43 = ((uint4(*((&kernelContext_5)->slots_0+slot_3)) ).w) == 0U;
    }
    if(_S43)
    {
        return;
    }
    uint _S44 = (uint4(*((&kernelContext_5)->islands_of_0+_S40)) ).x;
    uint2 _S45 = child_members_0(slot_3, &kernelContext_5);
    uint _S46 = 8U * slot_3;
    bool anchored_0 = (as_type<uint>(((float4(*((&kernelContext_5)->children_0+_S46)) ).w))) != 0U;
    float3 _S47 = float3(0.0f) ;
    uint j_1 = 0U;
    float mass_0 = 0.0f;
    float3 moment_0 = _S47;
    for(;;)
    {
        uint _S48 = _S45.y;
        _S38 = _S48;
        if(j_1 < _S48)
        {
        }
        else
        {
            break;
        }
        uint4 _S49 = part_range_0(_S44 + (&kernelContext_5)->work_0[_S39 + _S42 + _S45.x + j_1], &kernelContext_5);
        uint _S50 = _S49.x;
        k_2 = _S50;
        for(;;)
        {
            if(k_2 < (_S50 + _S49.y))
            {
            }
            else
            {
                break;
            }
            float4 _S51 = float4(*((&kernelContext_5)->parts_0+8U * k_2)) ;
            float _S52 = _S51.w;
            float mass_1 = mass_0 + _S52;
            float3 moment_1 = moment_0 + _S51.xyz * float3(_S52) ;
            k_2 = k_2 + 1U;
            mass_0 = mass_1;
            moment_0 = moment_1;
        }
        j_1 = j_1 + 1U;
    }
    float3 com_0 = moment_0 / float3(max(mass_0, 1.00000000317107685e-30f)) ;
    j_1 = 0U;
    float3 i0_0 = _S47;
    float3 i1_0 = _S47;
    float3 i2_0 = _S47;
    for(;;)
    {
        if(j_1 < _S38)
        {
        }
        else
        {
            break;
        }
        uint4 _S53 = part_range_0(_S44 + (&kernelContext_5)->work_0[_S39 + _S42 + _S45.x + j_1], &kernelContext_5);
        uint _S54 = _S53.x;
        k_2 = _S54;
        for(;;)
        {
            if(k_2 < (_S54 + _S53.y))
            {
            }
            else
            {
                break;
            }
            uint _S55 = 8U * k_2;
            float4 _S56 = float4(*((&kernelContext_5)->parts_0+_S55)) ;
            float3 d_0 = _S56.xyz - com_0;
            float dd_0 = dot(d_0, d_0);
            float3 _S57 = float3(_S56.w) ;
            float3 i0_1 = i0_0 + ((float4(*((&kernelContext_5)->parts_0+(_S55 + 1U))) ).xyz + (float3(dd_0, 0.0f, 0.0f) - float3(d_0.x)  * d_0) * _S57);
            float3 i1_1 = i1_0 + ((float4(*((&kernelContext_5)->parts_0+(_S55 + 2U))) ).xyz + (float3(0.0f, dd_0, 0.0f) - float3(d_0.y)  * d_0) * _S57);
            float3 i2_1 = i2_0 + ((float4(*((&kernelContext_5)->parts_0+(_S55 + 3U))) ).xyz + (float3(0.0f, 0.0f, dd_0) - float3(d_0.z)  * d_0) * _S57);
            k_2 = k_2 + 1U;
            i0_0 = i0_1;
            i1_0 = i1_1;
            i2_0 = i2_1;
        }
        j_1 = j_1 + 1U;
    }
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(i0_0, i1_0, i2_0, &principal_0, &moments_1);
    thread Quat_0 _S58 = principal_0;
    Quat_0 _S59 = conj_0(&_S58);
    uint _S60 = 5U * _S40;
    Quat_0 pose_rot_0 = quat_of_0(float4(*((&kernelContext_5)->frames_0+_S60)) );
    float3 pose_pos_0 = (float4(*((&kernelContext_5)->frames_0+(_S60 + 4U))) ).xyz;
    bool _S61 = !anchored_0;
    float inv_mass_0;
    float3 pose_pos_1;
    if(_S61)
    {
        j_1 = 0U;
        inv_mass_0 = 0.0f;
        pose_pos_1 = _S47;
        for(;;)
        {
            if(j_1 < _S38)
            {
            }
            else
            {
                break;
            }
            uint c_2 = _S44 + (&kernelContext_5)->work_0[_S39 + _S42 + _S45.x + j_1];
            float cm_0 = (float4(((&kernelContext_5)->chunks_0+c_2)->center_0) ).w;
            float sm_0 = inv_mass_0 + cm_0;
            float3 su_0 = pose_pos_1 + (float4(*((&kernelContext_5)->state_0+4U * c_2)) ).xyz * float3(cm_0) ;
            j_1 = j_1 + 1U;
            inv_mass_0 = sm_0;
            pose_pos_1 = su_0;
        }
        float3 _S62 = pose_pos_1 / float3(inv_mass_0) ;
        thread Quat_0 _S63 = pose_rot_0;
        float3 _S64 = rotate_0(&_S63, _S62);
        pose_pos_1 = pose_pos_0 + _S64;
    }
    else
    {
        pose_pos_1 = pose_pos_0;
    }
    uint4 _S65 = uint4(*((&kernelContext_5)->slots_0+slot_3)) ;
    thread Quat_0 _S66 = pose_rot_0;
    thread Quat_0 _S67 = principal_0;
    Quat_0 _S68 = quat_mul_0(&_S66, &_S67);
    thread Quat_0 _S69 = _S68;
    Quat_0 _S70 = normalized_0(&_S69);
    thread Quat_0 _S71 = _S70;
    Quat_0 _S72 = rot_to_rubble_0(&_S71);
    if(anchored_0)
    {
        _S43 = true;
    }
    else
    {
        _S43 = !(mass_0 > 0.0f);
    }
    if(_S43)
    {
        inv_mass_0 = 0.0f;
    }
    else
    {
        inv_mass_0 = 1.0f / mass_0;
    }
    float3 inv_inertia_0;
    if(_S61)
    {
        float _S73;
        if((moments_1.x) > 0.0f)
        {
            _S73 = 1.0f / moments_1.x;
        }
        else
        {
            _S73 = 0.0f;
        }
        float _S74;
        if((moments_1.y) > 0.0f)
        {
            _S74 = 1.0f / moments_1.y;
        }
        else
        {
            _S74 = 0.0f;
        }
        float _S75;
        if((moments_1.z) > 0.0f)
        {
            _S75 = 1.0f / moments_1.z;
        }
        else
        {
            _S75 = 0.0f;
        }
        inv_inertia_0 = float3(_S73, _S74, _S75);
    }
    else
    {
        inv_inertia_0 = _S47;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S47;
    }
    else
    {
        velocity_0 = to_rubble_0((float4(*((&kernelContext_5)->children_0+(_S46 + 5U))) ).xyz);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S47;
    }
    else
    {
        spin_0 = to_rubble_0((float4(*((&kernelContext_5)->children_0+(_S46 + 6U))) ).xyz);
    }
    uint flags_0;
    if(anchored_0)
    {
        flags_0 = 0U;
    }
    else
    {
        flags_0 = 9U;
    }
    thread Quat_0 _S76 = pose_rot_0;
    float3 _S77 = rotate_0(&_S76, com_0);
    float4 _S78 = float4(to_rubble_0(pose_pos_1 + _S77), inv_mass_0);
    thread Quat_0 _S79 = _S72;
    float4 _S80 = quat_vec_0(&_S79);
    array<float4, int(7)> _S81 = { { _S78, _S80, float4(inv_inertia_0, (as_type<float>((flags_0)))), float4(velocity_0, 0.0f), float4(spin_0, 0.0f), float4(0.0f) , float4(0.0f, 0.0f, 0.0f, 1.0f) } };
    if(((&kernelContext_5)->params_0->live_0) != 0U)
    {
        j_1 = _S65.x;
    }
    else
    {
        j_1 = _S65.x - *((&kernelContext_5)->control_0+4U);
    }
    uint _S82 = 7U * j_1;
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
        *((&kernelContext_5)->bodies_0+(_S82 + w_1)) = packed_float4(_S81[w_1]) ;
        w_1 = w_1 + 1U;
    }
    if(((&kernelContext_5)->params_0->live_0) != 0U)
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
            *((&kernelContext_5)->bodies_0+(7U * ((&kernelContext_5)->params_0->max_bodies_0 + _S65.x) + w_1)) = packed_float4(_S81[w_1]) ;
            w_1 = w_1 + 1U;
        }
    }
    uint _S83 = 2U * slot_3;
    *((&kernelContext_5)->extras_0+_S83) = packed_float4(float4(com_0, mass_0)) ;
    packed_float4 device* _S84 = (&kernelContext_5)->extras_0+(_S83 + 1U);
    thread Quat_0 _S85 = principal_0;
    float4 _S86 = quat_vec_0(&_S85);
    *_S84 = packed_float4(_S86) ;
    uint _S87 = _S65.y;
    j_1 = 0U;
    uint col_0 = _S87;
    for(;;)
    {
        if(j_1 < _S38)
        {
        }
        else
        {
            break;
        }
        uint4 _S88 = part_range_0(_S44 + (&kernelContext_5)->work_0[_S39 + _S42 + _S45.x + j_1], &kernelContext_5);
        uint _S89 = _S88.x;
        k_2 = _S89;
        uint col_1 = col_0;
        for(;;)
        {
            if(k_2 < (_S89 + _S88.y))
            {
            }
            else
            {
                break;
            }
            uint _S90 = 8U * k_2;
            float4 _S91 = float4(*((&kernelContext_5)->parts_0+(_S90 + 4U))) ;
            Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_5)->parts_0+(_S90 + 5U))) );
            float4 _S92 = float4(*((&kernelContext_5)->parts_0+(_S90 + 6U))) ;
            float4 _S93 = float4(*((&kernelContext_5)->parts_0+(_S90 + 7U))) ;
            if(((&kernelContext_5)->params_0->live_0) != 0U)
            {
                flags_0 = col_1;
            }
            else
            {
                flags_0 = col_1 - *((&kernelContext_5)->control_0+5U);
            }
            uint cw_0 = 4U * flags_0;
            *((&kernelContext_5)->colliders_0+cw_0) = packed_float4(float4((as_type<float>((_S65.x))), _S92.x, _S92.y, _S92.z)) ;
            packed_float4 device* _S94 = (&kernelContext_5)->colliders_0+(cw_0 + 1U);
            float3 _S95 = _S91.xyz - com_0;
            thread Quat_0 _S96 = _S59;
            float3 _S97 = rotate_0(&_S96, _S95);
            *_S94 = packed_float4(float4(_S97, _S91.w)) ;
            packed_float4 device* _S98 = (&kernelContext_5)->colliders_0+(cw_0 + 2U);
            thread Quat_0 _S99 = _S59;
            thread Quat_0 _S100 = rot_1;
            Quat_0 _S101 = quat_mul_0(&_S99, &_S100);
            thread Quat_0 _S102 = _S101;
            Quat_0 _S103 = normalized_0(&_S102);
            thread Quat_0 _S104 = _S103;
            float4 _S105 = quat_vec_0(&_S104);
            *_S98 = packed_float4(_S105) ;
            *((&kernelContext_5)->colliders_0+(cw_0 + 3U)) = packed_float4(_S93) ;
            uint _S106 = col_1 + 1U;
            k_2 = k_2 + 1U;
            col_1 = _S106;
        }
        j_1 = j_1 + 1U;
        col_0 = col_1;
    }
    return;
}

[[kernel]] void spawn_retire(uint3 id_2 [[thread_position_in_grid]], uint device* control_3 [[buffer(15)]], packed_uint4 device* parents_3 [[buffer(1)]], packed_uint4 device* islands_of_3 [[buffer(12)]], packed_float4 device* children_3 [[buffer(3)]], uint device* work_3 [[buffer(2)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(7)]], uint device* structure_base_3 [[buffer(14)]], packed_uint4 device* part_ranges_3 [[buffer(5)]], SpawnParams_0 constant* params_3 [[buffer(0)]], packed_uint4 device* slots_3 [[buffer(9)]], packed_float4 device* parts_3 [[buffer(6)]], packed_float4 device* frames_3 [[buffer(4)]], packed_float4 device* state_3 [[buffer(8)]], packed_float4 device* bodies_3 [[buffer(10)]], packed_float4 device* extras_3 [[buffer(13)]], packed_float4 device* colliders_3 [[buffer(11)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->control_0 = control_3;
    (&kernelContext_6)->parents_0 = parents_3;
    (&kernelContext_6)->islands_of_0 = islands_of_3;
    (&kernelContext_6)->children_0 = children_3;
    (&kernelContext_6)->work_0 = work_3;
    (&kernelContext_6)->chunks_0 = chunks_3;
    (&kernelContext_6)->structure_base_0 = structure_base_3;
    (&kernelContext_6)->part_ranges_0 = part_ranges_3;
    (&kernelContext_6)->params_0 = params_3;
    (&kernelContext_6)->slots_0 = slots_3;
    (&kernelContext_6)->parts_0 = parts_3;
    (&kernelContext_6)->frames_0 = frames_3;
    (&kernelContext_6)->state_0 = state_3;
    (&kernelContext_6)->bodies_0 = bodies_3;
    (&kernelContext_6)->extras_0 = extras_3;
    (&kernelContext_6)->colliders_0 = colliders_3;
    uint p_3 = id_2.x;
    bool _S107;
    if(p_3 >= (*(control_3+1U)))
    {
        _S107 = true;
    }
    else
    {
        _S107 = ((&kernelContext_6)->params_0->live_0) == 0U;
    }
    if(_S107)
    {
        _S107 = true;
    }
    else
    {
        _S107 = ((uint4(*((&kernelContext_6)->parents_0+p_3)) ).z) < 2U;
    }
    if(_S107)
    {
        _S107 = true;
    }
    else
    {
        _S107 = (*((&kernelContext_6)->control_0+int(0))) != 0U;
    }
    if(_S107)
    {
        return;
    }
    uint4 _S108 = uint4(*((&kernelContext_6)->islands_of_0+p_3)) ;
    uint _S109 = _S108.y;
    uint _S110 = 7U * _S109;
    thread array<float4, int(7)> record_0;
    uint w_2 = 0U;
    for(;;)
    {
        if(w_2 < 7U)
        {
        }
        else
        {
            break;
        }
        record_0[w_2] = float4(*((&kernelContext_6)->bodies_0+(_S110 + w_2))) ;
        w_2 = w_2 + 1U;
    }
    record_0[int(0)].w = 0.0f;
    record_0[int(2)] = float4(0.0f, 0.0f, 0.0f, (as_type<float>((0U))));
    float3 _S111 = float3(0.0f) ;
    record_0[int(3)].xyz = _S111;
    record_0[int(4)].xyz = _S111;
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
        w_2 = 0U;
        for(;;)
        {
            if(w_2 < 7U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_6)->bodies_0+(7U * (_S109 + region_0 * (&kernelContext_6)->params_0->max_bodies_0) + w_2)) = packed_float4(record_0[w_2]) ;
            w_2 = w_2 + 1U;
        }
        region_0 = region_0 + 1U;
    }
    uint c_3 = _S108.z;
    for(;;)
    {
        if(c_3 < (_S108.w))
        {
        }
        else
        {
            break;
        }
        uint _S112 = 4U * c_3;
        float4 _S113 = float4(*((&kernelContext_6)->colliders_0+_S112)) ;
        *((&kernelContext_6)->colliders_0+_S112) = packed_float4(float4(_S113.x, _S113.y, (as_type<float>(((as_type<uint>((_S113.z))) & 65535U))), (as_type<float>((32768U))))) ;
        c_3 = c_3 + 1U;
    }
    return;
}

