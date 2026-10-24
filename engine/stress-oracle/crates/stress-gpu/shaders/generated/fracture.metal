#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint w_body_0()
{
    return 0U;
}

struct FractureParams_0
{
    uint chunk_count_0;
    uint bond_count_0;
    uint body_count_0;
    uint max_bodies_0;
    uint step_0;
    uint pool_cap_0;
    uint impact_peak_0;
    uint parity_0;
    float dt_0;
    float band_0;
    float scale_m_0;
    float scale_s_0;
    float scale_j_0;
    float strike_speed_0;
    uint shot_count_0;
    float pad3_0;
};

struct Record_natural_0
{
    packed_uint4 ids_0;
    packed_float4 point_0;
    packed_float4 impulse_0;
    packed_float4 other_0;
    packed_uint4 key_0;
};

struct ScanParams_0
{
    uint n_0;
    uint in_base_0;
    uint out_base_0;
    uint block_base_0;
};

struct ViewParams_0
{
    uint mode_0;
    uint shot_count_1;
    uint impactor_base_0;
    uint pad0_0;
    float rest_speed_0;
    float pad1_0;
    float pad2_0;
    float pad3_1;
};

struct KernelContext_0
{
    FractureParams_0 constant* params_0;
    uint device* work_0;
    packed_float4 device* frames_0;
    packed_float4 device* bodies_0;
    packed_float4 device* chunks_0;
    uint device* sort_0;
    atomic<uint> device* atom_0;
    Record_natural_0 device* manifolds_0;
    packed_float4 device* loads_0;
    packed_uint4 device* bonds_0;
    uint device* args_0;
    uint device* info_0;
    ScanParams_0 constant* scan_0;
    packed_float4 device* parts_0;
    packed_float4 device* colliders_0;
    packed_uint4 device* collider_info_0;
    packed_float4 device* view_chunks_0;
    ViewParams_0 constant* view_0;
    packed_float4 device* objects_0;
    uint device* shot_bodies_0;
    float device* lines_0;
    packed_float4 device* geometry_0;
    packed_float4 device* shots_0;
    array<uint, int(1024)> threadgroup* g_scan_0;
};

uint body_of_0(uint c_0, KernelContext_0 thread* kernelContext_0)
{
    return *(kernelContext_0->work_0+(w_body_0() + c_0));
}

struct Quat_0
{
    float w_0;
    float x_0;
    float y_0;
    float z_0;
};

Quat_0 quat_of_0(float4 q_0)
{
    thread Quat_0 r_0;
    (&r_0)->x_0 = q_0.x;
    (&r_0)->y_0 = q_0.y;
    (&r_0)->z_0 = q_0.z;
    (&r_0)->w_0 = q_0.w;
    return r_0;
}

Quat_0 conj_0(const Quat_0 thread* q_1)
{
    thread Quat_0 c_1;
    (&c_1)->w_0 = q_1->w_0;
    (&c_1)->x_0 = - q_1->x_0;
    (&c_1)->y_0 = - q_1->y_0;
    (&c_1)->z_0 = - q_1->z_0;
    return c_1;
}

Quat_0 quat_mul_0(const Quat_0 thread* a_0, const Quat_0 thread* o_0)
{
    thread Quat_0 r_1;
    (&r_1)->w_0 = a_0->w_0 * o_0->w_0 - a_0->x_0 * o_0->x_0 - a_0->y_0 * o_0->y_0 - a_0->z_0 * o_0->z_0;
    (&r_1)->x_0 = a_0->w_0 * o_0->x_0 + a_0->x_0 * o_0->w_0 + a_0->y_0 * o_0->z_0 - a_0->z_0 * o_0->y_0;
    (&r_1)->y_0 = a_0->w_0 * o_0->y_0 - a_0->x_0 * o_0->z_0 + a_0->y_0 * o_0->w_0 + a_0->z_0 * o_0->x_0;
    (&r_1)->z_0 = a_0->w_0 * o_0->z_0 + a_0->x_0 * o_0->y_0 - a_0->y_0 * o_0->x_0 + a_0->z_0 * o_0->w_0;
    return r_1;
}

Quat_0 normalized_0(const Quat_0 thread* q_2)
{
    float n_1 = sqrt(q_2->w_0 * q_2->w_0 + q_2->x_0 * q_2->x_0 + q_2->y_0 * q_2->y_0 + q_2->z_0 * q_2->z_0);
    thread Quat_0 r_2;
    (&r_2)->w_0 = q_2->w_0 / n_1;
    (&r_2)->x_0 = q_2->x_0 / n_1;
    (&r_2)->y_0 = q_2->y_0 / n_1;
    (&r_2)->z_0 = q_2->z_0 / n_1;
    return r_2;
}

Quat_0 rot_from_rubble_0(const Quat_0 thread* q_3)
{
    thread Quat_0 axes_0;
    (&axes_0)->w_0 = 0.70710676908493042f;
    (&axes_0)->x_0 = 0.70710676908493042f;
    (&axes_0)->y_0 = 0.0f;
    (&axes_0)->z_0 = 0.0f;
    thread Quat_0 _S1 = axes_0;
    Quat_0 _S2 = quat_mul_0(&_S1, q_3);
    thread Quat_0 _S3 = _S2;
    Quat_0 _S4 = normalized_0(&_S3);
    return _S4;
}

float3 from_rubble_0(float3 v_0)
{
    return float3(v_0.x, - v_0.z, v_0.y);
}

float3 rotate_0(const Quat_0 thread* q_4, float3 v_1)
{
    float3 qv_0 = float3(q_4->x_0, q_4->y_0, q_4->z_0);
    float3 t_0 = cross(qv_0, v_1) * float3(2.0f) ;
    return v_1 + t_0 * float3(q_4->w_0)  + cross(qv_0, t_0);
}

float3 rotate_1(const Quat_0 thread* q_5, float3 v_2)
{
    float3 qv_1 = float3(q_5->x_0, q_5->y_0, q_5->z_0);
    float3 t_1 = cross(qv_1, v_2) * float3(2.0f) ;
    return v_2 + t_1 * float3(q_5->w_0)  + cross(qv_1, t_1);
}

void structure_pose_0(uint slot_0, uint b_0, Quat_0 thread* q_6, float3 thread* origin_0, KernelContext_0 thread* kernelContext_1)
{
    uint _S5 = 2U * slot_0;
    thread Quat_0 _S6 = quat_of_0(float4(*(kernelContext_1->frames_0+(_S5 + 1U))) );
    Quat_0 _S7 = conj_0(&_S6);
    float3 com_0 = (float4(*(kernelContext_1->frames_0+_S5)) ).xyz;
    thread Quat_0 _S8 = quat_of_0(float4(*(kernelContext_1->bodies_0+(b_0 + 1U))) );
    Quat_0 _S9 = rot_from_rubble_0(&_S8);
    thread Quat_0 _S10 = _S9;
    thread Quat_0 _S11 = _S7;
    Quat_0 _S12 = quat_mul_0(&_S10, &_S11);
    thread Quat_0 _S13 = _S12;
    Quat_0 _S14 = normalized_0(&_S13);
    *q_6 = _S14;
    float3 _S15 = from_rubble_0((float4(*(kernelContext_1->bodies_0+b_0)) ).xyz);
    thread Quat_0 _S16 = _S14;
    float3 _S17 = rotate_0(&_S16, com_0);
    *origin_0 = _S15 - _S17;
    return;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

uint w_alive_0(KernelContext_0 thread* kernelContext_2)
{
    return kernelContext_2->params_0->chunk_count_0;
}

uint w_label_0(KernelContext_0 thread* kernelContext_3)
{
    uint _S18 = w_alive_0(kernelContext_3);
    return _S18 + kernelContext_3->params_0->bond_count_0;
}

uint w_slot_0(KernelContext_0 thread* kernelContext_4)
{
    uint _S19 = w_label_0(kernelContext_4);
    return _S19 + kernelContext_4->params_0->chunk_count_0;
}

uint w_new_0(KernelContext_0 thread* kernelContext_5)
{
    uint _S20 = w_slot_0(kernelContext_5);
    return _S20 + kernelContext_5->params_0->chunk_count_0;
}

uint w_rank_0(KernelContext_0 thread* kernelContext_6)
{
    uint _S21 = w_new_0(kernelContext_6);
    return _S21 + kernelContext_6->params_0->chunk_count_0;
}

uint w_free_0(KernelContext_0 thread* kernelContext_7)
{
    uint _S22 = w_rank_0(kernelContext_7);
    return _S22 + kernelContext_7->params_0->chunk_count_0;
}

uint w_free_rank_0(KernelContext_0 thread* kernelContext_8)
{
    uint _S23 = w_free_0(kernelContext_8);
    return _S23 + kernelContext_8->params_0->body_count_0;
}

uint w_free_list_0(KernelContext_0 thread* kernelContext_9)
{
    uint _S24 = w_free_rank_0(kernelContext_9);
    return _S24 + kernelContext_9->params_0->body_count_0;
}

uint w_pulses_0(KernelContext_0 thread* kernelContext_10)
{
    uint _S25 = w_free_list_0(kernelContext_10);
    return _S25 + kernelContext_10->params_0->body_count_0 + 2050U;
}

uint a_sums_0(KernelContext_0 thread* kernelContext_11)
{
    return 3U * kernelContext_11->params_0->chunk_count_0;
}

uint a_dirty_0(KernelContext_0 thread* kernelContext_12)
{
    uint _S26 = a_sums_0(kernelContext_12);
    return _S26 + 20U * kernelContext_12->params_0->chunk_count_0;
}

uint a_best_count_0(KernelContext_0 thread* kernelContext_13)
{
    uint _S27 = a_dirty_0(kernelContext_13);
    return _S27 + kernelContext_13->params_0->body_count_0;
}

uint a_best_root_0(KernelContext_0 thread* kernelContext_14)
{
    uint _S28 = a_best_count_0(kernelContext_14);
    return _S28 + kernelContext_14->params_0->body_count_0;
}

uint a_counters_0(KernelContext_0 thread* kernelContext_15)
{
    uint _S29 = a_best_root_0(kernelContext_15);
    return _S29 + kernelContext_15->params_0->body_count_0;
}

uint a_counters_1(KernelContext_0 thread* kernelContext_16)
{
    uint _S30 = a_best_root_0(kernelContext_16);
    return _S30 + kernelContext_16->params_0->body_count_0;
}

uint l_pool_0(uint half_0, KernelContext_0 thread* kernelContext_17)
{
    return kernelContext_17->params_0->chunk_count_0 + 3U * kernelContext_17->params_0->pool_cap_0 * half_0;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S31 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S31, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S32 = max(0.0f, _S31);
    float _S33 = min(dt_1, _S31 + duration_0);
    bool _S34;
    if(_S33 > _S32)
    {
        _S34 = duration_0 > 0.0f;
    }
    else
    {
        _S34 = false;
    }
    float sum_1;
    if(_S34)
    {
        float w_1 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_1 * (cos(w_1 * (_S32 - _S31)) - cos(w_1 * (_S33 - _S31)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

float hertz_duration_0(float m_0, float r_3, float e_0, float v_3)
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_3 * e_0 * e_0 * max(abs(v_3), 0.00100000004749745f)), 0.20000000298023224f);
}

[[kernel]] void fr_loads(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], uint device* sort_1 [[buffer(11)]], atomic<uint> device* atom_1 [[buffer(5)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], packed_float4 device* loads_1 [[buffer(12)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]], packed_float4 device* view_chunks_1 [[buffer(17)]], ViewParams_0 constant* view_1 [[buffer(16)]], packed_float4 device* objects_1 [[buffer(18)]], uint device* shot_bodies_1 [[buffer(20)]], float device* lines_1 [[buffer(19)]], packed_float4 device* geometry_1 [[buffer(21)]], packed_float4 device* shots_1 [[buffer(22)]])
{
    float3 f_total_0;
    float m_1;
    uint p_0;
    uint kept_0;
    uint q3_0;
    bool overflow_0;
    bool overflow_1;
    float3 f_total_1;
    float3 t_total_0;
    thread KernelContext_0 kernelContext_18;
    (&kernelContext_18)->params_0 = params_1;
    (&kernelContext_18)->work_0 = work_1;
    (&kernelContext_18)->frames_0 = frames_1;
    (&kernelContext_18)->bodies_0 = bodies_1;
    (&kernelContext_18)->chunks_0 = chunks_1;
    (&kernelContext_18)->sort_0 = sort_1;
    (&kernelContext_18)->atom_0 = atom_1;
    (&kernelContext_18)->manifolds_0 = manifolds_1;
    (&kernelContext_18)->loads_0 = loads_1;
    (&kernelContext_18)->bonds_0 = bonds_1;
    (&kernelContext_18)->args_0 = args_1;
    (&kernelContext_18)->info_0 = info_1;
    (&kernelContext_18)->scan_0 = scan_1;
    (&kernelContext_18)->parts_0 = parts_1;
    (&kernelContext_18)->colliders_0 = colliders_1;
    (&kernelContext_18)->collider_info_0 = collider_info_1;
    (&kernelContext_18)->view_chunks_0 = view_chunks_1;
    (&kernelContext_18)->view_0 = view_1;
    (&kernelContext_18)->objects_0 = objects_1;
    (&kernelContext_18)->shot_bodies_0 = shot_bodies_1;
    (&kernelContext_18)->lines_0 = lines_1;
    (&kernelContext_18)->geometry_0 = geometry_1;
    (&kernelContext_18)->shots_0 = shots_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_18)->g_scan_0 = &g_scan_1;
    uint k_0 = id_0.x;
    if(k_0 >= (params_1->chunk_count_0))
    {
        return;
    }
    uint _S35 = body_of_0(k_0, &kernelContext_18);
    uint b_1 = 7U * _S35;
    thread Quat_0 q_7;
    thread float3 origin_1;
    structure_pose_0(_S35, b_1, &q_7, &origin_1, &kernelContext_18);
    uint _S36 = 4U * k_0;
    float4 _S37 = float4(*((&kernelContext_18)->chunks_0+_S36)) ;
    float3 _S38 = _S37.xyz;
    thread Quat_0 _S39 = q_7;
    float3 _S40 = rotate_0(&_S39, _S38);
    float3 centre_0 = origin_1 + _S40;
    float inv_mass_0 = (float4(*((&kernelContext_18)->bodies_0+b_1)) ).w;
    float m_body_0;
    if(inv_mass_0 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_0;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint from_0 = (&kernelContext_18)->params_0->parity_0;
    uint to_0 = 1U - (&kernelContext_18)->params_0->parity_0;
    uint _S41 = w_pulses_0(&kernelContext_18);
    uint _S42 = _S41 + _S36;
    uint _S43 = _S42 + 2U * from_0;
    uint off_0 = *((&kernelContext_18)->work_0+_S43);
    uint device* _S44 = (&kernelContext_18)->work_0+(_S43 + 1U);
    uint carried_0 = *_S44;
    uint room_0 = *_S44 + (&kernelContext_18)->sort_0[params_1->chunk_count_0 + 1U + k_0];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S45 = a_counters_0(&kernelContext_18);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_18)->atom_0+(_S45 + 5U + to_0), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_18)->params_0->pool_cap_0))
        {
            uint _S46 = atomic_fetch_or_explicit((&kernelContext_18)->atom_0+(_S45 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_18)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_18)->params_0->pool_cap_0 - base_1;
            }
            else
            {
                base_0 = 0U;
            }
        }
        else
        {
            base_0 = room_0;
        }
        uint _S47 = base_0;
        base_0 = base_1;
        room_1 = _S47;
    }
    else
    {
        base_0 = 0U;
        room_1 = room_0;
    }
    uint _S48 = params_1->chunk_count_0;
    float _S49 = _S37.w;
    float4 _S50 = float4(*((&kernelContext_18)->chunks_0+(_S36 + 3U))) ;
    float _S51 = (&kernelContext_18)->params_0->band_0;
    float _S52 = (&kernelContext_18)->params_0->dt_0;
    bool _S53 = ((&kernelContext_18)->params_0->impact_peak_0) != 0U;
    uint _S54 = l_pool_0(from_0, &kernelContext_18);
    uint _S55 = _S54 + 3U * off_0;
    uint _S56 = l_pool_0(to_0, &kernelContext_18);
    uint _S57 = _S56 + 3U * base_0;
    for(;;)
    {
        float3 _S58 = float3(0.0f) ;
        uint _S59 = (&kernelContext_18)->sort_0[k_0];
        uint _S60 = (&kernelContext_18)->sort_0[_S48 + 1U + k_0];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_1 = 0U;
        f_total_0 = _S58;
        float3 t_total_1 = _S58;
        for(;;)
        {
            if(step_1 < _S60)
            {
            }
            else
            {
                break;
            }
            p_0 = 4294967295U;
            kept_0 = 0U;
            q3_0 = 0U;
            for(;;)
            {
                if(q3_0 < _S60)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_18)->manifolds_0[_S59 + q3_0].ids_0) ).y + 1U;
                if(first_0)
                {
                    overflow_0 = true;
                }
                else
                {
                    overflow_0 = key_1 > last_0;
                }
                if(overflow_0)
                {
                    if(p_0 == 4294967295U)
                    {
                        overflow_1 = true;
                    }
                    else
                    {
                        overflow_1 = key_1 < kept_0;
                    }
                }
                else
                {
                    overflow_1 = false;
                }
                if(overflow_1)
                {
                    p_0 = q3_0;
                    kept_0 = key_1;
                }
                q3_0 = q3_0 + 1U;
            }
            if(p_0 == 4294967295U)
            {
                break;
            }
            Record_natural_0 r_4 = (&kernelContext_18)->manifolds_0[_S59 + p_0];
            float4 _S61 = float4(r_4.impulse_0) ;
            float3 _S62 = _S61.xyz;
            float j_0 = length(_S62);
            if(j_0 == 0.0f)
            {
                uint _S63 = step_1 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S63;
                continue;
            }
            float m_other_0 = (float4(r_4.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S49;
                }
                else
                {
                    m_1 = m_other_0;
                }
            }
            else
            {
                if(isinf(m_other_0))
                {
                    m_1 = m_body_0;
                }
                else
                {
                    m_1 = m_body_0 * m_other_0 / (m_body_0 + m_other_0);
                }
            }
            float _S64 = min(j_0, 2.0f * m_1 * max(_S61.w - _S51, 0.0f));
            if(_S64 < j_0)
            {
                float3 f_0 = _S62 / float3(j_0)  * float3(((j_0 - _S64) / _S52)) ;
                float3 _S65 = t_total_1 + cross((float4(r_4.point_0) ).xyz - centre_0, f_0);
                f_total_1 = f_total_0 + f_0;
                t_total_0 = _S65;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            uint _S63 = step_1 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S63;
        }
        p_0 = 0U;
        kept_0 = 0U;
        for(;;)
        {
            if(p_0 < carried_0)
            {
            }
            else
            {
                break;
            }
            uint _S66 = _S55 + 3U * p_0;
            float4 _S67 = float4(*((&kernelContext_18)->loads_0+_S66)) ;
            thread float4 a_1 = _S67;
            float4 _S68 = float4(*((&kernelContext_18)->loads_0+(_S66 + 1U))) ;
            float4 _S69 = float4(*((&kernelContext_18)->loads_0+(_S66 + 2U))) ;
            float _S70 = _S68.w;
            float mean_0 = pulse_mean_0(_S67.x, _S67.y, _S67.z, _S67.w, _S70, _S52);
            if(mean_0 != 0.0f)
            {
                float3 f_1 = _S69.xyz * float3(mean_0) ;
                float3 _S71 = t_total_1 + cross(_S68.xyz - centre_0, f_1);
                f_total_0 = f_total_0 + f_1;
                t_total_1 = _S71;
            }
            if((a_1.x + a_1.z + _S70) > _S52)
            {
                a_1.x = a_1.x - _S52;
                uint _S72 = _S57 + 3U * kept_0;
                *((&kernelContext_18)->loads_0+_S72) = packed_float4(a_1) ;
                *((&kernelContext_18)->loads_0+(_S72 + 1U)) = packed_float4(_S68) ;
                *((&kernelContext_18)->loads_0+(_S72 + 2U)) = packed_float4(_S69) ;
                kept_0 = kept_0 + 1U;
            }
            p_0 = p_0 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S60)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_5 = (&kernelContext_18)->manifolds_0[_S59 + q3_0];
            float4 _S73 = float4(r_5.impulse_0) ;
            float3 _S74 = _S73.xyz;
            float j_1 = length(_S74);
            if(j_1 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S75 = float4(r_5.other_0) ;
            float m_other_1 = _S75.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S49;
                }
                else
                {
                    m_1 = m_other_1;
                }
            }
            else
            {
                if(isinf(m_other_1))
                {
                    m_1 = m_body_0;
                }
                else
                {
                    m_1 = m_body_0 * m_other_1 / (m_body_0 + m_other_1);
                }
            }
            float _S76 = _S73.w;
            float _S77 = min(j_1, 2.0f * m_1 * max(_S76 - _S51, 0.0f));
            if(_S77 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S78 = _S50.y;
            float duration_1 = hertz_duration_0(m_1, min(_S75.z, _S50.z), 1.0f / ((1.0f - _S78 * _S78) / _S50.x + 1.0f / _S75.y), _S76);
            float peak_1 = 3.14159274101257324f * _S77 / (2.0f * duration_1);
            float3 dir_0 = _S74 / float3(j_1) ;
            float mean_1;
            if(_S53)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S52);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_2 = dir_0 * float3(mean_1) ;
                float3 _S79 = t_total_1 + cross((float4(r_5.point_0) ).xyz - centre_0, f_2);
                f_total_1 = f_total_0 + f_2;
                t_total_0 = _S79;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            uint kept_1;
            if(duration_1 > _S52)
            {
                if(kept_0 < room_1)
                {
                    uint _S80 = _S57 + 3U * kept_0;
                    *((&kernelContext_18)->loads_0+_S80) = packed_float4(float4(- _S52, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_18)->loads_0+(_S80 + 1U)) = packed_float4(float4((float4(r_5.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_18)->loads_0+(_S80 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
                    kept_1 = kept_0 + 1U;
                    overflow_1 = overflow_0;
                }
                else
                {
                    kept_1 = kept_0;
                    overflow_1 = true;
                }
            }
            else
            {
                kept_1 = kept_0;
                overflow_1 = overflow_0;
            }
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            kept_0 = kept_1;
            q3_0 = q3_0 + 1U;
            overflow_0 = overflow_1;
        }
        break;
    }
    if(overflow_0)
    {
        uint _S81 = a_counters_0(&kernelContext_18);
        uint _S82 = atomic_fetch_or_explicit((&kernelContext_18)->atom_0+(_S81 + 3U), 8U, memory_order_relaxed);
    }
    uint _S83 = _S42 + 2U * to_0;
    *((&kernelContext_18)->work_0+_S83) = base_0;
    *((&kernelContext_18)->work_0+(_S83 + 1U)) = kept_0;
    *((&kernelContext_18)->loads_0+k_0) = packed_float4(float4(length(f_total_0), 0.0f, 0.0f, 0.0f)) ;
    return;
}

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], atomic<uint> device* atom_2 [[buffer(5)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]], packed_float4 device* geometry_2 [[buffer(21)]], packed_float4 device* shots_2 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_19;
    (&kernelContext_19)->params_0 = params_2;
    (&kernelContext_19)->work_0 = work_2;
    (&kernelContext_19)->frames_0 = frames_2;
    (&kernelContext_19)->bodies_0 = bodies_2;
    (&kernelContext_19)->chunks_0 = chunks_2;
    (&kernelContext_19)->sort_0 = sort_2;
    (&kernelContext_19)->atom_0 = atom_2;
    (&kernelContext_19)->manifolds_0 = manifolds_2;
    (&kernelContext_19)->loads_0 = loads_2;
    (&kernelContext_19)->bonds_0 = bonds_2;
    (&kernelContext_19)->args_0 = args_2;
    (&kernelContext_19)->info_0 = info_2;
    (&kernelContext_19)->scan_0 = scan_2;
    (&kernelContext_19)->parts_0 = parts_2;
    (&kernelContext_19)->colliders_0 = colliders_2;
    (&kernelContext_19)->collider_info_0 = collider_info_2;
    (&kernelContext_19)->view_chunks_0 = view_chunks_2;
    (&kernelContext_19)->view_0 = view_2;
    (&kernelContext_19)->objects_0 = objects_2;
    (&kernelContext_19)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_19)->lines_0 = lines_2;
    (&kernelContext_19)->geometry_0 = geometry_2;
    (&kernelContext_19)->shots_0 = shots_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_19)->g_scan_0 = &g_scan_2;
    uint i_0 = id_1.x;
    bool _S84;
    if(i_0 >= (params_2->bond_count_0))
    {
        _S84 = true;
    }
    else
    {
        uint _S85 = w_alive_0(&kernelContext_19);
        _S84 = (*((&kernelContext_19)->work_0+(_S85 + i_0))) == 0U;
    }
    if(_S84)
    {
        return;
    }
    uint4 _S86 = uint4(*((&kernelContext_19)->bonds_0+i_0)) ;
    uint _S87 = _S86.x;
    if((max((float4(*((&kernelContext_19)->loads_0+_S87)) ).x, (float4(*((&kernelContext_19)->loads_0+_S86.y)) ).x)) > (as_type<float>((_S86.z))))
    {
        uint _S88 = w_alive_0(&kernelContext_19);
        *((&kernelContext_19)->work_0+(_S88 + i_0)) = 0U;
        uint _S89 = a_dirty_0(&kernelContext_19);
        uint _S90 = body_of_0(_S87, &kernelContext_19);
        uint _S91 = atomic_fetch_or_explicit((&kernelContext_19)->atom_0+(_S89 + _S90), 1U, memory_order_relaxed);
        uint _S92 = a_counters_0(&kernelContext_19);
        uint _S93 = atomic_fetch_add_explicit((&kernelContext_19)->atom_0+_S92, 1U, memory_order_relaxed);
        uint _S94 = atomic_fetch_add_explicit((&kernelContext_19)->atom_0+(_S92 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint groups_of_0(uint n_2, uint size_0)
{
    uint _S95 = (n_2 + size_0 - 1U) / size_0;
    return _S95;
}

void set_args_0(uint at_0, uint x_1, KernelContext_0 thread* kernelContext_20)
{
    uint _S96 = 3U * at_0;
    *(kernelContext_20->args_0+_S96) = x_1;
    *(kernelContext_20->args_0+(_S96 + 1U)) = 1U;
    *(kernelContext_20->args_0+(_S96 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_21;
    (&kernelContext_21)->params_0 = params_3;
    (&kernelContext_21)->work_0 = work_3;
    (&kernelContext_21)->frames_0 = frames_3;
    (&kernelContext_21)->bodies_0 = bodies_3;
    (&kernelContext_21)->chunks_0 = chunks_3;
    (&kernelContext_21)->sort_0 = sort_3;
    (&kernelContext_21)->atom_0 = atom_3;
    (&kernelContext_21)->manifolds_0 = manifolds_3;
    (&kernelContext_21)->loads_0 = loads_3;
    (&kernelContext_21)->bonds_0 = bonds_3;
    (&kernelContext_21)->args_0 = args_3;
    (&kernelContext_21)->info_0 = info_3;
    (&kernelContext_21)->scan_0 = scan_3;
    (&kernelContext_21)->parts_0 = parts_3;
    (&kernelContext_21)->colliders_0 = colliders_3;
    (&kernelContext_21)->collider_info_0 = collider_info_3;
    (&kernelContext_21)->view_chunks_0 = view_chunks_3;
    (&kernelContext_21)->view_0 = view_3;
    (&kernelContext_21)->objects_0 = objects_3;
    (&kernelContext_21)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_21)->lines_0 = lines_3;
    (&kernelContext_21)->geometry_0 = geometry_3;
    (&kernelContext_21)->shots_0 = shots_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_21)->g_scan_0 = &g_scan_3;
    uint _S97 = a_counters_0(&kernelContext_21);
    uint _S98 = atomic_load_explicit((&kernelContext_21)->atom_0+_S97, memory_order_relaxed);
    bool run_0 = _S98 != 0U;
    atomic_store_explicit((&kernelContext_21)->atom_0+_S97, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_21)->atom_0+(_S97 + 5U + (&kernelContext_21)->params_0->parity_0), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S99 = atomic_fetch_add_explicit((&kernelContext_21)->atom_0+(_S97 + 4U), 1U, memory_order_relaxed);
    }
    uint _S100;
    if(run_0)
    {
        uint _S101 = groups_of_0((&kernelContext_21)->params_0->chunk_count_0, 64U);
        _S100 = _S101;
    }
    else
    {
        _S100 = 0U;
    }
    set_args_0(0U, _S100, &kernelContext_21);
    if(run_0)
    {
        uint _S102 = groups_of_0((&kernelContext_21)->params_0->bond_count_0, 64U);
        _S100 = _S102;
    }
    else
    {
        _S100 = 0U;
    }
    set_args_0(1U, _S100, &kernelContext_21);
    if(run_0)
    {
        uint _S103 = groups_of_0((&kernelContext_21)->params_0->body_count_0, 64U);
        _S100 = _S103;
    }
    else
    {
        _S100 = 0U;
    }
    set_args_0(2U, _S100, &kernelContext_21);
    if(run_0)
    {
        uint _S104 = groups_of_0((&kernelContext_21)->params_0->chunk_count_0, 1024U);
        _S100 = _S104;
    }
    else
    {
        _S100 = 0U;
    }
    set_args_0(3U, _S100, &kernelContext_21);
    if(run_0)
    {
        uint _S105 = groups_of_0((&kernelContext_21)->params_0->body_count_0, 1024U);
        _S100 = _S105;
    }
    else
    {
        _S100 = 0U;
    }
    set_args_0(4U, _S100, &kernelContext_21);
    int _S106;
    if(run_0)
    {
        _S106 = int(1);
    }
    else
    {
        _S106 = int(0);
    }
    set_args_0(5U, uint(_S106), &kernelContext_21);
    return;
}

bool dirty_0(uint body_0, KernelContext_0 thread* kernelContext_22)
{
    uint _S107 = a_dirty_0(kernelContext_22);
    uint _S108 = atomic_load_explicit(kernelContext_22->atom_0+(_S107 + body_0), memory_order_relaxed);
    return _S108 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_23)
{
    return kernelContext_23->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_24)
{
    return 2U * kernelContext_24->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_25;
    (&kernelContext_25)->params_0 = params_4;
    (&kernelContext_25)->work_0 = work_4;
    (&kernelContext_25)->frames_0 = frames_4;
    (&kernelContext_25)->bodies_0 = bodies_4;
    (&kernelContext_25)->chunks_0 = chunks_4;
    (&kernelContext_25)->sort_0 = sort_4;
    (&kernelContext_25)->atom_0 = atom_4;
    (&kernelContext_25)->manifolds_0 = manifolds_4;
    (&kernelContext_25)->loads_0 = loads_4;
    (&kernelContext_25)->bonds_0 = bonds_4;
    (&kernelContext_25)->args_0 = args_4;
    (&kernelContext_25)->info_0 = info_4;
    (&kernelContext_25)->scan_0 = scan_4;
    (&kernelContext_25)->parts_0 = parts_4;
    (&kernelContext_25)->colliders_0 = colliders_4;
    (&kernelContext_25)->collider_info_0 = collider_info_4;
    (&kernelContext_25)->view_chunks_0 = view_chunks_4;
    (&kernelContext_25)->view_0 = view_4;
    (&kernelContext_25)->objects_0 = objects_4;
    (&kernelContext_25)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_25)->lines_0 = lines_4;
    (&kernelContext_25)->geometry_0 = geometry_4;
    (&kernelContext_25)->shots_0 = shots_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_25)->g_scan_0 = &g_scan_4;
    uint c_2 = id_3.x;
    bool _S109;
    if(c_2 >= (params_4->chunk_count_0))
    {
        _S109 = true;
    }
    else
    {
        uint _S110 = body_of_0(c_2, &kernelContext_25);
        bool _S111 = dirty_0(_S110, &kernelContext_25);
        _S109 = !_S111;
    }
    if(_S109)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_25)->atom_0+(a_parent_0() + c_2), c_2, memory_order_relaxed);
    uint _S112 = a_count_0(&kernelContext_25);
    atomic_store_explicit((&kernelContext_25)->atom_0+(_S112 + c_2), 0U, memory_order_relaxed);
    uint _S113 = a_anchor_0(&kernelContext_25);
    atomic_store_explicit((&kernelContext_25)->atom_0+(_S113 + c_2), 0U, memory_order_relaxed);
    uint w_2 = 0U;
    for(;;)
    {
        if(w_2 < 20U)
        {
        }
        else
        {
            break;
        }
        uint _S114 = a_sums_0(&kernelContext_25);
        atomic_store_explicit((&kernelContext_25)->atom_0+(_S114 + 20U * c_2 + w_2), 0U, memory_order_relaxed);
        w_2 = w_2 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_26;
    (&kernelContext_26)->params_0 = params_5;
    (&kernelContext_26)->work_0 = work_5;
    (&kernelContext_26)->frames_0 = frames_5;
    (&kernelContext_26)->bodies_0 = bodies_5;
    (&kernelContext_26)->chunks_0 = chunks_5;
    (&kernelContext_26)->sort_0 = sort_5;
    (&kernelContext_26)->atom_0 = atom_5;
    (&kernelContext_26)->manifolds_0 = manifolds_5;
    (&kernelContext_26)->loads_0 = loads_5;
    (&kernelContext_26)->bonds_0 = bonds_5;
    (&kernelContext_26)->args_0 = args_5;
    (&kernelContext_26)->info_0 = info_5;
    (&kernelContext_26)->scan_0 = scan_5;
    (&kernelContext_26)->parts_0 = parts_5;
    (&kernelContext_26)->colliders_0 = colliders_5;
    (&kernelContext_26)->collider_info_0 = collider_info_5;
    (&kernelContext_26)->view_chunks_0 = view_chunks_5;
    (&kernelContext_26)->view_0 = view_5;
    (&kernelContext_26)->objects_0 = objects_5;
    (&kernelContext_26)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_26)->lines_0 = lines_5;
    (&kernelContext_26)->geometry_0 = geometry_5;
    (&kernelContext_26)->shots_0 = shots_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_26)->g_scan_0 = &g_scan_5;
    uint s_0 = id_4.x;
    bool _S115;
    if(s_0 >= (params_5->body_count_0))
    {
        _S115 = true;
    }
    else
    {
        bool _S116 = dirty_0(s_0, &kernelContext_26);
        _S115 = !_S116;
    }
    if(_S115)
    {
        return;
    }
    uint _S117 = a_best_count_0(&kernelContext_26);
    atomic_store_explicit((&kernelContext_26)->atom_0+(_S117 + s_0), 0U, memory_order_relaxed);
    uint _S118 = a_best_root_0(&kernelContext_26);
    atomic_store_explicit((&kernelContext_26)->atom_0+(_S118 + s_0), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_2, KernelContext_0 thread* kernelContext_27)
{
    uint _S119 = a_parent_0();
    uint _S120 = atomic_load_explicit(kernelContext_27->atom_0+(_S119 + x_2), memory_order_relaxed);
    uint p_1 = _S120;
    uint _S121 = x_2;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S122;
        if(guard_0 < 1048576U)
        {
            _S122 = p_1 != _S121;
        }
        else
        {
            _S122 = false;
        }
        if(_S122)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_27->atom_0+(_S119 + p_1), memory_order_relaxed);
        if(gp_0 != p_1)
        {
            uint _S123;
            {
            uint _metal_cas_comparand = p_1;
            atomic_compare_exchange_weak_explicit(kernelContext_27->atom_0+(_S119 + _S121), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S123 = _metal_cas_comparand;
            }
        }
        uint _S124 = guard_0 + 1U;
        _S121 = p_1;
        p_1 = gp_0;
        guard_0 = _S124;
    }
    return _S121;
}

[[kernel]] void fr_uf_union(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_28;
    (&kernelContext_28)->params_0 = params_6;
    (&kernelContext_28)->work_0 = work_6;
    (&kernelContext_28)->frames_0 = frames_6;
    (&kernelContext_28)->bodies_0 = bodies_6;
    (&kernelContext_28)->chunks_0 = chunks_6;
    (&kernelContext_28)->sort_0 = sort_6;
    (&kernelContext_28)->atom_0 = atom_6;
    (&kernelContext_28)->manifolds_0 = manifolds_6;
    (&kernelContext_28)->loads_0 = loads_6;
    (&kernelContext_28)->bonds_0 = bonds_6;
    (&kernelContext_28)->args_0 = args_6;
    (&kernelContext_28)->info_0 = info_6;
    (&kernelContext_28)->scan_0 = scan_6;
    (&kernelContext_28)->parts_0 = parts_6;
    (&kernelContext_28)->colliders_0 = colliders_6;
    (&kernelContext_28)->collider_info_0 = collider_info_6;
    (&kernelContext_28)->view_chunks_0 = view_chunks_6;
    (&kernelContext_28)->view_0 = view_6;
    (&kernelContext_28)->objects_0 = objects_6;
    (&kernelContext_28)->shot_bodies_0 = shot_bodies_6;
    (&kernelContext_28)->lines_0 = lines_6;
    (&kernelContext_28)->geometry_0 = geometry_6;
    (&kernelContext_28)->shots_0 = shots_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_28)->g_scan_0 = &g_scan_6;
    uint i_1 = id_5.x;
    bool _S125;
    if(i_1 >= (params_6->bond_count_0))
    {
        _S125 = true;
    }
    else
    {
        uint _S126 = w_alive_0(&kernelContext_28);
        _S125 = (*((&kernelContext_28)->work_0+(_S126 + i_1))) == 0U;
    }
    if(_S125)
    {
        return;
    }
    uint4 _S127 = uint4(*((&kernelContext_28)->bonds_0+i_1)) ;
    uint _S128 = _S127.x;
    uint _S129 = body_of_0(_S128, &kernelContext_28);
    bool _S130 = dirty_0(_S129, &kernelContext_28);
    if(!_S130)
    {
        return;
    }
    uint _S131 = _S127.y;
    uint a_2 = _S128;
    uint b_2 = _S131;
    uint guard_1 = 0U;
    for(;;)
    {
        if(guard_1 < 4096U)
        {
        }
        else
        {
            break;
        }
        uint _S132 = find_0(a_2, &kernelContext_28);
        uint _S133 = find_0(b_2, &kernelContext_28);
        if(_S132 == _S133)
        {
            return;
        }
        uint _S134 = max(_S132, _S133);
        uint _S135 = min(_S132, _S133);
        uint was_0;
        {
        uint _metal_cas_comparand = _S134;
        atomic_compare_exchange_weak_explicit((&kernelContext_28)->atom_0+(a_parent_0() + _S134), &_metal_cas_comparand, _S135, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S134)
        {
            return;
        }
        uint _S136 = guard_1 + 1U;
        a_2 = _S134;
        b_2 = _S135;
        guard_1 = _S136;
    }
    uint _S137 = a_counters_0(&kernelContext_28);
    uint _S138 = atomic_fetch_or_explicit((&kernelContext_28)->atom_0+(_S137 + 3U), 4U, memory_order_relaxed);
    return;
}

void add64_0(uint at_1, float value_0, float scale_0, KernelContext_0 thread* kernelContext_29)
{
    float x_3 = round(value_0 * scale_0);
    float x_4;
    if(!((abs(x_3)) < 1.073741824e+09f))
    {
        uint _S139 = a_counters_0(kernelContext_29);
        uint _S140 = atomic_fetch_or_explicit(kernelContext_29->atom_0+(_S139 + 3U), 2U, memory_order_relaxed);
        x_4 = 0.0f;
    }
    else
    {
        x_4 = x_3;
    }
    int v_4 = int(x_4);
    uint lo_0 = (as_type<uint>((v_4)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_29->atom_0+at_1, lo_0, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_0) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S141;
    if(v_4 < int(0))
    {
        _S141 = 4294967295U;
    }
    else
    {
        _S141 = 0U;
    }
    uint hi_0 = _S141 + carry_0;
    if(hi_0 != 0U)
    {
        uint _S142 = atomic_fetch_add_explicit(kernelContext_29->atom_0+(at_1 + 1U), hi_0, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_30;
    (&kernelContext_30)->params_0 = params_7;
    (&kernelContext_30)->work_0 = work_7;
    (&kernelContext_30)->frames_0 = frames_7;
    (&kernelContext_30)->bodies_0 = bodies_7;
    (&kernelContext_30)->chunks_0 = chunks_7;
    (&kernelContext_30)->sort_0 = sort_7;
    (&kernelContext_30)->atom_0 = atom_7;
    (&kernelContext_30)->manifolds_0 = manifolds_7;
    (&kernelContext_30)->loads_0 = loads_7;
    (&kernelContext_30)->bonds_0 = bonds_7;
    (&kernelContext_30)->args_0 = args_7;
    (&kernelContext_30)->info_0 = info_7;
    (&kernelContext_30)->scan_0 = scan_7;
    (&kernelContext_30)->parts_0 = parts_7;
    (&kernelContext_30)->colliders_0 = colliders_7;
    (&kernelContext_30)->collider_info_0 = collider_info_7;
    (&kernelContext_30)->view_chunks_0 = view_chunks_7;
    (&kernelContext_30)->view_0 = view_7;
    (&kernelContext_30)->objects_0 = objects_7;
    (&kernelContext_30)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_30)->lines_0 = lines_7;
    (&kernelContext_30)->geometry_0 = geometry_7;
    (&kernelContext_30)->shots_0 = shots_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_30)->g_scan_0 = &g_scan_7;
    uint c_3 = id_6.x;
    bool _S143;
    if(c_3 >= (params_7->chunk_count_0))
    {
        _S143 = true;
    }
    else
    {
        uint _S144 = body_of_0(c_3, &kernelContext_30);
        bool _S145 = dirty_0(_S144, &kernelContext_30);
        _S143 = !_S145;
    }
    if(_S143)
    {
        return;
    }
    uint _S146 = find_0(c_3, &kernelContext_30);
    uint _S147 = w_label_0(&kernelContext_30);
    *((&kernelContext_30)->work_0+(_S147 + c_3)) = _S146;
    uint _S148 = a_count_0(&kernelContext_30);
    uint _S149 = atomic_fetch_add_explicit((&kernelContext_30)->atom_0+(_S148 + _S146), 1U, memory_order_relaxed);
    if(((&kernelContext_30)->info_0[c_3] & 1U) != 0U)
    {
        uint _S150 = a_anchor_0(&kernelContext_30);
        uint _S151 = atomic_fetch_or_explicit((&kernelContext_30)->atom_0+(_S150 + _S146), 1U, memory_order_relaxed);
    }
    uint _S152 = 4U * c_3;
    float4 _S153 = float4(*((&kernelContext_30)->chunks_0+_S152)) ;
    float4 _S154 = float4(*((&kernelContext_30)->chunks_0+(_S152 + 1U))) ;
    float4 _S155 = float4(*((&kernelContext_30)->chunks_0+(_S152 + 2U))) ;
    float m_2 = _S153.w;
    float3 d_0 = _S153.xyz - (float4(*((&kernelContext_30)->chunks_0+4U * _S146)) ).xyz;
    float dd_0 = dot(d_0, d_0);
    uint _S156 = a_sums_0(&kernelContext_30);
    uint at_2 = _S156 + 20U * _S146;
    add64_0(at_2, m_2, (&kernelContext_30)->params_0->scale_m_0, &kernelContext_30);
    float _S157 = d_0.x;
    float _S158 = m_2 * _S157;
    add64_0(at_2 + 2U, _S158, (&kernelContext_30)->params_0->scale_s_0, &kernelContext_30);
    float _S159 = d_0.y;
    float _S160 = m_2 * _S159;
    add64_0(at_2 + 4U, _S160, (&kernelContext_30)->params_0->scale_s_0, &kernelContext_30);
    float _S161 = d_0.z;
    add64_0(at_2 + 6U, m_2 * _S161, (&kernelContext_30)->params_0->scale_s_0, &kernelContext_30);
    add64_0(at_2 + 8U, _S154.x + m_2 * (dd_0 - _S157 * _S157), (&kernelContext_30)->params_0->scale_j_0, &kernelContext_30);
    add64_0(at_2 + 10U, _S154.y + m_2 * (dd_0 - _S159 * _S159), (&kernelContext_30)->params_0->scale_j_0, &kernelContext_30);
    add64_0(at_2 + 12U, _S154.z + m_2 * (dd_0 - _S161 * _S161), (&kernelContext_30)->params_0->scale_j_0, &kernelContext_30);
    add64_0(at_2 + 14U, _S154.w - _S158 * _S159, (&kernelContext_30)->params_0->scale_j_0, &kernelContext_30);
    add64_0(at_2 + 16U, _S155.x - _S158 * _S161, (&kernelContext_30)->params_0->scale_j_0, &kernelContext_30);
    add64_0(at_2 + 18U, _S155.y - _S160 * _S161, (&kernelContext_30)->params_0->scale_j_0, &kernelContext_30);
    return;
}

bool is_root_0(uint c_4, KernelContext_0 thread* kernelContext_31)
{
    bool _S162;
    if(c_4 < (kernelContext_31->params_0->chunk_count_0))
    {
        uint _S163 = body_of_0(c_4, kernelContext_31);
        bool _S164 = dirty_0(_S163, kernelContext_31);
        _S162 = _S164;
    }
    else
    {
        _S162 = false;
    }
    if(_S162)
    {
        uint _S165 = w_label_0(kernelContext_31);
        _S162 = (*(kernelContext_31->work_0+(_S165 + c_4))) == c_4;
    }
    else
    {
        _S162 = false;
    }
    return _S162;
}

[[kernel]] void fr_best_count(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_32;
    (&kernelContext_32)->params_0 = params_8;
    (&kernelContext_32)->work_0 = work_8;
    (&kernelContext_32)->frames_0 = frames_8;
    (&kernelContext_32)->bodies_0 = bodies_8;
    (&kernelContext_32)->chunks_0 = chunks_8;
    (&kernelContext_32)->sort_0 = sort_8;
    (&kernelContext_32)->atom_0 = atom_8;
    (&kernelContext_32)->manifolds_0 = manifolds_8;
    (&kernelContext_32)->loads_0 = loads_8;
    (&kernelContext_32)->bonds_0 = bonds_8;
    (&kernelContext_32)->args_0 = args_8;
    (&kernelContext_32)->info_0 = info_8;
    (&kernelContext_32)->scan_0 = scan_8;
    (&kernelContext_32)->parts_0 = parts_8;
    (&kernelContext_32)->colliders_0 = colliders_8;
    (&kernelContext_32)->collider_info_0 = collider_info_8;
    (&kernelContext_32)->view_chunks_0 = view_chunks_8;
    (&kernelContext_32)->view_0 = view_8;
    (&kernelContext_32)->objects_0 = objects_8;
    (&kernelContext_32)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_32)->lines_0 = lines_8;
    (&kernelContext_32)->geometry_0 = geometry_8;
    (&kernelContext_32)->shots_0 = shots_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_32)->g_scan_0 = &g_scan_8;
    uint c_5 = id_7.x;
    bool _S166 = is_root_0(c_5, &kernelContext_32);
    if(!_S166)
    {
        return;
    }
    uint _S167 = a_best_count_0(&kernelContext_32);
    uint _S168 = body_of_0(c_5, &kernelContext_32);
    atomic<uint> device* _S169 = (&kernelContext_32)->atom_0+(_S167 + _S168);
    uint _S170 = a_count_0(&kernelContext_32);
    uint _S171 = atomic_load_explicit((&kernelContext_32)->atom_0+(_S170 + c_5), memory_order_relaxed);
    uint _S172 = atomic_fetch_max_explicit(_S169, _S171, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_33;
    (&kernelContext_33)->params_0 = params_9;
    (&kernelContext_33)->work_0 = work_9;
    (&kernelContext_33)->frames_0 = frames_9;
    (&kernelContext_33)->bodies_0 = bodies_9;
    (&kernelContext_33)->chunks_0 = chunks_9;
    (&kernelContext_33)->sort_0 = sort_9;
    (&kernelContext_33)->atom_0 = atom_9;
    (&kernelContext_33)->manifolds_0 = manifolds_9;
    (&kernelContext_33)->loads_0 = loads_9;
    (&kernelContext_33)->bonds_0 = bonds_9;
    (&kernelContext_33)->args_0 = args_9;
    (&kernelContext_33)->info_0 = info_9;
    (&kernelContext_33)->scan_0 = scan_9;
    (&kernelContext_33)->parts_0 = parts_9;
    (&kernelContext_33)->colliders_0 = colliders_9;
    (&kernelContext_33)->collider_info_0 = collider_info_9;
    (&kernelContext_33)->view_chunks_0 = view_chunks_9;
    (&kernelContext_33)->view_0 = view_9;
    (&kernelContext_33)->objects_0 = objects_9;
    (&kernelContext_33)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_33)->lines_0 = lines_9;
    (&kernelContext_33)->geometry_0 = geometry_9;
    (&kernelContext_33)->shots_0 = shots_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_33)->g_scan_0 = &g_scan_9;
    uint c_6 = id_8.x;
    bool _S173 = is_root_0(c_6, &kernelContext_33);
    if(!_S173)
    {
        return;
    }
    uint _S174 = body_of_0(c_6, &kernelContext_33);
    uint _S175 = a_count_0(&kernelContext_33);
    uint _S176 = atomic_load_explicit((&kernelContext_33)->atom_0+(_S175 + c_6), memory_order_relaxed);
    uint _S177 = a_best_count_0(&kernelContext_33);
    uint _S178 = atomic_load_explicit((&kernelContext_33)->atom_0+(_S177 + _S174), memory_order_relaxed);
    if(_S176 == _S178)
    {
        uint _S179 = a_best_root_0(&kernelContext_33);
        uint _S180 = atomic_fetch_min_explicit((&kernelContext_33)->atom_0+(_S179 + _S174), c_6, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_34;
    (&kernelContext_34)->params_0 = params_10;
    (&kernelContext_34)->work_0 = work_10;
    (&kernelContext_34)->frames_0 = frames_10;
    (&kernelContext_34)->bodies_0 = bodies_10;
    (&kernelContext_34)->chunks_0 = chunks_10;
    (&kernelContext_34)->sort_0 = sort_10;
    (&kernelContext_34)->atom_0 = atom_10;
    (&kernelContext_34)->manifolds_0 = manifolds_10;
    (&kernelContext_34)->loads_0 = loads_10;
    (&kernelContext_34)->bonds_0 = bonds_10;
    (&kernelContext_34)->args_0 = args_10;
    (&kernelContext_34)->info_0 = info_10;
    (&kernelContext_34)->scan_0 = scan_10;
    (&kernelContext_34)->parts_0 = parts_10;
    (&kernelContext_34)->colliders_0 = colliders_10;
    (&kernelContext_34)->collider_info_0 = collider_info_10;
    (&kernelContext_34)->view_chunks_0 = view_chunks_10;
    (&kernelContext_34)->view_0 = view_10;
    (&kernelContext_34)->objects_0 = objects_10;
    (&kernelContext_34)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_34)->lines_0 = lines_10;
    (&kernelContext_34)->geometry_0 = geometry_10;
    (&kernelContext_34)->shots_0 = shots_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_34)->g_scan_0 = &g_scan_10;
    uint c_7 = id_9.x;
    if(c_7 >= (params_10->chunk_count_0))
    {
        return;
    }
    uint _S181 = w_new_0(&kernelContext_34);
    uint device* _S182 = (&kernelContext_34)->work_0+(_S181 + c_7);
    bool _S183 = is_root_0(c_7, &kernelContext_34);
    bool _S184;
    if(_S183)
    {
        uint _S185 = a_best_root_0(&kernelContext_34);
        uint _S186 = body_of_0(c_7, &kernelContext_34);
        uint _S187 = atomic_load_explicit((&kernelContext_34)->atom_0+(_S185 + _S186), memory_order_relaxed);
        _S184 = _S187 != c_7;
    }
    else
    {
        _S184 = false;
    }
    uint _S188;
    if(_S184)
    {
        _S188 = 1U;
    }
    else
    {
        _S188 = 0U;
    }
    *_S182 = _S188;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]])
{
    uint step_2;
    uint add_0;
    thread KernelContext_0 kernelContext_35;
    (&kernelContext_35)->params_0 = params_11;
    (&kernelContext_35)->work_0 = work_11;
    (&kernelContext_35)->frames_0 = frames_11;
    (&kernelContext_35)->bodies_0 = bodies_11;
    (&kernelContext_35)->chunks_0 = chunks_11;
    (&kernelContext_35)->sort_0 = sort_11;
    (&kernelContext_35)->atom_0 = atom_11;
    (&kernelContext_35)->manifolds_0 = manifolds_11;
    (&kernelContext_35)->loads_0 = loads_11;
    (&kernelContext_35)->bonds_0 = bonds_11;
    (&kernelContext_35)->args_0 = args_11;
    (&kernelContext_35)->info_0 = info_11;
    (&kernelContext_35)->scan_0 = scan_11;
    (&kernelContext_35)->parts_0 = parts_11;
    (&kernelContext_35)->colliders_0 = colliders_11;
    (&kernelContext_35)->collider_info_0 = collider_info_11;
    (&kernelContext_35)->view_chunks_0 = view_chunks_11;
    (&kernelContext_35)->view_0 = view_11;
    (&kernelContext_35)->objects_0 = objects_11;
    (&kernelContext_35)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_35)->lines_0 = lines_11;
    (&kernelContext_35)->geometry_0 = geometry_11;
    (&kernelContext_35)->shots_0 = shots_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_35)->g_scan_0 = &g_scan_11;
    uint t_2 = thread_0.x;
    uint _S189 = group_0.x;
    uint _S190 = 1024U * _S189 + 4U * t_2;
    thread array<uint, int(4)> v_5;
    uint j_2 = 0U;
    uint sum_2 = 0U;
    for(;;)
    {
        if(j_2 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S190 + j_2) < ((&kernelContext_35)->scan_0->n_0))
        {
            step_2 = *((&kernelContext_35)->work_0+((&kernelContext_35)->scan_0->in_base_0 + _S190 + j_2));
        }
        else
        {
            step_2 = 0U;
        }
        v_5[j_2] = step_2;
        uint sum_3 = sum_2 + step_2;
        j_2 = j_2 + 1U;
        sum_2 = sum_3;
    }
    (*(&kernelContext_35)->g_scan_0)[t_2] = sum_2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    step_2 = 1U;
    for(;;)
    {
        if(step_2 < 256U)
        {
        }
        else
        {
            break;
        }
        if(t_2 >= step_2)
        {
            add_0 = (*(&kernelContext_35)->g_scan_0)[t_2 - step_2];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_35)->g_scan_0)[t_2] = (*(&kernelContext_35)->g_scan_0)[t_2] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_2 = step_2 << 1U;
    }
    uint _S191 = (*(&kernelContext_35)->g_scan_0)[t_2] - sum_2;
    j_2 = 0U;
    add_0 = _S191;
    for(;;)
    {
        if(j_2 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S190 + j_2) < ((&kernelContext_35)->scan_0->n_0))
        {
            *((&kernelContext_35)->work_0+((&kernelContext_35)->scan_0->out_base_0 + _S190 + j_2)) = add_0;
        }
        uint run_1 = add_0 + v_5[j_2];
        j_2 = j_2 + 1U;
        add_0 = run_1;
    }
    if(t_2 == 255U)
    {
        *((&kernelContext_35)->work_0+((&kernelContext_35)->scan_0->block_base_0 + _S189)) = (*(&kernelContext_35)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_36;
    (&kernelContext_36)->params_0 = params_12;
    (&kernelContext_36)->work_0 = work_12;
    (&kernelContext_36)->frames_0 = frames_12;
    (&kernelContext_36)->bodies_0 = bodies_12;
    (&kernelContext_36)->chunks_0 = chunks_12;
    (&kernelContext_36)->sort_0 = sort_12;
    (&kernelContext_36)->atom_0 = atom_12;
    (&kernelContext_36)->manifolds_0 = manifolds_12;
    (&kernelContext_36)->loads_0 = loads_12;
    (&kernelContext_36)->bonds_0 = bonds_12;
    (&kernelContext_36)->args_0 = args_12;
    (&kernelContext_36)->info_0 = info_12;
    (&kernelContext_36)->scan_0 = scan_12;
    (&kernelContext_36)->parts_0 = parts_12;
    (&kernelContext_36)->colliders_0 = colliders_12;
    (&kernelContext_36)->collider_info_0 = collider_info_12;
    (&kernelContext_36)->view_chunks_0 = view_chunks_12;
    (&kernelContext_36)->view_0 = view_12;
    (&kernelContext_36)->objects_0 = objects_12;
    (&kernelContext_36)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_36)->lines_0 = lines_12;
    (&kernelContext_36)->geometry_0 = geometry_12;
    (&kernelContext_36)->shots_0 = shots_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_36)->g_scan_0 = &g_scan_12;
    uint t_3 = thread_1.x;
    uint blocks_0 = (scan_12->n_0 + 1023U) / 1024U;
    bool _S192 = t_3 < blocks_0;
    uint v_6;
    if(_S192)
    {
        v_6 = *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->block_base_0 + t_3));
    }
    else
    {
        v_6 = 0U;
    }
    (*(&kernelContext_36)->g_scan_0)[t_3] = v_6;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint step_3 = 1U;
    for(;;)
    {
        if(step_3 < 1024U)
        {
        }
        else
        {
            break;
        }
        uint add_1;
        if(t_3 >= step_3)
        {
            add_1 = (*(&kernelContext_36)->g_scan_0)[t_3 - step_3];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_36)->g_scan_0)[t_3] = (*(&kernelContext_36)->g_scan_0)[t_3] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_3 = step_3 << 1U;
    }
    if(_S192)
    {
        *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->block_base_0 + t_3)) = (*(&kernelContext_36)->g_scan_0)[t_3] - v_6;
    }
    if(t_3 == 1023U)
    {
        *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_36)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_37;
    (&kernelContext_37)->params_0 = params_13;
    (&kernelContext_37)->work_0 = work_13;
    (&kernelContext_37)->frames_0 = frames_13;
    (&kernelContext_37)->bodies_0 = bodies_13;
    (&kernelContext_37)->chunks_0 = chunks_13;
    (&kernelContext_37)->sort_0 = sort_13;
    (&kernelContext_37)->atom_0 = atom_13;
    (&kernelContext_37)->manifolds_0 = manifolds_13;
    (&kernelContext_37)->loads_0 = loads_13;
    (&kernelContext_37)->bonds_0 = bonds_13;
    (&kernelContext_37)->args_0 = args_13;
    (&kernelContext_37)->info_0 = info_13;
    (&kernelContext_37)->scan_0 = scan_13;
    (&kernelContext_37)->parts_0 = parts_13;
    (&kernelContext_37)->colliders_0 = colliders_13;
    (&kernelContext_37)->collider_info_0 = collider_info_13;
    (&kernelContext_37)->view_chunks_0 = view_chunks_13;
    (&kernelContext_37)->view_0 = view_13;
    (&kernelContext_37)->objects_0 = objects_13;
    (&kernelContext_37)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_37)->lines_0 = lines_13;
    (&kernelContext_37)->geometry_0 = geometry_13;
    (&kernelContext_37)->shots_0 = shots_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_37)->g_scan_0 = &g_scan_13;
    uint i_2 = id_10.x;
    if(i_2 >= (scan_13->n_0))
    {
        return;
    }
    uint device* _S193 = (&kernelContext_37)->work_0+((&kernelContext_37)->scan_0->out_base_0 + i_2);
    *_S193 = *_S193 + *((&kernelContext_37)->work_0+((&kernelContext_37)->scan_0->block_base_0 + i_2 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_38;
    (&kernelContext_38)->params_0 = params_14;
    (&kernelContext_38)->work_0 = work_14;
    (&kernelContext_38)->frames_0 = frames_14;
    (&kernelContext_38)->bodies_0 = bodies_14;
    (&kernelContext_38)->chunks_0 = chunks_14;
    (&kernelContext_38)->sort_0 = sort_14;
    (&kernelContext_38)->atom_0 = atom_14;
    (&kernelContext_38)->manifolds_0 = manifolds_14;
    (&kernelContext_38)->loads_0 = loads_14;
    (&kernelContext_38)->bonds_0 = bonds_14;
    (&kernelContext_38)->args_0 = args_14;
    (&kernelContext_38)->info_0 = info_14;
    (&kernelContext_38)->scan_0 = scan_14;
    (&kernelContext_38)->parts_0 = parts_14;
    (&kernelContext_38)->colliders_0 = colliders_14;
    (&kernelContext_38)->collider_info_0 = collider_info_14;
    (&kernelContext_38)->view_chunks_0 = view_chunks_14;
    (&kernelContext_38)->view_0 = view_14;
    (&kernelContext_38)->objects_0 = objects_14;
    (&kernelContext_38)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_38)->lines_0 = lines_14;
    (&kernelContext_38)->geometry_0 = geometry_14;
    (&kernelContext_38)->shots_0 = shots_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_38)->g_scan_0 = &g_scan_14;
    uint s_1 = id_11.x;
    bool _S194;
    if(s_1 >= (params_14->body_count_0))
    {
        _S194 = true;
    }
    else
    {
        uint _S195 = w_free_0(&kernelContext_38);
        _S194 = (*((&kernelContext_38)->work_0+(_S195 + s_1))) == 0U;
    }
    if(_S194)
    {
        return;
    }
    uint _S196 = w_free_list_0(&kernelContext_38);
    uint _S197 = w_free_rank_0(&kernelContext_38);
    *((&kernelContext_38)->work_0+(_S196 + *((&kernelContext_38)->work_0+(_S197 + s_1)))) = s_1;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_39)
{
    uint _S198 = w_free_list_0(kernelContext_39);
    return _S198 + kernelContext_39->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_40)
{
    uint _S199 = scan_new_blocks_0(kernelContext_40);
    return _S199 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_41)
{
    uint blocks_1 = (kernelContext_41->params_0->body_count_0 + 1023U) / 1024U;
    uint _S200 = scan_free_blocks_0(kernelContext_41);
    return *(kernelContext_41->work_0+(_S200 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_42;
    (&kernelContext_42)->params_0 = params_15;
    (&kernelContext_42)->work_0 = work_15;
    (&kernelContext_42)->frames_0 = frames_15;
    (&kernelContext_42)->bodies_0 = bodies_15;
    (&kernelContext_42)->chunks_0 = chunks_15;
    (&kernelContext_42)->sort_0 = sort_15;
    (&kernelContext_42)->atom_0 = atom_15;
    (&kernelContext_42)->manifolds_0 = manifolds_15;
    (&kernelContext_42)->loads_0 = loads_15;
    (&kernelContext_42)->bonds_0 = bonds_15;
    (&kernelContext_42)->args_0 = args_15;
    (&kernelContext_42)->info_0 = info_15;
    (&kernelContext_42)->scan_0 = scan_15;
    (&kernelContext_42)->parts_0 = parts_15;
    (&kernelContext_42)->colliders_0 = colliders_15;
    (&kernelContext_42)->collider_info_0 = collider_info_15;
    (&kernelContext_42)->view_chunks_0 = view_chunks_15;
    (&kernelContext_42)->view_0 = view_15;
    (&kernelContext_42)->objects_0 = objects_15;
    (&kernelContext_42)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_42)->lines_0 = lines_15;
    (&kernelContext_42)->geometry_0 = geometry_15;
    (&kernelContext_42)->shots_0 = shots_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_42)->g_scan_0 = &g_scan_15;
    uint c_8 = id_12.x;
    bool _S201 = is_root_0(c_8, &kernelContext_42);
    if(!_S201)
    {
        return;
    }
    uint _S202 = body_of_0(c_8, &kernelContext_42);
    uint _S203 = a_best_root_0(&kernelContext_42);
    uint _S204 = atomic_load_explicit((&kernelContext_42)->atom_0+(_S203 + _S202), memory_order_relaxed);
    uint slot_1;
    if(_S204 != c_8)
    {
        uint _S205 = w_rank_0(&kernelContext_42);
        uint device* _S206 = (&kernelContext_42)->work_0+(_S205 + c_8);
        uint k_1 = *_S206;
        uint _S207 = free_total_0(&kernelContext_42);
        if((*_S206) < _S207)
        {
            uint _S208 = w_free_list_0(&kernelContext_42);
            uint device* _S209 = (&kernelContext_42)->work_0+(_S208 + k_1);
            uint slot_2 = *_S209;
            uint _S210 = w_free_0(&kernelContext_42);
            *((&kernelContext_42)->work_0+(_S210 + *_S209)) = 0U;
            uint _S211 = a_counters_0(&kernelContext_42);
            uint _S212 = atomic_fetch_add_explicit((&kernelContext_42)->atom_0+(_S211 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S213 = a_counters_0(&kernelContext_42);
            uint _S214 = atomic_fetch_or_explicit((&kernelContext_42)->atom_0+(_S213 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S202;
        }
    }
    else
    {
        slot_1 = _S202;
    }
    uint _S215 = w_slot_0(&kernelContext_42);
    *((&kernelContext_42)->work_0+(_S215 + c_8)) = slot_1;
    return;
}

float read64_0(uint at_3, float scale_1, KernelContext_0 thread* kernelContext_43)
{
    uint _S216 = atomic_load_explicit(kernelContext_43->atom_0+(at_3 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S216)));
    uint lo_1 = atomic_load_explicit(kernelContext_43->atom_0+at_3, memory_order_relaxed);
    return (float(hi_1) * 4.294967296e+09f + float(lo_1)) / scale_1;
}

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
    thread Quat_0 q_8;
    if(m22_0 <= 0.0f)
    {
        float dif10_0 = m11_0 - m00_0;
        float omm22_0 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            float f_3 = omm22_0 - dif10_0;
            float inv_0 = 0.5f / sqrt(f_3);
            (&q_8)->x_0 = f_3 * inv_0;
            (&q_8)->y_0 = (m01_0 + m10_0) * inv_0;
            (&q_8)->z_0 = (m02_0 + m20_0) * inv_0;
            (&q_8)->w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            float f_4 = omm22_0 + dif10_0;
            float inv_1 = 0.5f / sqrt(f_4);
            (&q_8)->x_0 = (m01_0 + m10_0) * inv_1;
            (&q_8)->y_0 = f_4 * inv_1;
            (&q_8)->z_0 = (m12_0 + m21_0) * inv_1;
            (&q_8)->w_0 = (m20_0 - m02_0) * inv_1;
        }
    }
    else
    {
        float sum10_0 = m11_0 + m00_0;
        float opm22_0 = 1.0f + m22_0;
        if(sum10_0 <= 0.0f)
        {
            float f_5 = opm22_0 - sum10_0;
            float inv_2 = 0.5f / sqrt(f_5);
            (&q_8)->x_0 = (m02_0 + m20_0) * inv_2;
            (&q_8)->y_0 = (m12_0 + m21_0) * inv_2;
            (&q_8)->z_0 = f_5 * inv_2;
            (&q_8)->w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            float f_6 = opm22_0 + sum10_0;
            float inv_3 = 0.5f / sqrt(f_6);
            (&q_8)->x_0 = (m12_0 - m21_0) * inv_3;
            (&q_8)->y_0 = (m20_0 - m02_0) * inv_3;
            (&q_8)->z_0 = (m01_0 - m10_0) * inv_3;
            (&q_8)->w_0 = f_6 * inv_3;
        }
    }
    return q_8;
}

void principal_axes_0(float3 r0_0, float3 r1_0, float3 r2_0, Quat_0 thread* rot_0, float3 thread* moments_0)
{
    array<float, int(3)> _S217 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S218 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S219 = { { r2_0.x, r2_0.y, r2_0.z } };
    thread array<array<float, int(3)>, int(3)> a_3;
    a_3[int(0)] = _S217;
    a_3[int(1)] = _S218;
    a_3[int(2)] = _S219;
    array<float, int(3)> _S220 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S221 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S222 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_7;
    v_7[int(0)] = _S220;
    v_7[int(1)] = _S221;
    v_7[int(2)] = _S222;
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
        uint q_9;
        if((abs(a_3[int(0)][int(2)])) > (abs(a_3[0U][1U])))
        {
            q_9 = 2U;
        }
        else
        {
            q_9 = 1U;
        }
        uint p_2;
        uint q_10;
        if((abs(a_3[int(1)][int(2)])) > (abs(a_3[0U][q_9])))
        {
            p_2 = 1U;
            q_10 = 2U;
        }
        else
        {
            p_2 = 0U;
            q_10 = q_9;
        }
        if((abs(a_3[p_2][q_10])) <= (9.99999982451670044e-15f * max(max(abs(a_3[int(0)][int(0)]), abs(a_3[int(1)][int(1)])), max(abs(a_3[int(2)][int(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        float theta_0 = 0.5f * atan2(2.0f * a_3[p_2][q_10], a_3[q_10][q_10] - a_3[p_2][p_2]);
        float s_2 = sin(theta_0);
        float c_9 = cos(theta_0);
        float app_0 = a_3[p_2][p_2];
        float aqq_0 = a_3[q_10][q_10];
        float apq_0 = a_3[p_2][q_10];
        uint k_2 = 0U;
        for(;;)
        {
            if(k_2 < 3U)
            {
            }
            else
            {
                break;
            }
            bool _S223;
            if(k_2 != p_2)
            {
                _S223 = k_2 != q_10;
            }
            else
            {
                _S223 = false;
            }
            if(_S223)
            {
                float kp_0 = a_3[k_2][p_2];
                float kq_0 = a_3[k_2][q_10];
                float _S224 = c_9 * a_3[k_2][p_2] - s_2 * a_3[k_2][q_10];
                a_3[k_2][p_2] = _S224;
                a_3[p_2][k_2] = _S224;
                float _S225 = s_2 * kp_0 + c_9 * kq_0;
                a_3[k_2][q_10] = _S225;
                a_3[q_10][k_2] = _S225;
            }
            k_2 = k_2 + 1U;
        }
        float _S226 = c_9 * c_9;
        float _S227 = 2.0f * s_2 * c_9 * apq_0;
        float _S228 = s_2 * s_2;
        a_3[p_2][p_2] = _S226 * app_0 - _S227 + _S228 * aqq_0;
        a_3[q_10][q_10] = _S228 * app_0 + _S227 + _S226 * aqq_0;
        a_3[p_2][q_10] = 0.0f;
        a_3[q_10][p_2] = 0.0f;
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
            float kp_1 = v_7[row_0][p_2];
            float kq_1 = v_7[row_0][q_10];
            v_7[row_0][p_2] = c_9 * v_7[row_0][p_2] - s_2 * v_7[row_0][q_10];
            v_7[row_0][q_10] = s_2 * kp_1 + c_9 * kq_1;
            row_0 = row_0 + 1U;
        }
        it_0 = it_0 + 1U;
    }
    float3 x_5 = float3(v_7[int(0)][int(0)], v_7[int(1)][int(0)], v_7[int(2)][int(0)]);
    float3 y_1 = float3(v_7[int(0)][int(1)], v_7[int(1)][int(1)], v_7[int(2)][int(1)]);
    float3 z_1 = float3(v_7[int(0)][int(2)], v_7[int(1)][int(2)], v_7[int(2)][int(2)]);
    float3 z_2;
    if((dot(cross(x_5, y_1), z_1)) < 0.0f)
    {
        z_2 = - z_1;
    }
    else
    {
        z_2 = z_1;
    }
    thread Quat_0 _S229 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S230 = normalized_0(&_S229);
    *rot_0 = _S230;
    *moments_0 = float3(a_3[int(0)][int(0)], a_3[int(1)][int(1)], a_3[int(2)][int(2)]);
    return;
}

float3 to_rubble_0(float3 v_8)
{
    return float3(v_8.x, v_8.z, - v_8.y);
}

Quat_0 rot_to_rubble_0(const Quat_0 thread* q_11)
{
    thread Quat_0 axes_1;
    (&axes_1)->w_0 = 0.70710676908493042f;
    (&axes_1)->x_0 = -0.70710676908493042f;
    (&axes_1)->y_0 = 0.0f;
    (&axes_1)->z_0 = 0.0f;
    thread Quat_0 _S231 = axes_1;
    Quat_0 _S232 = quat_mul_0(&_S231, q_11);
    thread Quat_0 _S233 = _S232;
    Quat_0 _S234 = normalized_0(&_S233);
    return _S234;
}

uint stage_0(uint c_10, KernelContext_0 thread* kernelContext_44)
{
    return 2U * kernelContext_44->params_0->body_count_0 + 9U * c_10;
}

float4 quat_vec_0(const Quat_0 thread* q_12)
{
    return float4(q_12->x_0, q_12->y_0, q_12->z_0, q_12->w_0);
}

[[kernel]] void fr_piece(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_45;
    (&kernelContext_45)->params_0 = params_16;
    (&kernelContext_45)->work_0 = work_16;
    (&kernelContext_45)->frames_0 = frames_16;
    (&kernelContext_45)->bodies_0 = bodies_16;
    (&kernelContext_45)->chunks_0 = chunks_16;
    (&kernelContext_45)->sort_0 = sort_16;
    (&kernelContext_45)->atom_0 = atom_16;
    (&kernelContext_45)->manifolds_0 = manifolds_16;
    (&kernelContext_45)->loads_0 = loads_16;
    (&kernelContext_45)->bonds_0 = bonds_16;
    (&kernelContext_45)->args_0 = args_16;
    (&kernelContext_45)->info_0 = info_16;
    (&kernelContext_45)->scan_0 = scan_16;
    (&kernelContext_45)->parts_0 = parts_16;
    (&kernelContext_45)->colliders_0 = colliders_16;
    (&kernelContext_45)->collider_info_0 = collider_info_16;
    (&kernelContext_45)->view_chunks_0 = view_chunks_16;
    (&kernelContext_45)->view_0 = view_16;
    (&kernelContext_45)->objects_0 = objects_16;
    (&kernelContext_45)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_45)->lines_0 = lines_16;
    (&kernelContext_45)->geometry_0 = geometry_16;
    (&kernelContext_45)->shots_0 = shots_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_45)->g_scan_0 = &g_scan_16;
    uint c_11 = id_13.x;
    bool _S235 = is_root_0(c_11, &kernelContext_45);
    if(!_S235)
    {
        return;
    }
    uint _S236 = body_of_0(c_11, &kernelContext_45);
    uint pb_0 = 7U * _S236;
    uint _S237 = a_sums_0(&kernelContext_45);
    uint at_4 = _S237 + 20U * c_11;
    float _S238 = read64_0(at_4, (&kernelContext_45)->params_0->scale_m_0, &kernelContext_45);
    float _S239 = read64_0(at_4 + 2U, (&kernelContext_45)->params_0->scale_s_0, &kernelContext_45);
    float _S240 = read64_0(at_4 + 4U, (&kernelContext_45)->params_0->scale_s_0, &kernelContext_45);
    float _S241 = read64_0(at_4 + 6U, (&kernelContext_45)->params_0->scale_s_0, &kernelContext_45);
    float3 d_1 = float3(_S239, _S240, _S241) / float3(max(_S238, 1.00000000317107685e-30f)) ;
    float _S242 = read64_0(at_4 + 8U, (&kernelContext_45)->params_0->scale_j_0, &kernelContext_45);
    float _S243 = read64_0(at_4 + 10U, (&kernelContext_45)->params_0->scale_j_0, &kernelContext_45);
    float _S244 = read64_0(at_4 + 12U, (&kernelContext_45)->params_0->scale_j_0, &kernelContext_45);
    float _S245 = read64_0(at_4 + 14U, (&kernelContext_45)->params_0->scale_j_0, &kernelContext_45);
    float _S246 = read64_0(at_4 + 16U, (&kernelContext_45)->params_0->scale_j_0, &kernelContext_45);
    float _S247 = read64_0(at_4 + 18U, (&kernelContext_45)->params_0->scale_j_0, &kernelContext_45);
    float dd_1 = dot(d_1, d_1);
    float _S248 = d_1.x;
    float _S249 = _S238 * _S248;
    float _S250 = d_1.y;
    float _S251 = _S245 + _S249 * _S250;
    float _S252 = d_1.z;
    float _S253 = _S246 + _S249 * _S252;
    float _S254 = _S247 + _S238 * _S250 * _S252;
    float3 com_1 = (float4(*((&kernelContext_45)->chunks_0+4U * c_11)) ).xyz + d_1;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S242 - _S238 * (dd_1 - _S248 * _S248), _S251, _S253), float3(_S251, _S243 - _S238 * (dd_1 - _S250 * _S250), _S254), float3(_S253, _S254, _S244 - _S238 * (dd_1 - _S252 * _S252)), &principal_0, &moments_1);
    uint _S255 = a_anchor_0(&kernelContext_45);
    uint _S256 = atomic_load_explicit((&kernelContext_45)->atom_0+(_S255 + c_11), memory_order_relaxed);
    bool anchored_0 = _S256 != 0U;
    thread Quat_0 q_13;
    thread float3 origin_2;
    structure_pose_0(_S236, pb_0, &q_13, &origin_2, &kernelContext_45);
    float3 p_parent_0 = (float4(*((&kernelContext_45)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_45)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_45)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S257 = q_13;
    float3 _S258 = rotate_0(&_S257, com_1);
    float3 pos_0 = to_rubble_0(origin_2 + _S258);
    thread Quat_0 _S259 = q_13;
    thread Quat_0 _S260 = principal_0;
    Quat_0 _S261 = quat_mul_0(&_S259, &_S260);
    thread Quat_0 _S262 = _S261;
    Quat_0 _S263 = normalized_0(&_S262);
    thread Quat_0 _S264 = _S263;
    Quat_0 _S265 = rot_to_rubble_0(&_S264);
    bool _S266;
    if(anchored_0)
    {
        _S266 = true;
    }
    else
    {
        _S266 = !(_S238 > 0.0f);
    }
    float inv_mass_1;
    if(_S266)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / _S238;
    }
    float3 _S267 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S268;
        if((moments_1.x) > 0.0f)
        {
            _S268 = 1.0f / moments_1.x;
        }
        else
        {
            _S268 = 0.0f;
        }
        float _S269;
        if((moments_1.y) > 0.0f)
        {
            _S269 = 1.0f / moments_1.y;
        }
        else
        {
            _S269 = 0.0f;
        }
        float _S270;
        if((moments_1.z) > 0.0f)
        {
            _S270 = 1.0f / moments_1.z;
        }
        else
        {
            _S270 = 0.0f;
        }
        inv_inertia_0 = float3(_S268, _S269, _S270);
    }
    else
    {
        inv_inertia_0 = _S267;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S267;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S267;
    }
    else
    {
        spin_0 = w_parent_0;
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
    uint _S271 = stage_0(c_11, &kernelContext_45);
    *((&kernelContext_45)->frames_0+_S271) = packed_float4(float4(pos_0, inv_mass_1)) ;
    packed_float4 device* _S272 = (&kernelContext_45)->frames_0+(_S271 + 1U);
    thread Quat_0 _S273 = _S265;
    float4 _S274 = quat_vec_0(&_S273);
    *_S272 = packed_float4(_S274) ;
    *((&kernelContext_45)->frames_0+(_S271 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_45)->frames_0+(_S271 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_45)->frames_0+(_S271 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_45)->params_0->step_0))))) ;
    *((&kernelContext_45)->frames_0+(_S271 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_45)->frames_0+(_S271 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_45)->frames_0+(_S271 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_45)->frames_0+2U * _S236)) ).w)) ;
    packed_float4 device* _S275 = (&kernelContext_45)->frames_0+(_S271 + 8U);
    thread Quat_0 _S276 = principal_0;
    float4 _S277 = quat_vec_0(&_S276);
    *_S275 = packed_float4(_S277) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_46;
    (&kernelContext_46)->params_0 = params_17;
    (&kernelContext_46)->work_0 = work_17;
    (&kernelContext_46)->frames_0 = frames_17;
    (&kernelContext_46)->bodies_0 = bodies_17;
    (&kernelContext_46)->chunks_0 = chunks_17;
    (&kernelContext_46)->sort_0 = sort_17;
    (&kernelContext_46)->atom_0 = atom_17;
    (&kernelContext_46)->manifolds_0 = manifolds_17;
    (&kernelContext_46)->loads_0 = loads_17;
    (&kernelContext_46)->bonds_0 = bonds_17;
    (&kernelContext_46)->args_0 = args_17;
    (&kernelContext_46)->info_0 = info_17;
    (&kernelContext_46)->scan_0 = scan_17;
    (&kernelContext_46)->parts_0 = parts_17;
    (&kernelContext_46)->colliders_0 = colliders_17;
    (&kernelContext_46)->collider_info_0 = collider_info_17;
    (&kernelContext_46)->view_chunks_0 = view_chunks_17;
    (&kernelContext_46)->view_0 = view_17;
    (&kernelContext_46)->objects_0 = objects_17;
    (&kernelContext_46)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_46)->lines_0 = lines_17;
    (&kernelContext_46)->geometry_0 = geometry_17;
    (&kernelContext_46)->shots_0 = shots_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_46)->g_scan_0 = &g_scan_17;
    uint c_12 = id_14.x;
    bool _S278 = is_root_0(c_12, &kernelContext_46);
    if(!_S278)
    {
        return;
    }
    uint _S279 = w_slot_0(&kernelContext_46);
    uint slot_3 = *((&kernelContext_46)->work_0+(_S279 + c_12));
    uint _S280 = stage_0(c_12, &kernelContext_46);
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
        uint w_3 = 0U;
        for(;;)
        {
            if(w_3 < 7U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_46)->bodies_0+(7U * (slot_3 + region_0 * (&kernelContext_46)->params_0->max_bodies_0) + w_3)) = packed_float4(float4(*((&kernelContext_46)->frames_0+(_S280 + w_3))) ) ;
            w_3 = w_3 + 1U;
        }
        region_0 = region_0 + 1U;
    }
    uint _S281 = 2U * slot_3;
    *((&kernelContext_46)->frames_0+_S281) = packed_float4(float4(*((&kernelContext_46)->frames_0+(_S280 + 7U))) ) ;
    *((&kernelContext_46)->frames_0+(_S281 + 1U)) = packed_float4(float4(*((&kernelContext_46)->frames_0+(_S280 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_47;
    (&kernelContext_47)->params_0 = params_18;
    (&kernelContext_47)->work_0 = work_18;
    (&kernelContext_47)->frames_0 = frames_18;
    (&kernelContext_47)->bodies_0 = bodies_18;
    (&kernelContext_47)->chunks_0 = chunks_18;
    (&kernelContext_47)->sort_0 = sort_18;
    (&kernelContext_47)->atom_0 = atom_18;
    (&kernelContext_47)->manifolds_0 = manifolds_18;
    (&kernelContext_47)->loads_0 = loads_18;
    (&kernelContext_47)->bonds_0 = bonds_18;
    (&kernelContext_47)->args_0 = args_18;
    (&kernelContext_47)->info_0 = info_18;
    (&kernelContext_47)->scan_0 = scan_18;
    (&kernelContext_47)->parts_0 = parts_18;
    (&kernelContext_47)->colliders_0 = colliders_18;
    (&kernelContext_47)->collider_info_0 = collider_info_18;
    (&kernelContext_47)->view_chunks_0 = view_chunks_18;
    (&kernelContext_47)->view_0 = view_18;
    (&kernelContext_47)->objects_0 = objects_18;
    (&kernelContext_47)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_47)->lines_0 = lines_18;
    (&kernelContext_47)->geometry_0 = geometry_18;
    (&kernelContext_47)->shots_0 = shots_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_47)->g_scan_0 = &g_scan_18;
    uint c_13 = id_15.x;
    bool _S282;
    if(c_13 >= (params_18->chunk_count_0))
    {
        _S282 = true;
    }
    else
    {
        uint _S283 = body_of_0(c_13, &kernelContext_47);
        bool _S284 = dirty_0(_S283, &kernelContext_47);
        _S282 = !_S284;
    }
    if(_S282)
    {
        return;
    }
    uint _S285 = w_label_0(&kernelContext_47);
    uint device* _S286 = (&kernelContext_47)->work_0+(_S285 + c_13);
    uint r_6 = *_S286;
    uint _S287 = w_slot_0(&kernelContext_47);
    uint device* _S288 = (&kernelContext_47)->work_0+(_S287 + *_S286);
    uint slot_4 = *_S288;
    *((&kernelContext_47)->work_0+(w_body_0() + c_13)) = *_S288;
    uint _S289 = 2U * slot_4;
    float3 _S290 = (float4(*((&kernelContext_47)->frames_0+_S289)) ).xyz;
    thread Quat_0 _S291 = quat_of_0(float4(*((&kernelContext_47)->frames_0+(_S289 + 1U))) );
    Quat_0 _S292 = conj_0(&_S291);
    float inv_mass_2 = (float4(*((&kernelContext_47)->bodies_0+7U * slot_4)) ).w;
    float _S293;
    if(inv_mass_2 > 0.0f)
    {
        _S293 = 1.0f / inv_mass_2;
    }
    else
    {
        _S293 = float_bits_0(2139095040U);
    }
    float4 _S294 = float4(*((&kernelContext_47)->chunks_0+(4U * c_13 + 2U))) ;
    uint first_1 = (as_type<uint>((_S294.z)));
    uint _S295 = (as_type<uint>((_S294.w)));
    float4 _S296 = float4(*((&kernelContext_47)->chunks_0+(4U * r_6 + 3U))) ;
    uint p_3 = first_1;
    for(;;)
    {
        if(p_3 < (first_1 + _S295))
        {
        }
        else
        {
            break;
        }
        uint _S297 = 2U * p_3;
        float4 _S298 = float4(*((&kernelContext_47)->parts_0+_S297)) ;
        uint col_0 = (as_type<uint>((_S298.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_47)->parts_0+(_S297 + 1U))) );
        uint _S299 = 4U * col_0;
        float4 _S300 = float4(*((&kernelContext_47)->colliders_0+_S299)) ;
        *((&kernelContext_47)->colliders_0+_S299) = packed_float4(float4((as_type<float>((slot_4))), _S300.y, _S300.z, _S300.w)) ;
        uint _S301 = _S299 + 1U;
        packed_float4 device* _S302 = (&kernelContext_47)->colliders_0+_S301;
        float3 _S303 = _S298.xyz - _S290;
        thread Quat_0 _S304 = _S292;
        float3 _S305 = rotate_0(&_S304, _S303);
        *_S302 = packed_float4(float4(_S305, (float4(*((&kernelContext_47)->colliders_0+_S301)) ).w)) ;
        packed_float4 device* _S306 = (&kernelContext_47)->colliders_0+(_S299 + 2U);
        thread Quat_0 _S307 = _S292;
        thread Quat_0 _S308 = rot_1;
        Quat_0 _S309 = quat_mul_0(&_S307, &_S308);
        thread Quat_0 _S310 = _S309;
        Quat_0 _S311 = normalized_0(&_S310);
        thread Quat_0 _S312 = _S311;
        float4 _S313 = quat_vec_0(&_S312);
        *_S306 = packed_float4(_S313) ;
        uint _S314 = 2U * col_0;
        uint4 _S315 = uint4(*((&kernelContext_47)->collider_info_0+_S314)) ;
        *((&kernelContext_47)->collider_info_0+_S314) = packed_uint4(uint4(_S315.x, slot_4, _S315.z, _S315.w)) ;
        *((&kernelContext_47)->collider_info_0+(_S314 + 1U)) = packed_uint4(uint4((as_type<uint>((_S293))), (as_type<uint>((_S296.x))), (as_type<uint>((_S296.w))), 0U)) ;
        p_3 = p_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_48;
    (&kernelContext_48)->params_0 = params_19;
    (&kernelContext_48)->work_0 = work_19;
    (&kernelContext_48)->frames_0 = frames_19;
    (&kernelContext_48)->bodies_0 = bodies_19;
    (&kernelContext_48)->chunks_0 = chunks_19;
    (&kernelContext_48)->sort_0 = sort_19;
    (&kernelContext_48)->atom_0 = atom_19;
    (&kernelContext_48)->manifolds_0 = manifolds_19;
    (&kernelContext_48)->loads_0 = loads_19;
    (&kernelContext_48)->bonds_0 = bonds_19;
    (&kernelContext_48)->args_0 = args_19;
    (&kernelContext_48)->info_0 = info_19;
    (&kernelContext_48)->scan_0 = scan_19;
    (&kernelContext_48)->parts_0 = parts_19;
    (&kernelContext_48)->colliders_0 = colliders_19;
    (&kernelContext_48)->collider_info_0 = collider_info_19;
    (&kernelContext_48)->view_chunks_0 = view_chunks_19;
    (&kernelContext_48)->view_0 = view_19;
    (&kernelContext_48)->objects_0 = objects_19;
    (&kernelContext_48)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_48)->lines_0 = lines_19;
    (&kernelContext_48)->geometry_0 = geometry_19;
    (&kernelContext_48)->shots_0 = shots_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_48)->g_scan_0 = &g_scan_19;
    uint s_3 = id_16.x;
    if(s_3 >= (params_19->body_count_0))
    {
        return;
    }
    uint _S316 = a_dirty_0(&kernelContext_48);
    atomic_store_explicit((&kernelContext_48)->atom_0+(_S316 + s_3), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_0 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_7 = fract(h_0);
    float q_14 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_7);
    float t_4 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_7));
    uint k_3 = uint(h_0);
    float3 _S317;
    if(k_3 == 0U)
    {
        _S317 = float3(0.92000001668930054f, t_4, 0.34959998726844788f);
    }
    else
    {
        if(k_3 == 1U)
        {
            _S317 = float3(q_14, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_3 == 2U)
            {
                _S317 = float3(0.34959998726844788f, 0.92000001668930054f, t_4);
            }
            else
            {
                if(k_3 == 3U)
                {
                    _S317 = float3(0.34959998726844788f, q_14, 0.92000001668930054f);
                }
                else
                {
                    if(k_3 == 4U)
                    {
                        _S317 = float3(t_4, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S317 = float3(0.92000001668930054f, 0.34959998726844788f, q_14);
                    }
                }
            }
        }
    }
    return _S317;
}

void write_object_0(uint o_1, const Quat_0 thread* q_15, float3 t_5, float4 color_0, KernelContext_0 thread* kernelContext_49)
{
    float3 _S318 = rotate_0(q_15, float3(1.0f, 0.0f, 0.0f));
    float3 _S319 = rotate_0(q_15, float3(0.0f, 1.0f, 0.0f));
    float3 _S320 = rotate_0(q_15, float3(0.0f, 0.0f, 1.0f));
    uint _S321 = 4U * o_1;
    *(kernelContext_49->objects_0+_S321) = packed_float4(float4(_S318.x, _S319.x, _S320.x, t_5.x)) ;
    *(kernelContext_49->objects_0+(_S321 + 1U)) = packed_float4(float4(_S318.y, _S319.y, _S320.y, t_5.y)) ;
    *(kernelContext_49->objects_0+(_S321 + 2U)) = packed_float4(float4(_S318.z, _S319.z, _S320.z, t_5.z)) ;
    *(kernelContext_49->objects_0+(_S321 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_50;
    (&kernelContext_50)->params_0 = params_20;
    (&kernelContext_50)->work_0 = work_20;
    (&kernelContext_50)->frames_0 = frames_20;
    (&kernelContext_50)->bodies_0 = bodies_20;
    (&kernelContext_50)->chunks_0 = chunks_20;
    (&kernelContext_50)->sort_0 = sort_20;
    (&kernelContext_50)->atom_0 = atom_20;
    (&kernelContext_50)->manifolds_0 = manifolds_20;
    (&kernelContext_50)->loads_0 = loads_20;
    (&kernelContext_50)->bonds_0 = bonds_20;
    (&kernelContext_50)->args_0 = args_20;
    (&kernelContext_50)->info_0 = info_20;
    (&kernelContext_50)->scan_0 = scan_20;
    (&kernelContext_50)->parts_0 = parts_20;
    (&kernelContext_50)->colliders_0 = colliders_20;
    (&kernelContext_50)->collider_info_0 = collider_info_20;
    (&kernelContext_50)->view_chunks_0 = view_chunks_20;
    (&kernelContext_50)->view_0 = view_20;
    (&kernelContext_50)->objects_0 = objects_20;
    (&kernelContext_50)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_50)->lines_0 = lines_20;
    (&kernelContext_50)->geometry_0 = geometry_20;
    (&kernelContext_50)->shots_0 = shots_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_50)->g_scan_0 = &g_scan_20;
    uint k_4 = id_17.x;
    if(k_4 >= (params_20->chunk_count_0))
    {
        return;
    }
    uint _S322 = body_of_0(k_4, &kernelContext_50);
    uint b_3 = 7U * _S322;
    thread Quat_0 q_16;
    thread float3 origin_3;
    structure_pose_0(_S322, b_3, &q_16, &origin_3, &kernelContext_50);
    uint _S323 = 3U * k_4;
    float4 _S324 = float4(*((&kernelContext_50)->view_chunks_0+_S323)) ;
    uint _S325 = _S323 + 2U;
    float3 base_2 = (float4(*((&kernelContext_50)->view_chunks_0+_S325)) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_50)->bodies_0+b_3)) ).w) == 0.0f;
    bool _S326;
    if(((&kernelContext_50)->view_0->mode_0) == 0U)
    {
        _S326 = true;
    }
    else
    {
        _S326 = ((&kernelContext_50)->view_0->mode_0) == 3U;
    }
    float surface_0;
    if(_S326)
    {
        surface_0 = max((float4(*((&kernelContext_50)->view_chunks_0+_S325)) ).w, 1.0f);
    }
    else
    {
        surface_0 = 1.0f;
    }
    float3 color_1;
    if(((&kernelContext_50)->view_0->mode_0) == 1U)
    {
        float _S327 = max(length((float4(*((&kernelContext_50)->bodies_0+(b_3 + 3U))) ).xyz), length((float4(*((&kernelContext_50)->bodies_0+(b_3 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S327 > ((&kernelContext_50)->view_0->rest_speed_0))
            {
                color_1 = float3(1.0f, 0.44999998807907104f, 0.11999999731779099f);
            }
            else
            {
                color_1 = float3(0.25f, 0.44999998807907104f, 0.85000002384185791f);
            }
        }
    }
    else
    {
        if(((&kernelContext_50)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S322);
        }
        else
        {
            if(((&kernelContext_50)->view_0->mode_0) == 3U)
            {
                color_1 = base_2;
            }
            else
            {
                if(fixed_0)
                {
                    color_1 = base_2;
                }
                else
                {
                    color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S322);
                }
            }
        }
    }
    uint _S328 = (as_type<uint>((_S324.w)));
    Quat_0 _S329 = quat_of_0(float4(*((&kernelContext_50)->view_chunks_0+(_S323 + 1U))) );
    thread Quat_0 _S330 = q_16;
    thread Quat_0 _S331 = _S329;
    Quat_0 _S332 = quat_mul_0(&_S330, &_S331);
    thread Quat_0 _S333 = _S332;
    Quat_0 _S334 = normalized_0(&_S333);
    float3 _S335 = _S324.xyz;
    thread Quat_0 _S336 = q_16;
    float3 _S337 = rotate_0(&_S336, _S335);
    float3 _S338 = origin_3 + _S337;
    float4 _S339 = float4(color_1, surface_0);
    thread Quat_0 _S340 = _S334;
    write_object_0(_S328, &_S340, _S338, _S339, &kernelContext_50);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_51;
    (&kernelContext_51)->params_0 = params_21;
    (&kernelContext_51)->work_0 = work_21;
    (&kernelContext_51)->frames_0 = frames_21;
    (&kernelContext_51)->bodies_0 = bodies_21;
    (&kernelContext_51)->chunks_0 = chunks_21;
    (&kernelContext_51)->sort_0 = sort_21;
    (&kernelContext_51)->atom_0 = atom_21;
    (&kernelContext_51)->manifolds_0 = manifolds_21;
    (&kernelContext_51)->loads_0 = loads_21;
    (&kernelContext_51)->bonds_0 = bonds_21;
    (&kernelContext_51)->args_0 = args_21;
    (&kernelContext_51)->info_0 = info_21;
    (&kernelContext_51)->scan_0 = scan_21;
    (&kernelContext_51)->parts_0 = parts_21;
    (&kernelContext_51)->colliders_0 = colliders_21;
    (&kernelContext_51)->collider_info_0 = collider_info_21;
    (&kernelContext_51)->view_chunks_0 = view_chunks_21;
    (&kernelContext_51)->view_0 = view_21;
    (&kernelContext_51)->objects_0 = objects_21;
    (&kernelContext_51)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_51)->lines_0 = lines_21;
    (&kernelContext_51)->geometry_0 = geometry_21;
    (&kernelContext_51)->shots_0 = shots_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_51)->g_scan_0 = &g_scan_21;
    uint i_3 = id_18.x;
    if(i_3 >= (view_21->shot_count_1))
    {
        return;
    }
    uint b_4 = 7U * (&kernelContext_51)->shot_bodies_0[i_3];
    uint o_2 = (&kernelContext_51)->view_0->impactor_base_0 + i_3;
    if(((float4(*((&kernelContext_51)->bodies_0+b_4)) ).w) == 0.0f)
    {
        uint w_4 = 0U;
        for(;;)
        {
            if(w_4 < 4U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_51)->objects_0+(4U * o_2 + w_4)) = packed_float4(float4(0.0f) ) ;
            w_4 = w_4 + 1U;
        }
        return;
    }
    thread Quat_0 _S341 = quat_of_0(float4(*((&kernelContext_51)->bodies_0+(b_4 + 1U))) );
    Quat_0 _S342 = rot_from_rubble_0(&_S341);
    float3 _S343 = from_rubble_0((float4(*((&kernelContext_51)->bodies_0+b_4)) ).xyz);
    float4 _S344 = float4(0.30000001192092896f, 0.31000000238418579f, 0.34000000357627869f, 4.0f);
    thread Quat_0 _S345 = _S342;
    write_object_0(o_2, &_S345, _S343, _S344, &kernelContext_51);
    return;
}

float3 chunk_world_0(uint k_5, KernelContext_0 thread* kernelContext_52)
{
    uint _S346 = body_of_0(k_5, kernelContext_52);
    thread Quat_0 q_17;
    thread float3 origin_4;
    structure_pose_0(_S346, 7U * _S346, &q_17, &origin_4, kernelContext_52);
    float3 _S347 = (float4(*(kernelContext_52->view_chunks_0+3U * k_5)) ).xyz;
    thread Quat_0 _S348 = q_17;
    float3 _S349 = rotate_0(&_S348, _S347);
    return origin_4 + _S349;
}

void write_line_vertex_0(uint v_9, float3 p_4, KernelContext_0 thread* kernelContext_53)
{
    uint _S350 = 7U * v_9;
    *(kernelContext_53->lines_0+_S350) = p_4.x;
    *(kernelContext_53->lines_0+(_S350 + 1U)) = p_4.y;
    *(kernelContext_53->lines_0+(_S350 + 2U)) = p_4.z;
    *(kernelContext_53->lines_0+(_S350 + 3U)) = 1.0f;
    *(kernelContext_53->lines_0+(_S350 + 4U)) = 0.86000001430511475f;
    *(kernelContext_53->lines_0+(_S350 + 5U)) = 0.25f;
    *(kernelContext_53->lines_0+(_S350 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_54;
    (&kernelContext_54)->params_0 = params_22;
    (&kernelContext_54)->work_0 = work_22;
    (&kernelContext_54)->frames_0 = frames_22;
    (&kernelContext_54)->bodies_0 = bodies_22;
    (&kernelContext_54)->chunks_0 = chunks_22;
    (&kernelContext_54)->sort_0 = sort_22;
    (&kernelContext_54)->atom_0 = atom_22;
    (&kernelContext_54)->manifolds_0 = manifolds_22;
    (&kernelContext_54)->loads_0 = loads_22;
    (&kernelContext_54)->bonds_0 = bonds_22;
    (&kernelContext_54)->args_0 = args_22;
    (&kernelContext_54)->info_0 = info_22;
    (&kernelContext_54)->scan_0 = scan_22;
    (&kernelContext_54)->parts_0 = parts_22;
    (&kernelContext_54)->colliders_0 = colliders_22;
    (&kernelContext_54)->collider_info_0 = collider_info_22;
    (&kernelContext_54)->view_chunks_0 = view_chunks_22;
    (&kernelContext_54)->view_0 = view_22;
    (&kernelContext_54)->objects_0 = objects_22;
    (&kernelContext_54)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_54)->lines_0 = lines_22;
    (&kernelContext_54)->geometry_0 = geometry_22;
    (&kernelContext_54)->shots_0 = shots_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_54)->g_scan_0 = &g_scan_22;
    uint i_4 = id_19.x;
    if(i_4 >= (params_22->bond_count_0))
    {
        return;
    }
    uint4 _S351 = uint4(*((&kernelContext_54)->bonds_0+i_4)) ;
    float3 _S352 = chunk_world_0(_S351.x, &kernelContext_54);
    uint _S353 = w_alive_0(&kernelContext_54);
    float3 b_5;
    if((*((&kernelContext_54)->work_0+(_S353 + i_4))) != 0U)
    {
        float3 _S354 = chunk_world_0(_S351.y, &kernelContext_54);
        b_5 = _S354;
    }
    else
    {
        b_5 = _S352;
    }
    uint _S355 = 2U * i_4;
    write_line_vertex_0(_S355, _S352, &kernelContext_54);
    write_line_vertex_0(_S355 + 1U, b_5, &kernelContext_54);
    return;
}

uint a_strike_0(KernelContext_0 thread* kernelContext_55)
{
    uint _S356 = a_counters_1(kernelContext_55);
    return _S356 + 16U;
}

uint strike_at_0(uint b_6, KernelContext_0 thread* kernelContext_56)
{
    return 2U * kernelContext_56->params_0->body_count_0 + 9U * kernelContext_56->params_0->chunk_count_0 + 3U * b_6;
}

[[kernel]] void fr_strike_reset(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], atomic<uint> device* atom_23 [[buffer(5)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_57;
    (&kernelContext_57)->params_0 = params_23;
    (&kernelContext_57)->work_0 = work_23;
    (&kernelContext_57)->frames_0 = frames_23;
    (&kernelContext_57)->bodies_0 = bodies_23;
    (&kernelContext_57)->chunks_0 = chunks_23;
    (&kernelContext_57)->sort_0 = sort_23;
    (&kernelContext_57)->atom_0 = atom_23;
    (&kernelContext_57)->manifolds_0 = manifolds_23;
    (&kernelContext_57)->loads_0 = loads_23;
    (&kernelContext_57)->bonds_0 = bonds_23;
    (&kernelContext_57)->args_0 = args_23;
    (&kernelContext_57)->info_0 = info_23;
    (&kernelContext_57)->scan_0 = scan_23;
    (&kernelContext_57)->parts_0 = parts_23;
    (&kernelContext_57)->colliders_0 = colliders_23;
    (&kernelContext_57)->collider_info_0 = collider_info_23;
    (&kernelContext_57)->view_chunks_0 = view_chunks_23;
    (&kernelContext_57)->view_0 = view_23;
    (&kernelContext_57)->objects_0 = objects_23;
    (&kernelContext_57)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_57)->lines_0 = lines_23;
    (&kernelContext_57)->geometry_0 = geometry_23;
    (&kernelContext_57)->shots_0 = shots_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_57)->g_scan_0 = &g_scan_23;
    uint b_7 = id_20.x;
    if(b_7 >= (params_23->body_count_0))
    {
        return;
    }
    uint _S357 = a_strike_0(&kernelContext_57);
    uint at_5 = _S357 + 9U * b_7;
    uint w_5 = 0U;
    for(;;)
    {
        if(w_5 < 3U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit((&kernelContext_57)->atom_0+(at_5 + w_5), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_57)->atom_0+(at_5 + 3U + w_5), 0U, memory_order_relaxed);
        w_5 = w_5 + 1U;
    }
    atomic_store_explicit((&kernelContext_57)->atom_0+(at_5 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_57)->atom_0+(at_5 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_57)->atom_0+(at_5 + 8U), 4294967295U, memory_order_relaxed);
    uint _S358 = strike_at_0(b_7, &kernelContext_57);
    (*((&kernelContext_57)->frames_0+_S358))[int(3)] = 0.0f;
    return;
}

float3 chunk_box_world_0(uint k_6, Quat_0 thread* q_18, KernelContext_0 thread* kernelContext_58)
{
    uint _S359 = body_of_0(k_6, kernelContext_58);
    thread float3 origin_5;
    structure_pose_0(_S359, 7U * _S359, q_18, &origin_5, kernelContext_58);
    float3 _S360 = (float4(*(kernelContext_58->geometry_0+3U * k_6)) ).xyz;
    thread Quat_0 _S361 = *q_18;
    float3 _S362 = rotate_0(&_S361, _S360);
    return origin_5 + _S362;
}

uint ordered_0(float f_8)
{
    uint u_0 = (as_type<uint>((f_8)));
    uint _S363;
    if((u_0 & 2147483648U) != 0U)
    {
        _S363 = ~u_0;
    }
    else
    {
        _S363 = u_0 | 2147483648U;
    }
    return _S363;
}

[[kernel]] void fr_strike_bounds(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], atomic<uint> device* atom_24 [[buffer(5)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_59;
    (&kernelContext_59)->params_0 = params_24;
    (&kernelContext_59)->work_0 = work_24;
    (&kernelContext_59)->frames_0 = frames_24;
    (&kernelContext_59)->bodies_0 = bodies_24;
    (&kernelContext_59)->chunks_0 = chunks_24;
    (&kernelContext_59)->sort_0 = sort_24;
    (&kernelContext_59)->atom_0 = atom_24;
    (&kernelContext_59)->manifolds_0 = manifolds_24;
    (&kernelContext_59)->loads_0 = loads_24;
    (&kernelContext_59)->bonds_0 = bonds_24;
    (&kernelContext_59)->args_0 = args_24;
    (&kernelContext_59)->info_0 = info_24;
    (&kernelContext_59)->scan_0 = scan_24;
    (&kernelContext_59)->parts_0 = parts_24;
    (&kernelContext_59)->colliders_0 = colliders_24;
    (&kernelContext_59)->collider_info_0 = collider_info_24;
    (&kernelContext_59)->view_chunks_0 = view_chunks_24;
    (&kernelContext_59)->view_0 = view_24;
    (&kernelContext_59)->objects_0 = objects_24;
    (&kernelContext_59)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_59)->lines_0 = lines_24;
    (&kernelContext_59)->geometry_0 = geometry_24;
    (&kernelContext_59)->shots_0 = shots_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_59)->g_scan_0 = &g_scan_24;
    uint k_7 = id_21.x;
    bool _S364;
    if(k_7 >= (params_24->chunk_count_0))
    {
        _S364 = true;
    }
    else
    {
        _S364 = ((&kernelContext_59)->params_0->shot_count_0) == 0U;
    }
    if(_S364)
    {
        return;
    }
    thread Quat_0 q_19;
    float3 _S365 = chunk_box_world_0(k_7, &q_19, &kernelContext_59);
    float _S366 = (float4(*((&kernelContext_59)->geometry_0+3U * k_7)) ).w;
    uint _S367 = a_strike_0(&kernelContext_59);
    uint _S368 = body_of_0(k_7, &kernelContext_59);
    uint _S369 = _S367 + 9U * _S368;
    uint ax_0 = 0U;
    for(;;)
    {
        if(ax_0 < 3U)
        {
        }
        else
        {
            break;
        }
        uint _S370 = ax_0;
        uint _S371 = atomic_fetch_min_explicit((&kernelContext_59)->atom_0+(_S369 + ax_0), ordered_0(_S365[ax_0] - _S366), memory_order_relaxed);
        uint _S372 = atomic_fetch_max_explicit((&kernelContext_59)->atom_0+(_S369 + 3U + ax_0), ordered_0(_S365[_S370] + _S366), memory_order_relaxed);
        ax_0 = ax_0 + 1U;
    }
    return;
}

float unordered_0(uint u_1)
{
    uint _S373;
    if((u_1 & 2147483648U) != 0U)
    {
        _S373 = u_1 & 2147483647U;
    }
    else
    {
        _S373 = ~u_1;
    }
    return (as_type<float>((_S373)));
}

void shot_motion_0(uint k_8, float3 thread* c_14, float3 thread* vs_0, Quat_0 thread* q_20, KernelContext_0 thread* kernelContext_60)
{
    uint _S374 = 4U * k_8;
    uint slot_6 = (as_type<uint>(((float4(*(kernelContext_60->shots_0+_S374)) ).x)));
    uint sb_0 = 7U * slot_6;
    *c_14 = from_rubble_0((float4(*(kernelContext_60->bodies_0+sb_0)) ).xyz);
    *vs_0 = from_rubble_0((float4(*(kernelContext_60->bodies_0+(sb_0 + 3U))) ).xyz);
    uint _S375 = sb_0 + 1U;
    thread Quat_0 _S376 = quat_of_0(float4(*(kernelContext_60->bodies_0+_S375)) );
    Quat_0 _S377 = rot_from_rubble_0(&_S376);
    *q_20 = _S377;
    float4 _S378 = float4(*(kernelContext_60->shots_0+(_S374 + 2U))) ;
    float _S379 = _S378.w;
    if(_S379 == 0.0f)
    {
        return;
    }
    float3 p_5;
    if(_S379 == 1.0f)
    {
        thread Quat_0 qs_0;
        thread float3 os_0;
        structure_pose_0(slot_6, sb_0, &qs_0, &os_0, kernelContext_60);
        float3 _S380 = _S378.xyz;
        thread Quat_0 _S381 = qs_0;
        float3 _S382 = rotate_0(&_S381, _S380);
        p_5 = os_0 + _S382;
    }
    else
    {
        Quat_0 qb_0 = quat_of_0(float4(*(kernelContext_60->bodies_0+_S375)) );
        float3 _S383 = (float4(*(kernelContext_60->bodies_0+sb_0)) ).xyz;
        float3 _S384 = _S378.xyz;
        thread Quat_0 _S385 = qb_0;
        float3 _S386 = rotate_0(&_S385, _S384);
        float3 _S387 = from_rubble_0(_S383 + _S386);
        Quat_0 _S388 = quat_of_0(float4(*(kernelContext_60->shots_0+(_S374 + 3U))) );
        thread Quat_0 _S389 = qb_0;
        thread Quat_0 _S390 = _S388;
        Quat_0 _S391 = quat_mul_0(&_S389, &_S390);
        thread Quat_0 _S392 = _S391;
        Quat_0 _S393 = rot_from_rubble_0(&_S392);
        thread Quat_0 _S394 = _S393;
        Quat_0 _S395 = normalized_0(&_S394);
        *q_20 = _S395;
        p_5 = _S387;
    }
    *vs_0 = *vs_0 + cross(from_rubble_0((float4(*(kernelContext_60->bodies_0+(sb_0 + 4U))) ).xyz), p_5 - *c_14);
    *c_14 = p_5;
    return;
}

[[kernel]] void fr_strike_find(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], atomic<uint> device* atom_25 [[buffer(5)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_61;
    (&kernelContext_61)->params_0 = params_25;
    (&kernelContext_61)->work_0 = work_25;
    (&kernelContext_61)->frames_0 = frames_25;
    (&kernelContext_61)->bodies_0 = bodies_25;
    (&kernelContext_61)->chunks_0 = chunks_25;
    (&kernelContext_61)->sort_0 = sort_25;
    (&kernelContext_61)->atom_0 = atom_25;
    (&kernelContext_61)->manifolds_0 = manifolds_25;
    (&kernelContext_61)->loads_0 = loads_25;
    (&kernelContext_61)->bonds_0 = bonds_25;
    (&kernelContext_61)->args_0 = args_25;
    (&kernelContext_61)->info_0 = info_25;
    (&kernelContext_61)->scan_0 = scan_25;
    (&kernelContext_61)->parts_0 = parts_25;
    (&kernelContext_61)->colliders_0 = colliders_25;
    (&kernelContext_61)->collider_info_0 = collider_info_25;
    (&kernelContext_61)->view_chunks_0 = view_chunks_25;
    (&kernelContext_61)->view_0 = view_25;
    (&kernelContext_61)->objects_0 = objects_25;
    (&kernelContext_61)->shot_bodies_0 = shot_bodies_25;
    (&kernelContext_61)->lines_0 = lines_25;
    (&kernelContext_61)->geometry_0 = geometry_25;
    (&kernelContext_61)->shots_0 = shots_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_61)->g_scan_0 = &g_scan_25;
    uint b_8 = id_22.x;
    bool _S396;
    if(b_8 >= (params_25->body_count_0))
    {
        _S396 = true;
    }
    else
    {
        _S396 = ((&kernelContext_61)->params_0->shot_count_0) == 0U;
    }
    if(_S396)
    {
        return;
    }
    uint _S397 = a_strike_0(&kernelContext_61);
    uint at_6 = _S397 + 9U * b_8;
    uint _S398 = atomic_load_explicit((&kernelContext_61)->atom_0+at_6, memory_order_relaxed);
    if(_S398 == 4294967295U)
    {
        return;
    }
    uint _S399 = atomic_load_explicit((&kernelContext_61)->atom_0+at_6, memory_order_relaxed);
    float _S400 = unordered_0(_S399);
    uint _S401 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 1U), memory_order_relaxed);
    float _S402 = unordered_0(_S401);
    uint _S403 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 2U), memory_order_relaxed);
    float3 _S404 = float3(_S400, _S402, unordered_0(_S403));
    uint _S405 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 3U), memory_order_relaxed);
    float _S406 = unordered_0(_S405);
    uint _S407 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 4U), memory_order_relaxed);
    float _S408 = unordered_0(_S407);
    uint _S409 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 5U), memory_order_relaxed);
    float3 _S410 = float3(_S406, _S408, unordered_0(_S409));
    float3 _S411 = from_rubble_0((float4(*((&kernelContext_61)->bodies_0+(7U * b_8 + 3U))) ).xyz);
    float _S412 = float_bits_0(2139095040U);
    uint k_9 = 0U;
    for(;;)
    {
        if(k_9 < ((&kernelContext_61)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S413 = float4(*((&kernelContext_61)->shots_0+4U * k_9)) ;
        uint slot_7 = (as_type<uint>((_S413.x)));
        if(((float4(*((&kernelContext_61)->bodies_0+7U * slot_7)) ).w) == 0.0f)
        {
            _S396 = true;
        }
        else
        {
            _S396 = slot_7 == b_8;
        }
        if(_S396)
        {
            k_9 = k_9 + 1U;
            continue;
        }
        thread float3 c_15;
        thread float3 vs_1;
        thread Quat_0 qs_1;
        shot_motion_0(k_9, &c_15, &vs_1, &qs_1, &kernelContext_61);
        float3 rel_0 = vs_1 - _S411;
        if(!((length(rel_0)) > ((&kernelContext_61)->params_0->strike_speed_0)))
        {
            k_9 = k_9 + 1U;
            continue;
        }
        float3 _S414 = float3(_S413.y) ;
        float3 _S415 = _S404 - _S414;
        float3 _S416 = _S410 + _S414;
        float _S417 = (&kernelContext_61)->params_0->dt_0;
        float t0_0 = 0.0f;
        float t1_0 = _S417;
        uint ax_1 = 0U;
        for(;;)
        {
            if(ax_1 < 3U)
            {
            }
            else
            {
                break;
            }
            uint _S418 = ax_1;
            if((abs(rel_0[ax_1])) < 9.999999960041972e-13f)
            {
                bool _S419;
                if((c_15[ax_1]) < (_S415[ax_1]))
                {
                    _S419 = true;
                }
                else
                {
                    _S419 = (c_15[ax_1]) > (_S416[ax_1]);
                }
                float t0_1;
                if(_S419)
                {
                    t0_1 = _S412;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_1 = ax_1 + 1U;
                continue;
            }
            float ta_0 = (_S415[ax_1] - c_15[ax_1]) / rel_0[_S418];
            float tb_0 = (_S416[ax_1] - c_15[ax_1]) / rel_0[_S418];
            float _S420 = min(t1_0, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_0 = _S420;
            ax_1 = ax_1 + 1U;
        }
        if(t0_0 <= t1_0)
        {
            uint _S421 = strike_at_0(b_8, &kernelContext_61);
            *((&kernelContext_61)->frames_0+_S421) = packed_float4(float4(c_15, float(k_9 + 1U))) ;
            *((&kernelContext_61)->frames_0+(_S421 + 1U)) = packed_float4(float4(rel_0, t0_0)) ;
            return;
        }
        k_9 = k_9 + 1U;
    }
    return;
}

bool struck_0(uint b_9, uint thread* shot_0, KernelContext_0 thread* kernelContext_62)
{
    uint _S422 = strike_at_0(b_9, kernelContext_62);
    uint k_10 = uint((float4(*(kernelContext_62->frames_0+_S422)) ).w);
    *shot_0 = k_10 - 1U;
    return k_10 != 0U;
}

bool shot_is_box_0(uint k_11, KernelContext_0 thread* kernelContext_63)
{
    return ((float4(*(kernelContext_63->shots_0+(4U * k_11 + 2U))) ).w) == 2.0f;
}

float box_extent_0(const Quat_0 thread* q_21, float3 half_1, float3 axis_0)
{
    float _S423 = half_1.x;
    float3 _S424 = rotate_1(q_21, float3(1.0f, 0.0f, 0.0f));
    float _S425 = _S423 * abs(dot(_S424, axis_0));
    float _S426 = half_1.y;
    float3 _S427 = rotate_1(q_21, float3(0.0f, 1.0f, 0.0f));
    float _S428 = _S425 + _S426 * abs(dot(_S427, axis_0));
    float _S429 = half_1.z;
    float3 _S430 = rotate_1(q_21, float3(0.0f, 0.0f, 1.0f));
    return _S428 + _S429 * abs(dot(_S430, axis_0));
}

void sweep_axis_0(float3 axis_1, float3 o_3, float3 rel_1, float ext_0, float thread* t_in_0, float thread* t_out_0)
{
    float inf_0 = float_bits_0(2139095040U);
    float p_6 = dot(o_3, axis_1);
    float dv_0 = dot(rel_1, axis_1);
    if((abs(dv_0)) < 9.999999960041972e-13f)
    {
        if((abs(p_6)) > ext_0)
        {
            *t_in_0 = inf_0;
        }
        return;
    }
    float ta_1 = (- ext_0 - p_6) / dv_0;
    float tb_1 = (ext_0 - p_6) / dv_0;
    *t_in_0 = max(*t_in_0, min(ta_1, tb_1));
    *t_out_0 = min(*t_out_0, max(ta_1, tb_1));
    return;
}

[[kernel]] void fr_strike_entry(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], atomic<uint> device* atom_26 [[buffer(5)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_64;
    (&kernelContext_64)->params_0 = params_26;
    (&kernelContext_64)->work_0 = work_26;
    (&kernelContext_64)->frames_0 = frames_26;
    (&kernelContext_64)->bodies_0 = bodies_26;
    (&kernelContext_64)->chunks_0 = chunks_26;
    (&kernelContext_64)->sort_0 = sort_26;
    (&kernelContext_64)->atom_0 = atom_26;
    (&kernelContext_64)->manifolds_0 = manifolds_26;
    (&kernelContext_64)->loads_0 = loads_26;
    (&kernelContext_64)->bonds_0 = bonds_26;
    (&kernelContext_64)->args_0 = args_26;
    (&kernelContext_64)->info_0 = info_26;
    (&kernelContext_64)->scan_0 = scan_26;
    (&kernelContext_64)->parts_0 = parts_26;
    (&kernelContext_64)->colliders_0 = colliders_26;
    (&kernelContext_64)->collider_info_0 = collider_info_26;
    (&kernelContext_64)->view_chunks_0 = view_chunks_26;
    (&kernelContext_64)->view_0 = view_26;
    (&kernelContext_64)->objects_0 = objects_26;
    (&kernelContext_64)->shot_bodies_0 = shot_bodies_26;
    (&kernelContext_64)->lines_0 = lines_26;
    (&kernelContext_64)->geometry_0 = geometry_26;
    (&kernelContext_64)->shots_0 = shots_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_64)->g_scan_0 = &g_scan_26;
    uint k_12 = id_23.x;
    bool _S431;
    if(k_12 >= (params_26->chunk_count_0))
    {
        _S431 = true;
    }
    else
    {
        _S431 = ((&kernelContext_64)->params_0->shot_count_0) == 0U;
    }
    if(_S431)
    {
        return;
    }
    uint _S432 = body_of_0(k_12, &kernelContext_64);
    thread uint shot_1;
    bool _S433 = struck_0(_S432, &shot_1, &kernelContext_64);
    if(!_S433)
    {
        return;
    }
    uint _S434 = strike_at_0(_S432, &kernelContext_64);
    float4 _S435 = float4(*((&kernelContext_64)->frames_0+(_S434 + 1U))) ;
    float3 from_1 = (float4(*((&kernelContext_64)->frames_0+_S434)) ).xyz;
    float3 rel_2 = _S435.xyz;
    float3 d_2 = rel_2 / float3(max(length(rel_2), 9.999999960041972e-13f)) ;
    float _S436 = (float4(*((&kernelContext_64)->shots_0+4U * shot_1)) ).y;
    thread Quat_0 q_22;
    float3 _S437 = chunk_box_world_0(k_12, &q_22, &kernelContext_64);
    uint _S438 = 3U * k_12;
    Quat_0 _S439 = quat_of_0(float4(*((&kernelContext_64)->geometry_0+(_S438 + 2U))) );
    thread Quat_0 _S440 = q_22;
    thread Quat_0 _S441 = _S439;
    Quat_0 _S442 = quat_mul_0(&_S440, &_S441);
    thread Quat_0 _S443 = _S442;
    Quat_0 _S444 = normalized_0(&_S443);
    float3 _S445 = (float4(*((&kernelContext_64)->geometry_0+(_S438 + 1U))) ).xyz;
    float _S446 = float_bits_0(2139095040U);
    thread float t_in_1 = 0.0f;
    thread float t_out_1 = (&kernelContext_64)->params_0->dt_0;
    bool _S447 = shot_is_box_0(shot_1, &kernelContext_64);
    uint i_5;
    if(_S447)
    {
        thread float3 c_16;
        thread float3 vs_2;
        thread Quat_0 qs_2;
        shot_motion_0(shot_1, &c_16, &vs_2, &qs_2, &kernelContext_64);
        float3 _S448 = (float4(*((&kernelContext_64)->shots_0+(4U * shot_1 + 1U))) ).yzw;
        float3 _S449 = from_1 - _S437;
        i_5 = 0U;
        for(;;)
        {
            if(i_5 < 3U)
            {
            }
            else
            {
                break;
            }
            float3 _S450 = float3(0.0f) ;
            thread float3 e_1 = _S450;
            e_1[i_5] = 1.0f;
            thread Quat_0 _S451 = _S444;
            float3 _S452 = rotate_0(&_S451, e_1);
            thread Quat_0 _S453 = qs_2;
            float3 _S454 = rotate_0(&_S453, e_1);
            thread Quat_0 _S455 = qs_2;
            float _S456 = box_extent_0(&_S455, _S448, _S452);
            sweep_axis_0(_S452, _S449, rel_2, _S445[i_5] + _S456, &t_in_1, &t_out_1);
            thread Quat_0 _S457 = _S444;
            float _S458 = box_extent_0(&_S457, _S445, _S454);
            sweep_axis_0(_S454, _S449, rel_2, _S458 + _S448[i_5], &t_in_1, &t_out_1);
            uint j_3 = 0U;
            for(;;)
            {
                if(j_3 < 3U)
                {
                }
                else
                {
                    break;
                }
                thread float3 f_9 = _S450;
                f_9[j_3] = 1.0f;
                thread Quat_0 _S459 = qs_2;
                float3 _S460 = rotate_0(&_S459, f_9);
                float3 n_3 = cross(_S452, _S460);
                float len_0 = length(n_3);
                if(len_0 < 9.99999997475242708e-07f)
                {
                    j_3 = j_3 + 1U;
                    continue;
                }
                float3 n_4 = n_3 / float3(len_0) ;
                thread Quat_0 _S461 = _S444;
                float _S462 = box_extent_0(&_S461, _S445, n_4);
                thread Quat_0 _S463 = qs_2;
                float _S464 = box_extent_0(&_S463, _S448, n_4);
                sweep_axis_0(n_4, _S449, rel_2, _S462 + _S464, &t_in_1, &t_out_1);
                j_3 = j_3 + 1U;
            }
            i_5 = i_5 + 1U;
        }
    }
    else
    {
        i_5 = 0U;
        for(;;)
        {
            if(i_5 < 3U)
            {
            }
            else
            {
                break;
            }
            thread float3 e_2 = float3(0.0f) ;
            e_2[i_5] = 1.0f;
            thread Quat_0 _S465 = _S444;
            float3 _S466 = rotate_0(&_S465, e_2);
            float o_4 = dot(from_1 - _S437, _S466);
            float dv_1 = dot(rel_2, _S466);
            float ext_1 = _S445[i_5] + _S436;
            if((abs(dv_1)) < 9.999999960041972e-13f)
            {
                if((abs(o_4)) > ext_1)
                {
                    t_in_1 = _S446;
                }
                i_5 = i_5 + 1U;
                continue;
            }
            float ta_2 = (- ext_1 - o_4) / dv_1;
            float tb_2 = (ext_1 - o_4) / dv_1;
            t_in_1 = max(t_in_1, min(ta_2, tb_2));
            t_out_1 = min(t_out_1, max(ta_2, tb_2));
            i_5 = i_5 + 1U;
        }
    }
    uint _S467 = a_strike_0(&kernelContext_64);
    uint at_7 = _S467 + 9U * _S432;
    if(t_in_1 <= t_out_1)
    {
        uint _S468 = atomic_fetch_min_explicit((&kernelContext_64)->atom_0+(at_7 + 6U), (as_type<uint>((t_in_1))), memory_order_relaxed);
    }
    float3 r_7 = _S437 - (from_1 + rel_2 * float3(_S435.w) );
    float along_0 = dot(r_7, d_2);
    float dist_0 = length(r_7 - d_2 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f;
    uint _S469 = stage_0(k_12, &kernelContext_64);
    *((&kernelContext_64)->frames_0+_S469) = packed_float4(float4(dist_0, 0.0f, 0.0f, 0.0f)) ;
    uint _S470 = atomic_fetch_min_explicit((&kernelContext_64)->atom_0+(at_7 + 7U), (as_type<uint>((dist_0))), memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], atomic<uint> device* atom_27 [[buffer(5)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_65;
    (&kernelContext_65)->params_0 = params_27;
    (&kernelContext_65)->work_0 = work_27;
    (&kernelContext_65)->frames_0 = frames_27;
    (&kernelContext_65)->bodies_0 = bodies_27;
    (&kernelContext_65)->chunks_0 = chunks_27;
    (&kernelContext_65)->sort_0 = sort_27;
    (&kernelContext_65)->atom_0 = atom_27;
    (&kernelContext_65)->manifolds_0 = manifolds_27;
    (&kernelContext_65)->loads_0 = loads_27;
    (&kernelContext_65)->bonds_0 = bonds_27;
    (&kernelContext_65)->args_0 = args_27;
    (&kernelContext_65)->info_0 = info_27;
    (&kernelContext_65)->scan_0 = scan_27;
    (&kernelContext_65)->parts_0 = parts_27;
    (&kernelContext_65)->colliders_0 = colliders_27;
    (&kernelContext_65)->collider_info_0 = collider_info_27;
    (&kernelContext_65)->view_chunks_0 = view_chunks_27;
    (&kernelContext_65)->view_0 = view_27;
    (&kernelContext_65)->objects_0 = objects_27;
    (&kernelContext_65)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_65)->lines_0 = lines_27;
    (&kernelContext_65)->geometry_0 = geometry_27;
    (&kernelContext_65)->shots_0 = shots_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_65)->g_scan_0 = &g_scan_27;
    uint k_13 = id_24.x;
    bool _S471;
    if(k_13 >= (params_27->chunk_count_0))
    {
        _S471 = true;
    }
    else
    {
        _S471 = ((&kernelContext_65)->params_0->shot_count_0) == 0U;
    }
    if(_S471)
    {
        return;
    }
    uint _S472 = body_of_0(k_13, &kernelContext_65);
    thread uint shot_2;
    bool _S473 = struck_0(_S472, &shot_2, &kernelContext_65);
    if(!_S473)
    {
        return;
    }
    uint _S474 = stage_0(k_13, &kernelContext_65);
    float dist_1 = (float4(*((&kernelContext_65)->frames_0+_S474)) ).x;
    uint _S475 = a_strike_0(&kernelContext_65);
    uint at_8 = _S475 + 9U * _S472;
    uint _S476 = (as_type<uint>((dist_1)));
    uint _S477 = atomic_load_explicit((&kernelContext_65)->atom_0+(at_8 + 7U), memory_order_relaxed);
    if(_S476 == _S477)
    {
        uint _S478 = atomic_fetch_min_explicit((&kernelContext_65)->atom_0+(at_8 + 8U), k_13, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], atomic<uint> device* atom_28 [[buffer(5)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_66;
    (&kernelContext_66)->params_0 = params_28;
    (&kernelContext_66)->work_0 = work_28;
    (&kernelContext_66)->frames_0 = frames_28;
    (&kernelContext_66)->bodies_0 = bodies_28;
    (&kernelContext_66)->chunks_0 = chunks_28;
    (&kernelContext_66)->sort_0 = sort_28;
    (&kernelContext_66)->atom_0 = atom_28;
    (&kernelContext_66)->manifolds_0 = manifolds_28;
    (&kernelContext_66)->loads_0 = loads_28;
    (&kernelContext_66)->bonds_0 = bonds_28;
    (&kernelContext_66)->args_0 = args_28;
    (&kernelContext_66)->info_0 = info_28;
    (&kernelContext_66)->scan_0 = scan_28;
    (&kernelContext_66)->parts_0 = parts_28;
    (&kernelContext_66)->colliders_0 = colliders_28;
    (&kernelContext_66)->collider_info_0 = collider_info_28;
    (&kernelContext_66)->view_chunks_0 = view_chunks_28;
    (&kernelContext_66)->view_0 = view_28;
    (&kernelContext_66)->objects_0 = objects_28;
    (&kernelContext_66)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_66)->lines_0 = lines_28;
    (&kernelContext_66)->geometry_0 = geometry_28;
    (&kernelContext_66)->shots_0 = shots_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_66)->g_scan_0 = &g_scan_28;
    uint b_10 = id_25.x;
    bool _S479;
    if(b_10 >= (params_28->body_count_0))
    {
        _S479 = true;
    }
    else
    {
        _S479 = ((&kernelContext_66)->params_0->shot_count_0) == 0U;
    }
    if(_S479)
    {
        return;
    }
    thread uint shot_3;
    bool _S480 = struck_0(b_10, &shot_3, &kernelContext_66);
    if(!_S480)
    {
        return;
    }
    uint _S481 = a_strike_0(&kernelContext_66);
    uint at_9 = _S481 + 9U * b_10;
    uint _S482 = atomic_load_explicit((&kernelContext_66)->atom_0+(at_9 + 6U), memory_order_relaxed);
    float t_in_2 = (as_type<float>((_S482)));
    uint entry_0 = atomic_load_explicit((&kernelContext_66)->atom_0+(at_9 + 8U), memory_order_relaxed);
    uint _S483 = strike_at_0(b_10, &kernelContext_66);
    if(!(t_in_2 <= ((&kernelContext_66)->params_0->dt_0)))
    {
        _S479 = true;
    }
    else
    {
        _S479 = entry_0 == 4294967295U;
    }
    if(_S479)
    {
        (*((&kernelContext_66)->frames_0+_S483))[int(3)] = 0.0f;
        return;
    }
    float4 _S484 = float4(*((&kernelContext_66)->frames_0+_S483)) ;
    float3 rel_3 = (float4(*((&kernelContext_66)->frames_0+(_S483 + 1U))) ).xyz;
    float speed_0 = length(rel_3);
    float3 d_3 = rel_3 / float3(max(speed_0, 9.999999960041972e-13f)) ;
    float4 _S485 = float4(*((&kernelContext_66)->shots_0+4U * shot_3)) ;
    float _S486 = _S485.y;
    bool _S487 = shot_is_box_0(shot_3, &kernelContext_66);
    float radius_0;
    if(_S487)
    {
        thread float3 c_17;
        thread float3 vs_3;
        thread Quat_0 qs_3;
        shot_motion_0(shot_3, &c_17, &vs_3, &qs_3, &kernelContext_66);
        float3 _S488 = (float4(*((&kernelContext_66)->shots_0+(4U * shot_3 + 1U))) ).yzw;
        thread Quat_0 _S489 = qs_3;
        float _S490 = box_extent_0(&_S489, _S488, d_3);
        radius_0 = _S490;
    }
    else
    {
        radius_0 = _S486;
    }
    float _S491 = max(_S485.z, 0.00100000004749745f);
    float m_imp_0 = _S485.w;
    float e_imp_0 = (float4(*((&kernelContext_66)->shots_0+(4U * shot_3 + 1U))) ).x;
    float3 contact_0 = _S484.xyz + rel_3 * float3(t_in_2)  + d_3 * float3(radius_0) ;
    float inv_mass_3 = (float4(*((&kernelContext_66)->bodies_0+7U * b_10)) ).w;
    bool anchored_1 = inv_mass_3 == 0.0f;
    float m_body_1;
    if(anchored_1)
    {
        m_body_1 = 0.0f;
    }
    else
    {
        m_body_1 = 1.0f / inv_mass_3;
    }
    float m_eff_0;
    if(anchored_1)
    {
        m_eff_0 = m_imp_0;
    }
    else
    {
        m_eff_0 = m_imp_0 * m_body_1 / (m_imp_0 + m_body_1);
    }
    float modulus_0 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_66)->geometry_0+(3U * entry_0 + 1U))) ).w);
    *((&kernelContext_66)->frames_0+(_S483 + 2U)) = packed_float4(float4(contact_0, m_eff_0 * speed_0 / (2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (_S491 * modulus_0 * modulus_0 * max(speed_0, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

[[kernel]] void fr_strike_breaks(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], atomic<uint> device* atom_29 [[buffer(5)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_67;
    (&kernelContext_67)->params_0 = params_29;
    (&kernelContext_67)->work_0 = work_29;
    (&kernelContext_67)->frames_0 = frames_29;
    (&kernelContext_67)->bodies_0 = bodies_29;
    (&kernelContext_67)->chunks_0 = chunks_29;
    (&kernelContext_67)->sort_0 = sort_29;
    (&kernelContext_67)->atom_0 = atom_29;
    (&kernelContext_67)->manifolds_0 = manifolds_29;
    (&kernelContext_67)->loads_0 = loads_29;
    (&kernelContext_67)->bonds_0 = bonds_29;
    (&kernelContext_67)->args_0 = args_29;
    (&kernelContext_67)->info_0 = info_29;
    (&kernelContext_67)->scan_0 = scan_29;
    (&kernelContext_67)->parts_0 = parts_29;
    (&kernelContext_67)->colliders_0 = colliders_29;
    (&kernelContext_67)->collider_info_0 = collider_info_29;
    (&kernelContext_67)->view_chunks_0 = view_chunks_29;
    (&kernelContext_67)->view_0 = view_29;
    (&kernelContext_67)->objects_0 = objects_29;
    (&kernelContext_67)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_67)->lines_0 = lines_29;
    (&kernelContext_67)->geometry_0 = geometry_29;
    (&kernelContext_67)->shots_0 = shots_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_67)->g_scan_0 = &g_scan_29;
    uint i_6 = id_26.x;
    bool _S492;
    if(i_6 >= (params_29->bond_count_0))
    {
        _S492 = true;
    }
    else
    {
        _S492 = ((&kernelContext_67)->params_0->shot_count_0) == 0U;
    }
    if(_S492)
    {
        _S492 = true;
    }
    else
    {
        uint _S493 = w_alive_0(&kernelContext_67);
        _S492 = (*((&kernelContext_67)->work_0+(_S493 + i_6))) == 0U;
    }
    if(_S492)
    {
        return;
    }
    uint4 _S494 = uint4(*((&kernelContext_67)->bonds_0+i_6)) ;
    uint _S495 = _S494.x;
    uint _S496 = body_of_0(_S495, &kernelContext_67);
    thread uint shot_4;
    bool _S497 = struck_0(_S496, &shot_4, &kernelContext_67);
    if(!_S497)
    {
        return;
    }
    uint _S498 = strike_at_0(_S496, &kernelContext_67);
    float4 _S499 = float4(*((&kernelContext_67)->frames_0+(_S498 + 2U))) ;
    thread Quat_0 q_23;
    float3 _S500 = chunk_box_world_0(_S495, &q_23, &kernelContext_67);
    float3 _S501 = chunk_box_world_0(_S494.y, &q_23, &kernelContext_67);
    float dist_2 = length(float3(0.5f)  * (_S500 + _S501) - _S499.xyz);
    if((_S499.w * min(1.0f, (as_type<float>((_S494.w))) / (6.28318548202514648f * max(dist_2 * dist_2, 9.999999960041972e-13f)))) > (as_type<float>((_S494.z))))
    {
        uint _S502 = w_alive_0(&kernelContext_67);
        *((&kernelContext_67)->work_0+(_S502 + i_6)) = 0U;
        uint _S503 = a_dirty_0(&kernelContext_67);
        uint _S504 = atomic_fetch_or_explicit((&kernelContext_67)->atom_0+(_S503 + _S496), 1U, memory_order_relaxed);
        uint _S505 = a_counters_0(&kernelContext_67);
        uint _S506 = atomic_fetch_add_explicit((&kernelContext_67)->atom_0+_S505, 1U, memory_order_relaxed);
        uint _S507 = atomic_fetch_add_explicit((&kernelContext_67)->atom_0+(_S505 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

