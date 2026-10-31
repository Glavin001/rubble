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
    uint chunk_used_0;
    uint bond_used_0;
    uint pad4_0;
    uint pad5_0;
    uint pad6_0;
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
    float pad3_0;
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

bool live_0(uint c_1, KernelContext_0 thread* kernelContext_1)
{
    bool _S1;
    if(c_1 < (kernelContext_1->params_0->chunk_used_0))
    {
        uint _S2 = body_of_0(c_1, kernelContext_1);
        _S1 = _S2 != 4294967295U;
    }
    else
    {
        _S1 = false;
    }
    return _S1;
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
    thread Quat_0 c_2;
    (&c_2)->w_0 = q_1->w_0;
    (&c_2)->x_0 = - q_1->x_0;
    (&c_2)->y_0 = - q_1->y_0;
    (&c_2)->z_0 = - q_1->z_0;
    return c_2;
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
    thread Quat_0 _S3 = axes_0;
    Quat_0 _S4 = quat_mul_0(&_S3, q_3);
    thread Quat_0 _S5 = _S4;
    Quat_0 _S6 = normalized_0(&_S5);
    return _S6;
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

void structure_pose_0(uint slot_0, uint b_0, Quat_0 thread* q_5, float3 thread* origin_0, KernelContext_0 thread* kernelContext_2)
{
    uint _S7 = 2U * slot_0;
    thread Quat_0 _S8 = quat_of_0(float4(*(kernelContext_2->frames_0+(_S7 + 1U))) );
    Quat_0 _S9 = conj_0(&_S8);
    float3 com_0 = (float4(*(kernelContext_2->frames_0+_S7)) ).xyz;
    thread Quat_0 _S10 = quat_of_0(float4(*(kernelContext_2->bodies_0+(b_0 + 1U))) );
    Quat_0 _S11 = rot_from_rubble_0(&_S10);
    thread Quat_0 _S12 = _S11;
    thread Quat_0 _S13 = _S9;
    Quat_0 _S14 = quat_mul_0(&_S12, &_S13);
    thread Quat_0 _S15 = _S14;
    Quat_0 _S16 = normalized_0(&_S15);
    *q_5 = _S16;
    float3 _S17 = from_rubble_0((float4(*(kernelContext_2->bodies_0+b_0)) ).xyz);
    thread Quat_0 _S18 = _S16;
    float3 _S19 = rotate_0(&_S18, com_0);
    *origin_0 = _S17 - _S19;
    return;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

uint w_alive_0(KernelContext_0 thread* kernelContext_3)
{
    return kernelContext_3->params_0->chunk_count_0;
}

uint w_label_0(KernelContext_0 thread* kernelContext_4)
{
    uint _S20 = w_alive_0(kernelContext_4);
    return _S20 + kernelContext_4->params_0->bond_count_0;
}

uint w_slot_0(KernelContext_0 thread* kernelContext_5)
{
    uint _S21 = w_label_0(kernelContext_5);
    return _S21 + kernelContext_5->params_0->chunk_count_0;
}

uint w_new_0(KernelContext_0 thread* kernelContext_6)
{
    uint _S22 = w_slot_0(kernelContext_6);
    return _S22 + kernelContext_6->params_0->chunk_count_0;
}

uint w_rank_0(KernelContext_0 thread* kernelContext_7)
{
    uint _S23 = w_new_0(kernelContext_7);
    return _S23 + kernelContext_7->params_0->chunk_count_0;
}

uint w_free_0(KernelContext_0 thread* kernelContext_8)
{
    uint _S24 = w_rank_0(kernelContext_8);
    return _S24 + kernelContext_8->params_0->chunk_count_0;
}

uint w_free_rank_0(KernelContext_0 thread* kernelContext_9)
{
    uint _S25 = w_free_0(kernelContext_9);
    return _S25 + kernelContext_9->params_0->body_count_0;
}

uint w_free_list_0(KernelContext_0 thread* kernelContext_10)
{
    uint _S26 = w_free_rank_0(kernelContext_10);
    return _S26 + kernelContext_10->params_0->body_count_0;
}

uint w_pulses_0(KernelContext_0 thread* kernelContext_11)
{
    uint _S27 = w_free_list_0(kernelContext_11);
    return _S27 + kernelContext_11->params_0->body_count_0 + 2050U;
}

uint a_sums_0(KernelContext_0 thread* kernelContext_12)
{
    return 3U * kernelContext_12->params_0->chunk_count_0;
}

uint a_dirty_0(KernelContext_0 thread* kernelContext_13)
{
    uint _S28 = a_sums_0(kernelContext_13);
    return _S28 + 20U * kernelContext_13->params_0->chunk_count_0;
}

uint a_best_count_0(KernelContext_0 thread* kernelContext_14)
{
    uint _S29 = a_dirty_0(kernelContext_14);
    return _S29 + kernelContext_14->params_0->body_count_0;
}

uint a_best_root_0(KernelContext_0 thread* kernelContext_15)
{
    uint _S30 = a_best_count_0(kernelContext_15);
    return _S30 + kernelContext_15->params_0->body_count_0;
}

uint a_counters_0(KernelContext_0 thread* kernelContext_16)
{
    uint _S31 = a_best_root_0(kernelContext_16);
    return _S31 + kernelContext_16->params_0->body_count_0;
}

uint a_counters_1(KernelContext_0 thread* kernelContext_17)
{
    uint _S32 = a_best_root_0(kernelContext_17);
    return _S32 + kernelContext_17->params_0->body_count_0;
}

uint l_pool_0(uint half_0, KernelContext_0 thread* kernelContext_18)
{
    return kernelContext_18->params_0->chunk_count_0 + 3U * kernelContext_18->params_0->pool_cap_0 * half_0;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S33 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S33, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S34 = max(0.0f, _S33);
    float _S35 = min(dt_1, _S33 + duration_0);
    bool _S36;
    if(_S35 > _S34)
    {
        _S36 = duration_0 > 0.0f;
    }
    else
    {
        _S36 = false;
    }
    float sum_1;
    if(_S36)
    {
        float w_1 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_1 * (cos(w_1 * (_S34 - _S33)) - cos(w_1 * (_S35 - _S33)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

float hertz_duration_0(float m_0, float r_3, float e_0, float v_2)
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_3 * e_0 * e_0 * max(abs(v_2), 0.00100000004749745f)), 0.20000000298023224f);
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
    thread KernelContext_0 kernelContext_19;
    (&kernelContext_19)->params_0 = params_1;
    (&kernelContext_19)->work_0 = work_1;
    (&kernelContext_19)->frames_0 = frames_1;
    (&kernelContext_19)->bodies_0 = bodies_1;
    (&kernelContext_19)->chunks_0 = chunks_1;
    (&kernelContext_19)->sort_0 = sort_1;
    (&kernelContext_19)->atom_0 = atom_1;
    (&kernelContext_19)->manifolds_0 = manifolds_1;
    (&kernelContext_19)->loads_0 = loads_1;
    (&kernelContext_19)->bonds_0 = bonds_1;
    (&kernelContext_19)->args_0 = args_1;
    (&kernelContext_19)->info_0 = info_1;
    (&kernelContext_19)->scan_0 = scan_1;
    (&kernelContext_19)->parts_0 = parts_1;
    (&kernelContext_19)->colliders_0 = colliders_1;
    (&kernelContext_19)->collider_info_0 = collider_info_1;
    (&kernelContext_19)->view_chunks_0 = view_chunks_1;
    (&kernelContext_19)->view_0 = view_1;
    (&kernelContext_19)->objects_0 = objects_1;
    (&kernelContext_19)->shot_bodies_0 = shot_bodies_1;
    (&kernelContext_19)->lines_0 = lines_1;
    (&kernelContext_19)->geometry_0 = geometry_1;
    (&kernelContext_19)->shots_0 = shots_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_19)->g_scan_0 = &g_scan_1;
    uint k_0 = id_0.x;
    bool _S37 = live_0(k_0, &kernelContext_19);
    if(!_S37)
    {
        return;
    }
    uint _S38 = body_of_0(k_0, &kernelContext_19);
    uint b_1 = 7U * _S38;
    thread Quat_0 q_6;
    thread float3 origin_1;
    structure_pose_0(_S38, b_1, &q_6, &origin_1, &kernelContext_19);
    uint _S39 = 4U * k_0;
    float4 _S40 = float4(*((&kernelContext_19)->chunks_0+_S39)) ;
    float3 _S41 = _S40.xyz;
    thread Quat_0 _S42 = q_6;
    float3 _S43 = rotate_0(&_S42, _S41);
    float3 centre_0 = origin_1 + _S43;
    float inv_mass_0 = (float4(*((&kernelContext_19)->bodies_0+b_1)) ).w;
    float m_body_0;
    if(inv_mass_0 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_0;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint from_0 = (&kernelContext_19)->params_0->parity_0;
    uint to_0 = 1U - (&kernelContext_19)->params_0->parity_0;
    uint _S44 = w_pulses_0(&kernelContext_19);
    uint _S45 = _S44 + _S39;
    uint _S46 = _S45 + 2U * from_0;
    uint off_0 = *((&kernelContext_19)->work_0+_S46);
    uint device* _S47 = (&kernelContext_19)->work_0+(_S46 + 1U);
    uint carried_0 = *_S47;
    uint room_0 = *_S47 + (&kernelContext_19)->sort_0[(&kernelContext_19)->params_0->chunk_count_0 + 1U + k_0];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S48 = a_counters_0(&kernelContext_19);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_19)->atom_0+(_S48 + 5U + to_0), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_19)->params_0->pool_cap_0))
        {
            uint _S49 = atomic_fetch_or_explicit((&kernelContext_19)->atom_0+(_S48 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_19)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_19)->params_0->pool_cap_0 - base_1;
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
        uint _S50 = base_0;
        base_0 = base_1;
        room_1 = _S50;
    }
    else
    {
        base_0 = 0U;
        room_1 = room_0;
    }
    uint _S51 = (&kernelContext_19)->params_0->chunk_count_0;
    float _S52 = _S40.w;
    float4 _S53 = float4(*((&kernelContext_19)->chunks_0+(_S39 + 3U))) ;
    float _S54 = (&kernelContext_19)->params_0->band_0;
    float _S55 = (&kernelContext_19)->params_0->dt_0;
    bool _S56 = ((&kernelContext_19)->params_0->impact_peak_0) != 0U;
    uint _S57 = l_pool_0(from_0, &kernelContext_19);
    uint _S58 = _S57 + 3U * off_0;
    uint _S59 = l_pool_0(to_0, &kernelContext_19);
    uint _S60 = _S59 + 3U * base_0;
    for(;;)
    {
        float3 _S61 = float3(0.0f) ;
        uint _S62 = (&kernelContext_19)->sort_0[k_0];
        uint _S63 = (&kernelContext_19)->sort_0[_S51 + 1U + k_0];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_1 = 0U;
        f_total_0 = _S61;
        float3 t_total_1 = _S61;
        for(;;)
        {
            if(step_1 < _S63)
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
                if(q3_0 < _S63)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_19)->manifolds_0[_S62 + q3_0].ids_0) ).y + 1U;
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
            Record_natural_0 r_4 = (&kernelContext_19)->manifolds_0[_S62 + p_0];
            float4 _S64 = float4(r_4.impulse_0) ;
            float3 _S65 = _S64.xyz;
            float j_0 = length(_S65);
            if(j_0 == 0.0f)
            {
                uint _S66 = step_1 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S66;
                continue;
            }
            float m_other_0 = (float4(r_4.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S52;
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
            float _S67 = min(j_0, 2.0f * m_1 * max(_S64.w - _S54, 0.0f));
            if(_S67 < j_0)
            {
                float3 f_0 = _S65 / float3(j_0)  * float3(((j_0 - _S67) / _S55)) ;
                float3 _S68 = t_total_1 + cross((float4(r_4.point_0) ).xyz - centre_0, f_0);
                f_total_1 = f_total_0 + f_0;
                t_total_0 = _S68;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            uint _S66 = step_1 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S66;
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
            uint _S69 = _S58 + 3U * p_0;
            float4 _S70 = float4(*((&kernelContext_19)->loads_0+_S69)) ;
            thread float4 a_1 = _S70;
            float4 _S71 = float4(*((&kernelContext_19)->loads_0+(_S69 + 1U))) ;
            float4 _S72 = float4(*((&kernelContext_19)->loads_0+(_S69 + 2U))) ;
            float _S73 = _S71.w;
            float mean_0 = pulse_mean_0(_S70.x, _S70.y, _S70.z, _S70.w, _S73, _S55);
            if(mean_0 != 0.0f)
            {
                float3 f_1 = _S72.xyz * float3(mean_0) ;
                float3 _S74 = t_total_1 + cross(_S71.xyz - centre_0, f_1);
                f_total_0 = f_total_0 + f_1;
                t_total_1 = _S74;
            }
            if((a_1.x + a_1.z + _S73) > _S55)
            {
                a_1.x = a_1.x - _S55;
                uint _S75 = _S60 + 3U * kept_0;
                *((&kernelContext_19)->loads_0+_S75) = packed_float4(a_1) ;
                *((&kernelContext_19)->loads_0+(_S75 + 1U)) = packed_float4(_S71) ;
                *((&kernelContext_19)->loads_0+(_S75 + 2U)) = packed_float4(_S72) ;
                kept_0 = kept_0 + 1U;
            }
            p_0 = p_0 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S63)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_5 = (&kernelContext_19)->manifolds_0[_S62 + q3_0];
            float4 _S76 = float4(r_5.impulse_0) ;
            float3 _S77 = _S76.xyz;
            float j_1 = length(_S77);
            if(j_1 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S78 = float4(r_5.other_0) ;
            float m_other_1 = _S78.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S52;
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
            float _S79 = _S76.w;
            float _S80 = min(j_1, 2.0f * m_1 * max(_S79 - _S54, 0.0f));
            if(_S80 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S81 = _S53.y;
            float duration_1 = hertz_duration_0(m_1, min(_S78.z, _S53.z), 1.0f / ((1.0f - _S81 * _S81) / _S53.x + 1.0f / _S78.y), _S79);
            float peak_1 = 3.14159274101257324f * _S80 / (2.0f * duration_1);
            float3 dir_0 = _S77 / float3(j_1) ;
            float mean_1;
            if(_S56)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S55);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_2 = dir_0 * float3(mean_1) ;
                float3 _S82 = t_total_1 + cross((float4(r_5.point_0) ).xyz - centre_0, f_2);
                f_total_1 = f_total_0 + f_2;
                t_total_0 = _S82;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            uint kept_1;
            if(duration_1 > _S55)
            {
                if(kept_0 < room_1)
                {
                    uint _S83 = _S60 + 3U * kept_0;
                    *((&kernelContext_19)->loads_0+_S83) = packed_float4(float4(- _S55, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_19)->loads_0+(_S83 + 1U)) = packed_float4(float4((float4(r_5.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_19)->loads_0+(_S83 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
        uint _S84 = a_counters_0(&kernelContext_19);
        uint _S85 = atomic_fetch_or_explicit((&kernelContext_19)->atom_0+(_S84 + 3U), 8U, memory_order_relaxed);
    }
    uint _S86 = _S45 + 2U * to_0;
    *((&kernelContext_19)->work_0+_S86) = base_0;
    *((&kernelContext_19)->work_0+(_S86 + 1U)) = kept_0;
    *((&kernelContext_19)->loads_0+k_0) = packed_float4(float4(length(f_total_0), 0.0f, 0.0f, 0.0f)) ;
    return;
}

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], atomic<uint> device* atom_2 [[buffer(5)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]], packed_float4 device* geometry_2 [[buffer(21)]], packed_float4 device* shots_2 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_20;
    (&kernelContext_20)->params_0 = params_2;
    (&kernelContext_20)->work_0 = work_2;
    (&kernelContext_20)->frames_0 = frames_2;
    (&kernelContext_20)->bodies_0 = bodies_2;
    (&kernelContext_20)->chunks_0 = chunks_2;
    (&kernelContext_20)->sort_0 = sort_2;
    (&kernelContext_20)->atom_0 = atom_2;
    (&kernelContext_20)->manifolds_0 = manifolds_2;
    (&kernelContext_20)->loads_0 = loads_2;
    (&kernelContext_20)->bonds_0 = bonds_2;
    (&kernelContext_20)->args_0 = args_2;
    (&kernelContext_20)->info_0 = info_2;
    (&kernelContext_20)->scan_0 = scan_2;
    (&kernelContext_20)->parts_0 = parts_2;
    (&kernelContext_20)->colliders_0 = colliders_2;
    (&kernelContext_20)->collider_info_0 = collider_info_2;
    (&kernelContext_20)->view_chunks_0 = view_chunks_2;
    (&kernelContext_20)->view_0 = view_2;
    (&kernelContext_20)->objects_0 = objects_2;
    (&kernelContext_20)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_20)->lines_0 = lines_2;
    (&kernelContext_20)->geometry_0 = geometry_2;
    (&kernelContext_20)->shots_0 = shots_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_20)->g_scan_0 = &g_scan_2;
    uint i_0 = id_1.x;
    bool _S87;
    if(i_0 >= (params_2->bond_used_0))
    {
        _S87 = true;
    }
    else
    {
        uint _S88 = w_alive_0(&kernelContext_20);
        _S87 = (*((&kernelContext_20)->work_0+(_S88 + i_0))) == 0U;
    }
    if(_S87)
    {
        return;
    }
    uint4 _S89 = uint4(*((&kernelContext_20)->bonds_0+i_0)) ;
    uint _S90 = _S89.x;
    if((max((float4(*((&kernelContext_20)->loads_0+_S90)) ).x, (float4(*((&kernelContext_20)->loads_0+_S89.y)) ).x)) > (as_type<float>((_S89.z))))
    {
        uint _S91 = w_alive_0(&kernelContext_20);
        *((&kernelContext_20)->work_0+(_S91 + i_0)) = 0U;
        uint _S92 = a_dirty_0(&kernelContext_20);
        uint _S93 = body_of_0(_S90, &kernelContext_20);
        uint _S94 = atomic_fetch_or_explicit((&kernelContext_20)->atom_0+(_S92 + _S93), 1U, memory_order_relaxed);
        uint _S95 = a_counters_0(&kernelContext_20);
        uint _S96 = atomic_fetch_add_explicit((&kernelContext_20)->atom_0+_S95, 1U, memory_order_relaxed);
        uint _S97 = atomic_fetch_add_explicit((&kernelContext_20)->atom_0+(_S95 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint groups_of_0(uint n_2, uint size_0)
{
    uint _S98 = (n_2 + size_0 - 1U) / size_0;
    return _S98;
}

void set_args_0(uint at_0, uint x_1, KernelContext_0 thread* kernelContext_21)
{
    uint _S99 = 3U * at_0;
    *(kernelContext_21->args_0+_S99) = x_1;
    *(kernelContext_21->args_0+(_S99 + 1U)) = 1U;
    *(kernelContext_21->args_0+(_S99 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_22;
    (&kernelContext_22)->params_0 = params_3;
    (&kernelContext_22)->work_0 = work_3;
    (&kernelContext_22)->frames_0 = frames_3;
    (&kernelContext_22)->bodies_0 = bodies_3;
    (&kernelContext_22)->chunks_0 = chunks_3;
    (&kernelContext_22)->sort_0 = sort_3;
    (&kernelContext_22)->atom_0 = atom_3;
    (&kernelContext_22)->manifolds_0 = manifolds_3;
    (&kernelContext_22)->loads_0 = loads_3;
    (&kernelContext_22)->bonds_0 = bonds_3;
    (&kernelContext_22)->args_0 = args_3;
    (&kernelContext_22)->info_0 = info_3;
    (&kernelContext_22)->scan_0 = scan_3;
    (&kernelContext_22)->parts_0 = parts_3;
    (&kernelContext_22)->colliders_0 = colliders_3;
    (&kernelContext_22)->collider_info_0 = collider_info_3;
    (&kernelContext_22)->view_chunks_0 = view_chunks_3;
    (&kernelContext_22)->view_0 = view_3;
    (&kernelContext_22)->objects_0 = objects_3;
    (&kernelContext_22)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_22)->lines_0 = lines_3;
    (&kernelContext_22)->geometry_0 = geometry_3;
    (&kernelContext_22)->shots_0 = shots_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_22)->g_scan_0 = &g_scan_3;
    uint _S100 = a_counters_0(&kernelContext_22);
    uint _S101 = atomic_load_explicit((&kernelContext_22)->atom_0+_S100, memory_order_relaxed);
    bool run_0 = _S101 != 0U;
    atomic_store_explicit((&kernelContext_22)->atom_0+_S100, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_22)->atom_0+(_S100 + 5U + (&kernelContext_22)->params_0->parity_0), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S102 = atomic_fetch_add_explicit((&kernelContext_22)->atom_0+(_S100 + 4U), 1U, memory_order_relaxed);
    }
    uint _S103;
    if(run_0)
    {
        uint _S104 = groups_of_0((&kernelContext_22)->params_0->chunk_used_0, 64U);
        _S103 = _S104;
    }
    else
    {
        _S103 = 0U;
    }
    set_args_0(0U, _S103, &kernelContext_22);
    if(run_0)
    {
        uint _S105 = groups_of_0((&kernelContext_22)->params_0->bond_used_0, 64U);
        _S103 = _S105;
    }
    else
    {
        _S103 = 0U;
    }
    set_args_0(1U, _S103, &kernelContext_22);
    if(run_0)
    {
        uint _S106 = groups_of_0((&kernelContext_22)->params_0->body_count_0, 64U);
        _S103 = _S106;
    }
    else
    {
        _S103 = 0U;
    }
    set_args_0(2U, _S103, &kernelContext_22);
    if(run_0)
    {
        uint _S107 = groups_of_0((&kernelContext_22)->params_0->chunk_used_0, 1024U);
        _S103 = _S107;
    }
    else
    {
        _S103 = 0U;
    }
    set_args_0(3U, _S103, &kernelContext_22);
    if(run_0)
    {
        uint _S108 = groups_of_0((&kernelContext_22)->params_0->body_count_0, 1024U);
        _S103 = _S108;
    }
    else
    {
        _S103 = 0U;
    }
    set_args_0(4U, _S103, &kernelContext_22);
    int _S109;
    if(run_0)
    {
        _S109 = int(1);
    }
    else
    {
        _S109 = int(0);
    }
    set_args_0(5U, uint(_S109), &kernelContext_22);
    return;
}

bool dirty_0(uint body_0, KernelContext_0 thread* kernelContext_23)
{
    uint _S110 = a_dirty_0(kernelContext_23);
    uint _S111 = atomic_load_explicit(kernelContext_23->atom_0+(_S110 + body_0), memory_order_relaxed);
    return _S111 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_24)
{
    return kernelContext_24->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_25)
{
    return 2U * kernelContext_25->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_26;
    (&kernelContext_26)->params_0 = params_4;
    (&kernelContext_26)->work_0 = work_4;
    (&kernelContext_26)->frames_0 = frames_4;
    (&kernelContext_26)->bodies_0 = bodies_4;
    (&kernelContext_26)->chunks_0 = chunks_4;
    (&kernelContext_26)->sort_0 = sort_4;
    (&kernelContext_26)->atom_0 = atom_4;
    (&kernelContext_26)->manifolds_0 = manifolds_4;
    (&kernelContext_26)->loads_0 = loads_4;
    (&kernelContext_26)->bonds_0 = bonds_4;
    (&kernelContext_26)->args_0 = args_4;
    (&kernelContext_26)->info_0 = info_4;
    (&kernelContext_26)->scan_0 = scan_4;
    (&kernelContext_26)->parts_0 = parts_4;
    (&kernelContext_26)->colliders_0 = colliders_4;
    (&kernelContext_26)->collider_info_0 = collider_info_4;
    (&kernelContext_26)->view_chunks_0 = view_chunks_4;
    (&kernelContext_26)->view_0 = view_4;
    (&kernelContext_26)->objects_0 = objects_4;
    (&kernelContext_26)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_26)->lines_0 = lines_4;
    (&kernelContext_26)->geometry_0 = geometry_4;
    (&kernelContext_26)->shots_0 = shots_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_26)->g_scan_0 = &g_scan_4;
    uint c_3 = id_3.x;
    bool _S112 = live_0(c_3, &kernelContext_26);
    bool _S113;
    if(!_S112)
    {
        _S113 = true;
    }
    else
    {
        uint _S114 = body_of_0(c_3, &kernelContext_26);
        bool _S115 = dirty_0(_S114, &kernelContext_26);
        _S113 = !_S115;
    }
    if(_S113)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_26)->atom_0+(a_parent_0() + c_3), c_3, memory_order_relaxed);
    uint _S116 = a_count_0(&kernelContext_26);
    atomic_store_explicit((&kernelContext_26)->atom_0+(_S116 + c_3), 0U, memory_order_relaxed);
    uint _S117 = a_anchor_0(&kernelContext_26);
    atomic_store_explicit((&kernelContext_26)->atom_0+(_S117 + c_3), 0U, memory_order_relaxed);
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
        uint _S118 = a_sums_0(&kernelContext_26);
        atomic_store_explicit((&kernelContext_26)->atom_0+(_S118 + 20U * c_3 + w_2), 0U, memory_order_relaxed);
        w_2 = w_2 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_27;
    (&kernelContext_27)->params_0 = params_5;
    (&kernelContext_27)->work_0 = work_5;
    (&kernelContext_27)->frames_0 = frames_5;
    (&kernelContext_27)->bodies_0 = bodies_5;
    (&kernelContext_27)->chunks_0 = chunks_5;
    (&kernelContext_27)->sort_0 = sort_5;
    (&kernelContext_27)->atom_0 = atom_5;
    (&kernelContext_27)->manifolds_0 = manifolds_5;
    (&kernelContext_27)->loads_0 = loads_5;
    (&kernelContext_27)->bonds_0 = bonds_5;
    (&kernelContext_27)->args_0 = args_5;
    (&kernelContext_27)->info_0 = info_5;
    (&kernelContext_27)->scan_0 = scan_5;
    (&kernelContext_27)->parts_0 = parts_5;
    (&kernelContext_27)->colliders_0 = colliders_5;
    (&kernelContext_27)->collider_info_0 = collider_info_5;
    (&kernelContext_27)->view_chunks_0 = view_chunks_5;
    (&kernelContext_27)->view_0 = view_5;
    (&kernelContext_27)->objects_0 = objects_5;
    (&kernelContext_27)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_27)->lines_0 = lines_5;
    (&kernelContext_27)->geometry_0 = geometry_5;
    (&kernelContext_27)->shots_0 = shots_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_27)->g_scan_0 = &g_scan_5;
    uint s_0 = id_4.x;
    bool _S119;
    if(s_0 >= (params_5->body_count_0))
    {
        _S119 = true;
    }
    else
    {
        bool _S120 = dirty_0(s_0, &kernelContext_27);
        _S119 = !_S120;
    }
    if(_S119)
    {
        return;
    }
    uint _S121 = a_best_count_0(&kernelContext_27);
    atomic_store_explicit((&kernelContext_27)->atom_0+(_S121 + s_0), 0U, memory_order_relaxed);
    uint _S122 = a_best_root_0(&kernelContext_27);
    atomic_store_explicit((&kernelContext_27)->atom_0+(_S122 + s_0), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_2, KernelContext_0 thread* kernelContext_28)
{
    uint _S123 = a_parent_0();
    uint _S124 = atomic_load_explicit(kernelContext_28->atom_0+(_S123 + x_2), memory_order_relaxed);
    uint p_1 = _S124;
    uint _S125 = x_2;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S126;
        if(guard_0 < 1048576U)
        {
            _S126 = p_1 != _S125;
        }
        else
        {
            _S126 = false;
        }
        if(_S126)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_28->atom_0+(_S123 + p_1), memory_order_relaxed);
        if(gp_0 != p_1)
        {
            uint _S127;
            {
            uint _metal_cas_comparand = p_1;
            atomic_compare_exchange_weak_explicit(kernelContext_28->atom_0+(_S123 + _S125), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S127 = _metal_cas_comparand;
            }
        }
        uint _S128 = guard_0 + 1U;
        _S125 = p_1;
        p_1 = gp_0;
        guard_0 = _S128;
    }
    return _S125;
}

[[kernel]] void fr_uf_union(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_29;
    (&kernelContext_29)->params_0 = params_6;
    (&kernelContext_29)->work_0 = work_6;
    (&kernelContext_29)->frames_0 = frames_6;
    (&kernelContext_29)->bodies_0 = bodies_6;
    (&kernelContext_29)->chunks_0 = chunks_6;
    (&kernelContext_29)->sort_0 = sort_6;
    (&kernelContext_29)->atom_0 = atom_6;
    (&kernelContext_29)->manifolds_0 = manifolds_6;
    (&kernelContext_29)->loads_0 = loads_6;
    (&kernelContext_29)->bonds_0 = bonds_6;
    (&kernelContext_29)->args_0 = args_6;
    (&kernelContext_29)->info_0 = info_6;
    (&kernelContext_29)->scan_0 = scan_6;
    (&kernelContext_29)->parts_0 = parts_6;
    (&kernelContext_29)->colliders_0 = colliders_6;
    (&kernelContext_29)->collider_info_0 = collider_info_6;
    (&kernelContext_29)->view_chunks_0 = view_chunks_6;
    (&kernelContext_29)->view_0 = view_6;
    (&kernelContext_29)->objects_0 = objects_6;
    (&kernelContext_29)->shot_bodies_0 = shot_bodies_6;
    (&kernelContext_29)->lines_0 = lines_6;
    (&kernelContext_29)->geometry_0 = geometry_6;
    (&kernelContext_29)->shots_0 = shots_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_29)->g_scan_0 = &g_scan_6;
    uint i_1 = id_5.x;
    bool _S129;
    if(i_1 >= (params_6->bond_used_0))
    {
        _S129 = true;
    }
    else
    {
        uint _S130 = w_alive_0(&kernelContext_29);
        _S129 = (*((&kernelContext_29)->work_0+(_S130 + i_1))) == 0U;
    }
    if(_S129)
    {
        return;
    }
    uint4 _S131 = uint4(*((&kernelContext_29)->bonds_0+i_1)) ;
    uint _S132 = _S131.x;
    uint _S133 = body_of_0(_S132, &kernelContext_29);
    bool _S134 = dirty_0(_S133, &kernelContext_29);
    if(!_S134)
    {
        return;
    }
    uint _S135 = _S131.y;
    uint a_2 = _S132;
    uint b_2 = _S135;
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
        uint _S136 = find_0(a_2, &kernelContext_29);
        uint _S137 = find_0(b_2, &kernelContext_29);
        if(_S136 == _S137)
        {
            return;
        }
        uint _S138 = max(_S136, _S137);
        uint _S139 = min(_S136, _S137);
        uint was_0;
        {
        uint _metal_cas_comparand = _S138;
        atomic_compare_exchange_weak_explicit((&kernelContext_29)->atom_0+(a_parent_0() + _S138), &_metal_cas_comparand, _S139, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S138)
        {
            return;
        }
        uint _S140 = guard_1 + 1U;
        a_2 = _S138;
        b_2 = _S139;
        guard_1 = _S140;
    }
    uint _S141 = a_counters_0(&kernelContext_29);
    uint _S142 = atomic_fetch_or_explicit((&kernelContext_29)->atom_0+(_S141 + 3U), 4U, memory_order_relaxed);
    return;
}

void add64_0(uint at_1, float value_0, float scale_0, KernelContext_0 thread* kernelContext_30)
{
    float x_3 = round(value_0 * scale_0);
    float x_4;
    if(!((abs(x_3)) < 1.073741824e+09f))
    {
        uint _S143 = a_counters_0(kernelContext_30);
        uint _S144 = atomic_fetch_or_explicit(kernelContext_30->atom_0+(_S143 + 3U), 2U, memory_order_relaxed);
        x_4 = 0.0f;
    }
    else
    {
        x_4 = x_3;
    }
    int v_3 = int(x_4);
    uint lo_0 = (as_type<uint>((v_3)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_30->atom_0+at_1, lo_0, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_0) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S145;
    if(v_3 < int(0))
    {
        _S145 = 4294967295U;
    }
    else
    {
        _S145 = 0U;
    }
    uint hi_0 = _S145 + carry_0;
    if(hi_0 != 0U)
    {
        uint _S146 = atomic_fetch_add_explicit(kernelContext_30->atom_0+(at_1 + 1U), hi_0, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_31;
    (&kernelContext_31)->params_0 = params_7;
    (&kernelContext_31)->work_0 = work_7;
    (&kernelContext_31)->frames_0 = frames_7;
    (&kernelContext_31)->bodies_0 = bodies_7;
    (&kernelContext_31)->chunks_0 = chunks_7;
    (&kernelContext_31)->sort_0 = sort_7;
    (&kernelContext_31)->atom_0 = atom_7;
    (&kernelContext_31)->manifolds_0 = manifolds_7;
    (&kernelContext_31)->loads_0 = loads_7;
    (&kernelContext_31)->bonds_0 = bonds_7;
    (&kernelContext_31)->args_0 = args_7;
    (&kernelContext_31)->info_0 = info_7;
    (&kernelContext_31)->scan_0 = scan_7;
    (&kernelContext_31)->parts_0 = parts_7;
    (&kernelContext_31)->colliders_0 = colliders_7;
    (&kernelContext_31)->collider_info_0 = collider_info_7;
    (&kernelContext_31)->view_chunks_0 = view_chunks_7;
    (&kernelContext_31)->view_0 = view_7;
    (&kernelContext_31)->objects_0 = objects_7;
    (&kernelContext_31)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_31)->lines_0 = lines_7;
    (&kernelContext_31)->geometry_0 = geometry_7;
    (&kernelContext_31)->shots_0 = shots_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_31)->g_scan_0 = &g_scan_7;
    uint c_4 = id_6.x;
    bool _S147 = live_0(c_4, &kernelContext_31);
    bool _S148;
    if(!_S147)
    {
        _S148 = true;
    }
    else
    {
        uint _S149 = body_of_0(c_4, &kernelContext_31);
        bool _S150 = dirty_0(_S149, &kernelContext_31);
        _S148 = !_S150;
    }
    if(_S148)
    {
        return;
    }
    uint _S151 = find_0(c_4, &kernelContext_31);
    uint _S152 = w_label_0(&kernelContext_31);
    *((&kernelContext_31)->work_0+(_S152 + c_4)) = _S151;
    uint _S153 = a_count_0(&kernelContext_31);
    uint _S154 = atomic_fetch_add_explicit((&kernelContext_31)->atom_0+(_S153 + _S151), 1U, memory_order_relaxed);
    if(((&kernelContext_31)->info_0[c_4] & 1U) != 0U)
    {
        uint _S155 = a_anchor_0(&kernelContext_31);
        uint _S156 = atomic_fetch_or_explicit((&kernelContext_31)->atom_0+(_S155 + _S151), 1U, memory_order_relaxed);
    }
    uint _S157 = 4U * c_4;
    float4 _S158 = float4(*((&kernelContext_31)->chunks_0+_S157)) ;
    float4 _S159 = float4(*((&kernelContext_31)->chunks_0+(_S157 + 1U))) ;
    float4 _S160 = float4(*((&kernelContext_31)->chunks_0+(_S157 + 2U))) ;
    float m_2 = _S158.w;
    float3 d_0 = _S158.xyz - (float4(*((&kernelContext_31)->chunks_0+4U * _S151)) ).xyz;
    float dd_0 = dot(d_0, d_0);
    uint _S161 = a_sums_0(&kernelContext_31);
    uint at_2 = _S161 + 20U * _S151;
    add64_0(at_2, m_2, (&kernelContext_31)->params_0->scale_m_0, &kernelContext_31);
    float _S162 = d_0.x;
    float _S163 = m_2 * _S162;
    add64_0(at_2 + 2U, _S163, (&kernelContext_31)->params_0->scale_s_0, &kernelContext_31);
    float _S164 = d_0.y;
    float _S165 = m_2 * _S164;
    add64_0(at_2 + 4U, _S165, (&kernelContext_31)->params_0->scale_s_0, &kernelContext_31);
    float _S166 = d_0.z;
    add64_0(at_2 + 6U, m_2 * _S166, (&kernelContext_31)->params_0->scale_s_0, &kernelContext_31);
    add64_0(at_2 + 8U, _S159.x + m_2 * (dd_0 - _S162 * _S162), (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    add64_0(at_2 + 10U, _S159.y + m_2 * (dd_0 - _S164 * _S164), (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    add64_0(at_2 + 12U, _S159.z + m_2 * (dd_0 - _S166 * _S166), (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    add64_0(at_2 + 14U, _S159.w - _S163 * _S164, (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    add64_0(at_2 + 16U, _S160.x - _S163 * _S166, (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    add64_0(at_2 + 18U, _S160.y - _S165 * _S166, (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    return;
}

bool is_root_0(uint c_5, KernelContext_0 thread* kernelContext_32)
{
    bool _S167 = live_0(c_5, kernelContext_32);
    bool _S168;
    if(_S167)
    {
        uint _S169 = body_of_0(c_5, kernelContext_32);
        bool _S170 = dirty_0(_S169, kernelContext_32);
        _S168 = _S170;
    }
    else
    {
        _S168 = false;
    }
    if(_S168)
    {
        uint _S171 = w_label_0(kernelContext_32);
        _S168 = (*(kernelContext_32->work_0+(_S171 + c_5))) == c_5;
    }
    else
    {
        _S168 = false;
    }
    return _S168;
}

[[kernel]] void fr_best_count(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_33;
    (&kernelContext_33)->params_0 = params_8;
    (&kernelContext_33)->work_0 = work_8;
    (&kernelContext_33)->frames_0 = frames_8;
    (&kernelContext_33)->bodies_0 = bodies_8;
    (&kernelContext_33)->chunks_0 = chunks_8;
    (&kernelContext_33)->sort_0 = sort_8;
    (&kernelContext_33)->atom_0 = atom_8;
    (&kernelContext_33)->manifolds_0 = manifolds_8;
    (&kernelContext_33)->loads_0 = loads_8;
    (&kernelContext_33)->bonds_0 = bonds_8;
    (&kernelContext_33)->args_0 = args_8;
    (&kernelContext_33)->info_0 = info_8;
    (&kernelContext_33)->scan_0 = scan_8;
    (&kernelContext_33)->parts_0 = parts_8;
    (&kernelContext_33)->colliders_0 = colliders_8;
    (&kernelContext_33)->collider_info_0 = collider_info_8;
    (&kernelContext_33)->view_chunks_0 = view_chunks_8;
    (&kernelContext_33)->view_0 = view_8;
    (&kernelContext_33)->objects_0 = objects_8;
    (&kernelContext_33)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_33)->lines_0 = lines_8;
    (&kernelContext_33)->geometry_0 = geometry_8;
    (&kernelContext_33)->shots_0 = shots_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_33)->g_scan_0 = &g_scan_8;
    uint c_6 = id_7.x;
    bool _S172 = is_root_0(c_6, &kernelContext_33);
    if(!_S172)
    {
        return;
    }
    uint _S173 = a_best_count_0(&kernelContext_33);
    uint _S174 = body_of_0(c_6, &kernelContext_33);
    atomic<uint> device* _S175 = (&kernelContext_33)->atom_0+(_S173 + _S174);
    uint _S176 = a_count_0(&kernelContext_33);
    uint _S177 = atomic_load_explicit((&kernelContext_33)->atom_0+(_S176 + c_6), memory_order_relaxed);
    uint _S178 = atomic_fetch_max_explicit(_S175, _S177, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_34;
    (&kernelContext_34)->params_0 = params_9;
    (&kernelContext_34)->work_0 = work_9;
    (&kernelContext_34)->frames_0 = frames_9;
    (&kernelContext_34)->bodies_0 = bodies_9;
    (&kernelContext_34)->chunks_0 = chunks_9;
    (&kernelContext_34)->sort_0 = sort_9;
    (&kernelContext_34)->atom_0 = atom_9;
    (&kernelContext_34)->manifolds_0 = manifolds_9;
    (&kernelContext_34)->loads_0 = loads_9;
    (&kernelContext_34)->bonds_0 = bonds_9;
    (&kernelContext_34)->args_0 = args_9;
    (&kernelContext_34)->info_0 = info_9;
    (&kernelContext_34)->scan_0 = scan_9;
    (&kernelContext_34)->parts_0 = parts_9;
    (&kernelContext_34)->colliders_0 = colliders_9;
    (&kernelContext_34)->collider_info_0 = collider_info_9;
    (&kernelContext_34)->view_chunks_0 = view_chunks_9;
    (&kernelContext_34)->view_0 = view_9;
    (&kernelContext_34)->objects_0 = objects_9;
    (&kernelContext_34)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_34)->lines_0 = lines_9;
    (&kernelContext_34)->geometry_0 = geometry_9;
    (&kernelContext_34)->shots_0 = shots_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_34)->g_scan_0 = &g_scan_9;
    uint c_7 = id_8.x;
    bool _S179 = is_root_0(c_7, &kernelContext_34);
    if(!_S179)
    {
        return;
    }
    uint _S180 = body_of_0(c_7, &kernelContext_34);
    uint _S181 = a_count_0(&kernelContext_34);
    uint _S182 = atomic_load_explicit((&kernelContext_34)->atom_0+(_S181 + c_7), memory_order_relaxed);
    uint _S183 = a_best_count_0(&kernelContext_34);
    uint _S184 = atomic_load_explicit((&kernelContext_34)->atom_0+(_S183 + _S180), memory_order_relaxed);
    if(_S182 == _S184)
    {
        uint _S185 = a_best_root_0(&kernelContext_34);
        uint _S186 = atomic_fetch_min_explicit((&kernelContext_34)->atom_0+(_S185 + _S180), c_7, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_35;
    (&kernelContext_35)->params_0 = params_10;
    (&kernelContext_35)->work_0 = work_10;
    (&kernelContext_35)->frames_0 = frames_10;
    (&kernelContext_35)->bodies_0 = bodies_10;
    (&kernelContext_35)->chunks_0 = chunks_10;
    (&kernelContext_35)->sort_0 = sort_10;
    (&kernelContext_35)->atom_0 = atom_10;
    (&kernelContext_35)->manifolds_0 = manifolds_10;
    (&kernelContext_35)->loads_0 = loads_10;
    (&kernelContext_35)->bonds_0 = bonds_10;
    (&kernelContext_35)->args_0 = args_10;
    (&kernelContext_35)->info_0 = info_10;
    (&kernelContext_35)->scan_0 = scan_10;
    (&kernelContext_35)->parts_0 = parts_10;
    (&kernelContext_35)->colliders_0 = colliders_10;
    (&kernelContext_35)->collider_info_0 = collider_info_10;
    (&kernelContext_35)->view_chunks_0 = view_chunks_10;
    (&kernelContext_35)->view_0 = view_10;
    (&kernelContext_35)->objects_0 = objects_10;
    (&kernelContext_35)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_35)->lines_0 = lines_10;
    (&kernelContext_35)->geometry_0 = geometry_10;
    (&kernelContext_35)->shots_0 = shots_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_35)->g_scan_0 = &g_scan_10;
    uint c_8 = id_9.x;
    if(c_8 >= (params_10->chunk_used_0))
    {
        return;
    }
    uint _S187 = w_new_0(&kernelContext_35);
    uint device* _S188 = (&kernelContext_35)->work_0+(_S187 + c_8);
    bool _S189 = is_root_0(c_8, &kernelContext_35);
    bool _S190;
    if(_S189)
    {
        uint _S191 = a_best_root_0(&kernelContext_35);
        uint _S192 = body_of_0(c_8, &kernelContext_35);
        uint _S193 = atomic_load_explicit((&kernelContext_35)->atom_0+(_S191 + _S192), memory_order_relaxed);
        _S190 = _S193 != c_8;
    }
    else
    {
        _S190 = false;
    }
    uint _S194;
    if(_S190)
    {
        _S194 = 1U;
    }
    else
    {
        _S194 = 0U;
    }
    *_S188 = _S194;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]])
{
    uint step_2;
    uint add_0;
    thread KernelContext_0 kernelContext_36;
    (&kernelContext_36)->params_0 = params_11;
    (&kernelContext_36)->work_0 = work_11;
    (&kernelContext_36)->frames_0 = frames_11;
    (&kernelContext_36)->bodies_0 = bodies_11;
    (&kernelContext_36)->chunks_0 = chunks_11;
    (&kernelContext_36)->sort_0 = sort_11;
    (&kernelContext_36)->atom_0 = atom_11;
    (&kernelContext_36)->manifolds_0 = manifolds_11;
    (&kernelContext_36)->loads_0 = loads_11;
    (&kernelContext_36)->bonds_0 = bonds_11;
    (&kernelContext_36)->args_0 = args_11;
    (&kernelContext_36)->info_0 = info_11;
    (&kernelContext_36)->scan_0 = scan_11;
    (&kernelContext_36)->parts_0 = parts_11;
    (&kernelContext_36)->colliders_0 = colliders_11;
    (&kernelContext_36)->collider_info_0 = collider_info_11;
    (&kernelContext_36)->view_chunks_0 = view_chunks_11;
    (&kernelContext_36)->view_0 = view_11;
    (&kernelContext_36)->objects_0 = objects_11;
    (&kernelContext_36)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_36)->lines_0 = lines_11;
    (&kernelContext_36)->geometry_0 = geometry_11;
    (&kernelContext_36)->shots_0 = shots_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_36)->g_scan_0 = &g_scan_11;
    uint t_1 = thread_0.x;
    uint _S195 = group_0.x;
    uint _S196 = 1024U * _S195 + 4U * t_1;
    thread array<uint, int(4)> v_4;
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
        if((_S196 + j_2) < ((&kernelContext_36)->scan_0->n_0))
        {
            step_2 = *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->in_base_0 + _S196 + j_2));
        }
        else
        {
            step_2 = 0U;
        }
        v_4[j_2] = step_2;
        uint sum_3 = sum_2 + step_2;
        j_2 = j_2 + 1U;
        sum_2 = sum_3;
    }
    (*(&kernelContext_36)->g_scan_0)[t_1] = sum_2;
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
        if(t_1 >= step_2)
        {
            add_0 = (*(&kernelContext_36)->g_scan_0)[t_1 - step_2];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_36)->g_scan_0)[t_1] = (*(&kernelContext_36)->g_scan_0)[t_1] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_2 = step_2 << 1U;
    }
    uint _S197 = (*(&kernelContext_36)->g_scan_0)[t_1] - sum_2;
    j_2 = 0U;
    add_0 = _S197;
    for(;;)
    {
        if(j_2 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S196 + j_2) < ((&kernelContext_36)->scan_0->n_0))
        {
            *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->out_base_0 + _S196 + j_2)) = add_0;
        }
        uint run_1 = add_0 + v_4[j_2];
        j_2 = j_2 + 1U;
        add_0 = run_1;
    }
    if(t_1 == 255U)
    {
        *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->block_base_0 + _S195)) = (*(&kernelContext_36)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_37;
    (&kernelContext_37)->params_0 = params_12;
    (&kernelContext_37)->work_0 = work_12;
    (&kernelContext_37)->frames_0 = frames_12;
    (&kernelContext_37)->bodies_0 = bodies_12;
    (&kernelContext_37)->chunks_0 = chunks_12;
    (&kernelContext_37)->sort_0 = sort_12;
    (&kernelContext_37)->atom_0 = atom_12;
    (&kernelContext_37)->manifolds_0 = manifolds_12;
    (&kernelContext_37)->loads_0 = loads_12;
    (&kernelContext_37)->bonds_0 = bonds_12;
    (&kernelContext_37)->args_0 = args_12;
    (&kernelContext_37)->info_0 = info_12;
    (&kernelContext_37)->scan_0 = scan_12;
    (&kernelContext_37)->parts_0 = parts_12;
    (&kernelContext_37)->colliders_0 = colliders_12;
    (&kernelContext_37)->collider_info_0 = collider_info_12;
    (&kernelContext_37)->view_chunks_0 = view_chunks_12;
    (&kernelContext_37)->view_0 = view_12;
    (&kernelContext_37)->objects_0 = objects_12;
    (&kernelContext_37)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_37)->lines_0 = lines_12;
    (&kernelContext_37)->geometry_0 = geometry_12;
    (&kernelContext_37)->shots_0 = shots_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_37)->g_scan_0 = &g_scan_12;
    uint t_2 = thread_1.x;
    uint blocks_0 = (scan_12->n_0 + 1023U) / 1024U;
    bool _S198 = t_2 < blocks_0;
    uint v_5;
    if(_S198)
    {
        v_5 = *((&kernelContext_37)->work_0+((&kernelContext_37)->scan_0->block_base_0 + t_2));
    }
    else
    {
        v_5 = 0U;
    }
    (*(&kernelContext_37)->g_scan_0)[t_2] = v_5;
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
        if(t_2 >= step_3)
        {
            add_1 = (*(&kernelContext_37)->g_scan_0)[t_2 - step_3];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_37)->g_scan_0)[t_2] = (*(&kernelContext_37)->g_scan_0)[t_2] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_3 = step_3 << 1U;
    }
    if(_S198)
    {
        *((&kernelContext_37)->work_0+((&kernelContext_37)->scan_0->block_base_0 + t_2)) = (*(&kernelContext_37)->g_scan_0)[t_2] - v_5;
    }
    if(t_2 == 1023U)
    {
        *((&kernelContext_37)->work_0+((&kernelContext_37)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_37)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_38;
    (&kernelContext_38)->params_0 = params_13;
    (&kernelContext_38)->work_0 = work_13;
    (&kernelContext_38)->frames_0 = frames_13;
    (&kernelContext_38)->bodies_0 = bodies_13;
    (&kernelContext_38)->chunks_0 = chunks_13;
    (&kernelContext_38)->sort_0 = sort_13;
    (&kernelContext_38)->atom_0 = atom_13;
    (&kernelContext_38)->manifolds_0 = manifolds_13;
    (&kernelContext_38)->loads_0 = loads_13;
    (&kernelContext_38)->bonds_0 = bonds_13;
    (&kernelContext_38)->args_0 = args_13;
    (&kernelContext_38)->info_0 = info_13;
    (&kernelContext_38)->scan_0 = scan_13;
    (&kernelContext_38)->parts_0 = parts_13;
    (&kernelContext_38)->colliders_0 = colliders_13;
    (&kernelContext_38)->collider_info_0 = collider_info_13;
    (&kernelContext_38)->view_chunks_0 = view_chunks_13;
    (&kernelContext_38)->view_0 = view_13;
    (&kernelContext_38)->objects_0 = objects_13;
    (&kernelContext_38)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_38)->lines_0 = lines_13;
    (&kernelContext_38)->geometry_0 = geometry_13;
    (&kernelContext_38)->shots_0 = shots_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_38)->g_scan_0 = &g_scan_13;
    uint i_2 = id_10.x;
    if(i_2 >= (scan_13->n_0))
    {
        return;
    }
    uint device* _S199 = (&kernelContext_38)->work_0+((&kernelContext_38)->scan_0->out_base_0 + i_2);
    *_S199 = *_S199 + *((&kernelContext_38)->work_0+((&kernelContext_38)->scan_0->block_base_0 + i_2 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_39;
    (&kernelContext_39)->params_0 = params_14;
    (&kernelContext_39)->work_0 = work_14;
    (&kernelContext_39)->frames_0 = frames_14;
    (&kernelContext_39)->bodies_0 = bodies_14;
    (&kernelContext_39)->chunks_0 = chunks_14;
    (&kernelContext_39)->sort_0 = sort_14;
    (&kernelContext_39)->atom_0 = atom_14;
    (&kernelContext_39)->manifolds_0 = manifolds_14;
    (&kernelContext_39)->loads_0 = loads_14;
    (&kernelContext_39)->bonds_0 = bonds_14;
    (&kernelContext_39)->args_0 = args_14;
    (&kernelContext_39)->info_0 = info_14;
    (&kernelContext_39)->scan_0 = scan_14;
    (&kernelContext_39)->parts_0 = parts_14;
    (&kernelContext_39)->colliders_0 = colliders_14;
    (&kernelContext_39)->collider_info_0 = collider_info_14;
    (&kernelContext_39)->view_chunks_0 = view_chunks_14;
    (&kernelContext_39)->view_0 = view_14;
    (&kernelContext_39)->objects_0 = objects_14;
    (&kernelContext_39)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_39)->lines_0 = lines_14;
    (&kernelContext_39)->geometry_0 = geometry_14;
    (&kernelContext_39)->shots_0 = shots_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_39)->g_scan_0 = &g_scan_14;
    uint s_1 = id_11.x;
    bool _S200;
    if(s_1 >= (params_14->body_count_0))
    {
        _S200 = true;
    }
    else
    {
        uint _S201 = w_free_0(&kernelContext_39);
        _S200 = (*((&kernelContext_39)->work_0+(_S201 + s_1))) == 0U;
    }
    if(_S200)
    {
        return;
    }
    uint _S202 = w_free_list_0(&kernelContext_39);
    uint _S203 = w_free_rank_0(&kernelContext_39);
    *((&kernelContext_39)->work_0+(_S202 + *((&kernelContext_39)->work_0+(_S203 + s_1)))) = s_1;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_40)
{
    uint _S204 = w_free_list_0(kernelContext_40);
    return _S204 + kernelContext_40->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_41)
{
    uint _S205 = scan_new_blocks_0(kernelContext_41);
    return _S205 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_42)
{
    uint blocks_1 = (kernelContext_42->params_0->body_count_0 + 1023U) / 1024U;
    uint _S206 = scan_free_blocks_0(kernelContext_42);
    return *(kernelContext_42->work_0+(_S206 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_43;
    (&kernelContext_43)->params_0 = params_15;
    (&kernelContext_43)->work_0 = work_15;
    (&kernelContext_43)->frames_0 = frames_15;
    (&kernelContext_43)->bodies_0 = bodies_15;
    (&kernelContext_43)->chunks_0 = chunks_15;
    (&kernelContext_43)->sort_0 = sort_15;
    (&kernelContext_43)->atom_0 = atom_15;
    (&kernelContext_43)->manifolds_0 = manifolds_15;
    (&kernelContext_43)->loads_0 = loads_15;
    (&kernelContext_43)->bonds_0 = bonds_15;
    (&kernelContext_43)->args_0 = args_15;
    (&kernelContext_43)->info_0 = info_15;
    (&kernelContext_43)->scan_0 = scan_15;
    (&kernelContext_43)->parts_0 = parts_15;
    (&kernelContext_43)->colliders_0 = colliders_15;
    (&kernelContext_43)->collider_info_0 = collider_info_15;
    (&kernelContext_43)->view_chunks_0 = view_chunks_15;
    (&kernelContext_43)->view_0 = view_15;
    (&kernelContext_43)->objects_0 = objects_15;
    (&kernelContext_43)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_43)->lines_0 = lines_15;
    (&kernelContext_43)->geometry_0 = geometry_15;
    (&kernelContext_43)->shots_0 = shots_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_43)->g_scan_0 = &g_scan_15;
    uint c_9 = id_12.x;
    bool _S207 = is_root_0(c_9, &kernelContext_43);
    if(!_S207)
    {
        return;
    }
    uint _S208 = body_of_0(c_9, &kernelContext_43);
    uint _S209 = a_best_root_0(&kernelContext_43);
    uint _S210 = atomic_load_explicit((&kernelContext_43)->atom_0+(_S209 + _S208), memory_order_relaxed);
    uint slot_1;
    if(_S210 != c_9)
    {
        uint _S211 = w_rank_0(&kernelContext_43);
        uint device* _S212 = (&kernelContext_43)->work_0+(_S211 + c_9);
        uint k_1 = *_S212;
        uint _S213 = free_total_0(&kernelContext_43);
        if((*_S212) < _S213)
        {
            uint _S214 = w_free_list_0(&kernelContext_43);
            uint device* _S215 = (&kernelContext_43)->work_0+(_S214 + k_1);
            uint slot_2 = *_S215;
            uint _S216 = w_free_0(&kernelContext_43);
            *((&kernelContext_43)->work_0+(_S216 + *_S215)) = 0U;
            uint _S217 = a_counters_0(&kernelContext_43);
            uint _S218 = atomic_fetch_add_explicit((&kernelContext_43)->atom_0+(_S217 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S219 = a_counters_0(&kernelContext_43);
            uint _S220 = atomic_fetch_or_explicit((&kernelContext_43)->atom_0+(_S219 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S208;
        }
    }
    else
    {
        slot_1 = _S208;
    }
    uint _S221 = w_slot_0(&kernelContext_43);
    *((&kernelContext_43)->work_0+(_S221 + c_9)) = slot_1;
    return;
}

float read64_0(uint at_3, float scale_1, KernelContext_0 thread* kernelContext_44)
{
    uint _S222 = atomic_load_explicit(kernelContext_44->atom_0+(at_3 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S222)));
    uint lo_1 = atomic_load_explicit(kernelContext_44->atom_0+at_3, memory_order_relaxed);
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
    thread Quat_0 q_7;
    if(m22_0 <= 0.0f)
    {
        float dif10_0 = m11_0 - m00_0;
        float omm22_0 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            float f_3 = omm22_0 - dif10_0;
            float inv_0 = 0.5f / sqrt(f_3);
            (&q_7)->x_0 = f_3 * inv_0;
            (&q_7)->y_0 = (m01_0 + m10_0) * inv_0;
            (&q_7)->z_0 = (m02_0 + m20_0) * inv_0;
            (&q_7)->w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            float f_4 = omm22_0 + dif10_0;
            float inv_1 = 0.5f / sqrt(f_4);
            (&q_7)->x_0 = (m01_0 + m10_0) * inv_1;
            (&q_7)->y_0 = f_4 * inv_1;
            (&q_7)->z_0 = (m12_0 + m21_0) * inv_1;
            (&q_7)->w_0 = (m20_0 - m02_0) * inv_1;
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
            (&q_7)->x_0 = (m02_0 + m20_0) * inv_2;
            (&q_7)->y_0 = (m12_0 + m21_0) * inv_2;
            (&q_7)->z_0 = f_5 * inv_2;
            (&q_7)->w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            float f_6 = opm22_0 + sum10_0;
            float inv_3 = 0.5f / sqrt(f_6);
            (&q_7)->x_0 = (m12_0 - m21_0) * inv_3;
            (&q_7)->y_0 = (m20_0 - m02_0) * inv_3;
            (&q_7)->z_0 = (m01_0 - m10_0) * inv_3;
            (&q_7)->w_0 = f_6 * inv_3;
        }
    }
    return q_7;
}

void principal_axes_0(float3 r0_0, float3 r1_0, float3 r2_0, Quat_0 thread* rot_0, float3 thread* moments_0)
{
    array<float, int(3)> _S223 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S224 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S225 = { { r2_0.x, r2_0.y, r2_0.z } };
    thread array<array<float, int(3)>, int(3)> a_3;
    a_3[int(0)] = _S223;
    a_3[int(1)] = _S224;
    a_3[int(2)] = _S225;
    array<float, int(3)> _S226 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S227 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S228 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_6;
    v_6[int(0)] = _S226;
    v_6[int(1)] = _S227;
    v_6[int(2)] = _S228;
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
        uint q_8;
        if((abs(a_3[int(0)][int(2)])) > (abs(a_3[0U][1U])))
        {
            q_8 = 2U;
        }
        else
        {
            q_8 = 1U;
        }
        uint p_2;
        uint q_9;
        if((abs(a_3[int(1)][int(2)])) > (abs(a_3[0U][q_8])))
        {
            p_2 = 1U;
            q_9 = 2U;
        }
        else
        {
            p_2 = 0U;
            q_9 = q_8;
        }
        if((abs(a_3[p_2][q_9])) <= (9.99999982451670044e-15f * max(max(abs(a_3[int(0)][int(0)]), abs(a_3[int(1)][int(1)])), max(abs(a_3[int(2)][int(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        float theta_0 = 0.5f * atan2(2.0f * a_3[p_2][q_9], a_3[q_9][q_9] - a_3[p_2][p_2]);
        float s_2 = sin(theta_0);
        float c_10 = cos(theta_0);
        float app_0 = a_3[p_2][p_2];
        float aqq_0 = a_3[q_9][q_9];
        float apq_0 = a_3[p_2][q_9];
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
            bool _S229;
            if(k_2 != p_2)
            {
                _S229 = k_2 != q_9;
            }
            else
            {
                _S229 = false;
            }
            if(_S229)
            {
                float kp_0 = a_3[k_2][p_2];
                float kq_0 = a_3[k_2][q_9];
                float _S230 = c_10 * a_3[k_2][p_2] - s_2 * a_3[k_2][q_9];
                a_3[k_2][p_2] = _S230;
                a_3[p_2][k_2] = _S230;
                float _S231 = s_2 * kp_0 + c_10 * kq_0;
                a_3[k_2][q_9] = _S231;
                a_3[q_9][k_2] = _S231;
            }
            k_2 = k_2 + 1U;
        }
        float _S232 = c_10 * c_10;
        float _S233 = 2.0f * s_2 * c_10 * apq_0;
        float _S234 = s_2 * s_2;
        a_3[p_2][p_2] = _S232 * app_0 - _S233 + _S234 * aqq_0;
        a_3[q_9][q_9] = _S234 * app_0 + _S233 + _S232 * aqq_0;
        a_3[p_2][q_9] = 0.0f;
        a_3[q_9][p_2] = 0.0f;
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
            float kp_1 = v_6[row_0][p_2];
            float kq_1 = v_6[row_0][q_9];
            v_6[row_0][p_2] = c_10 * v_6[row_0][p_2] - s_2 * v_6[row_0][q_9];
            v_6[row_0][q_9] = s_2 * kp_1 + c_10 * kq_1;
            row_0 = row_0 + 1U;
        }
        it_0 = it_0 + 1U;
    }
    float3 x_5 = float3(v_6[int(0)][int(0)], v_6[int(1)][int(0)], v_6[int(2)][int(0)]);
    float3 y_1 = float3(v_6[int(0)][int(1)], v_6[int(1)][int(1)], v_6[int(2)][int(1)]);
    float3 z_1 = float3(v_6[int(0)][int(2)], v_6[int(1)][int(2)], v_6[int(2)][int(2)]);
    float3 z_2;
    if((dot(cross(x_5, y_1), z_1)) < 0.0f)
    {
        z_2 = - z_1;
    }
    else
    {
        z_2 = z_1;
    }
    thread Quat_0 _S235 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S236 = normalized_0(&_S235);
    *rot_0 = _S236;
    *moments_0 = float3(a_3[int(0)][int(0)], a_3[int(1)][int(1)], a_3[int(2)][int(2)]);
    return;
}

float3 to_rubble_0(float3 v_7)
{
    return float3(v_7.x, v_7.z, - v_7.y);
}

Quat_0 rot_to_rubble_0(const Quat_0 thread* q_10)
{
    thread Quat_0 axes_1;
    (&axes_1)->w_0 = 0.70710676908493042f;
    (&axes_1)->x_0 = -0.70710676908493042f;
    (&axes_1)->y_0 = 0.0f;
    (&axes_1)->z_0 = 0.0f;
    thread Quat_0 _S237 = axes_1;
    Quat_0 _S238 = quat_mul_0(&_S237, q_10);
    thread Quat_0 _S239 = _S238;
    Quat_0 _S240 = normalized_0(&_S239);
    return _S240;
}

uint stage_0(uint c_11, KernelContext_0 thread* kernelContext_45)
{
    return 2U * kernelContext_45->params_0->body_count_0 + 9U * c_11;
}

float4 quat_vec_0(const Quat_0 thread* q_11)
{
    return float4(q_11->x_0, q_11->y_0, q_11->z_0, q_11->w_0);
}

[[kernel]] void fr_piece(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_46;
    (&kernelContext_46)->params_0 = params_16;
    (&kernelContext_46)->work_0 = work_16;
    (&kernelContext_46)->frames_0 = frames_16;
    (&kernelContext_46)->bodies_0 = bodies_16;
    (&kernelContext_46)->chunks_0 = chunks_16;
    (&kernelContext_46)->sort_0 = sort_16;
    (&kernelContext_46)->atom_0 = atom_16;
    (&kernelContext_46)->manifolds_0 = manifolds_16;
    (&kernelContext_46)->loads_0 = loads_16;
    (&kernelContext_46)->bonds_0 = bonds_16;
    (&kernelContext_46)->args_0 = args_16;
    (&kernelContext_46)->info_0 = info_16;
    (&kernelContext_46)->scan_0 = scan_16;
    (&kernelContext_46)->parts_0 = parts_16;
    (&kernelContext_46)->colliders_0 = colliders_16;
    (&kernelContext_46)->collider_info_0 = collider_info_16;
    (&kernelContext_46)->view_chunks_0 = view_chunks_16;
    (&kernelContext_46)->view_0 = view_16;
    (&kernelContext_46)->objects_0 = objects_16;
    (&kernelContext_46)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_46)->lines_0 = lines_16;
    (&kernelContext_46)->geometry_0 = geometry_16;
    (&kernelContext_46)->shots_0 = shots_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_46)->g_scan_0 = &g_scan_16;
    uint c_12 = id_13.x;
    bool _S241 = is_root_0(c_12, &kernelContext_46);
    if(!_S241)
    {
        return;
    }
    uint _S242 = body_of_0(c_12, &kernelContext_46);
    uint pb_0 = 7U * _S242;
    uint _S243 = a_sums_0(&kernelContext_46);
    uint at_4 = _S243 + 20U * c_12;
    float _S244 = read64_0(at_4, (&kernelContext_46)->params_0->scale_m_0, &kernelContext_46);
    float _S245 = read64_0(at_4 + 2U, (&kernelContext_46)->params_0->scale_s_0, &kernelContext_46);
    float _S246 = read64_0(at_4 + 4U, (&kernelContext_46)->params_0->scale_s_0, &kernelContext_46);
    float _S247 = read64_0(at_4 + 6U, (&kernelContext_46)->params_0->scale_s_0, &kernelContext_46);
    float3 d_1 = float3(_S245, _S246, _S247) / float3(max(_S244, 1.00000000317107685e-30f)) ;
    float _S248 = read64_0(at_4 + 8U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float _S249 = read64_0(at_4 + 10U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float _S250 = read64_0(at_4 + 12U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float _S251 = read64_0(at_4 + 14U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float _S252 = read64_0(at_4 + 16U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float _S253 = read64_0(at_4 + 18U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float dd_1 = dot(d_1, d_1);
    float _S254 = d_1.x;
    float _S255 = _S244 * _S254;
    float _S256 = d_1.y;
    float _S257 = _S251 + _S255 * _S256;
    float _S258 = d_1.z;
    float _S259 = _S252 + _S255 * _S258;
    float _S260 = _S253 + _S244 * _S256 * _S258;
    float3 com_1 = (float4(*((&kernelContext_46)->chunks_0+4U * c_12)) ).xyz + d_1;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S248 - _S244 * (dd_1 - _S254 * _S254), _S257, _S259), float3(_S257, _S249 - _S244 * (dd_1 - _S256 * _S256), _S260), float3(_S259, _S260, _S250 - _S244 * (dd_1 - _S258 * _S258)), &principal_0, &moments_1);
    uint _S261 = a_anchor_0(&kernelContext_46);
    uint _S262 = atomic_load_explicit((&kernelContext_46)->atom_0+(_S261 + c_12), memory_order_relaxed);
    bool anchored_0 = _S262 != 0U;
    thread Quat_0 q_12;
    thread float3 origin_2;
    structure_pose_0(_S242, pb_0, &q_12, &origin_2, &kernelContext_46);
    float3 p_parent_0 = (float4(*((&kernelContext_46)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_46)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_46)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S263 = q_12;
    float3 _S264 = rotate_0(&_S263, com_1);
    float3 pos_0 = to_rubble_0(origin_2 + _S264);
    thread Quat_0 _S265 = q_12;
    thread Quat_0 _S266 = principal_0;
    Quat_0 _S267 = quat_mul_0(&_S265, &_S266);
    thread Quat_0 _S268 = _S267;
    Quat_0 _S269 = normalized_0(&_S268);
    thread Quat_0 _S270 = _S269;
    Quat_0 _S271 = rot_to_rubble_0(&_S270);
    bool _S272;
    if(anchored_0)
    {
        _S272 = true;
    }
    else
    {
        _S272 = !(_S244 > 0.0f);
    }
    float inv_mass_1;
    if(_S272)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / _S244;
    }
    float3 _S273 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S274;
        if((moments_1.x) > 0.0f)
        {
            _S274 = 1.0f / moments_1.x;
        }
        else
        {
            _S274 = 0.0f;
        }
        float _S275;
        if((moments_1.y) > 0.0f)
        {
            _S275 = 1.0f / moments_1.y;
        }
        else
        {
            _S275 = 0.0f;
        }
        float _S276;
        if((moments_1.z) > 0.0f)
        {
            _S276 = 1.0f / moments_1.z;
        }
        else
        {
            _S276 = 0.0f;
        }
        inv_inertia_0 = float3(_S274, _S275, _S276);
    }
    else
    {
        inv_inertia_0 = _S273;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S273;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S273;
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
    uint _S277 = stage_0(c_12, &kernelContext_46);
    *((&kernelContext_46)->frames_0+_S277) = packed_float4(float4(pos_0, inv_mass_1)) ;
    packed_float4 device* _S278 = (&kernelContext_46)->frames_0+(_S277 + 1U);
    thread Quat_0 _S279 = _S271;
    float4 _S280 = quat_vec_0(&_S279);
    *_S278 = packed_float4(_S280) ;
    *((&kernelContext_46)->frames_0+(_S277 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_46)->frames_0+(_S277 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_46)->frames_0+(_S277 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_46)->params_0->step_0))))) ;
    *((&kernelContext_46)->frames_0+(_S277 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_46)->frames_0+(_S277 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_46)->frames_0+(_S277 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_46)->frames_0+2U * _S242)) ).w)) ;
    packed_float4 device* _S281 = (&kernelContext_46)->frames_0+(_S277 + 8U);
    thread Quat_0 _S282 = principal_0;
    float4 _S283 = quat_vec_0(&_S282);
    *_S281 = packed_float4(_S283) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_47;
    (&kernelContext_47)->params_0 = params_17;
    (&kernelContext_47)->work_0 = work_17;
    (&kernelContext_47)->frames_0 = frames_17;
    (&kernelContext_47)->bodies_0 = bodies_17;
    (&kernelContext_47)->chunks_0 = chunks_17;
    (&kernelContext_47)->sort_0 = sort_17;
    (&kernelContext_47)->atom_0 = atom_17;
    (&kernelContext_47)->manifolds_0 = manifolds_17;
    (&kernelContext_47)->loads_0 = loads_17;
    (&kernelContext_47)->bonds_0 = bonds_17;
    (&kernelContext_47)->args_0 = args_17;
    (&kernelContext_47)->info_0 = info_17;
    (&kernelContext_47)->scan_0 = scan_17;
    (&kernelContext_47)->parts_0 = parts_17;
    (&kernelContext_47)->colliders_0 = colliders_17;
    (&kernelContext_47)->collider_info_0 = collider_info_17;
    (&kernelContext_47)->view_chunks_0 = view_chunks_17;
    (&kernelContext_47)->view_0 = view_17;
    (&kernelContext_47)->objects_0 = objects_17;
    (&kernelContext_47)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_47)->lines_0 = lines_17;
    (&kernelContext_47)->geometry_0 = geometry_17;
    (&kernelContext_47)->shots_0 = shots_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_47)->g_scan_0 = &g_scan_17;
    uint c_13 = id_14.x;
    bool _S284 = is_root_0(c_13, &kernelContext_47);
    if(!_S284)
    {
        return;
    }
    uint _S285 = w_slot_0(&kernelContext_47);
    uint slot_3 = *((&kernelContext_47)->work_0+(_S285 + c_13));
    uint _S286 = stage_0(c_13, &kernelContext_47);
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
            *((&kernelContext_47)->bodies_0+(7U * (slot_3 + region_0 * (&kernelContext_47)->params_0->max_bodies_0) + w_3)) = packed_float4(float4(*((&kernelContext_47)->frames_0+(_S286 + w_3))) ) ;
            w_3 = w_3 + 1U;
        }
        region_0 = region_0 + 1U;
    }
    uint _S287 = 2U * slot_3;
    *((&kernelContext_47)->frames_0+_S287) = packed_float4(float4(*((&kernelContext_47)->frames_0+(_S286 + 7U))) ) ;
    *((&kernelContext_47)->frames_0+(_S287 + 1U)) = packed_float4(float4(*((&kernelContext_47)->frames_0+(_S286 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_48;
    (&kernelContext_48)->params_0 = params_18;
    (&kernelContext_48)->work_0 = work_18;
    (&kernelContext_48)->frames_0 = frames_18;
    (&kernelContext_48)->bodies_0 = bodies_18;
    (&kernelContext_48)->chunks_0 = chunks_18;
    (&kernelContext_48)->sort_0 = sort_18;
    (&kernelContext_48)->atom_0 = atom_18;
    (&kernelContext_48)->manifolds_0 = manifolds_18;
    (&kernelContext_48)->loads_0 = loads_18;
    (&kernelContext_48)->bonds_0 = bonds_18;
    (&kernelContext_48)->args_0 = args_18;
    (&kernelContext_48)->info_0 = info_18;
    (&kernelContext_48)->scan_0 = scan_18;
    (&kernelContext_48)->parts_0 = parts_18;
    (&kernelContext_48)->colliders_0 = colliders_18;
    (&kernelContext_48)->collider_info_0 = collider_info_18;
    (&kernelContext_48)->view_chunks_0 = view_chunks_18;
    (&kernelContext_48)->view_0 = view_18;
    (&kernelContext_48)->objects_0 = objects_18;
    (&kernelContext_48)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_48)->lines_0 = lines_18;
    (&kernelContext_48)->geometry_0 = geometry_18;
    (&kernelContext_48)->shots_0 = shots_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_48)->g_scan_0 = &g_scan_18;
    uint c_14 = id_15.x;
    bool _S288 = live_0(c_14, &kernelContext_48);
    bool _S289;
    if(!_S288)
    {
        _S289 = true;
    }
    else
    {
        uint _S290 = body_of_0(c_14, &kernelContext_48);
        bool _S291 = dirty_0(_S290, &kernelContext_48);
        _S289 = !_S291;
    }
    if(_S289)
    {
        return;
    }
    uint _S292 = w_label_0(&kernelContext_48);
    uint device* _S293 = (&kernelContext_48)->work_0+(_S292 + c_14);
    uint r_6 = *_S293;
    uint _S294 = w_slot_0(&kernelContext_48);
    uint device* _S295 = (&kernelContext_48)->work_0+(_S294 + *_S293);
    uint slot_4 = *_S295;
    *((&kernelContext_48)->work_0+(w_body_0() + c_14)) = *_S295;
    uint _S296 = 2U * slot_4;
    float3 _S297 = (float4(*((&kernelContext_48)->frames_0+_S296)) ).xyz;
    thread Quat_0 _S298 = quat_of_0(float4(*((&kernelContext_48)->frames_0+(_S296 + 1U))) );
    Quat_0 _S299 = conj_0(&_S298);
    float inv_mass_2 = (float4(*((&kernelContext_48)->bodies_0+7U * slot_4)) ).w;
    float _S300;
    if(inv_mass_2 > 0.0f)
    {
        _S300 = 1.0f / inv_mass_2;
    }
    else
    {
        _S300 = float_bits_0(2139095040U);
    }
    float4 _S301 = float4(*((&kernelContext_48)->chunks_0+(4U * c_14 + 2U))) ;
    uint first_1 = (as_type<uint>((_S301.z)));
    uint _S302 = (as_type<uint>((_S301.w)));
    float4 _S303 = float4(*((&kernelContext_48)->chunks_0+(4U * r_6 + 3U))) ;
    uint p_3 = first_1;
    for(;;)
    {
        if(p_3 < (first_1 + _S302))
        {
        }
        else
        {
            break;
        }
        uint _S304 = 2U * p_3;
        float4 _S305 = float4(*((&kernelContext_48)->parts_0+_S304)) ;
        uint col_0 = (as_type<uint>((_S305.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_48)->parts_0+(_S304 + 1U))) );
        uint _S306 = 4U * col_0;
        float4 _S307 = float4(*((&kernelContext_48)->colliders_0+_S306)) ;
        *((&kernelContext_48)->colliders_0+_S306) = packed_float4(float4((as_type<float>((slot_4))), _S307.y, _S307.z, _S307.w)) ;
        uint _S308 = _S306 + 1U;
        packed_float4 device* _S309 = (&kernelContext_48)->colliders_0+_S308;
        float3 _S310 = _S305.xyz - _S297;
        thread Quat_0 _S311 = _S299;
        float3 _S312 = rotate_0(&_S311, _S310);
        *_S309 = packed_float4(float4(_S312, (float4(*((&kernelContext_48)->colliders_0+_S308)) ).w)) ;
        packed_float4 device* _S313 = (&kernelContext_48)->colliders_0+(_S306 + 2U);
        thread Quat_0 _S314 = _S299;
        thread Quat_0 _S315 = rot_1;
        Quat_0 _S316 = quat_mul_0(&_S314, &_S315);
        thread Quat_0 _S317 = _S316;
        Quat_0 _S318 = normalized_0(&_S317);
        thread Quat_0 _S319 = _S318;
        float4 _S320 = quat_vec_0(&_S319);
        *_S313 = packed_float4(_S320) ;
        uint _S321 = 2U * col_0;
        uint4 _S322 = uint4(*((&kernelContext_48)->collider_info_0+_S321)) ;
        *((&kernelContext_48)->collider_info_0+_S321) = packed_uint4(uint4(_S322.x, slot_4, _S322.z, _S322.w)) ;
        *((&kernelContext_48)->collider_info_0+(_S321 + 1U)) = packed_uint4(uint4((as_type<uint>((_S300))), (as_type<uint>((_S303.x))), (as_type<uint>((_S303.w))), 0U)) ;
        p_3 = p_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_49;
    (&kernelContext_49)->params_0 = params_19;
    (&kernelContext_49)->work_0 = work_19;
    (&kernelContext_49)->frames_0 = frames_19;
    (&kernelContext_49)->bodies_0 = bodies_19;
    (&kernelContext_49)->chunks_0 = chunks_19;
    (&kernelContext_49)->sort_0 = sort_19;
    (&kernelContext_49)->atom_0 = atom_19;
    (&kernelContext_49)->manifolds_0 = manifolds_19;
    (&kernelContext_49)->loads_0 = loads_19;
    (&kernelContext_49)->bonds_0 = bonds_19;
    (&kernelContext_49)->args_0 = args_19;
    (&kernelContext_49)->info_0 = info_19;
    (&kernelContext_49)->scan_0 = scan_19;
    (&kernelContext_49)->parts_0 = parts_19;
    (&kernelContext_49)->colliders_0 = colliders_19;
    (&kernelContext_49)->collider_info_0 = collider_info_19;
    (&kernelContext_49)->view_chunks_0 = view_chunks_19;
    (&kernelContext_49)->view_0 = view_19;
    (&kernelContext_49)->objects_0 = objects_19;
    (&kernelContext_49)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_49)->lines_0 = lines_19;
    (&kernelContext_49)->geometry_0 = geometry_19;
    (&kernelContext_49)->shots_0 = shots_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_49)->g_scan_0 = &g_scan_19;
    uint s_3 = id_16.x;
    if(s_3 >= (params_19->body_count_0))
    {
        return;
    }
    uint _S323 = a_dirty_0(&kernelContext_49);
    atomic_store_explicit((&kernelContext_49)->atom_0+(_S323 + s_3), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_0 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_7 = fract(h_0);
    float q_13 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_7);
    float t_3 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_7));
    uint k_3 = uint(h_0);
    float3 _S324;
    if(k_3 == 0U)
    {
        _S324 = float3(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_3 == 1U)
        {
            _S324 = float3(q_13, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_3 == 2U)
            {
                _S324 = float3(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_3 == 3U)
                {
                    _S324 = float3(0.34959998726844788f, q_13, 0.92000001668930054f);
                }
                else
                {
                    if(k_3 == 4U)
                    {
                        _S324 = float3(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S324 = float3(0.92000001668930054f, 0.34959998726844788f, q_13);
                    }
                }
            }
        }
    }
    return _S324;
}

void write_object_0(uint o_1, const Quat_0 thread* q_14, float3 t_4, float4 color_0, KernelContext_0 thread* kernelContext_50)
{
    float3 _S325 = rotate_0(q_14, float3(1.0f, 0.0f, 0.0f));
    float3 _S326 = rotate_0(q_14, float3(0.0f, 1.0f, 0.0f));
    float3 _S327 = rotate_0(q_14, float3(0.0f, 0.0f, 1.0f));
    uint _S328 = 4U * o_1;
    *(kernelContext_50->objects_0+_S328) = packed_float4(float4(_S325.x, _S326.x, _S327.x, t_4.x)) ;
    *(kernelContext_50->objects_0+(_S328 + 1U)) = packed_float4(float4(_S325.y, _S326.y, _S327.y, t_4.y)) ;
    *(kernelContext_50->objects_0+(_S328 + 2U)) = packed_float4(float4(_S325.z, _S326.z, _S327.z, t_4.z)) ;
    *(kernelContext_50->objects_0+(_S328 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_51;
    (&kernelContext_51)->params_0 = params_20;
    (&kernelContext_51)->work_0 = work_20;
    (&kernelContext_51)->frames_0 = frames_20;
    (&kernelContext_51)->bodies_0 = bodies_20;
    (&kernelContext_51)->chunks_0 = chunks_20;
    (&kernelContext_51)->sort_0 = sort_20;
    (&kernelContext_51)->atom_0 = atom_20;
    (&kernelContext_51)->manifolds_0 = manifolds_20;
    (&kernelContext_51)->loads_0 = loads_20;
    (&kernelContext_51)->bonds_0 = bonds_20;
    (&kernelContext_51)->args_0 = args_20;
    (&kernelContext_51)->info_0 = info_20;
    (&kernelContext_51)->scan_0 = scan_20;
    (&kernelContext_51)->parts_0 = parts_20;
    (&kernelContext_51)->colliders_0 = colliders_20;
    (&kernelContext_51)->collider_info_0 = collider_info_20;
    (&kernelContext_51)->view_chunks_0 = view_chunks_20;
    (&kernelContext_51)->view_0 = view_20;
    (&kernelContext_51)->objects_0 = objects_20;
    (&kernelContext_51)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_51)->lines_0 = lines_20;
    (&kernelContext_51)->geometry_0 = geometry_20;
    (&kernelContext_51)->shots_0 = shots_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_51)->g_scan_0 = &g_scan_20;
    uint k_4 = id_17.x;
    bool _S329 = live_0(k_4, &kernelContext_51);
    if(!_S329)
    {
        return;
    }
    uint _S330 = body_of_0(k_4, &kernelContext_51);
    uint b_3 = 7U * _S330;
    thread Quat_0 q_15;
    thread float3 origin_3;
    structure_pose_0(_S330, b_3, &q_15, &origin_3, &kernelContext_51);
    uint _S331 = 3U * k_4;
    float4 _S332 = float4(*((&kernelContext_51)->view_chunks_0+_S331)) ;
    float3 base_2 = (float4(*((&kernelContext_51)->view_chunks_0+(_S331 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_51)->bodies_0+b_3)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_51)->view_0->mode_0) == 1U)
    {
        float _S333 = max(length((float4(*((&kernelContext_51)->bodies_0+(b_3 + 3U))) ).xyz), length((float4(*((&kernelContext_51)->bodies_0+(b_3 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S333 > ((&kernelContext_51)->view_0->rest_speed_0))
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
        if(((&kernelContext_51)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S330);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S330);
            }
        }
    }
    uint _S334 = (as_type<uint>((_S332.w)));
    Quat_0 _S335 = quat_of_0(float4(*((&kernelContext_51)->view_chunks_0+(_S331 + 1U))) );
    thread Quat_0 _S336 = q_15;
    thread Quat_0 _S337 = _S335;
    Quat_0 _S338 = quat_mul_0(&_S336, &_S337);
    thread Quat_0 _S339 = _S338;
    Quat_0 _S340 = normalized_0(&_S339);
    float3 _S341 = _S332.xyz;
    thread Quat_0 _S342 = q_15;
    float3 _S343 = rotate_0(&_S342, _S341);
    float3 _S344 = origin_3 + _S343;
    float4 _S345 = float4(color_1, 1.0f);
    thread Quat_0 _S346 = _S340;
    write_object_0(_S334, &_S346, _S344, _S345, &kernelContext_51);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_52;
    (&kernelContext_52)->params_0 = params_21;
    (&kernelContext_52)->work_0 = work_21;
    (&kernelContext_52)->frames_0 = frames_21;
    (&kernelContext_52)->bodies_0 = bodies_21;
    (&kernelContext_52)->chunks_0 = chunks_21;
    (&kernelContext_52)->sort_0 = sort_21;
    (&kernelContext_52)->atom_0 = atom_21;
    (&kernelContext_52)->manifolds_0 = manifolds_21;
    (&kernelContext_52)->loads_0 = loads_21;
    (&kernelContext_52)->bonds_0 = bonds_21;
    (&kernelContext_52)->args_0 = args_21;
    (&kernelContext_52)->info_0 = info_21;
    (&kernelContext_52)->scan_0 = scan_21;
    (&kernelContext_52)->parts_0 = parts_21;
    (&kernelContext_52)->colliders_0 = colliders_21;
    (&kernelContext_52)->collider_info_0 = collider_info_21;
    (&kernelContext_52)->view_chunks_0 = view_chunks_21;
    (&kernelContext_52)->view_0 = view_21;
    (&kernelContext_52)->objects_0 = objects_21;
    (&kernelContext_52)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_52)->lines_0 = lines_21;
    (&kernelContext_52)->geometry_0 = geometry_21;
    (&kernelContext_52)->shots_0 = shots_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_52)->g_scan_0 = &g_scan_21;
    uint i_3 = id_18.x;
    if(i_3 >= (view_21->shot_count_1))
    {
        return;
    }
    uint b_4 = 7U * (&kernelContext_52)->shot_bodies_0[i_3];
    uint o_2 = (&kernelContext_52)->view_0->impactor_base_0 + i_3;
    if(((float4(*((&kernelContext_52)->bodies_0+b_4)) ).w) == 0.0f)
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
            *((&kernelContext_52)->objects_0+(4U * o_2 + w_4)) = packed_float4(float4(0.0f) ) ;
            w_4 = w_4 + 1U;
        }
        return;
    }
    thread Quat_0 _S347 = quat_of_0(float4(*((&kernelContext_52)->bodies_0+(b_4 + 1U))) );
    Quat_0 _S348 = rot_from_rubble_0(&_S347);
    float3 _S349 = from_rubble_0((float4(*((&kernelContext_52)->bodies_0+b_4)) ).xyz);
    float4 _S350 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S351 = _S348;
    write_object_0(o_2, &_S351, _S349, _S350, &kernelContext_52);
    return;
}

float3 chunk_world_0(uint k_5, KernelContext_0 thread* kernelContext_53)
{
    uint _S352 = body_of_0(k_5, kernelContext_53);
    thread Quat_0 q_16;
    thread float3 origin_4;
    structure_pose_0(_S352, 7U * _S352, &q_16, &origin_4, kernelContext_53);
    float3 _S353 = (float4(*(kernelContext_53->view_chunks_0+3U * k_5)) ).xyz;
    thread Quat_0 _S354 = q_16;
    float3 _S355 = rotate_0(&_S354, _S353);
    return origin_4 + _S355;
}

void write_line_vertex_0(uint v_8, float3 p_4, KernelContext_0 thread* kernelContext_54)
{
    uint _S356 = 7U * v_8;
    *(kernelContext_54->lines_0+_S356) = p_4.x;
    *(kernelContext_54->lines_0+(_S356 + 1U)) = p_4.y;
    *(kernelContext_54->lines_0+(_S356 + 2U)) = p_4.z;
    *(kernelContext_54->lines_0+(_S356 + 3U)) = 1.0f;
    *(kernelContext_54->lines_0+(_S356 + 4U)) = 0.86000001430511475f;
    *(kernelContext_54->lines_0+(_S356 + 5U)) = 0.25f;
    *(kernelContext_54->lines_0+(_S356 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_55;
    (&kernelContext_55)->params_0 = params_22;
    (&kernelContext_55)->work_0 = work_22;
    (&kernelContext_55)->frames_0 = frames_22;
    (&kernelContext_55)->bodies_0 = bodies_22;
    (&kernelContext_55)->chunks_0 = chunks_22;
    (&kernelContext_55)->sort_0 = sort_22;
    (&kernelContext_55)->atom_0 = atom_22;
    (&kernelContext_55)->manifolds_0 = manifolds_22;
    (&kernelContext_55)->loads_0 = loads_22;
    (&kernelContext_55)->bonds_0 = bonds_22;
    (&kernelContext_55)->args_0 = args_22;
    (&kernelContext_55)->info_0 = info_22;
    (&kernelContext_55)->scan_0 = scan_22;
    (&kernelContext_55)->parts_0 = parts_22;
    (&kernelContext_55)->colliders_0 = colliders_22;
    (&kernelContext_55)->collider_info_0 = collider_info_22;
    (&kernelContext_55)->view_chunks_0 = view_chunks_22;
    (&kernelContext_55)->view_0 = view_22;
    (&kernelContext_55)->objects_0 = objects_22;
    (&kernelContext_55)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_55)->lines_0 = lines_22;
    (&kernelContext_55)->geometry_0 = geometry_22;
    (&kernelContext_55)->shots_0 = shots_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_55)->g_scan_0 = &g_scan_22;
    uint i_4 = id_19.x;
    if(i_4 >= (params_22->bond_used_0))
    {
        return;
    }
    uint4 _S357 = uint4(*((&kernelContext_55)->bonds_0+i_4)) ;
    float3 _S358 = chunk_world_0(_S357.x, &kernelContext_55);
    uint _S359 = w_alive_0(&kernelContext_55);
    float3 b_5;
    if((*((&kernelContext_55)->work_0+(_S359 + i_4))) != 0U)
    {
        float3 _S360 = chunk_world_0(_S357.y, &kernelContext_55);
        b_5 = _S360;
    }
    else
    {
        b_5 = _S358;
    }
    uint _S361 = 2U * i_4;
    write_line_vertex_0(_S361, _S358, &kernelContext_55);
    write_line_vertex_0(_S361 + 1U, b_5, &kernelContext_55);
    return;
}

uint a_strike_0(KernelContext_0 thread* kernelContext_56)
{
    uint _S362 = a_counters_1(kernelContext_56);
    return _S362 + 16U;
}

uint strike_at_0(uint b_6, KernelContext_0 thread* kernelContext_57)
{
    return 2U * kernelContext_57->params_0->body_count_0 + 9U * kernelContext_57->params_0->chunk_count_0 + 3U * b_6;
}

[[kernel]] void fr_strike_reset(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], atomic<uint> device* atom_23 [[buffer(5)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_58;
    (&kernelContext_58)->params_0 = params_23;
    (&kernelContext_58)->work_0 = work_23;
    (&kernelContext_58)->frames_0 = frames_23;
    (&kernelContext_58)->bodies_0 = bodies_23;
    (&kernelContext_58)->chunks_0 = chunks_23;
    (&kernelContext_58)->sort_0 = sort_23;
    (&kernelContext_58)->atom_0 = atom_23;
    (&kernelContext_58)->manifolds_0 = manifolds_23;
    (&kernelContext_58)->loads_0 = loads_23;
    (&kernelContext_58)->bonds_0 = bonds_23;
    (&kernelContext_58)->args_0 = args_23;
    (&kernelContext_58)->info_0 = info_23;
    (&kernelContext_58)->scan_0 = scan_23;
    (&kernelContext_58)->parts_0 = parts_23;
    (&kernelContext_58)->colliders_0 = colliders_23;
    (&kernelContext_58)->collider_info_0 = collider_info_23;
    (&kernelContext_58)->view_chunks_0 = view_chunks_23;
    (&kernelContext_58)->view_0 = view_23;
    (&kernelContext_58)->objects_0 = objects_23;
    (&kernelContext_58)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_58)->lines_0 = lines_23;
    (&kernelContext_58)->geometry_0 = geometry_23;
    (&kernelContext_58)->shots_0 = shots_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_58)->g_scan_0 = &g_scan_23;
    uint b_7 = id_20.x;
    if(b_7 >= (params_23->body_count_0))
    {
        return;
    }
    uint _S363 = a_strike_0(&kernelContext_58);
    uint at_5 = _S363 + 9U * b_7;
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
        atomic_store_explicit((&kernelContext_58)->atom_0+(at_5 + w_5), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_58)->atom_0+(at_5 + 3U + w_5), 0U, memory_order_relaxed);
        w_5 = w_5 + 1U;
    }
    atomic_store_explicit((&kernelContext_58)->atom_0+(at_5 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_58)->atom_0+(at_5 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_58)->atom_0+(at_5 + 8U), 4294967295U, memory_order_relaxed);
    uint _S364 = strike_at_0(b_7, &kernelContext_58);
    (*((&kernelContext_58)->frames_0+_S364))[int(3)] = 0.0f;
    return;
}

float3 chunk_box_world_0(uint k_6, Quat_0 thread* q_17, KernelContext_0 thread* kernelContext_59)
{
    uint _S365 = body_of_0(k_6, kernelContext_59);
    thread float3 origin_5;
    structure_pose_0(_S365, 7U * _S365, q_17, &origin_5, kernelContext_59);
    float3 _S366 = (float4(*(kernelContext_59->geometry_0+3U * k_6)) ).xyz;
    thread Quat_0 _S367 = *q_17;
    float3 _S368 = rotate_0(&_S367, _S366);
    return origin_5 + _S368;
}

uint ordered_0(float f_8)
{
    uint u_0 = (as_type<uint>((f_8)));
    uint _S369;
    if((u_0 & 2147483648U) != 0U)
    {
        _S369 = ~u_0;
    }
    else
    {
        _S369 = u_0 | 2147483648U;
    }
    return _S369;
}

[[kernel]] void fr_strike_bounds(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], atomic<uint> device* atom_24 [[buffer(5)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_60;
    (&kernelContext_60)->params_0 = params_24;
    (&kernelContext_60)->work_0 = work_24;
    (&kernelContext_60)->frames_0 = frames_24;
    (&kernelContext_60)->bodies_0 = bodies_24;
    (&kernelContext_60)->chunks_0 = chunks_24;
    (&kernelContext_60)->sort_0 = sort_24;
    (&kernelContext_60)->atom_0 = atom_24;
    (&kernelContext_60)->manifolds_0 = manifolds_24;
    (&kernelContext_60)->loads_0 = loads_24;
    (&kernelContext_60)->bonds_0 = bonds_24;
    (&kernelContext_60)->args_0 = args_24;
    (&kernelContext_60)->info_0 = info_24;
    (&kernelContext_60)->scan_0 = scan_24;
    (&kernelContext_60)->parts_0 = parts_24;
    (&kernelContext_60)->colliders_0 = colliders_24;
    (&kernelContext_60)->collider_info_0 = collider_info_24;
    (&kernelContext_60)->view_chunks_0 = view_chunks_24;
    (&kernelContext_60)->view_0 = view_24;
    (&kernelContext_60)->objects_0 = objects_24;
    (&kernelContext_60)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_60)->lines_0 = lines_24;
    (&kernelContext_60)->geometry_0 = geometry_24;
    (&kernelContext_60)->shots_0 = shots_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_60)->g_scan_0 = &g_scan_24;
    uint k_7 = id_21.x;
    bool _S370 = live_0(k_7, &kernelContext_60);
    bool _S371;
    if(!_S370)
    {
        _S371 = true;
    }
    else
    {
        _S371 = ((&kernelContext_60)->params_0->shot_count_0) == 0U;
    }
    if(_S371)
    {
        return;
    }
    thread Quat_0 q_18;
    float3 _S372 = chunk_box_world_0(k_7, &q_18, &kernelContext_60);
    float _S373 = (float4(*((&kernelContext_60)->geometry_0+3U * k_7)) ).w;
    uint _S374 = a_strike_0(&kernelContext_60);
    uint _S375 = body_of_0(k_7, &kernelContext_60);
    uint _S376 = _S374 + 9U * _S375;
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
        uint _S377 = ax_0;
        uint _S378 = atomic_fetch_min_explicit((&kernelContext_60)->atom_0+(_S376 + ax_0), ordered_0(_S372[ax_0] - _S373), memory_order_relaxed);
        uint _S379 = atomic_fetch_max_explicit((&kernelContext_60)->atom_0+(_S376 + 3U + ax_0), ordered_0(_S372[_S377] + _S373), memory_order_relaxed);
        ax_0 = ax_0 + 1U;
    }
    return;
}

float unordered_0(uint u_1)
{
    uint _S380;
    if((u_1 & 2147483648U) != 0U)
    {
        _S380 = u_1 & 2147483647U;
    }
    else
    {
        _S380 = ~u_1;
    }
    return (as_type<float>((_S380)));
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
    bool _S381;
    if(b_8 >= (params_25->body_count_0))
    {
        _S381 = true;
    }
    else
    {
        _S381 = ((&kernelContext_61)->params_0->shot_count_0) == 0U;
    }
    if(_S381)
    {
        return;
    }
    uint _S382 = a_strike_0(&kernelContext_61);
    uint at_6 = _S382 + 9U * b_8;
    uint _S383 = atomic_load_explicit((&kernelContext_61)->atom_0+at_6, memory_order_relaxed);
    if(_S383 == 4294967295U)
    {
        return;
    }
    uint _S384 = atomic_load_explicit((&kernelContext_61)->atom_0+at_6, memory_order_relaxed);
    float _S385 = unordered_0(_S384);
    uint _S386 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 1U), memory_order_relaxed);
    float _S387 = unordered_0(_S386);
    uint _S388 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 2U), memory_order_relaxed);
    float3 _S389 = float3(_S385, _S387, unordered_0(_S388));
    uint _S390 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 3U), memory_order_relaxed);
    float _S391 = unordered_0(_S390);
    uint _S392 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 4U), memory_order_relaxed);
    float _S393 = unordered_0(_S392);
    uint _S394 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 5U), memory_order_relaxed);
    float3 _S395 = float3(_S391, _S393, unordered_0(_S394));
    float3 _S396 = from_rubble_0((float4(*((&kernelContext_61)->bodies_0+(7U * b_8 + 3U))) ).xyz);
    float _S397 = float_bits_0(2139095040U);
    uint k_8 = 0U;
    for(;;)
    {
        if(k_8 < ((&kernelContext_61)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S398 = float4(*((&kernelContext_61)->shots_0+2U * k_8)) ;
        uint sb_0 = 7U * (as_type<uint>((_S398.x)));
        if(((float4(*((&kernelContext_61)->bodies_0+sb_0)) ).w) == 0.0f)
        {
            k_8 = k_8 + 1U;
            continue;
        }
        float3 c_15 = from_rubble_0((float4(*((&kernelContext_61)->bodies_0+sb_0)) ).xyz);
        float3 rel_0 = from_rubble_0((float4(*((&kernelContext_61)->bodies_0+(sb_0 + 3U))) ).xyz) - _S396;
        if(!((length(rel_0)) > ((&kernelContext_61)->params_0->strike_speed_0)))
        {
            k_8 = k_8 + 1U;
            continue;
        }
        float3 _S399 = float3(_S398.y) ;
        float3 _S400 = _S389 - _S399;
        float3 _S401 = _S395 + _S399;
        float _S402 = (&kernelContext_61)->params_0->dt_0;
        float t0_0 = 0.0f;
        float t1_0 = _S402;
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
            uint _S403 = ax_1;
            if((abs(rel_0[ax_1])) < 9.999999960041972e-13f)
            {
                uint _S404 = ax_1;
                if((c_15[ax_1]) < (_S400[ax_1]))
                {
                    _S381 = true;
                }
                else
                {
                    _S381 = (c_15[_S404]) > (_S401[ax_1]);
                }
                float t0_1;
                if(_S381)
                {
                    t0_1 = _S397;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_1 = ax_1 + 1U;
                continue;
            }
            float ta_0 = (_S400[ax_1] - c_15[ax_1]) / rel_0[_S403];
            float tb_0 = (_S401[ax_1] - c_15[ax_1]) / rel_0[_S403];
            float _S405 = min(t1_0, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_0 = _S405;
            ax_1 = ax_1 + 1U;
        }
        if(t0_0 <= t1_0)
        {
            uint _S406 = strike_at_0(b_8, &kernelContext_61);
            *((&kernelContext_61)->frames_0+_S406) = packed_float4(float4(c_15, float(k_8 + 1U))) ;
            *((&kernelContext_61)->frames_0+(_S406 + 1U)) = packed_float4(float4(rel_0, t0_0)) ;
            return;
        }
        k_8 = k_8 + 1U;
    }
    return;
}

bool struck_0(uint b_9, uint thread* shot_0, KernelContext_0 thread* kernelContext_62)
{
    uint _S407 = strike_at_0(b_9, kernelContext_62);
    uint k_9 = uint((float4(*(kernelContext_62->frames_0+_S407)) ).w);
    *shot_0 = k_9 - 1U;
    return k_9 != 0U;
}

[[kernel]] void fr_strike_entry(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], atomic<uint> device* atom_26 [[buffer(5)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_63;
    (&kernelContext_63)->params_0 = params_26;
    (&kernelContext_63)->work_0 = work_26;
    (&kernelContext_63)->frames_0 = frames_26;
    (&kernelContext_63)->bodies_0 = bodies_26;
    (&kernelContext_63)->chunks_0 = chunks_26;
    (&kernelContext_63)->sort_0 = sort_26;
    (&kernelContext_63)->atom_0 = atom_26;
    (&kernelContext_63)->manifolds_0 = manifolds_26;
    (&kernelContext_63)->loads_0 = loads_26;
    (&kernelContext_63)->bonds_0 = bonds_26;
    (&kernelContext_63)->args_0 = args_26;
    (&kernelContext_63)->info_0 = info_26;
    (&kernelContext_63)->scan_0 = scan_26;
    (&kernelContext_63)->parts_0 = parts_26;
    (&kernelContext_63)->colliders_0 = colliders_26;
    (&kernelContext_63)->collider_info_0 = collider_info_26;
    (&kernelContext_63)->view_chunks_0 = view_chunks_26;
    (&kernelContext_63)->view_0 = view_26;
    (&kernelContext_63)->objects_0 = objects_26;
    (&kernelContext_63)->shot_bodies_0 = shot_bodies_26;
    (&kernelContext_63)->lines_0 = lines_26;
    (&kernelContext_63)->geometry_0 = geometry_26;
    (&kernelContext_63)->shots_0 = shots_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_63)->g_scan_0 = &g_scan_26;
    uint k_10 = id_23.x;
    bool _S408 = live_0(k_10, &kernelContext_63);
    bool _S409;
    if(!_S408)
    {
        _S409 = true;
    }
    else
    {
        _S409 = ((&kernelContext_63)->params_0->shot_count_0) == 0U;
    }
    if(_S409)
    {
        return;
    }
    uint _S410 = body_of_0(k_10, &kernelContext_63);
    thread uint shot_1;
    bool _S411 = struck_0(_S410, &shot_1, &kernelContext_63);
    if(!_S411)
    {
        return;
    }
    uint _S412 = strike_at_0(_S410, &kernelContext_63);
    float4 _S413 = float4(*((&kernelContext_63)->frames_0+(_S412 + 1U))) ;
    float3 from_1 = (float4(*((&kernelContext_63)->frames_0+_S412)) ).xyz;
    float3 rel_1 = _S413.xyz;
    float3 d_2 = rel_1 / float3(max(length(rel_1), 9.999999960041972e-13f)) ;
    float _S414 = (float4(*((&kernelContext_63)->shots_0+2U * shot_1)) ).y;
    thread Quat_0 q_19;
    float3 _S415 = chunk_box_world_0(k_10, &q_19, &kernelContext_63);
    uint _S416 = 3U * k_10;
    Quat_0 _S417 = quat_of_0(float4(*((&kernelContext_63)->geometry_0+(_S416 + 2U))) );
    thread Quat_0 _S418 = q_19;
    thread Quat_0 _S419 = _S417;
    Quat_0 _S420 = quat_mul_0(&_S418, &_S419);
    thread Quat_0 _S421 = _S420;
    Quat_0 _S422 = normalized_0(&_S421);
    float3 _S423 = (float4(*((&kernelContext_63)->geometry_0+(_S416 + 1U))) ).xyz;
    float _S424 = float_bits_0(2139095040U);
    float _S425 = (&kernelContext_63)->params_0->dt_0;
    float t_in_0 = 0.0f;
    float t_out_0 = _S425;
    uint ax_2 = 0U;
    for(;;)
    {
        if(ax_2 < 3U)
        {
        }
        else
        {
            break;
        }
        thread float3 e_1 = float3(0.0f) ;
        e_1[ax_2] = 1.0f;
        thread Quat_0 _S426 = _S422;
        float3 _S427 = rotate_0(&_S426, e_1);
        float o_3 = dot(from_1 - _S415, _S427);
        float dv_0 = dot(rel_1, _S427);
        float ext_0 = _S423[ax_2] + _S414;
        if((abs(dv_0)) < 9.999999960041972e-13f)
        {
            float t_in_1;
            if((abs(o_3)) > ext_0)
            {
                t_in_1 = _S424;
            }
            else
            {
                t_in_1 = t_in_0;
            }
            t_in_0 = t_in_1;
            ax_2 = ax_2 + 1U;
            continue;
        }
        float ta_1 = (- ext_0 - o_3) / dv_0;
        float tb_1 = (ext_0 - o_3) / dv_0;
        float _S428 = min(t_out_0, max(ta_1, tb_1));
        t_in_0 = max(t_in_0, min(ta_1, tb_1));
        t_out_0 = _S428;
        ax_2 = ax_2 + 1U;
    }
    uint _S429 = a_strike_0(&kernelContext_63);
    uint at_7 = _S429 + 9U * _S410;
    if(t_in_0 <= t_out_0)
    {
        uint _S430 = atomic_fetch_min_explicit((&kernelContext_63)->atom_0+(at_7 + 6U), (as_type<uint>((t_in_0))), memory_order_relaxed);
    }
    float3 r_7 = _S415 - (from_1 + rel_1 * float3(_S413.w) );
    float along_0 = dot(r_7, d_2);
    uint _S431 = atomic_fetch_min_explicit((&kernelContext_63)->atom_0+(at_7 + 7U), (as_type<uint>((length(r_7 - d_2 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f))), memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], atomic<uint> device* atom_27 [[buffer(5)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_64;
    (&kernelContext_64)->params_0 = params_27;
    (&kernelContext_64)->work_0 = work_27;
    (&kernelContext_64)->frames_0 = frames_27;
    (&kernelContext_64)->bodies_0 = bodies_27;
    (&kernelContext_64)->chunks_0 = chunks_27;
    (&kernelContext_64)->sort_0 = sort_27;
    (&kernelContext_64)->atom_0 = atom_27;
    (&kernelContext_64)->manifolds_0 = manifolds_27;
    (&kernelContext_64)->loads_0 = loads_27;
    (&kernelContext_64)->bonds_0 = bonds_27;
    (&kernelContext_64)->args_0 = args_27;
    (&kernelContext_64)->info_0 = info_27;
    (&kernelContext_64)->scan_0 = scan_27;
    (&kernelContext_64)->parts_0 = parts_27;
    (&kernelContext_64)->colliders_0 = colliders_27;
    (&kernelContext_64)->collider_info_0 = collider_info_27;
    (&kernelContext_64)->view_chunks_0 = view_chunks_27;
    (&kernelContext_64)->view_0 = view_27;
    (&kernelContext_64)->objects_0 = objects_27;
    (&kernelContext_64)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_64)->lines_0 = lines_27;
    (&kernelContext_64)->geometry_0 = geometry_27;
    (&kernelContext_64)->shots_0 = shots_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_64)->g_scan_0 = &g_scan_27;
    uint k_11 = id_24.x;
    bool _S432 = live_0(k_11, &kernelContext_64);
    bool _S433;
    if(!_S432)
    {
        _S433 = true;
    }
    else
    {
        _S433 = ((&kernelContext_64)->params_0->shot_count_0) == 0U;
    }
    if(_S433)
    {
        return;
    }
    uint _S434 = body_of_0(k_11, &kernelContext_64);
    thread uint shot_2;
    bool _S435 = struck_0(_S434, &shot_2, &kernelContext_64);
    if(!_S435)
    {
        return;
    }
    uint _S436 = strike_at_0(_S434, &kernelContext_64);
    float4 _S437 = float4(*((&kernelContext_64)->frames_0+_S436)) ;
    float4 _S438 = float4(*((&kernelContext_64)->frames_0+(_S436 + 1U))) ;
    float3 _S439 = _S438.xyz;
    float3 d_3 = _S439 / float3(max(length(_S439), 9.999999960041972e-13f)) ;
    thread Quat_0 q_20;
    float3 _S440 = chunk_box_world_0(k_11, &q_20, &kernelContext_64);
    float3 r_8 = _S440 - (_S437.xyz + _S439 * float3(_S438.w) );
    float along_1 = dot(r_8, d_3);
    float dist_0 = length(r_8 - d_3 * float3(along_1) ) + abs(along_1) * 0.00100000004749745f;
    uint _S441 = a_strike_0(&kernelContext_64);
    uint at_8 = _S441 + 9U * _S434;
    uint _S442 = (as_type<uint>((dist_0)));
    uint _S443 = atomic_load_explicit((&kernelContext_64)->atom_0+(at_8 + 7U), memory_order_relaxed);
    if(_S442 == _S443)
    {
        uint _S444 = atomic_fetch_min_explicit((&kernelContext_64)->atom_0+(at_8 + 8U), k_11, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], atomic<uint> device* atom_28 [[buffer(5)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_65;
    (&kernelContext_65)->params_0 = params_28;
    (&kernelContext_65)->work_0 = work_28;
    (&kernelContext_65)->frames_0 = frames_28;
    (&kernelContext_65)->bodies_0 = bodies_28;
    (&kernelContext_65)->chunks_0 = chunks_28;
    (&kernelContext_65)->sort_0 = sort_28;
    (&kernelContext_65)->atom_0 = atom_28;
    (&kernelContext_65)->manifolds_0 = manifolds_28;
    (&kernelContext_65)->loads_0 = loads_28;
    (&kernelContext_65)->bonds_0 = bonds_28;
    (&kernelContext_65)->args_0 = args_28;
    (&kernelContext_65)->info_0 = info_28;
    (&kernelContext_65)->scan_0 = scan_28;
    (&kernelContext_65)->parts_0 = parts_28;
    (&kernelContext_65)->colliders_0 = colliders_28;
    (&kernelContext_65)->collider_info_0 = collider_info_28;
    (&kernelContext_65)->view_chunks_0 = view_chunks_28;
    (&kernelContext_65)->view_0 = view_28;
    (&kernelContext_65)->objects_0 = objects_28;
    (&kernelContext_65)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_65)->lines_0 = lines_28;
    (&kernelContext_65)->geometry_0 = geometry_28;
    (&kernelContext_65)->shots_0 = shots_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_65)->g_scan_0 = &g_scan_28;
    uint b_10 = id_25.x;
    bool _S445;
    if(b_10 >= (params_28->body_count_0))
    {
        _S445 = true;
    }
    else
    {
        _S445 = ((&kernelContext_65)->params_0->shot_count_0) == 0U;
    }
    if(_S445)
    {
        return;
    }
    thread uint shot_3;
    bool _S446 = struck_0(b_10, &shot_3, &kernelContext_65);
    if(!_S446)
    {
        return;
    }
    uint _S447 = a_strike_0(&kernelContext_65);
    uint at_9 = _S447 + 9U * b_10;
    uint _S448 = atomic_load_explicit((&kernelContext_65)->atom_0+(at_9 + 6U), memory_order_relaxed);
    float t_in_2 = (as_type<float>((_S448)));
    uint entry_0 = atomic_load_explicit((&kernelContext_65)->atom_0+(at_9 + 8U), memory_order_relaxed);
    uint _S449 = strike_at_0(b_10, &kernelContext_65);
    if(!(t_in_2 <= ((&kernelContext_65)->params_0->dt_0)))
    {
        _S445 = true;
    }
    else
    {
        _S445 = entry_0 == 4294967295U;
    }
    if(_S445)
    {
        (*((&kernelContext_65)->frames_0+_S449))[int(3)] = 0.0f;
        return;
    }
    float3 rel_2 = (float4(*((&kernelContext_65)->frames_0+(_S449 + 1U))) ).xyz;
    float speed_0 = length(rel_2);
    float4 _S450 = float4(*((&kernelContext_65)->shots_0+2U * shot_3)) ;
    float _S451 = max(_S450.z, 0.00100000004749745f);
    float m_imp_0 = _S450.w;
    float e_imp_0 = (float4(*((&kernelContext_65)->shots_0+(2U * shot_3 + 1U))) ).x;
    float3 contact_0 = (float4(*((&kernelContext_65)->frames_0+_S449)) ).xyz + rel_2 * float3(t_in_2)  + rel_2 / float3(max(speed_0, 9.999999960041972e-13f))  * float3(_S450.y) ;
    float inv_mass_3 = (float4(*((&kernelContext_65)->bodies_0+7U * b_10)) ).w;
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
    float modulus_0 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_65)->geometry_0+(3U * entry_0 + 1U))) ).w);
    *((&kernelContext_65)->frames_0+(_S449 + 2U)) = packed_float4(float4(contact_0, m_eff_0 * speed_0 / (2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (_S451 * modulus_0 * modulus_0 * max(speed_0, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

[[kernel]] void fr_strike_breaks(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], atomic<uint> device* atom_29 [[buffer(5)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]])
{
    thread KernelContext_0 kernelContext_66;
    (&kernelContext_66)->params_0 = params_29;
    (&kernelContext_66)->work_0 = work_29;
    (&kernelContext_66)->frames_0 = frames_29;
    (&kernelContext_66)->bodies_0 = bodies_29;
    (&kernelContext_66)->chunks_0 = chunks_29;
    (&kernelContext_66)->sort_0 = sort_29;
    (&kernelContext_66)->atom_0 = atom_29;
    (&kernelContext_66)->manifolds_0 = manifolds_29;
    (&kernelContext_66)->loads_0 = loads_29;
    (&kernelContext_66)->bonds_0 = bonds_29;
    (&kernelContext_66)->args_0 = args_29;
    (&kernelContext_66)->info_0 = info_29;
    (&kernelContext_66)->scan_0 = scan_29;
    (&kernelContext_66)->parts_0 = parts_29;
    (&kernelContext_66)->colliders_0 = colliders_29;
    (&kernelContext_66)->collider_info_0 = collider_info_29;
    (&kernelContext_66)->view_chunks_0 = view_chunks_29;
    (&kernelContext_66)->view_0 = view_29;
    (&kernelContext_66)->objects_0 = objects_29;
    (&kernelContext_66)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_66)->lines_0 = lines_29;
    (&kernelContext_66)->geometry_0 = geometry_29;
    (&kernelContext_66)->shots_0 = shots_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_66)->g_scan_0 = &g_scan_29;
    uint i_5 = id_26.x;
    bool _S452;
    if(i_5 >= (params_29->bond_used_0))
    {
        _S452 = true;
    }
    else
    {
        _S452 = ((&kernelContext_66)->params_0->shot_count_0) == 0U;
    }
    if(_S452)
    {
        _S452 = true;
    }
    else
    {
        uint _S453 = w_alive_0(&kernelContext_66);
        _S452 = (*((&kernelContext_66)->work_0+(_S453 + i_5))) == 0U;
    }
    if(_S452)
    {
        return;
    }
    uint4 _S454 = uint4(*((&kernelContext_66)->bonds_0+i_5)) ;
    uint _S455 = _S454.x;
    uint _S456 = body_of_0(_S455, &kernelContext_66);
    thread uint shot_4;
    bool _S457 = struck_0(_S456, &shot_4, &kernelContext_66);
    if(!_S457)
    {
        return;
    }
    uint _S458 = strike_at_0(_S456, &kernelContext_66);
    float4 _S459 = float4(*((&kernelContext_66)->frames_0+(_S458 + 2U))) ;
    thread Quat_0 q_21;
    float3 _S460 = chunk_box_world_0(_S455, &q_21, &kernelContext_66);
    float3 _S461 = chunk_box_world_0(_S454.y, &q_21, &kernelContext_66);
    float dist_1 = length(float3(0.5f)  * (_S460 + _S461) - _S459.xyz);
    if((_S459.w * min(1.0f, (as_type<float>((_S454.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((_S454.z))))
    {
        uint _S462 = w_alive_0(&kernelContext_66);
        *((&kernelContext_66)->work_0+(_S462 + i_5)) = 0U;
        uint _S463 = a_dirty_0(&kernelContext_66);
        uint _S464 = atomic_fetch_or_explicit((&kernelContext_66)->atom_0+(_S463 + _S456), 1U, memory_order_relaxed);
        uint _S465 = a_counters_0(&kernelContext_66);
        uint _S466 = atomic_fetch_add_explicit((&kernelContext_66)->atom_0+_S465, 1U, memory_order_relaxed);
        uint _S467 = atomic_fetch_add_explicit((&kernelContext_66)->atom_0+(_S465 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

