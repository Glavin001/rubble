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
    float pad1_0;
    float pad2_0;
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
    uint shot_count_0;
    uint impactor_base_0;
    uint pad0_0;
    float rest_speed_0;
    float pad1_1;
    float pad2_1;
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

void structure_pose_0(uint slot_0, uint b_0, Quat_0 thread* q_5, float3 thread* origin_0, KernelContext_0 thread* kernelContext_1)
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
    *q_5 = _S14;
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

uint l_pool_0(uint half_0, KernelContext_0 thread* kernelContext_16)
{
    return kernelContext_16->params_0->chunk_count_0 + 3U * kernelContext_16->params_0->pool_cap_0 * half_0;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S30 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S30, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S31 = max(0.0f, _S30);
    float _S32 = min(dt_1, _S30 + duration_0);
    bool _S33;
    if(_S32 > _S31)
    {
        _S33 = duration_0 > 0.0f;
    }
    else
    {
        _S33 = false;
    }
    float sum_1;
    if(_S33)
    {
        float w_1 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_1 * (cos(w_1 * (_S31 - _S30)) - cos(w_1 * (_S32 - _S30)));
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

[[kernel]] void fr_loads(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], uint device* sort_1 [[buffer(11)]], atomic<uint> device* atom_1 [[buffer(5)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], packed_float4 device* loads_1 [[buffer(12)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]], packed_float4 device* view_chunks_1 [[buffer(17)]], ViewParams_0 constant* view_1 [[buffer(16)]], packed_float4 device* objects_1 [[buffer(18)]], uint device* shot_bodies_1 [[buffer(20)]], float device* lines_1 [[buffer(19)]])
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
    thread KernelContext_0 kernelContext_17;
    (&kernelContext_17)->params_0 = params_1;
    (&kernelContext_17)->work_0 = work_1;
    (&kernelContext_17)->frames_0 = frames_1;
    (&kernelContext_17)->bodies_0 = bodies_1;
    (&kernelContext_17)->chunks_0 = chunks_1;
    (&kernelContext_17)->sort_0 = sort_1;
    (&kernelContext_17)->atom_0 = atom_1;
    (&kernelContext_17)->manifolds_0 = manifolds_1;
    (&kernelContext_17)->loads_0 = loads_1;
    (&kernelContext_17)->bonds_0 = bonds_1;
    (&kernelContext_17)->args_0 = args_1;
    (&kernelContext_17)->info_0 = info_1;
    (&kernelContext_17)->scan_0 = scan_1;
    (&kernelContext_17)->parts_0 = parts_1;
    (&kernelContext_17)->colliders_0 = colliders_1;
    (&kernelContext_17)->collider_info_0 = collider_info_1;
    (&kernelContext_17)->view_chunks_0 = view_chunks_1;
    (&kernelContext_17)->view_0 = view_1;
    (&kernelContext_17)->objects_0 = objects_1;
    (&kernelContext_17)->shot_bodies_0 = shot_bodies_1;
    (&kernelContext_17)->lines_0 = lines_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_17)->g_scan_0 = &g_scan_1;
    uint k_0 = id_0.x;
    if(k_0 >= (params_1->chunk_count_0))
    {
        return;
    }
    uint _S34 = body_of_0(k_0, &kernelContext_17);
    uint b_1 = 7U * _S34;
    thread Quat_0 q_6;
    thread float3 origin_1;
    structure_pose_0(_S34, b_1, &q_6, &origin_1, &kernelContext_17);
    uint _S35 = 4U * k_0;
    float4 _S36 = float4(*((&kernelContext_17)->chunks_0+_S35)) ;
    float3 _S37 = _S36.xyz;
    thread Quat_0 _S38 = q_6;
    float3 _S39 = rotate_0(&_S38, _S37);
    float3 centre_0 = origin_1 + _S39;
    float inv_mass_0 = (float4(*((&kernelContext_17)->bodies_0+b_1)) ).w;
    float m_body_0;
    if(inv_mass_0 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_0;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint from_0 = (&kernelContext_17)->params_0->parity_0;
    uint to_0 = 1U - (&kernelContext_17)->params_0->parity_0;
    uint _S40 = w_pulses_0(&kernelContext_17);
    uint _S41 = _S40 + _S35;
    uint _S42 = _S41 + 2U * from_0;
    uint off_0 = *((&kernelContext_17)->work_0+_S42);
    uint device* _S43 = (&kernelContext_17)->work_0+(_S42 + 1U);
    uint carried_0 = *_S43;
    uint room_0 = *_S43 + (&kernelContext_17)->sort_0[params_1->chunk_count_0 + 1U + k_0];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S44 = a_counters_0(&kernelContext_17);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_17)->atom_0+(_S44 + 5U + to_0), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_17)->params_0->pool_cap_0))
        {
            uint _S45 = atomic_fetch_or_explicit((&kernelContext_17)->atom_0+(_S44 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_17)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_17)->params_0->pool_cap_0 - base_1;
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
        uint _S46 = base_0;
        base_0 = base_1;
        room_1 = _S46;
    }
    else
    {
        base_0 = 0U;
        room_1 = room_0;
    }
    uint _S47 = params_1->chunk_count_0;
    float _S48 = _S36.w;
    float4 _S49 = float4(*((&kernelContext_17)->chunks_0+(_S35 + 3U))) ;
    float _S50 = (&kernelContext_17)->params_0->band_0;
    float _S51 = (&kernelContext_17)->params_0->dt_0;
    bool _S52 = ((&kernelContext_17)->params_0->impact_peak_0) != 0U;
    uint _S53 = l_pool_0(from_0, &kernelContext_17);
    uint _S54 = _S53 + 3U * off_0;
    uint _S55 = l_pool_0(to_0, &kernelContext_17);
    uint _S56 = _S55 + 3U * base_0;
    for(;;)
    {
        float3 _S57 = float3(0.0f) ;
        uint _S58 = (&kernelContext_17)->sort_0[k_0];
        uint _S59 = (&kernelContext_17)->sort_0[_S47 + 1U + k_0];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_1 = 0U;
        f_total_0 = _S57;
        float3 t_total_1 = _S57;
        for(;;)
        {
            if(step_1 < _S59)
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
                if(q3_0 < _S59)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_17)->manifolds_0[_S58 + q3_0].ids_0) ).y + 1U;
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
            Record_natural_0 r_4 = (&kernelContext_17)->manifolds_0[_S58 + p_0];
            float4 _S60 = float4(r_4.impulse_0) ;
            float3 _S61 = _S60.xyz;
            float j_0 = length(_S61);
            if(j_0 == 0.0f)
            {
                uint _S62 = step_1 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S62;
                continue;
            }
            float m_other_0 = (float4(r_4.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S48;
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
            float _S63 = min(j_0, 2.0f * m_1 * max(_S60.w - _S50, 0.0f));
            if(_S63 < j_0)
            {
                float3 f_0 = _S61 / float3(j_0)  * float3(((j_0 - _S63) / _S51)) ;
                float3 _S64 = t_total_1 + cross((float4(r_4.point_0) ).xyz - centre_0, f_0);
                f_total_1 = f_total_0 + f_0;
                t_total_0 = _S64;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            uint _S62 = step_1 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S62;
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
            uint _S65 = _S54 + 3U * p_0;
            float4 _S66 = float4(*((&kernelContext_17)->loads_0+_S65)) ;
            thread float4 a_1 = _S66;
            float4 _S67 = float4(*((&kernelContext_17)->loads_0+(_S65 + 1U))) ;
            float4 _S68 = float4(*((&kernelContext_17)->loads_0+(_S65 + 2U))) ;
            float _S69 = _S67.w;
            float mean_0 = pulse_mean_0(_S66.x, _S66.y, _S66.z, _S66.w, _S69, _S51);
            if(mean_0 != 0.0f)
            {
                float3 f_1 = _S68.xyz * float3(mean_0) ;
                float3 _S70 = t_total_1 + cross(_S67.xyz - centre_0, f_1);
                f_total_0 = f_total_0 + f_1;
                t_total_1 = _S70;
            }
            if((a_1.x + a_1.z + _S69) > _S51)
            {
                a_1.x = a_1.x - _S51;
                uint _S71 = _S56 + 3U * kept_0;
                *((&kernelContext_17)->loads_0+_S71) = packed_float4(a_1) ;
                *((&kernelContext_17)->loads_0+(_S71 + 1U)) = packed_float4(_S67) ;
                *((&kernelContext_17)->loads_0+(_S71 + 2U)) = packed_float4(_S68) ;
                kept_0 = kept_0 + 1U;
            }
            p_0 = p_0 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S59)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_5 = (&kernelContext_17)->manifolds_0[_S58 + q3_0];
            float4 _S72 = float4(r_5.impulse_0) ;
            float3 _S73 = _S72.xyz;
            float j_1 = length(_S73);
            if(j_1 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S74 = float4(r_5.other_0) ;
            float m_other_1 = _S74.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S48;
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
            float _S75 = _S72.w;
            float _S76 = min(j_1, 2.0f * m_1 * max(_S75 - _S50, 0.0f));
            if(_S76 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S77 = _S49.y;
            float duration_1 = hertz_duration_0(m_1, min(_S74.z, _S49.z), 1.0f / ((1.0f - _S77 * _S77) / _S49.x + 1.0f / _S74.y), _S75);
            float peak_1 = 3.14159274101257324f * _S76 / (2.0f * duration_1);
            float3 dir_0 = _S73 / float3(j_1) ;
            float mean_1;
            if(_S52)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S51);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_2 = dir_0 * float3(mean_1) ;
                float3 _S78 = t_total_1 + cross((float4(r_5.point_0) ).xyz - centre_0, f_2);
                f_total_1 = f_total_0 + f_2;
                t_total_0 = _S78;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            uint kept_1;
            if(duration_1 > _S51)
            {
                if(kept_0 < room_1)
                {
                    uint _S79 = _S56 + 3U * kept_0;
                    *((&kernelContext_17)->loads_0+_S79) = packed_float4(float4(- _S51, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_17)->loads_0+(_S79 + 1U)) = packed_float4(float4((float4(r_5.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_17)->loads_0+(_S79 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
        uint _S80 = a_counters_0(&kernelContext_17);
        uint _S81 = atomic_fetch_or_explicit((&kernelContext_17)->atom_0+(_S80 + 3U), 8U, memory_order_relaxed);
    }
    uint _S82 = _S41 + 2U * to_0;
    *((&kernelContext_17)->work_0+_S82) = base_0;
    *((&kernelContext_17)->work_0+(_S82 + 1U)) = kept_0;
    *((&kernelContext_17)->loads_0+k_0) = packed_float4(float4(length(f_total_0), 0.0f, 0.0f, 0.0f)) ;
    return;
}

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], atomic<uint> device* atom_2 [[buffer(5)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_18;
    (&kernelContext_18)->params_0 = params_2;
    (&kernelContext_18)->work_0 = work_2;
    (&kernelContext_18)->frames_0 = frames_2;
    (&kernelContext_18)->bodies_0 = bodies_2;
    (&kernelContext_18)->chunks_0 = chunks_2;
    (&kernelContext_18)->sort_0 = sort_2;
    (&kernelContext_18)->atom_0 = atom_2;
    (&kernelContext_18)->manifolds_0 = manifolds_2;
    (&kernelContext_18)->loads_0 = loads_2;
    (&kernelContext_18)->bonds_0 = bonds_2;
    (&kernelContext_18)->args_0 = args_2;
    (&kernelContext_18)->info_0 = info_2;
    (&kernelContext_18)->scan_0 = scan_2;
    (&kernelContext_18)->parts_0 = parts_2;
    (&kernelContext_18)->colliders_0 = colliders_2;
    (&kernelContext_18)->collider_info_0 = collider_info_2;
    (&kernelContext_18)->view_chunks_0 = view_chunks_2;
    (&kernelContext_18)->view_0 = view_2;
    (&kernelContext_18)->objects_0 = objects_2;
    (&kernelContext_18)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_18)->lines_0 = lines_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_18)->g_scan_0 = &g_scan_2;
    uint i_0 = id_1.x;
    bool _S83;
    if(i_0 >= (params_2->bond_count_0))
    {
        _S83 = true;
    }
    else
    {
        uint _S84 = w_alive_0(&kernelContext_18);
        _S83 = (*((&kernelContext_18)->work_0+(_S84 + i_0))) == 0U;
    }
    if(_S83)
    {
        return;
    }
    uint4 _S85 = uint4(*((&kernelContext_18)->bonds_0+i_0)) ;
    uint _S86 = _S85.x;
    if((max((float4(*((&kernelContext_18)->loads_0+_S86)) ).x, (float4(*((&kernelContext_18)->loads_0+_S85.y)) ).x)) > (as_type<float>((_S85.z))))
    {
        uint _S87 = w_alive_0(&kernelContext_18);
        *((&kernelContext_18)->work_0+(_S87 + i_0)) = 0U;
        uint _S88 = a_dirty_0(&kernelContext_18);
        uint _S89 = body_of_0(_S86, &kernelContext_18);
        uint _S90 = atomic_fetch_or_explicit((&kernelContext_18)->atom_0+(_S88 + _S89), 1U, memory_order_relaxed);
        uint _S91 = a_counters_0(&kernelContext_18);
        uint _S92 = atomic_fetch_add_explicit((&kernelContext_18)->atom_0+_S91, 1U, memory_order_relaxed);
        uint _S93 = atomic_fetch_add_explicit((&kernelContext_18)->atom_0+(_S91 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint groups_of_0(uint n_2, uint size_0)
{
    uint _S94 = (n_2 + size_0 - 1U) / size_0;
    return _S94;
}

void set_args_0(uint at_0, uint x_1, KernelContext_0 thread* kernelContext_19)
{
    uint _S95 = 3U * at_0;
    *(kernelContext_19->args_0+_S95) = x_1;
    *(kernelContext_19->args_0+(_S95 + 1U)) = 1U;
    *(kernelContext_19->args_0+(_S95 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_20;
    (&kernelContext_20)->params_0 = params_3;
    (&kernelContext_20)->work_0 = work_3;
    (&kernelContext_20)->frames_0 = frames_3;
    (&kernelContext_20)->bodies_0 = bodies_3;
    (&kernelContext_20)->chunks_0 = chunks_3;
    (&kernelContext_20)->sort_0 = sort_3;
    (&kernelContext_20)->atom_0 = atom_3;
    (&kernelContext_20)->manifolds_0 = manifolds_3;
    (&kernelContext_20)->loads_0 = loads_3;
    (&kernelContext_20)->bonds_0 = bonds_3;
    (&kernelContext_20)->args_0 = args_3;
    (&kernelContext_20)->info_0 = info_3;
    (&kernelContext_20)->scan_0 = scan_3;
    (&kernelContext_20)->parts_0 = parts_3;
    (&kernelContext_20)->colliders_0 = colliders_3;
    (&kernelContext_20)->collider_info_0 = collider_info_3;
    (&kernelContext_20)->view_chunks_0 = view_chunks_3;
    (&kernelContext_20)->view_0 = view_3;
    (&kernelContext_20)->objects_0 = objects_3;
    (&kernelContext_20)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_20)->lines_0 = lines_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_20)->g_scan_0 = &g_scan_3;
    uint _S96 = a_counters_0(&kernelContext_20);
    uint _S97 = atomic_load_explicit((&kernelContext_20)->atom_0+_S96, memory_order_relaxed);
    bool run_0 = _S97 != 0U;
    atomic_store_explicit((&kernelContext_20)->atom_0+_S96, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_20)->atom_0+(_S96 + 5U + (&kernelContext_20)->params_0->parity_0), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S98 = atomic_fetch_add_explicit((&kernelContext_20)->atom_0+(_S96 + 4U), 1U, memory_order_relaxed);
    }
    uint _S99;
    if(run_0)
    {
        uint _S100 = groups_of_0((&kernelContext_20)->params_0->chunk_count_0, 64U);
        _S99 = _S100;
    }
    else
    {
        _S99 = 0U;
    }
    set_args_0(0U, _S99, &kernelContext_20);
    if(run_0)
    {
        uint _S101 = groups_of_0((&kernelContext_20)->params_0->bond_count_0, 64U);
        _S99 = _S101;
    }
    else
    {
        _S99 = 0U;
    }
    set_args_0(1U, _S99, &kernelContext_20);
    if(run_0)
    {
        uint _S102 = groups_of_0((&kernelContext_20)->params_0->body_count_0, 64U);
        _S99 = _S102;
    }
    else
    {
        _S99 = 0U;
    }
    set_args_0(2U, _S99, &kernelContext_20);
    if(run_0)
    {
        uint _S103 = groups_of_0((&kernelContext_20)->params_0->chunk_count_0, 1024U);
        _S99 = _S103;
    }
    else
    {
        _S99 = 0U;
    }
    set_args_0(3U, _S99, &kernelContext_20);
    if(run_0)
    {
        uint _S104 = groups_of_0((&kernelContext_20)->params_0->body_count_0, 1024U);
        _S99 = _S104;
    }
    else
    {
        _S99 = 0U;
    }
    set_args_0(4U, _S99, &kernelContext_20);
    int _S105;
    if(run_0)
    {
        _S105 = int(1);
    }
    else
    {
        _S105 = int(0);
    }
    set_args_0(5U, uint(_S105), &kernelContext_20);
    return;
}

bool dirty_0(uint body_0, KernelContext_0 thread* kernelContext_21)
{
    uint _S106 = a_dirty_0(kernelContext_21);
    uint _S107 = atomic_load_explicit(kernelContext_21->atom_0+(_S106 + body_0), memory_order_relaxed);
    return _S107 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_22)
{
    return kernelContext_22->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_23)
{
    return 2U * kernelContext_23->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_24;
    (&kernelContext_24)->params_0 = params_4;
    (&kernelContext_24)->work_0 = work_4;
    (&kernelContext_24)->frames_0 = frames_4;
    (&kernelContext_24)->bodies_0 = bodies_4;
    (&kernelContext_24)->chunks_0 = chunks_4;
    (&kernelContext_24)->sort_0 = sort_4;
    (&kernelContext_24)->atom_0 = atom_4;
    (&kernelContext_24)->manifolds_0 = manifolds_4;
    (&kernelContext_24)->loads_0 = loads_4;
    (&kernelContext_24)->bonds_0 = bonds_4;
    (&kernelContext_24)->args_0 = args_4;
    (&kernelContext_24)->info_0 = info_4;
    (&kernelContext_24)->scan_0 = scan_4;
    (&kernelContext_24)->parts_0 = parts_4;
    (&kernelContext_24)->colliders_0 = colliders_4;
    (&kernelContext_24)->collider_info_0 = collider_info_4;
    (&kernelContext_24)->view_chunks_0 = view_chunks_4;
    (&kernelContext_24)->view_0 = view_4;
    (&kernelContext_24)->objects_0 = objects_4;
    (&kernelContext_24)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_24)->lines_0 = lines_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_24)->g_scan_0 = &g_scan_4;
    uint c_2 = id_3.x;
    bool _S108;
    if(c_2 >= (params_4->chunk_count_0))
    {
        _S108 = true;
    }
    else
    {
        uint _S109 = body_of_0(c_2, &kernelContext_24);
        bool _S110 = dirty_0(_S109, &kernelContext_24);
        _S108 = !_S110;
    }
    if(_S108)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_24)->atom_0+(a_parent_0() + c_2), c_2, memory_order_relaxed);
    uint _S111 = a_count_0(&kernelContext_24);
    atomic_store_explicit((&kernelContext_24)->atom_0+(_S111 + c_2), 0U, memory_order_relaxed);
    uint _S112 = a_anchor_0(&kernelContext_24);
    atomic_store_explicit((&kernelContext_24)->atom_0+(_S112 + c_2), 0U, memory_order_relaxed);
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
        uint _S113 = a_sums_0(&kernelContext_24);
        atomic_store_explicit((&kernelContext_24)->atom_0+(_S113 + 20U * c_2 + w_2), 0U, memory_order_relaxed);
        w_2 = w_2 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_25;
    (&kernelContext_25)->params_0 = params_5;
    (&kernelContext_25)->work_0 = work_5;
    (&kernelContext_25)->frames_0 = frames_5;
    (&kernelContext_25)->bodies_0 = bodies_5;
    (&kernelContext_25)->chunks_0 = chunks_5;
    (&kernelContext_25)->sort_0 = sort_5;
    (&kernelContext_25)->atom_0 = atom_5;
    (&kernelContext_25)->manifolds_0 = manifolds_5;
    (&kernelContext_25)->loads_0 = loads_5;
    (&kernelContext_25)->bonds_0 = bonds_5;
    (&kernelContext_25)->args_0 = args_5;
    (&kernelContext_25)->info_0 = info_5;
    (&kernelContext_25)->scan_0 = scan_5;
    (&kernelContext_25)->parts_0 = parts_5;
    (&kernelContext_25)->colliders_0 = colliders_5;
    (&kernelContext_25)->collider_info_0 = collider_info_5;
    (&kernelContext_25)->view_chunks_0 = view_chunks_5;
    (&kernelContext_25)->view_0 = view_5;
    (&kernelContext_25)->objects_0 = objects_5;
    (&kernelContext_25)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_25)->lines_0 = lines_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_25)->g_scan_0 = &g_scan_5;
    uint s_0 = id_4.x;
    bool _S114;
    if(s_0 >= (params_5->body_count_0))
    {
        _S114 = true;
    }
    else
    {
        bool _S115 = dirty_0(s_0, &kernelContext_25);
        _S114 = !_S115;
    }
    if(_S114)
    {
        return;
    }
    uint _S116 = a_best_count_0(&kernelContext_25);
    atomic_store_explicit((&kernelContext_25)->atom_0+(_S116 + s_0), 0U, memory_order_relaxed);
    uint _S117 = a_best_root_0(&kernelContext_25);
    atomic_store_explicit((&kernelContext_25)->atom_0+(_S117 + s_0), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_2, KernelContext_0 thread* kernelContext_26)
{
    uint _S118 = a_parent_0();
    uint _S119 = atomic_load_explicit(kernelContext_26->atom_0+(_S118 + x_2), memory_order_relaxed);
    uint p_1 = _S119;
    uint _S120 = x_2;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S121;
        if(guard_0 < 1048576U)
        {
            _S121 = p_1 != _S120;
        }
        else
        {
            _S121 = false;
        }
        if(_S121)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_26->atom_0+(_S118 + p_1), memory_order_relaxed);
        if(gp_0 != p_1)
        {
            uint _S122;
            {
            uint _metal_cas_comparand = p_1;
            atomic_compare_exchange_weak_explicit(kernelContext_26->atom_0+(_S118 + _S120), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S122 = _metal_cas_comparand;
            }
        }
        uint _S123 = guard_0 + 1U;
        _S120 = p_1;
        p_1 = gp_0;
        guard_0 = _S123;
    }
    return _S120;
}

[[kernel]] void fr_uf_union(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_27;
    (&kernelContext_27)->params_0 = params_6;
    (&kernelContext_27)->work_0 = work_6;
    (&kernelContext_27)->frames_0 = frames_6;
    (&kernelContext_27)->bodies_0 = bodies_6;
    (&kernelContext_27)->chunks_0 = chunks_6;
    (&kernelContext_27)->sort_0 = sort_6;
    (&kernelContext_27)->atom_0 = atom_6;
    (&kernelContext_27)->manifolds_0 = manifolds_6;
    (&kernelContext_27)->loads_0 = loads_6;
    (&kernelContext_27)->bonds_0 = bonds_6;
    (&kernelContext_27)->args_0 = args_6;
    (&kernelContext_27)->info_0 = info_6;
    (&kernelContext_27)->scan_0 = scan_6;
    (&kernelContext_27)->parts_0 = parts_6;
    (&kernelContext_27)->colliders_0 = colliders_6;
    (&kernelContext_27)->collider_info_0 = collider_info_6;
    (&kernelContext_27)->view_chunks_0 = view_chunks_6;
    (&kernelContext_27)->view_0 = view_6;
    (&kernelContext_27)->objects_0 = objects_6;
    (&kernelContext_27)->shot_bodies_0 = shot_bodies_6;
    (&kernelContext_27)->lines_0 = lines_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_27)->g_scan_0 = &g_scan_6;
    uint i_1 = id_5.x;
    bool _S124;
    if(i_1 >= (params_6->bond_count_0))
    {
        _S124 = true;
    }
    else
    {
        uint _S125 = w_alive_0(&kernelContext_27);
        _S124 = (*((&kernelContext_27)->work_0+(_S125 + i_1))) == 0U;
    }
    if(_S124)
    {
        return;
    }
    uint4 _S126 = uint4(*((&kernelContext_27)->bonds_0+i_1)) ;
    uint _S127 = _S126.x;
    uint _S128 = body_of_0(_S127, &kernelContext_27);
    bool _S129 = dirty_0(_S128, &kernelContext_27);
    if(!_S129)
    {
        return;
    }
    uint _S130 = _S126.y;
    uint a_2 = _S127;
    uint b_2 = _S130;
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
        uint _S131 = find_0(a_2, &kernelContext_27);
        uint _S132 = find_0(b_2, &kernelContext_27);
        if(_S131 == _S132)
        {
            return;
        }
        uint _S133 = max(_S131, _S132);
        uint _S134 = min(_S131, _S132);
        uint was_0;
        {
        uint _metal_cas_comparand = _S133;
        atomic_compare_exchange_weak_explicit((&kernelContext_27)->atom_0+(a_parent_0() + _S133), &_metal_cas_comparand, _S134, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S133)
        {
            return;
        }
        uint _S135 = guard_1 + 1U;
        a_2 = _S133;
        b_2 = _S134;
        guard_1 = _S135;
    }
    uint _S136 = a_counters_0(&kernelContext_27);
    uint _S137 = atomic_fetch_or_explicit((&kernelContext_27)->atom_0+(_S136 + 3U), 4U, memory_order_relaxed);
    return;
}

void add64_0(uint at_1, float value_0, float scale_0, KernelContext_0 thread* kernelContext_28)
{
    float x_3 = round(value_0 * scale_0);
    float x_4;
    if(!((abs(x_3)) < 1.073741824e+09f))
    {
        uint _S138 = a_counters_0(kernelContext_28);
        uint _S139 = atomic_fetch_or_explicit(kernelContext_28->atom_0+(_S138 + 3U), 2U, memory_order_relaxed);
        x_4 = 0.0f;
    }
    else
    {
        x_4 = x_3;
    }
    int v_3 = int(x_4);
    uint lo_0 = (as_type<uint>((v_3)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_28->atom_0+at_1, lo_0, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_0) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S140;
    if(v_3 < int(0))
    {
        _S140 = 4294967295U;
    }
    else
    {
        _S140 = 0U;
    }
    uint hi_0 = _S140 + carry_0;
    if(hi_0 != 0U)
    {
        uint _S141 = atomic_fetch_add_explicit(kernelContext_28->atom_0+(at_1 + 1U), hi_0, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_29;
    (&kernelContext_29)->params_0 = params_7;
    (&kernelContext_29)->work_0 = work_7;
    (&kernelContext_29)->frames_0 = frames_7;
    (&kernelContext_29)->bodies_0 = bodies_7;
    (&kernelContext_29)->chunks_0 = chunks_7;
    (&kernelContext_29)->sort_0 = sort_7;
    (&kernelContext_29)->atom_0 = atom_7;
    (&kernelContext_29)->manifolds_0 = manifolds_7;
    (&kernelContext_29)->loads_0 = loads_7;
    (&kernelContext_29)->bonds_0 = bonds_7;
    (&kernelContext_29)->args_0 = args_7;
    (&kernelContext_29)->info_0 = info_7;
    (&kernelContext_29)->scan_0 = scan_7;
    (&kernelContext_29)->parts_0 = parts_7;
    (&kernelContext_29)->colliders_0 = colliders_7;
    (&kernelContext_29)->collider_info_0 = collider_info_7;
    (&kernelContext_29)->view_chunks_0 = view_chunks_7;
    (&kernelContext_29)->view_0 = view_7;
    (&kernelContext_29)->objects_0 = objects_7;
    (&kernelContext_29)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_29)->lines_0 = lines_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_29)->g_scan_0 = &g_scan_7;
    uint c_3 = id_6.x;
    bool _S142;
    if(c_3 >= (params_7->chunk_count_0))
    {
        _S142 = true;
    }
    else
    {
        uint _S143 = body_of_0(c_3, &kernelContext_29);
        bool _S144 = dirty_0(_S143, &kernelContext_29);
        _S142 = !_S144;
    }
    if(_S142)
    {
        return;
    }
    uint _S145 = find_0(c_3, &kernelContext_29);
    uint _S146 = w_label_0(&kernelContext_29);
    *((&kernelContext_29)->work_0+(_S146 + c_3)) = _S145;
    uint _S147 = a_count_0(&kernelContext_29);
    uint _S148 = atomic_fetch_add_explicit((&kernelContext_29)->atom_0+(_S147 + _S145), 1U, memory_order_relaxed);
    if(((&kernelContext_29)->info_0[c_3] & 1U) != 0U)
    {
        uint _S149 = a_anchor_0(&kernelContext_29);
        uint _S150 = atomic_fetch_or_explicit((&kernelContext_29)->atom_0+(_S149 + _S145), 1U, memory_order_relaxed);
    }
    uint _S151 = 4U * c_3;
    float4 _S152 = float4(*((&kernelContext_29)->chunks_0+_S151)) ;
    float4 _S153 = float4(*((&kernelContext_29)->chunks_0+(_S151 + 1U))) ;
    float4 _S154 = float4(*((&kernelContext_29)->chunks_0+(_S151 + 2U))) ;
    float m_2 = _S152.w;
    float3 d_0 = _S152.xyz - (float4(*((&kernelContext_29)->chunks_0+4U * _S145)) ).xyz;
    float dd_0 = dot(d_0, d_0);
    uint _S155 = a_sums_0(&kernelContext_29);
    uint at_2 = _S155 + 20U * _S145;
    add64_0(at_2, m_2, (&kernelContext_29)->params_0->scale_m_0, &kernelContext_29);
    float _S156 = d_0.x;
    float _S157 = m_2 * _S156;
    add64_0(at_2 + 2U, _S157, (&kernelContext_29)->params_0->scale_s_0, &kernelContext_29);
    float _S158 = d_0.y;
    float _S159 = m_2 * _S158;
    add64_0(at_2 + 4U, _S159, (&kernelContext_29)->params_0->scale_s_0, &kernelContext_29);
    float _S160 = d_0.z;
    add64_0(at_2 + 6U, m_2 * _S160, (&kernelContext_29)->params_0->scale_s_0, &kernelContext_29);
    add64_0(at_2 + 8U, _S153.x + m_2 * (dd_0 - _S156 * _S156), (&kernelContext_29)->params_0->scale_j_0, &kernelContext_29);
    add64_0(at_2 + 10U, _S153.y + m_2 * (dd_0 - _S158 * _S158), (&kernelContext_29)->params_0->scale_j_0, &kernelContext_29);
    add64_0(at_2 + 12U, _S153.z + m_2 * (dd_0 - _S160 * _S160), (&kernelContext_29)->params_0->scale_j_0, &kernelContext_29);
    add64_0(at_2 + 14U, _S153.w - _S157 * _S158, (&kernelContext_29)->params_0->scale_j_0, &kernelContext_29);
    add64_0(at_2 + 16U, _S154.x - _S157 * _S160, (&kernelContext_29)->params_0->scale_j_0, &kernelContext_29);
    add64_0(at_2 + 18U, _S154.y - _S159 * _S160, (&kernelContext_29)->params_0->scale_j_0, &kernelContext_29);
    return;
}

bool is_root_0(uint c_4, KernelContext_0 thread* kernelContext_30)
{
    bool _S161;
    if(c_4 < (kernelContext_30->params_0->chunk_count_0))
    {
        uint _S162 = body_of_0(c_4, kernelContext_30);
        bool _S163 = dirty_0(_S162, kernelContext_30);
        _S161 = _S163;
    }
    else
    {
        _S161 = false;
    }
    if(_S161)
    {
        uint _S164 = w_label_0(kernelContext_30);
        _S161 = (*(kernelContext_30->work_0+(_S164 + c_4))) == c_4;
    }
    else
    {
        _S161 = false;
    }
    return _S161;
}

[[kernel]] void fr_best_count(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_31;
    (&kernelContext_31)->params_0 = params_8;
    (&kernelContext_31)->work_0 = work_8;
    (&kernelContext_31)->frames_0 = frames_8;
    (&kernelContext_31)->bodies_0 = bodies_8;
    (&kernelContext_31)->chunks_0 = chunks_8;
    (&kernelContext_31)->sort_0 = sort_8;
    (&kernelContext_31)->atom_0 = atom_8;
    (&kernelContext_31)->manifolds_0 = manifolds_8;
    (&kernelContext_31)->loads_0 = loads_8;
    (&kernelContext_31)->bonds_0 = bonds_8;
    (&kernelContext_31)->args_0 = args_8;
    (&kernelContext_31)->info_0 = info_8;
    (&kernelContext_31)->scan_0 = scan_8;
    (&kernelContext_31)->parts_0 = parts_8;
    (&kernelContext_31)->colliders_0 = colliders_8;
    (&kernelContext_31)->collider_info_0 = collider_info_8;
    (&kernelContext_31)->view_chunks_0 = view_chunks_8;
    (&kernelContext_31)->view_0 = view_8;
    (&kernelContext_31)->objects_0 = objects_8;
    (&kernelContext_31)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_31)->lines_0 = lines_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_31)->g_scan_0 = &g_scan_8;
    uint c_5 = id_7.x;
    bool _S165 = is_root_0(c_5, &kernelContext_31);
    if(!_S165)
    {
        return;
    }
    uint _S166 = a_best_count_0(&kernelContext_31);
    uint _S167 = body_of_0(c_5, &kernelContext_31);
    atomic<uint> device* _S168 = (&kernelContext_31)->atom_0+(_S166 + _S167);
    uint _S169 = a_count_0(&kernelContext_31);
    uint _S170 = atomic_load_explicit((&kernelContext_31)->atom_0+(_S169 + c_5), memory_order_relaxed);
    uint _S171 = atomic_fetch_max_explicit(_S168, _S170, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_32;
    (&kernelContext_32)->params_0 = params_9;
    (&kernelContext_32)->work_0 = work_9;
    (&kernelContext_32)->frames_0 = frames_9;
    (&kernelContext_32)->bodies_0 = bodies_9;
    (&kernelContext_32)->chunks_0 = chunks_9;
    (&kernelContext_32)->sort_0 = sort_9;
    (&kernelContext_32)->atom_0 = atom_9;
    (&kernelContext_32)->manifolds_0 = manifolds_9;
    (&kernelContext_32)->loads_0 = loads_9;
    (&kernelContext_32)->bonds_0 = bonds_9;
    (&kernelContext_32)->args_0 = args_9;
    (&kernelContext_32)->info_0 = info_9;
    (&kernelContext_32)->scan_0 = scan_9;
    (&kernelContext_32)->parts_0 = parts_9;
    (&kernelContext_32)->colliders_0 = colliders_9;
    (&kernelContext_32)->collider_info_0 = collider_info_9;
    (&kernelContext_32)->view_chunks_0 = view_chunks_9;
    (&kernelContext_32)->view_0 = view_9;
    (&kernelContext_32)->objects_0 = objects_9;
    (&kernelContext_32)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_32)->lines_0 = lines_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_32)->g_scan_0 = &g_scan_9;
    uint c_6 = id_8.x;
    bool _S172 = is_root_0(c_6, &kernelContext_32);
    if(!_S172)
    {
        return;
    }
    uint _S173 = body_of_0(c_6, &kernelContext_32);
    uint _S174 = a_count_0(&kernelContext_32);
    uint _S175 = atomic_load_explicit((&kernelContext_32)->atom_0+(_S174 + c_6), memory_order_relaxed);
    uint _S176 = a_best_count_0(&kernelContext_32);
    uint _S177 = atomic_load_explicit((&kernelContext_32)->atom_0+(_S176 + _S173), memory_order_relaxed);
    if(_S175 == _S177)
    {
        uint _S178 = a_best_root_0(&kernelContext_32);
        uint _S179 = atomic_fetch_min_explicit((&kernelContext_32)->atom_0+(_S178 + _S173), c_6, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_33;
    (&kernelContext_33)->params_0 = params_10;
    (&kernelContext_33)->work_0 = work_10;
    (&kernelContext_33)->frames_0 = frames_10;
    (&kernelContext_33)->bodies_0 = bodies_10;
    (&kernelContext_33)->chunks_0 = chunks_10;
    (&kernelContext_33)->sort_0 = sort_10;
    (&kernelContext_33)->atom_0 = atom_10;
    (&kernelContext_33)->manifolds_0 = manifolds_10;
    (&kernelContext_33)->loads_0 = loads_10;
    (&kernelContext_33)->bonds_0 = bonds_10;
    (&kernelContext_33)->args_0 = args_10;
    (&kernelContext_33)->info_0 = info_10;
    (&kernelContext_33)->scan_0 = scan_10;
    (&kernelContext_33)->parts_0 = parts_10;
    (&kernelContext_33)->colliders_0 = colliders_10;
    (&kernelContext_33)->collider_info_0 = collider_info_10;
    (&kernelContext_33)->view_chunks_0 = view_chunks_10;
    (&kernelContext_33)->view_0 = view_10;
    (&kernelContext_33)->objects_0 = objects_10;
    (&kernelContext_33)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_33)->lines_0 = lines_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_33)->g_scan_0 = &g_scan_10;
    uint c_7 = id_9.x;
    if(c_7 >= (params_10->chunk_count_0))
    {
        return;
    }
    uint _S180 = w_new_0(&kernelContext_33);
    uint device* _S181 = (&kernelContext_33)->work_0+(_S180 + c_7);
    bool _S182 = is_root_0(c_7, &kernelContext_33);
    bool _S183;
    if(_S182)
    {
        uint _S184 = a_best_root_0(&kernelContext_33);
        uint _S185 = body_of_0(c_7, &kernelContext_33);
        uint _S186 = atomic_load_explicit((&kernelContext_33)->atom_0+(_S184 + _S185), memory_order_relaxed);
        _S183 = _S186 != c_7;
    }
    else
    {
        _S183 = false;
    }
    uint _S187;
    if(_S183)
    {
        _S187 = 1U;
    }
    else
    {
        _S187 = 0U;
    }
    *_S181 = _S187;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]])
{
    uint step_2;
    uint add_0;
    thread KernelContext_0 kernelContext_34;
    (&kernelContext_34)->params_0 = params_11;
    (&kernelContext_34)->work_0 = work_11;
    (&kernelContext_34)->frames_0 = frames_11;
    (&kernelContext_34)->bodies_0 = bodies_11;
    (&kernelContext_34)->chunks_0 = chunks_11;
    (&kernelContext_34)->sort_0 = sort_11;
    (&kernelContext_34)->atom_0 = atom_11;
    (&kernelContext_34)->manifolds_0 = manifolds_11;
    (&kernelContext_34)->loads_0 = loads_11;
    (&kernelContext_34)->bonds_0 = bonds_11;
    (&kernelContext_34)->args_0 = args_11;
    (&kernelContext_34)->info_0 = info_11;
    (&kernelContext_34)->scan_0 = scan_11;
    (&kernelContext_34)->parts_0 = parts_11;
    (&kernelContext_34)->colliders_0 = colliders_11;
    (&kernelContext_34)->collider_info_0 = collider_info_11;
    (&kernelContext_34)->view_chunks_0 = view_chunks_11;
    (&kernelContext_34)->view_0 = view_11;
    (&kernelContext_34)->objects_0 = objects_11;
    (&kernelContext_34)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_34)->lines_0 = lines_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_34)->g_scan_0 = &g_scan_11;
    uint t_1 = thread_0.x;
    uint _S188 = group_0.x;
    uint _S189 = 1024U * _S188 + 4U * t_1;
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
        if((_S189 + j_2) < ((&kernelContext_34)->scan_0->n_0))
        {
            step_2 = *((&kernelContext_34)->work_0+((&kernelContext_34)->scan_0->in_base_0 + _S189 + j_2));
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
    (*(&kernelContext_34)->g_scan_0)[t_1] = sum_2;
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
            add_0 = (*(&kernelContext_34)->g_scan_0)[t_1 - step_2];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_34)->g_scan_0)[t_1] = (*(&kernelContext_34)->g_scan_0)[t_1] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_2 = step_2 << 1U;
    }
    uint _S190 = (*(&kernelContext_34)->g_scan_0)[t_1] - sum_2;
    j_2 = 0U;
    add_0 = _S190;
    for(;;)
    {
        if(j_2 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S189 + j_2) < ((&kernelContext_34)->scan_0->n_0))
        {
            *((&kernelContext_34)->work_0+((&kernelContext_34)->scan_0->out_base_0 + _S189 + j_2)) = add_0;
        }
        uint run_1 = add_0 + v_4[j_2];
        j_2 = j_2 + 1U;
        add_0 = run_1;
    }
    if(t_1 == 255U)
    {
        *((&kernelContext_34)->work_0+((&kernelContext_34)->scan_0->block_base_0 + _S188)) = (*(&kernelContext_34)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_35;
    (&kernelContext_35)->params_0 = params_12;
    (&kernelContext_35)->work_0 = work_12;
    (&kernelContext_35)->frames_0 = frames_12;
    (&kernelContext_35)->bodies_0 = bodies_12;
    (&kernelContext_35)->chunks_0 = chunks_12;
    (&kernelContext_35)->sort_0 = sort_12;
    (&kernelContext_35)->atom_0 = atom_12;
    (&kernelContext_35)->manifolds_0 = manifolds_12;
    (&kernelContext_35)->loads_0 = loads_12;
    (&kernelContext_35)->bonds_0 = bonds_12;
    (&kernelContext_35)->args_0 = args_12;
    (&kernelContext_35)->info_0 = info_12;
    (&kernelContext_35)->scan_0 = scan_12;
    (&kernelContext_35)->parts_0 = parts_12;
    (&kernelContext_35)->colliders_0 = colliders_12;
    (&kernelContext_35)->collider_info_0 = collider_info_12;
    (&kernelContext_35)->view_chunks_0 = view_chunks_12;
    (&kernelContext_35)->view_0 = view_12;
    (&kernelContext_35)->objects_0 = objects_12;
    (&kernelContext_35)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_35)->lines_0 = lines_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_35)->g_scan_0 = &g_scan_12;
    uint t_2 = thread_1.x;
    uint blocks_0 = (scan_12->n_0 + 1023U) / 1024U;
    bool _S191 = t_2 < blocks_0;
    uint v_5;
    if(_S191)
    {
        v_5 = *((&kernelContext_35)->work_0+((&kernelContext_35)->scan_0->block_base_0 + t_2));
    }
    else
    {
        v_5 = 0U;
    }
    (*(&kernelContext_35)->g_scan_0)[t_2] = v_5;
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
            add_1 = (*(&kernelContext_35)->g_scan_0)[t_2 - step_3];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_35)->g_scan_0)[t_2] = (*(&kernelContext_35)->g_scan_0)[t_2] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_3 = step_3 << 1U;
    }
    if(_S191)
    {
        *((&kernelContext_35)->work_0+((&kernelContext_35)->scan_0->block_base_0 + t_2)) = (*(&kernelContext_35)->g_scan_0)[t_2] - v_5;
    }
    if(t_2 == 1023U)
    {
        *((&kernelContext_35)->work_0+((&kernelContext_35)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_35)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_36;
    (&kernelContext_36)->params_0 = params_13;
    (&kernelContext_36)->work_0 = work_13;
    (&kernelContext_36)->frames_0 = frames_13;
    (&kernelContext_36)->bodies_0 = bodies_13;
    (&kernelContext_36)->chunks_0 = chunks_13;
    (&kernelContext_36)->sort_0 = sort_13;
    (&kernelContext_36)->atom_0 = atom_13;
    (&kernelContext_36)->manifolds_0 = manifolds_13;
    (&kernelContext_36)->loads_0 = loads_13;
    (&kernelContext_36)->bonds_0 = bonds_13;
    (&kernelContext_36)->args_0 = args_13;
    (&kernelContext_36)->info_0 = info_13;
    (&kernelContext_36)->scan_0 = scan_13;
    (&kernelContext_36)->parts_0 = parts_13;
    (&kernelContext_36)->colliders_0 = colliders_13;
    (&kernelContext_36)->collider_info_0 = collider_info_13;
    (&kernelContext_36)->view_chunks_0 = view_chunks_13;
    (&kernelContext_36)->view_0 = view_13;
    (&kernelContext_36)->objects_0 = objects_13;
    (&kernelContext_36)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_36)->lines_0 = lines_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_36)->g_scan_0 = &g_scan_13;
    uint i_2 = id_10.x;
    if(i_2 >= (scan_13->n_0))
    {
        return;
    }
    uint device* _S192 = (&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->out_base_0 + i_2);
    *_S192 = *_S192 + *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->block_base_0 + i_2 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_37;
    (&kernelContext_37)->params_0 = params_14;
    (&kernelContext_37)->work_0 = work_14;
    (&kernelContext_37)->frames_0 = frames_14;
    (&kernelContext_37)->bodies_0 = bodies_14;
    (&kernelContext_37)->chunks_0 = chunks_14;
    (&kernelContext_37)->sort_0 = sort_14;
    (&kernelContext_37)->atom_0 = atom_14;
    (&kernelContext_37)->manifolds_0 = manifolds_14;
    (&kernelContext_37)->loads_0 = loads_14;
    (&kernelContext_37)->bonds_0 = bonds_14;
    (&kernelContext_37)->args_0 = args_14;
    (&kernelContext_37)->info_0 = info_14;
    (&kernelContext_37)->scan_0 = scan_14;
    (&kernelContext_37)->parts_0 = parts_14;
    (&kernelContext_37)->colliders_0 = colliders_14;
    (&kernelContext_37)->collider_info_0 = collider_info_14;
    (&kernelContext_37)->view_chunks_0 = view_chunks_14;
    (&kernelContext_37)->view_0 = view_14;
    (&kernelContext_37)->objects_0 = objects_14;
    (&kernelContext_37)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_37)->lines_0 = lines_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_37)->g_scan_0 = &g_scan_14;
    uint s_1 = id_11.x;
    bool _S193;
    if(s_1 >= (params_14->body_count_0))
    {
        _S193 = true;
    }
    else
    {
        uint _S194 = w_free_0(&kernelContext_37);
        _S193 = (*((&kernelContext_37)->work_0+(_S194 + s_1))) == 0U;
    }
    if(_S193)
    {
        return;
    }
    uint _S195 = w_free_list_0(&kernelContext_37);
    uint _S196 = w_free_rank_0(&kernelContext_37);
    *((&kernelContext_37)->work_0+(_S195 + *((&kernelContext_37)->work_0+(_S196 + s_1)))) = s_1;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_38)
{
    uint _S197 = w_free_list_0(kernelContext_38);
    return _S197 + kernelContext_38->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_39)
{
    uint _S198 = scan_new_blocks_0(kernelContext_39);
    return _S198 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_40)
{
    uint blocks_1 = (kernelContext_40->params_0->body_count_0 + 1023U) / 1024U;
    uint _S199 = scan_free_blocks_0(kernelContext_40);
    return *(kernelContext_40->work_0+(_S199 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_41;
    (&kernelContext_41)->params_0 = params_15;
    (&kernelContext_41)->work_0 = work_15;
    (&kernelContext_41)->frames_0 = frames_15;
    (&kernelContext_41)->bodies_0 = bodies_15;
    (&kernelContext_41)->chunks_0 = chunks_15;
    (&kernelContext_41)->sort_0 = sort_15;
    (&kernelContext_41)->atom_0 = atom_15;
    (&kernelContext_41)->manifolds_0 = manifolds_15;
    (&kernelContext_41)->loads_0 = loads_15;
    (&kernelContext_41)->bonds_0 = bonds_15;
    (&kernelContext_41)->args_0 = args_15;
    (&kernelContext_41)->info_0 = info_15;
    (&kernelContext_41)->scan_0 = scan_15;
    (&kernelContext_41)->parts_0 = parts_15;
    (&kernelContext_41)->colliders_0 = colliders_15;
    (&kernelContext_41)->collider_info_0 = collider_info_15;
    (&kernelContext_41)->view_chunks_0 = view_chunks_15;
    (&kernelContext_41)->view_0 = view_15;
    (&kernelContext_41)->objects_0 = objects_15;
    (&kernelContext_41)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_41)->lines_0 = lines_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_41)->g_scan_0 = &g_scan_15;
    uint c_8 = id_12.x;
    bool _S200 = is_root_0(c_8, &kernelContext_41);
    if(!_S200)
    {
        return;
    }
    uint _S201 = body_of_0(c_8, &kernelContext_41);
    uint _S202 = a_best_root_0(&kernelContext_41);
    uint _S203 = atomic_load_explicit((&kernelContext_41)->atom_0+(_S202 + _S201), memory_order_relaxed);
    uint slot_1;
    if(_S203 != c_8)
    {
        uint _S204 = w_rank_0(&kernelContext_41);
        uint device* _S205 = (&kernelContext_41)->work_0+(_S204 + c_8);
        uint k_1 = *_S205;
        uint _S206 = free_total_0(&kernelContext_41);
        if((*_S205) < _S206)
        {
            uint _S207 = w_free_list_0(&kernelContext_41);
            uint device* _S208 = (&kernelContext_41)->work_0+(_S207 + k_1);
            uint slot_2 = *_S208;
            uint _S209 = w_free_0(&kernelContext_41);
            *((&kernelContext_41)->work_0+(_S209 + *_S208)) = 0U;
            uint _S210 = a_counters_0(&kernelContext_41);
            uint _S211 = atomic_fetch_add_explicit((&kernelContext_41)->atom_0+(_S210 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S212 = a_counters_0(&kernelContext_41);
            uint _S213 = atomic_fetch_or_explicit((&kernelContext_41)->atom_0+(_S212 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S201;
        }
    }
    else
    {
        slot_1 = _S201;
    }
    uint _S214 = w_slot_0(&kernelContext_41);
    *((&kernelContext_41)->work_0+(_S214 + c_8)) = slot_1;
    return;
}

float read64_0(uint at_3, float scale_1, KernelContext_0 thread* kernelContext_42)
{
    uint _S215 = atomic_load_explicit(kernelContext_42->atom_0+(at_3 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S215)));
    uint lo_1 = atomic_load_explicit(kernelContext_42->atom_0+at_3, memory_order_relaxed);
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
    array<float, int(3)> _S216 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S217 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S218 = { { r2_0.x, r2_0.y, r2_0.z } };
    thread array<array<float, int(3)>, int(3)> a_3;
    a_3[int(0)] = _S216;
    a_3[int(1)] = _S217;
    a_3[int(2)] = _S218;
    array<float, int(3)> _S219 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S220 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S221 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_6;
    v_6[int(0)] = _S219;
    v_6[int(1)] = _S220;
    v_6[int(2)] = _S221;
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
        float c_9 = cos(theta_0);
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
            bool _S222;
            if(k_2 != p_2)
            {
                _S222 = k_2 != q_9;
            }
            else
            {
                _S222 = false;
            }
            if(_S222)
            {
                float kp_0 = a_3[k_2][p_2];
                float kq_0 = a_3[k_2][q_9];
                float _S223 = c_9 * a_3[k_2][p_2] - s_2 * a_3[k_2][q_9];
                a_3[k_2][p_2] = _S223;
                a_3[p_2][k_2] = _S223;
                float _S224 = s_2 * kp_0 + c_9 * kq_0;
                a_3[k_2][q_9] = _S224;
                a_3[q_9][k_2] = _S224;
            }
            k_2 = k_2 + 1U;
        }
        float _S225 = c_9 * c_9;
        float _S226 = 2.0f * s_2 * c_9 * apq_0;
        float _S227 = s_2 * s_2;
        a_3[p_2][p_2] = _S225 * app_0 - _S226 + _S227 * aqq_0;
        a_3[q_9][q_9] = _S227 * app_0 + _S226 + _S225 * aqq_0;
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
            v_6[row_0][p_2] = c_9 * v_6[row_0][p_2] - s_2 * v_6[row_0][q_9];
            v_6[row_0][q_9] = s_2 * kp_1 + c_9 * kq_1;
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
    thread Quat_0 _S228 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S229 = normalized_0(&_S228);
    *rot_0 = _S229;
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
    thread Quat_0 _S230 = axes_1;
    Quat_0 _S231 = quat_mul_0(&_S230, q_10);
    thread Quat_0 _S232 = _S231;
    Quat_0 _S233 = normalized_0(&_S232);
    return _S233;
}

uint stage_0(uint c_10, KernelContext_0 thread* kernelContext_43)
{
    return 2U * kernelContext_43->params_0->body_count_0 + 9U * c_10;
}

float4 quat_vec_0(const Quat_0 thread* q_11)
{
    return float4(q_11->x_0, q_11->y_0, q_11->z_0, q_11->w_0);
}

[[kernel]] void fr_piece(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_44;
    (&kernelContext_44)->params_0 = params_16;
    (&kernelContext_44)->work_0 = work_16;
    (&kernelContext_44)->frames_0 = frames_16;
    (&kernelContext_44)->bodies_0 = bodies_16;
    (&kernelContext_44)->chunks_0 = chunks_16;
    (&kernelContext_44)->sort_0 = sort_16;
    (&kernelContext_44)->atom_0 = atom_16;
    (&kernelContext_44)->manifolds_0 = manifolds_16;
    (&kernelContext_44)->loads_0 = loads_16;
    (&kernelContext_44)->bonds_0 = bonds_16;
    (&kernelContext_44)->args_0 = args_16;
    (&kernelContext_44)->info_0 = info_16;
    (&kernelContext_44)->scan_0 = scan_16;
    (&kernelContext_44)->parts_0 = parts_16;
    (&kernelContext_44)->colliders_0 = colliders_16;
    (&kernelContext_44)->collider_info_0 = collider_info_16;
    (&kernelContext_44)->view_chunks_0 = view_chunks_16;
    (&kernelContext_44)->view_0 = view_16;
    (&kernelContext_44)->objects_0 = objects_16;
    (&kernelContext_44)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_44)->lines_0 = lines_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_44)->g_scan_0 = &g_scan_16;
    uint c_11 = id_13.x;
    bool _S234 = is_root_0(c_11, &kernelContext_44);
    if(!_S234)
    {
        return;
    }
    uint _S235 = body_of_0(c_11, &kernelContext_44);
    uint pb_0 = 7U * _S235;
    uint _S236 = a_sums_0(&kernelContext_44);
    uint at_4 = _S236 + 20U * c_11;
    float _S237 = read64_0(at_4, (&kernelContext_44)->params_0->scale_m_0, &kernelContext_44);
    float _S238 = read64_0(at_4 + 2U, (&kernelContext_44)->params_0->scale_s_0, &kernelContext_44);
    float _S239 = read64_0(at_4 + 4U, (&kernelContext_44)->params_0->scale_s_0, &kernelContext_44);
    float _S240 = read64_0(at_4 + 6U, (&kernelContext_44)->params_0->scale_s_0, &kernelContext_44);
    float3 d_1 = float3(_S238, _S239, _S240) / float3(max(_S237, 1.00000000317107685e-30f)) ;
    float _S241 = read64_0(at_4 + 8U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float _S242 = read64_0(at_4 + 10U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float _S243 = read64_0(at_4 + 12U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float _S244 = read64_0(at_4 + 14U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float _S245 = read64_0(at_4 + 16U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float _S246 = read64_0(at_4 + 18U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float dd_1 = dot(d_1, d_1);
    float _S247 = d_1.x;
    float _S248 = _S237 * _S247;
    float _S249 = d_1.y;
    float _S250 = _S244 + _S248 * _S249;
    float _S251 = d_1.z;
    float _S252 = _S245 + _S248 * _S251;
    float _S253 = _S246 + _S237 * _S249 * _S251;
    float3 com_1 = (float4(*((&kernelContext_44)->chunks_0+4U * c_11)) ).xyz + d_1;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S241 - _S237 * (dd_1 - _S247 * _S247), _S250, _S252), float3(_S250, _S242 - _S237 * (dd_1 - _S249 * _S249), _S253), float3(_S252, _S253, _S243 - _S237 * (dd_1 - _S251 * _S251)), &principal_0, &moments_1);
    uint _S254 = a_anchor_0(&kernelContext_44);
    uint _S255 = atomic_load_explicit((&kernelContext_44)->atom_0+(_S254 + c_11), memory_order_relaxed);
    bool anchored_0 = _S255 != 0U;
    thread Quat_0 q_12;
    thread float3 origin_2;
    structure_pose_0(_S235, pb_0, &q_12, &origin_2, &kernelContext_44);
    float3 p_parent_0 = (float4(*((&kernelContext_44)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_44)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_44)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S256 = q_12;
    float3 _S257 = rotate_0(&_S256, com_1);
    float3 pos_0 = to_rubble_0(origin_2 + _S257);
    thread Quat_0 _S258 = q_12;
    thread Quat_0 _S259 = principal_0;
    Quat_0 _S260 = quat_mul_0(&_S258, &_S259);
    thread Quat_0 _S261 = _S260;
    Quat_0 _S262 = normalized_0(&_S261);
    thread Quat_0 _S263 = _S262;
    Quat_0 _S264 = rot_to_rubble_0(&_S263);
    bool _S265;
    if(anchored_0)
    {
        _S265 = true;
    }
    else
    {
        _S265 = !(_S237 > 0.0f);
    }
    float inv_mass_1;
    if(_S265)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / _S237;
    }
    float3 _S266 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S267;
        if((moments_1.x) > 0.0f)
        {
            _S267 = 1.0f / moments_1.x;
        }
        else
        {
            _S267 = 0.0f;
        }
        float _S268;
        if((moments_1.y) > 0.0f)
        {
            _S268 = 1.0f / moments_1.y;
        }
        else
        {
            _S268 = 0.0f;
        }
        float _S269;
        if((moments_1.z) > 0.0f)
        {
            _S269 = 1.0f / moments_1.z;
        }
        else
        {
            _S269 = 0.0f;
        }
        inv_inertia_0 = float3(_S267, _S268, _S269);
    }
    else
    {
        inv_inertia_0 = _S266;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S266;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S266;
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
    uint _S270 = stage_0(c_11, &kernelContext_44);
    *((&kernelContext_44)->frames_0+_S270) = packed_float4(float4(pos_0, inv_mass_1)) ;
    packed_float4 device* _S271 = (&kernelContext_44)->frames_0+(_S270 + 1U);
    thread Quat_0 _S272 = _S264;
    float4 _S273 = quat_vec_0(&_S272);
    *_S271 = packed_float4(_S273) ;
    *((&kernelContext_44)->frames_0+(_S270 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_44)->frames_0+(_S270 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_44)->frames_0+(_S270 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_44)->params_0->step_0))))) ;
    *((&kernelContext_44)->frames_0+(_S270 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_44)->frames_0+(_S270 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_44)->frames_0+(_S270 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_44)->frames_0+2U * _S235)) ).w)) ;
    packed_float4 device* _S274 = (&kernelContext_44)->frames_0+(_S270 + 8U);
    thread Quat_0 _S275 = principal_0;
    float4 _S276 = quat_vec_0(&_S275);
    *_S274 = packed_float4(_S276) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_45;
    (&kernelContext_45)->params_0 = params_17;
    (&kernelContext_45)->work_0 = work_17;
    (&kernelContext_45)->frames_0 = frames_17;
    (&kernelContext_45)->bodies_0 = bodies_17;
    (&kernelContext_45)->chunks_0 = chunks_17;
    (&kernelContext_45)->sort_0 = sort_17;
    (&kernelContext_45)->atom_0 = atom_17;
    (&kernelContext_45)->manifolds_0 = manifolds_17;
    (&kernelContext_45)->loads_0 = loads_17;
    (&kernelContext_45)->bonds_0 = bonds_17;
    (&kernelContext_45)->args_0 = args_17;
    (&kernelContext_45)->info_0 = info_17;
    (&kernelContext_45)->scan_0 = scan_17;
    (&kernelContext_45)->parts_0 = parts_17;
    (&kernelContext_45)->colliders_0 = colliders_17;
    (&kernelContext_45)->collider_info_0 = collider_info_17;
    (&kernelContext_45)->view_chunks_0 = view_chunks_17;
    (&kernelContext_45)->view_0 = view_17;
    (&kernelContext_45)->objects_0 = objects_17;
    (&kernelContext_45)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_45)->lines_0 = lines_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_45)->g_scan_0 = &g_scan_17;
    uint c_12 = id_14.x;
    bool _S277 = is_root_0(c_12, &kernelContext_45);
    if(!_S277)
    {
        return;
    }
    uint _S278 = w_slot_0(&kernelContext_45);
    uint slot_3 = *((&kernelContext_45)->work_0+(_S278 + c_12));
    uint _S279 = stage_0(c_12, &kernelContext_45);
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
            *((&kernelContext_45)->bodies_0+(7U * (slot_3 + region_0 * (&kernelContext_45)->params_0->max_bodies_0) + w_3)) = packed_float4(float4(*((&kernelContext_45)->frames_0+(_S279 + w_3))) ) ;
            w_3 = w_3 + 1U;
        }
        region_0 = region_0 + 1U;
    }
    uint _S280 = 2U * slot_3;
    *((&kernelContext_45)->frames_0+_S280) = packed_float4(float4(*((&kernelContext_45)->frames_0+(_S279 + 7U))) ) ;
    *((&kernelContext_45)->frames_0+(_S280 + 1U)) = packed_float4(float4(*((&kernelContext_45)->frames_0+(_S279 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_46;
    (&kernelContext_46)->params_0 = params_18;
    (&kernelContext_46)->work_0 = work_18;
    (&kernelContext_46)->frames_0 = frames_18;
    (&kernelContext_46)->bodies_0 = bodies_18;
    (&kernelContext_46)->chunks_0 = chunks_18;
    (&kernelContext_46)->sort_0 = sort_18;
    (&kernelContext_46)->atom_0 = atom_18;
    (&kernelContext_46)->manifolds_0 = manifolds_18;
    (&kernelContext_46)->loads_0 = loads_18;
    (&kernelContext_46)->bonds_0 = bonds_18;
    (&kernelContext_46)->args_0 = args_18;
    (&kernelContext_46)->info_0 = info_18;
    (&kernelContext_46)->scan_0 = scan_18;
    (&kernelContext_46)->parts_0 = parts_18;
    (&kernelContext_46)->colliders_0 = colliders_18;
    (&kernelContext_46)->collider_info_0 = collider_info_18;
    (&kernelContext_46)->view_chunks_0 = view_chunks_18;
    (&kernelContext_46)->view_0 = view_18;
    (&kernelContext_46)->objects_0 = objects_18;
    (&kernelContext_46)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_46)->lines_0 = lines_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_46)->g_scan_0 = &g_scan_18;
    uint c_13 = id_15.x;
    bool _S281;
    if(c_13 >= (params_18->chunk_count_0))
    {
        _S281 = true;
    }
    else
    {
        uint _S282 = body_of_0(c_13, &kernelContext_46);
        bool _S283 = dirty_0(_S282, &kernelContext_46);
        _S281 = !_S283;
    }
    if(_S281)
    {
        return;
    }
    uint _S284 = w_label_0(&kernelContext_46);
    uint device* _S285 = (&kernelContext_46)->work_0+(_S284 + c_13);
    uint r_6 = *_S285;
    uint _S286 = w_slot_0(&kernelContext_46);
    uint device* _S287 = (&kernelContext_46)->work_0+(_S286 + *_S285);
    uint slot_4 = *_S287;
    *((&kernelContext_46)->work_0+(w_body_0() + c_13)) = *_S287;
    uint _S288 = 2U * slot_4;
    float3 _S289 = (float4(*((&kernelContext_46)->frames_0+_S288)) ).xyz;
    thread Quat_0 _S290 = quat_of_0(float4(*((&kernelContext_46)->frames_0+(_S288 + 1U))) );
    Quat_0 _S291 = conj_0(&_S290);
    float inv_mass_2 = (float4(*((&kernelContext_46)->bodies_0+7U * slot_4)) ).w;
    float _S292;
    if(inv_mass_2 > 0.0f)
    {
        _S292 = 1.0f / inv_mass_2;
    }
    else
    {
        _S292 = float_bits_0(2139095040U);
    }
    float4 _S293 = float4(*((&kernelContext_46)->chunks_0+(4U * c_13 + 2U))) ;
    uint first_1 = (as_type<uint>((_S293.z)));
    uint _S294 = (as_type<uint>((_S293.w)));
    float4 _S295 = float4(*((&kernelContext_46)->chunks_0+(4U * r_6 + 3U))) ;
    uint p_3 = first_1;
    for(;;)
    {
        if(p_3 < (first_1 + _S294))
        {
        }
        else
        {
            break;
        }
        uint _S296 = 2U * p_3;
        float4 _S297 = float4(*((&kernelContext_46)->parts_0+_S296)) ;
        uint col_0 = (as_type<uint>((_S297.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_46)->parts_0+(_S296 + 1U))) );
        uint _S298 = 4U * col_0;
        float4 _S299 = float4(*((&kernelContext_46)->colliders_0+_S298)) ;
        *((&kernelContext_46)->colliders_0+_S298) = packed_float4(float4((as_type<float>((slot_4))), _S299.y, _S299.z, _S299.w)) ;
        uint _S300 = _S298 + 1U;
        packed_float4 device* _S301 = (&kernelContext_46)->colliders_0+_S300;
        float3 _S302 = _S297.xyz - _S289;
        thread Quat_0 _S303 = _S291;
        float3 _S304 = rotate_0(&_S303, _S302);
        *_S301 = packed_float4(float4(_S304, (float4(*((&kernelContext_46)->colliders_0+_S300)) ).w)) ;
        packed_float4 device* _S305 = (&kernelContext_46)->colliders_0+(_S298 + 2U);
        thread Quat_0 _S306 = _S291;
        thread Quat_0 _S307 = rot_1;
        Quat_0 _S308 = quat_mul_0(&_S306, &_S307);
        thread Quat_0 _S309 = _S308;
        Quat_0 _S310 = normalized_0(&_S309);
        thread Quat_0 _S311 = _S310;
        float4 _S312 = quat_vec_0(&_S311);
        *_S305 = packed_float4(_S312) ;
        uint _S313 = 2U * col_0;
        uint4 _S314 = uint4(*((&kernelContext_46)->collider_info_0+_S313)) ;
        *((&kernelContext_46)->collider_info_0+_S313) = packed_uint4(uint4(_S314.x, slot_4, _S314.z, _S314.w)) ;
        *((&kernelContext_46)->collider_info_0+(_S313 + 1U)) = packed_uint4(uint4((as_type<uint>((_S292))), (as_type<uint>((_S295.x))), (as_type<uint>((_S295.w))), 0U)) ;
        p_3 = p_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_47;
    (&kernelContext_47)->params_0 = params_19;
    (&kernelContext_47)->work_0 = work_19;
    (&kernelContext_47)->frames_0 = frames_19;
    (&kernelContext_47)->bodies_0 = bodies_19;
    (&kernelContext_47)->chunks_0 = chunks_19;
    (&kernelContext_47)->sort_0 = sort_19;
    (&kernelContext_47)->atom_0 = atom_19;
    (&kernelContext_47)->manifolds_0 = manifolds_19;
    (&kernelContext_47)->loads_0 = loads_19;
    (&kernelContext_47)->bonds_0 = bonds_19;
    (&kernelContext_47)->args_0 = args_19;
    (&kernelContext_47)->info_0 = info_19;
    (&kernelContext_47)->scan_0 = scan_19;
    (&kernelContext_47)->parts_0 = parts_19;
    (&kernelContext_47)->colliders_0 = colliders_19;
    (&kernelContext_47)->collider_info_0 = collider_info_19;
    (&kernelContext_47)->view_chunks_0 = view_chunks_19;
    (&kernelContext_47)->view_0 = view_19;
    (&kernelContext_47)->objects_0 = objects_19;
    (&kernelContext_47)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_47)->lines_0 = lines_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_47)->g_scan_0 = &g_scan_19;
    uint s_3 = id_16.x;
    if(s_3 >= (params_19->body_count_0))
    {
        return;
    }
    uint _S315 = a_dirty_0(&kernelContext_47);
    atomic_store_explicit((&kernelContext_47)->atom_0+(_S315 + s_3), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_0 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_7 = fract(h_0);
    float q_13 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_7);
    float t_3 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_7));
    uint k_3 = uint(h_0);
    float3 _S316;
    if(k_3 == 0U)
    {
        _S316 = float3(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_3 == 1U)
        {
            _S316 = float3(q_13, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_3 == 2U)
            {
                _S316 = float3(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_3 == 3U)
                {
                    _S316 = float3(0.34959998726844788f, q_13, 0.92000001668930054f);
                }
                else
                {
                    if(k_3 == 4U)
                    {
                        _S316 = float3(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S316 = float3(0.92000001668930054f, 0.34959998726844788f, q_13);
                    }
                }
            }
        }
    }
    return _S316;
}

void write_object_0(uint o_1, const Quat_0 thread* q_14, float3 t_4, float4 color_0, KernelContext_0 thread* kernelContext_48)
{
    float3 _S317 = rotate_0(q_14, float3(1.0f, 0.0f, 0.0f));
    float3 _S318 = rotate_0(q_14, float3(0.0f, 1.0f, 0.0f));
    float3 _S319 = rotate_0(q_14, float3(0.0f, 0.0f, 1.0f));
    uint _S320 = 4U * o_1;
    *(kernelContext_48->objects_0+_S320) = packed_float4(float4(_S317.x, _S318.x, _S319.x, t_4.x)) ;
    *(kernelContext_48->objects_0+(_S320 + 1U)) = packed_float4(float4(_S317.y, _S318.y, _S319.y, t_4.y)) ;
    *(kernelContext_48->objects_0+(_S320 + 2U)) = packed_float4(float4(_S317.z, _S318.z, _S319.z, t_4.z)) ;
    *(kernelContext_48->objects_0+(_S320 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_49;
    (&kernelContext_49)->params_0 = params_20;
    (&kernelContext_49)->work_0 = work_20;
    (&kernelContext_49)->frames_0 = frames_20;
    (&kernelContext_49)->bodies_0 = bodies_20;
    (&kernelContext_49)->chunks_0 = chunks_20;
    (&kernelContext_49)->sort_0 = sort_20;
    (&kernelContext_49)->atom_0 = atom_20;
    (&kernelContext_49)->manifolds_0 = manifolds_20;
    (&kernelContext_49)->loads_0 = loads_20;
    (&kernelContext_49)->bonds_0 = bonds_20;
    (&kernelContext_49)->args_0 = args_20;
    (&kernelContext_49)->info_0 = info_20;
    (&kernelContext_49)->scan_0 = scan_20;
    (&kernelContext_49)->parts_0 = parts_20;
    (&kernelContext_49)->colliders_0 = colliders_20;
    (&kernelContext_49)->collider_info_0 = collider_info_20;
    (&kernelContext_49)->view_chunks_0 = view_chunks_20;
    (&kernelContext_49)->view_0 = view_20;
    (&kernelContext_49)->objects_0 = objects_20;
    (&kernelContext_49)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_49)->lines_0 = lines_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_49)->g_scan_0 = &g_scan_20;
    uint k_4 = id_17.x;
    if(k_4 >= (params_20->chunk_count_0))
    {
        return;
    }
    uint _S321 = body_of_0(k_4, &kernelContext_49);
    uint b_3 = 7U * _S321;
    thread Quat_0 q_15;
    thread float3 origin_3;
    structure_pose_0(_S321, b_3, &q_15, &origin_3, &kernelContext_49);
    uint _S322 = 3U * k_4;
    float4 _S323 = float4(*((&kernelContext_49)->view_chunks_0+_S322)) ;
    float3 base_2 = (float4(*((&kernelContext_49)->view_chunks_0+(_S322 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_49)->bodies_0+b_3)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_49)->view_0->mode_0) == 1U)
    {
        float _S324 = max(length((float4(*((&kernelContext_49)->bodies_0+(b_3 + 3U))) ).xyz), length((float4(*((&kernelContext_49)->bodies_0+(b_3 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S324 > ((&kernelContext_49)->view_0->rest_speed_0))
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
        if(((&kernelContext_49)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S321);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S321);
            }
        }
    }
    uint _S325 = (as_type<uint>((_S323.w)));
    Quat_0 _S326 = quat_of_0(float4(*((&kernelContext_49)->view_chunks_0+(_S322 + 1U))) );
    thread Quat_0 _S327 = q_15;
    thread Quat_0 _S328 = _S326;
    Quat_0 _S329 = quat_mul_0(&_S327, &_S328);
    thread Quat_0 _S330 = _S329;
    Quat_0 _S331 = normalized_0(&_S330);
    float3 _S332 = _S323.xyz;
    thread Quat_0 _S333 = q_15;
    float3 _S334 = rotate_0(&_S333, _S332);
    float3 _S335 = origin_3 + _S334;
    float4 _S336 = float4(color_1, 1.0f);
    thread Quat_0 _S337 = _S331;
    write_object_0(_S325, &_S337, _S335, _S336, &kernelContext_49);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_50;
    (&kernelContext_50)->params_0 = params_21;
    (&kernelContext_50)->work_0 = work_21;
    (&kernelContext_50)->frames_0 = frames_21;
    (&kernelContext_50)->bodies_0 = bodies_21;
    (&kernelContext_50)->chunks_0 = chunks_21;
    (&kernelContext_50)->sort_0 = sort_21;
    (&kernelContext_50)->atom_0 = atom_21;
    (&kernelContext_50)->manifolds_0 = manifolds_21;
    (&kernelContext_50)->loads_0 = loads_21;
    (&kernelContext_50)->bonds_0 = bonds_21;
    (&kernelContext_50)->args_0 = args_21;
    (&kernelContext_50)->info_0 = info_21;
    (&kernelContext_50)->scan_0 = scan_21;
    (&kernelContext_50)->parts_0 = parts_21;
    (&kernelContext_50)->colliders_0 = colliders_21;
    (&kernelContext_50)->collider_info_0 = collider_info_21;
    (&kernelContext_50)->view_chunks_0 = view_chunks_21;
    (&kernelContext_50)->view_0 = view_21;
    (&kernelContext_50)->objects_0 = objects_21;
    (&kernelContext_50)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_50)->lines_0 = lines_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_50)->g_scan_0 = &g_scan_21;
    uint i_3 = id_18.x;
    if(i_3 >= (view_21->shot_count_0))
    {
        return;
    }
    uint b_4 = 7U * (&kernelContext_50)->shot_bodies_0[i_3];
    uint o_2 = (&kernelContext_50)->view_0->impactor_base_0 + i_3;
    if(((float4(*((&kernelContext_50)->bodies_0+b_4)) ).w) == 0.0f)
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
            *((&kernelContext_50)->objects_0+(4U * o_2 + w_4)) = packed_float4(float4(0.0f) ) ;
            w_4 = w_4 + 1U;
        }
        return;
    }
    thread Quat_0 _S338 = quat_of_0(float4(*((&kernelContext_50)->bodies_0+(b_4 + 1U))) );
    Quat_0 _S339 = rot_from_rubble_0(&_S338);
    float3 _S340 = from_rubble_0((float4(*((&kernelContext_50)->bodies_0+b_4)) ).xyz);
    float4 _S341 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S342 = _S339;
    write_object_0(o_2, &_S342, _S340, _S341, &kernelContext_50);
    return;
}

float3 chunk_world_0(uint k_5, KernelContext_0 thread* kernelContext_51)
{
    uint _S343 = body_of_0(k_5, kernelContext_51);
    thread Quat_0 q_16;
    thread float3 origin_4;
    structure_pose_0(_S343, 7U * _S343, &q_16, &origin_4, kernelContext_51);
    float3 _S344 = (float4(*(kernelContext_51->view_chunks_0+3U * k_5)) ).xyz;
    thread Quat_0 _S345 = q_16;
    float3 _S346 = rotate_0(&_S345, _S344);
    return origin_4 + _S346;
}

void write_line_vertex_0(uint v_8, float3 p_4, KernelContext_0 thread* kernelContext_52)
{
    uint _S347 = 7U * v_8;
    *(kernelContext_52->lines_0+_S347) = p_4.x;
    *(kernelContext_52->lines_0+(_S347 + 1U)) = p_4.y;
    *(kernelContext_52->lines_0+(_S347 + 2U)) = p_4.z;
    *(kernelContext_52->lines_0+(_S347 + 3U)) = 1.0f;
    *(kernelContext_52->lines_0+(_S347 + 4U)) = 0.86000001430511475f;
    *(kernelContext_52->lines_0+(_S347 + 5U)) = 0.25f;
    *(kernelContext_52->lines_0+(_S347 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_53;
    (&kernelContext_53)->params_0 = params_22;
    (&kernelContext_53)->work_0 = work_22;
    (&kernelContext_53)->frames_0 = frames_22;
    (&kernelContext_53)->bodies_0 = bodies_22;
    (&kernelContext_53)->chunks_0 = chunks_22;
    (&kernelContext_53)->sort_0 = sort_22;
    (&kernelContext_53)->atom_0 = atom_22;
    (&kernelContext_53)->manifolds_0 = manifolds_22;
    (&kernelContext_53)->loads_0 = loads_22;
    (&kernelContext_53)->bonds_0 = bonds_22;
    (&kernelContext_53)->args_0 = args_22;
    (&kernelContext_53)->info_0 = info_22;
    (&kernelContext_53)->scan_0 = scan_22;
    (&kernelContext_53)->parts_0 = parts_22;
    (&kernelContext_53)->colliders_0 = colliders_22;
    (&kernelContext_53)->collider_info_0 = collider_info_22;
    (&kernelContext_53)->view_chunks_0 = view_chunks_22;
    (&kernelContext_53)->view_0 = view_22;
    (&kernelContext_53)->objects_0 = objects_22;
    (&kernelContext_53)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_53)->lines_0 = lines_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_53)->g_scan_0 = &g_scan_22;
    uint i_4 = id_19.x;
    if(i_4 >= (params_22->bond_count_0))
    {
        return;
    }
    uint4 _S348 = uint4(*((&kernelContext_53)->bonds_0+i_4)) ;
    float3 _S349 = chunk_world_0(_S348.x, &kernelContext_53);
    uint _S350 = w_alive_0(&kernelContext_53);
    float3 b_5;
    if((*((&kernelContext_53)->work_0+(_S350 + i_4))) != 0U)
    {
        float3 _S351 = chunk_world_0(_S348.y, &kernelContext_53);
        b_5 = _S351;
    }
    else
    {
        b_5 = _S349;
    }
    uint _S352 = 2U * i_4;
    write_line_vertex_0(_S352, _S349, &kernelContext_53);
    write_line_vertex_0(_S352 + 1U, b_5, &kernelContext_53);
    return;
}

