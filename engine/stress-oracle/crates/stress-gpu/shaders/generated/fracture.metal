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
    uint load_model_0;
    uint strike_model_0;
    uint fracture_work_0;
    uint sweeps_0;
    float gx_0;
    float gy_0;
    float gz_0;
    float lp_c1_0;
    float lp_c2_0;
    uint ext_chunks_0;
    uint ext_adj_0;
    uint ext_mat_0;
    uint il_passes_0;
    float lp_scale_0;
    float pad4_0;
    float pad5_0;
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
    packed_float4 device* lp_0;
    packed_float4 device* ext_0;
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

float hertz_duration_0(float m_0, float r_3, float e_0, float v_2)
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_3 * e_0 * e_0 * max(abs(v_2), 0.00100000004749745f)), 0.20000000298023224f);
}

uint l_vec_0(KernelContext_0 thread* kernelContext_18)
{
    return kernelContext_18->params_0->chunk_count_0 + 6U * kernelContext_18->params_0->pool_cap_0;
}

[[kernel]] void fr_loads(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], uint device* sort_1 [[buffer(11)]], atomic<uint> device* atom_1 [[buffer(5)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], packed_float4 device* loads_1 [[buffer(12)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]], packed_float4 device* view_chunks_1 [[buffer(17)]], ViewParams_0 constant* view_1 [[buffer(16)]], packed_float4 device* objects_1 [[buffer(18)]], uint device* shot_bodies_1 [[buffer(20)]], float device* lines_1 [[buffer(19)]], packed_float4 device* geometry_1 [[buffer(21)]], packed_float4 device* shots_1 [[buffer(22)]], packed_float4 device* lp_1 [[buffer(24)]], packed_float4 device* ext_1 [[buffer(23)]])
{
    float3 f_total_0;
    float3 t_total_0;
    float m_1;
    uint p_0;
    uint kept_0;
    uint q3_0;
    bool overflow_0;
    bool overflow_1;
    float3 f_total_1;
    float3 t_total_1;
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
    (&kernelContext_19)->lp_0 = lp_1;
    (&kernelContext_19)->ext_0 = ext_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_19)->g_scan_0 = &g_scan_1;
    uint k_0 = id_0.x;
    if(k_0 >= (params_1->chunk_count_0))
    {
        return;
    }
    uint _S35 = body_of_0(k_0, &kernelContext_19);
    uint b_1 = 7U * _S35;
    thread Quat_0 q_6;
    thread float3 origin_1;
    structure_pose_0(_S35, b_1, &q_6, &origin_1, &kernelContext_19);
    uint _S36 = 4U * k_0;
    float4 _S37 = float4(*((&kernelContext_19)->chunks_0+_S36)) ;
    float3 _S38 = _S37.xyz;
    thread Quat_0 _S39 = q_6;
    float3 _S40 = rotate_0(&_S39, _S38);
    float3 centre_0 = origin_1 + _S40;
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
    uint _S41 = w_pulses_0(&kernelContext_19);
    uint _S42 = _S41 + _S36;
    uint _S43 = _S42 + 2U * from_0;
    uint off_0 = *((&kernelContext_19)->work_0+_S43);
    uint device* _S44 = (&kernelContext_19)->work_0+(_S43 + 1U);
    uint carried_0 = *_S44;
    uint room_0 = *_S44 + (&kernelContext_19)->sort_0[params_1->chunk_count_0 + 1U + k_0];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S45 = a_counters_0(&kernelContext_19);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_19)->atom_0+(_S45 + 5U + to_0), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_19)->params_0->pool_cap_0))
        {
            uint _S46 = atomic_fetch_or_explicit((&kernelContext_19)->atom_0+(_S45 + 3U), 8U, memory_order_relaxed);
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
    float4 _S50 = float4(*((&kernelContext_19)->chunks_0+(_S36 + 3U))) ;
    float _S51 = (&kernelContext_19)->params_0->band_0;
    float _S52 = (&kernelContext_19)->params_0->dt_0;
    bool _S53 = ((&kernelContext_19)->params_0->impact_peak_0) != 0U;
    uint _S54 = l_pool_0(from_0, &kernelContext_19);
    uint _S55 = _S54 + 3U * off_0;
    uint _S56 = l_pool_0(to_0, &kernelContext_19);
    uint _S57 = _S56 + 3U * base_0;
    for(;;)
    {
        float3 _S58 = float3(0.0f) ;
        uint _S59 = (&kernelContext_19)->sort_0[k_0];
        uint _S60 = (&kernelContext_19)->sort_0[_S48 + 1U + k_0];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_1 = 0U;
        f_total_0 = _S58;
        t_total_0 = _S58;
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
                uint key_1 = (uint4((&kernelContext_19)->manifolds_0[_S59 + q3_0].ids_0) ).y + 1U;
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
            Record_natural_0 r_4 = (&kernelContext_19)->manifolds_0[_S59 + p_0];
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
                float3 _S65 = t_total_0 + cross((float4(r_4.point_0) ).xyz - centre_0, f_0);
                f_total_1 = f_total_0 + f_0;
                t_total_1 = _S65;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
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
            float4 _S67 = float4(*((&kernelContext_19)->loads_0+_S66)) ;
            thread float4 a_1 = _S67;
            float4 _S68 = float4(*((&kernelContext_19)->loads_0+(_S66 + 1U))) ;
            float4 _S69 = float4(*((&kernelContext_19)->loads_0+(_S66 + 2U))) ;
            float _S70 = _S68.w;
            float mean_0 = pulse_mean_0(_S67.x, _S67.y, _S67.z, _S67.w, _S70, _S52);
            if(mean_0 != 0.0f)
            {
                float3 f_1 = _S69.xyz * float3(mean_0) ;
                float3 _S71 = t_total_0 + cross(_S68.xyz - centre_0, f_1);
                f_total_0 = f_total_0 + f_1;
                t_total_0 = _S71;
            }
            if((a_1.x + a_1.z + _S70) > _S52)
            {
                a_1.x = a_1.x - _S52;
                uint _S72 = _S57 + 3U * kept_0;
                *((&kernelContext_19)->loads_0+_S72) = packed_float4(a_1) ;
                *((&kernelContext_19)->loads_0+(_S72 + 1U)) = packed_float4(_S68) ;
                *((&kernelContext_19)->loads_0+(_S72 + 2U)) = packed_float4(_S69) ;
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
            Record_natural_0 r_5 = (&kernelContext_19)->manifolds_0[_S59 + q3_0];
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
                float3 _S79 = t_total_0 + cross((float4(r_5.point_0) ).xyz - centre_0, f_2);
                f_total_1 = f_total_0 + f_2;
                t_total_1 = _S79;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            uint kept_1;
            if(duration_1 > _S52)
            {
                if(kept_0 < room_1)
                {
                    uint _S80 = _S57 + 3U * kept_0;
                    *((&kernelContext_19)->loads_0+_S80) = packed_float4(float4(- _S52, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_19)->loads_0+(_S80 + 1U)) = packed_float4(float4((float4(r_5.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_19)->loads_0+(_S80 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
            t_total_0 = t_total_1;
            kept_0 = kept_1;
            q3_0 = q3_0 + 1U;
            overflow_0 = overflow_1;
        }
        break;
    }
    if(overflow_0)
    {
        uint _S81 = a_counters_0(&kernelContext_19);
        uint _S82 = atomic_fetch_or_explicit((&kernelContext_19)->atom_0+(_S81 + 3U), 8U, memory_order_relaxed);
    }
    uint _S83 = _S42 + 2U * to_0;
    *((&kernelContext_19)->work_0+_S83) = base_0;
    *((&kernelContext_19)->work_0+(_S83 + 1U)) = kept_0;
    *((&kernelContext_19)->loads_0+k_0) = packed_float4(float4(length(f_total_0), 0.0f, 0.0f, 0.0f)) ;
    if(((&kernelContext_19)->params_0->load_model_0) == 1U)
    {
        uint _S84 = l_vec_0(&kernelContext_19);
        uint _S85 = _S84 + 2U * k_0;
        *((&kernelContext_19)->loads_0+_S85) = packed_float4(float4(f_total_0, 0.0f)) ;
        *((&kernelContext_19)->loads_0+(_S85 + 1U)) = packed_float4(float4(t_total_0, 0.0f)) ;
    }
    return;
}

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], atomic<uint> device* atom_2 [[buffer(5)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]], packed_float4 device* geometry_2 [[buffer(21)]], packed_float4 device* shots_2 [[buffer(22)]], packed_float4 device* lp_2 [[buffer(24)]], packed_float4 device* ext_2 [[buffer(23)]])
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
    (&kernelContext_20)->lp_0 = lp_2;
    (&kernelContext_20)->ext_0 = ext_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_20)->g_scan_0 = &g_scan_2;
    uint i_0 = id_1.x;
    bool _S86;
    if(i_0 >= (params_2->bond_count_0))
    {
        _S86 = true;
    }
    else
    {
        uint _S87 = w_alive_0(&kernelContext_20);
        _S86 = (*((&kernelContext_20)->work_0+(_S87 + i_0))) == 0U;
    }
    if(_S86)
    {
        return;
    }
    uint4 _S88 = uint4(*((&kernelContext_20)->bonds_0+i_0)) ;
    uint _S89 = _S88.x;
    if((max((float4(*((&kernelContext_20)->loads_0+_S89)) ).x, (float4(*((&kernelContext_20)->loads_0+_S88.y)) ).x)) > (as_type<float>((_S88.z))))
    {
        uint _S90 = w_alive_0(&kernelContext_20);
        *((&kernelContext_20)->work_0+(_S90 + i_0)) = 0U;
        uint _S91 = a_dirty_0(&kernelContext_20);
        uint _S92 = body_of_0(_S89, &kernelContext_20);
        uint _S93 = atomic_fetch_or_explicit((&kernelContext_20)->atom_0+(_S91 + _S92), 1U, memory_order_relaxed);
        uint _S94 = a_counters_0(&kernelContext_20);
        uint _S95 = atomic_fetch_add_explicit((&kernelContext_20)->atom_0+_S94, 1U, memory_order_relaxed);
        uint _S96 = atomic_fetch_add_explicit((&kernelContext_20)->atom_0+(_S94 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint groups_of_0(uint n_2, uint size_0)
{
    uint _S97 = (n_2 + size_0 - 1U) / size_0;
    return _S97;
}

void set_args_0(uint at_0, uint x_1, KernelContext_0 thread* kernelContext_21)
{
    uint _S98 = 3U * at_0;
    *(kernelContext_21->args_0+_S98) = x_1;
    *(kernelContext_21->args_0+(_S98 + 1U)) = 1U;
    *(kernelContext_21->args_0+(_S98 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], packed_float4 device* lp_3 [[buffer(24)]], packed_float4 device* ext_3 [[buffer(23)]])
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
    (&kernelContext_22)->lp_0 = lp_3;
    (&kernelContext_22)->ext_0 = ext_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_22)->g_scan_0 = &g_scan_3;
    uint _S99 = a_counters_0(&kernelContext_22);
    uint _S100 = atomic_load_explicit((&kernelContext_22)->atom_0+_S99, memory_order_relaxed);
    bool run_0 = _S100 != 0U;
    atomic_store_explicit((&kernelContext_22)->atom_0+_S99, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_22)->atom_0+(_S99 + 5U + (&kernelContext_22)->params_0->parity_0), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S101 = atomic_fetch_add_explicit((&kernelContext_22)->atom_0+(_S99 + 4U), 1U, memory_order_relaxed);
    }
    uint _S102;
    if(run_0)
    {
        uint _S103 = groups_of_0((&kernelContext_22)->params_0->chunk_count_0, 64U);
        _S102 = _S103;
    }
    else
    {
        _S102 = 0U;
    }
    set_args_0(0U, _S102, &kernelContext_22);
    if(run_0)
    {
        uint _S104 = groups_of_0((&kernelContext_22)->params_0->bond_count_0, 64U);
        _S102 = _S104;
    }
    else
    {
        _S102 = 0U;
    }
    set_args_0(1U, _S102, &kernelContext_22);
    if(run_0)
    {
        uint _S105 = groups_of_0((&kernelContext_22)->params_0->body_count_0, 64U);
        _S102 = _S105;
    }
    else
    {
        _S102 = 0U;
    }
    set_args_0(2U, _S102, &kernelContext_22);
    if(run_0)
    {
        uint _S106 = groups_of_0((&kernelContext_22)->params_0->chunk_count_0, 1024U);
        _S102 = _S106;
    }
    else
    {
        _S102 = 0U;
    }
    set_args_0(3U, _S102, &kernelContext_22);
    if(run_0)
    {
        uint _S107 = groups_of_0((&kernelContext_22)->params_0->body_count_0, 1024U);
        _S102 = _S107;
    }
    else
    {
        _S102 = 0U;
    }
    set_args_0(4U, _S102, &kernelContext_22);
    int _S108;
    if(run_0)
    {
        _S108 = int(1);
    }
    else
    {
        _S108 = int(0);
    }
    set_args_0(5U, uint(_S108), &kernelContext_22);
    return;
}

bool dirty_0(uint body_0, KernelContext_0 thread* kernelContext_23)
{
    uint _S109 = a_dirty_0(kernelContext_23);
    uint _S110 = atomic_load_explicit(kernelContext_23->atom_0+(_S109 + body_0), memory_order_relaxed);
    return _S110 != 0U;
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

[[kernel]] void fr_uf_init(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], packed_float4 device* lp_4 [[buffer(24)]], packed_float4 device* ext_4 [[buffer(23)]])
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
    (&kernelContext_26)->lp_0 = lp_4;
    (&kernelContext_26)->ext_0 = ext_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_26)->g_scan_0 = &g_scan_4;
    uint c_2 = id_3.x;
    bool _S111;
    if(c_2 >= (params_4->chunk_count_0))
    {
        _S111 = true;
    }
    else
    {
        uint _S112 = body_of_0(c_2, &kernelContext_26);
        bool _S113 = dirty_0(_S112, &kernelContext_26);
        _S111 = !_S113;
    }
    if(_S111)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_26)->atom_0+(a_parent_0() + c_2), c_2, memory_order_relaxed);
    uint _S114 = a_count_0(&kernelContext_26);
    atomic_store_explicit((&kernelContext_26)->atom_0+(_S114 + c_2), 0U, memory_order_relaxed);
    uint _S115 = a_anchor_0(&kernelContext_26);
    atomic_store_explicit((&kernelContext_26)->atom_0+(_S115 + c_2), 0U, memory_order_relaxed);
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
        uint _S116 = a_sums_0(&kernelContext_26);
        atomic_store_explicit((&kernelContext_26)->atom_0+(_S116 + 20U * c_2 + w_2), 0U, memory_order_relaxed);
        w_2 = w_2 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], packed_float4 device* lp_5 [[buffer(24)]], packed_float4 device* ext_5 [[buffer(23)]])
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
    (&kernelContext_27)->lp_0 = lp_5;
    (&kernelContext_27)->ext_0 = ext_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_27)->g_scan_0 = &g_scan_5;
    uint s_0 = id_4.x;
    bool _S117;
    if(s_0 >= (params_5->body_count_0))
    {
        _S117 = true;
    }
    else
    {
        bool _S118 = dirty_0(s_0, &kernelContext_27);
        _S117 = !_S118;
    }
    if(_S117)
    {
        return;
    }
    uint _S119 = a_best_count_0(&kernelContext_27);
    atomic_store_explicit((&kernelContext_27)->atom_0+(_S119 + s_0), 0U, memory_order_relaxed);
    uint _S120 = a_best_root_0(&kernelContext_27);
    atomic_store_explicit((&kernelContext_27)->atom_0+(_S120 + s_0), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_2, KernelContext_0 thread* kernelContext_28)
{
    uint _S121 = a_parent_0();
    uint _S122 = atomic_load_explicit(kernelContext_28->atom_0+(_S121 + x_2), memory_order_relaxed);
    uint p_1 = _S122;
    uint _S123 = x_2;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S124;
        if(guard_0 < 1048576U)
        {
            _S124 = p_1 != _S123;
        }
        else
        {
            _S124 = false;
        }
        if(_S124)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_28->atom_0+(_S121 + p_1), memory_order_relaxed);
        if(gp_0 != p_1)
        {
            uint _S125;
            {
            uint _metal_cas_comparand = p_1;
            atomic_compare_exchange_weak_explicit(kernelContext_28->atom_0+(_S121 + _S123), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S125 = _metal_cas_comparand;
            }
        }
        uint _S126 = guard_0 + 1U;
        _S123 = p_1;
        p_1 = gp_0;
        guard_0 = _S126;
    }
    return _S123;
}

[[kernel]] void fr_uf_union(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], packed_float4 device* lp_6 [[buffer(24)]], packed_float4 device* ext_6 [[buffer(23)]])
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
    (&kernelContext_29)->lp_0 = lp_6;
    (&kernelContext_29)->ext_0 = ext_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_29)->g_scan_0 = &g_scan_6;
    uint i_1 = id_5.x;
    bool _S127;
    if(i_1 >= (params_6->bond_count_0))
    {
        _S127 = true;
    }
    else
    {
        uint _S128 = w_alive_0(&kernelContext_29);
        _S127 = (*((&kernelContext_29)->work_0+(_S128 + i_1))) == 0U;
    }
    if(_S127)
    {
        return;
    }
    uint4 _S129 = uint4(*((&kernelContext_29)->bonds_0+i_1)) ;
    uint _S130 = _S129.x;
    uint _S131 = body_of_0(_S130, &kernelContext_29);
    bool _S132 = dirty_0(_S131, &kernelContext_29);
    if(!_S132)
    {
        return;
    }
    uint _S133 = _S129.y;
    uint a_2 = _S130;
    uint b_2 = _S133;
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
        uint _S134 = find_0(a_2, &kernelContext_29);
        uint _S135 = find_0(b_2, &kernelContext_29);
        if(_S134 == _S135)
        {
            return;
        }
        uint _S136 = max(_S134, _S135);
        uint _S137 = min(_S134, _S135);
        uint was_0;
        {
        uint _metal_cas_comparand = _S136;
        atomic_compare_exchange_weak_explicit((&kernelContext_29)->atom_0+(a_parent_0() + _S136), &_metal_cas_comparand, _S137, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S136)
        {
            return;
        }
        uint _S138 = guard_1 + 1U;
        a_2 = _S136;
        b_2 = _S137;
        guard_1 = _S138;
    }
    uint _S139 = a_counters_0(&kernelContext_29);
    uint _S140 = atomic_fetch_or_explicit((&kernelContext_29)->atom_0+(_S139 + 3U), 4U, memory_order_relaxed);
    return;
}

void add64_0(uint at_1, float value_0, float scale_0, KernelContext_0 thread* kernelContext_30)
{
    float x_3 = round(value_0 * scale_0);
    float x_4;
    if(!((abs(x_3)) < 1.073741824e+09f))
    {
        uint _S141 = a_counters_0(kernelContext_30);
        uint _S142 = atomic_fetch_or_explicit(kernelContext_30->atom_0+(_S141 + 3U), 2U, memory_order_relaxed);
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
    uint _S143;
    if(v_3 < int(0))
    {
        _S143 = 4294967295U;
    }
    else
    {
        _S143 = 0U;
    }
    uint hi_0 = _S143 + carry_0;
    if(hi_0 != 0U)
    {
        uint _S144 = atomic_fetch_add_explicit(kernelContext_30->atom_0+(at_1 + 1U), hi_0, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], packed_float4 device* lp_7 [[buffer(24)]], packed_float4 device* ext_7 [[buffer(23)]])
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
    (&kernelContext_31)->lp_0 = lp_7;
    (&kernelContext_31)->ext_0 = ext_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_31)->g_scan_0 = &g_scan_7;
    uint c_3 = id_6.x;
    bool _S145;
    if(c_3 >= (params_7->chunk_count_0))
    {
        _S145 = true;
    }
    else
    {
        uint _S146 = body_of_0(c_3, &kernelContext_31);
        bool _S147 = dirty_0(_S146, &kernelContext_31);
        _S145 = !_S147;
    }
    if(_S145)
    {
        return;
    }
    uint _S148 = find_0(c_3, &kernelContext_31);
    uint _S149 = w_label_0(&kernelContext_31);
    *((&kernelContext_31)->work_0+(_S149 + c_3)) = _S148;
    uint _S150 = a_count_0(&kernelContext_31);
    uint _S151 = atomic_fetch_add_explicit((&kernelContext_31)->atom_0+(_S150 + _S148), 1U, memory_order_relaxed);
    if(((&kernelContext_31)->info_0[c_3] & 1U) != 0U)
    {
        uint _S152 = a_anchor_0(&kernelContext_31);
        uint _S153 = atomic_fetch_or_explicit((&kernelContext_31)->atom_0+(_S152 + _S148), 1U, memory_order_relaxed);
    }
    uint _S154 = 4U * c_3;
    float4 _S155 = float4(*((&kernelContext_31)->chunks_0+_S154)) ;
    float4 _S156 = float4(*((&kernelContext_31)->chunks_0+(_S154 + 1U))) ;
    float4 _S157 = float4(*((&kernelContext_31)->chunks_0+(_S154 + 2U))) ;
    float m_2 = _S155.w;
    float3 d_0 = _S155.xyz - (float4(*((&kernelContext_31)->chunks_0+4U * _S148)) ).xyz;
    float dd_0 = dot(d_0, d_0);
    uint _S158 = a_sums_0(&kernelContext_31);
    uint at_2 = _S158 + 20U * _S148;
    add64_0(at_2, m_2, (&kernelContext_31)->params_0->scale_m_0, &kernelContext_31);
    float _S159 = d_0.x;
    float _S160 = m_2 * _S159;
    add64_0(at_2 + 2U, _S160, (&kernelContext_31)->params_0->scale_s_0, &kernelContext_31);
    float _S161 = d_0.y;
    float _S162 = m_2 * _S161;
    add64_0(at_2 + 4U, _S162, (&kernelContext_31)->params_0->scale_s_0, &kernelContext_31);
    float _S163 = d_0.z;
    add64_0(at_2 + 6U, m_2 * _S163, (&kernelContext_31)->params_0->scale_s_0, &kernelContext_31);
    add64_0(at_2 + 8U, _S156.x + m_2 * (dd_0 - _S159 * _S159), (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    add64_0(at_2 + 10U, _S156.y + m_2 * (dd_0 - _S161 * _S161), (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    add64_0(at_2 + 12U, _S156.z + m_2 * (dd_0 - _S163 * _S163), (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    add64_0(at_2 + 14U, _S156.w - _S160 * _S161, (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    add64_0(at_2 + 16U, _S157.x - _S160 * _S163, (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    add64_0(at_2 + 18U, _S157.y - _S162 * _S163, (&kernelContext_31)->params_0->scale_j_0, &kernelContext_31);
    return;
}

bool is_root_0(uint c_4, KernelContext_0 thread* kernelContext_32)
{
    bool _S164;
    if(c_4 < (kernelContext_32->params_0->chunk_count_0))
    {
        uint _S165 = body_of_0(c_4, kernelContext_32);
        bool _S166 = dirty_0(_S165, kernelContext_32);
        _S164 = _S166;
    }
    else
    {
        _S164 = false;
    }
    if(_S164)
    {
        uint _S167 = w_label_0(kernelContext_32);
        _S164 = (*(kernelContext_32->work_0+(_S167 + c_4))) == c_4;
    }
    else
    {
        _S164 = false;
    }
    return _S164;
}

[[kernel]] void fr_best_count(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], packed_float4 device* lp_8 [[buffer(24)]], packed_float4 device* ext_8 [[buffer(23)]])
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
    (&kernelContext_33)->lp_0 = lp_8;
    (&kernelContext_33)->ext_0 = ext_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_33)->g_scan_0 = &g_scan_8;
    uint c_5 = id_7.x;
    bool _S168 = is_root_0(c_5, &kernelContext_33);
    if(!_S168)
    {
        return;
    }
    uint _S169 = a_best_count_0(&kernelContext_33);
    uint _S170 = body_of_0(c_5, &kernelContext_33);
    atomic<uint> device* _S171 = (&kernelContext_33)->atom_0+(_S169 + _S170);
    uint _S172 = a_count_0(&kernelContext_33);
    uint _S173 = atomic_load_explicit((&kernelContext_33)->atom_0+(_S172 + c_5), memory_order_relaxed);
    uint _S174 = atomic_fetch_max_explicit(_S171, _S173, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], packed_float4 device* lp_9 [[buffer(24)]], packed_float4 device* ext_9 [[buffer(23)]])
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
    (&kernelContext_34)->lp_0 = lp_9;
    (&kernelContext_34)->ext_0 = ext_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_34)->g_scan_0 = &g_scan_9;
    uint c_6 = id_8.x;
    bool _S175 = is_root_0(c_6, &kernelContext_34);
    if(!_S175)
    {
        return;
    }
    uint _S176 = body_of_0(c_6, &kernelContext_34);
    uint _S177 = a_count_0(&kernelContext_34);
    uint _S178 = atomic_load_explicit((&kernelContext_34)->atom_0+(_S177 + c_6), memory_order_relaxed);
    uint _S179 = a_best_count_0(&kernelContext_34);
    uint _S180 = atomic_load_explicit((&kernelContext_34)->atom_0+(_S179 + _S176), memory_order_relaxed);
    if(_S178 == _S180)
    {
        uint _S181 = a_best_root_0(&kernelContext_34);
        uint _S182 = atomic_fetch_min_explicit((&kernelContext_34)->atom_0+(_S181 + _S176), c_6, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], packed_float4 device* lp_10 [[buffer(24)]], packed_float4 device* ext_10 [[buffer(23)]])
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
    (&kernelContext_35)->lp_0 = lp_10;
    (&kernelContext_35)->ext_0 = ext_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_35)->g_scan_0 = &g_scan_10;
    uint c_7 = id_9.x;
    if(c_7 >= (params_10->chunk_count_0))
    {
        return;
    }
    uint _S183 = w_new_0(&kernelContext_35);
    uint device* _S184 = (&kernelContext_35)->work_0+(_S183 + c_7);
    bool _S185 = is_root_0(c_7, &kernelContext_35);
    bool _S186;
    if(_S185)
    {
        uint _S187 = a_best_root_0(&kernelContext_35);
        uint _S188 = body_of_0(c_7, &kernelContext_35);
        uint _S189 = atomic_load_explicit((&kernelContext_35)->atom_0+(_S187 + _S188), memory_order_relaxed);
        _S186 = _S189 != c_7;
    }
    else
    {
        _S186 = false;
    }
    uint _S190;
    if(_S186)
    {
        _S190 = 1U;
    }
    else
    {
        _S190 = 0U;
    }
    *_S184 = _S190;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], packed_float4 device* lp_11 [[buffer(24)]], packed_float4 device* ext_11 [[buffer(23)]])
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
    (&kernelContext_36)->lp_0 = lp_11;
    (&kernelContext_36)->ext_0 = ext_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_36)->g_scan_0 = &g_scan_11;
    uint t_1 = thread_0.x;
    uint _S191 = group_0.x;
    uint _S192 = 1024U * _S191 + 4U * t_1;
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
        if((_S192 + j_2) < ((&kernelContext_36)->scan_0->n_0))
        {
            step_2 = *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->in_base_0 + _S192 + j_2));
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
    uint _S193 = (*(&kernelContext_36)->g_scan_0)[t_1] - sum_2;
    j_2 = 0U;
    add_0 = _S193;
    for(;;)
    {
        if(j_2 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S192 + j_2) < ((&kernelContext_36)->scan_0->n_0))
        {
            *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->out_base_0 + _S192 + j_2)) = add_0;
        }
        uint run_1 = add_0 + v_4[j_2];
        j_2 = j_2 + 1U;
        add_0 = run_1;
    }
    if(t_1 == 255U)
    {
        *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->block_base_0 + _S191)) = (*(&kernelContext_36)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], packed_float4 device* lp_12 [[buffer(24)]], packed_float4 device* ext_12 [[buffer(23)]])
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
    (&kernelContext_37)->lp_0 = lp_12;
    (&kernelContext_37)->ext_0 = ext_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_37)->g_scan_0 = &g_scan_12;
    uint t_2 = thread_1.x;
    uint blocks_0 = (scan_12->n_0 + 1023U) / 1024U;
    bool _S194 = t_2 < blocks_0;
    uint v_5;
    if(_S194)
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
    if(_S194)
    {
        *((&kernelContext_37)->work_0+((&kernelContext_37)->scan_0->block_base_0 + t_2)) = (*(&kernelContext_37)->g_scan_0)[t_2] - v_5;
    }
    if(t_2 == 1023U)
    {
        *((&kernelContext_37)->work_0+((&kernelContext_37)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_37)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], packed_float4 device* lp_13 [[buffer(24)]], packed_float4 device* ext_13 [[buffer(23)]])
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
    (&kernelContext_38)->lp_0 = lp_13;
    (&kernelContext_38)->ext_0 = ext_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_38)->g_scan_0 = &g_scan_13;
    uint i_2 = id_10.x;
    if(i_2 >= (scan_13->n_0))
    {
        return;
    }
    uint device* _S195 = (&kernelContext_38)->work_0+((&kernelContext_38)->scan_0->out_base_0 + i_2);
    *_S195 = *_S195 + *((&kernelContext_38)->work_0+((&kernelContext_38)->scan_0->block_base_0 + i_2 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], packed_float4 device* lp_14 [[buffer(24)]], packed_float4 device* ext_14 [[buffer(23)]])
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
    (&kernelContext_39)->lp_0 = lp_14;
    (&kernelContext_39)->ext_0 = ext_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_39)->g_scan_0 = &g_scan_14;
    uint s_1 = id_11.x;
    bool _S196;
    if(s_1 >= (params_14->body_count_0))
    {
        _S196 = true;
    }
    else
    {
        uint _S197 = w_free_0(&kernelContext_39);
        _S196 = (*((&kernelContext_39)->work_0+(_S197 + s_1))) == 0U;
    }
    if(_S196)
    {
        return;
    }
    uint _S198 = w_free_list_0(&kernelContext_39);
    uint _S199 = w_free_rank_0(&kernelContext_39);
    *((&kernelContext_39)->work_0+(_S198 + *((&kernelContext_39)->work_0+(_S199 + s_1)))) = s_1;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_40)
{
    uint _S200 = w_free_list_0(kernelContext_40);
    return _S200 + kernelContext_40->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_41)
{
    uint _S201 = scan_new_blocks_0(kernelContext_41);
    return _S201 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_42)
{
    uint blocks_1 = (kernelContext_42->params_0->body_count_0 + 1023U) / 1024U;
    uint _S202 = scan_free_blocks_0(kernelContext_42);
    return *(kernelContext_42->work_0+(_S202 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], packed_float4 device* lp_15 [[buffer(24)]], packed_float4 device* ext_15 [[buffer(23)]])
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
    (&kernelContext_43)->lp_0 = lp_15;
    (&kernelContext_43)->ext_0 = ext_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_43)->g_scan_0 = &g_scan_15;
    uint c_8 = id_12.x;
    bool _S203 = is_root_0(c_8, &kernelContext_43);
    if(!_S203)
    {
        return;
    }
    uint _S204 = body_of_0(c_8, &kernelContext_43);
    uint _S205 = a_best_root_0(&kernelContext_43);
    uint _S206 = atomic_load_explicit((&kernelContext_43)->atom_0+(_S205 + _S204), memory_order_relaxed);
    uint slot_1;
    if(_S206 != c_8)
    {
        uint _S207 = w_rank_0(&kernelContext_43);
        uint device* _S208 = (&kernelContext_43)->work_0+(_S207 + c_8);
        uint k_1 = *_S208;
        uint _S209 = free_total_0(&kernelContext_43);
        if((*_S208) < _S209)
        {
            uint _S210 = w_free_list_0(&kernelContext_43);
            uint device* _S211 = (&kernelContext_43)->work_0+(_S210 + k_1);
            uint slot_2 = *_S211;
            uint _S212 = w_free_0(&kernelContext_43);
            *((&kernelContext_43)->work_0+(_S212 + *_S211)) = 0U;
            uint _S213 = a_counters_0(&kernelContext_43);
            uint _S214 = atomic_fetch_add_explicit((&kernelContext_43)->atom_0+(_S213 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S215 = a_counters_0(&kernelContext_43);
            uint _S216 = atomic_fetch_or_explicit((&kernelContext_43)->atom_0+(_S215 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S204;
        }
    }
    else
    {
        slot_1 = _S204;
    }
    uint _S217 = w_slot_0(&kernelContext_43);
    *((&kernelContext_43)->work_0+(_S217 + c_8)) = slot_1;
    return;
}

float read64_0(uint at_3, float scale_1, KernelContext_0 thread* kernelContext_44)
{
    uint _S218 = atomic_load_explicit(kernelContext_44->atom_0+(at_3 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S218)));
    uint lo_1 = atomic_load_explicit(kernelContext_44->atom_0+at_3, memory_order_relaxed);
    return (float(hi_1) * 4.294967296e+09f + float(lo_1)) / scale_1;
}

float read64_1(uint at_4, float scale_2, KernelContext_0 thread* kernelContext_45)
{
    uint _S219 = atomic_load_explicit(kernelContext_45->atom_0+(at_4 + 1U), memory_order_relaxed);
    int hi_2 = (as_type<int>((_S219)));
    uint lo_2 = atomic_load_explicit(kernelContext_45->atom_0+at_4, memory_order_relaxed);
    return (float(hi_2) * 4.294967296e+09f + float(lo_2)) / scale_2;
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
    array<float, int(3)> _S220 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S221 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S222 = { { r2_0.x, r2_0.y, r2_0.z } };
    thread array<array<float, int(3)>, int(3)> a_3;
    a_3[int(0)] = _S220;
    a_3[int(1)] = _S221;
    a_3[int(2)] = _S222;
    array<float, int(3)> _S223 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S224 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S225 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_6;
    v_6[int(0)] = _S223;
    v_6[int(1)] = _S224;
    v_6[int(2)] = _S225;
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
            bool _S226;
            if(k_2 != p_2)
            {
                _S226 = k_2 != q_9;
            }
            else
            {
                _S226 = false;
            }
            if(_S226)
            {
                float kp_0 = a_3[k_2][p_2];
                float kq_0 = a_3[k_2][q_9];
                float _S227 = c_9 * a_3[k_2][p_2] - s_2 * a_3[k_2][q_9];
                a_3[k_2][p_2] = _S227;
                a_3[p_2][k_2] = _S227;
                float _S228 = s_2 * kp_0 + c_9 * kq_0;
                a_3[k_2][q_9] = _S228;
                a_3[q_9][k_2] = _S228;
            }
            k_2 = k_2 + 1U;
        }
        float _S229 = c_9 * c_9;
        float _S230 = 2.0f * s_2 * c_9 * apq_0;
        float _S231 = s_2 * s_2;
        a_3[p_2][p_2] = _S229 * app_0 - _S230 + _S231 * aqq_0;
        a_3[q_9][q_9] = _S231 * app_0 + _S230 + _S229 * aqq_0;
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
    thread Quat_0 _S232 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S233 = normalized_0(&_S232);
    *rot_0 = _S233;
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
    thread Quat_0 _S234 = axes_1;
    Quat_0 _S235 = quat_mul_0(&_S234, q_10);
    thread Quat_0 _S236 = _S235;
    Quat_0 _S237 = normalized_0(&_S236);
    return _S237;
}

uint stage_0(uint c_10, KernelContext_0 thread* kernelContext_46)
{
    return 2U * kernelContext_46->params_0->body_count_0 + 9U * c_10;
}

float4 quat_vec_0(const Quat_0 thread* q_11)
{
    return float4(q_11->x_0, q_11->y_0, q_11->z_0, q_11->w_0);
}

[[kernel]] void fr_piece(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], packed_float4 device* lp_16 [[buffer(24)]], packed_float4 device* ext_16 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_47;
    (&kernelContext_47)->params_0 = params_16;
    (&kernelContext_47)->work_0 = work_16;
    (&kernelContext_47)->frames_0 = frames_16;
    (&kernelContext_47)->bodies_0 = bodies_16;
    (&kernelContext_47)->chunks_0 = chunks_16;
    (&kernelContext_47)->sort_0 = sort_16;
    (&kernelContext_47)->atom_0 = atom_16;
    (&kernelContext_47)->manifolds_0 = manifolds_16;
    (&kernelContext_47)->loads_0 = loads_16;
    (&kernelContext_47)->bonds_0 = bonds_16;
    (&kernelContext_47)->args_0 = args_16;
    (&kernelContext_47)->info_0 = info_16;
    (&kernelContext_47)->scan_0 = scan_16;
    (&kernelContext_47)->parts_0 = parts_16;
    (&kernelContext_47)->colliders_0 = colliders_16;
    (&kernelContext_47)->collider_info_0 = collider_info_16;
    (&kernelContext_47)->view_chunks_0 = view_chunks_16;
    (&kernelContext_47)->view_0 = view_16;
    (&kernelContext_47)->objects_0 = objects_16;
    (&kernelContext_47)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_47)->lines_0 = lines_16;
    (&kernelContext_47)->geometry_0 = geometry_16;
    (&kernelContext_47)->shots_0 = shots_16;
    (&kernelContext_47)->lp_0 = lp_16;
    (&kernelContext_47)->ext_0 = ext_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_47)->g_scan_0 = &g_scan_16;
    uint c_11 = id_13.x;
    bool _S238 = is_root_0(c_11, &kernelContext_47);
    if(!_S238)
    {
        return;
    }
    uint _S239 = body_of_0(c_11, &kernelContext_47);
    uint pb_0 = 7U * _S239;
    uint _S240 = a_sums_0(&kernelContext_47);
    uint at_5 = _S240 + 20U * c_11;
    float _S241 = read64_0(at_5, (&kernelContext_47)->params_0->scale_m_0, &kernelContext_47);
    float _S242 = read64_0(at_5 + 2U, (&kernelContext_47)->params_0->scale_s_0, &kernelContext_47);
    float _S243 = read64_0(at_5 + 4U, (&kernelContext_47)->params_0->scale_s_0, &kernelContext_47);
    float _S244 = read64_0(at_5 + 6U, (&kernelContext_47)->params_0->scale_s_0, &kernelContext_47);
    float3 d_1 = float3(_S242, _S243, _S244) / float3(max(_S241, 1.00000000317107685e-30f)) ;
    float _S245 = read64_0(at_5 + 8U, (&kernelContext_47)->params_0->scale_j_0, &kernelContext_47);
    float _S246 = read64_0(at_5 + 10U, (&kernelContext_47)->params_0->scale_j_0, &kernelContext_47);
    float _S247 = read64_0(at_5 + 12U, (&kernelContext_47)->params_0->scale_j_0, &kernelContext_47);
    float _S248 = read64_0(at_5 + 14U, (&kernelContext_47)->params_0->scale_j_0, &kernelContext_47);
    float _S249 = read64_0(at_5 + 16U, (&kernelContext_47)->params_0->scale_j_0, &kernelContext_47);
    float _S250 = read64_0(at_5 + 18U, (&kernelContext_47)->params_0->scale_j_0, &kernelContext_47);
    float dd_1 = dot(d_1, d_1);
    float _S251 = d_1.x;
    float _S252 = _S241 * _S251;
    float _S253 = d_1.y;
    float _S254 = _S248 + _S252 * _S253;
    float _S255 = d_1.z;
    float _S256 = _S249 + _S252 * _S255;
    float _S257 = _S250 + _S241 * _S253 * _S255;
    float3 com_1 = (float4(*((&kernelContext_47)->chunks_0+4U * c_11)) ).xyz + d_1;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S245 - _S241 * (dd_1 - _S251 * _S251), _S254, _S256), float3(_S254, _S246 - _S241 * (dd_1 - _S253 * _S253), _S257), float3(_S256, _S257, _S247 - _S241 * (dd_1 - _S255 * _S255)), &principal_0, &moments_1);
    uint _S258 = a_anchor_0(&kernelContext_47);
    uint _S259 = atomic_load_explicit((&kernelContext_47)->atom_0+(_S258 + c_11), memory_order_relaxed);
    bool anchored_0 = _S259 != 0U;
    thread Quat_0 q_12;
    thread float3 origin_2;
    structure_pose_0(_S239, pb_0, &q_12, &origin_2, &kernelContext_47);
    float3 p_parent_0 = (float4(*((&kernelContext_47)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_47)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_47)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S260 = q_12;
    float3 _S261 = rotate_0(&_S260, com_1);
    float3 pos_0 = to_rubble_0(origin_2 + _S261);
    thread Quat_0 _S262 = q_12;
    thread Quat_0 _S263 = principal_0;
    Quat_0 _S264 = quat_mul_0(&_S262, &_S263);
    thread Quat_0 _S265 = _S264;
    Quat_0 _S266 = normalized_0(&_S265);
    thread Quat_0 _S267 = _S266;
    Quat_0 _S268 = rot_to_rubble_0(&_S267);
    bool _S269;
    if(anchored_0)
    {
        _S269 = true;
    }
    else
    {
        _S269 = !(_S241 > 0.0f);
    }
    float inv_mass_1;
    if(_S269)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / _S241;
    }
    float3 _S270 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S271;
        if((moments_1.x) > 0.0f)
        {
            _S271 = 1.0f / moments_1.x;
        }
        else
        {
            _S271 = 0.0f;
        }
        float _S272;
        if((moments_1.y) > 0.0f)
        {
            _S272 = 1.0f / moments_1.y;
        }
        else
        {
            _S272 = 0.0f;
        }
        float _S273;
        if((moments_1.z) > 0.0f)
        {
            _S273 = 1.0f / moments_1.z;
        }
        else
        {
            _S273 = 0.0f;
        }
        inv_inertia_0 = float3(_S271, _S272, _S273);
    }
    else
    {
        inv_inertia_0 = _S270;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S270;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S270;
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
    uint _S274 = stage_0(c_11, &kernelContext_47);
    *((&kernelContext_47)->frames_0+_S274) = packed_float4(float4(pos_0, inv_mass_1)) ;
    packed_float4 device* _S275 = (&kernelContext_47)->frames_0+(_S274 + 1U);
    thread Quat_0 _S276 = _S268;
    float4 _S277 = quat_vec_0(&_S276);
    *_S275 = packed_float4(_S277) ;
    *((&kernelContext_47)->frames_0+(_S274 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_47)->frames_0+(_S274 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_47)->frames_0+(_S274 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_47)->params_0->step_0))))) ;
    *((&kernelContext_47)->frames_0+(_S274 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_47)->frames_0+(_S274 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_47)->frames_0+(_S274 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_47)->frames_0+2U * _S239)) ).w)) ;
    packed_float4 device* _S278 = (&kernelContext_47)->frames_0+(_S274 + 8U);
    thread Quat_0 _S279 = principal_0;
    float4 _S280 = quat_vec_0(&_S279);
    *_S278 = packed_float4(_S280) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], packed_float4 device* lp_17 [[buffer(24)]], packed_float4 device* ext_17 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_48;
    (&kernelContext_48)->params_0 = params_17;
    (&kernelContext_48)->work_0 = work_17;
    (&kernelContext_48)->frames_0 = frames_17;
    (&kernelContext_48)->bodies_0 = bodies_17;
    (&kernelContext_48)->chunks_0 = chunks_17;
    (&kernelContext_48)->sort_0 = sort_17;
    (&kernelContext_48)->atom_0 = atom_17;
    (&kernelContext_48)->manifolds_0 = manifolds_17;
    (&kernelContext_48)->loads_0 = loads_17;
    (&kernelContext_48)->bonds_0 = bonds_17;
    (&kernelContext_48)->args_0 = args_17;
    (&kernelContext_48)->info_0 = info_17;
    (&kernelContext_48)->scan_0 = scan_17;
    (&kernelContext_48)->parts_0 = parts_17;
    (&kernelContext_48)->colliders_0 = colliders_17;
    (&kernelContext_48)->collider_info_0 = collider_info_17;
    (&kernelContext_48)->view_chunks_0 = view_chunks_17;
    (&kernelContext_48)->view_0 = view_17;
    (&kernelContext_48)->objects_0 = objects_17;
    (&kernelContext_48)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_48)->lines_0 = lines_17;
    (&kernelContext_48)->geometry_0 = geometry_17;
    (&kernelContext_48)->shots_0 = shots_17;
    (&kernelContext_48)->lp_0 = lp_17;
    (&kernelContext_48)->ext_0 = ext_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_48)->g_scan_0 = &g_scan_17;
    uint c_12 = id_14.x;
    bool _S281 = is_root_0(c_12, &kernelContext_48);
    if(!_S281)
    {
        return;
    }
    uint _S282 = w_slot_0(&kernelContext_48);
    uint slot_3 = *((&kernelContext_48)->work_0+(_S282 + c_12));
    uint _S283 = stage_0(c_12, &kernelContext_48);
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
            *((&kernelContext_48)->bodies_0+(7U * (slot_3 + region_0 * (&kernelContext_48)->params_0->max_bodies_0) + w_3)) = packed_float4(float4(*((&kernelContext_48)->frames_0+(_S283 + w_3))) ) ;
            w_3 = w_3 + 1U;
        }
        region_0 = region_0 + 1U;
    }
    uint _S284 = 2U * slot_3;
    *((&kernelContext_48)->frames_0+_S284) = packed_float4(float4(*((&kernelContext_48)->frames_0+(_S283 + 7U))) ) ;
    *((&kernelContext_48)->frames_0+(_S284 + 1U)) = packed_float4(float4(*((&kernelContext_48)->frames_0+(_S283 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], packed_float4 device* lp_18 [[buffer(24)]], packed_float4 device* ext_18 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_49;
    (&kernelContext_49)->params_0 = params_18;
    (&kernelContext_49)->work_0 = work_18;
    (&kernelContext_49)->frames_0 = frames_18;
    (&kernelContext_49)->bodies_0 = bodies_18;
    (&kernelContext_49)->chunks_0 = chunks_18;
    (&kernelContext_49)->sort_0 = sort_18;
    (&kernelContext_49)->atom_0 = atom_18;
    (&kernelContext_49)->manifolds_0 = manifolds_18;
    (&kernelContext_49)->loads_0 = loads_18;
    (&kernelContext_49)->bonds_0 = bonds_18;
    (&kernelContext_49)->args_0 = args_18;
    (&kernelContext_49)->info_0 = info_18;
    (&kernelContext_49)->scan_0 = scan_18;
    (&kernelContext_49)->parts_0 = parts_18;
    (&kernelContext_49)->colliders_0 = colliders_18;
    (&kernelContext_49)->collider_info_0 = collider_info_18;
    (&kernelContext_49)->view_chunks_0 = view_chunks_18;
    (&kernelContext_49)->view_0 = view_18;
    (&kernelContext_49)->objects_0 = objects_18;
    (&kernelContext_49)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_49)->lines_0 = lines_18;
    (&kernelContext_49)->geometry_0 = geometry_18;
    (&kernelContext_49)->shots_0 = shots_18;
    (&kernelContext_49)->lp_0 = lp_18;
    (&kernelContext_49)->ext_0 = ext_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_49)->g_scan_0 = &g_scan_18;
    uint c_13 = id_15.x;
    bool _S285;
    if(c_13 >= (params_18->chunk_count_0))
    {
        _S285 = true;
    }
    else
    {
        uint _S286 = body_of_0(c_13, &kernelContext_49);
        bool _S287 = dirty_0(_S286, &kernelContext_49);
        _S285 = !_S287;
    }
    if(_S285)
    {
        return;
    }
    uint _S288 = w_label_0(&kernelContext_49);
    uint device* _S289 = (&kernelContext_49)->work_0+(_S288 + c_13);
    uint r_6 = *_S289;
    uint _S290 = w_slot_0(&kernelContext_49);
    uint device* _S291 = (&kernelContext_49)->work_0+(_S290 + *_S289);
    uint slot_4 = *_S291;
    *((&kernelContext_49)->work_0+(w_body_0() + c_13)) = *_S291;
    uint _S292 = 2U * slot_4;
    float3 _S293 = (float4(*((&kernelContext_49)->frames_0+_S292)) ).xyz;
    thread Quat_0 _S294 = quat_of_0(float4(*((&kernelContext_49)->frames_0+(_S292 + 1U))) );
    Quat_0 _S295 = conj_0(&_S294);
    float inv_mass_2 = (float4(*((&kernelContext_49)->bodies_0+7U * slot_4)) ).w;
    float _S296;
    if(inv_mass_2 > 0.0f)
    {
        _S296 = 1.0f / inv_mass_2;
    }
    else
    {
        _S296 = float_bits_0(2139095040U);
    }
    float4 _S297 = float4(*((&kernelContext_49)->chunks_0+(4U * c_13 + 2U))) ;
    uint first_1 = (as_type<uint>((_S297.z)));
    uint _S298 = (as_type<uint>((_S297.w)));
    float4 _S299 = float4(*((&kernelContext_49)->chunks_0+(4U * r_6 + 3U))) ;
    uint p_3 = first_1;
    for(;;)
    {
        if(p_3 < (first_1 + _S298))
        {
        }
        else
        {
            break;
        }
        uint _S300 = 2U * p_3;
        float4 _S301 = float4(*((&kernelContext_49)->parts_0+_S300)) ;
        uint col_0 = (as_type<uint>((_S301.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_49)->parts_0+(_S300 + 1U))) );
        uint _S302 = 4U * col_0;
        float4 _S303 = float4(*((&kernelContext_49)->colliders_0+_S302)) ;
        *((&kernelContext_49)->colliders_0+_S302) = packed_float4(float4((as_type<float>((slot_4))), _S303.y, _S303.z, _S303.w)) ;
        uint _S304 = _S302 + 1U;
        packed_float4 device* _S305 = (&kernelContext_49)->colliders_0+_S304;
        float3 _S306 = _S301.xyz - _S293;
        thread Quat_0 _S307 = _S295;
        float3 _S308 = rotate_0(&_S307, _S306);
        *_S305 = packed_float4(float4(_S308, (float4(*((&kernelContext_49)->colliders_0+_S304)) ).w)) ;
        packed_float4 device* _S309 = (&kernelContext_49)->colliders_0+(_S302 + 2U);
        thread Quat_0 _S310 = _S295;
        thread Quat_0 _S311 = rot_1;
        Quat_0 _S312 = quat_mul_0(&_S310, &_S311);
        thread Quat_0 _S313 = _S312;
        Quat_0 _S314 = normalized_0(&_S313);
        thread Quat_0 _S315 = _S314;
        float4 _S316 = quat_vec_0(&_S315);
        *_S309 = packed_float4(_S316) ;
        uint _S317 = 2U * col_0;
        uint4 _S318 = uint4(*((&kernelContext_49)->collider_info_0+_S317)) ;
        *((&kernelContext_49)->collider_info_0+_S317) = packed_uint4(uint4(_S318.x, slot_4, _S318.z, _S318.w)) ;
        *((&kernelContext_49)->collider_info_0+(_S317 + 1U)) = packed_uint4(uint4((as_type<uint>((_S296))), (as_type<uint>((_S299.x))), (as_type<uint>((_S299.w))), 0U)) ;
        p_3 = p_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], packed_float4 device* lp_19 [[buffer(24)]], packed_float4 device* ext_19 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_50;
    (&kernelContext_50)->params_0 = params_19;
    (&kernelContext_50)->work_0 = work_19;
    (&kernelContext_50)->frames_0 = frames_19;
    (&kernelContext_50)->bodies_0 = bodies_19;
    (&kernelContext_50)->chunks_0 = chunks_19;
    (&kernelContext_50)->sort_0 = sort_19;
    (&kernelContext_50)->atom_0 = atom_19;
    (&kernelContext_50)->manifolds_0 = manifolds_19;
    (&kernelContext_50)->loads_0 = loads_19;
    (&kernelContext_50)->bonds_0 = bonds_19;
    (&kernelContext_50)->args_0 = args_19;
    (&kernelContext_50)->info_0 = info_19;
    (&kernelContext_50)->scan_0 = scan_19;
    (&kernelContext_50)->parts_0 = parts_19;
    (&kernelContext_50)->colliders_0 = colliders_19;
    (&kernelContext_50)->collider_info_0 = collider_info_19;
    (&kernelContext_50)->view_chunks_0 = view_chunks_19;
    (&kernelContext_50)->view_0 = view_19;
    (&kernelContext_50)->objects_0 = objects_19;
    (&kernelContext_50)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_50)->lines_0 = lines_19;
    (&kernelContext_50)->geometry_0 = geometry_19;
    (&kernelContext_50)->shots_0 = shots_19;
    (&kernelContext_50)->lp_0 = lp_19;
    (&kernelContext_50)->ext_0 = ext_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_50)->g_scan_0 = &g_scan_19;
    uint s_3 = id_16.x;
    if(s_3 >= (params_19->body_count_0))
    {
        return;
    }
    uint _S319 = a_dirty_0(&kernelContext_50);
    atomic_store_explicit((&kernelContext_50)->atom_0+(_S319 + s_3), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_0 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_7 = fract(h_0);
    float q_13 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_7);
    float t_3 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_7));
    uint k_3 = uint(h_0);
    float3 _S320;
    if(k_3 == 0U)
    {
        _S320 = float3(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_3 == 1U)
        {
            _S320 = float3(q_13, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_3 == 2U)
            {
                _S320 = float3(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_3 == 3U)
                {
                    _S320 = float3(0.34959998726844788f, q_13, 0.92000001668930054f);
                }
                else
                {
                    if(k_3 == 4U)
                    {
                        _S320 = float3(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S320 = float3(0.92000001668930054f, 0.34959998726844788f, q_13);
                    }
                }
            }
        }
    }
    return _S320;
}

void write_object_0(uint o_1, const Quat_0 thread* q_14, float3 t_4, float4 color_0, KernelContext_0 thread* kernelContext_51)
{
    float3 _S321 = rotate_0(q_14, float3(1.0f, 0.0f, 0.0f));
    float3 _S322 = rotate_0(q_14, float3(0.0f, 1.0f, 0.0f));
    float3 _S323 = rotate_0(q_14, float3(0.0f, 0.0f, 1.0f));
    uint _S324 = 4U * o_1;
    *(kernelContext_51->objects_0+_S324) = packed_float4(float4(_S321.x, _S322.x, _S323.x, t_4.x)) ;
    *(kernelContext_51->objects_0+(_S324 + 1U)) = packed_float4(float4(_S321.y, _S322.y, _S323.y, t_4.y)) ;
    *(kernelContext_51->objects_0+(_S324 + 2U)) = packed_float4(float4(_S321.z, _S322.z, _S323.z, t_4.z)) ;
    *(kernelContext_51->objects_0+(_S324 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], packed_float4 device* lp_20 [[buffer(24)]], packed_float4 device* ext_20 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_52;
    (&kernelContext_52)->params_0 = params_20;
    (&kernelContext_52)->work_0 = work_20;
    (&kernelContext_52)->frames_0 = frames_20;
    (&kernelContext_52)->bodies_0 = bodies_20;
    (&kernelContext_52)->chunks_0 = chunks_20;
    (&kernelContext_52)->sort_0 = sort_20;
    (&kernelContext_52)->atom_0 = atom_20;
    (&kernelContext_52)->manifolds_0 = manifolds_20;
    (&kernelContext_52)->loads_0 = loads_20;
    (&kernelContext_52)->bonds_0 = bonds_20;
    (&kernelContext_52)->args_0 = args_20;
    (&kernelContext_52)->info_0 = info_20;
    (&kernelContext_52)->scan_0 = scan_20;
    (&kernelContext_52)->parts_0 = parts_20;
    (&kernelContext_52)->colliders_0 = colliders_20;
    (&kernelContext_52)->collider_info_0 = collider_info_20;
    (&kernelContext_52)->view_chunks_0 = view_chunks_20;
    (&kernelContext_52)->view_0 = view_20;
    (&kernelContext_52)->objects_0 = objects_20;
    (&kernelContext_52)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_52)->lines_0 = lines_20;
    (&kernelContext_52)->geometry_0 = geometry_20;
    (&kernelContext_52)->shots_0 = shots_20;
    (&kernelContext_52)->lp_0 = lp_20;
    (&kernelContext_52)->ext_0 = ext_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_52)->g_scan_0 = &g_scan_20;
    uint k_4 = id_17.x;
    if(k_4 >= (params_20->chunk_count_0))
    {
        return;
    }
    uint _S325 = body_of_0(k_4, &kernelContext_52);
    uint b_3 = 7U * _S325;
    thread Quat_0 q_15;
    thread float3 origin_3;
    structure_pose_0(_S325, b_3, &q_15, &origin_3, &kernelContext_52);
    uint _S326 = 3U * k_4;
    float4 _S327 = float4(*((&kernelContext_52)->view_chunks_0+_S326)) ;
    float3 base_2 = (float4(*((&kernelContext_52)->view_chunks_0+(_S326 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_52)->bodies_0+b_3)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_52)->view_0->mode_0) == 1U)
    {
        float _S328 = max(length((float4(*((&kernelContext_52)->bodies_0+(b_3 + 3U))) ).xyz), length((float4(*((&kernelContext_52)->bodies_0+(b_3 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S328 > ((&kernelContext_52)->view_0->rest_speed_0))
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
        if(((&kernelContext_52)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S325);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S325);
            }
        }
    }
    uint _S329 = (as_type<uint>((_S327.w)));
    Quat_0 _S330 = quat_of_0(float4(*((&kernelContext_52)->view_chunks_0+(_S326 + 1U))) );
    thread Quat_0 _S331 = q_15;
    thread Quat_0 _S332 = _S330;
    Quat_0 _S333 = quat_mul_0(&_S331, &_S332);
    thread Quat_0 _S334 = _S333;
    Quat_0 _S335 = normalized_0(&_S334);
    float3 _S336 = _S327.xyz;
    thread Quat_0 _S337 = q_15;
    float3 _S338 = rotate_0(&_S337, _S336);
    float3 _S339 = origin_3 + _S338;
    float4 _S340 = float4(color_1, 1.0f);
    thread Quat_0 _S341 = _S335;
    write_object_0(_S329, &_S341, _S339, _S340, &kernelContext_52);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], packed_float4 device* lp_21 [[buffer(24)]], packed_float4 device* ext_21 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_53;
    (&kernelContext_53)->params_0 = params_21;
    (&kernelContext_53)->work_0 = work_21;
    (&kernelContext_53)->frames_0 = frames_21;
    (&kernelContext_53)->bodies_0 = bodies_21;
    (&kernelContext_53)->chunks_0 = chunks_21;
    (&kernelContext_53)->sort_0 = sort_21;
    (&kernelContext_53)->atom_0 = atom_21;
    (&kernelContext_53)->manifolds_0 = manifolds_21;
    (&kernelContext_53)->loads_0 = loads_21;
    (&kernelContext_53)->bonds_0 = bonds_21;
    (&kernelContext_53)->args_0 = args_21;
    (&kernelContext_53)->info_0 = info_21;
    (&kernelContext_53)->scan_0 = scan_21;
    (&kernelContext_53)->parts_0 = parts_21;
    (&kernelContext_53)->colliders_0 = colliders_21;
    (&kernelContext_53)->collider_info_0 = collider_info_21;
    (&kernelContext_53)->view_chunks_0 = view_chunks_21;
    (&kernelContext_53)->view_0 = view_21;
    (&kernelContext_53)->objects_0 = objects_21;
    (&kernelContext_53)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_53)->lines_0 = lines_21;
    (&kernelContext_53)->geometry_0 = geometry_21;
    (&kernelContext_53)->shots_0 = shots_21;
    (&kernelContext_53)->lp_0 = lp_21;
    (&kernelContext_53)->ext_0 = ext_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_53)->g_scan_0 = &g_scan_21;
    uint i_3 = id_18.x;
    if(i_3 >= (view_21->shot_count_1))
    {
        return;
    }
    uint b_4 = 7U * (&kernelContext_53)->shot_bodies_0[i_3];
    uint o_2 = (&kernelContext_53)->view_0->impactor_base_0 + i_3;
    if(((float4(*((&kernelContext_53)->bodies_0+b_4)) ).w) == 0.0f)
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
            *((&kernelContext_53)->objects_0+(4U * o_2 + w_4)) = packed_float4(float4(0.0f) ) ;
            w_4 = w_4 + 1U;
        }
        return;
    }
    thread Quat_0 _S342 = quat_of_0(float4(*((&kernelContext_53)->bodies_0+(b_4 + 1U))) );
    Quat_0 _S343 = rot_from_rubble_0(&_S342);
    float3 _S344 = from_rubble_0((float4(*((&kernelContext_53)->bodies_0+b_4)) ).xyz);
    float4 _S345 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S346 = _S343;
    write_object_0(o_2, &_S346, _S344, _S345, &kernelContext_53);
    return;
}

float3 chunk_world_0(uint k_5, KernelContext_0 thread* kernelContext_54)
{
    uint _S347 = body_of_0(k_5, kernelContext_54);
    thread Quat_0 q_16;
    thread float3 origin_4;
    structure_pose_0(_S347, 7U * _S347, &q_16, &origin_4, kernelContext_54);
    float3 _S348 = (float4(*(kernelContext_54->view_chunks_0+3U * k_5)) ).xyz;
    thread Quat_0 _S349 = q_16;
    float3 _S350 = rotate_0(&_S349, _S348);
    return origin_4 + _S350;
}

void write_line_vertex_0(uint v_8, float3 p_4, KernelContext_0 thread* kernelContext_55)
{
    uint _S351 = 7U * v_8;
    *(kernelContext_55->lines_0+_S351) = p_4.x;
    *(kernelContext_55->lines_0+(_S351 + 1U)) = p_4.y;
    *(kernelContext_55->lines_0+(_S351 + 2U)) = p_4.z;
    *(kernelContext_55->lines_0+(_S351 + 3U)) = 1.0f;
    *(kernelContext_55->lines_0+(_S351 + 4U)) = 0.86000001430511475f;
    *(kernelContext_55->lines_0+(_S351 + 5U)) = 0.25f;
    *(kernelContext_55->lines_0+(_S351 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], packed_float4 device* lp_22 [[buffer(24)]], packed_float4 device* ext_22 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_56;
    (&kernelContext_56)->params_0 = params_22;
    (&kernelContext_56)->work_0 = work_22;
    (&kernelContext_56)->frames_0 = frames_22;
    (&kernelContext_56)->bodies_0 = bodies_22;
    (&kernelContext_56)->chunks_0 = chunks_22;
    (&kernelContext_56)->sort_0 = sort_22;
    (&kernelContext_56)->atom_0 = atom_22;
    (&kernelContext_56)->manifolds_0 = manifolds_22;
    (&kernelContext_56)->loads_0 = loads_22;
    (&kernelContext_56)->bonds_0 = bonds_22;
    (&kernelContext_56)->args_0 = args_22;
    (&kernelContext_56)->info_0 = info_22;
    (&kernelContext_56)->scan_0 = scan_22;
    (&kernelContext_56)->parts_0 = parts_22;
    (&kernelContext_56)->colliders_0 = colliders_22;
    (&kernelContext_56)->collider_info_0 = collider_info_22;
    (&kernelContext_56)->view_chunks_0 = view_chunks_22;
    (&kernelContext_56)->view_0 = view_22;
    (&kernelContext_56)->objects_0 = objects_22;
    (&kernelContext_56)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_56)->lines_0 = lines_22;
    (&kernelContext_56)->geometry_0 = geometry_22;
    (&kernelContext_56)->shots_0 = shots_22;
    (&kernelContext_56)->lp_0 = lp_22;
    (&kernelContext_56)->ext_0 = ext_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_56)->g_scan_0 = &g_scan_22;
    uint i_4 = id_19.x;
    if(i_4 >= (params_22->bond_count_0))
    {
        return;
    }
    uint4 _S352 = uint4(*((&kernelContext_56)->bonds_0+i_4)) ;
    float3 _S353 = chunk_world_0(_S352.x, &kernelContext_56);
    uint _S354 = w_alive_0(&kernelContext_56);
    float3 b_5;
    if((*((&kernelContext_56)->work_0+(_S354 + i_4))) != 0U)
    {
        float3 _S355 = chunk_world_0(_S352.y, &kernelContext_56);
        b_5 = _S355;
    }
    else
    {
        b_5 = _S353;
    }
    uint _S356 = 2U * i_4;
    write_line_vertex_0(_S356, _S353, &kernelContext_56);
    write_line_vertex_0(_S356 + 1U, b_5, &kernelContext_56);
    return;
}

uint a_strike_0(KernelContext_0 thread* kernelContext_57)
{
    uint _S357 = a_counters_1(kernelContext_57);
    return _S357 + 16U;
}

uint strike_at_0(uint b_6, KernelContext_0 thread* kernelContext_58)
{
    return 2U * kernelContext_58->params_0->body_count_0 + 9U * kernelContext_58->params_0->chunk_count_0 + 3U * b_6;
}

[[kernel]] void fr_strike_reset(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], atomic<uint> device* atom_23 [[buffer(5)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], packed_float4 device* lp_23 [[buffer(24)]], packed_float4 device* ext_23 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_59;
    (&kernelContext_59)->params_0 = params_23;
    (&kernelContext_59)->work_0 = work_23;
    (&kernelContext_59)->frames_0 = frames_23;
    (&kernelContext_59)->bodies_0 = bodies_23;
    (&kernelContext_59)->chunks_0 = chunks_23;
    (&kernelContext_59)->sort_0 = sort_23;
    (&kernelContext_59)->atom_0 = atom_23;
    (&kernelContext_59)->manifolds_0 = manifolds_23;
    (&kernelContext_59)->loads_0 = loads_23;
    (&kernelContext_59)->bonds_0 = bonds_23;
    (&kernelContext_59)->args_0 = args_23;
    (&kernelContext_59)->info_0 = info_23;
    (&kernelContext_59)->scan_0 = scan_23;
    (&kernelContext_59)->parts_0 = parts_23;
    (&kernelContext_59)->colliders_0 = colliders_23;
    (&kernelContext_59)->collider_info_0 = collider_info_23;
    (&kernelContext_59)->view_chunks_0 = view_chunks_23;
    (&kernelContext_59)->view_0 = view_23;
    (&kernelContext_59)->objects_0 = objects_23;
    (&kernelContext_59)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_59)->lines_0 = lines_23;
    (&kernelContext_59)->geometry_0 = geometry_23;
    (&kernelContext_59)->shots_0 = shots_23;
    (&kernelContext_59)->lp_0 = lp_23;
    (&kernelContext_59)->ext_0 = ext_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_59)->g_scan_0 = &g_scan_23;
    uint b_7 = id_20.x;
    if(b_7 >= (params_23->body_count_0))
    {
        return;
    }
    uint _S358 = a_strike_0(&kernelContext_59);
    uint at_6 = _S358 + 9U * b_7;
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
        atomic_store_explicit((&kernelContext_59)->atom_0+(at_6 + w_5), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_59)->atom_0+(at_6 + 3U + w_5), 0U, memory_order_relaxed);
        w_5 = w_5 + 1U;
    }
    atomic_store_explicit((&kernelContext_59)->atom_0+(at_6 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_59)->atom_0+(at_6 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_59)->atom_0+(at_6 + 8U), 4294967295U, memory_order_relaxed);
    uint _S359 = strike_at_0(b_7, &kernelContext_59);
    (*((&kernelContext_59)->frames_0+_S359))[int(3)] = 0.0f;
    return;
}

float3 chunk_box_world_0(uint k_6, Quat_0 thread* q_17, KernelContext_0 thread* kernelContext_60)
{
    uint _S360 = body_of_0(k_6, kernelContext_60);
    thread float3 origin_5;
    structure_pose_0(_S360, 7U * _S360, q_17, &origin_5, kernelContext_60);
    float3 _S361 = (float4(*(kernelContext_60->geometry_0+3U * k_6)) ).xyz;
    thread Quat_0 _S362 = *q_17;
    float3 _S363 = rotate_0(&_S362, _S361);
    return origin_5 + _S363;
}

uint ordered_0(float f_8)
{
    uint u_0 = (as_type<uint>((f_8)));
    uint _S364;
    if((u_0 & 2147483648U) != 0U)
    {
        _S364 = ~u_0;
    }
    else
    {
        _S364 = u_0 | 2147483648U;
    }
    return _S364;
}

[[kernel]] void fr_strike_bounds(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], atomic<uint> device* atom_24 [[buffer(5)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], packed_float4 device* lp_24 [[buffer(24)]], packed_float4 device* ext_24 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_61;
    (&kernelContext_61)->params_0 = params_24;
    (&kernelContext_61)->work_0 = work_24;
    (&kernelContext_61)->frames_0 = frames_24;
    (&kernelContext_61)->bodies_0 = bodies_24;
    (&kernelContext_61)->chunks_0 = chunks_24;
    (&kernelContext_61)->sort_0 = sort_24;
    (&kernelContext_61)->atom_0 = atom_24;
    (&kernelContext_61)->manifolds_0 = manifolds_24;
    (&kernelContext_61)->loads_0 = loads_24;
    (&kernelContext_61)->bonds_0 = bonds_24;
    (&kernelContext_61)->args_0 = args_24;
    (&kernelContext_61)->info_0 = info_24;
    (&kernelContext_61)->scan_0 = scan_24;
    (&kernelContext_61)->parts_0 = parts_24;
    (&kernelContext_61)->colliders_0 = colliders_24;
    (&kernelContext_61)->collider_info_0 = collider_info_24;
    (&kernelContext_61)->view_chunks_0 = view_chunks_24;
    (&kernelContext_61)->view_0 = view_24;
    (&kernelContext_61)->objects_0 = objects_24;
    (&kernelContext_61)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_61)->lines_0 = lines_24;
    (&kernelContext_61)->geometry_0 = geometry_24;
    (&kernelContext_61)->shots_0 = shots_24;
    (&kernelContext_61)->lp_0 = lp_24;
    (&kernelContext_61)->ext_0 = ext_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_61)->g_scan_0 = &g_scan_24;
    uint k_7 = id_21.x;
    bool _S365;
    if(k_7 >= (params_24->chunk_count_0))
    {
        _S365 = true;
    }
    else
    {
        _S365 = ((&kernelContext_61)->params_0->shot_count_0) == 0U;
    }
    if(_S365)
    {
        return;
    }
    thread Quat_0 q_18;
    float3 _S366 = chunk_box_world_0(k_7, &q_18, &kernelContext_61);
    float _S367 = (float4(*((&kernelContext_61)->geometry_0+3U * k_7)) ).w;
    uint _S368 = a_strike_0(&kernelContext_61);
    uint _S369 = body_of_0(k_7, &kernelContext_61);
    uint _S370 = _S368 + 9U * _S369;
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
        uint _S371 = ax_0;
        uint _S372 = atomic_fetch_min_explicit((&kernelContext_61)->atom_0+(_S370 + ax_0), ordered_0(_S366[ax_0] - _S367), memory_order_relaxed);
        uint _S373 = atomic_fetch_max_explicit((&kernelContext_61)->atom_0+(_S370 + 3U + ax_0), ordered_0(_S366[_S371] + _S367), memory_order_relaxed);
        ax_0 = ax_0 + 1U;
    }
    return;
}

float unordered_0(uint u_1)
{
    uint _S374;
    if((u_1 & 2147483648U) != 0U)
    {
        _S374 = u_1 & 2147483647U;
    }
    else
    {
        _S374 = ~u_1;
    }
    return (as_type<float>((_S374)));
}

[[kernel]] void fr_strike_find(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], atomic<uint> device* atom_25 [[buffer(5)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], packed_float4 device* lp_25 [[buffer(24)]], packed_float4 device* ext_25 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_62;
    (&kernelContext_62)->params_0 = params_25;
    (&kernelContext_62)->work_0 = work_25;
    (&kernelContext_62)->frames_0 = frames_25;
    (&kernelContext_62)->bodies_0 = bodies_25;
    (&kernelContext_62)->chunks_0 = chunks_25;
    (&kernelContext_62)->sort_0 = sort_25;
    (&kernelContext_62)->atom_0 = atom_25;
    (&kernelContext_62)->manifolds_0 = manifolds_25;
    (&kernelContext_62)->loads_0 = loads_25;
    (&kernelContext_62)->bonds_0 = bonds_25;
    (&kernelContext_62)->args_0 = args_25;
    (&kernelContext_62)->info_0 = info_25;
    (&kernelContext_62)->scan_0 = scan_25;
    (&kernelContext_62)->parts_0 = parts_25;
    (&kernelContext_62)->colliders_0 = colliders_25;
    (&kernelContext_62)->collider_info_0 = collider_info_25;
    (&kernelContext_62)->view_chunks_0 = view_chunks_25;
    (&kernelContext_62)->view_0 = view_25;
    (&kernelContext_62)->objects_0 = objects_25;
    (&kernelContext_62)->shot_bodies_0 = shot_bodies_25;
    (&kernelContext_62)->lines_0 = lines_25;
    (&kernelContext_62)->geometry_0 = geometry_25;
    (&kernelContext_62)->shots_0 = shots_25;
    (&kernelContext_62)->lp_0 = lp_25;
    (&kernelContext_62)->ext_0 = ext_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_62)->g_scan_0 = &g_scan_25;
    uint b_8 = id_22.x;
    bool _S375;
    if(b_8 >= (params_25->body_count_0))
    {
        _S375 = true;
    }
    else
    {
        _S375 = ((&kernelContext_62)->params_0->shot_count_0) == 0U;
    }
    if(_S375)
    {
        return;
    }
    uint _S376 = a_strike_0(&kernelContext_62);
    uint at_7 = _S376 + 9U * b_8;
    uint _S377 = atomic_load_explicit((&kernelContext_62)->atom_0+at_7, memory_order_relaxed);
    if(_S377 == 4294967295U)
    {
        return;
    }
    uint _S378 = atomic_load_explicit((&kernelContext_62)->atom_0+at_7, memory_order_relaxed);
    float _S379 = unordered_0(_S378);
    uint _S380 = atomic_load_explicit((&kernelContext_62)->atom_0+(at_7 + 1U), memory_order_relaxed);
    float _S381 = unordered_0(_S380);
    uint _S382 = atomic_load_explicit((&kernelContext_62)->atom_0+(at_7 + 2U), memory_order_relaxed);
    float3 _S383 = float3(_S379, _S381, unordered_0(_S382));
    uint _S384 = atomic_load_explicit((&kernelContext_62)->atom_0+(at_7 + 3U), memory_order_relaxed);
    float _S385 = unordered_0(_S384);
    uint _S386 = atomic_load_explicit((&kernelContext_62)->atom_0+(at_7 + 4U), memory_order_relaxed);
    float _S387 = unordered_0(_S386);
    uint _S388 = atomic_load_explicit((&kernelContext_62)->atom_0+(at_7 + 5U), memory_order_relaxed);
    float3 _S389 = float3(_S385, _S387, unordered_0(_S388));
    float3 _S390 = from_rubble_0((float4(*((&kernelContext_62)->bodies_0+(7U * b_8 + 3U))) ).xyz);
    float _S391 = float_bits_0(2139095040U);
    uint k_8 = 0U;
    for(;;)
    {
        if(k_8 < ((&kernelContext_62)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S392 = float4(*((&kernelContext_62)->shots_0+2U * k_8)) ;
        uint sb_0 = 7U * (as_type<uint>((_S392.x)));
        if(((float4(*((&kernelContext_62)->bodies_0+sb_0)) ).w) == 0.0f)
        {
            k_8 = k_8 + 1U;
            continue;
        }
        float3 c_14 = from_rubble_0((float4(*((&kernelContext_62)->bodies_0+sb_0)) ).xyz);
        float3 rel_0 = from_rubble_0((float4(*((&kernelContext_62)->bodies_0+(sb_0 + 3U))) ).xyz) - _S390;
        if(!((length(rel_0)) > ((&kernelContext_62)->params_0->strike_speed_0)))
        {
            k_8 = k_8 + 1U;
            continue;
        }
        float3 _S393 = float3(_S392.y) ;
        float3 _S394 = _S383 - _S393;
        float3 _S395 = _S389 + _S393;
        float _S396 = (&kernelContext_62)->params_0->dt_0;
        float t0_0 = 0.0f;
        float t1_0 = _S396;
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
            uint _S397 = ax_1;
            if((abs(rel_0[ax_1])) < 9.999999960041972e-13f)
            {
                uint _S398 = ax_1;
                if((c_14[ax_1]) < (_S394[ax_1]))
                {
                    _S375 = true;
                }
                else
                {
                    _S375 = (c_14[_S398]) > (_S395[ax_1]);
                }
                float t0_1;
                if(_S375)
                {
                    t0_1 = _S391;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_1 = ax_1 + 1U;
                continue;
            }
            float ta_0 = (_S394[ax_1] - c_14[ax_1]) / rel_0[_S397];
            float tb_0 = (_S395[ax_1] - c_14[ax_1]) / rel_0[_S397];
            float _S399 = min(t1_0, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_0 = _S399;
            ax_1 = ax_1 + 1U;
        }
        if(t0_0 <= t1_0)
        {
            uint _S400 = strike_at_0(b_8, &kernelContext_62);
            *((&kernelContext_62)->frames_0+_S400) = packed_float4(float4(c_14, float(k_8 + 1U))) ;
            *((&kernelContext_62)->frames_0+(_S400 + 1U)) = packed_float4(float4(rel_0, t0_0)) ;
            return;
        }
        k_8 = k_8 + 1U;
    }
    return;
}

bool struck_0(uint b_9, uint thread* shot_0, KernelContext_0 thread* kernelContext_63)
{
    uint _S401 = strike_at_0(b_9, kernelContext_63);
    uint k_9 = uint((float4(*(kernelContext_63->frames_0+_S401)) ).w);
    *shot_0 = k_9 - 1U;
    return k_9 != 0U;
}

[[kernel]] void fr_strike_entry(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], atomic<uint> device* atom_26 [[buffer(5)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], packed_float4 device* lp_26 [[buffer(24)]], packed_float4 device* ext_26 [[buffer(23)]])
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
    (&kernelContext_64)->lp_0 = lp_26;
    (&kernelContext_64)->ext_0 = ext_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_64)->g_scan_0 = &g_scan_26;
    uint k_10 = id_23.x;
    bool _S402;
    if(k_10 >= (params_26->chunk_count_0))
    {
        _S402 = true;
    }
    else
    {
        _S402 = ((&kernelContext_64)->params_0->shot_count_0) == 0U;
    }
    if(_S402)
    {
        return;
    }
    uint _S403 = body_of_0(k_10, &kernelContext_64);
    thread uint shot_1;
    bool _S404 = struck_0(_S403, &shot_1, &kernelContext_64);
    if(!_S404)
    {
        return;
    }
    uint _S405 = strike_at_0(_S403, &kernelContext_64);
    float4 _S406 = float4(*((&kernelContext_64)->frames_0+(_S405 + 1U))) ;
    float3 from_1 = (float4(*((&kernelContext_64)->frames_0+_S405)) ).xyz;
    float3 rel_1 = _S406.xyz;
    float3 d_2 = rel_1 / float3(max(length(rel_1), 9.999999960041972e-13f)) ;
    float _S407 = (float4(*((&kernelContext_64)->shots_0+2U * shot_1)) ).y;
    thread Quat_0 q_19;
    float3 _S408 = chunk_box_world_0(k_10, &q_19, &kernelContext_64);
    uint _S409 = 3U * k_10;
    Quat_0 _S410 = quat_of_0(float4(*((&kernelContext_64)->geometry_0+(_S409 + 2U))) );
    thread Quat_0 _S411 = q_19;
    thread Quat_0 _S412 = _S410;
    Quat_0 _S413 = quat_mul_0(&_S411, &_S412);
    thread Quat_0 _S414 = _S413;
    Quat_0 _S415 = normalized_0(&_S414);
    float3 _S416 = (float4(*((&kernelContext_64)->geometry_0+(_S409 + 1U))) ).xyz;
    float _S417 = float_bits_0(2139095040U);
    float _S418 = (&kernelContext_64)->params_0->dt_0;
    float t_in_0 = 0.0f;
    float t_out_0 = _S418;
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
        thread Quat_0 _S419 = _S415;
        float3 _S420 = rotate_0(&_S419, e_1);
        float o_3 = dot(from_1 - _S408, _S420);
        float dv_0 = dot(rel_1, _S420);
        float ext_27 = _S416[ax_2] + _S407;
        if((abs(dv_0)) < 9.999999960041972e-13f)
        {
            float t_in_1;
            if((abs(o_3)) > ext_27)
            {
                t_in_1 = _S417;
            }
            else
            {
                t_in_1 = t_in_0;
            }
            t_in_0 = t_in_1;
            ax_2 = ax_2 + 1U;
            continue;
        }
        float ta_1 = (- ext_27 - o_3) / dv_0;
        float tb_1 = (ext_27 - o_3) / dv_0;
        float _S421 = min(t_out_0, max(ta_1, tb_1));
        t_in_0 = max(t_in_0, min(ta_1, tb_1));
        t_out_0 = _S421;
        ax_2 = ax_2 + 1U;
    }
    uint _S422 = a_strike_0(&kernelContext_64);
    uint at_8 = _S422 + 9U * _S403;
    if(t_in_0 <= t_out_0)
    {
        uint _S423 = atomic_fetch_min_explicit((&kernelContext_64)->atom_0+(at_8 + 6U), (as_type<uint>((t_in_0))), memory_order_relaxed);
    }
    float3 r_7 = _S408 - (from_1 + rel_1 * float3(_S406.w) );
    float along_0 = dot(r_7, d_2);
    uint _S424 = atomic_fetch_min_explicit((&kernelContext_64)->atom_0+(at_8 + 7U), (as_type<uint>((length(r_7 - d_2 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f))), memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], atomic<uint> device* atom_27 [[buffer(5)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], packed_float4 device* lp_27 [[buffer(24)]], packed_float4 device* ext_28 [[buffer(23)]])
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
    (&kernelContext_65)->lp_0 = lp_27;
    (&kernelContext_65)->ext_0 = ext_28;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_65)->g_scan_0 = &g_scan_27;
    uint k_11 = id_24.x;
    bool _S425;
    if(k_11 >= (params_27->chunk_count_0))
    {
        _S425 = true;
    }
    else
    {
        _S425 = ((&kernelContext_65)->params_0->shot_count_0) == 0U;
    }
    if(_S425)
    {
        return;
    }
    uint _S426 = body_of_0(k_11, &kernelContext_65);
    thread uint shot_2;
    bool _S427 = struck_0(_S426, &shot_2, &kernelContext_65);
    if(!_S427)
    {
        return;
    }
    uint _S428 = strike_at_0(_S426, &kernelContext_65);
    float4 _S429 = float4(*((&kernelContext_65)->frames_0+_S428)) ;
    float4 _S430 = float4(*((&kernelContext_65)->frames_0+(_S428 + 1U))) ;
    float3 _S431 = _S430.xyz;
    float3 d_3 = _S431 / float3(max(length(_S431), 9.999999960041972e-13f)) ;
    thread Quat_0 q_20;
    float3 _S432 = chunk_box_world_0(k_11, &q_20, &kernelContext_65);
    float3 r_8 = _S432 - (_S429.xyz + _S431 * float3(_S430.w) );
    float along_1 = dot(r_8, d_3);
    float dist_0 = length(r_8 - d_3 * float3(along_1) ) + abs(along_1) * 0.00100000004749745f;
    uint _S433 = a_strike_0(&kernelContext_65);
    uint at_9 = _S433 + 9U * _S426;
    uint _S434 = (as_type<uint>((dist_0)));
    uint _S435 = atomic_load_explicit((&kernelContext_65)->atom_0+(at_9 + 7U), memory_order_relaxed);
    if(_S434 == _S435)
    {
        uint _S436 = atomic_fetch_min_explicit((&kernelContext_65)->atom_0+(at_9 + 8U), k_11, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], atomic<uint> device* atom_28 [[buffer(5)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], packed_float4 device* lp_28 [[buffer(24)]], packed_float4 device* ext_29 [[buffer(23)]])
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
    (&kernelContext_66)->lp_0 = lp_28;
    (&kernelContext_66)->ext_0 = ext_29;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_66)->g_scan_0 = &g_scan_28;
    uint b_10 = id_25.x;
    bool _S437;
    if(b_10 >= (params_28->body_count_0))
    {
        _S437 = true;
    }
    else
    {
        _S437 = ((&kernelContext_66)->params_0->shot_count_0) == 0U;
    }
    if(_S437)
    {
        return;
    }
    thread uint shot_3;
    bool _S438 = struck_0(b_10, &shot_3, &kernelContext_66);
    if(!_S438)
    {
        return;
    }
    uint _S439 = a_strike_0(&kernelContext_66);
    uint at_10 = _S439 + 9U * b_10;
    uint _S440 = atomic_load_explicit((&kernelContext_66)->atom_0+(at_10 + 6U), memory_order_relaxed);
    float t_in_2 = (as_type<float>((_S440)));
    uint entry_0 = atomic_load_explicit((&kernelContext_66)->atom_0+(at_10 + 8U), memory_order_relaxed);
    uint _S441 = strike_at_0(b_10, &kernelContext_66);
    if(!(t_in_2 <= ((&kernelContext_66)->params_0->dt_0)))
    {
        _S437 = true;
    }
    else
    {
        _S437 = entry_0 == 4294967295U;
    }
    if(_S437)
    {
        (*((&kernelContext_66)->frames_0+_S441))[int(3)] = 0.0f;
        return;
    }
    float3 rel_2 = (float4(*((&kernelContext_66)->frames_0+(_S441 + 1U))) ).xyz;
    float speed_0 = length(rel_2);
    float4 _S442 = float4(*((&kernelContext_66)->shots_0+2U * shot_3)) ;
    float _S443 = max(_S442.z, 0.00100000004749745f);
    float m_imp_0 = _S442.w;
    float e_imp_0 = (float4(*((&kernelContext_66)->shots_0+(2U * shot_3 + 1U))) ).x;
    float3 contact_0 = (float4(*((&kernelContext_66)->frames_0+_S441)) ).xyz + rel_2 * float3(t_in_2)  + rel_2 / float3(max(speed_0, 9.999999960041972e-13f))  * float3(_S442.y) ;
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
    *((&kernelContext_66)->frames_0+(_S441 + 2U)) = packed_float4(float4(contact_0, m_eff_0 * speed_0 / (2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (_S443 * modulus_0 * modulus_0 * max(speed_0, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

[[kernel]] void fr_strike_breaks(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], atomic<uint> device* atom_29 [[buffer(5)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], packed_float4 device* lp_29 [[buffer(24)]], packed_float4 device* ext_30 [[buffer(23)]])
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
    (&kernelContext_67)->lp_0 = lp_29;
    (&kernelContext_67)->ext_0 = ext_30;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_67)->g_scan_0 = &g_scan_29;
    uint i_5 = id_26.x;
    bool _S444;
    if(i_5 >= (params_29->bond_count_0))
    {
        _S444 = true;
    }
    else
    {
        _S444 = ((&kernelContext_67)->params_0->shot_count_0) == 0U;
    }
    if(_S444)
    {
        _S444 = true;
    }
    else
    {
        uint _S445 = w_alive_0(&kernelContext_67);
        _S444 = (*((&kernelContext_67)->work_0+(_S445 + i_5))) == 0U;
    }
    if(_S444)
    {
        return;
    }
    uint4 _S446 = uint4(*((&kernelContext_67)->bonds_0+i_5)) ;
    uint _S447 = _S446.x;
    uint _S448 = body_of_0(_S447, &kernelContext_67);
    thread uint shot_4;
    bool _S449 = struck_0(_S448, &shot_4, &kernelContext_67);
    if(!_S449)
    {
        return;
    }
    uint _S450 = strike_at_0(_S448, &kernelContext_67);
    float4 _S451 = float4(*((&kernelContext_67)->frames_0+(_S450 + 2U))) ;
    thread Quat_0 q_21;
    float3 _S452 = chunk_box_world_0(_S447, &q_21, &kernelContext_67);
    float3 _S453 = chunk_box_world_0(_S446.y, &q_21, &kernelContext_67);
    float dist_1 = length(float3(0.5f)  * (_S452 + _S453) - _S451.xyz);
    if((_S451.w * min(1.0f, (as_type<float>((_S446.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((_S446.z))))
    {
        uint _S454 = w_alive_0(&kernelContext_67);
        *((&kernelContext_67)->work_0+(_S454 + i_5)) = 0U;
        uint _S455 = a_dirty_0(&kernelContext_67);
        uint _S456 = atomic_fetch_or_explicit((&kernelContext_67)->atom_0+(_S455 + _S448), 1U, memory_order_relaxed);
        uint _S457 = a_counters_0(&kernelContext_67);
        uint _S458 = atomic_fetch_add_explicit((&kernelContext_67)->atom_0+_S457, 1U, memory_order_relaxed);
        uint _S459 = atomic_fetch_add_explicit((&kernelContext_67)->atom_0+(_S457 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_lp_0(uint b_11, KernelContext_0 thread* kernelContext_68)
{
    uint _S460 = a_strike_0(kernelContext_68);
    return _S460 + 9U * kernelContext_68->params_0->body_count_0 + 14U * b_11;
}

[[kernel]] void fr_lp_clear(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], atomic<uint> device* atom_30 [[buffer(5)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], packed_float4 device* lp_30 [[buffer(24)]], packed_float4 device* ext_31 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_69;
    (&kernelContext_69)->params_0 = params_30;
    (&kernelContext_69)->work_0 = work_30;
    (&kernelContext_69)->frames_0 = frames_30;
    (&kernelContext_69)->bodies_0 = bodies_30;
    (&kernelContext_69)->chunks_0 = chunks_30;
    (&kernelContext_69)->sort_0 = sort_30;
    (&kernelContext_69)->atom_0 = atom_30;
    (&kernelContext_69)->manifolds_0 = manifolds_30;
    (&kernelContext_69)->loads_0 = loads_30;
    (&kernelContext_69)->bonds_0 = bonds_30;
    (&kernelContext_69)->args_0 = args_30;
    (&kernelContext_69)->info_0 = info_30;
    (&kernelContext_69)->scan_0 = scan_30;
    (&kernelContext_69)->parts_0 = parts_30;
    (&kernelContext_69)->colliders_0 = colliders_30;
    (&kernelContext_69)->collider_info_0 = collider_info_30;
    (&kernelContext_69)->view_chunks_0 = view_chunks_30;
    (&kernelContext_69)->view_0 = view_30;
    (&kernelContext_69)->objects_0 = objects_30;
    (&kernelContext_69)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_69)->lines_0 = lines_30;
    (&kernelContext_69)->geometry_0 = geometry_30;
    (&kernelContext_69)->shots_0 = shots_30;
    (&kernelContext_69)->lp_0 = lp_30;
    (&kernelContext_69)->ext_0 = ext_31;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_69)->g_scan_0 = &g_scan_30;
    uint b_12 = id_27.x;
    if(b_12 >= (params_30->body_count_0))
    {
        return;
    }
    uint _S461 = a_lp_0(b_12, &kernelContext_69);
    uint w_6 = 0U;
    for(;;)
    {
        if(w_6 < 12U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit((&kernelContext_69)->atom_0+(_S461 + w_6), 0U, memory_order_relaxed);
        w_6 = w_6 + 1U;
    }
    atomic_store_explicit((&kernelContext_69)->atom_0+(_S461 + 12U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_69)->atom_0+(_S461 + 13U), 0U, memory_order_relaxed);
    return;
}

uint lp_f_0(uint k_12)
{
    return 14U * k_12 + 6U;
}

void add64w_0(uint at_11, float value_1, float scale_3, KernelContext_0 thread* kernelContext_70)
{
    float x_6 = round(value_1 * scale_3);
    float x_7;
    if(!((abs(x_6)) < 3.99999993722699776e+18f))
    {
        uint _S462 = a_counters_0(kernelContext_70);
        uint _S463 = atomic_fetch_or_explicit(kernelContext_70->atom_0+(_S462 + 3U), 2U, memory_order_relaxed);
        x_7 = 0.0f;
    }
    else
    {
        x_7 = x_6;
    }
    float u_2 = abs(x_7);
    float h_1 = floor(u_2 * 2.32830643653869629e-10f);
    uint lo_3 = uint(u_2 - h_1 * 4.294967296e+09f);
    uint hi_3 = uint(h_1);
    uint lo_4;
    uint hi_4;
    if(x_7 < 0.0f)
    {
        uint lo_5 = ~lo_3 + 1U;
        uint _S464 = ~hi_3;
        if(lo_5 == 0U)
        {
            lo_4 = 1U;
        }
        else
        {
            lo_4 = 0U;
        }
        uint _S465 = _S464 + lo_4;
        lo_4 = lo_5;
        hi_4 = _S465;
    }
    else
    {
        lo_4 = lo_3;
        hi_4 = hi_3;
    }
    uint old_1 = atomic_fetch_add_explicit(kernelContext_70->atom_0+at_11, lo_4, memory_order_relaxed);
    uint carry_1;
    if((old_1 + lo_4) < old_1)
    {
        carry_1 = 1U;
    }
    else
    {
        carry_1 = 0U;
    }
    uint hi_5 = hi_4 + carry_1;
    if(hi_5 != 0U)
    {
        uint _S466 = atomic_fetch_add_explicit(kernelContext_70->atom_0+(at_11 + 1U), hi_5, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_lp_sum(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], atomic<uint> device* atom_31 [[buffer(5)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_31 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], packed_float4 device* lp_31 [[buffer(24)]], packed_float4 device* ext_32 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_71;
    (&kernelContext_71)->params_0 = params_31;
    (&kernelContext_71)->work_0 = work_31;
    (&kernelContext_71)->frames_0 = frames_31;
    (&kernelContext_71)->bodies_0 = bodies_31;
    (&kernelContext_71)->chunks_0 = chunks_31;
    (&kernelContext_71)->sort_0 = sort_31;
    (&kernelContext_71)->atom_0 = atom_31;
    (&kernelContext_71)->manifolds_0 = manifolds_31;
    (&kernelContext_71)->loads_0 = loads_31;
    (&kernelContext_71)->bonds_0 = bonds_31;
    (&kernelContext_71)->args_0 = args_31;
    (&kernelContext_71)->info_0 = info_31;
    (&kernelContext_71)->scan_0 = scan_31;
    (&kernelContext_71)->parts_0 = parts_31;
    (&kernelContext_71)->colliders_0 = colliders_31;
    (&kernelContext_71)->collider_info_0 = collider_info_31;
    (&kernelContext_71)->view_chunks_0 = view_chunks_31;
    (&kernelContext_71)->view_0 = view_31;
    (&kernelContext_71)->objects_0 = objects_31;
    (&kernelContext_71)->shot_bodies_0 = shot_bodies_31;
    (&kernelContext_71)->lines_0 = lines_31;
    (&kernelContext_71)->geometry_0 = geometry_31;
    (&kernelContext_71)->shots_0 = shots_31;
    (&kernelContext_71)->lp_0 = lp_31;
    (&kernelContext_71)->ext_0 = ext_32;
    threadgroup array<uint, int(1024)> g_scan_31;
    (&kernelContext_71)->g_scan_0 = &g_scan_31;
    uint k_13 = id_28.x;
    if(k_13 >= (params_31->chunk_count_0))
    {
        return;
    }
    uint _S467 = body_of_0(k_13, &kernelContext_71);
    thread Quat_0 q_22;
    thread float3 origin_6;
    structure_pose_0(_S467, 7U * _S467, &q_22, &origin_6, &kernelContext_71);
    thread Quat_0 _S468 = q_22;
    Quat_0 _S469 = conj_0(&_S468);
    float4 _S470 = float4(*((&kernelContext_71)->chunks_0+4U * k_13)) ;
    uint _S471 = l_vec_0(&kernelContext_71);
    uint _S472 = _S471 + 2U * k_13;
    float3 _S473 = (float4(*((&kernelContext_71)->loads_0+_S472)) ).xyz;
    thread Quat_0 _S474 = _S469;
    float3 _S475 = rotate_0(&_S474, _S473);
    float _S476 = _S470.w;
    float3 _S477 = float3((&kernelContext_71)->params_0->gx_0, (&kernelContext_71)->params_0->gy_0, (&kernelContext_71)->params_0->gz_0);
    thread Quat_0 _S478 = _S469;
    float3 _S479 = rotate_0(&_S478, _S477);
    float3 f_9 = _S475 + float3(_S476)  * _S479;
    float3 _S480 = (float4(*((&kernelContext_71)->loads_0+(_S472 + 1U))) ).xyz;
    thread Quat_0 _S481 = _S469;
    float3 _S482 = rotate_0(&_S481, _S480);
    uint _S483 = lp_f_0(k_13);
    *((&kernelContext_71)->lp_0+_S483) = packed_float4(float4(f_9, 0.0f)) ;
    *((&kernelContext_71)->lp_0+(_S483 + 1U)) = packed_float4(float4(_S482, 0.0f)) ;
    float3 _S484 = _S482 + cross(_S470.xyz - (float4(*((&kernelContext_71)->frames_0+2U * _S467)) ).xyz, f_9);
    uint _S485 = a_lp_0(_S467, &kernelContext_71);
    uint a_4 = 0U;
    for(;;)
    {
        if(a_4 < 3U)
        {
        }
        else
        {
            break;
        }
        uint _S486 = 2U * a_4;
        add64w_0(_S485 + _S486, f_9[a_4], (&kernelContext_71)->params_0->lp_scale_0, &kernelContext_71);
        add64w_0(_S485 + 6U + _S486, _S484[a_4], (&kernelContext_71)->params_0->lp_scale_0, &kernelContext_71);
        a_4 = a_4 + 1U;
    }
    uint _S487 = atomic_fetch_min_explicit((&kernelContext_71)->atom_0+(_S485 + 12U), k_13, memory_order_relaxed);
    if(((&kernelContext_71)->info_0[k_13] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_71)->atom_0+(_S485 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

float3 read64w3_0(uint at_12, float scale_4, KernelContext_0 thread* kernelContext_72)
{
    float _S488 = read64_1(at_12, scale_4, kernelContext_72);
    float _S489 = read64_1(at_12 + 2U, scale_4, kernelContext_72);
    float _S490 = read64_1(at_12 + 4U, scale_4, kernelContext_72);
    return float3(_S488, _S489, _S490);
}

[[kernel]] void fr_lp_relief(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], atomic<uint> device* atom_32 [[buffer(5)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_32 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], packed_float4 device* lp_32 [[buffer(24)]], packed_float4 device* ext_33 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_73;
    (&kernelContext_73)->params_0 = params_32;
    (&kernelContext_73)->work_0 = work_32;
    (&kernelContext_73)->frames_0 = frames_32;
    (&kernelContext_73)->bodies_0 = bodies_32;
    (&kernelContext_73)->chunks_0 = chunks_32;
    (&kernelContext_73)->sort_0 = sort_32;
    (&kernelContext_73)->atom_0 = atom_32;
    (&kernelContext_73)->manifolds_0 = manifolds_32;
    (&kernelContext_73)->loads_0 = loads_32;
    (&kernelContext_73)->bonds_0 = bonds_32;
    (&kernelContext_73)->args_0 = args_32;
    (&kernelContext_73)->info_0 = info_32;
    (&kernelContext_73)->scan_0 = scan_32;
    (&kernelContext_73)->parts_0 = parts_32;
    (&kernelContext_73)->colliders_0 = colliders_32;
    (&kernelContext_73)->collider_info_0 = collider_info_32;
    (&kernelContext_73)->view_chunks_0 = view_chunks_32;
    (&kernelContext_73)->view_0 = view_32;
    (&kernelContext_73)->objects_0 = objects_32;
    (&kernelContext_73)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_73)->lines_0 = lines_32;
    (&kernelContext_73)->geometry_0 = geometry_32;
    (&kernelContext_73)->shots_0 = shots_32;
    (&kernelContext_73)->lp_0 = lp_32;
    (&kernelContext_73)->ext_0 = ext_33;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_73)->g_scan_0 = &g_scan_32;
    uint k_14 = id_29.x;
    if(k_14 >= (params_32->chunk_count_0))
    {
        return;
    }
    uint _S491 = body_of_0(k_14, &kernelContext_73);
    uint _S492 = a_lp_0(_S491, &kernelContext_73);
    uint _S493 = atomic_load_explicit((&kernelContext_73)->atom_0+(_S492 + 13U), memory_order_relaxed);
    bool supported_0 = _S493 != 0U;
    bool fixed_1;
    if(((&kernelContext_73)->info_0[k_14] & 1U) != 0U)
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            uint _S494 = atomic_load_explicit((&kernelContext_73)->atom_0+(_S492 + 12U), memory_order_relaxed);
            fixed_1 = _S494 == k_14;
        }
        else
        {
            fixed_1 = false;
        }
    }
    uint _S495 = lp_f_0(k_14);
    thread float4 f_10 = float4(*((&kernelContext_73)->lp_0+_S495)) ;
    uint _S496 = _S495 + 1U;
    thread float4 t_5 = float4(*((&kernelContext_73)->lp_0+_S496)) ;
    uint b_13 = 7U * _S491;
    float inv_mass_4 = (float4(*((&kernelContext_73)->bodies_0+b_13)) ).w;
    bool _S497;
    if(!supported_0)
    {
        _S497 = inv_mass_4 > 0.0f;
    }
    else
    {
        _S497 = false;
    }
    if(_S497)
    {
        float3 _S498 = read64w3_0(_S492, (&kernelContext_73)->params_0->lp_scale_0, &kernelContext_73);
        float3 _S499 = read64w3_0(_S492 + 6U, (&kernelContext_73)->params_0->lp_scale_0, &kernelContext_73);
        uint _S500 = 2U * _S491;
        Quat_0 p_5 = quat_of_0(float4(*((&kernelContext_73)->frames_0+(_S500 + 1U))) );
        float3 _S501 = (float4(*((&kernelContext_73)->bodies_0+(b_13 + 2U))) ).xyz;
        thread Quat_0 _S502 = p_5;
        Quat_0 _S503 = conj_0(&_S502);
        thread Quat_0 _S504 = _S503;
        float3 _S505 = rotate_0(&_S504, _S499);
        float3 _S506 = _S501 * _S505;
        thread Quat_0 _S507 = p_5;
        float3 _S508 = rotate_0(&_S507, _S506);
        uint _S509 = 4U * k_14;
        float4 _S510 = float4(*((&kernelContext_73)->chunks_0+_S509)) ;
        float4 _S511 = float4(*((&kernelContext_73)->chunks_0+(_S509 + 1U))) ;
        float4 _S512 = float4(*((&kernelContext_73)->chunks_0+(_S509 + 2U))) ;
        f_10.xyz = f_10.xyz - float3(_S510.w)  * (_S498 * float3(inv_mass_4)  + cross(_S508, _S510.xyz - (float4(*((&kernelContext_73)->frames_0+_S500)) ).xyz));
        float _S513 = _S508.x;
        float _S514 = _S511.w;
        float _S515 = _S508.y;
        float _S516 = _S512.x;
        float _S517 = _S508.z;
        float _S518 = _S512.y;
        t_5.xyz = t_5.xyz - float3(_S511.x * _S513 + _S514 * _S515 + _S516 * _S517, _S514 * _S513 + _S511.y * _S515 + _S518 * _S517, _S516 * _S513 + _S518 * _S515 + _S511.z * _S517);
    }
    float _S519;
    if(fixed_1)
    {
        _S519 = 1.0f;
    }
    else
    {
        _S519 = 0.0f;
    }
    f_10.w = _S519;
    *((&kernelContext_73)->lp_0+_S495) = packed_float4(f_10) ;
    *((&kernelContext_73)->lp_0+_S496) = packed_float4(t_5) ;
    return;
}

uint adj_bond_0(uint at_13, KernelContext_0 thread* kernelContext_74)
{
    return (as_type<uint>(((float4(*(kernelContext_74->ext_0+(kernelContext_74->params_0->ext_adj_0 + at_13 / 4U))) )[at_13 % 4U])));
}

struct JointBond_0
{
    float4 geom0_0;
    float4 geom1_0;
    float4 stiff0_0;
    float4 stiff1_0;
    float4 rebar0_0;
    float4 rebar1_0;
    uint4 ids_1;
};

struct LpBond_0
{
    float3 centroid_0;
    float3 n_3;
    float3 t1_1;
    float3 t2_0;
    JointBond_0 law_0;
};

LpBond_0 lp_bond_0(uint i_6, KernelContext_0 thread* kernelContext_75)
{
    uint o_4 = 10U * i_6;
    thread LpBond_0 b_14;
    (&b_14)->centroid_0 = (float4(*(kernelContext_75->ext_0+o_4)) ).xyz;
    float3 _S520 = (float4(*(kernelContext_75->ext_0+(o_4 + 1U))) ).xyz;
    (&b_14)->n_3 = _S520;
    float3 _S521 = (float4(*(kernelContext_75->ext_0+(o_4 + 2U))) ).xyz;
    (&b_14)->t1_1 = _S521;
    (&b_14)->t2_0 = cross(_S520, _S521);
    (&(&b_14)->law_0)->geom0_0 = float4(*(kernelContext_75->ext_0+(o_4 + 3U))) ;
    (&(&b_14)->law_0)->geom1_0 = float4(*(kernelContext_75->ext_0+(o_4 + 4U))) ;
    (&(&b_14)->law_0)->stiff0_0 = float4(*(kernelContext_75->ext_0+(o_4 + 5U))) ;
    (&(&b_14)->law_0)->stiff1_0 = float4(*(kernelContext_75->ext_0+(o_4 + 6U))) ;
    (&(&b_14)->law_0)->rebar0_0 = float4(*(kernelContext_75->ext_0+(o_4 + 7U))) ;
    (&(&b_14)->law_0)->rebar1_0 = float4(*(kernelContext_75->ext_0+(o_4 + 8U))) ;
    (&(&b_14)->law_0)->ids_1 = (as_type<uint4>((float4(*(kernelContext_75->ext_0+(o_4 + 9U))) )));
    return b_14;
}

uint tri_0(uint i_7, uint j_3)
{
    return i_7 * (i_7 + 1U) / 2U + j_3;
}

uint lp_l_0(uint k_15)
{
    return 14U * k_15 + 8U;
}

[[kernel]] void fr_lp_block(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], atomic<uint> device* atom_33 [[buffer(5)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], packed_float4 device* lp_33 [[buffer(24)]], packed_float4 device* ext_34 [[buffer(23)]])
{
    uint q_23;
    uint a_5;
    uint p2_0;
    uint q2_0;
    uint a2_0;
    float s_4;
    thread KernelContext_0 kernelContext_76;
    (&kernelContext_76)->params_0 = params_33;
    (&kernelContext_76)->work_0 = work_33;
    (&kernelContext_76)->frames_0 = frames_33;
    (&kernelContext_76)->bodies_0 = bodies_33;
    (&kernelContext_76)->chunks_0 = chunks_33;
    (&kernelContext_76)->sort_0 = sort_33;
    (&kernelContext_76)->atom_0 = atom_33;
    (&kernelContext_76)->manifolds_0 = manifolds_33;
    (&kernelContext_76)->loads_0 = loads_33;
    (&kernelContext_76)->bonds_0 = bonds_33;
    (&kernelContext_76)->args_0 = args_33;
    (&kernelContext_76)->info_0 = info_33;
    (&kernelContext_76)->scan_0 = scan_33;
    (&kernelContext_76)->parts_0 = parts_33;
    (&kernelContext_76)->colliders_0 = colliders_33;
    (&kernelContext_76)->collider_info_0 = collider_info_33;
    (&kernelContext_76)->view_chunks_0 = view_chunks_33;
    (&kernelContext_76)->view_0 = view_33;
    (&kernelContext_76)->objects_0 = objects_33;
    (&kernelContext_76)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_76)->lines_0 = lines_33;
    (&kernelContext_76)->geometry_0 = geometry_33;
    (&kernelContext_76)->shots_0 = shots_33;
    (&kernelContext_76)->lp_0 = lp_33;
    (&kernelContext_76)->ext_0 = ext_34;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_76)->g_scan_0 = &g_scan_33;
    uint k_16 = id_30.x;
    bool ok_0;
    if(k_16 >= (params_33->chunk_count_0))
    {
        ok_0 = true;
    }
    else
    {
        ok_0 = ((float4(*((&kernelContext_76)->lp_0+lp_f_0(k_16))) ).w) != 0.0f;
    }
    if(ok_0)
    {
        return;
    }
    thread array<array<float, int(6)>, int(6)> d_4;
    uint p_6 = 0U;
    for(;;)
    {
        if(p_6 < 6U)
        {
        }
        else
        {
            break;
        }
        q_23 = 0U;
        for(;;)
        {
            if(q_23 < 6U)
            {
            }
            else
            {
                break;
            }
            d_4[p_6][q_23] = 0.0f;
            q_23 = q_23 + 1U;
        }
        p_6 = p_6 + 1U;
    }
    uint _S522 = 4U * k_16;
    float4 _S523 = float4(*((&kernelContext_76)->ext_0+((&kernelContext_76)->params_0->ext_chunks_0 + _S522))) ;
    uint _S524 = (as_type<uint>((_S523.x)));
    uint _S525 = (as_type<uint>((_S523.y)));
    q_23 = 0U;
    for(;;)
    {
        if(q_23 < _S525)
        {
        }
        else
        {
            break;
        }
        uint _S526 = adj_bond_0(_S524 + q_23, &kernelContext_76);
        uint _S527 = w_alive_0(&kernelContext_76);
        if((*((&kernelContext_76)->work_0+(_S527 + _S526))) == 0U)
        {
            q_23 = q_23 + 1U;
            continue;
        }
        LpBond_0 _S528 = lp_bond_0(_S526, &kernelContext_76);
        float3 _S529 = _S528.centroid_0 - (float4(*((&kernelContext_76)->chunks_0+_S522)) ).xyz;
        float _S530 = _S528.law_0.stiff0_0.y;
        array<float, int(6)> _S531 = { { _S530, _S530, _S528.law_0.stiff0_0.x, _S528.law_0.stiff0_0.z, _S528.law_0.stiff0_0.w, _S528.law_0.stiff1_0.x } };
        array<float3, int(3)> _S532 = { { _S528.t1_1, _S528.t2_0, _S528.n_3 } };
        thread array<array<float, int(6)>, int(6)> j_4;
        a_5 = 0U;
        for(;;)
        {
            if(a_5 < 3U)
            {
            }
            else
            {
                break;
            }
            float3 c_15 = cross(_S529, _S532[a_5]);
            float _S533 = _S532[a_5].x;
            j_4[a_5][int(0)] = _S533;
            float _S534 = _S532[a_5].y;
            j_4[a_5][int(1)] = _S534;
            float _S535 = _S532[a_5].z;
            j_4[a_5][int(2)] = _S535;
            j_4[a_5][int(3)] = c_15.x;
            j_4[a_5][int(4)] = c_15.y;
            j_4[a_5][int(5)] = c_15.z;
            j_4[3U + a_5][int(0)] = 0.0f;
            j_4[3U + a_5][int(1)] = 0.0f;
            j_4[3U + a_5][int(2)] = 0.0f;
            j_4[3U + a_5][int(3)] = _S533;
            j_4[3U + a_5][int(4)] = _S534;
            j_4[3U + a_5][int(5)] = _S535;
            a_5 = a_5 + 1U;
        }
        p2_0 = 0U;
        for(;;)
        {
            if(p2_0 < 6U)
            {
            }
            else
            {
                break;
            }
            q2_0 = 0U;
            for(;;)
            {
                if(q2_0 < 6U)
                {
                }
                else
                {
                    break;
                }
                a2_0 = 0U;
                s_4 = 0.0f;
                for(;;)
                {
                    if(a2_0 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_5 = s_4 + j_4[a2_0][p2_0] * _S531[a2_0] * j_4[a2_0][q2_0];
                    a2_0 = a2_0 + 1U;
                    s_4 = s_5;
                }
                d_4[p2_0][q2_0] = d_4[p2_0][q2_0] + s_4;
                q2_0 = q2_0 + 1U;
            }
            p2_0 = p2_0 + 1U;
        }
        q_23 = q_23 + 1U;
    }
    thread array<float, int(21)> l_0;
    a_5 = 0U;
    for(;;)
    {
        if(a_5 < 21U)
        {
        }
        else
        {
            break;
        }
        l_0[a_5] = 0.0f;
        a_5 = a_5 + 1U;
    }
    ok_0 = true;
    p2_0 = 0U;
    for(;;)
    {
        if(p2_0 < 6U)
        {
        }
        else
        {
            break;
        }
        bool ok_1 = ok_0;
        q2_0 = 0U;
        for(;;)
        {
            if(q2_0 <= p2_0)
            {
            }
            else
            {
                break;
            }
            a2_0 = 0U;
            s_4 = d_4[p2_0][q2_0];
            for(;;)
            {
                if(a2_0 < q2_0)
                {
                }
                else
                {
                    break;
                }
                float s_6 = s_4 - l_0[tri_0(p2_0, a2_0)] * l_0[tri_0(q2_0, a2_0)];
                a2_0 = a2_0 + 1U;
                s_4 = s_6;
            }
            if(p2_0 == q2_0)
            {
                bool _S536;
                if(ok_1)
                {
                    _S536 = s_4 > 0.0f;
                }
                else
                {
                    _S536 = false;
                }
                l_0[tri_0(p2_0, p2_0)] = sqrt(max(s_4, 1.00000000317107685e-30f));
                ok_1 = _S536;
            }
            else
            {
                l_0[tri_0(p2_0, q2_0)] = s_4 / l_0[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + 1U;
        }
        uint _S537 = p2_0 + 1U;
        ok_0 = ok_1;
        p2_0 = _S537;
    }
    if(!ok_0)
    {
        (*((&kernelContext_76)->lp_0+lp_f_0(k_16)))[int(3)] = 1.0f;
        uint _S538 = a_counters_0(&kernelContext_76);
        uint _S539 = atomic_fetch_or_explicit((&kernelContext_76)->atom_0+(_S538 + 3U), 32U, memory_order_relaxed);
        return;
    }
    q2_0 = 0U;
    for(;;)
    {
        if(q2_0 < 6U)
        {
        }
        else
        {
            break;
        }
        packed_float4 device* _S540 = (&kernelContext_76)->lp_0+(lp_l_0(k_16) + q2_0);
        uint _S541 = 4U * q2_0;
        float _S542 = l_0[_S541];
        uint _S543 = _S541 + 1U;
        if(_S543 < 21U)
        {
            s_4 = l_0[_S543];
        }
        else
        {
            s_4 = 0.0f;
        }
        uint _S544 = _S541 + 2U;
        float _S545;
        if(_S544 < 21U)
        {
            _S545 = l_0[_S544];
        }
        else
        {
            _S545 = 0.0f;
        }
        uint _S546 = _S541 + 3U;
        float _S547;
        if(_S546 < 21U)
        {
            _S547 = l_0[_S546];
        }
        else
        {
            _S547 = 0.0f;
        }
        *_S540 = packed_float4(float4(_S542, s_4, _S545, _S547)) ;
        q2_0 = q2_0 + 1U;
    }
    return;
}

uint lp_x_0(uint k_17, uint copy_0)
{
    return 14U * k_17 + 2U * copy_0;
}

uint lp_d_0(uint k_18)
{
    return 14U * k_18 + 4U;
}

void lp_force_0(const LpBond_0 thread* b_15, uint ka_0, uint kb_0, uint c_16, float3 thread* q_lin_0, float3 thread* q_ang_0, KernelContext_0 thread* kernelContext_77)
{
    uint _S548 = lp_x_0(ka_0, c_16);
    uint _S549 = lp_x_0(kb_0, c_16);
    float3 _S550 = (float4(*(kernelContext_77->lp_0+(_S549 + 1U))) ).xyz;
    float3 _S551 = (float4(*(kernelContext_77->lp_0+(_S548 + 1U))) ).xyz;
    float3 delta_0 = (float4(*(kernelContext_77->lp_0+_S549)) ).xyz + cross(_S550, b_15->centroid_0 - (float4(*(kernelContext_77->chunks_0+4U * kb_0)) ).xyz) - ((float4(*(kernelContext_77->lp_0+_S548)) ).xyz + cross(_S551, b_15->centroid_0 - (float4(*(kernelContext_77->chunks_0+4U * ka_0)) ).xyz));
    float3 phi_0 = _S550 - _S551;
    float4 _S552 = (&b_15->law_0)->stiff0_0;
    float kt_0 = (&b_15->law_0)->stiff1_0.x;
    float _S553 = (&b_15->law_0)->stiff0_0.y;
    float3 _S554 = b_15->t1_1;
    float3 _S555 = b_15->t2_0;
    float3 _S556 = b_15->n_3;
    *q_lin_0 = float3(_S553 * dot(delta_0, b_15->t1_1), _S553 * dot(delta_0, b_15->t2_0), (&b_15->law_0)->stiff0_0.x * dot(delta_0, b_15->n_3));
    *q_ang_0 = float3(_S552.z * dot(phi_0, _S554), _S552.w * dot(phi_0, _S555), kt_0 * dot(phi_0, _S556));
    return;
}

void lp_sweep_0(uint k_19, uint from_2, KernelContext_0 thread* kernelContext_78)
{
    float s_7;
    uint to_1 = 1U - from_2;
    uint _S557 = lp_x_0(k_19, from_2);
    float4 _S558 = float4(*(kernelContext_78->lp_0+_S557)) ;
    float4 _S559 = float4(*(kernelContext_78->lp_0+(_S557 + 1U))) ;
    uint _S560 = lp_f_0(k_19);
    float4 _S561 = float4(*(kernelContext_78->lp_0+_S560)) ;
    if((_S561.w) != 0.0f)
    {
        uint _S562 = lp_x_0(k_19, to_1);
        *(kernelContext_78->lp_0+_S562) = packed_float4(_S558) ;
        *(kernelContext_78->lp_0+(_S562 + 1U)) = packed_float4(_S559) ;
        uint _S563 = lp_d_0(k_19);
        packed_float4 _S564 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_78->lp_0+_S563) = _S564;
        *(kernelContext_78->lp_0+(_S563 + 1U)) = _S564;
        return;
    }
    float _S565 = _S561.y;
    float _S566 = _S561.z;
    thread array<float, int(6)> r_9;
    r_9[int(0)] = _S561.x;
    r_9[int(1)] = _S565;
    r_9[int(2)] = _S566;
    r_9[int(3)] = 0.0f;
    r_9[int(4)] = 0.0f;
    r_9[int(5)] = 0.0f;
    float3 t_6 = (float4(*(kernelContext_78->lp_0+(_S560 + 1U))) ).xyz;
    r_9[int(3)] = t_6.x;
    r_9[int(4)] = t_6.y;
    r_9[int(5)] = t_6.z;
    uint _S567 = 4U * k_19;
    float4 _S568 = float4(*(kernelContext_78->ext_0+(kernelContext_78->params_0->ext_chunks_0 + _S567))) ;
    uint _S569 = (as_type<uint>((_S568.x)));
    uint _S570 = (as_type<uint>((_S568.y)));
    uint e_2 = 0U;
    for(;;)
    {
        if(e_2 < _S570)
        {
        }
        else
        {
            break;
        }
        uint _S571 = adj_bond_0(_S569 + e_2, kernelContext_78);
        uint _S572 = w_alive_0(kernelContext_78);
        if((*(kernelContext_78->work_0+(_S572 + _S571))) == 0U)
        {
            e_2 = e_2 + 1U;
            continue;
        }
        uint4 _S573 = uint4(*(kernelContext_78->bonds_0+_S571)) ;
        LpBond_0 _S574 = lp_bond_0(_S571, kernelContext_78);
        uint _S575 = _S573.x;
        uint _S576 = _S573.y;
        thread LpBond_0 _S577 = _S574;
        thread float3 q_lin_1;
        thread float3 q_ang_1;
        lp_force_0(&_S577, _S575, _S576, from_2, &q_lin_1, &q_ang_1, kernelContext_78);
        float3 fb_0 = _S574.t1_1 * float3(q_lin_1.x)  + _S574.t2_0 * float3(q_lin_1.y)  + _S574.n_3 * float3(q_lin_1.z) ;
        float3 mb_0 = _S574.t1_1 * float3(q_ang_1.x)  + _S574.t2_0 * float3(q_ang_1.y)  + _S574.n_3 * float3(q_ang_1.z) ;
        float3 arm_0 = _S574.centroid_0 - (float4(*(kernelContext_78->chunks_0+_S567)) ).xyz;
        bool _S578 = k_19 == _S575;
        float3 lf_0;
        if(_S578)
        {
            lf_0 = fb_0;
        }
        else
        {
            lf_0 = - fb_0;
        }
        float3 _S579;
        if(_S578)
        {
            _S579 = mb_0;
        }
        else
        {
            _S579 = - mb_0;
        }
        float3 lm_0 = _S579 + cross(arm_0, lf_0);
        r_9[int(0)] = r_9[int(0)] + lf_0.x;
        r_9[int(1)] = r_9[int(1)] + lf_0.y;
        r_9[int(2)] = r_9[int(2)] + lf_0.z;
        r_9[int(3)] = r_9[int(3)] + lm_0.x;
        r_9[int(4)] = r_9[int(4)] + lm_0.y;
        r_9[int(5)] = r_9[int(5)] + lm_0.z;
        e_2 = e_2 + 1U;
    }
    thread array<float, int(21)> l_1;
    uint z0_0 = 0U;
    for(;;)
    {
        if(z0_0 < 21U)
        {
        }
        else
        {
            break;
        }
        l_1[z0_0] = 0.0f;
        z0_0 = z0_0 + 1U;
    }
    uint w_7 = 0U;
    for(;;)
    {
        if(w_7 < 6U)
        {
        }
        else
        {
            break;
        }
        float4 _S580 = float4(*(kernelContext_78->lp_0+(lp_l_0(k_19) + w_7))) ;
        uint _S581 = 4U * w_7;
        l_1[_S581] = _S580.x;
        uint _S582 = _S581 + 1U;
        if(_S582 < 21U)
        {
            l_1[_S582] = _S580.y;
        }
        uint _S583 = _S581 + 2U;
        if(_S583 < 21U)
        {
            l_1[_S583] = _S580.z;
        }
        uint _S584 = _S581 + 3U;
        if(_S584 < 21U)
        {
            l_1[_S584] = _S580.w;
        }
        w_7 = w_7 + 1U;
    }
    thread array<float, int(6)> y_2;
    y_2[int(0)] = 0.0f;
    y_2[int(1)] = 0.0f;
    y_2[int(2)] = 0.0f;
    y_2[int(3)] = 0.0f;
    y_2[int(4)] = 0.0f;
    y_2[int(5)] = 0.0f;
    uint i2_0 = 0U;
    for(;;)
    {
        if(i2_0 < 6U)
        {
        }
        else
        {
            break;
        }
        uint p_7 = 0U;
        s_7 = r_9[i2_0];
        for(;;)
        {
            if(p_7 < i2_0)
            {
            }
            else
            {
                break;
            }
            float s_8 = s_7 - l_1[tri_0(i2_0, p_7)] * y_2[p_7];
            p_7 = p_7 + 1U;
            s_7 = s_8;
        }
        y_2[i2_0] = s_7 / l_1[tri_0(i2_0, i2_0)];
        i2_0 = i2_0 + 1U;
    }
    thread array<float, int(6)> z_3;
    z_3[int(0)] = 0.0f;
    z_3[int(1)] = 0.0f;
    z_3[int(2)] = 0.0f;
    z_3[int(3)] = 0.0f;
    z_3[int(4)] = 0.0f;
    z_3[int(5)] = 0.0f;
    int i3_0 = int(5);
    for(;;)
    {
        if(i3_0 >= int(0))
        {
        }
        else
        {
            break;
        }
        int p2_1 = i3_0 + int(1);
        s_7 = y_2[i3_0];
        for(;;)
        {
            if(p2_1 < int(6))
            {
            }
            else
            {
                break;
            }
            float s_9 = s_7 - l_1[tri_0(uint(p2_1), uint(i3_0))] * z_3[p2_1];
            p2_1 = p2_1 + int(1);
            s_7 = s_9;
        }
        uint _S585 = uint(i3_0);
        z_3[i3_0] = s_7 / l_1[tri_0(_S585, _S585)];
        i3_0 = i3_0 - int(1);
    }
    uint _S586 = lp_d_0(k_19);
    float4 _S587 = float4(*(kernelContext_78->lp_0+_S586)) ;
    thread float4 d0_0 = _S587;
    uint _S588 = _S586 + 1U;
    float4 _S589 = float4(*(kernelContext_78->lp_0+_S588)) ;
    thread float4 d1_0 = _S589;
    d0_0.xyz = float3(kernelContext_78->params_0->lp_c1_0)  * _S587.xyz + float3(kernelContext_78->params_0->lp_c2_0)  * float3(z_3[int(0)], z_3[int(1)], z_3[int(2)]);
    d1_0.xyz = float3(kernelContext_78->params_0->lp_c1_0)  * _S589.xyz + float3(kernelContext_78->params_0->lp_c2_0)  * float3(z_3[int(3)], z_3[int(4)], z_3[int(5)]);
    *(kernelContext_78->lp_0+_S586) = packed_float4(d0_0) ;
    *(kernelContext_78->lp_0+_S588) = packed_float4(d1_0) ;
    uint _S590 = lp_x_0(k_19, to_1);
    *(kernelContext_78->lp_0+_S590) = packed_float4(float4(_S558.xyz + d0_0.xyz, 0.0f)) ;
    *(kernelContext_78->lp_0+(_S590 + 1U)) = packed_float4(float4(_S559.xyz + d1_0.xyz, 0.0f)) ;
    return;
}

[[kernel]] void fr_lp_sweep_even(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], atomic<uint> device* atom_34 [[buffer(5)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], packed_float4 device* lp_34 [[buffer(24)]], packed_float4 device* ext_35 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_79;
    (&kernelContext_79)->params_0 = params_34;
    (&kernelContext_79)->work_0 = work_34;
    (&kernelContext_79)->frames_0 = frames_34;
    (&kernelContext_79)->bodies_0 = bodies_34;
    (&kernelContext_79)->chunks_0 = chunks_34;
    (&kernelContext_79)->sort_0 = sort_34;
    (&kernelContext_79)->atom_0 = atom_34;
    (&kernelContext_79)->manifolds_0 = manifolds_34;
    (&kernelContext_79)->loads_0 = loads_34;
    (&kernelContext_79)->bonds_0 = bonds_34;
    (&kernelContext_79)->args_0 = args_34;
    (&kernelContext_79)->info_0 = info_34;
    (&kernelContext_79)->scan_0 = scan_34;
    (&kernelContext_79)->parts_0 = parts_34;
    (&kernelContext_79)->colliders_0 = colliders_34;
    (&kernelContext_79)->collider_info_0 = collider_info_34;
    (&kernelContext_79)->view_chunks_0 = view_chunks_34;
    (&kernelContext_79)->view_0 = view_34;
    (&kernelContext_79)->objects_0 = objects_34;
    (&kernelContext_79)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_79)->lines_0 = lines_34;
    (&kernelContext_79)->geometry_0 = geometry_34;
    (&kernelContext_79)->shots_0 = shots_34;
    (&kernelContext_79)->lp_0 = lp_34;
    (&kernelContext_79)->ext_0 = ext_35;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_79)->g_scan_0 = &g_scan_34;
    uint _S591 = id_31.x;
    if(_S591 < (params_34->chunk_count_0))
    {
        lp_sweep_0(_S591, 0U, &kernelContext_79);
    }
    return;
}

[[kernel]] void fr_lp_sweep_odd(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], atomic<uint> device* atom_35 [[buffer(5)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], packed_float4 device* lp_35 [[buffer(24)]], packed_float4 device* ext_36 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_80;
    (&kernelContext_80)->params_0 = params_35;
    (&kernelContext_80)->work_0 = work_35;
    (&kernelContext_80)->frames_0 = frames_35;
    (&kernelContext_80)->bodies_0 = bodies_35;
    (&kernelContext_80)->chunks_0 = chunks_35;
    (&kernelContext_80)->sort_0 = sort_35;
    (&kernelContext_80)->atom_0 = atom_35;
    (&kernelContext_80)->manifolds_0 = manifolds_35;
    (&kernelContext_80)->loads_0 = loads_35;
    (&kernelContext_80)->bonds_0 = bonds_35;
    (&kernelContext_80)->args_0 = args_35;
    (&kernelContext_80)->info_0 = info_35;
    (&kernelContext_80)->scan_0 = scan_35;
    (&kernelContext_80)->parts_0 = parts_35;
    (&kernelContext_80)->colliders_0 = colliders_35;
    (&kernelContext_80)->collider_info_0 = collider_info_35;
    (&kernelContext_80)->view_chunks_0 = view_chunks_35;
    (&kernelContext_80)->view_0 = view_35;
    (&kernelContext_80)->objects_0 = objects_35;
    (&kernelContext_80)->shot_bodies_0 = shot_bodies_35;
    (&kernelContext_80)->lines_0 = lines_35;
    (&kernelContext_80)->geometry_0 = geometry_35;
    (&kernelContext_80)->shots_0 = shots_35;
    (&kernelContext_80)->lp_0 = lp_35;
    (&kernelContext_80)->ext_0 = ext_36;
    threadgroup array<uint, int(1024)> g_scan_35;
    (&kernelContext_80)->g_scan_0 = &g_scan_35;
    uint _S592 = id_32.x;
    if(_S592 < (params_35->chunk_count_0))
    {
        lp_sweep_0(_S592, 1U, &kernelContext_80);
    }
    return;
}

struct Measures_0
{
    float tension_0;
    float shear_0;
    float normal_compression_0;
    float compression_0;
    float compressive_force_0;
};

Measures_0 stress_measures_0(const JointBond_0 thread* b_16, float3 q_lin_2, float3 q_ang_2)
{
    float area_0 = b_16->geom0_0.x;
    float _S593 = q_lin_2.z;
    float axial_0 = (metal::fast::divide((_S593), (area_0)));
    float _S594 = (metal::fast::divide((abs(q_ang_2.x)), (b_16->geom1_0.x)));
    float _S595 = (metal::fast::divide((abs(q_ang_2.y)), (b_16->geom1_0.y)));
    float bending_0 = _S594 + _S595;
    float _S596 = q_lin_2.x;
    float _S597 = q_lin_2.y;
    float _S598 = (metal::fast::sqrt((_S596 * _S596 + _S597 * _S597)));
    float _S599 = (metal::fast::divide((_S598), (area_0)));
    float _S600 = (metal::fast::divide((abs(q_ang_2.z)), (b_16->geom0_0.w)));
    float shear_1 = _S599 + _S600;
    thread Measures_0 m_3;
    (&m_3)->tension_0 = axial_0 + bending_0;
    (&m_3)->shear_0 = shear_1;
    float _S601 = - axial_0;
    (&m_3)->normal_compression_0 = max(_S601, 0.0f);
    (&m_3)->compression_0 = _S601 + bending_0;
    (&m_3)->compressive_force_0 = max(- _S593, 0.0f);
    return m_3;
}

struct JointMaterial_0
{
    float4 strength_0;
    float4 energy_0;
    float4 dif_0;
    float4 misc_0;
    uint4 kind_flags_0;
};

JointMaterial_0 lp_material_0(uint m_4, KernelContext_0 thread* kernelContext_81)
{
    uint o_5 = kernelContext_81->params_0->ext_mat_0 + 5U * m_4;
    thread JointMaterial_0 j_5;
    (&j_5)->strength_0 = float4(*(kernelContext_81->ext_0+o_5)) ;
    (&j_5)->energy_0 = float4(*(kernelContext_81->ext_0+(o_5 + 1U))) ;
    (&j_5)->dif_0 = float4(*(kernelContext_81->ext_0+(o_5 + 2U))) ;
    (&j_5)->misc_0 = float4(*(kernelContext_81->ext_0+(o_5 + 3U))) ;
    (&j_5)->kind_flags_0 = (as_type<uint4>((float4(*(kernelContext_81->ext_0+(o_5 + 4U))) )));
    return j_5;
}

float infinity_0()
{
    return float_bits_0(2139095040U);
}

float4 failure_indices_0(const JointMaterial_0 thread* mat_0, const JointBond_0 thread* b_17, const Measures_0 thread* m_5, float multiplier_0)
{
    float fc_0 = mat_0->strength_0.y * multiplier_0;
    float _S602 = min(mat_0->strength_0.z * multiplier_0 + mat_0->strength_0.w * m_5->normal_compression_0, mat_0->energy_0.x * multiplier_0);
    thread float4 idx_0;
    float _S603 = (metal::fast::divide((m_5->tension_0), (mat_0->strength_0.x * multiplier_0)));
    idx_0.x = max(_S603, 0.0f);
    float _S604;
    if(_S602 > 0.0f)
    {
        float _S605 = (metal::fast::divide((m_5->shear_0), (_S602)));
        _S604 = _S605;
    }
    else
    {
        _S604 = infinity_0();
    }
    idx_0.y = _S604;
    float _S606 = (metal::fast::divide((m_5->compression_0), (fc_0)));
    idx_0.z = max(_S606, 0.0f);
    float _S607 = b_17->stiff1_0.y;
    if(_S607 > 0.0f)
    {
        float _S608 = (metal::fast::divide((m_5->compressive_force_0), (_S607)));
        _S604 = _S608;
    }
    else
    {
        _S604 = 0.0f;
    }
    idx_0.w = _S604;
    return idx_0;
}

[[kernel]] void fr_lp_breaks(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], atomic<uint> device* atom_36 [[buffer(5)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], packed_float4 device* lp_36 [[buffer(24)]], packed_float4 device* ext_37 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_82;
    (&kernelContext_82)->params_0 = params_36;
    (&kernelContext_82)->work_0 = work_36;
    (&kernelContext_82)->frames_0 = frames_36;
    (&kernelContext_82)->bodies_0 = bodies_36;
    (&kernelContext_82)->chunks_0 = chunks_36;
    (&kernelContext_82)->sort_0 = sort_36;
    (&kernelContext_82)->atom_0 = atom_36;
    (&kernelContext_82)->manifolds_0 = manifolds_36;
    (&kernelContext_82)->loads_0 = loads_36;
    (&kernelContext_82)->bonds_0 = bonds_36;
    (&kernelContext_82)->args_0 = args_36;
    (&kernelContext_82)->info_0 = info_36;
    (&kernelContext_82)->scan_0 = scan_36;
    (&kernelContext_82)->parts_0 = parts_36;
    (&kernelContext_82)->colliders_0 = colliders_36;
    (&kernelContext_82)->collider_info_0 = collider_info_36;
    (&kernelContext_82)->view_chunks_0 = view_chunks_36;
    (&kernelContext_82)->view_0 = view_36;
    (&kernelContext_82)->objects_0 = objects_36;
    (&kernelContext_82)->shot_bodies_0 = shot_bodies_36;
    (&kernelContext_82)->lines_0 = lines_36;
    (&kernelContext_82)->geometry_0 = geometry_36;
    (&kernelContext_82)->shots_0 = shots_36;
    (&kernelContext_82)->lp_0 = lp_36;
    (&kernelContext_82)->ext_0 = ext_37;
    threadgroup array<uint, int(1024)> g_scan_36;
    (&kernelContext_82)->g_scan_0 = &g_scan_36;
    uint i_8 = id_33.x;
    bool _S609;
    if(i_8 >= (params_36->bond_count_0))
    {
        _S609 = true;
    }
    else
    {
        uint _S610 = w_alive_0(&kernelContext_82);
        _S609 = (*((&kernelContext_82)->work_0+(_S610 + i_8))) == 0U;
    }
    if(_S609)
    {
        return;
    }
    uint4 _S611 = uint4(*((&kernelContext_82)->bonds_0+i_8)) ;
    LpBond_0 _S612 = lp_bond_0(i_8, &kernelContext_82);
    uint _S613 = _S611.x;
    uint _S614 = _S611.y;
    thread LpBond_0 _S615 = _S612;
    thread float3 q_lin_3;
    thread float3 q_ang_3;
    lp_force_0(&_S615, _S613, _S614, 0U, &q_lin_3, &q_ang_3, &kernelContext_82);
    thread JointBond_0 _S616 = _S612.law_0;
    Measures_0 _S617 = stress_measures_0(&_S616, q_lin_3, q_ang_3);
    JointMaterial_0 _S618 = lp_material_0(_S612.law_0.ids_1.x, &kernelContext_82);
    float _S619 = _S612.law_0.geom1_0.w;
    thread JointMaterial_0 _S620 = _S618;
    thread JointBond_0 _S621 = _S612.law_0;
    thread Measures_0 _S622 = _S617;
    float4 _S623 = failure_indices_0(&_S620, &_S621, &_S622, _S619);
    float _S624 = max(max(_S623.x, _S623.y), max(_S623.z, _S623.w));
    uint _S625 = a_counters_0(&kernelContext_82);
    uint _S626 = atomic_fetch_max_explicit((&kernelContext_82)->atom_0+(_S625 + 11U), (as_type<uint>((max(_S624, 0.0f)))), memory_order_relaxed);
    if(_S624 >= 1.0f)
    {
        uint _S627 = w_alive_0(&kernelContext_82);
        *((&kernelContext_82)->work_0+(_S627 + i_8)) = 0U;
        uint _S628 = a_dirty_0(&kernelContext_82);
        uint _S629 = body_of_0(_S613, &kernelContext_82);
        uint _S630 = atomic_fetch_or_explicit((&kernelContext_82)->atom_0+(_S628 + _S629), 1U, memory_order_relaxed);
        uint _S631 = atomic_fetch_add_explicit((&kernelContext_82)->atom_0+_S625, 1U, memory_order_relaxed);
        uint _S632 = atomic_fetch_add_explicit((&kernelContext_82)->atom_0+(_S625 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_first_0(uint shot_5, KernelContext_0 thread* kernelContext_83)
{
    uint _S633 = a_strike_0(kernelContext_83);
    return _S633 + 27U * kernelContext_83->params_0->body_count_0 + shot_5;
}

[[kernel]] void fr_il_first_reset(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], atomic<uint> device* atom_37 [[buffer(5)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], packed_float4 device* lp_37 [[buffer(24)]], packed_float4 device* ext_38 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_84;
    (&kernelContext_84)->params_0 = params_37;
    (&kernelContext_84)->work_0 = work_37;
    (&kernelContext_84)->frames_0 = frames_37;
    (&kernelContext_84)->bodies_0 = bodies_37;
    (&kernelContext_84)->chunks_0 = chunks_37;
    (&kernelContext_84)->sort_0 = sort_37;
    (&kernelContext_84)->atom_0 = atom_37;
    (&kernelContext_84)->manifolds_0 = manifolds_37;
    (&kernelContext_84)->loads_0 = loads_37;
    (&kernelContext_84)->bonds_0 = bonds_37;
    (&kernelContext_84)->args_0 = args_37;
    (&kernelContext_84)->info_0 = info_37;
    (&kernelContext_84)->scan_0 = scan_37;
    (&kernelContext_84)->parts_0 = parts_37;
    (&kernelContext_84)->colliders_0 = colliders_37;
    (&kernelContext_84)->collider_info_0 = collider_info_37;
    (&kernelContext_84)->view_chunks_0 = view_chunks_37;
    (&kernelContext_84)->view_0 = view_37;
    (&kernelContext_84)->objects_0 = objects_37;
    (&kernelContext_84)->shot_bodies_0 = shot_bodies_37;
    (&kernelContext_84)->lines_0 = lines_37;
    (&kernelContext_84)->geometry_0 = geometry_37;
    (&kernelContext_84)->shots_0 = shots_37;
    (&kernelContext_84)->lp_0 = lp_37;
    (&kernelContext_84)->ext_0 = ext_38;
    threadgroup array<uint, int(1024)> g_scan_37;
    (&kernelContext_84)->g_scan_0 = &g_scan_37;
    uint _S634 = id_34.x;
    if(_S634 < (params_37->shot_count_0))
    {
        uint _S635 = a_first_0(_S634, &kernelContext_84);
        atomic_store_explicit((&kernelContext_84)->atom_0+_S635, 2139095040U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], atomic<uint> device* atom_38 [[buffer(5)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], packed_float4 device* lp_38 [[buffer(24)]], packed_float4 device* ext_39 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_85;
    (&kernelContext_85)->params_0 = params_38;
    (&kernelContext_85)->work_0 = work_38;
    (&kernelContext_85)->frames_0 = frames_38;
    (&kernelContext_85)->bodies_0 = bodies_38;
    (&kernelContext_85)->chunks_0 = chunks_38;
    (&kernelContext_85)->sort_0 = sort_38;
    (&kernelContext_85)->atom_0 = atom_38;
    (&kernelContext_85)->manifolds_0 = manifolds_38;
    (&kernelContext_85)->loads_0 = loads_38;
    (&kernelContext_85)->bonds_0 = bonds_38;
    (&kernelContext_85)->args_0 = args_38;
    (&kernelContext_85)->info_0 = info_38;
    (&kernelContext_85)->scan_0 = scan_38;
    (&kernelContext_85)->parts_0 = parts_38;
    (&kernelContext_85)->colliders_0 = colliders_38;
    (&kernelContext_85)->collider_info_0 = collider_info_38;
    (&kernelContext_85)->view_chunks_0 = view_chunks_38;
    (&kernelContext_85)->view_0 = view_38;
    (&kernelContext_85)->objects_0 = objects_38;
    (&kernelContext_85)->shot_bodies_0 = shot_bodies_38;
    (&kernelContext_85)->lines_0 = lines_38;
    (&kernelContext_85)->geometry_0 = geometry_38;
    (&kernelContext_85)->shots_0 = shots_38;
    (&kernelContext_85)->lp_0 = lp_38;
    (&kernelContext_85)->ext_0 = ext_39;
    threadgroup array<uint, int(1024)> g_scan_38;
    (&kernelContext_85)->g_scan_0 = &g_scan_38;
    uint b_18 = id_35.x;
    bool _S636;
    if(b_18 >= (params_38->body_count_0))
    {
        _S636 = true;
    }
    else
    {
        _S636 = ((&kernelContext_85)->params_0->shot_count_0) == 0U;
    }
    if(_S636)
    {
        return;
    }
    thread uint shot_6;
    bool _S637 = struck_0(b_18, &shot_6, &kernelContext_85);
    if(!_S637)
    {
        return;
    }
    uint _S638 = a_first_0(shot_6, &kernelContext_85);
    atomic<uint> device* _S639 = (&kernelContext_85)->atom_0+_S638;
    uint _S640 = a_strike_0(&kernelContext_85);
    uint _S641 = atomic_load_explicit((&kernelContext_85)->atom_0+(_S640 + 9U * b_18 + 6U), memory_order_relaxed);
    uint _S642 = atomic_fetch_min_explicit(_S639, _S641, memory_order_relaxed);
    return;
}

void il_path_0(uint b_19, float3 thread* contact_1, float3 thread* d_5, float thread* speed_1, KernelContext_0 thread* kernelContext_86)
{
    uint _S643 = strike_at_0(b_19, kernelContext_86);
    float3 rel_3 = (float4(*(kernelContext_86->frames_0+(_S643 + 1U))) ).xyz;
    float _S644 = length(rel_3);
    *speed_1 = _S644;
    *d_5 = rel_3 / float3(max(_S644, 9.999999960041972e-13f)) ;
    *contact_1 = (float4(*(kernelContext_86->frames_0+(_S643 + 2U))) ).xyz;
    return;
}

float2 il_interval_0(uint k_20, float3 p_8, float3 d_6, float pad_0, KernelContext_0 thread* kernelContext_87)
{
    thread Quat_0 q_24;
    float3 _S645 = chunk_box_world_0(k_20, &q_24, kernelContext_87);
    uint _S646 = 3U * k_20;
    Quat_0 _S647 = quat_of_0(float4(*(kernelContext_87->geometry_0+(_S646 + 2U))) );
    thread Quat_0 _S648 = q_24;
    thread Quat_0 _S649 = _S647;
    Quat_0 _S650 = quat_mul_0(&_S648, &_S649);
    thread Quat_0 _S651 = _S650;
    Quat_0 _S652 = normalized_0(&_S651);
    float3 _S653 = (float4(*(kernelContext_87->geometry_0+(_S646 + 1U))) ).xyz;
    float inf_0 = float_bits_0(2139095040U);
    float t0_2 = - inf_0;
    float t1_2 = inf_0;
    uint ax_3 = 0U;
    for(;;)
    {
        if(ax_3 < 3U)
        {
        }
        else
        {
            break;
        }
        thread float3 e_3 = float3(0.0f) ;
        e_3[ax_3] = 1.0f;
        thread Quat_0 _S654 = _S652;
        float3 _S655 = rotate_0(&_S654, e_3);
        float o_6 = dot(p_8 - _S645, _S655);
        float dv_1 = dot(d_6, _S655);
        float ext_40 = _S653[ax_3] + pad_0;
        if((abs(dv_1)) < 9.99999971718068537e-10f)
        {
            if((abs(o_6)) > ext_40)
            {
                return float2(1.0f, 0.0f);
            }
            ax_3 = ax_3 + 1U;
            continue;
        }
        float ta_2 = (- ext_40 - o_6) / dv_1;
        float tb_2 = (ext_40 - o_6) / dv_1;
        float _S656 = min(t1_2, max(ta_2, tb_2));
        t0_2 = max(t0_2, min(ta_2, tb_2));
        t1_2 = _S656;
        ax_3 = ax_3 + 1U;
    }
    return float2(t0_2, t1_2);
}

uint a_il_0(uint b_20, KernelContext_0 thread* kernelContext_88)
{
    uint _S657 = a_strike_0(kernelContext_88);
    return _S657 + 23U * kernelContext_88->params_0->body_count_0 + 4U * b_20;
}

[[kernel]] void fr_il_begin(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], atomic<uint> device* atom_39 [[buffer(5)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], packed_float4 device* lp_39 [[buffer(24)]], packed_float4 device* ext_41 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_89;
    (&kernelContext_89)->params_0 = params_39;
    (&kernelContext_89)->work_0 = work_39;
    (&kernelContext_89)->frames_0 = frames_39;
    (&kernelContext_89)->bodies_0 = bodies_39;
    (&kernelContext_89)->chunks_0 = chunks_39;
    (&kernelContext_89)->sort_0 = sort_39;
    (&kernelContext_89)->atom_0 = atom_39;
    (&kernelContext_89)->manifolds_0 = manifolds_39;
    (&kernelContext_89)->loads_0 = loads_39;
    (&kernelContext_89)->bonds_0 = bonds_39;
    (&kernelContext_89)->args_0 = args_39;
    (&kernelContext_89)->info_0 = info_39;
    (&kernelContext_89)->scan_0 = scan_39;
    (&kernelContext_89)->parts_0 = parts_39;
    (&kernelContext_89)->colliders_0 = colliders_39;
    (&kernelContext_89)->collider_info_0 = collider_info_39;
    (&kernelContext_89)->view_chunks_0 = view_chunks_39;
    (&kernelContext_89)->view_0 = view_39;
    (&kernelContext_89)->objects_0 = objects_39;
    (&kernelContext_89)->shot_bodies_0 = shot_bodies_39;
    (&kernelContext_89)->lines_0 = lines_39;
    (&kernelContext_89)->geometry_0 = geometry_39;
    (&kernelContext_89)->shots_0 = shots_39;
    (&kernelContext_89)->lp_0 = lp_39;
    (&kernelContext_89)->ext_0 = ext_41;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_89)->g_scan_0 = &g_scan_39;
    uint b_21 = id_36.x;
    bool _S658;
    if(b_21 >= (params_39->body_count_0))
    {
        _S658 = true;
    }
    else
    {
        _S658 = ((&kernelContext_89)->params_0->shot_count_0) == 0U;
    }
    if(_S658)
    {
        return;
    }
    thread uint shot_7;
    bool _S659 = struck_0(b_21, &shot_7, &kernelContext_89);
    if(!_S659)
    {
        return;
    }
    uint _S660 = a_strike_0(&kernelContext_89);
    uint _S661 = _S660 + 9U * b_21;
    uint _S662 = atomic_load_explicit((&kernelContext_89)->atom_0+(_S661 + 6U), memory_order_relaxed);
    uint _S663 = a_first_0(shot_7, &kernelContext_89);
    uint _S664 = atomic_load_explicit((&kernelContext_89)->atom_0+_S663, memory_order_relaxed);
    if(_S662 != _S664)
    {
        uint _S665 = strike_at_0(b_21, &kernelContext_89);
        (*((&kernelContext_89)->frames_0+_S665))[int(3)] = 0.0f;
        return;
    }
    thread float3 contact_2;
    thread float3 d_7;
    thread float speed_2;
    il_path_0(b_21, &contact_2, &d_7, &speed_2, &kernelContext_89);
    uint entry_1 = atomic_load_explicit((&kernelContext_89)->atom_0+(_S661 + 8U), memory_order_relaxed);
    float2 _S666 = il_interval_0(entry_1, contact_2, d_7, (float4(*((&kernelContext_89)->shots_0+2U * shot_7)) ).z, &kernelContext_89);
    uint end_0 = (as_type<uint>((max(_S666.y, 0.0f))));
    uint _S667 = a_il_0(b_21, &kernelContext_89);
    atomic_store_explicit((&kernelContext_89)->atom_0+_S667, end_0, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_89)->atom_0+(_S667 + 1U), end_0, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_89)->atom_0+(_S667 + 2U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_89)->atom_0+(_S667 + 3U), 0U, memory_order_relaxed);
    return;
}

void il_extend_0(uint k_21, uint from_3, KernelContext_0 thread* kernelContext_90)
{
    bool _S668;
    if(k_21 >= (kernelContext_90->params_0->chunk_count_0))
    {
        _S668 = true;
    }
    else
    {
        _S668 = (kernelContext_90->params_0->shot_count_0) == 0U;
    }
    if(_S668)
    {
        return;
    }
    uint _S669 = body_of_0(k_21, kernelContext_90);
    thread uint shot_8;
    bool _S670 = struck_0(_S669, &shot_8, kernelContext_90);
    if(!_S670)
    {
        return;
    }
    thread float3 contact_3;
    thread float3 d_8;
    thread float speed_3;
    il_path_0(_S669, &contact_3, &d_8, &speed_3, kernelContext_90);
    float pad_1 = (float4(*(kernelContext_90->shots_0+2U * shot_8)) ).z;
    uint _S671 = a_il_0(_S669, kernelContext_90);
    uint _S672 = atomic_load_explicit(kernelContext_90->atom_0+(_S671 + from_3), memory_order_relaxed);
    float end_1 = (as_type<float>((_S672)));
    uint to_2 = _S671 + 1U - from_3;
    uint _S673 = atomic_fetch_max_explicit(kernelContext_90->atom_0+to_2, (as_type<uint>((end_1))), memory_order_relaxed);
    float2 _S674 = il_interval_0(k_21, contact_3, d_8, pad_1, kernelContext_90);
    float _S675 = _S674.x;
    float _S676 = _S674.y;
    if(_S675 <= _S676)
    {
        _S668 = _S675 <= (end_1 + 0.00100000004749745f + 0.01999999955296516f * pad_1);
    }
    else
    {
        _S668 = false;
    }
    if(_S668)
    {
        _S668 = _S676 > end_1;
    }
    else
    {
        _S668 = false;
    }
    if(_S668)
    {
        uint _S677 = atomic_fetch_max_explicit(kernelContext_90->atom_0+to_2, (as_type<uint>((_S676))), memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_extend_even(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], atomic<uint> device* atom_40 [[buffer(5)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], packed_float4 device* lp_40 [[buffer(24)]], packed_float4 device* ext_42 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_91;
    (&kernelContext_91)->params_0 = params_40;
    (&kernelContext_91)->work_0 = work_40;
    (&kernelContext_91)->frames_0 = frames_40;
    (&kernelContext_91)->bodies_0 = bodies_40;
    (&kernelContext_91)->chunks_0 = chunks_40;
    (&kernelContext_91)->sort_0 = sort_40;
    (&kernelContext_91)->atom_0 = atom_40;
    (&kernelContext_91)->manifolds_0 = manifolds_40;
    (&kernelContext_91)->loads_0 = loads_40;
    (&kernelContext_91)->bonds_0 = bonds_40;
    (&kernelContext_91)->args_0 = args_40;
    (&kernelContext_91)->info_0 = info_40;
    (&kernelContext_91)->scan_0 = scan_40;
    (&kernelContext_91)->parts_0 = parts_40;
    (&kernelContext_91)->colliders_0 = colliders_40;
    (&kernelContext_91)->collider_info_0 = collider_info_40;
    (&kernelContext_91)->view_chunks_0 = view_chunks_40;
    (&kernelContext_91)->view_0 = view_40;
    (&kernelContext_91)->objects_0 = objects_40;
    (&kernelContext_91)->shot_bodies_0 = shot_bodies_40;
    (&kernelContext_91)->lines_0 = lines_40;
    (&kernelContext_91)->geometry_0 = geometry_40;
    (&kernelContext_91)->shots_0 = shots_40;
    (&kernelContext_91)->lp_0 = lp_40;
    (&kernelContext_91)->ext_0 = ext_42;
    threadgroup array<uint, int(1024)> g_scan_40;
    (&kernelContext_91)->g_scan_0 = &g_scan_40;
    il_extend_0(id_37.x, 0U, &kernelContext_91);
    return;
}

[[kernel]] void fr_il_extend_odd(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], atomic<uint> device* atom_41 [[buffer(5)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], packed_float4 device* lp_41 [[buffer(24)]], packed_float4 device* ext_43 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_92;
    (&kernelContext_92)->params_0 = params_41;
    (&kernelContext_92)->work_0 = work_41;
    (&kernelContext_92)->frames_0 = frames_41;
    (&kernelContext_92)->bodies_0 = bodies_41;
    (&kernelContext_92)->chunks_0 = chunks_41;
    (&kernelContext_92)->sort_0 = sort_41;
    (&kernelContext_92)->atom_0 = atom_41;
    (&kernelContext_92)->manifolds_0 = manifolds_41;
    (&kernelContext_92)->loads_0 = loads_41;
    (&kernelContext_92)->bonds_0 = bonds_41;
    (&kernelContext_92)->args_0 = args_41;
    (&kernelContext_92)->info_0 = info_41;
    (&kernelContext_92)->scan_0 = scan_41;
    (&kernelContext_92)->parts_0 = parts_41;
    (&kernelContext_92)->colliders_0 = colliders_41;
    (&kernelContext_92)->collider_info_0 = collider_info_41;
    (&kernelContext_92)->view_chunks_0 = view_chunks_41;
    (&kernelContext_92)->view_0 = view_41;
    (&kernelContext_92)->objects_0 = objects_41;
    (&kernelContext_92)->shot_bodies_0 = shot_bodies_41;
    (&kernelContext_92)->lines_0 = lines_41;
    (&kernelContext_92)->geometry_0 = geometry_41;
    (&kernelContext_92)->shots_0 = shots_41;
    (&kernelContext_92)->lp_0 = lp_41;
    (&kernelContext_92)->ext_0 = ext_43;
    threadgroup array<uint, int(1024)> g_scan_41;
    (&kernelContext_92)->g_scan_0 = &g_scan_41;
    il_extend_0(id_38.x, 1U, &kernelContext_92);
    return;
}

uint w_marks_0(KernelContext_0 thread* kernelContext_93)
{
    uint _S678 = w_pulses_0(kernelContext_93);
    return _S678 + 4U * kernelContext_93->params_0->chunk_count_0;
}

uint lp_body_0(uint b_22, KernelContext_0 thread* kernelContext_94)
{
    return 14U * kernelContext_94->params_0->chunk_count_0 + 2U * b_22;
}

float dif_factor_0(const JointMaterial_0 thread* mat_1, float strain_rate_0)
{
    float r_10 = abs(strain_rate_0);
    float4 _S679 = mat_1->dif_0;
    float ref_0 = mat_1->dif_0.x;
    if(r_10 <= ref_0)
    {
        return 1.0f;
    }
    float _S680 = _S679.z;
    float f_11;
    if(r_10 <= _S680)
    {
        float _S681 = (metal::fast::divide((r_10), (ref_0)));
        float _S682 = (metal::fast::pow((_S681), (_S679.y)));
        f_11 = _S682;
    }
    else
    {
        float _S683 = (metal::fast::divide((_S680), (ref_0)));
        float _S684 = (metal::fast::pow((_S683), (_S679.y)));
        float _S685 = (metal::fast::divide((r_10), (_S680)));
        float _S686 = (metal::fast::pow((_S685), (_S679.w)));
        f_11 = _S684 * _S686;
    }
    return clamp(f_11, 1.0f, mat_1->misc_0.x);
}

void il_impulse_0(uint b_23, uint sb_1, float m_shot_0, float3 contact_4, float3 d_9, float j_6, KernelContext_0 thread* kernelContext_95)
{
    float3 jr_0 = to_rubble_0(d_9) * float3(j_6) ;
    uint at_14 = 7U * b_23;
    float inv_mass_5 = (float4(*(kernelContext_95->bodies_0+at_14)) ).w;
    float3 _S687 = jr_0 * float3(inv_mass_5) ;
    float3 _S688 = float3(0.0f) ;
    float3 dw_0;
    if(inv_mass_5 > 0.0f)
    {
        Quat_0 q_25 = quat_of_0(float4(*(kernelContext_95->bodies_0+(at_14 + 1U))) );
        float3 r_11 = to_rubble_0(contact_4) - (float4(*(kernelContext_95->bodies_0+at_14)) ).xyz;
        float3 _S689 = (float4(*(kernelContext_95->bodies_0+(at_14 + 2U))) ).xyz;
        thread Quat_0 _S690 = q_25;
        Quat_0 _S691 = conj_0(&_S690);
        float3 _S692 = cross(r_11, jr_0);
        thread Quat_0 _S693 = _S691;
        float3 _S694 = rotate_0(&_S693, _S692);
        float3 _S695 = _S689 * _S694;
        thread Quat_0 _S696 = q_25;
        float3 _S697 = rotate_0(&_S696, _S695);
        dw_0 = _S697;
    }
    else
    {
        dw_0 = _S688;
    }
    uint region_1 = 0U;
    for(;;)
    {
        if(region_1 < 2U)
        {
        }
        else
        {
            break;
        }
        uint s_at_0 = 7U * (sb_1 + region_1 * kernelContext_95->params_0->max_bodies_0) + 3U;
        (*(kernelContext_95->bodies_0+s_at_0)).xyz = (float4(*(kernelContext_95->bodies_0+s_at_0)) ).xyz - jr_0 / float3(m_shot_0) ;
        uint w_at_0 = 7U * (b_23 + region_1 * kernelContext_95->params_0->max_bodies_0);
        uint _S698 = w_at_0 + 3U;
        (*(kernelContext_95->bodies_0+_S698)).xyz = (float4(*(kernelContext_95->bodies_0+_S698)) ).xyz + _S687;
        uint _S699 = w_at_0 + 4U;
        (*(kernelContext_95->bodies_0+_S699)).xyz = (float4(*(kernelContext_95->bodies_0+_S699)) ).xyz + dw_0;
        region_1 = region_1 + 1U;
    }
    return;
}

[[kernel]] void fr_il_strike(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], atomic<uint> device* atom_42 [[buffer(5)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], packed_float4 device* lp_42 [[buffer(24)]], packed_float4 device* ext_44 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_96;
    (&kernelContext_96)->params_0 = params_42;
    (&kernelContext_96)->work_0 = work_42;
    (&kernelContext_96)->frames_0 = frames_42;
    (&kernelContext_96)->bodies_0 = bodies_42;
    (&kernelContext_96)->chunks_0 = chunks_42;
    (&kernelContext_96)->sort_0 = sort_42;
    (&kernelContext_96)->atom_0 = atom_42;
    (&kernelContext_96)->manifolds_0 = manifolds_42;
    (&kernelContext_96)->loads_0 = loads_42;
    (&kernelContext_96)->bonds_0 = bonds_42;
    (&kernelContext_96)->args_0 = args_42;
    (&kernelContext_96)->info_0 = info_42;
    (&kernelContext_96)->scan_0 = scan_42;
    (&kernelContext_96)->parts_0 = parts_42;
    (&kernelContext_96)->colliders_0 = colliders_42;
    (&kernelContext_96)->collider_info_0 = collider_info_42;
    (&kernelContext_96)->view_chunks_0 = view_chunks_42;
    (&kernelContext_96)->view_0 = view_42;
    (&kernelContext_96)->objects_0 = objects_42;
    (&kernelContext_96)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_96)->lines_0 = lines_42;
    (&kernelContext_96)->geometry_0 = geometry_42;
    (&kernelContext_96)->shots_0 = shots_42;
    (&kernelContext_96)->lp_0 = lp_42;
    (&kernelContext_96)->ext_0 = ext_44;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_96)->g_scan_0 = &g_scan_42;
    uint b_24 = id_39.x;
    bool damage_0;
    if(b_24 >= (params_42->body_count_0))
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ((&kernelContext_96)->params_0->shot_count_0) == 0U;
    }
    if(damage_0)
    {
        return;
    }
    thread uint shot_9;
    bool _S700 = struck_0(b_24, &shot_9, &kernelContext_96);
    if(!_S700)
    {
        return;
    }
    uint _S701 = a_strike_0(&kernelContext_96);
    uint at_15 = _S701 + 9U * b_24;
    uint _S702 = atomic_load_explicit((&kernelContext_96)->atom_0+(at_15 + 6U), memory_order_relaxed);
    float t_in_3 = (as_type<float>((_S702)));
    uint entry_2 = atomic_load_explicit((&kernelContext_96)->atom_0+(at_15 + 8U), memory_order_relaxed);
    thread float3 contact_5;
    thread float3 d_10;
    thread float speed_4;
    il_path_0(b_24, &contact_5, &d_10, &speed_4, &kernelContext_96);
    float4 _S703 = float4(*((&kernelContext_96)->shots_0+2U * shot_9)) ;
    float _S704 = max(_S703.z, 0.00100000004749745f);
    float m_imp_1 = _S703.w;
    float e_imp_1 = (float4(*((&kernelContext_96)->shots_0+(2U * shot_9 + 1U))) ).x;
    float inv_mass_6 = (float4(*((&kernelContext_96)->bodies_0+7U * b_24)) ).w;
    float m_6;
    if(inv_mass_6 == 0.0f)
    {
        m_6 = m_imp_1;
    }
    else
    {
        m_6 = m_imp_1 / (1.0f + m_imp_1 * inv_mass_6);
    }
    uint launch_0 = (as_type<uint>(((float4(*((&kernelContext_96)->shots_0+(2U * shot_9 + 1U))) ).y)));
    if(launch_0 != 0U)
    {
        uint _S705 = w_marks_0(&kernelContext_96);
        damage_0 = (*((&kernelContext_96)->work_0+(_S705 + entry_2))) == launch_0;
    }
    else
    {
        damage_0 = false;
    }
    if(damage_0)
    {
        uint _S706 = lp_body_0(b_24, &kernelContext_96);
        packed_float4 _S707 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_96)->lp_0+_S706) = _S707;
        *((&kernelContext_96)->lp_0+(_S706 + 1U)) = _S707;
        return;
    }
    uint _S708 = a_il_0(b_24, &kernelContext_96);
    uint _S709 = atomic_load_explicit((&kernelContext_96)->atom_0+_S708, memory_order_relaxed);
    uint _S710 = atomic_load_explicit((&kernelContext_96)->atom_0+(_S708 + 1U), memory_order_relaxed);
    if(_S709 != _S710)
    {
        uint _S711 = a_counters_0(&kernelContext_96);
        uint _S712 = atomic_fetch_or_explicit((&kernelContext_96)->atom_0+(_S711 + 3U), 16U, memory_order_relaxed);
    }
    float _S713 = max((as_type<float>((_S709))) - _S704, 9.99999997475242708e-07f);
    if(!(t_in_3 > 0.0f))
    {
        uint _S714 = lp_body_0(b_24, &kernelContext_96);
        *((&kernelContext_96)->lp_0+_S714) = packed_float4(float4(_S713, 0.0f, 0.0f, 0.0f)) ;
        *((&kernelContext_96)->lp_0+(_S714 + 1U)) = packed_float4(float4(0.0f) ) ;
        return;
    }
    uint o_7 = (&kernelContext_96)->params_0->ext_chunks_0 + 4U * entry_2;
    float4 _S715 = float4(*((&kernelContext_96)->ext_0+(o_7 + 1U))) ;
    thread JointMaterial_0 dm_0;
    float4 _S716 = float4(0.0f) ;
    (&dm_0)->strength_0 = _S716;
    (&dm_0)->energy_0 = _S716;
    (&dm_0)->dif_0 = float4(*((&kernelContext_96)->ext_0+(o_7 + 2U))) ;
    float4 _S717 = float4(*((&kernelContext_96)->ext_0+(o_7 + 3U))) ;
    float _S718 = _S717.x;
    (&dm_0)->misc_0 = float4(_S718, 0.0f, 0.0f, 0.0f);
    (&dm_0)->kind_flags_0 = uint4(0U) ;
    float nu_0 = _S717.y;
    float e_target_0 = _S717.z / (1.0f - nu_0 * nu_0);
    float e_star_0 = 1.0f / (1.0f / e_imp_1 + 1.0f / e_target_0);
    float _S719 = max(speed_4, 0.00100000004749745f);
    float _S720 = sqrt(_S704);
    float h0_0 = pow(15.0f * m_6 * _S719 * _S719 / (16.0f * e_star_0 * _S720), 0.40000000596046448f);
    float t_c_0 = 2.94000005722045898f * h0_0 / _S719;
    float force_0 = 1.33333337306976318f * e_star_0 * _S720 * pow(h0_0, 1.5f);
    float a_6 = sqrt(_S704 * h0_0);
    float p0_0 = 3.0f * force_0 / (6.28318548202514648f * a_6 * a_6);
    float ring_0 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    float k_22;
    if(_S718 > 0.0f)
    {
        float _S721 = _S719 / _S713;
        thread JointMaterial_0 _S722 = dm_0;
        float _S723 = dif_factor_0(&_S722, _S721);
        k_22 = _S723;
    }
    else
    {
        k_22 = 1.0f;
    }
    float ft_0 = k_22 * _S715.y;
    float fc_1 = k_22 * _S715.z;
    float tau_0 = k_22 * _S715.w;
    if(p0_0 > fc_1)
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ring_0 > ft_0;
    }
    float _S724 = fc_1 * (3.14159274101257324f * _S704 * _S704);
    float _S725 = min(_S724 * _S713, 1.57079637050628662f * (2.0f * _S704 + _S713) * _S713 * _S713 * tau_0);
    float _S726 = 2.0f * _S725;
    float v_bl_0 = sqrt(_S726 / m_6);
    bool perforate_0;
    if(damage_0)
    {
        perforate_0 = _S719 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    float r2_1 = _S704 + _S713;
    float _S727 = _S715.x;
    float plug_0 = _S727 * 3.14159274101257324f * _S713 / 3.0f * (_S704 * _S704 + _S704 * r2_1 + r2_1 * r2_1);
    float v_res_0;
    if(perforate_0)
    {
        v_res_0 = m_imp_1 / (m_imp_1 + plug_0) * sqrt(max(_S719 * _S719 - v_bl_0 * v_bl_0, 0.0f));
    }
    else
    {
        v_res_0 = 0.0f;
    }
    bool spall_0;
    if(damage_0)
    {
        spall_0 = !perforate_0;
    }
    else
    {
        spall_0 = false;
    }
    float depth_0;
    if(spall_0)
    {
        depth_0 = min(0.5f * m_6 * _S719 * _S719 / _S724, _S713);
    }
    else
    {
        depth_0 = 0.0f;
    }
    bool crater_0 = depth_0 > a_6;
    float c_17 = sqrt(e_target_0 / _S727);
    float z_t_0 = _S727 * c_17;
    float rho_p_0 = m_imp_1 / (4.18879032135009766f * _S704 * _S704 * _S704);
    float z_p_0 = rho_p_0 * sqrt(e_imp_1 / rho_p_0);
    float mean_pressure_0 = force_0 / (3.14159274101257324f * a_6 * a_6);
    float _S728 = min(min(z_t_0 * _S719 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_1);
    float source_radius_0;
    if(mean_pressure_0 > fc_1)
    {
        source_radius_0 = max(a_6, _S704);
    }
    else
    {
        source_radius_0 = a_6;
    }
    float pulse_0 = c_17 * t_c_0;
    float back_0 = _S728 * min(source_radius_0 / _S713, 1.0f) * min(2.0f * _S713 / pulse_0, 1.0f);
    if(damage_0)
    {
        spall_0 = !perforate_0;
    }
    else
    {
        spall_0 = false;
    }
    if(spall_0)
    {
        spall_0 = back_0 > ft_0;
    }
    else
    {
        spall_0 = false;
    }
    float scab_0;
    if(spall_0)
    {
        scab_0 = 0.5f * min(pulse_0, _S713);
    }
    else
    {
        scab_0 = 0.0f;
    }
    uint region_2;
    if(damage_0)
    {
        region_2 = 1U;
    }
    else
    {
        region_2 = 0U;
    }
    uint _S729;
    if(crater_0)
    {
        _S729 = 2U;
    }
    else
    {
        _S729 = 0U;
    }
    uint _S730 = region_2 | _S729;
    if(spall_0)
    {
        region_2 = 4U;
    }
    else
    {
        region_2 = 0U;
    }
    uint _S731 = _S730 | region_2;
    if(perforate_0)
    {
        region_2 = 8U;
    }
    else
    {
        region_2 = 0U;
    }
    uint outcome_0 = _S731 | region_2;
    uint _S732 = lp_body_0(b_24, &kernelContext_96);
    *((&kernelContext_96)->lp_0+_S732) = packed_float4(float4(_S713, v_bl_0, v_res_0, (as_type<float>((outcome_0))))) ;
    packed_float4 device* _S733 = (&kernelContext_96)->lp_0+(_S732 + 1U);
    if(crater_0)
    {
        k_22 = depth_0;
    }
    else
    {
        k_22 = 0.0f;
    }
    *_S733 = packed_float4(float4(k_22, scab_0, a_6, _S725)) ;
    if(perforate_0)
    {
        uint _S734 = a_counters_0(&kernelContext_96);
        uint _S735 = atomic_fetch_add_explicit((&kernelContext_96)->atom_0+(_S734 + 10U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_0)
        {
            uint _S736 = a_counters_0(&kernelContext_96);
            uint _S737 = atomic_fetch_add_explicit((&kernelContext_96)->atom_0+(_S736 + 9U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_0)
            {
                uint _S738 = a_counters_0(&kernelContext_96);
                uint _S739 = atomic_fetch_add_explicit((&kernelContext_96)->atom_0+(_S738 + 8U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S740 = a_counters_0(&kernelContext_96);
                uint _S741 = atomic_fetch_add_explicit((&kernelContext_96)->atom_0+(_S740 + 7U), 1U, memory_order_relaxed);
            }
        }
    }
    uint sb_2 = (as_type<uint>((_S703.x)));
    if(perforate_0)
    {
        if(((&kernelContext_96)->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_24, sb_2, m_imp_1, contact_5, d_10, m_6 * (_S719 - sqrt(max(_S719 * _S719 - _S726 / m_6, 0.0f))), &kernelContext_96);
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_24, sb_2, m_imp_1, contact_5, d_10, m_6 * _S719, &kernelContext_96);
            float3 _S742 = to_rubble_0(d_10) * float3((_S719 * clamp(t_in_3, 0.0f, (&kernelContext_96)->params_0->dt_0))) ;
            region_2 = 0U;
            for(;;)
            {
                if(region_2 < 2U)
                {
                }
                else
                {
                    break;
                }
                uint _S743 = 7U * (sb_2 + region_2 * (&kernelContext_96)->params_0->max_bodies_0);
                (*((&kernelContext_96)->bodies_0+_S743)).xyz = (float4(*((&kernelContext_96)->bodies_0+_S743)) ).xyz + _S742;
                region_2 = region_2 + 1U;
            }
        }
    }
    return;
}

struct IlPlace_0
{
    float along_2;
    float radial_0;
    bool entry_3;
};

IlPlace_0 il_place_0(uint k_23, uint entry_4, float3 contact_6, float3 d_11, KernelContext_0 thread* kernelContext_97)
{
    uint _S744 = body_of_0(k_23, kernelContext_97);
    thread Quat_0 q_26;
    thread float3 origin_7;
    structure_pose_0(_S744, 7U * _S744, &q_26, &origin_7, kernelContext_97);
    float3 _S745 = (float4(*(kernelContext_97->chunks_0+4U * k_23)) ).xyz;
    thread Quat_0 _S746 = q_26;
    float3 _S747 = rotate_0(&_S746, _S745);
    float3 rel_4 = origin_7 + _S747 - contact_6;
    thread IlPlace_0 p_9;
    float _S748 = dot(rel_4, d_11);
    (&p_9)->along_2 = _S748;
    (&p_9)->radial_0 = length(rel_4 - d_11 * float3(_S748) );
    (&p_9)->entry_3 = k_23 == entry_4;
    return p_9;
}

bool il_plug_0(uint outcome_1, float r_12, float h_2, const IlPlace_0 thread* p_10)
{
    bool _S749;
    if((outcome_1 & 8U) != 0U)
    {
        if(p_10->entry_3)
        {
            _S749 = true;
        }
        else
        {
            float _S750 = p_10->along_2;
            if((p_10->along_2) >= 0.0f)
            {
                _S749 = _S750 <= h_2;
            }
            else
            {
                _S749 = false;
            }
            if(_S749)
            {
                _S749 = (p_10->radial_0) <= (r_12 + _S750);
            }
            else
            {
                _S749 = false;
            }
        }
    }
    else
    {
        _S749 = false;
    }
    return _S749;
}

bool il_scab_0(uint outcome_2, float r_13, float h_3, float scab_1, const IlPlace_0 thread* p_11)
{
    bool _S751;
    if((outcome_2 & 4U) != 0U)
    {
        _S751 = (p_11->along_2) >= (h_3 - scab_1);
    }
    else
    {
        _S751 = false;
    }
    if(_S751)
    {
        _S751 = (p_11->along_2) <= h_3;
    }
    else
    {
        _S751 = false;
    }
    if(_S751)
    {
        _S751 = (p_11->radial_0) <= (r_13 + h_3);
    }
    else
    {
        _S751 = false;
    }
    return _S751;
}

bool il_crater_0(uint outcome_3, float r_14, float depth_1, float a_7, const IlPlace_0 thread* p_12)
{
    bool _S752;
    if((outcome_3 & 2U) != 0U)
    {
        _S752 = (p_12->along_2) >= 0.0f;
    }
    else
    {
        _S752 = false;
    }
    if(_S752)
    {
        _S752 = (p_12->along_2) <= depth_1;
    }
    else
    {
        _S752 = false;
    }
    if(_S752)
    {
        _S752 = (p_12->radial_0) <= (max(r_14, a_7));
    }
    else
    {
        _S752 = false;
    }
    return _S752;
}

uint w_carry_0(KernelContext_0 thread* kernelContext_98)
{
    uint _S753 = w_marks_0(kernelContext_98);
    return _S753 + kernelContext_98->params_0->chunk_count_0;
}

[[kernel]] void fr_il_mark(uint3 id_40 [[thread_position_in_grid]], FractureParams_0 constant* params_43 [[buffer(0)]], uint device* work_43 [[buffer(4)]], packed_float4 device* frames_43 [[buffer(6)]], packed_float4 device* bodies_43 [[buffer(7)]], packed_float4 device* chunks_43 [[buffer(1)]], uint device* sort_43 [[buffer(11)]], atomic<uint> device* atom_43 [[buffer(5)]], Record_natural_0 device* manifolds_43 [[buffer(10)]], packed_float4 device* loads_43 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_43 [[buffer(13)]], uint device* info_43 [[buffer(15)]], ScanParams_0 constant* scan_43 [[buffer(14)]], packed_float4 device* parts_43 [[buffer(3)]], packed_float4 device* colliders_43 [[buffer(8)]], packed_uint4 device* collider_info_43 [[buffer(9)]], packed_float4 device* view_chunks_43 [[buffer(17)]], ViewParams_0 constant* view_43 [[buffer(16)]], packed_float4 device* objects_43 [[buffer(18)]], uint device* shot_bodies_43 [[buffer(20)]], float device* lines_43 [[buffer(19)]], packed_float4 device* geometry_43 [[buffer(21)]], packed_float4 device* shots_43 [[buffer(22)]], packed_float4 device* lp_43 [[buffer(24)]], packed_float4 device* ext_45 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_43;
    (&kernelContext_99)->work_0 = work_43;
    (&kernelContext_99)->frames_0 = frames_43;
    (&kernelContext_99)->bodies_0 = bodies_43;
    (&kernelContext_99)->chunks_0 = chunks_43;
    (&kernelContext_99)->sort_0 = sort_43;
    (&kernelContext_99)->atom_0 = atom_43;
    (&kernelContext_99)->manifolds_0 = manifolds_43;
    (&kernelContext_99)->loads_0 = loads_43;
    (&kernelContext_99)->bonds_0 = bonds_43;
    (&kernelContext_99)->args_0 = args_43;
    (&kernelContext_99)->info_0 = info_43;
    (&kernelContext_99)->scan_0 = scan_43;
    (&kernelContext_99)->parts_0 = parts_43;
    (&kernelContext_99)->colliders_0 = colliders_43;
    (&kernelContext_99)->collider_info_0 = collider_info_43;
    (&kernelContext_99)->view_chunks_0 = view_chunks_43;
    (&kernelContext_99)->view_0 = view_43;
    (&kernelContext_99)->objects_0 = objects_43;
    (&kernelContext_99)->shot_bodies_0 = shot_bodies_43;
    (&kernelContext_99)->lines_0 = lines_43;
    (&kernelContext_99)->geometry_0 = geometry_43;
    (&kernelContext_99)->shots_0 = shots_43;
    (&kernelContext_99)->lp_0 = lp_43;
    (&kernelContext_99)->ext_0 = ext_45;
    threadgroup array<uint, int(1024)> g_scan_43;
    (&kernelContext_99)->g_scan_0 = &g_scan_43;
    uint k_24 = id_40.x;
    bool path_0;
    if(k_24 >= (params_43->chunk_count_0))
    {
        path_0 = true;
    }
    else
    {
        path_0 = ((&kernelContext_99)->params_0->shot_count_0) == 0U;
    }
    if(path_0)
    {
        return;
    }
    uint _S754 = body_of_0(k_24, &kernelContext_99);
    thread uint shot_10;
    bool _S755 = struck_0(_S754, &shot_10, &kernelContext_99);
    if(!_S755)
    {
        return;
    }
    uint _S756 = lp_body_0(_S754, &kernelContext_99);
    float4 _S757 = float4(*((&kernelContext_99)->lp_0+_S756)) ;
    float4 _S758 = float4(*((&kernelContext_99)->lp_0+(_S756 + 1U))) ;
    uint outcome_4 = (as_type<uint>((_S757.w)));
    if((outcome_4 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_7;
    thread float3 d_12;
    thread float speed_5;
    il_path_0(_S754, &contact_7, &d_12, &speed_5, &kernelContext_99);
    uint _S759 = a_strike_0(&kernelContext_99);
    uint entry_5 = atomic_load_explicit((&kernelContext_99)->atom_0+(_S759 + 9U * _S754 + 8U), memory_order_relaxed);
    float _S760 = max((float4(*((&kernelContext_99)->shots_0+2U * shot_10)) ).z, 0.00100000004749745f);
    IlPlace_0 _S761 = il_place_0(k_24, entry_5, contact_7, d_12, &kernelContext_99);
    float _S762 = _S757.x;
    thread IlPlace_0 _S763 = _S761;
    bool _S764 = il_plug_0(outcome_4, _S760, _S762, &_S763);
    float bound_0 = (float4(*((&kernelContext_99)->shots_0+2U * shot_10)) ).y;
    float2 _S765 = il_interval_0(k_24, contact_7 - d_12 * float3(bound_0) , d_12, bound_0, &kernelContext_99);
    float _S766 = _S765.x;
    float _S767 = _S765.y;
    if(_S766 <= _S767)
    {
        path_0 = _S766 <= (_S762 + 2.0f * bound_0);
    }
    else
    {
        path_0 = false;
    }
    if(path_0)
    {
        path_0 = _S767 >= 0.0f;
    }
    else
    {
        path_0 = false;
    }
    if(_S764)
    {
        path_0 = true;
    }
    if(path_0)
    {
        path_0 = true;
    }
    else
    {
        float _S768 = _S758.y;
        thread IlPlace_0 _S769 = _S761;
        bool _S770 = il_scab_0(outcome_4, _S760, _S762, _S768, &_S769);
        path_0 = _S770;
    }
    if(path_0)
    {
        path_0 = true;
    }
    else
    {
        float _S771 = _S758.x;
        float _S772 = _S758.z;
        thread IlPlace_0 _S773 = _S761;
        bool _S774 = il_crater_0(outcome_4, _S760, _S771, _S772, &_S773);
        path_0 = _S774;
    }
    if(path_0)
    {
        uint _S775 = w_marks_0(&kernelContext_99);
        *((&kernelContext_99)->work_0+(_S775 + k_24)) = (as_type<uint>(((float4(*((&kernelContext_99)->shots_0+(2U * shot_10 + 1U))) ).y)));
    }
    if(_S764)
    {
        uint _S776 = w_carry_0(&kernelContext_99);
        *((&kernelContext_99)->work_0+(_S776 + k_24)) = _S754 + 1U;
        uint _S777 = a_il_0(_S754, &kernelContext_99);
        add64_0(_S777 + 2U, (float4(*((&kernelContext_99)->chunks_0+4U * k_24)) ).w, (&kernelContext_99)->params_0->scale_m_0, &kernelContext_99);
    }
    return;
}

[[kernel]] void fr_il_carry(uint3 id_41 [[thread_position_in_grid]], FractureParams_0 constant* params_44 [[buffer(0)]], uint device* work_44 [[buffer(4)]], packed_float4 device* frames_44 [[buffer(6)]], packed_float4 device* bodies_44 [[buffer(7)]], packed_float4 device* chunks_44 [[buffer(1)]], uint device* sort_44 [[buffer(11)]], atomic<uint> device* atom_44 [[buffer(5)]], Record_natural_0 device* manifolds_44 [[buffer(10)]], packed_float4 device* loads_44 [[buffer(12)]], packed_uint4 device* bonds_44 [[buffer(2)]], uint device* args_44 [[buffer(13)]], uint device* info_44 [[buffer(15)]], ScanParams_0 constant* scan_44 [[buffer(14)]], packed_float4 device* parts_44 [[buffer(3)]], packed_float4 device* colliders_44 [[buffer(8)]], packed_uint4 device* collider_info_44 [[buffer(9)]], packed_float4 device* view_chunks_44 [[buffer(17)]], ViewParams_0 constant* view_44 [[buffer(16)]], packed_float4 device* objects_44 [[buffer(18)]], uint device* shot_bodies_44 [[buffer(20)]], float device* lines_44 [[buffer(19)]], packed_float4 device* geometry_44 [[buffer(21)]], packed_float4 device* shots_44 [[buffer(22)]], packed_float4 device* lp_44 [[buffer(24)]], packed_float4 device* ext_46 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_100;
    (&kernelContext_100)->params_0 = params_44;
    (&kernelContext_100)->work_0 = work_44;
    (&kernelContext_100)->frames_0 = frames_44;
    (&kernelContext_100)->bodies_0 = bodies_44;
    (&kernelContext_100)->chunks_0 = chunks_44;
    (&kernelContext_100)->sort_0 = sort_44;
    (&kernelContext_100)->atom_0 = atom_44;
    (&kernelContext_100)->manifolds_0 = manifolds_44;
    (&kernelContext_100)->loads_0 = loads_44;
    (&kernelContext_100)->bonds_0 = bonds_44;
    (&kernelContext_100)->args_0 = args_44;
    (&kernelContext_100)->info_0 = info_44;
    (&kernelContext_100)->scan_0 = scan_44;
    (&kernelContext_100)->parts_0 = parts_44;
    (&kernelContext_100)->colliders_0 = colliders_44;
    (&kernelContext_100)->collider_info_0 = collider_info_44;
    (&kernelContext_100)->view_chunks_0 = view_chunks_44;
    (&kernelContext_100)->view_0 = view_44;
    (&kernelContext_100)->objects_0 = objects_44;
    (&kernelContext_100)->shot_bodies_0 = shot_bodies_44;
    (&kernelContext_100)->lines_0 = lines_44;
    (&kernelContext_100)->geometry_0 = geometry_44;
    (&kernelContext_100)->shots_0 = shots_44;
    (&kernelContext_100)->lp_0 = lp_44;
    (&kernelContext_100)->ext_0 = ext_46;
    threadgroup array<uint, int(1024)> g_scan_44;
    (&kernelContext_100)->g_scan_0 = &g_scan_44;
    uint k_25 = id_41.x;
    bool _S778;
    if(k_25 >= (params_44->chunk_count_0))
    {
        _S778 = true;
    }
    else
    {
        _S778 = ((&kernelContext_100)->params_0->shot_count_0) == 0U;
    }
    if(_S778)
    {
        return;
    }
    uint _S779 = w_carry_0(&kernelContext_100);
    uint _S780 = _S779 + k_25;
    uint device* _S781 = (&kernelContext_100)->work_0+_S780;
    uint c_18 = *_S781;
    if((*_S781) == 0U)
    {
        return;
    }
    *((&kernelContext_100)->work_0+_S780) = 0U;
    uint b0_0 = c_18 - 1U;
    thread uint shot_11;
    bool _S782 = struck_0(b0_0, &shot_11, &kernelContext_100);
    if(!_S782)
    {
        return;
    }
    thread float3 contact_8;
    thread float3 d_13;
    thread float speed_6;
    il_path_0(b0_0, &contact_8, &d_13, &speed_6, &kernelContext_100);
    float3 dr_0 = to_rubble_0(d_13);
    uint sb_3 = (as_type<uint>(((float4(*((&kernelContext_100)->shots_0+2U * shot_11)) ).x)));
    float m_s_0 = (float4(*((&kernelContext_100)->shots_0+2U * shot_11)) ).w;
    uint _S783 = a_il_0(b0_0, &kernelContext_100);
    float _S784 = read64_0(_S783 + 2U, (&kernelContext_100)->params_0->scale_m_0, &kernelContext_100);
    float3 v_wall_0 = (float4(*((&kernelContext_100)->bodies_0+(7U * b0_0 + 3U))) ).xyz;
    uint _S785 = 7U * sb_3;
    float3 v_shot_0 = (float4(*((&kernelContext_100)->bodies_0+(_S785 + 3U))) ).xyz;
    float closing_0 = dot(v_shot_0 - v_wall_0, dr_0);
    float v_r_0 = m_s_0 * closing_0 / (m_s_0 + _S784);
    float3 _S786 = v_shot_0 + dr_0 * float3((v_r_0 - closing_0)) ;
    float3 _S787 = v_wall_0 + dr_0 * float3(v_r_0) ;
    uint _S788 = a_strike_0(&kernelContext_100);
    uint _S789 = atomic_load_explicit((&kernelContext_100)->atom_0+(_S788 + 9U * b0_0 + 6U), memory_order_relaxed);
    float3 _S790 = (float4(*((&kernelContext_100)->bodies_0+_S785)) ).xyz + dr_0 * float3(((closing_0 - v_r_0) * clamp((as_type<float>((_S789))), 0.0f, (&kernelContext_100)->params_0->dt_0))) ;
    uint _S791 = body_of_0(k_25, &kernelContext_100);
    uint region_3 = 0U;
    for(;;)
    {
        if(region_3 < 2U)
        {
        }
        else
        {
            break;
        }
        uint s_at_1 = 7U * (sb_3 + region_3 * (&kernelContext_100)->params_0->max_bodies_0);
        (*((&kernelContext_100)->bodies_0+s_at_1)).xyz = _S790;
        (*((&kernelContext_100)->bodies_0+(s_at_1 + 3U))).xyz = _S786;
        (*((&kernelContext_100)->bodies_0+(7U * (_S791 + region_3 * (&kernelContext_100)->params_0->max_bodies_0) + 3U))).xyz = _S787;
        region_3 = region_3 + 1U;
    }
    return;
}

[[kernel]] void fr_il_breaks(uint3 id_42 [[thread_position_in_grid]], FractureParams_0 constant* params_45 [[buffer(0)]], uint device* work_45 [[buffer(4)]], packed_float4 device* frames_45 [[buffer(6)]], packed_float4 device* bodies_45 [[buffer(7)]], packed_float4 device* chunks_45 [[buffer(1)]], uint device* sort_45 [[buffer(11)]], atomic<uint> device* atom_45 [[buffer(5)]], Record_natural_0 device* manifolds_45 [[buffer(10)]], packed_float4 device* loads_45 [[buffer(12)]], packed_uint4 device* bonds_45 [[buffer(2)]], uint device* args_45 [[buffer(13)]], uint device* info_45 [[buffer(15)]], ScanParams_0 constant* scan_45 [[buffer(14)]], packed_float4 device* parts_45 [[buffer(3)]], packed_float4 device* colliders_45 [[buffer(8)]], packed_uint4 device* collider_info_45 [[buffer(9)]], packed_float4 device* view_chunks_45 [[buffer(17)]], ViewParams_0 constant* view_45 [[buffer(16)]], packed_float4 device* objects_45 [[buffer(18)]], uint device* shot_bodies_45 [[buffer(20)]], float device* lines_45 [[buffer(19)]], packed_float4 device* geometry_45 [[buffer(21)]], packed_float4 device* shots_45 [[buffer(22)]], packed_float4 device* lp_45 [[buffer(24)]], packed_float4 device* ext_47 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_101;
    (&kernelContext_101)->params_0 = params_45;
    (&kernelContext_101)->work_0 = work_45;
    (&kernelContext_101)->frames_0 = frames_45;
    (&kernelContext_101)->bodies_0 = bodies_45;
    (&kernelContext_101)->chunks_0 = chunks_45;
    (&kernelContext_101)->sort_0 = sort_45;
    (&kernelContext_101)->atom_0 = atom_45;
    (&kernelContext_101)->manifolds_0 = manifolds_45;
    (&kernelContext_101)->loads_0 = loads_45;
    (&kernelContext_101)->bonds_0 = bonds_45;
    (&kernelContext_101)->args_0 = args_45;
    (&kernelContext_101)->info_0 = info_45;
    (&kernelContext_101)->scan_0 = scan_45;
    (&kernelContext_101)->parts_0 = parts_45;
    (&kernelContext_101)->colliders_0 = colliders_45;
    (&kernelContext_101)->collider_info_0 = collider_info_45;
    (&kernelContext_101)->view_chunks_0 = view_chunks_45;
    (&kernelContext_101)->view_0 = view_45;
    (&kernelContext_101)->objects_0 = objects_45;
    (&kernelContext_101)->shot_bodies_0 = shot_bodies_45;
    (&kernelContext_101)->lines_0 = lines_45;
    (&kernelContext_101)->geometry_0 = geometry_45;
    (&kernelContext_101)->shots_0 = shots_45;
    (&kernelContext_101)->lp_0 = lp_45;
    (&kernelContext_101)->ext_0 = ext_47;
    threadgroup array<uint, int(1024)> g_scan_45;
    (&kernelContext_101)->g_scan_0 = &g_scan_45;
    uint i_9 = id_42.x;
    bool cut_0;
    if(i_9 >= (params_45->bond_count_0))
    {
        cut_0 = true;
    }
    else
    {
        cut_0 = ((&kernelContext_101)->params_0->shot_count_0) == 0U;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        uint _S792 = w_alive_0(&kernelContext_101);
        cut_0 = (*((&kernelContext_101)->work_0+(_S792 + i_9))) == 0U;
    }
    if(cut_0)
    {
        return;
    }
    uint4 _S793 = uint4(*((&kernelContext_101)->bonds_0+i_9)) ;
    uint _S794 = _S793.x;
    uint _S795 = body_of_0(_S794, &kernelContext_101);
    thread uint shot_12;
    bool _S796 = struck_0(_S795, &shot_12, &kernelContext_101);
    if(!_S796)
    {
        return;
    }
    uint _S797 = lp_body_0(_S795, &kernelContext_101);
    float4 _S798 = float4(*((&kernelContext_101)->lp_0+_S797)) ;
    float4 _S799 = float4(*((&kernelContext_101)->lp_0+(_S797 + 1U))) ;
    uint outcome_5 = (as_type<uint>((_S798.w)));
    if((outcome_5 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_9;
    thread float3 d_14;
    thread float speed_7;
    il_path_0(_S795, &contact_9, &d_14, &speed_7, &kernelContext_101);
    uint _S800 = a_strike_0(&kernelContext_101);
    uint entry_6 = atomic_load_explicit((&kernelContext_101)->atom_0+(_S800 + 9U * _S795 + 8U), memory_order_relaxed);
    float _S801 = max((float4(*((&kernelContext_101)->shots_0+2U * shot_12)) ).z, 0.00100000004749745f);
    float h_4 = _S798.x;
    IlPlace_0 _S802 = il_place_0(_S794, entry_6, contact_9, d_14, &kernelContext_101);
    IlPlace_0 _S803 = il_place_0(_S793.y, entry_6, contact_9, d_14, &kernelContext_101);
    thread IlPlace_0 _S804 = _S802;
    bool _S805 = il_plug_0(outcome_5, _S801, h_4, &_S804);
    thread IlPlace_0 _S806 = _S803;
    bool _S807 = il_plug_0(outcome_5, _S801, h_4, &_S806);
    if(_S805 != _S807)
    {
        cut_0 = true;
    }
    else
    {
        float _S808 = _S799.y;
        thread IlPlace_0 _S809 = _S802;
        bool _S810 = il_scab_0(outcome_5, _S801, h_4, _S808, &_S809);
        thread IlPlace_0 _S811 = _S803;
        bool _S812 = il_scab_0(outcome_5, _S801, h_4, _S808, &_S811);
        cut_0 = _S810 != _S812;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S813 = _S799.x;
        float _S814 = _S799.z;
        thread IlPlace_0 _S815 = _S802;
        bool _S816 = il_crater_0(outcome_5, _S801, _S813, _S814, &_S815);
        cut_0 = _S816;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S817 = _S799.x;
        float _S818 = _S799.z;
        thread IlPlace_0 _S819 = _S803;
        bool _S820 = il_crater_0(outcome_5, _S801, _S817, _S818, &_S819);
        cut_0 = _S820;
    }
    if(cut_0)
    {
        uint _S821 = w_alive_0(&kernelContext_101);
        *((&kernelContext_101)->work_0+(_S821 + i_9)) = 0U;
        uint _S822 = a_dirty_0(&kernelContext_101);
        uint _S823 = atomic_fetch_or_explicit((&kernelContext_101)->atom_0+(_S822 + _S795), 1U, memory_order_relaxed);
        uint _S824 = a_counters_0(&kernelContext_101);
        uint _S825 = atomic_fetch_add_explicit((&kernelContext_101)->atom_0+_S824, 1U, memory_order_relaxed);
        uint _S826 = atomic_fetch_add_explicit((&kernelContext_101)->atom_0+(_S824 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

