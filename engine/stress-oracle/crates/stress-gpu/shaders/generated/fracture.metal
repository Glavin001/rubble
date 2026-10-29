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

struct CrushParams_0
{
    uint enabled_0;
    uint mode_1;
    uint fragments_0;
    uint pool_size_0;
    uint max_events_0;
    uint t_bonds_0;
    uint t_adj_0;
    uint t_list_0;
    uint t_pool_0;
    uint r_bill_0;
    uint r_shot_0;
    uint r_events_0;
    float park_y_0;
    float park_step_0;
    uint flow_0;
    uint t_normal_0;
    uint r_flow_0;
    uint r_blast_0;
    float pad0_1;
    float pad1_1;
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
    CrushParams_0 constant* crush_0;
    packed_float4 device* crush_rw_0;
    packed_float4 device* crush_tab_0;
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

[[kernel]] void fr_loads(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], uint device* sort_1 [[buffer(11)]], atomic<uint> device* atom_1 [[buffer(5)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], packed_float4 device* loads_1 [[buffer(12)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]], packed_float4 device* view_chunks_1 [[buffer(17)]], ViewParams_0 constant* view_1 [[buffer(16)]], packed_float4 device* objects_1 [[buffer(18)]], uint device* shot_bodies_1 [[buffer(20)]], float device* lines_1 [[buffer(19)]], packed_float4 device* geometry_1 [[buffer(21)]], packed_float4 device* shots_1 [[buffer(22)]], CrushParams_0 constant* crush_1 [[buffer(25)]], packed_float4 device* crush_rw_1 [[buffer(24)]], packed_float4 device* crush_tab_1 [[buffer(23)]])
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
    (&kernelContext_18)->crush_0 = crush_1;
    (&kernelContext_18)->crush_rw_0 = crush_rw_1;
    (&kernelContext_18)->crush_tab_0 = crush_tab_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_18)->g_scan_0 = &g_scan_1;
    uint k_0 = id_0.x;
    if(k_0 >= (params_1->chunk_count_0))
    {
        return;
    }
    uint _S35 = body_of_0(k_0, &kernelContext_18);
    uint b_1 = 7U * _S35;
    thread Quat_0 q_6;
    thread float3 origin_1;
    structure_pose_0(_S35, b_1, &q_6, &origin_1, &kernelContext_18);
    uint _S36 = 4U * k_0;
    float4 _S37 = float4(*((&kernelContext_18)->chunks_0+_S36)) ;
    float3 _S38 = _S37.xyz;
    thread Quat_0 _S39 = q_6;
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

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], atomic<uint> device* atom_2 [[buffer(5)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]], packed_float4 device* geometry_2 [[buffer(21)]], packed_float4 device* shots_2 [[buffer(22)]], CrushParams_0 constant* crush_2 [[buffer(25)]], packed_float4 device* crush_rw_2 [[buffer(24)]], packed_float4 device* crush_tab_2 [[buffer(23)]])
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
    (&kernelContext_19)->crush_0 = crush_2;
    (&kernelContext_19)->crush_rw_0 = crush_rw_2;
    (&kernelContext_19)->crush_tab_0 = crush_tab_2;
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

[[kernel]] void fr_gate(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], CrushParams_0 constant* crush_3 [[buffer(25)]], packed_float4 device* crush_rw_3 [[buffer(24)]], packed_float4 device* crush_tab_3 [[buffer(23)]])
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
    (&kernelContext_21)->crush_0 = crush_3;
    (&kernelContext_21)->crush_rw_0 = crush_rw_3;
    (&kernelContext_21)->crush_tab_0 = crush_tab_3;
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

[[kernel]] void fr_uf_init(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], CrushParams_0 constant* crush_4 [[buffer(25)]], packed_float4 device* crush_rw_4 [[buffer(24)]], packed_float4 device* crush_tab_4 [[buffer(23)]])
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
    (&kernelContext_25)->crush_0 = crush_4;
    (&kernelContext_25)->crush_rw_0 = crush_rw_4;
    (&kernelContext_25)->crush_tab_0 = crush_tab_4;
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

[[kernel]] void fr_body_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], CrushParams_0 constant* crush_5 [[buffer(25)]], packed_float4 device* crush_rw_5 [[buffer(24)]], packed_float4 device* crush_tab_5 [[buffer(23)]])
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
    (&kernelContext_26)->crush_0 = crush_5;
    (&kernelContext_26)->crush_rw_0 = crush_rw_5;
    (&kernelContext_26)->crush_tab_0 = crush_tab_5;
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

[[kernel]] void fr_uf_union(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], CrushParams_0 constant* crush_6 [[buffer(25)]], packed_float4 device* crush_rw_6 [[buffer(24)]], packed_float4 device* crush_tab_6 [[buffer(23)]])
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
    (&kernelContext_28)->crush_0 = crush_6;
    (&kernelContext_28)->crush_rw_0 = crush_rw_6;
    (&kernelContext_28)->crush_tab_0 = crush_tab_6;
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
    int v_3 = int(x_4);
    uint lo_0 = (as_type<uint>((v_3)));
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
    if(v_3 < int(0))
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

[[kernel]] void fr_uf_flatten(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], CrushParams_0 constant* crush_7 [[buffer(25)]], packed_float4 device* crush_rw_7 [[buffer(24)]], packed_float4 device* crush_tab_7 [[buffer(23)]])
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
    (&kernelContext_30)->crush_0 = crush_7;
    (&kernelContext_30)->crush_rw_0 = crush_rw_7;
    (&kernelContext_30)->crush_tab_0 = crush_tab_7;
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

[[kernel]] void fr_best_count(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], CrushParams_0 constant* crush_8 [[buffer(25)]], packed_float4 device* crush_rw_8 [[buffer(24)]], packed_float4 device* crush_tab_8 [[buffer(23)]])
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
    (&kernelContext_32)->crush_0 = crush_8;
    (&kernelContext_32)->crush_rw_0 = crush_rw_8;
    (&kernelContext_32)->crush_tab_0 = crush_tab_8;
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

[[kernel]] void fr_best_root(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], CrushParams_0 constant* crush_9 [[buffer(25)]], packed_float4 device* crush_rw_9 [[buffer(24)]], packed_float4 device* crush_tab_9 [[buffer(23)]])
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
    (&kernelContext_33)->crush_0 = crush_9;
    (&kernelContext_33)->crush_rw_0 = crush_rw_9;
    (&kernelContext_33)->crush_tab_0 = crush_tab_9;
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

[[kernel]] void fr_mark(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], CrushParams_0 constant* crush_10 [[buffer(25)]], packed_float4 device* crush_rw_10 [[buffer(24)]], packed_float4 device* crush_tab_10 [[buffer(23)]])
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
    (&kernelContext_34)->crush_0 = crush_10;
    (&kernelContext_34)->crush_rw_0 = crush_rw_10;
    (&kernelContext_34)->crush_tab_0 = crush_tab_10;
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

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], CrushParams_0 constant* crush_11 [[buffer(25)]], packed_float4 device* crush_rw_11 [[buffer(24)]], packed_float4 device* crush_tab_11 [[buffer(23)]])
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
    (&kernelContext_35)->crush_0 = crush_11;
    (&kernelContext_35)->crush_rw_0 = crush_rw_11;
    (&kernelContext_35)->crush_tab_0 = crush_tab_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_35)->g_scan_0 = &g_scan_11;
    uint t_1 = thread_0.x;
    uint _S189 = group_0.x;
    uint _S190 = 1024U * _S189 + 4U * t_1;
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
        if((_S190 + j_2) < ((&kernelContext_35)->scan_0->n_0))
        {
            step_2 = *((&kernelContext_35)->work_0+((&kernelContext_35)->scan_0->in_base_0 + _S190 + j_2));
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
    (*(&kernelContext_35)->g_scan_0)[t_1] = sum_2;
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
            add_0 = (*(&kernelContext_35)->g_scan_0)[t_1 - step_2];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_35)->g_scan_0)[t_1] = (*(&kernelContext_35)->g_scan_0)[t_1] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_2 = step_2 << 1U;
    }
    uint _S191 = (*(&kernelContext_35)->g_scan_0)[t_1] - sum_2;
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
        uint run_1 = add_0 + v_4[j_2];
        j_2 = j_2 + 1U;
        add_0 = run_1;
    }
    if(t_1 == 255U)
    {
        *((&kernelContext_35)->work_0+((&kernelContext_35)->scan_0->block_base_0 + _S189)) = (*(&kernelContext_35)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], CrushParams_0 constant* crush_12 [[buffer(25)]], packed_float4 device* crush_rw_12 [[buffer(24)]], packed_float4 device* crush_tab_12 [[buffer(23)]])
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
    (&kernelContext_36)->crush_0 = crush_12;
    (&kernelContext_36)->crush_rw_0 = crush_rw_12;
    (&kernelContext_36)->crush_tab_0 = crush_tab_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_36)->g_scan_0 = &g_scan_12;
    uint t_2 = thread_1.x;
    uint blocks_0 = (scan_12->n_0 + 1023U) / 1024U;
    bool _S192 = t_2 < blocks_0;
    uint v_5;
    if(_S192)
    {
        v_5 = *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->block_base_0 + t_2));
    }
    else
    {
        v_5 = 0U;
    }
    (*(&kernelContext_36)->g_scan_0)[t_2] = v_5;
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
            add_1 = (*(&kernelContext_36)->g_scan_0)[t_2 - step_3];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_36)->g_scan_0)[t_2] = (*(&kernelContext_36)->g_scan_0)[t_2] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_3 = step_3 << 1U;
    }
    if(_S192)
    {
        *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->block_base_0 + t_2)) = (*(&kernelContext_36)->g_scan_0)[t_2] - v_5;
    }
    if(t_2 == 1023U)
    {
        *((&kernelContext_36)->work_0+((&kernelContext_36)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_36)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], CrushParams_0 constant* crush_13 [[buffer(25)]], packed_float4 device* crush_rw_13 [[buffer(24)]], packed_float4 device* crush_tab_13 [[buffer(23)]])
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
    (&kernelContext_37)->crush_0 = crush_13;
    (&kernelContext_37)->crush_rw_0 = crush_rw_13;
    (&kernelContext_37)->crush_tab_0 = crush_tab_13;
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

[[kernel]] void fr_freelist(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], CrushParams_0 constant* crush_14 [[buffer(25)]], packed_float4 device* crush_rw_14 [[buffer(24)]], packed_float4 device* crush_tab_14 [[buffer(23)]])
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
    (&kernelContext_38)->crush_0 = crush_14;
    (&kernelContext_38)->crush_rw_0 = crush_rw_14;
    (&kernelContext_38)->crush_tab_0 = crush_tab_14;
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

uint scan_new_blocks_1(KernelContext_0 thread* kernelContext_40)
{
    uint _S199 = w_free_list_0(kernelContext_40);
    return _S199 + kernelContext_40->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_41)
{
    uint _S200 = scan_new_blocks_1(kernelContext_41);
    return _S200 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_42)
{
    uint blocks_1 = (kernelContext_42->params_0->body_count_0 + 1023U) / 1024U;
    uint _S201 = scan_free_blocks_0(kernelContext_42);
    return *(kernelContext_42->work_0+(_S201 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], CrushParams_0 constant* crush_15 [[buffer(25)]], packed_float4 device* crush_rw_15 [[buffer(24)]], packed_float4 device* crush_tab_15 [[buffer(23)]])
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
    (&kernelContext_43)->crush_0 = crush_15;
    (&kernelContext_43)->crush_rw_0 = crush_rw_15;
    (&kernelContext_43)->crush_tab_0 = crush_tab_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_43)->g_scan_0 = &g_scan_15;
    uint c_8 = id_12.x;
    bool _S202 = is_root_0(c_8, &kernelContext_43);
    if(!_S202)
    {
        return;
    }
    uint _S203 = body_of_0(c_8, &kernelContext_43);
    uint _S204 = a_best_root_0(&kernelContext_43);
    uint _S205 = atomic_load_explicit((&kernelContext_43)->atom_0+(_S204 + _S203), memory_order_relaxed);
    uint slot_1;
    if(_S205 != c_8)
    {
        uint _S206 = w_rank_0(&kernelContext_43);
        uint device* _S207 = (&kernelContext_43)->work_0+(_S206 + c_8);
        uint k_1 = *_S207;
        uint _S208 = free_total_0(&kernelContext_43);
        if((*_S207) < _S208)
        {
            uint _S209 = w_free_list_0(&kernelContext_43);
            uint device* _S210 = (&kernelContext_43)->work_0+(_S209 + k_1);
            uint slot_2 = *_S210;
            uint _S211 = w_free_0(&kernelContext_43);
            *((&kernelContext_43)->work_0+(_S211 + *_S210)) = 0U;
            uint _S212 = a_counters_0(&kernelContext_43);
            uint _S213 = atomic_fetch_add_explicit((&kernelContext_43)->atom_0+(_S212 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S214 = a_counters_0(&kernelContext_43);
            uint _S215 = atomic_fetch_or_explicit((&kernelContext_43)->atom_0+(_S214 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S203;
        }
    }
    else
    {
        slot_1 = _S203;
    }
    uint _S216 = w_slot_0(&kernelContext_43);
    *((&kernelContext_43)->work_0+(_S216 + c_8)) = slot_1;
    return;
}

float read64_0(uint at_3, float scale_1, KernelContext_0 thread* kernelContext_44)
{
    uint _S217 = atomic_load_explicit(kernelContext_44->atom_0+(at_3 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S217)));
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
    array<float, int(3)> _S218 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S219 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S220 = { { r2_0.x, r2_0.y, r2_0.z } };
    thread array<array<float, int(3)>, int(3)> a_3;
    a_3[int(0)] = _S218;
    a_3[int(1)] = _S219;
    a_3[int(2)] = _S220;
    array<float, int(3)> _S221 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S222 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S223 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_6;
    v_6[int(0)] = _S221;
    v_6[int(1)] = _S222;
    v_6[int(2)] = _S223;
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
            bool _S224;
            if(k_2 != p_2)
            {
                _S224 = k_2 != q_9;
            }
            else
            {
                _S224 = false;
            }
            if(_S224)
            {
                float kp_0 = a_3[k_2][p_2];
                float kq_0 = a_3[k_2][q_9];
                float _S225 = c_9 * a_3[k_2][p_2] - s_2 * a_3[k_2][q_9];
                a_3[k_2][p_2] = _S225;
                a_3[p_2][k_2] = _S225;
                float _S226 = s_2 * kp_0 + c_9 * kq_0;
                a_3[k_2][q_9] = _S226;
                a_3[q_9][k_2] = _S226;
            }
            k_2 = k_2 + 1U;
        }
        float _S227 = c_9 * c_9;
        float _S228 = 2.0f * s_2 * c_9 * apq_0;
        float _S229 = s_2 * s_2;
        a_3[p_2][p_2] = _S227 * app_0 - _S228 + _S229 * aqq_0;
        a_3[q_9][q_9] = _S229 * app_0 + _S228 + _S227 * aqq_0;
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
    thread Quat_0 _S230 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S231 = normalized_0(&_S230);
    *rot_0 = _S231;
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
    thread Quat_0 _S232 = axes_1;
    Quat_0 _S233 = quat_mul_0(&_S232, q_10);
    thread Quat_0 _S234 = _S233;
    Quat_0 _S235 = normalized_0(&_S234);
    return _S235;
}

uint stage_0(uint c_10, KernelContext_0 thread* kernelContext_45)
{
    return 2U * kernelContext_45->params_0->body_count_0 + 9U * c_10;
}

float4 quat_vec_0(const Quat_0 thread* q_11)
{
    return float4(q_11->x_0, q_11->y_0, q_11->z_0, q_11->w_0);
}

[[kernel]] void fr_piece(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], CrushParams_0 constant* crush_16 [[buffer(25)]], packed_float4 device* crush_rw_16 [[buffer(24)]], packed_float4 device* crush_tab_16 [[buffer(23)]])
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
    (&kernelContext_46)->crush_0 = crush_16;
    (&kernelContext_46)->crush_rw_0 = crush_rw_16;
    (&kernelContext_46)->crush_tab_0 = crush_tab_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_46)->g_scan_0 = &g_scan_16;
    uint c_11 = id_13.x;
    bool _S236 = is_root_0(c_11, &kernelContext_46);
    if(!_S236)
    {
        return;
    }
    uint _S237 = body_of_0(c_11, &kernelContext_46);
    uint pb_0 = 7U * _S237;
    uint _S238 = a_sums_0(&kernelContext_46);
    uint at_4 = _S238 + 20U * c_11;
    float _S239 = read64_0(at_4, (&kernelContext_46)->params_0->scale_m_0, &kernelContext_46);
    float _S240 = read64_0(at_4 + 2U, (&kernelContext_46)->params_0->scale_s_0, &kernelContext_46);
    float _S241 = read64_0(at_4 + 4U, (&kernelContext_46)->params_0->scale_s_0, &kernelContext_46);
    float _S242 = read64_0(at_4 + 6U, (&kernelContext_46)->params_0->scale_s_0, &kernelContext_46);
    float3 d_1 = float3(_S240, _S241, _S242) / float3(max(_S239, 1.00000000317107685e-30f)) ;
    float _S243 = read64_0(at_4 + 8U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float _S244 = read64_0(at_4 + 10U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float _S245 = read64_0(at_4 + 12U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float _S246 = read64_0(at_4 + 14U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float _S247 = read64_0(at_4 + 16U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float _S248 = read64_0(at_4 + 18U, (&kernelContext_46)->params_0->scale_j_0, &kernelContext_46);
    float dd_1 = dot(d_1, d_1);
    float _S249 = d_1.x;
    float _S250 = _S239 * _S249;
    float _S251 = d_1.y;
    float _S252 = _S246 + _S250 * _S251;
    float _S253 = d_1.z;
    float _S254 = _S247 + _S250 * _S253;
    float _S255 = _S248 + _S239 * _S251 * _S253;
    float3 com_1 = (float4(*((&kernelContext_46)->chunks_0+4U * c_11)) ).xyz + d_1;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S243 - _S239 * (dd_1 - _S249 * _S249), _S252, _S254), float3(_S252, _S244 - _S239 * (dd_1 - _S251 * _S251), _S255), float3(_S254, _S255, _S245 - _S239 * (dd_1 - _S253 * _S253)), &principal_0, &moments_1);
    uint _S256 = a_anchor_0(&kernelContext_46);
    uint _S257 = atomic_load_explicit((&kernelContext_46)->atom_0+(_S256 + c_11), memory_order_relaxed);
    bool anchored_0 = _S257 != 0U;
    thread Quat_0 q_12;
    thread float3 origin_2;
    structure_pose_0(_S237, pb_0, &q_12, &origin_2, &kernelContext_46);
    float3 p_parent_0 = (float4(*((&kernelContext_46)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_46)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_46)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S258 = q_12;
    float3 _S259 = rotate_0(&_S258, com_1);
    float3 pos_0 = to_rubble_0(origin_2 + _S259);
    thread Quat_0 _S260 = q_12;
    thread Quat_0 _S261 = principal_0;
    Quat_0 _S262 = quat_mul_0(&_S260, &_S261);
    thread Quat_0 _S263 = _S262;
    Quat_0 _S264 = normalized_0(&_S263);
    thread Quat_0 _S265 = _S264;
    Quat_0 _S266 = rot_to_rubble_0(&_S265);
    bool _S267;
    if(anchored_0)
    {
        _S267 = true;
    }
    else
    {
        _S267 = !(_S239 > 0.0f);
    }
    float inv_mass_1;
    if(_S267)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / _S239;
    }
    float3 _S268 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S269;
        if((moments_1.x) > 0.0f)
        {
            _S269 = 1.0f / moments_1.x;
        }
        else
        {
            _S269 = 0.0f;
        }
        float _S270;
        if((moments_1.y) > 0.0f)
        {
            _S270 = 1.0f / moments_1.y;
        }
        else
        {
            _S270 = 0.0f;
        }
        float _S271;
        if((moments_1.z) > 0.0f)
        {
            _S271 = 1.0f / moments_1.z;
        }
        else
        {
            _S271 = 0.0f;
        }
        inv_inertia_0 = float3(_S269, _S270, _S271);
    }
    else
    {
        inv_inertia_0 = _S268;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S268;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S268;
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
    uint _S272 = stage_0(c_11, &kernelContext_46);
    *((&kernelContext_46)->frames_0+_S272) = packed_float4(float4(pos_0, inv_mass_1)) ;
    packed_float4 device* _S273 = (&kernelContext_46)->frames_0+(_S272 + 1U);
    thread Quat_0 _S274 = _S266;
    float4 _S275 = quat_vec_0(&_S274);
    *_S273 = packed_float4(_S275) ;
    *((&kernelContext_46)->frames_0+(_S272 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_46)->frames_0+(_S272 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_46)->frames_0+(_S272 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_46)->params_0->step_0))))) ;
    *((&kernelContext_46)->frames_0+(_S272 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_46)->frames_0+(_S272 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_46)->frames_0+(_S272 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_46)->frames_0+2U * _S237)) ).w)) ;
    packed_float4 device* _S276 = (&kernelContext_46)->frames_0+(_S272 + 8U);
    thread Quat_0 _S277 = principal_0;
    float4 _S278 = quat_vec_0(&_S277);
    *_S276 = packed_float4(_S278) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], CrushParams_0 constant* crush_17 [[buffer(25)]], packed_float4 device* crush_rw_17 [[buffer(24)]], packed_float4 device* crush_tab_17 [[buffer(23)]])
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
    (&kernelContext_47)->crush_0 = crush_17;
    (&kernelContext_47)->crush_rw_0 = crush_rw_17;
    (&kernelContext_47)->crush_tab_0 = crush_tab_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_47)->g_scan_0 = &g_scan_17;
    uint c_12 = id_14.x;
    bool _S279 = is_root_0(c_12, &kernelContext_47);
    if(!_S279)
    {
        return;
    }
    uint _S280 = w_slot_0(&kernelContext_47);
    uint slot_3 = *((&kernelContext_47)->work_0+(_S280 + c_12));
    uint _S281 = stage_0(c_12, &kernelContext_47);
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
            *((&kernelContext_47)->bodies_0+(7U * (slot_3 + region_0 * (&kernelContext_47)->params_0->max_bodies_0) + w_3)) = packed_float4(float4(*((&kernelContext_47)->frames_0+(_S281 + w_3))) ) ;
            w_3 = w_3 + 1U;
        }
        region_0 = region_0 + 1U;
    }
    uint _S282 = 2U * slot_3;
    *((&kernelContext_47)->frames_0+_S282) = packed_float4(float4(*((&kernelContext_47)->frames_0+(_S281 + 7U))) ) ;
    *((&kernelContext_47)->frames_0+(_S282 + 1U)) = packed_float4(float4(*((&kernelContext_47)->frames_0+(_S281 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], CrushParams_0 constant* crush_18 [[buffer(25)]], packed_float4 device* crush_rw_18 [[buffer(24)]], packed_float4 device* crush_tab_18 [[buffer(23)]])
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
    (&kernelContext_48)->crush_0 = crush_18;
    (&kernelContext_48)->crush_rw_0 = crush_rw_18;
    (&kernelContext_48)->crush_tab_0 = crush_tab_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_48)->g_scan_0 = &g_scan_18;
    uint c_13 = id_15.x;
    bool _S283;
    if(c_13 >= (params_18->chunk_count_0))
    {
        _S283 = true;
    }
    else
    {
        uint _S284 = body_of_0(c_13, &kernelContext_48);
        bool _S285 = dirty_0(_S284, &kernelContext_48);
        _S283 = !_S285;
    }
    if(_S283)
    {
        return;
    }
    uint _S286 = w_label_0(&kernelContext_48);
    uint device* _S287 = (&kernelContext_48)->work_0+(_S286 + c_13);
    uint r_6 = *_S287;
    uint _S288 = w_slot_0(&kernelContext_48);
    uint device* _S289 = (&kernelContext_48)->work_0+(_S288 + *_S287);
    uint slot_4 = *_S289;
    *((&kernelContext_48)->work_0+(w_body_0() + c_13)) = *_S289;
    uint _S290 = 2U * slot_4;
    float3 _S291 = (float4(*((&kernelContext_48)->frames_0+_S290)) ).xyz;
    thread Quat_0 _S292 = quat_of_0(float4(*((&kernelContext_48)->frames_0+(_S290 + 1U))) );
    Quat_0 _S293 = conj_0(&_S292);
    float inv_mass_2 = (float4(*((&kernelContext_48)->bodies_0+7U * slot_4)) ).w;
    float _S294;
    if(inv_mass_2 > 0.0f)
    {
        _S294 = 1.0f / inv_mass_2;
    }
    else
    {
        _S294 = float_bits_0(2139095040U);
    }
    float4 _S295 = float4(*((&kernelContext_48)->chunks_0+(4U * c_13 + 2U))) ;
    uint first_1 = (as_type<uint>((_S295.z)));
    uint _S296 = (as_type<uint>((_S295.w)));
    float4 _S297 = float4(*((&kernelContext_48)->chunks_0+(4U * r_6 + 3U))) ;
    uint p_3 = first_1;
    for(;;)
    {
        if(p_3 < (first_1 + _S296))
        {
        }
        else
        {
            break;
        }
        uint _S298 = 2U * p_3;
        float4 _S299 = float4(*((&kernelContext_48)->parts_0+_S298)) ;
        uint col_0 = (as_type<uint>((_S299.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_48)->parts_0+(_S298 + 1U))) );
        uint _S300 = 4U * col_0;
        float4 _S301 = float4(*((&kernelContext_48)->colliders_0+_S300)) ;
        *((&kernelContext_48)->colliders_0+_S300) = packed_float4(float4((as_type<float>((slot_4))), _S301.y, _S301.z, _S301.w)) ;
        uint _S302 = _S300 + 1U;
        packed_float4 device* _S303 = (&kernelContext_48)->colliders_0+_S302;
        float3 _S304 = _S299.xyz - _S291;
        thread Quat_0 _S305 = _S293;
        float3 _S306 = rotate_0(&_S305, _S304);
        *_S303 = packed_float4(float4(_S306, (float4(*((&kernelContext_48)->colliders_0+_S302)) ).w)) ;
        packed_float4 device* _S307 = (&kernelContext_48)->colliders_0+(_S300 + 2U);
        thread Quat_0 _S308 = _S293;
        thread Quat_0 _S309 = rot_1;
        Quat_0 _S310 = quat_mul_0(&_S308, &_S309);
        thread Quat_0 _S311 = _S310;
        Quat_0 _S312 = normalized_0(&_S311);
        thread Quat_0 _S313 = _S312;
        float4 _S314 = quat_vec_0(&_S313);
        *_S307 = packed_float4(_S314) ;
        uint _S315 = 2U * col_0;
        uint4 _S316 = uint4(*((&kernelContext_48)->collider_info_0+_S315)) ;
        *((&kernelContext_48)->collider_info_0+_S315) = packed_uint4(uint4(_S316.x, slot_4, _S316.z, _S316.w)) ;
        *((&kernelContext_48)->collider_info_0+(_S315 + 1U)) = packed_uint4(uint4((as_type<uint>((_S294))), (as_type<uint>((_S297.x))), (as_type<uint>((_S297.w))), 0U)) ;
        p_3 = p_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], CrushParams_0 constant* crush_19 [[buffer(25)]], packed_float4 device* crush_rw_19 [[buffer(24)]], packed_float4 device* crush_tab_19 [[buffer(23)]])
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
    (&kernelContext_49)->crush_0 = crush_19;
    (&kernelContext_49)->crush_rw_0 = crush_rw_19;
    (&kernelContext_49)->crush_tab_0 = crush_tab_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_49)->g_scan_0 = &g_scan_19;
    uint s_3 = id_16.x;
    if(s_3 >= (params_19->body_count_0))
    {
        return;
    }
    uint _S317 = a_dirty_0(&kernelContext_49);
    atomic_store_explicit((&kernelContext_49)->atom_0+(_S317 + s_3), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_0 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_7 = fract(h_0);
    float q_13 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_7);
    float t_3 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_7));
    uint k_3 = uint(h_0);
    float3 _S318;
    if(k_3 == 0U)
    {
        _S318 = float3(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_3 == 1U)
        {
            _S318 = float3(q_13, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_3 == 2U)
            {
                _S318 = float3(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_3 == 3U)
                {
                    _S318 = float3(0.34959998726844788f, q_13, 0.92000001668930054f);
                }
                else
                {
                    if(k_3 == 4U)
                    {
                        _S318 = float3(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S318 = float3(0.92000001668930054f, 0.34959998726844788f, q_13);
                    }
                }
            }
        }
    }
    return _S318;
}

void write_object_0(uint o_1, const Quat_0 thread* q_14, float3 t_4, float4 color_0, KernelContext_0 thread* kernelContext_50)
{
    float3 _S319 = rotate_0(q_14, float3(1.0f, 0.0f, 0.0f));
    float3 _S320 = rotate_0(q_14, float3(0.0f, 1.0f, 0.0f));
    float3 _S321 = rotate_0(q_14, float3(0.0f, 0.0f, 1.0f));
    uint _S322 = 4U * o_1;
    *(kernelContext_50->objects_0+_S322) = packed_float4(float4(_S319.x, _S320.x, _S321.x, t_4.x)) ;
    *(kernelContext_50->objects_0+(_S322 + 1U)) = packed_float4(float4(_S319.y, _S320.y, _S321.y, t_4.y)) ;
    *(kernelContext_50->objects_0+(_S322 + 2U)) = packed_float4(float4(_S319.z, _S320.z, _S321.z, t_4.z)) ;
    *(kernelContext_50->objects_0+(_S322 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], CrushParams_0 constant* crush_20 [[buffer(25)]], packed_float4 device* crush_rw_20 [[buffer(24)]], packed_float4 device* crush_tab_20 [[buffer(23)]])
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
    (&kernelContext_51)->crush_0 = crush_20;
    (&kernelContext_51)->crush_rw_0 = crush_rw_20;
    (&kernelContext_51)->crush_tab_0 = crush_tab_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_51)->g_scan_0 = &g_scan_20;
    uint k_4 = id_17.x;
    if(k_4 >= (params_20->chunk_count_0))
    {
        return;
    }
    uint _S323 = body_of_0(k_4, &kernelContext_51);
    uint b_3 = 7U * _S323;
    thread Quat_0 q_15;
    thread float3 origin_3;
    structure_pose_0(_S323, b_3, &q_15, &origin_3, &kernelContext_51);
    uint _S324 = 3U * k_4;
    float4 _S325 = float4(*((&kernelContext_51)->view_chunks_0+_S324)) ;
    float3 base_2 = (float4(*((&kernelContext_51)->view_chunks_0+(_S324 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_51)->bodies_0+b_3)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_51)->view_0->mode_0) == 1U)
    {
        float _S326 = max(length((float4(*((&kernelContext_51)->bodies_0+(b_3 + 3U))) ).xyz), length((float4(*((&kernelContext_51)->bodies_0+(b_3 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S326 > ((&kernelContext_51)->view_0->rest_speed_0))
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
            color_1 = slot_color_0(_S323);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S323);
            }
        }
    }
    uint _S327 = (as_type<uint>((_S325.w)));
    Quat_0 _S328 = quat_of_0(float4(*((&kernelContext_51)->view_chunks_0+(_S324 + 1U))) );
    thread Quat_0 _S329 = q_15;
    thread Quat_0 _S330 = _S328;
    Quat_0 _S331 = quat_mul_0(&_S329, &_S330);
    thread Quat_0 _S332 = _S331;
    Quat_0 _S333 = normalized_0(&_S332);
    float3 _S334 = _S325.xyz;
    thread Quat_0 _S335 = q_15;
    float3 _S336 = rotate_0(&_S335, _S334);
    float3 _S337 = origin_3 + _S336;
    float4 _S338 = float4(color_1, 1.0f);
    thread Quat_0 _S339 = _S333;
    write_object_0(_S327, &_S339, _S337, _S338, &kernelContext_51);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], CrushParams_0 constant* crush_21 [[buffer(25)]], packed_float4 device* crush_rw_21 [[buffer(24)]], packed_float4 device* crush_tab_21 [[buffer(23)]])
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
    (&kernelContext_52)->crush_0 = crush_21;
    (&kernelContext_52)->crush_rw_0 = crush_rw_21;
    (&kernelContext_52)->crush_tab_0 = crush_tab_21;
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
    thread Quat_0 _S340 = quat_of_0(float4(*((&kernelContext_52)->bodies_0+(b_4 + 1U))) );
    Quat_0 _S341 = rot_from_rubble_0(&_S340);
    float3 _S342 = from_rubble_0((float4(*((&kernelContext_52)->bodies_0+b_4)) ).xyz);
    float4 _S343 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S344 = _S341;
    write_object_0(o_2, &_S344, _S342, _S343, &kernelContext_52);
    return;
}

float3 chunk_world_0(uint k_5, KernelContext_0 thread* kernelContext_53)
{
    uint _S345 = body_of_0(k_5, kernelContext_53);
    thread Quat_0 q_16;
    thread float3 origin_4;
    structure_pose_0(_S345, 7U * _S345, &q_16, &origin_4, kernelContext_53);
    float3 _S346 = (float4(*(kernelContext_53->view_chunks_0+3U * k_5)) ).xyz;
    thread Quat_0 _S347 = q_16;
    float3 _S348 = rotate_0(&_S347, _S346);
    return origin_4 + _S348;
}

void write_line_vertex_0(uint v_8, float3 p_4, KernelContext_0 thread* kernelContext_54)
{
    uint _S349 = 7U * v_8;
    *(kernelContext_54->lines_0+_S349) = p_4.x;
    *(kernelContext_54->lines_0+(_S349 + 1U)) = p_4.y;
    *(kernelContext_54->lines_0+(_S349 + 2U)) = p_4.z;
    *(kernelContext_54->lines_0+(_S349 + 3U)) = 1.0f;
    *(kernelContext_54->lines_0+(_S349 + 4U)) = 0.86000001430511475f;
    *(kernelContext_54->lines_0+(_S349 + 5U)) = 0.25f;
    *(kernelContext_54->lines_0+(_S349 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], CrushParams_0 constant* crush_22 [[buffer(25)]], packed_float4 device* crush_rw_22 [[buffer(24)]], packed_float4 device* crush_tab_22 [[buffer(23)]])
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
    (&kernelContext_55)->crush_0 = crush_22;
    (&kernelContext_55)->crush_rw_0 = crush_rw_22;
    (&kernelContext_55)->crush_tab_0 = crush_tab_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_55)->g_scan_0 = &g_scan_22;
    uint i_4 = id_19.x;
    if(i_4 >= (params_22->bond_count_0))
    {
        return;
    }
    uint4 _S350 = uint4(*((&kernelContext_55)->bonds_0+i_4)) ;
    float3 _S351 = chunk_world_0(_S350.x, &kernelContext_55);
    uint _S352 = w_alive_0(&kernelContext_55);
    float3 b_5;
    if((*((&kernelContext_55)->work_0+(_S352 + i_4))) != 0U)
    {
        float3 _S353 = chunk_world_0(_S350.y, &kernelContext_55);
        b_5 = _S353;
    }
    else
    {
        b_5 = _S351;
    }
    uint _S354 = 2U * i_4;
    write_line_vertex_0(_S354, _S351, &kernelContext_55);
    write_line_vertex_0(_S354 + 1U, b_5, &kernelContext_55);
    return;
}

uint a_strike_0(KernelContext_0 thread* kernelContext_56)
{
    uint _S355 = a_counters_1(kernelContext_56);
    return _S355 + 16U;
}

uint strike_at_0(uint b_6, KernelContext_0 thread* kernelContext_57)
{
    return 2U * kernelContext_57->params_0->body_count_0 + 9U * kernelContext_57->params_0->chunk_count_0 + 3U * b_6;
}

[[kernel]] void fr_strike_reset(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], atomic<uint> device* atom_23 [[buffer(5)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], CrushParams_0 constant* crush_23 [[buffer(25)]], packed_float4 device* crush_rw_23 [[buffer(24)]], packed_float4 device* crush_tab_23 [[buffer(23)]])
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
    (&kernelContext_58)->crush_0 = crush_23;
    (&kernelContext_58)->crush_rw_0 = crush_rw_23;
    (&kernelContext_58)->crush_tab_0 = crush_tab_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_58)->g_scan_0 = &g_scan_23;
    uint b_7 = id_20.x;
    if(b_7 >= (params_23->body_count_0))
    {
        return;
    }
    uint _S356 = a_strike_0(&kernelContext_58);
    uint at_5 = _S356 + 9U * b_7;
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
    uint _S357 = strike_at_0(b_7, &kernelContext_58);
    (*((&kernelContext_58)->frames_0+_S357))[int(3)] = 0.0f;
    return;
}

float3 chunk_box_world_0(uint k_6, Quat_0 thread* q_17, KernelContext_0 thread* kernelContext_59)
{
    uint _S358 = body_of_0(k_6, kernelContext_59);
    thread float3 origin_5;
    structure_pose_0(_S358, 7U * _S358, q_17, &origin_5, kernelContext_59);
    float3 _S359 = (float4(*(kernelContext_59->geometry_0+3U * k_6)) ).xyz;
    thread Quat_0 _S360 = *q_17;
    float3 _S361 = rotate_0(&_S360, _S359);
    return origin_5 + _S361;
}

uint ordered_0(float f_8)
{
    uint u_0 = (as_type<uint>((f_8)));
    uint _S362;
    if((u_0 & 2147483648U) != 0U)
    {
        _S362 = ~u_0;
    }
    else
    {
        _S362 = u_0 | 2147483648U;
    }
    return _S362;
}

[[kernel]] void fr_strike_bounds(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], atomic<uint> device* atom_24 [[buffer(5)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], CrushParams_0 constant* crush_24 [[buffer(25)]], packed_float4 device* crush_rw_24 [[buffer(24)]], packed_float4 device* crush_tab_24 [[buffer(23)]])
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
    (&kernelContext_60)->crush_0 = crush_24;
    (&kernelContext_60)->crush_rw_0 = crush_rw_24;
    (&kernelContext_60)->crush_tab_0 = crush_tab_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_60)->g_scan_0 = &g_scan_24;
    uint k_7 = id_21.x;
    bool _S363;
    if(k_7 >= (params_24->chunk_count_0))
    {
        _S363 = true;
    }
    else
    {
        _S363 = ((&kernelContext_60)->params_0->shot_count_0) == 0U;
    }
    if(_S363)
    {
        return;
    }
    thread Quat_0 q_18;
    float3 _S364 = chunk_box_world_0(k_7, &q_18, &kernelContext_60);
    float _S365 = (float4(*((&kernelContext_60)->geometry_0+3U * k_7)) ).w;
    uint _S366 = a_strike_0(&kernelContext_60);
    uint _S367 = body_of_0(k_7, &kernelContext_60);
    uint _S368 = _S366 + 9U * _S367;
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
        uint _S369 = ax_0;
        uint _S370 = atomic_fetch_min_explicit((&kernelContext_60)->atom_0+(_S368 + ax_0), ordered_0(_S364[ax_0] - _S365), memory_order_relaxed);
        uint _S371 = atomic_fetch_max_explicit((&kernelContext_60)->atom_0+(_S368 + 3U + ax_0), ordered_0(_S364[_S369] + _S365), memory_order_relaxed);
        ax_0 = ax_0 + 1U;
    }
    return;
}

float unordered_0(uint u_1)
{
    uint _S372;
    if((u_1 & 2147483648U) != 0U)
    {
        _S372 = u_1 & 2147483647U;
    }
    else
    {
        _S372 = ~u_1;
    }
    return (as_type<float>((_S372)));
}

[[kernel]] void fr_strike_find(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], atomic<uint> device* atom_25 [[buffer(5)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], CrushParams_0 constant* crush_25 [[buffer(25)]], packed_float4 device* crush_rw_25 [[buffer(24)]], packed_float4 device* crush_tab_25 [[buffer(23)]])
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
    (&kernelContext_61)->crush_0 = crush_25;
    (&kernelContext_61)->crush_rw_0 = crush_rw_25;
    (&kernelContext_61)->crush_tab_0 = crush_tab_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_61)->g_scan_0 = &g_scan_25;
    uint b_8 = id_22.x;
    bool _S373;
    if(b_8 >= (params_25->body_count_0))
    {
        _S373 = true;
    }
    else
    {
        _S373 = ((&kernelContext_61)->params_0->shot_count_0) == 0U;
    }
    if(_S373)
    {
        return;
    }
    uint _S374 = a_strike_0(&kernelContext_61);
    uint at_6 = _S374 + 9U * b_8;
    uint _S375 = atomic_load_explicit((&kernelContext_61)->atom_0+at_6, memory_order_relaxed);
    if(_S375 == 4294967295U)
    {
        return;
    }
    uint _S376 = atomic_load_explicit((&kernelContext_61)->atom_0+at_6, memory_order_relaxed);
    float _S377 = unordered_0(_S376);
    uint _S378 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 1U), memory_order_relaxed);
    float _S379 = unordered_0(_S378);
    uint _S380 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 2U), memory_order_relaxed);
    float3 _S381 = float3(_S377, _S379, unordered_0(_S380));
    uint _S382 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 3U), memory_order_relaxed);
    float _S383 = unordered_0(_S382);
    uint _S384 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 4U), memory_order_relaxed);
    float _S385 = unordered_0(_S384);
    uint _S386 = atomic_load_explicit((&kernelContext_61)->atom_0+(at_6 + 5U), memory_order_relaxed);
    float3 _S387 = float3(_S383, _S385, unordered_0(_S386));
    float3 _S388 = from_rubble_0((float4(*((&kernelContext_61)->bodies_0+(7U * b_8 + 3U))) ).xyz);
    float _S389 = float_bits_0(2139095040U);
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
        float4 _S390 = float4(*((&kernelContext_61)->shots_0+2U * k_8)) ;
        uint sb_0 = 7U * (as_type<uint>((_S390.x)));
        if(((float4(*((&kernelContext_61)->bodies_0+sb_0)) ).w) == 0.0f)
        {
            k_8 = k_8 + 1U;
            continue;
        }
        float3 c_14 = from_rubble_0((float4(*((&kernelContext_61)->bodies_0+sb_0)) ).xyz);
        float3 rel_0 = from_rubble_0((float4(*((&kernelContext_61)->bodies_0+(sb_0 + 3U))) ).xyz) - _S388;
        if(!((length(rel_0)) > ((&kernelContext_61)->params_0->strike_speed_0)))
        {
            k_8 = k_8 + 1U;
            continue;
        }
        float3 _S391 = float3(_S390.y) ;
        float3 _S392 = _S381 - _S391;
        float3 _S393 = _S387 + _S391;
        float _S394 = (&kernelContext_61)->params_0->dt_0;
        float t0_0 = 0.0f;
        float t1_0 = _S394;
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
            uint _S395 = ax_1;
            if((abs(rel_0[ax_1])) < 9.999999960041972e-13f)
            {
                uint _S396 = ax_1;
                if((c_14[ax_1]) < (_S392[ax_1]))
                {
                    _S373 = true;
                }
                else
                {
                    _S373 = (c_14[_S396]) > (_S393[ax_1]);
                }
                float t0_1;
                if(_S373)
                {
                    t0_1 = _S389;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_1 = ax_1 + 1U;
                continue;
            }
            float ta_0 = (_S392[ax_1] - c_14[ax_1]) / rel_0[_S395];
            float tb_0 = (_S393[ax_1] - c_14[ax_1]) / rel_0[_S395];
            float _S397 = min(t1_0, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_0 = _S397;
            ax_1 = ax_1 + 1U;
        }
        if(t0_0 <= t1_0)
        {
            uint _S398 = strike_at_0(b_8, &kernelContext_61);
            *((&kernelContext_61)->frames_0+_S398) = packed_float4(float4(c_14, float(k_8 + 1U))) ;
            *((&kernelContext_61)->frames_0+(_S398 + 1U)) = packed_float4(float4(rel_0, t0_0)) ;
            return;
        }
        k_8 = k_8 + 1U;
    }
    return;
}

bool struck_0(uint b_9, uint thread* shot_0, KernelContext_0 thread* kernelContext_62)
{
    uint _S399 = strike_at_0(b_9, kernelContext_62);
    uint k_9 = uint((float4(*(kernelContext_62->frames_0+_S399)) ).w);
    *shot_0 = k_9 - 1U;
    return k_9 != 0U;
}

uint a_strike_dist_0(KernelContext_0 thread* kernelContext_63)
{
    uint _S400 = a_strike_0(kernelContext_63);
    return _S400 + 9U * kernelContext_63->params_0->body_count_0;
}

[[kernel]] void fr_strike_entry(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], atomic<uint> device* atom_26 [[buffer(5)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], CrushParams_0 constant* crush_26 [[buffer(25)]], packed_float4 device* crush_rw_26 [[buffer(24)]], packed_float4 device* crush_tab_26 [[buffer(23)]])
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
    (&kernelContext_64)->crush_0 = crush_26;
    (&kernelContext_64)->crush_rw_0 = crush_rw_26;
    (&kernelContext_64)->crush_tab_0 = crush_tab_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_64)->g_scan_0 = &g_scan_26;
    uint k_10 = id_23.x;
    bool _S401;
    if(k_10 >= (params_26->chunk_count_0))
    {
        _S401 = true;
    }
    else
    {
        _S401 = ((&kernelContext_64)->params_0->shot_count_0) == 0U;
    }
    if(_S401)
    {
        return;
    }
    uint _S402 = body_of_0(k_10, &kernelContext_64);
    thread uint shot_1;
    bool _S403 = struck_0(_S402, &shot_1, &kernelContext_64);
    if(!_S403)
    {
        return;
    }
    uint _S404 = strike_at_0(_S402, &kernelContext_64);
    float4 _S405 = float4(*((&kernelContext_64)->frames_0+(_S404 + 1U))) ;
    float3 from_1 = (float4(*((&kernelContext_64)->frames_0+_S404)) ).xyz;
    float3 rel_1 = _S405.xyz;
    float3 d_2 = rel_1 / float3(max(length(rel_1), 9.999999960041972e-13f)) ;
    float _S406 = (float4(*((&kernelContext_64)->shots_0+2U * shot_1)) ).y;
    thread Quat_0 q_19;
    float3 _S407 = chunk_box_world_0(k_10, &q_19, &kernelContext_64);
    uint _S408 = 3U * k_10;
    Quat_0 _S409 = quat_of_0(float4(*((&kernelContext_64)->geometry_0+(_S408 + 2U))) );
    thread Quat_0 _S410 = q_19;
    thread Quat_0 _S411 = _S409;
    Quat_0 _S412 = quat_mul_0(&_S410, &_S411);
    thread Quat_0 _S413 = _S412;
    Quat_0 _S414 = normalized_0(&_S413);
    float3 _S415 = (float4(*((&kernelContext_64)->geometry_0+(_S408 + 1U))) ).xyz;
    float _S416 = float_bits_0(2139095040U);
    float _S417 = (&kernelContext_64)->params_0->dt_0;
    float t_in_0 = 0.0f;
    float t_out_0 = _S417;
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
        thread Quat_0 _S418 = _S414;
        float3 _S419 = rotate_0(&_S418, e_1);
        float o_3 = dot(from_1 - _S407, _S419);
        float dv_0 = dot(rel_1, _S419);
        float ext_0 = _S415[ax_2] + _S406;
        if((abs(dv_0)) < 9.999999960041972e-13f)
        {
            float t_in_1;
            if((abs(o_3)) > ext_0)
            {
                t_in_1 = _S416;
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
        float _S420 = min(t_out_0, max(ta_1, tb_1));
        t_in_0 = max(t_in_0, min(ta_1, tb_1));
        t_out_0 = _S420;
        ax_2 = ax_2 + 1U;
    }
    uint _S421 = a_strike_0(&kernelContext_64);
    uint at_7 = _S421 + 9U * _S402;
    if(t_in_0 <= t_out_0)
    {
        uint _S422 = atomic_fetch_min_explicit((&kernelContext_64)->atom_0+(at_7 + 6U), (as_type<uint>((t_in_0))), memory_order_relaxed);
    }
    float3 r_7 = _S407 - (from_1 + rel_1 * float3(_S405.w) );
    float along_0 = dot(r_7, d_2);
    float dist_0 = length(r_7 - d_2 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f;
    uint _S423 = a_strike_dist_0(&kernelContext_64);
    uint _S424 = (as_type<uint>((dist_0)));
    atomic_store_explicit((&kernelContext_64)->atom_0+(_S423 + k_10), _S424, memory_order_relaxed);
    uint _S425 = atomic_fetch_min_explicit((&kernelContext_64)->atom_0+(at_7 + 7U), _S424, memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], atomic<uint> device* atom_27 [[buffer(5)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], CrushParams_0 constant* crush_27 [[buffer(25)]], packed_float4 device* crush_rw_27 [[buffer(24)]], packed_float4 device* crush_tab_27 [[buffer(23)]])
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
    (&kernelContext_65)->crush_0 = crush_27;
    (&kernelContext_65)->crush_rw_0 = crush_rw_27;
    (&kernelContext_65)->crush_tab_0 = crush_tab_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_65)->g_scan_0 = &g_scan_27;
    uint k_11 = id_24.x;
    bool _S426;
    if(k_11 >= (params_27->chunk_count_0))
    {
        _S426 = true;
    }
    else
    {
        _S426 = ((&kernelContext_65)->params_0->shot_count_0) == 0U;
    }
    if(_S426)
    {
        return;
    }
    uint _S427 = body_of_0(k_11, &kernelContext_65);
    thread uint shot_2;
    bool _S428 = struck_0(_S427, &shot_2, &kernelContext_65);
    if(!_S428)
    {
        return;
    }
    uint _S429 = a_strike_0(&kernelContext_65);
    uint at_8 = _S429 + 9U * _S427;
    uint _S430 = a_strike_dist_0(&kernelContext_65);
    uint _S431 = atomic_load_explicit((&kernelContext_65)->atom_0+(_S430 + k_11), memory_order_relaxed);
    uint _S432 = atomic_load_explicit((&kernelContext_65)->atom_0+(at_8 + 7U), memory_order_relaxed);
    if(_S431 == _S432)
    {
        uint _S433 = atomic_fetch_min_explicit((&kernelContext_65)->atom_0+(at_8 + 8U), k_11, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], atomic<uint> device* atom_28 [[buffer(5)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], CrushParams_0 constant* crush_28 [[buffer(25)]], packed_float4 device* crush_rw_28 [[buffer(24)]], packed_float4 device* crush_tab_28 [[buffer(23)]])
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
    (&kernelContext_66)->crush_0 = crush_28;
    (&kernelContext_66)->crush_rw_0 = crush_rw_28;
    (&kernelContext_66)->crush_tab_0 = crush_tab_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_66)->g_scan_0 = &g_scan_28;
    uint b_10 = id_25.x;
    bool _S434;
    if(b_10 >= (params_28->body_count_0))
    {
        _S434 = true;
    }
    else
    {
        _S434 = ((&kernelContext_66)->params_0->shot_count_0) == 0U;
    }
    if(_S434)
    {
        return;
    }
    thread uint shot_3;
    bool _S435 = struck_0(b_10, &shot_3, &kernelContext_66);
    if(!_S435)
    {
        return;
    }
    uint _S436 = a_strike_0(&kernelContext_66);
    uint at_9 = _S436 + 9U * b_10;
    uint _S437 = atomic_load_explicit((&kernelContext_66)->atom_0+(at_9 + 6U), memory_order_relaxed);
    float t_in_2 = (as_type<float>((_S437)));
    uint entry_0 = atomic_load_explicit((&kernelContext_66)->atom_0+(at_9 + 8U), memory_order_relaxed);
    uint _S438 = strike_at_0(b_10, &kernelContext_66);
    if(!(t_in_2 <= ((&kernelContext_66)->params_0->dt_0)))
    {
        _S434 = true;
    }
    else
    {
        _S434 = entry_0 == 4294967295U;
    }
    if(_S434)
    {
        (*((&kernelContext_66)->frames_0+_S438))[int(3)] = 0.0f;
        return;
    }
    float3 rel_2 = (float4(*((&kernelContext_66)->frames_0+(_S438 + 1U))) ).xyz;
    float speed_0 = length(rel_2);
    float4 _S439 = float4(*((&kernelContext_66)->shots_0+2U * shot_3)) ;
    float _S440 = max(_S439.z, 0.00100000004749745f);
    float m_imp_0 = _S439.w;
    float e_imp_0 = (float4(*((&kernelContext_66)->shots_0+(2U * shot_3 + 1U))) ).x;
    float3 contact_0 = (float4(*((&kernelContext_66)->frames_0+_S438)) ).xyz + rel_2 * float3(t_in_2)  + rel_2 / float3(max(speed_0, 9.999999960041972e-13f))  * float3(_S439.y) ;
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
    *((&kernelContext_66)->frames_0+(_S438 + 2U)) = packed_float4(float4(contact_0, m_eff_0 * speed_0 / (2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (_S440 * modulus_0 * modulus_0 * max(speed_0, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

[[kernel]] void fr_strike_breaks(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], atomic<uint> device* atom_29 [[buffer(5)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], CrushParams_0 constant* crush_29 [[buffer(25)]], packed_float4 device* crush_rw_29 [[buffer(24)]], packed_float4 device* crush_tab_29 [[buffer(23)]])
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
    (&kernelContext_67)->crush_0 = crush_29;
    (&kernelContext_67)->crush_rw_0 = crush_rw_29;
    (&kernelContext_67)->crush_tab_0 = crush_tab_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_67)->g_scan_0 = &g_scan_29;
    uint i_5 = id_26.x;
    bool _S441;
    if(i_5 >= (params_29->bond_count_0))
    {
        _S441 = true;
    }
    else
    {
        _S441 = ((&kernelContext_67)->params_0->shot_count_0) == 0U;
    }
    if(_S441)
    {
        _S441 = true;
    }
    else
    {
        uint _S442 = w_alive_0(&kernelContext_67);
        _S441 = (*((&kernelContext_67)->work_0+(_S442 + i_5))) == 0U;
    }
    if(_S441)
    {
        return;
    }
    uint4 _S443 = uint4(*((&kernelContext_67)->bonds_0+i_5)) ;
    uint _S444 = _S443.x;
    uint _S445 = body_of_0(_S444, &kernelContext_67);
    thread uint shot_4;
    bool _S446 = struck_0(_S445, &shot_4, &kernelContext_67);
    if(!_S446)
    {
        return;
    }
    uint _S447 = strike_at_0(_S445, &kernelContext_67);
    float4 _S448 = float4(*((&kernelContext_67)->frames_0+(_S447 + 2U))) ;
    thread Quat_0 q_20;
    float3 _S449 = chunk_box_world_0(_S444, &q_20, &kernelContext_67);
    float3 _S450 = chunk_box_world_0(_S443.y, &q_20, &kernelContext_67);
    float dist_1 = length(float3(0.5f)  * (_S449 + _S450) - _S448.xyz);
    if((_S448.w * min(1.0f, (as_type<float>((_S443.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((_S443.z))))
    {
        uint _S451 = w_alive_0(&kernelContext_67);
        *((&kernelContext_67)->work_0+(_S451 + i_5)) = 0U;
        uint _S452 = a_dirty_0(&kernelContext_67);
        uint _S453 = atomic_fetch_or_explicit((&kernelContext_67)->atom_0+(_S452 + _S445), 1U, memory_order_relaxed);
        uint _S454 = a_counters_0(&kernelContext_67);
        uint _S455 = atomic_fetch_add_explicit((&kernelContext_67)->atom_0+_S454, 1U, memory_order_relaxed);
        uint _S456 = atomic_fetch_add_explicit((&kernelContext_67)->atom_0+(_S454 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

bool crushable_0(uint k_12, KernelContext_0 thread* kernelContext_68)
{
    return ((as_type<uint>(((float4(*(kernelContext_68->crush_tab_0+(2U * k_12 + 1U))) ).z))) & 1U) != 0U;
}

uint crush_flags_0(uint k_13, KernelContext_0 thread* kernelContext_69)
{
    return (as_type<uint>(((float4(*(kernelContext_69->crush_rw_0+k_13)) ).w)));
}

void crush_record_0(uint k_14, float4 step_4, uint cause_0, KernelContext_0 thread* kernelContext_70)
{
    float4 _S457 = float4(*(kernelContext_70->crush_rw_0+k_14)) ;
    uint flags_1 = (as_type<uint>((_S457.w)));
    uint flags_2;
    if((step_4.w) != 0.0f)
    {
        flags_2 = ((flags_1 | 1U) | 2U) | (cause_0 << 4U);
    }
    else
    {
        flags_2 = flags_1;
    }
    *(kernelContext_70->crush_rw_0+k_14) = packed_float4(float4(step_4.x, max(_S457.y, step_4.y), max(_S457.z, step_4.z), (as_type<float>((flags_2))))) ;
    return;
}

[[kernel]] void fr_crush_blast(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], atomic<uint> device* atom_30 [[buffer(5)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], CrushParams_0 constant* crush_30 [[buffer(25)]], packed_float4 device* crush_rw_30 [[buffer(24)]], packed_float4 device* crush_tab_30 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_71;
    (&kernelContext_71)->params_0 = params_30;
    (&kernelContext_71)->work_0 = work_30;
    (&kernelContext_71)->frames_0 = frames_30;
    (&kernelContext_71)->bodies_0 = bodies_30;
    (&kernelContext_71)->chunks_0 = chunks_30;
    (&kernelContext_71)->sort_0 = sort_30;
    (&kernelContext_71)->atom_0 = atom_30;
    (&kernelContext_71)->manifolds_0 = manifolds_30;
    (&kernelContext_71)->loads_0 = loads_30;
    (&kernelContext_71)->bonds_0 = bonds_30;
    (&kernelContext_71)->args_0 = args_30;
    (&kernelContext_71)->info_0 = info_30;
    (&kernelContext_71)->scan_0 = scan_30;
    (&kernelContext_71)->parts_0 = parts_30;
    (&kernelContext_71)->colliders_0 = colliders_30;
    (&kernelContext_71)->collider_info_0 = collider_info_30;
    (&kernelContext_71)->view_chunks_0 = view_chunks_30;
    (&kernelContext_71)->view_0 = view_30;
    (&kernelContext_71)->objects_0 = objects_30;
    (&kernelContext_71)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_71)->lines_0 = lines_30;
    (&kernelContext_71)->geometry_0 = geometry_30;
    (&kernelContext_71)->shots_0 = shots_30;
    (&kernelContext_71)->crush_0 = crush_30;
    (&kernelContext_71)->crush_rw_0 = crush_rw_30;
    (&kernelContext_71)->crush_tab_0 = crush_tab_30;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_71)->g_scan_0 = &g_scan_30;
    uint k_15 = id_27.x;
    bool _S458;
    if(k_15 >= (params_30->chunk_count_0))
    {
        _S458 = true;
    }
    else
    {
        _S458 = ((&kernelContext_71)->crush_0->enabled_0) == 0U;
    }
    if(_S458)
    {
        return;
    }
    float4 _S459 = float4(*((&kernelContext_71)->crush_rw_0+((&kernelContext_71)->crush_0->r_blast_0 + k_15))) ;
    float _S460 = _S459.x;
    if(!(_S460 > 0.0f))
    {
        _S458 = !((_S459.y) > 0.0f);
    }
    else
    {
        _S458 = false;
    }
    if(_S458)
    {
        _S458 = !((_S459.z) > 0.0f);
    }
    else
    {
        _S458 = false;
    }
    if(_S458)
    {
        return;
    }
    *((&kernelContext_71)->crush_rw_0+((&kernelContext_71)->crush_0->r_blast_0 + k_15)) = packed_float4(float4(0.0f) ) ;
    bool _S461 = crushable_0(k_15, &kernelContext_71);
    if(!_S461)
    {
        _S458 = true;
    }
    else
    {
        uint _S462 = crush_flags_0(k_15, &kernelContext_71);
        _S458 = (_S462 & 1U) != 0U;
    }
    if(_S458)
    {
        return;
    }
    float d_3 = (float4(*((&kernelContext_71)->crush_rw_0+k_15)) ).x + _S460;
    float4 step_5;
    if(d_3 >= 1.0f)
    {
        step_5 = float4(1.0f, _S459.y, _S459.z, 1.0f);
    }
    else
    {
        step_5 = float4(d_3, _S459.y, _S459.z, 0.0f);
    }
    crush_record_0(k_15, step_5, 3U, &kernelContext_71);
    return;
}

struct Sym_0
{
    float3 d_4;
    float3 o_4;
};

Sym_0 sym_zero_0()
{
    thread Sym_0 s_4;
    float3 _S463 = float3(0.0f) ;
    (&s_4)->d_4 = _S463;
    (&s_4)->o_4 = _S463;
    return s_4;
}

Sym_0 sym_outer_0(float3 r_8, float3 f_9)
{
    thread Sym_0 s_5;
    (&s_5)->d_4 = r_8 * f_9;
    float _S464 = r_8.x;
    float _S465 = f_9.y;
    float _S466 = r_8.y;
    float _S467 = f_9.x;
    float _S468 = f_9.z;
    float _S469 = r_8.z;
    (&s_5)->o_4 = float3(0.5f)  * float3(_S464 * _S465 + _S466 * _S467, _S464 * _S468 + _S469 * _S467, _S466 * _S468 + _S469 * _S465);
    return s_5;
}

Sym_0 sym_add_0(const Sym_0 thread* a_4, const Sym_0 thread* b_11)
{
    thread Sym_0 s_6;
    (&s_6)->d_4 = a_4->d_4 + b_11->d_4;
    (&s_6)->o_4 = a_4->o_4 + b_11->o_4;
    return s_6;
}

void add_load_0(Sym_0 thread* w_6, float3 thread* f_sum_0, float4 thread* spread_0, float3 r_9, float3 f_10)
{
    Sym_0 _S470 = sym_outer_0(r_9, f_10);
    thread Sym_0 _S471 = *w_6;
    thread Sym_0 _S472 = _S470;
    Sym_0 _S473 = sym_add_0(&_S471, &_S472);
    *w_6 = _S473;
    *f_sum_0 = *f_sum_0 + f_10;
    float m_3 = length(f_10);
    *spread_0 = *spread_0 + float4(float3(m_3)  * r_9, m_3);
    return;
}

void flow_virial_0(uint k_16, const Quat_0 thread* q_21, float3 origin_6, float3 centre_1, Sym_0 thread* wf_0, float thread* total_0, KernelContext_0 thread* kernelContext_72)
{
    *wf_0 = sym_zero_0();
    *total_0 = 0.0f;
    float4 _S474 = float4(*(kernelContext_72->crush_tab_0+(kernelContext_72->crush_0->t_adj_0 + k_16))) ;
    uint first_2 = (as_type<uint>((_S474.x)));
    uint _S475 = (as_type<uint>((_S474.y)));
    uint j_3 = first_2;
    for(;;)
    {
        if(j_3 < (first_2 + _S475))
        {
        }
        else
        {
            break;
        }
        uint bi_0 = (as_type<uint>(((float4(*(kernelContext_72->crush_tab_0+(kernelContext_72->crush_0->t_list_0 + j_3 / 4U))) )[j_3 % 4U])));
        uint _S476 = w_alive_0(kernelContext_72);
        if((*(kernelContext_72->work_0+(_S476 + bi_0))) == 0U)
        {
            j_3 = j_3 + 1U;
            continue;
        }
        float l_0 = (*(kernelContext_72->crush_rw_0+(kernelContext_72->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U];
        if(!(((*(kernelContext_72->crush_rw_0+(kernelContext_72->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U]) > 0.0f))
        {
            j_3 = j_3 + 1U;
            continue;
        }
        float3 _S477 = rotate_0(q_21, (float4(*(kernelContext_72->crush_tab_0+(kernelContext_72->crush_0->t_bonds_0 + bi_0))) ).xyz);
        float3 r_10 = origin_6 + _S477 - centre_1;
        float3 _S478 = rotate_0(q_21, (float4(*(kernelContext_72->crush_tab_0+(kernelContext_72->crush_0->t_normal_0 + bi_0))) ).xyz);
        float3 out_n_0;
        if((dot(_S478, r_10)) >= 0.0f)
        {
            out_n_0 = _S478;
        }
        else
        {
            out_n_0 = - _S478;
        }
        Sym_0 _S479 = sym_outer_0(r_10, float3(- l_0)  * out_n_0);
        thread Sym_0 _S480 = *wf_0;
        thread Sym_0 _S481 = _S479;
        Sym_0 _S482 = sym_add_0(&_S480, &_S481);
        *wf_0 = _S482;
        *total_0 = *total_0 + l_0;
        j_3 = j_3 + 1U;
    }
    return;
}

float4 bond_centroid_0(uint k_17, const Quat_0 thread* q_22, float3 origin_7, float3 centre_2, KernelContext_0 thread* kernelContext_73)
{
    float4 _S483 = float4(*(kernelContext_73->crush_tab_0+(kernelContext_73->crush_0->t_adj_0 + k_17))) ;
    uint first_3 = (as_type<uint>((_S483.x)));
    uint _S484 = (as_type<uint>((_S483.y)));
    float3 _S485 = float3(0.0f) ;
    uint j_4 = first_3;
    float3 sum_4 = _S485;
    float area_0 = 0.0f;
    for(;;)
    {
        if(j_4 < (first_3 + _S484))
        {
        }
        else
        {
            break;
        }
        uint bi_1 = (as_type<uint>(((float4(*(kernelContext_73->crush_tab_0+(kernelContext_73->crush_0->t_list_0 + j_4 / 4U))) )[j_4 % 4U])));
        uint _S486 = w_alive_0(kernelContext_73);
        if((*(kernelContext_73->work_0+(_S486 + bi_1))) == 0U)
        {
            j_4 = j_4 + 1U;
            continue;
        }
        float4 _S487 = float4(*(kernelContext_73->crush_tab_0+(kernelContext_73->crush_0->t_bonds_0 + bi_1))) ;
        float3 _S488 = rotate_0(q_22, _S487.xyz);
        float _S489 = _S487.w;
        float area_1 = area_0 + _S489;
        sum_4 = sum_4 + float3(_S489)  * (origin_7 + _S488 - centre_2);
        area_0 = area_1;
        j_4 = j_4 + 1U;
    }
    if(area_0 > 0.0f)
    {
        sum_4 = sum_4 / float3(area_0) ;
    }
    else
    {
        sum_4 = _S485;
    }
    return float4(sum_4, area_0);
}

Sym_0 balanced_0(const Sym_0 thread* w_7, float3 f_11, float4 spread_1, float4 bonds_31)
{
    float3 at_10;
    if((bonds_31.w) > 0.0f)
    {
        at_10 = bonds_31.xyz;
    }
    else
    {
        float _S490 = spread_1.w;
        if(_S490 > 0.0f)
        {
            at_10 = spread_1.xyz / float3(_S490) ;
        }
        else
        {
            at_10 = float3(0.0f) ;
        }
    }
    thread Sym_0 _S491 = sym_outer_0(at_10, - f_11);
    Sym_0 _S492 = sym_add_0(w_7, &_S491);
    return _S492;
}

float sym_norm2_0(const Sym_0 thread* s_7)
{
    return dot(s_7->d_4, s_7->d_4) + 2.0f * dot(s_7->o_4, s_7->o_4);
}

float2 crush_invariants_0(const Sym_0 thread* v_9, float volume_0)
{
    float inv_4 = 1.0f / volume_0;
    float xx_0 = v_9->d_4.x * inv_4;
    float yy_0 = v_9->d_4.y * inv_4;
    float zz_0 = v_9->d_4.z * inv_4;
    float xy_0 = v_9->o_4.x * inv_4;
    float xz_0 = v_9->o_4.y * inv_4;
    float yz_0 = v_9->o_4.z * inv_4;
    float p_5 = - (xx_0 + yy_0 + zz_0) / 3.0f;
    float dx_0 = xx_0 + p_5;
    float dy_0 = yy_0 + p_5;
    float dz_0 = zz_0 + p_5;
    return float2(p_5, sqrt(1.5f * max(dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0 + 2.0f * (xy_0 * xy_0 + xz_0 * xz_0 + yz_0 * yz_0), 0.0f)));
}

float crush_excess_0(float2 pq_0, float4 m0_0)
{
    float _S493 = pq_0.x;
    if(!(_S493 > 0.0f))
    {
        return 0.0f;
    }
    float _S494 = max(pq_0.y - (m0_0.y + m0_0.z * _S493), _S493 - m0_0.x);
    float _S495;
    if(_S494 > 0.0f)
    {
        _S495 = _S494;
    }
    else
    {
        _S495 = 0.0f;
    }
    return _S495;
}

float4 crush_two_0(const Sym_0 thread* a_5, float dt_a_0, const Sym_0 thread* b_12, float dt_b_0, uint k_18, float damage_0, KernelContext_0 thread* kernelContext_74)
{
    uint _S496 = 2U * k_18;
    float4 _S497 = float4(*(kernelContext_74->crush_tab_0+_S496)) ;
    float4 _S498 = float4(*(kernelContext_74->crush_tab_0+(_S496 + 1U))) ;
    float _S499 = _S498.y;
    float2 _S500 = crush_invariants_0(a_5, _S499);
    float2 _S501 = crush_invariants_0(b_12, _S499);
    float ea_0 = crush_excess_0(_S500, _S497);
    bool _S502 = dt_b_0 > 0.0f;
    float eb_0;
    if(_S502)
    {
        eb_0 = crush_excess_0(_S501, _S497);
    }
    else
    {
        eb_0 = 0.0f;
    }
    float d_5 = damage_0 + (ea_0 * ea_0 * dt_a_0 + eb_0 * eb_0 * dt_b_0) / (_S498.x * _S497.w);
    float2 peak_2;
    if(_S502)
    {
        peak_2 = max(_S500, _S501);
    }
    else
    {
        peak_2 = _S500;
    }
    if(d_5 >= 1.0f)
    {
        return float4(1.0f, peak_2, 1.0f);
    }
    return float4(d_5, peak_2, 0.0f);
}

void crush_loads_flow_0(uint k_19, KernelContext_0 thread* kernelContext_75)
{
    uint _S503 = body_of_0(k_19, kernelContext_75);
    uint b_13 = 7U * _S503;
    thread Quat_0 q_23;
    thread float3 origin_8;
    structure_pose_0(_S503, b_13, &q_23, &origin_8, kernelContext_75);
    uint _S504 = 4U * k_19;
    float4 _S505 = float4(*(kernelContext_75->chunks_0+_S504)) ;
    float4 _S506 = float4(*(kernelContext_75->chunks_0+(_S504 + 3U))) ;
    float3 _S507 = _S505.xyz;
    thread Quat_0 _S508 = q_23;
    float3 _S509 = rotate_0(&_S508, _S507);
    float3 centre_3 = origin_8 + _S509;
    float inv_mass_4 = (float4(*(kernelContext_75->bodies_0+b_13)) ).w;
    float _S510;
    if(inv_mass_4 > 0.0f)
    {
        _S510 = 1.0f / inv_mass_4;
    }
    else
    {
        _S510 = float_bits_0(2139095040U);
    }
    float dt_2 = kernelContext_75->params_0->dt_0;
    Sym_0 _S511 = sym_zero_0();
    thread Sym_0 wa_0 = _S511;
    float3 _S512 = float3(0.0f) ;
    thread float3 fa_0 = _S512;
    float4 _S513 = float4(0.0f) ;
    thread float4 sa_0 = _S513;
    thread Sym_0 wb_0 = _S511;
    thread float3 fb_0 = _S512;
    thread float4 sb_1 = _S513;
    uint _S514 = kernelContext_75->sort_0[k_19];
    uint _S515 = kernelContext_75->sort_0[kernelContext_75->params_0->chunk_count_0 + 1U + k_19];
    float tau_0 = 0.0f;
    uint i_6 = 0U;
    for(;;)
    {
        if(i_6 < _S515)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_11 = kernelContext_75->manifolds_0[_S514 + i_6];
        float4 _S516 = float4(r_11.impulse_0) ;
        float3 _S517 = _S516.xyz;
        float j_5 = length(_S517);
        if(j_5 == 0.0f)
        {
            i_6 = i_6 + 1U;
            continue;
        }
        float4 _S518 = float4(r_11.other_0) ;
        float m_other_2 = _S518.x;
        float m_4;
        if(isinf(_S510))
        {
            if(isinf(m_other_2))
            {
                m_4 = _S505.w;
            }
            else
            {
                m_4 = m_other_2;
            }
        }
        else
        {
            if(isinf(m_other_2))
            {
                m_4 = _S510;
            }
            else
            {
                m_4 = _S510 * m_other_2 / (_S510 + m_other_2);
            }
        }
        float _S519 = _S516.w;
        float _S520 = min(j_5, 2.0f * m_4 * max(_S519 - kernelContext_75->params_0->band_0, 0.0f));
        float3 rr_0 = (float4(r_11.point_0) ).xyz - centre_3;
        float3 dir_1 = _S517 / float3(j_5) ;
        if(_S520 < j_5)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, rr_0, dir_1 * float3(((j_5 - _S520) / dt_2)) );
        }
        float tau_1;
        if(_S520 > 0.0f)
        {
            float _S521 = _S506.y;
            float duration_2 = hertz_duration_0(m_4, min(_S518.z, _S506.z), 1.0f / ((1.0f - _S521 * _S521) / _S506.x + 1.0f / _S518.y), _S519);
            add_load_0(&wb_0, &fb_0, &sb_1, rr_0, dir_1 * float3((3.14159274101257324f * _S520 / (2.0f * duration_2))) );
            tau_1 = max(tau_0, min(duration_2, dt_2));
        }
        else
        {
            tau_1 = tau_0;
        }
        tau_0 = tau_1;
        i_6 = i_6 + 1U;
    }
    uint from_2 = kernelContext_75->params_0->parity_0;
    uint _S522 = w_pulses_0(kernelContext_75);
    uint _S523 = _S522 + _S504 + 2U * from_2;
    uint off_1 = *(kernelContext_75->work_0+_S523);
    uint _S524 = *(kernelContext_75->work_0+(_S523 + 1U));
    uint _S525 = l_pool_0(from_2, kernelContext_75);
    uint _S526 = _S525 + 3U * off_1;
    uint p_6 = 0U;
    for(;;)
    {
        if(p_6 < _S524)
        {
        }
        else
        {
            break;
        }
        uint _S527 = _S526 + 3U * p_6;
        float4 _S528 = float4(*(kernelContext_75->loads_0+_S527)) ;
        float4 _S529 = float4(*(kernelContext_75->loads_0+(_S527 + 1U))) ;
        float4 _S530 = float4(*(kernelContext_75->loads_0+(_S527 + 2U))) ;
        float mean_2 = pulse_mean_0(_S528.x, _S528.y, _S528.z, _S528.w, _S529.w, dt_2);
        if(mean_2 != 0.0f)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, _S529.xyz - centre_3, _S530.xyz * float3(mean_2) );
        }
        p_6 = p_6 + 1U;
    }
    thread Quat_0 _S531 = q_23;
    thread Sym_0 wf_1;
    thread float flow_total_0;
    flow_virial_0(k_19, &_S531, origin_8, centre_3, &wf_1, &flow_total_0, kernelContext_75);
    bool contact_1;
    if((sa_0.w) > 0.0f)
    {
        contact_1 = true;
    }
    else
    {
        contact_1 = (sb_1.w) > 0.0f;
    }
    bool _S532;
    if(!contact_1)
    {
        _S532 = !(flow_total_0 > 0.0f);
    }
    else
    {
        _S532 = false;
    }
    if(_S532)
    {
        return;
    }
    Sym_0 _S533 = wf_1;
    Sym_0 _S534 = wf_1;
    Sym_0 peak_contact_0;
    Sym_0 base_3;
    Sym_0 peak_3;
    if(contact_1)
    {
        thread Quat_0 _S535 = q_23;
        float4 _S536 = bond_centroid_0(k_19, &_S535, origin_8, centre_3, kernelContext_75);
        thread Sym_0 _S537 = wa_0;
        Sym_0 _S538 = balanced_0(&_S537, fa_0, sa_0, _S536);
        thread Sym_0 _S539 = _S538;
        thread Sym_0 _S540 = wf_1;
        Sym_0 _S541 = sym_add_0(&_S539, &_S540);
        thread Sym_0 _S542 = wa_0;
        thread Sym_0 _S543 = wb_0;
        Sym_0 _S544 = sym_add_0(&_S542, &_S543);
        float3 _S545 = fa_0 + fb_0;
        float4 _S546 = sa_0 + sb_1;
        thread Sym_0 _S547 = _S544;
        Sym_0 _S548 = balanced_0(&_S547, _S545, _S546, _S536);
        thread Sym_0 _S549 = _S548;
        thread Sym_0 _S550 = wf_1;
        Sym_0 _S551 = sym_add_0(&_S549, &_S550);
        peak_contact_0 = _S548;
        base_3 = _S541;
        peak_3 = _S551;
    }
    else
    {
        peak_contact_0 = _S511;
        base_3 = _S533;
        peak_3 = _S534;
    }
    thread Sym_0 _S552 = wf_1;
    float _S553 = sym_norm2_0(&_S552);
    thread Sym_0 _S554 = peak_contact_0;
    float _S555 = sym_norm2_0(&_S554);
    uint cause_1;
    if(_S553 >= _S555)
    {
        cause_1 = 4U;
    }
    else
    {
        cause_1 = 1U;
    }
    float _S556 = dt_2 - tau_0;
    float _S557 = (float4(*(kernelContext_75->crush_rw_0+k_19)) ).x;
    thread Sym_0 _S558 = base_3;
    thread Sym_0 _S559 = peak_3;
    float4 _S560 = crush_two_0(&_S558, _S556, &_S559, tau_0, k_19, _S557, kernelContext_75);
    crush_record_0(k_19, _S560, cause_1, kernelContext_75);
    return;
}

[[kernel]] void fr_crush_loads(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], atomic<uint> device* atom_31 [[buffer(5)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_32 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], CrushParams_0 constant* crush_31 [[buffer(25)]], packed_float4 device* crush_rw_31 [[buffer(24)]], packed_float4 device* crush_tab_31 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_76;
    (&kernelContext_76)->params_0 = params_31;
    (&kernelContext_76)->work_0 = work_31;
    (&kernelContext_76)->frames_0 = frames_31;
    (&kernelContext_76)->bodies_0 = bodies_31;
    (&kernelContext_76)->chunks_0 = chunks_31;
    (&kernelContext_76)->sort_0 = sort_31;
    (&kernelContext_76)->atom_0 = atom_31;
    (&kernelContext_76)->manifolds_0 = manifolds_31;
    (&kernelContext_76)->loads_0 = loads_31;
    (&kernelContext_76)->bonds_0 = bonds_32;
    (&kernelContext_76)->args_0 = args_31;
    (&kernelContext_76)->info_0 = info_31;
    (&kernelContext_76)->scan_0 = scan_31;
    (&kernelContext_76)->parts_0 = parts_31;
    (&kernelContext_76)->colliders_0 = colliders_31;
    (&kernelContext_76)->collider_info_0 = collider_info_31;
    (&kernelContext_76)->view_chunks_0 = view_chunks_31;
    (&kernelContext_76)->view_0 = view_31;
    (&kernelContext_76)->objects_0 = objects_31;
    (&kernelContext_76)->shot_bodies_0 = shot_bodies_31;
    (&kernelContext_76)->lines_0 = lines_31;
    (&kernelContext_76)->geometry_0 = geometry_31;
    (&kernelContext_76)->shots_0 = shots_31;
    (&kernelContext_76)->crush_0 = crush_31;
    (&kernelContext_76)->crush_rw_0 = crush_rw_31;
    (&kernelContext_76)->crush_tab_0 = crush_tab_31;
    threadgroup array<uint, int(1024)> g_scan_31;
    (&kernelContext_76)->g_scan_0 = &g_scan_31;
    uint k_20 = id_28.x;
    bool _S561;
    if(k_20 >= (params_31->chunk_count_0))
    {
        _S561 = true;
    }
    else
    {
        _S561 = ((&kernelContext_76)->crush_0->enabled_0) == 0U;
    }
    if(_S561)
    {
        _S561 = true;
    }
    else
    {
        bool _S562 = crushable_0(k_20, &kernelContext_76);
        _S561 = !_S562;
    }
    if(_S561)
    {
        _S561 = true;
    }
    else
    {
        uint _S563 = crush_flags_0(k_20, &kernelContext_76);
        _S561 = (_S563 & 1U) != 0U;
    }
    if(_S561)
    {
        return;
    }
    if(((&kernelContext_76)->crush_0->flow_0) != 0U)
    {
        crush_loads_flow_0(k_20, &kernelContext_76);
        return;
    }
    uint _S564 = body_of_0(k_20, &kernelContext_76);
    uint b_14 = 7U * _S564;
    thread Quat_0 q_24;
    thread float3 origin_9;
    structure_pose_0(_S564, b_14, &q_24, &origin_9, &kernelContext_76);
    uint _S565 = 4U * k_20;
    float4 _S566 = float4(*((&kernelContext_76)->chunks_0+_S565)) ;
    float4 _S567 = float4(*((&kernelContext_76)->chunks_0+(_S565 + 3U))) ;
    float3 _S568 = _S566.xyz;
    thread Quat_0 _S569 = q_24;
    float3 _S570 = rotate_0(&_S569, _S568);
    float3 centre_4 = origin_9 + _S570;
    float inv_mass_5 = (float4(*((&kernelContext_76)->bodies_0+b_14)) ).w;
    float _S571;
    if(inv_mass_5 > 0.0f)
    {
        _S571 = 1.0f / inv_mass_5;
    }
    else
    {
        _S571 = float_bits_0(2139095040U);
    }
    float dt_3 = (&kernelContext_76)->params_0->dt_0;
    Sym_0 _S572 = sym_zero_0();
    thread Sym_0 wa_1 = _S572;
    float3 _S573 = float3(0.0f) ;
    thread float3 fa_1 = _S573;
    float4 _S574 = float4(0.0f) ;
    thread float4 sa_1 = _S574;
    thread Sym_0 wb_1 = _S572;
    thread float3 fb_1 = _S573;
    thread float4 sb_2 = _S574;
    uint _S575 = (&kernelContext_76)->sort_0[k_20];
    uint _S576 = (&kernelContext_76)->sort_0[params_31->chunk_count_0 + 1U + k_20];
    float tau_2 = 0.0f;
    uint i_7 = 0U;
    for(;;)
    {
        if(i_7 < _S576)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_12 = (&kernelContext_76)->manifolds_0[_S575 + i_7];
        float4 _S577 = float4(r_12.impulse_0) ;
        float3 _S578 = _S577.xyz;
        float j_6 = length(_S578);
        if(j_6 == 0.0f)
        {
            i_7 = i_7 + 1U;
            continue;
        }
        float4 _S579 = float4(r_12.other_0) ;
        float m_other_3 = _S579.x;
        float m_5;
        if(isinf(_S571))
        {
            if(isinf(m_other_3))
            {
                m_5 = _S566.w;
            }
            else
            {
                m_5 = m_other_3;
            }
        }
        else
        {
            if(isinf(m_other_3))
            {
                m_5 = _S571;
            }
            else
            {
                m_5 = _S571 * m_other_3 / (_S571 + m_other_3);
            }
        }
        float _S580 = _S577.w;
        float _S581 = min(j_6, 2.0f * m_5 * max(_S580 - (&kernelContext_76)->params_0->band_0, 0.0f));
        float3 rr_1 = (float4(r_12.point_0) ).xyz - centre_4;
        float3 dir_2 = _S578 / float3(j_6) ;
        if(_S581 < j_6)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, rr_1, dir_2 * float3(((j_6 - _S581) / dt_3)) );
        }
        float tau_3;
        if(_S581 > 0.0f)
        {
            float _S582 = _S567.y;
            float duration_3 = hertz_duration_0(m_5, min(_S579.z, _S567.z), 1.0f / ((1.0f - _S582 * _S582) / _S567.x + 1.0f / _S579.y), _S580);
            add_load_0(&wb_1, &fb_1, &sb_2, rr_1, dir_2 * float3((3.14159274101257324f * _S581 / (2.0f * duration_3))) );
            tau_3 = max(tau_2, min(duration_3, dt_3));
        }
        else
        {
            tau_3 = tau_2;
        }
        tau_2 = tau_3;
        i_7 = i_7 + 1U;
    }
    uint from_3 = (&kernelContext_76)->params_0->parity_0;
    uint _S583 = w_pulses_0(&kernelContext_76);
    uint _S584 = _S583 + _S565 + 2U * from_3;
    uint device* _S585 = (&kernelContext_76)->work_0+_S584;
    uint _S586 = *((&kernelContext_76)->work_0+(_S584 + 1U));
    uint _S587 = l_pool_0(from_3, &kernelContext_76);
    uint _S588 = _S587 + 3U * *_S585;
    uint p_7 = 0U;
    for(;;)
    {
        if(p_7 < _S586)
        {
        }
        else
        {
            break;
        }
        uint _S589 = _S588 + 3U * p_7;
        float4 _S590 = float4(*((&kernelContext_76)->loads_0+_S589)) ;
        float4 _S591 = float4(*((&kernelContext_76)->loads_0+(_S589 + 1U))) ;
        float4 _S592 = float4(*((&kernelContext_76)->loads_0+(_S589 + 2U))) ;
        float mean_3 = pulse_mean_0(_S590.x, _S590.y, _S590.z, _S590.w, _S591.w, dt_3);
        if(mean_3 != 0.0f)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, _S591.xyz - centre_4, _S592.xyz * float3(mean_3) );
        }
        p_7 = p_7 + 1U;
    }
    if(!((sa_1.w) > 0.0f))
    {
        _S561 = !((sb_2.w) > 0.0f);
    }
    else
    {
        _S561 = false;
    }
    if(_S561)
    {
        return;
    }
    thread Quat_0 _S593 = q_24;
    float4 _S594 = bond_centroid_0(k_20, &_S593, origin_9, centre_4, &kernelContext_76);
    thread Sym_0 _S595 = wa_1;
    Sym_0 _S596 = balanced_0(&_S595, fa_1, sa_1, _S594);
    thread Sym_0 _S597 = wa_1;
    thread Sym_0 _S598 = wb_1;
    Sym_0 _S599 = sym_add_0(&_S597, &_S598);
    float3 _S600 = fa_1 + fb_1;
    float4 _S601 = sa_1 + sb_2;
    thread Sym_0 _S602 = _S599;
    Sym_0 _S603 = balanced_0(&_S602, _S600, _S601, _S594);
    float _S604 = dt_3 - tau_2;
    float _S605 = (float4(*((&kernelContext_76)->crush_rw_0+k_20)) ).x;
    thread Sym_0 _S606 = _S596;
    thread Sym_0 _S607 = _S603;
    float4 _S608 = crush_two_0(&_S606, _S604, &_S607, tau_2, k_20, _S605, &kernelContext_76);
    crush_record_0(k_20, _S608, 1U, &kernelContext_76);
    return;
}

uint a_cover_0(KernelContext_0 thread* kernelContext_77)
{
    uint _S609 = a_strike_dist_0(kernelContext_77);
    return _S609 + kernelContext_77->params_0->chunk_count_0;
}

[[kernel]] void fr_strike_patch(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], atomic<uint> device* atom_32 [[buffer(5)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], CrushParams_0 constant* crush_32 [[buffer(25)]], packed_float4 device* crush_rw_32 [[buffer(24)]], packed_float4 device* crush_tab_32 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_78;
    (&kernelContext_78)->params_0 = params_32;
    (&kernelContext_78)->work_0 = work_32;
    (&kernelContext_78)->frames_0 = frames_32;
    (&kernelContext_78)->bodies_0 = bodies_32;
    (&kernelContext_78)->chunks_0 = chunks_32;
    (&kernelContext_78)->sort_0 = sort_32;
    (&kernelContext_78)->atom_0 = atom_32;
    (&kernelContext_78)->manifolds_0 = manifolds_32;
    (&kernelContext_78)->loads_0 = loads_32;
    (&kernelContext_78)->bonds_0 = bonds_33;
    (&kernelContext_78)->args_0 = args_32;
    (&kernelContext_78)->info_0 = info_32;
    (&kernelContext_78)->scan_0 = scan_32;
    (&kernelContext_78)->parts_0 = parts_32;
    (&kernelContext_78)->colliders_0 = colliders_32;
    (&kernelContext_78)->collider_info_0 = collider_info_32;
    (&kernelContext_78)->view_chunks_0 = view_chunks_32;
    (&kernelContext_78)->view_0 = view_32;
    (&kernelContext_78)->objects_0 = objects_32;
    (&kernelContext_78)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_78)->lines_0 = lines_32;
    (&kernelContext_78)->geometry_0 = geometry_32;
    (&kernelContext_78)->shots_0 = shots_32;
    (&kernelContext_78)->crush_0 = crush_32;
    (&kernelContext_78)->crush_rw_0 = crush_rw_32;
    (&kernelContext_78)->crush_tab_0 = crush_tab_32;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_78)->g_scan_0 = &g_scan_32;
    uint b_15 = id_29.x;
    bool _S610;
    if(b_15 >= (params_32->body_count_0))
    {
        _S610 = true;
    }
    else
    {
        _S610 = ((&kernelContext_78)->params_0->shot_count_0) == 0U;
    }
    if(_S610)
    {
        _S610 = true;
    }
    else
    {
        _S610 = ((&kernelContext_78)->crush_0->enabled_0) == 0U;
    }
    if(_S610)
    {
        return;
    }
    uint bill_0 = (&kernelContext_78)->crush_0->r_bill_0 + 4U * b_15;
    packed_float4 _S611 = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_78)->crush_rw_0+bill_0) = _S611;
    uint _S612 = bill_0 + 3U;
    *((&kernelContext_78)->crush_rw_0+_S612) = _S611;
    thread uint shot_5;
    bool _S613 = struck_0(b_15, &shot_5, &kernelContext_78);
    if(!_S613)
    {
        return;
    }
    uint _S614 = a_strike_0(&kernelContext_78);
    uint _S615 = _S614 + 9U * b_15;
    uint k_21 = atomic_load_explicit((&kernelContext_78)->atom_0+(_S615 + 8U), memory_order_relaxed);
    uint _S616 = atomic_load_explicit((&kernelContext_78)->atom_0+(_S615 + 6U), memory_order_relaxed);
    float t_in_3 = (as_type<float>((_S616)));
    if(!(t_in_3 <= ((&kernelContext_78)->params_0->dt_0)))
    {
        _S610 = true;
    }
    else
    {
        _S610 = k_21 == 4294967295U;
    }
    if(_S610)
    {
        return;
    }
    uint _S617 = strike_at_0(b_15, &kernelContext_78);
    float4 _S618 = float4(*((&kernelContext_78)->frames_0+_S617)) ;
    float3 _S619 = (float4(*((&kernelContext_78)->frames_0+(_S617 + 1U))) ).xyz;
    float speed_1 = length(_S619);
    float3 d_6 = _S619 / float3(max(speed_1, 9.999999960041972e-13f)) ;
    float4 _S620 = float4(*((&kernelContext_78)->shots_0+2U * shot_5)) ;
    float radius_0 = _S620.y;
    float _S621 = max(_S620.z, 0.00100000004749745f);
    float m_imp_1 = _S620.w;
    float inv_mass_6 = (float4(*((&kernelContext_78)->bodies_0+7U * b_15)) ).w;
    bool _S622 = inv_mass_6 == 0.0f;
    float m_body_2;
    if(_S622)
    {
        m_body_2 = 0.0f;
    }
    else
    {
        m_body_2 = 1.0f / inv_mass_6;
    }
    float m_eff_1;
    if(_S622)
    {
        m_eff_1 = m_imp_1;
    }
    else
    {
        m_eff_1 = m_imp_1 * m_body_2 / (m_imp_1 + m_body_2);
    }
    float modulus_1 = 1.0f / (1.0f / (float4(*((&kernelContext_78)->shots_0+(2U * shot_5 + 1U))) ).x + 1.0f / (float4(*((&kernelContext_78)->geometry_0+(3U * k_21 + 1U))) ).w);
    float _S623 = max(speed_1, 0.00100000004749745f);
    float tau_4 = 2.86999988555908203f * pow(m_eff_1 * m_eff_1 / (_S621 * modulus_1 * modulus_1 * _S623), 0.20000000298023224f);
    float force_0 = m_eff_1 * speed_1 / tau_4;
    float3 contact_2 = _S618.xyz + _S619 * float3(t_in_3)  + d_6 * float3(radius_0) ;
    float delta_0 = pow(15.0f * m_eff_1 * _S623 * _S623 / (16.0f * modulus_1 * sqrt(_S621)), 0.40000000596046448f);
    float a_h_0 = sqrt(radius_0 * delta_0);
    bool _S624 = crushable_0(k_21, &kernelContext_78);
    float p_cap_0;
    if(_S624)
    {
        p_cap_0 = (float4(*((&kernelContext_78)->crush_tab_0+2U * k_21)) ).x;
    }
    else
    {
        p_cap_0 = 0.0f;
    }
    float a_p_0;
    if(p_cap_0 > 0.0f)
    {
        a_p_0 = sqrt(force_0 / (3.14159274101257324f * p_cap_0));
    }
    else
    {
        a_p_0 = 0.0f;
    }
    float _S625 = min(radius_0, max(a_h_0, a_p_0));
    float _S626 = max(delta_0, radius_0 - sqrt(max(radius_0 * radius_0 - _S625 * _S625, 0.0f)));
    *((&kernelContext_78)->crush_rw_0+(bill_0 + 1U)) = packed_float4(float4(contact_2, 0.0f)) ;
    *((&kernelContext_78)->crush_rw_0+(bill_0 + 2U)) = packed_float4(float4(d_6, _S625)) ;
    *((&kernelContext_78)->crush_rw_0+_S612) = packed_float4(float4(force_0, tau_4, _S626, float(k_21 + 1U))) ;
    uint _S627 = a_cover_0(&kernelContext_78);
    uint _S628 = _S627 + 128U * b_15;
    uint s_8 = 0U;
    for(;;)
    {
        if(s_8 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S629 = _S628 + 2U * s_8;
        atomic_store_explicit((&kernelContext_78)->atom_0+_S629, 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_78)->atom_0+(_S629 + 1U), 4294967295U, memory_order_relaxed);
        s_8 = s_8 + 1U;
    }
    return;
}

bool cover_patch_0(uint k_22, uint thread* b_16, float4 thread* p1_0, float4 thread* p2_0, KernelContext_0 thread* kernelContext_79)
{
    uint _S630 = body_of_0(k_22, kernelContext_79);
    *b_16 = _S630;
    float4 _S631 = float4(0.0f) ;
    *p1_0 = _S631;
    *p2_0 = _S631;
    thread uint shot_6;
    bool _S632 = struck_0(*b_16, &shot_6, kernelContext_79);
    bool _S633;
    if(!_S632)
    {
        _S633 = true;
    }
    else
    {
        uint _S634 = crush_flags_0(k_22, kernelContext_79);
        _S633 = (_S634 & 1U) != 0U;
    }
    if(_S633)
    {
        return false;
    }
    uint bill_1 = kernelContext_79->crush_0->r_bill_0 + 4U * *b_16;
    if(((float4(*(kernelContext_79->crush_rw_0+(bill_1 + 3U))) ).w) == 0.0f)
    {
        return false;
    }
    *p1_0 = float4((float4(*(kernelContext_79->crush_rw_0+(bill_1 + 1U))) ).xyz, (float4(*(kernelContext_79->shots_0+2U * shot_6)) ).y);
    *p2_0 = float4(*(kernelContext_79->crush_rw_0+(bill_1 + 2U))) ;
    return true;
}

float3 cover_offset_0(uint s_9, float3 d_7, float a_6, float thread* rho_0)
{
    uint ring_0 = s_9 / 8U;
    uint sector_0 = s_9 % 8U;
    *rho_0 = a_6 * sqrt((float(ring_0) + 0.5f) / 8.0f);
    float theta_1 = 6.28318548202514648f * (float(sector_0) + 0.5f * float(ring_0 & 1U)) / 8.0f;
    float3 e_2;
    if((abs(d_7.x)) < 0.57735002040863037f)
    {
        e_2 = float3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_7.y)) < 0.57735002040863037f)
        {
            e_2 = float3(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_2 = float3(0.0f, 0.0f, 1.0f);
        }
    }
    float3 u_2 = normalize(cross(d_7, e_2));
    return float3(*rho_0)  * (float3(cos(theta_1))  * u_2 + float3(sin(theta_1))  * cross(d_7, u_2));
}

float cover_depth_0(uint k_23, uint s_10, float3 contact_3, float3 d_8, float a_7, float radius_1, KernelContext_0 thread* kernelContext_80)
{
    thread float rho_1;
    float _S635 = 2.0f * radius_1;
    float3 _S636 = contact_3 + cover_offset_0(s_10, d_8, a_7, &rho_1) - d_8 * float3(_S635) ;
    thread Quat_0 q_25;
    float3 _S637 = chunk_box_world_0(k_23, &q_25, kernelContext_80);
    uint _S638 = 3U * k_23;
    Quat_0 _S639 = quat_of_0(float4(*(kernelContext_80->geometry_0+(_S638 + 2U))) );
    thread Quat_0 _S640 = q_25;
    thread Quat_0 _S641 = _S639;
    Quat_0 _S642 = quat_mul_0(&_S640, &_S641);
    thread Quat_0 _S643 = _S642;
    Quat_0 _S644 = normalized_0(&_S643);
    float3 _S645 = (float4(*(kernelContext_80->geometry_0+(_S638 + 1U))) ).xyz;
    float t0_2 = -1.00000001504746622e+30f;
    float t1_1 = 1.00000001504746622e+30f;
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
        thread Quat_0 _S646 = _S644;
        float3 _S647 = rotate_0(&_S646, e_3);
        float o_5 = dot(_S636 - _S637, _S647);
        float dv_1 = dot(d_8, _S647);
        if((abs(dv_1)) < 9.99999971718068537e-10f)
        {
            if((abs(o_5)) > (_S645[ax_3]))
            {
                return 1.00000001504746622e+30f;
            }
            ax_3 = ax_3 + 1U;
            continue;
        }
        float ta_2 = (- _S645[ax_3] - o_5) / dv_1;
        float tb_2 = (_S645[ax_3] - o_5) / dv_1;
        float _S648 = min(t1_1, max(ta_2, tb_2));
        t0_2 = max(t0_2, min(ta_2, tb_2));
        t1_1 = _S648;
        ax_3 = ax_3 + 1U;
    }
    bool _S649;
    if(!(t0_2 <= t1_1))
    {
        _S649 = true;
    }
    else
    {
        _S649 = t1_1 < 0.0f;
    }
    if(_S649)
    {
        return 1.00000001504746622e+30f;
    }
    return t0_2 - _S635;
}

[[kernel]] void fr_strike_cover(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], atomic<uint> device* atom_33 [[buffer(5)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], CrushParams_0 constant* crush_33 [[buffer(25)]], packed_float4 device* crush_rw_33 [[buffer(24)]], packed_float4 device* crush_tab_33 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_81;
    (&kernelContext_81)->params_0 = params_33;
    (&kernelContext_81)->work_0 = work_33;
    (&kernelContext_81)->frames_0 = frames_33;
    (&kernelContext_81)->bodies_0 = bodies_33;
    (&kernelContext_81)->chunks_0 = chunks_33;
    (&kernelContext_81)->sort_0 = sort_33;
    (&kernelContext_81)->atom_0 = atom_33;
    (&kernelContext_81)->manifolds_0 = manifolds_33;
    (&kernelContext_81)->loads_0 = loads_33;
    (&kernelContext_81)->bonds_0 = bonds_34;
    (&kernelContext_81)->args_0 = args_33;
    (&kernelContext_81)->info_0 = info_33;
    (&kernelContext_81)->scan_0 = scan_33;
    (&kernelContext_81)->parts_0 = parts_33;
    (&kernelContext_81)->colliders_0 = colliders_33;
    (&kernelContext_81)->collider_info_0 = collider_info_33;
    (&kernelContext_81)->view_chunks_0 = view_chunks_33;
    (&kernelContext_81)->view_0 = view_33;
    (&kernelContext_81)->objects_0 = objects_33;
    (&kernelContext_81)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_81)->lines_0 = lines_33;
    (&kernelContext_81)->geometry_0 = geometry_33;
    (&kernelContext_81)->shots_0 = shots_33;
    (&kernelContext_81)->crush_0 = crush_33;
    (&kernelContext_81)->crush_rw_0 = crush_rw_33;
    (&kernelContext_81)->crush_tab_0 = crush_tab_33;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_81)->g_scan_0 = &g_scan_33;
    uint k_24 = id_30.x;
    bool _S650;
    if(k_24 >= (params_33->chunk_count_0))
    {
        _S650 = true;
    }
    else
    {
        _S650 = ((&kernelContext_81)->params_0->shot_count_0) == 0U;
    }
    if(_S650)
    {
        _S650 = true;
    }
    else
    {
        _S650 = ((&kernelContext_81)->crush_0->enabled_0) == 0U;
    }
    if(_S650)
    {
        return;
    }
    thread uint b_17;
    thread float4 p1_1;
    thread float4 p2_1;
    bool _S651 = cover_patch_0(k_24, &b_17, &p1_1, &p2_1, &kernelContext_81);
    if(!_S651)
    {
        return;
    }
    uint _S652 = a_cover_0(&kernelContext_81);
    uint _S653 = _S652 + 128U * b_17;
    uint s_11 = 0U;
    for(;;)
    {
        if(s_11 < 64U)
        {
        }
        else
        {
            break;
        }
        float _S654 = cover_depth_0(k_24, s_11, p1_1.xyz, p2_1.xyz, p2_1.w, p1_1.w, &kernelContext_81);
        if(_S654 < 5.0000000752373311e+29f)
        {
            uint _S655 = atomic_fetch_min_explicit((&kernelContext_81)->atom_0+(_S653 + 2U * s_11), ordered_0(_S654), memory_order_relaxed);
        }
        s_11 = s_11 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_claim(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], atomic<uint> device* atom_34 [[buffer(5)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], CrushParams_0 constant* crush_34 [[buffer(25)]], packed_float4 device* crush_rw_34 [[buffer(24)]], packed_float4 device* crush_tab_34 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_82;
    (&kernelContext_82)->params_0 = params_34;
    (&kernelContext_82)->work_0 = work_34;
    (&kernelContext_82)->frames_0 = frames_34;
    (&kernelContext_82)->bodies_0 = bodies_34;
    (&kernelContext_82)->chunks_0 = chunks_34;
    (&kernelContext_82)->sort_0 = sort_34;
    (&kernelContext_82)->atom_0 = atom_34;
    (&kernelContext_82)->manifolds_0 = manifolds_34;
    (&kernelContext_82)->loads_0 = loads_34;
    (&kernelContext_82)->bonds_0 = bonds_35;
    (&kernelContext_82)->args_0 = args_34;
    (&kernelContext_82)->info_0 = info_34;
    (&kernelContext_82)->scan_0 = scan_34;
    (&kernelContext_82)->parts_0 = parts_34;
    (&kernelContext_82)->colliders_0 = colliders_34;
    (&kernelContext_82)->collider_info_0 = collider_info_34;
    (&kernelContext_82)->view_chunks_0 = view_chunks_34;
    (&kernelContext_82)->view_0 = view_34;
    (&kernelContext_82)->objects_0 = objects_34;
    (&kernelContext_82)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_82)->lines_0 = lines_34;
    (&kernelContext_82)->geometry_0 = geometry_34;
    (&kernelContext_82)->shots_0 = shots_34;
    (&kernelContext_82)->crush_0 = crush_34;
    (&kernelContext_82)->crush_rw_0 = crush_rw_34;
    (&kernelContext_82)->crush_tab_0 = crush_tab_34;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_82)->g_scan_0 = &g_scan_34;
    uint k_25 = id_31.x;
    bool _S656;
    if(k_25 >= (params_34->chunk_count_0))
    {
        _S656 = true;
    }
    else
    {
        _S656 = ((&kernelContext_82)->params_0->shot_count_0) == 0U;
    }
    if(_S656)
    {
        _S656 = true;
    }
    else
    {
        _S656 = ((&kernelContext_82)->crush_0->enabled_0) == 0U;
    }
    if(_S656)
    {
        return;
    }
    thread uint b_18;
    thread float4 p1_2;
    thread float4 p2_2;
    bool _S657 = cover_patch_0(k_25, &b_18, &p1_2, &p2_2, &kernelContext_82);
    if(!_S657)
    {
        return;
    }
    uint _S658 = a_cover_0(&kernelContext_82);
    uint _S659 = _S658 + 128U * b_18;
    uint s_12 = 0U;
    for(;;)
    {
        if(s_12 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S660 = _S659 + 2U * s_12;
        uint first_4 = atomic_load_explicit((&kernelContext_82)->atom_0+_S660, memory_order_relaxed);
        if(first_4 == 4294967295U)
        {
            s_12 = s_12 + 1U;
            continue;
        }
        float _S661 = cover_depth_0(k_25, s_12, p1_2.xyz, p2_2.xyz, p2_2.w, p1_2.w, &kernelContext_82);
        if(_S661 < 5.0000000752373311e+29f)
        {
            _S656 = _S661 <= (unordered_0(first_4) + 0.00009999999747379f);
        }
        else
        {
            _S656 = false;
        }
        if(_S656)
        {
            uint _S662 = atomic_fetch_min_explicit((&kernelContext_82)->atom_0+(_S660 + 1U), k_25, memory_order_relaxed);
        }
        s_12 = s_12 + 1U;
    }
    return;
}

float strike_load_0(uint k_26, uint slot_6, float3 f_12, float3 at_11, float tau_5, float budget_0, KernelContext_0 thread* kernelContext_83)
{
    bool _S663 = crushable_0(k_26, kernelContext_83);
    bool _S664;
    if(!_S663)
    {
        _S664 = true;
    }
    else
    {
        uint _S665 = crush_flags_0(k_26, kernelContext_83);
        _S664 = (_S665 & 1U) != 0U;
    }
    if(_S664)
    {
        _S664 = true;
    }
    else
    {
        _S664 = !(budget_0 > 0.0f);
    }
    if(_S664)
    {
        return 0.0f;
    }
    thread Quat_0 q_26;
    thread float3 origin_10;
    structure_pose_0(slot_6, 7U * slot_6, &q_26, &origin_10, kernelContext_83);
    float3 _S666 = (float4(*(kernelContext_83->chunks_0+4U * k_26)) ).xyz;
    thread Quat_0 _S667 = q_26;
    float3 _S668 = rotate_0(&_S667, _S666);
    float3 centre_5 = origin_10 + _S668;
    thread Quat_0 _S669 = q_26;
    float4 _S670 = bond_centroid_0(k_26, &_S669, origin_10, centre_5, kernelContext_83);
    if(!((_S670.w) > 0.0f))
    {
        return 0.0f;
    }
    Sym_0 _S671 = sym_outer_0(_S670.xyz, - f_12);
    thread Sym_0 _S672 = sym_outer_0(at_11 - centre_5, f_12);
    thread Sym_0 _S673 = _S671;
    Sym_0 _S674 = sym_add_0(&_S672, &_S673);
    float before_0 = (float4(*(kernelContext_83->crush_rw_0+k_26)) ).x;
    float _S675 = min(tau_5, kernelContext_83->params_0->dt_0);
    thread Sym_0 _S676 = _S674;
    thread Sym_0 _S677 = _S674;
    float4 _S678 = crush_two_0(&_S676, _S675, &_S677, 0.0f, k_26, before_0, kernelContext_83);
    uint _S679 = 2U * k_26;
    float per_0 = (float4(*(kernelContext_83->crush_tab_0+_S679)) ).w * (float4(*(kernelContext_83->crush_tab_0+(_S679 + 1U))) ).y;
    float energy_0 = (_S678.x - before_0) * per_0;
    float energy_1;
    float4 step_6;
    if(energy_0 > budget_0)
    {
        float d_9 = before_0 + budget_0 / per_0;
        float _S680 = min(d_9, 1.0f);
        float _S681 = _S678.y;
        float _S682 = _S678.z;
        if(d_9 >= 1.0f)
        {
            energy_1 = 1.0f;
        }
        else
        {
            energy_1 = 0.0f;
        }
        step_6 = float4(_S680, _S681, _S682, energy_1);
        energy_1 = budget_0;
    }
    else
    {
        step_6 = _S678;
        energy_1 = energy_0;
    }
    crush_record_0(k_26, step_6, 2U, kernelContext_83);
    return energy_1;
}

bool cell_in_0(uint2 mask_0, uint c_15)
{
    return (((mask_0[c_15 / 32U]) >> (c_15 % 32U)) & 1U) != 0U;
}

float strike_crush_body_0(uint b_19, uint s_13, float budget_1, KernelContext_0 thread* kernelContext_84)
{
    uint total_1;
    bool _S683;
    uint bill_2 = kernelContext_84->crush_0->r_bill_0 + 4U * b_19;
    uint _S684 = bill_2 + 1U;
    float4 _S685 = float4(*(kernelContext_84->crush_rw_0+(bill_2 + 2U))) ;
    float4 _S686 = float4(*(kernelContext_84->crush_rw_0+(bill_2 + 3U))) ;
    float3 contact_4 = (float4(*(kernelContext_84->crush_rw_0+_S684)) ).xyz;
    float3 d_10 = _S685.xyz;
    float _S687 = _S685.w;
    float _S688 = (float4(*(kernelContext_84->shots_0+2U * s_13)) ).y;
    uint _S689 = a_cover_0(kernelContext_84);
    uint _S690 = _S689 + 128U * b_19;
    float first_5 = 1.00000001504746622e+30f;
    uint c_16 = 0U;
    for(;;)
    {
        if(c_16 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S691 = _S690 + 2U * c_16;
        uint _S692 = atomic_load_explicit(kernelContext_84->atom_0+(_S691 + 1U), memory_order_relaxed);
        if(_S692 == 4294967295U)
        {
            c_16 = c_16 + 1U;
            continue;
        }
        thread float rho_2;
        float3 _S693 = cover_offset_0(c_16, d_10, _S687, &rho_2);
        uint _S694 = atomic_load_explicit(kernelContext_84->atom_0+_S691, memory_order_relaxed);
        first_5 = min(first_5, unordered_0(_S694) + _S688 - sqrt(max(_S688 * _S688 - rho_2 * rho_2, 0.0f)));
        c_16 = c_16 + 1U;
    }
    thread uint2 counted_0 = uint2(0U) ;
    c_16 = 0U;
    uint total_2 = 0U;
    for(;;)
    {
        if(c_16 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S695 = _S690 + 2U * c_16;
        uint _S696 = atomic_load_explicit(kernelContext_84->atom_0+(_S695 + 1U), memory_order_relaxed);
        if(_S696 == 4294967295U)
        {
            c_16 = c_16 + 1U;
            continue;
        }
        thread float rho_3;
        float3 _S697 = cover_offset_0(c_16, d_10, _S687, &rho_3);
        uint _S698 = atomic_load_explicit(kernelContext_84->atom_0+_S695, memory_order_relaxed);
        if((unordered_0(_S698) + _S688 - sqrt(max(_S688 * _S688 - rho_3 * rho_3, 0.0f)) - first_5) <= (_S686.z))
        {
            counted_0[c_16 / 32U] = (counted_0[c_16 / 32U]) | (1U << (c_16 % 32U));
            total_1 = total_2 + 1U;
        }
        else
        {
            total_1 = total_2;
        }
        total_2 = total_1;
        c_16 = c_16 + 1U;
    }
    float3 _S699 = float3(0.0f) ;
    float energy_2;
    float3 point_1;
    if(total_2 == 0U)
    {
        float _S700 = strike_load_0(uint(_S686.w) - 1U, b_19, d_10 * float3(_S686.x) , contact_4, _S686.y, budget_1, kernelContext_84);
        float3 _S701 = float3(_S700)  * contact_4;
        energy_2 = _S700;
        point_1 = _S701;
    }
    else
    {
        energy_2 = 0.0f;
        point_1 = _S699;
    }
    c_16 = 0U;
    for(;;)
    {
        if(c_16 < 64U)
        {
        }
        else
        {
            break;
        }
        if(!cell_in_0(counted_0, c_16))
        {
            c_16 = c_16 + 1U;
            continue;
        }
        uint k_27 = atomic_load_explicit(kernelContext_84->atom_0+(_S690 + 2U * c_16 + 1U), memory_order_relaxed);
        bool seen_0 = false;
        total_1 = 0U;
        for(;;)
        {
            if(total_1 < c_16)
            {
            }
            else
            {
                break;
            }
            if(seen_0)
            {
                seen_0 = true;
            }
            else
            {
                if(cell_in_0(counted_0, total_1))
                {
                    uint _S702 = atomic_load_explicit(kernelContext_84->atom_0+(_S690 + 2U * total_1 + 1U), memory_order_relaxed);
                    _S683 = _S702 == k_27;
                }
                else
                {
                    _S683 = false;
                }
                seen_0 = _S683;
            }
            total_1 = total_1 + 1U;
        }
        if(seen_0)
        {
            c_16 = c_16 + 1U;
            continue;
        }
        uint e_4 = c_16;
        float3 at_12 = _S699;
        uint n_3 = 0U;
        for(;;)
        {
            if(e_4 < 64U)
            {
            }
            else
            {
                break;
            }
            if(!cell_in_0(counted_0, e_4))
            {
                _S683 = true;
            }
            else
            {
                uint _S703 = atomic_load_explicit(kernelContext_84->atom_0+(_S690 + 2U * e_4 + 1U), memory_order_relaxed);
                _S683 = _S703 != k_27;
            }
            if(_S683)
            {
                e_4 = e_4 + 1U;
                continue;
            }
            thread float rho_4;
            float3 _S704 = contact_4 + cover_offset_0(e_4, d_10, _S687, &rho_4);
            uint _S705 = atomic_load_explicit(kernelContext_84->atom_0+(_S690 + 2U * e_4), memory_order_relaxed);
            uint _S706 = n_3 + 1U;
            at_12 = at_12 + (_S704 + d_10 * float3(unordered_0(_S705)) );
            n_3 = _S706;
            e_4 = e_4 + 1U;
        }
        float _S707 = float(n_3);
        float3 at_13 = at_12 / float3(_S707) ;
        float _S708 = strike_load_0(k_27, b_19, d_10 * float3((_S686.x * _S707 / float(total_2))) , at_13, _S686.y, budget_1 - energy_2, kernelContext_84);
        float3 point_2 = point_1 + float3(_S708)  * at_13;
        energy_2 = energy_2 + _S708;
        point_1 = point_2;
        c_16 = c_16 + 1U;
    }
    *(kernelContext_84->crush_rw_0+bill_2) = packed_float4(float4(energy_2, float(s_13 + 1U), 0.0f, 0.0f)) ;
    if(energy_2 > 0.0f)
    {
        *(kernelContext_84->crush_rw_0+_S684) = packed_float4(float4(point_1 / float3(energy_2) , 0.0f)) ;
    }
    return energy_2;
}

[[kernel]] void fr_strike_crush(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], atomic<uint> device* atom_35 [[buffer(5)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], CrushParams_0 constant* crush_35 [[buffer(25)]], packed_float4 device* crush_rw_35 [[buffer(24)]], packed_float4 device* crush_tab_35 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_85;
    (&kernelContext_85)->params_0 = params_35;
    (&kernelContext_85)->work_0 = work_35;
    (&kernelContext_85)->frames_0 = frames_35;
    (&kernelContext_85)->bodies_0 = bodies_35;
    (&kernelContext_85)->chunks_0 = chunks_35;
    (&kernelContext_85)->sort_0 = sort_35;
    (&kernelContext_85)->atom_0 = atom_35;
    (&kernelContext_85)->manifolds_0 = manifolds_35;
    (&kernelContext_85)->loads_0 = loads_35;
    (&kernelContext_85)->bonds_0 = bonds_36;
    (&kernelContext_85)->args_0 = args_35;
    (&kernelContext_85)->info_0 = info_35;
    (&kernelContext_85)->scan_0 = scan_35;
    (&kernelContext_85)->parts_0 = parts_35;
    (&kernelContext_85)->colliders_0 = colliders_35;
    (&kernelContext_85)->collider_info_0 = collider_info_35;
    (&kernelContext_85)->view_chunks_0 = view_chunks_35;
    (&kernelContext_85)->view_0 = view_35;
    (&kernelContext_85)->objects_0 = objects_35;
    (&kernelContext_85)->shot_bodies_0 = shot_bodies_35;
    (&kernelContext_85)->lines_0 = lines_35;
    (&kernelContext_85)->geometry_0 = geometry_35;
    (&kernelContext_85)->shots_0 = shots_35;
    (&kernelContext_85)->crush_0 = crush_35;
    (&kernelContext_85)->crush_rw_0 = crush_rw_35;
    (&kernelContext_85)->crush_tab_0 = crush_tab_35;
    threadgroup array<uint, int(1024)> g_scan_35;
    (&kernelContext_85)->g_scan_0 = &g_scan_35;
    uint s_14 = id_32.x;
    bool _S709;
    if(s_14 >= (params_35->shot_count_0))
    {
        _S709 = true;
    }
    else
    {
        _S709 = ((&kernelContext_85)->crush_0->enabled_0) == 0U;
    }
    if(_S709)
    {
        return;
    }
    uint sb_3 = 7U * (as_type<uint>(((float4(*((&kernelContext_85)->shots_0+2U * s_14)) ).x)));
    float inv_mass_7 = (float4(*((&kernelContext_85)->bodies_0+sb_3)) ).w;
    float3 v_10 = (float4(*((&kernelContext_85)->bodies_0+(sb_3 + 3U))) ).xyz;
    float budget_2;
    if(inv_mass_7 > 0.0f)
    {
        budget_2 = 0.5f * dot(v_10, v_10) / inv_mass_7;
    }
    else
    {
        budget_2 = 1.00000001504746622e+30f;
    }
    uint b_20 = 0U;
    for(;;)
    {
        if(b_20 < ((&kernelContext_85)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        thread uint shot_7;
        bool _S710 = struck_0(b_20, &shot_7, &kernelContext_85);
        if(!_S710)
        {
            _S709 = true;
        }
        else
        {
            _S709 = shot_7 != s_14;
        }
        bool _S711;
        if(_S709)
        {
            _S711 = true;
        }
        else
        {
            _S711 = ((float4(*((&kernelContext_85)->crush_rw_0+((&kernelContext_85)->crush_0->r_bill_0 + 4U * b_20 + 3U))) ).w) == 0.0f;
        }
        if(_S711)
        {
            b_20 = b_20 + 1U;
            continue;
        }
        float _S712 = strike_crush_body_0(b_20, s_14, budget_2, &kernelContext_85);
        budget_2 = budget_2 - _S712;
        b_20 = b_20 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bill(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], atomic<uint> device* atom_36 [[buffer(5)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], CrushParams_0 constant* crush_36 [[buffer(25)]], packed_float4 device* crush_rw_36 [[buffer(24)]], packed_float4 device* crush_tab_36 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_86;
    (&kernelContext_86)->params_0 = params_36;
    (&kernelContext_86)->work_0 = work_36;
    (&kernelContext_86)->frames_0 = frames_36;
    (&kernelContext_86)->bodies_0 = bodies_36;
    (&kernelContext_86)->chunks_0 = chunks_36;
    (&kernelContext_86)->sort_0 = sort_36;
    (&kernelContext_86)->atom_0 = atom_36;
    (&kernelContext_86)->manifolds_0 = manifolds_36;
    (&kernelContext_86)->loads_0 = loads_36;
    (&kernelContext_86)->bonds_0 = bonds_37;
    (&kernelContext_86)->args_0 = args_36;
    (&kernelContext_86)->info_0 = info_36;
    (&kernelContext_86)->scan_0 = scan_36;
    (&kernelContext_86)->parts_0 = parts_36;
    (&kernelContext_86)->colliders_0 = colliders_36;
    (&kernelContext_86)->collider_info_0 = collider_info_36;
    (&kernelContext_86)->view_chunks_0 = view_chunks_36;
    (&kernelContext_86)->view_0 = view_36;
    (&kernelContext_86)->objects_0 = objects_36;
    (&kernelContext_86)->shot_bodies_0 = shot_bodies_36;
    (&kernelContext_86)->lines_0 = lines_36;
    (&kernelContext_86)->geometry_0 = geometry_36;
    (&kernelContext_86)->shots_0 = shots_36;
    (&kernelContext_86)->crush_0 = crush_36;
    (&kernelContext_86)->crush_rw_0 = crush_rw_36;
    (&kernelContext_86)->crush_tab_0 = crush_tab_36;
    threadgroup array<uint, int(1024)> g_scan_36;
    (&kernelContext_86)->g_scan_0 = &g_scan_36;
    uint s_15 = id_33.x;
    bool _S713;
    if(s_15 >= (params_36->shot_count_0))
    {
        _S713 = true;
    }
    else
    {
        _S713 = ((&kernelContext_86)->crush_0->enabled_0) == 0U;
    }
    if(_S713)
    {
        return;
    }
    uint b_21 = 0U;
    float energy_3 = 0.0f;
    for(;;)
    {
        if(b_21 < ((&kernelContext_86)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S714 = float4(*((&kernelContext_86)->crush_rw_0+((&kernelContext_86)->crush_0->r_bill_0 + 4U * b_21))) ;
        if((_S714.y) == float(s_15 + 1U))
        {
            energy_3 = energy_3 + _S714.x;
        }
        b_21 = b_21 + 1U;
    }
    *((&kernelContext_86)->crush_rw_0+((&kernelContext_86)->crush_0->r_shot_0 + s_15)) = packed_float4(float4(0.0f) ) ;
    if(!(energy_3 > 0.0f))
    {
        return;
    }
    uint sb_4 = 7U * (as_type<uint>(((float4(*((&kernelContext_86)->shots_0+2U * s_15)) ).x)));
    float inv_mass_8 = (float4(*((&kernelContext_86)->bodies_0+sb_4)) ).w;
    if(!(inv_mass_8 > 0.0f))
    {
        return;
    }
    float m_6 = 1.0f / inv_mass_8;
    float3 v_11 = (float4(*((&kernelContext_86)->bodies_0+(sb_4 + 3U))) ).xyz;
    float speed_2 = length(v_11);
    float j_7 = m_6 * (speed_2 - sqrt(max(speed_2 * speed_2 - 2.0f * energy_3 / m_6, 0.0f)));
    bool _S715 = speed_2 > 0.0f;
    float3 _S716;
    if(_S715)
    {
        _S716 = - v_11 * float3((j_7 / (m_6 * speed_2))) ;
    }
    else
    {
        _S716 = float3(0.0f) ;
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
        uint w_8 = sb_4 + region_1 * 7U * (&kernelContext_86)->params_0->max_bodies_0 + 3U;
        *((&kernelContext_86)->bodies_0+w_8) = packed_float4(float4((float4(*((&kernelContext_86)->bodies_0+w_8)) ).xyz + _S716, (float4(*((&kernelContext_86)->bodies_0+w_8)) ).w)) ;
        region_1 = region_1 + 1U;
    }
    packed_float4 device* _S717 = (&kernelContext_86)->crush_rw_0+((&kernelContext_86)->crush_0->r_shot_0 + s_15);
    if(_S715)
    {
        _S716 = v_11 * float3((j_7 / speed_2)) ;
    }
    else
    {
        _S716 = float3(0.0f) ;
    }
    *_S717 = packed_float4(float4(_S716, energy_3)) ;
    return;
}

float3 inverse_rotate_0(const Quat_0 thread* q_27, float3 v_12)
{
    thread Quat_0 c_17;
    (&c_17)->w_0 = q_27->w_0;
    (&c_17)->x_0 = - q_27->x_0;
    (&c_17)->y_0 = - q_27->y_0;
    (&c_17)->z_0 = - q_27->z_0;
    thread Quat_0 _S718 = c_17;
    float3 _S719 = rotate_0(&_S718, v_12);
    return _S719;
}

[[kernel]] void fr_crush_push(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], atomic<uint> device* atom_37 [[buffer(5)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], CrushParams_0 constant* crush_37 [[buffer(25)]], packed_float4 device* crush_rw_37 [[buffer(24)]], packed_float4 device* crush_tab_37 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_87;
    (&kernelContext_87)->params_0 = params_37;
    (&kernelContext_87)->work_0 = work_37;
    (&kernelContext_87)->frames_0 = frames_37;
    (&kernelContext_87)->bodies_0 = bodies_37;
    (&kernelContext_87)->chunks_0 = chunks_37;
    (&kernelContext_87)->sort_0 = sort_37;
    (&kernelContext_87)->atom_0 = atom_37;
    (&kernelContext_87)->manifolds_0 = manifolds_37;
    (&kernelContext_87)->loads_0 = loads_37;
    (&kernelContext_87)->bonds_0 = bonds_38;
    (&kernelContext_87)->args_0 = args_37;
    (&kernelContext_87)->info_0 = info_37;
    (&kernelContext_87)->scan_0 = scan_37;
    (&kernelContext_87)->parts_0 = parts_37;
    (&kernelContext_87)->colliders_0 = colliders_37;
    (&kernelContext_87)->collider_info_0 = collider_info_37;
    (&kernelContext_87)->view_chunks_0 = view_chunks_37;
    (&kernelContext_87)->view_0 = view_37;
    (&kernelContext_87)->objects_0 = objects_37;
    (&kernelContext_87)->shot_bodies_0 = shot_bodies_37;
    (&kernelContext_87)->lines_0 = lines_37;
    (&kernelContext_87)->geometry_0 = geometry_37;
    (&kernelContext_87)->shots_0 = shots_37;
    (&kernelContext_87)->crush_0 = crush_37;
    (&kernelContext_87)->crush_rw_0 = crush_rw_37;
    (&kernelContext_87)->crush_tab_0 = crush_tab_37;
    threadgroup array<uint, int(1024)> g_scan_37;
    (&kernelContext_87)->g_scan_0 = &g_scan_37;
    uint b_22 = id_34.x;
    bool _S720;
    if(b_22 >= (params_37->body_count_0))
    {
        _S720 = true;
    }
    else
    {
        _S720 = ((&kernelContext_87)->params_0->shot_count_0) == 0U;
    }
    if(_S720)
    {
        _S720 = true;
    }
    else
    {
        _S720 = ((&kernelContext_87)->crush_0->enabled_0) == 0U;
    }
    if(_S720)
    {
        return;
    }
    uint _S721 = 4U * b_22;
    float4 _S722 = float4(*((&kernelContext_87)->crush_rw_0+((&kernelContext_87)->crush_0->r_bill_0 + _S721))) ;
    float _S723 = _S722.x;
    if(!(_S723 > 0.0f))
    {
        return;
    }
    float4 _S724 = float4(*((&kernelContext_87)->crush_rw_0+((&kernelContext_87)->crush_0->r_shot_0 + (uint(_S722.y) - 1U)))) ;
    uint bw_0 = 7U * b_22;
    float inv_mass_9 = (float4(*((&kernelContext_87)->bodies_0+bw_0)) ).w;
    float _S725 = _S724.w;
    if(!(_S725 > 0.0f))
    {
        _S720 = true;
    }
    else
    {
        _S720 = inv_mass_9 == 0.0f;
    }
    if(_S720)
    {
        return;
    }
    float3 j_8 = _S724.xyz * float3((_S723 / _S725)) ;
    Quat_0 o_6 = quat_of_0(float4(*((&kernelContext_87)->bodies_0+(bw_0 + 1U))) );
    float3 inv_i_0 = (float4(*((&kernelContext_87)->bodies_0+(bw_0 + 2U))) ).xyz;
    float3 _S726 = cross(to_rubble_0((float4(*((&kernelContext_87)->crush_rw_0+((&kernelContext_87)->crush_0->r_bill_0 + _S721 + 1U))) ).xyz) - (float4(*((&kernelContext_87)->bodies_0+bw_0)) ).xyz, j_8);
    thread Quat_0 _S727 = o_6;
    float3 _S728 = inverse_rotate_0(&_S727, _S726);
    float3 _S729 = inv_i_0 * _S728;
    thread Quat_0 _S730 = o_6;
    float3 _S731 = rotate_0(&_S730, _S729);
    uint region_2 = 0U;
    for(;;)
    {
        if(region_2 < 2U)
        {
        }
        else
        {
            break;
        }
        uint w_9 = bw_0 + region_2 * 7U * (&kernelContext_87)->params_0->max_bodies_0;
        uint _S732 = w_9 + 3U;
        *((&kernelContext_87)->bodies_0+_S732) = packed_float4(float4((float4(*((&kernelContext_87)->bodies_0+_S732)) ).xyz + j_8 * float3(inv_mass_9) , (float4(*((&kernelContext_87)->bodies_0+_S732)) ).w)) ;
        uint _S733 = w_9 + 4U;
        *((&kernelContext_87)->bodies_0+_S733) = packed_float4(float4((float4(*((&kernelContext_87)->bodies_0+_S733)) ).xyz + _S731, (float4(*((&kernelContext_87)->bodies_0+_S733)) ).w)) ;
        region_2 = region_2 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bonds(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], atomic<uint> device* atom_38 [[buffer(5)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], CrushParams_0 constant* crush_38 [[buffer(25)]], packed_float4 device* crush_rw_38 [[buffer(24)]], packed_float4 device* crush_tab_38 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_88;
    (&kernelContext_88)->params_0 = params_38;
    (&kernelContext_88)->work_0 = work_38;
    (&kernelContext_88)->frames_0 = frames_38;
    (&kernelContext_88)->bodies_0 = bodies_38;
    (&kernelContext_88)->chunks_0 = chunks_38;
    (&kernelContext_88)->sort_0 = sort_38;
    (&kernelContext_88)->atom_0 = atom_38;
    (&kernelContext_88)->manifolds_0 = manifolds_38;
    (&kernelContext_88)->loads_0 = loads_38;
    (&kernelContext_88)->bonds_0 = bonds_39;
    (&kernelContext_88)->args_0 = args_38;
    (&kernelContext_88)->info_0 = info_38;
    (&kernelContext_88)->scan_0 = scan_38;
    (&kernelContext_88)->parts_0 = parts_38;
    (&kernelContext_88)->colliders_0 = colliders_38;
    (&kernelContext_88)->collider_info_0 = collider_info_38;
    (&kernelContext_88)->view_chunks_0 = view_chunks_38;
    (&kernelContext_88)->view_0 = view_38;
    (&kernelContext_88)->objects_0 = objects_38;
    (&kernelContext_88)->shot_bodies_0 = shot_bodies_38;
    (&kernelContext_88)->lines_0 = lines_38;
    (&kernelContext_88)->geometry_0 = geometry_38;
    (&kernelContext_88)->shots_0 = shots_38;
    (&kernelContext_88)->crush_0 = crush_38;
    (&kernelContext_88)->crush_rw_0 = crush_rw_38;
    (&kernelContext_88)->crush_tab_0 = crush_tab_38;
    threadgroup array<uint, int(1024)> g_scan_38;
    (&kernelContext_88)->g_scan_0 = &g_scan_38;
    uint i_8 = id_35.x;
    bool _S734;
    if(i_8 >= (params_38->bond_count_0))
    {
        _S734 = true;
    }
    else
    {
        _S734 = ((&kernelContext_88)->crush_0->enabled_0) == 0U;
    }
    if(_S734)
    {
        _S734 = true;
    }
    else
    {
        uint _S735 = w_alive_0(&kernelContext_88);
        _S734 = (*((&kernelContext_88)->work_0+(_S735 + i_8))) == 0U;
    }
    if(_S734)
    {
        return;
    }
    uint4 _S736 = uint4(*((&kernelContext_88)->bonds_0+i_8)) ;
    uint _S737 = _S736.x;
    if((((as_type<uint>(((float4(*((&kernelContext_88)->crush_rw_0+_S737)) ).w))) | (as_type<uint>(((float4(*((&kernelContext_88)->crush_rw_0+_S736.y)) ).w)))) & 2U) == 0U)
    {
        return;
    }
    uint _S738 = w_alive_0(&kernelContext_88);
    *((&kernelContext_88)->work_0+(_S738 + i_8)) = 0U;
    uint _S739 = a_dirty_0(&kernelContext_88);
    uint _S740 = body_of_0(_S737, &kernelContext_88);
    uint _S741 = atomic_fetch_or_explicit((&kernelContext_88)->atom_0+(_S739 + _S740), 1U, memory_order_relaxed);
    uint _S742 = a_counters_0(&kernelContext_88);
    uint _S743 = atomic_fetch_add_explicit((&kernelContext_88)->atom_0+_S742, 1U, memory_order_relaxed);
    uint _S744 = atomic_fetch_add_explicit((&kernelContext_88)->atom_0+(_S742 + 1U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_crush_mark(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], atomic<uint> device* atom_39 [[buffer(5)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], CrushParams_0 constant* crush_39 [[buffer(25)]], packed_float4 device* crush_rw_39 [[buffer(24)]], packed_float4 device* crush_tab_39 [[buffer(23)]])
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
    (&kernelContext_89)->bonds_0 = bonds_40;
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
    (&kernelContext_89)->crush_0 = crush_39;
    (&kernelContext_89)->crush_rw_0 = crush_rw_39;
    (&kernelContext_89)->crush_tab_0 = crush_tab_39;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_89)->g_scan_0 = &g_scan_39;
    uint c_18 = id_36.x;
    bool _S745;
    if(c_18 >= (params_39->chunk_count_0))
    {
        _S745 = true;
    }
    else
    {
        _S745 = ((&kernelContext_89)->crush_0->enabled_0) == 0U;
    }
    if(_S745)
    {
        return;
    }
    uint _S746 = crush_flags_0(c_18, &kernelContext_89);
    uint now_0;
    if((_S746 & 2U) != 0U)
    {
        now_0 = 1U;
    }
    else
    {
        now_0 = 0U;
    }
    uint _S747 = w_new_0(&kernelContext_89);
    *((&kernelContext_89)->work_0+(_S747 + c_18)) = now_0;
    if(now_0 != 0U)
    {
        uint _S748 = a_counters_0(&kernelContext_89);
        uint _S749 = atomic_fetch_add_explicit((&kernelContext_89)->atom_0+(_S748 + 10U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_crush_gate(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], atomic<uint> device* atom_40 [[buffer(5)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], CrushParams_0 constant* crush_40 [[buffer(25)]], packed_float4 device* crush_rw_40 [[buffer(24)]], packed_float4 device* crush_tab_40 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_90;
    (&kernelContext_90)->params_0 = params_40;
    (&kernelContext_90)->work_0 = work_40;
    (&kernelContext_90)->frames_0 = frames_40;
    (&kernelContext_90)->bodies_0 = bodies_40;
    (&kernelContext_90)->chunks_0 = chunks_40;
    (&kernelContext_90)->sort_0 = sort_40;
    (&kernelContext_90)->atom_0 = atom_40;
    (&kernelContext_90)->manifolds_0 = manifolds_40;
    (&kernelContext_90)->loads_0 = loads_40;
    (&kernelContext_90)->bonds_0 = bonds_41;
    (&kernelContext_90)->args_0 = args_40;
    (&kernelContext_90)->info_0 = info_40;
    (&kernelContext_90)->scan_0 = scan_40;
    (&kernelContext_90)->parts_0 = parts_40;
    (&kernelContext_90)->colliders_0 = colliders_40;
    (&kernelContext_90)->collider_info_0 = collider_info_40;
    (&kernelContext_90)->view_chunks_0 = view_chunks_40;
    (&kernelContext_90)->view_0 = view_40;
    (&kernelContext_90)->objects_0 = objects_40;
    (&kernelContext_90)->shot_bodies_0 = shot_bodies_40;
    (&kernelContext_90)->lines_0 = lines_40;
    (&kernelContext_90)->geometry_0 = geometry_40;
    (&kernelContext_90)->shots_0 = shots_40;
    (&kernelContext_90)->crush_0 = crush_40;
    (&kernelContext_90)->crush_rw_0 = crush_rw_40;
    (&kernelContext_90)->crush_tab_0 = crush_tab_40;
    threadgroup array<uint, int(1024)> g_scan_40;
    (&kernelContext_90)->g_scan_0 = &g_scan_40;
    bool run_2;
    if((crush_40->enabled_0) != 0U)
    {
        uint _S750 = a_counters_0(&kernelContext_90);
        uint _S751 = atomic_load_explicit((&kernelContext_90)->atom_0+(_S750 + 10U), memory_order_relaxed);
        run_2 = _S751 != 0U;
    }
    else
    {
        run_2 = false;
    }
    uint _S752 = a_counters_0(&kernelContext_90);
    atomic_store_explicit((&kernelContext_90)->atom_0+(_S752 + 10U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_90)->atom_0+(_S752 + 7U), 0U, memory_order_relaxed);
    uint _S753;
    if(run_2)
    {
        uint _S754 = groups_of_0((&kernelContext_90)->params_0->chunk_count_0, 64U);
        _S753 = _S754;
    }
    else
    {
        _S753 = 0U;
    }
    set_args_0(6U, _S753, &kernelContext_90);
    if(run_2)
    {
        uint _S755 = groups_of_0((&kernelContext_90)->params_0->chunk_count_0, 1024U);
        _S753 = _S755;
    }
    else
    {
        _S753 = 0U;
    }
    set_args_0(7U, _S753, &kernelContext_90);
    int _S756;
    if(run_2)
    {
        _S756 = int(1);
    }
    else
    {
        _S756 = int(0);
    }
    set_args_0(8U, uint(_S756), &kernelContext_90);
    return;
}

void write_body_0(uint slot_7, float3 pos_1, float inv_mass_10, const Quat_0 thread* orientation_0, float3 inv_inertia_1, uint flags_3, float3 v_13, float3 w_10, KernelContext_0 thread* kernelContext_91)
{
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
        uint b_23 = 7U * (slot_7 + region_3 * kernelContext_91->params_0->max_bodies_0);
        *(kernelContext_91->bodies_0+b_23) = packed_float4(float4(pos_1, inv_mass_10)) ;
        packed_float4 device* _S757 = kernelContext_91->bodies_0+(b_23 + 1U);
        float4 _S758 = quat_vec_0(orientation_0);
        *_S757 = packed_float4(_S758) ;
        *(kernelContext_91->bodies_0+(b_23 + 2U)) = packed_float4(float4(inv_inertia_1, (as_type<float>((flags_3))))) ;
        *(kernelContext_91->bodies_0+(b_23 + 3U)) = packed_float4(float4(v_13, 0.0f)) ;
        *(kernelContext_91->bodies_0+(b_23 + 4U)) = packed_float4(float4(w_10, (as_type<float>((kernelContext_91->params_0->step_0))))) ;
        *(kernelContext_91->bodies_0+(b_23 + 5U)) = packed_float4(float4(0.0f) ) ;
        *(kernelContext_91->bodies_0+(b_23 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
        region_3 = region_3 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_apply(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], atomic<uint> device* atom_41 [[buffer(5)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], CrushParams_0 constant* crush_41 [[buffer(25)]], packed_float4 device* crush_rw_41 [[buffer(24)]], packed_float4 device* crush_tab_41 [[buffer(23)]])
{
    uint ax_4;
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
    (&kernelContext_92)->bonds_0 = bonds_42;
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
    (&kernelContext_92)->crush_0 = crush_41;
    (&kernelContext_92)->crush_rw_0 = crush_rw_41;
    (&kernelContext_92)->crush_tab_0 = crush_tab_41;
    threadgroup array<uint, int(1024)> g_scan_41;
    (&kernelContext_92)->g_scan_0 = &g_scan_41;
    uint k_28 = id_38.x;
    bool _S759;
    if(k_28 >= (params_41->chunk_count_0))
    {
        _S759 = true;
    }
    else
    {
        _S759 = ((&kernelContext_92)->crush_0->enabled_0) == 0U;
    }
    if(_S759)
    {
        return;
    }
    float4 _S760 = float4(*((&kernelContext_92)->crush_rw_0+k_28)) ;
    uint flags_4 = (as_type<uint>((_S760.w)));
    if((flags_4 & 2U) == 0U)
    {
        return;
    }
    *((&kernelContext_92)->crush_rw_0+k_28) = packed_float4(float4(_S760.xyz, (as_type<float>((flags_4 & 4294967293U))))) ;
    uint _S761 = body_of_0(k_28, &kernelContext_92);
    uint bw_1 = 7U * _S761;
    float4 _S762 = float4(*((&kernelContext_92)->bodies_0+bw_1)) ;
    Quat_0 orientation_1 = quat_of_0(float4(*((&kernelContext_92)->bodies_0+(bw_1 + 1U))) );
    float3 v_14 = (float4(*((&kernelContext_92)->bodies_0+(bw_1 + 3U))) ).xyz;
    float3 w_11 = (float4(*((&kernelContext_92)->bodies_0+(bw_1 + 4U))) ).xyz;
    uint _S763 = 4U * k_28;
    float mass_0 = (float4(*((&kernelContext_92)->chunks_0+_S763)) ).w;
    float volume_1 = (float4(*((&kernelContext_92)->crush_tab_0+(2U * k_28 + 1U))) ).y;
    uint _S764 = w_rank_0(&kernelContext_92);
    uint device* _S765 = (&kernelContext_92)->work_0+(_S764 + k_28);
    uint rank_0 = *_S765;
    if((*_S765) >= ((&kernelContext_92)->crush_0->max_events_0))
    {
        uint _S766 = a_counters_0(&kernelContext_92);
        uint _S767 = atomic_fetch_or_explicit((&kernelContext_92)->atom_0+(_S766 + 3U), 32U, memory_order_relaxed);
    }
    float3 _S768 = float3(0.0f) ;
    if(((&kernelContext_92)->crush_0->mode_1) == 1U)
    {
        _S759 = ((&kernelContext_92)->crush_0->fragments_0) > 0U;
    }
    else
    {
        _S759 = false;
    }
    uint first_6;
    float3 half_1;
    if(_S759)
    {
        uint _S769 = a_counters_0(&kernelContext_92);
        uint _S770 = atomic_load_explicit((&kernelContext_92)->atom_0+(_S769 + 8U), memory_order_relaxed);
        uint base_4 = _S770 + rank_0 * (&kernelContext_92)->crush_0->fragments_0;
        if((base_4 + (&kernelContext_92)->crush_0->fragments_0) > ((&kernelContext_92)->crush_0->pool_size_0))
        {
            uint _S771 = atomic_fetch_or_explicit((&kernelContext_92)->atom_0+(_S769 + 3U), 16U, memory_order_relaxed);
            first_6 = 4294967295U;
        }
        else
        {
            first_6 = base_4;
        }
        thread Quat_0 q_28;
        thread float3 origin_11;
        structure_pose_0(_S761, bw_1, &q_28, &origin_11, &kernelContext_92);
        float3 _S772 = (float4(*((&kernelContext_92)->chunks_0+_S763)) ).xyz;
        thread Quat_0 _S773 = q_28;
        float3 _S774 = rotate_0(&_S773, _S772);
        float3 _S775 = origin_11 + _S774;
        uint _S776 = 3U * k_28;
        Quat_0 _S777 = quat_of_0(float4(*((&kernelContext_92)->geometry_0+(_S776 + 2U))) );
        thread Quat_0 _S778 = q_28;
        thread Quat_0 _S779 = _S777;
        Quat_0 _S780 = quat_mul_0(&_S778, &_S779);
        thread Quat_0 _S781 = _S780;
        Quat_0 _S782 = normalized_0(&_S781);
        float3 h_1 = (float4(*((&kernelContext_92)->geometry_0+(_S776 + 1U))) ).xyz;
        thread uint3 g_0 = uint3(int3(int(1)) );
        thread float3 cell_0 = h_1;
        uint n_4 = (&kernelContext_92)->crush_0->fragments_0;
        for(;;)
        {
            if(n_4 > 1U)
            {
            }
            else
            {
                break;
            }
            if((cell_0.x) >= (cell_0.y))
            {
                _S759 = (cell_0.x) >= (cell_0.z);
            }
            else
            {
                _S759 = false;
            }
            if(_S759)
            {
                ax_4 = 0U;
            }
            else
            {
                if((cell_0.y) >= (cell_0.z))
                {
                    ax_4 = 1U;
                }
                else
                {
                    ax_4 = 2U;
                }
            }
            g_0[ax_4] = g_0[ax_4] * 2U;
            cell_0[ax_4] = cell_0[ax_4] * 0.5f;
            n_4 = n_4 >> 1U;
        }
        float box_volume_0 = 8.0f * h_1.x * h_1.y * h_1.z;
        float shrink_0;
        if(box_volume_0 > 0.0f)
        {
            shrink_0 = min(0.94999998807907104f, pow(volume_1 / box_volume_0, 0.3333333432674408f));
        }
        else
        {
            shrink_0 = 0.94999998807907104f;
        }
        float3 half_2 = cell_0 * float3(shrink_0) ;
        float mf_0 = mass_0 / float((&kernelContext_92)->crush_0->fragments_0);
        float _S783 = half_2.y;
        float _S784 = _S783 * _S783;
        float _S785 = half_2.z;
        float _S786 = _S785 * _S785;
        float _S787 = half_2.x;
        float _S788 = _S787 * _S787;
        float3 inertia_0 = float3((mf_0 / 3.0f))  * float3(_S784 + _S786, _S788 + _S786, _S788 + _S784);
        float3 _S789 = float3(1.0f / inertia_0.x, 1.0f / inertia_0.y, 1.0f / inertia_0.z);
        thread Quat_0 _S790 = _S782;
        Quat_0 _S791 = rot_to_rubble_0(&_S790);
        float _S792 = (float4(*((&kernelContext_92)->chunks_0+(_S763 + 3U))) ).x;
        float _S793 = min(_S787, min(_S783, _S785));
        ax_4 = 0U;
        for(;;)
        {
            if(ax_4 < ((&kernelContext_92)->crush_0->fragments_0))
            {
                _S759 = first_6 != 4294967295U;
            }
            else
            {
                _S759 = false;
            }
            if(_S759)
            {
            }
            else
            {
                break;
            }
            uint _S794 = ax_4 % g_0.x;
            uint _S795 = ax_4 / g_0.x;
            uint _S796 = _S795 % g_0.y;
            uint _S797 = ax_4 / (g_0.x * g_0.y);
            float3 offset_0 = - h_1 + cell_0 * (float3(2.0f)  * float3(uint3(_S794, _S796, _S797)) + float3(1.0f) );
            thread Quat_0 _S798 = _S782;
            float3 _S799 = rotate_0(&_S798, offset_0);
            float3 x_6 = to_rubble_0(_S775 + _S799);
            float4 _S800 = float4(*((&kernelContext_92)->crush_tab_0+((&kernelContext_92)->crush_0->t_pool_0 + first_6 + ax_4))) ;
            uint fs_0 = (as_type<uint>((_S800.x)));
            uint fc_0 = (as_type<uint>((_S800.y)));
            float _S801 = 1.0f / mf_0;
            float3 _S802 = v_14 + cross(w_11, x_6 - _S762.xyz);
            thread Quat_0 _S803 = _S791;
            write_body_0(fs_0, x_6, _S801, &_S803, _S789, 9U, _S802, w_11, &kernelContext_92);
            uint _S804 = 4U * fc_0;
            uint _S805 = _S804 + 1U;
            *((&kernelContext_92)->colliders_0+_S805) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_92)->colliders_0+_S805)) ).w)) ;
            *((&kernelContext_92)->colliders_0+(_S804 + 2U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
            *((&kernelContext_92)->colliders_0+(_S804 + 3U)) = packed_float4(float4(half_2, length(half_2))) ;
            uint _S806 = 2U * fc_0;
            *((&kernelContext_92)->collider_info_0+_S806) = packed_uint4(uint4(4294967295U, fs_0, 2U, (uint4(*((&kernelContext_92)->collider_info_0+_S806)) ).w)) ;
            *((&kernelContext_92)->collider_info_0+(_S806 + 1U)) = packed_uint4(uint4((as_type<uint>((mf_0))), (as_type<uint>((_S792))), (as_type<uint>((_S793))), 0U)) ;
            ax_4 = ax_4 + 1U;
        }
        half_1 = half_2;
    }
    else
    {
        first_6 = 4294967295U;
        half_1 = _S768;
    }
    if(rank_0 < ((&kernelContext_92)->crush_0->max_events_0))
    {
        uint ev_0 = (&kernelContext_92)->crush_0->r_events_0 + 6U * rank_0;
        *((&kernelContext_92)->crush_rw_0+ev_0) = packed_float4(float4((as_type<float>((k_28))), (as_type<float>((first_6))), (as_type<float>((flags_4 >> 4U))), _S760.x)) ;
        *((&kernelContext_92)->crush_rw_0+(ev_0 + 1U)) = packed_float4(float4(_S762.xyz, mass_0)) ;
        packed_float4 device* _S807 = (&kernelContext_92)->crush_rw_0+(ev_0 + 2U);
        thread Quat_0 _S808 = orientation_1;
        float4 _S809 = quat_vec_0(&_S808);
        *_S807 = packed_float4(_S809) ;
        *((&kernelContext_92)->crush_rw_0+(ev_0 + 3U)) = packed_float4(float4(v_14, _S760.y)) ;
        *((&kernelContext_92)->crush_rw_0+(ev_0 + 4U)) = packed_float4(float4(w_11, _S760.z)) ;
        *((&kernelContext_92)->crush_rw_0+(ev_0 + 5U)) = packed_float4(float4(half_1, volume_1)) ;
    }
    float3 park_0 = float3(float(_S761 % 1024U) * (&kernelContext_92)->crush_0->park_step_0, (&kernelContext_92)->crush_0->park_y_0, float(_S761 / 1024U) * (&kernelContext_92)->crush_0->park_step_0);
    thread Quat_0 _S810 = orientation_1;
    write_body_0(_S761, park_0, 0.0f, &_S810, _S768, 0U, _S768, _S768, &kernelContext_92);
    return;
}

[[kernel]] void fr_crush_end(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], atomic<uint> device* atom_42 [[buffer(5)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], CrushParams_0 constant* crush_42 [[buffer(25)]], packed_float4 device* crush_rw_42 [[buffer(24)]], packed_float4 device* crush_tab_42 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_93;
    (&kernelContext_93)->params_0 = params_42;
    (&kernelContext_93)->work_0 = work_42;
    (&kernelContext_93)->frames_0 = frames_42;
    (&kernelContext_93)->bodies_0 = bodies_42;
    (&kernelContext_93)->chunks_0 = chunks_42;
    (&kernelContext_93)->sort_0 = sort_42;
    (&kernelContext_93)->atom_0 = atom_42;
    (&kernelContext_93)->manifolds_0 = manifolds_42;
    (&kernelContext_93)->loads_0 = loads_42;
    (&kernelContext_93)->bonds_0 = bonds_43;
    (&kernelContext_93)->args_0 = args_42;
    (&kernelContext_93)->info_0 = info_42;
    (&kernelContext_93)->scan_0 = scan_42;
    (&kernelContext_93)->parts_0 = parts_42;
    (&kernelContext_93)->colliders_0 = colliders_42;
    (&kernelContext_93)->collider_info_0 = collider_info_42;
    (&kernelContext_93)->view_chunks_0 = view_chunks_42;
    (&kernelContext_93)->view_0 = view_42;
    (&kernelContext_93)->objects_0 = objects_42;
    (&kernelContext_93)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_93)->lines_0 = lines_42;
    (&kernelContext_93)->geometry_0 = geometry_42;
    (&kernelContext_93)->shots_0 = shots_42;
    (&kernelContext_93)->crush_0 = crush_42;
    (&kernelContext_93)->crush_rw_0 = crush_rw_42;
    (&kernelContext_93)->crush_tab_0 = crush_tab_42;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_93)->g_scan_0 = &g_scan_42;
    uint blocks_2 = (params_42->chunk_count_0 + 1023U) / 1024U;
    uint _S811 = scan_new_blocks_0(&kernelContext_93);
    uint device* _S812 = (&kernelContext_93)->work_0+(_S811 + blocks_2);
    uint total_3 = *_S812;
    uint _S813 = a_counters_0(&kernelContext_93);
    atomic_store_explicit((&kernelContext_93)->atom_0+(_S813 + 7U), *_S812, memory_order_relaxed);
    uint _S814 = atomic_fetch_add_explicit((&kernelContext_93)->atom_0+(_S813 + 9U), total_3, memory_order_relaxed);
    if(((&kernelContext_93)->crush_0->mode_1) == 1U)
    {
        uint _S815 = atomic_fetch_add_explicit((&kernelContext_93)->atom_0+(_S813 + 8U), total_3 * (&kernelContext_93)->crush_0->fragments_0, memory_order_relaxed);
    }
    return;
}

