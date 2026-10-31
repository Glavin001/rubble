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

uint w_host_breaks_0(KernelContext_0 thread* kernelContext_20)
{
    uint _S95 = w_pulses_0(kernelContext_20);
    return _S95 + 4U * kernelContext_20->params_0->chunk_count_0;
}

[[kernel]] void fr_host_breaks(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], CrushParams_0 constant* crush_3 [[buffer(25)]], packed_float4 device* crush_rw_3 [[buffer(24)]], packed_float4 device* crush_tab_3 [[buffer(23)]])
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
    uint i_1 = id_2.x;
    uint _S96 = w_host_breaks_0(&kernelContext_21);
    if(i_1 >= (*((&kernelContext_21)->work_0+_S96)))
    {
        return;
    }
    uint device* _S97 = (&kernelContext_21)->work_0+(_S96 + 1U + i_1);
    uint b_2 = *_S97;
    bool _S98;
    if((*_S97) >= ((&kernelContext_21)->params_0->bond_count_0))
    {
        _S98 = true;
    }
    else
    {
        uint _S99 = w_alive_0(&kernelContext_21);
        _S98 = (*((&kernelContext_21)->work_0+(_S99 + b_2))) == 0U;
    }
    if(_S98)
    {
        return;
    }
    uint _S100 = w_alive_0(&kernelContext_21);
    *((&kernelContext_21)->work_0+(_S100 + b_2)) = 0U;
    uint _S101 = a_dirty_0(&kernelContext_21);
    uint _S102 = body_of_0((uint4(*((&kernelContext_21)->bonds_0+b_2)) ).x, &kernelContext_21);
    uint _S103 = atomic_fetch_or_explicit((&kernelContext_21)->atom_0+(_S101 + _S102), 1U, memory_order_relaxed);
    uint _S104 = a_counters_0(&kernelContext_21);
    uint _S105 = atomic_fetch_add_explicit((&kernelContext_21)->atom_0+_S104, 1U, memory_order_relaxed);
    uint _S106 = atomic_fetch_add_explicit((&kernelContext_21)->atom_0+(_S104 + 1U), 1U, memory_order_relaxed);
    return;
}

uint groups_of_0(uint n_2, uint size_0)
{
    uint _S107 = (n_2 + size_0 - 1U) / size_0;
    return _S107;
}

void set_args_0(uint at_0, uint x_1, KernelContext_0 thread* kernelContext_22)
{
    uint _S108 = 3U * at_0;
    *(kernelContext_22->args_0+_S108) = x_1;
    *(kernelContext_22->args_0+(_S108 + 1U)) = 1U;
    *(kernelContext_22->args_0+(_S108 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], CrushParams_0 constant* crush_4 [[buffer(25)]], packed_float4 device* crush_rw_4 [[buffer(24)]], packed_float4 device* crush_tab_4 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_23;
    (&kernelContext_23)->params_0 = params_4;
    (&kernelContext_23)->work_0 = work_4;
    (&kernelContext_23)->frames_0 = frames_4;
    (&kernelContext_23)->bodies_0 = bodies_4;
    (&kernelContext_23)->chunks_0 = chunks_4;
    (&kernelContext_23)->sort_0 = sort_4;
    (&kernelContext_23)->atom_0 = atom_4;
    (&kernelContext_23)->manifolds_0 = manifolds_4;
    (&kernelContext_23)->loads_0 = loads_4;
    (&kernelContext_23)->bonds_0 = bonds_4;
    (&kernelContext_23)->args_0 = args_4;
    (&kernelContext_23)->info_0 = info_4;
    (&kernelContext_23)->scan_0 = scan_4;
    (&kernelContext_23)->parts_0 = parts_4;
    (&kernelContext_23)->colliders_0 = colliders_4;
    (&kernelContext_23)->collider_info_0 = collider_info_4;
    (&kernelContext_23)->view_chunks_0 = view_chunks_4;
    (&kernelContext_23)->view_0 = view_4;
    (&kernelContext_23)->objects_0 = objects_4;
    (&kernelContext_23)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_23)->lines_0 = lines_4;
    (&kernelContext_23)->geometry_0 = geometry_4;
    (&kernelContext_23)->shots_0 = shots_4;
    (&kernelContext_23)->crush_0 = crush_4;
    (&kernelContext_23)->crush_rw_0 = crush_rw_4;
    (&kernelContext_23)->crush_tab_0 = crush_tab_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_23)->g_scan_0 = &g_scan_4;
    uint _S109 = a_counters_0(&kernelContext_23);
    uint _S110 = atomic_load_explicit((&kernelContext_23)->atom_0+_S109, memory_order_relaxed);
    bool run_0 = _S110 != 0U;
    atomic_store_explicit((&kernelContext_23)->atom_0+_S109, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_23)->atom_0+(_S109 + 5U + (&kernelContext_23)->params_0->parity_0), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S111 = atomic_fetch_add_explicit((&kernelContext_23)->atom_0+(_S109 + 4U), 1U, memory_order_relaxed);
    }
    uint _S112;
    if(run_0)
    {
        uint _S113 = groups_of_0((&kernelContext_23)->params_0->chunk_count_0, 64U);
        _S112 = _S113;
    }
    else
    {
        _S112 = 0U;
    }
    set_args_0(0U, _S112, &kernelContext_23);
    if(run_0)
    {
        uint _S114 = groups_of_0((&kernelContext_23)->params_0->bond_count_0, 64U);
        _S112 = _S114;
    }
    else
    {
        _S112 = 0U;
    }
    set_args_0(1U, _S112, &kernelContext_23);
    if(run_0)
    {
        uint _S115 = groups_of_0((&kernelContext_23)->params_0->body_count_0, 64U);
        _S112 = _S115;
    }
    else
    {
        _S112 = 0U;
    }
    set_args_0(2U, _S112, &kernelContext_23);
    if(run_0)
    {
        uint _S116 = groups_of_0((&kernelContext_23)->params_0->chunk_count_0, 1024U);
        _S112 = _S116;
    }
    else
    {
        _S112 = 0U;
    }
    set_args_0(3U, _S112, &kernelContext_23);
    if(run_0)
    {
        uint _S117 = groups_of_0((&kernelContext_23)->params_0->body_count_0, 1024U);
        _S112 = _S117;
    }
    else
    {
        _S112 = 0U;
    }
    set_args_0(4U, _S112, &kernelContext_23);
    int _S118;
    if(run_0)
    {
        _S118 = int(1);
    }
    else
    {
        _S118 = int(0);
    }
    set_args_0(5U, uint(_S118), &kernelContext_23);
    return;
}

bool dirty_0(uint body_0, KernelContext_0 thread* kernelContext_24)
{
    uint _S119 = a_dirty_0(kernelContext_24);
    uint _S120 = atomic_load_explicit(kernelContext_24->atom_0+(_S119 + body_0), memory_order_relaxed);
    return _S120 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_25)
{
    return kernelContext_25->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_26)
{
    return 2U * kernelContext_26->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], CrushParams_0 constant* crush_5 [[buffer(25)]], packed_float4 device* crush_rw_5 [[buffer(24)]], packed_float4 device* crush_tab_5 [[buffer(23)]])
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
    (&kernelContext_27)->crush_0 = crush_5;
    (&kernelContext_27)->crush_rw_0 = crush_rw_5;
    (&kernelContext_27)->crush_tab_0 = crush_tab_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_27)->g_scan_0 = &g_scan_5;
    uint c_2 = id_4.x;
    bool _S121;
    if(c_2 >= (params_5->chunk_count_0))
    {
        _S121 = true;
    }
    else
    {
        uint _S122 = body_of_0(c_2, &kernelContext_27);
        bool _S123 = dirty_0(_S122, &kernelContext_27);
        _S121 = !_S123;
    }
    if(_S121)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_27)->atom_0+(a_parent_0() + c_2), c_2, memory_order_relaxed);
    uint _S124 = a_count_0(&kernelContext_27);
    atomic_store_explicit((&kernelContext_27)->atom_0+(_S124 + c_2), 0U, memory_order_relaxed);
    uint _S125 = a_anchor_0(&kernelContext_27);
    atomic_store_explicit((&kernelContext_27)->atom_0+(_S125 + c_2), 0U, memory_order_relaxed);
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
        uint _S126 = a_sums_0(&kernelContext_27);
        atomic_store_explicit((&kernelContext_27)->atom_0+(_S126 + 20U * c_2 + w_2), 0U, memory_order_relaxed);
        w_2 = w_2 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], CrushParams_0 constant* crush_6 [[buffer(25)]], packed_float4 device* crush_rw_6 [[buffer(24)]], packed_float4 device* crush_tab_6 [[buffer(23)]])
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
    uint s_0 = id_5.x;
    bool _S127;
    if(s_0 >= (params_6->body_count_0))
    {
        _S127 = true;
    }
    else
    {
        bool _S128 = dirty_0(s_0, &kernelContext_28);
        _S127 = !_S128;
    }
    if(_S127)
    {
        return;
    }
    uint _S129 = a_best_count_0(&kernelContext_28);
    atomic_store_explicit((&kernelContext_28)->atom_0+(_S129 + s_0), 0U, memory_order_relaxed);
    uint _S130 = a_best_root_0(&kernelContext_28);
    atomic_store_explicit((&kernelContext_28)->atom_0+(_S130 + s_0), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_2, KernelContext_0 thread* kernelContext_29)
{
    uint _S131 = a_parent_0();
    uint _S132 = atomic_load_explicit(kernelContext_29->atom_0+(_S131 + x_2), memory_order_relaxed);
    uint p_1 = _S132;
    uint _S133 = x_2;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S134;
        if(guard_0 < 1048576U)
        {
            _S134 = p_1 != _S133;
        }
        else
        {
            _S134 = false;
        }
        if(_S134)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_29->atom_0+(_S131 + p_1), memory_order_relaxed);
        if(gp_0 != p_1)
        {
            uint _S135;
            {
            uint _metal_cas_comparand = p_1;
            atomic_compare_exchange_weak_explicit(kernelContext_29->atom_0+(_S131 + _S133), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S135 = _metal_cas_comparand;
            }
        }
        uint _S136 = guard_0 + 1U;
        _S133 = p_1;
        p_1 = gp_0;
        guard_0 = _S136;
    }
    return _S133;
}

[[kernel]] void fr_uf_union(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], CrushParams_0 constant* crush_7 [[buffer(25)]], packed_float4 device* crush_rw_7 [[buffer(24)]], packed_float4 device* crush_tab_7 [[buffer(23)]])
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
    uint i_2 = id_6.x;
    bool _S137;
    if(i_2 >= (params_7->bond_count_0))
    {
        _S137 = true;
    }
    else
    {
        uint _S138 = w_alive_0(&kernelContext_30);
        _S137 = (*((&kernelContext_30)->work_0+(_S138 + i_2))) == 0U;
    }
    if(_S137)
    {
        return;
    }
    uint4 _S139 = uint4(*((&kernelContext_30)->bonds_0+i_2)) ;
    uint _S140 = _S139.x;
    uint _S141 = body_of_0(_S140, &kernelContext_30);
    bool _S142 = dirty_0(_S141, &kernelContext_30);
    if(!_S142)
    {
        return;
    }
    uint _S143 = _S139.y;
    uint a_2 = _S140;
    uint b_3 = _S143;
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
        uint _S144 = find_0(a_2, &kernelContext_30);
        uint _S145 = find_0(b_3, &kernelContext_30);
        if(_S144 == _S145)
        {
            return;
        }
        uint _S146 = max(_S144, _S145);
        uint _S147 = min(_S144, _S145);
        uint was_0;
        {
        uint _metal_cas_comparand = _S146;
        atomic_compare_exchange_weak_explicit((&kernelContext_30)->atom_0+(a_parent_0() + _S146), &_metal_cas_comparand, _S147, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S146)
        {
            return;
        }
        uint _S148 = guard_1 + 1U;
        a_2 = _S146;
        b_3 = _S147;
        guard_1 = _S148;
    }
    uint _S149 = a_counters_0(&kernelContext_30);
    uint _S150 = atomic_fetch_or_explicit((&kernelContext_30)->atom_0+(_S149 + 3U), 4U, memory_order_relaxed);
    return;
}

void add64_0(uint at_1, float value_0, float scale_0, KernelContext_0 thread* kernelContext_31)
{
    float x_3 = round(value_0 * scale_0);
    float x_4;
    if(!((abs(x_3)) < 1.073741824e+09f))
    {
        uint _S151 = a_counters_0(kernelContext_31);
        uint _S152 = atomic_fetch_or_explicit(kernelContext_31->atom_0+(_S151 + 3U), 2U, memory_order_relaxed);
        x_4 = 0.0f;
    }
    else
    {
        x_4 = x_3;
    }
    int v_4 = int(x_4);
    uint lo_0 = (as_type<uint>((v_4)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_31->atom_0+at_1, lo_0, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_0) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S153;
    if(v_4 < int(0))
    {
        _S153 = 4294967295U;
    }
    else
    {
        _S153 = 0U;
    }
    uint hi_0 = _S153 + carry_0;
    if(hi_0 != 0U)
    {
        uint _S154 = atomic_fetch_add_explicit(kernelContext_31->atom_0+(at_1 + 1U), hi_0, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], CrushParams_0 constant* crush_8 [[buffer(25)]], packed_float4 device* crush_rw_8 [[buffer(24)]], packed_float4 device* crush_tab_8 [[buffer(23)]])
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
    uint c_3 = id_7.x;
    bool _S155;
    if(c_3 >= (params_8->chunk_count_0))
    {
        _S155 = true;
    }
    else
    {
        uint _S156 = body_of_0(c_3, &kernelContext_32);
        bool _S157 = dirty_0(_S156, &kernelContext_32);
        _S155 = !_S157;
    }
    if(_S155)
    {
        return;
    }
    uint _S158 = find_0(c_3, &kernelContext_32);
    uint _S159 = w_label_0(&kernelContext_32);
    *((&kernelContext_32)->work_0+(_S159 + c_3)) = _S158;
    uint _S160 = a_count_0(&kernelContext_32);
    uint _S161 = atomic_fetch_add_explicit((&kernelContext_32)->atom_0+(_S160 + _S158), 1U, memory_order_relaxed);
    if(((&kernelContext_32)->info_0[c_3] & 1U) != 0U)
    {
        uint _S162 = a_anchor_0(&kernelContext_32);
        uint _S163 = atomic_fetch_or_explicit((&kernelContext_32)->atom_0+(_S162 + _S158), 1U, memory_order_relaxed);
    }
    uint _S164 = 4U * c_3;
    float4 _S165 = float4(*((&kernelContext_32)->chunks_0+_S164)) ;
    float4 _S166 = float4(*((&kernelContext_32)->chunks_0+(_S164 + 1U))) ;
    float4 _S167 = float4(*((&kernelContext_32)->chunks_0+(_S164 + 2U))) ;
    float m_2 = _S165.w;
    float3 d_0 = _S165.xyz - (float4(*((&kernelContext_32)->chunks_0+4U * _S158)) ).xyz;
    float dd_0 = dot(d_0, d_0);
    uint _S168 = a_sums_0(&kernelContext_32);
    uint at_2 = _S168 + 20U * _S158;
    add64_0(at_2, m_2, (&kernelContext_32)->params_0->scale_m_0, &kernelContext_32);
    float _S169 = d_0.x;
    float _S170 = m_2 * _S169;
    add64_0(at_2 + 2U, _S170, (&kernelContext_32)->params_0->scale_s_0, &kernelContext_32);
    float _S171 = d_0.y;
    float _S172 = m_2 * _S171;
    add64_0(at_2 + 4U, _S172, (&kernelContext_32)->params_0->scale_s_0, &kernelContext_32);
    float _S173 = d_0.z;
    add64_0(at_2 + 6U, m_2 * _S173, (&kernelContext_32)->params_0->scale_s_0, &kernelContext_32);
    add64_0(at_2 + 8U, _S166.x + m_2 * (dd_0 - _S169 * _S169), (&kernelContext_32)->params_0->scale_j_0, &kernelContext_32);
    add64_0(at_2 + 10U, _S166.y + m_2 * (dd_0 - _S171 * _S171), (&kernelContext_32)->params_0->scale_j_0, &kernelContext_32);
    add64_0(at_2 + 12U, _S166.z + m_2 * (dd_0 - _S173 * _S173), (&kernelContext_32)->params_0->scale_j_0, &kernelContext_32);
    add64_0(at_2 + 14U, _S166.w - _S170 * _S171, (&kernelContext_32)->params_0->scale_j_0, &kernelContext_32);
    add64_0(at_2 + 16U, _S167.x - _S170 * _S173, (&kernelContext_32)->params_0->scale_j_0, &kernelContext_32);
    add64_0(at_2 + 18U, _S167.y - _S172 * _S173, (&kernelContext_32)->params_0->scale_j_0, &kernelContext_32);
    return;
}

bool is_root_0(uint c_4, KernelContext_0 thread* kernelContext_33)
{
    bool _S174;
    if(c_4 < (kernelContext_33->params_0->chunk_count_0))
    {
        uint _S175 = body_of_0(c_4, kernelContext_33);
        bool _S176 = dirty_0(_S175, kernelContext_33);
        _S174 = _S176;
    }
    else
    {
        _S174 = false;
    }
    if(_S174)
    {
        uint _S177 = w_label_0(kernelContext_33);
        _S174 = (*(kernelContext_33->work_0+(_S177 + c_4))) == c_4;
    }
    else
    {
        _S174 = false;
    }
    return _S174;
}

[[kernel]] void fr_best_count(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], CrushParams_0 constant* crush_9 [[buffer(25)]], packed_float4 device* crush_rw_9 [[buffer(24)]], packed_float4 device* crush_tab_9 [[buffer(23)]])
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
    (&kernelContext_34)->crush_0 = crush_9;
    (&kernelContext_34)->crush_rw_0 = crush_rw_9;
    (&kernelContext_34)->crush_tab_0 = crush_tab_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_34)->g_scan_0 = &g_scan_9;
    uint c_5 = id_8.x;
    bool _S178 = is_root_0(c_5, &kernelContext_34);
    if(!_S178)
    {
        return;
    }
    uint _S179 = a_best_count_0(&kernelContext_34);
    uint _S180 = body_of_0(c_5, &kernelContext_34);
    atomic<uint> device* _S181 = (&kernelContext_34)->atom_0+(_S179 + _S180);
    uint _S182 = a_count_0(&kernelContext_34);
    uint _S183 = atomic_load_explicit((&kernelContext_34)->atom_0+(_S182 + c_5), memory_order_relaxed);
    uint _S184 = atomic_fetch_max_explicit(_S181, _S183, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], CrushParams_0 constant* crush_10 [[buffer(25)]], packed_float4 device* crush_rw_10 [[buffer(24)]], packed_float4 device* crush_tab_10 [[buffer(23)]])
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
    (&kernelContext_35)->crush_0 = crush_10;
    (&kernelContext_35)->crush_rw_0 = crush_rw_10;
    (&kernelContext_35)->crush_tab_0 = crush_tab_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_35)->g_scan_0 = &g_scan_10;
    uint c_6 = id_9.x;
    bool _S185 = is_root_0(c_6, &kernelContext_35);
    if(!_S185)
    {
        return;
    }
    uint _S186 = body_of_0(c_6, &kernelContext_35);
    uint _S187 = a_count_0(&kernelContext_35);
    uint _S188 = atomic_load_explicit((&kernelContext_35)->atom_0+(_S187 + c_6), memory_order_relaxed);
    uint _S189 = a_best_count_0(&kernelContext_35);
    uint _S190 = atomic_load_explicit((&kernelContext_35)->atom_0+(_S189 + _S186), memory_order_relaxed);
    if(_S188 == _S190)
    {
        uint _S191 = a_best_root_0(&kernelContext_35);
        uint _S192 = atomic_fetch_min_explicit((&kernelContext_35)->atom_0+(_S191 + _S186), c_6, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], CrushParams_0 constant* crush_11 [[buffer(25)]], packed_float4 device* crush_rw_11 [[buffer(24)]], packed_float4 device* crush_tab_11 [[buffer(23)]])
{
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
    (&kernelContext_36)->crush_0 = crush_11;
    (&kernelContext_36)->crush_rw_0 = crush_rw_11;
    (&kernelContext_36)->crush_tab_0 = crush_tab_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_36)->g_scan_0 = &g_scan_11;
    uint c_7 = id_10.x;
    if(c_7 >= (params_11->chunk_count_0))
    {
        return;
    }
    uint _S193 = w_new_0(&kernelContext_36);
    uint device* _S194 = (&kernelContext_36)->work_0+(_S193 + c_7);
    bool _S195 = is_root_0(c_7, &kernelContext_36);
    bool _S196;
    if(_S195)
    {
        uint _S197 = a_best_root_0(&kernelContext_36);
        uint _S198 = body_of_0(c_7, &kernelContext_36);
        uint _S199 = atomic_load_explicit((&kernelContext_36)->atom_0+(_S197 + _S198), memory_order_relaxed);
        _S196 = _S199 != c_7;
    }
    else
    {
        _S196 = false;
    }
    uint _S200;
    if(_S196)
    {
        _S200 = 1U;
    }
    else
    {
        _S200 = 0U;
    }
    *_S194 = _S200;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], CrushParams_0 constant* crush_12 [[buffer(25)]], packed_float4 device* crush_rw_12 [[buffer(24)]], packed_float4 device* crush_tab_12 [[buffer(23)]])
{
    uint step_2;
    uint add_0;
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
    (&kernelContext_37)->crush_0 = crush_12;
    (&kernelContext_37)->crush_rw_0 = crush_rw_12;
    (&kernelContext_37)->crush_tab_0 = crush_tab_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_37)->g_scan_0 = &g_scan_12;
    uint t_2 = thread_0.x;
    uint _S201 = group_0.x;
    uint _S202 = 1024U * _S201 + 4U * t_2;
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
        if((_S202 + j_2) < ((&kernelContext_37)->scan_0->n_0))
        {
            step_2 = *((&kernelContext_37)->work_0+((&kernelContext_37)->scan_0->in_base_0 + _S202 + j_2));
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
    (*(&kernelContext_37)->g_scan_0)[t_2] = sum_2;
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
            add_0 = (*(&kernelContext_37)->g_scan_0)[t_2 - step_2];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_37)->g_scan_0)[t_2] = (*(&kernelContext_37)->g_scan_0)[t_2] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_2 = step_2 << 1U;
    }
    uint _S203 = (*(&kernelContext_37)->g_scan_0)[t_2] - sum_2;
    j_2 = 0U;
    add_0 = _S203;
    for(;;)
    {
        if(j_2 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S202 + j_2) < ((&kernelContext_37)->scan_0->n_0))
        {
            *((&kernelContext_37)->work_0+((&kernelContext_37)->scan_0->out_base_0 + _S202 + j_2)) = add_0;
        }
        uint run_1 = add_0 + v_5[j_2];
        j_2 = j_2 + 1U;
        add_0 = run_1;
    }
    if(t_2 == 255U)
    {
        *((&kernelContext_37)->work_0+((&kernelContext_37)->scan_0->block_base_0 + _S201)) = (*(&kernelContext_37)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], CrushParams_0 constant* crush_13 [[buffer(25)]], packed_float4 device* crush_rw_13 [[buffer(24)]], packed_float4 device* crush_tab_13 [[buffer(23)]])
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
    (&kernelContext_38)->crush_0 = crush_13;
    (&kernelContext_38)->crush_rw_0 = crush_rw_13;
    (&kernelContext_38)->crush_tab_0 = crush_tab_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_38)->g_scan_0 = &g_scan_13;
    uint t_3 = thread_1.x;
    uint blocks_0 = (scan_13->n_0 + 1023U) / 1024U;
    bool _S204 = t_3 < blocks_0;
    uint v_6;
    if(_S204)
    {
        v_6 = *((&kernelContext_38)->work_0+((&kernelContext_38)->scan_0->block_base_0 + t_3));
    }
    else
    {
        v_6 = 0U;
    }
    (*(&kernelContext_38)->g_scan_0)[t_3] = v_6;
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
            add_1 = (*(&kernelContext_38)->g_scan_0)[t_3 - step_3];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_38)->g_scan_0)[t_3] = (*(&kernelContext_38)->g_scan_0)[t_3] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_3 = step_3 << 1U;
    }
    if(_S204)
    {
        *((&kernelContext_38)->work_0+((&kernelContext_38)->scan_0->block_base_0 + t_3)) = (*(&kernelContext_38)->g_scan_0)[t_3] - v_6;
    }
    if(t_3 == 1023U)
    {
        *((&kernelContext_38)->work_0+((&kernelContext_38)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_38)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], CrushParams_0 constant* crush_14 [[buffer(25)]], packed_float4 device* crush_rw_14 [[buffer(24)]], packed_float4 device* crush_tab_14 [[buffer(23)]])
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
    (&kernelContext_39)->crush_0 = crush_14;
    (&kernelContext_39)->crush_rw_0 = crush_rw_14;
    (&kernelContext_39)->crush_tab_0 = crush_tab_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_39)->g_scan_0 = &g_scan_14;
    uint i_3 = id_11.x;
    if(i_3 >= (scan_14->n_0))
    {
        return;
    }
    uint device* _S205 = (&kernelContext_39)->work_0+((&kernelContext_39)->scan_0->out_base_0 + i_3);
    *_S205 = *_S205 + *((&kernelContext_39)->work_0+((&kernelContext_39)->scan_0->block_base_0 + i_3 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], CrushParams_0 constant* crush_15 [[buffer(25)]], packed_float4 device* crush_rw_15 [[buffer(24)]], packed_float4 device* crush_tab_15 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_40;
    (&kernelContext_40)->params_0 = params_15;
    (&kernelContext_40)->work_0 = work_15;
    (&kernelContext_40)->frames_0 = frames_15;
    (&kernelContext_40)->bodies_0 = bodies_15;
    (&kernelContext_40)->chunks_0 = chunks_15;
    (&kernelContext_40)->sort_0 = sort_15;
    (&kernelContext_40)->atom_0 = atom_15;
    (&kernelContext_40)->manifolds_0 = manifolds_15;
    (&kernelContext_40)->loads_0 = loads_15;
    (&kernelContext_40)->bonds_0 = bonds_15;
    (&kernelContext_40)->args_0 = args_15;
    (&kernelContext_40)->info_0 = info_15;
    (&kernelContext_40)->scan_0 = scan_15;
    (&kernelContext_40)->parts_0 = parts_15;
    (&kernelContext_40)->colliders_0 = colliders_15;
    (&kernelContext_40)->collider_info_0 = collider_info_15;
    (&kernelContext_40)->view_chunks_0 = view_chunks_15;
    (&kernelContext_40)->view_0 = view_15;
    (&kernelContext_40)->objects_0 = objects_15;
    (&kernelContext_40)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_40)->lines_0 = lines_15;
    (&kernelContext_40)->geometry_0 = geometry_15;
    (&kernelContext_40)->shots_0 = shots_15;
    (&kernelContext_40)->crush_0 = crush_15;
    (&kernelContext_40)->crush_rw_0 = crush_rw_15;
    (&kernelContext_40)->crush_tab_0 = crush_tab_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_40)->g_scan_0 = &g_scan_15;
    uint s_1 = id_12.x;
    bool _S206;
    if(s_1 >= (params_15->body_count_0))
    {
        _S206 = true;
    }
    else
    {
        uint _S207 = w_free_0(&kernelContext_40);
        _S206 = (*((&kernelContext_40)->work_0+(_S207 + s_1))) == 0U;
    }
    if(_S206)
    {
        return;
    }
    uint _S208 = w_free_list_0(&kernelContext_40);
    uint _S209 = w_free_rank_0(&kernelContext_40);
    *((&kernelContext_40)->work_0+(_S208 + *((&kernelContext_40)->work_0+(_S209 + s_1)))) = s_1;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_41)
{
    uint _S210 = w_free_list_0(kernelContext_41);
    return _S210 + kernelContext_41->params_0->body_count_0;
}

uint scan_new_blocks_1(KernelContext_0 thread* kernelContext_42)
{
    uint _S211 = w_free_list_0(kernelContext_42);
    return _S211 + kernelContext_42->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_43)
{
    uint _S212 = scan_new_blocks_1(kernelContext_43);
    return _S212 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_44)
{
    uint blocks_1 = (kernelContext_44->params_0->body_count_0 + 1023U) / 1024U;
    uint _S213 = scan_free_blocks_0(kernelContext_44);
    return *(kernelContext_44->work_0+(_S213 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], CrushParams_0 constant* crush_16 [[buffer(25)]], packed_float4 device* crush_rw_16 [[buffer(24)]], packed_float4 device* crush_tab_16 [[buffer(23)]])
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
    (&kernelContext_45)->crush_0 = crush_16;
    (&kernelContext_45)->crush_rw_0 = crush_rw_16;
    (&kernelContext_45)->crush_tab_0 = crush_tab_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_45)->g_scan_0 = &g_scan_16;
    uint c_8 = id_13.x;
    bool _S214 = is_root_0(c_8, &kernelContext_45);
    if(!_S214)
    {
        return;
    }
    uint _S215 = body_of_0(c_8, &kernelContext_45);
    uint _S216 = a_best_root_0(&kernelContext_45);
    uint _S217 = atomic_load_explicit((&kernelContext_45)->atom_0+(_S216 + _S215), memory_order_relaxed);
    uint slot_1;
    if(_S217 != c_8)
    {
        uint _S218 = w_rank_0(&kernelContext_45);
        uint device* _S219 = (&kernelContext_45)->work_0+(_S218 + c_8);
        uint k_1 = *_S219;
        uint _S220 = free_total_0(&kernelContext_45);
        if((*_S219) < _S220)
        {
            uint _S221 = w_free_list_0(&kernelContext_45);
            uint device* _S222 = (&kernelContext_45)->work_0+(_S221 + k_1);
            uint slot_2 = *_S222;
            uint _S223 = w_free_0(&kernelContext_45);
            *((&kernelContext_45)->work_0+(_S223 + *_S222)) = 0U;
            uint _S224 = a_counters_0(&kernelContext_45);
            uint _S225 = atomic_fetch_add_explicit((&kernelContext_45)->atom_0+(_S224 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S226 = a_counters_0(&kernelContext_45);
            uint _S227 = atomic_fetch_or_explicit((&kernelContext_45)->atom_0+(_S226 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S215;
        }
    }
    else
    {
        slot_1 = _S215;
    }
    uint _S228 = w_slot_0(&kernelContext_45);
    *((&kernelContext_45)->work_0+(_S228 + c_8)) = slot_1;
    return;
}

float read64_0(uint at_3, float scale_1, KernelContext_0 thread* kernelContext_46)
{
    uint _S229 = atomic_load_explicit(kernelContext_46->atom_0+(at_3 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S229)));
    uint lo_1 = atomic_load_explicit(kernelContext_46->atom_0+at_3, memory_order_relaxed);
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
    array<float, int(3)> _S230 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S231 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S232 = { { r2_0.x, r2_0.y, r2_0.z } };
    thread array<array<float, int(3)>, int(3)> a_3;
    a_3[int(0)] = _S230;
    a_3[int(1)] = _S231;
    a_3[int(2)] = _S232;
    array<float, int(3)> _S233 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S234 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S235 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_7;
    v_7[int(0)] = _S233;
    v_7[int(1)] = _S234;
    v_7[int(2)] = _S235;
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
            bool _S236;
            if(k_2 != p_2)
            {
                _S236 = k_2 != q_10;
            }
            else
            {
                _S236 = false;
            }
            if(_S236)
            {
                float kp_0 = a_3[k_2][p_2];
                float kq_0 = a_3[k_2][q_10];
                float _S237 = c_9 * a_3[k_2][p_2] - s_2 * a_3[k_2][q_10];
                a_3[k_2][p_2] = _S237;
                a_3[p_2][k_2] = _S237;
                float _S238 = s_2 * kp_0 + c_9 * kq_0;
                a_3[k_2][q_10] = _S238;
                a_3[q_10][k_2] = _S238;
            }
            k_2 = k_2 + 1U;
        }
        float _S239 = c_9 * c_9;
        float _S240 = 2.0f * s_2 * c_9 * apq_0;
        float _S241 = s_2 * s_2;
        a_3[p_2][p_2] = _S239 * app_0 - _S240 + _S241 * aqq_0;
        a_3[q_10][q_10] = _S241 * app_0 + _S240 + _S239 * aqq_0;
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
    thread Quat_0 _S242 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S243 = normalized_0(&_S242);
    *rot_0 = _S243;
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
    thread Quat_0 _S244 = axes_1;
    Quat_0 _S245 = quat_mul_0(&_S244, q_11);
    thread Quat_0 _S246 = _S245;
    Quat_0 _S247 = normalized_0(&_S246);
    return _S247;
}

uint stage_0(uint c_10, KernelContext_0 thread* kernelContext_47)
{
    return 2U * kernelContext_47->params_0->body_count_0 + 9U * c_10;
}

float4 quat_vec_0(const Quat_0 thread* q_12)
{
    return float4(q_12->x_0, q_12->y_0, q_12->z_0, q_12->w_0);
}

[[kernel]] void fr_piece(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], CrushParams_0 constant* crush_17 [[buffer(25)]], packed_float4 device* crush_rw_17 [[buffer(24)]], packed_float4 device* crush_tab_17 [[buffer(23)]])
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
    (&kernelContext_48)->crush_0 = crush_17;
    (&kernelContext_48)->crush_rw_0 = crush_rw_17;
    (&kernelContext_48)->crush_tab_0 = crush_tab_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_48)->g_scan_0 = &g_scan_17;
    uint c_11 = id_14.x;
    bool _S248 = is_root_0(c_11, &kernelContext_48);
    if(!_S248)
    {
        return;
    }
    uint _S249 = body_of_0(c_11, &kernelContext_48);
    uint pb_0 = 7U * _S249;
    uint _S250 = a_sums_0(&kernelContext_48);
    uint at_4 = _S250 + 20U * c_11;
    float _S251 = read64_0(at_4, (&kernelContext_48)->params_0->scale_m_0, &kernelContext_48);
    float _S252 = read64_0(at_4 + 2U, (&kernelContext_48)->params_0->scale_s_0, &kernelContext_48);
    float _S253 = read64_0(at_4 + 4U, (&kernelContext_48)->params_0->scale_s_0, &kernelContext_48);
    float _S254 = read64_0(at_4 + 6U, (&kernelContext_48)->params_0->scale_s_0, &kernelContext_48);
    float3 d_1 = float3(_S252, _S253, _S254) / float3(max(_S251, 1.00000000317107685e-30f)) ;
    float _S255 = read64_0(at_4 + 8U, (&kernelContext_48)->params_0->scale_j_0, &kernelContext_48);
    float _S256 = read64_0(at_4 + 10U, (&kernelContext_48)->params_0->scale_j_0, &kernelContext_48);
    float _S257 = read64_0(at_4 + 12U, (&kernelContext_48)->params_0->scale_j_0, &kernelContext_48);
    float _S258 = read64_0(at_4 + 14U, (&kernelContext_48)->params_0->scale_j_0, &kernelContext_48);
    float _S259 = read64_0(at_4 + 16U, (&kernelContext_48)->params_0->scale_j_0, &kernelContext_48);
    float _S260 = read64_0(at_4 + 18U, (&kernelContext_48)->params_0->scale_j_0, &kernelContext_48);
    float dd_1 = dot(d_1, d_1);
    float _S261 = d_1.x;
    float _S262 = _S251 * _S261;
    float _S263 = d_1.y;
    float _S264 = _S258 + _S262 * _S263;
    float _S265 = d_1.z;
    float _S266 = _S259 + _S262 * _S265;
    float _S267 = _S260 + _S251 * _S263 * _S265;
    float3 com_1 = (float4(*((&kernelContext_48)->chunks_0+4U * c_11)) ).xyz + d_1;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S255 - _S251 * (dd_1 - _S261 * _S261), _S264, _S266), float3(_S264, _S256 - _S251 * (dd_1 - _S263 * _S263), _S267), float3(_S266, _S267, _S257 - _S251 * (dd_1 - _S265 * _S265)), &principal_0, &moments_1);
    uint _S268 = a_anchor_0(&kernelContext_48);
    uint _S269 = atomic_load_explicit((&kernelContext_48)->atom_0+(_S268 + c_11), memory_order_relaxed);
    bool anchored_0 = _S269 != 0U;
    thread Quat_0 q_13;
    thread float3 origin_2;
    structure_pose_0(_S249, pb_0, &q_13, &origin_2, &kernelContext_48);
    float3 p_parent_0 = (float4(*((&kernelContext_48)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_48)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_48)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S270 = q_13;
    float3 _S271 = rotate_0(&_S270, com_1);
    float3 pos_0 = to_rubble_0(origin_2 + _S271);
    thread Quat_0 _S272 = q_13;
    thread Quat_0 _S273 = principal_0;
    Quat_0 _S274 = quat_mul_0(&_S272, &_S273);
    thread Quat_0 _S275 = _S274;
    Quat_0 _S276 = normalized_0(&_S275);
    thread Quat_0 _S277 = _S276;
    Quat_0 _S278 = rot_to_rubble_0(&_S277);
    bool _S279;
    if(anchored_0)
    {
        _S279 = true;
    }
    else
    {
        _S279 = !(_S251 > 0.0f);
    }
    float inv_mass_1;
    if(_S279)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / _S251;
    }
    float3 _S280 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S281;
        if((moments_1.x) > 0.0f)
        {
            _S281 = 1.0f / moments_1.x;
        }
        else
        {
            _S281 = 0.0f;
        }
        float _S282;
        if((moments_1.y) > 0.0f)
        {
            _S282 = 1.0f / moments_1.y;
        }
        else
        {
            _S282 = 0.0f;
        }
        float _S283;
        if((moments_1.z) > 0.0f)
        {
            _S283 = 1.0f / moments_1.z;
        }
        else
        {
            _S283 = 0.0f;
        }
        inv_inertia_0 = float3(_S281, _S282, _S283);
    }
    else
    {
        inv_inertia_0 = _S280;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S280;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S280;
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
    uint _S284 = stage_0(c_11, &kernelContext_48);
    *((&kernelContext_48)->frames_0+_S284) = packed_float4(float4(pos_0, inv_mass_1)) ;
    packed_float4 device* _S285 = (&kernelContext_48)->frames_0+(_S284 + 1U);
    thread Quat_0 _S286 = _S278;
    float4 _S287 = quat_vec_0(&_S286);
    *_S285 = packed_float4(_S287) ;
    *((&kernelContext_48)->frames_0+(_S284 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_48)->frames_0+(_S284 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_48)->frames_0+(_S284 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_48)->params_0->step_0))))) ;
    *((&kernelContext_48)->frames_0+(_S284 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_48)->frames_0+(_S284 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_48)->frames_0+(_S284 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_48)->frames_0+2U * _S249)) ).w)) ;
    packed_float4 device* _S288 = (&kernelContext_48)->frames_0+(_S284 + 8U);
    thread Quat_0 _S289 = principal_0;
    float4 _S290 = quat_vec_0(&_S289);
    *_S288 = packed_float4(_S290) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], CrushParams_0 constant* crush_18 [[buffer(25)]], packed_float4 device* crush_rw_18 [[buffer(24)]], packed_float4 device* crush_tab_18 [[buffer(23)]])
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
    (&kernelContext_49)->crush_0 = crush_18;
    (&kernelContext_49)->crush_rw_0 = crush_rw_18;
    (&kernelContext_49)->crush_tab_0 = crush_tab_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_49)->g_scan_0 = &g_scan_18;
    uint c_12 = id_15.x;
    bool _S291 = is_root_0(c_12, &kernelContext_49);
    if(!_S291)
    {
        return;
    }
    uint _S292 = w_slot_0(&kernelContext_49);
    uint slot_3 = *((&kernelContext_49)->work_0+(_S292 + c_12));
    uint _S293 = stage_0(c_12, &kernelContext_49);
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
            *((&kernelContext_49)->bodies_0+(7U * (slot_3 + region_0 * (&kernelContext_49)->params_0->max_bodies_0) + w_3)) = packed_float4(float4(*((&kernelContext_49)->frames_0+(_S293 + w_3))) ) ;
            w_3 = w_3 + 1U;
        }
        region_0 = region_0 + 1U;
    }
    uint _S294 = 2U * slot_3;
    *((&kernelContext_49)->frames_0+_S294) = packed_float4(float4(*((&kernelContext_49)->frames_0+(_S293 + 7U))) ) ;
    *((&kernelContext_49)->frames_0+(_S294 + 1U)) = packed_float4(float4(*((&kernelContext_49)->frames_0+(_S293 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], CrushParams_0 constant* crush_19 [[buffer(25)]], packed_float4 device* crush_rw_19 [[buffer(24)]], packed_float4 device* crush_tab_19 [[buffer(23)]])
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
    (&kernelContext_50)->crush_0 = crush_19;
    (&kernelContext_50)->crush_rw_0 = crush_rw_19;
    (&kernelContext_50)->crush_tab_0 = crush_tab_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_50)->g_scan_0 = &g_scan_19;
    uint c_13 = id_16.x;
    bool _S295;
    if(c_13 >= (params_19->chunk_count_0))
    {
        _S295 = true;
    }
    else
    {
        uint _S296 = body_of_0(c_13, &kernelContext_50);
        bool _S297 = dirty_0(_S296, &kernelContext_50);
        _S295 = !_S297;
    }
    if(_S295)
    {
        return;
    }
    uint _S298 = w_label_0(&kernelContext_50);
    uint device* _S299 = (&kernelContext_50)->work_0+(_S298 + c_13);
    uint r_6 = *_S299;
    uint _S300 = w_slot_0(&kernelContext_50);
    uint device* _S301 = (&kernelContext_50)->work_0+(_S300 + *_S299);
    uint slot_4 = *_S301;
    *((&kernelContext_50)->work_0+(w_body_0() + c_13)) = *_S301;
    uint _S302 = 2U * slot_4;
    float3 _S303 = (float4(*((&kernelContext_50)->frames_0+_S302)) ).xyz;
    thread Quat_0 _S304 = quat_of_0(float4(*((&kernelContext_50)->frames_0+(_S302 + 1U))) );
    Quat_0 _S305 = conj_0(&_S304);
    float inv_mass_2 = (float4(*((&kernelContext_50)->bodies_0+7U * slot_4)) ).w;
    float _S306;
    if(inv_mass_2 > 0.0f)
    {
        _S306 = 1.0f / inv_mass_2;
    }
    else
    {
        _S306 = float_bits_0(2139095040U);
    }
    float4 _S307 = float4(*((&kernelContext_50)->chunks_0+(4U * c_13 + 2U))) ;
    uint first_1 = (as_type<uint>((_S307.z)));
    uint _S308 = (as_type<uint>((_S307.w)));
    float4 _S309 = float4(*((&kernelContext_50)->chunks_0+(4U * r_6 + 3U))) ;
    uint p_3 = first_1;
    for(;;)
    {
        if(p_3 < (first_1 + _S308))
        {
        }
        else
        {
            break;
        }
        uint _S310 = 2U * p_3;
        float4 _S311 = float4(*((&kernelContext_50)->parts_0+_S310)) ;
        uint col_0 = (as_type<uint>((_S311.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_50)->parts_0+(_S310 + 1U))) );
        uint _S312 = 4U * col_0;
        float4 _S313 = float4(*((&kernelContext_50)->colliders_0+_S312)) ;
        *((&kernelContext_50)->colliders_0+_S312) = packed_float4(float4((as_type<float>((slot_4))), _S313.y, _S313.z, _S313.w)) ;
        uint _S314 = _S312 + 1U;
        packed_float4 device* _S315 = (&kernelContext_50)->colliders_0+_S314;
        float3 _S316 = _S311.xyz - _S303;
        thread Quat_0 _S317 = _S305;
        float3 _S318 = rotate_0(&_S317, _S316);
        *_S315 = packed_float4(float4(_S318, (float4(*((&kernelContext_50)->colliders_0+_S314)) ).w)) ;
        packed_float4 device* _S319 = (&kernelContext_50)->colliders_0+(_S312 + 2U);
        thread Quat_0 _S320 = _S305;
        thread Quat_0 _S321 = rot_1;
        Quat_0 _S322 = quat_mul_0(&_S320, &_S321);
        thread Quat_0 _S323 = _S322;
        Quat_0 _S324 = normalized_0(&_S323);
        thread Quat_0 _S325 = _S324;
        float4 _S326 = quat_vec_0(&_S325);
        *_S319 = packed_float4(_S326) ;
        uint _S327 = 2U * col_0;
        uint4 _S328 = uint4(*((&kernelContext_50)->collider_info_0+_S327)) ;
        *((&kernelContext_50)->collider_info_0+_S327) = packed_uint4(uint4(_S328.x, slot_4, _S328.z, _S328.w)) ;
        *((&kernelContext_50)->collider_info_0+(_S327 + 1U)) = packed_uint4(uint4((as_type<uint>((_S306))), (as_type<uint>((_S309.x))), (as_type<uint>((_S309.w))), 0U)) ;
        p_3 = p_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], CrushParams_0 constant* crush_20 [[buffer(25)]], packed_float4 device* crush_rw_20 [[buffer(24)]], packed_float4 device* crush_tab_20 [[buffer(23)]])
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
    uint s_3 = id_17.x;
    if(s_3 >= (params_20->body_count_0))
    {
        return;
    }
    uint _S329 = a_dirty_0(&kernelContext_51);
    atomic_store_explicit((&kernelContext_51)->atom_0+(_S329 + s_3), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_0 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_7 = fract(h_0);
    float q_14 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_7);
    float t_4 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_7));
    uint k_3 = uint(h_0);
    float3 _S330;
    if(k_3 == 0U)
    {
        _S330 = float3(0.92000001668930054f, t_4, 0.34959998726844788f);
    }
    else
    {
        if(k_3 == 1U)
        {
            _S330 = float3(q_14, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_3 == 2U)
            {
                _S330 = float3(0.34959998726844788f, 0.92000001668930054f, t_4);
            }
            else
            {
                if(k_3 == 3U)
                {
                    _S330 = float3(0.34959998726844788f, q_14, 0.92000001668930054f);
                }
                else
                {
                    if(k_3 == 4U)
                    {
                        _S330 = float3(t_4, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S330 = float3(0.92000001668930054f, 0.34959998726844788f, q_14);
                    }
                }
            }
        }
    }
    return _S330;
}

void write_object_0(uint o_1, const Quat_0 thread* q_15, float3 t_5, float4 color_0, KernelContext_0 thread* kernelContext_52)
{
    float3 _S331 = rotate_0(q_15, float3(1.0f, 0.0f, 0.0f));
    float3 _S332 = rotate_0(q_15, float3(0.0f, 1.0f, 0.0f));
    float3 _S333 = rotate_0(q_15, float3(0.0f, 0.0f, 1.0f));
    uint _S334 = 4U * o_1;
    *(kernelContext_52->objects_0+_S334) = packed_float4(float4(_S331.x, _S332.x, _S333.x, t_5.x)) ;
    *(kernelContext_52->objects_0+(_S334 + 1U)) = packed_float4(float4(_S331.y, _S332.y, _S333.y, t_5.y)) ;
    *(kernelContext_52->objects_0+(_S334 + 2U)) = packed_float4(float4(_S331.z, _S332.z, _S333.z, t_5.z)) ;
    *(kernelContext_52->objects_0+(_S334 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], CrushParams_0 constant* crush_21 [[buffer(25)]], packed_float4 device* crush_rw_21 [[buffer(24)]], packed_float4 device* crush_tab_21 [[buffer(23)]])
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
    (&kernelContext_53)->crush_0 = crush_21;
    (&kernelContext_53)->crush_rw_0 = crush_rw_21;
    (&kernelContext_53)->crush_tab_0 = crush_tab_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_53)->g_scan_0 = &g_scan_21;
    uint k_4 = id_18.x;
    if(k_4 >= (params_21->chunk_count_0))
    {
        return;
    }
    uint _S335 = body_of_0(k_4, &kernelContext_53);
    uint b_4 = 7U * _S335;
    thread Quat_0 q_16;
    thread float3 origin_3;
    structure_pose_0(_S335, b_4, &q_16, &origin_3, &kernelContext_53);
    uint _S336 = 3U * k_4;
    float4 _S337 = float4(*((&kernelContext_53)->view_chunks_0+_S336)) ;
    uint _S338 = _S336 + 2U;
    float3 base_2 = (float4(*((&kernelContext_53)->view_chunks_0+_S338)) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_53)->bodies_0+b_4)) ).w) == 0.0f;
    bool _S339;
    if(((&kernelContext_53)->view_0->mode_0) == 0U)
    {
        _S339 = true;
    }
    else
    {
        _S339 = ((&kernelContext_53)->view_0->mode_0) == 3U;
    }
    float surface_0;
    if(_S339)
    {
        surface_0 = max((float4(*((&kernelContext_53)->view_chunks_0+_S338)) ).w, 1.0f);
    }
    else
    {
        surface_0 = 1.0f;
    }
    float3 color_1;
    if(((&kernelContext_53)->view_0->mode_0) == 1U)
    {
        float _S340 = max(length((float4(*((&kernelContext_53)->bodies_0+(b_4 + 3U))) ).xyz), length((float4(*((&kernelContext_53)->bodies_0+(b_4 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S340 > ((&kernelContext_53)->view_0->rest_speed_0))
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
        if(((&kernelContext_53)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S335);
        }
        else
        {
            if(((&kernelContext_53)->view_0->mode_0) == 3U)
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
                    color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S335);
                }
            }
        }
    }
    uint _S341 = (as_type<uint>((_S337.w)));
    Quat_0 _S342 = quat_of_0(float4(*((&kernelContext_53)->view_chunks_0+(_S336 + 1U))) );
    thread Quat_0 _S343 = q_16;
    thread Quat_0 _S344 = _S342;
    Quat_0 _S345 = quat_mul_0(&_S343, &_S344);
    thread Quat_0 _S346 = _S345;
    Quat_0 _S347 = normalized_0(&_S346);
    float3 _S348 = _S337.xyz;
    thread Quat_0 _S349 = q_16;
    float3 _S350 = rotate_0(&_S349, _S348);
    float3 _S351 = origin_3 + _S350;
    float4 _S352 = float4(color_1, surface_0);
    thread Quat_0 _S353 = _S347;
    write_object_0(_S341, &_S353, _S351, _S352, &kernelContext_53);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], CrushParams_0 constant* crush_22 [[buffer(25)]], packed_float4 device* crush_rw_22 [[buffer(24)]], packed_float4 device* crush_tab_22 [[buffer(23)]])
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
    (&kernelContext_54)->crush_0 = crush_22;
    (&kernelContext_54)->crush_rw_0 = crush_rw_22;
    (&kernelContext_54)->crush_tab_0 = crush_tab_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_54)->g_scan_0 = &g_scan_22;
    uint i_4 = id_19.x;
    if(i_4 >= (view_22->shot_count_1))
    {
        return;
    }
    uint b_5 = 7U * (&kernelContext_54)->shot_bodies_0[i_4];
    uint o_2 = (&kernelContext_54)->view_0->impactor_base_0 + i_4;
    if(((float4(*((&kernelContext_54)->bodies_0+b_5)) ).w) == 0.0f)
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
            *((&kernelContext_54)->objects_0+(4U * o_2 + w_4)) = packed_float4(float4(0.0f) ) ;
            w_4 = w_4 + 1U;
        }
        return;
    }
    thread Quat_0 _S354 = quat_of_0(float4(*((&kernelContext_54)->bodies_0+(b_5 + 1U))) );
    Quat_0 _S355 = rot_from_rubble_0(&_S354);
    float3 _S356 = from_rubble_0((float4(*((&kernelContext_54)->bodies_0+b_5)) ).xyz);
    float4 _S357 = float4(0.30000001192092896f, 0.31000000238418579f, 0.34000000357627869f, 4.0f);
    thread Quat_0 _S358 = _S355;
    write_object_0(o_2, &_S358, _S356, _S357, &kernelContext_54);
    return;
}

float3 chunk_world_0(uint k_5, KernelContext_0 thread* kernelContext_55)
{
    uint _S359 = body_of_0(k_5, kernelContext_55);
    thread Quat_0 q_17;
    thread float3 origin_4;
    structure_pose_0(_S359, 7U * _S359, &q_17, &origin_4, kernelContext_55);
    float3 _S360 = (float4(*(kernelContext_55->view_chunks_0+3U * k_5)) ).xyz;
    thread Quat_0 _S361 = q_17;
    float3 _S362 = rotate_0(&_S361, _S360);
    return origin_4 + _S362;
}

void write_line_vertex_0(uint v_9, float3 p_4, KernelContext_0 thread* kernelContext_56)
{
    uint _S363 = 7U * v_9;
    *(kernelContext_56->lines_0+_S363) = p_4.x;
    *(kernelContext_56->lines_0+(_S363 + 1U)) = p_4.y;
    *(kernelContext_56->lines_0+(_S363 + 2U)) = p_4.z;
    *(kernelContext_56->lines_0+(_S363 + 3U)) = 1.0f;
    *(kernelContext_56->lines_0+(_S363 + 4U)) = 0.86000001430511475f;
    *(kernelContext_56->lines_0+(_S363 + 5U)) = 0.25f;
    *(kernelContext_56->lines_0+(_S363 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], atomic<uint> device* atom_23 [[buffer(5)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], CrushParams_0 constant* crush_23 [[buffer(25)]], packed_float4 device* crush_rw_23 [[buffer(24)]], packed_float4 device* crush_tab_23 [[buffer(23)]])
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
    (&kernelContext_57)->crush_0 = crush_23;
    (&kernelContext_57)->crush_rw_0 = crush_rw_23;
    (&kernelContext_57)->crush_tab_0 = crush_tab_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_57)->g_scan_0 = &g_scan_23;
    uint i_5 = id_20.x;
    if(i_5 >= (params_23->bond_count_0))
    {
        return;
    }
    uint4 _S364 = uint4(*((&kernelContext_57)->bonds_0+i_5)) ;
    float3 _S365 = chunk_world_0(_S364.x, &kernelContext_57);
    uint _S366 = w_alive_0(&kernelContext_57);
    float3 b_6;
    if((*((&kernelContext_57)->work_0+(_S366 + i_5))) != 0U)
    {
        float3 _S367 = chunk_world_0(_S364.y, &kernelContext_57);
        b_6 = _S367;
    }
    else
    {
        b_6 = _S365;
    }
    uint _S368 = 2U * i_5;
    write_line_vertex_0(_S368, _S365, &kernelContext_57);
    write_line_vertex_0(_S368 + 1U, b_6, &kernelContext_57);
    return;
}

uint a_strike_0(KernelContext_0 thread* kernelContext_58)
{
    uint _S369 = a_counters_1(kernelContext_58);
    return _S369 + 16U;
}

uint strike_at_0(uint b_7, KernelContext_0 thread* kernelContext_59)
{
    return 2U * kernelContext_59->params_0->body_count_0 + 9U * kernelContext_59->params_0->chunk_count_0 + 3U * b_7;
}

[[kernel]] void fr_strike_reset(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], atomic<uint> device* atom_24 [[buffer(5)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], CrushParams_0 constant* crush_24 [[buffer(25)]], packed_float4 device* crush_rw_24 [[buffer(24)]], packed_float4 device* crush_tab_24 [[buffer(23)]])
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
    uint b_8 = id_21.x;
    if(b_8 >= (params_24->body_count_0))
    {
        return;
    }
    uint _S370 = a_strike_0(&kernelContext_60);
    uint at_5 = _S370 + 9U * b_8;
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
        atomic_store_explicit((&kernelContext_60)->atom_0+(at_5 + w_5), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_60)->atom_0+(at_5 + 3U + w_5), 0U, memory_order_relaxed);
        w_5 = w_5 + 1U;
    }
    atomic_store_explicit((&kernelContext_60)->atom_0+(at_5 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_60)->atom_0+(at_5 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_60)->atom_0+(at_5 + 8U), 4294967295U, memory_order_relaxed);
    uint _S371 = strike_at_0(b_8, &kernelContext_60);
    (*((&kernelContext_60)->frames_0+_S371))[int(3)] = 0.0f;
    return;
}

float3 chunk_box_world_0(uint k_6, Quat_0 thread* q_18, KernelContext_0 thread* kernelContext_61)
{
    uint _S372 = body_of_0(k_6, kernelContext_61);
    thread float3 origin_5;
    structure_pose_0(_S372, 7U * _S372, q_18, &origin_5, kernelContext_61);
    float3 _S373 = (float4(*(kernelContext_61->geometry_0+3U * k_6)) ).xyz;
    thread Quat_0 _S374 = *q_18;
    float3 _S375 = rotate_0(&_S374, _S373);
    return origin_5 + _S375;
}

uint ordered_0(float f_8)
{
    uint u_0 = (as_type<uint>((f_8)));
    uint _S376;
    if((u_0 & 2147483648U) != 0U)
    {
        _S376 = ~u_0;
    }
    else
    {
        _S376 = u_0 | 2147483648U;
    }
    return _S376;
}

[[kernel]] void fr_strike_bounds(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], atomic<uint> device* atom_25 [[buffer(5)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], CrushParams_0 constant* crush_25 [[buffer(25)]], packed_float4 device* crush_rw_25 [[buffer(24)]], packed_float4 device* crush_tab_25 [[buffer(23)]])
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
    (&kernelContext_62)->crush_0 = crush_25;
    (&kernelContext_62)->crush_rw_0 = crush_rw_25;
    (&kernelContext_62)->crush_tab_0 = crush_tab_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_62)->g_scan_0 = &g_scan_25;
    uint k_7 = id_22.x;
    bool _S377;
    if(k_7 >= (params_25->chunk_count_0))
    {
        _S377 = true;
    }
    else
    {
        _S377 = ((&kernelContext_62)->params_0->shot_count_0) == 0U;
    }
    if(_S377)
    {
        return;
    }
    thread Quat_0 q_19;
    float3 _S378 = chunk_box_world_0(k_7, &q_19, &kernelContext_62);
    float _S379 = (float4(*((&kernelContext_62)->geometry_0+3U * k_7)) ).w;
    uint _S380 = a_strike_0(&kernelContext_62);
    uint _S381 = body_of_0(k_7, &kernelContext_62);
    uint _S382 = _S380 + 9U * _S381;
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
        uint _S383 = ax_0;
        uint _S384 = atomic_fetch_min_explicit((&kernelContext_62)->atom_0+(_S382 + ax_0), ordered_0(_S378[ax_0] - _S379), memory_order_relaxed);
        uint _S385 = atomic_fetch_max_explicit((&kernelContext_62)->atom_0+(_S382 + 3U + ax_0), ordered_0(_S378[_S383] + _S379), memory_order_relaxed);
        ax_0 = ax_0 + 1U;
    }
    return;
}

float unordered_0(uint u_1)
{
    uint _S386;
    if((u_1 & 2147483648U) != 0U)
    {
        _S386 = u_1 & 2147483647U;
    }
    else
    {
        _S386 = ~u_1;
    }
    return (as_type<float>((_S386)));
}

void shot_motion_0(uint k_8, float3 thread* c_14, float3 thread* vs_0, Quat_0 thread* q_20, KernelContext_0 thread* kernelContext_63)
{
    uint _S387 = 4U * k_8;
    uint slot_6 = (as_type<uint>(((float4(*(kernelContext_63->shots_0+_S387)) ).x)));
    uint sb_0 = 7U * slot_6;
    *c_14 = from_rubble_0((float4(*(kernelContext_63->bodies_0+sb_0)) ).xyz);
    *vs_0 = from_rubble_0((float4(*(kernelContext_63->bodies_0+(sb_0 + 3U))) ).xyz);
    uint _S388 = sb_0 + 1U;
    thread Quat_0 _S389 = quat_of_0(float4(*(kernelContext_63->bodies_0+_S388)) );
    Quat_0 _S390 = rot_from_rubble_0(&_S389);
    *q_20 = _S390;
    float4 _S391 = float4(*(kernelContext_63->shots_0+(_S387 + 2U))) ;
    float _S392 = _S391.w;
    if(_S392 == 0.0f)
    {
        return;
    }
    float3 p_5;
    if(_S392 == 1.0f)
    {
        thread Quat_0 qs_0;
        thread float3 os_0;
        structure_pose_0(slot_6, sb_0, &qs_0, &os_0, kernelContext_63);
        float3 _S393 = _S391.xyz;
        thread Quat_0 _S394 = qs_0;
        float3 _S395 = rotate_0(&_S394, _S393);
        p_5 = os_0 + _S395;
    }
    else
    {
        Quat_0 qb_0 = quat_of_0(float4(*(kernelContext_63->bodies_0+_S388)) );
        float3 _S396 = (float4(*(kernelContext_63->bodies_0+sb_0)) ).xyz;
        float3 _S397 = _S391.xyz;
        thread Quat_0 _S398 = qb_0;
        float3 _S399 = rotate_0(&_S398, _S397);
        float3 _S400 = from_rubble_0(_S396 + _S399);
        Quat_0 _S401 = quat_of_0(float4(*(kernelContext_63->shots_0+(_S387 + 3U))) );
        thread Quat_0 _S402 = qb_0;
        thread Quat_0 _S403 = _S401;
        Quat_0 _S404 = quat_mul_0(&_S402, &_S403);
        thread Quat_0 _S405 = _S404;
        Quat_0 _S406 = rot_from_rubble_0(&_S405);
        thread Quat_0 _S407 = _S406;
        Quat_0 _S408 = normalized_0(&_S407);
        *q_20 = _S408;
        p_5 = _S400;
    }
    *vs_0 = *vs_0 + cross(from_rubble_0((float4(*(kernelContext_63->bodies_0+(sb_0 + 4U))) ).xyz), p_5 - *c_14);
    *c_14 = p_5;
    return;
}

[[kernel]] void fr_strike_find(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], atomic<uint> device* atom_26 [[buffer(5)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], CrushParams_0 constant* crush_26 [[buffer(25)]], packed_float4 device* crush_rw_26 [[buffer(24)]], packed_float4 device* crush_tab_26 [[buffer(23)]])
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
    uint b_9 = id_23.x;
    bool _S409;
    if(b_9 >= (params_26->body_count_0))
    {
        _S409 = true;
    }
    else
    {
        _S409 = ((&kernelContext_64)->params_0->shot_count_0) == 0U;
    }
    if(_S409)
    {
        return;
    }
    uint _S410 = a_strike_0(&kernelContext_64);
    uint at_6 = _S410 + 9U * b_9;
    uint _S411 = atomic_load_explicit((&kernelContext_64)->atom_0+at_6, memory_order_relaxed);
    if(_S411 == 4294967295U)
    {
        return;
    }
    uint _S412 = atomic_load_explicit((&kernelContext_64)->atom_0+at_6, memory_order_relaxed);
    float _S413 = unordered_0(_S412);
    uint _S414 = atomic_load_explicit((&kernelContext_64)->atom_0+(at_6 + 1U), memory_order_relaxed);
    float _S415 = unordered_0(_S414);
    uint _S416 = atomic_load_explicit((&kernelContext_64)->atom_0+(at_6 + 2U), memory_order_relaxed);
    float3 _S417 = float3(_S413, _S415, unordered_0(_S416));
    uint _S418 = atomic_load_explicit((&kernelContext_64)->atom_0+(at_6 + 3U), memory_order_relaxed);
    float _S419 = unordered_0(_S418);
    uint _S420 = atomic_load_explicit((&kernelContext_64)->atom_0+(at_6 + 4U), memory_order_relaxed);
    float _S421 = unordered_0(_S420);
    uint _S422 = atomic_load_explicit((&kernelContext_64)->atom_0+(at_6 + 5U), memory_order_relaxed);
    float3 _S423 = float3(_S419, _S421, unordered_0(_S422));
    float3 _S424 = from_rubble_0((float4(*((&kernelContext_64)->bodies_0+(7U * b_9 + 3U))) ).xyz);
    float _S425 = float_bits_0(2139095040U);
    uint k_9 = 0U;
    for(;;)
    {
        if(k_9 < ((&kernelContext_64)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S426 = float4(*((&kernelContext_64)->shots_0+4U * k_9)) ;
        uint slot_7 = (as_type<uint>((_S426.x)));
        if(((float4(*((&kernelContext_64)->bodies_0+7U * slot_7)) ).w) == 0.0f)
        {
            _S409 = true;
        }
        else
        {
            _S409 = slot_7 == b_9;
        }
        if(_S409)
        {
            k_9 = k_9 + 1U;
            continue;
        }
        thread float3 c_15;
        thread float3 vs_1;
        thread Quat_0 qs_1;
        shot_motion_0(k_9, &c_15, &vs_1, &qs_1, &kernelContext_64);
        float3 rel_0 = vs_1 - _S424;
        if(!((length(rel_0)) > ((&kernelContext_64)->params_0->strike_speed_0)))
        {
            k_9 = k_9 + 1U;
            continue;
        }
        float3 _S427 = float3(_S426.y) ;
        float3 _S428 = _S417 - _S427;
        float3 _S429 = _S423 + _S427;
        float _S430 = (&kernelContext_64)->params_0->dt_0;
        float t0_0 = 0.0f;
        float t1_0 = _S430;
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
            uint _S431 = ax_1;
            if((abs(rel_0[ax_1])) < 9.999999960041972e-13f)
            {
                bool _S432;
                if((c_15[ax_1]) < (_S428[ax_1]))
                {
                    _S432 = true;
                }
                else
                {
                    _S432 = (c_15[ax_1]) > (_S429[ax_1]);
                }
                float t0_1;
                if(_S432)
                {
                    t0_1 = _S425;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_1 = ax_1 + 1U;
                continue;
            }
            float ta_0 = (_S428[ax_1] - c_15[ax_1]) / rel_0[_S431];
            float tb_0 = (_S429[ax_1] - c_15[ax_1]) / rel_0[_S431];
            float _S433 = min(t1_0, max(ta_0, tb_0));
            t0_0 = max(t0_0, min(ta_0, tb_0));
            t1_0 = _S433;
            ax_1 = ax_1 + 1U;
        }
        if(t0_0 <= t1_0)
        {
            uint _S434 = strike_at_0(b_9, &kernelContext_64);
            *((&kernelContext_64)->frames_0+_S434) = packed_float4(float4(c_15, float(k_9 + 1U))) ;
            *((&kernelContext_64)->frames_0+(_S434 + 1U)) = packed_float4(float4(rel_0, t0_0)) ;
            return;
        }
        k_9 = k_9 + 1U;
    }
    return;
}

bool struck_0(uint b_10, uint thread* shot_0, KernelContext_0 thread* kernelContext_65)
{
    uint _S435 = strike_at_0(b_10, kernelContext_65);
    uint k_10 = uint((float4(*(kernelContext_65->frames_0+_S435)) ).w);
    *shot_0 = k_10 - 1U;
    return k_10 != 0U;
}

bool shot_is_box_0(uint k_11, KernelContext_0 thread* kernelContext_66)
{
    return ((float4(*(kernelContext_66->shots_0+(4U * k_11 + 2U))) ).w) == 2.0f;
}

float box_extent_0(const Quat_0 thread* q_21, float3 half_1, float3 axis_0)
{
    float _S436 = half_1.x;
    float3 _S437 = rotate_1(q_21, float3(1.0f, 0.0f, 0.0f));
    float _S438 = _S436 * abs(dot(_S437, axis_0));
    float _S439 = half_1.y;
    float3 _S440 = rotate_1(q_21, float3(0.0f, 1.0f, 0.0f));
    float _S441 = _S438 + _S439 * abs(dot(_S440, axis_0));
    float _S442 = half_1.z;
    float3 _S443 = rotate_1(q_21, float3(0.0f, 0.0f, 1.0f));
    return _S441 + _S442 * abs(dot(_S443, axis_0));
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

uint a_strike_dist_0(KernelContext_0 thread* kernelContext_67)
{
    uint _S444 = a_strike_0(kernelContext_67);
    return _S444 + 9U * kernelContext_67->params_0->body_count_0;
}

[[kernel]] void fr_strike_entry(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], atomic<uint> device* atom_27 [[buffer(5)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], CrushParams_0 constant* crush_27 [[buffer(25)]], packed_float4 device* crush_rw_27 [[buffer(24)]], packed_float4 device* crush_tab_27 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_68;
    (&kernelContext_68)->params_0 = params_27;
    (&kernelContext_68)->work_0 = work_27;
    (&kernelContext_68)->frames_0 = frames_27;
    (&kernelContext_68)->bodies_0 = bodies_27;
    (&kernelContext_68)->chunks_0 = chunks_27;
    (&kernelContext_68)->sort_0 = sort_27;
    (&kernelContext_68)->atom_0 = atom_27;
    (&kernelContext_68)->manifolds_0 = manifolds_27;
    (&kernelContext_68)->loads_0 = loads_27;
    (&kernelContext_68)->bonds_0 = bonds_27;
    (&kernelContext_68)->args_0 = args_27;
    (&kernelContext_68)->info_0 = info_27;
    (&kernelContext_68)->scan_0 = scan_27;
    (&kernelContext_68)->parts_0 = parts_27;
    (&kernelContext_68)->colliders_0 = colliders_27;
    (&kernelContext_68)->collider_info_0 = collider_info_27;
    (&kernelContext_68)->view_chunks_0 = view_chunks_27;
    (&kernelContext_68)->view_0 = view_27;
    (&kernelContext_68)->objects_0 = objects_27;
    (&kernelContext_68)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_68)->lines_0 = lines_27;
    (&kernelContext_68)->geometry_0 = geometry_27;
    (&kernelContext_68)->shots_0 = shots_27;
    (&kernelContext_68)->crush_0 = crush_27;
    (&kernelContext_68)->crush_rw_0 = crush_rw_27;
    (&kernelContext_68)->crush_tab_0 = crush_tab_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_68)->g_scan_0 = &g_scan_27;
    uint k_12 = id_24.x;
    bool _S445;
    if(k_12 >= (params_27->chunk_count_0))
    {
        _S445 = true;
    }
    else
    {
        _S445 = ((&kernelContext_68)->params_0->shot_count_0) == 0U;
    }
    if(_S445)
    {
        return;
    }
    uint _S446 = body_of_0(k_12, &kernelContext_68);
    thread uint shot_1;
    bool _S447 = struck_0(_S446, &shot_1, &kernelContext_68);
    if(!_S447)
    {
        return;
    }
    uint _S448 = strike_at_0(_S446, &kernelContext_68);
    float4 _S449 = float4(*((&kernelContext_68)->frames_0+(_S448 + 1U))) ;
    float3 from_1 = (float4(*((&kernelContext_68)->frames_0+_S448)) ).xyz;
    float3 rel_2 = _S449.xyz;
    float3 d_2 = rel_2 / float3(max(length(rel_2), 9.999999960041972e-13f)) ;
    float _S450 = (float4(*((&kernelContext_68)->shots_0+4U * shot_1)) ).y;
    thread Quat_0 q_22;
    float3 _S451 = chunk_box_world_0(k_12, &q_22, &kernelContext_68);
    uint _S452 = 3U * k_12;
    Quat_0 _S453 = quat_of_0(float4(*((&kernelContext_68)->geometry_0+(_S452 + 2U))) );
    thread Quat_0 _S454 = q_22;
    thread Quat_0 _S455 = _S453;
    Quat_0 _S456 = quat_mul_0(&_S454, &_S455);
    thread Quat_0 _S457 = _S456;
    Quat_0 _S458 = normalized_0(&_S457);
    float3 _S459 = (float4(*((&kernelContext_68)->geometry_0+(_S452 + 1U))) ).xyz;
    float _S460 = float_bits_0(2139095040U);
    thread float t_in_1 = 0.0f;
    thread float t_out_1 = (&kernelContext_68)->params_0->dt_0;
    bool _S461 = shot_is_box_0(shot_1, &kernelContext_68);
    uint i_6;
    if(_S461)
    {
        thread float3 c_16;
        thread float3 vs_2;
        thread Quat_0 qs_2;
        shot_motion_0(shot_1, &c_16, &vs_2, &qs_2, &kernelContext_68);
        float3 _S462 = (float4(*((&kernelContext_68)->shots_0+(4U * shot_1 + 1U))) ).yzw;
        float3 _S463 = from_1 - _S451;
        i_6 = 0U;
        for(;;)
        {
            if(i_6 < 3U)
            {
            }
            else
            {
                break;
            }
            float3 _S464 = float3(0.0f) ;
            thread float3 e_1 = _S464;
            e_1[i_6] = 1.0f;
            thread Quat_0 _S465 = _S458;
            float3 _S466 = rotate_0(&_S465, e_1);
            thread Quat_0 _S467 = qs_2;
            float3 _S468 = rotate_0(&_S467, e_1);
            thread Quat_0 _S469 = qs_2;
            float _S470 = box_extent_0(&_S469, _S462, _S466);
            sweep_axis_0(_S466, _S463, rel_2, _S459[i_6] + _S470, &t_in_1, &t_out_1);
            thread Quat_0 _S471 = _S458;
            float _S472 = box_extent_0(&_S471, _S459, _S468);
            sweep_axis_0(_S468, _S463, rel_2, _S472 + _S462[i_6], &t_in_1, &t_out_1);
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
                thread float3 f_9 = _S464;
                f_9[j_3] = 1.0f;
                thread Quat_0 _S473 = qs_2;
                float3 _S474 = rotate_0(&_S473, f_9);
                float3 n_3 = cross(_S466, _S474);
                float len_0 = length(n_3);
                if(len_0 < 9.99999997475242708e-07f)
                {
                    j_3 = j_3 + 1U;
                    continue;
                }
                float3 n_4 = n_3 / float3(len_0) ;
                thread Quat_0 _S475 = _S458;
                float _S476 = box_extent_0(&_S475, _S459, n_4);
                thread Quat_0 _S477 = qs_2;
                float _S478 = box_extent_0(&_S477, _S462, n_4);
                sweep_axis_0(n_4, _S463, rel_2, _S476 + _S478, &t_in_1, &t_out_1);
                j_3 = j_3 + 1U;
            }
            i_6 = i_6 + 1U;
        }
    }
    else
    {
        i_6 = 0U;
        for(;;)
        {
            if(i_6 < 3U)
            {
            }
            else
            {
                break;
            }
            thread float3 e_2 = float3(0.0f) ;
            e_2[i_6] = 1.0f;
            thread Quat_0 _S479 = _S458;
            float3 _S480 = rotate_0(&_S479, e_2);
            float o_4 = dot(from_1 - _S451, _S480);
            float dv_1 = dot(rel_2, _S480);
            float ext_1 = _S459[i_6] + _S450;
            if((abs(dv_1)) < 9.999999960041972e-13f)
            {
                if((abs(o_4)) > ext_1)
                {
                    t_in_1 = _S460;
                }
                i_6 = i_6 + 1U;
                continue;
            }
            float ta_2 = (- ext_1 - o_4) / dv_1;
            float tb_2 = (ext_1 - o_4) / dv_1;
            t_in_1 = max(t_in_1, min(ta_2, tb_2));
            t_out_1 = min(t_out_1, max(ta_2, tb_2));
            i_6 = i_6 + 1U;
        }
    }
    uint _S481 = a_strike_0(&kernelContext_68);
    uint at_7 = _S481 + 9U * _S446;
    if(t_in_1 <= t_out_1)
    {
        uint _S482 = atomic_fetch_min_explicit((&kernelContext_68)->atom_0+(at_7 + 6U), (as_type<uint>((t_in_1))), memory_order_relaxed);
    }
    float3 r_7 = _S451 - (from_1 + rel_2 * float3(_S449.w) );
    float along_0 = dot(r_7, d_2);
    float dist_0 = length(r_7 - d_2 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f;
    uint _S483 = a_strike_dist_0(&kernelContext_68);
    uint _S484 = (as_type<uint>((dist_0)));
    atomic_store_explicit((&kernelContext_68)->atom_0+(_S483 + k_12), _S484, memory_order_relaxed);
    uint _S485 = atomic_fetch_min_explicit((&kernelContext_68)->atom_0+(at_7 + 7U), _S484, memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], atomic<uint> device* atom_28 [[buffer(5)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], CrushParams_0 constant* crush_28 [[buffer(25)]], packed_float4 device* crush_rw_28 [[buffer(24)]], packed_float4 device* crush_tab_28 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_69;
    (&kernelContext_69)->params_0 = params_28;
    (&kernelContext_69)->work_0 = work_28;
    (&kernelContext_69)->frames_0 = frames_28;
    (&kernelContext_69)->bodies_0 = bodies_28;
    (&kernelContext_69)->chunks_0 = chunks_28;
    (&kernelContext_69)->sort_0 = sort_28;
    (&kernelContext_69)->atom_0 = atom_28;
    (&kernelContext_69)->manifolds_0 = manifolds_28;
    (&kernelContext_69)->loads_0 = loads_28;
    (&kernelContext_69)->bonds_0 = bonds_28;
    (&kernelContext_69)->args_0 = args_28;
    (&kernelContext_69)->info_0 = info_28;
    (&kernelContext_69)->scan_0 = scan_28;
    (&kernelContext_69)->parts_0 = parts_28;
    (&kernelContext_69)->colliders_0 = colliders_28;
    (&kernelContext_69)->collider_info_0 = collider_info_28;
    (&kernelContext_69)->view_chunks_0 = view_chunks_28;
    (&kernelContext_69)->view_0 = view_28;
    (&kernelContext_69)->objects_0 = objects_28;
    (&kernelContext_69)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_69)->lines_0 = lines_28;
    (&kernelContext_69)->geometry_0 = geometry_28;
    (&kernelContext_69)->shots_0 = shots_28;
    (&kernelContext_69)->crush_0 = crush_28;
    (&kernelContext_69)->crush_rw_0 = crush_rw_28;
    (&kernelContext_69)->crush_tab_0 = crush_tab_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_69)->g_scan_0 = &g_scan_28;
    uint k_13 = id_25.x;
    bool _S486;
    if(k_13 >= (params_28->chunk_count_0))
    {
        _S486 = true;
    }
    else
    {
        _S486 = ((&kernelContext_69)->params_0->shot_count_0) == 0U;
    }
    if(_S486)
    {
        return;
    }
    uint _S487 = body_of_0(k_13, &kernelContext_69);
    thread uint shot_2;
    bool _S488 = struck_0(_S487, &shot_2, &kernelContext_69);
    if(!_S488)
    {
        return;
    }
    uint _S489 = a_strike_0(&kernelContext_69);
    uint at_8 = _S489 + 9U * _S487;
    uint _S490 = a_strike_dist_0(&kernelContext_69);
    uint _S491 = atomic_load_explicit((&kernelContext_69)->atom_0+(_S490 + k_13), memory_order_relaxed);
    uint _S492 = atomic_load_explicit((&kernelContext_69)->atom_0+(at_8 + 7U), memory_order_relaxed);
    if(_S491 == _S492)
    {
        uint _S493 = atomic_fetch_min_explicit((&kernelContext_69)->atom_0+(at_8 + 8U), k_13, memory_order_relaxed);
    }
    return;
}

float shot_lead_0(uint k_14, float3 d_3, KernelContext_0 thread* kernelContext_70)
{
    bool _S494 = shot_is_box_0(k_14, kernelContext_70);
    if(!_S494)
    {
        return (float4(*(kernelContext_70->shots_0+4U * k_14)) ).y;
    }
    thread float3 c_17;
    thread float3 vs_3;
    thread Quat_0 qs_3;
    shot_motion_0(k_14, &c_17, &vs_3, &qs_3, kernelContext_70);
    float3 _S495 = (float4(*(kernelContext_70->shots_0+(4U * k_14 + 1U))) ).yzw;
    thread Quat_0 _S496 = qs_3;
    float _S497 = box_extent_0(&_S496, _S495, d_3);
    return _S497;
}

[[kernel]] void fr_strike_force(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], atomic<uint> device* atom_29 [[buffer(5)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], CrushParams_0 constant* crush_29 [[buffer(25)]], packed_float4 device* crush_rw_29 [[buffer(24)]], packed_float4 device* crush_tab_29 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_71;
    (&kernelContext_71)->params_0 = params_29;
    (&kernelContext_71)->work_0 = work_29;
    (&kernelContext_71)->frames_0 = frames_29;
    (&kernelContext_71)->bodies_0 = bodies_29;
    (&kernelContext_71)->chunks_0 = chunks_29;
    (&kernelContext_71)->sort_0 = sort_29;
    (&kernelContext_71)->atom_0 = atom_29;
    (&kernelContext_71)->manifolds_0 = manifolds_29;
    (&kernelContext_71)->loads_0 = loads_29;
    (&kernelContext_71)->bonds_0 = bonds_29;
    (&kernelContext_71)->args_0 = args_29;
    (&kernelContext_71)->info_0 = info_29;
    (&kernelContext_71)->scan_0 = scan_29;
    (&kernelContext_71)->parts_0 = parts_29;
    (&kernelContext_71)->colliders_0 = colliders_29;
    (&kernelContext_71)->collider_info_0 = collider_info_29;
    (&kernelContext_71)->view_chunks_0 = view_chunks_29;
    (&kernelContext_71)->view_0 = view_29;
    (&kernelContext_71)->objects_0 = objects_29;
    (&kernelContext_71)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_71)->lines_0 = lines_29;
    (&kernelContext_71)->geometry_0 = geometry_29;
    (&kernelContext_71)->shots_0 = shots_29;
    (&kernelContext_71)->crush_0 = crush_29;
    (&kernelContext_71)->crush_rw_0 = crush_rw_29;
    (&kernelContext_71)->crush_tab_0 = crush_tab_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_71)->g_scan_0 = &g_scan_29;
    uint b_11 = id_26.x;
    bool _S498;
    if(b_11 >= (params_29->body_count_0))
    {
        _S498 = true;
    }
    else
    {
        _S498 = ((&kernelContext_71)->params_0->shot_count_0) == 0U;
    }
    if(_S498)
    {
        return;
    }
    thread uint shot_3;
    bool _S499 = struck_0(b_11, &shot_3, &kernelContext_71);
    if(!_S499)
    {
        return;
    }
    uint _S500 = a_strike_0(&kernelContext_71);
    uint at_9 = _S500 + 9U * b_11;
    uint _S501 = atomic_load_explicit((&kernelContext_71)->atom_0+(at_9 + 6U), memory_order_relaxed);
    float t_in_2 = (as_type<float>((_S501)));
    uint entry_0 = atomic_load_explicit((&kernelContext_71)->atom_0+(at_9 + 8U), memory_order_relaxed);
    uint _S502 = strike_at_0(b_11, &kernelContext_71);
    if(!(t_in_2 <= ((&kernelContext_71)->params_0->dt_0)))
    {
        _S498 = true;
    }
    else
    {
        _S498 = entry_0 == 4294967295U;
    }
    if(_S498)
    {
        (*((&kernelContext_71)->frames_0+_S502))[int(3)] = 0.0f;
        return;
    }
    float4 _S503 = float4(*((&kernelContext_71)->frames_0+_S502)) ;
    float3 rel_3 = (float4(*((&kernelContext_71)->frames_0+(_S502 + 1U))) ).xyz;
    float speed_0 = length(rel_3);
    float3 d_4 = rel_3 / float3(max(speed_0, 9.999999960041972e-13f)) ;
    float4 _S504 = float4(*((&kernelContext_71)->shots_0+4U * shot_3)) ;
    float _S505 = shot_lead_0(shot_3, d_4, &kernelContext_71);
    float _S506 = max(_S504.z, 0.00100000004749745f);
    float m_imp_0 = _S504.w;
    float e_imp_0 = (float4(*((&kernelContext_71)->shots_0+(4U * shot_3 + 1U))) ).x;
    float3 contact_0 = _S503.xyz + rel_3 * float3(t_in_2)  + d_4 * float3(_S505) ;
    float inv_mass_3 = (float4(*((&kernelContext_71)->bodies_0+7U * b_11)) ).w;
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
    float modulus_0 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_71)->geometry_0+(3U * entry_0 + 1U))) ).w);
    *((&kernelContext_71)->frames_0+(_S502 + 2U)) = packed_float4(float4(contact_0, m_eff_0 * speed_0 / (2.86999988555908203f * pow(m_eff_0 * m_eff_0 / (_S506 * modulus_0 * modulus_0 * max(speed_0, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

[[kernel]] void fr_strike_breaks(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], atomic<uint> device* atom_30 [[buffer(5)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], CrushParams_0 constant* crush_30 [[buffer(25)]], packed_float4 device* crush_rw_30 [[buffer(24)]], packed_float4 device* crush_tab_30 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_72;
    (&kernelContext_72)->params_0 = params_30;
    (&kernelContext_72)->work_0 = work_30;
    (&kernelContext_72)->frames_0 = frames_30;
    (&kernelContext_72)->bodies_0 = bodies_30;
    (&kernelContext_72)->chunks_0 = chunks_30;
    (&kernelContext_72)->sort_0 = sort_30;
    (&kernelContext_72)->atom_0 = atom_30;
    (&kernelContext_72)->manifolds_0 = manifolds_30;
    (&kernelContext_72)->loads_0 = loads_30;
    (&kernelContext_72)->bonds_0 = bonds_30;
    (&kernelContext_72)->args_0 = args_30;
    (&kernelContext_72)->info_0 = info_30;
    (&kernelContext_72)->scan_0 = scan_30;
    (&kernelContext_72)->parts_0 = parts_30;
    (&kernelContext_72)->colliders_0 = colliders_30;
    (&kernelContext_72)->collider_info_0 = collider_info_30;
    (&kernelContext_72)->view_chunks_0 = view_chunks_30;
    (&kernelContext_72)->view_0 = view_30;
    (&kernelContext_72)->objects_0 = objects_30;
    (&kernelContext_72)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_72)->lines_0 = lines_30;
    (&kernelContext_72)->geometry_0 = geometry_30;
    (&kernelContext_72)->shots_0 = shots_30;
    (&kernelContext_72)->crush_0 = crush_30;
    (&kernelContext_72)->crush_rw_0 = crush_rw_30;
    (&kernelContext_72)->crush_tab_0 = crush_tab_30;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_72)->g_scan_0 = &g_scan_30;
    uint i_7 = id_27.x;
    bool _S507;
    if(i_7 >= (params_30->bond_count_0))
    {
        _S507 = true;
    }
    else
    {
        _S507 = ((&kernelContext_72)->params_0->shot_count_0) == 0U;
    }
    if(_S507)
    {
        _S507 = true;
    }
    else
    {
        uint _S508 = w_alive_0(&kernelContext_72);
        _S507 = (*((&kernelContext_72)->work_0+(_S508 + i_7))) == 0U;
    }
    if(_S507)
    {
        return;
    }
    uint4 _S509 = uint4(*((&kernelContext_72)->bonds_0+i_7)) ;
    uint _S510 = _S509.x;
    uint _S511 = body_of_0(_S510, &kernelContext_72);
    thread uint shot_4;
    bool _S512 = struck_0(_S511, &shot_4, &kernelContext_72);
    if(!_S512)
    {
        return;
    }
    uint _S513 = strike_at_0(_S511, &kernelContext_72);
    float4 _S514 = float4(*((&kernelContext_72)->frames_0+(_S513 + 2U))) ;
    thread Quat_0 q_23;
    float3 _S515 = chunk_box_world_0(_S510, &q_23, &kernelContext_72);
    float3 _S516 = chunk_box_world_0(_S509.y, &q_23, &kernelContext_72);
    float dist_1 = length(float3(0.5f)  * (_S515 + _S516) - _S514.xyz);
    if((_S514.w * min(1.0f, (as_type<float>((_S509.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((_S509.z))))
    {
        uint _S517 = w_alive_0(&kernelContext_72);
        *((&kernelContext_72)->work_0+(_S517 + i_7)) = 0U;
        uint _S518 = a_dirty_0(&kernelContext_72);
        uint _S519 = atomic_fetch_or_explicit((&kernelContext_72)->atom_0+(_S518 + _S511), 1U, memory_order_relaxed);
        uint _S520 = a_counters_0(&kernelContext_72);
        uint _S521 = atomic_fetch_add_explicit((&kernelContext_72)->atom_0+_S520, 1U, memory_order_relaxed);
        uint _S522 = atomic_fetch_add_explicit((&kernelContext_72)->atom_0+(_S520 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

bool crushable_0(uint k_15, KernelContext_0 thread* kernelContext_73)
{
    return ((as_type<uint>(((float4(*(kernelContext_73->crush_tab_0+(2U * k_15 + 1U))) ).z))) & 1U) != 0U;
}

uint crush_flags_0(uint k_16, KernelContext_0 thread* kernelContext_74)
{
    return (as_type<uint>(((float4(*(kernelContext_74->crush_rw_0+k_16)) ).w)));
}

void crush_record_0(uint k_17, float4 step_4, uint cause_0, KernelContext_0 thread* kernelContext_75)
{
    float4 _S523 = float4(*(kernelContext_75->crush_rw_0+k_17)) ;
    uint flags_1 = (as_type<uint>((_S523.w)));
    uint flags_2;
    if((step_4.w) != 0.0f)
    {
        flags_2 = ((flags_1 | 1U) | 2U) | (cause_0 << 4U);
    }
    else
    {
        flags_2 = flags_1;
    }
    *(kernelContext_75->crush_rw_0+k_17) = packed_float4(float4(step_4.x, max(_S523.y, step_4.y), max(_S523.z, step_4.z), (as_type<float>((flags_2))))) ;
    return;
}

[[kernel]] void fr_crush_blast(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], atomic<uint> device* atom_31 [[buffer(5)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_31 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], CrushParams_0 constant* crush_31 [[buffer(25)]], packed_float4 device* crush_rw_31 [[buffer(24)]], packed_float4 device* crush_tab_31 [[buffer(23)]])
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
    (&kernelContext_76)->bonds_0 = bonds_31;
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
    uint k_18 = id_28.x;
    bool _S524;
    if(k_18 >= (params_31->chunk_count_0))
    {
        _S524 = true;
    }
    else
    {
        _S524 = ((&kernelContext_76)->crush_0->enabled_0) == 0U;
    }
    if(_S524)
    {
        return;
    }
    float4 _S525 = float4(*((&kernelContext_76)->crush_rw_0+((&kernelContext_76)->crush_0->r_blast_0 + k_18))) ;
    float _S526 = _S525.x;
    if(!(_S526 > 0.0f))
    {
        _S524 = !((_S525.y) > 0.0f);
    }
    else
    {
        _S524 = false;
    }
    if(_S524)
    {
        _S524 = !((_S525.z) > 0.0f);
    }
    else
    {
        _S524 = false;
    }
    if(_S524)
    {
        return;
    }
    *((&kernelContext_76)->crush_rw_0+((&kernelContext_76)->crush_0->r_blast_0 + k_18)) = packed_float4(float4(0.0f) ) ;
    bool _S527 = crushable_0(k_18, &kernelContext_76);
    if(!_S527)
    {
        _S524 = true;
    }
    else
    {
        uint _S528 = crush_flags_0(k_18, &kernelContext_76);
        _S524 = (_S528 & 1U) != 0U;
    }
    if(_S524)
    {
        return;
    }
    float d_5 = (float4(*((&kernelContext_76)->crush_rw_0+k_18)) ).x + _S526;
    float4 step_5;
    if(d_5 >= 1.0f)
    {
        step_5 = float4(1.0f, _S525.y, _S525.z, 1.0f);
    }
    else
    {
        step_5 = float4(d_5, _S525.y, _S525.z, 0.0f);
    }
    crush_record_0(k_18, step_5, 3U, &kernelContext_76);
    return;
}

struct Sym_0
{
    float3 d_6;
    float3 o_5;
};

Sym_0 sym_zero_0()
{
    thread Sym_0 s_4;
    float3 _S529 = float3(0.0f) ;
    (&s_4)->d_6 = _S529;
    (&s_4)->o_5 = _S529;
    return s_4;
}

Sym_0 sym_outer_0(float3 r_8, float3 f_10)
{
    thread Sym_0 s_5;
    (&s_5)->d_6 = r_8 * f_10;
    float _S530 = r_8.x;
    float _S531 = f_10.y;
    float _S532 = r_8.y;
    float _S533 = f_10.x;
    float _S534 = f_10.z;
    float _S535 = r_8.z;
    (&s_5)->o_5 = float3(0.5f)  * float3(_S530 * _S531 + _S532 * _S533, _S530 * _S534 + _S535 * _S533, _S532 * _S534 + _S535 * _S531);
    return s_5;
}

Sym_0 sym_add_0(const Sym_0 thread* a_4, const Sym_0 thread* b_12)
{
    thread Sym_0 s_6;
    (&s_6)->d_6 = a_4->d_6 + b_12->d_6;
    (&s_6)->o_5 = a_4->o_5 + b_12->o_5;
    return s_6;
}

void add_load_0(Sym_0 thread* w_6, float3 thread* f_sum_0, float4 thread* spread_0, float3 r_9, float3 f_11)
{
    Sym_0 _S536 = sym_outer_0(r_9, f_11);
    thread Sym_0 _S537 = *w_6;
    thread Sym_0 _S538 = _S536;
    Sym_0 _S539 = sym_add_0(&_S537, &_S538);
    *w_6 = _S539;
    *f_sum_0 = *f_sum_0 + f_11;
    float m_3 = length(f_11);
    *spread_0 = *spread_0 + float4(float3(m_3)  * r_9, m_3);
    return;
}

void flow_virial_0(uint k_19, const Quat_0 thread* q_24, float3 origin_6, float3 centre_1, Sym_0 thread* wf_0, float thread* total_0, KernelContext_0 thread* kernelContext_77)
{
    *wf_0 = sym_zero_0();
    *total_0 = 0.0f;
    float4 _S540 = float4(*(kernelContext_77->crush_tab_0+(kernelContext_77->crush_0->t_adj_0 + k_19))) ;
    uint first_2 = (as_type<uint>((_S540.x)));
    uint _S541 = (as_type<uint>((_S540.y)));
    uint j_4 = first_2;
    for(;;)
    {
        if(j_4 < (first_2 + _S541))
        {
        }
        else
        {
            break;
        }
        uint bi_0 = (as_type<uint>(((float4(*(kernelContext_77->crush_tab_0+(kernelContext_77->crush_0->t_list_0 + j_4 / 4U))) )[j_4 % 4U])));
        uint _S542 = w_alive_0(kernelContext_77);
        if((*(kernelContext_77->work_0+(_S542 + bi_0))) == 0U)
        {
            j_4 = j_4 + 1U;
            continue;
        }
        float l_0 = (*(kernelContext_77->crush_rw_0+(kernelContext_77->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U];
        if(!(((*(kernelContext_77->crush_rw_0+(kernelContext_77->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U]) > 0.0f))
        {
            j_4 = j_4 + 1U;
            continue;
        }
        float3 _S543 = rotate_0(q_24, (float4(*(kernelContext_77->crush_tab_0+(kernelContext_77->crush_0->t_bonds_0 + bi_0))) ).xyz);
        float3 r_10 = origin_6 + _S543 - centre_1;
        float3 _S544 = rotate_0(q_24, (float4(*(kernelContext_77->crush_tab_0+(kernelContext_77->crush_0->t_normal_0 + bi_0))) ).xyz);
        float3 out_n_0;
        if((dot(_S544, r_10)) >= 0.0f)
        {
            out_n_0 = _S544;
        }
        else
        {
            out_n_0 = - _S544;
        }
        Sym_0 _S545 = sym_outer_0(r_10, float3(- l_0)  * out_n_0);
        thread Sym_0 _S546 = *wf_0;
        thread Sym_0 _S547 = _S545;
        Sym_0 _S548 = sym_add_0(&_S546, &_S547);
        *wf_0 = _S548;
        *total_0 = *total_0 + l_0;
        j_4 = j_4 + 1U;
    }
    return;
}

float4 bond_centroid_0(uint k_20, const Quat_0 thread* q_25, float3 origin_7, float3 centre_2, KernelContext_0 thread* kernelContext_78)
{
    float4 _S549 = float4(*(kernelContext_78->crush_tab_0+(kernelContext_78->crush_0->t_adj_0 + k_20))) ;
    uint first_3 = (as_type<uint>((_S549.x)));
    uint _S550 = (as_type<uint>((_S549.y)));
    float3 _S551 = float3(0.0f) ;
    uint j_5 = first_3;
    float3 sum_4 = _S551;
    float area_0 = 0.0f;
    for(;;)
    {
        if(j_5 < (first_3 + _S550))
        {
        }
        else
        {
            break;
        }
        uint bi_1 = (as_type<uint>(((float4(*(kernelContext_78->crush_tab_0+(kernelContext_78->crush_0->t_list_0 + j_5 / 4U))) )[j_5 % 4U])));
        uint _S552 = w_alive_0(kernelContext_78);
        if((*(kernelContext_78->work_0+(_S552 + bi_1))) == 0U)
        {
            j_5 = j_5 + 1U;
            continue;
        }
        float4 _S553 = float4(*(kernelContext_78->crush_tab_0+(kernelContext_78->crush_0->t_bonds_0 + bi_1))) ;
        float3 _S554 = rotate_0(q_25, _S553.xyz);
        float _S555 = _S553.w;
        float area_1 = area_0 + _S555;
        sum_4 = sum_4 + float3(_S555)  * (origin_7 + _S554 - centre_2);
        area_0 = area_1;
        j_5 = j_5 + 1U;
    }
    if(area_0 > 0.0f)
    {
        sum_4 = sum_4 / float3(area_0) ;
    }
    else
    {
        sum_4 = _S551;
    }
    return float4(sum_4, area_0);
}

Sym_0 balanced_0(const Sym_0 thread* w_7, float3 f_12, float4 spread_1, float4 bonds_32)
{
    float3 at_10;
    if((bonds_32.w) > 0.0f)
    {
        at_10 = bonds_32.xyz;
    }
    else
    {
        float _S556 = spread_1.w;
        if(_S556 > 0.0f)
        {
            at_10 = spread_1.xyz / float3(_S556) ;
        }
        else
        {
            at_10 = float3(0.0f) ;
        }
    }
    thread Sym_0 _S557 = sym_outer_0(at_10, - f_12);
    Sym_0 _S558 = sym_add_0(w_7, &_S557);
    return _S558;
}

float sym_norm2_0(const Sym_0 thread* s_7)
{
    return dot(s_7->d_6, s_7->d_6) + 2.0f * dot(s_7->o_5, s_7->o_5);
}

float2 crush_invariants_0(const Sym_0 thread* v_10, float volume_0)
{
    float inv_4 = 1.0f / volume_0;
    float xx_0 = v_10->d_6.x * inv_4;
    float yy_0 = v_10->d_6.y * inv_4;
    float zz_0 = v_10->d_6.z * inv_4;
    float xy_0 = v_10->o_5.x * inv_4;
    float xz_0 = v_10->o_5.y * inv_4;
    float yz_0 = v_10->o_5.z * inv_4;
    float p_7 = - (xx_0 + yy_0 + zz_0) / 3.0f;
    float dx_0 = xx_0 + p_7;
    float dy_0 = yy_0 + p_7;
    float dz_0 = zz_0 + p_7;
    return float2(p_7, sqrt(1.5f * max(dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0 + 2.0f * (xy_0 * xy_0 + xz_0 * xz_0 + yz_0 * yz_0), 0.0f)));
}

float crush_excess_0(float2 pq_0, float4 m0_0)
{
    float _S559 = pq_0.x;
    if(!(_S559 > 0.0f))
    {
        return 0.0f;
    }
    float _S560 = max(pq_0.y - (m0_0.y + m0_0.z * _S559), _S559 - m0_0.x);
    float _S561;
    if(_S560 > 0.0f)
    {
        _S561 = _S560;
    }
    else
    {
        _S561 = 0.0f;
    }
    return _S561;
}

float4 crush_two_0(const Sym_0 thread* a_5, float dt_a_0, const Sym_0 thread* b_13, float dt_b_0, uint k_21, float damage_0, KernelContext_0 thread* kernelContext_79)
{
    uint _S562 = 2U * k_21;
    float4 _S563 = float4(*(kernelContext_79->crush_tab_0+_S562)) ;
    float4 _S564 = float4(*(kernelContext_79->crush_tab_0+(_S562 + 1U))) ;
    float _S565 = _S564.y;
    float2 _S566 = crush_invariants_0(a_5, _S565);
    float2 _S567 = crush_invariants_0(b_13, _S565);
    float ea_0 = crush_excess_0(_S566, _S563);
    bool _S568 = dt_b_0 > 0.0f;
    float eb_0;
    if(_S568)
    {
        eb_0 = crush_excess_0(_S567, _S563);
    }
    else
    {
        eb_0 = 0.0f;
    }
    float d_7 = damage_0 + (ea_0 * ea_0 * dt_a_0 + eb_0 * eb_0 * dt_b_0) / (_S564.x * _S563.w);
    float2 peak_2;
    if(_S568)
    {
        peak_2 = max(_S566, _S567);
    }
    else
    {
        peak_2 = _S566;
    }
    if(d_7 >= 1.0f)
    {
        return float4(1.0f, peak_2, 1.0f);
    }
    return float4(d_7, peak_2, 0.0f);
}

void crush_loads_flow_0(uint k_22, KernelContext_0 thread* kernelContext_80)
{
    uint _S569 = body_of_0(k_22, kernelContext_80);
    uint b_14 = 7U * _S569;
    thread Quat_0 q_26;
    thread float3 origin_8;
    structure_pose_0(_S569, b_14, &q_26, &origin_8, kernelContext_80);
    uint _S570 = 4U * k_22;
    float4 _S571 = float4(*(kernelContext_80->chunks_0+_S570)) ;
    float4 _S572 = float4(*(kernelContext_80->chunks_0+(_S570 + 3U))) ;
    float3 _S573 = _S571.xyz;
    thread Quat_0 _S574 = q_26;
    float3 _S575 = rotate_0(&_S574, _S573);
    float3 centre_3 = origin_8 + _S575;
    float inv_mass_4 = (float4(*(kernelContext_80->bodies_0+b_14)) ).w;
    float _S576;
    if(inv_mass_4 > 0.0f)
    {
        _S576 = 1.0f / inv_mass_4;
    }
    else
    {
        _S576 = float_bits_0(2139095040U);
    }
    float dt_2 = kernelContext_80->params_0->dt_0;
    Sym_0 _S577 = sym_zero_0();
    thread Sym_0 wa_0 = _S577;
    float3 _S578 = float3(0.0f) ;
    thread float3 fa_0 = _S578;
    float4 _S579 = float4(0.0f) ;
    thread float4 sa_0 = _S579;
    thread Sym_0 wb_0 = _S577;
    thread float3 fb_0 = _S578;
    thread float4 sb_1 = _S579;
    uint _S580 = kernelContext_80->sort_0[k_22];
    uint _S581 = kernelContext_80->sort_0[kernelContext_80->params_0->chunk_count_0 + 1U + k_22];
    float tau_0 = 0.0f;
    uint i_8 = 0U;
    for(;;)
    {
        if(i_8 < _S581)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_11 = kernelContext_80->manifolds_0[_S580 + i_8];
        float4 _S582 = float4(r_11.impulse_0) ;
        float3 _S583 = _S582.xyz;
        float j_6 = length(_S583);
        if(j_6 == 0.0f)
        {
            i_8 = i_8 + 1U;
            continue;
        }
        float4 _S584 = float4(r_11.other_0) ;
        float m_other_2 = _S584.x;
        float m_4;
        if(isinf(_S576))
        {
            if(isinf(m_other_2))
            {
                m_4 = _S571.w;
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
                m_4 = _S576;
            }
            else
            {
                m_4 = _S576 * m_other_2 / (_S576 + m_other_2);
            }
        }
        float _S585 = _S582.w;
        float _S586 = min(j_6, 2.0f * m_4 * max(_S585 - kernelContext_80->params_0->band_0, 0.0f));
        float3 rr_0 = (float4(r_11.point_0) ).xyz - centre_3;
        float3 dir_1 = _S583 / float3(j_6) ;
        if(_S586 < j_6)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, rr_0, dir_1 * float3(((j_6 - _S586) / dt_2)) );
        }
        float tau_1;
        if(_S586 > 0.0f)
        {
            float _S587 = _S572.y;
            float duration_2 = hertz_duration_0(m_4, min(_S584.z, _S572.z), 1.0f / ((1.0f - _S587 * _S587) / _S572.x + 1.0f / _S584.y), _S585);
            add_load_0(&wb_0, &fb_0, &sb_1, rr_0, dir_1 * float3((3.14159274101257324f * _S586 / (2.0f * duration_2))) );
            tau_1 = max(tau_0, min(duration_2, dt_2));
        }
        else
        {
            tau_1 = tau_0;
        }
        tau_0 = tau_1;
        i_8 = i_8 + 1U;
    }
    uint from_2 = kernelContext_80->params_0->parity_0;
    uint _S588 = w_pulses_0(kernelContext_80);
    uint _S589 = _S588 + _S570 + 2U * from_2;
    uint off_1 = *(kernelContext_80->work_0+_S589);
    uint _S590 = *(kernelContext_80->work_0+(_S589 + 1U));
    uint _S591 = l_pool_0(from_2, kernelContext_80);
    uint _S592 = _S591 + 3U * off_1;
    uint p_8 = 0U;
    for(;;)
    {
        if(p_8 < _S590)
        {
        }
        else
        {
            break;
        }
        uint _S593 = _S592 + 3U * p_8;
        float4 _S594 = float4(*(kernelContext_80->loads_0+_S593)) ;
        float4 _S595 = float4(*(kernelContext_80->loads_0+(_S593 + 1U))) ;
        float4 _S596 = float4(*(kernelContext_80->loads_0+(_S593 + 2U))) ;
        float mean_2 = pulse_mean_0(_S594.x, _S594.y, _S594.z, _S594.w, _S595.w, dt_2);
        if(mean_2 != 0.0f)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, _S595.xyz - centre_3, _S596.xyz * float3(mean_2) );
        }
        p_8 = p_8 + 1U;
    }
    thread Quat_0 _S597 = q_26;
    thread Sym_0 wf_1;
    thread float flow_total_0;
    flow_virial_0(k_22, &_S597, origin_8, centre_3, &wf_1, &flow_total_0, kernelContext_80);
    bool contact_1;
    if((sa_0.w) > 0.0f)
    {
        contact_1 = true;
    }
    else
    {
        contact_1 = (sb_1.w) > 0.0f;
    }
    bool _S598;
    if(!contact_1)
    {
        _S598 = !(flow_total_0 > 0.0f);
    }
    else
    {
        _S598 = false;
    }
    if(_S598)
    {
        return;
    }
    Sym_0 _S599 = wf_1;
    Sym_0 _S600 = wf_1;
    Sym_0 peak_contact_0;
    Sym_0 base_3;
    Sym_0 peak_3;
    if(contact_1)
    {
        thread Quat_0 _S601 = q_26;
        float4 _S602 = bond_centroid_0(k_22, &_S601, origin_8, centre_3, kernelContext_80);
        thread Sym_0 _S603 = wa_0;
        Sym_0 _S604 = balanced_0(&_S603, fa_0, sa_0, _S602);
        thread Sym_0 _S605 = _S604;
        thread Sym_0 _S606 = wf_1;
        Sym_0 _S607 = sym_add_0(&_S605, &_S606);
        thread Sym_0 _S608 = wa_0;
        thread Sym_0 _S609 = wb_0;
        Sym_0 _S610 = sym_add_0(&_S608, &_S609);
        float3 _S611 = fa_0 + fb_0;
        float4 _S612 = sa_0 + sb_1;
        thread Sym_0 _S613 = _S610;
        Sym_0 _S614 = balanced_0(&_S613, _S611, _S612, _S602);
        thread Sym_0 _S615 = _S614;
        thread Sym_0 _S616 = wf_1;
        Sym_0 _S617 = sym_add_0(&_S615, &_S616);
        peak_contact_0 = _S614;
        base_3 = _S607;
        peak_3 = _S617;
    }
    else
    {
        peak_contact_0 = _S577;
        base_3 = _S599;
        peak_3 = _S600;
    }
    thread Sym_0 _S618 = wf_1;
    float _S619 = sym_norm2_0(&_S618);
    thread Sym_0 _S620 = peak_contact_0;
    float _S621 = sym_norm2_0(&_S620);
    uint cause_1;
    if(_S619 >= _S621)
    {
        cause_1 = 4U;
    }
    else
    {
        cause_1 = 1U;
    }
    float _S622 = dt_2 - tau_0;
    float _S623 = (float4(*(kernelContext_80->crush_rw_0+k_22)) ).x;
    thread Sym_0 _S624 = base_3;
    thread Sym_0 _S625 = peak_3;
    float4 _S626 = crush_two_0(&_S624, _S622, &_S625, tau_0, k_22, _S623, kernelContext_80);
    crush_record_0(k_22, _S626, cause_1, kernelContext_80);
    return;
}

[[kernel]] void fr_crush_loads(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], atomic<uint> device* atom_32 [[buffer(5)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], CrushParams_0 constant* crush_32 [[buffer(25)]], packed_float4 device* crush_rw_32 [[buffer(24)]], packed_float4 device* crush_tab_32 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_81;
    (&kernelContext_81)->params_0 = params_32;
    (&kernelContext_81)->work_0 = work_32;
    (&kernelContext_81)->frames_0 = frames_32;
    (&kernelContext_81)->bodies_0 = bodies_32;
    (&kernelContext_81)->chunks_0 = chunks_32;
    (&kernelContext_81)->sort_0 = sort_32;
    (&kernelContext_81)->atom_0 = atom_32;
    (&kernelContext_81)->manifolds_0 = manifolds_32;
    (&kernelContext_81)->loads_0 = loads_32;
    (&kernelContext_81)->bonds_0 = bonds_33;
    (&kernelContext_81)->args_0 = args_32;
    (&kernelContext_81)->info_0 = info_32;
    (&kernelContext_81)->scan_0 = scan_32;
    (&kernelContext_81)->parts_0 = parts_32;
    (&kernelContext_81)->colliders_0 = colliders_32;
    (&kernelContext_81)->collider_info_0 = collider_info_32;
    (&kernelContext_81)->view_chunks_0 = view_chunks_32;
    (&kernelContext_81)->view_0 = view_32;
    (&kernelContext_81)->objects_0 = objects_32;
    (&kernelContext_81)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_81)->lines_0 = lines_32;
    (&kernelContext_81)->geometry_0 = geometry_32;
    (&kernelContext_81)->shots_0 = shots_32;
    (&kernelContext_81)->crush_0 = crush_32;
    (&kernelContext_81)->crush_rw_0 = crush_rw_32;
    (&kernelContext_81)->crush_tab_0 = crush_tab_32;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_81)->g_scan_0 = &g_scan_32;
    uint k_23 = id_29.x;
    bool _S627;
    if(k_23 >= (params_32->chunk_count_0))
    {
        _S627 = true;
    }
    else
    {
        _S627 = ((&kernelContext_81)->crush_0->enabled_0) == 0U;
    }
    if(_S627)
    {
        _S627 = true;
    }
    else
    {
        bool _S628 = crushable_0(k_23, &kernelContext_81);
        _S627 = !_S628;
    }
    if(_S627)
    {
        _S627 = true;
    }
    else
    {
        uint _S629 = crush_flags_0(k_23, &kernelContext_81);
        _S627 = (_S629 & 1U) != 0U;
    }
    if(_S627)
    {
        return;
    }
    if(((&kernelContext_81)->crush_0->flow_0) != 0U)
    {
        crush_loads_flow_0(k_23, &kernelContext_81);
        return;
    }
    uint _S630 = body_of_0(k_23, &kernelContext_81);
    uint b_15 = 7U * _S630;
    thread Quat_0 q_27;
    thread float3 origin_9;
    structure_pose_0(_S630, b_15, &q_27, &origin_9, &kernelContext_81);
    uint _S631 = 4U * k_23;
    float4 _S632 = float4(*((&kernelContext_81)->chunks_0+_S631)) ;
    float4 _S633 = float4(*((&kernelContext_81)->chunks_0+(_S631 + 3U))) ;
    float3 _S634 = _S632.xyz;
    thread Quat_0 _S635 = q_27;
    float3 _S636 = rotate_0(&_S635, _S634);
    float3 centre_4 = origin_9 + _S636;
    float inv_mass_5 = (float4(*((&kernelContext_81)->bodies_0+b_15)) ).w;
    float _S637;
    if(inv_mass_5 > 0.0f)
    {
        _S637 = 1.0f / inv_mass_5;
    }
    else
    {
        _S637 = float_bits_0(2139095040U);
    }
    float dt_3 = (&kernelContext_81)->params_0->dt_0;
    Sym_0 _S638 = sym_zero_0();
    thread Sym_0 wa_1 = _S638;
    float3 _S639 = float3(0.0f) ;
    thread float3 fa_1 = _S639;
    float4 _S640 = float4(0.0f) ;
    thread float4 sa_1 = _S640;
    thread Sym_0 wb_1 = _S638;
    thread float3 fb_1 = _S639;
    thread float4 sb_2 = _S640;
    uint _S641 = (&kernelContext_81)->sort_0[k_23];
    uint _S642 = (&kernelContext_81)->sort_0[params_32->chunk_count_0 + 1U + k_23];
    float tau_2 = 0.0f;
    uint i_9 = 0U;
    for(;;)
    {
        if(i_9 < _S642)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_12 = (&kernelContext_81)->manifolds_0[_S641 + i_9];
        float4 _S643 = float4(r_12.impulse_0) ;
        float3 _S644 = _S643.xyz;
        float j_7 = length(_S644);
        if(j_7 == 0.0f)
        {
            i_9 = i_9 + 1U;
            continue;
        }
        float4 _S645 = float4(r_12.other_0) ;
        float m_other_3 = _S645.x;
        float m_5;
        if(isinf(_S637))
        {
            if(isinf(m_other_3))
            {
                m_5 = _S632.w;
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
                m_5 = _S637;
            }
            else
            {
                m_5 = _S637 * m_other_3 / (_S637 + m_other_3);
            }
        }
        float _S646 = _S643.w;
        float _S647 = min(j_7, 2.0f * m_5 * max(_S646 - (&kernelContext_81)->params_0->band_0, 0.0f));
        float3 rr_1 = (float4(r_12.point_0) ).xyz - centre_4;
        float3 dir_2 = _S644 / float3(j_7) ;
        if(_S647 < j_7)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, rr_1, dir_2 * float3(((j_7 - _S647) / dt_3)) );
        }
        float tau_3;
        if(_S647 > 0.0f)
        {
            float _S648 = _S633.y;
            float duration_3 = hertz_duration_0(m_5, min(_S645.z, _S633.z), 1.0f / ((1.0f - _S648 * _S648) / _S633.x + 1.0f / _S645.y), _S646);
            add_load_0(&wb_1, &fb_1, &sb_2, rr_1, dir_2 * float3((3.14159274101257324f * _S647 / (2.0f * duration_3))) );
            tau_3 = max(tau_2, min(duration_3, dt_3));
        }
        else
        {
            tau_3 = tau_2;
        }
        tau_2 = tau_3;
        i_9 = i_9 + 1U;
    }
    uint from_3 = (&kernelContext_81)->params_0->parity_0;
    uint _S649 = w_pulses_0(&kernelContext_81);
    uint _S650 = _S649 + _S631 + 2U * from_3;
    uint device* _S651 = (&kernelContext_81)->work_0+_S650;
    uint _S652 = *((&kernelContext_81)->work_0+(_S650 + 1U));
    uint _S653 = l_pool_0(from_3, &kernelContext_81);
    uint _S654 = _S653 + 3U * *_S651;
    uint p_9 = 0U;
    for(;;)
    {
        if(p_9 < _S652)
        {
        }
        else
        {
            break;
        }
        uint _S655 = _S654 + 3U * p_9;
        float4 _S656 = float4(*((&kernelContext_81)->loads_0+_S655)) ;
        float4 _S657 = float4(*((&kernelContext_81)->loads_0+(_S655 + 1U))) ;
        float4 _S658 = float4(*((&kernelContext_81)->loads_0+(_S655 + 2U))) ;
        float mean_3 = pulse_mean_0(_S656.x, _S656.y, _S656.z, _S656.w, _S657.w, dt_3);
        if(mean_3 != 0.0f)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, _S657.xyz - centre_4, _S658.xyz * float3(mean_3) );
        }
        p_9 = p_9 + 1U;
    }
    if(!((sa_1.w) > 0.0f))
    {
        _S627 = !((sb_2.w) > 0.0f);
    }
    else
    {
        _S627 = false;
    }
    if(_S627)
    {
        return;
    }
    thread Quat_0 _S659 = q_27;
    float4 _S660 = bond_centroid_0(k_23, &_S659, origin_9, centre_4, &kernelContext_81);
    thread Sym_0 _S661 = wa_1;
    Sym_0 _S662 = balanced_0(&_S661, fa_1, sa_1, _S660);
    thread Sym_0 _S663 = wa_1;
    thread Sym_0 _S664 = wb_1;
    Sym_0 _S665 = sym_add_0(&_S663, &_S664);
    float3 _S666 = fa_1 + fb_1;
    float4 _S667 = sa_1 + sb_2;
    thread Sym_0 _S668 = _S665;
    Sym_0 _S669 = balanced_0(&_S668, _S666, _S667, _S660);
    float _S670 = dt_3 - tau_2;
    float _S671 = (float4(*((&kernelContext_81)->crush_rw_0+k_23)) ).x;
    thread Sym_0 _S672 = _S662;
    thread Sym_0 _S673 = _S669;
    float4 _S674 = crush_two_0(&_S672, _S670, &_S673, tau_2, k_23, _S671, &kernelContext_81);
    crush_record_0(k_23, _S674, 1U, &kernelContext_81);
    return;
}

uint a_cover_0(KernelContext_0 thread* kernelContext_82)
{
    uint _S675 = a_strike_dist_0(kernelContext_82);
    return _S675 + kernelContext_82->params_0->chunk_count_0;
}

[[kernel]] void fr_strike_patch(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], atomic<uint> device* atom_33 [[buffer(5)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], CrushParams_0 constant* crush_33 [[buffer(25)]], packed_float4 device* crush_rw_33 [[buffer(24)]], packed_float4 device* crush_tab_33 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_83;
    (&kernelContext_83)->params_0 = params_33;
    (&kernelContext_83)->work_0 = work_33;
    (&kernelContext_83)->frames_0 = frames_33;
    (&kernelContext_83)->bodies_0 = bodies_33;
    (&kernelContext_83)->chunks_0 = chunks_33;
    (&kernelContext_83)->sort_0 = sort_33;
    (&kernelContext_83)->atom_0 = atom_33;
    (&kernelContext_83)->manifolds_0 = manifolds_33;
    (&kernelContext_83)->loads_0 = loads_33;
    (&kernelContext_83)->bonds_0 = bonds_34;
    (&kernelContext_83)->args_0 = args_33;
    (&kernelContext_83)->info_0 = info_33;
    (&kernelContext_83)->scan_0 = scan_33;
    (&kernelContext_83)->parts_0 = parts_33;
    (&kernelContext_83)->colliders_0 = colliders_33;
    (&kernelContext_83)->collider_info_0 = collider_info_33;
    (&kernelContext_83)->view_chunks_0 = view_chunks_33;
    (&kernelContext_83)->view_0 = view_33;
    (&kernelContext_83)->objects_0 = objects_33;
    (&kernelContext_83)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_83)->lines_0 = lines_33;
    (&kernelContext_83)->geometry_0 = geometry_33;
    (&kernelContext_83)->shots_0 = shots_33;
    (&kernelContext_83)->crush_0 = crush_33;
    (&kernelContext_83)->crush_rw_0 = crush_rw_33;
    (&kernelContext_83)->crush_tab_0 = crush_tab_33;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_83)->g_scan_0 = &g_scan_33;
    uint b_16 = id_30.x;
    bool _S676;
    if(b_16 >= (params_33->body_count_0))
    {
        _S676 = true;
    }
    else
    {
        _S676 = ((&kernelContext_83)->params_0->shot_count_0) == 0U;
    }
    if(_S676)
    {
        _S676 = true;
    }
    else
    {
        _S676 = ((&kernelContext_83)->crush_0->enabled_0) == 0U;
    }
    if(_S676)
    {
        return;
    }
    uint bill_0 = (&kernelContext_83)->crush_0->r_bill_0 + 4U * b_16;
    packed_float4 _S677 = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_83)->crush_rw_0+bill_0) = _S677;
    uint _S678 = bill_0 + 3U;
    *((&kernelContext_83)->crush_rw_0+_S678) = _S677;
    thread uint shot_5;
    bool _S679 = struck_0(b_16, &shot_5, &kernelContext_83);
    if(!_S679)
    {
        return;
    }
    uint _S680 = a_strike_0(&kernelContext_83);
    uint _S681 = _S680 + 9U * b_16;
    uint k_24 = atomic_load_explicit((&kernelContext_83)->atom_0+(_S681 + 8U), memory_order_relaxed);
    uint _S682 = atomic_load_explicit((&kernelContext_83)->atom_0+(_S681 + 6U), memory_order_relaxed);
    float t_in_3 = (as_type<float>((_S682)));
    if(!(t_in_3 <= ((&kernelContext_83)->params_0->dt_0)))
    {
        _S676 = true;
    }
    else
    {
        _S676 = k_24 == 4294967295U;
    }
    if(_S676)
    {
        return;
    }
    uint _S683 = strike_at_0(b_16, &kernelContext_83);
    float4 _S684 = float4(*((&kernelContext_83)->frames_0+_S683)) ;
    float3 _S685 = (float4(*((&kernelContext_83)->frames_0+(_S683 + 1U))) ).xyz;
    float speed_1 = length(_S685);
    float3 d_8 = _S685 / float3(max(speed_1, 9.999999960041972e-13f)) ;
    float4 _S686 = float4(*((&kernelContext_83)->shots_0+4U * shot_5)) ;
    float _S687 = shot_lead_0(shot_5, d_8, &kernelContext_83);
    float _S688 = max(_S686.z, 0.00100000004749745f);
    float m_imp_1 = _S686.w;
    float inv_mass_6 = (float4(*((&kernelContext_83)->bodies_0+7U * b_16)) ).w;
    bool _S689 = inv_mass_6 == 0.0f;
    float m_body_2;
    if(_S689)
    {
        m_body_2 = 0.0f;
    }
    else
    {
        m_body_2 = 1.0f / inv_mass_6;
    }
    float m_eff_1;
    if(_S689)
    {
        m_eff_1 = m_imp_1;
    }
    else
    {
        m_eff_1 = m_imp_1 * m_body_2 / (m_imp_1 + m_body_2);
    }
    float modulus_1 = 1.0f / (1.0f / (float4(*((&kernelContext_83)->shots_0+(4U * shot_5 + 1U))) ).x + 1.0f / (float4(*((&kernelContext_83)->geometry_0+(3U * k_24 + 1U))) ).w);
    float _S690 = max(speed_1, 0.00100000004749745f);
    float tau_4 = 2.86999988555908203f * pow(m_eff_1 * m_eff_1 / (_S688 * modulus_1 * modulus_1 * _S690), 0.20000000298023224f);
    float force_0 = m_eff_1 * speed_1 / tau_4;
    float3 contact_2 = _S684.xyz + _S685 * float3(t_in_3)  + d_8 * float3(_S687) ;
    float delta_0 = pow(15.0f * m_eff_1 * _S690 * _S690 / (16.0f * modulus_1 * sqrt(_S688)), 0.40000000596046448f);
    float a_h_0 = sqrt(_S687 * delta_0);
    bool _S691 = crushable_0(k_24, &kernelContext_83);
    float p_cap_0;
    if(_S691)
    {
        p_cap_0 = (float4(*((&kernelContext_83)->crush_tab_0+2U * k_24)) ).x;
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
    float _S692 = min(_S687, max(a_h_0, a_p_0));
    float _S693 = max(delta_0, _S687 - sqrt(max(_S687 * _S687 - _S692 * _S692, 0.0f)));
    *((&kernelContext_83)->crush_rw_0+(bill_0 + 1U)) = packed_float4(float4(contact_2, _S687)) ;
    *((&kernelContext_83)->crush_rw_0+(bill_0 + 2U)) = packed_float4(float4(d_8, _S692)) ;
    *((&kernelContext_83)->crush_rw_0+_S678) = packed_float4(float4(force_0, tau_4, _S693, float(k_24 + 1U))) ;
    uint _S694 = a_cover_0(&kernelContext_83);
    uint _S695 = _S694 + 128U * b_16;
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
        uint _S696 = _S695 + 2U * s_8;
        atomic_store_explicit((&kernelContext_83)->atom_0+_S696, 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_83)->atom_0+(_S696 + 1U), 4294967295U, memory_order_relaxed);
        s_8 = s_8 + 1U;
    }
    return;
}

bool cover_patch_0(uint k_25, uint thread* b_17, float4 thread* p1_0, float4 thread* p2_0, KernelContext_0 thread* kernelContext_84)
{
    uint _S697 = body_of_0(k_25, kernelContext_84);
    *b_17 = _S697;
    float4 _S698 = float4(0.0f) ;
    *p1_0 = _S698;
    *p2_0 = _S698;
    thread uint shot_6;
    bool _S699 = struck_0(*b_17, &shot_6, kernelContext_84);
    bool _S700;
    if(!_S699)
    {
        _S700 = true;
    }
    else
    {
        uint _S701 = crush_flags_0(k_25, kernelContext_84);
        _S700 = (_S701 & 1U) != 0U;
    }
    if(_S700)
    {
        return false;
    }
    uint bill_1 = kernelContext_84->crush_0->r_bill_0 + 4U * *b_17;
    if(((float4(*(kernelContext_84->crush_rw_0+(bill_1 + 3U))) ).w) == 0.0f)
    {
        return false;
    }
    *p1_0 = float4(*(kernelContext_84->crush_rw_0+(bill_1 + 1U))) ;
    *p2_0 = float4(*(kernelContext_84->crush_rw_0+(bill_1 + 2U))) ;
    return true;
}

float3 cover_offset_0(uint s_9, float3 d_9, float a_6, float thread* rho_0)
{
    uint ring_0 = s_9 / 8U;
    uint sector_0 = s_9 % 8U;
    *rho_0 = a_6 * sqrt((float(ring_0) + 0.5f) / 8.0f);
    float theta_1 = 6.28318548202514648f * (float(sector_0) + 0.5f * float(ring_0 & 1U)) / 8.0f;
    float3 e_3;
    if((abs(d_9.x)) < 0.57735002040863037f)
    {
        e_3 = float3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_9.y)) < 0.57735002040863037f)
        {
            e_3 = float3(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_3 = float3(0.0f, 0.0f, 1.0f);
        }
    }
    float3 u_2 = normalize(cross(d_9, e_3));
    return float3(*rho_0)  * (float3(cos(theta_1))  * u_2 + float3(sin(theta_1))  * cross(d_9, u_2));
}

float cover_depth_0(uint k_26, uint s_10, float3 contact_3, float3 d_10, float a_7, float radius_0, KernelContext_0 thread* kernelContext_85)
{
    thread float rho_1;
    float _S702 = 2.0f * radius_0;
    float3 _S703 = contact_3 + cover_offset_0(s_10, d_10, a_7, &rho_1) - d_10 * float3(_S702) ;
    thread Quat_0 q_28;
    float3 _S704 = chunk_box_world_0(k_26, &q_28, kernelContext_85);
    uint _S705 = 3U * k_26;
    Quat_0 _S706 = quat_of_0(float4(*(kernelContext_85->geometry_0+(_S705 + 2U))) );
    thread Quat_0 _S707 = q_28;
    thread Quat_0 _S708 = _S706;
    Quat_0 _S709 = quat_mul_0(&_S707, &_S708);
    thread Quat_0 _S710 = _S709;
    Quat_0 _S711 = normalized_0(&_S710);
    float3 _S712 = (float4(*(kernelContext_85->geometry_0+(_S705 + 1U))) ).xyz;
    float t0_2 = -1.00000001504746622e+30f;
    float t1_1 = 1.00000001504746622e+30f;
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
        thread float3 e_4 = float3(0.0f) ;
        e_4[ax_2] = 1.0f;
        thread Quat_0 _S713 = _S711;
        float3 _S714 = rotate_0(&_S713, e_4);
        float o_6 = dot(_S703 - _S704, _S714);
        float dv_2 = dot(d_10, _S714);
        if((abs(dv_2)) < 9.99999971718068537e-10f)
        {
            if((abs(o_6)) > (_S712[ax_2]))
            {
                return 1.00000001504746622e+30f;
            }
            ax_2 = ax_2 + 1U;
            continue;
        }
        float ta_3 = (- _S712[ax_2] - o_6) / dv_2;
        float tb_3 = (_S712[ax_2] - o_6) / dv_2;
        float _S715 = min(t1_1, max(ta_3, tb_3));
        t0_2 = max(t0_2, min(ta_3, tb_3));
        t1_1 = _S715;
        ax_2 = ax_2 + 1U;
    }
    bool _S716;
    if(!(t0_2 <= t1_1))
    {
        _S716 = true;
    }
    else
    {
        _S716 = t1_1 < 0.0f;
    }
    if(_S716)
    {
        return 1.00000001504746622e+30f;
    }
    return t0_2 - _S702;
}

[[kernel]] void fr_strike_cover(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], atomic<uint> device* atom_34 [[buffer(5)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], CrushParams_0 constant* crush_34 [[buffer(25)]], packed_float4 device* crush_rw_34 [[buffer(24)]], packed_float4 device* crush_tab_34 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_86;
    (&kernelContext_86)->params_0 = params_34;
    (&kernelContext_86)->work_0 = work_34;
    (&kernelContext_86)->frames_0 = frames_34;
    (&kernelContext_86)->bodies_0 = bodies_34;
    (&kernelContext_86)->chunks_0 = chunks_34;
    (&kernelContext_86)->sort_0 = sort_34;
    (&kernelContext_86)->atom_0 = atom_34;
    (&kernelContext_86)->manifolds_0 = manifolds_34;
    (&kernelContext_86)->loads_0 = loads_34;
    (&kernelContext_86)->bonds_0 = bonds_35;
    (&kernelContext_86)->args_0 = args_34;
    (&kernelContext_86)->info_0 = info_34;
    (&kernelContext_86)->scan_0 = scan_34;
    (&kernelContext_86)->parts_0 = parts_34;
    (&kernelContext_86)->colliders_0 = colliders_34;
    (&kernelContext_86)->collider_info_0 = collider_info_34;
    (&kernelContext_86)->view_chunks_0 = view_chunks_34;
    (&kernelContext_86)->view_0 = view_34;
    (&kernelContext_86)->objects_0 = objects_34;
    (&kernelContext_86)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_86)->lines_0 = lines_34;
    (&kernelContext_86)->geometry_0 = geometry_34;
    (&kernelContext_86)->shots_0 = shots_34;
    (&kernelContext_86)->crush_0 = crush_34;
    (&kernelContext_86)->crush_rw_0 = crush_rw_34;
    (&kernelContext_86)->crush_tab_0 = crush_tab_34;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_86)->g_scan_0 = &g_scan_34;
    uint k_27 = id_31.x;
    bool _S717;
    if(k_27 >= (params_34->chunk_count_0))
    {
        _S717 = true;
    }
    else
    {
        _S717 = ((&kernelContext_86)->params_0->shot_count_0) == 0U;
    }
    if(_S717)
    {
        _S717 = true;
    }
    else
    {
        _S717 = ((&kernelContext_86)->crush_0->enabled_0) == 0U;
    }
    if(_S717)
    {
        return;
    }
    thread uint b_18;
    thread float4 p1_1;
    thread float4 p2_1;
    bool _S718 = cover_patch_0(k_27, &b_18, &p1_1, &p2_1, &kernelContext_86);
    if(!_S718)
    {
        return;
    }
    uint _S719 = a_cover_0(&kernelContext_86);
    uint _S720 = _S719 + 128U * b_18;
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
        float _S721 = cover_depth_0(k_27, s_11, p1_1.xyz, p2_1.xyz, p2_1.w, p1_1.w, &kernelContext_86);
        if(_S721 < 5.0000000752373311e+29f)
        {
            uint _S722 = atomic_fetch_min_explicit((&kernelContext_86)->atom_0+(_S720 + 2U * s_11), ordered_0(_S721), memory_order_relaxed);
        }
        s_11 = s_11 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_claim(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], atomic<uint> device* atom_35 [[buffer(5)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], CrushParams_0 constant* crush_35 [[buffer(25)]], packed_float4 device* crush_rw_35 [[buffer(24)]], packed_float4 device* crush_tab_35 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_87;
    (&kernelContext_87)->params_0 = params_35;
    (&kernelContext_87)->work_0 = work_35;
    (&kernelContext_87)->frames_0 = frames_35;
    (&kernelContext_87)->bodies_0 = bodies_35;
    (&kernelContext_87)->chunks_0 = chunks_35;
    (&kernelContext_87)->sort_0 = sort_35;
    (&kernelContext_87)->atom_0 = atom_35;
    (&kernelContext_87)->manifolds_0 = manifolds_35;
    (&kernelContext_87)->loads_0 = loads_35;
    (&kernelContext_87)->bonds_0 = bonds_36;
    (&kernelContext_87)->args_0 = args_35;
    (&kernelContext_87)->info_0 = info_35;
    (&kernelContext_87)->scan_0 = scan_35;
    (&kernelContext_87)->parts_0 = parts_35;
    (&kernelContext_87)->colliders_0 = colliders_35;
    (&kernelContext_87)->collider_info_0 = collider_info_35;
    (&kernelContext_87)->view_chunks_0 = view_chunks_35;
    (&kernelContext_87)->view_0 = view_35;
    (&kernelContext_87)->objects_0 = objects_35;
    (&kernelContext_87)->shot_bodies_0 = shot_bodies_35;
    (&kernelContext_87)->lines_0 = lines_35;
    (&kernelContext_87)->geometry_0 = geometry_35;
    (&kernelContext_87)->shots_0 = shots_35;
    (&kernelContext_87)->crush_0 = crush_35;
    (&kernelContext_87)->crush_rw_0 = crush_rw_35;
    (&kernelContext_87)->crush_tab_0 = crush_tab_35;
    threadgroup array<uint, int(1024)> g_scan_35;
    (&kernelContext_87)->g_scan_0 = &g_scan_35;
    uint k_28 = id_32.x;
    bool _S723;
    if(k_28 >= (params_35->chunk_count_0))
    {
        _S723 = true;
    }
    else
    {
        _S723 = ((&kernelContext_87)->params_0->shot_count_0) == 0U;
    }
    if(_S723)
    {
        _S723 = true;
    }
    else
    {
        _S723 = ((&kernelContext_87)->crush_0->enabled_0) == 0U;
    }
    if(_S723)
    {
        return;
    }
    thread uint b_19;
    thread float4 p1_2;
    thread float4 p2_2;
    bool _S724 = cover_patch_0(k_28, &b_19, &p1_2, &p2_2, &kernelContext_87);
    if(!_S724)
    {
        return;
    }
    uint _S725 = a_cover_0(&kernelContext_87);
    uint _S726 = _S725 + 128U * b_19;
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
        uint _S727 = _S726 + 2U * s_12;
        uint first_4 = atomic_load_explicit((&kernelContext_87)->atom_0+_S727, memory_order_relaxed);
        if(first_4 == 4294967295U)
        {
            s_12 = s_12 + 1U;
            continue;
        }
        float _S728 = cover_depth_0(k_28, s_12, p1_2.xyz, p2_2.xyz, p2_2.w, p1_2.w, &kernelContext_87);
        if(_S728 < 5.0000000752373311e+29f)
        {
            _S723 = _S728 <= (unordered_0(first_4) + 0.00009999999747379f);
        }
        else
        {
            _S723 = false;
        }
        if(_S723)
        {
            uint _S729 = atomic_fetch_min_explicit((&kernelContext_87)->atom_0+(_S727 + 1U), k_28, memory_order_relaxed);
        }
        s_12 = s_12 + 1U;
    }
    return;
}

bool same_body_0(uint j_8, uint k_29, KernelContext_0 thread* kernelContext_88)
{
    return (as_type<uint>(((float4(*(kernelContext_88->shots_0+4U * j_8)) ).x))) == (as_type<uint>(((float4(*(kernelContext_88->shots_0+4U * k_29)) ).x)));
}

bool same_body_1(uint j_9, uint k_30, KernelContext_0 thread* kernelContext_89)
{
    return (as_type<uint>(((float4(*(kernelContext_89->shots_0+4U * j_9)) ).x))) == (as_type<uint>(((float4(*(kernelContext_89->shots_0+4U * k_30)) ).x)));
}

uint lead_of_0(uint k_31, KernelContext_0 thread* kernelContext_90)
{
    uint j_10 = 0U;
    for(;;)
    {
        if(j_10 < k_31)
        {
        }
        else
        {
            break;
        }
        bool _S730 = same_body_1(j_10, k_31, kernelContext_90);
        if(_S730)
        {
            return j_10;
        }
        j_10 = j_10 + 1U;
    }
    return k_31;
}

uint lead_of_1(uint k_32, KernelContext_0 thread* kernelContext_91)
{
    uint j_11 = 0U;
    for(;;)
    {
        if(j_11 < k_32)
        {
        }
        else
        {
            break;
        }
        bool _S731 = same_body_1(j_11, k_32, kernelContext_91);
        if(_S731)
        {
            return j_11;
        }
        j_11 = j_11 + 1U;
    }
    return k_32;
}

bool lead_shot_0(uint k_33, KernelContext_0 thread* kernelContext_92)
{
    uint _S732 = lead_of_1(k_33, kernelContext_92);
    return _S732 == k_33;
}

float strike_load_0(uint k_34, uint slot_8, float3 f_13, float3 at_11, float tau_5, float budget_0, KernelContext_0 thread* kernelContext_93)
{
    bool _S733 = crushable_0(k_34, kernelContext_93);
    bool _S734;
    if(!_S733)
    {
        _S734 = true;
    }
    else
    {
        uint _S735 = crush_flags_0(k_34, kernelContext_93);
        _S734 = (_S735 & 1U) != 0U;
    }
    if(_S734)
    {
        _S734 = true;
    }
    else
    {
        _S734 = !(budget_0 > 0.0f);
    }
    if(_S734)
    {
        return 0.0f;
    }
    thread Quat_0 q_29;
    thread float3 origin_10;
    structure_pose_0(slot_8, 7U * slot_8, &q_29, &origin_10, kernelContext_93);
    float3 _S736 = (float4(*(kernelContext_93->chunks_0+4U * k_34)) ).xyz;
    thread Quat_0 _S737 = q_29;
    float3 _S738 = rotate_0(&_S737, _S736);
    float3 centre_5 = origin_10 + _S738;
    thread Quat_0 _S739 = q_29;
    float4 _S740 = bond_centroid_0(k_34, &_S739, origin_10, centre_5, kernelContext_93);
    if(!((_S740.w) > 0.0f))
    {
        return 0.0f;
    }
    Sym_0 _S741 = sym_outer_0(_S740.xyz, - f_13);
    thread Sym_0 _S742 = sym_outer_0(at_11 - centre_5, f_13);
    thread Sym_0 _S743 = _S741;
    Sym_0 _S744 = sym_add_0(&_S742, &_S743);
    float before_0 = (float4(*(kernelContext_93->crush_rw_0+k_34)) ).x;
    float _S745 = min(tau_5, kernelContext_93->params_0->dt_0);
    thread Sym_0 _S746 = _S744;
    thread Sym_0 _S747 = _S744;
    float4 _S748 = crush_two_0(&_S746, _S745, &_S747, 0.0f, k_34, before_0, kernelContext_93);
    uint _S749 = 2U * k_34;
    float per_0 = (float4(*(kernelContext_93->crush_tab_0+_S749)) ).w * (float4(*(kernelContext_93->crush_tab_0+(_S749 + 1U))) ).y;
    float energy_0 = (_S748.x - before_0) * per_0;
    float energy_1;
    float4 step_6;
    if(energy_0 > budget_0)
    {
        float d_11 = before_0 + budget_0 / per_0;
        float _S750 = min(d_11, 1.0f);
        float _S751 = _S748.y;
        float _S752 = _S748.z;
        if(d_11 >= 1.0f)
        {
            energy_1 = 1.0f;
        }
        else
        {
            energy_1 = 0.0f;
        }
        step_6 = float4(_S750, _S751, _S752, energy_1);
        energy_1 = budget_0;
    }
    else
    {
        step_6 = _S748;
        energy_1 = energy_0;
    }
    crush_record_0(k_34, step_6, 2U, kernelContext_93);
    return energy_1;
}

bool cell_in_0(uint2 mask_0, uint c_18)
{
    return (((mask_0[c_18 / 32U]) >> (c_18 % 32U)) & 1U) != 0U;
}

float strike_crush_body_0(uint b_20, uint s_13, float budget_1, KernelContext_0 thread* kernelContext_94)
{
    uint total_1;
    bool _S753;
    uint bill_2 = kernelContext_94->crush_0->r_bill_0 + 4U * b_20;
    uint _S754 = bill_2 + 1U;
    float4 _S755 = float4(*(kernelContext_94->crush_rw_0+_S754)) ;
    float4 _S756 = float4(*(kernelContext_94->crush_rw_0+(bill_2 + 2U))) ;
    float4 _S757 = float4(*(kernelContext_94->crush_rw_0+(bill_2 + 3U))) ;
    float3 contact_4 = _S755.xyz;
    float3 d_12 = _S756.xyz;
    float _S758 = _S756.w;
    float _S759 = _S755.w;
    uint _S760 = a_cover_0(kernelContext_94);
    uint _S761 = _S760 + 128U * b_20;
    float first_5 = 1.00000001504746622e+30f;
    uint c_19 = 0U;
    for(;;)
    {
        if(c_19 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S762 = _S761 + 2U * c_19;
        uint _S763 = atomic_load_explicit(kernelContext_94->atom_0+(_S762 + 1U), memory_order_relaxed);
        if(_S763 == 4294967295U)
        {
            c_19 = c_19 + 1U;
            continue;
        }
        thread float rho_2;
        float3 _S764 = cover_offset_0(c_19, d_12, _S758, &rho_2);
        uint _S765 = atomic_load_explicit(kernelContext_94->atom_0+_S762, memory_order_relaxed);
        first_5 = min(first_5, unordered_0(_S765) + _S759 - sqrt(max(_S759 * _S759 - rho_2 * rho_2, 0.0f)));
        c_19 = c_19 + 1U;
    }
    thread uint2 counted_0 = uint2(0U) ;
    c_19 = 0U;
    uint total_2 = 0U;
    for(;;)
    {
        if(c_19 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S766 = _S761 + 2U * c_19;
        uint _S767 = atomic_load_explicit(kernelContext_94->atom_0+(_S766 + 1U), memory_order_relaxed);
        if(_S767 == 4294967295U)
        {
            c_19 = c_19 + 1U;
            continue;
        }
        thread float rho_3;
        float3 _S768 = cover_offset_0(c_19, d_12, _S758, &rho_3);
        uint _S769 = atomic_load_explicit(kernelContext_94->atom_0+_S766, memory_order_relaxed);
        if((unordered_0(_S769) + _S759 - sqrt(max(_S759 * _S759 - rho_3 * rho_3, 0.0f)) - first_5) <= (_S757.z))
        {
            counted_0[c_19 / 32U] = (counted_0[c_19 / 32U]) | (1U << (c_19 % 32U));
            total_1 = total_2 + 1U;
        }
        else
        {
            total_1 = total_2;
        }
        total_2 = total_1;
        c_19 = c_19 + 1U;
    }
    float3 _S770 = float3(0.0f) ;
    float energy_2;
    float3 point_1;
    if(total_2 == 0U)
    {
        float _S771 = strike_load_0(uint(_S757.w) - 1U, b_20, d_12 * float3(_S757.x) , contact_4, _S757.y, budget_1, kernelContext_94);
        float3 _S772 = float3(_S771)  * contact_4;
        energy_2 = _S771;
        point_1 = _S772;
    }
    else
    {
        energy_2 = 0.0f;
        point_1 = _S770;
    }
    c_19 = 0U;
    for(;;)
    {
        if(c_19 < 64U)
        {
        }
        else
        {
            break;
        }
        if(!cell_in_0(counted_0, c_19))
        {
            c_19 = c_19 + 1U;
            continue;
        }
        uint k_35 = atomic_load_explicit(kernelContext_94->atom_0+(_S761 + 2U * c_19 + 1U), memory_order_relaxed);
        bool seen_0 = false;
        total_1 = 0U;
        for(;;)
        {
            if(total_1 < c_19)
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
                    uint _S773 = atomic_load_explicit(kernelContext_94->atom_0+(_S761 + 2U * total_1 + 1U), memory_order_relaxed);
                    _S753 = _S773 == k_35;
                }
                else
                {
                    _S753 = false;
                }
                seen_0 = _S753;
            }
            total_1 = total_1 + 1U;
        }
        if(seen_0)
        {
            c_19 = c_19 + 1U;
            continue;
        }
        uint e_5 = c_19;
        float3 at_12 = _S770;
        uint n_5 = 0U;
        for(;;)
        {
            if(e_5 < 64U)
            {
            }
            else
            {
                break;
            }
            if(!cell_in_0(counted_0, e_5))
            {
                _S753 = true;
            }
            else
            {
                uint _S774 = atomic_load_explicit(kernelContext_94->atom_0+(_S761 + 2U * e_5 + 1U), memory_order_relaxed);
                _S753 = _S774 != k_35;
            }
            if(_S753)
            {
                e_5 = e_5 + 1U;
                continue;
            }
            thread float rho_4;
            float3 _S775 = contact_4 + cover_offset_0(e_5, d_12, _S758, &rho_4);
            uint _S776 = atomic_load_explicit(kernelContext_94->atom_0+(_S761 + 2U * e_5), memory_order_relaxed);
            uint _S777 = n_5 + 1U;
            at_12 = at_12 + (_S775 + d_12 * float3(unordered_0(_S776)) );
            n_5 = _S777;
            e_5 = e_5 + 1U;
        }
        float _S778 = float(n_5);
        float3 at_13 = at_12 / float3(_S778) ;
        float _S779 = strike_load_0(k_35, b_20, d_12 * float3((_S757.x * _S778 / float(total_2))) , at_13, _S757.y, budget_1 - energy_2, kernelContext_94);
        float3 point_2 = point_1 + float3(_S779)  * at_13;
        energy_2 = energy_2 + _S779;
        point_1 = point_2;
        c_19 = c_19 + 1U;
    }
    *(kernelContext_94->crush_rw_0+bill_2) = packed_float4(float4(energy_2, float(s_13 + 1U), 0.0f, 0.0f)) ;
    if(energy_2 > 0.0f)
    {
        *(kernelContext_94->crush_rw_0+_S754) = packed_float4(float4(point_1 / float3(energy_2) , 0.0f)) ;
    }
    return energy_2;
}

[[kernel]] void fr_strike_crush(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], atomic<uint> device* atom_36 [[buffer(5)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], CrushParams_0 constant* crush_36 [[buffer(25)]], packed_float4 device* crush_rw_36 [[buffer(24)]], packed_float4 device* crush_tab_36 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_95;
    (&kernelContext_95)->params_0 = params_36;
    (&kernelContext_95)->work_0 = work_36;
    (&kernelContext_95)->frames_0 = frames_36;
    (&kernelContext_95)->bodies_0 = bodies_36;
    (&kernelContext_95)->chunks_0 = chunks_36;
    (&kernelContext_95)->sort_0 = sort_36;
    (&kernelContext_95)->atom_0 = atom_36;
    (&kernelContext_95)->manifolds_0 = manifolds_36;
    (&kernelContext_95)->loads_0 = loads_36;
    (&kernelContext_95)->bonds_0 = bonds_37;
    (&kernelContext_95)->args_0 = args_36;
    (&kernelContext_95)->info_0 = info_36;
    (&kernelContext_95)->scan_0 = scan_36;
    (&kernelContext_95)->parts_0 = parts_36;
    (&kernelContext_95)->colliders_0 = colliders_36;
    (&kernelContext_95)->collider_info_0 = collider_info_36;
    (&kernelContext_95)->view_chunks_0 = view_chunks_36;
    (&kernelContext_95)->view_0 = view_36;
    (&kernelContext_95)->objects_0 = objects_36;
    (&kernelContext_95)->shot_bodies_0 = shot_bodies_36;
    (&kernelContext_95)->lines_0 = lines_36;
    (&kernelContext_95)->geometry_0 = geometry_36;
    (&kernelContext_95)->shots_0 = shots_36;
    (&kernelContext_95)->crush_0 = crush_36;
    (&kernelContext_95)->crush_rw_0 = crush_rw_36;
    (&kernelContext_95)->crush_tab_0 = crush_tab_36;
    threadgroup array<uint, int(1024)> g_scan_36;
    (&kernelContext_95)->g_scan_0 = &g_scan_36;
    uint s_14 = id_33.x;
    bool _S780;
    if(s_14 >= (params_36->shot_count_0))
    {
        _S780 = true;
    }
    else
    {
        _S780 = ((&kernelContext_95)->crush_0->enabled_0) == 0U;
    }
    if(_S780)
    {
        _S780 = true;
    }
    else
    {
        bool _S781 = lead_shot_0(s_14, &kernelContext_95);
        _S780 = !_S781;
    }
    if(_S780)
    {
        return;
    }
    uint sb_3 = 7U * (as_type<uint>(((float4(*((&kernelContext_95)->shots_0+4U * s_14)) ).x)));
    float inv_mass_7 = (float4(*((&kernelContext_95)->bodies_0+sb_3)) ).w;
    float3 v_11 = (float4(*((&kernelContext_95)->bodies_0+(sb_3 + 3U))) ).xyz;
    float budget_2;
    if(inv_mass_7 > 0.0f)
    {
        budget_2 = 0.5f * dot(v_11, v_11) / inv_mass_7;
    }
    else
    {
        budget_2 = 1.00000001504746622e+30f;
    }
    uint b_21 = 0U;
    for(;;)
    {
        if(b_21 < ((&kernelContext_95)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        thread uint shot_7;
        bool _S782 = struck_0(b_21, &shot_7, &kernelContext_95);
        if(!_S782)
        {
            _S780 = true;
        }
        else
        {
            bool _S783 = same_body_0(shot_7, s_14, &kernelContext_95);
            _S780 = !_S783;
        }
        bool _S784;
        if(_S780)
        {
            _S784 = true;
        }
        else
        {
            _S784 = ((float4(*((&kernelContext_95)->crush_rw_0+((&kernelContext_95)->crush_0->r_bill_0 + 4U * b_21 + 3U))) ).w) == 0.0f;
        }
        if(_S784)
        {
            b_21 = b_21 + 1U;
            continue;
        }
        float _S785 = strike_crush_body_0(b_21, shot_7, budget_2, &kernelContext_95);
        budget_2 = budget_2 - _S785;
        b_21 = b_21 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bill(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], atomic<uint> device* atom_37 [[buffer(5)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], CrushParams_0 constant* crush_37 [[buffer(25)]], packed_float4 device* crush_rw_37 [[buffer(24)]], packed_float4 device* crush_tab_37 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_96;
    (&kernelContext_96)->params_0 = params_37;
    (&kernelContext_96)->work_0 = work_37;
    (&kernelContext_96)->frames_0 = frames_37;
    (&kernelContext_96)->bodies_0 = bodies_37;
    (&kernelContext_96)->chunks_0 = chunks_37;
    (&kernelContext_96)->sort_0 = sort_37;
    (&kernelContext_96)->atom_0 = atom_37;
    (&kernelContext_96)->manifolds_0 = manifolds_37;
    (&kernelContext_96)->loads_0 = loads_37;
    (&kernelContext_96)->bonds_0 = bonds_38;
    (&kernelContext_96)->args_0 = args_37;
    (&kernelContext_96)->info_0 = info_37;
    (&kernelContext_96)->scan_0 = scan_37;
    (&kernelContext_96)->parts_0 = parts_37;
    (&kernelContext_96)->colliders_0 = colliders_37;
    (&kernelContext_96)->collider_info_0 = collider_info_37;
    (&kernelContext_96)->view_chunks_0 = view_chunks_37;
    (&kernelContext_96)->view_0 = view_37;
    (&kernelContext_96)->objects_0 = objects_37;
    (&kernelContext_96)->shot_bodies_0 = shot_bodies_37;
    (&kernelContext_96)->lines_0 = lines_37;
    (&kernelContext_96)->geometry_0 = geometry_37;
    (&kernelContext_96)->shots_0 = shots_37;
    (&kernelContext_96)->crush_0 = crush_37;
    (&kernelContext_96)->crush_rw_0 = crush_rw_37;
    (&kernelContext_96)->crush_tab_0 = crush_tab_37;
    threadgroup array<uint, int(1024)> g_scan_37;
    (&kernelContext_96)->g_scan_0 = &g_scan_37;
    uint s_15 = id_34.x;
    bool _S786;
    if(s_15 >= (params_37->shot_count_0))
    {
        _S786 = true;
    }
    else
    {
        _S786 = ((&kernelContext_96)->crush_0->enabled_0) == 0U;
    }
    if(_S786)
    {
        return;
    }
    *((&kernelContext_96)->crush_rw_0+((&kernelContext_96)->crush_0->r_shot_0 + s_15)) = packed_float4(float4(0.0f) ) ;
    bool _S787 = lead_shot_0(s_15, &kernelContext_96);
    if(!_S787)
    {
        return;
    }
    uint b_22 = 0U;
    float energy_3 = 0.0f;
    for(;;)
    {
        if(b_22 < ((&kernelContext_96)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S788 = float4(*((&kernelContext_96)->crush_rw_0+((&kernelContext_96)->crush_0->r_bill_0 + 4U * b_22))) ;
        float _S789 = _S788.y;
        if(_S789 > 0.0f)
        {
            bool _S790 = same_body_0(uint(_S789) - 1U, s_15, &kernelContext_96);
            _S786 = _S790;
        }
        else
        {
            _S786 = false;
        }
        if(_S786)
        {
            energy_3 = energy_3 + _S788.x;
        }
        b_22 = b_22 + 1U;
    }
    if(!(energy_3 > 0.0f))
    {
        return;
    }
    uint sb_4 = 7U * (as_type<uint>(((float4(*((&kernelContext_96)->shots_0+4U * s_15)) ).x)));
    float inv_mass_8 = (float4(*((&kernelContext_96)->bodies_0+sb_4)) ).w;
    if(!(inv_mass_8 > 0.0f))
    {
        return;
    }
    float m_6 = 1.0f / inv_mass_8;
    float3 v_12 = (float4(*((&kernelContext_96)->bodies_0+(sb_4 + 3U))) ).xyz;
    float speed_2 = length(v_12);
    float j_12 = m_6 * (speed_2 - sqrt(max(speed_2 * speed_2 - 2.0f * energy_3 / m_6, 0.0f)));
    bool _S791 = speed_2 > 0.0f;
    float3 _S792;
    if(_S791)
    {
        _S792 = - v_12 * float3((j_12 / (m_6 * speed_2))) ;
    }
    else
    {
        _S792 = float3(0.0f) ;
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
        uint w_8 = sb_4 + region_1 * 7U * (&kernelContext_96)->params_0->max_bodies_0 + 3U;
        *((&kernelContext_96)->bodies_0+w_8) = packed_float4(float4((float4(*((&kernelContext_96)->bodies_0+w_8)) ).xyz + _S792, (float4(*((&kernelContext_96)->bodies_0+w_8)) ).w)) ;
        region_1 = region_1 + 1U;
    }
    packed_float4 device* _S793 = (&kernelContext_96)->crush_rw_0+((&kernelContext_96)->crush_0->r_shot_0 + s_15);
    if(_S791)
    {
        _S792 = v_12 * float3((j_12 / speed_2)) ;
    }
    else
    {
        _S792 = float3(0.0f) ;
    }
    *_S793 = packed_float4(float4(_S792, energy_3)) ;
    return;
}

float3 inverse_rotate_0(const Quat_0 thread* q_30, float3 v_13)
{
    thread Quat_0 c_20;
    (&c_20)->w_0 = q_30->w_0;
    (&c_20)->x_0 = - q_30->x_0;
    (&c_20)->y_0 = - q_30->y_0;
    (&c_20)->z_0 = - q_30->z_0;
    thread Quat_0 _S794 = c_20;
    float3 _S795 = rotate_0(&_S794, v_13);
    return _S795;
}

[[kernel]] void fr_crush_push(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], atomic<uint> device* atom_38 [[buffer(5)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], CrushParams_0 constant* crush_38 [[buffer(25)]], packed_float4 device* crush_rw_38 [[buffer(24)]], packed_float4 device* crush_tab_38 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_97;
    (&kernelContext_97)->params_0 = params_38;
    (&kernelContext_97)->work_0 = work_38;
    (&kernelContext_97)->frames_0 = frames_38;
    (&kernelContext_97)->bodies_0 = bodies_38;
    (&kernelContext_97)->chunks_0 = chunks_38;
    (&kernelContext_97)->sort_0 = sort_38;
    (&kernelContext_97)->atom_0 = atom_38;
    (&kernelContext_97)->manifolds_0 = manifolds_38;
    (&kernelContext_97)->loads_0 = loads_38;
    (&kernelContext_97)->bonds_0 = bonds_39;
    (&kernelContext_97)->args_0 = args_38;
    (&kernelContext_97)->info_0 = info_38;
    (&kernelContext_97)->scan_0 = scan_38;
    (&kernelContext_97)->parts_0 = parts_38;
    (&kernelContext_97)->colliders_0 = colliders_38;
    (&kernelContext_97)->collider_info_0 = collider_info_38;
    (&kernelContext_97)->view_chunks_0 = view_chunks_38;
    (&kernelContext_97)->view_0 = view_38;
    (&kernelContext_97)->objects_0 = objects_38;
    (&kernelContext_97)->shot_bodies_0 = shot_bodies_38;
    (&kernelContext_97)->lines_0 = lines_38;
    (&kernelContext_97)->geometry_0 = geometry_38;
    (&kernelContext_97)->shots_0 = shots_38;
    (&kernelContext_97)->crush_0 = crush_38;
    (&kernelContext_97)->crush_rw_0 = crush_rw_38;
    (&kernelContext_97)->crush_tab_0 = crush_tab_38;
    threadgroup array<uint, int(1024)> g_scan_38;
    (&kernelContext_97)->g_scan_0 = &g_scan_38;
    uint b_23 = id_35.x;
    bool _S796;
    if(b_23 >= (params_38->body_count_0))
    {
        _S796 = true;
    }
    else
    {
        _S796 = ((&kernelContext_97)->params_0->shot_count_0) == 0U;
    }
    if(_S796)
    {
        _S796 = true;
    }
    else
    {
        _S796 = ((&kernelContext_97)->crush_0->enabled_0) == 0U;
    }
    if(_S796)
    {
        return;
    }
    uint _S797 = 4U * b_23;
    float4 _S798 = float4(*((&kernelContext_97)->crush_rw_0+((&kernelContext_97)->crush_0->r_bill_0 + _S797))) ;
    float _S799 = _S798.x;
    if(!(_S799 > 0.0f))
    {
        return;
    }
    uint _S800 = lead_of_0(uint(_S798.y) - 1U, &kernelContext_97);
    float4 _S801 = float4(*((&kernelContext_97)->crush_rw_0+((&kernelContext_97)->crush_0->r_shot_0 + _S800))) ;
    uint bw_0 = 7U * b_23;
    float inv_mass_9 = (float4(*((&kernelContext_97)->bodies_0+bw_0)) ).w;
    float _S802 = _S801.w;
    if(!(_S802 > 0.0f))
    {
        _S796 = true;
    }
    else
    {
        _S796 = inv_mass_9 == 0.0f;
    }
    if(_S796)
    {
        return;
    }
    float3 j_13 = _S801.xyz * float3((_S799 / _S802)) ;
    Quat_0 o_7 = quat_of_0(float4(*((&kernelContext_97)->bodies_0+(bw_0 + 1U))) );
    float3 inv_i_0 = (float4(*((&kernelContext_97)->bodies_0+(bw_0 + 2U))) ).xyz;
    float3 _S803 = cross(to_rubble_0((float4(*((&kernelContext_97)->crush_rw_0+((&kernelContext_97)->crush_0->r_bill_0 + _S797 + 1U))) ).xyz) - (float4(*((&kernelContext_97)->bodies_0+bw_0)) ).xyz, j_13);
    thread Quat_0 _S804 = o_7;
    float3 _S805 = inverse_rotate_0(&_S804, _S803);
    float3 _S806 = inv_i_0 * _S805;
    thread Quat_0 _S807 = o_7;
    float3 _S808 = rotate_0(&_S807, _S806);
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
        uint w_9 = bw_0 + region_2 * 7U * (&kernelContext_97)->params_0->max_bodies_0;
        uint _S809 = w_9 + 3U;
        *((&kernelContext_97)->bodies_0+_S809) = packed_float4(float4((float4(*((&kernelContext_97)->bodies_0+_S809)) ).xyz + j_13 * float3(inv_mass_9) , (float4(*((&kernelContext_97)->bodies_0+_S809)) ).w)) ;
        uint _S810 = w_9 + 4U;
        *((&kernelContext_97)->bodies_0+_S810) = packed_float4(float4((float4(*((&kernelContext_97)->bodies_0+_S810)) ).xyz + _S808, (float4(*((&kernelContext_97)->bodies_0+_S810)) ).w)) ;
        region_2 = region_2 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bonds(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], atomic<uint> device* atom_39 [[buffer(5)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], CrushParams_0 constant* crush_39 [[buffer(25)]], packed_float4 device* crush_rw_39 [[buffer(24)]], packed_float4 device* crush_tab_39 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_98;
    (&kernelContext_98)->params_0 = params_39;
    (&kernelContext_98)->work_0 = work_39;
    (&kernelContext_98)->frames_0 = frames_39;
    (&kernelContext_98)->bodies_0 = bodies_39;
    (&kernelContext_98)->chunks_0 = chunks_39;
    (&kernelContext_98)->sort_0 = sort_39;
    (&kernelContext_98)->atom_0 = atom_39;
    (&kernelContext_98)->manifolds_0 = manifolds_39;
    (&kernelContext_98)->loads_0 = loads_39;
    (&kernelContext_98)->bonds_0 = bonds_40;
    (&kernelContext_98)->args_0 = args_39;
    (&kernelContext_98)->info_0 = info_39;
    (&kernelContext_98)->scan_0 = scan_39;
    (&kernelContext_98)->parts_0 = parts_39;
    (&kernelContext_98)->colliders_0 = colliders_39;
    (&kernelContext_98)->collider_info_0 = collider_info_39;
    (&kernelContext_98)->view_chunks_0 = view_chunks_39;
    (&kernelContext_98)->view_0 = view_39;
    (&kernelContext_98)->objects_0 = objects_39;
    (&kernelContext_98)->shot_bodies_0 = shot_bodies_39;
    (&kernelContext_98)->lines_0 = lines_39;
    (&kernelContext_98)->geometry_0 = geometry_39;
    (&kernelContext_98)->shots_0 = shots_39;
    (&kernelContext_98)->crush_0 = crush_39;
    (&kernelContext_98)->crush_rw_0 = crush_rw_39;
    (&kernelContext_98)->crush_tab_0 = crush_tab_39;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_98)->g_scan_0 = &g_scan_39;
    uint i_10 = id_36.x;
    bool _S811;
    if(i_10 >= (params_39->bond_count_0))
    {
        _S811 = true;
    }
    else
    {
        _S811 = ((&kernelContext_98)->crush_0->enabled_0) == 0U;
    }
    if(_S811)
    {
        _S811 = true;
    }
    else
    {
        uint _S812 = w_alive_0(&kernelContext_98);
        _S811 = (*((&kernelContext_98)->work_0+(_S812 + i_10))) == 0U;
    }
    if(_S811)
    {
        return;
    }
    uint4 _S813 = uint4(*((&kernelContext_98)->bonds_0+i_10)) ;
    uint _S814 = _S813.x;
    if((((as_type<uint>(((float4(*((&kernelContext_98)->crush_rw_0+_S814)) ).w))) | (as_type<uint>(((float4(*((&kernelContext_98)->crush_rw_0+_S813.y)) ).w)))) & 2U) == 0U)
    {
        return;
    }
    uint _S815 = w_alive_0(&kernelContext_98);
    *((&kernelContext_98)->work_0+(_S815 + i_10)) = 0U;
    uint _S816 = a_dirty_0(&kernelContext_98);
    uint _S817 = body_of_0(_S814, &kernelContext_98);
    uint _S818 = atomic_fetch_or_explicit((&kernelContext_98)->atom_0+(_S816 + _S817), 1U, memory_order_relaxed);
    uint _S819 = a_counters_0(&kernelContext_98);
    uint _S820 = atomic_fetch_add_explicit((&kernelContext_98)->atom_0+_S819, 1U, memory_order_relaxed);
    uint _S821 = atomic_fetch_add_explicit((&kernelContext_98)->atom_0+(_S819 + 1U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_crush_mark(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], atomic<uint> device* atom_40 [[buffer(5)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], CrushParams_0 constant* crush_40 [[buffer(25)]], packed_float4 device* crush_rw_40 [[buffer(24)]], packed_float4 device* crush_tab_40 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_40;
    (&kernelContext_99)->work_0 = work_40;
    (&kernelContext_99)->frames_0 = frames_40;
    (&kernelContext_99)->bodies_0 = bodies_40;
    (&kernelContext_99)->chunks_0 = chunks_40;
    (&kernelContext_99)->sort_0 = sort_40;
    (&kernelContext_99)->atom_0 = atom_40;
    (&kernelContext_99)->manifolds_0 = manifolds_40;
    (&kernelContext_99)->loads_0 = loads_40;
    (&kernelContext_99)->bonds_0 = bonds_41;
    (&kernelContext_99)->args_0 = args_40;
    (&kernelContext_99)->info_0 = info_40;
    (&kernelContext_99)->scan_0 = scan_40;
    (&kernelContext_99)->parts_0 = parts_40;
    (&kernelContext_99)->colliders_0 = colliders_40;
    (&kernelContext_99)->collider_info_0 = collider_info_40;
    (&kernelContext_99)->view_chunks_0 = view_chunks_40;
    (&kernelContext_99)->view_0 = view_40;
    (&kernelContext_99)->objects_0 = objects_40;
    (&kernelContext_99)->shot_bodies_0 = shot_bodies_40;
    (&kernelContext_99)->lines_0 = lines_40;
    (&kernelContext_99)->geometry_0 = geometry_40;
    (&kernelContext_99)->shots_0 = shots_40;
    (&kernelContext_99)->crush_0 = crush_40;
    (&kernelContext_99)->crush_rw_0 = crush_rw_40;
    (&kernelContext_99)->crush_tab_0 = crush_tab_40;
    threadgroup array<uint, int(1024)> g_scan_40;
    (&kernelContext_99)->g_scan_0 = &g_scan_40;
    uint c_21 = id_37.x;
    bool _S822;
    if(c_21 >= (params_40->chunk_count_0))
    {
        _S822 = true;
    }
    else
    {
        _S822 = ((&kernelContext_99)->crush_0->enabled_0) == 0U;
    }
    if(_S822)
    {
        return;
    }
    uint _S823 = crush_flags_0(c_21, &kernelContext_99);
    uint now_0;
    if((_S823 & 2U) != 0U)
    {
        now_0 = 1U;
    }
    else
    {
        now_0 = 0U;
    }
    uint _S824 = w_new_0(&kernelContext_99);
    *((&kernelContext_99)->work_0+(_S824 + c_21)) = now_0;
    if(now_0 != 0U)
    {
        uint _S825 = a_counters_0(&kernelContext_99);
        uint _S826 = atomic_fetch_add_explicit((&kernelContext_99)->atom_0+(_S825 + 10U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_crush_gate(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], atomic<uint> device* atom_41 [[buffer(5)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], CrushParams_0 constant* crush_41 [[buffer(25)]], packed_float4 device* crush_rw_41 [[buffer(24)]], packed_float4 device* crush_tab_41 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_100;
    (&kernelContext_100)->params_0 = params_41;
    (&kernelContext_100)->work_0 = work_41;
    (&kernelContext_100)->frames_0 = frames_41;
    (&kernelContext_100)->bodies_0 = bodies_41;
    (&kernelContext_100)->chunks_0 = chunks_41;
    (&kernelContext_100)->sort_0 = sort_41;
    (&kernelContext_100)->atom_0 = atom_41;
    (&kernelContext_100)->manifolds_0 = manifolds_41;
    (&kernelContext_100)->loads_0 = loads_41;
    (&kernelContext_100)->bonds_0 = bonds_42;
    (&kernelContext_100)->args_0 = args_41;
    (&kernelContext_100)->info_0 = info_41;
    (&kernelContext_100)->scan_0 = scan_41;
    (&kernelContext_100)->parts_0 = parts_41;
    (&kernelContext_100)->colliders_0 = colliders_41;
    (&kernelContext_100)->collider_info_0 = collider_info_41;
    (&kernelContext_100)->view_chunks_0 = view_chunks_41;
    (&kernelContext_100)->view_0 = view_41;
    (&kernelContext_100)->objects_0 = objects_41;
    (&kernelContext_100)->shot_bodies_0 = shot_bodies_41;
    (&kernelContext_100)->lines_0 = lines_41;
    (&kernelContext_100)->geometry_0 = geometry_41;
    (&kernelContext_100)->shots_0 = shots_41;
    (&kernelContext_100)->crush_0 = crush_41;
    (&kernelContext_100)->crush_rw_0 = crush_rw_41;
    (&kernelContext_100)->crush_tab_0 = crush_tab_41;
    threadgroup array<uint, int(1024)> g_scan_41;
    (&kernelContext_100)->g_scan_0 = &g_scan_41;
    bool run_2;
    if((crush_41->enabled_0) != 0U)
    {
        uint _S827 = a_counters_0(&kernelContext_100);
        uint _S828 = atomic_load_explicit((&kernelContext_100)->atom_0+(_S827 + 10U), memory_order_relaxed);
        run_2 = _S828 != 0U;
    }
    else
    {
        run_2 = false;
    }
    uint _S829 = a_counters_0(&kernelContext_100);
    atomic_store_explicit((&kernelContext_100)->atom_0+(_S829 + 10U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_100)->atom_0+(_S829 + 7U), 0U, memory_order_relaxed);
    uint _S830;
    if(run_2)
    {
        uint _S831 = groups_of_0((&kernelContext_100)->params_0->chunk_count_0, 64U);
        _S830 = _S831;
    }
    else
    {
        _S830 = 0U;
    }
    set_args_0(6U, _S830, &kernelContext_100);
    if(run_2)
    {
        uint _S832 = groups_of_0((&kernelContext_100)->params_0->chunk_count_0, 1024U);
        _S830 = _S832;
    }
    else
    {
        _S830 = 0U;
    }
    set_args_0(7U, _S830, &kernelContext_100);
    int _S833;
    if(run_2)
    {
        _S833 = int(1);
    }
    else
    {
        _S833 = int(0);
    }
    set_args_0(8U, uint(_S833), &kernelContext_100);
    return;
}

void write_body_0(uint slot_9, float3 pos_1, float inv_mass_10, const Quat_0 thread* orientation_0, float3 inv_inertia_1, uint flags_3, float3 v_14, float3 w_10, KernelContext_0 thread* kernelContext_101)
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
        uint b_24 = 7U * (slot_9 + region_3 * kernelContext_101->params_0->max_bodies_0);
        *(kernelContext_101->bodies_0+b_24) = packed_float4(float4(pos_1, inv_mass_10)) ;
        packed_float4 device* _S834 = kernelContext_101->bodies_0+(b_24 + 1U);
        float4 _S835 = quat_vec_0(orientation_0);
        *_S834 = packed_float4(_S835) ;
        *(kernelContext_101->bodies_0+(b_24 + 2U)) = packed_float4(float4(inv_inertia_1, (as_type<float>((flags_3))))) ;
        *(kernelContext_101->bodies_0+(b_24 + 3U)) = packed_float4(float4(v_14, 0.0f)) ;
        *(kernelContext_101->bodies_0+(b_24 + 4U)) = packed_float4(float4(w_10, (as_type<float>((kernelContext_101->params_0->step_0))))) ;
        *(kernelContext_101->bodies_0+(b_24 + 5U)) = packed_float4(float4(0.0f) ) ;
        *(kernelContext_101->bodies_0+(b_24 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
        region_3 = region_3 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_apply(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], atomic<uint> device* atom_42 [[buffer(5)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], CrushParams_0 constant* crush_42 [[buffer(25)]], packed_float4 device* crush_rw_42 [[buffer(24)]], packed_float4 device* crush_tab_42 [[buffer(23)]])
{
    uint ax_3;
    thread KernelContext_0 kernelContext_102;
    (&kernelContext_102)->params_0 = params_42;
    (&kernelContext_102)->work_0 = work_42;
    (&kernelContext_102)->frames_0 = frames_42;
    (&kernelContext_102)->bodies_0 = bodies_42;
    (&kernelContext_102)->chunks_0 = chunks_42;
    (&kernelContext_102)->sort_0 = sort_42;
    (&kernelContext_102)->atom_0 = atom_42;
    (&kernelContext_102)->manifolds_0 = manifolds_42;
    (&kernelContext_102)->loads_0 = loads_42;
    (&kernelContext_102)->bonds_0 = bonds_43;
    (&kernelContext_102)->args_0 = args_42;
    (&kernelContext_102)->info_0 = info_42;
    (&kernelContext_102)->scan_0 = scan_42;
    (&kernelContext_102)->parts_0 = parts_42;
    (&kernelContext_102)->colliders_0 = colliders_42;
    (&kernelContext_102)->collider_info_0 = collider_info_42;
    (&kernelContext_102)->view_chunks_0 = view_chunks_42;
    (&kernelContext_102)->view_0 = view_42;
    (&kernelContext_102)->objects_0 = objects_42;
    (&kernelContext_102)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_102)->lines_0 = lines_42;
    (&kernelContext_102)->geometry_0 = geometry_42;
    (&kernelContext_102)->shots_0 = shots_42;
    (&kernelContext_102)->crush_0 = crush_42;
    (&kernelContext_102)->crush_rw_0 = crush_rw_42;
    (&kernelContext_102)->crush_tab_0 = crush_tab_42;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_102)->g_scan_0 = &g_scan_42;
    uint k_36 = id_39.x;
    bool _S836;
    if(k_36 >= (params_42->chunk_count_0))
    {
        _S836 = true;
    }
    else
    {
        _S836 = ((&kernelContext_102)->crush_0->enabled_0) == 0U;
    }
    if(_S836)
    {
        return;
    }
    float4 _S837 = float4(*((&kernelContext_102)->crush_rw_0+k_36)) ;
    uint flags_4 = (as_type<uint>((_S837.w)));
    if((flags_4 & 2U) == 0U)
    {
        return;
    }
    *((&kernelContext_102)->crush_rw_0+k_36) = packed_float4(float4(_S837.xyz, (as_type<float>((flags_4 & 4294967293U))))) ;
    uint _S838 = body_of_0(k_36, &kernelContext_102);
    uint bw_1 = 7U * _S838;
    float4 _S839 = float4(*((&kernelContext_102)->bodies_0+bw_1)) ;
    Quat_0 orientation_1 = quat_of_0(float4(*((&kernelContext_102)->bodies_0+(bw_1 + 1U))) );
    float3 v_15 = (float4(*((&kernelContext_102)->bodies_0+(bw_1 + 3U))) ).xyz;
    float3 w_11 = (float4(*((&kernelContext_102)->bodies_0+(bw_1 + 4U))) ).xyz;
    uint _S840 = 4U * k_36;
    float mass_0 = (float4(*((&kernelContext_102)->chunks_0+_S840)) ).w;
    float volume_1 = (float4(*((&kernelContext_102)->crush_tab_0+(2U * k_36 + 1U))) ).y;
    uint _S841 = w_rank_0(&kernelContext_102);
    uint device* _S842 = (&kernelContext_102)->work_0+(_S841 + k_36);
    uint rank_0 = *_S842;
    if((*_S842) >= ((&kernelContext_102)->crush_0->max_events_0))
    {
        uint _S843 = a_counters_0(&kernelContext_102);
        uint _S844 = atomic_fetch_or_explicit((&kernelContext_102)->atom_0+(_S843 + 3U), 32U, memory_order_relaxed);
    }
    float3 _S845 = float3(0.0f) ;
    if(((&kernelContext_102)->crush_0->mode_1) == 1U)
    {
        _S836 = ((&kernelContext_102)->crush_0->fragments_0) > 0U;
    }
    else
    {
        _S836 = false;
    }
    uint first_6;
    float3 half_2;
    if(_S836)
    {
        uint _S846 = a_counters_0(&kernelContext_102);
        uint _S847 = atomic_load_explicit((&kernelContext_102)->atom_0+(_S846 + 8U), memory_order_relaxed);
        uint base_4 = _S847 + rank_0 * (&kernelContext_102)->crush_0->fragments_0;
        if((base_4 + (&kernelContext_102)->crush_0->fragments_0) > ((&kernelContext_102)->crush_0->pool_size_0))
        {
            uint _S848 = atomic_fetch_or_explicit((&kernelContext_102)->atom_0+(_S846 + 3U), 16U, memory_order_relaxed);
            first_6 = 4294967295U;
        }
        else
        {
            first_6 = base_4;
        }
        thread Quat_0 q_31;
        thread float3 origin_11;
        structure_pose_0(_S838, bw_1, &q_31, &origin_11, &kernelContext_102);
        float3 _S849 = (float4(*((&kernelContext_102)->chunks_0+_S840)) ).xyz;
        thread Quat_0 _S850 = q_31;
        float3 _S851 = rotate_0(&_S850, _S849);
        float3 _S852 = origin_11 + _S851;
        uint _S853 = 3U * k_36;
        Quat_0 _S854 = quat_of_0(float4(*((&kernelContext_102)->geometry_0+(_S853 + 2U))) );
        thread Quat_0 _S855 = q_31;
        thread Quat_0 _S856 = _S854;
        Quat_0 _S857 = quat_mul_0(&_S855, &_S856);
        thread Quat_0 _S858 = _S857;
        Quat_0 _S859 = normalized_0(&_S858);
        float3 h_1 = (float4(*((&kernelContext_102)->geometry_0+(_S853 + 1U))) ).xyz;
        thread uint3 g_0 = uint3(int3(int(1)) );
        thread float3 cell_0 = h_1;
        uint n_6 = (&kernelContext_102)->crush_0->fragments_0;
        for(;;)
        {
            if(n_6 > 1U)
            {
            }
            else
            {
                break;
            }
            if((cell_0.x) >= (cell_0.y))
            {
                _S836 = (cell_0.x) >= (cell_0.z);
            }
            else
            {
                _S836 = false;
            }
            if(_S836)
            {
                ax_3 = 0U;
            }
            else
            {
                if((cell_0.y) >= (cell_0.z))
                {
                    ax_3 = 1U;
                }
                else
                {
                    ax_3 = 2U;
                }
            }
            g_0[ax_3] = g_0[ax_3] * 2U;
            cell_0[ax_3] = cell_0[ax_3] * 0.5f;
            n_6 = n_6 >> 1U;
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
        float3 half_3 = cell_0 * float3(shrink_0) ;
        float mf_0 = mass_0 / float((&kernelContext_102)->crush_0->fragments_0);
        float _S860 = half_3.y;
        float _S861 = _S860 * _S860;
        float _S862 = half_3.z;
        float _S863 = _S862 * _S862;
        float _S864 = half_3.x;
        float _S865 = _S864 * _S864;
        float3 inertia_0 = float3((mf_0 / 3.0f))  * float3(_S861 + _S863, _S865 + _S863, _S865 + _S861);
        float3 _S866 = float3(1.0f / inertia_0.x, 1.0f / inertia_0.y, 1.0f / inertia_0.z);
        thread Quat_0 _S867 = _S859;
        Quat_0 _S868 = rot_to_rubble_0(&_S867);
        float _S869 = (float4(*((&kernelContext_102)->chunks_0+(_S840 + 3U))) ).x;
        float _S870 = min(_S864, min(_S860, _S862));
        ax_3 = 0U;
        for(;;)
        {
            if(ax_3 < ((&kernelContext_102)->crush_0->fragments_0))
            {
                _S836 = first_6 != 4294967295U;
            }
            else
            {
                _S836 = false;
            }
            if(_S836)
            {
            }
            else
            {
                break;
            }
            uint _S871 = ax_3 % g_0.x;
            uint _S872 = ax_3 / g_0.x;
            uint _S873 = _S872 % g_0.y;
            uint _S874 = ax_3 / (g_0.x * g_0.y);
            float3 offset_0 = - h_1 + cell_0 * (float3(2.0f)  * float3(uint3(_S871, _S873, _S874)) + float3(1.0f) );
            thread Quat_0 _S875 = _S859;
            float3 _S876 = rotate_0(&_S875, offset_0);
            float3 x_6 = to_rubble_0(_S852 + _S876);
            float4 _S877 = float4(*((&kernelContext_102)->crush_tab_0+((&kernelContext_102)->crush_0->t_pool_0 + first_6 + ax_3))) ;
            uint fs_0 = (as_type<uint>((_S877.x)));
            uint fc_0 = (as_type<uint>((_S877.y)));
            float _S878 = 1.0f / mf_0;
            float3 _S879 = v_15 + cross(w_11, x_6 - _S839.xyz);
            thread Quat_0 _S880 = _S868;
            write_body_0(fs_0, x_6, _S878, &_S880, _S866, 9U, _S879, w_11, &kernelContext_102);
            uint _S881 = 4U * fc_0;
            uint _S882 = _S881 + 1U;
            *((&kernelContext_102)->colliders_0+_S882) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_102)->colliders_0+_S882)) ).w)) ;
            *((&kernelContext_102)->colliders_0+(_S881 + 2U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
            *((&kernelContext_102)->colliders_0+(_S881 + 3U)) = packed_float4(float4(half_3, length(half_3))) ;
            uint _S883 = 2U * fc_0;
            *((&kernelContext_102)->collider_info_0+_S883) = packed_uint4(uint4(4294967295U, fs_0, 2U, (uint4(*((&kernelContext_102)->collider_info_0+_S883)) ).w)) ;
            *((&kernelContext_102)->collider_info_0+(_S883 + 1U)) = packed_uint4(uint4((as_type<uint>((mf_0))), (as_type<uint>((_S869))), (as_type<uint>((_S870))), 0U)) ;
            ax_3 = ax_3 + 1U;
        }
        half_2 = half_3;
    }
    else
    {
        first_6 = 4294967295U;
        half_2 = _S845;
    }
    if(rank_0 < ((&kernelContext_102)->crush_0->max_events_0))
    {
        uint ev_0 = (&kernelContext_102)->crush_0->r_events_0 + 6U * rank_0;
        *((&kernelContext_102)->crush_rw_0+ev_0) = packed_float4(float4((as_type<float>((k_36))), (as_type<float>((first_6))), (as_type<float>((flags_4 >> 4U))), _S837.x)) ;
        *((&kernelContext_102)->crush_rw_0+(ev_0 + 1U)) = packed_float4(float4(_S839.xyz, mass_0)) ;
        packed_float4 device* _S884 = (&kernelContext_102)->crush_rw_0+(ev_0 + 2U);
        thread Quat_0 _S885 = orientation_1;
        float4 _S886 = quat_vec_0(&_S885);
        *_S884 = packed_float4(_S886) ;
        *((&kernelContext_102)->crush_rw_0+(ev_0 + 3U)) = packed_float4(float4(v_15, _S837.y)) ;
        *((&kernelContext_102)->crush_rw_0+(ev_0 + 4U)) = packed_float4(float4(w_11, _S837.z)) ;
        *((&kernelContext_102)->crush_rw_0+(ev_0 + 5U)) = packed_float4(float4(half_2, volume_1)) ;
    }
    float3 park_0 = float3(float(_S838 % 1024U) * (&kernelContext_102)->crush_0->park_step_0, (&kernelContext_102)->crush_0->park_y_0, float(_S838 / 1024U) * (&kernelContext_102)->crush_0->park_step_0);
    thread Quat_0 _S887 = orientation_1;
    write_body_0(_S838, park_0, 0.0f, &_S887, _S845, 0U, _S845, _S845, &kernelContext_102);
    return;
}

[[kernel]] void fr_crush_end(uint3 id_40 [[thread_position_in_grid]], FractureParams_0 constant* params_43 [[buffer(0)]], uint device* work_43 [[buffer(4)]], packed_float4 device* frames_43 [[buffer(6)]], packed_float4 device* bodies_43 [[buffer(7)]], packed_float4 device* chunks_43 [[buffer(1)]], uint device* sort_43 [[buffer(11)]], atomic<uint> device* atom_43 [[buffer(5)]], Record_natural_0 device* manifolds_43 [[buffer(10)]], packed_float4 device* loads_43 [[buffer(12)]], packed_uint4 device* bonds_44 [[buffer(2)]], uint device* args_43 [[buffer(13)]], uint device* info_43 [[buffer(15)]], ScanParams_0 constant* scan_43 [[buffer(14)]], packed_float4 device* parts_43 [[buffer(3)]], packed_float4 device* colliders_43 [[buffer(8)]], packed_uint4 device* collider_info_43 [[buffer(9)]], packed_float4 device* view_chunks_43 [[buffer(17)]], ViewParams_0 constant* view_43 [[buffer(16)]], packed_float4 device* objects_43 [[buffer(18)]], uint device* shot_bodies_43 [[buffer(20)]], float device* lines_43 [[buffer(19)]], packed_float4 device* geometry_43 [[buffer(21)]], packed_float4 device* shots_43 [[buffer(22)]], CrushParams_0 constant* crush_43 [[buffer(25)]], packed_float4 device* crush_rw_43 [[buffer(24)]], packed_float4 device* crush_tab_43 [[buffer(23)]])
{
    thread KernelContext_0 kernelContext_103;
    (&kernelContext_103)->params_0 = params_43;
    (&kernelContext_103)->work_0 = work_43;
    (&kernelContext_103)->frames_0 = frames_43;
    (&kernelContext_103)->bodies_0 = bodies_43;
    (&kernelContext_103)->chunks_0 = chunks_43;
    (&kernelContext_103)->sort_0 = sort_43;
    (&kernelContext_103)->atom_0 = atom_43;
    (&kernelContext_103)->manifolds_0 = manifolds_43;
    (&kernelContext_103)->loads_0 = loads_43;
    (&kernelContext_103)->bonds_0 = bonds_44;
    (&kernelContext_103)->args_0 = args_43;
    (&kernelContext_103)->info_0 = info_43;
    (&kernelContext_103)->scan_0 = scan_43;
    (&kernelContext_103)->parts_0 = parts_43;
    (&kernelContext_103)->colliders_0 = colliders_43;
    (&kernelContext_103)->collider_info_0 = collider_info_43;
    (&kernelContext_103)->view_chunks_0 = view_chunks_43;
    (&kernelContext_103)->view_0 = view_43;
    (&kernelContext_103)->objects_0 = objects_43;
    (&kernelContext_103)->shot_bodies_0 = shot_bodies_43;
    (&kernelContext_103)->lines_0 = lines_43;
    (&kernelContext_103)->geometry_0 = geometry_43;
    (&kernelContext_103)->shots_0 = shots_43;
    (&kernelContext_103)->crush_0 = crush_43;
    (&kernelContext_103)->crush_rw_0 = crush_rw_43;
    (&kernelContext_103)->crush_tab_0 = crush_tab_43;
    threadgroup array<uint, int(1024)> g_scan_43;
    (&kernelContext_103)->g_scan_0 = &g_scan_43;
    uint blocks_2 = (params_43->chunk_count_0 + 1023U) / 1024U;
    uint _S888 = scan_new_blocks_0(&kernelContext_103);
    uint device* _S889 = (&kernelContext_103)->work_0+(_S888 + blocks_2);
    uint total_3 = *_S889;
    uint _S890 = a_counters_0(&kernelContext_103);
    atomic_store_explicit((&kernelContext_103)->atom_0+(_S890 + 7U), *_S889, memory_order_relaxed);
    uint _S891 = atomic_fetch_add_explicit((&kernelContext_103)->atom_0+(_S890 + 9U), total_3, memory_order_relaxed);
    if(((&kernelContext_103)->crush_0->mode_1) == 1U)
    {
        uint _S892 = atomic_fetch_add_explicit((&kernelContext_103)->atom_0+(_S890 + 8U), total_3 * (&kernelContext_103)->crush_0->fragments_0, memory_order_relaxed);
    }
    return;
}

