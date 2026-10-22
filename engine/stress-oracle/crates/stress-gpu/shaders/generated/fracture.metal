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
    uint pulse_cap_0;
    uint impact_peak_0;
    uint pad0_0;
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

struct KernelContext_0
{
    FractureParams_0 constant* params_0;
    uint device* work_0;
    packed_float4 device* frames_0;
    packed_float4 device* bodies_0;
    packed_float4 device* chunks_0;
    Record_natural_0 device* manifolds_0;
    uint device* sort_0;
    packed_float4 device* loads_0;
    atomic<uint> device* atom_0;
    packed_uint4 device* bonds_0;
    uint device* args_0;
    uint device* info_0;
    ScanParams_0 constant* scan_0;
    packed_float4 device* parts_0;
    packed_float4 device* colliders_0;
    packed_uint4 device* collider_info_0;
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

uint l_pulse_base_0(KernelContext_0 thread* kernelContext_2)
{
    return kernelContext_2->params_0->chunk_count_0;
}

uint l_pulse_count_base_0(KernelContext_0 thread* kernelContext_3)
{
    return kernelContext_3->params_0->chunk_count_0 + 3U * kernelContext_3->params_0->pulse_cap_0 * kernelContext_3->params_0->chunk_count_0;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S18 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S18, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S19 = max(0.0f, _S18);
    float _S20 = min(dt_1, _S18 + duration_0);
    bool _S21;
    if(_S20 > _S19)
    {
        _S21 = duration_0 > 0.0f;
    }
    else
    {
        _S21 = false;
    }
    float sum_1;
    if(_S21)
    {
        float w_1 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_1 * (cos(w_1 * (_S19 - _S18)) - cos(w_1 * (_S20 - _S18)));
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

uint a_sums_0(KernelContext_0 thread* kernelContext_4)
{
    return 3U * kernelContext_4->params_0->chunk_count_0;
}

uint a_dirty_0(KernelContext_0 thread* kernelContext_5)
{
    uint _S22 = a_sums_0(kernelContext_5);
    return _S22 + 20U * kernelContext_5->params_0->chunk_count_0;
}

uint a_best_count_0(KernelContext_0 thread* kernelContext_6)
{
    uint _S23 = a_dirty_0(kernelContext_6);
    return _S23 + kernelContext_6->params_0->body_count_0;
}

uint a_best_root_0(KernelContext_0 thread* kernelContext_7)
{
    uint _S24 = a_best_count_0(kernelContext_7);
    return _S24 + kernelContext_7->params_0->body_count_0;
}

uint a_counters_0(KernelContext_0 thread* kernelContext_8)
{
    uint _S25 = a_best_root_0(kernelContext_8);
    return _S25 + kernelContext_8->params_0->body_count_0;
}

[[kernel]] void fr_loads(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], uint device* sort_1 [[buffer(11)]], packed_float4 device* loads_1 [[buffer(12)]], atomic<uint> device* atom_1 [[buffer(5)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]])
{
    float3 f_total_0;
    float m_1;
    bool overflow_0;
    bool overflow_1;
    uint p_0;
    uint kept_0;
    uint q3_0;
    float3 f_total_1;
    float3 t_total_0;
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->params_0 = params_1;
    (&kernelContext_9)->work_0 = work_1;
    (&kernelContext_9)->frames_0 = frames_1;
    (&kernelContext_9)->bodies_0 = bodies_1;
    (&kernelContext_9)->chunks_0 = chunks_1;
    (&kernelContext_9)->manifolds_0 = manifolds_1;
    (&kernelContext_9)->sort_0 = sort_1;
    (&kernelContext_9)->loads_0 = loads_1;
    (&kernelContext_9)->atom_0 = atom_1;
    (&kernelContext_9)->bonds_0 = bonds_1;
    (&kernelContext_9)->args_0 = args_1;
    (&kernelContext_9)->info_0 = info_1;
    (&kernelContext_9)->scan_0 = scan_1;
    (&kernelContext_9)->parts_0 = parts_1;
    (&kernelContext_9)->colliders_0 = colliders_1;
    (&kernelContext_9)->collider_info_0 = collider_info_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_9)->g_scan_0 = &g_scan_1;
    uint k_0 = id_0.x;
    if(k_0 >= (params_1->chunk_count_0))
    {
        return;
    }
    uint _S26 = body_of_0(k_0, &kernelContext_9);
    uint b_1 = 7U * _S26;
    thread Quat_0 q_6;
    thread float3 origin_1;
    structure_pose_0(_S26, b_1, &q_6, &origin_1, &kernelContext_9);
    uint _S27 = 4U * k_0;
    float4 _S28 = float4(*((&kernelContext_9)->chunks_0+_S27)) ;
    float3 _S29 = _S28.xyz;
    thread Quat_0 _S30 = q_6;
    float3 _S31 = rotate_0(&_S30, _S29);
    float3 centre_0 = origin_1 + _S31;
    float inv_mass_0 = (float4(*((&kernelContext_9)->bodies_0+b_1)) ).w;
    float m_body_0;
    if(inv_mass_0 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_0;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint _S32 = params_1->chunk_count_0;
    float _S33 = _S28.w;
    float4 _S34 = float4(*((&kernelContext_9)->chunks_0+(_S27 + 3U))) ;
    float _S35 = (&kernelContext_9)->params_0->band_0;
    float _S36 = (&kernelContext_9)->params_0->dt_0;
    bool _S37 = ((&kernelContext_9)->params_0->impact_peak_0) != 0U;
    uint _S38 = l_pulse_base_0(&kernelContext_9);
    uint _S39 = l_pulse_count_base_0(&kernelContext_9);
    uint _S40 = (&kernelContext_9)->params_0->pulse_cap_0;
    for(;;)
    {
        float3 _S41 = float3(0.0f) ;
        uint _S42 = (&kernelContext_9)->sort_0[k_0];
        uint _S43 = (&kernelContext_9)->sort_0[_S32 + 1U + k_0];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_1 = 0U;
        f_total_0 = _S41;
        float3 t_total_1 = _S41;
        for(;;)
        {
            if(step_1 < _S43)
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
                if(q3_0 < _S43)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_9)->manifolds_0[_S42 + q3_0].ids_0) ).y + 1U;
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
            Record_natural_0 r_4 = (&kernelContext_9)->manifolds_0[_S42 + p_0];
            float4 _S44 = float4(r_4.impulse_0) ;
            float3 _S45 = _S44.xyz;
            float j_0 = length(_S45);
            if(j_0 == 0.0f)
            {
                uint _S46 = step_1 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S46;
                continue;
            }
            float m_other_0 = (float4(r_4.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S33;
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
            float _S47 = min(j_0, 2.0f * m_1 * max(_S44.w - _S35, 0.0f));
            if(_S47 < j_0)
            {
                float3 f_0 = _S45 / float3(j_0)  * float3(((j_0 - _S47) / _S36)) ;
                float3 _S48 = t_total_1 + cross((float4(r_4.point_0) ).xyz - centre_0, f_0);
                f_total_1 = f_total_0 + f_0;
                t_total_0 = _S48;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            uint _S46 = step_1 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S46;
        }
        uint _S49 = _S38 + 3U * _S40 * k_0;
        uint _S50 = _S39 + k_0;
        uint _S51 = (as_type<uint>(((float4(*((&kernelContext_9)->loads_0+_S50)) ).x)));
        p_0 = 0U;
        kept_0 = 0U;
        for(;;)
        {
            if(p_0 < _S51)
            {
            }
            else
            {
                break;
            }
            uint _S52 = _S49 + 3U * p_0;
            float4 _S53 = float4(*((&kernelContext_9)->loads_0+_S52)) ;
            thread float4 a_1 = _S53;
            float4 _S54 = float4(*((&kernelContext_9)->loads_0+(_S52 + 1U))) ;
            float4 _S55 = float4(*((&kernelContext_9)->loads_0+(_S52 + 2U))) ;
            float _S56 = _S54.w;
            float mean_0 = pulse_mean_0(_S53.x, _S53.y, _S53.z, _S53.w, _S56, _S36);
            if(mean_0 != 0.0f)
            {
                float3 f_1 = _S55.xyz * float3(mean_0) ;
                float3 _S57 = t_total_1 + cross(_S54.xyz - centre_0, f_1);
                f_total_0 = f_total_0 + f_1;
                t_total_1 = _S57;
            }
            if((a_1.x + a_1.z + _S56) > _S36)
            {
                a_1.x = a_1.x - _S36;
                uint _S58 = _S49 + 3U * kept_0;
                *((&kernelContext_9)->loads_0+_S58) = packed_float4(a_1) ;
                *((&kernelContext_9)->loads_0+(_S58 + 1U)) = packed_float4(_S54) ;
                *((&kernelContext_9)->loads_0+(_S58 + 2U)) = packed_float4(_S55) ;
                kept_0 = kept_0 + 1U;
            }
            p_0 = p_0 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S43)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_5 = (&kernelContext_9)->manifolds_0[_S42 + q3_0];
            float4 _S59 = float4(r_5.impulse_0) ;
            float3 _S60 = _S59.xyz;
            float j_1 = length(_S60);
            if(j_1 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S61 = float4(r_5.other_0) ;
            float m_other_1 = _S61.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S33;
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
            float _S62 = _S59.w;
            float _S63 = min(j_1, 2.0f * m_1 * max(_S62 - _S35, 0.0f));
            if(_S63 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S64 = _S34.y;
            float duration_1 = hertz_duration_0(m_1, min(_S61.z, _S34.z), 1.0f / ((1.0f - _S64 * _S64) / _S34.x + 1.0f / _S61.y), _S62);
            float peak_1 = 3.14159274101257324f * _S63 / (2.0f * duration_1);
            float3 dir_0 = _S60 / float3(j_1) ;
            float mean_1;
            if(_S37)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S36);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_2 = dir_0 * float3(mean_1) ;
                float3 _S65 = t_total_1 + cross((float4(r_5.point_0) ).xyz - centre_0, f_2);
                f_total_1 = f_total_0 + f_2;
                t_total_0 = _S65;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_0 = t_total_1;
            }
            uint kept_1;
            if(duration_1 > _S36)
            {
                if(kept_0 < _S40)
                {
                    uint _S66 = _S49 + 3U * kept_0;
                    *((&kernelContext_9)->loads_0+_S66) = packed_float4(float4(- _S36, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_9)->loads_0+(_S66 + 1U)) = packed_float4(float4((float4(r_5.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_9)->loads_0+(_S66 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
            kept_0 = kept_1;
            f_total_0 = f_total_1;
            t_total_1 = t_total_0;
            q3_0 = q3_0 + 1U;
            overflow_0 = overflow_1;
        }
        *((&kernelContext_9)->loads_0+_S50) = packed_float4(float4((as_type<float>((kept_0))), 0.0f, 0.0f, 0.0f)) ;
        break;
    }
    if(overflow_0)
    {
        uint _S67 = a_counters_0(&kernelContext_9);
        uint _S68 = atomic_fetch_or_explicit((&kernelContext_9)->atom_0+(_S67 + 3U), 8U, memory_order_relaxed);
    }
    *((&kernelContext_9)->loads_0+k_0) = packed_float4(float4(length(f_total_0), 0.0f, 0.0f, 0.0f)) ;
    return;
}

uint w_alive_0(KernelContext_0 thread* kernelContext_10)
{
    return kernelContext_10->params_0->chunk_count_0;
}

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], uint device* sort_2 [[buffer(11)]], packed_float4 device* loads_2 [[buffer(12)]], atomic<uint> device* atom_2 [[buffer(5)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_11;
    (&kernelContext_11)->params_0 = params_2;
    (&kernelContext_11)->work_0 = work_2;
    (&kernelContext_11)->frames_0 = frames_2;
    (&kernelContext_11)->bodies_0 = bodies_2;
    (&kernelContext_11)->chunks_0 = chunks_2;
    (&kernelContext_11)->manifolds_0 = manifolds_2;
    (&kernelContext_11)->sort_0 = sort_2;
    (&kernelContext_11)->loads_0 = loads_2;
    (&kernelContext_11)->atom_0 = atom_2;
    (&kernelContext_11)->bonds_0 = bonds_2;
    (&kernelContext_11)->args_0 = args_2;
    (&kernelContext_11)->info_0 = info_2;
    (&kernelContext_11)->scan_0 = scan_2;
    (&kernelContext_11)->parts_0 = parts_2;
    (&kernelContext_11)->colliders_0 = colliders_2;
    (&kernelContext_11)->collider_info_0 = collider_info_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_11)->g_scan_0 = &g_scan_2;
    uint i_0 = id_1.x;
    bool _S69;
    if(i_0 >= (params_2->bond_count_0))
    {
        _S69 = true;
    }
    else
    {
        uint _S70 = w_alive_0(&kernelContext_11);
        _S69 = (*((&kernelContext_11)->work_0+(_S70 + i_0))) == 0U;
    }
    if(_S69)
    {
        return;
    }
    uint4 _S71 = uint4(*((&kernelContext_11)->bonds_0+i_0)) ;
    uint _S72 = _S71.x;
    if((max((float4(*((&kernelContext_11)->loads_0+_S72)) ).x, (float4(*((&kernelContext_11)->loads_0+_S71.y)) ).x)) > (as_type<float>((_S71.z))))
    {
        uint _S73 = w_alive_0(&kernelContext_11);
        *((&kernelContext_11)->work_0+(_S73 + i_0)) = 0U;
        uint _S74 = a_dirty_0(&kernelContext_11);
        uint _S75 = body_of_0(_S72, &kernelContext_11);
        uint _S76 = atomic_fetch_or_explicit((&kernelContext_11)->atom_0+(_S74 + _S75), 1U, memory_order_relaxed);
        uint _S77 = a_counters_0(&kernelContext_11);
        uint _S78 = atomic_fetch_add_explicit((&kernelContext_11)->atom_0+_S77, 1U, memory_order_relaxed);
        uint _S79 = atomic_fetch_add_explicit((&kernelContext_11)->atom_0+(_S77 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint groups_of_0(uint n_2, uint size_0)
{
    uint _S80 = (n_2 + size_0 - 1U) / size_0;
    return _S80;
}

void set_args_0(uint at_0, uint x_1, KernelContext_0 thread* kernelContext_12)
{
    uint _S81 = 3U * at_0;
    *(kernelContext_12->args_0+_S81) = x_1;
    *(kernelContext_12->args_0+(_S81 + 1U)) = 1U;
    *(kernelContext_12->args_0+(_S81 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], uint device* sort_3 [[buffer(11)]], packed_float4 device* loads_3 [[buffer(12)]], atomic<uint> device* atom_3 [[buffer(5)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_13;
    (&kernelContext_13)->params_0 = params_3;
    (&kernelContext_13)->work_0 = work_3;
    (&kernelContext_13)->frames_0 = frames_3;
    (&kernelContext_13)->bodies_0 = bodies_3;
    (&kernelContext_13)->chunks_0 = chunks_3;
    (&kernelContext_13)->manifolds_0 = manifolds_3;
    (&kernelContext_13)->sort_0 = sort_3;
    (&kernelContext_13)->loads_0 = loads_3;
    (&kernelContext_13)->atom_0 = atom_3;
    (&kernelContext_13)->bonds_0 = bonds_3;
    (&kernelContext_13)->args_0 = args_3;
    (&kernelContext_13)->info_0 = info_3;
    (&kernelContext_13)->scan_0 = scan_3;
    (&kernelContext_13)->parts_0 = parts_3;
    (&kernelContext_13)->colliders_0 = colliders_3;
    (&kernelContext_13)->collider_info_0 = collider_info_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_13)->g_scan_0 = &g_scan_3;
    uint _S82 = a_counters_0(&kernelContext_13);
    uint _S83 = atomic_load_explicit((&kernelContext_13)->atom_0+_S82, memory_order_relaxed);
    bool run_0 = _S83 != 0U;
    atomic_store_explicit((&kernelContext_13)->atom_0+_S82, 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S84 = atomic_fetch_add_explicit((&kernelContext_13)->atom_0+(_S82 + 4U), 1U, memory_order_relaxed);
    }
    uint _S85;
    if(run_0)
    {
        uint _S86 = groups_of_0((&kernelContext_13)->params_0->chunk_count_0, 64U);
        _S85 = _S86;
    }
    else
    {
        _S85 = 0U;
    }
    set_args_0(0U, _S85, &kernelContext_13);
    if(run_0)
    {
        uint _S87 = groups_of_0((&kernelContext_13)->params_0->bond_count_0, 64U);
        _S85 = _S87;
    }
    else
    {
        _S85 = 0U;
    }
    set_args_0(1U, _S85, &kernelContext_13);
    if(run_0)
    {
        uint _S88 = groups_of_0((&kernelContext_13)->params_0->body_count_0, 64U);
        _S85 = _S88;
    }
    else
    {
        _S85 = 0U;
    }
    set_args_0(2U, _S85, &kernelContext_13);
    if(run_0)
    {
        uint _S89 = groups_of_0((&kernelContext_13)->params_0->chunk_count_0, 1024U);
        _S85 = _S89;
    }
    else
    {
        _S85 = 0U;
    }
    set_args_0(3U, _S85, &kernelContext_13);
    if(run_0)
    {
        uint _S90 = groups_of_0((&kernelContext_13)->params_0->body_count_0, 1024U);
        _S85 = _S90;
    }
    else
    {
        _S85 = 0U;
    }
    set_args_0(4U, _S85, &kernelContext_13);
    int _S91;
    if(run_0)
    {
        _S91 = int(1);
    }
    else
    {
        _S91 = int(0);
    }
    set_args_0(5U, uint(_S91), &kernelContext_13);
    return;
}

bool dirty_0(uint body_0, KernelContext_0 thread* kernelContext_14)
{
    uint _S92 = a_dirty_0(kernelContext_14);
    uint _S93 = atomic_load_explicit(kernelContext_14->atom_0+(_S92 + body_0), memory_order_relaxed);
    return _S93 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_15)
{
    return kernelContext_15->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_16)
{
    return 2U * kernelContext_16->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], uint device* sort_4 [[buffer(11)]], packed_float4 device* loads_4 [[buffer(12)]], atomic<uint> device* atom_4 [[buffer(5)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_17;
    (&kernelContext_17)->params_0 = params_4;
    (&kernelContext_17)->work_0 = work_4;
    (&kernelContext_17)->frames_0 = frames_4;
    (&kernelContext_17)->bodies_0 = bodies_4;
    (&kernelContext_17)->chunks_0 = chunks_4;
    (&kernelContext_17)->manifolds_0 = manifolds_4;
    (&kernelContext_17)->sort_0 = sort_4;
    (&kernelContext_17)->loads_0 = loads_4;
    (&kernelContext_17)->atom_0 = atom_4;
    (&kernelContext_17)->bonds_0 = bonds_4;
    (&kernelContext_17)->args_0 = args_4;
    (&kernelContext_17)->info_0 = info_4;
    (&kernelContext_17)->scan_0 = scan_4;
    (&kernelContext_17)->parts_0 = parts_4;
    (&kernelContext_17)->colliders_0 = colliders_4;
    (&kernelContext_17)->collider_info_0 = collider_info_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_17)->g_scan_0 = &g_scan_4;
    uint c_2 = id_3.x;
    bool _S94;
    if(c_2 >= (params_4->chunk_count_0))
    {
        _S94 = true;
    }
    else
    {
        uint _S95 = body_of_0(c_2, &kernelContext_17);
        bool _S96 = dirty_0(_S95, &kernelContext_17);
        _S94 = !_S96;
    }
    if(_S94)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_17)->atom_0+(a_parent_0() + c_2), c_2, memory_order_relaxed);
    uint _S97 = a_count_0(&kernelContext_17);
    atomic_store_explicit((&kernelContext_17)->atom_0+(_S97 + c_2), 0U, memory_order_relaxed);
    uint _S98 = a_anchor_0(&kernelContext_17);
    atomic_store_explicit((&kernelContext_17)->atom_0+(_S98 + c_2), 0U, memory_order_relaxed);
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
        uint _S99 = a_sums_0(&kernelContext_17);
        atomic_store_explicit((&kernelContext_17)->atom_0+(_S99 + 20U * c_2 + w_2), 0U, memory_order_relaxed);
        w_2 = w_2 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], uint device* sort_5 [[buffer(11)]], packed_float4 device* loads_5 [[buffer(12)]], atomic<uint> device* atom_5 [[buffer(5)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_18;
    (&kernelContext_18)->params_0 = params_5;
    (&kernelContext_18)->work_0 = work_5;
    (&kernelContext_18)->frames_0 = frames_5;
    (&kernelContext_18)->bodies_0 = bodies_5;
    (&kernelContext_18)->chunks_0 = chunks_5;
    (&kernelContext_18)->manifolds_0 = manifolds_5;
    (&kernelContext_18)->sort_0 = sort_5;
    (&kernelContext_18)->loads_0 = loads_5;
    (&kernelContext_18)->atom_0 = atom_5;
    (&kernelContext_18)->bonds_0 = bonds_5;
    (&kernelContext_18)->args_0 = args_5;
    (&kernelContext_18)->info_0 = info_5;
    (&kernelContext_18)->scan_0 = scan_5;
    (&kernelContext_18)->parts_0 = parts_5;
    (&kernelContext_18)->colliders_0 = colliders_5;
    (&kernelContext_18)->collider_info_0 = collider_info_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_18)->g_scan_0 = &g_scan_5;
    uint s_0 = id_4.x;
    bool _S100;
    if(s_0 >= (params_5->body_count_0))
    {
        _S100 = true;
    }
    else
    {
        bool _S101 = dirty_0(s_0, &kernelContext_18);
        _S100 = !_S101;
    }
    if(_S100)
    {
        return;
    }
    uint _S102 = a_best_count_0(&kernelContext_18);
    atomic_store_explicit((&kernelContext_18)->atom_0+(_S102 + s_0), 0U, memory_order_relaxed);
    uint _S103 = a_best_root_0(&kernelContext_18);
    atomic_store_explicit((&kernelContext_18)->atom_0+(_S103 + s_0), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_2, KernelContext_0 thread* kernelContext_19)
{
    uint _S104 = a_parent_0();
    uint _S105 = atomic_load_explicit(kernelContext_19->atom_0+(_S104 + x_2), memory_order_relaxed);
    uint p_1 = _S105;
    uint _S106 = x_2;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S107;
        if(guard_0 < 1048576U)
        {
            _S107 = p_1 != _S106;
        }
        else
        {
            _S107 = false;
        }
        if(_S107)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_19->atom_0+(_S104 + p_1), memory_order_relaxed);
        if(gp_0 != p_1)
        {
            uint _S108;
            {
            uint _metal_cas_comparand = p_1;
            atomic_compare_exchange_weak_explicit(kernelContext_19->atom_0+(_S104 + _S106), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S108 = _metal_cas_comparand;
            }
        }
        uint _S109 = guard_0 + 1U;
        _S106 = p_1;
        p_1 = gp_0;
        guard_0 = _S109;
    }
    return _S106;
}

[[kernel]] void fr_uf_union(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], uint device* sort_6 [[buffer(11)]], packed_float4 device* loads_6 [[buffer(12)]], atomic<uint> device* atom_6 [[buffer(5)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_20;
    (&kernelContext_20)->params_0 = params_6;
    (&kernelContext_20)->work_0 = work_6;
    (&kernelContext_20)->frames_0 = frames_6;
    (&kernelContext_20)->bodies_0 = bodies_6;
    (&kernelContext_20)->chunks_0 = chunks_6;
    (&kernelContext_20)->manifolds_0 = manifolds_6;
    (&kernelContext_20)->sort_0 = sort_6;
    (&kernelContext_20)->loads_0 = loads_6;
    (&kernelContext_20)->atom_0 = atom_6;
    (&kernelContext_20)->bonds_0 = bonds_6;
    (&kernelContext_20)->args_0 = args_6;
    (&kernelContext_20)->info_0 = info_6;
    (&kernelContext_20)->scan_0 = scan_6;
    (&kernelContext_20)->parts_0 = parts_6;
    (&kernelContext_20)->colliders_0 = colliders_6;
    (&kernelContext_20)->collider_info_0 = collider_info_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_20)->g_scan_0 = &g_scan_6;
    uint i_1 = id_5.x;
    bool _S110;
    if(i_1 >= (params_6->bond_count_0))
    {
        _S110 = true;
    }
    else
    {
        uint _S111 = w_alive_0(&kernelContext_20);
        _S110 = (*((&kernelContext_20)->work_0+(_S111 + i_1))) == 0U;
    }
    if(_S110)
    {
        return;
    }
    uint4 _S112 = uint4(*((&kernelContext_20)->bonds_0+i_1)) ;
    uint _S113 = _S112.x;
    uint _S114 = body_of_0(_S113, &kernelContext_20);
    bool _S115 = dirty_0(_S114, &kernelContext_20);
    if(!_S115)
    {
        return;
    }
    uint _S116 = _S112.y;
    uint a_2 = _S113;
    uint b_2 = _S116;
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
        uint _S117 = find_0(a_2, &kernelContext_20);
        uint _S118 = find_0(b_2, &kernelContext_20);
        if(_S117 == _S118)
        {
            return;
        }
        uint _S119 = max(_S117, _S118);
        uint _S120 = min(_S117, _S118);
        uint was_0;
        {
        uint _metal_cas_comparand = _S119;
        atomic_compare_exchange_weak_explicit((&kernelContext_20)->atom_0+(a_parent_0() + _S119), &_metal_cas_comparand, _S120, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S119)
        {
            return;
        }
        uint _S121 = guard_1 + 1U;
        a_2 = _S119;
        b_2 = _S120;
        guard_1 = _S121;
    }
    uint _S122 = a_counters_0(&kernelContext_20);
    uint _S123 = atomic_fetch_or_explicit((&kernelContext_20)->atom_0+(_S122 + 3U), 4U, memory_order_relaxed);
    return;
}

uint w_label_0(KernelContext_0 thread* kernelContext_21)
{
    uint _S124 = w_alive_0(kernelContext_21);
    return _S124 + kernelContext_21->params_0->bond_count_0;
}

void add64_0(uint at_1, float value_0, float scale_0, KernelContext_0 thread* kernelContext_22)
{
    float x_3 = round(value_0 * scale_0);
    float x_4;
    if(!((abs(x_3)) < 1.073741824e+09f))
    {
        uint _S125 = a_counters_0(kernelContext_22);
        uint _S126 = atomic_fetch_or_explicit(kernelContext_22->atom_0+(_S125 + 3U), 2U, memory_order_relaxed);
        x_4 = 0.0f;
    }
    else
    {
        x_4 = x_3;
    }
    int v_3 = int(x_4);
    uint lo_0 = (as_type<uint>((v_3)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_22->atom_0+at_1, lo_0, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_0) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S127;
    if(v_3 < int(0))
    {
        _S127 = 4294967295U;
    }
    else
    {
        _S127 = 0U;
    }
    uint hi_0 = _S127 + carry_0;
    if(hi_0 != 0U)
    {
        uint _S128 = atomic_fetch_add_explicit(kernelContext_22->atom_0+(at_1 + 1U), hi_0, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], uint device* sort_7 [[buffer(11)]], packed_float4 device* loads_7 [[buffer(12)]], atomic<uint> device* atom_7 [[buffer(5)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_23;
    (&kernelContext_23)->params_0 = params_7;
    (&kernelContext_23)->work_0 = work_7;
    (&kernelContext_23)->frames_0 = frames_7;
    (&kernelContext_23)->bodies_0 = bodies_7;
    (&kernelContext_23)->chunks_0 = chunks_7;
    (&kernelContext_23)->manifolds_0 = manifolds_7;
    (&kernelContext_23)->sort_0 = sort_7;
    (&kernelContext_23)->loads_0 = loads_7;
    (&kernelContext_23)->atom_0 = atom_7;
    (&kernelContext_23)->bonds_0 = bonds_7;
    (&kernelContext_23)->args_0 = args_7;
    (&kernelContext_23)->info_0 = info_7;
    (&kernelContext_23)->scan_0 = scan_7;
    (&kernelContext_23)->parts_0 = parts_7;
    (&kernelContext_23)->colliders_0 = colliders_7;
    (&kernelContext_23)->collider_info_0 = collider_info_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_23)->g_scan_0 = &g_scan_7;
    uint c_3 = id_6.x;
    bool _S129;
    if(c_3 >= (params_7->chunk_count_0))
    {
        _S129 = true;
    }
    else
    {
        uint _S130 = body_of_0(c_3, &kernelContext_23);
        bool _S131 = dirty_0(_S130, &kernelContext_23);
        _S129 = !_S131;
    }
    if(_S129)
    {
        return;
    }
    uint _S132 = find_0(c_3, &kernelContext_23);
    uint _S133 = w_label_0(&kernelContext_23);
    *((&kernelContext_23)->work_0+(_S133 + c_3)) = _S132;
    uint _S134 = a_count_0(&kernelContext_23);
    uint _S135 = atomic_fetch_add_explicit((&kernelContext_23)->atom_0+(_S134 + _S132), 1U, memory_order_relaxed);
    if(((&kernelContext_23)->info_0[c_3] & 1U) != 0U)
    {
        uint _S136 = a_anchor_0(&kernelContext_23);
        uint _S137 = atomic_fetch_or_explicit((&kernelContext_23)->atom_0+(_S136 + _S132), 1U, memory_order_relaxed);
    }
    uint _S138 = 4U * c_3;
    float4 _S139 = float4(*((&kernelContext_23)->chunks_0+_S138)) ;
    float4 _S140 = float4(*((&kernelContext_23)->chunks_0+(_S138 + 1U))) ;
    float4 _S141 = float4(*((&kernelContext_23)->chunks_0+(_S138 + 2U))) ;
    float m_2 = _S139.w;
    float3 d_0 = _S139.xyz - (float4(*((&kernelContext_23)->chunks_0+4U * _S132)) ).xyz;
    float dd_0 = dot(d_0, d_0);
    uint _S142 = a_sums_0(&kernelContext_23);
    uint at_2 = _S142 + 20U * _S132;
    add64_0(at_2, m_2, (&kernelContext_23)->params_0->scale_m_0, &kernelContext_23);
    float _S143 = d_0.x;
    float _S144 = m_2 * _S143;
    add64_0(at_2 + 2U, _S144, (&kernelContext_23)->params_0->scale_s_0, &kernelContext_23);
    float _S145 = d_0.y;
    float _S146 = m_2 * _S145;
    add64_0(at_2 + 4U, _S146, (&kernelContext_23)->params_0->scale_s_0, &kernelContext_23);
    float _S147 = d_0.z;
    add64_0(at_2 + 6U, m_2 * _S147, (&kernelContext_23)->params_0->scale_s_0, &kernelContext_23);
    add64_0(at_2 + 8U, _S140.x + m_2 * (dd_0 - _S143 * _S143), (&kernelContext_23)->params_0->scale_j_0, &kernelContext_23);
    add64_0(at_2 + 10U, _S140.y + m_2 * (dd_0 - _S145 * _S145), (&kernelContext_23)->params_0->scale_j_0, &kernelContext_23);
    add64_0(at_2 + 12U, _S140.z + m_2 * (dd_0 - _S147 * _S147), (&kernelContext_23)->params_0->scale_j_0, &kernelContext_23);
    add64_0(at_2 + 14U, _S140.w - _S144 * _S145, (&kernelContext_23)->params_0->scale_j_0, &kernelContext_23);
    add64_0(at_2 + 16U, _S141.x - _S144 * _S147, (&kernelContext_23)->params_0->scale_j_0, &kernelContext_23);
    add64_0(at_2 + 18U, _S141.y - _S146 * _S147, (&kernelContext_23)->params_0->scale_j_0, &kernelContext_23);
    return;
}

bool is_root_0(uint c_4, KernelContext_0 thread* kernelContext_24)
{
    bool _S148;
    if(c_4 < (kernelContext_24->params_0->chunk_count_0))
    {
        uint _S149 = body_of_0(c_4, kernelContext_24);
        bool _S150 = dirty_0(_S149, kernelContext_24);
        _S148 = _S150;
    }
    else
    {
        _S148 = false;
    }
    if(_S148)
    {
        uint _S151 = w_label_0(kernelContext_24);
        _S148 = (*(kernelContext_24->work_0+(_S151 + c_4))) == c_4;
    }
    else
    {
        _S148 = false;
    }
    return _S148;
}

[[kernel]] void fr_best_count(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], uint device* sort_8 [[buffer(11)]], packed_float4 device* loads_8 [[buffer(12)]], atomic<uint> device* atom_8 [[buffer(5)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_25;
    (&kernelContext_25)->params_0 = params_8;
    (&kernelContext_25)->work_0 = work_8;
    (&kernelContext_25)->frames_0 = frames_8;
    (&kernelContext_25)->bodies_0 = bodies_8;
    (&kernelContext_25)->chunks_0 = chunks_8;
    (&kernelContext_25)->manifolds_0 = manifolds_8;
    (&kernelContext_25)->sort_0 = sort_8;
    (&kernelContext_25)->loads_0 = loads_8;
    (&kernelContext_25)->atom_0 = atom_8;
    (&kernelContext_25)->bonds_0 = bonds_8;
    (&kernelContext_25)->args_0 = args_8;
    (&kernelContext_25)->info_0 = info_8;
    (&kernelContext_25)->scan_0 = scan_8;
    (&kernelContext_25)->parts_0 = parts_8;
    (&kernelContext_25)->colliders_0 = colliders_8;
    (&kernelContext_25)->collider_info_0 = collider_info_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_25)->g_scan_0 = &g_scan_8;
    uint c_5 = id_7.x;
    bool _S152 = is_root_0(c_5, &kernelContext_25);
    if(!_S152)
    {
        return;
    }
    uint _S153 = a_best_count_0(&kernelContext_25);
    uint _S154 = body_of_0(c_5, &kernelContext_25);
    atomic<uint> device* _S155 = (&kernelContext_25)->atom_0+(_S153 + _S154);
    uint _S156 = a_count_0(&kernelContext_25);
    uint _S157 = atomic_load_explicit((&kernelContext_25)->atom_0+(_S156 + c_5), memory_order_relaxed);
    uint _S158 = atomic_fetch_max_explicit(_S155, _S157, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], uint device* sort_9 [[buffer(11)]], packed_float4 device* loads_9 [[buffer(12)]], atomic<uint> device* atom_9 [[buffer(5)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_26;
    (&kernelContext_26)->params_0 = params_9;
    (&kernelContext_26)->work_0 = work_9;
    (&kernelContext_26)->frames_0 = frames_9;
    (&kernelContext_26)->bodies_0 = bodies_9;
    (&kernelContext_26)->chunks_0 = chunks_9;
    (&kernelContext_26)->manifolds_0 = manifolds_9;
    (&kernelContext_26)->sort_0 = sort_9;
    (&kernelContext_26)->loads_0 = loads_9;
    (&kernelContext_26)->atom_0 = atom_9;
    (&kernelContext_26)->bonds_0 = bonds_9;
    (&kernelContext_26)->args_0 = args_9;
    (&kernelContext_26)->info_0 = info_9;
    (&kernelContext_26)->scan_0 = scan_9;
    (&kernelContext_26)->parts_0 = parts_9;
    (&kernelContext_26)->colliders_0 = colliders_9;
    (&kernelContext_26)->collider_info_0 = collider_info_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_26)->g_scan_0 = &g_scan_9;
    uint c_6 = id_8.x;
    bool _S159 = is_root_0(c_6, &kernelContext_26);
    if(!_S159)
    {
        return;
    }
    uint _S160 = body_of_0(c_6, &kernelContext_26);
    uint _S161 = a_count_0(&kernelContext_26);
    uint _S162 = atomic_load_explicit((&kernelContext_26)->atom_0+(_S161 + c_6), memory_order_relaxed);
    uint _S163 = a_best_count_0(&kernelContext_26);
    uint _S164 = atomic_load_explicit((&kernelContext_26)->atom_0+(_S163 + _S160), memory_order_relaxed);
    if(_S162 == _S164)
    {
        uint _S165 = a_best_root_0(&kernelContext_26);
        uint _S166 = atomic_fetch_min_explicit((&kernelContext_26)->atom_0+(_S165 + _S160), c_6, memory_order_relaxed);
    }
    return;
}

uint w_slot_0(KernelContext_0 thread* kernelContext_27)
{
    uint _S167 = w_label_0(kernelContext_27);
    return _S167 + kernelContext_27->params_0->chunk_count_0;
}

uint w_new_0(KernelContext_0 thread* kernelContext_28)
{
    uint _S168 = w_slot_0(kernelContext_28);
    return _S168 + kernelContext_28->params_0->chunk_count_0;
}

[[kernel]] void fr_mark(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], uint device* sort_10 [[buffer(11)]], packed_float4 device* loads_10 [[buffer(12)]], atomic<uint> device* atom_10 [[buffer(5)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_29;
    (&kernelContext_29)->params_0 = params_10;
    (&kernelContext_29)->work_0 = work_10;
    (&kernelContext_29)->frames_0 = frames_10;
    (&kernelContext_29)->bodies_0 = bodies_10;
    (&kernelContext_29)->chunks_0 = chunks_10;
    (&kernelContext_29)->manifolds_0 = manifolds_10;
    (&kernelContext_29)->sort_0 = sort_10;
    (&kernelContext_29)->loads_0 = loads_10;
    (&kernelContext_29)->atom_0 = atom_10;
    (&kernelContext_29)->bonds_0 = bonds_10;
    (&kernelContext_29)->args_0 = args_10;
    (&kernelContext_29)->info_0 = info_10;
    (&kernelContext_29)->scan_0 = scan_10;
    (&kernelContext_29)->parts_0 = parts_10;
    (&kernelContext_29)->colliders_0 = colliders_10;
    (&kernelContext_29)->collider_info_0 = collider_info_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_29)->g_scan_0 = &g_scan_10;
    uint c_7 = id_9.x;
    if(c_7 >= (params_10->chunk_count_0))
    {
        return;
    }
    uint _S169 = w_new_0(&kernelContext_29);
    uint device* _S170 = (&kernelContext_29)->work_0+(_S169 + c_7);
    bool _S171 = is_root_0(c_7, &kernelContext_29);
    bool _S172;
    if(_S171)
    {
        uint _S173 = a_best_root_0(&kernelContext_29);
        uint _S174 = body_of_0(c_7, &kernelContext_29);
        uint _S175 = atomic_load_explicit((&kernelContext_29)->atom_0+(_S173 + _S174), memory_order_relaxed);
        _S172 = _S175 != c_7;
    }
    else
    {
        _S172 = false;
    }
    uint _S176;
    if(_S172)
    {
        _S176 = 1U;
    }
    else
    {
        _S176 = 0U;
    }
    *_S170 = _S176;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], uint device* sort_11 [[buffer(11)]], packed_float4 device* loads_11 [[buffer(12)]], atomic<uint> device* atom_11 [[buffer(5)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]])
{
    uint step_2;
    uint add_0;
    thread KernelContext_0 kernelContext_30;
    (&kernelContext_30)->params_0 = params_11;
    (&kernelContext_30)->work_0 = work_11;
    (&kernelContext_30)->frames_0 = frames_11;
    (&kernelContext_30)->bodies_0 = bodies_11;
    (&kernelContext_30)->chunks_0 = chunks_11;
    (&kernelContext_30)->manifolds_0 = manifolds_11;
    (&kernelContext_30)->sort_0 = sort_11;
    (&kernelContext_30)->loads_0 = loads_11;
    (&kernelContext_30)->atom_0 = atom_11;
    (&kernelContext_30)->bonds_0 = bonds_11;
    (&kernelContext_30)->args_0 = args_11;
    (&kernelContext_30)->info_0 = info_11;
    (&kernelContext_30)->scan_0 = scan_11;
    (&kernelContext_30)->parts_0 = parts_11;
    (&kernelContext_30)->colliders_0 = colliders_11;
    (&kernelContext_30)->collider_info_0 = collider_info_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_30)->g_scan_0 = &g_scan_11;
    uint t_1 = thread_0.x;
    uint _S177 = group_0.x;
    uint _S178 = 1024U * _S177 + 4U * t_1;
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
        if((_S178 + j_2) < ((&kernelContext_30)->scan_0->n_0))
        {
            step_2 = *((&kernelContext_30)->work_0+((&kernelContext_30)->scan_0->in_base_0 + _S178 + j_2));
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
    (*(&kernelContext_30)->g_scan_0)[t_1] = sum_2;
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
            add_0 = (*(&kernelContext_30)->g_scan_0)[t_1 - step_2];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_30)->g_scan_0)[t_1] = (*(&kernelContext_30)->g_scan_0)[t_1] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_2 = step_2 << 1U;
    }
    uint _S179 = (*(&kernelContext_30)->g_scan_0)[t_1] - sum_2;
    j_2 = 0U;
    add_0 = _S179;
    for(;;)
    {
        if(j_2 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S178 + j_2) < ((&kernelContext_30)->scan_0->n_0))
        {
            *((&kernelContext_30)->work_0+((&kernelContext_30)->scan_0->out_base_0 + _S178 + j_2)) = add_0;
        }
        uint run_1 = add_0 + v_4[j_2];
        j_2 = j_2 + 1U;
        add_0 = run_1;
    }
    if(t_1 == 255U)
    {
        *((&kernelContext_30)->work_0+((&kernelContext_30)->scan_0->block_base_0 + _S177)) = (*(&kernelContext_30)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], uint device* sort_12 [[buffer(11)]], packed_float4 device* loads_12 [[buffer(12)]], atomic<uint> device* atom_12 [[buffer(5)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_31;
    (&kernelContext_31)->params_0 = params_12;
    (&kernelContext_31)->work_0 = work_12;
    (&kernelContext_31)->frames_0 = frames_12;
    (&kernelContext_31)->bodies_0 = bodies_12;
    (&kernelContext_31)->chunks_0 = chunks_12;
    (&kernelContext_31)->manifolds_0 = manifolds_12;
    (&kernelContext_31)->sort_0 = sort_12;
    (&kernelContext_31)->loads_0 = loads_12;
    (&kernelContext_31)->atom_0 = atom_12;
    (&kernelContext_31)->bonds_0 = bonds_12;
    (&kernelContext_31)->args_0 = args_12;
    (&kernelContext_31)->info_0 = info_12;
    (&kernelContext_31)->scan_0 = scan_12;
    (&kernelContext_31)->parts_0 = parts_12;
    (&kernelContext_31)->colliders_0 = colliders_12;
    (&kernelContext_31)->collider_info_0 = collider_info_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_31)->g_scan_0 = &g_scan_12;
    uint t_2 = thread_1.x;
    uint blocks_0 = (scan_12->n_0 + 1023U) / 1024U;
    bool _S180 = t_2 < blocks_0;
    uint v_5;
    if(_S180)
    {
        v_5 = *((&kernelContext_31)->work_0+((&kernelContext_31)->scan_0->block_base_0 + t_2));
    }
    else
    {
        v_5 = 0U;
    }
    (*(&kernelContext_31)->g_scan_0)[t_2] = v_5;
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
            add_1 = (*(&kernelContext_31)->g_scan_0)[t_2 - step_3];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_31)->g_scan_0)[t_2] = (*(&kernelContext_31)->g_scan_0)[t_2] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_3 = step_3 << 1U;
    }
    if(_S180)
    {
        *((&kernelContext_31)->work_0+((&kernelContext_31)->scan_0->block_base_0 + t_2)) = (*(&kernelContext_31)->g_scan_0)[t_2] - v_5;
    }
    if(t_2 == 1023U)
    {
        *((&kernelContext_31)->work_0+((&kernelContext_31)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_31)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], uint device* sort_13 [[buffer(11)]], packed_float4 device* loads_13 [[buffer(12)]], atomic<uint> device* atom_13 [[buffer(5)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_32;
    (&kernelContext_32)->params_0 = params_13;
    (&kernelContext_32)->work_0 = work_13;
    (&kernelContext_32)->frames_0 = frames_13;
    (&kernelContext_32)->bodies_0 = bodies_13;
    (&kernelContext_32)->chunks_0 = chunks_13;
    (&kernelContext_32)->manifolds_0 = manifolds_13;
    (&kernelContext_32)->sort_0 = sort_13;
    (&kernelContext_32)->loads_0 = loads_13;
    (&kernelContext_32)->atom_0 = atom_13;
    (&kernelContext_32)->bonds_0 = bonds_13;
    (&kernelContext_32)->args_0 = args_13;
    (&kernelContext_32)->info_0 = info_13;
    (&kernelContext_32)->scan_0 = scan_13;
    (&kernelContext_32)->parts_0 = parts_13;
    (&kernelContext_32)->colliders_0 = colliders_13;
    (&kernelContext_32)->collider_info_0 = collider_info_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_32)->g_scan_0 = &g_scan_13;
    uint i_2 = id_10.x;
    if(i_2 >= (scan_13->n_0))
    {
        return;
    }
    uint device* _S181 = (&kernelContext_32)->work_0+((&kernelContext_32)->scan_0->out_base_0 + i_2);
    *_S181 = *_S181 + *((&kernelContext_32)->work_0+((&kernelContext_32)->scan_0->block_base_0 + i_2 / 1024U));
    return;
}

uint w_rank_0(KernelContext_0 thread* kernelContext_33)
{
    uint _S182 = w_new_0(kernelContext_33);
    return _S182 + kernelContext_33->params_0->chunk_count_0;
}

uint w_free_0(KernelContext_0 thread* kernelContext_34)
{
    uint _S183 = w_rank_0(kernelContext_34);
    return _S183 + kernelContext_34->params_0->chunk_count_0;
}

uint w_free_rank_0(KernelContext_0 thread* kernelContext_35)
{
    uint _S184 = w_free_0(kernelContext_35);
    return _S184 + kernelContext_35->params_0->body_count_0;
}

uint w_free_list_0(KernelContext_0 thread* kernelContext_36)
{
    uint _S185 = w_free_rank_0(kernelContext_36);
    return _S185 + kernelContext_36->params_0->body_count_0;
}

[[kernel]] void fr_freelist(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], uint device* sort_14 [[buffer(11)]], packed_float4 device* loads_14 [[buffer(12)]], atomic<uint> device* atom_14 [[buffer(5)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_37;
    (&kernelContext_37)->params_0 = params_14;
    (&kernelContext_37)->work_0 = work_14;
    (&kernelContext_37)->frames_0 = frames_14;
    (&kernelContext_37)->bodies_0 = bodies_14;
    (&kernelContext_37)->chunks_0 = chunks_14;
    (&kernelContext_37)->manifolds_0 = manifolds_14;
    (&kernelContext_37)->sort_0 = sort_14;
    (&kernelContext_37)->loads_0 = loads_14;
    (&kernelContext_37)->atom_0 = atom_14;
    (&kernelContext_37)->bonds_0 = bonds_14;
    (&kernelContext_37)->args_0 = args_14;
    (&kernelContext_37)->info_0 = info_14;
    (&kernelContext_37)->scan_0 = scan_14;
    (&kernelContext_37)->parts_0 = parts_14;
    (&kernelContext_37)->colliders_0 = colliders_14;
    (&kernelContext_37)->collider_info_0 = collider_info_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_37)->g_scan_0 = &g_scan_14;
    uint s_1 = id_11.x;
    bool _S186;
    if(s_1 >= (params_14->body_count_0))
    {
        _S186 = true;
    }
    else
    {
        uint _S187 = w_free_0(&kernelContext_37);
        _S186 = (*((&kernelContext_37)->work_0+(_S187 + s_1))) == 0U;
    }
    if(_S186)
    {
        return;
    }
    uint _S188 = w_free_list_0(&kernelContext_37);
    uint _S189 = w_free_rank_0(&kernelContext_37);
    *((&kernelContext_37)->work_0+(_S188 + *((&kernelContext_37)->work_0+(_S189 + s_1)))) = s_1;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_38)
{
    uint _S190 = w_free_list_0(kernelContext_38);
    return _S190 + kernelContext_38->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_39)
{
    uint _S191 = scan_new_blocks_0(kernelContext_39);
    return _S191 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_40)
{
    uint blocks_1 = (kernelContext_40->params_0->body_count_0 + 1023U) / 1024U;
    uint _S192 = scan_free_blocks_0(kernelContext_40);
    return *(kernelContext_40->work_0+(_S192 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], uint device* sort_15 [[buffer(11)]], packed_float4 device* loads_15 [[buffer(12)]], atomic<uint> device* atom_15 [[buffer(5)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_41;
    (&kernelContext_41)->params_0 = params_15;
    (&kernelContext_41)->work_0 = work_15;
    (&kernelContext_41)->frames_0 = frames_15;
    (&kernelContext_41)->bodies_0 = bodies_15;
    (&kernelContext_41)->chunks_0 = chunks_15;
    (&kernelContext_41)->manifolds_0 = manifolds_15;
    (&kernelContext_41)->sort_0 = sort_15;
    (&kernelContext_41)->loads_0 = loads_15;
    (&kernelContext_41)->atom_0 = atom_15;
    (&kernelContext_41)->bonds_0 = bonds_15;
    (&kernelContext_41)->args_0 = args_15;
    (&kernelContext_41)->info_0 = info_15;
    (&kernelContext_41)->scan_0 = scan_15;
    (&kernelContext_41)->parts_0 = parts_15;
    (&kernelContext_41)->colliders_0 = colliders_15;
    (&kernelContext_41)->collider_info_0 = collider_info_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_41)->g_scan_0 = &g_scan_15;
    uint c_8 = id_12.x;
    bool _S193 = is_root_0(c_8, &kernelContext_41);
    if(!_S193)
    {
        return;
    }
    uint _S194 = body_of_0(c_8, &kernelContext_41);
    uint _S195 = a_best_root_0(&kernelContext_41);
    uint _S196 = atomic_load_explicit((&kernelContext_41)->atom_0+(_S195 + _S194), memory_order_relaxed);
    uint slot_1;
    if(_S196 != c_8)
    {
        uint _S197 = w_rank_0(&kernelContext_41);
        uint device* _S198 = (&kernelContext_41)->work_0+(_S197 + c_8);
        uint k_1 = *_S198;
        uint _S199 = free_total_0(&kernelContext_41);
        if((*_S198) < _S199)
        {
            uint _S200 = w_free_list_0(&kernelContext_41);
            uint device* _S201 = (&kernelContext_41)->work_0+(_S200 + k_1);
            uint slot_2 = *_S201;
            uint _S202 = w_free_0(&kernelContext_41);
            *((&kernelContext_41)->work_0+(_S202 + *_S201)) = 0U;
            uint _S203 = a_counters_0(&kernelContext_41);
            uint _S204 = atomic_fetch_add_explicit((&kernelContext_41)->atom_0+(_S203 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S205 = a_counters_0(&kernelContext_41);
            uint _S206 = atomic_fetch_or_explicit((&kernelContext_41)->atom_0+(_S205 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S194;
        }
    }
    else
    {
        slot_1 = _S194;
    }
    uint _S207 = w_slot_0(&kernelContext_41);
    *((&kernelContext_41)->work_0+(_S207 + c_8)) = slot_1;
    return;
}

float read64_0(uint at_3, float scale_1, KernelContext_0 thread* kernelContext_42)
{
    uint _S208 = atomic_load_explicit(kernelContext_42->atom_0+(at_3 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S208)));
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
    array<float, int(3)> _S209 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S210 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S211 = { { r2_0.x, r2_0.y, r2_0.z } };
    thread array<array<float, int(3)>, int(3)> a_3;
    a_3[int(0)] = _S209;
    a_3[int(1)] = _S210;
    a_3[int(2)] = _S211;
    array<float, int(3)> _S212 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S213 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S214 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_6;
    v_6[int(0)] = _S212;
    v_6[int(1)] = _S213;
    v_6[int(2)] = _S214;
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
            bool _S215;
            if(k_2 != p_2)
            {
                _S215 = k_2 != q_9;
            }
            else
            {
                _S215 = false;
            }
            if(_S215)
            {
                float kp_0 = a_3[k_2][p_2];
                float kq_0 = a_3[k_2][q_9];
                float _S216 = c_9 * a_3[k_2][p_2] - s_2 * a_3[k_2][q_9];
                a_3[k_2][p_2] = _S216;
                a_3[p_2][k_2] = _S216;
                float _S217 = s_2 * kp_0 + c_9 * kq_0;
                a_3[k_2][q_9] = _S217;
                a_3[q_9][k_2] = _S217;
            }
            k_2 = k_2 + 1U;
        }
        float _S218 = c_9 * c_9;
        float _S219 = 2.0f * s_2 * c_9 * apq_0;
        float _S220 = s_2 * s_2;
        a_3[p_2][p_2] = _S218 * app_0 - _S219 + _S220 * aqq_0;
        a_3[q_9][q_9] = _S220 * app_0 + _S219 + _S218 * aqq_0;
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
    thread Quat_0 _S221 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S222 = normalized_0(&_S221);
    *rot_0 = _S222;
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
    thread Quat_0 _S223 = axes_1;
    Quat_0 _S224 = quat_mul_0(&_S223, q_10);
    thread Quat_0 _S225 = _S224;
    Quat_0 _S226 = normalized_0(&_S225);
    return _S226;
}

uint stage_0(uint c_10, KernelContext_0 thread* kernelContext_43)
{
    return 2U * kernelContext_43->params_0->body_count_0 + 9U * c_10;
}

float4 quat_vec_0(const Quat_0 thread* q_11)
{
    return float4(q_11->x_0, q_11->y_0, q_11->z_0, q_11->w_0);
}

[[kernel]] void fr_piece(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], uint device* sort_16 [[buffer(11)]], packed_float4 device* loads_16 [[buffer(12)]], atomic<uint> device* atom_16 [[buffer(5)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_44;
    (&kernelContext_44)->params_0 = params_16;
    (&kernelContext_44)->work_0 = work_16;
    (&kernelContext_44)->frames_0 = frames_16;
    (&kernelContext_44)->bodies_0 = bodies_16;
    (&kernelContext_44)->chunks_0 = chunks_16;
    (&kernelContext_44)->manifolds_0 = manifolds_16;
    (&kernelContext_44)->sort_0 = sort_16;
    (&kernelContext_44)->loads_0 = loads_16;
    (&kernelContext_44)->atom_0 = atom_16;
    (&kernelContext_44)->bonds_0 = bonds_16;
    (&kernelContext_44)->args_0 = args_16;
    (&kernelContext_44)->info_0 = info_16;
    (&kernelContext_44)->scan_0 = scan_16;
    (&kernelContext_44)->parts_0 = parts_16;
    (&kernelContext_44)->colliders_0 = colliders_16;
    (&kernelContext_44)->collider_info_0 = collider_info_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_44)->g_scan_0 = &g_scan_16;
    uint c_11 = id_13.x;
    bool _S227 = is_root_0(c_11, &kernelContext_44);
    if(!_S227)
    {
        return;
    }
    uint _S228 = body_of_0(c_11, &kernelContext_44);
    uint pb_0 = 7U * _S228;
    uint _S229 = a_sums_0(&kernelContext_44);
    uint at_4 = _S229 + 20U * c_11;
    float _S230 = read64_0(at_4, (&kernelContext_44)->params_0->scale_m_0, &kernelContext_44);
    float _S231 = read64_0(at_4 + 2U, (&kernelContext_44)->params_0->scale_s_0, &kernelContext_44);
    float _S232 = read64_0(at_4 + 4U, (&kernelContext_44)->params_0->scale_s_0, &kernelContext_44);
    float _S233 = read64_0(at_4 + 6U, (&kernelContext_44)->params_0->scale_s_0, &kernelContext_44);
    float3 d_1 = float3(_S231, _S232, _S233) / float3(max(_S230, 1.00000000317107685e-30f)) ;
    float _S234 = read64_0(at_4 + 8U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float _S235 = read64_0(at_4 + 10U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float _S236 = read64_0(at_4 + 12U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float _S237 = read64_0(at_4 + 14U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float _S238 = read64_0(at_4 + 16U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float _S239 = read64_0(at_4 + 18U, (&kernelContext_44)->params_0->scale_j_0, &kernelContext_44);
    float dd_1 = dot(d_1, d_1);
    float _S240 = d_1.x;
    float _S241 = _S230 * _S240;
    float _S242 = d_1.y;
    float _S243 = _S237 + _S241 * _S242;
    float _S244 = d_1.z;
    float _S245 = _S238 + _S241 * _S244;
    float _S246 = _S239 + _S230 * _S242 * _S244;
    float3 com_1 = (float4(*((&kernelContext_44)->chunks_0+4U * c_11)) ).xyz + d_1;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S234 - _S230 * (dd_1 - _S240 * _S240), _S243, _S245), float3(_S243, _S235 - _S230 * (dd_1 - _S242 * _S242), _S246), float3(_S245, _S246, _S236 - _S230 * (dd_1 - _S244 * _S244)), &principal_0, &moments_1);
    uint _S247 = a_anchor_0(&kernelContext_44);
    uint _S248 = atomic_load_explicit((&kernelContext_44)->atom_0+(_S247 + c_11), memory_order_relaxed);
    bool anchored_0 = _S248 != 0U;
    thread Quat_0 q_12;
    thread float3 origin_2;
    structure_pose_0(_S228, pb_0, &q_12, &origin_2, &kernelContext_44);
    float3 p_parent_0 = (float4(*((&kernelContext_44)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_44)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_44)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S249 = q_12;
    float3 _S250 = rotate_0(&_S249, com_1);
    float3 pos_0 = to_rubble_0(origin_2 + _S250);
    thread Quat_0 _S251 = q_12;
    thread Quat_0 _S252 = principal_0;
    Quat_0 _S253 = quat_mul_0(&_S251, &_S252);
    thread Quat_0 _S254 = _S253;
    Quat_0 _S255 = normalized_0(&_S254);
    thread Quat_0 _S256 = _S255;
    Quat_0 _S257 = rot_to_rubble_0(&_S256);
    bool _S258;
    if(anchored_0)
    {
        _S258 = true;
    }
    else
    {
        _S258 = !(_S230 > 0.0f);
    }
    float inv_mass_1;
    if(_S258)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / _S230;
    }
    float3 _S259 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S260;
        if((moments_1.x) > 0.0f)
        {
            _S260 = 1.0f / moments_1.x;
        }
        else
        {
            _S260 = 0.0f;
        }
        float _S261;
        if((moments_1.y) > 0.0f)
        {
            _S261 = 1.0f / moments_1.y;
        }
        else
        {
            _S261 = 0.0f;
        }
        float _S262;
        if((moments_1.z) > 0.0f)
        {
            _S262 = 1.0f / moments_1.z;
        }
        else
        {
            _S262 = 0.0f;
        }
        inv_inertia_0 = float3(_S260, _S261, _S262);
    }
    else
    {
        inv_inertia_0 = _S259;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S259;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S259;
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
    uint _S263 = stage_0(c_11, &kernelContext_44);
    *((&kernelContext_44)->frames_0+_S263) = packed_float4(float4(pos_0, inv_mass_1)) ;
    packed_float4 device* _S264 = (&kernelContext_44)->frames_0+(_S263 + 1U);
    thread Quat_0 _S265 = _S257;
    float4 _S266 = quat_vec_0(&_S265);
    *_S264 = packed_float4(_S266) ;
    *((&kernelContext_44)->frames_0+(_S263 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_44)->frames_0+(_S263 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_44)->frames_0+(_S263 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_44)->params_0->step_0))))) ;
    *((&kernelContext_44)->frames_0+(_S263 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_44)->frames_0+(_S263 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_44)->frames_0+(_S263 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_44)->frames_0+2U * _S228)) ).w)) ;
    packed_float4 device* _S267 = (&kernelContext_44)->frames_0+(_S263 + 8U);
    thread Quat_0 _S268 = principal_0;
    float4 _S269 = quat_vec_0(&_S268);
    *_S267 = packed_float4(_S269) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], uint device* sort_17 [[buffer(11)]], packed_float4 device* loads_17 [[buffer(12)]], atomic<uint> device* atom_17 [[buffer(5)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_45;
    (&kernelContext_45)->params_0 = params_17;
    (&kernelContext_45)->work_0 = work_17;
    (&kernelContext_45)->frames_0 = frames_17;
    (&kernelContext_45)->bodies_0 = bodies_17;
    (&kernelContext_45)->chunks_0 = chunks_17;
    (&kernelContext_45)->manifolds_0 = manifolds_17;
    (&kernelContext_45)->sort_0 = sort_17;
    (&kernelContext_45)->loads_0 = loads_17;
    (&kernelContext_45)->atom_0 = atom_17;
    (&kernelContext_45)->bonds_0 = bonds_17;
    (&kernelContext_45)->args_0 = args_17;
    (&kernelContext_45)->info_0 = info_17;
    (&kernelContext_45)->scan_0 = scan_17;
    (&kernelContext_45)->parts_0 = parts_17;
    (&kernelContext_45)->colliders_0 = colliders_17;
    (&kernelContext_45)->collider_info_0 = collider_info_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_45)->g_scan_0 = &g_scan_17;
    uint c_12 = id_14.x;
    bool _S270 = is_root_0(c_12, &kernelContext_45);
    if(!_S270)
    {
        return;
    }
    uint _S271 = w_slot_0(&kernelContext_45);
    uint slot_3 = *((&kernelContext_45)->work_0+(_S271 + c_12));
    uint _S272 = stage_0(c_12, &kernelContext_45);
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
            *((&kernelContext_45)->bodies_0+(7U * (slot_3 + region_0 * (&kernelContext_45)->params_0->max_bodies_0) + w_3)) = packed_float4(float4(*((&kernelContext_45)->frames_0+(_S272 + w_3))) ) ;
            w_3 = w_3 + 1U;
        }
        region_0 = region_0 + 1U;
    }
    uint _S273 = 2U * slot_3;
    *((&kernelContext_45)->frames_0+_S273) = packed_float4(float4(*((&kernelContext_45)->frames_0+(_S272 + 7U))) ) ;
    *((&kernelContext_45)->frames_0+(_S273 + 1U)) = packed_float4(float4(*((&kernelContext_45)->frames_0+(_S272 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], uint device* sort_18 [[buffer(11)]], packed_float4 device* loads_18 [[buffer(12)]], atomic<uint> device* atom_18 [[buffer(5)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_46;
    (&kernelContext_46)->params_0 = params_18;
    (&kernelContext_46)->work_0 = work_18;
    (&kernelContext_46)->frames_0 = frames_18;
    (&kernelContext_46)->bodies_0 = bodies_18;
    (&kernelContext_46)->chunks_0 = chunks_18;
    (&kernelContext_46)->manifolds_0 = manifolds_18;
    (&kernelContext_46)->sort_0 = sort_18;
    (&kernelContext_46)->loads_0 = loads_18;
    (&kernelContext_46)->atom_0 = atom_18;
    (&kernelContext_46)->bonds_0 = bonds_18;
    (&kernelContext_46)->args_0 = args_18;
    (&kernelContext_46)->info_0 = info_18;
    (&kernelContext_46)->scan_0 = scan_18;
    (&kernelContext_46)->parts_0 = parts_18;
    (&kernelContext_46)->colliders_0 = colliders_18;
    (&kernelContext_46)->collider_info_0 = collider_info_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_46)->g_scan_0 = &g_scan_18;
    uint c_13 = id_15.x;
    bool _S274;
    if(c_13 >= (params_18->chunk_count_0))
    {
        _S274 = true;
    }
    else
    {
        uint _S275 = body_of_0(c_13, &kernelContext_46);
        bool _S276 = dirty_0(_S275, &kernelContext_46);
        _S274 = !_S276;
    }
    if(_S274)
    {
        return;
    }
    uint _S277 = w_label_0(&kernelContext_46);
    uint device* _S278 = (&kernelContext_46)->work_0+(_S277 + c_13);
    uint r_6 = *_S278;
    uint _S279 = w_slot_0(&kernelContext_46);
    uint device* _S280 = (&kernelContext_46)->work_0+(_S279 + *_S278);
    uint slot_4 = *_S280;
    *((&kernelContext_46)->work_0+(w_body_0() + c_13)) = *_S280;
    uint _S281 = 2U * slot_4;
    float3 _S282 = (float4(*((&kernelContext_46)->frames_0+_S281)) ).xyz;
    thread Quat_0 _S283 = quat_of_0(float4(*((&kernelContext_46)->frames_0+(_S281 + 1U))) );
    Quat_0 _S284 = conj_0(&_S283);
    float inv_mass_2 = (float4(*((&kernelContext_46)->bodies_0+7U * slot_4)) ).w;
    float _S285;
    if(inv_mass_2 > 0.0f)
    {
        _S285 = 1.0f / inv_mass_2;
    }
    else
    {
        _S285 = float_bits_0(2139095040U);
    }
    float4 _S286 = float4(*((&kernelContext_46)->chunks_0+(4U * c_13 + 2U))) ;
    uint first_1 = (as_type<uint>((_S286.z)));
    uint _S287 = (as_type<uint>((_S286.w)));
    float4 _S288 = float4(*((&kernelContext_46)->chunks_0+(4U * r_6 + 3U))) ;
    uint p_3 = first_1;
    for(;;)
    {
        if(p_3 < (first_1 + _S287))
        {
        }
        else
        {
            break;
        }
        uint _S289 = 2U * p_3;
        float4 _S290 = float4(*((&kernelContext_46)->parts_0+_S289)) ;
        uint col_0 = (as_type<uint>((_S290.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_46)->parts_0+(_S289 + 1U))) );
        uint _S291 = 4U * col_0;
        float4 _S292 = float4(*((&kernelContext_46)->colliders_0+_S291)) ;
        *((&kernelContext_46)->colliders_0+_S291) = packed_float4(float4((as_type<float>((slot_4))), _S292.y, _S292.z, _S292.w)) ;
        uint _S293 = _S291 + 1U;
        packed_float4 device* _S294 = (&kernelContext_46)->colliders_0+_S293;
        float3 _S295 = _S290.xyz - _S282;
        thread Quat_0 _S296 = _S284;
        float3 _S297 = rotate_0(&_S296, _S295);
        *_S294 = packed_float4(float4(_S297, (float4(*((&kernelContext_46)->colliders_0+_S293)) ).w)) ;
        packed_float4 device* _S298 = (&kernelContext_46)->colliders_0+(_S291 + 2U);
        thread Quat_0 _S299 = _S284;
        thread Quat_0 _S300 = rot_1;
        Quat_0 _S301 = quat_mul_0(&_S299, &_S300);
        thread Quat_0 _S302 = _S301;
        Quat_0 _S303 = normalized_0(&_S302);
        thread Quat_0 _S304 = _S303;
        float4 _S305 = quat_vec_0(&_S304);
        *_S298 = packed_float4(_S305) ;
        uint _S306 = 2U * col_0;
        uint4 _S307 = uint4(*((&kernelContext_46)->collider_info_0+_S306)) ;
        *((&kernelContext_46)->collider_info_0+_S306) = packed_uint4(uint4(_S307.x, slot_4, _S307.z, _S307.w)) ;
        *((&kernelContext_46)->collider_info_0+(_S306 + 1U)) = packed_uint4(uint4((as_type<uint>((_S285))), (as_type<uint>((_S288.x))), (as_type<uint>((_S288.w))), 0U)) ;
        p_3 = p_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], uint device* sort_19 [[buffer(11)]], packed_float4 device* loads_19 [[buffer(12)]], atomic<uint> device* atom_19 [[buffer(5)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_47;
    (&kernelContext_47)->params_0 = params_19;
    (&kernelContext_47)->work_0 = work_19;
    (&kernelContext_47)->frames_0 = frames_19;
    (&kernelContext_47)->bodies_0 = bodies_19;
    (&kernelContext_47)->chunks_0 = chunks_19;
    (&kernelContext_47)->manifolds_0 = manifolds_19;
    (&kernelContext_47)->sort_0 = sort_19;
    (&kernelContext_47)->loads_0 = loads_19;
    (&kernelContext_47)->atom_0 = atom_19;
    (&kernelContext_47)->bonds_0 = bonds_19;
    (&kernelContext_47)->args_0 = args_19;
    (&kernelContext_47)->info_0 = info_19;
    (&kernelContext_47)->scan_0 = scan_19;
    (&kernelContext_47)->parts_0 = parts_19;
    (&kernelContext_47)->colliders_0 = colliders_19;
    (&kernelContext_47)->collider_info_0 = collider_info_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_47)->g_scan_0 = &g_scan_19;
    uint s_3 = id_16.x;
    if(s_3 >= (params_19->body_count_0))
    {
        return;
    }
    uint _S308 = a_dirty_0(&kernelContext_47);
    atomic_store_explicit((&kernelContext_47)->atom_0+(_S308 + s_3), 0U, memory_order_relaxed);
    return;
}

