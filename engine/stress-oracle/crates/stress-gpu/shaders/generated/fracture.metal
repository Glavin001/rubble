#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
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
    float lp_scale_0;
    uint cs_sweeps_0;
    uint cs_row_cap_0;
    float cs_c1_0;
    float cs_c2_0;
    uint cs_row_offset_0;
    float pad7_0;
    float pad8_0;
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
    packed_float4 device* lp_sides_0;
    uint device* engine_counters_0;
    packed_float4 device* engine_contacts_0;
    packed_float4 device* cs_anchors_0;
    array<uint, int(1024)> threadgroup* g_scan_0;
    array<float, int(64)> threadgroup* g_il_best_0;
};

uint lp_body_0(uint b_0, KernelContext_0 thread* kernelContext_0)
{
    return 22U * kernelContext_0->params_0->chunk_count_0 + 7U * b_0;
}

float il_t_in_0(uint b_1, KernelContext_0 thread* kernelContext_1)
{
    uint _S1 = lp_body_0(b_1, kernelContext_1);
    return (float4(*(kernelContext_1->lp_0+(_S1 + 5U))) ).w;
}

uint a_sums_0(KernelContext_0 thread* kernelContext_2)
{
    return 3U * kernelContext_2->params_0->chunk_count_0;
}

uint a_dirty_0(KernelContext_0 thread* kernelContext_3)
{
    uint _S2 = a_sums_0(kernelContext_3);
    return _S2 + 20U * kernelContext_3->params_0->chunk_count_0;
}

uint a_best_count_0(KernelContext_0 thread* kernelContext_4)
{
    uint _S3 = a_dirty_0(kernelContext_4);
    return _S3 + kernelContext_4->params_0->body_count_0;
}

uint a_best_root_0(KernelContext_0 thread* kernelContext_5)
{
    uint _S4 = a_best_count_0(kernelContext_5);
    return _S4 + kernelContext_5->params_0->body_count_0;
}

uint a_counters_0(KernelContext_0 thread* kernelContext_6)
{
    uint _S5 = a_best_root_0(kernelContext_6);
    return _S5 + kernelContext_6->params_0->body_count_0;
}

uint a_strike_0(KernelContext_0 thread* kernelContext_7)
{
    uint _S6 = a_counters_0(kernelContext_7);
    return _S6 + 16U;
}

float il_t_in_1(uint b_2, KernelContext_0 thread* kernelContext_8)
{
    uint _S7 = a_strike_0(kernelContext_8);
    uint _S8 = atomic_load_explicit(kernelContext_8->atom_0+(_S7 + 9U * b_2 + 6U), memory_order_relaxed);
    return (as_type<float>((_S8)));
}

uint w_body_0()
{
    return 0U;
}

uint body_of_0(uint c_0, KernelContext_0 thread* kernelContext_9)
{
    return *(kernelContext_9->work_0+(w_body_0() + c_0));
}

uint a_first_0(uint shot_0, KernelContext_0 thread* kernelContext_10)
{
    uint _S9 = a_strike_0(kernelContext_10);
    return _S9 + 29U * kernelContext_10->params_0->body_count_0 + shot_0;
}

uint a_il_cursor_0(KernelContext_0 thread* kernelContext_11)
{
    uint _S10 = a_first_0(0U, kernelContext_11);
    return _S10 + 64U;
}

uint a_cs_0(uint b_3, KernelContext_0 thread* kernelContext_12)
{
    uint _S11 = a_il_cursor_0(kernelContext_12);
    return _S11 + 1U + 12U * b_3;
}

bool cs_struck_0(uint b_4, KernelContext_0 thread* kernelContext_13)
{
    bool _S12;
    if(b_4 < (kernelContext_13->params_0->body_count_0))
    {
        uint _S13 = a_cs_0(b_4, kernelContext_13);
        uint _S14 = atomic_load_explicit(kernelContext_13->atom_0+(_S13 + 1U), memory_order_relaxed);
        _S12 = _S14 != 0U;
    }
    else
    {
        _S12 = false;
    }
    return _S12;
}

bool il_struck_0(uint b_5, KernelContext_0 thread* kernelContext_14)
{
    bool _S15 = cs_struck_0(b_5, kernelContext_14);
    return _S15;
}

void il_path_0(uint b_6, float3 thread* contact_0, float3 thread* d_0, float thread* speed_0, KernelContext_0 thread* kernelContext_15)
{
    uint _S16 = lp_body_0(b_6, kernelContext_15);
    *contact_0 = (float4(*(kernelContext_15->lp_0+(_S16 + 2U))) ).xyz;
    float4 _S17 = float4(*(kernelContext_15->lp_0+(_S16 + 3U))) ;
    *d_0 = _S17.xyz;
    *speed_0 = _S17.w;
    return;
}

struct IlStriker_0
{
    uint body_0;
    float bound_0;
    float radius_0;
    float mass_0;
    float m_eff_0;
    float modulus_0;
    uint launch_0;
};

IlStriker_0 il_striker_0(uint b_7, KernelContext_0 thread* kernelContext_16)
{
    uint _S18 = lp_body_0(b_7, kernelContext_16);
    float4 _S19 = float4(*(kernelContext_16->lp_0+(_S18 + 4U))) ;
    float4 _S20 = float4(*(kernelContext_16->lp_0+(_S18 + 5U))) ;
    thread IlStriker_0 s_0;
    (&s_0)->body_0 = (as_type<uint>((_S19.w)));
    float _S21 = _S19.y;
    (&s_0)->bound_0 = 1.00999999046325684f * _S21 + 0.00100000004749745f;
    (&s_0)->radius_0 = _S21;
    (&s_0)->mass_0 = _S20.z;
    (&s_0)->m_eff_0 = _S19.x;
    (&s_0)->modulus_0 = _S19.z;
    (&s_0)->launch_0 = (as_type<uint>(((float4(*(kernelContext_16->lp_0+(_S18 + 6U))) ).y)));
    return s_0;
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
    thread Quat_0 _S22 = axes_0;
    Quat_0 _S23 = quat_mul_0(&_S22, q_3);
    thread Quat_0 _S24 = _S23;
    Quat_0 _S25 = normalized_0(&_S24);
    return _S25;
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

void structure_pose_0(uint slot_0, uint b_8, Quat_0 thread* q_5, float3 thread* origin_0, KernelContext_0 thread* kernelContext_17)
{
    uint _S26 = 2U * slot_0;
    thread Quat_0 _S27 = quat_of_0(float4(*(kernelContext_17->frames_0+(_S26 + 1U))) );
    Quat_0 _S28 = conj_0(&_S27);
    float3 com_0 = (float4(*(kernelContext_17->frames_0+_S26)) ).xyz;
    thread Quat_0 _S29 = quat_of_0(float4(*(kernelContext_17->bodies_0+(b_8 + 1U))) );
    Quat_0 _S30 = rot_from_rubble_0(&_S29);
    thread Quat_0 _S31 = _S30;
    thread Quat_0 _S32 = _S28;
    Quat_0 _S33 = quat_mul_0(&_S31, &_S32);
    thread Quat_0 _S34 = _S33;
    Quat_0 _S35 = normalized_0(&_S34);
    *q_5 = _S35;
    float3 _S36 = from_rubble_0((float4(*(kernelContext_17->bodies_0+b_8)) ).xyz);
    thread Quat_0 _S37 = _S35;
    float3 _S38 = rotate_0(&_S37, com_0);
    *origin_0 = _S36 - _S38;
    return;
}

float3 chunk_box_world_0(uint k_0, Quat_0 thread* q_6, KernelContext_0 thread* kernelContext_18)
{
    uint _S39 = body_of_0(k_0, kernelContext_18);
    thread float3 origin_1;
    structure_pose_0(_S39, 7U * _S39, q_6, &origin_1, kernelContext_18);
    float3 _S40 = (float4(*(kernelContext_18->geometry_0+3U * k_0)) ).xyz;
    thread Quat_0 _S41 = *q_6;
    float3 _S42 = rotate_0(&_S41, _S40);
    return origin_1 + _S42;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

float2 il_interval_0(uint k_1, float3 p_0, float3 d_1, float pad_0, KernelContext_0 thread* kernelContext_19)
{
    thread Quat_0 q_7;
    float3 _S43 = chunk_box_world_0(k_1, &q_7, kernelContext_19);
    uint _S44 = 3U * k_1;
    Quat_0 _S45 = quat_of_0(float4(*(kernelContext_19->geometry_0+(_S44 + 2U))) );
    thread Quat_0 _S46 = q_7;
    thread Quat_0 _S47 = _S45;
    Quat_0 _S48 = quat_mul_0(&_S46, &_S47);
    thread Quat_0 _S49 = _S48;
    Quat_0 _S50 = normalized_0(&_S49);
    float3 _S51 = (float4(*(kernelContext_19->geometry_0+(_S44 + 1U))) ).xyz;
    float inf_0 = float_bits_0(2139095040U);
    float t0_0 = - inf_0;
    float t1_0 = inf_0;
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
        thread float3 e_0 = float3(0.0f) ;
        e_0[ax_0] = 1.0f;
        thread Quat_0 _S52 = _S50;
        float3 _S53 = rotate_0(&_S52, e_0);
        float o_1 = dot(p_0 - _S43, _S53);
        float dv_0 = dot(d_1, _S53);
        float ext_1 = _S51[ax_0] + pad_0;
        if((abs(dv_0)) < 9.99999971718068537e-10f)
        {
            if((abs(o_1)) > ext_1)
            {
                return float2(1.0f, 0.0f);
            }
            ax_0 = ax_0 + 1U;
            continue;
        }
        float ta_0 = (- ext_1 - o_1) / dv_0;
        float tb_0 = (ext_1 - o_1) / dv_0;
        float _S54 = min(t1_0, max(ta_0, tb_0));
        t0_0 = max(t0_0, min(ta_0, tb_0));
        t1_0 = _S54;
        ax_0 = ax_0 + 1U;
    }
    return float2(t0_0, t1_0);
}

uint a_il_0(uint b_9, KernelContext_0 thread* kernelContext_20)
{
    uint _S55 = a_strike_0(kernelContext_20);
    return _S55 + 23U * kernelContext_20->params_0->body_count_0 + 6U * b_9;
}

bool il_tube_0(uint k_2, uint thread* b_10, float2 thread* t_1, KernelContext_0 thread* kernelContext_21)
{
    *b_10 = 0U;
    *t_1 = float2(1.0f, 0.0f);
    if(k_2 >= (kernelContext_21->params_0->chunk_count_0))
    {
        return false;
    }
    uint _S56 = body_of_0(k_2, kernelContext_21);
    *b_10 = _S56;
    bool _S57 = il_struck_0(_S56, kernelContext_21);
    if(!_S57)
    {
        return false;
    }
    thread float3 contact_1;
    thread float3 d_2;
    thread float speed_1;
    il_path_0(*b_10, &contact_1, &d_2, &speed_1, kernelContext_21);
    IlStriker_0 _S58 = il_striker_0(*b_10, kernelContext_21);
    float2 _S59 = il_interval_0(k_2, contact_1, d_2, _S58.radius_0, kernelContext_21);
    *t_1 = _S59;
    bool _S60;
    if((_S59.x) <= (_S59.y))
    {
        float _S61 = (*t_1).y;
        uint _S62 = a_il_0(*b_10, kernelContext_21);
        uint _S63 = atomic_load_explicit(kernelContext_21->atom_0+_S62, memory_order_relaxed);
        _S60 = _S61 > (as_type<float>((_S63)));
    }
    else
    {
        _S60 = false;
    }
    return _S60;
}

uint strike_at_0(uint b_11, KernelContext_0 thread* kernelContext_22)
{
    return 2U * kernelContext_22->params_0->body_count_0 + 9U * kernelContext_22->params_0->chunk_count_0 + 3U * b_11;
}

bool struck_0(uint b_12, uint thread* shot_1, KernelContext_0 thread* kernelContext_23)
{
    uint _S64 = strike_at_0(b_12, kernelContext_23);
    uint k_3 = uint((float4(*(kernelContext_23->frames_0+_S64)) ).w);
    *shot_1 = k_3 - 1U;
    return k_3 != 0U;
}

bool il_struck_1(uint b_13, KernelContext_0 thread* kernelContext_24)
{
    thread uint shot_2;
    bool _S65;
    if((kernelContext_24->params_0->shot_count_0) > 0U)
    {
        _S65 = b_13 < (kernelContext_24->params_0->body_count_0);
    }
    else
    {
        _S65 = false;
    }
    if(_S65)
    {
        bool _S66 = struck_0(b_13, &shot_2, kernelContext_24);
        _S65 = _S66;
    }
    else
    {
        _S65 = false;
    }
    return _S65;
}

void il_path_1(uint b_14, float3 thread* contact_2, float3 thread* d_3, float thread* speed_2, KernelContext_0 thread* kernelContext_25)
{
    uint _S67 = strike_at_0(b_14, kernelContext_25);
    float3 rel_0 = (float4(*(kernelContext_25->frames_0+(_S67 + 1U))) ).xyz;
    float _S68 = length(rel_0);
    *speed_2 = _S68;
    *d_3 = rel_0 / float3(max(_S68, 9.999999960041972e-13f)) ;
    *contact_2 = (float4(*(kernelContext_25->frames_0+(_S67 + 2U))) ).xyz;
    return;
}

IlStriker_0 il_striker_1(uint b_15, KernelContext_0 thread* kernelContext_26)
{
    uint _S69 = strike_at_0(b_15, kernelContext_26);
    uint _S70 = 2U * (uint((float4(*(kernelContext_26->frames_0+_S69)) ).w) - 1U);
    float4 _S71 = float4(*(kernelContext_26->shots_0+_S70)) ;
    thread IlStriker_0 s_1;
    (&s_1)->body_0 = (as_type<uint>((_S71.x)));
    (&s_1)->bound_0 = _S71.y;
    (&s_1)->radius_0 = _S71.z;
    (&s_1)->mass_0 = _S71.w;
    float inv_mass_0 = (float4(*(kernelContext_26->bodies_0+7U * b_15)) ).w;
    float _S72;
    if(inv_mass_0 == 0.0f)
    {
        _S72 = (&s_1)->mass_0;
    }
    else
    {
        _S72 = (&s_1)->mass_0 / (1.0f + (&s_1)->mass_0 * inv_mass_0);
    }
    (&s_1)->m_eff_0 = _S72;
    uint _S73 = _S70 + 1U;
    (&s_1)->modulus_0 = (float4(*(kernelContext_26->shots_0+_S73)) ).x;
    (&s_1)->launch_0 = (as_type<uint>(((float4(*(kernelContext_26->shots_0+_S73)) ).y)));
    return s_1;
}

bool il_tube_1(uint k_4, uint thread* b_16, float2 thread* t_2, KernelContext_0 thread* kernelContext_27)
{
    *b_16 = 0U;
    *t_2 = float2(1.0f, 0.0f);
    if(k_4 >= (kernelContext_27->params_0->chunk_count_0))
    {
        return false;
    }
    uint _S74 = body_of_0(k_4, kernelContext_27);
    *b_16 = _S74;
    bool _S75 = il_struck_1(_S74, kernelContext_27);
    if(!_S75)
    {
        return false;
    }
    thread float3 contact_3;
    thread float3 d_4;
    thread float speed_3;
    il_path_1(*b_16, &contact_3, &d_4, &speed_3, kernelContext_27);
    IlStriker_0 _S76 = il_striker_1(*b_16, kernelContext_27);
    float2 _S77 = il_interval_0(k_4, contact_3, d_4, _S76.radius_0, kernelContext_27);
    *t_2 = _S77;
    bool _S78;
    if((_S77.x) <= (_S77.y))
    {
        float _S79 = (*t_2).y;
        uint _S80 = a_il_0(*b_16, kernelContext_27);
        uint _S81 = atomic_load_explicit(kernelContext_27->atom_0+_S80, memory_order_relaxed);
        _S78 = _S79 > (as_type<float>((_S81)));
    }
    else
    {
        _S78 = false;
    }
    return _S78;
}

uint il_entry_0(uint b_17, KernelContext_0 thread* kernelContext_28)
{
    uint _S82 = a_cs_0(b_17, kernelContext_28);
    uint _S83 = atomic_load_explicit(kernelContext_28->atom_0+(_S82 + 1U), memory_order_relaxed);
    return _S83 - 1U;
}

uint il_entry_1(uint b_18, KernelContext_0 thread* kernelContext_29)
{
    uint _S84 = a_strike_0(kernelContext_29);
    uint _S85 = atomic_load_explicit(kernelContext_29->atom_0+(_S84 + 9U * b_18 + 8U), memory_order_relaxed);
    return _S85;
}

uint cs_rows_0(KernelContext_0 thread* kernelContext_30)
{
    return min(kernelContext_30->engine_counters_0[int(2)], kernelContext_30->params_0->cs_row_cap_0);
}

uint cs_row_0(uint r_3, KernelContext_0 thread* kernelContext_31)
{
    return 8U * (kernelContext_31->params_0->cs_row_offset_0 + r_3);
}

uint w_alive_0(KernelContext_0 thread* kernelContext_32)
{
    return kernelContext_32->params_0->chunk_count_0;
}

uint w_label_0(KernelContext_0 thread* kernelContext_33)
{
    uint _S86 = w_alive_0(kernelContext_33);
    return _S86 + kernelContext_33->params_0->bond_count_0;
}

uint w_slot_0(KernelContext_0 thread* kernelContext_34)
{
    uint _S87 = w_label_0(kernelContext_34);
    return _S87 + kernelContext_34->params_0->chunk_count_0;
}

uint w_new_0(KernelContext_0 thread* kernelContext_35)
{
    uint _S88 = w_slot_0(kernelContext_35);
    return _S88 + kernelContext_35->params_0->chunk_count_0;
}

uint w_rank_0(KernelContext_0 thread* kernelContext_36)
{
    uint _S89 = w_new_0(kernelContext_36);
    return _S89 + kernelContext_36->params_0->chunk_count_0;
}

uint w_free_0(KernelContext_0 thread* kernelContext_37)
{
    uint _S90 = w_rank_0(kernelContext_37);
    return _S90 + kernelContext_37->params_0->chunk_count_0;
}

uint w_free_rank_0(KernelContext_0 thread* kernelContext_38)
{
    uint _S91 = w_free_0(kernelContext_38);
    return _S91 + kernelContext_38->params_0->body_count_0;
}

uint w_free_list_0(KernelContext_0 thread* kernelContext_39)
{
    uint _S92 = w_free_rank_0(kernelContext_39);
    return _S92 + kernelContext_39->params_0->body_count_0;
}

uint w_pulses_0(KernelContext_0 thread* kernelContext_40)
{
    uint _S93 = w_free_list_0(kernelContext_40);
    return _S93 + kernelContext_40->params_0->body_count_0 + 2050U;
}

uint w_marks_0(KernelContext_0 thread* kernelContext_41)
{
    uint _S94 = w_pulses_0(kernelContext_41);
    return _S94 + 4U * kernelContext_41->params_0->chunk_count_0;
}

float cs_angular_0(uint b_19, float3 r_4, float3 u_0, KernelContext_0 thread* kernelContext_42)
{
    uint at_0 = 7U * b_19;
    thread Quat_0 _S95 = quat_of_0(float4(*(kernelContext_42->bodies_0+(at_0 + 1U))) );
    Quat_0 _S96 = conj_0(&_S95);
    float3 _S97 = cross(r_4, u_0);
    thread Quat_0 _S98 = _S96;
    float3 _S99 = rotate_0(&_S98, _S97);
    return dot(_S99 * (float4(*(kernelContext_42->bodies_0+(at_0 + 2U))) ).xyz, _S99);
}

bool cs_side_0(uint r_5, uint side_0, uint thread* k_5, uint thread* s_2, uint thread* ob_0, uint thread* co_0, float3 thread* p_1, float3 thread* u_1, float thread* v_2, float thread* m_eff_1, float thread* t_in_0, KernelContext_0 thread* kernelContext_43)
{
    *t_in_0 = 0.0f;
    *k_5 = 4294967295U;
    *s_2 = 4294967295U;
    *ob_0 = 4294967295U;
    *co_0 = 4294967295U;
    float3 _S100 = float3(0.0f) ;
    *p_1 = _S100;
    *u_1 = _S100;
    *v_2 = 0.0f;
    *m_eff_1 = 0.0f;
    uint _S101 = cs_row_0(r_5, kernelContext_43);
    uint4 ids_1 = (as_type<uint4>((float4(*(kernelContext_43->engine_contacts_0+_S101)) )));
    if((ids_1.w) == 0U)
    {
        return false;
    }
    bool _S102 = side_0 == 0U;
    uint cm_0;
    if(_S102)
    {
        cm_0 = ids_1.x;
    }
    else
    {
        cm_0 = ids_1.y;
    }
    uint _S103;
    if(_S102)
    {
        _S103 = ids_1.y;
    }
    else
    {
        _S103 = ids_1.x;
    }
    *co_0 = _S103;
    uint _S104 = (uint4(*(kernelContext_43->collider_info_0+2U * cm_0)) ).x;
    *k_5 = _S104;
    bool _S105;
    if(_S104 == 4294967295U)
    {
        _S105 = true;
    }
    else
    {
        _S105 = (*k_5) >= (kernelContext_43->params_0->chunk_count_0);
    }
    if(_S105)
    {
        _S105 = true;
    }
    else
    {
        _S105 = ((uint4(*(kernelContext_43->collider_info_0+2U * *co_0)) ).z) != 2U;
    }
    if(_S105)
    {
        return false;
    }
    uint _S106 = body_of_0(*k_5, kernelContext_43);
    *s_2 = _S106;
    uint _S107 = (as_type<uint>(((float4(*(kernelContext_43->colliders_0+4U * *co_0)) ).x)));
    *ob_0 = _S107;
    if(_S107 == (*s_2))
    {
        _S105 = true;
    }
    else
    {
        _S105 = (*ob_0) >= (kernelContext_43->params_0->max_bodies_0);
    }
    if(_S105)
    {
        return false;
    }
    uint launch_1 = (uint4(*(kernelContext_43->collider_info_0+(2U * *co_0 + 1U))) ).w;
    if(launch_1 != 0U)
    {
        uint _S108 = w_marks_0(kernelContext_43);
        _S105 = (*(kernelContext_43->work_0+(_S108 + *k_5))) == launch_1;
    }
    else
    {
        _S105 = false;
    }
    if(_S105)
    {
        return false;
    }
    uint bs_0 = 7U * *s_2;
    uint bo_0 = 7U * *ob_0;
    float inv_o_0 = (float4(*(kernelContext_43->bodies_0+bo_0)) ).w;
    if(!(inv_o_0 > 0.0f))
    {
        return false;
    }
    float3 n_2 = (float4(*(kernelContext_43->engine_contacts_0+(_S101 + 1U))) ).xyz;
    float3 _S109;
    if(_S102)
    {
        _S109 = - n_2;
    }
    else
    {
        _S109 = n_2;
    }
    *u_1 = _S109;
    Quat_0 _S110 = quat_of_0(float4(*(kernelContext_43->bodies_0+(bs_0 + 1U))) );
    if(_S102)
    {
        _S109 = (float4(*(kernelContext_43->engine_contacts_0+(_S101 + 2U))) ).xyz;
    }
    else
    {
        _S109 = (float4(*(kernelContext_43->engine_contacts_0+(_S101 + 3U))) ).xyz;
    }
    thread Quat_0 _S111 = _S110;
    float3 _S112 = rotate_0(&_S111, _S109);
    Quat_0 _S113 = quat_of_0(float4(*(kernelContext_43->bodies_0+(bo_0 + 1U))) );
    if(_S102)
    {
        _S109 = (float4(*(kernelContext_43->engine_contacts_0+(_S101 + 3U))) ).xyz;
    }
    else
    {
        _S109 = (float4(*(kernelContext_43->engine_contacts_0+(_S101 + 2U))) ).xyz;
    }
    thread Quat_0 _S114 = _S113;
    float3 _S115 = rotate_0(&_S114, _S109);
    float3 _S116 = (float4(*(kernelContext_43->bodies_0+bs_0)) ).xyz + _S112;
    *p_1 = _S116;
    float gap_0 = dot(_S116 - ((float4(*(kernelContext_43->bodies_0+bo_0)) ).xyz + _S115), *u_1);
    float _S117 = dot((float4(*(kernelContext_43->bodies_0+(bo_0 + 3U))) ).xyz + cross((float4(*(kernelContext_43->bodies_0+(bo_0 + 4U))) ).xyz, _S115) - ((float4(*(kernelContext_43->bodies_0+(bs_0 + 3U))) ).xyz + cross((float4(*(kernelContext_43->bodies_0+(bs_0 + 4U))) ).xyz, _S112)), *u_1);
    *v_2 = _S117;
    if(!(_S117 > (kernelContext_43->params_0->strike_speed_0)))
    {
        return false;
    }
    float _S118 = (float4(*(kernelContext_43->bodies_0+bs_0)) ).w;
    float _S119 = cs_angular_0(*s_2, _S112, *u_1, kernelContext_43);
    float _S120 = _S118 + _S119 + inv_o_0;
    float _S121 = cs_angular_0(*ob_0, _S115, *u_1, kernelContext_43);
    float w_1 = _S120 + _S121;
    if(!(w_1 > 0.0f))
    {
        return false;
    }
    *m_eff_1 = 1.0f / w_1;
    *t_in_0 = clamp(gap_0 / *v_2, 0.0f, kernelContext_43->params_0->dt_0);
    return true;
}

uint3 cs_identity_0(uint r_6, uint side_1, KernelContext_0 thread* kernelContext_44)
{
    uint _S122 = cs_row_0(r_6, kernelContext_44);
    uint4 ids_2 = (as_type<uint4>((float4(*(kernelContext_44->engine_contacts_0+_S122)) )));
    uint3 _S123;
    if(side_1 == 0U)
    {
        _S123 = uint3(ids_2.x, ids_2.y, 2U * ids_2.z);
    }
    else
    {
        _S123 = uint3(ids_2.y, ids_2.x, 2U * ids_2.z + 1U);
    }
    return _S123;
}

void cs_pick_0(uint r_7, KernelContext_0 thread* kernelContext_45)
{
    uint _S124 = cs_rows_0(kernelContext_45);
    if(r_7 >= _S124)
    {
        return;
    }
    uint side_2 = 0U;
    for(;;)
    {
        if(side_2 < 2U)
        {
        }
        else
        {
            break;
        }
        thread uint k_6;
        thread uint s_3;
        thread uint ob_1;
        thread uint co_1;
        thread float3 p_2;
        thread float3 u_2;
        thread float v_3;
        thread float m_eff_2;
        thread float t_in_1;
        bool _S125 = cs_side_0(r_7, side_2, &k_6, &s_3, &ob_1, &co_1, &p_2, &u_2, &v_3, &m_eff_2, &t_in_1, kernelContext_45);
        bool _S126;
        if(!_S125)
        {
            _S126 = true;
        }
        else
        {
            uint _S127 = (as_type<uint>((m_eff_2 * v_3)));
            uint _S128 = a_cs_0(s_3, kernelContext_45);
            uint _S129 = atomic_load_explicit(kernelContext_45->atom_0+_S128, memory_order_relaxed);
            _S126 = _S127 != _S129;
        }
        if(_S126)
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint3 _S130 = cs_identity_0(r_7, side_2, kernelContext_45);
        uint _S131 = a_cs_0(s_3, kernelContext_45);
        uint _S132 = atomic_load_explicit(kernelContext_45->atom_0+(_S131 + 2U), memory_order_relaxed);
        if((~_S132) != (_S130.x))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint _S133 = atomic_load_explicit(kernelContext_45->atom_0+(_S131 + 9U), memory_order_relaxed);
        if((~_S133) != (_S130.y))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint _S134 = atomic_load_explicit(kernelContext_45->atom_0+(_S131 + 10U), memory_order_relaxed);
        if((~_S134) != (_S130.z))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        atomic_store_explicit(kernelContext_45->atom_0+(_S131 + 11U), 2U * r_7 + side_2 + 1U, memory_order_relaxed);
        side_2 = side_2 + 1U;
    }
    return;
}

void cs_pick_1(uint r_8, KernelContext_0 thread* kernelContext_46)
{
    uint _S135 = cs_rows_0(kernelContext_46);
    if(r_8 >= _S135)
    {
        return;
    }
    uint side_3 = 0U;
    for(;;)
    {
        if(side_3 < 2U)
        {
        }
        else
        {
            break;
        }
        thread uint k_7;
        thread uint s_4;
        thread uint ob_2;
        thread uint co_2;
        thread float3 p_3;
        thread float3 u_3;
        thread float v_4;
        thread float m_eff_3;
        thread float t_in_2;
        bool _S136 = cs_side_0(r_8, side_3, &k_7, &s_4, &ob_2, &co_2, &p_3, &u_3, &v_4, &m_eff_3, &t_in_2, kernelContext_46);
        bool _S137;
        if(!_S136)
        {
            _S137 = true;
        }
        else
        {
            uint _S138 = (as_type<uint>((m_eff_3 * v_4)));
            uint _S139 = a_cs_0(s_4, kernelContext_46);
            uint _S140 = atomic_load_explicit(kernelContext_46->atom_0+_S139, memory_order_relaxed);
            _S137 = _S138 != _S140;
        }
        if(_S137)
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint3 _S141 = cs_identity_0(r_8, side_3, kernelContext_46);
        uint _S142 = a_cs_0(s_4, kernelContext_46);
        uint _S143 = atomic_load_explicit(kernelContext_46->atom_0+(_S142 + 2U), memory_order_relaxed);
        if((~_S143) != (_S141.x))
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint _S144 = atomic_load_explicit(kernelContext_46->atom_0+(_S142 + 9U), memory_order_relaxed);
        if((~_S144) != (_S141.y))
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint _S145 = atomic_fetch_max_explicit(kernelContext_46->atom_0+(_S142 + 10U), ~_S141.z, memory_order_relaxed);
        side_3 = side_3 + 1U;
    }
    return;
}

void cs_pick_2(uint r_9, KernelContext_0 thread* kernelContext_47)
{
    uint _S146 = cs_rows_0(kernelContext_47);
    if(r_9 >= _S146)
    {
        return;
    }
    uint side_4 = 0U;
    for(;;)
    {
        if(side_4 < 2U)
        {
        }
        else
        {
            break;
        }
        thread uint k_8;
        thread uint s_5;
        thread uint ob_3;
        thread uint co_3;
        thread float3 p_4;
        thread float3 u_4;
        thread float v_5;
        thread float m_eff_4;
        thread float t_in_3;
        bool _S147 = cs_side_0(r_9, side_4, &k_8, &s_5, &ob_3, &co_3, &p_4, &u_4, &v_5, &m_eff_4, &t_in_3, kernelContext_47);
        bool _S148;
        if(!_S147)
        {
            _S148 = true;
        }
        else
        {
            uint _S149 = (as_type<uint>((m_eff_4 * v_5)));
            uint _S150 = a_cs_0(s_5, kernelContext_47);
            uint _S151 = atomic_load_explicit(kernelContext_47->atom_0+_S150, memory_order_relaxed);
            _S148 = _S149 != _S151;
        }
        if(_S148)
        {
            side_4 = side_4 + 1U;
            continue;
        }
        uint3 _S152 = cs_identity_0(r_9, side_4, kernelContext_47);
        uint _S153 = a_cs_0(s_5, kernelContext_47);
        uint _S154 = atomic_load_explicit(kernelContext_47->atom_0+(_S153 + 2U), memory_order_relaxed);
        if((~_S154) != (_S152.x))
        {
            side_4 = side_4 + 1U;
            continue;
        }
        uint _S155 = atomic_fetch_max_explicit(kernelContext_47->atom_0+(_S153 + 9U), ~_S152.y, memory_order_relaxed);
        side_4 = side_4 + 1U;
    }
    return;
}

void cs_pick_3(uint r_10, KernelContext_0 thread* kernelContext_48)
{
    uint _S156 = cs_rows_0(kernelContext_48);
    if(r_10 >= _S156)
    {
        return;
    }
    uint side_5 = 0U;
    for(;;)
    {
        if(side_5 < 2U)
        {
        }
        else
        {
            break;
        }
        thread uint k_9;
        thread uint s_6;
        thread uint ob_4;
        thread uint co_4;
        thread float3 p_5;
        thread float3 u_5;
        thread float v_6;
        thread float m_eff_5;
        thread float t_in_4;
        bool _S157 = cs_side_0(r_10, side_5, &k_9, &s_6, &ob_4, &co_4, &p_5, &u_5, &v_6, &m_eff_5, &t_in_4, kernelContext_48);
        bool _S158;
        if(!_S157)
        {
            _S158 = true;
        }
        else
        {
            uint _S159 = (as_type<uint>((m_eff_5 * v_6)));
            uint _S160 = a_cs_0(s_6, kernelContext_48);
            uint _S161 = atomic_load_explicit(kernelContext_48->atom_0+_S160, memory_order_relaxed);
            _S158 = _S159 != _S161;
        }
        if(_S158)
        {
            side_5 = side_5 + 1U;
            continue;
        }
        uint3 _S162 = cs_identity_0(r_10, side_5, kernelContext_48);
        uint _S163 = a_cs_0(s_6, kernelContext_48);
        uint _S164 = atomic_fetch_max_explicit(kernelContext_48->atom_0+(_S163 + 2U), ~_S162.x, memory_order_relaxed);
        side_5 = side_5 + 1U;
    }
    return;
}

struct IlPlace_0
{
    float along_0;
    float radial_0;
    bool entry_0;
    bool crossed_0;
};

bool il_crossed_0(uint k_10, float3 a_1, float3 d_5, float length_0, float r_11, KernelContext_0 thread* kernelContext_49)
{
    thread Quat_0 q_8;
    float3 _S165 = chunk_box_world_0(k_10, &q_8, kernelContext_49);
    uint _S166 = 3U * k_10;
    Quat_0 _S167 = quat_of_0(float4(*(kernelContext_49->geometry_0+(_S166 + 2U))) );
    thread Quat_0 _S168 = q_8;
    thread Quat_0 _S169 = _S167;
    Quat_0 _S170 = quat_mul_0(&_S168, &_S169);
    thread Quat_0 _S171 = _S170;
    Quat_0 _S172 = normalized_0(&_S171);
    float3 half_0 = (float4(*(kernelContext_49->geometry_0+(_S166 + 1U))) ).xyz;
    thread Quat_0 _S173 = _S172;
    Quat_0 _S174 = conj_0(&_S173);
    float3 _S175 = a_1 - _S165;
    thread Quat_0 _S176 = _S174;
    float3 _S177 = rotate_0(&_S176, _S175);
    thread Quat_0 _S178 = _S174;
    float3 _S179 = rotate_0(&_S178, d_5);
    float lo_0 = 0.0f;
    float hi_0 = length_0;
    uint pass_0 = 0U;
    for(;;)
    {
        if(pass_0 < 40U)
        {
        }
        else
        {
            break;
        }
        float _S180 = hi_0 - lo_0;
        float m1_0 = lo_0 + _S180 / 3.0f;
        float m2_0 = hi_0 - _S180 / 3.0f;
        float3 _S181 = float3(0.0f) ;
        float3 e1_0 = max(abs(_S177 + _S179 * float3(m1_0) ) - half_0, _S181);
        float3 e2_0 = max(abs(_S177 + _S179 * float3(m2_0) ) - half_0, _S181);
        if((dot(e1_0, e1_0)) <= (dot(e2_0, e2_0)))
        {
            hi_0 = m2_0;
        }
        else
        {
            lo_0 = m1_0;
        }
        pass_0 = pass_0 + 1U;
    }
    float3 e_1 = max(abs(_S177 + _S179 * float3((0.5f * (lo_0 + hi_0))) ) - half_0, float3(0.0f) );
    return (dot(e_1, e_1)) < (0.99980002641677856f * r_11 * r_11);
}

IlPlace_0 il_place_0(uint k_11, uint entry_1, float3 contact_4, float3 d_6, float r_12, float h_0, KernelContext_0 thread* kernelContext_50)
{
    uint _S182 = body_of_0(k_11, kernelContext_50);
    thread Quat_0 q_9;
    thread float3 origin_2;
    structure_pose_0(_S182, 7U * _S182, &q_9, &origin_2, kernelContext_50);
    float3 _S183 = (float4(*(kernelContext_50->chunks_0+4U * k_11)) ).xyz;
    thread Quat_0 _S184 = q_9;
    float3 _S185 = rotate_0(&_S184, _S183);
    float3 rel_1 = origin_2 + _S185 - contact_4;
    thread IlPlace_0 p_6;
    float _S186 = dot(rel_1, d_6);
    (&p_6)->along_0 = _S186;
    (&p_6)->radial_0 = length(rel_1 - d_6 * float3(_S186) );
    (&p_6)->entry_0 = k_11 == entry_1;
    bool _S187 = il_crossed_0(k_11, contact_4 - d_6 * float3(r_12) , d_6, h_0 + 2.0f * r_12, r_12, kernelContext_50);
    (&p_6)->crossed_0 = _S187;
    return p_6;
}

bool il_plug_0(uint outcome_0, float r_13, float h_1, const IlPlace_0 thread* p_7)
{
    bool _S188;
    if((outcome_0 & 8U) != 0U)
    {
        if(p_7->entry_0)
        {
            _S188 = true;
        }
        else
        {
            _S188 = p_7->crossed_0;
        }
        if(_S188)
        {
            _S188 = true;
        }
        else
        {
            float _S189 = p_7->along_0;
            if((p_7->along_0) >= 0.0f)
            {
                _S188 = _S189 <= h_1;
            }
            else
            {
                _S188 = false;
            }
            if(_S188)
            {
                _S188 = (p_7->radial_0) <= (r_13 + _S189);
            }
            else
            {
                _S188 = false;
            }
        }
    }
    else
    {
        _S188 = false;
    }
    return _S188;
}

bool il_scab_0(uint outcome_1, float r_14, float h_2, float scab_0, const IlPlace_0 thread* p_8)
{
    bool _S190;
    if((outcome_1 & 4U) != 0U)
    {
        _S190 = (p_8->along_0) >= (h_2 - scab_0);
    }
    else
    {
        _S190 = false;
    }
    if(_S190)
    {
        _S190 = (p_8->along_0) <= h_2;
    }
    else
    {
        _S190 = false;
    }
    if(_S190)
    {
        _S190 = (p_8->radial_0) <= (r_14 + h_2);
    }
    else
    {
        _S190 = false;
    }
    return _S190;
}

bool il_crater_0(uint outcome_2, float r_15, float depth_0, float a_2, const IlPlace_0 thread* p_9)
{
    bool _S191;
    if((outcome_2 & 2U) != 0U)
    {
        _S191 = (p_9->along_0) >= 0.0f;
    }
    else
    {
        _S191 = false;
    }
    if(_S191)
    {
        _S191 = (p_9->along_0) <= depth_0;
    }
    else
    {
        _S191 = false;
    }
    if(_S191)
    {
        _S191 = (p_9->radial_0) <= (max(r_15, a_2));
    }
    else
    {
        _S191 = false;
    }
    return _S191;
}

uint a_counters_1(KernelContext_0 thread* kernelContext_51)
{
    uint _S192 = a_best_root_0(kernelContext_51);
    return _S192 + kernelContext_51->params_0->body_count_0;
}

void il_breaks_0(uint i_0, KernelContext_0 thread* kernelContext_52)
{
    bool cut_0;
    if(i_0 >= (kernelContext_52->params_0->bond_count_0))
    {
        cut_0 = true;
    }
    else
    {
        uint _S193 = w_alive_0(kernelContext_52);
        cut_0 = (*(kernelContext_52->work_0+(_S193 + i_0))) == 0U;
    }
    if(cut_0)
    {
        return;
    }
    uint4 _S194 = uint4(*(kernelContext_52->bonds_0+i_0)) ;
    uint _S195 = _S194.x;
    uint _S196 = body_of_0(_S195, kernelContext_52);
    bool _S197 = il_struck_0(_S196, kernelContext_52);
    if(!_S197)
    {
        return;
    }
    uint _S198 = lp_body_0(_S196, kernelContext_52);
    float4 _S199 = float4(*(kernelContext_52->lp_0+_S198)) ;
    float4 _S200 = float4(*(kernelContext_52->lp_0+(_S198 + 1U))) ;
    uint outcome_3 = (as_type<uint>((_S199.w)));
    if((outcome_3 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_5;
    thread float3 d_7;
    thread float speed_4;
    il_path_0(_S196, &contact_5, &d_7, &speed_4, kernelContext_52);
    uint _S201 = il_entry_0(_S196, kernelContext_52);
    IlStriker_0 _S202 = il_striker_0(_S196, kernelContext_52);
    float _S203 = max(_S202.radius_0, 0.00100000004749745f);
    float h_3 = _S199.x;
    IlPlace_0 _S204 = il_place_0(_S195, _S201, contact_5, d_7, _S203, h_3, kernelContext_52);
    IlPlace_0 _S205 = il_place_0(_S194.y, _S201, contact_5, d_7, _S203, h_3, kernelContext_52);
    thread IlPlace_0 _S206 = _S204;
    bool _S207 = il_plug_0(outcome_3, _S203, h_3, &_S206);
    thread IlPlace_0 _S208 = _S205;
    bool _S209 = il_plug_0(outcome_3, _S203, h_3, &_S208);
    if(_S207 != _S209)
    {
        cut_0 = true;
    }
    else
    {
        float _S210 = _S200.y;
        thread IlPlace_0 _S211 = _S204;
        bool _S212 = il_scab_0(outcome_3, _S203, h_3, _S210, &_S211);
        thread IlPlace_0 _S213 = _S205;
        bool _S214 = il_scab_0(outcome_3, _S203, h_3, _S210, &_S213);
        cut_0 = _S212 != _S214;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S215 = _S200.x;
        float _S216 = _S200.z;
        thread IlPlace_0 _S217 = _S204;
        bool _S218 = il_crater_0(outcome_3, _S203, _S215, _S216, &_S217);
        cut_0 = _S218;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S219 = _S200.x;
        float _S220 = _S200.z;
        thread IlPlace_0 _S221 = _S205;
        bool _S222 = il_crater_0(outcome_3, _S203, _S219, _S220, &_S221);
        cut_0 = _S222;
    }
    if(cut_0)
    {
        uint _S223 = w_alive_0(kernelContext_52);
        *(kernelContext_52->work_0+(_S223 + i_0)) = 0U;
        uint _S224 = a_dirty_0(kernelContext_52);
        uint _S225 = atomic_fetch_or_explicit(kernelContext_52->atom_0+(_S224 + _S196), 1U, memory_order_relaxed);
        uint _S226 = a_counters_1(kernelContext_52);
        uint _S227 = atomic_fetch_add_explicit(kernelContext_52->atom_0+_S226, 1U, memory_order_relaxed);
        uint _S228 = atomic_fetch_add_explicit(kernelContext_52->atom_0+(_S226 + 1U), 1U, memory_order_relaxed);
        uint _S229 = atomic_fetch_add_explicit(kernelContext_52->atom_0+(_S226 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

uint w_carry_0(KernelContext_0 thread* kernelContext_53)
{
    uint _S230 = w_marks_0(kernelContext_53);
    return _S230 + kernelContext_53->params_0->chunk_count_0;
}

float3 to_rubble_0(float3 v_7)
{
    return float3(v_7.x, v_7.z, - v_7.y);
}

float read64_0(uint at_1, float scale_0, KernelContext_0 thread* kernelContext_54)
{
    uint _S231 = atomic_load_explicit(kernelContext_54->atom_0+(at_1 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S231)));
    uint lo_1 = atomic_load_explicit(kernelContext_54->atom_0+at_1, memory_order_relaxed);
    return (float(hi_1) * 4.294967296e+09f + float(lo_1)) / scale_0;
}

void il_carry_0(uint k_12, KernelContext_0 thread* kernelContext_55)
{
    if(k_12 >= (kernelContext_55->params_0->chunk_count_0))
    {
        return;
    }
    uint _S232 = w_carry_0(kernelContext_55);
    uint _S233 = _S232 + k_12;
    uint device* _S234 = kernelContext_55->work_0+_S233;
    uint c_2 = *_S234;
    if((*_S234) == 0U)
    {
        return;
    }
    *(kernelContext_55->work_0+_S233) = 0U;
    uint b0_0 = c_2 - 1U;
    bool _S235 = il_struck_0(b0_0, kernelContext_55);
    if(!_S235)
    {
        return;
    }
    thread float3 contact_6;
    thread float3 d_8;
    thread float speed_5;
    il_path_0(b0_0, &contact_6, &d_8, &speed_5, kernelContext_55);
    float3 dr_0 = to_rubble_0(d_8);
    IlStriker_0 _S236 = il_striker_0(b0_0, kernelContext_55);
    uint _S237 = a_il_0(b0_0, kernelContext_55);
    float _S238 = read64_0(_S237 + 2U, kernelContext_55->params_0->scale_m_0, kernelContext_55);
    float3 v_wall_0 = (float4(*(kernelContext_55->bodies_0+(7U * b0_0 + 3U))) ).xyz;
    uint _S239 = 7U * _S236.body_0;
    float3 v_shot_0 = (float4(*(kernelContext_55->bodies_0+(_S239 + 3U))) ).xyz;
    float closing_0 = dot(v_shot_0 - v_wall_0, dr_0);
    float v_r_0 = _S236.mass_0 * closing_0 / (_S236.mass_0 + _S238);
    float3 _S240 = v_shot_0 + dr_0 * float3((v_r_0 - closing_0)) ;
    float3 _S241 = v_wall_0 + dr_0 * float3(v_r_0) ;
    float _S242 = il_t_in_0(b0_0, kernelContext_55);
    float3 _S243 = (float4(*(kernelContext_55->bodies_0+_S239)) ).xyz + dr_0 * float3(((speed_5 - v_r_0) * clamp(_S242, 0.0f, kernelContext_55->params_0->dt_0))) ;
    uint _S244 = body_of_0(k_12, kernelContext_55);
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
        uint s_at_0 = 7U * (_S236.body_0 + region_0 * kernelContext_55->params_0->max_bodies_0);
        (*(kernelContext_55->bodies_0+s_at_0)).xyz = _S243;
        (*(kernelContext_55->bodies_0+(s_at_0 + 3U))).xyz = _S240;
        (*(kernelContext_55->bodies_0+(7U * (_S244 + region_0 * kernelContext_55->params_0->max_bodies_0) + 3U))).xyz = _S241;
        region_0 = region_0 + 1U;
    }
    return;
}

void add64_0(uint at_2, float value_0, float scale_1, KernelContext_0 thread* kernelContext_56)
{
    float x_1 = round(value_0 * scale_1);
    float x_2;
    if(!((abs(x_1)) < 1.073741824e+09f))
    {
        uint _S245 = a_counters_1(kernelContext_56);
        uint _S246 = atomic_fetch_or_explicit(kernelContext_56->atom_0+(_S245 + 3U), 2U, memory_order_relaxed);
        x_2 = 0.0f;
    }
    else
    {
        x_2 = x_1;
    }
    int v_8 = int(x_2);
    uint lo_2 = (as_type<uint>((v_8)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_56->atom_0+at_2, lo_2, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_2) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S247;
    if(v_8 < int(0))
    {
        _S247 = 4294967295U;
    }
    else
    {
        _S247 = 0U;
    }
    uint hi_2 = _S247 + carry_0;
    if(hi_2 != 0U)
    {
        uint _S248 = atomic_fetch_add_explicit(kernelContext_56->atom_0+(at_2 + 1U), hi_2, memory_order_relaxed);
    }
    return;
}

void il_mark_0(uint k_13, KernelContext_0 thread* kernelContext_57)
{
    if(k_13 >= (kernelContext_57->params_0->chunk_count_0))
    {
        return;
    }
    uint _S249 = body_of_0(k_13, kernelContext_57);
    bool _S250 = il_struck_0(_S249, kernelContext_57);
    if(!_S250)
    {
        return;
    }
    uint _S251 = lp_body_0(_S249, kernelContext_57);
    float4 _S252 = float4(*(kernelContext_57->lp_0+_S251)) ;
    float4 _S253 = float4(*(kernelContext_57->lp_0+(_S251 + 1U))) ;
    uint outcome_4 = (as_type<uint>((_S252.w)));
    if((outcome_4 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_7;
    thread float3 d_9;
    thread float speed_6;
    il_path_0(_S249, &contact_7, &d_9, &speed_6, kernelContext_57);
    uint _S254 = il_entry_0(_S249, kernelContext_57);
    IlStriker_0 _S255 = il_striker_0(_S249, kernelContext_57);
    float _S256 = max(_S255.radius_0, 0.00100000004749745f);
    float _S257 = _S252.x;
    IlPlace_0 _S258 = il_place_0(k_13, _S254, contact_7, d_9, _S256, _S257, kernelContext_57);
    thread IlPlace_0 _S259 = _S258;
    bool _S260 = il_plug_0(outcome_4, _S256, _S257, &_S259);
    if((_S255.launch_0) != 0U)
    {
        float2 _S261 = il_interval_0(k_13, contact_7 - d_9 * float3(_S255.bound_0) , d_9, _S255.bound_0, kernelContext_57);
        float _S262 = _S261.x;
        float _S263 = _S261.y;
        bool path_0;
        if(_S262 <= _S263)
        {
            path_0 = _S262 <= (_S257 + 2.0f * _S255.bound_0);
        }
        else
        {
            path_0 = false;
        }
        if(path_0)
        {
            path_0 = _S263 >= 0.0f;
        }
        else
        {
            path_0 = false;
        }
        if(_S260)
        {
            path_0 = true;
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            float _S264 = _S253.y;
            thread IlPlace_0 _S265 = _S258;
            bool _S266 = il_scab_0(outcome_4, _S256, _S257, _S264, &_S265);
            path_0 = _S266;
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            float _S267 = _S253.x;
            float _S268 = _S253.z;
            thread IlPlace_0 _S269 = _S258;
            bool _S270 = il_crater_0(outcome_4, _S256, _S267, _S268, &_S269);
            path_0 = _S270;
        }
        if(path_0)
        {
            uint _S271 = w_marks_0(kernelContext_57);
            *(kernelContext_57->work_0+(_S271 + k_13)) = _S255.launch_0;
        }
    }
    if(_S260)
    {
        uint _S272 = w_carry_0(kernelContext_57);
        *(kernelContext_57->work_0+(_S272 + k_13)) = _S249 + 1U;
        uint _S273 = a_il_0(_S249, kernelContext_57);
        add64_0(_S273 + 2U, (float4(*(kernelContext_57->chunks_0+4U * k_13)) ).w, kernelContext_57->params_0->scale_m_0, kernelContext_57);
    }
    return;
}

struct JointMaterial_0
{
    float4 strength_0;
    float4 energy_0;
    float4 dif_0;
    float4 misc_0;
    uint4 kind_flags_0;
};

float dif_factor_0(const JointMaterial_0 thread* mat_0, float strain_rate_0)
{
    float r_16 = abs(strain_rate_0);
    float4 _S274 = mat_0->dif_0;
    float ref_0 = mat_0->dif_0.x;
    if(r_16 <= ref_0)
    {
        return 1.0f;
    }
    float _S275 = _S274.z;
    float f_0;
    if(r_16 <= _S275)
    {
        float _S276 = (metal::fast::divide((r_16), (ref_0)));
        float _S277 = (metal::fast::pow((_S276), (_S274.y)));
        f_0 = _S277;
    }
    else
    {
        float _S278 = (metal::fast::divide((_S275), (ref_0)));
        float _S279 = (metal::fast::pow((_S278), (_S274.y)));
        float _S280 = (metal::fast::divide((r_16), (_S275)));
        float _S281 = (metal::fast::pow((_S280), (_S274.w)));
        f_0 = _S279 * _S281;
    }
    return clamp(f_0, 1.0f, mat_0->misc_0.x);
}

void il_impulse_0(uint b_20, uint sb_0, float m_shot_0, float3 contact_8, float3 d_10, float j_0, KernelContext_0 thread* kernelContext_58)
{
    float3 jr_0 = to_rubble_0(d_10) * float3(j_0) ;
    uint at_3 = 7U * b_20;
    float inv_mass_1 = (float4(*(kernelContext_58->bodies_0+at_3)) ).w;
    float3 _S282 = jr_0 * float3(inv_mass_1) ;
    float3 _S283 = float3(0.0f) ;
    float3 dw_0;
    if(inv_mass_1 > 0.0f)
    {
        Quat_0 q_10 = quat_of_0(float4(*(kernelContext_58->bodies_0+(at_3 + 1U))) );
        float3 r_17 = to_rubble_0(contact_8) - (float4(*(kernelContext_58->bodies_0+at_3)) ).xyz;
        float3 _S284 = (float4(*(kernelContext_58->bodies_0+(at_3 + 2U))) ).xyz;
        thread Quat_0 _S285 = q_10;
        Quat_0 _S286 = conj_0(&_S285);
        float3 _S287 = cross(r_17, jr_0);
        thread Quat_0 _S288 = _S286;
        float3 _S289 = rotate_0(&_S288, _S287);
        float3 _S290 = _S284 * _S289;
        thread Quat_0 _S291 = q_10;
        float3 _S292 = rotate_0(&_S291, _S290);
        dw_0 = _S292;
    }
    else
    {
        dw_0 = _S283;
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
        uint s_at_1 = 7U * (sb_0 + region_1 * kernelContext_58->params_0->max_bodies_0) + 3U;
        (*(kernelContext_58->bodies_0+s_at_1)).xyz = (float4(*(kernelContext_58->bodies_0+s_at_1)) ).xyz - jr_0 / float3(m_shot_0) ;
        uint w_at_0 = 7U * (b_20 + region_1 * kernelContext_58->params_0->max_bodies_0);
        uint _S293 = w_at_0 + 3U;
        (*(kernelContext_58->bodies_0+_S293)).xyz = (float4(*(kernelContext_58->bodies_0+_S293)) ).xyz + _S282;
        uint _S294 = w_at_0 + 4U;
        (*(kernelContext_58->bodies_0+_S294)).xyz = (float4(*(kernelContext_58->bodies_0+_S294)) ).xyz + dw_0;
        region_1 = region_1 + 1U;
    }
    return;
}

void il_strike_0(uint b_21, KernelContext_0 thread* kernelContext_59)
{
    bool damage_0;
    if(b_21 >= (kernelContext_59->params_0->body_count_0))
    {
        damage_0 = true;
    }
    else
    {
        bool _S295 = il_struck_0(b_21, kernelContext_59);
        damage_0 = !_S295;
    }
    if(damage_0)
    {
        return;
    }
    float _S296 = il_t_in_0(b_21, kernelContext_59);
    uint _S297 = il_entry_0(b_21, kernelContext_59);
    thread float3 contact_9;
    thread float3 d_11;
    thread float speed_7;
    il_path_0(b_21, &contact_9, &d_11, &speed_7, kernelContext_59);
    IlStriker_0 _S298 = il_striker_0(b_21, kernelContext_59);
    float _S299 = max(_S298.radius_0, 0.00100000004749745f);
    if((_S298.launch_0) != 0U)
    {
        uint _S300 = w_marks_0(kernelContext_59);
        damage_0 = (*(kernelContext_59->work_0+(_S300 + _S297))) == (_S298.launch_0);
    }
    else
    {
        damage_0 = false;
    }
    if(damage_0)
    {
        uint _S301 = lp_body_0(b_21, kernelContext_59);
        packed_float4 _S302 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_59->lp_0+_S301) = _S302;
        *(kernelContext_59->lp_0+(_S301 + 1U)) = _S302;
        return;
    }
    uint _S303 = a_il_0(b_21, kernelContext_59);
    uint _S304 = atomic_load_explicit(kernelContext_59->atom_0+_S303, memory_order_relaxed);
    float _S305 = max((as_type<float>((_S304))) - _S299, 9.99999997475242708e-07f);
    uint o_2 = kernelContext_59->params_0->ext_chunks_0 + 4U * _S297;
    float4 _S306 = float4(*(kernelContext_59->ext_0+(o_2 + 1U))) ;
    thread JointMaterial_0 dm_0;
    float4 _S307 = float4(0.0f) ;
    (&dm_0)->strength_0 = _S307;
    (&dm_0)->energy_0 = _S307;
    (&dm_0)->dif_0 = float4(*(kernelContext_59->ext_0+(o_2 + 2U))) ;
    float4 _S308 = float4(*(kernelContext_59->ext_0+(o_2 + 3U))) ;
    float _S309 = _S308.x;
    (&dm_0)->misc_0 = float4(_S309, 0.0f, 0.0f, 0.0f);
    (&dm_0)->kind_flags_0 = uint4(0U) ;
    float nu_0 = _S308.y;
    float e_target_0 = _S308.z / (1.0f - nu_0 * nu_0);
    float e_star_0 = 1.0f / (1.0f / _S298.modulus_0 + 1.0f / e_target_0);
    float _S310 = max(speed_7, 0.00100000004749745f);
    float _S311 = sqrt(_S299);
    float h0_0 = pow(15.0f * _S298.m_eff_0 * _S310 * _S310 / (16.0f * e_star_0 * _S311), 0.40000000596046448f);
    float t_c_0 = 2.94000005722045898f * h0_0 / _S310;
    float force_0 = 1.33333337306976318f * e_star_0 * _S311 * pow(h0_0, 1.5f);
    float a_3 = sqrt(_S299 * h0_0);
    float p0_0 = 3.0f * force_0 / (6.28318548202514648f * a_3 * a_3);
    float ring_0 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    float k_14;
    if(_S309 > 0.0f)
    {
        float _S312 = _S310 / _S305;
        thread JointMaterial_0 _S313 = dm_0;
        float _S314 = dif_factor_0(&_S313, _S312);
        k_14 = _S314;
    }
    else
    {
        k_14 = 1.0f;
    }
    float ft_0 = k_14 * _S306.y;
    float fc_0 = k_14 * _S306.z;
    float tau_0 = k_14 * _S306.w;
    if(p0_0 > fc_0)
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ring_0 > ft_0;
    }
    float _S315 = fc_0 * (3.14159274101257324f * _S299 * _S299);
    float _S316 = min(_S315 * _S305, 1.57079637050628662f * (2.0f * _S299 + _S305) * _S305 * _S305 * tau_0);
    float _S317 = 2.0f * _S316;
    float v_bl_0 = sqrt(_S317 / _S298.m_eff_0);
    bool perforate_0;
    if(damage_0)
    {
        perforate_0 = _S310 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    float r2_0 = _S299 + _S305;
    float _S318 = _S306.x;
    float plug_0 = _S318 * 3.14159274101257324f * _S305 / 3.0f * (_S299 * _S299 + _S299 * r2_0 + r2_0 * r2_0);
    float v_res_0;
    if(perforate_0)
    {
        v_res_0 = _S298.mass_0 / (_S298.mass_0 + plug_0) * sqrt(max(_S310 * _S310 - v_bl_0 * v_bl_0, 0.0f));
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
    float depth_1;
    if(spall_0)
    {
        depth_1 = min(0.5f * _S298.m_eff_0 * _S310 * _S310 / _S315, _S305);
    }
    else
    {
        depth_1 = 0.0f;
    }
    bool crater_0 = depth_1 > a_3;
    float c_3 = sqrt(e_target_0 / _S318);
    float z_t_0 = _S318 * c_3;
    float rho_p_0 = _S298.mass_0 / (4.18879032135009766f * _S299 * _S299 * _S299);
    float z_p_0 = rho_p_0 * sqrt(_S298.modulus_0 / rho_p_0);
    float mean_pressure_0 = force_0 / (3.14159274101257324f * a_3 * a_3);
    float _S319 = min(min(z_t_0 * _S310 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_0);
    float source_radius_0;
    if(mean_pressure_0 > fc_0)
    {
        source_radius_0 = max(a_3, _S299);
    }
    else
    {
        source_radius_0 = a_3;
    }
    float pulse_0 = c_3 * t_c_0;
    float back_0 = _S319 * min(source_radius_0 / _S305, 1.0f) * min(2.0f * _S305 / pulse_0, 1.0f);
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
    float scab_1;
    if(spall_0)
    {
        scab_1 = 0.5f * min(pulse_0, _S305);
    }
    else
    {
        scab_1 = 0.0f;
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
    uint _S320;
    if(crater_0)
    {
        _S320 = 2U;
    }
    else
    {
        _S320 = 0U;
    }
    uint _S321 = region_2 | _S320;
    if(spall_0)
    {
        region_2 = 4U;
    }
    else
    {
        region_2 = 0U;
    }
    uint _S322 = _S321 | region_2;
    if(perforate_0)
    {
        region_2 = 8U;
    }
    else
    {
        region_2 = 0U;
    }
    uint outcome_5 = _S322 | region_2;
    uint _S323 = lp_body_0(b_21, kernelContext_59);
    *(kernelContext_59->lp_0+_S323) = packed_float4(float4(_S305, v_bl_0, v_res_0, (as_type<float>((outcome_5))))) ;
    packed_float4 device* _S324 = kernelContext_59->lp_0+(_S323 + 1U);
    if(crater_0)
    {
        k_14 = depth_1;
    }
    else
    {
        k_14 = 0.0f;
    }
    *_S324 = packed_float4(float4(k_14, scab_1, a_3, _S316)) ;
    if(perforate_0)
    {
        uint _S325 = a_counters_1(kernelContext_59);
        uint _S326 = atomic_fetch_add_explicit(kernelContext_59->atom_0+(_S325 + 10U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_0)
        {
            uint _S327 = a_counters_1(kernelContext_59);
            uint _S328 = atomic_fetch_add_explicit(kernelContext_59->atom_0+(_S327 + 9U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_0)
            {
                uint _S329 = a_counters_1(kernelContext_59);
                uint _S330 = atomic_fetch_add_explicit(kernelContext_59->atom_0+(_S329 + 8U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S331 = a_counters_1(kernelContext_59);
                uint _S332 = atomic_fetch_add_explicit(kernelContext_59->atom_0+(_S331 + 7U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_0)
    {
        if((kernelContext_59->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_21, _S298.body_0, _S298.mass_0, contact_9, d_11, _S298.m_eff_0 * (_S310 - sqrt(max(_S310 * _S310 - _S317 / _S298.m_eff_0, 0.0f))), kernelContext_59);
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_21, _S298.body_0, _S298.mass_0, contact_9, d_11, _S298.m_eff_0 * _S310, kernelContext_59);
            float3 _S333 = to_rubble_0(d_11) * float3((_S310 * clamp(_S296, 0.0f, kernelContext_59->params_0->dt_0))) ;
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
                uint _S334 = 7U * (_S298.body_0 + region_2 * kernelContext_59->params_0->max_bodies_0);
                (*(kernelContext_59->bodies_0+_S334)).xyz = (float4(*(kernelContext_59->bodies_0+_S334)) ).xyz + _S333;
                region_2 = region_2 + 1U;
            }
        }
    }
    return;
}

uint group_uniform_0(uint value_1, uint thread_0)
{
    return value_1;
}

float group_uniform_float_0(float value_2, uint thread_1)
{
    return value_2;
}

uint w_il_tube_0(KernelContext_0 thread* kernelContext_60)
{
    uint _S335 = w_carry_0(kernelContext_60);
    return _S335 + 2U * kernelContext_60->params_0->chunk_count_0 + kernelContext_60->params_0->body_count_0;
}

void il_run_0(uint b_22, uint i_1, KernelContext_0 thread* kernelContext_61)
{
    bool _S336 = il_struck_0(b_22, kernelContext_61);
    uint j_1;
    if(_S336)
    {
        j_1 = 1U;
    }
    else
    {
        j_1 = 0U;
    }
    if((group_uniform_0(j_1, i_1)) == 0U)
    {
        return;
    }
    IlStriker_0 _S337 = il_striker_0(b_22, kernelContext_61);
    float _S338 = 0.00100000004749745f + 0.01999999955296516f * _S337.radius_0;
    uint _S339 = a_il_0(b_22, kernelContext_61);
    uint _S340 = atomic_load_explicit(kernelContext_61->atom_0+(_S339 + 4U), memory_order_relaxed);
    uint _S341 = atomic_load_explicit(kernelContext_61->atom_0+(_S339 + 1U), memory_order_relaxed);
    uint _S342 = atomic_load_explicit(kernelContext_61->atom_0+_S339, memory_order_relaxed);
    float end_0 = group_uniform_float_0((as_type<float>((_S342))), i_1);
    for(;;)
    {
        float best_0 = end_0;
        j_1 = i_1;
        for(;;)
        {
            if(j_1 < _S341)
            {
            }
            else
            {
                break;
            }
            uint _S343 = w_il_tube_0(kernelContext_61);
            uint at_4 = _S343 + 2U * (_S340 + j_1);
            if((as_type<float>((*(kernelContext_61->work_0+at_4)))) <= (end_0 + _S338))
            {
                best_0 = max(best_0, (as_type<float>((*(kernelContext_61->work_0+(at_4 + 1U))))));
            }
            j_1 = j_1 + 64U;
        }
        (*kernelContext_61->g_il_best_0)[i_1] = best_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint step_1 = 32U;
        for(;;)
        {
            if(step_1 > 0U)
            {
            }
            else
            {
                break;
            }
            if(i_1 < step_1)
            {
                (*kernelContext_61->g_il_best_0)[i_1] = max((*kernelContext_61->g_il_best_0)[i_1], (*kernelContext_61->g_il_best_0)[i_1 + step_1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_1 = step_1 >> 1U;
        }
        float next_0 = group_uniform_float_0((*kernelContext_61->g_il_best_0)[int(0)], i_1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!(next_0 > end_0))
        {
            break;
        }
        end_0 = next_0;
    }
    if(i_1 == 0U)
    {
        atomic_store_explicit(kernelContext_61->atom_0+_S339, (as_type<uint>((end_0))), memory_order_relaxed);
    }
    return;
}

void il_tube_collect_0(uint k_15, KernelContext_0 thread* kernelContext_62)
{
    thread uint b_23;
    thread float2 t_3;
    bool _S344 = il_tube_0(k_15, &b_23, &t_3, kernelContext_62);
    if(!_S344)
    {
        return;
    }
    uint _S345 = w_il_tube_0(kernelContext_62);
    uint _S346 = a_il_0(b_23, kernelContext_62);
    uint _S347 = atomic_load_explicit(kernelContext_62->atom_0+(_S346 + 4U), memory_order_relaxed);
    uint _S348 = a_il_0(b_23, kernelContext_62);
    uint _S349 = atomic_fetch_add_explicit(kernelContext_62->atom_0+(_S348 + 5U), 1U, memory_order_relaxed);
    uint at_5 = _S345 + 2U * (_S347 + _S349);
    *(kernelContext_62->work_0+at_5) = (as_type<uint>((t_3.x)));
    *(kernelContext_62->work_0+(at_5 + 1U)) = (as_type<uint>((t_3.y)));
    return;
}

uint a_il_cursor_1(KernelContext_0 thread* kernelContext_63)
{
    uint _S350 = a_first_0(0U, kernelContext_63);
    return _S350 + 64U;
}

void il_tube_base_0(uint b_24, KernelContext_0 thread* kernelContext_64)
{
    bool _S351;
    if(b_24 < (kernelContext_64->params_0->body_count_0))
    {
        bool _S352 = il_struck_0(b_24, kernelContext_64);
        _S351 = _S352;
    }
    else
    {
        _S351 = false;
    }
    if(_S351)
    {
        uint _S353 = a_il_0(b_24, kernelContext_64);
        atomic<uint> device* _S354 = kernelContext_64->atom_0+(_S353 + 4U);
        uint _S355 = a_il_cursor_1(kernelContext_64);
        atomic<uint> device* _S356 = kernelContext_64->atom_0+_S355;
        uint _S357 = atomic_load_explicit(kernelContext_64->atom_0+(_S353 + 1U), memory_order_relaxed);
        uint _S358 = atomic_fetch_add_explicit(_S356, _S357, memory_order_relaxed);
        atomic_store_explicit(_S354, _S358, memory_order_relaxed);
    }
    return;
}

void il_tube_count_0(uint k_16, KernelContext_0 thread* kernelContext_65)
{
    thread uint b_25;
    thread float2 t_4;
    bool _S359 = il_tube_0(k_16, &b_25, &t_4, kernelContext_65);
    if(_S359)
    {
        uint _S360 = a_il_0(b_25, kernelContext_65);
        uint _S361 = atomic_fetch_add_explicit(kernelContext_65->atom_0+(_S360 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

void il_begin_0(uint b_26, KernelContext_0 thread* kernelContext_66)
{
    bool _S362;
    if(b_26 >= (kernelContext_66->params_0->body_count_0))
    {
        _S362 = true;
    }
    else
    {
        bool _S363 = il_struck_0(b_26, kernelContext_66);
        _S362 = !_S363;
    }
    if(_S362)
    {
        return;
    }
    thread float3 contact_10;
    thread float3 d_12;
    thread float speed_8;
    il_path_0(b_26, &contact_10, &d_12, &speed_8, kernelContext_66);
    uint _S364 = il_entry_0(b_26, kernelContext_66);
    IlStriker_0 _S365 = il_striker_0(b_26, kernelContext_66);
    float2 _S366 = il_interval_0(_S364, contact_10, d_12, _S365.radius_0, kernelContext_66);
    uint _S367 = a_il_0(b_26, kernelContext_66);
    atomic_store_explicit(kernelContext_66->atom_0+_S367, (as_type<uint>((max(_S366.y, 0.0f)))), memory_order_relaxed);
    uint i_2 = 1U;
    for(;;)
    {
        if(i_2 < 6U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit(kernelContext_66->atom_0+(_S367 + i_2), 0U, memory_order_relaxed);
        i_2 = i_2 + 1U;
    }
    return;
}

void il_breaks_1(uint i_3, KernelContext_0 thread* kernelContext_67)
{
    bool cut_1;
    if(i_3 >= (kernelContext_67->params_0->bond_count_0))
    {
        cut_1 = true;
    }
    else
    {
        uint _S368 = w_alive_0(kernelContext_67);
        cut_1 = (*(kernelContext_67->work_0+(_S368 + i_3))) == 0U;
    }
    if(cut_1)
    {
        return;
    }
    uint4 _S369 = uint4(*(kernelContext_67->bonds_0+i_3)) ;
    uint _S370 = _S369.x;
    uint _S371 = body_of_0(_S370, kernelContext_67);
    bool _S372 = il_struck_1(_S371, kernelContext_67);
    if(!_S372)
    {
        return;
    }
    uint _S373 = lp_body_0(_S371, kernelContext_67);
    float4 _S374 = float4(*(kernelContext_67->lp_0+_S373)) ;
    float4 _S375 = float4(*(kernelContext_67->lp_0+(_S373 + 1U))) ;
    uint outcome_6 = (as_type<uint>((_S374.w)));
    if((outcome_6 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_11;
    thread float3 d_13;
    thread float speed_9;
    il_path_1(_S371, &contact_11, &d_13, &speed_9, kernelContext_67);
    uint _S376 = il_entry_1(_S371, kernelContext_67);
    IlStriker_0 _S377 = il_striker_1(_S371, kernelContext_67);
    float _S378 = max(_S377.radius_0, 0.00100000004749745f);
    float h_4 = _S374.x;
    IlPlace_0 _S379 = il_place_0(_S370, _S376, contact_11, d_13, _S378, h_4, kernelContext_67);
    IlPlace_0 _S380 = il_place_0(_S369.y, _S376, contact_11, d_13, _S378, h_4, kernelContext_67);
    thread IlPlace_0 _S381 = _S379;
    bool _S382 = il_plug_0(outcome_6, _S378, h_4, &_S381);
    thread IlPlace_0 _S383 = _S380;
    bool _S384 = il_plug_0(outcome_6, _S378, h_4, &_S383);
    if(_S382 != _S384)
    {
        cut_1 = true;
    }
    else
    {
        float _S385 = _S375.y;
        thread IlPlace_0 _S386 = _S379;
        bool _S387 = il_scab_0(outcome_6, _S378, h_4, _S385, &_S386);
        thread IlPlace_0 _S388 = _S380;
        bool _S389 = il_scab_0(outcome_6, _S378, h_4, _S385, &_S388);
        cut_1 = _S387 != _S389;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S390 = _S375.x;
        float _S391 = _S375.z;
        thread IlPlace_0 _S392 = _S379;
        bool _S393 = il_crater_0(outcome_6, _S378, _S390, _S391, &_S392);
        cut_1 = _S393;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S394 = _S375.x;
        float _S395 = _S375.z;
        thread IlPlace_0 _S396 = _S380;
        bool _S397 = il_crater_0(outcome_6, _S378, _S394, _S395, &_S396);
        cut_1 = _S397;
    }
    if(cut_1)
    {
        uint _S398 = w_alive_0(kernelContext_67);
        *(kernelContext_67->work_0+(_S398 + i_3)) = 0U;
        uint _S399 = a_dirty_0(kernelContext_67);
        uint _S400 = atomic_fetch_or_explicit(kernelContext_67->atom_0+(_S399 + _S371), 1U, memory_order_relaxed);
        uint _S401 = a_counters_1(kernelContext_67);
        uint _S402 = atomic_fetch_add_explicit(kernelContext_67->atom_0+_S401, 1U, memory_order_relaxed);
        uint _S403 = atomic_fetch_add_explicit(kernelContext_67->atom_0+(_S401 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

void il_carry_1(uint k_17, KernelContext_0 thread* kernelContext_68)
{
    if(k_17 >= (kernelContext_68->params_0->chunk_count_0))
    {
        return;
    }
    uint _S404 = w_carry_0(kernelContext_68);
    uint _S405 = _S404 + k_17;
    uint device* _S406 = kernelContext_68->work_0+_S405;
    uint c_4 = *_S406;
    if((*_S406) == 0U)
    {
        return;
    }
    *(kernelContext_68->work_0+_S405) = 0U;
    uint b0_1 = c_4 - 1U;
    bool _S407 = il_struck_1(b0_1, kernelContext_68);
    if(!_S407)
    {
        return;
    }
    thread float3 contact_12;
    thread float3 d_14;
    thread float speed_10;
    il_path_1(b0_1, &contact_12, &d_14, &speed_10, kernelContext_68);
    float3 dr_1 = to_rubble_0(d_14);
    IlStriker_0 _S408 = il_striker_1(b0_1, kernelContext_68);
    uint _S409 = a_il_0(b0_1, kernelContext_68);
    float _S410 = read64_0(_S409 + 2U, kernelContext_68->params_0->scale_m_0, kernelContext_68);
    float3 v_wall_1 = (float4(*(kernelContext_68->bodies_0+(7U * b0_1 + 3U))) ).xyz;
    uint _S411 = 7U * _S408.body_0;
    float3 v_shot_1 = (float4(*(kernelContext_68->bodies_0+(_S411 + 3U))) ).xyz;
    float closing_1 = dot(v_shot_1 - v_wall_1, dr_1);
    float v_r_1 = _S408.mass_0 * closing_1 / (_S408.mass_0 + _S410);
    float3 _S412 = v_shot_1 + dr_1 * float3((v_r_1 - closing_1)) ;
    float3 _S413 = v_wall_1 + dr_1 * float3(v_r_1) ;
    float _S414 = il_t_in_1(b0_1, kernelContext_68);
    float3 _S415 = (float4(*(kernelContext_68->bodies_0+_S411)) ).xyz + dr_1 * float3(((speed_10 - v_r_1) * clamp(_S414, 0.0f, kernelContext_68->params_0->dt_0))) ;
    uint _S416 = body_of_0(k_17, kernelContext_68);
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
        uint s_at_2 = 7U * (_S408.body_0 + region_3 * kernelContext_68->params_0->max_bodies_0);
        (*(kernelContext_68->bodies_0+s_at_2)).xyz = _S415;
        (*(kernelContext_68->bodies_0+(s_at_2 + 3U))).xyz = _S412;
        (*(kernelContext_68->bodies_0+(7U * (_S416 + region_3 * kernelContext_68->params_0->max_bodies_0) + 3U))).xyz = _S413;
        region_3 = region_3 + 1U;
    }
    return;
}

void il_mark_1(uint k_18, KernelContext_0 thread* kernelContext_69)
{
    if(k_18 >= (kernelContext_69->params_0->chunk_count_0))
    {
        return;
    }
    uint _S417 = body_of_0(k_18, kernelContext_69);
    bool _S418 = il_struck_1(_S417, kernelContext_69);
    if(!_S418)
    {
        return;
    }
    uint _S419 = lp_body_0(_S417, kernelContext_69);
    float4 _S420 = float4(*(kernelContext_69->lp_0+_S419)) ;
    float4 _S421 = float4(*(kernelContext_69->lp_0+(_S419 + 1U))) ;
    uint outcome_7 = (as_type<uint>((_S420.w)));
    if((outcome_7 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_13;
    thread float3 d_15;
    thread float speed_11;
    il_path_1(_S417, &contact_13, &d_15, &speed_11, kernelContext_69);
    uint _S422 = il_entry_1(_S417, kernelContext_69);
    IlStriker_0 _S423 = il_striker_1(_S417, kernelContext_69);
    float _S424 = max(_S423.radius_0, 0.00100000004749745f);
    float _S425 = _S420.x;
    IlPlace_0 _S426 = il_place_0(k_18, _S422, contact_13, d_15, _S424, _S425, kernelContext_69);
    thread IlPlace_0 _S427 = _S426;
    bool _S428 = il_plug_0(outcome_7, _S424, _S425, &_S427);
    if((_S423.launch_0) != 0U)
    {
        float2 _S429 = il_interval_0(k_18, contact_13 - d_15 * float3(_S423.bound_0) , d_15, _S423.bound_0, kernelContext_69);
        float _S430 = _S429.x;
        float _S431 = _S429.y;
        bool path_1;
        if(_S430 <= _S431)
        {
            path_1 = _S430 <= (_S425 + 2.0f * _S423.bound_0);
        }
        else
        {
            path_1 = false;
        }
        if(path_1)
        {
            path_1 = _S431 >= 0.0f;
        }
        else
        {
            path_1 = false;
        }
        if(_S428)
        {
            path_1 = true;
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            float _S432 = _S421.y;
            thread IlPlace_0 _S433 = _S426;
            bool _S434 = il_scab_0(outcome_7, _S424, _S425, _S432, &_S433);
            path_1 = _S434;
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            float _S435 = _S421.x;
            float _S436 = _S421.z;
            thread IlPlace_0 _S437 = _S426;
            bool _S438 = il_crater_0(outcome_7, _S424, _S435, _S436, &_S437);
            path_1 = _S438;
        }
        if(path_1)
        {
            uint _S439 = w_marks_0(kernelContext_69);
            *(kernelContext_69->work_0+(_S439 + k_18)) = _S423.launch_0;
        }
    }
    if(_S428)
    {
        uint _S440 = w_carry_0(kernelContext_69);
        *(kernelContext_69->work_0+(_S440 + k_18)) = _S417 + 1U;
        uint _S441 = a_il_0(_S417, kernelContext_69);
        add64_0(_S441 + 2U, (float4(*(kernelContext_69->chunks_0+4U * k_18)) ).w, kernelContext_69->params_0->scale_m_0, kernelContext_69);
    }
    return;
}

void il_strike_1(uint b_27, KernelContext_0 thread* kernelContext_70)
{
    bool damage_1;
    if(b_27 >= (kernelContext_70->params_0->body_count_0))
    {
        damage_1 = true;
    }
    else
    {
        bool _S442 = il_struck_1(b_27, kernelContext_70);
        damage_1 = !_S442;
    }
    if(damage_1)
    {
        return;
    }
    float _S443 = il_t_in_1(b_27, kernelContext_70);
    uint _S444 = il_entry_1(b_27, kernelContext_70);
    thread float3 contact_14;
    thread float3 d_16;
    thread float speed_12;
    il_path_1(b_27, &contact_14, &d_16, &speed_12, kernelContext_70);
    IlStriker_0 _S445 = il_striker_1(b_27, kernelContext_70);
    float _S446 = max(_S445.radius_0, 0.00100000004749745f);
    if((_S445.launch_0) != 0U)
    {
        uint _S447 = w_marks_0(kernelContext_70);
        damage_1 = (*(kernelContext_70->work_0+(_S447 + _S444))) == (_S445.launch_0);
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        uint _S448 = lp_body_0(b_27, kernelContext_70);
        packed_float4 _S449 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_70->lp_0+_S448) = _S449;
        *(kernelContext_70->lp_0+(_S448 + 1U)) = _S449;
        return;
    }
    uint _S450 = a_il_0(b_27, kernelContext_70);
    uint _S451 = atomic_load_explicit(kernelContext_70->atom_0+_S450, memory_order_relaxed);
    float _S452 = max((as_type<float>((_S451))) - _S446, 9.99999997475242708e-07f);
    if(!(_S443 > 0.0f))
    {
        uint _S453 = lp_body_0(b_27, kernelContext_70);
        *(kernelContext_70->lp_0+_S453) = packed_float4(float4(_S452, 0.0f, 0.0f, 0.0f)) ;
        *(kernelContext_70->lp_0+(_S453 + 1U)) = packed_float4(float4(0.0f) ) ;
        return;
    }
    uint o_3 = kernelContext_70->params_0->ext_chunks_0 + 4U * _S444;
    float4 _S454 = float4(*(kernelContext_70->ext_0+(o_3 + 1U))) ;
    thread JointMaterial_0 dm_1;
    float4 _S455 = float4(0.0f) ;
    (&dm_1)->strength_0 = _S455;
    (&dm_1)->energy_0 = _S455;
    (&dm_1)->dif_0 = float4(*(kernelContext_70->ext_0+(o_3 + 2U))) ;
    float4 _S456 = float4(*(kernelContext_70->ext_0+(o_3 + 3U))) ;
    float _S457 = _S456.x;
    (&dm_1)->misc_0 = float4(_S457, 0.0f, 0.0f, 0.0f);
    (&dm_1)->kind_flags_0 = uint4(0U) ;
    float nu_1 = _S456.y;
    float e_target_1 = _S456.z / (1.0f - nu_1 * nu_1);
    float e_star_1 = 1.0f / (1.0f / _S445.modulus_0 + 1.0f / e_target_1);
    float _S458 = max(speed_12, 0.00100000004749745f);
    float _S459 = sqrt(_S446);
    float h0_1 = pow(15.0f * _S445.m_eff_0 * _S458 * _S458 / (16.0f * e_star_1 * _S459), 0.40000000596046448f);
    float t_c_1 = 2.94000005722045898f * h0_1 / _S458;
    float force_1 = 1.33333337306976318f * e_star_1 * _S459 * pow(h0_1, 1.5f);
    float a_4 = sqrt(_S446 * h0_1);
    float p0_1 = 3.0f * force_1 / (6.28318548202514648f * a_4 * a_4);
    float ring_1 = (1.0f - 2.0f * nu_1) * p0_1 / 3.0f;
    float k_19;
    if(_S457 > 0.0f)
    {
        float _S460 = _S458 / _S452;
        thread JointMaterial_0 _S461 = dm_1;
        float _S462 = dif_factor_0(&_S461, _S460);
        k_19 = _S462;
    }
    else
    {
        k_19 = 1.0f;
    }
    float ft_1 = k_19 * _S454.y;
    float fc_1 = k_19 * _S454.z;
    float tau_1 = k_19 * _S454.w;
    if(p0_1 > fc_1)
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = ring_1 > ft_1;
    }
    float _S463 = fc_1 * (3.14159274101257324f * _S446 * _S446);
    float _S464 = min(_S463 * _S452, 1.57079637050628662f * (2.0f * _S446 + _S452) * _S452 * _S452 * tau_1);
    float _S465 = 2.0f * _S464;
    float v_bl_1 = sqrt(_S465 / _S445.m_eff_0);
    bool perforate_1;
    if(damage_1)
    {
        perforate_1 = _S458 >= v_bl_1;
    }
    else
    {
        perforate_1 = false;
    }
    float r2_1 = _S446 + _S452;
    float _S466 = _S454.x;
    float plug_1 = _S466 * 3.14159274101257324f * _S452 / 3.0f * (_S446 * _S446 + _S446 * r2_1 + r2_1 * r2_1);
    float v_res_1;
    if(perforate_1)
    {
        v_res_1 = _S445.mass_0 / (_S445.mass_0 + plug_1) * sqrt(max(_S458 * _S458 - v_bl_1 * v_bl_1, 0.0f));
    }
    else
    {
        v_res_1 = 0.0f;
    }
    bool spall_1;
    if(damage_1)
    {
        spall_1 = !perforate_1;
    }
    else
    {
        spall_1 = false;
    }
    float depth_2;
    if(spall_1)
    {
        depth_2 = min(0.5f * _S445.m_eff_0 * _S458 * _S458 / _S463, _S452);
    }
    else
    {
        depth_2 = 0.0f;
    }
    bool crater_1 = depth_2 > a_4;
    float c_5 = sqrt(e_target_1 / _S466);
    float z_t_1 = _S466 * c_5;
    float rho_p_1 = _S445.mass_0 / (4.18879032135009766f * _S446 * _S446 * _S446);
    float z_p_1 = rho_p_1 * sqrt(_S445.modulus_0 / rho_p_1);
    float mean_pressure_1 = force_1 / (3.14159274101257324f * a_4 * a_4);
    float _S467 = min(min(z_t_1 * _S458 * z_p_1 / (z_p_1 + z_t_1), mean_pressure_1), fc_1);
    float source_radius_1;
    if(mean_pressure_1 > fc_1)
    {
        source_radius_1 = max(a_4, _S446);
    }
    else
    {
        source_radius_1 = a_4;
    }
    float pulse_1 = c_5 * t_c_1;
    float back_1 = _S467 * min(source_radius_1 / _S452, 1.0f) * min(2.0f * _S452 / pulse_1, 1.0f);
    if(damage_1)
    {
        spall_1 = !perforate_1;
    }
    else
    {
        spall_1 = false;
    }
    if(spall_1)
    {
        spall_1 = back_1 > ft_1;
    }
    else
    {
        spall_1 = false;
    }
    float scab_2;
    if(spall_1)
    {
        scab_2 = 0.5f * min(pulse_1, _S452);
    }
    else
    {
        scab_2 = 0.0f;
    }
    uint region_4;
    if(damage_1)
    {
        region_4 = 1U;
    }
    else
    {
        region_4 = 0U;
    }
    uint _S468;
    if(crater_1)
    {
        _S468 = 2U;
    }
    else
    {
        _S468 = 0U;
    }
    uint _S469 = region_4 | _S468;
    if(spall_1)
    {
        region_4 = 4U;
    }
    else
    {
        region_4 = 0U;
    }
    uint _S470 = _S469 | region_4;
    if(perforate_1)
    {
        region_4 = 8U;
    }
    else
    {
        region_4 = 0U;
    }
    uint outcome_8 = _S470 | region_4;
    uint _S471 = lp_body_0(b_27, kernelContext_70);
    *(kernelContext_70->lp_0+_S471) = packed_float4(float4(_S452, v_bl_1, v_res_1, (as_type<float>((outcome_8))))) ;
    packed_float4 device* _S472 = kernelContext_70->lp_0+(_S471 + 1U);
    if(crater_1)
    {
        k_19 = depth_2;
    }
    else
    {
        k_19 = 0.0f;
    }
    *_S472 = packed_float4(float4(k_19, scab_2, a_4, _S464)) ;
    if(perforate_1)
    {
        uint _S473 = a_counters_1(kernelContext_70);
        uint _S474 = atomic_fetch_add_explicit(kernelContext_70->atom_0+(_S473 + 10U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_1)
        {
            uint _S475 = a_counters_1(kernelContext_70);
            uint _S476 = atomic_fetch_add_explicit(kernelContext_70->atom_0+(_S475 + 9U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_1)
            {
                uint _S477 = a_counters_1(kernelContext_70);
                uint _S478 = atomic_fetch_add_explicit(kernelContext_70->atom_0+(_S477 + 8U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S479 = a_counters_1(kernelContext_70);
                uint _S480 = atomic_fetch_add_explicit(kernelContext_70->atom_0+(_S479 + 7U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_1)
    {
        if((kernelContext_70->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_27, _S445.body_0, _S445.mass_0, contact_14, d_16, _S445.m_eff_0 * (_S458 - sqrt(max(_S458 * _S458 - _S465 / _S445.m_eff_0, 0.0f))), kernelContext_70);
        }
    }
    else
    {
        if(damage_1)
        {
            il_impulse_0(b_27, _S445.body_0, _S445.mass_0, contact_14, d_16, _S445.m_eff_0 * _S458, kernelContext_70);
            float3 _S481 = to_rubble_0(d_16) * float3((_S458 * clamp(_S443, 0.0f, kernelContext_70->params_0->dt_0))) ;
            region_4 = 0U;
            for(;;)
            {
                if(region_4 < 2U)
                {
                }
                else
                {
                    break;
                }
                uint _S482 = 7U * (_S445.body_0 + region_4 * kernelContext_70->params_0->max_bodies_0);
                (*(kernelContext_70->bodies_0+_S482)).xyz = (float4(*(kernelContext_70->bodies_0+_S482)) ).xyz + _S481;
                region_4 = region_4 + 1U;
            }
        }
    }
    return;
}

void il_run_1(uint b_28, uint i_4, KernelContext_0 thread* kernelContext_71)
{
    bool _S483 = il_struck_1(b_28, kernelContext_71);
    uint j_2;
    if(_S483)
    {
        j_2 = 1U;
    }
    else
    {
        j_2 = 0U;
    }
    if((group_uniform_0(j_2, i_4)) == 0U)
    {
        return;
    }
    IlStriker_0 _S484 = il_striker_1(b_28, kernelContext_71);
    float _S485 = 0.00100000004749745f + 0.01999999955296516f * _S484.radius_0;
    uint _S486 = a_il_0(b_28, kernelContext_71);
    uint _S487 = atomic_load_explicit(kernelContext_71->atom_0+(_S486 + 4U), memory_order_relaxed);
    uint _S488 = atomic_load_explicit(kernelContext_71->atom_0+(_S486 + 1U), memory_order_relaxed);
    uint _S489 = atomic_load_explicit(kernelContext_71->atom_0+_S486, memory_order_relaxed);
    float end_1 = group_uniform_float_0((as_type<float>((_S489))), i_4);
    for(;;)
    {
        float best_1 = end_1;
        j_2 = i_4;
        for(;;)
        {
            if(j_2 < _S488)
            {
            }
            else
            {
                break;
            }
            uint _S490 = w_il_tube_0(kernelContext_71);
            uint at_6 = _S490 + 2U * (_S487 + j_2);
            if((as_type<float>((*(kernelContext_71->work_0+at_6)))) <= (end_1 + _S485))
            {
                best_1 = max(best_1, (as_type<float>((*(kernelContext_71->work_0+(at_6 + 1U))))));
            }
            j_2 = j_2 + 64U;
        }
        (*kernelContext_71->g_il_best_0)[i_4] = best_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint step_2 = 32U;
        for(;;)
        {
            if(step_2 > 0U)
            {
            }
            else
            {
                break;
            }
            if(i_4 < step_2)
            {
                (*kernelContext_71->g_il_best_0)[i_4] = max((*kernelContext_71->g_il_best_0)[i_4], (*kernelContext_71->g_il_best_0)[i_4 + step_2]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_2 = step_2 >> 1U;
        }
        float next_1 = group_uniform_float_0((*kernelContext_71->g_il_best_0)[int(0)], i_4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!(next_1 > end_1))
        {
            break;
        }
        end_1 = next_1;
    }
    if(i_4 == 0U)
    {
        atomic_store_explicit(kernelContext_71->atom_0+_S486, (as_type<uint>((end_1))), memory_order_relaxed);
    }
    return;
}

void il_tube_collect_1(uint k_20, KernelContext_0 thread* kernelContext_72)
{
    thread uint b_29;
    thread float2 t_5;
    bool _S491 = il_tube_1(k_20, &b_29, &t_5, kernelContext_72);
    if(!_S491)
    {
        return;
    }
    uint _S492 = w_il_tube_0(kernelContext_72);
    uint _S493 = a_il_0(b_29, kernelContext_72);
    uint _S494 = atomic_load_explicit(kernelContext_72->atom_0+(_S493 + 4U), memory_order_relaxed);
    uint _S495 = a_il_0(b_29, kernelContext_72);
    uint _S496 = atomic_fetch_add_explicit(kernelContext_72->atom_0+(_S495 + 5U), 1U, memory_order_relaxed);
    uint at_7 = _S492 + 2U * (_S494 + _S496);
    *(kernelContext_72->work_0+at_7) = (as_type<uint>((t_5.x)));
    *(kernelContext_72->work_0+(at_7 + 1U)) = (as_type<uint>((t_5.y)));
    return;
}

void il_tube_base_1(uint b_30, KernelContext_0 thread* kernelContext_73)
{
    bool _S497;
    if(b_30 < (kernelContext_73->params_0->body_count_0))
    {
        bool _S498 = il_struck_1(b_30, kernelContext_73);
        _S497 = _S498;
    }
    else
    {
        _S497 = false;
    }
    if(_S497)
    {
        uint _S499 = a_il_0(b_30, kernelContext_73);
        atomic<uint> device* _S500 = kernelContext_73->atom_0+(_S499 + 4U);
        uint _S501 = a_il_cursor_1(kernelContext_73);
        atomic<uint> device* _S502 = kernelContext_73->atom_0+_S501;
        uint _S503 = atomic_load_explicit(kernelContext_73->atom_0+(_S499 + 1U), memory_order_relaxed);
        uint _S504 = atomic_fetch_add_explicit(_S502, _S503, memory_order_relaxed);
        atomic_store_explicit(_S500, _S504, memory_order_relaxed);
    }
    return;
}

void il_tube_count_1(uint k_21, KernelContext_0 thread* kernelContext_74)
{
    thread uint b_31;
    thread float2 t_6;
    bool _S505 = il_tube_1(k_21, &b_31, &t_6, kernelContext_74);
    if(_S505)
    {
        uint _S506 = a_il_0(b_31, kernelContext_74);
        uint _S507 = atomic_fetch_add_explicit(kernelContext_74->atom_0+(_S506 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_first_1(uint shot_3, KernelContext_0 thread* kernelContext_75)
{
    uint _S508 = a_strike_0(kernelContext_75);
    return _S508 + 29U * kernelContext_75->params_0->body_count_0 + shot_3;
}

void il_begin_1(uint b_32, KernelContext_0 thread* kernelContext_76)
{
    bool _S509;
    if(b_32 >= (kernelContext_76->params_0->body_count_0))
    {
        _S509 = true;
    }
    else
    {
        bool _S510 = il_struck_1(b_32, kernelContext_76);
        _S509 = !_S510;
    }
    if(_S509)
    {
        return;
    }
    thread uint shot_4;
    bool _S511 = struck_0(b_32, &shot_4, kernelContext_76);
    uint _S512 = a_strike_0(kernelContext_76);
    uint _S513 = atomic_load_explicit(kernelContext_76->atom_0+(_S512 + 9U * b_32 + 6U), memory_order_relaxed);
    uint _S514 = a_first_1(shot_4, kernelContext_76);
    uint _S515 = atomic_load_explicit(kernelContext_76->atom_0+_S514, memory_order_relaxed);
    if(_S513 != _S515)
    {
        uint _S516 = strike_at_0(b_32, kernelContext_76);
        (*(kernelContext_76->frames_0+_S516))[int(3)] = 0.0f;
        return;
    }
    thread float3 contact_15;
    thread float3 d_17;
    thread float speed_13;
    il_path_1(b_32, &contact_15, &d_17, &speed_13, kernelContext_76);
    uint _S517 = il_entry_1(b_32, kernelContext_76);
    IlStriker_0 _S518 = il_striker_1(b_32, kernelContext_76);
    float2 _S519 = il_interval_0(_S517, contact_15, d_17, _S518.radius_0, kernelContext_76);
    uint _S520 = a_il_0(b_32, kernelContext_76);
    atomic_store_explicit(kernelContext_76->atom_0+_S520, (as_type<uint>((max(_S519.y, 0.0f)))), memory_order_relaxed);
    uint i_5 = 1U;
    for(;;)
    {
        if(i_5 < 6U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit(kernelContext_76->atom_0+(_S520 + i_5), 0U, memory_order_relaxed);
        i_5 = i_5 + 1U;
    }
    return;
}

uint l_pool_0(uint half_1, KernelContext_0 thread* kernelContext_77)
{
    return kernelContext_77->params_0->chunk_count_0 + 3U * kernelContext_77->params_0->pool_cap_0 * half_1;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S521 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S521, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S522 = max(0.0f, _S521);
    float _S523 = min(dt_1, _S521 + duration_0);
    bool _S524;
    if(_S523 > _S522)
    {
        _S524 = duration_0 > 0.0f;
    }
    else
    {
        _S524 = false;
    }
    float sum_1;
    if(_S524)
    {
        float w_2 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S522 - _S521)) - cos(w_2 * (_S523 - _S521)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

float hertz_duration_0(float m_0, float r_18, float e_2, float v_9)
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_18 * e_2 * e_2 * max(abs(v_9), 0.00100000004749745f)), 0.20000000298023224f);
}

uint l_vec_0(KernelContext_0 thread* kernelContext_78)
{
    return kernelContext_78->params_0->chunk_count_0 + 6U * kernelContext_78->params_0->pool_cap_0;
}

[[kernel]] void fr_loads(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], uint device* sort_1 [[buffer(11)]], atomic<uint> device* atom_1 [[buffer(5)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], packed_float4 device* loads_1 [[buffer(12)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]], packed_float4 device* view_chunks_1 [[buffer(17)]], ViewParams_0 constant* view_1 [[buffer(16)]], packed_float4 device* objects_1 [[buffer(18)]], uint device* shot_bodies_1 [[buffer(20)]], float device* lines_1 [[buffer(19)]], packed_float4 device* geometry_1 [[buffer(21)]], packed_float4 device* shots_1 [[buffer(22)]], packed_float4 device* lp_1 [[buffer(24)]], packed_float4 device* ext_2 [[buffer(23)]], packed_float4 device* lp_sides_1 [[buffer(28)]], uint device* engine_counters_1 [[buffer(26)]], packed_float4 device* engine_contacts_1 [[buffer(25)]], packed_float4 device* cs_anchors_1 [[buffer(27)]])
{
    float3 f_total_0;
    float3 t_total_0;
    float m_1;
    uint p_10;
    uint kept_0;
    uint q3_0;
    bool overflow_0;
    bool overflow_1;
    float3 f_total_1;
    float3 t_total_1;
    thread KernelContext_0 kernelContext_79;
    (&kernelContext_79)->params_0 = params_1;
    (&kernelContext_79)->work_0 = work_1;
    (&kernelContext_79)->frames_0 = frames_1;
    (&kernelContext_79)->bodies_0 = bodies_1;
    (&kernelContext_79)->chunks_0 = chunks_1;
    (&kernelContext_79)->sort_0 = sort_1;
    (&kernelContext_79)->atom_0 = atom_1;
    (&kernelContext_79)->manifolds_0 = manifolds_1;
    (&kernelContext_79)->loads_0 = loads_1;
    (&kernelContext_79)->bonds_0 = bonds_1;
    (&kernelContext_79)->args_0 = args_1;
    (&kernelContext_79)->info_0 = info_1;
    (&kernelContext_79)->scan_0 = scan_1;
    (&kernelContext_79)->parts_0 = parts_1;
    (&kernelContext_79)->colliders_0 = colliders_1;
    (&kernelContext_79)->collider_info_0 = collider_info_1;
    (&kernelContext_79)->view_chunks_0 = view_chunks_1;
    (&kernelContext_79)->view_0 = view_1;
    (&kernelContext_79)->objects_0 = objects_1;
    (&kernelContext_79)->shot_bodies_0 = shot_bodies_1;
    (&kernelContext_79)->lines_0 = lines_1;
    (&kernelContext_79)->geometry_0 = geometry_1;
    (&kernelContext_79)->shots_0 = shots_1;
    (&kernelContext_79)->lp_0 = lp_1;
    (&kernelContext_79)->ext_0 = ext_2;
    (&kernelContext_79)->lp_sides_0 = lp_sides_1;
    (&kernelContext_79)->engine_counters_0 = engine_counters_1;
    (&kernelContext_79)->engine_contacts_0 = engine_contacts_1;
    (&kernelContext_79)->cs_anchors_0 = cs_anchors_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_79)->g_scan_0 = &g_scan_1;
    threadgroup array<float, int(64)> g_il_best_1;
    (&kernelContext_79)->g_il_best_0 = &g_il_best_1;
    uint k_22 = id_0.x;
    if(k_22 >= (params_1->chunk_count_0))
    {
        return;
    }
    uint _S525 = body_of_0(k_22, &kernelContext_79);
    uint b_33 = 7U * _S525;
    thread Quat_0 q_11;
    thread float3 origin_3;
    structure_pose_0(_S525, b_33, &q_11, &origin_3, &kernelContext_79);
    uint _S526 = 4U * k_22;
    float4 _S527 = float4(*((&kernelContext_79)->chunks_0+_S526)) ;
    float3 _S528 = _S527.xyz;
    thread Quat_0 _S529 = q_11;
    float3 _S530 = rotate_0(&_S529, _S528);
    float3 centre_0 = origin_3 + _S530;
    float inv_mass_2 = (float4(*((&kernelContext_79)->bodies_0+b_33)) ).w;
    float m_body_0;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint from_0 = (&kernelContext_79)->params_0->parity_0;
    uint to_0 = 1U - (&kernelContext_79)->params_0->parity_0;
    uint _S531 = w_pulses_0(&kernelContext_79);
    uint _S532 = _S531 + _S526;
    uint _S533 = _S532 + 2U * from_0;
    uint off_0 = *((&kernelContext_79)->work_0+_S533);
    uint device* _S534 = (&kernelContext_79)->work_0+(_S533 + 1U);
    uint carried_0 = *_S534;
    uint room_0 = *_S534 + (&kernelContext_79)->sort_0[params_1->chunk_count_0 + 1U + k_22];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S535 = a_counters_1(&kernelContext_79);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_79)->atom_0+(_S535 + 5U + to_0), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_79)->params_0->pool_cap_0))
        {
            uint _S536 = atomic_fetch_or_explicit((&kernelContext_79)->atom_0+(_S535 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_79)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_79)->params_0->pool_cap_0 - base_1;
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
        uint _S537 = base_0;
        base_0 = base_1;
        room_1 = _S537;
    }
    else
    {
        base_0 = 0U;
        room_1 = room_0;
    }
    uint _S538 = params_1->chunk_count_0;
    float _S539 = _S527.w;
    float4 _S540 = float4(*((&kernelContext_79)->chunks_0+(_S526 + 3U))) ;
    float _S541 = (&kernelContext_79)->params_0->band_0;
    float _S542 = (&kernelContext_79)->params_0->dt_0;
    bool _S543 = ((&kernelContext_79)->params_0->impact_peak_0) != 0U;
    uint _S544 = l_pool_0(from_0, &kernelContext_79);
    uint _S545 = _S544 + 3U * off_0;
    uint _S546 = l_pool_0(to_0, &kernelContext_79);
    uint _S547 = _S546 + 3U * base_0;
    for(;;)
    {
        float3 _S548 = float3(0.0f) ;
        uint _S549 = (&kernelContext_79)->sort_0[k_22];
        uint _S550 = (&kernelContext_79)->sort_0[_S538 + 1U + k_22];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_3 = 0U;
        f_total_0 = _S548;
        t_total_0 = _S548;
        for(;;)
        {
            if(step_3 < _S550)
            {
            }
            else
            {
                break;
            }
            p_10 = 4294967295U;
            kept_0 = 0U;
            q3_0 = 0U;
            for(;;)
            {
                if(q3_0 < _S550)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_79)->manifolds_0[_S549 + q3_0].ids_0) ).y + 1U;
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
                    if(p_10 == 4294967295U)
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
                    p_10 = q3_0;
                    kept_0 = key_1;
                }
                q3_0 = q3_0 + 1U;
            }
            if(p_10 == 4294967295U)
            {
                break;
            }
            Record_natural_0 r_19 = (&kernelContext_79)->manifolds_0[_S549 + p_10];
            float4 _S551 = float4(r_19.impulse_0) ;
            float3 _S552 = _S551.xyz;
            float j_3 = length(_S552);
            if(j_3 == 0.0f)
            {
                uint _S553 = step_3 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_3 = _S553;
                continue;
            }
            float m_other_0 = (float4(r_19.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S539;
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
            float _S554 = min(j_3, 2.0f * m_1 * max(_S551.w - _S541, 0.0f));
            if(_S554 < j_3)
            {
                float3 f_1 = _S552 / float3(j_3)  * float3(((j_3 - _S554) / _S542)) ;
                float3 _S555 = t_total_0 + cross((float4(r_19.point_0) ).xyz - centre_0, f_1);
                f_total_1 = f_total_0 + f_1;
                t_total_1 = _S555;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            uint _S553 = step_3 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_3 = _S553;
        }
        p_10 = 0U;
        kept_0 = 0U;
        for(;;)
        {
            if(p_10 < carried_0)
            {
            }
            else
            {
                break;
            }
            uint _S556 = _S545 + 3U * p_10;
            float4 _S557 = float4(*((&kernelContext_79)->loads_0+_S556)) ;
            thread float4 a_5 = _S557;
            float4 _S558 = float4(*((&kernelContext_79)->loads_0+(_S556 + 1U))) ;
            float4 _S559 = float4(*((&kernelContext_79)->loads_0+(_S556 + 2U))) ;
            float _S560 = _S558.w;
            float mean_0 = pulse_mean_0(_S557.x, _S557.y, _S557.z, _S557.w, _S560, _S542);
            if(mean_0 != 0.0f)
            {
                float3 f_2 = _S559.xyz * float3(mean_0) ;
                float3 _S561 = t_total_0 + cross(_S558.xyz - centre_0, f_2);
                f_total_0 = f_total_0 + f_2;
                t_total_0 = _S561;
            }
            if((a_5.x + a_5.z + _S560) > _S542)
            {
                a_5.x = a_5.x - _S542;
                uint _S562 = _S547 + 3U * kept_0;
                *((&kernelContext_79)->loads_0+_S562) = packed_float4(a_5) ;
                *((&kernelContext_79)->loads_0+(_S562 + 1U)) = packed_float4(_S558) ;
                *((&kernelContext_79)->loads_0+(_S562 + 2U)) = packed_float4(_S559) ;
                kept_0 = kept_0 + 1U;
            }
            p_10 = p_10 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S550)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_20 = (&kernelContext_79)->manifolds_0[_S549 + q3_0];
            float4 _S563 = float4(r_20.impulse_0) ;
            float3 _S564 = _S563.xyz;
            float j_4 = length(_S564);
            if(j_4 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S565 = float4(r_20.other_0) ;
            float m_other_1 = _S565.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S539;
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
            float _S566 = _S563.w;
            float _S567 = min(j_4, 2.0f * m_1 * max(_S566 - _S541, 0.0f));
            if(_S567 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S568 = _S540.y;
            float duration_1 = hertz_duration_0(m_1, min(_S565.z, _S540.z), 1.0f / ((1.0f - _S568 * _S568) / _S540.x + 1.0f / _S565.y), _S566);
            float peak_1 = 3.14159274101257324f * _S567 / (2.0f * duration_1);
            float3 dir_0 = _S564 / float3(j_4) ;
            float mean_1;
            if(_S543)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S542);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_3 = dir_0 * float3(mean_1) ;
                float3 _S569 = t_total_0 + cross((float4(r_20.point_0) ).xyz - centre_0, f_3);
                f_total_1 = f_total_0 + f_3;
                t_total_1 = _S569;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            uint kept_1;
            if(duration_1 > _S542)
            {
                if(kept_0 < room_1)
                {
                    uint _S570 = _S547 + 3U * kept_0;
                    *((&kernelContext_79)->loads_0+_S570) = packed_float4(float4(- _S542, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_79)->loads_0+(_S570 + 1U)) = packed_float4(float4((float4(r_20.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_79)->loads_0+(_S570 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
        uint _S571 = a_counters_1(&kernelContext_79);
        uint _S572 = atomic_fetch_or_explicit((&kernelContext_79)->atom_0+(_S571 + 3U), 8U, memory_order_relaxed);
    }
    uint _S573 = _S532 + 2U * to_0;
    *((&kernelContext_79)->work_0+_S573) = base_0;
    *((&kernelContext_79)->work_0+(_S573 + 1U)) = kept_0;
    *((&kernelContext_79)->loads_0+k_22) = packed_float4(float4(length(f_total_0), 0.0f, 0.0f, 0.0f)) ;
    if(((&kernelContext_79)->params_0->load_model_0) == 1U)
    {
        uint _S574 = l_vec_0(&kernelContext_79);
        uint _S575 = _S574 + 2U * k_22;
        *((&kernelContext_79)->loads_0+_S575) = packed_float4(float4(f_total_0, 0.0f)) ;
        *((&kernelContext_79)->loads_0+(_S575 + 1U)) = packed_float4(float4(t_total_0, 0.0f)) ;
    }
    return;
}

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], atomic<uint> device* atom_2 [[buffer(5)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]], packed_float4 device* geometry_2 [[buffer(21)]], packed_float4 device* shots_2 [[buffer(22)]], packed_float4 device* lp_2 [[buffer(24)]], packed_float4 device* ext_3 [[buffer(23)]], packed_float4 device* lp_sides_2 [[buffer(28)]], uint device* engine_counters_2 [[buffer(26)]], packed_float4 device* engine_contacts_2 [[buffer(25)]], packed_float4 device* cs_anchors_2 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_80;
    (&kernelContext_80)->params_0 = params_2;
    (&kernelContext_80)->work_0 = work_2;
    (&kernelContext_80)->frames_0 = frames_2;
    (&kernelContext_80)->bodies_0 = bodies_2;
    (&kernelContext_80)->chunks_0 = chunks_2;
    (&kernelContext_80)->sort_0 = sort_2;
    (&kernelContext_80)->atom_0 = atom_2;
    (&kernelContext_80)->manifolds_0 = manifolds_2;
    (&kernelContext_80)->loads_0 = loads_2;
    (&kernelContext_80)->bonds_0 = bonds_2;
    (&kernelContext_80)->args_0 = args_2;
    (&kernelContext_80)->info_0 = info_2;
    (&kernelContext_80)->scan_0 = scan_2;
    (&kernelContext_80)->parts_0 = parts_2;
    (&kernelContext_80)->colliders_0 = colliders_2;
    (&kernelContext_80)->collider_info_0 = collider_info_2;
    (&kernelContext_80)->view_chunks_0 = view_chunks_2;
    (&kernelContext_80)->view_0 = view_2;
    (&kernelContext_80)->objects_0 = objects_2;
    (&kernelContext_80)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_80)->lines_0 = lines_2;
    (&kernelContext_80)->geometry_0 = geometry_2;
    (&kernelContext_80)->shots_0 = shots_2;
    (&kernelContext_80)->lp_0 = lp_2;
    (&kernelContext_80)->ext_0 = ext_3;
    (&kernelContext_80)->lp_sides_0 = lp_sides_2;
    (&kernelContext_80)->engine_counters_0 = engine_counters_2;
    (&kernelContext_80)->engine_contacts_0 = engine_contacts_2;
    (&kernelContext_80)->cs_anchors_0 = cs_anchors_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_80)->g_scan_0 = &g_scan_2;
    threadgroup array<float, int(64)> g_il_best_2;
    (&kernelContext_80)->g_il_best_0 = &g_il_best_2;
    uint i_6 = id_1.x;
    bool _S576;
    if(i_6 >= (params_2->bond_count_0))
    {
        _S576 = true;
    }
    else
    {
        uint _S577 = w_alive_0(&kernelContext_80);
        _S576 = (*((&kernelContext_80)->work_0+(_S577 + i_6))) == 0U;
    }
    if(_S576)
    {
        return;
    }
    uint4 _S578 = uint4(*((&kernelContext_80)->bonds_0+i_6)) ;
    uint _S579 = _S578.x;
    if((max((float4(*((&kernelContext_80)->loads_0+_S579)) ).x, (float4(*((&kernelContext_80)->loads_0+_S578.y)) ).x)) > (as_type<float>((_S578.z))))
    {
        uint _S580 = w_alive_0(&kernelContext_80);
        *((&kernelContext_80)->work_0+(_S580 + i_6)) = 0U;
        uint _S581 = a_dirty_0(&kernelContext_80);
        uint _S582 = body_of_0(_S579, &kernelContext_80);
        uint _S583 = atomic_fetch_or_explicit((&kernelContext_80)->atom_0+(_S581 + _S582), 1U, memory_order_relaxed);
        uint _S584 = a_counters_1(&kernelContext_80);
        uint _S585 = atomic_fetch_add_explicit((&kernelContext_80)->atom_0+_S584, 1U, memory_order_relaxed);
        uint _S586 = atomic_fetch_add_explicit((&kernelContext_80)->atom_0+(_S584 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint groups_of_0(uint n_3, uint size_0)
{
    uint _S587 = (n_3 + size_0 - 1U) / size_0;
    return _S587;
}

void set_args_0(uint at_8, uint x_3, KernelContext_0 thread* kernelContext_81)
{
    uint _S588 = 3U * at_8;
    *(kernelContext_81->args_0+_S588) = x_3;
    *(kernelContext_81->args_0+(_S588 + 1U)) = 1U;
    *(kernelContext_81->args_0+(_S588 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], packed_float4 device* lp_3 [[buffer(24)]], packed_float4 device* ext_4 [[buffer(23)]], packed_float4 device* lp_sides_3 [[buffer(28)]], uint device* engine_counters_3 [[buffer(26)]], packed_float4 device* engine_contacts_3 [[buffer(25)]], packed_float4 device* cs_anchors_3 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_82;
    (&kernelContext_82)->params_0 = params_3;
    (&kernelContext_82)->work_0 = work_3;
    (&kernelContext_82)->frames_0 = frames_3;
    (&kernelContext_82)->bodies_0 = bodies_3;
    (&kernelContext_82)->chunks_0 = chunks_3;
    (&kernelContext_82)->sort_0 = sort_3;
    (&kernelContext_82)->atom_0 = atom_3;
    (&kernelContext_82)->manifolds_0 = manifolds_3;
    (&kernelContext_82)->loads_0 = loads_3;
    (&kernelContext_82)->bonds_0 = bonds_3;
    (&kernelContext_82)->args_0 = args_3;
    (&kernelContext_82)->info_0 = info_3;
    (&kernelContext_82)->scan_0 = scan_3;
    (&kernelContext_82)->parts_0 = parts_3;
    (&kernelContext_82)->colliders_0 = colliders_3;
    (&kernelContext_82)->collider_info_0 = collider_info_3;
    (&kernelContext_82)->view_chunks_0 = view_chunks_3;
    (&kernelContext_82)->view_0 = view_3;
    (&kernelContext_82)->objects_0 = objects_3;
    (&kernelContext_82)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_82)->lines_0 = lines_3;
    (&kernelContext_82)->geometry_0 = geometry_3;
    (&kernelContext_82)->shots_0 = shots_3;
    (&kernelContext_82)->lp_0 = lp_3;
    (&kernelContext_82)->ext_0 = ext_4;
    (&kernelContext_82)->lp_sides_0 = lp_sides_3;
    (&kernelContext_82)->engine_counters_0 = engine_counters_3;
    (&kernelContext_82)->engine_contacts_0 = engine_contacts_3;
    (&kernelContext_82)->cs_anchors_0 = cs_anchors_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_82)->g_scan_0 = &g_scan_3;
    threadgroup array<float, int(64)> g_il_best_3;
    (&kernelContext_82)->g_il_best_0 = &g_il_best_3;
    uint _S589 = a_counters_1(&kernelContext_82);
    uint _S590 = atomic_load_explicit((&kernelContext_82)->atom_0+_S589, memory_order_relaxed);
    bool run_0 = _S590 != 0U;
    atomic_store_explicit((&kernelContext_82)->atom_0+_S589, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_82)->atom_0+(_S589 + 5U + (&kernelContext_82)->params_0->parity_0), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S591 = atomic_fetch_add_explicit((&kernelContext_82)->atom_0+(_S589 + 4U), 1U, memory_order_relaxed);
    }
    uint _S592;
    if(run_0)
    {
        uint _S593 = groups_of_0((&kernelContext_82)->params_0->chunk_count_0, 64U);
        _S592 = _S593;
    }
    else
    {
        _S592 = 0U;
    }
    set_args_0(0U, _S592, &kernelContext_82);
    if(run_0)
    {
        uint _S594 = groups_of_0((&kernelContext_82)->params_0->bond_count_0, 64U);
        _S592 = _S594;
    }
    else
    {
        _S592 = 0U;
    }
    set_args_0(1U, _S592, &kernelContext_82);
    if(run_0)
    {
        uint _S595 = groups_of_0((&kernelContext_82)->params_0->body_count_0, 64U);
        _S592 = _S595;
    }
    else
    {
        _S592 = 0U;
    }
    set_args_0(2U, _S592, &kernelContext_82);
    if(run_0)
    {
        uint _S596 = groups_of_0((&kernelContext_82)->params_0->chunk_count_0, 1024U);
        _S592 = _S596;
    }
    else
    {
        _S592 = 0U;
    }
    set_args_0(3U, _S592, &kernelContext_82);
    if(run_0)
    {
        uint _S597 = groups_of_0((&kernelContext_82)->params_0->body_count_0, 1024U);
        _S592 = _S597;
    }
    else
    {
        _S592 = 0U;
    }
    set_args_0(4U, _S592, &kernelContext_82);
    int _S598;
    if(run_0)
    {
        _S598 = int(1);
    }
    else
    {
        _S598 = int(0);
    }
    set_args_0(5U, uint(_S598), &kernelContext_82);
    return;
}

bool dirty_0(uint body_1, KernelContext_0 thread* kernelContext_83)
{
    uint _S599 = a_dirty_0(kernelContext_83);
    uint _S600 = atomic_load_explicit(kernelContext_83->atom_0+(_S599 + body_1), memory_order_relaxed);
    return _S600 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_84)
{
    return kernelContext_84->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_85)
{
    return 2U * kernelContext_85->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], packed_float4 device* lp_4 [[buffer(24)]], packed_float4 device* ext_5 [[buffer(23)]], packed_float4 device* lp_sides_4 [[buffer(28)]], uint device* engine_counters_4 [[buffer(26)]], packed_float4 device* engine_contacts_4 [[buffer(25)]], packed_float4 device* cs_anchors_4 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_86;
    (&kernelContext_86)->params_0 = params_4;
    (&kernelContext_86)->work_0 = work_4;
    (&kernelContext_86)->frames_0 = frames_4;
    (&kernelContext_86)->bodies_0 = bodies_4;
    (&kernelContext_86)->chunks_0 = chunks_4;
    (&kernelContext_86)->sort_0 = sort_4;
    (&kernelContext_86)->atom_0 = atom_4;
    (&kernelContext_86)->manifolds_0 = manifolds_4;
    (&kernelContext_86)->loads_0 = loads_4;
    (&kernelContext_86)->bonds_0 = bonds_4;
    (&kernelContext_86)->args_0 = args_4;
    (&kernelContext_86)->info_0 = info_4;
    (&kernelContext_86)->scan_0 = scan_4;
    (&kernelContext_86)->parts_0 = parts_4;
    (&kernelContext_86)->colliders_0 = colliders_4;
    (&kernelContext_86)->collider_info_0 = collider_info_4;
    (&kernelContext_86)->view_chunks_0 = view_chunks_4;
    (&kernelContext_86)->view_0 = view_4;
    (&kernelContext_86)->objects_0 = objects_4;
    (&kernelContext_86)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_86)->lines_0 = lines_4;
    (&kernelContext_86)->geometry_0 = geometry_4;
    (&kernelContext_86)->shots_0 = shots_4;
    (&kernelContext_86)->lp_0 = lp_4;
    (&kernelContext_86)->ext_0 = ext_5;
    (&kernelContext_86)->lp_sides_0 = lp_sides_4;
    (&kernelContext_86)->engine_counters_0 = engine_counters_4;
    (&kernelContext_86)->engine_contacts_0 = engine_contacts_4;
    (&kernelContext_86)->cs_anchors_0 = cs_anchors_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_86)->g_scan_0 = &g_scan_4;
    threadgroup array<float, int(64)> g_il_best_4;
    (&kernelContext_86)->g_il_best_0 = &g_il_best_4;
    uint c_6 = id_3.x;
    bool _S601;
    if(c_6 >= (params_4->chunk_count_0))
    {
        _S601 = true;
    }
    else
    {
        uint _S602 = body_of_0(c_6, &kernelContext_86);
        bool _S603 = dirty_0(_S602, &kernelContext_86);
        _S601 = !_S603;
    }
    if(_S601)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_86)->atom_0+(a_parent_0() + c_6), c_6, memory_order_relaxed);
    uint _S604 = a_count_0(&kernelContext_86);
    atomic_store_explicit((&kernelContext_86)->atom_0+(_S604 + c_6), 0U, memory_order_relaxed);
    uint _S605 = a_anchor_0(&kernelContext_86);
    atomic_store_explicit((&kernelContext_86)->atom_0+(_S605 + c_6), 0U, memory_order_relaxed);
    uint w_3 = 0U;
    for(;;)
    {
        if(w_3 < 20U)
        {
        }
        else
        {
            break;
        }
        uint _S606 = a_sums_0(&kernelContext_86);
        atomic_store_explicit((&kernelContext_86)->atom_0+(_S606 + 20U * c_6 + w_3), 0U, memory_order_relaxed);
        w_3 = w_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], packed_float4 device* lp_5 [[buffer(24)]], packed_float4 device* ext_6 [[buffer(23)]], packed_float4 device* lp_sides_5 [[buffer(28)]], uint device* engine_counters_5 [[buffer(26)]], packed_float4 device* engine_contacts_5 [[buffer(25)]], packed_float4 device* cs_anchors_5 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_87;
    (&kernelContext_87)->params_0 = params_5;
    (&kernelContext_87)->work_0 = work_5;
    (&kernelContext_87)->frames_0 = frames_5;
    (&kernelContext_87)->bodies_0 = bodies_5;
    (&kernelContext_87)->chunks_0 = chunks_5;
    (&kernelContext_87)->sort_0 = sort_5;
    (&kernelContext_87)->atom_0 = atom_5;
    (&kernelContext_87)->manifolds_0 = manifolds_5;
    (&kernelContext_87)->loads_0 = loads_5;
    (&kernelContext_87)->bonds_0 = bonds_5;
    (&kernelContext_87)->args_0 = args_5;
    (&kernelContext_87)->info_0 = info_5;
    (&kernelContext_87)->scan_0 = scan_5;
    (&kernelContext_87)->parts_0 = parts_5;
    (&kernelContext_87)->colliders_0 = colliders_5;
    (&kernelContext_87)->collider_info_0 = collider_info_5;
    (&kernelContext_87)->view_chunks_0 = view_chunks_5;
    (&kernelContext_87)->view_0 = view_5;
    (&kernelContext_87)->objects_0 = objects_5;
    (&kernelContext_87)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_87)->lines_0 = lines_5;
    (&kernelContext_87)->geometry_0 = geometry_5;
    (&kernelContext_87)->shots_0 = shots_5;
    (&kernelContext_87)->lp_0 = lp_5;
    (&kernelContext_87)->ext_0 = ext_6;
    (&kernelContext_87)->lp_sides_0 = lp_sides_5;
    (&kernelContext_87)->engine_counters_0 = engine_counters_5;
    (&kernelContext_87)->engine_contacts_0 = engine_contacts_5;
    (&kernelContext_87)->cs_anchors_0 = cs_anchors_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_87)->g_scan_0 = &g_scan_5;
    threadgroup array<float, int(64)> g_il_best_5;
    (&kernelContext_87)->g_il_best_0 = &g_il_best_5;
    uint s_7 = id_4.x;
    bool _S607;
    if(s_7 >= (params_5->body_count_0))
    {
        _S607 = true;
    }
    else
    {
        bool _S608 = dirty_0(s_7, &kernelContext_87);
        _S607 = !_S608;
    }
    if(_S607)
    {
        return;
    }
    uint _S609 = a_best_count_0(&kernelContext_87);
    atomic_store_explicit((&kernelContext_87)->atom_0+(_S609 + s_7), 0U, memory_order_relaxed);
    uint _S610 = a_best_root_0(&kernelContext_87);
    atomic_store_explicit((&kernelContext_87)->atom_0+(_S610 + s_7), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_4, KernelContext_0 thread* kernelContext_88)
{
    uint _S611 = a_parent_0();
    uint _S612 = atomic_load_explicit(kernelContext_88->atom_0+(_S611 + x_4), memory_order_relaxed);
    uint p_11 = _S612;
    uint _S613 = x_4;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S614;
        if(guard_0 < 1048576U)
        {
            _S614 = p_11 != _S613;
        }
        else
        {
            _S614 = false;
        }
        if(_S614)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_88->atom_0+(_S611 + p_11), memory_order_relaxed);
        if(gp_0 != p_11)
        {
            uint _S615;
            {
            uint _metal_cas_comparand = p_11;
            atomic_compare_exchange_weak_explicit(kernelContext_88->atom_0+(_S611 + _S613), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S615 = _metal_cas_comparand;
            }
        }
        uint _S616 = guard_0 + 1U;
        _S613 = p_11;
        p_11 = gp_0;
        guard_0 = _S616;
    }
    return _S613;
}

[[kernel]] void fr_uf_union(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], packed_float4 device* lp_6 [[buffer(24)]], packed_float4 device* ext_7 [[buffer(23)]], packed_float4 device* lp_sides_6 [[buffer(28)]], uint device* engine_counters_6 [[buffer(26)]], packed_float4 device* engine_contacts_6 [[buffer(25)]], packed_float4 device* cs_anchors_6 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_89;
    (&kernelContext_89)->params_0 = params_6;
    (&kernelContext_89)->work_0 = work_6;
    (&kernelContext_89)->frames_0 = frames_6;
    (&kernelContext_89)->bodies_0 = bodies_6;
    (&kernelContext_89)->chunks_0 = chunks_6;
    (&kernelContext_89)->sort_0 = sort_6;
    (&kernelContext_89)->atom_0 = atom_6;
    (&kernelContext_89)->manifolds_0 = manifolds_6;
    (&kernelContext_89)->loads_0 = loads_6;
    (&kernelContext_89)->bonds_0 = bonds_6;
    (&kernelContext_89)->args_0 = args_6;
    (&kernelContext_89)->info_0 = info_6;
    (&kernelContext_89)->scan_0 = scan_6;
    (&kernelContext_89)->parts_0 = parts_6;
    (&kernelContext_89)->colliders_0 = colliders_6;
    (&kernelContext_89)->collider_info_0 = collider_info_6;
    (&kernelContext_89)->view_chunks_0 = view_chunks_6;
    (&kernelContext_89)->view_0 = view_6;
    (&kernelContext_89)->objects_0 = objects_6;
    (&kernelContext_89)->shot_bodies_0 = shot_bodies_6;
    (&kernelContext_89)->lines_0 = lines_6;
    (&kernelContext_89)->geometry_0 = geometry_6;
    (&kernelContext_89)->shots_0 = shots_6;
    (&kernelContext_89)->lp_0 = lp_6;
    (&kernelContext_89)->ext_0 = ext_7;
    (&kernelContext_89)->lp_sides_0 = lp_sides_6;
    (&kernelContext_89)->engine_counters_0 = engine_counters_6;
    (&kernelContext_89)->engine_contacts_0 = engine_contacts_6;
    (&kernelContext_89)->cs_anchors_0 = cs_anchors_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_89)->g_scan_0 = &g_scan_6;
    threadgroup array<float, int(64)> g_il_best_6;
    (&kernelContext_89)->g_il_best_0 = &g_il_best_6;
    uint i_7 = id_5.x;
    bool _S617;
    if(i_7 >= (params_6->bond_count_0))
    {
        _S617 = true;
    }
    else
    {
        uint _S618 = w_alive_0(&kernelContext_89);
        _S617 = (*((&kernelContext_89)->work_0+(_S618 + i_7))) == 0U;
    }
    if(_S617)
    {
        return;
    }
    uint4 _S619 = uint4(*((&kernelContext_89)->bonds_0+i_7)) ;
    uint _S620 = _S619.x;
    uint _S621 = body_of_0(_S620, &kernelContext_89);
    bool _S622 = dirty_0(_S621, &kernelContext_89);
    if(!_S622)
    {
        return;
    }
    uint _S623 = _S619.y;
    uint a_6 = _S620;
    uint b_34 = _S623;
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
        uint _S624 = find_0(a_6, &kernelContext_89);
        uint _S625 = find_0(b_34, &kernelContext_89);
        if(_S624 == _S625)
        {
            return;
        }
        uint _S626 = max(_S624, _S625);
        uint _S627 = min(_S624, _S625);
        uint was_0;
        {
        uint _metal_cas_comparand = _S626;
        atomic_compare_exchange_weak_explicit((&kernelContext_89)->atom_0+(a_parent_0() + _S626), &_metal_cas_comparand, _S627, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S626)
        {
            return;
        }
        uint _S628 = guard_1 + 1U;
        a_6 = _S626;
        b_34 = _S627;
        guard_1 = _S628;
    }
    uint _S629 = a_counters_1(&kernelContext_89);
    uint _S630 = atomic_fetch_or_explicit((&kernelContext_89)->atom_0+(_S629 + 3U), 4U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], packed_float4 device* lp_7 [[buffer(24)]], packed_float4 device* ext_8 [[buffer(23)]], packed_float4 device* lp_sides_7 [[buffer(28)]], uint device* engine_counters_7 [[buffer(26)]], packed_float4 device* engine_contacts_7 [[buffer(25)]], packed_float4 device* cs_anchors_7 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_90;
    (&kernelContext_90)->params_0 = params_7;
    (&kernelContext_90)->work_0 = work_7;
    (&kernelContext_90)->frames_0 = frames_7;
    (&kernelContext_90)->bodies_0 = bodies_7;
    (&kernelContext_90)->chunks_0 = chunks_7;
    (&kernelContext_90)->sort_0 = sort_7;
    (&kernelContext_90)->atom_0 = atom_7;
    (&kernelContext_90)->manifolds_0 = manifolds_7;
    (&kernelContext_90)->loads_0 = loads_7;
    (&kernelContext_90)->bonds_0 = bonds_7;
    (&kernelContext_90)->args_0 = args_7;
    (&kernelContext_90)->info_0 = info_7;
    (&kernelContext_90)->scan_0 = scan_7;
    (&kernelContext_90)->parts_0 = parts_7;
    (&kernelContext_90)->colliders_0 = colliders_7;
    (&kernelContext_90)->collider_info_0 = collider_info_7;
    (&kernelContext_90)->view_chunks_0 = view_chunks_7;
    (&kernelContext_90)->view_0 = view_7;
    (&kernelContext_90)->objects_0 = objects_7;
    (&kernelContext_90)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_90)->lines_0 = lines_7;
    (&kernelContext_90)->geometry_0 = geometry_7;
    (&kernelContext_90)->shots_0 = shots_7;
    (&kernelContext_90)->lp_0 = lp_7;
    (&kernelContext_90)->ext_0 = ext_8;
    (&kernelContext_90)->lp_sides_0 = lp_sides_7;
    (&kernelContext_90)->engine_counters_0 = engine_counters_7;
    (&kernelContext_90)->engine_contacts_0 = engine_contacts_7;
    (&kernelContext_90)->cs_anchors_0 = cs_anchors_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_90)->g_scan_0 = &g_scan_7;
    threadgroup array<float, int(64)> g_il_best_7;
    (&kernelContext_90)->g_il_best_0 = &g_il_best_7;
    uint c_7 = id_6.x;
    bool _S631;
    if(c_7 >= (params_7->chunk_count_0))
    {
        _S631 = true;
    }
    else
    {
        uint _S632 = body_of_0(c_7, &kernelContext_90);
        bool _S633 = dirty_0(_S632, &kernelContext_90);
        _S631 = !_S633;
    }
    if(_S631)
    {
        return;
    }
    uint _S634 = find_0(c_7, &kernelContext_90);
    uint _S635 = w_label_0(&kernelContext_90);
    *((&kernelContext_90)->work_0+(_S635 + c_7)) = _S634;
    uint _S636 = a_count_0(&kernelContext_90);
    uint _S637 = atomic_fetch_add_explicit((&kernelContext_90)->atom_0+(_S636 + _S634), 1U, memory_order_relaxed);
    if(((&kernelContext_90)->info_0[c_7] & 1U) != 0U)
    {
        uint _S638 = a_anchor_0(&kernelContext_90);
        uint _S639 = atomic_fetch_or_explicit((&kernelContext_90)->atom_0+(_S638 + _S634), 1U, memory_order_relaxed);
    }
    uint _S640 = 4U * c_7;
    float4 _S641 = float4(*((&kernelContext_90)->chunks_0+_S640)) ;
    float4 _S642 = float4(*((&kernelContext_90)->chunks_0+(_S640 + 1U))) ;
    float4 _S643 = float4(*((&kernelContext_90)->chunks_0+(_S640 + 2U))) ;
    float m_2 = _S641.w;
    float3 d_18 = _S641.xyz - (float4(*((&kernelContext_90)->chunks_0+4U * _S634)) ).xyz;
    float dd_0 = dot(d_18, d_18);
    uint _S644 = a_sums_0(&kernelContext_90);
    uint at_9 = _S644 + 20U * _S634;
    add64_0(at_9, m_2, (&kernelContext_90)->params_0->scale_m_0, &kernelContext_90);
    float _S645 = d_18.x;
    float _S646 = m_2 * _S645;
    add64_0(at_9 + 2U, _S646, (&kernelContext_90)->params_0->scale_s_0, &kernelContext_90);
    float _S647 = d_18.y;
    float _S648 = m_2 * _S647;
    add64_0(at_9 + 4U, _S648, (&kernelContext_90)->params_0->scale_s_0, &kernelContext_90);
    float _S649 = d_18.z;
    add64_0(at_9 + 6U, m_2 * _S649, (&kernelContext_90)->params_0->scale_s_0, &kernelContext_90);
    add64_0(at_9 + 8U, _S642.x + m_2 * (dd_0 - _S645 * _S645), (&kernelContext_90)->params_0->scale_j_0, &kernelContext_90);
    add64_0(at_9 + 10U, _S642.y + m_2 * (dd_0 - _S647 * _S647), (&kernelContext_90)->params_0->scale_j_0, &kernelContext_90);
    add64_0(at_9 + 12U, _S642.z + m_2 * (dd_0 - _S649 * _S649), (&kernelContext_90)->params_0->scale_j_0, &kernelContext_90);
    add64_0(at_9 + 14U, _S642.w - _S646 * _S647, (&kernelContext_90)->params_0->scale_j_0, &kernelContext_90);
    add64_0(at_9 + 16U, _S643.x - _S646 * _S649, (&kernelContext_90)->params_0->scale_j_0, &kernelContext_90);
    add64_0(at_9 + 18U, _S643.y - _S648 * _S649, (&kernelContext_90)->params_0->scale_j_0, &kernelContext_90);
    return;
}

bool is_root_0(uint c_8, KernelContext_0 thread* kernelContext_91)
{
    bool _S650;
    if(c_8 < (kernelContext_91->params_0->chunk_count_0))
    {
        uint _S651 = body_of_0(c_8, kernelContext_91);
        bool _S652 = dirty_0(_S651, kernelContext_91);
        _S650 = _S652;
    }
    else
    {
        _S650 = false;
    }
    if(_S650)
    {
        uint _S653 = w_label_0(kernelContext_91);
        _S650 = (*(kernelContext_91->work_0+(_S653 + c_8))) == c_8;
    }
    else
    {
        _S650 = false;
    }
    return _S650;
}

[[kernel]] void fr_best_count(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], packed_float4 device* lp_8 [[buffer(24)]], packed_float4 device* ext_9 [[buffer(23)]], packed_float4 device* lp_sides_8 [[buffer(28)]], uint device* engine_counters_8 [[buffer(26)]], packed_float4 device* engine_contacts_8 [[buffer(25)]], packed_float4 device* cs_anchors_8 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_92;
    (&kernelContext_92)->params_0 = params_8;
    (&kernelContext_92)->work_0 = work_8;
    (&kernelContext_92)->frames_0 = frames_8;
    (&kernelContext_92)->bodies_0 = bodies_8;
    (&kernelContext_92)->chunks_0 = chunks_8;
    (&kernelContext_92)->sort_0 = sort_8;
    (&kernelContext_92)->atom_0 = atom_8;
    (&kernelContext_92)->manifolds_0 = manifolds_8;
    (&kernelContext_92)->loads_0 = loads_8;
    (&kernelContext_92)->bonds_0 = bonds_8;
    (&kernelContext_92)->args_0 = args_8;
    (&kernelContext_92)->info_0 = info_8;
    (&kernelContext_92)->scan_0 = scan_8;
    (&kernelContext_92)->parts_0 = parts_8;
    (&kernelContext_92)->colliders_0 = colliders_8;
    (&kernelContext_92)->collider_info_0 = collider_info_8;
    (&kernelContext_92)->view_chunks_0 = view_chunks_8;
    (&kernelContext_92)->view_0 = view_8;
    (&kernelContext_92)->objects_0 = objects_8;
    (&kernelContext_92)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_92)->lines_0 = lines_8;
    (&kernelContext_92)->geometry_0 = geometry_8;
    (&kernelContext_92)->shots_0 = shots_8;
    (&kernelContext_92)->lp_0 = lp_8;
    (&kernelContext_92)->ext_0 = ext_9;
    (&kernelContext_92)->lp_sides_0 = lp_sides_8;
    (&kernelContext_92)->engine_counters_0 = engine_counters_8;
    (&kernelContext_92)->engine_contacts_0 = engine_contacts_8;
    (&kernelContext_92)->cs_anchors_0 = cs_anchors_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_92)->g_scan_0 = &g_scan_8;
    threadgroup array<float, int(64)> g_il_best_8;
    (&kernelContext_92)->g_il_best_0 = &g_il_best_8;
    uint c_9 = id_7.x;
    bool _S654 = is_root_0(c_9, &kernelContext_92);
    if(!_S654)
    {
        return;
    }
    uint _S655 = a_best_count_0(&kernelContext_92);
    uint _S656 = body_of_0(c_9, &kernelContext_92);
    atomic<uint> device* _S657 = (&kernelContext_92)->atom_0+(_S655 + _S656);
    uint _S658 = a_count_0(&kernelContext_92);
    uint _S659 = atomic_load_explicit((&kernelContext_92)->atom_0+(_S658 + c_9), memory_order_relaxed);
    uint _S660 = atomic_fetch_max_explicit(_S657, _S659, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], packed_float4 device* lp_9 [[buffer(24)]], packed_float4 device* ext_10 [[buffer(23)]], packed_float4 device* lp_sides_9 [[buffer(28)]], uint device* engine_counters_9 [[buffer(26)]], packed_float4 device* engine_contacts_9 [[buffer(25)]], packed_float4 device* cs_anchors_9 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_93;
    (&kernelContext_93)->params_0 = params_9;
    (&kernelContext_93)->work_0 = work_9;
    (&kernelContext_93)->frames_0 = frames_9;
    (&kernelContext_93)->bodies_0 = bodies_9;
    (&kernelContext_93)->chunks_0 = chunks_9;
    (&kernelContext_93)->sort_0 = sort_9;
    (&kernelContext_93)->atom_0 = atom_9;
    (&kernelContext_93)->manifolds_0 = manifolds_9;
    (&kernelContext_93)->loads_0 = loads_9;
    (&kernelContext_93)->bonds_0 = bonds_9;
    (&kernelContext_93)->args_0 = args_9;
    (&kernelContext_93)->info_0 = info_9;
    (&kernelContext_93)->scan_0 = scan_9;
    (&kernelContext_93)->parts_0 = parts_9;
    (&kernelContext_93)->colliders_0 = colliders_9;
    (&kernelContext_93)->collider_info_0 = collider_info_9;
    (&kernelContext_93)->view_chunks_0 = view_chunks_9;
    (&kernelContext_93)->view_0 = view_9;
    (&kernelContext_93)->objects_0 = objects_9;
    (&kernelContext_93)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_93)->lines_0 = lines_9;
    (&kernelContext_93)->geometry_0 = geometry_9;
    (&kernelContext_93)->shots_0 = shots_9;
    (&kernelContext_93)->lp_0 = lp_9;
    (&kernelContext_93)->ext_0 = ext_10;
    (&kernelContext_93)->lp_sides_0 = lp_sides_9;
    (&kernelContext_93)->engine_counters_0 = engine_counters_9;
    (&kernelContext_93)->engine_contacts_0 = engine_contacts_9;
    (&kernelContext_93)->cs_anchors_0 = cs_anchors_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_93)->g_scan_0 = &g_scan_9;
    threadgroup array<float, int(64)> g_il_best_9;
    (&kernelContext_93)->g_il_best_0 = &g_il_best_9;
    uint c_10 = id_8.x;
    bool _S661 = is_root_0(c_10, &kernelContext_93);
    if(!_S661)
    {
        return;
    }
    uint _S662 = body_of_0(c_10, &kernelContext_93);
    uint _S663 = a_count_0(&kernelContext_93);
    uint _S664 = atomic_load_explicit((&kernelContext_93)->atom_0+(_S663 + c_10), memory_order_relaxed);
    uint _S665 = a_best_count_0(&kernelContext_93);
    uint _S666 = atomic_load_explicit((&kernelContext_93)->atom_0+(_S665 + _S662), memory_order_relaxed);
    if(_S664 == _S666)
    {
        uint _S667 = a_best_root_0(&kernelContext_93);
        uint _S668 = atomic_fetch_min_explicit((&kernelContext_93)->atom_0+(_S667 + _S662), c_10, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], packed_float4 device* lp_10 [[buffer(24)]], packed_float4 device* ext_11 [[buffer(23)]], packed_float4 device* lp_sides_10 [[buffer(28)]], uint device* engine_counters_10 [[buffer(26)]], packed_float4 device* engine_contacts_10 [[buffer(25)]], packed_float4 device* cs_anchors_10 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_94;
    (&kernelContext_94)->params_0 = params_10;
    (&kernelContext_94)->work_0 = work_10;
    (&kernelContext_94)->frames_0 = frames_10;
    (&kernelContext_94)->bodies_0 = bodies_10;
    (&kernelContext_94)->chunks_0 = chunks_10;
    (&kernelContext_94)->sort_0 = sort_10;
    (&kernelContext_94)->atom_0 = atom_10;
    (&kernelContext_94)->manifolds_0 = manifolds_10;
    (&kernelContext_94)->loads_0 = loads_10;
    (&kernelContext_94)->bonds_0 = bonds_10;
    (&kernelContext_94)->args_0 = args_10;
    (&kernelContext_94)->info_0 = info_10;
    (&kernelContext_94)->scan_0 = scan_10;
    (&kernelContext_94)->parts_0 = parts_10;
    (&kernelContext_94)->colliders_0 = colliders_10;
    (&kernelContext_94)->collider_info_0 = collider_info_10;
    (&kernelContext_94)->view_chunks_0 = view_chunks_10;
    (&kernelContext_94)->view_0 = view_10;
    (&kernelContext_94)->objects_0 = objects_10;
    (&kernelContext_94)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_94)->lines_0 = lines_10;
    (&kernelContext_94)->geometry_0 = geometry_10;
    (&kernelContext_94)->shots_0 = shots_10;
    (&kernelContext_94)->lp_0 = lp_10;
    (&kernelContext_94)->ext_0 = ext_11;
    (&kernelContext_94)->lp_sides_0 = lp_sides_10;
    (&kernelContext_94)->engine_counters_0 = engine_counters_10;
    (&kernelContext_94)->engine_contacts_0 = engine_contacts_10;
    (&kernelContext_94)->cs_anchors_0 = cs_anchors_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_94)->g_scan_0 = &g_scan_10;
    threadgroup array<float, int(64)> g_il_best_10;
    (&kernelContext_94)->g_il_best_0 = &g_il_best_10;
    uint c_11 = id_9.x;
    if(c_11 >= (params_10->chunk_count_0))
    {
        return;
    }
    uint _S669 = w_new_0(&kernelContext_94);
    uint device* _S670 = (&kernelContext_94)->work_0+(_S669 + c_11);
    bool _S671 = is_root_0(c_11, &kernelContext_94);
    bool _S672;
    if(_S671)
    {
        uint _S673 = a_best_root_0(&kernelContext_94);
        uint _S674 = body_of_0(c_11, &kernelContext_94);
        uint _S675 = atomic_load_explicit((&kernelContext_94)->atom_0+(_S673 + _S674), memory_order_relaxed);
        _S672 = _S675 != c_11;
    }
    else
    {
        _S672 = false;
    }
    uint _S676;
    if(_S672)
    {
        _S676 = 1U;
    }
    else
    {
        _S676 = 0U;
    }
    *_S670 = _S676;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], packed_float4 device* lp_11 [[buffer(24)]], packed_float4 device* ext_12 [[buffer(23)]], packed_float4 device* lp_sides_11 [[buffer(28)]], uint device* engine_counters_11 [[buffer(26)]], packed_float4 device* engine_contacts_11 [[buffer(25)]], packed_float4 device* cs_anchors_11 [[buffer(27)]])
{
    uint step_4;
    uint add_0;
    thread KernelContext_0 kernelContext_95;
    (&kernelContext_95)->params_0 = params_11;
    (&kernelContext_95)->work_0 = work_11;
    (&kernelContext_95)->frames_0 = frames_11;
    (&kernelContext_95)->bodies_0 = bodies_11;
    (&kernelContext_95)->chunks_0 = chunks_11;
    (&kernelContext_95)->sort_0 = sort_11;
    (&kernelContext_95)->atom_0 = atom_11;
    (&kernelContext_95)->manifolds_0 = manifolds_11;
    (&kernelContext_95)->loads_0 = loads_11;
    (&kernelContext_95)->bonds_0 = bonds_11;
    (&kernelContext_95)->args_0 = args_11;
    (&kernelContext_95)->info_0 = info_11;
    (&kernelContext_95)->scan_0 = scan_11;
    (&kernelContext_95)->parts_0 = parts_11;
    (&kernelContext_95)->colliders_0 = colliders_11;
    (&kernelContext_95)->collider_info_0 = collider_info_11;
    (&kernelContext_95)->view_chunks_0 = view_chunks_11;
    (&kernelContext_95)->view_0 = view_11;
    (&kernelContext_95)->objects_0 = objects_11;
    (&kernelContext_95)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_95)->lines_0 = lines_11;
    (&kernelContext_95)->geometry_0 = geometry_11;
    (&kernelContext_95)->shots_0 = shots_11;
    (&kernelContext_95)->lp_0 = lp_11;
    (&kernelContext_95)->ext_0 = ext_12;
    (&kernelContext_95)->lp_sides_0 = lp_sides_11;
    (&kernelContext_95)->engine_counters_0 = engine_counters_11;
    (&kernelContext_95)->engine_contacts_0 = engine_contacts_11;
    (&kernelContext_95)->cs_anchors_0 = cs_anchors_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_95)->g_scan_0 = &g_scan_11;
    threadgroup array<float, int(64)> g_il_best_11;
    (&kernelContext_95)->g_il_best_0 = &g_il_best_11;
    uint t_7 = thread_2.x;
    uint _S677 = group_0.x;
    uint _S678 = 1024U * _S677 + 4U * t_7;
    thread array<uint, int(4)> v_10;
    uint j_5 = 0U;
    uint sum_2 = 0U;
    for(;;)
    {
        if(j_5 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S678 + j_5) < ((&kernelContext_95)->scan_0->n_0))
        {
            step_4 = *((&kernelContext_95)->work_0+((&kernelContext_95)->scan_0->in_base_0 + _S678 + j_5));
        }
        else
        {
            step_4 = 0U;
        }
        v_10[j_5] = step_4;
        uint sum_3 = sum_2 + step_4;
        j_5 = j_5 + 1U;
        sum_2 = sum_3;
    }
    (*(&kernelContext_95)->g_scan_0)[t_7] = sum_2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    step_4 = 1U;
    for(;;)
    {
        if(step_4 < 256U)
        {
        }
        else
        {
            break;
        }
        if(t_7 >= step_4)
        {
            add_0 = (*(&kernelContext_95)->g_scan_0)[t_7 - step_4];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_95)->g_scan_0)[t_7] = (*(&kernelContext_95)->g_scan_0)[t_7] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_4 = step_4 << 1U;
    }
    uint _S679 = (*(&kernelContext_95)->g_scan_0)[t_7] - sum_2;
    j_5 = 0U;
    add_0 = _S679;
    for(;;)
    {
        if(j_5 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S678 + j_5) < ((&kernelContext_95)->scan_0->n_0))
        {
            *((&kernelContext_95)->work_0+((&kernelContext_95)->scan_0->out_base_0 + _S678 + j_5)) = add_0;
        }
        uint run_1 = add_0 + v_10[j_5];
        j_5 = j_5 + 1U;
        add_0 = run_1;
    }
    if(t_7 == 255U)
    {
        *((&kernelContext_95)->work_0+((&kernelContext_95)->scan_0->block_base_0 + _S677)) = (*(&kernelContext_95)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_3 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], packed_float4 device* lp_12 [[buffer(24)]], packed_float4 device* ext_13 [[buffer(23)]], packed_float4 device* lp_sides_12 [[buffer(28)]], uint device* engine_counters_12 [[buffer(26)]], packed_float4 device* engine_contacts_12 [[buffer(25)]], packed_float4 device* cs_anchors_12 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_96;
    (&kernelContext_96)->params_0 = params_12;
    (&kernelContext_96)->work_0 = work_12;
    (&kernelContext_96)->frames_0 = frames_12;
    (&kernelContext_96)->bodies_0 = bodies_12;
    (&kernelContext_96)->chunks_0 = chunks_12;
    (&kernelContext_96)->sort_0 = sort_12;
    (&kernelContext_96)->atom_0 = atom_12;
    (&kernelContext_96)->manifolds_0 = manifolds_12;
    (&kernelContext_96)->loads_0 = loads_12;
    (&kernelContext_96)->bonds_0 = bonds_12;
    (&kernelContext_96)->args_0 = args_12;
    (&kernelContext_96)->info_0 = info_12;
    (&kernelContext_96)->scan_0 = scan_12;
    (&kernelContext_96)->parts_0 = parts_12;
    (&kernelContext_96)->colliders_0 = colliders_12;
    (&kernelContext_96)->collider_info_0 = collider_info_12;
    (&kernelContext_96)->view_chunks_0 = view_chunks_12;
    (&kernelContext_96)->view_0 = view_12;
    (&kernelContext_96)->objects_0 = objects_12;
    (&kernelContext_96)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_96)->lines_0 = lines_12;
    (&kernelContext_96)->geometry_0 = geometry_12;
    (&kernelContext_96)->shots_0 = shots_12;
    (&kernelContext_96)->lp_0 = lp_12;
    (&kernelContext_96)->ext_0 = ext_13;
    (&kernelContext_96)->lp_sides_0 = lp_sides_12;
    (&kernelContext_96)->engine_counters_0 = engine_counters_12;
    (&kernelContext_96)->engine_contacts_0 = engine_contacts_12;
    (&kernelContext_96)->cs_anchors_0 = cs_anchors_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_96)->g_scan_0 = &g_scan_12;
    threadgroup array<float, int(64)> g_il_best_12;
    (&kernelContext_96)->g_il_best_0 = &g_il_best_12;
    uint t_8 = thread_3.x;
    uint blocks_0 = (scan_12->n_0 + 1023U) / 1024U;
    bool _S680 = t_8 < blocks_0;
    uint v_11;
    if(_S680)
    {
        v_11 = *((&kernelContext_96)->work_0+((&kernelContext_96)->scan_0->block_base_0 + t_8));
    }
    else
    {
        v_11 = 0U;
    }
    (*(&kernelContext_96)->g_scan_0)[t_8] = v_11;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint step_5 = 1U;
    for(;;)
    {
        if(step_5 < 1024U)
        {
        }
        else
        {
            break;
        }
        uint add_1;
        if(t_8 >= step_5)
        {
            add_1 = (*(&kernelContext_96)->g_scan_0)[t_8 - step_5];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_96)->g_scan_0)[t_8] = (*(&kernelContext_96)->g_scan_0)[t_8] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_5 = step_5 << 1U;
    }
    if(_S680)
    {
        *((&kernelContext_96)->work_0+((&kernelContext_96)->scan_0->block_base_0 + t_8)) = (*(&kernelContext_96)->g_scan_0)[t_8] - v_11;
    }
    if(t_8 == 1023U)
    {
        *((&kernelContext_96)->work_0+((&kernelContext_96)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_96)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], packed_float4 device* lp_13 [[buffer(24)]], packed_float4 device* ext_14 [[buffer(23)]], packed_float4 device* lp_sides_13 [[buffer(28)]], uint device* engine_counters_13 [[buffer(26)]], packed_float4 device* engine_contacts_13 [[buffer(25)]], packed_float4 device* cs_anchors_13 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_97;
    (&kernelContext_97)->params_0 = params_13;
    (&kernelContext_97)->work_0 = work_13;
    (&kernelContext_97)->frames_0 = frames_13;
    (&kernelContext_97)->bodies_0 = bodies_13;
    (&kernelContext_97)->chunks_0 = chunks_13;
    (&kernelContext_97)->sort_0 = sort_13;
    (&kernelContext_97)->atom_0 = atom_13;
    (&kernelContext_97)->manifolds_0 = manifolds_13;
    (&kernelContext_97)->loads_0 = loads_13;
    (&kernelContext_97)->bonds_0 = bonds_13;
    (&kernelContext_97)->args_0 = args_13;
    (&kernelContext_97)->info_0 = info_13;
    (&kernelContext_97)->scan_0 = scan_13;
    (&kernelContext_97)->parts_0 = parts_13;
    (&kernelContext_97)->colliders_0 = colliders_13;
    (&kernelContext_97)->collider_info_0 = collider_info_13;
    (&kernelContext_97)->view_chunks_0 = view_chunks_13;
    (&kernelContext_97)->view_0 = view_13;
    (&kernelContext_97)->objects_0 = objects_13;
    (&kernelContext_97)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_97)->lines_0 = lines_13;
    (&kernelContext_97)->geometry_0 = geometry_13;
    (&kernelContext_97)->shots_0 = shots_13;
    (&kernelContext_97)->lp_0 = lp_13;
    (&kernelContext_97)->ext_0 = ext_14;
    (&kernelContext_97)->lp_sides_0 = lp_sides_13;
    (&kernelContext_97)->engine_counters_0 = engine_counters_13;
    (&kernelContext_97)->engine_contacts_0 = engine_contacts_13;
    (&kernelContext_97)->cs_anchors_0 = cs_anchors_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_97)->g_scan_0 = &g_scan_13;
    threadgroup array<float, int(64)> g_il_best_13;
    (&kernelContext_97)->g_il_best_0 = &g_il_best_13;
    uint i_8 = id_10.x;
    if(i_8 >= (scan_13->n_0))
    {
        return;
    }
    uint device* _S681 = (&kernelContext_97)->work_0+((&kernelContext_97)->scan_0->out_base_0 + i_8);
    *_S681 = *_S681 + *((&kernelContext_97)->work_0+((&kernelContext_97)->scan_0->block_base_0 + i_8 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], packed_float4 device* lp_14 [[buffer(24)]], packed_float4 device* ext_15 [[buffer(23)]], packed_float4 device* lp_sides_14 [[buffer(28)]], uint device* engine_counters_14 [[buffer(26)]], packed_float4 device* engine_contacts_14 [[buffer(25)]], packed_float4 device* cs_anchors_14 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_98;
    (&kernelContext_98)->params_0 = params_14;
    (&kernelContext_98)->work_0 = work_14;
    (&kernelContext_98)->frames_0 = frames_14;
    (&kernelContext_98)->bodies_0 = bodies_14;
    (&kernelContext_98)->chunks_0 = chunks_14;
    (&kernelContext_98)->sort_0 = sort_14;
    (&kernelContext_98)->atom_0 = atom_14;
    (&kernelContext_98)->manifolds_0 = manifolds_14;
    (&kernelContext_98)->loads_0 = loads_14;
    (&kernelContext_98)->bonds_0 = bonds_14;
    (&kernelContext_98)->args_0 = args_14;
    (&kernelContext_98)->info_0 = info_14;
    (&kernelContext_98)->scan_0 = scan_14;
    (&kernelContext_98)->parts_0 = parts_14;
    (&kernelContext_98)->colliders_0 = colliders_14;
    (&kernelContext_98)->collider_info_0 = collider_info_14;
    (&kernelContext_98)->view_chunks_0 = view_chunks_14;
    (&kernelContext_98)->view_0 = view_14;
    (&kernelContext_98)->objects_0 = objects_14;
    (&kernelContext_98)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_98)->lines_0 = lines_14;
    (&kernelContext_98)->geometry_0 = geometry_14;
    (&kernelContext_98)->shots_0 = shots_14;
    (&kernelContext_98)->lp_0 = lp_14;
    (&kernelContext_98)->ext_0 = ext_15;
    (&kernelContext_98)->lp_sides_0 = lp_sides_14;
    (&kernelContext_98)->engine_counters_0 = engine_counters_14;
    (&kernelContext_98)->engine_contacts_0 = engine_contacts_14;
    (&kernelContext_98)->cs_anchors_0 = cs_anchors_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_98)->g_scan_0 = &g_scan_14;
    threadgroup array<float, int(64)> g_il_best_14;
    (&kernelContext_98)->g_il_best_0 = &g_il_best_14;
    uint s_8 = id_11.x;
    bool _S682;
    if(s_8 >= (params_14->body_count_0))
    {
        _S682 = true;
    }
    else
    {
        uint _S683 = w_free_0(&kernelContext_98);
        _S682 = (*((&kernelContext_98)->work_0+(_S683 + s_8))) == 0U;
    }
    if(_S682)
    {
        return;
    }
    uint _S684 = w_free_list_0(&kernelContext_98);
    uint _S685 = w_free_rank_0(&kernelContext_98);
    *((&kernelContext_98)->work_0+(_S684 + *((&kernelContext_98)->work_0+(_S685 + s_8)))) = s_8;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_99)
{
    uint _S686 = w_free_list_0(kernelContext_99);
    return _S686 + kernelContext_99->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_100)
{
    uint _S687 = scan_new_blocks_0(kernelContext_100);
    return _S687 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_101)
{
    uint blocks_1 = (kernelContext_101->params_0->body_count_0 + 1023U) / 1024U;
    uint _S688 = scan_free_blocks_0(kernelContext_101);
    return *(kernelContext_101->work_0+(_S688 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], packed_float4 device* lp_15 [[buffer(24)]], packed_float4 device* ext_16 [[buffer(23)]], packed_float4 device* lp_sides_15 [[buffer(28)]], uint device* engine_counters_15 [[buffer(26)]], packed_float4 device* engine_contacts_15 [[buffer(25)]], packed_float4 device* cs_anchors_15 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_102;
    (&kernelContext_102)->params_0 = params_15;
    (&kernelContext_102)->work_0 = work_15;
    (&kernelContext_102)->frames_0 = frames_15;
    (&kernelContext_102)->bodies_0 = bodies_15;
    (&kernelContext_102)->chunks_0 = chunks_15;
    (&kernelContext_102)->sort_0 = sort_15;
    (&kernelContext_102)->atom_0 = atom_15;
    (&kernelContext_102)->manifolds_0 = manifolds_15;
    (&kernelContext_102)->loads_0 = loads_15;
    (&kernelContext_102)->bonds_0 = bonds_15;
    (&kernelContext_102)->args_0 = args_15;
    (&kernelContext_102)->info_0 = info_15;
    (&kernelContext_102)->scan_0 = scan_15;
    (&kernelContext_102)->parts_0 = parts_15;
    (&kernelContext_102)->colliders_0 = colliders_15;
    (&kernelContext_102)->collider_info_0 = collider_info_15;
    (&kernelContext_102)->view_chunks_0 = view_chunks_15;
    (&kernelContext_102)->view_0 = view_15;
    (&kernelContext_102)->objects_0 = objects_15;
    (&kernelContext_102)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_102)->lines_0 = lines_15;
    (&kernelContext_102)->geometry_0 = geometry_15;
    (&kernelContext_102)->shots_0 = shots_15;
    (&kernelContext_102)->lp_0 = lp_15;
    (&kernelContext_102)->ext_0 = ext_16;
    (&kernelContext_102)->lp_sides_0 = lp_sides_15;
    (&kernelContext_102)->engine_counters_0 = engine_counters_15;
    (&kernelContext_102)->engine_contacts_0 = engine_contacts_15;
    (&kernelContext_102)->cs_anchors_0 = cs_anchors_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_102)->g_scan_0 = &g_scan_15;
    threadgroup array<float, int(64)> g_il_best_15;
    (&kernelContext_102)->g_il_best_0 = &g_il_best_15;
    uint c_12 = id_12.x;
    bool _S689 = is_root_0(c_12, &kernelContext_102);
    if(!_S689)
    {
        return;
    }
    uint _S690 = body_of_0(c_12, &kernelContext_102);
    uint _S691 = a_best_root_0(&kernelContext_102);
    uint _S692 = atomic_load_explicit((&kernelContext_102)->atom_0+(_S691 + _S690), memory_order_relaxed);
    uint slot_1;
    if(_S692 != c_12)
    {
        uint _S693 = w_rank_0(&kernelContext_102);
        uint device* _S694 = (&kernelContext_102)->work_0+(_S693 + c_12);
        uint k_23 = *_S694;
        uint _S695 = free_total_0(&kernelContext_102);
        if((*_S694) < _S695)
        {
            uint _S696 = w_free_list_0(&kernelContext_102);
            uint device* _S697 = (&kernelContext_102)->work_0+(_S696 + k_23);
            uint slot_2 = *_S697;
            uint _S698 = w_free_0(&kernelContext_102);
            *((&kernelContext_102)->work_0+(_S698 + *_S697)) = 0U;
            uint _S699 = a_counters_1(&kernelContext_102);
            uint _S700 = atomic_fetch_add_explicit((&kernelContext_102)->atom_0+(_S699 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S701 = a_counters_1(&kernelContext_102);
            uint _S702 = atomic_fetch_or_explicit((&kernelContext_102)->atom_0+(_S701 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S690;
        }
    }
    else
    {
        slot_1 = _S690;
    }
    uint _S703 = w_slot_0(&kernelContext_102);
    *((&kernelContext_102)->work_0+(_S703 + c_12)) = slot_1;
    return;
}

float read64_1(uint at_10, float scale_2, KernelContext_0 thread* kernelContext_103)
{
    uint _S704 = atomic_load_explicit(kernelContext_103->atom_0+(at_10 + 1U), memory_order_relaxed);
    int hi_3 = (as_type<int>((_S704)));
    uint lo_3 = atomic_load_explicit(kernelContext_103->atom_0+at_10, memory_order_relaxed);
    return (float(hi_3) * 4.294967296e+09f + float(lo_3)) / scale_2;
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
    thread Quat_0 q_12;
    if(m22_0 <= 0.0f)
    {
        float dif10_0 = m11_0 - m00_0;
        float omm22_0 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            float f_4 = omm22_0 - dif10_0;
            float inv_0 = 0.5f / sqrt(f_4);
            (&q_12)->x_0 = f_4 * inv_0;
            (&q_12)->y_0 = (m01_0 + m10_0) * inv_0;
            (&q_12)->z_0 = (m02_0 + m20_0) * inv_0;
            (&q_12)->w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            float f_5 = omm22_0 + dif10_0;
            float inv_1 = 0.5f / sqrt(f_5);
            (&q_12)->x_0 = (m01_0 + m10_0) * inv_1;
            (&q_12)->y_0 = f_5 * inv_1;
            (&q_12)->z_0 = (m12_0 + m21_0) * inv_1;
            (&q_12)->w_0 = (m20_0 - m02_0) * inv_1;
        }
    }
    else
    {
        float sum10_0 = m11_0 + m00_0;
        float opm22_0 = 1.0f + m22_0;
        if(sum10_0 <= 0.0f)
        {
            float f_6 = opm22_0 - sum10_0;
            float inv_2 = 0.5f / sqrt(f_6);
            (&q_12)->x_0 = (m02_0 + m20_0) * inv_2;
            (&q_12)->y_0 = (m12_0 + m21_0) * inv_2;
            (&q_12)->z_0 = f_6 * inv_2;
            (&q_12)->w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            float f_7 = opm22_0 + sum10_0;
            float inv_3 = 0.5f / sqrt(f_7);
            (&q_12)->x_0 = (m12_0 - m21_0) * inv_3;
            (&q_12)->y_0 = (m20_0 - m02_0) * inv_3;
            (&q_12)->z_0 = (m01_0 - m10_0) * inv_3;
            (&q_12)->w_0 = f_7 * inv_3;
        }
    }
    return q_12;
}

void principal_axes_0(float3 r0_0, float3 r1_0, float3 r2_2, Quat_0 thread* rot_0, float3 thread* moments_0)
{
    array<float, int(3)> _S705 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S706 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S707 = { { r2_2.x, r2_2.y, r2_2.z } };
    thread array<array<float, int(3)>, int(3)> a_7;
    a_7[int(0)] = _S705;
    a_7[int(1)] = _S706;
    a_7[int(2)] = _S707;
    array<float, int(3)> _S708 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S709 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S710 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_12;
    v_12[int(0)] = _S708;
    v_12[int(1)] = _S709;
    v_12[int(2)] = _S710;
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
        uint q_13;
        if((abs(a_7[int(0)][int(2)])) > (abs(a_7[0U][1U])))
        {
            q_13 = 2U;
        }
        else
        {
            q_13 = 1U;
        }
        uint p_12;
        uint q_14;
        if((abs(a_7[int(1)][int(2)])) > (abs(a_7[0U][q_13])))
        {
            p_12 = 1U;
            q_14 = 2U;
        }
        else
        {
            p_12 = 0U;
            q_14 = q_13;
        }
        if((abs(a_7[p_12][q_14])) <= (9.99999982451670044e-15f * max(max(abs(a_7[int(0)][int(0)]), abs(a_7[int(1)][int(1)])), max(abs(a_7[int(2)][int(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        float theta_0 = 0.5f * atan2(2.0f * a_7[p_12][q_14], a_7[q_14][q_14] - a_7[p_12][p_12]);
        float s_9 = sin(theta_0);
        float c_13 = cos(theta_0);
        float app_0 = a_7[p_12][p_12];
        float aqq_0 = a_7[q_14][q_14];
        float apq_0 = a_7[p_12][q_14];
        uint k_24 = 0U;
        for(;;)
        {
            if(k_24 < 3U)
            {
            }
            else
            {
                break;
            }
            bool _S711;
            if(k_24 != p_12)
            {
                _S711 = k_24 != q_14;
            }
            else
            {
                _S711 = false;
            }
            if(_S711)
            {
                float kp_0 = a_7[k_24][p_12];
                float kq_0 = a_7[k_24][q_14];
                float _S712 = c_13 * a_7[k_24][p_12] - s_9 * a_7[k_24][q_14];
                a_7[k_24][p_12] = _S712;
                a_7[p_12][k_24] = _S712;
                float _S713 = s_9 * kp_0 + c_13 * kq_0;
                a_7[k_24][q_14] = _S713;
                a_7[q_14][k_24] = _S713;
            }
            k_24 = k_24 + 1U;
        }
        float _S714 = c_13 * c_13;
        float _S715 = 2.0f * s_9 * c_13 * apq_0;
        float _S716 = s_9 * s_9;
        a_7[p_12][p_12] = _S714 * app_0 - _S715 + _S716 * aqq_0;
        a_7[q_14][q_14] = _S716 * app_0 + _S715 + _S714 * aqq_0;
        a_7[p_12][q_14] = 0.0f;
        a_7[q_14][p_12] = 0.0f;
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
            float kp_1 = v_12[row_0][p_12];
            float kq_1 = v_12[row_0][q_14];
            v_12[row_0][p_12] = c_13 * v_12[row_0][p_12] - s_9 * v_12[row_0][q_14];
            v_12[row_0][q_14] = s_9 * kp_1 + c_13 * kq_1;
            row_0 = row_0 + 1U;
        }
        it_0 = it_0 + 1U;
    }
    float3 x_5 = float3(v_12[int(0)][int(0)], v_12[int(1)][int(0)], v_12[int(2)][int(0)]);
    float3 y_1 = float3(v_12[int(0)][int(1)], v_12[int(1)][int(1)], v_12[int(2)][int(1)]);
    float3 z_1 = float3(v_12[int(0)][int(2)], v_12[int(1)][int(2)], v_12[int(2)][int(2)]);
    float3 z_2;
    if((dot(cross(x_5, y_1), z_1)) < 0.0f)
    {
        z_2 = - z_1;
    }
    else
    {
        z_2 = z_1;
    }
    thread Quat_0 _S717 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S718 = normalized_0(&_S717);
    *rot_0 = _S718;
    *moments_0 = float3(a_7[int(0)][int(0)], a_7[int(1)][int(1)], a_7[int(2)][int(2)]);
    return;
}

Quat_0 rot_to_rubble_0(const Quat_0 thread* q_15)
{
    thread Quat_0 axes_1;
    (&axes_1)->w_0 = 0.70710676908493042f;
    (&axes_1)->x_0 = -0.70710676908493042f;
    (&axes_1)->y_0 = 0.0f;
    (&axes_1)->z_0 = 0.0f;
    thread Quat_0 _S719 = axes_1;
    Quat_0 _S720 = quat_mul_0(&_S719, q_15);
    thread Quat_0 _S721 = _S720;
    Quat_0 _S722 = normalized_0(&_S721);
    return _S722;
}

uint stage_0(uint c_14, KernelContext_0 thread* kernelContext_104)
{
    return 2U * kernelContext_104->params_0->body_count_0 + 9U * c_14;
}

float4 quat_vec_0(const Quat_0 thread* q_16)
{
    return float4(q_16->x_0, q_16->y_0, q_16->z_0, q_16->w_0);
}

[[kernel]] void fr_piece(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], packed_float4 device* lp_16 [[buffer(24)]], packed_float4 device* ext_17 [[buffer(23)]], packed_float4 device* lp_sides_16 [[buffer(28)]], uint device* engine_counters_16 [[buffer(26)]], packed_float4 device* engine_contacts_16 [[buffer(25)]], packed_float4 device* cs_anchors_16 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_105;
    (&kernelContext_105)->params_0 = params_16;
    (&kernelContext_105)->work_0 = work_16;
    (&kernelContext_105)->frames_0 = frames_16;
    (&kernelContext_105)->bodies_0 = bodies_16;
    (&kernelContext_105)->chunks_0 = chunks_16;
    (&kernelContext_105)->sort_0 = sort_16;
    (&kernelContext_105)->atom_0 = atom_16;
    (&kernelContext_105)->manifolds_0 = manifolds_16;
    (&kernelContext_105)->loads_0 = loads_16;
    (&kernelContext_105)->bonds_0 = bonds_16;
    (&kernelContext_105)->args_0 = args_16;
    (&kernelContext_105)->info_0 = info_16;
    (&kernelContext_105)->scan_0 = scan_16;
    (&kernelContext_105)->parts_0 = parts_16;
    (&kernelContext_105)->colliders_0 = colliders_16;
    (&kernelContext_105)->collider_info_0 = collider_info_16;
    (&kernelContext_105)->view_chunks_0 = view_chunks_16;
    (&kernelContext_105)->view_0 = view_16;
    (&kernelContext_105)->objects_0 = objects_16;
    (&kernelContext_105)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_105)->lines_0 = lines_16;
    (&kernelContext_105)->geometry_0 = geometry_16;
    (&kernelContext_105)->shots_0 = shots_16;
    (&kernelContext_105)->lp_0 = lp_16;
    (&kernelContext_105)->ext_0 = ext_17;
    (&kernelContext_105)->lp_sides_0 = lp_sides_16;
    (&kernelContext_105)->engine_counters_0 = engine_counters_16;
    (&kernelContext_105)->engine_contacts_0 = engine_contacts_16;
    (&kernelContext_105)->cs_anchors_0 = cs_anchors_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_105)->g_scan_0 = &g_scan_16;
    threadgroup array<float, int(64)> g_il_best_16;
    (&kernelContext_105)->g_il_best_0 = &g_il_best_16;
    uint c_15 = id_13.x;
    bool _S723 = is_root_0(c_15, &kernelContext_105);
    if(!_S723)
    {
        return;
    }
    uint _S724 = body_of_0(c_15, &kernelContext_105);
    uint pb_0 = 7U * _S724;
    uint _S725 = a_sums_0(&kernelContext_105);
    uint at_11 = _S725 + 20U * c_15;
    float _S726 = read64_0(at_11, (&kernelContext_105)->params_0->scale_m_0, &kernelContext_105);
    float _S727 = read64_0(at_11 + 2U, (&kernelContext_105)->params_0->scale_s_0, &kernelContext_105);
    float _S728 = read64_0(at_11 + 4U, (&kernelContext_105)->params_0->scale_s_0, &kernelContext_105);
    float _S729 = read64_0(at_11 + 6U, (&kernelContext_105)->params_0->scale_s_0, &kernelContext_105);
    float3 d_19 = float3(_S727, _S728, _S729) / float3(max(_S726, 1.00000000317107685e-30f)) ;
    float _S730 = read64_0(at_11 + 8U, (&kernelContext_105)->params_0->scale_j_0, &kernelContext_105);
    float _S731 = read64_0(at_11 + 10U, (&kernelContext_105)->params_0->scale_j_0, &kernelContext_105);
    float _S732 = read64_0(at_11 + 12U, (&kernelContext_105)->params_0->scale_j_0, &kernelContext_105);
    float _S733 = read64_0(at_11 + 14U, (&kernelContext_105)->params_0->scale_j_0, &kernelContext_105);
    float _S734 = read64_0(at_11 + 16U, (&kernelContext_105)->params_0->scale_j_0, &kernelContext_105);
    float _S735 = read64_0(at_11 + 18U, (&kernelContext_105)->params_0->scale_j_0, &kernelContext_105);
    float dd_1 = dot(d_19, d_19);
    float _S736 = d_19.x;
    float _S737 = _S726 * _S736;
    float _S738 = d_19.y;
    float _S739 = _S733 + _S737 * _S738;
    float _S740 = d_19.z;
    float _S741 = _S734 + _S737 * _S740;
    float _S742 = _S735 + _S726 * _S738 * _S740;
    float3 com_1 = (float4(*((&kernelContext_105)->chunks_0+4U * c_15)) ).xyz + d_19;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S730 - _S726 * (dd_1 - _S736 * _S736), _S739, _S741), float3(_S739, _S731 - _S726 * (dd_1 - _S738 * _S738), _S742), float3(_S741, _S742, _S732 - _S726 * (dd_1 - _S740 * _S740)), &principal_0, &moments_1);
    uint _S743 = a_anchor_0(&kernelContext_105);
    uint _S744 = atomic_load_explicit((&kernelContext_105)->atom_0+(_S743 + c_15), memory_order_relaxed);
    bool anchored_0 = _S744 != 0U;
    thread Quat_0 q_17;
    thread float3 origin_4;
    structure_pose_0(_S724, pb_0, &q_17, &origin_4, &kernelContext_105);
    float3 p_parent_0 = (float4(*((&kernelContext_105)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_105)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_105)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S745 = q_17;
    float3 _S746 = rotate_0(&_S745, com_1);
    float3 pos_0 = to_rubble_0(origin_4 + _S746);
    thread Quat_0 _S747 = q_17;
    thread Quat_0 _S748 = principal_0;
    Quat_0 _S749 = quat_mul_0(&_S747, &_S748);
    thread Quat_0 _S750 = _S749;
    Quat_0 _S751 = normalized_0(&_S750);
    thread Quat_0 _S752 = _S751;
    Quat_0 _S753 = rot_to_rubble_0(&_S752);
    bool _S754;
    if(anchored_0)
    {
        _S754 = true;
    }
    else
    {
        _S754 = !(_S726 > 0.0f);
    }
    float inv_mass_3;
    if(_S754)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / _S726;
    }
    float3 _S755 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S756;
        if((moments_1.x) > 0.0f)
        {
            _S756 = 1.0f / moments_1.x;
        }
        else
        {
            _S756 = 0.0f;
        }
        float _S757;
        if((moments_1.y) > 0.0f)
        {
            _S757 = 1.0f / moments_1.y;
        }
        else
        {
            _S757 = 0.0f;
        }
        float _S758;
        if((moments_1.z) > 0.0f)
        {
            _S758 = 1.0f / moments_1.z;
        }
        else
        {
            _S758 = 0.0f;
        }
        inv_inertia_0 = float3(_S756, _S757, _S758);
    }
    else
    {
        inv_inertia_0 = _S755;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S755;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S755;
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
    uint _S759 = stage_0(c_15, &kernelContext_105);
    *((&kernelContext_105)->frames_0+_S759) = packed_float4(float4(pos_0, inv_mass_3)) ;
    packed_float4 device* _S760 = (&kernelContext_105)->frames_0+(_S759 + 1U);
    thread Quat_0 _S761 = _S753;
    float4 _S762 = quat_vec_0(&_S761);
    *_S760 = packed_float4(_S762) ;
    *((&kernelContext_105)->frames_0+(_S759 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_105)->frames_0+(_S759 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_105)->frames_0+(_S759 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_105)->params_0->step_0))))) ;
    *((&kernelContext_105)->frames_0+(_S759 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_105)->frames_0+(_S759 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_105)->frames_0+(_S759 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_105)->frames_0+2U * _S724)) ).w)) ;
    packed_float4 device* _S763 = (&kernelContext_105)->frames_0+(_S759 + 8U);
    thread Quat_0 _S764 = principal_0;
    float4 _S765 = quat_vec_0(&_S764);
    *_S763 = packed_float4(_S765) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], packed_float4 device* lp_17 [[buffer(24)]], packed_float4 device* ext_18 [[buffer(23)]], packed_float4 device* lp_sides_17 [[buffer(28)]], uint device* engine_counters_17 [[buffer(26)]], packed_float4 device* engine_contacts_17 [[buffer(25)]], packed_float4 device* cs_anchors_17 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_106;
    (&kernelContext_106)->params_0 = params_17;
    (&kernelContext_106)->work_0 = work_17;
    (&kernelContext_106)->frames_0 = frames_17;
    (&kernelContext_106)->bodies_0 = bodies_17;
    (&kernelContext_106)->chunks_0 = chunks_17;
    (&kernelContext_106)->sort_0 = sort_17;
    (&kernelContext_106)->atom_0 = atom_17;
    (&kernelContext_106)->manifolds_0 = manifolds_17;
    (&kernelContext_106)->loads_0 = loads_17;
    (&kernelContext_106)->bonds_0 = bonds_17;
    (&kernelContext_106)->args_0 = args_17;
    (&kernelContext_106)->info_0 = info_17;
    (&kernelContext_106)->scan_0 = scan_17;
    (&kernelContext_106)->parts_0 = parts_17;
    (&kernelContext_106)->colliders_0 = colliders_17;
    (&kernelContext_106)->collider_info_0 = collider_info_17;
    (&kernelContext_106)->view_chunks_0 = view_chunks_17;
    (&kernelContext_106)->view_0 = view_17;
    (&kernelContext_106)->objects_0 = objects_17;
    (&kernelContext_106)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_106)->lines_0 = lines_17;
    (&kernelContext_106)->geometry_0 = geometry_17;
    (&kernelContext_106)->shots_0 = shots_17;
    (&kernelContext_106)->lp_0 = lp_17;
    (&kernelContext_106)->ext_0 = ext_18;
    (&kernelContext_106)->lp_sides_0 = lp_sides_17;
    (&kernelContext_106)->engine_counters_0 = engine_counters_17;
    (&kernelContext_106)->engine_contacts_0 = engine_contacts_17;
    (&kernelContext_106)->cs_anchors_0 = cs_anchors_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_106)->g_scan_0 = &g_scan_17;
    threadgroup array<float, int(64)> g_il_best_17;
    (&kernelContext_106)->g_il_best_0 = &g_il_best_17;
    uint c_16 = id_14.x;
    bool _S766 = is_root_0(c_16, &kernelContext_106);
    if(!_S766)
    {
        return;
    }
    uint _S767 = w_slot_0(&kernelContext_106);
    uint slot_3 = *((&kernelContext_106)->work_0+(_S767 + c_16));
    uint _S768 = stage_0(c_16, &kernelContext_106);
    uint region_5 = 0U;
    for(;;)
    {
        if(region_5 < 2U)
        {
        }
        else
        {
            break;
        }
        uint w_4 = 0U;
        for(;;)
        {
            if(w_4 < 7U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_106)->bodies_0+(7U * (slot_3 + region_5 * (&kernelContext_106)->params_0->max_bodies_0) + w_4)) = packed_float4(float4(*((&kernelContext_106)->frames_0+(_S768 + w_4))) ) ;
            w_4 = w_4 + 1U;
        }
        region_5 = region_5 + 1U;
    }
    uint _S769 = 2U * slot_3;
    *((&kernelContext_106)->frames_0+_S769) = packed_float4(float4(*((&kernelContext_106)->frames_0+(_S768 + 7U))) ) ;
    *((&kernelContext_106)->frames_0+(_S769 + 1U)) = packed_float4(float4(*((&kernelContext_106)->frames_0+(_S768 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], packed_float4 device* lp_18 [[buffer(24)]], packed_float4 device* ext_19 [[buffer(23)]], packed_float4 device* lp_sides_18 [[buffer(28)]], uint device* engine_counters_18 [[buffer(26)]], packed_float4 device* engine_contacts_18 [[buffer(25)]], packed_float4 device* cs_anchors_18 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_107;
    (&kernelContext_107)->params_0 = params_18;
    (&kernelContext_107)->work_0 = work_18;
    (&kernelContext_107)->frames_0 = frames_18;
    (&kernelContext_107)->bodies_0 = bodies_18;
    (&kernelContext_107)->chunks_0 = chunks_18;
    (&kernelContext_107)->sort_0 = sort_18;
    (&kernelContext_107)->atom_0 = atom_18;
    (&kernelContext_107)->manifolds_0 = manifolds_18;
    (&kernelContext_107)->loads_0 = loads_18;
    (&kernelContext_107)->bonds_0 = bonds_18;
    (&kernelContext_107)->args_0 = args_18;
    (&kernelContext_107)->info_0 = info_18;
    (&kernelContext_107)->scan_0 = scan_18;
    (&kernelContext_107)->parts_0 = parts_18;
    (&kernelContext_107)->colliders_0 = colliders_18;
    (&kernelContext_107)->collider_info_0 = collider_info_18;
    (&kernelContext_107)->view_chunks_0 = view_chunks_18;
    (&kernelContext_107)->view_0 = view_18;
    (&kernelContext_107)->objects_0 = objects_18;
    (&kernelContext_107)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_107)->lines_0 = lines_18;
    (&kernelContext_107)->geometry_0 = geometry_18;
    (&kernelContext_107)->shots_0 = shots_18;
    (&kernelContext_107)->lp_0 = lp_18;
    (&kernelContext_107)->ext_0 = ext_19;
    (&kernelContext_107)->lp_sides_0 = lp_sides_18;
    (&kernelContext_107)->engine_counters_0 = engine_counters_18;
    (&kernelContext_107)->engine_contacts_0 = engine_contacts_18;
    (&kernelContext_107)->cs_anchors_0 = cs_anchors_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_107)->g_scan_0 = &g_scan_18;
    threadgroup array<float, int(64)> g_il_best_18;
    (&kernelContext_107)->g_il_best_0 = &g_il_best_18;
    uint c_17 = id_15.x;
    bool _S770;
    if(c_17 >= (params_18->chunk_count_0))
    {
        _S770 = true;
    }
    else
    {
        uint _S771 = body_of_0(c_17, &kernelContext_107);
        bool _S772 = dirty_0(_S771, &kernelContext_107);
        _S770 = !_S772;
    }
    if(_S770)
    {
        return;
    }
    uint _S773 = w_label_0(&kernelContext_107);
    uint device* _S774 = (&kernelContext_107)->work_0+(_S773 + c_17);
    uint r_21 = *_S774;
    uint _S775 = w_slot_0(&kernelContext_107);
    uint device* _S776 = (&kernelContext_107)->work_0+(_S775 + *_S774);
    uint slot_4 = *_S776;
    *((&kernelContext_107)->work_0+(w_body_0() + c_17)) = *_S776;
    uint _S777 = 2U * slot_4;
    float3 _S778 = (float4(*((&kernelContext_107)->frames_0+_S777)) ).xyz;
    thread Quat_0 _S779 = quat_of_0(float4(*((&kernelContext_107)->frames_0+(_S777 + 1U))) );
    Quat_0 _S780 = conj_0(&_S779);
    float inv_mass_4 = (float4(*((&kernelContext_107)->bodies_0+7U * slot_4)) ).w;
    float _S781;
    if(inv_mass_4 > 0.0f)
    {
        _S781 = 1.0f / inv_mass_4;
    }
    else
    {
        _S781 = float_bits_0(2139095040U);
    }
    float4 _S782 = float4(*((&kernelContext_107)->chunks_0+(4U * c_17 + 2U))) ;
    uint first_1 = (as_type<uint>((_S782.z)));
    uint _S783 = (as_type<uint>((_S782.w)));
    float4 _S784 = float4(*((&kernelContext_107)->chunks_0+(4U * r_21 + 3U))) ;
    uint p_13 = first_1;
    for(;;)
    {
        if(p_13 < (first_1 + _S783))
        {
        }
        else
        {
            break;
        }
        uint _S785 = 2U * p_13;
        float4 _S786 = float4(*((&kernelContext_107)->parts_0+_S785)) ;
        uint col_0 = (as_type<uint>((_S786.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_107)->parts_0+(_S785 + 1U))) );
        uint _S787 = 4U * col_0;
        float4 _S788 = float4(*((&kernelContext_107)->colliders_0+_S787)) ;
        *((&kernelContext_107)->colliders_0+_S787) = packed_float4(float4((as_type<float>((slot_4))), _S788.y, _S788.z, _S788.w)) ;
        uint _S789 = _S787 + 1U;
        packed_float4 device* _S790 = (&kernelContext_107)->colliders_0+_S789;
        float3 _S791 = _S786.xyz - _S778;
        thread Quat_0 _S792 = _S780;
        float3 _S793 = rotate_0(&_S792, _S791);
        *_S790 = packed_float4(float4(_S793, (float4(*((&kernelContext_107)->colliders_0+_S789)) ).w)) ;
        packed_float4 device* _S794 = (&kernelContext_107)->colliders_0+(_S787 + 2U);
        thread Quat_0 _S795 = _S780;
        thread Quat_0 _S796 = rot_1;
        Quat_0 _S797 = quat_mul_0(&_S795, &_S796);
        thread Quat_0 _S798 = _S797;
        Quat_0 _S799 = normalized_0(&_S798);
        thread Quat_0 _S800 = _S799;
        float4 _S801 = quat_vec_0(&_S800);
        *_S794 = packed_float4(_S801) ;
        uint _S802 = 2U * col_0;
        uint4 _S803 = uint4(*((&kernelContext_107)->collider_info_0+_S802)) ;
        *((&kernelContext_107)->collider_info_0+_S802) = packed_uint4(uint4(_S803.x, slot_4, _S803.z, _S803.w)) ;
        *((&kernelContext_107)->collider_info_0+(_S802 + 1U)) = packed_uint4(uint4((as_type<uint>((_S781))), (as_type<uint>((_S784.x))), (as_type<uint>((_S784.w))), 0U)) ;
        p_13 = p_13 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], packed_float4 device* lp_19 [[buffer(24)]], packed_float4 device* ext_20 [[buffer(23)]], packed_float4 device* lp_sides_19 [[buffer(28)]], uint device* engine_counters_19 [[buffer(26)]], packed_float4 device* engine_contacts_19 [[buffer(25)]], packed_float4 device* cs_anchors_19 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_108;
    (&kernelContext_108)->params_0 = params_19;
    (&kernelContext_108)->work_0 = work_19;
    (&kernelContext_108)->frames_0 = frames_19;
    (&kernelContext_108)->bodies_0 = bodies_19;
    (&kernelContext_108)->chunks_0 = chunks_19;
    (&kernelContext_108)->sort_0 = sort_19;
    (&kernelContext_108)->atom_0 = atom_19;
    (&kernelContext_108)->manifolds_0 = manifolds_19;
    (&kernelContext_108)->loads_0 = loads_19;
    (&kernelContext_108)->bonds_0 = bonds_19;
    (&kernelContext_108)->args_0 = args_19;
    (&kernelContext_108)->info_0 = info_19;
    (&kernelContext_108)->scan_0 = scan_19;
    (&kernelContext_108)->parts_0 = parts_19;
    (&kernelContext_108)->colliders_0 = colliders_19;
    (&kernelContext_108)->collider_info_0 = collider_info_19;
    (&kernelContext_108)->view_chunks_0 = view_chunks_19;
    (&kernelContext_108)->view_0 = view_19;
    (&kernelContext_108)->objects_0 = objects_19;
    (&kernelContext_108)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_108)->lines_0 = lines_19;
    (&kernelContext_108)->geometry_0 = geometry_19;
    (&kernelContext_108)->shots_0 = shots_19;
    (&kernelContext_108)->lp_0 = lp_19;
    (&kernelContext_108)->ext_0 = ext_20;
    (&kernelContext_108)->lp_sides_0 = lp_sides_19;
    (&kernelContext_108)->engine_counters_0 = engine_counters_19;
    (&kernelContext_108)->engine_contacts_0 = engine_contacts_19;
    (&kernelContext_108)->cs_anchors_0 = cs_anchors_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_108)->g_scan_0 = &g_scan_19;
    threadgroup array<float, int(64)> g_il_best_19;
    (&kernelContext_108)->g_il_best_0 = &g_il_best_19;
    uint s_10 = id_16.x;
    if(s_10 >= (params_19->body_count_0))
    {
        return;
    }
    uint _S804 = a_dirty_0(&kernelContext_108);
    atomic_store_explicit((&kernelContext_108)->atom_0+(_S804 + s_10), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_5 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_8 = fract(h_5);
    float q_18 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_8);
    float t_9 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_8));
    uint k_25 = uint(h_5);
    float3 _S805;
    if(k_25 == 0U)
    {
        _S805 = float3(0.92000001668930054f, t_9, 0.34959998726844788f);
    }
    else
    {
        if(k_25 == 1U)
        {
            _S805 = float3(q_18, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_25 == 2U)
            {
                _S805 = float3(0.34959998726844788f, 0.92000001668930054f, t_9);
            }
            else
            {
                if(k_25 == 3U)
                {
                    _S805 = float3(0.34959998726844788f, q_18, 0.92000001668930054f);
                }
                else
                {
                    if(k_25 == 4U)
                    {
                        _S805 = float3(t_9, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S805 = float3(0.92000001668930054f, 0.34959998726844788f, q_18);
                    }
                }
            }
        }
    }
    return _S805;
}

void write_object_0(uint o_4, const Quat_0 thread* q_19, float3 t_10, float4 color_0, KernelContext_0 thread* kernelContext_109)
{
    float3 _S806 = rotate_0(q_19, float3(1.0f, 0.0f, 0.0f));
    float3 _S807 = rotate_0(q_19, float3(0.0f, 1.0f, 0.0f));
    float3 _S808 = rotate_0(q_19, float3(0.0f, 0.0f, 1.0f));
    uint _S809 = 4U * o_4;
    *(kernelContext_109->objects_0+_S809) = packed_float4(float4(_S806.x, _S807.x, _S808.x, t_10.x)) ;
    *(kernelContext_109->objects_0+(_S809 + 1U)) = packed_float4(float4(_S806.y, _S807.y, _S808.y, t_10.y)) ;
    *(kernelContext_109->objects_0+(_S809 + 2U)) = packed_float4(float4(_S806.z, _S807.z, _S808.z, t_10.z)) ;
    *(kernelContext_109->objects_0+(_S809 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], packed_float4 device* lp_20 [[buffer(24)]], packed_float4 device* ext_21 [[buffer(23)]], packed_float4 device* lp_sides_20 [[buffer(28)]], uint device* engine_counters_20 [[buffer(26)]], packed_float4 device* engine_contacts_20 [[buffer(25)]], packed_float4 device* cs_anchors_20 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_110;
    (&kernelContext_110)->params_0 = params_20;
    (&kernelContext_110)->work_0 = work_20;
    (&kernelContext_110)->frames_0 = frames_20;
    (&kernelContext_110)->bodies_0 = bodies_20;
    (&kernelContext_110)->chunks_0 = chunks_20;
    (&kernelContext_110)->sort_0 = sort_20;
    (&kernelContext_110)->atom_0 = atom_20;
    (&kernelContext_110)->manifolds_0 = manifolds_20;
    (&kernelContext_110)->loads_0 = loads_20;
    (&kernelContext_110)->bonds_0 = bonds_20;
    (&kernelContext_110)->args_0 = args_20;
    (&kernelContext_110)->info_0 = info_20;
    (&kernelContext_110)->scan_0 = scan_20;
    (&kernelContext_110)->parts_0 = parts_20;
    (&kernelContext_110)->colliders_0 = colliders_20;
    (&kernelContext_110)->collider_info_0 = collider_info_20;
    (&kernelContext_110)->view_chunks_0 = view_chunks_20;
    (&kernelContext_110)->view_0 = view_20;
    (&kernelContext_110)->objects_0 = objects_20;
    (&kernelContext_110)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_110)->lines_0 = lines_20;
    (&kernelContext_110)->geometry_0 = geometry_20;
    (&kernelContext_110)->shots_0 = shots_20;
    (&kernelContext_110)->lp_0 = lp_20;
    (&kernelContext_110)->ext_0 = ext_21;
    (&kernelContext_110)->lp_sides_0 = lp_sides_20;
    (&kernelContext_110)->engine_counters_0 = engine_counters_20;
    (&kernelContext_110)->engine_contacts_0 = engine_contacts_20;
    (&kernelContext_110)->cs_anchors_0 = cs_anchors_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_110)->g_scan_0 = &g_scan_20;
    threadgroup array<float, int(64)> g_il_best_20;
    (&kernelContext_110)->g_il_best_0 = &g_il_best_20;
    uint k_26 = id_17.x;
    if(k_26 >= (params_20->chunk_count_0))
    {
        return;
    }
    uint _S810 = body_of_0(k_26, &kernelContext_110);
    uint b_35 = 7U * _S810;
    thread Quat_0 q_20;
    thread float3 origin_5;
    structure_pose_0(_S810, b_35, &q_20, &origin_5, &kernelContext_110);
    uint _S811 = 3U * k_26;
    float4 _S812 = float4(*((&kernelContext_110)->view_chunks_0+_S811)) ;
    float3 base_2 = (float4(*((&kernelContext_110)->view_chunks_0+(_S811 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_110)->bodies_0+b_35)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_110)->view_0->mode_0) == 1U)
    {
        float _S813 = max(length((float4(*((&kernelContext_110)->bodies_0+(b_35 + 3U))) ).xyz), length((float4(*((&kernelContext_110)->bodies_0+(b_35 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S813 > ((&kernelContext_110)->view_0->rest_speed_0))
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
        if(((&kernelContext_110)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S810);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S810);
            }
        }
    }
    uint _S814 = (as_type<uint>((_S812.w)));
    Quat_0 _S815 = quat_of_0(float4(*((&kernelContext_110)->view_chunks_0+(_S811 + 1U))) );
    thread Quat_0 _S816 = q_20;
    thread Quat_0 _S817 = _S815;
    Quat_0 _S818 = quat_mul_0(&_S816, &_S817);
    thread Quat_0 _S819 = _S818;
    Quat_0 _S820 = normalized_0(&_S819);
    float3 _S821 = _S812.xyz;
    thread Quat_0 _S822 = q_20;
    float3 _S823 = rotate_0(&_S822, _S821);
    float3 _S824 = origin_5 + _S823;
    float4 _S825 = float4(color_1, 1.0f);
    thread Quat_0 _S826 = _S820;
    write_object_0(_S814, &_S826, _S824, _S825, &kernelContext_110);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], packed_float4 device* lp_21 [[buffer(24)]], packed_float4 device* ext_22 [[buffer(23)]], packed_float4 device* lp_sides_21 [[buffer(28)]], uint device* engine_counters_21 [[buffer(26)]], packed_float4 device* engine_contacts_21 [[buffer(25)]], packed_float4 device* cs_anchors_21 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_111;
    (&kernelContext_111)->params_0 = params_21;
    (&kernelContext_111)->work_0 = work_21;
    (&kernelContext_111)->frames_0 = frames_21;
    (&kernelContext_111)->bodies_0 = bodies_21;
    (&kernelContext_111)->chunks_0 = chunks_21;
    (&kernelContext_111)->sort_0 = sort_21;
    (&kernelContext_111)->atom_0 = atom_21;
    (&kernelContext_111)->manifolds_0 = manifolds_21;
    (&kernelContext_111)->loads_0 = loads_21;
    (&kernelContext_111)->bonds_0 = bonds_21;
    (&kernelContext_111)->args_0 = args_21;
    (&kernelContext_111)->info_0 = info_21;
    (&kernelContext_111)->scan_0 = scan_21;
    (&kernelContext_111)->parts_0 = parts_21;
    (&kernelContext_111)->colliders_0 = colliders_21;
    (&kernelContext_111)->collider_info_0 = collider_info_21;
    (&kernelContext_111)->view_chunks_0 = view_chunks_21;
    (&kernelContext_111)->view_0 = view_21;
    (&kernelContext_111)->objects_0 = objects_21;
    (&kernelContext_111)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_111)->lines_0 = lines_21;
    (&kernelContext_111)->geometry_0 = geometry_21;
    (&kernelContext_111)->shots_0 = shots_21;
    (&kernelContext_111)->lp_0 = lp_21;
    (&kernelContext_111)->ext_0 = ext_22;
    (&kernelContext_111)->lp_sides_0 = lp_sides_21;
    (&kernelContext_111)->engine_counters_0 = engine_counters_21;
    (&kernelContext_111)->engine_contacts_0 = engine_contacts_21;
    (&kernelContext_111)->cs_anchors_0 = cs_anchors_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_111)->g_scan_0 = &g_scan_21;
    threadgroup array<float, int(64)> g_il_best_21;
    (&kernelContext_111)->g_il_best_0 = &g_il_best_21;
    uint i_9 = id_18.x;
    if(i_9 >= (view_21->shot_count_1))
    {
        return;
    }
    uint b_36 = 7U * (&kernelContext_111)->shot_bodies_0[i_9];
    uint o_5 = (&kernelContext_111)->view_0->impactor_base_0 + i_9;
    if(((float4(*((&kernelContext_111)->bodies_0+b_36)) ).w) == 0.0f)
    {
        uint w_5 = 0U;
        for(;;)
        {
            if(w_5 < 4U)
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_111)->objects_0+(4U * o_5 + w_5)) = packed_float4(float4(0.0f) ) ;
            w_5 = w_5 + 1U;
        }
        return;
    }
    thread Quat_0 _S827 = quat_of_0(float4(*((&kernelContext_111)->bodies_0+(b_36 + 1U))) );
    Quat_0 _S828 = rot_from_rubble_0(&_S827);
    float3 _S829 = from_rubble_0((float4(*((&kernelContext_111)->bodies_0+b_36)) ).xyz);
    float4 _S830 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S831 = _S828;
    write_object_0(o_5, &_S831, _S829, _S830, &kernelContext_111);
    return;
}

float3 chunk_world_0(uint k_27, KernelContext_0 thread* kernelContext_112)
{
    uint _S832 = body_of_0(k_27, kernelContext_112);
    thread Quat_0 q_21;
    thread float3 origin_6;
    structure_pose_0(_S832, 7U * _S832, &q_21, &origin_6, kernelContext_112);
    float3 _S833 = (float4(*(kernelContext_112->view_chunks_0+3U * k_27)) ).xyz;
    thread Quat_0 _S834 = q_21;
    float3 _S835 = rotate_0(&_S834, _S833);
    return origin_6 + _S835;
}

void write_line_vertex_0(uint v_13, float3 p_14, KernelContext_0 thread* kernelContext_113)
{
    uint _S836 = 7U * v_13;
    *(kernelContext_113->lines_0+_S836) = p_14.x;
    *(kernelContext_113->lines_0+(_S836 + 1U)) = p_14.y;
    *(kernelContext_113->lines_0+(_S836 + 2U)) = p_14.z;
    *(kernelContext_113->lines_0+(_S836 + 3U)) = 1.0f;
    *(kernelContext_113->lines_0+(_S836 + 4U)) = 0.86000001430511475f;
    *(kernelContext_113->lines_0+(_S836 + 5U)) = 0.25f;
    *(kernelContext_113->lines_0+(_S836 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], packed_float4 device* lp_22 [[buffer(24)]], packed_float4 device* ext_23 [[buffer(23)]], packed_float4 device* lp_sides_22 [[buffer(28)]], uint device* engine_counters_22 [[buffer(26)]], packed_float4 device* engine_contacts_22 [[buffer(25)]], packed_float4 device* cs_anchors_22 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_114;
    (&kernelContext_114)->params_0 = params_22;
    (&kernelContext_114)->work_0 = work_22;
    (&kernelContext_114)->frames_0 = frames_22;
    (&kernelContext_114)->bodies_0 = bodies_22;
    (&kernelContext_114)->chunks_0 = chunks_22;
    (&kernelContext_114)->sort_0 = sort_22;
    (&kernelContext_114)->atom_0 = atom_22;
    (&kernelContext_114)->manifolds_0 = manifolds_22;
    (&kernelContext_114)->loads_0 = loads_22;
    (&kernelContext_114)->bonds_0 = bonds_22;
    (&kernelContext_114)->args_0 = args_22;
    (&kernelContext_114)->info_0 = info_22;
    (&kernelContext_114)->scan_0 = scan_22;
    (&kernelContext_114)->parts_0 = parts_22;
    (&kernelContext_114)->colliders_0 = colliders_22;
    (&kernelContext_114)->collider_info_0 = collider_info_22;
    (&kernelContext_114)->view_chunks_0 = view_chunks_22;
    (&kernelContext_114)->view_0 = view_22;
    (&kernelContext_114)->objects_0 = objects_22;
    (&kernelContext_114)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_114)->lines_0 = lines_22;
    (&kernelContext_114)->geometry_0 = geometry_22;
    (&kernelContext_114)->shots_0 = shots_22;
    (&kernelContext_114)->lp_0 = lp_22;
    (&kernelContext_114)->ext_0 = ext_23;
    (&kernelContext_114)->lp_sides_0 = lp_sides_22;
    (&kernelContext_114)->engine_counters_0 = engine_counters_22;
    (&kernelContext_114)->engine_contacts_0 = engine_contacts_22;
    (&kernelContext_114)->cs_anchors_0 = cs_anchors_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_114)->g_scan_0 = &g_scan_22;
    threadgroup array<float, int(64)> g_il_best_22;
    (&kernelContext_114)->g_il_best_0 = &g_il_best_22;
    uint i_10 = id_19.x;
    if(i_10 >= (params_22->bond_count_0))
    {
        return;
    }
    uint4 _S837 = uint4(*((&kernelContext_114)->bonds_0+i_10)) ;
    float3 _S838 = chunk_world_0(_S837.x, &kernelContext_114);
    uint _S839 = w_alive_0(&kernelContext_114);
    float3 b_37;
    if((*((&kernelContext_114)->work_0+(_S839 + i_10))) != 0U)
    {
        float3 _S840 = chunk_world_0(_S837.y, &kernelContext_114);
        b_37 = _S840;
    }
    else
    {
        b_37 = _S838;
    }
    uint _S841 = 2U * i_10;
    write_line_vertex_0(_S841, _S838, &kernelContext_114);
    write_line_vertex_0(_S841 + 1U, b_37, &kernelContext_114);
    return;
}

[[kernel]] void fr_strike_reset(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], atomic<uint> device* atom_23 [[buffer(5)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], packed_float4 device* lp_23 [[buffer(24)]], packed_float4 device* ext_24 [[buffer(23)]], packed_float4 device* lp_sides_23 [[buffer(28)]], uint device* engine_counters_23 [[buffer(26)]], packed_float4 device* engine_contacts_23 [[buffer(25)]], packed_float4 device* cs_anchors_23 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_115;
    (&kernelContext_115)->params_0 = params_23;
    (&kernelContext_115)->work_0 = work_23;
    (&kernelContext_115)->frames_0 = frames_23;
    (&kernelContext_115)->bodies_0 = bodies_23;
    (&kernelContext_115)->chunks_0 = chunks_23;
    (&kernelContext_115)->sort_0 = sort_23;
    (&kernelContext_115)->atom_0 = atom_23;
    (&kernelContext_115)->manifolds_0 = manifolds_23;
    (&kernelContext_115)->loads_0 = loads_23;
    (&kernelContext_115)->bonds_0 = bonds_23;
    (&kernelContext_115)->args_0 = args_23;
    (&kernelContext_115)->info_0 = info_23;
    (&kernelContext_115)->scan_0 = scan_23;
    (&kernelContext_115)->parts_0 = parts_23;
    (&kernelContext_115)->colliders_0 = colliders_23;
    (&kernelContext_115)->collider_info_0 = collider_info_23;
    (&kernelContext_115)->view_chunks_0 = view_chunks_23;
    (&kernelContext_115)->view_0 = view_23;
    (&kernelContext_115)->objects_0 = objects_23;
    (&kernelContext_115)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_115)->lines_0 = lines_23;
    (&kernelContext_115)->geometry_0 = geometry_23;
    (&kernelContext_115)->shots_0 = shots_23;
    (&kernelContext_115)->lp_0 = lp_23;
    (&kernelContext_115)->ext_0 = ext_24;
    (&kernelContext_115)->lp_sides_0 = lp_sides_23;
    (&kernelContext_115)->engine_counters_0 = engine_counters_23;
    (&kernelContext_115)->engine_contacts_0 = engine_contacts_23;
    (&kernelContext_115)->cs_anchors_0 = cs_anchors_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_115)->g_scan_0 = &g_scan_23;
    threadgroup array<float, int(64)> g_il_best_23;
    (&kernelContext_115)->g_il_best_0 = &g_il_best_23;
    uint b_38 = id_20.x;
    if(b_38 >= (params_23->body_count_0))
    {
        return;
    }
    uint _S842 = a_strike_0(&kernelContext_115);
    uint at_12 = _S842 + 9U * b_38;
    uint w_6 = 0U;
    for(;;)
    {
        if(w_6 < 3U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit((&kernelContext_115)->atom_0+(at_12 + w_6), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_115)->atom_0+(at_12 + 3U + w_6), 0U, memory_order_relaxed);
        w_6 = w_6 + 1U;
    }
    atomic_store_explicit((&kernelContext_115)->atom_0+(at_12 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_115)->atom_0+(at_12 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_115)->atom_0+(at_12 + 8U), 4294967295U, memory_order_relaxed);
    uint _S843 = strike_at_0(b_38, &kernelContext_115);
    (*((&kernelContext_115)->frames_0+_S843))[int(3)] = 0.0f;
    return;
}

uint ordered_0(float f_9)
{
    uint u_6 = (as_type<uint>((f_9)));
    uint _S844;
    if((u_6 & 2147483648U) != 0U)
    {
        _S844 = ~u_6;
    }
    else
    {
        _S844 = u_6 | 2147483648U;
    }
    return _S844;
}

[[kernel]] void fr_strike_bounds(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], atomic<uint> device* atom_24 [[buffer(5)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], packed_float4 device* lp_24 [[buffer(24)]], packed_float4 device* ext_25 [[buffer(23)]], packed_float4 device* lp_sides_24 [[buffer(28)]], uint device* engine_counters_24 [[buffer(26)]], packed_float4 device* engine_contacts_24 [[buffer(25)]], packed_float4 device* cs_anchors_24 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_116;
    (&kernelContext_116)->params_0 = params_24;
    (&kernelContext_116)->work_0 = work_24;
    (&kernelContext_116)->frames_0 = frames_24;
    (&kernelContext_116)->bodies_0 = bodies_24;
    (&kernelContext_116)->chunks_0 = chunks_24;
    (&kernelContext_116)->sort_0 = sort_24;
    (&kernelContext_116)->atom_0 = atom_24;
    (&kernelContext_116)->manifolds_0 = manifolds_24;
    (&kernelContext_116)->loads_0 = loads_24;
    (&kernelContext_116)->bonds_0 = bonds_24;
    (&kernelContext_116)->args_0 = args_24;
    (&kernelContext_116)->info_0 = info_24;
    (&kernelContext_116)->scan_0 = scan_24;
    (&kernelContext_116)->parts_0 = parts_24;
    (&kernelContext_116)->colliders_0 = colliders_24;
    (&kernelContext_116)->collider_info_0 = collider_info_24;
    (&kernelContext_116)->view_chunks_0 = view_chunks_24;
    (&kernelContext_116)->view_0 = view_24;
    (&kernelContext_116)->objects_0 = objects_24;
    (&kernelContext_116)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_116)->lines_0 = lines_24;
    (&kernelContext_116)->geometry_0 = geometry_24;
    (&kernelContext_116)->shots_0 = shots_24;
    (&kernelContext_116)->lp_0 = lp_24;
    (&kernelContext_116)->ext_0 = ext_25;
    (&kernelContext_116)->lp_sides_0 = lp_sides_24;
    (&kernelContext_116)->engine_counters_0 = engine_counters_24;
    (&kernelContext_116)->engine_contacts_0 = engine_contacts_24;
    (&kernelContext_116)->cs_anchors_0 = cs_anchors_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_116)->g_scan_0 = &g_scan_24;
    threadgroup array<float, int(64)> g_il_best_24;
    (&kernelContext_116)->g_il_best_0 = &g_il_best_24;
    uint k_28 = id_21.x;
    bool _S845;
    if(k_28 >= (params_24->chunk_count_0))
    {
        _S845 = true;
    }
    else
    {
        _S845 = ((&kernelContext_116)->params_0->shot_count_0) == 0U;
    }
    if(_S845)
    {
        return;
    }
    thread Quat_0 q_22;
    float3 _S846 = chunk_box_world_0(k_28, &q_22, &kernelContext_116);
    float _S847 = (float4(*((&kernelContext_116)->geometry_0+3U * k_28)) ).w;
    uint _S848 = a_strike_0(&kernelContext_116);
    uint _S849 = body_of_0(k_28, &kernelContext_116);
    uint _S850 = _S848 + 9U * _S849;
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
        uint _S851 = ax_1;
        uint _S852 = atomic_fetch_min_explicit((&kernelContext_116)->atom_0+(_S850 + ax_1), ordered_0(_S846[ax_1] - _S847), memory_order_relaxed);
        uint _S853 = atomic_fetch_max_explicit((&kernelContext_116)->atom_0+(_S850 + 3U + ax_1), ordered_0(_S846[_S851] + _S847), memory_order_relaxed);
        ax_1 = ax_1 + 1U;
    }
    return;
}

float unordered_0(uint u_7)
{
    uint _S854;
    if((u_7 & 2147483648U) != 0U)
    {
        _S854 = u_7 & 2147483647U;
    }
    else
    {
        _S854 = ~u_7;
    }
    return (as_type<float>((_S854)));
}

[[kernel]] void fr_strike_find(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], atomic<uint> device* atom_25 [[buffer(5)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], packed_float4 device* lp_25 [[buffer(24)]], packed_float4 device* ext_26 [[buffer(23)]], packed_float4 device* lp_sides_25 [[buffer(28)]], uint device* engine_counters_25 [[buffer(26)]], packed_float4 device* engine_contacts_25 [[buffer(25)]], packed_float4 device* cs_anchors_25 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_117;
    (&kernelContext_117)->params_0 = params_25;
    (&kernelContext_117)->work_0 = work_25;
    (&kernelContext_117)->frames_0 = frames_25;
    (&kernelContext_117)->bodies_0 = bodies_25;
    (&kernelContext_117)->chunks_0 = chunks_25;
    (&kernelContext_117)->sort_0 = sort_25;
    (&kernelContext_117)->atom_0 = atom_25;
    (&kernelContext_117)->manifolds_0 = manifolds_25;
    (&kernelContext_117)->loads_0 = loads_25;
    (&kernelContext_117)->bonds_0 = bonds_25;
    (&kernelContext_117)->args_0 = args_25;
    (&kernelContext_117)->info_0 = info_25;
    (&kernelContext_117)->scan_0 = scan_25;
    (&kernelContext_117)->parts_0 = parts_25;
    (&kernelContext_117)->colliders_0 = colliders_25;
    (&kernelContext_117)->collider_info_0 = collider_info_25;
    (&kernelContext_117)->view_chunks_0 = view_chunks_25;
    (&kernelContext_117)->view_0 = view_25;
    (&kernelContext_117)->objects_0 = objects_25;
    (&kernelContext_117)->shot_bodies_0 = shot_bodies_25;
    (&kernelContext_117)->lines_0 = lines_25;
    (&kernelContext_117)->geometry_0 = geometry_25;
    (&kernelContext_117)->shots_0 = shots_25;
    (&kernelContext_117)->lp_0 = lp_25;
    (&kernelContext_117)->ext_0 = ext_26;
    (&kernelContext_117)->lp_sides_0 = lp_sides_25;
    (&kernelContext_117)->engine_counters_0 = engine_counters_25;
    (&kernelContext_117)->engine_contacts_0 = engine_contacts_25;
    (&kernelContext_117)->cs_anchors_0 = cs_anchors_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_117)->g_scan_0 = &g_scan_25;
    threadgroup array<float, int(64)> g_il_best_25;
    (&kernelContext_117)->g_il_best_0 = &g_il_best_25;
    uint b_39 = id_22.x;
    bool _S855;
    if(b_39 >= (params_25->body_count_0))
    {
        _S855 = true;
    }
    else
    {
        _S855 = ((&kernelContext_117)->params_0->shot_count_0) == 0U;
    }
    if(_S855)
    {
        return;
    }
    uint _S856 = a_strike_0(&kernelContext_117);
    uint at_13 = _S856 + 9U * b_39;
    uint _S857 = atomic_load_explicit((&kernelContext_117)->atom_0+at_13, memory_order_relaxed);
    if(_S857 == 4294967295U)
    {
        return;
    }
    uint _S858 = atomic_load_explicit((&kernelContext_117)->atom_0+at_13, memory_order_relaxed);
    float _S859 = unordered_0(_S858);
    uint _S860 = atomic_load_explicit((&kernelContext_117)->atom_0+(at_13 + 1U), memory_order_relaxed);
    float _S861 = unordered_0(_S860);
    uint _S862 = atomic_load_explicit((&kernelContext_117)->atom_0+(at_13 + 2U), memory_order_relaxed);
    float3 _S863 = float3(_S859, _S861, unordered_0(_S862));
    uint _S864 = atomic_load_explicit((&kernelContext_117)->atom_0+(at_13 + 3U), memory_order_relaxed);
    float _S865 = unordered_0(_S864);
    uint _S866 = atomic_load_explicit((&kernelContext_117)->atom_0+(at_13 + 4U), memory_order_relaxed);
    float _S867 = unordered_0(_S866);
    uint _S868 = atomic_load_explicit((&kernelContext_117)->atom_0+(at_13 + 5U), memory_order_relaxed);
    float3 _S869 = float3(_S865, _S867, unordered_0(_S868));
    float3 _S870 = from_rubble_0((float4(*((&kernelContext_117)->bodies_0+(7U * b_39 + 3U))) ).xyz);
    float _S871 = float_bits_0(2139095040U);
    uint k_29 = 0U;
    for(;;)
    {
        if(k_29 < ((&kernelContext_117)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S872 = float4(*((&kernelContext_117)->shots_0+2U * k_29)) ;
        uint sb_1 = 7U * (as_type<uint>((_S872.x)));
        if(((float4(*((&kernelContext_117)->bodies_0+sb_1)) ).w) == 0.0f)
        {
            k_29 = k_29 + 1U;
            continue;
        }
        float3 c_18 = from_rubble_0((float4(*((&kernelContext_117)->bodies_0+sb_1)) ).xyz);
        float3 rel_2 = from_rubble_0((float4(*((&kernelContext_117)->bodies_0+(sb_1 + 3U))) ).xyz) - _S870;
        if(!((length(rel_2)) > ((&kernelContext_117)->params_0->strike_speed_0)))
        {
            k_29 = k_29 + 1U;
            continue;
        }
        float3 _S873 = float3(_S872.y) ;
        float3 _S874 = _S863 - _S873;
        float3 _S875 = _S869 + _S873;
        float _S876 = (&kernelContext_117)->params_0->dt_0;
        float t0_1 = 0.0f;
        float t1_1 = _S876;
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
            uint _S877 = ax_2;
            if((abs(rel_2[ax_2])) < 9.999999960041972e-13f)
            {
                uint _S878 = ax_2;
                if((c_18[ax_2]) < (_S874[ax_2]))
                {
                    _S855 = true;
                }
                else
                {
                    _S855 = (c_18[_S878]) > (_S875[ax_2]);
                }
                float t0_2;
                if(_S855)
                {
                    t0_2 = _S871;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + 1U;
                continue;
            }
            float ta_1 = (_S874[ax_2] - c_18[ax_2]) / rel_2[_S877];
            float tb_1 = (_S875[ax_2] - c_18[ax_2]) / rel_2[_S877];
            float _S879 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S879;
            ax_2 = ax_2 + 1U;
        }
        if(t0_1 <= t1_1)
        {
            uint _S880 = strike_at_0(b_39, &kernelContext_117);
            *((&kernelContext_117)->frames_0+_S880) = packed_float4(float4(c_18, float(k_29 + 1U))) ;
            *((&kernelContext_117)->frames_0+(_S880 + 1U)) = packed_float4(float4(rel_2, t0_1)) ;
            return;
        }
        k_29 = k_29 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_entry(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], atomic<uint> device* atom_26 [[buffer(5)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], packed_float4 device* lp_26 [[buffer(24)]], packed_float4 device* ext_27 [[buffer(23)]], packed_float4 device* lp_sides_26 [[buffer(28)]], uint device* engine_counters_26 [[buffer(26)]], packed_float4 device* engine_contacts_26 [[buffer(25)]], packed_float4 device* cs_anchors_26 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_118;
    (&kernelContext_118)->params_0 = params_26;
    (&kernelContext_118)->work_0 = work_26;
    (&kernelContext_118)->frames_0 = frames_26;
    (&kernelContext_118)->bodies_0 = bodies_26;
    (&kernelContext_118)->chunks_0 = chunks_26;
    (&kernelContext_118)->sort_0 = sort_26;
    (&kernelContext_118)->atom_0 = atom_26;
    (&kernelContext_118)->manifolds_0 = manifolds_26;
    (&kernelContext_118)->loads_0 = loads_26;
    (&kernelContext_118)->bonds_0 = bonds_26;
    (&kernelContext_118)->args_0 = args_26;
    (&kernelContext_118)->info_0 = info_26;
    (&kernelContext_118)->scan_0 = scan_26;
    (&kernelContext_118)->parts_0 = parts_26;
    (&kernelContext_118)->colliders_0 = colliders_26;
    (&kernelContext_118)->collider_info_0 = collider_info_26;
    (&kernelContext_118)->view_chunks_0 = view_chunks_26;
    (&kernelContext_118)->view_0 = view_26;
    (&kernelContext_118)->objects_0 = objects_26;
    (&kernelContext_118)->shot_bodies_0 = shot_bodies_26;
    (&kernelContext_118)->lines_0 = lines_26;
    (&kernelContext_118)->geometry_0 = geometry_26;
    (&kernelContext_118)->shots_0 = shots_26;
    (&kernelContext_118)->lp_0 = lp_26;
    (&kernelContext_118)->ext_0 = ext_27;
    (&kernelContext_118)->lp_sides_0 = lp_sides_26;
    (&kernelContext_118)->engine_counters_0 = engine_counters_26;
    (&kernelContext_118)->engine_contacts_0 = engine_contacts_26;
    (&kernelContext_118)->cs_anchors_0 = cs_anchors_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_118)->g_scan_0 = &g_scan_26;
    threadgroup array<float, int(64)> g_il_best_26;
    (&kernelContext_118)->g_il_best_0 = &g_il_best_26;
    uint k_30 = id_23.x;
    bool _S881;
    if(k_30 >= (params_26->chunk_count_0))
    {
        _S881 = true;
    }
    else
    {
        _S881 = ((&kernelContext_118)->params_0->shot_count_0) == 0U;
    }
    if(_S881)
    {
        return;
    }
    uint _S882 = body_of_0(k_30, &kernelContext_118);
    thread uint shot_5;
    bool _S883 = struck_0(_S882, &shot_5, &kernelContext_118);
    if(!_S883)
    {
        return;
    }
    uint _S884 = strike_at_0(_S882, &kernelContext_118);
    float4 _S885 = float4(*((&kernelContext_118)->frames_0+(_S884 + 1U))) ;
    float3 from_1 = (float4(*((&kernelContext_118)->frames_0+_S884)) ).xyz;
    float3 rel_3 = _S885.xyz;
    float3 d_20 = rel_3 / float3(max(length(rel_3), 9.999999960041972e-13f)) ;
    float _S886 = (float4(*((&kernelContext_118)->shots_0+2U * shot_5)) ).y;
    thread Quat_0 q_23;
    float3 _S887 = chunk_box_world_0(k_30, &q_23, &kernelContext_118);
    uint _S888 = 3U * k_30;
    Quat_0 _S889 = quat_of_0(float4(*((&kernelContext_118)->geometry_0+(_S888 + 2U))) );
    thread Quat_0 _S890 = q_23;
    thread Quat_0 _S891 = _S889;
    Quat_0 _S892 = quat_mul_0(&_S890, &_S891);
    thread Quat_0 _S893 = _S892;
    Quat_0 _S894 = normalized_0(&_S893);
    float3 _S895 = (float4(*((&kernelContext_118)->geometry_0+(_S888 + 1U))) ).xyz;
    float _S896 = float_bits_0(2139095040U);
    float _S897 = (&kernelContext_118)->params_0->dt_0;
    float t_in_5 = 0.0f;
    float t_out_0 = _S897;
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
        thread Quat_0 _S898 = _S894;
        float3 _S899 = rotate_0(&_S898, e_3);
        float o_6 = dot(from_1 - _S887, _S899);
        float dv_1 = dot(rel_3, _S899);
        float ext_28 = _S895[ax_3] + _S886;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            float t_in_6;
            if((abs(o_6)) > ext_28)
            {
                t_in_6 = _S896;
            }
            else
            {
                t_in_6 = t_in_5;
            }
            t_in_5 = t_in_6;
            ax_3 = ax_3 + 1U;
            continue;
        }
        float ta_2 = (- ext_28 - o_6) / dv_1;
        float tb_2 = (ext_28 - o_6) / dv_1;
        float _S900 = min(t_out_0, max(ta_2, tb_2));
        t_in_5 = max(t_in_5, min(ta_2, tb_2));
        t_out_0 = _S900;
        ax_3 = ax_3 + 1U;
    }
    uint _S901 = a_strike_0(&kernelContext_118);
    uint at_14 = _S901 + 9U * _S882;
    if(t_in_5 <= t_out_0)
    {
        uint _S902 = atomic_fetch_min_explicit((&kernelContext_118)->atom_0+(at_14 + 6U), (as_type<uint>((t_in_5))), memory_order_relaxed);
    }
    float3 r_22 = _S887 - (from_1 + rel_3 * float3(_S885.w) );
    float along_1 = dot(r_22, d_20);
    uint _S903 = atomic_fetch_min_explicit((&kernelContext_118)->atom_0+(at_14 + 7U), (as_type<uint>((length(r_22 - d_20 * float3(along_1) ) + abs(along_1) * 0.00100000004749745f))), memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], atomic<uint> device* atom_27 [[buffer(5)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], packed_float4 device* lp_27 [[buffer(24)]], packed_float4 device* ext_29 [[buffer(23)]], packed_float4 device* lp_sides_27 [[buffer(28)]], uint device* engine_counters_27 [[buffer(26)]], packed_float4 device* engine_contacts_27 [[buffer(25)]], packed_float4 device* cs_anchors_27 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_119;
    (&kernelContext_119)->params_0 = params_27;
    (&kernelContext_119)->work_0 = work_27;
    (&kernelContext_119)->frames_0 = frames_27;
    (&kernelContext_119)->bodies_0 = bodies_27;
    (&kernelContext_119)->chunks_0 = chunks_27;
    (&kernelContext_119)->sort_0 = sort_27;
    (&kernelContext_119)->atom_0 = atom_27;
    (&kernelContext_119)->manifolds_0 = manifolds_27;
    (&kernelContext_119)->loads_0 = loads_27;
    (&kernelContext_119)->bonds_0 = bonds_27;
    (&kernelContext_119)->args_0 = args_27;
    (&kernelContext_119)->info_0 = info_27;
    (&kernelContext_119)->scan_0 = scan_27;
    (&kernelContext_119)->parts_0 = parts_27;
    (&kernelContext_119)->colliders_0 = colliders_27;
    (&kernelContext_119)->collider_info_0 = collider_info_27;
    (&kernelContext_119)->view_chunks_0 = view_chunks_27;
    (&kernelContext_119)->view_0 = view_27;
    (&kernelContext_119)->objects_0 = objects_27;
    (&kernelContext_119)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_119)->lines_0 = lines_27;
    (&kernelContext_119)->geometry_0 = geometry_27;
    (&kernelContext_119)->shots_0 = shots_27;
    (&kernelContext_119)->lp_0 = lp_27;
    (&kernelContext_119)->ext_0 = ext_29;
    (&kernelContext_119)->lp_sides_0 = lp_sides_27;
    (&kernelContext_119)->engine_counters_0 = engine_counters_27;
    (&kernelContext_119)->engine_contacts_0 = engine_contacts_27;
    (&kernelContext_119)->cs_anchors_0 = cs_anchors_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_119)->g_scan_0 = &g_scan_27;
    threadgroup array<float, int(64)> g_il_best_27;
    (&kernelContext_119)->g_il_best_0 = &g_il_best_27;
    uint k_31 = id_24.x;
    bool _S904;
    if(k_31 >= (params_27->chunk_count_0))
    {
        _S904 = true;
    }
    else
    {
        _S904 = ((&kernelContext_119)->params_0->shot_count_0) == 0U;
    }
    if(_S904)
    {
        return;
    }
    uint _S905 = body_of_0(k_31, &kernelContext_119);
    thread uint shot_6;
    bool _S906 = struck_0(_S905, &shot_6, &kernelContext_119);
    if(!_S906)
    {
        return;
    }
    uint _S907 = strike_at_0(_S905, &kernelContext_119);
    float4 _S908 = float4(*((&kernelContext_119)->frames_0+_S907)) ;
    float4 _S909 = float4(*((&kernelContext_119)->frames_0+(_S907 + 1U))) ;
    float3 _S910 = _S909.xyz;
    float3 d_21 = _S910 / float3(max(length(_S910), 9.999999960041972e-13f)) ;
    thread Quat_0 q_24;
    float3 _S911 = chunk_box_world_0(k_31, &q_24, &kernelContext_119);
    float3 r_23 = _S911 - (_S908.xyz + _S910 * float3(_S909.w) );
    float along_2 = dot(r_23, d_21);
    float dist_0 = length(r_23 - d_21 * float3(along_2) ) + abs(along_2) * 0.00100000004749745f;
    uint _S912 = a_strike_0(&kernelContext_119);
    uint at_15 = _S912 + 9U * _S905;
    uint _S913 = (as_type<uint>((dist_0)));
    uint _S914 = atomic_load_explicit((&kernelContext_119)->atom_0+(at_15 + 7U), memory_order_relaxed);
    if(_S913 == _S914)
    {
        uint _S915 = atomic_fetch_min_explicit((&kernelContext_119)->atom_0+(at_15 + 8U), k_31, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], atomic<uint> device* atom_28 [[buffer(5)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], packed_float4 device* lp_28 [[buffer(24)]], packed_float4 device* ext_30 [[buffer(23)]], packed_float4 device* lp_sides_28 [[buffer(28)]], uint device* engine_counters_28 [[buffer(26)]], packed_float4 device* engine_contacts_28 [[buffer(25)]], packed_float4 device* cs_anchors_28 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_120;
    (&kernelContext_120)->params_0 = params_28;
    (&kernelContext_120)->work_0 = work_28;
    (&kernelContext_120)->frames_0 = frames_28;
    (&kernelContext_120)->bodies_0 = bodies_28;
    (&kernelContext_120)->chunks_0 = chunks_28;
    (&kernelContext_120)->sort_0 = sort_28;
    (&kernelContext_120)->atom_0 = atom_28;
    (&kernelContext_120)->manifolds_0 = manifolds_28;
    (&kernelContext_120)->loads_0 = loads_28;
    (&kernelContext_120)->bonds_0 = bonds_28;
    (&kernelContext_120)->args_0 = args_28;
    (&kernelContext_120)->info_0 = info_28;
    (&kernelContext_120)->scan_0 = scan_28;
    (&kernelContext_120)->parts_0 = parts_28;
    (&kernelContext_120)->colliders_0 = colliders_28;
    (&kernelContext_120)->collider_info_0 = collider_info_28;
    (&kernelContext_120)->view_chunks_0 = view_chunks_28;
    (&kernelContext_120)->view_0 = view_28;
    (&kernelContext_120)->objects_0 = objects_28;
    (&kernelContext_120)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_120)->lines_0 = lines_28;
    (&kernelContext_120)->geometry_0 = geometry_28;
    (&kernelContext_120)->shots_0 = shots_28;
    (&kernelContext_120)->lp_0 = lp_28;
    (&kernelContext_120)->ext_0 = ext_30;
    (&kernelContext_120)->lp_sides_0 = lp_sides_28;
    (&kernelContext_120)->engine_counters_0 = engine_counters_28;
    (&kernelContext_120)->engine_contacts_0 = engine_contacts_28;
    (&kernelContext_120)->cs_anchors_0 = cs_anchors_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_120)->g_scan_0 = &g_scan_28;
    threadgroup array<float, int(64)> g_il_best_28;
    (&kernelContext_120)->g_il_best_0 = &g_il_best_28;
    uint b_40 = id_25.x;
    bool _S916;
    if(b_40 >= (params_28->body_count_0))
    {
        _S916 = true;
    }
    else
    {
        _S916 = ((&kernelContext_120)->params_0->shot_count_0) == 0U;
    }
    if(_S916)
    {
        return;
    }
    thread uint shot_7;
    bool _S917 = struck_0(b_40, &shot_7, &kernelContext_120);
    if(!_S917)
    {
        return;
    }
    uint _S918 = a_strike_0(&kernelContext_120);
    uint at_16 = _S918 + 9U * b_40;
    uint _S919 = atomic_load_explicit((&kernelContext_120)->atom_0+(at_16 + 6U), memory_order_relaxed);
    float t_in_7 = (as_type<float>((_S919)));
    uint entry_2 = atomic_load_explicit((&kernelContext_120)->atom_0+(at_16 + 8U), memory_order_relaxed);
    uint _S920 = strike_at_0(b_40, &kernelContext_120);
    if(!(t_in_7 <= ((&kernelContext_120)->params_0->dt_0)))
    {
        _S916 = true;
    }
    else
    {
        _S916 = entry_2 == 4294967295U;
    }
    if(_S916)
    {
        (*((&kernelContext_120)->frames_0+_S920))[int(3)] = 0.0f;
        return;
    }
    float3 rel_4 = (float4(*((&kernelContext_120)->frames_0+(_S920 + 1U))) ).xyz;
    float speed_14 = length(rel_4);
    float4 _S921 = float4(*((&kernelContext_120)->shots_0+2U * shot_7)) ;
    float _S922 = max(_S921.z, 0.00100000004749745f);
    float m_imp_0 = _S921.w;
    float e_imp_0 = (float4(*((&kernelContext_120)->shots_0+(2U * shot_7 + 1U))) ).x;
    float3 contact_16 = (float4(*((&kernelContext_120)->frames_0+_S920)) ).xyz + rel_4 * float3(t_in_7)  + rel_4 / float3(max(speed_14, 9.999999960041972e-13f))  * float3(_S921.y) ;
    float inv_mass_5 = (float4(*((&kernelContext_120)->bodies_0+7U * b_40)) ).w;
    bool anchored_1 = inv_mass_5 == 0.0f;
    float m_body_1;
    if(anchored_1)
    {
        m_body_1 = 0.0f;
    }
    else
    {
        m_body_1 = 1.0f / inv_mass_5;
    }
    float m_eff_6;
    if(anchored_1)
    {
        m_eff_6 = m_imp_0;
    }
    else
    {
        m_eff_6 = m_imp_0 * m_body_1 / (m_imp_0 + m_body_1);
    }
    float modulus_1 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_120)->geometry_0+(3U * entry_2 + 1U))) ).w);
    *((&kernelContext_120)->frames_0+(_S920 + 2U)) = packed_float4(float4(contact_16, m_eff_6 * speed_14 / (2.86999988555908203f * pow(m_eff_6 * m_eff_6 / (_S922 * modulus_1 * modulus_1 * max(speed_14, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

bool strike_share_breaks_0(uint4 bond_0, float3 contact_17, float force_2, KernelContext_0 thread* kernelContext_121)
{
    thread Quat_0 q_25;
    float3 _S923 = chunk_box_world_0(bond_0.x, &q_25, kernelContext_121);
    float3 _S924 = chunk_box_world_0(bond_0.y, &q_25, kernelContext_121);
    float dist_1 = length(float3(0.5f)  * (_S923 + _S924) - contact_17);
    return (force_2 * min(1.0f, (as_type<float>((bond_0.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((bond_0.z)));
}

[[kernel]] void fr_strike_breaks(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], atomic<uint> device* atom_29 [[buffer(5)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], packed_float4 device* lp_29 [[buffer(24)]], packed_float4 device* ext_31 [[buffer(23)]], packed_float4 device* lp_sides_29 [[buffer(28)]], uint device* engine_counters_29 [[buffer(26)]], packed_float4 device* engine_contacts_29 [[buffer(25)]], packed_float4 device* cs_anchors_29 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_122;
    (&kernelContext_122)->params_0 = params_29;
    (&kernelContext_122)->work_0 = work_29;
    (&kernelContext_122)->frames_0 = frames_29;
    (&kernelContext_122)->bodies_0 = bodies_29;
    (&kernelContext_122)->chunks_0 = chunks_29;
    (&kernelContext_122)->sort_0 = sort_29;
    (&kernelContext_122)->atom_0 = atom_29;
    (&kernelContext_122)->manifolds_0 = manifolds_29;
    (&kernelContext_122)->loads_0 = loads_29;
    (&kernelContext_122)->bonds_0 = bonds_29;
    (&kernelContext_122)->args_0 = args_29;
    (&kernelContext_122)->info_0 = info_29;
    (&kernelContext_122)->scan_0 = scan_29;
    (&kernelContext_122)->parts_0 = parts_29;
    (&kernelContext_122)->colliders_0 = colliders_29;
    (&kernelContext_122)->collider_info_0 = collider_info_29;
    (&kernelContext_122)->view_chunks_0 = view_chunks_29;
    (&kernelContext_122)->view_0 = view_29;
    (&kernelContext_122)->objects_0 = objects_29;
    (&kernelContext_122)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_122)->lines_0 = lines_29;
    (&kernelContext_122)->geometry_0 = geometry_29;
    (&kernelContext_122)->shots_0 = shots_29;
    (&kernelContext_122)->lp_0 = lp_29;
    (&kernelContext_122)->ext_0 = ext_31;
    (&kernelContext_122)->lp_sides_0 = lp_sides_29;
    (&kernelContext_122)->engine_counters_0 = engine_counters_29;
    (&kernelContext_122)->engine_contacts_0 = engine_contacts_29;
    (&kernelContext_122)->cs_anchors_0 = cs_anchors_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_122)->g_scan_0 = &g_scan_29;
    threadgroup array<float, int(64)> g_il_best_29;
    (&kernelContext_122)->g_il_best_0 = &g_il_best_29;
    uint i_11 = id_26.x;
    bool _S925;
    if(i_11 >= (params_29->bond_count_0))
    {
        _S925 = true;
    }
    else
    {
        _S925 = ((&kernelContext_122)->params_0->shot_count_0) == 0U;
    }
    if(_S925)
    {
        _S925 = true;
    }
    else
    {
        uint _S926 = w_alive_0(&kernelContext_122);
        _S925 = (*((&kernelContext_122)->work_0+(_S926 + i_11))) == 0U;
    }
    if(_S925)
    {
        return;
    }
    uint4 _S927 = uint4(*((&kernelContext_122)->bonds_0+i_11)) ;
    uint _S928 = body_of_0(_S927.x, &kernelContext_122);
    thread uint shot_8;
    bool _S929 = struck_0(_S928, &shot_8, &kernelContext_122);
    if(!_S929)
    {
        return;
    }
    uint _S930 = strike_at_0(_S928, &kernelContext_122);
    float4 _S931 = float4(*((&kernelContext_122)->frames_0+(_S930 + 2U))) ;
    bool _S932 = strike_share_breaks_0(_S927, _S931.xyz, _S931.w, &kernelContext_122);
    if(_S932)
    {
        uint _S933 = w_alive_0(&kernelContext_122);
        *((&kernelContext_122)->work_0+(_S933 + i_11)) = 0U;
        uint _S934 = a_dirty_0(&kernelContext_122);
        uint _S935 = atomic_fetch_or_explicit((&kernelContext_122)->atom_0+(_S934 + _S928), 1U, memory_order_relaxed);
        uint _S936 = a_counters_1(&kernelContext_122);
        uint _S937 = atomic_fetch_add_explicit((&kernelContext_122)->atom_0+_S936, 1U, memory_order_relaxed);
        uint _S938 = atomic_fetch_add_explicit((&kernelContext_122)->atom_0+(_S936 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_lp_0(uint b_41, KernelContext_0 thread* kernelContext_123)
{
    uint _S939 = a_strike_0(kernelContext_123);
    return _S939 + 9U * kernelContext_123->params_0->body_count_0 + 14U * b_41;
}

[[kernel]] void fr_lp_clear(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], atomic<uint> device* atom_30 [[buffer(5)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], packed_float4 device* lp_30 [[buffer(24)]], packed_float4 device* ext_32 [[buffer(23)]], packed_float4 device* lp_sides_30 [[buffer(28)]], uint device* engine_counters_30 [[buffer(26)]], packed_float4 device* engine_contacts_30 [[buffer(25)]], packed_float4 device* cs_anchors_30 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_124;
    (&kernelContext_124)->params_0 = params_30;
    (&kernelContext_124)->work_0 = work_30;
    (&kernelContext_124)->frames_0 = frames_30;
    (&kernelContext_124)->bodies_0 = bodies_30;
    (&kernelContext_124)->chunks_0 = chunks_30;
    (&kernelContext_124)->sort_0 = sort_30;
    (&kernelContext_124)->atom_0 = atom_30;
    (&kernelContext_124)->manifolds_0 = manifolds_30;
    (&kernelContext_124)->loads_0 = loads_30;
    (&kernelContext_124)->bonds_0 = bonds_30;
    (&kernelContext_124)->args_0 = args_30;
    (&kernelContext_124)->info_0 = info_30;
    (&kernelContext_124)->scan_0 = scan_30;
    (&kernelContext_124)->parts_0 = parts_30;
    (&kernelContext_124)->colliders_0 = colliders_30;
    (&kernelContext_124)->collider_info_0 = collider_info_30;
    (&kernelContext_124)->view_chunks_0 = view_chunks_30;
    (&kernelContext_124)->view_0 = view_30;
    (&kernelContext_124)->objects_0 = objects_30;
    (&kernelContext_124)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_124)->lines_0 = lines_30;
    (&kernelContext_124)->geometry_0 = geometry_30;
    (&kernelContext_124)->shots_0 = shots_30;
    (&kernelContext_124)->lp_0 = lp_30;
    (&kernelContext_124)->ext_0 = ext_32;
    (&kernelContext_124)->lp_sides_0 = lp_sides_30;
    (&kernelContext_124)->engine_counters_0 = engine_counters_30;
    (&kernelContext_124)->engine_contacts_0 = engine_contacts_30;
    (&kernelContext_124)->cs_anchors_0 = cs_anchors_30;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_124)->g_scan_0 = &g_scan_30;
    threadgroup array<float, int(64)> g_il_best_30;
    (&kernelContext_124)->g_il_best_0 = &g_il_best_30;
    uint b_42 = id_27.x;
    if(b_42 >= (params_30->body_count_0))
    {
        return;
    }
    uint _S940 = a_lp_0(b_42, &kernelContext_124);
    uint w_7 = 0U;
    for(;;)
    {
        if(w_7 < 12U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit((&kernelContext_124)->atom_0+(_S940 + w_7), 0U, memory_order_relaxed);
        w_7 = w_7 + 1U;
    }
    atomic_store_explicit((&kernelContext_124)->atom_0+(_S940 + 12U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_124)->atom_0+(_S940 + 13U), 0U, memory_order_relaxed);
    return;
}

uint st_f_0(uint k_32, uint state_0)
{
    return 22U * k_32 + state_0 + 6U;
}

uint lp_f_0(uint k_33)
{
    return st_f_0(k_33, 0U);
}

void add64w_0(uint at_17, float value_3, float scale_3, KernelContext_0 thread* kernelContext_125)
{
    float x_6 = round(value_3 * scale_3);
    float x_7;
    if(!((abs(x_6)) < 3.99999993722699776e+18f))
    {
        uint _S941 = a_counters_1(kernelContext_125);
        uint _S942 = atomic_fetch_or_explicit(kernelContext_125->atom_0+(_S941 + 3U), 2U, memory_order_relaxed);
        x_7 = 0.0f;
    }
    else
    {
        x_7 = x_6;
    }
    float u_8 = abs(x_7);
    float h_6 = floor(u_8 * 2.32830643653869629e-10f);
    uint lo_4 = uint(u_8 - h_6 * 4.294967296e+09f);
    uint hi_4 = uint(h_6);
    uint lo_5;
    uint hi_5;
    if(x_7 < 0.0f)
    {
        uint lo_6 = ~lo_4 + 1U;
        uint _S943 = ~hi_4;
        if(lo_6 == 0U)
        {
            lo_5 = 1U;
        }
        else
        {
            lo_5 = 0U;
        }
        uint _S944 = _S943 + lo_5;
        lo_5 = lo_6;
        hi_5 = _S944;
    }
    else
    {
        lo_5 = lo_4;
        hi_5 = hi_4;
    }
    uint old_1 = atomic_fetch_add_explicit(kernelContext_125->atom_0+at_17, lo_5, memory_order_relaxed);
    uint carry_1;
    if((old_1 + lo_5) < old_1)
    {
        carry_1 = 1U;
    }
    else
    {
        carry_1 = 0U;
    }
    uint hi_6 = hi_5 + carry_1;
    if(hi_6 != 0U)
    {
        uint _S945 = atomic_fetch_add_explicit(kernelContext_125->atom_0+(at_17 + 1U), hi_6, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_lp_sum(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], atomic<uint> device* atom_31 [[buffer(5)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_31 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], packed_float4 device* lp_31 [[buffer(24)]], packed_float4 device* ext_33 [[buffer(23)]], packed_float4 device* lp_sides_31 [[buffer(28)]], uint device* engine_counters_31 [[buffer(26)]], packed_float4 device* engine_contacts_31 [[buffer(25)]], packed_float4 device* cs_anchors_31 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_126;
    (&kernelContext_126)->params_0 = params_31;
    (&kernelContext_126)->work_0 = work_31;
    (&kernelContext_126)->frames_0 = frames_31;
    (&kernelContext_126)->bodies_0 = bodies_31;
    (&kernelContext_126)->chunks_0 = chunks_31;
    (&kernelContext_126)->sort_0 = sort_31;
    (&kernelContext_126)->atom_0 = atom_31;
    (&kernelContext_126)->manifolds_0 = manifolds_31;
    (&kernelContext_126)->loads_0 = loads_31;
    (&kernelContext_126)->bonds_0 = bonds_31;
    (&kernelContext_126)->args_0 = args_31;
    (&kernelContext_126)->info_0 = info_31;
    (&kernelContext_126)->scan_0 = scan_31;
    (&kernelContext_126)->parts_0 = parts_31;
    (&kernelContext_126)->colliders_0 = colliders_31;
    (&kernelContext_126)->collider_info_0 = collider_info_31;
    (&kernelContext_126)->view_chunks_0 = view_chunks_31;
    (&kernelContext_126)->view_0 = view_31;
    (&kernelContext_126)->objects_0 = objects_31;
    (&kernelContext_126)->shot_bodies_0 = shot_bodies_31;
    (&kernelContext_126)->lines_0 = lines_31;
    (&kernelContext_126)->geometry_0 = geometry_31;
    (&kernelContext_126)->shots_0 = shots_31;
    (&kernelContext_126)->lp_0 = lp_31;
    (&kernelContext_126)->ext_0 = ext_33;
    (&kernelContext_126)->lp_sides_0 = lp_sides_31;
    (&kernelContext_126)->engine_counters_0 = engine_counters_31;
    (&kernelContext_126)->engine_contacts_0 = engine_contacts_31;
    (&kernelContext_126)->cs_anchors_0 = cs_anchors_31;
    threadgroup array<uint, int(1024)> g_scan_31;
    (&kernelContext_126)->g_scan_0 = &g_scan_31;
    threadgroup array<float, int(64)> g_il_best_31;
    (&kernelContext_126)->g_il_best_0 = &g_il_best_31;
    uint k_34 = id_28.x;
    if(k_34 >= (params_31->chunk_count_0))
    {
        return;
    }
    uint _S946 = body_of_0(k_34, &kernelContext_126);
    thread Quat_0 q_26;
    thread float3 origin_7;
    structure_pose_0(_S946, 7U * _S946, &q_26, &origin_7, &kernelContext_126);
    thread Quat_0 _S947 = q_26;
    Quat_0 _S948 = conj_0(&_S947);
    float4 _S949 = float4(*((&kernelContext_126)->chunks_0+4U * k_34)) ;
    uint _S950 = l_vec_0(&kernelContext_126);
    uint _S951 = _S950 + 2U * k_34;
    float3 _S952 = (float4(*((&kernelContext_126)->loads_0+_S951)) ).xyz;
    thread Quat_0 _S953 = _S948;
    float3 _S954 = rotate_0(&_S953, _S952);
    float _S955 = _S949.w;
    float3 _S956 = float3((&kernelContext_126)->params_0->gx_0, (&kernelContext_126)->params_0->gy_0, (&kernelContext_126)->params_0->gz_0);
    thread Quat_0 _S957 = _S948;
    float3 _S958 = rotate_0(&_S957, _S956);
    float3 f_10 = _S954 + float3(_S955)  * _S958;
    float3 _S959 = (float4(*((&kernelContext_126)->loads_0+(_S951 + 1U))) ).xyz;
    thread Quat_0 _S960 = _S948;
    float3 _S961 = rotate_0(&_S960, _S959);
    uint _S962 = lp_f_0(k_34);
    *((&kernelContext_126)->lp_0+_S962) = packed_float4(float4(f_10, 0.0f)) ;
    *((&kernelContext_126)->lp_0+(_S962 + 1U)) = packed_float4(float4(_S961, 0.0f)) ;
    float3 _S963 = _S961 + cross(_S949.xyz - (float4(*((&kernelContext_126)->frames_0+2U * _S946)) ).xyz, f_10);
    uint _S964 = a_lp_0(_S946, &kernelContext_126);
    uint a_8 = 0U;
    for(;;)
    {
        if(a_8 < 3U)
        {
        }
        else
        {
            break;
        }
        uint _S965 = 2U * a_8;
        add64w_0(_S964 + _S965, f_10[a_8], (&kernelContext_126)->params_0->lp_scale_0, &kernelContext_126);
        add64w_0(_S964 + 6U + _S965, _S963[a_8], (&kernelContext_126)->params_0->lp_scale_0, &kernelContext_126);
        a_8 = a_8 + 1U;
    }
    uint _S966 = atomic_fetch_min_explicit((&kernelContext_126)->atom_0+(_S964 + 12U), k_34, memory_order_relaxed);
    if(((&kernelContext_126)->info_0[k_34] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_126)->atom_0+(_S964 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

float3 read64w3_0(uint at_18, float scale_4, KernelContext_0 thread* kernelContext_127)
{
    float _S967 = read64_1(at_18, scale_4, kernelContext_127);
    float _S968 = read64_1(at_18 + 2U, scale_4, kernelContext_127);
    float _S969 = read64_1(at_18 + 4U, scale_4, kernelContext_127);
    return float3(_S967, _S968, _S969);
}

[[kernel]] void fr_lp_relief(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], atomic<uint> device* atom_32 [[buffer(5)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_32 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], packed_float4 device* lp_32 [[buffer(24)]], packed_float4 device* ext_34 [[buffer(23)]], packed_float4 device* lp_sides_32 [[buffer(28)]], uint device* engine_counters_32 [[buffer(26)]], packed_float4 device* engine_contacts_32 [[buffer(25)]], packed_float4 device* cs_anchors_32 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_128;
    (&kernelContext_128)->params_0 = params_32;
    (&kernelContext_128)->work_0 = work_32;
    (&kernelContext_128)->frames_0 = frames_32;
    (&kernelContext_128)->bodies_0 = bodies_32;
    (&kernelContext_128)->chunks_0 = chunks_32;
    (&kernelContext_128)->sort_0 = sort_32;
    (&kernelContext_128)->atom_0 = atom_32;
    (&kernelContext_128)->manifolds_0 = manifolds_32;
    (&kernelContext_128)->loads_0 = loads_32;
    (&kernelContext_128)->bonds_0 = bonds_32;
    (&kernelContext_128)->args_0 = args_32;
    (&kernelContext_128)->info_0 = info_32;
    (&kernelContext_128)->scan_0 = scan_32;
    (&kernelContext_128)->parts_0 = parts_32;
    (&kernelContext_128)->colliders_0 = colliders_32;
    (&kernelContext_128)->collider_info_0 = collider_info_32;
    (&kernelContext_128)->view_chunks_0 = view_chunks_32;
    (&kernelContext_128)->view_0 = view_32;
    (&kernelContext_128)->objects_0 = objects_32;
    (&kernelContext_128)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_128)->lines_0 = lines_32;
    (&kernelContext_128)->geometry_0 = geometry_32;
    (&kernelContext_128)->shots_0 = shots_32;
    (&kernelContext_128)->lp_0 = lp_32;
    (&kernelContext_128)->ext_0 = ext_34;
    (&kernelContext_128)->lp_sides_0 = lp_sides_32;
    (&kernelContext_128)->engine_counters_0 = engine_counters_32;
    (&kernelContext_128)->engine_contacts_0 = engine_contacts_32;
    (&kernelContext_128)->cs_anchors_0 = cs_anchors_32;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_128)->g_scan_0 = &g_scan_32;
    threadgroup array<float, int(64)> g_il_best_32;
    (&kernelContext_128)->g_il_best_0 = &g_il_best_32;
    uint k_35 = id_29.x;
    if(k_35 >= (params_32->chunk_count_0))
    {
        return;
    }
    uint _S970 = body_of_0(k_35, &kernelContext_128);
    uint _S971 = a_lp_0(_S970, &kernelContext_128);
    uint _S972 = atomic_load_explicit((&kernelContext_128)->atom_0+(_S971 + 13U), memory_order_relaxed);
    bool supported_0 = _S972 != 0U;
    bool fixed_1;
    if(((&kernelContext_128)->info_0[k_35] & 1U) != 0U)
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            uint _S973 = atomic_load_explicit((&kernelContext_128)->atom_0+(_S971 + 12U), memory_order_relaxed);
            fixed_1 = _S973 == k_35;
        }
        else
        {
            fixed_1 = false;
        }
    }
    uint _S974 = lp_f_0(k_35);
    thread float4 f_11 = float4(*((&kernelContext_128)->lp_0+_S974)) ;
    uint _S975 = _S974 + 1U;
    thread float4 t_11 = float4(*((&kernelContext_128)->lp_0+_S975)) ;
    uint b_43 = 7U * _S970;
    float inv_mass_6 = (float4(*((&kernelContext_128)->bodies_0+b_43)) ).w;
    bool _S976;
    if(!supported_0)
    {
        _S976 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S976 = false;
    }
    if(_S976)
    {
        float3 _S977 = read64w3_0(_S971, (&kernelContext_128)->params_0->lp_scale_0, &kernelContext_128);
        float3 _S978 = read64w3_0(_S971 + 6U, (&kernelContext_128)->params_0->lp_scale_0, &kernelContext_128);
        uint _S979 = 2U * _S970;
        Quat_0 p_15 = quat_of_0(float4(*((&kernelContext_128)->frames_0+(_S979 + 1U))) );
        float3 _S980 = (float4(*((&kernelContext_128)->bodies_0+(b_43 + 2U))) ).xyz;
        thread Quat_0 _S981 = p_15;
        Quat_0 _S982 = conj_0(&_S981);
        thread Quat_0 _S983 = _S982;
        float3 _S984 = rotate_0(&_S983, _S978);
        float3 _S985 = _S980 * _S984;
        thread Quat_0 _S986 = p_15;
        float3 _S987 = rotate_0(&_S986, _S985);
        uint _S988 = 4U * k_35;
        float4 _S989 = float4(*((&kernelContext_128)->chunks_0+_S988)) ;
        float4 _S990 = float4(*((&kernelContext_128)->chunks_0+(_S988 + 1U))) ;
        float4 _S991 = float4(*((&kernelContext_128)->chunks_0+(_S988 + 2U))) ;
        f_11.xyz = f_11.xyz - float3(_S989.w)  * (_S977 * float3(inv_mass_6)  + cross(_S987, _S989.xyz - (float4(*((&kernelContext_128)->frames_0+_S979)) ).xyz));
        float _S992 = _S987.x;
        float _S993 = _S990.w;
        float _S994 = _S987.y;
        float _S995 = _S991.x;
        float _S996 = _S987.z;
        float _S997 = _S991.y;
        t_11.xyz = t_11.xyz - float3(_S990.x * _S992 + _S993 * _S994 + _S995 * _S996, _S993 * _S992 + _S990.y * _S994 + _S997 * _S996, _S995 * _S992 + _S997 * _S994 + _S990.z * _S996);
    }
    float _S998;
    if(fixed_1)
    {
        _S998 = 1.0f;
    }
    else
    {
        _S998 = 0.0f;
    }
    f_11.w = _S998;
    *((&kernelContext_128)->lp_0+_S974) = packed_float4(f_11) ;
    *((&kernelContext_128)->lp_0+_S975) = packed_float4(t_11) ;
    return;
}

uint adj_bond_0(uint at_19, KernelContext_0 thread* kernelContext_129)
{
    return (as_type<uint>(((float4(*(kernelContext_129->ext_0+(kernelContext_129->params_0->ext_adj_0 + at_19 / 4U))) )[at_19 % 4U])));
}

uint lp_intact_0(uint k_36, KernelContext_0 thread* kernelContext_130)
{
    float4 _S999 = float4(*(kernelContext_130->ext_0+(kernelContext_130->params_0->ext_chunks_0 + 4U * k_36))) ;
    uint _S1000 = (as_type<uint>((_S999.x)));
    uint _S1001 = (as_type<uint>((_S999.y)));
    uint e_4 = 0U;
    uint intact_0 = 0U;
    for(;;)
    {
        if(e_4 < _S1001)
        {
        }
        else
        {
            break;
        }
        uint _S1002 = w_alive_0(kernelContext_130);
        uint _S1003 = adj_bond_0(_S1000 + e_4, kernelContext_130);
        uint _S1004;
        if((*(kernelContext_130->work_0+(_S1002 + _S1003))) != 0U)
        {
            _S1004 = 1U;
        }
        else
        {
            _S1004 = 0U;
        }
        uint intact_1 = intact_0 + _S1004;
        e_4 = e_4 + 1U;
        intact_0 = intact_1;
    }
    return intact_0;
}

uint lp_l_0(uint k_37)
{
    return 22U * k_37 + 8U;
}

struct JointBond_0
{
    float4 geom0_0;
    float4 geom1_0;
    float4 stiff0_0;
    float4 stiff1_0;
    float4 rebar0_0;
    float4 rebar1_0;
    uint4 ids_3;
};

struct LpBond_0
{
    float3 centroid_0;
    float3 n_4;
    float3 t1_2;
    float3 t2_0;
    JointBond_0 law_0;
};

LpBond_0 lp_bond_0(uint i_12, KernelContext_0 thread* kernelContext_131)
{
    uint o_7 = 10U * i_12;
    thread LpBond_0 b_44;
    (&b_44)->centroid_0 = (float4(*(kernelContext_131->ext_0+o_7)) ).xyz;
    float3 _S1005 = (float4(*(kernelContext_131->ext_0+(o_7 + 1U))) ).xyz;
    (&b_44)->n_4 = _S1005;
    float3 _S1006 = (float4(*(kernelContext_131->ext_0+(o_7 + 2U))) ).xyz;
    (&b_44)->t1_2 = _S1006;
    (&b_44)->t2_0 = cross(_S1005, _S1006);
    (&(&b_44)->law_0)->geom0_0 = float4(*(kernelContext_131->ext_0+(o_7 + 3U))) ;
    (&(&b_44)->law_0)->geom1_0 = float4(*(kernelContext_131->ext_0+(o_7 + 4U))) ;
    (&(&b_44)->law_0)->stiff0_0 = float4(*(kernelContext_131->ext_0+(o_7 + 5U))) ;
    (&(&b_44)->law_0)->stiff1_0 = float4(*(kernelContext_131->ext_0+(o_7 + 6U))) ;
    (&(&b_44)->law_0)->rebar0_0 = float4(*(kernelContext_131->ext_0+(o_7 + 7U))) ;
    (&(&b_44)->law_0)->rebar1_0 = float4(*(kernelContext_131->ext_0+(o_7 + 8U))) ;
    (&(&b_44)->law_0)->ids_3 = (as_type<uint4>((float4(*(kernelContext_131->ext_0+(o_7 + 9U))) )));
    return b_44;
}

uint tri_0(uint i_13, uint j_6)
{
    return i_13 * (i_13 + 1U) / 2U + j_6;
}

bool lp_factor_0(uint k_38, KernelContext_0 thread* kernelContext_132)
{
    uint q_27;
    uint a_9;
    uint p2_0;
    uint q2_0;
    uint a2_0;
    float s_11;
    uint _S1007 = lp_intact_0(k_38, kernelContext_132);
    uint _S1008 = lp_l_0(k_38);
    uint _S1009 = _S1007 + 1U;
    if((as_type<uint>(((float4(*(kernelContext_132->lp_0+(_S1008 + 5U))) ).w))) == _S1009)
    {
        return true;
    }
    thread array<array<float, int(6)>, int(6)> d_22;
    uint p_16 = 0U;
    for(;;)
    {
        if(p_16 < 6U)
        {
        }
        else
        {
            break;
        }
        q_27 = 0U;
        for(;;)
        {
            if(q_27 < 6U)
            {
            }
            else
            {
                break;
            }
            d_22[p_16][q_27] = 0.0f;
            q_27 = q_27 + 1U;
        }
        p_16 = p_16 + 1U;
    }
    uint _S1010 = 4U * k_38;
    float4 _S1011 = float4(*(kernelContext_132->ext_0+(kernelContext_132->params_0->ext_chunks_0 + _S1010))) ;
    uint _S1012 = (as_type<uint>((_S1011.x)));
    uint _S1013 = (as_type<uint>((_S1011.y)));
    q_27 = 0U;
    for(;;)
    {
        if(q_27 < _S1013)
        {
        }
        else
        {
            break;
        }
        uint _S1014 = adj_bond_0(_S1012 + q_27, kernelContext_132);
        uint _S1015 = w_alive_0(kernelContext_132);
        if((*(kernelContext_132->work_0+(_S1015 + _S1014))) == 0U)
        {
            q_27 = q_27 + 1U;
            continue;
        }
        LpBond_0 _S1016 = lp_bond_0(_S1014, kernelContext_132);
        float3 _S1017 = _S1016.centroid_0 - (float4(*(kernelContext_132->chunks_0+_S1010)) ).xyz;
        float _S1018 = _S1016.law_0.stiff0_0.y;
        array<float, int(6)> _S1019 = { { _S1018, _S1018, _S1016.law_0.stiff0_0.x, _S1016.law_0.stiff0_0.z, _S1016.law_0.stiff0_0.w, _S1016.law_0.stiff1_0.x } };
        array<float3, int(3)> _S1020 = { { _S1016.t1_2, _S1016.t2_0, _S1016.n_4 } };
        thread array<array<float, int(6)>, int(6)> j_7;
        a_9 = 0U;
        for(;;)
        {
            if(a_9 < 3U)
            {
            }
            else
            {
                break;
            }
            float3 c_19 = cross(_S1017, _S1020[a_9]);
            float _S1021 = _S1020[a_9].x;
            j_7[a_9][int(0)] = _S1021;
            float _S1022 = _S1020[a_9].y;
            j_7[a_9][int(1)] = _S1022;
            float _S1023 = _S1020[a_9].z;
            j_7[a_9][int(2)] = _S1023;
            j_7[a_9][int(3)] = c_19.x;
            j_7[a_9][int(4)] = c_19.y;
            j_7[a_9][int(5)] = c_19.z;
            j_7[3U + a_9][int(0)] = 0.0f;
            j_7[3U + a_9][int(1)] = 0.0f;
            j_7[3U + a_9][int(2)] = 0.0f;
            j_7[3U + a_9][int(3)] = _S1021;
            j_7[3U + a_9][int(4)] = _S1022;
            j_7[3U + a_9][int(5)] = _S1023;
            a_9 = a_9 + 1U;
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
                s_11 = 0.0f;
                for(;;)
                {
                    if(a2_0 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_12 = s_11 + j_7[a2_0][p2_0] * _S1019[a2_0] * j_7[a2_0][q2_0];
                    a2_0 = a2_0 + 1U;
                    s_11 = s_12;
                }
                d_22[p2_0][q2_0] = d_22[p2_0][q2_0] + s_11;
                q2_0 = q2_0 + 1U;
            }
            p2_0 = p2_0 + 1U;
        }
        q_27 = q_27 + 1U;
    }
    thread array<float, int(21)> l_0;
    a_9 = 0U;
    for(;;)
    {
        if(a_9 < 21U)
        {
        }
        else
        {
            break;
        }
        l_0[a_9] = 0.0f;
        a_9 = a_9 + 1U;
    }
    bool ok_0 = true;
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
            s_11 = d_22[p2_0][q2_0];
            for(;;)
            {
                if(a2_0 < q2_0)
                {
                }
                else
                {
                    break;
                }
                float s_13 = s_11 - l_0[tri_0(p2_0, a2_0)] * l_0[tri_0(q2_0, a2_0)];
                a2_0 = a2_0 + 1U;
                s_11 = s_13;
            }
            if(p2_0 == q2_0)
            {
                bool _S1024;
                if(ok_1)
                {
                    _S1024 = s_11 > 0.0f;
                }
                else
                {
                    _S1024 = false;
                }
                l_0[tri_0(p2_0, p2_0)] = sqrt(max(s_11, 1.00000000317107685e-30f));
                ok_1 = _S1024;
            }
            else
            {
                l_0[tri_0(p2_0, q2_0)] = s_11 / l_0[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + 1U;
        }
        uint _S1025 = p2_0 + 1U;
        ok_0 = ok_1;
        p2_0 = _S1025;
    }
    if(!ok_0)
    {
        return false;
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
        packed_float4 device* _S1026 = kernelContext_132->lp_0+(_S1008 + q2_0);
        uint _S1027 = 4U * q2_0;
        float _S1028 = l_0[_S1027];
        uint _S1029 = _S1027 + 1U;
        if(_S1029 < 21U)
        {
            s_11 = l_0[_S1029];
        }
        else
        {
            s_11 = 0.0f;
        }
        uint _S1030 = _S1027 + 2U;
        float _S1031;
        if(_S1030 < 21U)
        {
            _S1031 = l_0[_S1030];
        }
        else
        {
            _S1031 = 0.0f;
        }
        float _S1032;
        if(q2_0 == 5U)
        {
            _S1032 = (as_type<float>((_S1009)));
        }
        else
        {
            _S1032 = l_0[_S1027 + 3U];
        }
        *_S1026 = packed_float4(float4(_S1028, s_11, _S1031, _S1032)) ;
        q2_0 = q2_0 + 1U;
    }
    return true;
}

[[kernel]] void fr_lp_block(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], atomic<uint> device* atom_33 [[buffer(5)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], packed_float4 device* lp_33 [[buffer(24)]], packed_float4 device* ext_35 [[buffer(23)]], packed_float4 device* lp_sides_33 [[buffer(28)]], uint device* engine_counters_33 [[buffer(26)]], packed_float4 device* engine_contacts_33 [[buffer(25)]], packed_float4 device* cs_anchors_33 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_133;
    (&kernelContext_133)->params_0 = params_33;
    (&kernelContext_133)->work_0 = work_33;
    (&kernelContext_133)->frames_0 = frames_33;
    (&kernelContext_133)->bodies_0 = bodies_33;
    (&kernelContext_133)->chunks_0 = chunks_33;
    (&kernelContext_133)->sort_0 = sort_33;
    (&kernelContext_133)->atom_0 = atom_33;
    (&kernelContext_133)->manifolds_0 = manifolds_33;
    (&kernelContext_133)->loads_0 = loads_33;
    (&kernelContext_133)->bonds_0 = bonds_33;
    (&kernelContext_133)->args_0 = args_33;
    (&kernelContext_133)->info_0 = info_33;
    (&kernelContext_133)->scan_0 = scan_33;
    (&kernelContext_133)->parts_0 = parts_33;
    (&kernelContext_133)->colliders_0 = colliders_33;
    (&kernelContext_133)->collider_info_0 = collider_info_33;
    (&kernelContext_133)->view_chunks_0 = view_chunks_33;
    (&kernelContext_133)->view_0 = view_33;
    (&kernelContext_133)->objects_0 = objects_33;
    (&kernelContext_133)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_133)->lines_0 = lines_33;
    (&kernelContext_133)->geometry_0 = geometry_33;
    (&kernelContext_133)->shots_0 = shots_33;
    (&kernelContext_133)->lp_0 = lp_33;
    (&kernelContext_133)->ext_0 = ext_35;
    (&kernelContext_133)->lp_sides_0 = lp_sides_33;
    (&kernelContext_133)->engine_counters_0 = engine_counters_33;
    (&kernelContext_133)->engine_contacts_0 = engine_contacts_33;
    (&kernelContext_133)->cs_anchors_0 = cs_anchors_33;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_133)->g_scan_0 = &g_scan_33;
    threadgroup array<float, int(64)> g_il_best_33;
    (&kernelContext_133)->g_il_best_0 = &g_il_best_33;
    uint k_39 = id_30.x;
    bool _S1033;
    if(k_39 >= (params_33->chunk_count_0))
    {
        _S1033 = true;
    }
    else
    {
        _S1033 = ((float4(*((&kernelContext_133)->lp_0+lp_f_0(k_39))) ).w) != 0.0f;
    }
    if(_S1033)
    {
        return;
    }
    bool _S1034 = lp_factor_0(k_39, &kernelContext_133);
    if(!_S1034)
    {
        (*((&kernelContext_133)->lp_0+lp_f_0(k_39)))[int(3)] = 1.0f;
        uint _S1035 = a_counters_1(&kernelContext_133);
        uint _S1036 = atomic_fetch_or_explicit((&kernelContext_133)->atom_0+(_S1035 + 3U), 32U, memory_order_relaxed);
    }
    return;
}

uint st_x_0(uint k_40, uint state_1, uint copy_0)
{
    return 22U * k_40 + state_1 + 2U * copy_0;
}

uint st_d_0(uint k_41, uint state_2)
{
    return 22U * k_41 + state_2 + 4U;
}

void st_sweep_0(uint k_42, uint state_3, uint from_2, float c1_1, float c2_1, KernelContext_0 thread* kernelContext_134)
{
    uint ka_0;
    uint kb_0;
    float s_14;
    uint to_1 = 1U - from_2;
    uint _S1037 = st_x_0(k_42, state_3, from_2);
    float4 _S1038 = float4(*(kernelContext_134->lp_0+_S1037)) ;
    float4 _S1039 = float4(*(kernelContext_134->lp_0+(_S1037 + 1U))) ;
    uint _S1040 = st_f_0(k_42, state_3);
    float4 _S1041 = float4(*(kernelContext_134->lp_0+_S1040)) ;
    if((_S1041.w) != 0.0f)
    {
        uint _S1042 = st_x_0(k_42, state_3, to_1);
        *(kernelContext_134->lp_0+_S1042) = packed_float4(_S1038) ;
        *(kernelContext_134->lp_0+(_S1042 + 1U)) = packed_float4(_S1039) ;
        uint _S1043 = st_d_0(k_42, state_3);
        packed_float4 _S1044 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_134->lp_0+_S1043) = _S1044;
        *(kernelContext_134->lp_0+(_S1043 + 1U)) = _S1044;
        return;
    }
    float _S1045 = _S1041.y;
    float _S1046 = _S1041.z;
    thread array<float, int(6)> r_24;
    r_24[int(0)] = _S1041.x;
    r_24[int(1)] = _S1045;
    r_24[int(2)] = _S1046;
    r_24[int(3)] = 0.0f;
    r_24[int(4)] = 0.0f;
    r_24[int(5)] = 0.0f;
    float3 t_12 = (float4(*(kernelContext_134->lp_0+(_S1040 + 1U))) ).xyz;
    r_24[int(3)] = t_12.x;
    r_24[int(4)] = t_12.y;
    r_24[int(5)] = t_12.z;
    float4 _S1047 = float4(*(kernelContext_134->ext_0+(kernelContext_134->params_0->ext_chunks_0 + 4U * k_42))) ;
    uint _S1048 = (as_type<uint>((_S1047.x)));
    uint _S1049 = (as_type<uint>((_S1047.y)));
    uint e_5 = 0U;
    for(;;)
    {
        if(e_5 < _S1049)
        {
        }
        else
        {
            break;
        }
        uint _S1050 = 5U * (_S1048 + e_5);
        float4 _S1051 = float4(*(kernelContext_134->lp_sides_0+_S1050)) ;
        uint _S1052 = w_alive_0(kernelContext_134);
        if((*(kernelContext_134->work_0+(_S1052 + (as_type<uint>((_S1051.w)))))) == 0U)
        {
            e_5 = e_5 + 1U;
            continue;
        }
        float4 _S1053 = float4(*(kernelContext_134->lp_sides_0+(_S1050 + 1U))) ;
        float4 _S1054 = float4(*(kernelContext_134->lp_sides_0+(_S1050 + 2U))) ;
        float4 _S1055 = float4(*(kernelContext_134->lp_sides_0+(_S1050 + 3U))) ;
        float4 _S1056 = float4(*(kernelContext_134->lp_sides_0+(_S1050 + 4U))) ;
        bool first_side_0 = (_S1056.w) != 0.0f;
        if(first_side_0)
        {
            ka_0 = k_42;
        }
        else
        {
            ka_0 = (as_type<uint>((_S1053.w)));
        }
        if(first_side_0)
        {
            kb_0 = (as_type<uint>((_S1053.w)));
        }
        else
        {
            kb_0 = k_42;
        }
        float3 ra_0 = _S1051.xyz;
        float3 rb_0 = _S1053.xyz;
        float3 t1_3 = _S1054.xyz;
        float3 n_5 = _S1055.xyz;
        float3 t2_1 = cross(n_5, t1_3);
        uint _S1057 = st_x_0(ka_0, state_3, from_2);
        uint _S1058 = st_x_0(kb_0, state_3, from_2);
        float3 _S1059 = (float4(*(kernelContext_134->lp_0+(_S1058 + 1U))) ).xyz;
        float3 _S1060 = (float4(*(kernelContext_134->lp_0+(_S1057 + 1U))) ).xyz;
        float3 delta_0 = (float4(*(kernelContext_134->lp_0+_S1058)) ).xyz + cross(_S1059, rb_0) - ((float4(*(kernelContext_134->lp_0+_S1057)) ).xyz + cross(_S1060, ra_0));
        float3 phi_0 = _S1059 - _S1060;
        float _S1061 = _S1054.w;
        float3 fb_0 = t1_3 * float3((_S1061 * dot(delta_0, t1_3)))  + t2_1 * float3((_S1061 * dot(delta_0, t2_1)))  + n_5 * float3((_S1055.w * dot(delta_0, n_5))) ;
        float3 mb_0 = t1_3 * float3((_S1056.x * dot(phi_0, t1_3)))  + t2_1 * float3((_S1056.y * dot(phi_0, t2_1)))  + n_5 * float3((_S1056.z * dot(phi_0, n_5))) ;
        float3 arm_0;
        if(first_side_0)
        {
            arm_0 = ra_0;
        }
        else
        {
            arm_0 = rb_0;
        }
        float3 lf_0;
        if(first_side_0)
        {
            lf_0 = fb_0;
        }
        else
        {
            lf_0 = - fb_0;
        }
        float3 _S1062;
        if(first_side_0)
        {
            _S1062 = mb_0;
        }
        else
        {
            _S1062 = - mb_0;
        }
        float3 lm_0 = _S1062 + cross(arm_0, lf_0);
        r_24[int(0)] = r_24[int(0)] + lf_0.x;
        r_24[int(1)] = r_24[int(1)] + lf_0.y;
        r_24[int(2)] = r_24[int(2)] + lf_0.z;
        r_24[int(3)] = r_24[int(3)] + lm_0.x;
        r_24[int(4)] = r_24[int(4)] + lm_0.y;
        r_24[int(5)] = r_24[int(5)] + lm_0.z;
        e_5 = e_5 + 1U;
    }
    thread array<float, int(21)> l_1;
    ka_0 = 0U;
    for(;;)
    {
        if(ka_0 < 21U)
        {
        }
        else
        {
            break;
        }
        l_1[ka_0] = 0.0f;
        ka_0 = ka_0 + 1U;
    }
    kb_0 = 0U;
    for(;;)
    {
        if(kb_0 < 6U)
        {
        }
        else
        {
            break;
        }
        float4 _S1063 = float4(*(kernelContext_134->lp_0+(lp_l_0(k_42) + kb_0))) ;
        uint _S1064 = 4U * kb_0;
        l_1[_S1064] = _S1063.x;
        uint _S1065 = _S1064 + 1U;
        if(_S1065 < 21U)
        {
            l_1[_S1065] = _S1063.y;
        }
        uint _S1066 = _S1064 + 2U;
        if(_S1066 < 21U)
        {
            l_1[_S1066] = _S1063.z;
        }
        uint _S1067 = _S1064 + 3U;
        if(_S1067 < 21U)
        {
            l_1[_S1067] = _S1063.w;
        }
        kb_0 = kb_0 + 1U;
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
        uint p_17 = 0U;
        s_14 = r_24[i2_0];
        for(;;)
        {
            if(p_17 < i2_0)
            {
            }
            else
            {
                break;
            }
            float s_15 = s_14 - l_1[tri_0(i2_0, p_17)] * y_2[p_17];
            p_17 = p_17 + 1U;
            s_14 = s_15;
        }
        y_2[i2_0] = s_14 / l_1[tri_0(i2_0, i2_0)];
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
        s_14 = y_2[i3_0];
        for(;;)
        {
            if(p2_1 < int(6))
            {
            }
            else
            {
                break;
            }
            float s_16 = s_14 - l_1[tri_0(uint(p2_1), uint(i3_0))] * z_3[p2_1];
            p2_1 = p2_1 + int(1);
            s_14 = s_16;
        }
        uint _S1068 = uint(i3_0);
        z_3[i3_0] = s_14 / l_1[tri_0(_S1068, _S1068)];
        i3_0 = i3_0 - int(1);
    }
    uint _S1069 = st_d_0(k_42, state_3);
    float4 _S1070 = float4(*(kernelContext_134->lp_0+_S1069)) ;
    thread float4 d0_0 = _S1070;
    uint _S1071 = _S1069 + 1U;
    float4 _S1072 = float4(*(kernelContext_134->lp_0+_S1071)) ;
    thread float4 d1_0 = _S1072;
    float3 _S1073 = float3(c1_1) ;
    float3 _S1074 = float3(c2_1) ;
    d0_0.xyz = _S1073 * _S1070.xyz + _S1074 * float3(z_3[int(0)], z_3[int(1)], z_3[int(2)]);
    d1_0.xyz = _S1073 * _S1072.xyz + _S1074 * float3(z_3[int(3)], z_3[int(4)], z_3[int(5)]);
    *(kernelContext_134->lp_0+_S1069) = packed_float4(d0_0) ;
    *(kernelContext_134->lp_0+_S1071) = packed_float4(d1_0) ;
    uint _S1075 = st_x_0(k_42, state_3, to_1);
    *(kernelContext_134->lp_0+_S1075) = packed_float4(float4(_S1038.xyz + d0_0.xyz, 0.0f)) ;
    *(kernelContext_134->lp_0+(_S1075 + 1U)) = packed_float4(float4(_S1039.xyz + d1_0.xyz, 0.0f)) ;
    return;
}

void lp_sweep_0(uint k_43, uint from_3, KernelContext_0 thread* kernelContext_135)
{
    st_sweep_0(k_43, 0U, from_3, kernelContext_135->params_0->lp_c1_0, kernelContext_135->params_0->lp_c2_0, kernelContext_135);
    return;
}

[[kernel]] void fr_lp_sweep_even(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], atomic<uint> device* atom_34 [[buffer(5)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], packed_float4 device* lp_34 [[buffer(24)]], packed_float4 device* ext_36 [[buffer(23)]], packed_float4 device* lp_sides_34 [[buffer(28)]], uint device* engine_counters_34 [[buffer(26)]], packed_float4 device* engine_contacts_34 [[buffer(25)]], packed_float4 device* cs_anchors_34 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_136;
    (&kernelContext_136)->params_0 = params_34;
    (&kernelContext_136)->work_0 = work_34;
    (&kernelContext_136)->frames_0 = frames_34;
    (&kernelContext_136)->bodies_0 = bodies_34;
    (&kernelContext_136)->chunks_0 = chunks_34;
    (&kernelContext_136)->sort_0 = sort_34;
    (&kernelContext_136)->atom_0 = atom_34;
    (&kernelContext_136)->manifolds_0 = manifolds_34;
    (&kernelContext_136)->loads_0 = loads_34;
    (&kernelContext_136)->bonds_0 = bonds_34;
    (&kernelContext_136)->args_0 = args_34;
    (&kernelContext_136)->info_0 = info_34;
    (&kernelContext_136)->scan_0 = scan_34;
    (&kernelContext_136)->parts_0 = parts_34;
    (&kernelContext_136)->colliders_0 = colliders_34;
    (&kernelContext_136)->collider_info_0 = collider_info_34;
    (&kernelContext_136)->view_chunks_0 = view_chunks_34;
    (&kernelContext_136)->view_0 = view_34;
    (&kernelContext_136)->objects_0 = objects_34;
    (&kernelContext_136)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_136)->lines_0 = lines_34;
    (&kernelContext_136)->geometry_0 = geometry_34;
    (&kernelContext_136)->shots_0 = shots_34;
    (&kernelContext_136)->lp_0 = lp_34;
    (&kernelContext_136)->ext_0 = ext_36;
    (&kernelContext_136)->lp_sides_0 = lp_sides_34;
    (&kernelContext_136)->engine_counters_0 = engine_counters_34;
    (&kernelContext_136)->engine_contacts_0 = engine_contacts_34;
    (&kernelContext_136)->cs_anchors_0 = cs_anchors_34;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_136)->g_scan_0 = &g_scan_34;
    threadgroup array<float, int(64)> g_il_best_34;
    (&kernelContext_136)->g_il_best_0 = &g_il_best_34;
    uint _S1076 = id_31.x;
    if(_S1076 < (params_34->chunk_count_0))
    {
        lp_sweep_0(_S1076, 0U, &kernelContext_136);
    }
    return;
}

[[kernel]] void fr_lp_sweep_odd(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], atomic<uint> device* atom_35 [[buffer(5)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], packed_float4 device* lp_35 [[buffer(24)]], packed_float4 device* ext_37 [[buffer(23)]], packed_float4 device* lp_sides_35 [[buffer(28)]], uint device* engine_counters_35 [[buffer(26)]], packed_float4 device* engine_contacts_35 [[buffer(25)]], packed_float4 device* cs_anchors_35 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_137;
    (&kernelContext_137)->params_0 = params_35;
    (&kernelContext_137)->work_0 = work_35;
    (&kernelContext_137)->frames_0 = frames_35;
    (&kernelContext_137)->bodies_0 = bodies_35;
    (&kernelContext_137)->chunks_0 = chunks_35;
    (&kernelContext_137)->sort_0 = sort_35;
    (&kernelContext_137)->atom_0 = atom_35;
    (&kernelContext_137)->manifolds_0 = manifolds_35;
    (&kernelContext_137)->loads_0 = loads_35;
    (&kernelContext_137)->bonds_0 = bonds_35;
    (&kernelContext_137)->args_0 = args_35;
    (&kernelContext_137)->info_0 = info_35;
    (&kernelContext_137)->scan_0 = scan_35;
    (&kernelContext_137)->parts_0 = parts_35;
    (&kernelContext_137)->colliders_0 = colliders_35;
    (&kernelContext_137)->collider_info_0 = collider_info_35;
    (&kernelContext_137)->view_chunks_0 = view_chunks_35;
    (&kernelContext_137)->view_0 = view_35;
    (&kernelContext_137)->objects_0 = objects_35;
    (&kernelContext_137)->shot_bodies_0 = shot_bodies_35;
    (&kernelContext_137)->lines_0 = lines_35;
    (&kernelContext_137)->geometry_0 = geometry_35;
    (&kernelContext_137)->shots_0 = shots_35;
    (&kernelContext_137)->lp_0 = lp_35;
    (&kernelContext_137)->ext_0 = ext_37;
    (&kernelContext_137)->lp_sides_0 = lp_sides_35;
    (&kernelContext_137)->engine_counters_0 = engine_counters_35;
    (&kernelContext_137)->engine_contacts_0 = engine_contacts_35;
    (&kernelContext_137)->cs_anchors_0 = cs_anchors_35;
    threadgroup array<uint, int(1024)> g_scan_35;
    (&kernelContext_137)->g_scan_0 = &g_scan_35;
    threadgroup array<float, int(64)> g_il_best_35;
    (&kernelContext_137)->g_il_best_0 = &g_il_best_35;
    uint _S1077 = id_32.x;
    if(_S1077 < (params_35->chunk_count_0))
    {
        lp_sweep_0(_S1077, 1U, &kernelContext_137);
    }
    return;
}

void st_force_0(const LpBond_0 thread* b_45, uint ka_1, uint kb_1, uint state_4, uint c_20, float3 thread* q_lin_0, float3 thread* q_ang_0, KernelContext_0 thread* kernelContext_138)
{
    uint _S1078 = st_x_0(ka_1, state_4, c_20);
    uint _S1079 = st_x_0(kb_1, state_4, c_20);
    float3 _S1080 = (float4(*(kernelContext_138->lp_0+(_S1079 + 1U))) ).xyz;
    float3 _S1081 = (float4(*(kernelContext_138->lp_0+(_S1078 + 1U))) ).xyz;
    float3 delta_1 = (float4(*(kernelContext_138->lp_0+_S1079)) ).xyz + cross(_S1080, b_45->centroid_0 - (float4(*(kernelContext_138->chunks_0+4U * kb_1)) ).xyz) - ((float4(*(kernelContext_138->lp_0+_S1078)) ).xyz + cross(_S1081, b_45->centroid_0 - (float4(*(kernelContext_138->chunks_0+4U * ka_1)) ).xyz));
    float3 phi_1 = _S1080 - _S1081;
    float4 _S1082 = (&b_45->law_0)->stiff0_0;
    float kt_0 = (&b_45->law_0)->stiff1_0.x;
    float _S1083 = (&b_45->law_0)->stiff0_0.y;
    float3 _S1084 = b_45->t1_2;
    float3 _S1085 = b_45->t2_0;
    float3 _S1086 = b_45->n_4;
    *q_lin_0 = float3(_S1083 * dot(delta_1, b_45->t1_2), _S1083 * dot(delta_1, b_45->t2_0), (&b_45->law_0)->stiff0_0.x * dot(delta_1, b_45->n_4));
    *q_ang_0 = float3(_S1082.z * dot(phi_1, _S1084), _S1082.w * dot(phi_1, _S1085), kt_0 * dot(phi_1, _S1086));
    return;
}

void st_force_1(const LpBond_0 thread* b_46, uint ka_2, uint kb_2, uint state_5, uint c_21, float3 thread* q_lin_1, float3 thread* q_ang_1, KernelContext_0 thread* kernelContext_139)
{
    uint _S1087 = st_x_0(ka_2, state_5, c_21);
    uint _S1088 = st_x_0(kb_2, state_5, c_21);
    float3 _S1089 = (float4(*(kernelContext_139->lp_0+(_S1088 + 1U))) ).xyz;
    float3 _S1090 = (float4(*(kernelContext_139->lp_0+(_S1087 + 1U))) ).xyz;
    float3 delta_2 = (float4(*(kernelContext_139->lp_0+_S1088)) ).xyz + cross(_S1089, b_46->centroid_0 - (float4(*(kernelContext_139->chunks_0+4U * kb_2)) ).xyz) - ((float4(*(kernelContext_139->lp_0+_S1087)) ).xyz + cross(_S1090, b_46->centroid_0 - (float4(*(kernelContext_139->chunks_0+4U * ka_2)) ).xyz));
    float3 phi_2 = _S1089 - _S1090;
    float4 _S1091 = (&b_46->law_0)->stiff0_0;
    float kt_1 = (&b_46->law_0)->stiff1_0.x;
    float _S1092 = (&b_46->law_0)->stiff0_0.y;
    float3 _S1093 = b_46->t1_2;
    float3 _S1094 = b_46->t2_0;
    float3 _S1095 = b_46->n_4;
    *q_lin_1 = float3(_S1092 * dot(delta_2, b_46->t1_2), _S1092 * dot(delta_2, b_46->t2_0), (&b_46->law_0)->stiff0_0.x * dot(delta_2, b_46->n_4));
    *q_ang_1 = float3(_S1091.z * dot(phi_2, _S1093), _S1091.w * dot(phi_2, _S1094), kt_1 * dot(phi_2, _S1095));
    return;
}

void lp_force_0(const LpBond_0 thread* b_47, uint ka_3, uint kb_3, uint c_22, float3 thread* q_lin_2, float3 thread* q_ang_2, KernelContext_0 thread* kernelContext_140)
{
    st_force_1(b_47, ka_3, kb_3, 0U, c_22, q_lin_2, q_ang_2, kernelContext_140);
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

Measures_0 stress_measures_0(const JointBond_0 thread* b_48, float3 q_lin_3, float3 q_ang_3)
{
    float area_0 = b_48->geom0_0.x;
    float _S1096 = q_lin_3.z;
    float axial_0 = (metal::fast::divide((_S1096), (area_0)));
    float _S1097 = (metal::fast::divide((abs(q_ang_3.x)), (b_48->geom1_0.x)));
    float _S1098 = (metal::fast::divide((abs(q_ang_3.y)), (b_48->geom1_0.y)));
    float bending_0 = _S1097 + _S1098;
    float _S1099 = q_lin_3.x;
    float _S1100 = q_lin_3.y;
    float _S1101 = (metal::fast::sqrt((_S1099 * _S1099 + _S1100 * _S1100)));
    float _S1102 = (metal::fast::divide((_S1101), (area_0)));
    float _S1103 = (metal::fast::divide((abs(q_ang_3.z)), (b_48->geom0_0.w)));
    float shear_1 = _S1102 + _S1103;
    thread Measures_0 m_3;
    (&m_3)->tension_0 = axial_0 + bending_0;
    (&m_3)->shear_0 = shear_1;
    float _S1104 = - axial_0;
    (&m_3)->normal_compression_0 = max(_S1104, 0.0f);
    (&m_3)->compression_0 = _S1104 + bending_0;
    (&m_3)->compressive_force_0 = max(- _S1096, 0.0f);
    return m_3;
}

JointMaterial_0 lp_material_0(uint m_4, KernelContext_0 thread* kernelContext_141)
{
    uint o_8 = kernelContext_141->params_0->ext_mat_0 + 5U * m_4;
    thread JointMaterial_0 j_8;
    (&j_8)->strength_0 = float4(*(kernelContext_141->ext_0+o_8)) ;
    (&j_8)->energy_0 = float4(*(kernelContext_141->ext_0+(o_8 + 1U))) ;
    (&j_8)->dif_0 = float4(*(kernelContext_141->ext_0+(o_8 + 2U))) ;
    (&j_8)->misc_0 = float4(*(kernelContext_141->ext_0+(o_8 + 3U))) ;
    (&j_8)->kind_flags_0 = (as_type<uint4>((float4(*(kernelContext_141->ext_0+(o_8 + 4U))) )));
    return j_8;
}

float infinity_0()
{
    return float_bits_0(2139095040U);
}

float4 failure_indices_0(const JointMaterial_0 thread* mat_1, const JointBond_0 thread* b_49, const Measures_0 thread* m_5, float multiplier_0)
{
    float fc_2 = mat_1->strength_0.y * multiplier_0;
    float _S1105 = min(mat_1->strength_0.z * multiplier_0 + mat_1->strength_0.w * m_5->normal_compression_0, mat_1->energy_0.x * multiplier_0);
    thread float4 idx_0;
    float _S1106 = (metal::fast::divide((m_5->tension_0), (mat_1->strength_0.x * multiplier_0)));
    idx_0.x = max(_S1106, 0.0f);
    float _S1107;
    if(_S1105 > 0.0f)
    {
        float _S1108 = (metal::fast::divide((m_5->shear_0), (_S1105)));
        _S1107 = _S1108;
    }
    else
    {
        _S1107 = infinity_0();
    }
    idx_0.y = _S1107;
    float _S1109 = (metal::fast::divide((m_5->compression_0), (fc_2)));
    idx_0.z = max(_S1109, 0.0f);
    float _S1110 = b_49->stiff1_0.y;
    if(_S1110 > 0.0f)
    {
        float _S1111 = (metal::fast::divide((m_5->compressive_force_0), (_S1110)));
        _S1107 = _S1111;
    }
    else
    {
        _S1107 = 0.0f;
    }
    idx_0.w = _S1107;
    return idx_0;
}

[[kernel]] void fr_lp_breaks(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], atomic<uint> device* atom_36 [[buffer(5)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], packed_float4 device* lp_36 [[buffer(24)]], packed_float4 device* ext_38 [[buffer(23)]], packed_float4 device* lp_sides_36 [[buffer(28)]], uint device* engine_counters_36 [[buffer(26)]], packed_float4 device* engine_contacts_36 [[buffer(25)]], packed_float4 device* cs_anchors_36 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_142;
    (&kernelContext_142)->params_0 = params_36;
    (&kernelContext_142)->work_0 = work_36;
    (&kernelContext_142)->frames_0 = frames_36;
    (&kernelContext_142)->bodies_0 = bodies_36;
    (&kernelContext_142)->chunks_0 = chunks_36;
    (&kernelContext_142)->sort_0 = sort_36;
    (&kernelContext_142)->atom_0 = atom_36;
    (&kernelContext_142)->manifolds_0 = manifolds_36;
    (&kernelContext_142)->loads_0 = loads_36;
    (&kernelContext_142)->bonds_0 = bonds_36;
    (&kernelContext_142)->args_0 = args_36;
    (&kernelContext_142)->info_0 = info_36;
    (&kernelContext_142)->scan_0 = scan_36;
    (&kernelContext_142)->parts_0 = parts_36;
    (&kernelContext_142)->colliders_0 = colliders_36;
    (&kernelContext_142)->collider_info_0 = collider_info_36;
    (&kernelContext_142)->view_chunks_0 = view_chunks_36;
    (&kernelContext_142)->view_0 = view_36;
    (&kernelContext_142)->objects_0 = objects_36;
    (&kernelContext_142)->shot_bodies_0 = shot_bodies_36;
    (&kernelContext_142)->lines_0 = lines_36;
    (&kernelContext_142)->geometry_0 = geometry_36;
    (&kernelContext_142)->shots_0 = shots_36;
    (&kernelContext_142)->lp_0 = lp_36;
    (&kernelContext_142)->ext_0 = ext_38;
    (&kernelContext_142)->lp_sides_0 = lp_sides_36;
    (&kernelContext_142)->engine_counters_0 = engine_counters_36;
    (&kernelContext_142)->engine_contacts_0 = engine_contacts_36;
    (&kernelContext_142)->cs_anchors_0 = cs_anchors_36;
    threadgroup array<uint, int(1024)> g_scan_36;
    (&kernelContext_142)->g_scan_0 = &g_scan_36;
    threadgroup array<float, int(64)> g_il_best_36;
    (&kernelContext_142)->g_il_best_0 = &g_il_best_36;
    uint i_14 = id_33.x;
    bool _S1112;
    if(i_14 >= (params_36->bond_count_0))
    {
        _S1112 = true;
    }
    else
    {
        uint _S1113 = w_alive_0(&kernelContext_142);
        _S1112 = (*((&kernelContext_142)->work_0+(_S1113 + i_14))) == 0U;
    }
    if(_S1112)
    {
        return;
    }
    uint4 _S1114 = uint4(*((&kernelContext_142)->bonds_0+i_14)) ;
    LpBond_0 _S1115 = lp_bond_0(i_14, &kernelContext_142);
    uint _S1116 = _S1114.x;
    uint _S1117 = _S1114.y;
    thread LpBond_0 _S1118 = _S1115;
    thread float3 q_lin_4;
    thread float3 q_ang_4;
    lp_force_0(&_S1118, _S1116, _S1117, 0U, &q_lin_4, &q_ang_4, &kernelContext_142);
    thread JointBond_0 _S1119 = _S1115.law_0;
    Measures_0 _S1120 = stress_measures_0(&_S1119, q_lin_4, q_ang_4);
    JointMaterial_0 _S1121 = lp_material_0(_S1115.law_0.ids_3.x, &kernelContext_142);
    float _S1122 = _S1115.law_0.geom1_0.w;
    thread JointMaterial_0 _S1123 = _S1121;
    thread JointBond_0 _S1124 = _S1115.law_0;
    thread Measures_0 _S1125 = _S1120;
    float4 _S1126 = failure_indices_0(&_S1123, &_S1124, &_S1125, _S1122);
    float _S1127 = max(max(_S1126.x, _S1126.y), max(_S1126.z, _S1126.w));
    uint _S1128 = a_counters_1(&kernelContext_142);
    uint _S1129 = atomic_fetch_max_explicit((&kernelContext_142)->atom_0+(_S1128 + 11U), (as_type<uint>((max(_S1127, 0.0f)))), memory_order_relaxed);
    if(_S1127 >= 1.0f)
    {
        uint _S1130 = w_alive_0(&kernelContext_142);
        *((&kernelContext_142)->work_0+(_S1130 + i_14)) = 0U;
        uint _S1131 = a_dirty_0(&kernelContext_142);
        uint _S1132 = body_of_0(_S1116, &kernelContext_142);
        uint _S1133 = atomic_fetch_or_explicit((&kernelContext_142)->atom_0+(_S1131 + _S1132), 1U, memory_order_relaxed);
        uint _S1134 = atomic_fetch_add_explicit((&kernelContext_142)->atom_0+_S1128, 1U, memory_order_relaxed);
        uint _S1135 = atomic_fetch_add_explicit((&kernelContext_142)->atom_0+(_S1128 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first_reset(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], atomic<uint> device* atom_37 [[buffer(5)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], packed_float4 device* lp_37 [[buffer(24)]], packed_float4 device* ext_39 [[buffer(23)]], packed_float4 device* lp_sides_37 [[buffer(28)]], uint device* engine_counters_37 [[buffer(26)]], packed_float4 device* engine_contacts_37 [[buffer(25)]], packed_float4 device* cs_anchors_37 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_143;
    (&kernelContext_143)->params_0 = params_37;
    (&kernelContext_143)->work_0 = work_37;
    (&kernelContext_143)->frames_0 = frames_37;
    (&kernelContext_143)->bodies_0 = bodies_37;
    (&kernelContext_143)->chunks_0 = chunks_37;
    (&kernelContext_143)->sort_0 = sort_37;
    (&kernelContext_143)->atom_0 = atom_37;
    (&kernelContext_143)->manifolds_0 = manifolds_37;
    (&kernelContext_143)->loads_0 = loads_37;
    (&kernelContext_143)->bonds_0 = bonds_37;
    (&kernelContext_143)->args_0 = args_37;
    (&kernelContext_143)->info_0 = info_37;
    (&kernelContext_143)->scan_0 = scan_37;
    (&kernelContext_143)->parts_0 = parts_37;
    (&kernelContext_143)->colliders_0 = colliders_37;
    (&kernelContext_143)->collider_info_0 = collider_info_37;
    (&kernelContext_143)->view_chunks_0 = view_chunks_37;
    (&kernelContext_143)->view_0 = view_37;
    (&kernelContext_143)->objects_0 = objects_37;
    (&kernelContext_143)->shot_bodies_0 = shot_bodies_37;
    (&kernelContext_143)->lines_0 = lines_37;
    (&kernelContext_143)->geometry_0 = geometry_37;
    (&kernelContext_143)->shots_0 = shots_37;
    (&kernelContext_143)->lp_0 = lp_37;
    (&kernelContext_143)->ext_0 = ext_39;
    (&kernelContext_143)->lp_sides_0 = lp_sides_37;
    (&kernelContext_143)->engine_counters_0 = engine_counters_37;
    (&kernelContext_143)->engine_contacts_0 = engine_contacts_37;
    (&kernelContext_143)->cs_anchors_0 = cs_anchors_37;
    threadgroup array<uint, int(1024)> g_scan_37;
    (&kernelContext_143)->g_scan_0 = &g_scan_37;
    threadgroup array<float, int(64)> g_il_best_37;
    (&kernelContext_143)->g_il_best_0 = &g_il_best_37;
    uint _S1136 = id_34.x;
    if(_S1136 == 0U)
    {
        uint _S1137 = a_il_cursor_1(&kernelContext_143);
        atomic_store_explicit((&kernelContext_143)->atom_0+_S1137, 0U, memory_order_relaxed);
    }
    if(_S1136 < ((&kernelContext_143)->params_0->shot_count_0))
    {
        uint _S1138 = a_first_1(_S1136, &kernelContext_143);
        atomic_store_explicit((&kernelContext_143)->atom_0+_S1138, 2139095040U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], atomic<uint> device* atom_38 [[buffer(5)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], packed_float4 device* lp_38 [[buffer(24)]], packed_float4 device* ext_40 [[buffer(23)]], packed_float4 device* lp_sides_38 [[buffer(28)]], uint device* engine_counters_38 [[buffer(26)]], packed_float4 device* engine_contacts_38 [[buffer(25)]], packed_float4 device* cs_anchors_38 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_144;
    (&kernelContext_144)->params_0 = params_38;
    (&kernelContext_144)->work_0 = work_38;
    (&kernelContext_144)->frames_0 = frames_38;
    (&kernelContext_144)->bodies_0 = bodies_38;
    (&kernelContext_144)->chunks_0 = chunks_38;
    (&kernelContext_144)->sort_0 = sort_38;
    (&kernelContext_144)->atom_0 = atom_38;
    (&kernelContext_144)->manifolds_0 = manifolds_38;
    (&kernelContext_144)->loads_0 = loads_38;
    (&kernelContext_144)->bonds_0 = bonds_38;
    (&kernelContext_144)->args_0 = args_38;
    (&kernelContext_144)->info_0 = info_38;
    (&kernelContext_144)->scan_0 = scan_38;
    (&kernelContext_144)->parts_0 = parts_38;
    (&kernelContext_144)->colliders_0 = colliders_38;
    (&kernelContext_144)->collider_info_0 = collider_info_38;
    (&kernelContext_144)->view_chunks_0 = view_chunks_38;
    (&kernelContext_144)->view_0 = view_38;
    (&kernelContext_144)->objects_0 = objects_38;
    (&kernelContext_144)->shot_bodies_0 = shot_bodies_38;
    (&kernelContext_144)->lines_0 = lines_38;
    (&kernelContext_144)->geometry_0 = geometry_38;
    (&kernelContext_144)->shots_0 = shots_38;
    (&kernelContext_144)->lp_0 = lp_38;
    (&kernelContext_144)->ext_0 = ext_40;
    (&kernelContext_144)->lp_sides_0 = lp_sides_38;
    (&kernelContext_144)->engine_counters_0 = engine_counters_38;
    (&kernelContext_144)->engine_contacts_0 = engine_contacts_38;
    (&kernelContext_144)->cs_anchors_0 = cs_anchors_38;
    threadgroup array<uint, int(1024)> g_scan_38;
    (&kernelContext_144)->g_scan_0 = &g_scan_38;
    threadgroup array<float, int(64)> g_il_best_38;
    (&kernelContext_144)->g_il_best_0 = &g_il_best_38;
    uint b_50 = id_35.x;
    bool _S1139;
    if(b_50 >= (params_38->body_count_0))
    {
        _S1139 = true;
    }
    else
    {
        _S1139 = ((&kernelContext_144)->params_0->shot_count_0) == 0U;
    }
    if(_S1139)
    {
        return;
    }
    thread uint shot_9;
    bool _S1140 = struck_0(b_50, &shot_9, &kernelContext_144);
    if(!_S1140)
    {
        return;
    }
    uint _S1141 = a_first_1(shot_9, &kernelContext_144);
    atomic<uint> device* _S1142 = (&kernelContext_144)->atom_0+_S1141;
    uint _S1143 = a_strike_0(&kernelContext_144);
    uint _S1144 = atomic_load_explicit((&kernelContext_144)->atom_0+(_S1143 + 9U * b_50 + 6U), memory_order_relaxed);
    uint _S1145 = atomic_fetch_min_explicit(_S1142, _S1144, memory_order_relaxed);
    return;
}

bool cs_struck_1(uint b_51, KernelContext_0 thread* kernelContext_145)
{
    bool _S1146;
    if(b_51 < (kernelContext_145->params_0->body_count_0))
    {
        uint _S1147 = a_cs_0(b_51, kernelContext_145);
        uint _S1148 = atomic_load_explicit(kernelContext_145->atom_0+(_S1147 + 1U), memory_order_relaxed);
        _S1146 = _S1148 != 0U;
    }
    else
    {
        _S1146 = false;
    }
    return _S1146;
}

[[kernel]] void fr_il_begin(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], atomic<uint> device* atom_39 [[buffer(5)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], packed_float4 device* lp_39 [[buffer(24)]], packed_float4 device* ext_41 [[buffer(23)]], packed_float4 device* lp_sides_39 [[buffer(28)]], uint device* engine_counters_39 [[buffer(26)]], packed_float4 device* engine_contacts_39 [[buffer(25)]], packed_float4 device* cs_anchors_39 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_146;
    (&kernelContext_146)->params_0 = params_39;
    (&kernelContext_146)->work_0 = work_39;
    (&kernelContext_146)->frames_0 = frames_39;
    (&kernelContext_146)->bodies_0 = bodies_39;
    (&kernelContext_146)->chunks_0 = chunks_39;
    (&kernelContext_146)->sort_0 = sort_39;
    (&kernelContext_146)->atom_0 = atom_39;
    (&kernelContext_146)->manifolds_0 = manifolds_39;
    (&kernelContext_146)->loads_0 = loads_39;
    (&kernelContext_146)->bonds_0 = bonds_39;
    (&kernelContext_146)->args_0 = args_39;
    (&kernelContext_146)->info_0 = info_39;
    (&kernelContext_146)->scan_0 = scan_39;
    (&kernelContext_146)->parts_0 = parts_39;
    (&kernelContext_146)->colliders_0 = colliders_39;
    (&kernelContext_146)->collider_info_0 = collider_info_39;
    (&kernelContext_146)->view_chunks_0 = view_chunks_39;
    (&kernelContext_146)->view_0 = view_39;
    (&kernelContext_146)->objects_0 = objects_39;
    (&kernelContext_146)->shot_bodies_0 = shot_bodies_39;
    (&kernelContext_146)->lines_0 = lines_39;
    (&kernelContext_146)->geometry_0 = geometry_39;
    (&kernelContext_146)->shots_0 = shots_39;
    (&kernelContext_146)->lp_0 = lp_39;
    (&kernelContext_146)->ext_0 = ext_41;
    (&kernelContext_146)->lp_sides_0 = lp_sides_39;
    (&kernelContext_146)->engine_counters_0 = engine_counters_39;
    (&kernelContext_146)->engine_contacts_0 = engine_contacts_39;
    (&kernelContext_146)->cs_anchors_0 = cs_anchors_39;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_146)->g_scan_0 = &g_scan_39;
    threadgroup array<float, int(64)> g_il_best_39;
    (&kernelContext_146)->g_il_best_0 = &g_il_best_39;
    if((params_39->shot_count_0) > 0U)
    {
        il_begin_1(id_36.x, &kernelContext_146);
    }
    return;
}

[[kernel]] void fr_il_tube_count(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], atomic<uint> device* atom_40 [[buffer(5)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], packed_float4 device* lp_40 [[buffer(24)]], packed_float4 device* ext_42 [[buffer(23)]], packed_float4 device* lp_sides_40 [[buffer(28)]], uint device* engine_counters_40 [[buffer(26)]], packed_float4 device* engine_contacts_40 [[buffer(25)]], packed_float4 device* cs_anchors_40 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_147;
    (&kernelContext_147)->params_0 = params_40;
    (&kernelContext_147)->work_0 = work_40;
    (&kernelContext_147)->frames_0 = frames_40;
    (&kernelContext_147)->bodies_0 = bodies_40;
    (&kernelContext_147)->chunks_0 = chunks_40;
    (&kernelContext_147)->sort_0 = sort_40;
    (&kernelContext_147)->atom_0 = atom_40;
    (&kernelContext_147)->manifolds_0 = manifolds_40;
    (&kernelContext_147)->loads_0 = loads_40;
    (&kernelContext_147)->bonds_0 = bonds_40;
    (&kernelContext_147)->args_0 = args_40;
    (&kernelContext_147)->info_0 = info_40;
    (&kernelContext_147)->scan_0 = scan_40;
    (&kernelContext_147)->parts_0 = parts_40;
    (&kernelContext_147)->colliders_0 = colliders_40;
    (&kernelContext_147)->collider_info_0 = collider_info_40;
    (&kernelContext_147)->view_chunks_0 = view_chunks_40;
    (&kernelContext_147)->view_0 = view_40;
    (&kernelContext_147)->objects_0 = objects_40;
    (&kernelContext_147)->shot_bodies_0 = shot_bodies_40;
    (&kernelContext_147)->lines_0 = lines_40;
    (&kernelContext_147)->geometry_0 = geometry_40;
    (&kernelContext_147)->shots_0 = shots_40;
    (&kernelContext_147)->lp_0 = lp_40;
    (&kernelContext_147)->ext_0 = ext_42;
    (&kernelContext_147)->lp_sides_0 = lp_sides_40;
    (&kernelContext_147)->engine_counters_0 = engine_counters_40;
    (&kernelContext_147)->engine_contacts_0 = engine_contacts_40;
    (&kernelContext_147)->cs_anchors_0 = cs_anchors_40;
    threadgroup array<uint, int(1024)> g_scan_40;
    (&kernelContext_147)->g_scan_0 = &g_scan_40;
    threadgroup array<float, int(64)> g_il_best_40;
    (&kernelContext_147)->g_il_best_0 = &g_il_best_40;
    if((params_40->shot_count_0) > 0U)
    {
        il_tube_count_1(id_37.x, &kernelContext_147);
    }
    return;
}

[[kernel]] void fr_il_tube_base(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], atomic<uint> device* atom_41 [[buffer(5)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], packed_float4 device* lp_41 [[buffer(24)]], packed_float4 device* ext_43 [[buffer(23)]], packed_float4 device* lp_sides_41 [[buffer(28)]], uint device* engine_counters_41 [[buffer(26)]], packed_float4 device* engine_contacts_41 [[buffer(25)]], packed_float4 device* cs_anchors_41 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_148;
    (&kernelContext_148)->params_0 = params_41;
    (&kernelContext_148)->work_0 = work_41;
    (&kernelContext_148)->frames_0 = frames_41;
    (&kernelContext_148)->bodies_0 = bodies_41;
    (&kernelContext_148)->chunks_0 = chunks_41;
    (&kernelContext_148)->sort_0 = sort_41;
    (&kernelContext_148)->atom_0 = atom_41;
    (&kernelContext_148)->manifolds_0 = manifolds_41;
    (&kernelContext_148)->loads_0 = loads_41;
    (&kernelContext_148)->bonds_0 = bonds_41;
    (&kernelContext_148)->args_0 = args_41;
    (&kernelContext_148)->info_0 = info_41;
    (&kernelContext_148)->scan_0 = scan_41;
    (&kernelContext_148)->parts_0 = parts_41;
    (&kernelContext_148)->colliders_0 = colliders_41;
    (&kernelContext_148)->collider_info_0 = collider_info_41;
    (&kernelContext_148)->view_chunks_0 = view_chunks_41;
    (&kernelContext_148)->view_0 = view_41;
    (&kernelContext_148)->objects_0 = objects_41;
    (&kernelContext_148)->shot_bodies_0 = shot_bodies_41;
    (&kernelContext_148)->lines_0 = lines_41;
    (&kernelContext_148)->geometry_0 = geometry_41;
    (&kernelContext_148)->shots_0 = shots_41;
    (&kernelContext_148)->lp_0 = lp_41;
    (&kernelContext_148)->ext_0 = ext_43;
    (&kernelContext_148)->lp_sides_0 = lp_sides_41;
    (&kernelContext_148)->engine_counters_0 = engine_counters_41;
    (&kernelContext_148)->engine_contacts_0 = engine_contacts_41;
    (&kernelContext_148)->cs_anchors_0 = cs_anchors_41;
    threadgroup array<uint, int(1024)> g_scan_41;
    (&kernelContext_148)->g_scan_0 = &g_scan_41;
    threadgroup array<float, int(64)> g_il_best_41;
    (&kernelContext_148)->g_il_best_0 = &g_il_best_41;
    if((params_41->shot_count_0) > 0U)
    {
        il_tube_base_1(id_38.x, &kernelContext_148);
    }
    return;
}

[[kernel]] void fr_il_tube_collect(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], atomic<uint> device* atom_42 [[buffer(5)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], packed_float4 device* lp_42 [[buffer(24)]], packed_float4 device* ext_44 [[buffer(23)]], packed_float4 device* lp_sides_42 [[buffer(28)]], uint device* engine_counters_42 [[buffer(26)]], packed_float4 device* engine_contacts_42 [[buffer(25)]], packed_float4 device* cs_anchors_42 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_149;
    (&kernelContext_149)->params_0 = params_42;
    (&kernelContext_149)->work_0 = work_42;
    (&kernelContext_149)->frames_0 = frames_42;
    (&kernelContext_149)->bodies_0 = bodies_42;
    (&kernelContext_149)->chunks_0 = chunks_42;
    (&kernelContext_149)->sort_0 = sort_42;
    (&kernelContext_149)->atom_0 = atom_42;
    (&kernelContext_149)->manifolds_0 = manifolds_42;
    (&kernelContext_149)->loads_0 = loads_42;
    (&kernelContext_149)->bonds_0 = bonds_42;
    (&kernelContext_149)->args_0 = args_42;
    (&kernelContext_149)->info_0 = info_42;
    (&kernelContext_149)->scan_0 = scan_42;
    (&kernelContext_149)->parts_0 = parts_42;
    (&kernelContext_149)->colliders_0 = colliders_42;
    (&kernelContext_149)->collider_info_0 = collider_info_42;
    (&kernelContext_149)->view_chunks_0 = view_chunks_42;
    (&kernelContext_149)->view_0 = view_42;
    (&kernelContext_149)->objects_0 = objects_42;
    (&kernelContext_149)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_149)->lines_0 = lines_42;
    (&kernelContext_149)->geometry_0 = geometry_42;
    (&kernelContext_149)->shots_0 = shots_42;
    (&kernelContext_149)->lp_0 = lp_42;
    (&kernelContext_149)->ext_0 = ext_44;
    (&kernelContext_149)->lp_sides_0 = lp_sides_42;
    (&kernelContext_149)->engine_counters_0 = engine_counters_42;
    (&kernelContext_149)->engine_contacts_0 = engine_contacts_42;
    (&kernelContext_149)->cs_anchors_0 = cs_anchors_42;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_149)->g_scan_0 = &g_scan_42;
    threadgroup array<float, int(64)> g_il_best_42;
    (&kernelContext_149)->g_il_best_0 = &g_il_best_42;
    if((params_42->shot_count_0) > 0U)
    {
        il_tube_collect_1(id_39.x, &kernelContext_149);
    }
    return;
}

[[kernel]] void fr_il_run(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_43 [[buffer(0)]], uint device* work_43 [[buffer(4)]], packed_float4 device* frames_43 [[buffer(6)]], packed_float4 device* bodies_43 [[buffer(7)]], packed_float4 device* chunks_43 [[buffer(1)]], uint device* sort_43 [[buffer(11)]], atomic<uint> device* atom_43 [[buffer(5)]], Record_natural_0 device* manifolds_43 [[buffer(10)]], packed_float4 device* loads_43 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_43 [[buffer(13)]], uint device* info_43 [[buffer(15)]], ScanParams_0 constant* scan_43 [[buffer(14)]], packed_float4 device* parts_43 [[buffer(3)]], packed_float4 device* colliders_43 [[buffer(8)]], packed_uint4 device* collider_info_43 [[buffer(9)]], packed_float4 device* view_chunks_43 [[buffer(17)]], ViewParams_0 constant* view_43 [[buffer(16)]], packed_float4 device* objects_43 [[buffer(18)]], uint device* shot_bodies_43 [[buffer(20)]], float device* lines_43 [[buffer(19)]], packed_float4 device* geometry_43 [[buffer(21)]], packed_float4 device* shots_43 [[buffer(22)]], packed_float4 device* lp_43 [[buffer(24)]], packed_float4 device* ext_45 [[buffer(23)]], packed_float4 device* lp_sides_43 [[buffer(28)]], uint device* engine_counters_43 [[buffer(26)]], packed_float4 device* engine_contacts_43 [[buffer(25)]], packed_float4 device* cs_anchors_43 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_150;
    (&kernelContext_150)->params_0 = params_43;
    (&kernelContext_150)->work_0 = work_43;
    (&kernelContext_150)->frames_0 = frames_43;
    (&kernelContext_150)->bodies_0 = bodies_43;
    (&kernelContext_150)->chunks_0 = chunks_43;
    (&kernelContext_150)->sort_0 = sort_43;
    (&kernelContext_150)->atom_0 = atom_43;
    (&kernelContext_150)->manifolds_0 = manifolds_43;
    (&kernelContext_150)->loads_0 = loads_43;
    (&kernelContext_150)->bonds_0 = bonds_43;
    (&kernelContext_150)->args_0 = args_43;
    (&kernelContext_150)->info_0 = info_43;
    (&kernelContext_150)->scan_0 = scan_43;
    (&kernelContext_150)->parts_0 = parts_43;
    (&kernelContext_150)->colliders_0 = colliders_43;
    (&kernelContext_150)->collider_info_0 = collider_info_43;
    (&kernelContext_150)->view_chunks_0 = view_chunks_43;
    (&kernelContext_150)->view_0 = view_43;
    (&kernelContext_150)->objects_0 = objects_43;
    (&kernelContext_150)->shot_bodies_0 = shot_bodies_43;
    (&kernelContext_150)->lines_0 = lines_43;
    (&kernelContext_150)->geometry_0 = geometry_43;
    (&kernelContext_150)->shots_0 = shots_43;
    (&kernelContext_150)->lp_0 = lp_43;
    (&kernelContext_150)->ext_0 = ext_45;
    (&kernelContext_150)->lp_sides_0 = lp_sides_43;
    (&kernelContext_150)->engine_counters_0 = engine_counters_43;
    (&kernelContext_150)->engine_contacts_0 = engine_contacts_43;
    (&kernelContext_150)->cs_anchors_0 = cs_anchors_43;
    threadgroup array<uint, int(1024)> g_scan_43;
    (&kernelContext_150)->g_scan_0 = &g_scan_43;
    threadgroup array<float, int(64)> g_il_best_43;
    (&kernelContext_150)->g_il_best_0 = &g_il_best_43;
    if((params_43->shot_count_0) > 0U)
    {
        il_run_1(group_1.x, thread_4.x, &kernelContext_150);
    }
    return;
}

[[kernel]] void fr_il_strike(uint3 id_40 [[thread_position_in_grid]], FractureParams_0 constant* params_44 [[buffer(0)]], uint device* work_44 [[buffer(4)]], packed_float4 device* frames_44 [[buffer(6)]], packed_float4 device* bodies_44 [[buffer(7)]], packed_float4 device* chunks_44 [[buffer(1)]], uint device* sort_44 [[buffer(11)]], atomic<uint> device* atom_44 [[buffer(5)]], Record_natural_0 device* manifolds_44 [[buffer(10)]], packed_float4 device* loads_44 [[buffer(12)]], packed_uint4 device* bonds_44 [[buffer(2)]], uint device* args_44 [[buffer(13)]], uint device* info_44 [[buffer(15)]], ScanParams_0 constant* scan_44 [[buffer(14)]], packed_float4 device* parts_44 [[buffer(3)]], packed_float4 device* colliders_44 [[buffer(8)]], packed_uint4 device* collider_info_44 [[buffer(9)]], packed_float4 device* view_chunks_44 [[buffer(17)]], ViewParams_0 constant* view_44 [[buffer(16)]], packed_float4 device* objects_44 [[buffer(18)]], uint device* shot_bodies_44 [[buffer(20)]], float device* lines_44 [[buffer(19)]], packed_float4 device* geometry_44 [[buffer(21)]], packed_float4 device* shots_44 [[buffer(22)]], packed_float4 device* lp_44 [[buffer(24)]], packed_float4 device* ext_46 [[buffer(23)]], packed_float4 device* lp_sides_44 [[buffer(28)]], uint device* engine_counters_44 [[buffer(26)]], packed_float4 device* engine_contacts_44 [[buffer(25)]], packed_float4 device* cs_anchors_44 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_151;
    (&kernelContext_151)->params_0 = params_44;
    (&kernelContext_151)->work_0 = work_44;
    (&kernelContext_151)->frames_0 = frames_44;
    (&kernelContext_151)->bodies_0 = bodies_44;
    (&kernelContext_151)->chunks_0 = chunks_44;
    (&kernelContext_151)->sort_0 = sort_44;
    (&kernelContext_151)->atom_0 = atom_44;
    (&kernelContext_151)->manifolds_0 = manifolds_44;
    (&kernelContext_151)->loads_0 = loads_44;
    (&kernelContext_151)->bonds_0 = bonds_44;
    (&kernelContext_151)->args_0 = args_44;
    (&kernelContext_151)->info_0 = info_44;
    (&kernelContext_151)->scan_0 = scan_44;
    (&kernelContext_151)->parts_0 = parts_44;
    (&kernelContext_151)->colliders_0 = colliders_44;
    (&kernelContext_151)->collider_info_0 = collider_info_44;
    (&kernelContext_151)->view_chunks_0 = view_chunks_44;
    (&kernelContext_151)->view_0 = view_44;
    (&kernelContext_151)->objects_0 = objects_44;
    (&kernelContext_151)->shot_bodies_0 = shot_bodies_44;
    (&kernelContext_151)->lines_0 = lines_44;
    (&kernelContext_151)->geometry_0 = geometry_44;
    (&kernelContext_151)->shots_0 = shots_44;
    (&kernelContext_151)->lp_0 = lp_44;
    (&kernelContext_151)->ext_0 = ext_46;
    (&kernelContext_151)->lp_sides_0 = lp_sides_44;
    (&kernelContext_151)->engine_counters_0 = engine_counters_44;
    (&kernelContext_151)->engine_contacts_0 = engine_contacts_44;
    (&kernelContext_151)->cs_anchors_0 = cs_anchors_44;
    threadgroup array<uint, int(1024)> g_scan_44;
    (&kernelContext_151)->g_scan_0 = &g_scan_44;
    threadgroup array<float, int(64)> g_il_best_44;
    (&kernelContext_151)->g_il_best_0 = &g_il_best_44;
    if((params_44->shot_count_0) > 0U)
    {
        il_strike_1(id_40.x, &kernelContext_151);
    }
    return;
}

[[kernel]] void fr_il_mark(uint3 id_41 [[thread_position_in_grid]], FractureParams_0 constant* params_45 [[buffer(0)]], uint device* work_45 [[buffer(4)]], packed_float4 device* frames_45 [[buffer(6)]], packed_float4 device* bodies_45 [[buffer(7)]], packed_float4 device* chunks_45 [[buffer(1)]], uint device* sort_45 [[buffer(11)]], atomic<uint> device* atom_45 [[buffer(5)]], Record_natural_0 device* manifolds_45 [[buffer(10)]], packed_float4 device* loads_45 [[buffer(12)]], packed_uint4 device* bonds_45 [[buffer(2)]], uint device* args_45 [[buffer(13)]], uint device* info_45 [[buffer(15)]], ScanParams_0 constant* scan_45 [[buffer(14)]], packed_float4 device* parts_45 [[buffer(3)]], packed_float4 device* colliders_45 [[buffer(8)]], packed_uint4 device* collider_info_45 [[buffer(9)]], packed_float4 device* view_chunks_45 [[buffer(17)]], ViewParams_0 constant* view_45 [[buffer(16)]], packed_float4 device* objects_45 [[buffer(18)]], uint device* shot_bodies_45 [[buffer(20)]], float device* lines_45 [[buffer(19)]], packed_float4 device* geometry_45 [[buffer(21)]], packed_float4 device* shots_45 [[buffer(22)]], packed_float4 device* lp_45 [[buffer(24)]], packed_float4 device* ext_47 [[buffer(23)]], packed_float4 device* lp_sides_45 [[buffer(28)]], uint device* engine_counters_45 [[buffer(26)]], packed_float4 device* engine_contacts_45 [[buffer(25)]], packed_float4 device* cs_anchors_45 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_152;
    (&kernelContext_152)->params_0 = params_45;
    (&kernelContext_152)->work_0 = work_45;
    (&kernelContext_152)->frames_0 = frames_45;
    (&kernelContext_152)->bodies_0 = bodies_45;
    (&kernelContext_152)->chunks_0 = chunks_45;
    (&kernelContext_152)->sort_0 = sort_45;
    (&kernelContext_152)->atom_0 = atom_45;
    (&kernelContext_152)->manifolds_0 = manifolds_45;
    (&kernelContext_152)->loads_0 = loads_45;
    (&kernelContext_152)->bonds_0 = bonds_45;
    (&kernelContext_152)->args_0 = args_45;
    (&kernelContext_152)->info_0 = info_45;
    (&kernelContext_152)->scan_0 = scan_45;
    (&kernelContext_152)->parts_0 = parts_45;
    (&kernelContext_152)->colliders_0 = colliders_45;
    (&kernelContext_152)->collider_info_0 = collider_info_45;
    (&kernelContext_152)->view_chunks_0 = view_chunks_45;
    (&kernelContext_152)->view_0 = view_45;
    (&kernelContext_152)->objects_0 = objects_45;
    (&kernelContext_152)->shot_bodies_0 = shot_bodies_45;
    (&kernelContext_152)->lines_0 = lines_45;
    (&kernelContext_152)->geometry_0 = geometry_45;
    (&kernelContext_152)->shots_0 = shots_45;
    (&kernelContext_152)->lp_0 = lp_45;
    (&kernelContext_152)->ext_0 = ext_47;
    (&kernelContext_152)->lp_sides_0 = lp_sides_45;
    (&kernelContext_152)->engine_counters_0 = engine_counters_45;
    (&kernelContext_152)->engine_contacts_0 = engine_contacts_45;
    (&kernelContext_152)->cs_anchors_0 = cs_anchors_45;
    threadgroup array<uint, int(1024)> g_scan_45;
    (&kernelContext_152)->g_scan_0 = &g_scan_45;
    threadgroup array<float, int(64)> g_il_best_45;
    (&kernelContext_152)->g_il_best_0 = &g_il_best_45;
    if((params_45->shot_count_0) > 0U)
    {
        il_mark_1(id_41.x, &kernelContext_152);
    }
    return;
}

[[kernel]] void fr_il_carry(uint3 id_42 [[thread_position_in_grid]], FractureParams_0 constant* params_46 [[buffer(0)]], uint device* work_46 [[buffer(4)]], packed_float4 device* frames_46 [[buffer(6)]], packed_float4 device* bodies_46 [[buffer(7)]], packed_float4 device* chunks_46 [[buffer(1)]], uint device* sort_46 [[buffer(11)]], atomic<uint> device* atom_46 [[buffer(5)]], Record_natural_0 device* manifolds_46 [[buffer(10)]], packed_float4 device* loads_46 [[buffer(12)]], packed_uint4 device* bonds_46 [[buffer(2)]], uint device* args_46 [[buffer(13)]], uint device* info_46 [[buffer(15)]], ScanParams_0 constant* scan_46 [[buffer(14)]], packed_float4 device* parts_46 [[buffer(3)]], packed_float4 device* colliders_46 [[buffer(8)]], packed_uint4 device* collider_info_46 [[buffer(9)]], packed_float4 device* view_chunks_46 [[buffer(17)]], ViewParams_0 constant* view_46 [[buffer(16)]], packed_float4 device* objects_46 [[buffer(18)]], uint device* shot_bodies_46 [[buffer(20)]], float device* lines_46 [[buffer(19)]], packed_float4 device* geometry_46 [[buffer(21)]], packed_float4 device* shots_46 [[buffer(22)]], packed_float4 device* lp_46 [[buffer(24)]], packed_float4 device* ext_48 [[buffer(23)]], packed_float4 device* lp_sides_46 [[buffer(28)]], uint device* engine_counters_46 [[buffer(26)]], packed_float4 device* engine_contacts_46 [[buffer(25)]], packed_float4 device* cs_anchors_46 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_153;
    (&kernelContext_153)->params_0 = params_46;
    (&kernelContext_153)->work_0 = work_46;
    (&kernelContext_153)->frames_0 = frames_46;
    (&kernelContext_153)->bodies_0 = bodies_46;
    (&kernelContext_153)->chunks_0 = chunks_46;
    (&kernelContext_153)->sort_0 = sort_46;
    (&kernelContext_153)->atom_0 = atom_46;
    (&kernelContext_153)->manifolds_0 = manifolds_46;
    (&kernelContext_153)->loads_0 = loads_46;
    (&kernelContext_153)->bonds_0 = bonds_46;
    (&kernelContext_153)->args_0 = args_46;
    (&kernelContext_153)->info_0 = info_46;
    (&kernelContext_153)->scan_0 = scan_46;
    (&kernelContext_153)->parts_0 = parts_46;
    (&kernelContext_153)->colliders_0 = colliders_46;
    (&kernelContext_153)->collider_info_0 = collider_info_46;
    (&kernelContext_153)->view_chunks_0 = view_chunks_46;
    (&kernelContext_153)->view_0 = view_46;
    (&kernelContext_153)->objects_0 = objects_46;
    (&kernelContext_153)->shot_bodies_0 = shot_bodies_46;
    (&kernelContext_153)->lines_0 = lines_46;
    (&kernelContext_153)->geometry_0 = geometry_46;
    (&kernelContext_153)->shots_0 = shots_46;
    (&kernelContext_153)->lp_0 = lp_46;
    (&kernelContext_153)->ext_0 = ext_48;
    (&kernelContext_153)->lp_sides_0 = lp_sides_46;
    (&kernelContext_153)->engine_counters_0 = engine_counters_46;
    (&kernelContext_153)->engine_contacts_0 = engine_contacts_46;
    (&kernelContext_153)->cs_anchors_0 = cs_anchors_46;
    threadgroup array<uint, int(1024)> g_scan_46;
    (&kernelContext_153)->g_scan_0 = &g_scan_46;
    threadgroup array<float, int(64)> g_il_best_46;
    (&kernelContext_153)->g_il_best_0 = &g_il_best_46;
    if((params_46->shot_count_0) > 0U)
    {
        il_carry_1(id_42.x, &kernelContext_153);
    }
    return;
}

[[kernel]] void fr_il_breaks(uint3 id_43 [[thread_position_in_grid]], FractureParams_0 constant* params_47 [[buffer(0)]], uint device* work_47 [[buffer(4)]], packed_float4 device* frames_47 [[buffer(6)]], packed_float4 device* bodies_47 [[buffer(7)]], packed_float4 device* chunks_47 [[buffer(1)]], uint device* sort_47 [[buffer(11)]], atomic<uint> device* atom_47 [[buffer(5)]], Record_natural_0 device* manifolds_47 [[buffer(10)]], packed_float4 device* loads_47 [[buffer(12)]], packed_uint4 device* bonds_47 [[buffer(2)]], uint device* args_47 [[buffer(13)]], uint device* info_47 [[buffer(15)]], ScanParams_0 constant* scan_47 [[buffer(14)]], packed_float4 device* parts_47 [[buffer(3)]], packed_float4 device* colliders_47 [[buffer(8)]], packed_uint4 device* collider_info_47 [[buffer(9)]], packed_float4 device* view_chunks_47 [[buffer(17)]], ViewParams_0 constant* view_47 [[buffer(16)]], packed_float4 device* objects_47 [[buffer(18)]], uint device* shot_bodies_47 [[buffer(20)]], float device* lines_47 [[buffer(19)]], packed_float4 device* geometry_47 [[buffer(21)]], packed_float4 device* shots_47 [[buffer(22)]], packed_float4 device* lp_47 [[buffer(24)]], packed_float4 device* ext_49 [[buffer(23)]], packed_float4 device* lp_sides_47 [[buffer(28)]], uint device* engine_counters_47 [[buffer(26)]], packed_float4 device* engine_contacts_47 [[buffer(25)]], packed_float4 device* cs_anchors_47 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_154;
    (&kernelContext_154)->params_0 = params_47;
    (&kernelContext_154)->work_0 = work_47;
    (&kernelContext_154)->frames_0 = frames_47;
    (&kernelContext_154)->bodies_0 = bodies_47;
    (&kernelContext_154)->chunks_0 = chunks_47;
    (&kernelContext_154)->sort_0 = sort_47;
    (&kernelContext_154)->atom_0 = atom_47;
    (&kernelContext_154)->manifolds_0 = manifolds_47;
    (&kernelContext_154)->loads_0 = loads_47;
    (&kernelContext_154)->bonds_0 = bonds_47;
    (&kernelContext_154)->args_0 = args_47;
    (&kernelContext_154)->info_0 = info_47;
    (&kernelContext_154)->scan_0 = scan_47;
    (&kernelContext_154)->parts_0 = parts_47;
    (&kernelContext_154)->colliders_0 = colliders_47;
    (&kernelContext_154)->collider_info_0 = collider_info_47;
    (&kernelContext_154)->view_chunks_0 = view_chunks_47;
    (&kernelContext_154)->view_0 = view_47;
    (&kernelContext_154)->objects_0 = objects_47;
    (&kernelContext_154)->shot_bodies_0 = shot_bodies_47;
    (&kernelContext_154)->lines_0 = lines_47;
    (&kernelContext_154)->geometry_0 = geometry_47;
    (&kernelContext_154)->shots_0 = shots_47;
    (&kernelContext_154)->lp_0 = lp_47;
    (&kernelContext_154)->ext_0 = ext_49;
    (&kernelContext_154)->lp_sides_0 = lp_sides_47;
    (&kernelContext_154)->engine_counters_0 = engine_counters_47;
    (&kernelContext_154)->engine_contacts_0 = engine_contacts_47;
    (&kernelContext_154)->cs_anchors_0 = cs_anchors_47;
    threadgroup array<uint, int(1024)> g_scan_47;
    (&kernelContext_154)->g_scan_0 = &g_scan_47;
    threadgroup array<float, int(64)> g_il_best_47;
    (&kernelContext_154)->g_il_best_0 = &g_il_best_47;
    if((params_47->shot_count_0) > 0U)
    {
        il_breaks_1(id_43.x, &kernelContext_154);
    }
    return;
}

[[kernel]] void fr_csl_begin(uint3 id_44 [[thread_position_in_grid]], FractureParams_0 constant* params_48 [[buffer(0)]], uint device* work_48 [[buffer(4)]], packed_float4 device* frames_48 [[buffer(6)]], packed_float4 device* bodies_48 [[buffer(7)]], packed_float4 device* chunks_48 [[buffer(1)]], uint device* sort_48 [[buffer(11)]], atomic<uint> device* atom_48 [[buffer(5)]], Record_natural_0 device* manifolds_48 [[buffer(10)]], packed_float4 device* loads_48 [[buffer(12)]], packed_uint4 device* bonds_48 [[buffer(2)]], uint device* args_48 [[buffer(13)]], uint device* info_48 [[buffer(15)]], ScanParams_0 constant* scan_48 [[buffer(14)]], packed_float4 device* parts_48 [[buffer(3)]], packed_float4 device* colliders_48 [[buffer(8)]], packed_uint4 device* collider_info_48 [[buffer(9)]], packed_float4 device* view_chunks_48 [[buffer(17)]], ViewParams_0 constant* view_48 [[buffer(16)]], packed_float4 device* objects_48 [[buffer(18)]], uint device* shot_bodies_48 [[buffer(20)]], float device* lines_48 [[buffer(19)]], packed_float4 device* geometry_48 [[buffer(21)]], packed_float4 device* shots_48 [[buffer(22)]], packed_float4 device* lp_48 [[buffer(24)]], packed_float4 device* ext_50 [[buffer(23)]], packed_float4 device* lp_sides_48 [[buffer(28)]], uint device* engine_counters_48 [[buffer(26)]], packed_float4 device* engine_contacts_48 [[buffer(25)]], packed_float4 device* cs_anchors_48 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_155;
    (&kernelContext_155)->params_0 = params_48;
    (&kernelContext_155)->work_0 = work_48;
    (&kernelContext_155)->frames_0 = frames_48;
    (&kernelContext_155)->bodies_0 = bodies_48;
    (&kernelContext_155)->chunks_0 = chunks_48;
    (&kernelContext_155)->sort_0 = sort_48;
    (&kernelContext_155)->atom_0 = atom_48;
    (&kernelContext_155)->manifolds_0 = manifolds_48;
    (&kernelContext_155)->loads_0 = loads_48;
    (&kernelContext_155)->bonds_0 = bonds_48;
    (&kernelContext_155)->args_0 = args_48;
    (&kernelContext_155)->info_0 = info_48;
    (&kernelContext_155)->scan_0 = scan_48;
    (&kernelContext_155)->parts_0 = parts_48;
    (&kernelContext_155)->colliders_0 = colliders_48;
    (&kernelContext_155)->collider_info_0 = collider_info_48;
    (&kernelContext_155)->view_chunks_0 = view_chunks_48;
    (&kernelContext_155)->view_0 = view_48;
    (&kernelContext_155)->objects_0 = objects_48;
    (&kernelContext_155)->shot_bodies_0 = shot_bodies_48;
    (&kernelContext_155)->lines_0 = lines_48;
    (&kernelContext_155)->geometry_0 = geometry_48;
    (&kernelContext_155)->shots_0 = shots_48;
    (&kernelContext_155)->lp_0 = lp_48;
    (&kernelContext_155)->ext_0 = ext_50;
    (&kernelContext_155)->lp_sides_0 = lp_sides_48;
    (&kernelContext_155)->engine_counters_0 = engine_counters_48;
    (&kernelContext_155)->engine_contacts_0 = engine_contacts_48;
    (&kernelContext_155)->cs_anchors_0 = cs_anchors_48;
    threadgroup array<uint, int(1024)> g_scan_48;
    (&kernelContext_155)->g_scan_0 = &g_scan_48;
    threadgroup array<float, int(64)> g_il_best_48;
    (&kernelContext_155)->g_il_best_0 = &g_il_best_48;
    il_begin_0(id_44.x, &kernelContext_155);
    return;
}

[[kernel]] void fr_csl_tube_count(uint3 id_45 [[thread_position_in_grid]], FractureParams_0 constant* params_49 [[buffer(0)]], uint device* work_49 [[buffer(4)]], packed_float4 device* frames_49 [[buffer(6)]], packed_float4 device* bodies_49 [[buffer(7)]], packed_float4 device* chunks_49 [[buffer(1)]], uint device* sort_49 [[buffer(11)]], atomic<uint> device* atom_49 [[buffer(5)]], Record_natural_0 device* manifolds_49 [[buffer(10)]], packed_float4 device* loads_49 [[buffer(12)]], packed_uint4 device* bonds_49 [[buffer(2)]], uint device* args_49 [[buffer(13)]], uint device* info_49 [[buffer(15)]], ScanParams_0 constant* scan_49 [[buffer(14)]], packed_float4 device* parts_49 [[buffer(3)]], packed_float4 device* colliders_49 [[buffer(8)]], packed_uint4 device* collider_info_49 [[buffer(9)]], packed_float4 device* view_chunks_49 [[buffer(17)]], ViewParams_0 constant* view_49 [[buffer(16)]], packed_float4 device* objects_49 [[buffer(18)]], uint device* shot_bodies_49 [[buffer(20)]], float device* lines_49 [[buffer(19)]], packed_float4 device* geometry_49 [[buffer(21)]], packed_float4 device* shots_49 [[buffer(22)]], packed_float4 device* lp_49 [[buffer(24)]], packed_float4 device* ext_51 [[buffer(23)]], packed_float4 device* lp_sides_49 [[buffer(28)]], uint device* engine_counters_49 [[buffer(26)]], packed_float4 device* engine_contacts_49 [[buffer(25)]], packed_float4 device* cs_anchors_49 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_156;
    (&kernelContext_156)->params_0 = params_49;
    (&kernelContext_156)->work_0 = work_49;
    (&kernelContext_156)->frames_0 = frames_49;
    (&kernelContext_156)->bodies_0 = bodies_49;
    (&kernelContext_156)->chunks_0 = chunks_49;
    (&kernelContext_156)->sort_0 = sort_49;
    (&kernelContext_156)->atom_0 = atom_49;
    (&kernelContext_156)->manifolds_0 = manifolds_49;
    (&kernelContext_156)->loads_0 = loads_49;
    (&kernelContext_156)->bonds_0 = bonds_49;
    (&kernelContext_156)->args_0 = args_49;
    (&kernelContext_156)->info_0 = info_49;
    (&kernelContext_156)->scan_0 = scan_49;
    (&kernelContext_156)->parts_0 = parts_49;
    (&kernelContext_156)->colliders_0 = colliders_49;
    (&kernelContext_156)->collider_info_0 = collider_info_49;
    (&kernelContext_156)->view_chunks_0 = view_chunks_49;
    (&kernelContext_156)->view_0 = view_49;
    (&kernelContext_156)->objects_0 = objects_49;
    (&kernelContext_156)->shot_bodies_0 = shot_bodies_49;
    (&kernelContext_156)->lines_0 = lines_49;
    (&kernelContext_156)->geometry_0 = geometry_49;
    (&kernelContext_156)->shots_0 = shots_49;
    (&kernelContext_156)->lp_0 = lp_49;
    (&kernelContext_156)->ext_0 = ext_51;
    (&kernelContext_156)->lp_sides_0 = lp_sides_49;
    (&kernelContext_156)->engine_counters_0 = engine_counters_49;
    (&kernelContext_156)->engine_contacts_0 = engine_contacts_49;
    (&kernelContext_156)->cs_anchors_0 = cs_anchors_49;
    threadgroup array<uint, int(1024)> g_scan_49;
    (&kernelContext_156)->g_scan_0 = &g_scan_49;
    threadgroup array<float, int(64)> g_il_best_49;
    (&kernelContext_156)->g_il_best_0 = &g_il_best_49;
    il_tube_count_0(id_45.x, &kernelContext_156);
    return;
}

[[kernel]] void fr_csl_tube_base(uint3 id_46 [[thread_position_in_grid]], FractureParams_0 constant* params_50 [[buffer(0)]], uint device* work_50 [[buffer(4)]], packed_float4 device* frames_50 [[buffer(6)]], packed_float4 device* bodies_50 [[buffer(7)]], packed_float4 device* chunks_50 [[buffer(1)]], uint device* sort_50 [[buffer(11)]], atomic<uint> device* atom_50 [[buffer(5)]], Record_natural_0 device* manifolds_50 [[buffer(10)]], packed_float4 device* loads_50 [[buffer(12)]], packed_uint4 device* bonds_50 [[buffer(2)]], uint device* args_50 [[buffer(13)]], uint device* info_50 [[buffer(15)]], ScanParams_0 constant* scan_50 [[buffer(14)]], packed_float4 device* parts_50 [[buffer(3)]], packed_float4 device* colliders_50 [[buffer(8)]], packed_uint4 device* collider_info_50 [[buffer(9)]], packed_float4 device* view_chunks_50 [[buffer(17)]], ViewParams_0 constant* view_50 [[buffer(16)]], packed_float4 device* objects_50 [[buffer(18)]], uint device* shot_bodies_50 [[buffer(20)]], float device* lines_50 [[buffer(19)]], packed_float4 device* geometry_50 [[buffer(21)]], packed_float4 device* shots_50 [[buffer(22)]], packed_float4 device* lp_50 [[buffer(24)]], packed_float4 device* ext_52 [[buffer(23)]], packed_float4 device* lp_sides_50 [[buffer(28)]], uint device* engine_counters_50 [[buffer(26)]], packed_float4 device* engine_contacts_50 [[buffer(25)]], packed_float4 device* cs_anchors_50 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_157;
    (&kernelContext_157)->params_0 = params_50;
    (&kernelContext_157)->work_0 = work_50;
    (&kernelContext_157)->frames_0 = frames_50;
    (&kernelContext_157)->bodies_0 = bodies_50;
    (&kernelContext_157)->chunks_0 = chunks_50;
    (&kernelContext_157)->sort_0 = sort_50;
    (&kernelContext_157)->atom_0 = atom_50;
    (&kernelContext_157)->manifolds_0 = manifolds_50;
    (&kernelContext_157)->loads_0 = loads_50;
    (&kernelContext_157)->bonds_0 = bonds_50;
    (&kernelContext_157)->args_0 = args_50;
    (&kernelContext_157)->info_0 = info_50;
    (&kernelContext_157)->scan_0 = scan_50;
    (&kernelContext_157)->parts_0 = parts_50;
    (&kernelContext_157)->colliders_0 = colliders_50;
    (&kernelContext_157)->collider_info_0 = collider_info_50;
    (&kernelContext_157)->view_chunks_0 = view_chunks_50;
    (&kernelContext_157)->view_0 = view_50;
    (&kernelContext_157)->objects_0 = objects_50;
    (&kernelContext_157)->shot_bodies_0 = shot_bodies_50;
    (&kernelContext_157)->lines_0 = lines_50;
    (&kernelContext_157)->geometry_0 = geometry_50;
    (&kernelContext_157)->shots_0 = shots_50;
    (&kernelContext_157)->lp_0 = lp_50;
    (&kernelContext_157)->ext_0 = ext_52;
    (&kernelContext_157)->lp_sides_0 = lp_sides_50;
    (&kernelContext_157)->engine_counters_0 = engine_counters_50;
    (&kernelContext_157)->engine_contacts_0 = engine_contacts_50;
    (&kernelContext_157)->cs_anchors_0 = cs_anchors_50;
    threadgroup array<uint, int(1024)> g_scan_50;
    (&kernelContext_157)->g_scan_0 = &g_scan_50;
    threadgroup array<float, int(64)> g_il_best_50;
    (&kernelContext_157)->g_il_best_0 = &g_il_best_50;
    il_tube_base_0(id_46.x, &kernelContext_157);
    return;
}

[[kernel]] void fr_csl_tube_collect(uint3 id_47 [[thread_position_in_grid]], FractureParams_0 constant* params_51 [[buffer(0)]], uint device* work_51 [[buffer(4)]], packed_float4 device* frames_51 [[buffer(6)]], packed_float4 device* bodies_51 [[buffer(7)]], packed_float4 device* chunks_51 [[buffer(1)]], uint device* sort_51 [[buffer(11)]], atomic<uint> device* atom_51 [[buffer(5)]], Record_natural_0 device* manifolds_51 [[buffer(10)]], packed_float4 device* loads_51 [[buffer(12)]], packed_uint4 device* bonds_51 [[buffer(2)]], uint device* args_51 [[buffer(13)]], uint device* info_51 [[buffer(15)]], ScanParams_0 constant* scan_51 [[buffer(14)]], packed_float4 device* parts_51 [[buffer(3)]], packed_float4 device* colliders_51 [[buffer(8)]], packed_uint4 device* collider_info_51 [[buffer(9)]], packed_float4 device* view_chunks_51 [[buffer(17)]], ViewParams_0 constant* view_51 [[buffer(16)]], packed_float4 device* objects_51 [[buffer(18)]], uint device* shot_bodies_51 [[buffer(20)]], float device* lines_51 [[buffer(19)]], packed_float4 device* geometry_51 [[buffer(21)]], packed_float4 device* shots_51 [[buffer(22)]], packed_float4 device* lp_51 [[buffer(24)]], packed_float4 device* ext_53 [[buffer(23)]], packed_float4 device* lp_sides_51 [[buffer(28)]], uint device* engine_counters_51 [[buffer(26)]], packed_float4 device* engine_contacts_51 [[buffer(25)]], packed_float4 device* cs_anchors_51 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_158;
    (&kernelContext_158)->params_0 = params_51;
    (&kernelContext_158)->work_0 = work_51;
    (&kernelContext_158)->frames_0 = frames_51;
    (&kernelContext_158)->bodies_0 = bodies_51;
    (&kernelContext_158)->chunks_0 = chunks_51;
    (&kernelContext_158)->sort_0 = sort_51;
    (&kernelContext_158)->atom_0 = atom_51;
    (&kernelContext_158)->manifolds_0 = manifolds_51;
    (&kernelContext_158)->loads_0 = loads_51;
    (&kernelContext_158)->bonds_0 = bonds_51;
    (&kernelContext_158)->args_0 = args_51;
    (&kernelContext_158)->info_0 = info_51;
    (&kernelContext_158)->scan_0 = scan_51;
    (&kernelContext_158)->parts_0 = parts_51;
    (&kernelContext_158)->colliders_0 = colliders_51;
    (&kernelContext_158)->collider_info_0 = collider_info_51;
    (&kernelContext_158)->view_chunks_0 = view_chunks_51;
    (&kernelContext_158)->view_0 = view_51;
    (&kernelContext_158)->objects_0 = objects_51;
    (&kernelContext_158)->shot_bodies_0 = shot_bodies_51;
    (&kernelContext_158)->lines_0 = lines_51;
    (&kernelContext_158)->geometry_0 = geometry_51;
    (&kernelContext_158)->shots_0 = shots_51;
    (&kernelContext_158)->lp_0 = lp_51;
    (&kernelContext_158)->ext_0 = ext_53;
    (&kernelContext_158)->lp_sides_0 = lp_sides_51;
    (&kernelContext_158)->engine_counters_0 = engine_counters_51;
    (&kernelContext_158)->engine_contacts_0 = engine_contacts_51;
    (&kernelContext_158)->cs_anchors_0 = cs_anchors_51;
    threadgroup array<uint, int(1024)> g_scan_51;
    (&kernelContext_158)->g_scan_0 = &g_scan_51;
    threadgroup array<float, int(64)> g_il_best_51;
    (&kernelContext_158)->g_il_best_0 = &g_il_best_51;
    il_tube_collect_0(id_47.x, &kernelContext_158);
    return;
}

uint w_cs_list_0(KernelContext_0 thread* kernelContext_159)
{
    uint _S1149 = w_carry_0(kernelContext_159);
    return _S1149 + kernelContext_159->params_0->chunk_count_0;
}

uint w_cs_bodies_0(KernelContext_0 thread* kernelContext_160)
{
    uint _S1150 = w_cs_list_0(kernelContext_160);
    return _S1150 + kernelContext_160->params_0->chunk_count_0;
}

[[kernel]] void fr_csl_run(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_52 [[buffer(0)]], uint device* work_52 [[buffer(4)]], packed_float4 device* frames_52 [[buffer(6)]], packed_float4 device* bodies_52 [[buffer(7)]], packed_float4 device* chunks_52 [[buffer(1)]], uint device* sort_52 [[buffer(11)]], atomic<uint> device* atom_52 [[buffer(5)]], Record_natural_0 device* manifolds_52 [[buffer(10)]], packed_float4 device* loads_52 [[buffer(12)]], packed_uint4 device* bonds_52 [[buffer(2)]], uint device* args_52 [[buffer(13)]], uint device* info_52 [[buffer(15)]], ScanParams_0 constant* scan_52 [[buffer(14)]], packed_float4 device* parts_52 [[buffer(3)]], packed_float4 device* colliders_52 [[buffer(8)]], packed_uint4 device* collider_info_52 [[buffer(9)]], packed_float4 device* view_chunks_52 [[buffer(17)]], ViewParams_0 constant* view_52 [[buffer(16)]], packed_float4 device* objects_52 [[buffer(18)]], uint device* shot_bodies_52 [[buffer(20)]], float device* lines_52 [[buffer(19)]], packed_float4 device* geometry_52 [[buffer(21)]], packed_float4 device* shots_52 [[buffer(22)]], packed_float4 device* lp_52 [[buffer(24)]], packed_float4 device* ext_54 [[buffer(23)]], packed_float4 device* lp_sides_52 [[buffer(28)]], uint device* engine_counters_52 [[buffer(26)]], packed_float4 device* engine_contacts_52 [[buffer(25)]], packed_float4 device* cs_anchors_52 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_161;
    (&kernelContext_161)->params_0 = params_52;
    (&kernelContext_161)->work_0 = work_52;
    (&kernelContext_161)->frames_0 = frames_52;
    (&kernelContext_161)->bodies_0 = bodies_52;
    (&kernelContext_161)->chunks_0 = chunks_52;
    (&kernelContext_161)->sort_0 = sort_52;
    (&kernelContext_161)->atom_0 = atom_52;
    (&kernelContext_161)->manifolds_0 = manifolds_52;
    (&kernelContext_161)->loads_0 = loads_52;
    (&kernelContext_161)->bonds_0 = bonds_52;
    (&kernelContext_161)->args_0 = args_52;
    (&kernelContext_161)->info_0 = info_52;
    (&kernelContext_161)->scan_0 = scan_52;
    (&kernelContext_161)->parts_0 = parts_52;
    (&kernelContext_161)->colliders_0 = colliders_52;
    (&kernelContext_161)->collider_info_0 = collider_info_52;
    (&kernelContext_161)->view_chunks_0 = view_chunks_52;
    (&kernelContext_161)->view_0 = view_52;
    (&kernelContext_161)->objects_0 = objects_52;
    (&kernelContext_161)->shot_bodies_0 = shot_bodies_52;
    (&kernelContext_161)->lines_0 = lines_52;
    (&kernelContext_161)->geometry_0 = geometry_52;
    (&kernelContext_161)->shots_0 = shots_52;
    (&kernelContext_161)->lp_0 = lp_52;
    (&kernelContext_161)->ext_0 = ext_54;
    (&kernelContext_161)->lp_sides_0 = lp_sides_52;
    (&kernelContext_161)->engine_counters_0 = engine_counters_52;
    (&kernelContext_161)->engine_contacts_0 = engine_contacts_52;
    (&kernelContext_161)->cs_anchors_0 = cs_anchors_52;
    threadgroup array<uint, int(1024)> g_scan_52;
    (&kernelContext_161)->g_scan_0 = &g_scan_52;
    threadgroup array<float, int(64)> g_il_best_52;
    (&kernelContext_161)->g_il_best_0 = &g_il_best_52;
    uint _S1151 = w_cs_bodies_0(&kernelContext_161);
    il_run_0(*((&kernelContext_161)->work_0+(_S1151 + group_2.x)), thread_5.x, &kernelContext_161);
    return;
}

[[kernel]] void fr_csl_strike(uint3 id_48 [[thread_position_in_grid]], FractureParams_0 constant* params_53 [[buffer(0)]], uint device* work_53 [[buffer(4)]], packed_float4 device* frames_53 [[buffer(6)]], packed_float4 device* bodies_53 [[buffer(7)]], packed_float4 device* chunks_53 [[buffer(1)]], uint device* sort_53 [[buffer(11)]], atomic<uint> device* atom_53 [[buffer(5)]], Record_natural_0 device* manifolds_53 [[buffer(10)]], packed_float4 device* loads_53 [[buffer(12)]], packed_uint4 device* bonds_53 [[buffer(2)]], uint device* args_53 [[buffer(13)]], uint device* info_53 [[buffer(15)]], ScanParams_0 constant* scan_53 [[buffer(14)]], packed_float4 device* parts_53 [[buffer(3)]], packed_float4 device* colliders_53 [[buffer(8)]], packed_uint4 device* collider_info_53 [[buffer(9)]], packed_float4 device* view_chunks_53 [[buffer(17)]], ViewParams_0 constant* view_53 [[buffer(16)]], packed_float4 device* objects_53 [[buffer(18)]], uint device* shot_bodies_53 [[buffer(20)]], float device* lines_53 [[buffer(19)]], packed_float4 device* geometry_53 [[buffer(21)]], packed_float4 device* shots_53 [[buffer(22)]], packed_float4 device* lp_53 [[buffer(24)]], packed_float4 device* ext_55 [[buffer(23)]], packed_float4 device* lp_sides_53 [[buffer(28)]], uint device* engine_counters_53 [[buffer(26)]], packed_float4 device* engine_contacts_53 [[buffer(25)]], packed_float4 device* cs_anchors_53 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_162;
    (&kernelContext_162)->params_0 = params_53;
    (&kernelContext_162)->work_0 = work_53;
    (&kernelContext_162)->frames_0 = frames_53;
    (&kernelContext_162)->bodies_0 = bodies_53;
    (&kernelContext_162)->chunks_0 = chunks_53;
    (&kernelContext_162)->sort_0 = sort_53;
    (&kernelContext_162)->atom_0 = atom_53;
    (&kernelContext_162)->manifolds_0 = manifolds_53;
    (&kernelContext_162)->loads_0 = loads_53;
    (&kernelContext_162)->bonds_0 = bonds_53;
    (&kernelContext_162)->args_0 = args_53;
    (&kernelContext_162)->info_0 = info_53;
    (&kernelContext_162)->scan_0 = scan_53;
    (&kernelContext_162)->parts_0 = parts_53;
    (&kernelContext_162)->colliders_0 = colliders_53;
    (&kernelContext_162)->collider_info_0 = collider_info_53;
    (&kernelContext_162)->view_chunks_0 = view_chunks_53;
    (&kernelContext_162)->view_0 = view_53;
    (&kernelContext_162)->objects_0 = objects_53;
    (&kernelContext_162)->shot_bodies_0 = shot_bodies_53;
    (&kernelContext_162)->lines_0 = lines_53;
    (&kernelContext_162)->geometry_0 = geometry_53;
    (&kernelContext_162)->shots_0 = shots_53;
    (&kernelContext_162)->lp_0 = lp_53;
    (&kernelContext_162)->ext_0 = ext_55;
    (&kernelContext_162)->lp_sides_0 = lp_sides_53;
    (&kernelContext_162)->engine_counters_0 = engine_counters_53;
    (&kernelContext_162)->engine_contacts_0 = engine_contacts_53;
    (&kernelContext_162)->cs_anchors_0 = cs_anchors_53;
    threadgroup array<uint, int(1024)> g_scan_53;
    (&kernelContext_162)->g_scan_0 = &g_scan_53;
    threadgroup array<float, int(64)> g_il_best_53;
    (&kernelContext_162)->g_il_best_0 = &g_il_best_53;
    il_strike_0(id_48.x, &kernelContext_162);
    return;
}

[[kernel]] void fr_csl_mark(uint3 id_49 [[thread_position_in_grid]], FractureParams_0 constant* params_54 [[buffer(0)]], uint device* work_54 [[buffer(4)]], packed_float4 device* frames_54 [[buffer(6)]], packed_float4 device* bodies_54 [[buffer(7)]], packed_float4 device* chunks_54 [[buffer(1)]], uint device* sort_54 [[buffer(11)]], atomic<uint> device* atom_54 [[buffer(5)]], Record_natural_0 device* manifolds_54 [[buffer(10)]], packed_float4 device* loads_54 [[buffer(12)]], packed_uint4 device* bonds_54 [[buffer(2)]], uint device* args_54 [[buffer(13)]], uint device* info_54 [[buffer(15)]], ScanParams_0 constant* scan_54 [[buffer(14)]], packed_float4 device* parts_54 [[buffer(3)]], packed_float4 device* colliders_54 [[buffer(8)]], packed_uint4 device* collider_info_54 [[buffer(9)]], packed_float4 device* view_chunks_54 [[buffer(17)]], ViewParams_0 constant* view_54 [[buffer(16)]], packed_float4 device* objects_54 [[buffer(18)]], uint device* shot_bodies_54 [[buffer(20)]], float device* lines_54 [[buffer(19)]], packed_float4 device* geometry_54 [[buffer(21)]], packed_float4 device* shots_54 [[buffer(22)]], packed_float4 device* lp_54 [[buffer(24)]], packed_float4 device* ext_56 [[buffer(23)]], packed_float4 device* lp_sides_54 [[buffer(28)]], uint device* engine_counters_54 [[buffer(26)]], packed_float4 device* engine_contacts_54 [[buffer(25)]], packed_float4 device* cs_anchors_54 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_163;
    (&kernelContext_163)->params_0 = params_54;
    (&kernelContext_163)->work_0 = work_54;
    (&kernelContext_163)->frames_0 = frames_54;
    (&kernelContext_163)->bodies_0 = bodies_54;
    (&kernelContext_163)->chunks_0 = chunks_54;
    (&kernelContext_163)->sort_0 = sort_54;
    (&kernelContext_163)->atom_0 = atom_54;
    (&kernelContext_163)->manifolds_0 = manifolds_54;
    (&kernelContext_163)->loads_0 = loads_54;
    (&kernelContext_163)->bonds_0 = bonds_54;
    (&kernelContext_163)->args_0 = args_54;
    (&kernelContext_163)->info_0 = info_54;
    (&kernelContext_163)->scan_0 = scan_54;
    (&kernelContext_163)->parts_0 = parts_54;
    (&kernelContext_163)->colliders_0 = colliders_54;
    (&kernelContext_163)->collider_info_0 = collider_info_54;
    (&kernelContext_163)->view_chunks_0 = view_chunks_54;
    (&kernelContext_163)->view_0 = view_54;
    (&kernelContext_163)->objects_0 = objects_54;
    (&kernelContext_163)->shot_bodies_0 = shot_bodies_54;
    (&kernelContext_163)->lines_0 = lines_54;
    (&kernelContext_163)->geometry_0 = geometry_54;
    (&kernelContext_163)->shots_0 = shots_54;
    (&kernelContext_163)->lp_0 = lp_54;
    (&kernelContext_163)->ext_0 = ext_56;
    (&kernelContext_163)->lp_sides_0 = lp_sides_54;
    (&kernelContext_163)->engine_counters_0 = engine_counters_54;
    (&kernelContext_163)->engine_contacts_0 = engine_contacts_54;
    (&kernelContext_163)->cs_anchors_0 = cs_anchors_54;
    threadgroup array<uint, int(1024)> g_scan_54;
    (&kernelContext_163)->g_scan_0 = &g_scan_54;
    threadgroup array<float, int(64)> g_il_best_54;
    (&kernelContext_163)->g_il_best_0 = &g_il_best_54;
    il_mark_0(id_49.x, &kernelContext_163);
    return;
}

[[kernel]] void fr_csl_carry(uint3 id_50 [[thread_position_in_grid]], FractureParams_0 constant* params_55 [[buffer(0)]], uint device* work_55 [[buffer(4)]], packed_float4 device* frames_55 [[buffer(6)]], packed_float4 device* bodies_55 [[buffer(7)]], packed_float4 device* chunks_55 [[buffer(1)]], uint device* sort_55 [[buffer(11)]], atomic<uint> device* atom_55 [[buffer(5)]], Record_natural_0 device* manifolds_55 [[buffer(10)]], packed_float4 device* loads_55 [[buffer(12)]], packed_uint4 device* bonds_55 [[buffer(2)]], uint device* args_55 [[buffer(13)]], uint device* info_55 [[buffer(15)]], ScanParams_0 constant* scan_55 [[buffer(14)]], packed_float4 device* parts_55 [[buffer(3)]], packed_float4 device* colliders_55 [[buffer(8)]], packed_uint4 device* collider_info_55 [[buffer(9)]], packed_float4 device* view_chunks_55 [[buffer(17)]], ViewParams_0 constant* view_55 [[buffer(16)]], packed_float4 device* objects_55 [[buffer(18)]], uint device* shot_bodies_55 [[buffer(20)]], float device* lines_55 [[buffer(19)]], packed_float4 device* geometry_55 [[buffer(21)]], packed_float4 device* shots_55 [[buffer(22)]], packed_float4 device* lp_55 [[buffer(24)]], packed_float4 device* ext_57 [[buffer(23)]], packed_float4 device* lp_sides_55 [[buffer(28)]], uint device* engine_counters_55 [[buffer(26)]], packed_float4 device* engine_contacts_55 [[buffer(25)]], packed_float4 device* cs_anchors_55 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_164;
    (&kernelContext_164)->params_0 = params_55;
    (&kernelContext_164)->work_0 = work_55;
    (&kernelContext_164)->frames_0 = frames_55;
    (&kernelContext_164)->bodies_0 = bodies_55;
    (&kernelContext_164)->chunks_0 = chunks_55;
    (&kernelContext_164)->sort_0 = sort_55;
    (&kernelContext_164)->atom_0 = atom_55;
    (&kernelContext_164)->manifolds_0 = manifolds_55;
    (&kernelContext_164)->loads_0 = loads_55;
    (&kernelContext_164)->bonds_0 = bonds_55;
    (&kernelContext_164)->args_0 = args_55;
    (&kernelContext_164)->info_0 = info_55;
    (&kernelContext_164)->scan_0 = scan_55;
    (&kernelContext_164)->parts_0 = parts_55;
    (&kernelContext_164)->colliders_0 = colliders_55;
    (&kernelContext_164)->collider_info_0 = collider_info_55;
    (&kernelContext_164)->view_chunks_0 = view_chunks_55;
    (&kernelContext_164)->view_0 = view_55;
    (&kernelContext_164)->objects_0 = objects_55;
    (&kernelContext_164)->shot_bodies_0 = shot_bodies_55;
    (&kernelContext_164)->lines_0 = lines_55;
    (&kernelContext_164)->geometry_0 = geometry_55;
    (&kernelContext_164)->shots_0 = shots_55;
    (&kernelContext_164)->lp_0 = lp_55;
    (&kernelContext_164)->ext_0 = ext_57;
    (&kernelContext_164)->lp_sides_0 = lp_sides_55;
    (&kernelContext_164)->engine_counters_0 = engine_counters_55;
    (&kernelContext_164)->engine_contacts_0 = engine_contacts_55;
    (&kernelContext_164)->cs_anchors_0 = cs_anchors_55;
    threadgroup array<uint, int(1024)> g_scan_55;
    (&kernelContext_164)->g_scan_0 = &g_scan_55;
    threadgroup array<float, int(64)> g_il_best_55;
    (&kernelContext_164)->g_il_best_0 = &g_il_best_55;
    il_carry_0(id_50.x, &kernelContext_164);
    return;
}

[[kernel]] void fr_csl_breaks(uint3 id_51 [[thread_position_in_grid]], FractureParams_0 constant* params_56 [[buffer(0)]], uint device* work_56 [[buffer(4)]], packed_float4 device* frames_56 [[buffer(6)]], packed_float4 device* bodies_56 [[buffer(7)]], packed_float4 device* chunks_56 [[buffer(1)]], uint device* sort_56 [[buffer(11)]], atomic<uint> device* atom_56 [[buffer(5)]], Record_natural_0 device* manifolds_56 [[buffer(10)]], packed_float4 device* loads_56 [[buffer(12)]], packed_uint4 device* bonds_56 [[buffer(2)]], uint device* args_56 [[buffer(13)]], uint device* info_56 [[buffer(15)]], ScanParams_0 constant* scan_56 [[buffer(14)]], packed_float4 device* parts_56 [[buffer(3)]], packed_float4 device* colliders_56 [[buffer(8)]], packed_uint4 device* collider_info_56 [[buffer(9)]], packed_float4 device* view_chunks_56 [[buffer(17)]], ViewParams_0 constant* view_56 [[buffer(16)]], packed_float4 device* objects_56 [[buffer(18)]], uint device* shot_bodies_56 [[buffer(20)]], float device* lines_56 [[buffer(19)]], packed_float4 device* geometry_56 [[buffer(21)]], packed_float4 device* shots_56 [[buffer(22)]], packed_float4 device* lp_56 [[buffer(24)]], packed_float4 device* ext_58 [[buffer(23)]], packed_float4 device* lp_sides_56 [[buffer(28)]], uint device* engine_counters_56 [[buffer(26)]], packed_float4 device* engine_contacts_56 [[buffer(25)]], packed_float4 device* cs_anchors_56 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_165;
    (&kernelContext_165)->params_0 = params_56;
    (&kernelContext_165)->work_0 = work_56;
    (&kernelContext_165)->frames_0 = frames_56;
    (&kernelContext_165)->bodies_0 = bodies_56;
    (&kernelContext_165)->chunks_0 = chunks_56;
    (&kernelContext_165)->sort_0 = sort_56;
    (&kernelContext_165)->atom_0 = atom_56;
    (&kernelContext_165)->manifolds_0 = manifolds_56;
    (&kernelContext_165)->loads_0 = loads_56;
    (&kernelContext_165)->bonds_0 = bonds_56;
    (&kernelContext_165)->args_0 = args_56;
    (&kernelContext_165)->info_0 = info_56;
    (&kernelContext_165)->scan_0 = scan_56;
    (&kernelContext_165)->parts_0 = parts_56;
    (&kernelContext_165)->colliders_0 = colliders_56;
    (&kernelContext_165)->collider_info_0 = collider_info_56;
    (&kernelContext_165)->view_chunks_0 = view_chunks_56;
    (&kernelContext_165)->view_0 = view_56;
    (&kernelContext_165)->objects_0 = objects_56;
    (&kernelContext_165)->shot_bodies_0 = shot_bodies_56;
    (&kernelContext_165)->lines_0 = lines_56;
    (&kernelContext_165)->geometry_0 = geometry_56;
    (&kernelContext_165)->shots_0 = shots_56;
    (&kernelContext_165)->lp_0 = lp_56;
    (&kernelContext_165)->ext_0 = ext_58;
    (&kernelContext_165)->lp_sides_0 = lp_sides_56;
    (&kernelContext_165)->engine_counters_0 = engine_counters_56;
    (&kernelContext_165)->engine_contacts_0 = engine_contacts_56;
    (&kernelContext_165)->cs_anchors_0 = cs_anchors_56;
    threadgroup array<uint, int(1024)> g_scan_56;
    (&kernelContext_165)->g_scan_0 = &g_scan_56;
    threadgroup array<float, int(64)> g_il_best_56;
    (&kernelContext_165)->g_il_best_0 = &g_il_best_56;
    il_breaks_0(id_51.x, &kernelContext_165);
    return;
}

[[kernel]] void fr_cs_begin(uint3 id_52 [[thread_position_in_grid]], FractureParams_0 constant* params_57 [[buffer(0)]], uint device* work_57 [[buffer(4)]], packed_float4 device* frames_57 [[buffer(6)]], packed_float4 device* bodies_57 [[buffer(7)]], packed_float4 device* chunks_57 [[buffer(1)]], uint device* sort_57 [[buffer(11)]], atomic<uint> device* atom_57 [[buffer(5)]], Record_natural_0 device* manifolds_57 [[buffer(10)]], packed_float4 device* loads_57 [[buffer(12)]], packed_uint4 device* bonds_57 [[buffer(2)]], uint device* args_57 [[buffer(13)]], uint device* info_57 [[buffer(15)]], ScanParams_0 constant* scan_57 [[buffer(14)]], packed_float4 device* parts_57 [[buffer(3)]], packed_float4 device* colliders_57 [[buffer(8)]], packed_uint4 device* collider_info_57 [[buffer(9)]], packed_float4 device* view_chunks_57 [[buffer(17)]], ViewParams_0 constant* view_57 [[buffer(16)]], packed_float4 device* objects_57 [[buffer(18)]], uint device* shot_bodies_57 [[buffer(20)]], float device* lines_57 [[buffer(19)]], packed_float4 device* geometry_57 [[buffer(21)]], packed_float4 device* shots_57 [[buffer(22)]], packed_float4 device* lp_57 [[buffer(24)]], packed_float4 device* ext_59 [[buffer(23)]], packed_float4 device* lp_sides_57 [[buffer(28)]], uint device* engine_counters_57 [[buffer(26)]], packed_float4 device* engine_contacts_57 [[buffer(25)]], packed_float4 device* cs_anchors_57 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_166;
    (&kernelContext_166)->params_0 = params_57;
    (&kernelContext_166)->work_0 = work_57;
    (&kernelContext_166)->frames_0 = frames_57;
    (&kernelContext_166)->bodies_0 = bodies_57;
    (&kernelContext_166)->chunks_0 = chunks_57;
    (&kernelContext_166)->sort_0 = sort_57;
    (&kernelContext_166)->atom_0 = atom_57;
    (&kernelContext_166)->manifolds_0 = manifolds_57;
    (&kernelContext_166)->loads_0 = loads_57;
    (&kernelContext_166)->bonds_0 = bonds_57;
    (&kernelContext_166)->args_0 = args_57;
    (&kernelContext_166)->info_0 = info_57;
    (&kernelContext_166)->scan_0 = scan_57;
    (&kernelContext_166)->parts_0 = parts_57;
    (&kernelContext_166)->colliders_0 = colliders_57;
    (&kernelContext_166)->collider_info_0 = collider_info_57;
    (&kernelContext_166)->view_chunks_0 = view_chunks_57;
    (&kernelContext_166)->view_0 = view_57;
    (&kernelContext_166)->objects_0 = objects_57;
    (&kernelContext_166)->shot_bodies_0 = shot_bodies_57;
    (&kernelContext_166)->lines_0 = lines_57;
    (&kernelContext_166)->geometry_0 = geometry_57;
    (&kernelContext_166)->shots_0 = shots_57;
    (&kernelContext_166)->lp_0 = lp_57;
    (&kernelContext_166)->ext_0 = ext_59;
    (&kernelContext_166)->lp_sides_0 = lp_sides_57;
    (&kernelContext_166)->engine_counters_0 = engine_counters_57;
    (&kernelContext_166)->engine_contacts_0 = engine_contacts_57;
    (&kernelContext_166)->cs_anchors_0 = cs_anchors_57;
    threadgroup array<uint, int(1024)> g_scan_57;
    (&kernelContext_166)->g_scan_0 = &g_scan_57;
    threadgroup array<float, int(64)> g_il_best_57;
    (&kernelContext_166)->g_il_best_0 = &g_il_best_57;
    uint _S1152 = a_counters_1(&kernelContext_166);
    atomic_store_explicit((&kernelContext_166)->atom_0+(_S1152 + 14U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_166)->atom_0+(_S1152 + 15U), 0U, memory_order_relaxed);
    uint _S1153 = a_il_cursor_1(&kernelContext_166);
    atomic_store_explicit((&kernelContext_166)->atom_0+_S1153, 0U, memory_order_relaxed);
    uint _S1154 = cs_rows_0(&kernelContext_166);
    uint _S1155 = groups_of_0(_S1154, 64U);
    set_args_0(7U, _S1155, &kernelContext_166);
    return;
}

[[kernel]] void fr_cs_contacts(uint3 id_53 [[thread_position_in_grid]], FractureParams_0 constant* params_58 [[buffer(0)]], uint device* work_58 [[buffer(4)]], packed_float4 device* frames_58 [[buffer(6)]], packed_float4 device* bodies_58 [[buffer(7)]], packed_float4 device* chunks_58 [[buffer(1)]], uint device* sort_58 [[buffer(11)]], atomic<uint> device* atom_58 [[buffer(5)]], Record_natural_0 device* manifolds_58 [[buffer(10)]], packed_float4 device* loads_58 [[buffer(12)]], packed_uint4 device* bonds_58 [[buffer(2)]], uint device* args_58 [[buffer(13)]], uint device* info_58 [[buffer(15)]], ScanParams_0 constant* scan_58 [[buffer(14)]], packed_float4 device* parts_58 [[buffer(3)]], packed_float4 device* colliders_58 [[buffer(8)]], packed_uint4 device* collider_info_58 [[buffer(9)]], packed_float4 device* view_chunks_58 [[buffer(17)]], ViewParams_0 constant* view_58 [[buffer(16)]], packed_float4 device* objects_58 [[buffer(18)]], uint device* shot_bodies_58 [[buffer(20)]], float device* lines_58 [[buffer(19)]], packed_float4 device* geometry_58 [[buffer(21)]], packed_float4 device* shots_58 [[buffer(22)]], packed_float4 device* lp_58 [[buffer(24)]], packed_float4 device* ext_60 [[buffer(23)]], packed_float4 device* lp_sides_58 [[buffer(28)]], uint device* engine_counters_58 [[buffer(26)]], packed_float4 device* engine_contacts_58 [[buffer(25)]], packed_float4 device* cs_anchors_58 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_167;
    (&kernelContext_167)->params_0 = params_58;
    (&kernelContext_167)->work_0 = work_58;
    (&kernelContext_167)->frames_0 = frames_58;
    (&kernelContext_167)->bodies_0 = bodies_58;
    (&kernelContext_167)->chunks_0 = chunks_58;
    (&kernelContext_167)->sort_0 = sort_58;
    (&kernelContext_167)->atom_0 = atom_58;
    (&kernelContext_167)->manifolds_0 = manifolds_58;
    (&kernelContext_167)->loads_0 = loads_58;
    (&kernelContext_167)->bonds_0 = bonds_58;
    (&kernelContext_167)->args_0 = args_58;
    (&kernelContext_167)->info_0 = info_58;
    (&kernelContext_167)->scan_0 = scan_58;
    (&kernelContext_167)->parts_0 = parts_58;
    (&kernelContext_167)->colliders_0 = colliders_58;
    (&kernelContext_167)->collider_info_0 = collider_info_58;
    (&kernelContext_167)->view_chunks_0 = view_chunks_58;
    (&kernelContext_167)->view_0 = view_58;
    (&kernelContext_167)->objects_0 = objects_58;
    (&kernelContext_167)->shot_bodies_0 = shot_bodies_58;
    (&kernelContext_167)->lines_0 = lines_58;
    (&kernelContext_167)->geometry_0 = geometry_58;
    (&kernelContext_167)->shots_0 = shots_58;
    (&kernelContext_167)->lp_0 = lp_58;
    (&kernelContext_167)->ext_0 = ext_60;
    (&kernelContext_167)->lp_sides_0 = lp_sides_58;
    (&kernelContext_167)->engine_counters_0 = engine_counters_58;
    (&kernelContext_167)->engine_contacts_0 = engine_contacts_58;
    (&kernelContext_167)->cs_anchors_0 = cs_anchors_58;
    threadgroup array<uint, int(1024)> g_scan_58;
    (&kernelContext_167)->g_scan_0 = &g_scan_58;
    threadgroup array<float, int(64)> g_il_best_58;
    (&kernelContext_167)->g_il_best_0 = &g_il_best_58;
    uint r_25 = id_53.x;
    uint _S1156 = cs_rows_0(&kernelContext_167);
    if(r_25 >= _S1156)
    {
        return;
    }
    uint side_6 = 0U;
    for(;;)
    {
        if(side_6 < 2U)
        {
        }
        else
        {
            break;
        }
        thread uint k_44;
        thread uint s_17;
        thread uint ob_5;
        thread uint co_5;
        thread float3 p_18;
        thread float3 u_9;
        thread float v_14;
        thread float m_eff_7;
        thread float t_in_8;
        bool _S1157 = cs_side_0(r_25, side_6, &k_44, &s_17, &ob_5, &co_5, &p_18, &u_9, &v_14, &m_eff_7, &t_in_8, &kernelContext_167);
        if(_S1157)
        {
            uint _S1158 = a_cs_0(s_17, &kernelContext_167);
            uint _S1159 = atomic_fetch_max_explicit((&kernelContext_167)->atom_0+_S1158, (as_type<uint>((m_eff_7 * v_14))), memory_order_relaxed);
            uint _S1160 = a_cs_0(s_17, &kernelContext_167);
            uint _S1161 = atomic_exchange_explicit((&kernelContext_167)->atom_0+(_S1160 + 8U), 1U, memory_order_relaxed);
            if(_S1161 == 0U)
            {
                uint _S1162 = w_cs_bodies_0(&kernelContext_167);
                uint _S1163 = a_counters_1(&kernelContext_167);
                uint _S1164 = atomic_fetch_add_explicit((&kernelContext_167)->atom_0+(_S1163 + 14U), 1U, memory_order_relaxed);
                *((&kernelContext_167)->work_0+(_S1162 + _S1164)) = s_17;
            }
        }
        side_6 = side_6 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_gate(uint3 id_54 [[thread_position_in_grid]], FractureParams_0 constant* params_59 [[buffer(0)]], uint device* work_59 [[buffer(4)]], packed_float4 device* frames_59 [[buffer(6)]], packed_float4 device* bodies_59 [[buffer(7)]], packed_float4 device* chunks_59 [[buffer(1)]], uint device* sort_59 [[buffer(11)]], atomic<uint> device* atom_59 [[buffer(5)]], Record_natural_0 device* manifolds_59 [[buffer(10)]], packed_float4 device* loads_59 [[buffer(12)]], packed_uint4 device* bonds_59 [[buffer(2)]], uint device* args_59 [[buffer(13)]], uint device* info_59 [[buffer(15)]], ScanParams_0 constant* scan_59 [[buffer(14)]], packed_float4 device* parts_59 [[buffer(3)]], packed_float4 device* colliders_59 [[buffer(8)]], packed_uint4 device* collider_info_59 [[buffer(9)]], packed_float4 device* view_chunks_59 [[buffer(17)]], ViewParams_0 constant* view_59 [[buffer(16)]], packed_float4 device* objects_59 [[buffer(18)]], uint device* shot_bodies_59 [[buffer(20)]], float device* lines_59 [[buffer(19)]], packed_float4 device* geometry_59 [[buffer(21)]], packed_float4 device* shots_59 [[buffer(22)]], packed_float4 device* lp_59 [[buffer(24)]], packed_float4 device* ext_61 [[buffer(23)]], packed_float4 device* lp_sides_59 [[buffer(28)]], uint device* engine_counters_59 [[buffer(26)]], packed_float4 device* engine_contacts_59 [[buffer(25)]], packed_float4 device* cs_anchors_59 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_168;
    (&kernelContext_168)->params_0 = params_59;
    (&kernelContext_168)->work_0 = work_59;
    (&kernelContext_168)->frames_0 = frames_59;
    (&kernelContext_168)->bodies_0 = bodies_59;
    (&kernelContext_168)->chunks_0 = chunks_59;
    (&kernelContext_168)->sort_0 = sort_59;
    (&kernelContext_168)->atom_0 = atom_59;
    (&kernelContext_168)->manifolds_0 = manifolds_59;
    (&kernelContext_168)->loads_0 = loads_59;
    (&kernelContext_168)->bonds_0 = bonds_59;
    (&kernelContext_168)->args_0 = args_59;
    (&kernelContext_168)->info_0 = info_59;
    (&kernelContext_168)->scan_0 = scan_59;
    (&kernelContext_168)->parts_0 = parts_59;
    (&kernelContext_168)->colliders_0 = colliders_59;
    (&kernelContext_168)->collider_info_0 = collider_info_59;
    (&kernelContext_168)->view_chunks_0 = view_chunks_59;
    (&kernelContext_168)->view_0 = view_59;
    (&kernelContext_168)->objects_0 = objects_59;
    (&kernelContext_168)->shot_bodies_0 = shot_bodies_59;
    (&kernelContext_168)->lines_0 = lines_59;
    (&kernelContext_168)->geometry_0 = geometry_59;
    (&kernelContext_168)->shots_0 = shots_59;
    (&kernelContext_168)->lp_0 = lp_59;
    (&kernelContext_168)->ext_0 = ext_61;
    (&kernelContext_168)->lp_sides_0 = lp_sides_59;
    (&kernelContext_168)->engine_counters_0 = engine_counters_59;
    (&kernelContext_168)->engine_contacts_0 = engine_contacts_59;
    (&kernelContext_168)->cs_anchors_0 = cs_anchors_59;
    threadgroup array<uint, int(1024)> g_scan_59;
    (&kernelContext_168)->g_scan_0 = &g_scan_59;
    threadgroup array<float, int(64)> g_il_best_59;
    (&kernelContext_168)->g_il_best_0 = &g_il_best_59;
    uint _S1165 = a_counters_1(&kernelContext_168);
    uint struck_1 = atomic_load_explicit((&kernelContext_168)->atom_0+(_S1165 + 14U), memory_order_relaxed);
    bool any_0 = struck_1 > 0U;
    uint _S1166;
    if(any_0)
    {
        uint _S1167 = cs_rows_0(&kernelContext_168);
        uint _S1168 = groups_of_0(_S1167, 64U);
        _S1166 = _S1168;
    }
    else
    {
        _S1166 = 0U;
    }
    set_args_0(8U, _S1166, &kernelContext_168);
    uint _S1169 = groups_of_0(struck_1, 64U);
    set_args_0(9U, _S1169, &kernelContext_168);
    set_args_0(6U, struck_1, &kernelContext_168);
    if(any_0)
    {
        uint _S1170 = groups_of_0((&kernelContext_168)->params_0->chunk_count_0, 64U);
        _S1166 = _S1170;
    }
    else
    {
        _S1166 = 0U;
    }
    set_args_0(10U, _S1166, &kernelContext_168);
    if(any_0)
    {
        uint _S1171 = groups_of_0((&kernelContext_168)->params_0->bond_count_0, 64U);
        _S1166 = _S1171;
    }
    else
    {
        _S1166 = 0U;
    }
    set_args_0(11U, _S1166, &kernelContext_168);
    if(any_0)
    {
        uint _S1172 = groups_of_0((&kernelContext_168)->params_0->body_count_0, 64U);
        _S1166 = _S1172;
    }
    else
    {
        _S1166 = 0U;
    }
    set_args_0(12U, _S1166, &kernelContext_168);
    return;
}

[[kernel]] void fr_cs_pick_a(uint3 id_55 [[thread_position_in_grid]], FractureParams_0 constant* params_60 [[buffer(0)]], uint device* work_60 [[buffer(4)]], packed_float4 device* frames_60 [[buffer(6)]], packed_float4 device* bodies_60 [[buffer(7)]], packed_float4 device* chunks_60 [[buffer(1)]], uint device* sort_60 [[buffer(11)]], atomic<uint> device* atom_60 [[buffer(5)]], Record_natural_0 device* manifolds_60 [[buffer(10)]], packed_float4 device* loads_60 [[buffer(12)]], packed_uint4 device* bonds_60 [[buffer(2)]], uint device* args_60 [[buffer(13)]], uint device* info_60 [[buffer(15)]], ScanParams_0 constant* scan_60 [[buffer(14)]], packed_float4 device* parts_60 [[buffer(3)]], packed_float4 device* colliders_60 [[buffer(8)]], packed_uint4 device* collider_info_60 [[buffer(9)]], packed_float4 device* view_chunks_60 [[buffer(17)]], ViewParams_0 constant* view_60 [[buffer(16)]], packed_float4 device* objects_60 [[buffer(18)]], uint device* shot_bodies_60 [[buffer(20)]], float device* lines_60 [[buffer(19)]], packed_float4 device* geometry_60 [[buffer(21)]], packed_float4 device* shots_60 [[buffer(22)]], packed_float4 device* lp_60 [[buffer(24)]], packed_float4 device* ext_62 [[buffer(23)]], packed_float4 device* lp_sides_60 [[buffer(28)]], uint device* engine_counters_60 [[buffer(26)]], packed_float4 device* engine_contacts_60 [[buffer(25)]], packed_float4 device* cs_anchors_60 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_169;
    (&kernelContext_169)->params_0 = params_60;
    (&kernelContext_169)->work_0 = work_60;
    (&kernelContext_169)->frames_0 = frames_60;
    (&kernelContext_169)->bodies_0 = bodies_60;
    (&kernelContext_169)->chunks_0 = chunks_60;
    (&kernelContext_169)->sort_0 = sort_60;
    (&kernelContext_169)->atom_0 = atom_60;
    (&kernelContext_169)->manifolds_0 = manifolds_60;
    (&kernelContext_169)->loads_0 = loads_60;
    (&kernelContext_169)->bonds_0 = bonds_60;
    (&kernelContext_169)->args_0 = args_60;
    (&kernelContext_169)->info_0 = info_60;
    (&kernelContext_169)->scan_0 = scan_60;
    (&kernelContext_169)->parts_0 = parts_60;
    (&kernelContext_169)->colliders_0 = colliders_60;
    (&kernelContext_169)->collider_info_0 = collider_info_60;
    (&kernelContext_169)->view_chunks_0 = view_chunks_60;
    (&kernelContext_169)->view_0 = view_60;
    (&kernelContext_169)->objects_0 = objects_60;
    (&kernelContext_169)->shot_bodies_0 = shot_bodies_60;
    (&kernelContext_169)->lines_0 = lines_60;
    (&kernelContext_169)->geometry_0 = geometry_60;
    (&kernelContext_169)->shots_0 = shots_60;
    (&kernelContext_169)->lp_0 = lp_60;
    (&kernelContext_169)->ext_0 = ext_62;
    (&kernelContext_169)->lp_sides_0 = lp_sides_60;
    (&kernelContext_169)->engine_counters_0 = engine_counters_60;
    (&kernelContext_169)->engine_contacts_0 = engine_contacts_60;
    (&kernelContext_169)->cs_anchors_0 = cs_anchors_60;
    threadgroup array<uint, int(1024)> g_scan_60;
    (&kernelContext_169)->g_scan_0 = &g_scan_60;
    threadgroup array<float, int(64)> g_il_best_60;
    (&kernelContext_169)->g_il_best_0 = &g_il_best_60;
    cs_pick_3(id_55.x, &kernelContext_169);
    return;
}

[[kernel]] void fr_cs_pick_b(uint3 id_56 [[thread_position_in_grid]], FractureParams_0 constant* params_61 [[buffer(0)]], uint device* work_61 [[buffer(4)]], packed_float4 device* frames_61 [[buffer(6)]], packed_float4 device* bodies_61 [[buffer(7)]], packed_float4 device* chunks_61 [[buffer(1)]], uint device* sort_61 [[buffer(11)]], atomic<uint> device* atom_61 [[buffer(5)]], Record_natural_0 device* manifolds_61 [[buffer(10)]], packed_float4 device* loads_61 [[buffer(12)]], packed_uint4 device* bonds_61 [[buffer(2)]], uint device* args_61 [[buffer(13)]], uint device* info_61 [[buffer(15)]], ScanParams_0 constant* scan_61 [[buffer(14)]], packed_float4 device* parts_61 [[buffer(3)]], packed_float4 device* colliders_61 [[buffer(8)]], packed_uint4 device* collider_info_61 [[buffer(9)]], packed_float4 device* view_chunks_61 [[buffer(17)]], ViewParams_0 constant* view_61 [[buffer(16)]], packed_float4 device* objects_61 [[buffer(18)]], uint device* shot_bodies_61 [[buffer(20)]], float device* lines_61 [[buffer(19)]], packed_float4 device* geometry_61 [[buffer(21)]], packed_float4 device* shots_61 [[buffer(22)]], packed_float4 device* lp_61 [[buffer(24)]], packed_float4 device* ext_63 [[buffer(23)]], packed_float4 device* lp_sides_61 [[buffer(28)]], uint device* engine_counters_61 [[buffer(26)]], packed_float4 device* engine_contacts_61 [[buffer(25)]], packed_float4 device* cs_anchors_61 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_170;
    (&kernelContext_170)->params_0 = params_61;
    (&kernelContext_170)->work_0 = work_61;
    (&kernelContext_170)->frames_0 = frames_61;
    (&kernelContext_170)->bodies_0 = bodies_61;
    (&kernelContext_170)->chunks_0 = chunks_61;
    (&kernelContext_170)->sort_0 = sort_61;
    (&kernelContext_170)->atom_0 = atom_61;
    (&kernelContext_170)->manifolds_0 = manifolds_61;
    (&kernelContext_170)->loads_0 = loads_61;
    (&kernelContext_170)->bonds_0 = bonds_61;
    (&kernelContext_170)->args_0 = args_61;
    (&kernelContext_170)->info_0 = info_61;
    (&kernelContext_170)->scan_0 = scan_61;
    (&kernelContext_170)->parts_0 = parts_61;
    (&kernelContext_170)->colliders_0 = colliders_61;
    (&kernelContext_170)->collider_info_0 = collider_info_61;
    (&kernelContext_170)->view_chunks_0 = view_chunks_61;
    (&kernelContext_170)->view_0 = view_61;
    (&kernelContext_170)->objects_0 = objects_61;
    (&kernelContext_170)->shot_bodies_0 = shot_bodies_61;
    (&kernelContext_170)->lines_0 = lines_61;
    (&kernelContext_170)->geometry_0 = geometry_61;
    (&kernelContext_170)->shots_0 = shots_61;
    (&kernelContext_170)->lp_0 = lp_61;
    (&kernelContext_170)->ext_0 = ext_63;
    (&kernelContext_170)->lp_sides_0 = lp_sides_61;
    (&kernelContext_170)->engine_counters_0 = engine_counters_61;
    (&kernelContext_170)->engine_contacts_0 = engine_contacts_61;
    (&kernelContext_170)->cs_anchors_0 = cs_anchors_61;
    threadgroup array<uint, int(1024)> g_scan_61;
    (&kernelContext_170)->g_scan_0 = &g_scan_61;
    threadgroup array<float, int(64)> g_il_best_61;
    (&kernelContext_170)->g_il_best_0 = &g_il_best_61;
    cs_pick_2(id_56.x, &kernelContext_170);
    return;
}

[[kernel]] void fr_cs_pick_c(uint3 id_57 [[thread_position_in_grid]], FractureParams_0 constant* params_62 [[buffer(0)]], uint device* work_62 [[buffer(4)]], packed_float4 device* frames_62 [[buffer(6)]], packed_float4 device* bodies_62 [[buffer(7)]], packed_float4 device* chunks_62 [[buffer(1)]], uint device* sort_62 [[buffer(11)]], atomic<uint> device* atom_62 [[buffer(5)]], Record_natural_0 device* manifolds_62 [[buffer(10)]], packed_float4 device* loads_62 [[buffer(12)]], packed_uint4 device* bonds_62 [[buffer(2)]], uint device* args_62 [[buffer(13)]], uint device* info_62 [[buffer(15)]], ScanParams_0 constant* scan_62 [[buffer(14)]], packed_float4 device* parts_62 [[buffer(3)]], packed_float4 device* colliders_62 [[buffer(8)]], packed_uint4 device* collider_info_62 [[buffer(9)]], packed_float4 device* view_chunks_62 [[buffer(17)]], ViewParams_0 constant* view_62 [[buffer(16)]], packed_float4 device* objects_62 [[buffer(18)]], uint device* shot_bodies_62 [[buffer(20)]], float device* lines_62 [[buffer(19)]], packed_float4 device* geometry_62 [[buffer(21)]], packed_float4 device* shots_62 [[buffer(22)]], packed_float4 device* lp_62 [[buffer(24)]], packed_float4 device* ext_64 [[buffer(23)]], packed_float4 device* lp_sides_62 [[buffer(28)]], uint device* engine_counters_62 [[buffer(26)]], packed_float4 device* engine_contacts_62 [[buffer(25)]], packed_float4 device* cs_anchors_62 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_171;
    (&kernelContext_171)->params_0 = params_62;
    (&kernelContext_171)->work_0 = work_62;
    (&kernelContext_171)->frames_0 = frames_62;
    (&kernelContext_171)->bodies_0 = bodies_62;
    (&kernelContext_171)->chunks_0 = chunks_62;
    (&kernelContext_171)->sort_0 = sort_62;
    (&kernelContext_171)->atom_0 = atom_62;
    (&kernelContext_171)->manifolds_0 = manifolds_62;
    (&kernelContext_171)->loads_0 = loads_62;
    (&kernelContext_171)->bonds_0 = bonds_62;
    (&kernelContext_171)->args_0 = args_62;
    (&kernelContext_171)->info_0 = info_62;
    (&kernelContext_171)->scan_0 = scan_62;
    (&kernelContext_171)->parts_0 = parts_62;
    (&kernelContext_171)->colliders_0 = colliders_62;
    (&kernelContext_171)->collider_info_0 = collider_info_62;
    (&kernelContext_171)->view_chunks_0 = view_chunks_62;
    (&kernelContext_171)->view_0 = view_62;
    (&kernelContext_171)->objects_0 = objects_62;
    (&kernelContext_171)->shot_bodies_0 = shot_bodies_62;
    (&kernelContext_171)->lines_0 = lines_62;
    (&kernelContext_171)->geometry_0 = geometry_62;
    (&kernelContext_171)->shots_0 = shots_62;
    (&kernelContext_171)->lp_0 = lp_62;
    (&kernelContext_171)->ext_0 = ext_64;
    (&kernelContext_171)->lp_sides_0 = lp_sides_62;
    (&kernelContext_171)->engine_counters_0 = engine_counters_62;
    (&kernelContext_171)->engine_contacts_0 = engine_contacts_62;
    (&kernelContext_171)->cs_anchors_0 = cs_anchors_62;
    threadgroup array<uint, int(1024)> g_scan_62;
    (&kernelContext_171)->g_scan_0 = &g_scan_62;
    threadgroup array<float, int(64)> g_il_best_62;
    (&kernelContext_171)->g_il_best_0 = &g_il_best_62;
    cs_pick_1(id_57.x, &kernelContext_171);
    return;
}

[[kernel]] void fr_cs_pick_d(uint3 id_58 [[thread_position_in_grid]], FractureParams_0 constant* params_63 [[buffer(0)]], uint device* work_63 [[buffer(4)]], packed_float4 device* frames_63 [[buffer(6)]], packed_float4 device* bodies_63 [[buffer(7)]], packed_float4 device* chunks_63 [[buffer(1)]], uint device* sort_63 [[buffer(11)]], atomic<uint> device* atom_63 [[buffer(5)]], Record_natural_0 device* manifolds_63 [[buffer(10)]], packed_float4 device* loads_63 [[buffer(12)]], packed_uint4 device* bonds_63 [[buffer(2)]], uint device* args_63 [[buffer(13)]], uint device* info_63 [[buffer(15)]], ScanParams_0 constant* scan_63 [[buffer(14)]], packed_float4 device* parts_63 [[buffer(3)]], packed_float4 device* colliders_63 [[buffer(8)]], packed_uint4 device* collider_info_63 [[buffer(9)]], packed_float4 device* view_chunks_63 [[buffer(17)]], ViewParams_0 constant* view_63 [[buffer(16)]], packed_float4 device* objects_63 [[buffer(18)]], uint device* shot_bodies_63 [[buffer(20)]], float device* lines_63 [[buffer(19)]], packed_float4 device* geometry_63 [[buffer(21)]], packed_float4 device* shots_63 [[buffer(22)]], packed_float4 device* lp_63 [[buffer(24)]], packed_float4 device* ext_65 [[buffer(23)]], packed_float4 device* lp_sides_63 [[buffer(28)]], uint device* engine_counters_63 [[buffer(26)]], packed_float4 device* engine_contacts_63 [[buffer(25)]], packed_float4 device* cs_anchors_63 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_172;
    (&kernelContext_172)->params_0 = params_63;
    (&kernelContext_172)->work_0 = work_63;
    (&kernelContext_172)->frames_0 = frames_63;
    (&kernelContext_172)->bodies_0 = bodies_63;
    (&kernelContext_172)->chunks_0 = chunks_63;
    (&kernelContext_172)->sort_0 = sort_63;
    (&kernelContext_172)->atom_0 = atom_63;
    (&kernelContext_172)->manifolds_0 = manifolds_63;
    (&kernelContext_172)->loads_0 = loads_63;
    (&kernelContext_172)->bonds_0 = bonds_63;
    (&kernelContext_172)->args_0 = args_63;
    (&kernelContext_172)->info_0 = info_63;
    (&kernelContext_172)->scan_0 = scan_63;
    (&kernelContext_172)->parts_0 = parts_63;
    (&kernelContext_172)->colliders_0 = colliders_63;
    (&kernelContext_172)->collider_info_0 = collider_info_63;
    (&kernelContext_172)->view_chunks_0 = view_chunks_63;
    (&kernelContext_172)->view_0 = view_63;
    (&kernelContext_172)->objects_0 = objects_63;
    (&kernelContext_172)->shot_bodies_0 = shot_bodies_63;
    (&kernelContext_172)->lines_0 = lines_63;
    (&kernelContext_172)->geometry_0 = geometry_63;
    (&kernelContext_172)->shots_0 = shots_63;
    (&kernelContext_172)->lp_0 = lp_63;
    (&kernelContext_172)->ext_0 = ext_65;
    (&kernelContext_172)->lp_sides_0 = lp_sides_63;
    (&kernelContext_172)->engine_counters_0 = engine_counters_63;
    (&kernelContext_172)->engine_contacts_0 = engine_contacts_63;
    (&kernelContext_172)->cs_anchors_0 = cs_anchors_63;
    threadgroup array<uint, int(1024)> g_scan_63;
    (&kernelContext_172)->g_scan_0 = &g_scan_63;
    threadgroup array<float, int(64)> g_il_best_63;
    (&kernelContext_172)->g_il_best_0 = &g_il_best_63;
    cs_pick_0(id_58.x, &kernelContext_172);
    return;
}

[[kernel]] void fr_cs_record(uint3 id_59 [[thread_position_in_grid]], FractureParams_0 constant* params_64 [[buffer(0)]], uint device* work_64 [[buffer(4)]], packed_float4 device* frames_64 [[buffer(6)]], packed_float4 device* bodies_64 [[buffer(7)]], packed_float4 device* chunks_64 [[buffer(1)]], uint device* sort_64 [[buffer(11)]], atomic<uint> device* atom_64 [[buffer(5)]], Record_natural_0 device* manifolds_64 [[buffer(10)]], packed_float4 device* loads_64 [[buffer(12)]], packed_uint4 device* bonds_64 [[buffer(2)]], uint device* args_64 [[buffer(13)]], uint device* info_64 [[buffer(15)]], ScanParams_0 constant* scan_64 [[buffer(14)]], packed_float4 device* parts_64 [[buffer(3)]], packed_float4 device* colliders_64 [[buffer(8)]], packed_uint4 device* collider_info_64 [[buffer(9)]], packed_float4 device* view_chunks_64 [[buffer(17)]], ViewParams_0 constant* view_64 [[buffer(16)]], packed_float4 device* objects_64 [[buffer(18)]], uint device* shot_bodies_64 [[buffer(20)]], float device* lines_64 [[buffer(19)]], packed_float4 device* geometry_64 [[buffer(21)]], packed_float4 device* shots_64 [[buffer(22)]], packed_float4 device* lp_64 [[buffer(24)]], packed_float4 device* ext_66 [[buffer(23)]], packed_float4 device* lp_sides_64 [[buffer(28)]], uint device* engine_counters_64 [[buffer(26)]], packed_float4 device* engine_contacts_64 [[buffer(25)]], packed_float4 device* cs_anchors_64 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_173;
    (&kernelContext_173)->params_0 = params_64;
    (&kernelContext_173)->work_0 = work_64;
    (&kernelContext_173)->frames_0 = frames_64;
    (&kernelContext_173)->bodies_0 = bodies_64;
    (&kernelContext_173)->chunks_0 = chunks_64;
    (&kernelContext_173)->sort_0 = sort_64;
    (&kernelContext_173)->atom_0 = atom_64;
    (&kernelContext_173)->manifolds_0 = manifolds_64;
    (&kernelContext_173)->loads_0 = loads_64;
    (&kernelContext_173)->bonds_0 = bonds_64;
    (&kernelContext_173)->args_0 = args_64;
    (&kernelContext_173)->info_0 = info_64;
    (&kernelContext_173)->scan_0 = scan_64;
    (&kernelContext_173)->parts_0 = parts_64;
    (&kernelContext_173)->colliders_0 = colliders_64;
    (&kernelContext_173)->collider_info_0 = collider_info_64;
    (&kernelContext_173)->view_chunks_0 = view_chunks_64;
    (&kernelContext_173)->view_0 = view_64;
    (&kernelContext_173)->objects_0 = objects_64;
    (&kernelContext_173)->shot_bodies_0 = shot_bodies_64;
    (&kernelContext_173)->lines_0 = lines_64;
    (&kernelContext_173)->geometry_0 = geometry_64;
    (&kernelContext_173)->shots_0 = shots_64;
    (&kernelContext_173)->lp_0 = lp_64;
    (&kernelContext_173)->ext_0 = ext_66;
    (&kernelContext_173)->lp_sides_0 = lp_sides_64;
    (&kernelContext_173)->engine_counters_0 = engine_counters_64;
    (&kernelContext_173)->engine_contacts_0 = engine_contacts_64;
    (&kernelContext_173)->cs_anchors_0 = cs_anchors_64;
    threadgroup array<uint, int(1024)> g_scan_64;
    (&kernelContext_173)->g_scan_0 = &g_scan_64;
    threadgroup array<float, int(64)> g_il_best_64;
    (&kernelContext_173)->g_il_best_0 = &g_il_best_64;
    uint _S1173 = id_59.x;
    uint _S1174 = a_counters_1(&kernelContext_173);
    uint _S1175 = atomic_load_explicit((&kernelContext_173)->atom_0+(_S1174 + 14U), memory_order_relaxed);
    if(_S1173 >= _S1175)
    {
        return;
    }
    uint _S1176 = w_cs_bodies_0(&kernelContext_173);
    uint device* _S1177 = (&kernelContext_173)->work_0+(_S1176 + _S1173);
    uint s_18 = *_S1177;
    uint _S1178 = a_cs_0(*_S1177, &kernelContext_173);
    uint _S1179 = atomic_load_explicit((&kernelContext_173)->atom_0+(_S1178 + 11U), memory_order_relaxed);
    uint key_2 = _S1179 - 1U;
    thread uint k_45;
    thread uint s2_0;
    thread uint ob_6;
    thread uint co_6;
    thread float3 p_19;
    thread float3 u_10;
    thread float v_15;
    thread float m_eff_8;
    thread float t_in_9;
    bool _S1180 = cs_side_0(key_2 / 2U, key_2 % 2U, &k_45, &s2_0, &ob_6, &co_6, &p_19, &u_10, &v_15, &m_eff_8, &t_in_9, &kernelContext_173);
    if(!_S1180)
    {
        return;
    }
    uint4 _S1181 = uint4(*((&kernelContext_173)->collider_info_0+(2U * co_6 + 1U))) ;
    float e_o_0 = (as_type<float>((_S1181.y)));
    float _S1182 = max((as_type<float>((_S1181.z))), 0.00100000004749745f);
    float j_9 = m_eff_8 * v_15;
    float tau_2 = hertz_duration_0(m_eff_8, _S1182, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / (float4(*((&kernelContext_173)->geometry_0+(3U * k_45 + 1U))) ).w), v_15);
    uint _S1183 = lp_body_0(s_18, &kernelContext_173);
    *((&kernelContext_173)->lp_0+(_S1183 + 2U)) = packed_float4(float4(from_rubble_0(p_19), 3.14159274101257324f * j_9 / (2.0f * tau_2))) ;
    *((&kernelContext_173)->lp_0+(_S1183 + 3U)) = packed_float4(float4(from_rubble_0(u_10), v_15)) ;
    *((&kernelContext_173)->lp_0+(_S1183 + 4U)) = packed_float4(float4(m_eff_8, _S1182, e_o_0, (as_type<float>((ob_6))))) ;
    *((&kernelContext_173)->lp_0+(_S1183 + 5U)) = packed_float4(float4((as_type<float>((k_45))), j_9 / tau_2, 1.0f / (float4(*((&kernelContext_173)->bodies_0+7U * ob_6)) ).w, t_in_9)) ;
    *((&kernelContext_173)->lp_0+(_S1183 + 6U)) = packed_float4(float4((as_type<float>((co_6))), (as_type<float>((_S1181.w))), 0.0f, 0.0f)) ;
    atomic_store_explicit((&kernelContext_173)->atom_0+(_S1178 + 1U), k_45 + 1U, memory_order_relaxed);
    uint _S1184 = atomic_fetch_add_explicit((&kernelContext_173)->atom_0+(_S1174 + 12U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_hertz(uint3 id_60 [[thread_position_in_grid]], FractureParams_0 constant* params_65 [[buffer(0)]], uint device* work_65 [[buffer(4)]], packed_float4 device* frames_65 [[buffer(6)]], packed_float4 device* bodies_65 [[buffer(7)]], packed_float4 device* chunks_65 [[buffer(1)]], uint device* sort_65 [[buffer(11)]], atomic<uint> device* atom_65 [[buffer(5)]], Record_natural_0 device* manifolds_65 [[buffer(10)]], packed_float4 device* loads_65 [[buffer(12)]], packed_uint4 device* bonds_65 [[buffer(2)]], uint device* args_65 [[buffer(13)]], uint device* info_65 [[buffer(15)]], ScanParams_0 constant* scan_65 [[buffer(14)]], packed_float4 device* parts_65 [[buffer(3)]], packed_float4 device* colliders_65 [[buffer(8)]], packed_uint4 device* collider_info_65 [[buffer(9)]], packed_float4 device* view_chunks_65 [[buffer(17)]], ViewParams_0 constant* view_65 [[buffer(16)]], packed_float4 device* objects_65 [[buffer(18)]], uint device* shot_bodies_65 [[buffer(20)]], float device* lines_65 [[buffer(19)]], packed_float4 device* geometry_65 [[buffer(21)]], packed_float4 device* shots_65 [[buffer(22)]], packed_float4 device* lp_65 [[buffer(24)]], packed_float4 device* ext_67 [[buffer(23)]], packed_float4 device* lp_sides_65 [[buffer(28)]], uint device* engine_counters_65 [[buffer(26)]], packed_float4 device* engine_contacts_65 [[buffer(25)]], packed_float4 device* cs_anchors_65 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_174;
    (&kernelContext_174)->params_0 = params_65;
    (&kernelContext_174)->work_0 = work_65;
    (&kernelContext_174)->frames_0 = frames_65;
    (&kernelContext_174)->bodies_0 = bodies_65;
    (&kernelContext_174)->chunks_0 = chunks_65;
    (&kernelContext_174)->sort_0 = sort_65;
    (&kernelContext_174)->atom_0 = atom_65;
    (&kernelContext_174)->manifolds_0 = manifolds_65;
    (&kernelContext_174)->loads_0 = loads_65;
    (&kernelContext_174)->bonds_0 = bonds_65;
    (&kernelContext_174)->args_0 = args_65;
    (&kernelContext_174)->info_0 = info_65;
    (&kernelContext_174)->scan_0 = scan_65;
    (&kernelContext_174)->parts_0 = parts_65;
    (&kernelContext_174)->colliders_0 = colliders_65;
    (&kernelContext_174)->collider_info_0 = collider_info_65;
    (&kernelContext_174)->view_chunks_0 = view_chunks_65;
    (&kernelContext_174)->view_0 = view_65;
    (&kernelContext_174)->objects_0 = objects_65;
    (&kernelContext_174)->shot_bodies_0 = shot_bodies_65;
    (&kernelContext_174)->lines_0 = lines_65;
    (&kernelContext_174)->geometry_0 = geometry_65;
    (&kernelContext_174)->shots_0 = shots_65;
    (&kernelContext_174)->lp_0 = lp_65;
    (&kernelContext_174)->ext_0 = ext_67;
    (&kernelContext_174)->lp_sides_0 = lp_sides_65;
    (&kernelContext_174)->engine_counters_0 = engine_counters_65;
    (&kernelContext_174)->engine_contacts_0 = engine_contacts_65;
    (&kernelContext_174)->cs_anchors_0 = cs_anchors_65;
    threadgroup array<uint, int(1024)> g_scan_65;
    (&kernelContext_174)->g_scan_0 = &g_scan_65;
    threadgroup array<float, int(64)> g_il_best_65;
    (&kernelContext_174)->g_il_best_0 = &g_il_best_65;
    uint i_15 = id_60.x;
    bool _S1185;
    if(i_15 >= (params_65->bond_count_0))
    {
        _S1185 = true;
    }
    else
    {
        uint _S1186 = w_alive_0(&kernelContext_174);
        _S1185 = (*((&kernelContext_174)->work_0+(_S1186 + i_15))) == 0U;
    }
    if(_S1185)
    {
        return;
    }
    uint4 _S1187 = uint4(*((&kernelContext_174)->bonds_0+i_15)) ;
    uint _S1188 = body_of_0(_S1187.x, &kernelContext_174);
    bool _S1189 = cs_struck_1(_S1188, &kernelContext_174);
    if(!_S1189)
    {
        return;
    }
    uint _S1190 = lp_body_0(_S1188, &kernelContext_174);
    bool _S1191 = strike_share_breaks_0(_S1187, (float4(*((&kernelContext_174)->lp_0+(_S1190 + 2U))) ).xyz, (float4(*((&kernelContext_174)->lp_0+(_S1190 + 5U))) ).y, &kernelContext_174);
    if(_S1191)
    {
        uint _S1192 = w_alive_0(&kernelContext_174);
        *((&kernelContext_174)->work_0+(_S1192 + i_15)) = 0U;
        uint _S1193 = a_dirty_0(&kernelContext_174);
        uint _S1194 = atomic_fetch_or_explicit((&kernelContext_174)->atom_0+(_S1193 + _S1188), 1U, memory_order_relaxed);
        uint _S1195 = a_counters_1(&kernelContext_174);
        uint _S1196 = atomic_fetch_add_explicit((&kernelContext_174)->atom_0+_S1195, 1U, memory_order_relaxed);
        uint _S1197 = atomic_fetch_add_explicit((&kernelContext_174)->atom_0+(_S1195 + 1U), 1U, memory_order_relaxed);
        uint _S1198 = atomic_fetch_add_explicit((&kernelContext_174)->atom_0+(_S1195 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_gate2(uint3 id_61 [[thread_position_in_grid]], FractureParams_0 constant* params_66 [[buffer(0)]], uint device* work_66 [[buffer(4)]], packed_float4 device* frames_66 [[buffer(6)]], packed_float4 device* bodies_66 [[buffer(7)]], packed_float4 device* chunks_66 [[buffer(1)]], uint device* sort_66 [[buffer(11)]], atomic<uint> device* atom_66 [[buffer(5)]], Record_natural_0 device* manifolds_66 [[buffer(10)]], packed_float4 device* loads_66 [[buffer(12)]], packed_uint4 device* bonds_66 [[buffer(2)]], uint device* args_66 [[buffer(13)]], uint device* info_66 [[buffer(15)]], ScanParams_0 constant* scan_66 [[buffer(14)]], packed_float4 device* parts_66 [[buffer(3)]], packed_float4 device* colliders_66 [[buffer(8)]], packed_uint4 device* collider_info_66 [[buffer(9)]], packed_float4 device* view_chunks_66 [[buffer(17)]], ViewParams_0 constant* view_66 [[buffer(16)]], packed_float4 device* objects_66 [[buffer(18)]], uint device* shot_bodies_66 [[buffer(20)]], float device* lines_66 [[buffer(19)]], packed_float4 device* geometry_66 [[buffer(21)]], packed_float4 device* shots_66 [[buffer(22)]], packed_float4 device* lp_66 [[buffer(24)]], packed_float4 device* ext_68 [[buffer(23)]], packed_float4 device* lp_sides_66 [[buffer(28)]], uint device* engine_counters_66 [[buffer(26)]], packed_float4 device* engine_contacts_66 [[buffer(25)]], packed_float4 device* cs_anchors_66 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_175;
    (&kernelContext_175)->params_0 = params_66;
    (&kernelContext_175)->work_0 = work_66;
    (&kernelContext_175)->frames_0 = frames_66;
    (&kernelContext_175)->bodies_0 = bodies_66;
    (&kernelContext_175)->chunks_0 = chunks_66;
    (&kernelContext_175)->sort_0 = sort_66;
    (&kernelContext_175)->atom_0 = atom_66;
    (&kernelContext_175)->manifolds_0 = manifolds_66;
    (&kernelContext_175)->loads_0 = loads_66;
    (&kernelContext_175)->bonds_0 = bonds_66;
    (&kernelContext_175)->args_0 = args_66;
    (&kernelContext_175)->info_0 = info_66;
    (&kernelContext_175)->scan_0 = scan_66;
    (&kernelContext_175)->parts_0 = parts_66;
    (&kernelContext_175)->colliders_0 = colliders_66;
    (&kernelContext_175)->collider_info_0 = collider_info_66;
    (&kernelContext_175)->view_chunks_0 = view_chunks_66;
    (&kernelContext_175)->view_0 = view_66;
    (&kernelContext_175)->objects_0 = objects_66;
    (&kernelContext_175)->shot_bodies_0 = shot_bodies_66;
    (&kernelContext_175)->lines_0 = lines_66;
    (&kernelContext_175)->geometry_0 = geometry_66;
    (&kernelContext_175)->shots_0 = shots_66;
    (&kernelContext_175)->lp_0 = lp_66;
    (&kernelContext_175)->ext_0 = ext_68;
    (&kernelContext_175)->lp_sides_0 = lp_sides_66;
    (&kernelContext_175)->engine_counters_0 = engine_counters_66;
    (&kernelContext_175)->engine_contacts_0 = engine_contacts_66;
    (&kernelContext_175)->cs_anchors_0 = cs_anchors_66;
    threadgroup array<uint, int(1024)> g_scan_66;
    (&kernelContext_175)->g_scan_0 = &g_scan_66;
    threadgroup array<float, int(64)> g_il_best_66;
    (&kernelContext_175)->g_il_best_0 = &g_il_best_66;
    uint _S1199 = a_counters_1(&kernelContext_175);
    uint _S1200 = atomic_load_explicit((&kernelContext_175)->atom_0+_S1199, memory_order_relaxed);
    uint _S1201;
    if(_S1200 > 0U)
    {
        uint _S1202 = cs_rows_0(&kernelContext_175);
        uint _S1203 = groups_of_0(_S1202, 64U);
        _S1201 = _S1203;
    }
    else
    {
        _S1201 = 0U;
    }
    set_args_0(13U, _S1201, &kernelContext_175);
    return;
}

[[kernel]] void fr_cs_save(uint3 id_62 [[thread_position_in_grid]], FractureParams_0 constant* params_67 [[buffer(0)]], uint device* work_67 [[buffer(4)]], packed_float4 device* frames_67 [[buffer(6)]], packed_float4 device* bodies_67 [[buffer(7)]], packed_float4 device* chunks_67 [[buffer(1)]], uint device* sort_67 [[buffer(11)]], atomic<uint> device* atom_67 [[buffer(5)]], Record_natural_0 device* manifolds_67 [[buffer(10)]], packed_float4 device* loads_67 [[buffer(12)]], packed_uint4 device* bonds_67 [[buffer(2)]], uint device* args_67 [[buffer(13)]], uint device* info_67 [[buffer(15)]], ScanParams_0 constant* scan_67 [[buffer(14)]], packed_float4 device* parts_67 [[buffer(3)]], packed_float4 device* colliders_67 [[buffer(8)]], packed_uint4 device* collider_info_67 [[buffer(9)]], packed_float4 device* view_chunks_67 [[buffer(17)]], ViewParams_0 constant* view_67 [[buffer(16)]], packed_float4 device* objects_67 [[buffer(18)]], uint device* shot_bodies_67 [[buffer(20)]], float device* lines_67 [[buffer(19)]], packed_float4 device* geometry_67 [[buffer(21)]], packed_float4 device* shots_67 [[buffer(22)]], packed_float4 device* lp_67 [[buffer(24)]], packed_float4 device* ext_69 [[buffer(23)]], packed_float4 device* lp_sides_67 [[buffer(28)]], uint device* engine_counters_67 [[buffer(26)]], packed_float4 device* engine_contacts_67 [[buffer(25)]], packed_float4 device* cs_anchors_67 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_176;
    (&kernelContext_176)->params_0 = params_67;
    (&kernelContext_176)->work_0 = work_67;
    (&kernelContext_176)->frames_0 = frames_67;
    (&kernelContext_176)->bodies_0 = bodies_67;
    (&kernelContext_176)->chunks_0 = chunks_67;
    (&kernelContext_176)->sort_0 = sort_67;
    (&kernelContext_176)->atom_0 = atom_67;
    (&kernelContext_176)->manifolds_0 = manifolds_67;
    (&kernelContext_176)->loads_0 = loads_67;
    (&kernelContext_176)->bonds_0 = bonds_67;
    (&kernelContext_176)->args_0 = args_67;
    (&kernelContext_176)->info_0 = info_67;
    (&kernelContext_176)->scan_0 = scan_67;
    (&kernelContext_176)->parts_0 = parts_67;
    (&kernelContext_176)->colliders_0 = colliders_67;
    (&kernelContext_176)->collider_info_0 = collider_info_67;
    (&kernelContext_176)->view_chunks_0 = view_chunks_67;
    (&kernelContext_176)->view_0 = view_67;
    (&kernelContext_176)->objects_0 = objects_67;
    (&kernelContext_176)->shot_bodies_0 = shot_bodies_67;
    (&kernelContext_176)->lines_0 = lines_67;
    (&kernelContext_176)->geometry_0 = geometry_67;
    (&kernelContext_176)->shots_0 = shots_67;
    (&kernelContext_176)->lp_0 = lp_67;
    (&kernelContext_176)->ext_0 = ext_69;
    (&kernelContext_176)->lp_sides_0 = lp_sides_67;
    (&kernelContext_176)->engine_counters_0 = engine_counters_67;
    (&kernelContext_176)->engine_contacts_0 = engine_contacts_67;
    (&kernelContext_176)->cs_anchors_0 = cs_anchors_67;
    threadgroup array<uint, int(1024)> g_scan_67;
    (&kernelContext_176)->g_scan_0 = &g_scan_67;
    threadgroup array<float, int(64)> g_il_best_67;
    (&kernelContext_176)->g_il_best_0 = &g_il_best_67;
    uint r_26 = id_62.x;
    uint _S1204 = cs_rows_0(&kernelContext_176);
    if(r_26 >= _S1204)
    {
        return;
    }
    uint _S1205 = cs_row_0(r_26, &kernelContext_176);
    uint4 _S1206 = (as_type<uint4>((float4(*((&kernelContext_176)->engine_contacts_0+_S1205)) )));
    uint side_7 = 0U;
    for(;;)
    {
        if(side_7 < 2U)
        {
        }
        else
        {
            break;
        }
        float4 _S1207 = float4(0.0f) ;
        bool _S1208 = side_7 == 0U;
        uint col_1;
        if(_S1208)
        {
            col_1 = _S1206.x;
        }
        else
        {
            col_1 = _S1206.y;
        }
        uint k_46;
        if((_S1206.w) != 0U)
        {
            k_46 = (uint4(*((&kernelContext_176)->collider_info_0+2U * col_1)) ).x;
        }
        else
        {
            k_46 = 4294967295U;
        }
        bool _S1209;
        if(k_46 != 4294967295U)
        {
            _S1209 = k_46 < ((&kernelContext_176)->params_0->chunk_count_0);
        }
        else
        {
            _S1209 = false;
        }
        bool _S1210;
        if(_S1209)
        {
            uint _S1211 = body_of_0(k_46, &kernelContext_176);
            bool _S1212 = dirty_0(_S1211, &kernelContext_176);
            _S1210 = _S1212;
        }
        else
        {
            _S1210 = false;
        }
        float4 saved_0;
        if(_S1210)
        {
            uint _S1213 = body_of_0(k_46, &kernelContext_176);
            uint at_20 = 7U * _S1213;
            float3 anchor_0;
            if(_S1208)
            {
                anchor_0 = (float4(*((&kernelContext_176)->engine_contacts_0+(_S1205 + 2U))) ).xyz;
            }
            else
            {
                anchor_0 = (float4(*((&kernelContext_176)->engine_contacts_0+(_S1205 + 3U))) ).xyz;
            }
            float3 _S1214 = (float4(*((&kernelContext_176)->bodies_0+at_20)) ).xyz;
            thread Quat_0 _S1215 = quat_of_0(float4(*((&kernelContext_176)->bodies_0+(at_20 + 1U))) );
            float3 _S1216 = rotate_0(&_S1215, anchor_0);
            saved_0 = float4(_S1214 + _S1216, 1.0f);
        }
        else
        {
            saved_0 = _S1207;
        }
        *((&kernelContext_176)->cs_anchors_0+(2U * r_26 + side_7)) = packed_float4(saved_0) ;
        side_7 = side_7 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_clean(uint3 id_63 [[thread_position_in_grid]], FractureParams_0 constant* params_68 [[buffer(0)]], uint device* work_68 [[buffer(4)]], packed_float4 device* frames_68 [[buffer(6)]], packed_float4 device* bodies_68 [[buffer(7)]], packed_float4 device* chunks_68 [[buffer(1)]], uint device* sort_68 [[buffer(11)]], atomic<uint> device* atom_68 [[buffer(5)]], Record_natural_0 device* manifolds_68 [[buffer(10)]], packed_float4 device* loads_68 [[buffer(12)]], packed_uint4 device* bonds_68 [[buffer(2)]], uint device* args_68 [[buffer(13)]], uint device* info_68 [[buffer(15)]], ScanParams_0 constant* scan_68 [[buffer(14)]], packed_float4 device* parts_68 [[buffer(3)]], packed_float4 device* colliders_68 [[buffer(8)]], packed_uint4 device* collider_info_68 [[buffer(9)]], packed_float4 device* view_chunks_68 [[buffer(17)]], ViewParams_0 constant* view_68 [[buffer(16)]], packed_float4 device* objects_68 [[buffer(18)]], uint device* shot_bodies_68 [[buffer(20)]], float device* lines_68 [[buffer(19)]], packed_float4 device* geometry_68 [[buffer(21)]], packed_float4 device* shots_68 [[buffer(22)]], packed_float4 device* lp_68 [[buffer(24)]], packed_float4 device* ext_70 [[buffer(23)]], packed_float4 device* lp_sides_68 [[buffer(28)]], uint device* engine_counters_68 [[buffer(26)]], packed_float4 device* engine_contacts_68 [[buffer(25)]], packed_float4 device* cs_anchors_68 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_177;
    (&kernelContext_177)->params_0 = params_68;
    (&kernelContext_177)->work_0 = work_68;
    (&kernelContext_177)->frames_0 = frames_68;
    (&kernelContext_177)->bodies_0 = bodies_68;
    (&kernelContext_177)->chunks_0 = chunks_68;
    (&kernelContext_177)->sort_0 = sort_68;
    (&kernelContext_177)->atom_0 = atom_68;
    (&kernelContext_177)->manifolds_0 = manifolds_68;
    (&kernelContext_177)->loads_0 = loads_68;
    (&kernelContext_177)->bonds_0 = bonds_68;
    (&kernelContext_177)->args_0 = args_68;
    (&kernelContext_177)->info_0 = info_68;
    (&kernelContext_177)->scan_0 = scan_68;
    (&kernelContext_177)->parts_0 = parts_68;
    (&kernelContext_177)->colliders_0 = colliders_68;
    (&kernelContext_177)->collider_info_0 = collider_info_68;
    (&kernelContext_177)->view_chunks_0 = view_chunks_68;
    (&kernelContext_177)->view_0 = view_68;
    (&kernelContext_177)->objects_0 = objects_68;
    (&kernelContext_177)->shot_bodies_0 = shot_bodies_68;
    (&kernelContext_177)->lines_0 = lines_68;
    (&kernelContext_177)->geometry_0 = geometry_68;
    (&kernelContext_177)->shots_0 = shots_68;
    (&kernelContext_177)->lp_0 = lp_68;
    (&kernelContext_177)->ext_0 = ext_70;
    (&kernelContext_177)->lp_sides_0 = lp_sides_68;
    (&kernelContext_177)->engine_counters_0 = engine_counters_68;
    (&kernelContext_177)->engine_contacts_0 = engine_contacts_68;
    (&kernelContext_177)->cs_anchors_0 = cs_anchors_68;
    threadgroup array<uint, int(1024)> g_scan_68;
    (&kernelContext_177)->g_scan_0 = &g_scan_68;
    threadgroup array<float, int(64)> g_il_best_68;
    (&kernelContext_177)->g_il_best_0 = &g_il_best_68;
    uint _S1217 = id_63.x;
    uint _S1218 = a_counters_1(&kernelContext_177);
    uint _S1219 = atomic_load_explicit((&kernelContext_177)->atom_0+(_S1218 + 14U), memory_order_relaxed);
    if(_S1217 >= _S1219)
    {
        return;
    }
    uint _S1220 = w_cs_bodies_0(&kernelContext_177);
    uint _S1221 = a_cs_0(*((&kernelContext_177)->work_0+(_S1220 + _S1217)), &kernelContext_177);
    uint w_8 = 0U;
    for(;;)
    {
        if(w_8 < 12U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit((&kernelContext_177)->atom_0+(_S1221 + w_8), 0U, memory_order_relaxed);
        w_8 = w_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_fix(uint3 id_64 [[thread_position_in_grid]], FractureParams_0 constant* params_69 [[buffer(0)]], uint device* work_69 [[buffer(4)]], packed_float4 device* frames_69 [[buffer(6)]], packed_float4 device* bodies_69 [[buffer(7)]], packed_float4 device* chunks_69 [[buffer(1)]], uint device* sort_69 [[buffer(11)]], atomic<uint> device* atom_69 [[buffer(5)]], Record_natural_0 device* manifolds_69 [[buffer(10)]], packed_float4 device* loads_69 [[buffer(12)]], packed_uint4 device* bonds_69 [[buffer(2)]], uint device* args_69 [[buffer(13)]], uint device* info_69 [[buffer(15)]], ScanParams_0 constant* scan_69 [[buffer(14)]], packed_float4 device* parts_69 [[buffer(3)]], packed_float4 device* colliders_69 [[buffer(8)]], packed_uint4 device* collider_info_69 [[buffer(9)]], packed_float4 device* view_chunks_69 [[buffer(17)]], ViewParams_0 constant* view_69 [[buffer(16)]], packed_float4 device* objects_69 [[buffer(18)]], uint device* shot_bodies_69 [[buffer(20)]], float device* lines_69 [[buffer(19)]], packed_float4 device* geometry_69 [[buffer(21)]], packed_float4 device* shots_69 [[buffer(22)]], packed_float4 device* lp_69 [[buffer(24)]], packed_float4 device* ext_71 [[buffer(23)]], packed_float4 device* lp_sides_69 [[buffer(28)]], uint device* engine_counters_69 [[buffer(26)]], packed_float4 device* engine_contacts_69 [[buffer(25)]], packed_float4 device* cs_anchors_69 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_178;
    (&kernelContext_178)->params_0 = params_69;
    (&kernelContext_178)->work_0 = work_69;
    (&kernelContext_178)->frames_0 = frames_69;
    (&kernelContext_178)->bodies_0 = bodies_69;
    (&kernelContext_178)->chunks_0 = chunks_69;
    (&kernelContext_178)->sort_0 = sort_69;
    (&kernelContext_178)->atom_0 = atom_69;
    (&kernelContext_178)->manifolds_0 = manifolds_69;
    (&kernelContext_178)->loads_0 = loads_69;
    (&kernelContext_178)->bonds_0 = bonds_69;
    (&kernelContext_178)->args_0 = args_69;
    (&kernelContext_178)->info_0 = info_69;
    (&kernelContext_178)->scan_0 = scan_69;
    (&kernelContext_178)->parts_0 = parts_69;
    (&kernelContext_178)->colliders_0 = colliders_69;
    (&kernelContext_178)->collider_info_0 = collider_info_69;
    (&kernelContext_178)->view_chunks_0 = view_chunks_69;
    (&kernelContext_178)->view_0 = view_69;
    (&kernelContext_178)->objects_0 = objects_69;
    (&kernelContext_178)->shot_bodies_0 = shot_bodies_69;
    (&kernelContext_178)->lines_0 = lines_69;
    (&kernelContext_178)->geometry_0 = geometry_69;
    (&kernelContext_178)->shots_0 = shots_69;
    (&kernelContext_178)->lp_0 = lp_69;
    (&kernelContext_178)->ext_0 = ext_71;
    (&kernelContext_178)->lp_sides_0 = lp_sides_69;
    (&kernelContext_178)->engine_counters_0 = engine_counters_69;
    (&kernelContext_178)->engine_contacts_0 = engine_contacts_69;
    (&kernelContext_178)->cs_anchors_0 = cs_anchors_69;
    threadgroup array<uint, int(1024)> g_scan_69;
    (&kernelContext_178)->g_scan_0 = &g_scan_69;
    threadgroup array<float, int(64)> g_il_best_69;
    (&kernelContext_178)->g_il_best_0 = &g_il_best_69;
    uint r_27 = id_64.x;
    uint _S1222 = cs_rows_0(&kernelContext_178);
    if(r_27 >= _S1222)
    {
        return;
    }
    uint _S1223 = cs_row_0(r_27, &kernelContext_178);
    uint4 _S1224 = (as_type<uint4>((float4(*((&kernelContext_178)->engine_contacts_0+_S1223)) )));
    bool moved_0 = false;
    uint side_8 = 0U;
    for(;;)
    {
        if(side_8 < 2U)
        {
        }
        else
        {
            break;
        }
        float4 _S1225 = float4(*((&kernelContext_178)->cs_anchors_0+(2U * r_27 + side_8))) ;
        if((_S1225.w) == 0.0f)
        {
            side_8 = side_8 + 1U;
            continue;
        }
        uint col_2;
        if(side_8 == 0U)
        {
            col_2 = _S1224.x;
        }
        else
        {
            col_2 = _S1224.y;
        }
        uint at_21 = 7U * (as_type<uint>(((float4(*((&kernelContext_178)->colliders_0+4U * col_2)) ).x)));
        thread Quat_0 _S1226 = quat_of_0(float4(*((&kernelContext_178)->bodies_0+(at_21 + 1U))) );
        Quat_0 _S1227 = conj_0(&_S1226);
        float3 _S1228 = _S1225.xyz - (float4(*((&kernelContext_178)->bodies_0+at_21)) ).xyz;
        thread Quat_0 _S1229 = _S1227;
        float3 _S1230 = rotate_0(&_S1229, _S1228);
        (*((&kernelContext_178)->engine_contacts_0+(_S1223 + 2U + side_8))).xyz = _S1230;
        moved_0 = true;
        side_8 = side_8 + 1U;
    }
    if(moved_0)
    {
        (*((&kernelContext_178)->engine_contacts_0+(_S1223 + 4U)))[int(3)] = 0.0f;
        (*((&kernelContext_178)->engine_contacts_0+(_S1223 + 5U)))[int(3)] = 0.0f;
        uint _S1231 = _S1223 + 6U;
        *((&kernelContext_178)->engine_contacts_0+_S1231) = packed_float4(float4(1.0f, 1.0f, 0.0f, (float4(*((&kernelContext_178)->engine_contacts_0+_S1231)) ).w)) ;
        (*((&kernelContext_178)->engine_contacts_0+(_S1223 + 7U)))[int(1)] = 0.0f;
    }
    return;
}

[[kernel]] void fr_cs_count(uint3 id_65 [[thread_position_in_grid]], FractureParams_0 constant* params_70 [[buffer(0)]], uint device* work_70 [[buffer(4)]], packed_float4 device* frames_70 [[buffer(6)]], packed_float4 device* bodies_70 [[buffer(7)]], packed_float4 device* chunks_70 [[buffer(1)]], uint device* sort_70 [[buffer(11)]], atomic<uint> device* atom_70 [[buffer(5)]], Record_natural_0 device* manifolds_70 [[buffer(10)]], packed_float4 device* loads_70 [[buffer(12)]], packed_uint4 device* bonds_70 [[buffer(2)]], uint device* args_70 [[buffer(13)]], uint device* info_70 [[buffer(15)]], ScanParams_0 constant* scan_70 [[buffer(14)]], packed_float4 device* parts_70 [[buffer(3)]], packed_float4 device* colliders_70 [[buffer(8)]], packed_uint4 device* collider_info_70 [[buffer(9)]], packed_float4 device* view_chunks_70 [[buffer(17)]], ViewParams_0 constant* view_70 [[buffer(16)]], packed_float4 device* objects_70 [[buffer(18)]], uint device* shot_bodies_70 [[buffer(20)]], float device* lines_70 [[buffer(19)]], packed_float4 device* geometry_70 [[buffer(21)]], packed_float4 device* shots_70 [[buffer(22)]], packed_float4 device* lp_70 [[buffer(24)]], packed_float4 device* ext_72 [[buffer(23)]], packed_float4 device* lp_sides_70 [[buffer(28)]], uint device* engine_counters_70 [[buffer(26)]], packed_float4 device* engine_contacts_70 [[buffer(25)]], packed_float4 device* cs_anchors_70 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_179;
    (&kernelContext_179)->params_0 = params_70;
    (&kernelContext_179)->work_0 = work_70;
    (&kernelContext_179)->frames_0 = frames_70;
    (&kernelContext_179)->bodies_0 = bodies_70;
    (&kernelContext_179)->chunks_0 = chunks_70;
    (&kernelContext_179)->sort_0 = sort_70;
    (&kernelContext_179)->atom_0 = atom_70;
    (&kernelContext_179)->manifolds_0 = manifolds_70;
    (&kernelContext_179)->loads_0 = loads_70;
    (&kernelContext_179)->bonds_0 = bonds_70;
    (&kernelContext_179)->args_0 = args_70;
    (&kernelContext_179)->info_0 = info_70;
    (&kernelContext_179)->scan_0 = scan_70;
    (&kernelContext_179)->parts_0 = parts_70;
    (&kernelContext_179)->colliders_0 = colliders_70;
    (&kernelContext_179)->collider_info_0 = collider_info_70;
    (&kernelContext_179)->view_chunks_0 = view_chunks_70;
    (&kernelContext_179)->view_0 = view_70;
    (&kernelContext_179)->objects_0 = objects_70;
    (&kernelContext_179)->shot_bodies_0 = shot_bodies_70;
    (&kernelContext_179)->lines_0 = lines_70;
    (&kernelContext_179)->geometry_0 = geometry_70;
    (&kernelContext_179)->shots_0 = shots_70;
    (&kernelContext_179)->lp_0 = lp_70;
    (&kernelContext_179)->ext_0 = ext_72;
    (&kernelContext_179)->lp_sides_0 = lp_sides_70;
    (&kernelContext_179)->engine_counters_0 = engine_counters_70;
    (&kernelContext_179)->engine_contacts_0 = engine_contacts_70;
    (&kernelContext_179)->cs_anchors_0 = cs_anchors_70;
    threadgroup array<uint, int(1024)> g_scan_70;
    (&kernelContext_179)->g_scan_0 = &g_scan_70;
    threadgroup array<float, int(64)> g_il_best_70;
    (&kernelContext_179)->g_il_best_0 = &g_il_best_70;
    uint k_47 = id_65.x;
    if(k_47 >= (params_70->chunk_count_0))
    {
        return;
    }
    uint _S1232 = body_of_0(k_47, &kernelContext_179);
    bool _S1233 = cs_struck_1(_S1232, &kernelContext_179);
    if(!_S1233)
    {
        return;
    }
    uint _S1234 = a_cs_0(_S1232, &kernelContext_179);
    uint _S1235 = atomic_fetch_max_explicit((&kernelContext_179)->atom_0+(_S1234 + 3U), ~k_47, memory_order_relaxed);
    if(((&kernelContext_179)->info_0[k_47] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_179)->atom_0+(_S1234 + 4U), 1U, memory_order_relaxed);
    }
    uint _S1236 = atomic_fetch_add_explicit((&kernelContext_179)->atom_0+(_S1234 + 5U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_base(uint3 id_66 [[thread_position_in_grid]], FractureParams_0 constant* params_71 [[buffer(0)]], uint device* work_71 [[buffer(4)]], packed_float4 device* frames_71 [[buffer(6)]], packed_float4 device* bodies_71 [[buffer(7)]], packed_float4 device* chunks_71 [[buffer(1)]], uint device* sort_71 [[buffer(11)]], atomic<uint> device* atom_71 [[buffer(5)]], Record_natural_0 device* manifolds_71 [[buffer(10)]], packed_float4 device* loads_71 [[buffer(12)]], packed_uint4 device* bonds_71 [[buffer(2)]], uint device* args_71 [[buffer(13)]], uint device* info_71 [[buffer(15)]], ScanParams_0 constant* scan_71 [[buffer(14)]], packed_float4 device* parts_71 [[buffer(3)]], packed_float4 device* colliders_71 [[buffer(8)]], packed_uint4 device* collider_info_71 [[buffer(9)]], packed_float4 device* view_chunks_71 [[buffer(17)]], ViewParams_0 constant* view_71 [[buffer(16)]], packed_float4 device* objects_71 [[buffer(18)]], uint device* shot_bodies_71 [[buffer(20)]], float device* lines_71 [[buffer(19)]], packed_float4 device* geometry_71 [[buffer(21)]], packed_float4 device* shots_71 [[buffer(22)]], packed_float4 device* lp_71 [[buffer(24)]], packed_float4 device* ext_73 [[buffer(23)]], packed_float4 device* lp_sides_71 [[buffer(28)]], uint device* engine_counters_71 [[buffer(26)]], packed_float4 device* engine_contacts_71 [[buffer(25)]], packed_float4 device* cs_anchors_71 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_180;
    (&kernelContext_180)->params_0 = params_71;
    (&kernelContext_180)->work_0 = work_71;
    (&kernelContext_180)->frames_0 = frames_71;
    (&kernelContext_180)->bodies_0 = bodies_71;
    (&kernelContext_180)->chunks_0 = chunks_71;
    (&kernelContext_180)->sort_0 = sort_71;
    (&kernelContext_180)->atom_0 = atom_71;
    (&kernelContext_180)->manifolds_0 = manifolds_71;
    (&kernelContext_180)->loads_0 = loads_71;
    (&kernelContext_180)->bonds_0 = bonds_71;
    (&kernelContext_180)->args_0 = args_71;
    (&kernelContext_180)->info_0 = info_71;
    (&kernelContext_180)->scan_0 = scan_71;
    (&kernelContext_180)->parts_0 = parts_71;
    (&kernelContext_180)->colliders_0 = colliders_71;
    (&kernelContext_180)->collider_info_0 = collider_info_71;
    (&kernelContext_180)->view_chunks_0 = view_chunks_71;
    (&kernelContext_180)->view_0 = view_71;
    (&kernelContext_180)->objects_0 = objects_71;
    (&kernelContext_180)->shot_bodies_0 = shot_bodies_71;
    (&kernelContext_180)->lines_0 = lines_71;
    (&kernelContext_180)->geometry_0 = geometry_71;
    (&kernelContext_180)->shots_0 = shots_71;
    (&kernelContext_180)->lp_0 = lp_71;
    (&kernelContext_180)->ext_0 = ext_73;
    (&kernelContext_180)->lp_sides_0 = lp_sides_71;
    (&kernelContext_180)->engine_counters_0 = engine_counters_71;
    (&kernelContext_180)->engine_contacts_0 = engine_contacts_71;
    (&kernelContext_180)->cs_anchors_0 = cs_anchors_71;
    threadgroup array<uint, int(1024)> g_scan_71;
    (&kernelContext_180)->g_scan_0 = &g_scan_71;
    threadgroup array<float, int(64)> g_il_best_71;
    (&kernelContext_180)->g_il_best_0 = &g_il_best_71;
    uint _S1237 = id_66.x;
    uint _S1238 = a_counters_1(&kernelContext_180);
    uint _S1239 = atomic_load_explicit((&kernelContext_180)->atom_0+(_S1238 + 14U), memory_order_relaxed);
    if(_S1237 >= _S1239)
    {
        return;
    }
    uint _S1240 = w_cs_bodies_0(&kernelContext_180);
    uint _S1241 = a_cs_0(*((&kernelContext_180)->work_0+(_S1240 + _S1237)), &kernelContext_180);
    atomic<uint> device* _S1242 = (&kernelContext_180)->atom_0+(_S1241 + 6U);
    atomic<uint> device* _S1243 = (&kernelContext_180)->atom_0+(_S1238 + 15U);
    uint _S1244 = atomic_load_explicit((&kernelContext_180)->atom_0+(_S1241 + 5U), memory_order_relaxed);
    uint _S1245 = atomic_fetch_add_explicit(_S1243, _S1244, memory_order_relaxed);
    atomic_store_explicit(_S1242, _S1245, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_collect(uint3 id_67 [[thread_position_in_grid]], FractureParams_0 constant* params_72 [[buffer(0)]], uint device* work_72 [[buffer(4)]], packed_float4 device* frames_72 [[buffer(6)]], packed_float4 device* bodies_72 [[buffer(7)]], packed_float4 device* chunks_72 [[buffer(1)]], uint device* sort_72 [[buffer(11)]], atomic<uint> device* atom_72 [[buffer(5)]], Record_natural_0 device* manifolds_72 [[buffer(10)]], packed_float4 device* loads_72 [[buffer(12)]], packed_uint4 device* bonds_72 [[buffer(2)]], uint device* args_72 [[buffer(13)]], uint device* info_72 [[buffer(15)]], ScanParams_0 constant* scan_72 [[buffer(14)]], packed_float4 device* parts_72 [[buffer(3)]], packed_float4 device* colliders_72 [[buffer(8)]], packed_uint4 device* collider_info_72 [[buffer(9)]], packed_float4 device* view_chunks_72 [[buffer(17)]], ViewParams_0 constant* view_72 [[buffer(16)]], packed_float4 device* objects_72 [[buffer(18)]], uint device* shot_bodies_72 [[buffer(20)]], float device* lines_72 [[buffer(19)]], packed_float4 device* geometry_72 [[buffer(21)]], packed_float4 device* shots_72 [[buffer(22)]], packed_float4 device* lp_72 [[buffer(24)]], packed_float4 device* ext_74 [[buffer(23)]], packed_float4 device* lp_sides_72 [[buffer(28)]], uint device* engine_counters_72 [[buffer(26)]], packed_float4 device* engine_contacts_72 [[buffer(25)]], packed_float4 device* cs_anchors_72 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_181;
    (&kernelContext_181)->params_0 = params_72;
    (&kernelContext_181)->work_0 = work_72;
    (&kernelContext_181)->frames_0 = frames_72;
    (&kernelContext_181)->bodies_0 = bodies_72;
    (&kernelContext_181)->chunks_0 = chunks_72;
    (&kernelContext_181)->sort_0 = sort_72;
    (&kernelContext_181)->atom_0 = atom_72;
    (&kernelContext_181)->manifolds_0 = manifolds_72;
    (&kernelContext_181)->loads_0 = loads_72;
    (&kernelContext_181)->bonds_0 = bonds_72;
    (&kernelContext_181)->args_0 = args_72;
    (&kernelContext_181)->info_0 = info_72;
    (&kernelContext_181)->scan_0 = scan_72;
    (&kernelContext_181)->parts_0 = parts_72;
    (&kernelContext_181)->colliders_0 = colliders_72;
    (&kernelContext_181)->collider_info_0 = collider_info_72;
    (&kernelContext_181)->view_chunks_0 = view_chunks_72;
    (&kernelContext_181)->view_0 = view_72;
    (&kernelContext_181)->objects_0 = objects_72;
    (&kernelContext_181)->shot_bodies_0 = shot_bodies_72;
    (&kernelContext_181)->lines_0 = lines_72;
    (&kernelContext_181)->geometry_0 = geometry_72;
    (&kernelContext_181)->shots_0 = shots_72;
    (&kernelContext_181)->lp_0 = lp_72;
    (&kernelContext_181)->ext_0 = ext_74;
    (&kernelContext_181)->lp_sides_0 = lp_sides_72;
    (&kernelContext_181)->engine_counters_0 = engine_counters_72;
    (&kernelContext_181)->engine_contacts_0 = engine_contacts_72;
    (&kernelContext_181)->cs_anchors_0 = cs_anchors_72;
    threadgroup array<uint, int(1024)> g_scan_72;
    (&kernelContext_181)->g_scan_0 = &g_scan_72;
    threadgroup array<float, int(64)> g_il_best_72;
    (&kernelContext_181)->g_il_best_0 = &g_il_best_72;
    uint k_48 = id_67.x;
    if(k_48 >= (params_72->chunk_count_0))
    {
        return;
    }
    uint _S1246 = body_of_0(k_48, &kernelContext_181);
    bool _S1247 = cs_struck_1(_S1246, &kernelContext_181);
    if(!_S1247)
    {
        return;
    }
    uint _S1248 = w_cs_list_0(&kernelContext_181);
    uint _S1249 = a_cs_0(_S1246, &kernelContext_181);
    uint _S1250 = atomic_load_explicit((&kernelContext_181)->atom_0+(_S1249 + 6U), memory_order_relaxed);
    uint _S1251 = _S1248 + _S1250;
    uint _S1252 = atomic_fetch_add_explicit((&kernelContext_181)->atom_0+(_S1249 + 7U), 1U, memory_order_relaxed);
    *((&kernelContext_181)->work_0+(_S1251 + _S1252)) = k_48;
    return;
}

void cs_load_0(uint k_49, uint s_19, KernelContext_0 thread* kernelContext_182)
{
    uint _S1253 = a_cs_0(s_19, kernelContext_182);
    uint _S1254 = atomic_load_explicit(kernelContext_182->atom_0+(_S1253 + 4U), memory_order_relaxed);
    bool supported_1 = _S1254 != 0U;
    bool fixed_2;
    if((kernelContext_182->info_0[k_49] & 1U) != 0U)
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            uint _S1255 = atomic_load_explicit(kernelContext_182->atom_0+(_S1253 + 3U), memory_order_relaxed);
            fixed_2 = (~_S1255) == k_49;
        }
        else
        {
            fixed_2 = false;
        }
    }
    uint _S1256 = lp_body_0(s_19, kernelContext_182);
    float4 _S1257 = float4(*(kernelContext_182->lp_0+(_S1256 + 2U))) ;
    float4 _S1258 = float4(*(kernelContext_182->lp_0+(_S1256 + 3U))) ;
    uint _S1259 = 7U * s_19;
    thread Quat_0 q_28;
    thread float3 origin_8;
    structure_pose_0(s_19, _S1259, &q_28, &origin_8, kernelContext_182);
    thread Quat_0 _S1260 = q_28;
    Quat_0 _S1261 = conj_0(&_S1260);
    float3 _S1262 = _S1258.xyz * float3(_S1257.w) ;
    thread Quat_0 _S1263 = _S1261;
    float3 _S1264 = rotate_0(&_S1263, _S1262);
    float3 _S1265 = _S1257.xyz - origin_8;
    thread Quat_0 _S1266 = _S1261;
    float3 _S1267 = rotate_0(&_S1266, _S1265);
    uint _S1268 = 4U * k_49;
    float4 _S1269 = float4(*(kernelContext_182->chunks_0+_S1268)) ;
    float3 _S1270 = float3(0.0f) ;
    uint _S1271 = atomic_load_explicit(kernelContext_182->atom_0+(_S1253 + 1U), memory_order_relaxed);
    float3 f_12;
    float3 t_13;
    if(_S1271 == (k_49 + 1U))
    {
        float3 _S1272 = cross(_S1267 - _S1269.xyz, _S1264);
        f_12 = _S1264;
        t_13 = _S1272;
    }
    else
    {
        f_12 = _S1270;
        t_13 = _S1270;
    }
    float inv_mass_7 = (float4(*(kernelContext_182->bodies_0+_S1259)) ).w;
    bool _S1273;
    if(!supported_1)
    {
        _S1273 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S1273 = false;
    }
    if(_S1273)
    {
        uint _S1274 = 2U * s_19;
        float3 com_2 = (float4(*(kernelContext_182->frames_0+_S1274)) ).xyz;
        Quat_0 p_20 = quat_of_0(float4(*(kernelContext_182->frames_0+(_S1274 + 1U))) );
        float3 _S1275 = (float4(*(kernelContext_182->bodies_0+(_S1259 + 2U))) ).xyz;
        thread Quat_0 _S1276 = p_20;
        Quat_0 _S1277 = conj_0(&_S1276);
        float3 _S1278 = cross(_S1267 - com_2, _S1264);
        thread Quat_0 _S1279 = _S1277;
        float3 _S1280 = rotate_0(&_S1279, _S1278);
        float3 _S1281 = _S1275 * _S1280;
        thread Quat_0 _S1282 = p_20;
        float3 _S1283 = rotate_0(&_S1282, _S1281);
        float4 _S1284 = float4(*(kernelContext_182->chunks_0+(_S1268 + 1U))) ;
        float4 _S1285 = float4(*(kernelContext_182->chunks_0+(_S1268 + 2U))) ;
        float _S1286 = _S1283.x;
        float _S1287 = _S1284.w;
        float _S1288 = _S1283.y;
        float _S1289 = _S1285.x;
        float _S1290 = _S1283.z;
        float _S1291 = _S1285.y;
        float3 t_14 = t_13 - float3(_S1284.x * _S1286 + _S1287 * _S1288 + _S1289 * _S1290, _S1287 * _S1286 + _S1284.y * _S1288 + _S1291 * _S1290, _S1289 * _S1286 + _S1291 * _S1288 + _S1284.z * _S1290);
        f_12 = f_12 - float3(_S1269.w)  * (_S1264 * float3(inv_mass_7)  + cross(_S1283, _S1269.xyz - com_2));
        t_13 = t_14;
    }
    uint _S1292 = st_f_0(k_49, 14U);
    packed_float4 device* _S1293 = kernelContext_182->lp_0+_S1292;
    float _S1294;
    if(fixed_2)
    {
        _S1294 = 1.0f;
    }
    else
    {
        _S1294 = 0.0f;
    }
    *_S1293 = packed_float4(float4(f_12, _S1294)) ;
    *(kernelContext_182->lp_0+(_S1292 + 1U)) = packed_float4(float4(t_13, 0.0f)) ;
    uint c_23 = 0U;
    for(;;)
    {
        if(c_23 < 2U)
        {
        }
        else
        {
            break;
        }
        uint _S1295 = st_x_0(k_49, 14U, c_23);
        packed_float4 _S1296 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_182->lp_0+_S1295) = _S1296;
        *(kernelContext_182->lp_0+(_S1295 + 1U)) = _S1296;
        c_23 = c_23 + 1U;
    }
    uint _S1297 = st_d_0(k_49, 14U);
    packed_float4 _S1298 = packed_float4(float4(0.0f) ) ;
    *(kernelContext_182->lp_0+_S1297) = _S1298;
    *(kernelContext_182->lp_0+(_S1297 + 1U)) = _S1298;
    return;
}

void cs_break_0(uint i_16, uint s_20, KernelContext_0 thread* kernelContext_183)
{
    uint4 _S1299 = uint4(*(kernelContext_183->bonds_0+i_16)) ;
    LpBond_0 _S1300 = lp_bond_0(i_16, kernelContext_183);
    uint _S1301 = _S1299.x;
    uint _S1302 = _S1299.y;
    thread LpBond_0 _S1303 = _S1300;
    thread float3 q_lin_5;
    thread float3 q_ang_5;
    st_force_0(&_S1303, _S1301, _S1302, 14U, 0U, &q_lin_5, &q_ang_5, kernelContext_183);
    if((kernelContext_183->params_0->load_model_0) == 1U)
    {
        thread LpBond_0 _S1304 = _S1300;
        thread float3 b_lin_0;
        thread float3 b_ang_0;
        lp_force_0(&_S1304, _S1301, _S1302, 0U, &b_lin_0, &b_ang_0, kernelContext_183);
        q_lin_5 = q_lin_5 + b_lin_0;
        q_ang_5 = q_ang_5 + b_ang_0;
    }
    thread JointBond_0 _S1305 = _S1300.law_0;
    Measures_0 _S1306 = stress_measures_0(&_S1305, q_lin_5, q_ang_5);
    JointMaterial_0 _S1307 = lp_material_0(_S1300.law_0.ids_3.x, kernelContext_183);
    float _S1308 = _S1300.law_0.geom1_0.w;
    thread JointMaterial_0 _S1309 = _S1307;
    thread JointBond_0 _S1310 = _S1300.law_0;
    thread Measures_0 _S1311 = _S1306;
    float4 _S1312 = failure_indices_0(&_S1309, &_S1310, &_S1311, _S1308);
    if((max(max(_S1312.x, _S1312.y), max(_S1312.z, _S1312.w))) >= 1.0f)
    {
        uint _S1313 = w_alive_0(kernelContext_183);
        *(kernelContext_183->work_0+(_S1313 + i_16)) = 0U;
        uint _S1314 = a_dirty_0(kernelContext_183);
        uint _S1315 = atomic_fetch_or_explicit(kernelContext_183->atom_0+(_S1314 + s_20), 1U, memory_order_relaxed);
        uint _S1316 = a_counters_1(kernelContext_183);
        uint _S1317 = atomic_fetch_add_explicit(kernelContext_183->atom_0+_S1316, 1U, memory_order_relaxed);
        uint _S1318 = atomic_fetch_add_explicit(kernelContext_183->atom_0+(_S1316 + 1U), 1U, memory_order_relaxed);
        uint _S1319 = atomic_fetch_add_explicit(kernelContext_183->atom_0+(_S1316 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_solve(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_73 [[buffer(0)]], uint device* work_73 [[buffer(4)]], packed_float4 device* frames_73 [[buffer(6)]], packed_float4 device* bodies_73 [[buffer(7)]], packed_float4 device* chunks_73 [[buffer(1)]], uint device* sort_73 [[buffer(11)]], atomic<uint> device* atom_73 [[buffer(5)]], Record_natural_0 device* manifolds_73 [[buffer(10)]], packed_float4 device* loads_73 [[buffer(12)]], packed_uint4 device* bonds_73 [[buffer(2)]], uint device* args_73 [[buffer(13)]], uint device* info_73 [[buffer(15)]], ScanParams_0 constant* scan_73 [[buffer(14)]], packed_float4 device* parts_73 [[buffer(3)]], packed_float4 device* colliders_73 [[buffer(8)]], packed_uint4 device* collider_info_73 [[buffer(9)]], packed_float4 device* view_chunks_73 [[buffer(17)]], ViewParams_0 constant* view_73 [[buffer(16)]], packed_float4 device* objects_73 [[buffer(18)]], uint device* shot_bodies_73 [[buffer(20)]], float device* lines_73 [[buffer(19)]], packed_float4 device* geometry_73 [[buffer(21)]], packed_float4 device* shots_73 [[buffer(22)]], packed_float4 device* lp_73 [[buffer(24)]], packed_float4 device* ext_75 [[buffer(23)]], packed_float4 device* lp_sides_73 [[buffer(28)]], uint device* engine_counters_73 [[buffer(26)]], packed_float4 device* engine_contacts_73 [[buffer(25)]], packed_float4 device* cs_anchors_73 [[buffer(27)]])
{
    bool _S1320;
    uint j3_0;
    thread KernelContext_0 kernelContext_184;
    (&kernelContext_184)->params_0 = params_73;
    (&kernelContext_184)->work_0 = work_73;
    (&kernelContext_184)->frames_0 = frames_73;
    (&kernelContext_184)->bodies_0 = bodies_73;
    (&kernelContext_184)->chunks_0 = chunks_73;
    (&kernelContext_184)->sort_0 = sort_73;
    (&kernelContext_184)->atom_0 = atom_73;
    (&kernelContext_184)->manifolds_0 = manifolds_73;
    (&kernelContext_184)->loads_0 = loads_73;
    (&kernelContext_184)->bonds_0 = bonds_73;
    (&kernelContext_184)->args_0 = args_73;
    (&kernelContext_184)->info_0 = info_73;
    (&kernelContext_184)->scan_0 = scan_73;
    (&kernelContext_184)->parts_0 = parts_73;
    (&kernelContext_184)->colliders_0 = colliders_73;
    (&kernelContext_184)->collider_info_0 = collider_info_73;
    (&kernelContext_184)->view_chunks_0 = view_chunks_73;
    (&kernelContext_184)->view_0 = view_73;
    (&kernelContext_184)->objects_0 = objects_73;
    (&kernelContext_184)->shot_bodies_0 = shot_bodies_73;
    (&kernelContext_184)->lines_0 = lines_73;
    (&kernelContext_184)->geometry_0 = geometry_73;
    (&kernelContext_184)->shots_0 = shots_73;
    (&kernelContext_184)->lp_0 = lp_73;
    (&kernelContext_184)->ext_0 = ext_75;
    (&kernelContext_184)->lp_sides_0 = lp_sides_73;
    (&kernelContext_184)->engine_counters_0 = engine_counters_73;
    (&kernelContext_184)->engine_contacts_0 = engine_contacts_73;
    (&kernelContext_184)->cs_anchors_0 = cs_anchors_73;
    threadgroup array<uint, int(1024)> g_scan_73;
    (&kernelContext_184)->g_scan_0 = &g_scan_73;
    threadgroup array<float, int(64)> g_il_best_73;
    (&kernelContext_184)->g_il_best_0 = &g_il_best_73;
    uint _S1321 = w_cs_bodies_0(&kernelContext_184);
    uint device* _S1322 = (&kernelContext_184)->work_0+(_S1321 + group_3.x);
    uint s_21 = *_S1322;
    uint t_15 = thread_6.x;
    uint _S1323 = a_cs_0(*_S1322, &kernelContext_184);
    uint _S1324 = atomic_load_explicit((&kernelContext_184)->atom_0+(_S1323 + 6U), memory_order_relaxed);
    uint _S1325 = atomic_load_explicit((&kernelContext_184)->atom_0+(_S1323 + 5U), memory_order_relaxed);
    uint j_10 = t_15;
    for(;;)
    {
        if(j_10 < _S1325)
        {
        }
        else
        {
            break;
        }
        uint _S1326 = w_cs_list_0(&kernelContext_184);
        cs_load_0(*((&kernelContext_184)->work_0+(_S1326 + _S1324 + j_10)), s_21, &kernelContext_184);
        j_10 = j_10 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint j2_0 = t_15;
    for(;;)
    {
        if(j2_0 < _S1325)
        {
        }
        else
        {
            break;
        }
        uint _S1327 = w_cs_list_0(&kernelContext_184);
        uint device* _S1328 = (&kernelContext_184)->work_0+(_S1327 + _S1324 + j2_0);
        uint k_50 = *_S1328;
        uint _S1329 = st_f_0(*_S1328, 14U);
        if(((float4(*((&kernelContext_184)->lp_0+_S1329)) ).w) == 0.0f)
        {
            bool _S1330 = lp_factor_0(k_50, &kernelContext_184);
            _S1320 = !_S1330;
        }
        else
        {
            _S1320 = false;
        }
        if(_S1320)
        {
            (*((&kernelContext_184)->lp_0+_S1329))[int(3)] = 1.0f;
            uint _S1331 = a_counters_1(&kernelContext_184);
            uint _S1332 = atomic_fetch_or_explicit((&kernelContext_184)->atom_0+(_S1331 + 3U), 32U, memory_order_relaxed);
        }
        j2_0 = j2_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint sweep_0 = 0U;
    for(;;)
    {
        if(sweep_0 < ((&kernelContext_184)->params_0->cs_sweeps_0))
        {
        }
        else
        {
            break;
        }
        j3_0 = t_15;
        for(;;)
        {
            if(j3_0 < _S1325)
            {
            }
            else
            {
                break;
            }
            uint _S1333 = w_cs_list_0(&kernelContext_184);
            st_sweep_0(*((&kernelContext_184)->work_0+(_S1333 + _S1324 + j3_0)), 14U, sweep_0 % 2U, (&kernelContext_184)->params_0->cs_c1_0, (&kernelContext_184)->params_0->cs_c2_0, &kernelContext_184);
            j3_0 = j3_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        sweep_0 = sweep_0 + 1U;
    }
    j3_0 = t_15;
    for(;;)
    {
        if(j3_0 < _S1325)
        {
        }
        else
        {
            break;
        }
        uint _S1334 = w_cs_list_0(&kernelContext_184);
        uint device* _S1335 = (&kernelContext_184)->work_0+(_S1334 + _S1324 + j3_0);
        uint k_51 = *_S1335;
        float4 _S1336 = float4(*((&kernelContext_184)->ext_0+((&kernelContext_184)->params_0->ext_chunks_0 + 4U * *_S1335))) ;
        uint _S1337 = (as_type<uint>((_S1336.x)));
        uint _S1338 = (as_type<uint>((_S1336.y)));
        uint e_6 = 0U;
        for(;;)
        {
            if(e_6 < _S1338)
            {
            }
            else
            {
                break;
            }
            uint _S1339 = adj_bond_0(_S1337 + e_6, &kernelContext_184);
            if(((uint4(*((&kernelContext_184)->bonds_0+_S1339)) ).x) == k_51)
            {
                uint _S1340 = w_alive_0(&kernelContext_184);
                _S1320 = (*((&kernelContext_184)->work_0+(_S1340 + _S1339))) != 0U;
            }
            else
            {
                _S1320 = false;
            }
            if(_S1320)
            {
                cs_break_0(_S1339, s_21, &kernelContext_184);
            }
            e_6 = e_6 + 1U;
        }
        j3_0 = j3_0 + 256U;
    }
    return;
}

