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
    uint ext_sides_0;
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
    uint particle_ccd_0;
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
    packed_float4 device* lp_0;
    packed_float4 device* ext_0;
    packed_float4 device* engine_contacts_0;
    CrushParams_0 constant* crush_0;
    packed_float4 device* crush_rw_0;
    packed_float4 device* crush_tab_0;
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
    return _S6 + 20U;
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

uint w_alive_0(KernelContext_0 thread* kernelContext_30)
{
    return kernelContext_30->params_0->chunk_count_0;
}

uint w_label_0(KernelContext_0 thread* kernelContext_31)
{
    uint _S86 = w_alive_0(kernelContext_31);
    return _S86 + kernelContext_31->params_0->bond_count_0;
}

uint w_slot_0(KernelContext_0 thread* kernelContext_32)
{
    uint _S87 = w_label_0(kernelContext_32);
    return _S87 + kernelContext_32->params_0->chunk_count_0;
}

uint w_new_0(KernelContext_0 thread* kernelContext_33)
{
    uint _S88 = w_slot_0(kernelContext_33);
    return _S88 + kernelContext_33->params_0->chunk_count_0;
}

uint w_rank_0(KernelContext_0 thread* kernelContext_34)
{
    uint _S89 = w_new_0(kernelContext_34);
    return _S89 + kernelContext_34->params_0->chunk_count_0;
}

uint w_free_0(KernelContext_0 thread* kernelContext_35)
{
    uint _S90 = w_rank_0(kernelContext_35);
    return _S90 + kernelContext_35->params_0->chunk_count_0;
}

uint w_free_rank_0(KernelContext_0 thread* kernelContext_36)
{
    uint _S91 = w_free_0(kernelContext_36);
    return _S91 + kernelContext_36->params_0->body_count_0;
}

uint w_free_list_0(KernelContext_0 thread* kernelContext_37)
{
    uint _S92 = w_free_rank_0(kernelContext_37);
    return _S92 + kernelContext_37->params_0->body_count_0;
}

uint w_pulses_0(KernelContext_0 thread* kernelContext_38)
{
    uint _S93 = w_free_list_0(kernelContext_38);
    return _S93 + kernelContext_38->params_0->body_count_0 + 2050U;
}

uint w_marks_0(KernelContext_0 thread* kernelContext_39)
{
    uint _S94 = w_pulses_0(kernelContext_39);
    return _S94 + 4U * kernelContext_39->params_0->chunk_count_0;
}

uint w_carry_0(KernelContext_0 thread* kernelContext_40)
{
    uint _S95 = w_marks_0(kernelContext_40);
    return _S95 + kernelContext_40->params_0->chunk_count_0;
}

uint w_il_tube_0(KernelContext_0 thread* kernelContext_41)
{
    uint _S96 = w_carry_0(kernelContext_41);
    return _S96 + 2U * kernelContext_41->params_0->chunk_count_0 + kernelContext_41->params_0->body_count_0;
}

uint w_host_breaks_0(KernelContext_0 thread* kernelContext_42)
{
    uint _S97 = w_il_tube_0(kernelContext_42);
    return _S97 + 2U * kernelContext_42->params_0->chunk_count_0;
}

uint w_cs_rows_0(KernelContext_0 thread* kernelContext_43)
{
    uint _S98 = w_host_breaks_0(kernelContext_43);
    return _S98 + 1U + kernelContext_43->params_0->bond_count_0;
}

uint cs_rows_0(KernelContext_0 thread* kernelContext_44)
{
    uint _S99 = w_cs_rows_0(kernelContext_44);
    return min(*(kernelContext_44->work_0+_S99), kernelContext_44->params_0->cs_row_cap_0);
}

uint cs_row_0(uint r_3, KernelContext_0 thread* kernelContext_45)
{
    return 8U * (kernelContext_45->params_0->cs_row_offset_0 + r_3);
}

float cs_angular_0(uint b_19, float3 r_4, float3 u_0, KernelContext_0 thread* kernelContext_46)
{
    uint at_0 = 7U * b_19;
    thread Quat_0 _S100 = quat_of_0(float4(*(kernelContext_46->bodies_0+(at_0 + 1U))) );
    Quat_0 _S101 = conj_0(&_S100);
    float3 _S102 = cross(r_4, u_0);
    thread Quat_0 _S103 = _S101;
    float3 _S104 = rotate_0(&_S103, _S102);
    return dot(_S104 * (float4(*(kernelContext_46->bodies_0+(at_0 + 2U))) ).xyz, _S104);
}

bool cs_side_0(uint r_5, uint side_0, uint thread* k_5, uint thread* s_2, uint thread* ob_0, uint thread* co_0, float3 thread* p_1, float3 thread* u_1, float thread* v_2, float thread* m_eff_1, float thread* t_in_0, KernelContext_0 thread* kernelContext_47)
{
    *t_in_0 = 0.0f;
    *k_5 = 4294967295U;
    *s_2 = 4294967295U;
    *ob_0 = 4294967295U;
    *co_0 = 4294967295U;
    float3 _S105 = float3(0.0f) ;
    *p_1 = _S105;
    *u_1 = _S105;
    *v_2 = 0.0f;
    *m_eff_1 = 0.0f;
    uint _S106 = cs_row_0(r_5, kernelContext_47);
    uint4 ids_1 = (as_type<uint4>((float4(*(kernelContext_47->engine_contacts_0+_S106)) )));
    if((ids_1.w) == 0U)
    {
        return false;
    }
    bool _S107 = side_0 == 0U;
    uint cm_0;
    if(_S107)
    {
        cm_0 = ids_1.x;
    }
    else
    {
        cm_0 = ids_1.y;
    }
    uint _S108;
    if(_S107)
    {
        _S108 = ids_1.y;
    }
    else
    {
        _S108 = ids_1.x;
    }
    *co_0 = _S108;
    uint _S109 = (uint4(*(kernelContext_47->collider_info_0+2U * cm_0)) ).x;
    *k_5 = _S109;
    bool _S110;
    if(_S109 == 4294967295U)
    {
        _S110 = true;
    }
    else
    {
        _S110 = (*k_5) >= (kernelContext_47->params_0->chunk_count_0);
    }
    if(_S110)
    {
        _S110 = true;
    }
    else
    {
        _S110 = ((uint4(*(kernelContext_47->collider_info_0+2U * *co_0)) ).z) != 2U;
    }
    if(_S110)
    {
        return false;
    }
    uint _S111 = body_of_0(*k_5, kernelContext_47);
    *s_2 = _S111;
    uint _S112 = (as_type<uint>(((float4(*(kernelContext_47->colliders_0+4U * *co_0)) ).x)));
    *ob_0 = _S112;
    if(_S112 == (*s_2))
    {
        _S110 = true;
    }
    else
    {
        _S110 = (*ob_0) >= (kernelContext_47->params_0->max_bodies_0);
    }
    if(_S110)
    {
        return false;
    }
    uint launch_1 = (uint4(*(kernelContext_47->collider_info_0+(2U * *co_0 + 1U))) ).w;
    if(launch_1 != 0U)
    {
        uint _S113 = w_marks_0(kernelContext_47);
        _S110 = (*(kernelContext_47->work_0+(_S113 + *k_5))) == launch_1;
    }
    else
    {
        _S110 = false;
    }
    if(_S110)
    {
        return false;
    }
    uint bs_0 = 7U * *s_2;
    uint bo_0 = 7U * *ob_0;
    float inv_o_0 = (float4(*(kernelContext_47->bodies_0+bo_0)) ).w;
    if(!(inv_o_0 > 0.0f))
    {
        return false;
    }
    float3 n_2 = (float4(*(kernelContext_47->engine_contacts_0+(_S106 + 1U))) ).xyz;
    float3 _S114;
    if(_S107)
    {
        _S114 = - n_2;
    }
    else
    {
        _S114 = n_2;
    }
    *u_1 = _S114;
    Quat_0 _S115 = quat_of_0(float4(*(kernelContext_47->bodies_0+(bs_0 + 1U))) );
    if(_S107)
    {
        _S114 = (float4(*(kernelContext_47->engine_contacts_0+(_S106 + 2U))) ).xyz;
    }
    else
    {
        _S114 = (float4(*(kernelContext_47->engine_contacts_0+(_S106 + 3U))) ).xyz;
    }
    thread Quat_0 _S116 = _S115;
    float3 _S117 = rotate_0(&_S116, _S114);
    Quat_0 _S118 = quat_of_0(float4(*(kernelContext_47->bodies_0+(bo_0 + 1U))) );
    if(_S107)
    {
        _S114 = (float4(*(kernelContext_47->engine_contacts_0+(_S106 + 3U))) ).xyz;
    }
    else
    {
        _S114 = (float4(*(kernelContext_47->engine_contacts_0+(_S106 + 2U))) ).xyz;
    }
    thread Quat_0 _S119 = _S118;
    float3 _S120 = rotate_0(&_S119, _S114);
    float3 _S121 = (float4(*(kernelContext_47->bodies_0+bs_0)) ).xyz + _S117;
    *p_1 = _S121;
    float gap_0 = dot(_S121 - ((float4(*(kernelContext_47->bodies_0+bo_0)) ).xyz + _S120), *u_1);
    float _S122 = dot((float4(*(kernelContext_47->bodies_0+(bo_0 + 3U))) ).xyz + cross((float4(*(kernelContext_47->bodies_0+(bo_0 + 4U))) ).xyz, _S120) - ((float4(*(kernelContext_47->bodies_0+(bs_0 + 3U))) ).xyz + cross((float4(*(kernelContext_47->bodies_0+(bs_0 + 4U))) ).xyz, _S117)), *u_1);
    *v_2 = _S122;
    if(!(_S122 > (kernelContext_47->params_0->strike_speed_0)))
    {
        return false;
    }
    float _S123 = (float4(*(kernelContext_47->bodies_0+bs_0)) ).w;
    float _S124 = cs_angular_0(*s_2, _S117, *u_1, kernelContext_47);
    float _S125 = _S123 + _S124 + inv_o_0;
    float _S126 = cs_angular_0(*ob_0, _S120, *u_1, kernelContext_47);
    float w_1 = _S125 + _S126;
    if(!(w_1 > 0.0f))
    {
        return false;
    }
    *m_eff_1 = 1.0f / w_1;
    *t_in_0 = clamp(gap_0 / *v_2, 0.0f, kernelContext_47->params_0->dt_0);
    return true;
}

uint3 cs_identity_0(uint r_6, uint side_1, KernelContext_0 thread* kernelContext_48)
{
    uint _S127 = cs_row_0(r_6, kernelContext_48);
    uint4 ids_2 = (as_type<uint4>((float4(*(kernelContext_48->engine_contacts_0+_S127)) )));
    uint3 _S128;
    if(side_1 == 0U)
    {
        _S128 = uint3(ids_2.x, ids_2.y, 2U * ids_2.z);
    }
    else
    {
        _S128 = uint3(ids_2.y, ids_2.x, 2U * ids_2.z + 1U);
    }
    return _S128;
}

void cs_pick_0(uint r_7, KernelContext_0 thread* kernelContext_49)
{
    uint _S129 = cs_rows_0(kernelContext_49);
    if(r_7 >= _S129)
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
        bool _S130 = cs_side_0(r_7, side_2, &k_6, &s_3, &ob_1, &co_1, &p_2, &u_2, &v_3, &m_eff_2, &t_in_1, kernelContext_49);
        bool _S131;
        if(!_S130)
        {
            _S131 = true;
        }
        else
        {
            uint _S132 = (as_type<uint>((m_eff_2 * v_3)));
            uint _S133 = a_cs_0(s_3, kernelContext_49);
            uint _S134 = atomic_load_explicit(kernelContext_49->atom_0+_S133, memory_order_relaxed);
            _S131 = _S132 != _S134;
        }
        if(_S131)
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint3 _S135 = cs_identity_0(r_7, side_2, kernelContext_49);
        uint _S136 = a_cs_0(s_3, kernelContext_49);
        uint _S137 = atomic_load_explicit(kernelContext_49->atom_0+(_S136 + 2U), memory_order_relaxed);
        if((~_S137) != (_S135.x))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint _S138 = atomic_load_explicit(kernelContext_49->atom_0+(_S136 + 9U), memory_order_relaxed);
        if((~_S138) != (_S135.y))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint _S139 = atomic_load_explicit(kernelContext_49->atom_0+(_S136 + 10U), memory_order_relaxed);
        if((~_S139) != (_S135.z))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        atomic_store_explicit(kernelContext_49->atom_0+(_S136 + 11U), 2U * r_7 + side_2 + 1U, memory_order_relaxed);
        side_2 = side_2 + 1U;
    }
    return;
}

void cs_pick_1(uint r_8, KernelContext_0 thread* kernelContext_50)
{
    uint _S140 = cs_rows_0(kernelContext_50);
    if(r_8 >= _S140)
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
        bool _S141 = cs_side_0(r_8, side_3, &k_7, &s_4, &ob_2, &co_2, &p_3, &u_3, &v_4, &m_eff_3, &t_in_2, kernelContext_50);
        bool _S142;
        if(!_S141)
        {
            _S142 = true;
        }
        else
        {
            uint _S143 = (as_type<uint>((m_eff_3 * v_4)));
            uint _S144 = a_cs_0(s_4, kernelContext_50);
            uint _S145 = atomic_load_explicit(kernelContext_50->atom_0+_S144, memory_order_relaxed);
            _S142 = _S143 != _S145;
        }
        if(_S142)
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint3 _S146 = cs_identity_0(r_8, side_3, kernelContext_50);
        uint _S147 = a_cs_0(s_4, kernelContext_50);
        uint _S148 = atomic_load_explicit(kernelContext_50->atom_0+(_S147 + 2U), memory_order_relaxed);
        if((~_S148) != (_S146.x))
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint _S149 = atomic_load_explicit(kernelContext_50->atom_0+(_S147 + 9U), memory_order_relaxed);
        if((~_S149) != (_S146.y))
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint _S150 = atomic_fetch_max_explicit(kernelContext_50->atom_0+(_S147 + 10U), ~_S146.z, memory_order_relaxed);
        side_3 = side_3 + 1U;
    }
    return;
}

void cs_pick_2(uint r_9, KernelContext_0 thread* kernelContext_51)
{
    uint _S151 = cs_rows_0(kernelContext_51);
    if(r_9 >= _S151)
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
        bool _S152 = cs_side_0(r_9, side_4, &k_8, &s_5, &ob_3, &co_3, &p_4, &u_4, &v_5, &m_eff_4, &t_in_3, kernelContext_51);
        bool _S153;
        if(!_S152)
        {
            _S153 = true;
        }
        else
        {
            uint _S154 = (as_type<uint>((m_eff_4 * v_5)));
            uint _S155 = a_cs_0(s_5, kernelContext_51);
            uint _S156 = atomic_load_explicit(kernelContext_51->atom_0+_S155, memory_order_relaxed);
            _S153 = _S154 != _S156;
        }
        if(_S153)
        {
            side_4 = side_4 + 1U;
            continue;
        }
        uint3 _S157 = cs_identity_0(r_9, side_4, kernelContext_51);
        uint _S158 = a_cs_0(s_5, kernelContext_51);
        uint _S159 = atomic_load_explicit(kernelContext_51->atom_0+(_S158 + 2U), memory_order_relaxed);
        if((~_S159) != (_S157.x))
        {
            side_4 = side_4 + 1U;
            continue;
        }
        uint _S160 = atomic_fetch_max_explicit(kernelContext_51->atom_0+(_S158 + 9U), ~_S157.y, memory_order_relaxed);
        side_4 = side_4 + 1U;
    }
    return;
}

void cs_pick_3(uint r_10, KernelContext_0 thread* kernelContext_52)
{
    uint _S161 = cs_rows_0(kernelContext_52);
    if(r_10 >= _S161)
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
        bool _S162 = cs_side_0(r_10, side_5, &k_9, &s_6, &ob_4, &co_4, &p_5, &u_5, &v_6, &m_eff_5, &t_in_4, kernelContext_52);
        bool _S163;
        if(!_S162)
        {
            _S163 = true;
        }
        else
        {
            uint _S164 = (as_type<uint>((m_eff_5 * v_6)));
            uint _S165 = a_cs_0(s_6, kernelContext_52);
            uint _S166 = atomic_load_explicit(kernelContext_52->atom_0+_S165, memory_order_relaxed);
            _S163 = _S164 != _S166;
        }
        if(_S163)
        {
            side_5 = side_5 + 1U;
            continue;
        }
        uint3 _S167 = cs_identity_0(r_10, side_5, kernelContext_52);
        uint _S168 = a_cs_0(s_6, kernelContext_52);
        uint _S169 = atomic_fetch_max_explicit(kernelContext_52->atom_0+(_S168 + 2U), ~_S167.x, memory_order_relaxed);
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

bool il_crossed_0(uint k_10, float3 a_1, float3 d_5, float length_0, float r_11, KernelContext_0 thread* kernelContext_53)
{
    thread Quat_0 q_8;
    float3 _S170 = chunk_box_world_0(k_10, &q_8, kernelContext_53);
    uint _S171 = 3U * k_10;
    Quat_0 _S172 = quat_of_0(float4(*(kernelContext_53->geometry_0+(_S171 + 2U))) );
    thread Quat_0 _S173 = q_8;
    thread Quat_0 _S174 = _S172;
    Quat_0 _S175 = quat_mul_0(&_S173, &_S174);
    thread Quat_0 _S176 = _S175;
    Quat_0 _S177 = normalized_0(&_S176);
    float3 half_0 = (float4(*(kernelContext_53->geometry_0+(_S171 + 1U))) ).xyz;
    thread Quat_0 _S178 = _S177;
    Quat_0 _S179 = conj_0(&_S178);
    float3 _S180 = a_1 - _S170;
    thread Quat_0 _S181 = _S179;
    float3 _S182 = rotate_0(&_S181, _S180);
    thread Quat_0 _S183 = _S179;
    float3 _S184 = rotate_0(&_S183, d_5);
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
        float _S185 = hi_0 - lo_0;
        float m1_0 = lo_0 + _S185 / 3.0f;
        float m2_0 = hi_0 - _S185 / 3.0f;
        float3 _S186 = float3(0.0f) ;
        float3 e1_0 = max(abs(_S182 + _S184 * float3(m1_0) ) - half_0, _S186);
        float3 e2_0 = max(abs(_S182 + _S184 * float3(m2_0) ) - half_0, _S186);
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
    float3 e_1 = max(abs(_S182 + _S184 * float3((0.5f * (lo_0 + hi_0))) ) - half_0, float3(0.0f) );
    return (dot(e_1, e_1)) < (0.99980002641677856f * r_11 * r_11);
}

IlPlace_0 il_place_0(uint k_11, uint entry_1, float3 contact_4, float3 d_6, float r_12, float h_0, KernelContext_0 thread* kernelContext_54)
{
    uint _S187 = body_of_0(k_11, kernelContext_54);
    thread Quat_0 q_9;
    thread float3 origin_2;
    structure_pose_0(_S187, 7U * _S187, &q_9, &origin_2, kernelContext_54);
    float3 _S188 = (float4(*(kernelContext_54->chunks_0+4U * k_11)) ).xyz;
    thread Quat_0 _S189 = q_9;
    float3 _S190 = rotate_0(&_S189, _S188);
    float3 rel_1 = origin_2 + _S190 - contact_4;
    thread IlPlace_0 p_6;
    float _S191 = dot(rel_1, d_6);
    (&p_6)->along_0 = _S191;
    (&p_6)->radial_0 = length(rel_1 - d_6 * float3(_S191) );
    (&p_6)->entry_0 = k_11 == entry_1;
    bool _S192 = il_crossed_0(k_11, contact_4 - d_6 * float3(r_12) , d_6, h_0 + 2.0f * r_12, r_12, kernelContext_54);
    (&p_6)->crossed_0 = _S192;
    return p_6;
}

bool il_plug_0(uint outcome_0, float r_13, float h_1, const IlPlace_0 thread* p_7)
{
    bool _S193;
    if((outcome_0 & 8U) != 0U)
    {
        if(p_7->entry_0)
        {
            _S193 = true;
        }
        else
        {
            _S193 = p_7->crossed_0;
        }
        if(_S193)
        {
            _S193 = true;
        }
        else
        {
            float _S194 = p_7->along_0;
            if((p_7->along_0) >= 0.0f)
            {
                _S193 = _S194 <= h_1;
            }
            else
            {
                _S193 = false;
            }
            if(_S193)
            {
                _S193 = (p_7->radial_0) <= (r_13 + _S194);
            }
            else
            {
                _S193 = false;
            }
        }
    }
    else
    {
        _S193 = false;
    }
    return _S193;
}

bool il_scab_0(uint outcome_1, float r_14, float h_2, float scab_0, const IlPlace_0 thread* p_8)
{
    bool _S195;
    if((outcome_1 & 4U) != 0U)
    {
        _S195 = (p_8->along_0) >= (h_2 - scab_0);
    }
    else
    {
        _S195 = false;
    }
    if(_S195)
    {
        _S195 = (p_8->along_0) <= h_2;
    }
    else
    {
        _S195 = false;
    }
    if(_S195)
    {
        _S195 = (p_8->radial_0) <= (r_14 + h_2);
    }
    else
    {
        _S195 = false;
    }
    return _S195;
}

bool il_crater_0(uint outcome_2, float r_15, float depth_0, float a_2, const IlPlace_0 thread* p_9)
{
    bool _S196;
    if((outcome_2 & 2U) != 0U)
    {
        _S196 = (p_9->along_0) >= 0.0f;
    }
    else
    {
        _S196 = false;
    }
    if(_S196)
    {
        _S196 = (p_9->along_0) <= depth_0;
    }
    else
    {
        _S196 = false;
    }
    if(_S196)
    {
        _S196 = (p_9->radial_0) <= (max(r_15, a_2));
    }
    else
    {
        _S196 = false;
    }
    return _S196;
}

uint a_counters_1(KernelContext_0 thread* kernelContext_55)
{
    uint _S197 = a_best_root_0(kernelContext_55);
    return _S197 + kernelContext_55->params_0->body_count_0;
}

void il_breaks_0(uint i_0, KernelContext_0 thread* kernelContext_56)
{
    bool cut_0;
    if(i_0 >= (kernelContext_56->params_0->bond_count_0))
    {
        cut_0 = true;
    }
    else
    {
        uint _S198 = w_alive_0(kernelContext_56);
        cut_0 = (*(kernelContext_56->work_0+(_S198 + i_0))) == 0U;
    }
    if(cut_0)
    {
        return;
    }
    uint4 _S199 = uint4(*(kernelContext_56->bonds_0+i_0)) ;
    uint _S200 = _S199.x;
    uint _S201 = body_of_0(_S200, kernelContext_56);
    bool _S202 = il_struck_0(_S201, kernelContext_56);
    if(!_S202)
    {
        return;
    }
    uint _S203 = lp_body_0(_S201, kernelContext_56);
    float4 _S204 = float4(*(kernelContext_56->lp_0+_S203)) ;
    float4 _S205 = float4(*(kernelContext_56->lp_0+(_S203 + 1U))) ;
    uint outcome_3 = (as_type<uint>((_S204.w)));
    if((outcome_3 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_5;
    thread float3 d_7;
    thread float speed_4;
    il_path_0(_S201, &contact_5, &d_7, &speed_4, kernelContext_56);
    uint _S206 = il_entry_0(_S201, kernelContext_56);
    IlStriker_0 _S207 = il_striker_0(_S201, kernelContext_56);
    float _S208 = max(_S207.radius_0, 0.00100000004749745f);
    float h_3 = _S204.x;
    IlPlace_0 _S209 = il_place_0(_S200, _S206, contact_5, d_7, _S208, h_3, kernelContext_56);
    IlPlace_0 _S210 = il_place_0(_S199.y, _S206, contact_5, d_7, _S208, h_3, kernelContext_56);
    thread IlPlace_0 _S211 = _S209;
    bool _S212 = il_plug_0(outcome_3, _S208, h_3, &_S211);
    thread IlPlace_0 _S213 = _S210;
    bool _S214 = il_plug_0(outcome_3, _S208, h_3, &_S213);
    if(_S212 != _S214)
    {
        cut_0 = true;
    }
    else
    {
        float _S215 = _S205.y;
        thread IlPlace_0 _S216 = _S209;
        bool _S217 = il_scab_0(outcome_3, _S208, h_3, _S215, &_S216);
        thread IlPlace_0 _S218 = _S210;
        bool _S219 = il_scab_0(outcome_3, _S208, h_3, _S215, &_S218);
        cut_0 = _S217 != _S219;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S220 = _S205.x;
        float _S221 = _S205.z;
        thread IlPlace_0 _S222 = _S209;
        bool _S223 = il_crater_0(outcome_3, _S208, _S220, _S221, &_S222);
        cut_0 = _S223;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S224 = _S205.x;
        float _S225 = _S205.z;
        thread IlPlace_0 _S226 = _S210;
        bool _S227 = il_crater_0(outcome_3, _S208, _S224, _S225, &_S226);
        cut_0 = _S227;
    }
    if(cut_0)
    {
        uint _S228 = w_alive_0(kernelContext_56);
        *(kernelContext_56->work_0+(_S228 + i_0)) = 0U;
        uint _S229 = a_dirty_0(kernelContext_56);
        uint _S230 = atomic_fetch_or_explicit(kernelContext_56->atom_0+(_S229 + _S201), 1U, memory_order_relaxed);
        uint _S231 = a_counters_1(kernelContext_56);
        uint _S232 = atomic_fetch_add_explicit(kernelContext_56->atom_0+_S231, 1U, memory_order_relaxed);
        uint _S233 = atomic_fetch_add_explicit(kernelContext_56->atom_0+(_S231 + 1U), 1U, memory_order_relaxed);
        uint _S234 = atomic_fetch_add_explicit(kernelContext_56->atom_0+(_S231 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

float3 to_rubble_0(float3 v_7)
{
    return float3(v_7.x, v_7.z, - v_7.y);
}

float read64_0(uint at_1, float scale_0, KernelContext_0 thread* kernelContext_57)
{
    uint _S235 = atomic_load_explicit(kernelContext_57->atom_0+(at_1 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S235)));
    uint lo_1 = atomic_load_explicit(kernelContext_57->atom_0+at_1, memory_order_relaxed);
    return (float(hi_1) * 4.294967296e+09f + float(lo_1)) / scale_0;
}

void il_carry_0(uint k_12, KernelContext_0 thread* kernelContext_58)
{
    if(k_12 >= (kernelContext_58->params_0->chunk_count_0))
    {
        return;
    }
    uint _S236 = w_carry_0(kernelContext_58);
    uint _S237 = _S236 + k_12;
    uint device* _S238 = kernelContext_58->work_0+_S237;
    uint c_2 = *_S238;
    if((*_S238) == 0U)
    {
        return;
    }
    *(kernelContext_58->work_0+_S237) = 0U;
    uint b0_0 = c_2 - 1U;
    bool _S239 = il_struck_0(b0_0, kernelContext_58);
    if(!_S239)
    {
        return;
    }
    thread float3 contact_6;
    thread float3 d_8;
    thread float speed_5;
    il_path_0(b0_0, &contact_6, &d_8, &speed_5, kernelContext_58);
    float3 dr_0 = to_rubble_0(d_8);
    IlStriker_0 _S240 = il_striker_0(b0_0, kernelContext_58);
    uint _S241 = a_il_0(b0_0, kernelContext_58);
    float _S242 = read64_0(_S241 + 2U, kernelContext_58->params_0->scale_m_0, kernelContext_58);
    float3 v_wall_0 = (float4(*(kernelContext_58->bodies_0+(7U * b0_0 + 3U))) ).xyz;
    uint _S243 = 7U * _S240.body_0;
    float3 v_shot_0 = (float4(*(kernelContext_58->bodies_0+(_S243 + 3U))) ).xyz;
    float closing_0 = dot(v_shot_0 - v_wall_0, dr_0);
    float v_r_0 = _S240.mass_0 * closing_0 / (_S240.mass_0 + _S242);
    float3 _S244 = v_shot_0 + dr_0 * float3((v_r_0 - closing_0)) ;
    float3 _S245 = v_wall_0 + dr_0 * float3(v_r_0) ;
    float _S246 = il_t_in_0(b0_0, kernelContext_58);
    float3 _S247 = (float4(*(kernelContext_58->bodies_0+_S243)) ).xyz + dr_0 * float3(((speed_5 - v_r_0) * clamp(_S246, 0.0f, kernelContext_58->params_0->dt_0))) ;
    uint _S248 = body_of_0(k_12, kernelContext_58);
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
        uint s_at_0 = 7U * (_S240.body_0 + region_0 * kernelContext_58->params_0->max_bodies_0);
        (*(kernelContext_58->bodies_0+s_at_0)).xyz = _S247;
        (*(kernelContext_58->bodies_0+(s_at_0 + 3U))).xyz = _S244;
        (*(kernelContext_58->bodies_0+(7U * (_S248 + region_0 * kernelContext_58->params_0->max_bodies_0) + 3U))).xyz = _S245;
        region_0 = region_0 + 1U;
    }
    return;
}

void add64_0(uint at_2, float value_0, float scale_1, KernelContext_0 thread* kernelContext_59)
{
    float x_1 = round(value_0 * scale_1);
    float x_2;
    if(!((abs(x_1)) < 1.073741824e+09f))
    {
        uint _S249 = a_counters_1(kernelContext_59);
        uint _S250 = atomic_fetch_or_explicit(kernelContext_59->atom_0+(_S249 + 3U), 2U, memory_order_relaxed);
        x_2 = 0.0f;
    }
    else
    {
        x_2 = x_1;
    }
    int v_8 = int(x_2);
    uint lo_2 = (as_type<uint>((v_8)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_59->atom_0+at_2, lo_2, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_2) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S251;
    if(v_8 < int(0))
    {
        _S251 = 4294967295U;
    }
    else
    {
        _S251 = 0U;
    }
    uint hi_2 = _S251 + carry_0;
    if(hi_2 != 0U)
    {
        uint _S252 = atomic_fetch_add_explicit(kernelContext_59->atom_0+(at_2 + 1U), hi_2, memory_order_relaxed);
    }
    return;
}

void il_mark_0(uint k_13, KernelContext_0 thread* kernelContext_60)
{
    if(k_13 >= (kernelContext_60->params_0->chunk_count_0))
    {
        return;
    }
    uint _S253 = body_of_0(k_13, kernelContext_60);
    bool _S254 = il_struck_0(_S253, kernelContext_60);
    if(!_S254)
    {
        return;
    }
    uint _S255 = lp_body_0(_S253, kernelContext_60);
    float4 _S256 = float4(*(kernelContext_60->lp_0+_S255)) ;
    float4 _S257 = float4(*(kernelContext_60->lp_0+(_S255 + 1U))) ;
    uint outcome_4 = (as_type<uint>((_S256.w)));
    if((outcome_4 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_7;
    thread float3 d_9;
    thread float speed_6;
    il_path_0(_S253, &contact_7, &d_9, &speed_6, kernelContext_60);
    uint _S258 = il_entry_0(_S253, kernelContext_60);
    IlStriker_0 _S259 = il_striker_0(_S253, kernelContext_60);
    float _S260 = max(_S259.radius_0, 0.00100000004749745f);
    float _S261 = _S256.x;
    IlPlace_0 _S262 = il_place_0(k_13, _S258, contact_7, d_9, _S260, _S261, kernelContext_60);
    thread IlPlace_0 _S263 = _S262;
    bool _S264 = il_plug_0(outcome_4, _S260, _S261, &_S263);
    if((_S259.launch_0) != 0U)
    {
        float2 _S265 = il_interval_0(k_13, contact_7 - d_9 * float3(_S259.bound_0) , d_9, _S259.bound_0, kernelContext_60);
        float _S266 = _S265.x;
        float _S267 = _S265.y;
        bool path_0;
        if(_S266 <= _S267)
        {
            path_0 = _S266 <= (_S261 + 2.0f * _S259.bound_0);
        }
        else
        {
            path_0 = false;
        }
        if(path_0)
        {
            path_0 = _S267 >= 0.0f;
        }
        else
        {
            path_0 = false;
        }
        if(_S264)
        {
            path_0 = true;
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            float _S268 = _S257.y;
            thread IlPlace_0 _S269 = _S262;
            bool _S270 = il_scab_0(outcome_4, _S260, _S261, _S268, &_S269);
            path_0 = _S270;
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            float _S271 = _S257.x;
            float _S272 = _S257.z;
            thread IlPlace_0 _S273 = _S262;
            bool _S274 = il_crater_0(outcome_4, _S260, _S271, _S272, &_S273);
            path_0 = _S274;
        }
        if(path_0)
        {
            uint _S275 = w_marks_0(kernelContext_60);
            *(kernelContext_60->work_0+(_S275 + k_13)) = _S259.launch_0;
        }
    }
    if(_S264)
    {
        uint _S276 = w_carry_0(kernelContext_60);
        *(kernelContext_60->work_0+(_S276 + k_13)) = _S253 + 1U;
        uint _S277 = a_il_0(_S253, kernelContext_60);
        add64_0(_S277 + 2U, (float4(*(kernelContext_60->chunks_0+4U * k_13)) ).w, kernelContext_60->params_0->scale_m_0, kernelContext_60);
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
    float4 _S278 = mat_0->dif_0;
    float ref_0 = mat_0->dif_0.x;
    if(r_16 <= ref_0)
    {
        return 1.0f;
    }
    float _S279 = _S278.z;
    float f_0;
    if(r_16 <= _S279)
    {
        float _S280 = (metal::fast::divide((r_16), (ref_0)));
        float _S281 = (metal::fast::pow((_S280), (_S278.y)));
        f_0 = _S281;
    }
    else
    {
        float _S282 = (metal::fast::divide((_S279), (ref_0)));
        float _S283 = (metal::fast::pow((_S282), (_S278.y)));
        float _S284 = (metal::fast::divide((r_16), (_S279)));
        float _S285 = (metal::fast::pow((_S284), (_S278.w)));
        f_0 = _S283 * _S285;
    }
    return clamp(f_0, 1.0f, mat_0->misc_0.x);
}

void il_impulse_0(uint b_20, uint sb_0, float m_shot_0, float3 contact_8, float3 d_10, float j_0, KernelContext_0 thread* kernelContext_61)
{
    float3 jr_0 = to_rubble_0(d_10) * float3(j_0) ;
    uint at_3 = 7U * b_20;
    float inv_mass_1 = (float4(*(kernelContext_61->bodies_0+at_3)) ).w;
    float3 _S286 = jr_0 * float3(inv_mass_1) ;
    float3 _S287 = float3(0.0f) ;
    float3 dw_0;
    if(inv_mass_1 > 0.0f)
    {
        Quat_0 q_10 = quat_of_0(float4(*(kernelContext_61->bodies_0+(at_3 + 1U))) );
        float3 r_17 = to_rubble_0(contact_8) - (float4(*(kernelContext_61->bodies_0+at_3)) ).xyz;
        float3 _S288 = (float4(*(kernelContext_61->bodies_0+(at_3 + 2U))) ).xyz;
        thread Quat_0 _S289 = q_10;
        Quat_0 _S290 = conj_0(&_S289);
        float3 _S291 = cross(r_17, jr_0);
        thread Quat_0 _S292 = _S290;
        float3 _S293 = rotate_0(&_S292, _S291);
        float3 _S294 = _S288 * _S293;
        thread Quat_0 _S295 = q_10;
        float3 _S296 = rotate_0(&_S295, _S294);
        dw_0 = _S296;
    }
    else
    {
        dw_0 = _S287;
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
        uint s_at_1 = 7U * (sb_0 + region_1 * kernelContext_61->params_0->max_bodies_0) + 3U;
        (*(kernelContext_61->bodies_0+s_at_1)).xyz = (float4(*(kernelContext_61->bodies_0+s_at_1)) ).xyz - jr_0 / float3(m_shot_0) ;
        uint w_at_0 = 7U * (b_20 + region_1 * kernelContext_61->params_0->max_bodies_0);
        uint _S297 = w_at_0 + 3U;
        (*(kernelContext_61->bodies_0+_S297)).xyz = (float4(*(kernelContext_61->bodies_0+_S297)) ).xyz + _S286;
        uint _S298 = w_at_0 + 4U;
        (*(kernelContext_61->bodies_0+_S298)).xyz = (float4(*(kernelContext_61->bodies_0+_S298)) ).xyz + dw_0;
        region_1 = region_1 + 1U;
    }
    return;
}

void il_strike_0(uint b_21, KernelContext_0 thread* kernelContext_62)
{
    bool damage_0;
    if(b_21 >= (kernelContext_62->params_0->body_count_0))
    {
        damage_0 = true;
    }
    else
    {
        bool _S299 = il_struck_0(b_21, kernelContext_62);
        damage_0 = !_S299;
    }
    if(damage_0)
    {
        return;
    }
    float _S300 = il_t_in_0(b_21, kernelContext_62);
    uint _S301 = il_entry_0(b_21, kernelContext_62);
    thread float3 contact_9;
    thread float3 d_11;
    thread float speed_7;
    il_path_0(b_21, &contact_9, &d_11, &speed_7, kernelContext_62);
    IlStriker_0 _S302 = il_striker_0(b_21, kernelContext_62);
    float _S303 = max(_S302.radius_0, 0.00100000004749745f);
    if((_S302.launch_0) != 0U)
    {
        uint _S304 = w_marks_0(kernelContext_62);
        damage_0 = (*(kernelContext_62->work_0+(_S304 + _S301))) == (_S302.launch_0);
    }
    else
    {
        damage_0 = false;
    }
    if(damage_0)
    {
        uint _S305 = lp_body_0(b_21, kernelContext_62);
        packed_float4 _S306 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_62->lp_0+_S305) = _S306;
        *(kernelContext_62->lp_0+(_S305 + 1U)) = _S306;
        return;
    }
    uint _S307 = a_il_0(b_21, kernelContext_62);
    uint _S308 = atomic_load_explicit(kernelContext_62->atom_0+_S307, memory_order_relaxed);
    float _S309 = max((as_type<float>((_S308))) - _S303, 9.99999997475242708e-07f);
    uint o_2 = kernelContext_62->params_0->ext_chunks_0 + 4U * _S301;
    float4 _S310 = float4(*(kernelContext_62->ext_0+(o_2 + 1U))) ;
    thread JointMaterial_0 dm_0;
    float4 _S311 = float4(0.0f) ;
    (&dm_0)->strength_0 = _S311;
    (&dm_0)->energy_0 = _S311;
    (&dm_0)->dif_0 = float4(*(kernelContext_62->ext_0+(o_2 + 2U))) ;
    float4 _S312 = float4(*(kernelContext_62->ext_0+(o_2 + 3U))) ;
    float _S313 = _S312.x;
    (&dm_0)->misc_0 = float4(_S313, 0.0f, 0.0f, 0.0f);
    (&dm_0)->kind_flags_0 = uint4(0U) ;
    float nu_0 = _S312.y;
    float e_target_0 = _S312.z / (1.0f - nu_0 * nu_0);
    float e_star_0 = 1.0f / (1.0f / _S302.modulus_0 + 1.0f / e_target_0);
    float _S314 = max(speed_7, 0.00100000004749745f);
    float _S315 = sqrt(_S303);
    float h0_0 = pow(15.0f * _S302.m_eff_0 * _S314 * _S314 / (16.0f * e_star_0 * _S315), 0.40000000596046448f);
    float t_c_0 = 2.94000005722045898f * h0_0 / _S314;
    float force_0 = 1.33333337306976318f * e_star_0 * _S315 * pow(h0_0, 1.5f);
    float a_3 = sqrt(_S303 * h0_0);
    float p0_0 = 3.0f * force_0 / (6.28318548202514648f * a_3 * a_3);
    float ring_0 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    float k_14;
    if(_S313 > 0.0f)
    {
        float _S316 = _S314 / _S309;
        thread JointMaterial_0 _S317 = dm_0;
        float _S318 = dif_factor_0(&_S317, _S316);
        k_14 = _S318;
    }
    else
    {
        k_14 = 1.0f;
    }
    float ft_0 = k_14 * _S310.y;
    float fc_0 = k_14 * _S310.z;
    float tau_0 = k_14 * _S310.w;
    if(p0_0 > fc_0)
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ring_0 > ft_0;
    }
    float _S319 = fc_0 * (3.14159274101257324f * _S303 * _S303);
    float _S320 = min(_S319 * _S309, 1.57079637050628662f * (2.0f * _S303 + _S309) * _S309 * _S309 * tau_0);
    float _S321 = 2.0f * _S320;
    float v_bl_0 = sqrt(_S321 / _S302.m_eff_0);
    bool perforate_0;
    if(damage_0)
    {
        perforate_0 = _S314 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    float r2_0 = _S303 + _S309;
    float _S322 = _S310.x;
    float plug_0 = _S322 * 3.14159274101257324f * _S309 / 3.0f * (_S303 * _S303 + _S303 * r2_0 + r2_0 * r2_0);
    float v_res_0;
    if(perforate_0)
    {
        v_res_0 = _S302.mass_0 / (_S302.mass_0 + plug_0) * sqrt(max(_S314 * _S314 - v_bl_0 * v_bl_0, 0.0f));
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
        depth_1 = min(0.5f * _S302.m_eff_0 * _S314 * _S314 / _S319, _S309);
    }
    else
    {
        depth_1 = 0.0f;
    }
    bool crater_0 = depth_1 > a_3;
    float c_3 = sqrt(e_target_0 / _S322);
    float z_t_0 = _S322 * c_3;
    float rho_p_0 = _S302.mass_0 / (4.18879032135009766f * _S303 * _S303 * _S303);
    float z_p_0 = rho_p_0 * sqrt(_S302.modulus_0 / rho_p_0);
    float mean_pressure_0 = force_0 / (3.14159274101257324f * a_3 * a_3);
    float _S323 = min(min(z_t_0 * _S314 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_0);
    float source_radius_0;
    if(mean_pressure_0 > fc_0)
    {
        source_radius_0 = max(a_3, _S303);
    }
    else
    {
        source_radius_0 = a_3;
    }
    float pulse_0 = c_3 * t_c_0;
    float back_0 = _S323 * min(source_radius_0 / _S309, 1.0f) * min(2.0f * _S309 / pulse_0, 1.0f);
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
        scab_1 = 0.5f * min(pulse_0, _S309);
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
    uint _S324;
    if(crater_0)
    {
        _S324 = 2U;
    }
    else
    {
        _S324 = 0U;
    }
    uint _S325 = region_2 | _S324;
    if(spall_0)
    {
        region_2 = 4U;
    }
    else
    {
        region_2 = 0U;
    }
    uint _S326 = _S325 | region_2;
    if(perforate_0)
    {
        region_2 = 8U;
    }
    else
    {
        region_2 = 0U;
    }
    uint outcome_5 = _S326 | region_2;
    uint _S327 = lp_body_0(b_21, kernelContext_62);
    *(kernelContext_62->lp_0+_S327) = packed_float4(float4(_S309, v_bl_0, v_res_0, (as_type<float>((outcome_5))))) ;
    packed_float4 device* _S328 = kernelContext_62->lp_0+(_S327 + 1U);
    if(crater_0)
    {
        k_14 = depth_1;
    }
    else
    {
        k_14 = 0.0f;
    }
    *_S328 = packed_float4(float4(k_14, scab_1, a_3, _S320)) ;
    if(perforate_0)
    {
        uint _S329 = a_counters_1(kernelContext_62);
        uint _S330 = atomic_fetch_add_explicit(kernelContext_62->atom_0+(_S329 + 14U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_0)
        {
            uint _S331 = a_counters_1(kernelContext_62);
            uint _S332 = atomic_fetch_add_explicit(kernelContext_62->atom_0+(_S331 + 13U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_0)
            {
                uint _S333 = a_counters_1(kernelContext_62);
                uint _S334 = atomic_fetch_add_explicit(kernelContext_62->atom_0+(_S333 + 12U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S335 = a_counters_1(kernelContext_62);
                uint _S336 = atomic_fetch_add_explicit(kernelContext_62->atom_0+(_S335 + 11U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_0)
    {
        if((kernelContext_62->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_21, _S302.body_0, _S302.mass_0, contact_9, d_11, _S302.m_eff_0 * (_S314 - sqrt(max(_S314 * _S314 - _S321 / _S302.m_eff_0, 0.0f))), kernelContext_62);
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_21, _S302.body_0, _S302.mass_0, contact_9, d_11, _S302.m_eff_0 * _S314, kernelContext_62);
            float3 _S337 = to_rubble_0(d_11) * float3((_S314 * clamp(_S300, 0.0f, kernelContext_62->params_0->dt_0))) ;
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
                uint _S338 = 7U * (_S302.body_0 + region_2 * kernelContext_62->params_0->max_bodies_0);
                (*(kernelContext_62->bodies_0+_S338)).xyz = (float4(*(kernelContext_62->bodies_0+_S338)) ).xyz + _S337;
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

void il_run_0(uint b_22, uint i_1, KernelContext_0 thread* kernelContext_63)
{
    bool _S339 = il_struck_0(b_22, kernelContext_63);
    uint j_1;
    if(_S339)
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
    IlStriker_0 _S340 = il_striker_0(b_22, kernelContext_63);
    float _S341 = 0.00100000004749745f + 0.01999999955296516f * _S340.radius_0;
    uint _S342 = a_il_0(b_22, kernelContext_63);
    uint _S343 = atomic_load_explicit(kernelContext_63->atom_0+(_S342 + 4U), memory_order_relaxed);
    uint _S344 = atomic_load_explicit(kernelContext_63->atom_0+(_S342 + 1U), memory_order_relaxed);
    uint _S345 = atomic_load_explicit(kernelContext_63->atom_0+_S342, memory_order_relaxed);
    float end_0 = group_uniform_float_0((as_type<float>((_S345))), i_1);
    for(;;)
    {
        float best_0 = end_0;
        j_1 = i_1;
        for(;;)
        {
            if(j_1 < _S344)
            {
            }
            else
            {
                break;
            }
            uint _S346 = w_il_tube_0(kernelContext_63);
            uint at_4 = _S346 + 2U * (_S343 + j_1);
            if((as_type<float>((*(kernelContext_63->work_0+at_4)))) <= (end_0 + _S341))
            {
                best_0 = max(best_0, (as_type<float>((*(kernelContext_63->work_0+(at_4 + 1U))))));
            }
            j_1 = j_1 + 64U;
        }
        (*kernelContext_63->g_il_best_0)[i_1] = best_0;
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
                (*kernelContext_63->g_il_best_0)[i_1] = max((*kernelContext_63->g_il_best_0)[i_1], (*kernelContext_63->g_il_best_0)[i_1 + step_1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_1 = step_1 >> 1U;
        }
        float next_0 = group_uniform_float_0((*kernelContext_63->g_il_best_0)[int(0)], i_1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!(next_0 > end_0))
        {
            break;
        }
        end_0 = next_0;
    }
    if(i_1 == 0U)
    {
        atomic_store_explicit(kernelContext_63->atom_0+_S342, (as_type<uint>((end_0))), memory_order_relaxed);
    }
    return;
}

void il_tube_collect_0(uint k_15, KernelContext_0 thread* kernelContext_64)
{
    thread uint b_23;
    thread float2 t_3;
    bool _S347 = il_tube_0(k_15, &b_23, &t_3, kernelContext_64);
    if(!_S347)
    {
        return;
    }
    uint _S348 = w_il_tube_0(kernelContext_64);
    uint _S349 = a_il_0(b_23, kernelContext_64);
    uint _S350 = atomic_load_explicit(kernelContext_64->atom_0+(_S349 + 4U), memory_order_relaxed);
    uint _S351 = a_il_0(b_23, kernelContext_64);
    uint _S352 = atomic_fetch_add_explicit(kernelContext_64->atom_0+(_S351 + 5U), 1U, memory_order_relaxed);
    uint at_5 = _S348 + 2U * (_S350 + _S352);
    *(kernelContext_64->work_0+at_5) = (as_type<uint>((t_3.x)));
    *(kernelContext_64->work_0+(at_5 + 1U)) = (as_type<uint>((t_3.y)));
    return;
}

uint a_il_cursor_1(KernelContext_0 thread* kernelContext_65)
{
    uint _S353 = a_first_0(0U, kernelContext_65);
    return _S353 + 64U;
}

void il_tube_base_0(uint b_24, KernelContext_0 thread* kernelContext_66)
{
    bool _S354;
    if(b_24 < (kernelContext_66->params_0->body_count_0))
    {
        bool _S355 = il_struck_0(b_24, kernelContext_66);
        _S354 = _S355;
    }
    else
    {
        _S354 = false;
    }
    if(_S354)
    {
        uint _S356 = a_il_0(b_24, kernelContext_66);
        atomic<uint> device* _S357 = kernelContext_66->atom_0+(_S356 + 4U);
        uint _S358 = a_il_cursor_1(kernelContext_66);
        atomic<uint> device* _S359 = kernelContext_66->atom_0+_S358;
        uint _S360 = atomic_load_explicit(kernelContext_66->atom_0+(_S356 + 1U), memory_order_relaxed);
        uint _S361 = atomic_fetch_add_explicit(_S359, _S360, memory_order_relaxed);
        atomic_store_explicit(_S357, _S361, memory_order_relaxed);
    }
    return;
}

void il_tube_count_0(uint k_16, KernelContext_0 thread* kernelContext_67)
{
    thread uint b_25;
    thread float2 t_4;
    bool _S362 = il_tube_0(k_16, &b_25, &t_4, kernelContext_67);
    if(_S362)
    {
        uint _S363 = a_il_0(b_25, kernelContext_67);
        uint _S364 = atomic_fetch_add_explicit(kernelContext_67->atom_0+(_S363 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

void il_begin_0(uint b_26, KernelContext_0 thread* kernelContext_68)
{
    bool _S365;
    if(b_26 >= (kernelContext_68->params_0->body_count_0))
    {
        _S365 = true;
    }
    else
    {
        bool _S366 = il_struck_0(b_26, kernelContext_68);
        _S365 = !_S366;
    }
    if(_S365)
    {
        return;
    }
    thread float3 contact_10;
    thread float3 d_12;
    thread float speed_8;
    il_path_0(b_26, &contact_10, &d_12, &speed_8, kernelContext_68);
    uint _S367 = il_entry_0(b_26, kernelContext_68);
    IlStriker_0 _S368 = il_striker_0(b_26, kernelContext_68);
    float2 _S369 = il_interval_0(_S367, contact_10, d_12, _S368.radius_0, kernelContext_68);
    uint _S370 = a_il_0(b_26, kernelContext_68);
    atomic_store_explicit(kernelContext_68->atom_0+_S370, (as_type<uint>((max(_S369.y, 0.0f)))), memory_order_relaxed);
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
        atomic_store_explicit(kernelContext_68->atom_0+(_S370 + i_2), 0U, memory_order_relaxed);
        i_2 = i_2 + 1U;
    }
    return;
}

void il_breaks_1(uint i_3, KernelContext_0 thread* kernelContext_69)
{
    bool cut_1;
    if(i_3 >= (kernelContext_69->params_0->bond_count_0))
    {
        cut_1 = true;
    }
    else
    {
        uint _S371 = w_alive_0(kernelContext_69);
        cut_1 = (*(kernelContext_69->work_0+(_S371 + i_3))) == 0U;
    }
    if(cut_1)
    {
        return;
    }
    uint4 _S372 = uint4(*(kernelContext_69->bonds_0+i_3)) ;
    uint _S373 = _S372.x;
    uint _S374 = body_of_0(_S373, kernelContext_69);
    bool _S375 = il_struck_1(_S374, kernelContext_69);
    if(!_S375)
    {
        return;
    }
    uint _S376 = lp_body_0(_S374, kernelContext_69);
    float4 _S377 = float4(*(kernelContext_69->lp_0+_S376)) ;
    float4 _S378 = float4(*(kernelContext_69->lp_0+(_S376 + 1U))) ;
    uint outcome_6 = (as_type<uint>((_S377.w)));
    if((outcome_6 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_11;
    thread float3 d_13;
    thread float speed_9;
    il_path_1(_S374, &contact_11, &d_13, &speed_9, kernelContext_69);
    uint _S379 = il_entry_1(_S374, kernelContext_69);
    IlStriker_0 _S380 = il_striker_1(_S374, kernelContext_69);
    float _S381 = max(_S380.radius_0, 0.00100000004749745f);
    float h_4 = _S377.x;
    IlPlace_0 _S382 = il_place_0(_S373, _S379, contact_11, d_13, _S381, h_4, kernelContext_69);
    IlPlace_0 _S383 = il_place_0(_S372.y, _S379, contact_11, d_13, _S381, h_4, kernelContext_69);
    thread IlPlace_0 _S384 = _S382;
    bool _S385 = il_plug_0(outcome_6, _S381, h_4, &_S384);
    thread IlPlace_0 _S386 = _S383;
    bool _S387 = il_plug_0(outcome_6, _S381, h_4, &_S386);
    if(_S385 != _S387)
    {
        cut_1 = true;
    }
    else
    {
        float _S388 = _S378.y;
        thread IlPlace_0 _S389 = _S382;
        bool _S390 = il_scab_0(outcome_6, _S381, h_4, _S388, &_S389);
        thread IlPlace_0 _S391 = _S383;
        bool _S392 = il_scab_0(outcome_6, _S381, h_4, _S388, &_S391);
        cut_1 = _S390 != _S392;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S393 = _S378.x;
        float _S394 = _S378.z;
        thread IlPlace_0 _S395 = _S382;
        bool _S396 = il_crater_0(outcome_6, _S381, _S393, _S394, &_S395);
        cut_1 = _S396;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S397 = _S378.x;
        float _S398 = _S378.z;
        thread IlPlace_0 _S399 = _S383;
        bool _S400 = il_crater_0(outcome_6, _S381, _S397, _S398, &_S399);
        cut_1 = _S400;
    }
    if(cut_1)
    {
        uint _S401 = w_alive_0(kernelContext_69);
        *(kernelContext_69->work_0+(_S401 + i_3)) = 0U;
        uint _S402 = a_dirty_0(kernelContext_69);
        uint _S403 = atomic_fetch_or_explicit(kernelContext_69->atom_0+(_S402 + _S374), 1U, memory_order_relaxed);
        uint _S404 = a_counters_1(kernelContext_69);
        uint _S405 = atomic_fetch_add_explicit(kernelContext_69->atom_0+_S404, 1U, memory_order_relaxed);
        uint _S406 = atomic_fetch_add_explicit(kernelContext_69->atom_0+(_S404 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

void il_carry_1(uint k_17, KernelContext_0 thread* kernelContext_70)
{
    if(k_17 >= (kernelContext_70->params_0->chunk_count_0))
    {
        return;
    }
    uint _S407 = w_carry_0(kernelContext_70);
    uint _S408 = _S407 + k_17;
    uint device* _S409 = kernelContext_70->work_0+_S408;
    uint c_4 = *_S409;
    if((*_S409) == 0U)
    {
        return;
    }
    *(kernelContext_70->work_0+_S408) = 0U;
    uint b0_1 = c_4 - 1U;
    bool _S410 = il_struck_1(b0_1, kernelContext_70);
    if(!_S410)
    {
        return;
    }
    thread float3 contact_12;
    thread float3 d_14;
    thread float speed_10;
    il_path_1(b0_1, &contact_12, &d_14, &speed_10, kernelContext_70);
    float3 dr_1 = to_rubble_0(d_14);
    IlStriker_0 _S411 = il_striker_1(b0_1, kernelContext_70);
    uint _S412 = a_il_0(b0_1, kernelContext_70);
    float _S413 = read64_0(_S412 + 2U, kernelContext_70->params_0->scale_m_0, kernelContext_70);
    float3 v_wall_1 = (float4(*(kernelContext_70->bodies_0+(7U * b0_1 + 3U))) ).xyz;
    uint _S414 = 7U * _S411.body_0;
    float3 v_shot_1 = (float4(*(kernelContext_70->bodies_0+(_S414 + 3U))) ).xyz;
    float closing_1 = dot(v_shot_1 - v_wall_1, dr_1);
    float v_r_1 = _S411.mass_0 * closing_1 / (_S411.mass_0 + _S413);
    float3 _S415 = v_shot_1 + dr_1 * float3((v_r_1 - closing_1)) ;
    float3 _S416 = v_wall_1 + dr_1 * float3(v_r_1) ;
    float _S417 = il_t_in_1(b0_1, kernelContext_70);
    float3 _S418 = (float4(*(kernelContext_70->bodies_0+_S414)) ).xyz + dr_1 * float3(((speed_10 - v_r_1) * clamp(_S417, 0.0f, kernelContext_70->params_0->dt_0))) ;
    uint _S419 = body_of_0(k_17, kernelContext_70);
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
        uint s_at_2 = 7U * (_S411.body_0 + region_3 * kernelContext_70->params_0->max_bodies_0);
        (*(kernelContext_70->bodies_0+s_at_2)).xyz = _S418;
        (*(kernelContext_70->bodies_0+(s_at_2 + 3U))).xyz = _S415;
        (*(kernelContext_70->bodies_0+(7U * (_S419 + region_3 * kernelContext_70->params_0->max_bodies_0) + 3U))).xyz = _S416;
        region_3 = region_3 + 1U;
    }
    return;
}

void il_mark_1(uint k_18, KernelContext_0 thread* kernelContext_71)
{
    if(k_18 >= (kernelContext_71->params_0->chunk_count_0))
    {
        return;
    }
    uint _S420 = body_of_0(k_18, kernelContext_71);
    bool _S421 = il_struck_1(_S420, kernelContext_71);
    if(!_S421)
    {
        return;
    }
    uint _S422 = lp_body_0(_S420, kernelContext_71);
    float4 _S423 = float4(*(kernelContext_71->lp_0+_S422)) ;
    float4 _S424 = float4(*(kernelContext_71->lp_0+(_S422 + 1U))) ;
    uint outcome_7 = (as_type<uint>((_S423.w)));
    if((outcome_7 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_13;
    thread float3 d_15;
    thread float speed_11;
    il_path_1(_S420, &contact_13, &d_15, &speed_11, kernelContext_71);
    uint _S425 = il_entry_1(_S420, kernelContext_71);
    IlStriker_0 _S426 = il_striker_1(_S420, kernelContext_71);
    float _S427 = max(_S426.radius_0, 0.00100000004749745f);
    float _S428 = _S423.x;
    IlPlace_0 _S429 = il_place_0(k_18, _S425, contact_13, d_15, _S427, _S428, kernelContext_71);
    thread IlPlace_0 _S430 = _S429;
    bool _S431 = il_plug_0(outcome_7, _S427, _S428, &_S430);
    if((_S426.launch_0) != 0U)
    {
        float2 _S432 = il_interval_0(k_18, contact_13 - d_15 * float3(_S426.bound_0) , d_15, _S426.bound_0, kernelContext_71);
        float _S433 = _S432.x;
        float _S434 = _S432.y;
        bool path_1;
        if(_S433 <= _S434)
        {
            path_1 = _S433 <= (_S428 + 2.0f * _S426.bound_0);
        }
        else
        {
            path_1 = false;
        }
        if(path_1)
        {
            path_1 = _S434 >= 0.0f;
        }
        else
        {
            path_1 = false;
        }
        if(_S431)
        {
            path_1 = true;
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            float _S435 = _S424.y;
            thread IlPlace_0 _S436 = _S429;
            bool _S437 = il_scab_0(outcome_7, _S427, _S428, _S435, &_S436);
            path_1 = _S437;
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            float _S438 = _S424.x;
            float _S439 = _S424.z;
            thread IlPlace_0 _S440 = _S429;
            bool _S441 = il_crater_0(outcome_7, _S427, _S438, _S439, &_S440);
            path_1 = _S441;
        }
        if(path_1)
        {
            uint _S442 = w_marks_0(kernelContext_71);
            *(kernelContext_71->work_0+(_S442 + k_18)) = _S426.launch_0;
        }
    }
    if(_S431)
    {
        uint _S443 = w_carry_0(kernelContext_71);
        *(kernelContext_71->work_0+(_S443 + k_18)) = _S420 + 1U;
        uint _S444 = a_il_0(_S420, kernelContext_71);
        add64_0(_S444 + 2U, (float4(*(kernelContext_71->chunks_0+4U * k_18)) ).w, kernelContext_71->params_0->scale_m_0, kernelContext_71);
    }
    return;
}

void il_strike_1(uint b_27, KernelContext_0 thread* kernelContext_72)
{
    bool damage_1;
    if(b_27 >= (kernelContext_72->params_0->body_count_0))
    {
        damage_1 = true;
    }
    else
    {
        bool _S445 = il_struck_1(b_27, kernelContext_72);
        damage_1 = !_S445;
    }
    if(damage_1)
    {
        return;
    }
    float _S446 = il_t_in_1(b_27, kernelContext_72);
    uint _S447 = il_entry_1(b_27, kernelContext_72);
    thread float3 contact_14;
    thread float3 d_16;
    thread float speed_12;
    il_path_1(b_27, &contact_14, &d_16, &speed_12, kernelContext_72);
    IlStriker_0 _S448 = il_striker_1(b_27, kernelContext_72);
    float _S449 = max(_S448.radius_0, 0.00100000004749745f);
    if((_S448.launch_0) != 0U)
    {
        uint _S450 = w_marks_0(kernelContext_72);
        damage_1 = (*(kernelContext_72->work_0+(_S450 + _S447))) == (_S448.launch_0);
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        uint _S451 = lp_body_0(b_27, kernelContext_72);
        packed_float4 _S452 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_72->lp_0+_S451) = _S452;
        *(kernelContext_72->lp_0+(_S451 + 1U)) = _S452;
        return;
    }
    uint _S453 = a_il_0(b_27, kernelContext_72);
    uint _S454 = atomic_load_explicit(kernelContext_72->atom_0+_S453, memory_order_relaxed);
    float _S455 = max((as_type<float>((_S454))) - _S449, 9.99999997475242708e-07f);
    if(!(_S446 > 0.0f))
    {
        uint _S456 = lp_body_0(b_27, kernelContext_72);
        *(kernelContext_72->lp_0+_S456) = packed_float4(float4(_S455, 0.0f, 0.0f, 0.0f)) ;
        *(kernelContext_72->lp_0+(_S456 + 1U)) = packed_float4(float4(0.0f) ) ;
        return;
    }
    uint o_3 = kernelContext_72->params_0->ext_chunks_0 + 4U * _S447;
    float4 _S457 = float4(*(kernelContext_72->ext_0+(o_3 + 1U))) ;
    thread JointMaterial_0 dm_1;
    float4 _S458 = float4(0.0f) ;
    (&dm_1)->strength_0 = _S458;
    (&dm_1)->energy_0 = _S458;
    (&dm_1)->dif_0 = float4(*(kernelContext_72->ext_0+(o_3 + 2U))) ;
    float4 _S459 = float4(*(kernelContext_72->ext_0+(o_3 + 3U))) ;
    float _S460 = _S459.x;
    (&dm_1)->misc_0 = float4(_S460, 0.0f, 0.0f, 0.0f);
    (&dm_1)->kind_flags_0 = uint4(0U) ;
    float nu_1 = _S459.y;
    float e_target_1 = _S459.z / (1.0f - nu_1 * nu_1);
    float e_star_1 = 1.0f / (1.0f / _S448.modulus_0 + 1.0f / e_target_1);
    float _S461 = max(speed_12, 0.00100000004749745f);
    float _S462 = sqrt(_S449);
    float h0_1 = pow(15.0f * _S448.m_eff_0 * _S461 * _S461 / (16.0f * e_star_1 * _S462), 0.40000000596046448f);
    float t_c_1 = 2.94000005722045898f * h0_1 / _S461;
    float force_1 = 1.33333337306976318f * e_star_1 * _S462 * pow(h0_1, 1.5f);
    float a_4 = sqrt(_S449 * h0_1);
    float p0_1 = 3.0f * force_1 / (6.28318548202514648f * a_4 * a_4);
    float ring_1 = (1.0f - 2.0f * nu_1) * p0_1 / 3.0f;
    float k_19;
    if(_S460 > 0.0f)
    {
        float _S463 = _S461 / _S455;
        thread JointMaterial_0 _S464 = dm_1;
        float _S465 = dif_factor_0(&_S464, _S463);
        k_19 = _S465;
    }
    else
    {
        k_19 = 1.0f;
    }
    float ft_1 = k_19 * _S457.y;
    float fc_1 = k_19 * _S457.z;
    float tau_1 = k_19 * _S457.w;
    if(p0_1 > fc_1)
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = ring_1 > ft_1;
    }
    float _S466 = fc_1 * (3.14159274101257324f * _S449 * _S449);
    float _S467 = min(_S466 * _S455, 1.57079637050628662f * (2.0f * _S449 + _S455) * _S455 * _S455 * tau_1);
    float _S468 = 2.0f * _S467;
    float v_bl_1 = sqrt(_S468 / _S448.m_eff_0);
    bool perforate_1;
    if(damage_1)
    {
        perforate_1 = _S461 >= v_bl_1;
    }
    else
    {
        perforate_1 = false;
    }
    float r2_1 = _S449 + _S455;
    float _S469 = _S457.x;
    float plug_1 = _S469 * 3.14159274101257324f * _S455 / 3.0f * (_S449 * _S449 + _S449 * r2_1 + r2_1 * r2_1);
    float v_res_1;
    if(perforate_1)
    {
        v_res_1 = _S448.mass_0 / (_S448.mass_0 + plug_1) * sqrt(max(_S461 * _S461 - v_bl_1 * v_bl_1, 0.0f));
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
        depth_2 = min(0.5f * _S448.m_eff_0 * _S461 * _S461 / _S466, _S455);
    }
    else
    {
        depth_2 = 0.0f;
    }
    bool crater_1 = depth_2 > a_4;
    float c_5 = sqrt(e_target_1 / _S469);
    float z_t_1 = _S469 * c_5;
    float rho_p_1 = _S448.mass_0 / (4.18879032135009766f * _S449 * _S449 * _S449);
    float z_p_1 = rho_p_1 * sqrt(_S448.modulus_0 / rho_p_1);
    float mean_pressure_1 = force_1 / (3.14159274101257324f * a_4 * a_4);
    float _S470 = min(min(z_t_1 * _S461 * z_p_1 / (z_p_1 + z_t_1), mean_pressure_1), fc_1);
    float source_radius_1;
    if(mean_pressure_1 > fc_1)
    {
        source_radius_1 = max(a_4, _S449);
    }
    else
    {
        source_radius_1 = a_4;
    }
    float pulse_1 = c_5 * t_c_1;
    float back_1 = _S470 * min(source_radius_1 / _S455, 1.0f) * min(2.0f * _S455 / pulse_1, 1.0f);
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
        scab_2 = 0.5f * min(pulse_1, _S455);
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
    uint _S471;
    if(crater_1)
    {
        _S471 = 2U;
    }
    else
    {
        _S471 = 0U;
    }
    uint _S472 = region_4 | _S471;
    if(spall_1)
    {
        region_4 = 4U;
    }
    else
    {
        region_4 = 0U;
    }
    uint _S473 = _S472 | region_4;
    if(perforate_1)
    {
        region_4 = 8U;
    }
    else
    {
        region_4 = 0U;
    }
    uint outcome_8 = _S473 | region_4;
    uint _S474 = lp_body_0(b_27, kernelContext_72);
    *(kernelContext_72->lp_0+_S474) = packed_float4(float4(_S455, v_bl_1, v_res_1, (as_type<float>((outcome_8))))) ;
    packed_float4 device* _S475 = kernelContext_72->lp_0+(_S474 + 1U);
    if(crater_1)
    {
        k_19 = depth_2;
    }
    else
    {
        k_19 = 0.0f;
    }
    *_S475 = packed_float4(float4(k_19, scab_2, a_4, _S467)) ;
    if(perforate_1)
    {
        uint _S476 = a_counters_1(kernelContext_72);
        uint _S477 = atomic_fetch_add_explicit(kernelContext_72->atom_0+(_S476 + 14U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_1)
        {
            uint _S478 = a_counters_1(kernelContext_72);
            uint _S479 = atomic_fetch_add_explicit(kernelContext_72->atom_0+(_S478 + 13U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_1)
            {
                uint _S480 = a_counters_1(kernelContext_72);
                uint _S481 = atomic_fetch_add_explicit(kernelContext_72->atom_0+(_S480 + 12U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S482 = a_counters_1(kernelContext_72);
                uint _S483 = atomic_fetch_add_explicit(kernelContext_72->atom_0+(_S482 + 11U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_1)
    {
        if((kernelContext_72->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_27, _S448.body_0, _S448.mass_0, contact_14, d_16, _S448.m_eff_0 * (_S461 - sqrt(max(_S461 * _S461 - _S468 / _S448.m_eff_0, 0.0f))), kernelContext_72);
        }
    }
    else
    {
        if(damage_1)
        {
            il_impulse_0(b_27, _S448.body_0, _S448.mass_0, contact_14, d_16, _S448.m_eff_0 * _S461, kernelContext_72);
            float3 _S484 = to_rubble_0(d_16) * float3((_S461 * clamp(_S446, 0.0f, kernelContext_72->params_0->dt_0))) ;
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
                uint _S485 = 7U * (_S448.body_0 + region_4 * kernelContext_72->params_0->max_bodies_0);
                (*(kernelContext_72->bodies_0+_S485)).xyz = (float4(*(kernelContext_72->bodies_0+_S485)) ).xyz + _S484;
                region_4 = region_4 + 1U;
            }
        }
    }
    return;
}

void il_run_1(uint b_28, uint i_4, KernelContext_0 thread* kernelContext_73)
{
    bool _S486 = il_struck_1(b_28, kernelContext_73);
    uint j_2;
    if(_S486)
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
    IlStriker_0 _S487 = il_striker_1(b_28, kernelContext_73);
    float _S488 = 0.00100000004749745f + 0.01999999955296516f * _S487.radius_0;
    uint _S489 = a_il_0(b_28, kernelContext_73);
    uint _S490 = atomic_load_explicit(kernelContext_73->atom_0+(_S489 + 4U), memory_order_relaxed);
    uint _S491 = atomic_load_explicit(kernelContext_73->atom_0+(_S489 + 1U), memory_order_relaxed);
    uint _S492 = atomic_load_explicit(kernelContext_73->atom_0+_S489, memory_order_relaxed);
    float end_1 = group_uniform_float_0((as_type<float>((_S492))), i_4);
    for(;;)
    {
        float best_1 = end_1;
        j_2 = i_4;
        for(;;)
        {
            if(j_2 < _S491)
            {
            }
            else
            {
                break;
            }
            uint _S493 = w_il_tube_0(kernelContext_73);
            uint at_6 = _S493 + 2U * (_S490 + j_2);
            if((as_type<float>((*(kernelContext_73->work_0+at_6)))) <= (end_1 + _S488))
            {
                best_1 = max(best_1, (as_type<float>((*(kernelContext_73->work_0+(at_6 + 1U))))));
            }
            j_2 = j_2 + 64U;
        }
        (*kernelContext_73->g_il_best_0)[i_4] = best_1;
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
                (*kernelContext_73->g_il_best_0)[i_4] = max((*kernelContext_73->g_il_best_0)[i_4], (*kernelContext_73->g_il_best_0)[i_4 + step_2]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_2 = step_2 >> 1U;
        }
        float next_1 = group_uniform_float_0((*kernelContext_73->g_il_best_0)[int(0)], i_4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!(next_1 > end_1))
        {
            break;
        }
        end_1 = next_1;
    }
    if(i_4 == 0U)
    {
        atomic_store_explicit(kernelContext_73->atom_0+_S489, (as_type<uint>((end_1))), memory_order_relaxed);
    }
    return;
}

void il_tube_collect_1(uint k_20, KernelContext_0 thread* kernelContext_74)
{
    thread uint b_29;
    thread float2 t_5;
    bool _S494 = il_tube_1(k_20, &b_29, &t_5, kernelContext_74);
    if(!_S494)
    {
        return;
    }
    uint _S495 = w_il_tube_0(kernelContext_74);
    uint _S496 = a_il_0(b_29, kernelContext_74);
    uint _S497 = atomic_load_explicit(kernelContext_74->atom_0+(_S496 + 4U), memory_order_relaxed);
    uint _S498 = a_il_0(b_29, kernelContext_74);
    uint _S499 = atomic_fetch_add_explicit(kernelContext_74->atom_0+(_S498 + 5U), 1U, memory_order_relaxed);
    uint at_7 = _S495 + 2U * (_S497 + _S499);
    *(kernelContext_74->work_0+at_7) = (as_type<uint>((t_5.x)));
    *(kernelContext_74->work_0+(at_7 + 1U)) = (as_type<uint>((t_5.y)));
    return;
}

void il_tube_base_1(uint b_30, KernelContext_0 thread* kernelContext_75)
{
    bool _S500;
    if(b_30 < (kernelContext_75->params_0->body_count_0))
    {
        bool _S501 = il_struck_1(b_30, kernelContext_75);
        _S500 = _S501;
    }
    else
    {
        _S500 = false;
    }
    if(_S500)
    {
        uint _S502 = a_il_0(b_30, kernelContext_75);
        atomic<uint> device* _S503 = kernelContext_75->atom_0+(_S502 + 4U);
        uint _S504 = a_il_cursor_1(kernelContext_75);
        atomic<uint> device* _S505 = kernelContext_75->atom_0+_S504;
        uint _S506 = atomic_load_explicit(kernelContext_75->atom_0+(_S502 + 1U), memory_order_relaxed);
        uint _S507 = atomic_fetch_add_explicit(_S505, _S506, memory_order_relaxed);
        atomic_store_explicit(_S503, _S507, memory_order_relaxed);
    }
    return;
}

void il_tube_count_1(uint k_21, KernelContext_0 thread* kernelContext_76)
{
    thread uint b_31;
    thread float2 t_6;
    bool _S508 = il_tube_1(k_21, &b_31, &t_6, kernelContext_76);
    if(_S508)
    {
        uint _S509 = a_il_0(b_31, kernelContext_76);
        uint _S510 = atomic_fetch_add_explicit(kernelContext_76->atom_0+(_S509 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_first_1(uint shot_3, KernelContext_0 thread* kernelContext_77)
{
    uint _S511 = a_strike_0(kernelContext_77);
    return _S511 + 29U * kernelContext_77->params_0->body_count_0 + shot_3;
}

void il_begin_1(uint b_32, KernelContext_0 thread* kernelContext_78)
{
    bool _S512;
    if(b_32 >= (kernelContext_78->params_0->body_count_0))
    {
        _S512 = true;
    }
    else
    {
        bool _S513 = il_struck_1(b_32, kernelContext_78);
        _S512 = !_S513;
    }
    if(_S512)
    {
        return;
    }
    thread uint shot_4;
    bool _S514 = struck_0(b_32, &shot_4, kernelContext_78);
    uint _S515 = a_strike_0(kernelContext_78);
    uint _S516 = atomic_load_explicit(kernelContext_78->atom_0+(_S515 + 9U * b_32 + 6U), memory_order_relaxed);
    uint _S517 = a_first_1(shot_4, kernelContext_78);
    uint _S518 = atomic_load_explicit(kernelContext_78->atom_0+_S517, memory_order_relaxed);
    if(_S516 != _S518)
    {
        uint _S519 = strike_at_0(b_32, kernelContext_78);
        (*(kernelContext_78->frames_0+_S519))[int(3)] = 0.0f;
        return;
    }
    thread float3 contact_15;
    thread float3 d_17;
    thread float speed_13;
    il_path_1(b_32, &contact_15, &d_17, &speed_13, kernelContext_78);
    uint _S520 = il_entry_1(b_32, kernelContext_78);
    IlStriker_0 _S521 = il_striker_1(b_32, kernelContext_78);
    float2 _S522 = il_interval_0(_S520, contact_15, d_17, _S521.radius_0, kernelContext_78);
    uint _S523 = a_il_0(b_32, kernelContext_78);
    atomic_store_explicit(kernelContext_78->atom_0+_S523, (as_type<uint>((max(_S522.y, 0.0f)))), memory_order_relaxed);
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
        atomic_store_explicit(kernelContext_78->atom_0+(_S523 + i_5), 0U, memory_order_relaxed);
        i_5 = i_5 + 1U;
    }
    return;
}

uint w_cs_rows_1(KernelContext_0 thread* kernelContext_79)
{
    uint _S524 = w_host_breaks_0(kernelContext_79);
    return _S524 + 1U + kernelContext_79->params_0->bond_count_0;
}

uint w_frame_0(KernelContext_0 thread* kernelContext_80)
{
    uint _S525 = w_cs_rows_1(kernelContext_80);
    return _S525 + 1U;
}

uint frame_parity_0(KernelContext_0 thread* kernelContext_81)
{
    uint _S526 = w_frame_0(kernelContext_81);
    return *(kernelContext_81->work_0+_S526);
}

uint l_pool_0(uint half_1, KernelContext_0 thread* kernelContext_82)
{
    return kernelContext_82->params_0->chunk_count_0 + 3U * kernelContext_82->params_0->pool_cap_0 * half_1;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S527 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S527, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S528 = max(0.0f, _S527);
    float _S529 = min(dt_1, _S527 + duration_0);
    bool _S530;
    if(_S529 > _S528)
    {
        _S530 = duration_0 > 0.0f;
    }
    else
    {
        _S530 = false;
    }
    float sum_1;
    if(_S530)
    {
        float w_2 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S528 - _S527)) - cos(w_2 * (_S529 - _S527)));
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

uint l_vec_0(KernelContext_0 thread* kernelContext_83)
{
    return kernelContext_83->params_0->chunk_count_0 + 6U * kernelContext_83->params_0->pool_cap_0;
}

[[kernel]] void fr_loads(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], uint device* sort_1 [[buffer(11)]], atomic<uint> device* atom_1 [[buffer(5)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], packed_float4 device* loads_1 [[buffer(12)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]], packed_float4 device* view_chunks_1 [[buffer(17)]], ViewParams_0 constant* view_1 [[buffer(16)]], packed_float4 device* objects_1 [[buffer(18)]], uint device* shot_bodies_1 [[buffer(20)]], float device* lines_1 [[buffer(19)]], packed_float4 device* geometry_1 [[buffer(21)]], packed_float4 device* shots_1 [[buffer(22)]], packed_float4 device* lp_1 [[buffer(24)]], packed_float4 device* ext_2 [[buffer(23)]], packed_float4 device* engine_contacts_1 [[buffer(25)]], CrushParams_0 constant* crush_1 [[buffer(28)]], packed_float4 device* crush_rw_1 [[buffer(27)]], packed_float4 device* crush_tab_1 [[buffer(26)]])
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
    thread KernelContext_0 kernelContext_84;
    (&kernelContext_84)->params_0 = params_1;
    (&kernelContext_84)->work_0 = work_1;
    (&kernelContext_84)->frames_0 = frames_1;
    (&kernelContext_84)->bodies_0 = bodies_1;
    (&kernelContext_84)->chunks_0 = chunks_1;
    (&kernelContext_84)->sort_0 = sort_1;
    (&kernelContext_84)->atom_0 = atom_1;
    (&kernelContext_84)->manifolds_0 = manifolds_1;
    (&kernelContext_84)->loads_0 = loads_1;
    (&kernelContext_84)->bonds_0 = bonds_1;
    (&kernelContext_84)->args_0 = args_1;
    (&kernelContext_84)->info_0 = info_1;
    (&kernelContext_84)->scan_0 = scan_1;
    (&kernelContext_84)->parts_0 = parts_1;
    (&kernelContext_84)->colliders_0 = colliders_1;
    (&kernelContext_84)->collider_info_0 = collider_info_1;
    (&kernelContext_84)->view_chunks_0 = view_chunks_1;
    (&kernelContext_84)->view_0 = view_1;
    (&kernelContext_84)->objects_0 = objects_1;
    (&kernelContext_84)->shot_bodies_0 = shot_bodies_1;
    (&kernelContext_84)->lines_0 = lines_1;
    (&kernelContext_84)->geometry_0 = geometry_1;
    (&kernelContext_84)->shots_0 = shots_1;
    (&kernelContext_84)->lp_0 = lp_1;
    (&kernelContext_84)->ext_0 = ext_2;
    (&kernelContext_84)->engine_contacts_0 = engine_contacts_1;
    (&kernelContext_84)->crush_0 = crush_1;
    (&kernelContext_84)->crush_rw_0 = crush_rw_1;
    (&kernelContext_84)->crush_tab_0 = crush_tab_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_84)->g_scan_0 = &g_scan_1;
    threadgroup array<float, int(64)> g_il_best_1;
    (&kernelContext_84)->g_il_best_0 = &g_il_best_1;
    uint k_22 = id_0.x;
    if(k_22 >= (params_1->chunk_count_0))
    {
        return;
    }
    uint _S531 = body_of_0(k_22, &kernelContext_84);
    uint b_33 = 7U * _S531;
    thread Quat_0 q_11;
    thread float3 origin_3;
    structure_pose_0(_S531, b_33, &q_11, &origin_3, &kernelContext_84);
    uint _S532 = 4U * k_22;
    float4 _S533 = float4(*((&kernelContext_84)->chunks_0+_S532)) ;
    float3 _S534 = _S533.xyz;
    thread Quat_0 _S535 = q_11;
    float3 _S536 = rotate_0(&_S535, _S534);
    float3 centre_0 = origin_3 + _S536;
    float inv_mass_2 = (float4(*((&kernelContext_84)->bodies_0+b_33)) ).w;
    float m_body_0;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint _S537 = frame_parity_0(&kernelContext_84);
    uint to_0 = 1U - _S537;
    uint _S538 = w_pulses_0(&kernelContext_84);
    uint _S539 = _S538 + _S532;
    uint _S540 = _S539 + 2U * _S537;
    uint off_0 = *((&kernelContext_84)->work_0+_S540);
    uint device* _S541 = (&kernelContext_84)->work_0+(_S540 + 1U);
    uint carried_0 = *_S541;
    uint room_0 = *_S541 + (&kernelContext_84)->sort_0[params_1->chunk_count_0 + 1U + k_22];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S542 = a_counters_1(&kernelContext_84);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_84)->atom_0+(_S542 + 5U + to_0), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_84)->params_0->pool_cap_0))
        {
            uint _S543 = atomic_fetch_or_explicit((&kernelContext_84)->atom_0+(_S542 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_84)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_84)->params_0->pool_cap_0 - base_1;
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
        uint _S544 = base_0;
        base_0 = base_1;
        room_1 = _S544;
    }
    else
    {
        base_0 = 0U;
        room_1 = room_0;
    }
    uint _S545 = params_1->chunk_count_0;
    float _S546 = _S533.w;
    float4 _S547 = float4(*((&kernelContext_84)->chunks_0+(_S532 + 3U))) ;
    float _S548 = (&kernelContext_84)->params_0->band_0;
    float _S549 = (&kernelContext_84)->params_0->dt_0;
    bool _S550 = ((&kernelContext_84)->params_0->impact_peak_0) != 0U;
    uint _S551 = l_pool_0(_S537, &kernelContext_84);
    uint _S552 = _S551 + 3U * off_0;
    uint _S553 = l_pool_0(to_0, &kernelContext_84);
    uint _S554 = _S553 + 3U * base_0;
    for(;;)
    {
        float3 _S555 = float3(0.0f) ;
        uint _S556 = (&kernelContext_84)->sort_0[k_22];
        uint _S557 = (&kernelContext_84)->sort_0[_S545 + 1U + k_22];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_3 = 0U;
        f_total_0 = _S555;
        t_total_0 = _S555;
        for(;;)
        {
            if(step_3 < _S557)
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
                if(q3_0 < _S557)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_84)->manifolds_0[_S556 + q3_0].ids_0) ).y + 1U;
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
            Record_natural_0 r_19 = (&kernelContext_84)->manifolds_0[_S556 + p_10];
            float4 _S558 = float4(r_19.impulse_0) ;
            float3 _S559 = _S558.xyz;
            float j_3 = length(_S559);
            if(j_3 == 0.0f)
            {
                uint _S560 = step_3 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_3 = _S560;
                continue;
            }
            float m_other_0 = (float4(r_19.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S546;
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
            float _S561 = min(j_3, 2.0f * m_1 * max(_S558.w - _S548, 0.0f));
            if(_S561 < j_3)
            {
                float3 f_1 = _S559 / float3(j_3)  * float3(((j_3 - _S561) / _S549)) ;
                float3 _S562 = t_total_0 + cross((float4(r_19.point_0) ).xyz - centre_0, f_1);
                f_total_1 = f_total_0 + f_1;
                t_total_1 = _S562;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            uint _S560 = step_3 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_3 = _S560;
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
            uint _S563 = _S552 + 3U * p_10;
            float4 _S564 = float4(*((&kernelContext_84)->loads_0+_S563)) ;
            thread float4 a_5 = _S564;
            float4 _S565 = float4(*((&kernelContext_84)->loads_0+(_S563 + 1U))) ;
            float4 _S566 = float4(*((&kernelContext_84)->loads_0+(_S563 + 2U))) ;
            float _S567 = _S565.w;
            float mean_0 = pulse_mean_0(_S564.x, _S564.y, _S564.z, _S564.w, _S567, _S549);
            if(mean_0 != 0.0f)
            {
                float3 f_2 = _S566.xyz * float3(mean_0) ;
                float3 _S568 = t_total_0 + cross(_S565.xyz - centre_0, f_2);
                f_total_0 = f_total_0 + f_2;
                t_total_0 = _S568;
            }
            if((a_5.x + a_5.z + _S567) > _S549)
            {
                a_5.x = a_5.x - _S549;
                uint _S569 = _S554 + 3U * kept_0;
                *((&kernelContext_84)->loads_0+_S569) = packed_float4(a_5) ;
                *((&kernelContext_84)->loads_0+(_S569 + 1U)) = packed_float4(_S565) ;
                *((&kernelContext_84)->loads_0+(_S569 + 2U)) = packed_float4(_S566) ;
                kept_0 = kept_0 + 1U;
            }
            p_10 = p_10 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S557)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_20 = (&kernelContext_84)->manifolds_0[_S556 + q3_0];
            float4 _S570 = float4(r_20.impulse_0) ;
            float3 _S571 = _S570.xyz;
            float j_4 = length(_S571);
            if(j_4 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S572 = float4(r_20.other_0) ;
            float m_other_1 = _S572.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S546;
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
            float _S573 = _S570.w;
            float _S574 = min(j_4, 2.0f * m_1 * max(_S573 - _S548, 0.0f));
            if(_S574 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S575 = _S547.y;
            float duration_1 = hertz_duration_0(m_1, min(_S572.z, _S547.z), 1.0f / ((1.0f - _S575 * _S575) / _S547.x + 1.0f / _S572.y), _S573);
            float peak_1 = 3.14159274101257324f * _S574 / (2.0f * duration_1);
            float3 dir_0 = _S571 / float3(j_4) ;
            float mean_1;
            if(_S550)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S549);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_3 = dir_0 * float3(mean_1) ;
                float3 _S576 = t_total_0 + cross((float4(r_20.point_0) ).xyz - centre_0, f_3);
                f_total_1 = f_total_0 + f_3;
                t_total_1 = _S576;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            uint kept_1;
            if(duration_1 > _S549)
            {
                if(kept_0 < room_1)
                {
                    uint _S577 = _S554 + 3U * kept_0;
                    *((&kernelContext_84)->loads_0+_S577) = packed_float4(float4(- _S549, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_84)->loads_0+(_S577 + 1U)) = packed_float4(float4((float4(r_20.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_84)->loads_0+(_S577 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
        uint _S578 = a_counters_1(&kernelContext_84);
        uint _S579 = atomic_fetch_or_explicit((&kernelContext_84)->atom_0+(_S578 + 3U), 8U, memory_order_relaxed);
    }
    uint _S580 = _S539 + 2U * to_0;
    *((&kernelContext_84)->work_0+_S580) = base_0;
    *((&kernelContext_84)->work_0+(_S580 + 1U)) = kept_0;
    *((&kernelContext_84)->loads_0+k_22) = packed_float4(float4(length(f_total_0), f_total_0)) ;
    if(((&kernelContext_84)->params_0->load_model_0) == 1U)
    {
        uint _S581 = l_vec_0(&kernelContext_84);
        uint _S582 = _S581 + 2U * k_22;
        *((&kernelContext_84)->loads_0+_S582) = packed_float4(float4(f_total_0, 0.0f)) ;
        *((&kernelContext_84)->loads_0+(_S582 + 1U)) = packed_float4(float4(t_total_0, 0.0f)) ;
    }
    return;
}

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], atomic<uint> device* atom_2 [[buffer(5)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]], packed_float4 device* geometry_2 [[buffer(21)]], packed_float4 device* shots_2 [[buffer(22)]], packed_float4 device* lp_2 [[buffer(24)]], packed_float4 device* ext_3 [[buffer(23)]], packed_float4 device* engine_contacts_2 [[buffer(25)]], CrushParams_0 constant* crush_2 [[buffer(28)]], packed_float4 device* crush_rw_2 [[buffer(27)]], packed_float4 device* crush_tab_2 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_85;
    (&kernelContext_85)->params_0 = params_2;
    (&kernelContext_85)->work_0 = work_2;
    (&kernelContext_85)->frames_0 = frames_2;
    (&kernelContext_85)->bodies_0 = bodies_2;
    (&kernelContext_85)->chunks_0 = chunks_2;
    (&kernelContext_85)->sort_0 = sort_2;
    (&kernelContext_85)->atom_0 = atom_2;
    (&kernelContext_85)->manifolds_0 = manifolds_2;
    (&kernelContext_85)->loads_0 = loads_2;
    (&kernelContext_85)->bonds_0 = bonds_2;
    (&kernelContext_85)->args_0 = args_2;
    (&kernelContext_85)->info_0 = info_2;
    (&kernelContext_85)->scan_0 = scan_2;
    (&kernelContext_85)->parts_0 = parts_2;
    (&kernelContext_85)->colliders_0 = colliders_2;
    (&kernelContext_85)->collider_info_0 = collider_info_2;
    (&kernelContext_85)->view_chunks_0 = view_chunks_2;
    (&kernelContext_85)->view_0 = view_2;
    (&kernelContext_85)->objects_0 = objects_2;
    (&kernelContext_85)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_85)->lines_0 = lines_2;
    (&kernelContext_85)->geometry_0 = geometry_2;
    (&kernelContext_85)->shots_0 = shots_2;
    (&kernelContext_85)->lp_0 = lp_2;
    (&kernelContext_85)->ext_0 = ext_3;
    (&kernelContext_85)->engine_contacts_0 = engine_contacts_2;
    (&kernelContext_85)->crush_0 = crush_2;
    (&kernelContext_85)->crush_rw_0 = crush_rw_2;
    (&kernelContext_85)->crush_tab_0 = crush_tab_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_85)->g_scan_0 = &g_scan_2;
    threadgroup array<float, int(64)> g_il_best_2;
    (&kernelContext_85)->g_il_best_0 = &g_il_best_2;
    uint i_6 = id_1.x;
    bool _S583;
    if(i_6 >= (params_2->bond_count_0))
    {
        _S583 = true;
    }
    else
    {
        uint _S584 = w_alive_0(&kernelContext_85);
        _S583 = (*((&kernelContext_85)->work_0+(_S584 + i_6))) == 0U;
    }
    if(_S583)
    {
        return;
    }
    uint4 _S585 = uint4(*((&kernelContext_85)->bonds_0+i_6)) ;
    uint _S586 = _S585.x;
    if((max((float4(*((&kernelContext_85)->loads_0+_S586)) ).x, (float4(*((&kernelContext_85)->loads_0+_S585.y)) ).x)) > (as_type<float>((_S585.z))))
    {
        uint _S587 = w_alive_0(&kernelContext_85);
        *((&kernelContext_85)->work_0+(_S587 + i_6)) = 0U;
        uint _S588 = a_dirty_0(&kernelContext_85);
        uint _S589 = body_of_0(_S586, &kernelContext_85);
        uint _S590 = atomic_fetch_or_explicit((&kernelContext_85)->atom_0+(_S588 + _S589), 1U, memory_order_relaxed);
        uint _S591 = a_counters_1(&kernelContext_85);
        uint _S592 = atomic_fetch_add_explicit((&kernelContext_85)->atom_0+_S591, 1U, memory_order_relaxed);
        uint _S593 = atomic_fetch_add_explicit((&kernelContext_85)->atom_0+(_S591 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_host_breaks(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], packed_float4 device* lp_3 [[buffer(24)]], packed_float4 device* ext_4 [[buffer(23)]], packed_float4 device* engine_contacts_3 [[buffer(25)]], CrushParams_0 constant* crush_3 [[buffer(28)]], packed_float4 device* crush_rw_3 [[buffer(27)]], packed_float4 device* crush_tab_3 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_86;
    (&kernelContext_86)->params_0 = params_3;
    (&kernelContext_86)->work_0 = work_3;
    (&kernelContext_86)->frames_0 = frames_3;
    (&kernelContext_86)->bodies_0 = bodies_3;
    (&kernelContext_86)->chunks_0 = chunks_3;
    (&kernelContext_86)->sort_0 = sort_3;
    (&kernelContext_86)->atom_0 = atom_3;
    (&kernelContext_86)->manifolds_0 = manifolds_3;
    (&kernelContext_86)->loads_0 = loads_3;
    (&kernelContext_86)->bonds_0 = bonds_3;
    (&kernelContext_86)->args_0 = args_3;
    (&kernelContext_86)->info_0 = info_3;
    (&kernelContext_86)->scan_0 = scan_3;
    (&kernelContext_86)->parts_0 = parts_3;
    (&kernelContext_86)->colliders_0 = colliders_3;
    (&kernelContext_86)->collider_info_0 = collider_info_3;
    (&kernelContext_86)->view_chunks_0 = view_chunks_3;
    (&kernelContext_86)->view_0 = view_3;
    (&kernelContext_86)->objects_0 = objects_3;
    (&kernelContext_86)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_86)->lines_0 = lines_3;
    (&kernelContext_86)->geometry_0 = geometry_3;
    (&kernelContext_86)->shots_0 = shots_3;
    (&kernelContext_86)->lp_0 = lp_3;
    (&kernelContext_86)->ext_0 = ext_4;
    (&kernelContext_86)->engine_contacts_0 = engine_contacts_3;
    (&kernelContext_86)->crush_0 = crush_3;
    (&kernelContext_86)->crush_rw_0 = crush_rw_3;
    (&kernelContext_86)->crush_tab_0 = crush_tab_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_86)->g_scan_0 = &g_scan_3;
    threadgroup array<float, int(64)> g_il_best_3;
    (&kernelContext_86)->g_il_best_0 = &g_il_best_3;
    uint i_7 = id_2.x;
    uint _S594 = w_host_breaks_0(&kernelContext_86);
    if(i_7 >= (*((&kernelContext_86)->work_0+_S594)))
    {
        return;
    }
    uint device* _S595 = (&kernelContext_86)->work_0+(_S594 + 1U + i_7);
    uint b_34 = *_S595;
    bool _S596;
    if((*_S595) >= ((&kernelContext_86)->params_0->bond_count_0))
    {
        _S596 = true;
    }
    else
    {
        uint _S597 = w_alive_0(&kernelContext_86);
        _S596 = (*((&kernelContext_86)->work_0+(_S597 + b_34))) == 0U;
    }
    if(_S596)
    {
        return;
    }
    uint _S598 = w_alive_0(&kernelContext_86);
    *((&kernelContext_86)->work_0+(_S598 + b_34)) = 0U;
    uint _S599 = a_dirty_0(&kernelContext_86);
    uint _S600 = body_of_0((uint4(*((&kernelContext_86)->bonds_0+b_34)) ).x, &kernelContext_86);
    uint _S601 = atomic_fetch_or_explicit((&kernelContext_86)->atom_0+(_S599 + _S600), 1U, memory_order_relaxed);
    uint _S602 = a_counters_1(&kernelContext_86);
    uint _S603 = atomic_fetch_add_explicit((&kernelContext_86)->atom_0+_S602, 1U, memory_order_relaxed);
    uint _S604 = atomic_fetch_add_explicit((&kernelContext_86)->atom_0+(_S602 + 1U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_frame_begin(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], packed_float4 device* lp_4 [[buffer(24)]], packed_float4 device* ext_5 [[buffer(23)]], packed_float4 device* engine_contacts_4 [[buffer(25)]], CrushParams_0 constant* crush_4 [[buffer(28)]], packed_float4 device* crush_rw_4 [[buffer(27)]], packed_float4 device* crush_tab_4 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_87;
    (&kernelContext_87)->params_0 = params_4;
    (&kernelContext_87)->work_0 = work_4;
    (&kernelContext_87)->frames_0 = frames_4;
    (&kernelContext_87)->bodies_0 = bodies_4;
    (&kernelContext_87)->chunks_0 = chunks_4;
    (&kernelContext_87)->sort_0 = sort_4;
    (&kernelContext_87)->atom_0 = atom_4;
    (&kernelContext_87)->manifolds_0 = manifolds_4;
    (&kernelContext_87)->loads_0 = loads_4;
    (&kernelContext_87)->bonds_0 = bonds_4;
    (&kernelContext_87)->args_0 = args_4;
    (&kernelContext_87)->info_0 = info_4;
    (&kernelContext_87)->scan_0 = scan_4;
    (&kernelContext_87)->parts_0 = parts_4;
    (&kernelContext_87)->colliders_0 = colliders_4;
    (&kernelContext_87)->collider_info_0 = collider_info_4;
    (&kernelContext_87)->view_chunks_0 = view_chunks_4;
    (&kernelContext_87)->view_0 = view_4;
    (&kernelContext_87)->objects_0 = objects_4;
    (&kernelContext_87)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_87)->lines_0 = lines_4;
    (&kernelContext_87)->geometry_0 = geometry_4;
    (&kernelContext_87)->shots_0 = shots_4;
    (&kernelContext_87)->lp_0 = lp_4;
    (&kernelContext_87)->ext_0 = ext_5;
    (&kernelContext_87)->engine_contacts_0 = engine_contacts_4;
    (&kernelContext_87)->crush_0 = crush_4;
    (&kernelContext_87)->crush_rw_0 = crush_rw_4;
    (&kernelContext_87)->crush_tab_0 = crush_tab_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_87)->g_scan_0 = &g_scan_4;
    threadgroup array<float, int(64)> g_il_best_4;
    (&kernelContext_87)->g_il_best_0 = &g_il_best_4;
    uint _S605 = w_frame_0(&kernelContext_87);
    *((&kernelContext_87)->work_0+_S605) = 1U - *((&kernelContext_87)->work_0+_S605);
    uint _S606 = _S605 + 1U;
    *((&kernelContext_87)->work_0+_S606) = *((&kernelContext_87)->work_0+_S606) + 1U;
    uint _S607 = _S605 + 2U;
    *((&kernelContext_87)->work_0+_S607) = *((&kernelContext_87)->work_0+_S607) + 1U;
    return;
}

uint groups_of_0(uint n_3, uint size_0)
{
    uint _S608 = (n_3 + size_0 - 1U) / size_0;
    return _S608;
}

void set_args_0(uint at_8, uint x_3, KernelContext_0 thread* kernelContext_88)
{
    uint _S609 = 3U * at_8;
    *(kernelContext_88->args_0+_S609) = x_3;
    *(kernelContext_88->args_0+(_S609 + 1U)) = 1U;
    *(kernelContext_88->args_0+(_S609 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], packed_float4 device* lp_5 [[buffer(24)]], packed_float4 device* ext_6 [[buffer(23)]], packed_float4 device* engine_contacts_5 [[buffer(25)]], CrushParams_0 constant* crush_5 [[buffer(28)]], packed_float4 device* crush_rw_5 [[buffer(27)]], packed_float4 device* crush_tab_5 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_89;
    (&kernelContext_89)->params_0 = params_5;
    (&kernelContext_89)->work_0 = work_5;
    (&kernelContext_89)->frames_0 = frames_5;
    (&kernelContext_89)->bodies_0 = bodies_5;
    (&kernelContext_89)->chunks_0 = chunks_5;
    (&kernelContext_89)->sort_0 = sort_5;
    (&kernelContext_89)->atom_0 = atom_5;
    (&kernelContext_89)->manifolds_0 = manifolds_5;
    (&kernelContext_89)->loads_0 = loads_5;
    (&kernelContext_89)->bonds_0 = bonds_5;
    (&kernelContext_89)->args_0 = args_5;
    (&kernelContext_89)->info_0 = info_5;
    (&kernelContext_89)->scan_0 = scan_5;
    (&kernelContext_89)->parts_0 = parts_5;
    (&kernelContext_89)->colliders_0 = colliders_5;
    (&kernelContext_89)->collider_info_0 = collider_info_5;
    (&kernelContext_89)->view_chunks_0 = view_chunks_5;
    (&kernelContext_89)->view_0 = view_5;
    (&kernelContext_89)->objects_0 = objects_5;
    (&kernelContext_89)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_89)->lines_0 = lines_5;
    (&kernelContext_89)->geometry_0 = geometry_5;
    (&kernelContext_89)->shots_0 = shots_5;
    (&kernelContext_89)->lp_0 = lp_5;
    (&kernelContext_89)->ext_0 = ext_6;
    (&kernelContext_89)->engine_contacts_0 = engine_contacts_5;
    (&kernelContext_89)->crush_0 = crush_5;
    (&kernelContext_89)->crush_rw_0 = crush_rw_5;
    (&kernelContext_89)->crush_tab_0 = crush_tab_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_89)->g_scan_0 = &g_scan_5;
    threadgroup array<float, int(64)> g_il_best_5;
    (&kernelContext_89)->g_il_best_0 = &g_il_best_5;
    uint _S610 = a_counters_1(&kernelContext_89);
    uint _S611 = atomic_load_explicit((&kernelContext_89)->atom_0+_S610, memory_order_relaxed);
    bool run_0 = _S611 != 0U;
    atomic_store_explicit((&kernelContext_89)->atom_0+_S610, 0U, memory_order_relaxed);
    uint _S612 = _S610 + 5U;
    uint _S613 = frame_parity_0(&kernelContext_89);
    atomic_store_explicit((&kernelContext_89)->atom_0+(_S612 + _S613), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S614 = atomic_fetch_add_explicit((&kernelContext_89)->atom_0+(_S610 + 4U), 1U, memory_order_relaxed);
    }
    uint _S615;
    if(run_0)
    {
        uint _S616 = groups_of_0((&kernelContext_89)->params_0->chunk_count_0, 64U);
        _S615 = _S616;
    }
    else
    {
        _S615 = 0U;
    }
    set_args_0(0U, _S615, &kernelContext_89);
    if(run_0)
    {
        uint _S617 = groups_of_0((&kernelContext_89)->params_0->bond_count_0, 64U);
        _S615 = _S617;
    }
    else
    {
        _S615 = 0U;
    }
    set_args_0(1U, _S615, &kernelContext_89);
    if(run_0)
    {
        uint _S618 = groups_of_0((&kernelContext_89)->params_0->body_count_0, 64U);
        _S615 = _S618;
    }
    else
    {
        _S615 = 0U;
    }
    set_args_0(2U, _S615, &kernelContext_89);
    if(run_0)
    {
        uint _S619 = groups_of_0((&kernelContext_89)->params_0->chunk_count_0, 1024U);
        _S615 = _S619;
    }
    else
    {
        _S615 = 0U;
    }
    set_args_0(3U, _S615, &kernelContext_89);
    if(run_0)
    {
        uint _S620 = groups_of_0((&kernelContext_89)->params_0->body_count_0, 1024U);
        _S615 = _S620;
    }
    else
    {
        _S615 = 0U;
    }
    set_args_0(4U, _S615, &kernelContext_89);
    int _S621;
    if(run_0)
    {
        _S621 = int(1);
    }
    else
    {
        _S621 = int(0);
    }
    set_args_0(5U, uint(_S621), &kernelContext_89);
    return;
}

bool dirty_0(uint body_1, KernelContext_0 thread* kernelContext_90)
{
    uint _S622 = a_dirty_0(kernelContext_90);
    uint _S623 = atomic_load_explicit(kernelContext_90->atom_0+(_S622 + body_1), memory_order_relaxed);
    return _S623 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_91)
{
    return kernelContext_91->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_92)
{
    return 2U * kernelContext_92->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], packed_float4 device* lp_6 [[buffer(24)]], packed_float4 device* ext_7 [[buffer(23)]], packed_float4 device* engine_contacts_6 [[buffer(25)]], CrushParams_0 constant* crush_6 [[buffer(28)]], packed_float4 device* crush_rw_6 [[buffer(27)]], packed_float4 device* crush_tab_6 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_93;
    (&kernelContext_93)->params_0 = params_6;
    (&kernelContext_93)->work_0 = work_6;
    (&kernelContext_93)->frames_0 = frames_6;
    (&kernelContext_93)->bodies_0 = bodies_6;
    (&kernelContext_93)->chunks_0 = chunks_6;
    (&kernelContext_93)->sort_0 = sort_6;
    (&kernelContext_93)->atom_0 = atom_6;
    (&kernelContext_93)->manifolds_0 = manifolds_6;
    (&kernelContext_93)->loads_0 = loads_6;
    (&kernelContext_93)->bonds_0 = bonds_6;
    (&kernelContext_93)->args_0 = args_6;
    (&kernelContext_93)->info_0 = info_6;
    (&kernelContext_93)->scan_0 = scan_6;
    (&kernelContext_93)->parts_0 = parts_6;
    (&kernelContext_93)->colliders_0 = colliders_6;
    (&kernelContext_93)->collider_info_0 = collider_info_6;
    (&kernelContext_93)->view_chunks_0 = view_chunks_6;
    (&kernelContext_93)->view_0 = view_6;
    (&kernelContext_93)->objects_0 = objects_6;
    (&kernelContext_93)->shot_bodies_0 = shot_bodies_6;
    (&kernelContext_93)->lines_0 = lines_6;
    (&kernelContext_93)->geometry_0 = geometry_6;
    (&kernelContext_93)->shots_0 = shots_6;
    (&kernelContext_93)->lp_0 = lp_6;
    (&kernelContext_93)->ext_0 = ext_7;
    (&kernelContext_93)->engine_contacts_0 = engine_contacts_6;
    (&kernelContext_93)->crush_0 = crush_6;
    (&kernelContext_93)->crush_rw_0 = crush_rw_6;
    (&kernelContext_93)->crush_tab_0 = crush_tab_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_93)->g_scan_0 = &g_scan_6;
    threadgroup array<float, int(64)> g_il_best_6;
    (&kernelContext_93)->g_il_best_0 = &g_il_best_6;
    uint c_6 = id_5.x;
    bool _S624;
    if(c_6 >= (params_6->chunk_count_0))
    {
        _S624 = true;
    }
    else
    {
        uint _S625 = body_of_0(c_6, &kernelContext_93);
        bool _S626 = dirty_0(_S625, &kernelContext_93);
        _S624 = !_S626;
    }
    if(_S624)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_93)->atom_0+(a_parent_0() + c_6), c_6, memory_order_relaxed);
    uint _S627 = a_count_0(&kernelContext_93);
    atomic_store_explicit((&kernelContext_93)->atom_0+(_S627 + c_6), 0U, memory_order_relaxed);
    uint _S628 = a_anchor_0(&kernelContext_93);
    atomic_store_explicit((&kernelContext_93)->atom_0+(_S628 + c_6), 0U, memory_order_relaxed);
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
        uint _S629 = a_sums_0(&kernelContext_93);
        atomic_store_explicit((&kernelContext_93)->atom_0+(_S629 + 20U * c_6 + w_3), 0U, memory_order_relaxed);
        w_3 = w_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], packed_float4 device* lp_7 [[buffer(24)]], packed_float4 device* ext_8 [[buffer(23)]], packed_float4 device* engine_contacts_7 [[buffer(25)]], CrushParams_0 constant* crush_7 [[buffer(28)]], packed_float4 device* crush_rw_7 [[buffer(27)]], packed_float4 device* crush_tab_7 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_94;
    (&kernelContext_94)->params_0 = params_7;
    (&kernelContext_94)->work_0 = work_7;
    (&kernelContext_94)->frames_0 = frames_7;
    (&kernelContext_94)->bodies_0 = bodies_7;
    (&kernelContext_94)->chunks_0 = chunks_7;
    (&kernelContext_94)->sort_0 = sort_7;
    (&kernelContext_94)->atom_0 = atom_7;
    (&kernelContext_94)->manifolds_0 = manifolds_7;
    (&kernelContext_94)->loads_0 = loads_7;
    (&kernelContext_94)->bonds_0 = bonds_7;
    (&kernelContext_94)->args_0 = args_7;
    (&kernelContext_94)->info_0 = info_7;
    (&kernelContext_94)->scan_0 = scan_7;
    (&kernelContext_94)->parts_0 = parts_7;
    (&kernelContext_94)->colliders_0 = colliders_7;
    (&kernelContext_94)->collider_info_0 = collider_info_7;
    (&kernelContext_94)->view_chunks_0 = view_chunks_7;
    (&kernelContext_94)->view_0 = view_7;
    (&kernelContext_94)->objects_0 = objects_7;
    (&kernelContext_94)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_94)->lines_0 = lines_7;
    (&kernelContext_94)->geometry_0 = geometry_7;
    (&kernelContext_94)->shots_0 = shots_7;
    (&kernelContext_94)->lp_0 = lp_7;
    (&kernelContext_94)->ext_0 = ext_8;
    (&kernelContext_94)->engine_contacts_0 = engine_contacts_7;
    (&kernelContext_94)->crush_0 = crush_7;
    (&kernelContext_94)->crush_rw_0 = crush_rw_7;
    (&kernelContext_94)->crush_tab_0 = crush_tab_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_94)->g_scan_0 = &g_scan_7;
    threadgroup array<float, int(64)> g_il_best_7;
    (&kernelContext_94)->g_il_best_0 = &g_il_best_7;
    uint s_7 = id_6.x;
    bool _S630;
    if(s_7 >= (params_7->body_count_0))
    {
        _S630 = true;
    }
    else
    {
        bool _S631 = dirty_0(s_7, &kernelContext_94);
        _S630 = !_S631;
    }
    if(_S630)
    {
        return;
    }
    uint _S632 = a_best_count_0(&kernelContext_94);
    atomic_store_explicit((&kernelContext_94)->atom_0+(_S632 + s_7), 0U, memory_order_relaxed);
    uint _S633 = a_best_root_0(&kernelContext_94);
    atomic_store_explicit((&kernelContext_94)->atom_0+(_S633 + s_7), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_4, KernelContext_0 thread* kernelContext_95)
{
    uint _S634 = a_parent_0();
    uint _S635 = atomic_load_explicit(kernelContext_95->atom_0+(_S634 + x_4), memory_order_relaxed);
    uint p_11 = _S635;
    uint _S636 = x_4;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S637;
        if(guard_0 < 1048576U)
        {
            _S637 = p_11 != _S636;
        }
        else
        {
            _S637 = false;
        }
        if(_S637)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_95->atom_0+(_S634 + p_11), memory_order_relaxed);
        if(gp_0 != p_11)
        {
            uint _S638;
            {
            uint _metal_cas_comparand = p_11;
            atomic_compare_exchange_weak_explicit(kernelContext_95->atom_0+(_S634 + _S636), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S638 = _metal_cas_comparand;
            }
        }
        uint _S639 = guard_0 + 1U;
        _S636 = p_11;
        p_11 = gp_0;
        guard_0 = _S639;
    }
    return _S636;
}

[[kernel]] void fr_uf_union(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], packed_float4 device* lp_8 [[buffer(24)]], packed_float4 device* ext_9 [[buffer(23)]], packed_float4 device* engine_contacts_8 [[buffer(25)]], CrushParams_0 constant* crush_8 [[buffer(28)]], packed_float4 device* crush_rw_8 [[buffer(27)]], packed_float4 device* crush_tab_8 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_96;
    (&kernelContext_96)->params_0 = params_8;
    (&kernelContext_96)->work_0 = work_8;
    (&kernelContext_96)->frames_0 = frames_8;
    (&kernelContext_96)->bodies_0 = bodies_8;
    (&kernelContext_96)->chunks_0 = chunks_8;
    (&kernelContext_96)->sort_0 = sort_8;
    (&kernelContext_96)->atom_0 = atom_8;
    (&kernelContext_96)->manifolds_0 = manifolds_8;
    (&kernelContext_96)->loads_0 = loads_8;
    (&kernelContext_96)->bonds_0 = bonds_8;
    (&kernelContext_96)->args_0 = args_8;
    (&kernelContext_96)->info_0 = info_8;
    (&kernelContext_96)->scan_0 = scan_8;
    (&kernelContext_96)->parts_0 = parts_8;
    (&kernelContext_96)->colliders_0 = colliders_8;
    (&kernelContext_96)->collider_info_0 = collider_info_8;
    (&kernelContext_96)->view_chunks_0 = view_chunks_8;
    (&kernelContext_96)->view_0 = view_8;
    (&kernelContext_96)->objects_0 = objects_8;
    (&kernelContext_96)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_96)->lines_0 = lines_8;
    (&kernelContext_96)->geometry_0 = geometry_8;
    (&kernelContext_96)->shots_0 = shots_8;
    (&kernelContext_96)->lp_0 = lp_8;
    (&kernelContext_96)->ext_0 = ext_9;
    (&kernelContext_96)->engine_contacts_0 = engine_contacts_8;
    (&kernelContext_96)->crush_0 = crush_8;
    (&kernelContext_96)->crush_rw_0 = crush_rw_8;
    (&kernelContext_96)->crush_tab_0 = crush_tab_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_96)->g_scan_0 = &g_scan_8;
    threadgroup array<float, int(64)> g_il_best_8;
    (&kernelContext_96)->g_il_best_0 = &g_il_best_8;
    uint i_8 = id_7.x;
    bool _S640;
    if(i_8 >= (params_8->bond_count_0))
    {
        _S640 = true;
    }
    else
    {
        uint _S641 = w_alive_0(&kernelContext_96);
        _S640 = (*((&kernelContext_96)->work_0+(_S641 + i_8))) == 0U;
    }
    if(_S640)
    {
        return;
    }
    uint4 _S642 = uint4(*((&kernelContext_96)->bonds_0+i_8)) ;
    uint _S643 = _S642.x;
    uint _S644 = body_of_0(_S643, &kernelContext_96);
    bool _S645 = dirty_0(_S644, &kernelContext_96);
    if(!_S645)
    {
        return;
    }
    uint _S646 = _S642.y;
    uint a_6 = _S643;
    uint b_35 = _S646;
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
        uint _S647 = find_0(a_6, &kernelContext_96);
        uint _S648 = find_0(b_35, &kernelContext_96);
        if(_S647 == _S648)
        {
            return;
        }
        uint _S649 = max(_S647, _S648);
        uint _S650 = min(_S647, _S648);
        uint was_0;
        {
        uint _metal_cas_comparand = _S649;
        atomic_compare_exchange_weak_explicit((&kernelContext_96)->atom_0+(a_parent_0() + _S649), &_metal_cas_comparand, _S650, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S649)
        {
            return;
        }
        uint _S651 = guard_1 + 1U;
        a_6 = _S649;
        b_35 = _S650;
        guard_1 = _S651;
    }
    uint _S652 = a_counters_1(&kernelContext_96);
    uint _S653 = atomic_fetch_or_explicit((&kernelContext_96)->atom_0+(_S652 + 3U), 4U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], packed_float4 device* lp_9 [[buffer(24)]], packed_float4 device* ext_10 [[buffer(23)]], packed_float4 device* engine_contacts_9 [[buffer(25)]], CrushParams_0 constant* crush_9 [[buffer(28)]], packed_float4 device* crush_rw_9 [[buffer(27)]], packed_float4 device* crush_tab_9 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_97;
    (&kernelContext_97)->params_0 = params_9;
    (&kernelContext_97)->work_0 = work_9;
    (&kernelContext_97)->frames_0 = frames_9;
    (&kernelContext_97)->bodies_0 = bodies_9;
    (&kernelContext_97)->chunks_0 = chunks_9;
    (&kernelContext_97)->sort_0 = sort_9;
    (&kernelContext_97)->atom_0 = atom_9;
    (&kernelContext_97)->manifolds_0 = manifolds_9;
    (&kernelContext_97)->loads_0 = loads_9;
    (&kernelContext_97)->bonds_0 = bonds_9;
    (&kernelContext_97)->args_0 = args_9;
    (&kernelContext_97)->info_0 = info_9;
    (&kernelContext_97)->scan_0 = scan_9;
    (&kernelContext_97)->parts_0 = parts_9;
    (&kernelContext_97)->colliders_0 = colliders_9;
    (&kernelContext_97)->collider_info_0 = collider_info_9;
    (&kernelContext_97)->view_chunks_0 = view_chunks_9;
    (&kernelContext_97)->view_0 = view_9;
    (&kernelContext_97)->objects_0 = objects_9;
    (&kernelContext_97)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_97)->lines_0 = lines_9;
    (&kernelContext_97)->geometry_0 = geometry_9;
    (&kernelContext_97)->shots_0 = shots_9;
    (&kernelContext_97)->lp_0 = lp_9;
    (&kernelContext_97)->ext_0 = ext_10;
    (&kernelContext_97)->engine_contacts_0 = engine_contacts_9;
    (&kernelContext_97)->crush_0 = crush_9;
    (&kernelContext_97)->crush_rw_0 = crush_rw_9;
    (&kernelContext_97)->crush_tab_0 = crush_tab_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_97)->g_scan_0 = &g_scan_9;
    threadgroup array<float, int(64)> g_il_best_9;
    (&kernelContext_97)->g_il_best_0 = &g_il_best_9;
    uint c_7 = id_8.x;
    bool _S654;
    if(c_7 >= (params_9->chunk_count_0))
    {
        _S654 = true;
    }
    else
    {
        uint _S655 = body_of_0(c_7, &kernelContext_97);
        bool _S656 = dirty_0(_S655, &kernelContext_97);
        _S654 = !_S656;
    }
    if(_S654)
    {
        return;
    }
    uint _S657 = find_0(c_7, &kernelContext_97);
    uint _S658 = w_label_0(&kernelContext_97);
    *((&kernelContext_97)->work_0+(_S658 + c_7)) = _S657;
    uint _S659 = a_count_0(&kernelContext_97);
    uint _S660 = atomic_fetch_add_explicit((&kernelContext_97)->atom_0+(_S659 + _S657), 1U, memory_order_relaxed);
    if(((&kernelContext_97)->info_0[c_7] & 1U) != 0U)
    {
        uint _S661 = a_anchor_0(&kernelContext_97);
        uint _S662 = atomic_fetch_or_explicit((&kernelContext_97)->atom_0+(_S661 + _S657), 1U, memory_order_relaxed);
    }
    uint _S663 = 4U * c_7;
    float4 _S664 = float4(*((&kernelContext_97)->chunks_0+_S663)) ;
    float4 _S665 = float4(*((&kernelContext_97)->chunks_0+(_S663 + 1U))) ;
    float4 _S666 = float4(*((&kernelContext_97)->chunks_0+(_S663 + 2U))) ;
    float m_2 = _S664.w;
    float3 d_18 = _S664.xyz - (float4(*((&kernelContext_97)->chunks_0+4U * _S657)) ).xyz;
    float dd_0 = dot(d_18, d_18);
    uint _S667 = a_sums_0(&kernelContext_97);
    uint at_9 = _S667 + 20U * _S657;
    add64_0(at_9, m_2, (&kernelContext_97)->params_0->scale_m_0, &kernelContext_97);
    float _S668 = d_18.x;
    float _S669 = m_2 * _S668;
    add64_0(at_9 + 2U, _S669, (&kernelContext_97)->params_0->scale_s_0, &kernelContext_97);
    float _S670 = d_18.y;
    float _S671 = m_2 * _S670;
    add64_0(at_9 + 4U, _S671, (&kernelContext_97)->params_0->scale_s_0, &kernelContext_97);
    float _S672 = d_18.z;
    add64_0(at_9 + 6U, m_2 * _S672, (&kernelContext_97)->params_0->scale_s_0, &kernelContext_97);
    add64_0(at_9 + 8U, _S665.x + m_2 * (dd_0 - _S668 * _S668), (&kernelContext_97)->params_0->scale_j_0, &kernelContext_97);
    add64_0(at_9 + 10U, _S665.y + m_2 * (dd_0 - _S670 * _S670), (&kernelContext_97)->params_0->scale_j_0, &kernelContext_97);
    add64_0(at_9 + 12U, _S665.z + m_2 * (dd_0 - _S672 * _S672), (&kernelContext_97)->params_0->scale_j_0, &kernelContext_97);
    add64_0(at_9 + 14U, _S665.w - _S669 * _S670, (&kernelContext_97)->params_0->scale_j_0, &kernelContext_97);
    add64_0(at_9 + 16U, _S666.x - _S669 * _S672, (&kernelContext_97)->params_0->scale_j_0, &kernelContext_97);
    add64_0(at_9 + 18U, _S666.y - _S671 * _S672, (&kernelContext_97)->params_0->scale_j_0, &kernelContext_97);
    return;
}

bool is_root_0(uint c_8, KernelContext_0 thread* kernelContext_98)
{
    bool _S673;
    if(c_8 < (kernelContext_98->params_0->chunk_count_0))
    {
        uint _S674 = body_of_0(c_8, kernelContext_98);
        bool _S675 = dirty_0(_S674, kernelContext_98);
        _S673 = _S675;
    }
    else
    {
        _S673 = false;
    }
    if(_S673)
    {
        uint _S676 = w_label_0(kernelContext_98);
        _S673 = (*(kernelContext_98->work_0+(_S676 + c_8))) == c_8;
    }
    else
    {
        _S673 = false;
    }
    return _S673;
}

[[kernel]] void fr_best_count(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], packed_float4 device* lp_10 [[buffer(24)]], packed_float4 device* ext_11 [[buffer(23)]], packed_float4 device* engine_contacts_10 [[buffer(25)]], CrushParams_0 constant* crush_10 [[buffer(28)]], packed_float4 device* crush_rw_10 [[buffer(27)]], packed_float4 device* crush_tab_10 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_10;
    (&kernelContext_99)->work_0 = work_10;
    (&kernelContext_99)->frames_0 = frames_10;
    (&kernelContext_99)->bodies_0 = bodies_10;
    (&kernelContext_99)->chunks_0 = chunks_10;
    (&kernelContext_99)->sort_0 = sort_10;
    (&kernelContext_99)->atom_0 = atom_10;
    (&kernelContext_99)->manifolds_0 = manifolds_10;
    (&kernelContext_99)->loads_0 = loads_10;
    (&kernelContext_99)->bonds_0 = bonds_10;
    (&kernelContext_99)->args_0 = args_10;
    (&kernelContext_99)->info_0 = info_10;
    (&kernelContext_99)->scan_0 = scan_10;
    (&kernelContext_99)->parts_0 = parts_10;
    (&kernelContext_99)->colliders_0 = colliders_10;
    (&kernelContext_99)->collider_info_0 = collider_info_10;
    (&kernelContext_99)->view_chunks_0 = view_chunks_10;
    (&kernelContext_99)->view_0 = view_10;
    (&kernelContext_99)->objects_0 = objects_10;
    (&kernelContext_99)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_99)->lines_0 = lines_10;
    (&kernelContext_99)->geometry_0 = geometry_10;
    (&kernelContext_99)->shots_0 = shots_10;
    (&kernelContext_99)->lp_0 = lp_10;
    (&kernelContext_99)->ext_0 = ext_11;
    (&kernelContext_99)->engine_contacts_0 = engine_contacts_10;
    (&kernelContext_99)->crush_0 = crush_10;
    (&kernelContext_99)->crush_rw_0 = crush_rw_10;
    (&kernelContext_99)->crush_tab_0 = crush_tab_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_99)->g_scan_0 = &g_scan_10;
    threadgroup array<float, int(64)> g_il_best_10;
    (&kernelContext_99)->g_il_best_0 = &g_il_best_10;
    uint c_9 = id_9.x;
    bool _S677 = is_root_0(c_9, &kernelContext_99);
    if(!_S677)
    {
        return;
    }
    uint _S678 = a_best_count_0(&kernelContext_99);
    uint _S679 = body_of_0(c_9, &kernelContext_99);
    atomic<uint> device* _S680 = (&kernelContext_99)->atom_0+(_S678 + _S679);
    uint _S681 = a_count_0(&kernelContext_99);
    uint _S682 = atomic_load_explicit((&kernelContext_99)->atom_0+(_S681 + c_9), memory_order_relaxed);
    uint _S683 = atomic_fetch_max_explicit(_S680, _S682, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], packed_float4 device* lp_11 [[buffer(24)]], packed_float4 device* ext_12 [[buffer(23)]], packed_float4 device* engine_contacts_11 [[buffer(25)]], CrushParams_0 constant* crush_11 [[buffer(28)]], packed_float4 device* crush_rw_11 [[buffer(27)]], packed_float4 device* crush_tab_11 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_100;
    (&kernelContext_100)->params_0 = params_11;
    (&kernelContext_100)->work_0 = work_11;
    (&kernelContext_100)->frames_0 = frames_11;
    (&kernelContext_100)->bodies_0 = bodies_11;
    (&kernelContext_100)->chunks_0 = chunks_11;
    (&kernelContext_100)->sort_0 = sort_11;
    (&kernelContext_100)->atom_0 = atom_11;
    (&kernelContext_100)->manifolds_0 = manifolds_11;
    (&kernelContext_100)->loads_0 = loads_11;
    (&kernelContext_100)->bonds_0 = bonds_11;
    (&kernelContext_100)->args_0 = args_11;
    (&kernelContext_100)->info_0 = info_11;
    (&kernelContext_100)->scan_0 = scan_11;
    (&kernelContext_100)->parts_0 = parts_11;
    (&kernelContext_100)->colliders_0 = colliders_11;
    (&kernelContext_100)->collider_info_0 = collider_info_11;
    (&kernelContext_100)->view_chunks_0 = view_chunks_11;
    (&kernelContext_100)->view_0 = view_11;
    (&kernelContext_100)->objects_0 = objects_11;
    (&kernelContext_100)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_100)->lines_0 = lines_11;
    (&kernelContext_100)->geometry_0 = geometry_11;
    (&kernelContext_100)->shots_0 = shots_11;
    (&kernelContext_100)->lp_0 = lp_11;
    (&kernelContext_100)->ext_0 = ext_12;
    (&kernelContext_100)->engine_contacts_0 = engine_contacts_11;
    (&kernelContext_100)->crush_0 = crush_11;
    (&kernelContext_100)->crush_rw_0 = crush_rw_11;
    (&kernelContext_100)->crush_tab_0 = crush_tab_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_100)->g_scan_0 = &g_scan_11;
    threadgroup array<float, int(64)> g_il_best_11;
    (&kernelContext_100)->g_il_best_0 = &g_il_best_11;
    uint c_10 = id_10.x;
    bool _S684 = is_root_0(c_10, &kernelContext_100);
    if(!_S684)
    {
        return;
    }
    uint _S685 = body_of_0(c_10, &kernelContext_100);
    uint _S686 = a_count_0(&kernelContext_100);
    uint _S687 = atomic_load_explicit((&kernelContext_100)->atom_0+(_S686 + c_10), memory_order_relaxed);
    uint _S688 = a_best_count_0(&kernelContext_100);
    uint _S689 = atomic_load_explicit((&kernelContext_100)->atom_0+(_S688 + _S685), memory_order_relaxed);
    if(_S687 == _S689)
    {
        uint _S690 = a_best_root_0(&kernelContext_100);
        uint _S691 = atomic_fetch_min_explicit((&kernelContext_100)->atom_0+(_S690 + _S685), c_10, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], packed_float4 device* lp_12 [[buffer(24)]], packed_float4 device* ext_13 [[buffer(23)]], packed_float4 device* engine_contacts_12 [[buffer(25)]], CrushParams_0 constant* crush_12 [[buffer(28)]], packed_float4 device* crush_rw_12 [[buffer(27)]], packed_float4 device* crush_tab_12 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_101;
    (&kernelContext_101)->params_0 = params_12;
    (&kernelContext_101)->work_0 = work_12;
    (&kernelContext_101)->frames_0 = frames_12;
    (&kernelContext_101)->bodies_0 = bodies_12;
    (&kernelContext_101)->chunks_0 = chunks_12;
    (&kernelContext_101)->sort_0 = sort_12;
    (&kernelContext_101)->atom_0 = atom_12;
    (&kernelContext_101)->manifolds_0 = manifolds_12;
    (&kernelContext_101)->loads_0 = loads_12;
    (&kernelContext_101)->bonds_0 = bonds_12;
    (&kernelContext_101)->args_0 = args_12;
    (&kernelContext_101)->info_0 = info_12;
    (&kernelContext_101)->scan_0 = scan_12;
    (&kernelContext_101)->parts_0 = parts_12;
    (&kernelContext_101)->colliders_0 = colliders_12;
    (&kernelContext_101)->collider_info_0 = collider_info_12;
    (&kernelContext_101)->view_chunks_0 = view_chunks_12;
    (&kernelContext_101)->view_0 = view_12;
    (&kernelContext_101)->objects_0 = objects_12;
    (&kernelContext_101)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_101)->lines_0 = lines_12;
    (&kernelContext_101)->geometry_0 = geometry_12;
    (&kernelContext_101)->shots_0 = shots_12;
    (&kernelContext_101)->lp_0 = lp_12;
    (&kernelContext_101)->ext_0 = ext_13;
    (&kernelContext_101)->engine_contacts_0 = engine_contacts_12;
    (&kernelContext_101)->crush_0 = crush_12;
    (&kernelContext_101)->crush_rw_0 = crush_rw_12;
    (&kernelContext_101)->crush_tab_0 = crush_tab_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_101)->g_scan_0 = &g_scan_12;
    threadgroup array<float, int(64)> g_il_best_12;
    (&kernelContext_101)->g_il_best_0 = &g_il_best_12;
    uint c_11 = id_11.x;
    if(c_11 >= (params_12->chunk_count_0))
    {
        return;
    }
    uint _S692 = w_new_0(&kernelContext_101);
    uint device* _S693 = (&kernelContext_101)->work_0+(_S692 + c_11);
    bool _S694 = is_root_0(c_11, &kernelContext_101);
    bool _S695;
    if(_S694)
    {
        uint _S696 = a_best_root_0(&kernelContext_101);
        uint _S697 = body_of_0(c_11, &kernelContext_101);
        uint _S698 = atomic_load_explicit((&kernelContext_101)->atom_0+(_S696 + _S697), memory_order_relaxed);
        _S695 = _S698 != c_11;
    }
    else
    {
        _S695 = false;
    }
    uint _S699;
    if(_S695)
    {
        _S699 = 1U;
    }
    else
    {
        _S699 = 0U;
    }
    *_S693 = _S699;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], packed_float4 device* lp_13 [[buffer(24)]], packed_float4 device* ext_14 [[buffer(23)]], packed_float4 device* engine_contacts_13 [[buffer(25)]], CrushParams_0 constant* crush_13 [[buffer(28)]], packed_float4 device* crush_rw_13 [[buffer(27)]], packed_float4 device* crush_tab_13 [[buffer(26)]])
{
    uint step_4;
    uint add_0;
    thread KernelContext_0 kernelContext_102;
    (&kernelContext_102)->params_0 = params_13;
    (&kernelContext_102)->work_0 = work_13;
    (&kernelContext_102)->frames_0 = frames_13;
    (&kernelContext_102)->bodies_0 = bodies_13;
    (&kernelContext_102)->chunks_0 = chunks_13;
    (&kernelContext_102)->sort_0 = sort_13;
    (&kernelContext_102)->atom_0 = atom_13;
    (&kernelContext_102)->manifolds_0 = manifolds_13;
    (&kernelContext_102)->loads_0 = loads_13;
    (&kernelContext_102)->bonds_0 = bonds_13;
    (&kernelContext_102)->args_0 = args_13;
    (&kernelContext_102)->info_0 = info_13;
    (&kernelContext_102)->scan_0 = scan_13;
    (&kernelContext_102)->parts_0 = parts_13;
    (&kernelContext_102)->colliders_0 = colliders_13;
    (&kernelContext_102)->collider_info_0 = collider_info_13;
    (&kernelContext_102)->view_chunks_0 = view_chunks_13;
    (&kernelContext_102)->view_0 = view_13;
    (&kernelContext_102)->objects_0 = objects_13;
    (&kernelContext_102)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_102)->lines_0 = lines_13;
    (&kernelContext_102)->geometry_0 = geometry_13;
    (&kernelContext_102)->shots_0 = shots_13;
    (&kernelContext_102)->lp_0 = lp_13;
    (&kernelContext_102)->ext_0 = ext_14;
    (&kernelContext_102)->engine_contacts_0 = engine_contacts_13;
    (&kernelContext_102)->crush_0 = crush_13;
    (&kernelContext_102)->crush_rw_0 = crush_rw_13;
    (&kernelContext_102)->crush_tab_0 = crush_tab_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_102)->g_scan_0 = &g_scan_13;
    threadgroup array<float, int(64)> g_il_best_13;
    (&kernelContext_102)->g_il_best_0 = &g_il_best_13;
    uint t_7 = thread_2.x;
    uint _S700 = group_0.x;
    uint _S701 = 1024U * _S700 + 4U * t_7;
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
        if((_S701 + j_5) < ((&kernelContext_102)->scan_0->n_0))
        {
            step_4 = *((&kernelContext_102)->work_0+((&kernelContext_102)->scan_0->in_base_0 + _S701 + j_5));
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
    (*(&kernelContext_102)->g_scan_0)[t_7] = sum_2;
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
            add_0 = (*(&kernelContext_102)->g_scan_0)[t_7 - step_4];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_102)->g_scan_0)[t_7] = (*(&kernelContext_102)->g_scan_0)[t_7] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_4 = step_4 << 1U;
    }
    uint _S702 = (*(&kernelContext_102)->g_scan_0)[t_7] - sum_2;
    j_5 = 0U;
    add_0 = _S702;
    for(;;)
    {
        if(j_5 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S701 + j_5) < ((&kernelContext_102)->scan_0->n_0))
        {
            *((&kernelContext_102)->work_0+((&kernelContext_102)->scan_0->out_base_0 + _S701 + j_5)) = add_0;
        }
        uint run_1 = add_0 + v_10[j_5];
        j_5 = j_5 + 1U;
        add_0 = run_1;
    }
    if(t_7 == 255U)
    {
        *((&kernelContext_102)->work_0+((&kernelContext_102)->scan_0->block_base_0 + _S700)) = (*(&kernelContext_102)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_3 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], packed_float4 device* lp_14 [[buffer(24)]], packed_float4 device* ext_15 [[buffer(23)]], packed_float4 device* engine_contacts_14 [[buffer(25)]], CrushParams_0 constant* crush_14 [[buffer(28)]], packed_float4 device* crush_rw_14 [[buffer(27)]], packed_float4 device* crush_tab_14 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_103;
    (&kernelContext_103)->params_0 = params_14;
    (&kernelContext_103)->work_0 = work_14;
    (&kernelContext_103)->frames_0 = frames_14;
    (&kernelContext_103)->bodies_0 = bodies_14;
    (&kernelContext_103)->chunks_0 = chunks_14;
    (&kernelContext_103)->sort_0 = sort_14;
    (&kernelContext_103)->atom_0 = atom_14;
    (&kernelContext_103)->manifolds_0 = manifolds_14;
    (&kernelContext_103)->loads_0 = loads_14;
    (&kernelContext_103)->bonds_0 = bonds_14;
    (&kernelContext_103)->args_0 = args_14;
    (&kernelContext_103)->info_0 = info_14;
    (&kernelContext_103)->scan_0 = scan_14;
    (&kernelContext_103)->parts_0 = parts_14;
    (&kernelContext_103)->colliders_0 = colliders_14;
    (&kernelContext_103)->collider_info_0 = collider_info_14;
    (&kernelContext_103)->view_chunks_0 = view_chunks_14;
    (&kernelContext_103)->view_0 = view_14;
    (&kernelContext_103)->objects_0 = objects_14;
    (&kernelContext_103)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_103)->lines_0 = lines_14;
    (&kernelContext_103)->geometry_0 = geometry_14;
    (&kernelContext_103)->shots_0 = shots_14;
    (&kernelContext_103)->lp_0 = lp_14;
    (&kernelContext_103)->ext_0 = ext_15;
    (&kernelContext_103)->engine_contacts_0 = engine_contacts_14;
    (&kernelContext_103)->crush_0 = crush_14;
    (&kernelContext_103)->crush_rw_0 = crush_rw_14;
    (&kernelContext_103)->crush_tab_0 = crush_tab_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_103)->g_scan_0 = &g_scan_14;
    threadgroup array<float, int(64)> g_il_best_14;
    (&kernelContext_103)->g_il_best_0 = &g_il_best_14;
    uint t_8 = thread_3.x;
    uint blocks_0 = (scan_14->n_0 + 1023U) / 1024U;
    bool _S703 = t_8 < blocks_0;
    uint v_11;
    if(_S703)
    {
        v_11 = *((&kernelContext_103)->work_0+((&kernelContext_103)->scan_0->block_base_0 + t_8));
    }
    else
    {
        v_11 = 0U;
    }
    (*(&kernelContext_103)->g_scan_0)[t_8] = v_11;
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
            add_1 = (*(&kernelContext_103)->g_scan_0)[t_8 - step_5];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_103)->g_scan_0)[t_8] = (*(&kernelContext_103)->g_scan_0)[t_8] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_5 = step_5 << 1U;
    }
    if(_S703)
    {
        *((&kernelContext_103)->work_0+((&kernelContext_103)->scan_0->block_base_0 + t_8)) = (*(&kernelContext_103)->g_scan_0)[t_8] - v_11;
    }
    if(t_8 == 1023U)
    {
        *((&kernelContext_103)->work_0+((&kernelContext_103)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_103)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], packed_float4 device* lp_15 [[buffer(24)]], packed_float4 device* ext_16 [[buffer(23)]], packed_float4 device* engine_contacts_15 [[buffer(25)]], CrushParams_0 constant* crush_15 [[buffer(28)]], packed_float4 device* crush_rw_15 [[buffer(27)]], packed_float4 device* crush_tab_15 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_104;
    (&kernelContext_104)->params_0 = params_15;
    (&kernelContext_104)->work_0 = work_15;
    (&kernelContext_104)->frames_0 = frames_15;
    (&kernelContext_104)->bodies_0 = bodies_15;
    (&kernelContext_104)->chunks_0 = chunks_15;
    (&kernelContext_104)->sort_0 = sort_15;
    (&kernelContext_104)->atom_0 = atom_15;
    (&kernelContext_104)->manifolds_0 = manifolds_15;
    (&kernelContext_104)->loads_0 = loads_15;
    (&kernelContext_104)->bonds_0 = bonds_15;
    (&kernelContext_104)->args_0 = args_15;
    (&kernelContext_104)->info_0 = info_15;
    (&kernelContext_104)->scan_0 = scan_15;
    (&kernelContext_104)->parts_0 = parts_15;
    (&kernelContext_104)->colliders_0 = colliders_15;
    (&kernelContext_104)->collider_info_0 = collider_info_15;
    (&kernelContext_104)->view_chunks_0 = view_chunks_15;
    (&kernelContext_104)->view_0 = view_15;
    (&kernelContext_104)->objects_0 = objects_15;
    (&kernelContext_104)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_104)->lines_0 = lines_15;
    (&kernelContext_104)->geometry_0 = geometry_15;
    (&kernelContext_104)->shots_0 = shots_15;
    (&kernelContext_104)->lp_0 = lp_15;
    (&kernelContext_104)->ext_0 = ext_16;
    (&kernelContext_104)->engine_contacts_0 = engine_contacts_15;
    (&kernelContext_104)->crush_0 = crush_15;
    (&kernelContext_104)->crush_rw_0 = crush_rw_15;
    (&kernelContext_104)->crush_tab_0 = crush_tab_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_104)->g_scan_0 = &g_scan_15;
    threadgroup array<float, int(64)> g_il_best_15;
    (&kernelContext_104)->g_il_best_0 = &g_il_best_15;
    uint i_9 = id_12.x;
    if(i_9 >= (scan_15->n_0))
    {
        return;
    }
    uint device* _S704 = (&kernelContext_104)->work_0+((&kernelContext_104)->scan_0->out_base_0 + i_9);
    *_S704 = *_S704 + *((&kernelContext_104)->work_0+((&kernelContext_104)->scan_0->block_base_0 + i_9 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], packed_float4 device* lp_16 [[buffer(24)]], packed_float4 device* ext_17 [[buffer(23)]], packed_float4 device* engine_contacts_16 [[buffer(25)]], CrushParams_0 constant* crush_16 [[buffer(28)]], packed_float4 device* crush_rw_16 [[buffer(27)]], packed_float4 device* crush_tab_16 [[buffer(26)]])
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
    (&kernelContext_105)->engine_contacts_0 = engine_contacts_16;
    (&kernelContext_105)->crush_0 = crush_16;
    (&kernelContext_105)->crush_rw_0 = crush_rw_16;
    (&kernelContext_105)->crush_tab_0 = crush_tab_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_105)->g_scan_0 = &g_scan_16;
    threadgroup array<float, int(64)> g_il_best_16;
    (&kernelContext_105)->g_il_best_0 = &g_il_best_16;
    uint s_8 = id_13.x;
    bool _S705;
    if(s_8 >= (params_16->body_count_0))
    {
        _S705 = true;
    }
    else
    {
        uint _S706 = w_free_0(&kernelContext_105);
        _S705 = (*((&kernelContext_105)->work_0+(_S706 + s_8))) == 0U;
    }
    if(_S705)
    {
        return;
    }
    uint _S707 = w_free_list_0(&kernelContext_105);
    uint _S708 = w_free_rank_0(&kernelContext_105);
    *((&kernelContext_105)->work_0+(_S707 + *((&kernelContext_105)->work_0+(_S708 + s_8)))) = s_8;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_106)
{
    uint _S709 = w_free_list_0(kernelContext_106);
    return _S709 + kernelContext_106->params_0->body_count_0;
}

uint scan_new_blocks_1(KernelContext_0 thread* kernelContext_107)
{
    uint _S710 = w_free_list_0(kernelContext_107);
    return _S710 + kernelContext_107->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_108)
{
    uint _S711 = scan_new_blocks_1(kernelContext_108);
    return _S711 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_109)
{
    uint blocks_1 = (kernelContext_109->params_0->body_count_0 + 1023U) / 1024U;
    uint _S712 = scan_free_blocks_0(kernelContext_109);
    return *(kernelContext_109->work_0+(_S712 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], packed_float4 device* lp_17 [[buffer(24)]], packed_float4 device* ext_18 [[buffer(23)]], packed_float4 device* engine_contacts_17 [[buffer(25)]], CrushParams_0 constant* crush_17 [[buffer(28)]], packed_float4 device* crush_rw_17 [[buffer(27)]], packed_float4 device* crush_tab_17 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_110;
    (&kernelContext_110)->params_0 = params_17;
    (&kernelContext_110)->work_0 = work_17;
    (&kernelContext_110)->frames_0 = frames_17;
    (&kernelContext_110)->bodies_0 = bodies_17;
    (&kernelContext_110)->chunks_0 = chunks_17;
    (&kernelContext_110)->sort_0 = sort_17;
    (&kernelContext_110)->atom_0 = atom_17;
    (&kernelContext_110)->manifolds_0 = manifolds_17;
    (&kernelContext_110)->loads_0 = loads_17;
    (&kernelContext_110)->bonds_0 = bonds_17;
    (&kernelContext_110)->args_0 = args_17;
    (&kernelContext_110)->info_0 = info_17;
    (&kernelContext_110)->scan_0 = scan_17;
    (&kernelContext_110)->parts_0 = parts_17;
    (&kernelContext_110)->colliders_0 = colliders_17;
    (&kernelContext_110)->collider_info_0 = collider_info_17;
    (&kernelContext_110)->view_chunks_0 = view_chunks_17;
    (&kernelContext_110)->view_0 = view_17;
    (&kernelContext_110)->objects_0 = objects_17;
    (&kernelContext_110)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_110)->lines_0 = lines_17;
    (&kernelContext_110)->geometry_0 = geometry_17;
    (&kernelContext_110)->shots_0 = shots_17;
    (&kernelContext_110)->lp_0 = lp_17;
    (&kernelContext_110)->ext_0 = ext_18;
    (&kernelContext_110)->engine_contacts_0 = engine_contacts_17;
    (&kernelContext_110)->crush_0 = crush_17;
    (&kernelContext_110)->crush_rw_0 = crush_rw_17;
    (&kernelContext_110)->crush_tab_0 = crush_tab_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_110)->g_scan_0 = &g_scan_17;
    threadgroup array<float, int(64)> g_il_best_17;
    (&kernelContext_110)->g_il_best_0 = &g_il_best_17;
    uint c_12 = id_14.x;
    bool _S713 = is_root_0(c_12, &kernelContext_110);
    if(!_S713)
    {
        return;
    }
    uint _S714 = body_of_0(c_12, &kernelContext_110);
    uint _S715 = a_best_root_0(&kernelContext_110);
    uint _S716 = atomic_load_explicit((&kernelContext_110)->atom_0+(_S715 + _S714), memory_order_relaxed);
    uint slot_1;
    if(_S716 != c_12)
    {
        uint _S717 = w_rank_0(&kernelContext_110);
        uint device* _S718 = (&kernelContext_110)->work_0+(_S717 + c_12);
        uint k_23 = *_S718;
        uint _S719 = free_total_0(&kernelContext_110);
        if((*_S718) < _S719)
        {
            uint _S720 = w_free_list_0(&kernelContext_110);
            uint device* _S721 = (&kernelContext_110)->work_0+(_S720 + k_23);
            uint slot_2 = *_S721;
            uint _S722 = w_free_0(&kernelContext_110);
            *((&kernelContext_110)->work_0+(_S722 + *_S721)) = 0U;
            uint _S723 = a_counters_1(&kernelContext_110);
            uint _S724 = atomic_fetch_add_explicit((&kernelContext_110)->atom_0+(_S723 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S725 = a_counters_1(&kernelContext_110);
            uint _S726 = atomic_fetch_or_explicit((&kernelContext_110)->atom_0+(_S725 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S714;
        }
    }
    else
    {
        slot_1 = _S714;
    }
    uint _S727 = w_slot_0(&kernelContext_110);
    *((&kernelContext_110)->work_0+(_S727 + c_12)) = slot_1;
    return;
}

float read64_1(uint at_10, float scale_2, KernelContext_0 thread* kernelContext_111)
{
    uint _S728 = atomic_load_explicit(kernelContext_111->atom_0+(at_10 + 1U), memory_order_relaxed);
    int hi_3 = (as_type<int>((_S728)));
    uint lo_3 = atomic_load_explicit(kernelContext_111->atom_0+at_10, memory_order_relaxed);
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
    array<float, int(3)> _S729 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S730 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S731 = { { r2_2.x, r2_2.y, r2_2.z } };
    thread array<array<float, int(3)>, int(3)> a_7;
    a_7[int(0)] = _S729;
    a_7[int(1)] = _S730;
    a_7[int(2)] = _S731;
    array<float, int(3)> _S732 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S733 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S734 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_12;
    v_12[int(0)] = _S732;
    v_12[int(1)] = _S733;
    v_12[int(2)] = _S734;
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
            bool _S735;
            if(k_24 != p_12)
            {
                _S735 = k_24 != q_14;
            }
            else
            {
                _S735 = false;
            }
            if(_S735)
            {
                float kp_0 = a_7[k_24][p_12];
                float kq_0 = a_7[k_24][q_14];
                float _S736 = c_13 * a_7[k_24][p_12] - s_9 * a_7[k_24][q_14];
                a_7[k_24][p_12] = _S736;
                a_7[p_12][k_24] = _S736;
                float _S737 = s_9 * kp_0 + c_13 * kq_0;
                a_7[k_24][q_14] = _S737;
                a_7[q_14][k_24] = _S737;
            }
            k_24 = k_24 + 1U;
        }
        float _S738 = c_13 * c_13;
        float _S739 = 2.0f * s_9 * c_13 * apq_0;
        float _S740 = s_9 * s_9;
        a_7[p_12][p_12] = _S738 * app_0 - _S739 + _S740 * aqq_0;
        a_7[q_14][q_14] = _S740 * app_0 + _S739 + _S738 * aqq_0;
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
    thread Quat_0 _S741 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S742 = normalized_0(&_S741);
    *rot_0 = _S742;
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
    thread Quat_0 _S743 = axes_1;
    Quat_0 _S744 = quat_mul_0(&_S743, q_15);
    thread Quat_0 _S745 = _S744;
    Quat_0 _S746 = normalized_0(&_S745);
    return _S746;
}

uint stage_0(uint c_14, KernelContext_0 thread* kernelContext_112)
{
    return 2U * kernelContext_112->params_0->body_count_0 + 9U * c_14;
}

float4 quat_vec_0(const Quat_0 thread* q_16)
{
    return float4(q_16->x_0, q_16->y_0, q_16->z_0, q_16->w_0);
}

uint frame_step_0(KernelContext_0 thread* kernelContext_113)
{
    uint _S747 = w_frame_0(kernelContext_113);
    return *(kernelContext_113->work_0+(_S747 + 1U));
}

[[kernel]] void fr_piece(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], packed_float4 device* lp_18 [[buffer(24)]], packed_float4 device* ext_19 [[buffer(23)]], packed_float4 device* engine_contacts_18 [[buffer(25)]], CrushParams_0 constant* crush_18 [[buffer(28)]], packed_float4 device* crush_rw_18 [[buffer(27)]], packed_float4 device* crush_tab_18 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_114;
    (&kernelContext_114)->params_0 = params_18;
    (&kernelContext_114)->work_0 = work_18;
    (&kernelContext_114)->frames_0 = frames_18;
    (&kernelContext_114)->bodies_0 = bodies_18;
    (&kernelContext_114)->chunks_0 = chunks_18;
    (&kernelContext_114)->sort_0 = sort_18;
    (&kernelContext_114)->atom_0 = atom_18;
    (&kernelContext_114)->manifolds_0 = manifolds_18;
    (&kernelContext_114)->loads_0 = loads_18;
    (&kernelContext_114)->bonds_0 = bonds_18;
    (&kernelContext_114)->args_0 = args_18;
    (&kernelContext_114)->info_0 = info_18;
    (&kernelContext_114)->scan_0 = scan_18;
    (&kernelContext_114)->parts_0 = parts_18;
    (&kernelContext_114)->colliders_0 = colliders_18;
    (&kernelContext_114)->collider_info_0 = collider_info_18;
    (&kernelContext_114)->view_chunks_0 = view_chunks_18;
    (&kernelContext_114)->view_0 = view_18;
    (&kernelContext_114)->objects_0 = objects_18;
    (&kernelContext_114)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_114)->lines_0 = lines_18;
    (&kernelContext_114)->geometry_0 = geometry_18;
    (&kernelContext_114)->shots_0 = shots_18;
    (&kernelContext_114)->lp_0 = lp_18;
    (&kernelContext_114)->ext_0 = ext_19;
    (&kernelContext_114)->engine_contacts_0 = engine_contacts_18;
    (&kernelContext_114)->crush_0 = crush_18;
    (&kernelContext_114)->crush_rw_0 = crush_rw_18;
    (&kernelContext_114)->crush_tab_0 = crush_tab_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_114)->g_scan_0 = &g_scan_18;
    threadgroup array<float, int(64)> g_il_best_18;
    (&kernelContext_114)->g_il_best_0 = &g_il_best_18;
    uint c_15 = id_15.x;
    bool _S748 = is_root_0(c_15, &kernelContext_114);
    if(!_S748)
    {
        return;
    }
    uint _S749 = body_of_0(c_15, &kernelContext_114);
    uint pb_0 = 7U * _S749;
    uint _S750 = a_sums_0(&kernelContext_114);
    uint at_11 = _S750 + 20U * c_15;
    float _S751 = read64_0(at_11, (&kernelContext_114)->params_0->scale_m_0, &kernelContext_114);
    float _S752 = read64_0(at_11 + 2U, (&kernelContext_114)->params_0->scale_s_0, &kernelContext_114);
    float _S753 = read64_0(at_11 + 4U, (&kernelContext_114)->params_0->scale_s_0, &kernelContext_114);
    float _S754 = read64_0(at_11 + 6U, (&kernelContext_114)->params_0->scale_s_0, &kernelContext_114);
    float3 d_19 = float3(_S752, _S753, _S754) / float3(max(_S751, 1.00000000317107685e-30f)) ;
    float _S755 = read64_0(at_11 + 8U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float _S756 = read64_0(at_11 + 10U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float _S757 = read64_0(at_11 + 12U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float _S758 = read64_0(at_11 + 14U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float _S759 = read64_0(at_11 + 16U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float _S760 = read64_0(at_11 + 18U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float dd_1 = dot(d_19, d_19);
    float _S761 = d_19.x;
    float _S762 = _S751 * _S761;
    float _S763 = d_19.y;
    float _S764 = _S758 + _S762 * _S763;
    float _S765 = d_19.z;
    float _S766 = _S759 + _S762 * _S765;
    float _S767 = _S760 + _S751 * _S763 * _S765;
    float3 com_1 = (float4(*((&kernelContext_114)->chunks_0+4U * c_15)) ).xyz + d_19;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S755 - _S751 * (dd_1 - _S761 * _S761), _S764, _S766), float3(_S764, _S756 - _S751 * (dd_1 - _S763 * _S763), _S767), float3(_S766, _S767, _S757 - _S751 * (dd_1 - _S765 * _S765)), &principal_0, &moments_1);
    uint _S768 = a_anchor_0(&kernelContext_114);
    uint _S769 = atomic_load_explicit((&kernelContext_114)->atom_0+(_S768 + c_15), memory_order_relaxed);
    bool anchored_0 = _S769 != 0U;
    thread Quat_0 q_17;
    thread float3 origin_4;
    structure_pose_0(_S749, pb_0, &q_17, &origin_4, &kernelContext_114);
    float3 p_parent_0 = (float4(*((&kernelContext_114)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_114)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_114)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S770 = q_17;
    float3 _S771 = rotate_0(&_S770, com_1);
    float3 pos_0 = to_rubble_0(origin_4 + _S771);
    thread Quat_0 _S772 = q_17;
    thread Quat_0 _S773 = principal_0;
    Quat_0 _S774 = quat_mul_0(&_S772, &_S773);
    thread Quat_0 _S775 = _S774;
    Quat_0 _S776 = normalized_0(&_S775);
    thread Quat_0 _S777 = _S776;
    Quat_0 _S778 = rot_to_rubble_0(&_S777);
    bool _S779;
    if(anchored_0)
    {
        _S779 = true;
    }
    else
    {
        _S779 = !(_S751 > 0.0f);
    }
    float inv_mass_3;
    if(_S779)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / _S751;
    }
    float3 _S780 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S781;
        if((moments_1.x) > 0.0f)
        {
            _S781 = 1.0f / moments_1.x;
        }
        else
        {
            _S781 = 0.0f;
        }
        float _S782;
        if((moments_1.y) > 0.0f)
        {
            _S782 = 1.0f / moments_1.y;
        }
        else
        {
            _S782 = 0.0f;
        }
        float _S783;
        if((moments_1.z) > 0.0f)
        {
            _S783 = 1.0f / moments_1.z;
        }
        else
        {
            _S783 = 0.0f;
        }
        inv_inertia_0 = float3(_S781, _S782, _S783);
    }
    else
    {
        inv_inertia_0 = _S780;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S780;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S780;
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
    uint _S784 = stage_0(c_15, &kernelContext_114);
    *((&kernelContext_114)->frames_0+_S784) = packed_float4(float4(pos_0, inv_mass_3)) ;
    packed_float4 device* _S785 = (&kernelContext_114)->frames_0+(_S784 + 1U);
    thread Quat_0 _S786 = _S778;
    float4 _S787 = quat_vec_0(&_S786);
    *_S785 = packed_float4(_S787) ;
    *((&kernelContext_114)->frames_0+(_S784 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_114)->frames_0+(_S784 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    packed_float4 device* _S788 = (&kernelContext_114)->frames_0+(_S784 + 4U);
    uint _S789 = frame_step_0(&kernelContext_114);
    *_S788 = packed_float4(float4(spin_0, (as_type<float>((_S789))))) ;
    *((&kernelContext_114)->frames_0+(_S784 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_114)->frames_0+(_S784 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_114)->frames_0+(_S784 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_114)->frames_0+2U * _S749)) ).w)) ;
    packed_float4 device* _S790 = (&kernelContext_114)->frames_0+(_S784 + 8U);
    thread Quat_0 _S791 = principal_0;
    float4 _S792 = quat_vec_0(&_S791);
    *_S790 = packed_float4(_S792) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], packed_float4 device* lp_19 [[buffer(24)]], packed_float4 device* ext_20 [[buffer(23)]], packed_float4 device* engine_contacts_19 [[buffer(25)]], CrushParams_0 constant* crush_19 [[buffer(28)]], packed_float4 device* crush_rw_19 [[buffer(27)]], packed_float4 device* crush_tab_19 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_115;
    (&kernelContext_115)->params_0 = params_19;
    (&kernelContext_115)->work_0 = work_19;
    (&kernelContext_115)->frames_0 = frames_19;
    (&kernelContext_115)->bodies_0 = bodies_19;
    (&kernelContext_115)->chunks_0 = chunks_19;
    (&kernelContext_115)->sort_0 = sort_19;
    (&kernelContext_115)->atom_0 = atom_19;
    (&kernelContext_115)->manifolds_0 = manifolds_19;
    (&kernelContext_115)->loads_0 = loads_19;
    (&kernelContext_115)->bonds_0 = bonds_19;
    (&kernelContext_115)->args_0 = args_19;
    (&kernelContext_115)->info_0 = info_19;
    (&kernelContext_115)->scan_0 = scan_19;
    (&kernelContext_115)->parts_0 = parts_19;
    (&kernelContext_115)->colliders_0 = colliders_19;
    (&kernelContext_115)->collider_info_0 = collider_info_19;
    (&kernelContext_115)->view_chunks_0 = view_chunks_19;
    (&kernelContext_115)->view_0 = view_19;
    (&kernelContext_115)->objects_0 = objects_19;
    (&kernelContext_115)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_115)->lines_0 = lines_19;
    (&kernelContext_115)->geometry_0 = geometry_19;
    (&kernelContext_115)->shots_0 = shots_19;
    (&kernelContext_115)->lp_0 = lp_19;
    (&kernelContext_115)->ext_0 = ext_20;
    (&kernelContext_115)->engine_contacts_0 = engine_contacts_19;
    (&kernelContext_115)->crush_0 = crush_19;
    (&kernelContext_115)->crush_rw_0 = crush_rw_19;
    (&kernelContext_115)->crush_tab_0 = crush_tab_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_115)->g_scan_0 = &g_scan_19;
    threadgroup array<float, int(64)> g_il_best_19;
    (&kernelContext_115)->g_il_best_0 = &g_il_best_19;
    uint c_16 = id_16.x;
    bool _S793 = is_root_0(c_16, &kernelContext_115);
    if(!_S793)
    {
        return;
    }
    uint _S794 = w_slot_0(&kernelContext_115);
    uint slot_3 = *((&kernelContext_115)->work_0+(_S794 + c_16));
    uint _S795 = stage_0(c_16, &kernelContext_115);
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
            *((&kernelContext_115)->bodies_0+(7U * (slot_3 + region_5 * (&kernelContext_115)->params_0->max_bodies_0) + w_4)) = packed_float4(float4(*((&kernelContext_115)->frames_0+(_S795 + w_4))) ) ;
            w_4 = w_4 + 1U;
        }
        region_5 = region_5 + 1U;
    }
    uint _S796 = 2U * slot_3;
    *((&kernelContext_115)->frames_0+_S796) = packed_float4(float4(*((&kernelContext_115)->frames_0+(_S795 + 7U))) ) ;
    *((&kernelContext_115)->frames_0+(_S796 + 1U)) = packed_float4(float4(*((&kernelContext_115)->frames_0+(_S795 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], packed_float4 device* lp_20 [[buffer(24)]], packed_float4 device* ext_21 [[buffer(23)]], packed_float4 device* engine_contacts_20 [[buffer(25)]], CrushParams_0 constant* crush_20 [[buffer(28)]], packed_float4 device* crush_rw_20 [[buffer(27)]], packed_float4 device* crush_tab_20 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_116;
    (&kernelContext_116)->params_0 = params_20;
    (&kernelContext_116)->work_0 = work_20;
    (&kernelContext_116)->frames_0 = frames_20;
    (&kernelContext_116)->bodies_0 = bodies_20;
    (&kernelContext_116)->chunks_0 = chunks_20;
    (&kernelContext_116)->sort_0 = sort_20;
    (&kernelContext_116)->atom_0 = atom_20;
    (&kernelContext_116)->manifolds_0 = manifolds_20;
    (&kernelContext_116)->loads_0 = loads_20;
    (&kernelContext_116)->bonds_0 = bonds_20;
    (&kernelContext_116)->args_0 = args_20;
    (&kernelContext_116)->info_0 = info_20;
    (&kernelContext_116)->scan_0 = scan_20;
    (&kernelContext_116)->parts_0 = parts_20;
    (&kernelContext_116)->colliders_0 = colliders_20;
    (&kernelContext_116)->collider_info_0 = collider_info_20;
    (&kernelContext_116)->view_chunks_0 = view_chunks_20;
    (&kernelContext_116)->view_0 = view_20;
    (&kernelContext_116)->objects_0 = objects_20;
    (&kernelContext_116)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_116)->lines_0 = lines_20;
    (&kernelContext_116)->geometry_0 = geometry_20;
    (&kernelContext_116)->shots_0 = shots_20;
    (&kernelContext_116)->lp_0 = lp_20;
    (&kernelContext_116)->ext_0 = ext_21;
    (&kernelContext_116)->engine_contacts_0 = engine_contacts_20;
    (&kernelContext_116)->crush_0 = crush_20;
    (&kernelContext_116)->crush_rw_0 = crush_rw_20;
    (&kernelContext_116)->crush_tab_0 = crush_tab_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_116)->g_scan_0 = &g_scan_20;
    threadgroup array<float, int(64)> g_il_best_20;
    (&kernelContext_116)->g_il_best_0 = &g_il_best_20;
    uint c_17 = id_17.x;
    bool _S797;
    if(c_17 >= (params_20->chunk_count_0))
    {
        _S797 = true;
    }
    else
    {
        uint _S798 = body_of_0(c_17, &kernelContext_116);
        bool _S799 = dirty_0(_S798, &kernelContext_116);
        _S797 = !_S799;
    }
    if(_S797)
    {
        return;
    }
    uint _S800 = w_label_0(&kernelContext_116);
    uint device* _S801 = (&kernelContext_116)->work_0+(_S800 + c_17);
    uint r_21 = *_S801;
    uint _S802 = w_slot_0(&kernelContext_116);
    uint device* _S803 = (&kernelContext_116)->work_0+(_S802 + *_S801);
    uint slot_4 = *_S803;
    *((&kernelContext_116)->work_0+(w_body_0() + c_17)) = *_S803;
    uint _S804 = 2U * slot_4;
    float3 _S805 = (float4(*((&kernelContext_116)->frames_0+_S804)) ).xyz;
    thread Quat_0 _S806 = quat_of_0(float4(*((&kernelContext_116)->frames_0+(_S804 + 1U))) );
    Quat_0 _S807 = conj_0(&_S806);
    float inv_mass_4 = (float4(*((&kernelContext_116)->bodies_0+7U * slot_4)) ).w;
    float _S808;
    if(inv_mass_4 > 0.0f)
    {
        _S808 = 1.0f / inv_mass_4;
    }
    else
    {
        _S808 = float_bits_0(2139095040U);
    }
    float4 _S809 = float4(*((&kernelContext_116)->chunks_0+(4U * c_17 + 2U))) ;
    uint first_1 = (as_type<uint>((_S809.z)));
    uint _S810 = (as_type<uint>((_S809.w)));
    float4 _S811 = float4(*((&kernelContext_116)->chunks_0+(4U * r_21 + 3U))) ;
    uint p_13 = first_1;
    for(;;)
    {
        if(p_13 < (first_1 + _S810))
        {
        }
        else
        {
            break;
        }
        uint _S812 = 2U * p_13;
        float4 _S813 = float4(*((&kernelContext_116)->parts_0+_S812)) ;
        uint col_0 = (as_type<uint>((_S813.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_116)->parts_0+(_S812 + 1U))) );
        uint _S814 = 4U * col_0;
        float4 _S815 = float4(*((&kernelContext_116)->colliders_0+_S814)) ;
        *((&kernelContext_116)->colliders_0+_S814) = packed_float4(float4((as_type<float>((slot_4))), _S815.y, _S815.z, _S815.w)) ;
        uint _S816 = _S814 + 1U;
        packed_float4 device* _S817 = (&kernelContext_116)->colliders_0+_S816;
        float3 _S818 = _S813.xyz - _S805;
        thread Quat_0 _S819 = _S807;
        float3 _S820 = rotate_0(&_S819, _S818);
        *_S817 = packed_float4(float4(_S820, (float4(*((&kernelContext_116)->colliders_0+_S816)) ).w)) ;
        packed_float4 device* _S821 = (&kernelContext_116)->colliders_0+(_S814 + 2U);
        thread Quat_0 _S822 = _S807;
        thread Quat_0 _S823 = rot_1;
        Quat_0 _S824 = quat_mul_0(&_S822, &_S823);
        thread Quat_0 _S825 = _S824;
        Quat_0 _S826 = normalized_0(&_S825);
        thread Quat_0 _S827 = _S826;
        float4 _S828 = quat_vec_0(&_S827);
        *_S821 = packed_float4(_S828) ;
        uint _S829 = 2U * col_0;
        uint4 _S830 = uint4(*((&kernelContext_116)->collider_info_0+_S829)) ;
        *((&kernelContext_116)->collider_info_0+_S829) = packed_uint4(uint4(_S830.x, slot_4, _S830.z, _S830.w)) ;
        *((&kernelContext_116)->collider_info_0+(_S829 + 1U)) = packed_uint4(uint4((as_type<uint>((_S808))), (as_type<uint>((_S811.x))), (as_type<uint>((_S811.w))), 0U)) ;
        p_13 = p_13 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], packed_float4 device* lp_21 [[buffer(24)]], packed_float4 device* ext_22 [[buffer(23)]], packed_float4 device* engine_contacts_21 [[buffer(25)]], CrushParams_0 constant* crush_21 [[buffer(28)]], packed_float4 device* crush_rw_21 [[buffer(27)]], packed_float4 device* crush_tab_21 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_117;
    (&kernelContext_117)->params_0 = params_21;
    (&kernelContext_117)->work_0 = work_21;
    (&kernelContext_117)->frames_0 = frames_21;
    (&kernelContext_117)->bodies_0 = bodies_21;
    (&kernelContext_117)->chunks_0 = chunks_21;
    (&kernelContext_117)->sort_0 = sort_21;
    (&kernelContext_117)->atom_0 = atom_21;
    (&kernelContext_117)->manifolds_0 = manifolds_21;
    (&kernelContext_117)->loads_0 = loads_21;
    (&kernelContext_117)->bonds_0 = bonds_21;
    (&kernelContext_117)->args_0 = args_21;
    (&kernelContext_117)->info_0 = info_21;
    (&kernelContext_117)->scan_0 = scan_21;
    (&kernelContext_117)->parts_0 = parts_21;
    (&kernelContext_117)->colliders_0 = colliders_21;
    (&kernelContext_117)->collider_info_0 = collider_info_21;
    (&kernelContext_117)->view_chunks_0 = view_chunks_21;
    (&kernelContext_117)->view_0 = view_21;
    (&kernelContext_117)->objects_0 = objects_21;
    (&kernelContext_117)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_117)->lines_0 = lines_21;
    (&kernelContext_117)->geometry_0 = geometry_21;
    (&kernelContext_117)->shots_0 = shots_21;
    (&kernelContext_117)->lp_0 = lp_21;
    (&kernelContext_117)->ext_0 = ext_22;
    (&kernelContext_117)->engine_contacts_0 = engine_contacts_21;
    (&kernelContext_117)->crush_0 = crush_21;
    (&kernelContext_117)->crush_rw_0 = crush_rw_21;
    (&kernelContext_117)->crush_tab_0 = crush_tab_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_117)->g_scan_0 = &g_scan_21;
    threadgroup array<float, int(64)> g_il_best_21;
    (&kernelContext_117)->g_il_best_0 = &g_il_best_21;
    uint s_10 = id_18.x;
    if(s_10 >= (params_21->body_count_0))
    {
        return;
    }
    uint _S831 = a_dirty_0(&kernelContext_117);
    atomic_store_explicit((&kernelContext_117)->atom_0+(_S831 + s_10), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_5 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_8 = fract(h_5);
    float q_18 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_8);
    float t_9 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_8));
    uint k_25 = uint(h_5);
    float3 _S832;
    if(k_25 == 0U)
    {
        _S832 = float3(0.92000001668930054f, t_9, 0.34959998726844788f);
    }
    else
    {
        if(k_25 == 1U)
        {
            _S832 = float3(q_18, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_25 == 2U)
            {
                _S832 = float3(0.34959998726844788f, 0.92000001668930054f, t_9);
            }
            else
            {
                if(k_25 == 3U)
                {
                    _S832 = float3(0.34959998726844788f, q_18, 0.92000001668930054f);
                }
                else
                {
                    if(k_25 == 4U)
                    {
                        _S832 = float3(t_9, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S832 = float3(0.92000001668930054f, 0.34959998726844788f, q_18);
                    }
                }
            }
        }
    }
    return _S832;
}

void write_object_0(uint o_4, const Quat_0 thread* q_19, float3 t_10, float4 color_0, KernelContext_0 thread* kernelContext_118)
{
    float3 _S833 = rotate_0(q_19, float3(1.0f, 0.0f, 0.0f));
    float3 _S834 = rotate_0(q_19, float3(0.0f, 1.0f, 0.0f));
    float3 _S835 = rotate_0(q_19, float3(0.0f, 0.0f, 1.0f));
    uint _S836 = 4U * o_4;
    *(kernelContext_118->objects_0+_S836) = packed_float4(float4(_S833.x, _S834.x, _S835.x, t_10.x)) ;
    *(kernelContext_118->objects_0+(_S836 + 1U)) = packed_float4(float4(_S833.y, _S834.y, _S835.y, t_10.y)) ;
    *(kernelContext_118->objects_0+(_S836 + 2U)) = packed_float4(float4(_S833.z, _S834.z, _S835.z, t_10.z)) ;
    *(kernelContext_118->objects_0+(_S836 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], packed_float4 device* lp_22 [[buffer(24)]], packed_float4 device* ext_23 [[buffer(23)]], packed_float4 device* engine_contacts_22 [[buffer(25)]], CrushParams_0 constant* crush_22 [[buffer(28)]], packed_float4 device* crush_rw_22 [[buffer(27)]], packed_float4 device* crush_tab_22 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_119;
    (&kernelContext_119)->params_0 = params_22;
    (&kernelContext_119)->work_0 = work_22;
    (&kernelContext_119)->frames_0 = frames_22;
    (&kernelContext_119)->bodies_0 = bodies_22;
    (&kernelContext_119)->chunks_0 = chunks_22;
    (&kernelContext_119)->sort_0 = sort_22;
    (&kernelContext_119)->atom_0 = atom_22;
    (&kernelContext_119)->manifolds_0 = manifolds_22;
    (&kernelContext_119)->loads_0 = loads_22;
    (&kernelContext_119)->bonds_0 = bonds_22;
    (&kernelContext_119)->args_0 = args_22;
    (&kernelContext_119)->info_0 = info_22;
    (&kernelContext_119)->scan_0 = scan_22;
    (&kernelContext_119)->parts_0 = parts_22;
    (&kernelContext_119)->colliders_0 = colliders_22;
    (&kernelContext_119)->collider_info_0 = collider_info_22;
    (&kernelContext_119)->view_chunks_0 = view_chunks_22;
    (&kernelContext_119)->view_0 = view_22;
    (&kernelContext_119)->objects_0 = objects_22;
    (&kernelContext_119)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_119)->lines_0 = lines_22;
    (&kernelContext_119)->geometry_0 = geometry_22;
    (&kernelContext_119)->shots_0 = shots_22;
    (&kernelContext_119)->lp_0 = lp_22;
    (&kernelContext_119)->ext_0 = ext_23;
    (&kernelContext_119)->engine_contacts_0 = engine_contacts_22;
    (&kernelContext_119)->crush_0 = crush_22;
    (&kernelContext_119)->crush_rw_0 = crush_rw_22;
    (&kernelContext_119)->crush_tab_0 = crush_tab_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_119)->g_scan_0 = &g_scan_22;
    threadgroup array<float, int(64)> g_il_best_22;
    (&kernelContext_119)->g_il_best_0 = &g_il_best_22;
    uint k_26 = id_19.x;
    if(k_26 >= (params_22->chunk_count_0))
    {
        return;
    }
    uint _S837 = body_of_0(k_26, &kernelContext_119);
    uint b_36 = 7U * _S837;
    thread Quat_0 q_20;
    thread float3 origin_5;
    structure_pose_0(_S837, b_36, &q_20, &origin_5, &kernelContext_119);
    uint _S838 = 3U * k_26;
    float4 _S839 = float4(*((&kernelContext_119)->view_chunks_0+_S838)) ;
    float3 base_2 = (float4(*((&kernelContext_119)->view_chunks_0+(_S838 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_119)->bodies_0+b_36)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_119)->view_0->mode_0) == 1U)
    {
        float _S840 = max(length((float4(*((&kernelContext_119)->bodies_0+(b_36 + 3U))) ).xyz), length((float4(*((&kernelContext_119)->bodies_0+(b_36 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S840 > ((&kernelContext_119)->view_0->rest_speed_0))
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
        if(((&kernelContext_119)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S837);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S837);
            }
        }
    }
    uint _S841 = (as_type<uint>((_S839.w)));
    Quat_0 _S842 = quat_of_0(float4(*((&kernelContext_119)->view_chunks_0+(_S838 + 1U))) );
    thread Quat_0 _S843 = q_20;
    thread Quat_0 _S844 = _S842;
    Quat_0 _S845 = quat_mul_0(&_S843, &_S844);
    thread Quat_0 _S846 = _S845;
    Quat_0 _S847 = normalized_0(&_S846);
    float3 _S848 = _S839.xyz;
    thread Quat_0 _S849 = q_20;
    float3 _S850 = rotate_0(&_S849, _S848);
    float3 _S851 = origin_5 + _S850;
    float4 _S852 = float4(color_1, 1.0f);
    thread Quat_0 _S853 = _S847;
    write_object_0(_S841, &_S853, _S851, _S852, &kernelContext_119);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], atomic<uint> device* atom_23 [[buffer(5)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], packed_float4 device* lp_23 [[buffer(24)]], packed_float4 device* ext_24 [[buffer(23)]], packed_float4 device* engine_contacts_23 [[buffer(25)]], CrushParams_0 constant* crush_23 [[buffer(28)]], packed_float4 device* crush_rw_23 [[buffer(27)]], packed_float4 device* crush_tab_23 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_120;
    (&kernelContext_120)->params_0 = params_23;
    (&kernelContext_120)->work_0 = work_23;
    (&kernelContext_120)->frames_0 = frames_23;
    (&kernelContext_120)->bodies_0 = bodies_23;
    (&kernelContext_120)->chunks_0 = chunks_23;
    (&kernelContext_120)->sort_0 = sort_23;
    (&kernelContext_120)->atom_0 = atom_23;
    (&kernelContext_120)->manifolds_0 = manifolds_23;
    (&kernelContext_120)->loads_0 = loads_23;
    (&kernelContext_120)->bonds_0 = bonds_23;
    (&kernelContext_120)->args_0 = args_23;
    (&kernelContext_120)->info_0 = info_23;
    (&kernelContext_120)->scan_0 = scan_23;
    (&kernelContext_120)->parts_0 = parts_23;
    (&kernelContext_120)->colliders_0 = colliders_23;
    (&kernelContext_120)->collider_info_0 = collider_info_23;
    (&kernelContext_120)->view_chunks_0 = view_chunks_23;
    (&kernelContext_120)->view_0 = view_23;
    (&kernelContext_120)->objects_0 = objects_23;
    (&kernelContext_120)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_120)->lines_0 = lines_23;
    (&kernelContext_120)->geometry_0 = geometry_23;
    (&kernelContext_120)->shots_0 = shots_23;
    (&kernelContext_120)->lp_0 = lp_23;
    (&kernelContext_120)->ext_0 = ext_24;
    (&kernelContext_120)->engine_contacts_0 = engine_contacts_23;
    (&kernelContext_120)->crush_0 = crush_23;
    (&kernelContext_120)->crush_rw_0 = crush_rw_23;
    (&kernelContext_120)->crush_tab_0 = crush_tab_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_120)->g_scan_0 = &g_scan_23;
    threadgroup array<float, int(64)> g_il_best_23;
    (&kernelContext_120)->g_il_best_0 = &g_il_best_23;
    uint i_10 = id_20.x;
    if(i_10 >= (view_23->shot_count_1))
    {
        return;
    }
    uint b_37 = 7U * (&kernelContext_120)->shot_bodies_0[i_10];
    uint o_5 = (&kernelContext_120)->view_0->impactor_base_0 + i_10;
    if(((float4(*((&kernelContext_120)->bodies_0+b_37)) ).w) == 0.0f)
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
            *((&kernelContext_120)->objects_0+(4U * o_5 + w_5)) = packed_float4(float4(0.0f) ) ;
            w_5 = w_5 + 1U;
        }
        return;
    }
    thread Quat_0 _S854 = quat_of_0(float4(*((&kernelContext_120)->bodies_0+(b_37 + 1U))) );
    Quat_0 _S855 = rot_from_rubble_0(&_S854);
    float3 _S856 = from_rubble_0((float4(*((&kernelContext_120)->bodies_0+b_37)) ).xyz);
    float4 _S857 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S858 = _S855;
    write_object_0(o_5, &_S858, _S856, _S857, &kernelContext_120);
    return;
}

float3 chunk_world_0(uint k_27, KernelContext_0 thread* kernelContext_121)
{
    uint _S859 = body_of_0(k_27, kernelContext_121);
    thread Quat_0 q_21;
    thread float3 origin_6;
    structure_pose_0(_S859, 7U * _S859, &q_21, &origin_6, kernelContext_121);
    float3 _S860 = (float4(*(kernelContext_121->view_chunks_0+3U * k_27)) ).xyz;
    thread Quat_0 _S861 = q_21;
    float3 _S862 = rotate_0(&_S861, _S860);
    return origin_6 + _S862;
}

void write_line_vertex_0(uint v_13, float3 p_14, KernelContext_0 thread* kernelContext_122)
{
    uint _S863 = 7U * v_13;
    *(kernelContext_122->lines_0+_S863) = p_14.x;
    *(kernelContext_122->lines_0+(_S863 + 1U)) = p_14.y;
    *(kernelContext_122->lines_0+(_S863 + 2U)) = p_14.z;
    *(kernelContext_122->lines_0+(_S863 + 3U)) = 1.0f;
    *(kernelContext_122->lines_0+(_S863 + 4U)) = 0.86000001430511475f;
    *(kernelContext_122->lines_0+(_S863 + 5U)) = 0.25f;
    *(kernelContext_122->lines_0+(_S863 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], atomic<uint> device* atom_24 [[buffer(5)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], packed_float4 device* lp_24 [[buffer(24)]], packed_float4 device* ext_25 [[buffer(23)]], packed_float4 device* engine_contacts_24 [[buffer(25)]], CrushParams_0 constant* crush_24 [[buffer(28)]], packed_float4 device* crush_rw_24 [[buffer(27)]], packed_float4 device* crush_tab_24 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_123;
    (&kernelContext_123)->params_0 = params_24;
    (&kernelContext_123)->work_0 = work_24;
    (&kernelContext_123)->frames_0 = frames_24;
    (&kernelContext_123)->bodies_0 = bodies_24;
    (&kernelContext_123)->chunks_0 = chunks_24;
    (&kernelContext_123)->sort_0 = sort_24;
    (&kernelContext_123)->atom_0 = atom_24;
    (&kernelContext_123)->manifolds_0 = manifolds_24;
    (&kernelContext_123)->loads_0 = loads_24;
    (&kernelContext_123)->bonds_0 = bonds_24;
    (&kernelContext_123)->args_0 = args_24;
    (&kernelContext_123)->info_0 = info_24;
    (&kernelContext_123)->scan_0 = scan_24;
    (&kernelContext_123)->parts_0 = parts_24;
    (&kernelContext_123)->colliders_0 = colliders_24;
    (&kernelContext_123)->collider_info_0 = collider_info_24;
    (&kernelContext_123)->view_chunks_0 = view_chunks_24;
    (&kernelContext_123)->view_0 = view_24;
    (&kernelContext_123)->objects_0 = objects_24;
    (&kernelContext_123)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_123)->lines_0 = lines_24;
    (&kernelContext_123)->geometry_0 = geometry_24;
    (&kernelContext_123)->shots_0 = shots_24;
    (&kernelContext_123)->lp_0 = lp_24;
    (&kernelContext_123)->ext_0 = ext_25;
    (&kernelContext_123)->engine_contacts_0 = engine_contacts_24;
    (&kernelContext_123)->crush_0 = crush_24;
    (&kernelContext_123)->crush_rw_0 = crush_rw_24;
    (&kernelContext_123)->crush_tab_0 = crush_tab_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_123)->g_scan_0 = &g_scan_24;
    threadgroup array<float, int(64)> g_il_best_24;
    (&kernelContext_123)->g_il_best_0 = &g_il_best_24;
    uint i_11 = id_21.x;
    if(i_11 >= (params_24->bond_count_0))
    {
        return;
    }
    uint4 _S864 = uint4(*((&kernelContext_123)->bonds_0+i_11)) ;
    float3 _S865 = chunk_world_0(_S864.x, &kernelContext_123);
    uint _S866 = w_alive_0(&kernelContext_123);
    float3 b_38;
    if((*((&kernelContext_123)->work_0+(_S866 + i_11))) != 0U)
    {
        float3 _S867 = chunk_world_0(_S864.y, &kernelContext_123);
        b_38 = _S867;
    }
    else
    {
        b_38 = _S865;
    }
    uint _S868 = 2U * i_11;
    write_line_vertex_0(_S868, _S865, &kernelContext_123);
    write_line_vertex_0(_S868 + 1U, b_38, &kernelContext_123);
    return;
}

[[kernel]] void fr_strike_reset(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], atomic<uint> device* atom_25 [[buffer(5)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], packed_float4 device* lp_25 [[buffer(24)]], packed_float4 device* ext_26 [[buffer(23)]], packed_float4 device* engine_contacts_25 [[buffer(25)]], CrushParams_0 constant* crush_25 [[buffer(28)]], packed_float4 device* crush_rw_25 [[buffer(27)]], packed_float4 device* crush_tab_25 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_124;
    (&kernelContext_124)->params_0 = params_25;
    (&kernelContext_124)->work_0 = work_25;
    (&kernelContext_124)->frames_0 = frames_25;
    (&kernelContext_124)->bodies_0 = bodies_25;
    (&kernelContext_124)->chunks_0 = chunks_25;
    (&kernelContext_124)->sort_0 = sort_25;
    (&kernelContext_124)->atom_0 = atom_25;
    (&kernelContext_124)->manifolds_0 = manifolds_25;
    (&kernelContext_124)->loads_0 = loads_25;
    (&kernelContext_124)->bonds_0 = bonds_25;
    (&kernelContext_124)->args_0 = args_25;
    (&kernelContext_124)->info_0 = info_25;
    (&kernelContext_124)->scan_0 = scan_25;
    (&kernelContext_124)->parts_0 = parts_25;
    (&kernelContext_124)->colliders_0 = colliders_25;
    (&kernelContext_124)->collider_info_0 = collider_info_25;
    (&kernelContext_124)->view_chunks_0 = view_chunks_25;
    (&kernelContext_124)->view_0 = view_25;
    (&kernelContext_124)->objects_0 = objects_25;
    (&kernelContext_124)->shot_bodies_0 = shot_bodies_25;
    (&kernelContext_124)->lines_0 = lines_25;
    (&kernelContext_124)->geometry_0 = geometry_25;
    (&kernelContext_124)->shots_0 = shots_25;
    (&kernelContext_124)->lp_0 = lp_25;
    (&kernelContext_124)->ext_0 = ext_26;
    (&kernelContext_124)->engine_contacts_0 = engine_contacts_25;
    (&kernelContext_124)->crush_0 = crush_25;
    (&kernelContext_124)->crush_rw_0 = crush_rw_25;
    (&kernelContext_124)->crush_tab_0 = crush_tab_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_124)->g_scan_0 = &g_scan_25;
    threadgroup array<float, int(64)> g_il_best_25;
    (&kernelContext_124)->g_il_best_0 = &g_il_best_25;
    uint b_39 = id_22.x;
    if(b_39 >= (params_25->body_count_0))
    {
        return;
    }
    uint _S869 = a_strike_0(&kernelContext_124);
    uint at_12 = _S869 + 9U * b_39;
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
        atomic_store_explicit((&kernelContext_124)->atom_0+(at_12 + w_6), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_124)->atom_0+(at_12 + 3U + w_6), 0U, memory_order_relaxed);
        w_6 = w_6 + 1U;
    }
    atomic_store_explicit((&kernelContext_124)->atom_0+(at_12 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_124)->atom_0+(at_12 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_124)->atom_0+(at_12 + 8U), 4294967295U, memory_order_relaxed);
    uint _S870 = strike_at_0(b_39, &kernelContext_124);
    (*((&kernelContext_124)->frames_0+_S870))[int(3)] = 0.0f;
    return;
}

uint ordered_0(float f_9)
{
    uint u_6 = (as_type<uint>((f_9)));
    uint _S871;
    if((u_6 & 2147483648U) != 0U)
    {
        _S871 = ~u_6;
    }
    else
    {
        _S871 = u_6 | 2147483648U;
    }
    return _S871;
}

[[kernel]] void fr_strike_bounds(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], atomic<uint> device* atom_26 [[buffer(5)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], packed_float4 device* lp_26 [[buffer(24)]], packed_float4 device* ext_27 [[buffer(23)]], packed_float4 device* engine_contacts_26 [[buffer(25)]], CrushParams_0 constant* crush_26 [[buffer(28)]], packed_float4 device* crush_rw_26 [[buffer(27)]], packed_float4 device* crush_tab_26 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_125;
    (&kernelContext_125)->params_0 = params_26;
    (&kernelContext_125)->work_0 = work_26;
    (&kernelContext_125)->frames_0 = frames_26;
    (&kernelContext_125)->bodies_0 = bodies_26;
    (&kernelContext_125)->chunks_0 = chunks_26;
    (&kernelContext_125)->sort_0 = sort_26;
    (&kernelContext_125)->atom_0 = atom_26;
    (&kernelContext_125)->manifolds_0 = manifolds_26;
    (&kernelContext_125)->loads_0 = loads_26;
    (&kernelContext_125)->bonds_0 = bonds_26;
    (&kernelContext_125)->args_0 = args_26;
    (&kernelContext_125)->info_0 = info_26;
    (&kernelContext_125)->scan_0 = scan_26;
    (&kernelContext_125)->parts_0 = parts_26;
    (&kernelContext_125)->colliders_0 = colliders_26;
    (&kernelContext_125)->collider_info_0 = collider_info_26;
    (&kernelContext_125)->view_chunks_0 = view_chunks_26;
    (&kernelContext_125)->view_0 = view_26;
    (&kernelContext_125)->objects_0 = objects_26;
    (&kernelContext_125)->shot_bodies_0 = shot_bodies_26;
    (&kernelContext_125)->lines_0 = lines_26;
    (&kernelContext_125)->geometry_0 = geometry_26;
    (&kernelContext_125)->shots_0 = shots_26;
    (&kernelContext_125)->lp_0 = lp_26;
    (&kernelContext_125)->ext_0 = ext_27;
    (&kernelContext_125)->engine_contacts_0 = engine_contacts_26;
    (&kernelContext_125)->crush_0 = crush_26;
    (&kernelContext_125)->crush_rw_0 = crush_rw_26;
    (&kernelContext_125)->crush_tab_0 = crush_tab_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_125)->g_scan_0 = &g_scan_26;
    threadgroup array<float, int(64)> g_il_best_26;
    (&kernelContext_125)->g_il_best_0 = &g_il_best_26;
    uint k_28 = id_23.x;
    bool _S872;
    if(k_28 >= (params_26->chunk_count_0))
    {
        _S872 = true;
    }
    else
    {
        _S872 = ((&kernelContext_125)->params_0->shot_count_0) == 0U;
    }
    if(_S872)
    {
        return;
    }
    thread Quat_0 q_22;
    float3 _S873 = chunk_box_world_0(k_28, &q_22, &kernelContext_125);
    float _S874 = (float4(*((&kernelContext_125)->geometry_0+3U * k_28)) ).w;
    uint _S875 = a_strike_0(&kernelContext_125);
    uint _S876 = body_of_0(k_28, &kernelContext_125);
    uint _S877 = _S875 + 9U * _S876;
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
        uint _S878 = ax_1;
        uint _S879 = atomic_fetch_min_explicit((&kernelContext_125)->atom_0+(_S877 + ax_1), ordered_0(_S873[ax_1] - _S874), memory_order_relaxed);
        uint _S880 = atomic_fetch_max_explicit((&kernelContext_125)->atom_0+(_S877 + 3U + ax_1), ordered_0(_S873[_S878] + _S874), memory_order_relaxed);
        ax_1 = ax_1 + 1U;
    }
    return;
}

float unordered_0(uint u_7)
{
    uint _S881;
    if((u_7 & 2147483648U) != 0U)
    {
        _S881 = u_7 & 2147483647U;
    }
    else
    {
        _S881 = ~u_7;
    }
    return (as_type<float>((_S881)));
}

[[kernel]] void fr_strike_find(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], atomic<uint> device* atom_27 [[buffer(5)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], packed_float4 device* lp_27 [[buffer(24)]], packed_float4 device* ext_28 [[buffer(23)]], packed_float4 device* engine_contacts_27 [[buffer(25)]], CrushParams_0 constant* crush_27 [[buffer(28)]], packed_float4 device* crush_rw_27 [[buffer(27)]], packed_float4 device* crush_tab_27 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_126;
    (&kernelContext_126)->params_0 = params_27;
    (&kernelContext_126)->work_0 = work_27;
    (&kernelContext_126)->frames_0 = frames_27;
    (&kernelContext_126)->bodies_0 = bodies_27;
    (&kernelContext_126)->chunks_0 = chunks_27;
    (&kernelContext_126)->sort_0 = sort_27;
    (&kernelContext_126)->atom_0 = atom_27;
    (&kernelContext_126)->manifolds_0 = manifolds_27;
    (&kernelContext_126)->loads_0 = loads_27;
    (&kernelContext_126)->bonds_0 = bonds_27;
    (&kernelContext_126)->args_0 = args_27;
    (&kernelContext_126)->info_0 = info_27;
    (&kernelContext_126)->scan_0 = scan_27;
    (&kernelContext_126)->parts_0 = parts_27;
    (&kernelContext_126)->colliders_0 = colliders_27;
    (&kernelContext_126)->collider_info_0 = collider_info_27;
    (&kernelContext_126)->view_chunks_0 = view_chunks_27;
    (&kernelContext_126)->view_0 = view_27;
    (&kernelContext_126)->objects_0 = objects_27;
    (&kernelContext_126)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_126)->lines_0 = lines_27;
    (&kernelContext_126)->geometry_0 = geometry_27;
    (&kernelContext_126)->shots_0 = shots_27;
    (&kernelContext_126)->lp_0 = lp_27;
    (&kernelContext_126)->ext_0 = ext_28;
    (&kernelContext_126)->engine_contacts_0 = engine_contacts_27;
    (&kernelContext_126)->crush_0 = crush_27;
    (&kernelContext_126)->crush_rw_0 = crush_rw_27;
    (&kernelContext_126)->crush_tab_0 = crush_tab_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_126)->g_scan_0 = &g_scan_27;
    threadgroup array<float, int(64)> g_il_best_27;
    (&kernelContext_126)->g_il_best_0 = &g_il_best_27;
    uint b_40 = id_24.x;
    bool _S882;
    if(b_40 >= (params_27->body_count_0))
    {
        _S882 = true;
    }
    else
    {
        _S882 = ((&kernelContext_126)->params_0->shot_count_0) == 0U;
    }
    if(_S882)
    {
        return;
    }
    uint _S883 = a_strike_0(&kernelContext_126);
    uint at_13 = _S883 + 9U * b_40;
    uint _S884 = atomic_load_explicit((&kernelContext_126)->atom_0+at_13, memory_order_relaxed);
    if(_S884 == 4294967295U)
    {
        return;
    }
    uint _S885 = atomic_load_explicit((&kernelContext_126)->atom_0+at_13, memory_order_relaxed);
    float _S886 = unordered_0(_S885);
    uint _S887 = atomic_load_explicit((&kernelContext_126)->atom_0+(at_13 + 1U), memory_order_relaxed);
    float _S888 = unordered_0(_S887);
    uint _S889 = atomic_load_explicit((&kernelContext_126)->atom_0+(at_13 + 2U), memory_order_relaxed);
    float3 _S890 = float3(_S886, _S888, unordered_0(_S889));
    uint _S891 = atomic_load_explicit((&kernelContext_126)->atom_0+(at_13 + 3U), memory_order_relaxed);
    float _S892 = unordered_0(_S891);
    uint _S893 = atomic_load_explicit((&kernelContext_126)->atom_0+(at_13 + 4U), memory_order_relaxed);
    float _S894 = unordered_0(_S893);
    uint _S895 = atomic_load_explicit((&kernelContext_126)->atom_0+(at_13 + 5U), memory_order_relaxed);
    float3 _S896 = float3(_S892, _S894, unordered_0(_S895));
    float3 _S897 = from_rubble_0((float4(*((&kernelContext_126)->bodies_0+(7U * b_40 + 3U))) ).xyz);
    float _S898 = float_bits_0(2139095040U);
    uint k_29 = 0U;
    for(;;)
    {
        if(k_29 < ((&kernelContext_126)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S899 = float4(*((&kernelContext_126)->shots_0+2U * k_29)) ;
        uint sb_1 = 7U * (as_type<uint>((_S899.x)));
        if(((float4(*((&kernelContext_126)->bodies_0+sb_1)) ).w) == 0.0f)
        {
            k_29 = k_29 + 1U;
            continue;
        }
        float3 c_18 = from_rubble_0((float4(*((&kernelContext_126)->bodies_0+sb_1)) ).xyz);
        float3 rel_2 = from_rubble_0((float4(*((&kernelContext_126)->bodies_0+(sb_1 + 3U))) ).xyz) - _S897;
        if(!((length(rel_2)) > ((&kernelContext_126)->params_0->strike_speed_0)))
        {
            k_29 = k_29 + 1U;
            continue;
        }
        float3 _S900 = float3(_S899.y) ;
        float3 _S901 = _S890 - _S900;
        float3 _S902 = _S896 + _S900;
        float _S903 = (&kernelContext_126)->params_0->dt_0;
        float t0_1 = 0.0f;
        float t1_1 = _S903;
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
            uint _S904 = ax_2;
            if((abs(rel_2[ax_2])) < 9.999999960041972e-13f)
            {
                uint _S905 = ax_2;
                if((c_18[ax_2]) < (_S901[ax_2]))
                {
                    _S882 = true;
                }
                else
                {
                    _S882 = (c_18[_S905]) > (_S902[ax_2]);
                }
                float t0_2;
                if(_S882)
                {
                    t0_2 = _S898;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + 1U;
                continue;
            }
            float ta_1 = (_S901[ax_2] - c_18[ax_2]) / rel_2[_S904];
            float tb_1 = (_S902[ax_2] - c_18[ax_2]) / rel_2[_S904];
            float _S906 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S906;
            ax_2 = ax_2 + 1U;
        }
        if(t0_1 <= t1_1)
        {
            uint _S907 = strike_at_0(b_40, &kernelContext_126);
            *((&kernelContext_126)->frames_0+_S907) = packed_float4(float4(c_18, float(k_29 + 1U))) ;
            *((&kernelContext_126)->frames_0+(_S907 + 1U)) = packed_float4(float4(rel_2, t0_1)) ;
            return;
        }
        k_29 = k_29 + 1U;
    }
    return;
}

uint a_strike_dist_0(KernelContext_0 thread* kernelContext_127)
{
    uint _S908 = a_cs_0(0U, kernelContext_127);
    return _S908 + 12U * kernelContext_127->params_0->body_count_0;
}

[[kernel]] void fr_strike_entry(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], atomic<uint> device* atom_28 [[buffer(5)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], packed_float4 device* lp_28 [[buffer(24)]], packed_float4 device* ext_29 [[buffer(23)]], packed_float4 device* engine_contacts_28 [[buffer(25)]], CrushParams_0 constant* crush_28 [[buffer(28)]], packed_float4 device* crush_rw_28 [[buffer(27)]], packed_float4 device* crush_tab_28 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_128;
    (&kernelContext_128)->params_0 = params_28;
    (&kernelContext_128)->work_0 = work_28;
    (&kernelContext_128)->frames_0 = frames_28;
    (&kernelContext_128)->bodies_0 = bodies_28;
    (&kernelContext_128)->chunks_0 = chunks_28;
    (&kernelContext_128)->sort_0 = sort_28;
    (&kernelContext_128)->atom_0 = atom_28;
    (&kernelContext_128)->manifolds_0 = manifolds_28;
    (&kernelContext_128)->loads_0 = loads_28;
    (&kernelContext_128)->bonds_0 = bonds_28;
    (&kernelContext_128)->args_0 = args_28;
    (&kernelContext_128)->info_0 = info_28;
    (&kernelContext_128)->scan_0 = scan_28;
    (&kernelContext_128)->parts_0 = parts_28;
    (&kernelContext_128)->colliders_0 = colliders_28;
    (&kernelContext_128)->collider_info_0 = collider_info_28;
    (&kernelContext_128)->view_chunks_0 = view_chunks_28;
    (&kernelContext_128)->view_0 = view_28;
    (&kernelContext_128)->objects_0 = objects_28;
    (&kernelContext_128)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_128)->lines_0 = lines_28;
    (&kernelContext_128)->geometry_0 = geometry_28;
    (&kernelContext_128)->shots_0 = shots_28;
    (&kernelContext_128)->lp_0 = lp_28;
    (&kernelContext_128)->ext_0 = ext_29;
    (&kernelContext_128)->engine_contacts_0 = engine_contacts_28;
    (&kernelContext_128)->crush_0 = crush_28;
    (&kernelContext_128)->crush_rw_0 = crush_rw_28;
    (&kernelContext_128)->crush_tab_0 = crush_tab_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_128)->g_scan_0 = &g_scan_28;
    threadgroup array<float, int(64)> g_il_best_28;
    (&kernelContext_128)->g_il_best_0 = &g_il_best_28;
    uint k_30 = id_25.x;
    bool _S909;
    if(k_30 >= (params_28->chunk_count_0))
    {
        _S909 = true;
    }
    else
    {
        _S909 = ((&kernelContext_128)->params_0->shot_count_0) == 0U;
    }
    if(_S909)
    {
        return;
    }
    uint _S910 = body_of_0(k_30, &kernelContext_128);
    thread uint shot_5;
    bool _S911 = struck_0(_S910, &shot_5, &kernelContext_128);
    if(!_S911)
    {
        return;
    }
    uint _S912 = strike_at_0(_S910, &kernelContext_128);
    float4 _S913 = float4(*((&kernelContext_128)->frames_0+(_S912 + 1U))) ;
    float3 from_0 = (float4(*((&kernelContext_128)->frames_0+_S912)) ).xyz;
    float3 rel_3 = _S913.xyz;
    float3 d_20 = rel_3 / float3(max(length(rel_3), 9.999999960041972e-13f)) ;
    float _S914 = (float4(*((&kernelContext_128)->shots_0+2U * shot_5)) ).y;
    thread Quat_0 q_23;
    float3 _S915 = chunk_box_world_0(k_30, &q_23, &kernelContext_128);
    uint _S916 = 3U * k_30;
    Quat_0 _S917 = quat_of_0(float4(*((&kernelContext_128)->geometry_0+(_S916 + 2U))) );
    thread Quat_0 _S918 = q_23;
    thread Quat_0 _S919 = _S917;
    Quat_0 _S920 = quat_mul_0(&_S918, &_S919);
    thread Quat_0 _S921 = _S920;
    Quat_0 _S922 = normalized_0(&_S921);
    float3 _S923 = (float4(*((&kernelContext_128)->geometry_0+(_S916 + 1U))) ).xyz;
    float _S924 = float_bits_0(2139095040U);
    float _S925 = (&kernelContext_128)->params_0->dt_0;
    float t_in_5 = 0.0f;
    float t_out_0 = _S925;
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
        thread Quat_0 _S926 = _S922;
        float3 _S927 = rotate_0(&_S926, e_3);
        float o_6 = dot(from_0 - _S915, _S927);
        float dv_1 = dot(rel_3, _S927);
        float ext_30 = _S923[ax_3] + _S914;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            float t_in_6;
            if((abs(o_6)) > ext_30)
            {
                t_in_6 = _S924;
            }
            else
            {
                t_in_6 = t_in_5;
            }
            t_in_5 = t_in_6;
            ax_3 = ax_3 + 1U;
            continue;
        }
        float ta_2 = (- ext_30 - o_6) / dv_1;
        float tb_2 = (ext_30 - o_6) / dv_1;
        float _S928 = min(t_out_0, max(ta_2, tb_2));
        t_in_5 = max(t_in_5, min(ta_2, tb_2));
        t_out_0 = _S928;
        ax_3 = ax_3 + 1U;
    }
    uint _S929 = a_strike_0(&kernelContext_128);
    uint at_14 = _S929 + 9U * _S910;
    if(t_in_5 <= t_out_0)
    {
        uint _S930 = atomic_fetch_min_explicit((&kernelContext_128)->atom_0+(at_14 + 6U), (as_type<uint>((t_in_5))), memory_order_relaxed);
    }
    float3 r_22 = _S915 - (from_0 + rel_3 * float3(_S913.w) );
    float along_1 = dot(r_22, d_20);
    float dist_0 = length(r_22 - d_20 * float3(along_1) ) + abs(along_1) * 0.00100000004749745f;
    uint _S931 = a_strike_dist_0(&kernelContext_128);
    uint _S932 = (as_type<uint>((dist_0)));
    atomic_store_explicit((&kernelContext_128)->atom_0+(_S931 + k_30), _S932, memory_order_relaxed);
    uint _S933 = atomic_fetch_min_explicit((&kernelContext_128)->atom_0+(at_14 + 7U), _S932, memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], atomic<uint> device* atom_29 [[buffer(5)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], packed_float4 device* lp_29 [[buffer(24)]], packed_float4 device* ext_31 [[buffer(23)]], packed_float4 device* engine_contacts_29 [[buffer(25)]], CrushParams_0 constant* crush_29 [[buffer(28)]], packed_float4 device* crush_rw_29 [[buffer(27)]], packed_float4 device* crush_tab_29 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_129;
    (&kernelContext_129)->params_0 = params_29;
    (&kernelContext_129)->work_0 = work_29;
    (&kernelContext_129)->frames_0 = frames_29;
    (&kernelContext_129)->bodies_0 = bodies_29;
    (&kernelContext_129)->chunks_0 = chunks_29;
    (&kernelContext_129)->sort_0 = sort_29;
    (&kernelContext_129)->atom_0 = atom_29;
    (&kernelContext_129)->manifolds_0 = manifolds_29;
    (&kernelContext_129)->loads_0 = loads_29;
    (&kernelContext_129)->bonds_0 = bonds_29;
    (&kernelContext_129)->args_0 = args_29;
    (&kernelContext_129)->info_0 = info_29;
    (&kernelContext_129)->scan_0 = scan_29;
    (&kernelContext_129)->parts_0 = parts_29;
    (&kernelContext_129)->colliders_0 = colliders_29;
    (&kernelContext_129)->collider_info_0 = collider_info_29;
    (&kernelContext_129)->view_chunks_0 = view_chunks_29;
    (&kernelContext_129)->view_0 = view_29;
    (&kernelContext_129)->objects_0 = objects_29;
    (&kernelContext_129)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_129)->lines_0 = lines_29;
    (&kernelContext_129)->geometry_0 = geometry_29;
    (&kernelContext_129)->shots_0 = shots_29;
    (&kernelContext_129)->lp_0 = lp_29;
    (&kernelContext_129)->ext_0 = ext_31;
    (&kernelContext_129)->engine_contacts_0 = engine_contacts_29;
    (&kernelContext_129)->crush_0 = crush_29;
    (&kernelContext_129)->crush_rw_0 = crush_rw_29;
    (&kernelContext_129)->crush_tab_0 = crush_tab_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_129)->g_scan_0 = &g_scan_29;
    threadgroup array<float, int(64)> g_il_best_29;
    (&kernelContext_129)->g_il_best_0 = &g_il_best_29;
    uint k_31 = id_26.x;
    bool _S934;
    if(k_31 >= (params_29->chunk_count_0))
    {
        _S934 = true;
    }
    else
    {
        _S934 = ((&kernelContext_129)->params_0->shot_count_0) == 0U;
    }
    if(_S934)
    {
        return;
    }
    uint _S935 = body_of_0(k_31, &kernelContext_129);
    thread uint shot_6;
    bool _S936 = struck_0(_S935, &shot_6, &kernelContext_129);
    if(!_S936)
    {
        return;
    }
    uint _S937 = a_strike_0(&kernelContext_129);
    uint at_15 = _S937 + 9U * _S935;
    uint _S938 = a_strike_dist_0(&kernelContext_129);
    uint _S939 = atomic_load_explicit((&kernelContext_129)->atom_0+(_S938 + k_31), memory_order_relaxed);
    uint _S940 = atomic_load_explicit((&kernelContext_129)->atom_0+(at_15 + 7U), memory_order_relaxed);
    if(_S939 == _S940)
    {
        uint _S941 = atomic_fetch_min_explicit((&kernelContext_129)->atom_0+(at_15 + 8U), k_31, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], atomic<uint> device* atom_30 [[buffer(5)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], packed_float4 device* lp_30 [[buffer(24)]], packed_float4 device* ext_32 [[buffer(23)]], packed_float4 device* engine_contacts_30 [[buffer(25)]], CrushParams_0 constant* crush_30 [[buffer(28)]], packed_float4 device* crush_rw_30 [[buffer(27)]], packed_float4 device* crush_tab_30 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_130;
    (&kernelContext_130)->params_0 = params_30;
    (&kernelContext_130)->work_0 = work_30;
    (&kernelContext_130)->frames_0 = frames_30;
    (&kernelContext_130)->bodies_0 = bodies_30;
    (&kernelContext_130)->chunks_0 = chunks_30;
    (&kernelContext_130)->sort_0 = sort_30;
    (&kernelContext_130)->atom_0 = atom_30;
    (&kernelContext_130)->manifolds_0 = manifolds_30;
    (&kernelContext_130)->loads_0 = loads_30;
    (&kernelContext_130)->bonds_0 = bonds_30;
    (&kernelContext_130)->args_0 = args_30;
    (&kernelContext_130)->info_0 = info_30;
    (&kernelContext_130)->scan_0 = scan_30;
    (&kernelContext_130)->parts_0 = parts_30;
    (&kernelContext_130)->colliders_0 = colliders_30;
    (&kernelContext_130)->collider_info_0 = collider_info_30;
    (&kernelContext_130)->view_chunks_0 = view_chunks_30;
    (&kernelContext_130)->view_0 = view_30;
    (&kernelContext_130)->objects_0 = objects_30;
    (&kernelContext_130)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_130)->lines_0 = lines_30;
    (&kernelContext_130)->geometry_0 = geometry_30;
    (&kernelContext_130)->shots_0 = shots_30;
    (&kernelContext_130)->lp_0 = lp_30;
    (&kernelContext_130)->ext_0 = ext_32;
    (&kernelContext_130)->engine_contacts_0 = engine_contacts_30;
    (&kernelContext_130)->crush_0 = crush_30;
    (&kernelContext_130)->crush_rw_0 = crush_rw_30;
    (&kernelContext_130)->crush_tab_0 = crush_tab_30;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_130)->g_scan_0 = &g_scan_30;
    threadgroup array<float, int(64)> g_il_best_30;
    (&kernelContext_130)->g_il_best_0 = &g_il_best_30;
    uint b_41 = id_27.x;
    bool _S942;
    if(b_41 >= (params_30->body_count_0))
    {
        _S942 = true;
    }
    else
    {
        _S942 = ((&kernelContext_130)->params_0->shot_count_0) == 0U;
    }
    if(_S942)
    {
        return;
    }
    thread uint shot_7;
    bool _S943 = struck_0(b_41, &shot_7, &kernelContext_130);
    if(!_S943)
    {
        return;
    }
    uint _S944 = a_strike_0(&kernelContext_130);
    uint at_16 = _S944 + 9U * b_41;
    uint _S945 = atomic_load_explicit((&kernelContext_130)->atom_0+(at_16 + 6U), memory_order_relaxed);
    float t_in_7 = (as_type<float>((_S945)));
    uint entry_2 = atomic_load_explicit((&kernelContext_130)->atom_0+(at_16 + 8U), memory_order_relaxed);
    uint _S946 = strike_at_0(b_41, &kernelContext_130);
    if(!(t_in_7 <= ((&kernelContext_130)->params_0->dt_0)))
    {
        _S942 = true;
    }
    else
    {
        _S942 = entry_2 == 4294967295U;
    }
    if(_S942)
    {
        (*((&kernelContext_130)->frames_0+_S946))[int(3)] = 0.0f;
        return;
    }
    float3 rel_4 = (float4(*((&kernelContext_130)->frames_0+(_S946 + 1U))) ).xyz;
    float speed_14 = length(rel_4);
    float4 _S947 = float4(*((&kernelContext_130)->shots_0+2U * shot_7)) ;
    float _S948 = max(_S947.z, 0.00100000004749745f);
    float m_imp_0 = _S947.w;
    float e_imp_0 = (float4(*((&kernelContext_130)->shots_0+(2U * shot_7 + 1U))) ).x;
    float3 contact_16 = (float4(*((&kernelContext_130)->frames_0+_S946)) ).xyz + rel_4 * float3(t_in_7)  + rel_4 / float3(max(speed_14, 9.999999960041972e-13f))  * float3(_S947.y) ;
    float inv_mass_5 = (float4(*((&kernelContext_130)->bodies_0+7U * b_41)) ).w;
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
    float modulus_1 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_130)->geometry_0+(3U * entry_2 + 1U))) ).w);
    *((&kernelContext_130)->frames_0+(_S946 + 2U)) = packed_float4(float4(contact_16, m_eff_6 * speed_14 / (2.86999988555908203f * pow(m_eff_6 * m_eff_6 / (_S948 * modulus_1 * modulus_1 * max(speed_14, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

bool strike_share_breaks_0(uint4 bond_0, float3 contact_17, float force_2, KernelContext_0 thread* kernelContext_131)
{
    thread Quat_0 q_24;
    float3 _S949 = chunk_box_world_0(bond_0.x, &q_24, kernelContext_131);
    float3 _S950 = chunk_box_world_0(bond_0.y, &q_24, kernelContext_131);
    float dist_1 = length(float3(0.5f)  * (_S949 + _S950) - contact_17);
    return (force_2 * min(1.0f, (as_type<float>((bond_0.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((bond_0.z)));
}

[[kernel]] void fr_strike_breaks(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], atomic<uint> device* atom_31 [[buffer(5)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_31 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], packed_float4 device* lp_31 [[buffer(24)]], packed_float4 device* ext_33 [[buffer(23)]], packed_float4 device* engine_contacts_31 [[buffer(25)]], CrushParams_0 constant* crush_31 [[buffer(28)]], packed_float4 device* crush_rw_31 [[buffer(27)]], packed_float4 device* crush_tab_31 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_132;
    (&kernelContext_132)->params_0 = params_31;
    (&kernelContext_132)->work_0 = work_31;
    (&kernelContext_132)->frames_0 = frames_31;
    (&kernelContext_132)->bodies_0 = bodies_31;
    (&kernelContext_132)->chunks_0 = chunks_31;
    (&kernelContext_132)->sort_0 = sort_31;
    (&kernelContext_132)->atom_0 = atom_31;
    (&kernelContext_132)->manifolds_0 = manifolds_31;
    (&kernelContext_132)->loads_0 = loads_31;
    (&kernelContext_132)->bonds_0 = bonds_31;
    (&kernelContext_132)->args_0 = args_31;
    (&kernelContext_132)->info_0 = info_31;
    (&kernelContext_132)->scan_0 = scan_31;
    (&kernelContext_132)->parts_0 = parts_31;
    (&kernelContext_132)->colliders_0 = colliders_31;
    (&kernelContext_132)->collider_info_0 = collider_info_31;
    (&kernelContext_132)->view_chunks_0 = view_chunks_31;
    (&kernelContext_132)->view_0 = view_31;
    (&kernelContext_132)->objects_0 = objects_31;
    (&kernelContext_132)->shot_bodies_0 = shot_bodies_31;
    (&kernelContext_132)->lines_0 = lines_31;
    (&kernelContext_132)->geometry_0 = geometry_31;
    (&kernelContext_132)->shots_0 = shots_31;
    (&kernelContext_132)->lp_0 = lp_31;
    (&kernelContext_132)->ext_0 = ext_33;
    (&kernelContext_132)->engine_contacts_0 = engine_contacts_31;
    (&kernelContext_132)->crush_0 = crush_31;
    (&kernelContext_132)->crush_rw_0 = crush_rw_31;
    (&kernelContext_132)->crush_tab_0 = crush_tab_31;
    threadgroup array<uint, int(1024)> g_scan_31;
    (&kernelContext_132)->g_scan_0 = &g_scan_31;
    threadgroup array<float, int(64)> g_il_best_31;
    (&kernelContext_132)->g_il_best_0 = &g_il_best_31;
    uint i_12 = id_28.x;
    bool _S951;
    if(i_12 >= (params_31->bond_count_0))
    {
        _S951 = true;
    }
    else
    {
        _S951 = ((&kernelContext_132)->params_0->shot_count_0) == 0U;
    }
    if(_S951)
    {
        _S951 = true;
    }
    else
    {
        uint _S952 = w_alive_0(&kernelContext_132);
        _S951 = (*((&kernelContext_132)->work_0+(_S952 + i_12))) == 0U;
    }
    if(_S951)
    {
        return;
    }
    uint4 _S953 = uint4(*((&kernelContext_132)->bonds_0+i_12)) ;
    uint _S954 = body_of_0(_S953.x, &kernelContext_132);
    thread uint shot_8;
    bool _S955 = struck_0(_S954, &shot_8, &kernelContext_132);
    if(!_S955)
    {
        return;
    }
    uint _S956 = strike_at_0(_S954, &kernelContext_132);
    float4 _S957 = float4(*((&kernelContext_132)->frames_0+(_S956 + 2U))) ;
    bool _S958 = strike_share_breaks_0(_S953, _S957.xyz, _S957.w, &kernelContext_132);
    if(_S958)
    {
        uint _S959 = w_alive_0(&kernelContext_132);
        *((&kernelContext_132)->work_0+(_S959 + i_12)) = 0U;
        uint _S960 = a_dirty_0(&kernelContext_132);
        uint _S961 = atomic_fetch_or_explicit((&kernelContext_132)->atom_0+(_S960 + _S954), 1U, memory_order_relaxed);
        uint _S962 = a_counters_1(&kernelContext_132);
        uint _S963 = atomic_fetch_add_explicit((&kernelContext_132)->atom_0+_S962, 1U, memory_order_relaxed);
        uint _S964 = atomic_fetch_add_explicit((&kernelContext_132)->atom_0+(_S962 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_lp_0(uint b_42, KernelContext_0 thread* kernelContext_133)
{
    uint _S965 = a_strike_0(kernelContext_133);
    return _S965 + 9U * kernelContext_133->params_0->body_count_0 + 14U * b_42;
}

[[kernel]] void fr_lp_clear(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], atomic<uint> device* atom_32 [[buffer(5)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_32 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], packed_float4 device* lp_32 [[buffer(24)]], packed_float4 device* ext_34 [[buffer(23)]], packed_float4 device* engine_contacts_32 [[buffer(25)]], CrushParams_0 constant* crush_32 [[buffer(28)]], packed_float4 device* crush_rw_32 [[buffer(27)]], packed_float4 device* crush_tab_32 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_134;
    (&kernelContext_134)->params_0 = params_32;
    (&kernelContext_134)->work_0 = work_32;
    (&kernelContext_134)->frames_0 = frames_32;
    (&kernelContext_134)->bodies_0 = bodies_32;
    (&kernelContext_134)->chunks_0 = chunks_32;
    (&kernelContext_134)->sort_0 = sort_32;
    (&kernelContext_134)->atom_0 = atom_32;
    (&kernelContext_134)->manifolds_0 = manifolds_32;
    (&kernelContext_134)->loads_0 = loads_32;
    (&kernelContext_134)->bonds_0 = bonds_32;
    (&kernelContext_134)->args_0 = args_32;
    (&kernelContext_134)->info_0 = info_32;
    (&kernelContext_134)->scan_0 = scan_32;
    (&kernelContext_134)->parts_0 = parts_32;
    (&kernelContext_134)->colliders_0 = colliders_32;
    (&kernelContext_134)->collider_info_0 = collider_info_32;
    (&kernelContext_134)->view_chunks_0 = view_chunks_32;
    (&kernelContext_134)->view_0 = view_32;
    (&kernelContext_134)->objects_0 = objects_32;
    (&kernelContext_134)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_134)->lines_0 = lines_32;
    (&kernelContext_134)->geometry_0 = geometry_32;
    (&kernelContext_134)->shots_0 = shots_32;
    (&kernelContext_134)->lp_0 = lp_32;
    (&kernelContext_134)->ext_0 = ext_34;
    (&kernelContext_134)->engine_contacts_0 = engine_contacts_32;
    (&kernelContext_134)->crush_0 = crush_32;
    (&kernelContext_134)->crush_rw_0 = crush_rw_32;
    (&kernelContext_134)->crush_tab_0 = crush_tab_32;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_134)->g_scan_0 = &g_scan_32;
    threadgroup array<float, int(64)> g_il_best_32;
    (&kernelContext_134)->g_il_best_0 = &g_il_best_32;
    uint b_43 = id_29.x;
    if(b_43 >= (params_32->body_count_0))
    {
        return;
    }
    uint _S966 = a_lp_0(b_43, &kernelContext_134);
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
        atomic_store_explicit((&kernelContext_134)->atom_0+(_S966 + w_7), 0U, memory_order_relaxed);
        w_7 = w_7 + 1U;
    }
    atomic_store_explicit((&kernelContext_134)->atom_0+(_S966 + 12U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_134)->atom_0+(_S966 + 13U), 0U, memory_order_relaxed);
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

void add64w_0(uint at_17, float value_3, float scale_3, KernelContext_0 thread* kernelContext_135)
{
    float x_6 = round(value_3 * scale_3);
    float x_7;
    if(!((abs(x_6)) < 3.99999993722699776e+18f))
    {
        uint _S967 = a_counters_1(kernelContext_135);
        uint _S968 = atomic_fetch_or_explicit(kernelContext_135->atom_0+(_S967 + 3U), 2U, memory_order_relaxed);
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
        uint _S969 = ~hi_4;
        if(lo_6 == 0U)
        {
            lo_5 = 1U;
        }
        else
        {
            lo_5 = 0U;
        }
        uint _S970 = _S969 + lo_5;
        lo_5 = lo_6;
        hi_5 = _S970;
    }
    else
    {
        lo_5 = lo_4;
        hi_5 = hi_4;
    }
    uint old_1 = atomic_fetch_add_explicit(kernelContext_135->atom_0+at_17, lo_5, memory_order_relaxed);
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
        uint _S971 = atomic_fetch_add_explicit(kernelContext_135->atom_0+(at_17 + 1U), hi_6, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_lp_sum(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], atomic<uint> device* atom_33 [[buffer(5)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], packed_float4 device* lp_33 [[buffer(24)]], packed_float4 device* ext_35 [[buffer(23)]], packed_float4 device* engine_contacts_33 [[buffer(25)]], CrushParams_0 constant* crush_33 [[buffer(28)]], packed_float4 device* crush_rw_33 [[buffer(27)]], packed_float4 device* crush_tab_33 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_136;
    (&kernelContext_136)->params_0 = params_33;
    (&kernelContext_136)->work_0 = work_33;
    (&kernelContext_136)->frames_0 = frames_33;
    (&kernelContext_136)->bodies_0 = bodies_33;
    (&kernelContext_136)->chunks_0 = chunks_33;
    (&kernelContext_136)->sort_0 = sort_33;
    (&kernelContext_136)->atom_0 = atom_33;
    (&kernelContext_136)->manifolds_0 = manifolds_33;
    (&kernelContext_136)->loads_0 = loads_33;
    (&kernelContext_136)->bonds_0 = bonds_33;
    (&kernelContext_136)->args_0 = args_33;
    (&kernelContext_136)->info_0 = info_33;
    (&kernelContext_136)->scan_0 = scan_33;
    (&kernelContext_136)->parts_0 = parts_33;
    (&kernelContext_136)->colliders_0 = colliders_33;
    (&kernelContext_136)->collider_info_0 = collider_info_33;
    (&kernelContext_136)->view_chunks_0 = view_chunks_33;
    (&kernelContext_136)->view_0 = view_33;
    (&kernelContext_136)->objects_0 = objects_33;
    (&kernelContext_136)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_136)->lines_0 = lines_33;
    (&kernelContext_136)->geometry_0 = geometry_33;
    (&kernelContext_136)->shots_0 = shots_33;
    (&kernelContext_136)->lp_0 = lp_33;
    (&kernelContext_136)->ext_0 = ext_35;
    (&kernelContext_136)->engine_contacts_0 = engine_contacts_33;
    (&kernelContext_136)->crush_0 = crush_33;
    (&kernelContext_136)->crush_rw_0 = crush_rw_33;
    (&kernelContext_136)->crush_tab_0 = crush_tab_33;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_136)->g_scan_0 = &g_scan_33;
    threadgroup array<float, int(64)> g_il_best_33;
    (&kernelContext_136)->g_il_best_0 = &g_il_best_33;
    uint k_34 = id_30.x;
    if(k_34 >= (params_33->chunk_count_0))
    {
        return;
    }
    uint _S972 = body_of_0(k_34, &kernelContext_136);
    thread Quat_0 q_25;
    thread float3 origin_7;
    structure_pose_0(_S972, 7U * _S972, &q_25, &origin_7, &kernelContext_136);
    thread Quat_0 _S973 = q_25;
    Quat_0 _S974 = conj_0(&_S973);
    float4 _S975 = float4(*((&kernelContext_136)->chunks_0+4U * k_34)) ;
    uint _S976 = l_vec_0(&kernelContext_136);
    uint _S977 = _S976 + 2U * k_34;
    float3 _S978 = (float4(*((&kernelContext_136)->loads_0+_S977)) ).xyz;
    thread Quat_0 _S979 = _S974;
    float3 _S980 = rotate_0(&_S979, _S978);
    float _S981 = _S975.w;
    float3 _S982 = float3((&kernelContext_136)->params_0->gx_0, (&kernelContext_136)->params_0->gy_0, (&kernelContext_136)->params_0->gz_0);
    thread Quat_0 _S983 = _S974;
    float3 _S984 = rotate_0(&_S983, _S982);
    float3 f_10 = _S980 + float3(_S981)  * _S984;
    float3 _S985 = (float4(*((&kernelContext_136)->loads_0+(_S977 + 1U))) ).xyz;
    thread Quat_0 _S986 = _S974;
    float3 _S987 = rotate_0(&_S986, _S985);
    uint _S988 = lp_f_0(k_34);
    *((&kernelContext_136)->lp_0+_S988) = packed_float4(float4(f_10, 0.0f)) ;
    *((&kernelContext_136)->lp_0+(_S988 + 1U)) = packed_float4(float4(_S987, 0.0f)) ;
    float3 _S989 = _S987 + cross(_S975.xyz - (float4(*((&kernelContext_136)->frames_0+2U * _S972)) ).xyz, f_10);
    uint _S990 = a_lp_0(_S972, &kernelContext_136);
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
        uint _S991 = 2U * a_8;
        add64w_0(_S990 + _S991, f_10[a_8], (&kernelContext_136)->params_0->lp_scale_0, &kernelContext_136);
        add64w_0(_S990 + 6U + _S991, _S989[a_8], (&kernelContext_136)->params_0->lp_scale_0, &kernelContext_136);
        a_8 = a_8 + 1U;
    }
    uint _S992 = atomic_fetch_min_explicit((&kernelContext_136)->atom_0+(_S990 + 12U), k_34, memory_order_relaxed);
    if(((&kernelContext_136)->info_0[k_34] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_136)->atom_0+(_S990 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

float3 read64w3_0(uint at_18, float scale_4, KernelContext_0 thread* kernelContext_137)
{
    float _S993 = read64_1(at_18, scale_4, kernelContext_137);
    float _S994 = read64_1(at_18 + 2U, scale_4, kernelContext_137);
    float _S995 = read64_1(at_18 + 4U, scale_4, kernelContext_137);
    return float3(_S993, _S994, _S995);
}

[[kernel]] void fr_lp_relief(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], atomic<uint> device* atom_34 [[buffer(5)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], packed_float4 device* lp_34 [[buffer(24)]], packed_float4 device* ext_36 [[buffer(23)]], packed_float4 device* engine_contacts_34 [[buffer(25)]], CrushParams_0 constant* crush_34 [[buffer(28)]], packed_float4 device* crush_rw_34 [[buffer(27)]], packed_float4 device* crush_tab_34 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_138;
    (&kernelContext_138)->params_0 = params_34;
    (&kernelContext_138)->work_0 = work_34;
    (&kernelContext_138)->frames_0 = frames_34;
    (&kernelContext_138)->bodies_0 = bodies_34;
    (&kernelContext_138)->chunks_0 = chunks_34;
    (&kernelContext_138)->sort_0 = sort_34;
    (&kernelContext_138)->atom_0 = atom_34;
    (&kernelContext_138)->manifolds_0 = manifolds_34;
    (&kernelContext_138)->loads_0 = loads_34;
    (&kernelContext_138)->bonds_0 = bonds_34;
    (&kernelContext_138)->args_0 = args_34;
    (&kernelContext_138)->info_0 = info_34;
    (&kernelContext_138)->scan_0 = scan_34;
    (&kernelContext_138)->parts_0 = parts_34;
    (&kernelContext_138)->colliders_0 = colliders_34;
    (&kernelContext_138)->collider_info_0 = collider_info_34;
    (&kernelContext_138)->view_chunks_0 = view_chunks_34;
    (&kernelContext_138)->view_0 = view_34;
    (&kernelContext_138)->objects_0 = objects_34;
    (&kernelContext_138)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_138)->lines_0 = lines_34;
    (&kernelContext_138)->geometry_0 = geometry_34;
    (&kernelContext_138)->shots_0 = shots_34;
    (&kernelContext_138)->lp_0 = lp_34;
    (&kernelContext_138)->ext_0 = ext_36;
    (&kernelContext_138)->engine_contacts_0 = engine_contacts_34;
    (&kernelContext_138)->crush_0 = crush_34;
    (&kernelContext_138)->crush_rw_0 = crush_rw_34;
    (&kernelContext_138)->crush_tab_0 = crush_tab_34;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_138)->g_scan_0 = &g_scan_34;
    threadgroup array<float, int(64)> g_il_best_34;
    (&kernelContext_138)->g_il_best_0 = &g_il_best_34;
    uint k_35 = id_31.x;
    if(k_35 >= (params_34->chunk_count_0))
    {
        return;
    }
    uint _S996 = body_of_0(k_35, &kernelContext_138);
    uint _S997 = a_lp_0(_S996, &kernelContext_138);
    uint _S998 = atomic_load_explicit((&kernelContext_138)->atom_0+(_S997 + 13U), memory_order_relaxed);
    bool supported_0 = _S998 != 0U;
    bool fixed_1;
    if(((&kernelContext_138)->info_0[k_35] & 1U) != 0U)
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            uint _S999 = atomic_load_explicit((&kernelContext_138)->atom_0+(_S997 + 12U), memory_order_relaxed);
            fixed_1 = _S999 == k_35;
        }
        else
        {
            fixed_1 = false;
        }
    }
    uint _S1000 = lp_f_0(k_35);
    thread float4 f_11 = float4(*((&kernelContext_138)->lp_0+_S1000)) ;
    uint _S1001 = _S1000 + 1U;
    thread float4 t_11 = float4(*((&kernelContext_138)->lp_0+_S1001)) ;
    uint b_44 = 7U * _S996;
    float inv_mass_6 = (float4(*((&kernelContext_138)->bodies_0+b_44)) ).w;
    bool _S1002;
    if(!supported_0)
    {
        _S1002 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S1002 = false;
    }
    if(_S1002)
    {
        float3 _S1003 = read64w3_0(_S997, (&kernelContext_138)->params_0->lp_scale_0, &kernelContext_138);
        float3 _S1004 = read64w3_0(_S997 + 6U, (&kernelContext_138)->params_0->lp_scale_0, &kernelContext_138);
        uint _S1005 = 2U * _S996;
        Quat_0 p_15 = quat_of_0(float4(*((&kernelContext_138)->frames_0+(_S1005 + 1U))) );
        float3 _S1006 = (float4(*((&kernelContext_138)->bodies_0+(b_44 + 2U))) ).xyz;
        thread Quat_0 _S1007 = p_15;
        Quat_0 _S1008 = conj_0(&_S1007);
        thread Quat_0 _S1009 = _S1008;
        float3 _S1010 = rotate_0(&_S1009, _S1004);
        float3 _S1011 = _S1006 * _S1010;
        thread Quat_0 _S1012 = p_15;
        float3 _S1013 = rotate_0(&_S1012, _S1011);
        uint _S1014 = 4U * k_35;
        float4 _S1015 = float4(*((&kernelContext_138)->chunks_0+_S1014)) ;
        float4 _S1016 = float4(*((&kernelContext_138)->chunks_0+(_S1014 + 1U))) ;
        float4 _S1017 = float4(*((&kernelContext_138)->chunks_0+(_S1014 + 2U))) ;
        f_11.xyz = f_11.xyz - float3(_S1015.w)  * (_S1003 * float3(inv_mass_6)  + cross(_S1013, _S1015.xyz - (float4(*((&kernelContext_138)->frames_0+_S1005)) ).xyz));
        float _S1018 = _S1013.x;
        float _S1019 = _S1016.w;
        float _S1020 = _S1013.y;
        float _S1021 = _S1017.x;
        float _S1022 = _S1013.z;
        float _S1023 = _S1017.y;
        t_11.xyz = t_11.xyz - float3(_S1016.x * _S1018 + _S1019 * _S1020 + _S1021 * _S1022, _S1019 * _S1018 + _S1016.y * _S1020 + _S1023 * _S1022, _S1021 * _S1018 + _S1023 * _S1020 + _S1016.z * _S1022);
    }
    float _S1024;
    if(fixed_1)
    {
        _S1024 = 1.0f;
    }
    else
    {
        _S1024 = 0.0f;
    }
    f_11.w = _S1024;
    *((&kernelContext_138)->lp_0+_S1000) = packed_float4(f_11) ;
    *((&kernelContext_138)->lp_0+_S1001) = packed_float4(t_11) ;
    return;
}

uint adj_bond_0(uint at_19, KernelContext_0 thread* kernelContext_139)
{
    return (as_type<uint>(((float4(*(kernelContext_139->ext_0+(kernelContext_139->params_0->ext_adj_0 + at_19 / 4U))) )[at_19 % 4U])));
}

uint lp_intact_0(uint k_36, KernelContext_0 thread* kernelContext_140)
{
    float4 _S1025 = float4(*(kernelContext_140->ext_0+(kernelContext_140->params_0->ext_chunks_0 + 4U * k_36))) ;
    uint _S1026 = (as_type<uint>((_S1025.x)));
    uint _S1027 = (as_type<uint>((_S1025.y)));
    uint e_4 = 0U;
    uint intact_0 = 0U;
    for(;;)
    {
        if(e_4 < _S1027)
        {
        }
        else
        {
            break;
        }
        uint _S1028 = w_alive_0(kernelContext_140);
        uint _S1029 = adj_bond_0(_S1026 + e_4, kernelContext_140);
        uint _S1030;
        if((*(kernelContext_140->work_0+(_S1028 + _S1029))) != 0U)
        {
            _S1030 = 1U;
        }
        else
        {
            _S1030 = 0U;
        }
        uint intact_1 = intact_0 + _S1030;
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

LpBond_0 lp_bond_0(uint i_13, KernelContext_0 thread* kernelContext_141)
{
    uint o_7 = 10U * i_13;
    thread LpBond_0 b_45;
    (&b_45)->centroid_0 = (float4(*(kernelContext_141->ext_0+o_7)) ).xyz;
    float3 _S1031 = (float4(*(kernelContext_141->ext_0+(o_7 + 1U))) ).xyz;
    (&b_45)->n_4 = _S1031;
    float3 _S1032 = (float4(*(kernelContext_141->ext_0+(o_7 + 2U))) ).xyz;
    (&b_45)->t1_2 = _S1032;
    (&b_45)->t2_0 = cross(_S1031, _S1032);
    (&(&b_45)->law_0)->geom0_0 = float4(*(kernelContext_141->ext_0+(o_7 + 3U))) ;
    (&(&b_45)->law_0)->geom1_0 = float4(*(kernelContext_141->ext_0+(o_7 + 4U))) ;
    (&(&b_45)->law_0)->stiff0_0 = float4(*(kernelContext_141->ext_0+(o_7 + 5U))) ;
    (&(&b_45)->law_0)->stiff1_0 = float4(*(kernelContext_141->ext_0+(o_7 + 6U))) ;
    (&(&b_45)->law_0)->rebar0_0 = float4(*(kernelContext_141->ext_0+(o_7 + 7U))) ;
    (&(&b_45)->law_0)->rebar1_0 = float4(*(kernelContext_141->ext_0+(o_7 + 8U))) ;
    (&(&b_45)->law_0)->ids_3 = (as_type<uint4>((float4(*(kernelContext_141->ext_0+(o_7 + 9U))) )));
    return b_45;
}

uint tri_0(uint i_14, uint j_6)
{
    return i_14 * (i_14 + 1U) / 2U + j_6;
}

bool lp_factor_0(uint k_38, KernelContext_0 thread* kernelContext_142)
{
    uint q_26;
    uint a_9;
    uint p2_0;
    uint q2_0;
    uint a2_0;
    float s_11;
    uint _S1033 = lp_intact_0(k_38, kernelContext_142);
    uint _S1034 = lp_l_0(k_38);
    uint _S1035 = _S1033 + 1U;
    if((as_type<uint>(((float4(*(kernelContext_142->lp_0+(_S1034 + 5U))) ).w))) == _S1035)
    {
        return true;
    }
    thread array<array<float, int(6)>, int(6)> d_21;
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
        q_26 = 0U;
        for(;;)
        {
            if(q_26 < 6U)
            {
            }
            else
            {
                break;
            }
            d_21[p_16][q_26] = 0.0f;
            q_26 = q_26 + 1U;
        }
        p_16 = p_16 + 1U;
    }
    uint _S1036 = 4U * k_38;
    float4 _S1037 = float4(*(kernelContext_142->ext_0+(kernelContext_142->params_0->ext_chunks_0 + _S1036))) ;
    uint _S1038 = (as_type<uint>((_S1037.x)));
    uint _S1039 = (as_type<uint>((_S1037.y)));
    q_26 = 0U;
    for(;;)
    {
        if(q_26 < _S1039)
        {
        }
        else
        {
            break;
        }
        uint _S1040 = adj_bond_0(_S1038 + q_26, kernelContext_142);
        uint _S1041 = w_alive_0(kernelContext_142);
        if((*(kernelContext_142->work_0+(_S1041 + _S1040))) == 0U)
        {
            q_26 = q_26 + 1U;
            continue;
        }
        LpBond_0 _S1042 = lp_bond_0(_S1040, kernelContext_142);
        float3 _S1043 = _S1042.centroid_0 - (float4(*(kernelContext_142->chunks_0+_S1036)) ).xyz;
        float _S1044 = _S1042.law_0.stiff0_0.y;
        array<float, int(6)> _S1045 = { { _S1044, _S1044, _S1042.law_0.stiff0_0.x, _S1042.law_0.stiff0_0.z, _S1042.law_0.stiff0_0.w, _S1042.law_0.stiff1_0.x } };
        array<float3, int(3)> _S1046 = { { _S1042.t1_2, _S1042.t2_0, _S1042.n_4 } };
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
            float3 c_19 = cross(_S1043, _S1046[a_9]);
            float _S1047 = _S1046[a_9].x;
            j_7[a_9][int(0)] = _S1047;
            float _S1048 = _S1046[a_9].y;
            j_7[a_9][int(1)] = _S1048;
            float _S1049 = _S1046[a_9].z;
            j_7[a_9][int(2)] = _S1049;
            j_7[a_9][int(3)] = c_19.x;
            j_7[a_9][int(4)] = c_19.y;
            j_7[a_9][int(5)] = c_19.z;
            j_7[3U + a_9][int(0)] = 0.0f;
            j_7[3U + a_9][int(1)] = 0.0f;
            j_7[3U + a_9][int(2)] = 0.0f;
            j_7[3U + a_9][int(3)] = _S1047;
            j_7[3U + a_9][int(4)] = _S1048;
            j_7[3U + a_9][int(5)] = _S1049;
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
                    float s_12 = s_11 + j_7[a2_0][p2_0] * _S1045[a2_0] * j_7[a2_0][q2_0];
                    a2_0 = a2_0 + 1U;
                    s_11 = s_12;
                }
                d_21[p2_0][q2_0] = d_21[p2_0][q2_0] + s_11;
                q2_0 = q2_0 + 1U;
            }
            p2_0 = p2_0 + 1U;
        }
        q_26 = q_26 + 1U;
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
            s_11 = d_21[p2_0][q2_0];
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
                bool _S1050;
                if(ok_1)
                {
                    _S1050 = s_11 > 0.0f;
                }
                else
                {
                    _S1050 = false;
                }
                l_0[tri_0(p2_0, p2_0)] = sqrt(max(s_11, 1.00000000317107685e-30f));
                ok_1 = _S1050;
            }
            else
            {
                l_0[tri_0(p2_0, q2_0)] = s_11 / l_0[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + 1U;
        }
        uint _S1051 = p2_0 + 1U;
        ok_0 = ok_1;
        p2_0 = _S1051;
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
        packed_float4 device* _S1052 = kernelContext_142->lp_0+(_S1034 + q2_0);
        uint _S1053 = 4U * q2_0;
        float _S1054 = l_0[_S1053];
        uint _S1055 = _S1053 + 1U;
        if(_S1055 < 21U)
        {
            s_11 = l_0[_S1055];
        }
        else
        {
            s_11 = 0.0f;
        }
        uint _S1056 = _S1053 + 2U;
        float _S1057;
        if(_S1056 < 21U)
        {
            _S1057 = l_0[_S1056];
        }
        else
        {
            _S1057 = 0.0f;
        }
        float _S1058;
        if(q2_0 == 5U)
        {
            _S1058 = (as_type<float>((_S1035)));
        }
        else
        {
            _S1058 = l_0[_S1053 + 3U];
        }
        *_S1052 = packed_float4(float4(_S1054, s_11, _S1057, _S1058)) ;
        q2_0 = q2_0 + 1U;
    }
    return true;
}

[[kernel]] void fr_lp_block(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], atomic<uint> device* atom_35 [[buffer(5)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], packed_float4 device* lp_35 [[buffer(24)]], packed_float4 device* ext_37 [[buffer(23)]], packed_float4 device* engine_contacts_35 [[buffer(25)]], CrushParams_0 constant* crush_35 [[buffer(28)]], packed_float4 device* crush_rw_35 [[buffer(27)]], packed_float4 device* crush_tab_35 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_143;
    (&kernelContext_143)->params_0 = params_35;
    (&kernelContext_143)->work_0 = work_35;
    (&kernelContext_143)->frames_0 = frames_35;
    (&kernelContext_143)->bodies_0 = bodies_35;
    (&kernelContext_143)->chunks_0 = chunks_35;
    (&kernelContext_143)->sort_0 = sort_35;
    (&kernelContext_143)->atom_0 = atom_35;
    (&kernelContext_143)->manifolds_0 = manifolds_35;
    (&kernelContext_143)->loads_0 = loads_35;
    (&kernelContext_143)->bonds_0 = bonds_35;
    (&kernelContext_143)->args_0 = args_35;
    (&kernelContext_143)->info_0 = info_35;
    (&kernelContext_143)->scan_0 = scan_35;
    (&kernelContext_143)->parts_0 = parts_35;
    (&kernelContext_143)->colliders_0 = colliders_35;
    (&kernelContext_143)->collider_info_0 = collider_info_35;
    (&kernelContext_143)->view_chunks_0 = view_chunks_35;
    (&kernelContext_143)->view_0 = view_35;
    (&kernelContext_143)->objects_0 = objects_35;
    (&kernelContext_143)->shot_bodies_0 = shot_bodies_35;
    (&kernelContext_143)->lines_0 = lines_35;
    (&kernelContext_143)->geometry_0 = geometry_35;
    (&kernelContext_143)->shots_0 = shots_35;
    (&kernelContext_143)->lp_0 = lp_35;
    (&kernelContext_143)->ext_0 = ext_37;
    (&kernelContext_143)->engine_contacts_0 = engine_contacts_35;
    (&kernelContext_143)->crush_0 = crush_35;
    (&kernelContext_143)->crush_rw_0 = crush_rw_35;
    (&kernelContext_143)->crush_tab_0 = crush_tab_35;
    threadgroup array<uint, int(1024)> g_scan_35;
    (&kernelContext_143)->g_scan_0 = &g_scan_35;
    threadgroup array<float, int(64)> g_il_best_35;
    (&kernelContext_143)->g_il_best_0 = &g_il_best_35;
    uint k_39 = id_32.x;
    bool _S1059;
    if(k_39 >= (params_35->chunk_count_0))
    {
        _S1059 = true;
    }
    else
    {
        _S1059 = ((float4(*((&kernelContext_143)->lp_0+lp_f_0(k_39))) ).w) != 0.0f;
    }
    if(_S1059)
    {
        return;
    }
    bool _S1060 = lp_factor_0(k_39, &kernelContext_143);
    if(!_S1060)
    {
        (*((&kernelContext_143)->lp_0+lp_f_0(k_39)))[int(3)] = 1.0f;
        uint _S1061 = a_counters_1(&kernelContext_143);
        uint _S1062 = atomic_fetch_or_explicit((&kernelContext_143)->atom_0+(_S1061 + 3U), 64U, memory_order_relaxed);
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

float4 lp_side_0(uint i_15, KernelContext_0 thread* kernelContext_144)
{
    return float4(*(kernelContext_144->ext_0+(kernelContext_144->params_0->ext_sides_0 + i_15))) ;
}

void st_sweep_0(uint k_42, uint state_3, uint from_1, float c1_1, float c2_1, KernelContext_0 thread* kernelContext_145)
{
    uint ka_0;
    uint kb_0;
    float s_14;
    uint to_1 = 1U - from_1;
    uint _S1063 = st_x_0(k_42, state_3, from_1);
    float4 _S1064 = float4(*(kernelContext_145->lp_0+_S1063)) ;
    float4 _S1065 = float4(*(kernelContext_145->lp_0+(_S1063 + 1U))) ;
    uint _S1066 = st_f_0(k_42, state_3);
    float4 _S1067 = float4(*(kernelContext_145->lp_0+_S1066)) ;
    if((_S1067.w) != 0.0f)
    {
        uint _S1068 = st_x_0(k_42, state_3, to_1);
        *(kernelContext_145->lp_0+_S1068) = packed_float4(_S1064) ;
        *(kernelContext_145->lp_0+(_S1068 + 1U)) = packed_float4(_S1065) ;
        uint _S1069 = st_d_0(k_42, state_3);
        packed_float4 _S1070 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_145->lp_0+_S1069) = _S1070;
        *(kernelContext_145->lp_0+(_S1069 + 1U)) = _S1070;
        return;
    }
    float _S1071 = _S1067.y;
    float _S1072 = _S1067.z;
    thread array<float, int(6)> r_23;
    r_23[int(0)] = _S1067.x;
    r_23[int(1)] = _S1071;
    r_23[int(2)] = _S1072;
    r_23[int(3)] = 0.0f;
    r_23[int(4)] = 0.0f;
    r_23[int(5)] = 0.0f;
    float3 t_12 = (float4(*(kernelContext_145->lp_0+(_S1066 + 1U))) ).xyz;
    r_23[int(3)] = t_12.x;
    r_23[int(4)] = t_12.y;
    r_23[int(5)] = t_12.z;
    float4 _S1073 = float4(*(kernelContext_145->ext_0+(kernelContext_145->params_0->ext_chunks_0 + 4U * k_42))) ;
    uint _S1074 = (as_type<uint>((_S1073.x)));
    uint _S1075 = (as_type<uint>((_S1073.y)));
    uint e_5 = 0U;
    for(;;)
    {
        if(e_5 < _S1075)
        {
        }
        else
        {
            break;
        }
        uint _S1076 = 5U * (_S1074 + e_5);
        float4 _S1077 = lp_side_0(_S1076, kernelContext_145);
        uint _S1078 = w_alive_0(kernelContext_145);
        if((*(kernelContext_145->work_0+(_S1078 + (as_type<uint>((_S1077.w)))))) == 0U)
        {
            e_5 = e_5 + 1U;
            continue;
        }
        float4 _S1079 = lp_side_0(_S1076 + 1U, kernelContext_145);
        float4 _S1080 = lp_side_0(_S1076 + 2U, kernelContext_145);
        float4 _S1081 = lp_side_0(_S1076 + 3U, kernelContext_145);
        float4 _S1082 = lp_side_0(_S1076 + 4U, kernelContext_145);
        bool first_side_0 = (_S1082.w) != 0.0f;
        if(first_side_0)
        {
            ka_0 = k_42;
        }
        else
        {
            ka_0 = (as_type<uint>((_S1079.w)));
        }
        if(first_side_0)
        {
            kb_0 = (as_type<uint>((_S1079.w)));
        }
        else
        {
            kb_0 = k_42;
        }
        float3 ra_0 = _S1077.xyz;
        float3 rb_0 = _S1079.xyz;
        float3 t1_3 = _S1080.xyz;
        float3 n_5 = _S1081.xyz;
        float3 t2_1 = cross(n_5, t1_3);
        uint _S1083 = st_x_0(ka_0, state_3, from_1);
        uint _S1084 = st_x_0(kb_0, state_3, from_1);
        float3 _S1085 = (float4(*(kernelContext_145->lp_0+(_S1084 + 1U))) ).xyz;
        float3 _S1086 = (float4(*(kernelContext_145->lp_0+(_S1083 + 1U))) ).xyz;
        float3 delta_0 = (float4(*(kernelContext_145->lp_0+_S1084)) ).xyz + cross(_S1085, rb_0) - ((float4(*(kernelContext_145->lp_0+_S1083)) ).xyz + cross(_S1086, ra_0));
        float3 phi_0 = _S1085 - _S1086;
        float _S1087 = _S1080.w;
        float3 fb_0 = t1_3 * float3((_S1087 * dot(delta_0, t1_3)))  + t2_1 * float3((_S1087 * dot(delta_0, t2_1)))  + n_5 * float3((_S1081.w * dot(delta_0, n_5))) ;
        float3 mb_0 = t1_3 * float3((_S1082.x * dot(phi_0, t1_3)))  + t2_1 * float3((_S1082.y * dot(phi_0, t2_1)))  + n_5 * float3((_S1082.z * dot(phi_0, n_5))) ;
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
        float3 _S1088;
        if(first_side_0)
        {
            _S1088 = mb_0;
        }
        else
        {
            _S1088 = - mb_0;
        }
        float3 lm_0 = _S1088 + cross(arm_0, lf_0);
        r_23[int(0)] = r_23[int(0)] + lf_0.x;
        r_23[int(1)] = r_23[int(1)] + lf_0.y;
        r_23[int(2)] = r_23[int(2)] + lf_0.z;
        r_23[int(3)] = r_23[int(3)] + lm_0.x;
        r_23[int(4)] = r_23[int(4)] + lm_0.y;
        r_23[int(5)] = r_23[int(5)] + lm_0.z;
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
        float4 _S1089 = float4(*(kernelContext_145->lp_0+(lp_l_0(k_42) + kb_0))) ;
        uint _S1090 = 4U * kb_0;
        l_1[_S1090] = _S1089.x;
        uint _S1091 = _S1090 + 1U;
        if(_S1091 < 21U)
        {
            l_1[_S1091] = _S1089.y;
        }
        uint _S1092 = _S1090 + 2U;
        if(_S1092 < 21U)
        {
            l_1[_S1092] = _S1089.z;
        }
        uint _S1093 = _S1090 + 3U;
        if(_S1093 < 21U)
        {
            l_1[_S1093] = _S1089.w;
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
        s_14 = r_23[i2_0];
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
        uint _S1094 = uint(i3_0);
        z_3[i3_0] = s_14 / l_1[tri_0(_S1094, _S1094)];
        i3_0 = i3_0 - int(1);
    }
    uint _S1095 = st_d_0(k_42, state_3);
    float4 _S1096 = float4(*(kernelContext_145->lp_0+_S1095)) ;
    thread float4 d0_0 = _S1096;
    uint _S1097 = _S1095 + 1U;
    float4 _S1098 = float4(*(kernelContext_145->lp_0+_S1097)) ;
    thread float4 d1_0 = _S1098;
    float3 _S1099 = float3(c1_1) ;
    float3 _S1100 = float3(c2_1) ;
    d0_0.xyz = _S1099 * _S1096.xyz + _S1100 * float3(z_3[int(0)], z_3[int(1)], z_3[int(2)]);
    d1_0.xyz = _S1099 * _S1098.xyz + _S1100 * float3(z_3[int(3)], z_3[int(4)], z_3[int(5)]);
    *(kernelContext_145->lp_0+_S1095) = packed_float4(d0_0) ;
    *(kernelContext_145->lp_0+_S1097) = packed_float4(d1_0) ;
    uint _S1101 = st_x_0(k_42, state_3, to_1);
    *(kernelContext_145->lp_0+_S1101) = packed_float4(float4(_S1064.xyz + d0_0.xyz, 0.0f)) ;
    *(kernelContext_145->lp_0+(_S1101 + 1U)) = packed_float4(float4(_S1065.xyz + d1_0.xyz, 0.0f)) ;
    return;
}

void lp_sweep_0(uint k_43, uint from_2, KernelContext_0 thread* kernelContext_146)
{
    st_sweep_0(k_43, 0U, from_2, kernelContext_146->params_0->lp_c1_0, kernelContext_146->params_0->lp_c2_0, kernelContext_146);
    return;
}

[[kernel]] void fr_lp_sweep_even(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], atomic<uint> device* atom_36 [[buffer(5)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], packed_float4 device* lp_36 [[buffer(24)]], packed_float4 device* ext_38 [[buffer(23)]], packed_float4 device* engine_contacts_36 [[buffer(25)]], CrushParams_0 constant* crush_36 [[buffer(28)]], packed_float4 device* crush_rw_36 [[buffer(27)]], packed_float4 device* crush_tab_36 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_147;
    (&kernelContext_147)->params_0 = params_36;
    (&kernelContext_147)->work_0 = work_36;
    (&kernelContext_147)->frames_0 = frames_36;
    (&kernelContext_147)->bodies_0 = bodies_36;
    (&kernelContext_147)->chunks_0 = chunks_36;
    (&kernelContext_147)->sort_0 = sort_36;
    (&kernelContext_147)->atom_0 = atom_36;
    (&kernelContext_147)->manifolds_0 = manifolds_36;
    (&kernelContext_147)->loads_0 = loads_36;
    (&kernelContext_147)->bonds_0 = bonds_36;
    (&kernelContext_147)->args_0 = args_36;
    (&kernelContext_147)->info_0 = info_36;
    (&kernelContext_147)->scan_0 = scan_36;
    (&kernelContext_147)->parts_0 = parts_36;
    (&kernelContext_147)->colliders_0 = colliders_36;
    (&kernelContext_147)->collider_info_0 = collider_info_36;
    (&kernelContext_147)->view_chunks_0 = view_chunks_36;
    (&kernelContext_147)->view_0 = view_36;
    (&kernelContext_147)->objects_0 = objects_36;
    (&kernelContext_147)->shot_bodies_0 = shot_bodies_36;
    (&kernelContext_147)->lines_0 = lines_36;
    (&kernelContext_147)->geometry_0 = geometry_36;
    (&kernelContext_147)->shots_0 = shots_36;
    (&kernelContext_147)->lp_0 = lp_36;
    (&kernelContext_147)->ext_0 = ext_38;
    (&kernelContext_147)->engine_contacts_0 = engine_contacts_36;
    (&kernelContext_147)->crush_0 = crush_36;
    (&kernelContext_147)->crush_rw_0 = crush_rw_36;
    (&kernelContext_147)->crush_tab_0 = crush_tab_36;
    threadgroup array<uint, int(1024)> g_scan_36;
    (&kernelContext_147)->g_scan_0 = &g_scan_36;
    threadgroup array<float, int(64)> g_il_best_36;
    (&kernelContext_147)->g_il_best_0 = &g_il_best_36;
    uint _S1102 = id_33.x;
    if(_S1102 < (params_36->chunk_count_0))
    {
        lp_sweep_0(_S1102, 0U, &kernelContext_147);
    }
    return;
}

[[kernel]] void fr_lp_sweep_odd(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], atomic<uint> device* atom_37 [[buffer(5)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], packed_float4 device* lp_37 [[buffer(24)]], packed_float4 device* ext_39 [[buffer(23)]], packed_float4 device* engine_contacts_37 [[buffer(25)]], CrushParams_0 constant* crush_37 [[buffer(28)]], packed_float4 device* crush_rw_37 [[buffer(27)]], packed_float4 device* crush_tab_37 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_148;
    (&kernelContext_148)->params_0 = params_37;
    (&kernelContext_148)->work_0 = work_37;
    (&kernelContext_148)->frames_0 = frames_37;
    (&kernelContext_148)->bodies_0 = bodies_37;
    (&kernelContext_148)->chunks_0 = chunks_37;
    (&kernelContext_148)->sort_0 = sort_37;
    (&kernelContext_148)->atom_0 = atom_37;
    (&kernelContext_148)->manifolds_0 = manifolds_37;
    (&kernelContext_148)->loads_0 = loads_37;
    (&kernelContext_148)->bonds_0 = bonds_37;
    (&kernelContext_148)->args_0 = args_37;
    (&kernelContext_148)->info_0 = info_37;
    (&kernelContext_148)->scan_0 = scan_37;
    (&kernelContext_148)->parts_0 = parts_37;
    (&kernelContext_148)->colliders_0 = colliders_37;
    (&kernelContext_148)->collider_info_0 = collider_info_37;
    (&kernelContext_148)->view_chunks_0 = view_chunks_37;
    (&kernelContext_148)->view_0 = view_37;
    (&kernelContext_148)->objects_0 = objects_37;
    (&kernelContext_148)->shot_bodies_0 = shot_bodies_37;
    (&kernelContext_148)->lines_0 = lines_37;
    (&kernelContext_148)->geometry_0 = geometry_37;
    (&kernelContext_148)->shots_0 = shots_37;
    (&kernelContext_148)->lp_0 = lp_37;
    (&kernelContext_148)->ext_0 = ext_39;
    (&kernelContext_148)->engine_contacts_0 = engine_contacts_37;
    (&kernelContext_148)->crush_0 = crush_37;
    (&kernelContext_148)->crush_rw_0 = crush_rw_37;
    (&kernelContext_148)->crush_tab_0 = crush_tab_37;
    threadgroup array<uint, int(1024)> g_scan_37;
    (&kernelContext_148)->g_scan_0 = &g_scan_37;
    threadgroup array<float, int(64)> g_il_best_37;
    (&kernelContext_148)->g_il_best_0 = &g_il_best_37;
    uint _S1103 = id_34.x;
    if(_S1103 < (params_37->chunk_count_0))
    {
        lp_sweep_0(_S1103, 1U, &kernelContext_148);
    }
    return;
}

void st_force_0(const LpBond_0 thread* b_46, uint ka_1, uint kb_1, uint state_4, uint c_20, float3 thread* q_lin_0, float3 thread* q_ang_0, KernelContext_0 thread* kernelContext_149)
{
    uint _S1104 = st_x_0(ka_1, state_4, c_20);
    uint _S1105 = st_x_0(kb_1, state_4, c_20);
    float3 _S1106 = (float4(*(kernelContext_149->lp_0+(_S1105 + 1U))) ).xyz;
    float3 _S1107 = (float4(*(kernelContext_149->lp_0+(_S1104 + 1U))) ).xyz;
    float3 delta_1 = (float4(*(kernelContext_149->lp_0+_S1105)) ).xyz + cross(_S1106, b_46->centroid_0 - (float4(*(kernelContext_149->chunks_0+4U * kb_1)) ).xyz) - ((float4(*(kernelContext_149->lp_0+_S1104)) ).xyz + cross(_S1107, b_46->centroid_0 - (float4(*(kernelContext_149->chunks_0+4U * ka_1)) ).xyz));
    float3 phi_1 = _S1106 - _S1107;
    float4 _S1108 = (&b_46->law_0)->stiff0_0;
    float kt_0 = (&b_46->law_0)->stiff1_0.x;
    float _S1109 = (&b_46->law_0)->stiff0_0.y;
    float3 _S1110 = b_46->t1_2;
    float3 _S1111 = b_46->t2_0;
    float3 _S1112 = b_46->n_4;
    *q_lin_0 = float3(_S1109 * dot(delta_1, b_46->t1_2), _S1109 * dot(delta_1, b_46->t2_0), (&b_46->law_0)->stiff0_0.x * dot(delta_1, b_46->n_4));
    *q_ang_0 = float3(_S1108.z * dot(phi_1, _S1110), _S1108.w * dot(phi_1, _S1111), kt_0 * dot(phi_1, _S1112));
    return;
}

void st_force_1(const LpBond_0 thread* b_47, uint ka_2, uint kb_2, uint state_5, uint c_21, float3 thread* q_lin_1, float3 thread* q_ang_1, KernelContext_0 thread* kernelContext_150)
{
    uint _S1113 = st_x_0(ka_2, state_5, c_21);
    uint _S1114 = st_x_0(kb_2, state_5, c_21);
    float3 _S1115 = (float4(*(kernelContext_150->lp_0+(_S1114 + 1U))) ).xyz;
    float3 _S1116 = (float4(*(kernelContext_150->lp_0+(_S1113 + 1U))) ).xyz;
    float3 delta_2 = (float4(*(kernelContext_150->lp_0+_S1114)) ).xyz + cross(_S1115, b_47->centroid_0 - (float4(*(kernelContext_150->chunks_0+4U * kb_2)) ).xyz) - ((float4(*(kernelContext_150->lp_0+_S1113)) ).xyz + cross(_S1116, b_47->centroid_0 - (float4(*(kernelContext_150->chunks_0+4U * ka_2)) ).xyz));
    float3 phi_2 = _S1115 - _S1116;
    float4 _S1117 = (&b_47->law_0)->stiff0_0;
    float kt_1 = (&b_47->law_0)->stiff1_0.x;
    float _S1118 = (&b_47->law_0)->stiff0_0.y;
    float3 _S1119 = b_47->t1_2;
    float3 _S1120 = b_47->t2_0;
    float3 _S1121 = b_47->n_4;
    *q_lin_1 = float3(_S1118 * dot(delta_2, b_47->t1_2), _S1118 * dot(delta_2, b_47->t2_0), (&b_47->law_0)->stiff0_0.x * dot(delta_2, b_47->n_4));
    *q_ang_1 = float3(_S1117.z * dot(phi_2, _S1119), _S1117.w * dot(phi_2, _S1120), kt_1 * dot(phi_2, _S1121));
    return;
}

void lp_force_0(const LpBond_0 thread* b_48, uint ka_3, uint kb_3, uint c_22, float3 thread* q_lin_2, float3 thread* q_ang_2, KernelContext_0 thread* kernelContext_151)
{
    st_force_1(b_48, ka_3, kb_3, 0U, c_22, q_lin_2, q_ang_2, kernelContext_151);
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

Measures_0 stress_measures_0(const JointBond_0 thread* b_49, float3 q_lin_3, float3 q_ang_3)
{
    float area_0 = b_49->geom0_0.x;
    float _S1122 = q_lin_3.z;
    float axial_0 = (metal::fast::divide((_S1122), (area_0)));
    float _S1123 = (metal::fast::divide((abs(q_ang_3.x)), (b_49->geom1_0.x)));
    float _S1124 = (metal::fast::divide((abs(q_ang_3.y)), (b_49->geom1_0.y)));
    float bending_0 = _S1123 + _S1124;
    float _S1125 = q_lin_3.x;
    float _S1126 = q_lin_3.y;
    float _S1127 = (metal::fast::sqrt((_S1125 * _S1125 + _S1126 * _S1126)));
    float _S1128 = (metal::fast::divide((_S1127), (area_0)));
    float _S1129 = (metal::fast::divide((abs(q_ang_3.z)), (b_49->geom0_0.w)));
    float shear_1 = _S1128 + _S1129;
    thread Measures_0 m_3;
    (&m_3)->tension_0 = axial_0 + bending_0;
    (&m_3)->shear_0 = shear_1;
    float _S1130 = - axial_0;
    (&m_3)->normal_compression_0 = max(_S1130, 0.0f);
    (&m_3)->compression_0 = _S1130 + bending_0;
    (&m_3)->compressive_force_0 = max(- _S1122, 0.0f);
    return m_3;
}

JointMaterial_0 lp_material_0(uint m_4, KernelContext_0 thread* kernelContext_152)
{
    uint o_8 = kernelContext_152->params_0->ext_mat_0 + 5U * m_4;
    thread JointMaterial_0 j_8;
    (&j_8)->strength_0 = float4(*(kernelContext_152->ext_0+o_8)) ;
    (&j_8)->energy_0 = float4(*(kernelContext_152->ext_0+(o_8 + 1U))) ;
    (&j_8)->dif_0 = float4(*(kernelContext_152->ext_0+(o_8 + 2U))) ;
    (&j_8)->misc_0 = float4(*(kernelContext_152->ext_0+(o_8 + 3U))) ;
    (&j_8)->kind_flags_0 = (as_type<uint4>((float4(*(kernelContext_152->ext_0+(o_8 + 4U))) )));
    return j_8;
}

float infinity_0()
{
    return float_bits_0(2139095040U);
}

float4 failure_indices_0(const JointMaterial_0 thread* mat_1, const JointBond_0 thread* b_50, const Measures_0 thread* m_5, float multiplier_0)
{
    float fc_2 = mat_1->strength_0.y * multiplier_0;
    float _S1131 = min(mat_1->strength_0.z * multiplier_0 + mat_1->strength_0.w * m_5->normal_compression_0, mat_1->energy_0.x * multiplier_0);
    thread float4 idx_0;
    float _S1132 = (metal::fast::divide((m_5->tension_0), (mat_1->strength_0.x * multiplier_0)));
    idx_0.x = max(_S1132, 0.0f);
    float _S1133;
    if(_S1131 > 0.0f)
    {
        float _S1134 = (metal::fast::divide((m_5->shear_0), (_S1131)));
        _S1133 = _S1134;
    }
    else
    {
        _S1133 = infinity_0();
    }
    idx_0.y = _S1133;
    float _S1135 = (metal::fast::divide((m_5->compression_0), (fc_2)));
    idx_0.z = max(_S1135, 0.0f);
    float _S1136 = b_50->stiff1_0.y;
    if(_S1136 > 0.0f)
    {
        float _S1137 = (metal::fast::divide((m_5->compressive_force_0), (_S1136)));
        _S1133 = _S1137;
    }
    else
    {
        _S1133 = 0.0f;
    }
    idx_0.w = _S1133;
    return idx_0;
}

[[kernel]] void fr_lp_breaks(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], atomic<uint> device* atom_38 [[buffer(5)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], packed_float4 device* lp_38 [[buffer(24)]], packed_float4 device* ext_40 [[buffer(23)]], packed_float4 device* engine_contacts_38 [[buffer(25)]], CrushParams_0 constant* crush_38 [[buffer(28)]], packed_float4 device* crush_rw_38 [[buffer(27)]], packed_float4 device* crush_tab_38 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_153;
    (&kernelContext_153)->params_0 = params_38;
    (&kernelContext_153)->work_0 = work_38;
    (&kernelContext_153)->frames_0 = frames_38;
    (&kernelContext_153)->bodies_0 = bodies_38;
    (&kernelContext_153)->chunks_0 = chunks_38;
    (&kernelContext_153)->sort_0 = sort_38;
    (&kernelContext_153)->atom_0 = atom_38;
    (&kernelContext_153)->manifolds_0 = manifolds_38;
    (&kernelContext_153)->loads_0 = loads_38;
    (&kernelContext_153)->bonds_0 = bonds_38;
    (&kernelContext_153)->args_0 = args_38;
    (&kernelContext_153)->info_0 = info_38;
    (&kernelContext_153)->scan_0 = scan_38;
    (&kernelContext_153)->parts_0 = parts_38;
    (&kernelContext_153)->colliders_0 = colliders_38;
    (&kernelContext_153)->collider_info_0 = collider_info_38;
    (&kernelContext_153)->view_chunks_0 = view_chunks_38;
    (&kernelContext_153)->view_0 = view_38;
    (&kernelContext_153)->objects_0 = objects_38;
    (&kernelContext_153)->shot_bodies_0 = shot_bodies_38;
    (&kernelContext_153)->lines_0 = lines_38;
    (&kernelContext_153)->geometry_0 = geometry_38;
    (&kernelContext_153)->shots_0 = shots_38;
    (&kernelContext_153)->lp_0 = lp_38;
    (&kernelContext_153)->ext_0 = ext_40;
    (&kernelContext_153)->engine_contacts_0 = engine_contacts_38;
    (&kernelContext_153)->crush_0 = crush_38;
    (&kernelContext_153)->crush_rw_0 = crush_rw_38;
    (&kernelContext_153)->crush_tab_0 = crush_tab_38;
    threadgroup array<uint, int(1024)> g_scan_38;
    (&kernelContext_153)->g_scan_0 = &g_scan_38;
    threadgroup array<float, int(64)> g_il_best_38;
    (&kernelContext_153)->g_il_best_0 = &g_il_best_38;
    uint i_16 = id_35.x;
    bool _S1138;
    if(i_16 >= (params_38->bond_count_0))
    {
        _S1138 = true;
    }
    else
    {
        uint _S1139 = w_alive_0(&kernelContext_153);
        _S1138 = (*((&kernelContext_153)->work_0+(_S1139 + i_16))) == 0U;
    }
    if(_S1138)
    {
        return;
    }
    uint4 _S1140 = uint4(*((&kernelContext_153)->bonds_0+i_16)) ;
    LpBond_0 _S1141 = lp_bond_0(i_16, &kernelContext_153);
    uint _S1142 = _S1140.x;
    uint _S1143 = _S1140.y;
    thread LpBond_0 _S1144 = _S1141;
    thread float3 q_lin_4;
    thread float3 q_ang_4;
    lp_force_0(&_S1144, _S1142, _S1143, 0U, &q_lin_4, &q_ang_4, &kernelContext_153);
    thread JointBond_0 _S1145 = _S1141.law_0;
    Measures_0 _S1146 = stress_measures_0(&_S1145, q_lin_4, q_ang_4);
    JointMaterial_0 _S1147 = lp_material_0(_S1141.law_0.ids_3.x, &kernelContext_153);
    float _S1148 = _S1141.law_0.geom1_0.w;
    thread JointMaterial_0 _S1149 = _S1147;
    thread JointBond_0 _S1150 = _S1141.law_0;
    thread Measures_0 _S1151 = _S1146;
    float4 _S1152 = failure_indices_0(&_S1149, &_S1150, &_S1151, _S1148);
    float _S1153 = max(max(_S1152.x, _S1152.y), max(_S1152.z, _S1152.w));
    uint _S1154 = a_counters_1(&kernelContext_153);
    uint _S1155 = atomic_fetch_max_explicit((&kernelContext_153)->atom_0+(_S1154 + 15U), (as_type<uint>((max(_S1153, 0.0f)))), memory_order_relaxed);
    if(_S1153 >= 1.0f)
    {
        uint _S1156 = w_alive_0(&kernelContext_153);
        *((&kernelContext_153)->work_0+(_S1156 + i_16)) = 0U;
        uint _S1157 = a_dirty_0(&kernelContext_153);
        uint _S1158 = body_of_0(_S1142, &kernelContext_153);
        uint _S1159 = atomic_fetch_or_explicit((&kernelContext_153)->atom_0+(_S1157 + _S1158), 1U, memory_order_relaxed);
        uint _S1160 = atomic_fetch_add_explicit((&kernelContext_153)->atom_0+_S1154, 1U, memory_order_relaxed);
        uint _S1161 = atomic_fetch_add_explicit((&kernelContext_153)->atom_0+(_S1154 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first_reset(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], atomic<uint> device* atom_39 [[buffer(5)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], packed_float4 device* lp_39 [[buffer(24)]], packed_float4 device* ext_41 [[buffer(23)]], packed_float4 device* engine_contacts_39 [[buffer(25)]], CrushParams_0 constant* crush_39 [[buffer(28)]], packed_float4 device* crush_rw_39 [[buffer(27)]], packed_float4 device* crush_tab_39 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_154;
    (&kernelContext_154)->params_0 = params_39;
    (&kernelContext_154)->work_0 = work_39;
    (&kernelContext_154)->frames_0 = frames_39;
    (&kernelContext_154)->bodies_0 = bodies_39;
    (&kernelContext_154)->chunks_0 = chunks_39;
    (&kernelContext_154)->sort_0 = sort_39;
    (&kernelContext_154)->atom_0 = atom_39;
    (&kernelContext_154)->manifolds_0 = manifolds_39;
    (&kernelContext_154)->loads_0 = loads_39;
    (&kernelContext_154)->bonds_0 = bonds_39;
    (&kernelContext_154)->args_0 = args_39;
    (&kernelContext_154)->info_0 = info_39;
    (&kernelContext_154)->scan_0 = scan_39;
    (&kernelContext_154)->parts_0 = parts_39;
    (&kernelContext_154)->colliders_0 = colliders_39;
    (&kernelContext_154)->collider_info_0 = collider_info_39;
    (&kernelContext_154)->view_chunks_0 = view_chunks_39;
    (&kernelContext_154)->view_0 = view_39;
    (&kernelContext_154)->objects_0 = objects_39;
    (&kernelContext_154)->shot_bodies_0 = shot_bodies_39;
    (&kernelContext_154)->lines_0 = lines_39;
    (&kernelContext_154)->geometry_0 = geometry_39;
    (&kernelContext_154)->shots_0 = shots_39;
    (&kernelContext_154)->lp_0 = lp_39;
    (&kernelContext_154)->ext_0 = ext_41;
    (&kernelContext_154)->engine_contacts_0 = engine_contacts_39;
    (&kernelContext_154)->crush_0 = crush_39;
    (&kernelContext_154)->crush_rw_0 = crush_rw_39;
    (&kernelContext_154)->crush_tab_0 = crush_tab_39;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_154)->g_scan_0 = &g_scan_39;
    threadgroup array<float, int(64)> g_il_best_39;
    (&kernelContext_154)->g_il_best_0 = &g_il_best_39;
    uint _S1162 = id_36.x;
    if(_S1162 == 0U)
    {
        uint _S1163 = a_il_cursor_1(&kernelContext_154);
        atomic_store_explicit((&kernelContext_154)->atom_0+_S1163, 0U, memory_order_relaxed);
    }
    if(_S1162 < ((&kernelContext_154)->params_0->shot_count_0))
    {
        uint _S1164 = a_first_1(_S1162, &kernelContext_154);
        atomic_store_explicit((&kernelContext_154)->atom_0+_S1164, 2139095040U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], atomic<uint> device* atom_40 [[buffer(5)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], packed_float4 device* lp_40 [[buffer(24)]], packed_float4 device* ext_42 [[buffer(23)]], packed_float4 device* engine_contacts_40 [[buffer(25)]], CrushParams_0 constant* crush_40 [[buffer(28)]], packed_float4 device* crush_rw_40 [[buffer(27)]], packed_float4 device* crush_tab_40 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_155;
    (&kernelContext_155)->params_0 = params_40;
    (&kernelContext_155)->work_0 = work_40;
    (&kernelContext_155)->frames_0 = frames_40;
    (&kernelContext_155)->bodies_0 = bodies_40;
    (&kernelContext_155)->chunks_0 = chunks_40;
    (&kernelContext_155)->sort_0 = sort_40;
    (&kernelContext_155)->atom_0 = atom_40;
    (&kernelContext_155)->manifolds_0 = manifolds_40;
    (&kernelContext_155)->loads_0 = loads_40;
    (&kernelContext_155)->bonds_0 = bonds_40;
    (&kernelContext_155)->args_0 = args_40;
    (&kernelContext_155)->info_0 = info_40;
    (&kernelContext_155)->scan_0 = scan_40;
    (&kernelContext_155)->parts_0 = parts_40;
    (&kernelContext_155)->colliders_0 = colliders_40;
    (&kernelContext_155)->collider_info_0 = collider_info_40;
    (&kernelContext_155)->view_chunks_0 = view_chunks_40;
    (&kernelContext_155)->view_0 = view_40;
    (&kernelContext_155)->objects_0 = objects_40;
    (&kernelContext_155)->shot_bodies_0 = shot_bodies_40;
    (&kernelContext_155)->lines_0 = lines_40;
    (&kernelContext_155)->geometry_0 = geometry_40;
    (&kernelContext_155)->shots_0 = shots_40;
    (&kernelContext_155)->lp_0 = lp_40;
    (&kernelContext_155)->ext_0 = ext_42;
    (&kernelContext_155)->engine_contacts_0 = engine_contacts_40;
    (&kernelContext_155)->crush_0 = crush_40;
    (&kernelContext_155)->crush_rw_0 = crush_rw_40;
    (&kernelContext_155)->crush_tab_0 = crush_tab_40;
    threadgroup array<uint, int(1024)> g_scan_40;
    (&kernelContext_155)->g_scan_0 = &g_scan_40;
    threadgroup array<float, int(64)> g_il_best_40;
    (&kernelContext_155)->g_il_best_0 = &g_il_best_40;
    uint b_51 = id_37.x;
    bool _S1165;
    if(b_51 >= (params_40->body_count_0))
    {
        _S1165 = true;
    }
    else
    {
        _S1165 = ((&kernelContext_155)->params_0->shot_count_0) == 0U;
    }
    if(_S1165)
    {
        return;
    }
    thread uint shot_9;
    bool _S1166 = struck_0(b_51, &shot_9, &kernelContext_155);
    if(!_S1166)
    {
        return;
    }
    uint _S1167 = a_first_1(shot_9, &kernelContext_155);
    atomic<uint> device* _S1168 = (&kernelContext_155)->atom_0+_S1167;
    uint _S1169 = a_strike_0(&kernelContext_155);
    uint _S1170 = atomic_load_explicit((&kernelContext_155)->atom_0+(_S1169 + 9U * b_51 + 6U), memory_order_relaxed);
    uint _S1171 = atomic_fetch_min_explicit(_S1168, _S1170, memory_order_relaxed);
    return;
}

bool cs_struck_1(uint b_52, KernelContext_0 thread* kernelContext_156)
{
    bool _S1172;
    if(b_52 < (kernelContext_156->params_0->body_count_0))
    {
        uint _S1173 = a_cs_0(b_52, kernelContext_156);
        uint _S1174 = atomic_load_explicit(kernelContext_156->atom_0+(_S1173 + 1U), memory_order_relaxed);
        _S1172 = _S1174 != 0U;
    }
    else
    {
        _S1172 = false;
    }
    return _S1172;
}

[[kernel]] void fr_il_begin(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], atomic<uint> device* atom_41 [[buffer(5)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], packed_float4 device* lp_41 [[buffer(24)]], packed_float4 device* ext_43 [[buffer(23)]], packed_float4 device* engine_contacts_41 [[buffer(25)]], CrushParams_0 constant* crush_41 [[buffer(28)]], packed_float4 device* crush_rw_41 [[buffer(27)]], packed_float4 device* crush_tab_41 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_157;
    (&kernelContext_157)->params_0 = params_41;
    (&kernelContext_157)->work_0 = work_41;
    (&kernelContext_157)->frames_0 = frames_41;
    (&kernelContext_157)->bodies_0 = bodies_41;
    (&kernelContext_157)->chunks_0 = chunks_41;
    (&kernelContext_157)->sort_0 = sort_41;
    (&kernelContext_157)->atom_0 = atom_41;
    (&kernelContext_157)->manifolds_0 = manifolds_41;
    (&kernelContext_157)->loads_0 = loads_41;
    (&kernelContext_157)->bonds_0 = bonds_41;
    (&kernelContext_157)->args_0 = args_41;
    (&kernelContext_157)->info_0 = info_41;
    (&kernelContext_157)->scan_0 = scan_41;
    (&kernelContext_157)->parts_0 = parts_41;
    (&kernelContext_157)->colliders_0 = colliders_41;
    (&kernelContext_157)->collider_info_0 = collider_info_41;
    (&kernelContext_157)->view_chunks_0 = view_chunks_41;
    (&kernelContext_157)->view_0 = view_41;
    (&kernelContext_157)->objects_0 = objects_41;
    (&kernelContext_157)->shot_bodies_0 = shot_bodies_41;
    (&kernelContext_157)->lines_0 = lines_41;
    (&kernelContext_157)->geometry_0 = geometry_41;
    (&kernelContext_157)->shots_0 = shots_41;
    (&kernelContext_157)->lp_0 = lp_41;
    (&kernelContext_157)->ext_0 = ext_43;
    (&kernelContext_157)->engine_contacts_0 = engine_contacts_41;
    (&kernelContext_157)->crush_0 = crush_41;
    (&kernelContext_157)->crush_rw_0 = crush_rw_41;
    (&kernelContext_157)->crush_tab_0 = crush_tab_41;
    threadgroup array<uint, int(1024)> g_scan_41;
    (&kernelContext_157)->g_scan_0 = &g_scan_41;
    threadgroup array<float, int(64)> g_il_best_41;
    (&kernelContext_157)->g_il_best_0 = &g_il_best_41;
    if((params_41->shot_count_0) > 0U)
    {
        il_begin_1(id_38.x, &kernelContext_157);
    }
    return;
}

[[kernel]] void fr_il_tube_count(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], atomic<uint> device* atom_42 [[buffer(5)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], packed_float4 device* lp_42 [[buffer(24)]], packed_float4 device* ext_44 [[buffer(23)]], packed_float4 device* engine_contacts_42 [[buffer(25)]], CrushParams_0 constant* crush_42 [[buffer(28)]], packed_float4 device* crush_rw_42 [[buffer(27)]], packed_float4 device* crush_tab_42 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_158;
    (&kernelContext_158)->params_0 = params_42;
    (&kernelContext_158)->work_0 = work_42;
    (&kernelContext_158)->frames_0 = frames_42;
    (&kernelContext_158)->bodies_0 = bodies_42;
    (&kernelContext_158)->chunks_0 = chunks_42;
    (&kernelContext_158)->sort_0 = sort_42;
    (&kernelContext_158)->atom_0 = atom_42;
    (&kernelContext_158)->manifolds_0 = manifolds_42;
    (&kernelContext_158)->loads_0 = loads_42;
    (&kernelContext_158)->bonds_0 = bonds_42;
    (&kernelContext_158)->args_0 = args_42;
    (&kernelContext_158)->info_0 = info_42;
    (&kernelContext_158)->scan_0 = scan_42;
    (&kernelContext_158)->parts_0 = parts_42;
    (&kernelContext_158)->colliders_0 = colliders_42;
    (&kernelContext_158)->collider_info_0 = collider_info_42;
    (&kernelContext_158)->view_chunks_0 = view_chunks_42;
    (&kernelContext_158)->view_0 = view_42;
    (&kernelContext_158)->objects_0 = objects_42;
    (&kernelContext_158)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_158)->lines_0 = lines_42;
    (&kernelContext_158)->geometry_0 = geometry_42;
    (&kernelContext_158)->shots_0 = shots_42;
    (&kernelContext_158)->lp_0 = lp_42;
    (&kernelContext_158)->ext_0 = ext_44;
    (&kernelContext_158)->engine_contacts_0 = engine_contacts_42;
    (&kernelContext_158)->crush_0 = crush_42;
    (&kernelContext_158)->crush_rw_0 = crush_rw_42;
    (&kernelContext_158)->crush_tab_0 = crush_tab_42;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_158)->g_scan_0 = &g_scan_42;
    threadgroup array<float, int(64)> g_il_best_42;
    (&kernelContext_158)->g_il_best_0 = &g_il_best_42;
    if((params_42->shot_count_0) > 0U)
    {
        il_tube_count_1(id_39.x, &kernelContext_158);
    }
    return;
}

[[kernel]] void fr_il_tube_base(uint3 id_40 [[thread_position_in_grid]], FractureParams_0 constant* params_43 [[buffer(0)]], uint device* work_43 [[buffer(4)]], packed_float4 device* frames_43 [[buffer(6)]], packed_float4 device* bodies_43 [[buffer(7)]], packed_float4 device* chunks_43 [[buffer(1)]], uint device* sort_43 [[buffer(11)]], atomic<uint> device* atom_43 [[buffer(5)]], Record_natural_0 device* manifolds_43 [[buffer(10)]], packed_float4 device* loads_43 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_43 [[buffer(13)]], uint device* info_43 [[buffer(15)]], ScanParams_0 constant* scan_43 [[buffer(14)]], packed_float4 device* parts_43 [[buffer(3)]], packed_float4 device* colliders_43 [[buffer(8)]], packed_uint4 device* collider_info_43 [[buffer(9)]], packed_float4 device* view_chunks_43 [[buffer(17)]], ViewParams_0 constant* view_43 [[buffer(16)]], packed_float4 device* objects_43 [[buffer(18)]], uint device* shot_bodies_43 [[buffer(20)]], float device* lines_43 [[buffer(19)]], packed_float4 device* geometry_43 [[buffer(21)]], packed_float4 device* shots_43 [[buffer(22)]], packed_float4 device* lp_43 [[buffer(24)]], packed_float4 device* ext_45 [[buffer(23)]], packed_float4 device* engine_contacts_43 [[buffer(25)]], CrushParams_0 constant* crush_43 [[buffer(28)]], packed_float4 device* crush_rw_43 [[buffer(27)]], packed_float4 device* crush_tab_43 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_159;
    (&kernelContext_159)->params_0 = params_43;
    (&kernelContext_159)->work_0 = work_43;
    (&kernelContext_159)->frames_0 = frames_43;
    (&kernelContext_159)->bodies_0 = bodies_43;
    (&kernelContext_159)->chunks_0 = chunks_43;
    (&kernelContext_159)->sort_0 = sort_43;
    (&kernelContext_159)->atom_0 = atom_43;
    (&kernelContext_159)->manifolds_0 = manifolds_43;
    (&kernelContext_159)->loads_0 = loads_43;
    (&kernelContext_159)->bonds_0 = bonds_43;
    (&kernelContext_159)->args_0 = args_43;
    (&kernelContext_159)->info_0 = info_43;
    (&kernelContext_159)->scan_0 = scan_43;
    (&kernelContext_159)->parts_0 = parts_43;
    (&kernelContext_159)->colliders_0 = colliders_43;
    (&kernelContext_159)->collider_info_0 = collider_info_43;
    (&kernelContext_159)->view_chunks_0 = view_chunks_43;
    (&kernelContext_159)->view_0 = view_43;
    (&kernelContext_159)->objects_0 = objects_43;
    (&kernelContext_159)->shot_bodies_0 = shot_bodies_43;
    (&kernelContext_159)->lines_0 = lines_43;
    (&kernelContext_159)->geometry_0 = geometry_43;
    (&kernelContext_159)->shots_0 = shots_43;
    (&kernelContext_159)->lp_0 = lp_43;
    (&kernelContext_159)->ext_0 = ext_45;
    (&kernelContext_159)->engine_contacts_0 = engine_contacts_43;
    (&kernelContext_159)->crush_0 = crush_43;
    (&kernelContext_159)->crush_rw_0 = crush_rw_43;
    (&kernelContext_159)->crush_tab_0 = crush_tab_43;
    threadgroup array<uint, int(1024)> g_scan_43;
    (&kernelContext_159)->g_scan_0 = &g_scan_43;
    threadgroup array<float, int(64)> g_il_best_43;
    (&kernelContext_159)->g_il_best_0 = &g_il_best_43;
    if((params_43->shot_count_0) > 0U)
    {
        il_tube_base_1(id_40.x, &kernelContext_159);
    }
    return;
}

[[kernel]] void fr_il_tube_collect(uint3 id_41 [[thread_position_in_grid]], FractureParams_0 constant* params_44 [[buffer(0)]], uint device* work_44 [[buffer(4)]], packed_float4 device* frames_44 [[buffer(6)]], packed_float4 device* bodies_44 [[buffer(7)]], packed_float4 device* chunks_44 [[buffer(1)]], uint device* sort_44 [[buffer(11)]], atomic<uint> device* atom_44 [[buffer(5)]], Record_natural_0 device* manifolds_44 [[buffer(10)]], packed_float4 device* loads_44 [[buffer(12)]], packed_uint4 device* bonds_44 [[buffer(2)]], uint device* args_44 [[buffer(13)]], uint device* info_44 [[buffer(15)]], ScanParams_0 constant* scan_44 [[buffer(14)]], packed_float4 device* parts_44 [[buffer(3)]], packed_float4 device* colliders_44 [[buffer(8)]], packed_uint4 device* collider_info_44 [[buffer(9)]], packed_float4 device* view_chunks_44 [[buffer(17)]], ViewParams_0 constant* view_44 [[buffer(16)]], packed_float4 device* objects_44 [[buffer(18)]], uint device* shot_bodies_44 [[buffer(20)]], float device* lines_44 [[buffer(19)]], packed_float4 device* geometry_44 [[buffer(21)]], packed_float4 device* shots_44 [[buffer(22)]], packed_float4 device* lp_44 [[buffer(24)]], packed_float4 device* ext_46 [[buffer(23)]], packed_float4 device* engine_contacts_44 [[buffer(25)]], CrushParams_0 constant* crush_44 [[buffer(28)]], packed_float4 device* crush_rw_44 [[buffer(27)]], packed_float4 device* crush_tab_44 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_160;
    (&kernelContext_160)->params_0 = params_44;
    (&kernelContext_160)->work_0 = work_44;
    (&kernelContext_160)->frames_0 = frames_44;
    (&kernelContext_160)->bodies_0 = bodies_44;
    (&kernelContext_160)->chunks_0 = chunks_44;
    (&kernelContext_160)->sort_0 = sort_44;
    (&kernelContext_160)->atom_0 = atom_44;
    (&kernelContext_160)->manifolds_0 = manifolds_44;
    (&kernelContext_160)->loads_0 = loads_44;
    (&kernelContext_160)->bonds_0 = bonds_44;
    (&kernelContext_160)->args_0 = args_44;
    (&kernelContext_160)->info_0 = info_44;
    (&kernelContext_160)->scan_0 = scan_44;
    (&kernelContext_160)->parts_0 = parts_44;
    (&kernelContext_160)->colliders_0 = colliders_44;
    (&kernelContext_160)->collider_info_0 = collider_info_44;
    (&kernelContext_160)->view_chunks_0 = view_chunks_44;
    (&kernelContext_160)->view_0 = view_44;
    (&kernelContext_160)->objects_0 = objects_44;
    (&kernelContext_160)->shot_bodies_0 = shot_bodies_44;
    (&kernelContext_160)->lines_0 = lines_44;
    (&kernelContext_160)->geometry_0 = geometry_44;
    (&kernelContext_160)->shots_0 = shots_44;
    (&kernelContext_160)->lp_0 = lp_44;
    (&kernelContext_160)->ext_0 = ext_46;
    (&kernelContext_160)->engine_contacts_0 = engine_contacts_44;
    (&kernelContext_160)->crush_0 = crush_44;
    (&kernelContext_160)->crush_rw_0 = crush_rw_44;
    (&kernelContext_160)->crush_tab_0 = crush_tab_44;
    threadgroup array<uint, int(1024)> g_scan_44;
    (&kernelContext_160)->g_scan_0 = &g_scan_44;
    threadgroup array<float, int(64)> g_il_best_44;
    (&kernelContext_160)->g_il_best_0 = &g_il_best_44;
    if((params_44->shot_count_0) > 0U)
    {
        il_tube_collect_1(id_41.x, &kernelContext_160);
    }
    return;
}

[[kernel]] void fr_il_run(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_45 [[buffer(0)]], uint device* work_45 [[buffer(4)]], packed_float4 device* frames_45 [[buffer(6)]], packed_float4 device* bodies_45 [[buffer(7)]], packed_float4 device* chunks_45 [[buffer(1)]], uint device* sort_45 [[buffer(11)]], atomic<uint> device* atom_45 [[buffer(5)]], Record_natural_0 device* manifolds_45 [[buffer(10)]], packed_float4 device* loads_45 [[buffer(12)]], packed_uint4 device* bonds_45 [[buffer(2)]], uint device* args_45 [[buffer(13)]], uint device* info_45 [[buffer(15)]], ScanParams_0 constant* scan_45 [[buffer(14)]], packed_float4 device* parts_45 [[buffer(3)]], packed_float4 device* colliders_45 [[buffer(8)]], packed_uint4 device* collider_info_45 [[buffer(9)]], packed_float4 device* view_chunks_45 [[buffer(17)]], ViewParams_0 constant* view_45 [[buffer(16)]], packed_float4 device* objects_45 [[buffer(18)]], uint device* shot_bodies_45 [[buffer(20)]], float device* lines_45 [[buffer(19)]], packed_float4 device* geometry_45 [[buffer(21)]], packed_float4 device* shots_45 [[buffer(22)]], packed_float4 device* lp_45 [[buffer(24)]], packed_float4 device* ext_47 [[buffer(23)]], packed_float4 device* engine_contacts_45 [[buffer(25)]], CrushParams_0 constant* crush_45 [[buffer(28)]], packed_float4 device* crush_rw_45 [[buffer(27)]], packed_float4 device* crush_tab_45 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_161;
    (&kernelContext_161)->params_0 = params_45;
    (&kernelContext_161)->work_0 = work_45;
    (&kernelContext_161)->frames_0 = frames_45;
    (&kernelContext_161)->bodies_0 = bodies_45;
    (&kernelContext_161)->chunks_0 = chunks_45;
    (&kernelContext_161)->sort_0 = sort_45;
    (&kernelContext_161)->atom_0 = atom_45;
    (&kernelContext_161)->manifolds_0 = manifolds_45;
    (&kernelContext_161)->loads_0 = loads_45;
    (&kernelContext_161)->bonds_0 = bonds_45;
    (&kernelContext_161)->args_0 = args_45;
    (&kernelContext_161)->info_0 = info_45;
    (&kernelContext_161)->scan_0 = scan_45;
    (&kernelContext_161)->parts_0 = parts_45;
    (&kernelContext_161)->colliders_0 = colliders_45;
    (&kernelContext_161)->collider_info_0 = collider_info_45;
    (&kernelContext_161)->view_chunks_0 = view_chunks_45;
    (&kernelContext_161)->view_0 = view_45;
    (&kernelContext_161)->objects_0 = objects_45;
    (&kernelContext_161)->shot_bodies_0 = shot_bodies_45;
    (&kernelContext_161)->lines_0 = lines_45;
    (&kernelContext_161)->geometry_0 = geometry_45;
    (&kernelContext_161)->shots_0 = shots_45;
    (&kernelContext_161)->lp_0 = lp_45;
    (&kernelContext_161)->ext_0 = ext_47;
    (&kernelContext_161)->engine_contacts_0 = engine_contacts_45;
    (&kernelContext_161)->crush_0 = crush_45;
    (&kernelContext_161)->crush_rw_0 = crush_rw_45;
    (&kernelContext_161)->crush_tab_0 = crush_tab_45;
    threadgroup array<uint, int(1024)> g_scan_45;
    (&kernelContext_161)->g_scan_0 = &g_scan_45;
    threadgroup array<float, int(64)> g_il_best_45;
    (&kernelContext_161)->g_il_best_0 = &g_il_best_45;
    if((params_45->shot_count_0) > 0U)
    {
        il_run_1(group_1.x, thread_4.x, &kernelContext_161);
    }
    return;
}

[[kernel]] void fr_il_strike(uint3 id_42 [[thread_position_in_grid]], FractureParams_0 constant* params_46 [[buffer(0)]], uint device* work_46 [[buffer(4)]], packed_float4 device* frames_46 [[buffer(6)]], packed_float4 device* bodies_46 [[buffer(7)]], packed_float4 device* chunks_46 [[buffer(1)]], uint device* sort_46 [[buffer(11)]], atomic<uint> device* atom_46 [[buffer(5)]], Record_natural_0 device* manifolds_46 [[buffer(10)]], packed_float4 device* loads_46 [[buffer(12)]], packed_uint4 device* bonds_46 [[buffer(2)]], uint device* args_46 [[buffer(13)]], uint device* info_46 [[buffer(15)]], ScanParams_0 constant* scan_46 [[buffer(14)]], packed_float4 device* parts_46 [[buffer(3)]], packed_float4 device* colliders_46 [[buffer(8)]], packed_uint4 device* collider_info_46 [[buffer(9)]], packed_float4 device* view_chunks_46 [[buffer(17)]], ViewParams_0 constant* view_46 [[buffer(16)]], packed_float4 device* objects_46 [[buffer(18)]], uint device* shot_bodies_46 [[buffer(20)]], float device* lines_46 [[buffer(19)]], packed_float4 device* geometry_46 [[buffer(21)]], packed_float4 device* shots_46 [[buffer(22)]], packed_float4 device* lp_46 [[buffer(24)]], packed_float4 device* ext_48 [[buffer(23)]], packed_float4 device* engine_contacts_46 [[buffer(25)]], CrushParams_0 constant* crush_46 [[buffer(28)]], packed_float4 device* crush_rw_46 [[buffer(27)]], packed_float4 device* crush_tab_46 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_162;
    (&kernelContext_162)->params_0 = params_46;
    (&kernelContext_162)->work_0 = work_46;
    (&kernelContext_162)->frames_0 = frames_46;
    (&kernelContext_162)->bodies_0 = bodies_46;
    (&kernelContext_162)->chunks_0 = chunks_46;
    (&kernelContext_162)->sort_0 = sort_46;
    (&kernelContext_162)->atom_0 = atom_46;
    (&kernelContext_162)->manifolds_0 = manifolds_46;
    (&kernelContext_162)->loads_0 = loads_46;
    (&kernelContext_162)->bonds_0 = bonds_46;
    (&kernelContext_162)->args_0 = args_46;
    (&kernelContext_162)->info_0 = info_46;
    (&kernelContext_162)->scan_0 = scan_46;
    (&kernelContext_162)->parts_0 = parts_46;
    (&kernelContext_162)->colliders_0 = colliders_46;
    (&kernelContext_162)->collider_info_0 = collider_info_46;
    (&kernelContext_162)->view_chunks_0 = view_chunks_46;
    (&kernelContext_162)->view_0 = view_46;
    (&kernelContext_162)->objects_0 = objects_46;
    (&kernelContext_162)->shot_bodies_0 = shot_bodies_46;
    (&kernelContext_162)->lines_0 = lines_46;
    (&kernelContext_162)->geometry_0 = geometry_46;
    (&kernelContext_162)->shots_0 = shots_46;
    (&kernelContext_162)->lp_0 = lp_46;
    (&kernelContext_162)->ext_0 = ext_48;
    (&kernelContext_162)->engine_contacts_0 = engine_contacts_46;
    (&kernelContext_162)->crush_0 = crush_46;
    (&kernelContext_162)->crush_rw_0 = crush_rw_46;
    (&kernelContext_162)->crush_tab_0 = crush_tab_46;
    threadgroup array<uint, int(1024)> g_scan_46;
    (&kernelContext_162)->g_scan_0 = &g_scan_46;
    threadgroup array<float, int(64)> g_il_best_46;
    (&kernelContext_162)->g_il_best_0 = &g_il_best_46;
    if((params_46->shot_count_0) > 0U)
    {
        il_strike_1(id_42.x, &kernelContext_162);
    }
    return;
}

[[kernel]] void fr_il_mark(uint3 id_43 [[thread_position_in_grid]], FractureParams_0 constant* params_47 [[buffer(0)]], uint device* work_47 [[buffer(4)]], packed_float4 device* frames_47 [[buffer(6)]], packed_float4 device* bodies_47 [[buffer(7)]], packed_float4 device* chunks_47 [[buffer(1)]], uint device* sort_47 [[buffer(11)]], atomic<uint> device* atom_47 [[buffer(5)]], Record_natural_0 device* manifolds_47 [[buffer(10)]], packed_float4 device* loads_47 [[buffer(12)]], packed_uint4 device* bonds_47 [[buffer(2)]], uint device* args_47 [[buffer(13)]], uint device* info_47 [[buffer(15)]], ScanParams_0 constant* scan_47 [[buffer(14)]], packed_float4 device* parts_47 [[buffer(3)]], packed_float4 device* colliders_47 [[buffer(8)]], packed_uint4 device* collider_info_47 [[buffer(9)]], packed_float4 device* view_chunks_47 [[buffer(17)]], ViewParams_0 constant* view_47 [[buffer(16)]], packed_float4 device* objects_47 [[buffer(18)]], uint device* shot_bodies_47 [[buffer(20)]], float device* lines_47 [[buffer(19)]], packed_float4 device* geometry_47 [[buffer(21)]], packed_float4 device* shots_47 [[buffer(22)]], packed_float4 device* lp_47 [[buffer(24)]], packed_float4 device* ext_49 [[buffer(23)]], packed_float4 device* engine_contacts_47 [[buffer(25)]], CrushParams_0 constant* crush_47 [[buffer(28)]], packed_float4 device* crush_rw_47 [[buffer(27)]], packed_float4 device* crush_tab_47 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_163;
    (&kernelContext_163)->params_0 = params_47;
    (&kernelContext_163)->work_0 = work_47;
    (&kernelContext_163)->frames_0 = frames_47;
    (&kernelContext_163)->bodies_0 = bodies_47;
    (&kernelContext_163)->chunks_0 = chunks_47;
    (&kernelContext_163)->sort_0 = sort_47;
    (&kernelContext_163)->atom_0 = atom_47;
    (&kernelContext_163)->manifolds_0 = manifolds_47;
    (&kernelContext_163)->loads_0 = loads_47;
    (&kernelContext_163)->bonds_0 = bonds_47;
    (&kernelContext_163)->args_0 = args_47;
    (&kernelContext_163)->info_0 = info_47;
    (&kernelContext_163)->scan_0 = scan_47;
    (&kernelContext_163)->parts_0 = parts_47;
    (&kernelContext_163)->colliders_0 = colliders_47;
    (&kernelContext_163)->collider_info_0 = collider_info_47;
    (&kernelContext_163)->view_chunks_0 = view_chunks_47;
    (&kernelContext_163)->view_0 = view_47;
    (&kernelContext_163)->objects_0 = objects_47;
    (&kernelContext_163)->shot_bodies_0 = shot_bodies_47;
    (&kernelContext_163)->lines_0 = lines_47;
    (&kernelContext_163)->geometry_0 = geometry_47;
    (&kernelContext_163)->shots_0 = shots_47;
    (&kernelContext_163)->lp_0 = lp_47;
    (&kernelContext_163)->ext_0 = ext_49;
    (&kernelContext_163)->engine_contacts_0 = engine_contacts_47;
    (&kernelContext_163)->crush_0 = crush_47;
    (&kernelContext_163)->crush_rw_0 = crush_rw_47;
    (&kernelContext_163)->crush_tab_0 = crush_tab_47;
    threadgroup array<uint, int(1024)> g_scan_47;
    (&kernelContext_163)->g_scan_0 = &g_scan_47;
    threadgroup array<float, int(64)> g_il_best_47;
    (&kernelContext_163)->g_il_best_0 = &g_il_best_47;
    if((params_47->shot_count_0) > 0U)
    {
        il_mark_1(id_43.x, &kernelContext_163);
    }
    return;
}

[[kernel]] void fr_il_carry(uint3 id_44 [[thread_position_in_grid]], FractureParams_0 constant* params_48 [[buffer(0)]], uint device* work_48 [[buffer(4)]], packed_float4 device* frames_48 [[buffer(6)]], packed_float4 device* bodies_48 [[buffer(7)]], packed_float4 device* chunks_48 [[buffer(1)]], uint device* sort_48 [[buffer(11)]], atomic<uint> device* atom_48 [[buffer(5)]], Record_natural_0 device* manifolds_48 [[buffer(10)]], packed_float4 device* loads_48 [[buffer(12)]], packed_uint4 device* bonds_48 [[buffer(2)]], uint device* args_48 [[buffer(13)]], uint device* info_48 [[buffer(15)]], ScanParams_0 constant* scan_48 [[buffer(14)]], packed_float4 device* parts_48 [[buffer(3)]], packed_float4 device* colliders_48 [[buffer(8)]], packed_uint4 device* collider_info_48 [[buffer(9)]], packed_float4 device* view_chunks_48 [[buffer(17)]], ViewParams_0 constant* view_48 [[buffer(16)]], packed_float4 device* objects_48 [[buffer(18)]], uint device* shot_bodies_48 [[buffer(20)]], float device* lines_48 [[buffer(19)]], packed_float4 device* geometry_48 [[buffer(21)]], packed_float4 device* shots_48 [[buffer(22)]], packed_float4 device* lp_48 [[buffer(24)]], packed_float4 device* ext_50 [[buffer(23)]], packed_float4 device* engine_contacts_48 [[buffer(25)]], CrushParams_0 constant* crush_48 [[buffer(28)]], packed_float4 device* crush_rw_48 [[buffer(27)]], packed_float4 device* crush_tab_48 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_164;
    (&kernelContext_164)->params_0 = params_48;
    (&kernelContext_164)->work_0 = work_48;
    (&kernelContext_164)->frames_0 = frames_48;
    (&kernelContext_164)->bodies_0 = bodies_48;
    (&kernelContext_164)->chunks_0 = chunks_48;
    (&kernelContext_164)->sort_0 = sort_48;
    (&kernelContext_164)->atom_0 = atom_48;
    (&kernelContext_164)->manifolds_0 = manifolds_48;
    (&kernelContext_164)->loads_0 = loads_48;
    (&kernelContext_164)->bonds_0 = bonds_48;
    (&kernelContext_164)->args_0 = args_48;
    (&kernelContext_164)->info_0 = info_48;
    (&kernelContext_164)->scan_0 = scan_48;
    (&kernelContext_164)->parts_0 = parts_48;
    (&kernelContext_164)->colliders_0 = colliders_48;
    (&kernelContext_164)->collider_info_0 = collider_info_48;
    (&kernelContext_164)->view_chunks_0 = view_chunks_48;
    (&kernelContext_164)->view_0 = view_48;
    (&kernelContext_164)->objects_0 = objects_48;
    (&kernelContext_164)->shot_bodies_0 = shot_bodies_48;
    (&kernelContext_164)->lines_0 = lines_48;
    (&kernelContext_164)->geometry_0 = geometry_48;
    (&kernelContext_164)->shots_0 = shots_48;
    (&kernelContext_164)->lp_0 = lp_48;
    (&kernelContext_164)->ext_0 = ext_50;
    (&kernelContext_164)->engine_contacts_0 = engine_contacts_48;
    (&kernelContext_164)->crush_0 = crush_48;
    (&kernelContext_164)->crush_rw_0 = crush_rw_48;
    (&kernelContext_164)->crush_tab_0 = crush_tab_48;
    threadgroup array<uint, int(1024)> g_scan_48;
    (&kernelContext_164)->g_scan_0 = &g_scan_48;
    threadgroup array<float, int(64)> g_il_best_48;
    (&kernelContext_164)->g_il_best_0 = &g_il_best_48;
    if((params_48->shot_count_0) > 0U)
    {
        il_carry_1(id_44.x, &kernelContext_164);
    }
    return;
}

[[kernel]] void fr_il_breaks(uint3 id_45 [[thread_position_in_grid]], FractureParams_0 constant* params_49 [[buffer(0)]], uint device* work_49 [[buffer(4)]], packed_float4 device* frames_49 [[buffer(6)]], packed_float4 device* bodies_49 [[buffer(7)]], packed_float4 device* chunks_49 [[buffer(1)]], uint device* sort_49 [[buffer(11)]], atomic<uint> device* atom_49 [[buffer(5)]], Record_natural_0 device* manifolds_49 [[buffer(10)]], packed_float4 device* loads_49 [[buffer(12)]], packed_uint4 device* bonds_49 [[buffer(2)]], uint device* args_49 [[buffer(13)]], uint device* info_49 [[buffer(15)]], ScanParams_0 constant* scan_49 [[buffer(14)]], packed_float4 device* parts_49 [[buffer(3)]], packed_float4 device* colliders_49 [[buffer(8)]], packed_uint4 device* collider_info_49 [[buffer(9)]], packed_float4 device* view_chunks_49 [[buffer(17)]], ViewParams_0 constant* view_49 [[buffer(16)]], packed_float4 device* objects_49 [[buffer(18)]], uint device* shot_bodies_49 [[buffer(20)]], float device* lines_49 [[buffer(19)]], packed_float4 device* geometry_49 [[buffer(21)]], packed_float4 device* shots_49 [[buffer(22)]], packed_float4 device* lp_49 [[buffer(24)]], packed_float4 device* ext_51 [[buffer(23)]], packed_float4 device* engine_contacts_49 [[buffer(25)]], CrushParams_0 constant* crush_49 [[buffer(28)]], packed_float4 device* crush_rw_49 [[buffer(27)]], packed_float4 device* crush_tab_49 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_165;
    (&kernelContext_165)->params_0 = params_49;
    (&kernelContext_165)->work_0 = work_49;
    (&kernelContext_165)->frames_0 = frames_49;
    (&kernelContext_165)->bodies_0 = bodies_49;
    (&kernelContext_165)->chunks_0 = chunks_49;
    (&kernelContext_165)->sort_0 = sort_49;
    (&kernelContext_165)->atom_0 = atom_49;
    (&kernelContext_165)->manifolds_0 = manifolds_49;
    (&kernelContext_165)->loads_0 = loads_49;
    (&kernelContext_165)->bonds_0 = bonds_49;
    (&kernelContext_165)->args_0 = args_49;
    (&kernelContext_165)->info_0 = info_49;
    (&kernelContext_165)->scan_0 = scan_49;
    (&kernelContext_165)->parts_0 = parts_49;
    (&kernelContext_165)->colliders_0 = colliders_49;
    (&kernelContext_165)->collider_info_0 = collider_info_49;
    (&kernelContext_165)->view_chunks_0 = view_chunks_49;
    (&kernelContext_165)->view_0 = view_49;
    (&kernelContext_165)->objects_0 = objects_49;
    (&kernelContext_165)->shot_bodies_0 = shot_bodies_49;
    (&kernelContext_165)->lines_0 = lines_49;
    (&kernelContext_165)->geometry_0 = geometry_49;
    (&kernelContext_165)->shots_0 = shots_49;
    (&kernelContext_165)->lp_0 = lp_49;
    (&kernelContext_165)->ext_0 = ext_51;
    (&kernelContext_165)->engine_contacts_0 = engine_contacts_49;
    (&kernelContext_165)->crush_0 = crush_49;
    (&kernelContext_165)->crush_rw_0 = crush_rw_49;
    (&kernelContext_165)->crush_tab_0 = crush_tab_49;
    threadgroup array<uint, int(1024)> g_scan_49;
    (&kernelContext_165)->g_scan_0 = &g_scan_49;
    threadgroup array<float, int(64)> g_il_best_49;
    (&kernelContext_165)->g_il_best_0 = &g_il_best_49;
    if((params_49->shot_count_0) > 0U)
    {
        il_breaks_1(id_45.x, &kernelContext_165);
    }
    return;
}

[[kernel]] void fr_csl_begin(uint3 id_46 [[thread_position_in_grid]], FractureParams_0 constant* params_50 [[buffer(0)]], uint device* work_50 [[buffer(4)]], packed_float4 device* frames_50 [[buffer(6)]], packed_float4 device* bodies_50 [[buffer(7)]], packed_float4 device* chunks_50 [[buffer(1)]], uint device* sort_50 [[buffer(11)]], atomic<uint> device* atom_50 [[buffer(5)]], Record_natural_0 device* manifolds_50 [[buffer(10)]], packed_float4 device* loads_50 [[buffer(12)]], packed_uint4 device* bonds_50 [[buffer(2)]], uint device* args_50 [[buffer(13)]], uint device* info_50 [[buffer(15)]], ScanParams_0 constant* scan_50 [[buffer(14)]], packed_float4 device* parts_50 [[buffer(3)]], packed_float4 device* colliders_50 [[buffer(8)]], packed_uint4 device* collider_info_50 [[buffer(9)]], packed_float4 device* view_chunks_50 [[buffer(17)]], ViewParams_0 constant* view_50 [[buffer(16)]], packed_float4 device* objects_50 [[buffer(18)]], uint device* shot_bodies_50 [[buffer(20)]], float device* lines_50 [[buffer(19)]], packed_float4 device* geometry_50 [[buffer(21)]], packed_float4 device* shots_50 [[buffer(22)]], packed_float4 device* lp_50 [[buffer(24)]], packed_float4 device* ext_52 [[buffer(23)]], packed_float4 device* engine_contacts_50 [[buffer(25)]], CrushParams_0 constant* crush_50 [[buffer(28)]], packed_float4 device* crush_rw_50 [[buffer(27)]], packed_float4 device* crush_tab_50 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_166;
    (&kernelContext_166)->params_0 = params_50;
    (&kernelContext_166)->work_0 = work_50;
    (&kernelContext_166)->frames_0 = frames_50;
    (&kernelContext_166)->bodies_0 = bodies_50;
    (&kernelContext_166)->chunks_0 = chunks_50;
    (&kernelContext_166)->sort_0 = sort_50;
    (&kernelContext_166)->atom_0 = atom_50;
    (&kernelContext_166)->manifolds_0 = manifolds_50;
    (&kernelContext_166)->loads_0 = loads_50;
    (&kernelContext_166)->bonds_0 = bonds_50;
    (&kernelContext_166)->args_0 = args_50;
    (&kernelContext_166)->info_0 = info_50;
    (&kernelContext_166)->scan_0 = scan_50;
    (&kernelContext_166)->parts_0 = parts_50;
    (&kernelContext_166)->colliders_0 = colliders_50;
    (&kernelContext_166)->collider_info_0 = collider_info_50;
    (&kernelContext_166)->view_chunks_0 = view_chunks_50;
    (&kernelContext_166)->view_0 = view_50;
    (&kernelContext_166)->objects_0 = objects_50;
    (&kernelContext_166)->shot_bodies_0 = shot_bodies_50;
    (&kernelContext_166)->lines_0 = lines_50;
    (&kernelContext_166)->geometry_0 = geometry_50;
    (&kernelContext_166)->shots_0 = shots_50;
    (&kernelContext_166)->lp_0 = lp_50;
    (&kernelContext_166)->ext_0 = ext_52;
    (&kernelContext_166)->engine_contacts_0 = engine_contacts_50;
    (&kernelContext_166)->crush_0 = crush_50;
    (&kernelContext_166)->crush_rw_0 = crush_rw_50;
    (&kernelContext_166)->crush_tab_0 = crush_tab_50;
    threadgroup array<uint, int(1024)> g_scan_50;
    (&kernelContext_166)->g_scan_0 = &g_scan_50;
    threadgroup array<float, int(64)> g_il_best_50;
    (&kernelContext_166)->g_il_best_0 = &g_il_best_50;
    il_begin_0(id_46.x, &kernelContext_166);
    return;
}

[[kernel]] void fr_csl_tube_count(uint3 id_47 [[thread_position_in_grid]], FractureParams_0 constant* params_51 [[buffer(0)]], uint device* work_51 [[buffer(4)]], packed_float4 device* frames_51 [[buffer(6)]], packed_float4 device* bodies_51 [[buffer(7)]], packed_float4 device* chunks_51 [[buffer(1)]], uint device* sort_51 [[buffer(11)]], atomic<uint> device* atom_51 [[buffer(5)]], Record_natural_0 device* manifolds_51 [[buffer(10)]], packed_float4 device* loads_51 [[buffer(12)]], packed_uint4 device* bonds_51 [[buffer(2)]], uint device* args_51 [[buffer(13)]], uint device* info_51 [[buffer(15)]], ScanParams_0 constant* scan_51 [[buffer(14)]], packed_float4 device* parts_51 [[buffer(3)]], packed_float4 device* colliders_51 [[buffer(8)]], packed_uint4 device* collider_info_51 [[buffer(9)]], packed_float4 device* view_chunks_51 [[buffer(17)]], ViewParams_0 constant* view_51 [[buffer(16)]], packed_float4 device* objects_51 [[buffer(18)]], uint device* shot_bodies_51 [[buffer(20)]], float device* lines_51 [[buffer(19)]], packed_float4 device* geometry_51 [[buffer(21)]], packed_float4 device* shots_51 [[buffer(22)]], packed_float4 device* lp_51 [[buffer(24)]], packed_float4 device* ext_53 [[buffer(23)]], packed_float4 device* engine_contacts_51 [[buffer(25)]], CrushParams_0 constant* crush_51 [[buffer(28)]], packed_float4 device* crush_rw_51 [[buffer(27)]], packed_float4 device* crush_tab_51 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_167;
    (&kernelContext_167)->params_0 = params_51;
    (&kernelContext_167)->work_0 = work_51;
    (&kernelContext_167)->frames_0 = frames_51;
    (&kernelContext_167)->bodies_0 = bodies_51;
    (&kernelContext_167)->chunks_0 = chunks_51;
    (&kernelContext_167)->sort_0 = sort_51;
    (&kernelContext_167)->atom_0 = atom_51;
    (&kernelContext_167)->manifolds_0 = manifolds_51;
    (&kernelContext_167)->loads_0 = loads_51;
    (&kernelContext_167)->bonds_0 = bonds_51;
    (&kernelContext_167)->args_0 = args_51;
    (&kernelContext_167)->info_0 = info_51;
    (&kernelContext_167)->scan_0 = scan_51;
    (&kernelContext_167)->parts_0 = parts_51;
    (&kernelContext_167)->colliders_0 = colliders_51;
    (&kernelContext_167)->collider_info_0 = collider_info_51;
    (&kernelContext_167)->view_chunks_0 = view_chunks_51;
    (&kernelContext_167)->view_0 = view_51;
    (&kernelContext_167)->objects_0 = objects_51;
    (&kernelContext_167)->shot_bodies_0 = shot_bodies_51;
    (&kernelContext_167)->lines_0 = lines_51;
    (&kernelContext_167)->geometry_0 = geometry_51;
    (&kernelContext_167)->shots_0 = shots_51;
    (&kernelContext_167)->lp_0 = lp_51;
    (&kernelContext_167)->ext_0 = ext_53;
    (&kernelContext_167)->engine_contacts_0 = engine_contacts_51;
    (&kernelContext_167)->crush_0 = crush_51;
    (&kernelContext_167)->crush_rw_0 = crush_rw_51;
    (&kernelContext_167)->crush_tab_0 = crush_tab_51;
    threadgroup array<uint, int(1024)> g_scan_51;
    (&kernelContext_167)->g_scan_0 = &g_scan_51;
    threadgroup array<float, int(64)> g_il_best_51;
    (&kernelContext_167)->g_il_best_0 = &g_il_best_51;
    il_tube_count_0(id_47.x, &kernelContext_167);
    return;
}

[[kernel]] void fr_csl_tube_base(uint3 id_48 [[thread_position_in_grid]], FractureParams_0 constant* params_52 [[buffer(0)]], uint device* work_52 [[buffer(4)]], packed_float4 device* frames_52 [[buffer(6)]], packed_float4 device* bodies_52 [[buffer(7)]], packed_float4 device* chunks_52 [[buffer(1)]], uint device* sort_52 [[buffer(11)]], atomic<uint> device* atom_52 [[buffer(5)]], Record_natural_0 device* manifolds_52 [[buffer(10)]], packed_float4 device* loads_52 [[buffer(12)]], packed_uint4 device* bonds_52 [[buffer(2)]], uint device* args_52 [[buffer(13)]], uint device* info_52 [[buffer(15)]], ScanParams_0 constant* scan_52 [[buffer(14)]], packed_float4 device* parts_52 [[buffer(3)]], packed_float4 device* colliders_52 [[buffer(8)]], packed_uint4 device* collider_info_52 [[buffer(9)]], packed_float4 device* view_chunks_52 [[buffer(17)]], ViewParams_0 constant* view_52 [[buffer(16)]], packed_float4 device* objects_52 [[buffer(18)]], uint device* shot_bodies_52 [[buffer(20)]], float device* lines_52 [[buffer(19)]], packed_float4 device* geometry_52 [[buffer(21)]], packed_float4 device* shots_52 [[buffer(22)]], packed_float4 device* lp_52 [[buffer(24)]], packed_float4 device* ext_54 [[buffer(23)]], packed_float4 device* engine_contacts_52 [[buffer(25)]], CrushParams_0 constant* crush_52 [[buffer(28)]], packed_float4 device* crush_rw_52 [[buffer(27)]], packed_float4 device* crush_tab_52 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_168;
    (&kernelContext_168)->params_0 = params_52;
    (&kernelContext_168)->work_0 = work_52;
    (&kernelContext_168)->frames_0 = frames_52;
    (&kernelContext_168)->bodies_0 = bodies_52;
    (&kernelContext_168)->chunks_0 = chunks_52;
    (&kernelContext_168)->sort_0 = sort_52;
    (&kernelContext_168)->atom_0 = atom_52;
    (&kernelContext_168)->manifolds_0 = manifolds_52;
    (&kernelContext_168)->loads_0 = loads_52;
    (&kernelContext_168)->bonds_0 = bonds_52;
    (&kernelContext_168)->args_0 = args_52;
    (&kernelContext_168)->info_0 = info_52;
    (&kernelContext_168)->scan_0 = scan_52;
    (&kernelContext_168)->parts_0 = parts_52;
    (&kernelContext_168)->colliders_0 = colliders_52;
    (&kernelContext_168)->collider_info_0 = collider_info_52;
    (&kernelContext_168)->view_chunks_0 = view_chunks_52;
    (&kernelContext_168)->view_0 = view_52;
    (&kernelContext_168)->objects_0 = objects_52;
    (&kernelContext_168)->shot_bodies_0 = shot_bodies_52;
    (&kernelContext_168)->lines_0 = lines_52;
    (&kernelContext_168)->geometry_0 = geometry_52;
    (&kernelContext_168)->shots_0 = shots_52;
    (&kernelContext_168)->lp_0 = lp_52;
    (&kernelContext_168)->ext_0 = ext_54;
    (&kernelContext_168)->engine_contacts_0 = engine_contacts_52;
    (&kernelContext_168)->crush_0 = crush_52;
    (&kernelContext_168)->crush_rw_0 = crush_rw_52;
    (&kernelContext_168)->crush_tab_0 = crush_tab_52;
    threadgroup array<uint, int(1024)> g_scan_52;
    (&kernelContext_168)->g_scan_0 = &g_scan_52;
    threadgroup array<float, int(64)> g_il_best_52;
    (&kernelContext_168)->g_il_best_0 = &g_il_best_52;
    il_tube_base_0(id_48.x, &kernelContext_168);
    return;
}

[[kernel]] void fr_csl_tube_collect(uint3 id_49 [[thread_position_in_grid]], FractureParams_0 constant* params_53 [[buffer(0)]], uint device* work_53 [[buffer(4)]], packed_float4 device* frames_53 [[buffer(6)]], packed_float4 device* bodies_53 [[buffer(7)]], packed_float4 device* chunks_53 [[buffer(1)]], uint device* sort_53 [[buffer(11)]], atomic<uint> device* atom_53 [[buffer(5)]], Record_natural_0 device* manifolds_53 [[buffer(10)]], packed_float4 device* loads_53 [[buffer(12)]], packed_uint4 device* bonds_53 [[buffer(2)]], uint device* args_53 [[buffer(13)]], uint device* info_53 [[buffer(15)]], ScanParams_0 constant* scan_53 [[buffer(14)]], packed_float4 device* parts_53 [[buffer(3)]], packed_float4 device* colliders_53 [[buffer(8)]], packed_uint4 device* collider_info_53 [[buffer(9)]], packed_float4 device* view_chunks_53 [[buffer(17)]], ViewParams_0 constant* view_53 [[buffer(16)]], packed_float4 device* objects_53 [[buffer(18)]], uint device* shot_bodies_53 [[buffer(20)]], float device* lines_53 [[buffer(19)]], packed_float4 device* geometry_53 [[buffer(21)]], packed_float4 device* shots_53 [[buffer(22)]], packed_float4 device* lp_53 [[buffer(24)]], packed_float4 device* ext_55 [[buffer(23)]], packed_float4 device* engine_contacts_53 [[buffer(25)]], CrushParams_0 constant* crush_53 [[buffer(28)]], packed_float4 device* crush_rw_53 [[buffer(27)]], packed_float4 device* crush_tab_53 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_169;
    (&kernelContext_169)->params_0 = params_53;
    (&kernelContext_169)->work_0 = work_53;
    (&kernelContext_169)->frames_0 = frames_53;
    (&kernelContext_169)->bodies_0 = bodies_53;
    (&kernelContext_169)->chunks_0 = chunks_53;
    (&kernelContext_169)->sort_0 = sort_53;
    (&kernelContext_169)->atom_0 = atom_53;
    (&kernelContext_169)->manifolds_0 = manifolds_53;
    (&kernelContext_169)->loads_0 = loads_53;
    (&kernelContext_169)->bonds_0 = bonds_53;
    (&kernelContext_169)->args_0 = args_53;
    (&kernelContext_169)->info_0 = info_53;
    (&kernelContext_169)->scan_0 = scan_53;
    (&kernelContext_169)->parts_0 = parts_53;
    (&kernelContext_169)->colliders_0 = colliders_53;
    (&kernelContext_169)->collider_info_0 = collider_info_53;
    (&kernelContext_169)->view_chunks_0 = view_chunks_53;
    (&kernelContext_169)->view_0 = view_53;
    (&kernelContext_169)->objects_0 = objects_53;
    (&kernelContext_169)->shot_bodies_0 = shot_bodies_53;
    (&kernelContext_169)->lines_0 = lines_53;
    (&kernelContext_169)->geometry_0 = geometry_53;
    (&kernelContext_169)->shots_0 = shots_53;
    (&kernelContext_169)->lp_0 = lp_53;
    (&kernelContext_169)->ext_0 = ext_55;
    (&kernelContext_169)->engine_contacts_0 = engine_contacts_53;
    (&kernelContext_169)->crush_0 = crush_53;
    (&kernelContext_169)->crush_rw_0 = crush_rw_53;
    (&kernelContext_169)->crush_tab_0 = crush_tab_53;
    threadgroup array<uint, int(1024)> g_scan_53;
    (&kernelContext_169)->g_scan_0 = &g_scan_53;
    threadgroup array<float, int(64)> g_il_best_53;
    (&kernelContext_169)->g_il_best_0 = &g_il_best_53;
    il_tube_collect_0(id_49.x, &kernelContext_169);
    return;
}

uint w_cs_list_0(KernelContext_0 thread* kernelContext_170)
{
    uint _S1175 = w_carry_0(kernelContext_170);
    return _S1175 + kernelContext_170->params_0->chunk_count_0;
}

uint w_cs_bodies_0(KernelContext_0 thread* kernelContext_171)
{
    uint _S1176 = w_cs_list_0(kernelContext_171);
    return _S1176 + kernelContext_171->params_0->chunk_count_0;
}

[[kernel]] void fr_csl_run(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_54 [[buffer(0)]], uint device* work_54 [[buffer(4)]], packed_float4 device* frames_54 [[buffer(6)]], packed_float4 device* bodies_54 [[buffer(7)]], packed_float4 device* chunks_54 [[buffer(1)]], uint device* sort_54 [[buffer(11)]], atomic<uint> device* atom_54 [[buffer(5)]], Record_natural_0 device* manifolds_54 [[buffer(10)]], packed_float4 device* loads_54 [[buffer(12)]], packed_uint4 device* bonds_54 [[buffer(2)]], uint device* args_54 [[buffer(13)]], uint device* info_54 [[buffer(15)]], ScanParams_0 constant* scan_54 [[buffer(14)]], packed_float4 device* parts_54 [[buffer(3)]], packed_float4 device* colliders_54 [[buffer(8)]], packed_uint4 device* collider_info_54 [[buffer(9)]], packed_float4 device* view_chunks_54 [[buffer(17)]], ViewParams_0 constant* view_54 [[buffer(16)]], packed_float4 device* objects_54 [[buffer(18)]], uint device* shot_bodies_54 [[buffer(20)]], float device* lines_54 [[buffer(19)]], packed_float4 device* geometry_54 [[buffer(21)]], packed_float4 device* shots_54 [[buffer(22)]], packed_float4 device* lp_54 [[buffer(24)]], packed_float4 device* ext_56 [[buffer(23)]], packed_float4 device* engine_contacts_54 [[buffer(25)]], CrushParams_0 constant* crush_54 [[buffer(28)]], packed_float4 device* crush_rw_54 [[buffer(27)]], packed_float4 device* crush_tab_54 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_172;
    (&kernelContext_172)->params_0 = params_54;
    (&kernelContext_172)->work_0 = work_54;
    (&kernelContext_172)->frames_0 = frames_54;
    (&kernelContext_172)->bodies_0 = bodies_54;
    (&kernelContext_172)->chunks_0 = chunks_54;
    (&kernelContext_172)->sort_0 = sort_54;
    (&kernelContext_172)->atom_0 = atom_54;
    (&kernelContext_172)->manifolds_0 = manifolds_54;
    (&kernelContext_172)->loads_0 = loads_54;
    (&kernelContext_172)->bonds_0 = bonds_54;
    (&kernelContext_172)->args_0 = args_54;
    (&kernelContext_172)->info_0 = info_54;
    (&kernelContext_172)->scan_0 = scan_54;
    (&kernelContext_172)->parts_0 = parts_54;
    (&kernelContext_172)->colliders_0 = colliders_54;
    (&kernelContext_172)->collider_info_0 = collider_info_54;
    (&kernelContext_172)->view_chunks_0 = view_chunks_54;
    (&kernelContext_172)->view_0 = view_54;
    (&kernelContext_172)->objects_0 = objects_54;
    (&kernelContext_172)->shot_bodies_0 = shot_bodies_54;
    (&kernelContext_172)->lines_0 = lines_54;
    (&kernelContext_172)->geometry_0 = geometry_54;
    (&kernelContext_172)->shots_0 = shots_54;
    (&kernelContext_172)->lp_0 = lp_54;
    (&kernelContext_172)->ext_0 = ext_56;
    (&kernelContext_172)->engine_contacts_0 = engine_contacts_54;
    (&kernelContext_172)->crush_0 = crush_54;
    (&kernelContext_172)->crush_rw_0 = crush_rw_54;
    (&kernelContext_172)->crush_tab_0 = crush_tab_54;
    threadgroup array<uint, int(1024)> g_scan_54;
    (&kernelContext_172)->g_scan_0 = &g_scan_54;
    threadgroup array<float, int(64)> g_il_best_54;
    (&kernelContext_172)->g_il_best_0 = &g_il_best_54;
    uint _S1177 = w_cs_bodies_0(&kernelContext_172);
    il_run_0(*((&kernelContext_172)->work_0+(_S1177 + group_2.x)), thread_5.x, &kernelContext_172);
    return;
}

[[kernel]] void fr_csl_strike(uint3 id_50 [[thread_position_in_grid]], FractureParams_0 constant* params_55 [[buffer(0)]], uint device* work_55 [[buffer(4)]], packed_float4 device* frames_55 [[buffer(6)]], packed_float4 device* bodies_55 [[buffer(7)]], packed_float4 device* chunks_55 [[buffer(1)]], uint device* sort_55 [[buffer(11)]], atomic<uint> device* atom_55 [[buffer(5)]], Record_natural_0 device* manifolds_55 [[buffer(10)]], packed_float4 device* loads_55 [[buffer(12)]], packed_uint4 device* bonds_55 [[buffer(2)]], uint device* args_55 [[buffer(13)]], uint device* info_55 [[buffer(15)]], ScanParams_0 constant* scan_55 [[buffer(14)]], packed_float4 device* parts_55 [[buffer(3)]], packed_float4 device* colliders_55 [[buffer(8)]], packed_uint4 device* collider_info_55 [[buffer(9)]], packed_float4 device* view_chunks_55 [[buffer(17)]], ViewParams_0 constant* view_55 [[buffer(16)]], packed_float4 device* objects_55 [[buffer(18)]], uint device* shot_bodies_55 [[buffer(20)]], float device* lines_55 [[buffer(19)]], packed_float4 device* geometry_55 [[buffer(21)]], packed_float4 device* shots_55 [[buffer(22)]], packed_float4 device* lp_55 [[buffer(24)]], packed_float4 device* ext_57 [[buffer(23)]], packed_float4 device* engine_contacts_55 [[buffer(25)]], CrushParams_0 constant* crush_55 [[buffer(28)]], packed_float4 device* crush_rw_55 [[buffer(27)]], packed_float4 device* crush_tab_55 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_173;
    (&kernelContext_173)->params_0 = params_55;
    (&kernelContext_173)->work_0 = work_55;
    (&kernelContext_173)->frames_0 = frames_55;
    (&kernelContext_173)->bodies_0 = bodies_55;
    (&kernelContext_173)->chunks_0 = chunks_55;
    (&kernelContext_173)->sort_0 = sort_55;
    (&kernelContext_173)->atom_0 = atom_55;
    (&kernelContext_173)->manifolds_0 = manifolds_55;
    (&kernelContext_173)->loads_0 = loads_55;
    (&kernelContext_173)->bonds_0 = bonds_55;
    (&kernelContext_173)->args_0 = args_55;
    (&kernelContext_173)->info_0 = info_55;
    (&kernelContext_173)->scan_0 = scan_55;
    (&kernelContext_173)->parts_0 = parts_55;
    (&kernelContext_173)->colliders_0 = colliders_55;
    (&kernelContext_173)->collider_info_0 = collider_info_55;
    (&kernelContext_173)->view_chunks_0 = view_chunks_55;
    (&kernelContext_173)->view_0 = view_55;
    (&kernelContext_173)->objects_0 = objects_55;
    (&kernelContext_173)->shot_bodies_0 = shot_bodies_55;
    (&kernelContext_173)->lines_0 = lines_55;
    (&kernelContext_173)->geometry_0 = geometry_55;
    (&kernelContext_173)->shots_0 = shots_55;
    (&kernelContext_173)->lp_0 = lp_55;
    (&kernelContext_173)->ext_0 = ext_57;
    (&kernelContext_173)->engine_contacts_0 = engine_contacts_55;
    (&kernelContext_173)->crush_0 = crush_55;
    (&kernelContext_173)->crush_rw_0 = crush_rw_55;
    (&kernelContext_173)->crush_tab_0 = crush_tab_55;
    threadgroup array<uint, int(1024)> g_scan_55;
    (&kernelContext_173)->g_scan_0 = &g_scan_55;
    threadgroup array<float, int(64)> g_il_best_55;
    (&kernelContext_173)->g_il_best_0 = &g_il_best_55;
    il_strike_0(id_50.x, &kernelContext_173);
    return;
}

[[kernel]] void fr_csl_mark(uint3 id_51 [[thread_position_in_grid]], FractureParams_0 constant* params_56 [[buffer(0)]], uint device* work_56 [[buffer(4)]], packed_float4 device* frames_56 [[buffer(6)]], packed_float4 device* bodies_56 [[buffer(7)]], packed_float4 device* chunks_56 [[buffer(1)]], uint device* sort_56 [[buffer(11)]], atomic<uint> device* atom_56 [[buffer(5)]], Record_natural_0 device* manifolds_56 [[buffer(10)]], packed_float4 device* loads_56 [[buffer(12)]], packed_uint4 device* bonds_56 [[buffer(2)]], uint device* args_56 [[buffer(13)]], uint device* info_56 [[buffer(15)]], ScanParams_0 constant* scan_56 [[buffer(14)]], packed_float4 device* parts_56 [[buffer(3)]], packed_float4 device* colliders_56 [[buffer(8)]], packed_uint4 device* collider_info_56 [[buffer(9)]], packed_float4 device* view_chunks_56 [[buffer(17)]], ViewParams_0 constant* view_56 [[buffer(16)]], packed_float4 device* objects_56 [[buffer(18)]], uint device* shot_bodies_56 [[buffer(20)]], float device* lines_56 [[buffer(19)]], packed_float4 device* geometry_56 [[buffer(21)]], packed_float4 device* shots_56 [[buffer(22)]], packed_float4 device* lp_56 [[buffer(24)]], packed_float4 device* ext_58 [[buffer(23)]], packed_float4 device* engine_contacts_56 [[buffer(25)]], CrushParams_0 constant* crush_56 [[buffer(28)]], packed_float4 device* crush_rw_56 [[buffer(27)]], packed_float4 device* crush_tab_56 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_174;
    (&kernelContext_174)->params_0 = params_56;
    (&kernelContext_174)->work_0 = work_56;
    (&kernelContext_174)->frames_0 = frames_56;
    (&kernelContext_174)->bodies_0 = bodies_56;
    (&kernelContext_174)->chunks_0 = chunks_56;
    (&kernelContext_174)->sort_0 = sort_56;
    (&kernelContext_174)->atom_0 = atom_56;
    (&kernelContext_174)->manifolds_0 = manifolds_56;
    (&kernelContext_174)->loads_0 = loads_56;
    (&kernelContext_174)->bonds_0 = bonds_56;
    (&kernelContext_174)->args_0 = args_56;
    (&kernelContext_174)->info_0 = info_56;
    (&kernelContext_174)->scan_0 = scan_56;
    (&kernelContext_174)->parts_0 = parts_56;
    (&kernelContext_174)->colliders_0 = colliders_56;
    (&kernelContext_174)->collider_info_0 = collider_info_56;
    (&kernelContext_174)->view_chunks_0 = view_chunks_56;
    (&kernelContext_174)->view_0 = view_56;
    (&kernelContext_174)->objects_0 = objects_56;
    (&kernelContext_174)->shot_bodies_0 = shot_bodies_56;
    (&kernelContext_174)->lines_0 = lines_56;
    (&kernelContext_174)->geometry_0 = geometry_56;
    (&kernelContext_174)->shots_0 = shots_56;
    (&kernelContext_174)->lp_0 = lp_56;
    (&kernelContext_174)->ext_0 = ext_58;
    (&kernelContext_174)->engine_contacts_0 = engine_contacts_56;
    (&kernelContext_174)->crush_0 = crush_56;
    (&kernelContext_174)->crush_rw_0 = crush_rw_56;
    (&kernelContext_174)->crush_tab_0 = crush_tab_56;
    threadgroup array<uint, int(1024)> g_scan_56;
    (&kernelContext_174)->g_scan_0 = &g_scan_56;
    threadgroup array<float, int(64)> g_il_best_56;
    (&kernelContext_174)->g_il_best_0 = &g_il_best_56;
    il_mark_0(id_51.x, &kernelContext_174);
    return;
}

[[kernel]] void fr_csl_carry(uint3 id_52 [[thread_position_in_grid]], FractureParams_0 constant* params_57 [[buffer(0)]], uint device* work_57 [[buffer(4)]], packed_float4 device* frames_57 [[buffer(6)]], packed_float4 device* bodies_57 [[buffer(7)]], packed_float4 device* chunks_57 [[buffer(1)]], uint device* sort_57 [[buffer(11)]], atomic<uint> device* atom_57 [[buffer(5)]], Record_natural_0 device* manifolds_57 [[buffer(10)]], packed_float4 device* loads_57 [[buffer(12)]], packed_uint4 device* bonds_57 [[buffer(2)]], uint device* args_57 [[buffer(13)]], uint device* info_57 [[buffer(15)]], ScanParams_0 constant* scan_57 [[buffer(14)]], packed_float4 device* parts_57 [[buffer(3)]], packed_float4 device* colliders_57 [[buffer(8)]], packed_uint4 device* collider_info_57 [[buffer(9)]], packed_float4 device* view_chunks_57 [[buffer(17)]], ViewParams_0 constant* view_57 [[buffer(16)]], packed_float4 device* objects_57 [[buffer(18)]], uint device* shot_bodies_57 [[buffer(20)]], float device* lines_57 [[buffer(19)]], packed_float4 device* geometry_57 [[buffer(21)]], packed_float4 device* shots_57 [[buffer(22)]], packed_float4 device* lp_57 [[buffer(24)]], packed_float4 device* ext_59 [[buffer(23)]], packed_float4 device* engine_contacts_57 [[buffer(25)]], CrushParams_0 constant* crush_57 [[buffer(28)]], packed_float4 device* crush_rw_57 [[buffer(27)]], packed_float4 device* crush_tab_57 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_175;
    (&kernelContext_175)->params_0 = params_57;
    (&kernelContext_175)->work_0 = work_57;
    (&kernelContext_175)->frames_0 = frames_57;
    (&kernelContext_175)->bodies_0 = bodies_57;
    (&kernelContext_175)->chunks_0 = chunks_57;
    (&kernelContext_175)->sort_0 = sort_57;
    (&kernelContext_175)->atom_0 = atom_57;
    (&kernelContext_175)->manifolds_0 = manifolds_57;
    (&kernelContext_175)->loads_0 = loads_57;
    (&kernelContext_175)->bonds_0 = bonds_57;
    (&kernelContext_175)->args_0 = args_57;
    (&kernelContext_175)->info_0 = info_57;
    (&kernelContext_175)->scan_0 = scan_57;
    (&kernelContext_175)->parts_0 = parts_57;
    (&kernelContext_175)->colliders_0 = colliders_57;
    (&kernelContext_175)->collider_info_0 = collider_info_57;
    (&kernelContext_175)->view_chunks_0 = view_chunks_57;
    (&kernelContext_175)->view_0 = view_57;
    (&kernelContext_175)->objects_0 = objects_57;
    (&kernelContext_175)->shot_bodies_0 = shot_bodies_57;
    (&kernelContext_175)->lines_0 = lines_57;
    (&kernelContext_175)->geometry_0 = geometry_57;
    (&kernelContext_175)->shots_0 = shots_57;
    (&kernelContext_175)->lp_0 = lp_57;
    (&kernelContext_175)->ext_0 = ext_59;
    (&kernelContext_175)->engine_contacts_0 = engine_contacts_57;
    (&kernelContext_175)->crush_0 = crush_57;
    (&kernelContext_175)->crush_rw_0 = crush_rw_57;
    (&kernelContext_175)->crush_tab_0 = crush_tab_57;
    threadgroup array<uint, int(1024)> g_scan_57;
    (&kernelContext_175)->g_scan_0 = &g_scan_57;
    threadgroup array<float, int(64)> g_il_best_57;
    (&kernelContext_175)->g_il_best_0 = &g_il_best_57;
    il_carry_0(id_52.x, &kernelContext_175);
    return;
}

[[kernel]] void fr_csl_breaks(uint3 id_53 [[thread_position_in_grid]], FractureParams_0 constant* params_58 [[buffer(0)]], uint device* work_58 [[buffer(4)]], packed_float4 device* frames_58 [[buffer(6)]], packed_float4 device* bodies_58 [[buffer(7)]], packed_float4 device* chunks_58 [[buffer(1)]], uint device* sort_58 [[buffer(11)]], atomic<uint> device* atom_58 [[buffer(5)]], Record_natural_0 device* manifolds_58 [[buffer(10)]], packed_float4 device* loads_58 [[buffer(12)]], packed_uint4 device* bonds_58 [[buffer(2)]], uint device* args_58 [[buffer(13)]], uint device* info_58 [[buffer(15)]], ScanParams_0 constant* scan_58 [[buffer(14)]], packed_float4 device* parts_58 [[buffer(3)]], packed_float4 device* colliders_58 [[buffer(8)]], packed_uint4 device* collider_info_58 [[buffer(9)]], packed_float4 device* view_chunks_58 [[buffer(17)]], ViewParams_0 constant* view_58 [[buffer(16)]], packed_float4 device* objects_58 [[buffer(18)]], uint device* shot_bodies_58 [[buffer(20)]], float device* lines_58 [[buffer(19)]], packed_float4 device* geometry_58 [[buffer(21)]], packed_float4 device* shots_58 [[buffer(22)]], packed_float4 device* lp_58 [[buffer(24)]], packed_float4 device* ext_60 [[buffer(23)]], packed_float4 device* engine_contacts_58 [[buffer(25)]], CrushParams_0 constant* crush_58 [[buffer(28)]], packed_float4 device* crush_rw_58 [[buffer(27)]], packed_float4 device* crush_tab_58 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_176;
    (&kernelContext_176)->params_0 = params_58;
    (&kernelContext_176)->work_0 = work_58;
    (&kernelContext_176)->frames_0 = frames_58;
    (&kernelContext_176)->bodies_0 = bodies_58;
    (&kernelContext_176)->chunks_0 = chunks_58;
    (&kernelContext_176)->sort_0 = sort_58;
    (&kernelContext_176)->atom_0 = atom_58;
    (&kernelContext_176)->manifolds_0 = manifolds_58;
    (&kernelContext_176)->loads_0 = loads_58;
    (&kernelContext_176)->bonds_0 = bonds_58;
    (&kernelContext_176)->args_0 = args_58;
    (&kernelContext_176)->info_0 = info_58;
    (&kernelContext_176)->scan_0 = scan_58;
    (&kernelContext_176)->parts_0 = parts_58;
    (&kernelContext_176)->colliders_0 = colliders_58;
    (&kernelContext_176)->collider_info_0 = collider_info_58;
    (&kernelContext_176)->view_chunks_0 = view_chunks_58;
    (&kernelContext_176)->view_0 = view_58;
    (&kernelContext_176)->objects_0 = objects_58;
    (&kernelContext_176)->shot_bodies_0 = shot_bodies_58;
    (&kernelContext_176)->lines_0 = lines_58;
    (&kernelContext_176)->geometry_0 = geometry_58;
    (&kernelContext_176)->shots_0 = shots_58;
    (&kernelContext_176)->lp_0 = lp_58;
    (&kernelContext_176)->ext_0 = ext_60;
    (&kernelContext_176)->engine_contacts_0 = engine_contacts_58;
    (&kernelContext_176)->crush_0 = crush_58;
    (&kernelContext_176)->crush_rw_0 = crush_rw_58;
    (&kernelContext_176)->crush_tab_0 = crush_tab_58;
    threadgroup array<uint, int(1024)> g_scan_58;
    (&kernelContext_176)->g_scan_0 = &g_scan_58;
    threadgroup array<float, int(64)> g_il_best_58;
    (&kernelContext_176)->g_il_best_0 = &g_il_best_58;
    il_breaks_0(id_53.x, &kernelContext_176);
    return;
}

[[kernel]] void fr_cs_begin(uint3 id_54 [[thread_position_in_grid]], FractureParams_0 constant* params_59 [[buffer(0)]], uint device* work_59 [[buffer(4)]], packed_float4 device* frames_59 [[buffer(6)]], packed_float4 device* bodies_59 [[buffer(7)]], packed_float4 device* chunks_59 [[buffer(1)]], uint device* sort_59 [[buffer(11)]], atomic<uint> device* atom_59 [[buffer(5)]], Record_natural_0 device* manifolds_59 [[buffer(10)]], packed_float4 device* loads_59 [[buffer(12)]], packed_uint4 device* bonds_59 [[buffer(2)]], uint device* args_59 [[buffer(13)]], uint device* info_59 [[buffer(15)]], ScanParams_0 constant* scan_59 [[buffer(14)]], packed_float4 device* parts_59 [[buffer(3)]], packed_float4 device* colliders_59 [[buffer(8)]], packed_uint4 device* collider_info_59 [[buffer(9)]], packed_float4 device* view_chunks_59 [[buffer(17)]], ViewParams_0 constant* view_59 [[buffer(16)]], packed_float4 device* objects_59 [[buffer(18)]], uint device* shot_bodies_59 [[buffer(20)]], float device* lines_59 [[buffer(19)]], packed_float4 device* geometry_59 [[buffer(21)]], packed_float4 device* shots_59 [[buffer(22)]], packed_float4 device* lp_59 [[buffer(24)]], packed_float4 device* ext_61 [[buffer(23)]], packed_float4 device* engine_contacts_59 [[buffer(25)]], CrushParams_0 constant* crush_59 [[buffer(28)]], packed_float4 device* crush_rw_59 [[buffer(27)]], packed_float4 device* crush_tab_59 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_177;
    (&kernelContext_177)->params_0 = params_59;
    (&kernelContext_177)->work_0 = work_59;
    (&kernelContext_177)->frames_0 = frames_59;
    (&kernelContext_177)->bodies_0 = bodies_59;
    (&kernelContext_177)->chunks_0 = chunks_59;
    (&kernelContext_177)->sort_0 = sort_59;
    (&kernelContext_177)->atom_0 = atom_59;
    (&kernelContext_177)->manifolds_0 = manifolds_59;
    (&kernelContext_177)->loads_0 = loads_59;
    (&kernelContext_177)->bonds_0 = bonds_59;
    (&kernelContext_177)->args_0 = args_59;
    (&kernelContext_177)->info_0 = info_59;
    (&kernelContext_177)->scan_0 = scan_59;
    (&kernelContext_177)->parts_0 = parts_59;
    (&kernelContext_177)->colliders_0 = colliders_59;
    (&kernelContext_177)->collider_info_0 = collider_info_59;
    (&kernelContext_177)->view_chunks_0 = view_chunks_59;
    (&kernelContext_177)->view_0 = view_59;
    (&kernelContext_177)->objects_0 = objects_59;
    (&kernelContext_177)->shot_bodies_0 = shot_bodies_59;
    (&kernelContext_177)->lines_0 = lines_59;
    (&kernelContext_177)->geometry_0 = geometry_59;
    (&kernelContext_177)->shots_0 = shots_59;
    (&kernelContext_177)->lp_0 = lp_59;
    (&kernelContext_177)->ext_0 = ext_61;
    (&kernelContext_177)->engine_contacts_0 = engine_contacts_59;
    (&kernelContext_177)->crush_0 = crush_59;
    (&kernelContext_177)->crush_rw_0 = crush_rw_59;
    (&kernelContext_177)->crush_tab_0 = crush_tab_59;
    threadgroup array<uint, int(1024)> g_scan_59;
    (&kernelContext_177)->g_scan_0 = &g_scan_59;
    threadgroup array<float, int(64)> g_il_best_59;
    (&kernelContext_177)->g_il_best_0 = &g_il_best_59;
    uint _S1178 = a_counters_1(&kernelContext_177);
    atomic_store_explicit((&kernelContext_177)->atom_0+(_S1178 + 18U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_177)->atom_0+(_S1178 + 19U), 0U, memory_order_relaxed);
    uint _S1179 = a_il_cursor_1(&kernelContext_177);
    atomic_store_explicit((&kernelContext_177)->atom_0+_S1179, 0U, memory_order_relaxed);
    uint _S1180 = cs_rows_0(&kernelContext_177);
    uint _S1181 = groups_of_0(_S1180, 64U);
    set_args_0(7U, _S1181, &kernelContext_177);
    return;
}

[[kernel]] void fr_cs_contacts(uint3 id_55 [[thread_position_in_grid]], FractureParams_0 constant* params_60 [[buffer(0)]], uint device* work_60 [[buffer(4)]], packed_float4 device* frames_60 [[buffer(6)]], packed_float4 device* bodies_60 [[buffer(7)]], packed_float4 device* chunks_60 [[buffer(1)]], uint device* sort_60 [[buffer(11)]], atomic<uint> device* atom_60 [[buffer(5)]], Record_natural_0 device* manifolds_60 [[buffer(10)]], packed_float4 device* loads_60 [[buffer(12)]], packed_uint4 device* bonds_60 [[buffer(2)]], uint device* args_60 [[buffer(13)]], uint device* info_60 [[buffer(15)]], ScanParams_0 constant* scan_60 [[buffer(14)]], packed_float4 device* parts_60 [[buffer(3)]], packed_float4 device* colliders_60 [[buffer(8)]], packed_uint4 device* collider_info_60 [[buffer(9)]], packed_float4 device* view_chunks_60 [[buffer(17)]], ViewParams_0 constant* view_60 [[buffer(16)]], packed_float4 device* objects_60 [[buffer(18)]], uint device* shot_bodies_60 [[buffer(20)]], float device* lines_60 [[buffer(19)]], packed_float4 device* geometry_60 [[buffer(21)]], packed_float4 device* shots_60 [[buffer(22)]], packed_float4 device* lp_60 [[buffer(24)]], packed_float4 device* ext_62 [[buffer(23)]], packed_float4 device* engine_contacts_60 [[buffer(25)]], CrushParams_0 constant* crush_60 [[buffer(28)]], packed_float4 device* crush_rw_60 [[buffer(27)]], packed_float4 device* crush_tab_60 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_178;
    (&kernelContext_178)->params_0 = params_60;
    (&kernelContext_178)->work_0 = work_60;
    (&kernelContext_178)->frames_0 = frames_60;
    (&kernelContext_178)->bodies_0 = bodies_60;
    (&kernelContext_178)->chunks_0 = chunks_60;
    (&kernelContext_178)->sort_0 = sort_60;
    (&kernelContext_178)->atom_0 = atom_60;
    (&kernelContext_178)->manifolds_0 = manifolds_60;
    (&kernelContext_178)->loads_0 = loads_60;
    (&kernelContext_178)->bonds_0 = bonds_60;
    (&kernelContext_178)->args_0 = args_60;
    (&kernelContext_178)->info_0 = info_60;
    (&kernelContext_178)->scan_0 = scan_60;
    (&kernelContext_178)->parts_0 = parts_60;
    (&kernelContext_178)->colliders_0 = colliders_60;
    (&kernelContext_178)->collider_info_0 = collider_info_60;
    (&kernelContext_178)->view_chunks_0 = view_chunks_60;
    (&kernelContext_178)->view_0 = view_60;
    (&kernelContext_178)->objects_0 = objects_60;
    (&kernelContext_178)->shot_bodies_0 = shot_bodies_60;
    (&kernelContext_178)->lines_0 = lines_60;
    (&kernelContext_178)->geometry_0 = geometry_60;
    (&kernelContext_178)->shots_0 = shots_60;
    (&kernelContext_178)->lp_0 = lp_60;
    (&kernelContext_178)->ext_0 = ext_62;
    (&kernelContext_178)->engine_contacts_0 = engine_contacts_60;
    (&kernelContext_178)->crush_0 = crush_60;
    (&kernelContext_178)->crush_rw_0 = crush_rw_60;
    (&kernelContext_178)->crush_tab_0 = crush_tab_60;
    threadgroup array<uint, int(1024)> g_scan_60;
    (&kernelContext_178)->g_scan_0 = &g_scan_60;
    threadgroup array<float, int(64)> g_il_best_60;
    (&kernelContext_178)->g_il_best_0 = &g_il_best_60;
    uint r_24 = id_55.x;
    uint _S1182 = cs_rows_0(&kernelContext_178);
    if(r_24 >= _S1182)
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
        bool _S1183 = cs_side_0(r_24, side_6, &k_44, &s_17, &ob_5, &co_5, &p_18, &u_9, &v_14, &m_eff_7, &t_in_8, &kernelContext_178);
        if(_S1183)
        {
            uint _S1184 = a_cs_0(s_17, &kernelContext_178);
            uint _S1185 = atomic_fetch_max_explicit((&kernelContext_178)->atom_0+_S1184, (as_type<uint>((m_eff_7 * v_14))), memory_order_relaxed);
            uint _S1186 = a_cs_0(s_17, &kernelContext_178);
            uint _S1187 = atomic_exchange_explicit((&kernelContext_178)->atom_0+(_S1186 + 8U), 1U, memory_order_relaxed);
            if(_S1187 == 0U)
            {
                uint _S1188 = w_cs_bodies_0(&kernelContext_178);
                uint _S1189 = a_counters_1(&kernelContext_178);
                uint _S1190 = atomic_fetch_add_explicit((&kernelContext_178)->atom_0+(_S1189 + 18U), 1U, memory_order_relaxed);
                *((&kernelContext_178)->work_0+(_S1188 + _S1190)) = s_17;
            }
        }
        side_6 = side_6 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_gate(uint3 id_56 [[thread_position_in_grid]], FractureParams_0 constant* params_61 [[buffer(0)]], uint device* work_61 [[buffer(4)]], packed_float4 device* frames_61 [[buffer(6)]], packed_float4 device* bodies_61 [[buffer(7)]], packed_float4 device* chunks_61 [[buffer(1)]], uint device* sort_61 [[buffer(11)]], atomic<uint> device* atom_61 [[buffer(5)]], Record_natural_0 device* manifolds_61 [[buffer(10)]], packed_float4 device* loads_61 [[buffer(12)]], packed_uint4 device* bonds_61 [[buffer(2)]], uint device* args_61 [[buffer(13)]], uint device* info_61 [[buffer(15)]], ScanParams_0 constant* scan_61 [[buffer(14)]], packed_float4 device* parts_61 [[buffer(3)]], packed_float4 device* colliders_61 [[buffer(8)]], packed_uint4 device* collider_info_61 [[buffer(9)]], packed_float4 device* view_chunks_61 [[buffer(17)]], ViewParams_0 constant* view_61 [[buffer(16)]], packed_float4 device* objects_61 [[buffer(18)]], uint device* shot_bodies_61 [[buffer(20)]], float device* lines_61 [[buffer(19)]], packed_float4 device* geometry_61 [[buffer(21)]], packed_float4 device* shots_61 [[buffer(22)]], packed_float4 device* lp_61 [[buffer(24)]], packed_float4 device* ext_63 [[buffer(23)]], packed_float4 device* engine_contacts_61 [[buffer(25)]], CrushParams_0 constant* crush_61 [[buffer(28)]], packed_float4 device* crush_rw_61 [[buffer(27)]], packed_float4 device* crush_tab_61 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_179;
    (&kernelContext_179)->params_0 = params_61;
    (&kernelContext_179)->work_0 = work_61;
    (&kernelContext_179)->frames_0 = frames_61;
    (&kernelContext_179)->bodies_0 = bodies_61;
    (&kernelContext_179)->chunks_0 = chunks_61;
    (&kernelContext_179)->sort_0 = sort_61;
    (&kernelContext_179)->atom_0 = atom_61;
    (&kernelContext_179)->manifolds_0 = manifolds_61;
    (&kernelContext_179)->loads_0 = loads_61;
    (&kernelContext_179)->bonds_0 = bonds_61;
    (&kernelContext_179)->args_0 = args_61;
    (&kernelContext_179)->info_0 = info_61;
    (&kernelContext_179)->scan_0 = scan_61;
    (&kernelContext_179)->parts_0 = parts_61;
    (&kernelContext_179)->colliders_0 = colliders_61;
    (&kernelContext_179)->collider_info_0 = collider_info_61;
    (&kernelContext_179)->view_chunks_0 = view_chunks_61;
    (&kernelContext_179)->view_0 = view_61;
    (&kernelContext_179)->objects_0 = objects_61;
    (&kernelContext_179)->shot_bodies_0 = shot_bodies_61;
    (&kernelContext_179)->lines_0 = lines_61;
    (&kernelContext_179)->geometry_0 = geometry_61;
    (&kernelContext_179)->shots_0 = shots_61;
    (&kernelContext_179)->lp_0 = lp_61;
    (&kernelContext_179)->ext_0 = ext_63;
    (&kernelContext_179)->engine_contacts_0 = engine_contacts_61;
    (&kernelContext_179)->crush_0 = crush_61;
    (&kernelContext_179)->crush_rw_0 = crush_rw_61;
    (&kernelContext_179)->crush_tab_0 = crush_tab_61;
    threadgroup array<uint, int(1024)> g_scan_61;
    (&kernelContext_179)->g_scan_0 = &g_scan_61;
    threadgroup array<float, int(64)> g_il_best_61;
    (&kernelContext_179)->g_il_best_0 = &g_il_best_61;
    uint _S1191 = a_counters_1(&kernelContext_179);
    uint struck_1 = atomic_load_explicit((&kernelContext_179)->atom_0+(_S1191 + 18U), memory_order_relaxed);
    bool any_0 = struck_1 > 0U;
    uint _S1192;
    if(any_0)
    {
        uint _S1193 = cs_rows_0(&kernelContext_179);
        uint _S1194 = groups_of_0(_S1193, 64U);
        _S1192 = _S1194;
    }
    else
    {
        _S1192 = 0U;
    }
    set_args_0(8U, _S1192, &kernelContext_179);
    uint _S1195 = groups_of_0(struck_1, 64U);
    set_args_0(9U, _S1195, &kernelContext_179);
    set_args_0(6U, struck_1, &kernelContext_179);
    if(any_0)
    {
        uint _S1196 = groups_of_0((&kernelContext_179)->params_0->chunk_count_0, 64U);
        _S1192 = _S1196;
    }
    else
    {
        _S1192 = 0U;
    }
    set_args_0(10U, _S1192, &kernelContext_179);
    if(any_0)
    {
        uint _S1197 = groups_of_0((&kernelContext_179)->params_0->bond_count_0, 64U);
        _S1192 = _S1197;
    }
    else
    {
        _S1192 = 0U;
    }
    set_args_0(11U, _S1192, &kernelContext_179);
    if(any_0)
    {
        uint _S1198 = groups_of_0((&kernelContext_179)->params_0->body_count_0, 64U);
        _S1192 = _S1198;
    }
    else
    {
        _S1192 = 0U;
    }
    set_args_0(12U, _S1192, &kernelContext_179);
    return;
}

[[kernel]] void fr_cs_pick_a(uint3 id_57 [[thread_position_in_grid]], FractureParams_0 constant* params_62 [[buffer(0)]], uint device* work_62 [[buffer(4)]], packed_float4 device* frames_62 [[buffer(6)]], packed_float4 device* bodies_62 [[buffer(7)]], packed_float4 device* chunks_62 [[buffer(1)]], uint device* sort_62 [[buffer(11)]], atomic<uint> device* atom_62 [[buffer(5)]], Record_natural_0 device* manifolds_62 [[buffer(10)]], packed_float4 device* loads_62 [[buffer(12)]], packed_uint4 device* bonds_62 [[buffer(2)]], uint device* args_62 [[buffer(13)]], uint device* info_62 [[buffer(15)]], ScanParams_0 constant* scan_62 [[buffer(14)]], packed_float4 device* parts_62 [[buffer(3)]], packed_float4 device* colliders_62 [[buffer(8)]], packed_uint4 device* collider_info_62 [[buffer(9)]], packed_float4 device* view_chunks_62 [[buffer(17)]], ViewParams_0 constant* view_62 [[buffer(16)]], packed_float4 device* objects_62 [[buffer(18)]], uint device* shot_bodies_62 [[buffer(20)]], float device* lines_62 [[buffer(19)]], packed_float4 device* geometry_62 [[buffer(21)]], packed_float4 device* shots_62 [[buffer(22)]], packed_float4 device* lp_62 [[buffer(24)]], packed_float4 device* ext_64 [[buffer(23)]], packed_float4 device* engine_contacts_62 [[buffer(25)]], CrushParams_0 constant* crush_62 [[buffer(28)]], packed_float4 device* crush_rw_62 [[buffer(27)]], packed_float4 device* crush_tab_62 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_180;
    (&kernelContext_180)->params_0 = params_62;
    (&kernelContext_180)->work_0 = work_62;
    (&kernelContext_180)->frames_0 = frames_62;
    (&kernelContext_180)->bodies_0 = bodies_62;
    (&kernelContext_180)->chunks_0 = chunks_62;
    (&kernelContext_180)->sort_0 = sort_62;
    (&kernelContext_180)->atom_0 = atom_62;
    (&kernelContext_180)->manifolds_0 = manifolds_62;
    (&kernelContext_180)->loads_0 = loads_62;
    (&kernelContext_180)->bonds_0 = bonds_62;
    (&kernelContext_180)->args_0 = args_62;
    (&kernelContext_180)->info_0 = info_62;
    (&kernelContext_180)->scan_0 = scan_62;
    (&kernelContext_180)->parts_0 = parts_62;
    (&kernelContext_180)->colliders_0 = colliders_62;
    (&kernelContext_180)->collider_info_0 = collider_info_62;
    (&kernelContext_180)->view_chunks_0 = view_chunks_62;
    (&kernelContext_180)->view_0 = view_62;
    (&kernelContext_180)->objects_0 = objects_62;
    (&kernelContext_180)->shot_bodies_0 = shot_bodies_62;
    (&kernelContext_180)->lines_0 = lines_62;
    (&kernelContext_180)->geometry_0 = geometry_62;
    (&kernelContext_180)->shots_0 = shots_62;
    (&kernelContext_180)->lp_0 = lp_62;
    (&kernelContext_180)->ext_0 = ext_64;
    (&kernelContext_180)->engine_contacts_0 = engine_contacts_62;
    (&kernelContext_180)->crush_0 = crush_62;
    (&kernelContext_180)->crush_rw_0 = crush_rw_62;
    (&kernelContext_180)->crush_tab_0 = crush_tab_62;
    threadgroup array<uint, int(1024)> g_scan_62;
    (&kernelContext_180)->g_scan_0 = &g_scan_62;
    threadgroup array<float, int(64)> g_il_best_62;
    (&kernelContext_180)->g_il_best_0 = &g_il_best_62;
    cs_pick_3(id_57.x, &kernelContext_180);
    return;
}

[[kernel]] void fr_cs_pick_b(uint3 id_58 [[thread_position_in_grid]], FractureParams_0 constant* params_63 [[buffer(0)]], uint device* work_63 [[buffer(4)]], packed_float4 device* frames_63 [[buffer(6)]], packed_float4 device* bodies_63 [[buffer(7)]], packed_float4 device* chunks_63 [[buffer(1)]], uint device* sort_63 [[buffer(11)]], atomic<uint> device* atom_63 [[buffer(5)]], Record_natural_0 device* manifolds_63 [[buffer(10)]], packed_float4 device* loads_63 [[buffer(12)]], packed_uint4 device* bonds_63 [[buffer(2)]], uint device* args_63 [[buffer(13)]], uint device* info_63 [[buffer(15)]], ScanParams_0 constant* scan_63 [[buffer(14)]], packed_float4 device* parts_63 [[buffer(3)]], packed_float4 device* colliders_63 [[buffer(8)]], packed_uint4 device* collider_info_63 [[buffer(9)]], packed_float4 device* view_chunks_63 [[buffer(17)]], ViewParams_0 constant* view_63 [[buffer(16)]], packed_float4 device* objects_63 [[buffer(18)]], uint device* shot_bodies_63 [[buffer(20)]], float device* lines_63 [[buffer(19)]], packed_float4 device* geometry_63 [[buffer(21)]], packed_float4 device* shots_63 [[buffer(22)]], packed_float4 device* lp_63 [[buffer(24)]], packed_float4 device* ext_65 [[buffer(23)]], packed_float4 device* engine_contacts_63 [[buffer(25)]], CrushParams_0 constant* crush_63 [[buffer(28)]], packed_float4 device* crush_rw_63 [[buffer(27)]], packed_float4 device* crush_tab_63 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_181;
    (&kernelContext_181)->params_0 = params_63;
    (&kernelContext_181)->work_0 = work_63;
    (&kernelContext_181)->frames_0 = frames_63;
    (&kernelContext_181)->bodies_0 = bodies_63;
    (&kernelContext_181)->chunks_0 = chunks_63;
    (&kernelContext_181)->sort_0 = sort_63;
    (&kernelContext_181)->atom_0 = atom_63;
    (&kernelContext_181)->manifolds_0 = manifolds_63;
    (&kernelContext_181)->loads_0 = loads_63;
    (&kernelContext_181)->bonds_0 = bonds_63;
    (&kernelContext_181)->args_0 = args_63;
    (&kernelContext_181)->info_0 = info_63;
    (&kernelContext_181)->scan_0 = scan_63;
    (&kernelContext_181)->parts_0 = parts_63;
    (&kernelContext_181)->colliders_0 = colliders_63;
    (&kernelContext_181)->collider_info_0 = collider_info_63;
    (&kernelContext_181)->view_chunks_0 = view_chunks_63;
    (&kernelContext_181)->view_0 = view_63;
    (&kernelContext_181)->objects_0 = objects_63;
    (&kernelContext_181)->shot_bodies_0 = shot_bodies_63;
    (&kernelContext_181)->lines_0 = lines_63;
    (&kernelContext_181)->geometry_0 = geometry_63;
    (&kernelContext_181)->shots_0 = shots_63;
    (&kernelContext_181)->lp_0 = lp_63;
    (&kernelContext_181)->ext_0 = ext_65;
    (&kernelContext_181)->engine_contacts_0 = engine_contacts_63;
    (&kernelContext_181)->crush_0 = crush_63;
    (&kernelContext_181)->crush_rw_0 = crush_rw_63;
    (&kernelContext_181)->crush_tab_0 = crush_tab_63;
    threadgroup array<uint, int(1024)> g_scan_63;
    (&kernelContext_181)->g_scan_0 = &g_scan_63;
    threadgroup array<float, int(64)> g_il_best_63;
    (&kernelContext_181)->g_il_best_0 = &g_il_best_63;
    cs_pick_2(id_58.x, &kernelContext_181);
    return;
}

[[kernel]] void fr_cs_pick_c(uint3 id_59 [[thread_position_in_grid]], FractureParams_0 constant* params_64 [[buffer(0)]], uint device* work_64 [[buffer(4)]], packed_float4 device* frames_64 [[buffer(6)]], packed_float4 device* bodies_64 [[buffer(7)]], packed_float4 device* chunks_64 [[buffer(1)]], uint device* sort_64 [[buffer(11)]], atomic<uint> device* atom_64 [[buffer(5)]], Record_natural_0 device* manifolds_64 [[buffer(10)]], packed_float4 device* loads_64 [[buffer(12)]], packed_uint4 device* bonds_64 [[buffer(2)]], uint device* args_64 [[buffer(13)]], uint device* info_64 [[buffer(15)]], ScanParams_0 constant* scan_64 [[buffer(14)]], packed_float4 device* parts_64 [[buffer(3)]], packed_float4 device* colliders_64 [[buffer(8)]], packed_uint4 device* collider_info_64 [[buffer(9)]], packed_float4 device* view_chunks_64 [[buffer(17)]], ViewParams_0 constant* view_64 [[buffer(16)]], packed_float4 device* objects_64 [[buffer(18)]], uint device* shot_bodies_64 [[buffer(20)]], float device* lines_64 [[buffer(19)]], packed_float4 device* geometry_64 [[buffer(21)]], packed_float4 device* shots_64 [[buffer(22)]], packed_float4 device* lp_64 [[buffer(24)]], packed_float4 device* ext_66 [[buffer(23)]], packed_float4 device* engine_contacts_64 [[buffer(25)]], CrushParams_0 constant* crush_64 [[buffer(28)]], packed_float4 device* crush_rw_64 [[buffer(27)]], packed_float4 device* crush_tab_64 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_182;
    (&kernelContext_182)->params_0 = params_64;
    (&kernelContext_182)->work_0 = work_64;
    (&kernelContext_182)->frames_0 = frames_64;
    (&kernelContext_182)->bodies_0 = bodies_64;
    (&kernelContext_182)->chunks_0 = chunks_64;
    (&kernelContext_182)->sort_0 = sort_64;
    (&kernelContext_182)->atom_0 = atom_64;
    (&kernelContext_182)->manifolds_0 = manifolds_64;
    (&kernelContext_182)->loads_0 = loads_64;
    (&kernelContext_182)->bonds_0 = bonds_64;
    (&kernelContext_182)->args_0 = args_64;
    (&kernelContext_182)->info_0 = info_64;
    (&kernelContext_182)->scan_0 = scan_64;
    (&kernelContext_182)->parts_0 = parts_64;
    (&kernelContext_182)->colliders_0 = colliders_64;
    (&kernelContext_182)->collider_info_0 = collider_info_64;
    (&kernelContext_182)->view_chunks_0 = view_chunks_64;
    (&kernelContext_182)->view_0 = view_64;
    (&kernelContext_182)->objects_0 = objects_64;
    (&kernelContext_182)->shot_bodies_0 = shot_bodies_64;
    (&kernelContext_182)->lines_0 = lines_64;
    (&kernelContext_182)->geometry_0 = geometry_64;
    (&kernelContext_182)->shots_0 = shots_64;
    (&kernelContext_182)->lp_0 = lp_64;
    (&kernelContext_182)->ext_0 = ext_66;
    (&kernelContext_182)->engine_contacts_0 = engine_contacts_64;
    (&kernelContext_182)->crush_0 = crush_64;
    (&kernelContext_182)->crush_rw_0 = crush_rw_64;
    (&kernelContext_182)->crush_tab_0 = crush_tab_64;
    threadgroup array<uint, int(1024)> g_scan_64;
    (&kernelContext_182)->g_scan_0 = &g_scan_64;
    threadgroup array<float, int(64)> g_il_best_64;
    (&kernelContext_182)->g_il_best_0 = &g_il_best_64;
    cs_pick_1(id_59.x, &kernelContext_182);
    return;
}

[[kernel]] void fr_cs_pick_d(uint3 id_60 [[thread_position_in_grid]], FractureParams_0 constant* params_65 [[buffer(0)]], uint device* work_65 [[buffer(4)]], packed_float4 device* frames_65 [[buffer(6)]], packed_float4 device* bodies_65 [[buffer(7)]], packed_float4 device* chunks_65 [[buffer(1)]], uint device* sort_65 [[buffer(11)]], atomic<uint> device* atom_65 [[buffer(5)]], Record_natural_0 device* manifolds_65 [[buffer(10)]], packed_float4 device* loads_65 [[buffer(12)]], packed_uint4 device* bonds_65 [[buffer(2)]], uint device* args_65 [[buffer(13)]], uint device* info_65 [[buffer(15)]], ScanParams_0 constant* scan_65 [[buffer(14)]], packed_float4 device* parts_65 [[buffer(3)]], packed_float4 device* colliders_65 [[buffer(8)]], packed_uint4 device* collider_info_65 [[buffer(9)]], packed_float4 device* view_chunks_65 [[buffer(17)]], ViewParams_0 constant* view_65 [[buffer(16)]], packed_float4 device* objects_65 [[buffer(18)]], uint device* shot_bodies_65 [[buffer(20)]], float device* lines_65 [[buffer(19)]], packed_float4 device* geometry_65 [[buffer(21)]], packed_float4 device* shots_65 [[buffer(22)]], packed_float4 device* lp_65 [[buffer(24)]], packed_float4 device* ext_67 [[buffer(23)]], packed_float4 device* engine_contacts_65 [[buffer(25)]], CrushParams_0 constant* crush_65 [[buffer(28)]], packed_float4 device* crush_rw_65 [[buffer(27)]], packed_float4 device* crush_tab_65 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_183;
    (&kernelContext_183)->params_0 = params_65;
    (&kernelContext_183)->work_0 = work_65;
    (&kernelContext_183)->frames_0 = frames_65;
    (&kernelContext_183)->bodies_0 = bodies_65;
    (&kernelContext_183)->chunks_0 = chunks_65;
    (&kernelContext_183)->sort_0 = sort_65;
    (&kernelContext_183)->atom_0 = atom_65;
    (&kernelContext_183)->manifolds_0 = manifolds_65;
    (&kernelContext_183)->loads_0 = loads_65;
    (&kernelContext_183)->bonds_0 = bonds_65;
    (&kernelContext_183)->args_0 = args_65;
    (&kernelContext_183)->info_0 = info_65;
    (&kernelContext_183)->scan_0 = scan_65;
    (&kernelContext_183)->parts_0 = parts_65;
    (&kernelContext_183)->colliders_0 = colliders_65;
    (&kernelContext_183)->collider_info_0 = collider_info_65;
    (&kernelContext_183)->view_chunks_0 = view_chunks_65;
    (&kernelContext_183)->view_0 = view_65;
    (&kernelContext_183)->objects_0 = objects_65;
    (&kernelContext_183)->shot_bodies_0 = shot_bodies_65;
    (&kernelContext_183)->lines_0 = lines_65;
    (&kernelContext_183)->geometry_0 = geometry_65;
    (&kernelContext_183)->shots_0 = shots_65;
    (&kernelContext_183)->lp_0 = lp_65;
    (&kernelContext_183)->ext_0 = ext_67;
    (&kernelContext_183)->engine_contacts_0 = engine_contacts_65;
    (&kernelContext_183)->crush_0 = crush_65;
    (&kernelContext_183)->crush_rw_0 = crush_rw_65;
    (&kernelContext_183)->crush_tab_0 = crush_tab_65;
    threadgroup array<uint, int(1024)> g_scan_65;
    (&kernelContext_183)->g_scan_0 = &g_scan_65;
    threadgroup array<float, int(64)> g_il_best_65;
    (&kernelContext_183)->g_il_best_0 = &g_il_best_65;
    cs_pick_0(id_60.x, &kernelContext_183);
    return;
}

[[kernel]] void fr_cs_record(uint3 id_61 [[thread_position_in_grid]], FractureParams_0 constant* params_66 [[buffer(0)]], uint device* work_66 [[buffer(4)]], packed_float4 device* frames_66 [[buffer(6)]], packed_float4 device* bodies_66 [[buffer(7)]], packed_float4 device* chunks_66 [[buffer(1)]], uint device* sort_66 [[buffer(11)]], atomic<uint> device* atom_66 [[buffer(5)]], Record_natural_0 device* manifolds_66 [[buffer(10)]], packed_float4 device* loads_66 [[buffer(12)]], packed_uint4 device* bonds_66 [[buffer(2)]], uint device* args_66 [[buffer(13)]], uint device* info_66 [[buffer(15)]], ScanParams_0 constant* scan_66 [[buffer(14)]], packed_float4 device* parts_66 [[buffer(3)]], packed_float4 device* colliders_66 [[buffer(8)]], packed_uint4 device* collider_info_66 [[buffer(9)]], packed_float4 device* view_chunks_66 [[buffer(17)]], ViewParams_0 constant* view_66 [[buffer(16)]], packed_float4 device* objects_66 [[buffer(18)]], uint device* shot_bodies_66 [[buffer(20)]], float device* lines_66 [[buffer(19)]], packed_float4 device* geometry_66 [[buffer(21)]], packed_float4 device* shots_66 [[buffer(22)]], packed_float4 device* lp_66 [[buffer(24)]], packed_float4 device* ext_68 [[buffer(23)]], packed_float4 device* engine_contacts_66 [[buffer(25)]], CrushParams_0 constant* crush_66 [[buffer(28)]], packed_float4 device* crush_rw_66 [[buffer(27)]], packed_float4 device* crush_tab_66 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_184;
    (&kernelContext_184)->params_0 = params_66;
    (&kernelContext_184)->work_0 = work_66;
    (&kernelContext_184)->frames_0 = frames_66;
    (&kernelContext_184)->bodies_0 = bodies_66;
    (&kernelContext_184)->chunks_0 = chunks_66;
    (&kernelContext_184)->sort_0 = sort_66;
    (&kernelContext_184)->atom_0 = atom_66;
    (&kernelContext_184)->manifolds_0 = manifolds_66;
    (&kernelContext_184)->loads_0 = loads_66;
    (&kernelContext_184)->bonds_0 = bonds_66;
    (&kernelContext_184)->args_0 = args_66;
    (&kernelContext_184)->info_0 = info_66;
    (&kernelContext_184)->scan_0 = scan_66;
    (&kernelContext_184)->parts_0 = parts_66;
    (&kernelContext_184)->colliders_0 = colliders_66;
    (&kernelContext_184)->collider_info_0 = collider_info_66;
    (&kernelContext_184)->view_chunks_0 = view_chunks_66;
    (&kernelContext_184)->view_0 = view_66;
    (&kernelContext_184)->objects_0 = objects_66;
    (&kernelContext_184)->shot_bodies_0 = shot_bodies_66;
    (&kernelContext_184)->lines_0 = lines_66;
    (&kernelContext_184)->geometry_0 = geometry_66;
    (&kernelContext_184)->shots_0 = shots_66;
    (&kernelContext_184)->lp_0 = lp_66;
    (&kernelContext_184)->ext_0 = ext_68;
    (&kernelContext_184)->engine_contacts_0 = engine_contacts_66;
    (&kernelContext_184)->crush_0 = crush_66;
    (&kernelContext_184)->crush_rw_0 = crush_rw_66;
    (&kernelContext_184)->crush_tab_0 = crush_tab_66;
    threadgroup array<uint, int(1024)> g_scan_66;
    (&kernelContext_184)->g_scan_0 = &g_scan_66;
    threadgroup array<float, int(64)> g_il_best_66;
    (&kernelContext_184)->g_il_best_0 = &g_il_best_66;
    uint _S1199 = id_61.x;
    uint _S1200 = a_counters_1(&kernelContext_184);
    uint _S1201 = atomic_load_explicit((&kernelContext_184)->atom_0+(_S1200 + 18U), memory_order_relaxed);
    if(_S1199 >= _S1201)
    {
        return;
    }
    uint _S1202 = w_cs_bodies_0(&kernelContext_184);
    uint device* _S1203 = (&kernelContext_184)->work_0+(_S1202 + _S1199);
    uint s_18 = *_S1203;
    uint _S1204 = a_cs_0(*_S1203, &kernelContext_184);
    uint _S1205 = atomic_load_explicit((&kernelContext_184)->atom_0+(_S1204 + 11U), memory_order_relaxed);
    uint key_2 = _S1205 - 1U;
    thread uint k_45;
    thread uint s2_0;
    thread uint ob_6;
    thread uint co_6;
    thread float3 p_19;
    thread float3 u_10;
    thread float v_15;
    thread float m_eff_8;
    thread float t_in_9;
    bool _S1206 = cs_side_0(key_2 / 2U, key_2 % 2U, &k_45, &s2_0, &ob_6, &co_6, &p_19, &u_10, &v_15, &m_eff_8, &t_in_9, &kernelContext_184);
    if(!_S1206)
    {
        return;
    }
    uint4 _S1207 = uint4(*((&kernelContext_184)->collider_info_0+(2U * co_6 + 1U))) ;
    float e_o_0 = (as_type<float>((_S1207.y)));
    float _S1208 = max((as_type<float>((_S1207.z))), 0.00100000004749745f);
    float j_9 = m_eff_8 * v_15;
    float tau_2 = hertz_duration_0(m_eff_8, _S1208, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / (float4(*((&kernelContext_184)->geometry_0+(3U * k_45 + 1U))) ).w), v_15);
    uint _S1209 = lp_body_0(s_18, &kernelContext_184);
    *((&kernelContext_184)->lp_0+(_S1209 + 2U)) = packed_float4(float4(from_rubble_0(p_19), 3.14159274101257324f * j_9 / (2.0f * tau_2))) ;
    *((&kernelContext_184)->lp_0+(_S1209 + 3U)) = packed_float4(float4(from_rubble_0(u_10), v_15)) ;
    *((&kernelContext_184)->lp_0+(_S1209 + 4U)) = packed_float4(float4(m_eff_8, _S1208, e_o_0, (as_type<float>((ob_6))))) ;
    *((&kernelContext_184)->lp_0+(_S1209 + 5U)) = packed_float4(float4((as_type<float>((k_45))), j_9 / tau_2, 1.0f / (float4(*((&kernelContext_184)->bodies_0+7U * ob_6)) ).w, t_in_9)) ;
    *((&kernelContext_184)->lp_0+(_S1209 + 6U)) = packed_float4(float4((as_type<float>((co_6))), (as_type<float>((_S1207.w))), 0.0f, 0.0f)) ;
    atomic_store_explicit((&kernelContext_184)->atom_0+(_S1204 + 1U), k_45 + 1U, memory_order_relaxed);
    uint _S1210 = atomic_fetch_add_explicit((&kernelContext_184)->atom_0+(_S1200 + 16U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_hertz(uint3 id_62 [[thread_position_in_grid]], FractureParams_0 constant* params_67 [[buffer(0)]], uint device* work_67 [[buffer(4)]], packed_float4 device* frames_67 [[buffer(6)]], packed_float4 device* bodies_67 [[buffer(7)]], packed_float4 device* chunks_67 [[buffer(1)]], uint device* sort_67 [[buffer(11)]], atomic<uint> device* atom_67 [[buffer(5)]], Record_natural_0 device* manifolds_67 [[buffer(10)]], packed_float4 device* loads_67 [[buffer(12)]], packed_uint4 device* bonds_67 [[buffer(2)]], uint device* args_67 [[buffer(13)]], uint device* info_67 [[buffer(15)]], ScanParams_0 constant* scan_67 [[buffer(14)]], packed_float4 device* parts_67 [[buffer(3)]], packed_float4 device* colliders_67 [[buffer(8)]], packed_uint4 device* collider_info_67 [[buffer(9)]], packed_float4 device* view_chunks_67 [[buffer(17)]], ViewParams_0 constant* view_67 [[buffer(16)]], packed_float4 device* objects_67 [[buffer(18)]], uint device* shot_bodies_67 [[buffer(20)]], float device* lines_67 [[buffer(19)]], packed_float4 device* geometry_67 [[buffer(21)]], packed_float4 device* shots_67 [[buffer(22)]], packed_float4 device* lp_67 [[buffer(24)]], packed_float4 device* ext_69 [[buffer(23)]], packed_float4 device* engine_contacts_67 [[buffer(25)]], CrushParams_0 constant* crush_67 [[buffer(28)]], packed_float4 device* crush_rw_67 [[buffer(27)]], packed_float4 device* crush_tab_67 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_185;
    (&kernelContext_185)->params_0 = params_67;
    (&kernelContext_185)->work_0 = work_67;
    (&kernelContext_185)->frames_0 = frames_67;
    (&kernelContext_185)->bodies_0 = bodies_67;
    (&kernelContext_185)->chunks_0 = chunks_67;
    (&kernelContext_185)->sort_0 = sort_67;
    (&kernelContext_185)->atom_0 = atom_67;
    (&kernelContext_185)->manifolds_0 = manifolds_67;
    (&kernelContext_185)->loads_0 = loads_67;
    (&kernelContext_185)->bonds_0 = bonds_67;
    (&kernelContext_185)->args_0 = args_67;
    (&kernelContext_185)->info_0 = info_67;
    (&kernelContext_185)->scan_0 = scan_67;
    (&kernelContext_185)->parts_0 = parts_67;
    (&kernelContext_185)->colliders_0 = colliders_67;
    (&kernelContext_185)->collider_info_0 = collider_info_67;
    (&kernelContext_185)->view_chunks_0 = view_chunks_67;
    (&kernelContext_185)->view_0 = view_67;
    (&kernelContext_185)->objects_0 = objects_67;
    (&kernelContext_185)->shot_bodies_0 = shot_bodies_67;
    (&kernelContext_185)->lines_0 = lines_67;
    (&kernelContext_185)->geometry_0 = geometry_67;
    (&kernelContext_185)->shots_0 = shots_67;
    (&kernelContext_185)->lp_0 = lp_67;
    (&kernelContext_185)->ext_0 = ext_69;
    (&kernelContext_185)->engine_contacts_0 = engine_contacts_67;
    (&kernelContext_185)->crush_0 = crush_67;
    (&kernelContext_185)->crush_rw_0 = crush_rw_67;
    (&kernelContext_185)->crush_tab_0 = crush_tab_67;
    threadgroup array<uint, int(1024)> g_scan_67;
    (&kernelContext_185)->g_scan_0 = &g_scan_67;
    threadgroup array<float, int(64)> g_il_best_67;
    (&kernelContext_185)->g_il_best_0 = &g_il_best_67;
    uint i_17 = id_62.x;
    bool _S1211;
    if(i_17 >= (params_67->bond_count_0))
    {
        _S1211 = true;
    }
    else
    {
        uint _S1212 = w_alive_0(&kernelContext_185);
        _S1211 = (*((&kernelContext_185)->work_0+(_S1212 + i_17))) == 0U;
    }
    if(_S1211)
    {
        return;
    }
    uint4 _S1213 = uint4(*((&kernelContext_185)->bonds_0+i_17)) ;
    uint _S1214 = body_of_0(_S1213.x, &kernelContext_185);
    bool _S1215 = cs_struck_1(_S1214, &kernelContext_185);
    if(!_S1215)
    {
        return;
    }
    uint _S1216 = lp_body_0(_S1214, &kernelContext_185);
    bool _S1217 = strike_share_breaks_0(_S1213, (float4(*((&kernelContext_185)->lp_0+(_S1216 + 2U))) ).xyz, (float4(*((&kernelContext_185)->lp_0+(_S1216 + 5U))) ).y, &kernelContext_185);
    if(_S1217)
    {
        uint _S1218 = w_alive_0(&kernelContext_185);
        *((&kernelContext_185)->work_0+(_S1218 + i_17)) = 0U;
        uint _S1219 = a_dirty_0(&kernelContext_185);
        uint _S1220 = atomic_fetch_or_explicit((&kernelContext_185)->atom_0+(_S1219 + _S1214), 1U, memory_order_relaxed);
        uint _S1221 = a_counters_1(&kernelContext_185);
        uint _S1222 = atomic_fetch_add_explicit((&kernelContext_185)->atom_0+_S1221, 1U, memory_order_relaxed);
        uint _S1223 = atomic_fetch_add_explicit((&kernelContext_185)->atom_0+(_S1221 + 1U), 1U, memory_order_relaxed);
        uint _S1224 = atomic_fetch_add_explicit((&kernelContext_185)->atom_0+(_S1221 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_gate2(uint3 id_63 [[thread_position_in_grid]], FractureParams_0 constant* params_68 [[buffer(0)]], uint device* work_68 [[buffer(4)]], packed_float4 device* frames_68 [[buffer(6)]], packed_float4 device* bodies_68 [[buffer(7)]], packed_float4 device* chunks_68 [[buffer(1)]], uint device* sort_68 [[buffer(11)]], atomic<uint> device* atom_68 [[buffer(5)]], Record_natural_0 device* manifolds_68 [[buffer(10)]], packed_float4 device* loads_68 [[buffer(12)]], packed_uint4 device* bonds_68 [[buffer(2)]], uint device* args_68 [[buffer(13)]], uint device* info_68 [[buffer(15)]], ScanParams_0 constant* scan_68 [[buffer(14)]], packed_float4 device* parts_68 [[buffer(3)]], packed_float4 device* colliders_68 [[buffer(8)]], packed_uint4 device* collider_info_68 [[buffer(9)]], packed_float4 device* view_chunks_68 [[buffer(17)]], ViewParams_0 constant* view_68 [[buffer(16)]], packed_float4 device* objects_68 [[buffer(18)]], uint device* shot_bodies_68 [[buffer(20)]], float device* lines_68 [[buffer(19)]], packed_float4 device* geometry_68 [[buffer(21)]], packed_float4 device* shots_68 [[buffer(22)]], packed_float4 device* lp_68 [[buffer(24)]], packed_float4 device* ext_70 [[buffer(23)]], packed_float4 device* engine_contacts_68 [[buffer(25)]], CrushParams_0 constant* crush_68 [[buffer(28)]], packed_float4 device* crush_rw_68 [[buffer(27)]], packed_float4 device* crush_tab_68 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_186;
    (&kernelContext_186)->params_0 = params_68;
    (&kernelContext_186)->work_0 = work_68;
    (&kernelContext_186)->frames_0 = frames_68;
    (&kernelContext_186)->bodies_0 = bodies_68;
    (&kernelContext_186)->chunks_0 = chunks_68;
    (&kernelContext_186)->sort_0 = sort_68;
    (&kernelContext_186)->atom_0 = atom_68;
    (&kernelContext_186)->manifolds_0 = manifolds_68;
    (&kernelContext_186)->loads_0 = loads_68;
    (&kernelContext_186)->bonds_0 = bonds_68;
    (&kernelContext_186)->args_0 = args_68;
    (&kernelContext_186)->info_0 = info_68;
    (&kernelContext_186)->scan_0 = scan_68;
    (&kernelContext_186)->parts_0 = parts_68;
    (&kernelContext_186)->colliders_0 = colliders_68;
    (&kernelContext_186)->collider_info_0 = collider_info_68;
    (&kernelContext_186)->view_chunks_0 = view_chunks_68;
    (&kernelContext_186)->view_0 = view_68;
    (&kernelContext_186)->objects_0 = objects_68;
    (&kernelContext_186)->shot_bodies_0 = shot_bodies_68;
    (&kernelContext_186)->lines_0 = lines_68;
    (&kernelContext_186)->geometry_0 = geometry_68;
    (&kernelContext_186)->shots_0 = shots_68;
    (&kernelContext_186)->lp_0 = lp_68;
    (&kernelContext_186)->ext_0 = ext_70;
    (&kernelContext_186)->engine_contacts_0 = engine_contacts_68;
    (&kernelContext_186)->crush_0 = crush_68;
    (&kernelContext_186)->crush_rw_0 = crush_rw_68;
    (&kernelContext_186)->crush_tab_0 = crush_tab_68;
    threadgroup array<uint, int(1024)> g_scan_68;
    (&kernelContext_186)->g_scan_0 = &g_scan_68;
    threadgroup array<float, int(64)> g_il_best_68;
    (&kernelContext_186)->g_il_best_0 = &g_il_best_68;
    uint _S1225 = a_counters_1(&kernelContext_186);
    uint _S1226 = atomic_load_explicit((&kernelContext_186)->atom_0+_S1225, memory_order_relaxed);
    uint _S1227;
    if(_S1226 > 0U)
    {
        uint _S1228 = cs_rows_0(&kernelContext_186);
        uint _S1229 = groups_of_0(_S1228, 64U);
        _S1227 = _S1229;
    }
    else
    {
        _S1227 = 0U;
    }
    set_args_0(13U, _S1227, &kernelContext_186);
    return;
}

uint lp_anchor_0(uint r_25, uint side_7, KernelContext_0 thread* kernelContext_187)
{
    return 22U * kernelContext_187->params_0->chunk_count_0 + 7U * kernelContext_187->params_0->body_count_0 + 2U * r_25 + side_7;
}

[[kernel]] void fr_cs_save(uint3 id_64 [[thread_position_in_grid]], FractureParams_0 constant* params_69 [[buffer(0)]], uint device* work_69 [[buffer(4)]], packed_float4 device* frames_69 [[buffer(6)]], packed_float4 device* bodies_69 [[buffer(7)]], packed_float4 device* chunks_69 [[buffer(1)]], uint device* sort_69 [[buffer(11)]], atomic<uint> device* atom_69 [[buffer(5)]], Record_natural_0 device* manifolds_69 [[buffer(10)]], packed_float4 device* loads_69 [[buffer(12)]], packed_uint4 device* bonds_69 [[buffer(2)]], uint device* args_69 [[buffer(13)]], uint device* info_69 [[buffer(15)]], ScanParams_0 constant* scan_69 [[buffer(14)]], packed_float4 device* parts_69 [[buffer(3)]], packed_float4 device* colliders_69 [[buffer(8)]], packed_uint4 device* collider_info_69 [[buffer(9)]], packed_float4 device* view_chunks_69 [[buffer(17)]], ViewParams_0 constant* view_69 [[buffer(16)]], packed_float4 device* objects_69 [[buffer(18)]], uint device* shot_bodies_69 [[buffer(20)]], float device* lines_69 [[buffer(19)]], packed_float4 device* geometry_69 [[buffer(21)]], packed_float4 device* shots_69 [[buffer(22)]], packed_float4 device* lp_69 [[buffer(24)]], packed_float4 device* ext_71 [[buffer(23)]], packed_float4 device* engine_contacts_69 [[buffer(25)]], CrushParams_0 constant* crush_69 [[buffer(28)]], packed_float4 device* crush_rw_69 [[buffer(27)]], packed_float4 device* crush_tab_69 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_188;
    (&kernelContext_188)->params_0 = params_69;
    (&kernelContext_188)->work_0 = work_69;
    (&kernelContext_188)->frames_0 = frames_69;
    (&kernelContext_188)->bodies_0 = bodies_69;
    (&kernelContext_188)->chunks_0 = chunks_69;
    (&kernelContext_188)->sort_0 = sort_69;
    (&kernelContext_188)->atom_0 = atom_69;
    (&kernelContext_188)->manifolds_0 = manifolds_69;
    (&kernelContext_188)->loads_0 = loads_69;
    (&kernelContext_188)->bonds_0 = bonds_69;
    (&kernelContext_188)->args_0 = args_69;
    (&kernelContext_188)->info_0 = info_69;
    (&kernelContext_188)->scan_0 = scan_69;
    (&kernelContext_188)->parts_0 = parts_69;
    (&kernelContext_188)->colliders_0 = colliders_69;
    (&kernelContext_188)->collider_info_0 = collider_info_69;
    (&kernelContext_188)->view_chunks_0 = view_chunks_69;
    (&kernelContext_188)->view_0 = view_69;
    (&kernelContext_188)->objects_0 = objects_69;
    (&kernelContext_188)->shot_bodies_0 = shot_bodies_69;
    (&kernelContext_188)->lines_0 = lines_69;
    (&kernelContext_188)->geometry_0 = geometry_69;
    (&kernelContext_188)->shots_0 = shots_69;
    (&kernelContext_188)->lp_0 = lp_69;
    (&kernelContext_188)->ext_0 = ext_71;
    (&kernelContext_188)->engine_contacts_0 = engine_contacts_69;
    (&kernelContext_188)->crush_0 = crush_69;
    (&kernelContext_188)->crush_rw_0 = crush_rw_69;
    (&kernelContext_188)->crush_tab_0 = crush_tab_69;
    threadgroup array<uint, int(1024)> g_scan_69;
    (&kernelContext_188)->g_scan_0 = &g_scan_69;
    threadgroup array<float, int(64)> g_il_best_69;
    (&kernelContext_188)->g_il_best_0 = &g_il_best_69;
    uint r_26 = id_64.x;
    uint _S1230 = cs_rows_0(&kernelContext_188);
    if(r_26 >= _S1230)
    {
        return;
    }
    uint _S1231 = cs_row_0(r_26, &kernelContext_188);
    uint4 _S1232 = (as_type<uint4>((float4(*((&kernelContext_188)->engine_contacts_0+_S1231)) )));
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
        float4 _S1233 = float4(0.0f) ;
        bool _S1234 = side_8 == 0U;
        uint col_1;
        if(_S1234)
        {
            col_1 = _S1232.x;
        }
        else
        {
            col_1 = _S1232.y;
        }
        uint k_46;
        if((_S1232.w) != 0U)
        {
            k_46 = (uint4(*((&kernelContext_188)->collider_info_0+2U * col_1)) ).x;
        }
        else
        {
            k_46 = 4294967295U;
        }
        bool _S1235;
        if(k_46 != 4294967295U)
        {
            _S1235 = k_46 < ((&kernelContext_188)->params_0->chunk_count_0);
        }
        else
        {
            _S1235 = false;
        }
        bool _S1236;
        if(_S1235)
        {
            uint _S1237 = body_of_0(k_46, &kernelContext_188);
            bool _S1238 = dirty_0(_S1237, &kernelContext_188);
            _S1236 = _S1238;
        }
        else
        {
            _S1236 = false;
        }
        float4 saved_0;
        if(_S1236)
        {
            uint _S1239 = body_of_0(k_46, &kernelContext_188);
            uint at_20 = 7U * _S1239;
            float3 anchor_0;
            if(_S1234)
            {
                anchor_0 = (float4(*((&kernelContext_188)->engine_contacts_0+(_S1231 + 2U))) ).xyz;
            }
            else
            {
                anchor_0 = (float4(*((&kernelContext_188)->engine_contacts_0+(_S1231 + 3U))) ).xyz;
            }
            float3 _S1240 = (float4(*((&kernelContext_188)->bodies_0+at_20)) ).xyz;
            thread Quat_0 _S1241 = quat_of_0(float4(*((&kernelContext_188)->bodies_0+(at_20 + 1U))) );
            float3 _S1242 = rotate_0(&_S1241, anchor_0);
            saved_0 = float4(_S1240 + _S1242, 1.0f);
        }
        else
        {
            saved_0 = _S1233;
        }
        uint _S1243 = lp_anchor_0(r_26, side_8, &kernelContext_188);
        *((&kernelContext_188)->lp_0+_S1243) = packed_float4(saved_0) ;
        side_8 = side_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_clean(uint3 id_65 [[thread_position_in_grid]], FractureParams_0 constant* params_70 [[buffer(0)]], uint device* work_70 [[buffer(4)]], packed_float4 device* frames_70 [[buffer(6)]], packed_float4 device* bodies_70 [[buffer(7)]], packed_float4 device* chunks_70 [[buffer(1)]], uint device* sort_70 [[buffer(11)]], atomic<uint> device* atom_70 [[buffer(5)]], Record_natural_0 device* manifolds_70 [[buffer(10)]], packed_float4 device* loads_70 [[buffer(12)]], packed_uint4 device* bonds_70 [[buffer(2)]], uint device* args_70 [[buffer(13)]], uint device* info_70 [[buffer(15)]], ScanParams_0 constant* scan_70 [[buffer(14)]], packed_float4 device* parts_70 [[buffer(3)]], packed_float4 device* colliders_70 [[buffer(8)]], packed_uint4 device* collider_info_70 [[buffer(9)]], packed_float4 device* view_chunks_70 [[buffer(17)]], ViewParams_0 constant* view_70 [[buffer(16)]], packed_float4 device* objects_70 [[buffer(18)]], uint device* shot_bodies_70 [[buffer(20)]], float device* lines_70 [[buffer(19)]], packed_float4 device* geometry_70 [[buffer(21)]], packed_float4 device* shots_70 [[buffer(22)]], packed_float4 device* lp_70 [[buffer(24)]], packed_float4 device* ext_72 [[buffer(23)]], packed_float4 device* engine_contacts_70 [[buffer(25)]], CrushParams_0 constant* crush_70 [[buffer(28)]], packed_float4 device* crush_rw_70 [[buffer(27)]], packed_float4 device* crush_tab_70 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_189;
    (&kernelContext_189)->params_0 = params_70;
    (&kernelContext_189)->work_0 = work_70;
    (&kernelContext_189)->frames_0 = frames_70;
    (&kernelContext_189)->bodies_0 = bodies_70;
    (&kernelContext_189)->chunks_0 = chunks_70;
    (&kernelContext_189)->sort_0 = sort_70;
    (&kernelContext_189)->atom_0 = atom_70;
    (&kernelContext_189)->manifolds_0 = manifolds_70;
    (&kernelContext_189)->loads_0 = loads_70;
    (&kernelContext_189)->bonds_0 = bonds_70;
    (&kernelContext_189)->args_0 = args_70;
    (&kernelContext_189)->info_0 = info_70;
    (&kernelContext_189)->scan_0 = scan_70;
    (&kernelContext_189)->parts_0 = parts_70;
    (&kernelContext_189)->colliders_0 = colliders_70;
    (&kernelContext_189)->collider_info_0 = collider_info_70;
    (&kernelContext_189)->view_chunks_0 = view_chunks_70;
    (&kernelContext_189)->view_0 = view_70;
    (&kernelContext_189)->objects_0 = objects_70;
    (&kernelContext_189)->shot_bodies_0 = shot_bodies_70;
    (&kernelContext_189)->lines_0 = lines_70;
    (&kernelContext_189)->geometry_0 = geometry_70;
    (&kernelContext_189)->shots_0 = shots_70;
    (&kernelContext_189)->lp_0 = lp_70;
    (&kernelContext_189)->ext_0 = ext_72;
    (&kernelContext_189)->engine_contacts_0 = engine_contacts_70;
    (&kernelContext_189)->crush_0 = crush_70;
    (&kernelContext_189)->crush_rw_0 = crush_rw_70;
    (&kernelContext_189)->crush_tab_0 = crush_tab_70;
    threadgroup array<uint, int(1024)> g_scan_70;
    (&kernelContext_189)->g_scan_0 = &g_scan_70;
    threadgroup array<float, int(64)> g_il_best_70;
    (&kernelContext_189)->g_il_best_0 = &g_il_best_70;
    uint _S1244 = id_65.x;
    uint _S1245 = a_counters_1(&kernelContext_189);
    uint _S1246 = atomic_load_explicit((&kernelContext_189)->atom_0+(_S1245 + 18U), memory_order_relaxed);
    if(_S1244 >= _S1246)
    {
        return;
    }
    uint _S1247 = w_cs_bodies_0(&kernelContext_189);
    uint _S1248 = a_cs_0(*((&kernelContext_189)->work_0+(_S1247 + _S1244)), &kernelContext_189);
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
        atomic_store_explicit((&kernelContext_189)->atom_0+(_S1248 + w_8), 0U, memory_order_relaxed);
        w_8 = w_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_fix(uint3 id_66 [[thread_position_in_grid]], FractureParams_0 constant* params_71 [[buffer(0)]], uint device* work_71 [[buffer(4)]], packed_float4 device* frames_71 [[buffer(6)]], packed_float4 device* bodies_71 [[buffer(7)]], packed_float4 device* chunks_71 [[buffer(1)]], uint device* sort_71 [[buffer(11)]], atomic<uint> device* atom_71 [[buffer(5)]], Record_natural_0 device* manifolds_71 [[buffer(10)]], packed_float4 device* loads_71 [[buffer(12)]], packed_uint4 device* bonds_71 [[buffer(2)]], uint device* args_71 [[buffer(13)]], uint device* info_71 [[buffer(15)]], ScanParams_0 constant* scan_71 [[buffer(14)]], packed_float4 device* parts_71 [[buffer(3)]], packed_float4 device* colliders_71 [[buffer(8)]], packed_uint4 device* collider_info_71 [[buffer(9)]], packed_float4 device* view_chunks_71 [[buffer(17)]], ViewParams_0 constant* view_71 [[buffer(16)]], packed_float4 device* objects_71 [[buffer(18)]], uint device* shot_bodies_71 [[buffer(20)]], float device* lines_71 [[buffer(19)]], packed_float4 device* geometry_71 [[buffer(21)]], packed_float4 device* shots_71 [[buffer(22)]], packed_float4 device* lp_71 [[buffer(24)]], packed_float4 device* ext_73 [[buffer(23)]], packed_float4 device* engine_contacts_71 [[buffer(25)]], CrushParams_0 constant* crush_71 [[buffer(28)]], packed_float4 device* crush_rw_71 [[buffer(27)]], packed_float4 device* crush_tab_71 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_190;
    (&kernelContext_190)->params_0 = params_71;
    (&kernelContext_190)->work_0 = work_71;
    (&kernelContext_190)->frames_0 = frames_71;
    (&kernelContext_190)->bodies_0 = bodies_71;
    (&kernelContext_190)->chunks_0 = chunks_71;
    (&kernelContext_190)->sort_0 = sort_71;
    (&kernelContext_190)->atom_0 = atom_71;
    (&kernelContext_190)->manifolds_0 = manifolds_71;
    (&kernelContext_190)->loads_0 = loads_71;
    (&kernelContext_190)->bonds_0 = bonds_71;
    (&kernelContext_190)->args_0 = args_71;
    (&kernelContext_190)->info_0 = info_71;
    (&kernelContext_190)->scan_0 = scan_71;
    (&kernelContext_190)->parts_0 = parts_71;
    (&kernelContext_190)->colliders_0 = colliders_71;
    (&kernelContext_190)->collider_info_0 = collider_info_71;
    (&kernelContext_190)->view_chunks_0 = view_chunks_71;
    (&kernelContext_190)->view_0 = view_71;
    (&kernelContext_190)->objects_0 = objects_71;
    (&kernelContext_190)->shot_bodies_0 = shot_bodies_71;
    (&kernelContext_190)->lines_0 = lines_71;
    (&kernelContext_190)->geometry_0 = geometry_71;
    (&kernelContext_190)->shots_0 = shots_71;
    (&kernelContext_190)->lp_0 = lp_71;
    (&kernelContext_190)->ext_0 = ext_73;
    (&kernelContext_190)->engine_contacts_0 = engine_contacts_71;
    (&kernelContext_190)->crush_0 = crush_71;
    (&kernelContext_190)->crush_rw_0 = crush_rw_71;
    (&kernelContext_190)->crush_tab_0 = crush_tab_71;
    threadgroup array<uint, int(1024)> g_scan_71;
    (&kernelContext_190)->g_scan_0 = &g_scan_71;
    threadgroup array<float, int(64)> g_il_best_71;
    (&kernelContext_190)->g_il_best_0 = &g_il_best_71;
    uint r_27 = id_66.x;
    uint _S1249 = cs_rows_0(&kernelContext_190);
    if(r_27 >= _S1249)
    {
        return;
    }
    uint _S1250 = cs_row_0(r_27, &kernelContext_190);
    uint4 _S1251 = (as_type<uint4>((float4(*((&kernelContext_190)->engine_contacts_0+_S1250)) )));
    bool moved_0 = false;
    uint side_9 = 0U;
    for(;;)
    {
        if(side_9 < 2U)
        {
        }
        else
        {
            break;
        }
        uint _S1252 = lp_anchor_0(r_27, side_9, &kernelContext_190);
        float4 _S1253 = float4(*((&kernelContext_190)->lp_0+_S1252)) ;
        if((_S1253.w) == 0.0f)
        {
            side_9 = side_9 + 1U;
            continue;
        }
        uint col_2;
        if(side_9 == 0U)
        {
            col_2 = _S1251.x;
        }
        else
        {
            col_2 = _S1251.y;
        }
        uint at_21 = 7U * (as_type<uint>(((float4(*((&kernelContext_190)->colliders_0+4U * col_2)) ).x)));
        thread Quat_0 _S1254 = quat_of_0(float4(*((&kernelContext_190)->bodies_0+(at_21 + 1U))) );
        Quat_0 _S1255 = conj_0(&_S1254);
        float3 _S1256 = _S1253.xyz - (float4(*((&kernelContext_190)->bodies_0+at_21)) ).xyz;
        thread Quat_0 _S1257 = _S1255;
        float3 _S1258 = rotate_0(&_S1257, _S1256);
        (*((&kernelContext_190)->engine_contacts_0+(_S1250 + 2U + side_9))).xyz = _S1258;
        moved_0 = true;
        side_9 = side_9 + 1U;
    }
    if(moved_0)
    {
        (*((&kernelContext_190)->engine_contacts_0+(_S1250 + 4U)))[int(3)] = 0.0f;
        (*((&kernelContext_190)->engine_contacts_0+(_S1250 + 5U)))[int(3)] = 0.0f;
        uint _S1259 = _S1250 + 6U;
        *((&kernelContext_190)->engine_contacts_0+_S1259) = packed_float4(float4(1.0f, 1.0f, 0.0f, (float4(*((&kernelContext_190)->engine_contacts_0+_S1259)) ).w)) ;
        (*((&kernelContext_190)->engine_contacts_0+(_S1250 + 7U)))[int(1)] = 0.0f;
    }
    return;
}

[[kernel]] void fr_cs_count(uint3 id_67 [[thread_position_in_grid]], FractureParams_0 constant* params_72 [[buffer(0)]], uint device* work_72 [[buffer(4)]], packed_float4 device* frames_72 [[buffer(6)]], packed_float4 device* bodies_72 [[buffer(7)]], packed_float4 device* chunks_72 [[buffer(1)]], uint device* sort_72 [[buffer(11)]], atomic<uint> device* atom_72 [[buffer(5)]], Record_natural_0 device* manifolds_72 [[buffer(10)]], packed_float4 device* loads_72 [[buffer(12)]], packed_uint4 device* bonds_72 [[buffer(2)]], uint device* args_72 [[buffer(13)]], uint device* info_72 [[buffer(15)]], ScanParams_0 constant* scan_72 [[buffer(14)]], packed_float4 device* parts_72 [[buffer(3)]], packed_float4 device* colliders_72 [[buffer(8)]], packed_uint4 device* collider_info_72 [[buffer(9)]], packed_float4 device* view_chunks_72 [[buffer(17)]], ViewParams_0 constant* view_72 [[buffer(16)]], packed_float4 device* objects_72 [[buffer(18)]], uint device* shot_bodies_72 [[buffer(20)]], float device* lines_72 [[buffer(19)]], packed_float4 device* geometry_72 [[buffer(21)]], packed_float4 device* shots_72 [[buffer(22)]], packed_float4 device* lp_72 [[buffer(24)]], packed_float4 device* ext_74 [[buffer(23)]], packed_float4 device* engine_contacts_72 [[buffer(25)]], CrushParams_0 constant* crush_72 [[buffer(28)]], packed_float4 device* crush_rw_72 [[buffer(27)]], packed_float4 device* crush_tab_72 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_191;
    (&kernelContext_191)->params_0 = params_72;
    (&kernelContext_191)->work_0 = work_72;
    (&kernelContext_191)->frames_0 = frames_72;
    (&kernelContext_191)->bodies_0 = bodies_72;
    (&kernelContext_191)->chunks_0 = chunks_72;
    (&kernelContext_191)->sort_0 = sort_72;
    (&kernelContext_191)->atom_0 = atom_72;
    (&kernelContext_191)->manifolds_0 = manifolds_72;
    (&kernelContext_191)->loads_0 = loads_72;
    (&kernelContext_191)->bonds_0 = bonds_72;
    (&kernelContext_191)->args_0 = args_72;
    (&kernelContext_191)->info_0 = info_72;
    (&kernelContext_191)->scan_0 = scan_72;
    (&kernelContext_191)->parts_0 = parts_72;
    (&kernelContext_191)->colliders_0 = colliders_72;
    (&kernelContext_191)->collider_info_0 = collider_info_72;
    (&kernelContext_191)->view_chunks_0 = view_chunks_72;
    (&kernelContext_191)->view_0 = view_72;
    (&kernelContext_191)->objects_0 = objects_72;
    (&kernelContext_191)->shot_bodies_0 = shot_bodies_72;
    (&kernelContext_191)->lines_0 = lines_72;
    (&kernelContext_191)->geometry_0 = geometry_72;
    (&kernelContext_191)->shots_0 = shots_72;
    (&kernelContext_191)->lp_0 = lp_72;
    (&kernelContext_191)->ext_0 = ext_74;
    (&kernelContext_191)->engine_contacts_0 = engine_contacts_72;
    (&kernelContext_191)->crush_0 = crush_72;
    (&kernelContext_191)->crush_rw_0 = crush_rw_72;
    (&kernelContext_191)->crush_tab_0 = crush_tab_72;
    threadgroup array<uint, int(1024)> g_scan_72;
    (&kernelContext_191)->g_scan_0 = &g_scan_72;
    threadgroup array<float, int(64)> g_il_best_72;
    (&kernelContext_191)->g_il_best_0 = &g_il_best_72;
    uint k_47 = id_67.x;
    if(k_47 >= (params_72->chunk_count_0))
    {
        return;
    }
    uint _S1260 = body_of_0(k_47, &kernelContext_191);
    bool _S1261 = cs_struck_1(_S1260, &kernelContext_191);
    if(!_S1261)
    {
        return;
    }
    uint _S1262 = a_cs_0(_S1260, &kernelContext_191);
    uint _S1263 = atomic_fetch_max_explicit((&kernelContext_191)->atom_0+(_S1262 + 3U), ~k_47, memory_order_relaxed);
    if(((&kernelContext_191)->info_0[k_47] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_191)->atom_0+(_S1262 + 4U), 1U, memory_order_relaxed);
    }
    uint _S1264 = atomic_fetch_add_explicit((&kernelContext_191)->atom_0+(_S1262 + 5U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_base(uint3 id_68 [[thread_position_in_grid]], FractureParams_0 constant* params_73 [[buffer(0)]], uint device* work_73 [[buffer(4)]], packed_float4 device* frames_73 [[buffer(6)]], packed_float4 device* bodies_73 [[buffer(7)]], packed_float4 device* chunks_73 [[buffer(1)]], uint device* sort_73 [[buffer(11)]], atomic<uint> device* atom_73 [[buffer(5)]], Record_natural_0 device* manifolds_73 [[buffer(10)]], packed_float4 device* loads_73 [[buffer(12)]], packed_uint4 device* bonds_73 [[buffer(2)]], uint device* args_73 [[buffer(13)]], uint device* info_73 [[buffer(15)]], ScanParams_0 constant* scan_73 [[buffer(14)]], packed_float4 device* parts_73 [[buffer(3)]], packed_float4 device* colliders_73 [[buffer(8)]], packed_uint4 device* collider_info_73 [[buffer(9)]], packed_float4 device* view_chunks_73 [[buffer(17)]], ViewParams_0 constant* view_73 [[buffer(16)]], packed_float4 device* objects_73 [[buffer(18)]], uint device* shot_bodies_73 [[buffer(20)]], float device* lines_73 [[buffer(19)]], packed_float4 device* geometry_73 [[buffer(21)]], packed_float4 device* shots_73 [[buffer(22)]], packed_float4 device* lp_73 [[buffer(24)]], packed_float4 device* ext_75 [[buffer(23)]], packed_float4 device* engine_contacts_73 [[buffer(25)]], CrushParams_0 constant* crush_73 [[buffer(28)]], packed_float4 device* crush_rw_73 [[buffer(27)]], packed_float4 device* crush_tab_73 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_192;
    (&kernelContext_192)->params_0 = params_73;
    (&kernelContext_192)->work_0 = work_73;
    (&kernelContext_192)->frames_0 = frames_73;
    (&kernelContext_192)->bodies_0 = bodies_73;
    (&kernelContext_192)->chunks_0 = chunks_73;
    (&kernelContext_192)->sort_0 = sort_73;
    (&kernelContext_192)->atom_0 = atom_73;
    (&kernelContext_192)->manifolds_0 = manifolds_73;
    (&kernelContext_192)->loads_0 = loads_73;
    (&kernelContext_192)->bonds_0 = bonds_73;
    (&kernelContext_192)->args_0 = args_73;
    (&kernelContext_192)->info_0 = info_73;
    (&kernelContext_192)->scan_0 = scan_73;
    (&kernelContext_192)->parts_0 = parts_73;
    (&kernelContext_192)->colliders_0 = colliders_73;
    (&kernelContext_192)->collider_info_0 = collider_info_73;
    (&kernelContext_192)->view_chunks_0 = view_chunks_73;
    (&kernelContext_192)->view_0 = view_73;
    (&kernelContext_192)->objects_0 = objects_73;
    (&kernelContext_192)->shot_bodies_0 = shot_bodies_73;
    (&kernelContext_192)->lines_0 = lines_73;
    (&kernelContext_192)->geometry_0 = geometry_73;
    (&kernelContext_192)->shots_0 = shots_73;
    (&kernelContext_192)->lp_0 = lp_73;
    (&kernelContext_192)->ext_0 = ext_75;
    (&kernelContext_192)->engine_contacts_0 = engine_contacts_73;
    (&kernelContext_192)->crush_0 = crush_73;
    (&kernelContext_192)->crush_rw_0 = crush_rw_73;
    (&kernelContext_192)->crush_tab_0 = crush_tab_73;
    threadgroup array<uint, int(1024)> g_scan_73;
    (&kernelContext_192)->g_scan_0 = &g_scan_73;
    threadgroup array<float, int(64)> g_il_best_73;
    (&kernelContext_192)->g_il_best_0 = &g_il_best_73;
    uint _S1265 = id_68.x;
    uint _S1266 = a_counters_1(&kernelContext_192);
    uint _S1267 = atomic_load_explicit((&kernelContext_192)->atom_0+(_S1266 + 18U), memory_order_relaxed);
    if(_S1265 >= _S1267)
    {
        return;
    }
    uint _S1268 = w_cs_bodies_0(&kernelContext_192);
    uint _S1269 = a_cs_0(*((&kernelContext_192)->work_0+(_S1268 + _S1265)), &kernelContext_192);
    atomic<uint> device* _S1270 = (&kernelContext_192)->atom_0+(_S1269 + 6U);
    atomic<uint> device* _S1271 = (&kernelContext_192)->atom_0+(_S1266 + 19U);
    uint _S1272 = atomic_load_explicit((&kernelContext_192)->atom_0+(_S1269 + 5U), memory_order_relaxed);
    uint _S1273 = atomic_fetch_add_explicit(_S1271, _S1272, memory_order_relaxed);
    atomic_store_explicit(_S1270, _S1273, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_collect(uint3 id_69 [[thread_position_in_grid]], FractureParams_0 constant* params_74 [[buffer(0)]], uint device* work_74 [[buffer(4)]], packed_float4 device* frames_74 [[buffer(6)]], packed_float4 device* bodies_74 [[buffer(7)]], packed_float4 device* chunks_74 [[buffer(1)]], uint device* sort_74 [[buffer(11)]], atomic<uint> device* atom_74 [[buffer(5)]], Record_natural_0 device* manifolds_74 [[buffer(10)]], packed_float4 device* loads_74 [[buffer(12)]], packed_uint4 device* bonds_74 [[buffer(2)]], uint device* args_74 [[buffer(13)]], uint device* info_74 [[buffer(15)]], ScanParams_0 constant* scan_74 [[buffer(14)]], packed_float4 device* parts_74 [[buffer(3)]], packed_float4 device* colliders_74 [[buffer(8)]], packed_uint4 device* collider_info_74 [[buffer(9)]], packed_float4 device* view_chunks_74 [[buffer(17)]], ViewParams_0 constant* view_74 [[buffer(16)]], packed_float4 device* objects_74 [[buffer(18)]], uint device* shot_bodies_74 [[buffer(20)]], float device* lines_74 [[buffer(19)]], packed_float4 device* geometry_74 [[buffer(21)]], packed_float4 device* shots_74 [[buffer(22)]], packed_float4 device* lp_74 [[buffer(24)]], packed_float4 device* ext_76 [[buffer(23)]], packed_float4 device* engine_contacts_74 [[buffer(25)]], CrushParams_0 constant* crush_74 [[buffer(28)]], packed_float4 device* crush_rw_74 [[buffer(27)]], packed_float4 device* crush_tab_74 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_193;
    (&kernelContext_193)->params_0 = params_74;
    (&kernelContext_193)->work_0 = work_74;
    (&kernelContext_193)->frames_0 = frames_74;
    (&kernelContext_193)->bodies_0 = bodies_74;
    (&kernelContext_193)->chunks_0 = chunks_74;
    (&kernelContext_193)->sort_0 = sort_74;
    (&kernelContext_193)->atom_0 = atom_74;
    (&kernelContext_193)->manifolds_0 = manifolds_74;
    (&kernelContext_193)->loads_0 = loads_74;
    (&kernelContext_193)->bonds_0 = bonds_74;
    (&kernelContext_193)->args_0 = args_74;
    (&kernelContext_193)->info_0 = info_74;
    (&kernelContext_193)->scan_0 = scan_74;
    (&kernelContext_193)->parts_0 = parts_74;
    (&kernelContext_193)->colliders_0 = colliders_74;
    (&kernelContext_193)->collider_info_0 = collider_info_74;
    (&kernelContext_193)->view_chunks_0 = view_chunks_74;
    (&kernelContext_193)->view_0 = view_74;
    (&kernelContext_193)->objects_0 = objects_74;
    (&kernelContext_193)->shot_bodies_0 = shot_bodies_74;
    (&kernelContext_193)->lines_0 = lines_74;
    (&kernelContext_193)->geometry_0 = geometry_74;
    (&kernelContext_193)->shots_0 = shots_74;
    (&kernelContext_193)->lp_0 = lp_74;
    (&kernelContext_193)->ext_0 = ext_76;
    (&kernelContext_193)->engine_contacts_0 = engine_contacts_74;
    (&kernelContext_193)->crush_0 = crush_74;
    (&kernelContext_193)->crush_rw_0 = crush_rw_74;
    (&kernelContext_193)->crush_tab_0 = crush_tab_74;
    threadgroup array<uint, int(1024)> g_scan_74;
    (&kernelContext_193)->g_scan_0 = &g_scan_74;
    threadgroup array<float, int(64)> g_il_best_74;
    (&kernelContext_193)->g_il_best_0 = &g_il_best_74;
    uint k_48 = id_69.x;
    if(k_48 >= (params_74->chunk_count_0))
    {
        return;
    }
    uint _S1274 = body_of_0(k_48, &kernelContext_193);
    bool _S1275 = cs_struck_1(_S1274, &kernelContext_193);
    if(!_S1275)
    {
        return;
    }
    uint _S1276 = w_cs_list_0(&kernelContext_193);
    uint _S1277 = a_cs_0(_S1274, &kernelContext_193);
    uint _S1278 = atomic_load_explicit((&kernelContext_193)->atom_0+(_S1277 + 6U), memory_order_relaxed);
    uint _S1279 = _S1276 + _S1278;
    uint _S1280 = atomic_fetch_add_explicit((&kernelContext_193)->atom_0+(_S1277 + 7U), 1U, memory_order_relaxed);
    *((&kernelContext_193)->work_0+(_S1279 + _S1280)) = k_48;
    return;
}

void cs_load_0(uint k_49, uint s_19, KernelContext_0 thread* kernelContext_194)
{
    uint _S1281 = a_cs_0(s_19, kernelContext_194);
    uint _S1282 = atomic_load_explicit(kernelContext_194->atom_0+(_S1281 + 4U), memory_order_relaxed);
    bool supported_1 = _S1282 != 0U;
    bool fixed_2;
    if((kernelContext_194->info_0[k_49] & 1U) != 0U)
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            uint _S1283 = atomic_load_explicit(kernelContext_194->atom_0+(_S1281 + 3U), memory_order_relaxed);
            fixed_2 = (~_S1283) == k_49;
        }
        else
        {
            fixed_2 = false;
        }
    }
    uint _S1284 = lp_body_0(s_19, kernelContext_194);
    float4 _S1285 = float4(*(kernelContext_194->lp_0+(_S1284 + 2U))) ;
    float4 _S1286 = float4(*(kernelContext_194->lp_0+(_S1284 + 3U))) ;
    uint _S1287 = 7U * s_19;
    thread Quat_0 q_27;
    thread float3 origin_8;
    structure_pose_0(s_19, _S1287, &q_27, &origin_8, kernelContext_194);
    thread Quat_0 _S1288 = q_27;
    Quat_0 _S1289 = conj_0(&_S1288);
    float3 _S1290 = _S1286.xyz * float3(_S1285.w) ;
    thread Quat_0 _S1291 = _S1289;
    float3 _S1292 = rotate_0(&_S1291, _S1290);
    float3 _S1293 = _S1285.xyz - origin_8;
    thread Quat_0 _S1294 = _S1289;
    float3 _S1295 = rotate_0(&_S1294, _S1293);
    uint _S1296 = 4U * k_49;
    float4 _S1297 = float4(*(kernelContext_194->chunks_0+_S1296)) ;
    float3 _S1298 = float3(0.0f) ;
    uint _S1299 = atomic_load_explicit(kernelContext_194->atom_0+(_S1281 + 1U), memory_order_relaxed);
    float3 f_12;
    float3 t_13;
    if(_S1299 == (k_49 + 1U))
    {
        float3 _S1300 = cross(_S1295 - _S1297.xyz, _S1292);
        f_12 = _S1292;
        t_13 = _S1300;
    }
    else
    {
        f_12 = _S1298;
        t_13 = _S1298;
    }
    float inv_mass_7 = (float4(*(kernelContext_194->bodies_0+_S1287)) ).w;
    bool _S1301;
    if(!supported_1)
    {
        _S1301 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S1301 = false;
    }
    if(_S1301)
    {
        uint _S1302 = 2U * s_19;
        float3 com_2 = (float4(*(kernelContext_194->frames_0+_S1302)) ).xyz;
        Quat_0 p_20 = quat_of_0(float4(*(kernelContext_194->frames_0+(_S1302 + 1U))) );
        float3 _S1303 = (float4(*(kernelContext_194->bodies_0+(_S1287 + 2U))) ).xyz;
        thread Quat_0 _S1304 = p_20;
        Quat_0 _S1305 = conj_0(&_S1304);
        float3 _S1306 = cross(_S1295 - com_2, _S1292);
        thread Quat_0 _S1307 = _S1305;
        float3 _S1308 = rotate_0(&_S1307, _S1306);
        float3 _S1309 = _S1303 * _S1308;
        thread Quat_0 _S1310 = p_20;
        float3 _S1311 = rotate_0(&_S1310, _S1309);
        float4 _S1312 = float4(*(kernelContext_194->chunks_0+(_S1296 + 1U))) ;
        float4 _S1313 = float4(*(kernelContext_194->chunks_0+(_S1296 + 2U))) ;
        float _S1314 = _S1311.x;
        float _S1315 = _S1312.w;
        float _S1316 = _S1311.y;
        float _S1317 = _S1313.x;
        float _S1318 = _S1311.z;
        float _S1319 = _S1313.y;
        float3 t_14 = t_13 - float3(_S1312.x * _S1314 + _S1315 * _S1316 + _S1317 * _S1318, _S1315 * _S1314 + _S1312.y * _S1316 + _S1319 * _S1318, _S1317 * _S1314 + _S1319 * _S1316 + _S1312.z * _S1318);
        f_12 = f_12 - float3(_S1297.w)  * (_S1292 * float3(inv_mass_7)  + cross(_S1311, _S1297.xyz - com_2));
        t_13 = t_14;
    }
    uint _S1320 = st_f_0(k_49, 14U);
    packed_float4 device* _S1321 = kernelContext_194->lp_0+_S1320;
    float _S1322;
    if(fixed_2)
    {
        _S1322 = 1.0f;
    }
    else
    {
        _S1322 = 0.0f;
    }
    *_S1321 = packed_float4(float4(f_12, _S1322)) ;
    *(kernelContext_194->lp_0+(_S1320 + 1U)) = packed_float4(float4(t_13, 0.0f)) ;
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
        uint _S1323 = st_x_0(k_49, 14U, c_23);
        packed_float4 _S1324 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_194->lp_0+_S1323) = _S1324;
        *(kernelContext_194->lp_0+(_S1323 + 1U)) = _S1324;
        c_23 = c_23 + 1U;
    }
    uint _S1325 = st_d_0(k_49, 14U);
    packed_float4 _S1326 = packed_float4(float4(0.0f) ) ;
    *(kernelContext_194->lp_0+_S1325) = _S1326;
    *(kernelContext_194->lp_0+(_S1325 + 1U)) = _S1326;
    return;
}

void cs_break_0(uint i_18, uint s_20, KernelContext_0 thread* kernelContext_195)
{
    uint4 _S1327 = uint4(*(kernelContext_195->bonds_0+i_18)) ;
    LpBond_0 _S1328 = lp_bond_0(i_18, kernelContext_195);
    uint _S1329 = _S1327.x;
    uint _S1330 = _S1327.y;
    thread LpBond_0 _S1331 = _S1328;
    thread float3 q_lin_5;
    thread float3 q_ang_5;
    st_force_0(&_S1331, _S1329, _S1330, 14U, 0U, &q_lin_5, &q_ang_5, kernelContext_195);
    if((kernelContext_195->params_0->load_model_0) == 1U)
    {
        thread LpBond_0 _S1332 = _S1328;
        thread float3 b_lin_0;
        thread float3 b_ang_0;
        lp_force_0(&_S1332, _S1329, _S1330, 0U, &b_lin_0, &b_ang_0, kernelContext_195);
        q_lin_5 = q_lin_5 + b_lin_0;
        q_ang_5 = q_ang_5 + b_ang_0;
    }
    thread JointBond_0 _S1333 = _S1328.law_0;
    Measures_0 _S1334 = stress_measures_0(&_S1333, q_lin_5, q_ang_5);
    JointMaterial_0 _S1335 = lp_material_0(_S1328.law_0.ids_3.x, kernelContext_195);
    float _S1336 = _S1328.law_0.geom1_0.w;
    thread JointMaterial_0 _S1337 = _S1335;
    thread JointBond_0 _S1338 = _S1328.law_0;
    thread Measures_0 _S1339 = _S1334;
    float4 _S1340 = failure_indices_0(&_S1337, &_S1338, &_S1339, _S1336);
    if((max(max(_S1340.x, _S1340.y), max(_S1340.z, _S1340.w))) >= 1.0f)
    {
        uint _S1341 = w_alive_0(kernelContext_195);
        *(kernelContext_195->work_0+(_S1341 + i_18)) = 0U;
        uint _S1342 = a_dirty_0(kernelContext_195);
        uint _S1343 = atomic_fetch_or_explicit(kernelContext_195->atom_0+(_S1342 + s_20), 1U, memory_order_relaxed);
        uint _S1344 = a_counters_1(kernelContext_195);
        uint _S1345 = atomic_fetch_add_explicit(kernelContext_195->atom_0+_S1344, 1U, memory_order_relaxed);
        uint _S1346 = atomic_fetch_add_explicit(kernelContext_195->atom_0+(_S1344 + 1U), 1U, memory_order_relaxed);
        uint _S1347 = atomic_fetch_add_explicit(kernelContext_195->atom_0+(_S1344 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_solve(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_75 [[buffer(0)]], uint device* work_75 [[buffer(4)]], packed_float4 device* frames_75 [[buffer(6)]], packed_float4 device* bodies_75 [[buffer(7)]], packed_float4 device* chunks_75 [[buffer(1)]], uint device* sort_75 [[buffer(11)]], atomic<uint> device* atom_75 [[buffer(5)]], Record_natural_0 device* manifolds_75 [[buffer(10)]], packed_float4 device* loads_75 [[buffer(12)]], packed_uint4 device* bonds_75 [[buffer(2)]], uint device* args_75 [[buffer(13)]], uint device* info_75 [[buffer(15)]], ScanParams_0 constant* scan_75 [[buffer(14)]], packed_float4 device* parts_75 [[buffer(3)]], packed_float4 device* colliders_75 [[buffer(8)]], packed_uint4 device* collider_info_75 [[buffer(9)]], packed_float4 device* view_chunks_75 [[buffer(17)]], ViewParams_0 constant* view_75 [[buffer(16)]], packed_float4 device* objects_75 [[buffer(18)]], uint device* shot_bodies_75 [[buffer(20)]], float device* lines_75 [[buffer(19)]], packed_float4 device* geometry_75 [[buffer(21)]], packed_float4 device* shots_75 [[buffer(22)]], packed_float4 device* lp_75 [[buffer(24)]], packed_float4 device* ext_77 [[buffer(23)]], packed_float4 device* engine_contacts_75 [[buffer(25)]], CrushParams_0 constant* crush_75 [[buffer(28)]], packed_float4 device* crush_rw_75 [[buffer(27)]], packed_float4 device* crush_tab_75 [[buffer(26)]])
{
    bool _S1348;
    uint j3_0;
    thread KernelContext_0 kernelContext_196;
    (&kernelContext_196)->params_0 = params_75;
    (&kernelContext_196)->work_0 = work_75;
    (&kernelContext_196)->frames_0 = frames_75;
    (&kernelContext_196)->bodies_0 = bodies_75;
    (&kernelContext_196)->chunks_0 = chunks_75;
    (&kernelContext_196)->sort_0 = sort_75;
    (&kernelContext_196)->atom_0 = atom_75;
    (&kernelContext_196)->manifolds_0 = manifolds_75;
    (&kernelContext_196)->loads_0 = loads_75;
    (&kernelContext_196)->bonds_0 = bonds_75;
    (&kernelContext_196)->args_0 = args_75;
    (&kernelContext_196)->info_0 = info_75;
    (&kernelContext_196)->scan_0 = scan_75;
    (&kernelContext_196)->parts_0 = parts_75;
    (&kernelContext_196)->colliders_0 = colliders_75;
    (&kernelContext_196)->collider_info_0 = collider_info_75;
    (&kernelContext_196)->view_chunks_0 = view_chunks_75;
    (&kernelContext_196)->view_0 = view_75;
    (&kernelContext_196)->objects_0 = objects_75;
    (&kernelContext_196)->shot_bodies_0 = shot_bodies_75;
    (&kernelContext_196)->lines_0 = lines_75;
    (&kernelContext_196)->geometry_0 = geometry_75;
    (&kernelContext_196)->shots_0 = shots_75;
    (&kernelContext_196)->lp_0 = lp_75;
    (&kernelContext_196)->ext_0 = ext_77;
    (&kernelContext_196)->engine_contacts_0 = engine_contacts_75;
    (&kernelContext_196)->crush_0 = crush_75;
    (&kernelContext_196)->crush_rw_0 = crush_rw_75;
    (&kernelContext_196)->crush_tab_0 = crush_tab_75;
    threadgroup array<uint, int(1024)> g_scan_75;
    (&kernelContext_196)->g_scan_0 = &g_scan_75;
    threadgroup array<float, int(64)> g_il_best_75;
    (&kernelContext_196)->g_il_best_0 = &g_il_best_75;
    uint _S1349 = w_cs_bodies_0(&kernelContext_196);
    uint device* _S1350 = (&kernelContext_196)->work_0+(_S1349 + group_3.x);
    uint s_21 = *_S1350;
    uint t_15 = thread_6.x;
    uint _S1351 = a_cs_0(*_S1350, &kernelContext_196);
    uint _S1352 = atomic_load_explicit((&kernelContext_196)->atom_0+(_S1351 + 6U), memory_order_relaxed);
    uint _S1353 = atomic_load_explicit((&kernelContext_196)->atom_0+(_S1351 + 5U), memory_order_relaxed);
    uint j_10 = t_15;
    for(;;)
    {
        if(j_10 < _S1353)
        {
        }
        else
        {
            break;
        }
        uint _S1354 = w_cs_list_0(&kernelContext_196);
        cs_load_0(*((&kernelContext_196)->work_0+(_S1354 + _S1352 + j_10)), s_21, &kernelContext_196);
        j_10 = j_10 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint j2_0 = t_15;
    for(;;)
    {
        if(j2_0 < _S1353)
        {
        }
        else
        {
            break;
        }
        uint _S1355 = w_cs_list_0(&kernelContext_196);
        uint device* _S1356 = (&kernelContext_196)->work_0+(_S1355 + _S1352 + j2_0);
        uint k_50 = *_S1356;
        uint _S1357 = st_f_0(*_S1356, 14U);
        if(((float4(*((&kernelContext_196)->lp_0+_S1357)) ).w) == 0.0f)
        {
            bool _S1358 = lp_factor_0(k_50, &kernelContext_196);
            _S1348 = !_S1358;
        }
        else
        {
            _S1348 = false;
        }
        if(_S1348)
        {
            (*((&kernelContext_196)->lp_0+_S1357))[int(3)] = 1.0f;
            uint _S1359 = a_counters_1(&kernelContext_196);
            uint _S1360 = atomic_fetch_or_explicit((&kernelContext_196)->atom_0+(_S1359 + 3U), 64U, memory_order_relaxed);
        }
        j2_0 = j2_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint sweep_0 = 0U;
    for(;;)
    {
        if(sweep_0 < ((&kernelContext_196)->params_0->cs_sweeps_0))
        {
        }
        else
        {
            break;
        }
        j3_0 = t_15;
        for(;;)
        {
            if(j3_0 < _S1353)
            {
            }
            else
            {
                break;
            }
            uint _S1361 = w_cs_list_0(&kernelContext_196);
            st_sweep_0(*((&kernelContext_196)->work_0+(_S1361 + _S1352 + j3_0)), 14U, sweep_0 % 2U, (&kernelContext_196)->params_0->cs_c1_0, (&kernelContext_196)->params_0->cs_c2_0, &kernelContext_196);
            j3_0 = j3_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        sweep_0 = sweep_0 + 1U;
    }
    j3_0 = t_15;
    for(;;)
    {
        if(j3_0 < _S1353)
        {
        }
        else
        {
            break;
        }
        uint _S1362 = w_cs_list_0(&kernelContext_196);
        uint device* _S1363 = (&kernelContext_196)->work_0+(_S1362 + _S1352 + j3_0);
        uint k_51 = *_S1363;
        float4 _S1364 = float4(*((&kernelContext_196)->ext_0+((&kernelContext_196)->params_0->ext_chunks_0 + 4U * *_S1363))) ;
        uint _S1365 = (as_type<uint>((_S1364.x)));
        uint _S1366 = (as_type<uint>((_S1364.y)));
        uint e_6 = 0U;
        for(;;)
        {
            if(e_6 < _S1366)
            {
            }
            else
            {
                break;
            }
            uint _S1367 = adj_bond_0(_S1365 + e_6, &kernelContext_196);
            if(((uint4(*((&kernelContext_196)->bonds_0+_S1367)) ).x) == k_51)
            {
                uint _S1368 = w_alive_0(&kernelContext_196);
                _S1348 = (*((&kernelContext_196)->work_0+(_S1368 + _S1367))) != 0U;
            }
            else
            {
                _S1348 = false;
            }
            if(_S1348)
            {
                cs_break_0(_S1367, s_21, &kernelContext_196);
            }
            e_6 = e_6 + 1U;
        }
        j3_0 = j3_0 + 256U;
    }
    return;
}

bool crushable_0(uint k_52, KernelContext_0 thread* kernelContext_197)
{
    return ((as_type<uint>(((float4(*(kernelContext_197->crush_tab_0+(2U * k_52 + 1U))) ).z))) & 1U) != 0U;
}

uint crush_flags_0(uint k_53, KernelContext_0 thread* kernelContext_198)
{
    return (as_type<uint>(((float4(*(kernelContext_198->crush_rw_0+k_53)) ).w)));
}

void crush_record_0(uint k_54, float4 step_6, uint cause_0, KernelContext_0 thread* kernelContext_199)
{
    float4 _S1369 = float4(*(kernelContext_199->crush_rw_0+k_54)) ;
    uint flags_1 = (as_type<uint>((_S1369.w)));
    uint flags_2;
    if((step_6.w) != 0.0f)
    {
        flags_2 = ((flags_1 | 1U) | 2U) | (cause_0 << 4U);
    }
    else
    {
        flags_2 = flags_1;
    }
    *(kernelContext_199->crush_rw_0+k_54) = packed_float4(float4(step_6.x, max(_S1369.y, step_6.y), max(_S1369.z, step_6.z), (as_type<float>((flags_2))))) ;
    return;
}

[[kernel]] void fr_crush_blast(uint3 id_70 [[thread_position_in_grid]], FractureParams_0 constant* params_76 [[buffer(0)]], uint device* work_76 [[buffer(4)]], packed_float4 device* frames_76 [[buffer(6)]], packed_float4 device* bodies_76 [[buffer(7)]], packed_float4 device* chunks_76 [[buffer(1)]], uint device* sort_76 [[buffer(11)]], atomic<uint> device* atom_76 [[buffer(5)]], Record_natural_0 device* manifolds_76 [[buffer(10)]], packed_float4 device* loads_76 [[buffer(12)]], packed_uint4 device* bonds_76 [[buffer(2)]], uint device* args_76 [[buffer(13)]], uint device* info_76 [[buffer(15)]], ScanParams_0 constant* scan_76 [[buffer(14)]], packed_float4 device* parts_76 [[buffer(3)]], packed_float4 device* colliders_76 [[buffer(8)]], packed_uint4 device* collider_info_76 [[buffer(9)]], packed_float4 device* view_chunks_76 [[buffer(17)]], ViewParams_0 constant* view_76 [[buffer(16)]], packed_float4 device* objects_76 [[buffer(18)]], uint device* shot_bodies_76 [[buffer(20)]], float device* lines_76 [[buffer(19)]], packed_float4 device* geometry_76 [[buffer(21)]], packed_float4 device* shots_76 [[buffer(22)]], packed_float4 device* lp_76 [[buffer(24)]], packed_float4 device* ext_78 [[buffer(23)]], packed_float4 device* engine_contacts_76 [[buffer(25)]], CrushParams_0 constant* crush_76 [[buffer(28)]], packed_float4 device* crush_rw_76 [[buffer(27)]], packed_float4 device* crush_tab_76 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_200;
    (&kernelContext_200)->params_0 = params_76;
    (&kernelContext_200)->work_0 = work_76;
    (&kernelContext_200)->frames_0 = frames_76;
    (&kernelContext_200)->bodies_0 = bodies_76;
    (&kernelContext_200)->chunks_0 = chunks_76;
    (&kernelContext_200)->sort_0 = sort_76;
    (&kernelContext_200)->atom_0 = atom_76;
    (&kernelContext_200)->manifolds_0 = manifolds_76;
    (&kernelContext_200)->loads_0 = loads_76;
    (&kernelContext_200)->bonds_0 = bonds_76;
    (&kernelContext_200)->args_0 = args_76;
    (&kernelContext_200)->info_0 = info_76;
    (&kernelContext_200)->scan_0 = scan_76;
    (&kernelContext_200)->parts_0 = parts_76;
    (&kernelContext_200)->colliders_0 = colliders_76;
    (&kernelContext_200)->collider_info_0 = collider_info_76;
    (&kernelContext_200)->view_chunks_0 = view_chunks_76;
    (&kernelContext_200)->view_0 = view_76;
    (&kernelContext_200)->objects_0 = objects_76;
    (&kernelContext_200)->shot_bodies_0 = shot_bodies_76;
    (&kernelContext_200)->lines_0 = lines_76;
    (&kernelContext_200)->geometry_0 = geometry_76;
    (&kernelContext_200)->shots_0 = shots_76;
    (&kernelContext_200)->lp_0 = lp_76;
    (&kernelContext_200)->ext_0 = ext_78;
    (&kernelContext_200)->engine_contacts_0 = engine_contacts_76;
    (&kernelContext_200)->crush_0 = crush_76;
    (&kernelContext_200)->crush_rw_0 = crush_rw_76;
    (&kernelContext_200)->crush_tab_0 = crush_tab_76;
    threadgroup array<uint, int(1024)> g_scan_76;
    (&kernelContext_200)->g_scan_0 = &g_scan_76;
    threadgroup array<float, int(64)> g_il_best_76;
    (&kernelContext_200)->g_il_best_0 = &g_il_best_76;
    uint k_55 = id_70.x;
    bool _S1370;
    if(k_55 >= (params_76->chunk_count_0))
    {
        _S1370 = true;
    }
    else
    {
        _S1370 = ((&kernelContext_200)->crush_0->enabled_0) == 0U;
    }
    if(_S1370)
    {
        return;
    }
    float4 _S1371 = float4(*((&kernelContext_200)->crush_rw_0+((&kernelContext_200)->crush_0->r_blast_0 + k_55))) ;
    float _S1372 = _S1371.x;
    if(!(_S1372 > 0.0f))
    {
        _S1370 = !((_S1371.y) > 0.0f);
    }
    else
    {
        _S1370 = false;
    }
    if(_S1370)
    {
        _S1370 = !((_S1371.z) > 0.0f);
    }
    else
    {
        _S1370 = false;
    }
    if(_S1370)
    {
        return;
    }
    *((&kernelContext_200)->crush_rw_0+((&kernelContext_200)->crush_0->r_blast_0 + k_55)) = packed_float4(float4(0.0f) ) ;
    bool _S1373 = crushable_0(k_55, &kernelContext_200);
    if(!_S1373)
    {
        _S1370 = true;
    }
    else
    {
        uint _S1374 = crush_flags_0(k_55, &kernelContext_200);
        _S1370 = (_S1374 & 1U) != 0U;
    }
    if(_S1370)
    {
        return;
    }
    float d_22 = (float4(*((&kernelContext_200)->crush_rw_0+k_55)) ).x + _S1372;
    float4 step_7;
    if(d_22 >= 1.0f)
    {
        step_7 = float4(1.0f, _S1371.y, _S1371.z, 1.0f);
    }
    else
    {
        step_7 = float4(d_22, _S1371.y, _S1371.z, 0.0f);
    }
    crush_record_0(k_55, step_7, 3U, &kernelContext_200);
    return;
}

struct Sym_0
{
    float3 d_23;
    float3 o_9;
};

Sym_0 sym_zero_0()
{
    thread Sym_0 s_22;
    float3 _S1375 = float3(0.0f) ;
    (&s_22)->d_23 = _S1375;
    (&s_22)->o_9 = _S1375;
    return s_22;
}

Sym_0 sym_outer_0(float3 r_28, float3 f_13)
{
    thread Sym_0 s_23;
    (&s_23)->d_23 = r_28 * f_13;
    float _S1376 = r_28.x;
    float _S1377 = f_13.y;
    float _S1378 = r_28.y;
    float _S1379 = f_13.x;
    float _S1380 = f_13.z;
    float _S1381 = r_28.z;
    (&s_23)->o_9 = float3(0.5f)  * float3(_S1376 * _S1377 + _S1378 * _S1379, _S1376 * _S1380 + _S1381 * _S1379, _S1378 * _S1380 + _S1381 * _S1377);
    return s_23;
}

Sym_0 sym_add_0(const Sym_0 thread* a_10, const Sym_0 thread* b_53)
{
    thread Sym_0 s_24;
    (&s_24)->d_23 = a_10->d_23 + b_53->d_23;
    (&s_24)->o_9 = a_10->o_9 + b_53->o_9;
    return s_24;
}

void add_load_0(Sym_0 thread* w_9, float3 thread* f_sum_0, float4 thread* spread_0, float3 r_29, float3 f_14)
{
    Sym_0 _S1382 = sym_outer_0(r_29, f_14);
    thread Sym_0 _S1383 = *w_9;
    thread Sym_0 _S1384 = _S1382;
    Sym_0 _S1385 = sym_add_0(&_S1383, &_S1384);
    *w_9 = _S1385;
    *f_sum_0 = *f_sum_0 + f_14;
    float m_6 = length(f_14);
    *spread_0 = *spread_0 + float4(float3(m_6)  * r_29, m_6);
    return;
}

void flow_virial_0(uint k_56, const Quat_0 thread* q_28, float3 origin_9, float3 centre_1, Sym_0 thread* wf_0, float thread* total_0, KernelContext_0 thread* kernelContext_201)
{
    *wf_0 = sym_zero_0();
    *total_0 = 0.0f;
    float4 _S1386 = float4(*(kernelContext_201->crush_tab_0+(kernelContext_201->crush_0->t_adj_0 + k_56))) ;
    uint first_2 = (as_type<uint>((_S1386.x)));
    uint _S1387 = (as_type<uint>((_S1386.y)));
    uint j_11 = first_2;
    for(;;)
    {
        if(j_11 < (first_2 + _S1387))
        {
        }
        else
        {
            break;
        }
        uint bi_0 = (as_type<uint>(((float4(*(kernelContext_201->crush_tab_0+(kernelContext_201->crush_0->t_list_0 + j_11 / 4U))) )[j_11 % 4U])));
        uint _S1388 = w_alive_0(kernelContext_201);
        if((*(kernelContext_201->work_0+(_S1388 + bi_0))) == 0U)
        {
            j_11 = j_11 + 1U;
            continue;
        }
        float l_2 = (*(kernelContext_201->crush_rw_0+(kernelContext_201->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U];
        if(!(((*(kernelContext_201->crush_rw_0+(kernelContext_201->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U]) > 0.0f))
        {
            j_11 = j_11 + 1U;
            continue;
        }
        float3 _S1389 = rotate_0(q_28, (float4(*(kernelContext_201->crush_tab_0+(kernelContext_201->crush_0->t_bonds_0 + bi_0))) ).xyz);
        float3 r_30 = origin_9 + _S1389 - centre_1;
        float3 _S1390 = rotate_0(q_28, (float4(*(kernelContext_201->crush_tab_0+(kernelContext_201->crush_0->t_normal_0 + bi_0))) ).xyz);
        float3 out_n_0;
        if((dot(_S1390, r_30)) >= 0.0f)
        {
            out_n_0 = _S1390;
        }
        else
        {
            out_n_0 = - _S1390;
        }
        Sym_0 _S1391 = sym_outer_0(r_30, float3(- l_2)  * out_n_0);
        thread Sym_0 _S1392 = *wf_0;
        thread Sym_0 _S1393 = _S1391;
        Sym_0 _S1394 = sym_add_0(&_S1392, &_S1393);
        *wf_0 = _S1394;
        *total_0 = *total_0 + l_2;
        j_11 = j_11 + 1U;
    }
    return;
}

float4 bond_centroid_0(uint k_57, const Quat_0 thread* q_29, float3 origin_10, float3 centre_2, KernelContext_0 thread* kernelContext_202)
{
    float4 _S1395 = float4(*(kernelContext_202->crush_tab_0+(kernelContext_202->crush_0->t_adj_0 + k_57))) ;
    uint first_3 = (as_type<uint>((_S1395.x)));
    uint _S1396 = (as_type<uint>((_S1395.y)));
    float3 _S1397 = float3(0.0f) ;
    uint j_12 = first_3;
    float3 sum_4 = _S1397;
    float area_1 = 0.0f;
    for(;;)
    {
        if(j_12 < (first_3 + _S1396))
        {
        }
        else
        {
            break;
        }
        uint bi_1 = (as_type<uint>(((float4(*(kernelContext_202->crush_tab_0+(kernelContext_202->crush_0->t_list_0 + j_12 / 4U))) )[j_12 % 4U])));
        uint _S1398 = w_alive_0(kernelContext_202);
        if((*(kernelContext_202->work_0+(_S1398 + bi_1))) == 0U)
        {
            j_12 = j_12 + 1U;
            continue;
        }
        float4 _S1399 = float4(*(kernelContext_202->crush_tab_0+(kernelContext_202->crush_0->t_bonds_0 + bi_1))) ;
        float3 _S1400 = rotate_0(q_29, _S1399.xyz);
        float _S1401 = _S1399.w;
        float area_2 = area_1 + _S1401;
        sum_4 = sum_4 + float3(_S1401)  * (origin_10 + _S1400 - centre_2);
        area_1 = area_2;
        j_12 = j_12 + 1U;
    }
    if(area_1 > 0.0f)
    {
        sum_4 = sum_4 / float3(area_1) ;
    }
    else
    {
        sum_4 = _S1397;
    }
    return float4(sum_4, area_1);
}

Sym_0 balanced_0(const Sym_0 thread* w_10, float3 f_15, float4 spread_1, float4 bonds_77)
{
    float3 at_22;
    if((bonds_77.w) > 0.0f)
    {
        at_22 = bonds_77.xyz;
    }
    else
    {
        float _S1402 = spread_1.w;
        if(_S1402 > 0.0f)
        {
            at_22 = spread_1.xyz / float3(_S1402) ;
        }
        else
        {
            at_22 = float3(0.0f) ;
        }
    }
    thread Sym_0 _S1403 = sym_outer_0(at_22, - f_15);
    Sym_0 _S1404 = sym_add_0(w_10, &_S1403);
    return _S1404;
}

float sym_norm2_0(const Sym_0 thread* s_25)
{
    return dot(s_25->d_23, s_25->d_23) + 2.0f * dot(s_25->o_9, s_25->o_9);
}

float2 crush_invariants_0(const Sym_0 thread* v_16, float volume_0)
{
    float inv_4 = 1.0f / volume_0;
    float xx_0 = v_16->d_23.x * inv_4;
    float yy_0 = v_16->d_23.y * inv_4;
    float zz_0 = v_16->d_23.z * inv_4;
    float xy_0 = v_16->o_9.x * inv_4;
    float xz_0 = v_16->o_9.y * inv_4;
    float yz_0 = v_16->o_9.z * inv_4;
    float p_21 = - (xx_0 + yy_0 + zz_0) / 3.0f;
    float dx_0 = xx_0 + p_21;
    float dy_0 = yy_0 + p_21;
    float dz_0 = zz_0 + p_21;
    return float2(p_21, sqrt(1.5f * max(dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0 + 2.0f * (xy_0 * xy_0 + xz_0 * xz_0 + yz_0 * yz_0), 0.0f)));
}

float crush_excess_0(float2 pq_0, float4 m0_0)
{
    float _S1405 = pq_0.x;
    if(!(_S1405 > 0.0f))
    {
        return 0.0f;
    }
    float _S1406 = max(pq_0.y - (m0_0.y + m0_0.z * _S1405), _S1405 - m0_0.x);
    float _S1407;
    if(_S1406 > 0.0f)
    {
        _S1407 = _S1406;
    }
    else
    {
        _S1407 = 0.0f;
    }
    return _S1407;
}

float4 crush_two_0(const Sym_0 thread* a_11, float dt_a_0, const Sym_0 thread* b_54, float dt_b_0, uint k_58, float damage_2, KernelContext_0 thread* kernelContext_203)
{
    uint _S1408 = 2U * k_58;
    float4 _S1409 = float4(*(kernelContext_203->crush_tab_0+_S1408)) ;
    float4 _S1410 = float4(*(kernelContext_203->crush_tab_0+(_S1408 + 1U))) ;
    float _S1411 = _S1410.y;
    float2 _S1412 = crush_invariants_0(a_11, _S1411);
    float2 _S1413 = crush_invariants_0(b_54, _S1411);
    float ea_0 = crush_excess_0(_S1412, _S1409);
    bool _S1414 = dt_b_0 > 0.0f;
    float eb_0;
    if(_S1414)
    {
        eb_0 = crush_excess_0(_S1413, _S1409);
    }
    else
    {
        eb_0 = 0.0f;
    }
    float d_24 = damage_2 + (ea_0 * ea_0 * dt_a_0 + eb_0 * eb_0 * dt_b_0) / (_S1410.x * _S1409.w);
    float2 peak_2;
    if(_S1414)
    {
        peak_2 = max(_S1412, _S1413);
    }
    else
    {
        peak_2 = _S1412;
    }
    if(d_24 >= 1.0f)
    {
        return float4(1.0f, peak_2, 1.0f);
    }
    return float4(d_24, peak_2, 0.0f);
}

void crush_loads_flow_0(uint k_59, KernelContext_0 thread* kernelContext_204)
{
    uint _S1415 = body_of_0(k_59, kernelContext_204);
    uint b_55 = 7U * _S1415;
    thread Quat_0 q_30;
    thread float3 origin_11;
    structure_pose_0(_S1415, b_55, &q_30, &origin_11, kernelContext_204);
    uint _S1416 = 4U * k_59;
    float4 _S1417 = float4(*(kernelContext_204->chunks_0+_S1416)) ;
    float4 _S1418 = float4(*(kernelContext_204->chunks_0+(_S1416 + 3U))) ;
    float3 _S1419 = _S1417.xyz;
    thread Quat_0 _S1420 = q_30;
    float3 _S1421 = rotate_0(&_S1420, _S1419);
    float3 centre_3 = origin_11 + _S1421;
    float inv_mass_8 = (float4(*(kernelContext_204->bodies_0+b_55)) ).w;
    float _S1422;
    if(inv_mass_8 > 0.0f)
    {
        _S1422 = 1.0f / inv_mass_8;
    }
    else
    {
        _S1422 = float_bits_0(2139095040U);
    }
    float dt_2 = kernelContext_204->params_0->dt_0;
    Sym_0 _S1423 = sym_zero_0();
    thread Sym_0 wa_0 = _S1423;
    float3 _S1424 = float3(0.0f) ;
    thread float3 fa_0 = _S1424;
    float4 _S1425 = float4(0.0f) ;
    thread float4 sa_0 = _S1425;
    thread Sym_0 wb_0 = _S1423;
    thread float3 fb_1 = _S1424;
    thread float4 sb_2 = _S1425;
    uint _S1426 = kernelContext_204->sort_0[k_59];
    uint _S1427 = kernelContext_204->sort_0[kernelContext_204->params_0->chunk_count_0 + 1U + k_59];
    float tau_3 = 0.0f;
    uint i_19 = 0U;
    for(;;)
    {
        if(i_19 < _S1427)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_31 = kernelContext_204->manifolds_0[_S1426 + i_19];
        float4 _S1428 = float4(r_31.impulse_0) ;
        float3 _S1429 = _S1428.xyz;
        float j_13 = length(_S1429);
        if(j_13 == 0.0f)
        {
            i_19 = i_19 + 1U;
            continue;
        }
        float4 _S1430 = float4(r_31.other_0) ;
        float m_other_2 = _S1430.x;
        float m_7;
        if(isinf(_S1422))
        {
            if(isinf(m_other_2))
            {
                m_7 = _S1417.w;
            }
            else
            {
                m_7 = m_other_2;
            }
        }
        else
        {
            if(isinf(m_other_2))
            {
                m_7 = _S1422;
            }
            else
            {
                m_7 = _S1422 * m_other_2 / (_S1422 + m_other_2);
            }
        }
        float _S1431 = _S1428.w;
        float _S1432 = min(j_13, 2.0f * m_7 * max(_S1431 - kernelContext_204->params_0->band_0, 0.0f));
        float3 rr_0 = (float4(r_31.point_0) ).xyz - centre_3;
        float3 dir_1 = _S1429 / float3(j_13) ;
        if(_S1432 < j_13)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, rr_0, dir_1 * float3(((j_13 - _S1432) / dt_2)) );
        }
        float tau_4;
        if(_S1432 > 0.0f)
        {
            float _S1433 = _S1418.y;
            float duration_2 = hertz_duration_0(m_7, min(_S1430.z, _S1418.z), 1.0f / ((1.0f - _S1433 * _S1433) / _S1418.x + 1.0f / _S1430.y), _S1431);
            add_load_0(&wb_0, &fb_1, &sb_2, rr_0, dir_1 * float3((3.14159274101257324f * _S1432 / (2.0f * duration_2))) );
            tau_4 = max(tau_3, min(duration_2, dt_2));
        }
        else
        {
            tau_4 = tau_3;
        }
        tau_3 = tau_4;
        i_19 = i_19 + 1U;
    }
    uint _S1434 = frame_parity_0(kernelContext_204);
    uint _S1435 = w_pulses_0(kernelContext_204);
    uint _S1436 = _S1435 + _S1416 + 2U * _S1434;
    uint off_1 = *(kernelContext_204->work_0+_S1436);
    uint _S1437 = *(kernelContext_204->work_0+(_S1436 + 1U));
    uint _S1438 = l_pool_0(_S1434, kernelContext_204);
    uint _S1439 = _S1438 + 3U * off_1;
    uint p_22 = 0U;
    for(;;)
    {
        if(p_22 < _S1437)
        {
        }
        else
        {
            break;
        }
        uint _S1440 = _S1439 + 3U * p_22;
        float4 _S1441 = float4(*(kernelContext_204->loads_0+_S1440)) ;
        float4 _S1442 = float4(*(kernelContext_204->loads_0+(_S1440 + 1U))) ;
        float4 _S1443 = float4(*(kernelContext_204->loads_0+(_S1440 + 2U))) ;
        float mean_2 = pulse_mean_0(_S1441.x, _S1441.y, _S1441.z, _S1441.w, _S1442.w, dt_2);
        if(mean_2 != 0.0f)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, _S1442.xyz - centre_3, _S1443.xyz * float3(mean_2) );
        }
        p_22 = p_22 + 1U;
    }
    thread Quat_0 _S1444 = q_30;
    thread Sym_0 wf_1;
    thread float flow_total_0;
    flow_virial_0(k_59, &_S1444, origin_11, centre_3, &wf_1, &flow_total_0, kernelContext_204);
    bool contact_18;
    if((sa_0.w) > 0.0f)
    {
        contact_18 = true;
    }
    else
    {
        contact_18 = (sb_2.w) > 0.0f;
    }
    bool _S1445;
    if(!contact_18)
    {
        _S1445 = !(flow_total_0 > 0.0f);
    }
    else
    {
        _S1445 = false;
    }
    if(_S1445)
    {
        return;
    }
    Sym_0 _S1446 = wf_1;
    Sym_0 _S1447 = wf_1;
    Sym_0 peak_contact_0;
    Sym_0 base_3;
    Sym_0 peak_3;
    if(contact_18)
    {
        thread Quat_0 _S1448 = q_30;
        float4 _S1449 = bond_centroid_0(k_59, &_S1448, origin_11, centre_3, kernelContext_204);
        thread Sym_0 _S1450 = wa_0;
        Sym_0 _S1451 = balanced_0(&_S1450, fa_0, sa_0, _S1449);
        thread Sym_0 _S1452 = _S1451;
        thread Sym_0 _S1453 = wf_1;
        Sym_0 _S1454 = sym_add_0(&_S1452, &_S1453);
        thread Sym_0 _S1455 = wa_0;
        thread Sym_0 _S1456 = wb_0;
        Sym_0 _S1457 = sym_add_0(&_S1455, &_S1456);
        float3 _S1458 = fa_0 + fb_1;
        float4 _S1459 = sa_0 + sb_2;
        thread Sym_0 _S1460 = _S1457;
        Sym_0 _S1461 = balanced_0(&_S1460, _S1458, _S1459, _S1449);
        thread Sym_0 _S1462 = _S1461;
        thread Sym_0 _S1463 = wf_1;
        Sym_0 _S1464 = sym_add_0(&_S1462, &_S1463);
        peak_contact_0 = _S1461;
        base_3 = _S1454;
        peak_3 = _S1464;
    }
    else
    {
        peak_contact_0 = _S1423;
        base_3 = _S1446;
        peak_3 = _S1447;
    }
    thread Sym_0 _S1465 = wf_1;
    float _S1466 = sym_norm2_0(&_S1465);
    thread Sym_0 _S1467 = peak_contact_0;
    float _S1468 = sym_norm2_0(&_S1467);
    uint cause_1;
    if(_S1466 >= _S1468)
    {
        cause_1 = 4U;
    }
    else
    {
        cause_1 = 1U;
    }
    float _S1469 = dt_2 - tau_3;
    float _S1470 = (float4(*(kernelContext_204->crush_rw_0+k_59)) ).x;
    thread Sym_0 _S1471 = base_3;
    thread Sym_0 _S1472 = peak_3;
    float4 _S1473 = crush_two_0(&_S1471, _S1469, &_S1472, tau_3, k_59, _S1470, kernelContext_204);
    crush_record_0(k_59, _S1473, cause_1, kernelContext_204);
    return;
}

[[kernel]] void fr_crush_loads(uint3 id_71 [[thread_position_in_grid]], FractureParams_0 constant* params_77 [[buffer(0)]], uint device* work_77 [[buffer(4)]], packed_float4 device* frames_77 [[buffer(6)]], packed_float4 device* bodies_77 [[buffer(7)]], packed_float4 device* chunks_77 [[buffer(1)]], uint device* sort_77 [[buffer(11)]], atomic<uint> device* atom_77 [[buffer(5)]], Record_natural_0 device* manifolds_77 [[buffer(10)]], packed_float4 device* loads_77 [[buffer(12)]], packed_uint4 device* bonds_78 [[buffer(2)]], uint device* args_77 [[buffer(13)]], uint device* info_77 [[buffer(15)]], ScanParams_0 constant* scan_77 [[buffer(14)]], packed_float4 device* parts_77 [[buffer(3)]], packed_float4 device* colliders_77 [[buffer(8)]], packed_uint4 device* collider_info_77 [[buffer(9)]], packed_float4 device* view_chunks_77 [[buffer(17)]], ViewParams_0 constant* view_77 [[buffer(16)]], packed_float4 device* objects_77 [[buffer(18)]], uint device* shot_bodies_77 [[buffer(20)]], float device* lines_77 [[buffer(19)]], packed_float4 device* geometry_77 [[buffer(21)]], packed_float4 device* shots_77 [[buffer(22)]], packed_float4 device* lp_77 [[buffer(24)]], packed_float4 device* ext_79 [[buffer(23)]], packed_float4 device* engine_contacts_77 [[buffer(25)]], CrushParams_0 constant* crush_77 [[buffer(28)]], packed_float4 device* crush_rw_77 [[buffer(27)]], packed_float4 device* crush_tab_77 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_205;
    (&kernelContext_205)->params_0 = params_77;
    (&kernelContext_205)->work_0 = work_77;
    (&kernelContext_205)->frames_0 = frames_77;
    (&kernelContext_205)->bodies_0 = bodies_77;
    (&kernelContext_205)->chunks_0 = chunks_77;
    (&kernelContext_205)->sort_0 = sort_77;
    (&kernelContext_205)->atom_0 = atom_77;
    (&kernelContext_205)->manifolds_0 = manifolds_77;
    (&kernelContext_205)->loads_0 = loads_77;
    (&kernelContext_205)->bonds_0 = bonds_78;
    (&kernelContext_205)->args_0 = args_77;
    (&kernelContext_205)->info_0 = info_77;
    (&kernelContext_205)->scan_0 = scan_77;
    (&kernelContext_205)->parts_0 = parts_77;
    (&kernelContext_205)->colliders_0 = colliders_77;
    (&kernelContext_205)->collider_info_0 = collider_info_77;
    (&kernelContext_205)->view_chunks_0 = view_chunks_77;
    (&kernelContext_205)->view_0 = view_77;
    (&kernelContext_205)->objects_0 = objects_77;
    (&kernelContext_205)->shot_bodies_0 = shot_bodies_77;
    (&kernelContext_205)->lines_0 = lines_77;
    (&kernelContext_205)->geometry_0 = geometry_77;
    (&kernelContext_205)->shots_0 = shots_77;
    (&kernelContext_205)->lp_0 = lp_77;
    (&kernelContext_205)->ext_0 = ext_79;
    (&kernelContext_205)->engine_contacts_0 = engine_contacts_77;
    (&kernelContext_205)->crush_0 = crush_77;
    (&kernelContext_205)->crush_rw_0 = crush_rw_77;
    (&kernelContext_205)->crush_tab_0 = crush_tab_77;
    threadgroup array<uint, int(1024)> g_scan_77;
    (&kernelContext_205)->g_scan_0 = &g_scan_77;
    threadgroup array<float, int(64)> g_il_best_77;
    (&kernelContext_205)->g_il_best_0 = &g_il_best_77;
    uint k_60 = id_71.x;
    bool _S1474;
    if(k_60 >= (params_77->chunk_count_0))
    {
        _S1474 = true;
    }
    else
    {
        _S1474 = ((&kernelContext_205)->crush_0->enabled_0) == 0U;
    }
    if(_S1474)
    {
        _S1474 = true;
    }
    else
    {
        bool _S1475 = crushable_0(k_60, &kernelContext_205);
        _S1474 = !_S1475;
    }
    if(_S1474)
    {
        _S1474 = true;
    }
    else
    {
        uint _S1476 = crush_flags_0(k_60, &kernelContext_205);
        _S1474 = (_S1476 & 1U) != 0U;
    }
    if(_S1474)
    {
        return;
    }
    if(((&kernelContext_205)->crush_0->flow_0) != 0U)
    {
        crush_loads_flow_0(k_60, &kernelContext_205);
        return;
    }
    uint _S1477 = body_of_0(k_60, &kernelContext_205);
    uint b_56 = 7U * _S1477;
    thread Quat_0 q_31;
    thread float3 origin_12;
    structure_pose_0(_S1477, b_56, &q_31, &origin_12, &kernelContext_205);
    uint _S1478 = 4U * k_60;
    float4 _S1479 = float4(*((&kernelContext_205)->chunks_0+_S1478)) ;
    float4 _S1480 = float4(*((&kernelContext_205)->chunks_0+(_S1478 + 3U))) ;
    float3 _S1481 = _S1479.xyz;
    thread Quat_0 _S1482 = q_31;
    float3 _S1483 = rotate_0(&_S1482, _S1481);
    float3 centre_4 = origin_12 + _S1483;
    float inv_mass_9 = (float4(*((&kernelContext_205)->bodies_0+b_56)) ).w;
    float _S1484;
    if(inv_mass_9 > 0.0f)
    {
        _S1484 = 1.0f / inv_mass_9;
    }
    else
    {
        _S1484 = float_bits_0(2139095040U);
    }
    float dt_3 = (&kernelContext_205)->params_0->dt_0;
    Sym_0 _S1485 = sym_zero_0();
    thread Sym_0 wa_1 = _S1485;
    float3 _S1486 = float3(0.0f) ;
    thread float3 fa_1 = _S1486;
    float4 _S1487 = float4(0.0f) ;
    thread float4 sa_1 = _S1487;
    thread Sym_0 wb_1 = _S1485;
    thread float3 fb_2 = _S1486;
    thread float4 sb_3 = _S1487;
    uint _S1488 = (&kernelContext_205)->sort_0[k_60];
    uint _S1489 = (&kernelContext_205)->sort_0[params_77->chunk_count_0 + 1U + k_60];
    float tau_5 = 0.0f;
    uint i_20 = 0U;
    for(;;)
    {
        if(i_20 < _S1489)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_32 = (&kernelContext_205)->manifolds_0[_S1488 + i_20];
        float4 _S1490 = float4(r_32.impulse_0) ;
        float3 _S1491 = _S1490.xyz;
        float j_14 = length(_S1491);
        if(j_14 == 0.0f)
        {
            i_20 = i_20 + 1U;
            continue;
        }
        float4 _S1492 = float4(r_32.other_0) ;
        float m_other_3 = _S1492.x;
        float m_8;
        if(isinf(_S1484))
        {
            if(isinf(m_other_3))
            {
                m_8 = _S1479.w;
            }
            else
            {
                m_8 = m_other_3;
            }
        }
        else
        {
            if(isinf(m_other_3))
            {
                m_8 = _S1484;
            }
            else
            {
                m_8 = _S1484 * m_other_3 / (_S1484 + m_other_3);
            }
        }
        float _S1493 = _S1490.w;
        float _S1494 = min(j_14, 2.0f * m_8 * max(_S1493 - (&kernelContext_205)->params_0->band_0, 0.0f));
        float3 rr_1 = (float4(r_32.point_0) ).xyz - centre_4;
        float3 dir_2 = _S1491 / float3(j_14) ;
        if(_S1494 < j_14)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, rr_1, dir_2 * float3(((j_14 - _S1494) / dt_3)) );
        }
        float tau_6;
        if(_S1494 > 0.0f)
        {
            float _S1495 = _S1480.y;
            float duration_3 = hertz_duration_0(m_8, min(_S1492.z, _S1480.z), 1.0f / ((1.0f - _S1495 * _S1495) / _S1480.x + 1.0f / _S1492.y), _S1493);
            add_load_0(&wb_1, &fb_2, &sb_3, rr_1, dir_2 * float3((3.14159274101257324f * _S1494 / (2.0f * duration_3))) );
            tau_6 = max(tau_5, min(duration_3, dt_3));
        }
        else
        {
            tau_6 = tau_5;
        }
        tau_5 = tau_6;
        i_20 = i_20 + 1U;
    }
    uint _S1496 = frame_parity_0(&kernelContext_205);
    uint _S1497 = w_pulses_0(&kernelContext_205);
    uint _S1498 = _S1497 + _S1478 + 2U * _S1496;
    uint device* _S1499 = (&kernelContext_205)->work_0+_S1498;
    uint _S1500 = *((&kernelContext_205)->work_0+(_S1498 + 1U));
    uint _S1501 = l_pool_0(_S1496, &kernelContext_205);
    uint _S1502 = _S1501 + 3U * *_S1499;
    uint p_23 = 0U;
    for(;;)
    {
        if(p_23 < _S1500)
        {
        }
        else
        {
            break;
        }
        uint _S1503 = _S1502 + 3U * p_23;
        float4 _S1504 = float4(*((&kernelContext_205)->loads_0+_S1503)) ;
        float4 _S1505 = float4(*((&kernelContext_205)->loads_0+(_S1503 + 1U))) ;
        float4 _S1506 = float4(*((&kernelContext_205)->loads_0+(_S1503 + 2U))) ;
        float mean_3 = pulse_mean_0(_S1504.x, _S1504.y, _S1504.z, _S1504.w, _S1505.w, dt_3);
        if(mean_3 != 0.0f)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, _S1505.xyz - centre_4, _S1506.xyz * float3(mean_3) );
        }
        p_23 = p_23 + 1U;
    }
    if(!((sa_1.w) > 0.0f))
    {
        _S1474 = !((sb_3.w) > 0.0f);
    }
    else
    {
        _S1474 = false;
    }
    if(_S1474)
    {
        return;
    }
    thread Quat_0 _S1507 = q_31;
    float4 _S1508 = bond_centroid_0(k_60, &_S1507, origin_12, centre_4, &kernelContext_205);
    thread Sym_0 _S1509 = wa_1;
    Sym_0 _S1510 = balanced_0(&_S1509, fa_1, sa_1, _S1508);
    thread Sym_0 _S1511 = wa_1;
    thread Sym_0 _S1512 = wb_1;
    Sym_0 _S1513 = sym_add_0(&_S1511, &_S1512);
    float3 _S1514 = fa_1 + fb_2;
    float4 _S1515 = sa_1 + sb_3;
    thread Sym_0 _S1516 = _S1513;
    Sym_0 _S1517 = balanced_0(&_S1516, _S1514, _S1515, _S1508);
    float _S1518 = dt_3 - tau_5;
    float _S1519 = (float4(*((&kernelContext_205)->crush_rw_0+k_60)) ).x;
    thread Sym_0 _S1520 = _S1510;
    thread Sym_0 _S1521 = _S1517;
    float4 _S1522 = crush_two_0(&_S1520, _S1518, &_S1521, tau_5, k_60, _S1519, &kernelContext_205);
    crush_record_0(k_60, _S1522, 1U, &kernelContext_205);
    return;
}

uint a_cover_0(KernelContext_0 thread* kernelContext_206)
{
    uint _S1523 = a_strike_dist_0(kernelContext_206);
    return _S1523 + kernelContext_206->params_0->chunk_count_0;
}

[[kernel]] void fr_strike_patch(uint3 id_72 [[thread_position_in_grid]], FractureParams_0 constant* params_78 [[buffer(0)]], uint device* work_78 [[buffer(4)]], packed_float4 device* frames_78 [[buffer(6)]], packed_float4 device* bodies_78 [[buffer(7)]], packed_float4 device* chunks_78 [[buffer(1)]], uint device* sort_78 [[buffer(11)]], atomic<uint> device* atom_78 [[buffer(5)]], Record_natural_0 device* manifolds_78 [[buffer(10)]], packed_float4 device* loads_78 [[buffer(12)]], packed_uint4 device* bonds_79 [[buffer(2)]], uint device* args_78 [[buffer(13)]], uint device* info_78 [[buffer(15)]], ScanParams_0 constant* scan_78 [[buffer(14)]], packed_float4 device* parts_78 [[buffer(3)]], packed_float4 device* colliders_78 [[buffer(8)]], packed_uint4 device* collider_info_78 [[buffer(9)]], packed_float4 device* view_chunks_78 [[buffer(17)]], ViewParams_0 constant* view_78 [[buffer(16)]], packed_float4 device* objects_78 [[buffer(18)]], uint device* shot_bodies_78 [[buffer(20)]], float device* lines_78 [[buffer(19)]], packed_float4 device* geometry_78 [[buffer(21)]], packed_float4 device* shots_78 [[buffer(22)]], packed_float4 device* lp_78 [[buffer(24)]], packed_float4 device* ext_80 [[buffer(23)]], packed_float4 device* engine_contacts_78 [[buffer(25)]], CrushParams_0 constant* crush_78 [[buffer(28)]], packed_float4 device* crush_rw_78 [[buffer(27)]], packed_float4 device* crush_tab_78 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_207;
    (&kernelContext_207)->params_0 = params_78;
    (&kernelContext_207)->work_0 = work_78;
    (&kernelContext_207)->frames_0 = frames_78;
    (&kernelContext_207)->bodies_0 = bodies_78;
    (&kernelContext_207)->chunks_0 = chunks_78;
    (&kernelContext_207)->sort_0 = sort_78;
    (&kernelContext_207)->atom_0 = atom_78;
    (&kernelContext_207)->manifolds_0 = manifolds_78;
    (&kernelContext_207)->loads_0 = loads_78;
    (&kernelContext_207)->bonds_0 = bonds_79;
    (&kernelContext_207)->args_0 = args_78;
    (&kernelContext_207)->info_0 = info_78;
    (&kernelContext_207)->scan_0 = scan_78;
    (&kernelContext_207)->parts_0 = parts_78;
    (&kernelContext_207)->colliders_0 = colliders_78;
    (&kernelContext_207)->collider_info_0 = collider_info_78;
    (&kernelContext_207)->view_chunks_0 = view_chunks_78;
    (&kernelContext_207)->view_0 = view_78;
    (&kernelContext_207)->objects_0 = objects_78;
    (&kernelContext_207)->shot_bodies_0 = shot_bodies_78;
    (&kernelContext_207)->lines_0 = lines_78;
    (&kernelContext_207)->geometry_0 = geometry_78;
    (&kernelContext_207)->shots_0 = shots_78;
    (&kernelContext_207)->lp_0 = lp_78;
    (&kernelContext_207)->ext_0 = ext_80;
    (&kernelContext_207)->engine_contacts_0 = engine_contacts_78;
    (&kernelContext_207)->crush_0 = crush_78;
    (&kernelContext_207)->crush_rw_0 = crush_rw_78;
    (&kernelContext_207)->crush_tab_0 = crush_tab_78;
    threadgroup array<uint, int(1024)> g_scan_78;
    (&kernelContext_207)->g_scan_0 = &g_scan_78;
    threadgroup array<float, int(64)> g_il_best_78;
    (&kernelContext_207)->g_il_best_0 = &g_il_best_78;
    uint b_57 = id_72.x;
    bool _S1524;
    if(b_57 >= (params_78->body_count_0))
    {
        _S1524 = true;
    }
    else
    {
        _S1524 = ((&kernelContext_207)->params_0->shot_count_0) == 0U;
    }
    if(_S1524)
    {
        _S1524 = true;
    }
    else
    {
        _S1524 = ((&kernelContext_207)->crush_0->enabled_0) == 0U;
    }
    if(_S1524)
    {
        return;
    }
    uint bill_0 = (&kernelContext_207)->crush_0->r_bill_0 + 4U * b_57;
    packed_float4 _S1525 = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_207)->crush_rw_0+bill_0) = _S1525;
    uint _S1526 = bill_0 + 3U;
    *((&kernelContext_207)->crush_rw_0+_S1526) = _S1525;
    thread uint shot_10;
    bool _S1527 = struck_0(b_57, &shot_10, &kernelContext_207);
    if(!_S1527)
    {
        return;
    }
    uint _S1528 = a_strike_0(&kernelContext_207);
    uint _S1529 = _S1528 + 9U * b_57;
    uint k_61 = atomic_load_explicit((&kernelContext_207)->atom_0+(_S1529 + 8U), memory_order_relaxed);
    uint _S1530 = atomic_load_explicit((&kernelContext_207)->atom_0+(_S1529 + 6U), memory_order_relaxed);
    float t_in_10 = (as_type<float>((_S1530)));
    if(!(t_in_10 <= ((&kernelContext_207)->params_0->dt_0)))
    {
        _S1524 = true;
    }
    else
    {
        _S1524 = k_61 == 4294967295U;
    }
    if(_S1524)
    {
        return;
    }
    uint _S1531 = strike_at_0(b_57, &kernelContext_207);
    float4 _S1532 = float4(*((&kernelContext_207)->frames_0+_S1531)) ;
    float3 _S1533 = (float4(*((&kernelContext_207)->frames_0+(_S1531 + 1U))) ).xyz;
    float speed_15 = length(_S1533);
    float3 d_25 = _S1533 / float3(max(speed_15, 9.999999960041972e-13f)) ;
    float4 _S1534 = float4(*((&kernelContext_207)->shots_0+2U * shot_10)) ;
    float radius_1 = _S1534.y;
    float _S1535 = max(_S1534.z, 0.00100000004749745f);
    float m_imp_1 = _S1534.w;
    float inv_mass_10 = (float4(*((&kernelContext_207)->bodies_0+7U * b_57)) ).w;
    bool _S1536 = inv_mass_10 == 0.0f;
    float m_body_2;
    if(_S1536)
    {
        m_body_2 = 0.0f;
    }
    else
    {
        m_body_2 = 1.0f / inv_mass_10;
    }
    float m_eff_9;
    if(_S1536)
    {
        m_eff_9 = m_imp_1;
    }
    else
    {
        m_eff_9 = m_imp_1 * m_body_2 / (m_imp_1 + m_body_2);
    }
    float modulus_2 = 1.0f / (1.0f / (float4(*((&kernelContext_207)->shots_0+(2U * shot_10 + 1U))) ).x + 1.0f / (float4(*((&kernelContext_207)->geometry_0+(3U * k_61 + 1U))) ).w);
    float _S1537 = max(speed_15, 0.00100000004749745f);
    float tau_7 = 2.86999988555908203f * pow(m_eff_9 * m_eff_9 / (_S1535 * modulus_2 * modulus_2 * _S1537), 0.20000000298023224f);
    float force_3 = m_eff_9 * speed_15 / tau_7;
    float3 contact_19 = _S1532.xyz + _S1533 * float3(t_in_10)  + d_25 * float3(radius_1) ;
    float delta_3 = pow(15.0f * m_eff_9 * _S1537 * _S1537 / (16.0f * modulus_2 * sqrt(_S1535)), 0.40000000596046448f);
    float a_h_0 = sqrt(radius_1 * delta_3);
    bool _S1538 = crushable_0(k_61, &kernelContext_207);
    float p_cap_0;
    if(_S1538)
    {
        p_cap_0 = (float4(*((&kernelContext_207)->crush_tab_0+2U * k_61)) ).x;
    }
    else
    {
        p_cap_0 = 0.0f;
    }
    float a_p_0;
    if(p_cap_0 > 0.0f)
    {
        a_p_0 = sqrt(force_3 / (3.14159274101257324f * p_cap_0));
    }
    else
    {
        a_p_0 = 0.0f;
    }
    float _S1539 = min(radius_1, max(a_h_0, a_p_0));
    float _S1540 = max(delta_3, radius_1 - sqrt(max(radius_1 * radius_1 - _S1539 * _S1539, 0.0f)));
    *((&kernelContext_207)->crush_rw_0+(bill_0 + 1U)) = packed_float4(float4(contact_19, 0.0f)) ;
    *((&kernelContext_207)->crush_rw_0+(bill_0 + 2U)) = packed_float4(float4(d_25, _S1539)) ;
    *((&kernelContext_207)->crush_rw_0+_S1526) = packed_float4(float4(force_3, tau_7, _S1540, float(k_61 + 1U))) ;
    uint _S1541 = a_cover_0(&kernelContext_207);
    uint _S1542 = _S1541 + 128U * b_57;
    uint s_26 = 0U;
    for(;;)
    {
        if(s_26 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S1543 = _S1542 + 2U * s_26;
        atomic_store_explicit((&kernelContext_207)->atom_0+_S1543, 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_207)->atom_0+(_S1543 + 1U), 4294967295U, memory_order_relaxed);
        s_26 = s_26 + 1U;
    }
    return;
}

bool cover_patch_0(uint k_62, uint thread* b_58, float4 thread* p1_0, float4 thread* p2_2, KernelContext_0 thread* kernelContext_208)
{
    uint _S1544 = body_of_0(k_62, kernelContext_208);
    *b_58 = _S1544;
    float4 _S1545 = float4(0.0f) ;
    *p1_0 = _S1545;
    *p2_2 = _S1545;
    thread uint shot_11;
    bool _S1546 = struck_0(*b_58, &shot_11, kernelContext_208);
    bool _S1547;
    if(!_S1546)
    {
        _S1547 = true;
    }
    else
    {
        uint _S1548 = crush_flags_0(k_62, kernelContext_208);
        _S1547 = (_S1548 & 1U) != 0U;
    }
    if(_S1547)
    {
        return false;
    }
    uint bill_1 = kernelContext_208->crush_0->r_bill_0 + 4U * *b_58;
    if(((float4(*(kernelContext_208->crush_rw_0+(bill_1 + 3U))) ).w) == 0.0f)
    {
        return false;
    }
    *p1_0 = float4((float4(*(kernelContext_208->crush_rw_0+(bill_1 + 1U))) ).xyz, (float4(*(kernelContext_208->shots_0+2U * shot_11)) ).y);
    *p2_2 = float4(*(kernelContext_208->crush_rw_0+(bill_1 + 2U))) ;
    return true;
}

float3 cover_offset_0(uint s_27, float3 d_26, float a_12, float thread* rho_0)
{
    uint ring_2 = s_27 / 8U;
    uint sector_0 = s_27 % 8U;
    *rho_0 = a_12 * sqrt((float(ring_2) + 0.5f) / 8.0f);
    float theta_1 = 6.28318548202514648f * (float(sector_0) + 0.5f * float(ring_2 & 1U)) / 8.0f;
    float3 e_7;
    if((abs(d_26.x)) < 0.57735002040863037f)
    {
        e_7 = float3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_26.y)) < 0.57735002040863037f)
        {
            e_7 = float3(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_7 = float3(0.0f, 0.0f, 1.0f);
        }
    }
    float3 u_11 = normalize(cross(d_26, e_7));
    return float3(*rho_0)  * (float3(cos(theta_1))  * u_11 + float3(sin(theta_1))  * cross(d_26, u_11));
}

float cover_depth_0(uint k_63, uint s_28, float3 contact_20, float3 d_27, float a_13, float radius_2, KernelContext_0 thread* kernelContext_209)
{
    thread float rho_1;
    float _S1549 = 2.0f * radius_2;
    float3 _S1550 = contact_20 + cover_offset_0(s_28, d_27, a_13, &rho_1) - d_27 * float3(_S1549) ;
    thread Quat_0 q_32;
    float3 _S1551 = chunk_box_world_0(k_63, &q_32, kernelContext_209);
    uint _S1552 = 3U * k_63;
    Quat_0 _S1553 = quat_of_0(float4(*(kernelContext_209->geometry_0+(_S1552 + 2U))) );
    thread Quat_0 _S1554 = q_32;
    thread Quat_0 _S1555 = _S1553;
    Quat_0 _S1556 = quat_mul_0(&_S1554, &_S1555);
    thread Quat_0 _S1557 = _S1556;
    Quat_0 _S1558 = normalized_0(&_S1557);
    float3 _S1559 = (float4(*(kernelContext_209->geometry_0+(_S1552 + 1U))) ).xyz;
    float t0_3 = -1.00000001504746622e+30f;
    float t1_4 = 1.00000001504746622e+30f;
    uint ax_4 = 0U;
    for(;;)
    {
        if(ax_4 < 3U)
        {
        }
        else
        {
            break;
        }
        thread float3 e_8 = float3(0.0f) ;
        e_8[ax_4] = 1.0f;
        thread Quat_0 _S1560 = _S1558;
        float3 _S1561 = rotate_0(&_S1560, e_8);
        float o_10 = dot(_S1550 - _S1551, _S1561);
        float dv_2 = dot(d_27, _S1561);
        if((abs(dv_2)) < 9.99999971718068537e-10f)
        {
            if((abs(o_10)) > (_S1559[ax_4]))
            {
                return 1.00000001504746622e+30f;
            }
            ax_4 = ax_4 + 1U;
            continue;
        }
        float ta_3 = (- _S1559[ax_4] - o_10) / dv_2;
        float tb_3 = (_S1559[ax_4] - o_10) / dv_2;
        float _S1562 = min(t1_4, max(ta_3, tb_3));
        t0_3 = max(t0_3, min(ta_3, tb_3));
        t1_4 = _S1562;
        ax_4 = ax_4 + 1U;
    }
    bool _S1563;
    if(!(t0_3 <= t1_4))
    {
        _S1563 = true;
    }
    else
    {
        _S1563 = t1_4 < 0.0f;
    }
    if(_S1563)
    {
        return 1.00000001504746622e+30f;
    }
    return t0_3 - _S1549;
}

[[kernel]] void fr_strike_cover(uint3 id_73 [[thread_position_in_grid]], FractureParams_0 constant* params_79 [[buffer(0)]], uint device* work_79 [[buffer(4)]], packed_float4 device* frames_79 [[buffer(6)]], packed_float4 device* bodies_79 [[buffer(7)]], packed_float4 device* chunks_79 [[buffer(1)]], uint device* sort_79 [[buffer(11)]], atomic<uint> device* atom_79 [[buffer(5)]], Record_natural_0 device* manifolds_79 [[buffer(10)]], packed_float4 device* loads_79 [[buffer(12)]], packed_uint4 device* bonds_80 [[buffer(2)]], uint device* args_79 [[buffer(13)]], uint device* info_79 [[buffer(15)]], ScanParams_0 constant* scan_79 [[buffer(14)]], packed_float4 device* parts_79 [[buffer(3)]], packed_float4 device* colliders_79 [[buffer(8)]], packed_uint4 device* collider_info_79 [[buffer(9)]], packed_float4 device* view_chunks_79 [[buffer(17)]], ViewParams_0 constant* view_79 [[buffer(16)]], packed_float4 device* objects_79 [[buffer(18)]], uint device* shot_bodies_79 [[buffer(20)]], float device* lines_79 [[buffer(19)]], packed_float4 device* geometry_79 [[buffer(21)]], packed_float4 device* shots_79 [[buffer(22)]], packed_float4 device* lp_79 [[buffer(24)]], packed_float4 device* ext_81 [[buffer(23)]], packed_float4 device* engine_contacts_79 [[buffer(25)]], CrushParams_0 constant* crush_79 [[buffer(28)]], packed_float4 device* crush_rw_79 [[buffer(27)]], packed_float4 device* crush_tab_79 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_210;
    (&kernelContext_210)->params_0 = params_79;
    (&kernelContext_210)->work_0 = work_79;
    (&kernelContext_210)->frames_0 = frames_79;
    (&kernelContext_210)->bodies_0 = bodies_79;
    (&kernelContext_210)->chunks_0 = chunks_79;
    (&kernelContext_210)->sort_0 = sort_79;
    (&kernelContext_210)->atom_0 = atom_79;
    (&kernelContext_210)->manifolds_0 = manifolds_79;
    (&kernelContext_210)->loads_0 = loads_79;
    (&kernelContext_210)->bonds_0 = bonds_80;
    (&kernelContext_210)->args_0 = args_79;
    (&kernelContext_210)->info_0 = info_79;
    (&kernelContext_210)->scan_0 = scan_79;
    (&kernelContext_210)->parts_0 = parts_79;
    (&kernelContext_210)->colliders_0 = colliders_79;
    (&kernelContext_210)->collider_info_0 = collider_info_79;
    (&kernelContext_210)->view_chunks_0 = view_chunks_79;
    (&kernelContext_210)->view_0 = view_79;
    (&kernelContext_210)->objects_0 = objects_79;
    (&kernelContext_210)->shot_bodies_0 = shot_bodies_79;
    (&kernelContext_210)->lines_0 = lines_79;
    (&kernelContext_210)->geometry_0 = geometry_79;
    (&kernelContext_210)->shots_0 = shots_79;
    (&kernelContext_210)->lp_0 = lp_79;
    (&kernelContext_210)->ext_0 = ext_81;
    (&kernelContext_210)->engine_contacts_0 = engine_contacts_79;
    (&kernelContext_210)->crush_0 = crush_79;
    (&kernelContext_210)->crush_rw_0 = crush_rw_79;
    (&kernelContext_210)->crush_tab_0 = crush_tab_79;
    threadgroup array<uint, int(1024)> g_scan_79;
    (&kernelContext_210)->g_scan_0 = &g_scan_79;
    threadgroup array<float, int(64)> g_il_best_79;
    (&kernelContext_210)->g_il_best_0 = &g_il_best_79;
    uint k_64 = id_73.x;
    bool _S1564;
    if(k_64 >= (params_79->chunk_count_0))
    {
        _S1564 = true;
    }
    else
    {
        _S1564 = ((&kernelContext_210)->params_0->shot_count_0) == 0U;
    }
    if(_S1564)
    {
        _S1564 = true;
    }
    else
    {
        _S1564 = ((&kernelContext_210)->crush_0->enabled_0) == 0U;
    }
    if(_S1564)
    {
        return;
    }
    thread uint b_59;
    thread float4 p1_1;
    thread float4 p2_3;
    bool _S1565 = cover_patch_0(k_64, &b_59, &p1_1, &p2_3, &kernelContext_210);
    if(!_S1565)
    {
        return;
    }
    uint _S1566 = a_cover_0(&kernelContext_210);
    uint _S1567 = _S1566 + 128U * b_59;
    uint s_29 = 0U;
    for(;;)
    {
        if(s_29 < 64U)
        {
        }
        else
        {
            break;
        }
        float _S1568 = cover_depth_0(k_64, s_29, p1_1.xyz, p2_3.xyz, p2_3.w, p1_1.w, &kernelContext_210);
        if(_S1568 < 5.0000000752373311e+29f)
        {
            uint _S1569 = atomic_fetch_min_explicit((&kernelContext_210)->atom_0+(_S1567 + 2U * s_29), ordered_0(_S1568), memory_order_relaxed);
        }
        s_29 = s_29 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_claim(uint3 id_74 [[thread_position_in_grid]], FractureParams_0 constant* params_80 [[buffer(0)]], uint device* work_80 [[buffer(4)]], packed_float4 device* frames_80 [[buffer(6)]], packed_float4 device* bodies_80 [[buffer(7)]], packed_float4 device* chunks_80 [[buffer(1)]], uint device* sort_80 [[buffer(11)]], atomic<uint> device* atom_80 [[buffer(5)]], Record_natural_0 device* manifolds_80 [[buffer(10)]], packed_float4 device* loads_80 [[buffer(12)]], packed_uint4 device* bonds_81 [[buffer(2)]], uint device* args_80 [[buffer(13)]], uint device* info_80 [[buffer(15)]], ScanParams_0 constant* scan_80 [[buffer(14)]], packed_float4 device* parts_80 [[buffer(3)]], packed_float4 device* colliders_80 [[buffer(8)]], packed_uint4 device* collider_info_80 [[buffer(9)]], packed_float4 device* view_chunks_80 [[buffer(17)]], ViewParams_0 constant* view_80 [[buffer(16)]], packed_float4 device* objects_80 [[buffer(18)]], uint device* shot_bodies_80 [[buffer(20)]], float device* lines_80 [[buffer(19)]], packed_float4 device* geometry_80 [[buffer(21)]], packed_float4 device* shots_80 [[buffer(22)]], packed_float4 device* lp_80 [[buffer(24)]], packed_float4 device* ext_82 [[buffer(23)]], packed_float4 device* engine_contacts_80 [[buffer(25)]], CrushParams_0 constant* crush_80 [[buffer(28)]], packed_float4 device* crush_rw_80 [[buffer(27)]], packed_float4 device* crush_tab_80 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_211;
    (&kernelContext_211)->params_0 = params_80;
    (&kernelContext_211)->work_0 = work_80;
    (&kernelContext_211)->frames_0 = frames_80;
    (&kernelContext_211)->bodies_0 = bodies_80;
    (&kernelContext_211)->chunks_0 = chunks_80;
    (&kernelContext_211)->sort_0 = sort_80;
    (&kernelContext_211)->atom_0 = atom_80;
    (&kernelContext_211)->manifolds_0 = manifolds_80;
    (&kernelContext_211)->loads_0 = loads_80;
    (&kernelContext_211)->bonds_0 = bonds_81;
    (&kernelContext_211)->args_0 = args_80;
    (&kernelContext_211)->info_0 = info_80;
    (&kernelContext_211)->scan_0 = scan_80;
    (&kernelContext_211)->parts_0 = parts_80;
    (&kernelContext_211)->colliders_0 = colliders_80;
    (&kernelContext_211)->collider_info_0 = collider_info_80;
    (&kernelContext_211)->view_chunks_0 = view_chunks_80;
    (&kernelContext_211)->view_0 = view_80;
    (&kernelContext_211)->objects_0 = objects_80;
    (&kernelContext_211)->shot_bodies_0 = shot_bodies_80;
    (&kernelContext_211)->lines_0 = lines_80;
    (&kernelContext_211)->geometry_0 = geometry_80;
    (&kernelContext_211)->shots_0 = shots_80;
    (&kernelContext_211)->lp_0 = lp_80;
    (&kernelContext_211)->ext_0 = ext_82;
    (&kernelContext_211)->engine_contacts_0 = engine_contacts_80;
    (&kernelContext_211)->crush_0 = crush_80;
    (&kernelContext_211)->crush_rw_0 = crush_rw_80;
    (&kernelContext_211)->crush_tab_0 = crush_tab_80;
    threadgroup array<uint, int(1024)> g_scan_80;
    (&kernelContext_211)->g_scan_0 = &g_scan_80;
    threadgroup array<float, int(64)> g_il_best_80;
    (&kernelContext_211)->g_il_best_0 = &g_il_best_80;
    uint k_65 = id_74.x;
    bool _S1570;
    if(k_65 >= (params_80->chunk_count_0))
    {
        _S1570 = true;
    }
    else
    {
        _S1570 = ((&kernelContext_211)->params_0->shot_count_0) == 0U;
    }
    if(_S1570)
    {
        _S1570 = true;
    }
    else
    {
        _S1570 = ((&kernelContext_211)->crush_0->enabled_0) == 0U;
    }
    if(_S1570)
    {
        return;
    }
    thread uint b_60;
    thread float4 p1_2;
    thread float4 p2_4;
    bool _S1571 = cover_patch_0(k_65, &b_60, &p1_2, &p2_4, &kernelContext_211);
    if(!_S1571)
    {
        return;
    }
    uint _S1572 = a_cover_0(&kernelContext_211);
    uint _S1573 = _S1572 + 128U * b_60;
    uint s_30 = 0U;
    for(;;)
    {
        if(s_30 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S1574 = _S1573 + 2U * s_30;
        uint first_4 = atomic_load_explicit((&kernelContext_211)->atom_0+_S1574, memory_order_relaxed);
        if(first_4 == 4294967295U)
        {
            s_30 = s_30 + 1U;
            continue;
        }
        float _S1575 = cover_depth_0(k_65, s_30, p1_2.xyz, p2_4.xyz, p2_4.w, p1_2.w, &kernelContext_211);
        if(_S1575 < 5.0000000752373311e+29f)
        {
            _S1570 = _S1575 <= (unordered_0(first_4) + 0.00009999999747379f);
        }
        else
        {
            _S1570 = false;
        }
        if(_S1570)
        {
            uint _S1576 = atomic_fetch_min_explicit((&kernelContext_211)->atom_0+(_S1574 + 1U), k_65, memory_order_relaxed);
        }
        s_30 = s_30 + 1U;
    }
    return;
}

float strike_load_0(uint k_66, uint slot_6, float3 f_16, float3 at_23, float tau_8, float budget_0, KernelContext_0 thread* kernelContext_212)
{
    bool _S1577 = crushable_0(k_66, kernelContext_212);
    bool _S1578;
    if(!_S1577)
    {
        _S1578 = true;
    }
    else
    {
        uint _S1579 = crush_flags_0(k_66, kernelContext_212);
        _S1578 = (_S1579 & 1U) != 0U;
    }
    if(_S1578)
    {
        _S1578 = true;
    }
    else
    {
        _S1578 = !(budget_0 > 0.0f);
    }
    if(_S1578)
    {
        return 0.0f;
    }
    thread Quat_0 q_33;
    thread float3 origin_13;
    structure_pose_0(slot_6, 7U * slot_6, &q_33, &origin_13, kernelContext_212);
    float3 _S1580 = (float4(*(kernelContext_212->chunks_0+4U * k_66)) ).xyz;
    thread Quat_0 _S1581 = q_33;
    float3 _S1582 = rotate_0(&_S1581, _S1580);
    float3 centre_5 = origin_13 + _S1582;
    thread Quat_0 _S1583 = q_33;
    float4 _S1584 = bond_centroid_0(k_66, &_S1583, origin_13, centre_5, kernelContext_212);
    if(!((_S1584.w) > 0.0f))
    {
        return 0.0f;
    }
    Sym_0 _S1585 = sym_outer_0(_S1584.xyz, - f_16);
    thread Sym_0 _S1586 = sym_outer_0(at_23 - centre_5, f_16);
    thread Sym_0 _S1587 = _S1585;
    Sym_0 _S1588 = sym_add_0(&_S1586, &_S1587);
    float before_0 = (float4(*(kernelContext_212->crush_rw_0+k_66)) ).x;
    float _S1589 = min(tau_8, kernelContext_212->params_0->dt_0);
    thread Sym_0 _S1590 = _S1588;
    thread Sym_0 _S1591 = _S1588;
    float4 _S1592 = crush_two_0(&_S1590, _S1589, &_S1591, 0.0f, k_66, before_0, kernelContext_212);
    uint _S1593 = 2U * k_66;
    float per_0 = (float4(*(kernelContext_212->crush_tab_0+_S1593)) ).w * (float4(*(kernelContext_212->crush_tab_0+(_S1593 + 1U))) ).y;
    float energy_1 = (_S1592.x - before_0) * per_0;
    float energy_2;
    float4 step_8;
    if(energy_1 > budget_0)
    {
        float d_28 = before_0 + budget_0 / per_0;
        float _S1594 = min(d_28, 1.0f);
        float _S1595 = _S1592.y;
        float _S1596 = _S1592.z;
        if(d_28 >= 1.0f)
        {
            energy_2 = 1.0f;
        }
        else
        {
            energy_2 = 0.0f;
        }
        step_8 = float4(_S1594, _S1595, _S1596, energy_2);
        energy_2 = budget_0;
    }
    else
    {
        step_8 = _S1592;
        energy_2 = energy_1;
    }
    crush_record_0(k_66, step_8, 2U, kernelContext_212);
    return energy_2;
}

bool cell_in_0(uint2 mask_0, uint c_24)
{
    return (((mask_0[c_24 / 32U]) >> (c_24 % 32U)) & 1U) != 0U;
}

float strike_crush_body_0(uint b_61, uint s_31, float budget_1, KernelContext_0 thread* kernelContext_213)
{
    uint total_1;
    bool _S1597;
    uint bill_2 = kernelContext_213->crush_0->r_bill_0 + 4U * b_61;
    uint _S1598 = bill_2 + 1U;
    float4 _S1599 = float4(*(kernelContext_213->crush_rw_0+(bill_2 + 2U))) ;
    float4 _S1600 = float4(*(kernelContext_213->crush_rw_0+(bill_2 + 3U))) ;
    float3 contact_21 = (float4(*(kernelContext_213->crush_rw_0+_S1598)) ).xyz;
    float3 d_29 = _S1599.xyz;
    float _S1601 = _S1599.w;
    float _S1602 = (float4(*(kernelContext_213->shots_0+2U * s_31)) ).y;
    uint _S1603 = a_cover_0(kernelContext_213);
    uint _S1604 = _S1603 + 128U * b_61;
    float first_5 = 1.00000001504746622e+30f;
    uint c_25 = 0U;
    for(;;)
    {
        if(c_25 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S1605 = _S1604 + 2U * c_25;
        uint _S1606 = atomic_load_explicit(kernelContext_213->atom_0+(_S1605 + 1U), memory_order_relaxed);
        if(_S1606 == 4294967295U)
        {
            c_25 = c_25 + 1U;
            continue;
        }
        thread float rho_2;
        float3 _S1607 = cover_offset_0(c_25, d_29, _S1601, &rho_2);
        uint _S1608 = atomic_load_explicit(kernelContext_213->atom_0+_S1605, memory_order_relaxed);
        first_5 = min(first_5, unordered_0(_S1608) + _S1602 - sqrt(max(_S1602 * _S1602 - rho_2 * rho_2, 0.0f)));
        c_25 = c_25 + 1U;
    }
    thread uint2 counted_0 = uint2(0U) ;
    c_25 = 0U;
    uint total_2 = 0U;
    for(;;)
    {
        if(c_25 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S1609 = _S1604 + 2U * c_25;
        uint _S1610 = atomic_load_explicit(kernelContext_213->atom_0+(_S1609 + 1U), memory_order_relaxed);
        if(_S1610 == 4294967295U)
        {
            c_25 = c_25 + 1U;
            continue;
        }
        thread float rho_3;
        float3 _S1611 = cover_offset_0(c_25, d_29, _S1601, &rho_3);
        uint _S1612 = atomic_load_explicit(kernelContext_213->atom_0+_S1609, memory_order_relaxed);
        if((unordered_0(_S1612) + _S1602 - sqrt(max(_S1602 * _S1602 - rho_3 * rho_3, 0.0f)) - first_5) <= (_S1600.z))
        {
            counted_0[c_25 / 32U] = (counted_0[c_25 / 32U]) | (1U << (c_25 % 32U));
            total_1 = total_2 + 1U;
        }
        else
        {
            total_1 = total_2;
        }
        total_2 = total_1;
        c_25 = c_25 + 1U;
    }
    float3 _S1613 = float3(0.0f) ;
    float energy_3;
    float3 point_1;
    if(total_2 == 0U)
    {
        float _S1614 = strike_load_0(uint(_S1600.w) - 1U, b_61, d_29 * float3(_S1600.x) , contact_21, _S1600.y, budget_1, kernelContext_213);
        float3 _S1615 = float3(_S1614)  * contact_21;
        energy_3 = _S1614;
        point_1 = _S1615;
    }
    else
    {
        energy_3 = 0.0f;
        point_1 = _S1613;
    }
    c_25 = 0U;
    for(;;)
    {
        if(c_25 < 64U)
        {
        }
        else
        {
            break;
        }
        if(!cell_in_0(counted_0, c_25))
        {
            c_25 = c_25 + 1U;
            continue;
        }
        uint k_67 = atomic_load_explicit(kernelContext_213->atom_0+(_S1604 + 2U * c_25 + 1U), memory_order_relaxed);
        bool seen_0 = false;
        total_1 = 0U;
        for(;;)
        {
            if(total_1 < c_25)
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
                    uint _S1616 = atomic_load_explicit(kernelContext_213->atom_0+(_S1604 + 2U * total_1 + 1U), memory_order_relaxed);
                    _S1597 = _S1616 == k_67;
                }
                else
                {
                    _S1597 = false;
                }
                seen_0 = _S1597;
            }
            total_1 = total_1 + 1U;
        }
        if(seen_0)
        {
            c_25 = c_25 + 1U;
            continue;
        }
        uint e_9 = c_25;
        float3 at_24 = _S1613;
        uint n_6 = 0U;
        for(;;)
        {
            if(e_9 < 64U)
            {
            }
            else
            {
                break;
            }
            if(!cell_in_0(counted_0, e_9))
            {
                _S1597 = true;
            }
            else
            {
                uint _S1617 = atomic_load_explicit(kernelContext_213->atom_0+(_S1604 + 2U * e_9 + 1U), memory_order_relaxed);
                _S1597 = _S1617 != k_67;
            }
            if(_S1597)
            {
                e_9 = e_9 + 1U;
                continue;
            }
            thread float rho_4;
            float3 _S1618 = contact_21 + cover_offset_0(e_9, d_29, _S1601, &rho_4);
            uint _S1619 = atomic_load_explicit(kernelContext_213->atom_0+(_S1604 + 2U * e_9), memory_order_relaxed);
            uint _S1620 = n_6 + 1U;
            at_24 = at_24 + (_S1618 + d_29 * float3(unordered_0(_S1619)) );
            n_6 = _S1620;
            e_9 = e_9 + 1U;
        }
        float _S1621 = float(n_6);
        float3 at_25 = at_24 / float3(_S1621) ;
        float _S1622 = strike_load_0(k_67, b_61, d_29 * float3((_S1600.x * _S1621 / float(total_2))) , at_25, _S1600.y, budget_1 - energy_3, kernelContext_213);
        float3 point_2 = point_1 + float3(_S1622)  * at_25;
        energy_3 = energy_3 + _S1622;
        point_1 = point_2;
        c_25 = c_25 + 1U;
    }
    *(kernelContext_213->crush_rw_0+bill_2) = packed_float4(float4(energy_3, float(s_31 + 1U), 0.0f, 0.0f)) ;
    if(energy_3 > 0.0f)
    {
        *(kernelContext_213->crush_rw_0+_S1598) = packed_float4(float4(point_1 / float3(energy_3) , 0.0f)) ;
    }
    return energy_3;
}

[[kernel]] void fr_strike_crush(uint3 id_75 [[thread_position_in_grid]], FractureParams_0 constant* params_81 [[buffer(0)]], uint device* work_81 [[buffer(4)]], packed_float4 device* frames_81 [[buffer(6)]], packed_float4 device* bodies_81 [[buffer(7)]], packed_float4 device* chunks_81 [[buffer(1)]], uint device* sort_81 [[buffer(11)]], atomic<uint> device* atom_81 [[buffer(5)]], Record_natural_0 device* manifolds_81 [[buffer(10)]], packed_float4 device* loads_81 [[buffer(12)]], packed_uint4 device* bonds_82 [[buffer(2)]], uint device* args_81 [[buffer(13)]], uint device* info_81 [[buffer(15)]], ScanParams_0 constant* scan_81 [[buffer(14)]], packed_float4 device* parts_81 [[buffer(3)]], packed_float4 device* colliders_81 [[buffer(8)]], packed_uint4 device* collider_info_81 [[buffer(9)]], packed_float4 device* view_chunks_81 [[buffer(17)]], ViewParams_0 constant* view_81 [[buffer(16)]], packed_float4 device* objects_81 [[buffer(18)]], uint device* shot_bodies_81 [[buffer(20)]], float device* lines_81 [[buffer(19)]], packed_float4 device* geometry_81 [[buffer(21)]], packed_float4 device* shots_81 [[buffer(22)]], packed_float4 device* lp_81 [[buffer(24)]], packed_float4 device* ext_83 [[buffer(23)]], packed_float4 device* engine_contacts_81 [[buffer(25)]], CrushParams_0 constant* crush_81 [[buffer(28)]], packed_float4 device* crush_rw_81 [[buffer(27)]], packed_float4 device* crush_tab_81 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_214;
    (&kernelContext_214)->params_0 = params_81;
    (&kernelContext_214)->work_0 = work_81;
    (&kernelContext_214)->frames_0 = frames_81;
    (&kernelContext_214)->bodies_0 = bodies_81;
    (&kernelContext_214)->chunks_0 = chunks_81;
    (&kernelContext_214)->sort_0 = sort_81;
    (&kernelContext_214)->atom_0 = atom_81;
    (&kernelContext_214)->manifolds_0 = manifolds_81;
    (&kernelContext_214)->loads_0 = loads_81;
    (&kernelContext_214)->bonds_0 = bonds_82;
    (&kernelContext_214)->args_0 = args_81;
    (&kernelContext_214)->info_0 = info_81;
    (&kernelContext_214)->scan_0 = scan_81;
    (&kernelContext_214)->parts_0 = parts_81;
    (&kernelContext_214)->colliders_0 = colliders_81;
    (&kernelContext_214)->collider_info_0 = collider_info_81;
    (&kernelContext_214)->view_chunks_0 = view_chunks_81;
    (&kernelContext_214)->view_0 = view_81;
    (&kernelContext_214)->objects_0 = objects_81;
    (&kernelContext_214)->shot_bodies_0 = shot_bodies_81;
    (&kernelContext_214)->lines_0 = lines_81;
    (&kernelContext_214)->geometry_0 = geometry_81;
    (&kernelContext_214)->shots_0 = shots_81;
    (&kernelContext_214)->lp_0 = lp_81;
    (&kernelContext_214)->ext_0 = ext_83;
    (&kernelContext_214)->engine_contacts_0 = engine_contacts_81;
    (&kernelContext_214)->crush_0 = crush_81;
    (&kernelContext_214)->crush_rw_0 = crush_rw_81;
    (&kernelContext_214)->crush_tab_0 = crush_tab_81;
    threadgroup array<uint, int(1024)> g_scan_81;
    (&kernelContext_214)->g_scan_0 = &g_scan_81;
    threadgroup array<float, int(64)> g_il_best_81;
    (&kernelContext_214)->g_il_best_0 = &g_il_best_81;
    uint s_32 = id_75.x;
    bool _S1623;
    if(s_32 >= (params_81->shot_count_0))
    {
        _S1623 = true;
    }
    else
    {
        _S1623 = ((&kernelContext_214)->crush_0->enabled_0) == 0U;
    }
    if(_S1623)
    {
        return;
    }
    uint sb_4 = 7U * (as_type<uint>(((float4(*((&kernelContext_214)->shots_0+2U * s_32)) ).x)));
    float inv_mass_11 = (float4(*((&kernelContext_214)->bodies_0+sb_4)) ).w;
    float3 v_17 = (float4(*((&kernelContext_214)->bodies_0+(sb_4 + 3U))) ).xyz;
    float budget_2;
    if(inv_mass_11 > 0.0f)
    {
        budget_2 = 0.5f * dot(v_17, v_17) / inv_mass_11;
    }
    else
    {
        budget_2 = 1.00000001504746622e+30f;
    }
    uint b_62 = 0U;
    for(;;)
    {
        if(b_62 < ((&kernelContext_214)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        thread uint shot_12;
        bool _S1624 = struck_0(b_62, &shot_12, &kernelContext_214);
        if(!_S1624)
        {
            _S1623 = true;
        }
        else
        {
            _S1623 = shot_12 != s_32;
        }
        bool _S1625;
        if(_S1623)
        {
            _S1625 = true;
        }
        else
        {
            _S1625 = ((float4(*((&kernelContext_214)->crush_rw_0+((&kernelContext_214)->crush_0->r_bill_0 + 4U * b_62 + 3U))) ).w) == 0.0f;
        }
        if(_S1625)
        {
            b_62 = b_62 + 1U;
            continue;
        }
        float _S1626 = strike_crush_body_0(b_62, s_32, budget_2, &kernelContext_214);
        budget_2 = budget_2 - _S1626;
        b_62 = b_62 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bill(uint3 id_76 [[thread_position_in_grid]], FractureParams_0 constant* params_82 [[buffer(0)]], uint device* work_82 [[buffer(4)]], packed_float4 device* frames_82 [[buffer(6)]], packed_float4 device* bodies_82 [[buffer(7)]], packed_float4 device* chunks_82 [[buffer(1)]], uint device* sort_82 [[buffer(11)]], atomic<uint> device* atom_82 [[buffer(5)]], Record_natural_0 device* manifolds_82 [[buffer(10)]], packed_float4 device* loads_82 [[buffer(12)]], packed_uint4 device* bonds_83 [[buffer(2)]], uint device* args_82 [[buffer(13)]], uint device* info_82 [[buffer(15)]], ScanParams_0 constant* scan_82 [[buffer(14)]], packed_float4 device* parts_82 [[buffer(3)]], packed_float4 device* colliders_82 [[buffer(8)]], packed_uint4 device* collider_info_82 [[buffer(9)]], packed_float4 device* view_chunks_82 [[buffer(17)]], ViewParams_0 constant* view_82 [[buffer(16)]], packed_float4 device* objects_82 [[buffer(18)]], uint device* shot_bodies_82 [[buffer(20)]], float device* lines_82 [[buffer(19)]], packed_float4 device* geometry_82 [[buffer(21)]], packed_float4 device* shots_82 [[buffer(22)]], packed_float4 device* lp_82 [[buffer(24)]], packed_float4 device* ext_84 [[buffer(23)]], packed_float4 device* engine_contacts_82 [[buffer(25)]], CrushParams_0 constant* crush_82 [[buffer(28)]], packed_float4 device* crush_rw_82 [[buffer(27)]], packed_float4 device* crush_tab_82 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_215;
    (&kernelContext_215)->params_0 = params_82;
    (&kernelContext_215)->work_0 = work_82;
    (&kernelContext_215)->frames_0 = frames_82;
    (&kernelContext_215)->bodies_0 = bodies_82;
    (&kernelContext_215)->chunks_0 = chunks_82;
    (&kernelContext_215)->sort_0 = sort_82;
    (&kernelContext_215)->atom_0 = atom_82;
    (&kernelContext_215)->manifolds_0 = manifolds_82;
    (&kernelContext_215)->loads_0 = loads_82;
    (&kernelContext_215)->bonds_0 = bonds_83;
    (&kernelContext_215)->args_0 = args_82;
    (&kernelContext_215)->info_0 = info_82;
    (&kernelContext_215)->scan_0 = scan_82;
    (&kernelContext_215)->parts_0 = parts_82;
    (&kernelContext_215)->colliders_0 = colliders_82;
    (&kernelContext_215)->collider_info_0 = collider_info_82;
    (&kernelContext_215)->view_chunks_0 = view_chunks_82;
    (&kernelContext_215)->view_0 = view_82;
    (&kernelContext_215)->objects_0 = objects_82;
    (&kernelContext_215)->shot_bodies_0 = shot_bodies_82;
    (&kernelContext_215)->lines_0 = lines_82;
    (&kernelContext_215)->geometry_0 = geometry_82;
    (&kernelContext_215)->shots_0 = shots_82;
    (&kernelContext_215)->lp_0 = lp_82;
    (&kernelContext_215)->ext_0 = ext_84;
    (&kernelContext_215)->engine_contacts_0 = engine_contacts_82;
    (&kernelContext_215)->crush_0 = crush_82;
    (&kernelContext_215)->crush_rw_0 = crush_rw_82;
    (&kernelContext_215)->crush_tab_0 = crush_tab_82;
    threadgroup array<uint, int(1024)> g_scan_82;
    (&kernelContext_215)->g_scan_0 = &g_scan_82;
    threadgroup array<float, int(64)> g_il_best_82;
    (&kernelContext_215)->g_il_best_0 = &g_il_best_82;
    uint s_33 = id_76.x;
    bool _S1627;
    if(s_33 >= (params_82->shot_count_0))
    {
        _S1627 = true;
    }
    else
    {
        _S1627 = ((&kernelContext_215)->crush_0->enabled_0) == 0U;
    }
    if(_S1627)
    {
        return;
    }
    uint b_63 = 0U;
    float energy_4 = 0.0f;
    for(;;)
    {
        if(b_63 < ((&kernelContext_215)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S1628 = float4(*((&kernelContext_215)->crush_rw_0+((&kernelContext_215)->crush_0->r_bill_0 + 4U * b_63))) ;
        if((_S1628.y) == float(s_33 + 1U))
        {
            energy_4 = energy_4 + _S1628.x;
        }
        b_63 = b_63 + 1U;
    }
    *((&kernelContext_215)->crush_rw_0+((&kernelContext_215)->crush_0->r_shot_0 + s_33)) = packed_float4(float4(0.0f) ) ;
    if(!(energy_4 > 0.0f))
    {
        return;
    }
    uint sb_5 = 7U * (as_type<uint>(((float4(*((&kernelContext_215)->shots_0+2U * s_33)) ).x)));
    float inv_mass_12 = (float4(*((&kernelContext_215)->bodies_0+sb_5)) ).w;
    if(!(inv_mass_12 > 0.0f))
    {
        return;
    }
    float m_9 = 1.0f / inv_mass_12;
    float3 v_18 = (float4(*((&kernelContext_215)->bodies_0+(sb_5 + 3U))) ).xyz;
    float speed_16 = length(v_18);
    float j_15 = m_9 * (speed_16 - sqrt(max(speed_16 * speed_16 - 2.0f * energy_4 / m_9, 0.0f)));
    bool _S1629 = speed_16 > 0.0f;
    float3 _S1630;
    if(_S1629)
    {
        _S1630 = - v_18 * float3((j_15 / (m_9 * speed_16))) ;
    }
    else
    {
        _S1630 = float3(0.0f) ;
    }
    uint region_6 = 0U;
    for(;;)
    {
        if(region_6 < 2U)
        {
        }
        else
        {
            break;
        }
        uint w_11 = sb_5 + region_6 * 7U * (&kernelContext_215)->params_0->max_bodies_0 + 3U;
        *((&kernelContext_215)->bodies_0+w_11) = packed_float4(float4((float4(*((&kernelContext_215)->bodies_0+w_11)) ).xyz + _S1630, (float4(*((&kernelContext_215)->bodies_0+w_11)) ).w)) ;
        region_6 = region_6 + 1U;
    }
    packed_float4 device* _S1631 = (&kernelContext_215)->crush_rw_0+((&kernelContext_215)->crush_0->r_shot_0 + s_33);
    if(_S1629)
    {
        _S1630 = v_18 * float3((j_15 / speed_16)) ;
    }
    else
    {
        _S1630 = float3(0.0f) ;
    }
    *_S1631 = packed_float4(float4(_S1630, energy_4)) ;
    return;
}

float3 inverse_rotate_0(const Quat_0 thread* q_34, float3 v_19)
{
    thread Quat_0 c_26;
    (&c_26)->w_0 = q_34->w_0;
    (&c_26)->x_0 = - q_34->x_0;
    (&c_26)->y_0 = - q_34->y_0;
    (&c_26)->z_0 = - q_34->z_0;
    thread Quat_0 _S1632 = c_26;
    float3 _S1633 = rotate_0(&_S1632, v_19);
    return _S1633;
}

[[kernel]] void fr_crush_push(uint3 id_77 [[thread_position_in_grid]], FractureParams_0 constant* params_83 [[buffer(0)]], uint device* work_83 [[buffer(4)]], packed_float4 device* frames_83 [[buffer(6)]], packed_float4 device* bodies_83 [[buffer(7)]], packed_float4 device* chunks_83 [[buffer(1)]], uint device* sort_83 [[buffer(11)]], atomic<uint> device* atom_83 [[buffer(5)]], Record_natural_0 device* manifolds_83 [[buffer(10)]], packed_float4 device* loads_83 [[buffer(12)]], packed_uint4 device* bonds_84 [[buffer(2)]], uint device* args_83 [[buffer(13)]], uint device* info_83 [[buffer(15)]], ScanParams_0 constant* scan_83 [[buffer(14)]], packed_float4 device* parts_83 [[buffer(3)]], packed_float4 device* colliders_83 [[buffer(8)]], packed_uint4 device* collider_info_83 [[buffer(9)]], packed_float4 device* view_chunks_83 [[buffer(17)]], ViewParams_0 constant* view_83 [[buffer(16)]], packed_float4 device* objects_83 [[buffer(18)]], uint device* shot_bodies_83 [[buffer(20)]], float device* lines_83 [[buffer(19)]], packed_float4 device* geometry_83 [[buffer(21)]], packed_float4 device* shots_83 [[buffer(22)]], packed_float4 device* lp_83 [[buffer(24)]], packed_float4 device* ext_85 [[buffer(23)]], packed_float4 device* engine_contacts_83 [[buffer(25)]], CrushParams_0 constant* crush_83 [[buffer(28)]], packed_float4 device* crush_rw_83 [[buffer(27)]], packed_float4 device* crush_tab_83 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_216;
    (&kernelContext_216)->params_0 = params_83;
    (&kernelContext_216)->work_0 = work_83;
    (&kernelContext_216)->frames_0 = frames_83;
    (&kernelContext_216)->bodies_0 = bodies_83;
    (&kernelContext_216)->chunks_0 = chunks_83;
    (&kernelContext_216)->sort_0 = sort_83;
    (&kernelContext_216)->atom_0 = atom_83;
    (&kernelContext_216)->manifolds_0 = manifolds_83;
    (&kernelContext_216)->loads_0 = loads_83;
    (&kernelContext_216)->bonds_0 = bonds_84;
    (&kernelContext_216)->args_0 = args_83;
    (&kernelContext_216)->info_0 = info_83;
    (&kernelContext_216)->scan_0 = scan_83;
    (&kernelContext_216)->parts_0 = parts_83;
    (&kernelContext_216)->colliders_0 = colliders_83;
    (&kernelContext_216)->collider_info_0 = collider_info_83;
    (&kernelContext_216)->view_chunks_0 = view_chunks_83;
    (&kernelContext_216)->view_0 = view_83;
    (&kernelContext_216)->objects_0 = objects_83;
    (&kernelContext_216)->shot_bodies_0 = shot_bodies_83;
    (&kernelContext_216)->lines_0 = lines_83;
    (&kernelContext_216)->geometry_0 = geometry_83;
    (&kernelContext_216)->shots_0 = shots_83;
    (&kernelContext_216)->lp_0 = lp_83;
    (&kernelContext_216)->ext_0 = ext_85;
    (&kernelContext_216)->engine_contacts_0 = engine_contacts_83;
    (&kernelContext_216)->crush_0 = crush_83;
    (&kernelContext_216)->crush_rw_0 = crush_rw_83;
    (&kernelContext_216)->crush_tab_0 = crush_tab_83;
    threadgroup array<uint, int(1024)> g_scan_83;
    (&kernelContext_216)->g_scan_0 = &g_scan_83;
    threadgroup array<float, int(64)> g_il_best_83;
    (&kernelContext_216)->g_il_best_0 = &g_il_best_83;
    uint b_64 = id_77.x;
    bool _S1634;
    if(b_64 >= (params_83->body_count_0))
    {
        _S1634 = true;
    }
    else
    {
        _S1634 = ((&kernelContext_216)->params_0->shot_count_0) == 0U;
    }
    if(_S1634)
    {
        _S1634 = true;
    }
    else
    {
        _S1634 = ((&kernelContext_216)->crush_0->enabled_0) == 0U;
    }
    if(_S1634)
    {
        return;
    }
    uint _S1635 = 4U * b_64;
    float4 _S1636 = float4(*((&kernelContext_216)->crush_rw_0+((&kernelContext_216)->crush_0->r_bill_0 + _S1635))) ;
    float _S1637 = _S1636.x;
    if(!(_S1637 > 0.0f))
    {
        return;
    }
    float4 _S1638 = float4(*((&kernelContext_216)->crush_rw_0+((&kernelContext_216)->crush_0->r_shot_0 + (uint(_S1636.y) - 1U)))) ;
    uint bw_0 = 7U * b_64;
    float inv_mass_13 = (float4(*((&kernelContext_216)->bodies_0+bw_0)) ).w;
    float _S1639 = _S1638.w;
    if(!(_S1639 > 0.0f))
    {
        _S1634 = true;
    }
    else
    {
        _S1634 = inv_mass_13 == 0.0f;
    }
    if(_S1634)
    {
        return;
    }
    float3 j_16 = _S1638.xyz * float3((_S1637 / _S1639)) ;
    Quat_0 o_11 = quat_of_0(float4(*((&kernelContext_216)->bodies_0+(bw_0 + 1U))) );
    float3 inv_i_0 = (float4(*((&kernelContext_216)->bodies_0+(bw_0 + 2U))) ).xyz;
    float3 _S1640 = cross(to_rubble_0((float4(*((&kernelContext_216)->crush_rw_0+((&kernelContext_216)->crush_0->r_bill_0 + _S1635 + 1U))) ).xyz) - (float4(*((&kernelContext_216)->bodies_0+bw_0)) ).xyz, j_16);
    thread Quat_0 _S1641 = o_11;
    float3 _S1642 = inverse_rotate_0(&_S1641, _S1640);
    float3 _S1643 = inv_i_0 * _S1642;
    thread Quat_0 _S1644 = o_11;
    float3 _S1645 = rotate_0(&_S1644, _S1643);
    uint region_7 = 0U;
    for(;;)
    {
        if(region_7 < 2U)
        {
        }
        else
        {
            break;
        }
        uint w_12 = bw_0 + region_7 * 7U * (&kernelContext_216)->params_0->max_bodies_0;
        uint _S1646 = w_12 + 3U;
        *((&kernelContext_216)->bodies_0+_S1646) = packed_float4(float4((float4(*((&kernelContext_216)->bodies_0+_S1646)) ).xyz + j_16 * float3(inv_mass_13) , (float4(*((&kernelContext_216)->bodies_0+_S1646)) ).w)) ;
        uint _S1647 = w_12 + 4U;
        *((&kernelContext_216)->bodies_0+_S1647) = packed_float4(float4((float4(*((&kernelContext_216)->bodies_0+_S1647)) ).xyz + _S1645, (float4(*((&kernelContext_216)->bodies_0+_S1647)) ).w)) ;
        region_7 = region_7 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bonds(uint3 id_78 [[thread_position_in_grid]], FractureParams_0 constant* params_84 [[buffer(0)]], uint device* work_84 [[buffer(4)]], packed_float4 device* frames_84 [[buffer(6)]], packed_float4 device* bodies_84 [[buffer(7)]], packed_float4 device* chunks_84 [[buffer(1)]], uint device* sort_84 [[buffer(11)]], atomic<uint> device* atom_84 [[buffer(5)]], Record_natural_0 device* manifolds_84 [[buffer(10)]], packed_float4 device* loads_84 [[buffer(12)]], packed_uint4 device* bonds_85 [[buffer(2)]], uint device* args_84 [[buffer(13)]], uint device* info_84 [[buffer(15)]], ScanParams_0 constant* scan_84 [[buffer(14)]], packed_float4 device* parts_84 [[buffer(3)]], packed_float4 device* colliders_84 [[buffer(8)]], packed_uint4 device* collider_info_84 [[buffer(9)]], packed_float4 device* view_chunks_84 [[buffer(17)]], ViewParams_0 constant* view_84 [[buffer(16)]], packed_float4 device* objects_84 [[buffer(18)]], uint device* shot_bodies_84 [[buffer(20)]], float device* lines_84 [[buffer(19)]], packed_float4 device* geometry_84 [[buffer(21)]], packed_float4 device* shots_84 [[buffer(22)]], packed_float4 device* lp_84 [[buffer(24)]], packed_float4 device* ext_86 [[buffer(23)]], packed_float4 device* engine_contacts_84 [[buffer(25)]], CrushParams_0 constant* crush_84 [[buffer(28)]], packed_float4 device* crush_rw_84 [[buffer(27)]], packed_float4 device* crush_tab_84 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_217;
    (&kernelContext_217)->params_0 = params_84;
    (&kernelContext_217)->work_0 = work_84;
    (&kernelContext_217)->frames_0 = frames_84;
    (&kernelContext_217)->bodies_0 = bodies_84;
    (&kernelContext_217)->chunks_0 = chunks_84;
    (&kernelContext_217)->sort_0 = sort_84;
    (&kernelContext_217)->atom_0 = atom_84;
    (&kernelContext_217)->manifolds_0 = manifolds_84;
    (&kernelContext_217)->loads_0 = loads_84;
    (&kernelContext_217)->bonds_0 = bonds_85;
    (&kernelContext_217)->args_0 = args_84;
    (&kernelContext_217)->info_0 = info_84;
    (&kernelContext_217)->scan_0 = scan_84;
    (&kernelContext_217)->parts_0 = parts_84;
    (&kernelContext_217)->colliders_0 = colliders_84;
    (&kernelContext_217)->collider_info_0 = collider_info_84;
    (&kernelContext_217)->view_chunks_0 = view_chunks_84;
    (&kernelContext_217)->view_0 = view_84;
    (&kernelContext_217)->objects_0 = objects_84;
    (&kernelContext_217)->shot_bodies_0 = shot_bodies_84;
    (&kernelContext_217)->lines_0 = lines_84;
    (&kernelContext_217)->geometry_0 = geometry_84;
    (&kernelContext_217)->shots_0 = shots_84;
    (&kernelContext_217)->lp_0 = lp_84;
    (&kernelContext_217)->ext_0 = ext_86;
    (&kernelContext_217)->engine_contacts_0 = engine_contacts_84;
    (&kernelContext_217)->crush_0 = crush_84;
    (&kernelContext_217)->crush_rw_0 = crush_rw_84;
    (&kernelContext_217)->crush_tab_0 = crush_tab_84;
    threadgroup array<uint, int(1024)> g_scan_84;
    (&kernelContext_217)->g_scan_0 = &g_scan_84;
    threadgroup array<float, int(64)> g_il_best_84;
    (&kernelContext_217)->g_il_best_0 = &g_il_best_84;
    uint i_21 = id_78.x;
    bool _S1648;
    if(i_21 >= (params_84->bond_count_0))
    {
        _S1648 = true;
    }
    else
    {
        _S1648 = ((&kernelContext_217)->crush_0->enabled_0) == 0U;
    }
    if(_S1648)
    {
        _S1648 = true;
    }
    else
    {
        uint _S1649 = w_alive_0(&kernelContext_217);
        _S1648 = (*((&kernelContext_217)->work_0+(_S1649 + i_21))) == 0U;
    }
    if(_S1648)
    {
        return;
    }
    uint4 _S1650 = uint4(*((&kernelContext_217)->bonds_0+i_21)) ;
    uint _S1651 = _S1650.x;
    if((((as_type<uint>(((float4(*((&kernelContext_217)->crush_rw_0+_S1651)) ).w))) | (as_type<uint>(((float4(*((&kernelContext_217)->crush_rw_0+_S1650.y)) ).w)))) & 2U) == 0U)
    {
        return;
    }
    uint _S1652 = w_alive_0(&kernelContext_217);
    *((&kernelContext_217)->work_0+(_S1652 + i_21)) = 0U;
    uint _S1653 = a_dirty_0(&kernelContext_217);
    uint _S1654 = body_of_0(_S1651, &kernelContext_217);
    uint _S1655 = atomic_fetch_or_explicit((&kernelContext_217)->atom_0+(_S1653 + _S1654), 1U, memory_order_relaxed);
    uint _S1656 = a_counters_1(&kernelContext_217);
    uint _S1657 = atomic_fetch_add_explicit((&kernelContext_217)->atom_0+_S1656, 1U, memory_order_relaxed);
    uint _S1658 = atomic_fetch_add_explicit((&kernelContext_217)->atom_0+(_S1656 + 1U), 1U, memory_order_relaxed);
    return;
}

uint crush_pieces_0(uint k_68, KernelContext_0 thread* kernelContext_218)
{
    if((kernelContext_218->crush_0->mode_1) == 1U)
    {
        return kernelContext_218->crush_0->fragments_0;
    }
    if((kernelContext_218->crush_0->mode_1) == 2U)
    {
        return uint((float4(*(kernelContext_218->crush_tab_0+(2U * k_68 + 1U))) ).w + 0.5f);
    }
    return 0U;
}

[[kernel]] void fr_crush_mark(uint3 id_79 [[thread_position_in_grid]], FractureParams_0 constant* params_85 [[buffer(0)]], uint device* work_85 [[buffer(4)]], packed_float4 device* frames_85 [[buffer(6)]], packed_float4 device* bodies_85 [[buffer(7)]], packed_float4 device* chunks_85 [[buffer(1)]], uint device* sort_85 [[buffer(11)]], atomic<uint> device* atom_85 [[buffer(5)]], Record_natural_0 device* manifolds_85 [[buffer(10)]], packed_float4 device* loads_85 [[buffer(12)]], packed_uint4 device* bonds_86 [[buffer(2)]], uint device* args_85 [[buffer(13)]], uint device* info_85 [[buffer(15)]], ScanParams_0 constant* scan_85 [[buffer(14)]], packed_float4 device* parts_85 [[buffer(3)]], packed_float4 device* colliders_85 [[buffer(8)]], packed_uint4 device* collider_info_85 [[buffer(9)]], packed_float4 device* view_chunks_85 [[buffer(17)]], ViewParams_0 constant* view_85 [[buffer(16)]], packed_float4 device* objects_85 [[buffer(18)]], uint device* shot_bodies_85 [[buffer(20)]], float device* lines_85 [[buffer(19)]], packed_float4 device* geometry_85 [[buffer(21)]], packed_float4 device* shots_85 [[buffer(22)]], packed_float4 device* lp_85 [[buffer(24)]], packed_float4 device* ext_87 [[buffer(23)]], packed_float4 device* engine_contacts_85 [[buffer(25)]], CrushParams_0 constant* crush_85 [[buffer(28)]], packed_float4 device* crush_rw_85 [[buffer(27)]], packed_float4 device* crush_tab_85 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_219;
    (&kernelContext_219)->params_0 = params_85;
    (&kernelContext_219)->work_0 = work_85;
    (&kernelContext_219)->frames_0 = frames_85;
    (&kernelContext_219)->bodies_0 = bodies_85;
    (&kernelContext_219)->chunks_0 = chunks_85;
    (&kernelContext_219)->sort_0 = sort_85;
    (&kernelContext_219)->atom_0 = atom_85;
    (&kernelContext_219)->manifolds_0 = manifolds_85;
    (&kernelContext_219)->loads_0 = loads_85;
    (&kernelContext_219)->bonds_0 = bonds_86;
    (&kernelContext_219)->args_0 = args_85;
    (&kernelContext_219)->info_0 = info_85;
    (&kernelContext_219)->scan_0 = scan_85;
    (&kernelContext_219)->parts_0 = parts_85;
    (&kernelContext_219)->colliders_0 = colliders_85;
    (&kernelContext_219)->collider_info_0 = collider_info_85;
    (&kernelContext_219)->view_chunks_0 = view_chunks_85;
    (&kernelContext_219)->view_0 = view_85;
    (&kernelContext_219)->objects_0 = objects_85;
    (&kernelContext_219)->shot_bodies_0 = shot_bodies_85;
    (&kernelContext_219)->lines_0 = lines_85;
    (&kernelContext_219)->geometry_0 = geometry_85;
    (&kernelContext_219)->shots_0 = shots_85;
    (&kernelContext_219)->lp_0 = lp_85;
    (&kernelContext_219)->ext_0 = ext_87;
    (&kernelContext_219)->engine_contacts_0 = engine_contacts_85;
    (&kernelContext_219)->crush_0 = crush_85;
    (&kernelContext_219)->crush_rw_0 = crush_rw_85;
    (&kernelContext_219)->crush_tab_0 = crush_tab_85;
    threadgroup array<uint, int(1024)> g_scan_85;
    (&kernelContext_219)->g_scan_0 = &g_scan_85;
    threadgroup array<float, int(64)> g_il_best_85;
    (&kernelContext_219)->g_il_best_0 = &g_il_best_85;
    uint c_27 = id_79.x;
    bool _S1659;
    if(c_27 >= (params_85->chunk_count_0))
    {
        _S1659 = true;
    }
    else
    {
        _S1659 = ((&kernelContext_219)->crush_0->enabled_0) == 0U;
    }
    if(_S1659)
    {
        return;
    }
    uint _S1660 = crush_flags_0(c_27, &kernelContext_219);
    uint now_0;
    if((_S1660 & 2U) != 0U)
    {
        now_0 = 1U;
    }
    else
    {
        now_0 = 0U;
    }
    uint _S1661 = w_new_0(&kernelContext_219);
    uint device* _S1662 = (&kernelContext_219)->work_0+(_S1661 + c_27);
    bool _S1663 = now_0 != 0U;
    if(_S1663)
    {
        uint _S1664 = crush_pieces_0(c_27, &kernelContext_219);
        now_0 = 1048576U | _S1664;
    }
    else
    {
        now_0 = 0U;
    }
    *_S1662 = now_0;
    if(_S1663)
    {
        uint _S1665 = a_counters_1(&kernelContext_219);
        uint _S1666 = atomic_fetch_add_explicit((&kernelContext_219)->atom_0+(_S1665 + 10U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_crush_gate(uint3 id_80 [[thread_position_in_grid]], FractureParams_0 constant* params_86 [[buffer(0)]], uint device* work_86 [[buffer(4)]], packed_float4 device* frames_86 [[buffer(6)]], packed_float4 device* bodies_86 [[buffer(7)]], packed_float4 device* chunks_86 [[buffer(1)]], uint device* sort_86 [[buffer(11)]], atomic<uint> device* atom_86 [[buffer(5)]], Record_natural_0 device* manifolds_86 [[buffer(10)]], packed_float4 device* loads_86 [[buffer(12)]], packed_uint4 device* bonds_87 [[buffer(2)]], uint device* args_86 [[buffer(13)]], uint device* info_86 [[buffer(15)]], ScanParams_0 constant* scan_86 [[buffer(14)]], packed_float4 device* parts_86 [[buffer(3)]], packed_float4 device* colliders_86 [[buffer(8)]], packed_uint4 device* collider_info_86 [[buffer(9)]], packed_float4 device* view_chunks_86 [[buffer(17)]], ViewParams_0 constant* view_86 [[buffer(16)]], packed_float4 device* objects_86 [[buffer(18)]], uint device* shot_bodies_86 [[buffer(20)]], float device* lines_86 [[buffer(19)]], packed_float4 device* geometry_86 [[buffer(21)]], packed_float4 device* shots_86 [[buffer(22)]], packed_float4 device* lp_86 [[buffer(24)]], packed_float4 device* ext_88 [[buffer(23)]], packed_float4 device* engine_contacts_86 [[buffer(25)]], CrushParams_0 constant* crush_86 [[buffer(28)]], packed_float4 device* crush_rw_86 [[buffer(27)]], packed_float4 device* crush_tab_86 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_220;
    (&kernelContext_220)->params_0 = params_86;
    (&kernelContext_220)->work_0 = work_86;
    (&kernelContext_220)->frames_0 = frames_86;
    (&kernelContext_220)->bodies_0 = bodies_86;
    (&kernelContext_220)->chunks_0 = chunks_86;
    (&kernelContext_220)->sort_0 = sort_86;
    (&kernelContext_220)->atom_0 = atom_86;
    (&kernelContext_220)->manifolds_0 = manifolds_86;
    (&kernelContext_220)->loads_0 = loads_86;
    (&kernelContext_220)->bonds_0 = bonds_87;
    (&kernelContext_220)->args_0 = args_86;
    (&kernelContext_220)->info_0 = info_86;
    (&kernelContext_220)->scan_0 = scan_86;
    (&kernelContext_220)->parts_0 = parts_86;
    (&kernelContext_220)->colliders_0 = colliders_86;
    (&kernelContext_220)->collider_info_0 = collider_info_86;
    (&kernelContext_220)->view_chunks_0 = view_chunks_86;
    (&kernelContext_220)->view_0 = view_86;
    (&kernelContext_220)->objects_0 = objects_86;
    (&kernelContext_220)->shot_bodies_0 = shot_bodies_86;
    (&kernelContext_220)->lines_0 = lines_86;
    (&kernelContext_220)->geometry_0 = geometry_86;
    (&kernelContext_220)->shots_0 = shots_86;
    (&kernelContext_220)->lp_0 = lp_86;
    (&kernelContext_220)->ext_0 = ext_88;
    (&kernelContext_220)->engine_contacts_0 = engine_contacts_86;
    (&kernelContext_220)->crush_0 = crush_86;
    (&kernelContext_220)->crush_rw_0 = crush_rw_86;
    (&kernelContext_220)->crush_tab_0 = crush_tab_86;
    threadgroup array<uint, int(1024)> g_scan_86;
    (&kernelContext_220)->g_scan_0 = &g_scan_86;
    threadgroup array<float, int(64)> g_il_best_86;
    (&kernelContext_220)->g_il_best_0 = &g_il_best_86;
    bool run_2;
    if((crush_86->enabled_0) != 0U)
    {
        uint _S1667 = a_counters_1(&kernelContext_220);
        uint _S1668 = atomic_load_explicit((&kernelContext_220)->atom_0+(_S1667 + 10U), memory_order_relaxed);
        run_2 = _S1668 != 0U;
    }
    else
    {
        run_2 = false;
    }
    uint _S1669 = a_counters_1(&kernelContext_220);
    atomic_store_explicit((&kernelContext_220)->atom_0+(_S1669 + 10U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_220)->atom_0+(_S1669 + 7U), 0U, memory_order_relaxed);
    uint _S1670;
    if(run_2)
    {
        uint _S1671 = groups_of_0((&kernelContext_220)->params_0->chunk_count_0, 64U);
        _S1670 = _S1671;
    }
    else
    {
        _S1670 = 0U;
    }
    set_args_0(14U, _S1670, &kernelContext_220);
    if(run_2)
    {
        uint _S1672 = groups_of_0((&kernelContext_220)->params_0->chunk_count_0, 1024U);
        _S1670 = _S1672;
    }
    else
    {
        _S1670 = 0U;
    }
    set_args_0(15U, _S1670, &kernelContext_220);
    int _S1673;
    if(run_2)
    {
        _S1673 = int(1);
    }
    else
    {
        _S1673 = int(0);
    }
    set_args_0(16U, uint(_S1673), &kernelContext_220);
    return;
}

void write_body_0(uint slot_7, float3 pos_1, float inv_mass_14, const Quat_0 thread* orientation_0, float3 inv_inertia_1, uint flags_3, float3 v_20, float3 w_13, KernelContext_0 thread* kernelContext_221)
{
    uint region_8 = 0U;
    for(;;)
    {
        if(region_8 < 2U)
        {
        }
        else
        {
            break;
        }
        uint b_65 = 7U * (slot_7 + region_8 * kernelContext_221->params_0->max_bodies_0);
        *(kernelContext_221->bodies_0+b_65) = packed_float4(float4(pos_1, inv_mass_14)) ;
        packed_float4 device* _S1674 = kernelContext_221->bodies_0+(b_65 + 1U);
        float4 _S1675 = quat_vec_0(orientation_0);
        *_S1674 = packed_float4(_S1675) ;
        *(kernelContext_221->bodies_0+(b_65 + 2U)) = packed_float4(float4(inv_inertia_1, (as_type<float>((flags_3))))) ;
        *(kernelContext_221->bodies_0+(b_65 + 3U)) = packed_float4(float4(v_20, 0.0f)) ;
        packed_float4 device* _S1676 = kernelContext_221->bodies_0+(b_65 + 4U);
        uint _S1677 = frame_step_0(kernelContext_221);
        *_S1676 = packed_float4(float4(w_13, (as_type<float>((_S1677))))) ;
        *(kernelContext_221->bodies_0+(b_65 + 5U)) = packed_float4(float4(0.0f) ) ;
        *(kernelContext_221->bodies_0+(b_65 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
        region_8 = region_8 + 1U;
    }
    return;
}

float3 crush_particle_spot_0(uint k_69, uint i_22)
{
    uint h_7 = (k_69 * 2654435761U) ^ (i_22 * 2246822519U);
    uint h_8 = (h_7 ^ (h_7 >> 15U)) * 739982445U;
    uint h_9 = (h_8 ^ (h_8 >> 12U)) * 695872825U;
    uint h_10 = h_9 ^ (h_9 >> 15U);
    return float3(float(h_10 & 1023U), float((h_10 >> 10U) & 1023U), float((h_10 * 2654435761U + 2135587861U) & 1023U)) / float3(511.5f)  - float3(1.0f) ;
}

[[kernel]] void fr_crush_apply(uint3 id_81 [[thread_position_in_grid]], FractureParams_0 constant* params_87 [[buffer(0)]], uint device* work_87 [[buffer(4)]], packed_float4 device* frames_87 [[buffer(6)]], packed_float4 device* bodies_87 [[buffer(7)]], packed_float4 device* chunks_87 [[buffer(1)]], uint device* sort_87 [[buffer(11)]], atomic<uint> device* atom_87 [[buffer(5)]], Record_natural_0 device* manifolds_87 [[buffer(10)]], packed_float4 device* loads_87 [[buffer(12)]], packed_uint4 device* bonds_88 [[buffer(2)]], uint device* args_87 [[buffer(13)]], uint device* info_87 [[buffer(15)]], ScanParams_0 constant* scan_87 [[buffer(14)]], packed_float4 device* parts_87 [[buffer(3)]], packed_float4 device* colliders_87 [[buffer(8)]], packed_uint4 device* collider_info_87 [[buffer(9)]], packed_float4 device* view_chunks_87 [[buffer(17)]], ViewParams_0 constant* view_87 [[buffer(16)]], packed_float4 device* objects_87 [[buffer(18)]], uint device* shot_bodies_87 [[buffer(20)]], float device* lines_87 [[buffer(19)]], packed_float4 device* geometry_87 [[buffer(21)]], packed_float4 device* shots_87 [[buffer(22)]], packed_float4 device* lp_87 [[buffer(24)]], packed_float4 device* ext_89 [[buffer(23)]], packed_float4 device* engine_contacts_87 [[buffer(25)]], CrushParams_0 constant* crush_87 [[buffer(28)]], packed_float4 device* crush_rw_87 [[buffer(27)]], packed_float4 device* crush_tab_87 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_222;
    (&kernelContext_222)->params_0 = params_87;
    (&kernelContext_222)->work_0 = work_87;
    (&kernelContext_222)->frames_0 = frames_87;
    (&kernelContext_222)->bodies_0 = bodies_87;
    (&kernelContext_222)->chunks_0 = chunks_87;
    (&kernelContext_222)->sort_0 = sort_87;
    (&kernelContext_222)->atom_0 = atom_87;
    (&kernelContext_222)->manifolds_0 = manifolds_87;
    (&kernelContext_222)->loads_0 = loads_87;
    (&kernelContext_222)->bonds_0 = bonds_88;
    (&kernelContext_222)->args_0 = args_87;
    (&kernelContext_222)->info_0 = info_87;
    (&kernelContext_222)->scan_0 = scan_87;
    (&kernelContext_222)->parts_0 = parts_87;
    (&kernelContext_222)->colliders_0 = colliders_87;
    (&kernelContext_222)->collider_info_0 = collider_info_87;
    (&kernelContext_222)->view_chunks_0 = view_chunks_87;
    (&kernelContext_222)->view_0 = view_87;
    (&kernelContext_222)->objects_0 = objects_87;
    (&kernelContext_222)->shot_bodies_0 = shot_bodies_87;
    (&kernelContext_222)->lines_0 = lines_87;
    (&kernelContext_222)->geometry_0 = geometry_87;
    (&kernelContext_222)->shots_0 = shots_87;
    (&kernelContext_222)->lp_0 = lp_87;
    (&kernelContext_222)->ext_0 = ext_89;
    (&kernelContext_222)->engine_contacts_0 = engine_contacts_87;
    (&kernelContext_222)->crush_0 = crush_87;
    (&kernelContext_222)->crush_rw_0 = crush_rw_87;
    (&kernelContext_222)->crush_tab_0 = crush_tab_87;
    threadgroup array<uint, int(1024)> g_scan_87;
    (&kernelContext_222)->g_scan_0 = &g_scan_87;
    threadgroup array<float, int(64)> g_il_best_87;
    (&kernelContext_222)->g_il_best_0 = &g_il_best_87;
    uint k_70 = id_81.x;
    bool _S1678;
    if(k_70 >= (params_87->chunk_count_0))
    {
        _S1678 = true;
    }
    else
    {
        _S1678 = ((&kernelContext_222)->crush_0->enabled_0) == 0U;
    }
    if(_S1678)
    {
        return;
    }
    float4 _S1679 = float4(*((&kernelContext_222)->crush_rw_0+k_70)) ;
    uint flags_4 = (as_type<uint>((_S1679.w)));
    if((flags_4 & 2U) == 0U)
    {
        return;
    }
    *((&kernelContext_222)->crush_rw_0+k_70) = packed_float4(float4(_S1679.xyz, (as_type<float>((flags_4 & 4294967293U))))) ;
    uint _S1680 = body_of_0(k_70, &kernelContext_222);
    uint bw_1 = 7U * _S1680;
    float4 _S1681 = float4(*((&kernelContext_222)->bodies_0+bw_1)) ;
    Quat_0 orientation_1 = quat_of_0(float4(*((&kernelContext_222)->bodies_0+(bw_1 + 1U))) );
    float3 v_21 = (float4(*((&kernelContext_222)->bodies_0+(bw_1 + 3U))) ).xyz;
    float3 w_14 = (float4(*((&kernelContext_222)->bodies_0+(bw_1 + 4U))) ).xyz;
    uint _S1682 = 4U * k_70;
    float mass_1 = (float4(*((&kernelContext_222)->chunks_0+_S1682)) ).w;
    float volume_1 = (float4(*((&kernelContext_222)->crush_tab_0+(2U * k_70 + 1U))) ).y;
    uint _S1683 = w_rank_0(&kernelContext_222);
    uint device* _S1684 = (&kernelContext_222)->work_0+(_S1683 + k_70);
    uint scanned_0 = *_S1684;
    uint rank_0 = *_S1684 / 1048576U;
    uint _S1685 = crush_pieces_0(k_70, &kernelContext_222);
    if(rank_0 >= ((&kernelContext_222)->crush_0->max_events_0))
    {
        uint _S1686 = a_counters_1(&kernelContext_222);
        uint _S1687 = atomic_fetch_or_explicit((&kernelContext_222)->atom_0+(_S1686 + 3U), 32U, memory_order_relaxed);
    }
    float3 _S1688 = float3(0.0f) ;
    if(((&kernelContext_222)->crush_0->mode_1) == 1U)
    {
        _S1678 = ((&kernelContext_222)->crush_0->fragments_0) > 0U;
    }
    else
    {
        _S1678 = false;
    }
    uint first_6;
    uint n_7;
    uint i_23;
    float3 half_2;
    if(_S1678)
    {
        uint _S1689 = a_counters_1(&kernelContext_222);
        uint _S1690 = atomic_load_explicit((&kernelContext_222)->atom_0+(_S1689 + 8U), memory_order_relaxed);
        uint base_4 = _S1690 + (scanned_0 & 1048575U);
        if((base_4 + (&kernelContext_222)->crush_0->fragments_0) > ((&kernelContext_222)->crush_0->pool_size_0))
        {
            uint _S1691 = atomic_fetch_or_explicit((&kernelContext_222)->atom_0+(_S1689 + 3U), 16U, memory_order_relaxed);
            first_6 = 4294967295U;
        }
        else
        {
            first_6 = base_4;
        }
        thread Quat_0 q_35;
        thread float3 origin_14;
        structure_pose_0(_S1680, bw_1, &q_35, &origin_14, &kernelContext_222);
        float3 _S1692 = (float4(*((&kernelContext_222)->chunks_0+_S1682)) ).xyz;
        thread Quat_0 _S1693 = q_35;
        float3 _S1694 = rotate_0(&_S1693, _S1692);
        float3 _S1695 = origin_14 + _S1694;
        uint _S1696 = 3U * k_70;
        Quat_0 _S1697 = quat_of_0(float4(*((&kernelContext_222)->geometry_0+(_S1696 + 2U))) );
        thread Quat_0 _S1698 = q_35;
        thread Quat_0 _S1699 = _S1697;
        Quat_0 _S1700 = quat_mul_0(&_S1698, &_S1699);
        thread Quat_0 _S1701 = _S1700;
        Quat_0 _S1702 = normalized_0(&_S1701);
        float3 h_11 = (float4(*((&kernelContext_222)->geometry_0+(_S1696 + 1U))) ).xyz;
        thread uint3 g_0 = uint3(int3(int(1)) );
        thread float3 cell_0 = h_11;
        n_7 = (&kernelContext_222)->crush_0->fragments_0;
        for(;;)
        {
            if(n_7 > 1U)
            {
            }
            else
            {
                break;
            }
            if((cell_0.x) >= (cell_0.y))
            {
                _S1678 = (cell_0.x) >= (cell_0.z);
            }
            else
            {
                _S1678 = false;
            }
            if(_S1678)
            {
                i_23 = 0U;
            }
            else
            {
                if((cell_0.y) >= (cell_0.z))
                {
                    i_23 = 1U;
                }
                else
                {
                    i_23 = 2U;
                }
            }
            g_0[i_23] = g_0[i_23] * 2U;
            cell_0[i_23] = cell_0[i_23] * 0.5f;
            n_7 = n_7 >> 1U;
        }
        float box_volume_0 = 8.0f * h_11.x * h_11.y * h_11.z;
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
        float mf_0 = mass_1 / float((&kernelContext_222)->crush_0->fragments_0);
        float _S1703 = half_3.y;
        float _S1704 = _S1703 * _S1703;
        float _S1705 = half_3.z;
        float _S1706 = _S1705 * _S1705;
        float _S1707 = half_3.x;
        float _S1708 = _S1707 * _S1707;
        float3 inertia_0 = float3((mf_0 / 3.0f))  * float3(_S1704 + _S1706, _S1708 + _S1706, _S1708 + _S1704);
        float3 _S1709 = float3(1.0f / inertia_0.x, 1.0f / inertia_0.y, 1.0f / inertia_0.z);
        thread Quat_0 _S1710 = _S1702;
        Quat_0 _S1711 = rot_to_rubble_0(&_S1710);
        float _S1712 = (float4(*((&kernelContext_222)->chunks_0+(_S1682 + 3U))) ).x;
        float _S1713 = min(_S1707, min(_S1703, _S1705));
        i_23 = 0U;
        for(;;)
        {
            if(i_23 < ((&kernelContext_222)->crush_0->fragments_0))
            {
                _S1678 = first_6 != 4294967295U;
            }
            else
            {
                _S1678 = false;
            }
            if(_S1678)
            {
            }
            else
            {
                break;
            }
            uint _S1714 = i_23 % g_0.x;
            uint _S1715 = i_23 / g_0.x;
            uint _S1716 = _S1715 % g_0.y;
            uint _S1717 = i_23 / (g_0.x * g_0.y);
            float3 offset_0 = - h_11 + cell_0 * (float3(2.0f)  * float3(uint3(_S1714, _S1716, _S1717)) + float3(1.0f) );
            thread Quat_0 _S1718 = _S1702;
            float3 _S1719 = rotate_0(&_S1718, offset_0);
            float3 x_8 = to_rubble_0(_S1695 + _S1719);
            float4 _S1720 = float4(*((&kernelContext_222)->crush_tab_0+((&kernelContext_222)->crush_0->t_pool_0 + first_6 + i_23))) ;
            uint fs_0 = (as_type<uint>((_S1720.x)));
            uint fc_3 = (as_type<uint>((_S1720.y)));
            float _S1721 = 1.0f / mf_0;
            float3 _S1722 = v_21 + cross(w_14, x_8 - _S1681.xyz);
            thread Quat_0 _S1723 = _S1711;
            write_body_0(fs_0, x_8, _S1721, &_S1723, _S1709, 9U, _S1722, w_14, &kernelContext_222);
            uint _S1724 = 4U * fc_3;
            uint _S1725 = _S1724 + 1U;
            *((&kernelContext_222)->colliders_0+_S1725) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_222)->colliders_0+_S1725)) ).w)) ;
            *((&kernelContext_222)->colliders_0+(_S1724 + 2U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
            *((&kernelContext_222)->colliders_0+(_S1724 + 3U)) = packed_float4(float4(half_3, length(half_3))) ;
            uint _S1726 = 2U * fc_3;
            *((&kernelContext_222)->collider_info_0+_S1726) = packed_uint4(uint4(4294967295U, fs_0, 2U, (uint4(*((&kernelContext_222)->collider_info_0+_S1726)) ).w)) ;
            *((&kernelContext_222)->collider_info_0+(_S1726 + 1U)) = packed_uint4(uint4((as_type<uint>((mf_0))), (as_type<uint>((_S1712))), (as_type<uint>((_S1713))), 0U)) ;
            i_23 = i_23 + 1U;
        }
        half_2 = half_3;
    }
    else
    {
        first_6 = 4294967295U;
        half_2 = _S1688;
    }
    if(((&kernelContext_222)->crush_0->mode_1) == 2U)
    {
        _S1678 = _S1685 > 0U;
    }
    else
    {
        _S1678 = false;
    }
    if(_S1678)
    {
        uint _S1727 = a_counters_1(&kernelContext_222);
        uint _S1728 = atomic_load_explicit((&kernelContext_222)->atom_0+(_S1727 + 8U), memory_order_relaxed);
        uint base_5 = _S1728 + (scanned_0 & 1048575U);
        if((base_5 + _S1685) > ((&kernelContext_222)->crush_0->pool_size_0))
        {
            uint _S1729 = atomic_fetch_or_explicit((&kernelContext_222)->atom_0+(_S1727 + 3U), 16U, memory_order_relaxed);
        }
        else
        {
            first_6 = base_5;
        }
        thread Quat_0 q_36;
        thread float3 origin_15;
        structure_pose_0(_S1680, bw_1, &q_36, &origin_15, &kernelContext_222);
        float3 _S1730 = (float4(*((&kernelContext_222)->chunks_0+_S1682)) ).xyz;
        thread Quat_0 _S1731 = q_36;
        float3 _S1732 = rotate_0(&_S1731, _S1730);
        float3 _S1733 = origin_15 + _S1732;
        uint _S1734 = 3U * k_70;
        Quat_0 _S1735 = quat_of_0(float4(*((&kernelContext_222)->geometry_0+(_S1734 + 2U))) );
        thread Quat_0 _S1736 = q_36;
        thread Quat_0 _S1737 = _S1735;
        Quat_0 _S1738 = quat_mul_0(&_S1736, &_S1737);
        thread Quat_0 _S1739 = _S1738;
        Quat_0 _S1740 = normalized_0(&_S1739);
        float3 h_12 = (float4(*((&kernelContext_222)->geometry_0+(_S1734 + 1U))) ).xyz;
        float _S1741 = float(_S1685);
        float _S1742 = max(pow(0.75f * volume_1 / (3.14159274101257324f * _S1741), 0.3333333432674408f), 0.0020000000949949f);
        float3 _S1743 = float3(_S1742, _S1742, _S1742);
        float _S1744 = mass_1 / _S1741;
        float _S1745 = (float4(*((&kernelContext_222)->chunks_0+(_S1682 + 3U))) ).x;
        float3 _S1746 = max(h_12 - float3(_S1742) , float3(0.25f)  * h_12);
        i_23 = 0U;
        float3 mean_4 = _S1688;
        for(;;)
        {
            if(i_23 < _S1685)
            {
            }
            else
            {
                break;
            }
            float3 mean_5 = mean_4 + crush_particle_spot_0(k_70, i_23);
            i_23 = i_23 + 1U;
            mean_4 = mean_5;
        }
        float3 mean_6 = mean_4 / float3(_S1741) ;
        i_23 = 0U;
        for(;;)
        {
            if(i_23 < _S1685)
            {
                _S1678 = first_6 != 4294967295U;
            }
            else
            {
                _S1678 = false;
            }
            if(_S1678)
            {
            }
            else
            {
                break;
            }
            float3 _S1747 = _S1746 * (crush_particle_spot_0(k_70, i_23) - mean_6);
            thread Quat_0 _S1748 = _S1740;
            float3 _S1749 = rotate_0(&_S1748, _S1747);
            float3 x_9 = to_rubble_0(_S1733 + _S1749);
            float4 _S1750 = float4(*((&kernelContext_222)->crush_tab_0+((&kernelContext_222)->crush_0->t_pool_0 + first_6 + i_23))) ;
            uint fs_1 = (as_type<uint>((_S1750.x)));
            uint fc_4 = (as_type<uint>((_S1750.y)));
            float _S1751 = 1.0f / _S1744;
            float4 _S1752 = float4(0.0f, 0.0f, 0.0f, 1.0f);
            Quat_0 _S1753 = quat_of_0(_S1752);
            if(((&kernelContext_222)->crush_0->particle_ccd_0) != 0U)
            {
                n_7 = 33024U;
            }
            else
            {
                n_7 = 0U;
            }
            uint _S1754 = 9U | n_7;
            float3 _S1755 = v_21 + cross(w_14, x_9 - _S1681.xyz);
            thread Quat_0 _S1756 = _S1753;
            write_body_0(fs_1, x_9, _S1751, &_S1756, _S1688, _S1754, _S1755, _S1688, &kernelContext_222);
            uint _S1757 = 4U * fc_4;
            uint _S1758 = _S1757 + 1U;
            *((&kernelContext_222)->colliders_0+_S1758) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_222)->colliders_0+_S1758)) ).w)) ;
            *((&kernelContext_222)->colliders_0+(_S1757 + 2U)) = packed_float4(_S1752) ;
            *((&kernelContext_222)->colliders_0+(_S1757 + 3U)) = packed_float4(float4(_S1742, 0.0f, 0.0f, _S1742)) ;
            uint _S1759 = 2U * fc_4;
            *((&kernelContext_222)->collider_info_0+_S1759) = packed_uint4(uint4(4294967295U, fs_1, 2U, (uint4(*((&kernelContext_222)->collider_info_0+_S1759)) ).w)) ;
            *((&kernelContext_222)->collider_info_0+(_S1759 + 1U)) = packed_uint4(uint4((as_type<uint>((_S1744))), (as_type<uint>((_S1745))), (as_type<uint>((_S1742))), 0U)) ;
            i_23 = i_23 + 1U;
        }
        half_2 = _S1743;
    }
    if(rank_0 < ((&kernelContext_222)->crush_0->max_events_0))
    {
        uint ev_0 = (&kernelContext_222)->crush_0->r_events_0 + 6U * rank_0;
        *((&kernelContext_222)->crush_rw_0+ev_0) = packed_float4(float4((as_type<float>((k_70))), (as_type<float>((first_6))), (as_type<float>((flags_4 >> 4U))), _S1679.x)) ;
        *((&kernelContext_222)->crush_rw_0+(ev_0 + 1U)) = packed_float4(float4(_S1681.xyz, mass_1)) ;
        packed_float4 device* _S1760 = (&kernelContext_222)->crush_rw_0+(ev_0 + 2U);
        thread Quat_0 _S1761 = orientation_1;
        float4 _S1762 = quat_vec_0(&_S1761);
        *_S1760 = packed_float4(_S1762) ;
        *((&kernelContext_222)->crush_rw_0+(ev_0 + 3U)) = packed_float4(float4(v_21, _S1679.y)) ;
        *((&kernelContext_222)->crush_rw_0+(ev_0 + 4U)) = packed_float4(float4(w_14, _S1679.z)) ;
        *((&kernelContext_222)->crush_rw_0+(ev_0 + 5U)) = packed_float4(float4(half_2, volume_1)) ;
    }
    float3 park_0 = float3(float(_S1680 % 1024U) * (&kernelContext_222)->crush_0->park_step_0, (&kernelContext_222)->crush_0->park_y_0, float(_S1680 / 1024U) * (&kernelContext_222)->crush_0->park_step_0);
    thread Quat_0 _S1763 = orientation_1;
    write_body_0(_S1680, park_0, 0.0f, &_S1763, _S1688, 0U, _S1688, _S1688, &kernelContext_222);
    return;
}

[[kernel]] void fr_crush_end(uint3 id_82 [[thread_position_in_grid]], FractureParams_0 constant* params_88 [[buffer(0)]], uint device* work_88 [[buffer(4)]], packed_float4 device* frames_88 [[buffer(6)]], packed_float4 device* bodies_88 [[buffer(7)]], packed_float4 device* chunks_88 [[buffer(1)]], uint device* sort_88 [[buffer(11)]], atomic<uint> device* atom_88 [[buffer(5)]], Record_natural_0 device* manifolds_88 [[buffer(10)]], packed_float4 device* loads_88 [[buffer(12)]], packed_uint4 device* bonds_89 [[buffer(2)]], uint device* args_88 [[buffer(13)]], uint device* info_88 [[buffer(15)]], ScanParams_0 constant* scan_88 [[buffer(14)]], packed_float4 device* parts_88 [[buffer(3)]], packed_float4 device* colliders_88 [[buffer(8)]], packed_uint4 device* collider_info_88 [[buffer(9)]], packed_float4 device* view_chunks_88 [[buffer(17)]], ViewParams_0 constant* view_88 [[buffer(16)]], packed_float4 device* objects_88 [[buffer(18)]], uint device* shot_bodies_88 [[buffer(20)]], float device* lines_88 [[buffer(19)]], packed_float4 device* geometry_88 [[buffer(21)]], packed_float4 device* shots_88 [[buffer(22)]], packed_float4 device* lp_88 [[buffer(24)]], packed_float4 device* ext_90 [[buffer(23)]], packed_float4 device* engine_contacts_88 [[buffer(25)]], CrushParams_0 constant* crush_88 [[buffer(28)]], packed_float4 device* crush_rw_88 [[buffer(27)]], packed_float4 device* crush_tab_88 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_223;
    (&kernelContext_223)->params_0 = params_88;
    (&kernelContext_223)->work_0 = work_88;
    (&kernelContext_223)->frames_0 = frames_88;
    (&kernelContext_223)->bodies_0 = bodies_88;
    (&kernelContext_223)->chunks_0 = chunks_88;
    (&kernelContext_223)->sort_0 = sort_88;
    (&kernelContext_223)->atom_0 = atom_88;
    (&kernelContext_223)->manifolds_0 = manifolds_88;
    (&kernelContext_223)->loads_0 = loads_88;
    (&kernelContext_223)->bonds_0 = bonds_89;
    (&kernelContext_223)->args_0 = args_88;
    (&kernelContext_223)->info_0 = info_88;
    (&kernelContext_223)->scan_0 = scan_88;
    (&kernelContext_223)->parts_0 = parts_88;
    (&kernelContext_223)->colliders_0 = colliders_88;
    (&kernelContext_223)->collider_info_0 = collider_info_88;
    (&kernelContext_223)->view_chunks_0 = view_chunks_88;
    (&kernelContext_223)->view_0 = view_88;
    (&kernelContext_223)->objects_0 = objects_88;
    (&kernelContext_223)->shot_bodies_0 = shot_bodies_88;
    (&kernelContext_223)->lines_0 = lines_88;
    (&kernelContext_223)->geometry_0 = geometry_88;
    (&kernelContext_223)->shots_0 = shots_88;
    (&kernelContext_223)->lp_0 = lp_88;
    (&kernelContext_223)->ext_0 = ext_90;
    (&kernelContext_223)->engine_contacts_0 = engine_contacts_88;
    (&kernelContext_223)->crush_0 = crush_88;
    (&kernelContext_223)->crush_rw_0 = crush_rw_88;
    (&kernelContext_223)->crush_tab_0 = crush_tab_88;
    threadgroup array<uint, int(1024)> g_scan_88;
    (&kernelContext_223)->g_scan_0 = &g_scan_88;
    threadgroup array<float, int(64)> g_il_best_88;
    (&kernelContext_223)->g_il_best_0 = &g_il_best_88;
    uint blocks_2 = (params_88->chunk_count_0 + 1023U) / 1024U;
    uint _S1764 = scan_new_blocks_0(&kernelContext_223);
    uint device* _S1765 = (&kernelContext_223)->work_0+(_S1764 + blocks_2);
    uint total_3 = *_S1765;
    uint events_0 = *_S1765 / 1048576U;
    uint _S1766 = a_counters_1(&kernelContext_223);
    atomic_store_explicit((&kernelContext_223)->atom_0+(_S1766 + 7U), events_0, memory_order_relaxed);
    uint _S1767 = atomic_fetch_add_explicit((&kernelContext_223)->atom_0+(_S1766 + 9U), events_0, memory_order_relaxed);
    uint _S1768 = atomic_fetch_add_explicit((&kernelContext_223)->atom_0+(_S1766 + 8U), total_3 & 1048575U, memory_order_relaxed);
    return;
}

