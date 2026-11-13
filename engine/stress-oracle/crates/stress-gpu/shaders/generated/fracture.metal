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
    uint chunk_used_0;
    uint bond_used_0;
    uint ev_broken_cap_0;
    uint ev_moved_cap_0;
    uint ev_frames_cap_0;
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
    float pad3_0;
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
    uint r_virial_0;
    uint bond_virial_0;
    float pad1_1;
};

struct KernelContext_0
{
    FractureParams_0 constant* params_0;
    atomic<uint> device* atom_0;
    uint device* work_0;
    packed_float4 device* frames_0;
    packed_float4 device* bodies_0;
    packed_float4 device* chunks_0;
    uint device* sort_0;
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
    return _S6 + 24U;
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

uint a_strike_dist_0(KernelContext_0 thread* kernelContext_56)
{
    uint _S198 = a_cs_0(0U, kernelContext_56);
    return _S198 + 12U * kernelContext_56->params_0->body_count_0;
}

uint a_events_0(KernelContext_0 thread* kernelContext_57)
{
    uint _S199 = a_strike_dist_0(kernelContext_57);
    return _S199 + kernelContext_57->params_0->chunk_count_0;
}

void event_broken_0(uint bond_0, KernelContext_0 thread* kernelContext_58)
{
    if((kernelContext_58->params_0->ev_broken_cap_0) == 0U)
    {
        return;
    }
    uint _S200 = a_counters_1(kernelContext_58);
    uint i_0 = atomic_fetch_add_explicit(kernelContext_58->atom_0+(_S200 + 20U), 1U, memory_order_relaxed);
    if(i_0 < (kernelContext_58->params_0->ev_broken_cap_0))
    {
        uint _S201 = a_events_0(kernelContext_58);
        atomic_store_explicit(kernelContext_58->atom_0+(_S201 + i_0), bond_0, memory_order_relaxed);
    }
    else
    {
        uint _S202 = atomic_fetch_or_explicit(kernelContext_58->atom_0+(_S200 + 3U), 128U, memory_order_relaxed);
    }
    return;
}

void il_breaks_0(uint i_1, KernelContext_0 thread* kernelContext_59)
{
    bool cut_0;
    if(i_1 >= (kernelContext_59->params_0->bond_count_0))
    {
        cut_0 = true;
    }
    else
    {
        uint _S203 = w_alive_0(kernelContext_59);
        cut_0 = (*(kernelContext_59->work_0+(_S203 + i_1))) == 0U;
    }
    if(cut_0)
    {
        return;
    }
    uint4 _S204 = uint4(*(kernelContext_59->bonds_0+i_1)) ;
    uint _S205 = _S204.x;
    uint _S206 = body_of_0(_S205, kernelContext_59);
    bool _S207 = il_struck_0(_S206, kernelContext_59);
    if(!_S207)
    {
        return;
    }
    uint _S208 = lp_body_0(_S206, kernelContext_59);
    float4 _S209 = float4(*(kernelContext_59->lp_0+_S208)) ;
    float4 _S210 = float4(*(kernelContext_59->lp_0+(_S208 + 1U))) ;
    uint outcome_3 = (as_type<uint>((_S209.w)));
    if((outcome_3 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_5;
    thread float3 d_7;
    thread float speed_4;
    il_path_0(_S206, &contact_5, &d_7, &speed_4, kernelContext_59);
    uint _S211 = il_entry_0(_S206, kernelContext_59);
    IlStriker_0 _S212 = il_striker_0(_S206, kernelContext_59);
    float _S213 = max(_S212.radius_0, 0.00100000004749745f);
    float h_3 = _S209.x;
    IlPlace_0 _S214 = il_place_0(_S205, _S211, contact_5, d_7, _S213, h_3, kernelContext_59);
    IlPlace_0 _S215 = il_place_0(_S204.y, _S211, contact_5, d_7, _S213, h_3, kernelContext_59);
    thread IlPlace_0 _S216 = _S214;
    bool _S217 = il_plug_0(outcome_3, _S213, h_3, &_S216);
    thread IlPlace_0 _S218 = _S215;
    bool _S219 = il_plug_0(outcome_3, _S213, h_3, &_S218);
    if(_S217 != _S219)
    {
        cut_0 = true;
    }
    else
    {
        float _S220 = _S210.y;
        thread IlPlace_0 _S221 = _S214;
        bool _S222 = il_scab_0(outcome_3, _S213, h_3, _S220, &_S221);
        thread IlPlace_0 _S223 = _S215;
        bool _S224 = il_scab_0(outcome_3, _S213, h_3, _S220, &_S223);
        cut_0 = _S222 != _S224;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S225 = _S210.x;
        float _S226 = _S210.z;
        thread IlPlace_0 _S227 = _S214;
        bool _S228 = il_crater_0(outcome_3, _S213, _S225, _S226, &_S227);
        cut_0 = _S228;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S229 = _S210.x;
        float _S230 = _S210.z;
        thread IlPlace_0 _S231 = _S215;
        bool _S232 = il_crater_0(outcome_3, _S213, _S229, _S230, &_S231);
        cut_0 = _S232;
    }
    if(cut_0)
    {
        uint _S233 = w_alive_0(kernelContext_59);
        *(kernelContext_59->work_0+(_S233 + i_1)) = 0U;
        uint _S234 = a_dirty_0(kernelContext_59);
        uint _S235 = atomic_fetch_or_explicit(kernelContext_59->atom_0+(_S234 + _S206), 1U, memory_order_relaxed);
        uint _S236 = a_counters_1(kernelContext_59);
        uint _S237 = atomic_fetch_add_explicit(kernelContext_59->atom_0+_S236, 1U, memory_order_relaxed);
        uint _S238 = atomic_fetch_add_explicit(kernelContext_59->atom_0+(_S236 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_1, kernelContext_59);
        uint _S239 = atomic_fetch_add_explicit(kernelContext_59->atom_0+(_S236 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

float3 to_rubble_0(float3 v_7)
{
    return float3(v_7.x, v_7.z, - v_7.y);
}

float read64_0(uint at_1, float scale_0, KernelContext_0 thread* kernelContext_60)
{
    uint _S240 = atomic_load_explicit(kernelContext_60->atom_0+(at_1 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S240)));
    uint lo_1 = atomic_load_explicit(kernelContext_60->atom_0+at_1, memory_order_relaxed);
    return (float(hi_1) * 4.294967296e+09f + float(lo_1)) / scale_0;
}

void il_carry_0(uint k_12, KernelContext_0 thread* kernelContext_61)
{
    if(k_12 >= (kernelContext_61->params_0->chunk_count_0))
    {
        return;
    }
    uint _S241 = w_carry_0(kernelContext_61);
    uint _S242 = _S241 + k_12;
    uint device* _S243 = kernelContext_61->work_0+_S242;
    uint c_2 = *_S243;
    if((*_S243) == 0U)
    {
        return;
    }
    *(kernelContext_61->work_0+_S242) = 0U;
    uint b0_0 = c_2 - 1U;
    bool _S244 = il_struck_0(b0_0, kernelContext_61);
    if(!_S244)
    {
        return;
    }
    thread float3 contact_6;
    thread float3 d_8;
    thread float speed_5;
    il_path_0(b0_0, &contact_6, &d_8, &speed_5, kernelContext_61);
    float3 dr_0 = to_rubble_0(d_8);
    IlStriker_0 _S245 = il_striker_0(b0_0, kernelContext_61);
    uint _S246 = a_il_0(b0_0, kernelContext_61);
    float _S247 = read64_0(_S246 + 2U, kernelContext_61->params_0->scale_m_0, kernelContext_61);
    float3 v_wall_0 = (float4(*(kernelContext_61->bodies_0+(7U * b0_0 + 3U))) ).xyz;
    uint _S248 = 7U * _S245.body_0;
    float3 v_shot_0 = (float4(*(kernelContext_61->bodies_0+(_S248 + 3U))) ).xyz;
    float closing_0 = dot(v_shot_0 - v_wall_0, dr_0);
    float v_r_0 = _S245.mass_0 * closing_0 / (_S245.mass_0 + _S247);
    float3 _S249 = v_shot_0 + dr_0 * float3((v_r_0 - closing_0)) ;
    float3 _S250 = v_wall_0 + dr_0 * float3(v_r_0) ;
    float _S251 = il_t_in_0(b0_0, kernelContext_61);
    float3 _S252 = (float4(*(kernelContext_61->bodies_0+_S248)) ).xyz + dr_0 * float3(((speed_5 - v_r_0) * clamp(_S251, 0.0f, kernelContext_61->params_0->dt_0))) ;
    uint _S253 = body_of_0(k_12, kernelContext_61);
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
        uint s_at_0 = 7U * (_S245.body_0 + region_0 * kernelContext_61->params_0->max_bodies_0);
        (*(kernelContext_61->bodies_0+s_at_0)).xyz = _S252;
        (*(kernelContext_61->bodies_0+(s_at_0 + 3U))).xyz = _S249;
        (*(kernelContext_61->bodies_0+(7U * (_S253 + region_0 * kernelContext_61->params_0->max_bodies_0) + 3U))).xyz = _S250;
        region_0 = region_0 + 1U;
    }
    return;
}

void add64_0(uint at_2, float value_0, float scale_1, KernelContext_0 thread* kernelContext_62)
{
    float x_1 = round(value_0 * scale_1);
    float x_2;
    if(!((abs(x_1)) < 1.073741824e+09f))
    {
        uint _S254 = a_counters_1(kernelContext_62);
        uint _S255 = atomic_fetch_or_explicit(kernelContext_62->atom_0+(_S254 + 3U), 2U, memory_order_relaxed);
        x_2 = 0.0f;
    }
    else
    {
        x_2 = x_1;
    }
    int v_8 = int(x_2);
    uint lo_2 = (as_type<uint>((v_8)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_62->atom_0+at_2, lo_2, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_2) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S256;
    if(v_8 < int(0))
    {
        _S256 = 4294967295U;
    }
    else
    {
        _S256 = 0U;
    }
    uint hi_2 = _S256 + carry_0;
    if(hi_2 != 0U)
    {
        uint _S257 = atomic_fetch_add_explicit(kernelContext_62->atom_0+(at_2 + 1U), hi_2, memory_order_relaxed);
    }
    return;
}

void il_mark_0(uint k_13, KernelContext_0 thread* kernelContext_63)
{
    if(k_13 >= (kernelContext_63->params_0->chunk_count_0))
    {
        return;
    }
    uint _S258 = body_of_0(k_13, kernelContext_63);
    bool _S259 = il_struck_0(_S258, kernelContext_63);
    if(!_S259)
    {
        return;
    }
    uint _S260 = lp_body_0(_S258, kernelContext_63);
    float4 _S261 = float4(*(kernelContext_63->lp_0+_S260)) ;
    float4 _S262 = float4(*(kernelContext_63->lp_0+(_S260 + 1U))) ;
    uint outcome_4 = (as_type<uint>((_S261.w)));
    if((outcome_4 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_7;
    thread float3 d_9;
    thread float speed_6;
    il_path_0(_S258, &contact_7, &d_9, &speed_6, kernelContext_63);
    uint _S263 = il_entry_0(_S258, kernelContext_63);
    IlStriker_0 _S264 = il_striker_0(_S258, kernelContext_63);
    float _S265 = max(_S264.radius_0, 0.00100000004749745f);
    float _S266 = _S261.x;
    IlPlace_0 _S267 = il_place_0(k_13, _S263, contact_7, d_9, _S265, _S266, kernelContext_63);
    thread IlPlace_0 _S268 = _S267;
    bool _S269 = il_plug_0(outcome_4, _S265, _S266, &_S268);
    if((_S264.launch_0) != 0U)
    {
        float2 _S270 = il_interval_0(k_13, contact_7 - d_9 * float3(_S264.bound_0) , d_9, _S264.bound_0, kernelContext_63);
        float _S271 = _S270.x;
        float _S272 = _S270.y;
        bool path_0;
        if(_S271 <= _S272)
        {
            path_0 = _S271 <= (_S266 + 2.0f * _S264.bound_0);
        }
        else
        {
            path_0 = false;
        }
        if(path_0)
        {
            path_0 = _S272 >= 0.0f;
        }
        else
        {
            path_0 = false;
        }
        if(_S269)
        {
            path_0 = true;
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            float _S273 = _S262.y;
            thread IlPlace_0 _S274 = _S267;
            bool _S275 = il_scab_0(outcome_4, _S265, _S266, _S273, &_S274);
            path_0 = _S275;
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            float _S276 = _S262.x;
            float _S277 = _S262.z;
            thread IlPlace_0 _S278 = _S267;
            bool _S279 = il_crater_0(outcome_4, _S265, _S276, _S277, &_S278);
            path_0 = _S279;
        }
        if(path_0)
        {
            uint _S280 = w_marks_0(kernelContext_63);
            *(kernelContext_63->work_0+(_S280 + k_13)) = _S264.launch_0;
        }
    }
    if(_S269)
    {
        uint _S281 = w_carry_0(kernelContext_63);
        *(kernelContext_63->work_0+(_S281 + k_13)) = _S258 + 1U;
        uint _S282 = a_il_0(_S258, kernelContext_63);
        add64_0(_S282 + 2U, (float4(*(kernelContext_63->chunks_0+4U * k_13)) ).w, kernelContext_63->params_0->scale_m_0, kernelContext_63);
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
    float4 _S283 = mat_0->dif_0;
    float ref_0 = mat_0->dif_0.x;
    if(r_16 <= ref_0)
    {
        return 1.0f;
    }
    float _S284 = _S283.z;
    float f_0;
    if(r_16 <= _S284)
    {
        float _S285 = (metal::fast::divide((r_16), (ref_0)));
        float _S286 = (metal::fast::pow((_S285), (_S283.y)));
        f_0 = _S286;
    }
    else
    {
        float _S287 = (metal::fast::divide((_S284), (ref_0)));
        float _S288 = (metal::fast::pow((_S287), (_S283.y)));
        float _S289 = (metal::fast::divide((r_16), (_S284)));
        float _S290 = (metal::fast::pow((_S289), (_S283.w)));
        f_0 = _S288 * _S290;
    }
    return clamp(f_0, 1.0f, mat_0->misc_0.x);
}

void il_impulse_0(uint b_20, uint sb_0, float m_shot_0, float3 contact_8, float3 d_10, float j_0, KernelContext_0 thread* kernelContext_64)
{
    float3 jr_0 = to_rubble_0(d_10) * float3(j_0) ;
    uint at_3 = 7U * b_20;
    float inv_mass_1 = (float4(*(kernelContext_64->bodies_0+at_3)) ).w;
    float3 _S291 = jr_0 * float3(inv_mass_1) ;
    float3 _S292 = float3(0.0f) ;
    float3 dw_0;
    if(inv_mass_1 > 0.0f)
    {
        Quat_0 q_10 = quat_of_0(float4(*(kernelContext_64->bodies_0+(at_3 + 1U))) );
        float3 r_17 = to_rubble_0(contact_8) - (float4(*(kernelContext_64->bodies_0+at_3)) ).xyz;
        float3 _S293 = (float4(*(kernelContext_64->bodies_0+(at_3 + 2U))) ).xyz;
        thread Quat_0 _S294 = q_10;
        Quat_0 _S295 = conj_0(&_S294);
        float3 _S296 = cross(r_17, jr_0);
        thread Quat_0 _S297 = _S295;
        float3 _S298 = rotate_0(&_S297, _S296);
        float3 _S299 = _S293 * _S298;
        thread Quat_0 _S300 = q_10;
        float3 _S301 = rotate_0(&_S300, _S299);
        dw_0 = _S301;
    }
    else
    {
        dw_0 = _S292;
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
        uint s_at_1 = 7U * (sb_0 + region_1 * kernelContext_64->params_0->max_bodies_0) + 3U;
        (*(kernelContext_64->bodies_0+s_at_1)).xyz = (float4(*(kernelContext_64->bodies_0+s_at_1)) ).xyz - jr_0 / float3(m_shot_0) ;
        uint w_at_0 = 7U * (b_20 + region_1 * kernelContext_64->params_0->max_bodies_0);
        uint _S302 = w_at_0 + 3U;
        (*(kernelContext_64->bodies_0+_S302)).xyz = (float4(*(kernelContext_64->bodies_0+_S302)) ).xyz + _S291;
        uint _S303 = w_at_0 + 4U;
        (*(kernelContext_64->bodies_0+_S303)).xyz = (float4(*(kernelContext_64->bodies_0+_S303)) ).xyz + dw_0;
        region_1 = region_1 + 1U;
    }
    return;
}

void il_strike_0(uint b_21, KernelContext_0 thread* kernelContext_65)
{
    bool damage_0;
    if(b_21 >= (kernelContext_65->params_0->body_count_0))
    {
        damage_0 = true;
    }
    else
    {
        bool _S304 = il_struck_0(b_21, kernelContext_65);
        damage_0 = !_S304;
    }
    if(damage_0)
    {
        return;
    }
    float _S305 = il_t_in_0(b_21, kernelContext_65);
    uint _S306 = il_entry_0(b_21, kernelContext_65);
    thread float3 contact_9;
    thread float3 d_11;
    thread float speed_7;
    il_path_0(b_21, &contact_9, &d_11, &speed_7, kernelContext_65);
    IlStriker_0 _S307 = il_striker_0(b_21, kernelContext_65);
    float _S308 = max(_S307.radius_0, 0.00100000004749745f);
    if((_S307.launch_0) != 0U)
    {
        uint _S309 = w_marks_0(kernelContext_65);
        damage_0 = (*(kernelContext_65->work_0+(_S309 + _S306))) == (_S307.launch_0);
    }
    else
    {
        damage_0 = false;
    }
    if(damage_0)
    {
        uint _S310 = lp_body_0(b_21, kernelContext_65);
        packed_float4 _S311 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_65->lp_0+_S310) = _S311;
        *(kernelContext_65->lp_0+(_S310 + 1U)) = _S311;
        return;
    }
    uint _S312 = a_il_0(b_21, kernelContext_65);
    uint _S313 = atomic_load_explicit(kernelContext_65->atom_0+_S312, memory_order_relaxed);
    float _S314 = max((as_type<float>((_S313))) - _S308, 9.99999997475242708e-07f);
    uint o_2 = kernelContext_65->params_0->ext_chunks_0 + 4U * _S306;
    float4 _S315 = float4(*(kernelContext_65->ext_0+(o_2 + 1U))) ;
    thread JointMaterial_0 dm_0;
    float4 _S316 = float4(0.0f) ;
    (&dm_0)->strength_0 = _S316;
    (&dm_0)->energy_0 = _S316;
    (&dm_0)->dif_0 = float4(*(kernelContext_65->ext_0+(o_2 + 2U))) ;
    float4 _S317 = float4(*(kernelContext_65->ext_0+(o_2 + 3U))) ;
    float _S318 = _S317.x;
    (&dm_0)->misc_0 = float4(_S318, 0.0f, 0.0f, 0.0f);
    (&dm_0)->kind_flags_0 = uint4(0U) ;
    float nu_0 = _S317.y;
    float e_target_0 = _S317.z / (1.0f - nu_0 * nu_0);
    float e_star_0 = 1.0f / (1.0f / _S307.modulus_0 + 1.0f / e_target_0);
    float _S319 = max(speed_7, 0.00100000004749745f);
    float _S320 = sqrt(_S308);
    float h0_0 = pow(15.0f * _S307.m_eff_0 * _S319 * _S319 / (16.0f * e_star_0 * _S320), 0.40000000596046448f);
    float t_c_0 = 2.94000005722045898f * h0_0 / _S319;
    float force_0 = 1.33333337306976318f * e_star_0 * _S320 * pow(h0_0, 1.5f);
    float a_3 = sqrt(_S308 * h0_0);
    float p0_0 = 3.0f * force_0 / (6.28318548202514648f * a_3 * a_3);
    float ring_0 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    float k_14;
    if(_S318 > 0.0f)
    {
        float _S321 = _S319 / _S314;
        thread JointMaterial_0 _S322 = dm_0;
        float _S323 = dif_factor_0(&_S322, _S321);
        k_14 = _S323;
    }
    else
    {
        k_14 = 1.0f;
    }
    float ft_0 = k_14 * _S315.y;
    float fc_0 = k_14 * _S315.z;
    float tau_0 = k_14 * _S315.w;
    if(p0_0 > fc_0)
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ring_0 > ft_0;
    }
    float _S324 = fc_0 * (3.14159274101257324f * _S308 * _S308);
    float _S325 = min(_S324 * _S314, 1.57079637050628662f * (2.0f * _S308 + _S314) * _S314 * _S314 * tau_0);
    float _S326 = 2.0f * _S325;
    float v_bl_0 = sqrt(_S326 / _S307.m_eff_0);
    bool perforate_0;
    if(damage_0)
    {
        perforate_0 = _S319 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    float r2_0 = _S308 + _S314;
    float _S327 = _S315.x;
    float plug_0 = _S327 * 3.14159274101257324f * _S314 / 3.0f * (_S308 * _S308 + _S308 * r2_0 + r2_0 * r2_0);
    float v_res_0;
    if(perforate_0)
    {
        v_res_0 = _S307.mass_0 / (_S307.mass_0 + plug_0) * sqrt(max(_S319 * _S319 - v_bl_0 * v_bl_0, 0.0f));
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
        depth_1 = min(0.5f * _S307.m_eff_0 * _S319 * _S319 / _S324, _S314);
    }
    else
    {
        depth_1 = 0.0f;
    }
    bool crater_0 = depth_1 > a_3;
    float c_3 = sqrt(e_target_0 / _S327);
    float z_t_0 = _S327 * c_3;
    float rho_p_0 = _S307.mass_0 / (4.18879032135009766f * _S308 * _S308 * _S308);
    float z_p_0 = rho_p_0 * sqrt(_S307.modulus_0 / rho_p_0);
    float mean_pressure_0 = force_0 / (3.14159274101257324f * a_3 * a_3);
    float _S328 = min(min(z_t_0 * _S319 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_0);
    float source_radius_0;
    if(mean_pressure_0 > fc_0)
    {
        source_radius_0 = max(a_3, _S308);
    }
    else
    {
        source_radius_0 = a_3;
    }
    float pulse_0 = c_3 * t_c_0;
    float back_0 = _S328 * min(source_radius_0 / _S314, 1.0f) * min(2.0f * _S314 / pulse_0, 1.0f);
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
        scab_1 = 0.5f * min(pulse_0, _S314);
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
    uint _S329;
    if(crater_0)
    {
        _S329 = 2U;
    }
    else
    {
        _S329 = 0U;
    }
    uint _S330 = region_2 | _S329;
    if(spall_0)
    {
        region_2 = 4U;
    }
    else
    {
        region_2 = 0U;
    }
    uint _S331 = _S330 | region_2;
    if(perforate_0)
    {
        region_2 = 8U;
    }
    else
    {
        region_2 = 0U;
    }
    uint outcome_5 = _S331 | region_2;
    uint _S332 = lp_body_0(b_21, kernelContext_65);
    *(kernelContext_65->lp_0+_S332) = packed_float4(float4(_S314, v_bl_0, v_res_0, (as_type<float>((outcome_5))))) ;
    packed_float4 device* _S333 = kernelContext_65->lp_0+(_S332 + 1U);
    if(crater_0)
    {
        k_14 = depth_1;
    }
    else
    {
        k_14 = 0.0f;
    }
    *_S333 = packed_float4(float4(k_14, scab_1, a_3, _S325)) ;
    if(perforate_0)
    {
        uint _S334 = a_counters_1(kernelContext_65);
        uint _S335 = atomic_fetch_add_explicit(kernelContext_65->atom_0+(_S334 + 14U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_0)
        {
            uint _S336 = a_counters_1(kernelContext_65);
            uint _S337 = atomic_fetch_add_explicit(kernelContext_65->atom_0+(_S336 + 13U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_0)
            {
                uint _S338 = a_counters_1(kernelContext_65);
                uint _S339 = atomic_fetch_add_explicit(kernelContext_65->atom_0+(_S338 + 12U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S340 = a_counters_1(kernelContext_65);
                uint _S341 = atomic_fetch_add_explicit(kernelContext_65->atom_0+(_S340 + 11U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_0)
    {
        if((kernelContext_65->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_21, _S307.body_0, _S307.mass_0, contact_9, d_11, _S307.m_eff_0 * (_S319 - sqrt(max(_S319 * _S319 - _S326 / _S307.m_eff_0, 0.0f))), kernelContext_65);
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_21, _S307.body_0, _S307.mass_0, contact_9, d_11, _S307.m_eff_0 * _S319, kernelContext_65);
            float3 _S342 = to_rubble_0(d_11) * float3((_S319 * clamp(_S305, 0.0f, kernelContext_65->params_0->dt_0))) ;
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
                uint _S343 = 7U * (_S307.body_0 + region_2 * kernelContext_65->params_0->max_bodies_0);
                (*(kernelContext_65->bodies_0+_S343)).xyz = (float4(*(kernelContext_65->bodies_0+_S343)) ).xyz + _S342;
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

void il_run_0(uint b_22, uint i_2, KernelContext_0 thread* kernelContext_66)
{
    bool _S344 = il_struck_0(b_22, kernelContext_66);
    uint j_1;
    if(_S344)
    {
        j_1 = 1U;
    }
    else
    {
        j_1 = 0U;
    }
    if((group_uniform_0(j_1, i_2)) == 0U)
    {
        return;
    }
    IlStriker_0 _S345 = il_striker_0(b_22, kernelContext_66);
    float _S346 = 0.00100000004749745f + 0.01999999955296516f * _S345.radius_0;
    uint _S347 = a_il_0(b_22, kernelContext_66);
    uint _S348 = atomic_load_explicit(kernelContext_66->atom_0+(_S347 + 4U), memory_order_relaxed);
    uint _S349 = atomic_load_explicit(kernelContext_66->atom_0+(_S347 + 1U), memory_order_relaxed);
    uint _S350 = atomic_load_explicit(kernelContext_66->atom_0+_S347, memory_order_relaxed);
    float end_0 = group_uniform_float_0((as_type<float>((_S350))), i_2);
    for(;;)
    {
        float best_0 = end_0;
        j_1 = i_2;
        for(;;)
        {
            if(j_1 < _S349)
            {
            }
            else
            {
                break;
            }
            uint _S351 = w_il_tube_0(kernelContext_66);
            uint at_4 = _S351 + 2U * (_S348 + j_1);
            if((as_type<float>((*(kernelContext_66->work_0+at_4)))) <= (end_0 + _S346))
            {
                best_0 = max(best_0, (as_type<float>((*(kernelContext_66->work_0+(at_4 + 1U))))));
            }
            j_1 = j_1 + 64U;
        }
        (*kernelContext_66->g_il_best_0)[i_2] = best_0;
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
            if(i_2 < step_1)
            {
                (*kernelContext_66->g_il_best_0)[i_2] = max((*kernelContext_66->g_il_best_0)[i_2], (*kernelContext_66->g_il_best_0)[i_2 + step_1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_1 = step_1 >> 1U;
        }
        float next_0 = group_uniform_float_0((*kernelContext_66->g_il_best_0)[int(0)], i_2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!(next_0 > end_0))
        {
            break;
        }
        end_0 = next_0;
    }
    if(i_2 == 0U)
    {
        atomic_store_explicit(kernelContext_66->atom_0+_S347, (as_type<uint>((end_0))), memory_order_relaxed);
    }
    return;
}

void il_tube_collect_0(uint k_15, KernelContext_0 thread* kernelContext_67)
{
    thread uint b_23;
    thread float2 t_3;
    bool _S352 = il_tube_0(k_15, &b_23, &t_3, kernelContext_67);
    if(!_S352)
    {
        return;
    }
    uint _S353 = w_il_tube_0(kernelContext_67);
    uint _S354 = a_il_0(b_23, kernelContext_67);
    uint _S355 = atomic_load_explicit(kernelContext_67->atom_0+(_S354 + 4U), memory_order_relaxed);
    uint _S356 = a_il_0(b_23, kernelContext_67);
    uint _S357 = atomic_fetch_add_explicit(kernelContext_67->atom_0+(_S356 + 5U), 1U, memory_order_relaxed);
    uint at_5 = _S353 + 2U * (_S355 + _S357);
    *(kernelContext_67->work_0+at_5) = (as_type<uint>((t_3.x)));
    *(kernelContext_67->work_0+(at_5 + 1U)) = (as_type<uint>((t_3.y)));
    return;
}

uint a_il_cursor_1(KernelContext_0 thread* kernelContext_68)
{
    uint _S358 = a_first_0(0U, kernelContext_68);
    return _S358 + 64U;
}

void il_tube_base_0(uint b_24, KernelContext_0 thread* kernelContext_69)
{
    bool _S359;
    if(b_24 < (kernelContext_69->params_0->body_count_0))
    {
        bool _S360 = il_struck_0(b_24, kernelContext_69);
        _S359 = _S360;
    }
    else
    {
        _S359 = false;
    }
    if(_S359)
    {
        uint _S361 = a_il_0(b_24, kernelContext_69);
        atomic<uint> device* _S362 = kernelContext_69->atom_0+(_S361 + 4U);
        uint _S363 = a_il_cursor_1(kernelContext_69);
        atomic<uint> device* _S364 = kernelContext_69->atom_0+_S363;
        uint _S365 = atomic_load_explicit(kernelContext_69->atom_0+(_S361 + 1U), memory_order_relaxed);
        uint _S366 = atomic_fetch_add_explicit(_S364, _S365, memory_order_relaxed);
        atomic_store_explicit(_S362, _S366, memory_order_relaxed);
    }
    return;
}

void il_tube_count_0(uint k_16, KernelContext_0 thread* kernelContext_70)
{
    thread uint b_25;
    thread float2 t_4;
    bool _S367 = il_tube_0(k_16, &b_25, &t_4, kernelContext_70);
    if(_S367)
    {
        uint _S368 = a_il_0(b_25, kernelContext_70);
        uint _S369 = atomic_fetch_add_explicit(kernelContext_70->atom_0+(_S368 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

void il_begin_0(uint b_26, KernelContext_0 thread* kernelContext_71)
{
    bool _S370;
    if(b_26 >= (kernelContext_71->params_0->body_count_0))
    {
        _S370 = true;
    }
    else
    {
        bool _S371 = il_struck_0(b_26, kernelContext_71);
        _S370 = !_S371;
    }
    if(_S370)
    {
        return;
    }
    thread float3 contact_10;
    thread float3 d_12;
    thread float speed_8;
    il_path_0(b_26, &contact_10, &d_12, &speed_8, kernelContext_71);
    uint _S372 = il_entry_0(b_26, kernelContext_71);
    IlStriker_0 _S373 = il_striker_0(b_26, kernelContext_71);
    float2 _S374 = il_interval_0(_S372, contact_10, d_12, _S373.radius_0, kernelContext_71);
    uint _S375 = a_il_0(b_26, kernelContext_71);
    atomic_store_explicit(kernelContext_71->atom_0+_S375, (as_type<uint>((max(_S374.y, 0.0f)))), memory_order_relaxed);
    uint i_3 = 1U;
    for(;;)
    {
        if(i_3 < 6U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit(kernelContext_71->atom_0+(_S375 + i_3), 0U, memory_order_relaxed);
        i_3 = i_3 + 1U;
    }
    return;
}

void il_breaks_1(uint i_4, KernelContext_0 thread* kernelContext_72)
{
    bool cut_1;
    if(i_4 >= (kernelContext_72->params_0->bond_count_0))
    {
        cut_1 = true;
    }
    else
    {
        uint _S376 = w_alive_0(kernelContext_72);
        cut_1 = (*(kernelContext_72->work_0+(_S376 + i_4))) == 0U;
    }
    if(cut_1)
    {
        return;
    }
    uint4 _S377 = uint4(*(kernelContext_72->bonds_0+i_4)) ;
    uint _S378 = _S377.x;
    uint _S379 = body_of_0(_S378, kernelContext_72);
    bool _S380 = il_struck_1(_S379, kernelContext_72);
    if(!_S380)
    {
        return;
    }
    uint _S381 = lp_body_0(_S379, kernelContext_72);
    float4 _S382 = float4(*(kernelContext_72->lp_0+_S381)) ;
    float4 _S383 = float4(*(kernelContext_72->lp_0+(_S381 + 1U))) ;
    uint outcome_6 = (as_type<uint>((_S382.w)));
    if((outcome_6 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_11;
    thread float3 d_13;
    thread float speed_9;
    il_path_1(_S379, &contact_11, &d_13, &speed_9, kernelContext_72);
    uint _S384 = il_entry_1(_S379, kernelContext_72);
    IlStriker_0 _S385 = il_striker_1(_S379, kernelContext_72);
    float _S386 = max(_S385.radius_0, 0.00100000004749745f);
    float h_4 = _S382.x;
    IlPlace_0 _S387 = il_place_0(_S378, _S384, contact_11, d_13, _S386, h_4, kernelContext_72);
    IlPlace_0 _S388 = il_place_0(_S377.y, _S384, contact_11, d_13, _S386, h_4, kernelContext_72);
    thread IlPlace_0 _S389 = _S387;
    bool _S390 = il_plug_0(outcome_6, _S386, h_4, &_S389);
    thread IlPlace_0 _S391 = _S388;
    bool _S392 = il_plug_0(outcome_6, _S386, h_4, &_S391);
    if(_S390 != _S392)
    {
        cut_1 = true;
    }
    else
    {
        float _S393 = _S383.y;
        thread IlPlace_0 _S394 = _S387;
        bool _S395 = il_scab_0(outcome_6, _S386, h_4, _S393, &_S394);
        thread IlPlace_0 _S396 = _S388;
        bool _S397 = il_scab_0(outcome_6, _S386, h_4, _S393, &_S396);
        cut_1 = _S395 != _S397;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S398 = _S383.x;
        float _S399 = _S383.z;
        thread IlPlace_0 _S400 = _S387;
        bool _S401 = il_crater_0(outcome_6, _S386, _S398, _S399, &_S400);
        cut_1 = _S401;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S402 = _S383.x;
        float _S403 = _S383.z;
        thread IlPlace_0 _S404 = _S388;
        bool _S405 = il_crater_0(outcome_6, _S386, _S402, _S403, &_S404);
        cut_1 = _S405;
    }
    if(cut_1)
    {
        uint _S406 = w_alive_0(kernelContext_72);
        *(kernelContext_72->work_0+(_S406 + i_4)) = 0U;
        uint _S407 = a_dirty_0(kernelContext_72);
        uint _S408 = atomic_fetch_or_explicit(kernelContext_72->atom_0+(_S407 + _S379), 1U, memory_order_relaxed);
        uint _S409 = a_counters_1(kernelContext_72);
        uint _S410 = atomic_fetch_add_explicit(kernelContext_72->atom_0+_S409, 1U, memory_order_relaxed);
        uint _S411 = atomic_fetch_add_explicit(kernelContext_72->atom_0+(_S409 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_4, kernelContext_72);
    }
    return;
}

void il_carry_1(uint k_17, KernelContext_0 thread* kernelContext_73)
{
    if(k_17 >= (kernelContext_73->params_0->chunk_count_0))
    {
        return;
    }
    uint _S412 = w_carry_0(kernelContext_73);
    uint _S413 = _S412 + k_17;
    uint device* _S414 = kernelContext_73->work_0+_S413;
    uint c_4 = *_S414;
    if((*_S414) == 0U)
    {
        return;
    }
    *(kernelContext_73->work_0+_S413) = 0U;
    uint b0_1 = c_4 - 1U;
    bool _S415 = il_struck_1(b0_1, kernelContext_73);
    if(!_S415)
    {
        return;
    }
    thread float3 contact_12;
    thread float3 d_14;
    thread float speed_10;
    il_path_1(b0_1, &contact_12, &d_14, &speed_10, kernelContext_73);
    float3 dr_1 = to_rubble_0(d_14);
    IlStriker_0 _S416 = il_striker_1(b0_1, kernelContext_73);
    uint _S417 = a_il_0(b0_1, kernelContext_73);
    float _S418 = read64_0(_S417 + 2U, kernelContext_73->params_0->scale_m_0, kernelContext_73);
    float3 v_wall_1 = (float4(*(kernelContext_73->bodies_0+(7U * b0_1 + 3U))) ).xyz;
    uint _S419 = 7U * _S416.body_0;
    float3 v_shot_1 = (float4(*(kernelContext_73->bodies_0+(_S419 + 3U))) ).xyz;
    float closing_1 = dot(v_shot_1 - v_wall_1, dr_1);
    float v_r_1 = _S416.mass_0 * closing_1 / (_S416.mass_0 + _S418);
    float3 _S420 = v_shot_1 + dr_1 * float3((v_r_1 - closing_1)) ;
    float3 _S421 = v_wall_1 + dr_1 * float3(v_r_1) ;
    float _S422 = il_t_in_1(b0_1, kernelContext_73);
    float3 _S423 = (float4(*(kernelContext_73->bodies_0+_S419)) ).xyz + dr_1 * float3(((speed_10 - v_r_1) * clamp(_S422, 0.0f, kernelContext_73->params_0->dt_0))) ;
    uint _S424 = body_of_0(k_17, kernelContext_73);
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
        uint s_at_2 = 7U * (_S416.body_0 + region_3 * kernelContext_73->params_0->max_bodies_0);
        (*(kernelContext_73->bodies_0+s_at_2)).xyz = _S423;
        (*(kernelContext_73->bodies_0+(s_at_2 + 3U))).xyz = _S420;
        (*(kernelContext_73->bodies_0+(7U * (_S424 + region_3 * kernelContext_73->params_0->max_bodies_0) + 3U))).xyz = _S421;
        region_3 = region_3 + 1U;
    }
    return;
}

void il_mark_1(uint k_18, KernelContext_0 thread* kernelContext_74)
{
    if(k_18 >= (kernelContext_74->params_0->chunk_count_0))
    {
        return;
    }
    uint _S425 = body_of_0(k_18, kernelContext_74);
    bool _S426 = il_struck_1(_S425, kernelContext_74);
    if(!_S426)
    {
        return;
    }
    uint _S427 = lp_body_0(_S425, kernelContext_74);
    float4 _S428 = float4(*(kernelContext_74->lp_0+_S427)) ;
    float4 _S429 = float4(*(kernelContext_74->lp_0+(_S427 + 1U))) ;
    uint outcome_7 = (as_type<uint>((_S428.w)));
    if((outcome_7 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_13;
    thread float3 d_15;
    thread float speed_11;
    il_path_1(_S425, &contact_13, &d_15, &speed_11, kernelContext_74);
    uint _S430 = il_entry_1(_S425, kernelContext_74);
    IlStriker_0 _S431 = il_striker_1(_S425, kernelContext_74);
    float _S432 = max(_S431.radius_0, 0.00100000004749745f);
    float _S433 = _S428.x;
    IlPlace_0 _S434 = il_place_0(k_18, _S430, contact_13, d_15, _S432, _S433, kernelContext_74);
    thread IlPlace_0 _S435 = _S434;
    bool _S436 = il_plug_0(outcome_7, _S432, _S433, &_S435);
    if((_S431.launch_0) != 0U)
    {
        float2 _S437 = il_interval_0(k_18, contact_13 - d_15 * float3(_S431.bound_0) , d_15, _S431.bound_0, kernelContext_74);
        float _S438 = _S437.x;
        float _S439 = _S437.y;
        bool path_1;
        if(_S438 <= _S439)
        {
            path_1 = _S438 <= (_S433 + 2.0f * _S431.bound_0);
        }
        else
        {
            path_1 = false;
        }
        if(path_1)
        {
            path_1 = _S439 >= 0.0f;
        }
        else
        {
            path_1 = false;
        }
        if(_S436)
        {
            path_1 = true;
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            float _S440 = _S429.y;
            thread IlPlace_0 _S441 = _S434;
            bool _S442 = il_scab_0(outcome_7, _S432, _S433, _S440, &_S441);
            path_1 = _S442;
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            float _S443 = _S429.x;
            float _S444 = _S429.z;
            thread IlPlace_0 _S445 = _S434;
            bool _S446 = il_crater_0(outcome_7, _S432, _S443, _S444, &_S445);
            path_1 = _S446;
        }
        if(path_1)
        {
            uint _S447 = w_marks_0(kernelContext_74);
            *(kernelContext_74->work_0+(_S447 + k_18)) = _S431.launch_0;
        }
    }
    if(_S436)
    {
        uint _S448 = w_carry_0(kernelContext_74);
        *(kernelContext_74->work_0+(_S448 + k_18)) = _S425 + 1U;
        uint _S449 = a_il_0(_S425, kernelContext_74);
        add64_0(_S449 + 2U, (float4(*(kernelContext_74->chunks_0+4U * k_18)) ).w, kernelContext_74->params_0->scale_m_0, kernelContext_74);
    }
    return;
}

void il_strike_1(uint b_27, KernelContext_0 thread* kernelContext_75)
{
    bool damage_1;
    if(b_27 >= (kernelContext_75->params_0->body_count_0))
    {
        damage_1 = true;
    }
    else
    {
        bool _S450 = il_struck_1(b_27, kernelContext_75);
        damage_1 = !_S450;
    }
    if(damage_1)
    {
        return;
    }
    float _S451 = il_t_in_1(b_27, kernelContext_75);
    uint _S452 = il_entry_1(b_27, kernelContext_75);
    thread float3 contact_14;
    thread float3 d_16;
    thread float speed_12;
    il_path_1(b_27, &contact_14, &d_16, &speed_12, kernelContext_75);
    IlStriker_0 _S453 = il_striker_1(b_27, kernelContext_75);
    float _S454 = max(_S453.radius_0, 0.00100000004749745f);
    if((_S453.launch_0) != 0U)
    {
        uint _S455 = w_marks_0(kernelContext_75);
        damage_1 = (*(kernelContext_75->work_0+(_S455 + _S452))) == (_S453.launch_0);
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        uint _S456 = lp_body_0(b_27, kernelContext_75);
        packed_float4 _S457 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_75->lp_0+_S456) = _S457;
        *(kernelContext_75->lp_0+(_S456 + 1U)) = _S457;
        return;
    }
    uint _S458 = a_il_0(b_27, kernelContext_75);
    uint _S459 = atomic_load_explicit(kernelContext_75->atom_0+_S458, memory_order_relaxed);
    float _S460 = max((as_type<float>((_S459))) - _S454, 9.99999997475242708e-07f);
    if(!(_S451 > 0.0f))
    {
        uint _S461 = lp_body_0(b_27, kernelContext_75);
        *(kernelContext_75->lp_0+_S461) = packed_float4(float4(_S460, 0.0f, 0.0f, 0.0f)) ;
        *(kernelContext_75->lp_0+(_S461 + 1U)) = packed_float4(float4(0.0f) ) ;
        return;
    }
    uint o_3 = kernelContext_75->params_0->ext_chunks_0 + 4U * _S452;
    float4 _S462 = float4(*(kernelContext_75->ext_0+(o_3 + 1U))) ;
    thread JointMaterial_0 dm_1;
    float4 _S463 = float4(0.0f) ;
    (&dm_1)->strength_0 = _S463;
    (&dm_1)->energy_0 = _S463;
    (&dm_1)->dif_0 = float4(*(kernelContext_75->ext_0+(o_3 + 2U))) ;
    float4 _S464 = float4(*(kernelContext_75->ext_0+(o_3 + 3U))) ;
    float _S465 = _S464.x;
    (&dm_1)->misc_0 = float4(_S465, 0.0f, 0.0f, 0.0f);
    (&dm_1)->kind_flags_0 = uint4(0U) ;
    float nu_1 = _S464.y;
    float e_target_1 = _S464.z / (1.0f - nu_1 * nu_1);
    float e_star_1 = 1.0f / (1.0f / _S453.modulus_0 + 1.0f / e_target_1);
    float _S466 = max(speed_12, 0.00100000004749745f);
    float _S467 = sqrt(_S454);
    float h0_1 = pow(15.0f * _S453.m_eff_0 * _S466 * _S466 / (16.0f * e_star_1 * _S467), 0.40000000596046448f);
    float t_c_1 = 2.94000005722045898f * h0_1 / _S466;
    float force_1 = 1.33333337306976318f * e_star_1 * _S467 * pow(h0_1, 1.5f);
    float a_4 = sqrt(_S454 * h0_1);
    float p0_1 = 3.0f * force_1 / (6.28318548202514648f * a_4 * a_4);
    float ring_1 = (1.0f - 2.0f * nu_1) * p0_1 / 3.0f;
    float k_19;
    if(_S465 > 0.0f)
    {
        float _S468 = _S466 / _S460;
        thread JointMaterial_0 _S469 = dm_1;
        float _S470 = dif_factor_0(&_S469, _S468);
        k_19 = _S470;
    }
    else
    {
        k_19 = 1.0f;
    }
    float ft_1 = k_19 * _S462.y;
    float fc_1 = k_19 * _S462.z;
    float tau_1 = k_19 * _S462.w;
    if(p0_1 > fc_1)
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = ring_1 > ft_1;
    }
    float _S471 = fc_1 * (3.14159274101257324f * _S454 * _S454);
    float _S472 = min(_S471 * _S460, 1.57079637050628662f * (2.0f * _S454 + _S460) * _S460 * _S460 * tau_1);
    float _S473 = 2.0f * _S472;
    float v_bl_1 = sqrt(_S473 / _S453.m_eff_0);
    bool perforate_1;
    if(damage_1)
    {
        perforate_1 = _S466 >= v_bl_1;
    }
    else
    {
        perforate_1 = false;
    }
    float r2_1 = _S454 + _S460;
    float _S474 = _S462.x;
    float plug_1 = _S474 * 3.14159274101257324f * _S460 / 3.0f * (_S454 * _S454 + _S454 * r2_1 + r2_1 * r2_1);
    float v_res_1;
    if(perforate_1)
    {
        v_res_1 = _S453.mass_0 / (_S453.mass_0 + plug_1) * sqrt(max(_S466 * _S466 - v_bl_1 * v_bl_1, 0.0f));
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
        depth_2 = min(0.5f * _S453.m_eff_0 * _S466 * _S466 / _S471, _S460);
    }
    else
    {
        depth_2 = 0.0f;
    }
    bool crater_1 = depth_2 > a_4;
    float c_5 = sqrt(e_target_1 / _S474);
    float z_t_1 = _S474 * c_5;
    float rho_p_1 = _S453.mass_0 / (4.18879032135009766f * _S454 * _S454 * _S454);
    float z_p_1 = rho_p_1 * sqrt(_S453.modulus_0 / rho_p_1);
    float mean_pressure_1 = force_1 / (3.14159274101257324f * a_4 * a_4);
    float _S475 = min(min(z_t_1 * _S466 * z_p_1 / (z_p_1 + z_t_1), mean_pressure_1), fc_1);
    float source_radius_1;
    if(mean_pressure_1 > fc_1)
    {
        source_radius_1 = max(a_4, _S454);
    }
    else
    {
        source_radius_1 = a_4;
    }
    float pulse_1 = c_5 * t_c_1;
    float back_1 = _S475 * min(source_radius_1 / _S460, 1.0f) * min(2.0f * _S460 / pulse_1, 1.0f);
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
        scab_2 = 0.5f * min(pulse_1, _S460);
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
    uint _S476;
    if(crater_1)
    {
        _S476 = 2U;
    }
    else
    {
        _S476 = 0U;
    }
    uint _S477 = region_4 | _S476;
    if(spall_1)
    {
        region_4 = 4U;
    }
    else
    {
        region_4 = 0U;
    }
    uint _S478 = _S477 | region_4;
    if(perforate_1)
    {
        region_4 = 8U;
    }
    else
    {
        region_4 = 0U;
    }
    uint outcome_8 = _S478 | region_4;
    uint _S479 = lp_body_0(b_27, kernelContext_75);
    *(kernelContext_75->lp_0+_S479) = packed_float4(float4(_S460, v_bl_1, v_res_1, (as_type<float>((outcome_8))))) ;
    packed_float4 device* _S480 = kernelContext_75->lp_0+(_S479 + 1U);
    if(crater_1)
    {
        k_19 = depth_2;
    }
    else
    {
        k_19 = 0.0f;
    }
    *_S480 = packed_float4(float4(k_19, scab_2, a_4, _S472)) ;
    if(perforate_1)
    {
        uint _S481 = a_counters_1(kernelContext_75);
        uint _S482 = atomic_fetch_add_explicit(kernelContext_75->atom_0+(_S481 + 14U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_1)
        {
            uint _S483 = a_counters_1(kernelContext_75);
            uint _S484 = atomic_fetch_add_explicit(kernelContext_75->atom_0+(_S483 + 13U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_1)
            {
                uint _S485 = a_counters_1(kernelContext_75);
                uint _S486 = atomic_fetch_add_explicit(kernelContext_75->atom_0+(_S485 + 12U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S487 = a_counters_1(kernelContext_75);
                uint _S488 = atomic_fetch_add_explicit(kernelContext_75->atom_0+(_S487 + 11U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_1)
    {
        if((kernelContext_75->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_27, _S453.body_0, _S453.mass_0, contact_14, d_16, _S453.m_eff_0 * (_S466 - sqrt(max(_S466 * _S466 - _S473 / _S453.m_eff_0, 0.0f))), kernelContext_75);
        }
    }
    else
    {
        if(damage_1)
        {
            il_impulse_0(b_27, _S453.body_0, _S453.mass_0, contact_14, d_16, _S453.m_eff_0 * _S466, kernelContext_75);
            float3 _S489 = to_rubble_0(d_16) * float3((_S466 * clamp(_S451, 0.0f, kernelContext_75->params_0->dt_0))) ;
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
                uint _S490 = 7U * (_S453.body_0 + region_4 * kernelContext_75->params_0->max_bodies_0);
                (*(kernelContext_75->bodies_0+_S490)).xyz = (float4(*(kernelContext_75->bodies_0+_S490)) ).xyz + _S489;
                region_4 = region_4 + 1U;
            }
        }
    }
    return;
}

void il_run_1(uint b_28, uint i_5, KernelContext_0 thread* kernelContext_76)
{
    bool _S491 = il_struck_1(b_28, kernelContext_76);
    uint j_2;
    if(_S491)
    {
        j_2 = 1U;
    }
    else
    {
        j_2 = 0U;
    }
    if((group_uniform_0(j_2, i_5)) == 0U)
    {
        return;
    }
    IlStriker_0 _S492 = il_striker_1(b_28, kernelContext_76);
    float _S493 = 0.00100000004749745f + 0.01999999955296516f * _S492.radius_0;
    uint _S494 = a_il_0(b_28, kernelContext_76);
    uint _S495 = atomic_load_explicit(kernelContext_76->atom_0+(_S494 + 4U), memory_order_relaxed);
    uint _S496 = atomic_load_explicit(kernelContext_76->atom_0+(_S494 + 1U), memory_order_relaxed);
    uint _S497 = atomic_load_explicit(kernelContext_76->atom_0+_S494, memory_order_relaxed);
    float end_1 = group_uniform_float_0((as_type<float>((_S497))), i_5);
    for(;;)
    {
        float best_1 = end_1;
        j_2 = i_5;
        for(;;)
        {
            if(j_2 < _S496)
            {
            }
            else
            {
                break;
            }
            uint _S498 = w_il_tube_0(kernelContext_76);
            uint at_6 = _S498 + 2U * (_S495 + j_2);
            if((as_type<float>((*(kernelContext_76->work_0+at_6)))) <= (end_1 + _S493))
            {
                best_1 = max(best_1, (as_type<float>((*(kernelContext_76->work_0+(at_6 + 1U))))));
            }
            j_2 = j_2 + 64U;
        }
        (*kernelContext_76->g_il_best_0)[i_5] = best_1;
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
            if(i_5 < step_2)
            {
                (*kernelContext_76->g_il_best_0)[i_5] = max((*kernelContext_76->g_il_best_0)[i_5], (*kernelContext_76->g_il_best_0)[i_5 + step_2]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_2 = step_2 >> 1U;
        }
        float next_1 = group_uniform_float_0((*kernelContext_76->g_il_best_0)[int(0)], i_5);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!(next_1 > end_1))
        {
            break;
        }
        end_1 = next_1;
    }
    if(i_5 == 0U)
    {
        atomic_store_explicit(kernelContext_76->atom_0+_S494, (as_type<uint>((end_1))), memory_order_relaxed);
    }
    return;
}

void il_tube_collect_1(uint k_20, KernelContext_0 thread* kernelContext_77)
{
    thread uint b_29;
    thread float2 t_5;
    bool _S499 = il_tube_1(k_20, &b_29, &t_5, kernelContext_77);
    if(!_S499)
    {
        return;
    }
    uint _S500 = w_il_tube_0(kernelContext_77);
    uint _S501 = a_il_0(b_29, kernelContext_77);
    uint _S502 = atomic_load_explicit(kernelContext_77->atom_0+(_S501 + 4U), memory_order_relaxed);
    uint _S503 = a_il_0(b_29, kernelContext_77);
    uint _S504 = atomic_fetch_add_explicit(kernelContext_77->atom_0+(_S503 + 5U), 1U, memory_order_relaxed);
    uint at_7 = _S500 + 2U * (_S502 + _S504);
    *(kernelContext_77->work_0+at_7) = (as_type<uint>((t_5.x)));
    *(kernelContext_77->work_0+(at_7 + 1U)) = (as_type<uint>((t_5.y)));
    return;
}

void il_tube_base_1(uint b_30, KernelContext_0 thread* kernelContext_78)
{
    bool _S505;
    if(b_30 < (kernelContext_78->params_0->body_count_0))
    {
        bool _S506 = il_struck_1(b_30, kernelContext_78);
        _S505 = _S506;
    }
    else
    {
        _S505 = false;
    }
    if(_S505)
    {
        uint _S507 = a_il_0(b_30, kernelContext_78);
        atomic<uint> device* _S508 = kernelContext_78->atom_0+(_S507 + 4U);
        uint _S509 = a_il_cursor_1(kernelContext_78);
        atomic<uint> device* _S510 = kernelContext_78->atom_0+_S509;
        uint _S511 = atomic_load_explicit(kernelContext_78->atom_0+(_S507 + 1U), memory_order_relaxed);
        uint _S512 = atomic_fetch_add_explicit(_S510, _S511, memory_order_relaxed);
        atomic_store_explicit(_S508, _S512, memory_order_relaxed);
    }
    return;
}

void il_tube_count_1(uint k_21, KernelContext_0 thread* kernelContext_79)
{
    thread uint b_31;
    thread float2 t_6;
    bool _S513 = il_tube_1(k_21, &b_31, &t_6, kernelContext_79);
    if(_S513)
    {
        uint _S514 = a_il_0(b_31, kernelContext_79);
        uint _S515 = atomic_fetch_add_explicit(kernelContext_79->atom_0+(_S514 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_first_1(uint shot_3, KernelContext_0 thread* kernelContext_80)
{
    uint _S516 = a_strike_0(kernelContext_80);
    return _S516 + 29U * kernelContext_80->params_0->body_count_0 + shot_3;
}

void il_begin_1(uint b_32, KernelContext_0 thread* kernelContext_81)
{
    bool _S517;
    if(b_32 >= (kernelContext_81->params_0->body_count_0))
    {
        _S517 = true;
    }
    else
    {
        bool _S518 = il_struck_1(b_32, kernelContext_81);
        _S517 = !_S518;
    }
    if(_S517)
    {
        return;
    }
    thread uint shot_4;
    bool _S519 = struck_0(b_32, &shot_4, kernelContext_81);
    uint _S520 = a_strike_0(kernelContext_81);
    uint _S521 = atomic_load_explicit(kernelContext_81->atom_0+(_S520 + 9U * b_32 + 6U), memory_order_relaxed);
    uint _S522 = a_first_1(shot_4, kernelContext_81);
    uint _S523 = atomic_load_explicit(kernelContext_81->atom_0+_S522, memory_order_relaxed);
    if(_S521 != _S523)
    {
        uint _S524 = strike_at_0(b_32, kernelContext_81);
        (*(kernelContext_81->frames_0+_S524))[int(3)] = 0.0f;
        return;
    }
    thread float3 contact_15;
    thread float3 d_17;
    thread float speed_13;
    il_path_1(b_32, &contact_15, &d_17, &speed_13, kernelContext_81);
    uint _S525 = il_entry_1(b_32, kernelContext_81);
    IlStriker_0 _S526 = il_striker_1(b_32, kernelContext_81);
    float2 _S527 = il_interval_0(_S525, contact_15, d_17, _S526.radius_0, kernelContext_81);
    uint _S528 = a_il_0(b_32, kernelContext_81);
    atomic_store_explicit(kernelContext_81->atom_0+_S528, (as_type<uint>((max(_S527.y, 0.0f)))), memory_order_relaxed);
    uint i_6 = 1U;
    for(;;)
    {
        if(i_6 < 6U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit(kernelContext_81->atom_0+(_S528 + i_6), 0U, memory_order_relaxed);
        i_6 = i_6 + 1U;
    }
    return;
}

[[kernel]] void fr_events_reset(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], atomic<uint> device* atom_1 [[buffer(5)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], uint device* sort_1 [[buffer(11)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], packed_float4 device* loads_1 [[buffer(12)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]], packed_float4 device* view_chunks_1 [[buffer(17)]], ViewParams_0 constant* view_1 [[buffer(16)]], packed_float4 device* objects_1 [[buffer(18)]], uint device* shot_bodies_1 [[buffer(20)]], float device* lines_1 [[buffer(19)]], packed_float4 device* geometry_1 [[buffer(21)]], packed_float4 device* shots_1 [[buffer(22)]], packed_float4 device* lp_1 [[buffer(24)]], packed_float4 device* ext_2 [[buffer(23)]], packed_float4 device* engine_contacts_1 [[buffer(25)]], CrushParams_0 constant* crush_1 [[buffer(28)]], packed_float4 device* crush_rw_1 [[buffer(27)]], packed_float4 device* crush_tab_1 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_82;
    (&kernelContext_82)->params_0 = params_1;
    (&kernelContext_82)->atom_0 = atom_1;
    (&kernelContext_82)->work_0 = work_1;
    (&kernelContext_82)->frames_0 = frames_1;
    (&kernelContext_82)->bodies_0 = bodies_1;
    (&kernelContext_82)->chunks_0 = chunks_1;
    (&kernelContext_82)->sort_0 = sort_1;
    (&kernelContext_82)->manifolds_0 = manifolds_1;
    (&kernelContext_82)->loads_0 = loads_1;
    (&kernelContext_82)->bonds_0 = bonds_1;
    (&kernelContext_82)->args_0 = args_1;
    (&kernelContext_82)->info_0 = info_1;
    (&kernelContext_82)->scan_0 = scan_1;
    (&kernelContext_82)->parts_0 = parts_1;
    (&kernelContext_82)->colliders_0 = colliders_1;
    (&kernelContext_82)->collider_info_0 = collider_info_1;
    (&kernelContext_82)->view_chunks_0 = view_chunks_1;
    (&kernelContext_82)->view_0 = view_1;
    (&kernelContext_82)->objects_0 = objects_1;
    (&kernelContext_82)->shot_bodies_0 = shot_bodies_1;
    (&kernelContext_82)->lines_0 = lines_1;
    (&kernelContext_82)->geometry_0 = geometry_1;
    (&kernelContext_82)->shots_0 = shots_1;
    (&kernelContext_82)->lp_0 = lp_1;
    (&kernelContext_82)->ext_0 = ext_2;
    (&kernelContext_82)->engine_contacts_0 = engine_contacts_1;
    (&kernelContext_82)->crush_0 = crush_1;
    (&kernelContext_82)->crush_rw_0 = crush_rw_1;
    (&kernelContext_82)->crush_tab_0 = crush_tab_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_82)->g_scan_0 = &g_scan_1;
    threadgroup array<float, int(64)> g_il_best_1;
    (&kernelContext_82)->g_il_best_0 = &g_il_best_1;
    uint _S529 = a_counters_1(&kernelContext_82);
    atomic_store_explicit((&kernelContext_82)->atom_0+(_S529 + 20U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_82)->atom_0+(_S529 + 21U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_82)->atom_0+(_S529 + 22U), 0U, memory_order_relaxed);
    return;
}

bool live_0(uint c_6, KernelContext_0 thread* kernelContext_83)
{
    bool _S530;
    if(c_6 < (kernelContext_83->params_0->chunk_used_0))
    {
        uint _S531 = body_of_0(c_6, kernelContext_83);
        _S530 = _S531 != 4294967295U;
    }
    else
    {
        _S530 = false;
    }
    return _S530;
}

uint w_cs_rows_1(KernelContext_0 thread* kernelContext_84)
{
    uint _S532 = w_host_breaks_0(kernelContext_84);
    return _S532 + 1U + kernelContext_84->params_0->bond_count_0;
}

uint w_frame_0(KernelContext_0 thread* kernelContext_85)
{
    uint _S533 = w_cs_rows_1(kernelContext_85);
    return _S533 + 1U;
}

uint frame_parity_0(KernelContext_0 thread* kernelContext_86)
{
    uint _S534 = w_frame_0(kernelContext_86);
    return *(kernelContext_86->work_0+_S534);
}

uint l_pool_0(uint half_1, KernelContext_0 thread* kernelContext_87)
{
    return kernelContext_87->params_0->chunk_count_0 + 3U * kernelContext_87->params_0->pool_cap_0 * half_1;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S535 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S535, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S536 = max(0.0f, _S535);
    float _S537 = min(dt_1, _S535 + duration_0);
    bool _S538;
    if(_S537 > _S536)
    {
        _S538 = duration_0 > 0.0f;
    }
    else
    {
        _S538 = false;
    }
    float sum_1;
    if(_S538)
    {
        float w_2 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S536 - _S535)) - cos(w_2 * (_S537 - _S535)));
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

uint l_vec_0(KernelContext_0 thread* kernelContext_88)
{
    return kernelContext_88->params_0->chunk_count_0 + 6U * kernelContext_88->params_0->pool_cap_0;
}

[[kernel]] void fr_loads(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], atomic<uint> device* atom_2 [[buffer(5)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]], packed_float4 device* geometry_2 [[buffer(21)]], packed_float4 device* shots_2 [[buffer(22)]], packed_float4 device* lp_2 [[buffer(24)]], packed_float4 device* ext_3 [[buffer(23)]], packed_float4 device* engine_contacts_2 [[buffer(25)]], CrushParams_0 constant* crush_2 [[buffer(28)]], packed_float4 device* crush_rw_2 [[buffer(27)]], packed_float4 device* crush_tab_2 [[buffer(26)]])
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
    thread KernelContext_0 kernelContext_89;
    (&kernelContext_89)->params_0 = params_2;
    (&kernelContext_89)->atom_0 = atom_2;
    (&kernelContext_89)->work_0 = work_2;
    (&kernelContext_89)->frames_0 = frames_2;
    (&kernelContext_89)->bodies_0 = bodies_2;
    (&kernelContext_89)->chunks_0 = chunks_2;
    (&kernelContext_89)->sort_0 = sort_2;
    (&kernelContext_89)->manifolds_0 = manifolds_2;
    (&kernelContext_89)->loads_0 = loads_2;
    (&kernelContext_89)->bonds_0 = bonds_2;
    (&kernelContext_89)->args_0 = args_2;
    (&kernelContext_89)->info_0 = info_2;
    (&kernelContext_89)->scan_0 = scan_2;
    (&kernelContext_89)->parts_0 = parts_2;
    (&kernelContext_89)->colliders_0 = colliders_2;
    (&kernelContext_89)->collider_info_0 = collider_info_2;
    (&kernelContext_89)->view_chunks_0 = view_chunks_2;
    (&kernelContext_89)->view_0 = view_2;
    (&kernelContext_89)->objects_0 = objects_2;
    (&kernelContext_89)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_89)->lines_0 = lines_2;
    (&kernelContext_89)->geometry_0 = geometry_2;
    (&kernelContext_89)->shots_0 = shots_2;
    (&kernelContext_89)->lp_0 = lp_2;
    (&kernelContext_89)->ext_0 = ext_3;
    (&kernelContext_89)->engine_contacts_0 = engine_contacts_2;
    (&kernelContext_89)->crush_0 = crush_2;
    (&kernelContext_89)->crush_rw_0 = crush_rw_2;
    (&kernelContext_89)->crush_tab_0 = crush_tab_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_89)->g_scan_0 = &g_scan_2;
    threadgroup array<float, int(64)> g_il_best_2;
    (&kernelContext_89)->g_il_best_0 = &g_il_best_2;
    uint k_22 = id_1.x;
    bool _S539 = live_0(k_22, &kernelContext_89);
    if(!_S539)
    {
        return;
    }
    uint _S540 = body_of_0(k_22, &kernelContext_89);
    uint b_33 = 7U * _S540;
    thread Quat_0 q_11;
    thread float3 origin_3;
    structure_pose_0(_S540, b_33, &q_11, &origin_3, &kernelContext_89);
    uint _S541 = 4U * k_22;
    float4 _S542 = float4(*((&kernelContext_89)->chunks_0+_S541)) ;
    float3 _S543 = _S542.xyz;
    thread Quat_0 _S544 = q_11;
    float3 _S545 = rotate_0(&_S544, _S543);
    float3 centre_0 = origin_3 + _S545;
    float inv_mass_2 = (float4(*((&kernelContext_89)->bodies_0+b_33)) ).w;
    float m_body_0;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint _S546 = frame_parity_0(&kernelContext_89);
    uint to_0 = 1U - _S546;
    uint _S547 = w_pulses_0(&kernelContext_89);
    uint _S548 = _S547 + _S541;
    uint _S549 = _S548 + 2U * _S546;
    uint off_0 = *((&kernelContext_89)->work_0+_S549);
    uint device* _S550 = (&kernelContext_89)->work_0+(_S549 + 1U);
    uint carried_0 = *_S550;
    uint room_0 = *_S550 + (&kernelContext_89)->sort_0[(&kernelContext_89)->params_0->chunk_count_0 + 1U + k_22];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S551 = a_counters_1(&kernelContext_89);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_89)->atom_0+(_S551 + 5U + to_0), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_89)->params_0->pool_cap_0))
        {
            uint _S552 = atomic_fetch_or_explicit((&kernelContext_89)->atom_0+(_S551 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_89)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_89)->params_0->pool_cap_0 - base_1;
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
        uint _S553 = base_0;
        base_0 = base_1;
        room_1 = _S553;
    }
    else
    {
        base_0 = 0U;
        room_1 = room_0;
    }
    uint _S554 = (&kernelContext_89)->params_0->chunk_count_0;
    float _S555 = _S542.w;
    float4 _S556 = float4(*((&kernelContext_89)->chunks_0+(_S541 + 3U))) ;
    float _S557 = (&kernelContext_89)->params_0->band_0;
    float _S558 = (&kernelContext_89)->params_0->dt_0;
    bool _S559 = ((&kernelContext_89)->params_0->impact_peak_0) != 0U;
    uint _S560 = l_pool_0(_S546, &kernelContext_89);
    uint _S561 = _S560 + 3U * off_0;
    uint _S562 = l_pool_0(to_0, &kernelContext_89);
    uint _S563 = _S562 + 3U * base_0;
    for(;;)
    {
        float3 _S564 = float3(0.0f) ;
        uint _S565 = (&kernelContext_89)->sort_0[k_22];
        uint _S566 = (&kernelContext_89)->sort_0[_S554 + 1U + k_22];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_3 = 0U;
        f_total_0 = _S564;
        t_total_0 = _S564;
        for(;;)
        {
            if(step_3 < _S566)
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
                if(q3_0 < _S566)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_89)->manifolds_0[_S565 + q3_0].ids_0) ).y + 1U;
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
            Record_natural_0 r_19 = (&kernelContext_89)->manifolds_0[_S565 + p_10];
            float4 _S567 = float4(r_19.impulse_0) ;
            float3 _S568 = _S567.xyz;
            float j_3 = length(_S568);
            if(j_3 == 0.0f)
            {
                uint _S569 = step_3 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_3 = _S569;
                continue;
            }
            float m_other_0 = (float4(r_19.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S555;
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
            float _S570 = min(j_3, 2.0f * m_1 * max(_S567.w - _S557, 0.0f));
            if(_S570 < j_3)
            {
                float3 f_1 = _S568 / float3(j_3)  * float3(((j_3 - _S570) / _S558)) ;
                float3 _S571 = t_total_0 + cross((float4(r_19.point_0) ).xyz - centre_0, f_1);
                f_total_1 = f_total_0 + f_1;
                t_total_1 = _S571;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            uint _S569 = step_3 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_3 = _S569;
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
            uint _S572 = _S561 + 3U * p_10;
            float4 _S573 = float4(*((&kernelContext_89)->loads_0+_S572)) ;
            thread float4 a_5 = _S573;
            float4 _S574 = float4(*((&kernelContext_89)->loads_0+(_S572 + 1U))) ;
            float4 _S575 = float4(*((&kernelContext_89)->loads_0+(_S572 + 2U))) ;
            float _S576 = _S574.w;
            float mean_0 = pulse_mean_0(_S573.x, _S573.y, _S573.z, _S573.w, _S576, _S558);
            if(mean_0 != 0.0f)
            {
                float3 f_2 = _S575.xyz * float3(mean_0) ;
                float3 _S577 = t_total_0 + cross(_S574.xyz - centre_0, f_2);
                f_total_0 = f_total_0 + f_2;
                t_total_0 = _S577;
            }
            if((a_5.x + a_5.z + _S576) > _S558)
            {
                a_5.x = a_5.x - _S558;
                uint _S578 = _S563 + 3U * kept_0;
                *((&kernelContext_89)->loads_0+_S578) = packed_float4(a_5) ;
                *((&kernelContext_89)->loads_0+(_S578 + 1U)) = packed_float4(_S574) ;
                *((&kernelContext_89)->loads_0+(_S578 + 2U)) = packed_float4(_S575) ;
                kept_0 = kept_0 + 1U;
            }
            p_10 = p_10 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S566)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_20 = (&kernelContext_89)->manifolds_0[_S565 + q3_0];
            float4 _S579 = float4(r_20.impulse_0) ;
            float3 _S580 = _S579.xyz;
            float j_4 = length(_S580);
            if(j_4 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S581 = float4(r_20.other_0) ;
            float m_other_1 = _S581.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S555;
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
            float _S582 = _S579.w;
            float _S583 = min(j_4, 2.0f * m_1 * max(_S582 - _S557, 0.0f));
            if(_S583 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S584 = _S556.y;
            float duration_1 = hertz_duration_0(m_1, min(_S581.z, _S556.z), 1.0f / ((1.0f - _S584 * _S584) / _S556.x + 1.0f / _S581.y), _S582);
            float peak_1 = 3.14159274101257324f * _S583 / (2.0f * duration_1);
            float3 dir_0 = _S580 / float3(j_4) ;
            float mean_1;
            if(_S559)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S558);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_3 = dir_0 * float3(mean_1) ;
                float3 _S585 = t_total_0 + cross((float4(r_20.point_0) ).xyz - centre_0, f_3);
                f_total_1 = f_total_0 + f_3;
                t_total_1 = _S585;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            uint kept_1;
            if(duration_1 > _S558)
            {
                if(kept_0 < room_1)
                {
                    uint _S586 = _S563 + 3U * kept_0;
                    *((&kernelContext_89)->loads_0+_S586) = packed_float4(float4(- _S558, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_89)->loads_0+(_S586 + 1U)) = packed_float4(float4((float4(r_20.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_89)->loads_0+(_S586 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
        uint _S587 = a_counters_1(&kernelContext_89);
        uint _S588 = atomic_fetch_or_explicit((&kernelContext_89)->atom_0+(_S587 + 3U), 8U, memory_order_relaxed);
    }
    uint _S589 = _S548 + 2U * to_0;
    *((&kernelContext_89)->work_0+_S589) = base_0;
    *((&kernelContext_89)->work_0+(_S589 + 1U)) = kept_0;
    *((&kernelContext_89)->loads_0+k_22) = packed_float4(float4(length(f_total_0), f_total_0)) ;
    if(((&kernelContext_89)->params_0->load_model_0) == 1U)
    {
        uint _S590 = l_vec_0(&kernelContext_89);
        uint _S591 = _S590 + 2U * k_22;
        *((&kernelContext_89)->loads_0+_S591) = packed_float4(float4(f_total_0, 0.0f)) ;
        *((&kernelContext_89)->loads_0+(_S591 + 1U)) = packed_float4(float4(t_total_0, 0.0f)) ;
    }
    return;
}

[[kernel]] void fr_breaks(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], atomic<uint> device* atom_3 [[buffer(5)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], packed_float4 device* lp_3 [[buffer(24)]], packed_float4 device* ext_4 [[buffer(23)]], packed_float4 device* engine_contacts_3 [[buffer(25)]], CrushParams_0 constant* crush_3 [[buffer(28)]], packed_float4 device* crush_rw_3 [[buffer(27)]], packed_float4 device* crush_tab_3 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_90;
    (&kernelContext_90)->params_0 = params_3;
    (&kernelContext_90)->atom_0 = atom_3;
    (&kernelContext_90)->work_0 = work_3;
    (&kernelContext_90)->frames_0 = frames_3;
    (&kernelContext_90)->bodies_0 = bodies_3;
    (&kernelContext_90)->chunks_0 = chunks_3;
    (&kernelContext_90)->sort_0 = sort_3;
    (&kernelContext_90)->manifolds_0 = manifolds_3;
    (&kernelContext_90)->loads_0 = loads_3;
    (&kernelContext_90)->bonds_0 = bonds_3;
    (&kernelContext_90)->args_0 = args_3;
    (&kernelContext_90)->info_0 = info_3;
    (&kernelContext_90)->scan_0 = scan_3;
    (&kernelContext_90)->parts_0 = parts_3;
    (&kernelContext_90)->colliders_0 = colliders_3;
    (&kernelContext_90)->collider_info_0 = collider_info_3;
    (&kernelContext_90)->view_chunks_0 = view_chunks_3;
    (&kernelContext_90)->view_0 = view_3;
    (&kernelContext_90)->objects_0 = objects_3;
    (&kernelContext_90)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_90)->lines_0 = lines_3;
    (&kernelContext_90)->geometry_0 = geometry_3;
    (&kernelContext_90)->shots_0 = shots_3;
    (&kernelContext_90)->lp_0 = lp_3;
    (&kernelContext_90)->ext_0 = ext_4;
    (&kernelContext_90)->engine_contacts_0 = engine_contacts_3;
    (&kernelContext_90)->crush_0 = crush_3;
    (&kernelContext_90)->crush_rw_0 = crush_rw_3;
    (&kernelContext_90)->crush_tab_0 = crush_tab_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_90)->g_scan_0 = &g_scan_3;
    threadgroup array<float, int(64)> g_il_best_3;
    (&kernelContext_90)->g_il_best_0 = &g_il_best_3;
    uint i_7 = id_2.x;
    bool _S592;
    if(i_7 >= (params_3->bond_used_0))
    {
        _S592 = true;
    }
    else
    {
        uint _S593 = w_alive_0(&kernelContext_90);
        _S592 = (*((&kernelContext_90)->work_0+(_S593 + i_7))) == 0U;
    }
    if(_S592)
    {
        return;
    }
    uint4 _S594 = uint4(*((&kernelContext_90)->bonds_0+i_7)) ;
    uint _S595 = _S594.x;
    if((max((float4(*((&kernelContext_90)->loads_0+_S595)) ).x, (float4(*((&kernelContext_90)->loads_0+_S594.y)) ).x)) > (as_type<float>((_S594.z))))
    {
        uint _S596 = w_alive_0(&kernelContext_90);
        *((&kernelContext_90)->work_0+(_S596 + i_7)) = 0U;
        uint _S597 = a_dirty_0(&kernelContext_90);
        uint _S598 = body_of_0(_S595, &kernelContext_90);
        uint _S599 = atomic_fetch_or_explicit((&kernelContext_90)->atom_0+(_S597 + _S598), 1U, memory_order_relaxed);
        uint _S600 = a_counters_1(&kernelContext_90);
        uint _S601 = atomic_fetch_add_explicit((&kernelContext_90)->atom_0+_S600, 1U, memory_order_relaxed);
        uint _S602 = atomic_fetch_add_explicit((&kernelContext_90)->atom_0+(_S600 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_7, &kernelContext_90);
    }
    return;
}

[[kernel]] void fr_host_breaks(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], atomic<uint> device* atom_4 [[buffer(5)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], packed_float4 device* lp_4 [[buffer(24)]], packed_float4 device* ext_5 [[buffer(23)]], packed_float4 device* engine_contacts_4 [[buffer(25)]], CrushParams_0 constant* crush_4 [[buffer(28)]], packed_float4 device* crush_rw_4 [[buffer(27)]], packed_float4 device* crush_tab_4 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_91;
    (&kernelContext_91)->params_0 = params_4;
    (&kernelContext_91)->atom_0 = atom_4;
    (&kernelContext_91)->work_0 = work_4;
    (&kernelContext_91)->frames_0 = frames_4;
    (&kernelContext_91)->bodies_0 = bodies_4;
    (&kernelContext_91)->chunks_0 = chunks_4;
    (&kernelContext_91)->sort_0 = sort_4;
    (&kernelContext_91)->manifolds_0 = manifolds_4;
    (&kernelContext_91)->loads_0 = loads_4;
    (&kernelContext_91)->bonds_0 = bonds_4;
    (&kernelContext_91)->args_0 = args_4;
    (&kernelContext_91)->info_0 = info_4;
    (&kernelContext_91)->scan_0 = scan_4;
    (&kernelContext_91)->parts_0 = parts_4;
    (&kernelContext_91)->colliders_0 = colliders_4;
    (&kernelContext_91)->collider_info_0 = collider_info_4;
    (&kernelContext_91)->view_chunks_0 = view_chunks_4;
    (&kernelContext_91)->view_0 = view_4;
    (&kernelContext_91)->objects_0 = objects_4;
    (&kernelContext_91)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_91)->lines_0 = lines_4;
    (&kernelContext_91)->geometry_0 = geometry_4;
    (&kernelContext_91)->shots_0 = shots_4;
    (&kernelContext_91)->lp_0 = lp_4;
    (&kernelContext_91)->ext_0 = ext_5;
    (&kernelContext_91)->engine_contacts_0 = engine_contacts_4;
    (&kernelContext_91)->crush_0 = crush_4;
    (&kernelContext_91)->crush_rw_0 = crush_rw_4;
    (&kernelContext_91)->crush_tab_0 = crush_tab_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_91)->g_scan_0 = &g_scan_4;
    threadgroup array<float, int(64)> g_il_best_4;
    (&kernelContext_91)->g_il_best_0 = &g_il_best_4;
    uint i_8 = id_3.x;
    uint _S603 = w_host_breaks_0(&kernelContext_91);
    if(i_8 >= (*((&kernelContext_91)->work_0+_S603)))
    {
        return;
    }
    uint device* _S604 = (&kernelContext_91)->work_0+(_S603 + 1U + i_8);
    uint b_34 = *_S604;
    bool _S605;
    if((*_S604) >= ((&kernelContext_91)->params_0->bond_count_0))
    {
        _S605 = true;
    }
    else
    {
        uint _S606 = w_alive_0(&kernelContext_91);
        _S605 = (*((&kernelContext_91)->work_0+(_S606 + b_34))) == 0U;
    }
    if(_S605)
    {
        return;
    }
    uint _S607 = w_alive_0(&kernelContext_91);
    *((&kernelContext_91)->work_0+(_S607 + b_34)) = 0U;
    uint _S608 = a_dirty_0(&kernelContext_91);
    uint _S609 = body_of_0((uint4(*((&kernelContext_91)->bonds_0+b_34)) ).x, &kernelContext_91);
    uint _S610 = atomic_fetch_or_explicit((&kernelContext_91)->atom_0+(_S608 + _S609), 1U, memory_order_relaxed);
    uint _S611 = a_counters_1(&kernelContext_91);
    uint _S612 = atomic_fetch_add_explicit((&kernelContext_91)->atom_0+_S611, 1U, memory_order_relaxed);
    uint _S613 = atomic_fetch_add_explicit((&kernelContext_91)->atom_0+(_S611 + 1U), 1U, memory_order_relaxed);
    event_broken_0(b_34, &kernelContext_91);
    return;
}

[[kernel]] void fr_frame_begin(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], atomic<uint> device* atom_5 [[buffer(5)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], packed_float4 device* lp_5 [[buffer(24)]], packed_float4 device* ext_6 [[buffer(23)]], packed_float4 device* engine_contacts_5 [[buffer(25)]], CrushParams_0 constant* crush_5 [[buffer(28)]], packed_float4 device* crush_rw_5 [[buffer(27)]], packed_float4 device* crush_tab_5 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_92;
    (&kernelContext_92)->params_0 = params_5;
    (&kernelContext_92)->atom_0 = atom_5;
    (&kernelContext_92)->work_0 = work_5;
    (&kernelContext_92)->frames_0 = frames_5;
    (&kernelContext_92)->bodies_0 = bodies_5;
    (&kernelContext_92)->chunks_0 = chunks_5;
    (&kernelContext_92)->sort_0 = sort_5;
    (&kernelContext_92)->manifolds_0 = manifolds_5;
    (&kernelContext_92)->loads_0 = loads_5;
    (&kernelContext_92)->bonds_0 = bonds_5;
    (&kernelContext_92)->args_0 = args_5;
    (&kernelContext_92)->info_0 = info_5;
    (&kernelContext_92)->scan_0 = scan_5;
    (&kernelContext_92)->parts_0 = parts_5;
    (&kernelContext_92)->colliders_0 = colliders_5;
    (&kernelContext_92)->collider_info_0 = collider_info_5;
    (&kernelContext_92)->view_chunks_0 = view_chunks_5;
    (&kernelContext_92)->view_0 = view_5;
    (&kernelContext_92)->objects_0 = objects_5;
    (&kernelContext_92)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_92)->lines_0 = lines_5;
    (&kernelContext_92)->geometry_0 = geometry_5;
    (&kernelContext_92)->shots_0 = shots_5;
    (&kernelContext_92)->lp_0 = lp_5;
    (&kernelContext_92)->ext_0 = ext_6;
    (&kernelContext_92)->engine_contacts_0 = engine_contacts_5;
    (&kernelContext_92)->crush_0 = crush_5;
    (&kernelContext_92)->crush_rw_0 = crush_rw_5;
    (&kernelContext_92)->crush_tab_0 = crush_tab_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_92)->g_scan_0 = &g_scan_5;
    threadgroup array<float, int(64)> g_il_best_5;
    (&kernelContext_92)->g_il_best_0 = &g_il_best_5;
    uint _S614 = w_frame_0(&kernelContext_92);
    *((&kernelContext_92)->work_0+_S614) = 1U - *((&kernelContext_92)->work_0+_S614);
    uint _S615 = _S614 + 1U;
    *((&kernelContext_92)->work_0+_S615) = *((&kernelContext_92)->work_0+_S615) + 1U;
    uint _S616 = _S614 + 2U;
    *((&kernelContext_92)->work_0+_S616) = *((&kernelContext_92)->work_0+_S616) + 1U;
    return;
}

uint groups_of_0(uint n_3, uint size_0)
{
    uint _S617 = (n_3 + size_0 - 1U) / size_0;
    return _S617;
}

void set_args_0(uint at_8, uint x_3, KernelContext_0 thread* kernelContext_93)
{
    uint _S618 = 3U * at_8;
    *(kernelContext_93->args_0+_S618) = x_3;
    *(kernelContext_93->args_0+(_S618 + 1U)) = 1U;
    *(kernelContext_93->args_0+(_S618 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], atomic<uint> device* atom_6 [[buffer(5)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], packed_float4 device* lp_6 [[buffer(24)]], packed_float4 device* ext_7 [[buffer(23)]], packed_float4 device* engine_contacts_6 [[buffer(25)]], CrushParams_0 constant* crush_6 [[buffer(28)]], packed_float4 device* crush_rw_6 [[buffer(27)]], packed_float4 device* crush_tab_6 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_94;
    (&kernelContext_94)->params_0 = params_6;
    (&kernelContext_94)->atom_0 = atom_6;
    (&kernelContext_94)->work_0 = work_6;
    (&kernelContext_94)->frames_0 = frames_6;
    (&kernelContext_94)->bodies_0 = bodies_6;
    (&kernelContext_94)->chunks_0 = chunks_6;
    (&kernelContext_94)->sort_0 = sort_6;
    (&kernelContext_94)->manifolds_0 = manifolds_6;
    (&kernelContext_94)->loads_0 = loads_6;
    (&kernelContext_94)->bonds_0 = bonds_6;
    (&kernelContext_94)->args_0 = args_6;
    (&kernelContext_94)->info_0 = info_6;
    (&kernelContext_94)->scan_0 = scan_6;
    (&kernelContext_94)->parts_0 = parts_6;
    (&kernelContext_94)->colliders_0 = colliders_6;
    (&kernelContext_94)->collider_info_0 = collider_info_6;
    (&kernelContext_94)->view_chunks_0 = view_chunks_6;
    (&kernelContext_94)->view_0 = view_6;
    (&kernelContext_94)->objects_0 = objects_6;
    (&kernelContext_94)->shot_bodies_0 = shot_bodies_6;
    (&kernelContext_94)->lines_0 = lines_6;
    (&kernelContext_94)->geometry_0 = geometry_6;
    (&kernelContext_94)->shots_0 = shots_6;
    (&kernelContext_94)->lp_0 = lp_6;
    (&kernelContext_94)->ext_0 = ext_7;
    (&kernelContext_94)->engine_contacts_0 = engine_contacts_6;
    (&kernelContext_94)->crush_0 = crush_6;
    (&kernelContext_94)->crush_rw_0 = crush_rw_6;
    (&kernelContext_94)->crush_tab_0 = crush_tab_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_94)->g_scan_0 = &g_scan_6;
    threadgroup array<float, int(64)> g_il_best_6;
    (&kernelContext_94)->g_il_best_0 = &g_il_best_6;
    uint _S619 = a_counters_1(&kernelContext_94);
    uint _S620 = atomic_load_explicit((&kernelContext_94)->atom_0+_S619, memory_order_relaxed);
    bool run_0 = _S620 != 0U;
    atomic_store_explicit((&kernelContext_94)->atom_0+_S619, 0U, memory_order_relaxed);
    uint _S621 = _S619 + 5U;
    uint _S622 = frame_parity_0(&kernelContext_94);
    atomic_store_explicit((&kernelContext_94)->atom_0+(_S621 + _S622), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S623 = atomic_fetch_add_explicit((&kernelContext_94)->atom_0+(_S619 + 4U), 1U, memory_order_relaxed);
    }
    uint _S624;
    if(run_0)
    {
        uint _S625 = groups_of_0((&kernelContext_94)->params_0->chunk_used_0, 64U);
        _S624 = _S625;
    }
    else
    {
        _S624 = 0U;
    }
    set_args_0(0U, _S624, &kernelContext_94);
    if(run_0)
    {
        uint _S626 = groups_of_0((&kernelContext_94)->params_0->bond_used_0, 64U);
        _S624 = _S626;
    }
    else
    {
        _S624 = 0U;
    }
    set_args_0(1U, _S624, &kernelContext_94);
    if(run_0)
    {
        uint _S627 = groups_of_0((&kernelContext_94)->params_0->body_count_0, 64U);
        _S624 = _S627;
    }
    else
    {
        _S624 = 0U;
    }
    set_args_0(2U, _S624, &kernelContext_94);
    if(run_0)
    {
        uint _S628 = groups_of_0((&kernelContext_94)->params_0->chunk_used_0, 1024U);
        _S624 = _S628;
    }
    else
    {
        _S624 = 0U;
    }
    set_args_0(3U, _S624, &kernelContext_94);
    if(run_0)
    {
        uint _S629 = groups_of_0((&kernelContext_94)->params_0->body_count_0, 1024U);
        _S624 = _S629;
    }
    else
    {
        _S624 = 0U;
    }
    set_args_0(4U, _S624, &kernelContext_94);
    int _S630;
    if(run_0)
    {
        _S630 = int(1);
    }
    else
    {
        _S630 = int(0);
    }
    set_args_0(5U, uint(_S630), &kernelContext_94);
    return;
}

bool dirty_0(uint body_1, KernelContext_0 thread* kernelContext_95)
{
    uint _S631 = a_dirty_0(kernelContext_95);
    uint _S632 = atomic_load_explicit(kernelContext_95->atom_0+(_S631 + body_1), memory_order_relaxed);
    return _S632 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_96)
{
    return kernelContext_96->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_97)
{
    return 2U * kernelContext_97->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], atomic<uint> device* atom_7 [[buffer(5)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], packed_float4 device* lp_7 [[buffer(24)]], packed_float4 device* ext_8 [[buffer(23)]], packed_float4 device* engine_contacts_7 [[buffer(25)]], CrushParams_0 constant* crush_7 [[buffer(28)]], packed_float4 device* crush_rw_7 [[buffer(27)]], packed_float4 device* crush_tab_7 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_98;
    (&kernelContext_98)->params_0 = params_7;
    (&kernelContext_98)->atom_0 = atom_7;
    (&kernelContext_98)->work_0 = work_7;
    (&kernelContext_98)->frames_0 = frames_7;
    (&kernelContext_98)->bodies_0 = bodies_7;
    (&kernelContext_98)->chunks_0 = chunks_7;
    (&kernelContext_98)->sort_0 = sort_7;
    (&kernelContext_98)->manifolds_0 = manifolds_7;
    (&kernelContext_98)->loads_0 = loads_7;
    (&kernelContext_98)->bonds_0 = bonds_7;
    (&kernelContext_98)->args_0 = args_7;
    (&kernelContext_98)->info_0 = info_7;
    (&kernelContext_98)->scan_0 = scan_7;
    (&kernelContext_98)->parts_0 = parts_7;
    (&kernelContext_98)->colliders_0 = colliders_7;
    (&kernelContext_98)->collider_info_0 = collider_info_7;
    (&kernelContext_98)->view_chunks_0 = view_chunks_7;
    (&kernelContext_98)->view_0 = view_7;
    (&kernelContext_98)->objects_0 = objects_7;
    (&kernelContext_98)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_98)->lines_0 = lines_7;
    (&kernelContext_98)->geometry_0 = geometry_7;
    (&kernelContext_98)->shots_0 = shots_7;
    (&kernelContext_98)->lp_0 = lp_7;
    (&kernelContext_98)->ext_0 = ext_8;
    (&kernelContext_98)->engine_contacts_0 = engine_contacts_7;
    (&kernelContext_98)->crush_0 = crush_7;
    (&kernelContext_98)->crush_rw_0 = crush_rw_7;
    (&kernelContext_98)->crush_tab_0 = crush_tab_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_98)->g_scan_0 = &g_scan_7;
    threadgroup array<float, int(64)> g_il_best_7;
    (&kernelContext_98)->g_il_best_0 = &g_il_best_7;
    uint c_7 = id_6.x;
    bool _S633 = live_0(c_7, &kernelContext_98);
    bool _S634;
    if(!_S633)
    {
        _S634 = true;
    }
    else
    {
        uint _S635 = body_of_0(c_7, &kernelContext_98);
        bool _S636 = dirty_0(_S635, &kernelContext_98);
        _S634 = !_S636;
    }
    if(_S634)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_98)->atom_0+(a_parent_0() + c_7), c_7, memory_order_relaxed);
    uint _S637 = a_count_0(&kernelContext_98);
    atomic_store_explicit((&kernelContext_98)->atom_0+(_S637 + c_7), 0U, memory_order_relaxed);
    uint _S638 = a_anchor_0(&kernelContext_98);
    atomic_store_explicit((&kernelContext_98)->atom_0+(_S638 + c_7), 0U, memory_order_relaxed);
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
        uint _S639 = a_sums_0(&kernelContext_98);
        atomic_store_explicit((&kernelContext_98)->atom_0+(_S639 + 20U * c_7 + w_3), 0U, memory_order_relaxed);
        w_3 = w_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], atomic<uint> device* atom_8 [[buffer(5)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], packed_float4 device* lp_8 [[buffer(24)]], packed_float4 device* ext_9 [[buffer(23)]], packed_float4 device* engine_contacts_8 [[buffer(25)]], CrushParams_0 constant* crush_8 [[buffer(28)]], packed_float4 device* crush_rw_8 [[buffer(27)]], packed_float4 device* crush_tab_8 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_8;
    (&kernelContext_99)->atom_0 = atom_8;
    (&kernelContext_99)->work_0 = work_8;
    (&kernelContext_99)->frames_0 = frames_8;
    (&kernelContext_99)->bodies_0 = bodies_8;
    (&kernelContext_99)->chunks_0 = chunks_8;
    (&kernelContext_99)->sort_0 = sort_8;
    (&kernelContext_99)->manifolds_0 = manifolds_8;
    (&kernelContext_99)->loads_0 = loads_8;
    (&kernelContext_99)->bonds_0 = bonds_8;
    (&kernelContext_99)->args_0 = args_8;
    (&kernelContext_99)->info_0 = info_8;
    (&kernelContext_99)->scan_0 = scan_8;
    (&kernelContext_99)->parts_0 = parts_8;
    (&kernelContext_99)->colliders_0 = colliders_8;
    (&kernelContext_99)->collider_info_0 = collider_info_8;
    (&kernelContext_99)->view_chunks_0 = view_chunks_8;
    (&kernelContext_99)->view_0 = view_8;
    (&kernelContext_99)->objects_0 = objects_8;
    (&kernelContext_99)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_99)->lines_0 = lines_8;
    (&kernelContext_99)->geometry_0 = geometry_8;
    (&kernelContext_99)->shots_0 = shots_8;
    (&kernelContext_99)->lp_0 = lp_8;
    (&kernelContext_99)->ext_0 = ext_9;
    (&kernelContext_99)->engine_contacts_0 = engine_contacts_8;
    (&kernelContext_99)->crush_0 = crush_8;
    (&kernelContext_99)->crush_rw_0 = crush_rw_8;
    (&kernelContext_99)->crush_tab_0 = crush_tab_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_99)->g_scan_0 = &g_scan_8;
    threadgroup array<float, int(64)> g_il_best_8;
    (&kernelContext_99)->g_il_best_0 = &g_il_best_8;
    uint s_7 = id_7.x;
    bool _S640;
    if(s_7 >= (params_8->body_count_0))
    {
        _S640 = true;
    }
    else
    {
        bool _S641 = dirty_0(s_7, &kernelContext_99);
        _S640 = !_S641;
    }
    if(_S640)
    {
        return;
    }
    uint _S642 = a_best_count_0(&kernelContext_99);
    atomic_store_explicit((&kernelContext_99)->atom_0+(_S642 + s_7), 0U, memory_order_relaxed);
    uint _S643 = a_best_root_0(&kernelContext_99);
    atomic_store_explicit((&kernelContext_99)->atom_0+(_S643 + s_7), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_4, KernelContext_0 thread* kernelContext_100)
{
    uint _S644 = a_parent_0();
    uint _S645 = atomic_load_explicit(kernelContext_100->atom_0+(_S644 + x_4), memory_order_relaxed);
    uint p_11 = _S645;
    uint _S646 = x_4;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S647;
        if(guard_0 < 1048576U)
        {
            _S647 = p_11 != _S646;
        }
        else
        {
            _S647 = false;
        }
        if(_S647)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_100->atom_0+(_S644 + p_11), memory_order_relaxed);
        if(gp_0 != p_11)
        {
            uint _S648;
            {
            uint _metal_cas_comparand = p_11;
            atomic_compare_exchange_weak_explicit(kernelContext_100->atom_0+(_S644 + _S646), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S648 = _metal_cas_comparand;
            }
        }
        uint _S649 = guard_0 + 1U;
        _S646 = p_11;
        p_11 = gp_0;
        guard_0 = _S649;
    }
    return _S646;
}

[[kernel]] void fr_uf_union(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], atomic<uint> device* atom_9 [[buffer(5)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], packed_float4 device* lp_9 [[buffer(24)]], packed_float4 device* ext_10 [[buffer(23)]], packed_float4 device* engine_contacts_9 [[buffer(25)]], CrushParams_0 constant* crush_9 [[buffer(28)]], packed_float4 device* crush_rw_9 [[buffer(27)]], packed_float4 device* crush_tab_9 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_101;
    (&kernelContext_101)->params_0 = params_9;
    (&kernelContext_101)->atom_0 = atom_9;
    (&kernelContext_101)->work_0 = work_9;
    (&kernelContext_101)->frames_0 = frames_9;
    (&kernelContext_101)->bodies_0 = bodies_9;
    (&kernelContext_101)->chunks_0 = chunks_9;
    (&kernelContext_101)->sort_0 = sort_9;
    (&kernelContext_101)->manifolds_0 = manifolds_9;
    (&kernelContext_101)->loads_0 = loads_9;
    (&kernelContext_101)->bonds_0 = bonds_9;
    (&kernelContext_101)->args_0 = args_9;
    (&kernelContext_101)->info_0 = info_9;
    (&kernelContext_101)->scan_0 = scan_9;
    (&kernelContext_101)->parts_0 = parts_9;
    (&kernelContext_101)->colliders_0 = colliders_9;
    (&kernelContext_101)->collider_info_0 = collider_info_9;
    (&kernelContext_101)->view_chunks_0 = view_chunks_9;
    (&kernelContext_101)->view_0 = view_9;
    (&kernelContext_101)->objects_0 = objects_9;
    (&kernelContext_101)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_101)->lines_0 = lines_9;
    (&kernelContext_101)->geometry_0 = geometry_9;
    (&kernelContext_101)->shots_0 = shots_9;
    (&kernelContext_101)->lp_0 = lp_9;
    (&kernelContext_101)->ext_0 = ext_10;
    (&kernelContext_101)->engine_contacts_0 = engine_contacts_9;
    (&kernelContext_101)->crush_0 = crush_9;
    (&kernelContext_101)->crush_rw_0 = crush_rw_9;
    (&kernelContext_101)->crush_tab_0 = crush_tab_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_101)->g_scan_0 = &g_scan_9;
    threadgroup array<float, int(64)> g_il_best_9;
    (&kernelContext_101)->g_il_best_0 = &g_il_best_9;
    uint i_9 = id_8.x;
    bool _S650;
    if(i_9 >= (params_9->bond_used_0))
    {
        _S650 = true;
    }
    else
    {
        uint _S651 = w_alive_0(&kernelContext_101);
        _S650 = (*((&kernelContext_101)->work_0+(_S651 + i_9))) == 0U;
    }
    if(_S650)
    {
        return;
    }
    uint4 _S652 = uint4(*((&kernelContext_101)->bonds_0+i_9)) ;
    uint _S653 = _S652.x;
    uint _S654 = body_of_0(_S653, &kernelContext_101);
    bool _S655 = dirty_0(_S654, &kernelContext_101);
    if(!_S655)
    {
        return;
    }
    uint _S656 = _S652.y;
    uint a_6 = _S653;
    uint b_35 = _S656;
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
        uint _S657 = find_0(a_6, &kernelContext_101);
        uint _S658 = find_0(b_35, &kernelContext_101);
        if(_S657 == _S658)
        {
            return;
        }
        uint _S659 = max(_S657, _S658);
        uint _S660 = min(_S657, _S658);
        uint was_0;
        {
        uint _metal_cas_comparand = _S659;
        atomic_compare_exchange_weak_explicit((&kernelContext_101)->atom_0+(a_parent_0() + _S659), &_metal_cas_comparand, _S660, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S659)
        {
            return;
        }
        uint _S661 = guard_1 + 1U;
        a_6 = _S659;
        b_35 = _S660;
        guard_1 = _S661;
    }
    uint _S662 = a_counters_1(&kernelContext_101);
    uint _S663 = atomic_fetch_or_explicit((&kernelContext_101)->atom_0+(_S662 + 3U), 4U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], atomic<uint> device* atom_10 [[buffer(5)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], packed_float4 device* lp_10 [[buffer(24)]], packed_float4 device* ext_11 [[buffer(23)]], packed_float4 device* engine_contacts_10 [[buffer(25)]], CrushParams_0 constant* crush_10 [[buffer(28)]], packed_float4 device* crush_rw_10 [[buffer(27)]], packed_float4 device* crush_tab_10 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_102;
    (&kernelContext_102)->params_0 = params_10;
    (&kernelContext_102)->atom_0 = atom_10;
    (&kernelContext_102)->work_0 = work_10;
    (&kernelContext_102)->frames_0 = frames_10;
    (&kernelContext_102)->bodies_0 = bodies_10;
    (&kernelContext_102)->chunks_0 = chunks_10;
    (&kernelContext_102)->sort_0 = sort_10;
    (&kernelContext_102)->manifolds_0 = manifolds_10;
    (&kernelContext_102)->loads_0 = loads_10;
    (&kernelContext_102)->bonds_0 = bonds_10;
    (&kernelContext_102)->args_0 = args_10;
    (&kernelContext_102)->info_0 = info_10;
    (&kernelContext_102)->scan_0 = scan_10;
    (&kernelContext_102)->parts_0 = parts_10;
    (&kernelContext_102)->colliders_0 = colliders_10;
    (&kernelContext_102)->collider_info_0 = collider_info_10;
    (&kernelContext_102)->view_chunks_0 = view_chunks_10;
    (&kernelContext_102)->view_0 = view_10;
    (&kernelContext_102)->objects_0 = objects_10;
    (&kernelContext_102)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_102)->lines_0 = lines_10;
    (&kernelContext_102)->geometry_0 = geometry_10;
    (&kernelContext_102)->shots_0 = shots_10;
    (&kernelContext_102)->lp_0 = lp_10;
    (&kernelContext_102)->ext_0 = ext_11;
    (&kernelContext_102)->engine_contacts_0 = engine_contacts_10;
    (&kernelContext_102)->crush_0 = crush_10;
    (&kernelContext_102)->crush_rw_0 = crush_rw_10;
    (&kernelContext_102)->crush_tab_0 = crush_tab_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_102)->g_scan_0 = &g_scan_10;
    threadgroup array<float, int(64)> g_il_best_10;
    (&kernelContext_102)->g_il_best_0 = &g_il_best_10;
    uint c_8 = id_9.x;
    bool _S664 = live_0(c_8, &kernelContext_102);
    bool _S665;
    if(!_S664)
    {
        _S665 = true;
    }
    else
    {
        uint _S666 = body_of_0(c_8, &kernelContext_102);
        bool _S667 = dirty_0(_S666, &kernelContext_102);
        _S665 = !_S667;
    }
    if(_S665)
    {
        return;
    }
    uint _S668 = find_0(c_8, &kernelContext_102);
    uint _S669 = w_label_0(&kernelContext_102);
    *((&kernelContext_102)->work_0+(_S669 + c_8)) = _S668;
    uint _S670 = a_count_0(&kernelContext_102);
    uint _S671 = atomic_fetch_add_explicit((&kernelContext_102)->atom_0+(_S670 + _S668), 1U, memory_order_relaxed);
    if(((&kernelContext_102)->info_0[c_8] & 1U) != 0U)
    {
        uint _S672 = a_anchor_0(&kernelContext_102);
        uint _S673 = atomic_fetch_or_explicit((&kernelContext_102)->atom_0+(_S672 + _S668), 1U, memory_order_relaxed);
    }
    uint _S674 = 4U * c_8;
    float4 _S675 = float4(*((&kernelContext_102)->chunks_0+_S674)) ;
    float4 _S676 = float4(*((&kernelContext_102)->chunks_0+(_S674 + 1U))) ;
    float4 _S677 = float4(*((&kernelContext_102)->chunks_0+(_S674 + 2U))) ;
    float m_2 = _S675.w;
    float3 d_18 = _S675.xyz - (float4(*((&kernelContext_102)->chunks_0+4U * _S668)) ).xyz;
    float dd_0 = dot(d_18, d_18);
    uint _S678 = a_sums_0(&kernelContext_102);
    uint at_9 = _S678 + 20U * _S668;
    add64_0(at_9, m_2, (&kernelContext_102)->params_0->scale_m_0, &kernelContext_102);
    float _S679 = d_18.x;
    float _S680 = m_2 * _S679;
    add64_0(at_9 + 2U, _S680, (&kernelContext_102)->params_0->scale_s_0, &kernelContext_102);
    float _S681 = d_18.y;
    float _S682 = m_2 * _S681;
    add64_0(at_9 + 4U, _S682, (&kernelContext_102)->params_0->scale_s_0, &kernelContext_102);
    float _S683 = d_18.z;
    add64_0(at_9 + 6U, m_2 * _S683, (&kernelContext_102)->params_0->scale_s_0, &kernelContext_102);
    add64_0(at_9 + 8U, _S676.x + m_2 * (dd_0 - _S679 * _S679), (&kernelContext_102)->params_0->scale_j_0, &kernelContext_102);
    add64_0(at_9 + 10U, _S676.y + m_2 * (dd_0 - _S681 * _S681), (&kernelContext_102)->params_0->scale_j_0, &kernelContext_102);
    add64_0(at_9 + 12U, _S676.z + m_2 * (dd_0 - _S683 * _S683), (&kernelContext_102)->params_0->scale_j_0, &kernelContext_102);
    add64_0(at_9 + 14U, _S676.w - _S680 * _S681, (&kernelContext_102)->params_0->scale_j_0, &kernelContext_102);
    add64_0(at_9 + 16U, _S677.x - _S680 * _S683, (&kernelContext_102)->params_0->scale_j_0, &kernelContext_102);
    add64_0(at_9 + 18U, _S677.y - _S682 * _S683, (&kernelContext_102)->params_0->scale_j_0, &kernelContext_102);
    return;
}

bool is_root_0(uint c_9, KernelContext_0 thread* kernelContext_103)
{
    bool _S684 = live_0(c_9, kernelContext_103);
    bool _S685;
    if(_S684)
    {
        uint _S686 = body_of_0(c_9, kernelContext_103);
        bool _S687 = dirty_0(_S686, kernelContext_103);
        _S685 = _S687;
    }
    else
    {
        _S685 = false;
    }
    if(_S685)
    {
        uint _S688 = w_label_0(kernelContext_103);
        _S685 = (*(kernelContext_103->work_0+(_S688 + c_9))) == c_9;
    }
    else
    {
        _S685 = false;
    }
    return _S685;
}

[[kernel]] void fr_best_count(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_11 [[buffer(0)]], atomic<uint> device* atom_11 [[buffer(5)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], packed_float4 device* lp_11 [[buffer(24)]], packed_float4 device* ext_12 [[buffer(23)]], packed_float4 device* engine_contacts_11 [[buffer(25)]], CrushParams_0 constant* crush_11 [[buffer(28)]], packed_float4 device* crush_rw_11 [[buffer(27)]], packed_float4 device* crush_tab_11 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_104;
    (&kernelContext_104)->params_0 = params_11;
    (&kernelContext_104)->atom_0 = atom_11;
    (&kernelContext_104)->work_0 = work_11;
    (&kernelContext_104)->frames_0 = frames_11;
    (&kernelContext_104)->bodies_0 = bodies_11;
    (&kernelContext_104)->chunks_0 = chunks_11;
    (&kernelContext_104)->sort_0 = sort_11;
    (&kernelContext_104)->manifolds_0 = manifolds_11;
    (&kernelContext_104)->loads_0 = loads_11;
    (&kernelContext_104)->bonds_0 = bonds_11;
    (&kernelContext_104)->args_0 = args_11;
    (&kernelContext_104)->info_0 = info_11;
    (&kernelContext_104)->scan_0 = scan_11;
    (&kernelContext_104)->parts_0 = parts_11;
    (&kernelContext_104)->colliders_0 = colliders_11;
    (&kernelContext_104)->collider_info_0 = collider_info_11;
    (&kernelContext_104)->view_chunks_0 = view_chunks_11;
    (&kernelContext_104)->view_0 = view_11;
    (&kernelContext_104)->objects_0 = objects_11;
    (&kernelContext_104)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_104)->lines_0 = lines_11;
    (&kernelContext_104)->geometry_0 = geometry_11;
    (&kernelContext_104)->shots_0 = shots_11;
    (&kernelContext_104)->lp_0 = lp_11;
    (&kernelContext_104)->ext_0 = ext_12;
    (&kernelContext_104)->engine_contacts_0 = engine_contacts_11;
    (&kernelContext_104)->crush_0 = crush_11;
    (&kernelContext_104)->crush_rw_0 = crush_rw_11;
    (&kernelContext_104)->crush_tab_0 = crush_tab_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_104)->g_scan_0 = &g_scan_11;
    threadgroup array<float, int(64)> g_il_best_11;
    (&kernelContext_104)->g_il_best_0 = &g_il_best_11;
    uint c_10 = id_10.x;
    bool _S689 = is_root_0(c_10, &kernelContext_104);
    if(!_S689)
    {
        return;
    }
    uint _S690 = a_best_count_0(&kernelContext_104);
    uint _S691 = body_of_0(c_10, &kernelContext_104);
    atomic<uint> device* _S692 = (&kernelContext_104)->atom_0+(_S690 + _S691);
    uint _S693 = a_count_0(&kernelContext_104);
    uint _S694 = atomic_load_explicit((&kernelContext_104)->atom_0+(_S693 + c_10), memory_order_relaxed);
    uint _S695 = atomic_fetch_max_explicit(_S692, _S694, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_12 [[buffer(0)]], atomic<uint> device* atom_12 [[buffer(5)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], packed_float4 device* lp_12 [[buffer(24)]], packed_float4 device* ext_13 [[buffer(23)]], packed_float4 device* engine_contacts_12 [[buffer(25)]], CrushParams_0 constant* crush_12 [[buffer(28)]], packed_float4 device* crush_rw_12 [[buffer(27)]], packed_float4 device* crush_tab_12 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_105;
    (&kernelContext_105)->params_0 = params_12;
    (&kernelContext_105)->atom_0 = atom_12;
    (&kernelContext_105)->work_0 = work_12;
    (&kernelContext_105)->frames_0 = frames_12;
    (&kernelContext_105)->bodies_0 = bodies_12;
    (&kernelContext_105)->chunks_0 = chunks_12;
    (&kernelContext_105)->sort_0 = sort_12;
    (&kernelContext_105)->manifolds_0 = manifolds_12;
    (&kernelContext_105)->loads_0 = loads_12;
    (&kernelContext_105)->bonds_0 = bonds_12;
    (&kernelContext_105)->args_0 = args_12;
    (&kernelContext_105)->info_0 = info_12;
    (&kernelContext_105)->scan_0 = scan_12;
    (&kernelContext_105)->parts_0 = parts_12;
    (&kernelContext_105)->colliders_0 = colliders_12;
    (&kernelContext_105)->collider_info_0 = collider_info_12;
    (&kernelContext_105)->view_chunks_0 = view_chunks_12;
    (&kernelContext_105)->view_0 = view_12;
    (&kernelContext_105)->objects_0 = objects_12;
    (&kernelContext_105)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_105)->lines_0 = lines_12;
    (&kernelContext_105)->geometry_0 = geometry_12;
    (&kernelContext_105)->shots_0 = shots_12;
    (&kernelContext_105)->lp_0 = lp_12;
    (&kernelContext_105)->ext_0 = ext_13;
    (&kernelContext_105)->engine_contacts_0 = engine_contacts_12;
    (&kernelContext_105)->crush_0 = crush_12;
    (&kernelContext_105)->crush_rw_0 = crush_rw_12;
    (&kernelContext_105)->crush_tab_0 = crush_tab_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_105)->g_scan_0 = &g_scan_12;
    threadgroup array<float, int(64)> g_il_best_12;
    (&kernelContext_105)->g_il_best_0 = &g_il_best_12;
    uint c_11 = id_11.x;
    bool _S696 = is_root_0(c_11, &kernelContext_105);
    if(!_S696)
    {
        return;
    }
    uint _S697 = body_of_0(c_11, &kernelContext_105);
    uint _S698 = a_count_0(&kernelContext_105);
    uint _S699 = atomic_load_explicit((&kernelContext_105)->atom_0+(_S698 + c_11), memory_order_relaxed);
    uint _S700 = a_best_count_0(&kernelContext_105);
    uint _S701 = atomic_load_explicit((&kernelContext_105)->atom_0+(_S700 + _S697), memory_order_relaxed);
    if(_S699 == _S701)
    {
        uint _S702 = a_best_root_0(&kernelContext_105);
        uint _S703 = atomic_fetch_min_explicit((&kernelContext_105)->atom_0+(_S702 + _S697), c_11, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], atomic<uint> device* atom_13 [[buffer(5)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], packed_float4 device* lp_13 [[buffer(24)]], packed_float4 device* ext_14 [[buffer(23)]], packed_float4 device* engine_contacts_13 [[buffer(25)]], CrushParams_0 constant* crush_13 [[buffer(28)]], packed_float4 device* crush_rw_13 [[buffer(27)]], packed_float4 device* crush_tab_13 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_106;
    (&kernelContext_106)->params_0 = params_13;
    (&kernelContext_106)->atom_0 = atom_13;
    (&kernelContext_106)->work_0 = work_13;
    (&kernelContext_106)->frames_0 = frames_13;
    (&kernelContext_106)->bodies_0 = bodies_13;
    (&kernelContext_106)->chunks_0 = chunks_13;
    (&kernelContext_106)->sort_0 = sort_13;
    (&kernelContext_106)->manifolds_0 = manifolds_13;
    (&kernelContext_106)->loads_0 = loads_13;
    (&kernelContext_106)->bonds_0 = bonds_13;
    (&kernelContext_106)->args_0 = args_13;
    (&kernelContext_106)->info_0 = info_13;
    (&kernelContext_106)->scan_0 = scan_13;
    (&kernelContext_106)->parts_0 = parts_13;
    (&kernelContext_106)->colliders_0 = colliders_13;
    (&kernelContext_106)->collider_info_0 = collider_info_13;
    (&kernelContext_106)->view_chunks_0 = view_chunks_13;
    (&kernelContext_106)->view_0 = view_13;
    (&kernelContext_106)->objects_0 = objects_13;
    (&kernelContext_106)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_106)->lines_0 = lines_13;
    (&kernelContext_106)->geometry_0 = geometry_13;
    (&kernelContext_106)->shots_0 = shots_13;
    (&kernelContext_106)->lp_0 = lp_13;
    (&kernelContext_106)->ext_0 = ext_14;
    (&kernelContext_106)->engine_contacts_0 = engine_contacts_13;
    (&kernelContext_106)->crush_0 = crush_13;
    (&kernelContext_106)->crush_rw_0 = crush_rw_13;
    (&kernelContext_106)->crush_tab_0 = crush_tab_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_106)->g_scan_0 = &g_scan_13;
    threadgroup array<float, int(64)> g_il_best_13;
    (&kernelContext_106)->g_il_best_0 = &g_il_best_13;
    uint c_12 = id_12.x;
    if(c_12 >= (params_13->chunk_used_0))
    {
        return;
    }
    uint _S704 = w_new_0(&kernelContext_106);
    uint device* _S705 = (&kernelContext_106)->work_0+(_S704 + c_12);
    bool _S706 = is_root_0(c_12, &kernelContext_106);
    bool _S707;
    if(_S706)
    {
        uint _S708 = a_best_root_0(&kernelContext_106);
        uint _S709 = body_of_0(c_12, &kernelContext_106);
        uint _S710 = atomic_load_explicit((&kernelContext_106)->atom_0+(_S708 + _S709), memory_order_relaxed);
        _S707 = _S710 != c_12;
    }
    else
    {
        _S707 = false;
    }
    uint _S711;
    if(_S707)
    {
        _S711 = 1U;
    }
    else
    {
        _S711 = 0U;
    }
    *_S705 = _S711;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_14 [[buffer(0)]], atomic<uint> device* atom_14 [[buffer(5)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], packed_float4 device* lp_14 [[buffer(24)]], packed_float4 device* ext_15 [[buffer(23)]], packed_float4 device* engine_contacts_14 [[buffer(25)]], CrushParams_0 constant* crush_14 [[buffer(28)]], packed_float4 device* crush_rw_14 [[buffer(27)]], packed_float4 device* crush_tab_14 [[buffer(26)]])
{
    uint step_4;
    uint add_0;
    thread KernelContext_0 kernelContext_107;
    (&kernelContext_107)->params_0 = params_14;
    (&kernelContext_107)->atom_0 = atom_14;
    (&kernelContext_107)->work_0 = work_14;
    (&kernelContext_107)->frames_0 = frames_14;
    (&kernelContext_107)->bodies_0 = bodies_14;
    (&kernelContext_107)->chunks_0 = chunks_14;
    (&kernelContext_107)->sort_0 = sort_14;
    (&kernelContext_107)->manifolds_0 = manifolds_14;
    (&kernelContext_107)->loads_0 = loads_14;
    (&kernelContext_107)->bonds_0 = bonds_14;
    (&kernelContext_107)->args_0 = args_14;
    (&kernelContext_107)->info_0 = info_14;
    (&kernelContext_107)->scan_0 = scan_14;
    (&kernelContext_107)->parts_0 = parts_14;
    (&kernelContext_107)->colliders_0 = colliders_14;
    (&kernelContext_107)->collider_info_0 = collider_info_14;
    (&kernelContext_107)->view_chunks_0 = view_chunks_14;
    (&kernelContext_107)->view_0 = view_14;
    (&kernelContext_107)->objects_0 = objects_14;
    (&kernelContext_107)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_107)->lines_0 = lines_14;
    (&kernelContext_107)->geometry_0 = geometry_14;
    (&kernelContext_107)->shots_0 = shots_14;
    (&kernelContext_107)->lp_0 = lp_14;
    (&kernelContext_107)->ext_0 = ext_15;
    (&kernelContext_107)->engine_contacts_0 = engine_contacts_14;
    (&kernelContext_107)->crush_0 = crush_14;
    (&kernelContext_107)->crush_rw_0 = crush_rw_14;
    (&kernelContext_107)->crush_tab_0 = crush_tab_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_107)->g_scan_0 = &g_scan_14;
    threadgroup array<float, int(64)> g_il_best_14;
    (&kernelContext_107)->g_il_best_0 = &g_il_best_14;
    uint t_7 = thread_2.x;
    uint _S712 = group_0.x;
    uint _S713 = 1024U * _S712 + 4U * t_7;
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
        if((_S713 + j_5) < ((&kernelContext_107)->scan_0->n_0))
        {
            step_4 = *((&kernelContext_107)->work_0+((&kernelContext_107)->scan_0->in_base_0 + _S713 + j_5));
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
    (*(&kernelContext_107)->g_scan_0)[t_7] = sum_2;
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
            add_0 = (*(&kernelContext_107)->g_scan_0)[t_7 - step_4];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_107)->g_scan_0)[t_7] = (*(&kernelContext_107)->g_scan_0)[t_7] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_4 = step_4 << 1U;
    }
    uint _S714 = (*(&kernelContext_107)->g_scan_0)[t_7] - sum_2;
    j_5 = 0U;
    add_0 = _S714;
    for(;;)
    {
        if(j_5 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S713 + j_5) < ((&kernelContext_107)->scan_0->n_0))
        {
            *((&kernelContext_107)->work_0+((&kernelContext_107)->scan_0->out_base_0 + _S713 + j_5)) = add_0;
        }
        uint run_1 = add_0 + v_10[j_5];
        j_5 = j_5 + 1U;
        add_0 = run_1;
    }
    if(t_7 == 255U)
    {
        *((&kernelContext_107)->work_0+((&kernelContext_107)->scan_0->block_base_0 + _S712)) = (*(&kernelContext_107)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_3 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_15 [[buffer(0)]], atomic<uint> device* atom_15 [[buffer(5)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], packed_float4 device* lp_15 [[buffer(24)]], packed_float4 device* ext_16 [[buffer(23)]], packed_float4 device* engine_contacts_15 [[buffer(25)]], CrushParams_0 constant* crush_15 [[buffer(28)]], packed_float4 device* crush_rw_15 [[buffer(27)]], packed_float4 device* crush_tab_15 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_108;
    (&kernelContext_108)->params_0 = params_15;
    (&kernelContext_108)->atom_0 = atom_15;
    (&kernelContext_108)->work_0 = work_15;
    (&kernelContext_108)->frames_0 = frames_15;
    (&kernelContext_108)->bodies_0 = bodies_15;
    (&kernelContext_108)->chunks_0 = chunks_15;
    (&kernelContext_108)->sort_0 = sort_15;
    (&kernelContext_108)->manifolds_0 = manifolds_15;
    (&kernelContext_108)->loads_0 = loads_15;
    (&kernelContext_108)->bonds_0 = bonds_15;
    (&kernelContext_108)->args_0 = args_15;
    (&kernelContext_108)->info_0 = info_15;
    (&kernelContext_108)->scan_0 = scan_15;
    (&kernelContext_108)->parts_0 = parts_15;
    (&kernelContext_108)->colliders_0 = colliders_15;
    (&kernelContext_108)->collider_info_0 = collider_info_15;
    (&kernelContext_108)->view_chunks_0 = view_chunks_15;
    (&kernelContext_108)->view_0 = view_15;
    (&kernelContext_108)->objects_0 = objects_15;
    (&kernelContext_108)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_108)->lines_0 = lines_15;
    (&kernelContext_108)->geometry_0 = geometry_15;
    (&kernelContext_108)->shots_0 = shots_15;
    (&kernelContext_108)->lp_0 = lp_15;
    (&kernelContext_108)->ext_0 = ext_16;
    (&kernelContext_108)->engine_contacts_0 = engine_contacts_15;
    (&kernelContext_108)->crush_0 = crush_15;
    (&kernelContext_108)->crush_rw_0 = crush_rw_15;
    (&kernelContext_108)->crush_tab_0 = crush_tab_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_108)->g_scan_0 = &g_scan_15;
    threadgroup array<float, int(64)> g_il_best_15;
    (&kernelContext_108)->g_il_best_0 = &g_il_best_15;
    uint t_8 = thread_3.x;
    uint blocks_0 = (scan_15->n_0 + 1023U) / 1024U;
    bool _S715 = t_8 < blocks_0;
    uint v_11;
    if(_S715)
    {
        v_11 = *((&kernelContext_108)->work_0+((&kernelContext_108)->scan_0->block_base_0 + t_8));
    }
    else
    {
        v_11 = 0U;
    }
    (*(&kernelContext_108)->g_scan_0)[t_8] = v_11;
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
            add_1 = (*(&kernelContext_108)->g_scan_0)[t_8 - step_5];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_108)->g_scan_0)[t_8] = (*(&kernelContext_108)->g_scan_0)[t_8] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_5 = step_5 << 1U;
    }
    if(_S715)
    {
        *((&kernelContext_108)->work_0+((&kernelContext_108)->scan_0->block_base_0 + t_8)) = (*(&kernelContext_108)->g_scan_0)[t_8] - v_11;
    }
    if(t_8 == 1023U)
    {
        *((&kernelContext_108)->work_0+((&kernelContext_108)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_108)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], atomic<uint> device* atom_16 [[buffer(5)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], packed_float4 device* lp_16 [[buffer(24)]], packed_float4 device* ext_17 [[buffer(23)]], packed_float4 device* engine_contacts_16 [[buffer(25)]], CrushParams_0 constant* crush_16 [[buffer(28)]], packed_float4 device* crush_rw_16 [[buffer(27)]], packed_float4 device* crush_tab_16 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_109;
    (&kernelContext_109)->params_0 = params_16;
    (&kernelContext_109)->atom_0 = atom_16;
    (&kernelContext_109)->work_0 = work_16;
    (&kernelContext_109)->frames_0 = frames_16;
    (&kernelContext_109)->bodies_0 = bodies_16;
    (&kernelContext_109)->chunks_0 = chunks_16;
    (&kernelContext_109)->sort_0 = sort_16;
    (&kernelContext_109)->manifolds_0 = manifolds_16;
    (&kernelContext_109)->loads_0 = loads_16;
    (&kernelContext_109)->bonds_0 = bonds_16;
    (&kernelContext_109)->args_0 = args_16;
    (&kernelContext_109)->info_0 = info_16;
    (&kernelContext_109)->scan_0 = scan_16;
    (&kernelContext_109)->parts_0 = parts_16;
    (&kernelContext_109)->colliders_0 = colliders_16;
    (&kernelContext_109)->collider_info_0 = collider_info_16;
    (&kernelContext_109)->view_chunks_0 = view_chunks_16;
    (&kernelContext_109)->view_0 = view_16;
    (&kernelContext_109)->objects_0 = objects_16;
    (&kernelContext_109)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_109)->lines_0 = lines_16;
    (&kernelContext_109)->geometry_0 = geometry_16;
    (&kernelContext_109)->shots_0 = shots_16;
    (&kernelContext_109)->lp_0 = lp_16;
    (&kernelContext_109)->ext_0 = ext_17;
    (&kernelContext_109)->engine_contacts_0 = engine_contacts_16;
    (&kernelContext_109)->crush_0 = crush_16;
    (&kernelContext_109)->crush_rw_0 = crush_rw_16;
    (&kernelContext_109)->crush_tab_0 = crush_tab_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_109)->g_scan_0 = &g_scan_16;
    threadgroup array<float, int(64)> g_il_best_16;
    (&kernelContext_109)->g_il_best_0 = &g_il_best_16;
    uint i_10 = id_13.x;
    if(i_10 >= (scan_16->n_0))
    {
        return;
    }
    uint device* _S716 = (&kernelContext_109)->work_0+((&kernelContext_109)->scan_0->out_base_0 + i_10);
    *_S716 = *_S716 + *((&kernelContext_109)->work_0+((&kernelContext_109)->scan_0->block_base_0 + i_10 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], atomic<uint> device* atom_17 [[buffer(5)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], packed_float4 device* lp_17 [[buffer(24)]], packed_float4 device* ext_18 [[buffer(23)]], packed_float4 device* engine_contacts_17 [[buffer(25)]], CrushParams_0 constant* crush_17 [[buffer(28)]], packed_float4 device* crush_rw_17 [[buffer(27)]], packed_float4 device* crush_tab_17 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_110;
    (&kernelContext_110)->params_0 = params_17;
    (&kernelContext_110)->atom_0 = atom_17;
    (&kernelContext_110)->work_0 = work_17;
    (&kernelContext_110)->frames_0 = frames_17;
    (&kernelContext_110)->bodies_0 = bodies_17;
    (&kernelContext_110)->chunks_0 = chunks_17;
    (&kernelContext_110)->sort_0 = sort_17;
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
    uint s_8 = id_14.x;
    bool _S717;
    if(s_8 >= (params_17->body_count_0))
    {
        _S717 = true;
    }
    else
    {
        uint _S718 = w_free_0(&kernelContext_110);
        _S717 = (*((&kernelContext_110)->work_0+(_S718 + s_8))) == 0U;
    }
    if(_S717)
    {
        return;
    }
    uint _S719 = w_free_list_0(&kernelContext_110);
    uint _S720 = w_free_rank_0(&kernelContext_110);
    *((&kernelContext_110)->work_0+(_S719 + *((&kernelContext_110)->work_0+(_S720 + s_8)))) = s_8;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_111)
{
    uint _S721 = w_free_list_0(kernelContext_111);
    return _S721 + kernelContext_111->params_0->body_count_0;
}

uint scan_new_blocks_1(KernelContext_0 thread* kernelContext_112)
{
    uint _S722 = w_free_list_0(kernelContext_112);
    return _S722 + kernelContext_112->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_113)
{
    uint _S723 = scan_new_blocks_1(kernelContext_113);
    return _S723 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_114)
{
    uint blocks_1 = (kernelContext_114->params_0->body_count_0 + 1023U) / 1024U;
    uint _S724 = scan_free_blocks_0(kernelContext_114);
    return *(kernelContext_114->work_0+(_S724 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], atomic<uint> device* atom_18 [[buffer(5)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], packed_float4 device* lp_18 [[buffer(24)]], packed_float4 device* ext_19 [[buffer(23)]], packed_float4 device* engine_contacts_18 [[buffer(25)]], CrushParams_0 constant* crush_18 [[buffer(28)]], packed_float4 device* crush_rw_18 [[buffer(27)]], packed_float4 device* crush_tab_18 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_115;
    (&kernelContext_115)->params_0 = params_18;
    (&kernelContext_115)->atom_0 = atom_18;
    (&kernelContext_115)->work_0 = work_18;
    (&kernelContext_115)->frames_0 = frames_18;
    (&kernelContext_115)->bodies_0 = bodies_18;
    (&kernelContext_115)->chunks_0 = chunks_18;
    (&kernelContext_115)->sort_0 = sort_18;
    (&kernelContext_115)->manifolds_0 = manifolds_18;
    (&kernelContext_115)->loads_0 = loads_18;
    (&kernelContext_115)->bonds_0 = bonds_18;
    (&kernelContext_115)->args_0 = args_18;
    (&kernelContext_115)->info_0 = info_18;
    (&kernelContext_115)->scan_0 = scan_18;
    (&kernelContext_115)->parts_0 = parts_18;
    (&kernelContext_115)->colliders_0 = colliders_18;
    (&kernelContext_115)->collider_info_0 = collider_info_18;
    (&kernelContext_115)->view_chunks_0 = view_chunks_18;
    (&kernelContext_115)->view_0 = view_18;
    (&kernelContext_115)->objects_0 = objects_18;
    (&kernelContext_115)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_115)->lines_0 = lines_18;
    (&kernelContext_115)->geometry_0 = geometry_18;
    (&kernelContext_115)->shots_0 = shots_18;
    (&kernelContext_115)->lp_0 = lp_18;
    (&kernelContext_115)->ext_0 = ext_19;
    (&kernelContext_115)->engine_contacts_0 = engine_contacts_18;
    (&kernelContext_115)->crush_0 = crush_18;
    (&kernelContext_115)->crush_rw_0 = crush_rw_18;
    (&kernelContext_115)->crush_tab_0 = crush_tab_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_115)->g_scan_0 = &g_scan_18;
    threadgroup array<float, int(64)> g_il_best_18;
    (&kernelContext_115)->g_il_best_0 = &g_il_best_18;
    uint c_13 = id_15.x;
    bool _S725 = is_root_0(c_13, &kernelContext_115);
    if(!_S725)
    {
        return;
    }
    uint _S726 = body_of_0(c_13, &kernelContext_115);
    uint _S727 = a_best_root_0(&kernelContext_115);
    uint _S728 = atomic_load_explicit((&kernelContext_115)->atom_0+(_S727 + _S726), memory_order_relaxed);
    uint slot_1;
    if(_S728 != c_13)
    {
        uint _S729 = w_rank_0(&kernelContext_115);
        uint device* _S730 = (&kernelContext_115)->work_0+(_S729 + c_13);
        uint k_23 = *_S730;
        uint _S731 = free_total_0(&kernelContext_115);
        if((*_S730) < _S731)
        {
            uint _S732 = w_free_list_0(&kernelContext_115);
            uint device* _S733 = (&kernelContext_115)->work_0+(_S732 + k_23);
            uint slot_2 = *_S733;
            uint _S734 = w_free_0(&kernelContext_115);
            *((&kernelContext_115)->work_0+(_S734 + *_S733)) = 0U;
            uint _S735 = a_counters_1(&kernelContext_115);
            uint _S736 = atomic_fetch_add_explicit((&kernelContext_115)->atom_0+(_S735 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S737 = a_counters_1(&kernelContext_115);
            uint _S738 = atomic_fetch_or_explicit((&kernelContext_115)->atom_0+(_S737 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S726;
        }
    }
    else
    {
        slot_1 = _S726;
    }
    uint _S739 = w_slot_0(&kernelContext_115);
    *((&kernelContext_115)->work_0+(_S739 + c_13)) = slot_1;
    return;
}

float read64_1(uint at_10, float scale_2, KernelContext_0 thread* kernelContext_116)
{
    uint _S740 = atomic_load_explicit(kernelContext_116->atom_0+(at_10 + 1U), memory_order_relaxed);
    int hi_3 = (as_type<int>((_S740)));
    uint lo_3 = atomic_load_explicit(kernelContext_116->atom_0+at_10, memory_order_relaxed);
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
    array<float, int(3)> _S741 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S742 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S743 = { { r2_2.x, r2_2.y, r2_2.z } };
    thread array<array<float, int(3)>, int(3)> a_7;
    a_7[int(0)] = _S741;
    a_7[int(1)] = _S742;
    a_7[int(2)] = _S743;
    array<float, int(3)> _S744 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S745 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S746 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_12;
    v_12[int(0)] = _S744;
    v_12[int(1)] = _S745;
    v_12[int(2)] = _S746;
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
        float c_14 = cos(theta_0);
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
            bool _S747;
            if(k_24 != p_12)
            {
                _S747 = k_24 != q_14;
            }
            else
            {
                _S747 = false;
            }
            if(_S747)
            {
                float kp_0 = a_7[k_24][p_12];
                float kq_0 = a_7[k_24][q_14];
                float _S748 = c_14 * a_7[k_24][p_12] - s_9 * a_7[k_24][q_14];
                a_7[k_24][p_12] = _S748;
                a_7[p_12][k_24] = _S748;
                float _S749 = s_9 * kp_0 + c_14 * kq_0;
                a_7[k_24][q_14] = _S749;
                a_7[q_14][k_24] = _S749;
            }
            k_24 = k_24 + 1U;
        }
        float _S750 = c_14 * c_14;
        float _S751 = 2.0f * s_9 * c_14 * apq_0;
        float _S752 = s_9 * s_9;
        a_7[p_12][p_12] = _S750 * app_0 - _S751 + _S752 * aqq_0;
        a_7[q_14][q_14] = _S752 * app_0 + _S751 + _S750 * aqq_0;
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
            v_12[row_0][p_12] = c_14 * v_12[row_0][p_12] - s_9 * v_12[row_0][q_14];
            v_12[row_0][q_14] = s_9 * kp_1 + c_14 * kq_1;
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
    thread Quat_0 _S753 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S754 = normalized_0(&_S753);
    *rot_0 = _S754;
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
    thread Quat_0 _S755 = axes_1;
    Quat_0 _S756 = quat_mul_0(&_S755, q_15);
    thread Quat_0 _S757 = _S756;
    Quat_0 _S758 = normalized_0(&_S757);
    return _S758;
}

uint stage_0(uint c_15, KernelContext_0 thread* kernelContext_117)
{
    return 2U * kernelContext_117->params_0->body_count_0 + 9U * c_15;
}

float4 quat_vec_0(const Quat_0 thread* q_16)
{
    return float4(q_16->x_0, q_16->y_0, q_16->z_0, q_16->w_0);
}

uint frame_step_0(KernelContext_0 thread* kernelContext_118)
{
    uint _S759 = w_frame_0(kernelContext_118);
    return *(kernelContext_118->work_0+(_S759 + 1U));
}

[[kernel]] void fr_piece(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], atomic<uint> device* atom_19 [[buffer(5)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], packed_float4 device* lp_19 [[buffer(24)]], packed_float4 device* ext_20 [[buffer(23)]], packed_float4 device* engine_contacts_19 [[buffer(25)]], CrushParams_0 constant* crush_19 [[buffer(28)]], packed_float4 device* crush_rw_19 [[buffer(27)]], packed_float4 device* crush_tab_19 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_119;
    (&kernelContext_119)->params_0 = params_19;
    (&kernelContext_119)->atom_0 = atom_19;
    (&kernelContext_119)->work_0 = work_19;
    (&kernelContext_119)->frames_0 = frames_19;
    (&kernelContext_119)->bodies_0 = bodies_19;
    (&kernelContext_119)->chunks_0 = chunks_19;
    (&kernelContext_119)->sort_0 = sort_19;
    (&kernelContext_119)->manifolds_0 = manifolds_19;
    (&kernelContext_119)->loads_0 = loads_19;
    (&kernelContext_119)->bonds_0 = bonds_19;
    (&kernelContext_119)->args_0 = args_19;
    (&kernelContext_119)->info_0 = info_19;
    (&kernelContext_119)->scan_0 = scan_19;
    (&kernelContext_119)->parts_0 = parts_19;
    (&kernelContext_119)->colliders_0 = colliders_19;
    (&kernelContext_119)->collider_info_0 = collider_info_19;
    (&kernelContext_119)->view_chunks_0 = view_chunks_19;
    (&kernelContext_119)->view_0 = view_19;
    (&kernelContext_119)->objects_0 = objects_19;
    (&kernelContext_119)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_119)->lines_0 = lines_19;
    (&kernelContext_119)->geometry_0 = geometry_19;
    (&kernelContext_119)->shots_0 = shots_19;
    (&kernelContext_119)->lp_0 = lp_19;
    (&kernelContext_119)->ext_0 = ext_20;
    (&kernelContext_119)->engine_contacts_0 = engine_contacts_19;
    (&kernelContext_119)->crush_0 = crush_19;
    (&kernelContext_119)->crush_rw_0 = crush_rw_19;
    (&kernelContext_119)->crush_tab_0 = crush_tab_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_119)->g_scan_0 = &g_scan_19;
    threadgroup array<float, int(64)> g_il_best_19;
    (&kernelContext_119)->g_il_best_0 = &g_il_best_19;
    uint c_16 = id_16.x;
    bool _S760 = is_root_0(c_16, &kernelContext_119);
    if(!_S760)
    {
        return;
    }
    uint _S761 = body_of_0(c_16, &kernelContext_119);
    uint pb_0 = 7U * _S761;
    uint _S762 = a_sums_0(&kernelContext_119);
    uint at_11 = _S762 + 20U * c_16;
    float _S763 = read64_0(at_11, (&kernelContext_119)->params_0->scale_m_0, &kernelContext_119);
    float _S764 = read64_0(at_11 + 2U, (&kernelContext_119)->params_0->scale_s_0, &kernelContext_119);
    float _S765 = read64_0(at_11 + 4U, (&kernelContext_119)->params_0->scale_s_0, &kernelContext_119);
    float _S766 = read64_0(at_11 + 6U, (&kernelContext_119)->params_0->scale_s_0, &kernelContext_119);
    float3 d_19 = float3(_S764, _S765, _S766) / float3(max(_S763, 1.00000000317107685e-30f)) ;
    float _S767 = read64_0(at_11 + 8U, (&kernelContext_119)->params_0->scale_j_0, &kernelContext_119);
    float _S768 = read64_0(at_11 + 10U, (&kernelContext_119)->params_0->scale_j_0, &kernelContext_119);
    float _S769 = read64_0(at_11 + 12U, (&kernelContext_119)->params_0->scale_j_0, &kernelContext_119);
    float _S770 = read64_0(at_11 + 14U, (&kernelContext_119)->params_0->scale_j_0, &kernelContext_119);
    float _S771 = read64_0(at_11 + 16U, (&kernelContext_119)->params_0->scale_j_0, &kernelContext_119);
    float _S772 = read64_0(at_11 + 18U, (&kernelContext_119)->params_0->scale_j_0, &kernelContext_119);
    float dd_1 = dot(d_19, d_19);
    float _S773 = d_19.x;
    float _S774 = _S763 * _S773;
    float _S775 = d_19.y;
    float _S776 = _S770 + _S774 * _S775;
    float _S777 = d_19.z;
    float _S778 = _S771 + _S774 * _S777;
    float _S779 = _S772 + _S763 * _S775 * _S777;
    float3 com_1 = (float4(*((&kernelContext_119)->chunks_0+4U * c_16)) ).xyz + d_19;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S767 - _S763 * (dd_1 - _S773 * _S773), _S776, _S778), float3(_S776, _S768 - _S763 * (dd_1 - _S775 * _S775), _S779), float3(_S778, _S779, _S769 - _S763 * (dd_1 - _S777 * _S777)), &principal_0, &moments_1);
    uint _S780 = a_anchor_0(&kernelContext_119);
    uint _S781 = atomic_load_explicit((&kernelContext_119)->atom_0+(_S780 + c_16), memory_order_relaxed);
    bool anchored_0 = _S781 != 0U;
    thread Quat_0 q_17;
    thread float3 origin_4;
    structure_pose_0(_S761, pb_0, &q_17, &origin_4, &kernelContext_119);
    float3 p_parent_0 = (float4(*((&kernelContext_119)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_119)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_119)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S782 = q_17;
    float3 _S783 = rotate_0(&_S782, com_1);
    float3 pos_0 = to_rubble_0(origin_4 + _S783);
    thread Quat_0 _S784 = q_17;
    thread Quat_0 _S785 = principal_0;
    Quat_0 _S786 = quat_mul_0(&_S784, &_S785);
    thread Quat_0 _S787 = _S786;
    Quat_0 _S788 = normalized_0(&_S787);
    thread Quat_0 _S789 = _S788;
    Quat_0 _S790 = rot_to_rubble_0(&_S789);
    bool _S791;
    if(anchored_0)
    {
        _S791 = true;
    }
    else
    {
        _S791 = !(_S763 > 0.0f);
    }
    float inv_mass_3;
    if(_S791)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / _S763;
    }
    float3 _S792 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S793;
        if((moments_1.x) > 0.0f)
        {
            _S793 = 1.0f / moments_1.x;
        }
        else
        {
            _S793 = 0.0f;
        }
        float _S794;
        if((moments_1.y) > 0.0f)
        {
            _S794 = 1.0f / moments_1.y;
        }
        else
        {
            _S794 = 0.0f;
        }
        float _S795;
        if((moments_1.z) > 0.0f)
        {
            _S795 = 1.0f / moments_1.z;
        }
        else
        {
            _S795 = 0.0f;
        }
        inv_inertia_0 = float3(_S793, _S794, _S795);
    }
    else
    {
        inv_inertia_0 = _S792;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S792;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S792;
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
    uint _S796 = stage_0(c_16, &kernelContext_119);
    *((&kernelContext_119)->frames_0+_S796) = packed_float4(float4(pos_0, inv_mass_3)) ;
    packed_float4 device* _S797 = (&kernelContext_119)->frames_0+(_S796 + 1U);
    thread Quat_0 _S798 = _S790;
    float4 _S799 = quat_vec_0(&_S798);
    *_S797 = packed_float4(_S799) ;
    *((&kernelContext_119)->frames_0+(_S796 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_119)->frames_0+(_S796 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    packed_float4 device* _S800 = (&kernelContext_119)->frames_0+(_S796 + 4U);
    uint _S801 = frame_step_0(&kernelContext_119);
    *_S800 = packed_float4(float4(spin_0, (as_type<float>((_S801))))) ;
    *((&kernelContext_119)->frames_0+(_S796 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_119)->frames_0+(_S796 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_119)->frames_0+(_S796 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_119)->frames_0+2U * _S761)) ).w)) ;
    packed_float4 device* _S802 = (&kernelContext_119)->frames_0+(_S796 + 8U);
    thread Quat_0 _S803 = principal_0;
    float4 _S804 = quat_vec_0(&_S803);
    *_S802 = packed_float4(_S804) ;
    return;
}

uint f_events_0(KernelContext_0 thread* kernelContext_120)
{
    return 2U * kernelContext_120->params_0->body_count_0 + 9U * kernelContext_120->params_0->chunk_count_0 + 3U * kernelContext_120->params_0->body_count_0;
}

[[kernel]] void fr_commit(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], atomic<uint> device* atom_20 [[buffer(5)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], packed_float4 device* lp_20 [[buffer(24)]], packed_float4 device* ext_21 [[buffer(23)]], packed_float4 device* engine_contacts_20 [[buffer(25)]], CrushParams_0 constant* crush_20 [[buffer(28)]], packed_float4 device* crush_rw_20 [[buffer(27)]], packed_float4 device* crush_tab_20 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_121;
    (&kernelContext_121)->params_0 = params_20;
    (&kernelContext_121)->atom_0 = atom_20;
    (&kernelContext_121)->work_0 = work_20;
    (&kernelContext_121)->frames_0 = frames_20;
    (&kernelContext_121)->bodies_0 = bodies_20;
    (&kernelContext_121)->chunks_0 = chunks_20;
    (&kernelContext_121)->sort_0 = sort_20;
    (&kernelContext_121)->manifolds_0 = manifolds_20;
    (&kernelContext_121)->loads_0 = loads_20;
    (&kernelContext_121)->bonds_0 = bonds_20;
    (&kernelContext_121)->args_0 = args_20;
    (&kernelContext_121)->info_0 = info_20;
    (&kernelContext_121)->scan_0 = scan_20;
    (&kernelContext_121)->parts_0 = parts_20;
    (&kernelContext_121)->colliders_0 = colliders_20;
    (&kernelContext_121)->collider_info_0 = collider_info_20;
    (&kernelContext_121)->view_chunks_0 = view_chunks_20;
    (&kernelContext_121)->view_0 = view_20;
    (&kernelContext_121)->objects_0 = objects_20;
    (&kernelContext_121)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_121)->lines_0 = lines_20;
    (&kernelContext_121)->geometry_0 = geometry_20;
    (&kernelContext_121)->shots_0 = shots_20;
    (&kernelContext_121)->lp_0 = lp_20;
    (&kernelContext_121)->ext_0 = ext_21;
    (&kernelContext_121)->engine_contacts_0 = engine_contacts_20;
    (&kernelContext_121)->crush_0 = crush_20;
    (&kernelContext_121)->crush_rw_0 = crush_rw_20;
    (&kernelContext_121)->crush_tab_0 = crush_tab_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_121)->g_scan_0 = &g_scan_20;
    threadgroup array<float, int(64)> g_il_best_20;
    (&kernelContext_121)->g_il_best_0 = &g_il_best_20;
    uint c_17 = id_17.x;
    bool _S805 = is_root_0(c_17, &kernelContext_121);
    if(!_S805)
    {
        return;
    }
    uint _S806 = w_slot_0(&kernelContext_121);
    uint slot_3 = *((&kernelContext_121)->work_0+(_S806 + c_17));
    uint _S807 = stage_0(c_17, &kernelContext_121);
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
            *((&kernelContext_121)->bodies_0+(7U * (slot_3 + region_5 * (&kernelContext_121)->params_0->max_bodies_0) + w_4)) = packed_float4(float4(*((&kernelContext_121)->frames_0+(_S807 + w_4))) ) ;
            w_4 = w_4 + 1U;
        }
        region_5 = region_5 + 1U;
    }
    uint _S808 = 2U * slot_3;
    uint _S809 = _S807 + 7U;
    *((&kernelContext_121)->frames_0+_S808) = packed_float4(float4(*((&kernelContext_121)->frames_0+_S809)) ) ;
    uint _S810 = _S807 + 8U;
    *((&kernelContext_121)->frames_0+(_S808 + 1U)) = packed_float4(float4(*((&kernelContext_121)->frames_0+_S810)) ) ;
    if(((&kernelContext_121)->params_0->ev_frames_cap_0) != 0U)
    {
        uint _S811 = a_counters_1(&kernelContext_121);
        uint i_11 = atomic_fetch_add_explicit((&kernelContext_121)->atom_0+(_S811 + 22U), 1U, memory_order_relaxed);
        if(i_11 < ((&kernelContext_121)->params_0->ev_frames_cap_0))
        {
            uint _S812 = f_events_0(&kernelContext_121);
            uint at_12 = _S812 + 3U * i_11;
            *((&kernelContext_121)->frames_0+at_12) = packed_float4(float4((as_type<float>((slot_3))), 0.0f, 0.0f, 0.0f)) ;
            *((&kernelContext_121)->frames_0+(at_12 + 1U)) = packed_float4(float4(*((&kernelContext_121)->frames_0+_S809)) ) ;
            *((&kernelContext_121)->frames_0+(at_12 + 2U)) = packed_float4(float4(*((&kernelContext_121)->frames_0+_S810)) ) ;
        }
        else
        {
            uint _S813 = atomic_fetch_or_explicit((&kernelContext_121)->atom_0+(_S811 + 3U), 128U, memory_order_relaxed);
        }
    }
    return;
}

uint a_moved_0(KernelContext_0 thread* kernelContext_122)
{
    uint _S814 = a_events_0(kernelContext_122);
    return _S814 + kernelContext_122->params_0->ev_broken_cap_0;
}

[[kernel]] void fr_reparent(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], atomic<uint> device* atom_21 [[buffer(5)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], packed_float4 device* lp_21 [[buffer(24)]], packed_float4 device* ext_22 [[buffer(23)]], packed_float4 device* engine_contacts_21 [[buffer(25)]], CrushParams_0 constant* crush_21 [[buffer(28)]], packed_float4 device* crush_rw_21 [[buffer(27)]], packed_float4 device* crush_tab_21 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_123;
    (&kernelContext_123)->params_0 = params_21;
    (&kernelContext_123)->atom_0 = atom_21;
    (&kernelContext_123)->work_0 = work_21;
    (&kernelContext_123)->frames_0 = frames_21;
    (&kernelContext_123)->bodies_0 = bodies_21;
    (&kernelContext_123)->chunks_0 = chunks_21;
    (&kernelContext_123)->sort_0 = sort_21;
    (&kernelContext_123)->manifolds_0 = manifolds_21;
    (&kernelContext_123)->loads_0 = loads_21;
    (&kernelContext_123)->bonds_0 = bonds_21;
    (&kernelContext_123)->args_0 = args_21;
    (&kernelContext_123)->info_0 = info_21;
    (&kernelContext_123)->scan_0 = scan_21;
    (&kernelContext_123)->parts_0 = parts_21;
    (&kernelContext_123)->colliders_0 = colliders_21;
    (&kernelContext_123)->collider_info_0 = collider_info_21;
    (&kernelContext_123)->view_chunks_0 = view_chunks_21;
    (&kernelContext_123)->view_0 = view_21;
    (&kernelContext_123)->objects_0 = objects_21;
    (&kernelContext_123)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_123)->lines_0 = lines_21;
    (&kernelContext_123)->geometry_0 = geometry_21;
    (&kernelContext_123)->shots_0 = shots_21;
    (&kernelContext_123)->lp_0 = lp_21;
    (&kernelContext_123)->ext_0 = ext_22;
    (&kernelContext_123)->engine_contacts_0 = engine_contacts_21;
    (&kernelContext_123)->crush_0 = crush_21;
    (&kernelContext_123)->crush_rw_0 = crush_rw_21;
    (&kernelContext_123)->crush_tab_0 = crush_tab_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_123)->g_scan_0 = &g_scan_21;
    threadgroup array<float, int(64)> g_il_best_21;
    (&kernelContext_123)->g_il_best_0 = &g_il_best_21;
    uint c_18 = id_18.x;
    bool _S815 = live_0(c_18, &kernelContext_123);
    bool _S816;
    if(!_S815)
    {
        _S816 = true;
    }
    else
    {
        uint _S817 = body_of_0(c_18, &kernelContext_123);
        bool _S818 = dirty_0(_S817, &kernelContext_123);
        _S816 = !_S818;
    }
    if(_S816)
    {
        return;
    }
    uint _S819 = w_label_0(&kernelContext_123);
    uint device* _S820 = (&kernelContext_123)->work_0+(_S819 + c_18);
    uint r_21 = *_S820;
    uint _S821 = w_slot_0(&kernelContext_123);
    uint device* _S822 = (&kernelContext_123)->work_0+(_S821 + *_S820);
    uint slot_4 = *_S822;
    uint _S823 = body_of_0(c_18, &kernelContext_123);
    if((*_S822) != _S823)
    {
        _S816 = ((&kernelContext_123)->params_0->ev_moved_cap_0) != 0U;
    }
    else
    {
        _S816 = false;
    }
    if(_S816)
    {
        uint _S824 = a_counters_1(&kernelContext_123);
        uint i_12 = atomic_fetch_add_explicit((&kernelContext_123)->atom_0+(_S824 + 21U), 1U, memory_order_relaxed);
        if(i_12 < ((&kernelContext_123)->params_0->ev_moved_cap_0))
        {
            uint _S825 = a_moved_0(&kernelContext_123);
            uint _S826 = _S825 + 2U * i_12;
            atomic_store_explicit((&kernelContext_123)->atom_0+_S826, c_18, memory_order_relaxed);
            atomic_store_explicit((&kernelContext_123)->atom_0+(_S826 + 1U), slot_4, memory_order_relaxed);
        }
        else
        {
            uint _S827 = atomic_fetch_or_explicit((&kernelContext_123)->atom_0+(_S824 + 3U), 128U, memory_order_relaxed);
        }
    }
    *((&kernelContext_123)->work_0+(w_body_0() + c_18)) = slot_4;
    uint _S828 = 2U * slot_4;
    float3 _S829 = (float4(*((&kernelContext_123)->frames_0+_S828)) ).xyz;
    thread Quat_0 _S830 = quat_of_0(float4(*((&kernelContext_123)->frames_0+(_S828 + 1U))) );
    Quat_0 _S831 = conj_0(&_S830);
    float inv_mass_4 = (float4(*((&kernelContext_123)->bodies_0+7U * slot_4)) ).w;
    float _S832;
    if(inv_mass_4 > 0.0f)
    {
        _S832 = 1.0f / inv_mass_4;
    }
    else
    {
        _S832 = float_bits_0(2139095040U);
    }
    float4 _S833 = float4(*((&kernelContext_123)->chunks_0+(4U * c_18 + 2U))) ;
    uint first_1 = (as_type<uint>((_S833.z)));
    uint _S834 = (as_type<uint>((_S833.w)));
    float4 _S835 = float4(*((&kernelContext_123)->chunks_0+(4U * r_21 + 3U))) ;
    uint p_13 = first_1;
    for(;;)
    {
        if(p_13 < (first_1 + _S834))
        {
        }
        else
        {
            break;
        }
        uint _S836 = 2U * p_13;
        float4 _S837 = float4(*((&kernelContext_123)->parts_0+_S836)) ;
        uint col_0 = (as_type<uint>((_S837.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_123)->parts_0+(_S836 + 1U))) );
        uint _S838 = 4U * col_0;
        float4 _S839 = float4(*((&kernelContext_123)->colliders_0+_S838)) ;
        *((&kernelContext_123)->colliders_0+_S838) = packed_float4(float4((as_type<float>((slot_4))), _S839.y, _S839.z, _S839.w)) ;
        uint _S840 = _S838 + 1U;
        packed_float4 device* _S841 = (&kernelContext_123)->colliders_0+_S840;
        float3 _S842 = _S837.xyz - _S829;
        thread Quat_0 _S843 = _S831;
        float3 _S844 = rotate_0(&_S843, _S842);
        *_S841 = packed_float4(float4(_S844, (float4(*((&kernelContext_123)->colliders_0+_S840)) ).w)) ;
        packed_float4 device* _S845 = (&kernelContext_123)->colliders_0+(_S838 + 2U);
        thread Quat_0 _S846 = _S831;
        thread Quat_0 _S847 = rot_1;
        Quat_0 _S848 = quat_mul_0(&_S846, &_S847);
        thread Quat_0 _S849 = _S848;
        Quat_0 _S850 = normalized_0(&_S849);
        thread Quat_0 _S851 = _S850;
        float4 _S852 = quat_vec_0(&_S851);
        *_S845 = packed_float4(_S852) ;
        uint _S853 = 2U * col_0;
        uint4 _S854 = uint4(*((&kernelContext_123)->collider_info_0+_S853)) ;
        *((&kernelContext_123)->collider_info_0+_S853) = packed_uint4(uint4(_S854.x, slot_4, _S854.z, _S854.w)) ;
        *((&kernelContext_123)->collider_info_0+(_S853 + 1U)) = packed_uint4(uint4((as_type<uint>((_S832))), (as_type<uint>((_S835.x))), (as_type<uint>((_S835.w))), 0U)) ;
        p_13 = p_13 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], atomic<uint> device* atom_22 [[buffer(5)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], packed_float4 device* lp_22 [[buffer(24)]], packed_float4 device* ext_23 [[buffer(23)]], packed_float4 device* engine_contacts_22 [[buffer(25)]], CrushParams_0 constant* crush_22 [[buffer(28)]], packed_float4 device* crush_rw_22 [[buffer(27)]], packed_float4 device* crush_tab_22 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_124;
    (&kernelContext_124)->params_0 = params_22;
    (&kernelContext_124)->atom_0 = atom_22;
    (&kernelContext_124)->work_0 = work_22;
    (&kernelContext_124)->frames_0 = frames_22;
    (&kernelContext_124)->bodies_0 = bodies_22;
    (&kernelContext_124)->chunks_0 = chunks_22;
    (&kernelContext_124)->sort_0 = sort_22;
    (&kernelContext_124)->manifolds_0 = manifolds_22;
    (&kernelContext_124)->loads_0 = loads_22;
    (&kernelContext_124)->bonds_0 = bonds_22;
    (&kernelContext_124)->args_0 = args_22;
    (&kernelContext_124)->info_0 = info_22;
    (&kernelContext_124)->scan_0 = scan_22;
    (&kernelContext_124)->parts_0 = parts_22;
    (&kernelContext_124)->colliders_0 = colliders_22;
    (&kernelContext_124)->collider_info_0 = collider_info_22;
    (&kernelContext_124)->view_chunks_0 = view_chunks_22;
    (&kernelContext_124)->view_0 = view_22;
    (&kernelContext_124)->objects_0 = objects_22;
    (&kernelContext_124)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_124)->lines_0 = lines_22;
    (&kernelContext_124)->geometry_0 = geometry_22;
    (&kernelContext_124)->shots_0 = shots_22;
    (&kernelContext_124)->lp_0 = lp_22;
    (&kernelContext_124)->ext_0 = ext_23;
    (&kernelContext_124)->engine_contacts_0 = engine_contacts_22;
    (&kernelContext_124)->crush_0 = crush_22;
    (&kernelContext_124)->crush_rw_0 = crush_rw_22;
    (&kernelContext_124)->crush_tab_0 = crush_tab_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_124)->g_scan_0 = &g_scan_22;
    threadgroup array<float, int(64)> g_il_best_22;
    (&kernelContext_124)->g_il_best_0 = &g_il_best_22;
    uint s_10 = id_19.x;
    if(s_10 >= (params_22->body_count_0))
    {
        return;
    }
    uint _S855 = a_dirty_0(&kernelContext_124);
    atomic_store_explicit((&kernelContext_124)->atom_0+(_S855 + s_10), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_5 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_8 = fract(h_5);
    float q_18 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_8);
    float t_9 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_8));
    uint k_25 = uint(h_5);
    float3 _S856;
    if(k_25 == 0U)
    {
        _S856 = float3(0.92000001668930054f, t_9, 0.34959998726844788f);
    }
    else
    {
        if(k_25 == 1U)
        {
            _S856 = float3(q_18, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_25 == 2U)
            {
                _S856 = float3(0.34959998726844788f, 0.92000001668930054f, t_9);
            }
            else
            {
                if(k_25 == 3U)
                {
                    _S856 = float3(0.34959998726844788f, q_18, 0.92000001668930054f);
                }
                else
                {
                    if(k_25 == 4U)
                    {
                        _S856 = float3(t_9, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S856 = float3(0.92000001668930054f, 0.34959998726844788f, q_18);
                    }
                }
            }
        }
    }
    return _S856;
}

void write_object_0(uint o_4, const Quat_0 thread* q_19, float3 t_10, float4 color_0, KernelContext_0 thread* kernelContext_125)
{
    float3 _S857 = rotate_0(q_19, float3(1.0f, 0.0f, 0.0f));
    float3 _S858 = rotate_0(q_19, float3(0.0f, 1.0f, 0.0f));
    float3 _S859 = rotate_0(q_19, float3(0.0f, 0.0f, 1.0f));
    uint _S860 = 4U * o_4;
    *(kernelContext_125->objects_0+_S860) = packed_float4(float4(_S857.x, _S858.x, _S859.x, t_10.x)) ;
    *(kernelContext_125->objects_0+(_S860 + 1U)) = packed_float4(float4(_S857.y, _S858.y, _S859.y, t_10.y)) ;
    *(kernelContext_125->objects_0+(_S860 + 2U)) = packed_float4(float4(_S857.z, _S858.z, _S859.z, t_10.z)) ;
    *(kernelContext_125->objects_0+(_S860 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], atomic<uint> device* atom_23 [[buffer(5)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], packed_float4 device* lp_23 [[buffer(24)]], packed_float4 device* ext_24 [[buffer(23)]], packed_float4 device* engine_contacts_23 [[buffer(25)]], CrushParams_0 constant* crush_23 [[buffer(28)]], packed_float4 device* crush_rw_23 [[buffer(27)]], packed_float4 device* crush_tab_23 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_126;
    (&kernelContext_126)->params_0 = params_23;
    (&kernelContext_126)->atom_0 = atom_23;
    (&kernelContext_126)->work_0 = work_23;
    (&kernelContext_126)->frames_0 = frames_23;
    (&kernelContext_126)->bodies_0 = bodies_23;
    (&kernelContext_126)->chunks_0 = chunks_23;
    (&kernelContext_126)->sort_0 = sort_23;
    (&kernelContext_126)->manifolds_0 = manifolds_23;
    (&kernelContext_126)->loads_0 = loads_23;
    (&kernelContext_126)->bonds_0 = bonds_23;
    (&kernelContext_126)->args_0 = args_23;
    (&kernelContext_126)->info_0 = info_23;
    (&kernelContext_126)->scan_0 = scan_23;
    (&kernelContext_126)->parts_0 = parts_23;
    (&kernelContext_126)->colliders_0 = colliders_23;
    (&kernelContext_126)->collider_info_0 = collider_info_23;
    (&kernelContext_126)->view_chunks_0 = view_chunks_23;
    (&kernelContext_126)->view_0 = view_23;
    (&kernelContext_126)->objects_0 = objects_23;
    (&kernelContext_126)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_126)->lines_0 = lines_23;
    (&kernelContext_126)->geometry_0 = geometry_23;
    (&kernelContext_126)->shots_0 = shots_23;
    (&kernelContext_126)->lp_0 = lp_23;
    (&kernelContext_126)->ext_0 = ext_24;
    (&kernelContext_126)->engine_contacts_0 = engine_contacts_23;
    (&kernelContext_126)->crush_0 = crush_23;
    (&kernelContext_126)->crush_rw_0 = crush_rw_23;
    (&kernelContext_126)->crush_tab_0 = crush_tab_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_126)->g_scan_0 = &g_scan_23;
    threadgroup array<float, int(64)> g_il_best_23;
    (&kernelContext_126)->g_il_best_0 = &g_il_best_23;
    uint k_26 = id_20.x;
    bool _S861 = live_0(k_26, &kernelContext_126);
    if(!_S861)
    {
        return;
    }
    uint _S862 = body_of_0(k_26, &kernelContext_126);
    uint b_36 = 7U * _S862;
    thread Quat_0 q_20;
    thread float3 origin_5;
    structure_pose_0(_S862, b_36, &q_20, &origin_5, &kernelContext_126);
    uint _S863 = 3U * k_26;
    float4 _S864 = float4(*((&kernelContext_126)->view_chunks_0+_S863)) ;
    float3 base_2 = (float4(*((&kernelContext_126)->view_chunks_0+(_S863 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_126)->bodies_0+b_36)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_126)->view_0->mode_0) == 1U)
    {
        float _S865 = max(length((float4(*((&kernelContext_126)->bodies_0+(b_36 + 3U))) ).xyz), length((float4(*((&kernelContext_126)->bodies_0+(b_36 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S865 > ((&kernelContext_126)->view_0->rest_speed_0))
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
        if(((&kernelContext_126)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S862);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S862);
            }
        }
    }
    uint _S866 = (as_type<uint>((_S864.w)));
    Quat_0 _S867 = quat_of_0(float4(*((&kernelContext_126)->view_chunks_0+(_S863 + 1U))) );
    thread Quat_0 _S868 = q_20;
    thread Quat_0 _S869 = _S867;
    Quat_0 _S870 = quat_mul_0(&_S868, &_S869);
    thread Quat_0 _S871 = _S870;
    Quat_0 _S872 = normalized_0(&_S871);
    float3 _S873 = _S864.xyz;
    thread Quat_0 _S874 = q_20;
    float3 _S875 = rotate_0(&_S874, _S873);
    float3 _S876 = origin_5 + _S875;
    float4 _S877 = float4(color_1, 1.0f);
    thread Quat_0 _S878 = _S872;
    write_object_0(_S866, &_S878, _S876, _S877, &kernelContext_126);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], atomic<uint> device* atom_24 [[buffer(5)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], packed_float4 device* lp_24 [[buffer(24)]], packed_float4 device* ext_25 [[buffer(23)]], packed_float4 device* engine_contacts_24 [[buffer(25)]], CrushParams_0 constant* crush_24 [[buffer(28)]], packed_float4 device* crush_rw_24 [[buffer(27)]], packed_float4 device* crush_tab_24 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_127;
    (&kernelContext_127)->params_0 = params_24;
    (&kernelContext_127)->atom_0 = atom_24;
    (&kernelContext_127)->work_0 = work_24;
    (&kernelContext_127)->frames_0 = frames_24;
    (&kernelContext_127)->bodies_0 = bodies_24;
    (&kernelContext_127)->chunks_0 = chunks_24;
    (&kernelContext_127)->sort_0 = sort_24;
    (&kernelContext_127)->manifolds_0 = manifolds_24;
    (&kernelContext_127)->loads_0 = loads_24;
    (&kernelContext_127)->bonds_0 = bonds_24;
    (&kernelContext_127)->args_0 = args_24;
    (&kernelContext_127)->info_0 = info_24;
    (&kernelContext_127)->scan_0 = scan_24;
    (&kernelContext_127)->parts_0 = parts_24;
    (&kernelContext_127)->colliders_0 = colliders_24;
    (&kernelContext_127)->collider_info_0 = collider_info_24;
    (&kernelContext_127)->view_chunks_0 = view_chunks_24;
    (&kernelContext_127)->view_0 = view_24;
    (&kernelContext_127)->objects_0 = objects_24;
    (&kernelContext_127)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_127)->lines_0 = lines_24;
    (&kernelContext_127)->geometry_0 = geometry_24;
    (&kernelContext_127)->shots_0 = shots_24;
    (&kernelContext_127)->lp_0 = lp_24;
    (&kernelContext_127)->ext_0 = ext_25;
    (&kernelContext_127)->engine_contacts_0 = engine_contacts_24;
    (&kernelContext_127)->crush_0 = crush_24;
    (&kernelContext_127)->crush_rw_0 = crush_rw_24;
    (&kernelContext_127)->crush_tab_0 = crush_tab_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_127)->g_scan_0 = &g_scan_24;
    threadgroup array<float, int(64)> g_il_best_24;
    (&kernelContext_127)->g_il_best_0 = &g_il_best_24;
    uint i_13 = id_21.x;
    if(i_13 >= (view_24->shot_count_1))
    {
        return;
    }
    uint b_37 = 7U * (&kernelContext_127)->shot_bodies_0[i_13];
    uint o_5 = (&kernelContext_127)->view_0->impactor_base_0 + i_13;
    if(((float4(*((&kernelContext_127)->bodies_0+b_37)) ).w) == 0.0f)
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
            *((&kernelContext_127)->objects_0+(4U * o_5 + w_5)) = packed_float4(float4(0.0f) ) ;
            w_5 = w_5 + 1U;
        }
        return;
    }
    thread Quat_0 _S879 = quat_of_0(float4(*((&kernelContext_127)->bodies_0+(b_37 + 1U))) );
    Quat_0 _S880 = rot_from_rubble_0(&_S879);
    float3 _S881 = from_rubble_0((float4(*((&kernelContext_127)->bodies_0+b_37)) ).xyz);
    float4 _S882 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S883 = _S880;
    write_object_0(o_5, &_S883, _S881, _S882, &kernelContext_127);
    return;
}

float3 chunk_world_0(uint k_27, KernelContext_0 thread* kernelContext_128)
{
    uint _S884 = body_of_0(k_27, kernelContext_128);
    thread Quat_0 q_21;
    thread float3 origin_6;
    structure_pose_0(_S884, 7U * _S884, &q_21, &origin_6, kernelContext_128);
    float3 _S885 = (float4(*(kernelContext_128->view_chunks_0+3U * k_27)) ).xyz;
    thread Quat_0 _S886 = q_21;
    float3 _S887 = rotate_0(&_S886, _S885);
    return origin_6 + _S887;
}

void write_line_vertex_0(uint v_13, float3 p_14, KernelContext_0 thread* kernelContext_129)
{
    uint _S888 = 7U * v_13;
    *(kernelContext_129->lines_0+_S888) = p_14.x;
    *(kernelContext_129->lines_0+(_S888 + 1U)) = p_14.y;
    *(kernelContext_129->lines_0+(_S888 + 2U)) = p_14.z;
    *(kernelContext_129->lines_0+(_S888 + 3U)) = 1.0f;
    *(kernelContext_129->lines_0+(_S888 + 4U)) = 0.86000001430511475f;
    *(kernelContext_129->lines_0+(_S888 + 5U)) = 0.25f;
    *(kernelContext_129->lines_0+(_S888 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], atomic<uint> device* atom_25 [[buffer(5)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], packed_float4 device* lp_25 [[buffer(24)]], packed_float4 device* ext_26 [[buffer(23)]], packed_float4 device* engine_contacts_25 [[buffer(25)]], CrushParams_0 constant* crush_25 [[buffer(28)]], packed_float4 device* crush_rw_25 [[buffer(27)]], packed_float4 device* crush_tab_25 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_130;
    (&kernelContext_130)->params_0 = params_25;
    (&kernelContext_130)->atom_0 = atom_25;
    (&kernelContext_130)->work_0 = work_25;
    (&kernelContext_130)->frames_0 = frames_25;
    (&kernelContext_130)->bodies_0 = bodies_25;
    (&kernelContext_130)->chunks_0 = chunks_25;
    (&kernelContext_130)->sort_0 = sort_25;
    (&kernelContext_130)->manifolds_0 = manifolds_25;
    (&kernelContext_130)->loads_0 = loads_25;
    (&kernelContext_130)->bonds_0 = bonds_25;
    (&kernelContext_130)->args_0 = args_25;
    (&kernelContext_130)->info_0 = info_25;
    (&kernelContext_130)->scan_0 = scan_25;
    (&kernelContext_130)->parts_0 = parts_25;
    (&kernelContext_130)->colliders_0 = colliders_25;
    (&kernelContext_130)->collider_info_0 = collider_info_25;
    (&kernelContext_130)->view_chunks_0 = view_chunks_25;
    (&kernelContext_130)->view_0 = view_25;
    (&kernelContext_130)->objects_0 = objects_25;
    (&kernelContext_130)->shot_bodies_0 = shot_bodies_25;
    (&kernelContext_130)->lines_0 = lines_25;
    (&kernelContext_130)->geometry_0 = geometry_25;
    (&kernelContext_130)->shots_0 = shots_25;
    (&kernelContext_130)->lp_0 = lp_25;
    (&kernelContext_130)->ext_0 = ext_26;
    (&kernelContext_130)->engine_contacts_0 = engine_contacts_25;
    (&kernelContext_130)->crush_0 = crush_25;
    (&kernelContext_130)->crush_rw_0 = crush_rw_25;
    (&kernelContext_130)->crush_tab_0 = crush_tab_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_130)->g_scan_0 = &g_scan_25;
    threadgroup array<float, int(64)> g_il_best_25;
    (&kernelContext_130)->g_il_best_0 = &g_il_best_25;
    uint i_14 = id_22.x;
    if(i_14 >= (params_25->bond_used_0))
    {
        return;
    }
    uint4 _S889 = uint4(*((&kernelContext_130)->bonds_0+i_14)) ;
    float3 _S890 = chunk_world_0(_S889.x, &kernelContext_130);
    uint _S891 = w_alive_0(&kernelContext_130);
    float3 b_38;
    if((*((&kernelContext_130)->work_0+(_S891 + i_14))) != 0U)
    {
        float3 _S892 = chunk_world_0(_S889.y, &kernelContext_130);
        b_38 = _S892;
    }
    else
    {
        b_38 = _S890;
    }
    uint _S893 = 2U * i_14;
    write_line_vertex_0(_S893, _S890, &kernelContext_130);
    write_line_vertex_0(_S893 + 1U, b_38, &kernelContext_130);
    return;
}

[[kernel]] void fr_strike_reset(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], atomic<uint> device* atom_26 [[buffer(5)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], packed_float4 device* lp_26 [[buffer(24)]], packed_float4 device* ext_27 [[buffer(23)]], packed_float4 device* engine_contacts_26 [[buffer(25)]], CrushParams_0 constant* crush_26 [[buffer(28)]], packed_float4 device* crush_rw_26 [[buffer(27)]], packed_float4 device* crush_tab_26 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_131;
    (&kernelContext_131)->params_0 = params_26;
    (&kernelContext_131)->atom_0 = atom_26;
    (&kernelContext_131)->work_0 = work_26;
    (&kernelContext_131)->frames_0 = frames_26;
    (&kernelContext_131)->bodies_0 = bodies_26;
    (&kernelContext_131)->chunks_0 = chunks_26;
    (&kernelContext_131)->sort_0 = sort_26;
    (&kernelContext_131)->manifolds_0 = manifolds_26;
    (&kernelContext_131)->loads_0 = loads_26;
    (&kernelContext_131)->bonds_0 = bonds_26;
    (&kernelContext_131)->args_0 = args_26;
    (&kernelContext_131)->info_0 = info_26;
    (&kernelContext_131)->scan_0 = scan_26;
    (&kernelContext_131)->parts_0 = parts_26;
    (&kernelContext_131)->colliders_0 = colliders_26;
    (&kernelContext_131)->collider_info_0 = collider_info_26;
    (&kernelContext_131)->view_chunks_0 = view_chunks_26;
    (&kernelContext_131)->view_0 = view_26;
    (&kernelContext_131)->objects_0 = objects_26;
    (&kernelContext_131)->shot_bodies_0 = shot_bodies_26;
    (&kernelContext_131)->lines_0 = lines_26;
    (&kernelContext_131)->geometry_0 = geometry_26;
    (&kernelContext_131)->shots_0 = shots_26;
    (&kernelContext_131)->lp_0 = lp_26;
    (&kernelContext_131)->ext_0 = ext_27;
    (&kernelContext_131)->engine_contacts_0 = engine_contacts_26;
    (&kernelContext_131)->crush_0 = crush_26;
    (&kernelContext_131)->crush_rw_0 = crush_rw_26;
    (&kernelContext_131)->crush_tab_0 = crush_tab_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_131)->g_scan_0 = &g_scan_26;
    threadgroup array<float, int(64)> g_il_best_26;
    (&kernelContext_131)->g_il_best_0 = &g_il_best_26;
    uint b_39 = id_23.x;
    if(b_39 >= (params_26->body_count_0))
    {
        return;
    }
    uint _S894 = a_strike_0(&kernelContext_131);
    uint at_13 = _S894 + 9U * b_39;
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
        atomic_store_explicit((&kernelContext_131)->atom_0+(at_13 + w_6), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_131)->atom_0+(at_13 + 3U + w_6), 0U, memory_order_relaxed);
        w_6 = w_6 + 1U;
    }
    atomic_store_explicit((&kernelContext_131)->atom_0+(at_13 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_131)->atom_0+(at_13 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_131)->atom_0+(at_13 + 8U), 4294967295U, memory_order_relaxed);
    uint _S895 = strike_at_0(b_39, &kernelContext_131);
    (*((&kernelContext_131)->frames_0+_S895))[int(3)] = 0.0f;
    return;
}

uint ordered_0(float f_9)
{
    uint u_6 = (as_type<uint>((f_9)));
    uint _S896;
    if((u_6 & 2147483648U) != 0U)
    {
        _S896 = ~u_6;
    }
    else
    {
        _S896 = u_6 | 2147483648U;
    }
    return _S896;
}

[[kernel]] void fr_strike_bounds(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], atomic<uint> device* atom_27 [[buffer(5)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], packed_float4 device* lp_27 [[buffer(24)]], packed_float4 device* ext_28 [[buffer(23)]], packed_float4 device* engine_contacts_27 [[buffer(25)]], CrushParams_0 constant* crush_27 [[buffer(28)]], packed_float4 device* crush_rw_27 [[buffer(27)]], packed_float4 device* crush_tab_27 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_132;
    (&kernelContext_132)->params_0 = params_27;
    (&kernelContext_132)->atom_0 = atom_27;
    (&kernelContext_132)->work_0 = work_27;
    (&kernelContext_132)->frames_0 = frames_27;
    (&kernelContext_132)->bodies_0 = bodies_27;
    (&kernelContext_132)->chunks_0 = chunks_27;
    (&kernelContext_132)->sort_0 = sort_27;
    (&kernelContext_132)->manifolds_0 = manifolds_27;
    (&kernelContext_132)->loads_0 = loads_27;
    (&kernelContext_132)->bonds_0 = bonds_27;
    (&kernelContext_132)->args_0 = args_27;
    (&kernelContext_132)->info_0 = info_27;
    (&kernelContext_132)->scan_0 = scan_27;
    (&kernelContext_132)->parts_0 = parts_27;
    (&kernelContext_132)->colliders_0 = colliders_27;
    (&kernelContext_132)->collider_info_0 = collider_info_27;
    (&kernelContext_132)->view_chunks_0 = view_chunks_27;
    (&kernelContext_132)->view_0 = view_27;
    (&kernelContext_132)->objects_0 = objects_27;
    (&kernelContext_132)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_132)->lines_0 = lines_27;
    (&kernelContext_132)->geometry_0 = geometry_27;
    (&kernelContext_132)->shots_0 = shots_27;
    (&kernelContext_132)->lp_0 = lp_27;
    (&kernelContext_132)->ext_0 = ext_28;
    (&kernelContext_132)->engine_contacts_0 = engine_contacts_27;
    (&kernelContext_132)->crush_0 = crush_27;
    (&kernelContext_132)->crush_rw_0 = crush_rw_27;
    (&kernelContext_132)->crush_tab_0 = crush_tab_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_132)->g_scan_0 = &g_scan_27;
    threadgroup array<float, int(64)> g_il_best_27;
    (&kernelContext_132)->g_il_best_0 = &g_il_best_27;
    uint k_28 = id_24.x;
    bool _S897 = live_0(k_28, &kernelContext_132);
    bool _S898;
    if(!_S897)
    {
        _S898 = true;
    }
    else
    {
        _S898 = ((&kernelContext_132)->params_0->shot_count_0) == 0U;
    }
    if(_S898)
    {
        return;
    }
    thread Quat_0 q_22;
    float3 _S899 = chunk_box_world_0(k_28, &q_22, &kernelContext_132);
    float _S900 = (float4(*((&kernelContext_132)->geometry_0+3U * k_28)) ).w;
    uint _S901 = a_strike_0(&kernelContext_132);
    uint _S902 = body_of_0(k_28, &kernelContext_132);
    uint _S903 = _S901 + 9U * _S902;
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
        uint _S904 = ax_1;
        uint _S905 = atomic_fetch_min_explicit((&kernelContext_132)->atom_0+(_S903 + ax_1), ordered_0(_S899[ax_1] - _S900), memory_order_relaxed);
        uint _S906 = atomic_fetch_max_explicit((&kernelContext_132)->atom_0+(_S903 + 3U + ax_1), ordered_0(_S899[_S904] + _S900), memory_order_relaxed);
        ax_1 = ax_1 + 1U;
    }
    return;
}

float unordered_0(uint u_7)
{
    uint _S907;
    if((u_7 & 2147483648U) != 0U)
    {
        _S907 = u_7 & 2147483647U;
    }
    else
    {
        _S907 = ~u_7;
    }
    return (as_type<float>((_S907)));
}

[[kernel]] void fr_strike_find(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], atomic<uint> device* atom_28 [[buffer(5)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], packed_float4 device* lp_28 [[buffer(24)]], packed_float4 device* ext_29 [[buffer(23)]], packed_float4 device* engine_contacts_28 [[buffer(25)]], CrushParams_0 constant* crush_28 [[buffer(28)]], packed_float4 device* crush_rw_28 [[buffer(27)]], packed_float4 device* crush_tab_28 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_133;
    (&kernelContext_133)->params_0 = params_28;
    (&kernelContext_133)->atom_0 = atom_28;
    (&kernelContext_133)->work_0 = work_28;
    (&kernelContext_133)->frames_0 = frames_28;
    (&kernelContext_133)->bodies_0 = bodies_28;
    (&kernelContext_133)->chunks_0 = chunks_28;
    (&kernelContext_133)->sort_0 = sort_28;
    (&kernelContext_133)->manifolds_0 = manifolds_28;
    (&kernelContext_133)->loads_0 = loads_28;
    (&kernelContext_133)->bonds_0 = bonds_28;
    (&kernelContext_133)->args_0 = args_28;
    (&kernelContext_133)->info_0 = info_28;
    (&kernelContext_133)->scan_0 = scan_28;
    (&kernelContext_133)->parts_0 = parts_28;
    (&kernelContext_133)->colliders_0 = colliders_28;
    (&kernelContext_133)->collider_info_0 = collider_info_28;
    (&kernelContext_133)->view_chunks_0 = view_chunks_28;
    (&kernelContext_133)->view_0 = view_28;
    (&kernelContext_133)->objects_0 = objects_28;
    (&kernelContext_133)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_133)->lines_0 = lines_28;
    (&kernelContext_133)->geometry_0 = geometry_28;
    (&kernelContext_133)->shots_0 = shots_28;
    (&kernelContext_133)->lp_0 = lp_28;
    (&kernelContext_133)->ext_0 = ext_29;
    (&kernelContext_133)->engine_contacts_0 = engine_contacts_28;
    (&kernelContext_133)->crush_0 = crush_28;
    (&kernelContext_133)->crush_rw_0 = crush_rw_28;
    (&kernelContext_133)->crush_tab_0 = crush_tab_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_133)->g_scan_0 = &g_scan_28;
    threadgroup array<float, int(64)> g_il_best_28;
    (&kernelContext_133)->g_il_best_0 = &g_il_best_28;
    uint b_40 = id_25.x;
    bool _S908;
    if(b_40 >= (params_28->body_count_0))
    {
        _S908 = true;
    }
    else
    {
        _S908 = ((&kernelContext_133)->params_0->shot_count_0) == 0U;
    }
    if(_S908)
    {
        return;
    }
    uint _S909 = a_strike_0(&kernelContext_133);
    uint at_14 = _S909 + 9U * b_40;
    uint _S910 = atomic_load_explicit((&kernelContext_133)->atom_0+at_14, memory_order_relaxed);
    if(_S910 == 4294967295U)
    {
        return;
    }
    uint _S911 = atomic_load_explicit((&kernelContext_133)->atom_0+at_14, memory_order_relaxed);
    float _S912 = unordered_0(_S911);
    uint _S913 = atomic_load_explicit((&kernelContext_133)->atom_0+(at_14 + 1U), memory_order_relaxed);
    float _S914 = unordered_0(_S913);
    uint _S915 = atomic_load_explicit((&kernelContext_133)->atom_0+(at_14 + 2U), memory_order_relaxed);
    float3 _S916 = float3(_S912, _S914, unordered_0(_S915));
    uint _S917 = atomic_load_explicit((&kernelContext_133)->atom_0+(at_14 + 3U), memory_order_relaxed);
    float _S918 = unordered_0(_S917);
    uint _S919 = atomic_load_explicit((&kernelContext_133)->atom_0+(at_14 + 4U), memory_order_relaxed);
    float _S920 = unordered_0(_S919);
    uint _S921 = atomic_load_explicit((&kernelContext_133)->atom_0+(at_14 + 5U), memory_order_relaxed);
    float3 _S922 = float3(_S918, _S920, unordered_0(_S921));
    float3 _S923 = from_rubble_0((float4(*((&kernelContext_133)->bodies_0+(7U * b_40 + 3U))) ).xyz);
    float _S924 = float_bits_0(2139095040U);
    uint k_29 = 0U;
    for(;;)
    {
        if(k_29 < ((&kernelContext_133)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S925 = float4(*((&kernelContext_133)->shots_0+2U * k_29)) ;
        uint sb_1 = 7U * (as_type<uint>((_S925.x)));
        if(((float4(*((&kernelContext_133)->bodies_0+sb_1)) ).w) == 0.0f)
        {
            k_29 = k_29 + 1U;
            continue;
        }
        float3 c_19 = from_rubble_0((float4(*((&kernelContext_133)->bodies_0+sb_1)) ).xyz);
        float3 rel_2 = from_rubble_0((float4(*((&kernelContext_133)->bodies_0+(sb_1 + 3U))) ).xyz) - _S923;
        if(!((length(rel_2)) > ((&kernelContext_133)->params_0->strike_speed_0)))
        {
            k_29 = k_29 + 1U;
            continue;
        }
        float3 _S926 = float3(_S925.y) ;
        float3 _S927 = _S916 - _S926;
        float3 _S928 = _S922 + _S926;
        float _S929 = (&kernelContext_133)->params_0->dt_0;
        float t0_1 = 0.0f;
        float t1_1 = _S929;
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
            uint _S930 = ax_2;
            if((abs(rel_2[ax_2])) < 9.999999960041972e-13f)
            {
                uint _S931 = ax_2;
                if((c_19[ax_2]) < (_S927[ax_2]))
                {
                    _S908 = true;
                }
                else
                {
                    _S908 = (c_19[_S931]) > (_S928[ax_2]);
                }
                float t0_2;
                if(_S908)
                {
                    t0_2 = _S924;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + 1U;
                continue;
            }
            float ta_1 = (_S927[ax_2] - c_19[ax_2]) / rel_2[_S930];
            float tb_1 = (_S928[ax_2] - c_19[ax_2]) / rel_2[_S930];
            float _S932 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S932;
            ax_2 = ax_2 + 1U;
        }
        if(t0_1 <= t1_1)
        {
            uint _S933 = strike_at_0(b_40, &kernelContext_133);
            *((&kernelContext_133)->frames_0+_S933) = packed_float4(float4(c_19, float(k_29 + 1U))) ;
            *((&kernelContext_133)->frames_0+(_S933 + 1U)) = packed_float4(float4(rel_2, t0_1)) ;
            return;
        }
        k_29 = k_29 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_entry(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], atomic<uint> device* atom_29 [[buffer(5)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], packed_float4 device* lp_29 [[buffer(24)]], packed_float4 device* ext_30 [[buffer(23)]], packed_float4 device* engine_contacts_29 [[buffer(25)]], CrushParams_0 constant* crush_29 [[buffer(28)]], packed_float4 device* crush_rw_29 [[buffer(27)]], packed_float4 device* crush_tab_29 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_134;
    (&kernelContext_134)->params_0 = params_29;
    (&kernelContext_134)->atom_0 = atom_29;
    (&kernelContext_134)->work_0 = work_29;
    (&kernelContext_134)->frames_0 = frames_29;
    (&kernelContext_134)->bodies_0 = bodies_29;
    (&kernelContext_134)->chunks_0 = chunks_29;
    (&kernelContext_134)->sort_0 = sort_29;
    (&kernelContext_134)->manifolds_0 = manifolds_29;
    (&kernelContext_134)->loads_0 = loads_29;
    (&kernelContext_134)->bonds_0 = bonds_29;
    (&kernelContext_134)->args_0 = args_29;
    (&kernelContext_134)->info_0 = info_29;
    (&kernelContext_134)->scan_0 = scan_29;
    (&kernelContext_134)->parts_0 = parts_29;
    (&kernelContext_134)->colliders_0 = colliders_29;
    (&kernelContext_134)->collider_info_0 = collider_info_29;
    (&kernelContext_134)->view_chunks_0 = view_chunks_29;
    (&kernelContext_134)->view_0 = view_29;
    (&kernelContext_134)->objects_0 = objects_29;
    (&kernelContext_134)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_134)->lines_0 = lines_29;
    (&kernelContext_134)->geometry_0 = geometry_29;
    (&kernelContext_134)->shots_0 = shots_29;
    (&kernelContext_134)->lp_0 = lp_29;
    (&kernelContext_134)->ext_0 = ext_30;
    (&kernelContext_134)->engine_contacts_0 = engine_contacts_29;
    (&kernelContext_134)->crush_0 = crush_29;
    (&kernelContext_134)->crush_rw_0 = crush_rw_29;
    (&kernelContext_134)->crush_tab_0 = crush_tab_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_134)->g_scan_0 = &g_scan_29;
    threadgroup array<float, int(64)> g_il_best_29;
    (&kernelContext_134)->g_il_best_0 = &g_il_best_29;
    uint k_30 = id_26.x;
    bool _S934 = live_0(k_30, &kernelContext_134);
    bool _S935;
    if(!_S934)
    {
        _S935 = true;
    }
    else
    {
        _S935 = ((&kernelContext_134)->params_0->shot_count_0) == 0U;
    }
    if(_S935)
    {
        return;
    }
    uint _S936 = body_of_0(k_30, &kernelContext_134);
    thread uint shot_5;
    bool _S937 = struck_0(_S936, &shot_5, &kernelContext_134);
    if(!_S937)
    {
        return;
    }
    uint _S938 = strike_at_0(_S936, &kernelContext_134);
    float4 _S939 = float4(*((&kernelContext_134)->frames_0+(_S938 + 1U))) ;
    float3 from_0 = (float4(*((&kernelContext_134)->frames_0+_S938)) ).xyz;
    float3 rel_3 = _S939.xyz;
    float3 d_20 = rel_3 / float3(max(length(rel_3), 9.999999960041972e-13f)) ;
    float _S940 = (float4(*((&kernelContext_134)->shots_0+2U * shot_5)) ).y;
    thread Quat_0 q_23;
    float3 _S941 = chunk_box_world_0(k_30, &q_23, &kernelContext_134);
    uint _S942 = 3U * k_30;
    Quat_0 _S943 = quat_of_0(float4(*((&kernelContext_134)->geometry_0+(_S942 + 2U))) );
    thread Quat_0 _S944 = q_23;
    thread Quat_0 _S945 = _S943;
    Quat_0 _S946 = quat_mul_0(&_S944, &_S945);
    thread Quat_0 _S947 = _S946;
    Quat_0 _S948 = normalized_0(&_S947);
    float3 _S949 = (float4(*((&kernelContext_134)->geometry_0+(_S942 + 1U))) ).xyz;
    float _S950 = float_bits_0(2139095040U);
    float _S951 = (&kernelContext_134)->params_0->dt_0;
    float t_in_5 = 0.0f;
    float t_out_0 = _S951;
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
        thread Quat_0 _S952 = _S948;
        float3 _S953 = rotate_0(&_S952, e_3);
        float o_6 = dot(from_0 - _S941, _S953);
        float dv_1 = dot(rel_3, _S953);
        float ext_31 = _S949[ax_3] + _S940;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            float t_in_6;
            if((abs(o_6)) > ext_31)
            {
                t_in_6 = _S950;
            }
            else
            {
                t_in_6 = t_in_5;
            }
            t_in_5 = t_in_6;
            ax_3 = ax_3 + 1U;
            continue;
        }
        float ta_2 = (- ext_31 - o_6) / dv_1;
        float tb_2 = (ext_31 - o_6) / dv_1;
        float _S954 = min(t_out_0, max(ta_2, tb_2));
        t_in_5 = max(t_in_5, min(ta_2, tb_2));
        t_out_0 = _S954;
        ax_3 = ax_3 + 1U;
    }
    uint _S955 = a_strike_0(&kernelContext_134);
    uint at_15 = _S955 + 9U * _S936;
    if(t_in_5 <= t_out_0)
    {
        uint _S956 = atomic_fetch_min_explicit((&kernelContext_134)->atom_0+(at_15 + 6U), (as_type<uint>((t_in_5))), memory_order_relaxed);
    }
    float3 r_22 = _S941 - (from_0 + rel_3 * float3(_S939.w) );
    float along_1 = dot(r_22, d_20);
    float dist_0 = length(r_22 - d_20 * float3(along_1) ) + abs(along_1) * 0.00100000004749745f;
    uint _S957 = a_strike_dist_0(&kernelContext_134);
    uint _S958 = (as_type<uint>((dist_0)));
    atomic_store_explicit((&kernelContext_134)->atom_0+(_S957 + k_30), _S958, memory_order_relaxed);
    uint _S959 = atomic_fetch_min_explicit((&kernelContext_134)->atom_0+(at_15 + 7U), _S958, memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], atomic<uint> device* atom_30 [[buffer(5)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], packed_float4 device* lp_30 [[buffer(24)]], packed_float4 device* ext_32 [[buffer(23)]], packed_float4 device* engine_contacts_30 [[buffer(25)]], CrushParams_0 constant* crush_30 [[buffer(28)]], packed_float4 device* crush_rw_30 [[buffer(27)]], packed_float4 device* crush_tab_30 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_135;
    (&kernelContext_135)->params_0 = params_30;
    (&kernelContext_135)->atom_0 = atom_30;
    (&kernelContext_135)->work_0 = work_30;
    (&kernelContext_135)->frames_0 = frames_30;
    (&kernelContext_135)->bodies_0 = bodies_30;
    (&kernelContext_135)->chunks_0 = chunks_30;
    (&kernelContext_135)->sort_0 = sort_30;
    (&kernelContext_135)->manifolds_0 = manifolds_30;
    (&kernelContext_135)->loads_0 = loads_30;
    (&kernelContext_135)->bonds_0 = bonds_30;
    (&kernelContext_135)->args_0 = args_30;
    (&kernelContext_135)->info_0 = info_30;
    (&kernelContext_135)->scan_0 = scan_30;
    (&kernelContext_135)->parts_0 = parts_30;
    (&kernelContext_135)->colliders_0 = colliders_30;
    (&kernelContext_135)->collider_info_0 = collider_info_30;
    (&kernelContext_135)->view_chunks_0 = view_chunks_30;
    (&kernelContext_135)->view_0 = view_30;
    (&kernelContext_135)->objects_0 = objects_30;
    (&kernelContext_135)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_135)->lines_0 = lines_30;
    (&kernelContext_135)->geometry_0 = geometry_30;
    (&kernelContext_135)->shots_0 = shots_30;
    (&kernelContext_135)->lp_0 = lp_30;
    (&kernelContext_135)->ext_0 = ext_32;
    (&kernelContext_135)->engine_contacts_0 = engine_contacts_30;
    (&kernelContext_135)->crush_0 = crush_30;
    (&kernelContext_135)->crush_rw_0 = crush_rw_30;
    (&kernelContext_135)->crush_tab_0 = crush_tab_30;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_135)->g_scan_0 = &g_scan_30;
    threadgroup array<float, int(64)> g_il_best_30;
    (&kernelContext_135)->g_il_best_0 = &g_il_best_30;
    uint k_31 = id_27.x;
    bool _S960 = live_0(k_31, &kernelContext_135);
    bool _S961;
    if(!_S960)
    {
        _S961 = true;
    }
    else
    {
        _S961 = ((&kernelContext_135)->params_0->shot_count_0) == 0U;
    }
    if(_S961)
    {
        return;
    }
    uint _S962 = body_of_0(k_31, &kernelContext_135);
    thread uint shot_6;
    bool _S963 = struck_0(_S962, &shot_6, &kernelContext_135);
    if(!_S963)
    {
        return;
    }
    uint _S964 = a_strike_0(&kernelContext_135);
    uint at_16 = _S964 + 9U * _S962;
    uint _S965 = a_strike_dist_0(&kernelContext_135);
    uint _S966 = atomic_load_explicit((&kernelContext_135)->atom_0+(_S965 + k_31), memory_order_relaxed);
    uint _S967 = atomic_load_explicit((&kernelContext_135)->atom_0+(at_16 + 7U), memory_order_relaxed);
    if(_S966 == _S967)
    {
        uint _S968 = atomic_fetch_min_explicit((&kernelContext_135)->atom_0+(at_16 + 8U), k_31, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], atomic<uint> device* atom_31 [[buffer(5)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_31 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], packed_float4 device* lp_31 [[buffer(24)]], packed_float4 device* ext_33 [[buffer(23)]], packed_float4 device* engine_contacts_31 [[buffer(25)]], CrushParams_0 constant* crush_31 [[buffer(28)]], packed_float4 device* crush_rw_31 [[buffer(27)]], packed_float4 device* crush_tab_31 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_136;
    (&kernelContext_136)->params_0 = params_31;
    (&kernelContext_136)->atom_0 = atom_31;
    (&kernelContext_136)->work_0 = work_31;
    (&kernelContext_136)->frames_0 = frames_31;
    (&kernelContext_136)->bodies_0 = bodies_31;
    (&kernelContext_136)->chunks_0 = chunks_31;
    (&kernelContext_136)->sort_0 = sort_31;
    (&kernelContext_136)->manifolds_0 = manifolds_31;
    (&kernelContext_136)->loads_0 = loads_31;
    (&kernelContext_136)->bonds_0 = bonds_31;
    (&kernelContext_136)->args_0 = args_31;
    (&kernelContext_136)->info_0 = info_31;
    (&kernelContext_136)->scan_0 = scan_31;
    (&kernelContext_136)->parts_0 = parts_31;
    (&kernelContext_136)->colliders_0 = colliders_31;
    (&kernelContext_136)->collider_info_0 = collider_info_31;
    (&kernelContext_136)->view_chunks_0 = view_chunks_31;
    (&kernelContext_136)->view_0 = view_31;
    (&kernelContext_136)->objects_0 = objects_31;
    (&kernelContext_136)->shot_bodies_0 = shot_bodies_31;
    (&kernelContext_136)->lines_0 = lines_31;
    (&kernelContext_136)->geometry_0 = geometry_31;
    (&kernelContext_136)->shots_0 = shots_31;
    (&kernelContext_136)->lp_0 = lp_31;
    (&kernelContext_136)->ext_0 = ext_33;
    (&kernelContext_136)->engine_contacts_0 = engine_contacts_31;
    (&kernelContext_136)->crush_0 = crush_31;
    (&kernelContext_136)->crush_rw_0 = crush_rw_31;
    (&kernelContext_136)->crush_tab_0 = crush_tab_31;
    threadgroup array<uint, int(1024)> g_scan_31;
    (&kernelContext_136)->g_scan_0 = &g_scan_31;
    threadgroup array<float, int(64)> g_il_best_31;
    (&kernelContext_136)->g_il_best_0 = &g_il_best_31;
    uint b_41 = id_28.x;
    bool _S969;
    if(b_41 >= (params_31->body_count_0))
    {
        _S969 = true;
    }
    else
    {
        _S969 = ((&kernelContext_136)->params_0->shot_count_0) == 0U;
    }
    if(_S969)
    {
        return;
    }
    thread uint shot_7;
    bool _S970 = struck_0(b_41, &shot_7, &kernelContext_136);
    if(!_S970)
    {
        return;
    }
    uint _S971 = a_strike_0(&kernelContext_136);
    uint at_17 = _S971 + 9U * b_41;
    uint _S972 = atomic_load_explicit((&kernelContext_136)->atom_0+(at_17 + 6U), memory_order_relaxed);
    float t_in_7 = (as_type<float>((_S972)));
    uint entry_2 = atomic_load_explicit((&kernelContext_136)->atom_0+(at_17 + 8U), memory_order_relaxed);
    uint _S973 = strike_at_0(b_41, &kernelContext_136);
    if(!(t_in_7 <= ((&kernelContext_136)->params_0->dt_0)))
    {
        _S969 = true;
    }
    else
    {
        _S969 = entry_2 == 4294967295U;
    }
    if(_S969)
    {
        (*((&kernelContext_136)->frames_0+_S973))[int(3)] = 0.0f;
        return;
    }
    float3 rel_4 = (float4(*((&kernelContext_136)->frames_0+(_S973 + 1U))) ).xyz;
    float speed_14 = length(rel_4);
    float4 _S974 = float4(*((&kernelContext_136)->shots_0+2U * shot_7)) ;
    float _S975 = max(_S974.z, 0.00100000004749745f);
    float m_imp_0 = _S974.w;
    float e_imp_0 = (float4(*((&kernelContext_136)->shots_0+(2U * shot_7 + 1U))) ).x;
    float3 contact_16 = (float4(*((&kernelContext_136)->frames_0+_S973)) ).xyz + rel_4 * float3(t_in_7)  + rel_4 / float3(max(speed_14, 9.999999960041972e-13f))  * float3(_S974.y) ;
    float inv_mass_5 = (float4(*((&kernelContext_136)->bodies_0+7U * b_41)) ).w;
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
    float modulus_1 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_136)->geometry_0+(3U * entry_2 + 1U))) ).w);
    *((&kernelContext_136)->frames_0+(_S973 + 2U)) = packed_float4(float4(contact_16, m_eff_6 * speed_14 / (2.86999988555908203f * pow(m_eff_6 * m_eff_6 / (_S975 * modulus_1 * modulus_1 * max(speed_14, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

bool strike_share_breaks_0(uint4 bond_1, float3 contact_17, float force_2, KernelContext_0 thread* kernelContext_137)
{
    thread Quat_0 q_24;
    float3 _S976 = chunk_box_world_0(bond_1.x, &q_24, kernelContext_137);
    float3 _S977 = chunk_box_world_0(bond_1.y, &q_24, kernelContext_137);
    float dist_1 = length(float3(0.5f)  * (_S976 + _S977) - contact_17);
    return (force_2 * min(1.0f, (as_type<float>((bond_1.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((bond_1.z)));
}

[[kernel]] void fr_strike_breaks(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], atomic<uint> device* atom_32 [[buffer(5)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_32 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], packed_float4 device* lp_32 [[buffer(24)]], packed_float4 device* ext_34 [[buffer(23)]], packed_float4 device* engine_contacts_32 [[buffer(25)]], CrushParams_0 constant* crush_32 [[buffer(28)]], packed_float4 device* crush_rw_32 [[buffer(27)]], packed_float4 device* crush_tab_32 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_138;
    (&kernelContext_138)->params_0 = params_32;
    (&kernelContext_138)->atom_0 = atom_32;
    (&kernelContext_138)->work_0 = work_32;
    (&kernelContext_138)->frames_0 = frames_32;
    (&kernelContext_138)->bodies_0 = bodies_32;
    (&kernelContext_138)->chunks_0 = chunks_32;
    (&kernelContext_138)->sort_0 = sort_32;
    (&kernelContext_138)->manifolds_0 = manifolds_32;
    (&kernelContext_138)->loads_0 = loads_32;
    (&kernelContext_138)->bonds_0 = bonds_32;
    (&kernelContext_138)->args_0 = args_32;
    (&kernelContext_138)->info_0 = info_32;
    (&kernelContext_138)->scan_0 = scan_32;
    (&kernelContext_138)->parts_0 = parts_32;
    (&kernelContext_138)->colliders_0 = colliders_32;
    (&kernelContext_138)->collider_info_0 = collider_info_32;
    (&kernelContext_138)->view_chunks_0 = view_chunks_32;
    (&kernelContext_138)->view_0 = view_32;
    (&kernelContext_138)->objects_0 = objects_32;
    (&kernelContext_138)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_138)->lines_0 = lines_32;
    (&kernelContext_138)->geometry_0 = geometry_32;
    (&kernelContext_138)->shots_0 = shots_32;
    (&kernelContext_138)->lp_0 = lp_32;
    (&kernelContext_138)->ext_0 = ext_34;
    (&kernelContext_138)->engine_contacts_0 = engine_contacts_32;
    (&kernelContext_138)->crush_0 = crush_32;
    (&kernelContext_138)->crush_rw_0 = crush_rw_32;
    (&kernelContext_138)->crush_tab_0 = crush_tab_32;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_138)->g_scan_0 = &g_scan_32;
    threadgroup array<float, int(64)> g_il_best_32;
    (&kernelContext_138)->g_il_best_0 = &g_il_best_32;
    uint i_15 = id_29.x;
    bool _S978;
    if(i_15 >= (params_32->bond_used_0))
    {
        _S978 = true;
    }
    else
    {
        _S978 = ((&kernelContext_138)->params_0->shot_count_0) == 0U;
    }
    if(_S978)
    {
        _S978 = true;
    }
    else
    {
        uint _S979 = w_alive_0(&kernelContext_138);
        _S978 = (*((&kernelContext_138)->work_0+(_S979 + i_15))) == 0U;
    }
    if(_S978)
    {
        return;
    }
    uint4 _S980 = uint4(*((&kernelContext_138)->bonds_0+i_15)) ;
    uint _S981 = body_of_0(_S980.x, &kernelContext_138);
    thread uint shot_8;
    bool _S982 = struck_0(_S981, &shot_8, &kernelContext_138);
    if(!_S982)
    {
        return;
    }
    uint _S983 = strike_at_0(_S981, &kernelContext_138);
    float4 _S984 = float4(*((&kernelContext_138)->frames_0+(_S983 + 2U))) ;
    bool _S985 = strike_share_breaks_0(_S980, _S984.xyz, _S984.w, &kernelContext_138);
    if(_S985)
    {
        uint _S986 = w_alive_0(&kernelContext_138);
        *((&kernelContext_138)->work_0+(_S986 + i_15)) = 0U;
        uint _S987 = a_dirty_0(&kernelContext_138);
        uint _S988 = atomic_fetch_or_explicit((&kernelContext_138)->atom_0+(_S987 + _S981), 1U, memory_order_relaxed);
        uint _S989 = a_counters_1(&kernelContext_138);
        uint _S990 = atomic_fetch_add_explicit((&kernelContext_138)->atom_0+_S989, 1U, memory_order_relaxed);
        uint _S991 = atomic_fetch_add_explicit((&kernelContext_138)->atom_0+(_S989 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_15, &kernelContext_138);
    }
    return;
}

uint a_lp_0(uint b_42, KernelContext_0 thread* kernelContext_139)
{
    uint _S992 = a_strike_0(kernelContext_139);
    return _S992 + 9U * kernelContext_139->params_0->body_count_0 + 14U * b_42;
}

[[kernel]] void fr_lp_clear(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], atomic<uint> device* atom_33 [[buffer(5)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], packed_float4 device* lp_33 [[buffer(24)]], packed_float4 device* ext_35 [[buffer(23)]], packed_float4 device* engine_contacts_33 [[buffer(25)]], CrushParams_0 constant* crush_33 [[buffer(28)]], packed_float4 device* crush_rw_33 [[buffer(27)]], packed_float4 device* crush_tab_33 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_140;
    (&kernelContext_140)->params_0 = params_33;
    (&kernelContext_140)->atom_0 = atom_33;
    (&kernelContext_140)->work_0 = work_33;
    (&kernelContext_140)->frames_0 = frames_33;
    (&kernelContext_140)->bodies_0 = bodies_33;
    (&kernelContext_140)->chunks_0 = chunks_33;
    (&kernelContext_140)->sort_0 = sort_33;
    (&kernelContext_140)->manifolds_0 = manifolds_33;
    (&kernelContext_140)->loads_0 = loads_33;
    (&kernelContext_140)->bonds_0 = bonds_33;
    (&kernelContext_140)->args_0 = args_33;
    (&kernelContext_140)->info_0 = info_33;
    (&kernelContext_140)->scan_0 = scan_33;
    (&kernelContext_140)->parts_0 = parts_33;
    (&kernelContext_140)->colliders_0 = colliders_33;
    (&kernelContext_140)->collider_info_0 = collider_info_33;
    (&kernelContext_140)->view_chunks_0 = view_chunks_33;
    (&kernelContext_140)->view_0 = view_33;
    (&kernelContext_140)->objects_0 = objects_33;
    (&kernelContext_140)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_140)->lines_0 = lines_33;
    (&kernelContext_140)->geometry_0 = geometry_33;
    (&kernelContext_140)->shots_0 = shots_33;
    (&kernelContext_140)->lp_0 = lp_33;
    (&kernelContext_140)->ext_0 = ext_35;
    (&kernelContext_140)->engine_contacts_0 = engine_contacts_33;
    (&kernelContext_140)->crush_0 = crush_33;
    (&kernelContext_140)->crush_rw_0 = crush_rw_33;
    (&kernelContext_140)->crush_tab_0 = crush_tab_33;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_140)->g_scan_0 = &g_scan_33;
    threadgroup array<float, int(64)> g_il_best_33;
    (&kernelContext_140)->g_il_best_0 = &g_il_best_33;
    uint b_43 = id_30.x;
    if(b_43 >= (params_33->body_count_0))
    {
        return;
    }
    uint _S993 = a_lp_0(b_43, &kernelContext_140);
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
        atomic_store_explicit((&kernelContext_140)->atom_0+(_S993 + w_7), 0U, memory_order_relaxed);
        w_7 = w_7 + 1U;
    }
    atomic_store_explicit((&kernelContext_140)->atom_0+(_S993 + 12U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_140)->atom_0+(_S993 + 13U), 0U, memory_order_relaxed);
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

void add64w_0(uint at_18, float value_3, float scale_3, KernelContext_0 thread* kernelContext_141)
{
    float x_6 = round(value_3 * scale_3);
    float x_7;
    if(!((abs(x_6)) < 3.99999993722699776e+18f))
    {
        uint _S994 = a_counters_1(kernelContext_141);
        uint _S995 = atomic_fetch_or_explicit(kernelContext_141->atom_0+(_S994 + 3U), 2U, memory_order_relaxed);
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
        uint _S996 = ~hi_4;
        if(lo_6 == 0U)
        {
            lo_5 = 1U;
        }
        else
        {
            lo_5 = 0U;
        }
        uint _S997 = _S996 + lo_5;
        lo_5 = lo_6;
        hi_5 = _S997;
    }
    else
    {
        lo_5 = lo_4;
        hi_5 = hi_4;
    }
    uint old_1 = atomic_fetch_add_explicit(kernelContext_141->atom_0+at_18, lo_5, memory_order_relaxed);
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
        uint _S998 = atomic_fetch_add_explicit(kernelContext_141->atom_0+(at_18 + 1U), hi_6, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_lp_sum(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], atomic<uint> device* atom_34 [[buffer(5)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], packed_float4 device* lp_34 [[buffer(24)]], packed_float4 device* ext_36 [[buffer(23)]], packed_float4 device* engine_contacts_34 [[buffer(25)]], CrushParams_0 constant* crush_34 [[buffer(28)]], packed_float4 device* crush_rw_34 [[buffer(27)]], packed_float4 device* crush_tab_34 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_142;
    (&kernelContext_142)->params_0 = params_34;
    (&kernelContext_142)->atom_0 = atom_34;
    (&kernelContext_142)->work_0 = work_34;
    (&kernelContext_142)->frames_0 = frames_34;
    (&kernelContext_142)->bodies_0 = bodies_34;
    (&kernelContext_142)->chunks_0 = chunks_34;
    (&kernelContext_142)->sort_0 = sort_34;
    (&kernelContext_142)->manifolds_0 = manifolds_34;
    (&kernelContext_142)->loads_0 = loads_34;
    (&kernelContext_142)->bonds_0 = bonds_34;
    (&kernelContext_142)->args_0 = args_34;
    (&kernelContext_142)->info_0 = info_34;
    (&kernelContext_142)->scan_0 = scan_34;
    (&kernelContext_142)->parts_0 = parts_34;
    (&kernelContext_142)->colliders_0 = colliders_34;
    (&kernelContext_142)->collider_info_0 = collider_info_34;
    (&kernelContext_142)->view_chunks_0 = view_chunks_34;
    (&kernelContext_142)->view_0 = view_34;
    (&kernelContext_142)->objects_0 = objects_34;
    (&kernelContext_142)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_142)->lines_0 = lines_34;
    (&kernelContext_142)->geometry_0 = geometry_34;
    (&kernelContext_142)->shots_0 = shots_34;
    (&kernelContext_142)->lp_0 = lp_34;
    (&kernelContext_142)->ext_0 = ext_36;
    (&kernelContext_142)->engine_contacts_0 = engine_contacts_34;
    (&kernelContext_142)->crush_0 = crush_34;
    (&kernelContext_142)->crush_rw_0 = crush_rw_34;
    (&kernelContext_142)->crush_tab_0 = crush_tab_34;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_142)->g_scan_0 = &g_scan_34;
    threadgroup array<float, int(64)> g_il_best_34;
    (&kernelContext_142)->g_il_best_0 = &g_il_best_34;
    uint k_34 = id_31.x;
    if(k_34 >= (params_34->chunk_count_0))
    {
        return;
    }
    uint _S999 = body_of_0(k_34, &kernelContext_142);
    thread Quat_0 q_25;
    thread float3 origin_7;
    structure_pose_0(_S999, 7U * _S999, &q_25, &origin_7, &kernelContext_142);
    thread Quat_0 _S1000 = q_25;
    Quat_0 _S1001 = conj_0(&_S1000);
    float4 _S1002 = float4(*((&kernelContext_142)->chunks_0+4U * k_34)) ;
    uint _S1003 = l_vec_0(&kernelContext_142);
    uint _S1004 = _S1003 + 2U * k_34;
    float3 _S1005 = (float4(*((&kernelContext_142)->loads_0+_S1004)) ).xyz;
    thread Quat_0 _S1006 = _S1001;
    float3 _S1007 = rotate_0(&_S1006, _S1005);
    float _S1008 = _S1002.w;
    float3 _S1009 = float3((&kernelContext_142)->params_0->gx_0, (&kernelContext_142)->params_0->gy_0, (&kernelContext_142)->params_0->gz_0);
    thread Quat_0 _S1010 = _S1001;
    float3 _S1011 = rotate_0(&_S1010, _S1009);
    float3 f_10 = _S1007 + float3(_S1008)  * _S1011;
    float3 _S1012 = (float4(*((&kernelContext_142)->loads_0+(_S1004 + 1U))) ).xyz;
    thread Quat_0 _S1013 = _S1001;
    float3 _S1014 = rotate_0(&_S1013, _S1012);
    uint _S1015 = lp_f_0(k_34);
    *((&kernelContext_142)->lp_0+_S1015) = packed_float4(float4(f_10, 0.0f)) ;
    *((&kernelContext_142)->lp_0+(_S1015 + 1U)) = packed_float4(float4(_S1014, 0.0f)) ;
    float3 _S1016 = _S1014 + cross(_S1002.xyz - (float4(*((&kernelContext_142)->frames_0+2U * _S999)) ).xyz, f_10);
    uint _S1017 = a_lp_0(_S999, &kernelContext_142);
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
        uint _S1018 = 2U * a_8;
        add64w_0(_S1017 + _S1018, f_10[a_8], (&kernelContext_142)->params_0->lp_scale_0, &kernelContext_142);
        add64w_0(_S1017 + 6U + _S1018, _S1016[a_8], (&kernelContext_142)->params_0->lp_scale_0, &kernelContext_142);
        a_8 = a_8 + 1U;
    }
    uint _S1019 = atomic_fetch_min_explicit((&kernelContext_142)->atom_0+(_S1017 + 12U), k_34, memory_order_relaxed);
    if(((&kernelContext_142)->info_0[k_34] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_142)->atom_0+(_S1017 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

float3 read64w3_0(uint at_19, float scale_4, KernelContext_0 thread* kernelContext_143)
{
    float _S1020 = read64_1(at_19, scale_4, kernelContext_143);
    float _S1021 = read64_1(at_19 + 2U, scale_4, kernelContext_143);
    float _S1022 = read64_1(at_19 + 4U, scale_4, kernelContext_143);
    return float3(_S1020, _S1021, _S1022);
}

[[kernel]] void fr_lp_relief(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], atomic<uint> device* atom_35 [[buffer(5)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], packed_float4 device* lp_35 [[buffer(24)]], packed_float4 device* ext_37 [[buffer(23)]], packed_float4 device* engine_contacts_35 [[buffer(25)]], CrushParams_0 constant* crush_35 [[buffer(28)]], packed_float4 device* crush_rw_35 [[buffer(27)]], packed_float4 device* crush_tab_35 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_144;
    (&kernelContext_144)->params_0 = params_35;
    (&kernelContext_144)->atom_0 = atom_35;
    (&kernelContext_144)->work_0 = work_35;
    (&kernelContext_144)->frames_0 = frames_35;
    (&kernelContext_144)->bodies_0 = bodies_35;
    (&kernelContext_144)->chunks_0 = chunks_35;
    (&kernelContext_144)->sort_0 = sort_35;
    (&kernelContext_144)->manifolds_0 = manifolds_35;
    (&kernelContext_144)->loads_0 = loads_35;
    (&kernelContext_144)->bonds_0 = bonds_35;
    (&kernelContext_144)->args_0 = args_35;
    (&kernelContext_144)->info_0 = info_35;
    (&kernelContext_144)->scan_0 = scan_35;
    (&kernelContext_144)->parts_0 = parts_35;
    (&kernelContext_144)->colliders_0 = colliders_35;
    (&kernelContext_144)->collider_info_0 = collider_info_35;
    (&kernelContext_144)->view_chunks_0 = view_chunks_35;
    (&kernelContext_144)->view_0 = view_35;
    (&kernelContext_144)->objects_0 = objects_35;
    (&kernelContext_144)->shot_bodies_0 = shot_bodies_35;
    (&kernelContext_144)->lines_0 = lines_35;
    (&kernelContext_144)->geometry_0 = geometry_35;
    (&kernelContext_144)->shots_0 = shots_35;
    (&kernelContext_144)->lp_0 = lp_35;
    (&kernelContext_144)->ext_0 = ext_37;
    (&kernelContext_144)->engine_contacts_0 = engine_contacts_35;
    (&kernelContext_144)->crush_0 = crush_35;
    (&kernelContext_144)->crush_rw_0 = crush_rw_35;
    (&kernelContext_144)->crush_tab_0 = crush_tab_35;
    threadgroup array<uint, int(1024)> g_scan_35;
    (&kernelContext_144)->g_scan_0 = &g_scan_35;
    threadgroup array<float, int(64)> g_il_best_35;
    (&kernelContext_144)->g_il_best_0 = &g_il_best_35;
    uint k_35 = id_32.x;
    if(k_35 >= (params_35->chunk_count_0))
    {
        return;
    }
    uint _S1023 = body_of_0(k_35, &kernelContext_144);
    uint _S1024 = a_lp_0(_S1023, &kernelContext_144);
    uint _S1025 = atomic_load_explicit((&kernelContext_144)->atom_0+(_S1024 + 13U), memory_order_relaxed);
    bool supported_0 = _S1025 != 0U;
    bool fixed_1;
    if(((&kernelContext_144)->info_0[k_35] & 1U) != 0U)
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            uint _S1026 = atomic_load_explicit((&kernelContext_144)->atom_0+(_S1024 + 12U), memory_order_relaxed);
            fixed_1 = _S1026 == k_35;
        }
        else
        {
            fixed_1 = false;
        }
    }
    uint _S1027 = lp_f_0(k_35);
    thread float4 f_11 = float4(*((&kernelContext_144)->lp_0+_S1027)) ;
    uint _S1028 = _S1027 + 1U;
    thread float4 t_11 = float4(*((&kernelContext_144)->lp_0+_S1028)) ;
    uint b_44 = 7U * _S1023;
    float inv_mass_6 = (float4(*((&kernelContext_144)->bodies_0+b_44)) ).w;
    bool _S1029;
    if(!supported_0)
    {
        _S1029 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S1029 = false;
    }
    if(_S1029)
    {
        float3 _S1030 = read64w3_0(_S1024, (&kernelContext_144)->params_0->lp_scale_0, &kernelContext_144);
        float3 _S1031 = read64w3_0(_S1024 + 6U, (&kernelContext_144)->params_0->lp_scale_0, &kernelContext_144);
        uint _S1032 = 2U * _S1023;
        Quat_0 p_15 = quat_of_0(float4(*((&kernelContext_144)->frames_0+(_S1032 + 1U))) );
        float3 _S1033 = (float4(*((&kernelContext_144)->bodies_0+(b_44 + 2U))) ).xyz;
        thread Quat_0 _S1034 = p_15;
        Quat_0 _S1035 = conj_0(&_S1034);
        thread Quat_0 _S1036 = _S1035;
        float3 _S1037 = rotate_0(&_S1036, _S1031);
        float3 _S1038 = _S1033 * _S1037;
        thread Quat_0 _S1039 = p_15;
        float3 _S1040 = rotate_0(&_S1039, _S1038);
        uint _S1041 = 4U * k_35;
        float4 _S1042 = float4(*((&kernelContext_144)->chunks_0+_S1041)) ;
        float4 _S1043 = float4(*((&kernelContext_144)->chunks_0+(_S1041 + 1U))) ;
        float4 _S1044 = float4(*((&kernelContext_144)->chunks_0+(_S1041 + 2U))) ;
        f_11.xyz = f_11.xyz - float3(_S1042.w)  * (_S1030 * float3(inv_mass_6)  + cross(_S1040, _S1042.xyz - (float4(*((&kernelContext_144)->frames_0+_S1032)) ).xyz));
        float _S1045 = _S1040.x;
        float _S1046 = _S1043.w;
        float _S1047 = _S1040.y;
        float _S1048 = _S1044.x;
        float _S1049 = _S1040.z;
        float _S1050 = _S1044.y;
        t_11.xyz = t_11.xyz - float3(_S1043.x * _S1045 + _S1046 * _S1047 + _S1048 * _S1049, _S1046 * _S1045 + _S1043.y * _S1047 + _S1050 * _S1049, _S1048 * _S1045 + _S1050 * _S1047 + _S1043.z * _S1049);
    }
    float _S1051;
    if(fixed_1)
    {
        _S1051 = 1.0f;
    }
    else
    {
        _S1051 = 0.0f;
    }
    f_11.w = _S1051;
    *((&kernelContext_144)->lp_0+_S1027) = packed_float4(f_11) ;
    *((&kernelContext_144)->lp_0+_S1028) = packed_float4(t_11) ;
    return;
}

uint adj_bond_0(uint at_20, KernelContext_0 thread* kernelContext_145)
{
    return (as_type<uint>(((float4(*(kernelContext_145->ext_0+(kernelContext_145->params_0->ext_adj_0 + at_20 / 4U))) )[at_20 % 4U])));
}

uint lp_intact_0(uint k_36, KernelContext_0 thread* kernelContext_146)
{
    float4 _S1052 = float4(*(kernelContext_146->ext_0+(kernelContext_146->params_0->ext_chunks_0 + 4U * k_36))) ;
    uint _S1053 = (as_type<uint>((_S1052.x)));
    uint _S1054 = (as_type<uint>((_S1052.y)));
    uint e_4 = 0U;
    uint intact_0 = 0U;
    for(;;)
    {
        if(e_4 < _S1054)
        {
        }
        else
        {
            break;
        }
        uint _S1055 = w_alive_0(kernelContext_146);
        uint _S1056 = adj_bond_0(_S1053 + e_4, kernelContext_146);
        uint _S1057;
        if((*(kernelContext_146->work_0+(_S1055 + _S1056))) != 0U)
        {
            _S1057 = 1U;
        }
        else
        {
            _S1057 = 0U;
        }
        uint intact_1 = intact_0 + _S1057;
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

LpBond_0 lp_bond_0(uint i_16, KernelContext_0 thread* kernelContext_147)
{
    uint o_7 = 10U * i_16;
    thread LpBond_0 b_45;
    (&b_45)->centroid_0 = (float4(*(kernelContext_147->ext_0+o_7)) ).xyz;
    float3 _S1058 = (float4(*(kernelContext_147->ext_0+(o_7 + 1U))) ).xyz;
    (&b_45)->n_4 = _S1058;
    float3 _S1059 = (float4(*(kernelContext_147->ext_0+(o_7 + 2U))) ).xyz;
    (&b_45)->t1_2 = _S1059;
    (&b_45)->t2_0 = cross(_S1058, _S1059);
    (&(&b_45)->law_0)->geom0_0 = float4(*(kernelContext_147->ext_0+(o_7 + 3U))) ;
    (&(&b_45)->law_0)->geom1_0 = float4(*(kernelContext_147->ext_0+(o_7 + 4U))) ;
    (&(&b_45)->law_0)->stiff0_0 = float4(*(kernelContext_147->ext_0+(o_7 + 5U))) ;
    (&(&b_45)->law_0)->stiff1_0 = float4(*(kernelContext_147->ext_0+(o_7 + 6U))) ;
    (&(&b_45)->law_0)->rebar0_0 = float4(*(kernelContext_147->ext_0+(o_7 + 7U))) ;
    (&(&b_45)->law_0)->rebar1_0 = float4(*(kernelContext_147->ext_0+(o_7 + 8U))) ;
    (&(&b_45)->law_0)->ids_3 = (as_type<uint4>((float4(*(kernelContext_147->ext_0+(o_7 + 9U))) )));
    return b_45;
}

uint tri_0(uint i_17, uint j_6)
{
    return i_17 * (i_17 + 1U) / 2U + j_6;
}

bool lp_factor_0(uint k_38, KernelContext_0 thread* kernelContext_148)
{
    uint q_26;
    uint a_9;
    uint p2_0;
    uint q2_0;
    uint a2_0;
    float s_11;
    uint _S1060 = lp_intact_0(k_38, kernelContext_148);
    uint _S1061 = lp_l_0(k_38);
    uint _S1062 = _S1060 + 1U;
    if((as_type<uint>(((float4(*(kernelContext_148->lp_0+(_S1061 + 5U))) ).w))) == _S1062)
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
    uint _S1063 = 4U * k_38;
    float4 _S1064 = float4(*(kernelContext_148->ext_0+(kernelContext_148->params_0->ext_chunks_0 + _S1063))) ;
    uint _S1065 = (as_type<uint>((_S1064.x)));
    uint _S1066 = (as_type<uint>((_S1064.y)));
    q_26 = 0U;
    for(;;)
    {
        if(q_26 < _S1066)
        {
        }
        else
        {
            break;
        }
        uint _S1067 = adj_bond_0(_S1065 + q_26, kernelContext_148);
        uint _S1068 = w_alive_0(kernelContext_148);
        if((*(kernelContext_148->work_0+(_S1068 + _S1067))) == 0U)
        {
            q_26 = q_26 + 1U;
            continue;
        }
        LpBond_0 _S1069 = lp_bond_0(_S1067, kernelContext_148);
        float3 _S1070 = _S1069.centroid_0 - (float4(*(kernelContext_148->chunks_0+_S1063)) ).xyz;
        float _S1071 = _S1069.law_0.stiff0_0.y;
        array<float, int(6)> _S1072 = { { _S1071, _S1071, _S1069.law_0.stiff0_0.x, _S1069.law_0.stiff0_0.z, _S1069.law_0.stiff0_0.w, _S1069.law_0.stiff1_0.x } };
        array<float3, int(3)> _S1073 = { { _S1069.t1_2, _S1069.t2_0, _S1069.n_4 } };
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
            float3 c_20 = cross(_S1070, _S1073[a_9]);
            float _S1074 = _S1073[a_9].x;
            j_7[a_9][int(0)] = _S1074;
            float _S1075 = _S1073[a_9].y;
            j_7[a_9][int(1)] = _S1075;
            float _S1076 = _S1073[a_9].z;
            j_7[a_9][int(2)] = _S1076;
            j_7[a_9][int(3)] = c_20.x;
            j_7[a_9][int(4)] = c_20.y;
            j_7[a_9][int(5)] = c_20.z;
            j_7[3U + a_9][int(0)] = 0.0f;
            j_7[3U + a_9][int(1)] = 0.0f;
            j_7[3U + a_9][int(2)] = 0.0f;
            j_7[3U + a_9][int(3)] = _S1074;
            j_7[3U + a_9][int(4)] = _S1075;
            j_7[3U + a_9][int(5)] = _S1076;
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
                    float s_12 = s_11 + j_7[a2_0][p2_0] * _S1072[a2_0] * j_7[a2_0][q2_0];
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
                bool _S1077;
                if(ok_1)
                {
                    _S1077 = s_11 > 0.0f;
                }
                else
                {
                    _S1077 = false;
                }
                l_0[tri_0(p2_0, p2_0)] = sqrt(max(s_11, 1.00000000317107685e-30f));
                ok_1 = _S1077;
            }
            else
            {
                l_0[tri_0(p2_0, q2_0)] = s_11 / l_0[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + 1U;
        }
        uint _S1078 = p2_0 + 1U;
        ok_0 = ok_1;
        p2_0 = _S1078;
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
        packed_float4 device* _S1079 = kernelContext_148->lp_0+(_S1061 + q2_0);
        uint _S1080 = 4U * q2_0;
        float _S1081 = l_0[_S1080];
        uint _S1082 = _S1080 + 1U;
        if(_S1082 < 21U)
        {
            s_11 = l_0[_S1082];
        }
        else
        {
            s_11 = 0.0f;
        }
        uint _S1083 = _S1080 + 2U;
        float _S1084;
        if(_S1083 < 21U)
        {
            _S1084 = l_0[_S1083];
        }
        else
        {
            _S1084 = 0.0f;
        }
        float _S1085;
        if(q2_0 == 5U)
        {
            _S1085 = (as_type<float>((_S1062)));
        }
        else
        {
            _S1085 = l_0[_S1080 + 3U];
        }
        *_S1079 = packed_float4(float4(_S1081, s_11, _S1084, _S1085)) ;
        q2_0 = q2_0 + 1U;
    }
    return true;
}

[[kernel]] void fr_lp_block(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], atomic<uint> device* atom_36 [[buffer(5)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], packed_float4 device* lp_36 [[buffer(24)]], packed_float4 device* ext_38 [[buffer(23)]], packed_float4 device* engine_contacts_36 [[buffer(25)]], CrushParams_0 constant* crush_36 [[buffer(28)]], packed_float4 device* crush_rw_36 [[buffer(27)]], packed_float4 device* crush_tab_36 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_149;
    (&kernelContext_149)->params_0 = params_36;
    (&kernelContext_149)->atom_0 = atom_36;
    (&kernelContext_149)->work_0 = work_36;
    (&kernelContext_149)->frames_0 = frames_36;
    (&kernelContext_149)->bodies_0 = bodies_36;
    (&kernelContext_149)->chunks_0 = chunks_36;
    (&kernelContext_149)->sort_0 = sort_36;
    (&kernelContext_149)->manifolds_0 = manifolds_36;
    (&kernelContext_149)->loads_0 = loads_36;
    (&kernelContext_149)->bonds_0 = bonds_36;
    (&kernelContext_149)->args_0 = args_36;
    (&kernelContext_149)->info_0 = info_36;
    (&kernelContext_149)->scan_0 = scan_36;
    (&kernelContext_149)->parts_0 = parts_36;
    (&kernelContext_149)->colliders_0 = colliders_36;
    (&kernelContext_149)->collider_info_0 = collider_info_36;
    (&kernelContext_149)->view_chunks_0 = view_chunks_36;
    (&kernelContext_149)->view_0 = view_36;
    (&kernelContext_149)->objects_0 = objects_36;
    (&kernelContext_149)->shot_bodies_0 = shot_bodies_36;
    (&kernelContext_149)->lines_0 = lines_36;
    (&kernelContext_149)->geometry_0 = geometry_36;
    (&kernelContext_149)->shots_0 = shots_36;
    (&kernelContext_149)->lp_0 = lp_36;
    (&kernelContext_149)->ext_0 = ext_38;
    (&kernelContext_149)->engine_contacts_0 = engine_contacts_36;
    (&kernelContext_149)->crush_0 = crush_36;
    (&kernelContext_149)->crush_rw_0 = crush_rw_36;
    (&kernelContext_149)->crush_tab_0 = crush_tab_36;
    threadgroup array<uint, int(1024)> g_scan_36;
    (&kernelContext_149)->g_scan_0 = &g_scan_36;
    threadgroup array<float, int(64)> g_il_best_36;
    (&kernelContext_149)->g_il_best_0 = &g_il_best_36;
    uint k_39 = id_33.x;
    bool _S1086;
    if(k_39 >= (params_36->chunk_count_0))
    {
        _S1086 = true;
    }
    else
    {
        _S1086 = ((float4(*((&kernelContext_149)->lp_0+lp_f_0(k_39))) ).w) != 0.0f;
    }
    if(_S1086)
    {
        return;
    }
    bool _S1087 = lp_factor_0(k_39, &kernelContext_149);
    if(!_S1087)
    {
        (*((&kernelContext_149)->lp_0+lp_f_0(k_39)))[int(3)] = 1.0f;
        uint _S1088 = a_counters_1(&kernelContext_149);
        uint _S1089 = atomic_fetch_or_explicit((&kernelContext_149)->atom_0+(_S1088 + 3U), 64U, memory_order_relaxed);
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

float4 lp_side_0(uint i_18, KernelContext_0 thread* kernelContext_150)
{
    return float4(*(kernelContext_150->ext_0+(kernelContext_150->params_0->ext_sides_0 + i_18))) ;
}

void st_sweep_0(uint k_42, uint state_3, uint from_1, float c1_1, float c2_1, KernelContext_0 thread* kernelContext_151)
{
    uint ka_0;
    uint kb_0;
    float s_14;
    uint to_1 = 1U - from_1;
    uint _S1090 = st_x_0(k_42, state_3, from_1);
    float4 _S1091 = float4(*(kernelContext_151->lp_0+_S1090)) ;
    float4 _S1092 = float4(*(kernelContext_151->lp_0+(_S1090 + 1U))) ;
    uint _S1093 = st_f_0(k_42, state_3);
    float4 _S1094 = float4(*(kernelContext_151->lp_0+_S1093)) ;
    if((_S1094.w) != 0.0f)
    {
        uint _S1095 = st_x_0(k_42, state_3, to_1);
        *(kernelContext_151->lp_0+_S1095) = packed_float4(_S1091) ;
        *(kernelContext_151->lp_0+(_S1095 + 1U)) = packed_float4(_S1092) ;
        uint _S1096 = st_d_0(k_42, state_3);
        packed_float4 _S1097 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_151->lp_0+_S1096) = _S1097;
        *(kernelContext_151->lp_0+(_S1096 + 1U)) = _S1097;
        return;
    }
    float _S1098 = _S1094.y;
    float _S1099 = _S1094.z;
    thread array<float, int(6)> r_23;
    r_23[int(0)] = _S1094.x;
    r_23[int(1)] = _S1098;
    r_23[int(2)] = _S1099;
    r_23[int(3)] = 0.0f;
    r_23[int(4)] = 0.0f;
    r_23[int(5)] = 0.0f;
    float3 t_12 = (float4(*(kernelContext_151->lp_0+(_S1093 + 1U))) ).xyz;
    r_23[int(3)] = t_12.x;
    r_23[int(4)] = t_12.y;
    r_23[int(5)] = t_12.z;
    float4 _S1100 = float4(*(kernelContext_151->ext_0+(kernelContext_151->params_0->ext_chunks_0 + 4U * k_42))) ;
    uint _S1101 = (as_type<uint>((_S1100.x)));
    uint _S1102 = (as_type<uint>((_S1100.y)));
    uint e_5 = 0U;
    for(;;)
    {
        if(e_5 < _S1102)
        {
        }
        else
        {
            break;
        }
        uint _S1103 = 5U * (_S1101 + e_5);
        float4 _S1104 = lp_side_0(_S1103, kernelContext_151);
        uint _S1105 = w_alive_0(kernelContext_151);
        if((*(kernelContext_151->work_0+(_S1105 + (as_type<uint>((_S1104.w)))))) == 0U)
        {
            e_5 = e_5 + 1U;
            continue;
        }
        float4 _S1106 = lp_side_0(_S1103 + 1U, kernelContext_151);
        float4 _S1107 = lp_side_0(_S1103 + 2U, kernelContext_151);
        float4 _S1108 = lp_side_0(_S1103 + 3U, kernelContext_151);
        float4 _S1109 = lp_side_0(_S1103 + 4U, kernelContext_151);
        bool first_side_0 = (_S1109.w) != 0.0f;
        if(first_side_0)
        {
            ka_0 = k_42;
        }
        else
        {
            ka_0 = (as_type<uint>((_S1106.w)));
        }
        if(first_side_0)
        {
            kb_0 = (as_type<uint>((_S1106.w)));
        }
        else
        {
            kb_0 = k_42;
        }
        float3 ra_0 = _S1104.xyz;
        float3 rb_0 = _S1106.xyz;
        float3 t1_3 = _S1107.xyz;
        float3 n_5 = _S1108.xyz;
        float3 t2_1 = cross(n_5, t1_3);
        uint _S1110 = st_x_0(ka_0, state_3, from_1);
        uint _S1111 = st_x_0(kb_0, state_3, from_1);
        float3 _S1112 = (float4(*(kernelContext_151->lp_0+(_S1111 + 1U))) ).xyz;
        float3 _S1113 = (float4(*(kernelContext_151->lp_0+(_S1110 + 1U))) ).xyz;
        float3 delta_0 = (float4(*(kernelContext_151->lp_0+_S1111)) ).xyz + cross(_S1112, rb_0) - ((float4(*(kernelContext_151->lp_0+_S1110)) ).xyz + cross(_S1113, ra_0));
        float3 phi_0 = _S1112 - _S1113;
        float _S1114 = _S1107.w;
        float3 fb_0 = t1_3 * float3((_S1114 * dot(delta_0, t1_3)))  + t2_1 * float3((_S1114 * dot(delta_0, t2_1)))  + n_5 * float3((_S1108.w * dot(delta_0, n_5))) ;
        float3 mb_0 = t1_3 * float3((_S1109.x * dot(phi_0, t1_3)))  + t2_1 * float3((_S1109.y * dot(phi_0, t2_1)))  + n_5 * float3((_S1109.z * dot(phi_0, n_5))) ;
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
        float3 _S1115;
        if(first_side_0)
        {
            _S1115 = mb_0;
        }
        else
        {
            _S1115 = - mb_0;
        }
        float3 lm_0 = _S1115 + cross(arm_0, lf_0);
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
        float4 _S1116 = float4(*(kernelContext_151->lp_0+(lp_l_0(k_42) + kb_0))) ;
        uint _S1117 = 4U * kb_0;
        l_1[_S1117] = _S1116.x;
        uint _S1118 = _S1117 + 1U;
        if(_S1118 < 21U)
        {
            l_1[_S1118] = _S1116.y;
        }
        uint _S1119 = _S1117 + 2U;
        if(_S1119 < 21U)
        {
            l_1[_S1119] = _S1116.z;
        }
        uint _S1120 = _S1117 + 3U;
        if(_S1120 < 21U)
        {
            l_1[_S1120] = _S1116.w;
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
        uint _S1121 = uint(i3_0);
        z_3[i3_0] = s_14 / l_1[tri_0(_S1121, _S1121)];
        i3_0 = i3_0 - int(1);
    }
    uint _S1122 = st_d_0(k_42, state_3);
    float4 _S1123 = float4(*(kernelContext_151->lp_0+_S1122)) ;
    thread float4 d0_0 = _S1123;
    uint _S1124 = _S1122 + 1U;
    float4 _S1125 = float4(*(kernelContext_151->lp_0+_S1124)) ;
    thread float4 d1_0 = _S1125;
    float3 _S1126 = float3(c1_1) ;
    float3 _S1127 = float3(c2_1) ;
    d0_0.xyz = _S1126 * _S1123.xyz + _S1127 * float3(z_3[int(0)], z_3[int(1)], z_3[int(2)]);
    d1_0.xyz = _S1126 * _S1125.xyz + _S1127 * float3(z_3[int(3)], z_3[int(4)], z_3[int(5)]);
    *(kernelContext_151->lp_0+_S1122) = packed_float4(d0_0) ;
    *(kernelContext_151->lp_0+_S1124) = packed_float4(d1_0) ;
    uint _S1128 = st_x_0(k_42, state_3, to_1);
    *(kernelContext_151->lp_0+_S1128) = packed_float4(float4(_S1091.xyz + d0_0.xyz, 0.0f)) ;
    *(kernelContext_151->lp_0+(_S1128 + 1U)) = packed_float4(float4(_S1092.xyz + d1_0.xyz, 0.0f)) ;
    return;
}

void lp_sweep_0(uint k_43, uint from_2, KernelContext_0 thread* kernelContext_152)
{
    st_sweep_0(k_43, 0U, from_2, kernelContext_152->params_0->lp_c1_0, kernelContext_152->params_0->lp_c2_0, kernelContext_152);
    return;
}

[[kernel]] void fr_lp_sweep_even(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], atomic<uint> device* atom_37 [[buffer(5)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], packed_float4 device* lp_37 [[buffer(24)]], packed_float4 device* ext_39 [[buffer(23)]], packed_float4 device* engine_contacts_37 [[buffer(25)]], CrushParams_0 constant* crush_37 [[buffer(28)]], packed_float4 device* crush_rw_37 [[buffer(27)]], packed_float4 device* crush_tab_37 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_153;
    (&kernelContext_153)->params_0 = params_37;
    (&kernelContext_153)->atom_0 = atom_37;
    (&kernelContext_153)->work_0 = work_37;
    (&kernelContext_153)->frames_0 = frames_37;
    (&kernelContext_153)->bodies_0 = bodies_37;
    (&kernelContext_153)->chunks_0 = chunks_37;
    (&kernelContext_153)->sort_0 = sort_37;
    (&kernelContext_153)->manifolds_0 = manifolds_37;
    (&kernelContext_153)->loads_0 = loads_37;
    (&kernelContext_153)->bonds_0 = bonds_37;
    (&kernelContext_153)->args_0 = args_37;
    (&kernelContext_153)->info_0 = info_37;
    (&kernelContext_153)->scan_0 = scan_37;
    (&kernelContext_153)->parts_0 = parts_37;
    (&kernelContext_153)->colliders_0 = colliders_37;
    (&kernelContext_153)->collider_info_0 = collider_info_37;
    (&kernelContext_153)->view_chunks_0 = view_chunks_37;
    (&kernelContext_153)->view_0 = view_37;
    (&kernelContext_153)->objects_0 = objects_37;
    (&kernelContext_153)->shot_bodies_0 = shot_bodies_37;
    (&kernelContext_153)->lines_0 = lines_37;
    (&kernelContext_153)->geometry_0 = geometry_37;
    (&kernelContext_153)->shots_0 = shots_37;
    (&kernelContext_153)->lp_0 = lp_37;
    (&kernelContext_153)->ext_0 = ext_39;
    (&kernelContext_153)->engine_contacts_0 = engine_contacts_37;
    (&kernelContext_153)->crush_0 = crush_37;
    (&kernelContext_153)->crush_rw_0 = crush_rw_37;
    (&kernelContext_153)->crush_tab_0 = crush_tab_37;
    threadgroup array<uint, int(1024)> g_scan_37;
    (&kernelContext_153)->g_scan_0 = &g_scan_37;
    threadgroup array<float, int(64)> g_il_best_37;
    (&kernelContext_153)->g_il_best_0 = &g_il_best_37;
    uint _S1129 = id_34.x;
    if(_S1129 < (params_37->chunk_count_0))
    {
        lp_sweep_0(_S1129, 0U, &kernelContext_153);
    }
    return;
}

[[kernel]] void fr_lp_sweep_odd(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], atomic<uint> device* atom_38 [[buffer(5)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], packed_float4 device* lp_38 [[buffer(24)]], packed_float4 device* ext_40 [[buffer(23)]], packed_float4 device* engine_contacts_38 [[buffer(25)]], CrushParams_0 constant* crush_38 [[buffer(28)]], packed_float4 device* crush_rw_38 [[buffer(27)]], packed_float4 device* crush_tab_38 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_154;
    (&kernelContext_154)->params_0 = params_38;
    (&kernelContext_154)->atom_0 = atom_38;
    (&kernelContext_154)->work_0 = work_38;
    (&kernelContext_154)->frames_0 = frames_38;
    (&kernelContext_154)->bodies_0 = bodies_38;
    (&kernelContext_154)->chunks_0 = chunks_38;
    (&kernelContext_154)->sort_0 = sort_38;
    (&kernelContext_154)->manifolds_0 = manifolds_38;
    (&kernelContext_154)->loads_0 = loads_38;
    (&kernelContext_154)->bonds_0 = bonds_38;
    (&kernelContext_154)->args_0 = args_38;
    (&kernelContext_154)->info_0 = info_38;
    (&kernelContext_154)->scan_0 = scan_38;
    (&kernelContext_154)->parts_0 = parts_38;
    (&kernelContext_154)->colliders_0 = colliders_38;
    (&kernelContext_154)->collider_info_0 = collider_info_38;
    (&kernelContext_154)->view_chunks_0 = view_chunks_38;
    (&kernelContext_154)->view_0 = view_38;
    (&kernelContext_154)->objects_0 = objects_38;
    (&kernelContext_154)->shot_bodies_0 = shot_bodies_38;
    (&kernelContext_154)->lines_0 = lines_38;
    (&kernelContext_154)->geometry_0 = geometry_38;
    (&kernelContext_154)->shots_0 = shots_38;
    (&kernelContext_154)->lp_0 = lp_38;
    (&kernelContext_154)->ext_0 = ext_40;
    (&kernelContext_154)->engine_contacts_0 = engine_contacts_38;
    (&kernelContext_154)->crush_0 = crush_38;
    (&kernelContext_154)->crush_rw_0 = crush_rw_38;
    (&kernelContext_154)->crush_tab_0 = crush_tab_38;
    threadgroup array<uint, int(1024)> g_scan_38;
    (&kernelContext_154)->g_scan_0 = &g_scan_38;
    threadgroup array<float, int(64)> g_il_best_38;
    (&kernelContext_154)->g_il_best_0 = &g_il_best_38;
    uint _S1130 = id_35.x;
    if(_S1130 < (params_38->chunk_count_0))
    {
        lp_sweep_0(_S1130, 1U, &kernelContext_154);
    }
    return;
}

void st_force_0(const LpBond_0 thread* b_46, uint ka_1, uint kb_1, uint state_4, uint c_21, float3 thread* q_lin_0, float3 thread* q_ang_0, KernelContext_0 thread* kernelContext_155)
{
    uint _S1131 = st_x_0(ka_1, state_4, c_21);
    uint _S1132 = st_x_0(kb_1, state_4, c_21);
    float3 _S1133 = (float4(*(kernelContext_155->lp_0+(_S1132 + 1U))) ).xyz;
    float3 _S1134 = (float4(*(kernelContext_155->lp_0+(_S1131 + 1U))) ).xyz;
    float3 delta_1 = (float4(*(kernelContext_155->lp_0+_S1132)) ).xyz + cross(_S1133, b_46->centroid_0 - (float4(*(kernelContext_155->chunks_0+4U * kb_1)) ).xyz) - ((float4(*(kernelContext_155->lp_0+_S1131)) ).xyz + cross(_S1134, b_46->centroid_0 - (float4(*(kernelContext_155->chunks_0+4U * ka_1)) ).xyz));
    float3 phi_1 = _S1133 - _S1134;
    float4 _S1135 = (&b_46->law_0)->stiff0_0;
    float kt_0 = (&b_46->law_0)->stiff1_0.x;
    float _S1136 = (&b_46->law_0)->stiff0_0.y;
    float3 _S1137 = b_46->t1_2;
    float3 _S1138 = b_46->t2_0;
    float3 _S1139 = b_46->n_4;
    *q_lin_0 = float3(_S1136 * dot(delta_1, b_46->t1_2), _S1136 * dot(delta_1, b_46->t2_0), (&b_46->law_0)->stiff0_0.x * dot(delta_1, b_46->n_4));
    *q_ang_0 = float3(_S1135.z * dot(phi_1, _S1137), _S1135.w * dot(phi_1, _S1138), kt_0 * dot(phi_1, _S1139));
    return;
}

void st_force_1(const LpBond_0 thread* b_47, uint ka_2, uint kb_2, uint state_5, uint c_22, float3 thread* q_lin_1, float3 thread* q_ang_1, KernelContext_0 thread* kernelContext_156)
{
    uint _S1140 = st_x_0(ka_2, state_5, c_22);
    uint _S1141 = st_x_0(kb_2, state_5, c_22);
    float3 _S1142 = (float4(*(kernelContext_156->lp_0+(_S1141 + 1U))) ).xyz;
    float3 _S1143 = (float4(*(kernelContext_156->lp_0+(_S1140 + 1U))) ).xyz;
    float3 delta_2 = (float4(*(kernelContext_156->lp_0+_S1141)) ).xyz + cross(_S1142, b_47->centroid_0 - (float4(*(kernelContext_156->chunks_0+4U * kb_2)) ).xyz) - ((float4(*(kernelContext_156->lp_0+_S1140)) ).xyz + cross(_S1143, b_47->centroid_0 - (float4(*(kernelContext_156->chunks_0+4U * ka_2)) ).xyz));
    float3 phi_2 = _S1142 - _S1143;
    float4 _S1144 = (&b_47->law_0)->stiff0_0;
    float kt_1 = (&b_47->law_0)->stiff1_0.x;
    float _S1145 = (&b_47->law_0)->stiff0_0.y;
    float3 _S1146 = b_47->t1_2;
    float3 _S1147 = b_47->t2_0;
    float3 _S1148 = b_47->n_4;
    *q_lin_1 = float3(_S1145 * dot(delta_2, b_47->t1_2), _S1145 * dot(delta_2, b_47->t2_0), (&b_47->law_0)->stiff0_0.x * dot(delta_2, b_47->n_4));
    *q_ang_1 = float3(_S1144.z * dot(phi_2, _S1146), _S1144.w * dot(phi_2, _S1147), kt_1 * dot(phi_2, _S1148));
    return;
}

void lp_force_0(const LpBond_0 thread* b_48, uint ka_3, uint kb_3, uint c_23, float3 thread* q_lin_2, float3 thread* q_ang_2, KernelContext_0 thread* kernelContext_157)
{
    st_force_1(b_48, ka_3, kb_3, 0U, c_23, q_lin_2, q_ang_2, kernelContext_157);
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
    float _S1149 = q_lin_3.z;
    float axial_0 = (metal::fast::divide((_S1149), (area_0)));
    float _S1150 = (metal::fast::divide((abs(q_ang_3.x)), (b_49->geom1_0.x)));
    float _S1151 = (metal::fast::divide((abs(q_ang_3.y)), (b_49->geom1_0.y)));
    float bending_0 = _S1150 + _S1151;
    float _S1152 = q_lin_3.x;
    float _S1153 = q_lin_3.y;
    float _S1154 = (metal::fast::sqrt((_S1152 * _S1152 + _S1153 * _S1153)));
    float _S1155 = (metal::fast::divide((_S1154), (area_0)));
    float _S1156 = (metal::fast::divide((abs(q_ang_3.z)), (b_49->geom0_0.w)));
    float shear_1 = _S1155 + _S1156;
    thread Measures_0 m_3;
    (&m_3)->tension_0 = axial_0 + bending_0;
    (&m_3)->shear_0 = shear_1;
    float _S1157 = - axial_0;
    (&m_3)->normal_compression_0 = max(_S1157, 0.0f);
    (&m_3)->compression_0 = _S1157 + bending_0;
    (&m_3)->compressive_force_0 = max(- _S1149, 0.0f);
    return m_3;
}

JointMaterial_0 lp_material_0(uint m_4, KernelContext_0 thread* kernelContext_158)
{
    uint o_8 = kernelContext_158->params_0->ext_mat_0 + 5U * m_4;
    thread JointMaterial_0 j_8;
    (&j_8)->strength_0 = float4(*(kernelContext_158->ext_0+o_8)) ;
    (&j_8)->energy_0 = float4(*(kernelContext_158->ext_0+(o_8 + 1U))) ;
    (&j_8)->dif_0 = float4(*(kernelContext_158->ext_0+(o_8 + 2U))) ;
    (&j_8)->misc_0 = float4(*(kernelContext_158->ext_0+(o_8 + 3U))) ;
    (&j_8)->kind_flags_0 = (as_type<uint4>((float4(*(kernelContext_158->ext_0+(o_8 + 4U))) )));
    return j_8;
}

float infinity_0()
{
    return float_bits_0(2139095040U);
}

float4 failure_indices_0(const JointMaterial_0 thread* mat_1, const JointBond_0 thread* b_50, const Measures_0 thread* m_5, float multiplier_0)
{
    float fc_2 = mat_1->strength_0.y * multiplier_0;
    float _S1158 = min(mat_1->strength_0.z * multiplier_0 + mat_1->strength_0.w * m_5->normal_compression_0, mat_1->energy_0.x * multiplier_0);
    thread float4 idx_0;
    float _S1159 = (metal::fast::divide((m_5->tension_0), (mat_1->strength_0.x * multiplier_0)));
    idx_0.x = max(_S1159, 0.0f);
    float _S1160;
    if(_S1158 > 0.0f)
    {
        float _S1161 = (metal::fast::divide((m_5->shear_0), (_S1158)));
        _S1160 = _S1161;
    }
    else
    {
        _S1160 = infinity_0();
    }
    idx_0.y = _S1160;
    float _S1162 = (metal::fast::divide((m_5->compression_0), (fc_2)));
    idx_0.z = max(_S1162, 0.0f);
    float _S1163 = b_50->stiff1_0.y;
    if(_S1163 > 0.0f)
    {
        float _S1164 = (metal::fast::divide((m_5->compressive_force_0), (_S1163)));
        _S1160 = _S1164;
    }
    else
    {
        _S1160 = 0.0f;
    }
    idx_0.w = _S1160;
    return idx_0;
}

[[kernel]] void fr_lp_breaks(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], atomic<uint> device* atom_39 [[buffer(5)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], packed_float4 device* lp_39 [[buffer(24)]], packed_float4 device* ext_41 [[buffer(23)]], packed_float4 device* engine_contacts_39 [[buffer(25)]], CrushParams_0 constant* crush_39 [[buffer(28)]], packed_float4 device* crush_rw_39 [[buffer(27)]], packed_float4 device* crush_tab_39 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_159;
    (&kernelContext_159)->params_0 = params_39;
    (&kernelContext_159)->atom_0 = atom_39;
    (&kernelContext_159)->work_0 = work_39;
    (&kernelContext_159)->frames_0 = frames_39;
    (&kernelContext_159)->bodies_0 = bodies_39;
    (&kernelContext_159)->chunks_0 = chunks_39;
    (&kernelContext_159)->sort_0 = sort_39;
    (&kernelContext_159)->manifolds_0 = manifolds_39;
    (&kernelContext_159)->loads_0 = loads_39;
    (&kernelContext_159)->bonds_0 = bonds_39;
    (&kernelContext_159)->args_0 = args_39;
    (&kernelContext_159)->info_0 = info_39;
    (&kernelContext_159)->scan_0 = scan_39;
    (&kernelContext_159)->parts_0 = parts_39;
    (&kernelContext_159)->colliders_0 = colliders_39;
    (&kernelContext_159)->collider_info_0 = collider_info_39;
    (&kernelContext_159)->view_chunks_0 = view_chunks_39;
    (&kernelContext_159)->view_0 = view_39;
    (&kernelContext_159)->objects_0 = objects_39;
    (&kernelContext_159)->shot_bodies_0 = shot_bodies_39;
    (&kernelContext_159)->lines_0 = lines_39;
    (&kernelContext_159)->geometry_0 = geometry_39;
    (&kernelContext_159)->shots_0 = shots_39;
    (&kernelContext_159)->lp_0 = lp_39;
    (&kernelContext_159)->ext_0 = ext_41;
    (&kernelContext_159)->engine_contacts_0 = engine_contacts_39;
    (&kernelContext_159)->crush_0 = crush_39;
    (&kernelContext_159)->crush_rw_0 = crush_rw_39;
    (&kernelContext_159)->crush_tab_0 = crush_tab_39;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_159)->g_scan_0 = &g_scan_39;
    threadgroup array<float, int(64)> g_il_best_39;
    (&kernelContext_159)->g_il_best_0 = &g_il_best_39;
    uint i_19 = id_36.x;
    bool _S1165;
    if(i_19 >= (params_39->bond_count_0))
    {
        _S1165 = true;
    }
    else
    {
        uint _S1166 = w_alive_0(&kernelContext_159);
        _S1165 = (*((&kernelContext_159)->work_0+(_S1166 + i_19))) == 0U;
    }
    if(_S1165)
    {
        return;
    }
    uint4 _S1167 = uint4(*((&kernelContext_159)->bonds_0+i_19)) ;
    LpBond_0 _S1168 = lp_bond_0(i_19, &kernelContext_159);
    uint _S1169 = _S1167.x;
    uint _S1170 = _S1167.y;
    thread LpBond_0 _S1171 = _S1168;
    thread float3 q_lin_4;
    thread float3 q_ang_4;
    lp_force_0(&_S1171, _S1169, _S1170, 0U, &q_lin_4, &q_ang_4, &kernelContext_159);
    thread JointBond_0 _S1172 = _S1168.law_0;
    Measures_0 _S1173 = stress_measures_0(&_S1172, q_lin_4, q_ang_4);
    JointMaterial_0 _S1174 = lp_material_0(_S1168.law_0.ids_3.x, &kernelContext_159);
    float _S1175 = _S1168.law_0.geom1_0.w;
    thread JointMaterial_0 _S1176 = _S1174;
    thread JointBond_0 _S1177 = _S1168.law_0;
    thread Measures_0 _S1178 = _S1173;
    float4 _S1179 = failure_indices_0(&_S1176, &_S1177, &_S1178, _S1175);
    float _S1180 = max(max(_S1179.x, _S1179.y), max(_S1179.z, _S1179.w));
    uint _S1181 = a_counters_1(&kernelContext_159);
    uint _S1182 = atomic_fetch_max_explicit((&kernelContext_159)->atom_0+(_S1181 + 15U), (as_type<uint>((max(_S1180, 0.0f)))), memory_order_relaxed);
    if(_S1180 >= 1.0f)
    {
        uint _S1183 = w_alive_0(&kernelContext_159);
        *((&kernelContext_159)->work_0+(_S1183 + i_19)) = 0U;
        uint _S1184 = a_dirty_0(&kernelContext_159);
        uint _S1185 = body_of_0(_S1169, &kernelContext_159);
        uint _S1186 = atomic_fetch_or_explicit((&kernelContext_159)->atom_0+(_S1184 + _S1185), 1U, memory_order_relaxed);
        uint _S1187 = atomic_fetch_add_explicit((&kernelContext_159)->atom_0+_S1181, 1U, memory_order_relaxed);
        uint _S1188 = atomic_fetch_add_explicit((&kernelContext_159)->atom_0+(_S1181 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_19, &kernelContext_159);
    }
    return;
}

uint lp_view_0(uint k_44, KernelContext_0 thread* kernelContext_160)
{
    return 22U * kernelContext_160->params_0->chunk_count_0 + 7U * kernelContext_160->params_0->body_count_0 + k_44;
}

[[kernel]] void fr_lp_view(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], atomic<uint> device* atom_40 [[buffer(5)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], packed_float4 device* lp_40 [[buffer(24)]], packed_float4 device* ext_42 [[buffer(23)]], packed_float4 device* engine_contacts_40 [[buffer(25)]], CrushParams_0 constant* crush_40 [[buffer(28)]], packed_float4 device* crush_rw_40 [[buffer(27)]], packed_float4 device* crush_tab_40 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_161;
    (&kernelContext_161)->params_0 = params_40;
    (&kernelContext_161)->atom_0 = atom_40;
    (&kernelContext_161)->work_0 = work_40;
    (&kernelContext_161)->frames_0 = frames_40;
    (&kernelContext_161)->bodies_0 = bodies_40;
    (&kernelContext_161)->chunks_0 = chunks_40;
    (&kernelContext_161)->sort_0 = sort_40;
    (&kernelContext_161)->manifolds_0 = manifolds_40;
    (&kernelContext_161)->loads_0 = loads_40;
    (&kernelContext_161)->bonds_0 = bonds_40;
    (&kernelContext_161)->args_0 = args_40;
    (&kernelContext_161)->info_0 = info_40;
    (&kernelContext_161)->scan_0 = scan_40;
    (&kernelContext_161)->parts_0 = parts_40;
    (&kernelContext_161)->colliders_0 = colliders_40;
    (&kernelContext_161)->collider_info_0 = collider_info_40;
    (&kernelContext_161)->view_chunks_0 = view_chunks_40;
    (&kernelContext_161)->view_0 = view_40;
    (&kernelContext_161)->objects_0 = objects_40;
    (&kernelContext_161)->shot_bodies_0 = shot_bodies_40;
    (&kernelContext_161)->lines_0 = lines_40;
    (&kernelContext_161)->geometry_0 = geometry_40;
    (&kernelContext_161)->shots_0 = shots_40;
    (&kernelContext_161)->lp_0 = lp_40;
    (&kernelContext_161)->ext_0 = ext_42;
    (&kernelContext_161)->engine_contacts_0 = engine_contacts_40;
    (&kernelContext_161)->crush_0 = crush_40;
    (&kernelContext_161)->crush_rw_0 = crush_rw_40;
    (&kernelContext_161)->crush_tab_0 = crush_tab_40;
    threadgroup array<uint, int(1024)> g_scan_40;
    (&kernelContext_161)->g_scan_0 = &g_scan_40;
    threadgroup array<float, int(64)> g_il_best_40;
    (&kernelContext_161)->g_il_best_0 = &g_il_best_40;
    uint k_45 = id_37.x;
    if(k_45 >= (params_40->chunk_count_0))
    {
        return;
    }
    float4 _S1189 = float4(*((&kernelContext_161)->ext_0+((&kernelContext_161)->params_0->ext_chunks_0 + 4U * k_45))) ;
    uint _S1190 = (as_type<uint>((_S1189.x)));
    uint _S1191 = (as_type<uint>((_S1189.y)));
    float4 worst_0 = float4(0.0f) ;
    uint e_6 = 0U;
    for(;;)
    {
        if(e_6 < _S1191)
        {
        }
        else
        {
            break;
        }
        float4 _S1192 = lp_side_0(5U * (_S1190 + e_6), &kernelContext_161);
        uint i_20 = (as_type<uint>((_S1192.w)));
        uint _S1193 = w_alive_0(&kernelContext_161);
        if((*((&kernelContext_161)->work_0+(_S1193 + i_20))) == 0U)
        {
            e_6 = e_6 + 1U;
            continue;
        }
        uint4 _S1194 = uint4(*((&kernelContext_161)->bonds_0+i_20)) ;
        LpBond_0 _S1195 = lp_bond_0(i_20, &kernelContext_161);
        uint _S1196 = _S1194.x;
        uint _S1197 = _S1194.y;
        thread LpBond_0 _S1198 = _S1195;
        thread float3 q_lin_5;
        thread float3 q_ang_5;
        lp_force_0(&_S1198, _S1196, _S1197, 0U, &q_lin_5, &q_ang_5, &kernelContext_161);
        thread JointBond_0 _S1199 = _S1195.law_0;
        Measures_0 _S1200 = stress_measures_0(&_S1199, q_lin_5, q_ang_5);
        JointMaterial_0 _S1201 = lp_material_0(_S1195.law_0.ids_3.x, &kernelContext_161);
        float _S1202 = _S1195.law_0.geom1_0.w;
        thread JointMaterial_0 _S1203 = _S1201;
        thread JointBond_0 _S1204 = _S1195.law_0;
        thread Measures_0 _S1205 = _S1200;
        float4 _S1206 = failure_indices_0(&_S1203, &_S1204, &_S1205, _S1202);
        worst_0 = max(worst_0, _S1206);
        e_6 = e_6 + 1U;
    }
    uint _S1207 = lp_view_0(k_45, &kernelContext_161);
    *((&kernelContext_161)->lp_0+_S1207) = packed_float4(worst_0) ;
    return;
}

[[kernel]] void fr_il_first_reset(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], atomic<uint> device* atom_41 [[buffer(5)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], packed_float4 device* lp_41 [[buffer(24)]], packed_float4 device* ext_43 [[buffer(23)]], packed_float4 device* engine_contacts_41 [[buffer(25)]], CrushParams_0 constant* crush_41 [[buffer(28)]], packed_float4 device* crush_rw_41 [[buffer(27)]], packed_float4 device* crush_tab_41 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_162;
    (&kernelContext_162)->params_0 = params_41;
    (&kernelContext_162)->atom_0 = atom_41;
    (&kernelContext_162)->work_0 = work_41;
    (&kernelContext_162)->frames_0 = frames_41;
    (&kernelContext_162)->bodies_0 = bodies_41;
    (&kernelContext_162)->chunks_0 = chunks_41;
    (&kernelContext_162)->sort_0 = sort_41;
    (&kernelContext_162)->manifolds_0 = manifolds_41;
    (&kernelContext_162)->loads_0 = loads_41;
    (&kernelContext_162)->bonds_0 = bonds_41;
    (&kernelContext_162)->args_0 = args_41;
    (&kernelContext_162)->info_0 = info_41;
    (&kernelContext_162)->scan_0 = scan_41;
    (&kernelContext_162)->parts_0 = parts_41;
    (&kernelContext_162)->colliders_0 = colliders_41;
    (&kernelContext_162)->collider_info_0 = collider_info_41;
    (&kernelContext_162)->view_chunks_0 = view_chunks_41;
    (&kernelContext_162)->view_0 = view_41;
    (&kernelContext_162)->objects_0 = objects_41;
    (&kernelContext_162)->shot_bodies_0 = shot_bodies_41;
    (&kernelContext_162)->lines_0 = lines_41;
    (&kernelContext_162)->geometry_0 = geometry_41;
    (&kernelContext_162)->shots_0 = shots_41;
    (&kernelContext_162)->lp_0 = lp_41;
    (&kernelContext_162)->ext_0 = ext_43;
    (&kernelContext_162)->engine_contacts_0 = engine_contacts_41;
    (&kernelContext_162)->crush_0 = crush_41;
    (&kernelContext_162)->crush_rw_0 = crush_rw_41;
    (&kernelContext_162)->crush_tab_0 = crush_tab_41;
    threadgroup array<uint, int(1024)> g_scan_41;
    (&kernelContext_162)->g_scan_0 = &g_scan_41;
    threadgroup array<float, int(64)> g_il_best_41;
    (&kernelContext_162)->g_il_best_0 = &g_il_best_41;
    uint _S1208 = id_38.x;
    if(_S1208 == 0U)
    {
        uint _S1209 = a_il_cursor_1(&kernelContext_162);
        atomic_store_explicit((&kernelContext_162)->atom_0+_S1209, 0U, memory_order_relaxed);
    }
    if(_S1208 < ((&kernelContext_162)->params_0->shot_count_0))
    {
        uint _S1210 = a_first_1(_S1208, &kernelContext_162);
        atomic_store_explicit((&kernelContext_162)->atom_0+_S1210, 2139095040U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], atomic<uint> device* atom_42 [[buffer(5)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], packed_float4 device* lp_42 [[buffer(24)]], packed_float4 device* ext_44 [[buffer(23)]], packed_float4 device* engine_contacts_42 [[buffer(25)]], CrushParams_0 constant* crush_42 [[buffer(28)]], packed_float4 device* crush_rw_42 [[buffer(27)]], packed_float4 device* crush_tab_42 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_163;
    (&kernelContext_163)->params_0 = params_42;
    (&kernelContext_163)->atom_0 = atom_42;
    (&kernelContext_163)->work_0 = work_42;
    (&kernelContext_163)->frames_0 = frames_42;
    (&kernelContext_163)->bodies_0 = bodies_42;
    (&kernelContext_163)->chunks_0 = chunks_42;
    (&kernelContext_163)->sort_0 = sort_42;
    (&kernelContext_163)->manifolds_0 = manifolds_42;
    (&kernelContext_163)->loads_0 = loads_42;
    (&kernelContext_163)->bonds_0 = bonds_42;
    (&kernelContext_163)->args_0 = args_42;
    (&kernelContext_163)->info_0 = info_42;
    (&kernelContext_163)->scan_0 = scan_42;
    (&kernelContext_163)->parts_0 = parts_42;
    (&kernelContext_163)->colliders_0 = colliders_42;
    (&kernelContext_163)->collider_info_0 = collider_info_42;
    (&kernelContext_163)->view_chunks_0 = view_chunks_42;
    (&kernelContext_163)->view_0 = view_42;
    (&kernelContext_163)->objects_0 = objects_42;
    (&kernelContext_163)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_163)->lines_0 = lines_42;
    (&kernelContext_163)->geometry_0 = geometry_42;
    (&kernelContext_163)->shots_0 = shots_42;
    (&kernelContext_163)->lp_0 = lp_42;
    (&kernelContext_163)->ext_0 = ext_44;
    (&kernelContext_163)->engine_contacts_0 = engine_contacts_42;
    (&kernelContext_163)->crush_0 = crush_42;
    (&kernelContext_163)->crush_rw_0 = crush_rw_42;
    (&kernelContext_163)->crush_tab_0 = crush_tab_42;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_163)->g_scan_0 = &g_scan_42;
    threadgroup array<float, int(64)> g_il_best_42;
    (&kernelContext_163)->g_il_best_0 = &g_il_best_42;
    uint b_51 = id_39.x;
    bool _S1211;
    if(b_51 >= (params_42->body_count_0))
    {
        _S1211 = true;
    }
    else
    {
        _S1211 = ((&kernelContext_163)->params_0->shot_count_0) == 0U;
    }
    if(_S1211)
    {
        return;
    }
    thread uint shot_9;
    bool _S1212 = struck_0(b_51, &shot_9, &kernelContext_163);
    if(!_S1212)
    {
        return;
    }
    uint _S1213 = a_first_1(shot_9, &kernelContext_163);
    atomic<uint> device* _S1214 = (&kernelContext_163)->atom_0+_S1213;
    uint _S1215 = a_strike_0(&kernelContext_163);
    uint _S1216 = atomic_load_explicit((&kernelContext_163)->atom_0+(_S1215 + 9U * b_51 + 6U), memory_order_relaxed);
    uint _S1217 = atomic_fetch_min_explicit(_S1214, _S1216, memory_order_relaxed);
    return;
}

bool cs_struck_1(uint b_52, KernelContext_0 thread* kernelContext_164)
{
    bool _S1218;
    if(b_52 < (kernelContext_164->params_0->body_count_0))
    {
        uint _S1219 = a_cs_0(b_52, kernelContext_164);
        uint _S1220 = atomic_load_explicit(kernelContext_164->atom_0+(_S1219 + 1U), memory_order_relaxed);
        _S1218 = _S1220 != 0U;
    }
    else
    {
        _S1218 = false;
    }
    return _S1218;
}

[[kernel]] void fr_il_begin(uint3 id_40 [[thread_position_in_grid]], FractureParams_0 constant* params_43 [[buffer(0)]], atomic<uint> device* atom_43 [[buffer(5)]], uint device* work_43 [[buffer(4)]], packed_float4 device* frames_43 [[buffer(6)]], packed_float4 device* bodies_43 [[buffer(7)]], packed_float4 device* chunks_43 [[buffer(1)]], uint device* sort_43 [[buffer(11)]], Record_natural_0 device* manifolds_43 [[buffer(10)]], packed_float4 device* loads_43 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_43 [[buffer(13)]], uint device* info_43 [[buffer(15)]], ScanParams_0 constant* scan_43 [[buffer(14)]], packed_float4 device* parts_43 [[buffer(3)]], packed_float4 device* colliders_43 [[buffer(8)]], packed_uint4 device* collider_info_43 [[buffer(9)]], packed_float4 device* view_chunks_43 [[buffer(17)]], ViewParams_0 constant* view_43 [[buffer(16)]], packed_float4 device* objects_43 [[buffer(18)]], uint device* shot_bodies_43 [[buffer(20)]], float device* lines_43 [[buffer(19)]], packed_float4 device* geometry_43 [[buffer(21)]], packed_float4 device* shots_43 [[buffer(22)]], packed_float4 device* lp_43 [[buffer(24)]], packed_float4 device* ext_45 [[buffer(23)]], packed_float4 device* engine_contacts_43 [[buffer(25)]], CrushParams_0 constant* crush_43 [[buffer(28)]], packed_float4 device* crush_rw_43 [[buffer(27)]], packed_float4 device* crush_tab_43 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_165;
    (&kernelContext_165)->params_0 = params_43;
    (&kernelContext_165)->atom_0 = atom_43;
    (&kernelContext_165)->work_0 = work_43;
    (&kernelContext_165)->frames_0 = frames_43;
    (&kernelContext_165)->bodies_0 = bodies_43;
    (&kernelContext_165)->chunks_0 = chunks_43;
    (&kernelContext_165)->sort_0 = sort_43;
    (&kernelContext_165)->manifolds_0 = manifolds_43;
    (&kernelContext_165)->loads_0 = loads_43;
    (&kernelContext_165)->bonds_0 = bonds_43;
    (&kernelContext_165)->args_0 = args_43;
    (&kernelContext_165)->info_0 = info_43;
    (&kernelContext_165)->scan_0 = scan_43;
    (&kernelContext_165)->parts_0 = parts_43;
    (&kernelContext_165)->colliders_0 = colliders_43;
    (&kernelContext_165)->collider_info_0 = collider_info_43;
    (&kernelContext_165)->view_chunks_0 = view_chunks_43;
    (&kernelContext_165)->view_0 = view_43;
    (&kernelContext_165)->objects_0 = objects_43;
    (&kernelContext_165)->shot_bodies_0 = shot_bodies_43;
    (&kernelContext_165)->lines_0 = lines_43;
    (&kernelContext_165)->geometry_0 = geometry_43;
    (&kernelContext_165)->shots_0 = shots_43;
    (&kernelContext_165)->lp_0 = lp_43;
    (&kernelContext_165)->ext_0 = ext_45;
    (&kernelContext_165)->engine_contacts_0 = engine_contacts_43;
    (&kernelContext_165)->crush_0 = crush_43;
    (&kernelContext_165)->crush_rw_0 = crush_rw_43;
    (&kernelContext_165)->crush_tab_0 = crush_tab_43;
    threadgroup array<uint, int(1024)> g_scan_43;
    (&kernelContext_165)->g_scan_0 = &g_scan_43;
    threadgroup array<float, int(64)> g_il_best_43;
    (&kernelContext_165)->g_il_best_0 = &g_il_best_43;
    if((params_43->shot_count_0) > 0U)
    {
        il_begin_1(id_40.x, &kernelContext_165);
    }
    return;
}

[[kernel]] void fr_il_tube_count(uint3 id_41 [[thread_position_in_grid]], FractureParams_0 constant* params_44 [[buffer(0)]], atomic<uint> device* atom_44 [[buffer(5)]], uint device* work_44 [[buffer(4)]], packed_float4 device* frames_44 [[buffer(6)]], packed_float4 device* bodies_44 [[buffer(7)]], packed_float4 device* chunks_44 [[buffer(1)]], uint device* sort_44 [[buffer(11)]], Record_natural_0 device* manifolds_44 [[buffer(10)]], packed_float4 device* loads_44 [[buffer(12)]], packed_uint4 device* bonds_44 [[buffer(2)]], uint device* args_44 [[buffer(13)]], uint device* info_44 [[buffer(15)]], ScanParams_0 constant* scan_44 [[buffer(14)]], packed_float4 device* parts_44 [[buffer(3)]], packed_float4 device* colliders_44 [[buffer(8)]], packed_uint4 device* collider_info_44 [[buffer(9)]], packed_float4 device* view_chunks_44 [[buffer(17)]], ViewParams_0 constant* view_44 [[buffer(16)]], packed_float4 device* objects_44 [[buffer(18)]], uint device* shot_bodies_44 [[buffer(20)]], float device* lines_44 [[buffer(19)]], packed_float4 device* geometry_44 [[buffer(21)]], packed_float4 device* shots_44 [[buffer(22)]], packed_float4 device* lp_44 [[buffer(24)]], packed_float4 device* ext_46 [[buffer(23)]], packed_float4 device* engine_contacts_44 [[buffer(25)]], CrushParams_0 constant* crush_44 [[buffer(28)]], packed_float4 device* crush_rw_44 [[buffer(27)]], packed_float4 device* crush_tab_44 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_166;
    (&kernelContext_166)->params_0 = params_44;
    (&kernelContext_166)->atom_0 = atom_44;
    (&kernelContext_166)->work_0 = work_44;
    (&kernelContext_166)->frames_0 = frames_44;
    (&kernelContext_166)->bodies_0 = bodies_44;
    (&kernelContext_166)->chunks_0 = chunks_44;
    (&kernelContext_166)->sort_0 = sort_44;
    (&kernelContext_166)->manifolds_0 = manifolds_44;
    (&kernelContext_166)->loads_0 = loads_44;
    (&kernelContext_166)->bonds_0 = bonds_44;
    (&kernelContext_166)->args_0 = args_44;
    (&kernelContext_166)->info_0 = info_44;
    (&kernelContext_166)->scan_0 = scan_44;
    (&kernelContext_166)->parts_0 = parts_44;
    (&kernelContext_166)->colliders_0 = colliders_44;
    (&kernelContext_166)->collider_info_0 = collider_info_44;
    (&kernelContext_166)->view_chunks_0 = view_chunks_44;
    (&kernelContext_166)->view_0 = view_44;
    (&kernelContext_166)->objects_0 = objects_44;
    (&kernelContext_166)->shot_bodies_0 = shot_bodies_44;
    (&kernelContext_166)->lines_0 = lines_44;
    (&kernelContext_166)->geometry_0 = geometry_44;
    (&kernelContext_166)->shots_0 = shots_44;
    (&kernelContext_166)->lp_0 = lp_44;
    (&kernelContext_166)->ext_0 = ext_46;
    (&kernelContext_166)->engine_contacts_0 = engine_contacts_44;
    (&kernelContext_166)->crush_0 = crush_44;
    (&kernelContext_166)->crush_rw_0 = crush_rw_44;
    (&kernelContext_166)->crush_tab_0 = crush_tab_44;
    threadgroup array<uint, int(1024)> g_scan_44;
    (&kernelContext_166)->g_scan_0 = &g_scan_44;
    threadgroup array<float, int(64)> g_il_best_44;
    (&kernelContext_166)->g_il_best_0 = &g_il_best_44;
    if((params_44->shot_count_0) > 0U)
    {
        il_tube_count_1(id_41.x, &kernelContext_166);
    }
    return;
}

[[kernel]] void fr_il_tube_base(uint3 id_42 [[thread_position_in_grid]], FractureParams_0 constant* params_45 [[buffer(0)]], atomic<uint> device* atom_45 [[buffer(5)]], uint device* work_45 [[buffer(4)]], packed_float4 device* frames_45 [[buffer(6)]], packed_float4 device* bodies_45 [[buffer(7)]], packed_float4 device* chunks_45 [[buffer(1)]], uint device* sort_45 [[buffer(11)]], Record_natural_0 device* manifolds_45 [[buffer(10)]], packed_float4 device* loads_45 [[buffer(12)]], packed_uint4 device* bonds_45 [[buffer(2)]], uint device* args_45 [[buffer(13)]], uint device* info_45 [[buffer(15)]], ScanParams_0 constant* scan_45 [[buffer(14)]], packed_float4 device* parts_45 [[buffer(3)]], packed_float4 device* colliders_45 [[buffer(8)]], packed_uint4 device* collider_info_45 [[buffer(9)]], packed_float4 device* view_chunks_45 [[buffer(17)]], ViewParams_0 constant* view_45 [[buffer(16)]], packed_float4 device* objects_45 [[buffer(18)]], uint device* shot_bodies_45 [[buffer(20)]], float device* lines_45 [[buffer(19)]], packed_float4 device* geometry_45 [[buffer(21)]], packed_float4 device* shots_45 [[buffer(22)]], packed_float4 device* lp_45 [[buffer(24)]], packed_float4 device* ext_47 [[buffer(23)]], packed_float4 device* engine_contacts_45 [[buffer(25)]], CrushParams_0 constant* crush_45 [[buffer(28)]], packed_float4 device* crush_rw_45 [[buffer(27)]], packed_float4 device* crush_tab_45 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_167;
    (&kernelContext_167)->params_0 = params_45;
    (&kernelContext_167)->atom_0 = atom_45;
    (&kernelContext_167)->work_0 = work_45;
    (&kernelContext_167)->frames_0 = frames_45;
    (&kernelContext_167)->bodies_0 = bodies_45;
    (&kernelContext_167)->chunks_0 = chunks_45;
    (&kernelContext_167)->sort_0 = sort_45;
    (&kernelContext_167)->manifolds_0 = manifolds_45;
    (&kernelContext_167)->loads_0 = loads_45;
    (&kernelContext_167)->bonds_0 = bonds_45;
    (&kernelContext_167)->args_0 = args_45;
    (&kernelContext_167)->info_0 = info_45;
    (&kernelContext_167)->scan_0 = scan_45;
    (&kernelContext_167)->parts_0 = parts_45;
    (&kernelContext_167)->colliders_0 = colliders_45;
    (&kernelContext_167)->collider_info_0 = collider_info_45;
    (&kernelContext_167)->view_chunks_0 = view_chunks_45;
    (&kernelContext_167)->view_0 = view_45;
    (&kernelContext_167)->objects_0 = objects_45;
    (&kernelContext_167)->shot_bodies_0 = shot_bodies_45;
    (&kernelContext_167)->lines_0 = lines_45;
    (&kernelContext_167)->geometry_0 = geometry_45;
    (&kernelContext_167)->shots_0 = shots_45;
    (&kernelContext_167)->lp_0 = lp_45;
    (&kernelContext_167)->ext_0 = ext_47;
    (&kernelContext_167)->engine_contacts_0 = engine_contacts_45;
    (&kernelContext_167)->crush_0 = crush_45;
    (&kernelContext_167)->crush_rw_0 = crush_rw_45;
    (&kernelContext_167)->crush_tab_0 = crush_tab_45;
    threadgroup array<uint, int(1024)> g_scan_45;
    (&kernelContext_167)->g_scan_0 = &g_scan_45;
    threadgroup array<float, int(64)> g_il_best_45;
    (&kernelContext_167)->g_il_best_0 = &g_il_best_45;
    if((params_45->shot_count_0) > 0U)
    {
        il_tube_base_1(id_42.x, &kernelContext_167);
    }
    return;
}

[[kernel]] void fr_il_tube_collect(uint3 id_43 [[thread_position_in_grid]], FractureParams_0 constant* params_46 [[buffer(0)]], atomic<uint> device* atom_46 [[buffer(5)]], uint device* work_46 [[buffer(4)]], packed_float4 device* frames_46 [[buffer(6)]], packed_float4 device* bodies_46 [[buffer(7)]], packed_float4 device* chunks_46 [[buffer(1)]], uint device* sort_46 [[buffer(11)]], Record_natural_0 device* manifolds_46 [[buffer(10)]], packed_float4 device* loads_46 [[buffer(12)]], packed_uint4 device* bonds_46 [[buffer(2)]], uint device* args_46 [[buffer(13)]], uint device* info_46 [[buffer(15)]], ScanParams_0 constant* scan_46 [[buffer(14)]], packed_float4 device* parts_46 [[buffer(3)]], packed_float4 device* colliders_46 [[buffer(8)]], packed_uint4 device* collider_info_46 [[buffer(9)]], packed_float4 device* view_chunks_46 [[buffer(17)]], ViewParams_0 constant* view_46 [[buffer(16)]], packed_float4 device* objects_46 [[buffer(18)]], uint device* shot_bodies_46 [[buffer(20)]], float device* lines_46 [[buffer(19)]], packed_float4 device* geometry_46 [[buffer(21)]], packed_float4 device* shots_46 [[buffer(22)]], packed_float4 device* lp_46 [[buffer(24)]], packed_float4 device* ext_48 [[buffer(23)]], packed_float4 device* engine_contacts_46 [[buffer(25)]], CrushParams_0 constant* crush_46 [[buffer(28)]], packed_float4 device* crush_rw_46 [[buffer(27)]], packed_float4 device* crush_tab_46 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_168;
    (&kernelContext_168)->params_0 = params_46;
    (&kernelContext_168)->atom_0 = atom_46;
    (&kernelContext_168)->work_0 = work_46;
    (&kernelContext_168)->frames_0 = frames_46;
    (&kernelContext_168)->bodies_0 = bodies_46;
    (&kernelContext_168)->chunks_0 = chunks_46;
    (&kernelContext_168)->sort_0 = sort_46;
    (&kernelContext_168)->manifolds_0 = manifolds_46;
    (&kernelContext_168)->loads_0 = loads_46;
    (&kernelContext_168)->bonds_0 = bonds_46;
    (&kernelContext_168)->args_0 = args_46;
    (&kernelContext_168)->info_0 = info_46;
    (&kernelContext_168)->scan_0 = scan_46;
    (&kernelContext_168)->parts_0 = parts_46;
    (&kernelContext_168)->colliders_0 = colliders_46;
    (&kernelContext_168)->collider_info_0 = collider_info_46;
    (&kernelContext_168)->view_chunks_0 = view_chunks_46;
    (&kernelContext_168)->view_0 = view_46;
    (&kernelContext_168)->objects_0 = objects_46;
    (&kernelContext_168)->shot_bodies_0 = shot_bodies_46;
    (&kernelContext_168)->lines_0 = lines_46;
    (&kernelContext_168)->geometry_0 = geometry_46;
    (&kernelContext_168)->shots_0 = shots_46;
    (&kernelContext_168)->lp_0 = lp_46;
    (&kernelContext_168)->ext_0 = ext_48;
    (&kernelContext_168)->engine_contacts_0 = engine_contacts_46;
    (&kernelContext_168)->crush_0 = crush_46;
    (&kernelContext_168)->crush_rw_0 = crush_rw_46;
    (&kernelContext_168)->crush_tab_0 = crush_tab_46;
    threadgroup array<uint, int(1024)> g_scan_46;
    (&kernelContext_168)->g_scan_0 = &g_scan_46;
    threadgroup array<float, int(64)> g_il_best_46;
    (&kernelContext_168)->g_il_best_0 = &g_il_best_46;
    if((params_46->shot_count_0) > 0U)
    {
        il_tube_collect_1(id_43.x, &kernelContext_168);
    }
    return;
}

[[kernel]] void fr_il_run(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_47 [[buffer(0)]], atomic<uint> device* atom_47 [[buffer(5)]], uint device* work_47 [[buffer(4)]], packed_float4 device* frames_47 [[buffer(6)]], packed_float4 device* bodies_47 [[buffer(7)]], packed_float4 device* chunks_47 [[buffer(1)]], uint device* sort_47 [[buffer(11)]], Record_natural_0 device* manifolds_47 [[buffer(10)]], packed_float4 device* loads_47 [[buffer(12)]], packed_uint4 device* bonds_47 [[buffer(2)]], uint device* args_47 [[buffer(13)]], uint device* info_47 [[buffer(15)]], ScanParams_0 constant* scan_47 [[buffer(14)]], packed_float4 device* parts_47 [[buffer(3)]], packed_float4 device* colliders_47 [[buffer(8)]], packed_uint4 device* collider_info_47 [[buffer(9)]], packed_float4 device* view_chunks_47 [[buffer(17)]], ViewParams_0 constant* view_47 [[buffer(16)]], packed_float4 device* objects_47 [[buffer(18)]], uint device* shot_bodies_47 [[buffer(20)]], float device* lines_47 [[buffer(19)]], packed_float4 device* geometry_47 [[buffer(21)]], packed_float4 device* shots_47 [[buffer(22)]], packed_float4 device* lp_47 [[buffer(24)]], packed_float4 device* ext_49 [[buffer(23)]], packed_float4 device* engine_contacts_47 [[buffer(25)]], CrushParams_0 constant* crush_47 [[buffer(28)]], packed_float4 device* crush_rw_47 [[buffer(27)]], packed_float4 device* crush_tab_47 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_169;
    (&kernelContext_169)->params_0 = params_47;
    (&kernelContext_169)->atom_0 = atom_47;
    (&kernelContext_169)->work_0 = work_47;
    (&kernelContext_169)->frames_0 = frames_47;
    (&kernelContext_169)->bodies_0 = bodies_47;
    (&kernelContext_169)->chunks_0 = chunks_47;
    (&kernelContext_169)->sort_0 = sort_47;
    (&kernelContext_169)->manifolds_0 = manifolds_47;
    (&kernelContext_169)->loads_0 = loads_47;
    (&kernelContext_169)->bonds_0 = bonds_47;
    (&kernelContext_169)->args_0 = args_47;
    (&kernelContext_169)->info_0 = info_47;
    (&kernelContext_169)->scan_0 = scan_47;
    (&kernelContext_169)->parts_0 = parts_47;
    (&kernelContext_169)->colliders_0 = colliders_47;
    (&kernelContext_169)->collider_info_0 = collider_info_47;
    (&kernelContext_169)->view_chunks_0 = view_chunks_47;
    (&kernelContext_169)->view_0 = view_47;
    (&kernelContext_169)->objects_0 = objects_47;
    (&kernelContext_169)->shot_bodies_0 = shot_bodies_47;
    (&kernelContext_169)->lines_0 = lines_47;
    (&kernelContext_169)->geometry_0 = geometry_47;
    (&kernelContext_169)->shots_0 = shots_47;
    (&kernelContext_169)->lp_0 = lp_47;
    (&kernelContext_169)->ext_0 = ext_49;
    (&kernelContext_169)->engine_contacts_0 = engine_contacts_47;
    (&kernelContext_169)->crush_0 = crush_47;
    (&kernelContext_169)->crush_rw_0 = crush_rw_47;
    (&kernelContext_169)->crush_tab_0 = crush_tab_47;
    threadgroup array<uint, int(1024)> g_scan_47;
    (&kernelContext_169)->g_scan_0 = &g_scan_47;
    threadgroup array<float, int(64)> g_il_best_47;
    (&kernelContext_169)->g_il_best_0 = &g_il_best_47;
    uint b_53 = group_1.y * 65535U + group_1.x;
    bool _S1221;
    if((params_47->shot_count_0) > 0U)
    {
        _S1221 = b_53 < ((&kernelContext_169)->params_0->body_count_0);
    }
    else
    {
        _S1221 = false;
    }
    if(_S1221)
    {
        il_run_1(b_53, thread_4.x, &kernelContext_169);
    }
    return;
}

[[kernel]] void fr_il_strike(uint3 id_44 [[thread_position_in_grid]], FractureParams_0 constant* params_48 [[buffer(0)]], atomic<uint> device* atom_48 [[buffer(5)]], uint device* work_48 [[buffer(4)]], packed_float4 device* frames_48 [[buffer(6)]], packed_float4 device* bodies_48 [[buffer(7)]], packed_float4 device* chunks_48 [[buffer(1)]], uint device* sort_48 [[buffer(11)]], Record_natural_0 device* manifolds_48 [[buffer(10)]], packed_float4 device* loads_48 [[buffer(12)]], packed_uint4 device* bonds_48 [[buffer(2)]], uint device* args_48 [[buffer(13)]], uint device* info_48 [[buffer(15)]], ScanParams_0 constant* scan_48 [[buffer(14)]], packed_float4 device* parts_48 [[buffer(3)]], packed_float4 device* colliders_48 [[buffer(8)]], packed_uint4 device* collider_info_48 [[buffer(9)]], packed_float4 device* view_chunks_48 [[buffer(17)]], ViewParams_0 constant* view_48 [[buffer(16)]], packed_float4 device* objects_48 [[buffer(18)]], uint device* shot_bodies_48 [[buffer(20)]], float device* lines_48 [[buffer(19)]], packed_float4 device* geometry_48 [[buffer(21)]], packed_float4 device* shots_48 [[buffer(22)]], packed_float4 device* lp_48 [[buffer(24)]], packed_float4 device* ext_50 [[buffer(23)]], packed_float4 device* engine_contacts_48 [[buffer(25)]], CrushParams_0 constant* crush_48 [[buffer(28)]], packed_float4 device* crush_rw_48 [[buffer(27)]], packed_float4 device* crush_tab_48 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_170;
    (&kernelContext_170)->params_0 = params_48;
    (&kernelContext_170)->atom_0 = atom_48;
    (&kernelContext_170)->work_0 = work_48;
    (&kernelContext_170)->frames_0 = frames_48;
    (&kernelContext_170)->bodies_0 = bodies_48;
    (&kernelContext_170)->chunks_0 = chunks_48;
    (&kernelContext_170)->sort_0 = sort_48;
    (&kernelContext_170)->manifolds_0 = manifolds_48;
    (&kernelContext_170)->loads_0 = loads_48;
    (&kernelContext_170)->bonds_0 = bonds_48;
    (&kernelContext_170)->args_0 = args_48;
    (&kernelContext_170)->info_0 = info_48;
    (&kernelContext_170)->scan_0 = scan_48;
    (&kernelContext_170)->parts_0 = parts_48;
    (&kernelContext_170)->colliders_0 = colliders_48;
    (&kernelContext_170)->collider_info_0 = collider_info_48;
    (&kernelContext_170)->view_chunks_0 = view_chunks_48;
    (&kernelContext_170)->view_0 = view_48;
    (&kernelContext_170)->objects_0 = objects_48;
    (&kernelContext_170)->shot_bodies_0 = shot_bodies_48;
    (&kernelContext_170)->lines_0 = lines_48;
    (&kernelContext_170)->geometry_0 = geometry_48;
    (&kernelContext_170)->shots_0 = shots_48;
    (&kernelContext_170)->lp_0 = lp_48;
    (&kernelContext_170)->ext_0 = ext_50;
    (&kernelContext_170)->engine_contacts_0 = engine_contacts_48;
    (&kernelContext_170)->crush_0 = crush_48;
    (&kernelContext_170)->crush_rw_0 = crush_rw_48;
    (&kernelContext_170)->crush_tab_0 = crush_tab_48;
    threadgroup array<uint, int(1024)> g_scan_48;
    (&kernelContext_170)->g_scan_0 = &g_scan_48;
    threadgroup array<float, int(64)> g_il_best_48;
    (&kernelContext_170)->g_il_best_0 = &g_il_best_48;
    if((params_48->shot_count_0) > 0U)
    {
        il_strike_1(id_44.x, &kernelContext_170);
    }
    return;
}

[[kernel]] void fr_il_mark(uint3 id_45 [[thread_position_in_grid]], FractureParams_0 constant* params_49 [[buffer(0)]], atomic<uint> device* atom_49 [[buffer(5)]], uint device* work_49 [[buffer(4)]], packed_float4 device* frames_49 [[buffer(6)]], packed_float4 device* bodies_49 [[buffer(7)]], packed_float4 device* chunks_49 [[buffer(1)]], uint device* sort_49 [[buffer(11)]], Record_natural_0 device* manifolds_49 [[buffer(10)]], packed_float4 device* loads_49 [[buffer(12)]], packed_uint4 device* bonds_49 [[buffer(2)]], uint device* args_49 [[buffer(13)]], uint device* info_49 [[buffer(15)]], ScanParams_0 constant* scan_49 [[buffer(14)]], packed_float4 device* parts_49 [[buffer(3)]], packed_float4 device* colliders_49 [[buffer(8)]], packed_uint4 device* collider_info_49 [[buffer(9)]], packed_float4 device* view_chunks_49 [[buffer(17)]], ViewParams_0 constant* view_49 [[buffer(16)]], packed_float4 device* objects_49 [[buffer(18)]], uint device* shot_bodies_49 [[buffer(20)]], float device* lines_49 [[buffer(19)]], packed_float4 device* geometry_49 [[buffer(21)]], packed_float4 device* shots_49 [[buffer(22)]], packed_float4 device* lp_49 [[buffer(24)]], packed_float4 device* ext_51 [[buffer(23)]], packed_float4 device* engine_contacts_49 [[buffer(25)]], CrushParams_0 constant* crush_49 [[buffer(28)]], packed_float4 device* crush_rw_49 [[buffer(27)]], packed_float4 device* crush_tab_49 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_171;
    (&kernelContext_171)->params_0 = params_49;
    (&kernelContext_171)->atom_0 = atom_49;
    (&kernelContext_171)->work_0 = work_49;
    (&kernelContext_171)->frames_0 = frames_49;
    (&kernelContext_171)->bodies_0 = bodies_49;
    (&kernelContext_171)->chunks_0 = chunks_49;
    (&kernelContext_171)->sort_0 = sort_49;
    (&kernelContext_171)->manifolds_0 = manifolds_49;
    (&kernelContext_171)->loads_0 = loads_49;
    (&kernelContext_171)->bonds_0 = bonds_49;
    (&kernelContext_171)->args_0 = args_49;
    (&kernelContext_171)->info_0 = info_49;
    (&kernelContext_171)->scan_0 = scan_49;
    (&kernelContext_171)->parts_0 = parts_49;
    (&kernelContext_171)->colliders_0 = colliders_49;
    (&kernelContext_171)->collider_info_0 = collider_info_49;
    (&kernelContext_171)->view_chunks_0 = view_chunks_49;
    (&kernelContext_171)->view_0 = view_49;
    (&kernelContext_171)->objects_0 = objects_49;
    (&kernelContext_171)->shot_bodies_0 = shot_bodies_49;
    (&kernelContext_171)->lines_0 = lines_49;
    (&kernelContext_171)->geometry_0 = geometry_49;
    (&kernelContext_171)->shots_0 = shots_49;
    (&kernelContext_171)->lp_0 = lp_49;
    (&kernelContext_171)->ext_0 = ext_51;
    (&kernelContext_171)->engine_contacts_0 = engine_contacts_49;
    (&kernelContext_171)->crush_0 = crush_49;
    (&kernelContext_171)->crush_rw_0 = crush_rw_49;
    (&kernelContext_171)->crush_tab_0 = crush_tab_49;
    threadgroup array<uint, int(1024)> g_scan_49;
    (&kernelContext_171)->g_scan_0 = &g_scan_49;
    threadgroup array<float, int(64)> g_il_best_49;
    (&kernelContext_171)->g_il_best_0 = &g_il_best_49;
    if((params_49->shot_count_0) > 0U)
    {
        il_mark_1(id_45.x, &kernelContext_171);
    }
    return;
}

[[kernel]] void fr_il_carry(uint3 id_46 [[thread_position_in_grid]], FractureParams_0 constant* params_50 [[buffer(0)]], atomic<uint> device* atom_50 [[buffer(5)]], uint device* work_50 [[buffer(4)]], packed_float4 device* frames_50 [[buffer(6)]], packed_float4 device* bodies_50 [[buffer(7)]], packed_float4 device* chunks_50 [[buffer(1)]], uint device* sort_50 [[buffer(11)]], Record_natural_0 device* manifolds_50 [[buffer(10)]], packed_float4 device* loads_50 [[buffer(12)]], packed_uint4 device* bonds_50 [[buffer(2)]], uint device* args_50 [[buffer(13)]], uint device* info_50 [[buffer(15)]], ScanParams_0 constant* scan_50 [[buffer(14)]], packed_float4 device* parts_50 [[buffer(3)]], packed_float4 device* colliders_50 [[buffer(8)]], packed_uint4 device* collider_info_50 [[buffer(9)]], packed_float4 device* view_chunks_50 [[buffer(17)]], ViewParams_0 constant* view_50 [[buffer(16)]], packed_float4 device* objects_50 [[buffer(18)]], uint device* shot_bodies_50 [[buffer(20)]], float device* lines_50 [[buffer(19)]], packed_float4 device* geometry_50 [[buffer(21)]], packed_float4 device* shots_50 [[buffer(22)]], packed_float4 device* lp_50 [[buffer(24)]], packed_float4 device* ext_52 [[buffer(23)]], packed_float4 device* engine_contacts_50 [[buffer(25)]], CrushParams_0 constant* crush_50 [[buffer(28)]], packed_float4 device* crush_rw_50 [[buffer(27)]], packed_float4 device* crush_tab_50 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_172;
    (&kernelContext_172)->params_0 = params_50;
    (&kernelContext_172)->atom_0 = atom_50;
    (&kernelContext_172)->work_0 = work_50;
    (&kernelContext_172)->frames_0 = frames_50;
    (&kernelContext_172)->bodies_0 = bodies_50;
    (&kernelContext_172)->chunks_0 = chunks_50;
    (&kernelContext_172)->sort_0 = sort_50;
    (&kernelContext_172)->manifolds_0 = manifolds_50;
    (&kernelContext_172)->loads_0 = loads_50;
    (&kernelContext_172)->bonds_0 = bonds_50;
    (&kernelContext_172)->args_0 = args_50;
    (&kernelContext_172)->info_0 = info_50;
    (&kernelContext_172)->scan_0 = scan_50;
    (&kernelContext_172)->parts_0 = parts_50;
    (&kernelContext_172)->colliders_0 = colliders_50;
    (&kernelContext_172)->collider_info_0 = collider_info_50;
    (&kernelContext_172)->view_chunks_0 = view_chunks_50;
    (&kernelContext_172)->view_0 = view_50;
    (&kernelContext_172)->objects_0 = objects_50;
    (&kernelContext_172)->shot_bodies_0 = shot_bodies_50;
    (&kernelContext_172)->lines_0 = lines_50;
    (&kernelContext_172)->geometry_0 = geometry_50;
    (&kernelContext_172)->shots_0 = shots_50;
    (&kernelContext_172)->lp_0 = lp_50;
    (&kernelContext_172)->ext_0 = ext_52;
    (&kernelContext_172)->engine_contacts_0 = engine_contacts_50;
    (&kernelContext_172)->crush_0 = crush_50;
    (&kernelContext_172)->crush_rw_0 = crush_rw_50;
    (&kernelContext_172)->crush_tab_0 = crush_tab_50;
    threadgroup array<uint, int(1024)> g_scan_50;
    (&kernelContext_172)->g_scan_0 = &g_scan_50;
    threadgroup array<float, int(64)> g_il_best_50;
    (&kernelContext_172)->g_il_best_0 = &g_il_best_50;
    if((params_50->shot_count_0) > 0U)
    {
        il_carry_1(id_46.x, &kernelContext_172);
    }
    return;
}

[[kernel]] void fr_il_breaks(uint3 id_47 [[thread_position_in_grid]], FractureParams_0 constant* params_51 [[buffer(0)]], atomic<uint> device* atom_51 [[buffer(5)]], uint device* work_51 [[buffer(4)]], packed_float4 device* frames_51 [[buffer(6)]], packed_float4 device* bodies_51 [[buffer(7)]], packed_float4 device* chunks_51 [[buffer(1)]], uint device* sort_51 [[buffer(11)]], Record_natural_0 device* manifolds_51 [[buffer(10)]], packed_float4 device* loads_51 [[buffer(12)]], packed_uint4 device* bonds_51 [[buffer(2)]], uint device* args_51 [[buffer(13)]], uint device* info_51 [[buffer(15)]], ScanParams_0 constant* scan_51 [[buffer(14)]], packed_float4 device* parts_51 [[buffer(3)]], packed_float4 device* colliders_51 [[buffer(8)]], packed_uint4 device* collider_info_51 [[buffer(9)]], packed_float4 device* view_chunks_51 [[buffer(17)]], ViewParams_0 constant* view_51 [[buffer(16)]], packed_float4 device* objects_51 [[buffer(18)]], uint device* shot_bodies_51 [[buffer(20)]], float device* lines_51 [[buffer(19)]], packed_float4 device* geometry_51 [[buffer(21)]], packed_float4 device* shots_51 [[buffer(22)]], packed_float4 device* lp_51 [[buffer(24)]], packed_float4 device* ext_53 [[buffer(23)]], packed_float4 device* engine_contacts_51 [[buffer(25)]], CrushParams_0 constant* crush_51 [[buffer(28)]], packed_float4 device* crush_rw_51 [[buffer(27)]], packed_float4 device* crush_tab_51 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_173;
    (&kernelContext_173)->params_0 = params_51;
    (&kernelContext_173)->atom_0 = atom_51;
    (&kernelContext_173)->work_0 = work_51;
    (&kernelContext_173)->frames_0 = frames_51;
    (&kernelContext_173)->bodies_0 = bodies_51;
    (&kernelContext_173)->chunks_0 = chunks_51;
    (&kernelContext_173)->sort_0 = sort_51;
    (&kernelContext_173)->manifolds_0 = manifolds_51;
    (&kernelContext_173)->loads_0 = loads_51;
    (&kernelContext_173)->bonds_0 = bonds_51;
    (&kernelContext_173)->args_0 = args_51;
    (&kernelContext_173)->info_0 = info_51;
    (&kernelContext_173)->scan_0 = scan_51;
    (&kernelContext_173)->parts_0 = parts_51;
    (&kernelContext_173)->colliders_0 = colliders_51;
    (&kernelContext_173)->collider_info_0 = collider_info_51;
    (&kernelContext_173)->view_chunks_0 = view_chunks_51;
    (&kernelContext_173)->view_0 = view_51;
    (&kernelContext_173)->objects_0 = objects_51;
    (&kernelContext_173)->shot_bodies_0 = shot_bodies_51;
    (&kernelContext_173)->lines_0 = lines_51;
    (&kernelContext_173)->geometry_0 = geometry_51;
    (&kernelContext_173)->shots_0 = shots_51;
    (&kernelContext_173)->lp_0 = lp_51;
    (&kernelContext_173)->ext_0 = ext_53;
    (&kernelContext_173)->engine_contacts_0 = engine_contacts_51;
    (&kernelContext_173)->crush_0 = crush_51;
    (&kernelContext_173)->crush_rw_0 = crush_rw_51;
    (&kernelContext_173)->crush_tab_0 = crush_tab_51;
    threadgroup array<uint, int(1024)> g_scan_51;
    (&kernelContext_173)->g_scan_0 = &g_scan_51;
    threadgroup array<float, int(64)> g_il_best_51;
    (&kernelContext_173)->g_il_best_0 = &g_il_best_51;
    if((params_51->shot_count_0) > 0U)
    {
        il_breaks_1(id_47.x, &kernelContext_173);
    }
    return;
}

[[kernel]] void fr_csl_begin(uint3 id_48 [[thread_position_in_grid]], FractureParams_0 constant* params_52 [[buffer(0)]], atomic<uint> device* atom_52 [[buffer(5)]], uint device* work_52 [[buffer(4)]], packed_float4 device* frames_52 [[buffer(6)]], packed_float4 device* bodies_52 [[buffer(7)]], packed_float4 device* chunks_52 [[buffer(1)]], uint device* sort_52 [[buffer(11)]], Record_natural_0 device* manifolds_52 [[buffer(10)]], packed_float4 device* loads_52 [[buffer(12)]], packed_uint4 device* bonds_52 [[buffer(2)]], uint device* args_52 [[buffer(13)]], uint device* info_52 [[buffer(15)]], ScanParams_0 constant* scan_52 [[buffer(14)]], packed_float4 device* parts_52 [[buffer(3)]], packed_float4 device* colliders_52 [[buffer(8)]], packed_uint4 device* collider_info_52 [[buffer(9)]], packed_float4 device* view_chunks_52 [[buffer(17)]], ViewParams_0 constant* view_52 [[buffer(16)]], packed_float4 device* objects_52 [[buffer(18)]], uint device* shot_bodies_52 [[buffer(20)]], float device* lines_52 [[buffer(19)]], packed_float4 device* geometry_52 [[buffer(21)]], packed_float4 device* shots_52 [[buffer(22)]], packed_float4 device* lp_52 [[buffer(24)]], packed_float4 device* ext_54 [[buffer(23)]], packed_float4 device* engine_contacts_52 [[buffer(25)]], CrushParams_0 constant* crush_52 [[buffer(28)]], packed_float4 device* crush_rw_52 [[buffer(27)]], packed_float4 device* crush_tab_52 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_174;
    (&kernelContext_174)->params_0 = params_52;
    (&kernelContext_174)->atom_0 = atom_52;
    (&kernelContext_174)->work_0 = work_52;
    (&kernelContext_174)->frames_0 = frames_52;
    (&kernelContext_174)->bodies_0 = bodies_52;
    (&kernelContext_174)->chunks_0 = chunks_52;
    (&kernelContext_174)->sort_0 = sort_52;
    (&kernelContext_174)->manifolds_0 = manifolds_52;
    (&kernelContext_174)->loads_0 = loads_52;
    (&kernelContext_174)->bonds_0 = bonds_52;
    (&kernelContext_174)->args_0 = args_52;
    (&kernelContext_174)->info_0 = info_52;
    (&kernelContext_174)->scan_0 = scan_52;
    (&kernelContext_174)->parts_0 = parts_52;
    (&kernelContext_174)->colliders_0 = colliders_52;
    (&kernelContext_174)->collider_info_0 = collider_info_52;
    (&kernelContext_174)->view_chunks_0 = view_chunks_52;
    (&kernelContext_174)->view_0 = view_52;
    (&kernelContext_174)->objects_0 = objects_52;
    (&kernelContext_174)->shot_bodies_0 = shot_bodies_52;
    (&kernelContext_174)->lines_0 = lines_52;
    (&kernelContext_174)->geometry_0 = geometry_52;
    (&kernelContext_174)->shots_0 = shots_52;
    (&kernelContext_174)->lp_0 = lp_52;
    (&kernelContext_174)->ext_0 = ext_54;
    (&kernelContext_174)->engine_contacts_0 = engine_contacts_52;
    (&kernelContext_174)->crush_0 = crush_52;
    (&kernelContext_174)->crush_rw_0 = crush_rw_52;
    (&kernelContext_174)->crush_tab_0 = crush_tab_52;
    threadgroup array<uint, int(1024)> g_scan_52;
    (&kernelContext_174)->g_scan_0 = &g_scan_52;
    threadgroup array<float, int(64)> g_il_best_52;
    (&kernelContext_174)->g_il_best_0 = &g_il_best_52;
    il_begin_0(id_48.x, &kernelContext_174);
    return;
}

[[kernel]] void fr_csl_tube_count(uint3 id_49 [[thread_position_in_grid]], FractureParams_0 constant* params_53 [[buffer(0)]], atomic<uint> device* atom_53 [[buffer(5)]], uint device* work_53 [[buffer(4)]], packed_float4 device* frames_53 [[buffer(6)]], packed_float4 device* bodies_53 [[buffer(7)]], packed_float4 device* chunks_53 [[buffer(1)]], uint device* sort_53 [[buffer(11)]], Record_natural_0 device* manifolds_53 [[buffer(10)]], packed_float4 device* loads_53 [[buffer(12)]], packed_uint4 device* bonds_53 [[buffer(2)]], uint device* args_53 [[buffer(13)]], uint device* info_53 [[buffer(15)]], ScanParams_0 constant* scan_53 [[buffer(14)]], packed_float4 device* parts_53 [[buffer(3)]], packed_float4 device* colliders_53 [[buffer(8)]], packed_uint4 device* collider_info_53 [[buffer(9)]], packed_float4 device* view_chunks_53 [[buffer(17)]], ViewParams_0 constant* view_53 [[buffer(16)]], packed_float4 device* objects_53 [[buffer(18)]], uint device* shot_bodies_53 [[buffer(20)]], float device* lines_53 [[buffer(19)]], packed_float4 device* geometry_53 [[buffer(21)]], packed_float4 device* shots_53 [[buffer(22)]], packed_float4 device* lp_53 [[buffer(24)]], packed_float4 device* ext_55 [[buffer(23)]], packed_float4 device* engine_contacts_53 [[buffer(25)]], CrushParams_0 constant* crush_53 [[buffer(28)]], packed_float4 device* crush_rw_53 [[buffer(27)]], packed_float4 device* crush_tab_53 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_175;
    (&kernelContext_175)->params_0 = params_53;
    (&kernelContext_175)->atom_0 = atom_53;
    (&kernelContext_175)->work_0 = work_53;
    (&kernelContext_175)->frames_0 = frames_53;
    (&kernelContext_175)->bodies_0 = bodies_53;
    (&kernelContext_175)->chunks_0 = chunks_53;
    (&kernelContext_175)->sort_0 = sort_53;
    (&kernelContext_175)->manifolds_0 = manifolds_53;
    (&kernelContext_175)->loads_0 = loads_53;
    (&kernelContext_175)->bonds_0 = bonds_53;
    (&kernelContext_175)->args_0 = args_53;
    (&kernelContext_175)->info_0 = info_53;
    (&kernelContext_175)->scan_0 = scan_53;
    (&kernelContext_175)->parts_0 = parts_53;
    (&kernelContext_175)->colliders_0 = colliders_53;
    (&kernelContext_175)->collider_info_0 = collider_info_53;
    (&kernelContext_175)->view_chunks_0 = view_chunks_53;
    (&kernelContext_175)->view_0 = view_53;
    (&kernelContext_175)->objects_0 = objects_53;
    (&kernelContext_175)->shot_bodies_0 = shot_bodies_53;
    (&kernelContext_175)->lines_0 = lines_53;
    (&kernelContext_175)->geometry_0 = geometry_53;
    (&kernelContext_175)->shots_0 = shots_53;
    (&kernelContext_175)->lp_0 = lp_53;
    (&kernelContext_175)->ext_0 = ext_55;
    (&kernelContext_175)->engine_contacts_0 = engine_contacts_53;
    (&kernelContext_175)->crush_0 = crush_53;
    (&kernelContext_175)->crush_rw_0 = crush_rw_53;
    (&kernelContext_175)->crush_tab_0 = crush_tab_53;
    threadgroup array<uint, int(1024)> g_scan_53;
    (&kernelContext_175)->g_scan_0 = &g_scan_53;
    threadgroup array<float, int(64)> g_il_best_53;
    (&kernelContext_175)->g_il_best_0 = &g_il_best_53;
    il_tube_count_0(id_49.x, &kernelContext_175);
    return;
}

[[kernel]] void fr_csl_tube_base(uint3 id_50 [[thread_position_in_grid]], FractureParams_0 constant* params_54 [[buffer(0)]], atomic<uint> device* atom_54 [[buffer(5)]], uint device* work_54 [[buffer(4)]], packed_float4 device* frames_54 [[buffer(6)]], packed_float4 device* bodies_54 [[buffer(7)]], packed_float4 device* chunks_54 [[buffer(1)]], uint device* sort_54 [[buffer(11)]], Record_natural_0 device* manifolds_54 [[buffer(10)]], packed_float4 device* loads_54 [[buffer(12)]], packed_uint4 device* bonds_54 [[buffer(2)]], uint device* args_54 [[buffer(13)]], uint device* info_54 [[buffer(15)]], ScanParams_0 constant* scan_54 [[buffer(14)]], packed_float4 device* parts_54 [[buffer(3)]], packed_float4 device* colliders_54 [[buffer(8)]], packed_uint4 device* collider_info_54 [[buffer(9)]], packed_float4 device* view_chunks_54 [[buffer(17)]], ViewParams_0 constant* view_54 [[buffer(16)]], packed_float4 device* objects_54 [[buffer(18)]], uint device* shot_bodies_54 [[buffer(20)]], float device* lines_54 [[buffer(19)]], packed_float4 device* geometry_54 [[buffer(21)]], packed_float4 device* shots_54 [[buffer(22)]], packed_float4 device* lp_54 [[buffer(24)]], packed_float4 device* ext_56 [[buffer(23)]], packed_float4 device* engine_contacts_54 [[buffer(25)]], CrushParams_0 constant* crush_54 [[buffer(28)]], packed_float4 device* crush_rw_54 [[buffer(27)]], packed_float4 device* crush_tab_54 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_176;
    (&kernelContext_176)->params_0 = params_54;
    (&kernelContext_176)->atom_0 = atom_54;
    (&kernelContext_176)->work_0 = work_54;
    (&kernelContext_176)->frames_0 = frames_54;
    (&kernelContext_176)->bodies_0 = bodies_54;
    (&kernelContext_176)->chunks_0 = chunks_54;
    (&kernelContext_176)->sort_0 = sort_54;
    (&kernelContext_176)->manifolds_0 = manifolds_54;
    (&kernelContext_176)->loads_0 = loads_54;
    (&kernelContext_176)->bonds_0 = bonds_54;
    (&kernelContext_176)->args_0 = args_54;
    (&kernelContext_176)->info_0 = info_54;
    (&kernelContext_176)->scan_0 = scan_54;
    (&kernelContext_176)->parts_0 = parts_54;
    (&kernelContext_176)->colliders_0 = colliders_54;
    (&kernelContext_176)->collider_info_0 = collider_info_54;
    (&kernelContext_176)->view_chunks_0 = view_chunks_54;
    (&kernelContext_176)->view_0 = view_54;
    (&kernelContext_176)->objects_0 = objects_54;
    (&kernelContext_176)->shot_bodies_0 = shot_bodies_54;
    (&kernelContext_176)->lines_0 = lines_54;
    (&kernelContext_176)->geometry_0 = geometry_54;
    (&kernelContext_176)->shots_0 = shots_54;
    (&kernelContext_176)->lp_0 = lp_54;
    (&kernelContext_176)->ext_0 = ext_56;
    (&kernelContext_176)->engine_contacts_0 = engine_contacts_54;
    (&kernelContext_176)->crush_0 = crush_54;
    (&kernelContext_176)->crush_rw_0 = crush_rw_54;
    (&kernelContext_176)->crush_tab_0 = crush_tab_54;
    threadgroup array<uint, int(1024)> g_scan_54;
    (&kernelContext_176)->g_scan_0 = &g_scan_54;
    threadgroup array<float, int(64)> g_il_best_54;
    (&kernelContext_176)->g_il_best_0 = &g_il_best_54;
    il_tube_base_0(id_50.x, &kernelContext_176);
    return;
}

[[kernel]] void fr_csl_tube_collect(uint3 id_51 [[thread_position_in_grid]], FractureParams_0 constant* params_55 [[buffer(0)]], atomic<uint> device* atom_55 [[buffer(5)]], uint device* work_55 [[buffer(4)]], packed_float4 device* frames_55 [[buffer(6)]], packed_float4 device* bodies_55 [[buffer(7)]], packed_float4 device* chunks_55 [[buffer(1)]], uint device* sort_55 [[buffer(11)]], Record_natural_0 device* manifolds_55 [[buffer(10)]], packed_float4 device* loads_55 [[buffer(12)]], packed_uint4 device* bonds_55 [[buffer(2)]], uint device* args_55 [[buffer(13)]], uint device* info_55 [[buffer(15)]], ScanParams_0 constant* scan_55 [[buffer(14)]], packed_float4 device* parts_55 [[buffer(3)]], packed_float4 device* colliders_55 [[buffer(8)]], packed_uint4 device* collider_info_55 [[buffer(9)]], packed_float4 device* view_chunks_55 [[buffer(17)]], ViewParams_0 constant* view_55 [[buffer(16)]], packed_float4 device* objects_55 [[buffer(18)]], uint device* shot_bodies_55 [[buffer(20)]], float device* lines_55 [[buffer(19)]], packed_float4 device* geometry_55 [[buffer(21)]], packed_float4 device* shots_55 [[buffer(22)]], packed_float4 device* lp_55 [[buffer(24)]], packed_float4 device* ext_57 [[buffer(23)]], packed_float4 device* engine_contacts_55 [[buffer(25)]], CrushParams_0 constant* crush_55 [[buffer(28)]], packed_float4 device* crush_rw_55 [[buffer(27)]], packed_float4 device* crush_tab_55 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_177;
    (&kernelContext_177)->params_0 = params_55;
    (&kernelContext_177)->atom_0 = atom_55;
    (&kernelContext_177)->work_0 = work_55;
    (&kernelContext_177)->frames_0 = frames_55;
    (&kernelContext_177)->bodies_0 = bodies_55;
    (&kernelContext_177)->chunks_0 = chunks_55;
    (&kernelContext_177)->sort_0 = sort_55;
    (&kernelContext_177)->manifolds_0 = manifolds_55;
    (&kernelContext_177)->loads_0 = loads_55;
    (&kernelContext_177)->bonds_0 = bonds_55;
    (&kernelContext_177)->args_0 = args_55;
    (&kernelContext_177)->info_0 = info_55;
    (&kernelContext_177)->scan_0 = scan_55;
    (&kernelContext_177)->parts_0 = parts_55;
    (&kernelContext_177)->colliders_0 = colliders_55;
    (&kernelContext_177)->collider_info_0 = collider_info_55;
    (&kernelContext_177)->view_chunks_0 = view_chunks_55;
    (&kernelContext_177)->view_0 = view_55;
    (&kernelContext_177)->objects_0 = objects_55;
    (&kernelContext_177)->shot_bodies_0 = shot_bodies_55;
    (&kernelContext_177)->lines_0 = lines_55;
    (&kernelContext_177)->geometry_0 = geometry_55;
    (&kernelContext_177)->shots_0 = shots_55;
    (&kernelContext_177)->lp_0 = lp_55;
    (&kernelContext_177)->ext_0 = ext_57;
    (&kernelContext_177)->engine_contacts_0 = engine_contacts_55;
    (&kernelContext_177)->crush_0 = crush_55;
    (&kernelContext_177)->crush_rw_0 = crush_rw_55;
    (&kernelContext_177)->crush_tab_0 = crush_tab_55;
    threadgroup array<uint, int(1024)> g_scan_55;
    (&kernelContext_177)->g_scan_0 = &g_scan_55;
    threadgroup array<float, int(64)> g_il_best_55;
    (&kernelContext_177)->g_il_best_0 = &g_il_best_55;
    il_tube_collect_0(id_51.x, &kernelContext_177);
    return;
}

uint w_cs_list_0(KernelContext_0 thread* kernelContext_178)
{
    uint _S1222 = w_carry_0(kernelContext_178);
    return _S1222 + kernelContext_178->params_0->chunk_count_0;
}

uint w_cs_bodies_0(KernelContext_0 thread* kernelContext_179)
{
    uint _S1223 = w_cs_list_0(kernelContext_179);
    return _S1223 + kernelContext_179->params_0->chunk_count_0;
}

[[kernel]] void fr_csl_run(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_56 [[buffer(0)]], atomic<uint> device* atom_56 [[buffer(5)]], uint device* work_56 [[buffer(4)]], packed_float4 device* frames_56 [[buffer(6)]], packed_float4 device* bodies_56 [[buffer(7)]], packed_float4 device* chunks_56 [[buffer(1)]], uint device* sort_56 [[buffer(11)]], Record_natural_0 device* manifolds_56 [[buffer(10)]], packed_float4 device* loads_56 [[buffer(12)]], packed_uint4 device* bonds_56 [[buffer(2)]], uint device* args_56 [[buffer(13)]], uint device* info_56 [[buffer(15)]], ScanParams_0 constant* scan_56 [[buffer(14)]], packed_float4 device* parts_56 [[buffer(3)]], packed_float4 device* colliders_56 [[buffer(8)]], packed_uint4 device* collider_info_56 [[buffer(9)]], packed_float4 device* view_chunks_56 [[buffer(17)]], ViewParams_0 constant* view_56 [[buffer(16)]], packed_float4 device* objects_56 [[buffer(18)]], uint device* shot_bodies_56 [[buffer(20)]], float device* lines_56 [[buffer(19)]], packed_float4 device* geometry_56 [[buffer(21)]], packed_float4 device* shots_56 [[buffer(22)]], packed_float4 device* lp_56 [[buffer(24)]], packed_float4 device* ext_58 [[buffer(23)]], packed_float4 device* engine_contacts_56 [[buffer(25)]], CrushParams_0 constant* crush_56 [[buffer(28)]], packed_float4 device* crush_rw_56 [[buffer(27)]], packed_float4 device* crush_tab_56 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_180;
    (&kernelContext_180)->params_0 = params_56;
    (&kernelContext_180)->atom_0 = atom_56;
    (&kernelContext_180)->work_0 = work_56;
    (&kernelContext_180)->frames_0 = frames_56;
    (&kernelContext_180)->bodies_0 = bodies_56;
    (&kernelContext_180)->chunks_0 = chunks_56;
    (&kernelContext_180)->sort_0 = sort_56;
    (&kernelContext_180)->manifolds_0 = manifolds_56;
    (&kernelContext_180)->loads_0 = loads_56;
    (&kernelContext_180)->bonds_0 = bonds_56;
    (&kernelContext_180)->args_0 = args_56;
    (&kernelContext_180)->info_0 = info_56;
    (&kernelContext_180)->scan_0 = scan_56;
    (&kernelContext_180)->parts_0 = parts_56;
    (&kernelContext_180)->colliders_0 = colliders_56;
    (&kernelContext_180)->collider_info_0 = collider_info_56;
    (&kernelContext_180)->view_chunks_0 = view_chunks_56;
    (&kernelContext_180)->view_0 = view_56;
    (&kernelContext_180)->objects_0 = objects_56;
    (&kernelContext_180)->shot_bodies_0 = shot_bodies_56;
    (&kernelContext_180)->lines_0 = lines_56;
    (&kernelContext_180)->geometry_0 = geometry_56;
    (&kernelContext_180)->shots_0 = shots_56;
    (&kernelContext_180)->lp_0 = lp_56;
    (&kernelContext_180)->ext_0 = ext_58;
    (&kernelContext_180)->engine_contacts_0 = engine_contacts_56;
    (&kernelContext_180)->crush_0 = crush_56;
    (&kernelContext_180)->crush_rw_0 = crush_rw_56;
    (&kernelContext_180)->crush_tab_0 = crush_tab_56;
    threadgroup array<uint, int(1024)> g_scan_56;
    (&kernelContext_180)->g_scan_0 = &g_scan_56;
    threadgroup array<float, int(64)> g_il_best_56;
    (&kernelContext_180)->g_il_best_0 = &g_il_best_56;
    uint _S1224 = w_cs_bodies_0(&kernelContext_180);
    il_run_0(*((&kernelContext_180)->work_0+(_S1224 + group_2.x)), thread_5.x, &kernelContext_180);
    return;
}

[[kernel]] void fr_csl_strike(uint3 id_52 [[thread_position_in_grid]], FractureParams_0 constant* params_57 [[buffer(0)]], atomic<uint> device* atom_57 [[buffer(5)]], uint device* work_57 [[buffer(4)]], packed_float4 device* frames_57 [[buffer(6)]], packed_float4 device* bodies_57 [[buffer(7)]], packed_float4 device* chunks_57 [[buffer(1)]], uint device* sort_57 [[buffer(11)]], Record_natural_0 device* manifolds_57 [[buffer(10)]], packed_float4 device* loads_57 [[buffer(12)]], packed_uint4 device* bonds_57 [[buffer(2)]], uint device* args_57 [[buffer(13)]], uint device* info_57 [[buffer(15)]], ScanParams_0 constant* scan_57 [[buffer(14)]], packed_float4 device* parts_57 [[buffer(3)]], packed_float4 device* colliders_57 [[buffer(8)]], packed_uint4 device* collider_info_57 [[buffer(9)]], packed_float4 device* view_chunks_57 [[buffer(17)]], ViewParams_0 constant* view_57 [[buffer(16)]], packed_float4 device* objects_57 [[buffer(18)]], uint device* shot_bodies_57 [[buffer(20)]], float device* lines_57 [[buffer(19)]], packed_float4 device* geometry_57 [[buffer(21)]], packed_float4 device* shots_57 [[buffer(22)]], packed_float4 device* lp_57 [[buffer(24)]], packed_float4 device* ext_59 [[buffer(23)]], packed_float4 device* engine_contacts_57 [[buffer(25)]], CrushParams_0 constant* crush_57 [[buffer(28)]], packed_float4 device* crush_rw_57 [[buffer(27)]], packed_float4 device* crush_tab_57 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_181;
    (&kernelContext_181)->params_0 = params_57;
    (&kernelContext_181)->atom_0 = atom_57;
    (&kernelContext_181)->work_0 = work_57;
    (&kernelContext_181)->frames_0 = frames_57;
    (&kernelContext_181)->bodies_0 = bodies_57;
    (&kernelContext_181)->chunks_0 = chunks_57;
    (&kernelContext_181)->sort_0 = sort_57;
    (&kernelContext_181)->manifolds_0 = manifolds_57;
    (&kernelContext_181)->loads_0 = loads_57;
    (&kernelContext_181)->bonds_0 = bonds_57;
    (&kernelContext_181)->args_0 = args_57;
    (&kernelContext_181)->info_0 = info_57;
    (&kernelContext_181)->scan_0 = scan_57;
    (&kernelContext_181)->parts_0 = parts_57;
    (&kernelContext_181)->colliders_0 = colliders_57;
    (&kernelContext_181)->collider_info_0 = collider_info_57;
    (&kernelContext_181)->view_chunks_0 = view_chunks_57;
    (&kernelContext_181)->view_0 = view_57;
    (&kernelContext_181)->objects_0 = objects_57;
    (&kernelContext_181)->shot_bodies_0 = shot_bodies_57;
    (&kernelContext_181)->lines_0 = lines_57;
    (&kernelContext_181)->geometry_0 = geometry_57;
    (&kernelContext_181)->shots_0 = shots_57;
    (&kernelContext_181)->lp_0 = lp_57;
    (&kernelContext_181)->ext_0 = ext_59;
    (&kernelContext_181)->engine_contacts_0 = engine_contacts_57;
    (&kernelContext_181)->crush_0 = crush_57;
    (&kernelContext_181)->crush_rw_0 = crush_rw_57;
    (&kernelContext_181)->crush_tab_0 = crush_tab_57;
    threadgroup array<uint, int(1024)> g_scan_57;
    (&kernelContext_181)->g_scan_0 = &g_scan_57;
    threadgroup array<float, int(64)> g_il_best_57;
    (&kernelContext_181)->g_il_best_0 = &g_il_best_57;
    il_strike_0(id_52.x, &kernelContext_181);
    return;
}

[[kernel]] void fr_csl_mark(uint3 id_53 [[thread_position_in_grid]], FractureParams_0 constant* params_58 [[buffer(0)]], atomic<uint> device* atom_58 [[buffer(5)]], uint device* work_58 [[buffer(4)]], packed_float4 device* frames_58 [[buffer(6)]], packed_float4 device* bodies_58 [[buffer(7)]], packed_float4 device* chunks_58 [[buffer(1)]], uint device* sort_58 [[buffer(11)]], Record_natural_0 device* manifolds_58 [[buffer(10)]], packed_float4 device* loads_58 [[buffer(12)]], packed_uint4 device* bonds_58 [[buffer(2)]], uint device* args_58 [[buffer(13)]], uint device* info_58 [[buffer(15)]], ScanParams_0 constant* scan_58 [[buffer(14)]], packed_float4 device* parts_58 [[buffer(3)]], packed_float4 device* colliders_58 [[buffer(8)]], packed_uint4 device* collider_info_58 [[buffer(9)]], packed_float4 device* view_chunks_58 [[buffer(17)]], ViewParams_0 constant* view_58 [[buffer(16)]], packed_float4 device* objects_58 [[buffer(18)]], uint device* shot_bodies_58 [[buffer(20)]], float device* lines_58 [[buffer(19)]], packed_float4 device* geometry_58 [[buffer(21)]], packed_float4 device* shots_58 [[buffer(22)]], packed_float4 device* lp_58 [[buffer(24)]], packed_float4 device* ext_60 [[buffer(23)]], packed_float4 device* engine_contacts_58 [[buffer(25)]], CrushParams_0 constant* crush_58 [[buffer(28)]], packed_float4 device* crush_rw_58 [[buffer(27)]], packed_float4 device* crush_tab_58 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_182;
    (&kernelContext_182)->params_0 = params_58;
    (&kernelContext_182)->atom_0 = atom_58;
    (&kernelContext_182)->work_0 = work_58;
    (&kernelContext_182)->frames_0 = frames_58;
    (&kernelContext_182)->bodies_0 = bodies_58;
    (&kernelContext_182)->chunks_0 = chunks_58;
    (&kernelContext_182)->sort_0 = sort_58;
    (&kernelContext_182)->manifolds_0 = manifolds_58;
    (&kernelContext_182)->loads_0 = loads_58;
    (&kernelContext_182)->bonds_0 = bonds_58;
    (&kernelContext_182)->args_0 = args_58;
    (&kernelContext_182)->info_0 = info_58;
    (&kernelContext_182)->scan_0 = scan_58;
    (&kernelContext_182)->parts_0 = parts_58;
    (&kernelContext_182)->colliders_0 = colliders_58;
    (&kernelContext_182)->collider_info_0 = collider_info_58;
    (&kernelContext_182)->view_chunks_0 = view_chunks_58;
    (&kernelContext_182)->view_0 = view_58;
    (&kernelContext_182)->objects_0 = objects_58;
    (&kernelContext_182)->shot_bodies_0 = shot_bodies_58;
    (&kernelContext_182)->lines_0 = lines_58;
    (&kernelContext_182)->geometry_0 = geometry_58;
    (&kernelContext_182)->shots_0 = shots_58;
    (&kernelContext_182)->lp_0 = lp_58;
    (&kernelContext_182)->ext_0 = ext_60;
    (&kernelContext_182)->engine_contacts_0 = engine_contacts_58;
    (&kernelContext_182)->crush_0 = crush_58;
    (&kernelContext_182)->crush_rw_0 = crush_rw_58;
    (&kernelContext_182)->crush_tab_0 = crush_tab_58;
    threadgroup array<uint, int(1024)> g_scan_58;
    (&kernelContext_182)->g_scan_0 = &g_scan_58;
    threadgroup array<float, int(64)> g_il_best_58;
    (&kernelContext_182)->g_il_best_0 = &g_il_best_58;
    il_mark_0(id_53.x, &kernelContext_182);
    return;
}

[[kernel]] void fr_csl_carry(uint3 id_54 [[thread_position_in_grid]], FractureParams_0 constant* params_59 [[buffer(0)]], atomic<uint> device* atom_59 [[buffer(5)]], uint device* work_59 [[buffer(4)]], packed_float4 device* frames_59 [[buffer(6)]], packed_float4 device* bodies_59 [[buffer(7)]], packed_float4 device* chunks_59 [[buffer(1)]], uint device* sort_59 [[buffer(11)]], Record_natural_0 device* manifolds_59 [[buffer(10)]], packed_float4 device* loads_59 [[buffer(12)]], packed_uint4 device* bonds_59 [[buffer(2)]], uint device* args_59 [[buffer(13)]], uint device* info_59 [[buffer(15)]], ScanParams_0 constant* scan_59 [[buffer(14)]], packed_float4 device* parts_59 [[buffer(3)]], packed_float4 device* colliders_59 [[buffer(8)]], packed_uint4 device* collider_info_59 [[buffer(9)]], packed_float4 device* view_chunks_59 [[buffer(17)]], ViewParams_0 constant* view_59 [[buffer(16)]], packed_float4 device* objects_59 [[buffer(18)]], uint device* shot_bodies_59 [[buffer(20)]], float device* lines_59 [[buffer(19)]], packed_float4 device* geometry_59 [[buffer(21)]], packed_float4 device* shots_59 [[buffer(22)]], packed_float4 device* lp_59 [[buffer(24)]], packed_float4 device* ext_61 [[buffer(23)]], packed_float4 device* engine_contacts_59 [[buffer(25)]], CrushParams_0 constant* crush_59 [[buffer(28)]], packed_float4 device* crush_rw_59 [[buffer(27)]], packed_float4 device* crush_tab_59 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_183;
    (&kernelContext_183)->params_0 = params_59;
    (&kernelContext_183)->atom_0 = atom_59;
    (&kernelContext_183)->work_0 = work_59;
    (&kernelContext_183)->frames_0 = frames_59;
    (&kernelContext_183)->bodies_0 = bodies_59;
    (&kernelContext_183)->chunks_0 = chunks_59;
    (&kernelContext_183)->sort_0 = sort_59;
    (&kernelContext_183)->manifolds_0 = manifolds_59;
    (&kernelContext_183)->loads_0 = loads_59;
    (&kernelContext_183)->bonds_0 = bonds_59;
    (&kernelContext_183)->args_0 = args_59;
    (&kernelContext_183)->info_0 = info_59;
    (&kernelContext_183)->scan_0 = scan_59;
    (&kernelContext_183)->parts_0 = parts_59;
    (&kernelContext_183)->colliders_0 = colliders_59;
    (&kernelContext_183)->collider_info_0 = collider_info_59;
    (&kernelContext_183)->view_chunks_0 = view_chunks_59;
    (&kernelContext_183)->view_0 = view_59;
    (&kernelContext_183)->objects_0 = objects_59;
    (&kernelContext_183)->shot_bodies_0 = shot_bodies_59;
    (&kernelContext_183)->lines_0 = lines_59;
    (&kernelContext_183)->geometry_0 = geometry_59;
    (&kernelContext_183)->shots_0 = shots_59;
    (&kernelContext_183)->lp_0 = lp_59;
    (&kernelContext_183)->ext_0 = ext_61;
    (&kernelContext_183)->engine_contacts_0 = engine_contacts_59;
    (&kernelContext_183)->crush_0 = crush_59;
    (&kernelContext_183)->crush_rw_0 = crush_rw_59;
    (&kernelContext_183)->crush_tab_0 = crush_tab_59;
    threadgroup array<uint, int(1024)> g_scan_59;
    (&kernelContext_183)->g_scan_0 = &g_scan_59;
    threadgroup array<float, int(64)> g_il_best_59;
    (&kernelContext_183)->g_il_best_0 = &g_il_best_59;
    il_carry_0(id_54.x, &kernelContext_183);
    return;
}

[[kernel]] void fr_csl_breaks(uint3 id_55 [[thread_position_in_grid]], FractureParams_0 constant* params_60 [[buffer(0)]], atomic<uint> device* atom_60 [[buffer(5)]], uint device* work_60 [[buffer(4)]], packed_float4 device* frames_60 [[buffer(6)]], packed_float4 device* bodies_60 [[buffer(7)]], packed_float4 device* chunks_60 [[buffer(1)]], uint device* sort_60 [[buffer(11)]], Record_natural_0 device* manifolds_60 [[buffer(10)]], packed_float4 device* loads_60 [[buffer(12)]], packed_uint4 device* bonds_60 [[buffer(2)]], uint device* args_60 [[buffer(13)]], uint device* info_60 [[buffer(15)]], ScanParams_0 constant* scan_60 [[buffer(14)]], packed_float4 device* parts_60 [[buffer(3)]], packed_float4 device* colliders_60 [[buffer(8)]], packed_uint4 device* collider_info_60 [[buffer(9)]], packed_float4 device* view_chunks_60 [[buffer(17)]], ViewParams_0 constant* view_60 [[buffer(16)]], packed_float4 device* objects_60 [[buffer(18)]], uint device* shot_bodies_60 [[buffer(20)]], float device* lines_60 [[buffer(19)]], packed_float4 device* geometry_60 [[buffer(21)]], packed_float4 device* shots_60 [[buffer(22)]], packed_float4 device* lp_60 [[buffer(24)]], packed_float4 device* ext_62 [[buffer(23)]], packed_float4 device* engine_contacts_60 [[buffer(25)]], CrushParams_0 constant* crush_60 [[buffer(28)]], packed_float4 device* crush_rw_60 [[buffer(27)]], packed_float4 device* crush_tab_60 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_184;
    (&kernelContext_184)->params_0 = params_60;
    (&kernelContext_184)->atom_0 = atom_60;
    (&kernelContext_184)->work_0 = work_60;
    (&kernelContext_184)->frames_0 = frames_60;
    (&kernelContext_184)->bodies_0 = bodies_60;
    (&kernelContext_184)->chunks_0 = chunks_60;
    (&kernelContext_184)->sort_0 = sort_60;
    (&kernelContext_184)->manifolds_0 = manifolds_60;
    (&kernelContext_184)->loads_0 = loads_60;
    (&kernelContext_184)->bonds_0 = bonds_60;
    (&kernelContext_184)->args_0 = args_60;
    (&kernelContext_184)->info_0 = info_60;
    (&kernelContext_184)->scan_0 = scan_60;
    (&kernelContext_184)->parts_0 = parts_60;
    (&kernelContext_184)->colliders_0 = colliders_60;
    (&kernelContext_184)->collider_info_0 = collider_info_60;
    (&kernelContext_184)->view_chunks_0 = view_chunks_60;
    (&kernelContext_184)->view_0 = view_60;
    (&kernelContext_184)->objects_0 = objects_60;
    (&kernelContext_184)->shot_bodies_0 = shot_bodies_60;
    (&kernelContext_184)->lines_0 = lines_60;
    (&kernelContext_184)->geometry_0 = geometry_60;
    (&kernelContext_184)->shots_0 = shots_60;
    (&kernelContext_184)->lp_0 = lp_60;
    (&kernelContext_184)->ext_0 = ext_62;
    (&kernelContext_184)->engine_contacts_0 = engine_contacts_60;
    (&kernelContext_184)->crush_0 = crush_60;
    (&kernelContext_184)->crush_rw_0 = crush_rw_60;
    (&kernelContext_184)->crush_tab_0 = crush_tab_60;
    threadgroup array<uint, int(1024)> g_scan_60;
    (&kernelContext_184)->g_scan_0 = &g_scan_60;
    threadgroup array<float, int(64)> g_il_best_60;
    (&kernelContext_184)->g_il_best_0 = &g_il_best_60;
    il_breaks_0(id_55.x, &kernelContext_184);
    return;
}

[[kernel]] void fr_cs_begin(uint3 id_56 [[thread_position_in_grid]], FractureParams_0 constant* params_61 [[buffer(0)]], atomic<uint> device* atom_61 [[buffer(5)]], uint device* work_61 [[buffer(4)]], packed_float4 device* frames_61 [[buffer(6)]], packed_float4 device* bodies_61 [[buffer(7)]], packed_float4 device* chunks_61 [[buffer(1)]], uint device* sort_61 [[buffer(11)]], Record_natural_0 device* manifolds_61 [[buffer(10)]], packed_float4 device* loads_61 [[buffer(12)]], packed_uint4 device* bonds_61 [[buffer(2)]], uint device* args_61 [[buffer(13)]], uint device* info_61 [[buffer(15)]], ScanParams_0 constant* scan_61 [[buffer(14)]], packed_float4 device* parts_61 [[buffer(3)]], packed_float4 device* colliders_61 [[buffer(8)]], packed_uint4 device* collider_info_61 [[buffer(9)]], packed_float4 device* view_chunks_61 [[buffer(17)]], ViewParams_0 constant* view_61 [[buffer(16)]], packed_float4 device* objects_61 [[buffer(18)]], uint device* shot_bodies_61 [[buffer(20)]], float device* lines_61 [[buffer(19)]], packed_float4 device* geometry_61 [[buffer(21)]], packed_float4 device* shots_61 [[buffer(22)]], packed_float4 device* lp_61 [[buffer(24)]], packed_float4 device* ext_63 [[buffer(23)]], packed_float4 device* engine_contacts_61 [[buffer(25)]], CrushParams_0 constant* crush_61 [[buffer(28)]], packed_float4 device* crush_rw_61 [[buffer(27)]], packed_float4 device* crush_tab_61 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_185;
    (&kernelContext_185)->params_0 = params_61;
    (&kernelContext_185)->atom_0 = atom_61;
    (&kernelContext_185)->work_0 = work_61;
    (&kernelContext_185)->frames_0 = frames_61;
    (&kernelContext_185)->bodies_0 = bodies_61;
    (&kernelContext_185)->chunks_0 = chunks_61;
    (&kernelContext_185)->sort_0 = sort_61;
    (&kernelContext_185)->manifolds_0 = manifolds_61;
    (&kernelContext_185)->loads_0 = loads_61;
    (&kernelContext_185)->bonds_0 = bonds_61;
    (&kernelContext_185)->args_0 = args_61;
    (&kernelContext_185)->info_0 = info_61;
    (&kernelContext_185)->scan_0 = scan_61;
    (&kernelContext_185)->parts_0 = parts_61;
    (&kernelContext_185)->colliders_0 = colliders_61;
    (&kernelContext_185)->collider_info_0 = collider_info_61;
    (&kernelContext_185)->view_chunks_0 = view_chunks_61;
    (&kernelContext_185)->view_0 = view_61;
    (&kernelContext_185)->objects_0 = objects_61;
    (&kernelContext_185)->shot_bodies_0 = shot_bodies_61;
    (&kernelContext_185)->lines_0 = lines_61;
    (&kernelContext_185)->geometry_0 = geometry_61;
    (&kernelContext_185)->shots_0 = shots_61;
    (&kernelContext_185)->lp_0 = lp_61;
    (&kernelContext_185)->ext_0 = ext_63;
    (&kernelContext_185)->engine_contacts_0 = engine_contacts_61;
    (&kernelContext_185)->crush_0 = crush_61;
    (&kernelContext_185)->crush_rw_0 = crush_rw_61;
    (&kernelContext_185)->crush_tab_0 = crush_tab_61;
    threadgroup array<uint, int(1024)> g_scan_61;
    (&kernelContext_185)->g_scan_0 = &g_scan_61;
    threadgroup array<float, int(64)> g_il_best_61;
    (&kernelContext_185)->g_il_best_0 = &g_il_best_61;
    uint _S1225 = a_counters_1(&kernelContext_185);
    atomic_store_explicit((&kernelContext_185)->atom_0+(_S1225 + 18U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_185)->atom_0+(_S1225 + 19U), 0U, memory_order_relaxed);
    uint _S1226 = a_il_cursor_1(&kernelContext_185);
    atomic_store_explicit((&kernelContext_185)->atom_0+_S1226, 0U, memory_order_relaxed);
    uint _S1227 = cs_rows_0(&kernelContext_185);
    uint _S1228 = groups_of_0(_S1227, 64U);
    set_args_0(7U, _S1228, &kernelContext_185);
    return;
}

[[kernel]] void fr_cs_contacts(uint3 id_57 [[thread_position_in_grid]], FractureParams_0 constant* params_62 [[buffer(0)]], atomic<uint> device* atom_62 [[buffer(5)]], uint device* work_62 [[buffer(4)]], packed_float4 device* frames_62 [[buffer(6)]], packed_float4 device* bodies_62 [[buffer(7)]], packed_float4 device* chunks_62 [[buffer(1)]], uint device* sort_62 [[buffer(11)]], Record_natural_0 device* manifolds_62 [[buffer(10)]], packed_float4 device* loads_62 [[buffer(12)]], packed_uint4 device* bonds_62 [[buffer(2)]], uint device* args_62 [[buffer(13)]], uint device* info_62 [[buffer(15)]], ScanParams_0 constant* scan_62 [[buffer(14)]], packed_float4 device* parts_62 [[buffer(3)]], packed_float4 device* colliders_62 [[buffer(8)]], packed_uint4 device* collider_info_62 [[buffer(9)]], packed_float4 device* view_chunks_62 [[buffer(17)]], ViewParams_0 constant* view_62 [[buffer(16)]], packed_float4 device* objects_62 [[buffer(18)]], uint device* shot_bodies_62 [[buffer(20)]], float device* lines_62 [[buffer(19)]], packed_float4 device* geometry_62 [[buffer(21)]], packed_float4 device* shots_62 [[buffer(22)]], packed_float4 device* lp_62 [[buffer(24)]], packed_float4 device* ext_64 [[buffer(23)]], packed_float4 device* engine_contacts_62 [[buffer(25)]], CrushParams_0 constant* crush_62 [[buffer(28)]], packed_float4 device* crush_rw_62 [[buffer(27)]], packed_float4 device* crush_tab_62 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_186;
    (&kernelContext_186)->params_0 = params_62;
    (&kernelContext_186)->atom_0 = atom_62;
    (&kernelContext_186)->work_0 = work_62;
    (&kernelContext_186)->frames_0 = frames_62;
    (&kernelContext_186)->bodies_0 = bodies_62;
    (&kernelContext_186)->chunks_0 = chunks_62;
    (&kernelContext_186)->sort_0 = sort_62;
    (&kernelContext_186)->manifolds_0 = manifolds_62;
    (&kernelContext_186)->loads_0 = loads_62;
    (&kernelContext_186)->bonds_0 = bonds_62;
    (&kernelContext_186)->args_0 = args_62;
    (&kernelContext_186)->info_0 = info_62;
    (&kernelContext_186)->scan_0 = scan_62;
    (&kernelContext_186)->parts_0 = parts_62;
    (&kernelContext_186)->colliders_0 = colliders_62;
    (&kernelContext_186)->collider_info_0 = collider_info_62;
    (&kernelContext_186)->view_chunks_0 = view_chunks_62;
    (&kernelContext_186)->view_0 = view_62;
    (&kernelContext_186)->objects_0 = objects_62;
    (&kernelContext_186)->shot_bodies_0 = shot_bodies_62;
    (&kernelContext_186)->lines_0 = lines_62;
    (&kernelContext_186)->geometry_0 = geometry_62;
    (&kernelContext_186)->shots_0 = shots_62;
    (&kernelContext_186)->lp_0 = lp_62;
    (&kernelContext_186)->ext_0 = ext_64;
    (&kernelContext_186)->engine_contacts_0 = engine_contacts_62;
    (&kernelContext_186)->crush_0 = crush_62;
    (&kernelContext_186)->crush_rw_0 = crush_rw_62;
    (&kernelContext_186)->crush_tab_0 = crush_tab_62;
    threadgroup array<uint, int(1024)> g_scan_62;
    (&kernelContext_186)->g_scan_0 = &g_scan_62;
    threadgroup array<float, int(64)> g_il_best_62;
    (&kernelContext_186)->g_il_best_0 = &g_il_best_62;
    uint r_24 = id_57.x;
    uint _S1229 = cs_rows_0(&kernelContext_186);
    if(r_24 >= _S1229)
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
        thread uint k_46;
        thread uint s_17;
        thread uint ob_5;
        thread uint co_5;
        thread float3 p_18;
        thread float3 u_9;
        thread float v_14;
        thread float m_eff_7;
        thread float t_in_8;
        bool _S1230 = cs_side_0(r_24, side_6, &k_46, &s_17, &ob_5, &co_5, &p_18, &u_9, &v_14, &m_eff_7, &t_in_8, &kernelContext_186);
        if(_S1230)
        {
            uint _S1231 = a_cs_0(s_17, &kernelContext_186);
            uint _S1232 = atomic_fetch_max_explicit((&kernelContext_186)->atom_0+_S1231, (as_type<uint>((m_eff_7 * v_14))), memory_order_relaxed);
            uint _S1233 = a_cs_0(s_17, &kernelContext_186);
            uint _S1234 = atomic_exchange_explicit((&kernelContext_186)->atom_0+(_S1233 + 8U), 1U, memory_order_relaxed);
            if(_S1234 == 0U)
            {
                uint _S1235 = w_cs_bodies_0(&kernelContext_186);
                uint _S1236 = a_counters_1(&kernelContext_186);
                uint _S1237 = atomic_fetch_add_explicit((&kernelContext_186)->atom_0+(_S1236 + 18U), 1U, memory_order_relaxed);
                *((&kernelContext_186)->work_0+(_S1235 + _S1237)) = s_17;
            }
        }
        side_6 = side_6 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_gate(uint3 id_58 [[thread_position_in_grid]], FractureParams_0 constant* params_63 [[buffer(0)]], atomic<uint> device* atom_63 [[buffer(5)]], uint device* work_63 [[buffer(4)]], packed_float4 device* frames_63 [[buffer(6)]], packed_float4 device* bodies_63 [[buffer(7)]], packed_float4 device* chunks_63 [[buffer(1)]], uint device* sort_63 [[buffer(11)]], Record_natural_0 device* manifolds_63 [[buffer(10)]], packed_float4 device* loads_63 [[buffer(12)]], packed_uint4 device* bonds_63 [[buffer(2)]], uint device* args_63 [[buffer(13)]], uint device* info_63 [[buffer(15)]], ScanParams_0 constant* scan_63 [[buffer(14)]], packed_float4 device* parts_63 [[buffer(3)]], packed_float4 device* colliders_63 [[buffer(8)]], packed_uint4 device* collider_info_63 [[buffer(9)]], packed_float4 device* view_chunks_63 [[buffer(17)]], ViewParams_0 constant* view_63 [[buffer(16)]], packed_float4 device* objects_63 [[buffer(18)]], uint device* shot_bodies_63 [[buffer(20)]], float device* lines_63 [[buffer(19)]], packed_float4 device* geometry_63 [[buffer(21)]], packed_float4 device* shots_63 [[buffer(22)]], packed_float4 device* lp_63 [[buffer(24)]], packed_float4 device* ext_65 [[buffer(23)]], packed_float4 device* engine_contacts_63 [[buffer(25)]], CrushParams_0 constant* crush_63 [[buffer(28)]], packed_float4 device* crush_rw_63 [[buffer(27)]], packed_float4 device* crush_tab_63 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_187;
    (&kernelContext_187)->params_0 = params_63;
    (&kernelContext_187)->atom_0 = atom_63;
    (&kernelContext_187)->work_0 = work_63;
    (&kernelContext_187)->frames_0 = frames_63;
    (&kernelContext_187)->bodies_0 = bodies_63;
    (&kernelContext_187)->chunks_0 = chunks_63;
    (&kernelContext_187)->sort_0 = sort_63;
    (&kernelContext_187)->manifolds_0 = manifolds_63;
    (&kernelContext_187)->loads_0 = loads_63;
    (&kernelContext_187)->bonds_0 = bonds_63;
    (&kernelContext_187)->args_0 = args_63;
    (&kernelContext_187)->info_0 = info_63;
    (&kernelContext_187)->scan_0 = scan_63;
    (&kernelContext_187)->parts_0 = parts_63;
    (&kernelContext_187)->colliders_0 = colliders_63;
    (&kernelContext_187)->collider_info_0 = collider_info_63;
    (&kernelContext_187)->view_chunks_0 = view_chunks_63;
    (&kernelContext_187)->view_0 = view_63;
    (&kernelContext_187)->objects_0 = objects_63;
    (&kernelContext_187)->shot_bodies_0 = shot_bodies_63;
    (&kernelContext_187)->lines_0 = lines_63;
    (&kernelContext_187)->geometry_0 = geometry_63;
    (&kernelContext_187)->shots_0 = shots_63;
    (&kernelContext_187)->lp_0 = lp_63;
    (&kernelContext_187)->ext_0 = ext_65;
    (&kernelContext_187)->engine_contacts_0 = engine_contacts_63;
    (&kernelContext_187)->crush_0 = crush_63;
    (&kernelContext_187)->crush_rw_0 = crush_rw_63;
    (&kernelContext_187)->crush_tab_0 = crush_tab_63;
    threadgroup array<uint, int(1024)> g_scan_63;
    (&kernelContext_187)->g_scan_0 = &g_scan_63;
    threadgroup array<float, int(64)> g_il_best_63;
    (&kernelContext_187)->g_il_best_0 = &g_il_best_63;
    uint _S1238 = a_counters_1(&kernelContext_187);
    uint struck_1 = atomic_load_explicit((&kernelContext_187)->atom_0+(_S1238 + 18U), memory_order_relaxed);
    bool any_0 = struck_1 > 0U;
    uint _S1239;
    if(any_0)
    {
        uint _S1240 = cs_rows_0(&kernelContext_187);
        uint _S1241 = groups_of_0(_S1240, 64U);
        _S1239 = _S1241;
    }
    else
    {
        _S1239 = 0U;
    }
    set_args_0(8U, _S1239, &kernelContext_187);
    uint _S1242 = groups_of_0(struck_1, 64U);
    set_args_0(9U, _S1242, &kernelContext_187);
    set_args_0(6U, struck_1, &kernelContext_187);
    if(any_0)
    {
        uint _S1243 = groups_of_0((&kernelContext_187)->params_0->chunk_count_0, 64U);
        _S1239 = _S1243;
    }
    else
    {
        _S1239 = 0U;
    }
    set_args_0(10U, _S1239, &kernelContext_187);
    if(any_0)
    {
        uint _S1244 = groups_of_0((&kernelContext_187)->params_0->bond_count_0, 64U);
        _S1239 = _S1244;
    }
    else
    {
        _S1239 = 0U;
    }
    set_args_0(11U, _S1239, &kernelContext_187);
    if(any_0)
    {
        uint _S1245 = groups_of_0((&kernelContext_187)->params_0->body_count_0, 64U);
        _S1239 = _S1245;
    }
    else
    {
        _S1239 = 0U;
    }
    set_args_0(12U, _S1239, &kernelContext_187);
    return;
}

[[kernel]] void fr_cs_pick_a(uint3 id_59 [[thread_position_in_grid]], FractureParams_0 constant* params_64 [[buffer(0)]], atomic<uint> device* atom_64 [[buffer(5)]], uint device* work_64 [[buffer(4)]], packed_float4 device* frames_64 [[buffer(6)]], packed_float4 device* bodies_64 [[buffer(7)]], packed_float4 device* chunks_64 [[buffer(1)]], uint device* sort_64 [[buffer(11)]], Record_natural_0 device* manifolds_64 [[buffer(10)]], packed_float4 device* loads_64 [[buffer(12)]], packed_uint4 device* bonds_64 [[buffer(2)]], uint device* args_64 [[buffer(13)]], uint device* info_64 [[buffer(15)]], ScanParams_0 constant* scan_64 [[buffer(14)]], packed_float4 device* parts_64 [[buffer(3)]], packed_float4 device* colliders_64 [[buffer(8)]], packed_uint4 device* collider_info_64 [[buffer(9)]], packed_float4 device* view_chunks_64 [[buffer(17)]], ViewParams_0 constant* view_64 [[buffer(16)]], packed_float4 device* objects_64 [[buffer(18)]], uint device* shot_bodies_64 [[buffer(20)]], float device* lines_64 [[buffer(19)]], packed_float4 device* geometry_64 [[buffer(21)]], packed_float4 device* shots_64 [[buffer(22)]], packed_float4 device* lp_64 [[buffer(24)]], packed_float4 device* ext_66 [[buffer(23)]], packed_float4 device* engine_contacts_64 [[buffer(25)]], CrushParams_0 constant* crush_64 [[buffer(28)]], packed_float4 device* crush_rw_64 [[buffer(27)]], packed_float4 device* crush_tab_64 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_188;
    (&kernelContext_188)->params_0 = params_64;
    (&kernelContext_188)->atom_0 = atom_64;
    (&kernelContext_188)->work_0 = work_64;
    (&kernelContext_188)->frames_0 = frames_64;
    (&kernelContext_188)->bodies_0 = bodies_64;
    (&kernelContext_188)->chunks_0 = chunks_64;
    (&kernelContext_188)->sort_0 = sort_64;
    (&kernelContext_188)->manifolds_0 = manifolds_64;
    (&kernelContext_188)->loads_0 = loads_64;
    (&kernelContext_188)->bonds_0 = bonds_64;
    (&kernelContext_188)->args_0 = args_64;
    (&kernelContext_188)->info_0 = info_64;
    (&kernelContext_188)->scan_0 = scan_64;
    (&kernelContext_188)->parts_0 = parts_64;
    (&kernelContext_188)->colliders_0 = colliders_64;
    (&kernelContext_188)->collider_info_0 = collider_info_64;
    (&kernelContext_188)->view_chunks_0 = view_chunks_64;
    (&kernelContext_188)->view_0 = view_64;
    (&kernelContext_188)->objects_0 = objects_64;
    (&kernelContext_188)->shot_bodies_0 = shot_bodies_64;
    (&kernelContext_188)->lines_0 = lines_64;
    (&kernelContext_188)->geometry_0 = geometry_64;
    (&kernelContext_188)->shots_0 = shots_64;
    (&kernelContext_188)->lp_0 = lp_64;
    (&kernelContext_188)->ext_0 = ext_66;
    (&kernelContext_188)->engine_contacts_0 = engine_contacts_64;
    (&kernelContext_188)->crush_0 = crush_64;
    (&kernelContext_188)->crush_rw_0 = crush_rw_64;
    (&kernelContext_188)->crush_tab_0 = crush_tab_64;
    threadgroup array<uint, int(1024)> g_scan_64;
    (&kernelContext_188)->g_scan_0 = &g_scan_64;
    threadgroup array<float, int(64)> g_il_best_64;
    (&kernelContext_188)->g_il_best_0 = &g_il_best_64;
    cs_pick_3(id_59.x, &kernelContext_188);
    return;
}

[[kernel]] void fr_cs_pick_b(uint3 id_60 [[thread_position_in_grid]], FractureParams_0 constant* params_65 [[buffer(0)]], atomic<uint> device* atom_65 [[buffer(5)]], uint device* work_65 [[buffer(4)]], packed_float4 device* frames_65 [[buffer(6)]], packed_float4 device* bodies_65 [[buffer(7)]], packed_float4 device* chunks_65 [[buffer(1)]], uint device* sort_65 [[buffer(11)]], Record_natural_0 device* manifolds_65 [[buffer(10)]], packed_float4 device* loads_65 [[buffer(12)]], packed_uint4 device* bonds_65 [[buffer(2)]], uint device* args_65 [[buffer(13)]], uint device* info_65 [[buffer(15)]], ScanParams_0 constant* scan_65 [[buffer(14)]], packed_float4 device* parts_65 [[buffer(3)]], packed_float4 device* colliders_65 [[buffer(8)]], packed_uint4 device* collider_info_65 [[buffer(9)]], packed_float4 device* view_chunks_65 [[buffer(17)]], ViewParams_0 constant* view_65 [[buffer(16)]], packed_float4 device* objects_65 [[buffer(18)]], uint device* shot_bodies_65 [[buffer(20)]], float device* lines_65 [[buffer(19)]], packed_float4 device* geometry_65 [[buffer(21)]], packed_float4 device* shots_65 [[buffer(22)]], packed_float4 device* lp_65 [[buffer(24)]], packed_float4 device* ext_67 [[buffer(23)]], packed_float4 device* engine_contacts_65 [[buffer(25)]], CrushParams_0 constant* crush_65 [[buffer(28)]], packed_float4 device* crush_rw_65 [[buffer(27)]], packed_float4 device* crush_tab_65 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_189;
    (&kernelContext_189)->params_0 = params_65;
    (&kernelContext_189)->atom_0 = atom_65;
    (&kernelContext_189)->work_0 = work_65;
    (&kernelContext_189)->frames_0 = frames_65;
    (&kernelContext_189)->bodies_0 = bodies_65;
    (&kernelContext_189)->chunks_0 = chunks_65;
    (&kernelContext_189)->sort_0 = sort_65;
    (&kernelContext_189)->manifolds_0 = manifolds_65;
    (&kernelContext_189)->loads_0 = loads_65;
    (&kernelContext_189)->bonds_0 = bonds_65;
    (&kernelContext_189)->args_0 = args_65;
    (&kernelContext_189)->info_0 = info_65;
    (&kernelContext_189)->scan_0 = scan_65;
    (&kernelContext_189)->parts_0 = parts_65;
    (&kernelContext_189)->colliders_0 = colliders_65;
    (&kernelContext_189)->collider_info_0 = collider_info_65;
    (&kernelContext_189)->view_chunks_0 = view_chunks_65;
    (&kernelContext_189)->view_0 = view_65;
    (&kernelContext_189)->objects_0 = objects_65;
    (&kernelContext_189)->shot_bodies_0 = shot_bodies_65;
    (&kernelContext_189)->lines_0 = lines_65;
    (&kernelContext_189)->geometry_0 = geometry_65;
    (&kernelContext_189)->shots_0 = shots_65;
    (&kernelContext_189)->lp_0 = lp_65;
    (&kernelContext_189)->ext_0 = ext_67;
    (&kernelContext_189)->engine_contacts_0 = engine_contacts_65;
    (&kernelContext_189)->crush_0 = crush_65;
    (&kernelContext_189)->crush_rw_0 = crush_rw_65;
    (&kernelContext_189)->crush_tab_0 = crush_tab_65;
    threadgroup array<uint, int(1024)> g_scan_65;
    (&kernelContext_189)->g_scan_0 = &g_scan_65;
    threadgroup array<float, int(64)> g_il_best_65;
    (&kernelContext_189)->g_il_best_0 = &g_il_best_65;
    cs_pick_2(id_60.x, &kernelContext_189);
    return;
}

[[kernel]] void fr_cs_pick_c(uint3 id_61 [[thread_position_in_grid]], FractureParams_0 constant* params_66 [[buffer(0)]], atomic<uint> device* atom_66 [[buffer(5)]], uint device* work_66 [[buffer(4)]], packed_float4 device* frames_66 [[buffer(6)]], packed_float4 device* bodies_66 [[buffer(7)]], packed_float4 device* chunks_66 [[buffer(1)]], uint device* sort_66 [[buffer(11)]], Record_natural_0 device* manifolds_66 [[buffer(10)]], packed_float4 device* loads_66 [[buffer(12)]], packed_uint4 device* bonds_66 [[buffer(2)]], uint device* args_66 [[buffer(13)]], uint device* info_66 [[buffer(15)]], ScanParams_0 constant* scan_66 [[buffer(14)]], packed_float4 device* parts_66 [[buffer(3)]], packed_float4 device* colliders_66 [[buffer(8)]], packed_uint4 device* collider_info_66 [[buffer(9)]], packed_float4 device* view_chunks_66 [[buffer(17)]], ViewParams_0 constant* view_66 [[buffer(16)]], packed_float4 device* objects_66 [[buffer(18)]], uint device* shot_bodies_66 [[buffer(20)]], float device* lines_66 [[buffer(19)]], packed_float4 device* geometry_66 [[buffer(21)]], packed_float4 device* shots_66 [[buffer(22)]], packed_float4 device* lp_66 [[buffer(24)]], packed_float4 device* ext_68 [[buffer(23)]], packed_float4 device* engine_contacts_66 [[buffer(25)]], CrushParams_0 constant* crush_66 [[buffer(28)]], packed_float4 device* crush_rw_66 [[buffer(27)]], packed_float4 device* crush_tab_66 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_190;
    (&kernelContext_190)->params_0 = params_66;
    (&kernelContext_190)->atom_0 = atom_66;
    (&kernelContext_190)->work_0 = work_66;
    (&kernelContext_190)->frames_0 = frames_66;
    (&kernelContext_190)->bodies_0 = bodies_66;
    (&kernelContext_190)->chunks_0 = chunks_66;
    (&kernelContext_190)->sort_0 = sort_66;
    (&kernelContext_190)->manifolds_0 = manifolds_66;
    (&kernelContext_190)->loads_0 = loads_66;
    (&kernelContext_190)->bonds_0 = bonds_66;
    (&kernelContext_190)->args_0 = args_66;
    (&kernelContext_190)->info_0 = info_66;
    (&kernelContext_190)->scan_0 = scan_66;
    (&kernelContext_190)->parts_0 = parts_66;
    (&kernelContext_190)->colliders_0 = colliders_66;
    (&kernelContext_190)->collider_info_0 = collider_info_66;
    (&kernelContext_190)->view_chunks_0 = view_chunks_66;
    (&kernelContext_190)->view_0 = view_66;
    (&kernelContext_190)->objects_0 = objects_66;
    (&kernelContext_190)->shot_bodies_0 = shot_bodies_66;
    (&kernelContext_190)->lines_0 = lines_66;
    (&kernelContext_190)->geometry_0 = geometry_66;
    (&kernelContext_190)->shots_0 = shots_66;
    (&kernelContext_190)->lp_0 = lp_66;
    (&kernelContext_190)->ext_0 = ext_68;
    (&kernelContext_190)->engine_contacts_0 = engine_contacts_66;
    (&kernelContext_190)->crush_0 = crush_66;
    (&kernelContext_190)->crush_rw_0 = crush_rw_66;
    (&kernelContext_190)->crush_tab_0 = crush_tab_66;
    threadgroup array<uint, int(1024)> g_scan_66;
    (&kernelContext_190)->g_scan_0 = &g_scan_66;
    threadgroup array<float, int(64)> g_il_best_66;
    (&kernelContext_190)->g_il_best_0 = &g_il_best_66;
    cs_pick_1(id_61.x, &kernelContext_190);
    return;
}

[[kernel]] void fr_cs_pick_d(uint3 id_62 [[thread_position_in_grid]], FractureParams_0 constant* params_67 [[buffer(0)]], atomic<uint> device* atom_67 [[buffer(5)]], uint device* work_67 [[buffer(4)]], packed_float4 device* frames_67 [[buffer(6)]], packed_float4 device* bodies_67 [[buffer(7)]], packed_float4 device* chunks_67 [[buffer(1)]], uint device* sort_67 [[buffer(11)]], Record_natural_0 device* manifolds_67 [[buffer(10)]], packed_float4 device* loads_67 [[buffer(12)]], packed_uint4 device* bonds_67 [[buffer(2)]], uint device* args_67 [[buffer(13)]], uint device* info_67 [[buffer(15)]], ScanParams_0 constant* scan_67 [[buffer(14)]], packed_float4 device* parts_67 [[buffer(3)]], packed_float4 device* colliders_67 [[buffer(8)]], packed_uint4 device* collider_info_67 [[buffer(9)]], packed_float4 device* view_chunks_67 [[buffer(17)]], ViewParams_0 constant* view_67 [[buffer(16)]], packed_float4 device* objects_67 [[buffer(18)]], uint device* shot_bodies_67 [[buffer(20)]], float device* lines_67 [[buffer(19)]], packed_float4 device* geometry_67 [[buffer(21)]], packed_float4 device* shots_67 [[buffer(22)]], packed_float4 device* lp_67 [[buffer(24)]], packed_float4 device* ext_69 [[buffer(23)]], packed_float4 device* engine_contacts_67 [[buffer(25)]], CrushParams_0 constant* crush_67 [[buffer(28)]], packed_float4 device* crush_rw_67 [[buffer(27)]], packed_float4 device* crush_tab_67 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_191;
    (&kernelContext_191)->params_0 = params_67;
    (&kernelContext_191)->atom_0 = atom_67;
    (&kernelContext_191)->work_0 = work_67;
    (&kernelContext_191)->frames_0 = frames_67;
    (&kernelContext_191)->bodies_0 = bodies_67;
    (&kernelContext_191)->chunks_0 = chunks_67;
    (&kernelContext_191)->sort_0 = sort_67;
    (&kernelContext_191)->manifolds_0 = manifolds_67;
    (&kernelContext_191)->loads_0 = loads_67;
    (&kernelContext_191)->bonds_0 = bonds_67;
    (&kernelContext_191)->args_0 = args_67;
    (&kernelContext_191)->info_0 = info_67;
    (&kernelContext_191)->scan_0 = scan_67;
    (&kernelContext_191)->parts_0 = parts_67;
    (&kernelContext_191)->colliders_0 = colliders_67;
    (&kernelContext_191)->collider_info_0 = collider_info_67;
    (&kernelContext_191)->view_chunks_0 = view_chunks_67;
    (&kernelContext_191)->view_0 = view_67;
    (&kernelContext_191)->objects_0 = objects_67;
    (&kernelContext_191)->shot_bodies_0 = shot_bodies_67;
    (&kernelContext_191)->lines_0 = lines_67;
    (&kernelContext_191)->geometry_0 = geometry_67;
    (&kernelContext_191)->shots_0 = shots_67;
    (&kernelContext_191)->lp_0 = lp_67;
    (&kernelContext_191)->ext_0 = ext_69;
    (&kernelContext_191)->engine_contacts_0 = engine_contacts_67;
    (&kernelContext_191)->crush_0 = crush_67;
    (&kernelContext_191)->crush_rw_0 = crush_rw_67;
    (&kernelContext_191)->crush_tab_0 = crush_tab_67;
    threadgroup array<uint, int(1024)> g_scan_67;
    (&kernelContext_191)->g_scan_0 = &g_scan_67;
    threadgroup array<float, int(64)> g_il_best_67;
    (&kernelContext_191)->g_il_best_0 = &g_il_best_67;
    cs_pick_0(id_62.x, &kernelContext_191);
    return;
}

[[kernel]] void fr_cs_record(uint3 id_63 [[thread_position_in_grid]], FractureParams_0 constant* params_68 [[buffer(0)]], atomic<uint> device* atom_68 [[buffer(5)]], uint device* work_68 [[buffer(4)]], packed_float4 device* frames_68 [[buffer(6)]], packed_float4 device* bodies_68 [[buffer(7)]], packed_float4 device* chunks_68 [[buffer(1)]], uint device* sort_68 [[buffer(11)]], Record_natural_0 device* manifolds_68 [[buffer(10)]], packed_float4 device* loads_68 [[buffer(12)]], packed_uint4 device* bonds_68 [[buffer(2)]], uint device* args_68 [[buffer(13)]], uint device* info_68 [[buffer(15)]], ScanParams_0 constant* scan_68 [[buffer(14)]], packed_float4 device* parts_68 [[buffer(3)]], packed_float4 device* colliders_68 [[buffer(8)]], packed_uint4 device* collider_info_68 [[buffer(9)]], packed_float4 device* view_chunks_68 [[buffer(17)]], ViewParams_0 constant* view_68 [[buffer(16)]], packed_float4 device* objects_68 [[buffer(18)]], uint device* shot_bodies_68 [[buffer(20)]], float device* lines_68 [[buffer(19)]], packed_float4 device* geometry_68 [[buffer(21)]], packed_float4 device* shots_68 [[buffer(22)]], packed_float4 device* lp_68 [[buffer(24)]], packed_float4 device* ext_70 [[buffer(23)]], packed_float4 device* engine_contacts_68 [[buffer(25)]], CrushParams_0 constant* crush_68 [[buffer(28)]], packed_float4 device* crush_rw_68 [[buffer(27)]], packed_float4 device* crush_tab_68 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_192;
    (&kernelContext_192)->params_0 = params_68;
    (&kernelContext_192)->atom_0 = atom_68;
    (&kernelContext_192)->work_0 = work_68;
    (&kernelContext_192)->frames_0 = frames_68;
    (&kernelContext_192)->bodies_0 = bodies_68;
    (&kernelContext_192)->chunks_0 = chunks_68;
    (&kernelContext_192)->sort_0 = sort_68;
    (&kernelContext_192)->manifolds_0 = manifolds_68;
    (&kernelContext_192)->loads_0 = loads_68;
    (&kernelContext_192)->bonds_0 = bonds_68;
    (&kernelContext_192)->args_0 = args_68;
    (&kernelContext_192)->info_0 = info_68;
    (&kernelContext_192)->scan_0 = scan_68;
    (&kernelContext_192)->parts_0 = parts_68;
    (&kernelContext_192)->colliders_0 = colliders_68;
    (&kernelContext_192)->collider_info_0 = collider_info_68;
    (&kernelContext_192)->view_chunks_0 = view_chunks_68;
    (&kernelContext_192)->view_0 = view_68;
    (&kernelContext_192)->objects_0 = objects_68;
    (&kernelContext_192)->shot_bodies_0 = shot_bodies_68;
    (&kernelContext_192)->lines_0 = lines_68;
    (&kernelContext_192)->geometry_0 = geometry_68;
    (&kernelContext_192)->shots_0 = shots_68;
    (&kernelContext_192)->lp_0 = lp_68;
    (&kernelContext_192)->ext_0 = ext_70;
    (&kernelContext_192)->engine_contacts_0 = engine_contacts_68;
    (&kernelContext_192)->crush_0 = crush_68;
    (&kernelContext_192)->crush_rw_0 = crush_rw_68;
    (&kernelContext_192)->crush_tab_0 = crush_tab_68;
    threadgroup array<uint, int(1024)> g_scan_68;
    (&kernelContext_192)->g_scan_0 = &g_scan_68;
    threadgroup array<float, int(64)> g_il_best_68;
    (&kernelContext_192)->g_il_best_0 = &g_il_best_68;
    uint _S1246 = id_63.x;
    uint _S1247 = a_counters_1(&kernelContext_192);
    uint _S1248 = atomic_load_explicit((&kernelContext_192)->atom_0+(_S1247 + 18U), memory_order_relaxed);
    if(_S1246 >= _S1248)
    {
        return;
    }
    uint _S1249 = w_cs_bodies_0(&kernelContext_192);
    uint device* _S1250 = (&kernelContext_192)->work_0+(_S1249 + _S1246);
    uint s_18 = *_S1250;
    uint _S1251 = a_cs_0(*_S1250, &kernelContext_192);
    uint _S1252 = atomic_load_explicit((&kernelContext_192)->atom_0+(_S1251 + 11U), memory_order_relaxed);
    uint key_2 = _S1252 - 1U;
    thread uint k_47;
    thread uint s2_0;
    thread uint ob_6;
    thread uint co_6;
    thread float3 p_19;
    thread float3 u_10;
    thread float v_15;
    thread float m_eff_8;
    thread float t_in_9;
    bool _S1253 = cs_side_0(key_2 / 2U, key_2 % 2U, &k_47, &s2_0, &ob_6, &co_6, &p_19, &u_10, &v_15, &m_eff_8, &t_in_9, &kernelContext_192);
    if(!_S1253)
    {
        return;
    }
    uint4 _S1254 = uint4(*((&kernelContext_192)->collider_info_0+(2U * co_6 + 1U))) ;
    float e_o_0 = (as_type<float>((_S1254.y)));
    float _S1255 = max((as_type<float>((_S1254.z))), 0.00100000004749745f);
    float j_9 = m_eff_8 * v_15;
    float tau_2 = hertz_duration_0(m_eff_8, _S1255, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / (float4(*((&kernelContext_192)->geometry_0+(3U * k_47 + 1U))) ).w), v_15);
    uint _S1256 = lp_body_0(s_18, &kernelContext_192);
    *((&kernelContext_192)->lp_0+(_S1256 + 2U)) = packed_float4(float4(from_rubble_0(p_19), 3.14159274101257324f * j_9 / (2.0f * tau_2))) ;
    *((&kernelContext_192)->lp_0+(_S1256 + 3U)) = packed_float4(float4(from_rubble_0(u_10), v_15)) ;
    *((&kernelContext_192)->lp_0+(_S1256 + 4U)) = packed_float4(float4(m_eff_8, _S1255, e_o_0, (as_type<float>((ob_6))))) ;
    *((&kernelContext_192)->lp_0+(_S1256 + 5U)) = packed_float4(float4((as_type<float>((k_47))), j_9 / tau_2, 1.0f / (float4(*((&kernelContext_192)->bodies_0+7U * ob_6)) ).w, t_in_9)) ;
    *((&kernelContext_192)->lp_0+(_S1256 + 6U)) = packed_float4(float4((as_type<float>((co_6))), (as_type<float>((_S1254.w))), 0.0f, 0.0f)) ;
    atomic_store_explicit((&kernelContext_192)->atom_0+(_S1251 + 1U), k_47 + 1U, memory_order_relaxed);
    uint _S1257 = atomic_fetch_add_explicit((&kernelContext_192)->atom_0+(_S1247 + 16U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_hertz(uint3 id_64 [[thread_position_in_grid]], FractureParams_0 constant* params_69 [[buffer(0)]], atomic<uint> device* atom_69 [[buffer(5)]], uint device* work_69 [[buffer(4)]], packed_float4 device* frames_69 [[buffer(6)]], packed_float4 device* bodies_69 [[buffer(7)]], packed_float4 device* chunks_69 [[buffer(1)]], uint device* sort_69 [[buffer(11)]], Record_natural_0 device* manifolds_69 [[buffer(10)]], packed_float4 device* loads_69 [[buffer(12)]], packed_uint4 device* bonds_69 [[buffer(2)]], uint device* args_69 [[buffer(13)]], uint device* info_69 [[buffer(15)]], ScanParams_0 constant* scan_69 [[buffer(14)]], packed_float4 device* parts_69 [[buffer(3)]], packed_float4 device* colliders_69 [[buffer(8)]], packed_uint4 device* collider_info_69 [[buffer(9)]], packed_float4 device* view_chunks_69 [[buffer(17)]], ViewParams_0 constant* view_69 [[buffer(16)]], packed_float4 device* objects_69 [[buffer(18)]], uint device* shot_bodies_69 [[buffer(20)]], float device* lines_69 [[buffer(19)]], packed_float4 device* geometry_69 [[buffer(21)]], packed_float4 device* shots_69 [[buffer(22)]], packed_float4 device* lp_69 [[buffer(24)]], packed_float4 device* ext_71 [[buffer(23)]], packed_float4 device* engine_contacts_69 [[buffer(25)]], CrushParams_0 constant* crush_69 [[buffer(28)]], packed_float4 device* crush_rw_69 [[buffer(27)]], packed_float4 device* crush_tab_69 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_193;
    (&kernelContext_193)->params_0 = params_69;
    (&kernelContext_193)->atom_0 = atom_69;
    (&kernelContext_193)->work_0 = work_69;
    (&kernelContext_193)->frames_0 = frames_69;
    (&kernelContext_193)->bodies_0 = bodies_69;
    (&kernelContext_193)->chunks_0 = chunks_69;
    (&kernelContext_193)->sort_0 = sort_69;
    (&kernelContext_193)->manifolds_0 = manifolds_69;
    (&kernelContext_193)->loads_0 = loads_69;
    (&kernelContext_193)->bonds_0 = bonds_69;
    (&kernelContext_193)->args_0 = args_69;
    (&kernelContext_193)->info_0 = info_69;
    (&kernelContext_193)->scan_0 = scan_69;
    (&kernelContext_193)->parts_0 = parts_69;
    (&kernelContext_193)->colliders_0 = colliders_69;
    (&kernelContext_193)->collider_info_0 = collider_info_69;
    (&kernelContext_193)->view_chunks_0 = view_chunks_69;
    (&kernelContext_193)->view_0 = view_69;
    (&kernelContext_193)->objects_0 = objects_69;
    (&kernelContext_193)->shot_bodies_0 = shot_bodies_69;
    (&kernelContext_193)->lines_0 = lines_69;
    (&kernelContext_193)->geometry_0 = geometry_69;
    (&kernelContext_193)->shots_0 = shots_69;
    (&kernelContext_193)->lp_0 = lp_69;
    (&kernelContext_193)->ext_0 = ext_71;
    (&kernelContext_193)->engine_contacts_0 = engine_contacts_69;
    (&kernelContext_193)->crush_0 = crush_69;
    (&kernelContext_193)->crush_rw_0 = crush_rw_69;
    (&kernelContext_193)->crush_tab_0 = crush_tab_69;
    threadgroup array<uint, int(1024)> g_scan_69;
    (&kernelContext_193)->g_scan_0 = &g_scan_69;
    threadgroup array<float, int(64)> g_il_best_69;
    (&kernelContext_193)->g_il_best_0 = &g_il_best_69;
    uint i_21 = id_64.x;
    bool _S1258;
    if(i_21 >= (params_69->bond_count_0))
    {
        _S1258 = true;
    }
    else
    {
        uint _S1259 = w_alive_0(&kernelContext_193);
        _S1258 = (*((&kernelContext_193)->work_0+(_S1259 + i_21))) == 0U;
    }
    if(_S1258)
    {
        return;
    }
    uint4 _S1260 = uint4(*((&kernelContext_193)->bonds_0+i_21)) ;
    uint _S1261 = body_of_0(_S1260.x, &kernelContext_193);
    bool _S1262 = cs_struck_1(_S1261, &kernelContext_193);
    if(!_S1262)
    {
        return;
    }
    uint _S1263 = lp_body_0(_S1261, &kernelContext_193);
    bool _S1264 = strike_share_breaks_0(_S1260, (float4(*((&kernelContext_193)->lp_0+(_S1263 + 2U))) ).xyz, (float4(*((&kernelContext_193)->lp_0+(_S1263 + 5U))) ).y, &kernelContext_193);
    if(_S1264)
    {
        uint _S1265 = w_alive_0(&kernelContext_193);
        *((&kernelContext_193)->work_0+(_S1265 + i_21)) = 0U;
        uint _S1266 = a_dirty_0(&kernelContext_193);
        uint _S1267 = atomic_fetch_or_explicit((&kernelContext_193)->atom_0+(_S1266 + _S1261), 1U, memory_order_relaxed);
        uint _S1268 = a_counters_1(&kernelContext_193);
        uint _S1269 = atomic_fetch_add_explicit((&kernelContext_193)->atom_0+_S1268, 1U, memory_order_relaxed);
        uint _S1270 = atomic_fetch_add_explicit((&kernelContext_193)->atom_0+(_S1268 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_21, &kernelContext_193);
        uint _S1271 = atomic_fetch_add_explicit((&kernelContext_193)->atom_0+(_S1268 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_gate2(uint3 id_65 [[thread_position_in_grid]], FractureParams_0 constant* params_70 [[buffer(0)]], atomic<uint> device* atom_70 [[buffer(5)]], uint device* work_70 [[buffer(4)]], packed_float4 device* frames_70 [[buffer(6)]], packed_float4 device* bodies_70 [[buffer(7)]], packed_float4 device* chunks_70 [[buffer(1)]], uint device* sort_70 [[buffer(11)]], Record_natural_0 device* manifolds_70 [[buffer(10)]], packed_float4 device* loads_70 [[buffer(12)]], packed_uint4 device* bonds_70 [[buffer(2)]], uint device* args_70 [[buffer(13)]], uint device* info_70 [[buffer(15)]], ScanParams_0 constant* scan_70 [[buffer(14)]], packed_float4 device* parts_70 [[buffer(3)]], packed_float4 device* colliders_70 [[buffer(8)]], packed_uint4 device* collider_info_70 [[buffer(9)]], packed_float4 device* view_chunks_70 [[buffer(17)]], ViewParams_0 constant* view_70 [[buffer(16)]], packed_float4 device* objects_70 [[buffer(18)]], uint device* shot_bodies_70 [[buffer(20)]], float device* lines_70 [[buffer(19)]], packed_float4 device* geometry_70 [[buffer(21)]], packed_float4 device* shots_70 [[buffer(22)]], packed_float4 device* lp_70 [[buffer(24)]], packed_float4 device* ext_72 [[buffer(23)]], packed_float4 device* engine_contacts_70 [[buffer(25)]], CrushParams_0 constant* crush_70 [[buffer(28)]], packed_float4 device* crush_rw_70 [[buffer(27)]], packed_float4 device* crush_tab_70 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_194;
    (&kernelContext_194)->params_0 = params_70;
    (&kernelContext_194)->atom_0 = atom_70;
    (&kernelContext_194)->work_0 = work_70;
    (&kernelContext_194)->frames_0 = frames_70;
    (&kernelContext_194)->bodies_0 = bodies_70;
    (&kernelContext_194)->chunks_0 = chunks_70;
    (&kernelContext_194)->sort_0 = sort_70;
    (&kernelContext_194)->manifolds_0 = manifolds_70;
    (&kernelContext_194)->loads_0 = loads_70;
    (&kernelContext_194)->bonds_0 = bonds_70;
    (&kernelContext_194)->args_0 = args_70;
    (&kernelContext_194)->info_0 = info_70;
    (&kernelContext_194)->scan_0 = scan_70;
    (&kernelContext_194)->parts_0 = parts_70;
    (&kernelContext_194)->colliders_0 = colliders_70;
    (&kernelContext_194)->collider_info_0 = collider_info_70;
    (&kernelContext_194)->view_chunks_0 = view_chunks_70;
    (&kernelContext_194)->view_0 = view_70;
    (&kernelContext_194)->objects_0 = objects_70;
    (&kernelContext_194)->shot_bodies_0 = shot_bodies_70;
    (&kernelContext_194)->lines_0 = lines_70;
    (&kernelContext_194)->geometry_0 = geometry_70;
    (&kernelContext_194)->shots_0 = shots_70;
    (&kernelContext_194)->lp_0 = lp_70;
    (&kernelContext_194)->ext_0 = ext_72;
    (&kernelContext_194)->engine_contacts_0 = engine_contacts_70;
    (&kernelContext_194)->crush_0 = crush_70;
    (&kernelContext_194)->crush_rw_0 = crush_rw_70;
    (&kernelContext_194)->crush_tab_0 = crush_tab_70;
    threadgroup array<uint, int(1024)> g_scan_70;
    (&kernelContext_194)->g_scan_0 = &g_scan_70;
    threadgroup array<float, int(64)> g_il_best_70;
    (&kernelContext_194)->g_il_best_0 = &g_il_best_70;
    uint _S1272 = a_counters_1(&kernelContext_194);
    uint _S1273 = atomic_load_explicit((&kernelContext_194)->atom_0+_S1272, memory_order_relaxed);
    uint _S1274;
    if(_S1273 > 0U)
    {
        uint _S1275 = cs_rows_0(&kernelContext_194);
        uint _S1276 = groups_of_0(_S1275, 64U);
        _S1274 = _S1276;
    }
    else
    {
        _S1274 = 0U;
    }
    set_args_0(13U, _S1274, &kernelContext_194);
    return;
}

uint lp_anchor_0(uint r_25, uint side_7, KernelContext_0 thread* kernelContext_195)
{
    return 22U * kernelContext_195->params_0->chunk_count_0 + 7U * kernelContext_195->params_0->body_count_0 + kernelContext_195->params_0->chunk_count_0 + 2U * r_25 + side_7;
}

[[kernel]] void fr_cs_save(uint3 id_66 [[thread_position_in_grid]], FractureParams_0 constant* params_71 [[buffer(0)]], atomic<uint> device* atom_71 [[buffer(5)]], uint device* work_71 [[buffer(4)]], packed_float4 device* frames_71 [[buffer(6)]], packed_float4 device* bodies_71 [[buffer(7)]], packed_float4 device* chunks_71 [[buffer(1)]], uint device* sort_71 [[buffer(11)]], Record_natural_0 device* manifolds_71 [[buffer(10)]], packed_float4 device* loads_71 [[buffer(12)]], packed_uint4 device* bonds_71 [[buffer(2)]], uint device* args_71 [[buffer(13)]], uint device* info_71 [[buffer(15)]], ScanParams_0 constant* scan_71 [[buffer(14)]], packed_float4 device* parts_71 [[buffer(3)]], packed_float4 device* colliders_71 [[buffer(8)]], packed_uint4 device* collider_info_71 [[buffer(9)]], packed_float4 device* view_chunks_71 [[buffer(17)]], ViewParams_0 constant* view_71 [[buffer(16)]], packed_float4 device* objects_71 [[buffer(18)]], uint device* shot_bodies_71 [[buffer(20)]], float device* lines_71 [[buffer(19)]], packed_float4 device* geometry_71 [[buffer(21)]], packed_float4 device* shots_71 [[buffer(22)]], packed_float4 device* lp_71 [[buffer(24)]], packed_float4 device* ext_73 [[buffer(23)]], packed_float4 device* engine_contacts_71 [[buffer(25)]], CrushParams_0 constant* crush_71 [[buffer(28)]], packed_float4 device* crush_rw_71 [[buffer(27)]], packed_float4 device* crush_tab_71 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_196;
    (&kernelContext_196)->params_0 = params_71;
    (&kernelContext_196)->atom_0 = atom_71;
    (&kernelContext_196)->work_0 = work_71;
    (&kernelContext_196)->frames_0 = frames_71;
    (&kernelContext_196)->bodies_0 = bodies_71;
    (&kernelContext_196)->chunks_0 = chunks_71;
    (&kernelContext_196)->sort_0 = sort_71;
    (&kernelContext_196)->manifolds_0 = manifolds_71;
    (&kernelContext_196)->loads_0 = loads_71;
    (&kernelContext_196)->bonds_0 = bonds_71;
    (&kernelContext_196)->args_0 = args_71;
    (&kernelContext_196)->info_0 = info_71;
    (&kernelContext_196)->scan_0 = scan_71;
    (&kernelContext_196)->parts_0 = parts_71;
    (&kernelContext_196)->colliders_0 = colliders_71;
    (&kernelContext_196)->collider_info_0 = collider_info_71;
    (&kernelContext_196)->view_chunks_0 = view_chunks_71;
    (&kernelContext_196)->view_0 = view_71;
    (&kernelContext_196)->objects_0 = objects_71;
    (&kernelContext_196)->shot_bodies_0 = shot_bodies_71;
    (&kernelContext_196)->lines_0 = lines_71;
    (&kernelContext_196)->geometry_0 = geometry_71;
    (&kernelContext_196)->shots_0 = shots_71;
    (&kernelContext_196)->lp_0 = lp_71;
    (&kernelContext_196)->ext_0 = ext_73;
    (&kernelContext_196)->engine_contacts_0 = engine_contacts_71;
    (&kernelContext_196)->crush_0 = crush_71;
    (&kernelContext_196)->crush_rw_0 = crush_rw_71;
    (&kernelContext_196)->crush_tab_0 = crush_tab_71;
    threadgroup array<uint, int(1024)> g_scan_71;
    (&kernelContext_196)->g_scan_0 = &g_scan_71;
    threadgroup array<float, int(64)> g_il_best_71;
    (&kernelContext_196)->g_il_best_0 = &g_il_best_71;
    uint r_26 = id_66.x;
    uint _S1277 = cs_rows_0(&kernelContext_196);
    if(r_26 >= _S1277)
    {
        return;
    }
    uint _S1278 = cs_row_0(r_26, &kernelContext_196);
    uint4 _S1279 = (as_type<uint4>((float4(*((&kernelContext_196)->engine_contacts_0+_S1278)) )));
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
        float4 _S1280 = float4(0.0f) ;
        bool _S1281 = side_8 == 0U;
        uint col_1;
        if(_S1281)
        {
            col_1 = _S1279.x;
        }
        else
        {
            col_1 = _S1279.y;
        }
        uint k_48;
        if((_S1279.w) != 0U)
        {
            k_48 = (uint4(*((&kernelContext_196)->collider_info_0+2U * col_1)) ).x;
        }
        else
        {
            k_48 = 4294967295U;
        }
        bool _S1282;
        if(k_48 != 4294967295U)
        {
            _S1282 = k_48 < ((&kernelContext_196)->params_0->chunk_count_0);
        }
        else
        {
            _S1282 = false;
        }
        bool _S1283;
        if(_S1282)
        {
            uint _S1284 = body_of_0(k_48, &kernelContext_196);
            bool _S1285 = dirty_0(_S1284, &kernelContext_196);
            _S1283 = _S1285;
        }
        else
        {
            _S1283 = false;
        }
        float4 saved_0;
        if(_S1283)
        {
            uint _S1286 = body_of_0(k_48, &kernelContext_196);
            uint at_21 = 7U * _S1286;
            float3 anchor_0;
            if(_S1281)
            {
                anchor_0 = (float4(*((&kernelContext_196)->engine_contacts_0+(_S1278 + 2U))) ).xyz;
            }
            else
            {
                anchor_0 = (float4(*((&kernelContext_196)->engine_contacts_0+(_S1278 + 3U))) ).xyz;
            }
            float3 _S1287 = (float4(*((&kernelContext_196)->bodies_0+at_21)) ).xyz;
            thread Quat_0 _S1288 = quat_of_0(float4(*((&kernelContext_196)->bodies_0+(at_21 + 1U))) );
            float3 _S1289 = rotate_0(&_S1288, anchor_0);
            saved_0 = float4(_S1287 + _S1289, 1.0f);
        }
        else
        {
            saved_0 = _S1280;
        }
        uint _S1290 = lp_anchor_0(r_26, side_8, &kernelContext_196);
        *((&kernelContext_196)->lp_0+_S1290) = packed_float4(saved_0) ;
        side_8 = side_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_clean(uint3 id_67 [[thread_position_in_grid]], FractureParams_0 constant* params_72 [[buffer(0)]], atomic<uint> device* atom_72 [[buffer(5)]], uint device* work_72 [[buffer(4)]], packed_float4 device* frames_72 [[buffer(6)]], packed_float4 device* bodies_72 [[buffer(7)]], packed_float4 device* chunks_72 [[buffer(1)]], uint device* sort_72 [[buffer(11)]], Record_natural_0 device* manifolds_72 [[buffer(10)]], packed_float4 device* loads_72 [[buffer(12)]], packed_uint4 device* bonds_72 [[buffer(2)]], uint device* args_72 [[buffer(13)]], uint device* info_72 [[buffer(15)]], ScanParams_0 constant* scan_72 [[buffer(14)]], packed_float4 device* parts_72 [[buffer(3)]], packed_float4 device* colliders_72 [[buffer(8)]], packed_uint4 device* collider_info_72 [[buffer(9)]], packed_float4 device* view_chunks_72 [[buffer(17)]], ViewParams_0 constant* view_72 [[buffer(16)]], packed_float4 device* objects_72 [[buffer(18)]], uint device* shot_bodies_72 [[buffer(20)]], float device* lines_72 [[buffer(19)]], packed_float4 device* geometry_72 [[buffer(21)]], packed_float4 device* shots_72 [[buffer(22)]], packed_float4 device* lp_72 [[buffer(24)]], packed_float4 device* ext_74 [[buffer(23)]], packed_float4 device* engine_contacts_72 [[buffer(25)]], CrushParams_0 constant* crush_72 [[buffer(28)]], packed_float4 device* crush_rw_72 [[buffer(27)]], packed_float4 device* crush_tab_72 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_197;
    (&kernelContext_197)->params_0 = params_72;
    (&kernelContext_197)->atom_0 = atom_72;
    (&kernelContext_197)->work_0 = work_72;
    (&kernelContext_197)->frames_0 = frames_72;
    (&kernelContext_197)->bodies_0 = bodies_72;
    (&kernelContext_197)->chunks_0 = chunks_72;
    (&kernelContext_197)->sort_0 = sort_72;
    (&kernelContext_197)->manifolds_0 = manifolds_72;
    (&kernelContext_197)->loads_0 = loads_72;
    (&kernelContext_197)->bonds_0 = bonds_72;
    (&kernelContext_197)->args_0 = args_72;
    (&kernelContext_197)->info_0 = info_72;
    (&kernelContext_197)->scan_0 = scan_72;
    (&kernelContext_197)->parts_0 = parts_72;
    (&kernelContext_197)->colliders_0 = colliders_72;
    (&kernelContext_197)->collider_info_0 = collider_info_72;
    (&kernelContext_197)->view_chunks_0 = view_chunks_72;
    (&kernelContext_197)->view_0 = view_72;
    (&kernelContext_197)->objects_0 = objects_72;
    (&kernelContext_197)->shot_bodies_0 = shot_bodies_72;
    (&kernelContext_197)->lines_0 = lines_72;
    (&kernelContext_197)->geometry_0 = geometry_72;
    (&kernelContext_197)->shots_0 = shots_72;
    (&kernelContext_197)->lp_0 = lp_72;
    (&kernelContext_197)->ext_0 = ext_74;
    (&kernelContext_197)->engine_contacts_0 = engine_contacts_72;
    (&kernelContext_197)->crush_0 = crush_72;
    (&kernelContext_197)->crush_rw_0 = crush_rw_72;
    (&kernelContext_197)->crush_tab_0 = crush_tab_72;
    threadgroup array<uint, int(1024)> g_scan_72;
    (&kernelContext_197)->g_scan_0 = &g_scan_72;
    threadgroup array<float, int(64)> g_il_best_72;
    (&kernelContext_197)->g_il_best_0 = &g_il_best_72;
    uint _S1291 = id_67.x;
    uint _S1292 = a_counters_1(&kernelContext_197);
    uint _S1293 = atomic_load_explicit((&kernelContext_197)->atom_0+(_S1292 + 18U), memory_order_relaxed);
    if(_S1291 >= _S1293)
    {
        return;
    }
    uint _S1294 = w_cs_bodies_0(&kernelContext_197);
    uint _S1295 = a_cs_0(*((&kernelContext_197)->work_0+(_S1294 + _S1291)), &kernelContext_197);
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
        atomic_store_explicit((&kernelContext_197)->atom_0+(_S1295 + w_8), 0U, memory_order_relaxed);
        w_8 = w_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_fix(uint3 id_68 [[thread_position_in_grid]], FractureParams_0 constant* params_73 [[buffer(0)]], atomic<uint> device* atom_73 [[buffer(5)]], uint device* work_73 [[buffer(4)]], packed_float4 device* frames_73 [[buffer(6)]], packed_float4 device* bodies_73 [[buffer(7)]], packed_float4 device* chunks_73 [[buffer(1)]], uint device* sort_73 [[buffer(11)]], Record_natural_0 device* manifolds_73 [[buffer(10)]], packed_float4 device* loads_73 [[buffer(12)]], packed_uint4 device* bonds_73 [[buffer(2)]], uint device* args_73 [[buffer(13)]], uint device* info_73 [[buffer(15)]], ScanParams_0 constant* scan_73 [[buffer(14)]], packed_float4 device* parts_73 [[buffer(3)]], packed_float4 device* colliders_73 [[buffer(8)]], packed_uint4 device* collider_info_73 [[buffer(9)]], packed_float4 device* view_chunks_73 [[buffer(17)]], ViewParams_0 constant* view_73 [[buffer(16)]], packed_float4 device* objects_73 [[buffer(18)]], uint device* shot_bodies_73 [[buffer(20)]], float device* lines_73 [[buffer(19)]], packed_float4 device* geometry_73 [[buffer(21)]], packed_float4 device* shots_73 [[buffer(22)]], packed_float4 device* lp_73 [[buffer(24)]], packed_float4 device* ext_75 [[buffer(23)]], packed_float4 device* engine_contacts_73 [[buffer(25)]], CrushParams_0 constant* crush_73 [[buffer(28)]], packed_float4 device* crush_rw_73 [[buffer(27)]], packed_float4 device* crush_tab_73 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_198;
    (&kernelContext_198)->params_0 = params_73;
    (&kernelContext_198)->atom_0 = atom_73;
    (&kernelContext_198)->work_0 = work_73;
    (&kernelContext_198)->frames_0 = frames_73;
    (&kernelContext_198)->bodies_0 = bodies_73;
    (&kernelContext_198)->chunks_0 = chunks_73;
    (&kernelContext_198)->sort_0 = sort_73;
    (&kernelContext_198)->manifolds_0 = manifolds_73;
    (&kernelContext_198)->loads_0 = loads_73;
    (&kernelContext_198)->bonds_0 = bonds_73;
    (&kernelContext_198)->args_0 = args_73;
    (&kernelContext_198)->info_0 = info_73;
    (&kernelContext_198)->scan_0 = scan_73;
    (&kernelContext_198)->parts_0 = parts_73;
    (&kernelContext_198)->colliders_0 = colliders_73;
    (&kernelContext_198)->collider_info_0 = collider_info_73;
    (&kernelContext_198)->view_chunks_0 = view_chunks_73;
    (&kernelContext_198)->view_0 = view_73;
    (&kernelContext_198)->objects_0 = objects_73;
    (&kernelContext_198)->shot_bodies_0 = shot_bodies_73;
    (&kernelContext_198)->lines_0 = lines_73;
    (&kernelContext_198)->geometry_0 = geometry_73;
    (&kernelContext_198)->shots_0 = shots_73;
    (&kernelContext_198)->lp_0 = lp_73;
    (&kernelContext_198)->ext_0 = ext_75;
    (&kernelContext_198)->engine_contacts_0 = engine_contacts_73;
    (&kernelContext_198)->crush_0 = crush_73;
    (&kernelContext_198)->crush_rw_0 = crush_rw_73;
    (&kernelContext_198)->crush_tab_0 = crush_tab_73;
    threadgroup array<uint, int(1024)> g_scan_73;
    (&kernelContext_198)->g_scan_0 = &g_scan_73;
    threadgroup array<float, int(64)> g_il_best_73;
    (&kernelContext_198)->g_il_best_0 = &g_il_best_73;
    uint r_27 = id_68.x;
    uint _S1296 = cs_rows_0(&kernelContext_198);
    if(r_27 >= _S1296)
    {
        return;
    }
    uint _S1297 = cs_row_0(r_27, &kernelContext_198);
    uint4 _S1298 = (as_type<uint4>((float4(*((&kernelContext_198)->engine_contacts_0+_S1297)) )));
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
        uint _S1299 = lp_anchor_0(r_27, side_9, &kernelContext_198);
        float4 _S1300 = float4(*((&kernelContext_198)->lp_0+_S1299)) ;
        if((_S1300.w) == 0.0f)
        {
            side_9 = side_9 + 1U;
            continue;
        }
        uint col_2;
        if(side_9 == 0U)
        {
            col_2 = _S1298.x;
        }
        else
        {
            col_2 = _S1298.y;
        }
        uint at_22 = 7U * (as_type<uint>(((float4(*((&kernelContext_198)->colliders_0+4U * col_2)) ).x)));
        thread Quat_0 _S1301 = quat_of_0(float4(*((&kernelContext_198)->bodies_0+(at_22 + 1U))) );
        Quat_0 _S1302 = conj_0(&_S1301);
        float3 _S1303 = _S1300.xyz - (float4(*((&kernelContext_198)->bodies_0+at_22)) ).xyz;
        thread Quat_0 _S1304 = _S1302;
        float3 _S1305 = rotate_0(&_S1304, _S1303);
        (*((&kernelContext_198)->engine_contacts_0+(_S1297 + 2U + side_9))).xyz = _S1305;
        moved_0 = true;
        side_9 = side_9 + 1U;
    }
    if(moved_0)
    {
        (*((&kernelContext_198)->engine_contacts_0+(_S1297 + 4U)))[int(3)] = 0.0f;
        (*((&kernelContext_198)->engine_contacts_0+(_S1297 + 5U)))[int(3)] = 0.0f;
        uint _S1306 = _S1297 + 6U;
        *((&kernelContext_198)->engine_contacts_0+_S1306) = packed_float4(float4(1.0f, 1.0f, 0.0f, (float4(*((&kernelContext_198)->engine_contacts_0+_S1306)) ).w)) ;
        (*((&kernelContext_198)->engine_contacts_0+(_S1297 + 7U)))[int(1)] = 0.0f;
    }
    return;
}

[[kernel]] void fr_cs_count(uint3 id_69 [[thread_position_in_grid]], FractureParams_0 constant* params_74 [[buffer(0)]], atomic<uint> device* atom_74 [[buffer(5)]], uint device* work_74 [[buffer(4)]], packed_float4 device* frames_74 [[buffer(6)]], packed_float4 device* bodies_74 [[buffer(7)]], packed_float4 device* chunks_74 [[buffer(1)]], uint device* sort_74 [[buffer(11)]], Record_natural_0 device* manifolds_74 [[buffer(10)]], packed_float4 device* loads_74 [[buffer(12)]], packed_uint4 device* bonds_74 [[buffer(2)]], uint device* args_74 [[buffer(13)]], uint device* info_74 [[buffer(15)]], ScanParams_0 constant* scan_74 [[buffer(14)]], packed_float4 device* parts_74 [[buffer(3)]], packed_float4 device* colliders_74 [[buffer(8)]], packed_uint4 device* collider_info_74 [[buffer(9)]], packed_float4 device* view_chunks_74 [[buffer(17)]], ViewParams_0 constant* view_74 [[buffer(16)]], packed_float4 device* objects_74 [[buffer(18)]], uint device* shot_bodies_74 [[buffer(20)]], float device* lines_74 [[buffer(19)]], packed_float4 device* geometry_74 [[buffer(21)]], packed_float4 device* shots_74 [[buffer(22)]], packed_float4 device* lp_74 [[buffer(24)]], packed_float4 device* ext_76 [[buffer(23)]], packed_float4 device* engine_contacts_74 [[buffer(25)]], CrushParams_0 constant* crush_74 [[buffer(28)]], packed_float4 device* crush_rw_74 [[buffer(27)]], packed_float4 device* crush_tab_74 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_199;
    (&kernelContext_199)->params_0 = params_74;
    (&kernelContext_199)->atom_0 = atom_74;
    (&kernelContext_199)->work_0 = work_74;
    (&kernelContext_199)->frames_0 = frames_74;
    (&kernelContext_199)->bodies_0 = bodies_74;
    (&kernelContext_199)->chunks_0 = chunks_74;
    (&kernelContext_199)->sort_0 = sort_74;
    (&kernelContext_199)->manifolds_0 = manifolds_74;
    (&kernelContext_199)->loads_0 = loads_74;
    (&kernelContext_199)->bonds_0 = bonds_74;
    (&kernelContext_199)->args_0 = args_74;
    (&kernelContext_199)->info_0 = info_74;
    (&kernelContext_199)->scan_0 = scan_74;
    (&kernelContext_199)->parts_0 = parts_74;
    (&kernelContext_199)->colliders_0 = colliders_74;
    (&kernelContext_199)->collider_info_0 = collider_info_74;
    (&kernelContext_199)->view_chunks_0 = view_chunks_74;
    (&kernelContext_199)->view_0 = view_74;
    (&kernelContext_199)->objects_0 = objects_74;
    (&kernelContext_199)->shot_bodies_0 = shot_bodies_74;
    (&kernelContext_199)->lines_0 = lines_74;
    (&kernelContext_199)->geometry_0 = geometry_74;
    (&kernelContext_199)->shots_0 = shots_74;
    (&kernelContext_199)->lp_0 = lp_74;
    (&kernelContext_199)->ext_0 = ext_76;
    (&kernelContext_199)->engine_contacts_0 = engine_contacts_74;
    (&kernelContext_199)->crush_0 = crush_74;
    (&kernelContext_199)->crush_rw_0 = crush_rw_74;
    (&kernelContext_199)->crush_tab_0 = crush_tab_74;
    threadgroup array<uint, int(1024)> g_scan_74;
    (&kernelContext_199)->g_scan_0 = &g_scan_74;
    threadgroup array<float, int(64)> g_il_best_74;
    (&kernelContext_199)->g_il_best_0 = &g_il_best_74;
    uint k_49 = id_69.x;
    if(k_49 >= (params_74->chunk_count_0))
    {
        return;
    }
    uint _S1307 = body_of_0(k_49, &kernelContext_199);
    bool _S1308 = cs_struck_1(_S1307, &kernelContext_199);
    if(!_S1308)
    {
        return;
    }
    uint _S1309 = a_cs_0(_S1307, &kernelContext_199);
    uint _S1310 = atomic_fetch_max_explicit((&kernelContext_199)->atom_0+(_S1309 + 3U), ~k_49, memory_order_relaxed);
    if(((&kernelContext_199)->info_0[k_49] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_199)->atom_0+(_S1309 + 4U), 1U, memory_order_relaxed);
    }
    uint _S1311 = atomic_fetch_add_explicit((&kernelContext_199)->atom_0+(_S1309 + 5U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_base(uint3 id_70 [[thread_position_in_grid]], FractureParams_0 constant* params_75 [[buffer(0)]], atomic<uint> device* atom_75 [[buffer(5)]], uint device* work_75 [[buffer(4)]], packed_float4 device* frames_75 [[buffer(6)]], packed_float4 device* bodies_75 [[buffer(7)]], packed_float4 device* chunks_75 [[buffer(1)]], uint device* sort_75 [[buffer(11)]], Record_natural_0 device* manifolds_75 [[buffer(10)]], packed_float4 device* loads_75 [[buffer(12)]], packed_uint4 device* bonds_75 [[buffer(2)]], uint device* args_75 [[buffer(13)]], uint device* info_75 [[buffer(15)]], ScanParams_0 constant* scan_75 [[buffer(14)]], packed_float4 device* parts_75 [[buffer(3)]], packed_float4 device* colliders_75 [[buffer(8)]], packed_uint4 device* collider_info_75 [[buffer(9)]], packed_float4 device* view_chunks_75 [[buffer(17)]], ViewParams_0 constant* view_75 [[buffer(16)]], packed_float4 device* objects_75 [[buffer(18)]], uint device* shot_bodies_75 [[buffer(20)]], float device* lines_75 [[buffer(19)]], packed_float4 device* geometry_75 [[buffer(21)]], packed_float4 device* shots_75 [[buffer(22)]], packed_float4 device* lp_75 [[buffer(24)]], packed_float4 device* ext_77 [[buffer(23)]], packed_float4 device* engine_contacts_75 [[buffer(25)]], CrushParams_0 constant* crush_75 [[buffer(28)]], packed_float4 device* crush_rw_75 [[buffer(27)]], packed_float4 device* crush_tab_75 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_200;
    (&kernelContext_200)->params_0 = params_75;
    (&kernelContext_200)->atom_0 = atom_75;
    (&kernelContext_200)->work_0 = work_75;
    (&kernelContext_200)->frames_0 = frames_75;
    (&kernelContext_200)->bodies_0 = bodies_75;
    (&kernelContext_200)->chunks_0 = chunks_75;
    (&kernelContext_200)->sort_0 = sort_75;
    (&kernelContext_200)->manifolds_0 = manifolds_75;
    (&kernelContext_200)->loads_0 = loads_75;
    (&kernelContext_200)->bonds_0 = bonds_75;
    (&kernelContext_200)->args_0 = args_75;
    (&kernelContext_200)->info_0 = info_75;
    (&kernelContext_200)->scan_0 = scan_75;
    (&kernelContext_200)->parts_0 = parts_75;
    (&kernelContext_200)->colliders_0 = colliders_75;
    (&kernelContext_200)->collider_info_0 = collider_info_75;
    (&kernelContext_200)->view_chunks_0 = view_chunks_75;
    (&kernelContext_200)->view_0 = view_75;
    (&kernelContext_200)->objects_0 = objects_75;
    (&kernelContext_200)->shot_bodies_0 = shot_bodies_75;
    (&kernelContext_200)->lines_0 = lines_75;
    (&kernelContext_200)->geometry_0 = geometry_75;
    (&kernelContext_200)->shots_0 = shots_75;
    (&kernelContext_200)->lp_0 = lp_75;
    (&kernelContext_200)->ext_0 = ext_77;
    (&kernelContext_200)->engine_contacts_0 = engine_contacts_75;
    (&kernelContext_200)->crush_0 = crush_75;
    (&kernelContext_200)->crush_rw_0 = crush_rw_75;
    (&kernelContext_200)->crush_tab_0 = crush_tab_75;
    threadgroup array<uint, int(1024)> g_scan_75;
    (&kernelContext_200)->g_scan_0 = &g_scan_75;
    threadgroup array<float, int(64)> g_il_best_75;
    (&kernelContext_200)->g_il_best_0 = &g_il_best_75;
    uint _S1312 = id_70.x;
    uint _S1313 = a_counters_1(&kernelContext_200);
    uint _S1314 = atomic_load_explicit((&kernelContext_200)->atom_0+(_S1313 + 18U), memory_order_relaxed);
    if(_S1312 >= _S1314)
    {
        return;
    }
    uint _S1315 = w_cs_bodies_0(&kernelContext_200);
    uint _S1316 = a_cs_0(*((&kernelContext_200)->work_0+(_S1315 + _S1312)), &kernelContext_200);
    atomic<uint> device* _S1317 = (&kernelContext_200)->atom_0+(_S1316 + 6U);
    atomic<uint> device* _S1318 = (&kernelContext_200)->atom_0+(_S1313 + 19U);
    uint _S1319 = atomic_load_explicit((&kernelContext_200)->atom_0+(_S1316 + 5U), memory_order_relaxed);
    uint _S1320 = atomic_fetch_add_explicit(_S1318, _S1319, memory_order_relaxed);
    atomic_store_explicit(_S1317, _S1320, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_collect(uint3 id_71 [[thread_position_in_grid]], FractureParams_0 constant* params_76 [[buffer(0)]], atomic<uint> device* atom_76 [[buffer(5)]], uint device* work_76 [[buffer(4)]], packed_float4 device* frames_76 [[buffer(6)]], packed_float4 device* bodies_76 [[buffer(7)]], packed_float4 device* chunks_76 [[buffer(1)]], uint device* sort_76 [[buffer(11)]], Record_natural_0 device* manifolds_76 [[buffer(10)]], packed_float4 device* loads_76 [[buffer(12)]], packed_uint4 device* bonds_76 [[buffer(2)]], uint device* args_76 [[buffer(13)]], uint device* info_76 [[buffer(15)]], ScanParams_0 constant* scan_76 [[buffer(14)]], packed_float4 device* parts_76 [[buffer(3)]], packed_float4 device* colliders_76 [[buffer(8)]], packed_uint4 device* collider_info_76 [[buffer(9)]], packed_float4 device* view_chunks_76 [[buffer(17)]], ViewParams_0 constant* view_76 [[buffer(16)]], packed_float4 device* objects_76 [[buffer(18)]], uint device* shot_bodies_76 [[buffer(20)]], float device* lines_76 [[buffer(19)]], packed_float4 device* geometry_76 [[buffer(21)]], packed_float4 device* shots_76 [[buffer(22)]], packed_float4 device* lp_76 [[buffer(24)]], packed_float4 device* ext_78 [[buffer(23)]], packed_float4 device* engine_contacts_76 [[buffer(25)]], CrushParams_0 constant* crush_76 [[buffer(28)]], packed_float4 device* crush_rw_76 [[buffer(27)]], packed_float4 device* crush_tab_76 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_201;
    (&kernelContext_201)->params_0 = params_76;
    (&kernelContext_201)->atom_0 = atom_76;
    (&kernelContext_201)->work_0 = work_76;
    (&kernelContext_201)->frames_0 = frames_76;
    (&kernelContext_201)->bodies_0 = bodies_76;
    (&kernelContext_201)->chunks_0 = chunks_76;
    (&kernelContext_201)->sort_0 = sort_76;
    (&kernelContext_201)->manifolds_0 = manifolds_76;
    (&kernelContext_201)->loads_0 = loads_76;
    (&kernelContext_201)->bonds_0 = bonds_76;
    (&kernelContext_201)->args_0 = args_76;
    (&kernelContext_201)->info_0 = info_76;
    (&kernelContext_201)->scan_0 = scan_76;
    (&kernelContext_201)->parts_0 = parts_76;
    (&kernelContext_201)->colliders_0 = colliders_76;
    (&kernelContext_201)->collider_info_0 = collider_info_76;
    (&kernelContext_201)->view_chunks_0 = view_chunks_76;
    (&kernelContext_201)->view_0 = view_76;
    (&kernelContext_201)->objects_0 = objects_76;
    (&kernelContext_201)->shot_bodies_0 = shot_bodies_76;
    (&kernelContext_201)->lines_0 = lines_76;
    (&kernelContext_201)->geometry_0 = geometry_76;
    (&kernelContext_201)->shots_0 = shots_76;
    (&kernelContext_201)->lp_0 = lp_76;
    (&kernelContext_201)->ext_0 = ext_78;
    (&kernelContext_201)->engine_contacts_0 = engine_contacts_76;
    (&kernelContext_201)->crush_0 = crush_76;
    (&kernelContext_201)->crush_rw_0 = crush_rw_76;
    (&kernelContext_201)->crush_tab_0 = crush_tab_76;
    threadgroup array<uint, int(1024)> g_scan_76;
    (&kernelContext_201)->g_scan_0 = &g_scan_76;
    threadgroup array<float, int(64)> g_il_best_76;
    (&kernelContext_201)->g_il_best_0 = &g_il_best_76;
    uint k_50 = id_71.x;
    if(k_50 >= (params_76->chunk_count_0))
    {
        return;
    }
    uint _S1321 = body_of_0(k_50, &kernelContext_201);
    bool _S1322 = cs_struck_1(_S1321, &kernelContext_201);
    if(!_S1322)
    {
        return;
    }
    uint _S1323 = w_cs_list_0(&kernelContext_201);
    uint _S1324 = a_cs_0(_S1321, &kernelContext_201);
    uint _S1325 = atomic_load_explicit((&kernelContext_201)->atom_0+(_S1324 + 6U), memory_order_relaxed);
    uint _S1326 = _S1323 + _S1325;
    uint _S1327 = atomic_fetch_add_explicit((&kernelContext_201)->atom_0+(_S1324 + 7U), 1U, memory_order_relaxed);
    *((&kernelContext_201)->work_0+(_S1326 + _S1327)) = k_50;
    return;
}

void cs_load_0(uint k_51, uint s_19, KernelContext_0 thread* kernelContext_202)
{
    uint _S1328 = a_cs_0(s_19, kernelContext_202);
    uint _S1329 = atomic_load_explicit(kernelContext_202->atom_0+(_S1328 + 4U), memory_order_relaxed);
    bool supported_1 = _S1329 != 0U;
    bool fixed_2;
    if((kernelContext_202->info_0[k_51] & 1U) != 0U)
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            uint _S1330 = atomic_load_explicit(kernelContext_202->atom_0+(_S1328 + 3U), memory_order_relaxed);
            fixed_2 = (~_S1330) == k_51;
        }
        else
        {
            fixed_2 = false;
        }
    }
    uint _S1331 = lp_body_0(s_19, kernelContext_202);
    float4 _S1332 = float4(*(kernelContext_202->lp_0+(_S1331 + 2U))) ;
    float4 _S1333 = float4(*(kernelContext_202->lp_0+(_S1331 + 3U))) ;
    uint _S1334 = 7U * s_19;
    thread Quat_0 q_27;
    thread float3 origin_8;
    structure_pose_0(s_19, _S1334, &q_27, &origin_8, kernelContext_202);
    thread Quat_0 _S1335 = q_27;
    Quat_0 _S1336 = conj_0(&_S1335);
    float3 _S1337 = _S1333.xyz * float3(_S1332.w) ;
    thread Quat_0 _S1338 = _S1336;
    float3 _S1339 = rotate_0(&_S1338, _S1337);
    float3 _S1340 = _S1332.xyz - origin_8;
    thread Quat_0 _S1341 = _S1336;
    float3 _S1342 = rotate_0(&_S1341, _S1340);
    uint _S1343 = 4U * k_51;
    float4 _S1344 = float4(*(kernelContext_202->chunks_0+_S1343)) ;
    float3 _S1345 = float3(0.0f) ;
    uint _S1346 = atomic_load_explicit(kernelContext_202->atom_0+(_S1328 + 1U), memory_order_relaxed);
    float3 f_12;
    float3 t_13;
    if(_S1346 == (k_51 + 1U))
    {
        float3 _S1347 = cross(_S1342 - _S1344.xyz, _S1339);
        f_12 = _S1339;
        t_13 = _S1347;
    }
    else
    {
        f_12 = _S1345;
        t_13 = _S1345;
    }
    float inv_mass_7 = (float4(*(kernelContext_202->bodies_0+_S1334)) ).w;
    bool _S1348;
    if(!supported_1)
    {
        _S1348 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S1348 = false;
    }
    if(_S1348)
    {
        uint _S1349 = 2U * s_19;
        float3 com_2 = (float4(*(kernelContext_202->frames_0+_S1349)) ).xyz;
        Quat_0 p_20 = quat_of_0(float4(*(kernelContext_202->frames_0+(_S1349 + 1U))) );
        float3 _S1350 = (float4(*(kernelContext_202->bodies_0+(_S1334 + 2U))) ).xyz;
        thread Quat_0 _S1351 = p_20;
        Quat_0 _S1352 = conj_0(&_S1351);
        float3 _S1353 = cross(_S1342 - com_2, _S1339);
        thread Quat_0 _S1354 = _S1352;
        float3 _S1355 = rotate_0(&_S1354, _S1353);
        float3 _S1356 = _S1350 * _S1355;
        thread Quat_0 _S1357 = p_20;
        float3 _S1358 = rotate_0(&_S1357, _S1356);
        float4 _S1359 = float4(*(kernelContext_202->chunks_0+(_S1343 + 1U))) ;
        float4 _S1360 = float4(*(kernelContext_202->chunks_0+(_S1343 + 2U))) ;
        float _S1361 = _S1358.x;
        float _S1362 = _S1359.w;
        float _S1363 = _S1358.y;
        float _S1364 = _S1360.x;
        float _S1365 = _S1358.z;
        float _S1366 = _S1360.y;
        float3 t_14 = t_13 - float3(_S1359.x * _S1361 + _S1362 * _S1363 + _S1364 * _S1365, _S1362 * _S1361 + _S1359.y * _S1363 + _S1366 * _S1365, _S1364 * _S1361 + _S1366 * _S1363 + _S1359.z * _S1365);
        f_12 = f_12 - float3(_S1344.w)  * (_S1339 * float3(inv_mass_7)  + cross(_S1358, _S1344.xyz - com_2));
        t_13 = t_14;
    }
    uint _S1367 = st_f_0(k_51, 14U);
    packed_float4 device* _S1368 = kernelContext_202->lp_0+_S1367;
    float _S1369;
    if(fixed_2)
    {
        _S1369 = 1.0f;
    }
    else
    {
        _S1369 = 0.0f;
    }
    *_S1368 = packed_float4(float4(f_12, _S1369)) ;
    *(kernelContext_202->lp_0+(_S1367 + 1U)) = packed_float4(float4(t_13, 0.0f)) ;
    uint c_24 = 0U;
    for(;;)
    {
        if(c_24 < 2U)
        {
        }
        else
        {
            break;
        }
        uint _S1370 = st_x_0(k_51, 14U, c_24);
        packed_float4 _S1371 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_202->lp_0+_S1370) = _S1371;
        *(kernelContext_202->lp_0+(_S1370 + 1U)) = _S1371;
        c_24 = c_24 + 1U;
    }
    uint _S1372 = st_d_0(k_51, 14U);
    packed_float4 _S1373 = packed_float4(float4(0.0f) ) ;
    *(kernelContext_202->lp_0+_S1372) = _S1373;
    *(kernelContext_202->lp_0+(_S1372 + 1U)) = _S1373;
    return;
}

void cs_break_0(uint i_22, uint s_20, KernelContext_0 thread* kernelContext_203)
{
    uint4 _S1374 = uint4(*(kernelContext_203->bonds_0+i_22)) ;
    LpBond_0 _S1375 = lp_bond_0(i_22, kernelContext_203);
    uint _S1376 = _S1374.x;
    uint _S1377 = _S1374.y;
    thread LpBond_0 _S1378 = _S1375;
    thread float3 q_lin_6;
    thread float3 q_ang_6;
    st_force_0(&_S1378, _S1376, _S1377, 14U, 0U, &q_lin_6, &q_ang_6, kernelContext_203);
    if((kernelContext_203->params_0->load_model_0) == 1U)
    {
        thread LpBond_0 _S1379 = _S1375;
        thread float3 b_lin_0;
        thread float3 b_ang_0;
        lp_force_0(&_S1379, _S1376, _S1377, 0U, &b_lin_0, &b_ang_0, kernelContext_203);
        q_lin_6 = q_lin_6 + b_lin_0;
        q_ang_6 = q_ang_6 + b_ang_0;
    }
    thread JointBond_0 _S1380 = _S1375.law_0;
    Measures_0 _S1381 = stress_measures_0(&_S1380, q_lin_6, q_ang_6);
    JointMaterial_0 _S1382 = lp_material_0(_S1375.law_0.ids_3.x, kernelContext_203);
    float _S1383 = _S1375.law_0.geom1_0.w;
    thread JointMaterial_0 _S1384 = _S1382;
    thread JointBond_0 _S1385 = _S1375.law_0;
    thread Measures_0 _S1386 = _S1381;
    float4 _S1387 = failure_indices_0(&_S1384, &_S1385, &_S1386, _S1383);
    if((max(max(_S1387.x, _S1387.y), max(_S1387.z, _S1387.w))) >= 1.0f)
    {
        uint _S1388 = w_alive_0(kernelContext_203);
        *(kernelContext_203->work_0+(_S1388 + i_22)) = 0U;
        uint _S1389 = a_dirty_0(kernelContext_203);
        uint _S1390 = atomic_fetch_or_explicit(kernelContext_203->atom_0+(_S1389 + s_20), 1U, memory_order_relaxed);
        uint _S1391 = a_counters_1(kernelContext_203);
        uint _S1392 = atomic_fetch_add_explicit(kernelContext_203->atom_0+_S1391, 1U, memory_order_relaxed);
        uint _S1393 = atomic_fetch_add_explicit(kernelContext_203->atom_0+(_S1391 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_22, kernelContext_203);
        uint _S1394 = atomic_fetch_add_explicit(kernelContext_203->atom_0+(_S1391 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_solve(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_77 [[buffer(0)]], atomic<uint> device* atom_77 [[buffer(5)]], uint device* work_77 [[buffer(4)]], packed_float4 device* frames_77 [[buffer(6)]], packed_float4 device* bodies_77 [[buffer(7)]], packed_float4 device* chunks_77 [[buffer(1)]], uint device* sort_77 [[buffer(11)]], Record_natural_0 device* manifolds_77 [[buffer(10)]], packed_float4 device* loads_77 [[buffer(12)]], packed_uint4 device* bonds_77 [[buffer(2)]], uint device* args_77 [[buffer(13)]], uint device* info_77 [[buffer(15)]], ScanParams_0 constant* scan_77 [[buffer(14)]], packed_float4 device* parts_77 [[buffer(3)]], packed_float4 device* colliders_77 [[buffer(8)]], packed_uint4 device* collider_info_77 [[buffer(9)]], packed_float4 device* view_chunks_77 [[buffer(17)]], ViewParams_0 constant* view_77 [[buffer(16)]], packed_float4 device* objects_77 [[buffer(18)]], uint device* shot_bodies_77 [[buffer(20)]], float device* lines_77 [[buffer(19)]], packed_float4 device* geometry_77 [[buffer(21)]], packed_float4 device* shots_77 [[buffer(22)]], packed_float4 device* lp_77 [[buffer(24)]], packed_float4 device* ext_79 [[buffer(23)]], packed_float4 device* engine_contacts_77 [[buffer(25)]], CrushParams_0 constant* crush_77 [[buffer(28)]], packed_float4 device* crush_rw_77 [[buffer(27)]], packed_float4 device* crush_tab_77 [[buffer(26)]])
{
    bool _S1395;
    uint j3_0;
    thread KernelContext_0 kernelContext_204;
    (&kernelContext_204)->params_0 = params_77;
    (&kernelContext_204)->atom_0 = atom_77;
    (&kernelContext_204)->work_0 = work_77;
    (&kernelContext_204)->frames_0 = frames_77;
    (&kernelContext_204)->bodies_0 = bodies_77;
    (&kernelContext_204)->chunks_0 = chunks_77;
    (&kernelContext_204)->sort_0 = sort_77;
    (&kernelContext_204)->manifolds_0 = manifolds_77;
    (&kernelContext_204)->loads_0 = loads_77;
    (&kernelContext_204)->bonds_0 = bonds_77;
    (&kernelContext_204)->args_0 = args_77;
    (&kernelContext_204)->info_0 = info_77;
    (&kernelContext_204)->scan_0 = scan_77;
    (&kernelContext_204)->parts_0 = parts_77;
    (&kernelContext_204)->colliders_0 = colliders_77;
    (&kernelContext_204)->collider_info_0 = collider_info_77;
    (&kernelContext_204)->view_chunks_0 = view_chunks_77;
    (&kernelContext_204)->view_0 = view_77;
    (&kernelContext_204)->objects_0 = objects_77;
    (&kernelContext_204)->shot_bodies_0 = shot_bodies_77;
    (&kernelContext_204)->lines_0 = lines_77;
    (&kernelContext_204)->geometry_0 = geometry_77;
    (&kernelContext_204)->shots_0 = shots_77;
    (&kernelContext_204)->lp_0 = lp_77;
    (&kernelContext_204)->ext_0 = ext_79;
    (&kernelContext_204)->engine_contacts_0 = engine_contacts_77;
    (&kernelContext_204)->crush_0 = crush_77;
    (&kernelContext_204)->crush_rw_0 = crush_rw_77;
    (&kernelContext_204)->crush_tab_0 = crush_tab_77;
    threadgroup array<uint, int(1024)> g_scan_77;
    (&kernelContext_204)->g_scan_0 = &g_scan_77;
    threadgroup array<float, int(64)> g_il_best_77;
    (&kernelContext_204)->g_il_best_0 = &g_il_best_77;
    uint _S1396 = w_cs_bodies_0(&kernelContext_204);
    uint device* _S1397 = (&kernelContext_204)->work_0+(_S1396 + group_3.x);
    uint s_21 = *_S1397;
    uint t_15 = thread_6.x;
    uint _S1398 = a_cs_0(*_S1397, &kernelContext_204);
    uint _S1399 = atomic_load_explicit((&kernelContext_204)->atom_0+(_S1398 + 6U), memory_order_relaxed);
    uint _S1400 = atomic_load_explicit((&kernelContext_204)->atom_0+(_S1398 + 5U), memory_order_relaxed);
    uint j_10 = t_15;
    for(;;)
    {
        if(j_10 < _S1400)
        {
        }
        else
        {
            break;
        }
        uint _S1401 = w_cs_list_0(&kernelContext_204);
        cs_load_0(*((&kernelContext_204)->work_0+(_S1401 + _S1399 + j_10)), s_21, &kernelContext_204);
        j_10 = j_10 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint j2_0 = t_15;
    for(;;)
    {
        if(j2_0 < _S1400)
        {
        }
        else
        {
            break;
        }
        uint _S1402 = w_cs_list_0(&kernelContext_204);
        uint device* _S1403 = (&kernelContext_204)->work_0+(_S1402 + _S1399 + j2_0);
        uint k_52 = *_S1403;
        uint _S1404 = st_f_0(*_S1403, 14U);
        if(((float4(*((&kernelContext_204)->lp_0+_S1404)) ).w) == 0.0f)
        {
            bool _S1405 = lp_factor_0(k_52, &kernelContext_204);
            _S1395 = !_S1405;
        }
        else
        {
            _S1395 = false;
        }
        if(_S1395)
        {
            (*((&kernelContext_204)->lp_0+_S1404))[int(3)] = 1.0f;
            uint _S1406 = a_counters_1(&kernelContext_204);
            uint _S1407 = atomic_fetch_or_explicit((&kernelContext_204)->atom_0+(_S1406 + 3U), 64U, memory_order_relaxed);
        }
        j2_0 = j2_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint sweep_0 = 0U;
    for(;;)
    {
        if(sweep_0 < ((&kernelContext_204)->params_0->cs_sweeps_0))
        {
        }
        else
        {
            break;
        }
        j3_0 = t_15;
        for(;;)
        {
            if(j3_0 < _S1400)
            {
            }
            else
            {
                break;
            }
            uint _S1408 = w_cs_list_0(&kernelContext_204);
            st_sweep_0(*((&kernelContext_204)->work_0+(_S1408 + _S1399 + j3_0)), 14U, sweep_0 % 2U, (&kernelContext_204)->params_0->cs_c1_0, (&kernelContext_204)->params_0->cs_c2_0, &kernelContext_204);
            j3_0 = j3_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        sweep_0 = sweep_0 + 1U;
    }
    j3_0 = t_15;
    for(;;)
    {
        if(j3_0 < _S1400)
        {
        }
        else
        {
            break;
        }
        uint _S1409 = w_cs_list_0(&kernelContext_204);
        uint device* _S1410 = (&kernelContext_204)->work_0+(_S1409 + _S1399 + j3_0);
        uint k_53 = *_S1410;
        float4 _S1411 = float4(*((&kernelContext_204)->ext_0+((&kernelContext_204)->params_0->ext_chunks_0 + 4U * *_S1410))) ;
        uint _S1412 = (as_type<uint>((_S1411.x)));
        uint _S1413 = (as_type<uint>((_S1411.y)));
        uint e_7 = 0U;
        for(;;)
        {
            if(e_7 < _S1413)
            {
            }
            else
            {
                break;
            }
            uint _S1414 = adj_bond_0(_S1412 + e_7, &kernelContext_204);
            if(((uint4(*((&kernelContext_204)->bonds_0+_S1414)) ).x) == k_53)
            {
                uint _S1415 = w_alive_0(&kernelContext_204);
                _S1395 = (*((&kernelContext_204)->work_0+(_S1415 + _S1414))) != 0U;
            }
            else
            {
                _S1395 = false;
            }
            if(_S1395)
            {
                cs_break_0(_S1414, s_21, &kernelContext_204);
            }
            e_7 = e_7 + 1U;
        }
        j3_0 = j3_0 + 256U;
    }
    return;
}

bool crushable_0(uint k_54, KernelContext_0 thread* kernelContext_205)
{
    return ((as_type<uint>(((float4(*(kernelContext_205->crush_tab_0+(2U * k_54 + 1U))) ).z))) & 1U) != 0U;
}

uint crush_flags_0(uint k_55, KernelContext_0 thread* kernelContext_206)
{
    return (as_type<uint>(((float4(*(kernelContext_206->crush_rw_0+k_55)) ).w)));
}

void crush_record_0(uint k_56, float4 step_6, uint cause_0, KernelContext_0 thread* kernelContext_207)
{
    float4 _S1416 = float4(*(kernelContext_207->crush_rw_0+k_56)) ;
    uint flags_1 = (as_type<uint>((_S1416.w)));
    uint flags_2;
    if((step_6.w) != 0.0f)
    {
        flags_2 = ((flags_1 | 1U) | 2U) | (cause_0 << 4U);
    }
    else
    {
        flags_2 = flags_1;
    }
    *(kernelContext_207->crush_rw_0+k_56) = packed_float4(float4(step_6.x, max(_S1416.y, step_6.y), max(_S1416.z, step_6.z), (as_type<float>((flags_2))))) ;
    return;
}

[[kernel]] void fr_crush_blast(uint3 id_72 [[thread_position_in_grid]], FractureParams_0 constant* params_78 [[buffer(0)]], atomic<uint> device* atom_78 [[buffer(5)]], uint device* work_78 [[buffer(4)]], packed_float4 device* frames_78 [[buffer(6)]], packed_float4 device* bodies_78 [[buffer(7)]], packed_float4 device* chunks_78 [[buffer(1)]], uint device* sort_78 [[buffer(11)]], Record_natural_0 device* manifolds_78 [[buffer(10)]], packed_float4 device* loads_78 [[buffer(12)]], packed_uint4 device* bonds_78 [[buffer(2)]], uint device* args_78 [[buffer(13)]], uint device* info_78 [[buffer(15)]], ScanParams_0 constant* scan_78 [[buffer(14)]], packed_float4 device* parts_78 [[buffer(3)]], packed_float4 device* colliders_78 [[buffer(8)]], packed_uint4 device* collider_info_78 [[buffer(9)]], packed_float4 device* view_chunks_78 [[buffer(17)]], ViewParams_0 constant* view_78 [[buffer(16)]], packed_float4 device* objects_78 [[buffer(18)]], uint device* shot_bodies_78 [[buffer(20)]], float device* lines_78 [[buffer(19)]], packed_float4 device* geometry_78 [[buffer(21)]], packed_float4 device* shots_78 [[buffer(22)]], packed_float4 device* lp_78 [[buffer(24)]], packed_float4 device* ext_80 [[buffer(23)]], packed_float4 device* engine_contacts_78 [[buffer(25)]], CrushParams_0 constant* crush_78 [[buffer(28)]], packed_float4 device* crush_rw_78 [[buffer(27)]], packed_float4 device* crush_tab_78 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_208;
    (&kernelContext_208)->params_0 = params_78;
    (&kernelContext_208)->atom_0 = atom_78;
    (&kernelContext_208)->work_0 = work_78;
    (&kernelContext_208)->frames_0 = frames_78;
    (&kernelContext_208)->bodies_0 = bodies_78;
    (&kernelContext_208)->chunks_0 = chunks_78;
    (&kernelContext_208)->sort_0 = sort_78;
    (&kernelContext_208)->manifolds_0 = manifolds_78;
    (&kernelContext_208)->loads_0 = loads_78;
    (&kernelContext_208)->bonds_0 = bonds_78;
    (&kernelContext_208)->args_0 = args_78;
    (&kernelContext_208)->info_0 = info_78;
    (&kernelContext_208)->scan_0 = scan_78;
    (&kernelContext_208)->parts_0 = parts_78;
    (&kernelContext_208)->colliders_0 = colliders_78;
    (&kernelContext_208)->collider_info_0 = collider_info_78;
    (&kernelContext_208)->view_chunks_0 = view_chunks_78;
    (&kernelContext_208)->view_0 = view_78;
    (&kernelContext_208)->objects_0 = objects_78;
    (&kernelContext_208)->shot_bodies_0 = shot_bodies_78;
    (&kernelContext_208)->lines_0 = lines_78;
    (&kernelContext_208)->geometry_0 = geometry_78;
    (&kernelContext_208)->shots_0 = shots_78;
    (&kernelContext_208)->lp_0 = lp_78;
    (&kernelContext_208)->ext_0 = ext_80;
    (&kernelContext_208)->engine_contacts_0 = engine_contacts_78;
    (&kernelContext_208)->crush_0 = crush_78;
    (&kernelContext_208)->crush_rw_0 = crush_rw_78;
    (&kernelContext_208)->crush_tab_0 = crush_tab_78;
    threadgroup array<uint, int(1024)> g_scan_78;
    (&kernelContext_208)->g_scan_0 = &g_scan_78;
    threadgroup array<float, int(64)> g_il_best_78;
    (&kernelContext_208)->g_il_best_0 = &g_il_best_78;
    uint k_57 = id_72.x;
    bool _S1417;
    if(k_57 >= (params_78->chunk_count_0))
    {
        _S1417 = true;
    }
    else
    {
        _S1417 = ((&kernelContext_208)->crush_0->enabled_0) == 0U;
    }
    if(_S1417)
    {
        return;
    }
    float4 _S1418 = float4(*((&kernelContext_208)->crush_rw_0+((&kernelContext_208)->crush_0->r_blast_0 + k_57))) ;
    float _S1419 = _S1418.x;
    if(!(_S1419 > 0.0f))
    {
        _S1417 = !((_S1418.y) > 0.0f);
    }
    else
    {
        _S1417 = false;
    }
    if(_S1417)
    {
        _S1417 = !((_S1418.z) > 0.0f);
    }
    else
    {
        _S1417 = false;
    }
    if(_S1417)
    {
        return;
    }
    *((&kernelContext_208)->crush_rw_0+((&kernelContext_208)->crush_0->r_blast_0 + k_57)) = packed_float4(float4(0.0f) ) ;
    bool _S1420 = crushable_0(k_57, &kernelContext_208);
    if(!_S1420)
    {
        _S1417 = true;
    }
    else
    {
        uint _S1421 = crush_flags_0(k_57, &kernelContext_208);
        _S1417 = (_S1421 & 1U) != 0U;
    }
    if(_S1417)
    {
        return;
    }
    float d_22 = (float4(*((&kernelContext_208)->crush_rw_0+k_57)) ).x + _S1419;
    float4 step_7;
    if(d_22 >= 1.0f)
    {
        step_7 = float4(1.0f, _S1418.y, _S1418.z, 1.0f);
    }
    else
    {
        step_7 = float4(d_22, _S1418.y, _S1418.z, 0.0f);
    }
    crush_record_0(k_57, step_7, 3U, &kernelContext_208);
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
    float3 _S1422 = float3(0.0f) ;
    (&s_22)->d_23 = _S1422;
    (&s_22)->o_9 = _S1422;
    return s_22;
}

Sym_0 sym_outer_0(float3 r_28, float3 f_13)
{
    thread Sym_0 s_23;
    (&s_23)->d_23 = r_28 * f_13;
    float _S1423 = r_28.x;
    float _S1424 = f_13.y;
    float _S1425 = r_28.y;
    float _S1426 = f_13.x;
    float _S1427 = f_13.z;
    float _S1428 = r_28.z;
    (&s_23)->o_9 = float3(0.5f)  * float3(_S1423 * _S1424 + _S1425 * _S1426, _S1423 * _S1427 + _S1428 * _S1426, _S1425 * _S1427 + _S1428 * _S1424);
    return s_23;
}

Sym_0 sym_add_0(const Sym_0 thread* a_10, const Sym_0 thread* b_54)
{
    thread Sym_0 s_24;
    (&s_24)->d_23 = a_10->d_23 + b_54->d_23;
    (&s_24)->o_9 = a_10->o_9 + b_54->o_9;
    return s_24;
}

void add_load_0(Sym_0 thread* w_9, float3 thread* f_sum_0, float4 thread* spread_0, float3 r_29, float3 f_14)
{
    Sym_0 _S1429 = sym_outer_0(r_29, f_14);
    thread Sym_0 _S1430 = *w_9;
    thread Sym_0 _S1431 = _S1429;
    Sym_0 _S1432 = sym_add_0(&_S1430, &_S1431);
    *w_9 = _S1432;
    *f_sum_0 = *f_sum_0 + f_14;
    float m_6 = length(f_14);
    *spread_0 = *spread_0 + float4(float3(m_6)  * r_29, m_6);
    return;
}

void flow_virial_0(uint k_58, const Quat_0 thread* q_28, float3 origin_9, float3 centre_1, Sym_0 thread* wf_0, float thread* total_0, KernelContext_0 thread* kernelContext_209)
{
    *wf_0 = sym_zero_0();
    *total_0 = 0.0f;
    float4 _S1433 = float4(*(kernelContext_209->crush_tab_0+(kernelContext_209->crush_0->t_adj_0 + k_58))) ;
    uint first_2 = (as_type<uint>((_S1433.x)));
    uint _S1434 = (as_type<uint>((_S1433.y)));
    uint j_11 = first_2;
    for(;;)
    {
        if(j_11 < (first_2 + _S1434))
        {
        }
        else
        {
            break;
        }
        uint bi_0 = (as_type<uint>(((float4(*(kernelContext_209->crush_tab_0+(kernelContext_209->crush_0->t_list_0 + j_11 / 4U))) )[j_11 % 4U])));
        uint _S1435 = w_alive_0(kernelContext_209);
        if((*(kernelContext_209->work_0+(_S1435 + bi_0))) == 0U)
        {
            j_11 = j_11 + 1U;
            continue;
        }
        float l_2 = (*(kernelContext_209->crush_rw_0+(kernelContext_209->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U];
        if(!(((*(kernelContext_209->crush_rw_0+(kernelContext_209->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U]) > 0.0f))
        {
            j_11 = j_11 + 1U;
            continue;
        }
        float3 _S1436 = rotate_0(q_28, (float4(*(kernelContext_209->crush_tab_0+(kernelContext_209->crush_0->t_bonds_0 + bi_0))) ).xyz);
        float3 r_30 = origin_9 + _S1436 - centre_1;
        float3 _S1437 = rotate_0(q_28, (float4(*(kernelContext_209->crush_tab_0+(kernelContext_209->crush_0->t_normal_0 + bi_0))) ).xyz);
        float3 out_n_0;
        if((dot(_S1437, r_30)) >= 0.0f)
        {
            out_n_0 = _S1437;
        }
        else
        {
            out_n_0 = - _S1437;
        }
        Sym_0 _S1438 = sym_outer_0(r_30, float3(- l_2)  * out_n_0);
        thread Sym_0 _S1439 = *wf_0;
        thread Sym_0 _S1440 = _S1438;
        Sym_0 _S1441 = sym_add_0(&_S1439, &_S1440);
        *wf_0 = _S1441;
        *total_0 = *total_0 + l_2;
        j_11 = j_11 + 1U;
    }
    return;
}

float4 bond_centroid_0(uint k_59, const Quat_0 thread* q_29, float3 origin_10, float3 centre_2, KernelContext_0 thread* kernelContext_210)
{
    float4 _S1442 = float4(*(kernelContext_210->crush_tab_0+(kernelContext_210->crush_0->t_adj_0 + k_59))) ;
    uint first_3 = (as_type<uint>((_S1442.x)));
    uint _S1443 = (as_type<uint>((_S1442.y)));
    float3 _S1444 = float3(0.0f) ;
    uint j_12 = first_3;
    float3 sum_4 = _S1444;
    float area_1 = 0.0f;
    for(;;)
    {
        if(j_12 < (first_3 + _S1443))
        {
        }
        else
        {
            break;
        }
        uint bi_1 = (as_type<uint>(((float4(*(kernelContext_210->crush_tab_0+(kernelContext_210->crush_0->t_list_0 + j_12 / 4U))) )[j_12 % 4U])));
        uint _S1445 = w_alive_0(kernelContext_210);
        if((*(kernelContext_210->work_0+(_S1445 + bi_1))) == 0U)
        {
            j_12 = j_12 + 1U;
            continue;
        }
        float4 _S1446 = float4(*(kernelContext_210->crush_tab_0+(kernelContext_210->crush_0->t_bonds_0 + bi_1))) ;
        float3 _S1447 = rotate_0(q_29, _S1446.xyz);
        float _S1448 = _S1446.w;
        float area_2 = area_1 + _S1448;
        sum_4 = sum_4 + float3(_S1448)  * (origin_10 + _S1447 - centre_2);
        area_1 = area_2;
        j_12 = j_12 + 1U;
    }
    if(area_1 > 0.0f)
    {
        sum_4 = sum_4 / float3(area_1) ;
    }
    else
    {
        sum_4 = _S1444;
    }
    return float4(sum_4, area_1);
}

Sym_0 balanced_0(const Sym_0 thread* w_10, float3 f_15, float4 spread_1, float4 bonds_79)
{
    float3 at_23;
    if((bonds_79.w) > 0.0f)
    {
        at_23 = bonds_79.xyz;
    }
    else
    {
        float _S1449 = spread_1.w;
        if(_S1449 > 0.0f)
        {
            at_23 = spread_1.xyz / float3(_S1449) ;
        }
        else
        {
            at_23 = float3(0.0f) ;
        }
    }
    thread Sym_0 _S1450 = sym_outer_0(at_23, - f_15);
    Sym_0 _S1451 = sym_add_0(w_10, &_S1450);
    return _S1451;
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
    float _S1452 = pq_0.x;
    if(!(_S1452 > 0.0f))
    {
        return 0.0f;
    }
    float _S1453 = max(pq_0.y - (m0_0.y + m0_0.z * _S1452), _S1452 - m0_0.x);
    float _S1454;
    if(_S1453 > 0.0f)
    {
        _S1454 = _S1453;
    }
    else
    {
        _S1454 = 0.0f;
    }
    return _S1454;
}

float4 crush_two_0(const Sym_0 thread* a_11, float dt_a_0, const Sym_0 thread* b_55, float dt_b_0, uint k_60, float damage_2, KernelContext_0 thread* kernelContext_211)
{
    uint _S1455 = 2U * k_60;
    float4 _S1456 = float4(*(kernelContext_211->crush_tab_0+_S1455)) ;
    float4 _S1457 = float4(*(kernelContext_211->crush_tab_0+(_S1455 + 1U))) ;
    float _S1458 = _S1457.y;
    float2 _S1459 = crush_invariants_0(a_11, _S1458);
    float2 _S1460 = crush_invariants_0(b_55, _S1458);
    float ea_0 = crush_excess_0(_S1459, _S1456);
    bool _S1461 = dt_b_0 > 0.0f;
    float eb_0;
    if(_S1461)
    {
        eb_0 = crush_excess_0(_S1460, _S1456);
    }
    else
    {
        eb_0 = 0.0f;
    }
    float d_24 = damage_2 + (ea_0 * ea_0 * dt_a_0 + eb_0 * eb_0 * dt_b_0) / (_S1457.x * _S1456.w);
    float2 peak_2;
    if(_S1461)
    {
        peak_2 = max(_S1459, _S1460);
    }
    else
    {
        peak_2 = _S1459;
    }
    if(d_24 >= 1.0f)
    {
        return float4(1.0f, peak_2, 1.0f);
    }
    return float4(d_24, peak_2, 0.0f);
}

void crush_loads_flow_0(uint k_61, KernelContext_0 thread* kernelContext_212)
{
    uint _S1462 = body_of_0(k_61, kernelContext_212);
    uint b_56 = 7U * _S1462;
    thread Quat_0 q_30;
    thread float3 origin_11;
    structure_pose_0(_S1462, b_56, &q_30, &origin_11, kernelContext_212);
    uint _S1463 = 4U * k_61;
    float4 _S1464 = float4(*(kernelContext_212->chunks_0+_S1463)) ;
    float4 _S1465 = float4(*(kernelContext_212->chunks_0+(_S1463 + 3U))) ;
    float3 _S1466 = _S1464.xyz;
    thread Quat_0 _S1467 = q_30;
    float3 _S1468 = rotate_0(&_S1467, _S1466);
    float3 centre_3 = origin_11 + _S1468;
    float inv_mass_8 = (float4(*(kernelContext_212->bodies_0+b_56)) ).w;
    float _S1469;
    if(inv_mass_8 > 0.0f)
    {
        _S1469 = 1.0f / inv_mass_8;
    }
    else
    {
        _S1469 = float_bits_0(2139095040U);
    }
    float dt_2 = kernelContext_212->params_0->dt_0;
    Sym_0 _S1470 = sym_zero_0();
    thread Sym_0 wa_0 = _S1470;
    float3 _S1471 = float3(0.0f) ;
    thread float3 fa_0 = _S1471;
    float4 _S1472 = float4(0.0f) ;
    thread float4 sa_0 = _S1472;
    thread Sym_0 wb_0 = _S1470;
    thread float3 fb_1 = _S1471;
    thread float4 sb_2 = _S1472;
    uint _S1473 = kernelContext_212->sort_0[k_61];
    uint _S1474 = kernelContext_212->sort_0[kernelContext_212->params_0->chunk_count_0 + 1U + k_61];
    float tau_3 = 0.0f;
    uint i_23 = 0U;
    for(;;)
    {
        if(i_23 < _S1474)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_31 = kernelContext_212->manifolds_0[_S1473 + i_23];
        float4 _S1475 = float4(r_31.impulse_0) ;
        float3 _S1476 = _S1475.xyz;
        float j_13 = length(_S1476);
        if(j_13 == 0.0f)
        {
            i_23 = i_23 + 1U;
            continue;
        }
        float4 _S1477 = float4(r_31.other_0) ;
        float m_other_2 = _S1477.x;
        float m_7;
        if(isinf(_S1469))
        {
            if(isinf(m_other_2))
            {
                m_7 = _S1464.w;
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
                m_7 = _S1469;
            }
            else
            {
                m_7 = _S1469 * m_other_2 / (_S1469 + m_other_2);
            }
        }
        float _S1478 = _S1475.w;
        float _S1479 = min(j_13, 2.0f * m_7 * max(_S1478 - kernelContext_212->params_0->band_0, 0.0f));
        float3 rr_0 = (float4(r_31.point_0) ).xyz - centre_3;
        float3 dir_1 = _S1476 / float3(j_13) ;
        if(_S1479 < j_13)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, rr_0, dir_1 * float3(((j_13 - _S1479) / dt_2)) );
        }
        float tau_4;
        if(_S1479 > 0.0f)
        {
            float _S1480 = _S1465.y;
            float duration_2 = hertz_duration_0(m_7, min(_S1477.z, _S1465.z), 1.0f / ((1.0f - _S1480 * _S1480) / _S1465.x + 1.0f / _S1477.y), _S1478);
            add_load_0(&wb_0, &fb_1, &sb_2, rr_0, dir_1 * float3((3.14159274101257324f * _S1479 / (2.0f * duration_2))) );
            tau_4 = max(tau_3, min(duration_2, dt_2));
        }
        else
        {
            tau_4 = tau_3;
        }
        tau_3 = tau_4;
        i_23 = i_23 + 1U;
    }
    uint _S1481 = frame_parity_0(kernelContext_212);
    uint _S1482 = w_pulses_0(kernelContext_212);
    uint _S1483 = _S1482 + _S1463 + 2U * _S1481;
    uint off_1 = *(kernelContext_212->work_0+_S1483);
    uint _S1484 = *(kernelContext_212->work_0+(_S1483 + 1U));
    uint _S1485 = l_pool_0(_S1481, kernelContext_212);
    uint _S1486 = _S1485 + 3U * off_1;
    uint p_22 = 0U;
    for(;;)
    {
        if(p_22 < _S1484)
        {
        }
        else
        {
            break;
        }
        uint _S1487 = _S1486 + 3U * p_22;
        float4 _S1488 = float4(*(kernelContext_212->loads_0+_S1487)) ;
        float4 _S1489 = float4(*(kernelContext_212->loads_0+(_S1487 + 1U))) ;
        float4 _S1490 = float4(*(kernelContext_212->loads_0+(_S1487 + 2U))) ;
        float mean_2 = pulse_mean_0(_S1488.x, _S1488.y, _S1488.z, _S1488.w, _S1489.w, dt_2);
        if(mean_2 != 0.0f)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, _S1489.xyz - centre_3, _S1490.xyz * float3(mean_2) );
        }
        p_22 = p_22 + 1U;
    }
    thread Quat_0 _S1491 = q_30;
    thread Sym_0 wf_1;
    thread float flow_total_0;
    flow_virial_0(k_61, &_S1491, origin_11, centre_3, &wf_1, &flow_total_0, kernelContext_212);
    bool contact_18;
    if((sa_0.w) > 0.0f)
    {
        contact_18 = true;
    }
    else
    {
        contact_18 = (sb_2.w) > 0.0f;
    }
    bool _S1492;
    if(!contact_18)
    {
        _S1492 = !(flow_total_0 > 0.0f);
    }
    else
    {
        _S1492 = false;
    }
    if(_S1492)
    {
        return;
    }
    Sym_0 _S1493 = wf_1;
    Sym_0 _S1494 = wf_1;
    Sym_0 peak_contact_0;
    Sym_0 base_3;
    Sym_0 peak_3;
    if(contact_18)
    {
        thread Quat_0 _S1495 = q_30;
        float4 _S1496 = bond_centroid_0(k_61, &_S1495, origin_11, centre_3, kernelContext_212);
        thread Sym_0 _S1497 = wa_0;
        Sym_0 _S1498 = balanced_0(&_S1497, fa_0, sa_0, _S1496);
        thread Sym_0 _S1499 = _S1498;
        thread Sym_0 _S1500 = wf_1;
        Sym_0 _S1501 = sym_add_0(&_S1499, &_S1500);
        thread Sym_0 _S1502 = wa_0;
        thread Sym_0 _S1503 = wb_0;
        Sym_0 _S1504 = sym_add_0(&_S1502, &_S1503);
        float3 _S1505 = fa_0 + fb_1;
        float4 _S1506 = sa_0 + sb_2;
        thread Sym_0 _S1507 = _S1504;
        Sym_0 _S1508 = balanced_0(&_S1507, _S1505, _S1506, _S1496);
        thread Sym_0 _S1509 = _S1508;
        thread Sym_0 _S1510 = wf_1;
        Sym_0 _S1511 = sym_add_0(&_S1509, &_S1510);
        peak_contact_0 = _S1508;
        base_3 = _S1501;
        peak_3 = _S1511;
    }
    else
    {
        peak_contact_0 = _S1470;
        base_3 = _S1493;
        peak_3 = _S1494;
    }
    thread Sym_0 _S1512 = wf_1;
    float _S1513 = sym_norm2_0(&_S1512);
    thread Sym_0 _S1514 = peak_contact_0;
    float _S1515 = sym_norm2_0(&_S1514);
    uint cause_1;
    if(_S1513 >= _S1515)
    {
        cause_1 = 4U;
    }
    else
    {
        cause_1 = 1U;
    }
    float _S1516 = dt_2 - tau_3;
    float _S1517 = (float4(*(kernelContext_212->crush_rw_0+k_61)) ).x;
    thread Sym_0 _S1518 = base_3;
    thread Sym_0 _S1519 = peak_3;
    float4 _S1520 = crush_two_0(&_S1518, _S1516, &_S1519, tau_3, k_61, _S1517, kernelContext_212);
    crush_record_0(k_61, _S1520, cause_1, kernelContext_212);
    return;
}

Sym_0 sym_rotated_0(const Quat_0 thread* q_31, float3 d_25, float3 o_10)
{
    float _S1521 = o_10.x;
    float _S1522 = o_10.y;
    float _S1523 = o_10.z;
    matrix<float,int(3),int(3)>  s_26 = matrix<float,int(3),int(3)> (d_25.x, _S1521, _S1522, _S1521, d_25.y, _S1523, _S1522, _S1523, d_25.z);
    float3 _S1524 = rotate_0(q_31, float3(1.0f, 0.0f, 0.0f));
    float3 _S1525 = rotate_0(q_31, float3(0.0f, 1.0f, 0.0f));
    float3 _S1526 = rotate_0(q_31, float3(0.0f, 0.0f, 1.0f));
    matrix<float,int(3),int(3)>  r_32 = matrix<float,int(3),int(3)> (_S1524.x, _S1525.x, _S1526.x, _S1524.y, _S1525.y, _S1526.y, _S1524.z, _S1525.z, _S1526.z);
    matrix<float,int(3),int(3)>  w_11 = (((transpose(r_32)) * ((((s_26) * (r_32))))));
    thread Sym_0 out_s_0;
    (&out_s_0)->d_23 = float3(w_11[int(0)][int(0)], w_11[int(1)][int(1)], w_11[int(2)][int(2)]);
    (&out_s_0)->o_9 = float3(w_11[int(0)][int(1)], w_11[int(0)][int(2)], w_11[int(1)][int(2)]);
    return out_s_0;
}

[[kernel]] void fr_crush_loads(uint3 id_73 [[thread_position_in_grid]], FractureParams_0 constant* params_79 [[buffer(0)]], atomic<uint> device* atom_79 [[buffer(5)]], uint device* work_79 [[buffer(4)]], packed_float4 device* frames_79 [[buffer(6)]], packed_float4 device* bodies_79 [[buffer(7)]], packed_float4 device* chunks_79 [[buffer(1)]], uint device* sort_79 [[buffer(11)]], Record_natural_0 device* manifolds_79 [[buffer(10)]], packed_float4 device* loads_79 [[buffer(12)]], packed_uint4 device* bonds_80 [[buffer(2)]], uint device* args_79 [[buffer(13)]], uint device* info_79 [[buffer(15)]], ScanParams_0 constant* scan_79 [[buffer(14)]], packed_float4 device* parts_79 [[buffer(3)]], packed_float4 device* colliders_79 [[buffer(8)]], packed_uint4 device* collider_info_79 [[buffer(9)]], packed_float4 device* view_chunks_79 [[buffer(17)]], ViewParams_0 constant* view_79 [[buffer(16)]], packed_float4 device* objects_79 [[buffer(18)]], uint device* shot_bodies_79 [[buffer(20)]], float device* lines_79 [[buffer(19)]], packed_float4 device* geometry_79 [[buffer(21)]], packed_float4 device* shots_79 [[buffer(22)]], packed_float4 device* lp_79 [[buffer(24)]], packed_float4 device* ext_81 [[buffer(23)]], packed_float4 device* engine_contacts_79 [[buffer(25)]], CrushParams_0 constant* crush_79 [[buffer(28)]], packed_float4 device* crush_rw_79 [[buffer(27)]], packed_float4 device* crush_tab_79 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_213;
    (&kernelContext_213)->params_0 = params_79;
    (&kernelContext_213)->atom_0 = atom_79;
    (&kernelContext_213)->work_0 = work_79;
    (&kernelContext_213)->frames_0 = frames_79;
    (&kernelContext_213)->bodies_0 = bodies_79;
    (&kernelContext_213)->chunks_0 = chunks_79;
    (&kernelContext_213)->sort_0 = sort_79;
    (&kernelContext_213)->manifolds_0 = manifolds_79;
    (&kernelContext_213)->loads_0 = loads_79;
    (&kernelContext_213)->bonds_0 = bonds_80;
    (&kernelContext_213)->args_0 = args_79;
    (&kernelContext_213)->info_0 = info_79;
    (&kernelContext_213)->scan_0 = scan_79;
    (&kernelContext_213)->parts_0 = parts_79;
    (&kernelContext_213)->colliders_0 = colliders_79;
    (&kernelContext_213)->collider_info_0 = collider_info_79;
    (&kernelContext_213)->view_chunks_0 = view_chunks_79;
    (&kernelContext_213)->view_0 = view_79;
    (&kernelContext_213)->objects_0 = objects_79;
    (&kernelContext_213)->shot_bodies_0 = shot_bodies_79;
    (&kernelContext_213)->lines_0 = lines_79;
    (&kernelContext_213)->geometry_0 = geometry_79;
    (&kernelContext_213)->shots_0 = shots_79;
    (&kernelContext_213)->lp_0 = lp_79;
    (&kernelContext_213)->ext_0 = ext_81;
    (&kernelContext_213)->engine_contacts_0 = engine_contacts_79;
    (&kernelContext_213)->crush_0 = crush_79;
    (&kernelContext_213)->crush_rw_0 = crush_rw_79;
    (&kernelContext_213)->crush_tab_0 = crush_tab_79;
    threadgroup array<uint, int(1024)> g_scan_79;
    (&kernelContext_213)->g_scan_0 = &g_scan_79;
    threadgroup array<float, int(64)> g_il_best_79;
    (&kernelContext_213)->g_il_best_0 = &g_il_best_79;
    uint k_62 = id_73.x;
    bool _S1527;
    if(k_62 >= (params_79->chunk_count_0))
    {
        _S1527 = true;
    }
    else
    {
        _S1527 = ((&kernelContext_213)->crush_0->enabled_0) == 0U;
    }
    if(_S1527)
    {
        _S1527 = true;
    }
    else
    {
        bool _S1528 = crushable_0(k_62, &kernelContext_213);
        _S1527 = !_S1528;
    }
    if(_S1527)
    {
        _S1527 = true;
    }
    else
    {
        uint _S1529 = crush_flags_0(k_62, &kernelContext_213);
        _S1527 = (_S1529 & 1U) != 0U;
    }
    if(_S1527)
    {
        return;
    }
    if(((&kernelContext_213)->crush_0->flow_0) != 0U)
    {
        crush_loads_flow_0(k_62, &kernelContext_213);
        return;
    }
    uint _S1530 = body_of_0(k_62, &kernelContext_213);
    uint b_57 = 7U * _S1530;
    thread Quat_0 q_32;
    thread float3 origin_12;
    structure_pose_0(_S1530, b_57, &q_32, &origin_12, &kernelContext_213);
    uint _S1531 = 4U * k_62;
    float4 _S1532 = float4(*((&kernelContext_213)->chunks_0+_S1531)) ;
    float4 _S1533 = float4(*((&kernelContext_213)->chunks_0+(_S1531 + 3U))) ;
    float3 _S1534 = _S1532.xyz;
    thread Quat_0 _S1535 = q_32;
    float3 _S1536 = rotate_0(&_S1535, _S1534);
    float3 centre_4 = origin_12 + _S1536;
    float inv_mass_9 = (float4(*((&kernelContext_213)->bodies_0+b_57)) ).w;
    float _S1537;
    if(inv_mass_9 > 0.0f)
    {
        _S1537 = 1.0f / inv_mass_9;
    }
    else
    {
        _S1537 = float_bits_0(2139095040U);
    }
    float dt_3 = (&kernelContext_213)->params_0->dt_0;
    Sym_0 _S1538 = sym_zero_0();
    thread Sym_0 wa_1 = _S1538;
    float3 _S1539 = float3(0.0f) ;
    thread float3 fa_1 = _S1539;
    float4 _S1540 = float4(0.0f) ;
    thread float4 sa_1 = _S1540;
    thread Sym_0 wb_1 = _S1538;
    thread float3 fb_2 = _S1539;
    thread float4 sb_3 = _S1540;
    uint _S1541 = (&kernelContext_213)->sort_0[k_62];
    uint _S1542 = (&kernelContext_213)->sort_0[params_79->chunk_count_0 + 1U + k_62];
    float tau_5 = 0.0f;
    uint i_24 = 0U;
    for(;;)
    {
        if(i_24 < _S1542)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_33 = (&kernelContext_213)->manifolds_0[_S1541 + i_24];
        float4 _S1543 = float4(r_33.impulse_0) ;
        float3 _S1544 = _S1543.xyz;
        float j_14 = length(_S1544);
        if(j_14 == 0.0f)
        {
            i_24 = i_24 + 1U;
            continue;
        }
        float4 _S1545 = float4(r_33.other_0) ;
        float m_other_3 = _S1545.x;
        float m_8;
        if(isinf(_S1537))
        {
            if(isinf(m_other_3))
            {
                m_8 = _S1532.w;
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
                m_8 = _S1537;
            }
            else
            {
                m_8 = _S1537 * m_other_3 / (_S1537 + m_other_3);
            }
        }
        float _S1546 = _S1543.w;
        float _S1547 = min(j_14, 2.0f * m_8 * max(_S1546 - (&kernelContext_213)->params_0->band_0, 0.0f));
        float3 rr_1 = (float4(r_33.point_0) ).xyz - centre_4;
        float3 dir_2 = _S1544 / float3(j_14) ;
        if(_S1547 < j_14)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, rr_1, dir_2 * float3(((j_14 - _S1547) / dt_3)) );
        }
        float tau_6;
        if(_S1547 > 0.0f)
        {
            float _S1548 = _S1533.y;
            float duration_3 = hertz_duration_0(m_8, min(_S1545.z, _S1533.z), 1.0f / ((1.0f - _S1548 * _S1548) / _S1533.x + 1.0f / _S1545.y), _S1546);
            add_load_0(&wb_1, &fb_2, &sb_3, rr_1, dir_2 * float3((3.14159274101257324f * _S1547 / (2.0f * duration_3))) );
            tau_6 = max(tau_5, min(duration_3, dt_3));
        }
        else
        {
            tau_6 = tau_5;
        }
        tau_5 = tau_6;
        i_24 = i_24 + 1U;
    }
    uint _S1549 = frame_parity_0(&kernelContext_213);
    uint _S1550 = w_pulses_0(&kernelContext_213);
    uint _S1551 = _S1550 + _S1531 + 2U * _S1549;
    uint device* _S1552 = (&kernelContext_213)->work_0+_S1551;
    uint _S1553 = *((&kernelContext_213)->work_0+(_S1551 + 1U));
    uint _S1554 = l_pool_0(_S1549, &kernelContext_213);
    uint _S1555 = _S1554 + 3U * *_S1552;
    uint p_23 = 0U;
    for(;;)
    {
        if(p_23 < _S1553)
        {
        }
        else
        {
            break;
        }
        uint _S1556 = _S1555 + 3U * p_23;
        float4 _S1557 = float4(*((&kernelContext_213)->loads_0+_S1556)) ;
        float4 _S1558 = float4(*((&kernelContext_213)->loads_0+(_S1556 + 1U))) ;
        float4 _S1559 = float4(*((&kernelContext_213)->loads_0+(_S1556 + 2U))) ;
        float mean_3 = pulse_mean_0(_S1557.x, _S1557.y, _S1557.z, _S1557.w, _S1558.w, dt_3);
        if(mean_3 != 0.0f)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, _S1558.xyz - centre_4, _S1559.xyz * float3(mean_3) );
        }
        p_23 = p_23 + 1U;
    }
    if(((&kernelContext_213)->crush_0->bond_virial_0) != 0U)
    {
        uint _S1560 = 2U * k_62;
        float3 _S1561 = (float4(*((&kernelContext_213)->crush_rw_0+((&kernelContext_213)->crush_0->r_virial_0 + _S1560))) ).xyz;
        float3 _S1562 = (float4(*((&kernelContext_213)->crush_rw_0+((&kernelContext_213)->crush_0->r_virial_0 + _S1560 + 1U))) ).xyz;
        thread Quat_0 _S1563 = q_32;
        Sym_0 _S1564 = sym_rotated_0(&_S1563, _S1561, _S1562);
        thread Sym_0 _S1565 = wa_1;
        thread Sym_0 _S1566 = _S1564;
        Sym_0 _S1567 = sym_add_0(&_S1565, &_S1566);
        thread Sym_0 _S1568 = wa_1;
        thread Sym_0 _S1569 = wb_1;
        Sym_0 _S1570 = sym_add_0(&_S1568, &_S1569);
        thread Sym_0 _S1571 = _S1570;
        thread Sym_0 _S1572 = _S1564;
        Sym_0 _S1573 = sym_add_0(&_S1571, &_S1572);
        float _S1574 = dt_3 - tau_5;
        float _S1575 = (float4(*((&kernelContext_213)->crush_rw_0+k_62)) ).x;
        thread Sym_0 _S1576 = _S1567;
        thread Sym_0 _S1577 = _S1573;
        float4 _S1578 = crush_two_0(&_S1576, _S1574, &_S1577, tau_5, k_62, _S1575, &kernelContext_213);
        crush_record_0(k_62, _S1578, 1U, &kernelContext_213);
        return;
    }
    if(!((sa_1.w) > 0.0f))
    {
        _S1527 = !((sb_3.w) > 0.0f);
    }
    else
    {
        _S1527 = false;
    }
    if(_S1527)
    {
        return;
    }
    thread Quat_0 _S1579 = q_32;
    float4 _S1580 = bond_centroid_0(k_62, &_S1579, origin_12, centre_4, &kernelContext_213);
    thread Sym_0 _S1581 = wa_1;
    Sym_0 _S1582 = balanced_0(&_S1581, fa_1, sa_1, _S1580);
    thread Sym_0 _S1583 = wa_1;
    thread Sym_0 _S1584 = wb_1;
    Sym_0 _S1585 = sym_add_0(&_S1583, &_S1584);
    float3 _S1586 = fa_1 + fb_2;
    float4 _S1587 = sa_1 + sb_3;
    thread Sym_0 _S1588 = _S1585;
    Sym_0 _S1589 = balanced_0(&_S1588, _S1586, _S1587, _S1580);
    float _S1590 = dt_3 - tau_5;
    float _S1591 = (float4(*((&kernelContext_213)->crush_rw_0+k_62)) ).x;
    thread Sym_0 _S1592 = _S1582;
    thread Sym_0 _S1593 = _S1589;
    float4 _S1594 = crush_two_0(&_S1592, _S1590, &_S1593, tau_5, k_62, _S1591, &kernelContext_213);
    crush_record_0(k_62, _S1594, 1U, &kernelContext_213);
    return;
}

uint a_cover_0(KernelContext_0 thread* kernelContext_214)
{
    uint _S1595 = a_moved_0(kernelContext_214);
    return _S1595 + 2U * kernelContext_214->params_0->ev_moved_cap_0;
}

[[kernel]] void fr_strike_patch(uint3 id_74 [[thread_position_in_grid]], FractureParams_0 constant* params_80 [[buffer(0)]], atomic<uint> device* atom_80 [[buffer(5)]], uint device* work_80 [[buffer(4)]], packed_float4 device* frames_80 [[buffer(6)]], packed_float4 device* bodies_80 [[buffer(7)]], packed_float4 device* chunks_80 [[buffer(1)]], uint device* sort_80 [[buffer(11)]], Record_natural_0 device* manifolds_80 [[buffer(10)]], packed_float4 device* loads_80 [[buffer(12)]], packed_uint4 device* bonds_81 [[buffer(2)]], uint device* args_80 [[buffer(13)]], uint device* info_80 [[buffer(15)]], ScanParams_0 constant* scan_80 [[buffer(14)]], packed_float4 device* parts_80 [[buffer(3)]], packed_float4 device* colliders_80 [[buffer(8)]], packed_uint4 device* collider_info_80 [[buffer(9)]], packed_float4 device* view_chunks_80 [[buffer(17)]], ViewParams_0 constant* view_80 [[buffer(16)]], packed_float4 device* objects_80 [[buffer(18)]], uint device* shot_bodies_80 [[buffer(20)]], float device* lines_80 [[buffer(19)]], packed_float4 device* geometry_80 [[buffer(21)]], packed_float4 device* shots_80 [[buffer(22)]], packed_float4 device* lp_80 [[buffer(24)]], packed_float4 device* ext_82 [[buffer(23)]], packed_float4 device* engine_contacts_80 [[buffer(25)]], CrushParams_0 constant* crush_80 [[buffer(28)]], packed_float4 device* crush_rw_80 [[buffer(27)]], packed_float4 device* crush_tab_80 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_215;
    (&kernelContext_215)->params_0 = params_80;
    (&kernelContext_215)->atom_0 = atom_80;
    (&kernelContext_215)->work_0 = work_80;
    (&kernelContext_215)->frames_0 = frames_80;
    (&kernelContext_215)->bodies_0 = bodies_80;
    (&kernelContext_215)->chunks_0 = chunks_80;
    (&kernelContext_215)->sort_0 = sort_80;
    (&kernelContext_215)->manifolds_0 = manifolds_80;
    (&kernelContext_215)->loads_0 = loads_80;
    (&kernelContext_215)->bonds_0 = bonds_81;
    (&kernelContext_215)->args_0 = args_80;
    (&kernelContext_215)->info_0 = info_80;
    (&kernelContext_215)->scan_0 = scan_80;
    (&kernelContext_215)->parts_0 = parts_80;
    (&kernelContext_215)->colliders_0 = colliders_80;
    (&kernelContext_215)->collider_info_0 = collider_info_80;
    (&kernelContext_215)->view_chunks_0 = view_chunks_80;
    (&kernelContext_215)->view_0 = view_80;
    (&kernelContext_215)->objects_0 = objects_80;
    (&kernelContext_215)->shot_bodies_0 = shot_bodies_80;
    (&kernelContext_215)->lines_0 = lines_80;
    (&kernelContext_215)->geometry_0 = geometry_80;
    (&kernelContext_215)->shots_0 = shots_80;
    (&kernelContext_215)->lp_0 = lp_80;
    (&kernelContext_215)->ext_0 = ext_82;
    (&kernelContext_215)->engine_contacts_0 = engine_contacts_80;
    (&kernelContext_215)->crush_0 = crush_80;
    (&kernelContext_215)->crush_rw_0 = crush_rw_80;
    (&kernelContext_215)->crush_tab_0 = crush_tab_80;
    threadgroup array<uint, int(1024)> g_scan_80;
    (&kernelContext_215)->g_scan_0 = &g_scan_80;
    threadgroup array<float, int(64)> g_il_best_80;
    (&kernelContext_215)->g_il_best_0 = &g_il_best_80;
    uint b_58 = id_74.x;
    bool _S1596;
    if(b_58 >= (params_80->body_count_0))
    {
        _S1596 = true;
    }
    else
    {
        _S1596 = ((&kernelContext_215)->params_0->shot_count_0) == 0U;
    }
    if(_S1596)
    {
        _S1596 = true;
    }
    else
    {
        _S1596 = ((&kernelContext_215)->crush_0->enabled_0) == 0U;
    }
    if(_S1596)
    {
        return;
    }
    uint bill_0 = (&kernelContext_215)->crush_0->r_bill_0 + 4U * b_58;
    packed_float4 _S1597 = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_215)->crush_rw_0+bill_0) = _S1597;
    uint _S1598 = bill_0 + 3U;
    *((&kernelContext_215)->crush_rw_0+_S1598) = _S1597;
    thread uint shot_10;
    bool _S1599 = struck_0(b_58, &shot_10, &kernelContext_215);
    if(!_S1599)
    {
        return;
    }
    uint _S1600 = a_strike_0(&kernelContext_215);
    uint _S1601 = _S1600 + 9U * b_58;
    uint k_63 = atomic_load_explicit((&kernelContext_215)->atom_0+(_S1601 + 8U), memory_order_relaxed);
    uint _S1602 = atomic_load_explicit((&kernelContext_215)->atom_0+(_S1601 + 6U), memory_order_relaxed);
    float t_in_10 = (as_type<float>((_S1602)));
    if(!(t_in_10 <= ((&kernelContext_215)->params_0->dt_0)))
    {
        _S1596 = true;
    }
    else
    {
        _S1596 = k_63 == 4294967295U;
    }
    if(_S1596)
    {
        return;
    }
    uint _S1603 = strike_at_0(b_58, &kernelContext_215);
    float4 _S1604 = float4(*((&kernelContext_215)->frames_0+_S1603)) ;
    float3 _S1605 = (float4(*((&kernelContext_215)->frames_0+(_S1603 + 1U))) ).xyz;
    float speed_15 = length(_S1605);
    float3 d_26 = _S1605 / float3(max(speed_15, 9.999999960041972e-13f)) ;
    float4 _S1606 = float4(*((&kernelContext_215)->shots_0+2U * shot_10)) ;
    float radius_1 = _S1606.y;
    float _S1607 = max(_S1606.z, 0.00100000004749745f);
    float m_imp_1 = _S1606.w;
    float inv_mass_10 = (float4(*((&kernelContext_215)->bodies_0+7U * b_58)) ).w;
    bool _S1608 = inv_mass_10 == 0.0f;
    float m_body_2;
    if(_S1608)
    {
        m_body_2 = 0.0f;
    }
    else
    {
        m_body_2 = 1.0f / inv_mass_10;
    }
    float m_eff_9;
    if(_S1608)
    {
        m_eff_9 = m_imp_1;
    }
    else
    {
        m_eff_9 = m_imp_1 * m_body_2 / (m_imp_1 + m_body_2);
    }
    float modulus_2 = 1.0f / (1.0f / (float4(*((&kernelContext_215)->shots_0+(2U * shot_10 + 1U))) ).x + 1.0f / (float4(*((&kernelContext_215)->geometry_0+(3U * k_63 + 1U))) ).w);
    float _S1609 = max(speed_15, 0.00100000004749745f);
    float tau_7 = 2.86999988555908203f * pow(m_eff_9 * m_eff_9 / (_S1607 * modulus_2 * modulus_2 * _S1609), 0.20000000298023224f);
    float force_3 = m_eff_9 * speed_15 / tau_7;
    float3 contact_19 = _S1604.xyz + _S1605 * float3(t_in_10)  + d_26 * float3(radius_1) ;
    float delta_3 = pow(15.0f * m_eff_9 * _S1609 * _S1609 / (16.0f * modulus_2 * sqrt(_S1607)), 0.40000000596046448f);
    float a_h_0 = sqrt(radius_1 * delta_3);
    bool _S1610 = crushable_0(k_63, &kernelContext_215);
    float p_cap_0;
    if(_S1610)
    {
        p_cap_0 = (float4(*((&kernelContext_215)->crush_tab_0+2U * k_63)) ).x;
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
    float _S1611 = min(radius_1, max(a_h_0, a_p_0));
    float _S1612 = max(delta_3, radius_1 - sqrt(max(radius_1 * radius_1 - _S1611 * _S1611, 0.0f)));
    *((&kernelContext_215)->crush_rw_0+(bill_0 + 1U)) = packed_float4(float4(contact_19, 0.0f)) ;
    *((&kernelContext_215)->crush_rw_0+(bill_0 + 2U)) = packed_float4(float4(d_26, _S1611)) ;
    *((&kernelContext_215)->crush_rw_0+_S1598) = packed_float4(float4(force_3, tau_7, _S1612, float(k_63 + 1U))) ;
    uint _S1613 = a_cover_0(&kernelContext_215);
    uint _S1614 = _S1613 + 128U * b_58;
    uint s_27 = 0U;
    for(;;)
    {
        if(s_27 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S1615 = _S1614 + 2U * s_27;
        atomic_store_explicit((&kernelContext_215)->atom_0+_S1615, 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_215)->atom_0+(_S1615 + 1U), 4294967295U, memory_order_relaxed);
        s_27 = s_27 + 1U;
    }
    return;
}

bool cover_patch_0(uint k_64, uint thread* b_59, float4 thread* p1_0, float4 thread* p2_2, KernelContext_0 thread* kernelContext_216)
{
    uint _S1616 = body_of_0(k_64, kernelContext_216);
    *b_59 = _S1616;
    float4 _S1617 = float4(0.0f) ;
    *p1_0 = _S1617;
    *p2_2 = _S1617;
    thread uint shot_11;
    bool _S1618 = struck_0(*b_59, &shot_11, kernelContext_216);
    bool _S1619;
    if(!_S1618)
    {
        _S1619 = true;
    }
    else
    {
        uint _S1620 = crush_flags_0(k_64, kernelContext_216);
        _S1619 = (_S1620 & 1U) != 0U;
    }
    if(_S1619)
    {
        return false;
    }
    uint bill_1 = kernelContext_216->crush_0->r_bill_0 + 4U * *b_59;
    if(((float4(*(kernelContext_216->crush_rw_0+(bill_1 + 3U))) ).w) == 0.0f)
    {
        return false;
    }
    *p1_0 = float4((float4(*(kernelContext_216->crush_rw_0+(bill_1 + 1U))) ).xyz, (float4(*(kernelContext_216->shots_0+2U * shot_11)) ).y);
    *p2_2 = float4(*(kernelContext_216->crush_rw_0+(bill_1 + 2U))) ;
    return true;
}

float3 cover_offset_0(uint s_28, float3 d_27, float a_12, float thread* rho_0)
{
    uint ring_2 = s_28 / 8U;
    uint sector_0 = s_28 % 8U;
    *rho_0 = a_12 * sqrt((float(ring_2) + 0.5f) / 8.0f);
    float theta_1 = 6.28318548202514648f * (float(sector_0) + 0.5f * float(ring_2 & 1U)) / 8.0f;
    float3 e_8;
    if((abs(d_27.x)) < 0.57735002040863037f)
    {
        e_8 = float3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_27.y)) < 0.57735002040863037f)
        {
            e_8 = float3(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_8 = float3(0.0f, 0.0f, 1.0f);
        }
    }
    float3 u_11 = normalize(cross(d_27, e_8));
    return float3(*rho_0)  * (float3(cos(theta_1))  * u_11 + float3(sin(theta_1))  * cross(d_27, u_11));
}

float cover_depth_0(uint k_65, uint s_29, float3 contact_20, float3 d_28, float a_13, float radius_2, KernelContext_0 thread* kernelContext_217)
{
    thread float rho_1;
    float _S1621 = 2.0f * radius_2;
    float3 _S1622 = contact_20 + cover_offset_0(s_29, d_28, a_13, &rho_1) - d_28 * float3(_S1621) ;
    thread Quat_0 q_33;
    float3 _S1623 = chunk_box_world_0(k_65, &q_33, kernelContext_217);
    uint _S1624 = 3U * k_65;
    Quat_0 _S1625 = quat_of_0(float4(*(kernelContext_217->geometry_0+(_S1624 + 2U))) );
    thread Quat_0 _S1626 = q_33;
    thread Quat_0 _S1627 = _S1625;
    Quat_0 _S1628 = quat_mul_0(&_S1626, &_S1627);
    thread Quat_0 _S1629 = _S1628;
    Quat_0 _S1630 = normalized_0(&_S1629);
    float3 _S1631 = (float4(*(kernelContext_217->geometry_0+(_S1624 + 1U))) ).xyz;
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
        thread float3 e_9 = float3(0.0f) ;
        e_9[ax_4] = 1.0f;
        thread Quat_0 _S1632 = _S1630;
        float3 _S1633 = rotate_0(&_S1632, e_9);
        float o_11 = dot(_S1622 - _S1623, _S1633);
        float dv_2 = dot(d_28, _S1633);
        if((abs(dv_2)) < 9.99999971718068537e-10f)
        {
            if((abs(o_11)) > (_S1631[ax_4]))
            {
                return 1.00000001504746622e+30f;
            }
            ax_4 = ax_4 + 1U;
            continue;
        }
        float ta_3 = (- _S1631[ax_4] - o_11) / dv_2;
        float tb_3 = (_S1631[ax_4] - o_11) / dv_2;
        float _S1634 = min(t1_4, max(ta_3, tb_3));
        t0_3 = max(t0_3, min(ta_3, tb_3));
        t1_4 = _S1634;
        ax_4 = ax_4 + 1U;
    }
    bool _S1635;
    if(!(t0_3 <= t1_4))
    {
        _S1635 = true;
    }
    else
    {
        _S1635 = t1_4 < 0.0f;
    }
    if(_S1635)
    {
        return 1.00000001504746622e+30f;
    }
    return t0_3 - _S1621;
}

[[kernel]] void fr_strike_cover(uint3 id_75 [[thread_position_in_grid]], FractureParams_0 constant* params_81 [[buffer(0)]], atomic<uint> device* atom_81 [[buffer(5)]], uint device* work_81 [[buffer(4)]], packed_float4 device* frames_81 [[buffer(6)]], packed_float4 device* bodies_81 [[buffer(7)]], packed_float4 device* chunks_81 [[buffer(1)]], uint device* sort_81 [[buffer(11)]], Record_natural_0 device* manifolds_81 [[buffer(10)]], packed_float4 device* loads_81 [[buffer(12)]], packed_uint4 device* bonds_82 [[buffer(2)]], uint device* args_81 [[buffer(13)]], uint device* info_81 [[buffer(15)]], ScanParams_0 constant* scan_81 [[buffer(14)]], packed_float4 device* parts_81 [[buffer(3)]], packed_float4 device* colliders_81 [[buffer(8)]], packed_uint4 device* collider_info_81 [[buffer(9)]], packed_float4 device* view_chunks_81 [[buffer(17)]], ViewParams_0 constant* view_81 [[buffer(16)]], packed_float4 device* objects_81 [[buffer(18)]], uint device* shot_bodies_81 [[buffer(20)]], float device* lines_81 [[buffer(19)]], packed_float4 device* geometry_81 [[buffer(21)]], packed_float4 device* shots_81 [[buffer(22)]], packed_float4 device* lp_81 [[buffer(24)]], packed_float4 device* ext_83 [[buffer(23)]], packed_float4 device* engine_contacts_81 [[buffer(25)]], CrushParams_0 constant* crush_81 [[buffer(28)]], packed_float4 device* crush_rw_81 [[buffer(27)]], packed_float4 device* crush_tab_81 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_218;
    (&kernelContext_218)->params_0 = params_81;
    (&kernelContext_218)->atom_0 = atom_81;
    (&kernelContext_218)->work_0 = work_81;
    (&kernelContext_218)->frames_0 = frames_81;
    (&kernelContext_218)->bodies_0 = bodies_81;
    (&kernelContext_218)->chunks_0 = chunks_81;
    (&kernelContext_218)->sort_0 = sort_81;
    (&kernelContext_218)->manifolds_0 = manifolds_81;
    (&kernelContext_218)->loads_0 = loads_81;
    (&kernelContext_218)->bonds_0 = bonds_82;
    (&kernelContext_218)->args_0 = args_81;
    (&kernelContext_218)->info_0 = info_81;
    (&kernelContext_218)->scan_0 = scan_81;
    (&kernelContext_218)->parts_0 = parts_81;
    (&kernelContext_218)->colliders_0 = colliders_81;
    (&kernelContext_218)->collider_info_0 = collider_info_81;
    (&kernelContext_218)->view_chunks_0 = view_chunks_81;
    (&kernelContext_218)->view_0 = view_81;
    (&kernelContext_218)->objects_0 = objects_81;
    (&kernelContext_218)->shot_bodies_0 = shot_bodies_81;
    (&kernelContext_218)->lines_0 = lines_81;
    (&kernelContext_218)->geometry_0 = geometry_81;
    (&kernelContext_218)->shots_0 = shots_81;
    (&kernelContext_218)->lp_0 = lp_81;
    (&kernelContext_218)->ext_0 = ext_83;
    (&kernelContext_218)->engine_contacts_0 = engine_contacts_81;
    (&kernelContext_218)->crush_0 = crush_81;
    (&kernelContext_218)->crush_rw_0 = crush_rw_81;
    (&kernelContext_218)->crush_tab_0 = crush_tab_81;
    threadgroup array<uint, int(1024)> g_scan_81;
    (&kernelContext_218)->g_scan_0 = &g_scan_81;
    threadgroup array<float, int(64)> g_il_best_81;
    (&kernelContext_218)->g_il_best_0 = &g_il_best_81;
    uint k_66 = id_75.x;
    bool _S1636;
    if(k_66 >= (params_81->chunk_count_0))
    {
        _S1636 = true;
    }
    else
    {
        _S1636 = ((&kernelContext_218)->params_0->shot_count_0) == 0U;
    }
    if(_S1636)
    {
        _S1636 = true;
    }
    else
    {
        _S1636 = ((&kernelContext_218)->crush_0->enabled_0) == 0U;
    }
    if(_S1636)
    {
        return;
    }
    thread uint b_60;
    thread float4 p1_1;
    thread float4 p2_3;
    bool _S1637 = cover_patch_0(k_66, &b_60, &p1_1, &p2_3, &kernelContext_218);
    if(!_S1637)
    {
        return;
    }
    uint _S1638 = a_cover_0(&kernelContext_218);
    uint _S1639 = _S1638 + 128U * b_60;
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
        float _S1640 = cover_depth_0(k_66, s_30, p1_1.xyz, p2_3.xyz, p2_3.w, p1_1.w, &kernelContext_218);
        if(_S1640 < 5.0000000752373311e+29f)
        {
            uint _S1641 = atomic_fetch_min_explicit((&kernelContext_218)->atom_0+(_S1639 + 2U * s_30), ordered_0(_S1640), memory_order_relaxed);
        }
        s_30 = s_30 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_claim(uint3 id_76 [[thread_position_in_grid]], FractureParams_0 constant* params_82 [[buffer(0)]], atomic<uint> device* atom_82 [[buffer(5)]], uint device* work_82 [[buffer(4)]], packed_float4 device* frames_82 [[buffer(6)]], packed_float4 device* bodies_82 [[buffer(7)]], packed_float4 device* chunks_82 [[buffer(1)]], uint device* sort_82 [[buffer(11)]], Record_natural_0 device* manifolds_82 [[buffer(10)]], packed_float4 device* loads_82 [[buffer(12)]], packed_uint4 device* bonds_83 [[buffer(2)]], uint device* args_82 [[buffer(13)]], uint device* info_82 [[buffer(15)]], ScanParams_0 constant* scan_82 [[buffer(14)]], packed_float4 device* parts_82 [[buffer(3)]], packed_float4 device* colliders_82 [[buffer(8)]], packed_uint4 device* collider_info_82 [[buffer(9)]], packed_float4 device* view_chunks_82 [[buffer(17)]], ViewParams_0 constant* view_82 [[buffer(16)]], packed_float4 device* objects_82 [[buffer(18)]], uint device* shot_bodies_82 [[buffer(20)]], float device* lines_82 [[buffer(19)]], packed_float4 device* geometry_82 [[buffer(21)]], packed_float4 device* shots_82 [[buffer(22)]], packed_float4 device* lp_82 [[buffer(24)]], packed_float4 device* ext_84 [[buffer(23)]], packed_float4 device* engine_contacts_82 [[buffer(25)]], CrushParams_0 constant* crush_82 [[buffer(28)]], packed_float4 device* crush_rw_82 [[buffer(27)]], packed_float4 device* crush_tab_82 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_219;
    (&kernelContext_219)->params_0 = params_82;
    (&kernelContext_219)->atom_0 = atom_82;
    (&kernelContext_219)->work_0 = work_82;
    (&kernelContext_219)->frames_0 = frames_82;
    (&kernelContext_219)->bodies_0 = bodies_82;
    (&kernelContext_219)->chunks_0 = chunks_82;
    (&kernelContext_219)->sort_0 = sort_82;
    (&kernelContext_219)->manifolds_0 = manifolds_82;
    (&kernelContext_219)->loads_0 = loads_82;
    (&kernelContext_219)->bonds_0 = bonds_83;
    (&kernelContext_219)->args_0 = args_82;
    (&kernelContext_219)->info_0 = info_82;
    (&kernelContext_219)->scan_0 = scan_82;
    (&kernelContext_219)->parts_0 = parts_82;
    (&kernelContext_219)->colliders_0 = colliders_82;
    (&kernelContext_219)->collider_info_0 = collider_info_82;
    (&kernelContext_219)->view_chunks_0 = view_chunks_82;
    (&kernelContext_219)->view_0 = view_82;
    (&kernelContext_219)->objects_0 = objects_82;
    (&kernelContext_219)->shot_bodies_0 = shot_bodies_82;
    (&kernelContext_219)->lines_0 = lines_82;
    (&kernelContext_219)->geometry_0 = geometry_82;
    (&kernelContext_219)->shots_0 = shots_82;
    (&kernelContext_219)->lp_0 = lp_82;
    (&kernelContext_219)->ext_0 = ext_84;
    (&kernelContext_219)->engine_contacts_0 = engine_contacts_82;
    (&kernelContext_219)->crush_0 = crush_82;
    (&kernelContext_219)->crush_rw_0 = crush_rw_82;
    (&kernelContext_219)->crush_tab_0 = crush_tab_82;
    threadgroup array<uint, int(1024)> g_scan_82;
    (&kernelContext_219)->g_scan_0 = &g_scan_82;
    threadgroup array<float, int(64)> g_il_best_82;
    (&kernelContext_219)->g_il_best_0 = &g_il_best_82;
    uint k_67 = id_76.x;
    bool _S1642;
    if(k_67 >= (params_82->chunk_count_0))
    {
        _S1642 = true;
    }
    else
    {
        _S1642 = ((&kernelContext_219)->params_0->shot_count_0) == 0U;
    }
    if(_S1642)
    {
        _S1642 = true;
    }
    else
    {
        _S1642 = ((&kernelContext_219)->crush_0->enabled_0) == 0U;
    }
    if(_S1642)
    {
        return;
    }
    thread uint b_61;
    thread float4 p1_2;
    thread float4 p2_4;
    bool _S1643 = cover_patch_0(k_67, &b_61, &p1_2, &p2_4, &kernelContext_219);
    if(!_S1643)
    {
        return;
    }
    uint _S1644 = a_cover_0(&kernelContext_219);
    uint _S1645 = _S1644 + 128U * b_61;
    uint s_31 = 0U;
    for(;;)
    {
        if(s_31 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S1646 = _S1645 + 2U * s_31;
        uint first_4 = atomic_load_explicit((&kernelContext_219)->atom_0+_S1646, memory_order_relaxed);
        if(first_4 == 4294967295U)
        {
            s_31 = s_31 + 1U;
            continue;
        }
        float _S1647 = cover_depth_0(k_67, s_31, p1_2.xyz, p2_4.xyz, p2_4.w, p1_2.w, &kernelContext_219);
        if(_S1647 < 5.0000000752373311e+29f)
        {
            _S1642 = _S1647 <= (unordered_0(first_4) + 0.00009999999747379f);
        }
        else
        {
            _S1642 = false;
        }
        if(_S1642)
        {
            uint _S1648 = atomic_fetch_min_explicit((&kernelContext_219)->atom_0+(_S1646 + 1U), k_67, memory_order_relaxed);
        }
        s_31 = s_31 + 1U;
    }
    return;
}

float strike_load_0(uint k_68, uint slot_6, float3 f_16, float3 at_24, float tau_8, float budget_0, KernelContext_0 thread* kernelContext_220)
{
    bool _S1649 = crushable_0(k_68, kernelContext_220);
    bool _S1650;
    if(!_S1649)
    {
        _S1650 = true;
    }
    else
    {
        uint _S1651 = crush_flags_0(k_68, kernelContext_220);
        _S1650 = (_S1651 & 1U) != 0U;
    }
    if(_S1650)
    {
        _S1650 = true;
    }
    else
    {
        _S1650 = !(budget_0 > 0.0f);
    }
    if(_S1650)
    {
        return 0.0f;
    }
    thread Quat_0 q_34;
    thread float3 origin_13;
    structure_pose_0(slot_6, 7U * slot_6, &q_34, &origin_13, kernelContext_220);
    float3 _S1652 = (float4(*(kernelContext_220->chunks_0+4U * k_68)) ).xyz;
    thread Quat_0 _S1653 = q_34;
    float3 _S1654 = rotate_0(&_S1653, _S1652);
    float3 centre_5 = origin_13 + _S1654;
    thread Quat_0 _S1655 = q_34;
    float4 _S1656 = bond_centroid_0(k_68, &_S1655, origin_13, centre_5, kernelContext_220);
    if(!((_S1656.w) > 0.0f))
    {
        return 0.0f;
    }
    Sym_0 _S1657 = sym_outer_0(_S1656.xyz, - f_16);
    thread Sym_0 _S1658 = sym_outer_0(at_24 - centre_5, f_16);
    thread Sym_0 _S1659 = _S1657;
    Sym_0 _S1660 = sym_add_0(&_S1658, &_S1659);
    float before_0 = (float4(*(kernelContext_220->crush_rw_0+k_68)) ).x;
    float _S1661 = min(tau_8, kernelContext_220->params_0->dt_0);
    thread Sym_0 _S1662 = _S1660;
    thread Sym_0 _S1663 = _S1660;
    float4 _S1664 = crush_two_0(&_S1662, _S1661, &_S1663, 0.0f, k_68, before_0, kernelContext_220);
    uint _S1665 = 2U * k_68;
    float per_0 = (float4(*(kernelContext_220->crush_tab_0+_S1665)) ).w * (float4(*(kernelContext_220->crush_tab_0+(_S1665 + 1U))) ).y;
    float energy_1 = (_S1664.x - before_0) * per_0;
    float energy_2;
    float4 step_8;
    if(energy_1 > budget_0)
    {
        float d_29 = before_0 + budget_0 / per_0;
        float _S1666 = min(d_29, 1.0f);
        float _S1667 = _S1664.y;
        float _S1668 = _S1664.z;
        if(d_29 >= 1.0f)
        {
            energy_2 = 1.0f;
        }
        else
        {
            energy_2 = 0.0f;
        }
        step_8 = float4(_S1666, _S1667, _S1668, energy_2);
        energy_2 = budget_0;
    }
    else
    {
        step_8 = _S1664;
        energy_2 = energy_1;
    }
    crush_record_0(k_68, step_8, 2U, kernelContext_220);
    return energy_2;
}

bool cell_in_0(uint2 mask_0, uint c_25)
{
    return (((mask_0[c_25 / 32U]) >> (c_25 % 32U)) & 1U) != 0U;
}

float strike_crush_body_0(uint b_62, uint s_32, float budget_1, KernelContext_0 thread* kernelContext_221)
{
    uint total_1;
    bool _S1669;
    uint bill_2 = kernelContext_221->crush_0->r_bill_0 + 4U * b_62;
    uint _S1670 = bill_2 + 1U;
    float4 _S1671 = float4(*(kernelContext_221->crush_rw_0+(bill_2 + 2U))) ;
    float4 _S1672 = float4(*(kernelContext_221->crush_rw_0+(bill_2 + 3U))) ;
    float3 contact_21 = (float4(*(kernelContext_221->crush_rw_0+_S1670)) ).xyz;
    float3 d_30 = _S1671.xyz;
    float _S1673 = _S1671.w;
    float _S1674 = (float4(*(kernelContext_221->shots_0+2U * s_32)) ).y;
    uint _S1675 = a_cover_0(kernelContext_221);
    uint _S1676 = _S1675 + 128U * b_62;
    float first_5 = 1.00000001504746622e+30f;
    uint c_26 = 0U;
    for(;;)
    {
        if(c_26 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S1677 = _S1676 + 2U * c_26;
        uint _S1678 = atomic_load_explicit(kernelContext_221->atom_0+(_S1677 + 1U), memory_order_relaxed);
        if(_S1678 == 4294967295U)
        {
            c_26 = c_26 + 1U;
            continue;
        }
        thread float rho_2;
        float3 _S1679 = cover_offset_0(c_26, d_30, _S1673, &rho_2);
        uint _S1680 = atomic_load_explicit(kernelContext_221->atom_0+_S1677, memory_order_relaxed);
        first_5 = min(first_5, unordered_0(_S1680) + _S1674 - sqrt(max(_S1674 * _S1674 - rho_2 * rho_2, 0.0f)));
        c_26 = c_26 + 1U;
    }
    thread uint2 counted_0 = uint2(0U) ;
    c_26 = 0U;
    uint total_2 = 0U;
    for(;;)
    {
        if(c_26 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S1681 = _S1676 + 2U * c_26;
        uint _S1682 = atomic_load_explicit(kernelContext_221->atom_0+(_S1681 + 1U), memory_order_relaxed);
        if(_S1682 == 4294967295U)
        {
            c_26 = c_26 + 1U;
            continue;
        }
        thread float rho_3;
        float3 _S1683 = cover_offset_0(c_26, d_30, _S1673, &rho_3);
        uint _S1684 = atomic_load_explicit(kernelContext_221->atom_0+_S1681, memory_order_relaxed);
        if((unordered_0(_S1684) + _S1674 - sqrt(max(_S1674 * _S1674 - rho_3 * rho_3, 0.0f)) - first_5) <= (_S1672.z))
        {
            counted_0[c_26 / 32U] = (counted_0[c_26 / 32U]) | (1U << (c_26 % 32U));
            total_1 = total_2 + 1U;
        }
        else
        {
            total_1 = total_2;
        }
        total_2 = total_1;
        c_26 = c_26 + 1U;
    }
    float3 _S1685 = float3(0.0f) ;
    float energy_3;
    float3 point_1;
    if(total_2 == 0U)
    {
        float _S1686 = strike_load_0(uint(_S1672.w) - 1U, b_62, d_30 * float3(_S1672.x) , contact_21, _S1672.y, budget_1, kernelContext_221);
        float3 _S1687 = float3(_S1686)  * contact_21;
        energy_3 = _S1686;
        point_1 = _S1687;
    }
    else
    {
        energy_3 = 0.0f;
        point_1 = _S1685;
    }
    c_26 = 0U;
    for(;;)
    {
        if(c_26 < 64U)
        {
        }
        else
        {
            break;
        }
        if(!cell_in_0(counted_0, c_26))
        {
            c_26 = c_26 + 1U;
            continue;
        }
        uint k_69 = atomic_load_explicit(kernelContext_221->atom_0+(_S1676 + 2U * c_26 + 1U), memory_order_relaxed);
        bool seen_0 = false;
        total_1 = 0U;
        for(;;)
        {
            if(total_1 < c_26)
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
                    uint _S1688 = atomic_load_explicit(kernelContext_221->atom_0+(_S1676 + 2U * total_1 + 1U), memory_order_relaxed);
                    _S1669 = _S1688 == k_69;
                }
                else
                {
                    _S1669 = false;
                }
                seen_0 = _S1669;
            }
            total_1 = total_1 + 1U;
        }
        if(seen_0)
        {
            c_26 = c_26 + 1U;
            continue;
        }
        uint e_10 = c_26;
        float3 at_25 = _S1685;
        uint n_6 = 0U;
        for(;;)
        {
            if(e_10 < 64U)
            {
            }
            else
            {
                break;
            }
            if(!cell_in_0(counted_0, e_10))
            {
                _S1669 = true;
            }
            else
            {
                uint _S1689 = atomic_load_explicit(kernelContext_221->atom_0+(_S1676 + 2U * e_10 + 1U), memory_order_relaxed);
                _S1669 = _S1689 != k_69;
            }
            if(_S1669)
            {
                e_10 = e_10 + 1U;
                continue;
            }
            thread float rho_4;
            float3 _S1690 = contact_21 + cover_offset_0(e_10, d_30, _S1673, &rho_4);
            uint _S1691 = atomic_load_explicit(kernelContext_221->atom_0+(_S1676 + 2U * e_10), memory_order_relaxed);
            uint _S1692 = n_6 + 1U;
            at_25 = at_25 + (_S1690 + d_30 * float3(unordered_0(_S1691)) );
            n_6 = _S1692;
            e_10 = e_10 + 1U;
        }
        float _S1693 = float(n_6);
        float3 at_26 = at_25 / float3(_S1693) ;
        float _S1694 = strike_load_0(k_69, b_62, d_30 * float3((_S1672.x * _S1693 / float(total_2))) , at_26, _S1672.y, budget_1 - energy_3, kernelContext_221);
        float3 point_2 = point_1 + float3(_S1694)  * at_26;
        energy_3 = energy_3 + _S1694;
        point_1 = point_2;
        c_26 = c_26 + 1U;
    }
    *(kernelContext_221->crush_rw_0+bill_2) = packed_float4(float4(energy_3, float(s_32 + 1U), 0.0f, 0.0f)) ;
    if(energy_3 > 0.0f)
    {
        *(kernelContext_221->crush_rw_0+_S1670) = packed_float4(float4(point_1 / float3(energy_3) , 0.0f)) ;
    }
    return energy_3;
}

[[kernel]] void fr_strike_crush(uint3 id_77 [[thread_position_in_grid]], FractureParams_0 constant* params_83 [[buffer(0)]], atomic<uint> device* atom_83 [[buffer(5)]], uint device* work_83 [[buffer(4)]], packed_float4 device* frames_83 [[buffer(6)]], packed_float4 device* bodies_83 [[buffer(7)]], packed_float4 device* chunks_83 [[buffer(1)]], uint device* sort_83 [[buffer(11)]], Record_natural_0 device* manifolds_83 [[buffer(10)]], packed_float4 device* loads_83 [[buffer(12)]], packed_uint4 device* bonds_84 [[buffer(2)]], uint device* args_83 [[buffer(13)]], uint device* info_83 [[buffer(15)]], ScanParams_0 constant* scan_83 [[buffer(14)]], packed_float4 device* parts_83 [[buffer(3)]], packed_float4 device* colliders_83 [[buffer(8)]], packed_uint4 device* collider_info_83 [[buffer(9)]], packed_float4 device* view_chunks_83 [[buffer(17)]], ViewParams_0 constant* view_83 [[buffer(16)]], packed_float4 device* objects_83 [[buffer(18)]], uint device* shot_bodies_83 [[buffer(20)]], float device* lines_83 [[buffer(19)]], packed_float4 device* geometry_83 [[buffer(21)]], packed_float4 device* shots_83 [[buffer(22)]], packed_float4 device* lp_83 [[buffer(24)]], packed_float4 device* ext_85 [[buffer(23)]], packed_float4 device* engine_contacts_83 [[buffer(25)]], CrushParams_0 constant* crush_83 [[buffer(28)]], packed_float4 device* crush_rw_83 [[buffer(27)]], packed_float4 device* crush_tab_83 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_222;
    (&kernelContext_222)->params_0 = params_83;
    (&kernelContext_222)->atom_0 = atom_83;
    (&kernelContext_222)->work_0 = work_83;
    (&kernelContext_222)->frames_0 = frames_83;
    (&kernelContext_222)->bodies_0 = bodies_83;
    (&kernelContext_222)->chunks_0 = chunks_83;
    (&kernelContext_222)->sort_0 = sort_83;
    (&kernelContext_222)->manifolds_0 = manifolds_83;
    (&kernelContext_222)->loads_0 = loads_83;
    (&kernelContext_222)->bonds_0 = bonds_84;
    (&kernelContext_222)->args_0 = args_83;
    (&kernelContext_222)->info_0 = info_83;
    (&kernelContext_222)->scan_0 = scan_83;
    (&kernelContext_222)->parts_0 = parts_83;
    (&kernelContext_222)->colliders_0 = colliders_83;
    (&kernelContext_222)->collider_info_0 = collider_info_83;
    (&kernelContext_222)->view_chunks_0 = view_chunks_83;
    (&kernelContext_222)->view_0 = view_83;
    (&kernelContext_222)->objects_0 = objects_83;
    (&kernelContext_222)->shot_bodies_0 = shot_bodies_83;
    (&kernelContext_222)->lines_0 = lines_83;
    (&kernelContext_222)->geometry_0 = geometry_83;
    (&kernelContext_222)->shots_0 = shots_83;
    (&kernelContext_222)->lp_0 = lp_83;
    (&kernelContext_222)->ext_0 = ext_85;
    (&kernelContext_222)->engine_contacts_0 = engine_contacts_83;
    (&kernelContext_222)->crush_0 = crush_83;
    (&kernelContext_222)->crush_rw_0 = crush_rw_83;
    (&kernelContext_222)->crush_tab_0 = crush_tab_83;
    threadgroup array<uint, int(1024)> g_scan_83;
    (&kernelContext_222)->g_scan_0 = &g_scan_83;
    threadgroup array<float, int(64)> g_il_best_83;
    (&kernelContext_222)->g_il_best_0 = &g_il_best_83;
    uint s_33 = id_77.x;
    bool _S1695;
    if(s_33 >= (params_83->shot_count_0))
    {
        _S1695 = true;
    }
    else
    {
        _S1695 = ((&kernelContext_222)->crush_0->enabled_0) == 0U;
    }
    if(_S1695)
    {
        return;
    }
    uint sb_4 = 7U * (as_type<uint>(((float4(*((&kernelContext_222)->shots_0+2U * s_33)) ).x)));
    float inv_mass_11 = (float4(*((&kernelContext_222)->bodies_0+sb_4)) ).w;
    float3 v_17 = (float4(*((&kernelContext_222)->bodies_0+(sb_4 + 3U))) ).xyz;
    float budget_2;
    if(inv_mass_11 > 0.0f)
    {
        budget_2 = 0.5f * dot(v_17, v_17) / inv_mass_11;
    }
    else
    {
        budget_2 = 1.00000001504746622e+30f;
    }
    uint b_63 = 0U;
    for(;;)
    {
        if(b_63 < ((&kernelContext_222)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        thread uint shot_12;
        bool _S1696 = struck_0(b_63, &shot_12, &kernelContext_222);
        if(!_S1696)
        {
            _S1695 = true;
        }
        else
        {
            _S1695 = shot_12 != s_33;
        }
        bool _S1697;
        if(_S1695)
        {
            _S1697 = true;
        }
        else
        {
            _S1697 = ((float4(*((&kernelContext_222)->crush_rw_0+((&kernelContext_222)->crush_0->r_bill_0 + 4U * b_63 + 3U))) ).w) == 0.0f;
        }
        if(_S1697)
        {
            b_63 = b_63 + 1U;
            continue;
        }
        float _S1698 = strike_crush_body_0(b_63, s_33, budget_2, &kernelContext_222);
        budget_2 = budget_2 - _S1698;
        b_63 = b_63 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bill(uint3 id_78 [[thread_position_in_grid]], FractureParams_0 constant* params_84 [[buffer(0)]], atomic<uint> device* atom_84 [[buffer(5)]], uint device* work_84 [[buffer(4)]], packed_float4 device* frames_84 [[buffer(6)]], packed_float4 device* bodies_84 [[buffer(7)]], packed_float4 device* chunks_84 [[buffer(1)]], uint device* sort_84 [[buffer(11)]], Record_natural_0 device* manifolds_84 [[buffer(10)]], packed_float4 device* loads_84 [[buffer(12)]], packed_uint4 device* bonds_85 [[buffer(2)]], uint device* args_84 [[buffer(13)]], uint device* info_84 [[buffer(15)]], ScanParams_0 constant* scan_84 [[buffer(14)]], packed_float4 device* parts_84 [[buffer(3)]], packed_float4 device* colliders_84 [[buffer(8)]], packed_uint4 device* collider_info_84 [[buffer(9)]], packed_float4 device* view_chunks_84 [[buffer(17)]], ViewParams_0 constant* view_84 [[buffer(16)]], packed_float4 device* objects_84 [[buffer(18)]], uint device* shot_bodies_84 [[buffer(20)]], float device* lines_84 [[buffer(19)]], packed_float4 device* geometry_84 [[buffer(21)]], packed_float4 device* shots_84 [[buffer(22)]], packed_float4 device* lp_84 [[buffer(24)]], packed_float4 device* ext_86 [[buffer(23)]], packed_float4 device* engine_contacts_84 [[buffer(25)]], CrushParams_0 constant* crush_84 [[buffer(28)]], packed_float4 device* crush_rw_84 [[buffer(27)]], packed_float4 device* crush_tab_84 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_223;
    (&kernelContext_223)->params_0 = params_84;
    (&kernelContext_223)->atom_0 = atom_84;
    (&kernelContext_223)->work_0 = work_84;
    (&kernelContext_223)->frames_0 = frames_84;
    (&kernelContext_223)->bodies_0 = bodies_84;
    (&kernelContext_223)->chunks_0 = chunks_84;
    (&kernelContext_223)->sort_0 = sort_84;
    (&kernelContext_223)->manifolds_0 = manifolds_84;
    (&kernelContext_223)->loads_0 = loads_84;
    (&kernelContext_223)->bonds_0 = bonds_85;
    (&kernelContext_223)->args_0 = args_84;
    (&kernelContext_223)->info_0 = info_84;
    (&kernelContext_223)->scan_0 = scan_84;
    (&kernelContext_223)->parts_0 = parts_84;
    (&kernelContext_223)->colliders_0 = colliders_84;
    (&kernelContext_223)->collider_info_0 = collider_info_84;
    (&kernelContext_223)->view_chunks_0 = view_chunks_84;
    (&kernelContext_223)->view_0 = view_84;
    (&kernelContext_223)->objects_0 = objects_84;
    (&kernelContext_223)->shot_bodies_0 = shot_bodies_84;
    (&kernelContext_223)->lines_0 = lines_84;
    (&kernelContext_223)->geometry_0 = geometry_84;
    (&kernelContext_223)->shots_0 = shots_84;
    (&kernelContext_223)->lp_0 = lp_84;
    (&kernelContext_223)->ext_0 = ext_86;
    (&kernelContext_223)->engine_contacts_0 = engine_contacts_84;
    (&kernelContext_223)->crush_0 = crush_84;
    (&kernelContext_223)->crush_rw_0 = crush_rw_84;
    (&kernelContext_223)->crush_tab_0 = crush_tab_84;
    threadgroup array<uint, int(1024)> g_scan_84;
    (&kernelContext_223)->g_scan_0 = &g_scan_84;
    threadgroup array<float, int(64)> g_il_best_84;
    (&kernelContext_223)->g_il_best_0 = &g_il_best_84;
    uint s_34 = id_78.x;
    bool _S1699;
    if(s_34 >= (params_84->shot_count_0))
    {
        _S1699 = true;
    }
    else
    {
        _S1699 = ((&kernelContext_223)->crush_0->enabled_0) == 0U;
    }
    if(_S1699)
    {
        return;
    }
    uint b_64 = 0U;
    float energy_4 = 0.0f;
    for(;;)
    {
        if(b_64 < ((&kernelContext_223)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S1700 = float4(*((&kernelContext_223)->crush_rw_0+((&kernelContext_223)->crush_0->r_bill_0 + 4U * b_64))) ;
        if((_S1700.y) == float(s_34 + 1U))
        {
            energy_4 = energy_4 + _S1700.x;
        }
        b_64 = b_64 + 1U;
    }
    *((&kernelContext_223)->crush_rw_0+((&kernelContext_223)->crush_0->r_shot_0 + s_34)) = packed_float4(float4(0.0f) ) ;
    if(!(energy_4 > 0.0f))
    {
        return;
    }
    uint sb_5 = 7U * (as_type<uint>(((float4(*((&kernelContext_223)->shots_0+2U * s_34)) ).x)));
    float inv_mass_12 = (float4(*((&kernelContext_223)->bodies_0+sb_5)) ).w;
    if(!(inv_mass_12 > 0.0f))
    {
        return;
    }
    float m_9 = 1.0f / inv_mass_12;
    float3 v_18 = (float4(*((&kernelContext_223)->bodies_0+(sb_5 + 3U))) ).xyz;
    float speed_16 = length(v_18);
    float j_15 = m_9 * (speed_16 - sqrt(max(speed_16 * speed_16 - 2.0f * energy_4 / m_9, 0.0f)));
    bool _S1701 = speed_16 > 0.0f;
    float3 _S1702;
    if(_S1701)
    {
        _S1702 = - v_18 * float3((j_15 / (m_9 * speed_16))) ;
    }
    else
    {
        _S1702 = float3(0.0f) ;
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
        uint w_12 = sb_5 + region_6 * 7U * (&kernelContext_223)->params_0->max_bodies_0 + 3U;
        *((&kernelContext_223)->bodies_0+w_12) = packed_float4(float4((float4(*((&kernelContext_223)->bodies_0+w_12)) ).xyz + _S1702, (float4(*((&kernelContext_223)->bodies_0+w_12)) ).w)) ;
        region_6 = region_6 + 1U;
    }
    packed_float4 device* _S1703 = (&kernelContext_223)->crush_rw_0+((&kernelContext_223)->crush_0->r_shot_0 + s_34);
    if(_S1701)
    {
        _S1702 = v_18 * float3((j_15 / speed_16)) ;
    }
    else
    {
        _S1702 = float3(0.0f) ;
    }
    *_S1703 = packed_float4(float4(_S1702, energy_4)) ;
    return;
}

float3 inverse_rotate_0(const Quat_0 thread* q_35, float3 v_19)
{
    thread Quat_0 c_27;
    (&c_27)->w_0 = q_35->w_0;
    (&c_27)->x_0 = - q_35->x_0;
    (&c_27)->y_0 = - q_35->y_0;
    (&c_27)->z_0 = - q_35->z_0;
    thread Quat_0 _S1704 = c_27;
    float3 _S1705 = rotate_0(&_S1704, v_19);
    return _S1705;
}

[[kernel]] void fr_crush_push(uint3 id_79 [[thread_position_in_grid]], FractureParams_0 constant* params_85 [[buffer(0)]], atomic<uint> device* atom_85 [[buffer(5)]], uint device* work_85 [[buffer(4)]], packed_float4 device* frames_85 [[buffer(6)]], packed_float4 device* bodies_85 [[buffer(7)]], packed_float4 device* chunks_85 [[buffer(1)]], uint device* sort_85 [[buffer(11)]], Record_natural_0 device* manifolds_85 [[buffer(10)]], packed_float4 device* loads_85 [[buffer(12)]], packed_uint4 device* bonds_86 [[buffer(2)]], uint device* args_85 [[buffer(13)]], uint device* info_85 [[buffer(15)]], ScanParams_0 constant* scan_85 [[buffer(14)]], packed_float4 device* parts_85 [[buffer(3)]], packed_float4 device* colliders_85 [[buffer(8)]], packed_uint4 device* collider_info_85 [[buffer(9)]], packed_float4 device* view_chunks_85 [[buffer(17)]], ViewParams_0 constant* view_85 [[buffer(16)]], packed_float4 device* objects_85 [[buffer(18)]], uint device* shot_bodies_85 [[buffer(20)]], float device* lines_85 [[buffer(19)]], packed_float4 device* geometry_85 [[buffer(21)]], packed_float4 device* shots_85 [[buffer(22)]], packed_float4 device* lp_85 [[buffer(24)]], packed_float4 device* ext_87 [[buffer(23)]], packed_float4 device* engine_contacts_85 [[buffer(25)]], CrushParams_0 constant* crush_85 [[buffer(28)]], packed_float4 device* crush_rw_85 [[buffer(27)]], packed_float4 device* crush_tab_85 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_224;
    (&kernelContext_224)->params_0 = params_85;
    (&kernelContext_224)->atom_0 = atom_85;
    (&kernelContext_224)->work_0 = work_85;
    (&kernelContext_224)->frames_0 = frames_85;
    (&kernelContext_224)->bodies_0 = bodies_85;
    (&kernelContext_224)->chunks_0 = chunks_85;
    (&kernelContext_224)->sort_0 = sort_85;
    (&kernelContext_224)->manifolds_0 = manifolds_85;
    (&kernelContext_224)->loads_0 = loads_85;
    (&kernelContext_224)->bonds_0 = bonds_86;
    (&kernelContext_224)->args_0 = args_85;
    (&kernelContext_224)->info_0 = info_85;
    (&kernelContext_224)->scan_0 = scan_85;
    (&kernelContext_224)->parts_0 = parts_85;
    (&kernelContext_224)->colliders_0 = colliders_85;
    (&kernelContext_224)->collider_info_0 = collider_info_85;
    (&kernelContext_224)->view_chunks_0 = view_chunks_85;
    (&kernelContext_224)->view_0 = view_85;
    (&kernelContext_224)->objects_0 = objects_85;
    (&kernelContext_224)->shot_bodies_0 = shot_bodies_85;
    (&kernelContext_224)->lines_0 = lines_85;
    (&kernelContext_224)->geometry_0 = geometry_85;
    (&kernelContext_224)->shots_0 = shots_85;
    (&kernelContext_224)->lp_0 = lp_85;
    (&kernelContext_224)->ext_0 = ext_87;
    (&kernelContext_224)->engine_contacts_0 = engine_contacts_85;
    (&kernelContext_224)->crush_0 = crush_85;
    (&kernelContext_224)->crush_rw_0 = crush_rw_85;
    (&kernelContext_224)->crush_tab_0 = crush_tab_85;
    threadgroup array<uint, int(1024)> g_scan_85;
    (&kernelContext_224)->g_scan_0 = &g_scan_85;
    threadgroup array<float, int(64)> g_il_best_85;
    (&kernelContext_224)->g_il_best_0 = &g_il_best_85;
    uint b_65 = id_79.x;
    bool _S1706;
    if(b_65 >= (params_85->body_count_0))
    {
        _S1706 = true;
    }
    else
    {
        _S1706 = ((&kernelContext_224)->params_0->shot_count_0) == 0U;
    }
    if(_S1706)
    {
        _S1706 = true;
    }
    else
    {
        _S1706 = ((&kernelContext_224)->crush_0->enabled_0) == 0U;
    }
    if(_S1706)
    {
        return;
    }
    uint _S1707 = 4U * b_65;
    float4 _S1708 = float4(*((&kernelContext_224)->crush_rw_0+((&kernelContext_224)->crush_0->r_bill_0 + _S1707))) ;
    float _S1709 = _S1708.x;
    if(!(_S1709 > 0.0f))
    {
        return;
    }
    float4 _S1710 = float4(*((&kernelContext_224)->crush_rw_0+((&kernelContext_224)->crush_0->r_shot_0 + (uint(_S1708.y) - 1U)))) ;
    uint bw_0 = 7U * b_65;
    float inv_mass_13 = (float4(*((&kernelContext_224)->bodies_0+bw_0)) ).w;
    float _S1711 = _S1710.w;
    if(!(_S1711 > 0.0f))
    {
        _S1706 = true;
    }
    else
    {
        _S1706 = inv_mass_13 == 0.0f;
    }
    if(_S1706)
    {
        return;
    }
    float3 j_16 = _S1710.xyz * float3((_S1709 / _S1711)) ;
    Quat_0 o_12 = quat_of_0(float4(*((&kernelContext_224)->bodies_0+(bw_0 + 1U))) );
    float3 inv_i_0 = (float4(*((&kernelContext_224)->bodies_0+(bw_0 + 2U))) ).xyz;
    float3 _S1712 = cross(to_rubble_0((float4(*((&kernelContext_224)->crush_rw_0+((&kernelContext_224)->crush_0->r_bill_0 + _S1707 + 1U))) ).xyz) - (float4(*((&kernelContext_224)->bodies_0+bw_0)) ).xyz, j_16);
    thread Quat_0 _S1713 = o_12;
    float3 _S1714 = inverse_rotate_0(&_S1713, _S1712);
    float3 _S1715 = inv_i_0 * _S1714;
    thread Quat_0 _S1716 = o_12;
    float3 _S1717 = rotate_0(&_S1716, _S1715);
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
        uint w_13 = bw_0 + region_7 * 7U * (&kernelContext_224)->params_0->max_bodies_0;
        uint _S1718 = w_13 + 3U;
        *((&kernelContext_224)->bodies_0+_S1718) = packed_float4(float4((float4(*((&kernelContext_224)->bodies_0+_S1718)) ).xyz + j_16 * float3(inv_mass_13) , (float4(*((&kernelContext_224)->bodies_0+_S1718)) ).w)) ;
        uint _S1719 = w_13 + 4U;
        *((&kernelContext_224)->bodies_0+_S1719) = packed_float4(float4((float4(*((&kernelContext_224)->bodies_0+_S1719)) ).xyz + _S1717, (float4(*((&kernelContext_224)->bodies_0+_S1719)) ).w)) ;
        region_7 = region_7 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bonds(uint3 id_80 [[thread_position_in_grid]], FractureParams_0 constant* params_86 [[buffer(0)]], atomic<uint> device* atom_86 [[buffer(5)]], uint device* work_86 [[buffer(4)]], packed_float4 device* frames_86 [[buffer(6)]], packed_float4 device* bodies_86 [[buffer(7)]], packed_float4 device* chunks_86 [[buffer(1)]], uint device* sort_86 [[buffer(11)]], Record_natural_0 device* manifolds_86 [[buffer(10)]], packed_float4 device* loads_86 [[buffer(12)]], packed_uint4 device* bonds_87 [[buffer(2)]], uint device* args_86 [[buffer(13)]], uint device* info_86 [[buffer(15)]], ScanParams_0 constant* scan_86 [[buffer(14)]], packed_float4 device* parts_86 [[buffer(3)]], packed_float4 device* colliders_86 [[buffer(8)]], packed_uint4 device* collider_info_86 [[buffer(9)]], packed_float4 device* view_chunks_86 [[buffer(17)]], ViewParams_0 constant* view_86 [[buffer(16)]], packed_float4 device* objects_86 [[buffer(18)]], uint device* shot_bodies_86 [[buffer(20)]], float device* lines_86 [[buffer(19)]], packed_float4 device* geometry_86 [[buffer(21)]], packed_float4 device* shots_86 [[buffer(22)]], packed_float4 device* lp_86 [[buffer(24)]], packed_float4 device* ext_88 [[buffer(23)]], packed_float4 device* engine_contacts_86 [[buffer(25)]], CrushParams_0 constant* crush_86 [[buffer(28)]], packed_float4 device* crush_rw_86 [[buffer(27)]], packed_float4 device* crush_tab_86 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_225;
    (&kernelContext_225)->params_0 = params_86;
    (&kernelContext_225)->atom_0 = atom_86;
    (&kernelContext_225)->work_0 = work_86;
    (&kernelContext_225)->frames_0 = frames_86;
    (&kernelContext_225)->bodies_0 = bodies_86;
    (&kernelContext_225)->chunks_0 = chunks_86;
    (&kernelContext_225)->sort_0 = sort_86;
    (&kernelContext_225)->manifolds_0 = manifolds_86;
    (&kernelContext_225)->loads_0 = loads_86;
    (&kernelContext_225)->bonds_0 = bonds_87;
    (&kernelContext_225)->args_0 = args_86;
    (&kernelContext_225)->info_0 = info_86;
    (&kernelContext_225)->scan_0 = scan_86;
    (&kernelContext_225)->parts_0 = parts_86;
    (&kernelContext_225)->colliders_0 = colliders_86;
    (&kernelContext_225)->collider_info_0 = collider_info_86;
    (&kernelContext_225)->view_chunks_0 = view_chunks_86;
    (&kernelContext_225)->view_0 = view_86;
    (&kernelContext_225)->objects_0 = objects_86;
    (&kernelContext_225)->shot_bodies_0 = shot_bodies_86;
    (&kernelContext_225)->lines_0 = lines_86;
    (&kernelContext_225)->geometry_0 = geometry_86;
    (&kernelContext_225)->shots_0 = shots_86;
    (&kernelContext_225)->lp_0 = lp_86;
    (&kernelContext_225)->ext_0 = ext_88;
    (&kernelContext_225)->engine_contacts_0 = engine_contacts_86;
    (&kernelContext_225)->crush_0 = crush_86;
    (&kernelContext_225)->crush_rw_0 = crush_rw_86;
    (&kernelContext_225)->crush_tab_0 = crush_tab_86;
    threadgroup array<uint, int(1024)> g_scan_86;
    (&kernelContext_225)->g_scan_0 = &g_scan_86;
    threadgroup array<float, int(64)> g_il_best_86;
    (&kernelContext_225)->g_il_best_0 = &g_il_best_86;
    uint i_25 = id_80.x;
    bool _S1720;
    if(i_25 >= (params_86->bond_count_0))
    {
        _S1720 = true;
    }
    else
    {
        _S1720 = ((&kernelContext_225)->crush_0->enabled_0) == 0U;
    }
    if(_S1720)
    {
        _S1720 = true;
    }
    else
    {
        uint _S1721 = w_alive_0(&kernelContext_225);
        _S1720 = (*((&kernelContext_225)->work_0+(_S1721 + i_25))) == 0U;
    }
    if(_S1720)
    {
        return;
    }
    uint4 _S1722 = uint4(*((&kernelContext_225)->bonds_0+i_25)) ;
    uint _S1723 = _S1722.x;
    if((((as_type<uint>(((float4(*((&kernelContext_225)->crush_rw_0+_S1723)) ).w))) | (as_type<uint>(((float4(*((&kernelContext_225)->crush_rw_0+_S1722.y)) ).w)))) & 2U) == 0U)
    {
        return;
    }
    uint _S1724 = w_alive_0(&kernelContext_225);
    *((&kernelContext_225)->work_0+(_S1724 + i_25)) = 0U;
    uint _S1725 = a_dirty_0(&kernelContext_225);
    uint _S1726 = body_of_0(_S1723, &kernelContext_225);
    uint _S1727 = atomic_fetch_or_explicit((&kernelContext_225)->atom_0+(_S1725 + _S1726), 1U, memory_order_relaxed);
    uint _S1728 = a_counters_1(&kernelContext_225);
    uint _S1729 = atomic_fetch_add_explicit((&kernelContext_225)->atom_0+_S1728, 1U, memory_order_relaxed);
    uint _S1730 = atomic_fetch_add_explicit((&kernelContext_225)->atom_0+(_S1728 + 1U), 1U, memory_order_relaxed);
    event_broken_0(i_25, &kernelContext_225);
    return;
}

uint crush_pieces_0(uint k_70, KernelContext_0 thread* kernelContext_226)
{
    if((kernelContext_226->crush_0->mode_1) == 1U)
    {
        return kernelContext_226->crush_0->fragments_0;
    }
    if((kernelContext_226->crush_0->mode_1) == 2U)
    {
        return uint((float4(*(kernelContext_226->crush_tab_0+(2U * k_70 + 1U))) ).w + 0.5f);
    }
    return 0U;
}

[[kernel]] void fr_crush_mark(uint3 id_81 [[thread_position_in_grid]], FractureParams_0 constant* params_87 [[buffer(0)]], atomic<uint> device* atom_87 [[buffer(5)]], uint device* work_87 [[buffer(4)]], packed_float4 device* frames_87 [[buffer(6)]], packed_float4 device* bodies_87 [[buffer(7)]], packed_float4 device* chunks_87 [[buffer(1)]], uint device* sort_87 [[buffer(11)]], Record_natural_0 device* manifolds_87 [[buffer(10)]], packed_float4 device* loads_87 [[buffer(12)]], packed_uint4 device* bonds_88 [[buffer(2)]], uint device* args_87 [[buffer(13)]], uint device* info_87 [[buffer(15)]], ScanParams_0 constant* scan_87 [[buffer(14)]], packed_float4 device* parts_87 [[buffer(3)]], packed_float4 device* colliders_87 [[buffer(8)]], packed_uint4 device* collider_info_87 [[buffer(9)]], packed_float4 device* view_chunks_87 [[buffer(17)]], ViewParams_0 constant* view_87 [[buffer(16)]], packed_float4 device* objects_87 [[buffer(18)]], uint device* shot_bodies_87 [[buffer(20)]], float device* lines_87 [[buffer(19)]], packed_float4 device* geometry_87 [[buffer(21)]], packed_float4 device* shots_87 [[buffer(22)]], packed_float4 device* lp_87 [[buffer(24)]], packed_float4 device* ext_89 [[buffer(23)]], packed_float4 device* engine_contacts_87 [[buffer(25)]], CrushParams_0 constant* crush_87 [[buffer(28)]], packed_float4 device* crush_rw_87 [[buffer(27)]], packed_float4 device* crush_tab_87 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_227;
    (&kernelContext_227)->params_0 = params_87;
    (&kernelContext_227)->atom_0 = atom_87;
    (&kernelContext_227)->work_0 = work_87;
    (&kernelContext_227)->frames_0 = frames_87;
    (&kernelContext_227)->bodies_0 = bodies_87;
    (&kernelContext_227)->chunks_0 = chunks_87;
    (&kernelContext_227)->sort_0 = sort_87;
    (&kernelContext_227)->manifolds_0 = manifolds_87;
    (&kernelContext_227)->loads_0 = loads_87;
    (&kernelContext_227)->bonds_0 = bonds_88;
    (&kernelContext_227)->args_0 = args_87;
    (&kernelContext_227)->info_0 = info_87;
    (&kernelContext_227)->scan_0 = scan_87;
    (&kernelContext_227)->parts_0 = parts_87;
    (&kernelContext_227)->colliders_0 = colliders_87;
    (&kernelContext_227)->collider_info_0 = collider_info_87;
    (&kernelContext_227)->view_chunks_0 = view_chunks_87;
    (&kernelContext_227)->view_0 = view_87;
    (&kernelContext_227)->objects_0 = objects_87;
    (&kernelContext_227)->shot_bodies_0 = shot_bodies_87;
    (&kernelContext_227)->lines_0 = lines_87;
    (&kernelContext_227)->geometry_0 = geometry_87;
    (&kernelContext_227)->shots_0 = shots_87;
    (&kernelContext_227)->lp_0 = lp_87;
    (&kernelContext_227)->ext_0 = ext_89;
    (&kernelContext_227)->engine_contacts_0 = engine_contacts_87;
    (&kernelContext_227)->crush_0 = crush_87;
    (&kernelContext_227)->crush_rw_0 = crush_rw_87;
    (&kernelContext_227)->crush_tab_0 = crush_tab_87;
    threadgroup array<uint, int(1024)> g_scan_87;
    (&kernelContext_227)->g_scan_0 = &g_scan_87;
    threadgroup array<float, int(64)> g_il_best_87;
    (&kernelContext_227)->g_il_best_0 = &g_il_best_87;
    uint c_28 = id_81.x;
    bool _S1731;
    if(c_28 >= (params_87->chunk_count_0))
    {
        _S1731 = true;
    }
    else
    {
        _S1731 = ((&kernelContext_227)->crush_0->enabled_0) == 0U;
    }
    if(_S1731)
    {
        return;
    }
    uint _S1732 = crush_flags_0(c_28, &kernelContext_227);
    uint now_0;
    if((_S1732 & 2U) != 0U)
    {
        now_0 = 1U;
    }
    else
    {
        now_0 = 0U;
    }
    uint _S1733 = w_new_0(&kernelContext_227);
    uint device* _S1734 = (&kernelContext_227)->work_0+(_S1733 + c_28);
    bool _S1735 = now_0 != 0U;
    if(_S1735)
    {
        uint _S1736 = crush_pieces_0(c_28, &kernelContext_227);
        now_0 = 1048576U | _S1736;
    }
    else
    {
        now_0 = 0U;
    }
    *_S1734 = now_0;
    if(_S1735)
    {
        uint _S1737 = a_counters_1(&kernelContext_227);
        uint _S1738 = atomic_fetch_add_explicit((&kernelContext_227)->atom_0+(_S1737 + 10U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_crush_gate(uint3 id_82 [[thread_position_in_grid]], FractureParams_0 constant* params_88 [[buffer(0)]], atomic<uint> device* atom_88 [[buffer(5)]], uint device* work_88 [[buffer(4)]], packed_float4 device* frames_88 [[buffer(6)]], packed_float4 device* bodies_88 [[buffer(7)]], packed_float4 device* chunks_88 [[buffer(1)]], uint device* sort_88 [[buffer(11)]], Record_natural_0 device* manifolds_88 [[buffer(10)]], packed_float4 device* loads_88 [[buffer(12)]], packed_uint4 device* bonds_89 [[buffer(2)]], uint device* args_88 [[buffer(13)]], uint device* info_88 [[buffer(15)]], ScanParams_0 constant* scan_88 [[buffer(14)]], packed_float4 device* parts_88 [[buffer(3)]], packed_float4 device* colliders_88 [[buffer(8)]], packed_uint4 device* collider_info_88 [[buffer(9)]], packed_float4 device* view_chunks_88 [[buffer(17)]], ViewParams_0 constant* view_88 [[buffer(16)]], packed_float4 device* objects_88 [[buffer(18)]], uint device* shot_bodies_88 [[buffer(20)]], float device* lines_88 [[buffer(19)]], packed_float4 device* geometry_88 [[buffer(21)]], packed_float4 device* shots_88 [[buffer(22)]], packed_float4 device* lp_88 [[buffer(24)]], packed_float4 device* ext_90 [[buffer(23)]], packed_float4 device* engine_contacts_88 [[buffer(25)]], CrushParams_0 constant* crush_88 [[buffer(28)]], packed_float4 device* crush_rw_88 [[buffer(27)]], packed_float4 device* crush_tab_88 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_228;
    (&kernelContext_228)->params_0 = params_88;
    (&kernelContext_228)->atom_0 = atom_88;
    (&kernelContext_228)->work_0 = work_88;
    (&kernelContext_228)->frames_0 = frames_88;
    (&kernelContext_228)->bodies_0 = bodies_88;
    (&kernelContext_228)->chunks_0 = chunks_88;
    (&kernelContext_228)->sort_0 = sort_88;
    (&kernelContext_228)->manifolds_0 = manifolds_88;
    (&kernelContext_228)->loads_0 = loads_88;
    (&kernelContext_228)->bonds_0 = bonds_89;
    (&kernelContext_228)->args_0 = args_88;
    (&kernelContext_228)->info_0 = info_88;
    (&kernelContext_228)->scan_0 = scan_88;
    (&kernelContext_228)->parts_0 = parts_88;
    (&kernelContext_228)->colliders_0 = colliders_88;
    (&kernelContext_228)->collider_info_0 = collider_info_88;
    (&kernelContext_228)->view_chunks_0 = view_chunks_88;
    (&kernelContext_228)->view_0 = view_88;
    (&kernelContext_228)->objects_0 = objects_88;
    (&kernelContext_228)->shot_bodies_0 = shot_bodies_88;
    (&kernelContext_228)->lines_0 = lines_88;
    (&kernelContext_228)->geometry_0 = geometry_88;
    (&kernelContext_228)->shots_0 = shots_88;
    (&kernelContext_228)->lp_0 = lp_88;
    (&kernelContext_228)->ext_0 = ext_90;
    (&kernelContext_228)->engine_contacts_0 = engine_contacts_88;
    (&kernelContext_228)->crush_0 = crush_88;
    (&kernelContext_228)->crush_rw_0 = crush_rw_88;
    (&kernelContext_228)->crush_tab_0 = crush_tab_88;
    threadgroup array<uint, int(1024)> g_scan_88;
    (&kernelContext_228)->g_scan_0 = &g_scan_88;
    threadgroup array<float, int(64)> g_il_best_88;
    (&kernelContext_228)->g_il_best_0 = &g_il_best_88;
    bool run_2;
    if((crush_88->enabled_0) != 0U)
    {
        uint _S1739 = a_counters_1(&kernelContext_228);
        uint _S1740 = atomic_load_explicit((&kernelContext_228)->atom_0+(_S1739 + 10U), memory_order_relaxed);
        run_2 = _S1740 != 0U;
    }
    else
    {
        run_2 = false;
    }
    uint _S1741 = a_counters_1(&kernelContext_228);
    atomic_store_explicit((&kernelContext_228)->atom_0+(_S1741 + 10U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_228)->atom_0+(_S1741 + 7U), 0U, memory_order_relaxed);
    uint _S1742;
    if(run_2)
    {
        uint _S1743 = groups_of_0((&kernelContext_228)->params_0->chunk_count_0, 64U);
        _S1742 = _S1743;
    }
    else
    {
        _S1742 = 0U;
    }
    set_args_0(14U, _S1742, &kernelContext_228);
    if(run_2)
    {
        uint _S1744 = groups_of_0((&kernelContext_228)->params_0->chunk_count_0, 1024U);
        _S1742 = _S1744;
    }
    else
    {
        _S1742 = 0U;
    }
    set_args_0(15U, _S1742, &kernelContext_228);
    int _S1745;
    if(run_2)
    {
        _S1745 = int(1);
    }
    else
    {
        _S1745 = int(0);
    }
    set_args_0(16U, uint(_S1745), &kernelContext_228);
    return;
}

void write_body_0(uint slot_7, float3 pos_1, float inv_mass_14, const Quat_0 thread* orientation_0, float3 inv_inertia_1, uint flags_3, float3 v_20, float3 w_14, KernelContext_0 thread* kernelContext_229)
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
        uint b_66 = 7U * (slot_7 + region_8 * kernelContext_229->params_0->max_bodies_0);
        *(kernelContext_229->bodies_0+b_66) = packed_float4(float4(pos_1, inv_mass_14)) ;
        packed_float4 device* _S1746 = kernelContext_229->bodies_0+(b_66 + 1U);
        float4 _S1747 = quat_vec_0(orientation_0);
        *_S1746 = packed_float4(_S1747) ;
        *(kernelContext_229->bodies_0+(b_66 + 2U)) = packed_float4(float4(inv_inertia_1, (as_type<float>((flags_3))))) ;
        *(kernelContext_229->bodies_0+(b_66 + 3U)) = packed_float4(float4(v_20, 0.0f)) ;
        packed_float4 device* _S1748 = kernelContext_229->bodies_0+(b_66 + 4U);
        uint _S1749 = frame_step_0(kernelContext_229);
        *_S1748 = packed_float4(float4(w_14, (as_type<float>((_S1749))))) ;
        *(kernelContext_229->bodies_0+(b_66 + 5U)) = packed_float4(float4(0.0f) ) ;
        *(kernelContext_229->bodies_0+(b_66 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
        region_8 = region_8 + 1U;
    }
    return;
}

float3 crush_particle_spot_0(uint k_71, uint i_26)
{
    uint h_7 = (k_71 * 2654435761U) ^ (i_26 * 2246822519U);
    uint h_8 = (h_7 ^ (h_7 >> 15U)) * 739982445U;
    uint h_9 = (h_8 ^ (h_8 >> 12U)) * 695872825U;
    uint h_10 = h_9 ^ (h_9 >> 15U);
    return float3(float(h_10 & 1023U), float((h_10 >> 10U) & 1023U), float((h_10 * 2654435761U + 2135587861U) & 1023U)) / float3(511.5f)  - float3(1.0f) ;
}

[[kernel]] void fr_crush_apply(uint3 id_83 [[thread_position_in_grid]], FractureParams_0 constant* params_89 [[buffer(0)]], atomic<uint> device* atom_89 [[buffer(5)]], uint device* work_89 [[buffer(4)]], packed_float4 device* frames_89 [[buffer(6)]], packed_float4 device* bodies_89 [[buffer(7)]], packed_float4 device* chunks_89 [[buffer(1)]], uint device* sort_89 [[buffer(11)]], Record_natural_0 device* manifolds_89 [[buffer(10)]], packed_float4 device* loads_89 [[buffer(12)]], packed_uint4 device* bonds_90 [[buffer(2)]], uint device* args_89 [[buffer(13)]], uint device* info_89 [[buffer(15)]], ScanParams_0 constant* scan_89 [[buffer(14)]], packed_float4 device* parts_89 [[buffer(3)]], packed_float4 device* colliders_89 [[buffer(8)]], packed_uint4 device* collider_info_89 [[buffer(9)]], packed_float4 device* view_chunks_89 [[buffer(17)]], ViewParams_0 constant* view_89 [[buffer(16)]], packed_float4 device* objects_89 [[buffer(18)]], uint device* shot_bodies_89 [[buffer(20)]], float device* lines_89 [[buffer(19)]], packed_float4 device* geometry_89 [[buffer(21)]], packed_float4 device* shots_89 [[buffer(22)]], packed_float4 device* lp_89 [[buffer(24)]], packed_float4 device* ext_91 [[buffer(23)]], packed_float4 device* engine_contacts_89 [[buffer(25)]], CrushParams_0 constant* crush_89 [[buffer(28)]], packed_float4 device* crush_rw_89 [[buffer(27)]], packed_float4 device* crush_tab_89 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_230;
    (&kernelContext_230)->params_0 = params_89;
    (&kernelContext_230)->atom_0 = atom_89;
    (&kernelContext_230)->work_0 = work_89;
    (&kernelContext_230)->frames_0 = frames_89;
    (&kernelContext_230)->bodies_0 = bodies_89;
    (&kernelContext_230)->chunks_0 = chunks_89;
    (&kernelContext_230)->sort_0 = sort_89;
    (&kernelContext_230)->manifolds_0 = manifolds_89;
    (&kernelContext_230)->loads_0 = loads_89;
    (&kernelContext_230)->bonds_0 = bonds_90;
    (&kernelContext_230)->args_0 = args_89;
    (&kernelContext_230)->info_0 = info_89;
    (&kernelContext_230)->scan_0 = scan_89;
    (&kernelContext_230)->parts_0 = parts_89;
    (&kernelContext_230)->colliders_0 = colliders_89;
    (&kernelContext_230)->collider_info_0 = collider_info_89;
    (&kernelContext_230)->view_chunks_0 = view_chunks_89;
    (&kernelContext_230)->view_0 = view_89;
    (&kernelContext_230)->objects_0 = objects_89;
    (&kernelContext_230)->shot_bodies_0 = shot_bodies_89;
    (&kernelContext_230)->lines_0 = lines_89;
    (&kernelContext_230)->geometry_0 = geometry_89;
    (&kernelContext_230)->shots_0 = shots_89;
    (&kernelContext_230)->lp_0 = lp_89;
    (&kernelContext_230)->ext_0 = ext_91;
    (&kernelContext_230)->engine_contacts_0 = engine_contacts_89;
    (&kernelContext_230)->crush_0 = crush_89;
    (&kernelContext_230)->crush_rw_0 = crush_rw_89;
    (&kernelContext_230)->crush_tab_0 = crush_tab_89;
    threadgroup array<uint, int(1024)> g_scan_89;
    (&kernelContext_230)->g_scan_0 = &g_scan_89;
    threadgroup array<float, int(64)> g_il_best_89;
    (&kernelContext_230)->g_il_best_0 = &g_il_best_89;
    uint k_72 = id_83.x;
    bool _S1750;
    if(k_72 >= (params_89->chunk_count_0))
    {
        _S1750 = true;
    }
    else
    {
        _S1750 = ((&kernelContext_230)->crush_0->enabled_0) == 0U;
    }
    if(_S1750)
    {
        return;
    }
    float4 _S1751 = float4(*((&kernelContext_230)->crush_rw_0+k_72)) ;
    uint flags_4 = (as_type<uint>((_S1751.w)));
    if((flags_4 & 2U) == 0U)
    {
        return;
    }
    *((&kernelContext_230)->crush_rw_0+k_72) = packed_float4(float4(_S1751.xyz, (as_type<float>((flags_4 & 4294967293U))))) ;
    uint _S1752 = body_of_0(k_72, &kernelContext_230);
    uint bw_1 = 7U * _S1752;
    float4 _S1753 = float4(*((&kernelContext_230)->bodies_0+bw_1)) ;
    Quat_0 orientation_1 = quat_of_0(float4(*((&kernelContext_230)->bodies_0+(bw_1 + 1U))) );
    float3 v_21 = (float4(*((&kernelContext_230)->bodies_0+(bw_1 + 3U))) ).xyz;
    float3 w_15 = (float4(*((&kernelContext_230)->bodies_0+(bw_1 + 4U))) ).xyz;
    uint _S1754 = 4U * k_72;
    float mass_1 = (float4(*((&kernelContext_230)->chunks_0+_S1754)) ).w;
    float volume_1 = (float4(*((&kernelContext_230)->crush_tab_0+(2U * k_72 + 1U))) ).y;
    uint _S1755 = w_rank_0(&kernelContext_230);
    uint device* _S1756 = (&kernelContext_230)->work_0+(_S1755 + k_72);
    uint scanned_0 = *_S1756;
    uint rank_0 = *_S1756 / 1048576U;
    uint _S1757 = crush_pieces_0(k_72, &kernelContext_230);
    if(rank_0 >= ((&kernelContext_230)->crush_0->max_events_0))
    {
        uint _S1758 = a_counters_1(&kernelContext_230);
        uint _S1759 = atomic_fetch_or_explicit((&kernelContext_230)->atom_0+(_S1758 + 3U), 32U, memory_order_relaxed);
    }
    float3 _S1760 = float3(0.0f) ;
    if(((&kernelContext_230)->crush_0->mode_1) == 1U)
    {
        _S1750 = ((&kernelContext_230)->crush_0->fragments_0) > 0U;
    }
    else
    {
        _S1750 = false;
    }
    uint first_6;
    uint n_7;
    uint i_27;
    float3 half_2;
    if(_S1750)
    {
        uint _S1761 = a_counters_1(&kernelContext_230);
        uint _S1762 = atomic_load_explicit((&kernelContext_230)->atom_0+(_S1761 + 8U), memory_order_relaxed);
        uint base_4 = _S1762 + (scanned_0 & 1048575U);
        if((base_4 + (&kernelContext_230)->crush_0->fragments_0) > ((&kernelContext_230)->crush_0->pool_size_0))
        {
            uint _S1763 = atomic_fetch_or_explicit((&kernelContext_230)->atom_0+(_S1761 + 3U), 16U, memory_order_relaxed);
            first_6 = 4294967295U;
        }
        else
        {
            first_6 = base_4;
        }
        thread Quat_0 q_36;
        thread float3 origin_14;
        structure_pose_0(_S1752, bw_1, &q_36, &origin_14, &kernelContext_230);
        float3 _S1764 = (float4(*((&kernelContext_230)->chunks_0+_S1754)) ).xyz;
        thread Quat_0 _S1765 = q_36;
        float3 _S1766 = rotate_0(&_S1765, _S1764);
        float3 _S1767 = origin_14 + _S1766;
        uint _S1768 = 3U * k_72;
        Quat_0 _S1769 = quat_of_0(float4(*((&kernelContext_230)->geometry_0+(_S1768 + 2U))) );
        thread Quat_0 _S1770 = q_36;
        thread Quat_0 _S1771 = _S1769;
        Quat_0 _S1772 = quat_mul_0(&_S1770, &_S1771);
        thread Quat_0 _S1773 = _S1772;
        Quat_0 _S1774 = normalized_0(&_S1773);
        float3 h_11 = (float4(*((&kernelContext_230)->geometry_0+(_S1768 + 1U))) ).xyz;
        thread uint3 g_0 = uint3(int3(int(1)) );
        thread float3 cell_0 = h_11;
        n_7 = (&kernelContext_230)->crush_0->fragments_0;
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
                _S1750 = (cell_0.x) >= (cell_0.z);
            }
            else
            {
                _S1750 = false;
            }
            if(_S1750)
            {
                i_27 = 0U;
            }
            else
            {
                if((cell_0.y) >= (cell_0.z))
                {
                    i_27 = 1U;
                }
                else
                {
                    i_27 = 2U;
                }
            }
            g_0[i_27] = g_0[i_27] * 2U;
            cell_0[i_27] = cell_0[i_27] * 0.5f;
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
        float mf_0 = mass_1 / float((&kernelContext_230)->crush_0->fragments_0);
        float _S1775 = half_3.y;
        float _S1776 = _S1775 * _S1775;
        float _S1777 = half_3.z;
        float _S1778 = _S1777 * _S1777;
        float _S1779 = half_3.x;
        float _S1780 = _S1779 * _S1779;
        float3 inertia_0 = float3((mf_0 / 3.0f))  * float3(_S1776 + _S1778, _S1780 + _S1778, _S1780 + _S1776);
        float3 _S1781 = float3(1.0f / inertia_0.x, 1.0f / inertia_0.y, 1.0f / inertia_0.z);
        thread Quat_0 _S1782 = _S1774;
        Quat_0 _S1783 = rot_to_rubble_0(&_S1782);
        float _S1784 = (float4(*((&kernelContext_230)->chunks_0+(_S1754 + 3U))) ).x;
        float _S1785 = min(_S1779, min(_S1775, _S1777));
        i_27 = 0U;
        for(;;)
        {
            if(i_27 < ((&kernelContext_230)->crush_0->fragments_0))
            {
                _S1750 = first_6 != 4294967295U;
            }
            else
            {
                _S1750 = false;
            }
            if(_S1750)
            {
            }
            else
            {
                break;
            }
            uint _S1786 = i_27 % g_0.x;
            uint _S1787 = i_27 / g_0.x;
            uint _S1788 = _S1787 % g_0.y;
            uint _S1789 = i_27 / (g_0.x * g_0.y);
            float3 offset_0 = - h_11 + cell_0 * (float3(2.0f)  * float3(uint3(_S1786, _S1788, _S1789)) + float3(1.0f) );
            thread Quat_0 _S1790 = _S1774;
            float3 _S1791 = rotate_0(&_S1790, offset_0);
            float3 x_8 = to_rubble_0(_S1767 + _S1791);
            float4 _S1792 = float4(*((&kernelContext_230)->crush_tab_0+((&kernelContext_230)->crush_0->t_pool_0 + first_6 + i_27))) ;
            uint fs_0 = (as_type<uint>((_S1792.x)));
            uint fc_3 = (as_type<uint>((_S1792.y)));
            float _S1793 = 1.0f / mf_0;
            float3 _S1794 = v_21 + cross(w_15, x_8 - _S1753.xyz);
            thread Quat_0 _S1795 = _S1783;
            write_body_0(fs_0, x_8, _S1793, &_S1795, _S1781, 9U, _S1794, w_15, &kernelContext_230);
            uint _S1796 = 4U * fc_3;
            uint _S1797 = _S1796 + 1U;
            *((&kernelContext_230)->colliders_0+_S1797) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_230)->colliders_0+_S1797)) ).w)) ;
            *((&kernelContext_230)->colliders_0+(_S1796 + 2U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
            *((&kernelContext_230)->colliders_0+(_S1796 + 3U)) = packed_float4(float4(half_3, length(half_3))) ;
            uint _S1798 = 2U * fc_3;
            *((&kernelContext_230)->collider_info_0+_S1798) = packed_uint4(uint4(4294967295U, fs_0, 2U, (uint4(*((&kernelContext_230)->collider_info_0+_S1798)) ).w)) ;
            *((&kernelContext_230)->collider_info_0+(_S1798 + 1U)) = packed_uint4(uint4((as_type<uint>((mf_0))), (as_type<uint>((_S1784))), (as_type<uint>((_S1785))), 0U)) ;
            i_27 = i_27 + 1U;
        }
        half_2 = half_3;
    }
    else
    {
        first_6 = 4294967295U;
        half_2 = _S1760;
    }
    if(((&kernelContext_230)->crush_0->mode_1) == 2U)
    {
        _S1750 = _S1757 > 0U;
    }
    else
    {
        _S1750 = false;
    }
    if(_S1750)
    {
        uint _S1799 = a_counters_1(&kernelContext_230);
        uint _S1800 = atomic_load_explicit((&kernelContext_230)->atom_0+(_S1799 + 8U), memory_order_relaxed);
        uint base_5 = _S1800 + (scanned_0 & 1048575U);
        if((base_5 + _S1757) > ((&kernelContext_230)->crush_0->pool_size_0))
        {
            uint _S1801 = atomic_fetch_or_explicit((&kernelContext_230)->atom_0+(_S1799 + 3U), 16U, memory_order_relaxed);
        }
        else
        {
            first_6 = base_5;
        }
        thread Quat_0 q_37;
        thread float3 origin_15;
        structure_pose_0(_S1752, bw_1, &q_37, &origin_15, &kernelContext_230);
        float3 _S1802 = (float4(*((&kernelContext_230)->chunks_0+_S1754)) ).xyz;
        thread Quat_0 _S1803 = q_37;
        float3 _S1804 = rotate_0(&_S1803, _S1802);
        float3 _S1805 = origin_15 + _S1804;
        uint _S1806 = 3U * k_72;
        Quat_0 _S1807 = quat_of_0(float4(*((&kernelContext_230)->geometry_0+(_S1806 + 2U))) );
        thread Quat_0 _S1808 = q_37;
        thread Quat_0 _S1809 = _S1807;
        Quat_0 _S1810 = quat_mul_0(&_S1808, &_S1809);
        thread Quat_0 _S1811 = _S1810;
        Quat_0 _S1812 = normalized_0(&_S1811);
        float3 h_12 = (float4(*((&kernelContext_230)->geometry_0+(_S1806 + 1U))) ).xyz;
        float _S1813 = float(_S1757);
        float _S1814 = max(pow(0.75f * volume_1 / (3.14159274101257324f * _S1813), 0.3333333432674408f), 0.0020000000949949f);
        float3 _S1815 = float3(_S1814, _S1814, _S1814);
        float _S1816 = mass_1 / _S1813;
        float _S1817 = (float4(*((&kernelContext_230)->chunks_0+(_S1754 + 3U))) ).x;
        float3 _S1818 = max(h_12 - float3(_S1814) , float3(0.25f)  * h_12);
        i_27 = 0U;
        float3 mean_4 = _S1760;
        for(;;)
        {
            if(i_27 < _S1757)
            {
            }
            else
            {
                break;
            }
            float3 mean_5 = mean_4 + crush_particle_spot_0(k_72, i_27);
            i_27 = i_27 + 1U;
            mean_4 = mean_5;
        }
        float3 mean_6 = mean_4 / float3(_S1813) ;
        i_27 = 0U;
        for(;;)
        {
            if(i_27 < _S1757)
            {
                _S1750 = first_6 != 4294967295U;
            }
            else
            {
                _S1750 = false;
            }
            if(_S1750)
            {
            }
            else
            {
                break;
            }
            float3 _S1819 = _S1818 * (crush_particle_spot_0(k_72, i_27) - mean_6);
            thread Quat_0 _S1820 = _S1812;
            float3 _S1821 = rotate_0(&_S1820, _S1819);
            float3 x_9 = to_rubble_0(_S1805 + _S1821);
            float4 _S1822 = float4(*((&kernelContext_230)->crush_tab_0+((&kernelContext_230)->crush_0->t_pool_0 + first_6 + i_27))) ;
            uint fs_1 = (as_type<uint>((_S1822.x)));
            uint fc_4 = (as_type<uint>((_S1822.y)));
            float _S1823 = 1.0f / _S1816;
            float4 _S1824 = float4(0.0f, 0.0f, 0.0f, 1.0f);
            Quat_0 _S1825 = quat_of_0(_S1824);
            if(((&kernelContext_230)->crush_0->particle_ccd_0) != 0U)
            {
                n_7 = 33024U;
            }
            else
            {
                n_7 = 0U;
            }
            uint _S1826 = 9U | n_7;
            float3 _S1827 = v_21 + cross(w_15, x_9 - _S1753.xyz);
            thread Quat_0 _S1828 = _S1825;
            write_body_0(fs_1, x_9, _S1823, &_S1828, _S1760, _S1826, _S1827, _S1760, &kernelContext_230);
            uint _S1829 = 4U * fc_4;
            uint _S1830 = _S1829 + 1U;
            *((&kernelContext_230)->colliders_0+_S1830) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_230)->colliders_0+_S1830)) ).w)) ;
            *((&kernelContext_230)->colliders_0+(_S1829 + 2U)) = packed_float4(_S1824) ;
            *((&kernelContext_230)->colliders_0+(_S1829 + 3U)) = packed_float4(float4(_S1814, 0.0f, 0.0f, _S1814)) ;
            uint _S1831 = 2U * fc_4;
            *((&kernelContext_230)->collider_info_0+_S1831) = packed_uint4(uint4(4294967295U, fs_1, 2U, (uint4(*((&kernelContext_230)->collider_info_0+_S1831)) ).w)) ;
            *((&kernelContext_230)->collider_info_0+(_S1831 + 1U)) = packed_uint4(uint4((as_type<uint>((_S1816))), (as_type<uint>((_S1817))), (as_type<uint>((_S1814))), 0U)) ;
            i_27 = i_27 + 1U;
        }
        half_2 = _S1815;
    }
    if(rank_0 < ((&kernelContext_230)->crush_0->max_events_0))
    {
        uint ev_0 = (&kernelContext_230)->crush_0->r_events_0 + 6U * rank_0;
        *((&kernelContext_230)->crush_rw_0+ev_0) = packed_float4(float4((as_type<float>((k_72))), (as_type<float>((first_6))), (as_type<float>((flags_4 >> 4U))), _S1751.x)) ;
        *((&kernelContext_230)->crush_rw_0+(ev_0 + 1U)) = packed_float4(float4(_S1753.xyz, mass_1)) ;
        packed_float4 device* _S1832 = (&kernelContext_230)->crush_rw_0+(ev_0 + 2U);
        thread Quat_0 _S1833 = orientation_1;
        float4 _S1834 = quat_vec_0(&_S1833);
        *_S1832 = packed_float4(_S1834) ;
        *((&kernelContext_230)->crush_rw_0+(ev_0 + 3U)) = packed_float4(float4(v_21, _S1751.y)) ;
        *((&kernelContext_230)->crush_rw_0+(ev_0 + 4U)) = packed_float4(float4(w_15, _S1751.z)) ;
        *((&kernelContext_230)->crush_rw_0+(ev_0 + 5U)) = packed_float4(float4(half_2, volume_1)) ;
    }
    float3 park_0 = float3(float(_S1752 % 1024U) * (&kernelContext_230)->crush_0->park_step_0, (&kernelContext_230)->crush_0->park_y_0, float(_S1752 / 1024U) * (&kernelContext_230)->crush_0->park_step_0);
    thread Quat_0 _S1835 = orientation_1;
    write_body_0(_S1752, park_0, 0.0f, &_S1835, _S1760, 0U, _S1760, _S1760, &kernelContext_230);
    return;
}

[[kernel]] void fr_crush_end(uint3 id_84 [[thread_position_in_grid]], FractureParams_0 constant* params_90 [[buffer(0)]], atomic<uint> device* atom_90 [[buffer(5)]], uint device* work_90 [[buffer(4)]], packed_float4 device* frames_90 [[buffer(6)]], packed_float4 device* bodies_90 [[buffer(7)]], packed_float4 device* chunks_90 [[buffer(1)]], uint device* sort_90 [[buffer(11)]], Record_natural_0 device* manifolds_90 [[buffer(10)]], packed_float4 device* loads_90 [[buffer(12)]], packed_uint4 device* bonds_91 [[buffer(2)]], uint device* args_90 [[buffer(13)]], uint device* info_90 [[buffer(15)]], ScanParams_0 constant* scan_90 [[buffer(14)]], packed_float4 device* parts_90 [[buffer(3)]], packed_float4 device* colliders_90 [[buffer(8)]], packed_uint4 device* collider_info_90 [[buffer(9)]], packed_float4 device* view_chunks_90 [[buffer(17)]], ViewParams_0 constant* view_90 [[buffer(16)]], packed_float4 device* objects_90 [[buffer(18)]], uint device* shot_bodies_90 [[buffer(20)]], float device* lines_90 [[buffer(19)]], packed_float4 device* geometry_90 [[buffer(21)]], packed_float4 device* shots_90 [[buffer(22)]], packed_float4 device* lp_90 [[buffer(24)]], packed_float4 device* ext_92 [[buffer(23)]], packed_float4 device* engine_contacts_90 [[buffer(25)]], CrushParams_0 constant* crush_90 [[buffer(28)]], packed_float4 device* crush_rw_90 [[buffer(27)]], packed_float4 device* crush_tab_90 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_231;
    (&kernelContext_231)->params_0 = params_90;
    (&kernelContext_231)->atom_0 = atom_90;
    (&kernelContext_231)->work_0 = work_90;
    (&kernelContext_231)->frames_0 = frames_90;
    (&kernelContext_231)->bodies_0 = bodies_90;
    (&kernelContext_231)->chunks_0 = chunks_90;
    (&kernelContext_231)->sort_0 = sort_90;
    (&kernelContext_231)->manifolds_0 = manifolds_90;
    (&kernelContext_231)->loads_0 = loads_90;
    (&kernelContext_231)->bonds_0 = bonds_91;
    (&kernelContext_231)->args_0 = args_90;
    (&kernelContext_231)->info_0 = info_90;
    (&kernelContext_231)->scan_0 = scan_90;
    (&kernelContext_231)->parts_0 = parts_90;
    (&kernelContext_231)->colliders_0 = colliders_90;
    (&kernelContext_231)->collider_info_0 = collider_info_90;
    (&kernelContext_231)->view_chunks_0 = view_chunks_90;
    (&kernelContext_231)->view_0 = view_90;
    (&kernelContext_231)->objects_0 = objects_90;
    (&kernelContext_231)->shot_bodies_0 = shot_bodies_90;
    (&kernelContext_231)->lines_0 = lines_90;
    (&kernelContext_231)->geometry_0 = geometry_90;
    (&kernelContext_231)->shots_0 = shots_90;
    (&kernelContext_231)->lp_0 = lp_90;
    (&kernelContext_231)->ext_0 = ext_92;
    (&kernelContext_231)->engine_contacts_0 = engine_contacts_90;
    (&kernelContext_231)->crush_0 = crush_90;
    (&kernelContext_231)->crush_rw_0 = crush_rw_90;
    (&kernelContext_231)->crush_tab_0 = crush_tab_90;
    threadgroup array<uint, int(1024)> g_scan_90;
    (&kernelContext_231)->g_scan_0 = &g_scan_90;
    threadgroup array<float, int(64)> g_il_best_90;
    (&kernelContext_231)->g_il_best_0 = &g_il_best_90;
    uint blocks_2 = (params_90->chunk_count_0 + 1023U) / 1024U;
    uint _S1836 = scan_new_blocks_0(&kernelContext_231);
    uint device* _S1837 = (&kernelContext_231)->work_0+(_S1836 + blocks_2);
    uint total_3 = *_S1837;
    uint events_0 = *_S1837 / 1048576U;
    uint _S1838 = a_counters_1(&kernelContext_231);
    atomic_store_explicit((&kernelContext_231)->atom_0+(_S1838 + 7U), events_0, memory_order_relaxed);
    uint _S1839 = atomic_fetch_add_explicit((&kernelContext_231)->atom_0+(_S1838 + 9U), events_0, memory_order_relaxed);
    uint _S1840 = atomic_fetch_add_explicit((&kernelContext_231)->atom_0+(_S1838 + 8U), total_3 & 1048575U, memory_order_relaxed);
    return;
}

