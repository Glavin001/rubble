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
    float strike_radius_0;
    uint gi_mode_0;
    float gi_scale_0;
    uint gi_sweeps_0;
    uint gi_pad_0;
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
    array<float, int(256)> threadgroup* g_gi_x_0;
    array<float, int(256)> threadgroup* g_gi_m_0;
    array<uint, int(1280)> threadgroup* g_gi_lo_0;
    array<uint, int(1280)> threadgroup* g_gi_hi_0;
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

bool dirty_0(uint body_1, KernelContext_0 thread* kernelContext_75)
{
    uint _S450 = a_dirty_0(kernelContext_75);
    uint _S451 = atomic_load_explicit(kernelContext_75->atom_0+(_S450 + body_1), memory_order_relaxed);
    return _S451 != 0U;
}

uint w_cs_list_0(KernelContext_0 thread* kernelContext_76)
{
    uint _S452 = w_carry_0(kernelContext_76);
    return _S452 + kernelContext_76->params_0->chunk_count_0;
}

uint w_cs_bodies_0(KernelContext_0 thread* kernelContext_77)
{
    uint _S453 = w_cs_list_0(kernelContext_77);
    return _S453 + kernelContext_77->params_0->chunk_count_0;
}

void gi_record_0(uint b_27, uint entry_2, float3 contact_14, float3 d_16, float v_9, float j_2, float tau_1, float local_0, KernelContext_0 thread* kernelContext_78)
{
    bool _S454 = dirty_0(b_27, kernelContext_78);
    if(_S454)
    {
        return;
    }
    uint _S455 = lp_body_0(b_27, kernelContext_78);
    *(kernelContext_78->lp_0+(_S455 + 2U)) = packed_float4(float4(contact_14, 3.14159274101257324f * j_2 / (2.0f * max(tau_1, 9.99999971718068537e-10f)))) ;
    *(kernelContext_78->lp_0+(_S455 + 3U)) = packed_float4(float4(d_16, v_9)) ;
    *(kernelContext_78->lp_0+(_S455 + 4U)) = packed_float4(float4(local_0, 0.0f, 0.0f, 0.0f)) ;
    *(kernelContext_78->lp_0+(_S455 + 6U)) = packed_float4(float4(0.0f, 0.0f, tau_1, 0.0f)) ;
    uint _S456 = a_cs_0(b_27, kernelContext_78);
    atomic_store_explicit(kernelContext_78->atom_0+(_S456 + 1U), entry_2 + 1U, memory_order_relaxed);
    uint _S457 = atomic_exchange_explicit(kernelContext_78->atom_0+(_S456 + 8U), 1U, memory_order_relaxed);
    if(_S457 == 0U)
    {
        uint _S458 = w_cs_bodies_0(kernelContext_78);
        uint _S459 = a_counters_1(kernelContext_78);
        uint _S460 = atomic_fetch_add_explicit(kernelContext_78->atom_0+(_S459 + 18U), 1U, memory_order_relaxed);
        *(kernelContext_78->work_0+(_S458 + _S460)) = b_27;
    }
    return;
}

void il_strike_1(uint b_28, KernelContext_0 thread* kernelContext_79)
{
    bool damage_1;
    if(b_28 >= (kernelContext_79->params_0->body_count_0))
    {
        damage_1 = true;
    }
    else
    {
        bool _S461 = il_struck_1(b_28, kernelContext_79);
        damage_1 = !_S461;
    }
    if(damage_1)
    {
        return;
    }
    float _S462 = il_t_in_1(b_28, kernelContext_79);
    uint _S463 = il_entry_1(b_28, kernelContext_79);
    thread float3 contact_15;
    thread float3 d_17;
    thread float speed_12;
    il_path_1(b_28, &contact_15, &d_17, &speed_12, kernelContext_79);
    IlStriker_0 _S464 = il_striker_1(b_28, kernelContext_79);
    float _S465 = max(_S464.radius_0, 0.00100000004749745f);
    if((_S464.launch_0) != 0U)
    {
        uint _S466 = w_marks_0(kernelContext_79);
        damage_1 = (*(kernelContext_79->work_0+(_S466 + _S463))) == (_S464.launch_0);
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        uint _S467 = lp_body_0(b_28, kernelContext_79);
        packed_float4 _S468 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_79->lp_0+_S467) = _S468;
        *(kernelContext_79->lp_0+(_S467 + 1U)) = _S468;
        return;
    }
    uint _S469 = a_il_0(b_28, kernelContext_79);
    uint _S470 = atomic_load_explicit(kernelContext_79->atom_0+_S469, memory_order_relaxed);
    float _S471 = max((as_type<float>((_S470))) - _S465, 9.99999997475242708e-07f);
    if(!(_S462 > 0.0f))
    {
        uint _S472 = lp_body_0(b_28, kernelContext_79);
        *(kernelContext_79->lp_0+_S472) = packed_float4(float4(_S471, 0.0f, 0.0f, 0.0f)) ;
        *(kernelContext_79->lp_0+(_S472 + 1U)) = packed_float4(float4(0.0f) ) ;
        return;
    }
    uint o_3 = kernelContext_79->params_0->ext_chunks_0 + 4U * _S463;
    float4 _S473 = float4(*(kernelContext_79->ext_0+(o_3 + 1U))) ;
    thread JointMaterial_0 dm_1;
    float4 _S474 = float4(0.0f) ;
    (&dm_1)->strength_0 = _S474;
    (&dm_1)->energy_0 = _S474;
    (&dm_1)->dif_0 = float4(*(kernelContext_79->ext_0+(o_3 + 2U))) ;
    float4 _S475 = float4(*(kernelContext_79->ext_0+(o_3 + 3U))) ;
    float _S476 = _S475.x;
    (&dm_1)->misc_0 = float4(_S476, 0.0f, 0.0f, 0.0f);
    (&dm_1)->kind_flags_0 = uint4(0U) ;
    float nu_1 = _S475.y;
    float e_target_1 = _S475.z / (1.0f - nu_1 * nu_1);
    float e_star_1 = 1.0f / (1.0f / _S464.modulus_0 + 1.0f / e_target_1);
    float _S477 = max(speed_12, 0.00100000004749745f);
    float _S478 = sqrt(_S465);
    float h0_1 = pow(15.0f * _S464.m_eff_0 * _S477 * _S477 / (16.0f * e_star_1 * _S478), 0.40000000596046448f);
    float t_c_1 = 2.94000005722045898f * h0_1 / _S477;
    float force_1 = 1.33333337306976318f * e_star_1 * _S478 * pow(h0_1, 1.5f);
    float a_4 = sqrt(_S465 * h0_1);
    float p0_1 = 3.0f * force_1 / (6.28318548202514648f * a_4 * a_4);
    float ring_1 = (1.0f - 2.0f * nu_1) * p0_1 / 3.0f;
    float k_19;
    if(_S476 > 0.0f)
    {
        float _S479 = _S477 / _S471;
        thread JointMaterial_0 _S480 = dm_1;
        float _S481 = dif_factor_0(&_S480, _S479);
        k_19 = _S481;
    }
    else
    {
        k_19 = 1.0f;
    }
    float ft_1 = k_19 * _S473.y;
    float fc_1 = k_19 * _S473.z;
    float tau_2 = k_19 * _S473.w;
    if(p0_1 > fc_1)
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = ring_1 > ft_1;
    }
    float _S482 = fc_1 * (3.14159274101257324f * _S465 * _S465);
    float _S483 = min(_S482 * _S471, 1.57079637050628662f * (2.0f * _S465 + _S471) * _S471 * _S471 * tau_2);
    float _S484 = 2.0f * _S483;
    float v_bl_1 = sqrt(_S484 / _S464.m_eff_0);
    bool perforate_1;
    if(damage_1)
    {
        perforate_1 = _S477 >= v_bl_1;
    }
    else
    {
        perforate_1 = false;
    }
    float r2_1 = _S465 + _S471;
    float _S485 = _S473.x;
    float plug_1 = _S485 * 3.14159274101257324f * _S471 / 3.0f * (_S465 * _S465 + _S465 * r2_1 + r2_1 * r2_1);
    float v_res_1;
    if(perforate_1)
    {
        v_res_1 = _S464.mass_0 / (_S464.mass_0 + plug_1) * sqrt(max(_S477 * _S477 - v_bl_1 * v_bl_1, 0.0f));
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
        depth_2 = min(0.5f * _S464.m_eff_0 * _S477 * _S477 / _S482, _S471);
    }
    else
    {
        depth_2 = 0.0f;
    }
    bool crater_1 = depth_2 > a_4;
    float c_5 = sqrt(e_target_1 / _S485);
    float z_t_1 = _S485 * c_5;
    float rho_p_1 = _S464.mass_0 / (4.18879032135009766f * _S465 * _S465 * _S465);
    float z_p_1 = rho_p_1 * sqrt(_S464.modulus_0 / rho_p_1);
    float mean_pressure_1 = force_1 / (3.14159274101257324f * a_4 * a_4);
    float _S486 = min(min(z_t_1 * _S477 * z_p_1 / (z_p_1 + z_t_1), mean_pressure_1), fc_1);
    float source_radius_1;
    if(mean_pressure_1 > fc_1)
    {
        source_radius_1 = max(a_4, _S465);
    }
    else
    {
        source_radius_1 = a_4;
    }
    float pulse_1 = c_5 * t_c_1;
    float back_1 = _S486 * min(source_radius_1 / _S471, 1.0f) * min(2.0f * _S471 / pulse_1, 1.0f);
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
        scab_2 = 0.5f * min(pulse_1, _S471);
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
    uint _S487;
    if(crater_1)
    {
        _S487 = 2U;
    }
    else
    {
        _S487 = 0U;
    }
    uint _S488 = region_4 | _S487;
    if(spall_1)
    {
        region_4 = 4U;
    }
    else
    {
        region_4 = 0U;
    }
    uint _S489 = _S488 | region_4;
    if(perforate_1)
    {
        region_4 = 8U;
    }
    else
    {
        region_4 = 0U;
    }
    uint outcome_8 = _S489 | region_4;
    uint _S490 = lp_body_0(b_28, kernelContext_79);
    *(kernelContext_79->lp_0+_S490) = packed_float4(float4(_S471, v_bl_1, v_res_1, (as_type<float>((outcome_8))))) ;
    packed_float4 device* _S491 = kernelContext_79->lp_0+(_S490 + 1U);
    if(crater_1)
    {
        k_19 = depth_2;
    }
    else
    {
        k_19 = 0.0f;
    }
    *_S491 = packed_float4(float4(k_19, scab_2, a_4, _S483)) ;
    if(perforate_1)
    {
        uint _S492 = a_counters_1(kernelContext_79);
        uint _S493 = atomic_fetch_add_explicit(kernelContext_79->atom_0+(_S492 + 14U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_1)
        {
            uint _S494 = a_counters_1(kernelContext_79);
            uint _S495 = atomic_fetch_add_explicit(kernelContext_79->atom_0+(_S494 + 13U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_1)
            {
                uint _S496 = a_counters_1(kernelContext_79);
                uint _S497 = atomic_fetch_add_explicit(kernelContext_79->atom_0+(_S496 + 12U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S498 = a_counters_1(kernelContext_79);
                uint _S499 = atomic_fetch_add_explicit(kernelContext_79->atom_0+(_S498 + 11U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_1)
    {
        if((kernelContext_79->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_28, _S464.body_0, _S464.mass_0, contact_15, d_17, _S464.m_eff_0 * (_S477 - sqrt(max(_S477 * _S477 - _S484 / _S464.m_eff_0, 0.0f))), kernelContext_79);
        }
    }
    else
    {
        if(damage_1)
        {
            il_impulse_0(b_28, _S464.body_0, _S464.mass_0, contact_15, d_17, _S464.m_eff_0 * _S477, kernelContext_79);
            float3 _S500 = to_rubble_0(d_17) * float3((_S477 * clamp(_S462, 0.0f, kernelContext_79->params_0->dt_0))) ;
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
                uint _S501 = 7U * (_S464.body_0 + region_4 * kernelContext_79->params_0->max_bodies_0);
                (*(kernelContext_79->bodies_0+_S501)).xyz = (float4(*(kernelContext_79->bodies_0+_S501)) ).xyz + _S500;
                region_4 = region_4 + 1U;
            }
        }
    }
    if((kernelContext_79->params_0->gi_mode_0) != 0U)
    {
        if(damage_1)
        {
            damage_1 = true;
        }
        else
        {
            damage_1 = (kernelContext_79->params_0->cs_sweeps_0) == 0U;
        }
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        float3 _S502 = contact_15;
        float3 _S503 = d_17;
        if(perforate_1)
        {
            k_19 = sqrt(max(_S477 * _S477 - v_bl_1 * v_bl_1, 0.0f));
        }
        else
        {
            k_19 = 0.0f;
        }
        float _S504 = _S464.m_eff_0 * (_S477 - k_19);
        if(perforate_1)
        {
            k_19 = _S471;
        }
        else
        {
            k_19 = depth_2;
        }
        gi_record_0(b_28, _S463, _S502, _S503, _S477, _S504, max(t_c_1, 2.0f * k_19 / max(_S477 + v_res_1, 9.99999971718068537e-10f)), _S471, kernelContext_79);
    }
    return;
}

void il_run_1(uint b_29, uint i_5, KernelContext_0 thread* kernelContext_80)
{
    bool _S505 = il_struck_1(b_29, kernelContext_80);
    uint j_3;
    if(_S505)
    {
        j_3 = 1U;
    }
    else
    {
        j_3 = 0U;
    }
    if((group_uniform_0(j_3, i_5)) == 0U)
    {
        return;
    }
    IlStriker_0 _S506 = il_striker_1(b_29, kernelContext_80);
    float _S507 = 0.00100000004749745f + 0.01999999955296516f * _S506.radius_0;
    uint _S508 = a_il_0(b_29, kernelContext_80);
    uint _S509 = atomic_load_explicit(kernelContext_80->atom_0+(_S508 + 4U), memory_order_relaxed);
    uint _S510 = atomic_load_explicit(kernelContext_80->atom_0+(_S508 + 1U), memory_order_relaxed);
    uint _S511 = atomic_load_explicit(kernelContext_80->atom_0+_S508, memory_order_relaxed);
    float end_1 = group_uniform_float_0((as_type<float>((_S511))), i_5);
    for(;;)
    {
        float best_1 = end_1;
        j_3 = i_5;
        for(;;)
        {
            if(j_3 < _S510)
            {
            }
            else
            {
                break;
            }
            uint _S512 = w_il_tube_0(kernelContext_80);
            uint at_6 = _S512 + 2U * (_S509 + j_3);
            if((as_type<float>((*(kernelContext_80->work_0+at_6)))) <= (end_1 + _S507))
            {
                best_1 = max(best_1, (as_type<float>((*(kernelContext_80->work_0+(at_6 + 1U))))));
            }
            j_3 = j_3 + 64U;
        }
        (*kernelContext_80->g_il_best_0)[i_5] = best_1;
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
                (*kernelContext_80->g_il_best_0)[i_5] = max((*kernelContext_80->g_il_best_0)[i_5], (*kernelContext_80->g_il_best_0)[i_5 + step_2]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_2 = step_2 >> 1U;
        }
        float next_1 = group_uniform_float_0((*kernelContext_80->g_il_best_0)[int(0)], i_5);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!(next_1 > end_1))
        {
            break;
        }
        end_1 = next_1;
    }
    if(i_5 == 0U)
    {
        atomic_store_explicit(kernelContext_80->atom_0+_S508, (as_type<uint>((end_1))), memory_order_relaxed);
    }
    return;
}

void il_tube_collect_1(uint k_20, KernelContext_0 thread* kernelContext_81)
{
    thread uint b_30;
    thread float2 t_5;
    bool _S513 = il_tube_1(k_20, &b_30, &t_5, kernelContext_81);
    if(!_S513)
    {
        return;
    }
    uint _S514 = w_il_tube_0(kernelContext_81);
    uint _S515 = a_il_0(b_30, kernelContext_81);
    uint _S516 = atomic_load_explicit(kernelContext_81->atom_0+(_S515 + 4U), memory_order_relaxed);
    uint _S517 = a_il_0(b_30, kernelContext_81);
    uint _S518 = atomic_fetch_add_explicit(kernelContext_81->atom_0+(_S517 + 5U), 1U, memory_order_relaxed);
    uint at_7 = _S514 + 2U * (_S516 + _S518);
    *(kernelContext_81->work_0+at_7) = (as_type<uint>((t_5.x)));
    *(kernelContext_81->work_0+(at_7 + 1U)) = (as_type<uint>((t_5.y)));
    return;
}

void il_tube_base_1(uint b_31, KernelContext_0 thread* kernelContext_82)
{
    bool _S519;
    if(b_31 < (kernelContext_82->params_0->body_count_0))
    {
        bool _S520 = il_struck_1(b_31, kernelContext_82);
        _S519 = _S520;
    }
    else
    {
        _S519 = false;
    }
    if(_S519)
    {
        uint _S521 = a_il_0(b_31, kernelContext_82);
        atomic<uint> device* _S522 = kernelContext_82->atom_0+(_S521 + 4U);
        uint _S523 = a_il_cursor_1(kernelContext_82);
        atomic<uint> device* _S524 = kernelContext_82->atom_0+_S523;
        uint _S525 = atomic_load_explicit(kernelContext_82->atom_0+(_S521 + 1U), memory_order_relaxed);
        uint _S526 = atomic_fetch_add_explicit(_S524, _S525, memory_order_relaxed);
        atomic_store_explicit(_S522, _S526, memory_order_relaxed);
    }
    return;
}

void il_tube_count_1(uint k_21, KernelContext_0 thread* kernelContext_83)
{
    thread uint b_32;
    thread float2 t_6;
    bool _S527 = il_tube_1(k_21, &b_32, &t_6, kernelContext_83);
    if(_S527)
    {
        uint _S528 = a_il_0(b_32, kernelContext_83);
        uint _S529 = atomic_fetch_add_explicit(kernelContext_83->atom_0+(_S528 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_first_1(uint shot_3, KernelContext_0 thread* kernelContext_84)
{
    uint _S530 = a_strike_0(kernelContext_84);
    return _S530 + 29U * kernelContext_84->params_0->body_count_0 + shot_3;
}

void il_begin_1(uint b_33, KernelContext_0 thread* kernelContext_85)
{
    bool _S531;
    if(b_33 >= (kernelContext_85->params_0->body_count_0))
    {
        _S531 = true;
    }
    else
    {
        bool _S532 = il_struck_1(b_33, kernelContext_85);
        _S531 = !_S532;
    }
    if(_S531)
    {
        return;
    }
    thread uint shot_4;
    bool _S533 = struck_0(b_33, &shot_4, kernelContext_85);
    uint _S534 = a_strike_0(kernelContext_85);
    uint _S535 = atomic_load_explicit(kernelContext_85->atom_0+(_S534 + 9U * b_33 + 6U), memory_order_relaxed);
    uint _S536 = a_first_1(shot_4, kernelContext_85);
    uint _S537 = atomic_load_explicit(kernelContext_85->atom_0+_S536, memory_order_relaxed);
    if(_S535 != _S537)
    {
        uint _S538 = strike_at_0(b_33, kernelContext_85);
        (*(kernelContext_85->frames_0+_S538))[int(3)] = 0.0f;
        return;
    }
    thread float3 contact_16;
    thread float3 d_18;
    thread float speed_13;
    il_path_1(b_33, &contact_16, &d_18, &speed_13, kernelContext_85);
    uint _S539 = il_entry_1(b_33, kernelContext_85);
    IlStriker_0 _S540 = il_striker_1(b_33, kernelContext_85);
    float2 _S541 = il_interval_0(_S539, contact_16, d_18, _S540.radius_0, kernelContext_85);
    uint _S542 = a_il_0(b_33, kernelContext_85);
    atomic_store_explicit(kernelContext_85->atom_0+_S542, (as_type<uint>((max(_S541.y, 0.0f)))), memory_order_relaxed);
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
        atomic_store_explicit(kernelContext_85->atom_0+(_S542 + i_6), 0U, memory_order_relaxed);
        i_6 = i_6 + 1U;
    }
    return;
}

[[kernel]] void fr_events_reset(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], atomic<uint> device* atom_1 [[buffer(5)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], uint device* sort_1 [[buffer(11)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], packed_float4 device* loads_1 [[buffer(12)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]], packed_float4 device* view_chunks_1 [[buffer(17)]], ViewParams_0 constant* view_1 [[buffer(16)]], packed_float4 device* objects_1 [[buffer(18)]], uint device* shot_bodies_1 [[buffer(20)]], float device* lines_1 [[buffer(19)]], packed_float4 device* geometry_1 [[buffer(21)]], packed_float4 device* shots_1 [[buffer(22)]], packed_float4 device* lp_1 [[buffer(24)]], packed_float4 device* ext_2 [[buffer(23)]], packed_float4 device* engine_contacts_1 [[buffer(25)]], CrushParams_0 constant* crush_1 [[buffer(28)]], packed_float4 device* crush_rw_1 [[buffer(27)]], packed_float4 device* crush_tab_1 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_86;
    (&kernelContext_86)->params_0 = params_1;
    (&kernelContext_86)->atom_0 = atom_1;
    (&kernelContext_86)->work_0 = work_1;
    (&kernelContext_86)->frames_0 = frames_1;
    (&kernelContext_86)->bodies_0 = bodies_1;
    (&kernelContext_86)->chunks_0 = chunks_1;
    (&kernelContext_86)->sort_0 = sort_1;
    (&kernelContext_86)->manifolds_0 = manifolds_1;
    (&kernelContext_86)->loads_0 = loads_1;
    (&kernelContext_86)->bonds_0 = bonds_1;
    (&kernelContext_86)->args_0 = args_1;
    (&kernelContext_86)->info_0 = info_1;
    (&kernelContext_86)->scan_0 = scan_1;
    (&kernelContext_86)->parts_0 = parts_1;
    (&kernelContext_86)->colliders_0 = colliders_1;
    (&kernelContext_86)->collider_info_0 = collider_info_1;
    (&kernelContext_86)->view_chunks_0 = view_chunks_1;
    (&kernelContext_86)->view_0 = view_1;
    (&kernelContext_86)->objects_0 = objects_1;
    (&kernelContext_86)->shot_bodies_0 = shot_bodies_1;
    (&kernelContext_86)->lines_0 = lines_1;
    (&kernelContext_86)->geometry_0 = geometry_1;
    (&kernelContext_86)->shots_0 = shots_1;
    (&kernelContext_86)->lp_0 = lp_1;
    (&kernelContext_86)->ext_0 = ext_2;
    (&kernelContext_86)->engine_contacts_0 = engine_contacts_1;
    (&kernelContext_86)->crush_0 = crush_1;
    (&kernelContext_86)->crush_rw_0 = crush_rw_1;
    (&kernelContext_86)->crush_tab_0 = crush_tab_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_86)->g_scan_0 = &g_scan_1;
    threadgroup array<float, int(64)> g_il_best_1;
    (&kernelContext_86)->g_il_best_0 = &g_il_best_1;
    threadgroup array<float, int(256)> g_gi_x_1;
    (&kernelContext_86)->g_gi_x_0 = &g_gi_x_1;
    threadgroup array<float, int(256)> g_gi_m_1;
    (&kernelContext_86)->g_gi_m_0 = &g_gi_m_1;
    threadgroup array<uint, int(1280)> g_gi_lo_1;
    (&kernelContext_86)->g_gi_lo_0 = &g_gi_lo_1;
    threadgroup array<uint, int(1280)> g_gi_hi_1;
    (&kernelContext_86)->g_gi_hi_0 = &g_gi_hi_1;
    uint _S543 = a_counters_1(&kernelContext_86);
    atomic_store_explicit((&kernelContext_86)->atom_0+(_S543 + 20U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_86)->atom_0+(_S543 + 21U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_86)->atom_0+(_S543 + 22U), 0U, memory_order_relaxed);
    return;
}

bool live_0(uint c_6, KernelContext_0 thread* kernelContext_87)
{
    bool _S544;
    if(c_6 < (kernelContext_87->params_0->chunk_used_0))
    {
        uint _S545 = body_of_0(c_6, kernelContext_87);
        _S544 = _S545 != 4294967295U;
    }
    else
    {
        _S544 = false;
    }
    return _S544;
}

uint w_cs_rows_1(KernelContext_0 thread* kernelContext_88)
{
    uint _S546 = w_host_breaks_0(kernelContext_88);
    return _S546 + 1U + kernelContext_88->params_0->bond_count_0;
}

uint w_frame_0(KernelContext_0 thread* kernelContext_89)
{
    uint _S547 = w_cs_rows_1(kernelContext_89);
    return _S547 + 1U;
}

uint frame_parity_0(KernelContext_0 thread* kernelContext_90)
{
    uint _S548 = w_frame_0(kernelContext_90);
    return *(kernelContext_90->work_0+_S548);
}

uint l_pool_0(uint half_1, KernelContext_0 thread* kernelContext_91)
{
    return kernelContext_91->params_0->chunk_count_0 + 3U * kernelContext_91->params_0->pool_cap_0 * half_1;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S549 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S549, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S550 = max(0.0f, _S549);
    float _S551 = min(dt_1, _S549 + duration_0);
    bool _S552;
    if(_S551 > _S550)
    {
        _S552 = duration_0 > 0.0f;
    }
    else
    {
        _S552 = false;
    }
    float sum_1;
    if(_S552)
    {
        float w_2 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S550 - _S549)) - cos(w_2 * (_S551 - _S549)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

float hertz_duration_0(float m_0, float r_18, float e_2, float v_10)
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_18 * e_2 * e_2 * max(abs(v_10), 0.00100000004749745f)), 0.20000000298023224f);
}

uint l_vec_0(KernelContext_0 thread* kernelContext_92)
{
    return kernelContext_92->params_0->chunk_count_0 + 6U * kernelContext_92->params_0->pool_cap_0;
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
    thread KernelContext_0 kernelContext_93;
    (&kernelContext_93)->params_0 = params_2;
    (&kernelContext_93)->atom_0 = atom_2;
    (&kernelContext_93)->work_0 = work_2;
    (&kernelContext_93)->frames_0 = frames_2;
    (&kernelContext_93)->bodies_0 = bodies_2;
    (&kernelContext_93)->chunks_0 = chunks_2;
    (&kernelContext_93)->sort_0 = sort_2;
    (&kernelContext_93)->manifolds_0 = manifolds_2;
    (&kernelContext_93)->loads_0 = loads_2;
    (&kernelContext_93)->bonds_0 = bonds_2;
    (&kernelContext_93)->args_0 = args_2;
    (&kernelContext_93)->info_0 = info_2;
    (&kernelContext_93)->scan_0 = scan_2;
    (&kernelContext_93)->parts_0 = parts_2;
    (&kernelContext_93)->colliders_0 = colliders_2;
    (&kernelContext_93)->collider_info_0 = collider_info_2;
    (&kernelContext_93)->view_chunks_0 = view_chunks_2;
    (&kernelContext_93)->view_0 = view_2;
    (&kernelContext_93)->objects_0 = objects_2;
    (&kernelContext_93)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_93)->lines_0 = lines_2;
    (&kernelContext_93)->geometry_0 = geometry_2;
    (&kernelContext_93)->shots_0 = shots_2;
    (&kernelContext_93)->lp_0 = lp_2;
    (&kernelContext_93)->ext_0 = ext_3;
    (&kernelContext_93)->engine_contacts_0 = engine_contacts_2;
    (&kernelContext_93)->crush_0 = crush_2;
    (&kernelContext_93)->crush_rw_0 = crush_rw_2;
    (&kernelContext_93)->crush_tab_0 = crush_tab_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_93)->g_scan_0 = &g_scan_2;
    threadgroup array<float, int(64)> g_il_best_2;
    (&kernelContext_93)->g_il_best_0 = &g_il_best_2;
    threadgroup array<float, int(256)> g_gi_x_2;
    (&kernelContext_93)->g_gi_x_0 = &g_gi_x_2;
    threadgroup array<float, int(256)> g_gi_m_2;
    (&kernelContext_93)->g_gi_m_0 = &g_gi_m_2;
    threadgroup array<uint, int(1280)> g_gi_lo_2;
    (&kernelContext_93)->g_gi_lo_0 = &g_gi_lo_2;
    threadgroup array<uint, int(1280)> g_gi_hi_2;
    (&kernelContext_93)->g_gi_hi_0 = &g_gi_hi_2;
    uint k_22 = id_1.x;
    bool _S553 = live_0(k_22, &kernelContext_93);
    if(!_S553)
    {
        return;
    }
    uint _S554 = body_of_0(k_22, &kernelContext_93);
    uint b_34 = 7U * _S554;
    thread Quat_0 q_11;
    thread float3 origin_3;
    structure_pose_0(_S554, b_34, &q_11, &origin_3, &kernelContext_93);
    uint _S555 = 4U * k_22;
    float4 _S556 = float4(*((&kernelContext_93)->chunks_0+_S555)) ;
    float3 _S557 = _S556.xyz;
    thread Quat_0 _S558 = q_11;
    float3 _S559 = rotate_0(&_S558, _S557);
    float3 centre_0 = origin_3 + _S559;
    float inv_mass_2 = (float4(*((&kernelContext_93)->bodies_0+b_34)) ).w;
    float m_body_0;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint _S560 = frame_parity_0(&kernelContext_93);
    uint to_0 = 1U - _S560;
    uint _S561 = w_pulses_0(&kernelContext_93);
    uint _S562 = _S561 + _S555;
    uint _S563 = _S562 + 2U * _S560;
    uint off_0 = *((&kernelContext_93)->work_0+_S563);
    uint device* _S564 = (&kernelContext_93)->work_0+(_S563 + 1U);
    uint carried_0 = *_S564;
    uint room_0 = *_S564 + (&kernelContext_93)->sort_0[(&kernelContext_93)->params_0->chunk_count_0 + 1U + k_22];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S565 = a_counters_1(&kernelContext_93);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_93)->atom_0+(_S565 + 5U + to_0), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_93)->params_0->pool_cap_0))
        {
            uint _S566 = atomic_fetch_or_explicit((&kernelContext_93)->atom_0+(_S565 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_93)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_93)->params_0->pool_cap_0 - base_1;
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
        uint _S567 = base_0;
        base_0 = base_1;
        room_1 = _S567;
    }
    else
    {
        base_0 = 0U;
        room_1 = room_0;
    }
    uint _S568 = (&kernelContext_93)->params_0->chunk_count_0;
    float _S569 = _S556.w;
    float4 _S570 = float4(*((&kernelContext_93)->chunks_0+(_S555 + 3U))) ;
    float _S571 = (&kernelContext_93)->params_0->band_0;
    float _S572 = (&kernelContext_93)->params_0->dt_0;
    bool _S573 = ((&kernelContext_93)->params_0->impact_peak_0) != 0U;
    uint _S574 = l_pool_0(_S560, &kernelContext_93);
    uint _S575 = _S574 + 3U * off_0;
    uint _S576 = l_pool_0(to_0, &kernelContext_93);
    uint _S577 = _S576 + 3U * base_0;
    for(;;)
    {
        float3 _S578 = float3(0.0f) ;
        uint _S579 = (&kernelContext_93)->sort_0[k_22];
        uint _S580 = (&kernelContext_93)->sort_0[_S568 + 1U + k_22];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_3 = 0U;
        f_total_0 = _S578;
        t_total_0 = _S578;
        for(;;)
        {
            if(step_3 < _S580)
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
                if(q3_0 < _S580)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_93)->manifolds_0[_S579 + q3_0].ids_0) ).y + 1U;
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
            Record_natural_0 r_19 = (&kernelContext_93)->manifolds_0[_S579 + p_10];
            float4 _S581 = float4(r_19.impulse_0) ;
            float3 _S582 = _S581.xyz;
            float j_4 = length(_S582);
            if(j_4 == 0.0f)
            {
                uint _S583 = step_3 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_3 = _S583;
                continue;
            }
            float m_other_0 = (float4(r_19.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S569;
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
            float _S584 = min(j_4, 2.0f * m_1 * max(_S581.w - _S571, 0.0f));
            if(_S584 < j_4)
            {
                float3 f_1 = _S582 / float3(j_4)  * float3(((j_4 - _S584) / _S572)) ;
                float3 _S585 = t_total_0 + cross((float4(r_19.point_0) ).xyz - centre_0, f_1);
                f_total_1 = f_total_0 + f_1;
                t_total_1 = _S585;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            uint _S583 = step_3 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_3 = _S583;
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
            uint _S586 = _S575 + 3U * p_10;
            float4 _S587 = float4(*((&kernelContext_93)->loads_0+_S586)) ;
            thread float4 a_5 = _S587;
            float4 _S588 = float4(*((&kernelContext_93)->loads_0+(_S586 + 1U))) ;
            float4 _S589 = float4(*((&kernelContext_93)->loads_0+(_S586 + 2U))) ;
            float _S590 = _S588.w;
            float mean_0 = pulse_mean_0(_S587.x, _S587.y, _S587.z, _S587.w, _S590, _S572);
            if(mean_0 != 0.0f)
            {
                float3 f_2 = _S589.xyz * float3(mean_0) ;
                float3 _S591 = t_total_0 + cross(_S588.xyz - centre_0, f_2);
                f_total_0 = f_total_0 + f_2;
                t_total_0 = _S591;
            }
            if((a_5.x + a_5.z + _S590) > _S572)
            {
                a_5.x = a_5.x - _S572;
                uint _S592 = _S577 + 3U * kept_0;
                *((&kernelContext_93)->loads_0+_S592) = packed_float4(a_5) ;
                *((&kernelContext_93)->loads_0+(_S592 + 1U)) = packed_float4(_S588) ;
                *((&kernelContext_93)->loads_0+(_S592 + 2U)) = packed_float4(_S589) ;
                kept_0 = kept_0 + 1U;
            }
            p_10 = p_10 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S580)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_20 = (&kernelContext_93)->manifolds_0[_S579 + q3_0];
            float4 _S593 = float4(r_20.impulse_0) ;
            float3 _S594 = _S593.xyz;
            float j_5 = length(_S594);
            if(j_5 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S595 = float4(r_20.other_0) ;
            float m_other_1 = _S595.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S569;
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
            float _S596 = _S593.w;
            float _S597 = min(j_5, 2.0f * m_1 * max(_S596 - _S571, 0.0f));
            if(_S597 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S598 = _S570.y;
            float duration_1 = hertz_duration_0(m_1, min(_S595.z, _S570.z), 1.0f / ((1.0f - _S598 * _S598) / _S570.x + 1.0f / _S595.y), _S596);
            float peak_1 = 3.14159274101257324f * _S597 / (2.0f * duration_1);
            float3 dir_0 = _S594 / float3(j_5) ;
            float mean_1;
            if(_S573)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S572);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_3 = dir_0 * float3(mean_1) ;
                float3 _S599 = t_total_0 + cross((float4(r_20.point_0) ).xyz - centre_0, f_3);
                f_total_1 = f_total_0 + f_3;
                t_total_1 = _S599;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            uint kept_1;
            if(duration_1 > _S572)
            {
                if(kept_0 < room_1)
                {
                    uint _S600 = _S577 + 3U * kept_0;
                    *((&kernelContext_93)->loads_0+_S600) = packed_float4(float4(- _S572, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_93)->loads_0+(_S600 + 1U)) = packed_float4(float4((float4(r_20.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_93)->loads_0+(_S600 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
        uint _S601 = a_counters_1(&kernelContext_93);
        uint _S602 = atomic_fetch_or_explicit((&kernelContext_93)->atom_0+(_S601 + 3U), 8U, memory_order_relaxed);
    }
    uint _S603 = _S562 + 2U * to_0;
    *((&kernelContext_93)->work_0+_S603) = base_0;
    *((&kernelContext_93)->work_0+(_S603 + 1U)) = kept_0;
    *((&kernelContext_93)->loads_0+k_22) = packed_float4(float4(length(f_total_0), f_total_0)) ;
    if(((&kernelContext_93)->params_0->load_model_0) == 1U)
    {
        uint _S604 = l_vec_0(&kernelContext_93);
        uint _S605 = _S604 + 2U * k_22;
        *((&kernelContext_93)->loads_0+_S605) = packed_float4(float4(f_total_0, 0.0f)) ;
        *((&kernelContext_93)->loads_0+(_S605 + 1U)) = packed_float4(float4(t_total_0, 0.0f)) ;
    }
    return;
}

[[kernel]] void fr_breaks(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], atomic<uint> device* atom_3 [[buffer(5)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], packed_float4 device* lp_3 [[buffer(24)]], packed_float4 device* ext_4 [[buffer(23)]], packed_float4 device* engine_contacts_3 [[buffer(25)]], CrushParams_0 constant* crush_3 [[buffer(28)]], packed_float4 device* crush_rw_3 [[buffer(27)]], packed_float4 device* crush_tab_3 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_94;
    (&kernelContext_94)->params_0 = params_3;
    (&kernelContext_94)->atom_0 = atom_3;
    (&kernelContext_94)->work_0 = work_3;
    (&kernelContext_94)->frames_0 = frames_3;
    (&kernelContext_94)->bodies_0 = bodies_3;
    (&kernelContext_94)->chunks_0 = chunks_3;
    (&kernelContext_94)->sort_0 = sort_3;
    (&kernelContext_94)->manifolds_0 = manifolds_3;
    (&kernelContext_94)->loads_0 = loads_3;
    (&kernelContext_94)->bonds_0 = bonds_3;
    (&kernelContext_94)->args_0 = args_3;
    (&kernelContext_94)->info_0 = info_3;
    (&kernelContext_94)->scan_0 = scan_3;
    (&kernelContext_94)->parts_0 = parts_3;
    (&kernelContext_94)->colliders_0 = colliders_3;
    (&kernelContext_94)->collider_info_0 = collider_info_3;
    (&kernelContext_94)->view_chunks_0 = view_chunks_3;
    (&kernelContext_94)->view_0 = view_3;
    (&kernelContext_94)->objects_0 = objects_3;
    (&kernelContext_94)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_94)->lines_0 = lines_3;
    (&kernelContext_94)->geometry_0 = geometry_3;
    (&kernelContext_94)->shots_0 = shots_3;
    (&kernelContext_94)->lp_0 = lp_3;
    (&kernelContext_94)->ext_0 = ext_4;
    (&kernelContext_94)->engine_contacts_0 = engine_contacts_3;
    (&kernelContext_94)->crush_0 = crush_3;
    (&kernelContext_94)->crush_rw_0 = crush_rw_3;
    (&kernelContext_94)->crush_tab_0 = crush_tab_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_94)->g_scan_0 = &g_scan_3;
    threadgroup array<float, int(64)> g_il_best_3;
    (&kernelContext_94)->g_il_best_0 = &g_il_best_3;
    threadgroup array<float, int(256)> g_gi_x_3;
    (&kernelContext_94)->g_gi_x_0 = &g_gi_x_3;
    threadgroup array<float, int(256)> g_gi_m_3;
    (&kernelContext_94)->g_gi_m_0 = &g_gi_m_3;
    threadgroup array<uint, int(1280)> g_gi_lo_3;
    (&kernelContext_94)->g_gi_lo_0 = &g_gi_lo_3;
    threadgroup array<uint, int(1280)> g_gi_hi_3;
    (&kernelContext_94)->g_gi_hi_0 = &g_gi_hi_3;
    uint i_7 = id_2.x;
    bool _S606;
    if(i_7 >= (params_3->bond_used_0))
    {
        _S606 = true;
    }
    else
    {
        uint _S607 = w_alive_0(&kernelContext_94);
        _S606 = (*((&kernelContext_94)->work_0+(_S607 + i_7))) == 0U;
    }
    if(_S606)
    {
        return;
    }
    uint4 _S608 = uint4(*((&kernelContext_94)->bonds_0+i_7)) ;
    uint _S609 = _S608.x;
    if((max((float4(*((&kernelContext_94)->loads_0+_S609)) ).x, (float4(*((&kernelContext_94)->loads_0+_S608.y)) ).x)) > (as_type<float>((_S608.z))))
    {
        uint _S610 = w_alive_0(&kernelContext_94);
        *((&kernelContext_94)->work_0+(_S610 + i_7)) = 0U;
        uint _S611 = a_dirty_0(&kernelContext_94);
        uint _S612 = body_of_0(_S609, &kernelContext_94);
        uint _S613 = atomic_fetch_or_explicit((&kernelContext_94)->atom_0+(_S611 + _S612), 1U, memory_order_relaxed);
        uint _S614 = a_counters_1(&kernelContext_94);
        uint _S615 = atomic_fetch_add_explicit((&kernelContext_94)->atom_0+_S614, 1U, memory_order_relaxed);
        uint _S616 = atomic_fetch_add_explicit((&kernelContext_94)->atom_0+(_S614 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_7, &kernelContext_94);
    }
    return;
}

uint a_moved_0(KernelContext_0 thread* kernelContext_95)
{
    uint _S617 = a_events_0(kernelContext_95);
    return _S617 + kernelContext_95->params_0->ev_broken_cap_0;
}

uint a_view_0(KernelContext_0 thread* kernelContext_96)
{
    uint _S618 = a_moved_0(kernelContext_96);
    return _S618 + 2U * kernelContext_96->params_0->ev_moved_cap_0;
}

[[kernel]] void fr_stress_clear(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], atomic<uint> device* atom_4 [[buffer(5)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], packed_float4 device* lp_4 [[buffer(24)]], packed_float4 device* ext_5 [[buffer(23)]], packed_float4 device* engine_contacts_4 [[buffer(25)]], CrushParams_0 constant* crush_4 [[buffer(28)]], packed_float4 device* crush_rw_4 [[buffer(27)]], packed_float4 device* crush_tab_4 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_97;
    (&kernelContext_97)->params_0 = params_4;
    (&kernelContext_97)->atom_0 = atom_4;
    (&kernelContext_97)->work_0 = work_4;
    (&kernelContext_97)->frames_0 = frames_4;
    (&kernelContext_97)->bodies_0 = bodies_4;
    (&kernelContext_97)->chunks_0 = chunks_4;
    (&kernelContext_97)->sort_0 = sort_4;
    (&kernelContext_97)->manifolds_0 = manifolds_4;
    (&kernelContext_97)->loads_0 = loads_4;
    (&kernelContext_97)->bonds_0 = bonds_4;
    (&kernelContext_97)->args_0 = args_4;
    (&kernelContext_97)->info_0 = info_4;
    (&kernelContext_97)->scan_0 = scan_4;
    (&kernelContext_97)->parts_0 = parts_4;
    (&kernelContext_97)->colliders_0 = colliders_4;
    (&kernelContext_97)->collider_info_0 = collider_info_4;
    (&kernelContext_97)->view_chunks_0 = view_chunks_4;
    (&kernelContext_97)->view_0 = view_4;
    (&kernelContext_97)->objects_0 = objects_4;
    (&kernelContext_97)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_97)->lines_0 = lines_4;
    (&kernelContext_97)->geometry_0 = geometry_4;
    (&kernelContext_97)->shots_0 = shots_4;
    (&kernelContext_97)->lp_0 = lp_4;
    (&kernelContext_97)->ext_0 = ext_5;
    (&kernelContext_97)->engine_contacts_0 = engine_contacts_4;
    (&kernelContext_97)->crush_0 = crush_4;
    (&kernelContext_97)->crush_rw_0 = crush_rw_4;
    (&kernelContext_97)->crush_tab_0 = crush_tab_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_97)->g_scan_0 = &g_scan_4;
    threadgroup array<float, int(64)> g_il_best_4;
    (&kernelContext_97)->g_il_best_0 = &g_il_best_4;
    threadgroup array<float, int(256)> g_gi_x_4;
    (&kernelContext_97)->g_gi_x_0 = &g_gi_x_4;
    threadgroup array<float, int(256)> g_gi_m_4;
    (&kernelContext_97)->g_gi_m_0 = &g_gi_m_4;
    threadgroup array<uint, int(1280)> g_gi_lo_4;
    (&kernelContext_97)->g_gi_lo_0 = &g_gi_lo_4;
    threadgroup array<uint, int(1280)> g_gi_hi_4;
    (&kernelContext_97)->g_gi_hi_0 = &g_gi_hi_4;
    uint _S619 = id_3.x;
    if(_S619 < (params_4->chunk_count_0))
    {
        uint _S620 = a_view_0(&kernelContext_97);
        atomic_store_explicit((&kernelContext_97)->atom_0+(_S620 + _S619), 0U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_stress_breaks(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], atomic<uint> device* atom_5 [[buffer(5)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], packed_float4 device* lp_5 [[buffer(24)]], packed_float4 device* ext_6 [[buffer(23)]], packed_float4 device* engine_contacts_5 [[buffer(25)]], CrushParams_0 constant* crush_5 [[buffer(28)]], packed_float4 device* crush_rw_5 [[buffer(27)]], packed_float4 device* crush_tab_5 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_98;
    (&kernelContext_98)->params_0 = params_5;
    (&kernelContext_98)->atom_0 = atom_5;
    (&kernelContext_98)->work_0 = work_5;
    (&kernelContext_98)->frames_0 = frames_5;
    (&kernelContext_98)->bodies_0 = bodies_5;
    (&kernelContext_98)->chunks_0 = chunks_5;
    (&kernelContext_98)->sort_0 = sort_5;
    (&kernelContext_98)->manifolds_0 = manifolds_5;
    (&kernelContext_98)->loads_0 = loads_5;
    (&kernelContext_98)->bonds_0 = bonds_5;
    (&kernelContext_98)->args_0 = args_5;
    (&kernelContext_98)->info_0 = info_5;
    (&kernelContext_98)->scan_0 = scan_5;
    (&kernelContext_98)->parts_0 = parts_5;
    (&kernelContext_98)->colliders_0 = colliders_5;
    (&kernelContext_98)->collider_info_0 = collider_info_5;
    (&kernelContext_98)->view_chunks_0 = view_chunks_5;
    (&kernelContext_98)->view_0 = view_5;
    (&kernelContext_98)->objects_0 = objects_5;
    (&kernelContext_98)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_98)->lines_0 = lines_5;
    (&kernelContext_98)->geometry_0 = geometry_5;
    (&kernelContext_98)->shots_0 = shots_5;
    (&kernelContext_98)->lp_0 = lp_5;
    (&kernelContext_98)->ext_0 = ext_6;
    (&kernelContext_98)->engine_contacts_0 = engine_contacts_5;
    (&kernelContext_98)->crush_0 = crush_5;
    (&kernelContext_98)->crush_rw_0 = crush_rw_5;
    (&kernelContext_98)->crush_tab_0 = crush_tab_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_98)->g_scan_0 = &g_scan_5;
    threadgroup array<float, int(64)> g_il_best_5;
    (&kernelContext_98)->g_il_best_0 = &g_il_best_5;
    threadgroup array<float, int(256)> g_gi_x_5;
    (&kernelContext_98)->g_gi_x_0 = &g_gi_x_5;
    threadgroup array<float, int(256)> g_gi_m_5;
    (&kernelContext_98)->g_gi_m_0 = &g_gi_m_5;
    threadgroup array<uint, int(1280)> g_gi_lo_5;
    (&kernelContext_98)->g_gi_lo_0 = &g_gi_lo_5;
    threadgroup array<uint, int(1280)> g_gi_hi_5;
    (&kernelContext_98)->g_gi_hi_0 = &g_gi_hi_5;
    uint i_8 = id_4.x;
    bool _S621;
    if(i_8 >= (params_5->bond_used_0))
    {
        _S621 = true;
    }
    else
    {
        uint _S622 = w_alive_0(&kernelContext_98);
        _S621 = (*((&kernelContext_98)->work_0+(_S622 + i_8))) == 0U;
    }
    if(_S621)
    {
        return;
    }
    uint4 _S623 = uint4(*((&kernelContext_98)->bonds_0+i_8)) ;
    float capacity_0 = (as_type<float>((_S623.z)));
    if(capacity_0 <= 0.0f)
    {
        return;
    }
    uint _S624 = _S623.x;
    uint _S625 = _S623.y;
    uint u_6 = (as_type<uint>((max(max((float4(*((&kernelContext_98)->loads_0+_S624)) ).x, (float4(*((&kernelContext_98)->loads_0+_S625)) ).x), 0.0f) / capacity_0)));
    uint _S626 = a_view_0(&kernelContext_98);
    uint _S627 = atomic_fetch_max_explicit((&kernelContext_98)->atom_0+(_S626 + _S624), u_6, memory_order_relaxed);
    uint _S628 = atomic_fetch_max_explicit((&kernelContext_98)->atom_0+(_S626 + _S625), u_6, memory_order_relaxed);
    return;
}

[[kernel]] void fr_host_breaks(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], atomic<uint> device* atom_6 [[buffer(5)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], packed_float4 device* lp_6 [[buffer(24)]], packed_float4 device* ext_7 [[buffer(23)]], packed_float4 device* engine_contacts_6 [[buffer(25)]], CrushParams_0 constant* crush_6 [[buffer(28)]], packed_float4 device* crush_rw_6 [[buffer(27)]], packed_float4 device* crush_tab_6 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_6;
    (&kernelContext_99)->atom_0 = atom_6;
    (&kernelContext_99)->work_0 = work_6;
    (&kernelContext_99)->frames_0 = frames_6;
    (&kernelContext_99)->bodies_0 = bodies_6;
    (&kernelContext_99)->chunks_0 = chunks_6;
    (&kernelContext_99)->sort_0 = sort_6;
    (&kernelContext_99)->manifolds_0 = manifolds_6;
    (&kernelContext_99)->loads_0 = loads_6;
    (&kernelContext_99)->bonds_0 = bonds_6;
    (&kernelContext_99)->args_0 = args_6;
    (&kernelContext_99)->info_0 = info_6;
    (&kernelContext_99)->scan_0 = scan_6;
    (&kernelContext_99)->parts_0 = parts_6;
    (&kernelContext_99)->colliders_0 = colliders_6;
    (&kernelContext_99)->collider_info_0 = collider_info_6;
    (&kernelContext_99)->view_chunks_0 = view_chunks_6;
    (&kernelContext_99)->view_0 = view_6;
    (&kernelContext_99)->objects_0 = objects_6;
    (&kernelContext_99)->shot_bodies_0 = shot_bodies_6;
    (&kernelContext_99)->lines_0 = lines_6;
    (&kernelContext_99)->geometry_0 = geometry_6;
    (&kernelContext_99)->shots_0 = shots_6;
    (&kernelContext_99)->lp_0 = lp_6;
    (&kernelContext_99)->ext_0 = ext_7;
    (&kernelContext_99)->engine_contacts_0 = engine_contacts_6;
    (&kernelContext_99)->crush_0 = crush_6;
    (&kernelContext_99)->crush_rw_0 = crush_rw_6;
    (&kernelContext_99)->crush_tab_0 = crush_tab_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_99)->g_scan_0 = &g_scan_6;
    threadgroup array<float, int(64)> g_il_best_6;
    (&kernelContext_99)->g_il_best_0 = &g_il_best_6;
    threadgroup array<float, int(256)> g_gi_x_6;
    (&kernelContext_99)->g_gi_x_0 = &g_gi_x_6;
    threadgroup array<float, int(256)> g_gi_m_6;
    (&kernelContext_99)->g_gi_m_0 = &g_gi_m_6;
    threadgroup array<uint, int(1280)> g_gi_lo_6;
    (&kernelContext_99)->g_gi_lo_0 = &g_gi_lo_6;
    threadgroup array<uint, int(1280)> g_gi_hi_6;
    (&kernelContext_99)->g_gi_hi_0 = &g_gi_hi_6;
    uint i_9 = id_5.x;
    uint _S629 = w_host_breaks_0(&kernelContext_99);
    if(i_9 >= (*((&kernelContext_99)->work_0+_S629)))
    {
        return;
    }
    uint device* _S630 = (&kernelContext_99)->work_0+(_S629 + 1U + i_9);
    uint b_35 = *_S630;
    bool _S631;
    if((*_S630) >= ((&kernelContext_99)->params_0->bond_count_0))
    {
        _S631 = true;
    }
    else
    {
        uint _S632 = w_alive_0(&kernelContext_99);
        _S631 = (*((&kernelContext_99)->work_0+(_S632 + b_35))) == 0U;
    }
    if(_S631)
    {
        return;
    }
    uint _S633 = w_alive_0(&kernelContext_99);
    *((&kernelContext_99)->work_0+(_S633 + b_35)) = 0U;
    uint _S634 = a_dirty_0(&kernelContext_99);
    uint _S635 = body_of_0((uint4(*((&kernelContext_99)->bonds_0+b_35)) ).x, &kernelContext_99);
    uint _S636 = atomic_fetch_or_explicit((&kernelContext_99)->atom_0+(_S634 + _S635), 1U, memory_order_relaxed);
    uint _S637 = a_counters_1(&kernelContext_99);
    uint _S638 = atomic_fetch_add_explicit((&kernelContext_99)->atom_0+_S637, 1U, memory_order_relaxed);
    uint _S639 = atomic_fetch_add_explicit((&kernelContext_99)->atom_0+(_S637 + 1U), 1U, memory_order_relaxed);
    event_broken_0(b_35, &kernelContext_99);
    return;
}

[[kernel]] void fr_frame_begin(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], atomic<uint> device* atom_7 [[buffer(5)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], packed_float4 device* lp_7 [[buffer(24)]], packed_float4 device* ext_8 [[buffer(23)]], packed_float4 device* engine_contacts_7 [[buffer(25)]], CrushParams_0 constant* crush_7 [[buffer(28)]], packed_float4 device* crush_rw_7 [[buffer(27)]], packed_float4 device* crush_tab_7 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_100;
    (&kernelContext_100)->params_0 = params_7;
    (&kernelContext_100)->atom_0 = atom_7;
    (&kernelContext_100)->work_0 = work_7;
    (&kernelContext_100)->frames_0 = frames_7;
    (&kernelContext_100)->bodies_0 = bodies_7;
    (&kernelContext_100)->chunks_0 = chunks_7;
    (&kernelContext_100)->sort_0 = sort_7;
    (&kernelContext_100)->manifolds_0 = manifolds_7;
    (&kernelContext_100)->loads_0 = loads_7;
    (&kernelContext_100)->bonds_0 = bonds_7;
    (&kernelContext_100)->args_0 = args_7;
    (&kernelContext_100)->info_0 = info_7;
    (&kernelContext_100)->scan_0 = scan_7;
    (&kernelContext_100)->parts_0 = parts_7;
    (&kernelContext_100)->colliders_0 = colliders_7;
    (&kernelContext_100)->collider_info_0 = collider_info_7;
    (&kernelContext_100)->view_chunks_0 = view_chunks_7;
    (&kernelContext_100)->view_0 = view_7;
    (&kernelContext_100)->objects_0 = objects_7;
    (&kernelContext_100)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_100)->lines_0 = lines_7;
    (&kernelContext_100)->geometry_0 = geometry_7;
    (&kernelContext_100)->shots_0 = shots_7;
    (&kernelContext_100)->lp_0 = lp_7;
    (&kernelContext_100)->ext_0 = ext_8;
    (&kernelContext_100)->engine_contacts_0 = engine_contacts_7;
    (&kernelContext_100)->crush_0 = crush_7;
    (&kernelContext_100)->crush_rw_0 = crush_rw_7;
    (&kernelContext_100)->crush_tab_0 = crush_tab_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_100)->g_scan_0 = &g_scan_7;
    threadgroup array<float, int(64)> g_il_best_7;
    (&kernelContext_100)->g_il_best_0 = &g_il_best_7;
    threadgroup array<float, int(256)> g_gi_x_7;
    (&kernelContext_100)->g_gi_x_0 = &g_gi_x_7;
    threadgroup array<float, int(256)> g_gi_m_7;
    (&kernelContext_100)->g_gi_m_0 = &g_gi_m_7;
    threadgroup array<uint, int(1280)> g_gi_lo_7;
    (&kernelContext_100)->g_gi_lo_0 = &g_gi_lo_7;
    threadgroup array<uint, int(1280)> g_gi_hi_7;
    (&kernelContext_100)->g_gi_hi_0 = &g_gi_hi_7;
    uint _S640 = w_frame_0(&kernelContext_100);
    *((&kernelContext_100)->work_0+_S640) = 1U - *((&kernelContext_100)->work_0+_S640);
    uint _S641 = _S640 + 1U;
    *((&kernelContext_100)->work_0+_S641) = *((&kernelContext_100)->work_0+_S641) + 1U;
    uint _S642 = _S640 + 2U;
    *((&kernelContext_100)->work_0+_S642) = *((&kernelContext_100)->work_0+_S642) + 1U;
    return;
}

uint groups_of_0(uint n_3, uint size_0)
{
    uint _S643 = (n_3 + size_0 - 1U) / size_0;
    return _S643;
}

void set_args_0(uint at_8, uint x_3, KernelContext_0 thread* kernelContext_101)
{
    uint _S644 = 3U * at_8;
    *(kernelContext_101->args_0+_S644) = x_3;
    *(kernelContext_101->args_0+(_S644 + 1U)) = 1U;
    *(kernelContext_101->args_0+(_S644 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], atomic<uint> device* atom_8 [[buffer(5)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], packed_float4 device* lp_8 [[buffer(24)]], packed_float4 device* ext_9 [[buffer(23)]], packed_float4 device* engine_contacts_8 [[buffer(25)]], CrushParams_0 constant* crush_8 [[buffer(28)]], packed_float4 device* crush_rw_8 [[buffer(27)]], packed_float4 device* crush_tab_8 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_102;
    (&kernelContext_102)->params_0 = params_8;
    (&kernelContext_102)->atom_0 = atom_8;
    (&kernelContext_102)->work_0 = work_8;
    (&kernelContext_102)->frames_0 = frames_8;
    (&kernelContext_102)->bodies_0 = bodies_8;
    (&kernelContext_102)->chunks_0 = chunks_8;
    (&kernelContext_102)->sort_0 = sort_8;
    (&kernelContext_102)->manifolds_0 = manifolds_8;
    (&kernelContext_102)->loads_0 = loads_8;
    (&kernelContext_102)->bonds_0 = bonds_8;
    (&kernelContext_102)->args_0 = args_8;
    (&kernelContext_102)->info_0 = info_8;
    (&kernelContext_102)->scan_0 = scan_8;
    (&kernelContext_102)->parts_0 = parts_8;
    (&kernelContext_102)->colliders_0 = colliders_8;
    (&kernelContext_102)->collider_info_0 = collider_info_8;
    (&kernelContext_102)->view_chunks_0 = view_chunks_8;
    (&kernelContext_102)->view_0 = view_8;
    (&kernelContext_102)->objects_0 = objects_8;
    (&kernelContext_102)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_102)->lines_0 = lines_8;
    (&kernelContext_102)->geometry_0 = geometry_8;
    (&kernelContext_102)->shots_0 = shots_8;
    (&kernelContext_102)->lp_0 = lp_8;
    (&kernelContext_102)->ext_0 = ext_9;
    (&kernelContext_102)->engine_contacts_0 = engine_contacts_8;
    (&kernelContext_102)->crush_0 = crush_8;
    (&kernelContext_102)->crush_rw_0 = crush_rw_8;
    (&kernelContext_102)->crush_tab_0 = crush_tab_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_102)->g_scan_0 = &g_scan_8;
    threadgroup array<float, int(64)> g_il_best_8;
    (&kernelContext_102)->g_il_best_0 = &g_il_best_8;
    threadgroup array<float, int(256)> g_gi_x_8;
    (&kernelContext_102)->g_gi_x_0 = &g_gi_x_8;
    threadgroup array<float, int(256)> g_gi_m_8;
    (&kernelContext_102)->g_gi_m_0 = &g_gi_m_8;
    threadgroup array<uint, int(1280)> g_gi_lo_8;
    (&kernelContext_102)->g_gi_lo_0 = &g_gi_lo_8;
    threadgroup array<uint, int(1280)> g_gi_hi_8;
    (&kernelContext_102)->g_gi_hi_0 = &g_gi_hi_8;
    uint _S645 = a_counters_1(&kernelContext_102);
    uint _S646 = atomic_load_explicit((&kernelContext_102)->atom_0+_S645, memory_order_relaxed);
    bool run_0 = _S646 != 0U;
    atomic_store_explicit((&kernelContext_102)->atom_0+_S645, 0U, memory_order_relaxed);
    uint _S647 = _S645 + 5U;
    uint _S648 = frame_parity_0(&kernelContext_102);
    atomic_store_explicit((&kernelContext_102)->atom_0+(_S647 + _S648), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S649 = atomic_fetch_add_explicit((&kernelContext_102)->atom_0+(_S645 + 4U), 1U, memory_order_relaxed);
    }
    uint _S650;
    if(run_0)
    {
        uint _S651 = groups_of_0((&kernelContext_102)->params_0->chunk_used_0, 64U);
        _S650 = _S651;
    }
    else
    {
        _S650 = 0U;
    }
    set_args_0(0U, _S650, &kernelContext_102);
    if(run_0)
    {
        uint _S652 = groups_of_0((&kernelContext_102)->params_0->bond_used_0, 64U);
        _S650 = _S652;
    }
    else
    {
        _S650 = 0U;
    }
    set_args_0(1U, _S650, &kernelContext_102);
    if(run_0)
    {
        uint _S653 = groups_of_0((&kernelContext_102)->params_0->body_count_0, 64U);
        _S650 = _S653;
    }
    else
    {
        _S650 = 0U;
    }
    set_args_0(2U, _S650, &kernelContext_102);
    if(run_0)
    {
        uint _S654 = groups_of_0((&kernelContext_102)->params_0->chunk_used_0, 1024U);
        _S650 = _S654;
    }
    else
    {
        _S650 = 0U;
    }
    set_args_0(3U, _S650, &kernelContext_102);
    if(run_0)
    {
        uint _S655 = groups_of_0((&kernelContext_102)->params_0->body_count_0, 1024U);
        _S650 = _S655;
    }
    else
    {
        _S650 = 0U;
    }
    set_args_0(4U, _S650, &kernelContext_102);
    int _S656;
    if(run_0)
    {
        _S656 = int(1);
    }
    else
    {
        _S656 = int(0);
    }
    set_args_0(5U, uint(_S656), &kernelContext_102);
    return;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_103)
{
    return kernelContext_103->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_104)
{
    return 2U * kernelContext_104->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], atomic<uint> device* atom_9 [[buffer(5)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], packed_float4 device* lp_9 [[buffer(24)]], packed_float4 device* ext_10 [[buffer(23)]], packed_float4 device* engine_contacts_9 [[buffer(25)]], CrushParams_0 constant* crush_9 [[buffer(28)]], packed_float4 device* crush_rw_9 [[buffer(27)]], packed_float4 device* crush_tab_9 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_105;
    (&kernelContext_105)->params_0 = params_9;
    (&kernelContext_105)->atom_0 = atom_9;
    (&kernelContext_105)->work_0 = work_9;
    (&kernelContext_105)->frames_0 = frames_9;
    (&kernelContext_105)->bodies_0 = bodies_9;
    (&kernelContext_105)->chunks_0 = chunks_9;
    (&kernelContext_105)->sort_0 = sort_9;
    (&kernelContext_105)->manifolds_0 = manifolds_9;
    (&kernelContext_105)->loads_0 = loads_9;
    (&kernelContext_105)->bonds_0 = bonds_9;
    (&kernelContext_105)->args_0 = args_9;
    (&kernelContext_105)->info_0 = info_9;
    (&kernelContext_105)->scan_0 = scan_9;
    (&kernelContext_105)->parts_0 = parts_9;
    (&kernelContext_105)->colliders_0 = colliders_9;
    (&kernelContext_105)->collider_info_0 = collider_info_9;
    (&kernelContext_105)->view_chunks_0 = view_chunks_9;
    (&kernelContext_105)->view_0 = view_9;
    (&kernelContext_105)->objects_0 = objects_9;
    (&kernelContext_105)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_105)->lines_0 = lines_9;
    (&kernelContext_105)->geometry_0 = geometry_9;
    (&kernelContext_105)->shots_0 = shots_9;
    (&kernelContext_105)->lp_0 = lp_9;
    (&kernelContext_105)->ext_0 = ext_10;
    (&kernelContext_105)->engine_contacts_0 = engine_contacts_9;
    (&kernelContext_105)->crush_0 = crush_9;
    (&kernelContext_105)->crush_rw_0 = crush_rw_9;
    (&kernelContext_105)->crush_tab_0 = crush_tab_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_105)->g_scan_0 = &g_scan_9;
    threadgroup array<float, int(64)> g_il_best_9;
    (&kernelContext_105)->g_il_best_0 = &g_il_best_9;
    threadgroup array<float, int(256)> g_gi_x_9;
    (&kernelContext_105)->g_gi_x_0 = &g_gi_x_9;
    threadgroup array<float, int(256)> g_gi_m_9;
    (&kernelContext_105)->g_gi_m_0 = &g_gi_m_9;
    threadgroup array<uint, int(1280)> g_gi_lo_9;
    (&kernelContext_105)->g_gi_lo_0 = &g_gi_lo_9;
    threadgroup array<uint, int(1280)> g_gi_hi_9;
    (&kernelContext_105)->g_gi_hi_0 = &g_gi_hi_9;
    uint c_7 = id_8.x;
    bool _S657 = live_0(c_7, &kernelContext_105);
    bool _S658;
    if(!_S657)
    {
        _S658 = true;
    }
    else
    {
        uint _S659 = body_of_0(c_7, &kernelContext_105);
        bool _S660 = dirty_0(_S659, &kernelContext_105);
        _S658 = !_S660;
    }
    if(_S658)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_105)->atom_0+(a_parent_0() + c_7), c_7, memory_order_relaxed);
    uint _S661 = a_count_0(&kernelContext_105);
    atomic_store_explicit((&kernelContext_105)->atom_0+(_S661 + c_7), 0U, memory_order_relaxed);
    uint _S662 = a_anchor_0(&kernelContext_105);
    atomic_store_explicit((&kernelContext_105)->atom_0+(_S662 + c_7), 0U, memory_order_relaxed);
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
        uint _S663 = a_sums_0(&kernelContext_105);
        atomic_store_explicit((&kernelContext_105)->atom_0+(_S663 + 20U * c_7 + w_3), 0U, memory_order_relaxed);
        w_3 = w_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], atomic<uint> device* atom_10 [[buffer(5)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], packed_float4 device* lp_10 [[buffer(24)]], packed_float4 device* ext_11 [[buffer(23)]], packed_float4 device* engine_contacts_10 [[buffer(25)]], CrushParams_0 constant* crush_10 [[buffer(28)]], packed_float4 device* crush_rw_10 [[buffer(27)]], packed_float4 device* crush_tab_10 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_106;
    (&kernelContext_106)->params_0 = params_10;
    (&kernelContext_106)->atom_0 = atom_10;
    (&kernelContext_106)->work_0 = work_10;
    (&kernelContext_106)->frames_0 = frames_10;
    (&kernelContext_106)->bodies_0 = bodies_10;
    (&kernelContext_106)->chunks_0 = chunks_10;
    (&kernelContext_106)->sort_0 = sort_10;
    (&kernelContext_106)->manifolds_0 = manifolds_10;
    (&kernelContext_106)->loads_0 = loads_10;
    (&kernelContext_106)->bonds_0 = bonds_10;
    (&kernelContext_106)->args_0 = args_10;
    (&kernelContext_106)->info_0 = info_10;
    (&kernelContext_106)->scan_0 = scan_10;
    (&kernelContext_106)->parts_0 = parts_10;
    (&kernelContext_106)->colliders_0 = colliders_10;
    (&kernelContext_106)->collider_info_0 = collider_info_10;
    (&kernelContext_106)->view_chunks_0 = view_chunks_10;
    (&kernelContext_106)->view_0 = view_10;
    (&kernelContext_106)->objects_0 = objects_10;
    (&kernelContext_106)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_106)->lines_0 = lines_10;
    (&kernelContext_106)->geometry_0 = geometry_10;
    (&kernelContext_106)->shots_0 = shots_10;
    (&kernelContext_106)->lp_0 = lp_10;
    (&kernelContext_106)->ext_0 = ext_11;
    (&kernelContext_106)->engine_contacts_0 = engine_contacts_10;
    (&kernelContext_106)->crush_0 = crush_10;
    (&kernelContext_106)->crush_rw_0 = crush_rw_10;
    (&kernelContext_106)->crush_tab_0 = crush_tab_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_106)->g_scan_0 = &g_scan_10;
    threadgroup array<float, int(64)> g_il_best_10;
    (&kernelContext_106)->g_il_best_0 = &g_il_best_10;
    threadgroup array<float, int(256)> g_gi_x_10;
    (&kernelContext_106)->g_gi_x_0 = &g_gi_x_10;
    threadgroup array<float, int(256)> g_gi_m_10;
    (&kernelContext_106)->g_gi_m_0 = &g_gi_m_10;
    threadgroup array<uint, int(1280)> g_gi_lo_10;
    (&kernelContext_106)->g_gi_lo_0 = &g_gi_lo_10;
    threadgroup array<uint, int(1280)> g_gi_hi_10;
    (&kernelContext_106)->g_gi_hi_0 = &g_gi_hi_10;
    uint s_7 = id_9.x;
    bool _S664;
    if(s_7 >= (params_10->body_count_0))
    {
        _S664 = true;
    }
    else
    {
        bool _S665 = dirty_0(s_7, &kernelContext_106);
        _S664 = !_S665;
    }
    if(_S664)
    {
        return;
    }
    uint _S666 = a_best_count_0(&kernelContext_106);
    atomic_store_explicit((&kernelContext_106)->atom_0+(_S666 + s_7), 0U, memory_order_relaxed);
    uint _S667 = a_best_root_0(&kernelContext_106);
    atomic_store_explicit((&kernelContext_106)->atom_0+(_S667 + s_7), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_4, KernelContext_0 thread* kernelContext_107)
{
    uint _S668 = a_parent_0();
    uint _S669 = atomic_load_explicit(kernelContext_107->atom_0+(_S668 + x_4), memory_order_relaxed);
    uint p_11 = _S669;
    uint _S670 = x_4;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S671;
        if(guard_0 < 1048576U)
        {
            _S671 = p_11 != _S670;
        }
        else
        {
            _S671 = false;
        }
        if(_S671)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_107->atom_0+(_S668 + p_11), memory_order_relaxed);
        if(gp_0 != p_11)
        {
            uint _S672;
            {
            uint _metal_cas_comparand = p_11;
            atomic_compare_exchange_weak_explicit(kernelContext_107->atom_0+(_S668 + _S670), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S672 = _metal_cas_comparand;
            }
        }
        uint _S673 = guard_0 + 1U;
        _S670 = p_11;
        p_11 = gp_0;
        guard_0 = _S673;
    }
    return _S670;
}

[[kernel]] void fr_uf_union(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_11 [[buffer(0)]], atomic<uint> device* atom_11 [[buffer(5)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], packed_float4 device* lp_11 [[buffer(24)]], packed_float4 device* ext_12 [[buffer(23)]], packed_float4 device* engine_contacts_11 [[buffer(25)]], CrushParams_0 constant* crush_11 [[buffer(28)]], packed_float4 device* crush_rw_11 [[buffer(27)]], packed_float4 device* crush_tab_11 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_108;
    (&kernelContext_108)->params_0 = params_11;
    (&kernelContext_108)->atom_0 = atom_11;
    (&kernelContext_108)->work_0 = work_11;
    (&kernelContext_108)->frames_0 = frames_11;
    (&kernelContext_108)->bodies_0 = bodies_11;
    (&kernelContext_108)->chunks_0 = chunks_11;
    (&kernelContext_108)->sort_0 = sort_11;
    (&kernelContext_108)->manifolds_0 = manifolds_11;
    (&kernelContext_108)->loads_0 = loads_11;
    (&kernelContext_108)->bonds_0 = bonds_11;
    (&kernelContext_108)->args_0 = args_11;
    (&kernelContext_108)->info_0 = info_11;
    (&kernelContext_108)->scan_0 = scan_11;
    (&kernelContext_108)->parts_0 = parts_11;
    (&kernelContext_108)->colliders_0 = colliders_11;
    (&kernelContext_108)->collider_info_0 = collider_info_11;
    (&kernelContext_108)->view_chunks_0 = view_chunks_11;
    (&kernelContext_108)->view_0 = view_11;
    (&kernelContext_108)->objects_0 = objects_11;
    (&kernelContext_108)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_108)->lines_0 = lines_11;
    (&kernelContext_108)->geometry_0 = geometry_11;
    (&kernelContext_108)->shots_0 = shots_11;
    (&kernelContext_108)->lp_0 = lp_11;
    (&kernelContext_108)->ext_0 = ext_12;
    (&kernelContext_108)->engine_contacts_0 = engine_contacts_11;
    (&kernelContext_108)->crush_0 = crush_11;
    (&kernelContext_108)->crush_rw_0 = crush_rw_11;
    (&kernelContext_108)->crush_tab_0 = crush_tab_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_108)->g_scan_0 = &g_scan_11;
    threadgroup array<float, int(64)> g_il_best_11;
    (&kernelContext_108)->g_il_best_0 = &g_il_best_11;
    threadgroup array<float, int(256)> g_gi_x_11;
    (&kernelContext_108)->g_gi_x_0 = &g_gi_x_11;
    threadgroup array<float, int(256)> g_gi_m_11;
    (&kernelContext_108)->g_gi_m_0 = &g_gi_m_11;
    threadgroup array<uint, int(1280)> g_gi_lo_11;
    (&kernelContext_108)->g_gi_lo_0 = &g_gi_lo_11;
    threadgroup array<uint, int(1280)> g_gi_hi_11;
    (&kernelContext_108)->g_gi_hi_0 = &g_gi_hi_11;
    uint i_10 = id_10.x;
    bool _S674;
    if(i_10 >= (params_11->bond_used_0))
    {
        _S674 = true;
    }
    else
    {
        uint _S675 = w_alive_0(&kernelContext_108);
        _S674 = (*((&kernelContext_108)->work_0+(_S675 + i_10))) == 0U;
    }
    if(_S674)
    {
        return;
    }
    uint4 _S676 = uint4(*((&kernelContext_108)->bonds_0+i_10)) ;
    uint _S677 = _S676.x;
    uint _S678 = body_of_0(_S677, &kernelContext_108);
    bool _S679 = dirty_0(_S678, &kernelContext_108);
    if(!_S679)
    {
        return;
    }
    uint _S680 = _S676.y;
    uint a_6 = _S677;
    uint b_36 = _S680;
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
        uint _S681 = find_0(a_6, &kernelContext_108);
        uint _S682 = find_0(b_36, &kernelContext_108);
        if(_S681 == _S682)
        {
            return;
        }
        uint _S683 = max(_S681, _S682);
        uint _S684 = min(_S681, _S682);
        uint was_0;
        {
        uint _metal_cas_comparand = _S683;
        atomic_compare_exchange_weak_explicit((&kernelContext_108)->atom_0+(a_parent_0() + _S683), &_metal_cas_comparand, _S684, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S683)
        {
            return;
        }
        uint _S685 = guard_1 + 1U;
        a_6 = _S683;
        b_36 = _S684;
        guard_1 = _S685;
    }
    uint _S686 = a_counters_1(&kernelContext_108);
    uint _S687 = atomic_fetch_or_explicit((&kernelContext_108)->atom_0+(_S686 + 3U), 4U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_12 [[buffer(0)]], atomic<uint> device* atom_12 [[buffer(5)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], packed_float4 device* lp_12 [[buffer(24)]], packed_float4 device* ext_13 [[buffer(23)]], packed_float4 device* engine_contacts_12 [[buffer(25)]], CrushParams_0 constant* crush_12 [[buffer(28)]], packed_float4 device* crush_rw_12 [[buffer(27)]], packed_float4 device* crush_tab_12 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_109;
    (&kernelContext_109)->params_0 = params_12;
    (&kernelContext_109)->atom_0 = atom_12;
    (&kernelContext_109)->work_0 = work_12;
    (&kernelContext_109)->frames_0 = frames_12;
    (&kernelContext_109)->bodies_0 = bodies_12;
    (&kernelContext_109)->chunks_0 = chunks_12;
    (&kernelContext_109)->sort_0 = sort_12;
    (&kernelContext_109)->manifolds_0 = manifolds_12;
    (&kernelContext_109)->loads_0 = loads_12;
    (&kernelContext_109)->bonds_0 = bonds_12;
    (&kernelContext_109)->args_0 = args_12;
    (&kernelContext_109)->info_0 = info_12;
    (&kernelContext_109)->scan_0 = scan_12;
    (&kernelContext_109)->parts_0 = parts_12;
    (&kernelContext_109)->colliders_0 = colliders_12;
    (&kernelContext_109)->collider_info_0 = collider_info_12;
    (&kernelContext_109)->view_chunks_0 = view_chunks_12;
    (&kernelContext_109)->view_0 = view_12;
    (&kernelContext_109)->objects_0 = objects_12;
    (&kernelContext_109)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_109)->lines_0 = lines_12;
    (&kernelContext_109)->geometry_0 = geometry_12;
    (&kernelContext_109)->shots_0 = shots_12;
    (&kernelContext_109)->lp_0 = lp_12;
    (&kernelContext_109)->ext_0 = ext_13;
    (&kernelContext_109)->engine_contacts_0 = engine_contacts_12;
    (&kernelContext_109)->crush_0 = crush_12;
    (&kernelContext_109)->crush_rw_0 = crush_rw_12;
    (&kernelContext_109)->crush_tab_0 = crush_tab_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_109)->g_scan_0 = &g_scan_12;
    threadgroup array<float, int(64)> g_il_best_12;
    (&kernelContext_109)->g_il_best_0 = &g_il_best_12;
    threadgroup array<float, int(256)> g_gi_x_12;
    (&kernelContext_109)->g_gi_x_0 = &g_gi_x_12;
    threadgroup array<float, int(256)> g_gi_m_12;
    (&kernelContext_109)->g_gi_m_0 = &g_gi_m_12;
    threadgroup array<uint, int(1280)> g_gi_lo_12;
    (&kernelContext_109)->g_gi_lo_0 = &g_gi_lo_12;
    threadgroup array<uint, int(1280)> g_gi_hi_12;
    (&kernelContext_109)->g_gi_hi_0 = &g_gi_hi_12;
    uint c_8 = id_11.x;
    bool _S688 = live_0(c_8, &kernelContext_109);
    bool _S689;
    if(!_S688)
    {
        _S689 = true;
    }
    else
    {
        uint _S690 = body_of_0(c_8, &kernelContext_109);
        bool _S691 = dirty_0(_S690, &kernelContext_109);
        _S689 = !_S691;
    }
    if(_S689)
    {
        return;
    }
    uint _S692 = find_0(c_8, &kernelContext_109);
    uint _S693 = w_label_0(&kernelContext_109);
    *((&kernelContext_109)->work_0+(_S693 + c_8)) = _S692;
    uint _S694 = a_count_0(&kernelContext_109);
    uint _S695 = atomic_fetch_add_explicit((&kernelContext_109)->atom_0+(_S694 + _S692), 1U, memory_order_relaxed);
    if(((&kernelContext_109)->info_0[c_8] & 1U) != 0U)
    {
        uint _S696 = a_anchor_0(&kernelContext_109);
        uint _S697 = atomic_fetch_or_explicit((&kernelContext_109)->atom_0+(_S696 + _S692), 1U, memory_order_relaxed);
    }
    uint _S698 = 4U * c_8;
    float4 _S699 = float4(*((&kernelContext_109)->chunks_0+_S698)) ;
    float4 _S700 = float4(*((&kernelContext_109)->chunks_0+(_S698 + 1U))) ;
    float4 _S701 = float4(*((&kernelContext_109)->chunks_0+(_S698 + 2U))) ;
    float m_2 = _S699.w;
    float3 d_19 = _S699.xyz - (float4(*((&kernelContext_109)->chunks_0+4U * _S692)) ).xyz;
    float dd_0 = dot(d_19, d_19);
    uint _S702 = a_sums_0(&kernelContext_109);
    uint at_9 = _S702 + 20U * _S692;
    add64_0(at_9, m_2, (&kernelContext_109)->params_0->scale_m_0, &kernelContext_109);
    float _S703 = d_19.x;
    float _S704 = m_2 * _S703;
    add64_0(at_9 + 2U, _S704, (&kernelContext_109)->params_0->scale_s_0, &kernelContext_109);
    float _S705 = d_19.y;
    float _S706 = m_2 * _S705;
    add64_0(at_9 + 4U, _S706, (&kernelContext_109)->params_0->scale_s_0, &kernelContext_109);
    float _S707 = d_19.z;
    add64_0(at_9 + 6U, m_2 * _S707, (&kernelContext_109)->params_0->scale_s_0, &kernelContext_109);
    add64_0(at_9 + 8U, _S700.x + m_2 * (dd_0 - _S703 * _S703), (&kernelContext_109)->params_0->scale_j_0, &kernelContext_109);
    add64_0(at_9 + 10U, _S700.y + m_2 * (dd_0 - _S705 * _S705), (&kernelContext_109)->params_0->scale_j_0, &kernelContext_109);
    add64_0(at_9 + 12U, _S700.z + m_2 * (dd_0 - _S707 * _S707), (&kernelContext_109)->params_0->scale_j_0, &kernelContext_109);
    add64_0(at_9 + 14U, _S700.w - _S704 * _S705, (&kernelContext_109)->params_0->scale_j_0, &kernelContext_109);
    add64_0(at_9 + 16U, _S701.x - _S704 * _S707, (&kernelContext_109)->params_0->scale_j_0, &kernelContext_109);
    add64_0(at_9 + 18U, _S701.y - _S706 * _S707, (&kernelContext_109)->params_0->scale_j_0, &kernelContext_109);
    return;
}

bool is_root_0(uint c_9, KernelContext_0 thread* kernelContext_110)
{
    bool _S708 = live_0(c_9, kernelContext_110);
    bool _S709;
    if(_S708)
    {
        uint _S710 = body_of_0(c_9, kernelContext_110);
        bool _S711 = dirty_0(_S710, kernelContext_110);
        _S709 = _S711;
    }
    else
    {
        _S709 = false;
    }
    if(_S709)
    {
        uint _S712 = w_label_0(kernelContext_110);
        _S709 = (*(kernelContext_110->work_0+(_S712 + c_9))) == c_9;
    }
    else
    {
        _S709 = false;
    }
    return _S709;
}

[[kernel]] void fr_best_count(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], atomic<uint> device* atom_13 [[buffer(5)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], packed_float4 device* lp_13 [[buffer(24)]], packed_float4 device* ext_14 [[buffer(23)]], packed_float4 device* engine_contacts_13 [[buffer(25)]], CrushParams_0 constant* crush_13 [[buffer(28)]], packed_float4 device* crush_rw_13 [[buffer(27)]], packed_float4 device* crush_tab_13 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_111;
    (&kernelContext_111)->params_0 = params_13;
    (&kernelContext_111)->atom_0 = atom_13;
    (&kernelContext_111)->work_0 = work_13;
    (&kernelContext_111)->frames_0 = frames_13;
    (&kernelContext_111)->bodies_0 = bodies_13;
    (&kernelContext_111)->chunks_0 = chunks_13;
    (&kernelContext_111)->sort_0 = sort_13;
    (&kernelContext_111)->manifolds_0 = manifolds_13;
    (&kernelContext_111)->loads_0 = loads_13;
    (&kernelContext_111)->bonds_0 = bonds_13;
    (&kernelContext_111)->args_0 = args_13;
    (&kernelContext_111)->info_0 = info_13;
    (&kernelContext_111)->scan_0 = scan_13;
    (&kernelContext_111)->parts_0 = parts_13;
    (&kernelContext_111)->colliders_0 = colliders_13;
    (&kernelContext_111)->collider_info_0 = collider_info_13;
    (&kernelContext_111)->view_chunks_0 = view_chunks_13;
    (&kernelContext_111)->view_0 = view_13;
    (&kernelContext_111)->objects_0 = objects_13;
    (&kernelContext_111)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_111)->lines_0 = lines_13;
    (&kernelContext_111)->geometry_0 = geometry_13;
    (&kernelContext_111)->shots_0 = shots_13;
    (&kernelContext_111)->lp_0 = lp_13;
    (&kernelContext_111)->ext_0 = ext_14;
    (&kernelContext_111)->engine_contacts_0 = engine_contacts_13;
    (&kernelContext_111)->crush_0 = crush_13;
    (&kernelContext_111)->crush_rw_0 = crush_rw_13;
    (&kernelContext_111)->crush_tab_0 = crush_tab_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_111)->g_scan_0 = &g_scan_13;
    threadgroup array<float, int(64)> g_il_best_13;
    (&kernelContext_111)->g_il_best_0 = &g_il_best_13;
    threadgroup array<float, int(256)> g_gi_x_13;
    (&kernelContext_111)->g_gi_x_0 = &g_gi_x_13;
    threadgroup array<float, int(256)> g_gi_m_13;
    (&kernelContext_111)->g_gi_m_0 = &g_gi_m_13;
    threadgroup array<uint, int(1280)> g_gi_lo_13;
    (&kernelContext_111)->g_gi_lo_0 = &g_gi_lo_13;
    threadgroup array<uint, int(1280)> g_gi_hi_13;
    (&kernelContext_111)->g_gi_hi_0 = &g_gi_hi_13;
    uint c_10 = id_12.x;
    bool _S713 = is_root_0(c_10, &kernelContext_111);
    if(!_S713)
    {
        return;
    }
    uint _S714 = a_best_count_0(&kernelContext_111);
    uint _S715 = body_of_0(c_10, &kernelContext_111);
    atomic<uint> device* _S716 = (&kernelContext_111)->atom_0+(_S714 + _S715);
    uint _S717 = a_count_0(&kernelContext_111);
    uint _S718 = atomic_load_explicit((&kernelContext_111)->atom_0+(_S717 + c_10), memory_order_relaxed);
    uint _S719 = atomic_fetch_max_explicit(_S716, _S718, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], atomic<uint> device* atom_14 [[buffer(5)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], packed_float4 device* lp_14 [[buffer(24)]], packed_float4 device* ext_15 [[buffer(23)]], packed_float4 device* engine_contacts_14 [[buffer(25)]], CrushParams_0 constant* crush_14 [[buffer(28)]], packed_float4 device* crush_rw_14 [[buffer(27)]], packed_float4 device* crush_tab_14 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_112;
    (&kernelContext_112)->params_0 = params_14;
    (&kernelContext_112)->atom_0 = atom_14;
    (&kernelContext_112)->work_0 = work_14;
    (&kernelContext_112)->frames_0 = frames_14;
    (&kernelContext_112)->bodies_0 = bodies_14;
    (&kernelContext_112)->chunks_0 = chunks_14;
    (&kernelContext_112)->sort_0 = sort_14;
    (&kernelContext_112)->manifolds_0 = manifolds_14;
    (&kernelContext_112)->loads_0 = loads_14;
    (&kernelContext_112)->bonds_0 = bonds_14;
    (&kernelContext_112)->args_0 = args_14;
    (&kernelContext_112)->info_0 = info_14;
    (&kernelContext_112)->scan_0 = scan_14;
    (&kernelContext_112)->parts_0 = parts_14;
    (&kernelContext_112)->colliders_0 = colliders_14;
    (&kernelContext_112)->collider_info_0 = collider_info_14;
    (&kernelContext_112)->view_chunks_0 = view_chunks_14;
    (&kernelContext_112)->view_0 = view_14;
    (&kernelContext_112)->objects_0 = objects_14;
    (&kernelContext_112)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_112)->lines_0 = lines_14;
    (&kernelContext_112)->geometry_0 = geometry_14;
    (&kernelContext_112)->shots_0 = shots_14;
    (&kernelContext_112)->lp_0 = lp_14;
    (&kernelContext_112)->ext_0 = ext_15;
    (&kernelContext_112)->engine_contacts_0 = engine_contacts_14;
    (&kernelContext_112)->crush_0 = crush_14;
    (&kernelContext_112)->crush_rw_0 = crush_rw_14;
    (&kernelContext_112)->crush_tab_0 = crush_tab_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_112)->g_scan_0 = &g_scan_14;
    threadgroup array<float, int(64)> g_il_best_14;
    (&kernelContext_112)->g_il_best_0 = &g_il_best_14;
    threadgroup array<float, int(256)> g_gi_x_14;
    (&kernelContext_112)->g_gi_x_0 = &g_gi_x_14;
    threadgroup array<float, int(256)> g_gi_m_14;
    (&kernelContext_112)->g_gi_m_0 = &g_gi_m_14;
    threadgroup array<uint, int(1280)> g_gi_lo_14;
    (&kernelContext_112)->g_gi_lo_0 = &g_gi_lo_14;
    threadgroup array<uint, int(1280)> g_gi_hi_14;
    (&kernelContext_112)->g_gi_hi_0 = &g_gi_hi_14;
    uint c_11 = id_13.x;
    bool _S720 = is_root_0(c_11, &kernelContext_112);
    if(!_S720)
    {
        return;
    }
    uint _S721 = body_of_0(c_11, &kernelContext_112);
    uint _S722 = a_count_0(&kernelContext_112);
    uint _S723 = atomic_load_explicit((&kernelContext_112)->atom_0+(_S722 + c_11), memory_order_relaxed);
    uint _S724 = a_best_count_0(&kernelContext_112);
    uint _S725 = atomic_load_explicit((&kernelContext_112)->atom_0+(_S724 + _S721), memory_order_relaxed);
    if(_S723 == _S725)
    {
        uint _S726 = a_best_root_0(&kernelContext_112);
        uint _S727 = atomic_fetch_min_explicit((&kernelContext_112)->atom_0+(_S726 + _S721), c_11, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], atomic<uint> device* atom_15 [[buffer(5)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], packed_float4 device* lp_15 [[buffer(24)]], packed_float4 device* ext_16 [[buffer(23)]], packed_float4 device* engine_contacts_15 [[buffer(25)]], CrushParams_0 constant* crush_15 [[buffer(28)]], packed_float4 device* crush_rw_15 [[buffer(27)]], packed_float4 device* crush_tab_15 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_113;
    (&kernelContext_113)->params_0 = params_15;
    (&kernelContext_113)->atom_0 = atom_15;
    (&kernelContext_113)->work_0 = work_15;
    (&kernelContext_113)->frames_0 = frames_15;
    (&kernelContext_113)->bodies_0 = bodies_15;
    (&kernelContext_113)->chunks_0 = chunks_15;
    (&kernelContext_113)->sort_0 = sort_15;
    (&kernelContext_113)->manifolds_0 = manifolds_15;
    (&kernelContext_113)->loads_0 = loads_15;
    (&kernelContext_113)->bonds_0 = bonds_15;
    (&kernelContext_113)->args_0 = args_15;
    (&kernelContext_113)->info_0 = info_15;
    (&kernelContext_113)->scan_0 = scan_15;
    (&kernelContext_113)->parts_0 = parts_15;
    (&kernelContext_113)->colliders_0 = colliders_15;
    (&kernelContext_113)->collider_info_0 = collider_info_15;
    (&kernelContext_113)->view_chunks_0 = view_chunks_15;
    (&kernelContext_113)->view_0 = view_15;
    (&kernelContext_113)->objects_0 = objects_15;
    (&kernelContext_113)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_113)->lines_0 = lines_15;
    (&kernelContext_113)->geometry_0 = geometry_15;
    (&kernelContext_113)->shots_0 = shots_15;
    (&kernelContext_113)->lp_0 = lp_15;
    (&kernelContext_113)->ext_0 = ext_16;
    (&kernelContext_113)->engine_contacts_0 = engine_contacts_15;
    (&kernelContext_113)->crush_0 = crush_15;
    (&kernelContext_113)->crush_rw_0 = crush_rw_15;
    (&kernelContext_113)->crush_tab_0 = crush_tab_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_113)->g_scan_0 = &g_scan_15;
    threadgroup array<float, int(64)> g_il_best_15;
    (&kernelContext_113)->g_il_best_0 = &g_il_best_15;
    threadgroup array<float, int(256)> g_gi_x_15;
    (&kernelContext_113)->g_gi_x_0 = &g_gi_x_15;
    threadgroup array<float, int(256)> g_gi_m_15;
    (&kernelContext_113)->g_gi_m_0 = &g_gi_m_15;
    threadgroup array<uint, int(1280)> g_gi_lo_15;
    (&kernelContext_113)->g_gi_lo_0 = &g_gi_lo_15;
    threadgroup array<uint, int(1280)> g_gi_hi_15;
    (&kernelContext_113)->g_gi_hi_0 = &g_gi_hi_15;
    uint c_12 = id_14.x;
    if(c_12 >= (params_15->chunk_used_0))
    {
        return;
    }
    uint _S728 = w_new_0(&kernelContext_113);
    uint device* _S729 = (&kernelContext_113)->work_0+(_S728 + c_12);
    bool _S730 = is_root_0(c_12, &kernelContext_113);
    bool _S731;
    if(_S730)
    {
        uint _S732 = a_best_root_0(&kernelContext_113);
        uint _S733 = body_of_0(c_12, &kernelContext_113);
        uint _S734 = atomic_load_explicit((&kernelContext_113)->atom_0+(_S732 + _S733), memory_order_relaxed);
        _S731 = _S734 != c_12;
    }
    else
    {
        _S731 = false;
    }
    uint _S735;
    if(_S731)
    {
        _S735 = 1U;
    }
    else
    {
        _S735 = 0U;
    }
    *_S729 = _S735;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_16 [[buffer(0)]], atomic<uint> device* atom_16 [[buffer(5)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], packed_float4 device* lp_16 [[buffer(24)]], packed_float4 device* ext_17 [[buffer(23)]], packed_float4 device* engine_contacts_16 [[buffer(25)]], CrushParams_0 constant* crush_16 [[buffer(28)]], packed_float4 device* crush_rw_16 [[buffer(27)]], packed_float4 device* crush_tab_16 [[buffer(26)]])
{
    uint step_4;
    uint add_0;
    thread KernelContext_0 kernelContext_114;
    (&kernelContext_114)->params_0 = params_16;
    (&kernelContext_114)->atom_0 = atom_16;
    (&kernelContext_114)->work_0 = work_16;
    (&kernelContext_114)->frames_0 = frames_16;
    (&kernelContext_114)->bodies_0 = bodies_16;
    (&kernelContext_114)->chunks_0 = chunks_16;
    (&kernelContext_114)->sort_0 = sort_16;
    (&kernelContext_114)->manifolds_0 = manifolds_16;
    (&kernelContext_114)->loads_0 = loads_16;
    (&kernelContext_114)->bonds_0 = bonds_16;
    (&kernelContext_114)->args_0 = args_16;
    (&kernelContext_114)->info_0 = info_16;
    (&kernelContext_114)->scan_0 = scan_16;
    (&kernelContext_114)->parts_0 = parts_16;
    (&kernelContext_114)->colliders_0 = colliders_16;
    (&kernelContext_114)->collider_info_0 = collider_info_16;
    (&kernelContext_114)->view_chunks_0 = view_chunks_16;
    (&kernelContext_114)->view_0 = view_16;
    (&kernelContext_114)->objects_0 = objects_16;
    (&kernelContext_114)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_114)->lines_0 = lines_16;
    (&kernelContext_114)->geometry_0 = geometry_16;
    (&kernelContext_114)->shots_0 = shots_16;
    (&kernelContext_114)->lp_0 = lp_16;
    (&kernelContext_114)->ext_0 = ext_17;
    (&kernelContext_114)->engine_contacts_0 = engine_contacts_16;
    (&kernelContext_114)->crush_0 = crush_16;
    (&kernelContext_114)->crush_rw_0 = crush_rw_16;
    (&kernelContext_114)->crush_tab_0 = crush_tab_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_114)->g_scan_0 = &g_scan_16;
    threadgroup array<float, int(64)> g_il_best_16;
    (&kernelContext_114)->g_il_best_0 = &g_il_best_16;
    threadgroup array<float, int(256)> g_gi_x_16;
    (&kernelContext_114)->g_gi_x_0 = &g_gi_x_16;
    threadgroup array<float, int(256)> g_gi_m_16;
    (&kernelContext_114)->g_gi_m_0 = &g_gi_m_16;
    threadgroup array<uint, int(1280)> g_gi_lo_16;
    (&kernelContext_114)->g_gi_lo_0 = &g_gi_lo_16;
    threadgroup array<uint, int(1280)> g_gi_hi_16;
    (&kernelContext_114)->g_gi_hi_0 = &g_gi_hi_16;
    uint t_7 = thread_2.x;
    uint _S736 = group_0.x;
    uint _S737 = 1024U * _S736 + 4U * t_7;
    thread array<uint, int(4)> v_11;
    uint j_6 = 0U;
    uint sum_2 = 0U;
    for(;;)
    {
        if(j_6 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S737 + j_6) < ((&kernelContext_114)->scan_0->n_0))
        {
            step_4 = *((&kernelContext_114)->work_0+((&kernelContext_114)->scan_0->in_base_0 + _S737 + j_6));
        }
        else
        {
            step_4 = 0U;
        }
        v_11[j_6] = step_4;
        uint sum_3 = sum_2 + step_4;
        j_6 = j_6 + 1U;
        sum_2 = sum_3;
    }
    (*(&kernelContext_114)->g_scan_0)[t_7] = sum_2;
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
            add_0 = (*(&kernelContext_114)->g_scan_0)[t_7 - step_4];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_114)->g_scan_0)[t_7] = (*(&kernelContext_114)->g_scan_0)[t_7] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_4 = step_4 << 1U;
    }
    uint _S738 = (*(&kernelContext_114)->g_scan_0)[t_7] - sum_2;
    j_6 = 0U;
    add_0 = _S738;
    for(;;)
    {
        if(j_6 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S737 + j_6) < ((&kernelContext_114)->scan_0->n_0))
        {
            *((&kernelContext_114)->work_0+((&kernelContext_114)->scan_0->out_base_0 + _S737 + j_6)) = add_0;
        }
        uint run_1 = add_0 + v_11[j_6];
        j_6 = j_6 + 1U;
        add_0 = run_1;
    }
    if(t_7 == 255U)
    {
        *((&kernelContext_114)->work_0+((&kernelContext_114)->scan_0->block_base_0 + _S736)) = (*(&kernelContext_114)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_3 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_17 [[buffer(0)]], atomic<uint> device* atom_17 [[buffer(5)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], packed_float4 device* lp_17 [[buffer(24)]], packed_float4 device* ext_18 [[buffer(23)]], packed_float4 device* engine_contacts_17 [[buffer(25)]], CrushParams_0 constant* crush_17 [[buffer(28)]], packed_float4 device* crush_rw_17 [[buffer(27)]], packed_float4 device* crush_tab_17 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_115;
    (&kernelContext_115)->params_0 = params_17;
    (&kernelContext_115)->atom_0 = atom_17;
    (&kernelContext_115)->work_0 = work_17;
    (&kernelContext_115)->frames_0 = frames_17;
    (&kernelContext_115)->bodies_0 = bodies_17;
    (&kernelContext_115)->chunks_0 = chunks_17;
    (&kernelContext_115)->sort_0 = sort_17;
    (&kernelContext_115)->manifolds_0 = manifolds_17;
    (&kernelContext_115)->loads_0 = loads_17;
    (&kernelContext_115)->bonds_0 = bonds_17;
    (&kernelContext_115)->args_0 = args_17;
    (&kernelContext_115)->info_0 = info_17;
    (&kernelContext_115)->scan_0 = scan_17;
    (&kernelContext_115)->parts_0 = parts_17;
    (&kernelContext_115)->colliders_0 = colliders_17;
    (&kernelContext_115)->collider_info_0 = collider_info_17;
    (&kernelContext_115)->view_chunks_0 = view_chunks_17;
    (&kernelContext_115)->view_0 = view_17;
    (&kernelContext_115)->objects_0 = objects_17;
    (&kernelContext_115)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_115)->lines_0 = lines_17;
    (&kernelContext_115)->geometry_0 = geometry_17;
    (&kernelContext_115)->shots_0 = shots_17;
    (&kernelContext_115)->lp_0 = lp_17;
    (&kernelContext_115)->ext_0 = ext_18;
    (&kernelContext_115)->engine_contacts_0 = engine_contacts_17;
    (&kernelContext_115)->crush_0 = crush_17;
    (&kernelContext_115)->crush_rw_0 = crush_rw_17;
    (&kernelContext_115)->crush_tab_0 = crush_tab_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_115)->g_scan_0 = &g_scan_17;
    threadgroup array<float, int(64)> g_il_best_17;
    (&kernelContext_115)->g_il_best_0 = &g_il_best_17;
    threadgroup array<float, int(256)> g_gi_x_17;
    (&kernelContext_115)->g_gi_x_0 = &g_gi_x_17;
    threadgroup array<float, int(256)> g_gi_m_17;
    (&kernelContext_115)->g_gi_m_0 = &g_gi_m_17;
    threadgroup array<uint, int(1280)> g_gi_lo_17;
    (&kernelContext_115)->g_gi_lo_0 = &g_gi_lo_17;
    threadgroup array<uint, int(1280)> g_gi_hi_17;
    (&kernelContext_115)->g_gi_hi_0 = &g_gi_hi_17;
    uint t_8 = thread_3.x;
    uint blocks_0 = (scan_17->n_0 + 1023U) / 1024U;
    bool _S739 = t_8 < blocks_0;
    uint v_12;
    if(_S739)
    {
        v_12 = *((&kernelContext_115)->work_0+((&kernelContext_115)->scan_0->block_base_0 + t_8));
    }
    else
    {
        v_12 = 0U;
    }
    (*(&kernelContext_115)->g_scan_0)[t_8] = v_12;
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
            add_1 = (*(&kernelContext_115)->g_scan_0)[t_8 - step_5];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_115)->g_scan_0)[t_8] = (*(&kernelContext_115)->g_scan_0)[t_8] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_5 = step_5 << 1U;
    }
    if(_S739)
    {
        *((&kernelContext_115)->work_0+((&kernelContext_115)->scan_0->block_base_0 + t_8)) = (*(&kernelContext_115)->g_scan_0)[t_8] - v_12;
    }
    if(t_8 == 1023U)
    {
        *((&kernelContext_115)->work_0+((&kernelContext_115)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_115)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], atomic<uint> device* atom_18 [[buffer(5)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], packed_float4 device* lp_18 [[buffer(24)]], packed_float4 device* ext_19 [[buffer(23)]], packed_float4 device* engine_contacts_18 [[buffer(25)]], CrushParams_0 constant* crush_18 [[buffer(28)]], packed_float4 device* crush_rw_18 [[buffer(27)]], packed_float4 device* crush_tab_18 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_116;
    (&kernelContext_116)->params_0 = params_18;
    (&kernelContext_116)->atom_0 = atom_18;
    (&kernelContext_116)->work_0 = work_18;
    (&kernelContext_116)->frames_0 = frames_18;
    (&kernelContext_116)->bodies_0 = bodies_18;
    (&kernelContext_116)->chunks_0 = chunks_18;
    (&kernelContext_116)->sort_0 = sort_18;
    (&kernelContext_116)->manifolds_0 = manifolds_18;
    (&kernelContext_116)->loads_0 = loads_18;
    (&kernelContext_116)->bonds_0 = bonds_18;
    (&kernelContext_116)->args_0 = args_18;
    (&kernelContext_116)->info_0 = info_18;
    (&kernelContext_116)->scan_0 = scan_18;
    (&kernelContext_116)->parts_0 = parts_18;
    (&kernelContext_116)->colliders_0 = colliders_18;
    (&kernelContext_116)->collider_info_0 = collider_info_18;
    (&kernelContext_116)->view_chunks_0 = view_chunks_18;
    (&kernelContext_116)->view_0 = view_18;
    (&kernelContext_116)->objects_0 = objects_18;
    (&kernelContext_116)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_116)->lines_0 = lines_18;
    (&kernelContext_116)->geometry_0 = geometry_18;
    (&kernelContext_116)->shots_0 = shots_18;
    (&kernelContext_116)->lp_0 = lp_18;
    (&kernelContext_116)->ext_0 = ext_19;
    (&kernelContext_116)->engine_contacts_0 = engine_contacts_18;
    (&kernelContext_116)->crush_0 = crush_18;
    (&kernelContext_116)->crush_rw_0 = crush_rw_18;
    (&kernelContext_116)->crush_tab_0 = crush_tab_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_116)->g_scan_0 = &g_scan_18;
    threadgroup array<float, int(64)> g_il_best_18;
    (&kernelContext_116)->g_il_best_0 = &g_il_best_18;
    threadgroup array<float, int(256)> g_gi_x_18;
    (&kernelContext_116)->g_gi_x_0 = &g_gi_x_18;
    threadgroup array<float, int(256)> g_gi_m_18;
    (&kernelContext_116)->g_gi_m_0 = &g_gi_m_18;
    threadgroup array<uint, int(1280)> g_gi_lo_18;
    (&kernelContext_116)->g_gi_lo_0 = &g_gi_lo_18;
    threadgroup array<uint, int(1280)> g_gi_hi_18;
    (&kernelContext_116)->g_gi_hi_0 = &g_gi_hi_18;
    uint i_11 = id_15.x;
    if(i_11 >= (scan_18->n_0))
    {
        return;
    }
    uint device* _S740 = (&kernelContext_116)->work_0+((&kernelContext_116)->scan_0->out_base_0 + i_11);
    *_S740 = *_S740 + *((&kernelContext_116)->work_0+((&kernelContext_116)->scan_0->block_base_0 + i_11 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], atomic<uint> device* atom_19 [[buffer(5)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], packed_float4 device* lp_19 [[buffer(24)]], packed_float4 device* ext_20 [[buffer(23)]], packed_float4 device* engine_contacts_19 [[buffer(25)]], CrushParams_0 constant* crush_19 [[buffer(28)]], packed_float4 device* crush_rw_19 [[buffer(27)]], packed_float4 device* crush_tab_19 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_117;
    (&kernelContext_117)->params_0 = params_19;
    (&kernelContext_117)->atom_0 = atom_19;
    (&kernelContext_117)->work_0 = work_19;
    (&kernelContext_117)->frames_0 = frames_19;
    (&kernelContext_117)->bodies_0 = bodies_19;
    (&kernelContext_117)->chunks_0 = chunks_19;
    (&kernelContext_117)->sort_0 = sort_19;
    (&kernelContext_117)->manifolds_0 = manifolds_19;
    (&kernelContext_117)->loads_0 = loads_19;
    (&kernelContext_117)->bonds_0 = bonds_19;
    (&kernelContext_117)->args_0 = args_19;
    (&kernelContext_117)->info_0 = info_19;
    (&kernelContext_117)->scan_0 = scan_19;
    (&kernelContext_117)->parts_0 = parts_19;
    (&kernelContext_117)->colliders_0 = colliders_19;
    (&kernelContext_117)->collider_info_0 = collider_info_19;
    (&kernelContext_117)->view_chunks_0 = view_chunks_19;
    (&kernelContext_117)->view_0 = view_19;
    (&kernelContext_117)->objects_0 = objects_19;
    (&kernelContext_117)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_117)->lines_0 = lines_19;
    (&kernelContext_117)->geometry_0 = geometry_19;
    (&kernelContext_117)->shots_0 = shots_19;
    (&kernelContext_117)->lp_0 = lp_19;
    (&kernelContext_117)->ext_0 = ext_20;
    (&kernelContext_117)->engine_contacts_0 = engine_contacts_19;
    (&kernelContext_117)->crush_0 = crush_19;
    (&kernelContext_117)->crush_rw_0 = crush_rw_19;
    (&kernelContext_117)->crush_tab_0 = crush_tab_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_117)->g_scan_0 = &g_scan_19;
    threadgroup array<float, int(64)> g_il_best_19;
    (&kernelContext_117)->g_il_best_0 = &g_il_best_19;
    threadgroup array<float, int(256)> g_gi_x_19;
    (&kernelContext_117)->g_gi_x_0 = &g_gi_x_19;
    threadgroup array<float, int(256)> g_gi_m_19;
    (&kernelContext_117)->g_gi_m_0 = &g_gi_m_19;
    threadgroup array<uint, int(1280)> g_gi_lo_19;
    (&kernelContext_117)->g_gi_lo_0 = &g_gi_lo_19;
    threadgroup array<uint, int(1280)> g_gi_hi_19;
    (&kernelContext_117)->g_gi_hi_0 = &g_gi_hi_19;
    uint s_8 = id_16.x;
    bool _S741;
    if(s_8 >= (params_19->body_count_0))
    {
        _S741 = true;
    }
    else
    {
        uint _S742 = w_free_0(&kernelContext_117);
        _S741 = (*((&kernelContext_117)->work_0+(_S742 + s_8))) == 0U;
    }
    if(_S741)
    {
        return;
    }
    uint _S743 = w_free_list_0(&kernelContext_117);
    uint _S744 = w_free_rank_0(&kernelContext_117);
    *((&kernelContext_117)->work_0+(_S743 + *((&kernelContext_117)->work_0+(_S744 + s_8)))) = s_8;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_118)
{
    uint _S745 = w_free_list_0(kernelContext_118);
    return _S745 + kernelContext_118->params_0->body_count_0;
}

uint scan_new_blocks_1(KernelContext_0 thread* kernelContext_119)
{
    uint _S746 = w_free_list_0(kernelContext_119);
    return _S746 + kernelContext_119->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_120)
{
    uint _S747 = scan_new_blocks_1(kernelContext_120);
    return _S747 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_121)
{
    uint blocks_1 = (kernelContext_121->params_0->body_count_0 + 1023U) / 1024U;
    uint _S748 = scan_free_blocks_0(kernelContext_121);
    return *(kernelContext_121->work_0+(_S748 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], atomic<uint> device* atom_20 [[buffer(5)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], packed_float4 device* lp_20 [[buffer(24)]], packed_float4 device* ext_21 [[buffer(23)]], packed_float4 device* engine_contacts_20 [[buffer(25)]], CrushParams_0 constant* crush_20 [[buffer(28)]], packed_float4 device* crush_rw_20 [[buffer(27)]], packed_float4 device* crush_tab_20 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_122;
    (&kernelContext_122)->params_0 = params_20;
    (&kernelContext_122)->atom_0 = atom_20;
    (&kernelContext_122)->work_0 = work_20;
    (&kernelContext_122)->frames_0 = frames_20;
    (&kernelContext_122)->bodies_0 = bodies_20;
    (&kernelContext_122)->chunks_0 = chunks_20;
    (&kernelContext_122)->sort_0 = sort_20;
    (&kernelContext_122)->manifolds_0 = manifolds_20;
    (&kernelContext_122)->loads_0 = loads_20;
    (&kernelContext_122)->bonds_0 = bonds_20;
    (&kernelContext_122)->args_0 = args_20;
    (&kernelContext_122)->info_0 = info_20;
    (&kernelContext_122)->scan_0 = scan_20;
    (&kernelContext_122)->parts_0 = parts_20;
    (&kernelContext_122)->colliders_0 = colliders_20;
    (&kernelContext_122)->collider_info_0 = collider_info_20;
    (&kernelContext_122)->view_chunks_0 = view_chunks_20;
    (&kernelContext_122)->view_0 = view_20;
    (&kernelContext_122)->objects_0 = objects_20;
    (&kernelContext_122)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_122)->lines_0 = lines_20;
    (&kernelContext_122)->geometry_0 = geometry_20;
    (&kernelContext_122)->shots_0 = shots_20;
    (&kernelContext_122)->lp_0 = lp_20;
    (&kernelContext_122)->ext_0 = ext_21;
    (&kernelContext_122)->engine_contacts_0 = engine_contacts_20;
    (&kernelContext_122)->crush_0 = crush_20;
    (&kernelContext_122)->crush_rw_0 = crush_rw_20;
    (&kernelContext_122)->crush_tab_0 = crush_tab_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_122)->g_scan_0 = &g_scan_20;
    threadgroup array<float, int(64)> g_il_best_20;
    (&kernelContext_122)->g_il_best_0 = &g_il_best_20;
    threadgroup array<float, int(256)> g_gi_x_20;
    (&kernelContext_122)->g_gi_x_0 = &g_gi_x_20;
    threadgroup array<float, int(256)> g_gi_m_20;
    (&kernelContext_122)->g_gi_m_0 = &g_gi_m_20;
    threadgroup array<uint, int(1280)> g_gi_lo_20;
    (&kernelContext_122)->g_gi_lo_0 = &g_gi_lo_20;
    threadgroup array<uint, int(1280)> g_gi_hi_20;
    (&kernelContext_122)->g_gi_hi_0 = &g_gi_hi_20;
    uint c_13 = id_17.x;
    bool _S749 = is_root_0(c_13, &kernelContext_122);
    if(!_S749)
    {
        return;
    }
    uint _S750 = body_of_0(c_13, &kernelContext_122);
    uint _S751 = a_best_root_0(&kernelContext_122);
    uint _S752 = atomic_load_explicit((&kernelContext_122)->atom_0+(_S751 + _S750), memory_order_relaxed);
    uint slot_1;
    if(_S752 != c_13)
    {
        uint _S753 = w_rank_0(&kernelContext_122);
        uint device* _S754 = (&kernelContext_122)->work_0+(_S753 + c_13);
        uint k_23 = *_S754;
        uint _S755 = free_total_0(&kernelContext_122);
        if((*_S754) < _S755)
        {
            uint _S756 = w_free_list_0(&kernelContext_122);
            uint device* _S757 = (&kernelContext_122)->work_0+(_S756 + k_23);
            uint slot_2 = *_S757;
            uint _S758 = w_free_0(&kernelContext_122);
            *((&kernelContext_122)->work_0+(_S758 + *_S757)) = 0U;
            uint _S759 = a_counters_1(&kernelContext_122);
            uint _S760 = atomic_fetch_add_explicit((&kernelContext_122)->atom_0+(_S759 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S761 = a_counters_1(&kernelContext_122);
            uint _S762 = atomic_fetch_or_explicit((&kernelContext_122)->atom_0+(_S761 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S750;
        }
    }
    else
    {
        slot_1 = _S750;
    }
    uint _S763 = w_slot_0(&kernelContext_122);
    *((&kernelContext_122)->work_0+(_S763 + c_13)) = slot_1;
    return;
}

float read64_1(uint at_10, float scale_2, KernelContext_0 thread* kernelContext_123)
{
    uint _S764 = atomic_load_explicit(kernelContext_123->atom_0+(at_10 + 1U), memory_order_relaxed);
    int hi_3 = (as_type<int>((_S764)));
    uint lo_3 = atomic_load_explicit(kernelContext_123->atom_0+at_10, memory_order_relaxed);
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
    array<float, int(3)> _S765 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S766 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S767 = { { r2_2.x, r2_2.y, r2_2.z } };
    thread array<array<float, int(3)>, int(3)> a_7;
    a_7[int(0)] = _S765;
    a_7[int(1)] = _S766;
    a_7[int(2)] = _S767;
    array<float, int(3)> _S768 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S769 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S770 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_13;
    v_13[int(0)] = _S768;
    v_13[int(1)] = _S769;
    v_13[int(2)] = _S770;
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
            bool _S771;
            if(k_24 != p_12)
            {
                _S771 = k_24 != q_14;
            }
            else
            {
                _S771 = false;
            }
            if(_S771)
            {
                float kp_0 = a_7[k_24][p_12];
                float kq_0 = a_7[k_24][q_14];
                float _S772 = c_14 * a_7[k_24][p_12] - s_9 * a_7[k_24][q_14];
                a_7[k_24][p_12] = _S772;
                a_7[p_12][k_24] = _S772;
                float _S773 = s_9 * kp_0 + c_14 * kq_0;
                a_7[k_24][q_14] = _S773;
                a_7[q_14][k_24] = _S773;
            }
            k_24 = k_24 + 1U;
        }
        float _S774 = c_14 * c_14;
        float _S775 = 2.0f * s_9 * c_14 * apq_0;
        float _S776 = s_9 * s_9;
        a_7[p_12][p_12] = _S774 * app_0 - _S775 + _S776 * aqq_0;
        a_7[q_14][q_14] = _S776 * app_0 + _S775 + _S774 * aqq_0;
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
            float kp_1 = v_13[row_0][p_12];
            float kq_1 = v_13[row_0][q_14];
            v_13[row_0][p_12] = c_14 * v_13[row_0][p_12] - s_9 * v_13[row_0][q_14];
            v_13[row_0][q_14] = s_9 * kp_1 + c_14 * kq_1;
            row_0 = row_0 + 1U;
        }
        it_0 = it_0 + 1U;
    }
    float3 x_5 = float3(v_13[int(0)][int(0)], v_13[int(1)][int(0)], v_13[int(2)][int(0)]);
    float3 y_1 = float3(v_13[int(0)][int(1)], v_13[int(1)][int(1)], v_13[int(2)][int(1)]);
    float3 z_1 = float3(v_13[int(0)][int(2)], v_13[int(1)][int(2)], v_13[int(2)][int(2)]);
    float3 z_2;
    if((dot(cross(x_5, y_1), z_1)) < 0.0f)
    {
        z_2 = - z_1;
    }
    else
    {
        z_2 = z_1;
    }
    thread Quat_0 _S777 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S778 = normalized_0(&_S777);
    *rot_0 = _S778;
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
    thread Quat_0 _S779 = axes_1;
    Quat_0 _S780 = quat_mul_0(&_S779, q_15);
    thread Quat_0 _S781 = _S780;
    Quat_0 _S782 = normalized_0(&_S781);
    return _S782;
}

uint stage_0(uint c_15, KernelContext_0 thread* kernelContext_124)
{
    return 2U * kernelContext_124->params_0->body_count_0 + 9U * c_15;
}

float4 quat_vec_0(const Quat_0 thread* q_16)
{
    return float4(q_16->x_0, q_16->y_0, q_16->z_0, q_16->w_0);
}

uint frame_step_0(KernelContext_0 thread* kernelContext_125)
{
    uint _S783 = w_frame_0(kernelContext_125);
    return *(kernelContext_125->work_0+(_S783 + 1U));
}

[[kernel]] void fr_piece(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], atomic<uint> device* atom_21 [[buffer(5)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], packed_float4 device* lp_21 [[buffer(24)]], packed_float4 device* ext_22 [[buffer(23)]], packed_float4 device* engine_contacts_21 [[buffer(25)]], CrushParams_0 constant* crush_21 [[buffer(28)]], packed_float4 device* crush_rw_21 [[buffer(27)]], packed_float4 device* crush_tab_21 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_126;
    (&kernelContext_126)->params_0 = params_21;
    (&kernelContext_126)->atom_0 = atom_21;
    (&kernelContext_126)->work_0 = work_21;
    (&kernelContext_126)->frames_0 = frames_21;
    (&kernelContext_126)->bodies_0 = bodies_21;
    (&kernelContext_126)->chunks_0 = chunks_21;
    (&kernelContext_126)->sort_0 = sort_21;
    (&kernelContext_126)->manifolds_0 = manifolds_21;
    (&kernelContext_126)->loads_0 = loads_21;
    (&kernelContext_126)->bonds_0 = bonds_21;
    (&kernelContext_126)->args_0 = args_21;
    (&kernelContext_126)->info_0 = info_21;
    (&kernelContext_126)->scan_0 = scan_21;
    (&kernelContext_126)->parts_0 = parts_21;
    (&kernelContext_126)->colliders_0 = colliders_21;
    (&kernelContext_126)->collider_info_0 = collider_info_21;
    (&kernelContext_126)->view_chunks_0 = view_chunks_21;
    (&kernelContext_126)->view_0 = view_21;
    (&kernelContext_126)->objects_0 = objects_21;
    (&kernelContext_126)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_126)->lines_0 = lines_21;
    (&kernelContext_126)->geometry_0 = geometry_21;
    (&kernelContext_126)->shots_0 = shots_21;
    (&kernelContext_126)->lp_0 = lp_21;
    (&kernelContext_126)->ext_0 = ext_22;
    (&kernelContext_126)->engine_contacts_0 = engine_contacts_21;
    (&kernelContext_126)->crush_0 = crush_21;
    (&kernelContext_126)->crush_rw_0 = crush_rw_21;
    (&kernelContext_126)->crush_tab_0 = crush_tab_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_126)->g_scan_0 = &g_scan_21;
    threadgroup array<float, int(64)> g_il_best_21;
    (&kernelContext_126)->g_il_best_0 = &g_il_best_21;
    threadgroup array<float, int(256)> g_gi_x_21;
    (&kernelContext_126)->g_gi_x_0 = &g_gi_x_21;
    threadgroup array<float, int(256)> g_gi_m_21;
    (&kernelContext_126)->g_gi_m_0 = &g_gi_m_21;
    threadgroup array<uint, int(1280)> g_gi_lo_21;
    (&kernelContext_126)->g_gi_lo_0 = &g_gi_lo_21;
    threadgroup array<uint, int(1280)> g_gi_hi_21;
    (&kernelContext_126)->g_gi_hi_0 = &g_gi_hi_21;
    uint c_16 = id_18.x;
    bool _S784 = is_root_0(c_16, &kernelContext_126);
    if(!_S784)
    {
        return;
    }
    uint _S785 = body_of_0(c_16, &kernelContext_126);
    uint pb_0 = 7U * _S785;
    uint _S786 = a_sums_0(&kernelContext_126);
    uint at_11 = _S786 + 20U * c_16;
    float _S787 = read64_0(at_11, (&kernelContext_126)->params_0->scale_m_0, &kernelContext_126);
    float _S788 = read64_0(at_11 + 2U, (&kernelContext_126)->params_0->scale_s_0, &kernelContext_126);
    float _S789 = read64_0(at_11 + 4U, (&kernelContext_126)->params_0->scale_s_0, &kernelContext_126);
    float _S790 = read64_0(at_11 + 6U, (&kernelContext_126)->params_0->scale_s_0, &kernelContext_126);
    float3 d_20 = float3(_S788, _S789, _S790) / float3(max(_S787, 1.00000000317107685e-30f)) ;
    float _S791 = read64_0(at_11 + 8U, (&kernelContext_126)->params_0->scale_j_0, &kernelContext_126);
    float _S792 = read64_0(at_11 + 10U, (&kernelContext_126)->params_0->scale_j_0, &kernelContext_126);
    float _S793 = read64_0(at_11 + 12U, (&kernelContext_126)->params_0->scale_j_0, &kernelContext_126);
    float _S794 = read64_0(at_11 + 14U, (&kernelContext_126)->params_0->scale_j_0, &kernelContext_126);
    float _S795 = read64_0(at_11 + 16U, (&kernelContext_126)->params_0->scale_j_0, &kernelContext_126);
    float _S796 = read64_0(at_11 + 18U, (&kernelContext_126)->params_0->scale_j_0, &kernelContext_126);
    float dd_1 = dot(d_20, d_20);
    float _S797 = d_20.x;
    float _S798 = _S787 * _S797;
    float _S799 = d_20.y;
    float _S800 = _S794 + _S798 * _S799;
    float _S801 = d_20.z;
    float _S802 = _S795 + _S798 * _S801;
    float _S803 = _S796 + _S787 * _S799 * _S801;
    float3 com_1 = (float4(*((&kernelContext_126)->chunks_0+4U * c_16)) ).xyz + d_20;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S791 - _S787 * (dd_1 - _S797 * _S797), _S800, _S802), float3(_S800, _S792 - _S787 * (dd_1 - _S799 * _S799), _S803), float3(_S802, _S803, _S793 - _S787 * (dd_1 - _S801 * _S801)), &principal_0, &moments_1);
    uint _S804 = a_anchor_0(&kernelContext_126);
    uint _S805 = atomic_load_explicit((&kernelContext_126)->atom_0+(_S804 + c_16), memory_order_relaxed);
    bool anchored_0 = _S805 != 0U;
    thread Quat_0 q_17;
    thread float3 origin_4;
    structure_pose_0(_S785, pb_0, &q_17, &origin_4, &kernelContext_126);
    float3 p_parent_0 = (float4(*((&kernelContext_126)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_126)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_126)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S806 = q_17;
    float3 _S807 = rotate_0(&_S806, com_1);
    float3 pos_0 = to_rubble_0(origin_4 + _S807);
    thread Quat_0 _S808 = q_17;
    thread Quat_0 _S809 = principal_0;
    Quat_0 _S810 = quat_mul_0(&_S808, &_S809);
    thread Quat_0 _S811 = _S810;
    Quat_0 _S812 = normalized_0(&_S811);
    thread Quat_0 _S813 = _S812;
    Quat_0 _S814 = rot_to_rubble_0(&_S813);
    bool _S815;
    if(anchored_0)
    {
        _S815 = true;
    }
    else
    {
        _S815 = !(_S787 > 0.0f);
    }
    float inv_mass_3;
    if(_S815)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / _S787;
    }
    float3 _S816 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S817;
        if((moments_1.x) > 0.0f)
        {
            _S817 = 1.0f / moments_1.x;
        }
        else
        {
            _S817 = 0.0f;
        }
        float _S818;
        if((moments_1.y) > 0.0f)
        {
            _S818 = 1.0f / moments_1.y;
        }
        else
        {
            _S818 = 0.0f;
        }
        float _S819;
        if((moments_1.z) > 0.0f)
        {
            _S819 = 1.0f / moments_1.z;
        }
        else
        {
            _S819 = 0.0f;
        }
        inv_inertia_0 = float3(_S817, _S818, _S819);
    }
    else
    {
        inv_inertia_0 = _S816;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S816;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S816;
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
    uint _S820 = stage_0(c_16, &kernelContext_126);
    *((&kernelContext_126)->frames_0+_S820) = packed_float4(float4(pos_0, inv_mass_3)) ;
    packed_float4 device* _S821 = (&kernelContext_126)->frames_0+(_S820 + 1U);
    thread Quat_0 _S822 = _S814;
    float4 _S823 = quat_vec_0(&_S822);
    *_S821 = packed_float4(_S823) ;
    *((&kernelContext_126)->frames_0+(_S820 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_126)->frames_0+(_S820 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    packed_float4 device* _S824 = (&kernelContext_126)->frames_0+(_S820 + 4U);
    uint _S825 = frame_step_0(&kernelContext_126);
    *_S824 = packed_float4(float4(spin_0, (as_type<float>((_S825))))) ;
    *((&kernelContext_126)->frames_0+(_S820 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_126)->frames_0+(_S820 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_126)->frames_0+(_S820 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_126)->frames_0+2U * _S785)) ).w)) ;
    packed_float4 device* _S826 = (&kernelContext_126)->frames_0+(_S820 + 8U);
    thread Quat_0 _S827 = principal_0;
    float4 _S828 = quat_vec_0(&_S827);
    *_S826 = packed_float4(_S828) ;
    return;
}

uint f_events_0(KernelContext_0 thread* kernelContext_127)
{
    return 2U * kernelContext_127->params_0->body_count_0 + 9U * kernelContext_127->params_0->chunk_count_0 + 3U * kernelContext_127->params_0->body_count_0;
}

[[kernel]] void fr_commit(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], atomic<uint> device* atom_22 [[buffer(5)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], packed_float4 device* lp_22 [[buffer(24)]], packed_float4 device* ext_23 [[buffer(23)]], packed_float4 device* engine_contacts_22 [[buffer(25)]], CrushParams_0 constant* crush_22 [[buffer(28)]], packed_float4 device* crush_rw_22 [[buffer(27)]], packed_float4 device* crush_tab_22 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_128;
    (&kernelContext_128)->params_0 = params_22;
    (&kernelContext_128)->atom_0 = atom_22;
    (&kernelContext_128)->work_0 = work_22;
    (&kernelContext_128)->frames_0 = frames_22;
    (&kernelContext_128)->bodies_0 = bodies_22;
    (&kernelContext_128)->chunks_0 = chunks_22;
    (&kernelContext_128)->sort_0 = sort_22;
    (&kernelContext_128)->manifolds_0 = manifolds_22;
    (&kernelContext_128)->loads_0 = loads_22;
    (&kernelContext_128)->bonds_0 = bonds_22;
    (&kernelContext_128)->args_0 = args_22;
    (&kernelContext_128)->info_0 = info_22;
    (&kernelContext_128)->scan_0 = scan_22;
    (&kernelContext_128)->parts_0 = parts_22;
    (&kernelContext_128)->colliders_0 = colliders_22;
    (&kernelContext_128)->collider_info_0 = collider_info_22;
    (&kernelContext_128)->view_chunks_0 = view_chunks_22;
    (&kernelContext_128)->view_0 = view_22;
    (&kernelContext_128)->objects_0 = objects_22;
    (&kernelContext_128)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_128)->lines_0 = lines_22;
    (&kernelContext_128)->geometry_0 = geometry_22;
    (&kernelContext_128)->shots_0 = shots_22;
    (&kernelContext_128)->lp_0 = lp_22;
    (&kernelContext_128)->ext_0 = ext_23;
    (&kernelContext_128)->engine_contacts_0 = engine_contacts_22;
    (&kernelContext_128)->crush_0 = crush_22;
    (&kernelContext_128)->crush_rw_0 = crush_rw_22;
    (&kernelContext_128)->crush_tab_0 = crush_tab_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_128)->g_scan_0 = &g_scan_22;
    threadgroup array<float, int(64)> g_il_best_22;
    (&kernelContext_128)->g_il_best_0 = &g_il_best_22;
    threadgroup array<float, int(256)> g_gi_x_22;
    (&kernelContext_128)->g_gi_x_0 = &g_gi_x_22;
    threadgroup array<float, int(256)> g_gi_m_22;
    (&kernelContext_128)->g_gi_m_0 = &g_gi_m_22;
    threadgroup array<uint, int(1280)> g_gi_lo_22;
    (&kernelContext_128)->g_gi_lo_0 = &g_gi_lo_22;
    threadgroup array<uint, int(1280)> g_gi_hi_22;
    (&kernelContext_128)->g_gi_hi_0 = &g_gi_hi_22;
    uint c_17 = id_19.x;
    bool _S829 = is_root_0(c_17, &kernelContext_128);
    if(!_S829)
    {
        return;
    }
    uint _S830 = w_slot_0(&kernelContext_128);
    uint slot_3 = *((&kernelContext_128)->work_0+(_S830 + c_17));
    uint _S831 = stage_0(c_17, &kernelContext_128);
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
            *((&kernelContext_128)->bodies_0+(7U * (slot_3 + region_5 * (&kernelContext_128)->params_0->max_bodies_0) + w_4)) = packed_float4(float4(*((&kernelContext_128)->frames_0+(_S831 + w_4))) ) ;
            w_4 = w_4 + 1U;
        }
        region_5 = region_5 + 1U;
    }
    uint _S832 = 2U * slot_3;
    uint _S833 = _S831 + 7U;
    *((&kernelContext_128)->frames_0+_S832) = packed_float4(float4(*((&kernelContext_128)->frames_0+_S833)) ) ;
    uint _S834 = _S831 + 8U;
    *((&kernelContext_128)->frames_0+(_S832 + 1U)) = packed_float4(float4(*((&kernelContext_128)->frames_0+_S834)) ) ;
    if(((&kernelContext_128)->params_0->ev_frames_cap_0) != 0U)
    {
        uint _S835 = a_counters_1(&kernelContext_128);
        uint i_12 = atomic_fetch_add_explicit((&kernelContext_128)->atom_0+(_S835 + 22U), 1U, memory_order_relaxed);
        if(i_12 < ((&kernelContext_128)->params_0->ev_frames_cap_0))
        {
            uint _S836 = f_events_0(&kernelContext_128);
            uint at_12 = _S836 + 3U * i_12;
            *((&kernelContext_128)->frames_0+at_12) = packed_float4(float4((as_type<float>((slot_3))), 0.0f, 0.0f, 0.0f)) ;
            *((&kernelContext_128)->frames_0+(at_12 + 1U)) = packed_float4(float4(*((&kernelContext_128)->frames_0+_S833)) ) ;
            *((&kernelContext_128)->frames_0+(at_12 + 2U)) = packed_float4(float4(*((&kernelContext_128)->frames_0+_S834)) ) ;
        }
        else
        {
            uint _S837 = atomic_fetch_or_explicit((&kernelContext_128)->atom_0+(_S835 + 3U), 128U, memory_order_relaxed);
        }
    }
    return;
}

[[kernel]] void fr_reparent(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], atomic<uint> device* atom_23 [[buffer(5)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], packed_float4 device* lp_23 [[buffer(24)]], packed_float4 device* ext_24 [[buffer(23)]], packed_float4 device* engine_contacts_23 [[buffer(25)]], CrushParams_0 constant* crush_23 [[buffer(28)]], packed_float4 device* crush_rw_23 [[buffer(27)]], packed_float4 device* crush_tab_23 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_129;
    (&kernelContext_129)->params_0 = params_23;
    (&kernelContext_129)->atom_0 = atom_23;
    (&kernelContext_129)->work_0 = work_23;
    (&kernelContext_129)->frames_0 = frames_23;
    (&kernelContext_129)->bodies_0 = bodies_23;
    (&kernelContext_129)->chunks_0 = chunks_23;
    (&kernelContext_129)->sort_0 = sort_23;
    (&kernelContext_129)->manifolds_0 = manifolds_23;
    (&kernelContext_129)->loads_0 = loads_23;
    (&kernelContext_129)->bonds_0 = bonds_23;
    (&kernelContext_129)->args_0 = args_23;
    (&kernelContext_129)->info_0 = info_23;
    (&kernelContext_129)->scan_0 = scan_23;
    (&kernelContext_129)->parts_0 = parts_23;
    (&kernelContext_129)->colliders_0 = colliders_23;
    (&kernelContext_129)->collider_info_0 = collider_info_23;
    (&kernelContext_129)->view_chunks_0 = view_chunks_23;
    (&kernelContext_129)->view_0 = view_23;
    (&kernelContext_129)->objects_0 = objects_23;
    (&kernelContext_129)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_129)->lines_0 = lines_23;
    (&kernelContext_129)->geometry_0 = geometry_23;
    (&kernelContext_129)->shots_0 = shots_23;
    (&kernelContext_129)->lp_0 = lp_23;
    (&kernelContext_129)->ext_0 = ext_24;
    (&kernelContext_129)->engine_contacts_0 = engine_contacts_23;
    (&kernelContext_129)->crush_0 = crush_23;
    (&kernelContext_129)->crush_rw_0 = crush_rw_23;
    (&kernelContext_129)->crush_tab_0 = crush_tab_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_129)->g_scan_0 = &g_scan_23;
    threadgroup array<float, int(64)> g_il_best_23;
    (&kernelContext_129)->g_il_best_0 = &g_il_best_23;
    threadgroup array<float, int(256)> g_gi_x_23;
    (&kernelContext_129)->g_gi_x_0 = &g_gi_x_23;
    threadgroup array<float, int(256)> g_gi_m_23;
    (&kernelContext_129)->g_gi_m_0 = &g_gi_m_23;
    threadgroup array<uint, int(1280)> g_gi_lo_23;
    (&kernelContext_129)->g_gi_lo_0 = &g_gi_lo_23;
    threadgroup array<uint, int(1280)> g_gi_hi_23;
    (&kernelContext_129)->g_gi_hi_0 = &g_gi_hi_23;
    uint c_18 = id_20.x;
    bool _S838 = live_0(c_18, &kernelContext_129);
    bool _S839;
    if(!_S838)
    {
        _S839 = true;
    }
    else
    {
        uint _S840 = body_of_0(c_18, &kernelContext_129);
        bool _S841 = dirty_0(_S840, &kernelContext_129);
        _S839 = !_S841;
    }
    if(_S839)
    {
        return;
    }
    uint _S842 = w_label_0(&kernelContext_129);
    uint device* _S843 = (&kernelContext_129)->work_0+(_S842 + c_18);
    uint r_21 = *_S843;
    uint _S844 = w_slot_0(&kernelContext_129);
    uint device* _S845 = (&kernelContext_129)->work_0+(_S844 + *_S843);
    uint slot_4 = *_S845;
    uint _S846 = body_of_0(c_18, &kernelContext_129);
    if((*_S845) != _S846)
    {
        _S839 = ((&kernelContext_129)->params_0->ev_moved_cap_0) != 0U;
    }
    else
    {
        _S839 = false;
    }
    if(_S839)
    {
        uint _S847 = a_counters_1(&kernelContext_129);
        uint i_13 = atomic_fetch_add_explicit((&kernelContext_129)->atom_0+(_S847 + 21U), 1U, memory_order_relaxed);
        if(i_13 < ((&kernelContext_129)->params_0->ev_moved_cap_0))
        {
            uint _S848 = a_moved_0(&kernelContext_129);
            uint _S849 = _S848 + 2U * i_13;
            atomic_store_explicit((&kernelContext_129)->atom_0+_S849, c_18, memory_order_relaxed);
            atomic_store_explicit((&kernelContext_129)->atom_0+(_S849 + 1U), slot_4, memory_order_relaxed);
        }
        else
        {
            uint _S850 = atomic_fetch_or_explicit((&kernelContext_129)->atom_0+(_S847 + 3U), 128U, memory_order_relaxed);
        }
    }
    *((&kernelContext_129)->work_0+(w_body_0() + c_18)) = slot_4;
    uint _S851 = 2U * slot_4;
    float3 _S852 = (float4(*((&kernelContext_129)->frames_0+_S851)) ).xyz;
    thread Quat_0 _S853 = quat_of_0(float4(*((&kernelContext_129)->frames_0+(_S851 + 1U))) );
    Quat_0 _S854 = conj_0(&_S853);
    float inv_mass_4 = (float4(*((&kernelContext_129)->bodies_0+7U * slot_4)) ).w;
    float _S855;
    if(inv_mass_4 > 0.0f)
    {
        _S855 = 1.0f / inv_mass_4;
    }
    else
    {
        _S855 = float_bits_0(2139095040U);
    }
    float4 _S856 = float4(*((&kernelContext_129)->chunks_0+(4U * c_18 + 2U))) ;
    uint first_1 = (as_type<uint>((_S856.z)));
    uint _S857 = (as_type<uint>((_S856.w)));
    float4 _S858 = float4(*((&kernelContext_129)->chunks_0+(4U * r_21 + 3U))) ;
    uint p_13 = first_1;
    for(;;)
    {
        if(p_13 < (first_1 + _S857))
        {
        }
        else
        {
            break;
        }
        uint _S859 = 2U * p_13;
        float4 _S860 = float4(*((&kernelContext_129)->parts_0+_S859)) ;
        uint col_0 = (as_type<uint>((_S860.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_129)->parts_0+(_S859 + 1U))) );
        uint _S861 = 4U * col_0;
        float4 _S862 = float4(*((&kernelContext_129)->colliders_0+_S861)) ;
        *((&kernelContext_129)->colliders_0+_S861) = packed_float4(float4((as_type<float>((slot_4))), _S862.y, _S862.z, _S862.w)) ;
        uint _S863 = _S861 + 1U;
        packed_float4 device* _S864 = (&kernelContext_129)->colliders_0+_S863;
        float3 _S865 = _S860.xyz - _S852;
        thread Quat_0 _S866 = _S854;
        float3 _S867 = rotate_0(&_S866, _S865);
        *_S864 = packed_float4(float4(_S867, (float4(*((&kernelContext_129)->colliders_0+_S863)) ).w)) ;
        packed_float4 device* _S868 = (&kernelContext_129)->colliders_0+(_S861 + 2U);
        thread Quat_0 _S869 = _S854;
        thread Quat_0 _S870 = rot_1;
        Quat_0 _S871 = quat_mul_0(&_S869, &_S870);
        thread Quat_0 _S872 = _S871;
        Quat_0 _S873 = normalized_0(&_S872);
        thread Quat_0 _S874 = _S873;
        float4 _S875 = quat_vec_0(&_S874);
        *_S868 = packed_float4(_S875) ;
        uint _S876 = 2U * col_0;
        uint4 _S877 = uint4(*((&kernelContext_129)->collider_info_0+_S876)) ;
        *((&kernelContext_129)->collider_info_0+_S876) = packed_uint4(uint4(_S877.x, slot_4, _S877.z, _S877.w)) ;
        *((&kernelContext_129)->collider_info_0+(_S876 + 1U)) = packed_uint4(uint4((as_type<uint>((_S855))), (as_type<uint>((_S858.x))), (as_type<uint>((_S858.w))), 0U)) ;
        p_13 = p_13 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], atomic<uint> device* atom_24 [[buffer(5)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], packed_float4 device* lp_24 [[buffer(24)]], packed_float4 device* ext_25 [[buffer(23)]], packed_float4 device* engine_contacts_24 [[buffer(25)]], CrushParams_0 constant* crush_24 [[buffer(28)]], packed_float4 device* crush_rw_24 [[buffer(27)]], packed_float4 device* crush_tab_24 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_130;
    (&kernelContext_130)->params_0 = params_24;
    (&kernelContext_130)->atom_0 = atom_24;
    (&kernelContext_130)->work_0 = work_24;
    (&kernelContext_130)->frames_0 = frames_24;
    (&kernelContext_130)->bodies_0 = bodies_24;
    (&kernelContext_130)->chunks_0 = chunks_24;
    (&kernelContext_130)->sort_0 = sort_24;
    (&kernelContext_130)->manifolds_0 = manifolds_24;
    (&kernelContext_130)->loads_0 = loads_24;
    (&kernelContext_130)->bonds_0 = bonds_24;
    (&kernelContext_130)->args_0 = args_24;
    (&kernelContext_130)->info_0 = info_24;
    (&kernelContext_130)->scan_0 = scan_24;
    (&kernelContext_130)->parts_0 = parts_24;
    (&kernelContext_130)->colliders_0 = colliders_24;
    (&kernelContext_130)->collider_info_0 = collider_info_24;
    (&kernelContext_130)->view_chunks_0 = view_chunks_24;
    (&kernelContext_130)->view_0 = view_24;
    (&kernelContext_130)->objects_0 = objects_24;
    (&kernelContext_130)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_130)->lines_0 = lines_24;
    (&kernelContext_130)->geometry_0 = geometry_24;
    (&kernelContext_130)->shots_0 = shots_24;
    (&kernelContext_130)->lp_0 = lp_24;
    (&kernelContext_130)->ext_0 = ext_25;
    (&kernelContext_130)->engine_contacts_0 = engine_contacts_24;
    (&kernelContext_130)->crush_0 = crush_24;
    (&kernelContext_130)->crush_rw_0 = crush_rw_24;
    (&kernelContext_130)->crush_tab_0 = crush_tab_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_130)->g_scan_0 = &g_scan_24;
    threadgroup array<float, int(64)> g_il_best_24;
    (&kernelContext_130)->g_il_best_0 = &g_il_best_24;
    threadgroup array<float, int(256)> g_gi_x_24;
    (&kernelContext_130)->g_gi_x_0 = &g_gi_x_24;
    threadgroup array<float, int(256)> g_gi_m_24;
    (&kernelContext_130)->g_gi_m_0 = &g_gi_m_24;
    threadgroup array<uint, int(1280)> g_gi_lo_24;
    (&kernelContext_130)->g_gi_lo_0 = &g_gi_lo_24;
    threadgroup array<uint, int(1280)> g_gi_hi_24;
    (&kernelContext_130)->g_gi_hi_0 = &g_gi_hi_24;
    uint s_10 = id_21.x;
    if(s_10 >= (params_24->body_count_0))
    {
        return;
    }
    uint _S878 = a_dirty_0(&kernelContext_130);
    atomic_store_explicit((&kernelContext_130)->atom_0+(_S878 + s_10), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_5 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_8 = fract(h_5);
    float q_18 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_8);
    float t_9 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_8));
    uint k_25 = uint(h_5);
    float3 _S879;
    if(k_25 == 0U)
    {
        _S879 = float3(0.92000001668930054f, t_9, 0.34959998726844788f);
    }
    else
    {
        if(k_25 == 1U)
        {
            _S879 = float3(q_18, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_25 == 2U)
            {
                _S879 = float3(0.34959998726844788f, 0.92000001668930054f, t_9);
            }
            else
            {
                if(k_25 == 3U)
                {
                    _S879 = float3(0.34959998726844788f, q_18, 0.92000001668930054f);
                }
                else
                {
                    if(k_25 == 4U)
                    {
                        _S879 = float3(t_9, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S879 = float3(0.92000001668930054f, 0.34959998726844788f, q_18);
                    }
                }
            }
        }
    }
    return _S879;
}

void write_object_0(uint o_4, const Quat_0 thread* q_19, float3 t_10, float4 color_0, KernelContext_0 thread* kernelContext_131)
{
    float3 _S880 = rotate_0(q_19, float3(1.0f, 0.0f, 0.0f));
    float3 _S881 = rotate_0(q_19, float3(0.0f, 1.0f, 0.0f));
    float3 _S882 = rotate_0(q_19, float3(0.0f, 0.0f, 1.0f));
    uint _S883 = 4U * o_4;
    *(kernelContext_131->objects_0+_S883) = packed_float4(float4(_S880.x, _S881.x, _S882.x, t_10.x)) ;
    *(kernelContext_131->objects_0+(_S883 + 1U)) = packed_float4(float4(_S880.y, _S881.y, _S882.y, t_10.y)) ;
    *(kernelContext_131->objects_0+(_S883 + 2U)) = packed_float4(float4(_S880.z, _S881.z, _S882.z, t_10.z)) ;
    *(kernelContext_131->objects_0+(_S883 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], atomic<uint> device* atom_25 [[buffer(5)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], packed_float4 device* lp_25 [[buffer(24)]], packed_float4 device* ext_26 [[buffer(23)]], packed_float4 device* engine_contacts_25 [[buffer(25)]], CrushParams_0 constant* crush_25 [[buffer(28)]], packed_float4 device* crush_rw_25 [[buffer(27)]], packed_float4 device* crush_tab_25 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_132;
    (&kernelContext_132)->params_0 = params_25;
    (&kernelContext_132)->atom_0 = atom_25;
    (&kernelContext_132)->work_0 = work_25;
    (&kernelContext_132)->frames_0 = frames_25;
    (&kernelContext_132)->bodies_0 = bodies_25;
    (&kernelContext_132)->chunks_0 = chunks_25;
    (&kernelContext_132)->sort_0 = sort_25;
    (&kernelContext_132)->manifolds_0 = manifolds_25;
    (&kernelContext_132)->loads_0 = loads_25;
    (&kernelContext_132)->bonds_0 = bonds_25;
    (&kernelContext_132)->args_0 = args_25;
    (&kernelContext_132)->info_0 = info_25;
    (&kernelContext_132)->scan_0 = scan_25;
    (&kernelContext_132)->parts_0 = parts_25;
    (&kernelContext_132)->colliders_0 = colliders_25;
    (&kernelContext_132)->collider_info_0 = collider_info_25;
    (&kernelContext_132)->view_chunks_0 = view_chunks_25;
    (&kernelContext_132)->view_0 = view_25;
    (&kernelContext_132)->objects_0 = objects_25;
    (&kernelContext_132)->shot_bodies_0 = shot_bodies_25;
    (&kernelContext_132)->lines_0 = lines_25;
    (&kernelContext_132)->geometry_0 = geometry_25;
    (&kernelContext_132)->shots_0 = shots_25;
    (&kernelContext_132)->lp_0 = lp_25;
    (&kernelContext_132)->ext_0 = ext_26;
    (&kernelContext_132)->engine_contacts_0 = engine_contacts_25;
    (&kernelContext_132)->crush_0 = crush_25;
    (&kernelContext_132)->crush_rw_0 = crush_rw_25;
    (&kernelContext_132)->crush_tab_0 = crush_tab_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_132)->g_scan_0 = &g_scan_25;
    threadgroup array<float, int(64)> g_il_best_25;
    (&kernelContext_132)->g_il_best_0 = &g_il_best_25;
    threadgroup array<float, int(256)> g_gi_x_25;
    (&kernelContext_132)->g_gi_x_0 = &g_gi_x_25;
    threadgroup array<float, int(256)> g_gi_m_25;
    (&kernelContext_132)->g_gi_m_0 = &g_gi_m_25;
    threadgroup array<uint, int(1280)> g_gi_lo_25;
    (&kernelContext_132)->g_gi_lo_0 = &g_gi_lo_25;
    threadgroup array<uint, int(1280)> g_gi_hi_25;
    (&kernelContext_132)->g_gi_hi_0 = &g_gi_hi_25;
    uint k_26 = id_22.x;
    bool _S884 = live_0(k_26, &kernelContext_132);
    if(!_S884)
    {
        return;
    }
    uint _S885 = body_of_0(k_26, &kernelContext_132);
    uint b_37 = 7U * _S885;
    thread Quat_0 q_20;
    thread float3 origin_5;
    structure_pose_0(_S885, b_37, &q_20, &origin_5, &kernelContext_132);
    uint _S886 = 3U * k_26;
    float4 _S887 = float4(*((&kernelContext_132)->view_chunks_0+_S886)) ;
    float3 base_2 = (float4(*((&kernelContext_132)->view_chunks_0+(_S886 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_132)->bodies_0+b_37)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_132)->view_0->mode_0) == 1U)
    {
        float _S888 = max(length((float4(*((&kernelContext_132)->bodies_0+(b_37 + 3U))) ).xyz), length((float4(*((&kernelContext_132)->bodies_0+(b_37 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S888 > ((&kernelContext_132)->view_0->rest_speed_0))
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
        if(((&kernelContext_132)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S885);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S885);
            }
        }
    }
    uint _S889 = (as_type<uint>((_S887.w)));
    Quat_0 _S890 = quat_of_0(float4(*((&kernelContext_132)->view_chunks_0+(_S886 + 1U))) );
    thread Quat_0 _S891 = q_20;
    thread Quat_0 _S892 = _S890;
    Quat_0 _S893 = quat_mul_0(&_S891, &_S892);
    thread Quat_0 _S894 = _S893;
    Quat_0 _S895 = normalized_0(&_S894);
    float3 _S896 = _S887.xyz;
    thread Quat_0 _S897 = q_20;
    float3 _S898 = rotate_0(&_S897, _S896);
    float3 _S899 = origin_5 + _S898;
    float4 _S900 = float4(color_1, 1.0f);
    thread Quat_0 _S901 = _S895;
    write_object_0(_S889, &_S901, _S899, _S900, &kernelContext_132);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], atomic<uint> device* atom_26 [[buffer(5)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], packed_float4 device* lp_26 [[buffer(24)]], packed_float4 device* ext_27 [[buffer(23)]], packed_float4 device* engine_contacts_26 [[buffer(25)]], CrushParams_0 constant* crush_26 [[buffer(28)]], packed_float4 device* crush_rw_26 [[buffer(27)]], packed_float4 device* crush_tab_26 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_133;
    (&kernelContext_133)->params_0 = params_26;
    (&kernelContext_133)->atom_0 = atom_26;
    (&kernelContext_133)->work_0 = work_26;
    (&kernelContext_133)->frames_0 = frames_26;
    (&kernelContext_133)->bodies_0 = bodies_26;
    (&kernelContext_133)->chunks_0 = chunks_26;
    (&kernelContext_133)->sort_0 = sort_26;
    (&kernelContext_133)->manifolds_0 = manifolds_26;
    (&kernelContext_133)->loads_0 = loads_26;
    (&kernelContext_133)->bonds_0 = bonds_26;
    (&kernelContext_133)->args_0 = args_26;
    (&kernelContext_133)->info_0 = info_26;
    (&kernelContext_133)->scan_0 = scan_26;
    (&kernelContext_133)->parts_0 = parts_26;
    (&kernelContext_133)->colliders_0 = colliders_26;
    (&kernelContext_133)->collider_info_0 = collider_info_26;
    (&kernelContext_133)->view_chunks_0 = view_chunks_26;
    (&kernelContext_133)->view_0 = view_26;
    (&kernelContext_133)->objects_0 = objects_26;
    (&kernelContext_133)->shot_bodies_0 = shot_bodies_26;
    (&kernelContext_133)->lines_0 = lines_26;
    (&kernelContext_133)->geometry_0 = geometry_26;
    (&kernelContext_133)->shots_0 = shots_26;
    (&kernelContext_133)->lp_0 = lp_26;
    (&kernelContext_133)->ext_0 = ext_27;
    (&kernelContext_133)->engine_contacts_0 = engine_contacts_26;
    (&kernelContext_133)->crush_0 = crush_26;
    (&kernelContext_133)->crush_rw_0 = crush_rw_26;
    (&kernelContext_133)->crush_tab_0 = crush_tab_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_133)->g_scan_0 = &g_scan_26;
    threadgroup array<float, int(64)> g_il_best_26;
    (&kernelContext_133)->g_il_best_0 = &g_il_best_26;
    threadgroup array<float, int(256)> g_gi_x_26;
    (&kernelContext_133)->g_gi_x_0 = &g_gi_x_26;
    threadgroup array<float, int(256)> g_gi_m_26;
    (&kernelContext_133)->g_gi_m_0 = &g_gi_m_26;
    threadgroup array<uint, int(1280)> g_gi_lo_26;
    (&kernelContext_133)->g_gi_lo_0 = &g_gi_lo_26;
    threadgroup array<uint, int(1280)> g_gi_hi_26;
    (&kernelContext_133)->g_gi_hi_0 = &g_gi_hi_26;
    uint i_14 = id_23.x;
    if(i_14 >= (view_26->shot_count_1))
    {
        return;
    }
    uint b_38 = 7U * (&kernelContext_133)->shot_bodies_0[i_14];
    uint o_5 = (&kernelContext_133)->view_0->impactor_base_0 + i_14;
    if(((float4(*((&kernelContext_133)->bodies_0+b_38)) ).w) == 0.0f)
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
            *((&kernelContext_133)->objects_0+(4U * o_5 + w_5)) = packed_float4(float4(0.0f) ) ;
            w_5 = w_5 + 1U;
        }
        return;
    }
    thread Quat_0 _S902 = quat_of_0(float4(*((&kernelContext_133)->bodies_0+(b_38 + 1U))) );
    Quat_0 _S903 = rot_from_rubble_0(&_S902);
    float3 _S904 = from_rubble_0((float4(*((&kernelContext_133)->bodies_0+b_38)) ).xyz);
    float4 _S905 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S906 = _S903;
    write_object_0(o_5, &_S906, _S904, _S905, &kernelContext_133);
    return;
}

float3 chunk_world_0(uint k_27, KernelContext_0 thread* kernelContext_134)
{
    uint _S907 = body_of_0(k_27, kernelContext_134);
    thread Quat_0 q_21;
    thread float3 origin_6;
    structure_pose_0(_S907, 7U * _S907, &q_21, &origin_6, kernelContext_134);
    float3 _S908 = (float4(*(kernelContext_134->view_chunks_0+3U * k_27)) ).xyz;
    thread Quat_0 _S909 = q_21;
    float3 _S910 = rotate_0(&_S909, _S908);
    return origin_6 + _S910;
}

void write_line_vertex_0(uint v_14, float3 p_14, KernelContext_0 thread* kernelContext_135)
{
    uint _S911 = 7U * v_14;
    *(kernelContext_135->lines_0+_S911) = p_14.x;
    *(kernelContext_135->lines_0+(_S911 + 1U)) = p_14.y;
    *(kernelContext_135->lines_0+(_S911 + 2U)) = p_14.z;
    *(kernelContext_135->lines_0+(_S911 + 3U)) = 1.0f;
    *(kernelContext_135->lines_0+(_S911 + 4U)) = 0.86000001430511475f;
    *(kernelContext_135->lines_0+(_S911 + 5U)) = 0.25f;
    *(kernelContext_135->lines_0+(_S911 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], atomic<uint> device* atom_27 [[buffer(5)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], packed_float4 device* lp_27 [[buffer(24)]], packed_float4 device* ext_28 [[buffer(23)]], packed_float4 device* engine_contacts_27 [[buffer(25)]], CrushParams_0 constant* crush_27 [[buffer(28)]], packed_float4 device* crush_rw_27 [[buffer(27)]], packed_float4 device* crush_tab_27 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_136;
    (&kernelContext_136)->params_0 = params_27;
    (&kernelContext_136)->atom_0 = atom_27;
    (&kernelContext_136)->work_0 = work_27;
    (&kernelContext_136)->frames_0 = frames_27;
    (&kernelContext_136)->bodies_0 = bodies_27;
    (&kernelContext_136)->chunks_0 = chunks_27;
    (&kernelContext_136)->sort_0 = sort_27;
    (&kernelContext_136)->manifolds_0 = manifolds_27;
    (&kernelContext_136)->loads_0 = loads_27;
    (&kernelContext_136)->bonds_0 = bonds_27;
    (&kernelContext_136)->args_0 = args_27;
    (&kernelContext_136)->info_0 = info_27;
    (&kernelContext_136)->scan_0 = scan_27;
    (&kernelContext_136)->parts_0 = parts_27;
    (&kernelContext_136)->colliders_0 = colliders_27;
    (&kernelContext_136)->collider_info_0 = collider_info_27;
    (&kernelContext_136)->view_chunks_0 = view_chunks_27;
    (&kernelContext_136)->view_0 = view_27;
    (&kernelContext_136)->objects_0 = objects_27;
    (&kernelContext_136)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_136)->lines_0 = lines_27;
    (&kernelContext_136)->geometry_0 = geometry_27;
    (&kernelContext_136)->shots_0 = shots_27;
    (&kernelContext_136)->lp_0 = lp_27;
    (&kernelContext_136)->ext_0 = ext_28;
    (&kernelContext_136)->engine_contacts_0 = engine_contacts_27;
    (&kernelContext_136)->crush_0 = crush_27;
    (&kernelContext_136)->crush_rw_0 = crush_rw_27;
    (&kernelContext_136)->crush_tab_0 = crush_tab_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_136)->g_scan_0 = &g_scan_27;
    threadgroup array<float, int(64)> g_il_best_27;
    (&kernelContext_136)->g_il_best_0 = &g_il_best_27;
    threadgroup array<float, int(256)> g_gi_x_27;
    (&kernelContext_136)->g_gi_x_0 = &g_gi_x_27;
    threadgroup array<float, int(256)> g_gi_m_27;
    (&kernelContext_136)->g_gi_m_0 = &g_gi_m_27;
    threadgroup array<uint, int(1280)> g_gi_lo_27;
    (&kernelContext_136)->g_gi_lo_0 = &g_gi_lo_27;
    threadgroup array<uint, int(1280)> g_gi_hi_27;
    (&kernelContext_136)->g_gi_hi_0 = &g_gi_hi_27;
    uint i_15 = id_24.x;
    if(i_15 >= (params_27->bond_used_0))
    {
        return;
    }
    uint4 _S912 = uint4(*((&kernelContext_136)->bonds_0+i_15)) ;
    float3 _S913 = chunk_world_0(_S912.x, &kernelContext_136);
    uint _S914 = w_alive_0(&kernelContext_136);
    float3 b_39;
    if((*((&kernelContext_136)->work_0+(_S914 + i_15))) != 0U)
    {
        float3 _S915 = chunk_world_0(_S912.y, &kernelContext_136);
        b_39 = _S915;
    }
    else
    {
        b_39 = _S913;
    }
    uint _S916 = 2U * i_15;
    write_line_vertex_0(_S916, _S913, &kernelContext_136);
    write_line_vertex_0(_S916 + 1U, b_39, &kernelContext_136);
    return;
}

[[kernel]] void fr_strike_reset(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], atomic<uint> device* atom_28 [[buffer(5)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], packed_float4 device* lp_28 [[buffer(24)]], packed_float4 device* ext_29 [[buffer(23)]], packed_float4 device* engine_contacts_28 [[buffer(25)]], CrushParams_0 constant* crush_28 [[buffer(28)]], packed_float4 device* crush_rw_28 [[buffer(27)]], packed_float4 device* crush_tab_28 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_137;
    (&kernelContext_137)->params_0 = params_28;
    (&kernelContext_137)->atom_0 = atom_28;
    (&kernelContext_137)->work_0 = work_28;
    (&kernelContext_137)->frames_0 = frames_28;
    (&kernelContext_137)->bodies_0 = bodies_28;
    (&kernelContext_137)->chunks_0 = chunks_28;
    (&kernelContext_137)->sort_0 = sort_28;
    (&kernelContext_137)->manifolds_0 = manifolds_28;
    (&kernelContext_137)->loads_0 = loads_28;
    (&kernelContext_137)->bonds_0 = bonds_28;
    (&kernelContext_137)->args_0 = args_28;
    (&kernelContext_137)->info_0 = info_28;
    (&kernelContext_137)->scan_0 = scan_28;
    (&kernelContext_137)->parts_0 = parts_28;
    (&kernelContext_137)->colliders_0 = colliders_28;
    (&kernelContext_137)->collider_info_0 = collider_info_28;
    (&kernelContext_137)->view_chunks_0 = view_chunks_28;
    (&kernelContext_137)->view_0 = view_28;
    (&kernelContext_137)->objects_0 = objects_28;
    (&kernelContext_137)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_137)->lines_0 = lines_28;
    (&kernelContext_137)->geometry_0 = geometry_28;
    (&kernelContext_137)->shots_0 = shots_28;
    (&kernelContext_137)->lp_0 = lp_28;
    (&kernelContext_137)->ext_0 = ext_29;
    (&kernelContext_137)->engine_contacts_0 = engine_contacts_28;
    (&kernelContext_137)->crush_0 = crush_28;
    (&kernelContext_137)->crush_rw_0 = crush_rw_28;
    (&kernelContext_137)->crush_tab_0 = crush_tab_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_137)->g_scan_0 = &g_scan_28;
    threadgroup array<float, int(64)> g_il_best_28;
    (&kernelContext_137)->g_il_best_0 = &g_il_best_28;
    threadgroup array<float, int(256)> g_gi_x_28;
    (&kernelContext_137)->g_gi_x_0 = &g_gi_x_28;
    threadgroup array<float, int(256)> g_gi_m_28;
    (&kernelContext_137)->g_gi_m_0 = &g_gi_m_28;
    threadgroup array<uint, int(1280)> g_gi_lo_28;
    (&kernelContext_137)->g_gi_lo_0 = &g_gi_lo_28;
    threadgroup array<uint, int(1280)> g_gi_hi_28;
    (&kernelContext_137)->g_gi_hi_0 = &g_gi_hi_28;
    uint b_40 = id_25.x;
    if(b_40 >= (params_28->body_count_0))
    {
        return;
    }
    uint _S917 = a_strike_0(&kernelContext_137);
    uint at_13 = _S917 + 9U * b_40;
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
        atomic_store_explicit((&kernelContext_137)->atom_0+(at_13 + w_6), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_137)->atom_0+(at_13 + 3U + w_6), 0U, memory_order_relaxed);
        w_6 = w_6 + 1U;
    }
    atomic_store_explicit((&kernelContext_137)->atom_0+(at_13 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_137)->atom_0+(at_13 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_137)->atom_0+(at_13 + 8U), 4294967295U, memory_order_relaxed);
    uint _S918 = strike_at_0(b_40, &kernelContext_137);
    (*((&kernelContext_137)->frames_0+_S918))[int(3)] = 0.0f;
    return;
}

uint ordered_0(float f_9)
{
    uint u_7 = (as_type<uint>((f_9)));
    uint _S919;
    if((u_7 & 2147483648U) != 0U)
    {
        _S919 = ~u_7;
    }
    else
    {
        _S919 = u_7 | 2147483648U;
    }
    return _S919;
}

[[kernel]] void fr_strike_bounds(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], atomic<uint> device* atom_29 [[buffer(5)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], packed_float4 device* lp_29 [[buffer(24)]], packed_float4 device* ext_30 [[buffer(23)]], packed_float4 device* engine_contacts_29 [[buffer(25)]], CrushParams_0 constant* crush_29 [[buffer(28)]], packed_float4 device* crush_rw_29 [[buffer(27)]], packed_float4 device* crush_tab_29 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_138;
    (&kernelContext_138)->params_0 = params_29;
    (&kernelContext_138)->atom_0 = atom_29;
    (&kernelContext_138)->work_0 = work_29;
    (&kernelContext_138)->frames_0 = frames_29;
    (&kernelContext_138)->bodies_0 = bodies_29;
    (&kernelContext_138)->chunks_0 = chunks_29;
    (&kernelContext_138)->sort_0 = sort_29;
    (&kernelContext_138)->manifolds_0 = manifolds_29;
    (&kernelContext_138)->loads_0 = loads_29;
    (&kernelContext_138)->bonds_0 = bonds_29;
    (&kernelContext_138)->args_0 = args_29;
    (&kernelContext_138)->info_0 = info_29;
    (&kernelContext_138)->scan_0 = scan_29;
    (&kernelContext_138)->parts_0 = parts_29;
    (&kernelContext_138)->colliders_0 = colliders_29;
    (&kernelContext_138)->collider_info_0 = collider_info_29;
    (&kernelContext_138)->view_chunks_0 = view_chunks_29;
    (&kernelContext_138)->view_0 = view_29;
    (&kernelContext_138)->objects_0 = objects_29;
    (&kernelContext_138)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_138)->lines_0 = lines_29;
    (&kernelContext_138)->geometry_0 = geometry_29;
    (&kernelContext_138)->shots_0 = shots_29;
    (&kernelContext_138)->lp_0 = lp_29;
    (&kernelContext_138)->ext_0 = ext_30;
    (&kernelContext_138)->engine_contacts_0 = engine_contacts_29;
    (&kernelContext_138)->crush_0 = crush_29;
    (&kernelContext_138)->crush_rw_0 = crush_rw_29;
    (&kernelContext_138)->crush_tab_0 = crush_tab_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_138)->g_scan_0 = &g_scan_29;
    threadgroup array<float, int(64)> g_il_best_29;
    (&kernelContext_138)->g_il_best_0 = &g_il_best_29;
    threadgroup array<float, int(256)> g_gi_x_29;
    (&kernelContext_138)->g_gi_x_0 = &g_gi_x_29;
    threadgroup array<float, int(256)> g_gi_m_29;
    (&kernelContext_138)->g_gi_m_0 = &g_gi_m_29;
    threadgroup array<uint, int(1280)> g_gi_lo_29;
    (&kernelContext_138)->g_gi_lo_0 = &g_gi_lo_29;
    threadgroup array<uint, int(1280)> g_gi_hi_29;
    (&kernelContext_138)->g_gi_hi_0 = &g_gi_hi_29;
    uint k_28 = id_26.x;
    bool _S920 = live_0(k_28, &kernelContext_138);
    bool _S921;
    if(!_S920)
    {
        _S921 = true;
    }
    else
    {
        _S921 = ((&kernelContext_138)->params_0->shot_count_0) == 0U;
    }
    if(_S921)
    {
        return;
    }
    thread Quat_0 q_22;
    float3 _S922 = chunk_box_world_0(k_28, &q_22, &kernelContext_138);
    float _S923 = (float4(*((&kernelContext_138)->geometry_0+3U * k_28)) ).w;
    uint _S924 = a_strike_0(&kernelContext_138);
    uint _S925 = body_of_0(k_28, &kernelContext_138);
    uint _S926 = _S924 + 9U * _S925;
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
        uint _S927 = ax_1;
        uint _S928 = atomic_fetch_min_explicit((&kernelContext_138)->atom_0+(_S926 + ax_1), ordered_0(_S922[ax_1] - _S923), memory_order_relaxed);
        uint _S929 = atomic_fetch_max_explicit((&kernelContext_138)->atom_0+(_S926 + 3U + ax_1), ordered_0(_S922[_S927] + _S923), memory_order_relaxed);
        ax_1 = ax_1 + 1U;
    }
    return;
}

float unordered_0(uint u_8)
{
    uint _S930;
    if((u_8 & 2147483648U) != 0U)
    {
        _S930 = u_8 & 2147483647U;
    }
    else
    {
        _S930 = ~u_8;
    }
    return (as_type<float>((_S930)));
}

[[kernel]] void fr_strike_find(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], atomic<uint> device* atom_30 [[buffer(5)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], packed_float4 device* lp_30 [[buffer(24)]], packed_float4 device* ext_31 [[buffer(23)]], packed_float4 device* engine_contacts_30 [[buffer(25)]], CrushParams_0 constant* crush_30 [[buffer(28)]], packed_float4 device* crush_rw_30 [[buffer(27)]], packed_float4 device* crush_tab_30 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_139;
    (&kernelContext_139)->params_0 = params_30;
    (&kernelContext_139)->atom_0 = atom_30;
    (&kernelContext_139)->work_0 = work_30;
    (&kernelContext_139)->frames_0 = frames_30;
    (&kernelContext_139)->bodies_0 = bodies_30;
    (&kernelContext_139)->chunks_0 = chunks_30;
    (&kernelContext_139)->sort_0 = sort_30;
    (&kernelContext_139)->manifolds_0 = manifolds_30;
    (&kernelContext_139)->loads_0 = loads_30;
    (&kernelContext_139)->bonds_0 = bonds_30;
    (&kernelContext_139)->args_0 = args_30;
    (&kernelContext_139)->info_0 = info_30;
    (&kernelContext_139)->scan_0 = scan_30;
    (&kernelContext_139)->parts_0 = parts_30;
    (&kernelContext_139)->colliders_0 = colliders_30;
    (&kernelContext_139)->collider_info_0 = collider_info_30;
    (&kernelContext_139)->view_chunks_0 = view_chunks_30;
    (&kernelContext_139)->view_0 = view_30;
    (&kernelContext_139)->objects_0 = objects_30;
    (&kernelContext_139)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_139)->lines_0 = lines_30;
    (&kernelContext_139)->geometry_0 = geometry_30;
    (&kernelContext_139)->shots_0 = shots_30;
    (&kernelContext_139)->lp_0 = lp_30;
    (&kernelContext_139)->ext_0 = ext_31;
    (&kernelContext_139)->engine_contacts_0 = engine_contacts_30;
    (&kernelContext_139)->crush_0 = crush_30;
    (&kernelContext_139)->crush_rw_0 = crush_rw_30;
    (&kernelContext_139)->crush_tab_0 = crush_tab_30;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_139)->g_scan_0 = &g_scan_30;
    threadgroup array<float, int(64)> g_il_best_30;
    (&kernelContext_139)->g_il_best_0 = &g_il_best_30;
    threadgroup array<float, int(256)> g_gi_x_30;
    (&kernelContext_139)->g_gi_x_0 = &g_gi_x_30;
    threadgroup array<float, int(256)> g_gi_m_30;
    (&kernelContext_139)->g_gi_m_0 = &g_gi_m_30;
    threadgroup array<uint, int(1280)> g_gi_lo_30;
    (&kernelContext_139)->g_gi_lo_0 = &g_gi_lo_30;
    threadgroup array<uint, int(1280)> g_gi_hi_30;
    (&kernelContext_139)->g_gi_hi_0 = &g_gi_hi_30;
    uint b_41 = id_27.x;
    bool _S931;
    if(b_41 >= (params_30->body_count_0))
    {
        _S931 = true;
    }
    else
    {
        _S931 = ((&kernelContext_139)->params_0->shot_count_0) == 0U;
    }
    if(_S931)
    {
        return;
    }
    uint _S932 = a_strike_0(&kernelContext_139);
    uint at_14 = _S932 + 9U * b_41;
    uint _S933 = atomic_load_explicit((&kernelContext_139)->atom_0+at_14, memory_order_relaxed);
    if(_S933 == 4294967295U)
    {
        return;
    }
    uint _S934 = atomic_load_explicit((&kernelContext_139)->atom_0+at_14, memory_order_relaxed);
    float _S935 = unordered_0(_S934);
    uint _S936 = atomic_load_explicit((&kernelContext_139)->atom_0+(at_14 + 1U), memory_order_relaxed);
    float _S937 = unordered_0(_S936);
    uint _S938 = atomic_load_explicit((&kernelContext_139)->atom_0+(at_14 + 2U), memory_order_relaxed);
    float3 _S939 = float3(_S935, _S937, unordered_0(_S938));
    uint _S940 = atomic_load_explicit((&kernelContext_139)->atom_0+(at_14 + 3U), memory_order_relaxed);
    float _S941 = unordered_0(_S940);
    uint _S942 = atomic_load_explicit((&kernelContext_139)->atom_0+(at_14 + 4U), memory_order_relaxed);
    float _S943 = unordered_0(_S942);
    uint _S944 = atomic_load_explicit((&kernelContext_139)->atom_0+(at_14 + 5U), memory_order_relaxed);
    float3 _S945 = float3(_S941, _S943, unordered_0(_S944));
    float3 _S946 = from_rubble_0((float4(*((&kernelContext_139)->bodies_0+(7U * b_41 + 3U))) ).xyz);
    float _S947 = float_bits_0(2139095040U);
    uint k_29 = 0U;
    for(;;)
    {
        if(k_29 < ((&kernelContext_139)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S948 = float4(*((&kernelContext_139)->shots_0+2U * k_29)) ;
        uint sb_1 = 7U * (as_type<uint>((_S948.x)));
        if(((float4(*((&kernelContext_139)->bodies_0+sb_1)) ).w) == 0.0f)
        {
            k_29 = k_29 + 1U;
            continue;
        }
        float3 c_19 = from_rubble_0((float4(*((&kernelContext_139)->bodies_0+sb_1)) ).xyz);
        float3 rel_2 = from_rubble_0((float4(*((&kernelContext_139)->bodies_0+(sb_1 + 3U))) ).xyz) - _S946;
        if(!((length(rel_2)) > ((&kernelContext_139)->params_0->strike_speed_0)))
        {
            k_29 = k_29 + 1U;
            continue;
        }
        float3 _S949 = float3(_S948.y) ;
        float3 _S950 = _S939 - _S949;
        float3 _S951 = _S945 + _S949;
        float _S952 = (&kernelContext_139)->params_0->dt_0;
        float t0_1 = 0.0f;
        float t1_1 = _S952;
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
            uint _S953 = ax_2;
            if((abs(rel_2[ax_2])) < 9.999999960041972e-13f)
            {
                uint _S954 = ax_2;
                if((c_19[ax_2]) < (_S950[ax_2]))
                {
                    _S931 = true;
                }
                else
                {
                    _S931 = (c_19[_S954]) > (_S951[ax_2]);
                }
                float t0_2;
                if(_S931)
                {
                    t0_2 = _S947;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + 1U;
                continue;
            }
            float ta_1 = (_S950[ax_2] - c_19[ax_2]) / rel_2[_S953];
            float tb_1 = (_S951[ax_2] - c_19[ax_2]) / rel_2[_S953];
            float _S955 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S955;
            ax_2 = ax_2 + 1U;
        }
        if(t0_1 <= t1_1)
        {
            uint _S956 = strike_at_0(b_41, &kernelContext_139);
            *((&kernelContext_139)->frames_0+_S956) = packed_float4(float4(c_19, float(k_29 + 1U))) ;
            *((&kernelContext_139)->frames_0+(_S956 + 1U)) = packed_float4(float4(rel_2, t0_1)) ;
            return;
        }
        k_29 = k_29 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_entry(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], atomic<uint> device* atom_31 [[buffer(5)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_31 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], packed_float4 device* lp_31 [[buffer(24)]], packed_float4 device* ext_32 [[buffer(23)]], packed_float4 device* engine_contacts_31 [[buffer(25)]], CrushParams_0 constant* crush_31 [[buffer(28)]], packed_float4 device* crush_rw_31 [[buffer(27)]], packed_float4 device* crush_tab_31 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_140;
    (&kernelContext_140)->params_0 = params_31;
    (&kernelContext_140)->atom_0 = atom_31;
    (&kernelContext_140)->work_0 = work_31;
    (&kernelContext_140)->frames_0 = frames_31;
    (&kernelContext_140)->bodies_0 = bodies_31;
    (&kernelContext_140)->chunks_0 = chunks_31;
    (&kernelContext_140)->sort_0 = sort_31;
    (&kernelContext_140)->manifolds_0 = manifolds_31;
    (&kernelContext_140)->loads_0 = loads_31;
    (&kernelContext_140)->bonds_0 = bonds_31;
    (&kernelContext_140)->args_0 = args_31;
    (&kernelContext_140)->info_0 = info_31;
    (&kernelContext_140)->scan_0 = scan_31;
    (&kernelContext_140)->parts_0 = parts_31;
    (&kernelContext_140)->colliders_0 = colliders_31;
    (&kernelContext_140)->collider_info_0 = collider_info_31;
    (&kernelContext_140)->view_chunks_0 = view_chunks_31;
    (&kernelContext_140)->view_0 = view_31;
    (&kernelContext_140)->objects_0 = objects_31;
    (&kernelContext_140)->shot_bodies_0 = shot_bodies_31;
    (&kernelContext_140)->lines_0 = lines_31;
    (&kernelContext_140)->geometry_0 = geometry_31;
    (&kernelContext_140)->shots_0 = shots_31;
    (&kernelContext_140)->lp_0 = lp_31;
    (&kernelContext_140)->ext_0 = ext_32;
    (&kernelContext_140)->engine_contacts_0 = engine_contacts_31;
    (&kernelContext_140)->crush_0 = crush_31;
    (&kernelContext_140)->crush_rw_0 = crush_rw_31;
    (&kernelContext_140)->crush_tab_0 = crush_tab_31;
    threadgroup array<uint, int(1024)> g_scan_31;
    (&kernelContext_140)->g_scan_0 = &g_scan_31;
    threadgroup array<float, int(64)> g_il_best_31;
    (&kernelContext_140)->g_il_best_0 = &g_il_best_31;
    threadgroup array<float, int(256)> g_gi_x_31;
    (&kernelContext_140)->g_gi_x_0 = &g_gi_x_31;
    threadgroup array<float, int(256)> g_gi_m_31;
    (&kernelContext_140)->g_gi_m_0 = &g_gi_m_31;
    threadgroup array<uint, int(1280)> g_gi_lo_31;
    (&kernelContext_140)->g_gi_lo_0 = &g_gi_lo_31;
    threadgroup array<uint, int(1280)> g_gi_hi_31;
    (&kernelContext_140)->g_gi_hi_0 = &g_gi_hi_31;
    uint k_30 = id_28.x;
    bool _S957 = live_0(k_30, &kernelContext_140);
    bool _S958;
    if(!_S957)
    {
        _S958 = true;
    }
    else
    {
        _S958 = ((&kernelContext_140)->params_0->shot_count_0) == 0U;
    }
    if(_S958)
    {
        return;
    }
    uint _S959 = body_of_0(k_30, &kernelContext_140);
    thread uint shot_5;
    bool _S960 = struck_0(_S959, &shot_5, &kernelContext_140);
    if(!_S960)
    {
        return;
    }
    uint _S961 = strike_at_0(_S959, &kernelContext_140);
    float4 _S962 = float4(*((&kernelContext_140)->frames_0+(_S961 + 1U))) ;
    float3 from_0 = (float4(*((&kernelContext_140)->frames_0+_S961)) ).xyz;
    float3 rel_3 = _S962.xyz;
    float3 d_21 = rel_3 / float3(max(length(rel_3), 9.999999960041972e-13f)) ;
    float _S963 = (float4(*((&kernelContext_140)->shots_0+2U * shot_5)) ).y;
    thread Quat_0 q_23;
    float3 _S964 = chunk_box_world_0(k_30, &q_23, &kernelContext_140);
    uint _S965 = 3U * k_30;
    Quat_0 _S966 = quat_of_0(float4(*((&kernelContext_140)->geometry_0+(_S965 + 2U))) );
    thread Quat_0 _S967 = q_23;
    thread Quat_0 _S968 = _S966;
    Quat_0 _S969 = quat_mul_0(&_S967, &_S968);
    thread Quat_0 _S970 = _S969;
    Quat_0 _S971 = normalized_0(&_S970);
    float3 _S972 = (float4(*((&kernelContext_140)->geometry_0+(_S965 + 1U))) ).xyz;
    float _S973 = float_bits_0(2139095040U);
    float _S974 = (&kernelContext_140)->params_0->dt_0;
    float t_in_5 = 0.0f;
    float t_out_0 = _S974;
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
        thread Quat_0 _S975 = _S971;
        float3 _S976 = rotate_0(&_S975, e_3);
        float o_6 = dot(from_0 - _S964, _S976);
        float dv_1 = dot(rel_3, _S976);
        float ext_33 = _S972[ax_3] + _S963;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            float t_in_6;
            if((abs(o_6)) > ext_33)
            {
                t_in_6 = _S973;
            }
            else
            {
                t_in_6 = t_in_5;
            }
            t_in_5 = t_in_6;
            ax_3 = ax_3 + 1U;
            continue;
        }
        float ta_2 = (- ext_33 - o_6) / dv_1;
        float tb_2 = (ext_33 - o_6) / dv_1;
        float _S977 = min(t_out_0, max(ta_2, tb_2));
        t_in_5 = max(t_in_5, min(ta_2, tb_2));
        t_out_0 = _S977;
        ax_3 = ax_3 + 1U;
    }
    uint _S978 = a_strike_0(&kernelContext_140);
    uint at_15 = _S978 + 9U * _S959;
    if(t_in_5 <= t_out_0)
    {
        uint _S979 = atomic_fetch_min_explicit((&kernelContext_140)->atom_0+(at_15 + 6U), (as_type<uint>((t_in_5))), memory_order_relaxed);
    }
    float3 r_22 = _S964 - (from_0 + rel_3 * float3(_S962.w) );
    float along_1 = dot(r_22, d_21);
    float dist_0 = length(r_22 - d_21 * float3(along_1) ) + abs(along_1) * 0.00100000004749745f;
    uint _S980 = a_strike_dist_0(&kernelContext_140);
    uint _S981 = (as_type<uint>((dist_0)));
    atomic_store_explicit((&kernelContext_140)->atom_0+(_S980 + k_30), _S981, memory_order_relaxed);
    uint _S982 = atomic_fetch_min_explicit((&kernelContext_140)->atom_0+(at_15 + 7U), _S981, memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], atomic<uint> device* atom_32 [[buffer(5)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_32 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], packed_float4 device* lp_32 [[buffer(24)]], packed_float4 device* ext_34 [[buffer(23)]], packed_float4 device* engine_contacts_32 [[buffer(25)]], CrushParams_0 constant* crush_32 [[buffer(28)]], packed_float4 device* crush_rw_32 [[buffer(27)]], packed_float4 device* crush_tab_32 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_141;
    (&kernelContext_141)->params_0 = params_32;
    (&kernelContext_141)->atom_0 = atom_32;
    (&kernelContext_141)->work_0 = work_32;
    (&kernelContext_141)->frames_0 = frames_32;
    (&kernelContext_141)->bodies_0 = bodies_32;
    (&kernelContext_141)->chunks_0 = chunks_32;
    (&kernelContext_141)->sort_0 = sort_32;
    (&kernelContext_141)->manifolds_0 = manifolds_32;
    (&kernelContext_141)->loads_0 = loads_32;
    (&kernelContext_141)->bonds_0 = bonds_32;
    (&kernelContext_141)->args_0 = args_32;
    (&kernelContext_141)->info_0 = info_32;
    (&kernelContext_141)->scan_0 = scan_32;
    (&kernelContext_141)->parts_0 = parts_32;
    (&kernelContext_141)->colliders_0 = colliders_32;
    (&kernelContext_141)->collider_info_0 = collider_info_32;
    (&kernelContext_141)->view_chunks_0 = view_chunks_32;
    (&kernelContext_141)->view_0 = view_32;
    (&kernelContext_141)->objects_0 = objects_32;
    (&kernelContext_141)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_141)->lines_0 = lines_32;
    (&kernelContext_141)->geometry_0 = geometry_32;
    (&kernelContext_141)->shots_0 = shots_32;
    (&kernelContext_141)->lp_0 = lp_32;
    (&kernelContext_141)->ext_0 = ext_34;
    (&kernelContext_141)->engine_contacts_0 = engine_contacts_32;
    (&kernelContext_141)->crush_0 = crush_32;
    (&kernelContext_141)->crush_rw_0 = crush_rw_32;
    (&kernelContext_141)->crush_tab_0 = crush_tab_32;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_141)->g_scan_0 = &g_scan_32;
    threadgroup array<float, int(64)> g_il_best_32;
    (&kernelContext_141)->g_il_best_0 = &g_il_best_32;
    threadgroup array<float, int(256)> g_gi_x_32;
    (&kernelContext_141)->g_gi_x_0 = &g_gi_x_32;
    threadgroup array<float, int(256)> g_gi_m_32;
    (&kernelContext_141)->g_gi_m_0 = &g_gi_m_32;
    threadgroup array<uint, int(1280)> g_gi_lo_32;
    (&kernelContext_141)->g_gi_lo_0 = &g_gi_lo_32;
    threadgroup array<uint, int(1280)> g_gi_hi_32;
    (&kernelContext_141)->g_gi_hi_0 = &g_gi_hi_32;
    uint k_31 = id_29.x;
    bool _S983 = live_0(k_31, &kernelContext_141);
    bool _S984;
    if(!_S983)
    {
        _S984 = true;
    }
    else
    {
        _S984 = ((&kernelContext_141)->params_0->shot_count_0) == 0U;
    }
    if(_S984)
    {
        return;
    }
    uint _S985 = body_of_0(k_31, &kernelContext_141);
    thread uint shot_6;
    bool _S986 = struck_0(_S985, &shot_6, &kernelContext_141);
    if(!_S986)
    {
        return;
    }
    uint _S987 = a_strike_0(&kernelContext_141);
    uint at_16 = _S987 + 9U * _S985;
    uint _S988 = a_strike_dist_0(&kernelContext_141);
    uint _S989 = atomic_load_explicit((&kernelContext_141)->atom_0+(_S988 + k_31), memory_order_relaxed);
    uint _S990 = atomic_load_explicit((&kernelContext_141)->atom_0+(at_16 + 7U), memory_order_relaxed);
    if(_S989 == _S990)
    {
        uint _S991 = atomic_fetch_min_explicit((&kernelContext_141)->atom_0+(at_16 + 8U), k_31, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], atomic<uint> device* atom_33 [[buffer(5)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], packed_float4 device* lp_33 [[buffer(24)]], packed_float4 device* ext_35 [[buffer(23)]], packed_float4 device* engine_contacts_33 [[buffer(25)]], CrushParams_0 constant* crush_33 [[buffer(28)]], packed_float4 device* crush_rw_33 [[buffer(27)]], packed_float4 device* crush_tab_33 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_142;
    (&kernelContext_142)->params_0 = params_33;
    (&kernelContext_142)->atom_0 = atom_33;
    (&kernelContext_142)->work_0 = work_33;
    (&kernelContext_142)->frames_0 = frames_33;
    (&kernelContext_142)->bodies_0 = bodies_33;
    (&kernelContext_142)->chunks_0 = chunks_33;
    (&kernelContext_142)->sort_0 = sort_33;
    (&kernelContext_142)->manifolds_0 = manifolds_33;
    (&kernelContext_142)->loads_0 = loads_33;
    (&kernelContext_142)->bonds_0 = bonds_33;
    (&kernelContext_142)->args_0 = args_33;
    (&kernelContext_142)->info_0 = info_33;
    (&kernelContext_142)->scan_0 = scan_33;
    (&kernelContext_142)->parts_0 = parts_33;
    (&kernelContext_142)->colliders_0 = colliders_33;
    (&kernelContext_142)->collider_info_0 = collider_info_33;
    (&kernelContext_142)->view_chunks_0 = view_chunks_33;
    (&kernelContext_142)->view_0 = view_33;
    (&kernelContext_142)->objects_0 = objects_33;
    (&kernelContext_142)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_142)->lines_0 = lines_33;
    (&kernelContext_142)->geometry_0 = geometry_33;
    (&kernelContext_142)->shots_0 = shots_33;
    (&kernelContext_142)->lp_0 = lp_33;
    (&kernelContext_142)->ext_0 = ext_35;
    (&kernelContext_142)->engine_contacts_0 = engine_contacts_33;
    (&kernelContext_142)->crush_0 = crush_33;
    (&kernelContext_142)->crush_rw_0 = crush_rw_33;
    (&kernelContext_142)->crush_tab_0 = crush_tab_33;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_142)->g_scan_0 = &g_scan_33;
    threadgroup array<float, int(64)> g_il_best_33;
    (&kernelContext_142)->g_il_best_0 = &g_il_best_33;
    threadgroup array<float, int(256)> g_gi_x_33;
    (&kernelContext_142)->g_gi_x_0 = &g_gi_x_33;
    threadgroup array<float, int(256)> g_gi_m_33;
    (&kernelContext_142)->g_gi_m_0 = &g_gi_m_33;
    threadgroup array<uint, int(1280)> g_gi_lo_33;
    (&kernelContext_142)->g_gi_lo_0 = &g_gi_lo_33;
    threadgroup array<uint, int(1280)> g_gi_hi_33;
    (&kernelContext_142)->g_gi_hi_0 = &g_gi_hi_33;
    uint b_42 = id_30.x;
    bool _S992;
    if(b_42 >= (params_33->body_count_0))
    {
        _S992 = true;
    }
    else
    {
        _S992 = ((&kernelContext_142)->params_0->shot_count_0) == 0U;
    }
    if(_S992)
    {
        return;
    }
    thread uint shot_7;
    bool _S993 = struck_0(b_42, &shot_7, &kernelContext_142);
    if(!_S993)
    {
        return;
    }
    uint _S994 = a_strike_0(&kernelContext_142);
    uint at_17 = _S994 + 9U * b_42;
    uint _S995 = atomic_load_explicit((&kernelContext_142)->atom_0+(at_17 + 6U), memory_order_relaxed);
    float t_in_7 = (as_type<float>((_S995)));
    uint entry_3 = atomic_load_explicit((&kernelContext_142)->atom_0+(at_17 + 8U), memory_order_relaxed);
    uint _S996 = strike_at_0(b_42, &kernelContext_142);
    if(!(t_in_7 <= ((&kernelContext_142)->params_0->dt_0)))
    {
        _S992 = true;
    }
    else
    {
        _S992 = entry_3 == 4294967295U;
    }
    if(_S992)
    {
        (*((&kernelContext_142)->frames_0+_S996))[int(3)] = 0.0f;
        return;
    }
    float3 rel_4 = (float4(*((&kernelContext_142)->frames_0+(_S996 + 1U))) ).xyz;
    float speed_14 = length(rel_4);
    float4 _S997 = float4(*((&kernelContext_142)->shots_0+2U * shot_7)) ;
    float _S998 = max(_S997.z, 0.00100000004749745f);
    float m_imp_0 = _S997.w;
    float e_imp_0 = (float4(*((&kernelContext_142)->shots_0+(2U * shot_7 + 1U))) ).x;
    float3 contact_17 = (float4(*((&kernelContext_142)->frames_0+_S996)) ).xyz + rel_4 * float3(t_in_7)  + rel_4 / float3(max(speed_14, 9.999999960041972e-13f))  * float3(_S997.y) ;
    float inv_mass_5 = (float4(*((&kernelContext_142)->bodies_0+7U * b_42)) ).w;
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
    float modulus_1 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_142)->geometry_0+(3U * entry_3 + 1U))) ).w);
    *((&kernelContext_142)->frames_0+(_S996 + 2U)) = packed_float4(float4(contact_17, m_eff_6 * speed_14 / (2.86999988555908203f * pow(m_eff_6 * m_eff_6 / (_S998 * modulus_1 * modulus_1 * max(speed_14, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

bool strike_share_breaks_0(uint4 bond_1, float3 contact_18, float force_2, KernelContext_0 thread* kernelContext_143)
{
    thread Quat_0 q_24;
    float3 _S999 = chunk_box_world_0(bond_1.x, &q_24, kernelContext_143);
    float3 _S1000 = chunk_box_world_0(bond_1.y, &q_24, kernelContext_143);
    float dist_1 = length(float3(0.5f)  * (_S999 + _S1000) - contact_18);
    return (force_2 * min(1.0f, (as_type<float>((bond_1.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((bond_1.z)));
}

[[kernel]] void fr_strike_breaks(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], atomic<uint> device* atom_34 [[buffer(5)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], packed_float4 device* lp_34 [[buffer(24)]], packed_float4 device* ext_36 [[buffer(23)]], packed_float4 device* engine_contacts_34 [[buffer(25)]], CrushParams_0 constant* crush_34 [[buffer(28)]], packed_float4 device* crush_rw_34 [[buffer(27)]], packed_float4 device* crush_tab_34 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_144;
    (&kernelContext_144)->params_0 = params_34;
    (&kernelContext_144)->atom_0 = atom_34;
    (&kernelContext_144)->work_0 = work_34;
    (&kernelContext_144)->frames_0 = frames_34;
    (&kernelContext_144)->bodies_0 = bodies_34;
    (&kernelContext_144)->chunks_0 = chunks_34;
    (&kernelContext_144)->sort_0 = sort_34;
    (&kernelContext_144)->manifolds_0 = manifolds_34;
    (&kernelContext_144)->loads_0 = loads_34;
    (&kernelContext_144)->bonds_0 = bonds_34;
    (&kernelContext_144)->args_0 = args_34;
    (&kernelContext_144)->info_0 = info_34;
    (&kernelContext_144)->scan_0 = scan_34;
    (&kernelContext_144)->parts_0 = parts_34;
    (&kernelContext_144)->colliders_0 = colliders_34;
    (&kernelContext_144)->collider_info_0 = collider_info_34;
    (&kernelContext_144)->view_chunks_0 = view_chunks_34;
    (&kernelContext_144)->view_0 = view_34;
    (&kernelContext_144)->objects_0 = objects_34;
    (&kernelContext_144)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_144)->lines_0 = lines_34;
    (&kernelContext_144)->geometry_0 = geometry_34;
    (&kernelContext_144)->shots_0 = shots_34;
    (&kernelContext_144)->lp_0 = lp_34;
    (&kernelContext_144)->ext_0 = ext_36;
    (&kernelContext_144)->engine_contacts_0 = engine_contacts_34;
    (&kernelContext_144)->crush_0 = crush_34;
    (&kernelContext_144)->crush_rw_0 = crush_rw_34;
    (&kernelContext_144)->crush_tab_0 = crush_tab_34;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_144)->g_scan_0 = &g_scan_34;
    threadgroup array<float, int(64)> g_il_best_34;
    (&kernelContext_144)->g_il_best_0 = &g_il_best_34;
    threadgroup array<float, int(256)> g_gi_x_34;
    (&kernelContext_144)->g_gi_x_0 = &g_gi_x_34;
    threadgroup array<float, int(256)> g_gi_m_34;
    (&kernelContext_144)->g_gi_m_0 = &g_gi_m_34;
    threadgroup array<uint, int(1280)> g_gi_lo_34;
    (&kernelContext_144)->g_gi_lo_0 = &g_gi_lo_34;
    threadgroup array<uint, int(1280)> g_gi_hi_34;
    (&kernelContext_144)->g_gi_hi_0 = &g_gi_hi_34;
    uint i_16 = id_31.x;
    bool _S1001;
    if(i_16 >= (params_34->bond_used_0))
    {
        _S1001 = true;
    }
    else
    {
        _S1001 = ((&kernelContext_144)->params_0->shot_count_0) == 0U;
    }
    if(_S1001)
    {
        _S1001 = true;
    }
    else
    {
        uint _S1002 = w_alive_0(&kernelContext_144);
        _S1001 = (*((&kernelContext_144)->work_0+(_S1002 + i_16))) == 0U;
    }
    if(_S1001)
    {
        return;
    }
    uint4 _S1003 = uint4(*((&kernelContext_144)->bonds_0+i_16)) ;
    uint _S1004 = body_of_0(_S1003.x, &kernelContext_144);
    thread uint shot_8;
    bool _S1005 = struck_0(_S1004, &shot_8, &kernelContext_144);
    if(!_S1005)
    {
        return;
    }
    uint _S1006 = strike_at_0(_S1004, &kernelContext_144);
    float4 _S1007 = float4(*((&kernelContext_144)->frames_0+(_S1006 + 2U))) ;
    bool _S1008 = strike_share_breaks_0(_S1003, _S1007.xyz, _S1007.w, &kernelContext_144);
    if(_S1008)
    {
        uint _S1009 = w_alive_0(&kernelContext_144);
        *((&kernelContext_144)->work_0+(_S1009 + i_16)) = 0U;
        uint _S1010 = a_dirty_0(&kernelContext_144);
        uint _S1011 = atomic_fetch_or_explicit((&kernelContext_144)->atom_0+(_S1010 + _S1004), 1U, memory_order_relaxed);
        uint _S1012 = a_counters_1(&kernelContext_144);
        uint _S1013 = atomic_fetch_add_explicit((&kernelContext_144)->atom_0+_S1012, 1U, memory_order_relaxed);
        uint _S1014 = atomic_fetch_add_explicit((&kernelContext_144)->atom_0+(_S1012 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_16, &kernelContext_144);
    }
    return;
}

[[kernel]] void fr_strike_radial(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], atomic<uint> device* atom_35 [[buffer(5)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], packed_float4 device* lp_35 [[buffer(24)]], packed_float4 device* ext_37 [[buffer(23)]], packed_float4 device* engine_contacts_35 [[buffer(25)]], CrushParams_0 constant* crush_35 [[buffer(28)]], packed_float4 device* crush_rw_35 [[buffer(27)]], packed_float4 device* crush_tab_35 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_145;
    (&kernelContext_145)->params_0 = params_35;
    (&kernelContext_145)->atom_0 = atom_35;
    (&kernelContext_145)->work_0 = work_35;
    (&kernelContext_145)->frames_0 = frames_35;
    (&kernelContext_145)->bodies_0 = bodies_35;
    (&kernelContext_145)->chunks_0 = chunks_35;
    (&kernelContext_145)->sort_0 = sort_35;
    (&kernelContext_145)->manifolds_0 = manifolds_35;
    (&kernelContext_145)->loads_0 = loads_35;
    (&kernelContext_145)->bonds_0 = bonds_35;
    (&kernelContext_145)->args_0 = args_35;
    (&kernelContext_145)->info_0 = info_35;
    (&kernelContext_145)->scan_0 = scan_35;
    (&kernelContext_145)->parts_0 = parts_35;
    (&kernelContext_145)->colliders_0 = colliders_35;
    (&kernelContext_145)->collider_info_0 = collider_info_35;
    (&kernelContext_145)->view_chunks_0 = view_chunks_35;
    (&kernelContext_145)->view_0 = view_35;
    (&kernelContext_145)->objects_0 = objects_35;
    (&kernelContext_145)->shot_bodies_0 = shot_bodies_35;
    (&kernelContext_145)->lines_0 = lines_35;
    (&kernelContext_145)->geometry_0 = geometry_35;
    (&kernelContext_145)->shots_0 = shots_35;
    (&kernelContext_145)->lp_0 = lp_35;
    (&kernelContext_145)->ext_0 = ext_37;
    (&kernelContext_145)->engine_contacts_0 = engine_contacts_35;
    (&kernelContext_145)->crush_0 = crush_35;
    (&kernelContext_145)->crush_rw_0 = crush_rw_35;
    (&kernelContext_145)->crush_tab_0 = crush_tab_35;
    threadgroup array<uint, int(1024)> g_scan_35;
    (&kernelContext_145)->g_scan_0 = &g_scan_35;
    threadgroup array<float, int(64)> g_il_best_35;
    (&kernelContext_145)->g_il_best_0 = &g_il_best_35;
    threadgroup array<float, int(256)> g_gi_x_35;
    (&kernelContext_145)->g_gi_x_0 = &g_gi_x_35;
    threadgroup array<float, int(256)> g_gi_m_35;
    (&kernelContext_145)->g_gi_m_0 = &g_gi_m_35;
    threadgroup array<uint, int(1280)> g_gi_lo_35;
    (&kernelContext_145)->g_gi_lo_0 = &g_gi_lo_35;
    threadgroup array<uint, int(1280)> g_gi_hi_35;
    (&kernelContext_145)->g_gi_hi_0 = &g_gi_hi_35;
    uint i_17 = id_32.x;
    bool _S1015;
    if(i_17 >= (params_35->bond_used_0))
    {
        _S1015 = true;
    }
    else
    {
        _S1015 = ((&kernelContext_145)->params_0->shot_count_0) == 0U;
    }
    if(_S1015)
    {
        _S1015 = true;
    }
    else
    {
        uint _S1016 = w_alive_0(&kernelContext_145);
        _S1015 = (*((&kernelContext_145)->work_0+(_S1016 + i_17))) == 0U;
    }
    if(_S1015)
    {
        return;
    }
    uint4 _S1017 = uint4(*((&kernelContext_145)->bonds_0+i_17)) ;
    uint _S1018 = _S1017.x;
    uint _S1019 = body_of_0(_S1018, &kernelContext_145);
    thread uint shot_9;
    bool _S1020 = struck_0(_S1019, &shot_9, &kernelContext_145);
    if(!_S1020)
    {
        return;
    }
    uint _S1021 = strike_at_0(_S1019, &kernelContext_145);
    float3 contact_19 = (float4(*((&kernelContext_145)->frames_0+(_S1021 + 2U))) ).xyz;
    thread Quat_0 q_25;
    float3 _S1022 = chunk_box_world_0(_S1018, &q_25, &kernelContext_145);
    float3 _S1023 = chunk_box_world_0(_S1017.y, &q_25, &kernelContext_145);
    if((length(float3(0.5f)  * (_S1022 + _S1023) - contact_19)) <= ((&kernelContext_145)->params_0->strike_radius_0))
    {
        uint _S1024 = w_alive_0(&kernelContext_145);
        *((&kernelContext_145)->work_0+(_S1024 + i_17)) = 0U;
        uint _S1025 = a_dirty_0(&kernelContext_145);
        uint _S1026 = atomic_fetch_or_explicit((&kernelContext_145)->atom_0+(_S1025 + _S1019), 1U, memory_order_relaxed);
        uint _S1027 = a_counters_1(&kernelContext_145);
        uint _S1028 = atomic_fetch_add_explicit((&kernelContext_145)->atom_0+_S1027, 1U, memory_order_relaxed);
        uint _S1029 = atomic_fetch_add_explicit((&kernelContext_145)->atom_0+(_S1027 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_17, &kernelContext_145);
    }
    return;
}

uint a_lp_0(uint b_43, KernelContext_0 thread* kernelContext_146)
{
    uint _S1030 = a_strike_0(kernelContext_146);
    return _S1030 + 9U * kernelContext_146->params_0->body_count_0 + 14U * b_43;
}

[[kernel]] void fr_lp_clear(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], atomic<uint> device* atom_36 [[buffer(5)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], packed_float4 device* lp_36 [[buffer(24)]], packed_float4 device* ext_38 [[buffer(23)]], packed_float4 device* engine_contacts_36 [[buffer(25)]], CrushParams_0 constant* crush_36 [[buffer(28)]], packed_float4 device* crush_rw_36 [[buffer(27)]], packed_float4 device* crush_tab_36 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_147;
    (&kernelContext_147)->params_0 = params_36;
    (&kernelContext_147)->atom_0 = atom_36;
    (&kernelContext_147)->work_0 = work_36;
    (&kernelContext_147)->frames_0 = frames_36;
    (&kernelContext_147)->bodies_0 = bodies_36;
    (&kernelContext_147)->chunks_0 = chunks_36;
    (&kernelContext_147)->sort_0 = sort_36;
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
    threadgroup array<float, int(256)> g_gi_x_36;
    (&kernelContext_147)->g_gi_x_0 = &g_gi_x_36;
    threadgroup array<float, int(256)> g_gi_m_36;
    (&kernelContext_147)->g_gi_m_0 = &g_gi_m_36;
    threadgroup array<uint, int(1280)> g_gi_lo_36;
    (&kernelContext_147)->g_gi_lo_0 = &g_gi_lo_36;
    threadgroup array<uint, int(1280)> g_gi_hi_36;
    (&kernelContext_147)->g_gi_hi_0 = &g_gi_hi_36;
    uint b_44 = id_33.x;
    if(b_44 >= (params_36->body_count_0))
    {
        return;
    }
    uint _S1031 = a_lp_0(b_44, &kernelContext_147);
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
        atomic_store_explicit((&kernelContext_147)->atom_0+(_S1031 + w_7), 0U, memory_order_relaxed);
        w_7 = w_7 + 1U;
    }
    atomic_store_explicit((&kernelContext_147)->atom_0+(_S1031 + 12U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_147)->atom_0+(_S1031 + 13U), 0U, memory_order_relaxed);
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

void add64w_0(uint at_18, float value_3, float scale_3, KernelContext_0 thread* kernelContext_148)
{
    float x_6 = round(value_3 * scale_3);
    float x_7;
    if(!((abs(x_6)) < 3.99999993722699776e+18f))
    {
        uint _S1032 = a_counters_1(kernelContext_148);
        uint _S1033 = atomic_fetch_or_explicit(kernelContext_148->atom_0+(_S1032 + 3U), 2U, memory_order_relaxed);
        x_7 = 0.0f;
    }
    else
    {
        x_7 = x_6;
    }
    float u_9 = abs(x_7);
    float h_6 = floor(u_9 * 2.32830643653869629e-10f);
    uint lo_4 = uint(u_9 - h_6 * 4.294967296e+09f);
    uint hi_4 = uint(h_6);
    uint lo_5;
    uint hi_5;
    if(x_7 < 0.0f)
    {
        uint lo_6 = ~lo_4 + 1U;
        uint _S1034 = ~hi_4;
        if(lo_6 == 0U)
        {
            lo_5 = 1U;
        }
        else
        {
            lo_5 = 0U;
        }
        uint _S1035 = _S1034 + lo_5;
        lo_5 = lo_6;
        hi_5 = _S1035;
    }
    else
    {
        lo_5 = lo_4;
        hi_5 = hi_4;
    }
    uint old_1 = atomic_fetch_add_explicit(kernelContext_148->atom_0+at_18, lo_5, memory_order_relaxed);
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
        uint _S1036 = atomic_fetch_add_explicit(kernelContext_148->atom_0+(at_18 + 1U), hi_6, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_lp_sum(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], atomic<uint> device* atom_37 [[buffer(5)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], packed_float4 device* lp_37 [[buffer(24)]], packed_float4 device* ext_39 [[buffer(23)]], packed_float4 device* engine_contacts_37 [[buffer(25)]], CrushParams_0 constant* crush_37 [[buffer(28)]], packed_float4 device* crush_rw_37 [[buffer(27)]], packed_float4 device* crush_tab_37 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_149;
    (&kernelContext_149)->params_0 = params_37;
    (&kernelContext_149)->atom_0 = atom_37;
    (&kernelContext_149)->work_0 = work_37;
    (&kernelContext_149)->frames_0 = frames_37;
    (&kernelContext_149)->bodies_0 = bodies_37;
    (&kernelContext_149)->chunks_0 = chunks_37;
    (&kernelContext_149)->sort_0 = sort_37;
    (&kernelContext_149)->manifolds_0 = manifolds_37;
    (&kernelContext_149)->loads_0 = loads_37;
    (&kernelContext_149)->bonds_0 = bonds_37;
    (&kernelContext_149)->args_0 = args_37;
    (&kernelContext_149)->info_0 = info_37;
    (&kernelContext_149)->scan_0 = scan_37;
    (&kernelContext_149)->parts_0 = parts_37;
    (&kernelContext_149)->colliders_0 = colliders_37;
    (&kernelContext_149)->collider_info_0 = collider_info_37;
    (&kernelContext_149)->view_chunks_0 = view_chunks_37;
    (&kernelContext_149)->view_0 = view_37;
    (&kernelContext_149)->objects_0 = objects_37;
    (&kernelContext_149)->shot_bodies_0 = shot_bodies_37;
    (&kernelContext_149)->lines_0 = lines_37;
    (&kernelContext_149)->geometry_0 = geometry_37;
    (&kernelContext_149)->shots_0 = shots_37;
    (&kernelContext_149)->lp_0 = lp_37;
    (&kernelContext_149)->ext_0 = ext_39;
    (&kernelContext_149)->engine_contacts_0 = engine_contacts_37;
    (&kernelContext_149)->crush_0 = crush_37;
    (&kernelContext_149)->crush_rw_0 = crush_rw_37;
    (&kernelContext_149)->crush_tab_0 = crush_tab_37;
    threadgroup array<uint, int(1024)> g_scan_37;
    (&kernelContext_149)->g_scan_0 = &g_scan_37;
    threadgroup array<float, int(64)> g_il_best_37;
    (&kernelContext_149)->g_il_best_0 = &g_il_best_37;
    threadgroup array<float, int(256)> g_gi_x_37;
    (&kernelContext_149)->g_gi_x_0 = &g_gi_x_37;
    threadgroup array<float, int(256)> g_gi_m_37;
    (&kernelContext_149)->g_gi_m_0 = &g_gi_m_37;
    threadgroup array<uint, int(1280)> g_gi_lo_37;
    (&kernelContext_149)->g_gi_lo_0 = &g_gi_lo_37;
    threadgroup array<uint, int(1280)> g_gi_hi_37;
    (&kernelContext_149)->g_gi_hi_0 = &g_gi_hi_37;
    uint k_34 = id_34.x;
    if(k_34 >= (params_37->chunk_count_0))
    {
        return;
    }
    uint _S1037 = body_of_0(k_34, &kernelContext_149);
    thread Quat_0 q_26;
    thread float3 origin_7;
    structure_pose_0(_S1037, 7U * _S1037, &q_26, &origin_7, &kernelContext_149);
    thread Quat_0 _S1038 = q_26;
    Quat_0 _S1039 = conj_0(&_S1038);
    float4 _S1040 = float4(*((&kernelContext_149)->chunks_0+4U * k_34)) ;
    uint _S1041 = l_vec_0(&kernelContext_149);
    uint _S1042 = _S1041 + 2U * k_34;
    float3 _S1043 = (float4(*((&kernelContext_149)->loads_0+_S1042)) ).xyz;
    thread Quat_0 _S1044 = _S1039;
    float3 _S1045 = rotate_0(&_S1044, _S1043);
    float _S1046 = _S1040.w;
    float3 _S1047 = float3((&kernelContext_149)->params_0->gx_0, (&kernelContext_149)->params_0->gy_0, (&kernelContext_149)->params_0->gz_0);
    thread Quat_0 _S1048 = _S1039;
    float3 _S1049 = rotate_0(&_S1048, _S1047);
    float3 f_10 = _S1045 + float3(_S1046)  * _S1049;
    float3 _S1050 = (float4(*((&kernelContext_149)->loads_0+(_S1042 + 1U))) ).xyz;
    thread Quat_0 _S1051 = _S1039;
    float3 _S1052 = rotate_0(&_S1051, _S1050);
    uint _S1053 = lp_f_0(k_34);
    *((&kernelContext_149)->lp_0+_S1053) = packed_float4(float4(f_10, 0.0f)) ;
    *((&kernelContext_149)->lp_0+(_S1053 + 1U)) = packed_float4(float4(_S1052, 0.0f)) ;
    float3 _S1054 = _S1052 + cross(_S1040.xyz - (float4(*((&kernelContext_149)->frames_0+2U * _S1037)) ).xyz, f_10);
    uint _S1055 = a_lp_0(_S1037, &kernelContext_149);
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
        uint _S1056 = 2U * a_8;
        add64w_0(_S1055 + _S1056, f_10[a_8], (&kernelContext_149)->params_0->lp_scale_0, &kernelContext_149);
        add64w_0(_S1055 + 6U + _S1056, _S1054[a_8], (&kernelContext_149)->params_0->lp_scale_0, &kernelContext_149);
        a_8 = a_8 + 1U;
    }
    uint _S1057 = atomic_fetch_min_explicit((&kernelContext_149)->atom_0+(_S1055 + 12U), k_34, memory_order_relaxed);
    if(((&kernelContext_149)->info_0[k_34] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_149)->atom_0+(_S1055 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

float3 read64w3_0(uint at_19, float scale_4, KernelContext_0 thread* kernelContext_150)
{
    float _S1058 = read64_1(at_19, scale_4, kernelContext_150);
    float _S1059 = read64_1(at_19 + 2U, scale_4, kernelContext_150);
    float _S1060 = read64_1(at_19 + 4U, scale_4, kernelContext_150);
    return float3(_S1058, _S1059, _S1060);
}

[[kernel]] void fr_lp_relief(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], atomic<uint> device* atom_38 [[buffer(5)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], packed_float4 device* lp_38 [[buffer(24)]], packed_float4 device* ext_40 [[buffer(23)]], packed_float4 device* engine_contacts_38 [[buffer(25)]], CrushParams_0 constant* crush_38 [[buffer(28)]], packed_float4 device* crush_rw_38 [[buffer(27)]], packed_float4 device* crush_tab_38 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_151;
    (&kernelContext_151)->params_0 = params_38;
    (&kernelContext_151)->atom_0 = atom_38;
    (&kernelContext_151)->work_0 = work_38;
    (&kernelContext_151)->frames_0 = frames_38;
    (&kernelContext_151)->bodies_0 = bodies_38;
    (&kernelContext_151)->chunks_0 = chunks_38;
    (&kernelContext_151)->sort_0 = sort_38;
    (&kernelContext_151)->manifolds_0 = manifolds_38;
    (&kernelContext_151)->loads_0 = loads_38;
    (&kernelContext_151)->bonds_0 = bonds_38;
    (&kernelContext_151)->args_0 = args_38;
    (&kernelContext_151)->info_0 = info_38;
    (&kernelContext_151)->scan_0 = scan_38;
    (&kernelContext_151)->parts_0 = parts_38;
    (&kernelContext_151)->colliders_0 = colliders_38;
    (&kernelContext_151)->collider_info_0 = collider_info_38;
    (&kernelContext_151)->view_chunks_0 = view_chunks_38;
    (&kernelContext_151)->view_0 = view_38;
    (&kernelContext_151)->objects_0 = objects_38;
    (&kernelContext_151)->shot_bodies_0 = shot_bodies_38;
    (&kernelContext_151)->lines_0 = lines_38;
    (&kernelContext_151)->geometry_0 = geometry_38;
    (&kernelContext_151)->shots_0 = shots_38;
    (&kernelContext_151)->lp_0 = lp_38;
    (&kernelContext_151)->ext_0 = ext_40;
    (&kernelContext_151)->engine_contacts_0 = engine_contacts_38;
    (&kernelContext_151)->crush_0 = crush_38;
    (&kernelContext_151)->crush_rw_0 = crush_rw_38;
    (&kernelContext_151)->crush_tab_0 = crush_tab_38;
    threadgroup array<uint, int(1024)> g_scan_38;
    (&kernelContext_151)->g_scan_0 = &g_scan_38;
    threadgroup array<float, int(64)> g_il_best_38;
    (&kernelContext_151)->g_il_best_0 = &g_il_best_38;
    threadgroup array<float, int(256)> g_gi_x_38;
    (&kernelContext_151)->g_gi_x_0 = &g_gi_x_38;
    threadgroup array<float, int(256)> g_gi_m_38;
    (&kernelContext_151)->g_gi_m_0 = &g_gi_m_38;
    threadgroup array<uint, int(1280)> g_gi_lo_38;
    (&kernelContext_151)->g_gi_lo_0 = &g_gi_lo_38;
    threadgroup array<uint, int(1280)> g_gi_hi_38;
    (&kernelContext_151)->g_gi_hi_0 = &g_gi_hi_38;
    uint k_35 = id_35.x;
    if(k_35 >= (params_38->chunk_count_0))
    {
        return;
    }
    uint _S1061 = body_of_0(k_35, &kernelContext_151);
    uint _S1062 = a_lp_0(_S1061, &kernelContext_151);
    uint _S1063 = atomic_load_explicit((&kernelContext_151)->atom_0+(_S1062 + 13U), memory_order_relaxed);
    bool supported_0 = _S1063 != 0U;
    bool fixed_1;
    if(((&kernelContext_151)->info_0[k_35] & 1U) != 0U)
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            uint _S1064 = atomic_load_explicit((&kernelContext_151)->atom_0+(_S1062 + 12U), memory_order_relaxed);
            fixed_1 = _S1064 == k_35;
        }
        else
        {
            fixed_1 = false;
        }
    }
    uint _S1065 = lp_f_0(k_35);
    thread float4 f_11 = float4(*((&kernelContext_151)->lp_0+_S1065)) ;
    uint _S1066 = _S1065 + 1U;
    thread float4 t_11 = float4(*((&kernelContext_151)->lp_0+_S1066)) ;
    uint b_45 = 7U * _S1061;
    float inv_mass_6 = (float4(*((&kernelContext_151)->bodies_0+b_45)) ).w;
    bool _S1067;
    if(!supported_0)
    {
        _S1067 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S1067 = false;
    }
    if(_S1067)
    {
        float3 _S1068 = read64w3_0(_S1062, (&kernelContext_151)->params_0->lp_scale_0, &kernelContext_151);
        float3 _S1069 = read64w3_0(_S1062 + 6U, (&kernelContext_151)->params_0->lp_scale_0, &kernelContext_151);
        uint _S1070 = 2U * _S1061;
        Quat_0 p_15 = quat_of_0(float4(*((&kernelContext_151)->frames_0+(_S1070 + 1U))) );
        float3 _S1071 = (float4(*((&kernelContext_151)->bodies_0+(b_45 + 2U))) ).xyz;
        thread Quat_0 _S1072 = p_15;
        Quat_0 _S1073 = conj_0(&_S1072);
        thread Quat_0 _S1074 = _S1073;
        float3 _S1075 = rotate_0(&_S1074, _S1069);
        float3 _S1076 = _S1071 * _S1075;
        thread Quat_0 _S1077 = p_15;
        float3 _S1078 = rotate_0(&_S1077, _S1076);
        uint _S1079 = 4U * k_35;
        float4 _S1080 = float4(*((&kernelContext_151)->chunks_0+_S1079)) ;
        float4 _S1081 = float4(*((&kernelContext_151)->chunks_0+(_S1079 + 1U))) ;
        float4 _S1082 = float4(*((&kernelContext_151)->chunks_0+(_S1079 + 2U))) ;
        f_11.xyz = f_11.xyz - float3(_S1080.w)  * (_S1068 * float3(inv_mass_6)  + cross(_S1078, _S1080.xyz - (float4(*((&kernelContext_151)->frames_0+_S1070)) ).xyz));
        float _S1083 = _S1078.x;
        float _S1084 = _S1081.w;
        float _S1085 = _S1078.y;
        float _S1086 = _S1082.x;
        float _S1087 = _S1078.z;
        float _S1088 = _S1082.y;
        t_11.xyz = t_11.xyz - float3(_S1081.x * _S1083 + _S1084 * _S1085 + _S1086 * _S1087, _S1084 * _S1083 + _S1081.y * _S1085 + _S1088 * _S1087, _S1086 * _S1083 + _S1088 * _S1085 + _S1081.z * _S1087);
    }
    float _S1089;
    if(fixed_1)
    {
        _S1089 = 1.0f;
    }
    else
    {
        _S1089 = 0.0f;
    }
    f_11.w = _S1089;
    *((&kernelContext_151)->lp_0+_S1065) = packed_float4(f_11) ;
    *((&kernelContext_151)->lp_0+_S1066) = packed_float4(t_11) ;
    return;
}

uint adj_bond_0(uint at_20, KernelContext_0 thread* kernelContext_152)
{
    return (as_type<uint>(((float4(*(kernelContext_152->ext_0+(kernelContext_152->params_0->ext_adj_0 + at_20 / 4U))) )[at_20 % 4U])));
}

uint lp_intact_0(uint k_36, KernelContext_0 thread* kernelContext_153)
{
    float4 _S1090 = float4(*(kernelContext_153->ext_0+(kernelContext_153->params_0->ext_chunks_0 + 4U * k_36))) ;
    uint _S1091 = (as_type<uint>((_S1090.x)));
    uint _S1092 = (as_type<uint>((_S1090.y)));
    uint e_4 = 0U;
    uint intact_0 = 0U;
    for(;;)
    {
        if(e_4 < _S1092)
        {
        }
        else
        {
            break;
        }
        uint _S1093 = w_alive_0(kernelContext_153);
        uint _S1094 = adj_bond_0(_S1091 + e_4, kernelContext_153);
        uint _S1095;
        if((*(kernelContext_153->work_0+(_S1093 + _S1094))) != 0U)
        {
            _S1095 = 1U;
        }
        else
        {
            _S1095 = 0U;
        }
        uint intact_1 = intact_0 + _S1095;
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

LpBond_0 lp_bond_0(uint i_18, KernelContext_0 thread* kernelContext_154)
{
    uint o_7 = 10U * i_18;
    thread LpBond_0 b_46;
    (&b_46)->centroid_0 = (float4(*(kernelContext_154->ext_0+o_7)) ).xyz;
    float3 _S1096 = (float4(*(kernelContext_154->ext_0+(o_7 + 1U))) ).xyz;
    (&b_46)->n_4 = _S1096;
    float3 _S1097 = (float4(*(kernelContext_154->ext_0+(o_7 + 2U))) ).xyz;
    (&b_46)->t1_2 = _S1097;
    (&b_46)->t2_0 = cross(_S1096, _S1097);
    (&(&b_46)->law_0)->geom0_0 = float4(*(kernelContext_154->ext_0+(o_7 + 3U))) ;
    (&(&b_46)->law_0)->geom1_0 = float4(*(kernelContext_154->ext_0+(o_7 + 4U))) ;
    (&(&b_46)->law_0)->stiff0_0 = float4(*(kernelContext_154->ext_0+(o_7 + 5U))) ;
    (&(&b_46)->law_0)->stiff1_0 = float4(*(kernelContext_154->ext_0+(o_7 + 6U))) ;
    (&(&b_46)->law_0)->rebar0_0 = float4(*(kernelContext_154->ext_0+(o_7 + 7U))) ;
    (&(&b_46)->law_0)->rebar1_0 = float4(*(kernelContext_154->ext_0+(o_7 + 8U))) ;
    (&(&b_46)->law_0)->ids_3 = (as_type<uint4>((float4(*(kernelContext_154->ext_0+(o_7 + 9U))) )));
    return b_46;
}

uint tri_0(uint i_19, uint j_7)
{
    return i_19 * (i_19 + 1U) / 2U + j_7;
}

bool lp_factor_shifted_0(uint k_38, const array<array<float, int(6)>, int(6)> thread* d_22, uint intact_2, KernelContext_0 thread* kernelContext_155)
{
    uint j2_0;
    float s_11;
    float s_12;
    float _S1098 = max((*d_22)[int(0)][int(0)], max((*d_22)[int(1)][int(1)], (*d_22)[int(2)][int(2)]));
    float _S1099 = max((*d_22)[int(3)][int(3)], max((*d_22)[int(4)][int(4)], (*d_22)[int(5)][int(5)]));
    bool ok_0;
    if(!(_S1098 > 0.0f))
    {
        ok_0 = true;
    }
    else
    {
        ok_0 = !(_S1099 > 0.0f);
    }
    if(ok_0)
    {
        return false;
    }
    uint attempt_0 = 0U;
    float shift_0 = 9.99999997475242708e-07f;
    for(;;)
    {
        if(attempt_0 < 4U)
        {
        }
        else
        {
            break;
        }
        thread array<float, int(21)> l_0;
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
            l_0[z0_0] = 0.0f;
            z0_0 = z0_0 + 1U;
        }
        ok_0 = true;
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
            bool ok_1 = ok_0;
            j2_0 = 0U;
            for(;;)
            {
                if(j2_0 <= i2_0)
                {
                }
                else
                {
                    break;
                }
                float _S1100 = (*d_22)[i2_0][j2_0];
                bool _S1101 = i2_0 == j2_0;
                if(_S1101)
                {
                    if(i2_0 < 3U)
                    {
                        s_11 = _S1098;
                    }
                    else
                    {
                        s_11 = _S1099;
                    }
                    s_11 = _S1100 + shift_0 * s_11;
                }
                else
                {
                    s_11 = _S1100;
                }
                uint p3_0 = 0U;
                s_12 = s_11;
                for(;;)
                {
                    if(p3_0 < j2_0)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_13 = s_12 - l_0[tri_0(i2_0, p3_0)] * l_0[tri_0(j2_0, p3_0)];
                    p3_0 = p3_0 + 1U;
                    s_12 = s_13;
                }
                if(_S1101)
                {
                    bool _S1102;
                    if(ok_1)
                    {
                        _S1102 = s_12 > 0.0f;
                    }
                    else
                    {
                        _S1102 = false;
                    }
                    l_0[tri_0(i2_0, i2_0)] = sqrt(max(s_12, 1.00000000317107685e-30f));
                    ok_1 = _S1102;
                }
                else
                {
                    l_0[tri_0(i2_0, j2_0)] = s_12 / l_0[tri_0(j2_0, j2_0)];
                }
                j2_0 = j2_0 + 1U;
            }
            uint _S1103 = i2_0 + 1U;
            ok_0 = ok_1;
            i2_0 = _S1103;
        }
        if(ok_0)
        {
            j2_0 = 0U;
            for(;;)
            {
                if(j2_0 < 6U)
                {
                }
                else
                {
                    break;
                }
                packed_float4 device* _S1104 = kernelContext_155->lp_0+(lp_l_0(k_38) + j2_0);
                uint _S1105 = 4U * j2_0;
                float _S1106 = l_0[_S1105];
                uint _S1107 = _S1105 + 1U;
                if(_S1107 < 21U)
                {
                    shift_0 = l_0[_S1107];
                }
                else
                {
                    shift_0 = 0.0f;
                }
                uint _S1108 = _S1105 + 2U;
                if(_S1108 < 21U)
                {
                    s_11 = l_0[_S1108];
                }
                else
                {
                    s_11 = 0.0f;
                }
                if(j2_0 == 5U)
                {
                    s_12 = (as_type<float>((intact_2 + 1U)));
                }
                else
                {
                    s_12 = l_0[_S1105 + 3U];
                }
                *_S1104 = packed_float4(float4(_S1106, shift_0, s_11, s_12)) ;
                j2_0 = j2_0 + 1U;
            }
            return true;
        }
        float shift_1 = shift_0 * 10.0f;
        attempt_0 = attempt_0 + 1U;
        shift_0 = shift_1;
    }
    return false;
}

bool lp_factor_0(uint k_39, KernelContext_0 thread* kernelContext_156)
{
    uint q_27;
    uint a_9;
    uint p2_0;
    uint q2_0;
    uint a2_0;
    float s_14;
    uint _S1109 = lp_intact_0(k_39, kernelContext_156);
    uint _S1110 = lp_l_0(k_39);
    uint _S1111 = _S1109 + 1U;
    if((as_type<uint>(((float4(*(kernelContext_156->lp_0+(_S1110 + 5U))) ).w))) == _S1111)
    {
        return true;
    }
    thread array<array<float, int(6)>, int(6)> d_23;
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
            d_23[p_16][q_27] = 0.0f;
            q_27 = q_27 + 1U;
        }
        p_16 = p_16 + 1U;
    }
    uint _S1112 = 4U * k_39;
    float4 _S1113 = float4(*(kernelContext_156->ext_0+(kernelContext_156->params_0->ext_chunks_0 + _S1112))) ;
    uint _S1114 = (as_type<uint>((_S1113.x)));
    uint _S1115 = (as_type<uint>((_S1113.y)));
    q_27 = 0U;
    for(;;)
    {
        if(q_27 < _S1115)
        {
        }
        else
        {
            break;
        }
        uint _S1116 = adj_bond_0(_S1114 + q_27, kernelContext_156);
        uint _S1117 = w_alive_0(kernelContext_156);
        if((*(kernelContext_156->work_0+(_S1117 + _S1116))) == 0U)
        {
            q_27 = q_27 + 1U;
            continue;
        }
        LpBond_0 _S1118 = lp_bond_0(_S1116, kernelContext_156);
        float3 _S1119 = _S1118.centroid_0 - (float4(*(kernelContext_156->chunks_0+_S1112)) ).xyz;
        float _S1120 = _S1118.law_0.stiff0_0.y;
        array<float, int(6)> _S1121 = { { _S1120, _S1120, _S1118.law_0.stiff0_0.x, _S1118.law_0.stiff0_0.z, _S1118.law_0.stiff0_0.w, _S1118.law_0.stiff1_0.x } };
        array<float3, int(3)> _S1122 = { { _S1118.t1_2, _S1118.t2_0, _S1118.n_4 } };
        thread array<array<float, int(6)>, int(6)> j_8;
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
            float3 c_20 = cross(_S1119, _S1122[a_9]);
            float _S1123 = _S1122[a_9].x;
            j_8[a_9][int(0)] = _S1123;
            float _S1124 = _S1122[a_9].y;
            j_8[a_9][int(1)] = _S1124;
            float _S1125 = _S1122[a_9].z;
            j_8[a_9][int(2)] = _S1125;
            j_8[a_9][int(3)] = c_20.x;
            j_8[a_9][int(4)] = c_20.y;
            j_8[a_9][int(5)] = c_20.z;
            j_8[3U + a_9][int(0)] = 0.0f;
            j_8[3U + a_9][int(1)] = 0.0f;
            j_8[3U + a_9][int(2)] = 0.0f;
            j_8[3U + a_9][int(3)] = _S1123;
            j_8[3U + a_9][int(4)] = _S1124;
            j_8[3U + a_9][int(5)] = _S1125;
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
                s_14 = 0.0f;
                for(;;)
                {
                    if(a2_0 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_15 = s_14 + j_8[a2_0][p2_0] * _S1121[a2_0] * j_8[a2_0][q2_0];
                    a2_0 = a2_0 + 1U;
                    s_14 = s_15;
                }
                d_23[p2_0][q2_0] = d_23[p2_0][q2_0] + s_14;
                q2_0 = q2_0 + 1U;
            }
            p2_0 = p2_0 + 1U;
        }
        q_27 = q_27 + 1U;
    }
    thread array<float, int(21)> l_1;
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
        l_1[a_9] = 0.0f;
        a_9 = a_9 + 1U;
    }
    bool ok_2 = true;
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
        bool ok_3 = ok_2;
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
            s_14 = d_23[p2_0][q2_0];
            for(;;)
            {
                if(a2_0 < q2_0)
                {
                }
                else
                {
                    break;
                }
                float s_16 = s_14 - l_1[tri_0(p2_0, a2_0)] * l_1[tri_0(q2_0, a2_0)];
                a2_0 = a2_0 + 1U;
                s_14 = s_16;
            }
            if(p2_0 == q2_0)
            {
                bool _S1126;
                if(ok_3)
                {
                    _S1126 = s_14 > 0.0f;
                }
                else
                {
                    _S1126 = false;
                }
                l_1[tri_0(p2_0, p2_0)] = sqrt(max(s_14, 1.00000000317107685e-30f));
                ok_3 = _S1126;
            }
            else
            {
                l_1[tri_0(p2_0, q2_0)] = s_14 / l_1[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + 1U;
        }
        uint _S1127 = p2_0 + 1U;
        ok_2 = ok_3;
        p2_0 = _S1127;
    }
    if(!ok_2)
    {
        thread array<array<float, int(6)>, int(6)> _S1128 = d_23;
        bool _S1129 = lp_factor_shifted_0(k_39, &_S1128, _S1109, kernelContext_156);
        return _S1129;
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
        packed_float4 device* _S1130 = kernelContext_156->lp_0+(_S1110 + q2_0);
        uint _S1131 = 4U * q2_0;
        float _S1132 = l_1[_S1131];
        uint _S1133 = _S1131 + 1U;
        if(_S1133 < 21U)
        {
            s_14 = l_1[_S1133];
        }
        else
        {
            s_14 = 0.0f;
        }
        uint _S1134 = _S1131 + 2U;
        float _S1135;
        if(_S1134 < 21U)
        {
            _S1135 = l_1[_S1134];
        }
        else
        {
            _S1135 = 0.0f;
        }
        float _S1136;
        if(q2_0 == 5U)
        {
            _S1136 = (as_type<float>((_S1111)));
        }
        else
        {
            _S1136 = l_1[_S1131 + 3U];
        }
        *_S1130 = packed_float4(float4(_S1132, s_14, _S1135, _S1136)) ;
        q2_0 = q2_0 + 1U;
    }
    return true;
}

[[kernel]] void fr_lp_block(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], atomic<uint> device* atom_39 [[buffer(5)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], packed_float4 device* lp_39 [[buffer(24)]], packed_float4 device* ext_41 [[buffer(23)]], packed_float4 device* engine_contacts_39 [[buffer(25)]], CrushParams_0 constant* crush_39 [[buffer(28)]], packed_float4 device* crush_rw_39 [[buffer(27)]], packed_float4 device* crush_tab_39 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_157;
    (&kernelContext_157)->params_0 = params_39;
    (&kernelContext_157)->atom_0 = atom_39;
    (&kernelContext_157)->work_0 = work_39;
    (&kernelContext_157)->frames_0 = frames_39;
    (&kernelContext_157)->bodies_0 = bodies_39;
    (&kernelContext_157)->chunks_0 = chunks_39;
    (&kernelContext_157)->sort_0 = sort_39;
    (&kernelContext_157)->manifolds_0 = manifolds_39;
    (&kernelContext_157)->loads_0 = loads_39;
    (&kernelContext_157)->bonds_0 = bonds_39;
    (&kernelContext_157)->args_0 = args_39;
    (&kernelContext_157)->info_0 = info_39;
    (&kernelContext_157)->scan_0 = scan_39;
    (&kernelContext_157)->parts_0 = parts_39;
    (&kernelContext_157)->colliders_0 = colliders_39;
    (&kernelContext_157)->collider_info_0 = collider_info_39;
    (&kernelContext_157)->view_chunks_0 = view_chunks_39;
    (&kernelContext_157)->view_0 = view_39;
    (&kernelContext_157)->objects_0 = objects_39;
    (&kernelContext_157)->shot_bodies_0 = shot_bodies_39;
    (&kernelContext_157)->lines_0 = lines_39;
    (&kernelContext_157)->geometry_0 = geometry_39;
    (&kernelContext_157)->shots_0 = shots_39;
    (&kernelContext_157)->lp_0 = lp_39;
    (&kernelContext_157)->ext_0 = ext_41;
    (&kernelContext_157)->engine_contacts_0 = engine_contacts_39;
    (&kernelContext_157)->crush_0 = crush_39;
    (&kernelContext_157)->crush_rw_0 = crush_rw_39;
    (&kernelContext_157)->crush_tab_0 = crush_tab_39;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_157)->g_scan_0 = &g_scan_39;
    threadgroup array<float, int(64)> g_il_best_39;
    (&kernelContext_157)->g_il_best_0 = &g_il_best_39;
    threadgroup array<float, int(256)> g_gi_x_39;
    (&kernelContext_157)->g_gi_x_0 = &g_gi_x_39;
    threadgroup array<float, int(256)> g_gi_m_39;
    (&kernelContext_157)->g_gi_m_0 = &g_gi_m_39;
    threadgroup array<uint, int(1280)> g_gi_lo_39;
    (&kernelContext_157)->g_gi_lo_0 = &g_gi_lo_39;
    threadgroup array<uint, int(1280)> g_gi_hi_39;
    (&kernelContext_157)->g_gi_hi_0 = &g_gi_hi_39;
    uint k_40 = id_36.x;
    bool _S1137;
    if(k_40 >= (params_39->chunk_count_0))
    {
        _S1137 = true;
    }
    else
    {
        _S1137 = ((float4(*((&kernelContext_157)->lp_0+lp_f_0(k_40))) ).w) != 0.0f;
    }
    if(_S1137)
    {
        return;
    }
    bool _S1138 = lp_factor_0(k_40, &kernelContext_157);
    if(!_S1138)
    {
        (*((&kernelContext_157)->lp_0+lp_f_0(k_40)))[int(3)] = 1.0f;
        uint _S1139 = a_counters_1(&kernelContext_157);
        uint _S1140 = atomic_fetch_or_explicit((&kernelContext_157)->atom_0+(_S1139 + 3U), 64U, memory_order_relaxed);
    }
    return;
}

uint st_x_0(uint k_41, uint state_1, uint copy_0)
{
    return 22U * k_41 + state_1 + 2U * copy_0;
}

uint st_d_0(uint k_42, uint state_2)
{
    return 22U * k_42 + state_2 + 4U;
}

float4 lp_side_0(uint i_20, KernelContext_0 thread* kernelContext_158)
{
    return float4(*(kernelContext_158->ext_0+(kernelContext_158->params_0->ext_sides_0 + i_20))) ;
}

void st_sweep_0(uint k_43, uint state_3, uint from_1, float c1_1, float c2_1, KernelContext_0 thread* kernelContext_159)
{
    uint ka_0;
    uint kb_0;
    float s_17;
    uint to_1 = 1U - from_1;
    uint _S1141 = st_x_0(k_43, state_3, from_1);
    float4 _S1142 = float4(*(kernelContext_159->lp_0+_S1141)) ;
    float4 _S1143 = float4(*(kernelContext_159->lp_0+(_S1141 + 1U))) ;
    uint _S1144 = st_f_0(k_43, state_3);
    float4 _S1145 = float4(*(kernelContext_159->lp_0+_S1144)) ;
    if((_S1145.w) != 0.0f)
    {
        uint _S1146 = st_x_0(k_43, state_3, to_1);
        *(kernelContext_159->lp_0+_S1146) = packed_float4(_S1142) ;
        *(kernelContext_159->lp_0+(_S1146 + 1U)) = packed_float4(_S1143) ;
        uint _S1147 = st_d_0(k_43, state_3);
        packed_float4 _S1148 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_159->lp_0+_S1147) = _S1148;
        *(kernelContext_159->lp_0+(_S1147 + 1U)) = _S1148;
        return;
    }
    float _S1149 = _S1145.y;
    float _S1150 = _S1145.z;
    thread array<float, int(6)> r_23;
    r_23[int(0)] = _S1145.x;
    r_23[int(1)] = _S1149;
    r_23[int(2)] = _S1150;
    r_23[int(3)] = 0.0f;
    r_23[int(4)] = 0.0f;
    r_23[int(5)] = 0.0f;
    float3 t_12 = (float4(*(kernelContext_159->lp_0+(_S1144 + 1U))) ).xyz;
    r_23[int(3)] = t_12.x;
    r_23[int(4)] = t_12.y;
    r_23[int(5)] = t_12.z;
    float4 _S1151 = float4(*(kernelContext_159->ext_0+(kernelContext_159->params_0->ext_chunks_0 + 4U * k_43))) ;
    uint _S1152 = (as_type<uint>((_S1151.x)));
    uint _S1153 = (as_type<uint>((_S1151.y)));
    uint e_5 = 0U;
    for(;;)
    {
        if(e_5 < _S1153)
        {
        }
        else
        {
            break;
        }
        uint _S1154 = 5U * (_S1152 + e_5);
        float4 _S1155 = lp_side_0(_S1154, kernelContext_159);
        uint _S1156 = w_alive_0(kernelContext_159);
        if((*(kernelContext_159->work_0+(_S1156 + (as_type<uint>((_S1155.w)))))) == 0U)
        {
            e_5 = e_5 + 1U;
            continue;
        }
        float4 _S1157 = lp_side_0(_S1154 + 1U, kernelContext_159);
        float4 _S1158 = lp_side_0(_S1154 + 2U, kernelContext_159);
        float4 _S1159 = lp_side_0(_S1154 + 3U, kernelContext_159);
        float4 _S1160 = lp_side_0(_S1154 + 4U, kernelContext_159);
        bool first_side_0 = (_S1160.w) != 0.0f;
        if(first_side_0)
        {
            ka_0 = k_43;
        }
        else
        {
            ka_0 = (as_type<uint>((_S1157.w)));
        }
        if(first_side_0)
        {
            kb_0 = (as_type<uint>((_S1157.w)));
        }
        else
        {
            kb_0 = k_43;
        }
        float3 ra_0 = _S1155.xyz;
        float3 rb_0 = _S1157.xyz;
        float3 t1_3 = _S1158.xyz;
        float3 n_5 = _S1159.xyz;
        float3 t2_1 = cross(n_5, t1_3);
        uint _S1161 = st_x_0(ka_0, state_3, from_1);
        uint _S1162 = st_x_0(kb_0, state_3, from_1);
        float3 _S1163 = (float4(*(kernelContext_159->lp_0+(_S1162 + 1U))) ).xyz;
        float3 _S1164 = (float4(*(kernelContext_159->lp_0+(_S1161 + 1U))) ).xyz;
        float3 delta_0 = (float4(*(kernelContext_159->lp_0+_S1162)) ).xyz + cross(_S1163, rb_0) - ((float4(*(kernelContext_159->lp_0+_S1161)) ).xyz + cross(_S1164, ra_0));
        float3 phi_0 = _S1163 - _S1164;
        float _S1165 = _S1158.w;
        float3 fb_0 = t1_3 * float3((_S1165 * dot(delta_0, t1_3)))  + t2_1 * float3((_S1165 * dot(delta_0, t2_1)))  + n_5 * float3((_S1159.w * dot(delta_0, n_5))) ;
        float3 mb_0 = t1_3 * float3((_S1160.x * dot(phi_0, t1_3)))  + t2_1 * float3((_S1160.y * dot(phi_0, t2_1)))  + n_5 * float3((_S1160.z * dot(phi_0, n_5))) ;
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
        float3 _S1166;
        if(first_side_0)
        {
            _S1166 = mb_0;
        }
        else
        {
            _S1166 = - mb_0;
        }
        float3 lm_0 = _S1166 + cross(arm_0, lf_0);
        r_23[int(0)] = r_23[int(0)] + lf_0.x;
        r_23[int(1)] = r_23[int(1)] + lf_0.y;
        r_23[int(2)] = r_23[int(2)] + lf_0.z;
        r_23[int(3)] = r_23[int(3)] + lm_0.x;
        r_23[int(4)] = r_23[int(4)] + lm_0.y;
        r_23[int(5)] = r_23[int(5)] + lm_0.z;
        e_5 = e_5 + 1U;
    }
    thread array<float, int(21)> l_2;
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
        l_2[ka_0] = 0.0f;
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
        float4 _S1167 = float4(*(kernelContext_159->lp_0+(lp_l_0(k_43) + kb_0))) ;
        uint _S1168 = 4U * kb_0;
        l_2[_S1168] = _S1167.x;
        uint _S1169 = _S1168 + 1U;
        if(_S1169 < 21U)
        {
            l_2[_S1169] = _S1167.y;
        }
        uint _S1170 = _S1168 + 2U;
        if(_S1170 < 21U)
        {
            l_2[_S1170] = _S1167.z;
        }
        uint _S1171 = _S1168 + 3U;
        if(_S1171 < 21U)
        {
            l_2[_S1171] = _S1167.w;
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
    uint i2_1 = 0U;
    for(;;)
    {
        if(i2_1 < 6U)
        {
        }
        else
        {
            break;
        }
        uint p_17 = 0U;
        s_17 = r_23[i2_1];
        for(;;)
        {
            if(p_17 < i2_1)
            {
            }
            else
            {
                break;
            }
            float s_18 = s_17 - l_2[tri_0(i2_1, p_17)] * y_2[p_17];
            p_17 = p_17 + 1U;
            s_17 = s_18;
        }
        y_2[i2_1] = s_17 / l_2[tri_0(i2_1, i2_1)];
        i2_1 = i2_1 + 1U;
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
        s_17 = y_2[i3_0];
        for(;;)
        {
            if(p2_1 < int(6))
            {
            }
            else
            {
                break;
            }
            float s_19 = s_17 - l_2[tri_0(uint(p2_1), uint(i3_0))] * z_3[p2_1];
            p2_1 = p2_1 + int(1);
            s_17 = s_19;
        }
        uint _S1172 = uint(i3_0);
        z_3[i3_0] = s_17 / l_2[tri_0(_S1172, _S1172)];
        i3_0 = i3_0 - int(1);
    }
    uint _S1173 = st_d_0(k_43, state_3);
    float4 _S1174 = float4(*(kernelContext_159->lp_0+_S1173)) ;
    thread float4 d0_0 = _S1174;
    uint _S1175 = _S1173 + 1U;
    float4 _S1176 = float4(*(kernelContext_159->lp_0+_S1175)) ;
    thread float4 d1_0 = _S1176;
    float3 _S1177 = float3(c1_1) ;
    float3 _S1178 = float3(c2_1) ;
    d0_0.xyz = _S1177 * _S1174.xyz + _S1178 * float3(z_3[int(0)], z_3[int(1)], z_3[int(2)]);
    d1_0.xyz = _S1177 * _S1176.xyz + _S1178 * float3(z_3[int(3)], z_3[int(4)], z_3[int(5)]);
    *(kernelContext_159->lp_0+_S1173) = packed_float4(d0_0) ;
    *(kernelContext_159->lp_0+_S1175) = packed_float4(d1_0) ;
    uint _S1179 = st_x_0(k_43, state_3, to_1);
    *(kernelContext_159->lp_0+_S1179) = packed_float4(float4(_S1142.xyz + d0_0.xyz, 0.0f)) ;
    *(kernelContext_159->lp_0+(_S1179 + 1U)) = packed_float4(float4(_S1143.xyz + d1_0.xyz, 0.0f)) ;
    return;
}

void lp_sweep_0(uint k_44, uint from_2, KernelContext_0 thread* kernelContext_160)
{
    st_sweep_0(k_44, 0U, from_2, kernelContext_160->params_0->lp_c1_0, kernelContext_160->params_0->lp_c2_0, kernelContext_160);
    return;
}

[[kernel]] void fr_lp_sweep_even(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], atomic<uint> device* atom_40 [[buffer(5)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], packed_float4 device* lp_40 [[buffer(24)]], packed_float4 device* ext_42 [[buffer(23)]], packed_float4 device* engine_contacts_40 [[buffer(25)]], CrushParams_0 constant* crush_40 [[buffer(28)]], packed_float4 device* crush_rw_40 [[buffer(27)]], packed_float4 device* crush_tab_40 [[buffer(26)]])
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
    threadgroup array<float, int(256)> g_gi_x_40;
    (&kernelContext_161)->g_gi_x_0 = &g_gi_x_40;
    threadgroup array<float, int(256)> g_gi_m_40;
    (&kernelContext_161)->g_gi_m_0 = &g_gi_m_40;
    threadgroup array<uint, int(1280)> g_gi_lo_40;
    (&kernelContext_161)->g_gi_lo_0 = &g_gi_lo_40;
    threadgroup array<uint, int(1280)> g_gi_hi_40;
    (&kernelContext_161)->g_gi_hi_0 = &g_gi_hi_40;
    uint _S1180 = id_37.x;
    if(_S1180 < (params_40->chunk_count_0))
    {
        lp_sweep_0(_S1180, 0U, &kernelContext_161);
    }
    return;
}

[[kernel]] void fr_lp_sweep_odd(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], atomic<uint> device* atom_41 [[buffer(5)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], packed_float4 device* lp_41 [[buffer(24)]], packed_float4 device* ext_43 [[buffer(23)]], packed_float4 device* engine_contacts_41 [[buffer(25)]], CrushParams_0 constant* crush_41 [[buffer(28)]], packed_float4 device* crush_rw_41 [[buffer(27)]], packed_float4 device* crush_tab_41 [[buffer(26)]])
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
    threadgroup array<float, int(256)> g_gi_x_41;
    (&kernelContext_162)->g_gi_x_0 = &g_gi_x_41;
    threadgroup array<float, int(256)> g_gi_m_41;
    (&kernelContext_162)->g_gi_m_0 = &g_gi_m_41;
    threadgroup array<uint, int(1280)> g_gi_lo_41;
    (&kernelContext_162)->g_gi_lo_0 = &g_gi_lo_41;
    threadgroup array<uint, int(1280)> g_gi_hi_41;
    (&kernelContext_162)->g_gi_hi_0 = &g_gi_hi_41;
    uint _S1181 = id_38.x;
    if(_S1181 < (params_41->chunk_count_0))
    {
        lp_sweep_0(_S1181, 1U, &kernelContext_162);
    }
    return;
}

void st_force_0(const LpBond_0 thread* b_47, uint ka_1, uint kb_1, uint state_4, uint c_21, float3 thread* q_lin_0, float3 thread* q_ang_0, KernelContext_0 thread* kernelContext_163)
{
    uint _S1182 = st_x_0(ka_1, state_4, c_21);
    uint _S1183 = st_x_0(kb_1, state_4, c_21);
    float3 _S1184 = (float4(*(kernelContext_163->lp_0+(_S1183 + 1U))) ).xyz;
    float3 _S1185 = (float4(*(kernelContext_163->lp_0+(_S1182 + 1U))) ).xyz;
    float3 delta_1 = (float4(*(kernelContext_163->lp_0+_S1183)) ).xyz + cross(_S1184, b_47->centroid_0 - (float4(*(kernelContext_163->chunks_0+4U * kb_1)) ).xyz) - ((float4(*(kernelContext_163->lp_0+_S1182)) ).xyz + cross(_S1185, b_47->centroid_0 - (float4(*(kernelContext_163->chunks_0+4U * ka_1)) ).xyz));
    float3 phi_1 = _S1184 - _S1185;
    float4 _S1186 = (&b_47->law_0)->stiff0_0;
    float kt_0 = (&b_47->law_0)->stiff1_0.x;
    float _S1187 = (&b_47->law_0)->stiff0_0.y;
    float3 _S1188 = b_47->t1_2;
    float3 _S1189 = b_47->t2_0;
    float3 _S1190 = b_47->n_4;
    *q_lin_0 = float3(_S1187 * dot(delta_1, b_47->t1_2), _S1187 * dot(delta_1, b_47->t2_0), (&b_47->law_0)->stiff0_0.x * dot(delta_1, b_47->n_4));
    *q_ang_0 = float3(_S1186.z * dot(phi_1, _S1188), _S1186.w * dot(phi_1, _S1189), kt_0 * dot(phi_1, _S1190));
    return;
}

void st_force_1(const LpBond_0 thread* b_48, uint ka_2, uint kb_2, uint state_5, uint c_22, float3 thread* q_lin_1, float3 thread* q_ang_1, KernelContext_0 thread* kernelContext_164)
{
    uint _S1191 = st_x_0(ka_2, state_5, c_22);
    uint _S1192 = st_x_0(kb_2, state_5, c_22);
    float3 _S1193 = (float4(*(kernelContext_164->lp_0+(_S1192 + 1U))) ).xyz;
    float3 _S1194 = (float4(*(kernelContext_164->lp_0+(_S1191 + 1U))) ).xyz;
    float3 delta_2 = (float4(*(kernelContext_164->lp_0+_S1192)) ).xyz + cross(_S1193, b_48->centroid_0 - (float4(*(kernelContext_164->chunks_0+4U * kb_2)) ).xyz) - ((float4(*(kernelContext_164->lp_0+_S1191)) ).xyz + cross(_S1194, b_48->centroid_0 - (float4(*(kernelContext_164->chunks_0+4U * ka_2)) ).xyz));
    float3 phi_2 = _S1193 - _S1194;
    float4 _S1195 = (&b_48->law_0)->stiff0_0;
    float kt_1 = (&b_48->law_0)->stiff1_0.x;
    float _S1196 = (&b_48->law_0)->stiff0_0.y;
    float3 _S1197 = b_48->t1_2;
    float3 _S1198 = b_48->t2_0;
    float3 _S1199 = b_48->n_4;
    *q_lin_1 = float3(_S1196 * dot(delta_2, b_48->t1_2), _S1196 * dot(delta_2, b_48->t2_0), (&b_48->law_0)->stiff0_0.x * dot(delta_2, b_48->n_4));
    *q_ang_1 = float3(_S1195.z * dot(phi_2, _S1197), _S1195.w * dot(phi_2, _S1198), kt_1 * dot(phi_2, _S1199));
    return;
}

void lp_force_0(const LpBond_0 thread* b_49, uint ka_3, uint kb_3, uint c_23, float3 thread* q_lin_2, float3 thread* q_ang_2, KernelContext_0 thread* kernelContext_165)
{
    st_force_1(b_49, ka_3, kb_3, 0U, c_23, q_lin_2, q_ang_2, kernelContext_165);
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

Measures_0 stress_measures_0(const JointBond_0 thread* b_50, float3 q_lin_3, float3 q_ang_3)
{
    float area_0 = b_50->geom0_0.x;
    float _S1200 = q_lin_3.z;
    float axial_0 = (metal::fast::divide((_S1200), (area_0)));
    float _S1201 = (metal::fast::divide((abs(q_ang_3.x)), (b_50->geom1_0.x)));
    float _S1202 = (metal::fast::divide((abs(q_ang_3.y)), (b_50->geom1_0.y)));
    float bending_0 = _S1201 + _S1202;
    float _S1203 = q_lin_3.x;
    float _S1204 = q_lin_3.y;
    float _S1205 = (metal::fast::sqrt((_S1203 * _S1203 + _S1204 * _S1204)));
    float _S1206 = (metal::fast::divide((_S1205), (area_0)));
    float _S1207 = (metal::fast::divide((abs(q_ang_3.z)), (b_50->geom0_0.w)));
    float shear_1 = _S1206 + _S1207;
    thread Measures_0 m_3;
    (&m_3)->tension_0 = axial_0 + bending_0;
    (&m_3)->shear_0 = shear_1;
    float _S1208 = - axial_0;
    (&m_3)->normal_compression_0 = max(_S1208, 0.0f);
    (&m_3)->compression_0 = _S1208 + bending_0;
    (&m_3)->compressive_force_0 = max(- _S1200, 0.0f);
    return m_3;
}

JointMaterial_0 lp_material_0(uint m_4, KernelContext_0 thread* kernelContext_166)
{
    uint o_8 = kernelContext_166->params_0->ext_mat_0 + 5U * m_4;
    thread JointMaterial_0 j_9;
    (&j_9)->strength_0 = float4(*(kernelContext_166->ext_0+o_8)) ;
    (&j_9)->energy_0 = float4(*(kernelContext_166->ext_0+(o_8 + 1U))) ;
    (&j_9)->dif_0 = float4(*(kernelContext_166->ext_0+(o_8 + 2U))) ;
    (&j_9)->misc_0 = float4(*(kernelContext_166->ext_0+(o_8 + 3U))) ;
    (&j_9)->kind_flags_0 = (as_type<uint4>((float4(*(kernelContext_166->ext_0+(o_8 + 4U))) )));
    return j_9;
}

float infinity_0()
{
    return float_bits_0(2139095040U);
}

float4 failure_indices_0(const JointMaterial_0 thread* mat_1, const JointBond_0 thread* b_51, const Measures_0 thread* m_5, float multiplier_0)
{
    float fc_2 = mat_1->strength_0.y * multiplier_0;
    float _S1209 = min(mat_1->strength_0.z * multiplier_0 + mat_1->strength_0.w * m_5->normal_compression_0, mat_1->energy_0.x * multiplier_0);
    thread float4 idx_0;
    float _S1210 = (metal::fast::divide((m_5->tension_0), (mat_1->strength_0.x * multiplier_0)));
    idx_0.x = max(_S1210, 0.0f);
    float _S1211;
    if(_S1209 > 0.0f)
    {
        float _S1212 = (metal::fast::divide((m_5->shear_0), (_S1209)));
        _S1211 = _S1212;
    }
    else
    {
        _S1211 = infinity_0();
    }
    idx_0.y = _S1211;
    float _S1213 = (metal::fast::divide((m_5->compression_0), (fc_2)));
    idx_0.z = max(_S1213, 0.0f);
    float _S1214 = b_51->stiff1_0.y;
    if(_S1214 > 0.0f)
    {
        float _S1215 = (metal::fast::divide((m_5->compressive_force_0), (_S1214)));
        _S1211 = _S1215;
    }
    else
    {
        _S1211 = 0.0f;
    }
    idx_0.w = _S1211;
    return idx_0;
}

[[kernel]] void fr_lp_breaks(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], atomic<uint> device* atom_42 [[buffer(5)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], packed_float4 device* lp_42 [[buffer(24)]], packed_float4 device* ext_44 [[buffer(23)]], packed_float4 device* engine_contacts_42 [[buffer(25)]], CrushParams_0 constant* crush_42 [[buffer(28)]], packed_float4 device* crush_rw_42 [[buffer(27)]], packed_float4 device* crush_tab_42 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_167;
    (&kernelContext_167)->params_0 = params_42;
    (&kernelContext_167)->atom_0 = atom_42;
    (&kernelContext_167)->work_0 = work_42;
    (&kernelContext_167)->frames_0 = frames_42;
    (&kernelContext_167)->bodies_0 = bodies_42;
    (&kernelContext_167)->chunks_0 = chunks_42;
    (&kernelContext_167)->sort_0 = sort_42;
    (&kernelContext_167)->manifolds_0 = manifolds_42;
    (&kernelContext_167)->loads_0 = loads_42;
    (&kernelContext_167)->bonds_0 = bonds_42;
    (&kernelContext_167)->args_0 = args_42;
    (&kernelContext_167)->info_0 = info_42;
    (&kernelContext_167)->scan_0 = scan_42;
    (&kernelContext_167)->parts_0 = parts_42;
    (&kernelContext_167)->colliders_0 = colliders_42;
    (&kernelContext_167)->collider_info_0 = collider_info_42;
    (&kernelContext_167)->view_chunks_0 = view_chunks_42;
    (&kernelContext_167)->view_0 = view_42;
    (&kernelContext_167)->objects_0 = objects_42;
    (&kernelContext_167)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_167)->lines_0 = lines_42;
    (&kernelContext_167)->geometry_0 = geometry_42;
    (&kernelContext_167)->shots_0 = shots_42;
    (&kernelContext_167)->lp_0 = lp_42;
    (&kernelContext_167)->ext_0 = ext_44;
    (&kernelContext_167)->engine_contacts_0 = engine_contacts_42;
    (&kernelContext_167)->crush_0 = crush_42;
    (&kernelContext_167)->crush_rw_0 = crush_rw_42;
    (&kernelContext_167)->crush_tab_0 = crush_tab_42;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_167)->g_scan_0 = &g_scan_42;
    threadgroup array<float, int(64)> g_il_best_42;
    (&kernelContext_167)->g_il_best_0 = &g_il_best_42;
    threadgroup array<float, int(256)> g_gi_x_42;
    (&kernelContext_167)->g_gi_x_0 = &g_gi_x_42;
    threadgroup array<float, int(256)> g_gi_m_42;
    (&kernelContext_167)->g_gi_m_0 = &g_gi_m_42;
    threadgroup array<uint, int(1280)> g_gi_lo_42;
    (&kernelContext_167)->g_gi_lo_0 = &g_gi_lo_42;
    threadgroup array<uint, int(1280)> g_gi_hi_42;
    (&kernelContext_167)->g_gi_hi_0 = &g_gi_hi_42;
    uint i_21 = id_39.x;
    bool _S1216;
    if(i_21 >= (params_42->bond_count_0))
    {
        _S1216 = true;
    }
    else
    {
        uint _S1217 = w_alive_0(&kernelContext_167);
        _S1216 = (*((&kernelContext_167)->work_0+(_S1217 + i_21))) == 0U;
    }
    if(_S1216)
    {
        return;
    }
    uint4 _S1218 = uint4(*((&kernelContext_167)->bonds_0+i_21)) ;
    LpBond_0 _S1219 = lp_bond_0(i_21, &kernelContext_167);
    uint _S1220 = _S1218.x;
    uint _S1221 = _S1218.y;
    thread LpBond_0 _S1222 = _S1219;
    thread float3 q_lin_4;
    thread float3 q_ang_4;
    lp_force_0(&_S1222, _S1220, _S1221, 0U, &q_lin_4, &q_ang_4, &kernelContext_167);
    thread JointBond_0 _S1223 = _S1219.law_0;
    Measures_0 _S1224 = stress_measures_0(&_S1223, q_lin_4, q_ang_4);
    JointMaterial_0 _S1225 = lp_material_0(_S1219.law_0.ids_3.x, &kernelContext_167);
    float _S1226 = _S1219.law_0.geom1_0.w;
    thread JointMaterial_0 _S1227 = _S1225;
    thread JointBond_0 _S1228 = _S1219.law_0;
    thread Measures_0 _S1229 = _S1224;
    float4 _S1230 = failure_indices_0(&_S1227, &_S1228, &_S1229, _S1226);
    float _S1231 = max(max(_S1230.x, _S1230.y), max(_S1230.z, _S1230.w));
    uint _S1232 = a_counters_1(&kernelContext_167);
    uint _S1233 = atomic_fetch_max_explicit((&kernelContext_167)->atom_0+(_S1232 + 15U), (as_type<uint>((max(_S1231, 0.0f)))), memory_order_relaxed);
    if(_S1231 >= 1.0f)
    {
        uint _S1234 = w_alive_0(&kernelContext_167);
        *((&kernelContext_167)->work_0+(_S1234 + i_21)) = 0U;
        uint _S1235 = a_dirty_0(&kernelContext_167);
        uint _S1236 = body_of_0(_S1220, &kernelContext_167);
        uint _S1237 = atomic_fetch_or_explicit((&kernelContext_167)->atom_0+(_S1235 + _S1236), 1U, memory_order_relaxed);
        uint _S1238 = atomic_fetch_add_explicit((&kernelContext_167)->atom_0+_S1232, 1U, memory_order_relaxed);
        uint _S1239 = atomic_fetch_add_explicit((&kernelContext_167)->atom_0+(_S1232 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_21, &kernelContext_167);
    }
    return;
}

uint lp_view_0(uint k_45, KernelContext_0 thread* kernelContext_168)
{
    return 22U * kernelContext_168->params_0->chunk_count_0 + 7U * kernelContext_168->params_0->body_count_0 + k_45;
}

[[kernel]] void fr_lp_view(uint3 id_40 [[thread_position_in_grid]], FractureParams_0 constant* params_43 [[buffer(0)]], atomic<uint> device* atom_43 [[buffer(5)]], uint device* work_43 [[buffer(4)]], packed_float4 device* frames_43 [[buffer(6)]], packed_float4 device* bodies_43 [[buffer(7)]], packed_float4 device* chunks_43 [[buffer(1)]], uint device* sort_43 [[buffer(11)]], Record_natural_0 device* manifolds_43 [[buffer(10)]], packed_float4 device* loads_43 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_43 [[buffer(13)]], uint device* info_43 [[buffer(15)]], ScanParams_0 constant* scan_43 [[buffer(14)]], packed_float4 device* parts_43 [[buffer(3)]], packed_float4 device* colliders_43 [[buffer(8)]], packed_uint4 device* collider_info_43 [[buffer(9)]], packed_float4 device* view_chunks_43 [[buffer(17)]], ViewParams_0 constant* view_43 [[buffer(16)]], packed_float4 device* objects_43 [[buffer(18)]], uint device* shot_bodies_43 [[buffer(20)]], float device* lines_43 [[buffer(19)]], packed_float4 device* geometry_43 [[buffer(21)]], packed_float4 device* shots_43 [[buffer(22)]], packed_float4 device* lp_43 [[buffer(24)]], packed_float4 device* ext_45 [[buffer(23)]], packed_float4 device* engine_contacts_43 [[buffer(25)]], CrushParams_0 constant* crush_43 [[buffer(28)]], packed_float4 device* crush_rw_43 [[buffer(27)]], packed_float4 device* crush_tab_43 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_169;
    (&kernelContext_169)->params_0 = params_43;
    (&kernelContext_169)->atom_0 = atom_43;
    (&kernelContext_169)->work_0 = work_43;
    (&kernelContext_169)->frames_0 = frames_43;
    (&kernelContext_169)->bodies_0 = bodies_43;
    (&kernelContext_169)->chunks_0 = chunks_43;
    (&kernelContext_169)->sort_0 = sort_43;
    (&kernelContext_169)->manifolds_0 = manifolds_43;
    (&kernelContext_169)->loads_0 = loads_43;
    (&kernelContext_169)->bonds_0 = bonds_43;
    (&kernelContext_169)->args_0 = args_43;
    (&kernelContext_169)->info_0 = info_43;
    (&kernelContext_169)->scan_0 = scan_43;
    (&kernelContext_169)->parts_0 = parts_43;
    (&kernelContext_169)->colliders_0 = colliders_43;
    (&kernelContext_169)->collider_info_0 = collider_info_43;
    (&kernelContext_169)->view_chunks_0 = view_chunks_43;
    (&kernelContext_169)->view_0 = view_43;
    (&kernelContext_169)->objects_0 = objects_43;
    (&kernelContext_169)->shot_bodies_0 = shot_bodies_43;
    (&kernelContext_169)->lines_0 = lines_43;
    (&kernelContext_169)->geometry_0 = geometry_43;
    (&kernelContext_169)->shots_0 = shots_43;
    (&kernelContext_169)->lp_0 = lp_43;
    (&kernelContext_169)->ext_0 = ext_45;
    (&kernelContext_169)->engine_contacts_0 = engine_contacts_43;
    (&kernelContext_169)->crush_0 = crush_43;
    (&kernelContext_169)->crush_rw_0 = crush_rw_43;
    (&kernelContext_169)->crush_tab_0 = crush_tab_43;
    threadgroup array<uint, int(1024)> g_scan_43;
    (&kernelContext_169)->g_scan_0 = &g_scan_43;
    threadgroup array<float, int(64)> g_il_best_43;
    (&kernelContext_169)->g_il_best_0 = &g_il_best_43;
    threadgroup array<float, int(256)> g_gi_x_43;
    (&kernelContext_169)->g_gi_x_0 = &g_gi_x_43;
    threadgroup array<float, int(256)> g_gi_m_43;
    (&kernelContext_169)->g_gi_m_0 = &g_gi_m_43;
    threadgroup array<uint, int(1280)> g_gi_lo_43;
    (&kernelContext_169)->g_gi_lo_0 = &g_gi_lo_43;
    threadgroup array<uint, int(1280)> g_gi_hi_43;
    (&kernelContext_169)->g_gi_hi_0 = &g_gi_hi_43;
    uint k_46 = id_40.x;
    if(k_46 >= (params_43->chunk_count_0))
    {
        return;
    }
    float4 _S1240 = float4(*((&kernelContext_169)->ext_0+((&kernelContext_169)->params_0->ext_chunks_0 + 4U * k_46))) ;
    uint _S1241 = (as_type<uint>((_S1240.x)));
    uint _S1242 = (as_type<uint>((_S1240.y)));
    float4 worst_0 = float4(0.0f) ;
    uint e_6 = 0U;
    for(;;)
    {
        if(e_6 < _S1242)
        {
        }
        else
        {
            break;
        }
        float4 _S1243 = lp_side_0(5U * (_S1241 + e_6), &kernelContext_169);
        uint i_22 = (as_type<uint>((_S1243.w)));
        uint _S1244 = w_alive_0(&kernelContext_169);
        if((*((&kernelContext_169)->work_0+(_S1244 + i_22))) == 0U)
        {
            e_6 = e_6 + 1U;
            continue;
        }
        uint4 _S1245 = uint4(*((&kernelContext_169)->bonds_0+i_22)) ;
        LpBond_0 _S1246 = lp_bond_0(i_22, &kernelContext_169);
        uint _S1247 = _S1245.x;
        uint _S1248 = _S1245.y;
        thread LpBond_0 _S1249 = _S1246;
        thread float3 q_lin_5;
        thread float3 q_ang_5;
        lp_force_0(&_S1249, _S1247, _S1248, 0U, &q_lin_5, &q_ang_5, &kernelContext_169);
        thread JointBond_0 _S1250 = _S1246.law_0;
        Measures_0 _S1251 = stress_measures_0(&_S1250, q_lin_5, q_ang_5);
        JointMaterial_0 _S1252 = lp_material_0(_S1246.law_0.ids_3.x, &kernelContext_169);
        float _S1253 = _S1246.law_0.geom1_0.w;
        thread JointMaterial_0 _S1254 = _S1252;
        thread JointBond_0 _S1255 = _S1246.law_0;
        thread Measures_0 _S1256 = _S1251;
        float4 _S1257 = failure_indices_0(&_S1254, &_S1255, &_S1256, _S1253);
        worst_0 = max(worst_0, _S1257);
        e_6 = e_6 + 1U;
    }
    uint _S1258 = lp_view_0(k_46, &kernelContext_169);
    *((&kernelContext_169)->lp_0+_S1258) = packed_float4(worst_0) ;
    return;
}

[[kernel]] void fr_il_first_reset(uint3 id_41 [[thread_position_in_grid]], FractureParams_0 constant* params_44 [[buffer(0)]], atomic<uint> device* atom_44 [[buffer(5)]], uint device* work_44 [[buffer(4)]], packed_float4 device* frames_44 [[buffer(6)]], packed_float4 device* bodies_44 [[buffer(7)]], packed_float4 device* chunks_44 [[buffer(1)]], uint device* sort_44 [[buffer(11)]], Record_natural_0 device* manifolds_44 [[buffer(10)]], packed_float4 device* loads_44 [[buffer(12)]], packed_uint4 device* bonds_44 [[buffer(2)]], uint device* args_44 [[buffer(13)]], uint device* info_44 [[buffer(15)]], ScanParams_0 constant* scan_44 [[buffer(14)]], packed_float4 device* parts_44 [[buffer(3)]], packed_float4 device* colliders_44 [[buffer(8)]], packed_uint4 device* collider_info_44 [[buffer(9)]], packed_float4 device* view_chunks_44 [[buffer(17)]], ViewParams_0 constant* view_44 [[buffer(16)]], packed_float4 device* objects_44 [[buffer(18)]], uint device* shot_bodies_44 [[buffer(20)]], float device* lines_44 [[buffer(19)]], packed_float4 device* geometry_44 [[buffer(21)]], packed_float4 device* shots_44 [[buffer(22)]], packed_float4 device* lp_44 [[buffer(24)]], packed_float4 device* ext_46 [[buffer(23)]], packed_float4 device* engine_contacts_44 [[buffer(25)]], CrushParams_0 constant* crush_44 [[buffer(28)]], packed_float4 device* crush_rw_44 [[buffer(27)]], packed_float4 device* crush_tab_44 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_170;
    (&kernelContext_170)->params_0 = params_44;
    (&kernelContext_170)->atom_0 = atom_44;
    (&kernelContext_170)->work_0 = work_44;
    (&kernelContext_170)->frames_0 = frames_44;
    (&kernelContext_170)->bodies_0 = bodies_44;
    (&kernelContext_170)->chunks_0 = chunks_44;
    (&kernelContext_170)->sort_0 = sort_44;
    (&kernelContext_170)->manifolds_0 = manifolds_44;
    (&kernelContext_170)->loads_0 = loads_44;
    (&kernelContext_170)->bonds_0 = bonds_44;
    (&kernelContext_170)->args_0 = args_44;
    (&kernelContext_170)->info_0 = info_44;
    (&kernelContext_170)->scan_0 = scan_44;
    (&kernelContext_170)->parts_0 = parts_44;
    (&kernelContext_170)->colliders_0 = colliders_44;
    (&kernelContext_170)->collider_info_0 = collider_info_44;
    (&kernelContext_170)->view_chunks_0 = view_chunks_44;
    (&kernelContext_170)->view_0 = view_44;
    (&kernelContext_170)->objects_0 = objects_44;
    (&kernelContext_170)->shot_bodies_0 = shot_bodies_44;
    (&kernelContext_170)->lines_0 = lines_44;
    (&kernelContext_170)->geometry_0 = geometry_44;
    (&kernelContext_170)->shots_0 = shots_44;
    (&kernelContext_170)->lp_0 = lp_44;
    (&kernelContext_170)->ext_0 = ext_46;
    (&kernelContext_170)->engine_contacts_0 = engine_contacts_44;
    (&kernelContext_170)->crush_0 = crush_44;
    (&kernelContext_170)->crush_rw_0 = crush_rw_44;
    (&kernelContext_170)->crush_tab_0 = crush_tab_44;
    threadgroup array<uint, int(1024)> g_scan_44;
    (&kernelContext_170)->g_scan_0 = &g_scan_44;
    threadgroup array<float, int(64)> g_il_best_44;
    (&kernelContext_170)->g_il_best_0 = &g_il_best_44;
    threadgroup array<float, int(256)> g_gi_x_44;
    (&kernelContext_170)->g_gi_x_0 = &g_gi_x_44;
    threadgroup array<float, int(256)> g_gi_m_44;
    (&kernelContext_170)->g_gi_m_0 = &g_gi_m_44;
    threadgroup array<uint, int(1280)> g_gi_lo_44;
    (&kernelContext_170)->g_gi_lo_0 = &g_gi_lo_44;
    threadgroup array<uint, int(1280)> g_gi_hi_44;
    (&kernelContext_170)->g_gi_hi_0 = &g_gi_hi_44;
    uint _S1259 = id_41.x;
    if(_S1259 == 0U)
    {
        uint _S1260 = a_il_cursor_1(&kernelContext_170);
        atomic_store_explicit((&kernelContext_170)->atom_0+_S1260, 0U, memory_order_relaxed);
    }
    if(_S1259 < ((&kernelContext_170)->params_0->shot_count_0))
    {
        uint _S1261 = a_first_1(_S1259, &kernelContext_170);
        atomic_store_explicit((&kernelContext_170)->atom_0+_S1261, 2139095040U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first(uint3 id_42 [[thread_position_in_grid]], FractureParams_0 constant* params_45 [[buffer(0)]], atomic<uint> device* atom_45 [[buffer(5)]], uint device* work_45 [[buffer(4)]], packed_float4 device* frames_45 [[buffer(6)]], packed_float4 device* bodies_45 [[buffer(7)]], packed_float4 device* chunks_45 [[buffer(1)]], uint device* sort_45 [[buffer(11)]], Record_natural_0 device* manifolds_45 [[buffer(10)]], packed_float4 device* loads_45 [[buffer(12)]], packed_uint4 device* bonds_45 [[buffer(2)]], uint device* args_45 [[buffer(13)]], uint device* info_45 [[buffer(15)]], ScanParams_0 constant* scan_45 [[buffer(14)]], packed_float4 device* parts_45 [[buffer(3)]], packed_float4 device* colliders_45 [[buffer(8)]], packed_uint4 device* collider_info_45 [[buffer(9)]], packed_float4 device* view_chunks_45 [[buffer(17)]], ViewParams_0 constant* view_45 [[buffer(16)]], packed_float4 device* objects_45 [[buffer(18)]], uint device* shot_bodies_45 [[buffer(20)]], float device* lines_45 [[buffer(19)]], packed_float4 device* geometry_45 [[buffer(21)]], packed_float4 device* shots_45 [[buffer(22)]], packed_float4 device* lp_45 [[buffer(24)]], packed_float4 device* ext_47 [[buffer(23)]], packed_float4 device* engine_contacts_45 [[buffer(25)]], CrushParams_0 constant* crush_45 [[buffer(28)]], packed_float4 device* crush_rw_45 [[buffer(27)]], packed_float4 device* crush_tab_45 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_171;
    (&kernelContext_171)->params_0 = params_45;
    (&kernelContext_171)->atom_0 = atom_45;
    (&kernelContext_171)->work_0 = work_45;
    (&kernelContext_171)->frames_0 = frames_45;
    (&kernelContext_171)->bodies_0 = bodies_45;
    (&kernelContext_171)->chunks_0 = chunks_45;
    (&kernelContext_171)->sort_0 = sort_45;
    (&kernelContext_171)->manifolds_0 = manifolds_45;
    (&kernelContext_171)->loads_0 = loads_45;
    (&kernelContext_171)->bonds_0 = bonds_45;
    (&kernelContext_171)->args_0 = args_45;
    (&kernelContext_171)->info_0 = info_45;
    (&kernelContext_171)->scan_0 = scan_45;
    (&kernelContext_171)->parts_0 = parts_45;
    (&kernelContext_171)->colliders_0 = colliders_45;
    (&kernelContext_171)->collider_info_0 = collider_info_45;
    (&kernelContext_171)->view_chunks_0 = view_chunks_45;
    (&kernelContext_171)->view_0 = view_45;
    (&kernelContext_171)->objects_0 = objects_45;
    (&kernelContext_171)->shot_bodies_0 = shot_bodies_45;
    (&kernelContext_171)->lines_0 = lines_45;
    (&kernelContext_171)->geometry_0 = geometry_45;
    (&kernelContext_171)->shots_0 = shots_45;
    (&kernelContext_171)->lp_0 = lp_45;
    (&kernelContext_171)->ext_0 = ext_47;
    (&kernelContext_171)->engine_contacts_0 = engine_contacts_45;
    (&kernelContext_171)->crush_0 = crush_45;
    (&kernelContext_171)->crush_rw_0 = crush_rw_45;
    (&kernelContext_171)->crush_tab_0 = crush_tab_45;
    threadgroup array<uint, int(1024)> g_scan_45;
    (&kernelContext_171)->g_scan_0 = &g_scan_45;
    threadgroup array<float, int(64)> g_il_best_45;
    (&kernelContext_171)->g_il_best_0 = &g_il_best_45;
    threadgroup array<float, int(256)> g_gi_x_45;
    (&kernelContext_171)->g_gi_x_0 = &g_gi_x_45;
    threadgroup array<float, int(256)> g_gi_m_45;
    (&kernelContext_171)->g_gi_m_0 = &g_gi_m_45;
    threadgroup array<uint, int(1280)> g_gi_lo_45;
    (&kernelContext_171)->g_gi_lo_0 = &g_gi_lo_45;
    threadgroup array<uint, int(1280)> g_gi_hi_45;
    (&kernelContext_171)->g_gi_hi_0 = &g_gi_hi_45;
    uint b_52 = id_42.x;
    bool _S1262;
    if(b_52 >= (params_45->body_count_0))
    {
        _S1262 = true;
    }
    else
    {
        _S1262 = ((&kernelContext_171)->params_0->shot_count_0) == 0U;
    }
    if(_S1262)
    {
        return;
    }
    thread uint shot_10;
    bool _S1263 = struck_0(b_52, &shot_10, &kernelContext_171);
    if(!_S1263)
    {
        return;
    }
    uint _S1264 = a_first_1(shot_10, &kernelContext_171);
    atomic<uint> device* _S1265 = (&kernelContext_171)->atom_0+_S1264;
    uint _S1266 = a_strike_0(&kernelContext_171);
    uint _S1267 = atomic_load_explicit((&kernelContext_171)->atom_0+(_S1266 + 9U * b_52 + 6U), memory_order_relaxed);
    uint _S1268 = atomic_fetch_min_explicit(_S1265, _S1267, memory_order_relaxed);
    return;
}

bool cs_struck_1(uint b_53, KernelContext_0 thread* kernelContext_172)
{
    bool _S1269;
    if(b_53 < (kernelContext_172->params_0->body_count_0))
    {
        uint _S1270 = a_cs_0(b_53, kernelContext_172);
        uint _S1271 = atomic_load_explicit(kernelContext_172->atom_0+(_S1270 + 1U), memory_order_relaxed);
        _S1269 = _S1271 != 0U;
    }
    else
    {
        _S1269 = false;
    }
    return _S1269;
}

[[kernel]] void fr_il_begin(uint3 id_43 [[thread_position_in_grid]], FractureParams_0 constant* params_46 [[buffer(0)]], atomic<uint> device* atom_46 [[buffer(5)]], uint device* work_46 [[buffer(4)]], packed_float4 device* frames_46 [[buffer(6)]], packed_float4 device* bodies_46 [[buffer(7)]], packed_float4 device* chunks_46 [[buffer(1)]], uint device* sort_46 [[buffer(11)]], Record_natural_0 device* manifolds_46 [[buffer(10)]], packed_float4 device* loads_46 [[buffer(12)]], packed_uint4 device* bonds_46 [[buffer(2)]], uint device* args_46 [[buffer(13)]], uint device* info_46 [[buffer(15)]], ScanParams_0 constant* scan_46 [[buffer(14)]], packed_float4 device* parts_46 [[buffer(3)]], packed_float4 device* colliders_46 [[buffer(8)]], packed_uint4 device* collider_info_46 [[buffer(9)]], packed_float4 device* view_chunks_46 [[buffer(17)]], ViewParams_0 constant* view_46 [[buffer(16)]], packed_float4 device* objects_46 [[buffer(18)]], uint device* shot_bodies_46 [[buffer(20)]], float device* lines_46 [[buffer(19)]], packed_float4 device* geometry_46 [[buffer(21)]], packed_float4 device* shots_46 [[buffer(22)]], packed_float4 device* lp_46 [[buffer(24)]], packed_float4 device* ext_48 [[buffer(23)]], packed_float4 device* engine_contacts_46 [[buffer(25)]], CrushParams_0 constant* crush_46 [[buffer(28)]], packed_float4 device* crush_rw_46 [[buffer(27)]], packed_float4 device* crush_tab_46 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_173;
    (&kernelContext_173)->params_0 = params_46;
    (&kernelContext_173)->atom_0 = atom_46;
    (&kernelContext_173)->work_0 = work_46;
    (&kernelContext_173)->frames_0 = frames_46;
    (&kernelContext_173)->bodies_0 = bodies_46;
    (&kernelContext_173)->chunks_0 = chunks_46;
    (&kernelContext_173)->sort_0 = sort_46;
    (&kernelContext_173)->manifolds_0 = manifolds_46;
    (&kernelContext_173)->loads_0 = loads_46;
    (&kernelContext_173)->bonds_0 = bonds_46;
    (&kernelContext_173)->args_0 = args_46;
    (&kernelContext_173)->info_0 = info_46;
    (&kernelContext_173)->scan_0 = scan_46;
    (&kernelContext_173)->parts_0 = parts_46;
    (&kernelContext_173)->colliders_0 = colliders_46;
    (&kernelContext_173)->collider_info_0 = collider_info_46;
    (&kernelContext_173)->view_chunks_0 = view_chunks_46;
    (&kernelContext_173)->view_0 = view_46;
    (&kernelContext_173)->objects_0 = objects_46;
    (&kernelContext_173)->shot_bodies_0 = shot_bodies_46;
    (&kernelContext_173)->lines_0 = lines_46;
    (&kernelContext_173)->geometry_0 = geometry_46;
    (&kernelContext_173)->shots_0 = shots_46;
    (&kernelContext_173)->lp_0 = lp_46;
    (&kernelContext_173)->ext_0 = ext_48;
    (&kernelContext_173)->engine_contacts_0 = engine_contacts_46;
    (&kernelContext_173)->crush_0 = crush_46;
    (&kernelContext_173)->crush_rw_0 = crush_rw_46;
    (&kernelContext_173)->crush_tab_0 = crush_tab_46;
    threadgroup array<uint, int(1024)> g_scan_46;
    (&kernelContext_173)->g_scan_0 = &g_scan_46;
    threadgroup array<float, int(64)> g_il_best_46;
    (&kernelContext_173)->g_il_best_0 = &g_il_best_46;
    threadgroup array<float, int(256)> g_gi_x_46;
    (&kernelContext_173)->g_gi_x_0 = &g_gi_x_46;
    threadgroup array<float, int(256)> g_gi_m_46;
    (&kernelContext_173)->g_gi_m_0 = &g_gi_m_46;
    threadgroup array<uint, int(1280)> g_gi_lo_46;
    (&kernelContext_173)->g_gi_lo_0 = &g_gi_lo_46;
    threadgroup array<uint, int(1280)> g_gi_hi_46;
    (&kernelContext_173)->g_gi_hi_0 = &g_gi_hi_46;
    if((params_46->shot_count_0) > 0U)
    {
        il_begin_1(id_43.x, &kernelContext_173);
    }
    return;
}

[[kernel]] void fr_il_tube_count(uint3 id_44 [[thread_position_in_grid]], FractureParams_0 constant* params_47 [[buffer(0)]], atomic<uint> device* atom_47 [[buffer(5)]], uint device* work_47 [[buffer(4)]], packed_float4 device* frames_47 [[buffer(6)]], packed_float4 device* bodies_47 [[buffer(7)]], packed_float4 device* chunks_47 [[buffer(1)]], uint device* sort_47 [[buffer(11)]], Record_natural_0 device* manifolds_47 [[buffer(10)]], packed_float4 device* loads_47 [[buffer(12)]], packed_uint4 device* bonds_47 [[buffer(2)]], uint device* args_47 [[buffer(13)]], uint device* info_47 [[buffer(15)]], ScanParams_0 constant* scan_47 [[buffer(14)]], packed_float4 device* parts_47 [[buffer(3)]], packed_float4 device* colliders_47 [[buffer(8)]], packed_uint4 device* collider_info_47 [[buffer(9)]], packed_float4 device* view_chunks_47 [[buffer(17)]], ViewParams_0 constant* view_47 [[buffer(16)]], packed_float4 device* objects_47 [[buffer(18)]], uint device* shot_bodies_47 [[buffer(20)]], float device* lines_47 [[buffer(19)]], packed_float4 device* geometry_47 [[buffer(21)]], packed_float4 device* shots_47 [[buffer(22)]], packed_float4 device* lp_47 [[buffer(24)]], packed_float4 device* ext_49 [[buffer(23)]], packed_float4 device* engine_contacts_47 [[buffer(25)]], CrushParams_0 constant* crush_47 [[buffer(28)]], packed_float4 device* crush_rw_47 [[buffer(27)]], packed_float4 device* crush_tab_47 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_174;
    (&kernelContext_174)->params_0 = params_47;
    (&kernelContext_174)->atom_0 = atom_47;
    (&kernelContext_174)->work_0 = work_47;
    (&kernelContext_174)->frames_0 = frames_47;
    (&kernelContext_174)->bodies_0 = bodies_47;
    (&kernelContext_174)->chunks_0 = chunks_47;
    (&kernelContext_174)->sort_0 = sort_47;
    (&kernelContext_174)->manifolds_0 = manifolds_47;
    (&kernelContext_174)->loads_0 = loads_47;
    (&kernelContext_174)->bonds_0 = bonds_47;
    (&kernelContext_174)->args_0 = args_47;
    (&kernelContext_174)->info_0 = info_47;
    (&kernelContext_174)->scan_0 = scan_47;
    (&kernelContext_174)->parts_0 = parts_47;
    (&kernelContext_174)->colliders_0 = colliders_47;
    (&kernelContext_174)->collider_info_0 = collider_info_47;
    (&kernelContext_174)->view_chunks_0 = view_chunks_47;
    (&kernelContext_174)->view_0 = view_47;
    (&kernelContext_174)->objects_0 = objects_47;
    (&kernelContext_174)->shot_bodies_0 = shot_bodies_47;
    (&kernelContext_174)->lines_0 = lines_47;
    (&kernelContext_174)->geometry_0 = geometry_47;
    (&kernelContext_174)->shots_0 = shots_47;
    (&kernelContext_174)->lp_0 = lp_47;
    (&kernelContext_174)->ext_0 = ext_49;
    (&kernelContext_174)->engine_contacts_0 = engine_contacts_47;
    (&kernelContext_174)->crush_0 = crush_47;
    (&kernelContext_174)->crush_rw_0 = crush_rw_47;
    (&kernelContext_174)->crush_tab_0 = crush_tab_47;
    threadgroup array<uint, int(1024)> g_scan_47;
    (&kernelContext_174)->g_scan_0 = &g_scan_47;
    threadgroup array<float, int(64)> g_il_best_47;
    (&kernelContext_174)->g_il_best_0 = &g_il_best_47;
    threadgroup array<float, int(256)> g_gi_x_47;
    (&kernelContext_174)->g_gi_x_0 = &g_gi_x_47;
    threadgroup array<float, int(256)> g_gi_m_47;
    (&kernelContext_174)->g_gi_m_0 = &g_gi_m_47;
    threadgroup array<uint, int(1280)> g_gi_lo_47;
    (&kernelContext_174)->g_gi_lo_0 = &g_gi_lo_47;
    threadgroup array<uint, int(1280)> g_gi_hi_47;
    (&kernelContext_174)->g_gi_hi_0 = &g_gi_hi_47;
    if((params_47->shot_count_0) > 0U)
    {
        il_tube_count_1(id_44.x, &kernelContext_174);
    }
    return;
}

[[kernel]] void fr_il_tube_base(uint3 id_45 [[thread_position_in_grid]], FractureParams_0 constant* params_48 [[buffer(0)]], atomic<uint> device* atom_48 [[buffer(5)]], uint device* work_48 [[buffer(4)]], packed_float4 device* frames_48 [[buffer(6)]], packed_float4 device* bodies_48 [[buffer(7)]], packed_float4 device* chunks_48 [[buffer(1)]], uint device* sort_48 [[buffer(11)]], Record_natural_0 device* manifolds_48 [[buffer(10)]], packed_float4 device* loads_48 [[buffer(12)]], packed_uint4 device* bonds_48 [[buffer(2)]], uint device* args_48 [[buffer(13)]], uint device* info_48 [[buffer(15)]], ScanParams_0 constant* scan_48 [[buffer(14)]], packed_float4 device* parts_48 [[buffer(3)]], packed_float4 device* colliders_48 [[buffer(8)]], packed_uint4 device* collider_info_48 [[buffer(9)]], packed_float4 device* view_chunks_48 [[buffer(17)]], ViewParams_0 constant* view_48 [[buffer(16)]], packed_float4 device* objects_48 [[buffer(18)]], uint device* shot_bodies_48 [[buffer(20)]], float device* lines_48 [[buffer(19)]], packed_float4 device* geometry_48 [[buffer(21)]], packed_float4 device* shots_48 [[buffer(22)]], packed_float4 device* lp_48 [[buffer(24)]], packed_float4 device* ext_50 [[buffer(23)]], packed_float4 device* engine_contacts_48 [[buffer(25)]], CrushParams_0 constant* crush_48 [[buffer(28)]], packed_float4 device* crush_rw_48 [[buffer(27)]], packed_float4 device* crush_tab_48 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_175;
    (&kernelContext_175)->params_0 = params_48;
    (&kernelContext_175)->atom_0 = atom_48;
    (&kernelContext_175)->work_0 = work_48;
    (&kernelContext_175)->frames_0 = frames_48;
    (&kernelContext_175)->bodies_0 = bodies_48;
    (&kernelContext_175)->chunks_0 = chunks_48;
    (&kernelContext_175)->sort_0 = sort_48;
    (&kernelContext_175)->manifolds_0 = manifolds_48;
    (&kernelContext_175)->loads_0 = loads_48;
    (&kernelContext_175)->bonds_0 = bonds_48;
    (&kernelContext_175)->args_0 = args_48;
    (&kernelContext_175)->info_0 = info_48;
    (&kernelContext_175)->scan_0 = scan_48;
    (&kernelContext_175)->parts_0 = parts_48;
    (&kernelContext_175)->colliders_0 = colliders_48;
    (&kernelContext_175)->collider_info_0 = collider_info_48;
    (&kernelContext_175)->view_chunks_0 = view_chunks_48;
    (&kernelContext_175)->view_0 = view_48;
    (&kernelContext_175)->objects_0 = objects_48;
    (&kernelContext_175)->shot_bodies_0 = shot_bodies_48;
    (&kernelContext_175)->lines_0 = lines_48;
    (&kernelContext_175)->geometry_0 = geometry_48;
    (&kernelContext_175)->shots_0 = shots_48;
    (&kernelContext_175)->lp_0 = lp_48;
    (&kernelContext_175)->ext_0 = ext_50;
    (&kernelContext_175)->engine_contacts_0 = engine_contacts_48;
    (&kernelContext_175)->crush_0 = crush_48;
    (&kernelContext_175)->crush_rw_0 = crush_rw_48;
    (&kernelContext_175)->crush_tab_0 = crush_tab_48;
    threadgroup array<uint, int(1024)> g_scan_48;
    (&kernelContext_175)->g_scan_0 = &g_scan_48;
    threadgroup array<float, int(64)> g_il_best_48;
    (&kernelContext_175)->g_il_best_0 = &g_il_best_48;
    threadgroup array<float, int(256)> g_gi_x_48;
    (&kernelContext_175)->g_gi_x_0 = &g_gi_x_48;
    threadgroup array<float, int(256)> g_gi_m_48;
    (&kernelContext_175)->g_gi_m_0 = &g_gi_m_48;
    threadgroup array<uint, int(1280)> g_gi_lo_48;
    (&kernelContext_175)->g_gi_lo_0 = &g_gi_lo_48;
    threadgroup array<uint, int(1280)> g_gi_hi_48;
    (&kernelContext_175)->g_gi_hi_0 = &g_gi_hi_48;
    if((params_48->shot_count_0) > 0U)
    {
        il_tube_base_1(id_45.x, &kernelContext_175);
    }
    return;
}

[[kernel]] void fr_il_tube_collect(uint3 id_46 [[thread_position_in_grid]], FractureParams_0 constant* params_49 [[buffer(0)]], atomic<uint> device* atom_49 [[buffer(5)]], uint device* work_49 [[buffer(4)]], packed_float4 device* frames_49 [[buffer(6)]], packed_float4 device* bodies_49 [[buffer(7)]], packed_float4 device* chunks_49 [[buffer(1)]], uint device* sort_49 [[buffer(11)]], Record_natural_0 device* manifolds_49 [[buffer(10)]], packed_float4 device* loads_49 [[buffer(12)]], packed_uint4 device* bonds_49 [[buffer(2)]], uint device* args_49 [[buffer(13)]], uint device* info_49 [[buffer(15)]], ScanParams_0 constant* scan_49 [[buffer(14)]], packed_float4 device* parts_49 [[buffer(3)]], packed_float4 device* colliders_49 [[buffer(8)]], packed_uint4 device* collider_info_49 [[buffer(9)]], packed_float4 device* view_chunks_49 [[buffer(17)]], ViewParams_0 constant* view_49 [[buffer(16)]], packed_float4 device* objects_49 [[buffer(18)]], uint device* shot_bodies_49 [[buffer(20)]], float device* lines_49 [[buffer(19)]], packed_float4 device* geometry_49 [[buffer(21)]], packed_float4 device* shots_49 [[buffer(22)]], packed_float4 device* lp_49 [[buffer(24)]], packed_float4 device* ext_51 [[buffer(23)]], packed_float4 device* engine_contacts_49 [[buffer(25)]], CrushParams_0 constant* crush_49 [[buffer(28)]], packed_float4 device* crush_rw_49 [[buffer(27)]], packed_float4 device* crush_tab_49 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_176;
    (&kernelContext_176)->params_0 = params_49;
    (&kernelContext_176)->atom_0 = atom_49;
    (&kernelContext_176)->work_0 = work_49;
    (&kernelContext_176)->frames_0 = frames_49;
    (&kernelContext_176)->bodies_0 = bodies_49;
    (&kernelContext_176)->chunks_0 = chunks_49;
    (&kernelContext_176)->sort_0 = sort_49;
    (&kernelContext_176)->manifolds_0 = manifolds_49;
    (&kernelContext_176)->loads_0 = loads_49;
    (&kernelContext_176)->bonds_0 = bonds_49;
    (&kernelContext_176)->args_0 = args_49;
    (&kernelContext_176)->info_0 = info_49;
    (&kernelContext_176)->scan_0 = scan_49;
    (&kernelContext_176)->parts_0 = parts_49;
    (&kernelContext_176)->colliders_0 = colliders_49;
    (&kernelContext_176)->collider_info_0 = collider_info_49;
    (&kernelContext_176)->view_chunks_0 = view_chunks_49;
    (&kernelContext_176)->view_0 = view_49;
    (&kernelContext_176)->objects_0 = objects_49;
    (&kernelContext_176)->shot_bodies_0 = shot_bodies_49;
    (&kernelContext_176)->lines_0 = lines_49;
    (&kernelContext_176)->geometry_0 = geometry_49;
    (&kernelContext_176)->shots_0 = shots_49;
    (&kernelContext_176)->lp_0 = lp_49;
    (&kernelContext_176)->ext_0 = ext_51;
    (&kernelContext_176)->engine_contacts_0 = engine_contacts_49;
    (&kernelContext_176)->crush_0 = crush_49;
    (&kernelContext_176)->crush_rw_0 = crush_rw_49;
    (&kernelContext_176)->crush_tab_0 = crush_tab_49;
    threadgroup array<uint, int(1024)> g_scan_49;
    (&kernelContext_176)->g_scan_0 = &g_scan_49;
    threadgroup array<float, int(64)> g_il_best_49;
    (&kernelContext_176)->g_il_best_0 = &g_il_best_49;
    threadgroup array<float, int(256)> g_gi_x_49;
    (&kernelContext_176)->g_gi_x_0 = &g_gi_x_49;
    threadgroup array<float, int(256)> g_gi_m_49;
    (&kernelContext_176)->g_gi_m_0 = &g_gi_m_49;
    threadgroup array<uint, int(1280)> g_gi_lo_49;
    (&kernelContext_176)->g_gi_lo_0 = &g_gi_lo_49;
    threadgroup array<uint, int(1280)> g_gi_hi_49;
    (&kernelContext_176)->g_gi_hi_0 = &g_gi_hi_49;
    if((params_49->shot_count_0) > 0U)
    {
        il_tube_collect_1(id_46.x, &kernelContext_176);
    }
    return;
}

[[kernel]] void fr_il_run(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_50 [[buffer(0)]], atomic<uint> device* atom_50 [[buffer(5)]], uint device* work_50 [[buffer(4)]], packed_float4 device* frames_50 [[buffer(6)]], packed_float4 device* bodies_50 [[buffer(7)]], packed_float4 device* chunks_50 [[buffer(1)]], uint device* sort_50 [[buffer(11)]], Record_natural_0 device* manifolds_50 [[buffer(10)]], packed_float4 device* loads_50 [[buffer(12)]], packed_uint4 device* bonds_50 [[buffer(2)]], uint device* args_50 [[buffer(13)]], uint device* info_50 [[buffer(15)]], ScanParams_0 constant* scan_50 [[buffer(14)]], packed_float4 device* parts_50 [[buffer(3)]], packed_float4 device* colliders_50 [[buffer(8)]], packed_uint4 device* collider_info_50 [[buffer(9)]], packed_float4 device* view_chunks_50 [[buffer(17)]], ViewParams_0 constant* view_50 [[buffer(16)]], packed_float4 device* objects_50 [[buffer(18)]], uint device* shot_bodies_50 [[buffer(20)]], float device* lines_50 [[buffer(19)]], packed_float4 device* geometry_50 [[buffer(21)]], packed_float4 device* shots_50 [[buffer(22)]], packed_float4 device* lp_50 [[buffer(24)]], packed_float4 device* ext_52 [[buffer(23)]], packed_float4 device* engine_contacts_50 [[buffer(25)]], CrushParams_0 constant* crush_50 [[buffer(28)]], packed_float4 device* crush_rw_50 [[buffer(27)]], packed_float4 device* crush_tab_50 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_177;
    (&kernelContext_177)->params_0 = params_50;
    (&kernelContext_177)->atom_0 = atom_50;
    (&kernelContext_177)->work_0 = work_50;
    (&kernelContext_177)->frames_0 = frames_50;
    (&kernelContext_177)->bodies_0 = bodies_50;
    (&kernelContext_177)->chunks_0 = chunks_50;
    (&kernelContext_177)->sort_0 = sort_50;
    (&kernelContext_177)->manifolds_0 = manifolds_50;
    (&kernelContext_177)->loads_0 = loads_50;
    (&kernelContext_177)->bonds_0 = bonds_50;
    (&kernelContext_177)->args_0 = args_50;
    (&kernelContext_177)->info_0 = info_50;
    (&kernelContext_177)->scan_0 = scan_50;
    (&kernelContext_177)->parts_0 = parts_50;
    (&kernelContext_177)->colliders_0 = colliders_50;
    (&kernelContext_177)->collider_info_0 = collider_info_50;
    (&kernelContext_177)->view_chunks_0 = view_chunks_50;
    (&kernelContext_177)->view_0 = view_50;
    (&kernelContext_177)->objects_0 = objects_50;
    (&kernelContext_177)->shot_bodies_0 = shot_bodies_50;
    (&kernelContext_177)->lines_0 = lines_50;
    (&kernelContext_177)->geometry_0 = geometry_50;
    (&kernelContext_177)->shots_0 = shots_50;
    (&kernelContext_177)->lp_0 = lp_50;
    (&kernelContext_177)->ext_0 = ext_52;
    (&kernelContext_177)->engine_contacts_0 = engine_contacts_50;
    (&kernelContext_177)->crush_0 = crush_50;
    (&kernelContext_177)->crush_rw_0 = crush_rw_50;
    (&kernelContext_177)->crush_tab_0 = crush_tab_50;
    threadgroup array<uint, int(1024)> g_scan_50;
    (&kernelContext_177)->g_scan_0 = &g_scan_50;
    threadgroup array<float, int(64)> g_il_best_50;
    (&kernelContext_177)->g_il_best_0 = &g_il_best_50;
    threadgroup array<float, int(256)> g_gi_x_50;
    (&kernelContext_177)->g_gi_x_0 = &g_gi_x_50;
    threadgroup array<float, int(256)> g_gi_m_50;
    (&kernelContext_177)->g_gi_m_0 = &g_gi_m_50;
    threadgroup array<uint, int(1280)> g_gi_lo_50;
    (&kernelContext_177)->g_gi_lo_0 = &g_gi_lo_50;
    threadgroup array<uint, int(1280)> g_gi_hi_50;
    (&kernelContext_177)->g_gi_hi_0 = &g_gi_hi_50;
    uint b_54 = group_1.y * 65535U + group_1.x;
    bool _S1272;
    if((params_50->shot_count_0) > 0U)
    {
        _S1272 = b_54 < ((&kernelContext_177)->params_0->body_count_0);
    }
    else
    {
        _S1272 = false;
    }
    if(_S1272)
    {
        il_run_1(b_54, thread_4.x, &kernelContext_177);
    }
    return;
}

[[kernel]] void fr_il_strike(uint3 id_47 [[thread_position_in_grid]], FractureParams_0 constant* params_51 [[buffer(0)]], atomic<uint> device* atom_51 [[buffer(5)]], uint device* work_51 [[buffer(4)]], packed_float4 device* frames_51 [[buffer(6)]], packed_float4 device* bodies_51 [[buffer(7)]], packed_float4 device* chunks_51 [[buffer(1)]], uint device* sort_51 [[buffer(11)]], Record_natural_0 device* manifolds_51 [[buffer(10)]], packed_float4 device* loads_51 [[buffer(12)]], packed_uint4 device* bonds_51 [[buffer(2)]], uint device* args_51 [[buffer(13)]], uint device* info_51 [[buffer(15)]], ScanParams_0 constant* scan_51 [[buffer(14)]], packed_float4 device* parts_51 [[buffer(3)]], packed_float4 device* colliders_51 [[buffer(8)]], packed_uint4 device* collider_info_51 [[buffer(9)]], packed_float4 device* view_chunks_51 [[buffer(17)]], ViewParams_0 constant* view_51 [[buffer(16)]], packed_float4 device* objects_51 [[buffer(18)]], uint device* shot_bodies_51 [[buffer(20)]], float device* lines_51 [[buffer(19)]], packed_float4 device* geometry_51 [[buffer(21)]], packed_float4 device* shots_51 [[buffer(22)]], packed_float4 device* lp_51 [[buffer(24)]], packed_float4 device* ext_53 [[buffer(23)]], packed_float4 device* engine_contacts_51 [[buffer(25)]], CrushParams_0 constant* crush_51 [[buffer(28)]], packed_float4 device* crush_rw_51 [[buffer(27)]], packed_float4 device* crush_tab_51 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_178;
    (&kernelContext_178)->params_0 = params_51;
    (&kernelContext_178)->atom_0 = atom_51;
    (&kernelContext_178)->work_0 = work_51;
    (&kernelContext_178)->frames_0 = frames_51;
    (&kernelContext_178)->bodies_0 = bodies_51;
    (&kernelContext_178)->chunks_0 = chunks_51;
    (&kernelContext_178)->sort_0 = sort_51;
    (&kernelContext_178)->manifolds_0 = manifolds_51;
    (&kernelContext_178)->loads_0 = loads_51;
    (&kernelContext_178)->bonds_0 = bonds_51;
    (&kernelContext_178)->args_0 = args_51;
    (&kernelContext_178)->info_0 = info_51;
    (&kernelContext_178)->scan_0 = scan_51;
    (&kernelContext_178)->parts_0 = parts_51;
    (&kernelContext_178)->colliders_0 = colliders_51;
    (&kernelContext_178)->collider_info_0 = collider_info_51;
    (&kernelContext_178)->view_chunks_0 = view_chunks_51;
    (&kernelContext_178)->view_0 = view_51;
    (&kernelContext_178)->objects_0 = objects_51;
    (&kernelContext_178)->shot_bodies_0 = shot_bodies_51;
    (&kernelContext_178)->lines_0 = lines_51;
    (&kernelContext_178)->geometry_0 = geometry_51;
    (&kernelContext_178)->shots_0 = shots_51;
    (&kernelContext_178)->lp_0 = lp_51;
    (&kernelContext_178)->ext_0 = ext_53;
    (&kernelContext_178)->engine_contacts_0 = engine_contacts_51;
    (&kernelContext_178)->crush_0 = crush_51;
    (&kernelContext_178)->crush_rw_0 = crush_rw_51;
    (&kernelContext_178)->crush_tab_0 = crush_tab_51;
    threadgroup array<uint, int(1024)> g_scan_51;
    (&kernelContext_178)->g_scan_0 = &g_scan_51;
    threadgroup array<float, int(64)> g_il_best_51;
    (&kernelContext_178)->g_il_best_0 = &g_il_best_51;
    threadgroup array<float, int(256)> g_gi_x_51;
    (&kernelContext_178)->g_gi_x_0 = &g_gi_x_51;
    threadgroup array<float, int(256)> g_gi_m_51;
    (&kernelContext_178)->g_gi_m_0 = &g_gi_m_51;
    threadgroup array<uint, int(1280)> g_gi_lo_51;
    (&kernelContext_178)->g_gi_lo_0 = &g_gi_lo_51;
    threadgroup array<uint, int(1280)> g_gi_hi_51;
    (&kernelContext_178)->g_gi_hi_0 = &g_gi_hi_51;
    if((params_51->shot_count_0) > 0U)
    {
        il_strike_1(id_47.x, &kernelContext_178);
    }
    return;
}

[[kernel]] void fr_il_mark(uint3 id_48 [[thread_position_in_grid]], FractureParams_0 constant* params_52 [[buffer(0)]], atomic<uint> device* atom_52 [[buffer(5)]], uint device* work_52 [[buffer(4)]], packed_float4 device* frames_52 [[buffer(6)]], packed_float4 device* bodies_52 [[buffer(7)]], packed_float4 device* chunks_52 [[buffer(1)]], uint device* sort_52 [[buffer(11)]], Record_natural_0 device* manifolds_52 [[buffer(10)]], packed_float4 device* loads_52 [[buffer(12)]], packed_uint4 device* bonds_52 [[buffer(2)]], uint device* args_52 [[buffer(13)]], uint device* info_52 [[buffer(15)]], ScanParams_0 constant* scan_52 [[buffer(14)]], packed_float4 device* parts_52 [[buffer(3)]], packed_float4 device* colliders_52 [[buffer(8)]], packed_uint4 device* collider_info_52 [[buffer(9)]], packed_float4 device* view_chunks_52 [[buffer(17)]], ViewParams_0 constant* view_52 [[buffer(16)]], packed_float4 device* objects_52 [[buffer(18)]], uint device* shot_bodies_52 [[buffer(20)]], float device* lines_52 [[buffer(19)]], packed_float4 device* geometry_52 [[buffer(21)]], packed_float4 device* shots_52 [[buffer(22)]], packed_float4 device* lp_52 [[buffer(24)]], packed_float4 device* ext_54 [[buffer(23)]], packed_float4 device* engine_contacts_52 [[buffer(25)]], CrushParams_0 constant* crush_52 [[buffer(28)]], packed_float4 device* crush_rw_52 [[buffer(27)]], packed_float4 device* crush_tab_52 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_179;
    (&kernelContext_179)->params_0 = params_52;
    (&kernelContext_179)->atom_0 = atom_52;
    (&kernelContext_179)->work_0 = work_52;
    (&kernelContext_179)->frames_0 = frames_52;
    (&kernelContext_179)->bodies_0 = bodies_52;
    (&kernelContext_179)->chunks_0 = chunks_52;
    (&kernelContext_179)->sort_0 = sort_52;
    (&kernelContext_179)->manifolds_0 = manifolds_52;
    (&kernelContext_179)->loads_0 = loads_52;
    (&kernelContext_179)->bonds_0 = bonds_52;
    (&kernelContext_179)->args_0 = args_52;
    (&kernelContext_179)->info_0 = info_52;
    (&kernelContext_179)->scan_0 = scan_52;
    (&kernelContext_179)->parts_0 = parts_52;
    (&kernelContext_179)->colliders_0 = colliders_52;
    (&kernelContext_179)->collider_info_0 = collider_info_52;
    (&kernelContext_179)->view_chunks_0 = view_chunks_52;
    (&kernelContext_179)->view_0 = view_52;
    (&kernelContext_179)->objects_0 = objects_52;
    (&kernelContext_179)->shot_bodies_0 = shot_bodies_52;
    (&kernelContext_179)->lines_0 = lines_52;
    (&kernelContext_179)->geometry_0 = geometry_52;
    (&kernelContext_179)->shots_0 = shots_52;
    (&kernelContext_179)->lp_0 = lp_52;
    (&kernelContext_179)->ext_0 = ext_54;
    (&kernelContext_179)->engine_contacts_0 = engine_contacts_52;
    (&kernelContext_179)->crush_0 = crush_52;
    (&kernelContext_179)->crush_rw_0 = crush_rw_52;
    (&kernelContext_179)->crush_tab_0 = crush_tab_52;
    threadgroup array<uint, int(1024)> g_scan_52;
    (&kernelContext_179)->g_scan_0 = &g_scan_52;
    threadgroup array<float, int(64)> g_il_best_52;
    (&kernelContext_179)->g_il_best_0 = &g_il_best_52;
    threadgroup array<float, int(256)> g_gi_x_52;
    (&kernelContext_179)->g_gi_x_0 = &g_gi_x_52;
    threadgroup array<float, int(256)> g_gi_m_52;
    (&kernelContext_179)->g_gi_m_0 = &g_gi_m_52;
    threadgroup array<uint, int(1280)> g_gi_lo_52;
    (&kernelContext_179)->g_gi_lo_0 = &g_gi_lo_52;
    threadgroup array<uint, int(1280)> g_gi_hi_52;
    (&kernelContext_179)->g_gi_hi_0 = &g_gi_hi_52;
    if((params_52->shot_count_0) > 0U)
    {
        il_mark_1(id_48.x, &kernelContext_179);
    }
    return;
}

[[kernel]] void fr_il_carry(uint3 id_49 [[thread_position_in_grid]], FractureParams_0 constant* params_53 [[buffer(0)]], atomic<uint> device* atom_53 [[buffer(5)]], uint device* work_53 [[buffer(4)]], packed_float4 device* frames_53 [[buffer(6)]], packed_float4 device* bodies_53 [[buffer(7)]], packed_float4 device* chunks_53 [[buffer(1)]], uint device* sort_53 [[buffer(11)]], Record_natural_0 device* manifolds_53 [[buffer(10)]], packed_float4 device* loads_53 [[buffer(12)]], packed_uint4 device* bonds_53 [[buffer(2)]], uint device* args_53 [[buffer(13)]], uint device* info_53 [[buffer(15)]], ScanParams_0 constant* scan_53 [[buffer(14)]], packed_float4 device* parts_53 [[buffer(3)]], packed_float4 device* colliders_53 [[buffer(8)]], packed_uint4 device* collider_info_53 [[buffer(9)]], packed_float4 device* view_chunks_53 [[buffer(17)]], ViewParams_0 constant* view_53 [[buffer(16)]], packed_float4 device* objects_53 [[buffer(18)]], uint device* shot_bodies_53 [[buffer(20)]], float device* lines_53 [[buffer(19)]], packed_float4 device* geometry_53 [[buffer(21)]], packed_float4 device* shots_53 [[buffer(22)]], packed_float4 device* lp_53 [[buffer(24)]], packed_float4 device* ext_55 [[buffer(23)]], packed_float4 device* engine_contacts_53 [[buffer(25)]], CrushParams_0 constant* crush_53 [[buffer(28)]], packed_float4 device* crush_rw_53 [[buffer(27)]], packed_float4 device* crush_tab_53 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_180;
    (&kernelContext_180)->params_0 = params_53;
    (&kernelContext_180)->atom_0 = atom_53;
    (&kernelContext_180)->work_0 = work_53;
    (&kernelContext_180)->frames_0 = frames_53;
    (&kernelContext_180)->bodies_0 = bodies_53;
    (&kernelContext_180)->chunks_0 = chunks_53;
    (&kernelContext_180)->sort_0 = sort_53;
    (&kernelContext_180)->manifolds_0 = manifolds_53;
    (&kernelContext_180)->loads_0 = loads_53;
    (&kernelContext_180)->bonds_0 = bonds_53;
    (&kernelContext_180)->args_0 = args_53;
    (&kernelContext_180)->info_0 = info_53;
    (&kernelContext_180)->scan_0 = scan_53;
    (&kernelContext_180)->parts_0 = parts_53;
    (&kernelContext_180)->colliders_0 = colliders_53;
    (&kernelContext_180)->collider_info_0 = collider_info_53;
    (&kernelContext_180)->view_chunks_0 = view_chunks_53;
    (&kernelContext_180)->view_0 = view_53;
    (&kernelContext_180)->objects_0 = objects_53;
    (&kernelContext_180)->shot_bodies_0 = shot_bodies_53;
    (&kernelContext_180)->lines_0 = lines_53;
    (&kernelContext_180)->geometry_0 = geometry_53;
    (&kernelContext_180)->shots_0 = shots_53;
    (&kernelContext_180)->lp_0 = lp_53;
    (&kernelContext_180)->ext_0 = ext_55;
    (&kernelContext_180)->engine_contacts_0 = engine_contacts_53;
    (&kernelContext_180)->crush_0 = crush_53;
    (&kernelContext_180)->crush_rw_0 = crush_rw_53;
    (&kernelContext_180)->crush_tab_0 = crush_tab_53;
    threadgroup array<uint, int(1024)> g_scan_53;
    (&kernelContext_180)->g_scan_0 = &g_scan_53;
    threadgroup array<float, int(64)> g_il_best_53;
    (&kernelContext_180)->g_il_best_0 = &g_il_best_53;
    threadgroup array<float, int(256)> g_gi_x_53;
    (&kernelContext_180)->g_gi_x_0 = &g_gi_x_53;
    threadgroup array<float, int(256)> g_gi_m_53;
    (&kernelContext_180)->g_gi_m_0 = &g_gi_m_53;
    threadgroup array<uint, int(1280)> g_gi_lo_53;
    (&kernelContext_180)->g_gi_lo_0 = &g_gi_lo_53;
    threadgroup array<uint, int(1280)> g_gi_hi_53;
    (&kernelContext_180)->g_gi_hi_0 = &g_gi_hi_53;
    if((params_53->shot_count_0) > 0U)
    {
        il_carry_1(id_49.x, &kernelContext_180);
    }
    return;
}

[[kernel]] void fr_il_breaks(uint3 id_50 [[thread_position_in_grid]], FractureParams_0 constant* params_54 [[buffer(0)]], atomic<uint> device* atom_54 [[buffer(5)]], uint device* work_54 [[buffer(4)]], packed_float4 device* frames_54 [[buffer(6)]], packed_float4 device* bodies_54 [[buffer(7)]], packed_float4 device* chunks_54 [[buffer(1)]], uint device* sort_54 [[buffer(11)]], Record_natural_0 device* manifolds_54 [[buffer(10)]], packed_float4 device* loads_54 [[buffer(12)]], packed_uint4 device* bonds_54 [[buffer(2)]], uint device* args_54 [[buffer(13)]], uint device* info_54 [[buffer(15)]], ScanParams_0 constant* scan_54 [[buffer(14)]], packed_float4 device* parts_54 [[buffer(3)]], packed_float4 device* colliders_54 [[buffer(8)]], packed_uint4 device* collider_info_54 [[buffer(9)]], packed_float4 device* view_chunks_54 [[buffer(17)]], ViewParams_0 constant* view_54 [[buffer(16)]], packed_float4 device* objects_54 [[buffer(18)]], uint device* shot_bodies_54 [[buffer(20)]], float device* lines_54 [[buffer(19)]], packed_float4 device* geometry_54 [[buffer(21)]], packed_float4 device* shots_54 [[buffer(22)]], packed_float4 device* lp_54 [[buffer(24)]], packed_float4 device* ext_56 [[buffer(23)]], packed_float4 device* engine_contacts_54 [[buffer(25)]], CrushParams_0 constant* crush_54 [[buffer(28)]], packed_float4 device* crush_rw_54 [[buffer(27)]], packed_float4 device* crush_tab_54 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_181;
    (&kernelContext_181)->params_0 = params_54;
    (&kernelContext_181)->atom_0 = atom_54;
    (&kernelContext_181)->work_0 = work_54;
    (&kernelContext_181)->frames_0 = frames_54;
    (&kernelContext_181)->bodies_0 = bodies_54;
    (&kernelContext_181)->chunks_0 = chunks_54;
    (&kernelContext_181)->sort_0 = sort_54;
    (&kernelContext_181)->manifolds_0 = manifolds_54;
    (&kernelContext_181)->loads_0 = loads_54;
    (&kernelContext_181)->bonds_0 = bonds_54;
    (&kernelContext_181)->args_0 = args_54;
    (&kernelContext_181)->info_0 = info_54;
    (&kernelContext_181)->scan_0 = scan_54;
    (&kernelContext_181)->parts_0 = parts_54;
    (&kernelContext_181)->colliders_0 = colliders_54;
    (&kernelContext_181)->collider_info_0 = collider_info_54;
    (&kernelContext_181)->view_chunks_0 = view_chunks_54;
    (&kernelContext_181)->view_0 = view_54;
    (&kernelContext_181)->objects_0 = objects_54;
    (&kernelContext_181)->shot_bodies_0 = shot_bodies_54;
    (&kernelContext_181)->lines_0 = lines_54;
    (&kernelContext_181)->geometry_0 = geometry_54;
    (&kernelContext_181)->shots_0 = shots_54;
    (&kernelContext_181)->lp_0 = lp_54;
    (&kernelContext_181)->ext_0 = ext_56;
    (&kernelContext_181)->engine_contacts_0 = engine_contacts_54;
    (&kernelContext_181)->crush_0 = crush_54;
    (&kernelContext_181)->crush_rw_0 = crush_rw_54;
    (&kernelContext_181)->crush_tab_0 = crush_tab_54;
    threadgroup array<uint, int(1024)> g_scan_54;
    (&kernelContext_181)->g_scan_0 = &g_scan_54;
    threadgroup array<float, int(64)> g_il_best_54;
    (&kernelContext_181)->g_il_best_0 = &g_il_best_54;
    threadgroup array<float, int(256)> g_gi_x_54;
    (&kernelContext_181)->g_gi_x_0 = &g_gi_x_54;
    threadgroup array<float, int(256)> g_gi_m_54;
    (&kernelContext_181)->g_gi_m_0 = &g_gi_m_54;
    threadgroup array<uint, int(1280)> g_gi_lo_54;
    (&kernelContext_181)->g_gi_lo_0 = &g_gi_lo_54;
    threadgroup array<uint, int(1280)> g_gi_hi_54;
    (&kernelContext_181)->g_gi_hi_0 = &g_gi_hi_54;
    if((params_54->shot_count_0) > 0U)
    {
        il_breaks_1(id_50.x, &kernelContext_181);
    }
    return;
}

[[kernel]] void fr_csl_begin(uint3 id_51 [[thread_position_in_grid]], FractureParams_0 constant* params_55 [[buffer(0)]], atomic<uint> device* atom_55 [[buffer(5)]], uint device* work_55 [[buffer(4)]], packed_float4 device* frames_55 [[buffer(6)]], packed_float4 device* bodies_55 [[buffer(7)]], packed_float4 device* chunks_55 [[buffer(1)]], uint device* sort_55 [[buffer(11)]], Record_natural_0 device* manifolds_55 [[buffer(10)]], packed_float4 device* loads_55 [[buffer(12)]], packed_uint4 device* bonds_55 [[buffer(2)]], uint device* args_55 [[buffer(13)]], uint device* info_55 [[buffer(15)]], ScanParams_0 constant* scan_55 [[buffer(14)]], packed_float4 device* parts_55 [[buffer(3)]], packed_float4 device* colliders_55 [[buffer(8)]], packed_uint4 device* collider_info_55 [[buffer(9)]], packed_float4 device* view_chunks_55 [[buffer(17)]], ViewParams_0 constant* view_55 [[buffer(16)]], packed_float4 device* objects_55 [[buffer(18)]], uint device* shot_bodies_55 [[buffer(20)]], float device* lines_55 [[buffer(19)]], packed_float4 device* geometry_55 [[buffer(21)]], packed_float4 device* shots_55 [[buffer(22)]], packed_float4 device* lp_55 [[buffer(24)]], packed_float4 device* ext_57 [[buffer(23)]], packed_float4 device* engine_contacts_55 [[buffer(25)]], CrushParams_0 constant* crush_55 [[buffer(28)]], packed_float4 device* crush_rw_55 [[buffer(27)]], packed_float4 device* crush_tab_55 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_182;
    (&kernelContext_182)->params_0 = params_55;
    (&kernelContext_182)->atom_0 = atom_55;
    (&kernelContext_182)->work_0 = work_55;
    (&kernelContext_182)->frames_0 = frames_55;
    (&kernelContext_182)->bodies_0 = bodies_55;
    (&kernelContext_182)->chunks_0 = chunks_55;
    (&kernelContext_182)->sort_0 = sort_55;
    (&kernelContext_182)->manifolds_0 = manifolds_55;
    (&kernelContext_182)->loads_0 = loads_55;
    (&kernelContext_182)->bonds_0 = bonds_55;
    (&kernelContext_182)->args_0 = args_55;
    (&kernelContext_182)->info_0 = info_55;
    (&kernelContext_182)->scan_0 = scan_55;
    (&kernelContext_182)->parts_0 = parts_55;
    (&kernelContext_182)->colliders_0 = colliders_55;
    (&kernelContext_182)->collider_info_0 = collider_info_55;
    (&kernelContext_182)->view_chunks_0 = view_chunks_55;
    (&kernelContext_182)->view_0 = view_55;
    (&kernelContext_182)->objects_0 = objects_55;
    (&kernelContext_182)->shot_bodies_0 = shot_bodies_55;
    (&kernelContext_182)->lines_0 = lines_55;
    (&kernelContext_182)->geometry_0 = geometry_55;
    (&kernelContext_182)->shots_0 = shots_55;
    (&kernelContext_182)->lp_0 = lp_55;
    (&kernelContext_182)->ext_0 = ext_57;
    (&kernelContext_182)->engine_contacts_0 = engine_contacts_55;
    (&kernelContext_182)->crush_0 = crush_55;
    (&kernelContext_182)->crush_rw_0 = crush_rw_55;
    (&kernelContext_182)->crush_tab_0 = crush_tab_55;
    threadgroup array<uint, int(1024)> g_scan_55;
    (&kernelContext_182)->g_scan_0 = &g_scan_55;
    threadgroup array<float, int(64)> g_il_best_55;
    (&kernelContext_182)->g_il_best_0 = &g_il_best_55;
    threadgroup array<float, int(256)> g_gi_x_55;
    (&kernelContext_182)->g_gi_x_0 = &g_gi_x_55;
    threadgroup array<float, int(256)> g_gi_m_55;
    (&kernelContext_182)->g_gi_m_0 = &g_gi_m_55;
    threadgroup array<uint, int(1280)> g_gi_lo_55;
    (&kernelContext_182)->g_gi_lo_0 = &g_gi_lo_55;
    threadgroup array<uint, int(1280)> g_gi_hi_55;
    (&kernelContext_182)->g_gi_hi_0 = &g_gi_hi_55;
    il_begin_0(id_51.x, &kernelContext_182);
    return;
}

[[kernel]] void fr_csl_tube_count(uint3 id_52 [[thread_position_in_grid]], FractureParams_0 constant* params_56 [[buffer(0)]], atomic<uint> device* atom_56 [[buffer(5)]], uint device* work_56 [[buffer(4)]], packed_float4 device* frames_56 [[buffer(6)]], packed_float4 device* bodies_56 [[buffer(7)]], packed_float4 device* chunks_56 [[buffer(1)]], uint device* sort_56 [[buffer(11)]], Record_natural_0 device* manifolds_56 [[buffer(10)]], packed_float4 device* loads_56 [[buffer(12)]], packed_uint4 device* bonds_56 [[buffer(2)]], uint device* args_56 [[buffer(13)]], uint device* info_56 [[buffer(15)]], ScanParams_0 constant* scan_56 [[buffer(14)]], packed_float4 device* parts_56 [[buffer(3)]], packed_float4 device* colliders_56 [[buffer(8)]], packed_uint4 device* collider_info_56 [[buffer(9)]], packed_float4 device* view_chunks_56 [[buffer(17)]], ViewParams_0 constant* view_56 [[buffer(16)]], packed_float4 device* objects_56 [[buffer(18)]], uint device* shot_bodies_56 [[buffer(20)]], float device* lines_56 [[buffer(19)]], packed_float4 device* geometry_56 [[buffer(21)]], packed_float4 device* shots_56 [[buffer(22)]], packed_float4 device* lp_56 [[buffer(24)]], packed_float4 device* ext_58 [[buffer(23)]], packed_float4 device* engine_contacts_56 [[buffer(25)]], CrushParams_0 constant* crush_56 [[buffer(28)]], packed_float4 device* crush_rw_56 [[buffer(27)]], packed_float4 device* crush_tab_56 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_183;
    (&kernelContext_183)->params_0 = params_56;
    (&kernelContext_183)->atom_0 = atom_56;
    (&kernelContext_183)->work_0 = work_56;
    (&kernelContext_183)->frames_0 = frames_56;
    (&kernelContext_183)->bodies_0 = bodies_56;
    (&kernelContext_183)->chunks_0 = chunks_56;
    (&kernelContext_183)->sort_0 = sort_56;
    (&kernelContext_183)->manifolds_0 = manifolds_56;
    (&kernelContext_183)->loads_0 = loads_56;
    (&kernelContext_183)->bonds_0 = bonds_56;
    (&kernelContext_183)->args_0 = args_56;
    (&kernelContext_183)->info_0 = info_56;
    (&kernelContext_183)->scan_0 = scan_56;
    (&kernelContext_183)->parts_0 = parts_56;
    (&kernelContext_183)->colliders_0 = colliders_56;
    (&kernelContext_183)->collider_info_0 = collider_info_56;
    (&kernelContext_183)->view_chunks_0 = view_chunks_56;
    (&kernelContext_183)->view_0 = view_56;
    (&kernelContext_183)->objects_0 = objects_56;
    (&kernelContext_183)->shot_bodies_0 = shot_bodies_56;
    (&kernelContext_183)->lines_0 = lines_56;
    (&kernelContext_183)->geometry_0 = geometry_56;
    (&kernelContext_183)->shots_0 = shots_56;
    (&kernelContext_183)->lp_0 = lp_56;
    (&kernelContext_183)->ext_0 = ext_58;
    (&kernelContext_183)->engine_contacts_0 = engine_contacts_56;
    (&kernelContext_183)->crush_0 = crush_56;
    (&kernelContext_183)->crush_rw_0 = crush_rw_56;
    (&kernelContext_183)->crush_tab_0 = crush_tab_56;
    threadgroup array<uint, int(1024)> g_scan_56;
    (&kernelContext_183)->g_scan_0 = &g_scan_56;
    threadgroup array<float, int(64)> g_il_best_56;
    (&kernelContext_183)->g_il_best_0 = &g_il_best_56;
    threadgroup array<float, int(256)> g_gi_x_56;
    (&kernelContext_183)->g_gi_x_0 = &g_gi_x_56;
    threadgroup array<float, int(256)> g_gi_m_56;
    (&kernelContext_183)->g_gi_m_0 = &g_gi_m_56;
    threadgroup array<uint, int(1280)> g_gi_lo_56;
    (&kernelContext_183)->g_gi_lo_0 = &g_gi_lo_56;
    threadgroup array<uint, int(1280)> g_gi_hi_56;
    (&kernelContext_183)->g_gi_hi_0 = &g_gi_hi_56;
    il_tube_count_0(id_52.x, &kernelContext_183);
    return;
}

[[kernel]] void fr_csl_tube_base(uint3 id_53 [[thread_position_in_grid]], FractureParams_0 constant* params_57 [[buffer(0)]], atomic<uint> device* atom_57 [[buffer(5)]], uint device* work_57 [[buffer(4)]], packed_float4 device* frames_57 [[buffer(6)]], packed_float4 device* bodies_57 [[buffer(7)]], packed_float4 device* chunks_57 [[buffer(1)]], uint device* sort_57 [[buffer(11)]], Record_natural_0 device* manifolds_57 [[buffer(10)]], packed_float4 device* loads_57 [[buffer(12)]], packed_uint4 device* bonds_57 [[buffer(2)]], uint device* args_57 [[buffer(13)]], uint device* info_57 [[buffer(15)]], ScanParams_0 constant* scan_57 [[buffer(14)]], packed_float4 device* parts_57 [[buffer(3)]], packed_float4 device* colliders_57 [[buffer(8)]], packed_uint4 device* collider_info_57 [[buffer(9)]], packed_float4 device* view_chunks_57 [[buffer(17)]], ViewParams_0 constant* view_57 [[buffer(16)]], packed_float4 device* objects_57 [[buffer(18)]], uint device* shot_bodies_57 [[buffer(20)]], float device* lines_57 [[buffer(19)]], packed_float4 device* geometry_57 [[buffer(21)]], packed_float4 device* shots_57 [[buffer(22)]], packed_float4 device* lp_57 [[buffer(24)]], packed_float4 device* ext_59 [[buffer(23)]], packed_float4 device* engine_contacts_57 [[buffer(25)]], CrushParams_0 constant* crush_57 [[buffer(28)]], packed_float4 device* crush_rw_57 [[buffer(27)]], packed_float4 device* crush_tab_57 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_184;
    (&kernelContext_184)->params_0 = params_57;
    (&kernelContext_184)->atom_0 = atom_57;
    (&kernelContext_184)->work_0 = work_57;
    (&kernelContext_184)->frames_0 = frames_57;
    (&kernelContext_184)->bodies_0 = bodies_57;
    (&kernelContext_184)->chunks_0 = chunks_57;
    (&kernelContext_184)->sort_0 = sort_57;
    (&kernelContext_184)->manifolds_0 = manifolds_57;
    (&kernelContext_184)->loads_0 = loads_57;
    (&kernelContext_184)->bonds_0 = bonds_57;
    (&kernelContext_184)->args_0 = args_57;
    (&kernelContext_184)->info_0 = info_57;
    (&kernelContext_184)->scan_0 = scan_57;
    (&kernelContext_184)->parts_0 = parts_57;
    (&kernelContext_184)->colliders_0 = colliders_57;
    (&kernelContext_184)->collider_info_0 = collider_info_57;
    (&kernelContext_184)->view_chunks_0 = view_chunks_57;
    (&kernelContext_184)->view_0 = view_57;
    (&kernelContext_184)->objects_0 = objects_57;
    (&kernelContext_184)->shot_bodies_0 = shot_bodies_57;
    (&kernelContext_184)->lines_0 = lines_57;
    (&kernelContext_184)->geometry_0 = geometry_57;
    (&kernelContext_184)->shots_0 = shots_57;
    (&kernelContext_184)->lp_0 = lp_57;
    (&kernelContext_184)->ext_0 = ext_59;
    (&kernelContext_184)->engine_contacts_0 = engine_contacts_57;
    (&kernelContext_184)->crush_0 = crush_57;
    (&kernelContext_184)->crush_rw_0 = crush_rw_57;
    (&kernelContext_184)->crush_tab_0 = crush_tab_57;
    threadgroup array<uint, int(1024)> g_scan_57;
    (&kernelContext_184)->g_scan_0 = &g_scan_57;
    threadgroup array<float, int(64)> g_il_best_57;
    (&kernelContext_184)->g_il_best_0 = &g_il_best_57;
    threadgroup array<float, int(256)> g_gi_x_57;
    (&kernelContext_184)->g_gi_x_0 = &g_gi_x_57;
    threadgroup array<float, int(256)> g_gi_m_57;
    (&kernelContext_184)->g_gi_m_0 = &g_gi_m_57;
    threadgroup array<uint, int(1280)> g_gi_lo_57;
    (&kernelContext_184)->g_gi_lo_0 = &g_gi_lo_57;
    threadgroup array<uint, int(1280)> g_gi_hi_57;
    (&kernelContext_184)->g_gi_hi_0 = &g_gi_hi_57;
    il_tube_base_0(id_53.x, &kernelContext_184);
    return;
}

[[kernel]] void fr_csl_tube_collect(uint3 id_54 [[thread_position_in_grid]], FractureParams_0 constant* params_58 [[buffer(0)]], atomic<uint> device* atom_58 [[buffer(5)]], uint device* work_58 [[buffer(4)]], packed_float4 device* frames_58 [[buffer(6)]], packed_float4 device* bodies_58 [[buffer(7)]], packed_float4 device* chunks_58 [[buffer(1)]], uint device* sort_58 [[buffer(11)]], Record_natural_0 device* manifolds_58 [[buffer(10)]], packed_float4 device* loads_58 [[buffer(12)]], packed_uint4 device* bonds_58 [[buffer(2)]], uint device* args_58 [[buffer(13)]], uint device* info_58 [[buffer(15)]], ScanParams_0 constant* scan_58 [[buffer(14)]], packed_float4 device* parts_58 [[buffer(3)]], packed_float4 device* colliders_58 [[buffer(8)]], packed_uint4 device* collider_info_58 [[buffer(9)]], packed_float4 device* view_chunks_58 [[buffer(17)]], ViewParams_0 constant* view_58 [[buffer(16)]], packed_float4 device* objects_58 [[buffer(18)]], uint device* shot_bodies_58 [[buffer(20)]], float device* lines_58 [[buffer(19)]], packed_float4 device* geometry_58 [[buffer(21)]], packed_float4 device* shots_58 [[buffer(22)]], packed_float4 device* lp_58 [[buffer(24)]], packed_float4 device* ext_60 [[buffer(23)]], packed_float4 device* engine_contacts_58 [[buffer(25)]], CrushParams_0 constant* crush_58 [[buffer(28)]], packed_float4 device* crush_rw_58 [[buffer(27)]], packed_float4 device* crush_tab_58 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_185;
    (&kernelContext_185)->params_0 = params_58;
    (&kernelContext_185)->atom_0 = atom_58;
    (&kernelContext_185)->work_0 = work_58;
    (&kernelContext_185)->frames_0 = frames_58;
    (&kernelContext_185)->bodies_0 = bodies_58;
    (&kernelContext_185)->chunks_0 = chunks_58;
    (&kernelContext_185)->sort_0 = sort_58;
    (&kernelContext_185)->manifolds_0 = manifolds_58;
    (&kernelContext_185)->loads_0 = loads_58;
    (&kernelContext_185)->bonds_0 = bonds_58;
    (&kernelContext_185)->args_0 = args_58;
    (&kernelContext_185)->info_0 = info_58;
    (&kernelContext_185)->scan_0 = scan_58;
    (&kernelContext_185)->parts_0 = parts_58;
    (&kernelContext_185)->colliders_0 = colliders_58;
    (&kernelContext_185)->collider_info_0 = collider_info_58;
    (&kernelContext_185)->view_chunks_0 = view_chunks_58;
    (&kernelContext_185)->view_0 = view_58;
    (&kernelContext_185)->objects_0 = objects_58;
    (&kernelContext_185)->shot_bodies_0 = shot_bodies_58;
    (&kernelContext_185)->lines_0 = lines_58;
    (&kernelContext_185)->geometry_0 = geometry_58;
    (&kernelContext_185)->shots_0 = shots_58;
    (&kernelContext_185)->lp_0 = lp_58;
    (&kernelContext_185)->ext_0 = ext_60;
    (&kernelContext_185)->engine_contacts_0 = engine_contacts_58;
    (&kernelContext_185)->crush_0 = crush_58;
    (&kernelContext_185)->crush_rw_0 = crush_rw_58;
    (&kernelContext_185)->crush_tab_0 = crush_tab_58;
    threadgroup array<uint, int(1024)> g_scan_58;
    (&kernelContext_185)->g_scan_0 = &g_scan_58;
    threadgroup array<float, int(64)> g_il_best_58;
    (&kernelContext_185)->g_il_best_0 = &g_il_best_58;
    threadgroup array<float, int(256)> g_gi_x_58;
    (&kernelContext_185)->g_gi_x_0 = &g_gi_x_58;
    threadgroup array<float, int(256)> g_gi_m_58;
    (&kernelContext_185)->g_gi_m_0 = &g_gi_m_58;
    threadgroup array<uint, int(1280)> g_gi_lo_58;
    (&kernelContext_185)->g_gi_lo_0 = &g_gi_lo_58;
    threadgroup array<uint, int(1280)> g_gi_hi_58;
    (&kernelContext_185)->g_gi_hi_0 = &g_gi_hi_58;
    il_tube_collect_0(id_54.x, &kernelContext_185);
    return;
}

[[kernel]] void fr_csl_run(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_59 [[buffer(0)]], atomic<uint> device* atom_59 [[buffer(5)]], uint device* work_59 [[buffer(4)]], packed_float4 device* frames_59 [[buffer(6)]], packed_float4 device* bodies_59 [[buffer(7)]], packed_float4 device* chunks_59 [[buffer(1)]], uint device* sort_59 [[buffer(11)]], Record_natural_0 device* manifolds_59 [[buffer(10)]], packed_float4 device* loads_59 [[buffer(12)]], packed_uint4 device* bonds_59 [[buffer(2)]], uint device* args_59 [[buffer(13)]], uint device* info_59 [[buffer(15)]], ScanParams_0 constant* scan_59 [[buffer(14)]], packed_float4 device* parts_59 [[buffer(3)]], packed_float4 device* colliders_59 [[buffer(8)]], packed_uint4 device* collider_info_59 [[buffer(9)]], packed_float4 device* view_chunks_59 [[buffer(17)]], ViewParams_0 constant* view_59 [[buffer(16)]], packed_float4 device* objects_59 [[buffer(18)]], uint device* shot_bodies_59 [[buffer(20)]], float device* lines_59 [[buffer(19)]], packed_float4 device* geometry_59 [[buffer(21)]], packed_float4 device* shots_59 [[buffer(22)]], packed_float4 device* lp_59 [[buffer(24)]], packed_float4 device* ext_61 [[buffer(23)]], packed_float4 device* engine_contacts_59 [[buffer(25)]], CrushParams_0 constant* crush_59 [[buffer(28)]], packed_float4 device* crush_rw_59 [[buffer(27)]], packed_float4 device* crush_tab_59 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_186;
    (&kernelContext_186)->params_0 = params_59;
    (&kernelContext_186)->atom_0 = atom_59;
    (&kernelContext_186)->work_0 = work_59;
    (&kernelContext_186)->frames_0 = frames_59;
    (&kernelContext_186)->bodies_0 = bodies_59;
    (&kernelContext_186)->chunks_0 = chunks_59;
    (&kernelContext_186)->sort_0 = sort_59;
    (&kernelContext_186)->manifolds_0 = manifolds_59;
    (&kernelContext_186)->loads_0 = loads_59;
    (&kernelContext_186)->bonds_0 = bonds_59;
    (&kernelContext_186)->args_0 = args_59;
    (&kernelContext_186)->info_0 = info_59;
    (&kernelContext_186)->scan_0 = scan_59;
    (&kernelContext_186)->parts_0 = parts_59;
    (&kernelContext_186)->colliders_0 = colliders_59;
    (&kernelContext_186)->collider_info_0 = collider_info_59;
    (&kernelContext_186)->view_chunks_0 = view_chunks_59;
    (&kernelContext_186)->view_0 = view_59;
    (&kernelContext_186)->objects_0 = objects_59;
    (&kernelContext_186)->shot_bodies_0 = shot_bodies_59;
    (&kernelContext_186)->lines_0 = lines_59;
    (&kernelContext_186)->geometry_0 = geometry_59;
    (&kernelContext_186)->shots_0 = shots_59;
    (&kernelContext_186)->lp_0 = lp_59;
    (&kernelContext_186)->ext_0 = ext_61;
    (&kernelContext_186)->engine_contacts_0 = engine_contacts_59;
    (&kernelContext_186)->crush_0 = crush_59;
    (&kernelContext_186)->crush_rw_0 = crush_rw_59;
    (&kernelContext_186)->crush_tab_0 = crush_tab_59;
    threadgroup array<uint, int(1024)> g_scan_59;
    (&kernelContext_186)->g_scan_0 = &g_scan_59;
    threadgroup array<float, int(64)> g_il_best_59;
    (&kernelContext_186)->g_il_best_0 = &g_il_best_59;
    threadgroup array<float, int(256)> g_gi_x_59;
    (&kernelContext_186)->g_gi_x_0 = &g_gi_x_59;
    threadgroup array<float, int(256)> g_gi_m_59;
    (&kernelContext_186)->g_gi_m_0 = &g_gi_m_59;
    threadgroup array<uint, int(1280)> g_gi_lo_59;
    (&kernelContext_186)->g_gi_lo_0 = &g_gi_lo_59;
    threadgroup array<uint, int(1280)> g_gi_hi_59;
    (&kernelContext_186)->g_gi_hi_0 = &g_gi_hi_59;
    uint _S1273 = w_cs_bodies_0(&kernelContext_186);
    il_run_0(*((&kernelContext_186)->work_0+(_S1273 + group_2.x)), thread_5.x, &kernelContext_186);
    return;
}

[[kernel]] void fr_csl_strike(uint3 id_55 [[thread_position_in_grid]], FractureParams_0 constant* params_60 [[buffer(0)]], atomic<uint> device* atom_60 [[buffer(5)]], uint device* work_60 [[buffer(4)]], packed_float4 device* frames_60 [[buffer(6)]], packed_float4 device* bodies_60 [[buffer(7)]], packed_float4 device* chunks_60 [[buffer(1)]], uint device* sort_60 [[buffer(11)]], Record_natural_0 device* manifolds_60 [[buffer(10)]], packed_float4 device* loads_60 [[buffer(12)]], packed_uint4 device* bonds_60 [[buffer(2)]], uint device* args_60 [[buffer(13)]], uint device* info_60 [[buffer(15)]], ScanParams_0 constant* scan_60 [[buffer(14)]], packed_float4 device* parts_60 [[buffer(3)]], packed_float4 device* colliders_60 [[buffer(8)]], packed_uint4 device* collider_info_60 [[buffer(9)]], packed_float4 device* view_chunks_60 [[buffer(17)]], ViewParams_0 constant* view_60 [[buffer(16)]], packed_float4 device* objects_60 [[buffer(18)]], uint device* shot_bodies_60 [[buffer(20)]], float device* lines_60 [[buffer(19)]], packed_float4 device* geometry_60 [[buffer(21)]], packed_float4 device* shots_60 [[buffer(22)]], packed_float4 device* lp_60 [[buffer(24)]], packed_float4 device* ext_62 [[buffer(23)]], packed_float4 device* engine_contacts_60 [[buffer(25)]], CrushParams_0 constant* crush_60 [[buffer(28)]], packed_float4 device* crush_rw_60 [[buffer(27)]], packed_float4 device* crush_tab_60 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_187;
    (&kernelContext_187)->params_0 = params_60;
    (&kernelContext_187)->atom_0 = atom_60;
    (&kernelContext_187)->work_0 = work_60;
    (&kernelContext_187)->frames_0 = frames_60;
    (&kernelContext_187)->bodies_0 = bodies_60;
    (&kernelContext_187)->chunks_0 = chunks_60;
    (&kernelContext_187)->sort_0 = sort_60;
    (&kernelContext_187)->manifolds_0 = manifolds_60;
    (&kernelContext_187)->loads_0 = loads_60;
    (&kernelContext_187)->bonds_0 = bonds_60;
    (&kernelContext_187)->args_0 = args_60;
    (&kernelContext_187)->info_0 = info_60;
    (&kernelContext_187)->scan_0 = scan_60;
    (&kernelContext_187)->parts_0 = parts_60;
    (&kernelContext_187)->colliders_0 = colliders_60;
    (&kernelContext_187)->collider_info_0 = collider_info_60;
    (&kernelContext_187)->view_chunks_0 = view_chunks_60;
    (&kernelContext_187)->view_0 = view_60;
    (&kernelContext_187)->objects_0 = objects_60;
    (&kernelContext_187)->shot_bodies_0 = shot_bodies_60;
    (&kernelContext_187)->lines_0 = lines_60;
    (&kernelContext_187)->geometry_0 = geometry_60;
    (&kernelContext_187)->shots_0 = shots_60;
    (&kernelContext_187)->lp_0 = lp_60;
    (&kernelContext_187)->ext_0 = ext_62;
    (&kernelContext_187)->engine_contacts_0 = engine_contacts_60;
    (&kernelContext_187)->crush_0 = crush_60;
    (&kernelContext_187)->crush_rw_0 = crush_rw_60;
    (&kernelContext_187)->crush_tab_0 = crush_tab_60;
    threadgroup array<uint, int(1024)> g_scan_60;
    (&kernelContext_187)->g_scan_0 = &g_scan_60;
    threadgroup array<float, int(64)> g_il_best_60;
    (&kernelContext_187)->g_il_best_0 = &g_il_best_60;
    threadgroup array<float, int(256)> g_gi_x_60;
    (&kernelContext_187)->g_gi_x_0 = &g_gi_x_60;
    threadgroup array<float, int(256)> g_gi_m_60;
    (&kernelContext_187)->g_gi_m_0 = &g_gi_m_60;
    threadgroup array<uint, int(1280)> g_gi_lo_60;
    (&kernelContext_187)->g_gi_lo_0 = &g_gi_lo_60;
    threadgroup array<uint, int(1280)> g_gi_hi_60;
    (&kernelContext_187)->g_gi_hi_0 = &g_gi_hi_60;
    il_strike_0(id_55.x, &kernelContext_187);
    return;
}

[[kernel]] void fr_csl_mark(uint3 id_56 [[thread_position_in_grid]], FractureParams_0 constant* params_61 [[buffer(0)]], atomic<uint> device* atom_61 [[buffer(5)]], uint device* work_61 [[buffer(4)]], packed_float4 device* frames_61 [[buffer(6)]], packed_float4 device* bodies_61 [[buffer(7)]], packed_float4 device* chunks_61 [[buffer(1)]], uint device* sort_61 [[buffer(11)]], Record_natural_0 device* manifolds_61 [[buffer(10)]], packed_float4 device* loads_61 [[buffer(12)]], packed_uint4 device* bonds_61 [[buffer(2)]], uint device* args_61 [[buffer(13)]], uint device* info_61 [[buffer(15)]], ScanParams_0 constant* scan_61 [[buffer(14)]], packed_float4 device* parts_61 [[buffer(3)]], packed_float4 device* colliders_61 [[buffer(8)]], packed_uint4 device* collider_info_61 [[buffer(9)]], packed_float4 device* view_chunks_61 [[buffer(17)]], ViewParams_0 constant* view_61 [[buffer(16)]], packed_float4 device* objects_61 [[buffer(18)]], uint device* shot_bodies_61 [[buffer(20)]], float device* lines_61 [[buffer(19)]], packed_float4 device* geometry_61 [[buffer(21)]], packed_float4 device* shots_61 [[buffer(22)]], packed_float4 device* lp_61 [[buffer(24)]], packed_float4 device* ext_63 [[buffer(23)]], packed_float4 device* engine_contacts_61 [[buffer(25)]], CrushParams_0 constant* crush_61 [[buffer(28)]], packed_float4 device* crush_rw_61 [[buffer(27)]], packed_float4 device* crush_tab_61 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_188;
    (&kernelContext_188)->params_0 = params_61;
    (&kernelContext_188)->atom_0 = atom_61;
    (&kernelContext_188)->work_0 = work_61;
    (&kernelContext_188)->frames_0 = frames_61;
    (&kernelContext_188)->bodies_0 = bodies_61;
    (&kernelContext_188)->chunks_0 = chunks_61;
    (&kernelContext_188)->sort_0 = sort_61;
    (&kernelContext_188)->manifolds_0 = manifolds_61;
    (&kernelContext_188)->loads_0 = loads_61;
    (&kernelContext_188)->bonds_0 = bonds_61;
    (&kernelContext_188)->args_0 = args_61;
    (&kernelContext_188)->info_0 = info_61;
    (&kernelContext_188)->scan_0 = scan_61;
    (&kernelContext_188)->parts_0 = parts_61;
    (&kernelContext_188)->colliders_0 = colliders_61;
    (&kernelContext_188)->collider_info_0 = collider_info_61;
    (&kernelContext_188)->view_chunks_0 = view_chunks_61;
    (&kernelContext_188)->view_0 = view_61;
    (&kernelContext_188)->objects_0 = objects_61;
    (&kernelContext_188)->shot_bodies_0 = shot_bodies_61;
    (&kernelContext_188)->lines_0 = lines_61;
    (&kernelContext_188)->geometry_0 = geometry_61;
    (&kernelContext_188)->shots_0 = shots_61;
    (&kernelContext_188)->lp_0 = lp_61;
    (&kernelContext_188)->ext_0 = ext_63;
    (&kernelContext_188)->engine_contacts_0 = engine_contacts_61;
    (&kernelContext_188)->crush_0 = crush_61;
    (&kernelContext_188)->crush_rw_0 = crush_rw_61;
    (&kernelContext_188)->crush_tab_0 = crush_tab_61;
    threadgroup array<uint, int(1024)> g_scan_61;
    (&kernelContext_188)->g_scan_0 = &g_scan_61;
    threadgroup array<float, int(64)> g_il_best_61;
    (&kernelContext_188)->g_il_best_0 = &g_il_best_61;
    threadgroup array<float, int(256)> g_gi_x_61;
    (&kernelContext_188)->g_gi_x_0 = &g_gi_x_61;
    threadgroup array<float, int(256)> g_gi_m_61;
    (&kernelContext_188)->g_gi_m_0 = &g_gi_m_61;
    threadgroup array<uint, int(1280)> g_gi_lo_61;
    (&kernelContext_188)->g_gi_lo_0 = &g_gi_lo_61;
    threadgroup array<uint, int(1280)> g_gi_hi_61;
    (&kernelContext_188)->g_gi_hi_0 = &g_gi_hi_61;
    il_mark_0(id_56.x, &kernelContext_188);
    return;
}

[[kernel]] void fr_csl_carry(uint3 id_57 [[thread_position_in_grid]], FractureParams_0 constant* params_62 [[buffer(0)]], atomic<uint> device* atom_62 [[buffer(5)]], uint device* work_62 [[buffer(4)]], packed_float4 device* frames_62 [[buffer(6)]], packed_float4 device* bodies_62 [[buffer(7)]], packed_float4 device* chunks_62 [[buffer(1)]], uint device* sort_62 [[buffer(11)]], Record_natural_0 device* manifolds_62 [[buffer(10)]], packed_float4 device* loads_62 [[buffer(12)]], packed_uint4 device* bonds_62 [[buffer(2)]], uint device* args_62 [[buffer(13)]], uint device* info_62 [[buffer(15)]], ScanParams_0 constant* scan_62 [[buffer(14)]], packed_float4 device* parts_62 [[buffer(3)]], packed_float4 device* colliders_62 [[buffer(8)]], packed_uint4 device* collider_info_62 [[buffer(9)]], packed_float4 device* view_chunks_62 [[buffer(17)]], ViewParams_0 constant* view_62 [[buffer(16)]], packed_float4 device* objects_62 [[buffer(18)]], uint device* shot_bodies_62 [[buffer(20)]], float device* lines_62 [[buffer(19)]], packed_float4 device* geometry_62 [[buffer(21)]], packed_float4 device* shots_62 [[buffer(22)]], packed_float4 device* lp_62 [[buffer(24)]], packed_float4 device* ext_64 [[buffer(23)]], packed_float4 device* engine_contacts_62 [[buffer(25)]], CrushParams_0 constant* crush_62 [[buffer(28)]], packed_float4 device* crush_rw_62 [[buffer(27)]], packed_float4 device* crush_tab_62 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_189;
    (&kernelContext_189)->params_0 = params_62;
    (&kernelContext_189)->atom_0 = atom_62;
    (&kernelContext_189)->work_0 = work_62;
    (&kernelContext_189)->frames_0 = frames_62;
    (&kernelContext_189)->bodies_0 = bodies_62;
    (&kernelContext_189)->chunks_0 = chunks_62;
    (&kernelContext_189)->sort_0 = sort_62;
    (&kernelContext_189)->manifolds_0 = manifolds_62;
    (&kernelContext_189)->loads_0 = loads_62;
    (&kernelContext_189)->bonds_0 = bonds_62;
    (&kernelContext_189)->args_0 = args_62;
    (&kernelContext_189)->info_0 = info_62;
    (&kernelContext_189)->scan_0 = scan_62;
    (&kernelContext_189)->parts_0 = parts_62;
    (&kernelContext_189)->colliders_0 = colliders_62;
    (&kernelContext_189)->collider_info_0 = collider_info_62;
    (&kernelContext_189)->view_chunks_0 = view_chunks_62;
    (&kernelContext_189)->view_0 = view_62;
    (&kernelContext_189)->objects_0 = objects_62;
    (&kernelContext_189)->shot_bodies_0 = shot_bodies_62;
    (&kernelContext_189)->lines_0 = lines_62;
    (&kernelContext_189)->geometry_0 = geometry_62;
    (&kernelContext_189)->shots_0 = shots_62;
    (&kernelContext_189)->lp_0 = lp_62;
    (&kernelContext_189)->ext_0 = ext_64;
    (&kernelContext_189)->engine_contacts_0 = engine_contacts_62;
    (&kernelContext_189)->crush_0 = crush_62;
    (&kernelContext_189)->crush_rw_0 = crush_rw_62;
    (&kernelContext_189)->crush_tab_0 = crush_tab_62;
    threadgroup array<uint, int(1024)> g_scan_62;
    (&kernelContext_189)->g_scan_0 = &g_scan_62;
    threadgroup array<float, int(64)> g_il_best_62;
    (&kernelContext_189)->g_il_best_0 = &g_il_best_62;
    threadgroup array<float, int(256)> g_gi_x_62;
    (&kernelContext_189)->g_gi_x_0 = &g_gi_x_62;
    threadgroup array<float, int(256)> g_gi_m_62;
    (&kernelContext_189)->g_gi_m_0 = &g_gi_m_62;
    threadgroup array<uint, int(1280)> g_gi_lo_62;
    (&kernelContext_189)->g_gi_lo_0 = &g_gi_lo_62;
    threadgroup array<uint, int(1280)> g_gi_hi_62;
    (&kernelContext_189)->g_gi_hi_0 = &g_gi_hi_62;
    il_carry_0(id_57.x, &kernelContext_189);
    return;
}

[[kernel]] void fr_csl_breaks(uint3 id_58 [[thread_position_in_grid]], FractureParams_0 constant* params_63 [[buffer(0)]], atomic<uint> device* atom_63 [[buffer(5)]], uint device* work_63 [[buffer(4)]], packed_float4 device* frames_63 [[buffer(6)]], packed_float4 device* bodies_63 [[buffer(7)]], packed_float4 device* chunks_63 [[buffer(1)]], uint device* sort_63 [[buffer(11)]], Record_natural_0 device* manifolds_63 [[buffer(10)]], packed_float4 device* loads_63 [[buffer(12)]], packed_uint4 device* bonds_63 [[buffer(2)]], uint device* args_63 [[buffer(13)]], uint device* info_63 [[buffer(15)]], ScanParams_0 constant* scan_63 [[buffer(14)]], packed_float4 device* parts_63 [[buffer(3)]], packed_float4 device* colliders_63 [[buffer(8)]], packed_uint4 device* collider_info_63 [[buffer(9)]], packed_float4 device* view_chunks_63 [[buffer(17)]], ViewParams_0 constant* view_63 [[buffer(16)]], packed_float4 device* objects_63 [[buffer(18)]], uint device* shot_bodies_63 [[buffer(20)]], float device* lines_63 [[buffer(19)]], packed_float4 device* geometry_63 [[buffer(21)]], packed_float4 device* shots_63 [[buffer(22)]], packed_float4 device* lp_63 [[buffer(24)]], packed_float4 device* ext_65 [[buffer(23)]], packed_float4 device* engine_contacts_63 [[buffer(25)]], CrushParams_0 constant* crush_63 [[buffer(28)]], packed_float4 device* crush_rw_63 [[buffer(27)]], packed_float4 device* crush_tab_63 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_190;
    (&kernelContext_190)->params_0 = params_63;
    (&kernelContext_190)->atom_0 = atom_63;
    (&kernelContext_190)->work_0 = work_63;
    (&kernelContext_190)->frames_0 = frames_63;
    (&kernelContext_190)->bodies_0 = bodies_63;
    (&kernelContext_190)->chunks_0 = chunks_63;
    (&kernelContext_190)->sort_0 = sort_63;
    (&kernelContext_190)->manifolds_0 = manifolds_63;
    (&kernelContext_190)->loads_0 = loads_63;
    (&kernelContext_190)->bonds_0 = bonds_63;
    (&kernelContext_190)->args_0 = args_63;
    (&kernelContext_190)->info_0 = info_63;
    (&kernelContext_190)->scan_0 = scan_63;
    (&kernelContext_190)->parts_0 = parts_63;
    (&kernelContext_190)->colliders_0 = colliders_63;
    (&kernelContext_190)->collider_info_0 = collider_info_63;
    (&kernelContext_190)->view_chunks_0 = view_chunks_63;
    (&kernelContext_190)->view_0 = view_63;
    (&kernelContext_190)->objects_0 = objects_63;
    (&kernelContext_190)->shot_bodies_0 = shot_bodies_63;
    (&kernelContext_190)->lines_0 = lines_63;
    (&kernelContext_190)->geometry_0 = geometry_63;
    (&kernelContext_190)->shots_0 = shots_63;
    (&kernelContext_190)->lp_0 = lp_63;
    (&kernelContext_190)->ext_0 = ext_65;
    (&kernelContext_190)->engine_contacts_0 = engine_contacts_63;
    (&kernelContext_190)->crush_0 = crush_63;
    (&kernelContext_190)->crush_rw_0 = crush_rw_63;
    (&kernelContext_190)->crush_tab_0 = crush_tab_63;
    threadgroup array<uint, int(1024)> g_scan_63;
    (&kernelContext_190)->g_scan_0 = &g_scan_63;
    threadgroup array<float, int(64)> g_il_best_63;
    (&kernelContext_190)->g_il_best_0 = &g_il_best_63;
    threadgroup array<float, int(256)> g_gi_x_63;
    (&kernelContext_190)->g_gi_x_0 = &g_gi_x_63;
    threadgroup array<float, int(256)> g_gi_m_63;
    (&kernelContext_190)->g_gi_m_0 = &g_gi_m_63;
    threadgroup array<uint, int(1280)> g_gi_lo_63;
    (&kernelContext_190)->g_gi_lo_0 = &g_gi_lo_63;
    threadgroup array<uint, int(1280)> g_gi_hi_63;
    (&kernelContext_190)->g_gi_hi_0 = &g_gi_hi_63;
    il_breaks_0(id_58.x, &kernelContext_190);
    return;
}

[[kernel]] void fr_cs_begin(uint3 id_59 [[thread_position_in_grid]], FractureParams_0 constant* params_64 [[buffer(0)]], atomic<uint> device* atom_64 [[buffer(5)]], uint device* work_64 [[buffer(4)]], packed_float4 device* frames_64 [[buffer(6)]], packed_float4 device* bodies_64 [[buffer(7)]], packed_float4 device* chunks_64 [[buffer(1)]], uint device* sort_64 [[buffer(11)]], Record_natural_0 device* manifolds_64 [[buffer(10)]], packed_float4 device* loads_64 [[buffer(12)]], packed_uint4 device* bonds_64 [[buffer(2)]], uint device* args_64 [[buffer(13)]], uint device* info_64 [[buffer(15)]], ScanParams_0 constant* scan_64 [[buffer(14)]], packed_float4 device* parts_64 [[buffer(3)]], packed_float4 device* colliders_64 [[buffer(8)]], packed_uint4 device* collider_info_64 [[buffer(9)]], packed_float4 device* view_chunks_64 [[buffer(17)]], ViewParams_0 constant* view_64 [[buffer(16)]], packed_float4 device* objects_64 [[buffer(18)]], uint device* shot_bodies_64 [[buffer(20)]], float device* lines_64 [[buffer(19)]], packed_float4 device* geometry_64 [[buffer(21)]], packed_float4 device* shots_64 [[buffer(22)]], packed_float4 device* lp_64 [[buffer(24)]], packed_float4 device* ext_66 [[buffer(23)]], packed_float4 device* engine_contacts_64 [[buffer(25)]], CrushParams_0 constant* crush_64 [[buffer(28)]], packed_float4 device* crush_rw_64 [[buffer(27)]], packed_float4 device* crush_tab_64 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_191;
    (&kernelContext_191)->params_0 = params_64;
    (&kernelContext_191)->atom_0 = atom_64;
    (&kernelContext_191)->work_0 = work_64;
    (&kernelContext_191)->frames_0 = frames_64;
    (&kernelContext_191)->bodies_0 = bodies_64;
    (&kernelContext_191)->chunks_0 = chunks_64;
    (&kernelContext_191)->sort_0 = sort_64;
    (&kernelContext_191)->manifolds_0 = manifolds_64;
    (&kernelContext_191)->loads_0 = loads_64;
    (&kernelContext_191)->bonds_0 = bonds_64;
    (&kernelContext_191)->args_0 = args_64;
    (&kernelContext_191)->info_0 = info_64;
    (&kernelContext_191)->scan_0 = scan_64;
    (&kernelContext_191)->parts_0 = parts_64;
    (&kernelContext_191)->colliders_0 = colliders_64;
    (&kernelContext_191)->collider_info_0 = collider_info_64;
    (&kernelContext_191)->view_chunks_0 = view_chunks_64;
    (&kernelContext_191)->view_0 = view_64;
    (&kernelContext_191)->objects_0 = objects_64;
    (&kernelContext_191)->shot_bodies_0 = shot_bodies_64;
    (&kernelContext_191)->lines_0 = lines_64;
    (&kernelContext_191)->geometry_0 = geometry_64;
    (&kernelContext_191)->shots_0 = shots_64;
    (&kernelContext_191)->lp_0 = lp_64;
    (&kernelContext_191)->ext_0 = ext_66;
    (&kernelContext_191)->engine_contacts_0 = engine_contacts_64;
    (&kernelContext_191)->crush_0 = crush_64;
    (&kernelContext_191)->crush_rw_0 = crush_rw_64;
    (&kernelContext_191)->crush_tab_0 = crush_tab_64;
    threadgroup array<uint, int(1024)> g_scan_64;
    (&kernelContext_191)->g_scan_0 = &g_scan_64;
    threadgroup array<float, int(64)> g_il_best_64;
    (&kernelContext_191)->g_il_best_0 = &g_il_best_64;
    threadgroup array<float, int(256)> g_gi_x_64;
    (&kernelContext_191)->g_gi_x_0 = &g_gi_x_64;
    threadgroup array<float, int(256)> g_gi_m_64;
    (&kernelContext_191)->g_gi_m_0 = &g_gi_m_64;
    threadgroup array<uint, int(1280)> g_gi_lo_64;
    (&kernelContext_191)->g_gi_lo_0 = &g_gi_lo_64;
    threadgroup array<uint, int(1280)> g_gi_hi_64;
    (&kernelContext_191)->g_gi_hi_0 = &g_gi_hi_64;
    uint _S1274 = a_counters_1(&kernelContext_191);
    atomic_store_explicit((&kernelContext_191)->atom_0+(_S1274 + 18U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_191)->atom_0+(_S1274 + 19U), 0U, memory_order_relaxed);
    uint _S1275 = a_il_cursor_1(&kernelContext_191);
    atomic_store_explicit((&kernelContext_191)->atom_0+_S1275, 0U, memory_order_relaxed);
    uint _S1276 = cs_rows_0(&kernelContext_191);
    uint _S1277 = groups_of_0(_S1276, 64U);
    set_args_0(7U, _S1277, &kernelContext_191);
    return;
}

[[kernel]] void fr_cs_contacts(uint3 id_60 [[thread_position_in_grid]], FractureParams_0 constant* params_65 [[buffer(0)]], atomic<uint> device* atom_65 [[buffer(5)]], uint device* work_65 [[buffer(4)]], packed_float4 device* frames_65 [[buffer(6)]], packed_float4 device* bodies_65 [[buffer(7)]], packed_float4 device* chunks_65 [[buffer(1)]], uint device* sort_65 [[buffer(11)]], Record_natural_0 device* manifolds_65 [[buffer(10)]], packed_float4 device* loads_65 [[buffer(12)]], packed_uint4 device* bonds_65 [[buffer(2)]], uint device* args_65 [[buffer(13)]], uint device* info_65 [[buffer(15)]], ScanParams_0 constant* scan_65 [[buffer(14)]], packed_float4 device* parts_65 [[buffer(3)]], packed_float4 device* colliders_65 [[buffer(8)]], packed_uint4 device* collider_info_65 [[buffer(9)]], packed_float4 device* view_chunks_65 [[buffer(17)]], ViewParams_0 constant* view_65 [[buffer(16)]], packed_float4 device* objects_65 [[buffer(18)]], uint device* shot_bodies_65 [[buffer(20)]], float device* lines_65 [[buffer(19)]], packed_float4 device* geometry_65 [[buffer(21)]], packed_float4 device* shots_65 [[buffer(22)]], packed_float4 device* lp_65 [[buffer(24)]], packed_float4 device* ext_67 [[buffer(23)]], packed_float4 device* engine_contacts_65 [[buffer(25)]], CrushParams_0 constant* crush_65 [[buffer(28)]], packed_float4 device* crush_rw_65 [[buffer(27)]], packed_float4 device* crush_tab_65 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_192;
    (&kernelContext_192)->params_0 = params_65;
    (&kernelContext_192)->atom_0 = atom_65;
    (&kernelContext_192)->work_0 = work_65;
    (&kernelContext_192)->frames_0 = frames_65;
    (&kernelContext_192)->bodies_0 = bodies_65;
    (&kernelContext_192)->chunks_0 = chunks_65;
    (&kernelContext_192)->sort_0 = sort_65;
    (&kernelContext_192)->manifolds_0 = manifolds_65;
    (&kernelContext_192)->loads_0 = loads_65;
    (&kernelContext_192)->bonds_0 = bonds_65;
    (&kernelContext_192)->args_0 = args_65;
    (&kernelContext_192)->info_0 = info_65;
    (&kernelContext_192)->scan_0 = scan_65;
    (&kernelContext_192)->parts_0 = parts_65;
    (&kernelContext_192)->colliders_0 = colliders_65;
    (&kernelContext_192)->collider_info_0 = collider_info_65;
    (&kernelContext_192)->view_chunks_0 = view_chunks_65;
    (&kernelContext_192)->view_0 = view_65;
    (&kernelContext_192)->objects_0 = objects_65;
    (&kernelContext_192)->shot_bodies_0 = shot_bodies_65;
    (&kernelContext_192)->lines_0 = lines_65;
    (&kernelContext_192)->geometry_0 = geometry_65;
    (&kernelContext_192)->shots_0 = shots_65;
    (&kernelContext_192)->lp_0 = lp_65;
    (&kernelContext_192)->ext_0 = ext_67;
    (&kernelContext_192)->engine_contacts_0 = engine_contacts_65;
    (&kernelContext_192)->crush_0 = crush_65;
    (&kernelContext_192)->crush_rw_0 = crush_rw_65;
    (&kernelContext_192)->crush_tab_0 = crush_tab_65;
    threadgroup array<uint, int(1024)> g_scan_65;
    (&kernelContext_192)->g_scan_0 = &g_scan_65;
    threadgroup array<float, int(64)> g_il_best_65;
    (&kernelContext_192)->g_il_best_0 = &g_il_best_65;
    threadgroup array<float, int(256)> g_gi_x_65;
    (&kernelContext_192)->g_gi_x_0 = &g_gi_x_65;
    threadgroup array<float, int(256)> g_gi_m_65;
    (&kernelContext_192)->g_gi_m_0 = &g_gi_m_65;
    threadgroup array<uint, int(1280)> g_gi_lo_65;
    (&kernelContext_192)->g_gi_lo_0 = &g_gi_lo_65;
    threadgroup array<uint, int(1280)> g_gi_hi_65;
    (&kernelContext_192)->g_gi_hi_0 = &g_gi_hi_65;
    uint r_24 = id_60.x;
    uint _S1278 = cs_rows_0(&kernelContext_192);
    if(r_24 >= _S1278)
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
        thread uint k_47;
        thread uint s_20;
        thread uint ob_5;
        thread uint co_5;
        thread float3 p_18;
        thread float3 u_10;
        thread float v_15;
        thread float m_eff_7;
        thread float t_in_8;
        bool _S1279 = cs_side_0(r_24, side_6, &k_47, &s_20, &ob_5, &co_5, &p_18, &u_10, &v_15, &m_eff_7, &t_in_8, &kernelContext_192);
        if(_S1279)
        {
            uint _S1280 = a_cs_0(s_20, &kernelContext_192);
            uint _S1281 = atomic_fetch_max_explicit((&kernelContext_192)->atom_0+_S1280, (as_type<uint>((m_eff_7 * v_15))), memory_order_relaxed);
            uint _S1282 = a_cs_0(s_20, &kernelContext_192);
            uint _S1283 = atomic_exchange_explicit((&kernelContext_192)->atom_0+(_S1282 + 8U), 1U, memory_order_relaxed);
            if(_S1283 == 0U)
            {
                uint _S1284 = w_cs_bodies_0(&kernelContext_192);
                uint _S1285 = a_counters_1(&kernelContext_192);
                uint _S1286 = atomic_fetch_add_explicit((&kernelContext_192)->atom_0+(_S1285 + 18U), 1U, memory_order_relaxed);
                *((&kernelContext_192)->work_0+(_S1284 + _S1286)) = s_20;
            }
        }
        side_6 = side_6 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_gate(uint3 id_61 [[thread_position_in_grid]], FractureParams_0 constant* params_66 [[buffer(0)]], atomic<uint> device* atom_66 [[buffer(5)]], uint device* work_66 [[buffer(4)]], packed_float4 device* frames_66 [[buffer(6)]], packed_float4 device* bodies_66 [[buffer(7)]], packed_float4 device* chunks_66 [[buffer(1)]], uint device* sort_66 [[buffer(11)]], Record_natural_0 device* manifolds_66 [[buffer(10)]], packed_float4 device* loads_66 [[buffer(12)]], packed_uint4 device* bonds_66 [[buffer(2)]], uint device* args_66 [[buffer(13)]], uint device* info_66 [[buffer(15)]], ScanParams_0 constant* scan_66 [[buffer(14)]], packed_float4 device* parts_66 [[buffer(3)]], packed_float4 device* colliders_66 [[buffer(8)]], packed_uint4 device* collider_info_66 [[buffer(9)]], packed_float4 device* view_chunks_66 [[buffer(17)]], ViewParams_0 constant* view_66 [[buffer(16)]], packed_float4 device* objects_66 [[buffer(18)]], uint device* shot_bodies_66 [[buffer(20)]], float device* lines_66 [[buffer(19)]], packed_float4 device* geometry_66 [[buffer(21)]], packed_float4 device* shots_66 [[buffer(22)]], packed_float4 device* lp_66 [[buffer(24)]], packed_float4 device* ext_68 [[buffer(23)]], packed_float4 device* engine_contacts_66 [[buffer(25)]], CrushParams_0 constant* crush_66 [[buffer(28)]], packed_float4 device* crush_rw_66 [[buffer(27)]], packed_float4 device* crush_tab_66 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_193;
    (&kernelContext_193)->params_0 = params_66;
    (&kernelContext_193)->atom_0 = atom_66;
    (&kernelContext_193)->work_0 = work_66;
    (&kernelContext_193)->frames_0 = frames_66;
    (&kernelContext_193)->bodies_0 = bodies_66;
    (&kernelContext_193)->chunks_0 = chunks_66;
    (&kernelContext_193)->sort_0 = sort_66;
    (&kernelContext_193)->manifolds_0 = manifolds_66;
    (&kernelContext_193)->loads_0 = loads_66;
    (&kernelContext_193)->bonds_0 = bonds_66;
    (&kernelContext_193)->args_0 = args_66;
    (&kernelContext_193)->info_0 = info_66;
    (&kernelContext_193)->scan_0 = scan_66;
    (&kernelContext_193)->parts_0 = parts_66;
    (&kernelContext_193)->colliders_0 = colliders_66;
    (&kernelContext_193)->collider_info_0 = collider_info_66;
    (&kernelContext_193)->view_chunks_0 = view_chunks_66;
    (&kernelContext_193)->view_0 = view_66;
    (&kernelContext_193)->objects_0 = objects_66;
    (&kernelContext_193)->shot_bodies_0 = shot_bodies_66;
    (&kernelContext_193)->lines_0 = lines_66;
    (&kernelContext_193)->geometry_0 = geometry_66;
    (&kernelContext_193)->shots_0 = shots_66;
    (&kernelContext_193)->lp_0 = lp_66;
    (&kernelContext_193)->ext_0 = ext_68;
    (&kernelContext_193)->engine_contacts_0 = engine_contacts_66;
    (&kernelContext_193)->crush_0 = crush_66;
    (&kernelContext_193)->crush_rw_0 = crush_rw_66;
    (&kernelContext_193)->crush_tab_0 = crush_tab_66;
    threadgroup array<uint, int(1024)> g_scan_66;
    (&kernelContext_193)->g_scan_0 = &g_scan_66;
    threadgroup array<float, int(64)> g_il_best_66;
    (&kernelContext_193)->g_il_best_0 = &g_il_best_66;
    threadgroup array<float, int(256)> g_gi_x_66;
    (&kernelContext_193)->g_gi_x_0 = &g_gi_x_66;
    threadgroup array<float, int(256)> g_gi_m_66;
    (&kernelContext_193)->g_gi_m_0 = &g_gi_m_66;
    threadgroup array<uint, int(1280)> g_gi_lo_66;
    (&kernelContext_193)->g_gi_lo_0 = &g_gi_lo_66;
    threadgroup array<uint, int(1280)> g_gi_hi_66;
    (&kernelContext_193)->g_gi_hi_0 = &g_gi_hi_66;
    uint _S1287 = a_counters_1(&kernelContext_193);
    uint struck_1 = atomic_load_explicit((&kernelContext_193)->atom_0+(_S1287 + 18U), memory_order_relaxed);
    bool any_0 = struck_1 > 0U;
    uint _S1288;
    if(any_0)
    {
        uint _S1289 = cs_rows_0(&kernelContext_193);
        uint _S1290 = groups_of_0(_S1289, 64U);
        _S1288 = _S1290;
    }
    else
    {
        _S1288 = 0U;
    }
    set_args_0(8U, _S1288, &kernelContext_193);
    uint _S1291 = groups_of_0(struck_1, 64U);
    set_args_0(9U, _S1291, &kernelContext_193);
    set_args_0(6U, struck_1, &kernelContext_193);
    if(any_0)
    {
        uint _S1292 = groups_of_0((&kernelContext_193)->params_0->chunk_count_0, 64U);
        _S1288 = _S1292;
    }
    else
    {
        _S1288 = 0U;
    }
    set_args_0(10U, _S1288, &kernelContext_193);
    if(any_0)
    {
        uint _S1293 = groups_of_0((&kernelContext_193)->params_0->bond_count_0, 64U);
        _S1288 = _S1293;
    }
    else
    {
        _S1288 = 0U;
    }
    set_args_0(11U, _S1288, &kernelContext_193);
    if(any_0)
    {
        uint _S1294 = groups_of_0((&kernelContext_193)->params_0->body_count_0, 64U);
        _S1288 = _S1294;
    }
    else
    {
        _S1288 = 0U;
    }
    set_args_0(12U, _S1288, &kernelContext_193);
    return;
}

[[kernel]] void fr_cs_pick_a(uint3 id_62 [[thread_position_in_grid]], FractureParams_0 constant* params_67 [[buffer(0)]], atomic<uint> device* atom_67 [[buffer(5)]], uint device* work_67 [[buffer(4)]], packed_float4 device* frames_67 [[buffer(6)]], packed_float4 device* bodies_67 [[buffer(7)]], packed_float4 device* chunks_67 [[buffer(1)]], uint device* sort_67 [[buffer(11)]], Record_natural_0 device* manifolds_67 [[buffer(10)]], packed_float4 device* loads_67 [[buffer(12)]], packed_uint4 device* bonds_67 [[buffer(2)]], uint device* args_67 [[buffer(13)]], uint device* info_67 [[buffer(15)]], ScanParams_0 constant* scan_67 [[buffer(14)]], packed_float4 device* parts_67 [[buffer(3)]], packed_float4 device* colliders_67 [[buffer(8)]], packed_uint4 device* collider_info_67 [[buffer(9)]], packed_float4 device* view_chunks_67 [[buffer(17)]], ViewParams_0 constant* view_67 [[buffer(16)]], packed_float4 device* objects_67 [[buffer(18)]], uint device* shot_bodies_67 [[buffer(20)]], float device* lines_67 [[buffer(19)]], packed_float4 device* geometry_67 [[buffer(21)]], packed_float4 device* shots_67 [[buffer(22)]], packed_float4 device* lp_67 [[buffer(24)]], packed_float4 device* ext_69 [[buffer(23)]], packed_float4 device* engine_contacts_67 [[buffer(25)]], CrushParams_0 constant* crush_67 [[buffer(28)]], packed_float4 device* crush_rw_67 [[buffer(27)]], packed_float4 device* crush_tab_67 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_194;
    (&kernelContext_194)->params_0 = params_67;
    (&kernelContext_194)->atom_0 = atom_67;
    (&kernelContext_194)->work_0 = work_67;
    (&kernelContext_194)->frames_0 = frames_67;
    (&kernelContext_194)->bodies_0 = bodies_67;
    (&kernelContext_194)->chunks_0 = chunks_67;
    (&kernelContext_194)->sort_0 = sort_67;
    (&kernelContext_194)->manifolds_0 = manifolds_67;
    (&kernelContext_194)->loads_0 = loads_67;
    (&kernelContext_194)->bonds_0 = bonds_67;
    (&kernelContext_194)->args_0 = args_67;
    (&kernelContext_194)->info_0 = info_67;
    (&kernelContext_194)->scan_0 = scan_67;
    (&kernelContext_194)->parts_0 = parts_67;
    (&kernelContext_194)->colliders_0 = colliders_67;
    (&kernelContext_194)->collider_info_0 = collider_info_67;
    (&kernelContext_194)->view_chunks_0 = view_chunks_67;
    (&kernelContext_194)->view_0 = view_67;
    (&kernelContext_194)->objects_0 = objects_67;
    (&kernelContext_194)->shot_bodies_0 = shot_bodies_67;
    (&kernelContext_194)->lines_0 = lines_67;
    (&kernelContext_194)->geometry_0 = geometry_67;
    (&kernelContext_194)->shots_0 = shots_67;
    (&kernelContext_194)->lp_0 = lp_67;
    (&kernelContext_194)->ext_0 = ext_69;
    (&kernelContext_194)->engine_contacts_0 = engine_contacts_67;
    (&kernelContext_194)->crush_0 = crush_67;
    (&kernelContext_194)->crush_rw_0 = crush_rw_67;
    (&kernelContext_194)->crush_tab_0 = crush_tab_67;
    threadgroup array<uint, int(1024)> g_scan_67;
    (&kernelContext_194)->g_scan_0 = &g_scan_67;
    threadgroup array<float, int(64)> g_il_best_67;
    (&kernelContext_194)->g_il_best_0 = &g_il_best_67;
    threadgroup array<float, int(256)> g_gi_x_67;
    (&kernelContext_194)->g_gi_x_0 = &g_gi_x_67;
    threadgroup array<float, int(256)> g_gi_m_67;
    (&kernelContext_194)->g_gi_m_0 = &g_gi_m_67;
    threadgroup array<uint, int(1280)> g_gi_lo_67;
    (&kernelContext_194)->g_gi_lo_0 = &g_gi_lo_67;
    threadgroup array<uint, int(1280)> g_gi_hi_67;
    (&kernelContext_194)->g_gi_hi_0 = &g_gi_hi_67;
    cs_pick_3(id_62.x, &kernelContext_194);
    return;
}

[[kernel]] void fr_cs_pick_b(uint3 id_63 [[thread_position_in_grid]], FractureParams_0 constant* params_68 [[buffer(0)]], atomic<uint> device* atom_68 [[buffer(5)]], uint device* work_68 [[buffer(4)]], packed_float4 device* frames_68 [[buffer(6)]], packed_float4 device* bodies_68 [[buffer(7)]], packed_float4 device* chunks_68 [[buffer(1)]], uint device* sort_68 [[buffer(11)]], Record_natural_0 device* manifolds_68 [[buffer(10)]], packed_float4 device* loads_68 [[buffer(12)]], packed_uint4 device* bonds_68 [[buffer(2)]], uint device* args_68 [[buffer(13)]], uint device* info_68 [[buffer(15)]], ScanParams_0 constant* scan_68 [[buffer(14)]], packed_float4 device* parts_68 [[buffer(3)]], packed_float4 device* colliders_68 [[buffer(8)]], packed_uint4 device* collider_info_68 [[buffer(9)]], packed_float4 device* view_chunks_68 [[buffer(17)]], ViewParams_0 constant* view_68 [[buffer(16)]], packed_float4 device* objects_68 [[buffer(18)]], uint device* shot_bodies_68 [[buffer(20)]], float device* lines_68 [[buffer(19)]], packed_float4 device* geometry_68 [[buffer(21)]], packed_float4 device* shots_68 [[buffer(22)]], packed_float4 device* lp_68 [[buffer(24)]], packed_float4 device* ext_70 [[buffer(23)]], packed_float4 device* engine_contacts_68 [[buffer(25)]], CrushParams_0 constant* crush_68 [[buffer(28)]], packed_float4 device* crush_rw_68 [[buffer(27)]], packed_float4 device* crush_tab_68 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_195;
    (&kernelContext_195)->params_0 = params_68;
    (&kernelContext_195)->atom_0 = atom_68;
    (&kernelContext_195)->work_0 = work_68;
    (&kernelContext_195)->frames_0 = frames_68;
    (&kernelContext_195)->bodies_0 = bodies_68;
    (&kernelContext_195)->chunks_0 = chunks_68;
    (&kernelContext_195)->sort_0 = sort_68;
    (&kernelContext_195)->manifolds_0 = manifolds_68;
    (&kernelContext_195)->loads_0 = loads_68;
    (&kernelContext_195)->bonds_0 = bonds_68;
    (&kernelContext_195)->args_0 = args_68;
    (&kernelContext_195)->info_0 = info_68;
    (&kernelContext_195)->scan_0 = scan_68;
    (&kernelContext_195)->parts_0 = parts_68;
    (&kernelContext_195)->colliders_0 = colliders_68;
    (&kernelContext_195)->collider_info_0 = collider_info_68;
    (&kernelContext_195)->view_chunks_0 = view_chunks_68;
    (&kernelContext_195)->view_0 = view_68;
    (&kernelContext_195)->objects_0 = objects_68;
    (&kernelContext_195)->shot_bodies_0 = shot_bodies_68;
    (&kernelContext_195)->lines_0 = lines_68;
    (&kernelContext_195)->geometry_0 = geometry_68;
    (&kernelContext_195)->shots_0 = shots_68;
    (&kernelContext_195)->lp_0 = lp_68;
    (&kernelContext_195)->ext_0 = ext_70;
    (&kernelContext_195)->engine_contacts_0 = engine_contacts_68;
    (&kernelContext_195)->crush_0 = crush_68;
    (&kernelContext_195)->crush_rw_0 = crush_rw_68;
    (&kernelContext_195)->crush_tab_0 = crush_tab_68;
    threadgroup array<uint, int(1024)> g_scan_68;
    (&kernelContext_195)->g_scan_0 = &g_scan_68;
    threadgroup array<float, int(64)> g_il_best_68;
    (&kernelContext_195)->g_il_best_0 = &g_il_best_68;
    threadgroup array<float, int(256)> g_gi_x_68;
    (&kernelContext_195)->g_gi_x_0 = &g_gi_x_68;
    threadgroup array<float, int(256)> g_gi_m_68;
    (&kernelContext_195)->g_gi_m_0 = &g_gi_m_68;
    threadgroup array<uint, int(1280)> g_gi_lo_68;
    (&kernelContext_195)->g_gi_lo_0 = &g_gi_lo_68;
    threadgroup array<uint, int(1280)> g_gi_hi_68;
    (&kernelContext_195)->g_gi_hi_0 = &g_gi_hi_68;
    cs_pick_2(id_63.x, &kernelContext_195);
    return;
}

[[kernel]] void fr_cs_pick_c(uint3 id_64 [[thread_position_in_grid]], FractureParams_0 constant* params_69 [[buffer(0)]], atomic<uint> device* atom_69 [[buffer(5)]], uint device* work_69 [[buffer(4)]], packed_float4 device* frames_69 [[buffer(6)]], packed_float4 device* bodies_69 [[buffer(7)]], packed_float4 device* chunks_69 [[buffer(1)]], uint device* sort_69 [[buffer(11)]], Record_natural_0 device* manifolds_69 [[buffer(10)]], packed_float4 device* loads_69 [[buffer(12)]], packed_uint4 device* bonds_69 [[buffer(2)]], uint device* args_69 [[buffer(13)]], uint device* info_69 [[buffer(15)]], ScanParams_0 constant* scan_69 [[buffer(14)]], packed_float4 device* parts_69 [[buffer(3)]], packed_float4 device* colliders_69 [[buffer(8)]], packed_uint4 device* collider_info_69 [[buffer(9)]], packed_float4 device* view_chunks_69 [[buffer(17)]], ViewParams_0 constant* view_69 [[buffer(16)]], packed_float4 device* objects_69 [[buffer(18)]], uint device* shot_bodies_69 [[buffer(20)]], float device* lines_69 [[buffer(19)]], packed_float4 device* geometry_69 [[buffer(21)]], packed_float4 device* shots_69 [[buffer(22)]], packed_float4 device* lp_69 [[buffer(24)]], packed_float4 device* ext_71 [[buffer(23)]], packed_float4 device* engine_contacts_69 [[buffer(25)]], CrushParams_0 constant* crush_69 [[buffer(28)]], packed_float4 device* crush_rw_69 [[buffer(27)]], packed_float4 device* crush_tab_69 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_196;
    (&kernelContext_196)->params_0 = params_69;
    (&kernelContext_196)->atom_0 = atom_69;
    (&kernelContext_196)->work_0 = work_69;
    (&kernelContext_196)->frames_0 = frames_69;
    (&kernelContext_196)->bodies_0 = bodies_69;
    (&kernelContext_196)->chunks_0 = chunks_69;
    (&kernelContext_196)->sort_0 = sort_69;
    (&kernelContext_196)->manifolds_0 = manifolds_69;
    (&kernelContext_196)->loads_0 = loads_69;
    (&kernelContext_196)->bonds_0 = bonds_69;
    (&kernelContext_196)->args_0 = args_69;
    (&kernelContext_196)->info_0 = info_69;
    (&kernelContext_196)->scan_0 = scan_69;
    (&kernelContext_196)->parts_0 = parts_69;
    (&kernelContext_196)->colliders_0 = colliders_69;
    (&kernelContext_196)->collider_info_0 = collider_info_69;
    (&kernelContext_196)->view_chunks_0 = view_chunks_69;
    (&kernelContext_196)->view_0 = view_69;
    (&kernelContext_196)->objects_0 = objects_69;
    (&kernelContext_196)->shot_bodies_0 = shot_bodies_69;
    (&kernelContext_196)->lines_0 = lines_69;
    (&kernelContext_196)->geometry_0 = geometry_69;
    (&kernelContext_196)->shots_0 = shots_69;
    (&kernelContext_196)->lp_0 = lp_69;
    (&kernelContext_196)->ext_0 = ext_71;
    (&kernelContext_196)->engine_contacts_0 = engine_contacts_69;
    (&kernelContext_196)->crush_0 = crush_69;
    (&kernelContext_196)->crush_rw_0 = crush_rw_69;
    (&kernelContext_196)->crush_tab_0 = crush_tab_69;
    threadgroup array<uint, int(1024)> g_scan_69;
    (&kernelContext_196)->g_scan_0 = &g_scan_69;
    threadgroup array<float, int(64)> g_il_best_69;
    (&kernelContext_196)->g_il_best_0 = &g_il_best_69;
    threadgroup array<float, int(256)> g_gi_x_69;
    (&kernelContext_196)->g_gi_x_0 = &g_gi_x_69;
    threadgroup array<float, int(256)> g_gi_m_69;
    (&kernelContext_196)->g_gi_m_0 = &g_gi_m_69;
    threadgroup array<uint, int(1280)> g_gi_lo_69;
    (&kernelContext_196)->g_gi_lo_0 = &g_gi_lo_69;
    threadgroup array<uint, int(1280)> g_gi_hi_69;
    (&kernelContext_196)->g_gi_hi_0 = &g_gi_hi_69;
    cs_pick_1(id_64.x, &kernelContext_196);
    return;
}

[[kernel]] void fr_cs_pick_d(uint3 id_65 [[thread_position_in_grid]], FractureParams_0 constant* params_70 [[buffer(0)]], atomic<uint> device* atom_70 [[buffer(5)]], uint device* work_70 [[buffer(4)]], packed_float4 device* frames_70 [[buffer(6)]], packed_float4 device* bodies_70 [[buffer(7)]], packed_float4 device* chunks_70 [[buffer(1)]], uint device* sort_70 [[buffer(11)]], Record_natural_0 device* manifolds_70 [[buffer(10)]], packed_float4 device* loads_70 [[buffer(12)]], packed_uint4 device* bonds_70 [[buffer(2)]], uint device* args_70 [[buffer(13)]], uint device* info_70 [[buffer(15)]], ScanParams_0 constant* scan_70 [[buffer(14)]], packed_float4 device* parts_70 [[buffer(3)]], packed_float4 device* colliders_70 [[buffer(8)]], packed_uint4 device* collider_info_70 [[buffer(9)]], packed_float4 device* view_chunks_70 [[buffer(17)]], ViewParams_0 constant* view_70 [[buffer(16)]], packed_float4 device* objects_70 [[buffer(18)]], uint device* shot_bodies_70 [[buffer(20)]], float device* lines_70 [[buffer(19)]], packed_float4 device* geometry_70 [[buffer(21)]], packed_float4 device* shots_70 [[buffer(22)]], packed_float4 device* lp_70 [[buffer(24)]], packed_float4 device* ext_72 [[buffer(23)]], packed_float4 device* engine_contacts_70 [[buffer(25)]], CrushParams_0 constant* crush_70 [[buffer(28)]], packed_float4 device* crush_rw_70 [[buffer(27)]], packed_float4 device* crush_tab_70 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_197;
    (&kernelContext_197)->params_0 = params_70;
    (&kernelContext_197)->atom_0 = atom_70;
    (&kernelContext_197)->work_0 = work_70;
    (&kernelContext_197)->frames_0 = frames_70;
    (&kernelContext_197)->bodies_0 = bodies_70;
    (&kernelContext_197)->chunks_0 = chunks_70;
    (&kernelContext_197)->sort_0 = sort_70;
    (&kernelContext_197)->manifolds_0 = manifolds_70;
    (&kernelContext_197)->loads_0 = loads_70;
    (&kernelContext_197)->bonds_0 = bonds_70;
    (&kernelContext_197)->args_0 = args_70;
    (&kernelContext_197)->info_0 = info_70;
    (&kernelContext_197)->scan_0 = scan_70;
    (&kernelContext_197)->parts_0 = parts_70;
    (&kernelContext_197)->colliders_0 = colliders_70;
    (&kernelContext_197)->collider_info_0 = collider_info_70;
    (&kernelContext_197)->view_chunks_0 = view_chunks_70;
    (&kernelContext_197)->view_0 = view_70;
    (&kernelContext_197)->objects_0 = objects_70;
    (&kernelContext_197)->shot_bodies_0 = shot_bodies_70;
    (&kernelContext_197)->lines_0 = lines_70;
    (&kernelContext_197)->geometry_0 = geometry_70;
    (&kernelContext_197)->shots_0 = shots_70;
    (&kernelContext_197)->lp_0 = lp_70;
    (&kernelContext_197)->ext_0 = ext_72;
    (&kernelContext_197)->engine_contacts_0 = engine_contacts_70;
    (&kernelContext_197)->crush_0 = crush_70;
    (&kernelContext_197)->crush_rw_0 = crush_rw_70;
    (&kernelContext_197)->crush_tab_0 = crush_tab_70;
    threadgroup array<uint, int(1024)> g_scan_70;
    (&kernelContext_197)->g_scan_0 = &g_scan_70;
    threadgroup array<float, int(64)> g_il_best_70;
    (&kernelContext_197)->g_il_best_0 = &g_il_best_70;
    threadgroup array<float, int(256)> g_gi_x_70;
    (&kernelContext_197)->g_gi_x_0 = &g_gi_x_70;
    threadgroup array<float, int(256)> g_gi_m_70;
    (&kernelContext_197)->g_gi_m_0 = &g_gi_m_70;
    threadgroup array<uint, int(1280)> g_gi_lo_70;
    (&kernelContext_197)->g_gi_lo_0 = &g_gi_lo_70;
    threadgroup array<uint, int(1280)> g_gi_hi_70;
    (&kernelContext_197)->g_gi_hi_0 = &g_gi_hi_70;
    cs_pick_0(id_65.x, &kernelContext_197);
    return;
}

[[kernel]] void fr_cs_record(uint3 id_66 [[thread_position_in_grid]], FractureParams_0 constant* params_71 [[buffer(0)]], atomic<uint> device* atom_71 [[buffer(5)]], uint device* work_71 [[buffer(4)]], packed_float4 device* frames_71 [[buffer(6)]], packed_float4 device* bodies_71 [[buffer(7)]], packed_float4 device* chunks_71 [[buffer(1)]], uint device* sort_71 [[buffer(11)]], Record_natural_0 device* manifolds_71 [[buffer(10)]], packed_float4 device* loads_71 [[buffer(12)]], packed_uint4 device* bonds_71 [[buffer(2)]], uint device* args_71 [[buffer(13)]], uint device* info_71 [[buffer(15)]], ScanParams_0 constant* scan_71 [[buffer(14)]], packed_float4 device* parts_71 [[buffer(3)]], packed_float4 device* colliders_71 [[buffer(8)]], packed_uint4 device* collider_info_71 [[buffer(9)]], packed_float4 device* view_chunks_71 [[buffer(17)]], ViewParams_0 constant* view_71 [[buffer(16)]], packed_float4 device* objects_71 [[buffer(18)]], uint device* shot_bodies_71 [[buffer(20)]], float device* lines_71 [[buffer(19)]], packed_float4 device* geometry_71 [[buffer(21)]], packed_float4 device* shots_71 [[buffer(22)]], packed_float4 device* lp_71 [[buffer(24)]], packed_float4 device* ext_73 [[buffer(23)]], packed_float4 device* engine_contacts_71 [[buffer(25)]], CrushParams_0 constant* crush_71 [[buffer(28)]], packed_float4 device* crush_rw_71 [[buffer(27)]], packed_float4 device* crush_tab_71 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_198;
    (&kernelContext_198)->params_0 = params_71;
    (&kernelContext_198)->atom_0 = atom_71;
    (&kernelContext_198)->work_0 = work_71;
    (&kernelContext_198)->frames_0 = frames_71;
    (&kernelContext_198)->bodies_0 = bodies_71;
    (&kernelContext_198)->chunks_0 = chunks_71;
    (&kernelContext_198)->sort_0 = sort_71;
    (&kernelContext_198)->manifolds_0 = manifolds_71;
    (&kernelContext_198)->loads_0 = loads_71;
    (&kernelContext_198)->bonds_0 = bonds_71;
    (&kernelContext_198)->args_0 = args_71;
    (&kernelContext_198)->info_0 = info_71;
    (&kernelContext_198)->scan_0 = scan_71;
    (&kernelContext_198)->parts_0 = parts_71;
    (&kernelContext_198)->colliders_0 = colliders_71;
    (&kernelContext_198)->collider_info_0 = collider_info_71;
    (&kernelContext_198)->view_chunks_0 = view_chunks_71;
    (&kernelContext_198)->view_0 = view_71;
    (&kernelContext_198)->objects_0 = objects_71;
    (&kernelContext_198)->shot_bodies_0 = shot_bodies_71;
    (&kernelContext_198)->lines_0 = lines_71;
    (&kernelContext_198)->geometry_0 = geometry_71;
    (&kernelContext_198)->shots_0 = shots_71;
    (&kernelContext_198)->lp_0 = lp_71;
    (&kernelContext_198)->ext_0 = ext_73;
    (&kernelContext_198)->engine_contacts_0 = engine_contacts_71;
    (&kernelContext_198)->crush_0 = crush_71;
    (&kernelContext_198)->crush_rw_0 = crush_rw_71;
    (&kernelContext_198)->crush_tab_0 = crush_tab_71;
    threadgroup array<uint, int(1024)> g_scan_71;
    (&kernelContext_198)->g_scan_0 = &g_scan_71;
    threadgroup array<float, int(64)> g_il_best_71;
    (&kernelContext_198)->g_il_best_0 = &g_il_best_71;
    threadgroup array<float, int(256)> g_gi_x_71;
    (&kernelContext_198)->g_gi_x_0 = &g_gi_x_71;
    threadgroup array<float, int(256)> g_gi_m_71;
    (&kernelContext_198)->g_gi_m_0 = &g_gi_m_71;
    threadgroup array<uint, int(1280)> g_gi_lo_71;
    (&kernelContext_198)->g_gi_lo_0 = &g_gi_lo_71;
    threadgroup array<uint, int(1280)> g_gi_hi_71;
    (&kernelContext_198)->g_gi_hi_0 = &g_gi_hi_71;
    uint _S1295 = id_66.x;
    uint _S1296 = a_counters_1(&kernelContext_198);
    uint _S1297 = atomic_load_explicit((&kernelContext_198)->atom_0+(_S1296 + 18U), memory_order_relaxed);
    if(_S1295 >= _S1297)
    {
        return;
    }
    uint _S1298 = w_cs_bodies_0(&kernelContext_198);
    uint device* _S1299 = (&kernelContext_198)->work_0+(_S1298 + _S1295);
    uint s_21 = *_S1299;
    uint _S1300 = a_cs_0(*_S1299, &kernelContext_198);
    uint _S1301 = atomic_load_explicit((&kernelContext_198)->atom_0+(_S1300 + 11U), memory_order_relaxed);
    uint key_2 = _S1301 - 1U;
    thread uint k_48;
    thread uint s2_0;
    thread uint ob_6;
    thread uint co_6;
    thread float3 p_19;
    thread float3 u_11;
    thread float v_16;
    thread float m_eff_8;
    thread float t_in_9;
    bool _S1302 = cs_side_0(key_2 / 2U, key_2 % 2U, &k_48, &s2_0, &ob_6, &co_6, &p_19, &u_11, &v_16, &m_eff_8, &t_in_9, &kernelContext_198);
    if(!_S1302)
    {
        return;
    }
    uint4 _S1303 = uint4(*((&kernelContext_198)->collider_info_0+(2U * co_6 + 1U))) ;
    float e_o_0 = (as_type<float>((_S1303.y)));
    float _S1304 = max((as_type<float>((_S1303.z))), 0.00100000004749745f);
    float j_10 = m_eff_8 * v_16;
    float tau_3 = hertz_duration_0(m_eff_8, _S1304, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / (float4(*((&kernelContext_198)->geometry_0+(3U * k_48 + 1U))) ).w), v_16);
    uint _S1305 = lp_body_0(s_21, &kernelContext_198);
    *((&kernelContext_198)->lp_0+(_S1305 + 2U)) = packed_float4(float4(from_rubble_0(p_19), 3.14159274101257324f * j_10 / (2.0f * tau_3))) ;
    *((&kernelContext_198)->lp_0+(_S1305 + 3U)) = packed_float4(float4(from_rubble_0(u_11), v_16)) ;
    *((&kernelContext_198)->lp_0+(_S1305 + 4U)) = packed_float4(float4(m_eff_8, _S1304, e_o_0, (as_type<float>((ob_6))))) ;
    *((&kernelContext_198)->lp_0+(_S1305 + 5U)) = packed_float4(float4((as_type<float>((k_48))), j_10 / tau_3, 1.0f / (float4(*((&kernelContext_198)->bodies_0+7U * ob_6)) ).w, t_in_9)) ;
    *((&kernelContext_198)->lp_0+(_S1305 + 6U)) = packed_float4(float4((as_type<float>((co_6))), (as_type<float>((_S1303.w))), tau_3, 0.0f)) ;
    atomic_store_explicit((&kernelContext_198)->atom_0+(_S1300 + 1U), k_48 + 1U, memory_order_relaxed);
    uint _S1306 = atomic_fetch_add_explicit((&kernelContext_198)->atom_0+(_S1296 + 16U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_hertz(uint3 id_67 [[thread_position_in_grid]], FractureParams_0 constant* params_72 [[buffer(0)]], atomic<uint> device* atom_72 [[buffer(5)]], uint device* work_72 [[buffer(4)]], packed_float4 device* frames_72 [[buffer(6)]], packed_float4 device* bodies_72 [[buffer(7)]], packed_float4 device* chunks_72 [[buffer(1)]], uint device* sort_72 [[buffer(11)]], Record_natural_0 device* manifolds_72 [[buffer(10)]], packed_float4 device* loads_72 [[buffer(12)]], packed_uint4 device* bonds_72 [[buffer(2)]], uint device* args_72 [[buffer(13)]], uint device* info_72 [[buffer(15)]], ScanParams_0 constant* scan_72 [[buffer(14)]], packed_float4 device* parts_72 [[buffer(3)]], packed_float4 device* colliders_72 [[buffer(8)]], packed_uint4 device* collider_info_72 [[buffer(9)]], packed_float4 device* view_chunks_72 [[buffer(17)]], ViewParams_0 constant* view_72 [[buffer(16)]], packed_float4 device* objects_72 [[buffer(18)]], uint device* shot_bodies_72 [[buffer(20)]], float device* lines_72 [[buffer(19)]], packed_float4 device* geometry_72 [[buffer(21)]], packed_float4 device* shots_72 [[buffer(22)]], packed_float4 device* lp_72 [[buffer(24)]], packed_float4 device* ext_74 [[buffer(23)]], packed_float4 device* engine_contacts_72 [[buffer(25)]], CrushParams_0 constant* crush_72 [[buffer(28)]], packed_float4 device* crush_rw_72 [[buffer(27)]], packed_float4 device* crush_tab_72 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_199;
    (&kernelContext_199)->params_0 = params_72;
    (&kernelContext_199)->atom_0 = atom_72;
    (&kernelContext_199)->work_0 = work_72;
    (&kernelContext_199)->frames_0 = frames_72;
    (&kernelContext_199)->bodies_0 = bodies_72;
    (&kernelContext_199)->chunks_0 = chunks_72;
    (&kernelContext_199)->sort_0 = sort_72;
    (&kernelContext_199)->manifolds_0 = manifolds_72;
    (&kernelContext_199)->loads_0 = loads_72;
    (&kernelContext_199)->bonds_0 = bonds_72;
    (&kernelContext_199)->args_0 = args_72;
    (&kernelContext_199)->info_0 = info_72;
    (&kernelContext_199)->scan_0 = scan_72;
    (&kernelContext_199)->parts_0 = parts_72;
    (&kernelContext_199)->colliders_0 = colliders_72;
    (&kernelContext_199)->collider_info_0 = collider_info_72;
    (&kernelContext_199)->view_chunks_0 = view_chunks_72;
    (&kernelContext_199)->view_0 = view_72;
    (&kernelContext_199)->objects_0 = objects_72;
    (&kernelContext_199)->shot_bodies_0 = shot_bodies_72;
    (&kernelContext_199)->lines_0 = lines_72;
    (&kernelContext_199)->geometry_0 = geometry_72;
    (&kernelContext_199)->shots_0 = shots_72;
    (&kernelContext_199)->lp_0 = lp_72;
    (&kernelContext_199)->ext_0 = ext_74;
    (&kernelContext_199)->engine_contacts_0 = engine_contacts_72;
    (&kernelContext_199)->crush_0 = crush_72;
    (&kernelContext_199)->crush_rw_0 = crush_rw_72;
    (&kernelContext_199)->crush_tab_0 = crush_tab_72;
    threadgroup array<uint, int(1024)> g_scan_72;
    (&kernelContext_199)->g_scan_0 = &g_scan_72;
    threadgroup array<float, int(64)> g_il_best_72;
    (&kernelContext_199)->g_il_best_0 = &g_il_best_72;
    threadgroup array<float, int(256)> g_gi_x_72;
    (&kernelContext_199)->g_gi_x_0 = &g_gi_x_72;
    threadgroup array<float, int(256)> g_gi_m_72;
    (&kernelContext_199)->g_gi_m_0 = &g_gi_m_72;
    threadgroup array<uint, int(1280)> g_gi_lo_72;
    (&kernelContext_199)->g_gi_lo_0 = &g_gi_lo_72;
    threadgroup array<uint, int(1280)> g_gi_hi_72;
    (&kernelContext_199)->g_gi_hi_0 = &g_gi_hi_72;
    uint i_23 = id_67.x;
    bool _S1307;
    if(i_23 >= (params_72->bond_count_0))
    {
        _S1307 = true;
    }
    else
    {
        uint _S1308 = w_alive_0(&kernelContext_199);
        _S1307 = (*((&kernelContext_199)->work_0+(_S1308 + i_23))) == 0U;
    }
    if(_S1307)
    {
        return;
    }
    uint4 _S1309 = uint4(*((&kernelContext_199)->bonds_0+i_23)) ;
    uint _S1310 = body_of_0(_S1309.x, &kernelContext_199);
    bool _S1311 = cs_struck_1(_S1310, &kernelContext_199);
    if(!_S1311)
    {
        return;
    }
    uint _S1312 = lp_body_0(_S1310, &kernelContext_199);
    bool _S1313 = strike_share_breaks_0(_S1309, (float4(*((&kernelContext_199)->lp_0+(_S1312 + 2U))) ).xyz, (float4(*((&kernelContext_199)->lp_0+(_S1312 + 5U))) ).y, &kernelContext_199);
    if(_S1313)
    {
        uint _S1314 = w_alive_0(&kernelContext_199);
        *((&kernelContext_199)->work_0+(_S1314 + i_23)) = 0U;
        uint _S1315 = a_dirty_0(&kernelContext_199);
        uint _S1316 = atomic_fetch_or_explicit((&kernelContext_199)->atom_0+(_S1315 + _S1310), 1U, memory_order_relaxed);
        uint _S1317 = a_counters_1(&kernelContext_199);
        uint _S1318 = atomic_fetch_add_explicit((&kernelContext_199)->atom_0+_S1317, 1U, memory_order_relaxed);
        uint _S1319 = atomic_fetch_add_explicit((&kernelContext_199)->atom_0+(_S1317 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_23, &kernelContext_199);
        uint _S1320 = atomic_fetch_add_explicit((&kernelContext_199)->atom_0+(_S1317 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_gate2(uint3 id_68 [[thread_position_in_grid]], FractureParams_0 constant* params_73 [[buffer(0)]], atomic<uint> device* atom_73 [[buffer(5)]], uint device* work_73 [[buffer(4)]], packed_float4 device* frames_73 [[buffer(6)]], packed_float4 device* bodies_73 [[buffer(7)]], packed_float4 device* chunks_73 [[buffer(1)]], uint device* sort_73 [[buffer(11)]], Record_natural_0 device* manifolds_73 [[buffer(10)]], packed_float4 device* loads_73 [[buffer(12)]], packed_uint4 device* bonds_73 [[buffer(2)]], uint device* args_73 [[buffer(13)]], uint device* info_73 [[buffer(15)]], ScanParams_0 constant* scan_73 [[buffer(14)]], packed_float4 device* parts_73 [[buffer(3)]], packed_float4 device* colliders_73 [[buffer(8)]], packed_uint4 device* collider_info_73 [[buffer(9)]], packed_float4 device* view_chunks_73 [[buffer(17)]], ViewParams_0 constant* view_73 [[buffer(16)]], packed_float4 device* objects_73 [[buffer(18)]], uint device* shot_bodies_73 [[buffer(20)]], float device* lines_73 [[buffer(19)]], packed_float4 device* geometry_73 [[buffer(21)]], packed_float4 device* shots_73 [[buffer(22)]], packed_float4 device* lp_73 [[buffer(24)]], packed_float4 device* ext_75 [[buffer(23)]], packed_float4 device* engine_contacts_73 [[buffer(25)]], CrushParams_0 constant* crush_73 [[buffer(28)]], packed_float4 device* crush_rw_73 [[buffer(27)]], packed_float4 device* crush_tab_73 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_200;
    (&kernelContext_200)->params_0 = params_73;
    (&kernelContext_200)->atom_0 = atom_73;
    (&kernelContext_200)->work_0 = work_73;
    (&kernelContext_200)->frames_0 = frames_73;
    (&kernelContext_200)->bodies_0 = bodies_73;
    (&kernelContext_200)->chunks_0 = chunks_73;
    (&kernelContext_200)->sort_0 = sort_73;
    (&kernelContext_200)->manifolds_0 = manifolds_73;
    (&kernelContext_200)->loads_0 = loads_73;
    (&kernelContext_200)->bonds_0 = bonds_73;
    (&kernelContext_200)->args_0 = args_73;
    (&kernelContext_200)->info_0 = info_73;
    (&kernelContext_200)->scan_0 = scan_73;
    (&kernelContext_200)->parts_0 = parts_73;
    (&kernelContext_200)->colliders_0 = colliders_73;
    (&kernelContext_200)->collider_info_0 = collider_info_73;
    (&kernelContext_200)->view_chunks_0 = view_chunks_73;
    (&kernelContext_200)->view_0 = view_73;
    (&kernelContext_200)->objects_0 = objects_73;
    (&kernelContext_200)->shot_bodies_0 = shot_bodies_73;
    (&kernelContext_200)->lines_0 = lines_73;
    (&kernelContext_200)->geometry_0 = geometry_73;
    (&kernelContext_200)->shots_0 = shots_73;
    (&kernelContext_200)->lp_0 = lp_73;
    (&kernelContext_200)->ext_0 = ext_75;
    (&kernelContext_200)->engine_contacts_0 = engine_contacts_73;
    (&kernelContext_200)->crush_0 = crush_73;
    (&kernelContext_200)->crush_rw_0 = crush_rw_73;
    (&kernelContext_200)->crush_tab_0 = crush_tab_73;
    threadgroup array<uint, int(1024)> g_scan_73;
    (&kernelContext_200)->g_scan_0 = &g_scan_73;
    threadgroup array<float, int(64)> g_il_best_73;
    (&kernelContext_200)->g_il_best_0 = &g_il_best_73;
    threadgroup array<float, int(256)> g_gi_x_73;
    (&kernelContext_200)->g_gi_x_0 = &g_gi_x_73;
    threadgroup array<float, int(256)> g_gi_m_73;
    (&kernelContext_200)->g_gi_m_0 = &g_gi_m_73;
    threadgroup array<uint, int(1280)> g_gi_lo_73;
    (&kernelContext_200)->g_gi_lo_0 = &g_gi_lo_73;
    threadgroup array<uint, int(1280)> g_gi_hi_73;
    (&kernelContext_200)->g_gi_hi_0 = &g_gi_hi_73;
    uint _S1321 = a_counters_1(&kernelContext_200);
    uint _S1322 = atomic_load_explicit((&kernelContext_200)->atom_0+_S1321, memory_order_relaxed);
    uint _S1323;
    if(_S1322 > 0U)
    {
        uint _S1324 = cs_rows_0(&kernelContext_200);
        uint _S1325 = groups_of_0(_S1324, 64U);
        _S1323 = _S1325;
    }
    else
    {
        _S1323 = 0U;
    }
    set_args_0(13U, _S1323, &kernelContext_200);
    return;
}

uint lp_anchor_0(uint r_25, uint side_7, KernelContext_0 thread* kernelContext_201)
{
    return 22U * kernelContext_201->params_0->chunk_count_0 + 7U * kernelContext_201->params_0->body_count_0 + kernelContext_201->params_0->chunk_count_0 + 2U * r_25 + side_7;
}

[[kernel]] void fr_cs_save(uint3 id_69 [[thread_position_in_grid]], FractureParams_0 constant* params_74 [[buffer(0)]], atomic<uint> device* atom_74 [[buffer(5)]], uint device* work_74 [[buffer(4)]], packed_float4 device* frames_74 [[buffer(6)]], packed_float4 device* bodies_74 [[buffer(7)]], packed_float4 device* chunks_74 [[buffer(1)]], uint device* sort_74 [[buffer(11)]], Record_natural_0 device* manifolds_74 [[buffer(10)]], packed_float4 device* loads_74 [[buffer(12)]], packed_uint4 device* bonds_74 [[buffer(2)]], uint device* args_74 [[buffer(13)]], uint device* info_74 [[buffer(15)]], ScanParams_0 constant* scan_74 [[buffer(14)]], packed_float4 device* parts_74 [[buffer(3)]], packed_float4 device* colliders_74 [[buffer(8)]], packed_uint4 device* collider_info_74 [[buffer(9)]], packed_float4 device* view_chunks_74 [[buffer(17)]], ViewParams_0 constant* view_74 [[buffer(16)]], packed_float4 device* objects_74 [[buffer(18)]], uint device* shot_bodies_74 [[buffer(20)]], float device* lines_74 [[buffer(19)]], packed_float4 device* geometry_74 [[buffer(21)]], packed_float4 device* shots_74 [[buffer(22)]], packed_float4 device* lp_74 [[buffer(24)]], packed_float4 device* ext_76 [[buffer(23)]], packed_float4 device* engine_contacts_74 [[buffer(25)]], CrushParams_0 constant* crush_74 [[buffer(28)]], packed_float4 device* crush_rw_74 [[buffer(27)]], packed_float4 device* crush_tab_74 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_202;
    (&kernelContext_202)->params_0 = params_74;
    (&kernelContext_202)->atom_0 = atom_74;
    (&kernelContext_202)->work_0 = work_74;
    (&kernelContext_202)->frames_0 = frames_74;
    (&kernelContext_202)->bodies_0 = bodies_74;
    (&kernelContext_202)->chunks_0 = chunks_74;
    (&kernelContext_202)->sort_0 = sort_74;
    (&kernelContext_202)->manifolds_0 = manifolds_74;
    (&kernelContext_202)->loads_0 = loads_74;
    (&kernelContext_202)->bonds_0 = bonds_74;
    (&kernelContext_202)->args_0 = args_74;
    (&kernelContext_202)->info_0 = info_74;
    (&kernelContext_202)->scan_0 = scan_74;
    (&kernelContext_202)->parts_0 = parts_74;
    (&kernelContext_202)->colliders_0 = colliders_74;
    (&kernelContext_202)->collider_info_0 = collider_info_74;
    (&kernelContext_202)->view_chunks_0 = view_chunks_74;
    (&kernelContext_202)->view_0 = view_74;
    (&kernelContext_202)->objects_0 = objects_74;
    (&kernelContext_202)->shot_bodies_0 = shot_bodies_74;
    (&kernelContext_202)->lines_0 = lines_74;
    (&kernelContext_202)->geometry_0 = geometry_74;
    (&kernelContext_202)->shots_0 = shots_74;
    (&kernelContext_202)->lp_0 = lp_74;
    (&kernelContext_202)->ext_0 = ext_76;
    (&kernelContext_202)->engine_contacts_0 = engine_contacts_74;
    (&kernelContext_202)->crush_0 = crush_74;
    (&kernelContext_202)->crush_rw_0 = crush_rw_74;
    (&kernelContext_202)->crush_tab_0 = crush_tab_74;
    threadgroup array<uint, int(1024)> g_scan_74;
    (&kernelContext_202)->g_scan_0 = &g_scan_74;
    threadgroup array<float, int(64)> g_il_best_74;
    (&kernelContext_202)->g_il_best_0 = &g_il_best_74;
    threadgroup array<float, int(256)> g_gi_x_74;
    (&kernelContext_202)->g_gi_x_0 = &g_gi_x_74;
    threadgroup array<float, int(256)> g_gi_m_74;
    (&kernelContext_202)->g_gi_m_0 = &g_gi_m_74;
    threadgroup array<uint, int(1280)> g_gi_lo_74;
    (&kernelContext_202)->g_gi_lo_0 = &g_gi_lo_74;
    threadgroup array<uint, int(1280)> g_gi_hi_74;
    (&kernelContext_202)->g_gi_hi_0 = &g_gi_hi_74;
    uint r_26 = id_69.x;
    uint _S1326 = cs_rows_0(&kernelContext_202);
    if(r_26 >= _S1326)
    {
        return;
    }
    uint _S1327 = cs_row_0(r_26, &kernelContext_202);
    uint4 _S1328 = (as_type<uint4>((float4(*((&kernelContext_202)->engine_contacts_0+_S1327)) )));
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
        float4 _S1329 = float4(0.0f) ;
        bool _S1330 = side_8 == 0U;
        uint col_1;
        if(_S1330)
        {
            col_1 = _S1328.x;
        }
        else
        {
            col_1 = _S1328.y;
        }
        uint k_49;
        if((_S1328.w) != 0U)
        {
            k_49 = (uint4(*((&kernelContext_202)->collider_info_0+2U * col_1)) ).x;
        }
        else
        {
            k_49 = 4294967295U;
        }
        bool _S1331;
        if(k_49 != 4294967295U)
        {
            _S1331 = k_49 < ((&kernelContext_202)->params_0->chunk_count_0);
        }
        else
        {
            _S1331 = false;
        }
        bool _S1332;
        if(_S1331)
        {
            uint _S1333 = body_of_0(k_49, &kernelContext_202);
            bool _S1334 = dirty_0(_S1333, &kernelContext_202);
            _S1332 = _S1334;
        }
        else
        {
            _S1332 = false;
        }
        float4 saved_0;
        if(_S1332)
        {
            uint _S1335 = body_of_0(k_49, &kernelContext_202);
            uint at_21 = 7U * _S1335;
            float3 anchor_0;
            if(_S1330)
            {
                anchor_0 = (float4(*((&kernelContext_202)->engine_contacts_0+(_S1327 + 2U))) ).xyz;
            }
            else
            {
                anchor_0 = (float4(*((&kernelContext_202)->engine_contacts_0+(_S1327 + 3U))) ).xyz;
            }
            float3 _S1336 = (float4(*((&kernelContext_202)->bodies_0+at_21)) ).xyz;
            thread Quat_0 _S1337 = quat_of_0(float4(*((&kernelContext_202)->bodies_0+(at_21 + 1U))) );
            float3 _S1338 = rotate_0(&_S1337, anchor_0);
            saved_0 = float4(_S1336 + _S1338, 1.0f);
        }
        else
        {
            saved_0 = _S1329;
        }
        uint _S1339 = lp_anchor_0(r_26, side_8, &kernelContext_202);
        *((&kernelContext_202)->lp_0+_S1339) = packed_float4(saved_0) ;
        side_8 = side_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_clean(uint3 id_70 [[thread_position_in_grid]], FractureParams_0 constant* params_75 [[buffer(0)]], atomic<uint> device* atom_75 [[buffer(5)]], uint device* work_75 [[buffer(4)]], packed_float4 device* frames_75 [[buffer(6)]], packed_float4 device* bodies_75 [[buffer(7)]], packed_float4 device* chunks_75 [[buffer(1)]], uint device* sort_75 [[buffer(11)]], Record_natural_0 device* manifolds_75 [[buffer(10)]], packed_float4 device* loads_75 [[buffer(12)]], packed_uint4 device* bonds_75 [[buffer(2)]], uint device* args_75 [[buffer(13)]], uint device* info_75 [[buffer(15)]], ScanParams_0 constant* scan_75 [[buffer(14)]], packed_float4 device* parts_75 [[buffer(3)]], packed_float4 device* colliders_75 [[buffer(8)]], packed_uint4 device* collider_info_75 [[buffer(9)]], packed_float4 device* view_chunks_75 [[buffer(17)]], ViewParams_0 constant* view_75 [[buffer(16)]], packed_float4 device* objects_75 [[buffer(18)]], uint device* shot_bodies_75 [[buffer(20)]], float device* lines_75 [[buffer(19)]], packed_float4 device* geometry_75 [[buffer(21)]], packed_float4 device* shots_75 [[buffer(22)]], packed_float4 device* lp_75 [[buffer(24)]], packed_float4 device* ext_77 [[buffer(23)]], packed_float4 device* engine_contacts_75 [[buffer(25)]], CrushParams_0 constant* crush_75 [[buffer(28)]], packed_float4 device* crush_rw_75 [[buffer(27)]], packed_float4 device* crush_tab_75 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_203;
    (&kernelContext_203)->params_0 = params_75;
    (&kernelContext_203)->atom_0 = atom_75;
    (&kernelContext_203)->work_0 = work_75;
    (&kernelContext_203)->frames_0 = frames_75;
    (&kernelContext_203)->bodies_0 = bodies_75;
    (&kernelContext_203)->chunks_0 = chunks_75;
    (&kernelContext_203)->sort_0 = sort_75;
    (&kernelContext_203)->manifolds_0 = manifolds_75;
    (&kernelContext_203)->loads_0 = loads_75;
    (&kernelContext_203)->bonds_0 = bonds_75;
    (&kernelContext_203)->args_0 = args_75;
    (&kernelContext_203)->info_0 = info_75;
    (&kernelContext_203)->scan_0 = scan_75;
    (&kernelContext_203)->parts_0 = parts_75;
    (&kernelContext_203)->colliders_0 = colliders_75;
    (&kernelContext_203)->collider_info_0 = collider_info_75;
    (&kernelContext_203)->view_chunks_0 = view_chunks_75;
    (&kernelContext_203)->view_0 = view_75;
    (&kernelContext_203)->objects_0 = objects_75;
    (&kernelContext_203)->shot_bodies_0 = shot_bodies_75;
    (&kernelContext_203)->lines_0 = lines_75;
    (&kernelContext_203)->geometry_0 = geometry_75;
    (&kernelContext_203)->shots_0 = shots_75;
    (&kernelContext_203)->lp_0 = lp_75;
    (&kernelContext_203)->ext_0 = ext_77;
    (&kernelContext_203)->engine_contacts_0 = engine_contacts_75;
    (&kernelContext_203)->crush_0 = crush_75;
    (&kernelContext_203)->crush_rw_0 = crush_rw_75;
    (&kernelContext_203)->crush_tab_0 = crush_tab_75;
    threadgroup array<uint, int(1024)> g_scan_75;
    (&kernelContext_203)->g_scan_0 = &g_scan_75;
    threadgroup array<float, int(64)> g_il_best_75;
    (&kernelContext_203)->g_il_best_0 = &g_il_best_75;
    threadgroup array<float, int(256)> g_gi_x_75;
    (&kernelContext_203)->g_gi_x_0 = &g_gi_x_75;
    threadgroup array<float, int(256)> g_gi_m_75;
    (&kernelContext_203)->g_gi_m_0 = &g_gi_m_75;
    threadgroup array<uint, int(1280)> g_gi_lo_75;
    (&kernelContext_203)->g_gi_lo_0 = &g_gi_lo_75;
    threadgroup array<uint, int(1280)> g_gi_hi_75;
    (&kernelContext_203)->g_gi_hi_0 = &g_gi_hi_75;
    uint _S1340 = id_70.x;
    uint _S1341 = a_counters_1(&kernelContext_203);
    uint _S1342 = atomic_load_explicit((&kernelContext_203)->atom_0+(_S1341 + 18U), memory_order_relaxed);
    if(_S1340 >= _S1342)
    {
        return;
    }
    uint _S1343 = w_cs_bodies_0(&kernelContext_203);
    uint _S1344 = a_cs_0(*((&kernelContext_203)->work_0+(_S1343 + _S1340)), &kernelContext_203);
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
        atomic_store_explicit((&kernelContext_203)->atom_0+(_S1344 + w_8), 0U, memory_order_relaxed);
        w_8 = w_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_fix(uint3 id_71 [[thread_position_in_grid]], FractureParams_0 constant* params_76 [[buffer(0)]], atomic<uint> device* atom_76 [[buffer(5)]], uint device* work_76 [[buffer(4)]], packed_float4 device* frames_76 [[buffer(6)]], packed_float4 device* bodies_76 [[buffer(7)]], packed_float4 device* chunks_76 [[buffer(1)]], uint device* sort_76 [[buffer(11)]], Record_natural_0 device* manifolds_76 [[buffer(10)]], packed_float4 device* loads_76 [[buffer(12)]], packed_uint4 device* bonds_76 [[buffer(2)]], uint device* args_76 [[buffer(13)]], uint device* info_76 [[buffer(15)]], ScanParams_0 constant* scan_76 [[buffer(14)]], packed_float4 device* parts_76 [[buffer(3)]], packed_float4 device* colliders_76 [[buffer(8)]], packed_uint4 device* collider_info_76 [[buffer(9)]], packed_float4 device* view_chunks_76 [[buffer(17)]], ViewParams_0 constant* view_76 [[buffer(16)]], packed_float4 device* objects_76 [[buffer(18)]], uint device* shot_bodies_76 [[buffer(20)]], float device* lines_76 [[buffer(19)]], packed_float4 device* geometry_76 [[buffer(21)]], packed_float4 device* shots_76 [[buffer(22)]], packed_float4 device* lp_76 [[buffer(24)]], packed_float4 device* ext_78 [[buffer(23)]], packed_float4 device* engine_contacts_76 [[buffer(25)]], CrushParams_0 constant* crush_76 [[buffer(28)]], packed_float4 device* crush_rw_76 [[buffer(27)]], packed_float4 device* crush_tab_76 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_204;
    (&kernelContext_204)->params_0 = params_76;
    (&kernelContext_204)->atom_0 = atom_76;
    (&kernelContext_204)->work_0 = work_76;
    (&kernelContext_204)->frames_0 = frames_76;
    (&kernelContext_204)->bodies_0 = bodies_76;
    (&kernelContext_204)->chunks_0 = chunks_76;
    (&kernelContext_204)->sort_0 = sort_76;
    (&kernelContext_204)->manifolds_0 = manifolds_76;
    (&kernelContext_204)->loads_0 = loads_76;
    (&kernelContext_204)->bonds_0 = bonds_76;
    (&kernelContext_204)->args_0 = args_76;
    (&kernelContext_204)->info_0 = info_76;
    (&kernelContext_204)->scan_0 = scan_76;
    (&kernelContext_204)->parts_0 = parts_76;
    (&kernelContext_204)->colliders_0 = colliders_76;
    (&kernelContext_204)->collider_info_0 = collider_info_76;
    (&kernelContext_204)->view_chunks_0 = view_chunks_76;
    (&kernelContext_204)->view_0 = view_76;
    (&kernelContext_204)->objects_0 = objects_76;
    (&kernelContext_204)->shot_bodies_0 = shot_bodies_76;
    (&kernelContext_204)->lines_0 = lines_76;
    (&kernelContext_204)->geometry_0 = geometry_76;
    (&kernelContext_204)->shots_0 = shots_76;
    (&kernelContext_204)->lp_0 = lp_76;
    (&kernelContext_204)->ext_0 = ext_78;
    (&kernelContext_204)->engine_contacts_0 = engine_contacts_76;
    (&kernelContext_204)->crush_0 = crush_76;
    (&kernelContext_204)->crush_rw_0 = crush_rw_76;
    (&kernelContext_204)->crush_tab_0 = crush_tab_76;
    threadgroup array<uint, int(1024)> g_scan_76;
    (&kernelContext_204)->g_scan_0 = &g_scan_76;
    threadgroup array<float, int(64)> g_il_best_76;
    (&kernelContext_204)->g_il_best_0 = &g_il_best_76;
    threadgroup array<float, int(256)> g_gi_x_76;
    (&kernelContext_204)->g_gi_x_0 = &g_gi_x_76;
    threadgroup array<float, int(256)> g_gi_m_76;
    (&kernelContext_204)->g_gi_m_0 = &g_gi_m_76;
    threadgroup array<uint, int(1280)> g_gi_lo_76;
    (&kernelContext_204)->g_gi_lo_0 = &g_gi_lo_76;
    threadgroup array<uint, int(1280)> g_gi_hi_76;
    (&kernelContext_204)->g_gi_hi_0 = &g_gi_hi_76;
    uint r_27 = id_71.x;
    uint _S1345 = cs_rows_0(&kernelContext_204);
    if(r_27 >= _S1345)
    {
        return;
    }
    uint _S1346 = cs_row_0(r_27, &kernelContext_204);
    uint4 _S1347 = (as_type<uint4>((float4(*((&kernelContext_204)->engine_contacts_0+_S1346)) )));
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
        uint _S1348 = lp_anchor_0(r_27, side_9, &kernelContext_204);
        float4 _S1349 = float4(*((&kernelContext_204)->lp_0+_S1348)) ;
        if((_S1349.w) == 0.0f)
        {
            side_9 = side_9 + 1U;
            continue;
        }
        uint col_2;
        if(side_9 == 0U)
        {
            col_2 = _S1347.x;
        }
        else
        {
            col_2 = _S1347.y;
        }
        uint at_22 = 7U * (as_type<uint>(((float4(*((&kernelContext_204)->colliders_0+4U * col_2)) ).x)));
        thread Quat_0 _S1350 = quat_of_0(float4(*((&kernelContext_204)->bodies_0+(at_22 + 1U))) );
        Quat_0 _S1351 = conj_0(&_S1350);
        float3 _S1352 = _S1349.xyz - (float4(*((&kernelContext_204)->bodies_0+at_22)) ).xyz;
        thread Quat_0 _S1353 = _S1351;
        float3 _S1354 = rotate_0(&_S1353, _S1352);
        (*((&kernelContext_204)->engine_contacts_0+(_S1346 + 2U + side_9))).xyz = _S1354;
        moved_0 = true;
        side_9 = side_9 + 1U;
    }
    if(moved_0)
    {
        (*((&kernelContext_204)->engine_contacts_0+(_S1346 + 4U)))[int(3)] = 0.0f;
        (*((&kernelContext_204)->engine_contacts_0+(_S1346 + 5U)))[int(3)] = 0.0f;
        uint _S1355 = _S1346 + 6U;
        *((&kernelContext_204)->engine_contacts_0+_S1355) = packed_float4(float4(1.0f, 1.0f, 0.0f, (float4(*((&kernelContext_204)->engine_contacts_0+_S1355)) ).w)) ;
        (*((&kernelContext_204)->engine_contacts_0+(_S1346 + 7U)))[int(1)] = 0.0f;
    }
    return;
}

[[kernel]] void fr_cs_count(uint3 id_72 [[thread_position_in_grid]], FractureParams_0 constant* params_77 [[buffer(0)]], atomic<uint> device* atom_77 [[buffer(5)]], uint device* work_77 [[buffer(4)]], packed_float4 device* frames_77 [[buffer(6)]], packed_float4 device* bodies_77 [[buffer(7)]], packed_float4 device* chunks_77 [[buffer(1)]], uint device* sort_77 [[buffer(11)]], Record_natural_0 device* manifolds_77 [[buffer(10)]], packed_float4 device* loads_77 [[buffer(12)]], packed_uint4 device* bonds_77 [[buffer(2)]], uint device* args_77 [[buffer(13)]], uint device* info_77 [[buffer(15)]], ScanParams_0 constant* scan_77 [[buffer(14)]], packed_float4 device* parts_77 [[buffer(3)]], packed_float4 device* colliders_77 [[buffer(8)]], packed_uint4 device* collider_info_77 [[buffer(9)]], packed_float4 device* view_chunks_77 [[buffer(17)]], ViewParams_0 constant* view_77 [[buffer(16)]], packed_float4 device* objects_77 [[buffer(18)]], uint device* shot_bodies_77 [[buffer(20)]], float device* lines_77 [[buffer(19)]], packed_float4 device* geometry_77 [[buffer(21)]], packed_float4 device* shots_77 [[buffer(22)]], packed_float4 device* lp_77 [[buffer(24)]], packed_float4 device* ext_79 [[buffer(23)]], packed_float4 device* engine_contacts_77 [[buffer(25)]], CrushParams_0 constant* crush_77 [[buffer(28)]], packed_float4 device* crush_rw_77 [[buffer(27)]], packed_float4 device* crush_tab_77 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_205;
    (&kernelContext_205)->params_0 = params_77;
    (&kernelContext_205)->atom_0 = atom_77;
    (&kernelContext_205)->work_0 = work_77;
    (&kernelContext_205)->frames_0 = frames_77;
    (&kernelContext_205)->bodies_0 = bodies_77;
    (&kernelContext_205)->chunks_0 = chunks_77;
    (&kernelContext_205)->sort_0 = sort_77;
    (&kernelContext_205)->manifolds_0 = manifolds_77;
    (&kernelContext_205)->loads_0 = loads_77;
    (&kernelContext_205)->bonds_0 = bonds_77;
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
    threadgroup array<float, int(256)> g_gi_x_77;
    (&kernelContext_205)->g_gi_x_0 = &g_gi_x_77;
    threadgroup array<float, int(256)> g_gi_m_77;
    (&kernelContext_205)->g_gi_m_0 = &g_gi_m_77;
    threadgroup array<uint, int(1280)> g_gi_lo_77;
    (&kernelContext_205)->g_gi_lo_0 = &g_gi_lo_77;
    threadgroup array<uint, int(1280)> g_gi_hi_77;
    (&kernelContext_205)->g_gi_hi_0 = &g_gi_hi_77;
    uint k_50 = id_72.x;
    if(k_50 >= (params_77->chunk_count_0))
    {
        return;
    }
    uint _S1356 = body_of_0(k_50, &kernelContext_205);
    bool _S1357 = cs_struck_1(_S1356, &kernelContext_205);
    if(!_S1357)
    {
        return;
    }
    uint _S1358 = a_cs_0(_S1356, &kernelContext_205);
    uint _S1359 = atomic_fetch_max_explicit((&kernelContext_205)->atom_0+(_S1358 + 3U), ~k_50, memory_order_relaxed);
    if(((&kernelContext_205)->info_0[k_50] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_205)->atom_0+(_S1358 + 4U), 1U, memory_order_relaxed);
    }
    uint _S1360 = atomic_fetch_add_explicit((&kernelContext_205)->atom_0+(_S1358 + 5U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_base(uint3 id_73 [[thread_position_in_grid]], FractureParams_0 constant* params_78 [[buffer(0)]], atomic<uint> device* atom_78 [[buffer(5)]], uint device* work_78 [[buffer(4)]], packed_float4 device* frames_78 [[buffer(6)]], packed_float4 device* bodies_78 [[buffer(7)]], packed_float4 device* chunks_78 [[buffer(1)]], uint device* sort_78 [[buffer(11)]], Record_natural_0 device* manifolds_78 [[buffer(10)]], packed_float4 device* loads_78 [[buffer(12)]], packed_uint4 device* bonds_78 [[buffer(2)]], uint device* args_78 [[buffer(13)]], uint device* info_78 [[buffer(15)]], ScanParams_0 constant* scan_78 [[buffer(14)]], packed_float4 device* parts_78 [[buffer(3)]], packed_float4 device* colliders_78 [[buffer(8)]], packed_uint4 device* collider_info_78 [[buffer(9)]], packed_float4 device* view_chunks_78 [[buffer(17)]], ViewParams_0 constant* view_78 [[buffer(16)]], packed_float4 device* objects_78 [[buffer(18)]], uint device* shot_bodies_78 [[buffer(20)]], float device* lines_78 [[buffer(19)]], packed_float4 device* geometry_78 [[buffer(21)]], packed_float4 device* shots_78 [[buffer(22)]], packed_float4 device* lp_78 [[buffer(24)]], packed_float4 device* ext_80 [[buffer(23)]], packed_float4 device* engine_contacts_78 [[buffer(25)]], CrushParams_0 constant* crush_78 [[buffer(28)]], packed_float4 device* crush_rw_78 [[buffer(27)]], packed_float4 device* crush_tab_78 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_206;
    (&kernelContext_206)->params_0 = params_78;
    (&kernelContext_206)->atom_0 = atom_78;
    (&kernelContext_206)->work_0 = work_78;
    (&kernelContext_206)->frames_0 = frames_78;
    (&kernelContext_206)->bodies_0 = bodies_78;
    (&kernelContext_206)->chunks_0 = chunks_78;
    (&kernelContext_206)->sort_0 = sort_78;
    (&kernelContext_206)->manifolds_0 = manifolds_78;
    (&kernelContext_206)->loads_0 = loads_78;
    (&kernelContext_206)->bonds_0 = bonds_78;
    (&kernelContext_206)->args_0 = args_78;
    (&kernelContext_206)->info_0 = info_78;
    (&kernelContext_206)->scan_0 = scan_78;
    (&kernelContext_206)->parts_0 = parts_78;
    (&kernelContext_206)->colliders_0 = colliders_78;
    (&kernelContext_206)->collider_info_0 = collider_info_78;
    (&kernelContext_206)->view_chunks_0 = view_chunks_78;
    (&kernelContext_206)->view_0 = view_78;
    (&kernelContext_206)->objects_0 = objects_78;
    (&kernelContext_206)->shot_bodies_0 = shot_bodies_78;
    (&kernelContext_206)->lines_0 = lines_78;
    (&kernelContext_206)->geometry_0 = geometry_78;
    (&kernelContext_206)->shots_0 = shots_78;
    (&kernelContext_206)->lp_0 = lp_78;
    (&kernelContext_206)->ext_0 = ext_80;
    (&kernelContext_206)->engine_contacts_0 = engine_contacts_78;
    (&kernelContext_206)->crush_0 = crush_78;
    (&kernelContext_206)->crush_rw_0 = crush_rw_78;
    (&kernelContext_206)->crush_tab_0 = crush_tab_78;
    threadgroup array<uint, int(1024)> g_scan_78;
    (&kernelContext_206)->g_scan_0 = &g_scan_78;
    threadgroup array<float, int(64)> g_il_best_78;
    (&kernelContext_206)->g_il_best_0 = &g_il_best_78;
    threadgroup array<float, int(256)> g_gi_x_78;
    (&kernelContext_206)->g_gi_x_0 = &g_gi_x_78;
    threadgroup array<float, int(256)> g_gi_m_78;
    (&kernelContext_206)->g_gi_m_0 = &g_gi_m_78;
    threadgroup array<uint, int(1280)> g_gi_lo_78;
    (&kernelContext_206)->g_gi_lo_0 = &g_gi_lo_78;
    threadgroup array<uint, int(1280)> g_gi_hi_78;
    (&kernelContext_206)->g_gi_hi_0 = &g_gi_hi_78;
    uint _S1361 = id_73.x;
    uint _S1362 = a_counters_1(&kernelContext_206);
    uint _S1363 = atomic_load_explicit((&kernelContext_206)->atom_0+(_S1362 + 18U), memory_order_relaxed);
    if(_S1361 >= _S1363)
    {
        return;
    }
    uint _S1364 = w_cs_bodies_0(&kernelContext_206);
    uint _S1365 = a_cs_0(*((&kernelContext_206)->work_0+(_S1364 + _S1361)), &kernelContext_206);
    atomic<uint> device* _S1366 = (&kernelContext_206)->atom_0+(_S1365 + 6U);
    atomic<uint> device* _S1367 = (&kernelContext_206)->atom_0+(_S1362 + 19U);
    uint _S1368 = atomic_load_explicit((&kernelContext_206)->atom_0+(_S1365 + 5U), memory_order_relaxed);
    uint _S1369 = atomic_fetch_add_explicit(_S1367, _S1368, memory_order_relaxed);
    atomic_store_explicit(_S1366, _S1369, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_collect(uint3 id_74 [[thread_position_in_grid]], FractureParams_0 constant* params_79 [[buffer(0)]], atomic<uint> device* atom_79 [[buffer(5)]], uint device* work_79 [[buffer(4)]], packed_float4 device* frames_79 [[buffer(6)]], packed_float4 device* bodies_79 [[buffer(7)]], packed_float4 device* chunks_79 [[buffer(1)]], uint device* sort_79 [[buffer(11)]], Record_natural_0 device* manifolds_79 [[buffer(10)]], packed_float4 device* loads_79 [[buffer(12)]], packed_uint4 device* bonds_79 [[buffer(2)]], uint device* args_79 [[buffer(13)]], uint device* info_79 [[buffer(15)]], ScanParams_0 constant* scan_79 [[buffer(14)]], packed_float4 device* parts_79 [[buffer(3)]], packed_float4 device* colliders_79 [[buffer(8)]], packed_uint4 device* collider_info_79 [[buffer(9)]], packed_float4 device* view_chunks_79 [[buffer(17)]], ViewParams_0 constant* view_79 [[buffer(16)]], packed_float4 device* objects_79 [[buffer(18)]], uint device* shot_bodies_79 [[buffer(20)]], float device* lines_79 [[buffer(19)]], packed_float4 device* geometry_79 [[buffer(21)]], packed_float4 device* shots_79 [[buffer(22)]], packed_float4 device* lp_79 [[buffer(24)]], packed_float4 device* ext_81 [[buffer(23)]], packed_float4 device* engine_contacts_79 [[buffer(25)]], CrushParams_0 constant* crush_79 [[buffer(28)]], packed_float4 device* crush_rw_79 [[buffer(27)]], packed_float4 device* crush_tab_79 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_207;
    (&kernelContext_207)->params_0 = params_79;
    (&kernelContext_207)->atom_0 = atom_79;
    (&kernelContext_207)->work_0 = work_79;
    (&kernelContext_207)->frames_0 = frames_79;
    (&kernelContext_207)->bodies_0 = bodies_79;
    (&kernelContext_207)->chunks_0 = chunks_79;
    (&kernelContext_207)->sort_0 = sort_79;
    (&kernelContext_207)->manifolds_0 = manifolds_79;
    (&kernelContext_207)->loads_0 = loads_79;
    (&kernelContext_207)->bonds_0 = bonds_79;
    (&kernelContext_207)->args_0 = args_79;
    (&kernelContext_207)->info_0 = info_79;
    (&kernelContext_207)->scan_0 = scan_79;
    (&kernelContext_207)->parts_0 = parts_79;
    (&kernelContext_207)->colliders_0 = colliders_79;
    (&kernelContext_207)->collider_info_0 = collider_info_79;
    (&kernelContext_207)->view_chunks_0 = view_chunks_79;
    (&kernelContext_207)->view_0 = view_79;
    (&kernelContext_207)->objects_0 = objects_79;
    (&kernelContext_207)->shot_bodies_0 = shot_bodies_79;
    (&kernelContext_207)->lines_0 = lines_79;
    (&kernelContext_207)->geometry_0 = geometry_79;
    (&kernelContext_207)->shots_0 = shots_79;
    (&kernelContext_207)->lp_0 = lp_79;
    (&kernelContext_207)->ext_0 = ext_81;
    (&kernelContext_207)->engine_contacts_0 = engine_contacts_79;
    (&kernelContext_207)->crush_0 = crush_79;
    (&kernelContext_207)->crush_rw_0 = crush_rw_79;
    (&kernelContext_207)->crush_tab_0 = crush_tab_79;
    threadgroup array<uint, int(1024)> g_scan_79;
    (&kernelContext_207)->g_scan_0 = &g_scan_79;
    threadgroup array<float, int(64)> g_il_best_79;
    (&kernelContext_207)->g_il_best_0 = &g_il_best_79;
    threadgroup array<float, int(256)> g_gi_x_79;
    (&kernelContext_207)->g_gi_x_0 = &g_gi_x_79;
    threadgroup array<float, int(256)> g_gi_m_79;
    (&kernelContext_207)->g_gi_m_0 = &g_gi_m_79;
    threadgroup array<uint, int(1280)> g_gi_lo_79;
    (&kernelContext_207)->g_gi_lo_0 = &g_gi_lo_79;
    threadgroup array<uint, int(1280)> g_gi_hi_79;
    (&kernelContext_207)->g_gi_hi_0 = &g_gi_hi_79;
    uint k_51 = id_74.x;
    if(k_51 >= (params_79->chunk_count_0))
    {
        return;
    }
    uint _S1370 = body_of_0(k_51, &kernelContext_207);
    bool _S1371 = cs_struck_1(_S1370, &kernelContext_207);
    if(!_S1371)
    {
        return;
    }
    uint _S1372 = w_cs_list_0(&kernelContext_207);
    uint _S1373 = a_cs_0(_S1370, &kernelContext_207);
    uint _S1374 = atomic_load_explicit((&kernelContext_207)->atom_0+(_S1373 + 6U), memory_order_relaxed);
    uint _S1375 = _S1372 + _S1374;
    uint _S1376 = atomic_fetch_add_explicit((&kernelContext_207)->atom_0+(_S1373 + 7U), 1U, memory_order_relaxed);
    *((&kernelContext_207)->work_0+(_S1375 + _S1376)) = k_51;
    return;
}

void cs_load_0(uint k_52, uint s_22, KernelContext_0 thread* kernelContext_208)
{
    uint _S1377 = a_cs_0(s_22, kernelContext_208);
    uint _S1378 = atomic_load_explicit(kernelContext_208->atom_0+(_S1377 + 4U), memory_order_relaxed);
    bool supported_1 = _S1378 != 0U;
    bool fixed_2;
    if((kernelContext_208->info_0[k_52] & 1U) != 0U)
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            uint _S1379 = atomic_load_explicit(kernelContext_208->atom_0+(_S1377 + 3U), memory_order_relaxed);
            fixed_2 = (~_S1379) == k_52;
        }
        else
        {
            fixed_2 = false;
        }
    }
    uint _S1380 = lp_body_0(s_22, kernelContext_208);
    float4 _S1381 = float4(*(kernelContext_208->lp_0+(_S1380 + 2U))) ;
    float4 _S1382 = float4(*(kernelContext_208->lp_0+(_S1380 + 3U))) ;
    uint _S1383 = 7U * s_22;
    thread Quat_0 q_28;
    thread float3 origin_8;
    structure_pose_0(s_22, _S1383, &q_28, &origin_8, kernelContext_208);
    thread Quat_0 _S1384 = q_28;
    Quat_0 _S1385 = conj_0(&_S1384);
    float3 _S1386 = _S1382.xyz * float3(_S1381.w) ;
    thread Quat_0 _S1387 = _S1385;
    float3 _S1388 = rotate_0(&_S1387, _S1386);
    float3 _S1389 = _S1381.xyz - origin_8;
    thread Quat_0 _S1390 = _S1385;
    float3 _S1391 = rotate_0(&_S1390, _S1389);
    uint _S1392 = 4U * k_52;
    float4 _S1393 = float4(*(kernelContext_208->chunks_0+_S1392)) ;
    float3 _S1394 = float3(0.0f) ;
    uint _S1395 = atomic_load_explicit(kernelContext_208->atom_0+(_S1377 + 1U), memory_order_relaxed);
    float3 f_12;
    float3 t_13;
    if(_S1395 == (k_52 + 1U))
    {
        float3 _S1396 = cross(_S1391 - _S1393.xyz, _S1388);
        f_12 = _S1388;
        t_13 = _S1396;
    }
    else
    {
        f_12 = _S1394;
        t_13 = _S1394;
    }
    float inv_mass_7 = (float4(*(kernelContext_208->bodies_0+_S1383)) ).w;
    bool _S1397;
    if(!supported_1)
    {
        _S1397 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S1397 = false;
    }
    if(_S1397)
    {
        uint _S1398 = 2U * s_22;
        float3 com_2 = (float4(*(kernelContext_208->frames_0+_S1398)) ).xyz;
        Quat_0 p_20 = quat_of_0(float4(*(kernelContext_208->frames_0+(_S1398 + 1U))) );
        float3 _S1399 = (float4(*(kernelContext_208->bodies_0+(_S1383 + 2U))) ).xyz;
        thread Quat_0 _S1400 = p_20;
        Quat_0 _S1401 = conj_0(&_S1400);
        float3 _S1402 = cross(_S1391 - com_2, _S1388);
        thread Quat_0 _S1403 = _S1401;
        float3 _S1404 = rotate_0(&_S1403, _S1402);
        float3 _S1405 = _S1399 * _S1404;
        thread Quat_0 _S1406 = p_20;
        float3 _S1407 = rotate_0(&_S1406, _S1405);
        float4 _S1408 = float4(*(kernelContext_208->chunks_0+(_S1392 + 1U))) ;
        float4 _S1409 = float4(*(kernelContext_208->chunks_0+(_S1392 + 2U))) ;
        float _S1410 = _S1407.x;
        float _S1411 = _S1408.w;
        float _S1412 = _S1407.y;
        float _S1413 = _S1409.x;
        float _S1414 = _S1407.z;
        float _S1415 = _S1409.y;
        float3 t_14 = t_13 - float3(_S1408.x * _S1410 + _S1411 * _S1412 + _S1413 * _S1414, _S1411 * _S1410 + _S1408.y * _S1412 + _S1415 * _S1414, _S1413 * _S1410 + _S1415 * _S1412 + _S1408.z * _S1414);
        f_12 = f_12 - float3(_S1393.w)  * (_S1388 * float3(inv_mass_7)  + cross(_S1407, _S1393.xyz - com_2));
        t_13 = t_14;
    }
    uint _S1416 = st_f_0(k_52, 14U);
    packed_float4 device* _S1417 = kernelContext_208->lp_0+_S1416;
    float _S1418;
    if(fixed_2)
    {
        _S1418 = 1.0f;
    }
    else
    {
        _S1418 = 0.0f;
    }
    *_S1417 = packed_float4(float4(f_12, _S1418)) ;
    *(kernelContext_208->lp_0+(_S1416 + 1U)) = packed_float4(float4(t_13, 0.0f)) ;
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
        uint _S1419 = st_x_0(k_52, 14U, c_24);
        packed_float4 _S1420 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_208->lp_0+_S1419) = _S1420;
        *(kernelContext_208->lp_0+(_S1419 + 1U)) = _S1420;
        c_24 = c_24 + 1U;
    }
    uint _S1421 = st_d_0(k_52, 14U);
    packed_float4 _S1422 = packed_float4(float4(0.0f) ) ;
    *(kernelContext_208->lp_0+_S1421) = _S1422;
    *(kernelContext_208->lp_0+(_S1421 + 1U)) = _S1422;
    return;
}

void cs_break_0(uint i_24, uint s_23, KernelContext_0 thread* kernelContext_209)
{
    uint4 _S1423 = uint4(*(kernelContext_209->bonds_0+i_24)) ;
    LpBond_0 _S1424 = lp_bond_0(i_24, kernelContext_209);
    uint _S1425 = _S1423.x;
    uint _S1426 = _S1423.y;
    thread LpBond_0 _S1427 = _S1424;
    thread float3 q_lin_6;
    thread float3 q_ang_6;
    st_force_0(&_S1427, _S1425, _S1426, 14U, 0U, &q_lin_6, &q_ang_6, kernelContext_209);
    if((kernelContext_209->params_0->load_model_0) == 1U)
    {
        thread LpBond_0 _S1428 = _S1424;
        thread float3 b_lin_0;
        thread float3 b_ang_0;
        lp_force_0(&_S1428, _S1425, _S1426, 0U, &b_lin_0, &b_ang_0, kernelContext_209);
        q_lin_6 = q_lin_6 + b_lin_0;
        q_ang_6 = q_ang_6 + b_ang_0;
    }
    thread JointBond_0 _S1429 = _S1424.law_0;
    Measures_0 _S1430 = stress_measures_0(&_S1429, q_lin_6, q_ang_6);
    JointMaterial_0 _S1431 = lp_material_0(_S1424.law_0.ids_3.x, kernelContext_209);
    float _S1432 = _S1424.law_0.geom1_0.w;
    thread JointMaterial_0 _S1433 = _S1431;
    thread JointBond_0 _S1434 = _S1424.law_0;
    thread Measures_0 _S1435 = _S1430;
    float4 _S1436 = failure_indices_0(&_S1433, &_S1434, &_S1435, _S1432);
    if((max(max(_S1436.x, _S1436.y), max(_S1436.z, _S1436.w))) >= 1.0f)
    {
        uint _S1437 = w_alive_0(kernelContext_209);
        *(kernelContext_209->work_0+(_S1437 + i_24)) = 0U;
        uint _S1438 = a_dirty_0(kernelContext_209);
        uint _S1439 = atomic_fetch_or_explicit(kernelContext_209->atom_0+(_S1438 + s_23), 1U, memory_order_relaxed);
        uint _S1440 = a_counters_1(kernelContext_209);
        uint _S1441 = atomic_fetch_add_explicit(kernelContext_209->atom_0+_S1440, 1U, memory_order_relaxed);
        uint _S1442 = atomic_fetch_add_explicit(kernelContext_209->atom_0+(_S1440 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_24, kernelContext_209);
        uint _S1443 = atomic_fetch_add_explicit(kernelContext_209->atom_0+(_S1440 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_solve(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_80 [[buffer(0)]], atomic<uint> device* atom_80 [[buffer(5)]], uint device* work_80 [[buffer(4)]], packed_float4 device* frames_80 [[buffer(6)]], packed_float4 device* bodies_80 [[buffer(7)]], packed_float4 device* chunks_80 [[buffer(1)]], uint device* sort_80 [[buffer(11)]], Record_natural_0 device* manifolds_80 [[buffer(10)]], packed_float4 device* loads_80 [[buffer(12)]], packed_uint4 device* bonds_80 [[buffer(2)]], uint device* args_80 [[buffer(13)]], uint device* info_80 [[buffer(15)]], ScanParams_0 constant* scan_80 [[buffer(14)]], packed_float4 device* parts_80 [[buffer(3)]], packed_float4 device* colliders_80 [[buffer(8)]], packed_uint4 device* collider_info_80 [[buffer(9)]], packed_float4 device* view_chunks_80 [[buffer(17)]], ViewParams_0 constant* view_80 [[buffer(16)]], packed_float4 device* objects_80 [[buffer(18)]], uint device* shot_bodies_80 [[buffer(20)]], float device* lines_80 [[buffer(19)]], packed_float4 device* geometry_80 [[buffer(21)]], packed_float4 device* shots_80 [[buffer(22)]], packed_float4 device* lp_80 [[buffer(24)]], packed_float4 device* ext_82 [[buffer(23)]], packed_float4 device* engine_contacts_80 [[buffer(25)]], CrushParams_0 constant* crush_80 [[buffer(28)]], packed_float4 device* crush_rw_80 [[buffer(27)]], packed_float4 device* crush_tab_80 [[buffer(26)]])
{
    bool _S1444;
    uint j3_0;
    thread KernelContext_0 kernelContext_210;
    (&kernelContext_210)->params_0 = params_80;
    (&kernelContext_210)->atom_0 = atom_80;
    (&kernelContext_210)->work_0 = work_80;
    (&kernelContext_210)->frames_0 = frames_80;
    (&kernelContext_210)->bodies_0 = bodies_80;
    (&kernelContext_210)->chunks_0 = chunks_80;
    (&kernelContext_210)->sort_0 = sort_80;
    (&kernelContext_210)->manifolds_0 = manifolds_80;
    (&kernelContext_210)->loads_0 = loads_80;
    (&kernelContext_210)->bonds_0 = bonds_80;
    (&kernelContext_210)->args_0 = args_80;
    (&kernelContext_210)->info_0 = info_80;
    (&kernelContext_210)->scan_0 = scan_80;
    (&kernelContext_210)->parts_0 = parts_80;
    (&kernelContext_210)->colliders_0 = colliders_80;
    (&kernelContext_210)->collider_info_0 = collider_info_80;
    (&kernelContext_210)->view_chunks_0 = view_chunks_80;
    (&kernelContext_210)->view_0 = view_80;
    (&kernelContext_210)->objects_0 = objects_80;
    (&kernelContext_210)->shot_bodies_0 = shot_bodies_80;
    (&kernelContext_210)->lines_0 = lines_80;
    (&kernelContext_210)->geometry_0 = geometry_80;
    (&kernelContext_210)->shots_0 = shots_80;
    (&kernelContext_210)->lp_0 = lp_80;
    (&kernelContext_210)->ext_0 = ext_82;
    (&kernelContext_210)->engine_contacts_0 = engine_contacts_80;
    (&kernelContext_210)->crush_0 = crush_80;
    (&kernelContext_210)->crush_rw_0 = crush_rw_80;
    (&kernelContext_210)->crush_tab_0 = crush_tab_80;
    threadgroup array<uint, int(1024)> g_scan_80;
    (&kernelContext_210)->g_scan_0 = &g_scan_80;
    threadgroup array<float, int(64)> g_il_best_80;
    (&kernelContext_210)->g_il_best_0 = &g_il_best_80;
    threadgroup array<float, int(256)> g_gi_x_80;
    (&kernelContext_210)->g_gi_x_0 = &g_gi_x_80;
    threadgroup array<float, int(256)> g_gi_m_80;
    (&kernelContext_210)->g_gi_m_0 = &g_gi_m_80;
    threadgroup array<uint, int(1280)> g_gi_lo_80;
    (&kernelContext_210)->g_gi_lo_0 = &g_gi_lo_80;
    threadgroup array<uint, int(1280)> g_gi_hi_80;
    (&kernelContext_210)->g_gi_hi_0 = &g_gi_hi_80;
    uint _S1445 = w_cs_bodies_0(&kernelContext_210);
    uint device* _S1446 = (&kernelContext_210)->work_0+(_S1445 + group_3.x);
    uint s_24 = *_S1446;
    uint t_15 = thread_6.x;
    uint _S1447 = a_cs_0(*_S1446, &kernelContext_210);
    uint _S1448 = atomic_load_explicit((&kernelContext_210)->atom_0+(_S1447 + 6U), memory_order_relaxed);
    uint _S1449 = atomic_load_explicit((&kernelContext_210)->atom_0+(_S1447 + 5U), memory_order_relaxed);
    uint j_11 = t_15;
    for(;;)
    {
        if(j_11 < _S1449)
        {
        }
        else
        {
            break;
        }
        uint _S1450 = w_cs_list_0(&kernelContext_210);
        cs_load_0(*((&kernelContext_210)->work_0+(_S1450 + _S1448 + j_11)), s_24, &kernelContext_210);
        j_11 = j_11 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint j2_1 = t_15;
    for(;;)
    {
        if(j2_1 < _S1449)
        {
        }
        else
        {
            break;
        }
        uint _S1451 = w_cs_list_0(&kernelContext_210);
        uint device* _S1452 = (&kernelContext_210)->work_0+(_S1451 + _S1448 + j2_1);
        uint k_53 = *_S1452;
        uint _S1453 = st_f_0(*_S1452, 14U);
        if(((float4(*((&kernelContext_210)->lp_0+_S1453)) ).w) == 0.0f)
        {
            bool _S1454 = lp_factor_0(k_53, &kernelContext_210);
            _S1444 = !_S1454;
        }
        else
        {
            _S1444 = false;
        }
        if(_S1444)
        {
            (*((&kernelContext_210)->lp_0+_S1453))[int(3)] = 1.0f;
            uint _S1455 = a_counters_1(&kernelContext_210);
            uint _S1456 = atomic_fetch_or_explicit((&kernelContext_210)->atom_0+(_S1455 + 3U), 64U, memory_order_relaxed);
        }
        j2_1 = j2_1 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint sweep_0 = 0U;
    for(;;)
    {
        if(sweep_0 < ((&kernelContext_210)->params_0->cs_sweeps_0))
        {
        }
        else
        {
            break;
        }
        j3_0 = t_15;
        for(;;)
        {
            if(j3_0 < _S1449)
            {
            }
            else
            {
                break;
            }
            uint _S1457 = w_cs_list_0(&kernelContext_210);
            st_sweep_0(*((&kernelContext_210)->work_0+(_S1457 + _S1448 + j3_0)), 14U, sweep_0 % 2U, (&kernelContext_210)->params_0->cs_c1_0, (&kernelContext_210)->params_0->cs_c2_0, &kernelContext_210);
            j3_0 = j3_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        sweep_0 = sweep_0 + 1U;
    }
    j3_0 = t_15;
    for(;;)
    {
        if(j3_0 < _S1449)
        {
        }
        else
        {
            break;
        }
        uint _S1458 = w_cs_list_0(&kernelContext_210);
        uint device* _S1459 = (&kernelContext_210)->work_0+(_S1458 + _S1448 + j3_0);
        uint k_54 = *_S1459;
        float4 _S1460 = float4(*((&kernelContext_210)->ext_0+((&kernelContext_210)->params_0->ext_chunks_0 + 4U * *_S1459))) ;
        uint _S1461 = (as_type<uint>((_S1460.x)));
        uint _S1462 = (as_type<uint>((_S1460.y)));
        uint e_7 = 0U;
        for(;;)
        {
            if(e_7 < _S1462)
            {
            }
            else
            {
                break;
            }
            uint _S1463 = adj_bond_0(_S1461 + e_7, &kernelContext_210);
            if(((uint4(*((&kernelContext_210)->bonds_0+_S1463)) ).x) == k_54)
            {
                uint _S1464 = w_alive_0(&kernelContext_210);
                _S1444 = (*((&kernelContext_210)->work_0+(_S1464 + _S1463))) != 0U;
            }
            else
            {
                _S1444 = false;
            }
            if(_S1444)
            {
                cs_break_0(_S1463, s_24, &kernelContext_210);
            }
            e_7 = e_7 + 1U;
        }
        j3_0 = j3_0 + 256U;
    }
    return;
}

[[kernel]] void fr_gi_begin(uint3 id_75 [[thread_position_in_grid]], FractureParams_0 constant* params_81 [[buffer(0)]], atomic<uint> device* atom_81 [[buffer(5)]], uint device* work_81 [[buffer(4)]], packed_float4 device* frames_81 [[buffer(6)]], packed_float4 device* bodies_81 [[buffer(7)]], packed_float4 device* chunks_81 [[buffer(1)]], uint device* sort_81 [[buffer(11)]], Record_natural_0 device* manifolds_81 [[buffer(10)]], packed_float4 device* loads_81 [[buffer(12)]], packed_uint4 device* bonds_81 [[buffer(2)]], uint device* args_81 [[buffer(13)]], uint device* info_81 [[buffer(15)]], ScanParams_0 constant* scan_81 [[buffer(14)]], packed_float4 device* parts_81 [[buffer(3)]], packed_float4 device* colliders_81 [[buffer(8)]], packed_uint4 device* collider_info_81 [[buffer(9)]], packed_float4 device* view_chunks_81 [[buffer(17)]], ViewParams_0 constant* view_81 [[buffer(16)]], packed_float4 device* objects_81 [[buffer(18)]], uint device* shot_bodies_81 [[buffer(20)]], float device* lines_81 [[buffer(19)]], packed_float4 device* geometry_81 [[buffer(21)]], packed_float4 device* shots_81 [[buffer(22)]], packed_float4 device* lp_81 [[buffer(24)]], packed_float4 device* ext_83 [[buffer(23)]], packed_float4 device* engine_contacts_81 [[buffer(25)]], CrushParams_0 constant* crush_81 [[buffer(28)]], packed_float4 device* crush_rw_81 [[buffer(27)]], packed_float4 device* crush_tab_81 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_211;
    (&kernelContext_211)->params_0 = params_81;
    (&kernelContext_211)->atom_0 = atom_81;
    (&kernelContext_211)->work_0 = work_81;
    (&kernelContext_211)->frames_0 = frames_81;
    (&kernelContext_211)->bodies_0 = bodies_81;
    (&kernelContext_211)->chunks_0 = chunks_81;
    (&kernelContext_211)->sort_0 = sort_81;
    (&kernelContext_211)->manifolds_0 = manifolds_81;
    (&kernelContext_211)->loads_0 = loads_81;
    (&kernelContext_211)->bonds_0 = bonds_81;
    (&kernelContext_211)->args_0 = args_81;
    (&kernelContext_211)->info_0 = info_81;
    (&kernelContext_211)->scan_0 = scan_81;
    (&kernelContext_211)->parts_0 = parts_81;
    (&kernelContext_211)->colliders_0 = colliders_81;
    (&kernelContext_211)->collider_info_0 = collider_info_81;
    (&kernelContext_211)->view_chunks_0 = view_chunks_81;
    (&kernelContext_211)->view_0 = view_81;
    (&kernelContext_211)->objects_0 = objects_81;
    (&kernelContext_211)->shot_bodies_0 = shot_bodies_81;
    (&kernelContext_211)->lines_0 = lines_81;
    (&kernelContext_211)->geometry_0 = geometry_81;
    (&kernelContext_211)->shots_0 = shots_81;
    (&kernelContext_211)->lp_0 = lp_81;
    (&kernelContext_211)->ext_0 = ext_83;
    (&kernelContext_211)->engine_contacts_0 = engine_contacts_81;
    (&kernelContext_211)->crush_0 = crush_81;
    (&kernelContext_211)->crush_rw_0 = crush_rw_81;
    (&kernelContext_211)->crush_tab_0 = crush_tab_81;
    threadgroup array<uint, int(1024)> g_scan_81;
    (&kernelContext_211)->g_scan_0 = &g_scan_81;
    threadgroup array<float, int(64)> g_il_best_81;
    (&kernelContext_211)->g_il_best_0 = &g_il_best_81;
    threadgroup array<float, int(256)> g_gi_x_81;
    (&kernelContext_211)->g_gi_x_0 = &g_gi_x_81;
    threadgroup array<float, int(256)> g_gi_m_81;
    (&kernelContext_211)->g_gi_m_0 = &g_gi_m_81;
    threadgroup array<uint, int(1280)> g_gi_lo_81;
    (&kernelContext_211)->g_gi_lo_0 = &g_gi_lo_81;
    threadgroup array<uint, int(1280)> g_gi_hi_81;
    (&kernelContext_211)->g_gi_hi_0 = &g_gi_hi_81;
    uint _S1465 = a_counters_1(&kernelContext_211);
    atomic_store_explicit((&kernelContext_211)->atom_0+(_S1465 + 18U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_211)->atom_0+(_S1465 + 19U), 0U, memory_order_relaxed);
    return;
}

void gi_acc_0(uint thread* lo_7, uint thread* hi_7, float value_4)
{
    float x_8 = round(value_4 * 1.073741824e+09f);
    float x_9;
    if(!((abs(x_8)) < 3.99999993722699776e+18f))
    {
        x_9 = 0.0f;
    }
    else
    {
        x_9 = x_8;
    }
    float u_12 = abs(x_9);
    float h_7 = floor(u_12 * 2.32830643653869629e-10f);
    uint l_3 = uint(u_12 - h_7 * 4.294967296e+09f);
    uint g_0 = uint(h_7);
    uint l_4;
    uint g_1;
    if(x_9 < 0.0f)
    {
        uint l_5 = ~l_3 + 1U;
        uint _S1466 = ~g_0;
        if(l_5 == 0U)
        {
            l_4 = 1U;
        }
        else
        {
            l_4 = 0U;
        }
        uint _S1467 = _S1466 + l_4;
        l_4 = l_5;
        g_1 = _S1467;
    }
    else
    {
        l_4 = l_3;
        g_1 = g_0;
    }
    uint sum_4 = *lo_7 + l_4;
    if(sum_4 < (*lo_7))
    {
        l_4 = 1U;
    }
    else
    {
        l_4 = 0U;
    }
    *hi_7 = *hi_7 + (g_1 + l_4);
    *lo_7 = sum_4;
    return;
}

float gi_value_0(uint lo_8, uint hi_8)
{
    bool negative_0 = (hi_8 & 2147483648U) != 0U;
    uint _S1468;
    uint _S1469;
    if(negative_0)
    {
        uint _S1470 = ~lo_8 + 1U;
        uint _S1471 = ~hi_8;
        if(_S1470 == 0U)
        {
            _S1468 = 1U;
        }
        else
        {
            _S1468 = 0U;
        }
        _S1468 = _S1471 + _S1468;
        _S1469 = _S1470;
    }
    else
    {
        _S1468 = hi_8;
        _S1469 = lo_8;
    }
    float v_17 = (float(_S1468) * 4.294967296e+09f + float(_S1469)) / 1.073741824e+09f;
    float _S1472;
    if(negative_0)
    {
        _S1472 = - v_17;
    }
    else
    {
        _S1472 = v_17;
    }
    return _S1472;
}

float gi_dlf_0(float r_28)
{
    if(!(r_28 > 0.0f))
    {
        return 0.0f;
    }
    if(r_28 > 33.0f)
    {
        return r_28 / (r_28 - 1.0f);
    }
    if((abs(r_28 - 1.0f)) < 0.00009999999747379f)
    {
        return 1.57079637050628662f;
    }
    float _S1473 = r_28 * r_28;
    float _S1474 = r_28 + 1.0f;
    uint _S1475 = uint(floor(0.5f * _S1474));
    float best_2 = abs(2.0f * r_28 * cos(1.57079637050628662f * r_28) / (1.0f - _S1473));
    uint l_6 = 1U;
    for(;;)
    {
        if(l_6 <= _S1475)
        {
        }
        else
        {
            break;
        }
        float a_10 = 6.28318548202514648f * float(l_6) / _S1474;
        float _S1476 = max(best_2, abs((sin(a_10) - sin(r_28 * a_10) / r_28) / (1.0f - 1.0f / _S1473)));
        uint _S1477 = l_6 + 1U;
        best_2 = _S1476;
        l_6 = _S1477;
    }
    return best_2;
}

float gi_factor_0(uint s_25, uint t_16, uint base_3, uint count_0, KernelContext_0 thread* kernelContext_212)
{
    bool _S1478;
    float x2_0 = 0.0f;
    float mm_0 = 0.0f;
    uint j_12 = t_16;
    for(;;)
    {
        if(j_12 < count_0)
        {
        }
        else
        {
            break;
        }
        uint _S1479 = w_cs_list_0(kernelContext_212);
        uint device* _S1480 = kernelContext_212->work_0+(_S1479 + base_3 + j_12);
        float3 u_13 = (float4(*(kernelContext_212->lp_0+st_x_0(*_S1480, 14U, 0U))) ).xyz;
        float _S1481 = max(x2_0, dot(u_13, u_13));
        float _S1482 = max(mm_0, (float4(*(kernelContext_212->chunks_0+4U * *_S1480)) ).w);
        uint j_13 = j_12 + 256U;
        x2_0 = _S1481;
        mm_0 = _S1482;
        j_12 = j_13;
    }
    (*kernelContext_212->g_gi_x_0)[t_16] = x2_0;
    (*kernelContext_212->g_gi_m_0)[t_16] = mm_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint step_6 = 128U;
    for(;;)
    {
        if(step_6 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_16 < step_6)
        {
            uint _S1483 = t_16 + step_6;
            (*kernelContext_212->g_gi_x_0)[t_16] = max((*kernelContext_212->g_gi_x_0)[t_16], (*kernelContext_212->g_gi_x_0)[_S1483]);
            (*kernelContext_212->g_gi_m_0)[t_16] = max((*kernelContext_212->g_gi_m_0)[t_16], (*kernelContext_212->g_gi_m_0)[_S1483]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_6 = step_6 >> 1U;
    }
    float x2_1 = (*kernelContext_212->g_gi_x_0)[int(0)];
    float mm_1 = (*kernelContext_212->g_gi_m_0)[int(0)];
    float xn_0 = sqrt((*kernelContext_212->g_gi_x_0)[int(0)]);
    uint _S1484 = lp_body_0(s_25, kernelContext_212);
    float f0_0 = (float4(*(kernelContext_212->lp_0+(_S1484 + 2U))) ).w;
    bool any_1;
    if(x2_1 > 0.0f)
    {
        any_1 = mm_1 > 0.0f;
    }
    else
    {
        any_1 = false;
    }
    if(any_1)
    {
        any_1 = f0_0 > 0.0f;
    }
    else
    {
        any_1 = false;
    }
    thread array<uint, int(5)> lo_9;
    thread array<uint, int(5)> hi_9;
    uint i_25 = 0U;
    for(;;)
    {
        if(i_25 < 5U)
        {
        }
        else
        {
            break;
        }
        lo_9[i_25] = 0U;
        hi_9[i_25] = 0U;
        i_25 = i_25 + 1U;
    }
    uint j2_2 = t_16;
    for(;;)
    {
        if(any_1)
        {
            _S1478 = j2_2 < count_0;
        }
        else
        {
            _S1478 = false;
        }
        if(_S1478)
        {
        }
        else
        {
            break;
        }
        uint _S1485 = w_cs_list_0(kernelContext_212);
        uint device* _S1486 = kernelContext_212->work_0+(_S1485 + base_3 + j2_2);
        uint _S1487 = st_x_0(*_S1486, 14U, 0U);
        float3 u_14 = (float4(*(kernelContext_212->lp_0+_S1487)) ).xyz;
        float3 th_0 = (float4(*(kernelContext_212->lp_0+(_S1487 + 1U))) ).xyz;
        uint _S1488 = st_f_0(*_S1486, 14U);
        uint _S1489 = 4U * *_S1486;
        float4 _S1490 = float4(*(kernelContext_212->chunks_0+_S1489)) ;
        float4 _S1491 = float4(*(kernelContext_212->chunks_0+(_S1489 + 1U))) ;
        float4 _S1492 = float4(*(kernelContext_212->chunks_0+(_S1489 + 2U))) ;
        float _S1493 = th_0.x;
        float _S1494 = _S1491.w;
        float _S1495 = th_0.y;
        float _S1496 = _S1492.x;
        float _S1497 = th_0.z;
        float _S1498 = _S1492.y;
        float3 ith_0 = float3(_S1491.x * _S1493 + _S1494 * _S1495 + _S1496 * _S1497, _S1494 * _S1493 + _S1491.y * _S1495 + _S1498 * _S1497, _S1496 * _S1493 + _S1498 * _S1495 + _S1491.z * _S1497);
        gi_acc_0(&lo_9[int(0)], &hi_9[int(0)], (dot((float4(*(kernelContext_212->lp_0+_S1488)) ).xyz, u_14) + dot((float4(*(kernelContext_212->lp_0+(_S1488 + 1U))) ).xyz, th_0)) / (f0_0 * xn_0));
        float _S1499 = _S1490.w;
        gi_acc_0(&lo_9[int(1)], &hi_9[int(1)], (_S1499 * dot(u_14, u_14) + dot(th_0, ith_0)) / (mm_1 * x2_1));
        float3 p_21 = u_14 * float3((_S1499 / (mm_1 * xn_0))) ;
        gi_acc_0(&lo_9[int(2)], &hi_9[int(2)], p_21.x);
        gi_acc_0(&lo_9[int(3)], &hi_9[int(3)], p_21.y);
        gi_acc_0(&lo_9[int(4)], &hi_9[int(4)], p_21.z);
        j2_2 = j2_2 + 256U;
    }
    uint i2_2 = 0U;
    for(;;)
    {
        if(i2_2 < 5U)
        {
        }
        else
        {
            break;
        }
        uint _S1500 = 5U * t_16 + i2_2;
        (*kernelContext_212->g_gi_lo_0)[_S1500] = lo_9[i2_2];
        (*kernelContext_212->g_gi_hi_0)[_S1500] = hi_9[i2_2];
        i2_2 = i2_2 + 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint step2_0 = 128U;
    for(;;)
    {
        if(step2_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_16 < step2_0)
        {
            uint i3_1 = 0U;
            for(;;)
            {
                if(i3_1 < 5U)
                {
                }
                else
                {
                    break;
                }
                uint a_11 = 5U * t_16 + i3_1;
                uint o_9 = 5U * (t_16 + step2_0) + i3_1;
                uint sum_5 = (*kernelContext_212->g_gi_lo_0)[a_11] + (*kernelContext_212->g_gi_lo_0)[o_9];
                uint _S1501 = (*kernelContext_212->g_gi_hi_0)[o_9];
                if(sum_5 < (*kernelContext_212->g_gi_lo_0)[a_11])
                {
                    j_12 = 1U;
                }
                else
                {
                    j_12 = 0U;
                }
                (*kernelContext_212->g_gi_hi_0)[a_11] = (*kernelContext_212->g_gi_hi_0)[a_11] + (_S1501 + j_12);
                (*kernelContext_212->g_gi_lo_0)[a_11] = sum_5;
                i3_1 = i3_1 + 1U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step2_0 = step2_0 >> 1U;
    }
    float work_sum_0 = gi_value_0((*kernelContext_212->g_gi_lo_0)[int(0)], (*kernelContext_212->g_gi_hi_0)[int(0)]);
    float inertia_0 = gi_value_0((*kernelContext_212->g_gi_lo_0)[int(1)], (*kernelContext_212->g_gi_hi_0)[int(1)]);
    float3 p_sum_0 = float3(gi_value_0((*kernelContext_212->g_gi_lo_0)[int(2)], (*kernelContext_212->g_gi_hi_0)[int(2)]), gi_value_0((*kernelContext_212->g_gi_lo_0)[int(3)], (*kernelContext_212->g_gi_hi_0)[int(3)]), gi_value_0((*kernelContext_212->g_gi_lo_0)[int(4)], (*kernelContext_212->g_gi_hi_0)[int(4)]));
    float inv_mass_8 = (float4(*(kernelContext_212->bodies_0+7U * s_25)) ).w;
    uint _S1502 = a_cs_0(s_25, kernelContext_212);
    uint _S1503 = atomic_load_explicit(kernelContext_212->atom_0+(_S1502 + 4U), memory_order_relaxed);
    if(_S1503 == 0U)
    {
        _S1478 = inv_mass_8 > 0.0f;
    }
    else
    {
        _S1478 = false;
    }
    float inertia_1;
    if(_S1478)
    {
        inertia_1 = inertia_0 - mm_1 * inv_mass_8 * dot(p_sum_0, p_sum_0);
    }
    else
    {
        inertia_1 = inertia_0;
    }
    float w_9 = f0_0 * xn_0 * work_sum_0;
    float m_6 = mm_1 * x2_1 * inertia_1;
    if(any_1)
    {
        any_1 = w_9 > 0.0f;
    }
    else
    {
        any_1 = false;
    }
    if(any_1)
    {
        any_1 = m_6 > 0.0f;
    }
    else
    {
        any_1 = false;
    }
    float omega_0;
    if(any_1)
    {
        omega_0 = sqrt(w_9 / m_6);
    }
    else
    {
        omega_0 = 0.0f;
    }
    uint _S1504 = _S1484 + 6U;
    float tau_4 = (float4(*(kernelContext_212->lp_0+_S1504)) ).z;
    if(t_16 == 0U)
    {
        any_1 = count_0 > 0U;
    }
    else
    {
        any_1 = false;
    }
    if(any_1)
    {
        (*(kernelContext_212->lp_0+_S1504))[int(3)] = omega_0;
    }
    if(omega_0 > 0.0f)
    {
        any_1 = tau_4 > 0.0f;
    }
    else
    {
        any_1 = false;
    }
    if(any_1)
    {
        x2_0 = gi_dlf_0(omega_0 * tau_4 / 3.14159274101257324f);
    }
    else
    {
        x2_0 = 1.0f;
    }
    return x2_0;
}

void gi_break_0(uint i_26, uint s_26, float scale_5, float3 contact_20, float local_1, KernelContext_0 thread* kernelContext_213)
{
    uint4 _S1505 = uint4(*(kernelContext_213->bonds_0+i_26)) ;
    LpBond_0 _S1506 = lp_bond_0(i_26, kernelContext_213);
    if((length(_S1506.centroid_0 - contact_20)) <= local_1)
    {
        return;
    }
    uint _S1507 = _S1505.x;
    uint _S1508 = _S1505.y;
    thread LpBond_0 _S1509 = _S1506;
    thread float3 q_lin_7;
    thread float3 q_ang_7;
    st_force_0(&_S1509, _S1507, _S1508, 14U, 0U, &q_lin_7, &q_ang_7, kernelContext_213);
    float3 _S1510 = float3(scale_5) ;
    q_lin_7 = q_lin_7 * _S1510;
    q_ang_7 = q_ang_7 * _S1510;
    if((kernelContext_213->params_0->load_model_0) == 1U)
    {
        thread LpBond_0 _S1511 = _S1506;
        thread float3 b_lin_1;
        thread float3 b_ang_1;
        lp_force_0(&_S1511, _S1507, _S1508, 0U, &b_lin_1, &b_ang_1, kernelContext_213);
        q_lin_7 = q_lin_7 + b_lin_1;
        q_ang_7 = q_ang_7 + b_ang_1;
    }
    thread JointBond_0 _S1512 = _S1506.law_0;
    Measures_0 _S1513 = stress_measures_0(&_S1512, q_lin_7, q_ang_7);
    JointMaterial_0 _S1514 = lp_material_0(_S1506.law_0.ids_3.x, kernelContext_213);
    float _S1515 = _S1506.law_0.geom1_0.w;
    thread JointMaterial_0 _S1516 = _S1514;
    thread JointBond_0 _S1517 = _S1506.law_0;
    thread Measures_0 _S1518 = _S1513;
    float4 _S1519 = failure_indices_0(&_S1516, &_S1517, &_S1518, _S1515);
    if((max(max(_S1519.x, _S1519.y), max(_S1519.z, _S1519.w))) >= 1.0f)
    {
        uint _S1520 = w_alive_0(kernelContext_213);
        *(kernelContext_213->work_0+(_S1520 + i_26)) = 0U;
        uint _S1521 = a_dirty_0(kernelContext_213);
        uint _S1522 = atomic_fetch_or_explicit(kernelContext_213->atom_0+(_S1521 + s_26), 1U, memory_order_relaxed);
        uint _S1523 = a_counters_1(kernelContext_213);
        uint _S1524 = atomic_fetch_add_explicit(kernelContext_213->atom_0+_S1523, 1U, memory_order_relaxed);
        uint _S1525 = atomic_fetch_add_explicit(kernelContext_213->atom_0+(_S1523 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_26, kernelContext_213);
        uint _S1526 = atomic_fetch_add_explicit(kernelContext_213->atom_0+(_S1523 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_gi_solve(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_82 [[buffer(0)]], atomic<uint> device* atom_82 [[buffer(5)]], uint device* work_82 [[buffer(4)]], packed_float4 device* frames_82 [[buffer(6)]], packed_float4 device* bodies_82 [[buffer(7)]], packed_float4 device* chunks_82 [[buffer(1)]], uint device* sort_82 [[buffer(11)]], Record_natural_0 device* manifolds_82 [[buffer(10)]], packed_float4 device* loads_82 [[buffer(12)]], packed_uint4 device* bonds_82 [[buffer(2)]], uint device* args_82 [[buffer(13)]], uint device* info_82 [[buffer(15)]], ScanParams_0 constant* scan_82 [[buffer(14)]], packed_float4 device* parts_82 [[buffer(3)]], packed_float4 device* colliders_82 [[buffer(8)]], packed_uint4 device* collider_info_82 [[buffer(9)]], packed_float4 device* view_chunks_82 [[buffer(17)]], ViewParams_0 constant* view_82 [[buffer(16)]], packed_float4 device* objects_82 [[buffer(18)]], uint device* shot_bodies_82 [[buffer(20)]], float device* lines_82 [[buffer(19)]], packed_float4 device* geometry_82 [[buffer(21)]], packed_float4 device* shots_82 [[buffer(22)]], packed_float4 device* lp_82 [[buffer(24)]], packed_float4 device* ext_84 [[buffer(23)]], packed_float4 device* engine_contacts_82 [[buffer(25)]], CrushParams_0 constant* crush_82 [[buffer(28)]], packed_float4 device* crush_rw_82 [[buffer(27)]], packed_float4 device* crush_tab_82 [[buffer(26)]])
{
    bool _S1527;
    uint j3_1;
    thread KernelContext_0 kernelContext_214;
    (&kernelContext_214)->params_0 = params_82;
    (&kernelContext_214)->atom_0 = atom_82;
    (&kernelContext_214)->work_0 = work_82;
    (&kernelContext_214)->frames_0 = frames_82;
    (&kernelContext_214)->bodies_0 = bodies_82;
    (&kernelContext_214)->chunks_0 = chunks_82;
    (&kernelContext_214)->sort_0 = sort_82;
    (&kernelContext_214)->manifolds_0 = manifolds_82;
    (&kernelContext_214)->loads_0 = loads_82;
    (&kernelContext_214)->bonds_0 = bonds_82;
    (&kernelContext_214)->args_0 = args_82;
    (&kernelContext_214)->info_0 = info_82;
    (&kernelContext_214)->scan_0 = scan_82;
    (&kernelContext_214)->parts_0 = parts_82;
    (&kernelContext_214)->colliders_0 = colliders_82;
    (&kernelContext_214)->collider_info_0 = collider_info_82;
    (&kernelContext_214)->view_chunks_0 = view_chunks_82;
    (&kernelContext_214)->view_0 = view_82;
    (&kernelContext_214)->objects_0 = objects_82;
    (&kernelContext_214)->shot_bodies_0 = shot_bodies_82;
    (&kernelContext_214)->lines_0 = lines_82;
    (&kernelContext_214)->geometry_0 = geometry_82;
    (&kernelContext_214)->shots_0 = shots_82;
    (&kernelContext_214)->lp_0 = lp_82;
    (&kernelContext_214)->ext_0 = ext_84;
    (&kernelContext_214)->engine_contacts_0 = engine_contacts_82;
    (&kernelContext_214)->crush_0 = crush_82;
    (&kernelContext_214)->crush_rw_0 = crush_rw_82;
    (&kernelContext_214)->crush_tab_0 = crush_tab_82;
    threadgroup array<uint, int(1024)> g_scan_82;
    (&kernelContext_214)->g_scan_0 = &g_scan_82;
    threadgroup array<float, int(64)> g_il_best_82;
    (&kernelContext_214)->g_il_best_0 = &g_il_best_82;
    threadgroup array<float, int(256)> g_gi_x_82;
    (&kernelContext_214)->g_gi_x_0 = &g_gi_x_82;
    threadgroup array<float, int(256)> g_gi_m_82;
    (&kernelContext_214)->g_gi_m_0 = &g_gi_m_82;
    threadgroup array<uint, int(1280)> g_gi_lo_82;
    (&kernelContext_214)->g_gi_lo_0 = &g_gi_lo_82;
    threadgroup array<uint, int(1280)> g_gi_hi_82;
    (&kernelContext_214)->g_gi_hi_0 = &g_gi_hi_82;
    uint t_17 = thread_7.x;
    uint _S1528 = group_4.x;
    uint _S1529 = a_counters_1(&kernelContext_214);
    uint _S1530 = atomic_load_explicit((&kernelContext_214)->atom_0+(_S1529 + 18U), memory_order_relaxed);
    uint j_14;
    if(_S1528 < _S1530)
    {
        j_14 = 1U;
    }
    else
    {
        j_14 = 0U;
    }
    if((group_uniform_0(j_14, t_17)) == 0U)
    {
        return;
    }
    uint _S1531 = w_cs_bodies_0(&kernelContext_214);
    uint device* _S1532 = (&kernelContext_214)->work_0+(_S1531 + _S1528);
    uint s_27 = *_S1532;
    uint _S1533 = a_cs_0(*_S1532, &kernelContext_214);
    uint base_4 = atomic_load_explicit((&kernelContext_214)->atom_0+(_S1533 + 6U), memory_order_relaxed);
    uint count_1 = atomic_load_explicit((&kernelContext_214)->atom_0+(_S1533 + 5U), memory_order_relaxed);
    j_14 = t_17;
    for(;;)
    {
        if(j_14 < count_1)
        {
        }
        else
        {
            break;
        }
        uint _S1534 = w_cs_list_0(&kernelContext_214);
        cs_load_0(*((&kernelContext_214)->work_0+(_S1534 + base_4 + j_14)), s_27, &kernelContext_214);
        j_14 = j_14 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint j2_3 = t_17;
    for(;;)
    {
        if(j2_3 < count_1)
        {
        }
        else
        {
            break;
        }
        uint _S1535 = w_cs_list_0(&kernelContext_214);
        uint device* _S1536 = (&kernelContext_214)->work_0+(_S1535 + base_4 + j2_3);
        uint k_55 = *_S1536;
        uint _S1537 = st_f_0(*_S1536, 14U);
        if(((float4(*((&kernelContext_214)->lp_0+_S1537)) ).w) == 0.0f)
        {
            bool _S1538 = lp_factor_0(k_55, &kernelContext_214);
            _S1527 = !_S1538;
        }
        else
        {
            _S1527 = false;
        }
        if(_S1527)
        {
            (*((&kernelContext_214)->lp_0+_S1537))[int(3)] = 1.0f;
            uint _S1539 = lp_intact_0(k_55, &kernelContext_214);
            if(_S1539 != 0U)
            {
                uint _S1540 = atomic_fetch_or_explicit((&kernelContext_214)->atom_0+(_S1529 + 3U), 64U, memory_order_relaxed);
            }
        }
        j2_3 = j2_3 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint _S1541 = group_uniform_0(clamp(2U * uint(ceil(32.0f * pow(float(max(count_1, 1U)), 0.3333333432674408f))), 256U, (&kernelContext_214)->params_0->gi_sweeps_0), t_17);
    uint sweep_1 = 0U;
    for(;;)
    {
        if(sweep_1 < _S1541)
        {
        }
        else
        {
            break;
        }
        j3_1 = t_17;
        for(;;)
        {
            if(j3_1 < count_1)
            {
            }
            else
            {
                break;
            }
            uint _S1542 = w_cs_list_0(&kernelContext_214);
            st_sweep_0(*((&kernelContext_214)->work_0+(_S1542 + base_4 + j3_1)), 14U, sweep_1 % 2U, (&kernelContext_214)->params_0->cs_c1_0, (&kernelContext_214)->params_0->cs_c2_0, &kernelContext_214);
            j3_1 = j3_1 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        sweep_1 = sweep_1 + 1U;
    }
    float scale_6 = (&kernelContext_214)->params_0->gi_scale_0;
    float scale_7;
    if(((&kernelContext_214)->params_0->gi_mode_0) == 2U)
    {
        float _S1543 = gi_factor_0(s_27, t_17, base_4, count_1, &kernelContext_214);
        float scale_8 = scale_6 * _S1543;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        scale_7 = scale_8;
    }
    else
    {
        scale_7 = scale_6;
    }
    uint _S1544 = atomic_load_explicit((&kernelContext_214)->atom_0+(_S1533 + 11U), memory_order_relaxed);
    float _S1545;
    if(_S1544 == 0U)
    {
        uint _S1546 = lp_body_0(s_27, &kernelContext_214);
        _S1545 = (float4(*((&kernelContext_214)->lp_0+(_S1546 + 4U))) ).x;
    }
    else
    {
        _S1545 = 0.0f;
    }
    thread Quat_0 sq_0;
    thread float3 s_origin_0;
    structure_pose_0(s_27, 7U * s_27, &sq_0, &s_origin_0, &kernelContext_214);
    thread Quat_0 _S1547 = sq_0;
    Quat_0 _S1548 = conj_0(&_S1547);
    uint _S1549 = lp_body_0(s_27, &kernelContext_214);
    float3 _S1550 = (float4(*((&kernelContext_214)->lp_0+(_S1549 + 2U))) ).xyz - s_origin_0;
    thread Quat_0 _S1551 = _S1548;
    float3 _S1552 = rotate_0(&_S1551, _S1550);
    j3_1 = t_17;
    for(;;)
    {
        if(j3_1 < count_1)
        {
        }
        else
        {
            break;
        }
        uint _S1553 = w_cs_list_0(&kernelContext_214);
        uint device* _S1554 = (&kernelContext_214)->work_0+(_S1553 + base_4 + j3_1);
        uint k_56 = *_S1554;
        float4 _S1555 = float4(*((&kernelContext_214)->ext_0+((&kernelContext_214)->params_0->ext_chunks_0 + 4U * *_S1554))) ;
        uint _S1556 = (as_type<uint>((_S1555.x)));
        uint _S1557 = (as_type<uint>((_S1555.y)));
        uint e_8 = 0U;
        for(;;)
        {
            if(e_8 < _S1557)
            {
            }
            else
            {
                break;
            }
            uint _S1558 = adj_bond_0(_S1556 + e_8, &kernelContext_214);
            if(((uint4(*((&kernelContext_214)->bonds_0+_S1558)) ).x) == k_56)
            {
                uint _S1559 = w_alive_0(&kernelContext_214);
                _S1527 = (*((&kernelContext_214)->work_0+(_S1559 + _S1558))) != 0U;
            }
            else
            {
                _S1527 = false;
            }
            if(_S1527)
            {
                gi_break_0(_S1558, s_27, scale_7, _S1552, _S1545, &kernelContext_214);
            }
            e_8 = e_8 + 1U;
        }
        j3_1 = j3_1 + 256U;
    }
    return;
}

bool crushable_0(uint k_57, KernelContext_0 thread* kernelContext_215)
{
    return ((as_type<uint>(((float4(*(kernelContext_215->crush_tab_0+(2U * k_57 + 1U))) ).z))) & 1U) != 0U;
}

uint crush_flags_0(uint k_58, KernelContext_0 thread* kernelContext_216)
{
    return (as_type<uint>(((float4(*(kernelContext_216->crush_rw_0+k_58)) ).w)));
}

void crush_record_0(uint k_59, float4 step_7, uint cause_0, KernelContext_0 thread* kernelContext_217)
{
    float4 _S1560 = float4(*(kernelContext_217->crush_rw_0+k_59)) ;
    uint flags_1 = (as_type<uint>((_S1560.w)));
    uint flags_2;
    if((step_7.w) != 0.0f)
    {
        flags_2 = ((flags_1 | 1U) | 2U) | (cause_0 << 4U);
    }
    else
    {
        flags_2 = flags_1;
    }
    *(kernelContext_217->crush_rw_0+k_59) = packed_float4(float4(step_7.x, max(_S1560.y, step_7.y), max(_S1560.z, step_7.z), (as_type<float>((flags_2))))) ;
    return;
}

[[kernel]] void fr_crush_blast(uint3 id_76 [[thread_position_in_grid]], FractureParams_0 constant* params_83 [[buffer(0)]], atomic<uint> device* atom_83 [[buffer(5)]], uint device* work_83 [[buffer(4)]], packed_float4 device* frames_83 [[buffer(6)]], packed_float4 device* bodies_83 [[buffer(7)]], packed_float4 device* chunks_83 [[buffer(1)]], uint device* sort_83 [[buffer(11)]], Record_natural_0 device* manifolds_83 [[buffer(10)]], packed_float4 device* loads_83 [[buffer(12)]], packed_uint4 device* bonds_83 [[buffer(2)]], uint device* args_83 [[buffer(13)]], uint device* info_83 [[buffer(15)]], ScanParams_0 constant* scan_83 [[buffer(14)]], packed_float4 device* parts_83 [[buffer(3)]], packed_float4 device* colliders_83 [[buffer(8)]], packed_uint4 device* collider_info_83 [[buffer(9)]], packed_float4 device* view_chunks_83 [[buffer(17)]], ViewParams_0 constant* view_83 [[buffer(16)]], packed_float4 device* objects_83 [[buffer(18)]], uint device* shot_bodies_83 [[buffer(20)]], float device* lines_83 [[buffer(19)]], packed_float4 device* geometry_83 [[buffer(21)]], packed_float4 device* shots_83 [[buffer(22)]], packed_float4 device* lp_83 [[buffer(24)]], packed_float4 device* ext_85 [[buffer(23)]], packed_float4 device* engine_contacts_83 [[buffer(25)]], CrushParams_0 constant* crush_83 [[buffer(28)]], packed_float4 device* crush_rw_83 [[buffer(27)]], packed_float4 device* crush_tab_83 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_218;
    (&kernelContext_218)->params_0 = params_83;
    (&kernelContext_218)->atom_0 = atom_83;
    (&kernelContext_218)->work_0 = work_83;
    (&kernelContext_218)->frames_0 = frames_83;
    (&kernelContext_218)->bodies_0 = bodies_83;
    (&kernelContext_218)->chunks_0 = chunks_83;
    (&kernelContext_218)->sort_0 = sort_83;
    (&kernelContext_218)->manifolds_0 = manifolds_83;
    (&kernelContext_218)->loads_0 = loads_83;
    (&kernelContext_218)->bonds_0 = bonds_83;
    (&kernelContext_218)->args_0 = args_83;
    (&kernelContext_218)->info_0 = info_83;
    (&kernelContext_218)->scan_0 = scan_83;
    (&kernelContext_218)->parts_0 = parts_83;
    (&kernelContext_218)->colliders_0 = colliders_83;
    (&kernelContext_218)->collider_info_0 = collider_info_83;
    (&kernelContext_218)->view_chunks_0 = view_chunks_83;
    (&kernelContext_218)->view_0 = view_83;
    (&kernelContext_218)->objects_0 = objects_83;
    (&kernelContext_218)->shot_bodies_0 = shot_bodies_83;
    (&kernelContext_218)->lines_0 = lines_83;
    (&kernelContext_218)->geometry_0 = geometry_83;
    (&kernelContext_218)->shots_0 = shots_83;
    (&kernelContext_218)->lp_0 = lp_83;
    (&kernelContext_218)->ext_0 = ext_85;
    (&kernelContext_218)->engine_contacts_0 = engine_contacts_83;
    (&kernelContext_218)->crush_0 = crush_83;
    (&kernelContext_218)->crush_rw_0 = crush_rw_83;
    (&kernelContext_218)->crush_tab_0 = crush_tab_83;
    threadgroup array<uint, int(1024)> g_scan_83;
    (&kernelContext_218)->g_scan_0 = &g_scan_83;
    threadgroup array<float, int(64)> g_il_best_83;
    (&kernelContext_218)->g_il_best_0 = &g_il_best_83;
    threadgroup array<float, int(256)> g_gi_x_83;
    (&kernelContext_218)->g_gi_x_0 = &g_gi_x_83;
    threadgroup array<float, int(256)> g_gi_m_83;
    (&kernelContext_218)->g_gi_m_0 = &g_gi_m_83;
    threadgroup array<uint, int(1280)> g_gi_lo_83;
    (&kernelContext_218)->g_gi_lo_0 = &g_gi_lo_83;
    threadgroup array<uint, int(1280)> g_gi_hi_83;
    (&kernelContext_218)->g_gi_hi_0 = &g_gi_hi_83;
    uint k_60 = id_76.x;
    bool _S1561;
    if(k_60 >= (params_83->chunk_count_0))
    {
        _S1561 = true;
    }
    else
    {
        _S1561 = ((&kernelContext_218)->crush_0->enabled_0) == 0U;
    }
    if(_S1561)
    {
        return;
    }
    float4 _S1562 = float4(*((&kernelContext_218)->crush_rw_0+((&kernelContext_218)->crush_0->r_blast_0 + k_60))) ;
    float _S1563 = _S1562.x;
    if(!(_S1563 > 0.0f))
    {
        _S1561 = !((_S1562.y) > 0.0f);
    }
    else
    {
        _S1561 = false;
    }
    if(_S1561)
    {
        _S1561 = !((_S1562.z) > 0.0f);
    }
    else
    {
        _S1561 = false;
    }
    if(_S1561)
    {
        return;
    }
    *((&kernelContext_218)->crush_rw_0+((&kernelContext_218)->crush_0->r_blast_0 + k_60)) = packed_float4(float4(0.0f) ) ;
    bool _S1564 = crushable_0(k_60, &kernelContext_218);
    if(!_S1564)
    {
        _S1561 = true;
    }
    else
    {
        uint _S1565 = crush_flags_0(k_60, &kernelContext_218);
        _S1561 = (_S1565 & 1U) != 0U;
    }
    if(_S1561)
    {
        return;
    }
    float d_24 = (float4(*((&kernelContext_218)->crush_rw_0+k_60)) ).x + _S1563;
    float4 step_8;
    if(d_24 >= 1.0f)
    {
        step_8 = float4(1.0f, _S1562.y, _S1562.z, 1.0f);
    }
    else
    {
        step_8 = float4(d_24, _S1562.y, _S1562.z, 0.0f);
    }
    crush_record_0(k_60, step_8, 3U, &kernelContext_218);
    return;
}

struct Sym_0
{
    float3 d_25;
    float3 o_10;
};

Sym_0 sym_zero_0()
{
    thread Sym_0 s_28;
    float3 _S1566 = float3(0.0f) ;
    (&s_28)->d_25 = _S1566;
    (&s_28)->o_10 = _S1566;
    return s_28;
}

Sym_0 sym_outer_0(float3 r_29, float3 f_13)
{
    thread Sym_0 s_29;
    (&s_29)->d_25 = r_29 * f_13;
    float _S1567 = r_29.x;
    float _S1568 = f_13.y;
    float _S1569 = r_29.y;
    float _S1570 = f_13.x;
    float _S1571 = f_13.z;
    float _S1572 = r_29.z;
    (&s_29)->o_10 = float3(0.5f)  * float3(_S1567 * _S1568 + _S1569 * _S1570, _S1567 * _S1571 + _S1572 * _S1570, _S1569 * _S1571 + _S1572 * _S1568);
    return s_29;
}

Sym_0 sym_add_0(const Sym_0 thread* a_12, const Sym_0 thread* b_55)
{
    thread Sym_0 s_30;
    (&s_30)->d_25 = a_12->d_25 + b_55->d_25;
    (&s_30)->o_10 = a_12->o_10 + b_55->o_10;
    return s_30;
}

void add_load_0(Sym_0 thread* w_10, float3 thread* f_sum_0, float4 thread* spread_0, float3 r_30, float3 f_14)
{
    Sym_0 _S1573 = sym_outer_0(r_30, f_14);
    thread Sym_0 _S1574 = *w_10;
    thread Sym_0 _S1575 = _S1573;
    Sym_0 _S1576 = sym_add_0(&_S1574, &_S1575);
    *w_10 = _S1576;
    *f_sum_0 = *f_sum_0 + f_14;
    float m_7 = length(f_14);
    *spread_0 = *spread_0 + float4(float3(m_7)  * r_30, m_7);
    return;
}

void flow_virial_0(uint k_61, const Quat_0 thread* q_29, float3 origin_9, float3 centre_1, Sym_0 thread* wf_0, float thread* total_0, KernelContext_0 thread* kernelContext_219)
{
    *wf_0 = sym_zero_0();
    *total_0 = 0.0f;
    float4 _S1577 = float4(*(kernelContext_219->crush_tab_0+(kernelContext_219->crush_0->t_adj_0 + k_61))) ;
    uint first_2 = (as_type<uint>((_S1577.x)));
    uint _S1578 = (as_type<uint>((_S1577.y)));
    uint j_15 = first_2;
    for(;;)
    {
        if(j_15 < (first_2 + _S1578))
        {
        }
        else
        {
            break;
        }
        uint bi_0 = (as_type<uint>(((float4(*(kernelContext_219->crush_tab_0+(kernelContext_219->crush_0->t_list_0 + j_15 / 4U))) )[j_15 % 4U])));
        uint _S1579 = w_alive_0(kernelContext_219);
        if((*(kernelContext_219->work_0+(_S1579 + bi_0))) == 0U)
        {
            j_15 = j_15 + 1U;
            continue;
        }
        float l_7 = (*(kernelContext_219->crush_rw_0+(kernelContext_219->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U];
        if(!(((*(kernelContext_219->crush_rw_0+(kernelContext_219->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U]) > 0.0f))
        {
            j_15 = j_15 + 1U;
            continue;
        }
        float3 _S1580 = rotate_0(q_29, (float4(*(kernelContext_219->crush_tab_0+(kernelContext_219->crush_0->t_bonds_0 + bi_0))) ).xyz);
        float3 r_31 = origin_9 + _S1580 - centre_1;
        float3 _S1581 = rotate_0(q_29, (float4(*(kernelContext_219->crush_tab_0+(kernelContext_219->crush_0->t_normal_0 + bi_0))) ).xyz);
        float3 out_n_0;
        if((dot(_S1581, r_31)) >= 0.0f)
        {
            out_n_0 = _S1581;
        }
        else
        {
            out_n_0 = - _S1581;
        }
        Sym_0 _S1582 = sym_outer_0(r_31, float3(- l_7)  * out_n_0);
        thread Sym_0 _S1583 = *wf_0;
        thread Sym_0 _S1584 = _S1582;
        Sym_0 _S1585 = sym_add_0(&_S1583, &_S1584);
        *wf_0 = _S1585;
        *total_0 = *total_0 + l_7;
        j_15 = j_15 + 1U;
    }
    return;
}

float4 bond_centroid_0(uint k_62, const Quat_0 thread* q_30, float3 origin_10, float3 centre_2, KernelContext_0 thread* kernelContext_220)
{
    float4 _S1586 = float4(*(kernelContext_220->crush_tab_0+(kernelContext_220->crush_0->t_adj_0 + k_62))) ;
    uint first_3 = (as_type<uint>((_S1586.x)));
    uint _S1587 = (as_type<uint>((_S1586.y)));
    float3 _S1588 = float3(0.0f) ;
    uint j_16 = first_3;
    float3 sum_6 = _S1588;
    float area_1 = 0.0f;
    for(;;)
    {
        if(j_16 < (first_3 + _S1587))
        {
        }
        else
        {
            break;
        }
        uint bi_1 = (as_type<uint>(((float4(*(kernelContext_220->crush_tab_0+(kernelContext_220->crush_0->t_list_0 + j_16 / 4U))) )[j_16 % 4U])));
        uint _S1589 = w_alive_0(kernelContext_220);
        if((*(kernelContext_220->work_0+(_S1589 + bi_1))) == 0U)
        {
            j_16 = j_16 + 1U;
            continue;
        }
        float4 _S1590 = float4(*(kernelContext_220->crush_tab_0+(kernelContext_220->crush_0->t_bonds_0 + bi_1))) ;
        float3 _S1591 = rotate_0(q_30, _S1590.xyz);
        float _S1592 = _S1590.w;
        float area_2 = area_1 + _S1592;
        sum_6 = sum_6 + float3(_S1592)  * (origin_10 + _S1591 - centre_2);
        area_1 = area_2;
        j_16 = j_16 + 1U;
    }
    if(area_1 > 0.0f)
    {
        sum_6 = sum_6 / float3(area_1) ;
    }
    else
    {
        sum_6 = _S1588;
    }
    return float4(sum_6, area_1);
}

Sym_0 balanced_0(const Sym_0 thread* w_11, float3 f_15, float4 spread_1, float4 bonds_84)
{
    float3 at_23;
    if((bonds_84.w) > 0.0f)
    {
        at_23 = bonds_84.xyz;
    }
    else
    {
        float _S1593 = spread_1.w;
        if(_S1593 > 0.0f)
        {
            at_23 = spread_1.xyz / float3(_S1593) ;
        }
        else
        {
            at_23 = float3(0.0f) ;
        }
    }
    thread Sym_0 _S1594 = sym_outer_0(at_23, - f_15);
    Sym_0 _S1595 = sym_add_0(w_11, &_S1594);
    return _S1595;
}

float sym_norm2_0(const Sym_0 thread* s_31)
{
    return dot(s_31->d_25, s_31->d_25) + 2.0f * dot(s_31->o_10, s_31->o_10);
}

float2 crush_invariants_0(const Sym_0 thread* v_18, float volume_0)
{
    float inv_4 = 1.0f / volume_0;
    float xx_0 = v_18->d_25.x * inv_4;
    float yy_0 = v_18->d_25.y * inv_4;
    float zz_0 = v_18->d_25.z * inv_4;
    float xy_0 = v_18->o_10.x * inv_4;
    float xz_0 = v_18->o_10.y * inv_4;
    float yz_0 = v_18->o_10.z * inv_4;
    float p_22 = - (xx_0 + yy_0 + zz_0) / 3.0f;
    float dx_0 = xx_0 + p_22;
    float dy_0 = yy_0 + p_22;
    float dz_0 = zz_0 + p_22;
    return float2(p_22, sqrt(1.5f * max(dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0 + 2.0f * (xy_0 * xy_0 + xz_0 * xz_0 + yz_0 * yz_0), 0.0f)));
}

float crush_excess_0(float2 pq_0, float4 m0_0)
{
    float _S1596 = pq_0.x;
    if(!(_S1596 > 0.0f))
    {
        return 0.0f;
    }
    float _S1597 = max(pq_0.y - (m0_0.y + m0_0.z * _S1596), _S1596 - m0_0.x);
    float _S1598;
    if(_S1597 > 0.0f)
    {
        _S1598 = _S1597;
    }
    else
    {
        _S1598 = 0.0f;
    }
    return _S1598;
}

float4 crush_two_0(const Sym_0 thread* a_13, float dt_a_0, const Sym_0 thread* b_56, float dt_b_0, uint k_63, float damage_2, KernelContext_0 thread* kernelContext_221)
{
    uint _S1599 = 2U * k_63;
    float4 _S1600 = float4(*(kernelContext_221->crush_tab_0+_S1599)) ;
    float4 _S1601 = float4(*(kernelContext_221->crush_tab_0+(_S1599 + 1U))) ;
    float _S1602 = _S1601.y;
    float2 _S1603 = crush_invariants_0(a_13, _S1602);
    float2 _S1604 = crush_invariants_0(b_56, _S1602);
    float ea_0 = crush_excess_0(_S1603, _S1600);
    bool _S1605 = dt_b_0 > 0.0f;
    float eb_0;
    if(_S1605)
    {
        eb_0 = crush_excess_0(_S1604, _S1600);
    }
    else
    {
        eb_0 = 0.0f;
    }
    float d_26 = damage_2 + (ea_0 * ea_0 * dt_a_0 + eb_0 * eb_0 * dt_b_0) / (_S1601.x * _S1600.w);
    float2 peak_2;
    if(_S1605)
    {
        peak_2 = max(_S1603, _S1604);
    }
    else
    {
        peak_2 = _S1603;
    }
    if(d_26 >= 1.0f)
    {
        return float4(1.0f, peak_2, 1.0f);
    }
    return float4(d_26, peak_2, 0.0f);
}

void crush_loads_flow_0(uint k_64, KernelContext_0 thread* kernelContext_222)
{
    uint _S1606 = body_of_0(k_64, kernelContext_222);
    uint b_57 = 7U * _S1606;
    thread Quat_0 q_31;
    thread float3 origin_11;
    structure_pose_0(_S1606, b_57, &q_31, &origin_11, kernelContext_222);
    uint _S1607 = 4U * k_64;
    float4 _S1608 = float4(*(kernelContext_222->chunks_0+_S1607)) ;
    float4 _S1609 = float4(*(kernelContext_222->chunks_0+(_S1607 + 3U))) ;
    float3 _S1610 = _S1608.xyz;
    thread Quat_0 _S1611 = q_31;
    float3 _S1612 = rotate_0(&_S1611, _S1610);
    float3 centre_3 = origin_11 + _S1612;
    float inv_mass_9 = (float4(*(kernelContext_222->bodies_0+b_57)) ).w;
    float _S1613;
    if(inv_mass_9 > 0.0f)
    {
        _S1613 = 1.0f / inv_mass_9;
    }
    else
    {
        _S1613 = float_bits_0(2139095040U);
    }
    float dt_2 = kernelContext_222->params_0->dt_0;
    Sym_0 _S1614 = sym_zero_0();
    thread Sym_0 wa_0 = _S1614;
    float3 _S1615 = float3(0.0f) ;
    thread float3 fa_0 = _S1615;
    float4 _S1616 = float4(0.0f) ;
    thread float4 sa_0 = _S1616;
    thread Sym_0 wb_0 = _S1614;
    thread float3 fb_1 = _S1615;
    thread float4 sb_2 = _S1616;
    uint _S1617 = kernelContext_222->sort_0[k_64];
    uint _S1618 = kernelContext_222->sort_0[kernelContext_222->params_0->chunk_count_0 + 1U + k_64];
    float tau_5 = 0.0f;
    uint i_27 = 0U;
    for(;;)
    {
        if(i_27 < _S1618)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_32 = kernelContext_222->manifolds_0[_S1617 + i_27];
        float4 _S1619 = float4(r_32.impulse_0) ;
        float3 _S1620 = _S1619.xyz;
        float j_17 = length(_S1620);
        if(j_17 == 0.0f)
        {
            i_27 = i_27 + 1U;
            continue;
        }
        float4 _S1621 = float4(r_32.other_0) ;
        float m_other_2 = _S1621.x;
        float m_8;
        if(isinf(_S1613))
        {
            if(isinf(m_other_2))
            {
                m_8 = _S1608.w;
            }
            else
            {
                m_8 = m_other_2;
            }
        }
        else
        {
            if(isinf(m_other_2))
            {
                m_8 = _S1613;
            }
            else
            {
                m_8 = _S1613 * m_other_2 / (_S1613 + m_other_2);
            }
        }
        float _S1622 = _S1619.w;
        float _S1623 = min(j_17, 2.0f * m_8 * max(_S1622 - kernelContext_222->params_0->band_0, 0.0f));
        float3 rr_0 = (float4(r_32.point_0) ).xyz - centre_3;
        float3 dir_1 = _S1620 / float3(j_17) ;
        if(_S1623 < j_17)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, rr_0, dir_1 * float3(((j_17 - _S1623) / dt_2)) );
        }
        float tau_6;
        if(_S1623 > 0.0f)
        {
            float _S1624 = _S1609.y;
            float duration_2 = hertz_duration_0(m_8, min(_S1621.z, _S1609.z), 1.0f / ((1.0f - _S1624 * _S1624) / _S1609.x + 1.0f / _S1621.y), _S1622);
            add_load_0(&wb_0, &fb_1, &sb_2, rr_0, dir_1 * float3((3.14159274101257324f * _S1623 / (2.0f * duration_2))) );
            tau_6 = max(tau_5, min(duration_2, dt_2));
        }
        else
        {
            tau_6 = tau_5;
        }
        tau_5 = tau_6;
        i_27 = i_27 + 1U;
    }
    uint _S1625 = frame_parity_0(kernelContext_222);
    uint _S1626 = w_pulses_0(kernelContext_222);
    uint _S1627 = _S1626 + _S1607 + 2U * _S1625;
    uint off_1 = *(kernelContext_222->work_0+_S1627);
    uint _S1628 = *(kernelContext_222->work_0+(_S1627 + 1U));
    uint _S1629 = l_pool_0(_S1625, kernelContext_222);
    uint _S1630 = _S1629 + 3U * off_1;
    uint p_23 = 0U;
    for(;;)
    {
        if(p_23 < _S1628)
        {
        }
        else
        {
            break;
        }
        uint _S1631 = _S1630 + 3U * p_23;
        float4 _S1632 = float4(*(kernelContext_222->loads_0+_S1631)) ;
        float4 _S1633 = float4(*(kernelContext_222->loads_0+(_S1631 + 1U))) ;
        float4 _S1634 = float4(*(kernelContext_222->loads_0+(_S1631 + 2U))) ;
        float mean_2 = pulse_mean_0(_S1632.x, _S1632.y, _S1632.z, _S1632.w, _S1633.w, dt_2);
        if(mean_2 != 0.0f)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, _S1633.xyz - centre_3, _S1634.xyz * float3(mean_2) );
        }
        p_23 = p_23 + 1U;
    }
    thread Quat_0 _S1635 = q_31;
    thread Sym_0 wf_1;
    thread float flow_total_0;
    flow_virial_0(k_64, &_S1635, origin_11, centre_3, &wf_1, &flow_total_0, kernelContext_222);
    bool contact_21;
    if((sa_0.w) > 0.0f)
    {
        contact_21 = true;
    }
    else
    {
        contact_21 = (sb_2.w) > 0.0f;
    }
    bool _S1636;
    if(!contact_21)
    {
        _S1636 = !(flow_total_0 > 0.0f);
    }
    else
    {
        _S1636 = false;
    }
    if(_S1636)
    {
        return;
    }
    Sym_0 _S1637 = wf_1;
    Sym_0 _S1638 = wf_1;
    Sym_0 peak_contact_0;
    Sym_0 base_5;
    Sym_0 peak_3;
    if(contact_21)
    {
        thread Quat_0 _S1639 = q_31;
        float4 _S1640 = bond_centroid_0(k_64, &_S1639, origin_11, centre_3, kernelContext_222);
        thread Sym_0 _S1641 = wa_0;
        Sym_0 _S1642 = balanced_0(&_S1641, fa_0, sa_0, _S1640);
        thread Sym_0 _S1643 = _S1642;
        thread Sym_0 _S1644 = wf_1;
        Sym_0 _S1645 = sym_add_0(&_S1643, &_S1644);
        thread Sym_0 _S1646 = wa_0;
        thread Sym_0 _S1647 = wb_0;
        Sym_0 _S1648 = sym_add_0(&_S1646, &_S1647);
        float3 _S1649 = fa_0 + fb_1;
        float4 _S1650 = sa_0 + sb_2;
        thread Sym_0 _S1651 = _S1648;
        Sym_0 _S1652 = balanced_0(&_S1651, _S1649, _S1650, _S1640);
        thread Sym_0 _S1653 = _S1652;
        thread Sym_0 _S1654 = wf_1;
        Sym_0 _S1655 = sym_add_0(&_S1653, &_S1654);
        peak_contact_0 = _S1652;
        base_5 = _S1645;
        peak_3 = _S1655;
    }
    else
    {
        peak_contact_0 = _S1614;
        base_5 = _S1637;
        peak_3 = _S1638;
    }
    thread Sym_0 _S1656 = wf_1;
    float _S1657 = sym_norm2_0(&_S1656);
    thread Sym_0 _S1658 = peak_contact_0;
    float _S1659 = sym_norm2_0(&_S1658);
    uint cause_1;
    if(_S1657 >= _S1659)
    {
        cause_1 = 4U;
    }
    else
    {
        cause_1 = 1U;
    }
    float _S1660 = dt_2 - tau_5;
    float _S1661 = (float4(*(kernelContext_222->crush_rw_0+k_64)) ).x;
    thread Sym_0 _S1662 = base_5;
    thread Sym_0 _S1663 = peak_3;
    float4 _S1664 = crush_two_0(&_S1662, _S1660, &_S1663, tau_5, k_64, _S1661, kernelContext_222);
    crush_record_0(k_64, _S1664, cause_1, kernelContext_222);
    return;
}

Sym_0 sym_rotated_0(const Quat_0 thread* q_32, float3 d_27, float3 o_11)
{
    float _S1665 = o_11.x;
    float _S1666 = o_11.y;
    float _S1667 = o_11.z;
    matrix<float,int(3),int(3)>  s_32 = matrix<float,int(3),int(3)> (d_27.x, _S1665, _S1666, _S1665, d_27.y, _S1667, _S1666, _S1667, d_27.z);
    float3 _S1668 = rotate_0(q_32, float3(1.0f, 0.0f, 0.0f));
    float3 _S1669 = rotate_0(q_32, float3(0.0f, 1.0f, 0.0f));
    float3 _S1670 = rotate_0(q_32, float3(0.0f, 0.0f, 1.0f));
    matrix<float,int(3),int(3)>  r_33 = matrix<float,int(3),int(3)> (_S1668.x, _S1669.x, _S1670.x, _S1668.y, _S1669.y, _S1670.y, _S1668.z, _S1669.z, _S1670.z);
    matrix<float,int(3),int(3)>  w_12 = (((transpose(r_33)) * ((((s_32) * (r_33))))));
    thread Sym_0 out_s_0;
    (&out_s_0)->d_25 = float3(w_12[int(0)][int(0)], w_12[int(1)][int(1)], w_12[int(2)][int(2)]);
    (&out_s_0)->o_10 = float3(w_12[int(0)][int(1)], w_12[int(0)][int(2)], w_12[int(1)][int(2)]);
    return out_s_0;
}

[[kernel]] void fr_crush_loads(uint3 id_77 [[thread_position_in_grid]], FractureParams_0 constant* params_84 [[buffer(0)]], atomic<uint> device* atom_84 [[buffer(5)]], uint device* work_84 [[buffer(4)]], packed_float4 device* frames_84 [[buffer(6)]], packed_float4 device* bodies_84 [[buffer(7)]], packed_float4 device* chunks_84 [[buffer(1)]], uint device* sort_84 [[buffer(11)]], Record_natural_0 device* manifolds_84 [[buffer(10)]], packed_float4 device* loads_84 [[buffer(12)]], packed_uint4 device* bonds_85 [[buffer(2)]], uint device* args_84 [[buffer(13)]], uint device* info_84 [[buffer(15)]], ScanParams_0 constant* scan_84 [[buffer(14)]], packed_float4 device* parts_84 [[buffer(3)]], packed_float4 device* colliders_84 [[buffer(8)]], packed_uint4 device* collider_info_84 [[buffer(9)]], packed_float4 device* view_chunks_84 [[buffer(17)]], ViewParams_0 constant* view_84 [[buffer(16)]], packed_float4 device* objects_84 [[buffer(18)]], uint device* shot_bodies_84 [[buffer(20)]], float device* lines_84 [[buffer(19)]], packed_float4 device* geometry_84 [[buffer(21)]], packed_float4 device* shots_84 [[buffer(22)]], packed_float4 device* lp_84 [[buffer(24)]], packed_float4 device* ext_86 [[buffer(23)]], packed_float4 device* engine_contacts_84 [[buffer(25)]], CrushParams_0 constant* crush_84 [[buffer(28)]], packed_float4 device* crush_rw_84 [[buffer(27)]], packed_float4 device* crush_tab_84 [[buffer(26)]])
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
    threadgroup array<float, int(256)> g_gi_x_84;
    (&kernelContext_223)->g_gi_x_0 = &g_gi_x_84;
    threadgroup array<float, int(256)> g_gi_m_84;
    (&kernelContext_223)->g_gi_m_0 = &g_gi_m_84;
    threadgroup array<uint, int(1280)> g_gi_lo_84;
    (&kernelContext_223)->g_gi_lo_0 = &g_gi_lo_84;
    threadgroup array<uint, int(1280)> g_gi_hi_84;
    (&kernelContext_223)->g_gi_hi_0 = &g_gi_hi_84;
    uint k_65 = id_77.x;
    bool _S1671;
    if(k_65 >= (params_84->chunk_count_0))
    {
        _S1671 = true;
    }
    else
    {
        _S1671 = ((&kernelContext_223)->crush_0->enabled_0) == 0U;
    }
    if(_S1671)
    {
        _S1671 = true;
    }
    else
    {
        bool _S1672 = crushable_0(k_65, &kernelContext_223);
        _S1671 = !_S1672;
    }
    if(_S1671)
    {
        _S1671 = true;
    }
    else
    {
        uint _S1673 = crush_flags_0(k_65, &kernelContext_223);
        _S1671 = (_S1673 & 1U) != 0U;
    }
    if(_S1671)
    {
        return;
    }
    if(((&kernelContext_223)->crush_0->flow_0) != 0U)
    {
        crush_loads_flow_0(k_65, &kernelContext_223);
        return;
    }
    uint _S1674 = body_of_0(k_65, &kernelContext_223);
    uint b_58 = 7U * _S1674;
    thread Quat_0 q_33;
    thread float3 origin_12;
    structure_pose_0(_S1674, b_58, &q_33, &origin_12, &kernelContext_223);
    uint _S1675 = 4U * k_65;
    float4 _S1676 = float4(*((&kernelContext_223)->chunks_0+_S1675)) ;
    float4 _S1677 = float4(*((&kernelContext_223)->chunks_0+(_S1675 + 3U))) ;
    float3 _S1678 = _S1676.xyz;
    thread Quat_0 _S1679 = q_33;
    float3 _S1680 = rotate_0(&_S1679, _S1678);
    float3 centre_4 = origin_12 + _S1680;
    float inv_mass_10 = (float4(*((&kernelContext_223)->bodies_0+b_58)) ).w;
    float _S1681;
    if(inv_mass_10 > 0.0f)
    {
        _S1681 = 1.0f / inv_mass_10;
    }
    else
    {
        _S1681 = float_bits_0(2139095040U);
    }
    float dt_3 = (&kernelContext_223)->params_0->dt_0;
    Sym_0 _S1682 = sym_zero_0();
    thread Sym_0 wa_1 = _S1682;
    float3 _S1683 = float3(0.0f) ;
    thread float3 fa_1 = _S1683;
    float4 _S1684 = float4(0.0f) ;
    thread float4 sa_1 = _S1684;
    thread Sym_0 wb_1 = _S1682;
    thread float3 fb_2 = _S1683;
    thread float4 sb_3 = _S1684;
    uint _S1685 = (&kernelContext_223)->sort_0[k_65];
    uint _S1686 = (&kernelContext_223)->sort_0[params_84->chunk_count_0 + 1U + k_65];
    float tau_7 = 0.0f;
    uint i_28 = 0U;
    for(;;)
    {
        if(i_28 < _S1686)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_34 = (&kernelContext_223)->manifolds_0[_S1685 + i_28];
        float4 _S1687 = float4(r_34.impulse_0) ;
        float3 _S1688 = _S1687.xyz;
        float j_18 = length(_S1688);
        if(j_18 == 0.0f)
        {
            i_28 = i_28 + 1U;
            continue;
        }
        float4 _S1689 = float4(r_34.other_0) ;
        float m_other_3 = _S1689.x;
        float m_9;
        if(isinf(_S1681))
        {
            if(isinf(m_other_3))
            {
                m_9 = _S1676.w;
            }
            else
            {
                m_9 = m_other_3;
            }
        }
        else
        {
            if(isinf(m_other_3))
            {
                m_9 = _S1681;
            }
            else
            {
                m_9 = _S1681 * m_other_3 / (_S1681 + m_other_3);
            }
        }
        float _S1690 = _S1687.w;
        float _S1691 = min(j_18, 2.0f * m_9 * max(_S1690 - (&kernelContext_223)->params_0->band_0, 0.0f));
        float3 rr_1 = (float4(r_34.point_0) ).xyz - centre_4;
        float3 dir_2 = _S1688 / float3(j_18) ;
        if(_S1691 < j_18)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, rr_1, dir_2 * float3(((j_18 - _S1691) / dt_3)) );
        }
        float tau_8;
        if(_S1691 > 0.0f)
        {
            float _S1692 = _S1677.y;
            float duration_3 = hertz_duration_0(m_9, min(_S1689.z, _S1677.z), 1.0f / ((1.0f - _S1692 * _S1692) / _S1677.x + 1.0f / _S1689.y), _S1690);
            add_load_0(&wb_1, &fb_2, &sb_3, rr_1, dir_2 * float3((3.14159274101257324f * _S1691 / (2.0f * duration_3))) );
            tau_8 = max(tau_7, min(duration_3, dt_3));
        }
        else
        {
            tau_8 = tau_7;
        }
        tau_7 = tau_8;
        i_28 = i_28 + 1U;
    }
    uint _S1693 = frame_parity_0(&kernelContext_223);
    uint _S1694 = w_pulses_0(&kernelContext_223);
    uint _S1695 = _S1694 + _S1675 + 2U * _S1693;
    uint device* _S1696 = (&kernelContext_223)->work_0+_S1695;
    uint _S1697 = *((&kernelContext_223)->work_0+(_S1695 + 1U));
    uint _S1698 = l_pool_0(_S1693, &kernelContext_223);
    uint _S1699 = _S1698 + 3U * *_S1696;
    uint p_24 = 0U;
    for(;;)
    {
        if(p_24 < _S1697)
        {
        }
        else
        {
            break;
        }
        uint _S1700 = _S1699 + 3U * p_24;
        float4 _S1701 = float4(*((&kernelContext_223)->loads_0+_S1700)) ;
        float4 _S1702 = float4(*((&kernelContext_223)->loads_0+(_S1700 + 1U))) ;
        float4 _S1703 = float4(*((&kernelContext_223)->loads_0+(_S1700 + 2U))) ;
        float mean_3 = pulse_mean_0(_S1701.x, _S1701.y, _S1701.z, _S1701.w, _S1702.w, dt_3);
        if(mean_3 != 0.0f)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, _S1702.xyz - centre_4, _S1703.xyz * float3(mean_3) );
        }
        p_24 = p_24 + 1U;
    }
    if(((&kernelContext_223)->crush_0->bond_virial_0) != 0U)
    {
        uint _S1704 = 2U * k_65;
        float3 _S1705 = (float4(*((&kernelContext_223)->crush_rw_0+((&kernelContext_223)->crush_0->r_virial_0 + _S1704))) ).xyz;
        float3 _S1706 = (float4(*((&kernelContext_223)->crush_rw_0+((&kernelContext_223)->crush_0->r_virial_0 + _S1704 + 1U))) ).xyz;
        thread Quat_0 _S1707 = q_33;
        Sym_0 _S1708 = sym_rotated_0(&_S1707, _S1705, _S1706);
        thread Sym_0 _S1709 = wa_1;
        thread Sym_0 _S1710 = _S1708;
        Sym_0 _S1711 = sym_add_0(&_S1709, &_S1710);
        thread Sym_0 _S1712 = wa_1;
        thread Sym_0 _S1713 = wb_1;
        Sym_0 _S1714 = sym_add_0(&_S1712, &_S1713);
        thread Sym_0 _S1715 = _S1714;
        thread Sym_0 _S1716 = _S1708;
        Sym_0 _S1717 = sym_add_0(&_S1715, &_S1716);
        float _S1718 = dt_3 - tau_7;
        float _S1719 = (float4(*((&kernelContext_223)->crush_rw_0+k_65)) ).x;
        thread Sym_0 _S1720 = _S1711;
        thread Sym_0 _S1721 = _S1717;
        float4 _S1722 = crush_two_0(&_S1720, _S1718, &_S1721, tau_7, k_65, _S1719, &kernelContext_223);
        crush_record_0(k_65, _S1722, 1U, &kernelContext_223);
        return;
    }
    if(!((sa_1.w) > 0.0f))
    {
        _S1671 = !((sb_3.w) > 0.0f);
    }
    else
    {
        _S1671 = false;
    }
    if(_S1671)
    {
        return;
    }
    thread Quat_0 _S1723 = q_33;
    float4 _S1724 = bond_centroid_0(k_65, &_S1723, origin_12, centre_4, &kernelContext_223);
    thread Sym_0 _S1725 = wa_1;
    Sym_0 _S1726 = balanced_0(&_S1725, fa_1, sa_1, _S1724);
    thread Sym_0 _S1727 = wa_1;
    thread Sym_0 _S1728 = wb_1;
    Sym_0 _S1729 = sym_add_0(&_S1727, &_S1728);
    float3 _S1730 = fa_1 + fb_2;
    float4 _S1731 = sa_1 + sb_3;
    thread Sym_0 _S1732 = _S1729;
    Sym_0 _S1733 = balanced_0(&_S1732, _S1730, _S1731, _S1724);
    float _S1734 = dt_3 - tau_7;
    float _S1735 = (float4(*((&kernelContext_223)->crush_rw_0+k_65)) ).x;
    thread Sym_0 _S1736 = _S1726;
    thread Sym_0 _S1737 = _S1733;
    float4 _S1738 = crush_two_0(&_S1736, _S1734, &_S1737, tau_7, k_65, _S1735, &kernelContext_223);
    crush_record_0(k_65, _S1738, 1U, &kernelContext_223);
    return;
}

uint a_cover_0(KernelContext_0 thread* kernelContext_224)
{
    uint _S1739 = a_moved_0(kernelContext_224);
    return _S1739 + 2U * kernelContext_224->params_0->ev_moved_cap_0 + kernelContext_224->params_0->chunk_count_0;
}

[[kernel]] void fr_strike_patch(uint3 id_78 [[thread_position_in_grid]], FractureParams_0 constant* params_85 [[buffer(0)]], atomic<uint> device* atom_85 [[buffer(5)]], uint device* work_85 [[buffer(4)]], packed_float4 device* frames_85 [[buffer(6)]], packed_float4 device* bodies_85 [[buffer(7)]], packed_float4 device* chunks_85 [[buffer(1)]], uint device* sort_85 [[buffer(11)]], Record_natural_0 device* manifolds_85 [[buffer(10)]], packed_float4 device* loads_85 [[buffer(12)]], packed_uint4 device* bonds_86 [[buffer(2)]], uint device* args_85 [[buffer(13)]], uint device* info_85 [[buffer(15)]], ScanParams_0 constant* scan_85 [[buffer(14)]], packed_float4 device* parts_85 [[buffer(3)]], packed_float4 device* colliders_85 [[buffer(8)]], packed_uint4 device* collider_info_85 [[buffer(9)]], packed_float4 device* view_chunks_85 [[buffer(17)]], ViewParams_0 constant* view_85 [[buffer(16)]], packed_float4 device* objects_85 [[buffer(18)]], uint device* shot_bodies_85 [[buffer(20)]], float device* lines_85 [[buffer(19)]], packed_float4 device* geometry_85 [[buffer(21)]], packed_float4 device* shots_85 [[buffer(22)]], packed_float4 device* lp_85 [[buffer(24)]], packed_float4 device* ext_87 [[buffer(23)]], packed_float4 device* engine_contacts_85 [[buffer(25)]], CrushParams_0 constant* crush_85 [[buffer(28)]], packed_float4 device* crush_rw_85 [[buffer(27)]], packed_float4 device* crush_tab_85 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_225;
    (&kernelContext_225)->params_0 = params_85;
    (&kernelContext_225)->atom_0 = atom_85;
    (&kernelContext_225)->work_0 = work_85;
    (&kernelContext_225)->frames_0 = frames_85;
    (&kernelContext_225)->bodies_0 = bodies_85;
    (&kernelContext_225)->chunks_0 = chunks_85;
    (&kernelContext_225)->sort_0 = sort_85;
    (&kernelContext_225)->manifolds_0 = manifolds_85;
    (&kernelContext_225)->loads_0 = loads_85;
    (&kernelContext_225)->bonds_0 = bonds_86;
    (&kernelContext_225)->args_0 = args_85;
    (&kernelContext_225)->info_0 = info_85;
    (&kernelContext_225)->scan_0 = scan_85;
    (&kernelContext_225)->parts_0 = parts_85;
    (&kernelContext_225)->colliders_0 = colliders_85;
    (&kernelContext_225)->collider_info_0 = collider_info_85;
    (&kernelContext_225)->view_chunks_0 = view_chunks_85;
    (&kernelContext_225)->view_0 = view_85;
    (&kernelContext_225)->objects_0 = objects_85;
    (&kernelContext_225)->shot_bodies_0 = shot_bodies_85;
    (&kernelContext_225)->lines_0 = lines_85;
    (&kernelContext_225)->geometry_0 = geometry_85;
    (&kernelContext_225)->shots_0 = shots_85;
    (&kernelContext_225)->lp_0 = lp_85;
    (&kernelContext_225)->ext_0 = ext_87;
    (&kernelContext_225)->engine_contacts_0 = engine_contacts_85;
    (&kernelContext_225)->crush_0 = crush_85;
    (&kernelContext_225)->crush_rw_0 = crush_rw_85;
    (&kernelContext_225)->crush_tab_0 = crush_tab_85;
    threadgroup array<uint, int(1024)> g_scan_85;
    (&kernelContext_225)->g_scan_0 = &g_scan_85;
    threadgroup array<float, int(64)> g_il_best_85;
    (&kernelContext_225)->g_il_best_0 = &g_il_best_85;
    threadgroup array<float, int(256)> g_gi_x_85;
    (&kernelContext_225)->g_gi_x_0 = &g_gi_x_85;
    threadgroup array<float, int(256)> g_gi_m_85;
    (&kernelContext_225)->g_gi_m_0 = &g_gi_m_85;
    threadgroup array<uint, int(1280)> g_gi_lo_85;
    (&kernelContext_225)->g_gi_lo_0 = &g_gi_lo_85;
    threadgroup array<uint, int(1280)> g_gi_hi_85;
    (&kernelContext_225)->g_gi_hi_0 = &g_gi_hi_85;
    uint b_59 = id_78.x;
    bool _S1740;
    if(b_59 >= (params_85->body_count_0))
    {
        _S1740 = true;
    }
    else
    {
        _S1740 = ((&kernelContext_225)->params_0->shot_count_0) == 0U;
    }
    if(_S1740)
    {
        _S1740 = true;
    }
    else
    {
        _S1740 = ((&kernelContext_225)->crush_0->enabled_0) == 0U;
    }
    if(_S1740)
    {
        return;
    }
    uint bill_0 = (&kernelContext_225)->crush_0->r_bill_0 + 4U * b_59;
    packed_float4 _S1741 = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_225)->crush_rw_0+bill_0) = _S1741;
    uint _S1742 = bill_0 + 3U;
    *((&kernelContext_225)->crush_rw_0+_S1742) = _S1741;
    thread uint shot_11;
    bool _S1743 = struck_0(b_59, &shot_11, &kernelContext_225);
    if(!_S1743)
    {
        return;
    }
    uint _S1744 = a_strike_0(&kernelContext_225);
    uint _S1745 = _S1744 + 9U * b_59;
    uint k_66 = atomic_load_explicit((&kernelContext_225)->atom_0+(_S1745 + 8U), memory_order_relaxed);
    uint _S1746 = atomic_load_explicit((&kernelContext_225)->atom_0+(_S1745 + 6U), memory_order_relaxed);
    float t_in_10 = (as_type<float>((_S1746)));
    if(!(t_in_10 <= ((&kernelContext_225)->params_0->dt_0)))
    {
        _S1740 = true;
    }
    else
    {
        _S1740 = k_66 == 4294967295U;
    }
    if(_S1740)
    {
        return;
    }
    uint _S1747 = strike_at_0(b_59, &kernelContext_225);
    float4 _S1748 = float4(*((&kernelContext_225)->frames_0+_S1747)) ;
    float3 _S1749 = (float4(*((&kernelContext_225)->frames_0+(_S1747 + 1U))) ).xyz;
    float speed_15 = length(_S1749);
    float3 d_28 = _S1749 / float3(max(speed_15, 9.999999960041972e-13f)) ;
    float4 _S1750 = float4(*((&kernelContext_225)->shots_0+2U * shot_11)) ;
    float radius_1 = _S1750.y;
    float _S1751 = max(_S1750.z, 0.00100000004749745f);
    float m_imp_1 = _S1750.w;
    float inv_mass_11 = (float4(*((&kernelContext_225)->bodies_0+7U * b_59)) ).w;
    bool _S1752 = inv_mass_11 == 0.0f;
    float m_body_2;
    if(_S1752)
    {
        m_body_2 = 0.0f;
    }
    else
    {
        m_body_2 = 1.0f / inv_mass_11;
    }
    float m_eff_9;
    if(_S1752)
    {
        m_eff_9 = m_imp_1;
    }
    else
    {
        m_eff_9 = m_imp_1 * m_body_2 / (m_imp_1 + m_body_2);
    }
    float modulus_2 = 1.0f / (1.0f / (float4(*((&kernelContext_225)->shots_0+(2U * shot_11 + 1U))) ).x + 1.0f / (float4(*((&kernelContext_225)->geometry_0+(3U * k_66 + 1U))) ).w);
    float _S1753 = max(speed_15, 0.00100000004749745f);
    float tau_9 = 2.86999988555908203f * pow(m_eff_9 * m_eff_9 / (_S1751 * modulus_2 * modulus_2 * _S1753), 0.20000000298023224f);
    float force_3 = m_eff_9 * speed_15 / tau_9;
    float3 contact_22 = _S1748.xyz + _S1749 * float3(t_in_10)  + d_28 * float3(radius_1) ;
    float delta_3 = pow(15.0f * m_eff_9 * _S1753 * _S1753 / (16.0f * modulus_2 * sqrt(_S1751)), 0.40000000596046448f);
    float a_h_0 = sqrt(radius_1 * delta_3);
    bool _S1754 = crushable_0(k_66, &kernelContext_225);
    float p_cap_0;
    if(_S1754)
    {
        p_cap_0 = (float4(*((&kernelContext_225)->crush_tab_0+2U * k_66)) ).x;
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
    float _S1755 = min(radius_1, max(a_h_0, a_p_0));
    float _S1756 = max(delta_3, radius_1 - sqrt(max(radius_1 * radius_1 - _S1755 * _S1755, 0.0f)));
    *((&kernelContext_225)->crush_rw_0+(bill_0 + 1U)) = packed_float4(float4(contact_22, 0.0f)) ;
    *((&kernelContext_225)->crush_rw_0+(bill_0 + 2U)) = packed_float4(float4(d_28, _S1755)) ;
    *((&kernelContext_225)->crush_rw_0+_S1742) = packed_float4(float4(force_3, tau_9, _S1756, float(k_66 + 1U))) ;
    uint _S1757 = a_cover_0(&kernelContext_225);
    uint _S1758 = _S1757 + 128U * b_59;
    uint s_33 = 0U;
    for(;;)
    {
        if(s_33 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S1759 = _S1758 + 2U * s_33;
        atomic_store_explicit((&kernelContext_225)->atom_0+_S1759, 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_225)->atom_0+(_S1759 + 1U), 4294967295U, memory_order_relaxed);
        s_33 = s_33 + 1U;
    }
    return;
}

bool cover_patch_0(uint k_67, uint thread* b_60, float4 thread* p1_0, float4 thread* p2_2, KernelContext_0 thread* kernelContext_226)
{
    uint _S1760 = body_of_0(k_67, kernelContext_226);
    *b_60 = _S1760;
    float4 _S1761 = float4(0.0f) ;
    *p1_0 = _S1761;
    *p2_2 = _S1761;
    thread uint shot_12;
    bool _S1762 = struck_0(*b_60, &shot_12, kernelContext_226);
    bool _S1763;
    if(!_S1762)
    {
        _S1763 = true;
    }
    else
    {
        uint _S1764 = crush_flags_0(k_67, kernelContext_226);
        _S1763 = (_S1764 & 1U) != 0U;
    }
    if(_S1763)
    {
        return false;
    }
    uint bill_1 = kernelContext_226->crush_0->r_bill_0 + 4U * *b_60;
    if(((float4(*(kernelContext_226->crush_rw_0+(bill_1 + 3U))) ).w) == 0.0f)
    {
        return false;
    }
    *p1_0 = float4((float4(*(kernelContext_226->crush_rw_0+(bill_1 + 1U))) ).xyz, (float4(*(kernelContext_226->shots_0+2U * shot_12)) ).y);
    *p2_2 = float4(*(kernelContext_226->crush_rw_0+(bill_1 + 2U))) ;
    return true;
}

float3 cover_offset_0(uint s_34, float3 d_29, float a_14, float thread* rho_0)
{
    uint ring_2 = s_34 / 8U;
    uint sector_0 = s_34 % 8U;
    *rho_0 = a_14 * sqrt((float(ring_2) + 0.5f) / 8.0f);
    float theta_1 = 6.28318548202514648f * (float(sector_0) + 0.5f * float(ring_2 & 1U)) / 8.0f;
    float3 e_9;
    if((abs(d_29.x)) < 0.57735002040863037f)
    {
        e_9 = float3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        if((abs(d_29.y)) < 0.57735002040863037f)
        {
            e_9 = float3(0.0f, 1.0f, 0.0f);
        }
        else
        {
            e_9 = float3(0.0f, 0.0f, 1.0f);
        }
    }
    float3 u_15 = normalize(cross(d_29, e_9));
    return float3(*rho_0)  * (float3(cos(theta_1))  * u_15 + float3(sin(theta_1))  * cross(d_29, u_15));
}

float cover_depth_0(uint k_68, uint s_35, float3 contact_23, float3 d_30, float a_15, float radius_2, KernelContext_0 thread* kernelContext_227)
{
    thread float rho_1;
    float _S1765 = 2.0f * radius_2;
    float3 _S1766 = contact_23 + cover_offset_0(s_35, d_30, a_15, &rho_1) - d_30 * float3(_S1765) ;
    thread Quat_0 q_34;
    float3 _S1767 = chunk_box_world_0(k_68, &q_34, kernelContext_227);
    uint _S1768 = 3U * k_68;
    Quat_0 _S1769 = quat_of_0(float4(*(kernelContext_227->geometry_0+(_S1768 + 2U))) );
    thread Quat_0 _S1770 = q_34;
    thread Quat_0 _S1771 = _S1769;
    Quat_0 _S1772 = quat_mul_0(&_S1770, &_S1771);
    thread Quat_0 _S1773 = _S1772;
    Quat_0 _S1774 = normalized_0(&_S1773);
    float3 _S1775 = (float4(*(kernelContext_227->geometry_0+(_S1768 + 1U))) ).xyz;
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
        thread float3 e_10 = float3(0.0f) ;
        e_10[ax_4] = 1.0f;
        thread Quat_0 _S1776 = _S1774;
        float3 _S1777 = rotate_0(&_S1776, e_10);
        float o_12 = dot(_S1766 - _S1767, _S1777);
        float dv_2 = dot(d_30, _S1777);
        if((abs(dv_2)) < 9.99999971718068537e-10f)
        {
            if((abs(o_12)) > (_S1775[ax_4]))
            {
                return 1.00000001504746622e+30f;
            }
            ax_4 = ax_4 + 1U;
            continue;
        }
        float ta_3 = (- _S1775[ax_4] - o_12) / dv_2;
        float tb_3 = (_S1775[ax_4] - o_12) / dv_2;
        float _S1778 = min(t1_4, max(ta_3, tb_3));
        t0_3 = max(t0_3, min(ta_3, tb_3));
        t1_4 = _S1778;
        ax_4 = ax_4 + 1U;
    }
    bool _S1779;
    if(!(t0_3 <= t1_4))
    {
        _S1779 = true;
    }
    else
    {
        _S1779 = t1_4 < 0.0f;
    }
    if(_S1779)
    {
        return 1.00000001504746622e+30f;
    }
    return t0_3 - _S1765;
}

[[kernel]] void fr_strike_cover(uint3 id_79 [[thread_position_in_grid]], FractureParams_0 constant* params_86 [[buffer(0)]], atomic<uint> device* atom_86 [[buffer(5)]], uint device* work_86 [[buffer(4)]], packed_float4 device* frames_86 [[buffer(6)]], packed_float4 device* bodies_86 [[buffer(7)]], packed_float4 device* chunks_86 [[buffer(1)]], uint device* sort_86 [[buffer(11)]], Record_natural_0 device* manifolds_86 [[buffer(10)]], packed_float4 device* loads_86 [[buffer(12)]], packed_uint4 device* bonds_87 [[buffer(2)]], uint device* args_86 [[buffer(13)]], uint device* info_86 [[buffer(15)]], ScanParams_0 constant* scan_86 [[buffer(14)]], packed_float4 device* parts_86 [[buffer(3)]], packed_float4 device* colliders_86 [[buffer(8)]], packed_uint4 device* collider_info_86 [[buffer(9)]], packed_float4 device* view_chunks_86 [[buffer(17)]], ViewParams_0 constant* view_86 [[buffer(16)]], packed_float4 device* objects_86 [[buffer(18)]], uint device* shot_bodies_86 [[buffer(20)]], float device* lines_86 [[buffer(19)]], packed_float4 device* geometry_86 [[buffer(21)]], packed_float4 device* shots_86 [[buffer(22)]], packed_float4 device* lp_86 [[buffer(24)]], packed_float4 device* ext_88 [[buffer(23)]], packed_float4 device* engine_contacts_86 [[buffer(25)]], CrushParams_0 constant* crush_86 [[buffer(28)]], packed_float4 device* crush_rw_86 [[buffer(27)]], packed_float4 device* crush_tab_86 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_228;
    (&kernelContext_228)->params_0 = params_86;
    (&kernelContext_228)->atom_0 = atom_86;
    (&kernelContext_228)->work_0 = work_86;
    (&kernelContext_228)->frames_0 = frames_86;
    (&kernelContext_228)->bodies_0 = bodies_86;
    (&kernelContext_228)->chunks_0 = chunks_86;
    (&kernelContext_228)->sort_0 = sort_86;
    (&kernelContext_228)->manifolds_0 = manifolds_86;
    (&kernelContext_228)->loads_0 = loads_86;
    (&kernelContext_228)->bonds_0 = bonds_87;
    (&kernelContext_228)->args_0 = args_86;
    (&kernelContext_228)->info_0 = info_86;
    (&kernelContext_228)->scan_0 = scan_86;
    (&kernelContext_228)->parts_0 = parts_86;
    (&kernelContext_228)->colliders_0 = colliders_86;
    (&kernelContext_228)->collider_info_0 = collider_info_86;
    (&kernelContext_228)->view_chunks_0 = view_chunks_86;
    (&kernelContext_228)->view_0 = view_86;
    (&kernelContext_228)->objects_0 = objects_86;
    (&kernelContext_228)->shot_bodies_0 = shot_bodies_86;
    (&kernelContext_228)->lines_0 = lines_86;
    (&kernelContext_228)->geometry_0 = geometry_86;
    (&kernelContext_228)->shots_0 = shots_86;
    (&kernelContext_228)->lp_0 = lp_86;
    (&kernelContext_228)->ext_0 = ext_88;
    (&kernelContext_228)->engine_contacts_0 = engine_contacts_86;
    (&kernelContext_228)->crush_0 = crush_86;
    (&kernelContext_228)->crush_rw_0 = crush_rw_86;
    (&kernelContext_228)->crush_tab_0 = crush_tab_86;
    threadgroup array<uint, int(1024)> g_scan_86;
    (&kernelContext_228)->g_scan_0 = &g_scan_86;
    threadgroup array<float, int(64)> g_il_best_86;
    (&kernelContext_228)->g_il_best_0 = &g_il_best_86;
    threadgroup array<float, int(256)> g_gi_x_86;
    (&kernelContext_228)->g_gi_x_0 = &g_gi_x_86;
    threadgroup array<float, int(256)> g_gi_m_86;
    (&kernelContext_228)->g_gi_m_0 = &g_gi_m_86;
    threadgroup array<uint, int(1280)> g_gi_lo_86;
    (&kernelContext_228)->g_gi_lo_0 = &g_gi_lo_86;
    threadgroup array<uint, int(1280)> g_gi_hi_86;
    (&kernelContext_228)->g_gi_hi_0 = &g_gi_hi_86;
    uint k_69 = id_79.x;
    bool _S1780;
    if(k_69 >= (params_86->chunk_count_0))
    {
        _S1780 = true;
    }
    else
    {
        _S1780 = ((&kernelContext_228)->params_0->shot_count_0) == 0U;
    }
    if(_S1780)
    {
        _S1780 = true;
    }
    else
    {
        _S1780 = ((&kernelContext_228)->crush_0->enabled_0) == 0U;
    }
    if(_S1780)
    {
        return;
    }
    thread uint b_61;
    thread float4 p1_1;
    thread float4 p2_3;
    bool _S1781 = cover_patch_0(k_69, &b_61, &p1_1, &p2_3, &kernelContext_228);
    if(!_S1781)
    {
        return;
    }
    uint _S1782 = a_cover_0(&kernelContext_228);
    uint _S1783 = _S1782 + 128U * b_61;
    uint s_36 = 0U;
    for(;;)
    {
        if(s_36 < 64U)
        {
        }
        else
        {
            break;
        }
        float _S1784 = cover_depth_0(k_69, s_36, p1_1.xyz, p2_3.xyz, p2_3.w, p1_1.w, &kernelContext_228);
        if(_S1784 < 5.0000000752373311e+29f)
        {
            uint _S1785 = atomic_fetch_min_explicit((&kernelContext_228)->atom_0+(_S1783 + 2U * s_36), ordered_0(_S1784), memory_order_relaxed);
        }
        s_36 = s_36 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_claim(uint3 id_80 [[thread_position_in_grid]], FractureParams_0 constant* params_87 [[buffer(0)]], atomic<uint> device* atom_87 [[buffer(5)]], uint device* work_87 [[buffer(4)]], packed_float4 device* frames_87 [[buffer(6)]], packed_float4 device* bodies_87 [[buffer(7)]], packed_float4 device* chunks_87 [[buffer(1)]], uint device* sort_87 [[buffer(11)]], Record_natural_0 device* manifolds_87 [[buffer(10)]], packed_float4 device* loads_87 [[buffer(12)]], packed_uint4 device* bonds_88 [[buffer(2)]], uint device* args_87 [[buffer(13)]], uint device* info_87 [[buffer(15)]], ScanParams_0 constant* scan_87 [[buffer(14)]], packed_float4 device* parts_87 [[buffer(3)]], packed_float4 device* colliders_87 [[buffer(8)]], packed_uint4 device* collider_info_87 [[buffer(9)]], packed_float4 device* view_chunks_87 [[buffer(17)]], ViewParams_0 constant* view_87 [[buffer(16)]], packed_float4 device* objects_87 [[buffer(18)]], uint device* shot_bodies_87 [[buffer(20)]], float device* lines_87 [[buffer(19)]], packed_float4 device* geometry_87 [[buffer(21)]], packed_float4 device* shots_87 [[buffer(22)]], packed_float4 device* lp_87 [[buffer(24)]], packed_float4 device* ext_89 [[buffer(23)]], packed_float4 device* engine_contacts_87 [[buffer(25)]], CrushParams_0 constant* crush_87 [[buffer(28)]], packed_float4 device* crush_rw_87 [[buffer(27)]], packed_float4 device* crush_tab_87 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_229;
    (&kernelContext_229)->params_0 = params_87;
    (&kernelContext_229)->atom_0 = atom_87;
    (&kernelContext_229)->work_0 = work_87;
    (&kernelContext_229)->frames_0 = frames_87;
    (&kernelContext_229)->bodies_0 = bodies_87;
    (&kernelContext_229)->chunks_0 = chunks_87;
    (&kernelContext_229)->sort_0 = sort_87;
    (&kernelContext_229)->manifolds_0 = manifolds_87;
    (&kernelContext_229)->loads_0 = loads_87;
    (&kernelContext_229)->bonds_0 = bonds_88;
    (&kernelContext_229)->args_0 = args_87;
    (&kernelContext_229)->info_0 = info_87;
    (&kernelContext_229)->scan_0 = scan_87;
    (&kernelContext_229)->parts_0 = parts_87;
    (&kernelContext_229)->colliders_0 = colliders_87;
    (&kernelContext_229)->collider_info_0 = collider_info_87;
    (&kernelContext_229)->view_chunks_0 = view_chunks_87;
    (&kernelContext_229)->view_0 = view_87;
    (&kernelContext_229)->objects_0 = objects_87;
    (&kernelContext_229)->shot_bodies_0 = shot_bodies_87;
    (&kernelContext_229)->lines_0 = lines_87;
    (&kernelContext_229)->geometry_0 = geometry_87;
    (&kernelContext_229)->shots_0 = shots_87;
    (&kernelContext_229)->lp_0 = lp_87;
    (&kernelContext_229)->ext_0 = ext_89;
    (&kernelContext_229)->engine_contacts_0 = engine_contacts_87;
    (&kernelContext_229)->crush_0 = crush_87;
    (&kernelContext_229)->crush_rw_0 = crush_rw_87;
    (&kernelContext_229)->crush_tab_0 = crush_tab_87;
    threadgroup array<uint, int(1024)> g_scan_87;
    (&kernelContext_229)->g_scan_0 = &g_scan_87;
    threadgroup array<float, int(64)> g_il_best_87;
    (&kernelContext_229)->g_il_best_0 = &g_il_best_87;
    threadgroup array<float, int(256)> g_gi_x_87;
    (&kernelContext_229)->g_gi_x_0 = &g_gi_x_87;
    threadgroup array<float, int(256)> g_gi_m_87;
    (&kernelContext_229)->g_gi_m_0 = &g_gi_m_87;
    threadgroup array<uint, int(1280)> g_gi_lo_87;
    (&kernelContext_229)->g_gi_lo_0 = &g_gi_lo_87;
    threadgroup array<uint, int(1280)> g_gi_hi_87;
    (&kernelContext_229)->g_gi_hi_0 = &g_gi_hi_87;
    uint k_70 = id_80.x;
    bool _S1786;
    if(k_70 >= (params_87->chunk_count_0))
    {
        _S1786 = true;
    }
    else
    {
        _S1786 = ((&kernelContext_229)->params_0->shot_count_0) == 0U;
    }
    if(_S1786)
    {
        _S1786 = true;
    }
    else
    {
        _S1786 = ((&kernelContext_229)->crush_0->enabled_0) == 0U;
    }
    if(_S1786)
    {
        return;
    }
    thread uint b_62;
    thread float4 p1_2;
    thread float4 p2_4;
    bool _S1787 = cover_patch_0(k_70, &b_62, &p1_2, &p2_4, &kernelContext_229);
    if(!_S1787)
    {
        return;
    }
    uint _S1788 = a_cover_0(&kernelContext_229);
    uint _S1789 = _S1788 + 128U * b_62;
    uint s_37 = 0U;
    for(;;)
    {
        if(s_37 < 64U)
        {
        }
        else
        {
            break;
        }
        uint _S1790 = _S1789 + 2U * s_37;
        uint first_4 = atomic_load_explicit((&kernelContext_229)->atom_0+_S1790, memory_order_relaxed);
        if(first_4 == 4294967295U)
        {
            s_37 = s_37 + 1U;
            continue;
        }
        float _S1791 = cover_depth_0(k_70, s_37, p1_2.xyz, p2_4.xyz, p2_4.w, p1_2.w, &kernelContext_229);
        if(_S1791 < 5.0000000752373311e+29f)
        {
            _S1786 = _S1791 <= (unordered_0(first_4) + 0.00009999999747379f);
        }
        else
        {
            _S1786 = false;
        }
        if(_S1786)
        {
            uint _S1792 = atomic_fetch_min_explicit((&kernelContext_229)->atom_0+(_S1790 + 1U), k_70, memory_order_relaxed);
        }
        s_37 = s_37 + 1U;
    }
    return;
}

float strike_load_0(uint k_71, uint slot_6, float3 f_16, float3 at_24, float tau_10, float budget_0, KernelContext_0 thread* kernelContext_230)
{
    bool _S1793 = crushable_0(k_71, kernelContext_230);
    bool _S1794;
    if(!_S1793)
    {
        _S1794 = true;
    }
    else
    {
        uint _S1795 = crush_flags_0(k_71, kernelContext_230);
        _S1794 = (_S1795 & 1U) != 0U;
    }
    if(_S1794)
    {
        _S1794 = true;
    }
    else
    {
        _S1794 = !(budget_0 > 0.0f);
    }
    if(_S1794)
    {
        return 0.0f;
    }
    thread Quat_0 q_35;
    thread float3 origin_13;
    structure_pose_0(slot_6, 7U * slot_6, &q_35, &origin_13, kernelContext_230);
    float3 _S1796 = (float4(*(kernelContext_230->chunks_0+4U * k_71)) ).xyz;
    thread Quat_0 _S1797 = q_35;
    float3 _S1798 = rotate_0(&_S1797, _S1796);
    float3 centre_5 = origin_13 + _S1798;
    thread Quat_0 _S1799 = q_35;
    float4 _S1800 = bond_centroid_0(k_71, &_S1799, origin_13, centre_5, kernelContext_230);
    if(!((_S1800.w) > 0.0f))
    {
        return 0.0f;
    }
    Sym_0 _S1801 = sym_outer_0(_S1800.xyz, - f_16);
    thread Sym_0 _S1802 = sym_outer_0(at_24 - centre_5, f_16);
    thread Sym_0 _S1803 = _S1801;
    Sym_0 _S1804 = sym_add_0(&_S1802, &_S1803);
    float before_0 = (float4(*(kernelContext_230->crush_rw_0+k_71)) ).x;
    float _S1805 = min(tau_10, kernelContext_230->params_0->dt_0);
    thread Sym_0 _S1806 = _S1804;
    thread Sym_0 _S1807 = _S1804;
    float4 _S1808 = crush_two_0(&_S1806, _S1805, &_S1807, 0.0f, k_71, before_0, kernelContext_230);
    uint _S1809 = 2U * k_71;
    float per_0 = (float4(*(kernelContext_230->crush_tab_0+_S1809)) ).w * (float4(*(kernelContext_230->crush_tab_0+(_S1809 + 1U))) ).y;
    float energy_1 = (_S1808.x - before_0) * per_0;
    float energy_2;
    float4 step_9;
    if(energy_1 > budget_0)
    {
        float d_31 = before_0 + budget_0 / per_0;
        float _S1810 = min(d_31, 1.0f);
        float _S1811 = _S1808.y;
        float _S1812 = _S1808.z;
        if(d_31 >= 1.0f)
        {
            energy_2 = 1.0f;
        }
        else
        {
            energy_2 = 0.0f;
        }
        step_9 = float4(_S1810, _S1811, _S1812, energy_2);
        energy_2 = budget_0;
    }
    else
    {
        step_9 = _S1808;
        energy_2 = energy_1;
    }
    crush_record_0(k_71, step_9, 2U, kernelContext_230);
    return energy_2;
}

bool cell_in_0(uint2 mask_0, uint c_25)
{
    return (((mask_0[c_25 / 32U]) >> (c_25 % 32U)) & 1U) != 0U;
}

float strike_crush_body_0(uint b_63, uint s_38, float budget_1, KernelContext_0 thread* kernelContext_231)
{
    uint total_1;
    bool _S1813;
    uint bill_2 = kernelContext_231->crush_0->r_bill_0 + 4U * b_63;
    uint _S1814 = bill_2 + 1U;
    float4 _S1815 = float4(*(kernelContext_231->crush_rw_0+(bill_2 + 2U))) ;
    float4 _S1816 = float4(*(kernelContext_231->crush_rw_0+(bill_2 + 3U))) ;
    float3 contact_24 = (float4(*(kernelContext_231->crush_rw_0+_S1814)) ).xyz;
    float3 d_32 = _S1815.xyz;
    float _S1817 = _S1815.w;
    float _S1818 = (float4(*(kernelContext_231->shots_0+2U * s_38)) ).y;
    uint _S1819 = a_cover_0(kernelContext_231);
    uint _S1820 = _S1819 + 128U * b_63;
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
        uint _S1821 = _S1820 + 2U * c_26;
        uint _S1822 = atomic_load_explicit(kernelContext_231->atom_0+(_S1821 + 1U), memory_order_relaxed);
        if(_S1822 == 4294967295U)
        {
            c_26 = c_26 + 1U;
            continue;
        }
        thread float rho_2;
        float3 _S1823 = cover_offset_0(c_26, d_32, _S1817, &rho_2);
        uint _S1824 = atomic_load_explicit(kernelContext_231->atom_0+_S1821, memory_order_relaxed);
        first_5 = min(first_5, unordered_0(_S1824) + _S1818 - sqrt(max(_S1818 * _S1818 - rho_2 * rho_2, 0.0f)));
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
        uint _S1825 = _S1820 + 2U * c_26;
        uint _S1826 = atomic_load_explicit(kernelContext_231->atom_0+(_S1825 + 1U), memory_order_relaxed);
        if(_S1826 == 4294967295U)
        {
            c_26 = c_26 + 1U;
            continue;
        }
        thread float rho_3;
        float3 _S1827 = cover_offset_0(c_26, d_32, _S1817, &rho_3);
        uint _S1828 = atomic_load_explicit(kernelContext_231->atom_0+_S1825, memory_order_relaxed);
        if((unordered_0(_S1828) + _S1818 - sqrt(max(_S1818 * _S1818 - rho_3 * rho_3, 0.0f)) - first_5) <= (_S1816.z))
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
    float3 _S1829 = float3(0.0f) ;
    float energy_3;
    float3 point_1;
    if(total_2 == 0U)
    {
        float _S1830 = strike_load_0(uint(_S1816.w) - 1U, b_63, d_32 * float3(_S1816.x) , contact_24, _S1816.y, budget_1, kernelContext_231);
        float3 _S1831 = float3(_S1830)  * contact_24;
        energy_3 = _S1830;
        point_1 = _S1831;
    }
    else
    {
        energy_3 = 0.0f;
        point_1 = _S1829;
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
        uint k_72 = atomic_load_explicit(kernelContext_231->atom_0+(_S1820 + 2U * c_26 + 1U), memory_order_relaxed);
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
                    uint _S1832 = atomic_load_explicit(kernelContext_231->atom_0+(_S1820 + 2U * total_1 + 1U), memory_order_relaxed);
                    _S1813 = _S1832 == k_72;
                }
                else
                {
                    _S1813 = false;
                }
                seen_0 = _S1813;
            }
            total_1 = total_1 + 1U;
        }
        if(seen_0)
        {
            c_26 = c_26 + 1U;
            continue;
        }
        uint e_11 = c_26;
        float3 at_25 = _S1829;
        uint n_6 = 0U;
        for(;;)
        {
            if(e_11 < 64U)
            {
            }
            else
            {
                break;
            }
            if(!cell_in_0(counted_0, e_11))
            {
                _S1813 = true;
            }
            else
            {
                uint _S1833 = atomic_load_explicit(kernelContext_231->atom_0+(_S1820 + 2U * e_11 + 1U), memory_order_relaxed);
                _S1813 = _S1833 != k_72;
            }
            if(_S1813)
            {
                e_11 = e_11 + 1U;
                continue;
            }
            thread float rho_4;
            float3 _S1834 = contact_24 + cover_offset_0(e_11, d_32, _S1817, &rho_4);
            uint _S1835 = atomic_load_explicit(kernelContext_231->atom_0+(_S1820 + 2U * e_11), memory_order_relaxed);
            uint _S1836 = n_6 + 1U;
            at_25 = at_25 + (_S1834 + d_32 * float3(unordered_0(_S1835)) );
            n_6 = _S1836;
            e_11 = e_11 + 1U;
        }
        float _S1837 = float(n_6);
        float3 at_26 = at_25 / float3(_S1837) ;
        float _S1838 = strike_load_0(k_72, b_63, d_32 * float3((_S1816.x * _S1837 / float(total_2))) , at_26, _S1816.y, budget_1 - energy_3, kernelContext_231);
        float3 point_2 = point_1 + float3(_S1838)  * at_26;
        energy_3 = energy_3 + _S1838;
        point_1 = point_2;
        c_26 = c_26 + 1U;
    }
    *(kernelContext_231->crush_rw_0+bill_2) = packed_float4(float4(energy_3, float(s_38 + 1U), 0.0f, 0.0f)) ;
    if(energy_3 > 0.0f)
    {
        *(kernelContext_231->crush_rw_0+_S1814) = packed_float4(float4(point_1 / float3(energy_3) , 0.0f)) ;
    }
    return energy_3;
}

[[kernel]] void fr_strike_crush(uint3 id_81 [[thread_position_in_grid]], FractureParams_0 constant* params_88 [[buffer(0)]], atomic<uint> device* atom_88 [[buffer(5)]], uint device* work_88 [[buffer(4)]], packed_float4 device* frames_88 [[buffer(6)]], packed_float4 device* bodies_88 [[buffer(7)]], packed_float4 device* chunks_88 [[buffer(1)]], uint device* sort_88 [[buffer(11)]], Record_natural_0 device* manifolds_88 [[buffer(10)]], packed_float4 device* loads_88 [[buffer(12)]], packed_uint4 device* bonds_89 [[buffer(2)]], uint device* args_88 [[buffer(13)]], uint device* info_88 [[buffer(15)]], ScanParams_0 constant* scan_88 [[buffer(14)]], packed_float4 device* parts_88 [[buffer(3)]], packed_float4 device* colliders_88 [[buffer(8)]], packed_uint4 device* collider_info_88 [[buffer(9)]], packed_float4 device* view_chunks_88 [[buffer(17)]], ViewParams_0 constant* view_88 [[buffer(16)]], packed_float4 device* objects_88 [[buffer(18)]], uint device* shot_bodies_88 [[buffer(20)]], float device* lines_88 [[buffer(19)]], packed_float4 device* geometry_88 [[buffer(21)]], packed_float4 device* shots_88 [[buffer(22)]], packed_float4 device* lp_88 [[buffer(24)]], packed_float4 device* ext_90 [[buffer(23)]], packed_float4 device* engine_contacts_88 [[buffer(25)]], CrushParams_0 constant* crush_88 [[buffer(28)]], packed_float4 device* crush_rw_88 [[buffer(27)]], packed_float4 device* crush_tab_88 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_232;
    (&kernelContext_232)->params_0 = params_88;
    (&kernelContext_232)->atom_0 = atom_88;
    (&kernelContext_232)->work_0 = work_88;
    (&kernelContext_232)->frames_0 = frames_88;
    (&kernelContext_232)->bodies_0 = bodies_88;
    (&kernelContext_232)->chunks_0 = chunks_88;
    (&kernelContext_232)->sort_0 = sort_88;
    (&kernelContext_232)->manifolds_0 = manifolds_88;
    (&kernelContext_232)->loads_0 = loads_88;
    (&kernelContext_232)->bonds_0 = bonds_89;
    (&kernelContext_232)->args_0 = args_88;
    (&kernelContext_232)->info_0 = info_88;
    (&kernelContext_232)->scan_0 = scan_88;
    (&kernelContext_232)->parts_0 = parts_88;
    (&kernelContext_232)->colliders_0 = colliders_88;
    (&kernelContext_232)->collider_info_0 = collider_info_88;
    (&kernelContext_232)->view_chunks_0 = view_chunks_88;
    (&kernelContext_232)->view_0 = view_88;
    (&kernelContext_232)->objects_0 = objects_88;
    (&kernelContext_232)->shot_bodies_0 = shot_bodies_88;
    (&kernelContext_232)->lines_0 = lines_88;
    (&kernelContext_232)->geometry_0 = geometry_88;
    (&kernelContext_232)->shots_0 = shots_88;
    (&kernelContext_232)->lp_0 = lp_88;
    (&kernelContext_232)->ext_0 = ext_90;
    (&kernelContext_232)->engine_contacts_0 = engine_contacts_88;
    (&kernelContext_232)->crush_0 = crush_88;
    (&kernelContext_232)->crush_rw_0 = crush_rw_88;
    (&kernelContext_232)->crush_tab_0 = crush_tab_88;
    threadgroup array<uint, int(1024)> g_scan_88;
    (&kernelContext_232)->g_scan_0 = &g_scan_88;
    threadgroup array<float, int(64)> g_il_best_88;
    (&kernelContext_232)->g_il_best_0 = &g_il_best_88;
    threadgroup array<float, int(256)> g_gi_x_88;
    (&kernelContext_232)->g_gi_x_0 = &g_gi_x_88;
    threadgroup array<float, int(256)> g_gi_m_88;
    (&kernelContext_232)->g_gi_m_0 = &g_gi_m_88;
    threadgroup array<uint, int(1280)> g_gi_lo_88;
    (&kernelContext_232)->g_gi_lo_0 = &g_gi_lo_88;
    threadgroup array<uint, int(1280)> g_gi_hi_88;
    (&kernelContext_232)->g_gi_hi_0 = &g_gi_hi_88;
    uint s_39 = id_81.x;
    bool _S1839;
    if(s_39 >= (params_88->shot_count_0))
    {
        _S1839 = true;
    }
    else
    {
        _S1839 = ((&kernelContext_232)->crush_0->enabled_0) == 0U;
    }
    if(_S1839)
    {
        return;
    }
    uint sb_4 = 7U * (as_type<uint>(((float4(*((&kernelContext_232)->shots_0+2U * s_39)) ).x)));
    float inv_mass_12 = (float4(*((&kernelContext_232)->bodies_0+sb_4)) ).w;
    float3 v_19 = (float4(*((&kernelContext_232)->bodies_0+(sb_4 + 3U))) ).xyz;
    float budget_2;
    if(inv_mass_12 > 0.0f)
    {
        budget_2 = 0.5f * dot(v_19, v_19) / inv_mass_12;
    }
    else
    {
        budget_2 = 1.00000001504746622e+30f;
    }
    uint b_64 = 0U;
    for(;;)
    {
        if(b_64 < ((&kernelContext_232)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        thread uint shot_13;
        bool _S1840 = struck_0(b_64, &shot_13, &kernelContext_232);
        if(!_S1840)
        {
            _S1839 = true;
        }
        else
        {
            _S1839 = shot_13 != s_39;
        }
        bool _S1841;
        if(_S1839)
        {
            _S1841 = true;
        }
        else
        {
            _S1841 = ((float4(*((&kernelContext_232)->crush_rw_0+((&kernelContext_232)->crush_0->r_bill_0 + 4U * b_64 + 3U))) ).w) == 0.0f;
        }
        if(_S1841)
        {
            b_64 = b_64 + 1U;
            continue;
        }
        float _S1842 = strike_crush_body_0(b_64, s_39, budget_2, &kernelContext_232);
        budget_2 = budget_2 - _S1842;
        b_64 = b_64 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bill(uint3 id_82 [[thread_position_in_grid]], FractureParams_0 constant* params_89 [[buffer(0)]], atomic<uint> device* atom_89 [[buffer(5)]], uint device* work_89 [[buffer(4)]], packed_float4 device* frames_89 [[buffer(6)]], packed_float4 device* bodies_89 [[buffer(7)]], packed_float4 device* chunks_89 [[buffer(1)]], uint device* sort_89 [[buffer(11)]], Record_natural_0 device* manifolds_89 [[buffer(10)]], packed_float4 device* loads_89 [[buffer(12)]], packed_uint4 device* bonds_90 [[buffer(2)]], uint device* args_89 [[buffer(13)]], uint device* info_89 [[buffer(15)]], ScanParams_0 constant* scan_89 [[buffer(14)]], packed_float4 device* parts_89 [[buffer(3)]], packed_float4 device* colliders_89 [[buffer(8)]], packed_uint4 device* collider_info_89 [[buffer(9)]], packed_float4 device* view_chunks_89 [[buffer(17)]], ViewParams_0 constant* view_89 [[buffer(16)]], packed_float4 device* objects_89 [[buffer(18)]], uint device* shot_bodies_89 [[buffer(20)]], float device* lines_89 [[buffer(19)]], packed_float4 device* geometry_89 [[buffer(21)]], packed_float4 device* shots_89 [[buffer(22)]], packed_float4 device* lp_89 [[buffer(24)]], packed_float4 device* ext_91 [[buffer(23)]], packed_float4 device* engine_contacts_89 [[buffer(25)]], CrushParams_0 constant* crush_89 [[buffer(28)]], packed_float4 device* crush_rw_89 [[buffer(27)]], packed_float4 device* crush_tab_89 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_233;
    (&kernelContext_233)->params_0 = params_89;
    (&kernelContext_233)->atom_0 = atom_89;
    (&kernelContext_233)->work_0 = work_89;
    (&kernelContext_233)->frames_0 = frames_89;
    (&kernelContext_233)->bodies_0 = bodies_89;
    (&kernelContext_233)->chunks_0 = chunks_89;
    (&kernelContext_233)->sort_0 = sort_89;
    (&kernelContext_233)->manifolds_0 = manifolds_89;
    (&kernelContext_233)->loads_0 = loads_89;
    (&kernelContext_233)->bonds_0 = bonds_90;
    (&kernelContext_233)->args_0 = args_89;
    (&kernelContext_233)->info_0 = info_89;
    (&kernelContext_233)->scan_0 = scan_89;
    (&kernelContext_233)->parts_0 = parts_89;
    (&kernelContext_233)->colliders_0 = colliders_89;
    (&kernelContext_233)->collider_info_0 = collider_info_89;
    (&kernelContext_233)->view_chunks_0 = view_chunks_89;
    (&kernelContext_233)->view_0 = view_89;
    (&kernelContext_233)->objects_0 = objects_89;
    (&kernelContext_233)->shot_bodies_0 = shot_bodies_89;
    (&kernelContext_233)->lines_0 = lines_89;
    (&kernelContext_233)->geometry_0 = geometry_89;
    (&kernelContext_233)->shots_0 = shots_89;
    (&kernelContext_233)->lp_0 = lp_89;
    (&kernelContext_233)->ext_0 = ext_91;
    (&kernelContext_233)->engine_contacts_0 = engine_contacts_89;
    (&kernelContext_233)->crush_0 = crush_89;
    (&kernelContext_233)->crush_rw_0 = crush_rw_89;
    (&kernelContext_233)->crush_tab_0 = crush_tab_89;
    threadgroup array<uint, int(1024)> g_scan_89;
    (&kernelContext_233)->g_scan_0 = &g_scan_89;
    threadgroup array<float, int(64)> g_il_best_89;
    (&kernelContext_233)->g_il_best_0 = &g_il_best_89;
    threadgroup array<float, int(256)> g_gi_x_89;
    (&kernelContext_233)->g_gi_x_0 = &g_gi_x_89;
    threadgroup array<float, int(256)> g_gi_m_89;
    (&kernelContext_233)->g_gi_m_0 = &g_gi_m_89;
    threadgroup array<uint, int(1280)> g_gi_lo_89;
    (&kernelContext_233)->g_gi_lo_0 = &g_gi_lo_89;
    threadgroup array<uint, int(1280)> g_gi_hi_89;
    (&kernelContext_233)->g_gi_hi_0 = &g_gi_hi_89;
    uint s_40 = id_82.x;
    bool _S1843;
    if(s_40 >= (params_89->shot_count_0))
    {
        _S1843 = true;
    }
    else
    {
        _S1843 = ((&kernelContext_233)->crush_0->enabled_0) == 0U;
    }
    if(_S1843)
    {
        return;
    }
    uint b_65 = 0U;
    float energy_4 = 0.0f;
    for(;;)
    {
        if(b_65 < ((&kernelContext_233)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S1844 = float4(*((&kernelContext_233)->crush_rw_0+((&kernelContext_233)->crush_0->r_bill_0 + 4U * b_65))) ;
        if((_S1844.y) == float(s_40 + 1U))
        {
            energy_4 = energy_4 + _S1844.x;
        }
        b_65 = b_65 + 1U;
    }
    *((&kernelContext_233)->crush_rw_0+((&kernelContext_233)->crush_0->r_shot_0 + s_40)) = packed_float4(float4(0.0f) ) ;
    if(!(energy_4 > 0.0f))
    {
        return;
    }
    uint sb_5 = 7U * (as_type<uint>(((float4(*((&kernelContext_233)->shots_0+2U * s_40)) ).x)));
    float inv_mass_13 = (float4(*((&kernelContext_233)->bodies_0+sb_5)) ).w;
    if(!(inv_mass_13 > 0.0f))
    {
        return;
    }
    float m_10 = 1.0f / inv_mass_13;
    float3 v_20 = (float4(*((&kernelContext_233)->bodies_0+(sb_5 + 3U))) ).xyz;
    float speed_16 = length(v_20);
    float j_19 = m_10 * (speed_16 - sqrt(max(speed_16 * speed_16 - 2.0f * energy_4 / m_10, 0.0f)));
    bool _S1845 = speed_16 > 0.0f;
    float3 _S1846;
    if(_S1845)
    {
        _S1846 = - v_20 * float3((j_19 / (m_10 * speed_16))) ;
    }
    else
    {
        _S1846 = float3(0.0f) ;
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
        uint w_13 = sb_5 + region_6 * 7U * (&kernelContext_233)->params_0->max_bodies_0 + 3U;
        *((&kernelContext_233)->bodies_0+w_13) = packed_float4(float4((float4(*((&kernelContext_233)->bodies_0+w_13)) ).xyz + _S1846, (float4(*((&kernelContext_233)->bodies_0+w_13)) ).w)) ;
        region_6 = region_6 + 1U;
    }
    packed_float4 device* _S1847 = (&kernelContext_233)->crush_rw_0+((&kernelContext_233)->crush_0->r_shot_0 + s_40);
    if(_S1845)
    {
        _S1846 = v_20 * float3((j_19 / speed_16)) ;
    }
    else
    {
        _S1846 = float3(0.0f) ;
    }
    *_S1847 = packed_float4(float4(_S1846, energy_4)) ;
    return;
}

float3 inverse_rotate_0(const Quat_0 thread* q_36, float3 v_21)
{
    thread Quat_0 c_27;
    (&c_27)->w_0 = q_36->w_0;
    (&c_27)->x_0 = - q_36->x_0;
    (&c_27)->y_0 = - q_36->y_0;
    (&c_27)->z_0 = - q_36->z_0;
    thread Quat_0 _S1848 = c_27;
    float3 _S1849 = rotate_0(&_S1848, v_21);
    return _S1849;
}

[[kernel]] void fr_crush_push(uint3 id_83 [[thread_position_in_grid]], FractureParams_0 constant* params_90 [[buffer(0)]], atomic<uint> device* atom_90 [[buffer(5)]], uint device* work_90 [[buffer(4)]], packed_float4 device* frames_90 [[buffer(6)]], packed_float4 device* bodies_90 [[buffer(7)]], packed_float4 device* chunks_90 [[buffer(1)]], uint device* sort_90 [[buffer(11)]], Record_natural_0 device* manifolds_90 [[buffer(10)]], packed_float4 device* loads_90 [[buffer(12)]], packed_uint4 device* bonds_91 [[buffer(2)]], uint device* args_90 [[buffer(13)]], uint device* info_90 [[buffer(15)]], ScanParams_0 constant* scan_90 [[buffer(14)]], packed_float4 device* parts_90 [[buffer(3)]], packed_float4 device* colliders_90 [[buffer(8)]], packed_uint4 device* collider_info_90 [[buffer(9)]], packed_float4 device* view_chunks_90 [[buffer(17)]], ViewParams_0 constant* view_90 [[buffer(16)]], packed_float4 device* objects_90 [[buffer(18)]], uint device* shot_bodies_90 [[buffer(20)]], float device* lines_90 [[buffer(19)]], packed_float4 device* geometry_90 [[buffer(21)]], packed_float4 device* shots_90 [[buffer(22)]], packed_float4 device* lp_90 [[buffer(24)]], packed_float4 device* ext_92 [[buffer(23)]], packed_float4 device* engine_contacts_90 [[buffer(25)]], CrushParams_0 constant* crush_90 [[buffer(28)]], packed_float4 device* crush_rw_90 [[buffer(27)]], packed_float4 device* crush_tab_90 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_234;
    (&kernelContext_234)->params_0 = params_90;
    (&kernelContext_234)->atom_0 = atom_90;
    (&kernelContext_234)->work_0 = work_90;
    (&kernelContext_234)->frames_0 = frames_90;
    (&kernelContext_234)->bodies_0 = bodies_90;
    (&kernelContext_234)->chunks_0 = chunks_90;
    (&kernelContext_234)->sort_0 = sort_90;
    (&kernelContext_234)->manifolds_0 = manifolds_90;
    (&kernelContext_234)->loads_0 = loads_90;
    (&kernelContext_234)->bonds_0 = bonds_91;
    (&kernelContext_234)->args_0 = args_90;
    (&kernelContext_234)->info_0 = info_90;
    (&kernelContext_234)->scan_0 = scan_90;
    (&kernelContext_234)->parts_0 = parts_90;
    (&kernelContext_234)->colliders_0 = colliders_90;
    (&kernelContext_234)->collider_info_0 = collider_info_90;
    (&kernelContext_234)->view_chunks_0 = view_chunks_90;
    (&kernelContext_234)->view_0 = view_90;
    (&kernelContext_234)->objects_0 = objects_90;
    (&kernelContext_234)->shot_bodies_0 = shot_bodies_90;
    (&kernelContext_234)->lines_0 = lines_90;
    (&kernelContext_234)->geometry_0 = geometry_90;
    (&kernelContext_234)->shots_0 = shots_90;
    (&kernelContext_234)->lp_0 = lp_90;
    (&kernelContext_234)->ext_0 = ext_92;
    (&kernelContext_234)->engine_contacts_0 = engine_contacts_90;
    (&kernelContext_234)->crush_0 = crush_90;
    (&kernelContext_234)->crush_rw_0 = crush_rw_90;
    (&kernelContext_234)->crush_tab_0 = crush_tab_90;
    threadgroup array<uint, int(1024)> g_scan_90;
    (&kernelContext_234)->g_scan_0 = &g_scan_90;
    threadgroup array<float, int(64)> g_il_best_90;
    (&kernelContext_234)->g_il_best_0 = &g_il_best_90;
    threadgroup array<float, int(256)> g_gi_x_90;
    (&kernelContext_234)->g_gi_x_0 = &g_gi_x_90;
    threadgroup array<float, int(256)> g_gi_m_90;
    (&kernelContext_234)->g_gi_m_0 = &g_gi_m_90;
    threadgroup array<uint, int(1280)> g_gi_lo_90;
    (&kernelContext_234)->g_gi_lo_0 = &g_gi_lo_90;
    threadgroup array<uint, int(1280)> g_gi_hi_90;
    (&kernelContext_234)->g_gi_hi_0 = &g_gi_hi_90;
    uint b_66 = id_83.x;
    bool _S1850;
    if(b_66 >= (params_90->body_count_0))
    {
        _S1850 = true;
    }
    else
    {
        _S1850 = ((&kernelContext_234)->params_0->shot_count_0) == 0U;
    }
    if(_S1850)
    {
        _S1850 = true;
    }
    else
    {
        _S1850 = ((&kernelContext_234)->crush_0->enabled_0) == 0U;
    }
    if(_S1850)
    {
        return;
    }
    uint _S1851 = 4U * b_66;
    float4 _S1852 = float4(*((&kernelContext_234)->crush_rw_0+((&kernelContext_234)->crush_0->r_bill_0 + _S1851))) ;
    float _S1853 = _S1852.x;
    if(!(_S1853 > 0.0f))
    {
        return;
    }
    float4 _S1854 = float4(*((&kernelContext_234)->crush_rw_0+((&kernelContext_234)->crush_0->r_shot_0 + (uint(_S1852.y) - 1U)))) ;
    uint bw_0 = 7U * b_66;
    float inv_mass_14 = (float4(*((&kernelContext_234)->bodies_0+bw_0)) ).w;
    float _S1855 = _S1854.w;
    if(!(_S1855 > 0.0f))
    {
        _S1850 = true;
    }
    else
    {
        _S1850 = inv_mass_14 == 0.0f;
    }
    if(_S1850)
    {
        return;
    }
    float3 j_20 = _S1854.xyz * float3((_S1853 / _S1855)) ;
    Quat_0 o_13 = quat_of_0(float4(*((&kernelContext_234)->bodies_0+(bw_0 + 1U))) );
    float3 inv_i_0 = (float4(*((&kernelContext_234)->bodies_0+(bw_0 + 2U))) ).xyz;
    float3 _S1856 = cross(to_rubble_0((float4(*((&kernelContext_234)->crush_rw_0+((&kernelContext_234)->crush_0->r_bill_0 + _S1851 + 1U))) ).xyz) - (float4(*((&kernelContext_234)->bodies_0+bw_0)) ).xyz, j_20);
    thread Quat_0 _S1857 = o_13;
    float3 _S1858 = inverse_rotate_0(&_S1857, _S1856);
    float3 _S1859 = inv_i_0 * _S1858;
    thread Quat_0 _S1860 = o_13;
    float3 _S1861 = rotate_0(&_S1860, _S1859);
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
        uint w_14 = bw_0 + region_7 * 7U * (&kernelContext_234)->params_0->max_bodies_0;
        uint _S1862 = w_14 + 3U;
        *((&kernelContext_234)->bodies_0+_S1862) = packed_float4(float4((float4(*((&kernelContext_234)->bodies_0+_S1862)) ).xyz + j_20 * float3(inv_mass_14) , (float4(*((&kernelContext_234)->bodies_0+_S1862)) ).w)) ;
        uint _S1863 = w_14 + 4U;
        *((&kernelContext_234)->bodies_0+_S1863) = packed_float4(float4((float4(*((&kernelContext_234)->bodies_0+_S1863)) ).xyz + _S1861, (float4(*((&kernelContext_234)->bodies_0+_S1863)) ).w)) ;
        region_7 = region_7 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bonds(uint3 id_84 [[thread_position_in_grid]], FractureParams_0 constant* params_91 [[buffer(0)]], atomic<uint> device* atom_91 [[buffer(5)]], uint device* work_91 [[buffer(4)]], packed_float4 device* frames_91 [[buffer(6)]], packed_float4 device* bodies_91 [[buffer(7)]], packed_float4 device* chunks_91 [[buffer(1)]], uint device* sort_91 [[buffer(11)]], Record_natural_0 device* manifolds_91 [[buffer(10)]], packed_float4 device* loads_91 [[buffer(12)]], packed_uint4 device* bonds_92 [[buffer(2)]], uint device* args_91 [[buffer(13)]], uint device* info_91 [[buffer(15)]], ScanParams_0 constant* scan_91 [[buffer(14)]], packed_float4 device* parts_91 [[buffer(3)]], packed_float4 device* colliders_91 [[buffer(8)]], packed_uint4 device* collider_info_91 [[buffer(9)]], packed_float4 device* view_chunks_91 [[buffer(17)]], ViewParams_0 constant* view_91 [[buffer(16)]], packed_float4 device* objects_91 [[buffer(18)]], uint device* shot_bodies_91 [[buffer(20)]], float device* lines_91 [[buffer(19)]], packed_float4 device* geometry_91 [[buffer(21)]], packed_float4 device* shots_91 [[buffer(22)]], packed_float4 device* lp_91 [[buffer(24)]], packed_float4 device* ext_93 [[buffer(23)]], packed_float4 device* engine_contacts_91 [[buffer(25)]], CrushParams_0 constant* crush_91 [[buffer(28)]], packed_float4 device* crush_rw_91 [[buffer(27)]], packed_float4 device* crush_tab_91 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_235;
    (&kernelContext_235)->params_0 = params_91;
    (&kernelContext_235)->atom_0 = atom_91;
    (&kernelContext_235)->work_0 = work_91;
    (&kernelContext_235)->frames_0 = frames_91;
    (&kernelContext_235)->bodies_0 = bodies_91;
    (&kernelContext_235)->chunks_0 = chunks_91;
    (&kernelContext_235)->sort_0 = sort_91;
    (&kernelContext_235)->manifolds_0 = manifolds_91;
    (&kernelContext_235)->loads_0 = loads_91;
    (&kernelContext_235)->bonds_0 = bonds_92;
    (&kernelContext_235)->args_0 = args_91;
    (&kernelContext_235)->info_0 = info_91;
    (&kernelContext_235)->scan_0 = scan_91;
    (&kernelContext_235)->parts_0 = parts_91;
    (&kernelContext_235)->colliders_0 = colliders_91;
    (&kernelContext_235)->collider_info_0 = collider_info_91;
    (&kernelContext_235)->view_chunks_0 = view_chunks_91;
    (&kernelContext_235)->view_0 = view_91;
    (&kernelContext_235)->objects_0 = objects_91;
    (&kernelContext_235)->shot_bodies_0 = shot_bodies_91;
    (&kernelContext_235)->lines_0 = lines_91;
    (&kernelContext_235)->geometry_0 = geometry_91;
    (&kernelContext_235)->shots_0 = shots_91;
    (&kernelContext_235)->lp_0 = lp_91;
    (&kernelContext_235)->ext_0 = ext_93;
    (&kernelContext_235)->engine_contacts_0 = engine_contacts_91;
    (&kernelContext_235)->crush_0 = crush_91;
    (&kernelContext_235)->crush_rw_0 = crush_rw_91;
    (&kernelContext_235)->crush_tab_0 = crush_tab_91;
    threadgroup array<uint, int(1024)> g_scan_91;
    (&kernelContext_235)->g_scan_0 = &g_scan_91;
    threadgroup array<float, int(64)> g_il_best_91;
    (&kernelContext_235)->g_il_best_0 = &g_il_best_91;
    threadgroup array<float, int(256)> g_gi_x_91;
    (&kernelContext_235)->g_gi_x_0 = &g_gi_x_91;
    threadgroup array<float, int(256)> g_gi_m_91;
    (&kernelContext_235)->g_gi_m_0 = &g_gi_m_91;
    threadgroup array<uint, int(1280)> g_gi_lo_91;
    (&kernelContext_235)->g_gi_lo_0 = &g_gi_lo_91;
    threadgroup array<uint, int(1280)> g_gi_hi_91;
    (&kernelContext_235)->g_gi_hi_0 = &g_gi_hi_91;
    uint i_29 = id_84.x;
    bool _S1864;
    if(i_29 >= (params_91->bond_count_0))
    {
        _S1864 = true;
    }
    else
    {
        _S1864 = ((&kernelContext_235)->crush_0->enabled_0) == 0U;
    }
    if(_S1864)
    {
        _S1864 = true;
    }
    else
    {
        uint _S1865 = w_alive_0(&kernelContext_235);
        _S1864 = (*((&kernelContext_235)->work_0+(_S1865 + i_29))) == 0U;
    }
    if(_S1864)
    {
        return;
    }
    uint4 _S1866 = uint4(*((&kernelContext_235)->bonds_0+i_29)) ;
    uint _S1867 = _S1866.x;
    if((((as_type<uint>(((float4(*((&kernelContext_235)->crush_rw_0+_S1867)) ).w))) | (as_type<uint>(((float4(*((&kernelContext_235)->crush_rw_0+_S1866.y)) ).w)))) & 2U) == 0U)
    {
        return;
    }
    uint _S1868 = w_alive_0(&kernelContext_235);
    *((&kernelContext_235)->work_0+(_S1868 + i_29)) = 0U;
    uint _S1869 = a_dirty_0(&kernelContext_235);
    uint _S1870 = body_of_0(_S1867, &kernelContext_235);
    uint _S1871 = atomic_fetch_or_explicit((&kernelContext_235)->atom_0+(_S1869 + _S1870), 1U, memory_order_relaxed);
    uint _S1872 = a_counters_1(&kernelContext_235);
    uint _S1873 = atomic_fetch_add_explicit((&kernelContext_235)->atom_0+_S1872, 1U, memory_order_relaxed);
    uint _S1874 = atomic_fetch_add_explicit((&kernelContext_235)->atom_0+(_S1872 + 1U), 1U, memory_order_relaxed);
    event_broken_0(i_29, &kernelContext_235);
    return;
}

uint crush_pieces_0(uint k_73, KernelContext_0 thread* kernelContext_236)
{
    if((kernelContext_236->crush_0->mode_1) == 1U)
    {
        return kernelContext_236->crush_0->fragments_0;
    }
    if((kernelContext_236->crush_0->mode_1) == 2U)
    {
        return uint((float4(*(kernelContext_236->crush_tab_0+(2U * k_73 + 1U))) ).w + 0.5f);
    }
    return 0U;
}

[[kernel]] void fr_crush_mark(uint3 id_85 [[thread_position_in_grid]], FractureParams_0 constant* params_92 [[buffer(0)]], atomic<uint> device* atom_92 [[buffer(5)]], uint device* work_92 [[buffer(4)]], packed_float4 device* frames_92 [[buffer(6)]], packed_float4 device* bodies_92 [[buffer(7)]], packed_float4 device* chunks_92 [[buffer(1)]], uint device* sort_92 [[buffer(11)]], Record_natural_0 device* manifolds_92 [[buffer(10)]], packed_float4 device* loads_92 [[buffer(12)]], packed_uint4 device* bonds_93 [[buffer(2)]], uint device* args_92 [[buffer(13)]], uint device* info_92 [[buffer(15)]], ScanParams_0 constant* scan_92 [[buffer(14)]], packed_float4 device* parts_92 [[buffer(3)]], packed_float4 device* colliders_92 [[buffer(8)]], packed_uint4 device* collider_info_92 [[buffer(9)]], packed_float4 device* view_chunks_92 [[buffer(17)]], ViewParams_0 constant* view_92 [[buffer(16)]], packed_float4 device* objects_92 [[buffer(18)]], uint device* shot_bodies_92 [[buffer(20)]], float device* lines_92 [[buffer(19)]], packed_float4 device* geometry_92 [[buffer(21)]], packed_float4 device* shots_92 [[buffer(22)]], packed_float4 device* lp_92 [[buffer(24)]], packed_float4 device* ext_94 [[buffer(23)]], packed_float4 device* engine_contacts_92 [[buffer(25)]], CrushParams_0 constant* crush_92 [[buffer(28)]], packed_float4 device* crush_rw_92 [[buffer(27)]], packed_float4 device* crush_tab_92 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_237;
    (&kernelContext_237)->params_0 = params_92;
    (&kernelContext_237)->atom_0 = atom_92;
    (&kernelContext_237)->work_0 = work_92;
    (&kernelContext_237)->frames_0 = frames_92;
    (&kernelContext_237)->bodies_0 = bodies_92;
    (&kernelContext_237)->chunks_0 = chunks_92;
    (&kernelContext_237)->sort_0 = sort_92;
    (&kernelContext_237)->manifolds_0 = manifolds_92;
    (&kernelContext_237)->loads_0 = loads_92;
    (&kernelContext_237)->bonds_0 = bonds_93;
    (&kernelContext_237)->args_0 = args_92;
    (&kernelContext_237)->info_0 = info_92;
    (&kernelContext_237)->scan_0 = scan_92;
    (&kernelContext_237)->parts_0 = parts_92;
    (&kernelContext_237)->colliders_0 = colliders_92;
    (&kernelContext_237)->collider_info_0 = collider_info_92;
    (&kernelContext_237)->view_chunks_0 = view_chunks_92;
    (&kernelContext_237)->view_0 = view_92;
    (&kernelContext_237)->objects_0 = objects_92;
    (&kernelContext_237)->shot_bodies_0 = shot_bodies_92;
    (&kernelContext_237)->lines_0 = lines_92;
    (&kernelContext_237)->geometry_0 = geometry_92;
    (&kernelContext_237)->shots_0 = shots_92;
    (&kernelContext_237)->lp_0 = lp_92;
    (&kernelContext_237)->ext_0 = ext_94;
    (&kernelContext_237)->engine_contacts_0 = engine_contacts_92;
    (&kernelContext_237)->crush_0 = crush_92;
    (&kernelContext_237)->crush_rw_0 = crush_rw_92;
    (&kernelContext_237)->crush_tab_0 = crush_tab_92;
    threadgroup array<uint, int(1024)> g_scan_92;
    (&kernelContext_237)->g_scan_0 = &g_scan_92;
    threadgroup array<float, int(64)> g_il_best_92;
    (&kernelContext_237)->g_il_best_0 = &g_il_best_92;
    threadgroup array<float, int(256)> g_gi_x_92;
    (&kernelContext_237)->g_gi_x_0 = &g_gi_x_92;
    threadgroup array<float, int(256)> g_gi_m_92;
    (&kernelContext_237)->g_gi_m_0 = &g_gi_m_92;
    threadgroup array<uint, int(1280)> g_gi_lo_92;
    (&kernelContext_237)->g_gi_lo_0 = &g_gi_lo_92;
    threadgroup array<uint, int(1280)> g_gi_hi_92;
    (&kernelContext_237)->g_gi_hi_0 = &g_gi_hi_92;
    uint c_28 = id_85.x;
    bool _S1875;
    if(c_28 >= (params_92->chunk_count_0))
    {
        _S1875 = true;
    }
    else
    {
        _S1875 = ((&kernelContext_237)->crush_0->enabled_0) == 0U;
    }
    if(_S1875)
    {
        return;
    }
    uint _S1876 = crush_flags_0(c_28, &kernelContext_237);
    uint now_0;
    if((_S1876 & 2U) != 0U)
    {
        now_0 = 1U;
    }
    else
    {
        now_0 = 0U;
    }
    uint _S1877 = w_new_0(&kernelContext_237);
    uint device* _S1878 = (&kernelContext_237)->work_0+(_S1877 + c_28);
    bool _S1879 = now_0 != 0U;
    if(_S1879)
    {
        uint _S1880 = crush_pieces_0(c_28, &kernelContext_237);
        now_0 = 1048576U | _S1880;
    }
    else
    {
        now_0 = 0U;
    }
    *_S1878 = now_0;
    if(_S1879)
    {
        uint _S1881 = a_counters_1(&kernelContext_237);
        uint _S1882 = atomic_fetch_add_explicit((&kernelContext_237)->atom_0+(_S1881 + 10U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_crush_gate(uint3 id_86 [[thread_position_in_grid]], FractureParams_0 constant* params_93 [[buffer(0)]], atomic<uint> device* atom_93 [[buffer(5)]], uint device* work_93 [[buffer(4)]], packed_float4 device* frames_93 [[buffer(6)]], packed_float4 device* bodies_93 [[buffer(7)]], packed_float4 device* chunks_93 [[buffer(1)]], uint device* sort_93 [[buffer(11)]], Record_natural_0 device* manifolds_93 [[buffer(10)]], packed_float4 device* loads_93 [[buffer(12)]], packed_uint4 device* bonds_94 [[buffer(2)]], uint device* args_93 [[buffer(13)]], uint device* info_93 [[buffer(15)]], ScanParams_0 constant* scan_93 [[buffer(14)]], packed_float4 device* parts_93 [[buffer(3)]], packed_float4 device* colliders_93 [[buffer(8)]], packed_uint4 device* collider_info_93 [[buffer(9)]], packed_float4 device* view_chunks_93 [[buffer(17)]], ViewParams_0 constant* view_93 [[buffer(16)]], packed_float4 device* objects_93 [[buffer(18)]], uint device* shot_bodies_93 [[buffer(20)]], float device* lines_93 [[buffer(19)]], packed_float4 device* geometry_93 [[buffer(21)]], packed_float4 device* shots_93 [[buffer(22)]], packed_float4 device* lp_93 [[buffer(24)]], packed_float4 device* ext_95 [[buffer(23)]], packed_float4 device* engine_contacts_93 [[buffer(25)]], CrushParams_0 constant* crush_93 [[buffer(28)]], packed_float4 device* crush_rw_93 [[buffer(27)]], packed_float4 device* crush_tab_93 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_238;
    (&kernelContext_238)->params_0 = params_93;
    (&kernelContext_238)->atom_0 = atom_93;
    (&kernelContext_238)->work_0 = work_93;
    (&kernelContext_238)->frames_0 = frames_93;
    (&kernelContext_238)->bodies_0 = bodies_93;
    (&kernelContext_238)->chunks_0 = chunks_93;
    (&kernelContext_238)->sort_0 = sort_93;
    (&kernelContext_238)->manifolds_0 = manifolds_93;
    (&kernelContext_238)->loads_0 = loads_93;
    (&kernelContext_238)->bonds_0 = bonds_94;
    (&kernelContext_238)->args_0 = args_93;
    (&kernelContext_238)->info_0 = info_93;
    (&kernelContext_238)->scan_0 = scan_93;
    (&kernelContext_238)->parts_0 = parts_93;
    (&kernelContext_238)->colliders_0 = colliders_93;
    (&kernelContext_238)->collider_info_0 = collider_info_93;
    (&kernelContext_238)->view_chunks_0 = view_chunks_93;
    (&kernelContext_238)->view_0 = view_93;
    (&kernelContext_238)->objects_0 = objects_93;
    (&kernelContext_238)->shot_bodies_0 = shot_bodies_93;
    (&kernelContext_238)->lines_0 = lines_93;
    (&kernelContext_238)->geometry_0 = geometry_93;
    (&kernelContext_238)->shots_0 = shots_93;
    (&kernelContext_238)->lp_0 = lp_93;
    (&kernelContext_238)->ext_0 = ext_95;
    (&kernelContext_238)->engine_contacts_0 = engine_contacts_93;
    (&kernelContext_238)->crush_0 = crush_93;
    (&kernelContext_238)->crush_rw_0 = crush_rw_93;
    (&kernelContext_238)->crush_tab_0 = crush_tab_93;
    threadgroup array<uint, int(1024)> g_scan_93;
    (&kernelContext_238)->g_scan_0 = &g_scan_93;
    threadgroup array<float, int(64)> g_il_best_93;
    (&kernelContext_238)->g_il_best_0 = &g_il_best_93;
    threadgroup array<float, int(256)> g_gi_x_93;
    (&kernelContext_238)->g_gi_x_0 = &g_gi_x_93;
    threadgroup array<float, int(256)> g_gi_m_93;
    (&kernelContext_238)->g_gi_m_0 = &g_gi_m_93;
    threadgroup array<uint, int(1280)> g_gi_lo_93;
    (&kernelContext_238)->g_gi_lo_0 = &g_gi_lo_93;
    threadgroup array<uint, int(1280)> g_gi_hi_93;
    (&kernelContext_238)->g_gi_hi_0 = &g_gi_hi_93;
    bool run_2;
    if((crush_93->enabled_0) != 0U)
    {
        uint _S1883 = a_counters_1(&kernelContext_238);
        uint _S1884 = atomic_load_explicit((&kernelContext_238)->atom_0+(_S1883 + 10U), memory_order_relaxed);
        run_2 = _S1884 != 0U;
    }
    else
    {
        run_2 = false;
    }
    uint _S1885 = a_counters_1(&kernelContext_238);
    atomic_store_explicit((&kernelContext_238)->atom_0+(_S1885 + 10U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_238)->atom_0+(_S1885 + 7U), 0U, memory_order_relaxed);
    uint _S1886;
    if(run_2)
    {
        uint _S1887 = groups_of_0((&kernelContext_238)->params_0->chunk_count_0, 64U);
        _S1886 = _S1887;
    }
    else
    {
        _S1886 = 0U;
    }
    set_args_0(14U, _S1886, &kernelContext_238);
    if(run_2)
    {
        uint _S1888 = groups_of_0((&kernelContext_238)->params_0->chunk_count_0, 1024U);
        _S1886 = _S1888;
    }
    else
    {
        _S1886 = 0U;
    }
    set_args_0(15U, _S1886, &kernelContext_238);
    int _S1889;
    if(run_2)
    {
        _S1889 = int(1);
    }
    else
    {
        _S1889 = int(0);
    }
    set_args_0(16U, uint(_S1889), &kernelContext_238);
    return;
}

void write_body_0(uint slot_7, float3 pos_1, float inv_mass_15, const Quat_0 thread* orientation_0, float3 inv_inertia_1, uint flags_3, float3 v_22, float3 w_15, KernelContext_0 thread* kernelContext_239)
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
        uint b_67 = 7U * (slot_7 + region_8 * kernelContext_239->params_0->max_bodies_0);
        *(kernelContext_239->bodies_0+b_67) = packed_float4(float4(pos_1, inv_mass_15)) ;
        packed_float4 device* _S1890 = kernelContext_239->bodies_0+(b_67 + 1U);
        float4 _S1891 = quat_vec_0(orientation_0);
        *_S1890 = packed_float4(_S1891) ;
        *(kernelContext_239->bodies_0+(b_67 + 2U)) = packed_float4(float4(inv_inertia_1, (as_type<float>((flags_3))))) ;
        *(kernelContext_239->bodies_0+(b_67 + 3U)) = packed_float4(float4(v_22, 0.0f)) ;
        packed_float4 device* _S1892 = kernelContext_239->bodies_0+(b_67 + 4U);
        uint _S1893 = frame_step_0(kernelContext_239);
        *_S1892 = packed_float4(float4(w_15, (as_type<float>((_S1893))))) ;
        *(kernelContext_239->bodies_0+(b_67 + 5U)) = packed_float4(float4(0.0f) ) ;
        *(kernelContext_239->bodies_0+(b_67 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
        region_8 = region_8 + 1U;
    }
    return;
}

float3 crush_particle_spot_0(uint k_74, uint i_30)
{
    uint h_8 = (k_74 * 2654435761U) ^ (i_30 * 2246822519U);
    uint h_9 = (h_8 ^ (h_8 >> 15U)) * 739982445U;
    uint h_10 = (h_9 ^ (h_9 >> 12U)) * 695872825U;
    uint h_11 = h_10 ^ (h_10 >> 15U);
    return float3(float(h_11 & 1023U), float((h_11 >> 10U) & 1023U), float((h_11 * 2654435761U + 2135587861U) & 1023U)) / float3(511.5f)  - float3(1.0f) ;
}

[[kernel]] void fr_crush_apply(uint3 id_87 [[thread_position_in_grid]], FractureParams_0 constant* params_94 [[buffer(0)]], atomic<uint> device* atom_94 [[buffer(5)]], uint device* work_94 [[buffer(4)]], packed_float4 device* frames_94 [[buffer(6)]], packed_float4 device* bodies_94 [[buffer(7)]], packed_float4 device* chunks_94 [[buffer(1)]], uint device* sort_94 [[buffer(11)]], Record_natural_0 device* manifolds_94 [[buffer(10)]], packed_float4 device* loads_94 [[buffer(12)]], packed_uint4 device* bonds_95 [[buffer(2)]], uint device* args_94 [[buffer(13)]], uint device* info_94 [[buffer(15)]], ScanParams_0 constant* scan_94 [[buffer(14)]], packed_float4 device* parts_94 [[buffer(3)]], packed_float4 device* colliders_94 [[buffer(8)]], packed_uint4 device* collider_info_94 [[buffer(9)]], packed_float4 device* view_chunks_94 [[buffer(17)]], ViewParams_0 constant* view_94 [[buffer(16)]], packed_float4 device* objects_94 [[buffer(18)]], uint device* shot_bodies_94 [[buffer(20)]], float device* lines_94 [[buffer(19)]], packed_float4 device* geometry_94 [[buffer(21)]], packed_float4 device* shots_94 [[buffer(22)]], packed_float4 device* lp_94 [[buffer(24)]], packed_float4 device* ext_96 [[buffer(23)]], packed_float4 device* engine_contacts_94 [[buffer(25)]], CrushParams_0 constant* crush_94 [[buffer(28)]], packed_float4 device* crush_rw_94 [[buffer(27)]], packed_float4 device* crush_tab_94 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_240;
    (&kernelContext_240)->params_0 = params_94;
    (&kernelContext_240)->atom_0 = atom_94;
    (&kernelContext_240)->work_0 = work_94;
    (&kernelContext_240)->frames_0 = frames_94;
    (&kernelContext_240)->bodies_0 = bodies_94;
    (&kernelContext_240)->chunks_0 = chunks_94;
    (&kernelContext_240)->sort_0 = sort_94;
    (&kernelContext_240)->manifolds_0 = manifolds_94;
    (&kernelContext_240)->loads_0 = loads_94;
    (&kernelContext_240)->bonds_0 = bonds_95;
    (&kernelContext_240)->args_0 = args_94;
    (&kernelContext_240)->info_0 = info_94;
    (&kernelContext_240)->scan_0 = scan_94;
    (&kernelContext_240)->parts_0 = parts_94;
    (&kernelContext_240)->colliders_0 = colliders_94;
    (&kernelContext_240)->collider_info_0 = collider_info_94;
    (&kernelContext_240)->view_chunks_0 = view_chunks_94;
    (&kernelContext_240)->view_0 = view_94;
    (&kernelContext_240)->objects_0 = objects_94;
    (&kernelContext_240)->shot_bodies_0 = shot_bodies_94;
    (&kernelContext_240)->lines_0 = lines_94;
    (&kernelContext_240)->geometry_0 = geometry_94;
    (&kernelContext_240)->shots_0 = shots_94;
    (&kernelContext_240)->lp_0 = lp_94;
    (&kernelContext_240)->ext_0 = ext_96;
    (&kernelContext_240)->engine_contacts_0 = engine_contacts_94;
    (&kernelContext_240)->crush_0 = crush_94;
    (&kernelContext_240)->crush_rw_0 = crush_rw_94;
    (&kernelContext_240)->crush_tab_0 = crush_tab_94;
    threadgroup array<uint, int(1024)> g_scan_94;
    (&kernelContext_240)->g_scan_0 = &g_scan_94;
    threadgroup array<float, int(64)> g_il_best_94;
    (&kernelContext_240)->g_il_best_0 = &g_il_best_94;
    threadgroup array<float, int(256)> g_gi_x_94;
    (&kernelContext_240)->g_gi_x_0 = &g_gi_x_94;
    threadgroup array<float, int(256)> g_gi_m_94;
    (&kernelContext_240)->g_gi_m_0 = &g_gi_m_94;
    threadgroup array<uint, int(1280)> g_gi_lo_94;
    (&kernelContext_240)->g_gi_lo_0 = &g_gi_lo_94;
    threadgroup array<uint, int(1280)> g_gi_hi_94;
    (&kernelContext_240)->g_gi_hi_0 = &g_gi_hi_94;
    uint k_75 = id_87.x;
    bool _S1894;
    if(k_75 >= (params_94->chunk_count_0))
    {
        _S1894 = true;
    }
    else
    {
        _S1894 = ((&kernelContext_240)->crush_0->enabled_0) == 0U;
    }
    if(_S1894)
    {
        return;
    }
    float4 _S1895 = float4(*((&kernelContext_240)->crush_rw_0+k_75)) ;
    uint flags_4 = (as_type<uint>((_S1895.w)));
    if((flags_4 & 2U) == 0U)
    {
        return;
    }
    *((&kernelContext_240)->crush_rw_0+k_75) = packed_float4(float4(_S1895.xyz, (as_type<float>((flags_4 & 4294967293U))))) ;
    uint _S1896 = body_of_0(k_75, &kernelContext_240);
    uint bw_1 = 7U * _S1896;
    float4 _S1897 = float4(*((&kernelContext_240)->bodies_0+bw_1)) ;
    Quat_0 orientation_1 = quat_of_0(float4(*((&kernelContext_240)->bodies_0+(bw_1 + 1U))) );
    float3 v_23 = (float4(*((&kernelContext_240)->bodies_0+(bw_1 + 3U))) ).xyz;
    float3 w_16 = (float4(*((&kernelContext_240)->bodies_0+(bw_1 + 4U))) ).xyz;
    uint _S1898 = 4U * k_75;
    float mass_1 = (float4(*((&kernelContext_240)->chunks_0+_S1898)) ).w;
    float volume_1 = (float4(*((&kernelContext_240)->crush_tab_0+(2U * k_75 + 1U))) ).y;
    uint _S1899 = w_rank_0(&kernelContext_240);
    uint device* _S1900 = (&kernelContext_240)->work_0+(_S1899 + k_75);
    uint scanned_0 = *_S1900;
    uint rank_0 = *_S1900 / 1048576U;
    uint _S1901 = crush_pieces_0(k_75, &kernelContext_240);
    if(rank_0 >= ((&kernelContext_240)->crush_0->max_events_0))
    {
        uint _S1902 = a_counters_1(&kernelContext_240);
        uint _S1903 = atomic_fetch_or_explicit((&kernelContext_240)->atom_0+(_S1902 + 3U), 32U, memory_order_relaxed);
    }
    float3 _S1904 = float3(0.0f) ;
    if(((&kernelContext_240)->crush_0->mode_1) == 1U)
    {
        _S1894 = ((&kernelContext_240)->crush_0->fragments_0) > 0U;
    }
    else
    {
        _S1894 = false;
    }
    uint first_6;
    uint n_7;
    uint i_31;
    float3 half_2;
    if(_S1894)
    {
        uint _S1905 = a_counters_1(&kernelContext_240);
        uint _S1906 = atomic_load_explicit((&kernelContext_240)->atom_0+(_S1905 + 8U), memory_order_relaxed);
        uint base_6 = _S1906 + (scanned_0 & 1048575U);
        if((base_6 + (&kernelContext_240)->crush_0->fragments_0) > ((&kernelContext_240)->crush_0->pool_size_0))
        {
            uint _S1907 = atomic_fetch_or_explicit((&kernelContext_240)->atom_0+(_S1905 + 3U), 16U, memory_order_relaxed);
            first_6 = 4294967295U;
        }
        else
        {
            first_6 = base_6;
        }
        thread Quat_0 q_37;
        thread float3 origin_14;
        structure_pose_0(_S1896, bw_1, &q_37, &origin_14, &kernelContext_240);
        float3 _S1908 = (float4(*((&kernelContext_240)->chunks_0+_S1898)) ).xyz;
        thread Quat_0 _S1909 = q_37;
        float3 _S1910 = rotate_0(&_S1909, _S1908);
        float3 _S1911 = origin_14 + _S1910;
        uint _S1912 = 3U * k_75;
        Quat_0 _S1913 = quat_of_0(float4(*((&kernelContext_240)->geometry_0+(_S1912 + 2U))) );
        thread Quat_0 _S1914 = q_37;
        thread Quat_0 _S1915 = _S1913;
        Quat_0 _S1916 = quat_mul_0(&_S1914, &_S1915);
        thread Quat_0 _S1917 = _S1916;
        Quat_0 _S1918 = normalized_0(&_S1917);
        float3 h_12 = (float4(*((&kernelContext_240)->geometry_0+(_S1912 + 1U))) ).xyz;
        thread uint3 g_2 = uint3(int3(int(1)) );
        thread float3 cell_0 = h_12;
        n_7 = (&kernelContext_240)->crush_0->fragments_0;
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
                _S1894 = (cell_0.x) >= (cell_0.z);
            }
            else
            {
                _S1894 = false;
            }
            if(_S1894)
            {
                i_31 = 0U;
            }
            else
            {
                if((cell_0.y) >= (cell_0.z))
                {
                    i_31 = 1U;
                }
                else
                {
                    i_31 = 2U;
                }
            }
            g_2[i_31] = g_2[i_31] * 2U;
            cell_0[i_31] = cell_0[i_31] * 0.5f;
            n_7 = n_7 >> 1U;
        }
        float box_volume_0 = 8.0f * h_12.x * h_12.y * h_12.z;
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
        float mf_0 = mass_1 / float((&kernelContext_240)->crush_0->fragments_0);
        float _S1919 = half_3.y;
        float _S1920 = _S1919 * _S1919;
        float _S1921 = half_3.z;
        float _S1922 = _S1921 * _S1921;
        float _S1923 = half_3.x;
        float _S1924 = _S1923 * _S1923;
        float3 inertia_2 = float3((mf_0 / 3.0f))  * float3(_S1920 + _S1922, _S1924 + _S1922, _S1924 + _S1920);
        float3 _S1925 = float3(1.0f / inertia_2.x, 1.0f / inertia_2.y, 1.0f / inertia_2.z);
        thread Quat_0 _S1926 = _S1918;
        Quat_0 _S1927 = rot_to_rubble_0(&_S1926);
        float _S1928 = (float4(*((&kernelContext_240)->chunks_0+(_S1898 + 3U))) ).x;
        float _S1929 = min(_S1923, min(_S1919, _S1921));
        i_31 = 0U;
        for(;;)
        {
            if(i_31 < ((&kernelContext_240)->crush_0->fragments_0))
            {
                _S1894 = first_6 != 4294967295U;
            }
            else
            {
                _S1894 = false;
            }
            if(_S1894)
            {
            }
            else
            {
                break;
            }
            uint _S1930 = i_31 % g_2.x;
            uint _S1931 = i_31 / g_2.x;
            uint _S1932 = _S1931 % g_2.y;
            uint _S1933 = i_31 / (g_2.x * g_2.y);
            float3 offset_0 = - h_12 + cell_0 * (float3(2.0f)  * float3(uint3(_S1930, _S1932, _S1933)) + float3(1.0f) );
            thread Quat_0 _S1934 = _S1918;
            float3 _S1935 = rotate_0(&_S1934, offset_0);
            float3 x_10 = to_rubble_0(_S1911 + _S1935);
            float4 _S1936 = float4(*((&kernelContext_240)->crush_tab_0+((&kernelContext_240)->crush_0->t_pool_0 + first_6 + i_31))) ;
            uint fs_0 = (as_type<uint>((_S1936.x)));
            uint fc_3 = (as_type<uint>((_S1936.y)));
            float _S1937 = 1.0f / mf_0;
            float3 _S1938 = v_23 + cross(w_16, x_10 - _S1897.xyz);
            thread Quat_0 _S1939 = _S1927;
            write_body_0(fs_0, x_10, _S1937, &_S1939, _S1925, 9U, _S1938, w_16, &kernelContext_240);
            uint _S1940 = 4U * fc_3;
            uint _S1941 = _S1940 + 1U;
            *((&kernelContext_240)->colliders_0+_S1941) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_240)->colliders_0+_S1941)) ).w)) ;
            *((&kernelContext_240)->colliders_0+(_S1940 + 2U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
            *((&kernelContext_240)->colliders_0+(_S1940 + 3U)) = packed_float4(float4(half_3, length(half_3))) ;
            uint _S1942 = 2U * fc_3;
            *((&kernelContext_240)->collider_info_0+_S1942) = packed_uint4(uint4(4294967295U, fs_0, 2U, (uint4(*((&kernelContext_240)->collider_info_0+_S1942)) ).w)) ;
            *((&kernelContext_240)->collider_info_0+(_S1942 + 1U)) = packed_uint4(uint4((as_type<uint>((mf_0))), (as_type<uint>((_S1928))), (as_type<uint>((_S1929))), 0U)) ;
            i_31 = i_31 + 1U;
        }
        half_2 = half_3;
    }
    else
    {
        first_6 = 4294967295U;
        half_2 = _S1904;
    }
    if(((&kernelContext_240)->crush_0->mode_1) == 2U)
    {
        _S1894 = _S1901 > 0U;
    }
    else
    {
        _S1894 = false;
    }
    if(_S1894)
    {
        uint _S1943 = a_counters_1(&kernelContext_240);
        uint _S1944 = atomic_load_explicit((&kernelContext_240)->atom_0+(_S1943 + 8U), memory_order_relaxed);
        uint base_7 = _S1944 + (scanned_0 & 1048575U);
        if((base_7 + _S1901) > ((&kernelContext_240)->crush_0->pool_size_0))
        {
            uint _S1945 = atomic_fetch_or_explicit((&kernelContext_240)->atom_0+(_S1943 + 3U), 16U, memory_order_relaxed);
        }
        else
        {
            first_6 = base_7;
        }
        thread Quat_0 q_38;
        thread float3 origin_15;
        structure_pose_0(_S1896, bw_1, &q_38, &origin_15, &kernelContext_240);
        float3 _S1946 = (float4(*((&kernelContext_240)->chunks_0+_S1898)) ).xyz;
        thread Quat_0 _S1947 = q_38;
        float3 _S1948 = rotate_0(&_S1947, _S1946);
        float3 _S1949 = origin_15 + _S1948;
        uint _S1950 = 3U * k_75;
        Quat_0 _S1951 = quat_of_0(float4(*((&kernelContext_240)->geometry_0+(_S1950 + 2U))) );
        thread Quat_0 _S1952 = q_38;
        thread Quat_0 _S1953 = _S1951;
        Quat_0 _S1954 = quat_mul_0(&_S1952, &_S1953);
        thread Quat_0 _S1955 = _S1954;
        Quat_0 _S1956 = normalized_0(&_S1955);
        float3 h_13 = (float4(*((&kernelContext_240)->geometry_0+(_S1950 + 1U))) ).xyz;
        float _S1957 = float(_S1901);
        float _S1958 = max(pow(0.75f * volume_1 / (3.14159274101257324f * _S1957), 0.3333333432674408f), 0.0020000000949949f);
        float3 _S1959 = float3(_S1958, _S1958, _S1958);
        float _S1960 = mass_1 / _S1957;
        float _S1961 = (float4(*((&kernelContext_240)->chunks_0+(_S1898 + 3U))) ).x;
        float3 _S1962 = max(h_13 - float3(_S1958) , float3(0.25f)  * h_13);
        i_31 = 0U;
        float3 mean_4 = _S1904;
        for(;;)
        {
            if(i_31 < _S1901)
            {
            }
            else
            {
                break;
            }
            float3 mean_5 = mean_4 + crush_particle_spot_0(k_75, i_31);
            i_31 = i_31 + 1U;
            mean_4 = mean_5;
        }
        float3 mean_6 = mean_4 / float3(_S1957) ;
        i_31 = 0U;
        for(;;)
        {
            if(i_31 < _S1901)
            {
                _S1894 = first_6 != 4294967295U;
            }
            else
            {
                _S1894 = false;
            }
            if(_S1894)
            {
            }
            else
            {
                break;
            }
            float3 _S1963 = _S1962 * (crush_particle_spot_0(k_75, i_31) - mean_6);
            thread Quat_0 _S1964 = _S1956;
            float3 _S1965 = rotate_0(&_S1964, _S1963);
            float3 x_11 = to_rubble_0(_S1949 + _S1965);
            float4 _S1966 = float4(*((&kernelContext_240)->crush_tab_0+((&kernelContext_240)->crush_0->t_pool_0 + first_6 + i_31))) ;
            uint fs_1 = (as_type<uint>((_S1966.x)));
            uint fc_4 = (as_type<uint>((_S1966.y)));
            float _S1967 = 1.0f / _S1960;
            float4 _S1968 = float4(0.0f, 0.0f, 0.0f, 1.0f);
            Quat_0 _S1969 = quat_of_0(_S1968);
            if(((&kernelContext_240)->crush_0->particle_ccd_0) != 0U)
            {
                n_7 = 33024U;
            }
            else
            {
                n_7 = 0U;
            }
            uint _S1970 = 9U | n_7;
            float3 _S1971 = v_23 + cross(w_16, x_11 - _S1897.xyz);
            thread Quat_0 _S1972 = _S1969;
            write_body_0(fs_1, x_11, _S1967, &_S1972, _S1904, _S1970, _S1971, _S1904, &kernelContext_240);
            uint _S1973 = 4U * fc_4;
            uint _S1974 = _S1973 + 1U;
            *((&kernelContext_240)->colliders_0+_S1974) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_240)->colliders_0+_S1974)) ).w)) ;
            *((&kernelContext_240)->colliders_0+(_S1973 + 2U)) = packed_float4(_S1968) ;
            *((&kernelContext_240)->colliders_0+(_S1973 + 3U)) = packed_float4(float4(_S1958, 0.0f, 0.0f, _S1958)) ;
            uint _S1975 = 2U * fc_4;
            *((&kernelContext_240)->collider_info_0+_S1975) = packed_uint4(uint4(4294967295U, fs_1, 2U, (uint4(*((&kernelContext_240)->collider_info_0+_S1975)) ).w)) ;
            *((&kernelContext_240)->collider_info_0+(_S1975 + 1U)) = packed_uint4(uint4((as_type<uint>((_S1960))), (as_type<uint>((_S1961))), (as_type<uint>((_S1958))), 0U)) ;
            i_31 = i_31 + 1U;
        }
        half_2 = _S1959;
    }
    if(rank_0 < ((&kernelContext_240)->crush_0->max_events_0))
    {
        uint ev_0 = (&kernelContext_240)->crush_0->r_events_0 + 6U * rank_0;
        *((&kernelContext_240)->crush_rw_0+ev_0) = packed_float4(float4((as_type<float>((k_75))), (as_type<float>((first_6))), (as_type<float>((flags_4 >> 4U))), _S1895.x)) ;
        *((&kernelContext_240)->crush_rw_0+(ev_0 + 1U)) = packed_float4(float4(_S1897.xyz, mass_1)) ;
        packed_float4 device* _S1976 = (&kernelContext_240)->crush_rw_0+(ev_0 + 2U);
        thread Quat_0 _S1977 = orientation_1;
        float4 _S1978 = quat_vec_0(&_S1977);
        *_S1976 = packed_float4(_S1978) ;
        *((&kernelContext_240)->crush_rw_0+(ev_0 + 3U)) = packed_float4(float4(v_23, _S1895.y)) ;
        *((&kernelContext_240)->crush_rw_0+(ev_0 + 4U)) = packed_float4(float4(w_16, _S1895.z)) ;
        *((&kernelContext_240)->crush_rw_0+(ev_0 + 5U)) = packed_float4(float4(half_2, volume_1)) ;
    }
    float3 park_0 = float3(float(_S1896 % 1024U) * (&kernelContext_240)->crush_0->park_step_0, (&kernelContext_240)->crush_0->park_y_0, float(_S1896 / 1024U) * (&kernelContext_240)->crush_0->park_step_0);
    thread Quat_0 _S1979 = orientation_1;
    write_body_0(_S1896, park_0, 0.0f, &_S1979, _S1904, 0U, _S1904, _S1904, &kernelContext_240);
    return;
}

[[kernel]] void fr_crush_end(uint3 id_88 [[thread_position_in_grid]], FractureParams_0 constant* params_95 [[buffer(0)]], atomic<uint> device* atom_95 [[buffer(5)]], uint device* work_95 [[buffer(4)]], packed_float4 device* frames_95 [[buffer(6)]], packed_float4 device* bodies_95 [[buffer(7)]], packed_float4 device* chunks_95 [[buffer(1)]], uint device* sort_95 [[buffer(11)]], Record_natural_0 device* manifolds_95 [[buffer(10)]], packed_float4 device* loads_95 [[buffer(12)]], packed_uint4 device* bonds_96 [[buffer(2)]], uint device* args_95 [[buffer(13)]], uint device* info_95 [[buffer(15)]], ScanParams_0 constant* scan_95 [[buffer(14)]], packed_float4 device* parts_95 [[buffer(3)]], packed_float4 device* colliders_95 [[buffer(8)]], packed_uint4 device* collider_info_95 [[buffer(9)]], packed_float4 device* view_chunks_95 [[buffer(17)]], ViewParams_0 constant* view_95 [[buffer(16)]], packed_float4 device* objects_95 [[buffer(18)]], uint device* shot_bodies_95 [[buffer(20)]], float device* lines_95 [[buffer(19)]], packed_float4 device* geometry_95 [[buffer(21)]], packed_float4 device* shots_95 [[buffer(22)]], packed_float4 device* lp_95 [[buffer(24)]], packed_float4 device* ext_97 [[buffer(23)]], packed_float4 device* engine_contacts_95 [[buffer(25)]], CrushParams_0 constant* crush_95 [[buffer(28)]], packed_float4 device* crush_rw_95 [[buffer(27)]], packed_float4 device* crush_tab_95 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_241;
    (&kernelContext_241)->params_0 = params_95;
    (&kernelContext_241)->atom_0 = atom_95;
    (&kernelContext_241)->work_0 = work_95;
    (&kernelContext_241)->frames_0 = frames_95;
    (&kernelContext_241)->bodies_0 = bodies_95;
    (&kernelContext_241)->chunks_0 = chunks_95;
    (&kernelContext_241)->sort_0 = sort_95;
    (&kernelContext_241)->manifolds_0 = manifolds_95;
    (&kernelContext_241)->loads_0 = loads_95;
    (&kernelContext_241)->bonds_0 = bonds_96;
    (&kernelContext_241)->args_0 = args_95;
    (&kernelContext_241)->info_0 = info_95;
    (&kernelContext_241)->scan_0 = scan_95;
    (&kernelContext_241)->parts_0 = parts_95;
    (&kernelContext_241)->colliders_0 = colliders_95;
    (&kernelContext_241)->collider_info_0 = collider_info_95;
    (&kernelContext_241)->view_chunks_0 = view_chunks_95;
    (&kernelContext_241)->view_0 = view_95;
    (&kernelContext_241)->objects_0 = objects_95;
    (&kernelContext_241)->shot_bodies_0 = shot_bodies_95;
    (&kernelContext_241)->lines_0 = lines_95;
    (&kernelContext_241)->geometry_0 = geometry_95;
    (&kernelContext_241)->shots_0 = shots_95;
    (&kernelContext_241)->lp_0 = lp_95;
    (&kernelContext_241)->ext_0 = ext_97;
    (&kernelContext_241)->engine_contacts_0 = engine_contacts_95;
    (&kernelContext_241)->crush_0 = crush_95;
    (&kernelContext_241)->crush_rw_0 = crush_rw_95;
    (&kernelContext_241)->crush_tab_0 = crush_tab_95;
    threadgroup array<uint, int(1024)> g_scan_95;
    (&kernelContext_241)->g_scan_0 = &g_scan_95;
    threadgroup array<float, int(64)> g_il_best_95;
    (&kernelContext_241)->g_il_best_0 = &g_il_best_95;
    threadgroup array<float, int(256)> g_gi_x_95;
    (&kernelContext_241)->g_gi_x_0 = &g_gi_x_95;
    threadgroup array<float, int(256)> g_gi_m_95;
    (&kernelContext_241)->g_gi_m_0 = &g_gi_m_95;
    threadgroup array<uint, int(1280)> g_gi_lo_95;
    (&kernelContext_241)->g_gi_lo_0 = &g_gi_lo_95;
    threadgroup array<uint, int(1280)> g_gi_hi_95;
    (&kernelContext_241)->g_gi_hi_0 = &g_gi_hi_95;
    uint blocks_2 = (params_95->chunk_count_0 + 1023U) / 1024U;
    uint _S1980 = scan_new_blocks_0(&kernelContext_241);
    uint device* _S1981 = (&kernelContext_241)->work_0+(_S1980 + blocks_2);
    uint total_3 = *_S1981;
    uint events_0 = *_S1981 / 1048576U;
    uint _S1982 = a_counters_1(&kernelContext_241);
    atomic_store_explicit((&kernelContext_241)->atom_0+(_S1982 + 7U), events_0, memory_order_relaxed);
    uint _S1983 = atomic_fetch_add_explicit((&kernelContext_241)->atom_0+(_S1982 + 9U), events_0, memory_order_relaxed);
    uint _S1984 = atomic_fetch_add_explicit((&kernelContext_241)->atom_0+(_S1982 + 8U), total_3 & 1048575U, memory_order_relaxed);
    return;
}

