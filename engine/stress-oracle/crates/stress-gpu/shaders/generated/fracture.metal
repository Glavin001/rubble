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
    uint list_atom_0;
    uint list_stamp_0;
    uint list_touched_0;
    uint pad9_0;
    uint pad10_0;
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

[[kernel]] void fr_events_reset(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], atomic<uint> device* atom_1 [[buffer(5)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], uint device* sort_1 [[buffer(11)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], packed_float4 device* loads_1 [[buffer(12)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]], packed_float4 device* view_chunks_1 [[buffer(17)]], ViewParams_0 constant* view_1 [[buffer(16)]], packed_float4 device* objects_1 [[buffer(18)]], uint device* shot_bodies_1 [[buffer(20)]], float device* lines_1 [[buffer(19)]], packed_float4 device* geometry_1 [[buffer(21)]], packed_float4 device* shots_1 [[buffer(22)]], packed_float4 device* lp_1 [[buffer(24)]], packed_float4 device* ext_2 [[buffer(23)]], packed_float4 device* engine_contacts_1 [[buffer(25)]], CrushParams_0 constant* crush_1 [[buffer(28)]], packed_float4 device* crush_rw_1 [[buffer(27)]], packed_float4 device* crush_tab_1 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_79;
    (&kernelContext_79)->params_0 = params_1;
    (&kernelContext_79)->atom_0 = atom_1;
    (&kernelContext_79)->work_0 = work_1;
    (&kernelContext_79)->frames_0 = frames_1;
    (&kernelContext_79)->bodies_0 = bodies_1;
    (&kernelContext_79)->chunks_0 = chunks_1;
    (&kernelContext_79)->sort_0 = sort_1;
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
    (&kernelContext_79)->engine_contacts_0 = engine_contacts_1;
    (&kernelContext_79)->crush_0 = crush_1;
    (&kernelContext_79)->crush_rw_0 = crush_rw_1;
    (&kernelContext_79)->crush_tab_0 = crush_tab_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_79)->g_scan_0 = &g_scan_1;
    threadgroup array<float, int(64)> g_il_best_1;
    (&kernelContext_79)->g_il_best_0 = &g_il_best_1;
    uint _S524 = a_counters_1(&kernelContext_79);
    atomic_store_explicit((&kernelContext_79)->atom_0+(_S524 + 20U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_79)->atom_0+(_S524 + 21U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_79)->atom_0+(_S524 + 22U), 0U, memory_order_relaxed);
    return;
}

bool live_0(uint c_6, KernelContext_0 thread* kernelContext_80)
{
    bool _S525;
    if(c_6 < (kernelContext_80->params_0->chunk_used_0))
    {
        uint _S526 = body_of_0(c_6, kernelContext_80);
        _S525 = _S526 != 4294967295U;
    }
    else
    {
        _S525 = false;
    }
    return _S525;
}

uint l_pool_0(uint half_1, KernelContext_0 thread* kernelContext_81)
{
    return kernelContext_81->params_0->chunk_count_0 + 3U * kernelContext_81->params_0->pool_cap_0 * half_1;
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

uint l_vec_0(KernelContext_0 thread* kernelContext_82)
{
    return kernelContext_82->params_0->chunk_count_0 + 6U * kernelContext_82->params_0->pool_cap_0;
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
    thread KernelContext_0 kernelContext_83;
    (&kernelContext_83)->params_0 = params_2;
    (&kernelContext_83)->atom_0 = atom_2;
    (&kernelContext_83)->work_0 = work_2;
    (&kernelContext_83)->frames_0 = frames_2;
    (&kernelContext_83)->bodies_0 = bodies_2;
    (&kernelContext_83)->chunks_0 = chunks_2;
    (&kernelContext_83)->sort_0 = sort_2;
    (&kernelContext_83)->manifolds_0 = manifolds_2;
    (&kernelContext_83)->loads_0 = loads_2;
    (&kernelContext_83)->bonds_0 = bonds_2;
    (&kernelContext_83)->args_0 = args_2;
    (&kernelContext_83)->info_0 = info_2;
    (&kernelContext_83)->scan_0 = scan_2;
    (&kernelContext_83)->parts_0 = parts_2;
    (&kernelContext_83)->colliders_0 = colliders_2;
    (&kernelContext_83)->collider_info_0 = collider_info_2;
    (&kernelContext_83)->view_chunks_0 = view_chunks_2;
    (&kernelContext_83)->view_0 = view_2;
    (&kernelContext_83)->objects_0 = objects_2;
    (&kernelContext_83)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_83)->lines_0 = lines_2;
    (&kernelContext_83)->geometry_0 = geometry_2;
    (&kernelContext_83)->shots_0 = shots_2;
    (&kernelContext_83)->lp_0 = lp_2;
    (&kernelContext_83)->ext_0 = ext_3;
    (&kernelContext_83)->engine_contacts_0 = engine_contacts_2;
    (&kernelContext_83)->crush_0 = crush_2;
    (&kernelContext_83)->crush_rw_0 = crush_rw_2;
    (&kernelContext_83)->crush_tab_0 = crush_tab_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_83)->g_scan_0 = &g_scan_2;
    threadgroup array<float, int(64)> g_il_best_2;
    (&kernelContext_83)->g_il_best_0 = &g_il_best_2;
    uint k_22 = id_1.x;
    bool _S531 = live_0(k_22, &kernelContext_83);
    if(!_S531)
    {
        return;
    }
    uint _S532 = body_of_0(k_22, &kernelContext_83);
    uint b_33 = 7U * _S532;
    thread Quat_0 q_11;
    thread float3 origin_3;
    structure_pose_0(_S532, b_33, &q_11, &origin_3, &kernelContext_83);
    uint _S533 = 4U * k_22;
    float4 _S534 = float4(*((&kernelContext_83)->chunks_0+_S533)) ;
    float3 _S535 = _S534.xyz;
    thread Quat_0 _S536 = q_11;
    float3 _S537 = rotate_0(&_S536, _S535);
    float3 centre_0 = origin_3 + _S537;
    float inv_mass_2 = (float4(*((&kernelContext_83)->bodies_0+b_33)) ).w;
    float m_body_0;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint from_0 = (&kernelContext_83)->params_0->parity_0;
    uint to_0 = 1U - (&kernelContext_83)->params_0->parity_0;
    uint _S538 = w_pulses_0(&kernelContext_83);
    uint _S539 = _S538 + _S533;
    uint _S540 = _S539 + 2U * from_0;
    uint off_0 = *((&kernelContext_83)->work_0+_S540);
    uint device* _S541 = (&kernelContext_83)->work_0+(_S540 + 1U);
    uint carried_0 = *_S541;
    uint room_0 = *_S541 + (&kernelContext_83)->sort_0[(&kernelContext_83)->params_0->chunk_count_0 + 1U + k_22];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S542 = a_counters_1(&kernelContext_83);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_83)->atom_0+(_S542 + 5U + to_0), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_83)->params_0->pool_cap_0))
        {
            uint _S543 = atomic_fetch_or_explicit((&kernelContext_83)->atom_0+(_S542 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_83)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_83)->params_0->pool_cap_0 - base_1;
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
    uint _S545 = (&kernelContext_83)->params_0->chunk_count_0;
    float _S546 = _S534.w;
    float4 _S547 = float4(*((&kernelContext_83)->chunks_0+(_S533 + 3U))) ;
    float _S548 = (&kernelContext_83)->params_0->band_0;
    float _S549 = (&kernelContext_83)->params_0->dt_0;
    bool _S550 = ((&kernelContext_83)->params_0->impact_peak_0) != 0U;
    uint _S551 = l_pool_0(from_0, &kernelContext_83);
    uint _S552 = _S551 + 3U * off_0;
    uint _S553 = l_pool_0(to_0, &kernelContext_83);
    uint _S554 = _S553 + 3U * base_0;
    for(;;)
    {
        float3 _S555 = float3(0.0f) ;
        uint _S556 = (&kernelContext_83)->sort_0[k_22];
        uint _S557 = (&kernelContext_83)->sort_0[_S545 + 1U + k_22];
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
                uint key_1 = (uint4((&kernelContext_83)->manifolds_0[_S556 + q3_0].ids_0) ).y + 1U;
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
            Record_natural_0 r_19 = (&kernelContext_83)->manifolds_0[_S556 + p_10];
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
            float4 _S564 = float4(*((&kernelContext_83)->loads_0+_S563)) ;
            thread float4 a_5 = _S564;
            float4 _S565 = float4(*((&kernelContext_83)->loads_0+(_S563 + 1U))) ;
            float4 _S566 = float4(*((&kernelContext_83)->loads_0+(_S563 + 2U))) ;
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
                *((&kernelContext_83)->loads_0+_S569) = packed_float4(a_5) ;
                *((&kernelContext_83)->loads_0+(_S569 + 1U)) = packed_float4(_S565) ;
                *((&kernelContext_83)->loads_0+(_S569 + 2U)) = packed_float4(_S566) ;
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
            Record_natural_0 r_20 = (&kernelContext_83)->manifolds_0[_S556 + q3_0];
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
                    *((&kernelContext_83)->loads_0+_S577) = packed_float4(float4(- _S549, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_83)->loads_0+(_S577 + 1U)) = packed_float4(float4((float4(r_20.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_83)->loads_0+(_S577 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
        uint _S578 = a_counters_1(&kernelContext_83);
        uint _S579 = atomic_fetch_or_explicit((&kernelContext_83)->atom_0+(_S578 + 3U), 8U, memory_order_relaxed);
    }
    uint _S580 = _S539 + 2U * to_0;
    *((&kernelContext_83)->work_0+_S580) = base_0;
    *((&kernelContext_83)->work_0+(_S580 + 1U)) = kept_0;
    *((&kernelContext_83)->loads_0+k_22) = packed_float4(float4(length(f_total_0), 0.0f, 0.0f, 0.0f)) ;
    if(((&kernelContext_83)->params_0->load_model_0) == 1U)
    {
        uint _S581 = l_vec_0(&kernelContext_83);
        uint _S582 = _S581 + 2U * k_22;
        *((&kernelContext_83)->loads_0+_S582) = packed_float4(float4(f_total_0, 0.0f)) ;
        *((&kernelContext_83)->loads_0+(_S582 + 1U)) = packed_float4(float4(t_total_0, 0.0f)) ;
    }
    return;
}

uint a_strike_dist_0(KernelContext_0 thread* kernelContext_84)
{
    uint _S583 = a_cs_0(0U, kernelContext_84);
    return _S583 + 12U * kernelContext_84->params_0->body_count_0;
}

uint a_events_0(KernelContext_0 thread* kernelContext_85)
{
    uint _S584 = a_strike_dist_0(kernelContext_85);
    return _S584 + kernelContext_85->params_0->chunk_count_0;
}

void event_broken_0(uint bond_0, KernelContext_0 thread* kernelContext_86)
{
    if((kernelContext_86->params_0->ev_broken_cap_0) == 0U)
    {
        return;
    }
    uint _S585 = a_counters_1(kernelContext_86);
    uint i_6 = atomic_fetch_add_explicit(kernelContext_86->atom_0+(_S585 + 20U), 1U, memory_order_relaxed);
    if(i_6 < (kernelContext_86->params_0->ev_broken_cap_0))
    {
        uint _S586 = a_events_0(kernelContext_86);
        atomic_store_explicit(kernelContext_86->atom_0+(_S586 + i_6), bond_0, memory_order_relaxed);
    }
    else
    {
        uint _S587 = atomic_fetch_or_explicit(kernelContext_86->atom_0+(_S585 + 3U), 128U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_breaks(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], atomic<uint> device* atom_3 [[buffer(5)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], packed_float4 device* lp_3 [[buffer(24)]], packed_float4 device* ext_4 [[buffer(23)]], packed_float4 device* engine_contacts_3 [[buffer(25)]], CrushParams_0 constant* crush_3 [[buffer(28)]], packed_float4 device* crush_rw_3 [[buffer(27)]], packed_float4 device* crush_tab_3 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_87;
    (&kernelContext_87)->params_0 = params_3;
    (&kernelContext_87)->atom_0 = atom_3;
    (&kernelContext_87)->work_0 = work_3;
    (&kernelContext_87)->frames_0 = frames_3;
    (&kernelContext_87)->bodies_0 = bodies_3;
    (&kernelContext_87)->chunks_0 = chunks_3;
    (&kernelContext_87)->sort_0 = sort_3;
    (&kernelContext_87)->manifolds_0 = manifolds_3;
    (&kernelContext_87)->loads_0 = loads_3;
    (&kernelContext_87)->bonds_0 = bonds_3;
    (&kernelContext_87)->args_0 = args_3;
    (&kernelContext_87)->info_0 = info_3;
    (&kernelContext_87)->scan_0 = scan_3;
    (&kernelContext_87)->parts_0 = parts_3;
    (&kernelContext_87)->colliders_0 = colliders_3;
    (&kernelContext_87)->collider_info_0 = collider_info_3;
    (&kernelContext_87)->view_chunks_0 = view_chunks_3;
    (&kernelContext_87)->view_0 = view_3;
    (&kernelContext_87)->objects_0 = objects_3;
    (&kernelContext_87)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_87)->lines_0 = lines_3;
    (&kernelContext_87)->geometry_0 = geometry_3;
    (&kernelContext_87)->shots_0 = shots_3;
    (&kernelContext_87)->lp_0 = lp_3;
    (&kernelContext_87)->ext_0 = ext_4;
    (&kernelContext_87)->engine_contacts_0 = engine_contacts_3;
    (&kernelContext_87)->crush_0 = crush_3;
    (&kernelContext_87)->crush_rw_0 = crush_rw_3;
    (&kernelContext_87)->crush_tab_0 = crush_tab_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_87)->g_scan_0 = &g_scan_3;
    threadgroup array<float, int(64)> g_il_best_3;
    (&kernelContext_87)->g_il_best_0 = &g_il_best_3;
    uint i_7 = id_2.x;
    bool _S588;
    if(i_7 >= (params_3->bond_used_0))
    {
        _S588 = true;
    }
    else
    {
        uint _S589 = w_alive_0(&kernelContext_87);
        _S588 = (*((&kernelContext_87)->work_0+(_S589 + i_7))) == 0U;
    }
    if(_S588)
    {
        return;
    }
    uint4 _S590 = uint4(*((&kernelContext_87)->bonds_0+i_7)) ;
    uint _S591 = _S590.x;
    if((max((float4(*((&kernelContext_87)->loads_0+_S591)) ).x, (float4(*((&kernelContext_87)->loads_0+_S590.y)) ).x)) > (as_type<float>((_S590.z))))
    {
        uint _S592 = w_alive_0(&kernelContext_87);
        *((&kernelContext_87)->work_0+(_S592 + i_7)) = 0U;
        uint _S593 = a_dirty_0(&kernelContext_87);
        uint _S594 = body_of_0(_S591, &kernelContext_87);
        uint _S595 = atomic_fetch_or_explicit((&kernelContext_87)->atom_0+(_S593 + _S594), 1U, memory_order_relaxed);
        uint _S596 = a_counters_1(&kernelContext_87);
        uint _S597 = atomic_fetch_add_explicit((&kernelContext_87)->atom_0+_S596, 1U, memory_order_relaxed);
        uint _S598 = atomic_fetch_add_explicit((&kernelContext_87)->atom_0+(_S596 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_7, &kernelContext_87);
    }
    return;
}

[[kernel]] void fr_host_breaks(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], atomic<uint> device* atom_4 [[buffer(5)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], packed_float4 device* lp_4 [[buffer(24)]], packed_float4 device* ext_5 [[buffer(23)]], packed_float4 device* engine_contacts_4 [[buffer(25)]], CrushParams_0 constant* crush_4 [[buffer(28)]], packed_float4 device* crush_rw_4 [[buffer(27)]], packed_float4 device* crush_tab_4 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_88;
    (&kernelContext_88)->params_0 = params_4;
    (&kernelContext_88)->atom_0 = atom_4;
    (&kernelContext_88)->work_0 = work_4;
    (&kernelContext_88)->frames_0 = frames_4;
    (&kernelContext_88)->bodies_0 = bodies_4;
    (&kernelContext_88)->chunks_0 = chunks_4;
    (&kernelContext_88)->sort_0 = sort_4;
    (&kernelContext_88)->manifolds_0 = manifolds_4;
    (&kernelContext_88)->loads_0 = loads_4;
    (&kernelContext_88)->bonds_0 = bonds_4;
    (&kernelContext_88)->args_0 = args_4;
    (&kernelContext_88)->info_0 = info_4;
    (&kernelContext_88)->scan_0 = scan_4;
    (&kernelContext_88)->parts_0 = parts_4;
    (&kernelContext_88)->colliders_0 = colliders_4;
    (&kernelContext_88)->collider_info_0 = collider_info_4;
    (&kernelContext_88)->view_chunks_0 = view_chunks_4;
    (&kernelContext_88)->view_0 = view_4;
    (&kernelContext_88)->objects_0 = objects_4;
    (&kernelContext_88)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_88)->lines_0 = lines_4;
    (&kernelContext_88)->geometry_0 = geometry_4;
    (&kernelContext_88)->shots_0 = shots_4;
    (&kernelContext_88)->lp_0 = lp_4;
    (&kernelContext_88)->ext_0 = ext_5;
    (&kernelContext_88)->engine_contacts_0 = engine_contacts_4;
    (&kernelContext_88)->crush_0 = crush_4;
    (&kernelContext_88)->crush_rw_0 = crush_rw_4;
    (&kernelContext_88)->crush_tab_0 = crush_tab_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_88)->g_scan_0 = &g_scan_4;
    threadgroup array<float, int(64)> g_il_best_4;
    (&kernelContext_88)->g_il_best_0 = &g_il_best_4;
    uint i_8 = id_3.x;
    uint _S599 = w_host_breaks_0(&kernelContext_88);
    if(i_8 >= (*((&kernelContext_88)->work_0+_S599)))
    {
        return;
    }
    uint device* _S600 = (&kernelContext_88)->work_0+(_S599 + 1U + i_8);
    uint b_34 = *_S600;
    bool _S601;
    if((*_S600) >= ((&kernelContext_88)->params_0->bond_count_0))
    {
        _S601 = true;
    }
    else
    {
        uint _S602 = w_alive_0(&kernelContext_88);
        _S601 = (*((&kernelContext_88)->work_0+(_S602 + b_34))) == 0U;
    }
    if(_S601)
    {
        return;
    }
    uint _S603 = w_alive_0(&kernelContext_88);
    *((&kernelContext_88)->work_0+(_S603 + b_34)) = 0U;
    uint _S604 = a_dirty_0(&kernelContext_88);
    uint _S605 = body_of_0((uint4(*((&kernelContext_88)->bonds_0+b_34)) ).x, &kernelContext_88);
    uint _S606 = atomic_fetch_or_explicit((&kernelContext_88)->atom_0+(_S604 + _S605), 1U, memory_order_relaxed);
    uint _S607 = a_counters_1(&kernelContext_88);
    uint _S608 = atomic_fetch_add_explicit((&kernelContext_88)->atom_0+_S607, 1U, memory_order_relaxed);
    uint _S609 = atomic_fetch_add_explicit((&kernelContext_88)->atom_0+(_S607 + 1U), 1U, memory_order_relaxed);
    event_broken_0(b_34, &kernelContext_88);
    return;
}

uint groups_of_0(uint n_3, uint size_0)
{
    uint _S610 = (n_3 + size_0 - 1U) / size_0;
    return _S610;
}

void set_args_0(uint at_8, uint x_3, KernelContext_0 thread* kernelContext_89)
{
    uint _S611 = 3U * at_8;
    *(kernelContext_89->args_0+_S611) = x_3;
    *(kernelContext_89->args_0+(_S611 + 1U)) = 1U;
    *(kernelContext_89->args_0+(_S611 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], atomic<uint> device* atom_5 [[buffer(5)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], packed_float4 device* lp_5 [[buffer(24)]], packed_float4 device* ext_6 [[buffer(23)]], packed_float4 device* engine_contacts_5 [[buffer(25)]], CrushParams_0 constant* crush_5 [[buffer(28)]], packed_float4 device* crush_rw_5 [[buffer(27)]], packed_float4 device* crush_tab_5 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_90;
    (&kernelContext_90)->params_0 = params_5;
    (&kernelContext_90)->atom_0 = atom_5;
    (&kernelContext_90)->work_0 = work_5;
    (&kernelContext_90)->frames_0 = frames_5;
    (&kernelContext_90)->bodies_0 = bodies_5;
    (&kernelContext_90)->chunks_0 = chunks_5;
    (&kernelContext_90)->sort_0 = sort_5;
    (&kernelContext_90)->manifolds_0 = manifolds_5;
    (&kernelContext_90)->loads_0 = loads_5;
    (&kernelContext_90)->bonds_0 = bonds_5;
    (&kernelContext_90)->args_0 = args_5;
    (&kernelContext_90)->info_0 = info_5;
    (&kernelContext_90)->scan_0 = scan_5;
    (&kernelContext_90)->parts_0 = parts_5;
    (&kernelContext_90)->colliders_0 = colliders_5;
    (&kernelContext_90)->collider_info_0 = collider_info_5;
    (&kernelContext_90)->view_chunks_0 = view_chunks_5;
    (&kernelContext_90)->view_0 = view_5;
    (&kernelContext_90)->objects_0 = objects_5;
    (&kernelContext_90)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_90)->lines_0 = lines_5;
    (&kernelContext_90)->geometry_0 = geometry_5;
    (&kernelContext_90)->shots_0 = shots_5;
    (&kernelContext_90)->lp_0 = lp_5;
    (&kernelContext_90)->ext_0 = ext_6;
    (&kernelContext_90)->engine_contacts_0 = engine_contacts_5;
    (&kernelContext_90)->crush_0 = crush_5;
    (&kernelContext_90)->crush_rw_0 = crush_rw_5;
    (&kernelContext_90)->crush_tab_0 = crush_tab_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_90)->g_scan_0 = &g_scan_5;
    threadgroup array<float, int(64)> g_il_best_5;
    (&kernelContext_90)->g_il_best_0 = &g_il_best_5;
    uint _S612 = a_counters_1(&kernelContext_90);
    uint _S613 = atomic_load_explicit((&kernelContext_90)->atom_0+_S612, memory_order_relaxed);
    bool run_0 = _S613 != 0U;
    atomic_store_explicit((&kernelContext_90)->atom_0+_S612, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_90)->atom_0+(_S612 + 5U + (&kernelContext_90)->params_0->parity_0), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S614 = atomic_fetch_add_explicit((&kernelContext_90)->atom_0+(_S612 + 4U), 1U, memory_order_relaxed);
    }
    uint _S615;
    if(run_0)
    {
        uint _S616 = groups_of_0((&kernelContext_90)->params_0->chunk_used_0, 64U);
        _S615 = _S616;
    }
    else
    {
        _S615 = 0U;
    }
    set_args_0(0U, _S615, &kernelContext_90);
    if(run_0)
    {
        uint _S617 = groups_of_0((&kernelContext_90)->params_0->bond_used_0, 64U);
        _S615 = _S617;
    }
    else
    {
        _S615 = 0U;
    }
    set_args_0(1U, _S615, &kernelContext_90);
    if(run_0)
    {
        uint _S618 = groups_of_0((&kernelContext_90)->params_0->body_count_0, 64U);
        _S615 = _S618;
    }
    else
    {
        _S615 = 0U;
    }
    set_args_0(2U, _S615, &kernelContext_90);
    if(run_0)
    {
        uint _S619 = groups_of_0((&kernelContext_90)->params_0->chunk_used_0, 1024U);
        _S615 = _S619;
    }
    else
    {
        _S615 = 0U;
    }
    set_args_0(3U, _S615, &kernelContext_90);
    if(run_0)
    {
        uint _S620 = groups_of_0((&kernelContext_90)->params_0->body_count_0, 1024U);
        _S615 = _S620;
    }
    else
    {
        _S615 = 0U;
    }
    set_args_0(4U, _S615, &kernelContext_90);
    int _S621;
    if(run_0)
    {
        _S621 = int(1);
    }
    else
    {
        _S621 = int(0);
    }
    set_args_0(5U, uint(_S621), &kernelContext_90);
    return;
}

bool dirty_0(uint body_1, KernelContext_0 thread* kernelContext_91)
{
    uint _S622 = a_dirty_0(kernelContext_91);
    uint _S623 = atomic_load_explicit(kernelContext_91->atom_0+(_S622 + body_1), memory_order_relaxed);
    return _S623 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_92)
{
    return kernelContext_92->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_93)
{
    return 2U * kernelContext_93->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], atomic<uint> device* atom_6 [[buffer(5)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], packed_float4 device* lp_6 [[buffer(24)]], packed_float4 device* ext_7 [[buffer(23)]], packed_float4 device* engine_contacts_6 [[buffer(25)]], CrushParams_0 constant* crush_6 [[buffer(28)]], packed_float4 device* crush_rw_6 [[buffer(27)]], packed_float4 device* crush_tab_6 [[buffer(26)]])
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
    uint c_7 = id_5.x;
    bool _S624 = live_0(c_7, &kernelContext_94);
    bool _S625;
    if(!_S624)
    {
        _S625 = true;
    }
    else
    {
        uint _S626 = body_of_0(c_7, &kernelContext_94);
        bool _S627 = dirty_0(_S626, &kernelContext_94);
        _S625 = !_S627;
    }
    if(_S625)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_94)->atom_0+(a_parent_0() + c_7), c_7, memory_order_relaxed);
    uint _S628 = a_count_0(&kernelContext_94);
    atomic_store_explicit((&kernelContext_94)->atom_0+(_S628 + c_7), 0U, memory_order_relaxed);
    uint _S629 = a_anchor_0(&kernelContext_94);
    atomic_store_explicit((&kernelContext_94)->atom_0+(_S629 + c_7), 0U, memory_order_relaxed);
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
        uint _S630 = a_sums_0(&kernelContext_94);
        atomic_store_explicit((&kernelContext_94)->atom_0+(_S630 + 20U * c_7 + w_3), 0U, memory_order_relaxed);
        w_3 = w_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], atomic<uint> device* atom_7 [[buffer(5)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], packed_float4 device* lp_7 [[buffer(24)]], packed_float4 device* ext_8 [[buffer(23)]], packed_float4 device* engine_contacts_7 [[buffer(25)]], CrushParams_0 constant* crush_7 [[buffer(28)]], packed_float4 device* crush_rw_7 [[buffer(27)]], packed_float4 device* crush_tab_7 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_95;
    (&kernelContext_95)->params_0 = params_7;
    (&kernelContext_95)->atom_0 = atom_7;
    (&kernelContext_95)->work_0 = work_7;
    (&kernelContext_95)->frames_0 = frames_7;
    (&kernelContext_95)->bodies_0 = bodies_7;
    (&kernelContext_95)->chunks_0 = chunks_7;
    (&kernelContext_95)->sort_0 = sort_7;
    (&kernelContext_95)->manifolds_0 = manifolds_7;
    (&kernelContext_95)->loads_0 = loads_7;
    (&kernelContext_95)->bonds_0 = bonds_7;
    (&kernelContext_95)->args_0 = args_7;
    (&kernelContext_95)->info_0 = info_7;
    (&kernelContext_95)->scan_0 = scan_7;
    (&kernelContext_95)->parts_0 = parts_7;
    (&kernelContext_95)->colliders_0 = colliders_7;
    (&kernelContext_95)->collider_info_0 = collider_info_7;
    (&kernelContext_95)->view_chunks_0 = view_chunks_7;
    (&kernelContext_95)->view_0 = view_7;
    (&kernelContext_95)->objects_0 = objects_7;
    (&kernelContext_95)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_95)->lines_0 = lines_7;
    (&kernelContext_95)->geometry_0 = geometry_7;
    (&kernelContext_95)->shots_0 = shots_7;
    (&kernelContext_95)->lp_0 = lp_7;
    (&kernelContext_95)->ext_0 = ext_8;
    (&kernelContext_95)->engine_contacts_0 = engine_contacts_7;
    (&kernelContext_95)->crush_0 = crush_7;
    (&kernelContext_95)->crush_rw_0 = crush_rw_7;
    (&kernelContext_95)->crush_tab_0 = crush_tab_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_95)->g_scan_0 = &g_scan_7;
    threadgroup array<float, int(64)> g_il_best_7;
    (&kernelContext_95)->g_il_best_0 = &g_il_best_7;
    uint s_7 = id_6.x;
    bool _S631;
    if(s_7 >= (params_7->body_count_0))
    {
        _S631 = true;
    }
    else
    {
        bool _S632 = dirty_0(s_7, &kernelContext_95);
        _S631 = !_S632;
    }
    if(_S631)
    {
        return;
    }
    uint _S633 = a_best_count_0(&kernelContext_95);
    atomic_store_explicit((&kernelContext_95)->atom_0+(_S633 + s_7), 0U, memory_order_relaxed);
    uint _S634 = a_best_root_0(&kernelContext_95);
    atomic_store_explicit((&kernelContext_95)->atom_0+(_S634 + s_7), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_4, KernelContext_0 thread* kernelContext_96)
{
    uint _S635 = a_parent_0();
    uint _S636 = atomic_load_explicit(kernelContext_96->atom_0+(_S635 + x_4), memory_order_relaxed);
    uint p_11 = _S636;
    uint _S637 = x_4;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S638;
        if(guard_0 < 1048576U)
        {
            _S638 = p_11 != _S637;
        }
        else
        {
            _S638 = false;
        }
        if(_S638)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_96->atom_0+(_S635 + p_11), memory_order_relaxed);
        if(gp_0 != p_11)
        {
            uint _S639;
            {
            uint _metal_cas_comparand = p_11;
            atomic_compare_exchange_weak_explicit(kernelContext_96->atom_0+(_S635 + _S637), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S639 = _metal_cas_comparand;
            }
        }
        uint _S640 = guard_0 + 1U;
        _S637 = p_11;
        p_11 = gp_0;
        guard_0 = _S640;
    }
    return _S637;
}

[[kernel]] void fr_uf_union(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], atomic<uint> device* atom_8 [[buffer(5)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], packed_float4 device* lp_8 [[buffer(24)]], packed_float4 device* ext_9 [[buffer(23)]], packed_float4 device* engine_contacts_8 [[buffer(25)]], CrushParams_0 constant* crush_8 [[buffer(28)]], packed_float4 device* crush_rw_8 [[buffer(27)]], packed_float4 device* crush_tab_8 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_97;
    (&kernelContext_97)->params_0 = params_8;
    (&kernelContext_97)->atom_0 = atom_8;
    (&kernelContext_97)->work_0 = work_8;
    (&kernelContext_97)->frames_0 = frames_8;
    (&kernelContext_97)->bodies_0 = bodies_8;
    (&kernelContext_97)->chunks_0 = chunks_8;
    (&kernelContext_97)->sort_0 = sort_8;
    (&kernelContext_97)->manifolds_0 = manifolds_8;
    (&kernelContext_97)->loads_0 = loads_8;
    (&kernelContext_97)->bonds_0 = bonds_8;
    (&kernelContext_97)->args_0 = args_8;
    (&kernelContext_97)->info_0 = info_8;
    (&kernelContext_97)->scan_0 = scan_8;
    (&kernelContext_97)->parts_0 = parts_8;
    (&kernelContext_97)->colliders_0 = colliders_8;
    (&kernelContext_97)->collider_info_0 = collider_info_8;
    (&kernelContext_97)->view_chunks_0 = view_chunks_8;
    (&kernelContext_97)->view_0 = view_8;
    (&kernelContext_97)->objects_0 = objects_8;
    (&kernelContext_97)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_97)->lines_0 = lines_8;
    (&kernelContext_97)->geometry_0 = geometry_8;
    (&kernelContext_97)->shots_0 = shots_8;
    (&kernelContext_97)->lp_0 = lp_8;
    (&kernelContext_97)->ext_0 = ext_9;
    (&kernelContext_97)->engine_contacts_0 = engine_contacts_8;
    (&kernelContext_97)->crush_0 = crush_8;
    (&kernelContext_97)->crush_rw_0 = crush_rw_8;
    (&kernelContext_97)->crush_tab_0 = crush_tab_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_97)->g_scan_0 = &g_scan_8;
    threadgroup array<float, int(64)> g_il_best_8;
    (&kernelContext_97)->g_il_best_0 = &g_il_best_8;
    uint i_9 = id_7.x;
    bool _S641;
    if(i_9 >= (params_8->bond_used_0))
    {
        _S641 = true;
    }
    else
    {
        uint _S642 = w_alive_0(&kernelContext_97);
        _S641 = (*((&kernelContext_97)->work_0+(_S642 + i_9))) == 0U;
    }
    if(_S641)
    {
        return;
    }
    uint4 _S643 = uint4(*((&kernelContext_97)->bonds_0+i_9)) ;
    uint _S644 = _S643.x;
    uint _S645 = body_of_0(_S644, &kernelContext_97);
    bool _S646 = dirty_0(_S645, &kernelContext_97);
    if(!_S646)
    {
        return;
    }
    uint _S647 = _S643.y;
    uint a_6 = _S644;
    uint b_35 = _S647;
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
        uint _S648 = find_0(a_6, &kernelContext_97);
        uint _S649 = find_0(b_35, &kernelContext_97);
        if(_S648 == _S649)
        {
            return;
        }
        uint _S650 = max(_S648, _S649);
        uint _S651 = min(_S648, _S649);
        uint was_0;
        {
        uint _metal_cas_comparand = _S650;
        atomic_compare_exchange_weak_explicit((&kernelContext_97)->atom_0+(a_parent_0() + _S650), &_metal_cas_comparand, _S651, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S650)
        {
            return;
        }
        uint _S652 = guard_1 + 1U;
        a_6 = _S650;
        b_35 = _S651;
        guard_1 = _S652;
    }
    uint _S653 = a_counters_1(&kernelContext_97);
    uint _S654 = atomic_fetch_or_explicit((&kernelContext_97)->atom_0+(_S653 + 3U), 4U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], atomic<uint> device* atom_9 [[buffer(5)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], packed_float4 device* lp_9 [[buffer(24)]], packed_float4 device* ext_10 [[buffer(23)]], packed_float4 device* engine_contacts_9 [[buffer(25)]], CrushParams_0 constant* crush_9 [[buffer(28)]], packed_float4 device* crush_rw_9 [[buffer(27)]], packed_float4 device* crush_tab_9 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_98;
    (&kernelContext_98)->params_0 = params_9;
    (&kernelContext_98)->atom_0 = atom_9;
    (&kernelContext_98)->work_0 = work_9;
    (&kernelContext_98)->frames_0 = frames_9;
    (&kernelContext_98)->bodies_0 = bodies_9;
    (&kernelContext_98)->chunks_0 = chunks_9;
    (&kernelContext_98)->sort_0 = sort_9;
    (&kernelContext_98)->manifolds_0 = manifolds_9;
    (&kernelContext_98)->loads_0 = loads_9;
    (&kernelContext_98)->bonds_0 = bonds_9;
    (&kernelContext_98)->args_0 = args_9;
    (&kernelContext_98)->info_0 = info_9;
    (&kernelContext_98)->scan_0 = scan_9;
    (&kernelContext_98)->parts_0 = parts_9;
    (&kernelContext_98)->colliders_0 = colliders_9;
    (&kernelContext_98)->collider_info_0 = collider_info_9;
    (&kernelContext_98)->view_chunks_0 = view_chunks_9;
    (&kernelContext_98)->view_0 = view_9;
    (&kernelContext_98)->objects_0 = objects_9;
    (&kernelContext_98)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_98)->lines_0 = lines_9;
    (&kernelContext_98)->geometry_0 = geometry_9;
    (&kernelContext_98)->shots_0 = shots_9;
    (&kernelContext_98)->lp_0 = lp_9;
    (&kernelContext_98)->ext_0 = ext_10;
    (&kernelContext_98)->engine_contacts_0 = engine_contacts_9;
    (&kernelContext_98)->crush_0 = crush_9;
    (&kernelContext_98)->crush_rw_0 = crush_rw_9;
    (&kernelContext_98)->crush_tab_0 = crush_tab_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_98)->g_scan_0 = &g_scan_9;
    threadgroup array<float, int(64)> g_il_best_9;
    (&kernelContext_98)->g_il_best_0 = &g_il_best_9;
    uint c_8 = id_8.x;
    bool _S655 = live_0(c_8, &kernelContext_98);
    bool _S656;
    if(!_S655)
    {
        _S656 = true;
    }
    else
    {
        uint _S657 = body_of_0(c_8, &kernelContext_98);
        bool _S658 = dirty_0(_S657, &kernelContext_98);
        _S656 = !_S658;
    }
    if(_S656)
    {
        return;
    }
    uint _S659 = find_0(c_8, &kernelContext_98);
    uint _S660 = w_label_0(&kernelContext_98);
    *((&kernelContext_98)->work_0+(_S660 + c_8)) = _S659;
    uint _S661 = a_count_0(&kernelContext_98);
    uint _S662 = atomic_fetch_add_explicit((&kernelContext_98)->atom_0+(_S661 + _S659), 1U, memory_order_relaxed);
    if(((&kernelContext_98)->info_0[c_8] & 1U) != 0U)
    {
        uint _S663 = a_anchor_0(&kernelContext_98);
        uint _S664 = atomic_fetch_or_explicit((&kernelContext_98)->atom_0+(_S663 + _S659), 1U, memory_order_relaxed);
    }
    uint _S665 = 4U * c_8;
    float4 _S666 = float4(*((&kernelContext_98)->chunks_0+_S665)) ;
    float4 _S667 = float4(*((&kernelContext_98)->chunks_0+(_S665 + 1U))) ;
    float4 _S668 = float4(*((&kernelContext_98)->chunks_0+(_S665 + 2U))) ;
    float m_2 = _S666.w;
    float3 d_18 = _S666.xyz - (float4(*((&kernelContext_98)->chunks_0+4U * _S659)) ).xyz;
    float dd_0 = dot(d_18, d_18);
    uint _S669 = a_sums_0(&kernelContext_98);
    uint at_9 = _S669 + 20U * _S659;
    add64_0(at_9, m_2, (&kernelContext_98)->params_0->scale_m_0, &kernelContext_98);
    float _S670 = d_18.x;
    float _S671 = m_2 * _S670;
    add64_0(at_9 + 2U, _S671, (&kernelContext_98)->params_0->scale_s_0, &kernelContext_98);
    float _S672 = d_18.y;
    float _S673 = m_2 * _S672;
    add64_0(at_9 + 4U, _S673, (&kernelContext_98)->params_0->scale_s_0, &kernelContext_98);
    float _S674 = d_18.z;
    add64_0(at_9 + 6U, m_2 * _S674, (&kernelContext_98)->params_0->scale_s_0, &kernelContext_98);
    add64_0(at_9 + 8U, _S667.x + m_2 * (dd_0 - _S670 * _S670), (&kernelContext_98)->params_0->scale_j_0, &kernelContext_98);
    add64_0(at_9 + 10U, _S667.y + m_2 * (dd_0 - _S672 * _S672), (&kernelContext_98)->params_0->scale_j_0, &kernelContext_98);
    add64_0(at_9 + 12U, _S667.z + m_2 * (dd_0 - _S674 * _S674), (&kernelContext_98)->params_0->scale_j_0, &kernelContext_98);
    add64_0(at_9 + 14U, _S667.w - _S671 * _S672, (&kernelContext_98)->params_0->scale_j_0, &kernelContext_98);
    add64_0(at_9 + 16U, _S668.x - _S671 * _S674, (&kernelContext_98)->params_0->scale_j_0, &kernelContext_98);
    add64_0(at_9 + 18U, _S668.y - _S673 * _S674, (&kernelContext_98)->params_0->scale_j_0, &kernelContext_98);
    return;
}

bool is_root_0(uint c_9, KernelContext_0 thread* kernelContext_99)
{
    bool _S675 = live_0(c_9, kernelContext_99);
    bool _S676;
    if(_S675)
    {
        uint _S677 = body_of_0(c_9, kernelContext_99);
        bool _S678 = dirty_0(_S677, kernelContext_99);
        _S676 = _S678;
    }
    else
    {
        _S676 = false;
    }
    if(_S676)
    {
        uint _S679 = w_label_0(kernelContext_99);
        _S676 = (*(kernelContext_99->work_0+(_S679 + c_9))) == c_9;
    }
    else
    {
        _S676 = false;
    }
    return _S676;
}

[[kernel]] void fr_best_count(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], atomic<uint> device* atom_10 [[buffer(5)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], packed_float4 device* lp_10 [[buffer(24)]], packed_float4 device* ext_11 [[buffer(23)]], packed_float4 device* engine_contacts_10 [[buffer(25)]], CrushParams_0 constant* crush_10 [[buffer(28)]], packed_float4 device* crush_rw_10 [[buffer(27)]], packed_float4 device* crush_tab_10 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_100;
    (&kernelContext_100)->params_0 = params_10;
    (&kernelContext_100)->atom_0 = atom_10;
    (&kernelContext_100)->work_0 = work_10;
    (&kernelContext_100)->frames_0 = frames_10;
    (&kernelContext_100)->bodies_0 = bodies_10;
    (&kernelContext_100)->chunks_0 = chunks_10;
    (&kernelContext_100)->sort_0 = sort_10;
    (&kernelContext_100)->manifolds_0 = manifolds_10;
    (&kernelContext_100)->loads_0 = loads_10;
    (&kernelContext_100)->bonds_0 = bonds_10;
    (&kernelContext_100)->args_0 = args_10;
    (&kernelContext_100)->info_0 = info_10;
    (&kernelContext_100)->scan_0 = scan_10;
    (&kernelContext_100)->parts_0 = parts_10;
    (&kernelContext_100)->colliders_0 = colliders_10;
    (&kernelContext_100)->collider_info_0 = collider_info_10;
    (&kernelContext_100)->view_chunks_0 = view_chunks_10;
    (&kernelContext_100)->view_0 = view_10;
    (&kernelContext_100)->objects_0 = objects_10;
    (&kernelContext_100)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_100)->lines_0 = lines_10;
    (&kernelContext_100)->geometry_0 = geometry_10;
    (&kernelContext_100)->shots_0 = shots_10;
    (&kernelContext_100)->lp_0 = lp_10;
    (&kernelContext_100)->ext_0 = ext_11;
    (&kernelContext_100)->engine_contacts_0 = engine_contacts_10;
    (&kernelContext_100)->crush_0 = crush_10;
    (&kernelContext_100)->crush_rw_0 = crush_rw_10;
    (&kernelContext_100)->crush_tab_0 = crush_tab_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_100)->g_scan_0 = &g_scan_10;
    threadgroup array<float, int(64)> g_il_best_10;
    (&kernelContext_100)->g_il_best_0 = &g_il_best_10;
    uint c_10 = id_9.x;
    bool _S680 = is_root_0(c_10, &kernelContext_100);
    if(!_S680)
    {
        return;
    }
    uint _S681 = a_best_count_0(&kernelContext_100);
    uint _S682 = body_of_0(c_10, &kernelContext_100);
    atomic<uint> device* _S683 = (&kernelContext_100)->atom_0+(_S681 + _S682);
    uint _S684 = a_count_0(&kernelContext_100);
    uint _S685 = atomic_load_explicit((&kernelContext_100)->atom_0+(_S684 + c_10), memory_order_relaxed);
    uint _S686 = atomic_fetch_max_explicit(_S683, _S685, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_11 [[buffer(0)]], atomic<uint> device* atom_11 [[buffer(5)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], packed_float4 device* lp_11 [[buffer(24)]], packed_float4 device* ext_12 [[buffer(23)]], packed_float4 device* engine_contacts_11 [[buffer(25)]], CrushParams_0 constant* crush_11 [[buffer(28)]], packed_float4 device* crush_rw_11 [[buffer(27)]], packed_float4 device* crush_tab_11 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_101;
    (&kernelContext_101)->params_0 = params_11;
    (&kernelContext_101)->atom_0 = atom_11;
    (&kernelContext_101)->work_0 = work_11;
    (&kernelContext_101)->frames_0 = frames_11;
    (&kernelContext_101)->bodies_0 = bodies_11;
    (&kernelContext_101)->chunks_0 = chunks_11;
    (&kernelContext_101)->sort_0 = sort_11;
    (&kernelContext_101)->manifolds_0 = manifolds_11;
    (&kernelContext_101)->loads_0 = loads_11;
    (&kernelContext_101)->bonds_0 = bonds_11;
    (&kernelContext_101)->args_0 = args_11;
    (&kernelContext_101)->info_0 = info_11;
    (&kernelContext_101)->scan_0 = scan_11;
    (&kernelContext_101)->parts_0 = parts_11;
    (&kernelContext_101)->colliders_0 = colliders_11;
    (&kernelContext_101)->collider_info_0 = collider_info_11;
    (&kernelContext_101)->view_chunks_0 = view_chunks_11;
    (&kernelContext_101)->view_0 = view_11;
    (&kernelContext_101)->objects_0 = objects_11;
    (&kernelContext_101)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_101)->lines_0 = lines_11;
    (&kernelContext_101)->geometry_0 = geometry_11;
    (&kernelContext_101)->shots_0 = shots_11;
    (&kernelContext_101)->lp_0 = lp_11;
    (&kernelContext_101)->ext_0 = ext_12;
    (&kernelContext_101)->engine_contacts_0 = engine_contacts_11;
    (&kernelContext_101)->crush_0 = crush_11;
    (&kernelContext_101)->crush_rw_0 = crush_rw_11;
    (&kernelContext_101)->crush_tab_0 = crush_tab_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_101)->g_scan_0 = &g_scan_11;
    threadgroup array<float, int(64)> g_il_best_11;
    (&kernelContext_101)->g_il_best_0 = &g_il_best_11;
    uint c_11 = id_10.x;
    bool _S687 = is_root_0(c_11, &kernelContext_101);
    if(!_S687)
    {
        return;
    }
    uint _S688 = body_of_0(c_11, &kernelContext_101);
    uint _S689 = a_count_0(&kernelContext_101);
    uint _S690 = atomic_load_explicit((&kernelContext_101)->atom_0+(_S689 + c_11), memory_order_relaxed);
    uint _S691 = a_best_count_0(&kernelContext_101);
    uint _S692 = atomic_load_explicit((&kernelContext_101)->atom_0+(_S691 + _S688), memory_order_relaxed);
    if(_S690 == _S692)
    {
        uint _S693 = a_best_root_0(&kernelContext_101);
        uint _S694 = atomic_fetch_min_explicit((&kernelContext_101)->atom_0+(_S693 + _S688), c_11, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_12 [[buffer(0)]], atomic<uint> device* atom_12 [[buffer(5)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], packed_float4 device* lp_12 [[buffer(24)]], packed_float4 device* ext_13 [[buffer(23)]], packed_float4 device* engine_contacts_12 [[buffer(25)]], CrushParams_0 constant* crush_12 [[buffer(28)]], packed_float4 device* crush_rw_12 [[buffer(27)]], packed_float4 device* crush_tab_12 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_102;
    (&kernelContext_102)->params_0 = params_12;
    (&kernelContext_102)->atom_0 = atom_12;
    (&kernelContext_102)->work_0 = work_12;
    (&kernelContext_102)->frames_0 = frames_12;
    (&kernelContext_102)->bodies_0 = bodies_12;
    (&kernelContext_102)->chunks_0 = chunks_12;
    (&kernelContext_102)->sort_0 = sort_12;
    (&kernelContext_102)->manifolds_0 = manifolds_12;
    (&kernelContext_102)->loads_0 = loads_12;
    (&kernelContext_102)->bonds_0 = bonds_12;
    (&kernelContext_102)->args_0 = args_12;
    (&kernelContext_102)->info_0 = info_12;
    (&kernelContext_102)->scan_0 = scan_12;
    (&kernelContext_102)->parts_0 = parts_12;
    (&kernelContext_102)->colliders_0 = colliders_12;
    (&kernelContext_102)->collider_info_0 = collider_info_12;
    (&kernelContext_102)->view_chunks_0 = view_chunks_12;
    (&kernelContext_102)->view_0 = view_12;
    (&kernelContext_102)->objects_0 = objects_12;
    (&kernelContext_102)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_102)->lines_0 = lines_12;
    (&kernelContext_102)->geometry_0 = geometry_12;
    (&kernelContext_102)->shots_0 = shots_12;
    (&kernelContext_102)->lp_0 = lp_12;
    (&kernelContext_102)->ext_0 = ext_13;
    (&kernelContext_102)->engine_contacts_0 = engine_contacts_12;
    (&kernelContext_102)->crush_0 = crush_12;
    (&kernelContext_102)->crush_rw_0 = crush_rw_12;
    (&kernelContext_102)->crush_tab_0 = crush_tab_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_102)->g_scan_0 = &g_scan_12;
    threadgroup array<float, int(64)> g_il_best_12;
    (&kernelContext_102)->g_il_best_0 = &g_il_best_12;
    uint c_12 = id_11.x;
    if(c_12 >= (params_12->chunk_used_0))
    {
        return;
    }
    uint _S695 = w_new_0(&kernelContext_102);
    uint device* _S696 = (&kernelContext_102)->work_0+(_S695 + c_12);
    bool _S697 = is_root_0(c_12, &kernelContext_102);
    bool _S698;
    if(_S697)
    {
        uint _S699 = a_best_root_0(&kernelContext_102);
        uint _S700 = body_of_0(c_12, &kernelContext_102);
        uint _S701 = atomic_load_explicit((&kernelContext_102)->atom_0+(_S699 + _S700), memory_order_relaxed);
        _S698 = _S701 != c_12;
    }
    else
    {
        _S698 = false;
    }
    uint _S702;
    if(_S698)
    {
        _S702 = 1U;
    }
    else
    {
        _S702 = 0U;
    }
    *_S696 = _S702;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_13 [[buffer(0)]], atomic<uint> device* atom_13 [[buffer(5)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], packed_float4 device* lp_13 [[buffer(24)]], packed_float4 device* ext_14 [[buffer(23)]], packed_float4 device* engine_contacts_13 [[buffer(25)]], CrushParams_0 constant* crush_13 [[buffer(28)]], packed_float4 device* crush_rw_13 [[buffer(27)]], packed_float4 device* crush_tab_13 [[buffer(26)]])
{
    uint step_4;
    uint add_0;
    thread KernelContext_0 kernelContext_103;
    (&kernelContext_103)->params_0 = params_13;
    (&kernelContext_103)->atom_0 = atom_13;
    (&kernelContext_103)->work_0 = work_13;
    (&kernelContext_103)->frames_0 = frames_13;
    (&kernelContext_103)->bodies_0 = bodies_13;
    (&kernelContext_103)->chunks_0 = chunks_13;
    (&kernelContext_103)->sort_0 = sort_13;
    (&kernelContext_103)->manifolds_0 = manifolds_13;
    (&kernelContext_103)->loads_0 = loads_13;
    (&kernelContext_103)->bonds_0 = bonds_13;
    (&kernelContext_103)->args_0 = args_13;
    (&kernelContext_103)->info_0 = info_13;
    (&kernelContext_103)->scan_0 = scan_13;
    (&kernelContext_103)->parts_0 = parts_13;
    (&kernelContext_103)->colliders_0 = colliders_13;
    (&kernelContext_103)->collider_info_0 = collider_info_13;
    (&kernelContext_103)->view_chunks_0 = view_chunks_13;
    (&kernelContext_103)->view_0 = view_13;
    (&kernelContext_103)->objects_0 = objects_13;
    (&kernelContext_103)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_103)->lines_0 = lines_13;
    (&kernelContext_103)->geometry_0 = geometry_13;
    (&kernelContext_103)->shots_0 = shots_13;
    (&kernelContext_103)->lp_0 = lp_13;
    (&kernelContext_103)->ext_0 = ext_14;
    (&kernelContext_103)->engine_contacts_0 = engine_contacts_13;
    (&kernelContext_103)->crush_0 = crush_13;
    (&kernelContext_103)->crush_rw_0 = crush_rw_13;
    (&kernelContext_103)->crush_tab_0 = crush_tab_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_103)->g_scan_0 = &g_scan_13;
    threadgroup array<float, int(64)> g_il_best_13;
    (&kernelContext_103)->g_il_best_0 = &g_il_best_13;
    uint t_7 = thread_2.x;
    uint _S703 = group_0.x;
    uint _S704 = 1024U * _S703 + 4U * t_7;
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
        if((_S704 + j_5) < ((&kernelContext_103)->scan_0->n_0))
        {
            step_4 = *((&kernelContext_103)->work_0+((&kernelContext_103)->scan_0->in_base_0 + _S704 + j_5));
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
    (*(&kernelContext_103)->g_scan_0)[t_7] = sum_2;
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
            add_0 = (*(&kernelContext_103)->g_scan_0)[t_7 - step_4];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_103)->g_scan_0)[t_7] = (*(&kernelContext_103)->g_scan_0)[t_7] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_4 = step_4 << 1U;
    }
    uint _S705 = (*(&kernelContext_103)->g_scan_0)[t_7] - sum_2;
    j_5 = 0U;
    add_0 = _S705;
    for(;;)
    {
        if(j_5 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S704 + j_5) < ((&kernelContext_103)->scan_0->n_0))
        {
            *((&kernelContext_103)->work_0+((&kernelContext_103)->scan_0->out_base_0 + _S704 + j_5)) = add_0;
        }
        uint run_1 = add_0 + v_10[j_5];
        j_5 = j_5 + 1U;
        add_0 = run_1;
    }
    if(t_7 == 255U)
    {
        *((&kernelContext_103)->work_0+((&kernelContext_103)->scan_0->block_base_0 + _S703)) = (*(&kernelContext_103)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_3 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_14 [[buffer(0)]], atomic<uint> device* atom_14 [[buffer(5)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], packed_float4 device* lp_14 [[buffer(24)]], packed_float4 device* ext_15 [[buffer(23)]], packed_float4 device* engine_contacts_14 [[buffer(25)]], CrushParams_0 constant* crush_14 [[buffer(28)]], packed_float4 device* crush_rw_14 [[buffer(27)]], packed_float4 device* crush_tab_14 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_104;
    (&kernelContext_104)->params_0 = params_14;
    (&kernelContext_104)->atom_0 = atom_14;
    (&kernelContext_104)->work_0 = work_14;
    (&kernelContext_104)->frames_0 = frames_14;
    (&kernelContext_104)->bodies_0 = bodies_14;
    (&kernelContext_104)->chunks_0 = chunks_14;
    (&kernelContext_104)->sort_0 = sort_14;
    (&kernelContext_104)->manifolds_0 = manifolds_14;
    (&kernelContext_104)->loads_0 = loads_14;
    (&kernelContext_104)->bonds_0 = bonds_14;
    (&kernelContext_104)->args_0 = args_14;
    (&kernelContext_104)->info_0 = info_14;
    (&kernelContext_104)->scan_0 = scan_14;
    (&kernelContext_104)->parts_0 = parts_14;
    (&kernelContext_104)->colliders_0 = colliders_14;
    (&kernelContext_104)->collider_info_0 = collider_info_14;
    (&kernelContext_104)->view_chunks_0 = view_chunks_14;
    (&kernelContext_104)->view_0 = view_14;
    (&kernelContext_104)->objects_0 = objects_14;
    (&kernelContext_104)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_104)->lines_0 = lines_14;
    (&kernelContext_104)->geometry_0 = geometry_14;
    (&kernelContext_104)->shots_0 = shots_14;
    (&kernelContext_104)->lp_0 = lp_14;
    (&kernelContext_104)->ext_0 = ext_15;
    (&kernelContext_104)->engine_contacts_0 = engine_contacts_14;
    (&kernelContext_104)->crush_0 = crush_14;
    (&kernelContext_104)->crush_rw_0 = crush_rw_14;
    (&kernelContext_104)->crush_tab_0 = crush_tab_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_104)->g_scan_0 = &g_scan_14;
    threadgroup array<float, int(64)> g_il_best_14;
    (&kernelContext_104)->g_il_best_0 = &g_il_best_14;
    uint t_8 = thread_3.x;
    uint blocks_0 = (scan_14->n_0 + 1023U) / 1024U;
    bool _S706 = t_8 < blocks_0;
    uint v_11;
    if(_S706)
    {
        v_11 = *((&kernelContext_104)->work_0+((&kernelContext_104)->scan_0->block_base_0 + t_8));
    }
    else
    {
        v_11 = 0U;
    }
    (*(&kernelContext_104)->g_scan_0)[t_8] = v_11;
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
            add_1 = (*(&kernelContext_104)->g_scan_0)[t_8 - step_5];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_104)->g_scan_0)[t_8] = (*(&kernelContext_104)->g_scan_0)[t_8] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_5 = step_5 << 1U;
    }
    if(_S706)
    {
        *((&kernelContext_104)->work_0+((&kernelContext_104)->scan_0->block_base_0 + t_8)) = (*(&kernelContext_104)->g_scan_0)[t_8] - v_11;
    }
    if(t_8 == 1023U)
    {
        *((&kernelContext_104)->work_0+((&kernelContext_104)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_104)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], atomic<uint> device* atom_15 [[buffer(5)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], packed_float4 device* lp_15 [[buffer(24)]], packed_float4 device* ext_16 [[buffer(23)]], packed_float4 device* engine_contacts_15 [[buffer(25)]], CrushParams_0 constant* crush_15 [[buffer(28)]], packed_float4 device* crush_rw_15 [[buffer(27)]], packed_float4 device* crush_tab_15 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_105;
    (&kernelContext_105)->params_0 = params_15;
    (&kernelContext_105)->atom_0 = atom_15;
    (&kernelContext_105)->work_0 = work_15;
    (&kernelContext_105)->frames_0 = frames_15;
    (&kernelContext_105)->bodies_0 = bodies_15;
    (&kernelContext_105)->chunks_0 = chunks_15;
    (&kernelContext_105)->sort_0 = sort_15;
    (&kernelContext_105)->manifolds_0 = manifolds_15;
    (&kernelContext_105)->loads_0 = loads_15;
    (&kernelContext_105)->bonds_0 = bonds_15;
    (&kernelContext_105)->args_0 = args_15;
    (&kernelContext_105)->info_0 = info_15;
    (&kernelContext_105)->scan_0 = scan_15;
    (&kernelContext_105)->parts_0 = parts_15;
    (&kernelContext_105)->colliders_0 = colliders_15;
    (&kernelContext_105)->collider_info_0 = collider_info_15;
    (&kernelContext_105)->view_chunks_0 = view_chunks_15;
    (&kernelContext_105)->view_0 = view_15;
    (&kernelContext_105)->objects_0 = objects_15;
    (&kernelContext_105)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_105)->lines_0 = lines_15;
    (&kernelContext_105)->geometry_0 = geometry_15;
    (&kernelContext_105)->shots_0 = shots_15;
    (&kernelContext_105)->lp_0 = lp_15;
    (&kernelContext_105)->ext_0 = ext_16;
    (&kernelContext_105)->engine_contacts_0 = engine_contacts_15;
    (&kernelContext_105)->crush_0 = crush_15;
    (&kernelContext_105)->crush_rw_0 = crush_rw_15;
    (&kernelContext_105)->crush_tab_0 = crush_tab_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_105)->g_scan_0 = &g_scan_15;
    threadgroup array<float, int(64)> g_il_best_15;
    (&kernelContext_105)->g_il_best_0 = &g_il_best_15;
    uint i_10 = id_12.x;
    if(i_10 >= (scan_15->n_0))
    {
        return;
    }
    uint device* _S707 = (&kernelContext_105)->work_0+((&kernelContext_105)->scan_0->out_base_0 + i_10);
    *_S707 = *_S707 + *((&kernelContext_105)->work_0+((&kernelContext_105)->scan_0->block_base_0 + i_10 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], atomic<uint> device* atom_16 [[buffer(5)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], packed_float4 device* lp_16 [[buffer(24)]], packed_float4 device* ext_17 [[buffer(23)]], packed_float4 device* engine_contacts_16 [[buffer(25)]], CrushParams_0 constant* crush_16 [[buffer(28)]], packed_float4 device* crush_rw_16 [[buffer(27)]], packed_float4 device* crush_tab_16 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_106;
    (&kernelContext_106)->params_0 = params_16;
    (&kernelContext_106)->atom_0 = atom_16;
    (&kernelContext_106)->work_0 = work_16;
    (&kernelContext_106)->frames_0 = frames_16;
    (&kernelContext_106)->bodies_0 = bodies_16;
    (&kernelContext_106)->chunks_0 = chunks_16;
    (&kernelContext_106)->sort_0 = sort_16;
    (&kernelContext_106)->manifolds_0 = manifolds_16;
    (&kernelContext_106)->loads_0 = loads_16;
    (&kernelContext_106)->bonds_0 = bonds_16;
    (&kernelContext_106)->args_0 = args_16;
    (&kernelContext_106)->info_0 = info_16;
    (&kernelContext_106)->scan_0 = scan_16;
    (&kernelContext_106)->parts_0 = parts_16;
    (&kernelContext_106)->colliders_0 = colliders_16;
    (&kernelContext_106)->collider_info_0 = collider_info_16;
    (&kernelContext_106)->view_chunks_0 = view_chunks_16;
    (&kernelContext_106)->view_0 = view_16;
    (&kernelContext_106)->objects_0 = objects_16;
    (&kernelContext_106)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_106)->lines_0 = lines_16;
    (&kernelContext_106)->geometry_0 = geometry_16;
    (&kernelContext_106)->shots_0 = shots_16;
    (&kernelContext_106)->lp_0 = lp_16;
    (&kernelContext_106)->ext_0 = ext_17;
    (&kernelContext_106)->engine_contacts_0 = engine_contacts_16;
    (&kernelContext_106)->crush_0 = crush_16;
    (&kernelContext_106)->crush_rw_0 = crush_rw_16;
    (&kernelContext_106)->crush_tab_0 = crush_tab_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_106)->g_scan_0 = &g_scan_16;
    threadgroup array<float, int(64)> g_il_best_16;
    (&kernelContext_106)->g_il_best_0 = &g_il_best_16;
    uint s_8 = id_13.x;
    bool _S708;
    if(s_8 >= (params_16->body_count_0))
    {
        _S708 = true;
    }
    else
    {
        uint _S709 = w_free_0(&kernelContext_106);
        _S708 = (*((&kernelContext_106)->work_0+(_S709 + s_8))) == 0U;
    }
    if(_S708)
    {
        return;
    }
    uint _S710 = w_free_list_0(&kernelContext_106);
    uint _S711 = w_free_rank_0(&kernelContext_106);
    *((&kernelContext_106)->work_0+(_S710 + *((&kernelContext_106)->work_0+(_S711 + s_8)))) = s_8;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_107)
{
    uint _S712 = w_free_list_0(kernelContext_107);
    return _S712 + kernelContext_107->params_0->body_count_0;
}

uint scan_new_blocks_1(KernelContext_0 thread* kernelContext_108)
{
    uint _S713 = w_free_list_0(kernelContext_108);
    return _S713 + kernelContext_108->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_109)
{
    uint _S714 = scan_new_blocks_1(kernelContext_109);
    return _S714 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_110)
{
    uint blocks_1 = (kernelContext_110->params_0->body_count_0 + 1023U) / 1024U;
    uint _S715 = scan_free_blocks_0(kernelContext_110);
    return *(kernelContext_110->work_0+(_S715 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], atomic<uint> device* atom_17 [[buffer(5)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], packed_float4 device* lp_17 [[buffer(24)]], packed_float4 device* ext_18 [[buffer(23)]], packed_float4 device* engine_contacts_17 [[buffer(25)]], CrushParams_0 constant* crush_17 [[buffer(28)]], packed_float4 device* crush_rw_17 [[buffer(27)]], packed_float4 device* crush_tab_17 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_111;
    (&kernelContext_111)->params_0 = params_17;
    (&kernelContext_111)->atom_0 = atom_17;
    (&kernelContext_111)->work_0 = work_17;
    (&kernelContext_111)->frames_0 = frames_17;
    (&kernelContext_111)->bodies_0 = bodies_17;
    (&kernelContext_111)->chunks_0 = chunks_17;
    (&kernelContext_111)->sort_0 = sort_17;
    (&kernelContext_111)->manifolds_0 = manifolds_17;
    (&kernelContext_111)->loads_0 = loads_17;
    (&kernelContext_111)->bonds_0 = bonds_17;
    (&kernelContext_111)->args_0 = args_17;
    (&kernelContext_111)->info_0 = info_17;
    (&kernelContext_111)->scan_0 = scan_17;
    (&kernelContext_111)->parts_0 = parts_17;
    (&kernelContext_111)->colliders_0 = colliders_17;
    (&kernelContext_111)->collider_info_0 = collider_info_17;
    (&kernelContext_111)->view_chunks_0 = view_chunks_17;
    (&kernelContext_111)->view_0 = view_17;
    (&kernelContext_111)->objects_0 = objects_17;
    (&kernelContext_111)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_111)->lines_0 = lines_17;
    (&kernelContext_111)->geometry_0 = geometry_17;
    (&kernelContext_111)->shots_0 = shots_17;
    (&kernelContext_111)->lp_0 = lp_17;
    (&kernelContext_111)->ext_0 = ext_18;
    (&kernelContext_111)->engine_contacts_0 = engine_contacts_17;
    (&kernelContext_111)->crush_0 = crush_17;
    (&kernelContext_111)->crush_rw_0 = crush_rw_17;
    (&kernelContext_111)->crush_tab_0 = crush_tab_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_111)->g_scan_0 = &g_scan_17;
    threadgroup array<float, int(64)> g_il_best_17;
    (&kernelContext_111)->g_il_best_0 = &g_il_best_17;
    uint c_13 = id_14.x;
    bool _S716 = is_root_0(c_13, &kernelContext_111);
    if(!_S716)
    {
        return;
    }
    uint _S717 = body_of_0(c_13, &kernelContext_111);
    uint _S718 = a_best_root_0(&kernelContext_111);
    uint _S719 = atomic_load_explicit((&kernelContext_111)->atom_0+(_S718 + _S717), memory_order_relaxed);
    uint slot_1;
    if(_S719 != c_13)
    {
        uint _S720 = w_rank_0(&kernelContext_111);
        uint device* _S721 = (&kernelContext_111)->work_0+(_S720 + c_13);
        uint k_23 = *_S721;
        uint _S722 = free_total_0(&kernelContext_111);
        if((*_S721) < _S722)
        {
            uint _S723 = w_free_list_0(&kernelContext_111);
            uint device* _S724 = (&kernelContext_111)->work_0+(_S723 + k_23);
            uint slot_2 = *_S724;
            uint _S725 = w_free_0(&kernelContext_111);
            *((&kernelContext_111)->work_0+(_S725 + *_S724)) = 0U;
            uint _S726 = a_counters_1(&kernelContext_111);
            uint _S727 = atomic_fetch_add_explicit((&kernelContext_111)->atom_0+(_S726 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S728 = a_counters_1(&kernelContext_111);
            uint _S729 = atomic_fetch_or_explicit((&kernelContext_111)->atom_0+(_S728 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S717;
        }
    }
    else
    {
        slot_1 = _S717;
    }
    uint _S730 = w_slot_0(&kernelContext_111);
    *((&kernelContext_111)->work_0+(_S730 + c_13)) = slot_1;
    return;
}

float read64_1(uint at_10, float scale_2, KernelContext_0 thread* kernelContext_112)
{
    uint _S731 = atomic_load_explicit(kernelContext_112->atom_0+(at_10 + 1U), memory_order_relaxed);
    int hi_3 = (as_type<int>((_S731)));
    uint lo_3 = atomic_load_explicit(kernelContext_112->atom_0+at_10, memory_order_relaxed);
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
    array<float, int(3)> _S732 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S733 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S734 = { { r2_2.x, r2_2.y, r2_2.z } };
    thread array<array<float, int(3)>, int(3)> a_7;
    a_7[int(0)] = _S732;
    a_7[int(1)] = _S733;
    a_7[int(2)] = _S734;
    array<float, int(3)> _S735 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S736 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S737 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_12;
    v_12[int(0)] = _S735;
    v_12[int(1)] = _S736;
    v_12[int(2)] = _S737;
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
            bool _S738;
            if(k_24 != p_12)
            {
                _S738 = k_24 != q_14;
            }
            else
            {
                _S738 = false;
            }
            if(_S738)
            {
                float kp_0 = a_7[k_24][p_12];
                float kq_0 = a_7[k_24][q_14];
                float _S739 = c_14 * a_7[k_24][p_12] - s_9 * a_7[k_24][q_14];
                a_7[k_24][p_12] = _S739;
                a_7[p_12][k_24] = _S739;
                float _S740 = s_9 * kp_0 + c_14 * kq_0;
                a_7[k_24][q_14] = _S740;
                a_7[q_14][k_24] = _S740;
            }
            k_24 = k_24 + 1U;
        }
        float _S741 = c_14 * c_14;
        float _S742 = 2.0f * s_9 * c_14 * apq_0;
        float _S743 = s_9 * s_9;
        a_7[p_12][p_12] = _S741 * app_0 - _S742 + _S743 * aqq_0;
        a_7[q_14][q_14] = _S743 * app_0 + _S742 + _S741 * aqq_0;
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
    thread Quat_0 _S744 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S745 = normalized_0(&_S744);
    *rot_0 = _S745;
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
    thread Quat_0 _S746 = axes_1;
    Quat_0 _S747 = quat_mul_0(&_S746, q_15);
    thread Quat_0 _S748 = _S747;
    Quat_0 _S749 = normalized_0(&_S748);
    return _S749;
}

uint stage_0(uint c_15, KernelContext_0 thread* kernelContext_113)
{
    return 2U * kernelContext_113->params_0->body_count_0 + 9U * c_15;
}

float4 quat_vec_0(const Quat_0 thread* q_16)
{
    return float4(q_16->x_0, q_16->y_0, q_16->z_0, q_16->w_0);
}

[[kernel]] void fr_piece(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], atomic<uint> device* atom_18 [[buffer(5)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], packed_float4 device* lp_18 [[buffer(24)]], packed_float4 device* ext_19 [[buffer(23)]], packed_float4 device* engine_contacts_18 [[buffer(25)]], CrushParams_0 constant* crush_18 [[buffer(28)]], packed_float4 device* crush_rw_18 [[buffer(27)]], packed_float4 device* crush_tab_18 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_114;
    (&kernelContext_114)->params_0 = params_18;
    (&kernelContext_114)->atom_0 = atom_18;
    (&kernelContext_114)->work_0 = work_18;
    (&kernelContext_114)->frames_0 = frames_18;
    (&kernelContext_114)->bodies_0 = bodies_18;
    (&kernelContext_114)->chunks_0 = chunks_18;
    (&kernelContext_114)->sort_0 = sort_18;
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
    uint c_16 = id_15.x;
    bool _S750 = is_root_0(c_16, &kernelContext_114);
    if(!_S750)
    {
        return;
    }
    uint _S751 = body_of_0(c_16, &kernelContext_114);
    uint pb_0 = 7U * _S751;
    uint _S752 = a_sums_0(&kernelContext_114);
    uint at_11 = _S752 + 20U * c_16;
    float _S753 = read64_0(at_11, (&kernelContext_114)->params_0->scale_m_0, &kernelContext_114);
    float _S754 = read64_0(at_11 + 2U, (&kernelContext_114)->params_0->scale_s_0, &kernelContext_114);
    float _S755 = read64_0(at_11 + 4U, (&kernelContext_114)->params_0->scale_s_0, &kernelContext_114);
    float _S756 = read64_0(at_11 + 6U, (&kernelContext_114)->params_0->scale_s_0, &kernelContext_114);
    float3 d_19 = float3(_S754, _S755, _S756) / float3(max(_S753, 1.00000000317107685e-30f)) ;
    float _S757 = read64_0(at_11 + 8U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float _S758 = read64_0(at_11 + 10U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float _S759 = read64_0(at_11 + 12U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float _S760 = read64_0(at_11 + 14U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float _S761 = read64_0(at_11 + 16U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float _S762 = read64_0(at_11 + 18U, (&kernelContext_114)->params_0->scale_j_0, &kernelContext_114);
    float dd_1 = dot(d_19, d_19);
    float _S763 = d_19.x;
    float _S764 = _S753 * _S763;
    float _S765 = d_19.y;
    float _S766 = _S760 + _S764 * _S765;
    float _S767 = d_19.z;
    float _S768 = _S761 + _S764 * _S767;
    float _S769 = _S762 + _S753 * _S765 * _S767;
    float3 com_1 = (float4(*((&kernelContext_114)->chunks_0+4U * c_16)) ).xyz + d_19;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S757 - _S753 * (dd_1 - _S763 * _S763), _S766, _S768), float3(_S766, _S758 - _S753 * (dd_1 - _S765 * _S765), _S769), float3(_S768, _S769, _S759 - _S753 * (dd_1 - _S767 * _S767)), &principal_0, &moments_1);
    uint _S770 = a_anchor_0(&kernelContext_114);
    uint _S771 = atomic_load_explicit((&kernelContext_114)->atom_0+(_S770 + c_16), memory_order_relaxed);
    bool anchored_0 = _S771 != 0U;
    thread Quat_0 q_17;
    thread float3 origin_4;
    structure_pose_0(_S751, pb_0, &q_17, &origin_4, &kernelContext_114);
    float3 p_parent_0 = (float4(*((&kernelContext_114)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_114)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_114)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S772 = q_17;
    float3 _S773 = rotate_0(&_S772, com_1);
    float3 pos_0 = to_rubble_0(origin_4 + _S773);
    thread Quat_0 _S774 = q_17;
    thread Quat_0 _S775 = principal_0;
    Quat_0 _S776 = quat_mul_0(&_S774, &_S775);
    thread Quat_0 _S777 = _S776;
    Quat_0 _S778 = normalized_0(&_S777);
    thread Quat_0 _S779 = _S778;
    Quat_0 _S780 = rot_to_rubble_0(&_S779);
    bool _S781;
    if(anchored_0)
    {
        _S781 = true;
    }
    else
    {
        _S781 = !(_S753 > 0.0f);
    }
    float inv_mass_3;
    if(_S781)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / _S753;
    }
    float3 _S782 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S783;
        if((moments_1.x) > 0.0f)
        {
            _S783 = 1.0f / moments_1.x;
        }
        else
        {
            _S783 = 0.0f;
        }
        float _S784;
        if((moments_1.y) > 0.0f)
        {
            _S784 = 1.0f / moments_1.y;
        }
        else
        {
            _S784 = 0.0f;
        }
        float _S785;
        if((moments_1.z) > 0.0f)
        {
            _S785 = 1.0f / moments_1.z;
        }
        else
        {
            _S785 = 0.0f;
        }
        inv_inertia_0 = float3(_S783, _S784, _S785);
    }
    else
    {
        inv_inertia_0 = _S782;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S782;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S782;
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
    uint _S786 = stage_0(c_16, &kernelContext_114);
    *((&kernelContext_114)->frames_0+_S786) = packed_float4(float4(pos_0, inv_mass_3)) ;
    packed_float4 device* _S787 = (&kernelContext_114)->frames_0+(_S786 + 1U);
    thread Quat_0 _S788 = _S780;
    float4 _S789 = quat_vec_0(&_S788);
    *_S787 = packed_float4(_S789) ;
    *((&kernelContext_114)->frames_0+(_S786 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_114)->frames_0+(_S786 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_114)->frames_0+(_S786 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_114)->params_0->step_0))))) ;
    *((&kernelContext_114)->frames_0+(_S786 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_114)->frames_0+(_S786 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_114)->frames_0+(_S786 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_114)->frames_0+2U * _S751)) ).w)) ;
    packed_float4 device* _S790 = (&kernelContext_114)->frames_0+(_S786 + 8U);
    thread Quat_0 _S791 = principal_0;
    float4 _S792 = quat_vec_0(&_S791);
    *_S790 = packed_float4(_S792) ;
    return;
}

uint f_events_0(KernelContext_0 thread* kernelContext_115)
{
    return 2U * kernelContext_115->params_0->body_count_0 + 9U * kernelContext_115->params_0->chunk_count_0 + 3U * kernelContext_115->params_0->body_count_0;
}

[[kernel]] void fr_commit(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], atomic<uint> device* atom_19 [[buffer(5)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], packed_float4 device* lp_19 [[buffer(24)]], packed_float4 device* ext_20 [[buffer(23)]], packed_float4 device* engine_contacts_19 [[buffer(25)]], CrushParams_0 constant* crush_19 [[buffer(28)]], packed_float4 device* crush_rw_19 [[buffer(27)]], packed_float4 device* crush_tab_19 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_116;
    (&kernelContext_116)->params_0 = params_19;
    (&kernelContext_116)->atom_0 = atom_19;
    (&kernelContext_116)->work_0 = work_19;
    (&kernelContext_116)->frames_0 = frames_19;
    (&kernelContext_116)->bodies_0 = bodies_19;
    (&kernelContext_116)->chunks_0 = chunks_19;
    (&kernelContext_116)->sort_0 = sort_19;
    (&kernelContext_116)->manifolds_0 = manifolds_19;
    (&kernelContext_116)->loads_0 = loads_19;
    (&kernelContext_116)->bonds_0 = bonds_19;
    (&kernelContext_116)->args_0 = args_19;
    (&kernelContext_116)->info_0 = info_19;
    (&kernelContext_116)->scan_0 = scan_19;
    (&kernelContext_116)->parts_0 = parts_19;
    (&kernelContext_116)->colliders_0 = colliders_19;
    (&kernelContext_116)->collider_info_0 = collider_info_19;
    (&kernelContext_116)->view_chunks_0 = view_chunks_19;
    (&kernelContext_116)->view_0 = view_19;
    (&kernelContext_116)->objects_0 = objects_19;
    (&kernelContext_116)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_116)->lines_0 = lines_19;
    (&kernelContext_116)->geometry_0 = geometry_19;
    (&kernelContext_116)->shots_0 = shots_19;
    (&kernelContext_116)->lp_0 = lp_19;
    (&kernelContext_116)->ext_0 = ext_20;
    (&kernelContext_116)->engine_contacts_0 = engine_contacts_19;
    (&kernelContext_116)->crush_0 = crush_19;
    (&kernelContext_116)->crush_rw_0 = crush_rw_19;
    (&kernelContext_116)->crush_tab_0 = crush_tab_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_116)->g_scan_0 = &g_scan_19;
    threadgroup array<float, int(64)> g_il_best_19;
    (&kernelContext_116)->g_il_best_0 = &g_il_best_19;
    uint c_17 = id_16.x;
    bool _S793 = is_root_0(c_17, &kernelContext_116);
    if(!_S793)
    {
        return;
    }
    uint _S794 = w_slot_0(&kernelContext_116);
    uint slot_3 = *((&kernelContext_116)->work_0+(_S794 + c_17));
    uint _S795 = stage_0(c_17, &kernelContext_116);
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
            *((&kernelContext_116)->bodies_0+(7U * (slot_3 + region_5 * (&kernelContext_116)->params_0->max_bodies_0) + w_4)) = packed_float4(float4(*((&kernelContext_116)->frames_0+(_S795 + w_4))) ) ;
            w_4 = w_4 + 1U;
        }
        region_5 = region_5 + 1U;
    }
    uint _S796 = 2U * slot_3;
    uint _S797 = _S795 + 7U;
    *((&kernelContext_116)->frames_0+_S796) = packed_float4(float4(*((&kernelContext_116)->frames_0+_S797)) ) ;
    uint _S798 = _S795 + 8U;
    *((&kernelContext_116)->frames_0+(_S796 + 1U)) = packed_float4(float4(*((&kernelContext_116)->frames_0+_S798)) ) ;
    if(((&kernelContext_116)->params_0->ev_frames_cap_0) != 0U)
    {
        uint _S799 = a_counters_1(&kernelContext_116);
        uint i_11 = atomic_fetch_add_explicit((&kernelContext_116)->atom_0+(_S799 + 22U), 1U, memory_order_relaxed);
        if(i_11 < ((&kernelContext_116)->params_0->ev_frames_cap_0))
        {
            uint _S800 = f_events_0(&kernelContext_116);
            uint at_12 = _S800 + 3U * i_11;
            *((&kernelContext_116)->frames_0+at_12) = packed_float4(float4((as_type<float>((slot_3))), 0.0f, 0.0f, 0.0f)) ;
            *((&kernelContext_116)->frames_0+(at_12 + 1U)) = packed_float4(float4(*((&kernelContext_116)->frames_0+_S797)) ) ;
            *((&kernelContext_116)->frames_0+(at_12 + 2U)) = packed_float4(float4(*((&kernelContext_116)->frames_0+_S798)) ) ;
        }
        else
        {
            uint _S801 = atomic_fetch_or_explicit((&kernelContext_116)->atom_0+(_S799 + 3U), 128U, memory_order_relaxed);
        }
    }
    return;
}

uint a_moved_0(KernelContext_0 thread* kernelContext_117)
{
    uint _S802 = a_events_0(kernelContext_117);
    return _S802 + kernelContext_117->params_0->ev_broken_cap_0;
}

[[kernel]] void fr_reparent(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], atomic<uint> device* atom_20 [[buffer(5)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], packed_float4 device* lp_20 [[buffer(24)]], packed_float4 device* ext_21 [[buffer(23)]], packed_float4 device* engine_contacts_20 [[buffer(25)]], CrushParams_0 constant* crush_20 [[buffer(28)]], packed_float4 device* crush_rw_20 [[buffer(27)]], packed_float4 device* crush_tab_20 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_118;
    (&kernelContext_118)->params_0 = params_20;
    (&kernelContext_118)->atom_0 = atom_20;
    (&kernelContext_118)->work_0 = work_20;
    (&kernelContext_118)->frames_0 = frames_20;
    (&kernelContext_118)->bodies_0 = bodies_20;
    (&kernelContext_118)->chunks_0 = chunks_20;
    (&kernelContext_118)->sort_0 = sort_20;
    (&kernelContext_118)->manifolds_0 = manifolds_20;
    (&kernelContext_118)->loads_0 = loads_20;
    (&kernelContext_118)->bonds_0 = bonds_20;
    (&kernelContext_118)->args_0 = args_20;
    (&kernelContext_118)->info_0 = info_20;
    (&kernelContext_118)->scan_0 = scan_20;
    (&kernelContext_118)->parts_0 = parts_20;
    (&kernelContext_118)->colliders_0 = colliders_20;
    (&kernelContext_118)->collider_info_0 = collider_info_20;
    (&kernelContext_118)->view_chunks_0 = view_chunks_20;
    (&kernelContext_118)->view_0 = view_20;
    (&kernelContext_118)->objects_0 = objects_20;
    (&kernelContext_118)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_118)->lines_0 = lines_20;
    (&kernelContext_118)->geometry_0 = geometry_20;
    (&kernelContext_118)->shots_0 = shots_20;
    (&kernelContext_118)->lp_0 = lp_20;
    (&kernelContext_118)->ext_0 = ext_21;
    (&kernelContext_118)->engine_contacts_0 = engine_contacts_20;
    (&kernelContext_118)->crush_0 = crush_20;
    (&kernelContext_118)->crush_rw_0 = crush_rw_20;
    (&kernelContext_118)->crush_tab_0 = crush_tab_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_118)->g_scan_0 = &g_scan_20;
    threadgroup array<float, int(64)> g_il_best_20;
    (&kernelContext_118)->g_il_best_0 = &g_il_best_20;
    uint c_18 = id_17.x;
    bool _S803 = live_0(c_18, &kernelContext_118);
    bool _S804;
    if(!_S803)
    {
        _S804 = true;
    }
    else
    {
        uint _S805 = body_of_0(c_18, &kernelContext_118);
        bool _S806 = dirty_0(_S805, &kernelContext_118);
        _S804 = !_S806;
    }
    if(_S804)
    {
        return;
    }
    uint _S807 = w_label_0(&kernelContext_118);
    uint device* _S808 = (&kernelContext_118)->work_0+(_S807 + c_18);
    uint r_21 = *_S808;
    uint _S809 = w_slot_0(&kernelContext_118);
    uint device* _S810 = (&kernelContext_118)->work_0+(_S809 + *_S808);
    uint slot_4 = *_S810;
    uint _S811 = body_of_0(c_18, &kernelContext_118);
    if((*_S810) != _S811)
    {
        _S804 = ((&kernelContext_118)->params_0->ev_moved_cap_0) != 0U;
    }
    else
    {
        _S804 = false;
    }
    if(_S804)
    {
        uint _S812 = a_counters_1(&kernelContext_118);
        uint i_12 = atomic_fetch_add_explicit((&kernelContext_118)->atom_0+(_S812 + 21U), 1U, memory_order_relaxed);
        if(i_12 < ((&kernelContext_118)->params_0->ev_moved_cap_0))
        {
            uint _S813 = a_moved_0(&kernelContext_118);
            uint _S814 = _S813 + 2U * i_12;
            atomic_store_explicit((&kernelContext_118)->atom_0+_S814, c_18, memory_order_relaxed);
            atomic_store_explicit((&kernelContext_118)->atom_0+(_S814 + 1U), slot_4, memory_order_relaxed);
        }
        else
        {
            uint _S815 = atomic_fetch_or_explicit((&kernelContext_118)->atom_0+(_S812 + 3U), 128U, memory_order_relaxed);
        }
    }
    *((&kernelContext_118)->work_0+(w_body_0() + c_18)) = slot_4;
    uint _S816 = 2U * slot_4;
    float3 _S817 = (float4(*((&kernelContext_118)->frames_0+_S816)) ).xyz;
    thread Quat_0 _S818 = quat_of_0(float4(*((&kernelContext_118)->frames_0+(_S816 + 1U))) );
    Quat_0 _S819 = conj_0(&_S818);
    float inv_mass_4 = (float4(*((&kernelContext_118)->bodies_0+7U * slot_4)) ).w;
    float _S820;
    if(inv_mass_4 > 0.0f)
    {
        _S820 = 1.0f / inv_mass_4;
    }
    else
    {
        _S820 = float_bits_0(2139095040U);
    }
    float4 _S821 = float4(*((&kernelContext_118)->chunks_0+(4U * c_18 + 2U))) ;
    uint first_1 = (as_type<uint>((_S821.z)));
    uint _S822 = (as_type<uint>((_S821.w)));
    float4 _S823 = float4(*((&kernelContext_118)->chunks_0+(4U * r_21 + 3U))) ;
    uint p_13 = first_1;
    for(;;)
    {
        if(p_13 < (first_1 + _S822))
        {
        }
        else
        {
            break;
        }
        uint _S824 = 2U * p_13;
        float4 _S825 = float4(*((&kernelContext_118)->parts_0+_S824)) ;
        uint col_0 = (as_type<uint>((_S825.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_118)->parts_0+(_S824 + 1U))) );
        uint _S826 = 4U * col_0;
        float4 _S827 = float4(*((&kernelContext_118)->colliders_0+_S826)) ;
        *((&kernelContext_118)->colliders_0+_S826) = packed_float4(float4((as_type<float>((slot_4))), _S827.y, _S827.z, _S827.w)) ;
        uint _S828 = _S826 + 1U;
        packed_float4 device* _S829 = (&kernelContext_118)->colliders_0+_S828;
        float3 _S830 = _S825.xyz - _S817;
        thread Quat_0 _S831 = _S819;
        float3 _S832 = rotate_0(&_S831, _S830);
        *_S829 = packed_float4(float4(_S832, (float4(*((&kernelContext_118)->colliders_0+_S828)) ).w)) ;
        packed_float4 device* _S833 = (&kernelContext_118)->colliders_0+(_S826 + 2U);
        thread Quat_0 _S834 = _S819;
        thread Quat_0 _S835 = rot_1;
        Quat_0 _S836 = quat_mul_0(&_S834, &_S835);
        thread Quat_0 _S837 = _S836;
        Quat_0 _S838 = normalized_0(&_S837);
        thread Quat_0 _S839 = _S838;
        float4 _S840 = quat_vec_0(&_S839);
        *_S833 = packed_float4(_S840) ;
        uint _S841 = 2U * col_0;
        uint4 _S842 = uint4(*((&kernelContext_118)->collider_info_0+_S841)) ;
        *((&kernelContext_118)->collider_info_0+_S841) = packed_uint4(uint4(_S842.x, slot_4, _S842.z, _S842.w)) ;
        *((&kernelContext_118)->collider_info_0+(_S841 + 1U)) = packed_uint4(uint4((as_type<uint>((_S820))), (as_type<uint>((_S823.x))), (as_type<uint>((_S823.w))), 0U)) ;
        p_13 = p_13 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], atomic<uint> device* atom_21 [[buffer(5)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], packed_float4 device* lp_21 [[buffer(24)]], packed_float4 device* ext_22 [[buffer(23)]], packed_float4 device* engine_contacts_21 [[buffer(25)]], CrushParams_0 constant* crush_21 [[buffer(28)]], packed_float4 device* crush_rw_21 [[buffer(27)]], packed_float4 device* crush_tab_21 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_119;
    (&kernelContext_119)->params_0 = params_21;
    (&kernelContext_119)->atom_0 = atom_21;
    (&kernelContext_119)->work_0 = work_21;
    (&kernelContext_119)->frames_0 = frames_21;
    (&kernelContext_119)->bodies_0 = bodies_21;
    (&kernelContext_119)->chunks_0 = chunks_21;
    (&kernelContext_119)->sort_0 = sort_21;
    (&kernelContext_119)->manifolds_0 = manifolds_21;
    (&kernelContext_119)->loads_0 = loads_21;
    (&kernelContext_119)->bonds_0 = bonds_21;
    (&kernelContext_119)->args_0 = args_21;
    (&kernelContext_119)->info_0 = info_21;
    (&kernelContext_119)->scan_0 = scan_21;
    (&kernelContext_119)->parts_0 = parts_21;
    (&kernelContext_119)->colliders_0 = colliders_21;
    (&kernelContext_119)->collider_info_0 = collider_info_21;
    (&kernelContext_119)->view_chunks_0 = view_chunks_21;
    (&kernelContext_119)->view_0 = view_21;
    (&kernelContext_119)->objects_0 = objects_21;
    (&kernelContext_119)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_119)->lines_0 = lines_21;
    (&kernelContext_119)->geometry_0 = geometry_21;
    (&kernelContext_119)->shots_0 = shots_21;
    (&kernelContext_119)->lp_0 = lp_21;
    (&kernelContext_119)->ext_0 = ext_22;
    (&kernelContext_119)->engine_contacts_0 = engine_contacts_21;
    (&kernelContext_119)->crush_0 = crush_21;
    (&kernelContext_119)->crush_rw_0 = crush_rw_21;
    (&kernelContext_119)->crush_tab_0 = crush_tab_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_119)->g_scan_0 = &g_scan_21;
    threadgroup array<float, int(64)> g_il_best_21;
    (&kernelContext_119)->g_il_best_0 = &g_il_best_21;
    uint s_10 = id_18.x;
    if(s_10 >= (params_21->body_count_0))
    {
        return;
    }
    uint _S843 = a_dirty_0(&kernelContext_119);
    atomic_store_explicit((&kernelContext_119)->atom_0+(_S843 + s_10), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_5 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_8 = fract(h_5);
    float q_18 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_8);
    float t_9 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_8));
    uint k_25 = uint(h_5);
    float3 _S844;
    if(k_25 == 0U)
    {
        _S844 = float3(0.92000001668930054f, t_9, 0.34959998726844788f);
    }
    else
    {
        if(k_25 == 1U)
        {
            _S844 = float3(q_18, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_25 == 2U)
            {
                _S844 = float3(0.34959998726844788f, 0.92000001668930054f, t_9);
            }
            else
            {
                if(k_25 == 3U)
                {
                    _S844 = float3(0.34959998726844788f, q_18, 0.92000001668930054f);
                }
                else
                {
                    if(k_25 == 4U)
                    {
                        _S844 = float3(t_9, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S844 = float3(0.92000001668930054f, 0.34959998726844788f, q_18);
                    }
                }
            }
        }
    }
    return _S844;
}

void write_object_0(uint o_4, const Quat_0 thread* q_19, float3 t_10, float4 color_0, KernelContext_0 thread* kernelContext_120)
{
    float3 _S845 = rotate_0(q_19, float3(1.0f, 0.0f, 0.0f));
    float3 _S846 = rotate_0(q_19, float3(0.0f, 1.0f, 0.0f));
    float3 _S847 = rotate_0(q_19, float3(0.0f, 0.0f, 1.0f));
    uint _S848 = 4U * o_4;
    *(kernelContext_120->objects_0+_S848) = packed_float4(float4(_S845.x, _S846.x, _S847.x, t_10.x)) ;
    *(kernelContext_120->objects_0+(_S848 + 1U)) = packed_float4(float4(_S845.y, _S846.y, _S847.y, t_10.y)) ;
    *(kernelContext_120->objects_0+(_S848 + 2U)) = packed_float4(float4(_S845.z, _S846.z, _S847.z, t_10.z)) ;
    *(kernelContext_120->objects_0+(_S848 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], atomic<uint> device* atom_22 [[buffer(5)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], packed_float4 device* lp_22 [[buffer(24)]], packed_float4 device* ext_23 [[buffer(23)]], packed_float4 device* engine_contacts_22 [[buffer(25)]], CrushParams_0 constant* crush_22 [[buffer(28)]], packed_float4 device* crush_rw_22 [[buffer(27)]], packed_float4 device* crush_tab_22 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_121;
    (&kernelContext_121)->params_0 = params_22;
    (&kernelContext_121)->atom_0 = atom_22;
    (&kernelContext_121)->work_0 = work_22;
    (&kernelContext_121)->frames_0 = frames_22;
    (&kernelContext_121)->bodies_0 = bodies_22;
    (&kernelContext_121)->chunks_0 = chunks_22;
    (&kernelContext_121)->sort_0 = sort_22;
    (&kernelContext_121)->manifolds_0 = manifolds_22;
    (&kernelContext_121)->loads_0 = loads_22;
    (&kernelContext_121)->bonds_0 = bonds_22;
    (&kernelContext_121)->args_0 = args_22;
    (&kernelContext_121)->info_0 = info_22;
    (&kernelContext_121)->scan_0 = scan_22;
    (&kernelContext_121)->parts_0 = parts_22;
    (&kernelContext_121)->colliders_0 = colliders_22;
    (&kernelContext_121)->collider_info_0 = collider_info_22;
    (&kernelContext_121)->view_chunks_0 = view_chunks_22;
    (&kernelContext_121)->view_0 = view_22;
    (&kernelContext_121)->objects_0 = objects_22;
    (&kernelContext_121)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_121)->lines_0 = lines_22;
    (&kernelContext_121)->geometry_0 = geometry_22;
    (&kernelContext_121)->shots_0 = shots_22;
    (&kernelContext_121)->lp_0 = lp_22;
    (&kernelContext_121)->ext_0 = ext_23;
    (&kernelContext_121)->engine_contacts_0 = engine_contacts_22;
    (&kernelContext_121)->crush_0 = crush_22;
    (&kernelContext_121)->crush_rw_0 = crush_rw_22;
    (&kernelContext_121)->crush_tab_0 = crush_tab_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_121)->g_scan_0 = &g_scan_22;
    threadgroup array<float, int(64)> g_il_best_22;
    (&kernelContext_121)->g_il_best_0 = &g_il_best_22;
    uint k_26 = id_19.x;
    bool _S849 = live_0(k_26, &kernelContext_121);
    if(!_S849)
    {
        return;
    }
    uint _S850 = body_of_0(k_26, &kernelContext_121);
    uint b_36 = 7U * _S850;
    thread Quat_0 q_20;
    thread float3 origin_5;
    structure_pose_0(_S850, b_36, &q_20, &origin_5, &kernelContext_121);
    uint _S851 = 3U * k_26;
    float4 _S852 = float4(*((&kernelContext_121)->view_chunks_0+_S851)) ;
    float3 base_2 = (float4(*((&kernelContext_121)->view_chunks_0+(_S851 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_121)->bodies_0+b_36)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_121)->view_0->mode_0) == 1U)
    {
        float _S853 = max(length((float4(*((&kernelContext_121)->bodies_0+(b_36 + 3U))) ).xyz), length((float4(*((&kernelContext_121)->bodies_0+(b_36 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S853 > ((&kernelContext_121)->view_0->rest_speed_0))
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
        if(((&kernelContext_121)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S850);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S850);
            }
        }
    }
    uint _S854 = (as_type<uint>((_S852.w)));
    Quat_0 _S855 = quat_of_0(float4(*((&kernelContext_121)->view_chunks_0+(_S851 + 1U))) );
    thread Quat_0 _S856 = q_20;
    thread Quat_0 _S857 = _S855;
    Quat_0 _S858 = quat_mul_0(&_S856, &_S857);
    thread Quat_0 _S859 = _S858;
    Quat_0 _S860 = normalized_0(&_S859);
    float3 _S861 = _S852.xyz;
    thread Quat_0 _S862 = q_20;
    float3 _S863 = rotate_0(&_S862, _S861);
    float3 _S864 = origin_5 + _S863;
    float4 _S865 = float4(color_1, 1.0f);
    thread Quat_0 _S866 = _S860;
    write_object_0(_S854, &_S866, _S864, _S865, &kernelContext_121);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], atomic<uint> device* atom_23 [[buffer(5)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], packed_float4 device* lp_23 [[buffer(24)]], packed_float4 device* ext_24 [[buffer(23)]], packed_float4 device* engine_contacts_23 [[buffer(25)]], CrushParams_0 constant* crush_23 [[buffer(28)]], packed_float4 device* crush_rw_23 [[buffer(27)]], packed_float4 device* crush_tab_23 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_122;
    (&kernelContext_122)->params_0 = params_23;
    (&kernelContext_122)->atom_0 = atom_23;
    (&kernelContext_122)->work_0 = work_23;
    (&kernelContext_122)->frames_0 = frames_23;
    (&kernelContext_122)->bodies_0 = bodies_23;
    (&kernelContext_122)->chunks_0 = chunks_23;
    (&kernelContext_122)->sort_0 = sort_23;
    (&kernelContext_122)->manifolds_0 = manifolds_23;
    (&kernelContext_122)->loads_0 = loads_23;
    (&kernelContext_122)->bonds_0 = bonds_23;
    (&kernelContext_122)->args_0 = args_23;
    (&kernelContext_122)->info_0 = info_23;
    (&kernelContext_122)->scan_0 = scan_23;
    (&kernelContext_122)->parts_0 = parts_23;
    (&kernelContext_122)->colliders_0 = colliders_23;
    (&kernelContext_122)->collider_info_0 = collider_info_23;
    (&kernelContext_122)->view_chunks_0 = view_chunks_23;
    (&kernelContext_122)->view_0 = view_23;
    (&kernelContext_122)->objects_0 = objects_23;
    (&kernelContext_122)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_122)->lines_0 = lines_23;
    (&kernelContext_122)->geometry_0 = geometry_23;
    (&kernelContext_122)->shots_0 = shots_23;
    (&kernelContext_122)->lp_0 = lp_23;
    (&kernelContext_122)->ext_0 = ext_24;
    (&kernelContext_122)->engine_contacts_0 = engine_contacts_23;
    (&kernelContext_122)->crush_0 = crush_23;
    (&kernelContext_122)->crush_rw_0 = crush_rw_23;
    (&kernelContext_122)->crush_tab_0 = crush_tab_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_122)->g_scan_0 = &g_scan_23;
    threadgroup array<float, int(64)> g_il_best_23;
    (&kernelContext_122)->g_il_best_0 = &g_il_best_23;
    uint i_13 = id_20.x;
    if(i_13 >= (view_23->shot_count_1))
    {
        return;
    }
    uint b_37 = 7U * (&kernelContext_122)->shot_bodies_0[i_13];
    uint o_5 = (&kernelContext_122)->view_0->impactor_base_0 + i_13;
    if(((float4(*((&kernelContext_122)->bodies_0+b_37)) ).w) == 0.0f)
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
            *((&kernelContext_122)->objects_0+(4U * o_5 + w_5)) = packed_float4(float4(0.0f) ) ;
            w_5 = w_5 + 1U;
        }
        return;
    }
    thread Quat_0 _S867 = quat_of_0(float4(*((&kernelContext_122)->bodies_0+(b_37 + 1U))) );
    Quat_0 _S868 = rot_from_rubble_0(&_S867);
    float3 _S869 = from_rubble_0((float4(*((&kernelContext_122)->bodies_0+b_37)) ).xyz);
    float4 _S870 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S871 = _S868;
    write_object_0(o_5, &_S871, _S869, _S870, &kernelContext_122);
    return;
}

float3 chunk_world_0(uint k_27, KernelContext_0 thread* kernelContext_123)
{
    uint _S872 = body_of_0(k_27, kernelContext_123);
    thread Quat_0 q_21;
    thread float3 origin_6;
    structure_pose_0(_S872, 7U * _S872, &q_21, &origin_6, kernelContext_123);
    float3 _S873 = (float4(*(kernelContext_123->view_chunks_0+3U * k_27)) ).xyz;
    thread Quat_0 _S874 = q_21;
    float3 _S875 = rotate_0(&_S874, _S873);
    return origin_6 + _S875;
}

void write_line_vertex_0(uint v_13, float3 p_14, KernelContext_0 thread* kernelContext_124)
{
    uint _S876 = 7U * v_13;
    *(kernelContext_124->lines_0+_S876) = p_14.x;
    *(kernelContext_124->lines_0+(_S876 + 1U)) = p_14.y;
    *(kernelContext_124->lines_0+(_S876 + 2U)) = p_14.z;
    *(kernelContext_124->lines_0+(_S876 + 3U)) = 1.0f;
    *(kernelContext_124->lines_0+(_S876 + 4U)) = 0.86000001430511475f;
    *(kernelContext_124->lines_0+(_S876 + 5U)) = 0.25f;
    *(kernelContext_124->lines_0+(_S876 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], atomic<uint> device* atom_24 [[buffer(5)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], packed_float4 device* lp_24 [[buffer(24)]], packed_float4 device* ext_25 [[buffer(23)]], packed_float4 device* engine_contacts_24 [[buffer(25)]], CrushParams_0 constant* crush_24 [[buffer(28)]], packed_float4 device* crush_rw_24 [[buffer(27)]], packed_float4 device* crush_tab_24 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_125;
    (&kernelContext_125)->params_0 = params_24;
    (&kernelContext_125)->atom_0 = atom_24;
    (&kernelContext_125)->work_0 = work_24;
    (&kernelContext_125)->frames_0 = frames_24;
    (&kernelContext_125)->bodies_0 = bodies_24;
    (&kernelContext_125)->chunks_0 = chunks_24;
    (&kernelContext_125)->sort_0 = sort_24;
    (&kernelContext_125)->manifolds_0 = manifolds_24;
    (&kernelContext_125)->loads_0 = loads_24;
    (&kernelContext_125)->bonds_0 = bonds_24;
    (&kernelContext_125)->args_0 = args_24;
    (&kernelContext_125)->info_0 = info_24;
    (&kernelContext_125)->scan_0 = scan_24;
    (&kernelContext_125)->parts_0 = parts_24;
    (&kernelContext_125)->colliders_0 = colliders_24;
    (&kernelContext_125)->collider_info_0 = collider_info_24;
    (&kernelContext_125)->view_chunks_0 = view_chunks_24;
    (&kernelContext_125)->view_0 = view_24;
    (&kernelContext_125)->objects_0 = objects_24;
    (&kernelContext_125)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_125)->lines_0 = lines_24;
    (&kernelContext_125)->geometry_0 = geometry_24;
    (&kernelContext_125)->shots_0 = shots_24;
    (&kernelContext_125)->lp_0 = lp_24;
    (&kernelContext_125)->ext_0 = ext_25;
    (&kernelContext_125)->engine_contacts_0 = engine_contacts_24;
    (&kernelContext_125)->crush_0 = crush_24;
    (&kernelContext_125)->crush_rw_0 = crush_rw_24;
    (&kernelContext_125)->crush_tab_0 = crush_tab_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_125)->g_scan_0 = &g_scan_24;
    threadgroup array<float, int(64)> g_il_best_24;
    (&kernelContext_125)->g_il_best_0 = &g_il_best_24;
    uint i_14 = id_21.x;
    if(i_14 >= (params_24->bond_used_0))
    {
        return;
    }
    uint4 _S877 = uint4(*((&kernelContext_125)->bonds_0+i_14)) ;
    float3 _S878 = chunk_world_0(_S877.x, &kernelContext_125);
    uint _S879 = w_alive_0(&kernelContext_125);
    float3 b_38;
    if((*((&kernelContext_125)->work_0+(_S879 + i_14))) != 0U)
    {
        float3 _S880 = chunk_world_0(_S877.y, &kernelContext_125);
        b_38 = _S880;
    }
    else
    {
        b_38 = _S878;
    }
    uint _S881 = 2U * i_14;
    write_line_vertex_0(_S881, _S878, &kernelContext_125);
    write_line_vertex_0(_S881 + 1U, b_38, &kernelContext_125);
    return;
}

[[kernel]] void fr_strike_reset(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], atomic<uint> device* atom_25 [[buffer(5)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], packed_float4 device* lp_25 [[buffer(24)]], packed_float4 device* ext_26 [[buffer(23)]], packed_float4 device* engine_contacts_25 [[buffer(25)]], CrushParams_0 constant* crush_25 [[buffer(28)]], packed_float4 device* crush_rw_25 [[buffer(27)]], packed_float4 device* crush_tab_25 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_126;
    (&kernelContext_126)->params_0 = params_25;
    (&kernelContext_126)->atom_0 = atom_25;
    (&kernelContext_126)->work_0 = work_25;
    (&kernelContext_126)->frames_0 = frames_25;
    (&kernelContext_126)->bodies_0 = bodies_25;
    (&kernelContext_126)->chunks_0 = chunks_25;
    (&kernelContext_126)->sort_0 = sort_25;
    (&kernelContext_126)->manifolds_0 = manifolds_25;
    (&kernelContext_126)->loads_0 = loads_25;
    (&kernelContext_126)->bonds_0 = bonds_25;
    (&kernelContext_126)->args_0 = args_25;
    (&kernelContext_126)->info_0 = info_25;
    (&kernelContext_126)->scan_0 = scan_25;
    (&kernelContext_126)->parts_0 = parts_25;
    (&kernelContext_126)->colliders_0 = colliders_25;
    (&kernelContext_126)->collider_info_0 = collider_info_25;
    (&kernelContext_126)->view_chunks_0 = view_chunks_25;
    (&kernelContext_126)->view_0 = view_25;
    (&kernelContext_126)->objects_0 = objects_25;
    (&kernelContext_126)->shot_bodies_0 = shot_bodies_25;
    (&kernelContext_126)->lines_0 = lines_25;
    (&kernelContext_126)->geometry_0 = geometry_25;
    (&kernelContext_126)->shots_0 = shots_25;
    (&kernelContext_126)->lp_0 = lp_25;
    (&kernelContext_126)->ext_0 = ext_26;
    (&kernelContext_126)->engine_contacts_0 = engine_contacts_25;
    (&kernelContext_126)->crush_0 = crush_25;
    (&kernelContext_126)->crush_rw_0 = crush_rw_25;
    (&kernelContext_126)->crush_tab_0 = crush_tab_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_126)->g_scan_0 = &g_scan_25;
    threadgroup array<float, int(64)> g_il_best_25;
    (&kernelContext_126)->g_il_best_0 = &g_il_best_25;
    uint b_39 = id_22.x;
    if(b_39 >= (params_25->body_count_0))
    {
        return;
    }
    uint _S882 = a_strike_0(&kernelContext_126);
    uint at_13 = _S882 + 9U * b_39;
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
        atomic_store_explicit((&kernelContext_126)->atom_0+(at_13 + w_6), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_126)->atom_0+(at_13 + 3U + w_6), 0U, memory_order_relaxed);
        w_6 = w_6 + 1U;
    }
    atomic_store_explicit((&kernelContext_126)->atom_0+(at_13 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_126)->atom_0+(at_13 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_126)->atom_0+(at_13 + 8U), 4294967295U, memory_order_relaxed);
    uint _S883 = strike_at_0(b_39, &kernelContext_126);
    (*((&kernelContext_126)->frames_0+_S883))[int(3)] = 0.0f;
    return;
}

uint ordered_0(float f_9)
{
    uint u_6 = (as_type<uint>((f_9)));
    uint _S884;
    if((u_6 & 2147483648U) != 0U)
    {
        _S884 = ~u_6;
    }
    else
    {
        _S884 = u_6 | 2147483648U;
    }
    return _S884;
}

[[kernel]] void fr_strike_bounds(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], atomic<uint> device* atom_26 [[buffer(5)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], packed_float4 device* lp_26 [[buffer(24)]], packed_float4 device* ext_27 [[buffer(23)]], packed_float4 device* engine_contacts_26 [[buffer(25)]], CrushParams_0 constant* crush_26 [[buffer(28)]], packed_float4 device* crush_rw_26 [[buffer(27)]], packed_float4 device* crush_tab_26 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_127;
    (&kernelContext_127)->params_0 = params_26;
    (&kernelContext_127)->atom_0 = atom_26;
    (&kernelContext_127)->work_0 = work_26;
    (&kernelContext_127)->frames_0 = frames_26;
    (&kernelContext_127)->bodies_0 = bodies_26;
    (&kernelContext_127)->chunks_0 = chunks_26;
    (&kernelContext_127)->sort_0 = sort_26;
    (&kernelContext_127)->manifolds_0 = manifolds_26;
    (&kernelContext_127)->loads_0 = loads_26;
    (&kernelContext_127)->bonds_0 = bonds_26;
    (&kernelContext_127)->args_0 = args_26;
    (&kernelContext_127)->info_0 = info_26;
    (&kernelContext_127)->scan_0 = scan_26;
    (&kernelContext_127)->parts_0 = parts_26;
    (&kernelContext_127)->colliders_0 = colliders_26;
    (&kernelContext_127)->collider_info_0 = collider_info_26;
    (&kernelContext_127)->view_chunks_0 = view_chunks_26;
    (&kernelContext_127)->view_0 = view_26;
    (&kernelContext_127)->objects_0 = objects_26;
    (&kernelContext_127)->shot_bodies_0 = shot_bodies_26;
    (&kernelContext_127)->lines_0 = lines_26;
    (&kernelContext_127)->geometry_0 = geometry_26;
    (&kernelContext_127)->shots_0 = shots_26;
    (&kernelContext_127)->lp_0 = lp_26;
    (&kernelContext_127)->ext_0 = ext_27;
    (&kernelContext_127)->engine_contacts_0 = engine_contacts_26;
    (&kernelContext_127)->crush_0 = crush_26;
    (&kernelContext_127)->crush_rw_0 = crush_rw_26;
    (&kernelContext_127)->crush_tab_0 = crush_tab_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_127)->g_scan_0 = &g_scan_26;
    threadgroup array<float, int(64)> g_il_best_26;
    (&kernelContext_127)->g_il_best_0 = &g_il_best_26;
    uint k_28 = id_23.x;
    bool _S885 = live_0(k_28, &kernelContext_127);
    bool _S886;
    if(!_S885)
    {
        _S886 = true;
    }
    else
    {
        _S886 = ((&kernelContext_127)->params_0->shot_count_0) == 0U;
    }
    if(_S886)
    {
        return;
    }
    thread Quat_0 q_22;
    float3 _S887 = chunk_box_world_0(k_28, &q_22, &kernelContext_127);
    float _S888 = (float4(*((&kernelContext_127)->geometry_0+3U * k_28)) ).w;
    uint _S889 = a_strike_0(&kernelContext_127);
    uint _S890 = body_of_0(k_28, &kernelContext_127);
    uint _S891 = _S889 + 9U * _S890;
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
        uint _S892 = ax_1;
        uint _S893 = atomic_fetch_min_explicit((&kernelContext_127)->atom_0+(_S891 + ax_1), ordered_0(_S887[ax_1] - _S888), memory_order_relaxed);
        uint _S894 = atomic_fetch_max_explicit((&kernelContext_127)->atom_0+(_S891 + 3U + ax_1), ordered_0(_S887[_S892] + _S888), memory_order_relaxed);
        ax_1 = ax_1 + 1U;
    }
    return;
}

float unordered_0(uint u_7)
{
    uint _S895;
    if((u_7 & 2147483648U) != 0U)
    {
        _S895 = u_7 & 2147483647U;
    }
    else
    {
        _S895 = ~u_7;
    }
    return (as_type<float>((_S895)));
}

[[kernel]] void fr_strike_find(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], atomic<uint> device* atom_27 [[buffer(5)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], packed_float4 device* lp_27 [[buffer(24)]], packed_float4 device* ext_28 [[buffer(23)]], packed_float4 device* engine_contacts_27 [[buffer(25)]], CrushParams_0 constant* crush_27 [[buffer(28)]], packed_float4 device* crush_rw_27 [[buffer(27)]], packed_float4 device* crush_tab_27 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_128;
    (&kernelContext_128)->params_0 = params_27;
    (&kernelContext_128)->atom_0 = atom_27;
    (&kernelContext_128)->work_0 = work_27;
    (&kernelContext_128)->frames_0 = frames_27;
    (&kernelContext_128)->bodies_0 = bodies_27;
    (&kernelContext_128)->chunks_0 = chunks_27;
    (&kernelContext_128)->sort_0 = sort_27;
    (&kernelContext_128)->manifolds_0 = manifolds_27;
    (&kernelContext_128)->loads_0 = loads_27;
    (&kernelContext_128)->bonds_0 = bonds_27;
    (&kernelContext_128)->args_0 = args_27;
    (&kernelContext_128)->info_0 = info_27;
    (&kernelContext_128)->scan_0 = scan_27;
    (&kernelContext_128)->parts_0 = parts_27;
    (&kernelContext_128)->colliders_0 = colliders_27;
    (&kernelContext_128)->collider_info_0 = collider_info_27;
    (&kernelContext_128)->view_chunks_0 = view_chunks_27;
    (&kernelContext_128)->view_0 = view_27;
    (&kernelContext_128)->objects_0 = objects_27;
    (&kernelContext_128)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_128)->lines_0 = lines_27;
    (&kernelContext_128)->geometry_0 = geometry_27;
    (&kernelContext_128)->shots_0 = shots_27;
    (&kernelContext_128)->lp_0 = lp_27;
    (&kernelContext_128)->ext_0 = ext_28;
    (&kernelContext_128)->engine_contacts_0 = engine_contacts_27;
    (&kernelContext_128)->crush_0 = crush_27;
    (&kernelContext_128)->crush_rw_0 = crush_rw_27;
    (&kernelContext_128)->crush_tab_0 = crush_tab_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_128)->g_scan_0 = &g_scan_27;
    threadgroup array<float, int(64)> g_il_best_27;
    (&kernelContext_128)->g_il_best_0 = &g_il_best_27;
    uint b_40 = id_24.x;
    bool _S896;
    if(b_40 >= (params_27->body_count_0))
    {
        _S896 = true;
    }
    else
    {
        _S896 = ((&kernelContext_128)->params_0->shot_count_0) == 0U;
    }
    if(_S896)
    {
        return;
    }
    uint _S897 = a_strike_0(&kernelContext_128);
    uint at_14 = _S897 + 9U * b_40;
    uint _S898 = atomic_load_explicit((&kernelContext_128)->atom_0+at_14, memory_order_relaxed);
    if(_S898 == 4294967295U)
    {
        return;
    }
    uint _S899 = atomic_load_explicit((&kernelContext_128)->atom_0+at_14, memory_order_relaxed);
    float _S900 = unordered_0(_S899);
    uint _S901 = atomic_load_explicit((&kernelContext_128)->atom_0+(at_14 + 1U), memory_order_relaxed);
    float _S902 = unordered_0(_S901);
    uint _S903 = atomic_load_explicit((&kernelContext_128)->atom_0+(at_14 + 2U), memory_order_relaxed);
    float3 _S904 = float3(_S900, _S902, unordered_0(_S903));
    uint _S905 = atomic_load_explicit((&kernelContext_128)->atom_0+(at_14 + 3U), memory_order_relaxed);
    float _S906 = unordered_0(_S905);
    uint _S907 = atomic_load_explicit((&kernelContext_128)->atom_0+(at_14 + 4U), memory_order_relaxed);
    float _S908 = unordered_0(_S907);
    uint _S909 = atomic_load_explicit((&kernelContext_128)->atom_0+(at_14 + 5U), memory_order_relaxed);
    float3 _S910 = float3(_S906, _S908, unordered_0(_S909));
    float3 _S911 = from_rubble_0((float4(*((&kernelContext_128)->bodies_0+(7U * b_40 + 3U))) ).xyz);
    float _S912 = float_bits_0(2139095040U);
    uint k_29 = 0U;
    for(;;)
    {
        if(k_29 < ((&kernelContext_128)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S913 = float4(*((&kernelContext_128)->shots_0+2U * k_29)) ;
        uint sb_1 = 7U * (as_type<uint>((_S913.x)));
        if(((float4(*((&kernelContext_128)->bodies_0+sb_1)) ).w) == 0.0f)
        {
            k_29 = k_29 + 1U;
            continue;
        }
        float3 c_19 = from_rubble_0((float4(*((&kernelContext_128)->bodies_0+sb_1)) ).xyz);
        float3 rel_2 = from_rubble_0((float4(*((&kernelContext_128)->bodies_0+(sb_1 + 3U))) ).xyz) - _S911;
        if(!((length(rel_2)) > ((&kernelContext_128)->params_0->strike_speed_0)))
        {
            k_29 = k_29 + 1U;
            continue;
        }
        float3 _S914 = float3(_S913.y) ;
        float3 _S915 = _S904 - _S914;
        float3 _S916 = _S910 + _S914;
        float _S917 = (&kernelContext_128)->params_0->dt_0;
        float t0_1 = 0.0f;
        float t1_1 = _S917;
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
            uint _S918 = ax_2;
            if((abs(rel_2[ax_2])) < 9.999999960041972e-13f)
            {
                uint _S919 = ax_2;
                if((c_19[ax_2]) < (_S915[ax_2]))
                {
                    _S896 = true;
                }
                else
                {
                    _S896 = (c_19[_S919]) > (_S916[ax_2]);
                }
                float t0_2;
                if(_S896)
                {
                    t0_2 = _S912;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + 1U;
                continue;
            }
            float ta_1 = (_S915[ax_2] - c_19[ax_2]) / rel_2[_S918];
            float tb_1 = (_S916[ax_2] - c_19[ax_2]) / rel_2[_S918];
            float _S920 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S920;
            ax_2 = ax_2 + 1U;
        }
        if(t0_1 <= t1_1)
        {
            uint _S921 = strike_at_0(b_40, &kernelContext_128);
            *((&kernelContext_128)->frames_0+_S921) = packed_float4(float4(c_19, float(k_29 + 1U))) ;
            *((&kernelContext_128)->frames_0+(_S921 + 1U)) = packed_float4(float4(rel_2, t0_1)) ;
            return;
        }
        k_29 = k_29 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_entry(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], atomic<uint> device* atom_28 [[buffer(5)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], packed_float4 device* lp_28 [[buffer(24)]], packed_float4 device* ext_29 [[buffer(23)]], packed_float4 device* engine_contacts_28 [[buffer(25)]], CrushParams_0 constant* crush_28 [[buffer(28)]], packed_float4 device* crush_rw_28 [[buffer(27)]], packed_float4 device* crush_tab_28 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_129;
    (&kernelContext_129)->params_0 = params_28;
    (&kernelContext_129)->atom_0 = atom_28;
    (&kernelContext_129)->work_0 = work_28;
    (&kernelContext_129)->frames_0 = frames_28;
    (&kernelContext_129)->bodies_0 = bodies_28;
    (&kernelContext_129)->chunks_0 = chunks_28;
    (&kernelContext_129)->sort_0 = sort_28;
    (&kernelContext_129)->manifolds_0 = manifolds_28;
    (&kernelContext_129)->loads_0 = loads_28;
    (&kernelContext_129)->bonds_0 = bonds_28;
    (&kernelContext_129)->args_0 = args_28;
    (&kernelContext_129)->info_0 = info_28;
    (&kernelContext_129)->scan_0 = scan_28;
    (&kernelContext_129)->parts_0 = parts_28;
    (&kernelContext_129)->colliders_0 = colliders_28;
    (&kernelContext_129)->collider_info_0 = collider_info_28;
    (&kernelContext_129)->view_chunks_0 = view_chunks_28;
    (&kernelContext_129)->view_0 = view_28;
    (&kernelContext_129)->objects_0 = objects_28;
    (&kernelContext_129)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_129)->lines_0 = lines_28;
    (&kernelContext_129)->geometry_0 = geometry_28;
    (&kernelContext_129)->shots_0 = shots_28;
    (&kernelContext_129)->lp_0 = lp_28;
    (&kernelContext_129)->ext_0 = ext_29;
    (&kernelContext_129)->engine_contacts_0 = engine_contacts_28;
    (&kernelContext_129)->crush_0 = crush_28;
    (&kernelContext_129)->crush_rw_0 = crush_rw_28;
    (&kernelContext_129)->crush_tab_0 = crush_tab_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_129)->g_scan_0 = &g_scan_28;
    threadgroup array<float, int(64)> g_il_best_28;
    (&kernelContext_129)->g_il_best_0 = &g_il_best_28;
    uint k_30 = id_25.x;
    bool _S922 = live_0(k_30, &kernelContext_129);
    bool _S923;
    if(!_S922)
    {
        _S923 = true;
    }
    else
    {
        _S923 = ((&kernelContext_129)->params_0->shot_count_0) == 0U;
    }
    if(_S923)
    {
        return;
    }
    uint _S924 = body_of_0(k_30, &kernelContext_129);
    thread uint shot_5;
    bool _S925 = struck_0(_S924, &shot_5, &kernelContext_129);
    if(!_S925)
    {
        return;
    }
    uint _S926 = strike_at_0(_S924, &kernelContext_129);
    float4 _S927 = float4(*((&kernelContext_129)->frames_0+(_S926 + 1U))) ;
    float3 from_1 = (float4(*((&kernelContext_129)->frames_0+_S926)) ).xyz;
    float3 rel_3 = _S927.xyz;
    float3 d_20 = rel_3 / float3(max(length(rel_3), 9.999999960041972e-13f)) ;
    float _S928 = (float4(*((&kernelContext_129)->shots_0+2U * shot_5)) ).y;
    thread Quat_0 q_23;
    float3 _S929 = chunk_box_world_0(k_30, &q_23, &kernelContext_129);
    uint _S930 = 3U * k_30;
    Quat_0 _S931 = quat_of_0(float4(*((&kernelContext_129)->geometry_0+(_S930 + 2U))) );
    thread Quat_0 _S932 = q_23;
    thread Quat_0 _S933 = _S931;
    Quat_0 _S934 = quat_mul_0(&_S932, &_S933);
    thread Quat_0 _S935 = _S934;
    Quat_0 _S936 = normalized_0(&_S935);
    float3 _S937 = (float4(*((&kernelContext_129)->geometry_0+(_S930 + 1U))) ).xyz;
    float _S938 = float_bits_0(2139095040U);
    float _S939 = (&kernelContext_129)->params_0->dt_0;
    float t_in_5 = 0.0f;
    float t_out_0 = _S939;
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
        thread Quat_0 _S940 = _S936;
        float3 _S941 = rotate_0(&_S940, e_3);
        float o_6 = dot(from_1 - _S929, _S941);
        float dv_1 = dot(rel_3, _S941);
        float ext_30 = _S937[ax_3] + _S928;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            float t_in_6;
            if((abs(o_6)) > ext_30)
            {
                t_in_6 = _S938;
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
        float _S942 = min(t_out_0, max(ta_2, tb_2));
        t_in_5 = max(t_in_5, min(ta_2, tb_2));
        t_out_0 = _S942;
        ax_3 = ax_3 + 1U;
    }
    uint _S943 = a_strike_0(&kernelContext_129);
    uint at_15 = _S943 + 9U * _S924;
    if(t_in_5 <= t_out_0)
    {
        uint _S944 = atomic_fetch_min_explicit((&kernelContext_129)->atom_0+(at_15 + 6U), (as_type<uint>((t_in_5))), memory_order_relaxed);
    }
    float3 r_22 = _S929 - (from_1 + rel_3 * float3(_S927.w) );
    float along_1 = dot(r_22, d_20);
    float dist_0 = length(r_22 - d_20 * float3(along_1) ) + abs(along_1) * 0.00100000004749745f;
    uint _S945 = a_strike_dist_0(&kernelContext_129);
    uint _S946 = (as_type<uint>((dist_0)));
    atomic_store_explicit((&kernelContext_129)->atom_0+(_S945 + k_30), _S946, memory_order_relaxed);
    uint _S947 = atomic_fetch_min_explicit((&kernelContext_129)->atom_0+(at_15 + 7U), _S946, memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], atomic<uint> device* atom_29 [[buffer(5)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], packed_float4 device* lp_29 [[buffer(24)]], packed_float4 device* ext_31 [[buffer(23)]], packed_float4 device* engine_contacts_29 [[buffer(25)]], CrushParams_0 constant* crush_29 [[buffer(28)]], packed_float4 device* crush_rw_29 [[buffer(27)]], packed_float4 device* crush_tab_29 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_130;
    (&kernelContext_130)->params_0 = params_29;
    (&kernelContext_130)->atom_0 = atom_29;
    (&kernelContext_130)->work_0 = work_29;
    (&kernelContext_130)->frames_0 = frames_29;
    (&kernelContext_130)->bodies_0 = bodies_29;
    (&kernelContext_130)->chunks_0 = chunks_29;
    (&kernelContext_130)->sort_0 = sort_29;
    (&kernelContext_130)->manifolds_0 = manifolds_29;
    (&kernelContext_130)->loads_0 = loads_29;
    (&kernelContext_130)->bonds_0 = bonds_29;
    (&kernelContext_130)->args_0 = args_29;
    (&kernelContext_130)->info_0 = info_29;
    (&kernelContext_130)->scan_0 = scan_29;
    (&kernelContext_130)->parts_0 = parts_29;
    (&kernelContext_130)->colliders_0 = colliders_29;
    (&kernelContext_130)->collider_info_0 = collider_info_29;
    (&kernelContext_130)->view_chunks_0 = view_chunks_29;
    (&kernelContext_130)->view_0 = view_29;
    (&kernelContext_130)->objects_0 = objects_29;
    (&kernelContext_130)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_130)->lines_0 = lines_29;
    (&kernelContext_130)->geometry_0 = geometry_29;
    (&kernelContext_130)->shots_0 = shots_29;
    (&kernelContext_130)->lp_0 = lp_29;
    (&kernelContext_130)->ext_0 = ext_31;
    (&kernelContext_130)->engine_contacts_0 = engine_contacts_29;
    (&kernelContext_130)->crush_0 = crush_29;
    (&kernelContext_130)->crush_rw_0 = crush_rw_29;
    (&kernelContext_130)->crush_tab_0 = crush_tab_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_130)->g_scan_0 = &g_scan_29;
    threadgroup array<float, int(64)> g_il_best_29;
    (&kernelContext_130)->g_il_best_0 = &g_il_best_29;
    uint k_31 = id_26.x;
    bool _S948 = live_0(k_31, &kernelContext_130);
    bool _S949;
    if(!_S948)
    {
        _S949 = true;
    }
    else
    {
        _S949 = ((&kernelContext_130)->params_0->shot_count_0) == 0U;
    }
    if(_S949)
    {
        return;
    }
    uint _S950 = body_of_0(k_31, &kernelContext_130);
    thread uint shot_6;
    bool _S951 = struck_0(_S950, &shot_6, &kernelContext_130);
    if(!_S951)
    {
        return;
    }
    uint _S952 = a_strike_0(&kernelContext_130);
    uint at_16 = _S952 + 9U * _S950;
    uint _S953 = a_strike_dist_0(&kernelContext_130);
    uint _S954 = atomic_load_explicit((&kernelContext_130)->atom_0+(_S953 + k_31), memory_order_relaxed);
    uint _S955 = atomic_load_explicit((&kernelContext_130)->atom_0+(at_16 + 7U), memory_order_relaxed);
    if(_S954 == _S955)
    {
        uint _S956 = atomic_fetch_min_explicit((&kernelContext_130)->atom_0+(at_16 + 8U), k_31, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], atomic<uint> device* atom_30 [[buffer(5)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], packed_float4 device* lp_30 [[buffer(24)]], packed_float4 device* ext_32 [[buffer(23)]], packed_float4 device* engine_contacts_30 [[buffer(25)]], CrushParams_0 constant* crush_30 [[buffer(28)]], packed_float4 device* crush_rw_30 [[buffer(27)]], packed_float4 device* crush_tab_30 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_131;
    (&kernelContext_131)->params_0 = params_30;
    (&kernelContext_131)->atom_0 = atom_30;
    (&kernelContext_131)->work_0 = work_30;
    (&kernelContext_131)->frames_0 = frames_30;
    (&kernelContext_131)->bodies_0 = bodies_30;
    (&kernelContext_131)->chunks_0 = chunks_30;
    (&kernelContext_131)->sort_0 = sort_30;
    (&kernelContext_131)->manifolds_0 = manifolds_30;
    (&kernelContext_131)->loads_0 = loads_30;
    (&kernelContext_131)->bonds_0 = bonds_30;
    (&kernelContext_131)->args_0 = args_30;
    (&kernelContext_131)->info_0 = info_30;
    (&kernelContext_131)->scan_0 = scan_30;
    (&kernelContext_131)->parts_0 = parts_30;
    (&kernelContext_131)->colliders_0 = colliders_30;
    (&kernelContext_131)->collider_info_0 = collider_info_30;
    (&kernelContext_131)->view_chunks_0 = view_chunks_30;
    (&kernelContext_131)->view_0 = view_30;
    (&kernelContext_131)->objects_0 = objects_30;
    (&kernelContext_131)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_131)->lines_0 = lines_30;
    (&kernelContext_131)->geometry_0 = geometry_30;
    (&kernelContext_131)->shots_0 = shots_30;
    (&kernelContext_131)->lp_0 = lp_30;
    (&kernelContext_131)->ext_0 = ext_32;
    (&kernelContext_131)->engine_contacts_0 = engine_contacts_30;
    (&kernelContext_131)->crush_0 = crush_30;
    (&kernelContext_131)->crush_rw_0 = crush_rw_30;
    (&kernelContext_131)->crush_tab_0 = crush_tab_30;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_131)->g_scan_0 = &g_scan_30;
    threadgroup array<float, int(64)> g_il_best_30;
    (&kernelContext_131)->g_il_best_0 = &g_il_best_30;
    uint b_41 = id_27.x;
    bool _S957;
    if(b_41 >= (params_30->body_count_0))
    {
        _S957 = true;
    }
    else
    {
        _S957 = ((&kernelContext_131)->params_0->shot_count_0) == 0U;
    }
    if(_S957)
    {
        return;
    }
    thread uint shot_7;
    bool _S958 = struck_0(b_41, &shot_7, &kernelContext_131);
    if(!_S958)
    {
        return;
    }
    uint _S959 = a_strike_0(&kernelContext_131);
    uint at_17 = _S959 + 9U * b_41;
    uint _S960 = atomic_load_explicit((&kernelContext_131)->atom_0+(at_17 + 6U), memory_order_relaxed);
    float t_in_7 = (as_type<float>((_S960)));
    uint entry_2 = atomic_load_explicit((&kernelContext_131)->atom_0+(at_17 + 8U), memory_order_relaxed);
    uint _S961 = strike_at_0(b_41, &kernelContext_131);
    if(!(t_in_7 <= ((&kernelContext_131)->params_0->dt_0)))
    {
        _S957 = true;
    }
    else
    {
        _S957 = entry_2 == 4294967295U;
    }
    if(_S957)
    {
        (*((&kernelContext_131)->frames_0+_S961))[int(3)] = 0.0f;
        return;
    }
    float3 rel_4 = (float4(*((&kernelContext_131)->frames_0+(_S961 + 1U))) ).xyz;
    float speed_14 = length(rel_4);
    float4 _S962 = float4(*((&kernelContext_131)->shots_0+2U * shot_7)) ;
    float _S963 = max(_S962.z, 0.00100000004749745f);
    float m_imp_0 = _S962.w;
    float e_imp_0 = (float4(*((&kernelContext_131)->shots_0+(2U * shot_7 + 1U))) ).x;
    float3 contact_16 = (float4(*((&kernelContext_131)->frames_0+_S961)) ).xyz + rel_4 * float3(t_in_7)  + rel_4 / float3(max(speed_14, 9.999999960041972e-13f))  * float3(_S962.y) ;
    float inv_mass_5 = (float4(*((&kernelContext_131)->bodies_0+7U * b_41)) ).w;
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
    float modulus_1 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_131)->geometry_0+(3U * entry_2 + 1U))) ).w);
    *((&kernelContext_131)->frames_0+(_S961 + 2U)) = packed_float4(float4(contact_16, m_eff_6 * speed_14 / (2.86999988555908203f * pow(m_eff_6 * m_eff_6 / (_S963 * modulus_1 * modulus_1 * max(speed_14, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

bool strike_share_breaks_0(uint4 bond_1, float3 contact_17, float force_2, KernelContext_0 thread* kernelContext_132)
{
    thread Quat_0 q_24;
    float3 _S964 = chunk_box_world_0(bond_1.x, &q_24, kernelContext_132);
    float3 _S965 = chunk_box_world_0(bond_1.y, &q_24, kernelContext_132);
    float dist_1 = length(float3(0.5f)  * (_S964 + _S965) - contact_17);
    return (force_2 * min(1.0f, (as_type<float>((bond_1.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((bond_1.z)));
}

[[kernel]] void fr_strike_breaks(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], atomic<uint> device* atom_31 [[buffer(5)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_31 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], packed_float4 device* lp_31 [[buffer(24)]], packed_float4 device* ext_33 [[buffer(23)]], packed_float4 device* engine_contacts_31 [[buffer(25)]], CrushParams_0 constant* crush_31 [[buffer(28)]], packed_float4 device* crush_rw_31 [[buffer(27)]], packed_float4 device* crush_tab_31 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_133;
    (&kernelContext_133)->params_0 = params_31;
    (&kernelContext_133)->atom_0 = atom_31;
    (&kernelContext_133)->work_0 = work_31;
    (&kernelContext_133)->frames_0 = frames_31;
    (&kernelContext_133)->bodies_0 = bodies_31;
    (&kernelContext_133)->chunks_0 = chunks_31;
    (&kernelContext_133)->sort_0 = sort_31;
    (&kernelContext_133)->manifolds_0 = manifolds_31;
    (&kernelContext_133)->loads_0 = loads_31;
    (&kernelContext_133)->bonds_0 = bonds_31;
    (&kernelContext_133)->args_0 = args_31;
    (&kernelContext_133)->info_0 = info_31;
    (&kernelContext_133)->scan_0 = scan_31;
    (&kernelContext_133)->parts_0 = parts_31;
    (&kernelContext_133)->colliders_0 = colliders_31;
    (&kernelContext_133)->collider_info_0 = collider_info_31;
    (&kernelContext_133)->view_chunks_0 = view_chunks_31;
    (&kernelContext_133)->view_0 = view_31;
    (&kernelContext_133)->objects_0 = objects_31;
    (&kernelContext_133)->shot_bodies_0 = shot_bodies_31;
    (&kernelContext_133)->lines_0 = lines_31;
    (&kernelContext_133)->geometry_0 = geometry_31;
    (&kernelContext_133)->shots_0 = shots_31;
    (&kernelContext_133)->lp_0 = lp_31;
    (&kernelContext_133)->ext_0 = ext_33;
    (&kernelContext_133)->engine_contacts_0 = engine_contacts_31;
    (&kernelContext_133)->crush_0 = crush_31;
    (&kernelContext_133)->crush_rw_0 = crush_rw_31;
    (&kernelContext_133)->crush_tab_0 = crush_tab_31;
    threadgroup array<uint, int(1024)> g_scan_31;
    (&kernelContext_133)->g_scan_0 = &g_scan_31;
    threadgroup array<float, int(64)> g_il_best_31;
    (&kernelContext_133)->g_il_best_0 = &g_il_best_31;
    uint i_15 = id_28.x;
    bool _S966;
    if(i_15 >= (params_31->bond_used_0))
    {
        _S966 = true;
    }
    else
    {
        _S966 = ((&kernelContext_133)->params_0->shot_count_0) == 0U;
    }
    if(_S966)
    {
        _S966 = true;
    }
    else
    {
        uint _S967 = w_alive_0(&kernelContext_133);
        _S966 = (*((&kernelContext_133)->work_0+(_S967 + i_15))) == 0U;
    }
    if(_S966)
    {
        return;
    }
    uint4 _S968 = uint4(*((&kernelContext_133)->bonds_0+i_15)) ;
    uint _S969 = body_of_0(_S968.x, &kernelContext_133);
    thread uint shot_8;
    bool _S970 = struck_0(_S969, &shot_8, &kernelContext_133);
    if(!_S970)
    {
        return;
    }
    uint _S971 = strike_at_0(_S969, &kernelContext_133);
    float4 _S972 = float4(*((&kernelContext_133)->frames_0+(_S971 + 2U))) ;
    bool _S973 = strike_share_breaks_0(_S968, _S972.xyz, _S972.w, &kernelContext_133);
    if(_S973)
    {
        uint _S974 = w_alive_0(&kernelContext_133);
        *((&kernelContext_133)->work_0+(_S974 + i_15)) = 0U;
        uint _S975 = a_dirty_0(&kernelContext_133);
        uint _S976 = atomic_fetch_or_explicit((&kernelContext_133)->atom_0+(_S975 + _S969), 1U, memory_order_relaxed);
        uint _S977 = a_counters_1(&kernelContext_133);
        uint _S978 = atomic_fetch_add_explicit((&kernelContext_133)->atom_0+_S977, 1U, memory_order_relaxed);
        uint _S979 = atomic_fetch_add_explicit((&kernelContext_133)->atom_0+(_S977 + 1U), 1U, memory_order_relaxed);
        event_broken_0(i_15, &kernelContext_133);
    }
    return;
}

uint a_lp_0(uint b_42, KernelContext_0 thread* kernelContext_134)
{
    uint _S980 = a_strike_0(kernelContext_134);
    return _S980 + 9U * kernelContext_134->params_0->body_count_0 + 14U * b_42;
}

[[kernel]] void fr_lp_clear(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], atomic<uint> device* atom_32 [[buffer(5)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_32 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], packed_float4 device* lp_32 [[buffer(24)]], packed_float4 device* ext_34 [[buffer(23)]], packed_float4 device* engine_contacts_32 [[buffer(25)]], CrushParams_0 constant* crush_32 [[buffer(28)]], packed_float4 device* crush_rw_32 [[buffer(27)]], packed_float4 device* crush_tab_32 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_135;
    (&kernelContext_135)->params_0 = params_32;
    (&kernelContext_135)->atom_0 = atom_32;
    (&kernelContext_135)->work_0 = work_32;
    (&kernelContext_135)->frames_0 = frames_32;
    (&kernelContext_135)->bodies_0 = bodies_32;
    (&kernelContext_135)->chunks_0 = chunks_32;
    (&kernelContext_135)->sort_0 = sort_32;
    (&kernelContext_135)->manifolds_0 = manifolds_32;
    (&kernelContext_135)->loads_0 = loads_32;
    (&kernelContext_135)->bonds_0 = bonds_32;
    (&kernelContext_135)->args_0 = args_32;
    (&kernelContext_135)->info_0 = info_32;
    (&kernelContext_135)->scan_0 = scan_32;
    (&kernelContext_135)->parts_0 = parts_32;
    (&kernelContext_135)->colliders_0 = colliders_32;
    (&kernelContext_135)->collider_info_0 = collider_info_32;
    (&kernelContext_135)->view_chunks_0 = view_chunks_32;
    (&kernelContext_135)->view_0 = view_32;
    (&kernelContext_135)->objects_0 = objects_32;
    (&kernelContext_135)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_135)->lines_0 = lines_32;
    (&kernelContext_135)->geometry_0 = geometry_32;
    (&kernelContext_135)->shots_0 = shots_32;
    (&kernelContext_135)->lp_0 = lp_32;
    (&kernelContext_135)->ext_0 = ext_34;
    (&kernelContext_135)->engine_contacts_0 = engine_contacts_32;
    (&kernelContext_135)->crush_0 = crush_32;
    (&kernelContext_135)->crush_rw_0 = crush_rw_32;
    (&kernelContext_135)->crush_tab_0 = crush_tab_32;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_135)->g_scan_0 = &g_scan_32;
    threadgroup array<float, int(64)> g_il_best_32;
    (&kernelContext_135)->g_il_best_0 = &g_il_best_32;
    uint b_43 = id_29.x;
    if(b_43 >= (params_32->body_count_0))
    {
        return;
    }
    uint _S981 = a_lp_0(b_43, &kernelContext_135);
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
        atomic_store_explicit((&kernelContext_135)->atom_0+(_S981 + w_7), 0U, memory_order_relaxed);
        w_7 = w_7 + 1U;
    }
    atomic_store_explicit((&kernelContext_135)->atom_0+(_S981 + 12U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_135)->atom_0+(_S981 + 13U), 0U, memory_order_relaxed);
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

void add64w_0(uint at_18, float value_3, float scale_3, KernelContext_0 thread* kernelContext_136)
{
    float x_6 = round(value_3 * scale_3);
    float x_7;
    if(!((abs(x_6)) < 3.99999993722699776e+18f))
    {
        uint _S982 = a_counters_1(kernelContext_136);
        uint _S983 = atomic_fetch_or_explicit(kernelContext_136->atom_0+(_S982 + 3U), 2U, memory_order_relaxed);
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
        uint _S984 = ~hi_4;
        if(lo_6 == 0U)
        {
            lo_5 = 1U;
        }
        else
        {
            lo_5 = 0U;
        }
        uint _S985 = _S984 + lo_5;
        lo_5 = lo_6;
        hi_5 = _S985;
    }
    else
    {
        lo_5 = lo_4;
        hi_5 = hi_4;
    }
    uint old_1 = atomic_fetch_add_explicit(kernelContext_136->atom_0+at_18, lo_5, memory_order_relaxed);
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
        uint _S986 = atomic_fetch_add_explicit(kernelContext_136->atom_0+(at_18 + 1U), hi_6, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_lp_sum(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], atomic<uint> device* atom_33 [[buffer(5)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], packed_float4 device* lp_33 [[buffer(24)]], packed_float4 device* ext_35 [[buffer(23)]], packed_float4 device* engine_contacts_33 [[buffer(25)]], CrushParams_0 constant* crush_33 [[buffer(28)]], packed_float4 device* crush_rw_33 [[buffer(27)]], packed_float4 device* crush_tab_33 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_137;
    (&kernelContext_137)->params_0 = params_33;
    (&kernelContext_137)->atom_0 = atom_33;
    (&kernelContext_137)->work_0 = work_33;
    (&kernelContext_137)->frames_0 = frames_33;
    (&kernelContext_137)->bodies_0 = bodies_33;
    (&kernelContext_137)->chunks_0 = chunks_33;
    (&kernelContext_137)->sort_0 = sort_33;
    (&kernelContext_137)->manifolds_0 = manifolds_33;
    (&kernelContext_137)->loads_0 = loads_33;
    (&kernelContext_137)->bonds_0 = bonds_33;
    (&kernelContext_137)->args_0 = args_33;
    (&kernelContext_137)->info_0 = info_33;
    (&kernelContext_137)->scan_0 = scan_33;
    (&kernelContext_137)->parts_0 = parts_33;
    (&kernelContext_137)->colliders_0 = colliders_33;
    (&kernelContext_137)->collider_info_0 = collider_info_33;
    (&kernelContext_137)->view_chunks_0 = view_chunks_33;
    (&kernelContext_137)->view_0 = view_33;
    (&kernelContext_137)->objects_0 = objects_33;
    (&kernelContext_137)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_137)->lines_0 = lines_33;
    (&kernelContext_137)->geometry_0 = geometry_33;
    (&kernelContext_137)->shots_0 = shots_33;
    (&kernelContext_137)->lp_0 = lp_33;
    (&kernelContext_137)->ext_0 = ext_35;
    (&kernelContext_137)->engine_contacts_0 = engine_contacts_33;
    (&kernelContext_137)->crush_0 = crush_33;
    (&kernelContext_137)->crush_rw_0 = crush_rw_33;
    (&kernelContext_137)->crush_tab_0 = crush_tab_33;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_137)->g_scan_0 = &g_scan_33;
    threadgroup array<float, int(64)> g_il_best_33;
    (&kernelContext_137)->g_il_best_0 = &g_il_best_33;
    uint k_34 = id_30.x;
    if(k_34 >= (params_33->chunk_count_0))
    {
        return;
    }
    uint _S987 = body_of_0(k_34, &kernelContext_137);
    thread Quat_0 q_25;
    thread float3 origin_7;
    structure_pose_0(_S987, 7U * _S987, &q_25, &origin_7, &kernelContext_137);
    thread Quat_0 _S988 = q_25;
    Quat_0 _S989 = conj_0(&_S988);
    float4 _S990 = float4(*((&kernelContext_137)->chunks_0+4U * k_34)) ;
    uint _S991 = l_vec_0(&kernelContext_137);
    uint _S992 = _S991 + 2U * k_34;
    float3 _S993 = (float4(*((&kernelContext_137)->loads_0+_S992)) ).xyz;
    thread Quat_0 _S994 = _S989;
    float3 _S995 = rotate_0(&_S994, _S993);
    float _S996 = _S990.w;
    float3 _S997 = float3((&kernelContext_137)->params_0->gx_0, (&kernelContext_137)->params_0->gy_0, (&kernelContext_137)->params_0->gz_0);
    thread Quat_0 _S998 = _S989;
    float3 _S999 = rotate_0(&_S998, _S997);
    float3 f_10 = _S995 + float3(_S996)  * _S999;
    float3 _S1000 = (float4(*((&kernelContext_137)->loads_0+(_S992 + 1U))) ).xyz;
    thread Quat_0 _S1001 = _S989;
    float3 _S1002 = rotate_0(&_S1001, _S1000);
    uint _S1003 = lp_f_0(k_34);
    *((&kernelContext_137)->lp_0+_S1003) = packed_float4(float4(f_10, 0.0f)) ;
    *((&kernelContext_137)->lp_0+(_S1003 + 1U)) = packed_float4(float4(_S1002, 0.0f)) ;
    float3 _S1004 = _S1002 + cross(_S990.xyz - (float4(*((&kernelContext_137)->frames_0+2U * _S987)) ).xyz, f_10);
    uint _S1005 = a_lp_0(_S987, &kernelContext_137);
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
        uint _S1006 = 2U * a_8;
        add64w_0(_S1005 + _S1006, f_10[a_8], (&kernelContext_137)->params_0->lp_scale_0, &kernelContext_137);
        add64w_0(_S1005 + 6U + _S1006, _S1004[a_8], (&kernelContext_137)->params_0->lp_scale_0, &kernelContext_137);
        a_8 = a_8 + 1U;
    }
    uint _S1007 = atomic_fetch_min_explicit((&kernelContext_137)->atom_0+(_S1005 + 12U), k_34, memory_order_relaxed);
    if(((&kernelContext_137)->info_0[k_34] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_137)->atom_0+(_S1005 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

float3 read64w3_0(uint at_19, float scale_4, KernelContext_0 thread* kernelContext_138)
{
    float _S1008 = read64_1(at_19, scale_4, kernelContext_138);
    float _S1009 = read64_1(at_19 + 2U, scale_4, kernelContext_138);
    float _S1010 = read64_1(at_19 + 4U, scale_4, kernelContext_138);
    return float3(_S1008, _S1009, _S1010);
}

[[kernel]] void fr_lp_relief(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], atomic<uint> device* atom_34 [[buffer(5)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], packed_float4 device* lp_34 [[buffer(24)]], packed_float4 device* ext_36 [[buffer(23)]], packed_float4 device* engine_contacts_34 [[buffer(25)]], CrushParams_0 constant* crush_34 [[buffer(28)]], packed_float4 device* crush_rw_34 [[buffer(27)]], packed_float4 device* crush_tab_34 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_139;
    (&kernelContext_139)->params_0 = params_34;
    (&kernelContext_139)->atom_0 = atom_34;
    (&kernelContext_139)->work_0 = work_34;
    (&kernelContext_139)->frames_0 = frames_34;
    (&kernelContext_139)->bodies_0 = bodies_34;
    (&kernelContext_139)->chunks_0 = chunks_34;
    (&kernelContext_139)->sort_0 = sort_34;
    (&kernelContext_139)->manifolds_0 = manifolds_34;
    (&kernelContext_139)->loads_0 = loads_34;
    (&kernelContext_139)->bonds_0 = bonds_34;
    (&kernelContext_139)->args_0 = args_34;
    (&kernelContext_139)->info_0 = info_34;
    (&kernelContext_139)->scan_0 = scan_34;
    (&kernelContext_139)->parts_0 = parts_34;
    (&kernelContext_139)->colliders_0 = colliders_34;
    (&kernelContext_139)->collider_info_0 = collider_info_34;
    (&kernelContext_139)->view_chunks_0 = view_chunks_34;
    (&kernelContext_139)->view_0 = view_34;
    (&kernelContext_139)->objects_0 = objects_34;
    (&kernelContext_139)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_139)->lines_0 = lines_34;
    (&kernelContext_139)->geometry_0 = geometry_34;
    (&kernelContext_139)->shots_0 = shots_34;
    (&kernelContext_139)->lp_0 = lp_34;
    (&kernelContext_139)->ext_0 = ext_36;
    (&kernelContext_139)->engine_contacts_0 = engine_contacts_34;
    (&kernelContext_139)->crush_0 = crush_34;
    (&kernelContext_139)->crush_rw_0 = crush_rw_34;
    (&kernelContext_139)->crush_tab_0 = crush_tab_34;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_139)->g_scan_0 = &g_scan_34;
    threadgroup array<float, int(64)> g_il_best_34;
    (&kernelContext_139)->g_il_best_0 = &g_il_best_34;
    uint k_35 = id_31.x;
    if(k_35 >= (params_34->chunk_count_0))
    {
        return;
    }
    uint _S1011 = body_of_0(k_35, &kernelContext_139);
    uint _S1012 = a_lp_0(_S1011, &kernelContext_139);
    uint _S1013 = atomic_load_explicit((&kernelContext_139)->atom_0+(_S1012 + 13U), memory_order_relaxed);
    bool supported_0 = _S1013 != 0U;
    bool fixed_1;
    if(((&kernelContext_139)->info_0[k_35] & 1U) != 0U)
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            uint _S1014 = atomic_load_explicit((&kernelContext_139)->atom_0+(_S1012 + 12U), memory_order_relaxed);
            fixed_1 = _S1014 == k_35;
        }
        else
        {
            fixed_1 = false;
        }
    }
    uint _S1015 = lp_f_0(k_35);
    thread float4 f_11 = float4(*((&kernelContext_139)->lp_0+_S1015)) ;
    uint _S1016 = _S1015 + 1U;
    thread float4 t_11 = float4(*((&kernelContext_139)->lp_0+_S1016)) ;
    uint b_44 = 7U * _S1011;
    float inv_mass_6 = (float4(*((&kernelContext_139)->bodies_0+b_44)) ).w;
    bool _S1017;
    if(!supported_0)
    {
        _S1017 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S1017 = false;
    }
    if(_S1017)
    {
        float3 _S1018 = read64w3_0(_S1012, (&kernelContext_139)->params_0->lp_scale_0, &kernelContext_139);
        float3 _S1019 = read64w3_0(_S1012 + 6U, (&kernelContext_139)->params_0->lp_scale_0, &kernelContext_139);
        uint _S1020 = 2U * _S1011;
        Quat_0 p_15 = quat_of_0(float4(*((&kernelContext_139)->frames_0+(_S1020 + 1U))) );
        float3 _S1021 = (float4(*((&kernelContext_139)->bodies_0+(b_44 + 2U))) ).xyz;
        thread Quat_0 _S1022 = p_15;
        Quat_0 _S1023 = conj_0(&_S1022);
        thread Quat_0 _S1024 = _S1023;
        float3 _S1025 = rotate_0(&_S1024, _S1019);
        float3 _S1026 = _S1021 * _S1025;
        thread Quat_0 _S1027 = p_15;
        float3 _S1028 = rotate_0(&_S1027, _S1026);
        uint _S1029 = 4U * k_35;
        float4 _S1030 = float4(*((&kernelContext_139)->chunks_0+_S1029)) ;
        float4 _S1031 = float4(*((&kernelContext_139)->chunks_0+(_S1029 + 1U))) ;
        float4 _S1032 = float4(*((&kernelContext_139)->chunks_0+(_S1029 + 2U))) ;
        f_11.xyz = f_11.xyz - float3(_S1030.w)  * (_S1018 * float3(inv_mass_6)  + cross(_S1028, _S1030.xyz - (float4(*((&kernelContext_139)->frames_0+_S1020)) ).xyz));
        float _S1033 = _S1028.x;
        float _S1034 = _S1031.w;
        float _S1035 = _S1028.y;
        float _S1036 = _S1032.x;
        float _S1037 = _S1028.z;
        float _S1038 = _S1032.y;
        t_11.xyz = t_11.xyz - float3(_S1031.x * _S1033 + _S1034 * _S1035 + _S1036 * _S1037, _S1034 * _S1033 + _S1031.y * _S1035 + _S1038 * _S1037, _S1036 * _S1033 + _S1038 * _S1035 + _S1031.z * _S1037);
    }
    float _S1039;
    if(fixed_1)
    {
        _S1039 = 1.0f;
    }
    else
    {
        _S1039 = 0.0f;
    }
    f_11.w = _S1039;
    *((&kernelContext_139)->lp_0+_S1015) = packed_float4(f_11) ;
    *((&kernelContext_139)->lp_0+_S1016) = packed_float4(t_11) ;
    return;
}

uint adj_bond_0(uint at_20, KernelContext_0 thread* kernelContext_140)
{
    return (as_type<uint>(((float4(*(kernelContext_140->ext_0+(kernelContext_140->params_0->ext_adj_0 + at_20 / 4U))) )[at_20 % 4U])));
}

uint lp_intact_0(uint k_36, KernelContext_0 thread* kernelContext_141)
{
    float4 _S1040 = float4(*(kernelContext_141->ext_0+(kernelContext_141->params_0->ext_chunks_0 + 4U * k_36))) ;
    uint _S1041 = (as_type<uint>((_S1040.x)));
    uint _S1042 = (as_type<uint>((_S1040.y)));
    uint e_4 = 0U;
    uint intact_0 = 0U;
    for(;;)
    {
        if(e_4 < _S1042)
        {
        }
        else
        {
            break;
        }
        uint _S1043 = w_alive_0(kernelContext_141);
        uint _S1044 = adj_bond_0(_S1041 + e_4, kernelContext_141);
        uint _S1045;
        if((*(kernelContext_141->work_0+(_S1043 + _S1044))) != 0U)
        {
            _S1045 = 1U;
        }
        else
        {
            _S1045 = 0U;
        }
        uint intact_1 = intact_0 + _S1045;
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

LpBond_0 lp_bond_0(uint i_16, KernelContext_0 thread* kernelContext_142)
{
    uint o_7 = 10U * i_16;
    thread LpBond_0 b_45;
    (&b_45)->centroid_0 = (float4(*(kernelContext_142->ext_0+o_7)) ).xyz;
    float3 _S1046 = (float4(*(kernelContext_142->ext_0+(o_7 + 1U))) ).xyz;
    (&b_45)->n_4 = _S1046;
    float3 _S1047 = (float4(*(kernelContext_142->ext_0+(o_7 + 2U))) ).xyz;
    (&b_45)->t1_2 = _S1047;
    (&b_45)->t2_0 = cross(_S1046, _S1047);
    (&(&b_45)->law_0)->geom0_0 = float4(*(kernelContext_142->ext_0+(o_7 + 3U))) ;
    (&(&b_45)->law_0)->geom1_0 = float4(*(kernelContext_142->ext_0+(o_7 + 4U))) ;
    (&(&b_45)->law_0)->stiff0_0 = float4(*(kernelContext_142->ext_0+(o_7 + 5U))) ;
    (&(&b_45)->law_0)->stiff1_0 = float4(*(kernelContext_142->ext_0+(o_7 + 6U))) ;
    (&(&b_45)->law_0)->rebar0_0 = float4(*(kernelContext_142->ext_0+(o_7 + 7U))) ;
    (&(&b_45)->law_0)->rebar1_0 = float4(*(kernelContext_142->ext_0+(o_7 + 8U))) ;
    (&(&b_45)->law_0)->ids_3 = (as_type<uint4>((float4(*(kernelContext_142->ext_0+(o_7 + 9U))) )));
    return b_45;
}

uint tri_0(uint i_17, uint j_6)
{
    return i_17 * (i_17 + 1U) / 2U + j_6;
}

bool lp_factor_0(uint k_38, KernelContext_0 thread* kernelContext_143)
{
    uint q_26;
    uint a_9;
    uint p2_0;
    uint q2_0;
    uint a2_0;
    float s_11;
    uint _S1048 = lp_intact_0(k_38, kernelContext_143);
    uint _S1049 = lp_l_0(k_38);
    uint _S1050 = _S1048 + 1U;
    if((as_type<uint>(((float4(*(kernelContext_143->lp_0+(_S1049 + 5U))) ).w))) == _S1050)
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
    uint _S1051 = 4U * k_38;
    float4 _S1052 = float4(*(kernelContext_143->ext_0+(kernelContext_143->params_0->ext_chunks_0 + _S1051))) ;
    uint _S1053 = (as_type<uint>((_S1052.x)));
    uint _S1054 = (as_type<uint>((_S1052.y)));
    q_26 = 0U;
    for(;;)
    {
        if(q_26 < _S1054)
        {
        }
        else
        {
            break;
        }
        uint _S1055 = adj_bond_0(_S1053 + q_26, kernelContext_143);
        uint _S1056 = w_alive_0(kernelContext_143);
        if((*(kernelContext_143->work_0+(_S1056 + _S1055))) == 0U)
        {
            q_26 = q_26 + 1U;
            continue;
        }
        LpBond_0 _S1057 = lp_bond_0(_S1055, kernelContext_143);
        float3 _S1058 = _S1057.centroid_0 - (float4(*(kernelContext_143->chunks_0+_S1051)) ).xyz;
        float _S1059 = _S1057.law_0.stiff0_0.y;
        array<float, int(6)> _S1060 = { { _S1059, _S1059, _S1057.law_0.stiff0_0.x, _S1057.law_0.stiff0_0.z, _S1057.law_0.stiff0_0.w, _S1057.law_0.stiff1_0.x } };
        array<float3, int(3)> _S1061 = { { _S1057.t1_2, _S1057.t2_0, _S1057.n_4 } };
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
            float3 c_20 = cross(_S1058, _S1061[a_9]);
            float _S1062 = _S1061[a_9].x;
            j_7[a_9][int(0)] = _S1062;
            float _S1063 = _S1061[a_9].y;
            j_7[a_9][int(1)] = _S1063;
            float _S1064 = _S1061[a_9].z;
            j_7[a_9][int(2)] = _S1064;
            j_7[a_9][int(3)] = c_20.x;
            j_7[a_9][int(4)] = c_20.y;
            j_7[a_9][int(5)] = c_20.z;
            j_7[3U + a_9][int(0)] = 0.0f;
            j_7[3U + a_9][int(1)] = 0.0f;
            j_7[3U + a_9][int(2)] = 0.0f;
            j_7[3U + a_9][int(3)] = _S1062;
            j_7[3U + a_9][int(4)] = _S1063;
            j_7[3U + a_9][int(5)] = _S1064;
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
                    float s_12 = s_11 + j_7[a2_0][p2_0] * _S1060[a2_0] * j_7[a2_0][q2_0];
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
                bool _S1065;
                if(ok_1)
                {
                    _S1065 = s_11 > 0.0f;
                }
                else
                {
                    _S1065 = false;
                }
                l_0[tri_0(p2_0, p2_0)] = sqrt(max(s_11, 1.00000000317107685e-30f));
                ok_1 = _S1065;
            }
            else
            {
                l_0[tri_0(p2_0, q2_0)] = s_11 / l_0[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + 1U;
        }
        uint _S1066 = p2_0 + 1U;
        ok_0 = ok_1;
        p2_0 = _S1066;
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
        packed_float4 device* _S1067 = kernelContext_143->lp_0+(_S1049 + q2_0);
        uint _S1068 = 4U * q2_0;
        float _S1069 = l_0[_S1068];
        uint _S1070 = _S1068 + 1U;
        if(_S1070 < 21U)
        {
            s_11 = l_0[_S1070];
        }
        else
        {
            s_11 = 0.0f;
        }
        uint _S1071 = _S1068 + 2U;
        float _S1072;
        if(_S1071 < 21U)
        {
            _S1072 = l_0[_S1071];
        }
        else
        {
            _S1072 = 0.0f;
        }
        float _S1073;
        if(q2_0 == 5U)
        {
            _S1073 = (as_type<float>((_S1050)));
        }
        else
        {
            _S1073 = l_0[_S1068 + 3U];
        }
        *_S1067 = packed_float4(float4(_S1069, s_11, _S1072, _S1073)) ;
        q2_0 = q2_0 + 1U;
    }
    return true;
}

[[kernel]] void fr_lp_block(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], atomic<uint> device* atom_35 [[buffer(5)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], packed_float4 device* lp_35 [[buffer(24)]], packed_float4 device* ext_37 [[buffer(23)]], packed_float4 device* engine_contacts_35 [[buffer(25)]], CrushParams_0 constant* crush_35 [[buffer(28)]], packed_float4 device* crush_rw_35 [[buffer(27)]], packed_float4 device* crush_tab_35 [[buffer(26)]])
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
    uint k_39 = id_32.x;
    bool _S1074;
    if(k_39 >= (params_35->chunk_count_0))
    {
        _S1074 = true;
    }
    else
    {
        _S1074 = ((float4(*((&kernelContext_144)->lp_0+lp_f_0(k_39))) ).w) != 0.0f;
    }
    if(_S1074)
    {
        return;
    }
    bool _S1075 = lp_factor_0(k_39, &kernelContext_144);
    if(!_S1075)
    {
        (*((&kernelContext_144)->lp_0+lp_f_0(k_39)))[int(3)] = 1.0f;
        uint _S1076 = a_counters_1(&kernelContext_144);
        uint _S1077 = atomic_fetch_or_explicit((&kernelContext_144)->atom_0+(_S1076 + 3U), 64U, memory_order_relaxed);
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

float4 lp_side_0(uint i_18, KernelContext_0 thread* kernelContext_145)
{
    return float4(*(kernelContext_145->ext_0+(kernelContext_145->params_0->ext_sides_0 + i_18))) ;
}

void st_sweep_0(uint k_42, uint state_3, uint from_2, float c1_1, float c2_1, KernelContext_0 thread* kernelContext_146)
{
    uint ka_0;
    uint kb_0;
    float s_14;
    uint to_1 = 1U - from_2;
    uint _S1078 = st_x_0(k_42, state_3, from_2);
    float4 _S1079 = float4(*(kernelContext_146->lp_0+_S1078)) ;
    float4 _S1080 = float4(*(kernelContext_146->lp_0+(_S1078 + 1U))) ;
    uint _S1081 = st_f_0(k_42, state_3);
    float4 _S1082 = float4(*(kernelContext_146->lp_0+_S1081)) ;
    if((_S1082.w) != 0.0f)
    {
        uint _S1083 = st_x_0(k_42, state_3, to_1);
        *(kernelContext_146->lp_0+_S1083) = packed_float4(_S1079) ;
        *(kernelContext_146->lp_0+(_S1083 + 1U)) = packed_float4(_S1080) ;
        uint _S1084 = st_d_0(k_42, state_3);
        packed_float4 _S1085 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_146->lp_0+_S1084) = _S1085;
        *(kernelContext_146->lp_0+(_S1084 + 1U)) = _S1085;
        return;
    }
    float _S1086 = _S1082.y;
    float _S1087 = _S1082.z;
    thread array<float, int(6)> r_23;
    r_23[int(0)] = _S1082.x;
    r_23[int(1)] = _S1086;
    r_23[int(2)] = _S1087;
    r_23[int(3)] = 0.0f;
    r_23[int(4)] = 0.0f;
    r_23[int(5)] = 0.0f;
    float3 t_12 = (float4(*(kernelContext_146->lp_0+(_S1081 + 1U))) ).xyz;
    r_23[int(3)] = t_12.x;
    r_23[int(4)] = t_12.y;
    r_23[int(5)] = t_12.z;
    float4 _S1088 = float4(*(kernelContext_146->ext_0+(kernelContext_146->params_0->ext_chunks_0 + 4U * k_42))) ;
    uint _S1089 = (as_type<uint>((_S1088.x)));
    uint _S1090 = (as_type<uint>((_S1088.y)));
    uint e_5 = 0U;
    for(;;)
    {
        if(e_5 < _S1090)
        {
        }
        else
        {
            break;
        }
        uint _S1091 = 5U * (_S1089 + e_5);
        float4 _S1092 = lp_side_0(_S1091, kernelContext_146);
        uint _S1093 = w_alive_0(kernelContext_146);
        if((*(kernelContext_146->work_0+(_S1093 + (as_type<uint>((_S1092.w)))))) == 0U)
        {
            e_5 = e_5 + 1U;
            continue;
        }
        float4 _S1094 = lp_side_0(_S1091 + 1U, kernelContext_146);
        float4 _S1095 = lp_side_0(_S1091 + 2U, kernelContext_146);
        float4 _S1096 = lp_side_0(_S1091 + 3U, kernelContext_146);
        float4 _S1097 = lp_side_0(_S1091 + 4U, kernelContext_146);
        bool first_side_0 = (_S1097.w) != 0.0f;
        if(first_side_0)
        {
            ka_0 = k_42;
        }
        else
        {
            ka_0 = (as_type<uint>((_S1094.w)));
        }
        if(first_side_0)
        {
            kb_0 = (as_type<uint>((_S1094.w)));
        }
        else
        {
            kb_0 = k_42;
        }
        float3 ra_0 = _S1092.xyz;
        float3 rb_0 = _S1094.xyz;
        float3 t1_3 = _S1095.xyz;
        float3 n_5 = _S1096.xyz;
        float3 t2_1 = cross(n_5, t1_3);
        uint _S1098 = st_x_0(ka_0, state_3, from_2);
        uint _S1099 = st_x_0(kb_0, state_3, from_2);
        float3 _S1100 = (float4(*(kernelContext_146->lp_0+(_S1099 + 1U))) ).xyz;
        float3 _S1101 = (float4(*(kernelContext_146->lp_0+(_S1098 + 1U))) ).xyz;
        float3 delta_0 = (float4(*(kernelContext_146->lp_0+_S1099)) ).xyz + cross(_S1100, rb_0) - ((float4(*(kernelContext_146->lp_0+_S1098)) ).xyz + cross(_S1101, ra_0));
        float3 phi_0 = _S1100 - _S1101;
        float _S1102 = _S1095.w;
        float3 fb_0 = t1_3 * float3((_S1102 * dot(delta_0, t1_3)))  + t2_1 * float3((_S1102 * dot(delta_0, t2_1)))  + n_5 * float3((_S1096.w * dot(delta_0, n_5))) ;
        float3 mb_0 = t1_3 * float3((_S1097.x * dot(phi_0, t1_3)))  + t2_1 * float3((_S1097.y * dot(phi_0, t2_1)))  + n_5 * float3((_S1097.z * dot(phi_0, n_5))) ;
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
        float3 _S1103;
        if(first_side_0)
        {
            _S1103 = mb_0;
        }
        else
        {
            _S1103 = - mb_0;
        }
        float3 lm_0 = _S1103 + cross(arm_0, lf_0);
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
        float4 _S1104 = float4(*(kernelContext_146->lp_0+(lp_l_0(k_42) + kb_0))) ;
        uint _S1105 = 4U * kb_0;
        l_1[_S1105] = _S1104.x;
        uint _S1106 = _S1105 + 1U;
        if(_S1106 < 21U)
        {
            l_1[_S1106] = _S1104.y;
        }
        uint _S1107 = _S1105 + 2U;
        if(_S1107 < 21U)
        {
            l_1[_S1107] = _S1104.z;
        }
        uint _S1108 = _S1105 + 3U;
        if(_S1108 < 21U)
        {
            l_1[_S1108] = _S1104.w;
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
        uint _S1109 = uint(i3_0);
        z_3[i3_0] = s_14 / l_1[tri_0(_S1109, _S1109)];
        i3_0 = i3_0 - int(1);
    }
    uint _S1110 = st_d_0(k_42, state_3);
    float4 _S1111 = float4(*(kernelContext_146->lp_0+_S1110)) ;
    thread float4 d0_0 = _S1111;
    uint _S1112 = _S1110 + 1U;
    float4 _S1113 = float4(*(kernelContext_146->lp_0+_S1112)) ;
    thread float4 d1_0 = _S1113;
    float3 _S1114 = float3(c1_1) ;
    float3 _S1115 = float3(c2_1) ;
    d0_0.xyz = _S1114 * _S1111.xyz + _S1115 * float3(z_3[int(0)], z_3[int(1)], z_3[int(2)]);
    d1_0.xyz = _S1114 * _S1113.xyz + _S1115 * float3(z_3[int(3)], z_3[int(4)], z_3[int(5)]);
    *(kernelContext_146->lp_0+_S1110) = packed_float4(d0_0) ;
    *(kernelContext_146->lp_0+_S1112) = packed_float4(d1_0) ;
    uint _S1116 = st_x_0(k_42, state_3, to_1);
    *(kernelContext_146->lp_0+_S1116) = packed_float4(float4(_S1079.xyz + d0_0.xyz, 0.0f)) ;
    *(kernelContext_146->lp_0+(_S1116 + 1U)) = packed_float4(float4(_S1080.xyz + d1_0.xyz, 0.0f)) ;
    return;
}

void lp_sweep_0(uint k_43, uint from_3, KernelContext_0 thread* kernelContext_147)
{
    st_sweep_0(k_43, 0U, from_3, kernelContext_147->params_0->lp_c1_0, kernelContext_147->params_0->lp_c2_0, kernelContext_147);
    return;
}

[[kernel]] void fr_lp_sweep_even(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], atomic<uint> device* atom_36 [[buffer(5)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], packed_float4 device* lp_36 [[buffer(24)]], packed_float4 device* ext_38 [[buffer(23)]], packed_float4 device* engine_contacts_36 [[buffer(25)]], CrushParams_0 constant* crush_36 [[buffer(28)]], packed_float4 device* crush_rw_36 [[buffer(27)]], packed_float4 device* crush_tab_36 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_148;
    (&kernelContext_148)->params_0 = params_36;
    (&kernelContext_148)->atom_0 = atom_36;
    (&kernelContext_148)->work_0 = work_36;
    (&kernelContext_148)->frames_0 = frames_36;
    (&kernelContext_148)->bodies_0 = bodies_36;
    (&kernelContext_148)->chunks_0 = chunks_36;
    (&kernelContext_148)->sort_0 = sort_36;
    (&kernelContext_148)->manifolds_0 = manifolds_36;
    (&kernelContext_148)->loads_0 = loads_36;
    (&kernelContext_148)->bonds_0 = bonds_36;
    (&kernelContext_148)->args_0 = args_36;
    (&kernelContext_148)->info_0 = info_36;
    (&kernelContext_148)->scan_0 = scan_36;
    (&kernelContext_148)->parts_0 = parts_36;
    (&kernelContext_148)->colliders_0 = colliders_36;
    (&kernelContext_148)->collider_info_0 = collider_info_36;
    (&kernelContext_148)->view_chunks_0 = view_chunks_36;
    (&kernelContext_148)->view_0 = view_36;
    (&kernelContext_148)->objects_0 = objects_36;
    (&kernelContext_148)->shot_bodies_0 = shot_bodies_36;
    (&kernelContext_148)->lines_0 = lines_36;
    (&kernelContext_148)->geometry_0 = geometry_36;
    (&kernelContext_148)->shots_0 = shots_36;
    (&kernelContext_148)->lp_0 = lp_36;
    (&kernelContext_148)->ext_0 = ext_38;
    (&kernelContext_148)->engine_contacts_0 = engine_contacts_36;
    (&kernelContext_148)->crush_0 = crush_36;
    (&kernelContext_148)->crush_rw_0 = crush_rw_36;
    (&kernelContext_148)->crush_tab_0 = crush_tab_36;
    threadgroup array<uint, int(1024)> g_scan_36;
    (&kernelContext_148)->g_scan_0 = &g_scan_36;
    threadgroup array<float, int(64)> g_il_best_36;
    (&kernelContext_148)->g_il_best_0 = &g_il_best_36;
    uint _S1117 = id_33.x;
    if(_S1117 < (params_36->chunk_count_0))
    {
        lp_sweep_0(_S1117, 0U, &kernelContext_148);
    }
    return;
}

[[kernel]] void fr_lp_sweep_odd(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], atomic<uint> device* atom_37 [[buffer(5)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], packed_float4 device* lp_37 [[buffer(24)]], packed_float4 device* ext_39 [[buffer(23)]], packed_float4 device* engine_contacts_37 [[buffer(25)]], CrushParams_0 constant* crush_37 [[buffer(28)]], packed_float4 device* crush_rw_37 [[buffer(27)]], packed_float4 device* crush_tab_37 [[buffer(26)]])
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
    uint _S1118 = id_34.x;
    if(_S1118 < (params_37->chunk_count_0))
    {
        lp_sweep_0(_S1118, 1U, &kernelContext_149);
    }
    return;
}

void st_force_0(const LpBond_0 thread* b_46, uint ka_1, uint kb_1, uint state_4, uint c_21, float3 thread* q_lin_0, float3 thread* q_ang_0, KernelContext_0 thread* kernelContext_150)
{
    uint _S1119 = st_x_0(ka_1, state_4, c_21);
    uint _S1120 = st_x_0(kb_1, state_4, c_21);
    float3 _S1121 = (float4(*(kernelContext_150->lp_0+(_S1120 + 1U))) ).xyz;
    float3 _S1122 = (float4(*(kernelContext_150->lp_0+(_S1119 + 1U))) ).xyz;
    float3 delta_1 = (float4(*(kernelContext_150->lp_0+_S1120)) ).xyz + cross(_S1121, b_46->centroid_0 - (float4(*(kernelContext_150->chunks_0+4U * kb_1)) ).xyz) - ((float4(*(kernelContext_150->lp_0+_S1119)) ).xyz + cross(_S1122, b_46->centroid_0 - (float4(*(kernelContext_150->chunks_0+4U * ka_1)) ).xyz));
    float3 phi_1 = _S1121 - _S1122;
    float4 _S1123 = (&b_46->law_0)->stiff0_0;
    float kt_0 = (&b_46->law_0)->stiff1_0.x;
    float _S1124 = (&b_46->law_0)->stiff0_0.y;
    float3 _S1125 = b_46->t1_2;
    float3 _S1126 = b_46->t2_0;
    float3 _S1127 = b_46->n_4;
    *q_lin_0 = float3(_S1124 * dot(delta_1, b_46->t1_2), _S1124 * dot(delta_1, b_46->t2_0), (&b_46->law_0)->stiff0_0.x * dot(delta_1, b_46->n_4));
    *q_ang_0 = float3(_S1123.z * dot(phi_1, _S1125), _S1123.w * dot(phi_1, _S1126), kt_0 * dot(phi_1, _S1127));
    return;
}

void st_force_1(const LpBond_0 thread* b_47, uint ka_2, uint kb_2, uint state_5, uint c_22, float3 thread* q_lin_1, float3 thread* q_ang_1, KernelContext_0 thread* kernelContext_151)
{
    uint _S1128 = st_x_0(ka_2, state_5, c_22);
    uint _S1129 = st_x_0(kb_2, state_5, c_22);
    float3 _S1130 = (float4(*(kernelContext_151->lp_0+(_S1129 + 1U))) ).xyz;
    float3 _S1131 = (float4(*(kernelContext_151->lp_0+(_S1128 + 1U))) ).xyz;
    float3 delta_2 = (float4(*(kernelContext_151->lp_0+_S1129)) ).xyz + cross(_S1130, b_47->centroid_0 - (float4(*(kernelContext_151->chunks_0+4U * kb_2)) ).xyz) - ((float4(*(kernelContext_151->lp_0+_S1128)) ).xyz + cross(_S1131, b_47->centroid_0 - (float4(*(kernelContext_151->chunks_0+4U * ka_2)) ).xyz));
    float3 phi_2 = _S1130 - _S1131;
    float4 _S1132 = (&b_47->law_0)->stiff0_0;
    float kt_1 = (&b_47->law_0)->stiff1_0.x;
    float _S1133 = (&b_47->law_0)->stiff0_0.y;
    float3 _S1134 = b_47->t1_2;
    float3 _S1135 = b_47->t2_0;
    float3 _S1136 = b_47->n_4;
    *q_lin_1 = float3(_S1133 * dot(delta_2, b_47->t1_2), _S1133 * dot(delta_2, b_47->t2_0), (&b_47->law_0)->stiff0_0.x * dot(delta_2, b_47->n_4));
    *q_ang_1 = float3(_S1132.z * dot(phi_2, _S1134), _S1132.w * dot(phi_2, _S1135), kt_1 * dot(phi_2, _S1136));
    return;
}

void lp_force_0(const LpBond_0 thread* b_48, uint ka_3, uint kb_3, uint c_23, float3 thread* q_lin_2, float3 thread* q_ang_2, KernelContext_0 thread* kernelContext_152)
{
    st_force_1(b_48, ka_3, kb_3, 0U, c_23, q_lin_2, q_ang_2, kernelContext_152);
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
    float _S1137 = q_lin_3.z;
    float axial_0 = (metal::fast::divide((_S1137), (area_0)));
    float _S1138 = (metal::fast::divide((abs(q_ang_3.x)), (b_49->geom1_0.x)));
    float _S1139 = (metal::fast::divide((abs(q_ang_3.y)), (b_49->geom1_0.y)));
    float bending_0 = _S1138 + _S1139;
    float _S1140 = q_lin_3.x;
    float _S1141 = q_lin_3.y;
    float _S1142 = (metal::fast::sqrt((_S1140 * _S1140 + _S1141 * _S1141)));
    float _S1143 = (metal::fast::divide((_S1142), (area_0)));
    float _S1144 = (metal::fast::divide((abs(q_ang_3.z)), (b_49->geom0_0.w)));
    float shear_1 = _S1143 + _S1144;
    thread Measures_0 m_3;
    (&m_3)->tension_0 = axial_0 + bending_0;
    (&m_3)->shear_0 = shear_1;
    float _S1145 = - axial_0;
    (&m_3)->normal_compression_0 = max(_S1145, 0.0f);
    (&m_3)->compression_0 = _S1145 + bending_0;
    (&m_3)->compressive_force_0 = max(- _S1137, 0.0f);
    return m_3;
}

JointMaterial_0 lp_material_0(uint m_4, KernelContext_0 thread* kernelContext_153)
{
    uint o_8 = kernelContext_153->params_0->ext_mat_0 + 5U * m_4;
    thread JointMaterial_0 j_8;
    (&j_8)->strength_0 = float4(*(kernelContext_153->ext_0+o_8)) ;
    (&j_8)->energy_0 = float4(*(kernelContext_153->ext_0+(o_8 + 1U))) ;
    (&j_8)->dif_0 = float4(*(kernelContext_153->ext_0+(o_8 + 2U))) ;
    (&j_8)->misc_0 = float4(*(kernelContext_153->ext_0+(o_8 + 3U))) ;
    (&j_8)->kind_flags_0 = (as_type<uint4>((float4(*(kernelContext_153->ext_0+(o_8 + 4U))) )));
    return j_8;
}

float infinity_0()
{
    return float_bits_0(2139095040U);
}

float4 failure_indices_0(const JointMaterial_0 thread* mat_1, const JointBond_0 thread* b_50, const Measures_0 thread* m_5, float multiplier_0)
{
    float fc_2 = mat_1->strength_0.y * multiplier_0;
    float _S1146 = min(mat_1->strength_0.z * multiplier_0 + mat_1->strength_0.w * m_5->normal_compression_0, mat_1->energy_0.x * multiplier_0);
    thread float4 idx_0;
    float _S1147 = (metal::fast::divide((m_5->tension_0), (mat_1->strength_0.x * multiplier_0)));
    idx_0.x = max(_S1147, 0.0f);
    float _S1148;
    if(_S1146 > 0.0f)
    {
        float _S1149 = (metal::fast::divide((m_5->shear_0), (_S1146)));
        _S1148 = _S1149;
    }
    else
    {
        _S1148 = infinity_0();
    }
    idx_0.y = _S1148;
    float _S1150 = (metal::fast::divide((m_5->compression_0), (fc_2)));
    idx_0.z = max(_S1150, 0.0f);
    float _S1151 = b_50->stiff1_0.y;
    if(_S1151 > 0.0f)
    {
        float _S1152 = (metal::fast::divide((m_5->compressive_force_0), (_S1151)));
        _S1148 = _S1152;
    }
    else
    {
        _S1148 = 0.0f;
    }
    idx_0.w = _S1148;
    return idx_0;
}

[[kernel]] void fr_lp_breaks(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], atomic<uint> device* atom_38 [[buffer(5)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], packed_float4 device* lp_38 [[buffer(24)]], packed_float4 device* ext_40 [[buffer(23)]], packed_float4 device* engine_contacts_38 [[buffer(25)]], CrushParams_0 constant* crush_38 [[buffer(28)]], packed_float4 device* crush_rw_38 [[buffer(27)]], packed_float4 device* crush_tab_38 [[buffer(26)]])
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
    uint i_19 = id_35.x;
    bool _S1153;
    if(i_19 >= (params_38->bond_count_0))
    {
        _S1153 = true;
    }
    else
    {
        uint _S1154 = w_alive_0(&kernelContext_154);
        _S1153 = (*((&kernelContext_154)->work_0+(_S1154 + i_19))) == 0U;
    }
    if(_S1153)
    {
        return;
    }
    uint4 _S1155 = uint4(*((&kernelContext_154)->bonds_0+i_19)) ;
    LpBond_0 _S1156 = lp_bond_0(i_19, &kernelContext_154);
    uint _S1157 = _S1155.x;
    uint _S1158 = _S1155.y;
    thread LpBond_0 _S1159 = _S1156;
    thread float3 q_lin_4;
    thread float3 q_ang_4;
    lp_force_0(&_S1159, _S1157, _S1158, 0U, &q_lin_4, &q_ang_4, &kernelContext_154);
    thread JointBond_0 _S1160 = _S1156.law_0;
    Measures_0 _S1161 = stress_measures_0(&_S1160, q_lin_4, q_ang_4);
    JointMaterial_0 _S1162 = lp_material_0(_S1156.law_0.ids_3.x, &kernelContext_154);
    float _S1163 = _S1156.law_0.geom1_0.w;
    thread JointMaterial_0 _S1164 = _S1162;
    thread JointBond_0 _S1165 = _S1156.law_0;
    thread Measures_0 _S1166 = _S1161;
    float4 _S1167 = failure_indices_0(&_S1164, &_S1165, &_S1166, _S1163);
    float _S1168 = max(max(_S1167.x, _S1167.y), max(_S1167.z, _S1167.w));
    uint _S1169 = a_counters_1(&kernelContext_154);
    uint _S1170 = atomic_fetch_max_explicit((&kernelContext_154)->atom_0+(_S1169 + 15U), (as_type<uint>((max(_S1168, 0.0f)))), memory_order_relaxed);
    if(_S1168 >= 1.0f)
    {
        uint _S1171 = w_alive_0(&kernelContext_154);
        *((&kernelContext_154)->work_0+(_S1171 + i_19)) = 0U;
        uint _S1172 = a_dirty_0(&kernelContext_154);
        uint _S1173 = body_of_0(_S1157, &kernelContext_154);
        uint _S1174 = atomic_fetch_or_explicit((&kernelContext_154)->atom_0+(_S1172 + _S1173), 1U, memory_order_relaxed);
        uint _S1175 = atomic_fetch_add_explicit((&kernelContext_154)->atom_0+_S1169, 1U, memory_order_relaxed);
        uint _S1176 = atomic_fetch_add_explicit((&kernelContext_154)->atom_0+(_S1169 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first_reset(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], atomic<uint> device* atom_39 [[buffer(5)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], packed_float4 device* lp_39 [[buffer(24)]], packed_float4 device* ext_41 [[buffer(23)]], packed_float4 device* engine_contacts_39 [[buffer(25)]], CrushParams_0 constant* crush_39 [[buffer(28)]], packed_float4 device* crush_rw_39 [[buffer(27)]], packed_float4 device* crush_tab_39 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_155;
    (&kernelContext_155)->params_0 = params_39;
    (&kernelContext_155)->atom_0 = atom_39;
    (&kernelContext_155)->work_0 = work_39;
    (&kernelContext_155)->frames_0 = frames_39;
    (&kernelContext_155)->bodies_0 = bodies_39;
    (&kernelContext_155)->chunks_0 = chunks_39;
    (&kernelContext_155)->sort_0 = sort_39;
    (&kernelContext_155)->manifolds_0 = manifolds_39;
    (&kernelContext_155)->loads_0 = loads_39;
    (&kernelContext_155)->bonds_0 = bonds_39;
    (&kernelContext_155)->args_0 = args_39;
    (&kernelContext_155)->info_0 = info_39;
    (&kernelContext_155)->scan_0 = scan_39;
    (&kernelContext_155)->parts_0 = parts_39;
    (&kernelContext_155)->colliders_0 = colliders_39;
    (&kernelContext_155)->collider_info_0 = collider_info_39;
    (&kernelContext_155)->view_chunks_0 = view_chunks_39;
    (&kernelContext_155)->view_0 = view_39;
    (&kernelContext_155)->objects_0 = objects_39;
    (&kernelContext_155)->shot_bodies_0 = shot_bodies_39;
    (&kernelContext_155)->lines_0 = lines_39;
    (&kernelContext_155)->geometry_0 = geometry_39;
    (&kernelContext_155)->shots_0 = shots_39;
    (&kernelContext_155)->lp_0 = lp_39;
    (&kernelContext_155)->ext_0 = ext_41;
    (&kernelContext_155)->engine_contacts_0 = engine_contacts_39;
    (&kernelContext_155)->crush_0 = crush_39;
    (&kernelContext_155)->crush_rw_0 = crush_rw_39;
    (&kernelContext_155)->crush_tab_0 = crush_tab_39;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_155)->g_scan_0 = &g_scan_39;
    threadgroup array<float, int(64)> g_il_best_39;
    (&kernelContext_155)->g_il_best_0 = &g_il_best_39;
    uint _S1177 = id_36.x;
    if(_S1177 == 0U)
    {
        uint _S1178 = a_il_cursor_1(&kernelContext_155);
        atomic_store_explicit((&kernelContext_155)->atom_0+_S1178, 0U, memory_order_relaxed);
    }
    if(_S1177 < ((&kernelContext_155)->params_0->shot_count_0))
    {
        uint _S1179 = a_first_1(_S1177, &kernelContext_155);
        atomic_store_explicit((&kernelContext_155)->atom_0+_S1179, 2139095040U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], atomic<uint> device* atom_40 [[buffer(5)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], packed_float4 device* lp_40 [[buffer(24)]], packed_float4 device* ext_42 [[buffer(23)]], packed_float4 device* engine_contacts_40 [[buffer(25)]], CrushParams_0 constant* crush_40 [[buffer(28)]], packed_float4 device* crush_rw_40 [[buffer(27)]], packed_float4 device* crush_tab_40 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_156;
    (&kernelContext_156)->params_0 = params_40;
    (&kernelContext_156)->atom_0 = atom_40;
    (&kernelContext_156)->work_0 = work_40;
    (&kernelContext_156)->frames_0 = frames_40;
    (&kernelContext_156)->bodies_0 = bodies_40;
    (&kernelContext_156)->chunks_0 = chunks_40;
    (&kernelContext_156)->sort_0 = sort_40;
    (&kernelContext_156)->manifolds_0 = manifolds_40;
    (&kernelContext_156)->loads_0 = loads_40;
    (&kernelContext_156)->bonds_0 = bonds_40;
    (&kernelContext_156)->args_0 = args_40;
    (&kernelContext_156)->info_0 = info_40;
    (&kernelContext_156)->scan_0 = scan_40;
    (&kernelContext_156)->parts_0 = parts_40;
    (&kernelContext_156)->colliders_0 = colliders_40;
    (&kernelContext_156)->collider_info_0 = collider_info_40;
    (&kernelContext_156)->view_chunks_0 = view_chunks_40;
    (&kernelContext_156)->view_0 = view_40;
    (&kernelContext_156)->objects_0 = objects_40;
    (&kernelContext_156)->shot_bodies_0 = shot_bodies_40;
    (&kernelContext_156)->lines_0 = lines_40;
    (&kernelContext_156)->geometry_0 = geometry_40;
    (&kernelContext_156)->shots_0 = shots_40;
    (&kernelContext_156)->lp_0 = lp_40;
    (&kernelContext_156)->ext_0 = ext_42;
    (&kernelContext_156)->engine_contacts_0 = engine_contacts_40;
    (&kernelContext_156)->crush_0 = crush_40;
    (&kernelContext_156)->crush_rw_0 = crush_rw_40;
    (&kernelContext_156)->crush_tab_0 = crush_tab_40;
    threadgroup array<uint, int(1024)> g_scan_40;
    (&kernelContext_156)->g_scan_0 = &g_scan_40;
    threadgroup array<float, int(64)> g_il_best_40;
    (&kernelContext_156)->g_il_best_0 = &g_il_best_40;
    uint b_51 = id_37.x;
    bool _S1180;
    if(b_51 >= (params_40->body_count_0))
    {
        _S1180 = true;
    }
    else
    {
        _S1180 = ((&kernelContext_156)->params_0->shot_count_0) == 0U;
    }
    if(_S1180)
    {
        return;
    }
    thread uint shot_9;
    bool _S1181 = struck_0(b_51, &shot_9, &kernelContext_156);
    if(!_S1181)
    {
        return;
    }
    uint _S1182 = a_first_1(shot_9, &kernelContext_156);
    atomic<uint> device* _S1183 = (&kernelContext_156)->atom_0+_S1182;
    uint _S1184 = a_strike_0(&kernelContext_156);
    uint _S1185 = atomic_load_explicit((&kernelContext_156)->atom_0+(_S1184 + 9U * b_51 + 6U), memory_order_relaxed);
    uint _S1186 = atomic_fetch_min_explicit(_S1183, _S1185, memory_order_relaxed);
    return;
}

bool cs_struck_1(uint b_52, KernelContext_0 thread* kernelContext_157)
{
    bool _S1187;
    if(b_52 < (kernelContext_157->params_0->body_count_0))
    {
        uint _S1188 = a_cs_0(b_52, kernelContext_157);
        uint _S1189 = atomic_load_explicit(kernelContext_157->atom_0+(_S1188 + 1U), memory_order_relaxed);
        _S1187 = _S1189 != 0U;
    }
    else
    {
        _S1187 = false;
    }
    return _S1187;
}

[[kernel]] void fr_il_begin(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], atomic<uint> device* atom_41 [[buffer(5)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], packed_float4 device* lp_41 [[buffer(24)]], packed_float4 device* ext_43 [[buffer(23)]], packed_float4 device* engine_contacts_41 [[buffer(25)]], CrushParams_0 constant* crush_41 [[buffer(28)]], packed_float4 device* crush_rw_41 [[buffer(27)]], packed_float4 device* crush_tab_41 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_158;
    (&kernelContext_158)->params_0 = params_41;
    (&kernelContext_158)->atom_0 = atom_41;
    (&kernelContext_158)->work_0 = work_41;
    (&kernelContext_158)->frames_0 = frames_41;
    (&kernelContext_158)->bodies_0 = bodies_41;
    (&kernelContext_158)->chunks_0 = chunks_41;
    (&kernelContext_158)->sort_0 = sort_41;
    (&kernelContext_158)->manifolds_0 = manifolds_41;
    (&kernelContext_158)->loads_0 = loads_41;
    (&kernelContext_158)->bonds_0 = bonds_41;
    (&kernelContext_158)->args_0 = args_41;
    (&kernelContext_158)->info_0 = info_41;
    (&kernelContext_158)->scan_0 = scan_41;
    (&kernelContext_158)->parts_0 = parts_41;
    (&kernelContext_158)->colliders_0 = colliders_41;
    (&kernelContext_158)->collider_info_0 = collider_info_41;
    (&kernelContext_158)->view_chunks_0 = view_chunks_41;
    (&kernelContext_158)->view_0 = view_41;
    (&kernelContext_158)->objects_0 = objects_41;
    (&kernelContext_158)->shot_bodies_0 = shot_bodies_41;
    (&kernelContext_158)->lines_0 = lines_41;
    (&kernelContext_158)->geometry_0 = geometry_41;
    (&kernelContext_158)->shots_0 = shots_41;
    (&kernelContext_158)->lp_0 = lp_41;
    (&kernelContext_158)->ext_0 = ext_43;
    (&kernelContext_158)->engine_contacts_0 = engine_contacts_41;
    (&kernelContext_158)->crush_0 = crush_41;
    (&kernelContext_158)->crush_rw_0 = crush_rw_41;
    (&kernelContext_158)->crush_tab_0 = crush_tab_41;
    threadgroup array<uint, int(1024)> g_scan_41;
    (&kernelContext_158)->g_scan_0 = &g_scan_41;
    threadgroup array<float, int(64)> g_il_best_41;
    (&kernelContext_158)->g_il_best_0 = &g_il_best_41;
    if((params_41->shot_count_0) > 0U)
    {
        il_begin_1(id_38.x, &kernelContext_158);
    }
    return;
}

[[kernel]] void fr_il_tube_count(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], atomic<uint> device* atom_42 [[buffer(5)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], packed_float4 device* lp_42 [[buffer(24)]], packed_float4 device* ext_44 [[buffer(23)]], packed_float4 device* engine_contacts_42 [[buffer(25)]], CrushParams_0 constant* crush_42 [[buffer(28)]], packed_float4 device* crush_rw_42 [[buffer(27)]], packed_float4 device* crush_tab_42 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_159;
    (&kernelContext_159)->params_0 = params_42;
    (&kernelContext_159)->atom_0 = atom_42;
    (&kernelContext_159)->work_0 = work_42;
    (&kernelContext_159)->frames_0 = frames_42;
    (&kernelContext_159)->bodies_0 = bodies_42;
    (&kernelContext_159)->chunks_0 = chunks_42;
    (&kernelContext_159)->sort_0 = sort_42;
    (&kernelContext_159)->manifolds_0 = manifolds_42;
    (&kernelContext_159)->loads_0 = loads_42;
    (&kernelContext_159)->bonds_0 = bonds_42;
    (&kernelContext_159)->args_0 = args_42;
    (&kernelContext_159)->info_0 = info_42;
    (&kernelContext_159)->scan_0 = scan_42;
    (&kernelContext_159)->parts_0 = parts_42;
    (&kernelContext_159)->colliders_0 = colliders_42;
    (&kernelContext_159)->collider_info_0 = collider_info_42;
    (&kernelContext_159)->view_chunks_0 = view_chunks_42;
    (&kernelContext_159)->view_0 = view_42;
    (&kernelContext_159)->objects_0 = objects_42;
    (&kernelContext_159)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_159)->lines_0 = lines_42;
    (&kernelContext_159)->geometry_0 = geometry_42;
    (&kernelContext_159)->shots_0 = shots_42;
    (&kernelContext_159)->lp_0 = lp_42;
    (&kernelContext_159)->ext_0 = ext_44;
    (&kernelContext_159)->engine_contacts_0 = engine_contacts_42;
    (&kernelContext_159)->crush_0 = crush_42;
    (&kernelContext_159)->crush_rw_0 = crush_rw_42;
    (&kernelContext_159)->crush_tab_0 = crush_tab_42;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_159)->g_scan_0 = &g_scan_42;
    threadgroup array<float, int(64)> g_il_best_42;
    (&kernelContext_159)->g_il_best_0 = &g_il_best_42;
    if((params_42->shot_count_0) > 0U)
    {
        il_tube_count_1(id_39.x, &kernelContext_159);
    }
    return;
}

[[kernel]] void fr_il_tube_base(uint3 id_40 [[thread_position_in_grid]], FractureParams_0 constant* params_43 [[buffer(0)]], atomic<uint> device* atom_43 [[buffer(5)]], uint device* work_43 [[buffer(4)]], packed_float4 device* frames_43 [[buffer(6)]], packed_float4 device* bodies_43 [[buffer(7)]], packed_float4 device* chunks_43 [[buffer(1)]], uint device* sort_43 [[buffer(11)]], Record_natural_0 device* manifolds_43 [[buffer(10)]], packed_float4 device* loads_43 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_43 [[buffer(13)]], uint device* info_43 [[buffer(15)]], ScanParams_0 constant* scan_43 [[buffer(14)]], packed_float4 device* parts_43 [[buffer(3)]], packed_float4 device* colliders_43 [[buffer(8)]], packed_uint4 device* collider_info_43 [[buffer(9)]], packed_float4 device* view_chunks_43 [[buffer(17)]], ViewParams_0 constant* view_43 [[buffer(16)]], packed_float4 device* objects_43 [[buffer(18)]], uint device* shot_bodies_43 [[buffer(20)]], float device* lines_43 [[buffer(19)]], packed_float4 device* geometry_43 [[buffer(21)]], packed_float4 device* shots_43 [[buffer(22)]], packed_float4 device* lp_43 [[buffer(24)]], packed_float4 device* ext_45 [[buffer(23)]], packed_float4 device* engine_contacts_43 [[buffer(25)]], CrushParams_0 constant* crush_43 [[buffer(28)]], packed_float4 device* crush_rw_43 [[buffer(27)]], packed_float4 device* crush_tab_43 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_160;
    (&kernelContext_160)->params_0 = params_43;
    (&kernelContext_160)->atom_0 = atom_43;
    (&kernelContext_160)->work_0 = work_43;
    (&kernelContext_160)->frames_0 = frames_43;
    (&kernelContext_160)->bodies_0 = bodies_43;
    (&kernelContext_160)->chunks_0 = chunks_43;
    (&kernelContext_160)->sort_0 = sort_43;
    (&kernelContext_160)->manifolds_0 = manifolds_43;
    (&kernelContext_160)->loads_0 = loads_43;
    (&kernelContext_160)->bonds_0 = bonds_43;
    (&kernelContext_160)->args_0 = args_43;
    (&kernelContext_160)->info_0 = info_43;
    (&kernelContext_160)->scan_0 = scan_43;
    (&kernelContext_160)->parts_0 = parts_43;
    (&kernelContext_160)->colliders_0 = colliders_43;
    (&kernelContext_160)->collider_info_0 = collider_info_43;
    (&kernelContext_160)->view_chunks_0 = view_chunks_43;
    (&kernelContext_160)->view_0 = view_43;
    (&kernelContext_160)->objects_0 = objects_43;
    (&kernelContext_160)->shot_bodies_0 = shot_bodies_43;
    (&kernelContext_160)->lines_0 = lines_43;
    (&kernelContext_160)->geometry_0 = geometry_43;
    (&kernelContext_160)->shots_0 = shots_43;
    (&kernelContext_160)->lp_0 = lp_43;
    (&kernelContext_160)->ext_0 = ext_45;
    (&kernelContext_160)->engine_contacts_0 = engine_contacts_43;
    (&kernelContext_160)->crush_0 = crush_43;
    (&kernelContext_160)->crush_rw_0 = crush_rw_43;
    (&kernelContext_160)->crush_tab_0 = crush_tab_43;
    threadgroup array<uint, int(1024)> g_scan_43;
    (&kernelContext_160)->g_scan_0 = &g_scan_43;
    threadgroup array<float, int(64)> g_il_best_43;
    (&kernelContext_160)->g_il_best_0 = &g_il_best_43;
    if((params_43->shot_count_0) > 0U)
    {
        il_tube_base_1(id_40.x, &kernelContext_160);
    }
    return;
}

[[kernel]] void fr_il_tube_collect(uint3 id_41 [[thread_position_in_grid]], FractureParams_0 constant* params_44 [[buffer(0)]], atomic<uint> device* atom_44 [[buffer(5)]], uint device* work_44 [[buffer(4)]], packed_float4 device* frames_44 [[buffer(6)]], packed_float4 device* bodies_44 [[buffer(7)]], packed_float4 device* chunks_44 [[buffer(1)]], uint device* sort_44 [[buffer(11)]], Record_natural_0 device* manifolds_44 [[buffer(10)]], packed_float4 device* loads_44 [[buffer(12)]], packed_uint4 device* bonds_44 [[buffer(2)]], uint device* args_44 [[buffer(13)]], uint device* info_44 [[buffer(15)]], ScanParams_0 constant* scan_44 [[buffer(14)]], packed_float4 device* parts_44 [[buffer(3)]], packed_float4 device* colliders_44 [[buffer(8)]], packed_uint4 device* collider_info_44 [[buffer(9)]], packed_float4 device* view_chunks_44 [[buffer(17)]], ViewParams_0 constant* view_44 [[buffer(16)]], packed_float4 device* objects_44 [[buffer(18)]], uint device* shot_bodies_44 [[buffer(20)]], float device* lines_44 [[buffer(19)]], packed_float4 device* geometry_44 [[buffer(21)]], packed_float4 device* shots_44 [[buffer(22)]], packed_float4 device* lp_44 [[buffer(24)]], packed_float4 device* ext_46 [[buffer(23)]], packed_float4 device* engine_contacts_44 [[buffer(25)]], CrushParams_0 constant* crush_44 [[buffer(28)]], packed_float4 device* crush_rw_44 [[buffer(27)]], packed_float4 device* crush_tab_44 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_161;
    (&kernelContext_161)->params_0 = params_44;
    (&kernelContext_161)->atom_0 = atom_44;
    (&kernelContext_161)->work_0 = work_44;
    (&kernelContext_161)->frames_0 = frames_44;
    (&kernelContext_161)->bodies_0 = bodies_44;
    (&kernelContext_161)->chunks_0 = chunks_44;
    (&kernelContext_161)->sort_0 = sort_44;
    (&kernelContext_161)->manifolds_0 = manifolds_44;
    (&kernelContext_161)->loads_0 = loads_44;
    (&kernelContext_161)->bonds_0 = bonds_44;
    (&kernelContext_161)->args_0 = args_44;
    (&kernelContext_161)->info_0 = info_44;
    (&kernelContext_161)->scan_0 = scan_44;
    (&kernelContext_161)->parts_0 = parts_44;
    (&kernelContext_161)->colliders_0 = colliders_44;
    (&kernelContext_161)->collider_info_0 = collider_info_44;
    (&kernelContext_161)->view_chunks_0 = view_chunks_44;
    (&kernelContext_161)->view_0 = view_44;
    (&kernelContext_161)->objects_0 = objects_44;
    (&kernelContext_161)->shot_bodies_0 = shot_bodies_44;
    (&kernelContext_161)->lines_0 = lines_44;
    (&kernelContext_161)->geometry_0 = geometry_44;
    (&kernelContext_161)->shots_0 = shots_44;
    (&kernelContext_161)->lp_0 = lp_44;
    (&kernelContext_161)->ext_0 = ext_46;
    (&kernelContext_161)->engine_contacts_0 = engine_contacts_44;
    (&kernelContext_161)->crush_0 = crush_44;
    (&kernelContext_161)->crush_rw_0 = crush_rw_44;
    (&kernelContext_161)->crush_tab_0 = crush_tab_44;
    threadgroup array<uint, int(1024)> g_scan_44;
    (&kernelContext_161)->g_scan_0 = &g_scan_44;
    threadgroup array<float, int(64)> g_il_best_44;
    (&kernelContext_161)->g_il_best_0 = &g_il_best_44;
    if((params_44->shot_count_0) > 0U)
    {
        il_tube_collect_1(id_41.x, &kernelContext_161);
    }
    return;
}

[[kernel]] void fr_il_run(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_45 [[buffer(0)]], atomic<uint> device* atom_45 [[buffer(5)]], uint device* work_45 [[buffer(4)]], packed_float4 device* frames_45 [[buffer(6)]], packed_float4 device* bodies_45 [[buffer(7)]], packed_float4 device* chunks_45 [[buffer(1)]], uint device* sort_45 [[buffer(11)]], Record_natural_0 device* manifolds_45 [[buffer(10)]], packed_float4 device* loads_45 [[buffer(12)]], packed_uint4 device* bonds_45 [[buffer(2)]], uint device* args_45 [[buffer(13)]], uint device* info_45 [[buffer(15)]], ScanParams_0 constant* scan_45 [[buffer(14)]], packed_float4 device* parts_45 [[buffer(3)]], packed_float4 device* colliders_45 [[buffer(8)]], packed_uint4 device* collider_info_45 [[buffer(9)]], packed_float4 device* view_chunks_45 [[buffer(17)]], ViewParams_0 constant* view_45 [[buffer(16)]], packed_float4 device* objects_45 [[buffer(18)]], uint device* shot_bodies_45 [[buffer(20)]], float device* lines_45 [[buffer(19)]], packed_float4 device* geometry_45 [[buffer(21)]], packed_float4 device* shots_45 [[buffer(22)]], packed_float4 device* lp_45 [[buffer(24)]], packed_float4 device* ext_47 [[buffer(23)]], packed_float4 device* engine_contacts_45 [[buffer(25)]], CrushParams_0 constant* crush_45 [[buffer(28)]], packed_float4 device* crush_rw_45 [[buffer(27)]], packed_float4 device* crush_tab_45 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_162;
    (&kernelContext_162)->params_0 = params_45;
    (&kernelContext_162)->atom_0 = atom_45;
    (&kernelContext_162)->work_0 = work_45;
    (&kernelContext_162)->frames_0 = frames_45;
    (&kernelContext_162)->bodies_0 = bodies_45;
    (&kernelContext_162)->chunks_0 = chunks_45;
    (&kernelContext_162)->sort_0 = sort_45;
    (&kernelContext_162)->manifolds_0 = manifolds_45;
    (&kernelContext_162)->loads_0 = loads_45;
    (&kernelContext_162)->bonds_0 = bonds_45;
    (&kernelContext_162)->args_0 = args_45;
    (&kernelContext_162)->info_0 = info_45;
    (&kernelContext_162)->scan_0 = scan_45;
    (&kernelContext_162)->parts_0 = parts_45;
    (&kernelContext_162)->colliders_0 = colliders_45;
    (&kernelContext_162)->collider_info_0 = collider_info_45;
    (&kernelContext_162)->view_chunks_0 = view_chunks_45;
    (&kernelContext_162)->view_0 = view_45;
    (&kernelContext_162)->objects_0 = objects_45;
    (&kernelContext_162)->shot_bodies_0 = shot_bodies_45;
    (&kernelContext_162)->lines_0 = lines_45;
    (&kernelContext_162)->geometry_0 = geometry_45;
    (&kernelContext_162)->shots_0 = shots_45;
    (&kernelContext_162)->lp_0 = lp_45;
    (&kernelContext_162)->ext_0 = ext_47;
    (&kernelContext_162)->engine_contacts_0 = engine_contacts_45;
    (&kernelContext_162)->crush_0 = crush_45;
    (&kernelContext_162)->crush_rw_0 = crush_rw_45;
    (&kernelContext_162)->crush_tab_0 = crush_tab_45;
    threadgroup array<uint, int(1024)> g_scan_45;
    (&kernelContext_162)->g_scan_0 = &g_scan_45;
    threadgroup array<float, int(64)> g_il_best_45;
    (&kernelContext_162)->g_il_best_0 = &g_il_best_45;
    if((params_45->shot_count_0) > 0U)
    {
        il_run_1(group_1.x, thread_4.x, &kernelContext_162);
    }
    return;
}

[[kernel]] void fr_il_strike(uint3 id_42 [[thread_position_in_grid]], FractureParams_0 constant* params_46 [[buffer(0)]], atomic<uint> device* atom_46 [[buffer(5)]], uint device* work_46 [[buffer(4)]], packed_float4 device* frames_46 [[buffer(6)]], packed_float4 device* bodies_46 [[buffer(7)]], packed_float4 device* chunks_46 [[buffer(1)]], uint device* sort_46 [[buffer(11)]], Record_natural_0 device* manifolds_46 [[buffer(10)]], packed_float4 device* loads_46 [[buffer(12)]], packed_uint4 device* bonds_46 [[buffer(2)]], uint device* args_46 [[buffer(13)]], uint device* info_46 [[buffer(15)]], ScanParams_0 constant* scan_46 [[buffer(14)]], packed_float4 device* parts_46 [[buffer(3)]], packed_float4 device* colliders_46 [[buffer(8)]], packed_uint4 device* collider_info_46 [[buffer(9)]], packed_float4 device* view_chunks_46 [[buffer(17)]], ViewParams_0 constant* view_46 [[buffer(16)]], packed_float4 device* objects_46 [[buffer(18)]], uint device* shot_bodies_46 [[buffer(20)]], float device* lines_46 [[buffer(19)]], packed_float4 device* geometry_46 [[buffer(21)]], packed_float4 device* shots_46 [[buffer(22)]], packed_float4 device* lp_46 [[buffer(24)]], packed_float4 device* ext_48 [[buffer(23)]], packed_float4 device* engine_contacts_46 [[buffer(25)]], CrushParams_0 constant* crush_46 [[buffer(28)]], packed_float4 device* crush_rw_46 [[buffer(27)]], packed_float4 device* crush_tab_46 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_163;
    (&kernelContext_163)->params_0 = params_46;
    (&kernelContext_163)->atom_0 = atom_46;
    (&kernelContext_163)->work_0 = work_46;
    (&kernelContext_163)->frames_0 = frames_46;
    (&kernelContext_163)->bodies_0 = bodies_46;
    (&kernelContext_163)->chunks_0 = chunks_46;
    (&kernelContext_163)->sort_0 = sort_46;
    (&kernelContext_163)->manifolds_0 = manifolds_46;
    (&kernelContext_163)->loads_0 = loads_46;
    (&kernelContext_163)->bonds_0 = bonds_46;
    (&kernelContext_163)->args_0 = args_46;
    (&kernelContext_163)->info_0 = info_46;
    (&kernelContext_163)->scan_0 = scan_46;
    (&kernelContext_163)->parts_0 = parts_46;
    (&kernelContext_163)->colliders_0 = colliders_46;
    (&kernelContext_163)->collider_info_0 = collider_info_46;
    (&kernelContext_163)->view_chunks_0 = view_chunks_46;
    (&kernelContext_163)->view_0 = view_46;
    (&kernelContext_163)->objects_0 = objects_46;
    (&kernelContext_163)->shot_bodies_0 = shot_bodies_46;
    (&kernelContext_163)->lines_0 = lines_46;
    (&kernelContext_163)->geometry_0 = geometry_46;
    (&kernelContext_163)->shots_0 = shots_46;
    (&kernelContext_163)->lp_0 = lp_46;
    (&kernelContext_163)->ext_0 = ext_48;
    (&kernelContext_163)->engine_contacts_0 = engine_contacts_46;
    (&kernelContext_163)->crush_0 = crush_46;
    (&kernelContext_163)->crush_rw_0 = crush_rw_46;
    (&kernelContext_163)->crush_tab_0 = crush_tab_46;
    threadgroup array<uint, int(1024)> g_scan_46;
    (&kernelContext_163)->g_scan_0 = &g_scan_46;
    threadgroup array<float, int(64)> g_il_best_46;
    (&kernelContext_163)->g_il_best_0 = &g_il_best_46;
    if((params_46->shot_count_0) > 0U)
    {
        il_strike_1(id_42.x, &kernelContext_163);
    }
    return;
}

[[kernel]] void fr_il_mark(uint3 id_43 [[thread_position_in_grid]], FractureParams_0 constant* params_47 [[buffer(0)]], atomic<uint> device* atom_47 [[buffer(5)]], uint device* work_47 [[buffer(4)]], packed_float4 device* frames_47 [[buffer(6)]], packed_float4 device* bodies_47 [[buffer(7)]], packed_float4 device* chunks_47 [[buffer(1)]], uint device* sort_47 [[buffer(11)]], Record_natural_0 device* manifolds_47 [[buffer(10)]], packed_float4 device* loads_47 [[buffer(12)]], packed_uint4 device* bonds_47 [[buffer(2)]], uint device* args_47 [[buffer(13)]], uint device* info_47 [[buffer(15)]], ScanParams_0 constant* scan_47 [[buffer(14)]], packed_float4 device* parts_47 [[buffer(3)]], packed_float4 device* colliders_47 [[buffer(8)]], packed_uint4 device* collider_info_47 [[buffer(9)]], packed_float4 device* view_chunks_47 [[buffer(17)]], ViewParams_0 constant* view_47 [[buffer(16)]], packed_float4 device* objects_47 [[buffer(18)]], uint device* shot_bodies_47 [[buffer(20)]], float device* lines_47 [[buffer(19)]], packed_float4 device* geometry_47 [[buffer(21)]], packed_float4 device* shots_47 [[buffer(22)]], packed_float4 device* lp_47 [[buffer(24)]], packed_float4 device* ext_49 [[buffer(23)]], packed_float4 device* engine_contacts_47 [[buffer(25)]], CrushParams_0 constant* crush_47 [[buffer(28)]], packed_float4 device* crush_rw_47 [[buffer(27)]], packed_float4 device* crush_tab_47 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_164;
    (&kernelContext_164)->params_0 = params_47;
    (&kernelContext_164)->atom_0 = atom_47;
    (&kernelContext_164)->work_0 = work_47;
    (&kernelContext_164)->frames_0 = frames_47;
    (&kernelContext_164)->bodies_0 = bodies_47;
    (&kernelContext_164)->chunks_0 = chunks_47;
    (&kernelContext_164)->sort_0 = sort_47;
    (&kernelContext_164)->manifolds_0 = manifolds_47;
    (&kernelContext_164)->loads_0 = loads_47;
    (&kernelContext_164)->bonds_0 = bonds_47;
    (&kernelContext_164)->args_0 = args_47;
    (&kernelContext_164)->info_0 = info_47;
    (&kernelContext_164)->scan_0 = scan_47;
    (&kernelContext_164)->parts_0 = parts_47;
    (&kernelContext_164)->colliders_0 = colliders_47;
    (&kernelContext_164)->collider_info_0 = collider_info_47;
    (&kernelContext_164)->view_chunks_0 = view_chunks_47;
    (&kernelContext_164)->view_0 = view_47;
    (&kernelContext_164)->objects_0 = objects_47;
    (&kernelContext_164)->shot_bodies_0 = shot_bodies_47;
    (&kernelContext_164)->lines_0 = lines_47;
    (&kernelContext_164)->geometry_0 = geometry_47;
    (&kernelContext_164)->shots_0 = shots_47;
    (&kernelContext_164)->lp_0 = lp_47;
    (&kernelContext_164)->ext_0 = ext_49;
    (&kernelContext_164)->engine_contacts_0 = engine_contacts_47;
    (&kernelContext_164)->crush_0 = crush_47;
    (&kernelContext_164)->crush_rw_0 = crush_rw_47;
    (&kernelContext_164)->crush_tab_0 = crush_tab_47;
    threadgroup array<uint, int(1024)> g_scan_47;
    (&kernelContext_164)->g_scan_0 = &g_scan_47;
    threadgroup array<float, int(64)> g_il_best_47;
    (&kernelContext_164)->g_il_best_0 = &g_il_best_47;
    if((params_47->shot_count_0) > 0U)
    {
        il_mark_1(id_43.x, &kernelContext_164);
    }
    return;
}

[[kernel]] void fr_il_carry(uint3 id_44 [[thread_position_in_grid]], FractureParams_0 constant* params_48 [[buffer(0)]], atomic<uint> device* atom_48 [[buffer(5)]], uint device* work_48 [[buffer(4)]], packed_float4 device* frames_48 [[buffer(6)]], packed_float4 device* bodies_48 [[buffer(7)]], packed_float4 device* chunks_48 [[buffer(1)]], uint device* sort_48 [[buffer(11)]], Record_natural_0 device* manifolds_48 [[buffer(10)]], packed_float4 device* loads_48 [[buffer(12)]], packed_uint4 device* bonds_48 [[buffer(2)]], uint device* args_48 [[buffer(13)]], uint device* info_48 [[buffer(15)]], ScanParams_0 constant* scan_48 [[buffer(14)]], packed_float4 device* parts_48 [[buffer(3)]], packed_float4 device* colliders_48 [[buffer(8)]], packed_uint4 device* collider_info_48 [[buffer(9)]], packed_float4 device* view_chunks_48 [[buffer(17)]], ViewParams_0 constant* view_48 [[buffer(16)]], packed_float4 device* objects_48 [[buffer(18)]], uint device* shot_bodies_48 [[buffer(20)]], float device* lines_48 [[buffer(19)]], packed_float4 device* geometry_48 [[buffer(21)]], packed_float4 device* shots_48 [[buffer(22)]], packed_float4 device* lp_48 [[buffer(24)]], packed_float4 device* ext_50 [[buffer(23)]], packed_float4 device* engine_contacts_48 [[buffer(25)]], CrushParams_0 constant* crush_48 [[buffer(28)]], packed_float4 device* crush_rw_48 [[buffer(27)]], packed_float4 device* crush_tab_48 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_165;
    (&kernelContext_165)->params_0 = params_48;
    (&kernelContext_165)->atom_0 = atom_48;
    (&kernelContext_165)->work_0 = work_48;
    (&kernelContext_165)->frames_0 = frames_48;
    (&kernelContext_165)->bodies_0 = bodies_48;
    (&kernelContext_165)->chunks_0 = chunks_48;
    (&kernelContext_165)->sort_0 = sort_48;
    (&kernelContext_165)->manifolds_0 = manifolds_48;
    (&kernelContext_165)->loads_0 = loads_48;
    (&kernelContext_165)->bonds_0 = bonds_48;
    (&kernelContext_165)->args_0 = args_48;
    (&kernelContext_165)->info_0 = info_48;
    (&kernelContext_165)->scan_0 = scan_48;
    (&kernelContext_165)->parts_0 = parts_48;
    (&kernelContext_165)->colliders_0 = colliders_48;
    (&kernelContext_165)->collider_info_0 = collider_info_48;
    (&kernelContext_165)->view_chunks_0 = view_chunks_48;
    (&kernelContext_165)->view_0 = view_48;
    (&kernelContext_165)->objects_0 = objects_48;
    (&kernelContext_165)->shot_bodies_0 = shot_bodies_48;
    (&kernelContext_165)->lines_0 = lines_48;
    (&kernelContext_165)->geometry_0 = geometry_48;
    (&kernelContext_165)->shots_0 = shots_48;
    (&kernelContext_165)->lp_0 = lp_48;
    (&kernelContext_165)->ext_0 = ext_50;
    (&kernelContext_165)->engine_contacts_0 = engine_contacts_48;
    (&kernelContext_165)->crush_0 = crush_48;
    (&kernelContext_165)->crush_rw_0 = crush_rw_48;
    (&kernelContext_165)->crush_tab_0 = crush_tab_48;
    threadgroup array<uint, int(1024)> g_scan_48;
    (&kernelContext_165)->g_scan_0 = &g_scan_48;
    threadgroup array<float, int(64)> g_il_best_48;
    (&kernelContext_165)->g_il_best_0 = &g_il_best_48;
    if((params_48->shot_count_0) > 0U)
    {
        il_carry_1(id_44.x, &kernelContext_165);
    }
    return;
}

[[kernel]] void fr_il_breaks(uint3 id_45 [[thread_position_in_grid]], FractureParams_0 constant* params_49 [[buffer(0)]], atomic<uint> device* atom_49 [[buffer(5)]], uint device* work_49 [[buffer(4)]], packed_float4 device* frames_49 [[buffer(6)]], packed_float4 device* bodies_49 [[buffer(7)]], packed_float4 device* chunks_49 [[buffer(1)]], uint device* sort_49 [[buffer(11)]], Record_natural_0 device* manifolds_49 [[buffer(10)]], packed_float4 device* loads_49 [[buffer(12)]], packed_uint4 device* bonds_49 [[buffer(2)]], uint device* args_49 [[buffer(13)]], uint device* info_49 [[buffer(15)]], ScanParams_0 constant* scan_49 [[buffer(14)]], packed_float4 device* parts_49 [[buffer(3)]], packed_float4 device* colliders_49 [[buffer(8)]], packed_uint4 device* collider_info_49 [[buffer(9)]], packed_float4 device* view_chunks_49 [[buffer(17)]], ViewParams_0 constant* view_49 [[buffer(16)]], packed_float4 device* objects_49 [[buffer(18)]], uint device* shot_bodies_49 [[buffer(20)]], float device* lines_49 [[buffer(19)]], packed_float4 device* geometry_49 [[buffer(21)]], packed_float4 device* shots_49 [[buffer(22)]], packed_float4 device* lp_49 [[buffer(24)]], packed_float4 device* ext_51 [[buffer(23)]], packed_float4 device* engine_contacts_49 [[buffer(25)]], CrushParams_0 constant* crush_49 [[buffer(28)]], packed_float4 device* crush_rw_49 [[buffer(27)]], packed_float4 device* crush_tab_49 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_166;
    (&kernelContext_166)->params_0 = params_49;
    (&kernelContext_166)->atom_0 = atom_49;
    (&kernelContext_166)->work_0 = work_49;
    (&kernelContext_166)->frames_0 = frames_49;
    (&kernelContext_166)->bodies_0 = bodies_49;
    (&kernelContext_166)->chunks_0 = chunks_49;
    (&kernelContext_166)->sort_0 = sort_49;
    (&kernelContext_166)->manifolds_0 = manifolds_49;
    (&kernelContext_166)->loads_0 = loads_49;
    (&kernelContext_166)->bonds_0 = bonds_49;
    (&kernelContext_166)->args_0 = args_49;
    (&kernelContext_166)->info_0 = info_49;
    (&kernelContext_166)->scan_0 = scan_49;
    (&kernelContext_166)->parts_0 = parts_49;
    (&kernelContext_166)->colliders_0 = colliders_49;
    (&kernelContext_166)->collider_info_0 = collider_info_49;
    (&kernelContext_166)->view_chunks_0 = view_chunks_49;
    (&kernelContext_166)->view_0 = view_49;
    (&kernelContext_166)->objects_0 = objects_49;
    (&kernelContext_166)->shot_bodies_0 = shot_bodies_49;
    (&kernelContext_166)->lines_0 = lines_49;
    (&kernelContext_166)->geometry_0 = geometry_49;
    (&kernelContext_166)->shots_0 = shots_49;
    (&kernelContext_166)->lp_0 = lp_49;
    (&kernelContext_166)->ext_0 = ext_51;
    (&kernelContext_166)->engine_contacts_0 = engine_contacts_49;
    (&kernelContext_166)->crush_0 = crush_49;
    (&kernelContext_166)->crush_rw_0 = crush_rw_49;
    (&kernelContext_166)->crush_tab_0 = crush_tab_49;
    threadgroup array<uint, int(1024)> g_scan_49;
    (&kernelContext_166)->g_scan_0 = &g_scan_49;
    threadgroup array<float, int(64)> g_il_best_49;
    (&kernelContext_166)->g_il_best_0 = &g_il_best_49;
    if((params_49->shot_count_0) > 0U)
    {
        il_breaks_1(id_45.x, &kernelContext_166);
    }
    return;
}

[[kernel]] void fr_csl_begin(uint3 id_46 [[thread_position_in_grid]], FractureParams_0 constant* params_50 [[buffer(0)]], atomic<uint> device* atom_50 [[buffer(5)]], uint device* work_50 [[buffer(4)]], packed_float4 device* frames_50 [[buffer(6)]], packed_float4 device* bodies_50 [[buffer(7)]], packed_float4 device* chunks_50 [[buffer(1)]], uint device* sort_50 [[buffer(11)]], Record_natural_0 device* manifolds_50 [[buffer(10)]], packed_float4 device* loads_50 [[buffer(12)]], packed_uint4 device* bonds_50 [[buffer(2)]], uint device* args_50 [[buffer(13)]], uint device* info_50 [[buffer(15)]], ScanParams_0 constant* scan_50 [[buffer(14)]], packed_float4 device* parts_50 [[buffer(3)]], packed_float4 device* colliders_50 [[buffer(8)]], packed_uint4 device* collider_info_50 [[buffer(9)]], packed_float4 device* view_chunks_50 [[buffer(17)]], ViewParams_0 constant* view_50 [[buffer(16)]], packed_float4 device* objects_50 [[buffer(18)]], uint device* shot_bodies_50 [[buffer(20)]], float device* lines_50 [[buffer(19)]], packed_float4 device* geometry_50 [[buffer(21)]], packed_float4 device* shots_50 [[buffer(22)]], packed_float4 device* lp_50 [[buffer(24)]], packed_float4 device* ext_52 [[buffer(23)]], packed_float4 device* engine_contacts_50 [[buffer(25)]], CrushParams_0 constant* crush_50 [[buffer(28)]], packed_float4 device* crush_rw_50 [[buffer(27)]], packed_float4 device* crush_tab_50 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_167;
    (&kernelContext_167)->params_0 = params_50;
    (&kernelContext_167)->atom_0 = atom_50;
    (&kernelContext_167)->work_0 = work_50;
    (&kernelContext_167)->frames_0 = frames_50;
    (&kernelContext_167)->bodies_0 = bodies_50;
    (&kernelContext_167)->chunks_0 = chunks_50;
    (&kernelContext_167)->sort_0 = sort_50;
    (&kernelContext_167)->manifolds_0 = manifolds_50;
    (&kernelContext_167)->loads_0 = loads_50;
    (&kernelContext_167)->bonds_0 = bonds_50;
    (&kernelContext_167)->args_0 = args_50;
    (&kernelContext_167)->info_0 = info_50;
    (&kernelContext_167)->scan_0 = scan_50;
    (&kernelContext_167)->parts_0 = parts_50;
    (&kernelContext_167)->colliders_0 = colliders_50;
    (&kernelContext_167)->collider_info_0 = collider_info_50;
    (&kernelContext_167)->view_chunks_0 = view_chunks_50;
    (&kernelContext_167)->view_0 = view_50;
    (&kernelContext_167)->objects_0 = objects_50;
    (&kernelContext_167)->shot_bodies_0 = shot_bodies_50;
    (&kernelContext_167)->lines_0 = lines_50;
    (&kernelContext_167)->geometry_0 = geometry_50;
    (&kernelContext_167)->shots_0 = shots_50;
    (&kernelContext_167)->lp_0 = lp_50;
    (&kernelContext_167)->ext_0 = ext_52;
    (&kernelContext_167)->engine_contacts_0 = engine_contacts_50;
    (&kernelContext_167)->crush_0 = crush_50;
    (&kernelContext_167)->crush_rw_0 = crush_rw_50;
    (&kernelContext_167)->crush_tab_0 = crush_tab_50;
    threadgroup array<uint, int(1024)> g_scan_50;
    (&kernelContext_167)->g_scan_0 = &g_scan_50;
    threadgroup array<float, int(64)> g_il_best_50;
    (&kernelContext_167)->g_il_best_0 = &g_il_best_50;
    il_begin_0(id_46.x, &kernelContext_167);
    return;
}

[[kernel]] void fr_csl_tube_count(uint3 id_47 [[thread_position_in_grid]], FractureParams_0 constant* params_51 [[buffer(0)]], atomic<uint> device* atom_51 [[buffer(5)]], uint device* work_51 [[buffer(4)]], packed_float4 device* frames_51 [[buffer(6)]], packed_float4 device* bodies_51 [[buffer(7)]], packed_float4 device* chunks_51 [[buffer(1)]], uint device* sort_51 [[buffer(11)]], Record_natural_0 device* manifolds_51 [[buffer(10)]], packed_float4 device* loads_51 [[buffer(12)]], packed_uint4 device* bonds_51 [[buffer(2)]], uint device* args_51 [[buffer(13)]], uint device* info_51 [[buffer(15)]], ScanParams_0 constant* scan_51 [[buffer(14)]], packed_float4 device* parts_51 [[buffer(3)]], packed_float4 device* colliders_51 [[buffer(8)]], packed_uint4 device* collider_info_51 [[buffer(9)]], packed_float4 device* view_chunks_51 [[buffer(17)]], ViewParams_0 constant* view_51 [[buffer(16)]], packed_float4 device* objects_51 [[buffer(18)]], uint device* shot_bodies_51 [[buffer(20)]], float device* lines_51 [[buffer(19)]], packed_float4 device* geometry_51 [[buffer(21)]], packed_float4 device* shots_51 [[buffer(22)]], packed_float4 device* lp_51 [[buffer(24)]], packed_float4 device* ext_53 [[buffer(23)]], packed_float4 device* engine_contacts_51 [[buffer(25)]], CrushParams_0 constant* crush_51 [[buffer(28)]], packed_float4 device* crush_rw_51 [[buffer(27)]], packed_float4 device* crush_tab_51 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_168;
    (&kernelContext_168)->params_0 = params_51;
    (&kernelContext_168)->atom_0 = atom_51;
    (&kernelContext_168)->work_0 = work_51;
    (&kernelContext_168)->frames_0 = frames_51;
    (&kernelContext_168)->bodies_0 = bodies_51;
    (&kernelContext_168)->chunks_0 = chunks_51;
    (&kernelContext_168)->sort_0 = sort_51;
    (&kernelContext_168)->manifolds_0 = manifolds_51;
    (&kernelContext_168)->loads_0 = loads_51;
    (&kernelContext_168)->bonds_0 = bonds_51;
    (&kernelContext_168)->args_0 = args_51;
    (&kernelContext_168)->info_0 = info_51;
    (&kernelContext_168)->scan_0 = scan_51;
    (&kernelContext_168)->parts_0 = parts_51;
    (&kernelContext_168)->colliders_0 = colliders_51;
    (&kernelContext_168)->collider_info_0 = collider_info_51;
    (&kernelContext_168)->view_chunks_0 = view_chunks_51;
    (&kernelContext_168)->view_0 = view_51;
    (&kernelContext_168)->objects_0 = objects_51;
    (&kernelContext_168)->shot_bodies_0 = shot_bodies_51;
    (&kernelContext_168)->lines_0 = lines_51;
    (&kernelContext_168)->geometry_0 = geometry_51;
    (&kernelContext_168)->shots_0 = shots_51;
    (&kernelContext_168)->lp_0 = lp_51;
    (&kernelContext_168)->ext_0 = ext_53;
    (&kernelContext_168)->engine_contacts_0 = engine_contacts_51;
    (&kernelContext_168)->crush_0 = crush_51;
    (&kernelContext_168)->crush_rw_0 = crush_rw_51;
    (&kernelContext_168)->crush_tab_0 = crush_tab_51;
    threadgroup array<uint, int(1024)> g_scan_51;
    (&kernelContext_168)->g_scan_0 = &g_scan_51;
    threadgroup array<float, int(64)> g_il_best_51;
    (&kernelContext_168)->g_il_best_0 = &g_il_best_51;
    il_tube_count_0(id_47.x, &kernelContext_168);
    return;
}

[[kernel]] void fr_csl_tube_base(uint3 id_48 [[thread_position_in_grid]], FractureParams_0 constant* params_52 [[buffer(0)]], atomic<uint> device* atom_52 [[buffer(5)]], uint device* work_52 [[buffer(4)]], packed_float4 device* frames_52 [[buffer(6)]], packed_float4 device* bodies_52 [[buffer(7)]], packed_float4 device* chunks_52 [[buffer(1)]], uint device* sort_52 [[buffer(11)]], Record_natural_0 device* manifolds_52 [[buffer(10)]], packed_float4 device* loads_52 [[buffer(12)]], packed_uint4 device* bonds_52 [[buffer(2)]], uint device* args_52 [[buffer(13)]], uint device* info_52 [[buffer(15)]], ScanParams_0 constant* scan_52 [[buffer(14)]], packed_float4 device* parts_52 [[buffer(3)]], packed_float4 device* colliders_52 [[buffer(8)]], packed_uint4 device* collider_info_52 [[buffer(9)]], packed_float4 device* view_chunks_52 [[buffer(17)]], ViewParams_0 constant* view_52 [[buffer(16)]], packed_float4 device* objects_52 [[buffer(18)]], uint device* shot_bodies_52 [[buffer(20)]], float device* lines_52 [[buffer(19)]], packed_float4 device* geometry_52 [[buffer(21)]], packed_float4 device* shots_52 [[buffer(22)]], packed_float4 device* lp_52 [[buffer(24)]], packed_float4 device* ext_54 [[buffer(23)]], packed_float4 device* engine_contacts_52 [[buffer(25)]], CrushParams_0 constant* crush_52 [[buffer(28)]], packed_float4 device* crush_rw_52 [[buffer(27)]], packed_float4 device* crush_tab_52 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_169;
    (&kernelContext_169)->params_0 = params_52;
    (&kernelContext_169)->atom_0 = atom_52;
    (&kernelContext_169)->work_0 = work_52;
    (&kernelContext_169)->frames_0 = frames_52;
    (&kernelContext_169)->bodies_0 = bodies_52;
    (&kernelContext_169)->chunks_0 = chunks_52;
    (&kernelContext_169)->sort_0 = sort_52;
    (&kernelContext_169)->manifolds_0 = manifolds_52;
    (&kernelContext_169)->loads_0 = loads_52;
    (&kernelContext_169)->bonds_0 = bonds_52;
    (&kernelContext_169)->args_0 = args_52;
    (&kernelContext_169)->info_0 = info_52;
    (&kernelContext_169)->scan_0 = scan_52;
    (&kernelContext_169)->parts_0 = parts_52;
    (&kernelContext_169)->colliders_0 = colliders_52;
    (&kernelContext_169)->collider_info_0 = collider_info_52;
    (&kernelContext_169)->view_chunks_0 = view_chunks_52;
    (&kernelContext_169)->view_0 = view_52;
    (&kernelContext_169)->objects_0 = objects_52;
    (&kernelContext_169)->shot_bodies_0 = shot_bodies_52;
    (&kernelContext_169)->lines_0 = lines_52;
    (&kernelContext_169)->geometry_0 = geometry_52;
    (&kernelContext_169)->shots_0 = shots_52;
    (&kernelContext_169)->lp_0 = lp_52;
    (&kernelContext_169)->ext_0 = ext_54;
    (&kernelContext_169)->engine_contacts_0 = engine_contacts_52;
    (&kernelContext_169)->crush_0 = crush_52;
    (&kernelContext_169)->crush_rw_0 = crush_rw_52;
    (&kernelContext_169)->crush_tab_0 = crush_tab_52;
    threadgroup array<uint, int(1024)> g_scan_52;
    (&kernelContext_169)->g_scan_0 = &g_scan_52;
    threadgroup array<float, int(64)> g_il_best_52;
    (&kernelContext_169)->g_il_best_0 = &g_il_best_52;
    il_tube_base_0(id_48.x, &kernelContext_169);
    return;
}

[[kernel]] void fr_csl_tube_collect(uint3 id_49 [[thread_position_in_grid]], FractureParams_0 constant* params_53 [[buffer(0)]], atomic<uint> device* atom_53 [[buffer(5)]], uint device* work_53 [[buffer(4)]], packed_float4 device* frames_53 [[buffer(6)]], packed_float4 device* bodies_53 [[buffer(7)]], packed_float4 device* chunks_53 [[buffer(1)]], uint device* sort_53 [[buffer(11)]], Record_natural_0 device* manifolds_53 [[buffer(10)]], packed_float4 device* loads_53 [[buffer(12)]], packed_uint4 device* bonds_53 [[buffer(2)]], uint device* args_53 [[buffer(13)]], uint device* info_53 [[buffer(15)]], ScanParams_0 constant* scan_53 [[buffer(14)]], packed_float4 device* parts_53 [[buffer(3)]], packed_float4 device* colliders_53 [[buffer(8)]], packed_uint4 device* collider_info_53 [[buffer(9)]], packed_float4 device* view_chunks_53 [[buffer(17)]], ViewParams_0 constant* view_53 [[buffer(16)]], packed_float4 device* objects_53 [[buffer(18)]], uint device* shot_bodies_53 [[buffer(20)]], float device* lines_53 [[buffer(19)]], packed_float4 device* geometry_53 [[buffer(21)]], packed_float4 device* shots_53 [[buffer(22)]], packed_float4 device* lp_53 [[buffer(24)]], packed_float4 device* ext_55 [[buffer(23)]], packed_float4 device* engine_contacts_53 [[buffer(25)]], CrushParams_0 constant* crush_53 [[buffer(28)]], packed_float4 device* crush_rw_53 [[buffer(27)]], packed_float4 device* crush_tab_53 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_170;
    (&kernelContext_170)->params_0 = params_53;
    (&kernelContext_170)->atom_0 = atom_53;
    (&kernelContext_170)->work_0 = work_53;
    (&kernelContext_170)->frames_0 = frames_53;
    (&kernelContext_170)->bodies_0 = bodies_53;
    (&kernelContext_170)->chunks_0 = chunks_53;
    (&kernelContext_170)->sort_0 = sort_53;
    (&kernelContext_170)->manifolds_0 = manifolds_53;
    (&kernelContext_170)->loads_0 = loads_53;
    (&kernelContext_170)->bonds_0 = bonds_53;
    (&kernelContext_170)->args_0 = args_53;
    (&kernelContext_170)->info_0 = info_53;
    (&kernelContext_170)->scan_0 = scan_53;
    (&kernelContext_170)->parts_0 = parts_53;
    (&kernelContext_170)->colliders_0 = colliders_53;
    (&kernelContext_170)->collider_info_0 = collider_info_53;
    (&kernelContext_170)->view_chunks_0 = view_chunks_53;
    (&kernelContext_170)->view_0 = view_53;
    (&kernelContext_170)->objects_0 = objects_53;
    (&kernelContext_170)->shot_bodies_0 = shot_bodies_53;
    (&kernelContext_170)->lines_0 = lines_53;
    (&kernelContext_170)->geometry_0 = geometry_53;
    (&kernelContext_170)->shots_0 = shots_53;
    (&kernelContext_170)->lp_0 = lp_53;
    (&kernelContext_170)->ext_0 = ext_55;
    (&kernelContext_170)->engine_contacts_0 = engine_contacts_53;
    (&kernelContext_170)->crush_0 = crush_53;
    (&kernelContext_170)->crush_rw_0 = crush_rw_53;
    (&kernelContext_170)->crush_tab_0 = crush_tab_53;
    threadgroup array<uint, int(1024)> g_scan_53;
    (&kernelContext_170)->g_scan_0 = &g_scan_53;
    threadgroup array<float, int(64)> g_il_best_53;
    (&kernelContext_170)->g_il_best_0 = &g_il_best_53;
    il_tube_collect_0(id_49.x, &kernelContext_170);
    return;
}

uint w_cs_list_0(KernelContext_0 thread* kernelContext_171)
{
    uint _S1190 = w_carry_0(kernelContext_171);
    return _S1190 + kernelContext_171->params_0->chunk_count_0;
}

uint w_cs_bodies_0(KernelContext_0 thread* kernelContext_172)
{
    uint _S1191 = w_cs_list_0(kernelContext_172);
    return _S1191 + kernelContext_172->params_0->chunk_count_0;
}

[[kernel]] void fr_csl_run(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_54 [[buffer(0)]], atomic<uint> device* atom_54 [[buffer(5)]], uint device* work_54 [[buffer(4)]], packed_float4 device* frames_54 [[buffer(6)]], packed_float4 device* bodies_54 [[buffer(7)]], packed_float4 device* chunks_54 [[buffer(1)]], uint device* sort_54 [[buffer(11)]], Record_natural_0 device* manifolds_54 [[buffer(10)]], packed_float4 device* loads_54 [[buffer(12)]], packed_uint4 device* bonds_54 [[buffer(2)]], uint device* args_54 [[buffer(13)]], uint device* info_54 [[buffer(15)]], ScanParams_0 constant* scan_54 [[buffer(14)]], packed_float4 device* parts_54 [[buffer(3)]], packed_float4 device* colliders_54 [[buffer(8)]], packed_uint4 device* collider_info_54 [[buffer(9)]], packed_float4 device* view_chunks_54 [[buffer(17)]], ViewParams_0 constant* view_54 [[buffer(16)]], packed_float4 device* objects_54 [[buffer(18)]], uint device* shot_bodies_54 [[buffer(20)]], float device* lines_54 [[buffer(19)]], packed_float4 device* geometry_54 [[buffer(21)]], packed_float4 device* shots_54 [[buffer(22)]], packed_float4 device* lp_54 [[buffer(24)]], packed_float4 device* ext_56 [[buffer(23)]], packed_float4 device* engine_contacts_54 [[buffer(25)]], CrushParams_0 constant* crush_54 [[buffer(28)]], packed_float4 device* crush_rw_54 [[buffer(27)]], packed_float4 device* crush_tab_54 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_173;
    (&kernelContext_173)->params_0 = params_54;
    (&kernelContext_173)->atom_0 = atom_54;
    (&kernelContext_173)->work_0 = work_54;
    (&kernelContext_173)->frames_0 = frames_54;
    (&kernelContext_173)->bodies_0 = bodies_54;
    (&kernelContext_173)->chunks_0 = chunks_54;
    (&kernelContext_173)->sort_0 = sort_54;
    (&kernelContext_173)->manifolds_0 = manifolds_54;
    (&kernelContext_173)->loads_0 = loads_54;
    (&kernelContext_173)->bonds_0 = bonds_54;
    (&kernelContext_173)->args_0 = args_54;
    (&kernelContext_173)->info_0 = info_54;
    (&kernelContext_173)->scan_0 = scan_54;
    (&kernelContext_173)->parts_0 = parts_54;
    (&kernelContext_173)->colliders_0 = colliders_54;
    (&kernelContext_173)->collider_info_0 = collider_info_54;
    (&kernelContext_173)->view_chunks_0 = view_chunks_54;
    (&kernelContext_173)->view_0 = view_54;
    (&kernelContext_173)->objects_0 = objects_54;
    (&kernelContext_173)->shot_bodies_0 = shot_bodies_54;
    (&kernelContext_173)->lines_0 = lines_54;
    (&kernelContext_173)->geometry_0 = geometry_54;
    (&kernelContext_173)->shots_0 = shots_54;
    (&kernelContext_173)->lp_0 = lp_54;
    (&kernelContext_173)->ext_0 = ext_56;
    (&kernelContext_173)->engine_contacts_0 = engine_contacts_54;
    (&kernelContext_173)->crush_0 = crush_54;
    (&kernelContext_173)->crush_rw_0 = crush_rw_54;
    (&kernelContext_173)->crush_tab_0 = crush_tab_54;
    threadgroup array<uint, int(1024)> g_scan_54;
    (&kernelContext_173)->g_scan_0 = &g_scan_54;
    threadgroup array<float, int(64)> g_il_best_54;
    (&kernelContext_173)->g_il_best_0 = &g_il_best_54;
    uint _S1192 = w_cs_bodies_0(&kernelContext_173);
    il_run_0(*((&kernelContext_173)->work_0+(_S1192 + group_2.x)), thread_5.x, &kernelContext_173);
    return;
}

[[kernel]] void fr_csl_strike(uint3 id_50 [[thread_position_in_grid]], FractureParams_0 constant* params_55 [[buffer(0)]], atomic<uint> device* atom_55 [[buffer(5)]], uint device* work_55 [[buffer(4)]], packed_float4 device* frames_55 [[buffer(6)]], packed_float4 device* bodies_55 [[buffer(7)]], packed_float4 device* chunks_55 [[buffer(1)]], uint device* sort_55 [[buffer(11)]], Record_natural_0 device* manifolds_55 [[buffer(10)]], packed_float4 device* loads_55 [[buffer(12)]], packed_uint4 device* bonds_55 [[buffer(2)]], uint device* args_55 [[buffer(13)]], uint device* info_55 [[buffer(15)]], ScanParams_0 constant* scan_55 [[buffer(14)]], packed_float4 device* parts_55 [[buffer(3)]], packed_float4 device* colliders_55 [[buffer(8)]], packed_uint4 device* collider_info_55 [[buffer(9)]], packed_float4 device* view_chunks_55 [[buffer(17)]], ViewParams_0 constant* view_55 [[buffer(16)]], packed_float4 device* objects_55 [[buffer(18)]], uint device* shot_bodies_55 [[buffer(20)]], float device* lines_55 [[buffer(19)]], packed_float4 device* geometry_55 [[buffer(21)]], packed_float4 device* shots_55 [[buffer(22)]], packed_float4 device* lp_55 [[buffer(24)]], packed_float4 device* ext_57 [[buffer(23)]], packed_float4 device* engine_contacts_55 [[buffer(25)]], CrushParams_0 constant* crush_55 [[buffer(28)]], packed_float4 device* crush_rw_55 [[buffer(27)]], packed_float4 device* crush_tab_55 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_174;
    (&kernelContext_174)->params_0 = params_55;
    (&kernelContext_174)->atom_0 = atom_55;
    (&kernelContext_174)->work_0 = work_55;
    (&kernelContext_174)->frames_0 = frames_55;
    (&kernelContext_174)->bodies_0 = bodies_55;
    (&kernelContext_174)->chunks_0 = chunks_55;
    (&kernelContext_174)->sort_0 = sort_55;
    (&kernelContext_174)->manifolds_0 = manifolds_55;
    (&kernelContext_174)->loads_0 = loads_55;
    (&kernelContext_174)->bonds_0 = bonds_55;
    (&kernelContext_174)->args_0 = args_55;
    (&kernelContext_174)->info_0 = info_55;
    (&kernelContext_174)->scan_0 = scan_55;
    (&kernelContext_174)->parts_0 = parts_55;
    (&kernelContext_174)->colliders_0 = colliders_55;
    (&kernelContext_174)->collider_info_0 = collider_info_55;
    (&kernelContext_174)->view_chunks_0 = view_chunks_55;
    (&kernelContext_174)->view_0 = view_55;
    (&kernelContext_174)->objects_0 = objects_55;
    (&kernelContext_174)->shot_bodies_0 = shot_bodies_55;
    (&kernelContext_174)->lines_0 = lines_55;
    (&kernelContext_174)->geometry_0 = geometry_55;
    (&kernelContext_174)->shots_0 = shots_55;
    (&kernelContext_174)->lp_0 = lp_55;
    (&kernelContext_174)->ext_0 = ext_57;
    (&kernelContext_174)->engine_contacts_0 = engine_contacts_55;
    (&kernelContext_174)->crush_0 = crush_55;
    (&kernelContext_174)->crush_rw_0 = crush_rw_55;
    (&kernelContext_174)->crush_tab_0 = crush_tab_55;
    threadgroup array<uint, int(1024)> g_scan_55;
    (&kernelContext_174)->g_scan_0 = &g_scan_55;
    threadgroup array<float, int(64)> g_il_best_55;
    (&kernelContext_174)->g_il_best_0 = &g_il_best_55;
    il_strike_0(id_50.x, &kernelContext_174);
    return;
}

[[kernel]] void fr_csl_mark(uint3 id_51 [[thread_position_in_grid]], FractureParams_0 constant* params_56 [[buffer(0)]], atomic<uint> device* atom_56 [[buffer(5)]], uint device* work_56 [[buffer(4)]], packed_float4 device* frames_56 [[buffer(6)]], packed_float4 device* bodies_56 [[buffer(7)]], packed_float4 device* chunks_56 [[buffer(1)]], uint device* sort_56 [[buffer(11)]], Record_natural_0 device* manifolds_56 [[buffer(10)]], packed_float4 device* loads_56 [[buffer(12)]], packed_uint4 device* bonds_56 [[buffer(2)]], uint device* args_56 [[buffer(13)]], uint device* info_56 [[buffer(15)]], ScanParams_0 constant* scan_56 [[buffer(14)]], packed_float4 device* parts_56 [[buffer(3)]], packed_float4 device* colliders_56 [[buffer(8)]], packed_uint4 device* collider_info_56 [[buffer(9)]], packed_float4 device* view_chunks_56 [[buffer(17)]], ViewParams_0 constant* view_56 [[buffer(16)]], packed_float4 device* objects_56 [[buffer(18)]], uint device* shot_bodies_56 [[buffer(20)]], float device* lines_56 [[buffer(19)]], packed_float4 device* geometry_56 [[buffer(21)]], packed_float4 device* shots_56 [[buffer(22)]], packed_float4 device* lp_56 [[buffer(24)]], packed_float4 device* ext_58 [[buffer(23)]], packed_float4 device* engine_contacts_56 [[buffer(25)]], CrushParams_0 constant* crush_56 [[buffer(28)]], packed_float4 device* crush_rw_56 [[buffer(27)]], packed_float4 device* crush_tab_56 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_175;
    (&kernelContext_175)->params_0 = params_56;
    (&kernelContext_175)->atom_0 = atom_56;
    (&kernelContext_175)->work_0 = work_56;
    (&kernelContext_175)->frames_0 = frames_56;
    (&kernelContext_175)->bodies_0 = bodies_56;
    (&kernelContext_175)->chunks_0 = chunks_56;
    (&kernelContext_175)->sort_0 = sort_56;
    (&kernelContext_175)->manifolds_0 = manifolds_56;
    (&kernelContext_175)->loads_0 = loads_56;
    (&kernelContext_175)->bonds_0 = bonds_56;
    (&kernelContext_175)->args_0 = args_56;
    (&kernelContext_175)->info_0 = info_56;
    (&kernelContext_175)->scan_0 = scan_56;
    (&kernelContext_175)->parts_0 = parts_56;
    (&kernelContext_175)->colliders_0 = colliders_56;
    (&kernelContext_175)->collider_info_0 = collider_info_56;
    (&kernelContext_175)->view_chunks_0 = view_chunks_56;
    (&kernelContext_175)->view_0 = view_56;
    (&kernelContext_175)->objects_0 = objects_56;
    (&kernelContext_175)->shot_bodies_0 = shot_bodies_56;
    (&kernelContext_175)->lines_0 = lines_56;
    (&kernelContext_175)->geometry_0 = geometry_56;
    (&kernelContext_175)->shots_0 = shots_56;
    (&kernelContext_175)->lp_0 = lp_56;
    (&kernelContext_175)->ext_0 = ext_58;
    (&kernelContext_175)->engine_contacts_0 = engine_contacts_56;
    (&kernelContext_175)->crush_0 = crush_56;
    (&kernelContext_175)->crush_rw_0 = crush_rw_56;
    (&kernelContext_175)->crush_tab_0 = crush_tab_56;
    threadgroup array<uint, int(1024)> g_scan_56;
    (&kernelContext_175)->g_scan_0 = &g_scan_56;
    threadgroup array<float, int(64)> g_il_best_56;
    (&kernelContext_175)->g_il_best_0 = &g_il_best_56;
    il_mark_0(id_51.x, &kernelContext_175);
    return;
}

[[kernel]] void fr_csl_carry(uint3 id_52 [[thread_position_in_grid]], FractureParams_0 constant* params_57 [[buffer(0)]], atomic<uint> device* atom_57 [[buffer(5)]], uint device* work_57 [[buffer(4)]], packed_float4 device* frames_57 [[buffer(6)]], packed_float4 device* bodies_57 [[buffer(7)]], packed_float4 device* chunks_57 [[buffer(1)]], uint device* sort_57 [[buffer(11)]], Record_natural_0 device* manifolds_57 [[buffer(10)]], packed_float4 device* loads_57 [[buffer(12)]], packed_uint4 device* bonds_57 [[buffer(2)]], uint device* args_57 [[buffer(13)]], uint device* info_57 [[buffer(15)]], ScanParams_0 constant* scan_57 [[buffer(14)]], packed_float4 device* parts_57 [[buffer(3)]], packed_float4 device* colliders_57 [[buffer(8)]], packed_uint4 device* collider_info_57 [[buffer(9)]], packed_float4 device* view_chunks_57 [[buffer(17)]], ViewParams_0 constant* view_57 [[buffer(16)]], packed_float4 device* objects_57 [[buffer(18)]], uint device* shot_bodies_57 [[buffer(20)]], float device* lines_57 [[buffer(19)]], packed_float4 device* geometry_57 [[buffer(21)]], packed_float4 device* shots_57 [[buffer(22)]], packed_float4 device* lp_57 [[buffer(24)]], packed_float4 device* ext_59 [[buffer(23)]], packed_float4 device* engine_contacts_57 [[buffer(25)]], CrushParams_0 constant* crush_57 [[buffer(28)]], packed_float4 device* crush_rw_57 [[buffer(27)]], packed_float4 device* crush_tab_57 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_176;
    (&kernelContext_176)->params_0 = params_57;
    (&kernelContext_176)->atom_0 = atom_57;
    (&kernelContext_176)->work_0 = work_57;
    (&kernelContext_176)->frames_0 = frames_57;
    (&kernelContext_176)->bodies_0 = bodies_57;
    (&kernelContext_176)->chunks_0 = chunks_57;
    (&kernelContext_176)->sort_0 = sort_57;
    (&kernelContext_176)->manifolds_0 = manifolds_57;
    (&kernelContext_176)->loads_0 = loads_57;
    (&kernelContext_176)->bonds_0 = bonds_57;
    (&kernelContext_176)->args_0 = args_57;
    (&kernelContext_176)->info_0 = info_57;
    (&kernelContext_176)->scan_0 = scan_57;
    (&kernelContext_176)->parts_0 = parts_57;
    (&kernelContext_176)->colliders_0 = colliders_57;
    (&kernelContext_176)->collider_info_0 = collider_info_57;
    (&kernelContext_176)->view_chunks_0 = view_chunks_57;
    (&kernelContext_176)->view_0 = view_57;
    (&kernelContext_176)->objects_0 = objects_57;
    (&kernelContext_176)->shot_bodies_0 = shot_bodies_57;
    (&kernelContext_176)->lines_0 = lines_57;
    (&kernelContext_176)->geometry_0 = geometry_57;
    (&kernelContext_176)->shots_0 = shots_57;
    (&kernelContext_176)->lp_0 = lp_57;
    (&kernelContext_176)->ext_0 = ext_59;
    (&kernelContext_176)->engine_contacts_0 = engine_contacts_57;
    (&kernelContext_176)->crush_0 = crush_57;
    (&kernelContext_176)->crush_rw_0 = crush_rw_57;
    (&kernelContext_176)->crush_tab_0 = crush_tab_57;
    threadgroup array<uint, int(1024)> g_scan_57;
    (&kernelContext_176)->g_scan_0 = &g_scan_57;
    threadgroup array<float, int(64)> g_il_best_57;
    (&kernelContext_176)->g_il_best_0 = &g_il_best_57;
    il_carry_0(id_52.x, &kernelContext_176);
    return;
}

[[kernel]] void fr_csl_breaks(uint3 id_53 [[thread_position_in_grid]], FractureParams_0 constant* params_58 [[buffer(0)]], atomic<uint> device* atom_58 [[buffer(5)]], uint device* work_58 [[buffer(4)]], packed_float4 device* frames_58 [[buffer(6)]], packed_float4 device* bodies_58 [[buffer(7)]], packed_float4 device* chunks_58 [[buffer(1)]], uint device* sort_58 [[buffer(11)]], Record_natural_0 device* manifolds_58 [[buffer(10)]], packed_float4 device* loads_58 [[buffer(12)]], packed_uint4 device* bonds_58 [[buffer(2)]], uint device* args_58 [[buffer(13)]], uint device* info_58 [[buffer(15)]], ScanParams_0 constant* scan_58 [[buffer(14)]], packed_float4 device* parts_58 [[buffer(3)]], packed_float4 device* colliders_58 [[buffer(8)]], packed_uint4 device* collider_info_58 [[buffer(9)]], packed_float4 device* view_chunks_58 [[buffer(17)]], ViewParams_0 constant* view_58 [[buffer(16)]], packed_float4 device* objects_58 [[buffer(18)]], uint device* shot_bodies_58 [[buffer(20)]], float device* lines_58 [[buffer(19)]], packed_float4 device* geometry_58 [[buffer(21)]], packed_float4 device* shots_58 [[buffer(22)]], packed_float4 device* lp_58 [[buffer(24)]], packed_float4 device* ext_60 [[buffer(23)]], packed_float4 device* engine_contacts_58 [[buffer(25)]], CrushParams_0 constant* crush_58 [[buffer(28)]], packed_float4 device* crush_rw_58 [[buffer(27)]], packed_float4 device* crush_tab_58 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_177;
    (&kernelContext_177)->params_0 = params_58;
    (&kernelContext_177)->atom_0 = atom_58;
    (&kernelContext_177)->work_0 = work_58;
    (&kernelContext_177)->frames_0 = frames_58;
    (&kernelContext_177)->bodies_0 = bodies_58;
    (&kernelContext_177)->chunks_0 = chunks_58;
    (&kernelContext_177)->sort_0 = sort_58;
    (&kernelContext_177)->manifolds_0 = manifolds_58;
    (&kernelContext_177)->loads_0 = loads_58;
    (&kernelContext_177)->bonds_0 = bonds_58;
    (&kernelContext_177)->args_0 = args_58;
    (&kernelContext_177)->info_0 = info_58;
    (&kernelContext_177)->scan_0 = scan_58;
    (&kernelContext_177)->parts_0 = parts_58;
    (&kernelContext_177)->colliders_0 = colliders_58;
    (&kernelContext_177)->collider_info_0 = collider_info_58;
    (&kernelContext_177)->view_chunks_0 = view_chunks_58;
    (&kernelContext_177)->view_0 = view_58;
    (&kernelContext_177)->objects_0 = objects_58;
    (&kernelContext_177)->shot_bodies_0 = shot_bodies_58;
    (&kernelContext_177)->lines_0 = lines_58;
    (&kernelContext_177)->geometry_0 = geometry_58;
    (&kernelContext_177)->shots_0 = shots_58;
    (&kernelContext_177)->lp_0 = lp_58;
    (&kernelContext_177)->ext_0 = ext_60;
    (&kernelContext_177)->engine_contacts_0 = engine_contacts_58;
    (&kernelContext_177)->crush_0 = crush_58;
    (&kernelContext_177)->crush_rw_0 = crush_rw_58;
    (&kernelContext_177)->crush_tab_0 = crush_tab_58;
    threadgroup array<uint, int(1024)> g_scan_58;
    (&kernelContext_177)->g_scan_0 = &g_scan_58;
    threadgroup array<float, int(64)> g_il_best_58;
    (&kernelContext_177)->g_il_best_0 = &g_il_best_58;
    il_breaks_0(id_53.x, &kernelContext_177);
    return;
}

[[kernel]] void fr_cs_begin(uint3 id_54 [[thread_position_in_grid]], FractureParams_0 constant* params_59 [[buffer(0)]], atomic<uint> device* atom_59 [[buffer(5)]], uint device* work_59 [[buffer(4)]], packed_float4 device* frames_59 [[buffer(6)]], packed_float4 device* bodies_59 [[buffer(7)]], packed_float4 device* chunks_59 [[buffer(1)]], uint device* sort_59 [[buffer(11)]], Record_natural_0 device* manifolds_59 [[buffer(10)]], packed_float4 device* loads_59 [[buffer(12)]], packed_uint4 device* bonds_59 [[buffer(2)]], uint device* args_59 [[buffer(13)]], uint device* info_59 [[buffer(15)]], ScanParams_0 constant* scan_59 [[buffer(14)]], packed_float4 device* parts_59 [[buffer(3)]], packed_float4 device* colliders_59 [[buffer(8)]], packed_uint4 device* collider_info_59 [[buffer(9)]], packed_float4 device* view_chunks_59 [[buffer(17)]], ViewParams_0 constant* view_59 [[buffer(16)]], packed_float4 device* objects_59 [[buffer(18)]], uint device* shot_bodies_59 [[buffer(20)]], float device* lines_59 [[buffer(19)]], packed_float4 device* geometry_59 [[buffer(21)]], packed_float4 device* shots_59 [[buffer(22)]], packed_float4 device* lp_59 [[buffer(24)]], packed_float4 device* ext_61 [[buffer(23)]], packed_float4 device* engine_contacts_59 [[buffer(25)]], CrushParams_0 constant* crush_59 [[buffer(28)]], packed_float4 device* crush_rw_59 [[buffer(27)]], packed_float4 device* crush_tab_59 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_178;
    (&kernelContext_178)->params_0 = params_59;
    (&kernelContext_178)->atom_0 = atom_59;
    (&kernelContext_178)->work_0 = work_59;
    (&kernelContext_178)->frames_0 = frames_59;
    (&kernelContext_178)->bodies_0 = bodies_59;
    (&kernelContext_178)->chunks_0 = chunks_59;
    (&kernelContext_178)->sort_0 = sort_59;
    (&kernelContext_178)->manifolds_0 = manifolds_59;
    (&kernelContext_178)->loads_0 = loads_59;
    (&kernelContext_178)->bonds_0 = bonds_59;
    (&kernelContext_178)->args_0 = args_59;
    (&kernelContext_178)->info_0 = info_59;
    (&kernelContext_178)->scan_0 = scan_59;
    (&kernelContext_178)->parts_0 = parts_59;
    (&kernelContext_178)->colliders_0 = colliders_59;
    (&kernelContext_178)->collider_info_0 = collider_info_59;
    (&kernelContext_178)->view_chunks_0 = view_chunks_59;
    (&kernelContext_178)->view_0 = view_59;
    (&kernelContext_178)->objects_0 = objects_59;
    (&kernelContext_178)->shot_bodies_0 = shot_bodies_59;
    (&kernelContext_178)->lines_0 = lines_59;
    (&kernelContext_178)->geometry_0 = geometry_59;
    (&kernelContext_178)->shots_0 = shots_59;
    (&kernelContext_178)->lp_0 = lp_59;
    (&kernelContext_178)->ext_0 = ext_61;
    (&kernelContext_178)->engine_contacts_0 = engine_contacts_59;
    (&kernelContext_178)->crush_0 = crush_59;
    (&kernelContext_178)->crush_rw_0 = crush_rw_59;
    (&kernelContext_178)->crush_tab_0 = crush_tab_59;
    threadgroup array<uint, int(1024)> g_scan_59;
    (&kernelContext_178)->g_scan_0 = &g_scan_59;
    threadgroup array<float, int(64)> g_il_best_59;
    (&kernelContext_178)->g_il_best_0 = &g_il_best_59;
    uint _S1193 = a_counters_1(&kernelContext_178);
    atomic_store_explicit((&kernelContext_178)->atom_0+(_S1193 + 18U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_178)->atom_0+(_S1193 + 19U), 0U, memory_order_relaxed);
    uint _S1194 = a_il_cursor_1(&kernelContext_178);
    atomic_store_explicit((&kernelContext_178)->atom_0+_S1194, 0U, memory_order_relaxed);
    uint _S1195 = cs_rows_0(&kernelContext_178);
    uint _S1196 = groups_of_0(_S1195, 64U);
    set_args_0(7U, _S1196, &kernelContext_178);
    return;
}

[[kernel]] void fr_cs_contacts(uint3 id_55 [[thread_position_in_grid]], FractureParams_0 constant* params_60 [[buffer(0)]], atomic<uint> device* atom_60 [[buffer(5)]], uint device* work_60 [[buffer(4)]], packed_float4 device* frames_60 [[buffer(6)]], packed_float4 device* bodies_60 [[buffer(7)]], packed_float4 device* chunks_60 [[buffer(1)]], uint device* sort_60 [[buffer(11)]], Record_natural_0 device* manifolds_60 [[buffer(10)]], packed_float4 device* loads_60 [[buffer(12)]], packed_uint4 device* bonds_60 [[buffer(2)]], uint device* args_60 [[buffer(13)]], uint device* info_60 [[buffer(15)]], ScanParams_0 constant* scan_60 [[buffer(14)]], packed_float4 device* parts_60 [[buffer(3)]], packed_float4 device* colliders_60 [[buffer(8)]], packed_uint4 device* collider_info_60 [[buffer(9)]], packed_float4 device* view_chunks_60 [[buffer(17)]], ViewParams_0 constant* view_60 [[buffer(16)]], packed_float4 device* objects_60 [[buffer(18)]], uint device* shot_bodies_60 [[buffer(20)]], float device* lines_60 [[buffer(19)]], packed_float4 device* geometry_60 [[buffer(21)]], packed_float4 device* shots_60 [[buffer(22)]], packed_float4 device* lp_60 [[buffer(24)]], packed_float4 device* ext_62 [[buffer(23)]], packed_float4 device* engine_contacts_60 [[buffer(25)]], CrushParams_0 constant* crush_60 [[buffer(28)]], packed_float4 device* crush_rw_60 [[buffer(27)]], packed_float4 device* crush_tab_60 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_179;
    (&kernelContext_179)->params_0 = params_60;
    (&kernelContext_179)->atom_0 = atom_60;
    (&kernelContext_179)->work_0 = work_60;
    (&kernelContext_179)->frames_0 = frames_60;
    (&kernelContext_179)->bodies_0 = bodies_60;
    (&kernelContext_179)->chunks_0 = chunks_60;
    (&kernelContext_179)->sort_0 = sort_60;
    (&kernelContext_179)->manifolds_0 = manifolds_60;
    (&kernelContext_179)->loads_0 = loads_60;
    (&kernelContext_179)->bonds_0 = bonds_60;
    (&kernelContext_179)->args_0 = args_60;
    (&kernelContext_179)->info_0 = info_60;
    (&kernelContext_179)->scan_0 = scan_60;
    (&kernelContext_179)->parts_0 = parts_60;
    (&kernelContext_179)->colliders_0 = colliders_60;
    (&kernelContext_179)->collider_info_0 = collider_info_60;
    (&kernelContext_179)->view_chunks_0 = view_chunks_60;
    (&kernelContext_179)->view_0 = view_60;
    (&kernelContext_179)->objects_0 = objects_60;
    (&kernelContext_179)->shot_bodies_0 = shot_bodies_60;
    (&kernelContext_179)->lines_0 = lines_60;
    (&kernelContext_179)->geometry_0 = geometry_60;
    (&kernelContext_179)->shots_0 = shots_60;
    (&kernelContext_179)->lp_0 = lp_60;
    (&kernelContext_179)->ext_0 = ext_62;
    (&kernelContext_179)->engine_contacts_0 = engine_contacts_60;
    (&kernelContext_179)->crush_0 = crush_60;
    (&kernelContext_179)->crush_rw_0 = crush_rw_60;
    (&kernelContext_179)->crush_tab_0 = crush_tab_60;
    threadgroup array<uint, int(1024)> g_scan_60;
    (&kernelContext_179)->g_scan_0 = &g_scan_60;
    threadgroup array<float, int(64)> g_il_best_60;
    (&kernelContext_179)->g_il_best_0 = &g_il_best_60;
    uint r_24 = id_55.x;
    uint _S1197 = cs_rows_0(&kernelContext_179);
    if(r_24 >= _S1197)
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
        bool _S1198 = cs_side_0(r_24, side_6, &k_44, &s_17, &ob_5, &co_5, &p_18, &u_9, &v_14, &m_eff_7, &t_in_8, &kernelContext_179);
        if(_S1198)
        {
            uint _S1199 = a_cs_0(s_17, &kernelContext_179);
            uint _S1200 = atomic_fetch_max_explicit((&kernelContext_179)->atom_0+_S1199, (as_type<uint>((m_eff_7 * v_14))), memory_order_relaxed);
            uint _S1201 = a_cs_0(s_17, &kernelContext_179);
            uint _S1202 = atomic_exchange_explicit((&kernelContext_179)->atom_0+(_S1201 + 8U), 1U, memory_order_relaxed);
            if(_S1202 == 0U)
            {
                uint _S1203 = w_cs_bodies_0(&kernelContext_179);
                uint _S1204 = a_counters_1(&kernelContext_179);
                uint _S1205 = atomic_fetch_add_explicit((&kernelContext_179)->atom_0+(_S1204 + 18U), 1U, memory_order_relaxed);
                *((&kernelContext_179)->work_0+(_S1203 + _S1205)) = s_17;
            }
        }
        side_6 = side_6 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_gate(uint3 id_56 [[thread_position_in_grid]], FractureParams_0 constant* params_61 [[buffer(0)]], atomic<uint> device* atom_61 [[buffer(5)]], uint device* work_61 [[buffer(4)]], packed_float4 device* frames_61 [[buffer(6)]], packed_float4 device* bodies_61 [[buffer(7)]], packed_float4 device* chunks_61 [[buffer(1)]], uint device* sort_61 [[buffer(11)]], Record_natural_0 device* manifolds_61 [[buffer(10)]], packed_float4 device* loads_61 [[buffer(12)]], packed_uint4 device* bonds_61 [[buffer(2)]], uint device* args_61 [[buffer(13)]], uint device* info_61 [[buffer(15)]], ScanParams_0 constant* scan_61 [[buffer(14)]], packed_float4 device* parts_61 [[buffer(3)]], packed_float4 device* colliders_61 [[buffer(8)]], packed_uint4 device* collider_info_61 [[buffer(9)]], packed_float4 device* view_chunks_61 [[buffer(17)]], ViewParams_0 constant* view_61 [[buffer(16)]], packed_float4 device* objects_61 [[buffer(18)]], uint device* shot_bodies_61 [[buffer(20)]], float device* lines_61 [[buffer(19)]], packed_float4 device* geometry_61 [[buffer(21)]], packed_float4 device* shots_61 [[buffer(22)]], packed_float4 device* lp_61 [[buffer(24)]], packed_float4 device* ext_63 [[buffer(23)]], packed_float4 device* engine_contacts_61 [[buffer(25)]], CrushParams_0 constant* crush_61 [[buffer(28)]], packed_float4 device* crush_rw_61 [[buffer(27)]], packed_float4 device* crush_tab_61 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_180;
    (&kernelContext_180)->params_0 = params_61;
    (&kernelContext_180)->atom_0 = atom_61;
    (&kernelContext_180)->work_0 = work_61;
    (&kernelContext_180)->frames_0 = frames_61;
    (&kernelContext_180)->bodies_0 = bodies_61;
    (&kernelContext_180)->chunks_0 = chunks_61;
    (&kernelContext_180)->sort_0 = sort_61;
    (&kernelContext_180)->manifolds_0 = manifolds_61;
    (&kernelContext_180)->loads_0 = loads_61;
    (&kernelContext_180)->bonds_0 = bonds_61;
    (&kernelContext_180)->args_0 = args_61;
    (&kernelContext_180)->info_0 = info_61;
    (&kernelContext_180)->scan_0 = scan_61;
    (&kernelContext_180)->parts_0 = parts_61;
    (&kernelContext_180)->colliders_0 = colliders_61;
    (&kernelContext_180)->collider_info_0 = collider_info_61;
    (&kernelContext_180)->view_chunks_0 = view_chunks_61;
    (&kernelContext_180)->view_0 = view_61;
    (&kernelContext_180)->objects_0 = objects_61;
    (&kernelContext_180)->shot_bodies_0 = shot_bodies_61;
    (&kernelContext_180)->lines_0 = lines_61;
    (&kernelContext_180)->geometry_0 = geometry_61;
    (&kernelContext_180)->shots_0 = shots_61;
    (&kernelContext_180)->lp_0 = lp_61;
    (&kernelContext_180)->ext_0 = ext_63;
    (&kernelContext_180)->engine_contacts_0 = engine_contacts_61;
    (&kernelContext_180)->crush_0 = crush_61;
    (&kernelContext_180)->crush_rw_0 = crush_rw_61;
    (&kernelContext_180)->crush_tab_0 = crush_tab_61;
    threadgroup array<uint, int(1024)> g_scan_61;
    (&kernelContext_180)->g_scan_0 = &g_scan_61;
    threadgroup array<float, int(64)> g_il_best_61;
    (&kernelContext_180)->g_il_best_0 = &g_il_best_61;
    uint _S1206 = a_counters_1(&kernelContext_180);
    uint struck_1 = atomic_load_explicit((&kernelContext_180)->atom_0+(_S1206 + 18U), memory_order_relaxed);
    bool any_0 = struck_1 > 0U;
    uint _S1207;
    if(any_0)
    {
        uint _S1208 = cs_rows_0(&kernelContext_180);
        uint _S1209 = groups_of_0(_S1208, 64U);
        _S1207 = _S1209;
    }
    else
    {
        _S1207 = 0U;
    }
    set_args_0(8U, _S1207, &kernelContext_180);
    uint _S1210 = groups_of_0(struck_1, 64U);
    set_args_0(9U, _S1210, &kernelContext_180);
    set_args_0(6U, struck_1, &kernelContext_180);
    if(any_0)
    {
        uint _S1211 = groups_of_0((&kernelContext_180)->params_0->chunk_count_0, 64U);
        _S1207 = _S1211;
    }
    else
    {
        _S1207 = 0U;
    }
    set_args_0(10U, _S1207, &kernelContext_180);
    if(any_0)
    {
        uint _S1212 = groups_of_0((&kernelContext_180)->params_0->bond_count_0, 64U);
        _S1207 = _S1212;
    }
    else
    {
        _S1207 = 0U;
    }
    set_args_0(11U, _S1207, &kernelContext_180);
    if(any_0)
    {
        uint _S1213 = groups_of_0((&kernelContext_180)->params_0->body_count_0, 64U);
        _S1207 = _S1213;
    }
    else
    {
        _S1207 = 0U;
    }
    set_args_0(12U, _S1207, &kernelContext_180);
    return;
}

[[kernel]] void fr_cs_pick_a(uint3 id_57 [[thread_position_in_grid]], FractureParams_0 constant* params_62 [[buffer(0)]], atomic<uint> device* atom_62 [[buffer(5)]], uint device* work_62 [[buffer(4)]], packed_float4 device* frames_62 [[buffer(6)]], packed_float4 device* bodies_62 [[buffer(7)]], packed_float4 device* chunks_62 [[buffer(1)]], uint device* sort_62 [[buffer(11)]], Record_natural_0 device* manifolds_62 [[buffer(10)]], packed_float4 device* loads_62 [[buffer(12)]], packed_uint4 device* bonds_62 [[buffer(2)]], uint device* args_62 [[buffer(13)]], uint device* info_62 [[buffer(15)]], ScanParams_0 constant* scan_62 [[buffer(14)]], packed_float4 device* parts_62 [[buffer(3)]], packed_float4 device* colliders_62 [[buffer(8)]], packed_uint4 device* collider_info_62 [[buffer(9)]], packed_float4 device* view_chunks_62 [[buffer(17)]], ViewParams_0 constant* view_62 [[buffer(16)]], packed_float4 device* objects_62 [[buffer(18)]], uint device* shot_bodies_62 [[buffer(20)]], float device* lines_62 [[buffer(19)]], packed_float4 device* geometry_62 [[buffer(21)]], packed_float4 device* shots_62 [[buffer(22)]], packed_float4 device* lp_62 [[buffer(24)]], packed_float4 device* ext_64 [[buffer(23)]], packed_float4 device* engine_contacts_62 [[buffer(25)]], CrushParams_0 constant* crush_62 [[buffer(28)]], packed_float4 device* crush_rw_62 [[buffer(27)]], packed_float4 device* crush_tab_62 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_181;
    (&kernelContext_181)->params_0 = params_62;
    (&kernelContext_181)->atom_0 = atom_62;
    (&kernelContext_181)->work_0 = work_62;
    (&kernelContext_181)->frames_0 = frames_62;
    (&kernelContext_181)->bodies_0 = bodies_62;
    (&kernelContext_181)->chunks_0 = chunks_62;
    (&kernelContext_181)->sort_0 = sort_62;
    (&kernelContext_181)->manifolds_0 = manifolds_62;
    (&kernelContext_181)->loads_0 = loads_62;
    (&kernelContext_181)->bonds_0 = bonds_62;
    (&kernelContext_181)->args_0 = args_62;
    (&kernelContext_181)->info_0 = info_62;
    (&kernelContext_181)->scan_0 = scan_62;
    (&kernelContext_181)->parts_0 = parts_62;
    (&kernelContext_181)->colliders_0 = colliders_62;
    (&kernelContext_181)->collider_info_0 = collider_info_62;
    (&kernelContext_181)->view_chunks_0 = view_chunks_62;
    (&kernelContext_181)->view_0 = view_62;
    (&kernelContext_181)->objects_0 = objects_62;
    (&kernelContext_181)->shot_bodies_0 = shot_bodies_62;
    (&kernelContext_181)->lines_0 = lines_62;
    (&kernelContext_181)->geometry_0 = geometry_62;
    (&kernelContext_181)->shots_0 = shots_62;
    (&kernelContext_181)->lp_0 = lp_62;
    (&kernelContext_181)->ext_0 = ext_64;
    (&kernelContext_181)->engine_contacts_0 = engine_contacts_62;
    (&kernelContext_181)->crush_0 = crush_62;
    (&kernelContext_181)->crush_rw_0 = crush_rw_62;
    (&kernelContext_181)->crush_tab_0 = crush_tab_62;
    threadgroup array<uint, int(1024)> g_scan_62;
    (&kernelContext_181)->g_scan_0 = &g_scan_62;
    threadgroup array<float, int(64)> g_il_best_62;
    (&kernelContext_181)->g_il_best_0 = &g_il_best_62;
    cs_pick_3(id_57.x, &kernelContext_181);
    return;
}

[[kernel]] void fr_cs_pick_b(uint3 id_58 [[thread_position_in_grid]], FractureParams_0 constant* params_63 [[buffer(0)]], atomic<uint> device* atom_63 [[buffer(5)]], uint device* work_63 [[buffer(4)]], packed_float4 device* frames_63 [[buffer(6)]], packed_float4 device* bodies_63 [[buffer(7)]], packed_float4 device* chunks_63 [[buffer(1)]], uint device* sort_63 [[buffer(11)]], Record_natural_0 device* manifolds_63 [[buffer(10)]], packed_float4 device* loads_63 [[buffer(12)]], packed_uint4 device* bonds_63 [[buffer(2)]], uint device* args_63 [[buffer(13)]], uint device* info_63 [[buffer(15)]], ScanParams_0 constant* scan_63 [[buffer(14)]], packed_float4 device* parts_63 [[buffer(3)]], packed_float4 device* colliders_63 [[buffer(8)]], packed_uint4 device* collider_info_63 [[buffer(9)]], packed_float4 device* view_chunks_63 [[buffer(17)]], ViewParams_0 constant* view_63 [[buffer(16)]], packed_float4 device* objects_63 [[buffer(18)]], uint device* shot_bodies_63 [[buffer(20)]], float device* lines_63 [[buffer(19)]], packed_float4 device* geometry_63 [[buffer(21)]], packed_float4 device* shots_63 [[buffer(22)]], packed_float4 device* lp_63 [[buffer(24)]], packed_float4 device* ext_65 [[buffer(23)]], packed_float4 device* engine_contacts_63 [[buffer(25)]], CrushParams_0 constant* crush_63 [[buffer(28)]], packed_float4 device* crush_rw_63 [[buffer(27)]], packed_float4 device* crush_tab_63 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_182;
    (&kernelContext_182)->params_0 = params_63;
    (&kernelContext_182)->atom_0 = atom_63;
    (&kernelContext_182)->work_0 = work_63;
    (&kernelContext_182)->frames_0 = frames_63;
    (&kernelContext_182)->bodies_0 = bodies_63;
    (&kernelContext_182)->chunks_0 = chunks_63;
    (&kernelContext_182)->sort_0 = sort_63;
    (&kernelContext_182)->manifolds_0 = manifolds_63;
    (&kernelContext_182)->loads_0 = loads_63;
    (&kernelContext_182)->bonds_0 = bonds_63;
    (&kernelContext_182)->args_0 = args_63;
    (&kernelContext_182)->info_0 = info_63;
    (&kernelContext_182)->scan_0 = scan_63;
    (&kernelContext_182)->parts_0 = parts_63;
    (&kernelContext_182)->colliders_0 = colliders_63;
    (&kernelContext_182)->collider_info_0 = collider_info_63;
    (&kernelContext_182)->view_chunks_0 = view_chunks_63;
    (&kernelContext_182)->view_0 = view_63;
    (&kernelContext_182)->objects_0 = objects_63;
    (&kernelContext_182)->shot_bodies_0 = shot_bodies_63;
    (&kernelContext_182)->lines_0 = lines_63;
    (&kernelContext_182)->geometry_0 = geometry_63;
    (&kernelContext_182)->shots_0 = shots_63;
    (&kernelContext_182)->lp_0 = lp_63;
    (&kernelContext_182)->ext_0 = ext_65;
    (&kernelContext_182)->engine_contacts_0 = engine_contacts_63;
    (&kernelContext_182)->crush_0 = crush_63;
    (&kernelContext_182)->crush_rw_0 = crush_rw_63;
    (&kernelContext_182)->crush_tab_0 = crush_tab_63;
    threadgroup array<uint, int(1024)> g_scan_63;
    (&kernelContext_182)->g_scan_0 = &g_scan_63;
    threadgroup array<float, int(64)> g_il_best_63;
    (&kernelContext_182)->g_il_best_0 = &g_il_best_63;
    cs_pick_2(id_58.x, &kernelContext_182);
    return;
}

[[kernel]] void fr_cs_pick_c(uint3 id_59 [[thread_position_in_grid]], FractureParams_0 constant* params_64 [[buffer(0)]], atomic<uint> device* atom_64 [[buffer(5)]], uint device* work_64 [[buffer(4)]], packed_float4 device* frames_64 [[buffer(6)]], packed_float4 device* bodies_64 [[buffer(7)]], packed_float4 device* chunks_64 [[buffer(1)]], uint device* sort_64 [[buffer(11)]], Record_natural_0 device* manifolds_64 [[buffer(10)]], packed_float4 device* loads_64 [[buffer(12)]], packed_uint4 device* bonds_64 [[buffer(2)]], uint device* args_64 [[buffer(13)]], uint device* info_64 [[buffer(15)]], ScanParams_0 constant* scan_64 [[buffer(14)]], packed_float4 device* parts_64 [[buffer(3)]], packed_float4 device* colliders_64 [[buffer(8)]], packed_uint4 device* collider_info_64 [[buffer(9)]], packed_float4 device* view_chunks_64 [[buffer(17)]], ViewParams_0 constant* view_64 [[buffer(16)]], packed_float4 device* objects_64 [[buffer(18)]], uint device* shot_bodies_64 [[buffer(20)]], float device* lines_64 [[buffer(19)]], packed_float4 device* geometry_64 [[buffer(21)]], packed_float4 device* shots_64 [[buffer(22)]], packed_float4 device* lp_64 [[buffer(24)]], packed_float4 device* ext_66 [[buffer(23)]], packed_float4 device* engine_contacts_64 [[buffer(25)]], CrushParams_0 constant* crush_64 [[buffer(28)]], packed_float4 device* crush_rw_64 [[buffer(27)]], packed_float4 device* crush_tab_64 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_183;
    (&kernelContext_183)->params_0 = params_64;
    (&kernelContext_183)->atom_0 = atom_64;
    (&kernelContext_183)->work_0 = work_64;
    (&kernelContext_183)->frames_0 = frames_64;
    (&kernelContext_183)->bodies_0 = bodies_64;
    (&kernelContext_183)->chunks_0 = chunks_64;
    (&kernelContext_183)->sort_0 = sort_64;
    (&kernelContext_183)->manifolds_0 = manifolds_64;
    (&kernelContext_183)->loads_0 = loads_64;
    (&kernelContext_183)->bonds_0 = bonds_64;
    (&kernelContext_183)->args_0 = args_64;
    (&kernelContext_183)->info_0 = info_64;
    (&kernelContext_183)->scan_0 = scan_64;
    (&kernelContext_183)->parts_0 = parts_64;
    (&kernelContext_183)->colliders_0 = colliders_64;
    (&kernelContext_183)->collider_info_0 = collider_info_64;
    (&kernelContext_183)->view_chunks_0 = view_chunks_64;
    (&kernelContext_183)->view_0 = view_64;
    (&kernelContext_183)->objects_0 = objects_64;
    (&kernelContext_183)->shot_bodies_0 = shot_bodies_64;
    (&kernelContext_183)->lines_0 = lines_64;
    (&kernelContext_183)->geometry_0 = geometry_64;
    (&kernelContext_183)->shots_0 = shots_64;
    (&kernelContext_183)->lp_0 = lp_64;
    (&kernelContext_183)->ext_0 = ext_66;
    (&kernelContext_183)->engine_contacts_0 = engine_contacts_64;
    (&kernelContext_183)->crush_0 = crush_64;
    (&kernelContext_183)->crush_rw_0 = crush_rw_64;
    (&kernelContext_183)->crush_tab_0 = crush_tab_64;
    threadgroup array<uint, int(1024)> g_scan_64;
    (&kernelContext_183)->g_scan_0 = &g_scan_64;
    threadgroup array<float, int(64)> g_il_best_64;
    (&kernelContext_183)->g_il_best_0 = &g_il_best_64;
    cs_pick_1(id_59.x, &kernelContext_183);
    return;
}

[[kernel]] void fr_cs_pick_d(uint3 id_60 [[thread_position_in_grid]], FractureParams_0 constant* params_65 [[buffer(0)]], atomic<uint> device* atom_65 [[buffer(5)]], uint device* work_65 [[buffer(4)]], packed_float4 device* frames_65 [[buffer(6)]], packed_float4 device* bodies_65 [[buffer(7)]], packed_float4 device* chunks_65 [[buffer(1)]], uint device* sort_65 [[buffer(11)]], Record_natural_0 device* manifolds_65 [[buffer(10)]], packed_float4 device* loads_65 [[buffer(12)]], packed_uint4 device* bonds_65 [[buffer(2)]], uint device* args_65 [[buffer(13)]], uint device* info_65 [[buffer(15)]], ScanParams_0 constant* scan_65 [[buffer(14)]], packed_float4 device* parts_65 [[buffer(3)]], packed_float4 device* colliders_65 [[buffer(8)]], packed_uint4 device* collider_info_65 [[buffer(9)]], packed_float4 device* view_chunks_65 [[buffer(17)]], ViewParams_0 constant* view_65 [[buffer(16)]], packed_float4 device* objects_65 [[buffer(18)]], uint device* shot_bodies_65 [[buffer(20)]], float device* lines_65 [[buffer(19)]], packed_float4 device* geometry_65 [[buffer(21)]], packed_float4 device* shots_65 [[buffer(22)]], packed_float4 device* lp_65 [[buffer(24)]], packed_float4 device* ext_67 [[buffer(23)]], packed_float4 device* engine_contacts_65 [[buffer(25)]], CrushParams_0 constant* crush_65 [[buffer(28)]], packed_float4 device* crush_rw_65 [[buffer(27)]], packed_float4 device* crush_tab_65 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_184;
    (&kernelContext_184)->params_0 = params_65;
    (&kernelContext_184)->atom_0 = atom_65;
    (&kernelContext_184)->work_0 = work_65;
    (&kernelContext_184)->frames_0 = frames_65;
    (&kernelContext_184)->bodies_0 = bodies_65;
    (&kernelContext_184)->chunks_0 = chunks_65;
    (&kernelContext_184)->sort_0 = sort_65;
    (&kernelContext_184)->manifolds_0 = manifolds_65;
    (&kernelContext_184)->loads_0 = loads_65;
    (&kernelContext_184)->bonds_0 = bonds_65;
    (&kernelContext_184)->args_0 = args_65;
    (&kernelContext_184)->info_0 = info_65;
    (&kernelContext_184)->scan_0 = scan_65;
    (&kernelContext_184)->parts_0 = parts_65;
    (&kernelContext_184)->colliders_0 = colliders_65;
    (&kernelContext_184)->collider_info_0 = collider_info_65;
    (&kernelContext_184)->view_chunks_0 = view_chunks_65;
    (&kernelContext_184)->view_0 = view_65;
    (&kernelContext_184)->objects_0 = objects_65;
    (&kernelContext_184)->shot_bodies_0 = shot_bodies_65;
    (&kernelContext_184)->lines_0 = lines_65;
    (&kernelContext_184)->geometry_0 = geometry_65;
    (&kernelContext_184)->shots_0 = shots_65;
    (&kernelContext_184)->lp_0 = lp_65;
    (&kernelContext_184)->ext_0 = ext_67;
    (&kernelContext_184)->engine_contacts_0 = engine_contacts_65;
    (&kernelContext_184)->crush_0 = crush_65;
    (&kernelContext_184)->crush_rw_0 = crush_rw_65;
    (&kernelContext_184)->crush_tab_0 = crush_tab_65;
    threadgroup array<uint, int(1024)> g_scan_65;
    (&kernelContext_184)->g_scan_0 = &g_scan_65;
    threadgroup array<float, int(64)> g_il_best_65;
    (&kernelContext_184)->g_il_best_0 = &g_il_best_65;
    cs_pick_0(id_60.x, &kernelContext_184);
    return;
}

[[kernel]] void fr_cs_record(uint3 id_61 [[thread_position_in_grid]], FractureParams_0 constant* params_66 [[buffer(0)]], atomic<uint> device* atom_66 [[buffer(5)]], uint device* work_66 [[buffer(4)]], packed_float4 device* frames_66 [[buffer(6)]], packed_float4 device* bodies_66 [[buffer(7)]], packed_float4 device* chunks_66 [[buffer(1)]], uint device* sort_66 [[buffer(11)]], Record_natural_0 device* manifolds_66 [[buffer(10)]], packed_float4 device* loads_66 [[buffer(12)]], packed_uint4 device* bonds_66 [[buffer(2)]], uint device* args_66 [[buffer(13)]], uint device* info_66 [[buffer(15)]], ScanParams_0 constant* scan_66 [[buffer(14)]], packed_float4 device* parts_66 [[buffer(3)]], packed_float4 device* colliders_66 [[buffer(8)]], packed_uint4 device* collider_info_66 [[buffer(9)]], packed_float4 device* view_chunks_66 [[buffer(17)]], ViewParams_0 constant* view_66 [[buffer(16)]], packed_float4 device* objects_66 [[buffer(18)]], uint device* shot_bodies_66 [[buffer(20)]], float device* lines_66 [[buffer(19)]], packed_float4 device* geometry_66 [[buffer(21)]], packed_float4 device* shots_66 [[buffer(22)]], packed_float4 device* lp_66 [[buffer(24)]], packed_float4 device* ext_68 [[buffer(23)]], packed_float4 device* engine_contacts_66 [[buffer(25)]], CrushParams_0 constant* crush_66 [[buffer(28)]], packed_float4 device* crush_rw_66 [[buffer(27)]], packed_float4 device* crush_tab_66 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_185;
    (&kernelContext_185)->params_0 = params_66;
    (&kernelContext_185)->atom_0 = atom_66;
    (&kernelContext_185)->work_0 = work_66;
    (&kernelContext_185)->frames_0 = frames_66;
    (&kernelContext_185)->bodies_0 = bodies_66;
    (&kernelContext_185)->chunks_0 = chunks_66;
    (&kernelContext_185)->sort_0 = sort_66;
    (&kernelContext_185)->manifolds_0 = manifolds_66;
    (&kernelContext_185)->loads_0 = loads_66;
    (&kernelContext_185)->bonds_0 = bonds_66;
    (&kernelContext_185)->args_0 = args_66;
    (&kernelContext_185)->info_0 = info_66;
    (&kernelContext_185)->scan_0 = scan_66;
    (&kernelContext_185)->parts_0 = parts_66;
    (&kernelContext_185)->colliders_0 = colliders_66;
    (&kernelContext_185)->collider_info_0 = collider_info_66;
    (&kernelContext_185)->view_chunks_0 = view_chunks_66;
    (&kernelContext_185)->view_0 = view_66;
    (&kernelContext_185)->objects_0 = objects_66;
    (&kernelContext_185)->shot_bodies_0 = shot_bodies_66;
    (&kernelContext_185)->lines_0 = lines_66;
    (&kernelContext_185)->geometry_0 = geometry_66;
    (&kernelContext_185)->shots_0 = shots_66;
    (&kernelContext_185)->lp_0 = lp_66;
    (&kernelContext_185)->ext_0 = ext_68;
    (&kernelContext_185)->engine_contacts_0 = engine_contacts_66;
    (&kernelContext_185)->crush_0 = crush_66;
    (&kernelContext_185)->crush_rw_0 = crush_rw_66;
    (&kernelContext_185)->crush_tab_0 = crush_tab_66;
    threadgroup array<uint, int(1024)> g_scan_66;
    (&kernelContext_185)->g_scan_0 = &g_scan_66;
    threadgroup array<float, int(64)> g_il_best_66;
    (&kernelContext_185)->g_il_best_0 = &g_il_best_66;
    uint _S1214 = id_61.x;
    uint _S1215 = a_counters_1(&kernelContext_185);
    uint _S1216 = atomic_load_explicit((&kernelContext_185)->atom_0+(_S1215 + 18U), memory_order_relaxed);
    if(_S1214 >= _S1216)
    {
        return;
    }
    uint _S1217 = w_cs_bodies_0(&kernelContext_185);
    uint device* _S1218 = (&kernelContext_185)->work_0+(_S1217 + _S1214);
    uint s_18 = *_S1218;
    uint _S1219 = a_cs_0(*_S1218, &kernelContext_185);
    uint _S1220 = atomic_load_explicit((&kernelContext_185)->atom_0+(_S1219 + 11U), memory_order_relaxed);
    uint key_2 = _S1220 - 1U;
    thread uint k_45;
    thread uint s2_0;
    thread uint ob_6;
    thread uint co_6;
    thread float3 p_19;
    thread float3 u_10;
    thread float v_15;
    thread float m_eff_8;
    thread float t_in_9;
    bool _S1221 = cs_side_0(key_2 / 2U, key_2 % 2U, &k_45, &s2_0, &ob_6, &co_6, &p_19, &u_10, &v_15, &m_eff_8, &t_in_9, &kernelContext_185);
    if(!_S1221)
    {
        return;
    }
    uint4 _S1222 = uint4(*((&kernelContext_185)->collider_info_0+(2U * co_6 + 1U))) ;
    float e_o_0 = (as_type<float>((_S1222.y)));
    float _S1223 = max((as_type<float>((_S1222.z))), 0.00100000004749745f);
    float j_9 = m_eff_8 * v_15;
    float tau_2 = hertz_duration_0(m_eff_8, _S1223, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / (float4(*((&kernelContext_185)->geometry_0+(3U * k_45 + 1U))) ).w), v_15);
    uint _S1224 = lp_body_0(s_18, &kernelContext_185);
    *((&kernelContext_185)->lp_0+(_S1224 + 2U)) = packed_float4(float4(from_rubble_0(p_19), 3.14159274101257324f * j_9 / (2.0f * tau_2))) ;
    *((&kernelContext_185)->lp_0+(_S1224 + 3U)) = packed_float4(float4(from_rubble_0(u_10), v_15)) ;
    *((&kernelContext_185)->lp_0+(_S1224 + 4U)) = packed_float4(float4(m_eff_8, _S1223, e_o_0, (as_type<float>((ob_6))))) ;
    *((&kernelContext_185)->lp_0+(_S1224 + 5U)) = packed_float4(float4((as_type<float>((k_45))), j_9 / tau_2, 1.0f / (float4(*((&kernelContext_185)->bodies_0+7U * ob_6)) ).w, t_in_9)) ;
    *((&kernelContext_185)->lp_0+(_S1224 + 6U)) = packed_float4(float4((as_type<float>((co_6))), (as_type<float>((_S1222.w))), 0.0f, 0.0f)) ;
    atomic_store_explicit((&kernelContext_185)->atom_0+(_S1219 + 1U), k_45 + 1U, memory_order_relaxed);
    uint _S1225 = atomic_fetch_add_explicit((&kernelContext_185)->atom_0+(_S1215 + 16U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_hertz(uint3 id_62 [[thread_position_in_grid]], FractureParams_0 constant* params_67 [[buffer(0)]], atomic<uint> device* atom_67 [[buffer(5)]], uint device* work_67 [[buffer(4)]], packed_float4 device* frames_67 [[buffer(6)]], packed_float4 device* bodies_67 [[buffer(7)]], packed_float4 device* chunks_67 [[buffer(1)]], uint device* sort_67 [[buffer(11)]], Record_natural_0 device* manifolds_67 [[buffer(10)]], packed_float4 device* loads_67 [[buffer(12)]], packed_uint4 device* bonds_67 [[buffer(2)]], uint device* args_67 [[buffer(13)]], uint device* info_67 [[buffer(15)]], ScanParams_0 constant* scan_67 [[buffer(14)]], packed_float4 device* parts_67 [[buffer(3)]], packed_float4 device* colliders_67 [[buffer(8)]], packed_uint4 device* collider_info_67 [[buffer(9)]], packed_float4 device* view_chunks_67 [[buffer(17)]], ViewParams_0 constant* view_67 [[buffer(16)]], packed_float4 device* objects_67 [[buffer(18)]], uint device* shot_bodies_67 [[buffer(20)]], float device* lines_67 [[buffer(19)]], packed_float4 device* geometry_67 [[buffer(21)]], packed_float4 device* shots_67 [[buffer(22)]], packed_float4 device* lp_67 [[buffer(24)]], packed_float4 device* ext_69 [[buffer(23)]], packed_float4 device* engine_contacts_67 [[buffer(25)]], CrushParams_0 constant* crush_67 [[buffer(28)]], packed_float4 device* crush_rw_67 [[buffer(27)]], packed_float4 device* crush_tab_67 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_186;
    (&kernelContext_186)->params_0 = params_67;
    (&kernelContext_186)->atom_0 = atom_67;
    (&kernelContext_186)->work_0 = work_67;
    (&kernelContext_186)->frames_0 = frames_67;
    (&kernelContext_186)->bodies_0 = bodies_67;
    (&kernelContext_186)->chunks_0 = chunks_67;
    (&kernelContext_186)->sort_0 = sort_67;
    (&kernelContext_186)->manifolds_0 = manifolds_67;
    (&kernelContext_186)->loads_0 = loads_67;
    (&kernelContext_186)->bonds_0 = bonds_67;
    (&kernelContext_186)->args_0 = args_67;
    (&kernelContext_186)->info_0 = info_67;
    (&kernelContext_186)->scan_0 = scan_67;
    (&kernelContext_186)->parts_0 = parts_67;
    (&kernelContext_186)->colliders_0 = colliders_67;
    (&kernelContext_186)->collider_info_0 = collider_info_67;
    (&kernelContext_186)->view_chunks_0 = view_chunks_67;
    (&kernelContext_186)->view_0 = view_67;
    (&kernelContext_186)->objects_0 = objects_67;
    (&kernelContext_186)->shot_bodies_0 = shot_bodies_67;
    (&kernelContext_186)->lines_0 = lines_67;
    (&kernelContext_186)->geometry_0 = geometry_67;
    (&kernelContext_186)->shots_0 = shots_67;
    (&kernelContext_186)->lp_0 = lp_67;
    (&kernelContext_186)->ext_0 = ext_69;
    (&kernelContext_186)->engine_contacts_0 = engine_contacts_67;
    (&kernelContext_186)->crush_0 = crush_67;
    (&kernelContext_186)->crush_rw_0 = crush_rw_67;
    (&kernelContext_186)->crush_tab_0 = crush_tab_67;
    threadgroup array<uint, int(1024)> g_scan_67;
    (&kernelContext_186)->g_scan_0 = &g_scan_67;
    threadgroup array<float, int(64)> g_il_best_67;
    (&kernelContext_186)->g_il_best_0 = &g_il_best_67;
    uint i_20 = id_62.x;
    bool _S1226;
    if(i_20 >= (params_67->bond_count_0))
    {
        _S1226 = true;
    }
    else
    {
        uint _S1227 = w_alive_0(&kernelContext_186);
        _S1226 = (*((&kernelContext_186)->work_0+(_S1227 + i_20))) == 0U;
    }
    if(_S1226)
    {
        return;
    }
    uint4 _S1228 = uint4(*((&kernelContext_186)->bonds_0+i_20)) ;
    uint _S1229 = body_of_0(_S1228.x, &kernelContext_186);
    bool _S1230 = cs_struck_1(_S1229, &kernelContext_186);
    if(!_S1230)
    {
        return;
    }
    uint _S1231 = lp_body_0(_S1229, &kernelContext_186);
    bool _S1232 = strike_share_breaks_0(_S1228, (float4(*((&kernelContext_186)->lp_0+(_S1231 + 2U))) ).xyz, (float4(*((&kernelContext_186)->lp_0+(_S1231 + 5U))) ).y, &kernelContext_186);
    if(_S1232)
    {
        uint _S1233 = w_alive_0(&kernelContext_186);
        *((&kernelContext_186)->work_0+(_S1233 + i_20)) = 0U;
        uint _S1234 = a_dirty_0(&kernelContext_186);
        uint _S1235 = atomic_fetch_or_explicit((&kernelContext_186)->atom_0+(_S1234 + _S1229), 1U, memory_order_relaxed);
        uint _S1236 = a_counters_1(&kernelContext_186);
        uint _S1237 = atomic_fetch_add_explicit((&kernelContext_186)->atom_0+_S1236, 1U, memory_order_relaxed);
        uint _S1238 = atomic_fetch_add_explicit((&kernelContext_186)->atom_0+(_S1236 + 1U), 1U, memory_order_relaxed);
        uint _S1239 = atomic_fetch_add_explicit((&kernelContext_186)->atom_0+(_S1236 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_gate2(uint3 id_63 [[thread_position_in_grid]], FractureParams_0 constant* params_68 [[buffer(0)]], atomic<uint> device* atom_68 [[buffer(5)]], uint device* work_68 [[buffer(4)]], packed_float4 device* frames_68 [[buffer(6)]], packed_float4 device* bodies_68 [[buffer(7)]], packed_float4 device* chunks_68 [[buffer(1)]], uint device* sort_68 [[buffer(11)]], Record_natural_0 device* manifolds_68 [[buffer(10)]], packed_float4 device* loads_68 [[buffer(12)]], packed_uint4 device* bonds_68 [[buffer(2)]], uint device* args_68 [[buffer(13)]], uint device* info_68 [[buffer(15)]], ScanParams_0 constant* scan_68 [[buffer(14)]], packed_float4 device* parts_68 [[buffer(3)]], packed_float4 device* colliders_68 [[buffer(8)]], packed_uint4 device* collider_info_68 [[buffer(9)]], packed_float4 device* view_chunks_68 [[buffer(17)]], ViewParams_0 constant* view_68 [[buffer(16)]], packed_float4 device* objects_68 [[buffer(18)]], uint device* shot_bodies_68 [[buffer(20)]], float device* lines_68 [[buffer(19)]], packed_float4 device* geometry_68 [[buffer(21)]], packed_float4 device* shots_68 [[buffer(22)]], packed_float4 device* lp_68 [[buffer(24)]], packed_float4 device* ext_70 [[buffer(23)]], packed_float4 device* engine_contacts_68 [[buffer(25)]], CrushParams_0 constant* crush_68 [[buffer(28)]], packed_float4 device* crush_rw_68 [[buffer(27)]], packed_float4 device* crush_tab_68 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_187;
    (&kernelContext_187)->params_0 = params_68;
    (&kernelContext_187)->atom_0 = atom_68;
    (&kernelContext_187)->work_0 = work_68;
    (&kernelContext_187)->frames_0 = frames_68;
    (&kernelContext_187)->bodies_0 = bodies_68;
    (&kernelContext_187)->chunks_0 = chunks_68;
    (&kernelContext_187)->sort_0 = sort_68;
    (&kernelContext_187)->manifolds_0 = manifolds_68;
    (&kernelContext_187)->loads_0 = loads_68;
    (&kernelContext_187)->bonds_0 = bonds_68;
    (&kernelContext_187)->args_0 = args_68;
    (&kernelContext_187)->info_0 = info_68;
    (&kernelContext_187)->scan_0 = scan_68;
    (&kernelContext_187)->parts_0 = parts_68;
    (&kernelContext_187)->colliders_0 = colliders_68;
    (&kernelContext_187)->collider_info_0 = collider_info_68;
    (&kernelContext_187)->view_chunks_0 = view_chunks_68;
    (&kernelContext_187)->view_0 = view_68;
    (&kernelContext_187)->objects_0 = objects_68;
    (&kernelContext_187)->shot_bodies_0 = shot_bodies_68;
    (&kernelContext_187)->lines_0 = lines_68;
    (&kernelContext_187)->geometry_0 = geometry_68;
    (&kernelContext_187)->shots_0 = shots_68;
    (&kernelContext_187)->lp_0 = lp_68;
    (&kernelContext_187)->ext_0 = ext_70;
    (&kernelContext_187)->engine_contacts_0 = engine_contacts_68;
    (&kernelContext_187)->crush_0 = crush_68;
    (&kernelContext_187)->crush_rw_0 = crush_rw_68;
    (&kernelContext_187)->crush_tab_0 = crush_tab_68;
    threadgroup array<uint, int(1024)> g_scan_68;
    (&kernelContext_187)->g_scan_0 = &g_scan_68;
    threadgroup array<float, int(64)> g_il_best_68;
    (&kernelContext_187)->g_il_best_0 = &g_il_best_68;
    uint _S1240 = a_counters_1(&kernelContext_187);
    uint _S1241 = atomic_load_explicit((&kernelContext_187)->atom_0+_S1240, memory_order_relaxed);
    uint _S1242;
    if(_S1241 > 0U)
    {
        uint _S1243 = cs_rows_0(&kernelContext_187);
        uint _S1244 = groups_of_0(_S1243, 64U);
        _S1242 = _S1244;
    }
    else
    {
        _S1242 = 0U;
    }
    set_args_0(13U, _S1242, &kernelContext_187);
    return;
}

uint lp_anchor_0(uint r_25, uint side_7, KernelContext_0 thread* kernelContext_188)
{
    return 22U * kernelContext_188->params_0->chunk_count_0 + 7U * kernelContext_188->params_0->body_count_0 + 2U * r_25 + side_7;
}

[[kernel]] void fr_cs_save(uint3 id_64 [[thread_position_in_grid]], FractureParams_0 constant* params_69 [[buffer(0)]], atomic<uint> device* atom_69 [[buffer(5)]], uint device* work_69 [[buffer(4)]], packed_float4 device* frames_69 [[buffer(6)]], packed_float4 device* bodies_69 [[buffer(7)]], packed_float4 device* chunks_69 [[buffer(1)]], uint device* sort_69 [[buffer(11)]], Record_natural_0 device* manifolds_69 [[buffer(10)]], packed_float4 device* loads_69 [[buffer(12)]], packed_uint4 device* bonds_69 [[buffer(2)]], uint device* args_69 [[buffer(13)]], uint device* info_69 [[buffer(15)]], ScanParams_0 constant* scan_69 [[buffer(14)]], packed_float4 device* parts_69 [[buffer(3)]], packed_float4 device* colliders_69 [[buffer(8)]], packed_uint4 device* collider_info_69 [[buffer(9)]], packed_float4 device* view_chunks_69 [[buffer(17)]], ViewParams_0 constant* view_69 [[buffer(16)]], packed_float4 device* objects_69 [[buffer(18)]], uint device* shot_bodies_69 [[buffer(20)]], float device* lines_69 [[buffer(19)]], packed_float4 device* geometry_69 [[buffer(21)]], packed_float4 device* shots_69 [[buffer(22)]], packed_float4 device* lp_69 [[buffer(24)]], packed_float4 device* ext_71 [[buffer(23)]], packed_float4 device* engine_contacts_69 [[buffer(25)]], CrushParams_0 constant* crush_69 [[buffer(28)]], packed_float4 device* crush_rw_69 [[buffer(27)]], packed_float4 device* crush_tab_69 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_189;
    (&kernelContext_189)->params_0 = params_69;
    (&kernelContext_189)->atom_0 = atom_69;
    (&kernelContext_189)->work_0 = work_69;
    (&kernelContext_189)->frames_0 = frames_69;
    (&kernelContext_189)->bodies_0 = bodies_69;
    (&kernelContext_189)->chunks_0 = chunks_69;
    (&kernelContext_189)->sort_0 = sort_69;
    (&kernelContext_189)->manifolds_0 = manifolds_69;
    (&kernelContext_189)->loads_0 = loads_69;
    (&kernelContext_189)->bonds_0 = bonds_69;
    (&kernelContext_189)->args_0 = args_69;
    (&kernelContext_189)->info_0 = info_69;
    (&kernelContext_189)->scan_0 = scan_69;
    (&kernelContext_189)->parts_0 = parts_69;
    (&kernelContext_189)->colliders_0 = colliders_69;
    (&kernelContext_189)->collider_info_0 = collider_info_69;
    (&kernelContext_189)->view_chunks_0 = view_chunks_69;
    (&kernelContext_189)->view_0 = view_69;
    (&kernelContext_189)->objects_0 = objects_69;
    (&kernelContext_189)->shot_bodies_0 = shot_bodies_69;
    (&kernelContext_189)->lines_0 = lines_69;
    (&kernelContext_189)->geometry_0 = geometry_69;
    (&kernelContext_189)->shots_0 = shots_69;
    (&kernelContext_189)->lp_0 = lp_69;
    (&kernelContext_189)->ext_0 = ext_71;
    (&kernelContext_189)->engine_contacts_0 = engine_contacts_69;
    (&kernelContext_189)->crush_0 = crush_69;
    (&kernelContext_189)->crush_rw_0 = crush_rw_69;
    (&kernelContext_189)->crush_tab_0 = crush_tab_69;
    threadgroup array<uint, int(1024)> g_scan_69;
    (&kernelContext_189)->g_scan_0 = &g_scan_69;
    threadgroup array<float, int(64)> g_il_best_69;
    (&kernelContext_189)->g_il_best_0 = &g_il_best_69;
    uint r_26 = id_64.x;
    uint _S1245 = cs_rows_0(&kernelContext_189);
    if(r_26 >= _S1245)
    {
        return;
    }
    uint _S1246 = cs_row_0(r_26, &kernelContext_189);
    uint4 _S1247 = (as_type<uint4>((float4(*((&kernelContext_189)->engine_contacts_0+_S1246)) )));
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
        float4 _S1248 = float4(0.0f) ;
        bool _S1249 = side_8 == 0U;
        uint col_1;
        if(_S1249)
        {
            col_1 = _S1247.x;
        }
        else
        {
            col_1 = _S1247.y;
        }
        uint k_46;
        if((_S1247.w) != 0U)
        {
            k_46 = (uint4(*((&kernelContext_189)->collider_info_0+2U * col_1)) ).x;
        }
        else
        {
            k_46 = 4294967295U;
        }
        bool _S1250;
        if(k_46 != 4294967295U)
        {
            _S1250 = k_46 < ((&kernelContext_189)->params_0->chunk_count_0);
        }
        else
        {
            _S1250 = false;
        }
        bool _S1251;
        if(_S1250)
        {
            uint _S1252 = body_of_0(k_46, &kernelContext_189);
            bool _S1253 = dirty_0(_S1252, &kernelContext_189);
            _S1251 = _S1253;
        }
        else
        {
            _S1251 = false;
        }
        float4 saved_0;
        if(_S1251)
        {
            uint _S1254 = body_of_0(k_46, &kernelContext_189);
            uint at_21 = 7U * _S1254;
            float3 anchor_0;
            if(_S1249)
            {
                anchor_0 = (float4(*((&kernelContext_189)->engine_contacts_0+(_S1246 + 2U))) ).xyz;
            }
            else
            {
                anchor_0 = (float4(*((&kernelContext_189)->engine_contacts_0+(_S1246 + 3U))) ).xyz;
            }
            float3 _S1255 = (float4(*((&kernelContext_189)->bodies_0+at_21)) ).xyz;
            thread Quat_0 _S1256 = quat_of_0(float4(*((&kernelContext_189)->bodies_0+(at_21 + 1U))) );
            float3 _S1257 = rotate_0(&_S1256, anchor_0);
            saved_0 = float4(_S1255 + _S1257, 1.0f);
        }
        else
        {
            saved_0 = _S1248;
        }
        uint _S1258 = lp_anchor_0(r_26, side_8, &kernelContext_189);
        *((&kernelContext_189)->lp_0+_S1258) = packed_float4(saved_0) ;
        side_8 = side_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_clean(uint3 id_65 [[thread_position_in_grid]], FractureParams_0 constant* params_70 [[buffer(0)]], atomic<uint> device* atom_70 [[buffer(5)]], uint device* work_70 [[buffer(4)]], packed_float4 device* frames_70 [[buffer(6)]], packed_float4 device* bodies_70 [[buffer(7)]], packed_float4 device* chunks_70 [[buffer(1)]], uint device* sort_70 [[buffer(11)]], Record_natural_0 device* manifolds_70 [[buffer(10)]], packed_float4 device* loads_70 [[buffer(12)]], packed_uint4 device* bonds_70 [[buffer(2)]], uint device* args_70 [[buffer(13)]], uint device* info_70 [[buffer(15)]], ScanParams_0 constant* scan_70 [[buffer(14)]], packed_float4 device* parts_70 [[buffer(3)]], packed_float4 device* colliders_70 [[buffer(8)]], packed_uint4 device* collider_info_70 [[buffer(9)]], packed_float4 device* view_chunks_70 [[buffer(17)]], ViewParams_0 constant* view_70 [[buffer(16)]], packed_float4 device* objects_70 [[buffer(18)]], uint device* shot_bodies_70 [[buffer(20)]], float device* lines_70 [[buffer(19)]], packed_float4 device* geometry_70 [[buffer(21)]], packed_float4 device* shots_70 [[buffer(22)]], packed_float4 device* lp_70 [[buffer(24)]], packed_float4 device* ext_72 [[buffer(23)]], packed_float4 device* engine_contacts_70 [[buffer(25)]], CrushParams_0 constant* crush_70 [[buffer(28)]], packed_float4 device* crush_rw_70 [[buffer(27)]], packed_float4 device* crush_tab_70 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_190;
    (&kernelContext_190)->params_0 = params_70;
    (&kernelContext_190)->atom_0 = atom_70;
    (&kernelContext_190)->work_0 = work_70;
    (&kernelContext_190)->frames_0 = frames_70;
    (&kernelContext_190)->bodies_0 = bodies_70;
    (&kernelContext_190)->chunks_0 = chunks_70;
    (&kernelContext_190)->sort_0 = sort_70;
    (&kernelContext_190)->manifolds_0 = manifolds_70;
    (&kernelContext_190)->loads_0 = loads_70;
    (&kernelContext_190)->bonds_0 = bonds_70;
    (&kernelContext_190)->args_0 = args_70;
    (&kernelContext_190)->info_0 = info_70;
    (&kernelContext_190)->scan_0 = scan_70;
    (&kernelContext_190)->parts_0 = parts_70;
    (&kernelContext_190)->colliders_0 = colliders_70;
    (&kernelContext_190)->collider_info_0 = collider_info_70;
    (&kernelContext_190)->view_chunks_0 = view_chunks_70;
    (&kernelContext_190)->view_0 = view_70;
    (&kernelContext_190)->objects_0 = objects_70;
    (&kernelContext_190)->shot_bodies_0 = shot_bodies_70;
    (&kernelContext_190)->lines_0 = lines_70;
    (&kernelContext_190)->geometry_0 = geometry_70;
    (&kernelContext_190)->shots_0 = shots_70;
    (&kernelContext_190)->lp_0 = lp_70;
    (&kernelContext_190)->ext_0 = ext_72;
    (&kernelContext_190)->engine_contacts_0 = engine_contacts_70;
    (&kernelContext_190)->crush_0 = crush_70;
    (&kernelContext_190)->crush_rw_0 = crush_rw_70;
    (&kernelContext_190)->crush_tab_0 = crush_tab_70;
    threadgroup array<uint, int(1024)> g_scan_70;
    (&kernelContext_190)->g_scan_0 = &g_scan_70;
    threadgroup array<float, int(64)> g_il_best_70;
    (&kernelContext_190)->g_il_best_0 = &g_il_best_70;
    uint _S1259 = id_65.x;
    uint _S1260 = a_counters_1(&kernelContext_190);
    uint _S1261 = atomic_load_explicit((&kernelContext_190)->atom_0+(_S1260 + 18U), memory_order_relaxed);
    if(_S1259 >= _S1261)
    {
        return;
    }
    uint _S1262 = w_cs_bodies_0(&kernelContext_190);
    uint _S1263 = a_cs_0(*((&kernelContext_190)->work_0+(_S1262 + _S1259)), &kernelContext_190);
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
        atomic_store_explicit((&kernelContext_190)->atom_0+(_S1263 + w_8), 0U, memory_order_relaxed);
        w_8 = w_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_fix(uint3 id_66 [[thread_position_in_grid]], FractureParams_0 constant* params_71 [[buffer(0)]], atomic<uint> device* atom_71 [[buffer(5)]], uint device* work_71 [[buffer(4)]], packed_float4 device* frames_71 [[buffer(6)]], packed_float4 device* bodies_71 [[buffer(7)]], packed_float4 device* chunks_71 [[buffer(1)]], uint device* sort_71 [[buffer(11)]], Record_natural_0 device* manifolds_71 [[buffer(10)]], packed_float4 device* loads_71 [[buffer(12)]], packed_uint4 device* bonds_71 [[buffer(2)]], uint device* args_71 [[buffer(13)]], uint device* info_71 [[buffer(15)]], ScanParams_0 constant* scan_71 [[buffer(14)]], packed_float4 device* parts_71 [[buffer(3)]], packed_float4 device* colliders_71 [[buffer(8)]], packed_uint4 device* collider_info_71 [[buffer(9)]], packed_float4 device* view_chunks_71 [[buffer(17)]], ViewParams_0 constant* view_71 [[buffer(16)]], packed_float4 device* objects_71 [[buffer(18)]], uint device* shot_bodies_71 [[buffer(20)]], float device* lines_71 [[buffer(19)]], packed_float4 device* geometry_71 [[buffer(21)]], packed_float4 device* shots_71 [[buffer(22)]], packed_float4 device* lp_71 [[buffer(24)]], packed_float4 device* ext_73 [[buffer(23)]], packed_float4 device* engine_contacts_71 [[buffer(25)]], CrushParams_0 constant* crush_71 [[buffer(28)]], packed_float4 device* crush_rw_71 [[buffer(27)]], packed_float4 device* crush_tab_71 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_191;
    (&kernelContext_191)->params_0 = params_71;
    (&kernelContext_191)->atom_0 = atom_71;
    (&kernelContext_191)->work_0 = work_71;
    (&kernelContext_191)->frames_0 = frames_71;
    (&kernelContext_191)->bodies_0 = bodies_71;
    (&kernelContext_191)->chunks_0 = chunks_71;
    (&kernelContext_191)->sort_0 = sort_71;
    (&kernelContext_191)->manifolds_0 = manifolds_71;
    (&kernelContext_191)->loads_0 = loads_71;
    (&kernelContext_191)->bonds_0 = bonds_71;
    (&kernelContext_191)->args_0 = args_71;
    (&kernelContext_191)->info_0 = info_71;
    (&kernelContext_191)->scan_0 = scan_71;
    (&kernelContext_191)->parts_0 = parts_71;
    (&kernelContext_191)->colliders_0 = colliders_71;
    (&kernelContext_191)->collider_info_0 = collider_info_71;
    (&kernelContext_191)->view_chunks_0 = view_chunks_71;
    (&kernelContext_191)->view_0 = view_71;
    (&kernelContext_191)->objects_0 = objects_71;
    (&kernelContext_191)->shot_bodies_0 = shot_bodies_71;
    (&kernelContext_191)->lines_0 = lines_71;
    (&kernelContext_191)->geometry_0 = geometry_71;
    (&kernelContext_191)->shots_0 = shots_71;
    (&kernelContext_191)->lp_0 = lp_71;
    (&kernelContext_191)->ext_0 = ext_73;
    (&kernelContext_191)->engine_contacts_0 = engine_contacts_71;
    (&kernelContext_191)->crush_0 = crush_71;
    (&kernelContext_191)->crush_rw_0 = crush_rw_71;
    (&kernelContext_191)->crush_tab_0 = crush_tab_71;
    threadgroup array<uint, int(1024)> g_scan_71;
    (&kernelContext_191)->g_scan_0 = &g_scan_71;
    threadgroup array<float, int(64)> g_il_best_71;
    (&kernelContext_191)->g_il_best_0 = &g_il_best_71;
    uint r_27 = id_66.x;
    uint _S1264 = cs_rows_0(&kernelContext_191);
    if(r_27 >= _S1264)
    {
        return;
    }
    uint _S1265 = cs_row_0(r_27, &kernelContext_191);
    uint4 _S1266 = (as_type<uint4>((float4(*((&kernelContext_191)->engine_contacts_0+_S1265)) )));
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
        uint _S1267 = lp_anchor_0(r_27, side_9, &kernelContext_191);
        float4 _S1268 = float4(*((&kernelContext_191)->lp_0+_S1267)) ;
        if((_S1268.w) == 0.0f)
        {
            side_9 = side_9 + 1U;
            continue;
        }
        uint col_2;
        if(side_9 == 0U)
        {
            col_2 = _S1266.x;
        }
        else
        {
            col_2 = _S1266.y;
        }
        uint at_22 = 7U * (as_type<uint>(((float4(*((&kernelContext_191)->colliders_0+4U * col_2)) ).x)));
        thread Quat_0 _S1269 = quat_of_0(float4(*((&kernelContext_191)->bodies_0+(at_22 + 1U))) );
        Quat_0 _S1270 = conj_0(&_S1269);
        float3 _S1271 = _S1268.xyz - (float4(*((&kernelContext_191)->bodies_0+at_22)) ).xyz;
        thread Quat_0 _S1272 = _S1270;
        float3 _S1273 = rotate_0(&_S1272, _S1271);
        (*((&kernelContext_191)->engine_contacts_0+(_S1265 + 2U + side_9))).xyz = _S1273;
        moved_0 = true;
        side_9 = side_9 + 1U;
    }
    if(moved_0)
    {
        (*((&kernelContext_191)->engine_contacts_0+(_S1265 + 4U)))[int(3)] = 0.0f;
        (*((&kernelContext_191)->engine_contacts_0+(_S1265 + 5U)))[int(3)] = 0.0f;
        uint _S1274 = _S1265 + 6U;
        *((&kernelContext_191)->engine_contacts_0+_S1274) = packed_float4(float4(1.0f, 1.0f, 0.0f, (float4(*((&kernelContext_191)->engine_contacts_0+_S1274)) ).w)) ;
        (*((&kernelContext_191)->engine_contacts_0+(_S1265 + 7U)))[int(1)] = 0.0f;
    }
    return;
}

[[kernel]] void fr_cs_count(uint3 id_67 [[thread_position_in_grid]], FractureParams_0 constant* params_72 [[buffer(0)]], atomic<uint> device* atom_72 [[buffer(5)]], uint device* work_72 [[buffer(4)]], packed_float4 device* frames_72 [[buffer(6)]], packed_float4 device* bodies_72 [[buffer(7)]], packed_float4 device* chunks_72 [[buffer(1)]], uint device* sort_72 [[buffer(11)]], Record_natural_0 device* manifolds_72 [[buffer(10)]], packed_float4 device* loads_72 [[buffer(12)]], packed_uint4 device* bonds_72 [[buffer(2)]], uint device* args_72 [[buffer(13)]], uint device* info_72 [[buffer(15)]], ScanParams_0 constant* scan_72 [[buffer(14)]], packed_float4 device* parts_72 [[buffer(3)]], packed_float4 device* colliders_72 [[buffer(8)]], packed_uint4 device* collider_info_72 [[buffer(9)]], packed_float4 device* view_chunks_72 [[buffer(17)]], ViewParams_0 constant* view_72 [[buffer(16)]], packed_float4 device* objects_72 [[buffer(18)]], uint device* shot_bodies_72 [[buffer(20)]], float device* lines_72 [[buffer(19)]], packed_float4 device* geometry_72 [[buffer(21)]], packed_float4 device* shots_72 [[buffer(22)]], packed_float4 device* lp_72 [[buffer(24)]], packed_float4 device* ext_74 [[buffer(23)]], packed_float4 device* engine_contacts_72 [[buffer(25)]], CrushParams_0 constant* crush_72 [[buffer(28)]], packed_float4 device* crush_rw_72 [[buffer(27)]], packed_float4 device* crush_tab_72 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_192;
    (&kernelContext_192)->params_0 = params_72;
    (&kernelContext_192)->atom_0 = atom_72;
    (&kernelContext_192)->work_0 = work_72;
    (&kernelContext_192)->frames_0 = frames_72;
    (&kernelContext_192)->bodies_0 = bodies_72;
    (&kernelContext_192)->chunks_0 = chunks_72;
    (&kernelContext_192)->sort_0 = sort_72;
    (&kernelContext_192)->manifolds_0 = manifolds_72;
    (&kernelContext_192)->loads_0 = loads_72;
    (&kernelContext_192)->bonds_0 = bonds_72;
    (&kernelContext_192)->args_0 = args_72;
    (&kernelContext_192)->info_0 = info_72;
    (&kernelContext_192)->scan_0 = scan_72;
    (&kernelContext_192)->parts_0 = parts_72;
    (&kernelContext_192)->colliders_0 = colliders_72;
    (&kernelContext_192)->collider_info_0 = collider_info_72;
    (&kernelContext_192)->view_chunks_0 = view_chunks_72;
    (&kernelContext_192)->view_0 = view_72;
    (&kernelContext_192)->objects_0 = objects_72;
    (&kernelContext_192)->shot_bodies_0 = shot_bodies_72;
    (&kernelContext_192)->lines_0 = lines_72;
    (&kernelContext_192)->geometry_0 = geometry_72;
    (&kernelContext_192)->shots_0 = shots_72;
    (&kernelContext_192)->lp_0 = lp_72;
    (&kernelContext_192)->ext_0 = ext_74;
    (&kernelContext_192)->engine_contacts_0 = engine_contacts_72;
    (&kernelContext_192)->crush_0 = crush_72;
    (&kernelContext_192)->crush_rw_0 = crush_rw_72;
    (&kernelContext_192)->crush_tab_0 = crush_tab_72;
    threadgroup array<uint, int(1024)> g_scan_72;
    (&kernelContext_192)->g_scan_0 = &g_scan_72;
    threadgroup array<float, int(64)> g_il_best_72;
    (&kernelContext_192)->g_il_best_0 = &g_il_best_72;
    uint k_47 = id_67.x;
    if(k_47 >= (params_72->chunk_count_0))
    {
        return;
    }
    uint _S1275 = body_of_0(k_47, &kernelContext_192);
    bool _S1276 = cs_struck_1(_S1275, &kernelContext_192);
    if(!_S1276)
    {
        return;
    }
    uint _S1277 = a_cs_0(_S1275, &kernelContext_192);
    uint _S1278 = atomic_fetch_max_explicit((&kernelContext_192)->atom_0+(_S1277 + 3U), ~k_47, memory_order_relaxed);
    if(((&kernelContext_192)->info_0[k_47] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_192)->atom_0+(_S1277 + 4U), 1U, memory_order_relaxed);
    }
    uint _S1279 = atomic_fetch_add_explicit((&kernelContext_192)->atom_0+(_S1277 + 5U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_base(uint3 id_68 [[thread_position_in_grid]], FractureParams_0 constant* params_73 [[buffer(0)]], atomic<uint> device* atom_73 [[buffer(5)]], uint device* work_73 [[buffer(4)]], packed_float4 device* frames_73 [[buffer(6)]], packed_float4 device* bodies_73 [[buffer(7)]], packed_float4 device* chunks_73 [[buffer(1)]], uint device* sort_73 [[buffer(11)]], Record_natural_0 device* manifolds_73 [[buffer(10)]], packed_float4 device* loads_73 [[buffer(12)]], packed_uint4 device* bonds_73 [[buffer(2)]], uint device* args_73 [[buffer(13)]], uint device* info_73 [[buffer(15)]], ScanParams_0 constant* scan_73 [[buffer(14)]], packed_float4 device* parts_73 [[buffer(3)]], packed_float4 device* colliders_73 [[buffer(8)]], packed_uint4 device* collider_info_73 [[buffer(9)]], packed_float4 device* view_chunks_73 [[buffer(17)]], ViewParams_0 constant* view_73 [[buffer(16)]], packed_float4 device* objects_73 [[buffer(18)]], uint device* shot_bodies_73 [[buffer(20)]], float device* lines_73 [[buffer(19)]], packed_float4 device* geometry_73 [[buffer(21)]], packed_float4 device* shots_73 [[buffer(22)]], packed_float4 device* lp_73 [[buffer(24)]], packed_float4 device* ext_75 [[buffer(23)]], packed_float4 device* engine_contacts_73 [[buffer(25)]], CrushParams_0 constant* crush_73 [[buffer(28)]], packed_float4 device* crush_rw_73 [[buffer(27)]], packed_float4 device* crush_tab_73 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_193;
    (&kernelContext_193)->params_0 = params_73;
    (&kernelContext_193)->atom_0 = atom_73;
    (&kernelContext_193)->work_0 = work_73;
    (&kernelContext_193)->frames_0 = frames_73;
    (&kernelContext_193)->bodies_0 = bodies_73;
    (&kernelContext_193)->chunks_0 = chunks_73;
    (&kernelContext_193)->sort_0 = sort_73;
    (&kernelContext_193)->manifolds_0 = manifolds_73;
    (&kernelContext_193)->loads_0 = loads_73;
    (&kernelContext_193)->bonds_0 = bonds_73;
    (&kernelContext_193)->args_0 = args_73;
    (&kernelContext_193)->info_0 = info_73;
    (&kernelContext_193)->scan_0 = scan_73;
    (&kernelContext_193)->parts_0 = parts_73;
    (&kernelContext_193)->colliders_0 = colliders_73;
    (&kernelContext_193)->collider_info_0 = collider_info_73;
    (&kernelContext_193)->view_chunks_0 = view_chunks_73;
    (&kernelContext_193)->view_0 = view_73;
    (&kernelContext_193)->objects_0 = objects_73;
    (&kernelContext_193)->shot_bodies_0 = shot_bodies_73;
    (&kernelContext_193)->lines_0 = lines_73;
    (&kernelContext_193)->geometry_0 = geometry_73;
    (&kernelContext_193)->shots_0 = shots_73;
    (&kernelContext_193)->lp_0 = lp_73;
    (&kernelContext_193)->ext_0 = ext_75;
    (&kernelContext_193)->engine_contacts_0 = engine_contacts_73;
    (&kernelContext_193)->crush_0 = crush_73;
    (&kernelContext_193)->crush_rw_0 = crush_rw_73;
    (&kernelContext_193)->crush_tab_0 = crush_tab_73;
    threadgroup array<uint, int(1024)> g_scan_73;
    (&kernelContext_193)->g_scan_0 = &g_scan_73;
    threadgroup array<float, int(64)> g_il_best_73;
    (&kernelContext_193)->g_il_best_0 = &g_il_best_73;
    uint _S1280 = id_68.x;
    uint _S1281 = a_counters_1(&kernelContext_193);
    uint _S1282 = atomic_load_explicit((&kernelContext_193)->atom_0+(_S1281 + 18U), memory_order_relaxed);
    if(_S1280 >= _S1282)
    {
        return;
    }
    uint _S1283 = w_cs_bodies_0(&kernelContext_193);
    uint _S1284 = a_cs_0(*((&kernelContext_193)->work_0+(_S1283 + _S1280)), &kernelContext_193);
    atomic<uint> device* _S1285 = (&kernelContext_193)->atom_0+(_S1284 + 6U);
    atomic<uint> device* _S1286 = (&kernelContext_193)->atom_0+(_S1281 + 19U);
    uint _S1287 = atomic_load_explicit((&kernelContext_193)->atom_0+(_S1284 + 5U), memory_order_relaxed);
    uint _S1288 = atomic_fetch_add_explicit(_S1286, _S1287, memory_order_relaxed);
    atomic_store_explicit(_S1285, _S1288, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_collect(uint3 id_69 [[thread_position_in_grid]], FractureParams_0 constant* params_74 [[buffer(0)]], atomic<uint> device* atom_74 [[buffer(5)]], uint device* work_74 [[buffer(4)]], packed_float4 device* frames_74 [[buffer(6)]], packed_float4 device* bodies_74 [[buffer(7)]], packed_float4 device* chunks_74 [[buffer(1)]], uint device* sort_74 [[buffer(11)]], Record_natural_0 device* manifolds_74 [[buffer(10)]], packed_float4 device* loads_74 [[buffer(12)]], packed_uint4 device* bonds_74 [[buffer(2)]], uint device* args_74 [[buffer(13)]], uint device* info_74 [[buffer(15)]], ScanParams_0 constant* scan_74 [[buffer(14)]], packed_float4 device* parts_74 [[buffer(3)]], packed_float4 device* colliders_74 [[buffer(8)]], packed_uint4 device* collider_info_74 [[buffer(9)]], packed_float4 device* view_chunks_74 [[buffer(17)]], ViewParams_0 constant* view_74 [[buffer(16)]], packed_float4 device* objects_74 [[buffer(18)]], uint device* shot_bodies_74 [[buffer(20)]], float device* lines_74 [[buffer(19)]], packed_float4 device* geometry_74 [[buffer(21)]], packed_float4 device* shots_74 [[buffer(22)]], packed_float4 device* lp_74 [[buffer(24)]], packed_float4 device* ext_76 [[buffer(23)]], packed_float4 device* engine_contacts_74 [[buffer(25)]], CrushParams_0 constant* crush_74 [[buffer(28)]], packed_float4 device* crush_rw_74 [[buffer(27)]], packed_float4 device* crush_tab_74 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_194;
    (&kernelContext_194)->params_0 = params_74;
    (&kernelContext_194)->atom_0 = atom_74;
    (&kernelContext_194)->work_0 = work_74;
    (&kernelContext_194)->frames_0 = frames_74;
    (&kernelContext_194)->bodies_0 = bodies_74;
    (&kernelContext_194)->chunks_0 = chunks_74;
    (&kernelContext_194)->sort_0 = sort_74;
    (&kernelContext_194)->manifolds_0 = manifolds_74;
    (&kernelContext_194)->loads_0 = loads_74;
    (&kernelContext_194)->bonds_0 = bonds_74;
    (&kernelContext_194)->args_0 = args_74;
    (&kernelContext_194)->info_0 = info_74;
    (&kernelContext_194)->scan_0 = scan_74;
    (&kernelContext_194)->parts_0 = parts_74;
    (&kernelContext_194)->colliders_0 = colliders_74;
    (&kernelContext_194)->collider_info_0 = collider_info_74;
    (&kernelContext_194)->view_chunks_0 = view_chunks_74;
    (&kernelContext_194)->view_0 = view_74;
    (&kernelContext_194)->objects_0 = objects_74;
    (&kernelContext_194)->shot_bodies_0 = shot_bodies_74;
    (&kernelContext_194)->lines_0 = lines_74;
    (&kernelContext_194)->geometry_0 = geometry_74;
    (&kernelContext_194)->shots_0 = shots_74;
    (&kernelContext_194)->lp_0 = lp_74;
    (&kernelContext_194)->ext_0 = ext_76;
    (&kernelContext_194)->engine_contacts_0 = engine_contacts_74;
    (&kernelContext_194)->crush_0 = crush_74;
    (&kernelContext_194)->crush_rw_0 = crush_rw_74;
    (&kernelContext_194)->crush_tab_0 = crush_tab_74;
    threadgroup array<uint, int(1024)> g_scan_74;
    (&kernelContext_194)->g_scan_0 = &g_scan_74;
    threadgroup array<float, int(64)> g_il_best_74;
    (&kernelContext_194)->g_il_best_0 = &g_il_best_74;
    uint k_48 = id_69.x;
    if(k_48 >= (params_74->chunk_count_0))
    {
        return;
    }
    uint _S1289 = body_of_0(k_48, &kernelContext_194);
    bool _S1290 = cs_struck_1(_S1289, &kernelContext_194);
    if(!_S1290)
    {
        return;
    }
    uint _S1291 = w_cs_list_0(&kernelContext_194);
    uint _S1292 = a_cs_0(_S1289, &kernelContext_194);
    uint _S1293 = atomic_load_explicit((&kernelContext_194)->atom_0+(_S1292 + 6U), memory_order_relaxed);
    uint _S1294 = _S1291 + _S1293;
    uint _S1295 = atomic_fetch_add_explicit((&kernelContext_194)->atom_0+(_S1292 + 7U), 1U, memory_order_relaxed);
    *((&kernelContext_194)->work_0+(_S1294 + _S1295)) = k_48;
    return;
}

void cs_load_0(uint k_49, uint s_19, KernelContext_0 thread* kernelContext_195)
{
    uint _S1296 = a_cs_0(s_19, kernelContext_195);
    uint _S1297 = atomic_load_explicit(kernelContext_195->atom_0+(_S1296 + 4U), memory_order_relaxed);
    bool supported_1 = _S1297 != 0U;
    bool fixed_2;
    if((kernelContext_195->info_0[k_49] & 1U) != 0U)
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            uint _S1298 = atomic_load_explicit(kernelContext_195->atom_0+(_S1296 + 3U), memory_order_relaxed);
            fixed_2 = (~_S1298) == k_49;
        }
        else
        {
            fixed_2 = false;
        }
    }
    uint _S1299 = lp_body_0(s_19, kernelContext_195);
    float4 _S1300 = float4(*(kernelContext_195->lp_0+(_S1299 + 2U))) ;
    float4 _S1301 = float4(*(kernelContext_195->lp_0+(_S1299 + 3U))) ;
    uint _S1302 = 7U * s_19;
    thread Quat_0 q_27;
    thread float3 origin_8;
    structure_pose_0(s_19, _S1302, &q_27, &origin_8, kernelContext_195);
    thread Quat_0 _S1303 = q_27;
    Quat_0 _S1304 = conj_0(&_S1303);
    float3 _S1305 = _S1301.xyz * float3(_S1300.w) ;
    thread Quat_0 _S1306 = _S1304;
    float3 _S1307 = rotate_0(&_S1306, _S1305);
    float3 _S1308 = _S1300.xyz - origin_8;
    thread Quat_0 _S1309 = _S1304;
    float3 _S1310 = rotate_0(&_S1309, _S1308);
    uint _S1311 = 4U * k_49;
    float4 _S1312 = float4(*(kernelContext_195->chunks_0+_S1311)) ;
    float3 _S1313 = float3(0.0f) ;
    uint _S1314 = atomic_load_explicit(kernelContext_195->atom_0+(_S1296 + 1U), memory_order_relaxed);
    float3 f_12;
    float3 t_13;
    if(_S1314 == (k_49 + 1U))
    {
        float3 _S1315 = cross(_S1310 - _S1312.xyz, _S1307);
        f_12 = _S1307;
        t_13 = _S1315;
    }
    else
    {
        f_12 = _S1313;
        t_13 = _S1313;
    }
    float inv_mass_7 = (float4(*(kernelContext_195->bodies_0+_S1302)) ).w;
    bool _S1316;
    if(!supported_1)
    {
        _S1316 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S1316 = false;
    }
    if(_S1316)
    {
        uint _S1317 = 2U * s_19;
        float3 com_2 = (float4(*(kernelContext_195->frames_0+_S1317)) ).xyz;
        Quat_0 p_20 = quat_of_0(float4(*(kernelContext_195->frames_0+(_S1317 + 1U))) );
        float3 _S1318 = (float4(*(kernelContext_195->bodies_0+(_S1302 + 2U))) ).xyz;
        thread Quat_0 _S1319 = p_20;
        Quat_0 _S1320 = conj_0(&_S1319);
        float3 _S1321 = cross(_S1310 - com_2, _S1307);
        thread Quat_0 _S1322 = _S1320;
        float3 _S1323 = rotate_0(&_S1322, _S1321);
        float3 _S1324 = _S1318 * _S1323;
        thread Quat_0 _S1325 = p_20;
        float3 _S1326 = rotate_0(&_S1325, _S1324);
        float4 _S1327 = float4(*(kernelContext_195->chunks_0+(_S1311 + 1U))) ;
        float4 _S1328 = float4(*(kernelContext_195->chunks_0+(_S1311 + 2U))) ;
        float _S1329 = _S1326.x;
        float _S1330 = _S1327.w;
        float _S1331 = _S1326.y;
        float _S1332 = _S1328.x;
        float _S1333 = _S1326.z;
        float _S1334 = _S1328.y;
        float3 t_14 = t_13 - float3(_S1327.x * _S1329 + _S1330 * _S1331 + _S1332 * _S1333, _S1330 * _S1329 + _S1327.y * _S1331 + _S1334 * _S1333, _S1332 * _S1329 + _S1334 * _S1331 + _S1327.z * _S1333);
        f_12 = f_12 - float3(_S1312.w)  * (_S1307 * float3(inv_mass_7)  + cross(_S1326, _S1312.xyz - com_2));
        t_13 = t_14;
    }
    uint _S1335 = st_f_0(k_49, 14U);
    packed_float4 device* _S1336 = kernelContext_195->lp_0+_S1335;
    float _S1337;
    if(fixed_2)
    {
        _S1337 = 1.0f;
    }
    else
    {
        _S1337 = 0.0f;
    }
    *_S1336 = packed_float4(float4(f_12, _S1337)) ;
    *(kernelContext_195->lp_0+(_S1335 + 1U)) = packed_float4(float4(t_13, 0.0f)) ;
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
        uint _S1338 = st_x_0(k_49, 14U, c_24);
        packed_float4 _S1339 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_195->lp_0+_S1338) = _S1339;
        *(kernelContext_195->lp_0+(_S1338 + 1U)) = _S1339;
        c_24 = c_24 + 1U;
    }
    uint _S1340 = st_d_0(k_49, 14U);
    packed_float4 _S1341 = packed_float4(float4(0.0f) ) ;
    *(kernelContext_195->lp_0+_S1340) = _S1341;
    *(kernelContext_195->lp_0+(_S1340 + 1U)) = _S1341;
    return;
}

void cs_break_0(uint i_21, uint s_20, KernelContext_0 thread* kernelContext_196)
{
    uint4 _S1342 = uint4(*(kernelContext_196->bonds_0+i_21)) ;
    LpBond_0 _S1343 = lp_bond_0(i_21, kernelContext_196);
    uint _S1344 = _S1342.x;
    uint _S1345 = _S1342.y;
    thread LpBond_0 _S1346 = _S1343;
    thread float3 q_lin_5;
    thread float3 q_ang_5;
    st_force_0(&_S1346, _S1344, _S1345, 14U, 0U, &q_lin_5, &q_ang_5, kernelContext_196);
    if((kernelContext_196->params_0->load_model_0) == 1U)
    {
        thread LpBond_0 _S1347 = _S1343;
        thread float3 b_lin_0;
        thread float3 b_ang_0;
        lp_force_0(&_S1347, _S1344, _S1345, 0U, &b_lin_0, &b_ang_0, kernelContext_196);
        q_lin_5 = q_lin_5 + b_lin_0;
        q_ang_5 = q_ang_5 + b_ang_0;
    }
    thread JointBond_0 _S1348 = _S1343.law_0;
    Measures_0 _S1349 = stress_measures_0(&_S1348, q_lin_5, q_ang_5);
    JointMaterial_0 _S1350 = lp_material_0(_S1343.law_0.ids_3.x, kernelContext_196);
    float _S1351 = _S1343.law_0.geom1_0.w;
    thread JointMaterial_0 _S1352 = _S1350;
    thread JointBond_0 _S1353 = _S1343.law_0;
    thread Measures_0 _S1354 = _S1349;
    float4 _S1355 = failure_indices_0(&_S1352, &_S1353, &_S1354, _S1351);
    if((max(max(_S1355.x, _S1355.y), max(_S1355.z, _S1355.w))) >= 1.0f)
    {
        uint _S1356 = w_alive_0(kernelContext_196);
        *(kernelContext_196->work_0+(_S1356 + i_21)) = 0U;
        uint _S1357 = a_dirty_0(kernelContext_196);
        uint _S1358 = atomic_fetch_or_explicit(kernelContext_196->atom_0+(_S1357 + s_20), 1U, memory_order_relaxed);
        uint _S1359 = a_counters_1(kernelContext_196);
        uint _S1360 = atomic_fetch_add_explicit(kernelContext_196->atom_0+_S1359, 1U, memory_order_relaxed);
        uint _S1361 = atomic_fetch_add_explicit(kernelContext_196->atom_0+(_S1359 + 1U), 1U, memory_order_relaxed);
        uint _S1362 = atomic_fetch_add_explicit(kernelContext_196->atom_0+(_S1359 + 17U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_solve(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_75 [[buffer(0)]], atomic<uint> device* atom_75 [[buffer(5)]], uint device* work_75 [[buffer(4)]], packed_float4 device* frames_75 [[buffer(6)]], packed_float4 device* bodies_75 [[buffer(7)]], packed_float4 device* chunks_75 [[buffer(1)]], uint device* sort_75 [[buffer(11)]], Record_natural_0 device* manifolds_75 [[buffer(10)]], packed_float4 device* loads_75 [[buffer(12)]], packed_uint4 device* bonds_75 [[buffer(2)]], uint device* args_75 [[buffer(13)]], uint device* info_75 [[buffer(15)]], ScanParams_0 constant* scan_75 [[buffer(14)]], packed_float4 device* parts_75 [[buffer(3)]], packed_float4 device* colliders_75 [[buffer(8)]], packed_uint4 device* collider_info_75 [[buffer(9)]], packed_float4 device* view_chunks_75 [[buffer(17)]], ViewParams_0 constant* view_75 [[buffer(16)]], packed_float4 device* objects_75 [[buffer(18)]], uint device* shot_bodies_75 [[buffer(20)]], float device* lines_75 [[buffer(19)]], packed_float4 device* geometry_75 [[buffer(21)]], packed_float4 device* shots_75 [[buffer(22)]], packed_float4 device* lp_75 [[buffer(24)]], packed_float4 device* ext_77 [[buffer(23)]], packed_float4 device* engine_contacts_75 [[buffer(25)]], CrushParams_0 constant* crush_75 [[buffer(28)]], packed_float4 device* crush_rw_75 [[buffer(27)]], packed_float4 device* crush_tab_75 [[buffer(26)]])
{
    bool _S1363;
    uint j3_0;
    thread KernelContext_0 kernelContext_197;
    (&kernelContext_197)->params_0 = params_75;
    (&kernelContext_197)->atom_0 = atom_75;
    (&kernelContext_197)->work_0 = work_75;
    (&kernelContext_197)->frames_0 = frames_75;
    (&kernelContext_197)->bodies_0 = bodies_75;
    (&kernelContext_197)->chunks_0 = chunks_75;
    (&kernelContext_197)->sort_0 = sort_75;
    (&kernelContext_197)->manifolds_0 = manifolds_75;
    (&kernelContext_197)->loads_0 = loads_75;
    (&kernelContext_197)->bonds_0 = bonds_75;
    (&kernelContext_197)->args_0 = args_75;
    (&kernelContext_197)->info_0 = info_75;
    (&kernelContext_197)->scan_0 = scan_75;
    (&kernelContext_197)->parts_0 = parts_75;
    (&kernelContext_197)->colliders_0 = colliders_75;
    (&kernelContext_197)->collider_info_0 = collider_info_75;
    (&kernelContext_197)->view_chunks_0 = view_chunks_75;
    (&kernelContext_197)->view_0 = view_75;
    (&kernelContext_197)->objects_0 = objects_75;
    (&kernelContext_197)->shot_bodies_0 = shot_bodies_75;
    (&kernelContext_197)->lines_0 = lines_75;
    (&kernelContext_197)->geometry_0 = geometry_75;
    (&kernelContext_197)->shots_0 = shots_75;
    (&kernelContext_197)->lp_0 = lp_75;
    (&kernelContext_197)->ext_0 = ext_77;
    (&kernelContext_197)->engine_contacts_0 = engine_contacts_75;
    (&kernelContext_197)->crush_0 = crush_75;
    (&kernelContext_197)->crush_rw_0 = crush_rw_75;
    (&kernelContext_197)->crush_tab_0 = crush_tab_75;
    threadgroup array<uint, int(1024)> g_scan_75;
    (&kernelContext_197)->g_scan_0 = &g_scan_75;
    threadgroup array<float, int(64)> g_il_best_75;
    (&kernelContext_197)->g_il_best_0 = &g_il_best_75;
    uint _S1364 = w_cs_bodies_0(&kernelContext_197);
    uint device* _S1365 = (&kernelContext_197)->work_0+(_S1364 + group_3.x);
    uint s_21 = *_S1365;
    uint t_15 = thread_6.x;
    uint _S1366 = a_cs_0(*_S1365, &kernelContext_197);
    uint _S1367 = atomic_load_explicit((&kernelContext_197)->atom_0+(_S1366 + 6U), memory_order_relaxed);
    uint _S1368 = atomic_load_explicit((&kernelContext_197)->atom_0+(_S1366 + 5U), memory_order_relaxed);
    uint j_10 = t_15;
    for(;;)
    {
        if(j_10 < _S1368)
        {
        }
        else
        {
            break;
        }
        uint _S1369 = w_cs_list_0(&kernelContext_197);
        cs_load_0(*((&kernelContext_197)->work_0+(_S1369 + _S1367 + j_10)), s_21, &kernelContext_197);
        j_10 = j_10 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint j2_0 = t_15;
    for(;;)
    {
        if(j2_0 < _S1368)
        {
        }
        else
        {
            break;
        }
        uint _S1370 = w_cs_list_0(&kernelContext_197);
        uint device* _S1371 = (&kernelContext_197)->work_0+(_S1370 + _S1367 + j2_0);
        uint k_50 = *_S1371;
        uint _S1372 = st_f_0(*_S1371, 14U);
        if(((float4(*((&kernelContext_197)->lp_0+_S1372)) ).w) == 0.0f)
        {
            bool _S1373 = lp_factor_0(k_50, &kernelContext_197);
            _S1363 = !_S1373;
        }
        else
        {
            _S1363 = false;
        }
        if(_S1363)
        {
            (*((&kernelContext_197)->lp_0+_S1372))[int(3)] = 1.0f;
            uint _S1374 = a_counters_1(&kernelContext_197);
            uint _S1375 = atomic_fetch_or_explicit((&kernelContext_197)->atom_0+(_S1374 + 3U), 64U, memory_order_relaxed);
        }
        j2_0 = j2_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint sweep_0 = 0U;
    for(;;)
    {
        if(sweep_0 < ((&kernelContext_197)->params_0->cs_sweeps_0))
        {
        }
        else
        {
            break;
        }
        j3_0 = t_15;
        for(;;)
        {
            if(j3_0 < _S1368)
            {
            }
            else
            {
                break;
            }
            uint _S1376 = w_cs_list_0(&kernelContext_197);
            st_sweep_0(*((&kernelContext_197)->work_0+(_S1376 + _S1367 + j3_0)), 14U, sweep_0 % 2U, (&kernelContext_197)->params_0->cs_c1_0, (&kernelContext_197)->params_0->cs_c2_0, &kernelContext_197);
            j3_0 = j3_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        sweep_0 = sweep_0 + 1U;
    }
    j3_0 = t_15;
    for(;;)
    {
        if(j3_0 < _S1368)
        {
        }
        else
        {
            break;
        }
        uint _S1377 = w_cs_list_0(&kernelContext_197);
        uint device* _S1378 = (&kernelContext_197)->work_0+(_S1377 + _S1367 + j3_0);
        uint k_51 = *_S1378;
        float4 _S1379 = float4(*((&kernelContext_197)->ext_0+((&kernelContext_197)->params_0->ext_chunks_0 + 4U * *_S1378))) ;
        uint _S1380 = (as_type<uint>((_S1379.x)));
        uint _S1381 = (as_type<uint>((_S1379.y)));
        uint e_6 = 0U;
        for(;;)
        {
            if(e_6 < _S1381)
            {
            }
            else
            {
                break;
            }
            uint _S1382 = adj_bond_0(_S1380 + e_6, &kernelContext_197);
            if(((uint4(*((&kernelContext_197)->bonds_0+_S1382)) ).x) == k_51)
            {
                uint _S1383 = w_alive_0(&kernelContext_197);
                _S1363 = (*((&kernelContext_197)->work_0+(_S1383 + _S1382))) != 0U;
            }
            else
            {
                _S1363 = false;
            }
            if(_S1363)
            {
                cs_break_0(_S1382, s_21, &kernelContext_197);
            }
            e_6 = e_6 + 1U;
        }
        j3_0 = j3_0 + 256U;
    }
    return;
}

bool crushable_0(uint k_52, KernelContext_0 thread* kernelContext_198)
{
    return ((as_type<uint>(((float4(*(kernelContext_198->crush_tab_0+(2U * k_52 + 1U))) ).z))) & 1U) != 0U;
}

uint crush_flags_0(uint k_53, KernelContext_0 thread* kernelContext_199)
{
    return (as_type<uint>(((float4(*(kernelContext_199->crush_rw_0+k_53)) ).w)));
}

void crush_record_0(uint k_54, float4 step_6, uint cause_0, KernelContext_0 thread* kernelContext_200)
{
    float4 _S1384 = float4(*(kernelContext_200->crush_rw_0+k_54)) ;
    uint flags_1 = (as_type<uint>((_S1384.w)));
    uint flags_2;
    if((step_6.w) != 0.0f)
    {
        flags_2 = ((flags_1 | 1U) | 2U) | (cause_0 << 4U);
    }
    else
    {
        flags_2 = flags_1;
    }
    *(kernelContext_200->crush_rw_0+k_54) = packed_float4(float4(step_6.x, max(_S1384.y, step_6.y), max(_S1384.z, step_6.z), (as_type<float>((flags_2))))) ;
    return;
}

[[kernel]] void fr_crush_blast(uint3 id_70 [[thread_position_in_grid]], FractureParams_0 constant* params_76 [[buffer(0)]], atomic<uint> device* atom_76 [[buffer(5)]], uint device* work_76 [[buffer(4)]], packed_float4 device* frames_76 [[buffer(6)]], packed_float4 device* bodies_76 [[buffer(7)]], packed_float4 device* chunks_76 [[buffer(1)]], uint device* sort_76 [[buffer(11)]], Record_natural_0 device* manifolds_76 [[buffer(10)]], packed_float4 device* loads_76 [[buffer(12)]], packed_uint4 device* bonds_76 [[buffer(2)]], uint device* args_76 [[buffer(13)]], uint device* info_76 [[buffer(15)]], ScanParams_0 constant* scan_76 [[buffer(14)]], packed_float4 device* parts_76 [[buffer(3)]], packed_float4 device* colliders_76 [[buffer(8)]], packed_uint4 device* collider_info_76 [[buffer(9)]], packed_float4 device* view_chunks_76 [[buffer(17)]], ViewParams_0 constant* view_76 [[buffer(16)]], packed_float4 device* objects_76 [[buffer(18)]], uint device* shot_bodies_76 [[buffer(20)]], float device* lines_76 [[buffer(19)]], packed_float4 device* geometry_76 [[buffer(21)]], packed_float4 device* shots_76 [[buffer(22)]], packed_float4 device* lp_76 [[buffer(24)]], packed_float4 device* ext_78 [[buffer(23)]], packed_float4 device* engine_contacts_76 [[buffer(25)]], CrushParams_0 constant* crush_76 [[buffer(28)]], packed_float4 device* crush_rw_76 [[buffer(27)]], packed_float4 device* crush_tab_76 [[buffer(26)]])
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
    uint k_55 = id_70.x;
    bool _S1385;
    if(k_55 >= (params_76->chunk_count_0))
    {
        _S1385 = true;
    }
    else
    {
        _S1385 = ((&kernelContext_201)->crush_0->enabled_0) == 0U;
    }
    if(_S1385)
    {
        return;
    }
    float4 _S1386 = float4(*((&kernelContext_201)->crush_rw_0+((&kernelContext_201)->crush_0->r_blast_0 + k_55))) ;
    float _S1387 = _S1386.x;
    if(!(_S1387 > 0.0f))
    {
        _S1385 = !((_S1386.y) > 0.0f);
    }
    else
    {
        _S1385 = false;
    }
    if(_S1385)
    {
        _S1385 = !((_S1386.z) > 0.0f);
    }
    else
    {
        _S1385 = false;
    }
    if(_S1385)
    {
        return;
    }
    *((&kernelContext_201)->crush_rw_0+((&kernelContext_201)->crush_0->r_blast_0 + k_55)) = packed_float4(float4(0.0f) ) ;
    bool _S1388 = crushable_0(k_55, &kernelContext_201);
    if(!_S1388)
    {
        _S1385 = true;
    }
    else
    {
        uint _S1389 = crush_flags_0(k_55, &kernelContext_201);
        _S1385 = (_S1389 & 1U) != 0U;
    }
    if(_S1385)
    {
        return;
    }
    float d_22 = (float4(*((&kernelContext_201)->crush_rw_0+k_55)) ).x + _S1387;
    float4 step_7;
    if(d_22 >= 1.0f)
    {
        step_7 = float4(1.0f, _S1386.y, _S1386.z, 1.0f);
    }
    else
    {
        step_7 = float4(d_22, _S1386.y, _S1386.z, 0.0f);
    }
    crush_record_0(k_55, step_7, 3U, &kernelContext_201);
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
    float3 _S1390 = float3(0.0f) ;
    (&s_22)->d_23 = _S1390;
    (&s_22)->o_9 = _S1390;
    return s_22;
}

Sym_0 sym_outer_0(float3 r_28, float3 f_13)
{
    thread Sym_0 s_23;
    (&s_23)->d_23 = r_28 * f_13;
    float _S1391 = r_28.x;
    float _S1392 = f_13.y;
    float _S1393 = r_28.y;
    float _S1394 = f_13.x;
    float _S1395 = f_13.z;
    float _S1396 = r_28.z;
    (&s_23)->o_9 = float3(0.5f)  * float3(_S1391 * _S1392 + _S1393 * _S1394, _S1391 * _S1395 + _S1396 * _S1394, _S1393 * _S1395 + _S1396 * _S1392);
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
    Sym_0 _S1397 = sym_outer_0(r_29, f_14);
    thread Sym_0 _S1398 = *w_9;
    thread Sym_0 _S1399 = _S1397;
    Sym_0 _S1400 = sym_add_0(&_S1398, &_S1399);
    *w_9 = _S1400;
    *f_sum_0 = *f_sum_0 + f_14;
    float m_6 = length(f_14);
    *spread_0 = *spread_0 + float4(float3(m_6)  * r_29, m_6);
    return;
}

void flow_virial_0(uint k_56, const Quat_0 thread* q_28, float3 origin_9, float3 centre_1, Sym_0 thread* wf_0, float thread* total_0, KernelContext_0 thread* kernelContext_202)
{
    *wf_0 = sym_zero_0();
    *total_0 = 0.0f;
    float4 _S1401 = float4(*(kernelContext_202->crush_tab_0+(kernelContext_202->crush_0->t_adj_0 + k_56))) ;
    uint first_2 = (as_type<uint>((_S1401.x)));
    uint _S1402 = (as_type<uint>((_S1401.y)));
    uint j_11 = first_2;
    for(;;)
    {
        if(j_11 < (first_2 + _S1402))
        {
        }
        else
        {
            break;
        }
        uint bi_0 = (as_type<uint>(((float4(*(kernelContext_202->crush_tab_0+(kernelContext_202->crush_0->t_list_0 + j_11 / 4U))) )[j_11 % 4U])));
        uint _S1403 = w_alive_0(kernelContext_202);
        if((*(kernelContext_202->work_0+(_S1403 + bi_0))) == 0U)
        {
            j_11 = j_11 + 1U;
            continue;
        }
        float l_2 = (*(kernelContext_202->crush_rw_0+(kernelContext_202->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U];
        if(!(((*(kernelContext_202->crush_rw_0+(kernelContext_202->crush_0->r_flow_0 + bi_0 / 4U)))[bi_0 % 4U]) > 0.0f))
        {
            j_11 = j_11 + 1U;
            continue;
        }
        float3 _S1404 = rotate_0(q_28, (float4(*(kernelContext_202->crush_tab_0+(kernelContext_202->crush_0->t_bonds_0 + bi_0))) ).xyz);
        float3 r_30 = origin_9 + _S1404 - centre_1;
        float3 _S1405 = rotate_0(q_28, (float4(*(kernelContext_202->crush_tab_0+(kernelContext_202->crush_0->t_normal_0 + bi_0))) ).xyz);
        float3 out_n_0;
        if((dot(_S1405, r_30)) >= 0.0f)
        {
            out_n_0 = _S1405;
        }
        else
        {
            out_n_0 = - _S1405;
        }
        Sym_0 _S1406 = sym_outer_0(r_30, float3(- l_2)  * out_n_0);
        thread Sym_0 _S1407 = *wf_0;
        thread Sym_0 _S1408 = _S1406;
        Sym_0 _S1409 = sym_add_0(&_S1407, &_S1408);
        *wf_0 = _S1409;
        *total_0 = *total_0 + l_2;
        j_11 = j_11 + 1U;
    }
    return;
}

float4 bond_centroid_0(uint k_57, const Quat_0 thread* q_29, float3 origin_10, float3 centre_2, KernelContext_0 thread* kernelContext_203)
{
    float4 _S1410 = float4(*(kernelContext_203->crush_tab_0+(kernelContext_203->crush_0->t_adj_0 + k_57))) ;
    uint first_3 = (as_type<uint>((_S1410.x)));
    uint _S1411 = (as_type<uint>((_S1410.y)));
    float3 _S1412 = float3(0.0f) ;
    uint j_12 = first_3;
    float3 sum_4 = _S1412;
    float area_1 = 0.0f;
    for(;;)
    {
        if(j_12 < (first_3 + _S1411))
        {
        }
        else
        {
            break;
        }
        uint bi_1 = (as_type<uint>(((float4(*(kernelContext_203->crush_tab_0+(kernelContext_203->crush_0->t_list_0 + j_12 / 4U))) )[j_12 % 4U])));
        uint _S1413 = w_alive_0(kernelContext_203);
        if((*(kernelContext_203->work_0+(_S1413 + bi_1))) == 0U)
        {
            j_12 = j_12 + 1U;
            continue;
        }
        float4 _S1414 = float4(*(kernelContext_203->crush_tab_0+(kernelContext_203->crush_0->t_bonds_0 + bi_1))) ;
        float3 _S1415 = rotate_0(q_29, _S1414.xyz);
        float _S1416 = _S1414.w;
        float area_2 = area_1 + _S1416;
        sum_4 = sum_4 + float3(_S1416)  * (origin_10 + _S1415 - centre_2);
        area_1 = area_2;
        j_12 = j_12 + 1U;
    }
    if(area_1 > 0.0f)
    {
        sum_4 = sum_4 / float3(area_1) ;
    }
    else
    {
        sum_4 = _S1412;
    }
    return float4(sum_4, area_1);
}

Sym_0 balanced_0(const Sym_0 thread* w_10, float3 f_15, float4 spread_1, float4 bonds_77)
{
    float3 at_23;
    if((bonds_77.w) > 0.0f)
    {
        at_23 = bonds_77.xyz;
    }
    else
    {
        float _S1417 = spread_1.w;
        if(_S1417 > 0.0f)
        {
            at_23 = spread_1.xyz / float3(_S1417) ;
        }
        else
        {
            at_23 = float3(0.0f) ;
        }
    }
    thread Sym_0 _S1418 = sym_outer_0(at_23, - f_15);
    Sym_0 _S1419 = sym_add_0(w_10, &_S1418);
    return _S1419;
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
    float _S1420 = pq_0.x;
    if(!(_S1420 > 0.0f))
    {
        return 0.0f;
    }
    float _S1421 = max(pq_0.y - (m0_0.y + m0_0.z * _S1420), _S1420 - m0_0.x);
    float _S1422;
    if(_S1421 > 0.0f)
    {
        _S1422 = _S1421;
    }
    else
    {
        _S1422 = 0.0f;
    }
    return _S1422;
}

float4 crush_two_0(const Sym_0 thread* a_11, float dt_a_0, const Sym_0 thread* b_54, float dt_b_0, uint k_58, float damage_2, KernelContext_0 thread* kernelContext_204)
{
    uint _S1423 = 2U * k_58;
    float4 _S1424 = float4(*(kernelContext_204->crush_tab_0+_S1423)) ;
    float4 _S1425 = float4(*(kernelContext_204->crush_tab_0+(_S1423 + 1U))) ;
    float _S1426 = _S1425.y;
    float2 _S1427 = crush_invariants_0(a_11, _S1426);
    float2 _S1428 = crush_invariants_0(b_54, _S1426);
    float ea_0 = crush_excess_0(_S1427, _S1424);
    bool _S1429 = dt_b_0 > 0.0f;
    float eb_0;
    if(_S1429)
    {
        eb_0 = crush_excess_0(_S1428, _S1424);
    }
    else
    {
        eb_0 = 0.0f;
    }
    float d_24 = damage_2 + (ea_0 * ea_0 * dt_a_0 + eb_0 * eb_0 * dt_b_0) / (_S1425.x * _S1424.w);
    float2 peak_2;
    if(_S1429)
    {
        peak_2 = max(_S1427, _S1428);
    }
    else
    {
        peak_2 = _S1427;
    }
    if(d_24 >= 1.0f)
    {
        return float4(1.0f, peak_2, 1.0f);
    }
    return float4(d_24, peak_2, 0.0f);
}

void crush_loads_flow_0(uint k_59, KernelContext_0 thread* kernelContext_205)
{
    uint _S1430 = body_of_0(k_59, kernelContext_205);
    uint b_55 = 7U * _S1430;
    thread Quat_0 q_30;
    thread float3 origin_11;
    structure_pose_0(_S1430, b_55, &q_30, &origin_11, kernelContext_205);
    uint _S1431 = 4U * k_59;
    float4 _S1432 = float4(*(kernelContext_205->chunks_0+_S1431)) ;
    float4 _S1433 = float4(*(kernelContext_205->chunks_0+(_S1431 + 3U))) ;
    float3 _S1434 = _S1432.xyz;
    thread Quat_0 _S1435 = q_30;
    float3 _S1436 = rotate_0(&_S1435, _S1434);
    float3 centre_3 = origin_11 + _S1436;
    float inv_mass_8 = (float4(*(kernelContext_205->bodies_0+b_55)) ).w;
    float _S1437;
    if(inv_mass_8 > 0.0f)
    {
        _S1437 = 1.0f / inv_mass_8;
    }
    else
    {
        _S1437 = float_bits_0(2139095040U);
    }
    float dt_2 = kernelContext_205->params_0->dt_0;
    Sym_0 _S1438 = sym_zero_0();
    thread Sym_0 wa_0 = _S1438;
    float3 _S1439 = float3(0.0f) ;
    thread float3 fa_0 = _S1439;
    float4 _S1440 = float4(0.0f) ;
    thread float4 sa_0 = _S1440;
    thread Sym_0 wb_0 = _S1438;
    thread float3 fb_1 = _S1439;
    thread float4 sb_2 = _S1440;
    uint _S1441 = kernelContext_205->sort_0[k_59];
    uint _S1442 = kernelContext_205->sort_0[kernelContext_205->params_0->chunk_count_0 + 1U + k_59];
    float tau_3 = 0.0f;
    uint i_22 = 0U;
    for(;;)
    {
        if(i_22 < _S1442)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_31 = kernelContext_205->manifolds_0[_S1441 + i_22];
        float4 _S1443 = float4(r_31.impulse_0) ;
        float3 _S1444 = _S1443.xyz;
        float j_13 = length(_S1444);
        if(j_13 == 0.0f)
        {
            i_22 = i_22 + 1U;
            continue;
        }
        float4 _S1445 = float4(r_31.other_0) ;
        float m_other_2 = _S1445.x;
        float m_7;
        if(isinf(_S1437))
        {
            if(isinf(m_other_2))
            {
                m_7 = _S1432.w;
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
                m_7 = _S1437;
            }
            else
            {
                m_7 = _S1437 * m_other_2 / (_S1437 + m_other_2);
            }
        }
        float _S1446 = _S1443.w;
        float _S1447 = min(j_13, 2.0f * m_7 * max(_S1446 - kernelContext_205->params_0->band_0, 0.0f));
        float3 rr_0 = (float4(r_31.point_0) ).xyz - centre_3;
        float3 dir_1 = _S1444 / float3(j_13) ;
        if(_S1447 < j_13)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, rr_0, dir_1 * float3(((j_13 - _S1447) / dt_2)) );
        }
        float tau_4;
        if(_S1447 > 0.0f)
        {
            float _S1448 = _S1433.y;
            float duration_2 = hertz_duration_0(m_7, min(_S1445.z, _S1433.z), 1.0f / ((1.0f - _S1448 * _S1448) / _S1433.x + 1.0f / _S1445.y), _S1446);
            add_load_0(&wb_0, &fb_1, &sb_2, rr_0, dir_1 * float3((3.14159274101257324f * _S1447 / (2.0f * duration_2))) );
            tau_4 = max(tau_3, min(duration_2, dt_2));
        }
        else
        {
            tau_4 = tau_3;
        }
        tau_3 = tau_4;
        i_22 = i_22 + 1U;
    }
    uint from_4 = kernelContext_205->params_0->parity_0;
    uint _S1449 = w_pulses_0(kernelContext_205);
    uint _S1450 = _S1449 + _S1431 + 2U * from_4;
    uint off_1 = *(kernelContext_205->work_0+_S1450);
    uint _S1451 = *(kernelContext_205->work_0+(_S1450 + 1U));
    uint _S1452 = l_pool_0(from_4, kernelContext_205);
    uint _S1453 = _S1452 + 3U * off_1;
    uint p_22 = 0U;
    for(;;)
    {
        if(p_22 < _S1451)
        {
        }
        else
        {
            break;
        }
        uint _S1454 = _S1453 + 3U * p_22;
        float4 _S1455 = float4(*(kernelContext_205->loads_0+_S1454)) ;
        float4 _S1456 = float4(*(kernelContext_205->loads_0+(_S1454 + 1U))) ;
        float4 _S1457 = float4(*(kernelContext_205->loads_0+(_S1454 + 2U))) ;
        float mean_2 = pulse_mean_0(_S1455.x, _S1455.y, _S1455.z, _S1455.w, _S1456.w, dt_2);
        if(mean_2 != 0.0f)
        {
            add_load_0(&wa_0, &fa_0, &sa_0, _S1456.xyz - centre_3, _S1457.xyz * float3(mean_2) );
        }
        p_22 = p_22 + 1U;
    }
    thread Quat_0 _S1458 = q_30;
    thread Sym_0 wf_1;
    thread float flow_total_0;
    flow_virial_0(k_59, &_S1458, origin_11, centre_3, &wf_1, &flow_total_0, kernelContext_205);
    bool contact_18;
    if((sa_0.w) > 0.0f)
    {
        contact_18 = true;
    }
    else
    {
        contact_18 = (sb_2.w) > 0.0f;
    }
    bool _S1459;
    if(!contact_18)
    {
        _S1459 = !(flow_total_0 > 0.0f);
    }
    else
    {
        _S1459 = false;
    }
    if(_S1459)
    {
        return;
    }
    Sym_0 _S1460 = wf_1;
    Sym_0 _S1461 = wf_1;
    Sym_0 peak_contact_0;
    Sym_0 base_3;
    Sym_0 peak_3;
    if(contact_18)
    {
        thread Quat_0 _S1462 = q_30;
        float4 _S1463 = bond_centroid_0(k_59, &_S1462, origin_11, centre_3, kernelContext_205);
        thread Sym_0 _S1464 = wa_0;
        Sym_0 _S1465 = balanced_0(&_S1464, fa_0, sa_0, _S1463);
        thread Sym_0 _S1466 = _S1465;
        thread Sym_0 _S1467 = wf_1;
        Sym_0 _S1468 = sym_add_0(&_S1466, &_S1467);
        thread Sym_0 _S1469 = wa_0;
        thread Sym_0 _S1470 = wb_0;
        Sym_0 _S1471 = sym_add_0(&_S1469, &_S1470);
        float3 _S1472 = fa_0 + fb_1;
        float4 _S1473 = sa_0 + sb_2;
        thread Sym_0 _S1474 = _S1471;
        Sym_0 _S1475 = balanced_0(&_S1474, _S1472, _S1473, _S1463);
        thread Sym_0 _S1476 = _S1475;
        thread Sym_0 _S1477 = wf_1;
        Sym_0 _S1478 = sym_add_0(&_S1476, &_S1477);
        peak_contact_0 = _S1475;
        base_3 = _S1468;
        peak_3 = _S1478;
    }
    else
    {
        peak_contact_0 = _S1438;
        base_3 = _S1460;
        peak_3 = _S1461;
    }
    thread Sym_0 _S1479 = wf_1;
    float _S1480 = sym_norm2_0(&_S1479);
    thread Sym_0 _S1481 = peak_contact_0;
    float _S1482 = sym_norm2_0(&_S1481);
    uint cause_1;
    if(_S1480 >= _S1482)
    {
        cause_1 = 4U;
    }
    else
    {
        cause_1 = 1U;
    }
    float _S1483 = dt_2 - tau_3;
    float _S1484 = (float4(*(kernelContext_205->crush_rw_0+k_59)) ).x;
    thread Sym_0 _S1485 = base_3;
    thread Sym_0 _S1486 = peak_3;
    float4 _S1487 = crush_two_0(&_S1485, _S1483, &_S1486, tau_3, k_59, _S1484, kernelContext_205);
    crush_record_0(k_59, _S1487, cause_1, kernelContext_205);
    return;
}

[[kernel]] void fr_crush_loads(uint3 id_71 [[thread_position_in_grid]], FractureParams_0 constant* params_77 [[buffer(0)]], atomic<uint> device* atom_77 [[buffer(5)]], uint device* work_77 [[buffer(4)]], packed_float4 device* frames_77 [[buffer(6)]], packed_float4 device* bodies_77 [[buffer(7)]], packed_float4 device* chunks_77 [[buffer(1)]], uint device* sort_77 [[buffer(11)]], Record_natural_0 device* manifolds_77 [[buffer(10)]], packed_float4 device* loads_77 [[buffer(12)]], packed_uint4 device* bonds_78 [[buffer(2)]], uint device* args_77 [[buffer(13)]], uint device* info_77 [[buffer(15)]], ScanParams_0 constant* scan_77 [[buffer(14)]], packed_float4 device* parts_77 [[buffer(3)]], packed_float4 device* colliders_77 [[buffer(8)]], packed_uint4 device* collider_info_77 [[buffer(9)]], packed_float4 device* view_chunks_77 [[buffer(17)]], ViewParams_0 constant* view_77 [[buffer(16)]], packed_float4 device* objects_77 [[buffer(18)]], uint device* shot_bodies_77 [[buffer(20)]], float device* lines_77 [[buffer(19)]], packed_float4 device* geometry_77 [[buffer(21)]], packed_float4 device* shots_77 [[buffer(22)]], packed_float4 device* lp_77 [[buffer(24)]], packed_float4 device* ext_79 [[buffer(23)]], packed_float4 device* engine_contacts_77 [[buffer(25)]], CrushParams_0 constant* crush_77 [[buffer(28)]], packed_float4 device* crush_rw_77 [[buffer(27)]], packed_float4 device* crush_tab_77 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_206;
    (&kernelContext_206)->params_0 = params_77;
    (&kernelContext_206)->atom_0 = atom_77;
    (&kernelContext_206)->work_0 = work_77;
    (&kernelContext_206)->frames_0 = frames_77;
    (&kernelContext_206)->bodies_0 = bodies_77;
    (&kernelContext_206)->chunks_0 = chunks_77;
    (&kernelContext_206)->sort_0 = sort_77;
    (&kernelContext_206)->manifolds_0 = manifolds_77;
    (&kernelContext_206)->loads_0 = loads_77;
    (&kernelContext_206)->bonds_0 = bonds_78;
    (&kernelContext_206)->args_0 = args_77;
    (&kernelContext_206)->info_0 = info_77;
    (&kernelContext_206)->scan_0 = scan_77;
    (&kernelContext_206)->parts_0 = parts_77;
    (&kernelContext_206)->colliders_0 = colliders_77;
    (&kernelContext_206)->collider_info_0 = collider_info_77;
    (&kernelContext_206)->view_chunks_0 = view_chunks_77;
    (&kernelContext_206)->view_0 = view_77;
    (&kernelContext_206)->objects_0 = objects_77;
    (&kernelContext_206)->shot_bodies_0 = shot_bodies_77;
    (&kernelContext_206)->lines_0 = lines_77;
    (&kernelContext_206)->geometry_0 = geometry_77;
    (&kernelContext_206)->shots_0 = shots_77;
    (&kernelContext_206)->lp_0 = lp_77;
    (&kernelContext_206)->ext_0 = ext_79;
    (&kernelContext_206)->engine_contacts_0 = engine_contacts_77;
    (&kernelContext_206)->crush_0 = crush_77;
    (&kernelContext_206)->crush_rw_0 = crush_rw_77;
    (&kernelContext_206)->crush_tab_0 = crush_tab_77;
    threadgroup array<uint, int(1024)> g_scan_77;
    (&kernelContext_206)->g_scan_0 = &g_scan_77;
    threadgroup array<float, int(64)> g_il_best_77;
    (&kernelContext_206)->g_il_best_0 = &g_il_best_77;
    uint k_60 = id_71.x;
    bool _S1488;
    if(k_60 >= (params_77->chunk_count_0))
    {
        _S1488 = true;
    }
    else
    {
        _S1488 = ((&kernelContext_206)->crush_0->enabled_0) == 0U;
    }
    if(_S1488)
    {
        _S1488 = true;
    }
    else
    {
        bool _S1489 = crushable_0(k_60, &kernelContext_206);
        _S1488 = !_S1489;
    }
    if(_S1488)
    {
        _S1488 = true;
    }
    else
    {
        uint _S1490 = crush_flags_0(k_60, &kernelContext_206);
        _S1488 = (_S1490 & 1U) != 0U;
    }
    if(_S1488)
    {
        return;
    }
    if(((&kernelContext_206)->crush_0->flow_0) != 0U)
    {
        crush_loads_flow_0(k_60, &kernelContext_206);
        return;
    }
    uint _S1491 = body_of_0(k_60, &kernelContext_206);
    uint b_56 = 7U * _S1491;
    thread Quat_0 q_31;
    thread float3 origin_12;
    structure_pose_0(_S1491, b_56, &q_31, &origin_12, &kernelContext_206);
    uint _S1492 = 4U * k_60;
    float4 _S1493 = float4(*((&kernelContext_206)->chunks_0+_S1492)) ;
    float4 _S1494 = float4(*((&kernelContext_206)->chunks_0+(_S1492 + 3U))) ;
    float3 _S1495 = _S1493.xyz;
    thread Quat_0 _S1496 = q_31;
    float3 _S1497 = rotate_0(&_S1496, _S1495);
    float3 centre_4 = origin_12 + _S1497;
    float inv_mass_9 = (float4(*((&kernelContext_206)->bodies_0+b_56)) ).w;
    float _S1498;
    if(inv_mass_9 > 0.0f)
    {
        _S1498 = 1.0f / inv_mass_9;
    }
    else
    {
        _S1498 = float_bits_0(2139095040U);
    }
    float dt_3 = (&kernelContext_206)->params_0->dt_0;
    Sym_0 _S1499 = sym_zero_0();
    thread Sym_0 wa_1 = _S1499;
    float3 _S1500 = float3(0.0f) ;
    thread float3 fa_1 = _S1500;
    float4 _S1501 = float4(0.0f) ;
    thread float4 sa_1 = _S1501;
    thread Sym_0 wb_1 = _S1499;
    thread float3 fb_2 = _S1500;
    thread float4 sb_3 = _S1501;
    uint _S1502 = (&kernelContext_206)->sort_0[k_60];
    uint _S1503 = (&kernelContext_206)->sort_0[params_77->chunk_count_0 + 1U + k_60];
    float tau_5 = 0.0f;
    uint i_23 = 0U;
    for(;;)
    {
        if(i_23 < _S1503)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_32 = (&kernelContext_206)->manifolds_0[_S1502 + i_23];
        float4 _S1504 = float4(r_32.impulse_0) ;
        float3 _S1505 = _S1504.xyz;
        float j_14 = length(_S1505);
        if(j_14 == 0.0f)
        {
            i_23 = i_23 + 1U;
            continue;
        }
        float4 _S1506 = float4(r_32.other_0) ;
        float m_other_3 = _S1506.x;
        float m_8;
        if(isinf(_S1498))
        {
            if(isinf(m_other_3))
            {
                m_8 = _S1493.w;
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
                m_8 = _S1498;
            }
            else
            {
                m_8 = _S1498 * m_other_3 / (_S1498 + m_other_3);
            }
        }
        float _S1507 = _S1504.w;
        float _S1508 = min(j_14, 2.0f * m_8 * max(_S1507 - (&kernelContext_206)->params_0->band_0, 0.0f));
        float3 rr_1 = (float4(r_32.point_0) ).xyz - centre_4;
        float3 dir_2 = _S1505 / float3(j_14) ;
        if(_S1508 < j_14)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, rr_1, dir_2 * float3(((j_14 - _S1508) / dt_3)) );
        }
        float tau_6;
        if(_S1508 > 0.0f)
        {
            float _S1509 = _S1494.y;
            float duration_3 = hertz_duration_0(m_8, min(_S1506.z, _S1494.z), 1.0f / ((1.0f - _S1509 * _S1509) / _S1494.x + 1.0f / _S1506.y), _S1507);
            add_load_0(&wb_1, &fb_2, &sb_3, rr_1, dir_2 * float3((3.14159274101257324f * _S1508 / (2.0f * duration_3))) );
            tau_6 = max(tau_5, min(duration_3, dt_3));
        }
        else
        {
            tau_6 = tau_5;
        }
        tau_5 = tau_6;
        i_23 = i_23 + 1U;
    }
    uint from_5 = (&kernelContext_206)->params_0->parity_0;
    uint _S1510 = w_pulses_0(&kernelContext_206);
    uint _S1511 = _S1510 + _S1492 + 2U * from_5;
    uint device* _S1512 = (&kernelContext_206)->work_0+_S1511;
    uint _S1513 = *((&kernelContext_206)->work_0+(_S1511 + 1U));
    uint _S1514 = l_pool_0(from_5, &kernelContext_206);
    uint _S1515 = _S1514 + 3U * *_S1512;
    uint p_23 = 0U;
    for(;;)
    {
        if(p_23 < _S1513)
        {
        }
        else
        {
            break;
        }
        uint _S1516 = _S1515 + 3U * p_23;
        float4 _S1517 = float4(*((&kernelContext_206)->loads_0+_S1516)) ;
        float4 _S1518 = float4(*((&kernelContext_206)->loads_0+(_S1516 + 1U))) ;
        float4 _S1519 = float4(*((&kernelContext_206)->loads_0+(_S1516 + 2U))) ;
        float mean_3 = pulse_mean_0(_S1517.x, _S1517.y, _S1517.z, _S1517.w, _S1518.w, dt_3);
        if(mean_3 != 0.0f)
        {
            add_load_0(&wa_1, &fa_1, &sa_1, _S1518.xyz - centre_4, _S1519.xyz * float3(mean_3) );
        }
        p_23 = p_23 + 1U;
    }
    if(!((sa_1.w) > 0.0f))
    {
        _S1488 = !((sb_3.w) > 0.0f);
    }
    else
    {
        _S1488 = false;
    }
    if(_S1488)
    {
        return;
    }
    thread Quat_0 _S1520 = q_31;
    float4 _S1521 = bond_centroid_0(k_60, &_S1520, origin_12, centre_4, &kernelContext_206);
    thread Sym_0 _S1522 = wa_1;
    Sym_0 _S1523 = balanced_0(&_S1522, fa_1, sa_1, _S1521);
    thread Sym_0 _S1524 = wa_1;
    thread Sym_0 _S1525 = wb_1;
    Sym_0 _S1526 = sym_add_0(&_S1524, &_S1525);
    float3 _S1527 = fa_1 + fb_2;
    float4 _S1528 = sa_1 + sb_3;
    thread Sym_0 _S1529 = _S1526;
    Sym_0 _S1530 = balanced_0(&_S1529, _S1527, _S1528, _S1521);
    float _S1531 = dt_3 - tau_5;
    float _S1532 = (float4(*((&kernelContext_206)->crush_rw_0+k_60)) ).x;
    thread Sym_0 _S1533 = _S1523;
    thread Sym_0 _S1534 = _S1530;
    float4 _S1535 = crush_two_0(&_S1533, _S1531, &_S1534, tau_5, k_60, _S1532, &kernelContext_206);
    crush_record_0(k_60, _S1535, 1U, &kernelContext_206);
    return;
}

uint a_cover_0(KernelContext_0 thread* kernelContext_207)
{
    uint _S1536 = a_moved_0(kernelContext_207);
    return _S1536 + 2U * kernelContext_207->params_0->ev_moved_cap_0;
}

[[kernel]] void fr_strike_patch(uint3 id_72 [[thread_position_in_grid]], FractureParams_0 constant* params_78 [[buffer(0)]], atomic<uint> device* atom_78 [[buffer(5)]], uint device* work_78 [[buffer(4)]], packed_float4 device* frames_78 [[buffer(6)]], packed_float4 device* bodies_78 [[buffer(7)]], packed_float4 device* chunks_78 [[buffer(1)]], uint device* sort_78 [[buffer(11)]], Record_natural_0 device* manifolds_78 [[buffer(10)]], packed_float4 device* loads_78 [[buffer(12)]], packed_uint4 device* bonds_79 [[buffer(2)]], uint device* args_78 [[buffer(13)]], uint device* info_78 [[buffer(15)]], ScanParams_0 constant* scan_78 [[buffer(14)]], packed_float4 device* parts_78 [[buffer(3)]], packed_float4 device* colliders_78 [[buffer(8)]], packed_uint4 device* collider_info_78 [[buffer(9)]], packed_float4 device* view_chunks_78 [[buffer(17)]], ViewParams_0 constant* view_78 [[buffer(16)]], packed_float4 device* objects_78 [[buffer(18)]], uint device* shot_bodies_78 [[buffer(20)]], float device* lines_78 [[buffer(19)]], packed_float4 device* geometry_78 [[buffer(21)]], packed_float4 device* shots_78 [[buffer(22)]], packed_float4 device* lp_78 [[buffer(24)]], packed_float4 device* ext_80 [[buffer(23)]], packed_float4 device* engine_contacts_78 [[buffer(25)]], CrushParams_0 constant* crush_78 [[buffer(28)]], packed_float4 device* crush_rw_78 [[buffer(27)]], packed_float4 device* crush_tab_78 [[buffer(26)]])
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
    (&kernelContext_208)->bonds_0 = bonds_79;
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
    uint b_57 = id_72.x;
    bool _S1537;
    if(b_57 >= (params_78->body_count_0))
    {
        _S1537 = true;
    }
    else
    {
        _S1537 = ((&kernelContext_208)->params_0->shot_count_0) == 0U;
    }
    if(_S1537)
    {
        _S1537 = true;
    }
    else
    {
        _S1537 = ((&kernelContext_208)->crush_0->enabled_0) == 0U;
    }
    if(_S1537)
    {
        return;
    }
    uint bill_0 = (&kernelContext_208)->crush_0->r_bill_0 + 4U * b_57;
    packed_float4 _S1538 = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_208)->crush_rw_0+bill_0) = _S1538;
    uint _S1539 = bill_0 + 3U;
    *((&kernelContext_208)->crush_rw_0+_S1539) = _S1538;
    thread uint shot_10;
    bool _S1540 = struck_0(b_57, &shot_10, &kernelContext_208);
    if(!_S1540)
    {
        return;
    }
    uint _S1541 = a_strike_0(&kernelContext_208);
    uint _S1542 = _S1541 + 9U * b_57;
    uint k_61 = atomic_load_explicit((&kernelContext_208)->atom_0+(_S1542 + 8U), memory_order_relaxed);
    uint _S1543 = atomic_load_explicit((&kernelContext_208)->atom_0+(_S1542 + 6U), memory_order_relaxed);
    float t_in_10 = (as_type<float>((_S1543)));
    if(!(t_in_10 <= ((&kernelContext_208)->params_0->dt_0)))
    {
        _S1537 = true;
    }
    else
    {
        _S1537 = k_61 == 4294967295U;
    }
    if(_S1537)
    {
        return;
    }
    uint _S1544 = strike_at_0(b_57, &kernelContext_208);
    float4 _S1545 = float4(*((&kernelContext_208)->frames_0+_S1544)) ;
    float3 _S1546 = (float4(*((&kernelContext_208)->frames_0+(_S1544 + 1U))) ).xyz;
    float speed_15 = length(_S1546);
    float3 d_25 = _S1546 / float3(max(speed_15, 9.999999960041972e-13f)) ;
    float4 _S1547 = float4(*((&kernelContext_208)->shots_0+2U * shot_10)) ;
    float radius_1 = _S1547.y;
    float _S1548 = max(_S1547.z, 0.00100000004749745f);
    float m_imp_1 = _S1547.w;
    float inv_mass_10 = (float4(*((&kernelContext_208)->bodies_0+7U * b_57)) ).w;
    bool _S1549 = inv_mass_10 == 0.0f;
    float m_body_2;
    if(_S1549)
    {
        m_body_2 = 0.0f;
    }
    else
    {
        m_body_2 = 1.0f / inv_mass_10;
    }
    float m_eff_9;
    if(_S1549)
    {
        m_eff_9 = m_imp_1;
    }
    else
    {
        m_eff_9 = m_imp_1 * m_body_2 / (m_imp_1 + m_body_2);
    }
    float modulus_2 = 1.0f / (1.0f / (float4(*((&kernelContext_208)->shots_0+(2U * shot_10 + 1U))) ).x + 1.0f / (float4(*((&kernelContext_208)->geometry_0+(3U * k_61 + 1U))) ).w);
    float _S1550 = max(speed_15, 0.00100000004749745f);
    float tau_7 = 2.86999988555908203f * pow(m_eff_9 * m_eff_9 / (_S1548 * modulus_2 * modulus_2 * _S1550), 0.20000000298023224f);
    float force_3 = m_eff_9 * speed_15 / tau_7;
    float3 contact_19 = _S1545.xyz + _S1546 * float3(t_in_10)  + d_25 * float3(radius_1) ;
    float delta_3 = pow(15.0f * m_eff_9 * _S1550 * _S1550 / (16.0f * modulus_2 * sqrt(_S1548)), 0.40000000596046448f);
    float a_h_0 = sqrt(radius_1 * delta_3);
    bool _S1551 = crushable_0(k_61, &kernelContext_208);
    float p_cap_0;
    if(_S1551)
    {
        p_cap_0 = (float4(*((&kernelContext_208)->crush_tab_0+2U * k_61)) ).x;
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
    float _S1552 = min(radius_1, max(a_h_0, a_p_0));
    float _S1553 = max(delta_3, radius_1 - sqrt(max(radius_1 * radius_1 - _S1552 * _S1552, 0.0f)));
    *((&kernelContext_208)->crush_rw_0+(bill_0 + 1U)) = packed_float4(float4(contact_19, 0.0f)) ;
    *((&kernelContext_208)->crush_rw_0+(bill_0 + 2U)) = packed_float4(float4(d_25, _S1552)) ;
    *((&kernelContext_208)->crush_rw_0+_S1539) = packed_float4(float4(force_3, tau_7, _S1553, float(k_61 + 1U))) ;
    uint _S1554 = a_cover_0(&kernelContext_208);
    uint _S1555 = _S1554 + 128U * b_57;
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
        uint _S1556 = _S1555 + 2U * s_26;
        atomic_store_explicit((&kernelContext_208)->atom_0+_S1556, 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_208)->atom_0+(_S1556 + 1U), 4294967295U, memory_order_relaxed);
        s_26 = s_26 + 1U;
    }
    return;
}

bool cover_patch_0(uint k_62, uint thread* b_58, float4 thread* p1_0, float4 thread* p2_2, KernelContext_0 thread* kernelContext_209)
{
    uint _S1557 = body_of_0(k_62, kernelContext_209);
    *b_58 = _S1557;
    float4 _S1558 = float4(0.0f) ;
    *p1_0 = _S1558;
    *p2_2 = _S1558;
    thread uint shot_11;
    bool _S1559 = struck_0(*b_58, &shot_11, kernelContext_209);
    bool _S1560;
    if(!_S1559)
    {
        _S1560 = true;
    }
    else
    {
        uint _S1561 = crush_flags_0(k_62, kernelContext_209);
        _S1560 = (_S1561 & 1U) != 0U;
    }
    if(_S1560)
    {
        return false;
    }
    uint bill_1 = kernelContext_209->crush_0->r_bill_0 + 4U * *b_58;
    if(((float4(*(kernelContext_209->crush_rw_0+(bill_1 + 3U))) ).w) == 0.0f)
    {
        return false;
    }
    *p1_0 = float4((float4(*(kernelContext_209->crush_rw_0+(bill_1 + 1U))) ).xyz, (float4(*(kernelContext_209->shots_0+2U * shot_11)) ).y);
    *p2_2 = float4(*(kernelContext_209->crush_rw_0+(bill_1 + 2U))) ;
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

float cover_depth_0(uint k_63, uint s_28, float3 contact_20, float3 d_27, float a_13, float radius_2, KernelContext_0 thread* kernelContext_210)
{
    thread float rho_1;
    float _S1562 = 2.0f * radius_2;
    float3 _S1563 = contact_20 + cover_offset_0(s_28, d_27, a_13, &rho_1) - d_27 * float3(_S1562) ;
    thread Quat_0 q_32;
    float3 _S1564 = chunk_box_world_0(k_63, &q_32, kernelContext_210);
    uint _S1565 = 3U * k_63;
    Quat_0 _S1566 = quat_of_0(float4(*(kernelContext_210->geometry_0+(_S1565 + 2U))) );
    thread Quat_0 _S1567 = q_32;
    thread Quat_0 _S1568 = _S1566;
    Quat_0 _S1569 = quat_mul_0(&_S1567, &_S1568);
    thread Quat_0 _S1570 = _S1569;
    Quat_0 _S1571 = normalized_0(&_S1570);
    float3 _S1572 = (float4(*(kernelContext_210->geometry_0+(_S1565 + 1U))) ).xyz;
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
        thread Quat_0 _S1573 = _S1571;
        float3 _S1574 = rotate_0(&_S1573, e_8);
        float o_10 = dot(_S1563 - _S1564, _S1574);
        float dv_2 = dot(d_27, _S1574);
        if((abs(dv_2)) < 9.99999971718068537e-10f)
        {
            if((abs(o_10)) > (_S1572[ax_4]))
            {
                return 1.00000001504746622e+30f;
            }
            ax_4 = ax_4 + 1U;
            continue;
        }
        float ta_3 = (- _S1572[ax_4] - o_10) / dv_2;
        float tb_3 = (_S1572[ax_4] - o_10) / dv_2;
        float _S1575 = min(t1_4, max(ta_3, tb_3));
        t0_3 = max(t0_3, min(ta_3, tb_3));
        t1_4 = _S1575;
        ax_4 = ax_4 + 1U;
    }
    bool _S1576;
    if(!(t0_3 <= t1_4))
    {
        _S1576 = true;
    }
    else
    {
        _S1576 = t1_4 < 0.0f;
    }
    if(_S1576)
    {
        return 1.00000001504746622e+30f;
    }
    return t0_3 - _S1562;
}

[[kernel]] void fr_strike_cover(uint3 id_73 [[thread_position_in_grid]], FractureParams_0 constant* params_79 [[buffer(0)]], atomic<uint> device* atom_79 [[buffer(5)]], uint device* work_79 [[buffer(4)]], packed_float4 device* frames_79 [[buffer(6)]], packed_float4 device* bodies_79 [[buffer(7)]], packed_float4 device* chunks_79 [[buffer(1)]], uint device* sort_79 [[buffer(11)]], Record_natural_0 device* manifolds_79 [[buffer(10)]], packed_float4 device* loads_79 [[buffer(12)]], packed_uint4 device* bonds_80 [[buffer(2)]], uint device* args_79 [[buffer(13)]], uint device* info_79 [[buffer(15)]], ScanParams_0 constant* scan_79 [[buffer(14)]], packed_float4 device* parts_79 [[buffer(3)]], packed_float4 device* colliders_79 [[buffer(8)]], packed_uint4 device* collider_info_79 [[buffer(9)]], packed_float4 device* view_chunks_79 [[buffer(17)]], ViewParams_0 constant* view_79 [[buffer(16)]], packed_float4 device* objects_79 [[buffer(18)]], uint device* shot_bodies_79 [[buffer(20)]], float device* lines_79 [[buffer(19)]], packed_float4 device* geometry_79 [[buffer(21)]], packed_float4 device* shots_79 [[buffer(22)]], packed_float4 device* lp_79 [[buffer(24)]], packed_float4 device* ext_81 [[buffer(23)]], packed_float4 device* engine_contacts_79 [[buffer(25)]], CrushParams_0 constant* crush_79 [[buffer(28)]], packed_float4 device* crush_rw_79 [[buffer(27)]], packed_float4 device* crush_tab_79 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_211;
    (&kernelContext_211)->params_0 = params_79;
    (&kernelContext_211)->atom_0 = atom_79;
    (&kernelContext_211)->work_0 = work_79;
    (&kernelContext_211)->frames_0 = frames_79;
    (&kernelContext_211)->bodies_0 = bodies_79;
    (&kernelContext_211)->chunks_0 = chunks_79;
    (&kernelContext_211)->sort_0 = sort_79;
    (&kernelContext_211)->manifolds_0 = manifolds_79;
    (&kernelContext_211)->loads_0 = loads_79;
    (&kernelContext_211)->bonds_0 = bonds_80;
    (&kernelContext_211)->args_0 = args_79;
    (&kernelContext_211)->info_0 = info_79;
    (&kernelContext_211)->scan_0 = scan_79;
    (&kernelContext_211)->parts_0 = parts_79;
    (&kernelContext_211)->colliders_0 = colliders_79;
    (&kernelContext_211)->collider_info_0 = collider_info_79;
    (&kernelContext_211)->view_chunks_0 = view_chunks_79;
    (&kernelContext_211)->view_0 = view_79;
    (&kernelContext_211)->objects_0 = objects_79;
    (&kernelContext_211)->shot_bodies_0 = shot_bodies_79;
    (&kernelContext_211)->lines_0 = lines_79;
    (&kernelContext_211)->geometry_0 = geometry_79;
    (&kernelContext_211)->shots_0 = shots_79;
    (&kernelContext_211)->lp_0 = lp_79;
    (&kernelContext_211)->ext_0 = ext_81;
    (&kernelContext_211)->engine_contacts_0 = engine_contacts_79;
    (&kernelContext_211)->crush_0 = crush_79;
    (&kernelContext_211)->crush_rw_0 = crush_rw_79;
    (&kernelContext_211)->crush_tab_0 = crush_tab_79;
    threadgroup array<uint, int(1024)> g_scan_79;
    (&kernelContext_211)->g_scan_0 = &g_scan_79;
    threadgroup array<float, int(64)> g_il_best_79;
    (&kernelContext_211)->g_il_best_0 = &g_il_best_79;
    uint k_64 = id_73.x;
    bool _S1577;
    if(k_64 >= (params_79->chunk_count_0))
    {
        _S1577 = true;
    }
    else
    {
        _S1577 = ((&kernelContext_211)->params_0->shot_count_0) == 0U;
    }
    if(_S1577)
    {
        _S1577 = true;
    }
    else
    {
        _S1577 = ((&kernelContext_211)->crush_0->enabled_0) == 0U;
    }
    if(_S1577)
    {
        return;
    }
    thread uint b_59;
    thread float4 p1_1;
    thread float4 p2_3;
    bool _S1578 = cover_patch_0(k_64, &b_59, &p1_1, &p2_3, &kernelContext_211);
    if(!_S1578)
    {
        return;
    }
    uint _S1579 = a_cover_0(&kernelContext_211);
    uint _S1580 = _S1579 + 128U * b_59;
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
        float _S1581 = cover_depth_0(k_64, s_29, p1_1.xyz, p2_3.xyz, p2_3.w, p1_1.w, &kernelContext_211);
        if(_S1581 < 5.0000000752373311e+29f)
        {
            uint _S1582 = atomic_fetch_min_explicit((&kernelContext_211)->atom_0+(_S1580 + 2U * s_29), ordered_0(_S1581), memory_order_relaxed);
        }
        s_29 = s_29 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_claim(uint3 id_74 [[thread_position_in_grid]], FractureParams_0 constant* params_80 [[buffer(0)]], atomic<uint> device* atom_80 [[buffer(5)]], uint device* work_80 [[buffer(4)]], packed_float4 device* frames_80 [[buffer(6)]], packed_float4 device* bodies_80 [[buffer(7)]], packed_float4 device* chunks_80 [[buffer(1)]], uint device* sort_80 [[buffer(11)]], Record_natural_0 device* manifolds_80 [[buffer(10)]], packed_float4 device* loads_80 [[buffer(12)]], packed_uint4 device* bonds_81 [[buffer(2)]], uint device* args_80 [[buffer(13)]], uint device* info_80 [[buffer(15)]], ScanParams_0 constant* scan_80 [[buffer(14)]], packed_float4 device* parts_80 [[buffer(3)]], packed_float4 device* colliders_80 [[buffer(8)]], packed_uint4 device* collider_info_80 [[buffer(9)]], packed_float4 device* view_chunks_80 [[buffer(17)]], ViewParams_0 constant* view_80 [[buffer(16)]], packed_float4 device* objects_80 [[buffer(18)]], uint device* shot_bodies_80 [[buffer(20)]], float device* lines_80 [[buffer(19)]], packed_float4 device* geometry_80 [[buffer(21)]], packed_float4 device* shots_80 [[buffer(22)]], packed_float4 device* lp_80 [[buffer(24)]], packed_float4 device* ext_82 [[buffer(23)]], packed_float4 device* engine_contacts_80 [[buffer(25)]], CrushParams_0 constant* crush_80 [[buffer(28)]], packed_float4 device* crush_rw_80 [[buffer(27)]], packed_float4 device* crush_tab_80 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_212;
    (&kernelContext_212)->params_0 = params_80;
    (&kernelContext_212)->atom_0 = atom_80;
    (&kernelContext_212)->work_0 = work_80;
    (&kernelContext_212)->frames_0 = frames_80;
    (&kernelContext_212)->bodies_0 = bodies_80;
    (&kernelContext_212)->chunks_0 = chunks_80;
    (&kernelContext_212)->sort_0 = sort_80;
    (&kernelContext_212)->manifolds_0 = manifolds_80;
    (&kernelContext_212)->loads_0 = loads_80;
    (&kernelContext_212)->bonds_0 = bonds_81;
    (&kernelContext_212)->args_0 = args_80;
    (&kernelContext_212)->info_0 = info_80;
    (&kernelContext_212)->scan_0 = scan_80;
    (&kernelContext_212)->parts_0 = parts_80;
    (&kernelContext_212)->colliders_0 = colliders_80;
    (&kernelContext_212)->collider_info_0 = collider_info_80;
    (&kernelContext_212)->view_chunks_0 = view_chunks_80;
    (&kernelContext_212)->view_0 = view_80;
    (&kernelContext_212)->objects_0 = objects_80;
    (&kernelContext_212)->shot_bodies_0 = shot_bodies_80;
    (&kernelContext_212)->lines_0 = lines_80;
    (&kernelContext_212)->geometry_0 = geometry_80;
    (&kernelContext_212)->shots_0 = shots_80;
    (&kernelContext_212)->lp_0 = lp_80;
    (&kernelContext_212)->ext_0 = ext_82;
    (&kernelContext_212)->engine_contacts_0 = engine_contacts_80;
    (&kernelContext_212)->crush_0 = crush_80;
    (&kernelContext_212)->crush_rw_0 = crush_rw_80;
    (&kernelContext_212)->crush_tab_0 = crush_tab_80;
    threadgroup array<uint, int(1024)> g_scan_80;
    (&kernelContext_212)->g_scan_0 = &g_scan_80;
    threadgroup array<float, int(64)> g_il_best_80;
    (&kernelContext_212)->g_il_best_0 = &g_il_best_80;
    uint k_65 = id_74.x;
    bool _S1583;
    if(k_65 >= (params_80->chunk_count_0))
    {
        _S1583 = true;
    }
    else
    {
        _S1583 = ((&kernelContext_212)->params_0->shot_count_0) == 0U;
    }
    if(_S1583)
    {
        _S1583 = true;
    }
    else
    {
        _S1583 = ((&kernelContext_212)->crush_0->enabled_0) == 0U;
    }
    if(_S1583)
    {
        return;
    }
    thread uint b_60;
    thread float4 p1_2;
    thread float4 p2_4;
    bool _S1584 = cover_patch_0(k_65, &b_60, &p1_2, &p2_4, &kernelContext_212);
    if(!_S1584)
    {
        return;
    }
    uint _S1585 = a_cover_0(&kernelContext_212);
    uint _S1586 = _S1585 + 128U * b_60;
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
        uint _S1587 = _S1586 + 2U * s_30;
        uint first_4 = atomic_load_explicit((&kernelContext_212)->atom_0+_S1587, memory_order_relaxed);
        if(first_4 == 4294967295U)
        {
            s_30 = s_30 + 1U;
            continue;
        }
        float _S1588 = cover_depth_0(k_65, s_30, p1_2.xyz, p2_4.xyz, p2_4.w, p1_2.w, &kernelContext_212);
        if(_S1588 < 5.0000000752373311e+29f)
        {
            _S1583 = _S1588 <= (unordered_0(first_4) + 0.00009999999747379f);
        }
        else
        {
            _S1583 = false;
        }
        if(_S1583)
        {
            uint _S1589 = atomic_fetch_min_explicit((&kernelContext_212)->atom_0+(_S1587 + 1U), k_65, memory_order_relaxed);
        }
        s_30 = s_30 + 1U;
    }
    return;
}

float strike_load_0(uint k_66, uint slot_6, float3 f_16, float3 at_24, float tau_8, float budget_0, KernelContext_0 thread* kernelContext_213)
{
    bool _S1590 = crushable_0(k_66, kernelContext_213);
    bool _S1591;
    if(!_S1590)
    {
        _S1591 = true;
    }
    else
    {
        uint _S1592 = crush_flags_0(k_66, kernelContext_213);
        _S1591 = (_S1592 & 1U) != 0U;
    }
    if(_S1591)
    {
        _S1591 = true;
    }
    else
    {
        _S1591 = !(budget_0 > 0.0f);
    }
    if(_S1591)
    {
        return 0.0f;
    }
    thread Quat_0 q_33;
    thread float3 origin_13;
    structure_pose_0(slot_6, 7U * slot_6, &q_33, &origin_13, kernelContext_213);
    float3 _S1593 = (float4(*(kernelContext_213->chunks_0+4U * k_66)) ).xyz;
    thread Quat_0 _S1594 = q_33;
    float3 _S1595 = rotate_0(&_S1594, _S1593);
    float3 centre_5 = origin_13 + _S1595;
    thread Quat_0 _S1596 = q_33;
    float4 _S1597 = bond_centroid_0(k_66, &_S1596, origin_13, centre_5, kernelContext_213);
    if(!((_S1597.w) > 0.0f))
    {
        return 0.0f;
    }
    Sym_0 _S1598 = sym_outer_0(_S1597.xyz, - f_16);
    thread Sym_0 _S1599 = sym_outer_0(at_24 - centre_5, f_16);
    thread Sym_0 _S1600 = _S1598;
    Sym_0 _S1601 = sym_add_0(&_S1599, &_S1600);
    float before_0 = (float4(*(kernelContext_213->crush_rw_0+k_66)) ).x;
    float _S1602 = min(tau_8, kernelContext_213->params_0->dt_0);
    thread Sym_0 _S1603 = _S1601;
    thread Sym_0 _S1604 = _S1601;
    float4 _S1605 = crush_two_0(&_S1603, _S1602, &_S1604, 0.0f, k_66, before_0, kernelContext_213);
    uint _S1606 = 2U * k_66;
    float per_0 = (float4(*(kernelContext_213->crush_tab_0+_S1606)) ).w * (float4(*(kernelContext_213->crush_tab_0+(_S1606 + 1U))) ).y;
    float energy_1 = (_S1605.x - before_0) * per_0;
    float energy_2;
    float4 step_8;
    if(energy_1 > budget_0)
    {
        float d_28 = before_0 + budget_0 / per_0;
        float _S1607 = min(d_28, 1.0f);
        float _S1608 = _S1605.y;
        float _S1609 = _S1605.z;
        if(d_28 >= 1.0f)
        {
            energy_2 = 1.0f;
        }
        else
        {
            energy_2 = 0.0f;
        }
        step_8 = float4(_S1607, _S1608, _S1609, energy_2);
        energy_2 = budget_0;
    }
    else
    {
        step_8 = _S1605;
        energy_2 = energy_1;
    }
    crush_record_0(k_66, step_8, 2U, kernelContext_213);
    return energy_2;
}

bool cell_in_0(uint2 mask_0, uint c_25)
{
    return (((mask_0[c_25 / 32U]) >> (c_25 % 32U)) & 1U) != 0U;
}

float strike_crush_body_0(uint b_61, uint s_31, float budget_1, KernelContext_0 thread* kernelContext_214)
{
    uint total_1;
    bool _S1610;
    uint bill_2 = kernelContext_214->crush_0->r_bill_0 + 4U * b_61;
    uint _S1611 = bill_2 + 1U;
    float4 _S1612 = float4(*(kernelContext_214->crush_rw_0+(bill_2 + 2U))) ;
    float4 _S1613 = float4(*(kernelContext_214->crush_rw_0+(bill_2 + 3U))) ;
    float3 contact_21 = (float4(*(kernelContext_214->crush_rw_0+_S1611)) ).xyz;
    float3 d_29 = _S1612.xyz;
    float _S1614 = _S1612.w;
    float _S1615 = (float4(*(kernelContext_214->shots_0+2U * s_31)) ).y;
    uint _S1616 = a_cover_0(kernelContext_214);
    uint _S1617 = _S1616 + 128U * b_61;
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
        uint _S1618 = _S1617 + 2U * c_26;
        uint _S1619 = atomic_load_explicit(kernelContext_214->atom_0+(_S1618 + 1U), memory_order_relaxed);
        if(_S1619 == 4294967295U)
        {
            c_26 = c_26 + 1U;
            continue;
        }
        thread float rho_2;
        float3 _S1620 = cover_offset_0(c_26, d_29, _S1614, &rho_2);
        uint _S1621 = atomic_load_explicit(kernelContext_214->atom_0+_S1618, memory_order_relaxed);
        first_5 = min(first_5, unordered_0(_S1621) + _S1615 - sqrt(max(_S1615 * _S1615 - rho_2 * rho_2, 0.0f)));
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
        uint _S1622 = _S1617 + 2U * c_26;
        uint _S1623 = atomic_load_explicit(kernelContext_214->atom_0+(_S1622 + 1U), memory_order_relaxed);
        if(_S1623 == 4294967295U)
        {
            c_26 = c_26 + 1U;
            continue;
        }
        thread float rho_3;
        float3 _S1624 = cover_offset_0(c_26, d_29, _S1614, &rho_3);
        uint _S1625 = atomic_load_explicit(kernelContext_214->atom_0+_S1622, memory_order_relaxed);
        if((unordered_0(_S1625) + _S1615 - sqrt(max(_S1615 * _S1615 - rho_3 * rho_3, 0.0f)) - first_5) <= (_S1613.z))
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
    float3 _S1626 = float3(0.0f) ;
    float energy_3;
    float3 point_1;
    if(total_2 == 0U)
    {
        float _S1627 = strike_load_0(uint(_S1613.w) - 1U, b_61, d_29 * float3(_S1613.x) , contact_21, _S1613.y, budget_1, kernelContext_214);
        float3 _S1628 = float3(_S1627)  * contact_21;
        energy_3 = _S1627;
        point_1 = _S1628;
    }
    else
    {
        energy_3 = 0.0f;
        point_1 = _S1626;
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
        uint k_67 = atomic_load_explicit(kernelContext_214->atom_0+(_S1617 + 2U * c_26 + 1U), memory_order_relaxed);
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
                    uint _S1629 = atomic_load_explicit(kernelContext_214->atom_0+(_S1617 + 2U * total_1 + 1U), memory_order_relaxed);
                    _S1610 = _S1629 == k_67;
                }
                else
                {
                    _S1610 = false;
                }
                seen_0 = _S1610;
            }
            total_1 = total_1 + 1U;
        }
        if(seen_0)
        {
            c_26 = c_26 + 1U;
            continue;
        }
        uint e_9 = c_26;
        float3 at_25 = _S1626;
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
                _S1610 = true;
            }
            else
            {
                uint _S1630 = atomic_load_explicit(kernelContext_214->atom_0+(_S1617 + 2U * e_9 + 1U), memory_order_relaxed);
                _S1610 = _S1630 != k_67;
            }
            if(_S1610)
            {
                e_9 = e_9 + 1U;
                continue;
            }
            thread float rho_4;
            float3 _S1631 = contact_21 + cover_offset_0(e_9, d_29, _S1614, &rho_4);
            uint _S1632 = atomic_load_explicit(kernelContext_214->atom_0+(_S1617 + 2U * e_9), memory_order_relaxed);
            uint _S1633 = n_6 + 1U;
            at_25 = at_25 + (_S1631 + d_29 * float3(unordered_0(_S1632)) );
            n_6 = _S1633;
            e_9 = e_9 + 1U;
        }
        float _S1634 = float(n_6);
        float3 at_26 = at_25 / float3(_S1634) ;
        float _S1635 = strike_load_0(k_67, b_61, d_29 * float3((_S1613.x * _S1634 / float(total_2))) , at_26, _S1613.y, budget_1 - energy_3, kernelContext_214);
        float3 point_2 = point_1 + float3(_S1635)  * at_26;
        energy_3 = energy_3 + _S1635;
        point_1 = point_2;
        c_26 = c_26 + 1U;
    }
    *(kernelContext_214->crush_rw_0+bill_2) = packed_float4(float4(energy_3, float(s_31 + 1U), 0.0f, 0.0f)) ;
    if(energy_3 > 0.0f)
    {
        *(kernelContext_214->crush_rw_0+_S1611) = packed_float4(float4(point_1 / float3(energy_3) , 0.0f)) ;
    }
    return energy_3;
}

[[kernel]] void fr_strike_crush(uint3 id_75 [[thread_position_in_grid]], FractureParams_0 constant* params_81 [[buffer(0)]], atomic<uint> device* atom_81 [[buffer(5)]], uint device* work_81 [[buffer(4)]], packed_float4 device* frames_81 [[buffer(6)]], packed_float4 device* bodies_81 [[buffer(7)]], packed_float4 device* chunks_81 [[buffer(1)]], uint device* sort_81 [[buffer(11)]], Record_natural_0 device* manifolds_81 [[buffer(10)]], packed_float4 device* loads_81 [[buffer(12)]], packed_uint4 device* bonds_82 [[buffer(2)]], uint device* args_81 [[buffer(13)]], uint device* info_81 [[buffer(15)]], ScanParams_0 constant* scan_81 [[buffer(14)]], packed_float4 device* parts_81 [[buffer(3)]], packed_float4 device* colliders_81 [[buffer(8)]], packed_uint4 device* collider_info_81 [[buffer(9)]], packed_float4 device* view_chunks_81 [[buffer(17)]], ViewParams_0 constant* view_81 [[buffer(16)]], packed_float4 device* objects_81 [[buffer(18)]], uint device* shot_bodies_81 [[buffer(20)]], float device* lines_81 [[buffer(19)]], packed_float4 device* geometry_81 [[buffer(21)]], packed_float4 device* shots_81 [[buffer(22)]], packed_float4 device* lp_81 [[buffer(24)]], packed_float4 device* ext_83 [[buffer(23)]], packed_float4 device* engine_contacts_81 [[buffer(25)]], CrushParams_0 constant* crush_81 [[buffer(28)]], packed_float4 device* crush_rw_81 [[buffer(27)]], packed_float4 device* crush_tab_81 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_215;
    (&kernelContext_215)->params_0 = params_81;
    (&kernelContext_215)->atom_0 = atom_81;
    (&kernelContext_215)->work_0 = work_81;
    (&kernelContext_215)->frames_0 = frames_81;
    (&kernelContext_215)->bodies_0 = bodies_81;
    (&kernelContext_215)->chunks_0 = chunks_81;
    (&kernelContext_215)->sort_0 = sort_81;
    (&kernelContext_215)->manifolds_0 = manifolds_81;
    (&kernelContext_215)->loads_0 = loads_81;
    (&kernelContext_215)->bonds_0 = bonds_82;
    (&kernelContext_215)->args_0 = args_81;
    (&kernelContext_215)->info_0 = info_81;
    (&kernelContext_215)->scan_0 = scan_81;
    (&kernelContext_215)->parts_0 = parts_81;
    (&kernelContext_215)->colliders_0 = colliders_81;
    (&kernelContext_215)->collider_info_0 = collider_info_81;
    (&kernelContext_215)->view_chunks_0 = view_chunks_81;
    (&kernelContext_215)->view_0 = view_81;
    (&kernelContext_215)->objects_0 = objects_81;
    (&kernelContext_215)->shot_bodies_0 = shot_bodies_81;
    (&kernelContext_215)->lines_0 = lines_81;
    (&kernelContext_215)->geometry_0 = geometry_81;
    (&kernelContext_215)->shots_0 = shots_81;
    (&kernelContext_215)->lp_0 = lp_81;
    (&kernelContext_215)->ext_0 = ext_83;
    (&kernelContext_215)->engine_contacts_0 = engine_contacts_81;
    (&kernelContext_215)->crush_0 = crush_81;
    (&kernelContext_215)->crush_rw_0 = crush_rw_81;
    (&kernelContext_215)->crush_tab_0 = crush_tab_81;
    threadgroup array<uint, int(1024)> g_scan_81;
    (&kernelContext_215)->g_scan_0 = &g_scan_81;
    threadgroup array<float, int(64)> g_il_best_81;
    (&kernelContext_215)->g_il_best_0 = &g_il_best_81;
    uint s_32 = id_75.x;
    bool _S1636;
    if(s_32 >= (params_81->shot_count_0))
    {
        _S1636 = true;
    }
    else
    {
        _S1636 = ((&kernelContext_215)->crush_0->enabled_0) == 0U;
    }
    if(_S1636)
    {
        return;
    }
    uint sb_4 = 7U * (as_type<uint>(((float4(*((&kernelContext_215)->shots_0+2U * s_32)) ).x)));
    float inv_mass_11 = (float4(*((&kernelContext_215)->bodies_0+sb_4)) ).w;
    float3 v_17 = (float4(*((&kernelContext_215)->bodies_0+(sb_4 + 3U))) ).xyz;
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
        if(b_62 < ((&kernelContext_215)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        thread uint shot_12;
        bool _S1637 = struck_0(b_62, &shot_12, &kernelContext_215);
        if(!_S1637)
        {
            _S1636 = true;
        }
        else
        {
            _S1636 = shot_12 != s_32;
        }
        bool _S1638;
        if(_S1636)
        {
            _S1638 = true;
        }
        else
        {
            _S1638 = ((float4(*((&kernelContext_215)->crush_rw_0+((&kernelContext_215)->crush_0->r_bill_0 + 4U * b_62 + 3U))) ).w) == 0.0f;
        }
        if(_S1638)
        {
            b_62 = b_62 + 1U;
            continue;
        }
        float _S1639 = strike_crush_body_0(b_62, s_32, budget_2, &kernelContext_215);
        budget_2 = budget_2 - _S1639;
        b_62 = b_62 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bill(uint3 id_76 [[thread_position_in_grid]], FractureParams_0 constant* params_82 [[buffer(0)]], atomic<uint> device* atom_82 [[buffer(5)]], uint device* work_82 [[buffer(4)]], packed_float4 device* frames_82 [[buffer(6)]], packed_float4 device* bodies_82 [[buffer(7)]], packed_float4 device* chunks_82 [[buffer(1)]], uint device* sort_82 [[buffer(11)]], Record_natural_0 device* manifolds_82 [[buffer(10)]], packed_float4 device* loads_82 [[buffer(12)]], packed_uint4 device* bonds_83 [[buffer(2)]], uint device* args_82 [[buffer(13)]], uint device* info_82 [[buffer(15)]], ScanParams_0 constant* scan_82 [[buffer(14)]], packed_float4 device* parts_82 [[buffer(3)]], packed_float4 device* colliders_82 [[buffer(8)]], packed_uint4 device* collider_info_82 [[buffer(9)]], packed_float4 device* view_chunks_82 [[buffer(17)]], ViewParams_0 constant* view_82 [[buffer(16)]], packed_float4 device* objects_82 [[buffer(18)]], uint device* shot_bodies_82 [[buffer(20)]], float device* lines_82 [[buffer(19)]], packed_float4 device* geometry_82 [[buffer(21)]], packed_float4 device* shots_82 [[buffer(22)]], packed_float4 device* lp_82 [[buffer(24)]], packed_float4 device* ext_84 [[buffer(23)]], packed_float4 device* engine_contacts_82 [[buffer(25)]], CrushParams_0 constant* crush_82 [[buffer(28)]], packed_float4 device* crush_rw_82 [[buffer(27)]], packed_float4 device* crush_tab_82 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_216;
    (&kernelContext_216)->params_0 = params_82;
    (&kernelContext_216)->atom_0 = atom_82;
    (&kernelContext_216)->work_0 = work_82;
    (&kernelContext_216)->frames_0 = frames_82;
    (&kernelContext_216)->bodies_0 = bodies_82;
    (&kernelContext_216)->chunks_0 = chunks_82;
    (&kernelContext_216)->sort_0 = sort_82;
    (&kernelContext_216)->manifolds_0 = manifolds_82;
    (&kernelContext_216)->loads_0 = loads_82;
    (&kernelContext_216)->bonds_0 = bonds_83;
    (&kernelContext_216)->args_0 = args_82;
    (&kernelContext_216)->info_0 = info_82;
    (&kernelContext_216)->scan_0 = scan_82;
    (&kernelContext_216)->parts_0 = parts_82;
    (&kernelContext_216)->colliders_0 = colliders_82;
    (&kernelContext_216)->collider_info_0 = collider_info_82;
    (&kernelContext_216)->view_chunks_0 = view_chunks_82;
    (&kernelContext_216)->view_0 = view_82;
    (&kernelContext_216)->objects_0 = objects_82;
    (&kernelContext_216)->shot_bodies_0 = shot_bodies_82;
    (&kernelContext_216)->lines_0 = lines_82;
    (&kernelContext_216)->geometry_0 = geometry_82;
    (&kernelContext_216)->shots_0 = shots_82;
    (&kernelContext_216)->lp_0 = lp_82;
    (&kernelContext_216)->ext_0 = ext_84;
    (&kernelContext_216)->engine_contacts_0 = engine_contacts_82;
    (&kernelContext_216)->crush_0 = crush_82;
    (&kernelContext_216)->crush_rw_0 = crush_rw_82;
    (&kernelContext_216)->crush_tab_0 = crush_tab_82;
    threadgroup array<uint, int(1024)> g_scan_82;
    (&kernelContext_216)->g_scan_0 = &g_scan_82;
    threadgroup array<float, int(64)> g_il_best_82;
    (&kernelContext_216)->g_il_best_0 = &g_il_best_82;
    uint s_33 = id_76.x;
    bool _S1640;
    if(s_33 >= (params_82->shot_count_0))
    {
        _S1640 = true;
    }
    else
    {
        _S1640 = ((&kernelContext_216)->crush_0->enabled_0) == 0U;
    }
    if(_S1640)
    {
        return;
    }
    uint b_63 = 0U;
    float energy_4 = 0.0f;
    for(;;)
    {
        if(b_63 < ((&kernelContext_216)->params_0->body_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S1641 = float4(*((&kernelContext_216)->crush_rw_0+((&kernelContext_216)->crush_0->r_bill_0 + 4U * b_63))) ;
        if((_S1641.y) == float(s_33 + 1U))
        {
            energy_4 = energy_4 + _S1641.x;
        }
        b_63 = b_63 + 1U;
    }
    *((&kernelContext_216)->crush_rw_0+((&kernelContext_216)->crush_0->r_shot_0 + s_33)) = packed_float4(float4(0.0f) ) ;
    if(!(energy_4 > 0.0f))
    {
        return;
    }
    uint sb_5 = 7U * (as_type<uint>(((float4(*((&kernelContext_216)->shots_0+2U * s_33)) ).x)));
    float inv_mass_12 = (float4(*((&kernelContext_216)->bodies_0+sb_5)) ).w;
    if(!(inv_mass_12 > 0.0f))
    {
        return;
    }
    float m_9 = 1.0f / inv_mass_12;
    float3 v_18 = (float4(*((&kernelContext_216)->bodies_0+(sb_5 + 3U))) ).xyz;
    float speed_16 = length(v_18);
    float j_15 = m_9 * (speed_16 - sqrt(max(speed_16 * speed_16 - 2.0f * energy_4 / m_9, 0.0f)));
    bool _S1642 = speed_16 > 0.0f;
    float3 _S1643;
    if(_S1642)
    {
        _S1643 = - v_18 * float3((j_15 / (m_9 * speed_16))) ;
    }
    else
    {
        _S1643 = float3(0.0f) ;
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
        uint w_11 = sb_5 + region_6 * 7U * (&kernelContext_216)->params_0->max_bodies_0 + 3U;
        *((&kernelContext_216)->bodies_0+w_11) = packed_float4(float4((float4(*((&kernelContext_216)->bodies_0+w_11)) ).xyz + _S1643, (float4(*((&kernelContext_216)->bodies_0+w_11)) ).w)) ;
        region_6 = region_6 + 1U;
    }
    packed_float4 device* _S1644 = (&kernelContext_216)->crush_rw_0+((&kernelContext_216)->crush_0->r_shot_0 + s_33);
    if(_S1642)
    {
        _S1643 = v_18 * float3((j_15 / speed_16)) ;
    }
    else
    {
        _S1643 = float3(0.0f) ;
    }
    *_S1644 = packed_float4(float4(_S1643, energy_4)) ;
    return;
}

float3 inverse_rotate_0(const Quat_0 thread* q_34, float3 v_19)
{
    thread Quat_0 c_27;
    (&c_27)->w_0 = q_34->w_0;
    (&c_27)->x_0 = - q_34->x_0;
    (&c_27)->y_0 = - q_34->y_0;
    (&c_27)->z_0 = - q_34->z_0;
    thread Quat_0 _S1645 = c_27;
    float3 _S1646 = rotate_0(&_S1645, v_19);
    return _S1646;
}

[[kernel]] void fr_crush_push(uint3 id_77 [[thread_position_in_grid]], FractureParams_0 constant* params_83 [[buffer(0)]], atomic<uint> device* atom_83 [[buffer(5)]], uint device* work_83 [[buffer(4)]], packed_float4 device* frames_83 [[buffer(6)]], packed_float4 device* bodies_83 [[buffer(7)]], packed_float4 device* chunks_83 [[buffer(1)]], uint device* sort_83 [[buffer(11)]], Record_natural_0 device* manifolds_83 [[buffer(10)]], packed_float4 device* loads_83 [[buffer(12)]], packed_uint4 device* bonds_84 [[buffer(2)]], uint device* args_83 [[buffer(13)]], uint device* info_83 [[buffer(15)]], ScanParams_0 constant* scan_83 [[buffer(14)]], packed_float4 device* parts_83 [[buffer(3)]], packed_float4 device* colliders_83 [[buffer(8)]], packed_uint4 device* collider_info_83 [[buffer(9)]], packed_float4 device* view_chunks_83 [[buffer(17)]], ViewParams_0 constant* view_83 [[buffer(16)]], packed_float4 device* objects_83 [[buffer(18)]], uint device* shot_bodies_83 [[buffer(20)]], float device* lines_83 [[buffer(19)]], packed_float4 device* geometry_83 [[buffer(21)]], packed_float4 device* shots_83 [[buffer(22)]], packed_float4 device* lp_83 [[buffer(24)]], packed_float4 device* ext_85 [[buffer(23)]], packed_float4 device* engine_contacts_83 [[buffer(25)]], CrushParams_0 constant* crush_83 [[buffer(28)]], packed_float4 device* crush_rw_83 [[buffer(27)]], packed_float4 device* crush_tab_83 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_217;
    (&kernelContext_217)->params_0 = params_83;
    (&kernelContext_217)->atom_0 = atom_83;
    (&kernelContext_217)->work_0 = work_83;
    (&kernelContext_217)->frames_0 = frames_83;
    (&kernelContext_217)->bodies_0 = bodies_83;
    (&kernelContext_217)->chunks_0 = chunks_83;
    (&kernelContext_217)->sort_0 = sort_83;
    (&kernelContext_217)->manifolds_0 = manifolds_83;
    (&kernelContext_217)->loads_0 = loads_83;
    (&kernelContext_217)->bonds_0 = bonds_84;
    (&kernelContext_217)->args_0 = args_83;
    (&kernelContext_217)->info_0 = info_83;
    (&kernelContext_217)->scan_0 = scan_83;
    (&kernelContext_217)->parts_0 = parts_83;
    (&kernelContext_217)->colliders_0 = colliders_83;
    (&kernelContext_217)->collider_info_0 = collider_info_83;
    (&kernelContext_217)->view_chunks_0 = view_chunks_83;
    (&kernelContext_217)->view_0 = view_83;
    (&kernelContext_217)->objects_0 = objects_83;
    (&kernelContext_217)->shot_bodies_0 = shot_bodies_83;
    (&kernelContext_217)->lines_0 = lines_83;
    (&kernelContext_217)->geometry_0 = geometry_83;
    (&kernelContext_217)->shots_0 = shots_83;
    (&kernelContext_217)->lp_0 = lp_83;
    (&kernelContext_217)->ext_0 = ext_85;
    (&kernelContext_217)->engine_contacts_0 = engine_contacts_83;
    (&kernelContext_217)->crush_0 = crush_83;
    (&kernelContext_217)->crush_rw_0 = crush_rw_83;
    (&kernelContext_217)->crush_tab_0 = crush_tab_83;
    threadgroup array<uint, int(1024)> g_scan_83;
    (&kernelContext_217)->g_scan_0 = &g_scan_83;
    threadgroup array<float, int(64)> g_il_best_83;
    (&kernelContext_217)->g_il_best_0 = &g_il_best_83;
    uint b_64 = id_77.x;
    bool _S1647;
    if(b_64 >= (params_83->body_count_0))
    {
        _S1647 = true;
    }
    else
    {
        _S1647 = ((&kernelContext_217)->params_0->shot_count_0) == 0U;
    }
    if(_S1647)
    {
        _S1647 = true;
    }
    else
    {
        _S1647 = ((&kernelContext_217)->crush_0->enabled_0) == 0U;
    }
    if(_S1647)
    {
        return;
    }
    uint _S1648 = 4U * b_64;
    float4 _S1649 = float4(*((&kernelContext_217)->crush_rw_0+((&kernelContext_217)->crush_0->r_bill_0 + _S1648))) ;
    float _S1650 = _S1649.x;
    if(!(_S1650 > 0.0f))
    {
        return;
    }
    float4 _S1651 = float4(*((&kernelContext_217)->crush_rw_0+((&kernelContext_217)->crush_0->r_shot_0 + (uint(_S1649.y) - 1U)))) ;
    uint bw_0 = 7U * b_64;
    float inv_mass_13 = (float4(*((&kernelContext_217)->bodies_0+bw_0)) ).w;
    float _S1652 = _S1651.w;
    if(!(_S1652 > 0.0f))
    {
        _S1647 = true;
    }
    else
    {
        _S1647 = inv_mass_13 == 0.0f;
    }
    if(_S1647)
    {
        return;
    }
    float3 j_16 = _S1651.xyz * float3((_S1650 / _S1652)) ;
    Quat_0 o_11 = quat_of_0(float4(*((&kernelContext_217)->bodies_0+(bw_0 + 1U))) );
    float3 inv_i_0 = (float4(*((&kernelContext_217)->bodies_0+(bw_0 + 2U))) ).xyz;
    float3 _S1653 = cross(to_rubble_0((float4(*((&kernelContext_217)->crush_rw_0+((&kernelContext_217)->crush_0->r_bill_0 + _S1648 + 1U))) ).xyz) - (float4(*((&kernelContext_217)->bodies_0+bw_0)) ).xyz, j_16);
    thread Quat_0 _S1654 = o_11;
    float3 _S1655 = inverse_rotate_0(&_S1654, _S1653);
    float3 _S1656 = inv_i_0 * _S1655;
    thread Quat_0 _S1657 = o_11;
    float3 _S1658 = rotate_0(&_S1657, _S1656);
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
        uint w_12 = bw_0 + region_7 * 7U * (&kernelContext_217)->params_0->max_bodies_0;
        uint _S1659 = w_12 + 3U;
        *((&kernelContext_217)->bodies_0+_S1659) = packed_float4(float4((float4(*((&kernelContext_217)->bodies_0+_S1659)) ).xyz + j_16 * float3(inv_mass_13) , (float4(*((&kernelContext_217)->bodies_0+_S1659)) ).w)) ;
        uint _S1660 = w_12 + 4U;
        *((&kernelContext_217)->bodies_0+_S1660) = packed_float4(float4((float4(*((&kernelContext_217)->bodies_0+_S1660)) ).xyz + _S1658, (float4(*((&kernelContext_217)->bodies_0+_S1660)) ).w)) ;
        region_7 = region_7 + 1U;
    }
    return;
}

[[kernel]] void fr_crush_bonds(uint3 id_78 [[thread_position_in_grid]], FractureParams_0 constant* params_84 [[buffer(0)]], atomic<uint> device* atom_84 [[buffer(5)]], uint device* work_84 [[buffer(4)]], packed_float4 device* frames_84 [[buffer(6)]], packed_float4 device* bodies_84 [[buffer(7)]], packed_float4 device* chunks_84 [[buffer(1)]], uint device* sort_84 [[buffer(11)]], Record_natural_0 device* manifolds_84 [[buffer(10)]], packed_float4 device* loads_84 [[buffer(12)]], packed_uint4 device* bonds_85 [[buffer(2)]], uint device* args_84 [[buffer(13)]], uint device* info_84 [[buffer(15)]], ScanParams_0 constant* scan_84 [[buffer(14)]], packed_float4 device* parts_84 [[buffer(3)]], packed_float4 device* colliders_84 [[buffer(8)]], packed_uint4 device* collider_info_84 [[buffer(9)]], packed_float4 device* view_chunks_84 [[buffer(17)]], ViewParams_0 constant* view_84 [[buffer(16)]], packed_float4 device* objects_84 [[buffer(18)]], uint device* shot_bodies_84 [[buffer(20)]], float device* lines_84 [[buffer(19)]], packed_float4 device* geometry_84 [[buffer(21)]], packed_float4 device* shots_84 [[buffer(22)]], packed_float4 device* lp_84 [[buffer(24)]], packed_float4 device* ext_86 [[buffer(23)]], packed_float4 device* engine_contacts_84 [[buffer(25)]], CrushParams_0 constant* crush_84 [[buffer(28)]], packed_float4 device* crush_rw_84 [[buffer(27)]], packed_float4 device* crush_tab_84 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_218;
    (&kernelContext_218)->params_0 = params_84;
    (&kernelContext_218)->atom_0 = atom_84;
    (&kernelContext_218)->work_0 = work_84;
    (&kernelContext_218)->frames_0 = frames_84;
    (&kernelContext_218)->bodies_0 = bodies_84;
    (&kernelContext_218)->chunks_0 = chunks_84;
    (&kernelContext_218)->sort_0 = sort_84;
    (&kernelContext_218)->manifolds_0 = manifolds_84;
    (&kernelContext_218)->loads_0 = loads_84;
    (&kernelContext_218)->bonds_0 = bonds_85;
    (&kernelContext_218)->args_0 = args_84;
    (&kernelContext_218)->info_0 = info_84;
    (&kernelContext_218)->scan_0 = scan_84;
    (&kernelContext_218)->parts_0 = parts_84;
    (&kernelContext_218)->colliders_0 = colliders_84;
    (&kernelContext_218)->collider_info_0 = collider_info_84;
    (&kernelContext_218)->view_chunks_0 = view_chunks_84;
    (&kernelContext_218)->view_0 = view_84;
    (&kernelContext_218)->objects_0 = objects_84;
    (&kernelContext_218)->shot_bodies_0 = shot_bodies_84;
    (&kernelContext_218)->lines_0 = lines_84;
    (&kernelContext_218)->geometry_0 = geometry_84;
    (&kernelContext_218)->shots_0 = shots_84;
    (&kernelContext_218)->lp_0 = lp_84;
    (&kernelContext_218)->ext_0 = ext_86;
    (&kernelContext_218)->engine_contacts_0 = engine_contacts_84;
    (&kernelContext_218)->crush_0 = crush_84;
    (&kernelContext_218)->crush_rw_0 = crush_rw_84;
    (&kernelContext_218)->crush_tab_0 = crush_tab_84;
    threadgroup array<uint, int(1024)> g_scan_84;
    (&kernelContext_218)->g_scan_0 = &g_scan_84;
    threadgroup array<float, int(64)> g_il_best_84;
    (&kernelContext_218)->g_il_best_0 = &g_il_best_84;
    uint i_24 = id_78.x;
    bool _S1661;
    if(i_24 >= (params_84->bond_count_0))
    {
        _S1661 = true;
    }
    else
    {
        _S1661 = ((&kernelContext_218)->crush_0->enabled_0) == 0U;
    }
    if(_S1661)
    {
        _S1661 = true;
    }
    else
    {
        uint _S1662 = w_alive_0(&kernelContext_218);
        _S1661 = (*((&kernelContext_218)->work_0+(_S1662 + i_24))) == 0U;
    }
    if(_S1661)
    {
        return;
    }
    uint4 _S1663 = uint4(*((&kernelContext_218)->bonds_0+i_24)) ;
    uint _S1664 = _S1663.x;
    if((((as_type<uint>(((float4(*((&kernelContext_218)->crush_rw_0+_S1664)) ).w))) | (as_type<uint>(((float4(*((&kernelContext_218)->crush_rw_0+_S1663.y)) ).w)))) & 2U) == 0U)
    {
        return;
    }
    uint _S1665 = w_alive_0(&kernelContext_218);
    *((&kernelContext_218)->work_0+(_S1665 + i_24)) = 0U;
    uint _S1666 = a_dirty_0(&kernelContext_218);
    uint _S1667 = body_of_0(_S1664, &kernelContext_218);
    uint _S1668 = atomic_fetch_or_explicit((&kernelContext_218)->atom_0+(_S1666 + _S1667), 1U, memory_order_relaxed);
    uint _S1669 = a_counters_1(&kernelContext_218);
    uint _S1670 = atomic_fetch_add_explicit((&kernelContext_218)->atom_0+_S1669, 1U, memory_order_relaxed);
    uint _S1671 = atomic_fetch_add_explicit((&kernelContext_218)->atom_0+(_S1669 + 1U), 1U, memory_order_relaxed);
    return;
}

uint crush_pieces_0(uint k_68, KernelContext_0 thread* kernelContext_219)
{
    if((kernelContext_219->crush_0->mode_1) == 1U)
    {
        return kernelContext_219->crush_0->fragments_0;
    }
    if((kernelContext_219->crush_0->mode_1) == 2U)
    {
        return uint((float4(*(kernelContext_219->crush_tab_0+(2U * k_68 + 1U))) ).w + 0.5f);
    }
    return 0U;
}

[[kernel]] void fr_crush_mark(uint3 id_79 [[thread_position_in_grid]], FractureParams_0 constant* params_85 [[buffer(0)]], atomic<uint> device* atom_85 [[buffer(5)]], uint device* work_85 [[buffer(4)]], packed_float4 device* frames_85 [[buffer(6)]], packed_float4 device* bodies_85 [[buffer(7)]], packed_float4 device* chunks_85 [[buffer(1)]], uint device* sort_85 [[buffer(11)]], Record_natural_0 device* manifolds_85 [[buffer(10)]], packed_float4 device* loads_85 [[buffer(12)]], packed_uint4 device* bonds_86 [[buffer(2)]], uint device* args_85 [[buffer(13)]], uint device* info_85 [[buffer(15)]], ScanParams_0 constant* scan_85 [[buffer(14)]], packed_float4 device* parts_85 [[buffer(3)]], packed_float4 device* colliders_85 [[buffer(8)]], packed_uint4 device* collider_info_85 [[buffer(9)]], packed_float4 device* view_chunks_85 [[buffer(17)]], ViewParams_0 constant* view_85 [[buffer(16)]], packed_float4 device* objects_85 [[buffer(18)]], uint device* shot_bodies_85 [[buffer(20)]], float device* lines_85 [[buffer(19)]], packed_float4 device* geometry_85 [[buffer(21)]], packed_float4 device* shots_85 [[buffer(22)]], packed_float4 device* lp_85 [[buffer(24)]], packed_float4 device* ext_87 [[buffer(23)]], packed_float4 device* engine_contacts_85 [[buffer(25)]], CrushParams_0 constant* crush_85 [[buffer(28)]], packed_float4 device* crush_rw_85 [[buffer(27)]], packed_float4 device* crush_tab_85 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_220;
    (&kernelContext_220)->params_0 = params_85;
    (&kernelContext_220)->atom_0 = atom_85;
    (&kernelContext_220)->work_0 = work_85;
    (&kernelContext_220)->frames_0 = frames_85;
    (&kernelContext_220)->bodies_0 = bodies_85;
    (&kernelContext_220)->chunks_0 = chunks_85;
    (&kernelContext_220)->sort_0 = sort_85;
    (&kernelContext_220)->manifolds_0 = manifolds_85;
    (&kernelContext_220)->loads_0 = loads_85;
    (&kernelContext_220)->bonds_0 = bonds_86;
    (&kernelContext_220)->args_0 = args_85;
    (&kernelContext_220)->info_0 = info_85;
    (&kernelContext_220)->scan_0 = scan_85;
    (&kernelContext_220)->parts_0 = parts_85;
    (&kernelContext_220)->colliders_0 = colliders_85;
    (&kernelContext_220)->collider_info_0 = collider_info_85;
    (&kernelContext_220)->view_chunks_0 = view_chunks_85;
    (&kernelContext_220)->view_0 = view_85;
    (&kernelContext_220)->objects_0 = objects_85;
    (&kernelContext_220)->shot_bodies_0 = shot_bodies_85;
    (&kernelContext_220)->lines_0 = lines_85;
    (&kernelContext_220)->geometry_0 = geometry_85;
    (&kernelContext_220)->shots_0 = shots_85;
    (&kernelContext_220)->lp_0 = lp_85;
    (&kernelContext_220)->ext_0 = ext_87;
    (&kernelContext_220)->engine_contacts_0 = engine_contacts_85;
    (&kernelContext_220)->crush_0 = crush_85;
    (&kernelContext_220)->crush_rw_0 = crush_rw_85;
    (&kernelContext_220)->crush_tab_0 = crush_tab_85;
    threadgroup array<uint, int(1024)> g_scan_85;
    (&kernelContext_220)->g_scan_0 = &g_scan_85;
    threadgroup array<float, int(64)> g_il_best_85;
    (&kernelContext_220)->g_il_best_0 = &g_il_best_85;
    uint c_28 = id_79.x;
    bool _S1672;
    if(c_28 >= (params_85->chunk_count_0))
    {
        _S1672 = true;
    }
    else
    {
        _S1672 = ((&kernelContext_220)->crush_0->enabled_0) == 0U;
    }
    if(_S1672)
    {
        return;
    }
    uint _S1673 = crush_flags_0(c_28, &kernelContext_220);
    uint now_0;
    if((_S1673 & 2U) != 0U)
    {
        now_0 = 1U;
    }
    else
    {
        now_0 = 0U;
    }
    uint _S1674 = w_new_0(&kernelContext_220);
    uint device* _S1675 = (&kernelContext_220)->work_0+(_S1674 + c_28);
    bool _S1676 = now_0 != 0U;
    if(_S1676)
    {
        uint _S1677 = crush_pieces_0(c_28, &kernelContext_220);
        now_0 = 1048576U | _S1677;
    }
    else
    {
        now_0 = 0U;
    }
    *_S1675 = now_0;
    if(_S1676)
    {
        uint _S1678 = a_counters_1(&kernelContext_220);
        uint _S1679 = atomic_fetch_add_explicit((&kernelContext_220)->atom_0+(_S1678 + 10U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_crush_gate(uint3 id_80 [[thread_position_in_grid]], FractureParams_0 constant* params_86 [[buffer(0)]], atomic<uint> device* atom_86 [[buffer(5)]], uint device* work_86 [[buffer(4)]], packed_float4 device* frames_86 [[buffer(6)]], packed_float4 device* bodies_86 [[buffer(7)]], packed_float4 device* chunks_86 [[buffer(1)]], uint device* sort_86 [[buffer(11)]], Record_natural_0 device* manifolds_86 [[buffer(10)]], packed_float4 device* loads_86 [[buffer(12)]], packed_uint4 device* bonds_87 [[buffer(2)]], uint device* args_86 [[buffer(13)]], uint device* info_86 [[buffer(15)]], ScanParams_0 constant* scan_86 [[buffer(14)]], packed_float4 device* parts_86 [[buffer(3)]], packed_float4 device* colliders_86 [[buffer(8)]], packed_uint4 device* collider_info_86 [[buffer(9)]], packed_float4 device* view_chunks_86 [[buffer(17)]], ViewParams_0 constant* view_86 [[buffer(16)]], packed_float4 device* objects_86 [[buffer(18)]], uint device* shot_bodies_86 [[buffer(20)]], float device* lines_86 [[buffer(19)]], packed_float4 device* geometry_86 [[buffer(21)]], packed_float4 device* shots_86 [[buffer(22)]], packed_float4 device* lp_86 [[buffer(24)]], packed_float4 device* ext_88 [[buffer(23)]], packed_float4 device* engine_contacts_86 [[buffer(25)]], CrushParams_0 constant* crush_86 [[buffer(28)]], packed_float4 device* crush_rw_86 [[buffer(27)]], packed_float4 device* crush_tab_86 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_221;
    (&kernelContext_221)->params_0 = params_86;
    (&kernelContext_221)->atom_0 = atom_86;
    (&kernelContext_221)->work_0 = work_86;
    (&kernelContext_221)->frames_0 = frames_86;
    (&kernelContext_221)->bodies_0 = bodies_86;
    (&kernelContext_221)->chunks_0 = chunks_86;
    (&kernelContext_221)->sort_0 = sort_86;
    (&kernelContext_221)->manifolds_0 = manifolds_86;
    (&kernelContext_221)->loads_0 = loads_86;
    (&kernelContext_221)->bonds_0 = bonds_87;
    (&kernelContext_221)->args_0 = args_86;
    (&kernelContext_221)->info_0 = info_86;
    (&kernelContext_221)->scan_0 = scan_86;
    (&kernelContext_221)->parts_0 = parts_86;
    (&kernelContext_221)->colliders_0 = colliders_86;
    (&kernelContext_221)->collider_info_0 = collider_info_86;
    (&kernelContext_221)->view_chunks_0 = view_chunks_86;
    (&kernelContext_221)->view_0 = view_86;
    (&kernelContext_221)->objects_0 = objects_86;
    (&kernelContext_221)->shot_bodies_0 = shot_bodies_86;
    (&kernelContext_221)->lines_0 = lines_86;
    (&kernelContext_221)->geometry_0 = geometry_86;
    (&kernelContext_221)->shots_0 = shots_86;
    (&kernelContext_221)->lp_0 = lp_86;
    (&kernelContext_221)->ext_0 = ext_88;
    (&kernelContext_221)->engine_contacts_0 = engine_contacts_86;
    (&kernelContext_221)->crush_0 = crush_86;
    (&kernelContext_221)->crush_rw_0 = crush_rw_86;
    (&kernelContext_221)->crush_tab_0 = crush_tab_86;
    threadgroup array<uint, int(1024)> g_scan_86;
    (&kernelContext_221)->g_scan_0 = &g_scan_86;
    threadgroup array<float, int(64)> g_il_best_86;
    (&kernelContext_221)->g_il_best_0 = &g_il_best_86;
    bool run_2;
    if((crush_86->enabled_0) != 0U)
    {
        uint _S1680 = a_counters_1(&kernelContext_221);
        uint _S1681 = atomic_load_explicit((&kernelContext_221)->atom_0+(_S1680 + 10U), memory_order_relaxed);
        run_2 = _S1681 != 0U;
    }
    else
    {
        run_2 = false;
    }
    uint _S1682 = a_counters_1(&kernelContext_221);
    atomic_store_explicit((&kernelContext_221)->atom_0+(_S1682 + 10U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_221)->atom_0+(_S1682 + 7U), 0U, memory_order_relaxed);
    uint _S1683;
    if(run_2)
    {
        uint _S1684 = groups_of_0((&kernelContext_221)->params_0->chunk_count_0, 64U);
        _S1683 = _S1684;
    }
    else
    {
        _S1683 = 0U;
    }
    set_args_0(14U, _S1683, &kernelContext_221);
    if(run_2)
    {
        uint _S1685 = groups_of_0((&kernelContext_221)->params_0->chunk_count_0, 1024U);
        _S1683 = _S1685;
    }
    else
    {
        _S1683 = 0U;
    }
    set_args_0(15U, _S1683, &kernelContext_221);
    int _S1686;
    if(run_2)
    {
        _S1686 = int(1);
    }
    else
    {
        _S1686 = int(0);
    }
    set_args_0(16U, uint(_S1686), &kernelContext_221);
    return;
}

void write_body_0(uint slot_7, float3 pos_1, float inv_mass_14, const Quat_0 thread* orientation_0, float3 inv_inertia_1, uint flags_3, float3 v_20, float3 w_13, KernelContext_0 thread* kernelContext_222)
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
        uint b_65 = 7U * (slot_7 + region_8 * kernelContext_222->params_0->max_bodies_0);
        *(kernelContext_222->bodies_0+b_65) = packed_float4(float4(pos_1, inv_mass_14)) ;
        packed_float4 device* _S1687 = kernelContext_222->bodies_0+(b_65 + 1U);
        float4 _S1688 = quat_vec_0(orientation_0);
        *_S1687 = packed_float4(_S1688) ;
        *(kernelContext_222->bodies_0+(b_65 + 2U)) = packed_float4(float4(inv_inertia_1, (as_type<float>((flags_3))))) ;
        *(kernelContext_222->bodies_0+(b_65 + 3U)) = packed_float4(float4(v_20, 0.0f)) ;
        *(kernelContext_222->bodies_0+(b_65 + 4U)) = packed_float4(float4(w_13, (as_type<float>((kernelContext_222->params_0->step_0))))) ;
        *(kernelContext_222->bodies_0+(b_65 + 5U)) = packed_float4(float4(0.0f) ) ;
        *(kernelContext_222->bodies_0+(b_65 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
        region_8 = region_8 + 1U;
    }
    return;
}

float3 crush_particle_spot_0(uint k_69, uint i_25)
{
    uint h_7 = (k_69 * 2654435761U) ^ (i_25 * 2246822519U);
    uint h_8 = (h_7 ^ (h_7 >> 15U)) * 739982445U;
    uint h_9 = (h_8 ^ (h_8 >> 12U)) * 695872825U;
    uint h_10 = h_9 ^ (h_9 >> 15U);
    return float3(float(h_10 & 1023U), float((h_10 >> 10U) & 1023U), float((h_10 * 2654435761U + 2135587861U) & 1023U)) / float3(511.5f)  - float3(1.0f) ;
}

[[kernel]] void fr_crush_apply(uint3 id_81 [[thread_position_in_grid]], FractureParams_0 constant* params_87 [[buffer(0)]], atomic<uint> device* atom_87 [[buffer(5)]], uint device* work_87 [[buffer(4)]], packed_float4 device* frames_87 [[buffer(6)]], packed_float4 device* bodies_87 [[buffer(7)]], packed_float4 device* chunks_87 [[buffer(1)]], uint device* sort_87 [[buffer(11)]], Record_natural_0 device* manifolds_87 [[buffer(10)]], packed_float4 device* loads_87 [[buffer(12)]], packed_uint4 device* bonds_88 [[buffer(2)]], uint device* args_87 [[buffer(13)]], uint device* info_87 [[buffer(15)]], ScanParams_0 constant* scan_87 [[buffer(14)]], packed_float4 device* parts_87 [[buffer(3)]], packed_float4 device* colliders_87 [[buffer(8)]], packed_uint4 device* collider_info_87 [[buffer(9)]], packed_float4 device* view_chunks_87 [[buffer(17)]], ViewParams_0 constant* view_87 [[buffer(16)]], packed_float4 device* objects_87 [[buffer(18)]], uint device* shot_bodies_87 [[buffer(20)]], float device* lines_87 [[buffer(19)]], packed_float4 device* geometry_87 [[buffer(21)]], packed_float4 device* shots_87 [[buffer(22)]], packed_float4 device* lp_87 [[buffer(24)]], packed_float4 device* ext_89 [[buffer(23)]], packed_float4 device* engine_contacts_87 [[buffer(25)]], CrushParams_0 constant* crush_87 [[buffer(28)]], packed_float4 device* crush_rw_87 [[buffer(27)]], packed_float4 device* crush_tab_87 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_223;
    (&kernelContext_223)->params_0 = params_87;
    (&kernelContext_223)->atom_0 = atom_87;
    (&kernelContext_223)->work_0 = work_87;
    (&kernelContext_223)->frames_0 = frames_87;
    (&kernelContext_223)->bodies_0 = bodies_87;
    (&kernelContext_223)->chunks_0 = chunks_87;
    (&kernelContext_223)->sort_0 = sort_87;
    (&kernelContext_223)->manifolds_0 = manifolds_87;
    (&kernelContext_223)->loads_0 = loads_87;
    (&kernelContext_223)->bonds_0 = bonds_88;
    (&kernelContext_223)->args_0 = args_87;
    (&kernelContext_223)->info_0 = info_87;
    (&kernelContext_223)->scan_0 = scan_87;
    (&kernelContext_223)->parts_0 = parts_87;
    (&kernelContext_223)->colliders_0 = colliders_87;
    (&kernelContext_223)->collider_info_0 = collider_info_87;
    (&kernelContext_223)->view_chunks_0 = view_chunks_87;
    (&kernelContext_223)->view_0 = view_87;
    (&kernelContext_223)->objects_0 = objects_87;
    (&kernelContext_223)->shot_bodies_0 = shot_bodies_87;
    (&kernelContext_223)->lines_0 = lines_87;
    (&kernelContext_223)->geometry_0 = geometry_87;
    (&kernelContext_223)->shots_0 = shots_87;
    (&kernelContext_223)->lp_0 = lp_87;
    (&kernelContext_223)->ext_0 = ext_89;
    (&kernelContext_223)->engine_contacts_0 = engine_contacts_87;
    (&kernelContext_223)->crush_0 = crush_87;
    (&kernelContext_223)->crush_rw_0 = crush_rw_87;
    (&kernelContext_223)->crush_tab_0 = crush_tab_87;
    threadgroup array<uint, int(1024)> g_scan_87;
    (&kernelContext_223)->g_scan_0 = &g_scan_87;
    threadgroup array<float, int(64)> g_il_best_87;
    (&kernelContext_223)->g_il_best_0 = &g_il_best_87;
    uint k_70 = id_81.x;
    bool _S1689;
    if(k_70 >= (params_87->chunk_count_0))
    {
        _S1689 = true;
    }
    else
    {
        _S1689 = ((&kernelContext_223)->crush_0->enabled_0) == 0U;
    }
    if(_S1689)
    {
        return;
    }
    float4 _S1690 = float4(*((&kernelContext_223)->crush_rw_0+k_70)) ;
    uint flags_4 = (as_type<uint>((_S1690.w)));
    if((flags_4 & 2U) == 0U)
    {
        return;
    }
    *((&kernelContext_223)->crush_rw_0+k_70) = packed_float4(float4(_S1690.xyz, (as_type<float>((flags_4 & 4294967293U))))) ;
    uint _S1691 = body_of_0(k_70, &kernelContext_223);
    uint bw_1 = 7U * _S1691;
    float4 _S1692 = float4(*((&kernelContext_223)->bodies_0+bw_1)) ;
    Quat_0 orientation_1 = quat_of_0(float4(*((&kernelContext_223)->bodies_0+(bw_1 + 1U))) );
    float3 v_21 = (float4(*((&kernelContext_223)->bodies_0+(bw_1 + 3U))) ).xyz;
    float3 w_14 = (float4(*((&kernelContext_223)->bodies_0+(bw_1 + 4U))) ).xyz;
    uint _S1693 = 4U * k_70;
    float mass_1 = (float4(*((&kernelContext_223)->chunks_0+_S1693)) ).w;
    float volume_1 = (float4(*((&kernelContext_223)->crush_tab_0+(2U * k_70 + 1U))) ).y;
    uint _S1694 = w_rank_0(&kernelContext_223);
    uint device* _S1695 = (&kernelContext_223)->work_0+(_S1694 + k_70);
    uint scanned_0 = *_S1695;
    uint rank_0 = *_S1695 / 1048576U;
    uint _S1696 = crush_pieces_0(k_70, &kernelContext_223);
    if(rank_0 >= ((&kernelContext_223)->crush_0->max_events_0))
    {
        uint _S1697 = a_counters_1(&kernelContext_223);
        uint _S1698 = atomic_fetch_or_explicit((&kernelContext_223)->atom_0+(_S1697 + 3U), 32U, memory_order_relaxed);
    }
    float3 _S1699 = float3(0.0f) ;
    if(((&kernelContext_223)->crush_0->mode_1) == 1U)
    {
        _S1689 = ((&kernelContext_223)->crush_0->fragments_0) > 0U;
    }
    else
    {
        _S1689 = false;
    }
    uint first_6;
    uint n_7;
    uint i_26;
    float3 half_2;
    if(_S1689)
    {
        uint _S1700 = a_counters_1(&kernelContext_223);
        uint _S1701 = atomic_load_explicit((&kernelContext_223)->atom_0+(_S1700 + 8U), memory_order_relaxed);
        uint base_4 = _S1701 + (scanned_0 & 1048575U);
        if((base_4 + (&kernelContext_223)->crush_0->fragments_0) > ((&kernelContext_223)->crush_0->pool_size_0))
        {
            uint _S1702 = atomic_fetch_or_explicit((&kernelContext_223)->atom_0+(_S1700 + 3U), 16U, memory_order_relaxed);
            first_6 = 4294967295U;
        }
        else
        {
            first_6 = base_4;
        }
        thread Quat_0 q_35;
        thread float3 origin_14;
        structure_pose_0(_S1691, bw_1, &q_35, &origin_14, &kernelContext_223);
        float3 _S1703 = (float4(*((&kernelContext_223)->chunks_0+_S1693)) ).xyz;
        thread Quat_0 _S1704 = q_35;
        float3 _S1705 = rotate_0(&_S1704, _S1703);
        float3 _S1706 = origin_14 + _S1705;
        uint _S1707 = 3U * k_70;
        Quat_0 _S1708 = quat_of_0(float4(*((&kernelContext_223)->geometry_0+(_S1707 + 2U))) );
        thread Quat_0 _S1709 = q_35;
        thread Quat_0 _S1710 = _S1708;
        Quat_0 _S1711 = quat_mul_0(&_S1709, &_S1710);
        thread Quat_0 _S1712 = _S1711;
        Quat_0 _S1713 = normalized_0(&_S1712);
        float3 h_11 = (float4(*((&kernelContext_223)->geometry_0+(_S1707 + 1U))) ).xyz;
        thread uint3 g_0 = uint3(int3(int(1)) );
        thread float3 cell_0 = h_11;
        n_7 = (&kernelContext_223)->crush_0->fragments_0;
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
                _S1689 = (cell_0.x) >= (cell_0.z);
            }
            else
            {
                _S1689 = false;
            }
            if(_S1689)
            {
                i_26 = 0U;
            }
            else
            {
                if((cell_0.y) >= (cell_0.z))
                {
                    i_26 = 1U;
                }
                else
                {
                    i_26 = 2U;
                }
            }
            g_0[i_26] = g_0[i_26] * 2U;
            cell_0[i_26] = cell_0[i_26] * 0.5f;
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
        float mf_0 = mass_1 / float((&kernelContext_223)->crush_0->fragments_0);
        float _S1714 = half_3.y;
        float _S1715 = _S1714 * _S1714;
        float _S1716 = half_3.z;
        float _S1717 = _S1716 * _S1716;
        float _S1718 = half_3.x;
        float _S1719 = _S1718 * _S1718;
        float3 inertia_0 = float3((mf_0 / 3.0f))  * float3(_S1715 + _S1717, _S1719 + _S1717, _S1719 + _S1715);
        float3 _S1720 = float3(1.0f / inertia_0.x, 1.0f / inertia_0.y, 1.0f / inertia_0.z);
        thread Quat_0 _S1721 = _S1713;
        Quat_0 _S1722 = rot_to_rubble_0(&_S1721);
        float _S1723 = (float4(*((&kernelContext_223)->chunks_0+(_S1693 + 3U))) ).x;
        float _S1724 = min(_S1718, min(_S1714, _S1716));
        i_26 = 0U;
        for(;;)
        {
            if(i_26 < ((&kernelContext_223)->crush_0->fragments_0))
            {
                _S1689 = first_6 != 4294967295U;
            }
            else
            {
                _S1689 = false;
            }
            if(_S1689)
            {
            }
            else
            {
                break;
            }
            uint _S1725 = i_26 % g_0.x;
            uint _S1726 = i_26 / g_0.x;
            uint _S1727 = _S1726 % g_0.y;
            uint _S1728 = i_26 / (g_0.x * g_0.y);
            float3 offset_0 = - h_11 + cell_0 * (float3(2.0f)  * float3(uint3(_S1725, _S1727, _S1728)) + float3(1.0f) );
            thread Quat_0 _S1729 = _S1713;
            float3 _S1730 = rotate_0(&_S1729, offset_0);
            float3 x_8 = to_rubble_0(_S1706 + _S1730);
            float4 _S1731 = float4(*((&kernelContext_223)->crush_tab_0+((&kernelContext_223)->crush_0->t_pool_0 + first_6 + i_26))) ;
            uint fs_0 = (as_type<uint>((_S1731.x)));
            uint fc_3 = (as_type<uint>((_S1731.y)));
            float _S1732 = 1.0f / mf_0;
            float3 _S1733 = v_21 + cross(w_14, x_8 - _S1692.xyz);
            thread Quat_0 _S1734 = _S1722;
            write_body_0(fs_0, x_8, _S1732, &_S1734, _S1720, 9U, _S1733, w_14, &kernelContext_223);
            uint _S1735 = 4U * fc_3;
            uint _S1736 = _S1735 + 1U;
            *((&kernelContext_223)->colliders_0+_S1736) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_223)->colliders_0+_S1736)) ).w)) ;
            *((&kernelContext_223)->colliders_0+(_S1735 + 2U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
            *((&kernelContext_223)->colliders_0+(_S1735 + 3U)) = packed_float4(float4(half_3, length(half_3))) ;
            uint _S1737 = 2U * fc_3;
            *((&kernelContext_223)->collider_info_0+_S1737) = packed_uint4(uint4(4294967295U, fs_0, 2U, (uint4(*((&kernelContext_223)->collider_info_0+_S1737)) ).w)) ;
            *((&kernelContext_223)->collider_info_0+(_S1737 + 1U)) = packed_uint4(uint4((as_type<uint>((mf_0))), (as_type<uint>((_S1723))), (as_type<uint>((_S1724))), 0U)) ;
            i_26 = i_26 + 1U;
        }
        half_2 = half_3;
    }
    else
    {
        first_6 = 4294967295U;
        half_2 = _S1699;
    }
    if(((&kernelContext_223)->crush_0->mode_1) == 2U)
    {
        _S1689 = _S1696 > 0U;
    }
    else
    {
        _S1689 = false;
    }
    if(_S1689)
    {
        uint _S1738 = a_counters_1(&kernelContext_223);
        uint _S1739 = atomic_load_explicit((&kernelContext_223)->atom_0+(_S1738 + 8U), memory_order_relaxed);
        uint base_5 = _S1739 + (scanned_0 & 1048575U);
        if((base_5 + _S1696) > ((&kernelContext_223)->crush_0->pool_size_0))
        {
            uint _S1740 = atomic_fetch_or_explicit((&kernelContext_223)->atom_0+(_S1738 + 3U), 16U, memory_order_relaxed);
        }
        else
        {
            first_6 = base_5;
        }
        thread Quat_0 q_36;
        thread float3 origin_15;
        structure_pose_0(_S1691, bw_1, &q_36, &origin_15, &kernelContext_223);
        float3 _S1741 = (float4(*((&kernelContext_223)->chunks_0+_S1693)) ).xyz;
        thread Quat_0 _S1742 = q_36;
        float3 _S1743 = rotate_0(&_S1742, _S1741);
        float3 _S1744 = origin_15 + _S1743;
        uint _S1745 = 3U * k_70;
        Quat_0 _S1746 = quat_of_0(float4(*((&kernelContext_223)->geometry_0+(_S1745 + 2U))) );
        thread Quat_0 _S1747 = q_36;
        thread Quat_0 _S1748 = _S1746;
        Quat_0 _S1749 = quat_mul_0(&_S1747, &_S1748);
        thread Quat_0 _S1750 = _S1749;
        Quat_0 _S1751 = normalized_0(&_S1750);
        float3 h_12 = (float4(*((&kernelContext_223)->geometry_0+(_S1745 + 1U))) ).xyz;
        float _S1752 = float(_S1696);
        float _S1753 = max(pow(0.75f * volume_1 / (3.14159274101257324f * _S1752), 0.3333333432674408f), 0.0020000000949949f);
        float3 _S1754 = float3(_S1753, _S1753, _S1753);
        float _S1755 = mass_1 / _S1752;
        float _S1756 = (float4(*((&kernelContext_223)->chunks_0+(_S1693 + 3U))) ).x;
        float3 _S1757 = max(h_12 - float3(_S1753) , float3(0.25f)  * h_12);
        i_26 = 0U;
        float3 mean_4 = _S1699;
        for(;;)
        {
            if(i_26 < _S1696)
            {
            }
            else
            {
                break;
            }
            float3 mean_5 = mean_4 + crush_particle_spot_0(k_70, i_26);
            i_26 = i_26 + 1U;
            mean_4 = mean_5;
        }
        float3 mean_6 = mean_4 / float3(_S1752) ;
        i_26 = 0U;
        for(;;)
        {
            if(i_26 < _S1696)
            {
                _S1689 = first_6 != 4294967295U;
            }
            else
            {
                _S1689 = false;
            }
            if(_S1689)
            {
            }
            else
            {
                break;
            }
            float3 _S1758 = _S1757 * (crush_particle_spot_0(k_70, i_26) - mean_6);
            thread Quat_0 _S1759 = _S1751;
            float3 _S1760 = rotate_0(&_S1759, _S1758);
            float3 x_9 = to_rubble_0(_S1744 + _S1760);
            float4 _S1761 = float4(*((&kernelContext_223)->crush_tab_0+((&kernelContext_223)->crush_0->t_pool_0 + first_6 + i_26))) ;
            uint fs_1 = (as_type<uint>((_S1761.x)));
            uint fc_4 = (as_type<uint>((_S1761.y)));
            float _S1762 = 1.0f / _S1755;
            float4 _S1763 = float4(0.0f, 0.0f, 0.0f, 1.0f);
            Quat_0 _S1764 = quat_of_0(_S1763);
            if(((&kernelContext_223)->crush_0->particle_ccd_0) != 0U)
            {
                n_7 = 33024U;
            }
            else
            {
                n_7 = 0U;
            }
            uint _S1765 = 9U | n_7;
            float3 _S1766 = v_21 + cross(w_14, x_9 - _S1692.xyz);
            thread Quat_0 _S1767 = _S1764;
            write_body_0(fs_1, x_9, _S1762, &_S1767, _S1699, _S1765, _S1766, _S1699, &kernelContext_223);
            uint _S1768 = 4U * fc_4;
            uint _S1769 = _S1768 + 1U;
            *((&kernelContext_223)->colliders_0+_S1769) = packed_float4(float4(0.0f, 0.0f, 0.0f, (float4(*((&kernelContext_223)->colliders_0+_S1769)) ).w)) ;
            *((&kernelContext_223)->colliders_0+(_S1768 + 2U)) = packed_float4(_S1763) ;
            *((&kernelContext_223)->colliders_0+(_S1768 + 3U)) = packed_float4(float4(_S1753, 0.0f, 0.0f, _S1753)) ;
            uint _S1770 = 2U * fc_4;
            *((&kernelContext_223)->collider_info_0+_S1770) = packed_uint4(uint4(4294967295U, fs_1, 2U, (uint4(*((&kernelContext_223)->collider_info_0+_S1770)) ).w)) ;
            *((&kernelContext_223)->collider_info_0+(_S1770 + 1U)) = packed_uint4(uint4((as_type<uint>((_S1755))), (as_type<uint>((_S1756))), (as_type<uint>((_S1753))), 0U)) ;
            i_26 = i_26 + 1U;
        }
        half_2 = _S1754;
    }
    if(rank_0 < ((&kernelContext_223)->crush_0->max_events_0))
    {
        uint ev_0 = (&kernelContext_223)->crush_0->r_events_0 + 6U * rank_0;
        *((&kernelContext_223)->crush_rw_0+ev_0) = packed_float4(float4((as_type<float>((k_70))), (as_type<float>((first_6))), (as_type<float>((flags_4 >> 4U))), _S1690.x)) ;
        *((&kernelContext_223)->crush_rw_0+(ev_0 + 1U)) = packed_float4(float4(_S1692.xyz, mass_1)) ;
        packed_float4 device* _S1771 = (&kernelContext_223)->crush_rw_0+(ev_0 + 2U);
        thread Quat_0 _S1772 = orientation_1;
        float4 _S1773 = quat_vec_0(&_S1772);
        *_S1771 = packed_float4(_S1773) ;
        *((&kernelContext_223)->crush_rw_0+(ev_0 + 3U)) = packed_float4(float4(v_21, _S1690.y)) ;
        *((&kernelContext_223)->crush_rw_0+(ev_0 + 4U)) = packed_float4(float4(w_14, _S1690.z)) ;
        *((&kernelContext_223)->crush_rw_0+(ev_0 + 5U)) = packed_float4(float4(half_2, volume_1)) ;
    }
    float3 park_0 = float3(float(_S1691 % 1024U) * (&kernelContext_223)->crush_0->park_step_0, (&kernelContext_223)->crush_0->park_y_0, float(_S1691 / 1024U) * (&kernelContext_223)->crush_0->park_step_0);
    thread Quat_0 _S1774 = orientation_1;
    write_body_0(_S1691, park_0, 0.0f, &_S1774, _S1699, 0U, _S1699, _S1699, &kernelContext_223);
    return;
}

[[kernel]] void fr_crush_end(uint3 id_82 [[thread_position_in_grid]], FractureParams_0 constant* params_88 [[buffer(0)]], atomic<uint> device* atom_88 [[buffer(5)]], uint device* work_88 [[buffer(4)]], packed_float4 device* frames_88 [[buffer(6)]], packed_float4 device* bodies_88 [[buffer(7)]], packed_float4 device* chunks_88 [[buffer(1)]], uint device* sort_88 [[buffer(11)]], Record_natural_0 device* manifolds_88 [[buffer(10)]], packed_float4 device* loads_88 [[buffer(12)]], packed_uint4 device* bonds_89 [[buffer(2)]], uint device* args_88 [[buffer(13)]], uint device* info_88 [[buffer(15)]], ScanParams_0 constant* scan_88 [[buffer(14)]], packed_float4 device* parts_88 [[buffer(3)]], packed_float4 device* colliders_88 [[buffer(8)]], packed_uint4 device* collider_info_88 [[buffer(9)]], packed_float4 device* view_chunks_88 [[buffer(17)]], ViewParams_0 constant* view_88 [[buffer(16)]], packed_float4 device* objects_88 [[buffer(18)]], uint device* shot_bodies_88 [[buffer(20)]], float device* lines_88 [[buffer(19)]], packed_float4 device* geometry_88 [[buffer(21)]], packed_float4 device* shots_88 [[buffer(22)]], packed_float4 device* lp_88 [[buffer(24)]], packed_float4 device* ext_90 [[buffer(23)]], packed_float4 device* engine_contacts_88 [[buffer(25)]], CrushParams_0 constant* crush_88 [[buffer(28)]], packed_float4 device* crush_rw_88 [[buffer(27)]], packed_float4 device* crush_tab_88 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_224;
    (&kernelContext_224)->params_0 = params_88;
    (&kernelContext_224)->atom_0 = atom_88;
    (&kernelContext_224)->work_0 = work_88;
    (&kernelContext_224)->frames_0 = frames_88;
    (&kernelContext_224)->bodies_0 = bodies_88;
    (&kernelContext_224)->chunks_0 = chunks_88;
    (&kernelContext_224)->sort_0 = sort_88;
    (&kernelContext_224)->manifolds_0 = manifolds_88;
    (&kernelContext_224)->loads_0 = loads_88;
    (&kernelContext_224)->bonds_0 = bonds_89;
    (&kernelContext_224)->args_0 = args_88;
    (&kernelContext_224)->info_0 = info_88;
    (&kernelContext_224)->scan_0 = scan_88;
    (&kernelContext_224)->parts_0 = parts_88;
    (&kernelContext_224)->colliders_0 = colliders_88;
    (&kernelContext_224)->collider_info_0 = collider_info_88;
    (&kernelContext_224)->view_chunks_0 = view_chunks_88;
    (&kernelContext_224)->view_0 = view_88;
    (&kernelContext_224)->objects_0 = objects_88;
    (&kernelContext_224)->shot_bodies_0 = shot_bodies_88;
    (&kernelContext_224)->lines_0 = lines_88;
    (&kernelContext_224)->geometry_0 = geometry_88;
    (&kernelContext_224)->shots_0 = shots_88;
    (&kernelContext_224)->lp_0 = lp_88;
    (&kernelContext_224)->ext_0 = ext_90;
    (&kernelContext_224)->engine_contacts_0 = engine_contacts_88;
    (&kernelContext_224)->crush_0 = crush_88;
    (&kernelContext_224)->crush_rw_0 = crush_rw_88;
    (&kernelContext_224)->crush_tab_0 = crush_tab_88;
    threadgroup array<uint, int(1024)> g_scan_88;
    (&kernelContext_224)->g_scan_0 = &g_scan_88;
    threadgroup array<float, int(64)> g_il_best_88;
    (&kernelContext_224)->g_il_best_0 = &g_il_best_88;
    uint blocks_2 = (params_88->chunk_count_0 + 1023U) / 1024U;
    uint _S1775 = scan_new_blocks_0(&kernelContext_224);
    uint device* _S1776 = (&kernelContext_224)->work_0+(_S1775 + blocks_2);
    uint total_3 = *_S1776;
    uint events_0 = *_S1776 / 1048576U;
    uint _S1777 = a_counters_1(&kernelContext_224);
    atomic_store_explicit((&kernelContext_224)->atom_0+(_S1777 + 7U), events_0, memory_order_relaxed);
    uint _S1778 = atomic_fetch_add_explicit((&kernelContext_224)->atom_0+(_S1777 + 9U), events_0, memory_order_relaxed);
    uint _S1779 = atomic_fetch_add_explicit((&kernelContext_224)->atom_0+(_S1777 + 8U), total_3 & 1048575U, memory_order_relaxed);
    return;
}

uint l_mark_0(uint par_0, uint k_71, KernelContext_0 thread* kernelContext_225)
{
    return kernelContext_225->params_0->list_atom_0 + 2U + par_0 * kernelContext_225->params_0->chunk_count_0 + k_71;
}

uint l_count_0(uint par_1, KernelContext_0 thread* kernelContext_226)
{
    return kernelContext_226->params_0->list_atom_0 + par_1;
}

uint l_entry_0(uint par_2, uint i_27, KernelContext_0 thread* kernelContext_227)
{
    return kernelContext_227->params_0->list_atom_0 + 2U + 2U * kernelContext_227->params_0->chunk_count_0 + par_2 * kernelContext_227->params_0->chunk_count_0 + i_27;
}

void list_append_0(uint par_3, uint k_72, uint stamp_0, KernelContext_0 thread* kernelContext_228)
{
    uint _S1780 = l_mark_0(par_3, k_72, kernelContext_228);
    uint _S1781 = atomic_exchange_explicit(kernelContext_228->atom_0+_S1780, stamp_0, memory_order_relaxed);
    if(_S1781 != stamp_0)
    {
        uint _S1782 = l_count_0(par_3, kernelContext_228);
        uint i_28 = atomic_fetch_add_explicit(kernelContext_228->atom_0+_S1782, 1U, memory_order_relaxed);
        uint _S1783 = l_entry_0(par_3, i_28, kernelContext_228);
        atomic_store_explicit(kernelContext_228->atom_0+_S1783, k_72, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_list_touched(uint3 id_83 [[thread_position_in_grid]], FractureParams_0 constant* params_89 [[buffer(0)]], atomic<uint> device* atom_89 [[buffer(5)]], uint device* work_89 [[buffer(4)]], packed_float4 device* frames_89 [[buffer(6)]], packed_float4 device* bodies_89 [[buffer(7)]], packed_float4 device* chunks_89 [[buffer(1)]], uint device* sort_89 [[buffer(11)]], Record_natural_0 device* manifolds_89 [[buffer(10)]], packed_float4 device* loads_89 [[buffer(12)]], packed_uint4 device* bonds_90 [[buffer(2)]], uint device* args_89 [[buffer(13)]], uint device* info_89 [[buffer(15)]], ScanParams_0 constant* scan_89 [[buffer(14)]], packed_float4 device* parts_89 [[buffer(3)]], packed_float4 device* colliders_89 [[buffer(8)]], packed_uint4 device* collider_info_89 [[buffer(9)]], packed_float4 device* view_chunks_89 [[buffer(17)]], ViewParams_0 constant* view_89 [[buffer(16)]], packed_float4 device* objects_89 [[buffer(18)]], uint device* shot_bodies_89 [[buffer(20)]], float device* lines_89 [[buffer(19)]], packed_float4 device* geometry_89 [[buffer(21)]], packed_float4 device* shots_89 [[buffer(22)]], packed_float4 device* lp_89 [[buffer(24)]], packed_float4 device* ext_91 [[buffer(23)]], packed_float4 device* engine_contacts_89 [[buffer(25)]], CrushParams_0 constant* crush_89 [[buffer(28)]], packed_float4 device* crush_rw_89 [[buffer(27)]], packed_float4 device* crush_tab_89 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_229;
    (&kernelContext_229)->params_0 = params_89;
    (&kernelContext_229)->atom_0 = atom_89;
    (&kernelContext_229)->work_0 = work_89;
    (&kernelContext_229)->frames_0 = frames_89;
    (&kernelContext_229)->bodies_0 = bodies_89;
    (&kernelContext_229)->chunks_0 = chunks_89;
    (&kernelContext_229)->sort_0 = sort_89;
    (&kernelContext_229)->manifolds_0 = manifolds_89;
    (&kernelContext_229)->loads_0 = loads_89;
    (&kernelContext_229)->bonds_0 = bonds_90;
    (&kernelContext_229)->args_0 = args_89;
    (&kernelContext_229)->info_0 = info_89;
    (&kernelContext_229)->scan_0 = scan_89;
    (&kernelContext_229)->parts_0 = parts_89;
    (&kernelContext_229)->colliders_0 = colliders_89;
    (&kernelContext_229)->collider_info_0 = collider_info_89;
    (&kernelContext_229)->view_chunks_0 = view_chunks_89;
    (&kernelContext_229)->view_0 = view_89;
    (&kernelContext_229)->objects_0 = objects_89;
    (&kernelContext_229)->shot_bodies_0 = shot_bodies_89;
    (&kernelContext_229)->lines_0 = lines_89;
    (&kernelContext_229)->geometry_0 = geometry_89;
    (&kernelContext_229)->shots_0 = shots_89;
    (&kernelContext_229)->lp_0 = lp_89;
    (&kernelContext_229)->ext_0 = ext_91;
    (&kernelContext_229)->engine_contacts_0 = engine_contacts_89;
    (&kernelContext_229)->crush_0 = crush_89;
    (&kernelContext_229)->crush_rw_0 = crush_rw_89;
    (&kernelContext_229)->crush_tab_0 = crush_tab_89;
    threadgroup array<uint, int(1024)> g_scan_89;
    (&kernelContext_229)->g_scan_0 = &g_scan_89;
    threadgroup array<float, int(64)> g_il_best_89;
    (&kernelContext_229)->g_il_best_0 = &g_il_best_89;
    uint _S1784 = id_83.x;
    if(_S1784 >= sort_89[params_89->list_touched_0])
    {
        return;
    }
    list_append_0((&kernelContext_229)->params_0->parity_0, (&kernelContext_229)->sort_0[params_89->list_touched_0 + 1U + _S1784], (&kernelContext_229)->params_0->list_stamp_0, &kernelContext_229);
    return;
}

[[kernel]] void fr_list_all(uint3 id_84 [[thread_position_in_grid]], FractureParams_0 constant* params_90 [[buffer(0)]], atomic<uint> device* atom_90 [[buffer(5)]], uint device* work_90 [[buffer(4)]], packed_float4 device* frames_90 [[buffer(6)]], packed_float4 device* bodies_90 [[buffer(7)]], packed_float4 device* chunks_90 [[buffer(1)]], uint device* sort_90 [[buffer(11)]], Record_natural_0 device* manifolds_90 [[buffer(10)]], packed_float4 device* loads_90 [[buffer(12)]], packed_uint4 device* bonds_91 [[buffer(2)]], uint device* args_90 [[buffer(13)]], uint device* info_90 [[buffer(15)]], ScanParams_0 constant* scan_90 [[buffer(14)]], packed_float4 device* parts_90 [[buffer(3)]], packed_float4 device* colliders_90 [[buffer(8)]], packed_uint4 device* collider_info_90 [[buffer(9)]], packed_float4 device* view_chunks_90 [[buffer(17)]], ViewParams_0 constant* view_90 [[buffer(16)]], packed_float4 device* objects_90 [[buffer(18)]], uint device* shot_bodies_90 [[buffer(20)]], float device* lines_90 [[buffer(19)]], packed_float4 device* geometry_90 [[buffer(21)]], packed_float4 device* shots_90 [[buffer(22)]], packed_float4 device* lp_90 [[buffer(24)]], packed_float4 device* ext_92 [[buffer(23)]], packed_float4 device* engine_contacts_90 [[buffer(25)]], CrushParams_0 constant* crush_90 [[buffer(28)]], packed_float4 device* crush_rw_90 [[buffer(27)]], packed_float4 device* crush_tab_90 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_230;
    (&kernelContext_230)->params_0 = params_90;
    (&kernelContext_230)->atom_0 = atom_90;
    (&kernelContext_230)->work_0 = work_90;
    (&kernelContext_230)->frames_0 = frames_90;
    (&kernelContext_230)->bodies_0 = bodies_90;
    (&kernelContext_230)->chunks_0 = chunks_90;
    (&kernelContext_230)->sort_0 = sort_90;
    (&kernelContext_230)->manifolds_0 = manifolds_90;
    (&kernelContext_230)->loads_0 = loads_90;
    (&kernelContext_230)->bonds_0 = bonds_91;
    (&kernelContext_230)->args_0 = args_90;
    (&kernelContext_230)->info_0 = info_90;
    (&kernelContext_230)->scan_0 = scan_90;
    (&kernelContext_230)->parts_0 = parts_90;
    (&kernelContext_230)->colliders_0 = colliders_90;
    (&kernelContext_230)->collider_info_0 = collider_info_90;
    (&kernelContext_230)->view_chunks_0 = view_chunks_90;
    (&kernelContext_230)->view_0 = view_90;
    (&kernelContext_230)->objects_0 = objects_90;
    (&kernelContext_230)->shot_bodies_0 = shot_bodies_90;
    (&kernelContext_230)->lines_0 = lines_90;
    (&kernelContext_230)->geometry_0 = geometry_90;
    (&kernelContext_230)->shots_0 = shots_90;
    (&kernelContext_230)->lp_0 = lp_90;
    (&kernelContext_230)->ext_0 = ext_92;
    (&kernelContext_230)->engine_contacts_0 = engine_contacts_90;
    (&kernelContext_230)->crush_0 = crush_90;
    (&kernelContext_230)->crush_rw_0 = crush_rw_90;
    (&kernelContext_230)->crush_tab_0 = crush_tab_90;
    threadgroup array<uint, int(1024)> g_scan_90;
    (&kernelContext_230)->g_scan_0 = &g_scan_90;
    threadgroup array<float, int(64)> g_il_best_90;
    (&kernelContext_230)->g_il_best_0 = &g_il_best_90;
    uint _S1785 = id_84.x;
    bool _S1786 = live_0(_S1785, &kernelContext_230);
    if(_S1786)
    {
        list_append_0((&kernelContext_230)->params_0->parity_0, _S1785, (&kernelContext_230)->params_0->list_stamp_0, &kernelContext_230);
    }
    return;
}

[[kernel]] void fr_list_args(uint3 id_85 [[thread_position_in_grid]], FractureParams_0 constant* params_91 [[buffer(0)]], atomic<uint> device* atom_91 [[buffer(5)]], uint device* work_91 [[buffer(4)]], packed_float4 device* frames_91 [[buffer(6)]], packed_float4 device* bodies_91 [[buffer(7)]], packed_float4 device* chunks_91 [[buffer(1)]], uint device* sort_91 [[buffer(11)]], Record_natural_0 device* manifolds_91 [[buffer(10)]], packed_float4 device* loads_91 [[buffer(12)]], packed_uint4 device* bonds_92 [[buffer(2)]], uint device* args_91 [[buffer(13)]], uint device* info_91 [[buffer(15)]], ScanParams_0 constant* scan_91 [[buffer(14)]], packed_float4 device* parts_91 [[buffer(3)]], packed_float4 device* colliders_91 [[buffer(8)]], packed_uint4 device* collider_info_91 [[buffer(9)]], packed_float4 device* view_chunks_91 [[buffer(17)]], ViewParams_0 constant* view_91 [[buffer(16)]], packed_float4 device* objects_91 [[buffer(18)]], uint device* shot_bodies_91 [[buffer(20)]], float device* lines_91 [[buffer(19)]], packed_float4 device* geometry_91 [[buffer(21)]], packed_float4 device* shots_91 [[buffer(22)]], packed_float4 device* lp_91 [[buffer(24)]], packed_float4 device* ext_93 [[buffer(23)]], packed_float4 device* engine_contacts_91 [[buffer(25)]], CrushParams_0 constant* crush_91 [[buffer(28)]], packed_float4 device* crush_rw_91 [[buffer(27)]], packed_float4 device* crush_tab_91 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_231;
    (&kernelContext_231)->params_0 = params_91;
    (&kernelContext_231)->atom_0 = atom_91;
    (&kernelContext_231)->work_0 = work_91;
    (&kernelContext_231)->frames_0 = frames_91;
    (&kernelContext_231)->bodies_0 = bodies_91;
    (&kernelContext_231)->chunks_0 = chunks_91;
    (&kernelContext_231)->sort_0 = sort_91;
    (&kernelContext_231)->manifolds_0 = manifolds_91;
    (&kernelContext_231)->loads_0 = loads_91;
    (&kernelContext_231)->bonds_0 = bonds_92;
    (&kernelContext_231)->args_0 = args_91;
    (&kernelContext_231)->info_0 = info_91;
    (&kernelContext_231)->scan_0 = scan_91;
    (&kernelContext_231)->parts_0 = parts_91;
    (&kernelContext_231)->colliders_0 = colliders_91;
    (&kernelContext_231)->collider_info_0 = collider_info_91;
    (&kernelContext_231)->view_chunks_0 = view_chunks_91;
    (&kernelContext_231)->view_0 = view_91;
    (&kernelContext_231)->objects_0 = objects_91;
    (&kernelContext_231)->shot_bodies_0 = shot_bodies_91;
    (&kernelContext_231)->lines_0 = lines_91;
    (&kernelContext_231)->geometry_0 = geometry_91;
    (&kernelContext_231)->shots_0 = shots_91;
    (&kernelContext_231)->lp_0 = lp_91;
    (&kernelContext_231)->ext_0 = ext_93;
    (&kernelContext_231)->engine_contacts_0 = engine_contacts_91;
    (&kernelContext_231)->crush_0 = crush_91;
    (&kernelContext_231)->crush_rw_0 = crush_rw_91;
    (&kernelContext_231)->crush_tab_0 = crush_tab_91;
    threadgroup array<uint, int(1024)> g_scan_91;
    (&kernelContext_231)->g_scan_0 = &g_scan_91;
    threadgroup array<float, int(64)> g_il_best_91;
    (&kernelContext_231)->g_il_best_0 = &g_il_best_91;
    uint _S1787 = l_count_0(params_91->parity_0, &kernelContext_231);
    uint _S1788 = atomic_load_explicit((&kernelContext_231)->atom_0+_S1787, memory_order_relaxed);
    uint _S1789 = groups_of_0(_S1788, 64U);
    set_args_0(17U, _S1789, &kernelContext_231);
    uint _S1790 = l_count_0(1U - params_91->parity_0, &kernelContext_231);
    atomic_store_explicit((&kernelContext_231)->atom_0+_S1790, 0U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_loads_listed(uint3 id_86 [[thread_position_in_grid]], FractureParams_0 constant* params_92 [[buffer(0)]], atomic<uint> device* atom_92 [[buffer(5)]], uint device* work_92 [[buffer(4)]], packed_float4 device* frames_92 [[buffer(6)]], packed_float4 device* bodies_92 [[buffer(7)]], packed_float4 device* chunks_92 [[buffer(1)]], uint device* sort_92 [[buffer(11)]], Record_natural_0 device* manifolds_92 [[buffer(10)]], packed_float4 device* loads_92 [[buffer(12)]], packed_uint4 device* bonds_93 [[buffer(2)]], uint device* args_92 [[buffer(13)]], uint device* info_92 [[buffer(15)]], ScanParams_0 constant* scan_92 [[buffer(14)]], packed_float4 device* parts_92 [[buffer(3)]], packed_float4 device* colliders_92 [[buffer(8)]], packed_uint4 device* collider_info_92 [[buffer(9)]], packed_float4 device* view_chunks_92 [[buffer(17)]], ViewParams_0 constant* view_92 [[buffer(16)]], packed_float4 device* objects_92 [[buffer(18)]], uint device* shot_bodies_92 [[buffer(20)]], float device* lines_92 [[buffer(19)]], packed_float4 device* geometry_92 [[buffer(21)]], packed_float4 device* shots_92 [[buffer(22)]], packed_float4 device* lp_92 [[buffer(24)]], packed_float4 device* ext_94 [[buffer(23)]], packed_float4 device* engine_contacts_92 [[buffer(25)]], CrushParams_0 constant* crush_92 [[buffer(28)]], packed_float4 device* crush_rw_92 [[buffer(27)]], packed_float4 device* crush_tab_92 [[buffer(26)]])
{
    bool first_7;
    float3 f_total_2;
    float m_10;
    uint p_24;
    uint kept_2;
    uint q3_1;
    bool overflow_2;
    bool overflow_3;
    float3 f_total_3;
    float3 t_total_2;
    thread KernelContext_0 kernelContext_232;
    (&kernelContext_232)->params_0 = params_92;
    (&kernelContext_232)->atom_0 = atom_92;
    (&kernelContext_232)->work_0 = work_92;
    (&kernelContext_232)->frames_0 = frames_92;
    (&kernelContext_232)->bodies_0 = bodies_92;
    (&kernelContext_232)->chunks_0 = chunks_92;
    (&kernelContext_232)->sort_0 = sort_92;
    (&kernelContext_232)->manifolds_0 = manifolds_92;
    (&kernelContext_232)->loads_0 = loads_92;
    (&kernelContext_232)->bonds_0 = bonds_93;
    (&kernelContext_232)->args_0 = args_92;
    (&kernelContext_232)->info_0 = info_92;
    (&kernelContext_232)->scan_0 = scan_92;
    (&kernelContext_232)->parts_0 = parts_92;
    (&kernelContext_232)->colliders_0 = colliders_92;
    (&kernelContext_232)->collider_info_0 = collider_info_92;
    (&kernelContext_232)->view_chunks_0 = view_chunks_92;
    (&kernelContext_232)->view_0 = view_92;
    (&kernelContext_232)->objects_0 = objects_92;
    (&kernelContext_232)->shot_bodies_0 = shot_bodies_92;
    (&kernelContext_232)->lines_0 = lines_92;
    (&kernelContext_232)->geometry_0 = geometry_92;
    (&kernelContext_232)->shots_0 = shots_92;
    (&kernelContext_232)->lp_0 = lp_92;
    (&kernelContext_232)->ext_0 = ext_94;
    (&kernelContext_232)->engine_contacts_0 = engine_contacts_92;
    (&kernelContext_232)->crush_0 = crush_92;
    (&kernelContext_232)->crush_rw_0 = crush_rw_92;
    (&kernelContext_232)->crush_tab_0 = crush_tab_92;
    threadgroup array<uint, int(1024)> g_scan_92;
    (&kernelContext_232)->g_scan_0 = &g_scan_92;
    threadgroup array<float, int(64)> g_il_best_92;
    (&kernelContext_232)->g_il_best_0 = &g_il_best_92;
    uint _S1791 = id_86.x;
    uint _S1792 = l_count_0(params_92->parity_0, &kernelContext_232);
    uint _S1793 = atomic_load_explicit((&kernelContext_232)->atom_0+_S1792, memory_order_relaxed);
    if(_S1791 >= _S1793)
    {
        return;
    }
    uint _S1794 = l_entry_0(params_92->parity_0, _S1791, &kernelContext_232);
    uint k_73 = atomic_load_explicit((&kernelContext_232)->atom_0+_S1794, memory_order_relaxed);
    bool _S1795 = live_0(k_73, &kernelContext_232);
    if(!_S1795)
    {
        return;
    }
    uint _S1796 = body_of_0(k_73, &kernelContext_232);
    uint b_66 = 7U * _S1796;
    thread Quat_0 q_37;
    thread float3 origin_16;
    structure_pose_0(_S1796, b_66, &q_37, &origin_16, &kernelContext_232);
    uint _S1797 = 4U * k_73;
    float4 _S1798 = float4(*((&kernelContext_232)->chunks_0+_S1797)) ;
    float3 _S1799 = _S1798.xyz;
    thread Quat_0 _S1800 = q_37;
    float3 _S1801 = rotate_0(&_S1800, _S1799);
    float3 centre_6 = origin_16 + _S1801;
    float inv_mass_15 = (float4(*((&kernelContext_232)->bodies_0+b_66)) ).w;
    float m_body_3;
    if(inv_mass_15 > 0.0f)
    {
        m_body_3 = 1.0f / inv_mass_15;
    }
    else
    {
        m_body_3 = float_bits_0(2139095040U);
    }
    uint from_6 = params_92->parity_0;
    uint to_2 = 1U - params_92->parity_0;
    uint _S1802 = w_pulses_0(&kernelContext_232);
    uint _S1803 = _S1802 + _S1797;
    uint _S1804 = _S1803 + 2U * params_92->parity_0;
    uint off_2 = *((&kernelContext_232)->work_0+_S1804);
    uint device* _S1805 = (&kernelContext_232)->work_0+(_S1804 + 1U);
    uint carried_1 = *_S1805;
    uint room_2 = *_S1805 + (&kernelContext_232)->sort_0[(&kernelContext_232)->params_0->chunk_count_0 + 1U + k_73];
    uint base_6;
    uint room_3;
    if(room_2 != 0U)
    {
        uint _S1806 = a_counters_1(&kernelContext_232);
        uint base_7 = atomic_fetch_add_explicit((&kernelContext_232)->atom_0+(_S1806 + 5U + to_2), room_2, memory_order_relaxed);
        if((base_7 + room_2) > ((&kernelContext_232)->params_0->pool_cap_0))
        {
            uint _S1807 = atomic_fetch_or_explicit((&kernelContext_232)->atom_0+(_S1806 + 3U), 8U, memory_order_relaxed);
            if(base_7 < ((&kernelContext_232)->params_0->pool_cap_0))
            {
                base_6 = (&kernelContext_232)->params_0->pool_cap_0 - base_7;
            }
            else
            {
                base_6 = 0U;
            }
        }
        else
        {
            base_6 = room_2;
        }
        uint _S1808 = base_6;
        base_6 = base_7;
        room_3 = _S1808;
    }
    else
    {
        base_6 = 0U;
        room_3 = room_2;
    }
    uint _S1809 = (&kernelContext_232)->params_0->chunk_count_0;
    float _S1810 = _S1798.w;
    float4 _S1811 = float4(*((&kernelContext_232)->chunks_0+(_S1797 + 3U))) ;
    float _S1812 = (&kernelContext_232)->params_0->band_0;
    float _S1813 = (&kernelContext_232)->params_0->dt_0;
    bool _S1814 = ((&kernelContext_232)->params_0->impact_peak_0) != 0U;
    uint _S1815 = l_pool_0(from_6, &kernelContext_232);
    uint _S1816 = _S1815 + 3U * off_2;
    uint _S1817 = l_pool_0(to_2, &kernelContext_232);
    uint _S1818 = _S1817 + 3U * base_6;
    for(;;)
    {
        float3 _S1819 = float3(0.0f) ;
        uint _S1820 = (&kernelContext_232)->sort_0[k_73];
        uint _S1821 = (&kernelContext_232)->sort_0[_S1809 + 1U + k_73];
        first_7 = true;
        uint last_1 = 0U;
        uint step_9 = 0U;
        f_total_2 = _S1819;
        float3 t_total_3 = _S1819;
        for(;;)
        {
            if(step_9 < _S1821)
            {
            }
            else
            {
                break;
            }
            p_24 = 4294967295U;
            kept_2 = 0U;
            q3_1 = 0U;
            for(;;)
            {
                if(q3_1 < _S1821)
                {
                }
                else
                {
                    break;
                }
                uint key_3 = (uint4((&kernelContext_232)->manifolds_0[_S1820 + q3_1].ids_0) ).y + 1U;
                if(first_7)
                {
                    overflow_2 = true;
                }
                else
                {
                    overflow_2 = key_3 > last_1;
                }
                if(overflow_2)
                {
                    if(p_24 == 4294967295U)
                    {
                        overflow_3 = true;
                    }
                    else
                    {
                        overflow_3 = key_3 < kept_2;
                    }
                }
                else
                {
                    overflow_3 = false;
                }
                if(overflow_3)
                {
                    p_24 = q3_1;
                    kept_2 = key_3;
                }
                q3_1 = q3_1 + 1U;
            }
            if(p_24 == 4294967295U)
            {
                break;
            }
            Record_natural_0 r_33 = (&kernelContext_232)->manifolds_0[_S1820 + p_24];
            float4 _S1822 = float4(r_33.impulse_0) ;
            float3 _S1823 = _S1822.xyz;
            float j_17 = length(_S1823);
            if(j_17 == 0.0f)
            {
                uint _S1824 = step_9 + 1U;
                first_7 = false;
                last_1 = kept_2;
                step_9 = _S1824;
                continue;
            }
            float m_other_4 = (float4(r_33.other_0) ).x;
            if(isinf(m_body_3))
            {
                if(isinf(m_other_4))
                {
                    m_10 = _S1810;
                }
                else
                {
                    m_10 = m_other_4;
                }
            }
            else
            {
                if(isinf(m_other_4))
                {
                    m_10 = m_body_3;
                }
                else
                {
                    m_10 = m_body_3 * m_other_4 / (m_body_3 + m_other_4);
                }
            }
            float _S1825 = min(j_17, 2.0f * m_10 * max(_S1822.w - _S1812, 0.0f));
            if(_S1825 < j_17)
            {
                float3 f_17 = _S1823 / float3(j_17)  * float3(((j_17 - _S1825) / _S1813)) ;
                float3 _S1826 = t_total_3 + cross((float4(r_33.point_0) ).xyz - centre_6, f_17);
                f_total_3 = f_total_2 + f_17;
                t_total_2 = _S1826;
            }
            else
            {
                f_total_3 = f_total_2;
                t_total_2 = t_total_3;
            }
            f_total_2 = f_total_3;
            t_total_3 = t_total_2;
            uint _S1824 = step_9 + 1U;
            first_7 = false;
            last_1 = kept_2;
            step_9 = _S1824;
        }
        p_24 = 0U;
        kept_2 = 0U;
        for(;;)
        {
            if(p_24 < carried_1)
            {
            }
            else
            {
                break;
            }
            uint _S1827 = _S1816 + 3U * p_24;
            float4 _S1828 = float4(*((&kernelContext_232)->loads_0+_S1827)) ;
            thread float4 a_14 = _S1828;
            float4 _S1829 = float4(*((&kernelContext_232)->loads_0+(_S1827 + 1U))) ;
            float4 _S1830 = float4(*((&kernelContext_232)->loads_0+(_S1827 + 2U))) ;
            float _S1831 = _S1829.w;
            float mean_7 = pulse_mean_0(_S1828.x, _S1828.y, _S1828.z, _S1828.w, _S1831, _S1813);
            if(mean_7 != 0.0f)
            {
                float3 f_18 = _S1830.xyz * float3(mean_7) ;
                float3 _S1832 = t_total_3 + cross(_S1829.xyz - centre_6, f_18);
                f_total_2 = f_total_2 + f_18;
                t_total_3 = _S1832;
            }
            if((a_14.x + a_14.z + _S1831) > _S1813)
            {
                a_14.x = a_14.x - _S1813;
                uint _S1833 = _S1818 + 3U * kept_2;
                *((&kernelContext_232)->loads_0+_S1833) = packed_float4(a_14) ;
                *((&kernelContext_232)->loads_0+(_S1833 + 1U)) = packed_float4(_S1829) ;
                *((&kernelContext_232)->loads_0+(_S1833 + 2U)) = packed_float4(_S1830) ;
                kept_2 = kept_2 + 1U;
            }
            p_24 = p_24 + 1U;
        }
        q3_1 = 0U;
        overflow_2 = false;
        for(;;)
        {
            if(q3_1 < _S1821)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_34 = (&kernelContext_232)->manifolds_0[_S1820 + q3_1];
            float4 _S1834 = float4(r_34.impulse_0) ;
            float3 _S1835 = _S1834.xyz;
            float j_18 = length(_S1835);
            if(j_18 == 0.0f)
            {
                overflow_3 = overflow_2;
                q3_1 = q3_1 + 1U;
                overflow_2 = overflow_3;
                continue;
            }
            float4 _S1836 = float4(r_34.other_0) ;
            float m_other_5 = _S1836.x;
            if(isinf(m_body_3))
            {
                if(isinf(m_other_5))
                {
                    m_10 = _S1810;
                }
                else
                {
                    m_10 = m_other_5;
                }
            }
            else
            {
                if(isinf(m_other_5))
                {
                    m_10 = m_body_3;
                }
                else
                {
                    m_10 = m_body_3 * m_other_5 / (m_body_3 + m_other_5);
                }
            }
            float _S1837 = _S1834.w;
            float _S1838 = min(j_18, 2.0f * m_10 * max(_S1837 - _S1812, 0.0f));
            if(_S1838 <= 0.0f)
            {
                overflow_3 = overflow_2;
                q3_1 = q3_1 + 1U;
                overflow_2 = overflow_3;
                continue;
            }
            float _S1839 = _S1811.y;
            float duration_4 = hertz_duration_0(m_10, min(_S1836.z, _S1811.z), 1.0f / ((1.0f - _S1839 * _S1839) / _S1811.x + 1.0f / _S1836.y), _S1837);
            float peak_4 = 3.14159274101257324f * _S1838 / (2.0f * duration_4);
            float3 dir_3 = _S1835 / float3(j_18) ;
            float mean_8;
            if(_S1814)
            {
                mean_8 = peak_4;
            }
            else
            {
                mean_8 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_4, duration_4, _S1813);
            }
            if(mean_8 != 0.0f)
            {
                float3 f_19 = dir_3 * float3(mean_8) ;
                float3 _S1840 = t_total_3 + cross((float4(r_34.point_0) ).xyz - centre_6, f_19);
                f_total_3 = f_total_2 + f_19;
                t_total_2 = _S1840;
            }
            else
            {
                f_total_3 = f_total_2;
                t_total_2 = t_total_3;
            }
            uint kept_3;
            if(duration_4 > _S1813)
            {
                if(kept_2 < room_3)
                {
                    uint _S1841 = _S1818 + 3U * kept_2;
                    *((&kernelContext_232)->loads_0+_S1841) = packed_float4(float4(- _S1813, 0.0f, 0.0f, peak_4)) ;
                    *((&kernelContext_232)->loads_0+(_S1841 + 1U)) = packed_float4(float4((float4(r_34.point_0) ).xyz, duration_4)) ;
                    *((&kernelContext_232)->loads_0+(_S1841 + 2U)) = packed_float4(float4(dir_3, 0.0f)) ;
                    kept_3 = kept_2 + 1U;
                    overflow_3 = overflow_2;
                }
                else
                {
                    kept_3 = kept_2;
                    overflow_3 = true;
                }
            }
            else
            {
                kept_3 = kept_2;
                overflow_3 = overflow_2;
            }
            f_total_2 = f_total_3;
            t_total_3 = t_total_2;
            kept_2 = kept_3;
            q3_1 = q3_1 + 1U;
            overflow_2 = overflow_3;
        }
        break;
    }
    if(overflow_2)
    {
        uint _S1842 = a_counters_1(&kernelContext_232);
        uint _S1843 = atomic_fetch_or_explicit((&kernelContext_232)->atom_0+(_S1842 + 3U), 8U, memory_order_relaxed);
    }
    uint _S1844 = _S1803 + 2U * to_2;
    *((&kernelContext_232)->work_0+_S1844) = base_6;
    *((&kernelContext_232)->work_0+(_S1844 + 1U)) = kept_2;
    float load_0 = length(f_total_2);
    *((&kernelContext_232)->loads_0+k_73) = packed_float4(float4(load_0, 0.0f, 0.0f, 0.0f)) ;
    if(base_6 != 0U)
    {
        first_7 = true;
    }
    else
    {
        first_7 = kept_2 != 0U;
    }
    if(first_7)
    {
        first_7 = true;
    }
    else
    {
        first_7 = off_2 != 0U;
    }
    if(first_7)
    {
        first_7 = true;
    }
    else
    {
        first_7 = carried_1 != 0U;
    }
    if(first_7)
    {
        first_7 = true;
    }
    else
    {
        first_7 = (as_type<uint>((load_0))) != 0U;
    }
    if(first_7)
    {
        list_append_0(to_2, k_73, (&kernelContext_232)->params_0->list_stamp_0 + 1U, &kernelContext_232);
    }
    return;
}

uint adj_first_0(uint k_74, KernelContext_0 thread* kernelContext_233)
{
    return kernelContext_233->info_0[kernelContext_233->params_0->chunk_count_0 + 2U * k_74];
}

uint adj_count_0(uint k_75, KernelContext_0 thread* kernelContext_234)
{
    return kernelContext_234->info_0[kernelContext_234->params_0->chunk_count_0 + 2U * k_75 + 1U];
}

uint adj_at_0(uint j_19, KernelContext_0 thread* kernelContext_235)
{
    return kernelContext_235->info_0[3U * kernelContext_235->params_0->chunk_count_0 + j_19];
}

[[kernel]] void fr_breaks_listed(uint3 id_87 [[thread_position_in_grid]], FractureParams_0 constant* params_93 [[buffer(0)]], atomic<uint> device* atom_93 [[buffer(5)]], uint device* work_93 [[buffer(4)]], packed_float4 device* frames_93 [[buffer(6)]], packed_float4 device* bodies_93 [[buffer(7)]], packed_float4 device* chunks_93 [[buffer(1)]], uint device* sort_93 [[buffer(11)]], Record_natural_0 device* manifolds_93 [[buffer(10)]], packed_float4 device* loads_93 [[buffer(12)]], packed_uint4 device* bonds_94 [[buffer(2)]], uint device* args_93 [[buffer(13)]], uint device* info_93 [[buffer(15)]], ScanParams_0 constant* scan_93 [[buffer(14)]], packed_float4 device* parts_93 [[buffer(3)]], packed_float4 device* colliders_93 [[buffer(8)]], packed_uint4 device* collider_info_93 [[buffer(9)]], packed_float4 device* view_chunks_93 [[buffer(17)]], ViewParams_0 constant* view_93 [[buffer(16)]], packed_float4 device* objects_93 [[buffer(18)]], uint device* shot_bodies_93 [[buffer(20)]], float device* lines_93 [[buffer(19)]], packed_float4 device* geometry_93 [[buffer(21)]], packed_float4 device* shots_93 [[buffer(22)]], packed_float4 device* lp_93 [[buffer(24)]], packed_float4 device* ext_95 [[buffer(23)]], packed_float4 device* engine_contacts_93 [[buffer(25)]], CrushParams_0 constant* crush_93 [[buffer(28)]], packed_float4 device* crush_rw_93 [[buffer(27)]], packed_float4 device* crush_tab_93 [[buffer(26)]])
{
    thread KernelContext_0 kernelContext_236;
    (&kernelContext_236)->params_0 = params_93;
    (&kernelContext_236)->atom_0 = atom_93;
    (&kernelContext_236)->work_0 = work_93;
    (&kernelContext_236)->frames_0 = frames_93;
    (&kernelContext_236)->bodies_0 = bodies_93;
    (&kernelContext_236)->chunks_0 = chunks_93;
    (&kernelContext_236)->sort_0 = sort_93;
    (&kernelContext_236)->manifolds_0 = manifolds_93;
    (&kernelContext_236)->loads_0 = loads_93;
    (&kernelContext_236)->bonds_0 = bonds_94;
    (&kernelContext_236)->args_0 = args_93;
    (&kernelContext_236)->info_0 = info_93;
    (&kernelContext_236)->scan_0 = scan_93;
    (&kernelContext_236)->parts_0 = parts_93;
    (&kernelContext_236)->colliders_0 = colliders_93;
    (&kernelContext_236)->collider_info_0 = collider_info_93;
    (&kernelContext_236)->view_chunks_0 = view_chunks_93;
    (&kernelContext_236)->view_0 = view_93;
    (&kernelContext_236)->objects_0 = objects_93;
    (&kernelContext_236)->shot_bodies_0 = shot_bodies_93;
    (&kernelContext_236)->lines_0 = lines_93;
    (&kernelContext_236)->geometry_0 = geometry_93;
    (&kernelContext_236)->shots_0 = shots_93;
    (&kernelContext_236)->lp_0 = lp_93;
    (&kernelContext_236)->ext_0 = ext_95;
    (&kernelContext_236)->engine_contacts_0 = engine_contacts_93;
    (&kernelContext_236)->crush_0 = crush_93;
    (&kernelContext_236)->crush_rw_0 = crush_rw_93;
    (&kernelContext_236)->crush_tab_0 = crush_tab_93;
    threadgroup array<uint, int(1024)> g_scan_93;
    (&kernelContext_236)->g_scan_0 = &g_scan_93;
    threadgroup array<float, int(64)> g_il_best_93;
    (&kernelContext_236)->g_il_best_0 = &g_il_best_93;
    uint par_4 = params_93->parity_0;
    uint _S1845 = id_87.x;
    uint _S1846 = l_count_0(params_93->parity_0, &kernelContext_236);
    uint _S1847 = atomic_load_explicit((&kernelContext_236)->atom_0+_S1846, memory_order_relaxed);
    if(_S1845 >= _S1847)
    {
        return;
    }
    uint _S1848 = l_entry_0(par_4, _S1845, &kernelContext_236);
    uint k_76 = atomic_load_explicit((&kernelContext_236)->atom_0+_S1848, memory_order_relaxed);
    uint _S1849 = adj_first_0(k_76, &kernelContext_236);
    uint _S1850 = adj_count_0(k_76, &kernelContext_236);
    uint j_20 = 0U;
    for(;;)
    {
        if(j_20 < _S1850)
        {
        }
        else
        {
            break;
        }
        uint _S1851 = adj_at_0(_S1849 + j_20, &kernelContext_236);
        bool _S1852;
        if(_S1851 >= ((&kernelContext_236)->params_0->bond_used_0))
        {
            _S1852 = true;
        }
        else
        {
            uint _S1853 = w_alive_0(&kernelContext_236);
            _S1852 = (*((&kernelContext_236)->work_0+(_S1853 + _S1851))) == 0U;
        }
        if(_S1852)
        {
            j_20 = j_20 + 1U;
            continue;
        }
        uint4 _S1854 = uint4(*((&kernelContext_236)->bonds_0+_S1851)) ;
        uint _S1855 = _S1854.x;
        uint other_1;
        if(_S1855 == k_76)
        {
            other_1 = _S1854.y;
        }
        else
        {
            other_1 = _S1855;
        }
        bool _S1856;
        if(other_1 < k_76)
        {
            uint _S1857 = l_mark_0(par_4, other_1, &kernelContext_236);
            uint _S1858 = atomic_load_explicit((&kernelContext_236)->atom_0+_S1857, memory_order_relaxed);
            _S1856 = _S1858 == ((&kernelContext_236)->params_0->list_stamp_0);
        }
        else
        {
            _S1856 = false;
        }
        if(_S1856)
        {
            j_20 = j_20 + 1U;
            continue;
        }
        if((max((float4(*((&kernelContext_236)->loads_0+_S1855)) ).x, (float4(*((&kernelContext_236)->loads_0+_S1854.y)) ).x)) > (as_type<float>((_S1854.z))))
        {
            uint _S1859 = w_alive_0(&kernelContext_236);
            *((&kernelContext_236)->work_0+(_S1859 + _S1851)) = 0U;
            uint _S1860 = a_dirty_0(&kernelContext_236);
            uint _S1861 = body_of_0(_S1855, &kernelContext_236);
            uint _S1862 = atomic_fetch_or_explicit((&kernelContext_236)->atom_0+(_S1860 + _S1861), 1U, memory_order_relaxed);
            uint _S1863 = a_counters_1(&kernelContext_236);
            uint _S1864 = atomic_fetch_add_explicit((&kernelContext_236)->atom_0+_S1863, 1U, memory_order_relaxed);
            uint _S1865 = atomic_fetch_add_explicit((&kernelContext_236)->atom_0+(_S1863 + 1U), 1U, memory_order_relaxed);
            event_broken_0(_S1851, &kernelContext_236);
        }
        j_20 = j_20 + 1U;
    }
    return;
}

