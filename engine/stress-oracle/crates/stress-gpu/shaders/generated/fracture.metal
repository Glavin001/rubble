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
    uint il_passes_0;
    float lp_scale_0;
    uint cs_sweeps_0;
    uint cs_row_cap_0;
    float cs_c1_0;
    float cs_c2_0;
    uint cs_row_offset_0;
    float pad7_0;
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

IlStriker_0 il_striker_0(uint b_3, KernelContext_0 thread* kernelContext_9)
{
    uint _S9 = lp_body_0(b_3, kernelContext_9);
    float4 _S10 = float4(*(kernelContext_9->lp_0+(_S9 + 4U))) ;
    float4 _S11 = float4(*(kernelContext_9->lp_0+(_S9 + 5U))) ;
    thread IlStriker_0 s_0;
    (&s_0)->body_0 = (as_type<uint>((_S10.w)));
    float _S12 = _S10.y;
    (&s_0)->bound_0 = 1.00999999046325684f * _S12 + 0.00100000004749745f;
    (&s_0)->radius_0 = _S12;
    (&s_0)->mass_0 = _S11.z;
    (&s_0)->m_eff_0 = _S10.x;
    (&s_0)->modulus_0 = _S10.z;
    (&s_0)->launch_0 = (as_type<uint>(((float4(*(kernelContext_9->lp_0+(_S9 + 6U))) ).y)));
    return s_0;
}

uint a_first_0(uint shot_0, KernelContext_0 thread* kernelContext_10)
{
    uint _S13 = a_strike_0(kernelContext_10);
    return _S13 + 27U * kernelContext_10->params_0->body_count_0 + shot_0;
}

uint a_cs_0(uint b_4, KernelContext_0 thread* kernelContext_11)
{
    uint _S14 = a_first_0(0U, kernelContext_11);
    return _S14 + 64U + 12U * b_4;
}

uint il_entry_0(uint b_5, KernelContext_0 thread* kernelContext_12)
{
    uint _S15 = a_cs_0(b_5, kernelContext_12);
    uint _S16 = atomic_load_explicit(kernelContext_12->atom_0+(_S15 + 1U), memory_order_relaxed);
    return _S16 - 1U;
}

void il_path_0(uint b_6, float3 thread* contact_0, float3 thread* d_0, float thread* speed_0, KernelContext_0 thread* kernelContext_13)
{
    uint _S17 = lp_body_0(b_6, kernelContext_13);
    *contact_0 = (float4(*(kernelContext_13->lp_0+(_S17 + 2U))) ).xyz;
    float4 _S18 = float4(*(kernelContext_13->lp_0+(_S17 + 3U))) ;
    *d_0 = _S18.xyz;
    *speed_0 = _S18.w;
    return;
}

bool cs_struck_0(uint b_7, KernelContext_0 thread* kernelContext_14)
{
    bool _S19;
    if(b_7 < (kernelContext_14->params_0->body_count_0))
    {
        uint _S20 = a_cs_0(b_7, kernelContext_14);
        uint _S21 = atomic_load_explicit(kernelContext_14->atom_0+(_S20 + 1U), memory_order_relaxed);
        _S19 = _S21 != 0U;
    }
    else
    {
        _S19 = false;
    }
    return _S19;
}

bool il_struck_0(uint b_8, KernelContext_0 thread* kernelContext_15)
{
    bool _S22 = cs_struck_0(b_8, kernelContext_15);
    return _S22;
}

uint strike_at_0(uint b_9, KernelContext_0 thread* kernelContext_16)
{
    return 2U * kernelContext_16->params_0->body_count_0 + 9U * kernelContext_16->params_0->chunk_count_0 + 3U * b_9;
}

IlStriker_0 il_striker_1(uint b_10, KernelContext_0 thread* kernelContext_17)
{
    uint _S23 = strike_at_0(b_10, kernelContext_17);
    uint _S24 = 2U * (uint((float4(*(kernelContext_17->frames_0+_S23)) ).w) - 1U);
    float4 _S25 = float4(*(kernelContext_17->shots_0+_S24)) ;
    thread IlStriker_0 s_1;
    (&s_1)->body_0 = (as_type<uint>((_S25.x)));
    (&s_1)->bound_0 = _S25.y;
    (&s_1)->radius_0 = _S25.z;
    (&s_1)->mass_0 = _S25.w;
    float inv_mass_0 = (float4(*(kernelContext_17->bodies_0+7U * b_10)) ).w;
    float _S26;
    if(inv_mass_0 == 0.0f)
    {
        _S26 = (&s_1)->mass_0;
    }
    else
    {
        _S26 = (&s_1)->mass_0 / (1.0f + (&s_1)->mass_0 * inv_mass_0);
    }
    (&s_1)->m_eff_0 = _S26;
    uint _S27 = _S24 + 1U;
    (&s_1)->modulus_0 = (float4(*(kernelContext_17->shots_0+_S27)) ).x;
    (&s_1)->launch_0 = (as_type<uint>(((float4(*(kernelContext_17->shots_0+_S27)) ).y)));
    return s_1;
}

uint il_entry_1(uint b_11, KernelContext_0 thread* kernelContext_18)
{
    uint _S28 = a_strike_0(kernelContext_18);
    uint _S29 = atomic_load_explicit(kernelContext_18->atom_0+(_S28 + 9U * b_11 + 8U), memory_order_relaxed);
    return _S29;
}

void il_path_1(uint b_12, float3 thread* contact_1, float3 thread* d_1, float thread* speed_1, KernelContext_0 thread* kernelContext_19)
{
    uint _S30 = strike_at_0(b_12, kernelContext_19);
    float3 rel_0 = (float4(*(kernelContext_19->frames_0+(_S30 + 1U))) ).xyz;
    float _S31 = length(rel_0);
    *speed_1 = _S31;
    *d_1 = rel_0 / float3(max(_S31, 9.999999960041972e-13f)) ;
    *contact_1 = (float4(*(kernelContext_19->frames_0+(_S30 + 2U))) ).xyz;
    return;
}

bool struck_0(uint b_13, uint thread* shot_1, KernelContext_0 thread* kernelContext_20)
{
    uint _S32 = strike_at_0(b_13, kernelContext_20);
    uint k_0 = uint((float4(*(kernelContext_20->frames_0+_S32)) ).w);
    *shot_1 = k_0 - 1U;
    return k_0 != 0U;
}

bool il_struck_1(uint b_14, KernelContext_0 thread* kernelContext_21)
{
    thread uint shot_2;
    bool _S33;
    if((kernelContext_21->params_0->shot_count_0) > 0U)
    {
        _S33 = b_14 < (kernelContext_21->params_0->body_count_0);
    }
    else
    {
        _S33 = false;
    }
    if(_S33)
    {
        bool _S34 = struck_0(b_14, &shot_2, kernelContext_21);
        _S33 = _S34;
    }
    else
    {
        _S33 = false;
    }
    return _S33;
}

uint cs_rows_0(KernelContext_0 thread* kernelContext_22)
{
    return min(kernelContext_22->engine_counters_0[int(2)], kernelContext_22->params_0->cs_row_cap_0);
}

uint cs_row_0(uint r_0, KernelContext_0 thread* kernelContext_23)
{
    return 8U * (kernelContext_23->params_0->cs_row_offset_0 + r_0);
}

uint w_body_0()
{
    return 0U;
}

uint body_of_0(uint c_0, KernelContext_0 thread* kernelContext_24)
{
    return *(kernelContext_24->work_0+(w_body_0() + c_0));
}

uint w_alive_0(KernelContext_0 thread* kernelContext_25)
{
    return kernelContext_25->params_0->chunk_count_0;
}

uint w_label_0(KernelContext_0 thread* kernelContext_26)
{
    uint _S35 = w_alive_0(kernelContext_26);
    return _S35 + kernelContext_26->params_0->bond_count_0;
}

uint w_slot_0(KernelContext_0 thread* kernelContext_27)
{
    uint _S36 = w_label_0(kernelContext_27);
    return _S36 + kernelContext_27->params_0->chunk_count_0;
}

uint w_new_0(KernelContext_0 thread* kernelContext_28)
{
    uint _S37 = w_slot_0(kernelContext_28);
    return _S37 + kernelContext_28->params_0->chunk_count_0;
}

uint w_rank_0(KernelContext_0 thread* kernelContext_29)
{
    uint _S38 = w_new_0(kernelContext_29);
    return _S38 + kernelContext_29->params_0->chunk_count_0;
}

uint w_free_0(KernelContext_0 thread* kernelContext_30)
{
    uint _S39 = w_rank_0(kernelContext_30);
    return _S39 + kernelContext_30->params_0->chunk_count_0;
}

uint w_free_rank_0(KernelContext_0 thread* kernelContext_31)
{
    uint _S40 = w_free_0(kernelContext_31);
    return _S40 + kernelContext_31->params_0->body_count_0;
}

uint w_free_list_0(KernelContext_0 thread* kernelContext_32)
{
    uint _S41 = w_free_rank_0(kernelContext_32);
    return _S41 + kernelContext_32->params_0->body_count_0;
}

uint w_pulses_0(KernelContext_0 thread* kernelContext_33)
{
    uint _S42 = w_free_list_0(kernelContext_33);
    return _S42 + kernelContext_33->params_0->body_count_0 + 2050U;
}

uint w_marks_0(KernelContext_0 thread* kernelContext_34)
{
    uint _S43 = w_pulses_0(kernelContext_34);
    return _S43 + 4U * kernelContext_34->params_0->chunk_count_0;
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
    thread Quat_0 r_1;
    (&r_1)->x_0 = q_0.x;
    (&r_1)->y_0 = q_0.y;
    (&r_1)->z_0 = q_0.z;
    (&r_1)->w_0 = q_0.w;
    return r_1;
}

float3 rotate_0(const Quat_0 thread* q_1, float3 v_0)
{
    float3 qv_0 = float3(q_1->x_0, q_1->y_0, q_1->z_0);
    float3 t_0 = cross(qv_0, v_0) * float3(2.0f) ;
    return v_0 + t_0 * float3(q_1->w_0)  + cross(qv_0, t_0);
}

Quat_0 conj_0(const Quat_0 thread* q_2)
{
    thread Quat_0 c_1;
    (&c_1)->w_0 = q_2->w_0;
    (&c_1)->x_0 = - q_2->x_0;
    (&c_1)->y_0 = - q_2->y_0;
    (&c_1)->z_0 = - q_2->z_0;
    return c_1;
}

float cs_angular_0(uint b_15, float3 r_2, float3 u_0, KernelContext_0 thread* kernelContext_35)
{
    uint at_0 = 7U * b_15;
    thread Quat_0 _S44 = quat_of_0(float4(*(kernelContext_35->bodies_0+(at_0 + 1U))) );
    Quat_0 _S45 = conj_0(&_S44);
    float3 _S46 = cross(r_2, u_0);
    thread Quat_0 _S47 = _S45;
    float3 _S48 = rotate_0(&_S47, _S46);
    return dot(_S48 * (float4(*(kernelContext_35->bodies_0+(at_0 + 2U))) ).xyz, _S48);
}

bool cs_side_0(uint r_3, uint side_0, uint thread* k_1, uint thread* s_2, uint thread* ob_0, uint thread* co_0, float3 thread* p_0, float3 thread* u_1, float thread* v_1, float thread* m_eff_1, float thread* t_in_0, KernelContext_0 thread* kernelContext_36)
{
    *t_in_0 = 0.0f;
    *k_1 = 4294967295U;
    *s_2 = 4294967295U;
    *ob_0 = 4294967295U;
    *co_0 = 4294967295U;
    float3 _S49 = float3(0.0f) ;
    *p_0 = _S49;
    *u_1 = _S49;
    *v_1 = 0.0f;
    *m_eff_1 = 0.0f;
    uint _S50 = cs_row_0(r_3, kernelContext_36);
    uint4 ids_1 = (as_type<uint4>((float4(*(kernelContext_36->engine_contacts_0+_S50)) )));
    if((ids_1.w) == 0U)
    {
        return false;
    }
    bool _S51 = side_0 == 0U;
    uint cm_0;
    if(_S51)
    {
        cm_0 = ids_1.x;
    }
    else
    {
        cm_0 = ids_1.y;
    }
    uint _S52;
    if(_S51)
    {
        _S52 = ids_1.y;
    }
    else
    {
        _S52 = ids_1.x;
    }
    *co_0 = _S52;
    uint _S53 = (uint4(*(kernelContext_36->collider_info_0+2U * cm_0)) ).x;
    *k_1 = _S53;
    bool _S54;
    if(_S53 == 4294967295U)
    {
        _S54 = true;
    }
    else
    {
        _S54 = (*k_1) >= (kernelContext_36->params_0->chunk_count_0);
    }
    if(_S54)
    {
        _S54 = true;
    }
    else
    {
        _S54 = ((uint4(*(kernelContext_36->collider_info_0+2U * *co_0)) ).z) != 2U;
    }
    if(_S54)
    {
        return false;
    }
    uint _S55 = body_of_0(*k_1, kernelContext_36);
    *s_2 = _S55;
    uint _S56 = (as_type<uint>(((float4(*(kernelContext_36->colliders_0+4U * *co_0)) ).x)));
    *ob_0 = _S56;
    if(_S56 == (*s_2))
    {
        _S54 = true;
    }
    else
    {
        _S54 = (*ob_0) >= (kernelContext_36->params_0->max_bodies_0);
    }
    if(_S54)
    {
        return false;
    }
    uint launch_1 = (uint4(*(kernelContext_36->collider_info_0+(2U * *co_0 + 1U))) ).w;
    if(launch_1 != 0U)
    {
        uint _S57 = w_marks_0(kernelContext_36);
        _S54 = (*(kernelContext_36->work_0+(_S57 + *k_1))) == launch_1;
    }
    else
    {
        _S54 = false;
    }
    if(_S54)
    {
        return false;
    }
    uint bs_0 = 7U * *s_2;
    uint bo_0 = 7U * *ob_0;
    float inv_o_0 = (float4(*(kernelContext_36->bodies_0+bo_0)) ).w;
    if(!(inv_o_0 > 0.0f))
    {
        return false;
    }
    float3 n_1 = (float4(*(kernelContext_36->engine_contacts_0+(_S50 + 1U))) ).xyz;
    float3 _S58;
    if(_S51)
    {
        _S58 = - n_1;
    }
    else
    {
        _S58 = n_1;
    }
    *u_1 = _S58;
    Quat_0 _S59 = quat_of_0(float4(*(kernelContext_36->bodies_0+(bs_0 + 1U))) );
    if(_S51)
    {
        _S58 = (float4(*(kernelContext_36->engine_contacts_0+(_S50 + 2U))) ).xyz;
    }
    else
    {
        _S58 = (float4(*(kernelContext_36->engine_contacts_0+(_S50 + 3U))) ).xyz;
    }
    thread Quat_0 _S60 = _S59;
    float3 _S61 = rotate_0(&_S60, _S58);
    Quat_0 _S62 = quat_of_0(float4(*(kernelContext_36->bodies_0+(bo_0 + 1U))) );
    if(_S51)
    {
        _S58 = (float4(*(kernelContext_36->engine_contacts_0+(_S50 + 3U))) ).xyz;
    }
    else
    {
        _S58 = (float4(*(kernelContext_36->engine_contacts_0+(_S50 + 2U))) ).xyz;
    }
    thread Quat_0 _S63 = _S62;
    float3 _S64 = rotate_0(&_S63, _S58);
    float3 _S65 = (float4(*(kernelContext_36->bodies_0+bs_0)) ).xyz + _S61;
    *p_0 = _S65;
    float gap_0 = dot(_S65 - ((float4(*(kernelContext_36->bodies_0+bo_0)) ).xyz + _S64), *u_1);
    float _S66 = dot((float4(*(kernelContext_36->bodies_0+(bo_0 + 3U))) ).xyz + cross((float4(*(kernelContext_36->bodies_0+(bo_0 + 4U))) ).xyz, _S64) - ((float4(*(kernelContext_36->bodies_0+(bs_0 + 3U))) ).xyz + cross((float4(*(kernelContext_36->bodies_0+(bs_0 + 4U))) ).xyz, _S61)), *u_1);
    *v_1 = _S66;
    if(!(_S66 > (kernelContext_36->params_0->strike_speed_0)))
    {
        return false;
    }
    float _S67 = (float4(*(kernelContext_36->bodies_0+bs_0)) ).w;
    float _S68 = cs_angular_0(*s_2, _S61, *u_1, kernelContext_36);
    float _S69 = _S67 + _S68 + inv_o_0;
    float _S70 = cs_angular_0(*ob_0, _S64, *u_1, kernelContext_36);
    float w_1 = _S69 + _S70;
    if(!(w_1 > 0.0f))
    {
        return false;
    }
    *m_eff_1 = 1.0f / w_1;
    *t_in_0 = clamp(gap_0 / *v_1, 0.0f, kernelContext_36->params_0->dt_0);
    return true;
}

uint3 cs_identity_0(uint r_4, uint side_1, KernelContext_0 thread* kernelContext_37)
{
    uint _S71 = cs_row_0(r_4, kernelContext_37);
    uint4 ids_2 = (as_type<uint4>((float4(*(kernelContext_37->engine_contacts_0+_S71)) )));
    uint3 _S72;
    if(side_1 == 0U)
    {
        _S72 = uint3(ids_2.x, ids_2.y, 2U * ids_2.z);
    }
    else
    {
        _S72 = uint3(ids_2.y, ids_2.x, 2U * ids_2.z + 1U);
    }
    return _S72;
}

void cs_pick_0(uint r_5, KernelContext_0 thread* kernelContext_38)
{
    uint _S73 = cs_rows_0(kernelContext_38);
    if(r_5 >= _S73)
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
        thread uint k_2;
        thread uint s_3;
        thread uint ob_1;
        thread uint co_1;
        thread float3 p_1;
        thread float3 u_2;
        thread float v_2;
        thread float m_eff_2;
        thread float t_in_1;
        bool _S74 = cs_side_0(r_5, side_2, &k_2, &s_3, &ob_1, &co_1, &p_1, &u_2, &v_2, &m_eff_2, &t_in_1, kernelContext_38);
        bool _S75;
        if(!_S74)
        {
            _S75 = true;
        }
        else
        {
            uint _S76 = (as_type<uint>((m_eff_2 * v_2)));
            uint _S77 = a_cs_0(s_3, kernelContext_38);
            uint _S78 = atomic_load_explicit(kernelContext_38->atom_0+_S77, memory_order_relaxed);
            _S75 = _S76 != _S78;
        }
        if(_S75)
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint3 _S79 = cs_identity_0(r_5, side_2, kernelContext_38);
        uint _S80 = a_cs_0(s_3, kernelContext_38);
        uint _S81 = atomic_load_explicit(kernelContext_38->atom_0+(_S80 + 2U), memory_order_relaxed);
        if((~_S81) != (_S79.x))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint _S82 = atomic_load_explicit(kernelContext_38->atom_0+(_S80 + 9U), memory_order_relaxed);
        if((~_S82) != (_S79.y))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint _S83 = atomic_load_explicit(kernelContext_38->atom_0+(_S80 + 10U), memory_order_relaxed);
        if((~_S83) != (_S79.z))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        atomic_store_explicit(kernelContext_38->atom_0+(_S80 + 11U), 2U * r_5 + side_2 + 1U, memory_order_relaxed);
        side_2 = side_2 + 1U;
    }
    return;
}

void cs_pick_1(uint r_6, KernelContext_0 thread* kernelContext_39)
{
    uint _S84 = cs_rows_0(kernelContext_39);
    if(r_6 >= _S84)
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
        thread uint k_3;
        thread uint s_4;
        thread uint ob_2;
        thread uint co_2;
        thread float3 p_2;
        thread float3 u_3;
        thread float v_3;
        thread float m_eff_3;
        thread float t_in_2;
        bool _S85 = cs_side_0(r_6, side_3, &k_3, &s_4, &ob_2, &co_2, &p_2, &u_3, &v_3, &m_eff_3, &t_in_2, kernelContext_39);
        bool _S86;
        if(!_S85)
        {
            _S86 = true;
        }
        else
        {
            uint _S87 = (as_type<uint>((m_eff_3 * v_3)));
            uint _S88 = a_cs_0(s_4, kernelContext_39);
            uint _S89 = atomic_load_explicit(kernelContext_39->atom_0+_S88, memory_order_relaxed);
            _S86 = _S87 != _S89;
        }
        if(_S86)
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint3 _S90 = cs_identity_0(r_6, side_3, kernelContext_39);
        uint _S91 = a_cs_0(s_4, kernelContext_39);
        uint _S92 = atomic_load_explicit(kernelContext_39->atom_0+(_S91 + 2U), memory_order_relaxed);
        if((~_S92) != (_S90.x))
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint _S93 = atomic_load_explicit(kernelContext_39->atom_0+(_S91 + 9U), memory_order_relaxed);
        if((~_S93) != (_S90.y))
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint _S94 = atomic_fetch_max_explicit(kernelContext_39->atom_0+(_S91 + 10U), ~_S90.z, memory_order_relaxed);
        side_3 = side_3 + 1U;
    }
    return;
}

void cs_pick_2(uint r_7, KernelContext_0 thread* kernelContext_40)
{
    uint _S95 = cs_rows_0(kernelContext_40);
    if(r_7 >= _S95)
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
        thread uint k_4;
        thread uint s_5;
        thread uint ob_3;
        thread uint co_3;
        thread float3 p_3;
        thread float3 u_4;
        thread float v_4;
        thread float m_eff_4;
        thread float t_in_3;
        bool _S96 = cs_side_0(r_7, side_4, &k_4, &s_5, &ob_3, &co_3, &p_3, &u_4, &v_4, &m_eff_4, &t_in_3, kernelContext_40);
        bool _S97;
        if(!_S96)
        {
            _S97 = true;
        }
        else
        {
            uint _S98 = (as_type<uint>((m_eff_4 * v_4)));
            uint _S99 = a_cs_0(s_5, kernelContext_40);
            uint _S100 = atomic_load_explicit(kernelContext_40->atom_0+_S99, memory_order_relaxed);
            _S97 = _S98 != _S100;
        }
        if(_S97)
        {
            side_4 = side_4 + 1U;
            continue;
        }
        uint3 _S101 = cs_identity_0(r_7, side_4, kernelContext_40);
        uint _S102 = a_cs_0(s_5, kernelContext_40);
        uint _S103 = atomic_load_explicit(kernelContext_40->atom_0+(_S102 + 2U), memory_order_relaxed);
        if((~_S103) != (_S101.x))
        {
            side_4 = side_4 + 1U;
            continue;
        }
        uint _S104 = atomic_fetch_max_explicit(kernelContext_40->atom_0+(_S102 + 9U), ~_S101.y, memory_order_relaxed);
        side_4 = side_4 + 1U;
    }
    return;
}

void cs_pick_3(uint r_8, KernelContext_0 thread* kernelContext_41)
{
    uint _S105 = cs_rows_0(kernelContext_41);
    if(r_8 >= _S105)
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
        thread uint k_5;
        thread uint s_6;
        thread uint ob_4;
        thread uint co_4;
        thread float3 p_4;
        thread float3 u_5;
        thread float v_5;
        thread float m_eff_5;
        thread float t_in_4;
        bool _S106 = cs_side_0(r_8, side_5, &k_5, &s_6, &ob_4, &co_4, &p_4, &u_5, &v_5, &m_eff_5, &t_in_4, kernelContext_41);
        bool _S107;
        if(!_S106)
        {
            _S107 = true;
        }
        else
        {
            uint _S108 = (as_type<uint>((m_eff_5 * v_5)));
            uint _S109 = a_cs_0(s_6, kernelContext_41);
            uint _S110 = atomic_load_explicit(kernelContext_41->atom_0+_S109, memory_order_relaxed);
            _S107 = _S108 != _S110;
        }
        if(_S107)
        {
            side_5 = side_5 + 1U;
            continue;
        }
        uint3 _S111 = cs_identity_0(r_8, side_5, kernelContext_41);
        uint _S112 = a_cs_0(s_6, kernelContext_41);
        uint _S113 = atomic_fetch_max_explicit(kernelContext_41->atom_0+(_S112 + 2U), ~_S111.x, memory_order_relaxed);
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

Quat_0 quat_mul_0(const Quat_0 thread* a_0, const Quat_0 thread* o_0)
{
    thread Quat_0 r_9;
    (&r_9)->w_0 = a_0->w_0 * o_0->w_0 - a_0->x_0 * o_0->x_0 - a_0->y_0 * o_0->y_0 - a_0->z_0 * o_0->z_0;
    (&r_9)->x_0 = a_0->w_0 * o_0->x_0 + a_0->x_0 * o_0->w_0 + a_0->y_0 * o_0->z_0 - a_0->z_0 * o_0->y_0;
    (&r_9)->y_0 = a_0->w_0 * o_0->y_0 - a_0->x_0 * o_0->z_0 + a_0->y_0 * o_0->w_0 + a_0->z_0 * o_0->x_0;
    (&r_9)->z_0 = a_0->w_0 * o_0->z_0 + a_0->x_0 * o_0->y_0 - a_0->y_0 * o_0->x_0 + a_0->z_0 * o_0->w_0;
    return r_9;
}

Quat_0 normalized_0(const Quat_0 thread* q_3)
{
    float n_2 = sqrt(q_3->w_0 * q_3->w_0 + q_3->x_0 * q_3->x_0 + q_3->y_0 * q_3->y_0 + q_3->z_0 * q_3->z_0);
    thread Quat_0 r_10;
    (&r_10)->w_0 = q_3->w_0 / n_2;
    (&r_10)->x_0 = q_3->x_0 / n_2;
    (&r_10)->y_0 = q_3->y_0 / n_2;
    (&r_10)->z_0 = q_3->z_0 / n_2;
    return r_10;
}

Quat_0 rot_from_rubble_0(const Quat_0 thread* q_4)
{
    thread Quat_0 axes_0;
    (&axes_0)->w_0 = 0.70710676908493042f;
    (&axes_0)->x_0 = 0.70710676908493042f;
    (&axes_0)->y_0 = 0.0f;
    (&axes_0)->z_0 = 0.0f;
    thread Quat_0 _S114 = axes_0;
    Quat_0 _S115 = quat_mul_0(&_S114, q_4);
    thread Quat_0 _S116 = _S115;
    Quat_0 _S117 = normalized_0(&_S116);
    return _S117;
}

float3 from_rubble_0(float3 v_6)
{
    return float3(v_6.x, - v_6.z, v_6.y);
}

void structure_pose_0(uint slot_0, uint b_16, Quat_0 thread* q_5, float3 thread* origin_0, KernelContext_0 thread* kernelContext_42)
{
    uint _S118 = 2U * slot_0;
    thread Quat_0 _S119 = quat_of_0(float4(*(kernelContext_42->frames_0+(_S118 + 1U))) );
    Quat_0 _S120 = conj_0(&_S119);
    float3 com_0 = (float4(*(kernelContext_42->frames_0+_S118)) ).xyz;
    thread Quat_0 _S121 = quat_of_0(float4(*(kernelContext_42->bodies_0+(b_16 + 1U))) );
    Quat_0 _S122 = rot_from_rubble_0(&_S121);
    thread Quat_0 _S123 = _S122;
    thread Quat_0 _S124 = _S120;
    Quat_0 _S125 = quat_mul_0(&_S123, &_S124);
    thread Quat_0 _S126 = _S125;
    Quat_0 _S127 = normalized_0(&_S126);
    *q_5 = _S127;
    float3 _S128 = from_rubble_0((float4(*(kernelContext_42->bodies_0+b_16)) ).xyz);
    thread Quat_0 _S129 = _S127;
    float3 _S130 = rotate_0(&_S129, com_0);
    *origin_0 = _S128 - _S130;
    return;
}

float3 chunk_box_world_0(uint k_6, Quat_0 thread* q_6, KernelContext_0 thread* kernelContext_43)
{
    uint _S131 = body_of_0(k_6, kernelContext_43);
    thread float3 origin_1;
    structure_pose_0(_S131, 7U * _S131, q_6, &origin_1, kernelContext_43);
    float3 _S132 = (float4(*(kernelContext_43->geometry_0+3U * k_6)) ).xyz;
    thread Quat_0 _S133 = *q_6;
    float3 _S134 = rotate_0(&_S133, _S132);
    return origin_1 + _S134;
}

bool il_crossed_0(uint k_7, float3 a_1, float3 d_2, float length_0, float r_11, KernelContext_0 thread* kernelContext_44)
{
    thread Quat_0 q_7;
    float3 _S135 = chunk_box_world_0(k_7, &q_7, kernelContext_44);
    uint _S136 = 3U * k_7;
    Quat_0 _S137 = quat_of_0(float4(*(kernelContext_44->geometry_0+(_S136 + 2U))) );
    thread Quat_0 _S138 = q_7;
    thread Quat_0 _S139 = _S137;
    Quat_0 _S140 = quat_mul_0(&_S138, &_S139);
    thread Quat_0 _S141 = _S140;
    Quat_0 _S142 = normalized_0(&_S141);
    float3 half_0 = (float4(*(kernelContext_44->geometry_0+(_S136 + 1U))) ).xyz;
    thread Quat_0 _S143 = _S142;
    Quat_0 _S144 = conj_0(&_S143);
    float3 _S145 = a_1 - _S135;
    thread Quat_0 _S146 = _S144;
    float3 _S147 = rotate_0(&_S146, _S145);
    thread Quat_0 _S148 = _S144;
    float3 _S149 = rotate_0(&_S148, d_2);
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
        float _S150 = hi_0 - lo_0;
        float m1_0 = lo_0 + _S150 / 3.0f;
        float m2_0 = hi_0 - _S150 / 3.0f;
        float3 _S151 = float3(0.0f) ;
        float3 e1_0 = max(abs(_S147 + _S149 * float3(m1_0) ) - half_0, _S151);
        float3 e2_0 = max(abs(_S147 + _S149 * float3(m2_0) ) - half_0, _S151);
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
    float3 e_0 = max(abs(_S147 + _S149 * float3((0.5f * (lo_0 + hi_0))) ) - half_0, float3(0.0f) );
    return (dot(e_0, e_0)) < (0.99980002641677856f * r_11 * r_11);
}

IlPlace_0 il_place_0(uint k_8, uint entry_1, float3 contact_2, float3 d_3, float r_12, float h_0, KernelContext_0 thread* kernelContext_45)
{
    uint _S152 = body_of_0(k_8, kernelContext_45);
    thread Quat_0 q_8;
    thread float3 origin_2;
    structure_pose_0(_S152, 7U * _S152, &q_8, &origin_2, kernelContext_45);
    float3 _S153 = (float4(*(kernelContext_45->chunks_0+4U * k_8)) ).xyz;
    thread Quat_0 _S154 = q_8;
    float3 _S155 = rotate_0(&_S154, _S153);
    float3 rel_1 = origin_2 + _S155 - contact_2;
    thread IlPlace_0 p_5;
    float _S156 = dot(rel_1, d_3);
    (&p_5)->along_0 = _S156;
    (&p_5)->radial_0 = length(rel_1 - d_3 * float3(_S156) );
    (&p_5)->entry_0 = k_8 == entry_1;
    bool _S157 = il_crossed_0(k_8, contact_2 - d_3 * float3(r_12) , d_3, h_0 + 2.0f * r_12, r_12, kernelContext_45);
    (&p_5)->crossed_0 = _S157;
    return p_5;
}

bool il_plug_0(uint outcome_0, float r_13, float h_1, const IlPlace_0 thread* p_6)
{
    bool _S158;
    if((outcome_0 & 8U) != 0U)
    {
        if(p_6->entry_0)
        {
            _S158 = true;
        }
        else
        {
            _S158 = p_6->crossed_0;
        }
        if(_S158)
        {
            _S158 = true;
        }
        else
        {
            float _S159 = p_6->along_0;
            if((p_6->along_0) >= 0.0f)
            {
                _S158 = _S159 <= h_1;
            }
            else
            {
                _S158 = false;
            }
            if(_S158)
            {
                _S158 = (p_6->radial_0) <= (r_13 + _S159);
            }
            else
            {
                _S158 = false;
            }
        }
    }
    else
    {
        _S158 = false;
    }
    return _S158;
}

bool il_scab_0(uint outcome_1, float r_14, float h_2, float scab_0, const IlPlace_0 thread* p_7)
{
    bool _S160;
    if((outcome_1 & 4U) != 0U)
    {
        _S160 = (p_7->along_0) >= (h_2 - scab_0);
    }
    else
    {
        _S160 = false;
    }
    if(_S160)
    {
        _S160 = (p_7->along_0) <= h_2;
    }
    else
    {
        _S160 = false;
    }
    if(_S160)
    {
        _S160 = (p_7->radial_0) <= (r_14 + h_2);
    }
    else
    {
        _S160 = false;
    }
    return _S160;
}

bool il_crater_0(uint outcome_2, float r_15, float depth_0, float a_2, const IlPlace_0 thread* p_8)
{
    bool _S161;
    if((outcome_2 & 2U) != 0U)
    {
        _S161 = (p_8->along_0) >= 0.0f;
    }
    else
    {
        _S161 = false;
    }
    if(_S161)
    {
        _S161 = (p_8->along_0) <= depth_0;
    }
    else
    {
        _S161 = false;
    }
    if(_S161)
    {
        _S161 = (p_8->radial_0) <= (max(r_15, a_2));
    }
    else
    {
        _S161 = false;
    }
    return _S161;
}

uint a_counters_1(KernelContext_0 thread* kernelContext_46)
{
    uint _S162 = a_best_root_0(kernelContext_46);
    return _S162 + kernelContext_46->params_0->body_count_0;
}

void il_breaks_0(uint i_0, KernelContext_0 thread* kernelContext_47)
{
    bool cut_0;
    if(i_0 >= (kernelContext_47->params_0->bond_count_0))
    {
        cut_0 = true;
    }
    else
    {
        uint _S163 = w_alive_0(kernelContext_47);
        cut_0 = (*(kernelContext_47->work_0+(_S163 + i_0))) == 0U;
    }
    if(cut_0)
    {
        return;
    }
    uint4 _S164 = uint4(*(kernelContext_47->bonds_0+i_0)) ;
    uint _S165 = _S164.x;
    uint _S166 = body_of_0(_S165, kernelContext_47);
    bool _S167 = il_struck_0(_S166, kernelContext_47);
    if(!_S167)
    {
        return;
    }
    uint _S168 = lp_body_0(_S166, kernelContext_47);
    float4 _S169 = float4(*(kernelContext_47->lp_0+_S168)) ;
    float4 _S170 = float4(*(kernelContext_47->lp_0+(_S168 + 1U))) ;
    uint outcome_3 = (as_type<uint>((_S169.w)));
    if((outcome_3 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_3;
    thread float3 d_4;
    thread float speed_2;
    il_path_0(_S166, &contact_3, &d_4, &speed_2, kernelContext_47);
    uint _S171 = il_entry_0(_S166, kernelContext_47);
    IlStriker_0 _S172 = il_striker_0(_S166, kernelContext_47);
    float _S173 = max(_S172.radius_0, 0.00100000004749745f);
    float h_3 = _S169.x;
    IlPlace_0 _S174 = il_place_0(_S165, _S171, contact_3, d_4, _S173, h_3, kernelContext_47);
    IlPlace_0 _S175 = il_place_0(_S164.y, _S171, contact_3, d_4, _S173, h_3, kernelContext_47);
    thread IlPlace_0 _S176 = _S174;
    bool _S177 = il_plug_0(outcome_3, _S173, h_3, &_S176);
    thread IlPlace_0 _S178 = _S175;
    bool _S179 = il_plug_0(outcome_3, _S173, h_3, &_S178);
    if(_S177 != _S179)
    {
        cut_0 = true;
    }
    else
    {
        float _S180 = _S170.y;
        thread IlPlace_0 _S181 = _S174;
        bool _S182 = il_scab_0(outcome_3, _S173, h_3, _S180, &_S181);
        thread IlPlace_0 _S183 = _S175;
        bool _S184 = il_scab_0(outcome_3, _S173, h_3, _S180, &_S183);
        cut_0 = _S182 != _S184;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S185 = _S170.x;
        float _S186 = _S170.z;
        thread IlPlace_0 _S187 = _S174;
        bool _S188 = il_crater_0(outcome_3, _S173, _S185, _S186, &_S187);
        cut_0 = _S188;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S189 = _S170.x;
        float _S190 = _S170.z;
        thread IlPlace_0 _S191 = _S175;
        bool _S192 = il_crater_0(outcome_3, _S173, _S189, _S190, &_S191);
        cut_0 = _S192;
    }
    if(cut_0)
    {
        uint _S193 = w_alive_0(kernelContext_47);
        *(kernelContext_47->work_0+(_S193 + i_0)) = 0U;
        uint _S194 = a_dirty_0(kernelContext_47);
        uint _S195 = atomic_fetch_or_explicit(kernelContext_47->atom_0+(_S194 + _S166), 1U, memory_order_relaxed);
        uint _S196 = a_counters_1(kernelContext_47);
        uint _S197 = atomic_fetch_add_explicit(kernelContext_47->atom_0+_S196, 1U, memory_order_relaxed);
        uint _S198 = atomic_fetch_add_explicit(kernelContext_47->atom_0+(_S196 + 1U), 1U, memory_order_relaxed);
        uint _S199 = atomic_fetch_add_explicit(kernelContext_47->atom_0+(_S196 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

uint w_carry_0(KernelContext_0 thread* kernelContext_48)
{
    uint _S200 = w_marks_0(kernelContext_48);
    return _S200 + kernelContext_48->params_0->chunk_count_0;
}

float3 to_rubble_0(float3 v_7)
{
    return float3(v_7.x, v_7.z, - v_7.y);
}

uint a_il_0(uint b_17, KernelContext_0 thread* kernelContext_49)
{
    uint _S201 = a_strike_0(kernelContext_49);
    return _S201 + 23U * kernelContext_49->params_0->body_count_0 + 4U * b_17;
}

float read64_0(uint at_1, float scale_0, KernelContext_0 thread* kernelContext_50)
{
    uint _S202 = atomic_load_explicit(kernelContext_50->atom_0+(at_1 + 1U), memory_order_relaxed);
    int hi_1 = (as_type<int>((_S202)));
    uint lo_1 = atomic_load_explicit(kernelContext_50->atom_0+at_1, memory_order_relaxed);
    return (float(hi_1) * 4.294967296e+09f + float(lo_1)) / scale_0;
}

void il_carry_0(uint k_9, KernelContext_0 thread* kernelContext_51)
{
    if(k_9 >= (kernelContext_51->params_0->chunk_count_0))
    {
        return;
    }
    uint _S203 = w_carry_0(kernelContext_51);
    uint _S204 = _S203 + k_9;
    uint device* _S205 = kernelContext_51->work_0+_S204;
    uint c_2 = *_S205;
    if((*_S205) == 0U)
    {
        return;
    }
    *(kernelContext_51->work_0+_S204) = 0U;
    uint b0_0 = c_2 - 1U;
    bool _S206 = il_struck_0(b0_0, kernelContext_51);
    if(!_S206)
    {
        return;
    }
    thread float3 contact_4;
    thread float3 d_5;
    thread float speed_3;
    il_path_0(b0_0, &contact_4, &d_5, &speed_3, kernelContext_51);
    float3 dr_0 = to_rubble_0(d_5);
    IlStriker_0 _S207 = il_striker_0(b0_0, kernelContext_51);
    uint _S208 = a_il_0(b0_0, kernelContext_51);
    float _S209 = read64_0(_S208 + 2U, kernelContext_51->params_0->scale_m_0, kernelContext_51);
    float3 v_wall_0 = (float4(*(kernelContext_51->bodies_0+(7U * b0_0 + 3U))) ).xyz;
    uint _S210 = 7U * _S207.body_0;
    float3 v_shot_0 = (float4(*(kernelContext_51->bodies_0+(_S210 + 3U))) ).xyz;
    float closing_0 = dot(v_shot_0 - v_wall_0, dr_0);
    float v_r_0 = _S207.mass_0 * closing_0 / (_S207.mass_0 + _S209);
    float3 _S211 = v_shot_0 + dr_0 * float3((v_r_0 - closing_0)) ;
    float3 _S212 = v_wall_0 + dr_0 * float3(v_r_0) ;
    float _S213 = il_t_in_0(b0_0, kernelContext_51);
    float3 _S214 = (float4(*(kernelContext_51->bodies_0+_S210)) ).xyz + dr_0 * float3(((speed_3 - v_r_0) * clamp(_S213, 0.0f, kernelContext_51->params_0->dt_0))) ;
    uint _S215 = body_of_0(k_9, kernelContext_51);
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
        uint s_at_0 = 7U * (_S207.body_0 + region_0 * kernelContext_51->params_0->max_bodies_0);
        (*(kernelContext_51->bodies_0+s_at_0)).xyz = _S214;
        (*(kernelContext_51->bodies_0+(s_at_0 + 3U))).xyz = _S211;
        (*(kernelContext_51->bodies_0+(7U * (_S215 + region_0 * kernelContext_51->params_0->max_bodies_0) + 3U))).xyz = _S212;
        region_0 = region_0 + 1U;
    }
    return;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

float2 il_interval_0(uint k_10, float3 p_9, float3 d_6, float pad_0, KernelContext_0 thread* kernelContext_52)
{
    thread Quat_0 q_9;
    float3 _S216 = chunk_box_world_0(k_10, &q_9, kernelContext_52);
    uint _S217 = 3U * k_10;
    Quat_0 _S218 = quat_of_0(float4(*(kernelContext_52->geometry_0+(_S217 + 2U))) );
    thread Quat_0 _S219 = q_9;
    thread Quat_0 _S220 = _S218;
    Quat_0 _S221 = quat_mul_0(&_S219, &_S220);
    thread Quat_0 _S222 = _S221;
    Quat_0 _S223 = normalized_0(&_S222);
    float3 _S224 = (float4(*(kernelContext_52->geometry_0+(_S217 + 1U))) ).xyz;
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
        thread float3 e_1 = float3(0.0f) ;
        e_1[ax_0] = 1.0f;
        thread Quat_0 _S225 = _S223;
        float3 _S226 = rotate_0(&_S225, e_1);
        float o_1 = dot(p_9 - _S216, _S226);
        float dv_0 = dot(d_6, _S226);
        float ext_1 = _S224[ax_0] + pad_0;
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
        float _S227 = min(t1_0, max(ta_0, tb_0));
        t0_0 = max(t0_0, min(ta_0, tb_0));
        t1_0 = _S227;
        ax_0 = ax_0 + 1U;
    }
    return float2(t0_0, t1_0);
}

void add64_0(uint at_2, float value_0, float scale_1, KernelContext_0 thread* kernelContext_53)
{
    float x_1 = round(value_0 * scale_1);
    float x_2;
    if(!((abs(x_1)) < 1.073741824e+09f))
    {
        uint _S228 = a_counters_1(kernelContext_53);
        uint _S229 = atomic_fetch_or_explicit(kernelContext_53->atom_0+(_S228 + 3U), 2U, memory_order_relaxed);
        x_2 = 0.0f;
    }
    else
    {
        x_2 = x_1;
    }
    int v_8 = int(x_2);
    uint lo_2 = (as_type<uint>((v_8)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_53->atom_0+at_2, lo_2, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_2) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S230;
    if(v_8 < int(0))
    {
        _S230 = 4294967295U;
    }
    else
    {
        _S230 = 0U;
    }
    uint hi_2 = _S230 + carry_0;
    if(hi_2 != 0U)
    {
        uint _S231 = atomic_fetch_add_explicit(kernelContext_53->atom_0+(at_2 + 1U), hi_2, memory_order_relaxed);
    }
    return;
}

void il_mark_0(uint k_11, KernelContext_0 thread* kernelContext_54)
{
    if(k_11 >= (kernelContext_54->params_0->chunk_count_0))
    {
        return;
    }
    uint _S232 = body_of_0(k_11, kernelContext_54);
    bool _S233 = il_struck_0(_S232, kernelContext_54);
    if(!_S233)
    {
        return;
    }
    uint _S234 = lp_body_0(_S232, kernelContext_54);
    float4 _S235 = float4(*(kernelContext_54->lp_0+_S234)) ;
    float4 _S236 = float4(*(kernelContext_54->lp_0+(_S234 + 1U))) ;
    uint outcome_4 = (as_type<uint>((_S235.w)));
    if((outcome_4 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_5;
    thread float3 d_7;
    thread float speed_4;
    il_path_0(_S232, &contact_5, &d_7, &speed_4, kernelContext_54);
    uint _S237 = il_entry_0(_S232, kernelContext_54);
    IlStriker_0 _S238 = il_striker_0(_S232, kernelContext_54);
    float _S239 = max(_S238.radius_0, 0.00100000004749745f);
    float _S240 = _S235.x;
    IlPlace_0 _S241 = il_place_0(k_11, _S237, contact_5, d_7, _S239, _S240, kernelContext_54);
    thread IlPlace_0 _S242 = _S241;
    bool _S243 = il_plug_0(outcome_4, _S239, _S240, &_S242);
    if((_S238.launch_0) != 0U)
    {
        float2 _S244 = il_interval_0(k_11, contact_5 - d_7 * float3(_S238.bound_0) , d_7, _S238.bound_0, kernelContext_54);
        float _S245 = _S244.x;
        float _S246 = _S244.y;
        bool path_0;
        if(_S245 <= _S246)
        {
            path_0 = _S245 <= (_S240 + 2.0f * _S238.bound_0);
        }
        else
        {
            path_0 = false;
        }
        if(path_0)
        {
            path_0 = _S246 >= 0.0f;
        }
        else
        {
            path_0 = false;
        }
        if(_S243)
        {
            path_0 = true;
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            float _S247 = _S236.y;
            thread IlPlace_0 _S248 = _S241;
            bool _S249 = il_scab_0(outcome_4, _S239, _S240, _S247, &_S248);
            path_0 = _S249;
        }
        if(path_0)
        {
            path_0 = true;
        }
        else
        {
            float _S250 = _S236.x;
            float _S251 = _S236.z;
            thread IlPlace_0 _S252 = _S241;
            bool _S253 = il_crater_0(outcome_4, _S239, _S250, _S251, &_S252);
            path_0 = _S253;
        }
        if(path_0)
        {
            uint _S254 = w_marks_0(kernelContext_54);
            *(kernelContext_54->work_0+(_S254 + k_11)) = _S238.launch_0;
        }
    }
    if(_S243)
    {
        uint _S255 = w_carry_0(kernelContext_54);
        *(kernelContext_54->work_0+(_S255 + k_11)) = _S232 + 1U;
        uint _S256 = a_il_0(_S232, kernelContext_54);
        add64_0(_S256 + 2U, (float4(*(kernelContext_54->chunks_0+4U * k_11)) ).w, kernelContext_54->params_0->scale_m_0, kernelContext_54);
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
    float4 _S257 = mat_0->dif_0;
    float ref_0 = mat_0->dif_0.x;
    if(r_16 <= ref_0)
    {
        return 1.0f;
    }
    float _S258 = _S257.z;
    float f_0;
    if(r_16 <= _S258)
    {
        float _S259 = (metal::fast::divide((r_16), (ref_0)));
        float _S260 = (metal::fast::pow((_S259), (_S257.y)));
        f_0 = _S260;
    }
    else
    {
        float _S261 = (metal::fast::divide((_S258), (ref_0)));
        float _S262 = (metal::fast::pow((_S261), (_S257.y)));
        float _S263 = (metal::fast::divide((r_16), (_S258)));
        float _S264 = (metal::fast::pow((_S263), (_S257.w)));
        f_0 = _S262 * _S264;
    }
    return clamp(f_0, 1.0f, mat_0->misc_0.x);
}

void il_impulse_0(uint b_18, uint sb_0, float m_shot_0, float3 contact_6, float3 d_8, float j_0, KernelContext_0 thread* kernelContext_55)
{
    float3 jr_0 = to_rubble_0(d_8) * float3(j_0) ;
    uint at_3 = 7U * b_18;
    float inv_mass_1 = (float4(*(kernelContext_55->bodies_0+at_3)) ).w;
    float3 _S265 = jr_0 * float3(inv_mass_1) ;
    float3 _S266 = float3(0.0f) ;
    float3 dw_0;
    if(inv_mass_1 > 0.0f)
    {
        Quat_0 q_10 = quat_of_0(float4(*(kernelContext_55->bodies_0+(at_3 + 1U))) );
        float3 r_17 = to_rubble_0(contact_6) - (float4(*(kernelContext_55->bodies_0+at_3)) ).xyz;
        float3 _S267 = (float4(*(kernelContext_55->bodies_0+(at_3 + 2U))) ).xyz;
        thread Quat_0 _S268 = q_10;
        Quat_0 _S269 = conj_0(&_S268);
        float3 _S270 = cross(r_17, jr_0);
        thread Quat_0 _S271 = _S269;
        float3 _S272 = rotate_0(&_S271, _S270);
        float3 _S273 = _S267 * _S272;
        thread Quat_0 _S274 = q_10;
        float3 _S275 = rotate_0(&_S274, _S273);
        dw_0 = _S275;
    }
    else
    {
        dw_0 = _S266;
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
        uint s_at_1 = 7U * (sb_0 + region_1 * kernelContext_55->params_0->max_bodies_0) + 3U;
        (*(kernelContext_55->bodies_0+s_at_1)).xyz = (float4(*(kernelContext_55->bodies_0+s_at_1)) ).xyz - jr_0 / float3(m_shot_0) ;
        uint w_at_0 = 7U * (b_18 + region_1 * kernelContext_55->params_0->max_bodies_0);
        uint _S276 = w_at_0 + 3U;
        (*(kernelContext_55->bodies_0+_S276)).xyz = (float4(*(kernelContext_55->bodies_0+_S276)) ).xyz + _S265;
        uint _S277 = w_at_0 + 4U;
        (*(kernelContext_55->bodies_0+_S277)).xyz = (float4(*(kernelContext_55->bodies_0+_S277)) ).xyz + dw_0;
        region_1 = region_1 + 1U;
    }
    return;
}

void il_strike_0(uint b_19, KernelContext_0 thread* kernelContext_56)
{
    bool damage_0;
    if(b_19 >= (kernelContext_56->params_0->body_count_0))
    {
        damage_0 = true;
    }
    else
    {
        bool _S278 = il_struck_0(b_19, kernelContext_56);
        damage_0 = !_S278;
    }
    if(damage_0)
    {
        return;
    }
    float _S279 = il_t_in_0(b_19, kernelContext_56);
    uint _S280 = il_entry_0(b_19, kernelContext_56);
    thread float3 contact_7;
    thread float3 d_9;
    thread float speed_5;
    il_path_0(b_19, &contact_7, &d_9, &speed_5, kernelContext_56);
    IlStriker_0 _S281 = il_striker_0(b_19, kernelContext_56);
    float _S282 = max(_S281.radius_0, 0.00100000004749745f);
    if((_S281.launch_0) != 0U)
    {
        uint _S283 = w_marks_0(kernelContext_56);
        damage_0 = (*(kernelContext_56->work_0+(_S283 + _S280))) == (_S281.launch_0);
    }
    else
    {
        damage_0 = false;
    }
    if(damage_0)
    {
        uint _S284 = lp_body_0(b_19, kernelContext_56);
        packed_float4 _S285 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_56->lp_0+_S284) = _S285;
        *(kernelContext_56->lp_0+(_S284 + 1U)) = _S285;
        return;
    }
    uint _S286 = a_il_0(b_19, kernelContext_56);
    uint _S287 = atomic_load_explicit(kernelContext_56->atom_0+_S286, memory_order_relaxed);
    uint _S288 = atomic_load_explicit(kernelContext_56->atom_0+(_S286 + 1U), memory_order_relaxed);
    if(_S287 != _S288)
    {
        uint _S289 = a_counters_1(kernelContext_56);
        uint _S290 = atomic_fetch_or_explicit(kernelContext_56->atom_0+(_S289 + 3U), 16U, memory_order_relaxed);
    }
    float _S291 = max((as_type<float>((_S287))) - _S282, 9.99999997475242708e-07f);
    uint o_2 = kernelContext_56->params_0->ext_chunks_0 + 4U * _S280;
    float4 _S292 = float4(*(kernelContext_56->ext_0+(o_2 + 1U))) ;
    thread JointMaterial_0 dm_0;
    float4 _S293 = float4(0.0f) ;
    (&dm_0)->strength_0 = _S293;
    (&dm_0)->energy_0 = _S293;
    (&dm_0)->dif_0 = float4(*(kernelContext_56->ext_0+(o_2 + 2U))) ;
    float4 _S294 = float4(*(kernelContext_56->ext_0+(o_2 + 3U))) ;
    float _S295 = _S294.x;
    (&dm_0)->misc_0 = float4(_S295, 0.0f, 0.0f, 0.0f);
    (&dm_0)->kind_flags_0 = uint4(0U) ;
    float nu_0 = _S294.y;
    float e_target_0 = _S294.z / (1.0f - nu_0 * nu_0);
    float e_star_0 = 1.0f / (1.0f / _S281.modulus_0 + 1.0f / e_target_0);
    float _S296 = max(speed_5, 0.00100000004749745f);
    float _S297 = sqrt(_S282);
    float h0_0 = pow(15.0f * _S281.m_eff_0 * _S296 * _S296 / (16.0f * e_star_0 * _S297), 0.40000000596046448f);
    float t_c_0 = 2.94000005722045898f * h0_0 / _S296;
    float force_0 = 1.33333337306976318f * e_star_0 * _S297 * pow(h0_0, 1.5f);
    float a_3 = sqrt(_S282 * h0_0);
    float p0_0 = 3.0f * force_0 / (6.28318548202514648f * a_3 * a_3);
    float ring_0 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    float k_12;
    if(_S295 > 0.0f)
    {
        float _S298 = _S296 / _S291;
        thread JointMaterial_0 _S299 = dm_0;
        float _S300 = dif_factor_0(&_S299, _S298);
        k_12 = _S300;
    }
    else
    {
        k_12 = 1.0f;
    }
    float ft_0 = k_12 * _S292.y;
    float fc_0 = k_12 * _S292.z;
    float tau_0 = k_12 * _S292.w;
    if(p0_0 > fc_0)
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ring_0 > ft_0;
    }
    float _S301 = fc_0 * (3.14159274101257324f * _S282 * _S282);
    float _S302 = min(_S301 * _S291, 1.57079637050628662f * (2.0f * _S282 + _S291) * _S291 * _S291 * tau_0);
    float _S303 = 2.0f * _S302;
    float v_bl_0 = sqrt(_S303 / _S281.m_eff_0);
    bool perforate_0;
    if(damage_0)
    {
        perforate_0 = _S296 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    float r2_0 = _S282 + _S291;
    float _S304 = _S292.x;
    float plug_0 = _S304 * 3.14159274101257324f * _S291 / 3.0f * (_S282 * _S282 + _S282 * r2_0 + r2_0 * r2_0);
    float v_res_0;
    if(perforate_0)
    {
        v_res_0 = _S281.mass_0 / (_S281.mass_0 + plug_0) * sqrt(max(_S296 * _S296 - v_bl_0 * v_bl_0, 0.0f));
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
        depth_1 = min(0.5f * _S281.m_eff_0 * _S296 * _S296 / _S301, _S291);
    }
    else
    {
        depth_1 = 0.0f;
    }
    bool crater_0 = depth_1 > a_3;
    float c_3 = sqrt(e_target_0 / _S304);
    float z_t_0 = _S304 * c_3;
    float rho_p_0 = _S281.mass_0 / (4.18879032135009766f * _S282 * _S282 * _S282);
    float z_p_0 = rho_p_0 * sqrt(_S281.modulus_0 / rho_p_0);
    float mean_pressure_0 = force_0 / (3.14159274101257324f * a_3 * a_3);
    float _S305 = min(min(z_t_0 * _S296 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_0);
    float source_radius_0;
    if(mean_pressure_0 > fc_0)
    {
        source_radius_0 = max(a_3, _S282);
    }
    else
    {
        source_radius_0 = a_3;
    }
    float pulse_0 = c_3 * t_c_0;
    float back_0 = _S305 * min(source_radius_0 / _S291, 1.0f) * min(2.0f * _S291 / pulse_0, 1.0f);
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
        scab_1 = 0.5f * min(pulse_0, _S291);
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
    uint _S306;
    if(crater_0)
    {
        _S306 = 2U;
    }
    else
    {
        _S306 = 0U;
    }
    uint _S307 = region_2 | _S306;
    if(spall_0)
    {
        region_2 = 4U;
    }
    else
    {
        region_2 = 0U;
    }
    uint _S308 = _S307 | region_2;
    if(perforate_0)
    {
        region_2 = 8U;
    }
    else
    {
        region_2 = 0U;
    }
    uint outcome_5 = _S308 | region_2;
    uint _S309 = lp_body_0(b_19, kernelContext_56);
    *(kernelContext_56->lp_0+_S309) = packed_float4(float4(_S291, v_bl_0, v_res_0, (as_type<float>((outcome_5))))) ;
    packed_float4 device* _S310 = kernelContext_56->lp_0+(_S309 + 1U);
    if(crater_0)
    {
        k_12 = depth_1;
    }
    else
    {
        k_12 = 0.0f;
    }
    *_S310 = packed_float4(float4(k_12, scab_1, a_3, _S302)) ;
    if(perforate_0)
    {
        uint _S311 = a_counters_1(kernelContext_56);
        uint _S312 = atomic_fetch_add_explicit(kernelContext_56->atom_0+(_S311 + 10U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_0)
        {
            uint _S313 = a_counters_1(kernelContext_56);
            uint _S314 = atomic_fetch_add_explicit(kernelContext_56->atom_0+(_S313 + 9U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_0)
            {
                uint _S315 = a_counters_1(kernelContext_56);
                uint _S316 = atomic_fetch_add_explicit(kernelContext_56->atom_0+(_S315 + 8U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S317 = a_counters_1(kernelContext_56);
                uint _S318 = atomic_fetch_add_explicit(kernelContext_56->atom_0+(_S317 + 7U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_0)
    {
        if((kernelContext_56->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_19, _S281.body_0, _S281.mass_0, contact_7, d_9, _S281.m_eff_0 * (_S296 - sqrt(max(_S296 * _S296 - _S303 / _S281.m_eff_0, 0.0f))), kernelContext_56);
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_19, _S281.body_0, _S281.mass_0, contact_7, d_9, _S281.m_eff_0 * _S296, kernelContext_56);
            float3 _S319 = to_rubble_0(d_9) * float3((_S296 * clamp(_S279, 0.0f, kernelContext_56->params_0->dt_0))) ;
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
                uint _S320 = 7U * (_S281.body_0 + region_2 * kernelContext_56->params_0->max_bodies_0);
                (*(kernelContext_56->bodies_0+_S320)).xyz = (float4(*(kernelContext_56->bodies_0+_S320)) ).xyz + _S319;
                region_2 = region_2 + 1U;
            }
        }
    }
    return;
}

void il_extend_0(uint k_13, uint from_0, KernelContext_0 thread* kernelContext_57)
{
    if(k_13 >= (kernelContext_57->params_0->chunk_count_0))
    {
        return;
    }
    uint _S321 = body_of_0(k_13, kernelContext_57);
    bool _S322 = il_struck_0(_S321, kernelContext_57);
    if(!_S322)
    {
        return;
    }
    thread float3 contact_8;
    thread float3 d_10;
    thread float speed_6;
    il_path_0(_S321, &contact_8, &d_10, &speed_6, kernelContext_57);
    IlStriker_0 _S323 = il_striker_0(_S321, kernelContext_57);
    uint _S324 = a_il_0(_S321, kernelContext_57);
    uint _S325 = atomic_load_explicit(kernelContext_57->atom_0+(_S324 + from_0), memory_order_relaxed);
    float end_0 = (as_type<float>((_S325)));
    uint to_0 = _S324 + 1U - from_0;
    uint _S326 = atomic_fetch_max_explicit(kernelContext_57->atom_0+to_0, (as_type<uint>((end_0))), memory_order_relaxed);
    float2 _S327 = il_interval_0(k_13, contact_8, d_10, _S323.radius_0, kernelContext_57);
    float _S328 = _S327.x;
    float _S329 = _S327.y;
    bool _S330;
    if(_S328 <= _S329)
    {
        _S330 = _S328 <= (end_0 + 0.00100000004749745f + 0.01999999955296516f * _S323.radius_0);
    }
    else
    {
        _S330 = false;
    }
    if(_S330)
    {
        _S330 = _S329 > end_0;
    }
    else
    {
        _S330 = false;
    }
    if(_S330)
    {
        uint _S331 = atomic_fetch_max_explicit(kernelContext_57->atom_0+to_0, (as_type<uint>((_S329))), memory_order_relaxed);
    }
    return;
}

void il_begin_0(uint b_20, KernelContext_0 thread* kernelContext_58)
{
    bool _S332;
    if(b_20 >= (kernelContext_58->params_0->body_count_0))
    {
        _S332 = true;
    }
    else
    {
        bool _S333 = il_struck_0(b_20, kernelContext_58);
        _S332 = !_S333;
    }
    if(_S332)
    {
        return;
    }
    thread float3 contact_9;
    thread float3 d_11;
    thread float speed_7;
    il_path_0(b_20, &contact_9, &d_11, &speed_7, kernelContext_58);
    uint _S334 = il_entry_0(b_20, kernelContext_58);
    IlStriker_0 _S335 = il_striker_0(b_20, kernelContext_58);
    float2 _S336 = il_interval_0(_S334, contact_9, d_11, _S335.radius_0, kernelContext_58);
    uint end_1 = (as_type<uint>((max(_S336.y, 0.0f))));
    uint _S337 = a_il_0(b_20, kernelContext_58);
    atomic_store_explicit(kernelContext_58->atom_0+_S337, end_1, memory_order_relaxed);
    atomic_store_explicit(kernelContext_58->atom_0+(_S337 + 1U), end_1, memory_order_relaxed);
    atomic_store_explicit(kernelContext_58->atom_0+(_S337 + 2U), 0U, memory_order_relaxed);
    atomic_store_explicit(kernelContext_58->atom_0+(_S337 + 3U), 0U, memory_order_relaxed);
    return;
}

void il_breaks_1(uint i_1, KernelContext_0 thread* kernelContext_59)
{
    bool cut_1;
    if(i_1 >= (kernelContext_59->params_0->bond_count_0))
    {
        cut_1 = true;
    }
    else
    {
        uint _S338 = w_alive_0(kernelContext_59);
        cut_1 = (*(kernelContext_59->work_0+(_S338 + i_1))) == 0U;
    }
    if(cut_1)
    {
        return;
    }
    uint4 _S339 = uint4(*(kernelContext_59->bonds_0+i_1)) ;
    uint _S340 = _S339.x;
    uint _S341 = body_of_0(_S340, kernelContext_59);
    bool _S342 = il_struck_1(_S341, kernelContext_59);
    if(!_S342)
    {
        return;
    }
    uint _S343 = lp_body_0(_S341, kernelContext_59);
    float4 _S344 = float4(*(kernelContext_59->lp_0+_S343)) ;
    float4 _S345 = float4(*(kernelContext_59->lp_0+(_S343 + 1U))) ;
    uint outcome_6 = (as_type<uint>((_S344.w)));
    if((outcome_6 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_10;
    thread float3 d_12;
    thread float speed_8;
    il_path_1(_S341, &contact_10, &d_12, &speed_8, kernelContext_59);
    uint _S346 = il_entry_1(_S341, kernelContext_59);
    IlStriker_0 _S347 = il_striker_1(_S341, kernelContext_59);
    float _S348 = max(_S347.radius_0, 0.00100000004749745f);
    float h_4 = _S344.x;
    IlPlace_0 _S349 = il_place_0(_S340, _S346, contact_10, d_12, _S348, h_4, kernelContext_59);
    IlPlace_0 _S350 = il_place_0(_S339.y, _S346, contact_10, d_12, _S348, h_4, kernelContext_59);
    thread IlPlace_0 _S351 = _S349;
    bool _S352 = il_plug_0(outcome_6, _S348, h_4, &_S351);
    thread IlPlace_0 _S353 = _S350;
    bool _S354 = il_plug_0(outcome_6, _S348, h_4, &_S353);
    if(_S352 != _S354)
    {
        cut_1 = true;
    }
    else
    {
        float _S355 = _S345.y;
        thread IlPlace_0 _S356 = _S349;
        bool _S357 = il_scab_0(outcome_6, _S348, h_4, _S355, &_S356);
        thread IlPlace_0 _S358 = _S350;
        bool _S359 = il_scab_0(outcome_6, _S348, h_4, _S355, &_S358);
        cut_1 = _S357 != _S359;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S360 = _S345.x;
        float _S361 = _S345.z;
        thread IlPlace_0 _S362 = _S349;
        bool _S363 = il_crater_0(outcome_6, _S348, _S360, _S361, &_S362);
        cut_1 = _S363;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S364 = _S345.x;
        float _S365 = _S345.z;
        thread IlPlace_0 _S366 = _S350;
        bool _S367 = il_crater_0(outcome_6, _S348, _S364, _S365, &_S366);
        cut_1 = _S367;
    }
    if(cut_1)
    {
        uint _S368 = w_alive_0(kernelContext_59);
        *(kernelContext_59->work_0+(_S368 + i_1)) = 0U;
        uint _S369 = a_dirty_0(kernelContext_59);
        uint _S370 = atomic_fetch_or_explicit(kernelContext_59->atom_0+(_S369 + _S341), 1U, memory_order_relaxed);
        uint _S371 = a_counters_1(kernelContext_59);
        uint _S372 = atomic_fetch_add_explicit(kernelContext_59->atom_0+_S371, 1U, memory_order_relaxed);
        uint _S373 = atomic_fetch_add_explicit(kernelContext_59->atom_0+(_S371 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

void il_carry_1(uint k_14, KernelContext_0 thread* kernelContext_60)
{
    if(k_14 >= (kernelContext_60->params_0->chunk_count_0))
    {
        return;
    }
    uint _S374 = w_carry_0(kernelContext_60);
    uint _S375 = _S374 + k_14;
    uint device* _S376 = kernelContext_60->work_0+_S375;
    uint c_4 = *_S376;
    if((*_S376) == 0U)
    {
        return;
    }
    *(kernelContext_60->work_0+_S375) = 0U;
    uint b0_1 = c_4 - 1U;
    bool _S377 = il_struck_1(b0_1, kernelContext_60);
    if(!_S377)
    {
        return;
    }
    thread float3 contact_11;
    thread float3 d_13;
    thread float speed_9;
    il_path_1(b0_1, &contact_11, &d_13, &speed_9, kernelContext_60);
    float3 dr_1 = to_rubble_0(d_13);
    IlStriker_0 _S378 = il_striker_1(b0_1, kernelContext_60);
    uint _S379 = a_il_0(b0_1, kernelContext_60);
    float _S380 = read64_0(_S379 + 2U, kernelContext_60->params_0->scale_m_0, kernelContext_60);
    float3 v_wall_1 = (float4(*(kernelContext_60->bodies_0+(7U * b0_1 + 3U))) ).xyz;
    uint _S381 = 7U * _S378.body_0;
    float3 v_shot_1 = (float4(*(kernelContext_60->bodies_0+(_S381 + 3U))) ).xyz;
    float closing_1 = dot(v_shot_1 - v_wall_1, dr_1);
    float v_r_1 = _S378.mass_0 * closing_1 / (_S378.mass_0 + _S380);
    float3 _S382 = v_shot_1 + dr_1 * float3((v_r_1 - closing_1)) ;
    float3 _S383 = v_wall_1 + dr_1 * float3(v_r_1) ;
    float _S384 = il_t_in_1(b0_1, kernelContext_60);
    float3 _S385 = (float4(*(kernelContext_60->bodies_0+_S381)) ).xyz + dr_1 * float3(((speed_9 - v_r_1) * clamp(_S384, 0.0f, kernelContext_60->params_0->dt_0))) ;
    uint _S386 = body_of_0(k_14, kernelContext_60);
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
        uint s_at_2 = 7U * (_S378.body_0 + region_3 * kernelContext_60->params_0->max_bodies_0);
        (*(kernelContext_60->bodies_0+s_at_2)).xyz = _S385;
        (*(kernelContext_60->bodies_0+(s_at_2 + 3U))).xyz = _S382;
        (*(kernelContext_60->bodies_0+(7U * (_S386 + region_3 * kernelContext_60->params_0->max_bodies_0) + 3U))).xyz = _S383;
        region_3 = region_3 + 1U;
    }
    return;
}

void il_mark_1(uint k_15, KernelContext_0 thread* kernelContext_61)
{
    if(k_15 >= (kernelContext_61->params_0->chunk_count_0))
    {
        return;
    }
    uint _S387 = body_of_0(k_15, kernelContext_61);
    bool _S388 = il_struck_1(_S387, kernelContext_61);
    if(!_S388)
    {
        return;
    }
    uint _S389 = lp_body_0(_S387, kernelContext_61);
    float4 _S390 = float4(*(kernelContext_61->lp_0+_S389)) ;
    float4 _S391 = float4(*(kernelContext_61->lp_0+(_S389 + 1U))) ;
    uint outcome_7 = (as_type<uint>((_S390.w)));
    if((outcome_7 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_12;
    thread float3 d_14;
    thread float speed_10;
    il_path_1(_S387, &contact_12, &d_14, &speed_10, kernelContext_61);
    uint _S392 = il_entry_1(_S387, kernelContext_61);
    IlStriker_0 _S393 = il_striker_1(_S387, kernelContext_61);
    float _S394 = max(_S393.radius_0, 0.00100000004749745f);
    float _S395 = _S390.x;
    IlPlace_0 _S396 = il_place_0(k_15, _S392, contact_12, d_14, _S394, _S395, kernelContext_61);
    thread IlPlace_0 _S397 = _S396;
    bool _S398 = il_plug_0(outcome_7, _S394, _S395, &_S397);
    if((_S393.launch_0) != 0U)
    {
        float2 _S399 = il_interval_0(k_15, contact_12 - d_14 * float3(_S393.bound_0) , d_14, _S393.bound_0, kernelContext_61);
        float _S400 = _S399.x;
        float _S401 = _S399.y;
        bool path_1;
        if(_S400 <= _S401)
        {
            path_1 = _S400 <= (_S395 + 2.0f * _S393.bound_0);
        }
        else
        {
            path_1 = false;
        }
        if(path_1)
        {
            path_1 = _S401 >= 0.0f;
        }
        else
        {
            path_1 = false;
        }
        if(_S398)
        {
            path_1 = true;
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            float _S402 = _S391.y;
            thread IlPlace_0 _S403 = _S396;
            bool _S404 = il_scab_0(outcome_7, _S394, _S395, _S402, &_S403);
            path_1 = _S404;
        }
        if(path_1)
        {
            path_1 = true;
        }
        else
        {
            float _S405 = _S391.x;
            float _S406 = _S391.z;
            thread IlPlace_0 _S407 = _S396;
            bool _S408 = il_crater_0(outcome_7, _S394, _S405, _S406, &_S407);
            path_1 = _S408;
        }
        if(path_1)
        {
            uint _S409 = w_marks_0(kernelContext_61);
            *(kernelContext_61->work_0+(_S409 + k_15)) = _S393.launch_0;
        }
    }
    if(_S398)
    {
        uint _S410 = w_carry_0(kernelContext_61);
        *(kernelContext_61->work_0+(_S410 + k_15)) = _S387 + 1U;
        uint _S411 = a_il_0(_S387, kernelContext_61);
        add64_0(_S411 + 2U, (float4(*(kernelContext_61->chunks_0+4U * k_15)) ).w, kernelContext_61->params_0->scale_m_0, kernelContext_61);
    }
    return;
}

void il_strike_1(uint b_21, KernelContext_0 thread* kernelContext_62)
{
    bool damage_1;
    if(b_21 >= (kernelContext_62->params_0->body_count_0))
    {
        damage_1 = true;
    }
    else
    {
        bool _S412 = il_struck_1(b_21, kernelContext_62);
        damage_1 = !_S412;
    }
    if(damage_1)
    {
        return;
    }
    float _S413 = il_t_in_1(b_21, kernelContext_62);
    uint _S414 = il_entry_1(b_21, kernelContext_62);
    thread float3 contact_13;
    thread float3 d_15;
    thread float speed_11;
    il_path_1(b_21, &contact_13, &d_15, &speed_11, kernelContext_62);
    IlStriker_0 _S415 = il_striker_1(b_21, kernelContext_62);
    float _S416 = max(_S415.radius_0, 0.00100000004749745f);
    if((_S415.launch_0) != 0U)
    {
        uint _S417 = w_marks_0(kernelContext_62);
        damage_1 = (*(kernelContext_62->work_0+(_S417 + _S414))) == (_S415.launch_0);
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        uint _S418 = lp_body_0(b_21, kernelContext_62);
        packed_float4 _S419 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_62->lp_0+_S418) = _S419;
        *(kernelContext_62->lp_0+(_S418 + 1U)) = _S419;
        return;
    }
    uint _S420 = a_il_0(b_21, kernelContext_62);
    uint _S421 = atomic_load_explicit(kernelContext_62->atom_0+_S420, memory_order_relaxed);
    uint _S422 = atomic_load_explicit(kernelContext_62->atom_0+(_S420 + 1U), memory_order_relaxed);
    if(_S421 != _S422)
    {
        uint _S423 = a_counters_1(kernelContext_62);
        uint _S424 = atomic_fetch_or_explicit(kernelContext_62->atom_0+(_S423 + 3U), 16U, memory_order_relaxed);
    }
    float _S425 = max((as_type<float>((_S421))) - _S416, 9.99999997475242708e-07f);
    if(!(_S413 > 0.0f))
    {
        uint _S426 = lp_body_0(b_21, kernelContext_62);
        *(kernelContext_62->lp_0+_S426) = packed_float4(float4(_S425, 0.0f, 0.0f, 0.0f)) ;
        *(kernelContext_62->lp_0+(_S426 + 1U)) = packed_float4(float4(0.0f) ) ;
        return;
    }
    uint o_3 = kernelContext_62->params_0->ext_chunks_0 + 4U * _S414;
    float4 _S427 = float4(*(kernelContext_62->ext_0+(o_3 + 1U))) ;
    thread JointMaterial_0 dm_1;
    float4 _S428 = float4(0.0f) ;
    (&dm_1)->strength_0 = _S428;
    (&dm_1)->energy_0 = _S428;
    (&dm_1)->dif_0 = float4(*(kernelContext_62->ext_0+(o_3 + 2U))) ;
    float4 _S429 = float4(*(kernelContext_62->ext_0+(o_3 + 3U))) ;
    float _S430 = _S429.x;
    (&dm_1)->misc_0 = float4(_S430, 0.0f, 0.0f, 0.0f);
    (&dm_1)->kind_flags_0 = uint4(0U) ;
    float nu_1 = _S429.y;
    float e_target_1 = _S429.z / (1.0f - nu_1 * nu_1);
    float e_star_1 = 1.0f / (1.0f / _S415.modulus_0 + 1.0f / e_target_1);
    float _S431 = max(speed_11, 0.00100000004749745f);
    float _S432 = sqrt(_S416);
    float h0_1 = pow(15.0f * _S415.m_eff_0 * _S431 * _S431 / (16.0f * e_star_1 * _S432), 0.40000000596046448f);
    float t_c_1 = 2.94000005722045898f * h0_1 / _S431;
    float force_1 = 1.33333337306976318f * e_star_1 * _S432 * pow(h0_1, 1.5f);
    float a_4 = sqrt(_S416 * h0_1);
    float p0_1 = 3.0f * force_1 / (6.28318548202514648f * a_4 * a_4);
    float ring_1 = (1.0f - 2.0f * nu_1) * p0_1 / 3.0f;
    float k_16;
    if(_S430 > 0.0f)
    {
        float _S433 = _S431 / _S425;
        thread JointMaterial_0 _S434 = dm_1;
        float _S435 = dif_factor_0(&_S434, _S433);
        k_16 = _S435;
    }
    else
    {
        k_16 = 1.0f;
    }
    float ft_1 = k_16 * _S427.y;
    float fc_1 = k_16 * _S427.z;
    float tau_1 = k_16 * _S427.w;
    if(p0_1 > fc_1)
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = ring_1 > ft_1;
    }
    float _S436 = fc_1 * (3.14159274101257324f * _S416 * _S416);
    float _S437 = min(_S436 * _S425, 1.57079637050628662f * (2.0f * _S416 + _S425) * _S425 * _S425 * tau_1);
    float _S438 = 2.0f * _S437;
    float v_bl_1 = sqrt(_S438 / _S415.m_eff_0);
    bool perforate_1;
    if(damage_1)
    {
        perforate_1 = _S431 >= v_bl_1;
    }
    else
    {
        perforate_1 = false;
    }
    float r2_1 = _S416 + _S425;
    float _S439 = _S427.x;
    float plug_1 = _S439 * 3.14159274101257324f * _S425 / 3.0f * (_S416 * _S416 + _S416 * r2_1 + r2_1 * r2_1);
    float v_res_1;
    if(perforate_1)
    {
        v_res_1 = _S415.mass_0 / (_S415.mass_0 + plug_1) * sqrt(max(_S431 * _S431 - v_bl_1 * v_bl_1, 0.0f));
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
        depth_2 = min(0.5f * _S415.m_eff_0 * _S431 * _S431 / _S436, _S425);
    }
    else
    {
        depth_2 = 0.0f;
    }
    bool crater_1 = depth_2 > a_4;
    float c_5 = sqrt(e_target_1 / _S439);
    float z_t_1 = _S439 * c_5;
    float rho_p_1 = _S415.mass_0 / (4.18879032135009766f * _S416 * _S416 * _S416);
    float z_p_1 = rho_p_1 * sqrt(_S415.modulus_0 / rho_p_1);
    float mean_pressure_1 = force_1 / (3.14159274101257324f * a_4 * a_4);
    float _S440 = min(min(z_t_1 * _S431 * z_p_1 / (z_p_1 + z_t_1), mean_pressure_1), fc_1);
    float source_radius_1;
    if(mean_pressure_1 > fc_1)
    {
        source_radius_1 = max(a_4, _S416);
    }
    else
    {
        source_radius_1 = a_4;
    }
    float pulse_1 = c_5 * t_c_1;
    float back_1 = _S440 * min(source_radius_1 / _S425, 1.0f) * min(2.0f * _S425 / pulse_1, 1.0f);
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
        scab_2 = 0.5f * min(pulse_1, _S425);
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
    uint _S441;
    if(crater_1)
    {
        _S441 = 2U;
    }
    else
    {
        _S441 = 0U;
    }
    uint _S442 = region_4 | _S441;
    if(spall_1)
    {
        region_4 = 4U;
    }
    else
    {
        region_4 = 0U;
    }
    uint _S443 = _S442 | region_4;
    if(perforate_1)
    {
        region_4 = 8U;
    }
    else
    {
        region_4 = 0U;
    }
    uint outcome_8 = _S443 | region_4;
    uint _S444 = lp_body_0(b_21, kernelContext_62);
    *(kernelContext_62->lp_0+_S444) = packed_float4(float4(_S425, v_bl_1, v_res_1, (as_type<float>((outcome_8))))) ;
    packed_float4 device* _S445 = kernelContext_62->lp_0+(_S444 + 1U);
    if(crater_1)
    {
        k_16 = depth_2;
    }
    else
    {
        k_16 = 0.0f;
    }
    *_S445 = packed_float4(float4(k_16, scab_2, a_4, _S437)) ;
    if(perforate_1)
    {
        uint _S446 = a_counters_1(kernelContext_62);
        uint _S447 = atomic_fetch_add_explicit(kernelContext_62->atom_0+(_S446 + 10U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_1)
        {
            uint _S448 = a_counters_1(kernelContext_62);
            uint _S449 = atomic_fetch_add_explicit(kernelContext_62->atom_0+(_S448 + 9U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_1)
            {
                uint _S450 = a_counters_1(kernelContext_62);
                uint _S451 = atomic_fetch_add_explicit(kernelContext_62->atom_0+(_S450 + 8U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S452 = a_counters_1(kernelContext_62);
                uint _S453 = atomic_fetch_add_explicit(kernelContext_62->atom_0+(_S452 + 7U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_1)
    {
        if((kernelContext_62->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_21, _S415.body_0, _S415.mass_0, contact_13, d_15, _S415.m_eff_0 * (_S431 - sqrt(max(_S431 * _S431 - _S438 / _S415.m_eff_0, 0.0f))), kernelContext_62);
        }
    }
    else
    {
        if(damage_1)
        {
            il_impulse_0(b_21, _S415.body_0, _S415.mass_0, contact_13, d_15, _S415.m_eff_0 * _S431, kernelContext_62);
            float3 _S454 = to_rubble_0(d_15) * float3((_S431 * clamp(_S413, 0.0f, kernelContext_62->params_0->dt_0))) ;
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
                uint _S455 = 7U * (_S415.body_0 + region_4 * kernelContext_62->params_0->max_bodies_0);
                (*(kernelContext_62->bodies_0+_S455)).xyz = (float4(*(kernelContext_62->bodies_0+_S455)) ).xyz + _S454;
                region_4 = region_4 + 1U;
            }
        }
    }
    return;
}

void il_extend_1(uint k_17, uint from_1, KernelContext_0 thread* kernelContext_63)
{
    if(k_17 >= (kernelContext_63->params_0->chunk_count_0))
    {
        return;
    }
    uint _S456 = body_of_0(k_17, kernelContext_63);
    bool _S457 = il_struck_1(_S456, kernelContext_63);
    if(!_S457)
    {
        return;
    }
    thread float3 contact_14;
    thread float3 d_16;
    thread float speed_12;
    il_path_1(_S456, &contact_14, &d_16, &speed_12, kernelContext_63);
    IlStriker_0 _S458 = il_striker_1(_S456, kernelContext_63);
    uint _S459 = a_il_0(_S456, kernelContext_63);
    uint _S460 = atomic_load_explicit(kernelContext_63->atom_0+(_S459 + from_1), memory_order_relaxed);
    float end_2 = (as_type<float>((_S460)));
    uint to_1 = _S459 + 1U - from_1;
    uint _S461 = atomic_fetch_max_explicit(kernelContext_63->atom_0+to_1, (as_type<uint>((end_2))), memory_order_relaxed);
    float2 _S462 = il_interval_0(k_17, contact_14, d_16, _S458.radius_0, kernelContext_63);
    float _S463 = _S462.x;
    float _S464 = _S462.y;
    bool _S465;
    if(_S463 <= _S464)
    {
        _S465 = _S463 <= (end_2 + 0.00100000004749745f + 0.01999999955296516f * _S458.radius_0);
    }
    else
    {
        _S465 = false;
    }
    if(_S465)
    {
        _S465 = _S464 > end_2;
    }
    else
    {
        _S465 = false;
    }
    if(_S465)
    {
        uint _S466 = atomic_fetch_max_explicit(kernelContext_63->atom_0+to_1, (as_type<uint>((_S464))), memory_order_relaxed);
    }
    return;
}

uint a_first_1(uint shot_3, KernelContext_0 thread* kernelContext_64)
{
    uint _S467 = a_strike_0(kernelContext_64);
    return _S467 + 27U * kernelContext_64->params_0->body_count_0 + shot_3;
}

void il_begin_1(uint b_22, KernelContext_0 thread* kernelContext_65)
{
    bool _S468;
    if(b_22 >= (kernelContext_65->params_0->body_count_0))
    {
        _S468 = true;
    }
    else
    {
        bool _S469 = il_struck_1(b_22, kernelContext_65);
        _S468 = !_S469;
    }
    if(_S468)
    {
        return;
    }
    thread uint shot_4;
    bool _S470 = struck_0(b_22, &shot_4, kernelContext_65);
    uint _S471 = a_strike_0(kernelContext_65);
    uint _S472 = atomic_load_explicit(kernelContext_65->atom_0+(_S471 + 9U * b_22 + 6U), memory_order_relaxed);
    uint _S473 = a_first_1(shot_4, kernelContext_65);
    uint _S474 = atomic_load_explicit(kernelContext_65->atom_0+_S473, memory_order_relaxed);
    if(_S472 != _S474)
    {
        uint _S475 = strike_at_0(b_22, kernelContext_65);
        (*(kernelContext_65->frames_0+_S475))[int(3)] = 0.0f;
        return;
    }
    thread float3 contact_15;
    thread float3 d_17;
    thread float speed_13;
    il_path_1(b_22, &contact_15, &d_17, &speed_13, kernelContext_65);
    uint _S476 = il_entry_1(b_22, kernelContext_65);
    IlStriker_0 _S477 = il_striker_1(b_22, kernelContext_65);
    float2 _S478 = il_interval_0(_S476, contact_15, d_17, _S477.radius_0, kernelContext_65);
    uint end_3 = (as_type<uint>((max(_S478.y, 0.0f))));
    uint _S479 = a_il_0(b_22, kernelContext_65);
    atomic_store_explicit(kernelContext_65->atom_0+_S479, end_3, memory_order_relaxed);
    atomic_store_explicit(kernelContext_65->atom_0+(_S479 + 1U), end_3, memory_order_relaxed);
    atomic_store_explicit(kernelContext_65->atom_0+(_S479 + 2U), 0U, memory_order_relaxed);
    atomic_store_explicit(kernelContext_65->atom_0+(_S479 + 3U), 0U, memory_order_relaxed);
    return;
}

uint l_pool_0(uint half_1, KernelContext_0 thread* kernelContext_66)
{
    return kernelContext_66->params_0->chunk_count_0 + 3U * kernelContext_66->params_0->pool_cap_0 * half_1;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S480 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S480, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S481 = max(0.0f, _S480);
    float _S482 = min(dt_1, _S480 + duration_0);
    bool _S483;
    if(_S482 > _S481)
    {
        _S483 = duration_0 > 0.0f;
    }
    else
    {
        _S483 = false;
    }
    float sum_1;
    if(_S483)
    {
        float w_2 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S481 - _S480)) - cos(w_2 * (_S482 - _S480)));
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

uint l_vec_0(KernelContext_0 thread* kernelContext_67)
{
    return kernelContext_67->params_0->chunk_count_0 + 6U * kernelContext_67->params_0->pool_cap_0;
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
    thread KernelContext_0 kernelContext_68;
    (&kernelContext_68)->params_0 = params_1;
    (&kernelContext_68)->work_0 = work_1;
    (&kernelContext_68)->frames_0 = frames_1;
    (&kernelContext_68)->bodies_0 = bodies_1;
    (&kernelContext_68)->chunks_0 = chunks_1;
    (&kernelContext_68)->sort_0 = sort_1;
    (&kernelContext_68)->atom_0 = atom_1;
    (&kernelContext_68)->manifolds_0 = manifolds_1;
    (&kernelContext_68)->loads_0 = loads_1;
    (&kernelContext_68)->bonds_0 = bonds_1;
    (&kernelContext_68)->args_0 = args_1;
    (&kernelContext_68)->info_0 = info_1;
    (&kernelContext_68)->scan_0 = scan_1;
    (&kernelContext_68)->parts_0 = parts_1;
    (&kernelContext_68)->colliders_0 = colliders_1;
    (&kernelContext_68)->collider_info_0 = collider_info_1;
    (&kernelContext_68)->view_chunks_0 = view_chunks_1;
    (&kernelContext_68)->view_0 = view_1;
    (&kernelContext_68)->objects_0 = objects_1;
    (&kernelContext_68)->shot_bodies_0 = shot_bodies_1;
    (&kernelContext_68)->lines_0 = lines_1;
    (&kernelContext_68)->geometry_0 = geometry_1;
    (&kernelContext_68)->shots_0 = shots_1;
    (&kernelContext_68)->lp_0 = lp_1;
    (&kernelContext_68)->ext_0 = ext_2;
    (&kernelContext_68)->lp_sides_0 = lp_sides_1;
    (&kernelContext_68)->engine_counters_0 = engine_counters_1;
    (&kernelContext_68)->engine_contacts_0 = engine_contacts_1;
    (&kernelContext_68)->cs_anchors_0 = cs_anchors_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_68)->g_scan_0 = &g_scan_1;
    uint k_18 = id_0.x;
    if(k_18 >= (params_1->chunk_count_0))
    {
        return;
    }
    uint _S484 = body_of_0(k_18, &kernelContext_68);
    uint b_23 = 7U * _S484;
    thread Quat_0 q_11;
    thread float3 origin_3;
    structure_pose_0(_S484, b_23, &q_11, &origin_3, &kernelContext_68);
    uint _S485 = 4U * k_18;
    float4 _S486 = float4(*((&kernelContext_68)->chunks_0+_S485)) ;
    float3 _S487 = _S486.xyz;
    thread Quat_0 _S488 = q_11;
    float3 _S489 = rotate_0(&_S488, _S487);
    float3 centre_0 = origin_3 + _S489;
    float inv_mass_2 = (float4(*((&kernelContext_68)->bodies_0+b_23)) ).w;
    float m_body_0;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint from_2 = (&kernelContext_68)->params_0->parity_0;
    uint to_2 = 1U - (&kernelContext_68)->params_0->parity_0;
    uint _S490 = w_pulses_0(&kernelContext_68);
    uint _S491 = _S490 + _S485;
    uint _S492 = _S491 + 2U * from_2;
    uint off_0 = *((&kernelContext_68)->work_0+_S492);
    uint device* _S493 = (&kernelContext_68)->work_0+(_S492 + 1U);
    uint carried_0 = *_S493;
    uint room_0 = *_S493 + (&kernelContext_68)->sort_0[params_1->chunk_count_0 + 1U + k_18];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S494 = a_counters_1(&kernelContext_68);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_68)->atom_0+(_S494 + 5U + to_2), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_68)->params_0->pool_cap_0))
        {
            uint _S495 = atomic_fetch_or_explicit((&kernelContext_68)->atom_0+(_S494 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_68)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_68)->params_0->pool_cap_0 - base_1;
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
        uint _S496 = base_0;
        base_0 = base_1;
        room_1 = _S496;
    }
    else
    {
        base_0 = 0U;
        room_1 = room_0;
    }
    uint _S497 = params_1->chunk_count_0;
    float _S498 = _S486.w;
    float4 _S499 = float4(*((&kernelContext_68)->chunks_0+(_S485 + 3U))) ;
    float _S500 = (&kernelContext_68)->params_0->band_0;
    float _S501 = (&kernelContext_68)->params_0->dt_0;
    bool _S502 = ((&kernelContext_68)->params_0->impact_peak_0) != 0U;
    uint _S503 = l_pool_0(from_2, &kernelContext_68);
    uint _S504 = _S503 + 3U * off_0;
    uint _S505 = l_pool_0(to_2, &kernelContext_68);
    uint _S506 = _S505 + 3U * base_0;
    for(;;)
    {
        float3 _S507 = float3(0.0f) ;
        uint _S508 = (&kernelContext_68)->sort_0[k_18];
        uint _S509 = (&kernelContext_68)->sort_0[_S497 + 1U + k_18];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_1 = 0U;
        f_total_0 = _S507;
        t_total_0 = _S507;
        for(;;)
        {
            if(step_1 < _S509)
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
                if(q3_0 < _S509)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_68)->manifolds_0[_S508 + q3_0].ids_0) ).y + 1U;
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
            Record_natural_0 r_19 = (&kernelContext_68)->manifolds_0[_S508 + p_10];
            float4 _S510 = float4(r_19.impulse_0) ;
            float3 _S511 = _S510.xyz;
            float j_1 = length(_S511);
            if(j_1 == 0.0f)
            {
                uint _S512 = step_1 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S512;
                continue;
            }
            float m_other_0 = (float4(r_19.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S498;
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
            float _S513 = min(j_1, 2.0f * m_1 * max(_S510.w - _S500, 0.0f));
            if(_S513 < j_1)
            {
                float3 f_1 = _S511 / float3(j_1)  * float3(((j_1 - _S513) / _S501)) ;
                float3 _S514 = t_total_0 + cross((float4(r_19.point_0) ).xyz - centre_0, f_1);
                f_total_1 = f_total_0 + f_1;
                t_total_1 = _S514;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            uint _S512 = step_1 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S512;
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
            uint _S515 = _S504 + 3U * p_10;
            float4 _S516 = float4(*((&kernelContext_68)->loads_0+_S515)) ;
            thread float4 a_5 = _S516;
            float4 _S517 = float4(*((&kernelContext_68)->loads_0+(_S515 + 1U))) ;
            float4 _S518 = float4(*((&kernelContext_68)->loads_0+(_S515 + 2U))) ;
            float _S519 = _S517.w;
            float mean_0 = pulse_mean_0(_S516.x, _S516.y, _S516.z, _S516.w, _S519, _S501);
            if(mean_0 != 0.0f)
            {
                float3 f_2 = _S518.xyz * float3(mean_0) ;
                float3 _S520 = t_total_0 + cross(_S517.xyz - centre_0, f_2);
                f_total_0 = f_total_0 + f_2;
                t_total_0 = _S520;
            }
            if((a_5.x + a_5.z + _S519) > _S501)
            {
                a_5.x = a_5.x - _S501;
                uint _S521 = _S506 + 3U * kept_0;
                *((&kernelContext_68)->loads_0+_S521) = packed_float4(a_5) ;
                *((&kernelContext_68)->loads_0+(_S521 + 1U)) = packed_float4(_S517) ;
                *((&kernelContext_68)->loads_0+(_S521 + 2U)) = packed_float4(_S518) ;
                kept_0 = kept_0 + 1U;
            }
            p_10 = p_10 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S509)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_20 = (&kernelContext_68)->manifolds_0[_S508 + q3_0];
            float4 _S522 = float4(r_20.impulse_0) ;
            float3 _S523 = _S522.xyz;
            float j_2 = length(_S523);
            if(j_2 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S524 = float4(r_20.other_0) ;
            float m_other_1 = _S524.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S498;
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
            float _S525 = _S522.w;
            float _S526 = min(j_2, 2.0f * m_1 * max(_S525 - _S500, 0.0f));
            if(_S526 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S527 = _S499.y;
            float duration_1 = hertz_duration_0(m_1, min(_S524.z, _S499.z), 1.0f / ((1.0f - _S527 * _S527) / _S499.x + 1.0f / _S524.y), _S525);
            float peak_1 = 3.14159274101257324f * _S526 / (2.0f * duration_1);
            float3 dir_0 = _S523 / float3(j_2) ;
            float mean_1;
            if(_S502)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S501);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_3 = dir_0 * float3(mean_1) ;
                float3 _S528 = t_total_0 + cross((float4(r_20.point_0) ).xyz - centre_0, f_3);
                f_total_1 = f_total_0 + f_3;
                t_total_1 = _S528;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            uint kept_1;
            if(duration_1 > _S501)
            {
                if(kept_0 < room_1)
                {
                    uint _S529 = _S506 + 3U * kept_0;
                    *((&kernelContext_68)->loads_0+_S529) = packed_float4(float4(- _S501, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_68)->loads_0+(_S529 + 1U)) = packed_float4(float4((float4(r_20.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_68)->loads_0+(_S529 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
        uint _S530 = a_counters_1(&kernelContext_68);
        uint _S531 = atomic_fetch_or_explicit((&kernelContext_68)->atom_0+(_S530 + 3U), 8U, memory_order_relaxed);
    }
    uint _S532 = _S491 + 2U * to_2;
    *((&kernelContext_68)->work_0+_S532) = base_0;
    *((&kernelContext_68)->work_0+(_S532 + 1U)) = kept_0;
    *((&kernelContext_68)->loads_0+k_18) = packed_float4(float4(length(f_total_0), 0.0f, 0.0f, 0.0f)) ;
    if(((&kernelContext_68)->params_0->load_model_0) == 1U)
    {
        uint _S533 = l_vec_0(&kernelContext_68);
        uint _S534 = _S533 + 2U * k_18;
        *((&kernelContext_68)->loads_0+_S534) = packed_float4(float4(f_total_0, 0.0f)) ;
        *((&kernelContext_68)->loads_0+(_S534 + 1U)) = packed_float4(float4(t_total_0, 0.0f)) ;
    }
    return;
}

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], atomic<uint> device* atom_2 [[buffer(5)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]], packed_float4 device* geometry_2 [[buffer(21)]], packed_float4 device* shots_2 [[buffer(22)]], packed_float4 device* lp_2 [[buffer(24)]], packed_float4 device* ext_3 [[buffer(23)]], packed_float4 device* lp_sides_2 [[buffer(28)]], uint device* engine_counters_2 [[buffer(26)]], packed_float4 device* engine_contacts_2 [[buffer(25)]], packed_float4 device* cs_anchors_2 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_69;
    (&kernelContext_69)->params_0 = params_2;
    (&kernelContext_69)->work_0 = work_2;
    (&kernelContext_69)->frames_0 = frames_2;
    (&kernelContext_69)->bodies_0 = bodies_2;
    (&kernelContext_69)->chunks_0 = chunks_2;
    (&kernelContext_69)->sort_0 = sort_2;
    (&kernelContext_69)->atom_0 = atom_2;
    (&kernelContext_69)->manifolds_0 = manifolds_2;
    (&kernelContext_69)->loads_0 = loads_2;
    (&kernelContext_69)->bonds_0 = bonds_2;
    (&kernelContext_69)->args_0 = args_2;
    (&kernelContext_69)->info_0 = info_2;
    (&kernelContext_69)->scan_0 = scan_2;
    (&kernelContext_69)->parts_0 = parts_2;
    (&kernelContext_69)->colliders_0 = colliders_2;
    (&kernelContext_69)->collider_info_0 = collider_info_2;
    (&kernelContext_69)->view_chunks_0 = view_chunks_2;
    (&kernelContext_69)->view_0 = view_2;
    (&kernelContext_69)->objects_0 = objects_2;
    (&kernelContext_69)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_69)->lines_0 = lines_2;
    (&kernelContext_69)->geometry_0 = geometry_2;
    (&kernelContext_69)->shots_0 = shots_2;
    (&kernelContext_69)->lp_0 = lp_2;
    (&kernelContext_69)->ext_0 = ext_3;
    (&kernelContext_69)->lp_sides_0 = lp_sides_2;
    (&kernelContext_69)->engine_counters_0 = engine_counters_2;
    (&kernelContext_69)->engine_contacts_0 = engine_contacts_2;
    (&kernelContext_69)->cs_anchors_0 = cs_anchors_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_69)->g_scan_0 = &g_scan_2;
    uint i_2 = id_1.x;
    bool _S535;
    if(i_2 >= (params_2->bond_count_0))
    {
        _S535 = true;
    }
    else
    {
        uint _S536 = w_alive_0(&kernelContext_69);
        _S535 = (*((&kernelContext_69)->work_0+(_S536 + i_2))) == 0U;
    }
    if(_S535)
    {
        return;
    }
    uint4 _S537 = uint4(*((&kernelContext_69)->bonds_0+i_2)) ;
    uint _S538 = _S537.x;
    if((max((float4(*((&kernelContext_69)->loads_0+_S538)) ).x, (float4(*((&kernelContext_69)->loads_0+_S537.y)) ).x)) > (as_type<float>((_S537.z))))
    {
        uint _S539 = w_alive_0(&kernelContext_69);
        *((&kernelContext_69)->work_0+(_S539 + i_2)) = 0U;
        uint _S540 = a_dirty_0(&kernelContext_69);
        uint _S541 = body_of_0(_S538, &kernelContext_69);
        uint _S542 = atomic_fetch_or_explicit((&kernelContext_69)->atom_0+(_S540 + _S541), 1U, memory_order_relaxed);
        uint _S543 = a_counters_1(&kernelContext_69);
        uint _S544 = atomic_fetch_add_explicit((&kernelContext_69)->atom_0+_S543, 1U, memory_order_relaxed);
        uint _S545 = atomic_fetch_add_explicit((&kernelContext_69)->atom_0+(_S543 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint groups_of_0(uint n_3, uint size_0)
{
    uint _S546 = (n_3 + size_0 - 1U) / size_0;
    return _S546;
}

void set_args_0(uint at_4, uint x_3, KernelContext_0 thread* kernelContext_70)
{
    uint _S547 = 3U * at_4;
    *(kernelContext_70->args_0+_S547) = x_3;
    *(kernelContext_70->args_0+(_S547 + 1U)) = 1U;
    *(kernelContext_70->args_0+(_S547 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], packed_float4 device* lp_3 [[buffer(24)]], packed_float4 device* ext_4 [[buffer(23)]], packed_float4 device* lp_sides_3 [[buffer(28)]], uint device* engine_counters_3 [[buffer(26)]], packed_float4 device* engine_contacts_3 [[buffer(25)]], packed_float4 device* cs_anchors_3 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_71;
    (&kernelContext_71)->params_0 = params_3;
    (&kernelContext_71)->work_0 = work_3;
    (&kernelContext_71)->frames_0 = frames_3;
    (&kernelContext_71)->bodies_0 = bodies_3;
    (&kernelContext_71)->chunks_0 = chunks_3;
    (&kernelContext_71)->sort_0 = sort_3;
    (&kernelContext_71)->atom_0 = atom_3;
    (&kernelContext_71)->manifolds_0 = manifolds_3;
    (&kernelContext_71)->loads_0 = loads_3;
    (&kernelContext_71)->bonds_0 = bonds_3;
    (&kernelContext_71)->args_0 = args_3;
    (&kernelContext_71)->info_0 = info_3;
    (&kernelContext_71)->scan_0 = scan_3;
    (&kernelContext_71)->parts_0 = parts_3;
    (&kernelContext_71)->colliders_0 = colliders_3;
    (&kernelContext_71)->collider_info_0 = collider_info_3;
    (&kernelContext_71)->view_chunks_0 = view_chunks_3;
    (&kernelContext_71)->view_0 = view_3;
    (&kernelContext_71)->objects_0 = objects_3;
    (&kernelContext_71)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_71)->lines_0 = lines_3;
    (&kernelContext_71)->geometry_0 = geometry_3;
    (&kernelContext_71)->shots_0 = shots_3;
    (&kernelContext_71)->lp_0 = lp_3;
    (&kernelContext_71)->ext_0 = ext_4;
    (&kernelContext_71)->lp_sides_0 = lp_sides_3;
    (&kernelContext_71)->engine_counters_0 = engine_counters_3;
    (&kernelContext_71)->engine_contacts_0 = engine_contacts_3;
    (&kernelContext_71)->cs_anchors_0 = cs_anchors_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_71)->g_scan_0 = &g_scan_3;
    uint _S548 = a_counters_1(&kernelContext_71);
    uint _S549 = atomic_load_explicit((&kernelContext_71)->atom_0+_S548, memory_order_relaxed);
    bool run_0 = _S549 != 0U;
    atomic_store_explicit((&kernelContext_71)->atom_0+_S548, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_71)->atom_0+(_S548 + 5U + (&kernelContext_71)->params_0->parity_0), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S550 = atomic_fetch_add_explicit((&kernelContext_71)->atom_0+(_S548 + 4U), 1U, memory_order_relaxed);
    }
    uint _S551;
    if(run_0)
    {
        uint _S552 = groups_of_0((&kernelContext_71)->params_0->chunk_count_0, 64U);
        _S551 = _S552;
    }
    else
    {
        _S551 = 0U;
    }
    set_args_0(0U, _S551, &kernelContext_71);
    if(run_0)
    {
        uint _S553 = groups_of_0((&kernelContext_71)->params_0->bond_count_0, 64U);
        _S551 = _S553;
    }
    else
    {
        _S551 = 0U;
    }
    set_args_0(1U, _S551, &kernelContext_71);
    if(run_0)
    {
        uint _S554 = groups_of_0((&kernelContext_71)->params_0->body_count_0, 64U);
        _S551 = _S554;
    }
    else
    {
        _S551 = 0U;
    }
    set_args_0(2U, _S551, &kernelContext_71);
    if(run_0)
    {
        uint _S555 = groups_of_0((&kernelContext_71)->params_0->chunk_count_0, 1024U);
        _S551 = _S555;
    }
    else
    {
        _S551 = 0U;
    }
    set_args_0(3U, _S551, &kernelContext_71);
    if(run_0)
    {
        uint _S556 = groups_of_0((&kernelContext_71)->params_0->body_count_0, 1024U);
        _S551 = _S556;
    }
    else
    {
        _S551 = 0U;
    }
    set_args_0(4U, _S551, &kernelContext_71);
    int _S557;
    if(run_0)
    {
        _S557 = int(1);
    }
    else
    {
        _S557 = int(0);
    }
    set_args_0(5U, uint(_S557), &kernelContext_71);
    return;
}

bool dirty_0(uint body_1, KernelContext_0 thread* kernelContext_72)
{
    uint _S558 = a_dirty_0(kernelContext_72);
    uint _S559 = atomic_load_explicit(kernelContext_72->atom_0+(_S558 + body_1), memory_order_relaxed);
    return _S559 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_73)
{
    return kernelContext_73->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_74)
{
    return 2U * kernelContext_74->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], packed_float4 device* lp_4 [[buffer(24)]], packed_float4 device* ext_5 [[buffer(23)]], packed_float4 device* lp_sides_4 [[buffer(28)]], uint device* engine_counters_4 [[buffer(26)]], packed_float4 device* engine_contacts_4 [[buffer(25)]], packed_float4 device* cs_anchors_4 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_75;
    (&kernelContext_75)->params_0 = params_4;
    (&kernelContext_75)->work_0 = work_4;
    (&kernelContext_75)->frames_0 = frames_4;
    (&kernelContext_75)->bodies_0 = bodies_4;
    (&kernelContext_75)->chunks_0 = chunks_4;
    (&kernelContext_75)->sort_0 = sort_4;
    (&kernelContext_75)->atom_0 = atom_4;
    (&kernelContext_75)->manifolds_0 = manifolds_4;
    (&kernelContext_75)->loads_0 = loads_4;
    (&kernelContext_75)->bonds_0 = bonds_4;
    (&kernelContext_75)->args_0 = args_4;
    (&kernelContext_75)->info_0 = info_4;
    (&kernelContext_75)->scan_0 = scan_4;
    (&kernelContext_75)->parts_0 = parts_4;
    (&kernelContext_75)->colliders_0 = colliders_4;
    (&kernelContext_75)->collider_info_0 = collider_info_4;
    (&kernelContext_75)->view_chunks_0 = view_chunks_4;
    (&kernelContext_75)->view_0 = view_4;
    (&kernelContext_75)->objects_0 = objects_4;
    (&kernelContext_75)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_75)->lines_0 = lines_4;
    (&kernelContext_75)->geometry_0 = geometry_4;
    (&kernelContext_75)->shots_0 = shots_4;
    (&kernelContext_75)->lp_0 = lp_4;
    (&kernelContext_75)->ext_0 = ext_5;
    (&kernelContext_75)->lp_sides_0 = lp_sides_4;
    (&kernelContext_75)->engine_counters_0 = engine_counters_4;
    (&kernelContext_75)->engine_contacts_0 = engine_contacts_4;
    (&kernelContext_75)->cs_anchors_0 = cs_anchors_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_75)->g_scan_0 = &g_scan_4;
    uint c_6 = id_3.x;
    bool _S560;
    if(c_6 >= (params_4->chunk_count_0))
    {
        _S560 = true;
    }
    else
    {
        uint _S561 = body_of_0(c_6, &kernelContext_75);
        bool _S562 = dirty_0(_S561, &kernelContext_75);
        _S560 = !_S562;
    }
    if(_S560)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_75)->atom_0+(a_parent_0() + c_6), c_6, memory_order_relaxed);
    uint _S563 = a_count_0(&kernelContext_75);
    atomic_store_explicit((&kernelContext_75)->atom_0+(_S563 + c_6), 0U, memory_order_relaxed);
    uint _S564 = a_anchor_0(&kernelContext_75);
    atomic_store_explicit((&kernelContext_75)->atom_0+(_S564 + c_6), 0U, memory_order_relaxed);
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
        uint _S565 = a_sums_0(&kernelContext_75);
        atomic_store_explicit((&kernelContext_75)->atom_0+(_S565 + 20U * c_6 + w_3), 0U, memory_order_relaxed);
        w_3 = w_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], packed_float4 device* lp_5 [[buffer(24)]], packed_float4 device* ext_6 [[buffer(23)]], packed_float4 device* lp_sides_5 [[buffer(28)]], uint device* engine_counters_5 [[buffer(26)]], packed_float4 device* engine_contacts_5 [[buffer(25)]], packed_float4 device* cs_anchors_5 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_76;
    (&kernelContext_76)->params_0 = params_5;
    (&kernelContext_76)->work_0 = work_5;
    (&kernelContext_76)->frames_0 = frames_5;
    (&kernelContext_76)->bodies_0 = bodies_5;
    (&kernelContext_76)->chunks_0 = chunks_5;
    (&kernelContext_76)->sort_0 = sort_5;
    (&kernelContext_76)->atom_0 = atom_5;
    (&kernelContext_76)->manifolds_0 = manifolds_5;
    (&kernelContext_76)->loads_0 = loads_5;
    (&kernelContext_76)->bonds_0 = bonds_5;
    (&kernelContext_76)->args_0 = args_5;
    (&kernelContext_76)->info_0 = info_5;
    (&kernelContext_76)->scan_0 = scan_5;
    (&kernelContext_76)->parts_0 = parts_5;
    (&kernelContext_76)->colliders_0 = colliders_5;
    (&kernelContext_76)->collider_info_0 = collider_info_5;
    (&kernelContext_76)->view_chunks_0 = view_chunks_5;
    (&kernelContext_76)->view_0 = view_5;
    (&kernelContext_76)->objects_0 = objects_5;
    (&kernelContext_76)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_76)->lines_0 = lines_5;
    (&kernelContext_76)->geometry_0 = geometry_5;
    (&kernelContext_76)->shots_0 = shots_5;
    (&kernelContext_76)->lp_0 = lp_5;
    (&kernelContext_76)->ext_0 = ext_6;
    (&kernelContext_76)->lp_sides_0 = lp_sides_5;
    (&kernelContext_76)->engine_counters_0 = engine_counters_5;
    (&kernelContext_76)->engine_contacts_0 = engine_contacts_5;
    (&kernelContext_76)->cs_anchors_0 = cs_anchors_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_76)->g_scan_0 = &g_scan_5;
    uint s_7 = id_4.x;
    bool _S566;
    if(s_7 >= (params_5->body_count_0))
    {
        _S566 = true;
    }
    else
    {
        bool _S567 = dirty_0(s_7, &kernelContext_76);
        _S566 = !_S567;
    }
    if(_S566)
    {
        return;
    }
    uint _S568 = a_best_count_0(&kernelContext_76);
    atomic_store_explicit((&kernelContext_76)->atom_0+(_S568 + s_7), 0U, memory_order_relaxed);
    uint _S569 = a_best_root_0(&kernelContext_76);
    atomic_store_explicit((&kernelContext_76)->atom_0+(_S569 + s_7), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_4, KernelContext_0 thread* kernelContext_77)
{
    uint _S570 = a_parent_0();
    uint _S571 = atomic_load_explicit(kernelContext_77->atom_0+(_S570 + x_4), memory_order_relaxed);
    uint p_11 = _S571;
    uint _S572 = x_4;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S573;
        if(guard_0 < 1048576U)
        {
            _S573 = p_11 != _S572;
        }
        else
        {
            _S573 = false;
        }
        if(_S573)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_77->atom_0+(_S570 + p_11), memory_order_relaxed);
        if(gp_0 != p_11)
        {
            uint _S574;
            {
            uint _metal_cas_comparand = p_11;
            atomic_compare_exchange_weak_explicit(kernelContext_77->atom_0+(_S570 + _S572), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S574 = _metal_cas_comparand;
            }
        }
        uint _S575 = guard_0 + 1U;
        _S572 = p_11;
        p_11 = gp_0;
        guard_0 = _S575;
    }
    return _S572;
}

[[kernel]] void fr_uf_union(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], packed_float4 device* lp_6 [[buffer(24)]], packed_float4 device* ext_7 [[buffer(23)]], packed_float4 device* lp_sides_6 [[buffer(28)]], uint device* engine_counters_6 [[buffer(26)]], packed_float4 device* engine_contacts_6 [[buffer(25)]], packed_float4 device* cs_anchors_6 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_78;
    (&kernelContext_78)->params_0 = params_6;
    (&kernelContext_78)->work_0 = work_6;
    (&kernelContext_78)->frames_0 = frames_6;
    (&kernelContext_78)->bodies_0 = bodies_6;
    (&kernelContext_78)->chunks_0 = chunks_6;
    (&kernelContext_78)->sort_0 = sort_6;
    (&kernelContext_78)->atom_0 = atom_6;
    (&kernelContext_78)->manifolds_0 = manifolds_6;
    (&kernelContext_78)->loads_0 = loads_6;
    (&kernelContext_78)->bonds_0 = bonds_6;
    (&kernelContext_78)->args_0 = args_6;
    (&kernelContext_78)->info_0 = info_6;
    (&kernelContext_78)->scan_0 = scan_6;
    (&kernelContext_78)->parts_0 = parts_6;
    (&kernelContext_78)->colliders_0 = colliders_6;
    (&kernelContext_78)->collider_info_0 = collider_info_6;
    (&kernelContext_78)->view_chunks_0 = view_chunks_6;
    (&kernelContext_78)->view_0 = view_6;
    (&kernelContext_78)->objects_0 = objects_6;
    (&kernelContext_78)->shot_bodies_0 = shot_bodies_6;
    (&kernelContext_78)->lines_0 = lines_6;
    (&kernelContext_78)->geometry_0 = geometry_6;
    (&kernelContext_78)->shots_0 = shots_6;
    (&kernelContext_78)->lp_0 = lp_6;
    (&kernelContext_78)->ext_0 = ext_7;
    (&kernelContext_78)->lp_sides_0 = lp_sides_6;
    (&kernelContext_78)->engine_counters_0 = engine_counters_6;
    (&kernelContext_78)->engine_contacts_0 = engine_contacts_6;
    (&kernelContext_78)->cs_anchors_0 = cs_anchors_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_78)->g_scan_0 = &g_scan_6;
    uint i_3 = id_5.x;
    bool _S576;
    if(i_3 >= (params_6->bond_count_0))
    {
        _S576 = true;
    }
    else
    {
        uint _S577 = w_alive_0(&kernelContext_78);
        _S576 = (*((&kernelContext_78)->work_0+(_S577 + i_3))) == 0U;
    }
    if(_S576)
    {
        return;
    }
    uint4 _S578 = uint4(*((&kernelContext_78)->bonds_0+i_3)) ;
    uint _S579 = _S578.x;
    uint _S580 = body_of_0(_S579, &kernelContext_78);
    bool _S581 = dirty_0(_S580, &kernelContext_78);
    if(!_S581)
    {
        return;
    }
    uint _S582 = _S578.y;
    uint a_6 = _S579;
    uint b_24 = _S582;
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
        uint _S583 = find_0(a_6, &kernelContext_78);
        uint _S584 = find_0(b_24, &kernelContext_78);
        if(_S583 == _S584)
        {
            return;
        }
        uint _S585 = max(_S583, _S584);
        uint _S586 = min(_S583, _S584);
        uint was_0;
        {
        uint _metal_cas_comparand = _S585;
        atomic_compare_exchange_weak_explicit((&kernelContext_78)->atom_0+(a_parent_0() + _S585), &_metal_cas_comparand, _S586, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S585)
        {
            return;
        }
        uint _S587 = guard_1 + 1U;
        a_6 = _S585;
        b_24 = _S586;
        guard_1 = _S587;
    }
    uint _S588 = a_counters_1(&kernelContext_78);
    uint _S589 = atomic_fetch_or_explicit((&kernelContext_78)->atom_0+(_S588 + 3U), 4U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], packed_float4 device* lp_7 [[buffer(24)]], packed_float4 device* ext_8 [[buffer(23)]], packed_float4 device* lp_sides_7 [[buffer(28)]], uint device* engine_counters_7 [[buffer(26)]], packed_float4 device* engine_contacts_7 [[buffer(25)]], packed_float4 device* cs_anchors_7 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_79;
    (&kernelContext_79)->params_0 = params_7;
    (&kernelContext_79)->work_0 = work_7;
    (&kernelContext_79)->frames_0 = frames_7;
    (&kernelContext_79)->bodies_0 = bodies_7;
    (&kernelContext_79)->chunks_0 = chunks_7;
    (&kernelContext_79)->sort_0 = sort_7;
    (&kernelContext_79)->atom_0 = atom_7;
    (&kernelContext_79)->manifolds_0 = manifolds_7;
    (&kernelContext_79)->loads_0 = loads_7;
    (&kernelContext_79)->bonds_0 = bonds_7;
    (&kernelContext_79)->args_0 = args_7;
    (&kernelContext_79)->info_0 = info_7;
    (&kernelContext_79)->scan_0 = scan_7;
    (&kernelContext_79)->parts_0 = parts_7;
    (&kernelContext_79)->colliders_0 = colliders_7;
    (&kernelContext_79)->collider_info_0 = collider_info_7;
    (&kernelContext_79)->view_chunks_0 = view_chunks_7;
    (&kernelContext_79)->view_0 = view_7;
    (&kernelContext_79)->objects_0 = objects_7;
    (&kernelContext_79)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_79)->lines_0 = lines_7;
    (&kernelContext_79)->geometry_0 = geometry_7;
    (&kernelContext_79)->shots_0 = shots_7;
    (&kernelContext_79)->lp_0 = lp_7;
    (&kernelContext_79)->ext_0 = ext_8;
    (&kernelContext_79)->lp_sides_0 = lp_sides_7;
    (&kernelContext_79)->engine_counters_0 = engine_counters_7;
    (&kernelContext_79)->engine_contacts_0 = engine_contacts_7;
    (&kernelContext_79)->cs_anchors_0 = cs_anchors_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_79)->g_scan_0 = &g_scan_7;
    uint c_7 = id_6.x;
    bool _S590;
    if(c_7 >= (params_7->chunk_count_0))
    {
        _S590 = true;
    }
    else
    {
        uint _S591 = body_of_0(c_7, &kernelContext_79);
        bool _S592 = dirty_0(_S591, &kernelContext_79);
        _S590 = !_S592;
    }
    if(_S590)
    {
        return;
    }
    uint _S593 = find_0(c_7, &kernelContext_79);
    uint _S594 = w_label_0(&kernelContext_79);
    *((&kernelContext_79)->work_0+(_S594 + c_7)) = _S593;
    uint _S595 = a_count_0(&kernelContext_79);
    uint _S596 = atomic_fetch_add_explicit((&kernelContext_79)->atom_0+(_S595 + _S593), 1U, memory_order_relaxed);
    if(((&kernelContext_79)->info_0[c_7] & 1U) != 0U)
    {
        uint _S597 = a_anchor_0(&kernelContext_79);
        uint _S598 = atomic_fetch_or_explicit((&kernelContext_79)->atom_0+(_S597 + _S593), 1U, memory_order_relaxed);
    }
    uint _S599 = 4U * c_7;
    float4 _S600 = float4(*((&kernelContext_79)->chunks_0+_S599)) ;
    float4 _S601 = float4(*((&kernelContext_79)->chunks_0+(_S599 + 1U))) ;
    float4 _S602 = float4(*((&kernelContext_79)->chunks_0+(_S599 + 2U))) ;
    float m_2 = _S600.w;
    float3 d_18 = _S600.xyz - (float4(*((&kernelContext_79)->chunks_0+4U * _S593)) ).xyz;
    float dd_0 = dot(d_18, d_18);
    uint _S603 = a_sums_0(&kernelContext_79);
    uint at_5 = _S603 + 20U * _S593;
    add64_0(at_5, m_2, (&kernelContext_79)->params_0->scale_m_0, &kernelContext_79);
    float _S604 = d_18.x;
    float _S605 = m_2 * _S604;
    add64_0(at_5 + 2U, _S605, (&kernelContext_79)->params_0->scale_s_0, &kernelContext_79);
    float _S606 = d_18.y;
    float _S607 = m_2 * _S606;
    add64_0(at_5 + 4U, _S607, (&kernelContext_79)->params_0->scale_s_0, &kernelContext_79);
    float _S608 = d_18.z;
    add64_0(at_5 + 6U, m_2 * _S608, (&kernelContext_79)->params_0->scale_s_0, &kernelContext_79);
    add64_0(at_5 + 8U, _S601.x + m_2 * (dd_0 - _S604 * _S604), (&kernelContext_79)->params_0->scale_j_0, &kernelContext_79);
    add64_0(at_5 + 10U, _S601.y + m_2 * (dd_0 - _S606 * _S606), (&kernelContext_79)->params_0->scale_j_0, &kernelContext_79);
    add64_0(at_5 + 12U, _S601.z + m_2 * (dd_0 - _S608 * _S608), (&kernelContext_79)->params_0->scale_j_0, &kernelContext_79);
    add64_0(at_5 + 14U, _S601.w - _S605 * _S606, (&kernelContext_79)->params_0->scale_j_0, &kernelContext_79);
    add64_0(at_5 + 16U, _S602.x - _S605 * _S608, (&kernelContext_79)->params_0->scale_j_0, &kernelContext_79);
    add64_0(at_5 + 18U, _S602.y - _S607 * _S608, (&kernelContext_79)->params_0->scale_j_0, &kernelContext_79);
    return;
}

bool is_root_0(uint c_8, KernelContext_0 thread* kernelContext_80)
{
    bool _S609;
    if(c_8 < (kernelContext_80->params_0->chunk_count_0))
    {
        uint _S610 = body_of_0(c_8, kernelContext_80);
        bool _S611 = dirty_0(_S610, kernelContext_80);
        _S609 = _S611;
    }
    else
    {
        _S609 = false;
    }
    if(_S609)
    {
        uint _S612 = w_label_0(kernelContext_80);
        _S609 = (*(kernelContext_80->work_0+(_S612 + c_8))) == c_8;
    }
    else
    {
        _S609 = false;
    }
    return _S609;
}

[[kernel]] void fr_best_count(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], packed_float4 device* lp_8 [[buffer(24)]], packed_float4 device* ext_9 [[buffer(23)]], packed_float4 device* lp_sides_8 [[buffer(28)]], uint device* engine_counters_8 [[buffer(26)]], packed_float4 device* engine_contacts_8 [[buffer(25)]], packed_float4 device* cs_anchors_8 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_81;
    (&kernelContext_81)->params_0 = params_8;
    (&kernelContext_81)->work_0 = work_8;
    (&kernelContext_81)->frames_0 = frames_8;
    (&kernelContext_81)->bodies_0 = bodies_8;
    (&kernelContext_81)->chunks_0 = chunks_8;
    (&kernelContext_81)->sort_0 = sort_8;
    (&kernelContext_81)->atom_0 = atom_8;
    (&kernelContext_81)->manifolds_0 = manifolds_8;
    (&kernelContext_81)->loads_0 = loads_8;
    (&kernelContext_81)->bonds_0 = bonds_8;
    (&kernelContext_81)->args_0 = args_8;
    (&kernelContext_81)->info_0 = info_8;
    (&kernelContext_81)->scan_0 = scan_8;
    (&kernelContext_81)->parts_0 = parts_8;
    (&kernelContext_81)->colliders_0 = colliders_8;
    (&kernelContext_81)->collider_info_0 = collider_info_8;
    (&kernelContext_81)->view_chunks_0 = view_chunks_8;
    (&kernelContext_81)->view_0 = view_8;
    (&kernelContext_81)->objects_0 = objects_8;
    (&kernelContext_81)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_81)->lines_0 = lines_8;
    (&kernelContext_81)->geometry_0 = geometry_8;
    (&kernelContext_81)->shots_0 = shots_8;
    (&kernelContext_81)->lp_0 = lp_8;
    (&kernelContext_81)->ext_0 = ext_9;
    (&kernelContext_81)->lp_sides_0 = lp_sides_8;
    (&kernelContext_81)->engine_counters_0 = engine_counters_8;
    (&kernelContext_81)->engine_contacts_0 = engine_contacts_8;
    (&kernelContext_81)->cs_anchors_0 = cs_anchors_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_81)->g_scan_0 = &g_scan_8;
    uint c_9 = id_7.x;
    bool _S613 = is_root_0(c_9, &kernelContext_81);
    if(!_S613)
    {
        return;
    }
    uint _S614 = a_best_count_0(&kernelContext_81);
    uint _S615 = body_of_0(c_9, &kernelContext_81);
    atomic<uint> device* _S616 = (&kernelContext_81)->atom_0+(_S614 + _S615);
    uint _S617 = a_count_0(&kernelContext_81);
    uint _S618 = atomic_load_explicit((&kernelContext_81)->atom_0+(_S617 + c_9), memory_order_relaxed);
    uint _S619 = atomic_fetch_max_explicit(_S616, _S618, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], packed_float4 device* lp_9 [[buffer(24)]], packed_float4 device* ext_10 [[buffer(23)]], packed_float4 device* lp_sides_9 [[buffer(28)]], uint device* engine_counters_9 [[buffer(26)]], packed_float4 device* engine_contacts_9 [[buffer(25)]], packed_float4 device* cs_anchors_9 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_82;
    (&kernelContext_82)->params_0 = params_9;
    (&kernelContext_82)->work_0 = work_9;
    (&kernelContext_82)->frames_0 = frames_9;
    (&kernelContext_82)->bodies_0 = bodies_9;
    (&kernelContext_82)->chunks_0 = chunks_9;
    (&kernelContext_82)->sort_0 = sort_9;
    (&kernelContext_82)->atom_0 = atom_9;
    (&kernelContext_82)->manifolds_0 = manifolds_9;
    (&kernelContext_82)->loads_0 = loads_9;
    (&kernelContext_82)->bonds_0 = bonds_9;
    (&kernelContext_82)->args_0 = args_9;
    (&kernelContext_82)->info_0 = info_9;
    (&kernelContext_82)->scan_0 = scan_9;
    (&kernelContext_82)->parts_0 = parts_9;
    (&kernelContext_82)->colliders_0 = colliders_9;
    (&kernelContext_82)->collider_info_0 = collider_info_9;
    (&kernelContext_82)->view_chunks_0 = view_chunks_9;
    (&kernelContext_82)->view_0 = view_9;
    (&kernelContext_82)->objects_0 = objects_9;
    (&kernelContext_82)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_82)->lines_0 = lines_9;
    (&kernelContext_82)->geometry_0 = geometry_9;
    (&kernelContext_82)->shots_0 = shots_9;
    (&kernelContext_82)->lp_0 = lp_9;
    (&kernelContext_82)->ext_0 = ext_10;
    (&kernelContext_82)->lp_sides_0 = lp_sides_9;
    (&kernelContext_82)->engine_counters_0 = engine_counters_9;
    (&kernelContext_82)->engine_contacts_0 = engine_contacts_9;
    (&kernelContext_82)->cs_anchors_0 = cs_anchors_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_82)->g_scan_0 = &g_scan_9;
    uint c_10 = id_8.x;
    bool _S620 = is_root_0(c_10, &kernelContext_82);
    if(!_S620)
    {
        return;
    }
    uint _S621 = body_of_0(c_10, &kernelContext_82);
    uint _S622 = a_count_0(&kernelContext_82);
    uint _S623 = atomic_load_explicit((&kernelContext_82)->atom_0+(_S622 + c_10), memory_order_relaxed);
    uint _S624 = a_best_count_0(&kernelContext_82);
    uint _S625 = atomic_load_explicit((&kernelContext_82)->atom_0+(_S624 + _S621), memory_order_relaxed);
    if(_S623 == _S625)
    {
        uint _S626 = a_best_root_0(&kernelContext_82);
        uint _S627 = atomic_fetch_min_explicit((&kernelContext_82)->atom_0+(_S626 + _S621), c_10, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], packed_float4 device* lp_10 [[buffer(24)]], packed_float4 device* ext_11 [[buffer(23)]], packed_float4 device* lp_sides_10 [[buffer(28)]], uint device* engine_counters_10 [[buffer(26)]], packed_float4 device* engine_contacts_10 [[buffer(25)]], packed_float4 device* cs_anchors_10 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_83;
    (&kernelContext_83)->params_0 = params_10;
    (&kernelContext_83)->work_0 = work_10;
    (&kernelContext_83)->frames_0 = frames_10;
    (&kernelContext_83)->bodies_0 = bodies_10;
    (&kernelContext_83)->chunks_0 = chunks_10;
    (&kernelContext_83)->sort_0 = sort_10;
    (&kernelContext_83)->atom_0 = atom_10;
    (&kernelContext_83)->manifolds_0 = manifolds_10;
    (&kernelContext_83)->loads_0 = loads_10;
    (&kernelContext_83)->bonds_0 = bonds_10;
    (&kernelContext_83)->args_0 = args_10;
    (&kernelContext_83)->info_0 = info_10;
    (&kernelContext_83)->scan_0 = scan_10;
    (&kernelContext_83)->parts_0 = parts_10;
    (&kernelContext_83)->colliders_0 = colliders_10;
    (&kernelContext_83)->collider_info_0 = collider_info_10;
    (&kernelContext_83)->view_chunks_0 = view_chunks_10;
    (&kernelContext_83)->view_0 = view_10;
    (&kernelContext_83)->objects_0 = objects_10;
    (&kernelContext_83)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_83)->lines_0 = lines_10;
    (&kernelContext_83)->geometry_0 = geometry_10;
    (&kernelContext_83)->shots_0 = shots_10;
    (&kernelContext_83)->lp_0 = lp_10;
    (&kernelContext_83)->ext_0 = ext_11;
    (&kernelContext_83)->lp_sides_0 = lp_sides_10;
    (&kernelContext_83)->engine_counters_0 = engine_counters_10;
    (&kernelContext_83)->engine_contacts_0 = engine_contacts_10;
    (&kernelContext_83)->cs_anchors_0 = cs_anchors_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_83)->g_scan_0 = &g_scan_10;
    uint c_11 = id_9.x;
    if(c_11 >= (params_10->chunk_count_0))
    {
        return;
    }
    uint _S628 = w_new_0(&kernelContext_83);
    uint device* _S629 = (&kernelContext_83)->work_0+(_S628 + c_11);
    bool _S630 = is_root_0(c_11, &kernelContext_83);
    bool _S631;
    if(_S630)
    {
        uint _S632 = a_best_root_0(&kernelContext_83);
        uint _S633 = body_of_0(c_11, &kernelContext_83);
        uint _S634 = atomic_load_explicit((&kernelContext_83)->atom_0+(_S632 + _S633), memory_order_relaxed);
        _S631 = _S634 != c_11;
    }
    else
    {
        _S631 = false;
    }
    uint _S635;
    if(_S631)
    {
        _S635 = 1U;
    }
    else
    {
        _S635 = 0U;
    }
    *_S629 = _S635;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], packed_float4 device* lp_11 [[buffer(24)]], packed_float4 device* ext_12 [[buffer(23)]], packed_float4 device* lp_sides_11 [[buffer(28)]], uint device* engine_counters_11 [[buffer(26)]], packed_float4 device* engine_contacts_11 [[buffer(25)]], packed_float4 device* cs_anchors_11 [[buffer(27)]])
{
    uint step_2;
    uint add_0;
    thread KernelContext_0 kernelContext_84;
    (&kernelContext_84)->params_0 = params_11;
    (&kernelContext_84)->work_0 = work_11;
    (&kernelContext_84)->frames_0 = frames_11;
    (&kernelContext_84)->bodies_0 = bodies_11;
    (&kernelContext_84)->chunks_0 = chunks_11;
    (&kernelContext_84)->sort_0 = sort_11;
    (&kernelContext_84)->atom_0 = atom_11;
    (&kernelContext_84)->manifolds_0 = manifolds_11;
    (&kernelContext_84)->loads_0 = loads_11;
    (&kernelContext_84)->bonds_0 = bonds_11;
    (&kernelContext_84)->args_0 = args_11;
    (&kernelContext_84)->info_0 = info_11;
    (&kernelContext_84)->scan_0 = scan_11;
    (&kernelContext_84)->parts_0 = parts_11;
    (&kernelContext_84)->colliders_0 = colliders_11;
    (&kernelContext_84)->collider_info_0 = collider_info_11;
    (&kernelContext_84)->view_chunks_0 = view_chunks_11;
    (&kernelContext_84)->view_0 = view_11;
    (&kernelContext_84)->objects_0 = objects_11;
    (&kernelContext_84)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_84)->lines_0 = lines_11;
    (&kernelContext_84)->geometry_0 = geometry_11;
    (&kernelContext_84)->shots_0 = shots_11;
    (&kernelContext_84)->lp_0 = lp_11;
    (&kernelContext_84)->ext_0 = ext_12;
    (&kernelContext_84)->lp_sides_0 = lp_sides_11;
    (&kernelContext_84)->engine_counters_0 = engine_counters_11;
    (&kernelContext_84)->engine_contacts_0 = engine_contacts_11;
    (&kernelContext_84)->cs_anchors_0 = cs_anchors_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_84)->g_scan_0 = &g_scan_11;
    uint t_1 = thread_0.x;
    uint _S636 = group_0.x;
    uint _S637 = 1024U * _S636 + 4U * t_1;
    thread array<uint, int(4)> v_10;
    uint j_3 = 0U;
    uint sum_2 = 0U;
    for(;;)
    {
        if(j_3 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S637 + j_3) < ((&kernelContext_84)->scan_0->n_0))
        {
            step_2 = *((&kernelContext_84)->work_0+((&kernelContext_84)->scan_0->in_base_0 + _S637 + j_3));
        }
        else
        {
            step_2 = 0U;
        }
        v_10[j_3] = step_2;
        uint sum_3 = sum_2 + step_2;
        j_3 = j_3 + 1U;
        sum_2 = sum_3;
    }
    (*(&kernelContext_84)->g_scan_0)[t_1] = sum_2;
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
            add_0 = (*(&kernelContext_84)->g_scan_0)[t_1 - step_2];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_84)->g_scan_0)[t_1] = (*(&kernelContext_84)->g_scan_0)[t_1] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_2 = step_2 << 1U;
    }
    uint _S638 = (*(&kernelContext_84)->g_scan_0)[t_1] - sum_2;
    j_3 = 0U;
    add_0 = _S638;
    for(;;)
    {
        if(j_3 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S637 + j_3) < ((&kernelContext_84)->scan_0->n_0))
        {
            *((&kernelContext_84)->work_0+((&kernelContext_84)->scan_0->out_base_0 + _S637 + j_3)) = add_0;
        }
        uint run_1 = add_0 + v_10[j_3];
        j_3 = j_3 + 1U;
        add_0 = run_1;
    }
    if(t_1 == 255U)
    {
        *((&kernelContext_84)->work_0+((&kernelContext_84)->scan_0->block_base_0 + _S636)) = (*(&kernelContext_84)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], packed_float4 device* lp_12 [[buffer(24)]], packed_float4 device* ext_13 [[buffer(23)]], packed_float4 device* lp_sides_12 [[buffer(28)]], uint device* engine_counters_12 [[buffer(26)]], packed_float4 device* engine_contacts_12 [[buffer(25)]], packed_float4 device* cs_anchors_12 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_85;
    (&kernelContext_85)->params_0 = params_12;
    (&kernelContext_85)->work_0 = work_12;
    (&kernelContext_85)->frames_0 = frames_12;
    (&kernelContext_85)->bodies_0 = bodies_12;
    (&kernelContext_85)->chunks_0 = chunks_12;
    (&kernelContext_85)->sort_0 = sort_12;
    (&kernelContext_85)->atom_0 = atom_12;
    (&kernelContext_85)->manifolds_0 = manifolds_12;
    (&kernelContext_85)->loads_0 = loads_12;
    (&kernelContext_85)->bonds_0 = bonds_12;
    (&kernelContext_85)->args_0 = args_12;
    (&kernelContext_85)->info_0 = info_12;
    (&kernelContext_85)->scan_0 = scan_12;
    (&kernelContext_85)->parts_0 = parts_12;
    (&kernelContext_85)->colliders_0 = colliders_12;
    (&kernelContext_85)->collider_info_0 = collider_info_12;
    (&kernelContext_85)->view_chunks_0 = view_chunks_12;
    (&kernelContext_85)->view_0 = view_12;
    (&kernelContext_85)->objects_0 = objects_12;
    (&kernelContext_85)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_85)->lines_0 = lines_12;
    (&kernelContext_85)->geometry_0 = geometry_12;
    (&kernelContext_85)->shots_0 = shots_12;
    (&kernelContext_85)->lp_0 = lp_12;
    (&kernelContext_85)->ext_0 = ext_13;
    (&kernelContext_85)->lp_sides_0 = lp_sides_12;
    (&kernelContext_85)->engine_counters_0 = engine_counters_12;
    (&kernelContext_85)->engine_contacts_0 = engine_contacts_12;
    (&kernelContext_85)->cs_anchors_0 = cs_anchors_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_85)->g_scan_0 = &g_scan_12;
    uint t_2 = thread_1.x;
    uint blocks_0 = (scan_12->n_0 + 1023U) / 1024U;
    bool _S639 = t_2 < blocks_0;
    uint v_11;
    if(_S639)
    {
        v_11 = *((&kernelContext_85)->work_0+((&kernelContext_85)->scan_0->block_base_0 + t_2));
    }
    else
    {
        v_11 = 0U;
    }
    (*(&kernelContext_85)->g_scan_0)[t_2] = v_11;
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
            add_1 = (*(&kernelContext_85)->g_scan_0)[t_2 - step_3];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_85)->g_scan_0)[t_2] = (*(&kernelContext_85)->g_scan_0)[t_2] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_3 = step_3 << 1U;
    }
    if(_S639)
    {
        *((&kernelContext_85)->work_0+((&kernelContext_85)->scan_0->block_base_0 + t_2)) = (*(&kernelContext_85)->g_scan_0)[t_2] - v_11;
    }
    if(t_2 == 1023U)
    {
        *((&kernelContext_85)->work_0+((&kernelContext_85)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_85)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], packed_float4 device* lp_13 [[buffer(24)]], packed_float4 device* ext_14 [[buffer(23)]], packed_float4 device* lp_sides_13 [[buffer(28)]], uint device* engine_counters_13 [[buffer(26)]], packed_float4 device* engine_contacts_13 [[buffer(25)]], packed_float4 device* cs_anchors_13 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_86;
    (&kernelContext_86)->params_0 = params_13;
    (&kernelContext_86)->work_0 = work_13;
    (&kernelContext_86)->frames_0 = frames_13;
    (&kernelContext_86)->bodies_0 = bodies_13;
    (&kernelContext_86)->chunks_0 = chunks_13;
    (&kernelContext_86)->sort_0 = sort_13;
    (&kernelContext_86)->atom_0 = atom_13;
    (&kernelContext_86)->manifolds_0 = manifolds_13;
    (&kernelContext_86)->loads_0 = loads_13;
    (&kernelContext_86)->bonds_0 = bonds_13;
    (&kernelContext_86)->args_0 = args_13;
    (&kernelContext_86)->info_0 = info_13;
    (&kernelContext_86)->scan_0 = scan_13;
    (&kernelContext_86)->parts_0 = parts_13;
    (&kernelContext_86)->colliders_0 = colliders_13;
    (&kernelContext_86)->collider_info_0 = collider_info_13;
    (&kernelContext_86)->view_chunks_0 = view_chunks_13;
    (&kernelContext_86)->view_0 = view_13;
    (&kernelContext_86)->objects_0 = objects_13;
    (&kernelContext_86)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_86)->lines_0 = lines_13;
    (&kernelContext_86)->geometry_0 = geometry_13;
    (&kernelContext_86)->shots_0 = shots_13;
    (&kernelContext_86)->lp_0 = lp_13;
    (&kernelContext_86)->ext_0 = ext_14;
    (&kernelContext_86)->lp_sides_0 = lp_sides_13;
    (&kernelContext_86)->engine_counters_0 = engine_counters_13;
    (&kernelContext_86)->engine_contacts_0 = engine_contacts_13;
    (&kernelContext_86)->cs_anchors_0 = cs_anchors_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_86)->g_scan_0 = &g_scan_13;
    uint i_4 = id_10.x;
    if(i_4 >= (scan_13->n_0))
    {
        return;
    }
    uint device* _S640 = (&kernelContext_86)->work_0+((&kernelContext_86)->scan_0->out_base_0 + i_4);
    *_S640 = *_S640 + *((&kernelContext_86)->work_0+((&kernelContext_86)->scan_0->block_base_0 + i_4 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], packed_float4 device* lp_14 [[buffer(24)]], packed_float4 device* ext_15 [[buffer(23)]], packed_float4 device* lp_sides_14 [[buffer(28)]], uint device* engine_counters_14 [[buffer(26)]], packed_float4 device* engine_contacts_14 [[buffer(25)]], packed_float4 device* cs_anchors_14 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_87;
    (&kernelContext_87)->params_0 = params_14;
    (&kernelContext_87)->work_0 = work_14;
    (&kernelContext_87)->frames_0 = frames_14;
    (&kernelContext_87)->bodies_0 = bodies_14;
    (&kernelContext_87)->chunks_0 = chunks_14;
    (&kernelContext_87)->sort_0 = sort_14;
    (&kernelContext_87)->atom_0 = atom_14;
    (&kernelContext_87)->manifolds_0 = manifolds_14;
    (&kernelContext_87)->loads_0 = loads_14;
    (&kernelContext_87)->bonds_0 = bonds_14;
    (&kernelContext_87)->args_0 = args_14;
    (&kernelContext_87)->info_0 = info_14;
    (&kernelContext_87)->scan_0 = scan_14;
    (&kernelContext_87)->parts_0 = parts_14;
    (&kernelContext_87)->colliders_0 = colliders_14;
    (&kernelContext_87)->collider_info_0 = collider_info_14;
    (&kernelContext_87)->view_chunks_0 = view_chunks_14;
    (&kernelContext_87)->view_0 = view_14;
    (&kernelContext_87)->objects_0 = objects_14;
    (&kernelContext_87)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_87)->lines_0 = lines_14;
    (&kernelContext_87)->geometry_0 = geometry_14;
    (&kernelContext_87)->shots_0 = shots_14;
    (&kernelContext_87)->lp_0 = lp_14;
    (&kernelContext_87)->ext_0 = ext_15;
    (&kernelContext_87)->lp_sides_0 = lp_sides_14;
    (&kernelContext_87)->engine_counters_0 = engine_counters_14;
    (&kernelContext_87)->engine_contacts_0 = engine_contacts_14;
    (&kernelContext_87)->cs_anchors_0 = cs_anchors_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_87)->g_scan_0 = &g_scan_14;
    uint s_8 = id_11.x;
    bool _S641;
    if(s_8 >= (params_14->body_count_0))
    {
        _S641 = true;
    }
    else
    {
        uint _S642 = w_free_0(&kernelContext_87);
        _S641 = (*((&kernelContext_87)->work_0+(_S642 + s_8))) == 0U;
    }
    if(_S641)
    {
        return;
    }
    uint _S643 = w_free_list_0(&kernelContext_87);
    uint _S644 = w_free_rank_0(&kernelContext_87);
    *((&kernelContext_87)->work_0+(_S643 + *((&kernelContext_87)->work_0+(_S644 + s_8)))) = s_8;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_88)
{
    uint _S645 = w_free_list_0(kernelContext_88);
    return _S645 + kernelContext_88->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_89)
{
    uint _S646 = scan_new_blocks_0(kernelContext_89);
    return _S646 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_90)
{
    uint blocks_1 = (kernelContext_90->params_0->body_count_0 + 1023U) / 1024U;
    uint _S647 = scan_free_blocks_0(kernelContext_90);
    return *(kernelContext_90->work_0+(_S647 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], packed_float4 device* lp_15 [[buffer(24)]], packed_float4 device* ext_16 [[buffer(23)]], packed_float4 device* lp_sides_15 [[buffer(28)]], uint device* engine_counters_15 [[buffer(26)]], packed_float4 device* engine_contacts_15 [[buffer(25)]], packed_float4 device* cs_anchors_15 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_91;
    (&kernelContext_91)->params_0 = params_15;
    (&kernelContext_91)->work_0 = work_15;
    (&kernelContext_91)->frames_0 = frames_15;
    (&kernelContext_91)->bodies_0 = bodies_15;
    (&kernelContext_91)->chunks_0 = chunks_15;
    (&kernelContext_91)->sort_0 = sort_15;
    (&kernelContext_91)->atom_0 = atom_15;
    (&kernelContext_91)->manifolds_0 = manifolds_15;
    (&kernelContext_91)->loads_0 = loads_15;
    (&kernelContext_91)->bonds_0 = bonds_15;
    (&kernelContext_91)->args_0 = args_15;
    (&kernelContext_91)->info_0 = info_15;
    (&kernelContext_91)->scan_0 = scan_15;
    (&kernelContext_91)->parts_0 = parts_15;
    (&kernelContext_91)->colliders_0 = colliders_15;
    (&kernelContext_91)->collider_info_0 = collider_info_15;
    (&kernelContext_91)->view_chunks_0 = view_chunks_15;
    (&kernelContext_91)->view_0 = view_15;
    (&kernelContext_91)->objects_0 = objects_15;
    (&kernelContext_91)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_91)->lines_0 = lines_15;
    (&kernelContext_91)->geometry_0 = geometry_15;
    (&kernelContext_91)->shots_0 = shots_15;
    (&kernelContext_91)->lp_0 = lp_15;
    (&kernelContext_91)->ext_0 = ext_16;
    (&kernelContext_91)->lp_sides_0 = lp_sides_15;
    (&kernelContext_91)->engine_counters_0 = engine_counters_15;
    (&kernelContext_91)->engine_contacts_0 = engine_contacts_15;
    (&kernelContext_91)->cs_anchors_0 = cs_anchors_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_91)->g_scan_0 = &g_scan_15;
    uint c_12 = id_12.x;
    bool _S648 = is_root_0(c_12, &kernelContext_91);
    if(!_S648)
    {
        return;
    }
    uint _S649 = body_of_0(c_12, &kernelContext_91);
    uint _S650 = a_best_root_0(&kernelContext_91);
    uint _S651 = atomic_load_explicit((&kernelContext_91)->atom_0+(_S650 + _S649), memory_order_relaxed);
    uint slot_1;
    if(_S651 != c_12)
    {
        uint _S652 = w_rank_0(&kernelContext_91);
        uint device* _S653 = (&kernelContext_91)->work_0+(_S652 + c_12);
        uint k_19 = *_S653;
        uint _S654 = free_total_0(&kernelContext_91);
        if((*_S653) < _S654)
        {
            uint _S655 = w_free_list_0(&kernelContext_91);
            uint device* _S656 = (&kernelContext_91)->work_0+(_S655 + k_19);
            uint slot_2 = *_S656;
            uint _S657 = w_free_0(&kernelContext_91);
            *((&kernelContext_91)->work_0+(_S657 + *_S656)) = 0U;
            uint _S658 = a_counters_1(&kernelContext_91);
            uint _S659 = atomic_fetch_add_explicit((&kernelContext_91)->atom_0+(_S658 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S660 = a_counters_1(&kernelContext_91);
            uint _S661 = atomic_fetch_or_explicit((&kernelContext_91)->atom_0+(_S660 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S649;
        }
    }
    else
    {
        slot_1 = _S649;
    }
    uint _S662 = w_slot_0(&kernelContext_91);
    *((&kernelContext_91)->work_0+(_S662 + c_12)) = slot_1;
    return;
}

float read64_1(uint at_6, float scale_2, KernelContext_0 thread* kernelContext_92)
{
    uint _S663 = atomic_load_explicit(kernelContext_92->atom_0+(at_6 + 1U), memory_order_relaxed);
    int hi_3 = (as_type<int>((_S663)));
    uint lo_3 = atomic_load_explicit(kernelContext_92->atom_0+at_6, memory_order_relaxed);
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
    array<float, int(3)> _S664 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S665 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S666 = { { r2_2.x, r2_2.y, r2_2.z } };
    thread array<array<float, int(3)>, int(3)> a_7;
    a_7[int(0)] = _S664;
    a_7[int(1)] = _S665;
    a_7[int(2)] = _S666;
    array<float, int(3)> _S667 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S668 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S669 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_12;
    v_12[int(0)] = _S667;
    v_12[int(1)] = _S668;
    v_12[int(2)] = _S669;
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
        uint k_20 = 0U;
        for(;;)
        {
            if(k_20 < 3U)
            {
            }
            else
            {
                break;
            }
            bool _S670;
            if(k_20 != p_12)
            {
                _S670 = k_20 != q_14;
            }
            else
            {
                _S670 = false;
            }
            if(_S670)
            {
                float kp_0 = a_7[k_20][p_12];
                float kq_0 = a_7[k_20][q_14];
                float _S671 = c_13 * a_7[k_20][p_12] - s_9 * a_7[k_20][q_14];
                a_7[k_20][p_12] = _S671;
                a_7[p_12][k_20] = _S671;
                float _S672 = s_9 * kp_0 + c_13 * kq_0;
                a_7[k_20][q_14] = _S672;
                a_7[q_14][k_20] = _S672;
            }
            k_20 = k_20 + 1U;
        }
        float _S673 = c_13 * c_13;
        float _S674 = 2.0f * s_9 * c_13 * apq_0;
        float _S675 = s_9 * s_9;
        a_7[p_12][p_12] = _S673 * app_0 - _S674 + _S675 * aqq_0;
        a_7[q_14][q_14] = _S675 * app_0 + _S674 + _S673 * aqq_0;
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
    thread Quat_0 _S676 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S677 = normalized_0(&_S676);
    *rot_0 = _S677;
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
    thread Quat_0 _S678 = axes_1;
    Quat_0 _S679 = quat_mul_0(&_S678, q_15);
    thread Quat_0 _S680 = _S679;
    Quat_0 _S681 = normalized_0(&_S680);
    return _S681;
}

uint stage_0(uint c_14, KernelContext_0 thread* kernelContext_93)
{
    return 2U * kernelContext_93->params_0->body_count_0 + 9U * c_14;
}

float4 quat_vec_0(const Quat_0 thread* q_16)
{
    return float4(q_16->x_0, q_16->y_0, q_16->z_0, q_16->w_0);
}

[[kernel]] void fr_piece(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], packed_float4 device* lp_16 [[buffer(24)]], packed_float4 device* ext_17 [[buffer(23)]], packed_float4 device* lp_sides_16 [[buffer(28)]], uint device* engine_counters_16 [[buffer(26)]], packed_float4 device* engine_contacts_16 [[buffer(25)]], packed_float4 device* cs_anchors_16 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_94;
    (&kernelContext_94)->params_0 = params_16;
    (&kernelContext_94)->work_0 = work_16;
    (&kernelContext_94)->frames_0 = frames_16;
    (&kernelContext_94)->bodies_0 = bodies_16;
    (&kernelContext_94)->chunks_0 = chunks_16;
    (&kernelContext_94)->sort_0 = sort_16;
    (&kernelContext_94)->atom_0 = atom_16;
    (&kernelContext_94)->manifolds_0 = manifolds_16;
    (&kernelContext_94)->loads_0 = loads_16;
    (&kernelContext_94)->bonds_0 = bonds_16;
    (&kernelContext_94)->args_0 = args_16;
    (&kernelContext_94)->info_0 = info_16;
    (&kernelContext_94)->scan_0 = scan_16;
    (&kernelContext_94)->parts_0 = parts_16;
    (&kernelContext_94)->colliders_0 = colliders_16;
    (&kernelContext_94)->collider_info_0 = collider_info_16;
    (&kernelContext_94)->view_chunks_0 = view_chunks_16;
    (&kernelContext_94)->view_0 = view_16;
    (&kernelContext_94)->objects_0 = objects_16;
    (&kernelContext_94)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_94)->lines_0 = lines_16;
    (&kernelContext_94)->geometry_0 = geometry_16;
    (&kernelContext_94)->shots_0 = shots_16;
    (&kernelContext_94)->lp_0 = lp_16;
    (&kernelContext_94)->ext_0 = ext_17;
    (&kernelContext_94)->lp_sides_0 = lp_sides_16;
    (&kernelContext_94)->engine_counters_0 = engine_counters_16;
    (&kernelContext_94)->engine_contacts_0 = engine_contacts_16;
    (&kernelContext_94)->cs_anchors_0 = cs_anchors_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_94)->g_scan_0 = &g_scan_16;
    uint c_15 = id_13.x;
    bool _S682 = is_root_0(c_15, &kernelContext_94);
    if(!_S682)
    {
        return;
    }
    uint _S683 = body_of_0(c_15, &kernelContext_94);
    uint pb_0 = 7U * _S683;
    uint _S684 = a_sums_0(&kernelContext_94);
    uint at_7 = _S684 + 20U * c_15;
    float _S685 = read64_0(at_7, (&kernelContext_94)->params_0->scale_m_0, &kernelContext_94);
    float _S686 = read64_0(at_7 + 2U, (&kernelContext_94)->params_0->scale_s_0, &kernelContext_94);
    float _S687 = read64_0(at_7 + 4U, (&kernelContext_94)->params_0->scale_s_0, &kernelContext_94);
    float _S688 = read64_0(at_7 + 6U, (&kernelContext_94)->params_0->scale_s_0, &kernelContext_94);
    float3 d_19 = float3(_S686, _S687, _S688) / float3(max(_S685, 1.00000000317107685e-30f)) ;
    float _S689 = read64_0(at_7 + 8U, (&kernelContext_94)->params_0->scale_j_0, &kernelContext_94);
    float _S690 = read64_0(at_7 + 10U, (&kernelContext_94)->params_0->scale_j_0, &kernelContext_94);
    float _S691 = read64_0(at_7 + 12U, (&kernelContext_94)->params_0->scale_j_0, &kernelContext_94);
    float _S692 = read64_0(at_7 + 14U, (&kernelContext_94)->params_0->scale_j_0, &kernelContext_94);
    float _S693 = read64_0(at_7 + 16U, (&kernelContext_94)->params_0->scale_j_0, &kernelContext_94);
    float _S694 = read64_0(at_7 + 18U, (&kernelContext_94)->params_0->scale_j_0, &kernelContext_94);
    float dd_1 = dot(d_19, d_19);
    float _S695 = d_19.x;
    float _S696 = _S685 * _S695;
    float _S697 = d_19.y;
    float _S698 = _S692 + _S696 * _S697;
    float _S699 = d_19.z;
    float _S700 = _S693 + _S696 * _S699;
    float _S701 = _S694 + _S685 * _S697 * _S699;
    float3 com_1 = (float4(*((&kernelContext_94)->chunks_0+4U * c_15)) ).xyz + d_19;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S689 - _S685 * (dd_1 - _S695 * _S695), _S698, _S700), float3(_S698, _S690 - _S685 * (dd_1 - _S697 * _S697), _S701), float3(_S700, _S701, _S691 - _S685 * (dd_1 - _S699 * _S699)), &principal_0, &moments_1);
    uint _S702 = a_anchor_0(&kernelContext_94);
    uint _S703 = atomic_load_explicit((&kernelContext_94)->atom_0+(_S702 + c_15), memory_order_relaxed);
    bool anchored_0 = _S703 != 0U;
    thread Quat_0 q_17;
    thread float3 origin_4;
    structure_pose_0(_S683, pb_0, &q_17, &origin_4, &kernelContext_94);
    float3 p_parent_0 = (float4(*((&kernelContext_94)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_94)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_94)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S704 = q_17;
    float3 _S705 = rotate_0(&_S704, com_1);
    float3 pos_0 = to_rubble_0(origin_4 + _S705);
    thread Quat_0 _S706 = q_17;
    thread Quat_0 _S707 = principal_0;
    Quat_0 _S708 = quat_mul_0(&_S706, &_S707);
    thread Quat_0 _S709 = _S708;
    Quat_0 _S710 = normalized_0(&_S709);
    thread Quat_0 _S711 = _S710;
    Quat_0 _S712 = rot_to_rubble_0(&_S711);
    bool _S713;
    if(anchored_0)
    {
        _S713 = true;
    }
    else
    {
        _S713 = !(_S685 > 0.0f);
    }
    float inv_mass_3;
    if(_S713)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / _S685;
    }
    float3 _S714 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S715;
        if((moments_1.x) > 0.0f)
        {
            _S715 = 1.0f / moments_1.x;
        }
        else
        {
            _S715 = 0.0f;
        }
        float _S716;
        if((moments_1.y) > 0.0f)
        {
            _S716 = 1.0f / moments_1.y;
        }
        else
        {
            _S716 = 0.0f;
        }
        float _S717;
        if((moments_1.z) > 0.0f)
        {
            _S717 = 1.0f / moments_1.z;
        }
        else
        {
            _S717 = 0.0f;
        }
        inv_inertia_0 = float3(_S715, _S716, _S717);
    }
    else
    {
        inv_inertia_0 = _S714;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S714;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S714;
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
    uint _S718 = stage_0(c_15, &kernelContext_94);
    *((&kernelContext_94)->frames_0+_S718) = packed_float4(float4(pos_0, inv_mass_3)) ;
    packed_float4 device* _S719 = (&kernelContext_94)->frames_0+(_S718 + 1U);
    thread Quat_0 _S720 = _S712;
    float4 _S721 = quat_vec_0(&_S720);
    *_S719 = packed_float4(_S721) ;
    *((&kernelContext_94)->frames_0+(_S718 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_94)->frames_0+(_S718 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_94)->frames_0+(_S718 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_94)->params_0->step_0))))) ;
    *((&kernelContext_94)->frames_0+(_S718 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_94)->frames_0+(_S718 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_94)->frames_0+(_S718 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_94)->frames_0+2U * _S683)) ).w)) ;
    packed_float4 device* _S722 = (&kernelContext_94)->frames_0+(_S718 + 8U);
    thread Quat_0 _S723 = principal_0;
    float4 _S724 = quat_vec_0(&_S723);
    *_S722 = packed_float4(_S724) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], packed_float4 device* lp_17 [[buffer(24)]], packed_float4 device* ext_18 [[buffer(23)]], packed_float4 device* lp_sides_17 [[buffer(28)]], uint device* engine_counters_17 [[buffer(26)]], packed_float4 device* engine_contacts_17 [[buffer(25)]], packed_float4 device* cs_anchors_17 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_95;
    (&kernelContext_95)->params_0 = params_17;
    (&kernelContext_95)->work_0 = work_17;
    (&kernelContext_95)->frames_0 = frames_17;
    (&kernelContext_95)->bodies_0 = bodies_17;
    (&kernelContext_95)->chunks_0 = chunks_17;
    (&kernelContext_95)->sort_0 = sort_17;
    (&kernelContext_95)->atom_0 = atom_17;
    (&kernelContext_95)->manifolds_0 = manifolds_17;
    (&kernelContext_95)->loads_0 = loads_17;
    (&kernelContext_95)->bonds_0 = bonds_17;
    (&kernelContext_95)->args_0 = args_17;
    (&kernelContext_95)->info_0 = info_17;
    (&kernelContext_95)->scan_0 = scan_17;
    (&kernelContext_95)->parts_0 = parts_17;
    (&kernelContext_95)->colliders_0 = colliders_17;
    (&kernelContext_95)->collider_info_0 = collider_info_17;
    (&kernelContext_95)->view_chunks_0 = view_chunks_17;
    (&kernelContext_95)->view_0 = view_17;
    (&kernelContext_95)->objects_0 = objects_17;
    (&kernelContext_95)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_95)->lines_0 = lines_17;
    (&kernelContext_95)->geometry_0 = geometry_17;
    (&kernelContext_95)->shots_0 = shots_17;
    (&kernelContext_95)->lp_0 = lp_17;
    (&kernelContext_95)->ext_0 = ext_18;
    (&kernelContext_95)->lp_sides_0 = lp_sides_17;
    (&kernelContext_95)->engine_counters_0 = engine_counters_17;
    (&kernelContext_95)->engine_contacts_0 = engine_contacts_17;
    (&kernelContext_95)->cs_anchors_0 = cs_anchors_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_95)->g_scan_0 = &g_scan_17;
    uint c_16 = id_14.x;
    bool _S725 = is_root_0(c_16, &kernelContext_95);
    if(!_S725)
    {
        return;
    }
    uint _S726 = w_slot_0(&kernelContext_95);
    uint slot_3 = *((&kernelContext_95)->work_0+(_S726 + c_16));
    uint _S727 = stage_0(c_16, &kernelContext_95);
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
            *((&kernelContext_95)->bodies_0+(7U * (slot_3 + region_5 * (&kernelContext_95)->params_0->max_bodies_0) + w_4)) = packed_float4(float4(*((&kernelContext_95)->frames_0+(_S727 + w_4))) ) ;
            w_4 = w_4 + 1U;
        }
        region_5 = region_5 + 1U;
    }
    uint _S728 = 2U * slot_3;
    *((&kernelContext_95)->frames_0+_S728) = packed_float4(float4(*((&kernelContext_95)->frames_0+(_S727 + 7U))) ) ;
    *((&kernelContext_95)->frames_0+(_S728 + 1U)) = packed_float4(float4(*((&kernelContext_95)->frames_0+(_S727 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], packed_float4 device* lp_18 [[buffer(24)]], packed_float4 device* ext_19 [[buffer(23)]], packed_float4 device* lp_sides_18 [[buffer(28)]], uint device* engine_counters_18 [[buffer(26)]], packed_float4 device* engine_contacts_18 [[buffer(25)]], packed_float4 device* cs_anchors_18 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_96;
    (&kernelContext_96)->params_0 = params_18;
    (&kernelContext_96)->work_0 = work_18;
    (&kernelContext_96)->frames_0 = frames_18;
    (&kernelContext_96)->bodies_0 = bodies_18;
    (&kernelContext_96)->chunks_0 = chunks_18;
    (&kernelContext_96)->sort_0 = sort_18;
    (&kernelContext_96)->atom_0 = atom_18;
    (&kernelContext_96)->manifolds_0 = manifolds_18;
    (&kernelContext_96)->loads_0 = loads_18;
    (&kernelContext_96)->bonds_0 = bonds_18;
    (&kernelContext_96)->args_0 = args_18;
    (&kernelContext_96)->info_0 = info_18;
    (&kernelContext_96)->scan_0 = scan_18;
    (&kernelContext_96)->parts_0 = parts_18;
    (&kernelContext_96)->colliders_0 = colliders_18;
    (&kernelContext_96)->collider_info_0 = collider_info_18;
    (&kernelContext_96)->view_chunks_0 = view_chunks_18;
    (&kernelContext_96)->view_0 = view_18;
    (&kernelContext_96)->objects_0 = objects_18;
    (&kernelContext_96)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_96)->lines_0 = lines_18;
    (&kernelContext_96)->geometry_0 = geometry_18;
    (&kernelContext_96)->shots_0 = shots_18;
    (&kernelContext_96)->lp_0 = lp_18;
    (&kernelContext_96)->ext_0 = ext_19;
    (&kernelContext_96)->lp_sides_0 = lp_sides_18;
    (&kernelContext_96)->engine_counters_0 = engine_counters_18;
    (&kernelContext_96)->engine_contacts_0 = engine_contacts_18;
    (&kernelContext_96)->cs_anchors_0 = cs_anchors_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_96)->g_scan_0 = &g_scan_18;
    uint c_17 = id_15.x;
    bool _S729;
    if(c_17 >= (params_18->chunk_count_0))
    {
        _S729 = true;
    }
    else
    {
        uint _S730 = body_of_0(c_17, &kernelContext_96);
        bool _S731 = dirty_0(_S730, &kernelContext_96);
        _S729 = !_S731;
    }
    if(_S729)
    {
        return;
    }
    uint _S732 = w_label_0(&kernelContext_96);
    uint device* _S733 = (&kernelContext_96)->work_0+(_S732 + c_17);
    uint r_21 = *_S733;
    uint _S734 = w_slot_0(&kernelContext_96);
    uint device* _S735 = (&kernelContext_96)->work_0+(_S734 + *_S733);
    uint slot_4 = *_S735;
    *((&kernelContext_96)->work_0+(w_body_0() + c_17)) = *_S735;
    uint _S736 = 2U * slot_4;
    float3 _S737 = (float4(*((&kernelContext_96)->frames_0+_S736)) ).xyz;
    thread Quat_0 _S738 = quat_of_0(float4(*((&kernelContext_96)->frames_0+(_S736 + 1U))) );
    Quat_0 _S739 = conj_0(&_S738);
    float inv_mass_4 = (float4(*((&kernelContext_96)->bodies_0+7U * slot_4)) ).w;
    float _S740;
    if(inv_mass_4 > 0.0f)
    {
        _S740 = 1.0f / inv_mass_4;
    }
    else
    {
        _S740 = float_bits_0(2139095040U);
    }
    float4 _S741 = float4(*((&kernelContext_96)->chunks_0+(4U * c_17 + 2U))) ;
    uint first_1 = (as_type<uint>((_S741.z)));
    uint _S742 = (as_type<uint>((_S741.w)));
    float4 _S743 = float4(*((&kernelContext_96)->chunks_0+(4U * r_21 + 3U))) ;
    uint p_13 = first_1;
    for(;;)
    {
        if(p_13 < (first_1 + _S742))
        {
        }
        else
        {
            break;
        }
        uint _S744 = 2U * p_13;
        float4 _S745 = float4(*((&kernelContext_96)->parts_0+_S744)) ;
        uint col_0 = (as_type<uint>((_S745.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_96)->parts_0+(_S744 + 1U))) );
        uint _S746 = 4U * col_0;
        float4 _S747 = float4(*((&kernelContext_96)->colliders_0+_S746)) ;
        *((&kernelContext_96)->colliders_0+_S746) = packed_float4(float4((as_type<float>((slot_4))), _S747.y, _S747.z, _S747.w)) ;
        uint _S748 = _S746 + 1U;
        packed_float4 device* _S749 = (&kernelContext_96)->colliders_0+_S748;
        float3 _S750 = _S745.xyz - _S737;
        thread Quat_0 _S751 = _S739;
        float3 _S752 = rotate_0(&_S751, _S750);
        *_S749 = packed_float4(float4(_S752, (float4(*((&kernelContext_96)->colliders_0+_S748)) ).w)) ;
        packed_float4 device* _S753 = (&kernelContext_96)->colliders_0+(_S746 + 2U);
        thread Quat_0 _S754 = _S739;
        thread Quat_0 _S755 = rot_1;
        Quat_0 _S756 = quat_mul_0(&_S754, &_S755);
        thread Quat_0 _S757 = _S756;
        Quat_0 _S758 = normalized_0(&_S757);
        thread Quat_0 _S759 = _S758;
        float4 _S760 = quat_vec_0(&_S759);
        *_S753 = packed_float4(_S760) ;
        uint _S761 = 2U * col_0;
        uint4 _S762 = uint4(*((&kernelContext_96)->collider_info_0+_S761)) ;
        *((&kernelContext_96)->collider_info_0+_S761) = packed_uint4(uint4(_S762.x, slot_4, _S762.z, _S762.w)) ;
        *((&kernelContext_96)->collider_info_0+(_S761 + 1U)) = packed_uint4(uint4((as_type<uint>((_S740))), (as_type<uint>((_S743.x))), (as_type<uint>((_S743.w))), 0U)) ;
        p_13 = p_13 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], packed_float4 device* lp_19 [[buffer(24)]], packed_float4 device* ext_20 [[buffer(23)]], packed_float4 device* lp_sides_19 [[buffer(28)]], uint device* engine_counters_19 [[buffer(26)]], packed_float4 device* engine_contacts_19 [[buffer(25)]], packed_float4 device* cs_anchors_19 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_97;
    (&kernelContext_97)->params_0 = params_19;
    (&kernelContext_97)->work_0 = work_19;
    (&kernelContext_97)->frames_0 = frames_19;
    (&kernelContext_97)->bodies_0 = bodies_19;
    (&kernelContext_97)->chunks_0 = chunks_19;
    (&kernelContext_97)->sort_0 = sort_19;
    (&kernelContext_97)->atom_0 = atom_19;
    (&kernelContext_97)->manifolds_0 = manifolds_19;
    (&kernelContext_97)->loads_0 = loads_19;
    (&kernelContext_97)->bonds_0 = bonds_19;
    (&kernelContext_97)->args_0 = args_19;
    (&kernelContext_97)->info_0 = info_19;
    (&kernelContext_97)->scan_0 = scan_19;
    (&kernelContext_97)->parts_0 = parts_19;
    (&kernelContext_97)->colliders_0 = colliders_19;
    (&kernelContext_97)->collider_info_0 = collider_info_19;
    (&kernelContext_97)->view_chunks_0 = view_chunks_19;
    (&kernelContext_97)->view_0 = view_19;
    (&kernelContext_97)->objects_0 = objects_19;
    (&kernelContext_97)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_97)->lines_0 = lines_19;
    (&kernelContext_97)->geometry_0 = geometry_19;
    (&kernelContext_97)->shots_0 = shots_19;
    (&kernelContext_97)->lp_0 = lp_19;
    (&kernelContext_97)->ext_0 = ext_20;
    (&kernelContext_97)->lp_sides_0 = lp_sides_19;
    (&kernelContext_97)->engine_counters_0 = engine_counters_19;
    (&kernelContext_97)->engine_contacts_0 = engine_contacts_19;
    (&kernelContext_97)->cs_anchors_0 = cs_anchors_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_97)->g_scan_0 = &g_scan_19;
    uint s_10 = id_16.x;
    if(s_10 >= (params_19->body_count_0))
    {
        return;
    }
    uint _S763 = a_dirty_0(&kernelContext_97);
    atomic_store_explicit((&kernelContext_97)->atom_0+(_S763 + s_10), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_5 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_8 = fract(h_5);
    float q_18 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_8);
    float t_3 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_8));
    uint k_21 = uint(h_5);
    float3 _S764;
    if(k_21 == 0U)
    {
        _S764 = float3(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_21 == 1U)
        {
            _S764 = float3(q_18, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_21 == 2U)
            {
                _S764 = float3(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_21 == 3U)
                {
                    _S764 = float3(0.34959998726844788f, q_18, 0.92000001668930054f);
                }
                else
                {
                    if(k_21 == 4U)
                    {
                        _S764 = float3(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S764 = float3(0.92000001668930054f, 0.34959998726844788f, q_18);
                    }
                }
            }
        }
    }
    return _S764;
}

void write_object_0(uint o_4, const Quat_0 thread* q_19, float3 t_4, float4 color_0, KernelContext_0 thread* kernelContext_98)
{
    float3 _S765 = rotate_0(q_19, float3(1.0f, 0.0f, 0.0f));
    float3 _S766 = rotate_0(q_19, float3(0.0f, 1.0f, 0.0f));
    float3 _S767 = rotate_0(q_19, float3(0.0f, 0.0f, 1.0f));
    uint _S768 = 4U * o_4;
    *(kernelContext_98->objects_0+_S768) = packed_float4(float4(_S765.x, _S766.x, _S767.x, t_4.x)) ;
    *(kernelContext_98->objects_0+(_S768 + 1U)) = packed_float4(float4(_S765.y, _S766.y, _S767.y, t_4.y)) ;
    *(kernelContext_98->objects_0+(_S768 + 2U)) = packed_float4(float4(_S765.z, _S766.z, _S767.z, t_4.z)) ;
    *(kernelContext_98->objects_0+(_S768 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], packed_float4 device* lp_20 [[buffer(24)]], packed_float4 device* ext_21 [[buffer(23)]], packed_float4 device* lp_sides_20 [[buffer(28)]], uint device* engine_counters_20 [[buffer(26)]], packed_float4 device* engine_contacts_20 [[buffer(25)]], packed_float4 device* cs_anchors_20 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_20;
    (&kernelContext_99)->work_0 = work_20;
    (&kernelContext_99)->frames_0 = frames_20;
    (&kernelContext_99)->bodies_0 = bodies_20;
    (&kernelContext_99)->chunks_0 = chunks_20;
    (&kernelContext_99)->sort_0 = sort_20;
    (&kernelContext_99)->atom_0 = atom_20;
    (&kernelContext_99)->manifolds_0 = manifolds_20;
    (&kernelContext_99)->loads_0 = loads_20;
    (&kernelContext_99)->bonds_0 = bonds_20;
    (&kernelContext_99)->args_0 = args_20;
    (&kernelContext_99)->info_0 = info_20;
    (&kernelContext_99)->scan_0 = scan_20;
    (&kernelContext_99)->parts_0 = parts_20;
    (&kernelContext_99)->colliders_0 = colliders_20;
    (&kernelContext_99)->collider_info_0 = collider_info_20;
    (&kernelContext_99)->view_chunks_0 = view_chunks_20;
    (&kernelContext_99)->view_0 = view_20;
    (&kernelContext_99)->objects_0 = objects_20;
    (&kernelContext_99)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_99)->lines_0 = lines_20;
    (&kernelContext_99)->geometry_0 = geometry_20;
    (&kernelContext_99)->shots_0 = shots_20;
    (&kernelContext_99)->lp_0 = lp_20;
    (&kernelContext_99)->ext_0 = ext_21;
    (&kernelContext_99)->lp_sides_0 = lp_sides_20;
    (&kernelContext_99)->engine_counters_0 = engine_counters_20;
    (&kernelContext_99)->engine_contacts_0 = engine_contacts_20;
    (&kernelContext_99)->cs_anchors_0 = cs_anchors_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_99)->g_scan_0 = &g_scan_20;
    uint k_22 = id_17.x;
    if(k_22 >= (params_20->chunk_count_0))
    {
        return;
    }
    uint _S769 = body_of_0(k_22, &kernelContext_99);
    uint b_25 = 7U * _S769;
    thread Quat_0 q_20;
    thread float3 origin_5;
    structure_pose_0(_S769, b_25, &q_20, &origin_5, &kernelContext_99);
    uint _S770 = 3U * k_22;
    float4 _S771 = float4(*((&kernelContext_99)->view_chunks_0+_S770)) ;
    float3 base_2 = (float4(*((&kernelContext_99)->view_chunks_0+(_S770 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_99)->bodies_0+b_25)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_99)->view_0->mode_0) == 1U)
    {
        float _S772 = max(length((float4(*((&kernelContext_99)->bodies_0+(b_25 + 3U))) ).xyz), length((float4(*((&kernelContext_99)->bodies_0+(b_25 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S772 > ((&kernelContext_99)->view_0->rest_speed_0))
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
        if(((&kernelContext_99)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S769);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S769);
            }
        }
    }
    uint _S773 = (as_type<uint>((_S771.w)));
    Quat_0 _S774 = quat_of_0(float4(*((&kernelContext_99)->view_chunks_0+(_S770 + 1U))) );
    thread Quat_0 _S775 = q_20;
    thread Quat_0 _S776 = _S774;
    Quat_0 _S777 = quat_mul_0(&_S775, &_S776);
    thread Quat_0 _S778 = _S777;
    Quat_0 _S779 = normalized_0(&_S778);
    float3 _S780 = _S771.xyz;
    thread Quat_0 _S781 = q_20;
    float3 _S782 = rotate_0(&_S781, _S780);
    float3 _S783 = origin_5 + _S782;
    float4 _S784 = float4(color_1, 1.0f);
    thread Quat_0 _S785 = _S779;
    write_object_0(_S773, &_S785, _S783, _S784, &kernelContext_99);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], packed_float4 device* lp_21 [[buffer(24)]], packed_float4 device* ext_22 [[buffer(23)]], packed_float4 device* lp_sides_21 [[buffer(28)]], uint device* engine_counters_21 [[buffer(26)]], packed_float4 device* engine_contacts_21 [[buffer(25)]], packed_float4 device* cs_anchors_21 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_100;
    (&kernelContext_100)->params_0 = params_21;
    (&kernelContext_100)->work_0 = work_21;
    (&kernelContext_100)->frames_0 = frames_21;
    (&kernelContext_100)->bodies_0 = bodies_21;
    (&kernelContext_100)->chunks_0 = chunks_21;
    (&kernelContext_100)->sort_0 = sort_21;
    (&kernelContext_100)->atom_0 = atom_21;
    (&kernelContext_100)->manifolds_0 = manifolds_21;
    (&kernelContext_100)->loads_0 = loads_21;
    (&kernelContext_100)->bonds_0 = bonds_21;
    (&kernelContext_100)->args_0 = args_21;
    (&kernelContext_100)->info_0 = info_21;
    (&kernelContext_100)->scan_0 = scan_21;
    (&kernelContext_100)->parts_0 = parts_21;
    (&kernelContext_100)->colliders_0 = colliders_21;
    (&kernelContext_100)->collider_info_0 = collider_info_21;
    (&kernelContext_100)->view_chunks_0 = view_chunks_21;
    (&kernelContext_100)->view_0 = view_21;
    (&kernelContext_100)->objects_0 = objects_21;
    (&kernelContext_100)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_100)->lines_0 = lines_21;
    (&kernelContext_100)->geometry_0 = geometry_21;
    (&kernelContext_100)->shots_0 = shots_21;
    (&kernelContext_100)->lp_0 = lp_21;
    (&kernelContext_100)->ext_0 = ext_22;
    (&kernelContext_100)->lp_sides_0 = lp_sides_21;
    (&kernelContext_100)->engine_counters_0 = engine_counters_21;
    (&kernelContext_100)->engine_contacts_0 = engine_contacts_21;
    (&kernelContext_100)->cs_anchors_0 = cs_anchors_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_100)->g_scan_0 = &g_scan_21;
    uint i_5 = id_18.x;
    if(i_5 >= (view_21->shot_count_1))
    {
        return;
    }
    uint b_26 = 7U * (&kernelContext_100)->shot_bodies_0[i_5];
    uint o_5 = (&kernelContext_100)->view_0->impactor_base_0 + i_5;
    if(((float4(*((&kernelContext_100)->bodies_0+b_26)) ).w) == 0.0f)
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
            *((&kernelContext_100)->objects_0+(4U * o_5 + w_5)) = packed_float4(float4(0.0f) ) ;
            w_5 = w_5 + 1U;
        }
        return;
    }
    thread Quat_0 _S786 = quat_of_0(float4(*((&kernelContext_100)->bodies_0+(b_26 + 1U))) );
    Quat_0 _S787 = rot_from_rubble_0(&_S786);
    float3 _S788 = from_rubble_0((float4(*((&kernelContext_100)->bodies_0+b_26)) ).xyz);
    float4 _S789 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S790 = _S787;
    write_object_0(o_5, &_S790, _S788, _S789, &kernelContext_100);
    return;
}

float3 chunk_world_0(uint k_23, KernelContext_0 thread* kernelContext_101)
{
    uint _S791 = body_of_0(k_23, kernelContext_101);
    thread Quat_0 q_21;
    thread float3 origin_6;
    structure_pose_0(_S791, 7U * _S791, &q_21, &origin_6, kernelContext_101);
    float3 _S792 = (float4(*(kernelContext_101->view_chunks_0+3U * k_23)) ).xyz;
    thread Quat_0 _S793 = q_21;
    float3 _S794 = rotate_0(&_S793, _S792);
    return origin_6 + _S794;
}

void write_line_vertex_0(uint v_13, float3 p_14, KernelContext_0 thread* kernelContext_102)
{
    uint _S795 = 7U * v_13;
    *(kernelContext_102->lines_0+_S795) = p_14.x;
    *(kernelContext_102->lines_0+(_S795 + 1U)) = p_14.y;
    *(kernelContext_102->lines_0+(_S795 + 2U)) = p_14.z;
    *(kernelContext_102->lines_0+(_S795 + 3U)) = 1.0f;
    *(kernelContext_102->lines_0+(_S795 + 4U)) = 0.86000001430511475f;
    *(kernelContext_102->lines_0+(_S795 + 5U)) = 0.25f;
    *(kernelContext_102->lines_0+(_S795 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], packed_float4 device* lp_22 [[buffer(24)]], packed_float4 device* ext_23 [[buffer(23)]], packed_float4 device* lp_sides_22 [[buffer(28)]], uint device* engine_counters_22 [[buffer(26)]], packed_float4 device* engine_contacts_22 [[buffer(25)]], packed_float4 device* cs_anchors_22 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_103;
    (&kernelContext_103)->params_0 = params_22;
    (&kernelContext_103)->work_0 = work_22;
    (&kernelContext_103)->frames_0 = frames_22;
    (&kernelContext_103)->bodies_0 = bodies_22;
    (&kernelContext_103)->chunks_0 = chunks_22;
    (&kernelContext_103)->sort_0 = sort_22;
    (&kernelContext_103)->atom_0 = atom_22;
    (&kernelContext_103)->manifolds_0 = manifolds_22;
    (&kernelContext_103)->loads_0 = loads_22;
    (&kernelContext_103)->bonds_0 = bonds_22;
    (&kernelContext_103)->args_0 = args_22;
    (&kernelContext_103)->info_0 = info_22;
    (&kernelContext_103)->scan_0 = scan_22;
    (&kernelContext_103)->parts_0 = parts_22;
    (&kernelContext_103)->colliders_0 = colliders_22;
    (&kernelContext_103)->collider_info_0 = collider_info_22;
    (&kernelContext_103)->view_chunks_0 = view_chunks_22;
    (&kernelContext_103)->view_0 = view_22;
    (&kernelContext_103)->objects_0 = objects_22;
    (&kernelContext_103)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_103)->lines_0 = lines_22;
    (&kernelContext_103)->geometry_0 = geometry_22;
    (&kernelContext_103)->shots_0 = shots_22;
    (&kernelContext_103)->lp_0 = lp_22;
    (&kernelContext_103)->ext_0 = ext_23;
    (&kernelContext_103)->lp_sides_0 = lp_sides_22;
    (&kernelContext_103)->engine_counters_0 = engine_counters_22;
    (&kernelContext_103)->engine_contacts_0 = engine_contacts_22;
    (&kernelContext_103)->cs_anchors_0 = cs_anchors_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_103)->g_scan_0 = &g_scan_22;
    uint i_6 = id_19.x;
    if(i_6 >= (params_22->bond_count_0))
    {
        return;
    }
    uint4 _S796 = uint4(*((&kernelContext_103)->bonds_0+i_6)) ;
    float3 _S797 = chunk_world_0(_S796.x, &kernelContext_103);
    uint _S798 = w_alive_0(&kernelContext_103);
    float3 b_27;
    if((*((&kernelContext_103)->work_0+(_S798 + i_6))) != 0U)
    {
        float3 _S799 = chunk_world_0(_S796.y, &kernelContext_103);
        b_27 = _S799;
    }
    else
    {
        b_27 = _S797;
    }
    uint _S800 = 2U * i_6;
    write_line_vertex_0(_S800, _S797, &kernelContext_103);
    write_line_vertex_0(_S800 + 1U, b_27, &kernelContext_103);
    return;
}

[[kernel]] void fr_strike_reset(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], atomic<uint> device* atom_23 [[buffer(5)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], packed_float4 device* lp_23 [[buffer(24)]], packed_float4 device* ext_24 [[buffer(23)]], packed_float4 device* lp_sides_23 [[buffer(28)]], uint device* engine_counters_23 [[buffer(26)]], packed_float4 device* engine_contacts_23 [[buffer(25)]], packed_float4 device* cs_anchors_23 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_104;
    (&kernelContext_104)->params_0 = params_23;
    (&kernelContext_104)->work_0 = work_23;
    (&kernelContext_104)->frames_0 = frames_23;
    (&kernelContext_104)->bodies_0 = bodies_23;
    (&kernelContext_104)->chunks_0 = chunks_23;
    (&kernelContext_104)->sort_0 = sort_23;
    (&kernelContext_104)->atom_0 = atom_23;
    (&kernelContext_104)->manifolds_0 = manifolds_23;
    (&kernelContext_104)->loads_0 = loads_23;
    (&kernelContext_104)->bonds_0 = bonds_23;
    (&kernelContext_104)->args_0 = args_23;
    (&kernelContext_104)->info_0 = info_23;
    (&kernelContext_104)->scan_0 = scan_23;
    (&kernelContext_104)->parts_0 = parts_23;
    (&kernelContext_104)->colliders_0 = colliders_23;
    (&kernelContext_104)->collider_info_0 = collider_info_23;
    (&kernelContext_104)->view_chunks_0 = view_chunks_23;
    (&kernelContext_104)->view_0 = view_23;
    (&kernelContext_104)->objects_0 = objects_23;
    (&kernelContext_104)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_104)->lines_0 = lines_23;
    (&kernelContext_104)->geometry_0 = geometry_23;
    (&kernelContext_104)->shots_0 = shots_23;
    (&kernelContext_104)->lp_0 = lp_23;
    (&kernelContext_104)->ext_0 = ext_24;
    (&kernelContext_104)->lp_sides_0 = lp_sides_23;
    (&kernelContext_104)->engine_counters_0 = engine_counters_23;
    (&kernelContext_104)->engine_contacts_0 = engine_contacts_23;
    (&kernelContext_104)->cs_anchors_0 = cs_anchors_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_104)->g_scan_0 = &g_scan_23;
    uint b_28 = id_20.x;
    if(b_28 >= (params_23->body_count_0))
    {
        return;
    }
    uint _S801 = a_strike_0(&kernelContext_104);
    uint at_8 = _S801 + 9U * b_28;
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
        atomic_store_explicit((&kernelContext_104)->atom_0+(at_8 + w_6), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_104)->atom_0+(at_8 + 3U + w_6), 0U, memory_order_relaxed);
        w_6 = w_6 + 1U;
    }
    atomic_store_explicit((&kernelContext_104)->atom_0+(at_8 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_104)->atom_0+(at_8 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_104)->atom_0+(at_8 + 8U), 4294967295U, memory_order_relaxed);
    uint _S802 = strike_at_0(b_28, &kernelContext_104);
    (*((&kernelContext_104)->frames_0+_S802))[int(3)] = 0.0f;
    return;
}

uint ordered_0(float f_9)
{
    uint u_6 = (as_type<uint>((f_9)));
    uint _S803;
    if((u_6 & 2147483648U) != 0U)
    {
        _S803 = ~u_6;
    }
    else
    {
        _S803 = u_6 | 2147483648U;
    }
    return _S803;
}

[[kernel]] void fr_strike_bounds(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], atomic<uint> device* atom_24 [[buffer(5)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], packed_float4 device* lp_24 [[buffer(24)]], packed_float4 device* ext_25 [[buffer(23)]], packed_float4 device* lp_sides_24 [[buffer(28)]], uint device* engine_counters_24 [[buffer(26)]], packed_float4 device* engine_contacts_24 [[buffer(25)]], packed_float4 device* cs_anchors_24 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_105;
    (&kernelContext_105)->params_0 = params_24;
    (&kernelContext_105)->work_0 = work_24;
    (&kernelContext_105)->frames_0 = frames_24;
    (&kernelContext_105)->bodies_0 = bodies_24;
    (&kernelContext_105)->chunks_0 = chunks_24;
    (&kernelContext_105)->sort_0 = sort_24;
    (&kernelContext_105)->atom_0 = atom_24;
    (&kernelContext_105)->manifolds_0 = manifolds_24;
    (&kernelContext_105)->loads_0 = loads_24;
    (&kernelContext_105)->bonds_0 = bonds_24;
    (&kernelContext_105)->args_0 = args_24;
    (&kernelContext_105)->info_0 = info_24;
    (&kernelContext_105)->scan_0 = scan_24;
    (&kernelContext_105)->parts_0 = parts_24;
    (&kernelContext_105)->colliders_0 = colliders_24;
    (&kernelContext_105)->collider_info_0 = collider_info_24;
    (&kernelContext_105)->view_chunks_0 = view_chunks_24;
    (&kernelContext_105)->view_0 = view_24;
    (&kernelContext_105)->objects_0 = objects_24;
    (&kernelContext_105)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_105)->lines_0 = lines_24;
    (&kernelContext_105)->geometry_0 = geometry_24;
    (&kernelContext_105)->shots_0 = shots_24;
    (&kernelContext_105)->lp_0 = lp_24;
    (&kernelContext_105)->ext_0 = ext_25;
    (&kernelContext_105)->lp_sides_0 = lp_sides_24;
    (&kernelContext_105)->engine_counters_0 = engine_counters_24;
    (&kernelContext_105)->engine_contacts_0 = engine_contacts_24;
    (&kernelContext_105)->cs_anchors_0 = cs_anchors_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_105)->g_scan_0 = &g_scan_24;
    uint k_24 = id_21.x;
    bool _S804;
    if(k_24 >= (params_24->chunk_count_0))
    {
        _S804 = true;
    }
    else
    {
        _S804 = ((&kernelContext_105)->params_0->shot_count_0) == 0U;
    }
    if(_S804)
    {
        return;
    }
    thread Quat_0 q_22;
    float3 _S805 = chunk_box_world_0(k_24, &q_22, &kernelContext_105);
    float _S806 = (float4(*((&kernelContext_105)->geometry_0+3U * k_24)) ).w;
    uint _S807 = a_strike_0(&kernelContext_105);
    uint _S808 = body_of_0(k_24, &kernelContext_105);
    uint _S809 = _S807 + 9U * _S808;
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
        uint _S810 = ax_1;
        uint _S811 = atomic_fetch_min_explicit((&kernelContext_105)->atom_0+(_S809 + ax_1), ordered_0(_S805[ax_1] - _S806), memory_order_relaxed);
        uint _S812 = atomic_fetch_max_explicit((&kernelContext_105)->atom_0+(_S809 + 3U + ax_1), ordered_0(_S805[_S810] + _S806), memory_order_relaxed);
        ax_1 = ax_1 + 1U;
    }
    return;
}

float unordered_0(uint u_7)
{
    uint _S813;
    if((u_7 & 2147483648U) != 0U)
    {
        _S813 = u_7 & 2147483647U;
    }
    else
    {
        _S813 = ~u_7;
    }
    return (as_type<float>((_S813)));
}

[[kernel]] void fr_strike_find(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], atomic<uint> device* atom_25 [[buffer(5)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], packed_float4 device* lp_25 [[buffer(24)]], packed_float4 device* ext_26 [[buffer(23)]], packed_float4 device* lp_sides_25 [[buffer(28)]], uint device* engine_counters_25 [[buffer(26)]], packed_float4 device* engine_contacts_25 [[buffer(25)]], packed_float4 device* cs_anchors_25 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_106;
    (&kernelContext_106)->params_0 = params_25;
    (&kernelContext_106)->work_0 = work_25;
    (&kernelContext_106)->frames_0 = frames_25;
    (&kernelContext_106)->bodies_0 = bodies_25;
    (&kernelContext_106)->chunks_0 = chunks_25;
    (&kernelContext_106)->sort_0 = sort_25;
    (&kernelContext_106)->atom_0 = atom_25;
    (&kernelContext_106)->manifolds_0 = manifolds_25;
    (&kernelContext_106)->loads_0 = loads_25;
    (&kernelContext_106)->bonds_0 = bonds_25;
    (&kernelContext_106)->args_0 = args_25;
    (&kernelContext_106)->info_0 = info_25;
    (&kernelContext_106)->scan_0 = scan_25;
    (&kernelContext_106)->parts_0 = parts_25;
    (&kernelContext_106)->colliders_0 = colliders_25;
    (&kernelContext_106)->collider_info_0 = collider_info_25;
    (&kernelContext_106)->view_chunks_0 = view_chunks_25;
    (&kernelContext_106)->view_0 = view_25;
    (&kernelContext_106)->objects_0 = objects_25;
    (&kernelContext_106)->shot_bodies_0 = shot_bodies_25;
    (&kernelContext_106)->lines_0 = lines_25;
    (&kernelContext_106)->geometry_0 = geometry_25;
    (&kernelContext_106)->shots_0 = shots_25;
    (&kernelContext_106)->lp_0 = lp_25;
    (&kernelContext_106)->ext_0 = ext_26;
    (&kernelContext_106)->lp_sides_0 = lp_sides_25;
    (&kernelContext_106)->engine_counters_0 = engine_counters_25;
    (&kernelContext_106)->engine_contacts_0 = engine_contacts_25;
    (&kernelContext_106)->cs_anchors_0 = cs_anchors_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_106)->g_scan_0 = &g_scan_25;
    uint b_29 = id_22.x;
    bool _S814;
    if(b_29 >= (params_25->body_count_0))
    {
        _S814 = true;
    }
    else
    {
        _S814 = ((&kernelContext_106)->params_0->shot_count_0) == 0U;
    }
    if(_S814)
    {
        return;
    }
    uint _S815 = a_strike_0(&kernelContext_106);
    uint at_9 = _S815 + 9U * b_29;
    uint _S816 = atomic_load_explicit((&kernelContext_106)->atom_0+at_9, memory_order_relaxed);
    if(_S816 == 4294967295U)
    {
        return;
    }
    uint _S817 = atomic_load_explicit((&kernelContext_106)->atom_0+at_9, memory_order_relaxed);
    float _S818 = unordered_0(_S817);
    uint _S819 = atomic_load_explicit((&kernelContext_106)->atom_0+(at_9 + 1U), memory_order_relaxed);
    float _S820 = unordered_0(_S819);
    uint _S821 = atomic_load_explicit((&kernelContext_106)->atom_0+(at_9 + 2U), memory_order_relaxed);
    float3 _S822 = float3(_S818, _S820, unordered_0(_S821));
    uint _S823 = atomic_load_explicit((&kernelContext_106)->atom_0+(at_9 + 3U), memory_order_relaxed);
    float _S824 = unordered_0(_S823);
    uint _S825 = atomic_load_explicit((&kernelContext_106)->atom_0+(at_9 + 4U), memory_order_relaxed);
    float _S826 = unordered_0(_S825);
    uint _S827 = atomic_load_explicit((&kernelContext_106)->atom_0+(at_9 + 5U), memory_order_relaxed);
    float3 _S828 = float3(_S824, _S826, unordered_0(_S827));
    float3 _S829 = from_rubble_0((float4(*((&kernelContext_106)->bodies_0+(7U * b_29 + 3U))) ).xyz);
    float _S830 = float_bits_0(2139095040U);
    uint k_25 = 0U;
    for(;;)
    {
        if(k_25 < ((&kernelContext_106)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S831 = float4(*((&kernelContext_106)->shots_0+2U * k_25)) ;
        uint sb_1 = 7U * (as_type<uint>((_S831.x)));
        if(((float4(*((&kernelContext_106)->bodies_0+sb_1)) ).w) == 0.0f)
        {
            k_25 = k_25 + 1U;
            continue;
        }
        float3 c_18 = from_rubble_0((float4(*((&kernelContext_106)->bodies_0+sb_1)) ).xyz);
        float3 rel_2 = from_rubble_0((float4(*((&kernelContext_106)->bodies_0+(sb_1 + 3U))) ).xyz) - _S829;
        if(!((length(rel_2)) > ((&kernelContext_106)->params_0->strike_speed_0)))
        {
            k_25 = k_25 + 1U;
            continue;
        }
        float3 _S832 = float3(_S831.y) ;
        float3 _S833 = _S822 - _S832;
        float3 _S834 = _S828 + _S832;
        float _S835 = (&kernelContext_106)->params_0->dt_0;
        float t0_1 = 0.0f;
        float t1_1 = _S835;
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
            uint _S836 = ax_2;
            if((abs(rel_2[ax_2])) < 9.999999960041972e-13f)
            {
                uint _S837 = ax_2;
                if((c_18[ax_2]) < (_S833[ax_2]))
                {
                    _S814 = true;
                }
                else
                {
                    _S814 = (c_18[_S837]) > (_S834[ax_2]);
                }
                float t0_2;
                if(_S814)
                {
                    t0_2 = _S830;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + 1U;
                continue;
            }
            float ta_1 = (_S833[ax_2] - c_18[ax_2]) / rel_2[_S836];
            float tb_1 = (_S834[ax_2] - c_18[ax_2]) / rel_2[_S836];
            float _S838 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S838;
            ax_2 = ax_2 + 1U;
        }
        if(t0_1 <= t1_1)
        {
            uint _S839 = strike_at_0(b_29, &kernelContext_106);
            *((&kernelContext_106)->frames_0+_S839) = packed_float4(float4(c_18, float(k_25 + 1U))) ;
            *((&kernelContext_106)->frames_0+(_S839 + 1U)) = packed_float4(float4(rel_2, t0_1)) ;
            return;
        }
        k_25 = k_25 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_entry(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], atomic<uint> device* atom_26 [[buffer(5)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], packed_float4 device* lp_26 [[buffer(24)]], packed_float4 device* ext_27 [[buffer(23)]], packed_float4 device* lp_sides_26 [[buffer(28)]], uint device* engine_counters_26 [[buffer(26)]], packed_float4 device* engine_contacts_26 [[buffer(25)]], packed_float4 device* cs_anchors_26 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_107;
    (&kernelContext_107)->params_0 = params_26;
    (&kernelContext_107)->work_0 = work_26;
    (&kernelContext_107)->frames_0 = frames_26;
    (&kernelContext_107)->bodies_0 = bodies_26;
    (&kernelContext_107)->chunks_0 = chunks_26;
    (&kernelContext_107)->sort_0 = sort_26;
    (&kernelContext_107)->atom_0 = atom_26;
    (&kernelContext_107)->manifolds_0 = manifolds_26;
    (&kernelContext_107)->loads_0 = loads_26;
    (&kernelContext_107)->bonds_0 = bonds_26;
    (&kernelContext_107)->args_0 = args_26;
    (&kernelContext_107)->info_0 = info_26;
    (&kernelContext_107)->scan_0 = scan_26;
    (&kernelContext_107)->parts_0 = parts_26;
    (&kernelContext_107)->colliders_0 = colliders_26;
    (&kernelContext_107)->collider_info_0 = collider_info_26;
    (&kernelContext_107)->view_chunks_0 = view_chunks_26;
    (&kernelContext_107)->view_0 = view_26;
    (&kernelContext_107)->objects_0 = objects_26;
    (&kernelContext_107)->shot_bodies_0 = shot_bodies_26;
    (&kernelContext_107)->lines_0 = lines_26;
    (&kernelContext_107)->geometry_0 = geometry_26;
    (&kernelContext_107)->shots_0 = shots_26;
    (&kernelContext_107)->lp_0 = lp_26;
    (&kernelContext_107)->ext_0 = ext_27;
    (&kernelContext_107)->lp_sides_0 = lp_sides_26;
    (&kernelContext_107)->engine_counters_0 = engine_counters_26;
    (&kernelContext_107)->engine_contacts_0 = engine_contacts_26;
    (&kernelContext_107)->cs_anchors_0 = cs_anchors_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_107)->g_scan_0 = &g_scan_26;
    uint k_26 = id_23.x;
    bool _S840;
    if(k_26 >= (params_26->chunk_count_0))
    {
        _S840 = true;
    }
    else
    {
        _S840 = ((&kernelContext_107)->params_0->shot_count_0) == 0U;
    }
    if(_S840)
    {
        return;
    }
    uint _S841 = body_of_0(k_26, &kernelContext_107);
    thread uint shot_5;
    bool _S842 = struck_0(_S841, &shot_5, &kernelContext_107);
    if(!_S842)
    {
        return;
    }
    uint _S843 = strike_at_0(_S841, &kernelContext_107);
    float4 _S844 = float4(*((&kernelContext_107)->frames_0+(_S843 + 1U))) ;
    float3 from_3 = (float4(*((&kernelContext_107)->frames_0+_S843)) ).xyz;
    float3 rel_3 = _S844.xyz;
    float3 d_20 = rel_3 / float3(max(length(rel_3), 9.999999960041972e-13f)) ;
    float _S845 = (float4(*((&kernelContext_107)->shots_0+2U * shot_5)) ).y;
    thread Quat_0 q_23;
    float3 _S846 = chunk_box_world_0(k_26, &q_23, &kernelContext_107);
    uint _S847 = 3U * k_26;
    Quat_0 _S848 = quat_of_0(float4(*((&kernelContext_107)->geometry_0+(_S847 + 2U))) );
    thread Quat_0 _S849 = q_23;
    thread Quat_0 _S850 = _S848;
    Quat_0 _S851 = quat_mul_0(&_S849, &_S850);
    thread Quat_0 _S852 = _S851;
    Quat_0 _S853 = normalized_0(&_S852);
    float3 _S854 = (float4(*((&kernelContext_107)->geometry_0+(_S847 + 1U))) ).xyz;
    float _S855 = float_bits_0(2139095040U);
    float _S856 = (&kernelContext_107)->params_0->dt_0;
    float t_in_5 = 0.0f;
    float t_out_0 = _S856;
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
        thread Quat_0 _S857 = _S853;
        float3 _S858 = rotate_0(&_S857, e_3);
        float o_6 = dot(from_3 - _S846, _S858);
        float dv_1 = dot(rel_3, _S858);
        float ext_28 = _S854[ax_3] + _S845;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            float t_in_6;
            if((abs(o_6)) > ext_28)
            {
                t_in_6 = _S855;
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
        float _S859 = min(t_out_0, max(ta_2, tb_2));
        t_in_5 = max(t_in_5, min(ta_2, tb_2));
        t_out_0 = _S859;
        ax_3 = ax_3 + 1U;
    }
    uint _S860 = a_strike_0(&kernelContext_107);
    uint at_10 = _S860 + 9U * _S841;
    if(t_in_5 <= t_out_0)
    {
        uint _S861 = atomic_fetch_min_explicit((&kernelContext_107)->atom_0+(at_10 + 6U), (as_type<uint>((t_in_5))), memory_order_relaxed);
    }
    float3 r_22 = _S846 - (from_3 + rel_3 * float3(_S844.w) );
    float along_1 = dot(r_22, d_20);
    uint _S862 = atomic_fetch_min_explicit((&kernelContext_107)->atom_0+(at_10 + 7U), (as_type<uint>((length(r_22 - d_20 * float3(along_1) ) + abs(along_1) * 0.00100000004749745f))), memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], atomic<uint> device* atom_27 [[buffer(5)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], packed_float4 device* lp_27 [[buffer(24)]], packed_float4 device* ext_29 [[buffer(23)]], packed_float4 device* lp_sides_27 [[buffer(28)]], uint device* engine_counters_27 [[buffer(26)]], packed_float4 device* engine_contacts_27 [[buffer(25)]], packed_float4 device* cs_anchors_27 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_108;
    (&kernelContext_108)->params_0 = params_27;
    (&kernelContext_108)->work_0 = work_27;
    (&kernelContext_108)->frames_0 = frames_27;
    (&kernelContext_108)->bodies_0 = bodies_27;
    (&kernelContext_108)->chunks_0 = chunks_27;
    (&kernelContext_108)->sort_0 = sort_27;
    (&kernelContext_108)->atom_0 = atom_27;
    (&kernelContext_108)->manifolds_0 = manifolds_27;
    (&kernelContext_108)->loads_0 = loads_27;
    (&kernelContext_108)->bonds_0 = bonds_27;
    (&kernelContext_108)->args_0 = args_27;
    (&kernelContext_108)->info_0 = info_27;
    (&kernelContext_108)->scan_0 = scan_27;
    (&kernelContext_108)->parts_0 = parts_27;
    (&kernelContext_108)->colliders_0 = colliders_27;
    (&kernelContext_108)->collider_info_0 = collider_info_27;
    (&kernelContext_108)->view_chunks_0 = view_chunks_27;
    (&kernelContext_108)->view_0 = view_27;
    (&kernelContext_108)->objects_0 = objects_27;
    (&kernelContext_108)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_108)->lines_0 = lines_27;
    (&kernelContext_108)->geometry_0 = geometry_27;
    (&kernelContext_108)->shots_0 = shots_27;
    (&kernelContext_108)->lp_0 = lp_27;
    (&kernelContext_108)->ext_0 = ext_29;
    (&kernelContext_108)->lp_sides_0 = lp_sides_27;
    (&kernelContext_108)->engine_counters_0 = engine_counters_27;
    (&kernelContext_108)->engine_contacts_0 = engine_contacts_27;
    (&kernelContext_108)->cs_anchors_0 = cs_anchors_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_108)->g_scan_0 = &g_scan_27;
    uint k_27 = id_24.x;
    bool _S863;
    if(k_27 >= (params_27->chunk_count_0))
    {
        _S863 = true;
    }
    else
    {
        _S863 = ((&kernelContext_108)->params_0->shot_count_0) == 0U;
    }
    if(_S863)
    {
        return;
    }
    uint _S864 = body_of_0(k_27, &kernelContext_108);
    thread uint shot_6;
    bool _S865 = struck_0(_S864, &shot_6, &kernelContext_108);
    if(!_S865)
    {
        return;
    }
    uint _S866 = strike_at_0(_S864, &kernelContext_108);
    float4 _S867 = float4(*((&kernelContext_108)->frames_0+_S866)) ;
    float4 _S868 = float4(*((&kernelContext_108)->frames_0+(_S866 + 1U))) ;
    float3 _S869 = _S868.xyz;
    float3 d_21 = _S869 / float3(max(length(_S869), 9.999999960041972e-13f)) ;
    thread Quat_0 q_24;
    float3 _S870 = chunk_box_world_0(k_27, &q_24, &kernelContext_108);
    float3 r_23 = _S870 - (_S867.xyz + _S869 * float3(_S868.w) );
    float along_2 = dot(r_23, d_21);
    float dist_0 = length(r_23 - d_21 * float3(along_2) ) + abs(along_2) * 0.00100000004749745f;
    uint _S871 = a_strike_0(&kernelContext_108);
    uint at_11 = _S871 + 9U * _S864;
    uint _S872 = (as_type<uint>((dist_0)));
    uint _S873 = atomic_load_explicit((&kernelContext_108)->atom_0+(at_11 + 7U), memory_order_relaxed);
    if(_S872 == _S873)
    {
        uint _S874 = atomic_fetch_min_explicit((&kernelContext_108)->atom_0+(at_11 + 8U), k_27, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], atomic<uint> device* atom_28 [[buffer(5)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], packed_float4 device* lp_28 [[buffer(24)]], packed_float4 device* ext_30 [[buffer(23)]], packed_float4 device* lp_sides_28 [[buffer(28)]], uint device* engine_counters_28 [[buffer(26)]], packed_float4 device* engine_contacts_28 [[buffer(25)]], packed_float4 device* cs_anchors_28 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_109;
    (&kernelContext_109)->params_0 = params_28;
    (&kernelContext_109)->work_0 = work_28;
    (&kernelContext_109)->frames_0 = frames_28;
    (&kernelContext_109)->bodies_0 = bodies_28;
    (&kernelContext_109)->chunks_0 = chunks_28;
    (&kernelContext_109)->sort_0 = sort_28;
    (&kernelContext_109)->atom_0 = atom_28;
    (&kernelContext_109)->manifolds_0 = manifolds_28;
    (&kernelContext_109)->loads_0 = loads_28;
    (&kernelContext_109)->bonds_0 = bonds_28;
    (&kernelContext_109)->args_0 = args_28;
    (&kernelContext_109)->info_0 = info_28;
    (&kernelContext_109)->scan_0 = scan_28;
    (&kernelContext_109)->parts_0 = parts_28;
    (&kernelContext_109)->colliders_0 = colliders_28;
    (&kernelContext_109)->collider_info_0 = collider_info_28;
    (&kernelContext_109)->view_chunks_0 = view_chunks_28;
    (&kernelContext_109)->view_0 = view_28;
    (&kernelContext_109)->objects_0 = objects_28;
    (&kernelContext_109)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_109)->lines_0 = lines_28;
    (&kernelContext_109)->geometry_0 = geometry_28;
    (&kernelContext_109)->shots_0 = shots_28;
    (&kernelContext_109)->lp_0 = lp_28;
    (&kernelContext_109)->ext_0 = ext_30;
    (&kernelContext_109)->lp_sides_0 = lp_sides_28;
    (&kernelContext_109)->engine_counters_0 = engine_counters_28;
    (&kernelContext_109)->engine_contacts_0 = engine_contacts_28;
    (&kernelContext_109)->cs_anchors_0 = cs_anchors_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_109)->g_scan_0 = &g_scan_28;
    uint b_30 = id_25.x;
    bool _S875;
    if(b_30 >= (params_28->body_count_0))
    {
        _S875 = true;
    }
    else
    {
        _S875 = ((&kernelContext_109)->params_0->shot_count_0) == 0U;
    }
    if(_S875)
    {
        return;
    }
    thread uint shot_7;
    bool _S876 = struck_0(b_30, &shot_7, &kernelContext_109);
    if(!_S876)
    {
        return;
    }
    uint _S877 = a_strike_0(&kernelContext_109);
    uint at_12 = _S877 + 9U * b_30;
    uint _S878 = atomic_load_explicit((&kernelContext_109)->atom_0+(at_12 + 6U), memory_order_relaxed);
    float t_in_7 = (as_type<float>((_S878)));
    uint entry_2 = atomic_load_explicit((&kernelContext_109)->atom_0+(at_12 + 8U), memory_order_relaxed);
    uint _S879 = strike_at_0(b_30, &kernelContext_109);
    if(!(t_in_7 <= ((&kernelContext_109)->params_0->dt_0)))
    {
        _S875 = true;
    }
    else
    {
        _S875 = entry_2 == 4294967295U;
    }
    if(_S875)
    {
        (*((&kernelContext_109)->frames_0+_S879))[int(3)] = 0.0f;
        return;
    }
    float3 rel_4 = (float4(*((&kernelContext_109)->frames_0+(_S879 + 1U))) ).xyz;
    float speed_14 = length(rel_4);
    float4 _S880 = float4(*((&kernelContext_109)->shots_0+2U * shot_7)) ;
    float _S881 = max(_S880.z, 0.00100000004749745f);
    float m_imp_0 = _S880.w;
    float e_imp_0 = (float4(*((&kernelContext_109)->shots_0+(2U * shot_7 + 1U))) ).x;
    float3 contact_16 = (float4(*((&kernelContext_109)->frames_0+_S879)) ).xyz + rel_4 * float3(t_in_7)  + rel_4 / float3(max(speed_14, 9.999999960041972e-13f))  * float3(_S880.y) ;
    float inv_mass_5 = (float4(*((&kernelContext_109)->bodies_0+7U * b_30)) ).w;
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
    float modulus_1 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_109)->geometry_0+(3U * entry_2 + 1U))) ).w);
    *((&kernelContext_109)->frames_0+(_S879 + 2U)) = packed_float4(float4(contact_16, m_eff_6 * speed_14 / (2.86999988555908203f * pow(m_eff_6 * m_eff_6 / (_S881 * modulus_1 * modulus_1 * max(speed_14, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

bool strike_share_breaks_0(uint4 bond_0, float3 contact_17, float force_2, KernelContext_0 thread* kernelContext_110)
{
    thread Quat_0 q_25;
    float3 _S882 = chunk_box_world_0(bond_0.x, &q_25, kernelContext_110);
    float3 _S883 = chunk_box_world_0(bond_0.y, &q_25, kernelContext_110);
    float dist_1 = length(float3(0.5f)  * (_S882 + _S883) - contact_17);
    return (force_2 * min(1.0f, (as_type<float>((bond_0.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((bond_0.z)));
}

[[kernel]] void fr_strike_breaks(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], atomic<uint> device* atom_29 [[buffer(5)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], packed_float4 device* lp_29 [[buffer(24)]], packed_float4 device* ext_31 [[buffer(23)]], packed_float4 device* lp_sides_29 [[buffer(28)]], uint device* engine_counters_29 [[buffer(26)]], packed_float4 device* engine_contacts_29 [[buffer(25)]], packed_float4 device* cs_anchors_29 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_111;
    (&kernelContext_111)->params_0 = params_29;
    (&kernelContext_111)->work_0 = work_29;
    (&kernelContext_111)->frames_0 = frames_29;
    (&kernelContext_111)->bodies_0 = bodies_29;
    (&kernelContext_111)->chunks_0 = chunks_29;
    (&kernelContext_111)->sort_0 = sort_29;
    (&kernelContext_111)->atom_0 = atom_29;
    (&kernelContext_111)->manifolds_0 = manifolds_29;
    (&kernelContext_111)->loads_0 = loads_29;
    (&kernelContext_111)->bonds_0 = bonds_29;
    (&kernelContext_111)->args_0 = args_29;
    (&kernelContext_111)->info_0 = info_29;
    (&kernelContext_111)->scan_0 = scan_29;
    (&kernelContext_111)->parts_0 = parts_29;
    (&kernelContext_111)->colliders_0 = colliders_29;
    (&kernelContext_111)->collider_info_0 = collider_info_29;
    (&kernelContext_111)->view_chunks_0 = view_chunks_29;
    (&kernelContext_111)->view_0 = view_29;
    (&kernelContext_111)->objects_0 = objects_29;
    (&kernelContext_111)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_111)->lines_0 = lines_29;
    (&kernelContext_111)->geometry_0 = geometry_29;
    (&kernelContext_111)->shots_0 = shots_29;
    (&kernelContext_111)->lp_0 = lp_29;
    (&kernelContext_111)->ext_0 = ext_31;
    (&kernelContext_111)->lp_sides_0 = lp_sides_29;
    (&kernelContext_111)->engine_counters_0 = engine_counters_29;
    (&kernelContext_111)->engine_contacts_0 = engine_contacts_29;
    (&kernelContext_111)->cs_anchors_0 = cs_anchors_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_111)->g_scan_0 = &g_scan_29;
    uint i_7 = id_26.x;
    bool _S884;
    if(i_7 >= (params_29->bond_count_0))
    {
        _S884 = true;
    }
    else
    {
        _S884 = ((&kernelContext_111)->params_0->shot_count_0) == 0U;
    }
    if(_S884)
    {
        _S884 = true;
    }
    else
    {
        uint _S885 = w_alive_0(&kernelContext_111);
        _S884 = (*((&kernelContext_111)->work_0+(_S885 + i_7))) == 0U;
    }
    if(_S884)
    {
        return;
    }
    uint4 _S886 = uint4(*((&kernelContext_111)->bonds_0+i_7)) ;
    uint _S887 = body_of_0(_S886.x, &kernelContext_111);
    thread uint shot_8;
    bool _S888 = struck_0(_S887, &shot_8, &kernelContext_111);
    if(!_S888)
    {
        return;
    }
    uint _S889 = strike_at_0(_S887, &kernelContext_111);
    float4 _S890 = float4(*((&kernelContext_111)->frames_0+(_S889 + 2U))) ;
    bool _S891 = strike_share_breaks_0(_S886, _S890.xyz, _S890.w, &kernelContext_111);
    if(_S891)
    {
        uint _S892 = w_alive_0(&kernelContext_111);
        *((&kernelContext_111)->work_0+(_S892 + i_7)) = 0U;
        uint _S893 = a_dirty_0(&kernelContext_111);
        uint _S894 = atomic_fetch_or_explicit((&kernelContext_111)->atom_0+(_S893 + _S887), 1U, memory_order_relaxed);
        uint _S895 = a_counters_1(&kernelContext_111);
        uint _S896 = atomic_fetch_add_explicit((&kernelContext_111)->atom_0+_S895, 1U, memory_order_relaxed);
        uint _S897 = atomic_fetch_add_explicit((&kernelContext_111)->atom_0+(_S895 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_lp_0(uint b_31, KernelContext_0 thread* kernelContext_112)
{
    uint _S898 = a_strike_0(kernelContext_112);
    return _S898 + 9U * kernelContext_112->params_0->body_count_0 + 14U * b_31;
}

[[kernel]] void fr_lp_clear(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], atomic<uint> device* atom_30 [[buffer(5)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], packed_float4 device* lp_30 [[buffer(24)]], packed_float4 device* ext_32 [[buffer(23)]], packed_float4 device* lp_sides_30 [[buffer(28)]], uint device* engine_counters_30 [[buffer(26)]], packed_float4 device* engine_contacts_30 [[buffer(25)]], packed_float4 device* cs_anchors_30 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_113;
    (&kernelContext_113)->params_0 = params_30;
    (&kernelContext_113)->work_0 = work_30;
    (&kernelContext_113)->frames_0 = frames_30;
    (&kernelContext_113)->bodies_0 = bodies_30;
    (&kernelContext_113)->chunks_0 = chunks_30;
    (&kernelContext_113)->sort_0 = sort_30;
    (&kernelContext_113)->atom_0 = atom_30;
    (&kernelContext_113)->manifolds_0 = manifolds_30;
    (&kernelContext_113)->loads_0 = loads_30;
    (&kernelContext_113)->bonds_0 = bonds_30;
    (&kernelContext_113)->args_0 = args_30;
    (&kernelContext_113)->info_0 = info_30;
    (&kernelContext_113)->scan_0 = scan_30;
    (&kernelContext_113)->parts_0 = parts_30;
    (&kernelContext_113)->colliders_0 = colliders_30;
    (&kernelContext_113)->collider_info_0 = collider_info_30;
    (&kernelContext_113)->view_chunks_0 = view_chunks_30;
    (&kernelContext_113)->view_0 = view_30;
    (&kernelContext_113)->objects_0 = objects_30;
    (&kernelContext_113)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_113)->lines_0 = lines_30;
    (&kernelContext_113)->geometry_0 = geometry_30;
    (&kernelContext_113)->shots_0 = shots_30;
    (&kernelContext_113)->lp_0 = lp_30;
    (&kernelContext_113)->ext_0 = ext_32;
    (&kernelContext_113)->lp_sides_0 = lp_sides_30;
    (&kernelContext_113)->engine_counters_0 = engine_counters_30;
    (&kernelContext_113)->engine_contacts_0 = engine_contacts_30;
    (&kernelContext_113)->cs_anchors_0 = cs_anchors_30;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_113)->g_scan_0 = &g_scan_30;
    uint b_32 = id_27.x;
    if(b_32 >= (params_30->body_count_0))
    {
        return;
    }
    uint _S899 = a_lp_0(b_32, &kernelContext_113);
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
        atomic_store_explicit((&kernelContext_113)->atom_0+(_S899 + w_7), 0U, memory_order_relaxed);
        w_7 = w_7 + 1U;
    }
    atomic_store_explicit((&kernelContext_113)->atom_0+(_S899 + 12U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_113)->atom_0+(_S899 + 13U), 0U, memory_order_relaxed);
    return;
}

uint st_f_0(uint k_28, uint state_0)
{
    return 22U * k_28 + state_0 + 6U;
}

uint lp_f_0(uint k_29)
{
    return st_f_0(k_29, 0U);
}

void add64w_0(uint at_13, float value_1, float scale_3, KernelContext_0 thread* kernelContext_114)
{
    float x_6 = round(value_1 * scale_3);
    float x_7;
    if(!((abs(x_6)) < 3.99999993722699776e+18f))
    {
        uint _S900 = a_counters_1(kernelContext_114);
        uint _S901 = atomic_fetch_or_explicit(kernelContext_114->atom_0+(_S900 + 3U), 2U, memory_order_relaxed);
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
        uint _S902 = ~hi_4;
        if(lo_6 == 0U)
        {
            lo_5 = 1U;
        }
        else
        {
            lo_5 = 0U;
        }
        uint _S903 = _S902 + lo_5;
        lo_5 = lo_6;
        hi_5 = _S903;
    }
    else
    {
        lo_5 = lo_4;
        hi_5 = hi_4;
    }
    uint old_1 = atomic_fetch_add_explicit(kernelContext_114->atom_0+at_13, lo_5, memory_order_relaxed);
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
        uint _S904 = atomic_fetch_add_explicit(kernelContext_114->atom_0+(at_13 + 1U), hi_6, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_lp_sum(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], atomic<uint> device* atom_31 [[buffer(5)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_31 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], packed_float4 device* lp_31 [[buffer(24)]], packed_float4 device* ext_33 [[buffer(23)]], packed_float4 device* lp_sides_31 [[buffer(28)]], uint device* engine_counters_31 [[buffer(26)]], packed_float4 device* engine_contacts_31 [[buffer(25)]], packed_float4 device* cs_anchors_31 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_115;
    (&kernelContext_115)->params_0 = params_31;
    (&kernelContext_115)->work_0 = work_31;
    (&kernelContext_115)->frames_0 = frames_31;
    (&kernelContext_115)->bodies_0 = bodies_31;
    (&kernelContext_115)->chunks_0 = chunks_31;
    (&kernelContext_115)->sort_0 = sort_31;
    (&kernelContext_115)->atom_0 = atom_31;
    (&kernelContext_115)->manifolds_0 = manifolds_31;
    (&kernelContext_115)->loads_0 = loads_31;
    (&kernelContext_115)->bonds_0 = bonds_31;
    (&kernelContext_115)->args_0 = args_31;
    (&kernelContext_115)->info_0 = info_31;
    (&kernelContext_115)->scan_0 = scan_31;
    (&kernelContext_115)->parts_0 = parts_31;
    (&kernelContext_115)->colliders_0 = colliders_31;
    (&kernelContext_115)->collider_info_0 = collider_info_31;
    (&kernelContext_115)->view_chunks_0 = view_chunks_31;
    (&kernelContext_115)->view_0 = view_31;
    (&kernelContext_115)->objects_0 = objects_31;
    (&kernelContext_115)->shot_bodies_0 = shot_bodies_31;
    (&kernelContext_115)->lines_0 = lines_31;
    (&kernelContext_115)->geometry_0 = geometry_31;
    (&kernelContext_115)->shots_0 = shots_31;
    (&kernelContext_115)->lp_0 = lp_31;
    (&kernelContext_115)->ext_0 = ext_33;
    (&kernelContext_115)->lp_sides_0 = lp_sides_31;
    (&kernelContext_115)->engine_counters_0 = engine_counters_31;
    (&kernelContext_115)->engine_contacts_0 = engine_contacts_31;
    (&kernelContext_115)->cs_anchors_0 = cs_anchors_31;
    threadgroup array<uint, int(1024)> g_scan_31;
    (&kernelContext_115)->g_scan_0 = &g_scan_31;
    uint k_30 = id_28.x;
    if(k_30 >= (params_31->chunk_count_0))
    {
        return;
    }
    uint _S905 = body_of_0(k_30, &kernelContext_115);
    thread Quat_0 q_26;
    thread float3 origin_7;
    structure_pose_0(_S905, 7U * _S905, &q_26, &origin_7, &kernelContext_115);
    thread Quat_0 _S906 = q_26;
    Quat_0 _S907 = conj_0(&_S906);
    float4 _S908 = float4(*((&kernelContext_115)->chunks_0+4U * k_30)) ;
    uint _S909 = l_vec_0(&kernelContext_115);
    uint _S910 = _S909 + 2U * k_30;
    float3 _S911 = (float4(*((&kernelContext_115)->loads_0+_S910)) ).xyz;
    thread Quat_0 _S912 = _S907;
    float3 _S913 = rotate_0(&_S912, _S911);
    float _S914 = _S908.w;
    float3 _S915 = float3((&kernelContext_115)->params_0->gx_0, (&kernelContext_115)->params_0->gy_0, (&kernelContext_115)->params_0->gz_0);
    thread Quat_0 _S916 = _S907;
    float3 _S917 = rotate_0(&_S916, _S915);
    float3 f_10 = _S913 + float3(_S914)  * _S917;
    float3 _S918 = (float4(*((&kernelContext_115)->loads_0+(_S910 + 1U))) ).xyz;
    thread Quat_0 _S919 = _S907;
    float3 _S920 = rotate_0(&_S919, _S918);
    uint _S921 = lp_f_0(k_30);
    *((&kernelContext_115)->lp_0+_S921) = packed_float4(float4(f_10, 0.0f)) ;
    *((&kernelContext_115)->lp_0+(_S921 + 1U)) = packed_float4(float4(_S920, 0.0f)) ;
    float3 _S922 = _S920 + cross(_S908.xyz - (float4(*((&kernelContext_115)->frames_0+2U * _S905)) ).xyz, f_10);
    uint _S923 = a_lp_0(_S905, &kernelContext_115);
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
        uint _S924 = 2U * a_8;
        add64w_0(_S923 + _S924, f_10[a_8], (&kernelContext_115)->params_0->lp_scale_0, &kernelContext_115);
        add64w_0(_S923 + 6U + _S924, _S922[a_8], (&kernelContext_115)->params_0->lp_scale_0, &kernelContext_115);
        a_8 = a_8 + 1U;
    }
    uint _S925 = atomic_fetch_min_explicit((&kernelContext_115)->atom_0+(_S923 + 12U), k_30, memory_order_relaxed);
    if(((&kernelContext_115)->info_0[k_30] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_115)->atom_0+(_S923 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

float3 read64w3_0(uint at_14, float scale_4, KernelContext_0 thread* kernelContext_116)
{
    float _S926 = read64_1(at_14, scale_4, kernelContext_116);
    float _S927 = read64_1(at_14 + 2U, scale_4, kernelContext_116);
    float _S928 = read64_1(at_14 + 4U, scale_4, kernelContext_116);
    return float3(_S926, _S927, _S928);
}

[[kernel]] void fr_lp_relief(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], atomic<uint> device* atom_32 [[buffer(5)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_32 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], packed_float4 device* lp_32 [[buffer(24)]], packed_float4 device* ext_34 [[buffer(23)]], packed_float4 device* lp_sides_32 [[buffer(28)]], uint device* engine_counters_32 [[buffer(26)]], packed_float4 device* engine_contacts_32 [[buffer(25)]], packed_float4 device* cs_anchors_32 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_117;
    (&kernelContext_117)->params_0 = params_32;
    (&kernelContext_117)->work_0 = work_32;
    (&kernelContext_117)->frames_0 = frames_32;
    (&kernelContext_117)->bodies_0 = bodies_32;
    (&kernelContext_117)->chunks_0 = chunks_32;
    (&kernelContext_117)->sort_0 = sort_32;
    (&kernelContext_117)->atom_0 = atom_32;
    (&kernelContext_117)->manifolds_0 = manifolds_32;
    (&kernelContext_117)->loads_0 = loads_32;
    (&kernelContext_117)->bonds_0 = bonds_32;
    (&kernelContext_117)->args_0 = args_32;
    (&kernelContext_117)->info_0 = info_32;
    (&kernelContext_117)->scan_0 = scan_32;
    (&kernelContext_117)->parts_0 = parts_32;
    (&kernelContext_117)->colliders_0 = colliders_32;
    (&kernelContext_117)->collider_info_0 = collider_info_32;
    (&kernelContext_117)->view_chunks_0 = view_chunks_32;
    (&kernelContext_117)->view_0 = view_32;
    (&kernelContext_117)->objects_0 = objects_32;
    (&kernelContext_117)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_117)->lines_0 = lines_32;
    (&kernelContext_117)->geometry_0 = geometry_32;
    (&kernelContext_117)->shots_0 = shots_32;
    (&kernelContext_117)->lp_0 = lp_32;
    (&kernelContext_117)->ext_0 = ext_34;
    (&kernelContext_117)->lp_sides_0 = lp_sides_32;
    (&kernelContext_117)->engine_counters_0 = engine_counters_32;
    (&kernelContext_117)->engine_contacts_0 = engine_contacts_32;
    (&kernelContext_117)->cs_anchors_0 = cs_anchors_32;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_117)->g_scan_0 = &g_scan_32;
    uint k_31 = id_29.x;
    if(k_31 >= (params_32->chunk_count_0))
    {
        return;
    }
    uint _S929 = body_of_0(k_31, &kernelContext_117);
    uint _S930 = a_lp_0(_S929, &kernelContext_117);
    uint _S931 = atomic_load_explicit((&kernelContext_117)->atom_0+(_S930 + 13U), memory_order_relaxed);
    bool supported_0 = _S931 != 0U;
    bool fixed_1;
    if(((&kernelContext_117)->info_0[k_31] & 1U) != 0U)
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            uint _S932 = atomic_load_explicit((&kernelContext_117)->atom_0+(_S930 + 12U), memory_order_relaxed);
            fixed_1 = _S932 == k_31;
        }
        else
        {
            fixed_1 = false;
        }
    }
    uint _S933 = lp_f_0(k_31);
    thread float4 f_11 = float4(*((&kernelContext_117)->lp_0+_S933)) ;
    uint _S934 = _S933 + 1U;
    thread float4 t_5 = float4(*((&kernelContext_117)->lp_0+_S934)) ;
    uint b_33 = 7U * _S929;
    float inv_mass_6 = (float4(*((&kernelContext_117)->bodies_0+b_33)) ).w;
    bool _S935;
    if(!supported_0)
    {
        _S935 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S935 = false;
    }
    if(_S935)
    {
        float3 _S936 = read64w3_0(_S930, (&kernelContext_117)->params_0->lp_scale_0, &kernelContext_117);
        float3 _S937 = read64w3_0(_S930 + 6U, (&kernelContext_117)->params_0->lp_scale_0, &kernelContext_117);
        uint _S938 = 2U * _S929;
        Quat_0 p_15 = quat_of_0(float4(*((&kernelContext_117)->frames_0+(_S938 + 1U))) );
        float3 _S939 = (float4(*((&kernelContext_117)->bodies_0+(b_33 + 2U))) ).xyz;
        thread Quat_0 _S940 = p_15;
        Quat_0 _S941 = conj_0(&_S940);
        thread Quat_0 _S942 = _S941;
        float3 _S943 = rotate_0(&_S942, _S937);
        float3 _S944 = _S939 * _S943;
        thread Quat_0 _S945 = p_15;
        float3 _S946 = rotate_0(&_S945, _S944);
        uint _S947 = 4U * k_31;
        float4 _S948 = float4(*((&kernelContext_117)->chunks_0+_S947)) ;
        float4 _S949 = float4(*((&kernelContext_117)->chunks_0+(_S947 + 1U))) ;
        float4 _S950 = float4(*((&kernelContext_117)->chunks_0+(_S947 + 2U))) ;
        f_11.xyz = f_11.xyz - float3(_S948.w)  * (_S936 * float3(inv_mass_6)  + cross(_S946, _S948.xyz - (float4(*((&kernelContext_117)->frames_0+_S938)) ).xyz));
        float _S951 = _S946.x;
        float _S952 = _S949.w;
        float _S953 = _S946.y;
        float _S954 = _S950.x;
        float _S955 = _S946.z;
        float _S956 = _S950.y;
        t_5.xyz = t_5.xyz - float3(_S949.x * _S951 + _S952 * _S953 + _S954 * _S955, _S952 * _S951 + _S949.y * _S953 + _S956 * _S955, _S954 * _S951 + _S956 * _S953 + _S949.z * _S955);
    }
    float _S957;
    if(fixed_1)
    {
        _S957 = 1.0f;
    }
    else
    {
        _S957 = 0.0f;
    }
    f_11.w = _S957;
    *((&kernelContext_117)->lp_0+_S933) = packed_float4(f_11) ;
    *((&kernelContext_117)->lp_0+_S934) = packed_float4(t_5) ;
    return;
}

uint adj_bond_0(uint at_15, KernelContext_0 thread* kernelContext_118)
{
    return (as_type<uint>(((float4(*(kernelContext_118->ext_0+(kernelContext_118->params_0->ext_adj_0 + at_15 / 4U))) )[at_15 % 4U])));
}

uint lp_intact_0(uint k_32, KernelContext_0 thread* kernelContext_119)
{
    float4 _S958 = float4(*(kernelContext_119->ext_0+(kernelContext_119->params_0->ext_chunks_0 + 4U * k_32))) ;
    uint _S959 = (as_type<uint>((_S958.x)));
    uint _S960 = (as_type<uint>((_S958.y)));
    uint e_4 = 0U;
    uint intact_0 = 0U;
    for(;;)
    {
        if(e_4 < _S960)
        {
        }
        else
        {
            break;
        }
        uint _S961 = w_alive_0(kernelContext_119);
        uint _S962 = adj_bond_0(_S959 + e_4, kernelContext_119);
        uint _S963;
        if((*(kernelContext_119->work_0+(_S961 + _S962))) != 0U)
        {
            _S963 = 1U;
        }
        else
        {
            _S963 = 0U;
        }
        uint intact_1 = intact_0 + _S963;
        e_4 = e_4 + 1U;
        intact_0 = intact_1;
    }
    return intact_0;
}

uint lp_l_0(uint k_33)
{
    return 22U * k_33 + 8U;
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

LpBond_0 lp_bond_0(uint i_8, KernelContext_0 thread* kernelContext_120)
{
    uint o_7 = 10U * i_8;
    thread LpBond_0 b_34;
    (&b_34)->centroid_0 = (float4(*(kernelContext_120->ext_0+o_7)) ).xyz;
    float3 _S964 = (float4(*(kernelContext_120->ext_0+(o_7 + 1U))) ).xyz;
    (&b_34)->n_4 = _S964;
    float3 _S965 = (float4(*(kernelContext_120->ext_0+(o_7 + 2U))) ).xyz;
    (&b_34)->t1_2 = _S965;
    (&b_34)->t2_0 = cross(_S964, _S965);
    (&(&b_34)->law_0)->geom0_0 = float4(*(kernelContext_120->ext_0+(o_7 + 3U))) ;
    (&(&b_34)->law_0)->geom1_0 = float4(*(kernelContext_120->ext_0+(o_7 + 4U))) ;
    (&(&b_34)->law_0)->stiff0_0 = float4(*(kernelContext_120->ext_0+(o_7 + 5U))) ;
    (&(&b_34)->law_0)->stiff1_0 = float4(*(kernelContext_120->ext_0+(o_7 + 6U))) ;
    (&(&b_34)->law_0)->rebar0_0 = float4(*(kernelContext_120->ext_0+(o_7 + 7U))) ;
    (&(&b_34)->law_0)->rebar1_0 = float4(*(kernelContext_120->ext_0+(o_7 + 8U))) ;
    (&(&b_34)->law_0)->ids_3 = (as_type<uint4>((float4(*(kernelContext_120->ext_0+(o_7 + 9U))) )));
    return b_34;
}

uint tri_0(uint i_9, uint j_4)
{
    return i_9 * (i_9 + 1U) / 2U + j_4;
}

bool lp_factor_0(uint k_34, KernelContext_0 thread* kernelContext_121)
{
    uint q_27;
    uint a_9;
    uint p2_0;
    uint q2_0;
    uint a2_0;
    float s_11;
    uint _S966 = lp_intact_0(k_34, kernelContext_121);
    uint _S967 = lp_l_0(k_34);
    uint _S968 = _S966 + 1U;
    if((as_type<uint>(((float4(*(kernelContext_121->lp_0+(_S967 + 5U))) ).w))) == _S968)
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
    uint _S969 = 4U * k_34;
    float4 _S970 = float4(*(kernelContext_121->ext_0+(kernelContext_121->params_0->ext_chunks_0 + _S969))) ;
    uint _S971 = (as_type<uint>((_S970.x)));
    uint _S972 = (as_type<uint>((_S970.y)));
    q_27 = 0U;
    for(;;)
    {
        if(q_27 < _S972)
        {
        }
        else
        {
            break;
        }
        uint _S973 = adj_bond_0(_S971 + q_27, kernelContext_121);
        uint _S974 = w_alive_0(kernelContext_121);
        if((*(kernelContext_121->work_0+(_S974 + _S973))) == 0U)
        {
            q_27 = q_27 + 1U;
            continue;
        }
        LpBond_0 _S975 = lp_bond_0(_S973, kernelContext_121);
        float3 _S976 = _S975.centroid_0 - (float4(*(kernelContext_121->chunks_0+_S969)) ).xyz;
        float _S977 = _S975.law_0.stiff0_0.y;
        array<float, int(6)> _S978 = { { _S977, _S977, _S975.law_0.stiff0_0.x, _S975.law_0.stiff0_0.z, _S975.law_0.stiff0_0.w, _S975.law_0.stiff1_0.x } };
        array<float3, int(3)> _S979 = { { _S975.t1_2, _S975.t2_0, _S975.n_4 } };
        thread array<array<float, int(6)>, int(6)> j_5;
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
            float3 c_19 = cross(_S976, _S979[a_9]);
            float _S980 = _S979[a_9].x;
            j_5[a_9][int(0)] = _S980;
            float _S981 = _S979[a_9].y;
            j_5[a_9][int(1)] = _S981;
            float _S982 = _S979[a_9].z;
            j_5[a_9][int(2)] = _S982;
            j_5[a_9][int(3)] = c_19.x;
            j_5[a_9][int(4)] = c_19.y;
            j_5[a_9][int(5)] = c_19.z;
            j_5[3U + a_9][int(0)] = 0.0f;
            j_5[3U + a_9][int(1)] = 0.0f;
            j_5[3U + a_9][int(2)] = 0.0f;
            j_5[3U + a_9][int(3)] = _S980;
            j_5[3U + a_9][int(4)] = _S981;
            j_5[3U + a_9][int(5)] = _S982;
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
                    float s_12 = s_11 + j_5[a2_0][p2_0] * _S978[a2_0] * j_5[a2_0][q2_0];
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
                bool _S983;
                if(ok_1)
                {
                    _S983 = s_11 > 0.0f;
                }
                else
                {
                    _S983 = false;
                }
                l_0[tri_0(p2_0, p2_0)] = sqrt(max(s_11, 1.00000000317107685e-30f));
                ok_1 = _S983;
            }
            else
            {
                l_0[tri_0(p2_0, q2_0)] = s_11 / l_0[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + 1U;
        }
        uint _S984 = p2_0 + 1U;
        ok_0 = ok_1;
        p2_0 = _S984;
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
        packed_float4 device* _S985 = kernelContext_121->lp_0+(_S967 + q2_0);
        uint _S986 = 4U * q2_0;
        float _S987 = l_0[_S986];
        uint _S988 = _S986 + 1U;
        if(_S988 < 21U)
        {
            s_11 = l_0[_S988];
        }
        else
        {
            s_11 = 0.0f;
        }
        uint _S989 = _S986 + 2U;
        float _S990;
        if(_S989 < 21U)
        {
            _S990 = l_0[_S989];
        }
        else
        {
            _S990 = 0.0f;
        }
        float _S991;
        if(q2_0 == 5U)
        {
            _S991 = (as_type<float>((_S968)));
        }
        else
        {
            _S991 = l_0[_S986 + 3U];
        }
        *_S985 = packed_float4(float4(_S987, s_11, _S990, _S991)) ;
        q2_0 = q2_0 + 1U;
    }
    return true;
}

[[kernel]] void fr_lp_block(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], atomic<uint> device* atom_33 [[buffer(5)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], packed_float4 device* lp_33 [[buffer(24)]], packed_float4 device* ext_35 [[buffer(23)]], packed_float4 device* lp_sides_33 [[buffer(28)]], uint device* engine_counters_33 [[buffer(26)]], packed_float4 device* engine_contacts_33 [[buffer(25)]], packed_float4 device* cs_anchors_33 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_122;
    (&kernelContext_122)->params_0 = params_33;
    (&kernelContext_122)->work_0 = work_33;
    (&kernelContext_122)->frames_0 = frames_33;
    (&kernelContext_122)->bodies_0 = bodies_33;
    (&kernelContext_122)->chunks_0 = chunks_33;
    (&kernelContext_122)->sort_0 = sort_33;
    (&kernelContext_122)->atom_0 = atom_33;
    (&kernelContext_122)->manifolds_0 = manifolds_33;
    (&kernelContext_122)->loads_0 = loads_33;
    (&kernelContext_122)->bonds_0 = bonds_33;
    (&kernelContext_122)->args_0 = args_33;
    (&kernelContext_122)->info_0 = info_33;
    (&kernelContext_122)->scan_0 = scan_33;
    (&kernelContext_122)->parts_0 = parts_33;
    (&kernelContext_122)->colliders_0 = colliders_33;
    (&kernelContext_122)->collider_info_0 = collider_info_33;
    (&kernelContext_122)->view_chunks_0 = view_chunks_33;
    (&kernelContext_122)->view_0 = view_33;
    (&kernelContext_122)->objects_0 = objects_33;
    (&kernelContext_122)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_122)->lines_0 = lines_33;
    (&kernelContext_122)->geometry_0 = geometry_33;
    (&kernelContext_122)->shots_0 = shots_33;
    (&kernelContext_122)->lp_0 = lp_33;
    (&kernelContext_122)->ext_0 = ext_35;
    (&kernelContext_122)->lp_sides_0 = lp_sides_33;
    (&kernelContext_122)->engine_counters_0 = engine_counters_33;
    (&kernelContext_122)->engine_contacts_0 = engine_contacts_33;
    (&kernelContext_122)->cs_anchors_0 = cs_anchors_33;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_122)->g_scan_0 = &g_scan_33;
    uint k_35 = id_30.x;
    bool _S992;
    if(k_35 >= (params_33->chunk_count_0))
    {
        _S992 = true;
    }
    else
    {
        _S992 = ((float4(*((&kernelContext_122)->lp_0+lp_f_0(k_35))) ).w) != 0.0f;
    }
    if(_S992)
    {
        return;
    }
    bool _S993 = lp_factor_0(k_35, &kernelContext_122);
    if(!_S993)
    {
        (*((&kernelContext_122)->lp_0+lp_f_0(k_35)))[int(3)] = 1.0f;
        uint _S994 = a_counters_1(&kernelContext_122);
        uint _S995 = atomic_fetch_or_explicit((&kernelContext_122)->atom_0+(_S994 + 3U), 32U, memory_order_relaxed);
    }
    return;
}

uint st_x_0(uint k_36, uint state_1, uint copy_0)
{
    return 22U * k_36 + state_1 + 2U * copy_0;
}

uint st_d_0(uint k_37, uint state_2)
{
    return 22U * k_37 + state_2 + 4U;
}

void st_sweep_0(uint k_38, uint state_3, uint from_4, float c1_1, float c2_1, KernelContext_0 thread* kernelContext_123)
{
    uint ka_0;
    uint kb_0;
    float s_14;
    uint to_3 = 1U - from_4;
    uint _S996 = st_x_0(k_38, state_3, from_4);
    float4 _S997 = float4(*(kernelContext_123->lp_0+_S996)) ;
    float4 _S998 = float4(*(kernelContext_123->lp_0+(_S996 + 1U))) ;
    uint _S999 = st_f_0(k_38, state_3);
    float4 _S1000 = float4(*(kernelContext_123->lp_0+_S999)) ;
    if((_S1000.w) != 0.0f)
    {
        uint _S1001 = st_x_0(k_38, state_3, to_3);
        *(kernelContext_123->lp_0+_S1001) = packed_float4(_S997) ;
        *(kernelContext_123->lp_0+(_S1001 + 1U)) = packed_float4(_S998) ;
        uint _S1002 = st_d_0(k_38, state_3);
        packed_float4 _S1003 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_123->lp_0+_S1002) = _S1003;
        *(kernelContext_123->lp_0+(_S1002 + 1U)) = _S1003;
        return;
    }
    float _S1004 = _S1000.y;
    float _S1005 = _S1000.z;
    thread array<float, int(6)> r_24;
    r_24[int(0)] = _S1000.x;
    r_24[int(1)] = _S1004;
    r_24[int(2)] = _S1005;
    r_24[int(3)] = 0.0f;
    r_24[int(4)] = 0.0f;
    r_24[int(5)] = 0.0f;
    float3 t_6 = (float4(*(kernelContext_123->lp_0+(_S999 + 1U))) ).xyz;
    r_24[int(3)] = t_6.x;
    r_24[int(4)] = t_6.y;
    r_24[int(5)] = t_6.z;
    float4 _S1006 = float4(*(kernelContext_123->ext_0+(kernelContext_123->params_0->ext_chunks_0 + 4U * k_38))) ;
    uint _S1007 = (as_type<uint>((_S1006.x)));
    uint _S1008 = (as_type<uint>((_S1006.y)));
    uint e_5 = 0U;
    for(;;)
    {
        if(e_5 < _S1008)
        {
        }
        else
        {
            break;
        }
        uint _S1009 = 5U * (_S1007 + e_5);
        float4 _S1010 = float4(*(kernelContext_123->lp_sides_0+_S1009)) ;
        uint _S1011 = w_alive_0(kernelContext_123);
        if((*(kernelContext_123->work_0+(_S1011 + (as_type<uint>((_S1010.w)))))) == 0U)
        {
            e_5 = e_5 + 1U;
            continue;
        }
        float4 _S1012 = float4(*(kernelContext_123->lp_sides_0+(_S1009 + 1U))) ;
        float4 _S1013 = float4(*(kernelContext_123->lp_sides_0+(_S1009 + 2U))) ;
        float4 _S1014 = float4(*(kernelContext_123->lp_sides_0+(_S1009 + 3U))) ;
        float4 _S1015 = float4(*(kernelContext_123->lp_sides_0+(_S1009 + 4U))) ;
        bool first_side_0 = (_S1015.w) != 0.0f;
        if(first_side_0)
        {
            ka_0 = k_38;
        }
        else
        {
            ka_0 = (as_type<uint>((_S1012.w)));
        }
        if(first_side_0)
        {
            kb_0 = (as_type<uint>((_S1012.w)));
        }
        else
        {
            kb_0 = k_38;
        }
        float3 ra_0 = _S1010.xyz;
        float3 rb_0 = _S1012.xyz;
        float3 t1_3 = _S1013.xyz;
        float3 n_5 = _S1014.xyz;
        float3 t2_1 = cross(n_5, t1_3);
        uint _S1016 = st_x_0(ka_0, state_3, from_4);
        uint _S1017 = st_x_0(kb_0, state_3, from_4);
        float3 _S1018 = (float4(*(kernelContext_123->lp_0+(_S1017 + 1U))) ).xyz;
        float3 _S1019 = (float4(*(kernelContext_123->lp_0+(_S1016 + 1U))) ).xyz;
        float3 delta_0 = (float4(*(kernelContext_123->lp_0+_S1017)) ).xyz + cross(_S1018, rb_0) - ((float4(*(kernelContext_123->lp_0+_S1016)) ).xyz + cross(_S1019, ra_0));
        float3 phi_0 = _S1018 - _S1019;
        float _S1020 = _S1013.w;
        float3 fb_0 = t1_3 * float3((_S1020 * dot(delta_0, t1_3)))  + t2_1 * float3((_S1020 * dot(delta_0, t2_1)))  + n_5 * float3((_S1014.w * dot(delta_0, n_5))) ;
        float3 mb_0 = t1_3 * float3((_S1015.x * dot(phi_0, t1_3)))  + t2_1 * float3((_S1015.y * dot(phi_0, t2_1)))  + n_5 * float3((_S1015.z * dot(phi_0, n_5))) ;
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
        float3 _S1021;
        if(first_side_0)
        {
            _S1021 = mb_0;
        }
        else
        {
            _S1021 = - mb_0;
        }
        float3 lm_0 = _S1021 + cross(arm_0, lf_0);
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
        float4 _S1022 = float4(*(kernelContext_123->lp_0+(lp_l_0(k_38) + kb_0))) ;
        uint _S1023 = 4U * kb_0;
        l_1[_S1023] = _S1022.x;
        uint _S1024 = _S1023 + 1U;
        if(_S1024 < 21U)
        {
            l_1[_S1024] = _S1022.y;
        }
        uint _S1025 = _S1023 + 2U;
        if(_S1025 < 21U)
        {
            l_1[_S1025] = _S1022.z;
        }
        uint _S1026 = _S1023 + 3U;
        if(_S1026 < 21U)
        {
            l_1[_S1026] = _S1022.w;
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
        uint _S1027 = uint(i3_0);
        z_3[i3_0] = s_14 / l_1[tri_0(_S1027, _S1027)];
        i3_0 = i3_0 - int(1);
    }
    uint _S1028 = st_d_0(k_38, state_3);
    float4 _S1029 = float4(*(kernelContext_123->lp_0+_S1028)) ;
    thread float4 d0_0 = _S1029;
    uint _S1030 = _S1028 + 1U;
    float4 _S1031 = float4(*(kernelContext_123->lp_0+_S1030)) ;
    thread float4 d1_0 = _S1031;
    float3 _S1032 = float3(c1_1) ;
    float3 _S1033 = float3(c2_1) ;
    d0_0.xyz = _S1032 * _S1029.xyz + _S1033 * float3(z_3[int(0)], z_3[int(1)], z_3[int(2)]);
    d1_0.xyz = _S1032 * _S1031.xyz + _S1033 * float3(z_3[int(3)], z_3[int(4)], z_3[int(5)]);
    *(kernelContext_123->lp_0+_S1028) = packed_float4(d0_0) ;
    *(kernelContext_123->lp_0+_S1030) = packed_float4(d1_0) ;
    uint _S1034 = st_x_0(k_38, state_3, to_3);
    *(kernelContext_123->lp_0+_S1034) = packed_float4(float4(_S997.xyz + d0_0.xyz, 0.0f)) ;
    *(kernelContext_123->lp_0+(_S1034 + 1U)) = packed_float4(float4(_S998.xyz + d1_0.xyz, 0.0f)) ;
    return;
}

void lp_sweep_0(uint k_39, uint from_5, KernelContext_0 thread* kernelContext_124)
{
    st_sweep_0(k_39, 0U, from_5, kernelContext_124->params_0->lp_c1_0, kernelContext_124->params_0->lp_c2_0, kernelContext_124);
    return;
}

[[kernel]] void fr_lp_sweep_even(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], atomic<uint> device* atom_34 [[buffer(5)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], packed_float4 device* lp_34 [[buffer(24)]], packed_float4 device* ext_36 [[buffer(23)]], packed_float4 device* lp_sides_34 [[buffer(28)]], uint device* engine_counters_34 [[buffer(26)]], packed_float4 device* engine_contacts_34 [[buffer(25)]], packed_float4 device* cs_anchors_34 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_125;
    (&kernelContext_125)->params_0 = params_34;
    (&kernelContext_125)->work_0 = work_34;
    (&kernelContext_125)->frames_0 = frames_34;
    (&kernelContext_125)->bodies_0 = bodies_34;
    (&kernelContext_125)->chunks_0 = chunks_34;
    (&kernelContext_125)->sort_0 = sort_34;
    (&kernelContext_125)->atom_0 = atom_34;
    (&kernelContext_125)->manifolds_0 = manifolds_34;
    (&kernelContext_125)->loads_0 = loads_34;
    (&kernelContext_125)->bonds_0 = bonds_34;
    (&kernelContext_125)->args_0 = args_34;
    (&kernelContext_125)->info_0 = info_34;
    (&kernelContext_125)->scan_0 = scan_34;
    (&kernelContext_125)->parts_0 = parts_34;
    (&kernelContext_125)->colliders_0 = colliders_34;
    (&kernelContext_125)->collider_info_0 = collider_info_34;
    (&kernelContext_125)->view_chunks_0 = view_chunks_34;
    (&kernelContext_125)->view_0 = view_34;
    (&kernelContext_125)->objects_0 = objects_34;
    (&kernelContext_125)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_125)->lines_0 = lines_34;
    (&kernelContext_125)->geometry_0 = geometry_34;
    (&kernelContext_125)->shots_0 = shots_34;
    (&kernelContext_125)->lp_0 = lp_34;
    (&kernelContext_125)->ext_0 = ext_36;
    (&kernelContext_125)->lp_sides_0 = lp_sides_34;
    (&kernelContext_125)->engine_counters_0 = engine_counters_34;
    (&kernelContext_125)->engine_contacts_0 = engine_contacts_34;
    (&kernelContext_125)->cs_anchors_0 = cs_anchors_34;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_125)->g_scan_0 = &g_scan_34;
    uint _S1035 = id_31.x;
    if(_S1035 < (params_34->chunk_count_0))
    {
        lp_sweep_0(_S1035, 0U, &kernelContext_125);
    }
    return;
}

[[kernel]] void fr_lp_sweep_odd(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], atomic<uint> device* atom_35 [[buffer(5)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], packed_float4 device* lp_35 [[buffer(24)]], packed_float4 device* ext_37 [[buffer(23)]], packed_float4 device* lp_sides_35 [[buffer(28)]], uint device* engine_counters_35 [[buffer(26)]], packed_float4 device* engine_contacts_35 [[buffer(25)]], packed_float4 device* cs_anchors_35 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_126;
    (&kernelContext_126)->params_0 = params_35;
    (&kernelContext_126)->work_0 = work_35;
    (&kernelContext_126)->frames_0 = frames_35;
    (&kernelContext_126)->bodies_0 = bodies_35;
    (&kernelContext_126)->chunks_0 = chunks_35;
    (&kernelContext_126)->sort_0 = sort_35;
    (&kernelContext_126)->atom_0 = atom_35;
    (&kernelContext_126)->manifolds_0 = manifolds_35;
    (&kernelContext_126)->loads_0 = loads_35;
    (&kernelContext_126)->bonds_0 = bonds_35;
    (&kernelContext_126)->args_0 = args_35;
    (&kernelContext_126)->info_0 = info_35;
    (&kernelContext_126)->scan_0 = scan_35;
    (&kernelContext_126)->parts_0 = parts_35;
    (&kernelContext_126)->colliders_0 = colliders_35;
    (&kernelContext_126)->collider_info_0 = collider_info_35;
    (&kernelContext_126)->view_chunks_0 = view_chunks_35;
    (&kernelContext_126)->view_0 = view_35;
    (&kernelContext_126)->objects_0 = objects_35;
    (&kernelContext_126)->shot_bodies_0 = shot_bodies_35;
    (&kernelContext_126)->lines_0 = lines_35;
    (&kernelContext_126)->geometry_0 = geometry_35;
    (&kernelContext_126)->shots_0 = shots_35;
    (&kernelContext_126)->lp_0 = lp_35;
    (&kernelContext_126)->ext_0 = ext_37;
    (&kernelContext_126)->lp_sides_0 = lp_sides_35;
    (&kernelContext_126)->engine_counters_0 = engine_counters_35;
    (&kernelContext_126)->engine_contacts_0 = engine_contacts_35;
    (&kernelContext_126)->cs_anchors_0 = cs_anchors_35;
    threadgroup array<uint, int(1024)> g_scan_35;
    (&kernelContext_126)->g_scan_0 = &g_scan_35;
    uint _S1036 = id_32.x;
    if(_S1036 < (params_35->chunk_count_0))
    {
        lp_sweep_0(_S1036, 1U, &kernelContext_126);
    }
    return;
}

void st_force_0(const LpBond_0 thread* b_35, uint ka_1, uint kb_1, uint state_4, uint c_20, float3 thread* q_lin_0, float3 thread* q_ang_0, KernelContext_0 thread* kernelContext_127)
{
    uint _S1037 = st_x_0(ka_1, state_4, c_20);
    uint _S1038 = st_x_0(kb_1, state_4, c_20);
    float3 _S1039 = (float4(*(kernelContext_127->lp_0+(_S1038 + 1U))) ).xyz;
    float3 _S1040 = (float4(*(kernelContext_127->lp_0+(_S1037 + 1U))) ).xyz;
    float3 delta_1 = (float4(*(kernelContext_127->lp_0+_S1038)) ).xyz + cross(_S1039, b_35->centroid_0 - (float4(*(kernelContext_127->chunks_0+4U * kb_1)) ).xyz) - ((float4(*(kernelContext_127->lp_0+_S1037)) ).xyz + cross(_S1040, b_35->centroid_0 - (float4(*(kernelContext_127->chunks_0+4U * ka_1)) ).xyz));
    float3 phi_1 = _S1039 - _S1040;
    float4 _S1041 = (&b_35->law_0)->stiff0_0;
    float kt_0 = (&b_35->law_0)->stiff1_0.x;
    float _S1042 = (&b_35->law_0)->stiff0_0.y;
    float3 _S1043 = b_35->t1_2;
    float3 _S1044 = b_35->t2_0;
    float3 _S1045 = b_35->n_4;
    *q_lin_0 = float3(_S1042 * dot(delta_1, b_35->t1_2), _S1042 * dot(delta_1, b_35->t2_0), (&b_35->law_0)->stiff0_0.x * dot(delta_1, b_35->n_4));
    *q_ang_0 = float3(_S1041.z * dot(phi_1, _S1043), _S1041.w * dot(phi_1, _S1044), kt_0 * dot(phi_1, _S1045));
    return;
}

void st_force_1(const LpBond_0 thread* b_36, uint ka_2, uint kb_2, uint state_5, uint c_21, float3 thread* q_lin_1, float3 thread* q_ang_1, KernelContext_0 thread* kernelContext_128)
{
    uint _S1046 = st_x_0(ka_2, state_5, c_21);
    uint _S1047 = st_x_0(kb_2, state_5, c_21);
    float3 _S1048 = (float4(*(kernelContext_128->lp_0+(_S1047 + 1U))) ).xyz;
    float3 _S1049 = (float4(*(kernelContext_128->lp_0+(_S1046 + 1U))) ).xyz;
    float3 delta_2 = (float4(*(kernelContext_128->lp_0+_S1047)) ).xyz + cross(_S1048, b_36->centroid_0 - (float4(*(kernelContext_128->chunks_0+4U * kb_2)) ).xyz) - ((float4(*(kernelContext_128->lp_0+_S1046)) ).xyz + cross(_S1049, b_36->centroid_0 - (float4(*(kernelContext_128->chunks_0+4U * ka_2)) ).xyz));
    float3 phi_2 = _S1048 - _S1049;
    float4 _S1050 = (&b_36->law_0)->stiff0_0;
    float kt_1 = (&b_36->law_0)->stiff1_0.x;
    float _S1051 = (&b_36->law_0)->stiff0_0.y;
    float3 _S1052 = b_36->t1_2;
    float3 _S1053 = b_36->t2_0;
    float3 _S1054 = b_36->n_4;
    *q_lin_1 = float3(_S1051 * dot(delta_2, b_36->t1_2), _S1051 * dot(delta_2, b_36->t2_0), (&b_36->law_0)->stiff0_0.x * dot(delta_2, b_36->n_4));
    *q_ang_1 = float3(_S1050.z * dot(phi_2, _S1052), _S1050.w * dot(phi_2, _S1053), kt_1 * dot(phi_2, _S1054));
    return;
}

void lp_force_0(const LpBond_0 thread* b_37, uint ka_3, uint kb_3, uint c_22, float3 thread* q_lin_2, float3 thread* q_ang_2, KernelContext_0 thread* kernelContext_129)
{
    st_force_1(b_37, ka_3, kb_3, 0U, c_22, q_lin_2, q_ang_2, kernelContext_129);
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

Measures_0 stress_measures_0(const JointBond_0 thread* b_38, float3 q_lin_3, float3 q_ang_3)
{
    float area_0 = b_38->geom0_0.x;
    float _S1055 = q_lin_3.z;
    float axial_0 = (metal::fast::divide((_S1055), (area_0)));
    float _S1056 = (metal::fast::divide((abs(q_ang_3.x)), (b_38->geom1_0.x)));
    float _S1057 = (metal::fast::divide((abs(q_ang_3.y)), (b_38->geom1_0.y)));
    float bending_0 = _S1056 + _S1057;
    float _S1058 = q_lin_3.x;
    float _S1059 = q_lin_3.y;
    float _S1060 = (metal::fast::sqrt((_S1058 * _S1058 + _S1059 * _S1059)));
    float _S1061 = (metal::fast::divide((_S1060), (area_0)));
    float _S1062 = (metal::fast::divide((abs(q_ang_3.z)), (b_38->geom0_0.w)));
    float shear_1 = _S1061 + _S1062;
    thread Measures_0 m_3;
    (&m_3)->tension_0 = axial_0 + bending_0;
    (&m_3)->shear_0 = shear_1;
    float _S1063 = - axial_0;
    (&m_3)->normal_compression_0 = max(_S1063, 0.0f);
    (&m_3)->compression_0 = _S1063 + bending_0;
    (&m_3)->compressive_force_0 = max(- _S1055, 0.0f);
    return m_3;
}

JointMaterial_0 lp_material_0(uint m_4, KernelContext_0 thread* kernelContext_130)
{
    uint o_8 = kernelContext_130->params_0->ext_mat_0 + 5U * m_4;
    thread JointMaterial_0 j_6;
    (&j_6)->strength_0 = float4(*(kernelContext_130->ext_0+o_8)) ;
    (&j_6)->energy_0 = float4(*(kernelContext_130->ext_0+(o_8 + 1U))) ;
    (&j_6)->dif_0 = float4(*(kernelContext_130->ext_0+(o_8 + 2U))) ;
    (&j_6)->misc_0 = float4(*(kernelContext_130->ext_0+(o_8 + 3U))) ;
    (&j_6)->kind_flags_0 = (as_type<uint4>((float4(*(kernelContext_130->ext_0+(o_8 + 4U))) )));
    return j_6;
}

float infinity_0()
{
    return float_bits_0(2139095040U);
}

float4 failure_indices_0(const JointMaterial_0 thread* mat_1, const JointBond_0 thread* b_39, const Measures_0 thread* m_5, float multiplier_0)
{
    float fc_2 = mat_1->strength_0.y * multiplier_0;
    float _S1064 = min(mat_1->strength_0.z * multiplier_0 + mat_1->strength_0.w * m_5->normal_compression_0, mat_1->energy_0.x * multiplier_0);
    thread float4 idx_0;
    float _S1065 = (metal::fast::divide((m_5->tension_0), (mat_1->strength_0.x * multiplier_0)));
    idx_0.x = max(_S1065, 0.0f);
    float _S1066;
    if(_S1064 > 0.0f)
    {
        float _S1067 = (metal::fast::divide((m_5->shear_0), (_S1064)));
        _S1066 = _S1067;
    }
    else
    {
        _S1066 = infinity_0();
    }
    idx_0.y = _S1066;
    float _S1068 = (metal::fast::divide((m_5->compression_0), (fc_2)));
    idx_0.z = max(_S1068, 0.0f);
    float _S1069 = b_39->stiff1_0.y;
    if(_S1069 > 0.0f)
    {
        float _S1070 = (metal::fast::divide((m_5->compressive_force_0), (_S1069)));
        _S1066 = _S1070;
    }
    else
    {
        _S1066 = 0.0f;
    }
    idx_0.w = _S1066;
    return idx_0;
}

[[kernel]] void fr_lp_breaks(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], atomic<uint> device* atom_36 [[buffer(5)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], packed_float4 device* lp_36 [[buffer(24)]], packed_float4 device* ext_38 [[buffer(23)]], packed_float4 device* lp_sides_36 [[buffer(28)]], uint device* engine_counters_36 [[buffer(26)]], packed_float4 device* engine_contacts_36 [[buffer(25)]], packed_float4 device* cs_anchors_36 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_131;
    (&kernelContext_131)->params_0 = params_36;
    (&kernelContext_131)->work_0 = work_36;
    (&kernelContext_131)->frames_0 = frames_36;
    (&kernelContext_131)->bodies_0 = bodies_36;
    (&kernelContext_131)->chunks_0 = chunks_36;
    (&kernelContext_131)->sort_0 = sort_36;
    (&kernelContext_131)->atom_0 = atom_36;
    (&kernelContext_131)->manifolds_0 = manifolds_36;
    (&kernelContext_131)->loads_0 = loads_36;
    (&kernelContext_131)->bonds_0 = bonds_36;
    (&kernelContext_131)->args_0 = args_36;
    (&kernelContext_131)->info_0 = info_36;
    (&kernelContext_131)->scan_0 = scan_36;
    (&kernelContext_131)->parts_0 = parts_36;
    (&kernelContext_131)->colliders_0 = colliders_36;
    (&kernelContext_131)->collider_info_0 = collider_info_36;
    (&kernelContext_131)->view_chunks_0 = view_chunks_36;
    (&kernelContext_131)->view_0 = view_36;
    (&kernelContext_131)->objects_0 = objects_36;
    (&kernelContext_131)->shot_bodies_0 = shot_bodies_36;
    (&kernelContext_131)->lines_0 = lines_36;
    (&kernelContext_131)->geometry_0 = geometry_36;
    (&kernelContext_131)->shots_0 = shots_36;
    (&kernelContext_131)->lp_0 = lp_36;
    (&kernelContext_131)->ext_0 = ext_38;
    (&kernelContext_131)->lp_sides_0 = lp_sides_36;
    (&kernelContext_131)->engine_counters_0 = engine_counters_36;
    (&kernelContext_131)->engine_contacts_0 = engine_contacts_36;
    (&kernelContext_131)->cs_anchors_0 = cs_anchors_36;
    threadgroup array<uint, int(1024)> g_scan_36;
    (&kernelContext_131)->g_scan_0 = &g_scan_36;
    uint i_10 = id_33.x;
    bool _S1071;
    if(i_10 >= (params_36->bond_count_0))
    {
        _S1071 = true;
    }
    else
    {
        uint _S1072 = w_alive_0(&kernelContext_131);
        _S1071 = (*((&kernelContext_131)->work_0+(_S1072 + i_10))) == 0U;
    }
    if(_S1071)
    {
        return;
    }
    uint4 _S1073 = uint4(*((&kernelContext_131)->bonds_0+i_10)) ;
    LpBond_0 _S1074 = lp_bond_0(i_10, &kernelContext_131);
    uint _S1075 = _S1073.x;
    uint _S1076 = _S1073.y;
    thread LpBond_0 _S1077 = _S1074;
    thread float3 q_lin_4;
    thread float3 q_ang_4;
    lp_force_0(&_S1077, _S1075, _S1076, 0U, &q_lin_4, &q_ang_4, &kernelContext_131);
    thread JointBond_0 _S1078 = _S1074.law_0;
    Measures_0 _S1079 = stress_measures_0(&_S1078, q_lin_4, q_ang_4);
    JointMaterial_0 _S1080 = lp_material_0(_S1074.law_0.ids_3.x, &kernelContext_131);
    float _S1081 = _S1074.law_0.geom1_0.w;
    thread JointMaterial_0 _S1082 = _S1080;
    thread JointBond_0 _S1083 = _S1074.law_0;
    thread Measures_0 _S1084 = _S1079;
    float4 _S1085 = failure_indices_0(&_S1082, &_S1083, &_S1084, _S1081);
    float _S1086 = max(max(_S1085.x, _S1085.y), max(_S1085.z, _S1085.w));
    uint _S1087 = a_counters_1(&kernelContext_131);
    uint _S1088 = atomic_fetch_max_explicit((&kernelContext_131)->atom_0+(_S1087 + 11U), (as_type<uint>((max(_S1086, 0.0f)))), memory_order_relaxed);
    if(_S1086 >= 1.0f)
    {
        uint _S1089 = w_alive_0(&kernelContext_131);
        *((&kernelContext_131)->work_0+(_S1089 + i_10)) = 0U;
        uint _S1090 = a_dirty_0(&kernelContext_131);
        uint _S1091 = body_of_0(_S1075, &kernelContext_131);
        uint _S1092 = atomic_fetch_or_explicit((&kernelContext_131)->atom_0+(_S1090 + _S1091), 1U, memory_order_relaxed);
        uint _S1093 = atomic_fetch_add_explicit((&kernelContext_131)->atom_0+_S1087, 1U, memory_order_relaxed);
        uint _S1094 = atomic_fetch_add_explicit((&kernelContext_131)->atom_0+(_S1087 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first_reset(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], atomic<uint> device* atom_37 [[buffer(5)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], packed_float4 device* lp_37 [[buffer(24)]], packed_float4 device* ext_39 [[buffer(23)]], packed_float4 device* lp_sides_37 [[buffer(28)]], uint device* engine_counters_37 [[buffer(26)]], packed_float4 device* engine_contacts_37 [[buffer(25)]], packed_float4 device* cs_anchors_37 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_132;
    (&kernelContext_132)->params_0 = params_37;
    (&kernelContext_132)->work_0 = work_37;
    (&kernelContext_132)->frames_0 = frames_37;
    (&kernelContext_132)->bodies_0 = bodies_37;
    (&kernelContext_132)->chunks_0 = chunks_37;
    (&kernelContext_132)->sort_0 = sort_37;
    (&kernelContext_132)->atom_0 = atom_37;
    (&kernelContext_132)->manifolds_0 = manifolds_37;
    (&kernelContext_132)->loads_0 = loads_37;
    (&kernelContext_132)->bonds_0 = bonds_37;
    (&kernelContext_132)->args_0 = args_37;
    (&kernelContext_132)->info_0 = info_37;
    (&kernelContext_132)->scan_0 = scan_37;
    (&kernelContext_132)->parts_0 = parts_37;
    (&kernelContext_132)->colliders_0 = colliders_37;
    (&kernelContext_132)->collider_info_0 = collider_info_37;
    (&kernelContext_132)->view_chunks_0 = view_chunks_37;
    (&kernelContext_132)->view_0 = view_37;
    (&kernelContext_132)->objects_0 = objects_37;
    (&kernelContext_132)->shot_bodies_0 = shot_bodies_37;
    (&kernelContext_132)->lines_0 = lines_37;
    (&kernelContext_132)->geometry_0 = geometry_37;
    (&kernelContext_132)->shots_0 = shots_37;
    (&kernelContext_132)->lp_0 = lp_37;
    (&kernelContext_132)->ext_0 = ext_39;
    (&kernelContext_132)->lp_sides_0 = lp_sides_37;
    (&kernelContext_132)->engine_counters_0 = engine_counters_37;
    (&kernelContext_132)->engine_contacts_0 = engine_contacts_37;
    (&kernelContext_132)->cs_anchors_0 = cs_anchors_37;
    threadgroup array<uint, int(1024)> g_scan_37;
    (&kernelContext_132)->g_scan_0 = &g_scan_37;
    uint _S1095 = id_34.x;
    if(_S1095 < (params_37->shot_count_0))
    {
        uint _S1096 = a_first_1(_S1095, &kernelContext_132);
        atomic_store_explicit((&kernelContext_132)->atom_0+_S1096, 2139095040U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], atomic<uint> device* atom_38 [[buffer(5)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], packed_float4 device* lp_38 [[buffer(24)]], packed_float4 device* ext_40 [[buffer(23)]], packed_float4 device* lp_sides_38 [[buffer(28)]], uint device* engine_counters_38 [[buffer(26)]], packed_float4 device* engine_contacts_38 [[buffer(25)]], packed_float4 device* cs_anchors_38 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_133;
    (&kernelContext_133)->params_0 = params_38;
    (&kernelContext_133)->work_0 = work_38;
    (&kernelContext_133)->frames_0 = frames_38;
    (&kernelContext_133)->bodies_0 = bodies_38;
    (&kernelContext_133)->chunks_0 = chunks_38;
    (&kernelContext_133)->sort_0 = sort_38;
    (&kernelContext_133)->atom_0 = atom_38;
    (&kernelContext_133)->manifolds_0 = manifolds_38;
    (&kernelContext_133)->loads_0 = loads_38;
    (&kernelContext_133)->bonds_0 = bonds_38;
    (&kernelContext_133)->args_0 = args_38;
    (&kernelContext_133)->info_0 = info_38;
    (&kernelContext_133)->scan_0 = scan_38;
    (&kernelContext_133)->parts_0 = parts_38;
    (&kernelContext_133)->colliders_0 = colliders_38;
    (&kernelContext_133)->collider_info_0 = collider_info_38;
    (&kernelContext_133)->view_chunks_0 = view_chunks_38;
    (&kernelContext_133)->view_0 = view_38;
    (&kernelContext_133)->objects_0 = objects_38;
    (&kernelContext_133)->shot_bodies_0 = shot_bodies_38;
    (&kernelContext_133)->lines_0 = lines_38;
    (&kernelContext_133)->geometry_0 = geometry_38;
    (&kernelContext_133)->shots_0 = shots_38;
    (&kernelContext_133)->lp_0 = lp_38;
    (&kernelContext_133)->ext_0 = ext_40;
    (&kernelContext_133)->lp_sides_0 = lp_sides_38;
    (&kernelContext_133)->engine_counters_0 = engine_counters_38;
    (&kernelContext_133)->engine_contacts_0 = engine_contacts_38;
    (&kernelContext_133)->cs_anchors_0 = cs_anchors_38;
    threadgroup array<uint, int(1024)> g_scan_38;
    (&kernelContext_133)->g_scan_0 = &g_scan_38;
    uint b_40 = id_35.x;
    bool _S1097;
    if(b_40 >= (params_38->body_count_0))
    {
        _S1097 = true;
    }
    else
    {
        _S1097 = ((&kernelContext_133)->params_0->shot_count_0) == 0U;
    }
    if(_S1097)
    {
        return;
    }
    thread uint shot_9;
    bool _S1098 = struck_0(b_40, &shot_9, &kernelContext_133);
    if(!_S1098)
    {
        return;
    }
    uint _S1099 = a_first_1(shot_9, &kernelContext_133);
    atomic<uint> device* _S1100 = (&kernelContext_133)->atom_0+_S1099;
    uint _S1101 = a_strike_0(&kernelContext_133);
    uint _S1102 = atomic_load_explicit((&kernelContext_133)->atom_0+(_S1101 + 9U * b_40 + 6U), memory_order_relaxed);
    uint _S1103 = atomic_fetch_min_explicit(_S1100, _S1102, memory_order_relaxed);
    return;
}

bool cs_struck_1(uint b_41, KernelContext_0 thread* kernelContext_134)
{
    bool _S1104;
    if(b_41 < (kernelContext_134->params_0->body_count_0))
    {
        uint _S1105 = a_cs_0(b_41, kernelContext_134);
        uint _S1106 = atomic_load_explicit(kernelContext_134->atom_0+(_S1105 + 1U), memory_order_relaxed);
        _S1104 = _S1106 != 0U;
    }
    else
    {
        _S1104 = false;
    }
    return _S1104;
}

[[kernel]] void fr_il_begin(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], atomic<uint> device* atom_39 [[buffer(5)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], packed_float4 device* lp_39 [[buffer(24)]], packed_float4 device* ext_41 [[buffer(23)]], packed_float4 device* lp_sides_39 [[buffer(28)]], uint device* engine_counters_39 [[buffer(26)]], packed_float4 device* engine_contacts_39 [[buffer(25)]], packed_float4 device* cs_anchors_39 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_135;
    (&kernelContext_135)->params_0 = params_39;
    (&kernelContext_135)->work_0 = work_39;
    (&kernelContext_135)->frames_0 = frames_39;
    (&kernelContext_135)->bodies_0 = bodies_39;
    (&kernelContext_135)->chunks_0 = chunks_39;
    (&kernelContext_135)->sort_0 = sort_39;
    (&kernelContext_135)->atom_0 = atom_39;
    (&kernelContext_135)->manifolds_0 = manifolds_39;
    (&kernelContext_135)->loads_0 = loads_39;
    (&kernelContext_135)->bonds_0 = bonds_39;
    (&kernelContext_135)->args_0 = args_39;
    (&kernelContext_135)->info_0 = info_39;
    (&kernelContext_135)->scan_0 = scan_39;
    (&kernelContext_135)->parts_0 = parts_39;
    (&kernelContext_135)->colliders_0 = colliders_39;
    (&kernelContext_135)->collider_info_0 = collider_info_39;
    (&kernelContext_135)->view_chunks_0 = view_chunks_39;
    (&kernelContext_135)->view_0 = view_39;
    (&kernelContext_135)->objects_0 = objects_39;
    (&kernelContext_135)->shot_bodies_0 = shot_bodies_39;
    (&kernelContext_135)->lines_0 = lines_39;
    (&kernelContext_135)->geometry_0 = geometry_39;
    (&kernelContext_135)->shots_0 = shots_39;
    (&kernelContext_135)->lp_0 = lp_39;
    (&kernelContext_135)->ext_0 = ext_41;
    (&kernelContext_135)->lp_sides_0 = lp_sides_39;
    (&kernelContext_135)->engine_counters_0 = engine_counters_39;
    (&kernelContext_135)->engine_contacts_0 = engine_contacts_39;
    (&kernelContext_135)->cs_anchors_0 = cs_anchors_39;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_135)->g_scan_0 = &g_scan_39;
    if((params_39->shot_count_0) > 0U)
    {
        il_begin_1(id_36.x, &kernelContext_135);
    }
    return;
}

[[kernel]] void fr_il_extend_even(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], atomic<uint> device* atom_40 [[buffer(5)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], packed_float4 device* lp_40 [[buffer(24)]], packed_float4 device* ext_42 [[buffer(23)]], packed_float4 device* lp_sides_40 [[buffer(28)]], uint device* engine_counters_40 [[buffer(26)]], packed_float4 device* engine_contacts_40 [[buffer(25)]], packed_float4 device* cs_anchors_40 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_136;
    (&kernelContext_136)->params_0 = params_40;
    (&kernelContext_136)->work_0 = work_40;
    (&kernelContext_136)->frames_0 = frames_40;
    (&kernelContext_136)->bodies_0 = bodies_40;
    (&kernelContext_136)->chunks_0 = chunks_40;
    (&kernelContext_136)->sort_0 = sort_40;
    (&kernelContext_136)->atom_0 = atom_40;
    (&kernelContext_136)->manifolds_0 = manifolds_40;
    (&kernelContext_136)->loads_0 = loads_40;
    (&kernelContext_136)->bonds_0 = bonds_40;
    (&kernelContext_136)->args_0 = args_40;
    (&kernelContext_136)->info_0 = info_40;
    (&kernelContext_136)->scan_0 = scan_40;
    (&kernelContext_136)->parts_0 = parts_40;
    (&kernelContext_136)->colliders_0 = colliders_40;
    (&kernelContext_136)->collider_info_0 = collider_info_40;
    (&kernelContext_136)->view_chunks_0 = view_chunks_40;
    (&kernelContext_136)->view_0 = view_40;
    (&kernelContext_136)->objects_0 = objects_40;
    (&kernelContext_136)->shot_bodies_0 = shot_bodies_40;
    (&kernelContext_136)->lines_0 = lines_40;
    (&kernelContext_136)->geometry_0 = geometry_40;
    (&kernelContext_136)->shots_0 = shots_40;
    (&kernelContext_136)->lp_0 = lp_40;
    (&kernelContext_136)->ext_0 = ext_42;
    (&kernelContext_136)->lp_sides_0 = lp_sides_40;
    (&kernelContext_136)->engine_counters_0 = engine_counters_40;
    (&kernelContext_136)->engine_contacts_0 = engine_contacts_40;
    (&kernelContext_136)->cs_anchors_0 = cs_anchors_40;
    threadgroup array<uint, int(1024)> g_scan_40;
    (&kernelContext_136)->g_scan_0 = &g_scan_40;
    if((params_40->shot_count_0) > 0U)
    {
        il_extend_1(id_37.x, 0U, &kernelContext_136);
    }
    return;
}

[[kernel]] void fr_il_extend_odd(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], atomic<uint> device* atom_41 [[buffer(5)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], packed_float4 device* lp_41 [[buffer(24)]], packed_float4 device* ext_43 [[buffer(23)]], packed_float4 device* lp_sides_41 [[buffer(28)]], uint device* engine_counters_41 [[buffer(26)]], packed_float4 device* engine_contacts_41 [[buffer(25)]], packed_float4 device* cs_anchors_41 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_137;
    (&kernelContext_137)->params_0 = params_41;
    (&kernelContext_137)->work_0 = work_41;
    (&kernelContext_137)->frames_0 = frames_41;
    (&kernelContext_137)->bodies_0 = bodies_41;
    (&kernelContext_137)->chunks_0 = chunks_41;
    (&kernelContext_137)->sort_0 = sort_41;
    (&kernelContext_137)->atom_0 = atom_41;
    (&kernelContext_137)->manifolds_0 = manifolds_41;
    (&kernelContext_137)->loads_0 = loads_41;
    (&kernelContext_137)->bonds_0 = bonds_41;
    (&kernelContext_137)->args_0 = args_41;
    (&kernelContext_137)->info_0 = info_41;
    (&kernelContext_137)->scan_0 = scan_41;
    (&kernelContext_137)->parts_0 = parts_41;
    (&kernelContext_137)->colliders_0 = colliders_41;
    (&kernelContext_137)->collider_info_0 = collider_info_41;
    (&kernelContext_137)->view_chunks_0 = view_chunks_41;
    (&kernelContext_137)->view_0 = view_41;
    (&kernelContext_137)->objects_0 = objects_41;
    (&kernelContext_137)->shot_bodies_0 = shot_bodies_41;
    (&kernelContext_137)->lines_0 = lines_41;
    (&kernelContext_137)->geometry_0 = geometry_41;
    (&kernelContext_137)->shots_0 = shots_41;
    (&kernelContext_137)->lp_0 = lp_41;
    (&kernelContext_137)->ext_0 = ext_43;
    (&kernelContext_137)->lp_sides_0 = lp_sides_41;
    (&kernelContext_137)->engine_counters_0 = engine_counters_41;
    (&kernelContext_137)->engine_contacts_0 = engine_contacts_41;
    (&kernelContext_137)->cs_anchors_0 = cs_anchors_41;
    threadgroup array<uint, int(1024)> g_scan_41;
    (&kernelContext_137)->g_scan_0 = &g_scan_41;
    if((params_41->shot_count_0) > 0U)
    {
        il_extend_1(id_38.x, 1U, &kernelContext_137);
    }
    return;
}

[[kernel]] void fr_il_strike(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], atomic<uint> device* atom_42 [[buffer(5)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], packed_float4 device* lp_42 [[buffer(24)]], packed_float4 device* ext_44 [[buffer(23)]], packed_float4 device* lp_sides_42 [[buffer(28)]], uint device* engine_counters_42 [[buffer(26)]], packed_float4 device* engine_contacts_42 [[buffer(25)]], packed_float4 device* cs_anchors_42 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_138;
    (&kernelContext_138)->params_0 = params_42;
    (&kernelContext_138)->work_0 = work_42;
    (&kernelContext_138)->frames_0 = frames_42;
    (&kernelContext_138)->bodies_0 = bodies_42;
    (&kernelContext_138)->chunks_0 = chunks_42;
    (&kernelContext_138)->sort_0 = sort_42;
    (&kernelContext_138)->atom_0 = atom_42;
    (&kernelContext_138)->manifolds_0 = manifolds_42;
    (&kernelContext_138)->loads_0 = loads_42;
    (&kernelContext_138)->bonds_0 = bonds_42;
    (&kernelContext_138)->args_0 = args_42;
    (&kernelContext_138)->info_0 = info_42;
    (&kernelContext_138)->scan_0 = scan_42;
    (&kernelContext_138)->parts_0 = parts_42;
    (&kernelContext_138)->colliders_0 = colliders_42;
    (&kernelContext_138)->collider_info_0 = collider_info_42;
    (&kernelContext_138)->view_chunks_0 = view_chunks_42;
    (&kernelContext_138)->view_0 = view_42;
    (&kernelContext_138)->objects_0 = objects_42;
    (&kernelContext_138)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_138)->lines_0 = lines_42;
    (&kernelContext_138)->geometry_0 = geometry_42;
    (&kernelContext_138)->shots_0 = shots_42;
    (&kernelContext_138)->lp_0 = lp_42;
    (&kernelContext_138)->ext_0 = ext_44;
    (&kernelContext_138)->lp_sides_0 = lp_sides_42;
    (&kernelContext_138)->engine_counters_0 = engine_counters_42;
    (&kernelContext_138)->engine_contacts_0 = engine_contacts_42;
    (&kernelContext_138)->cs_anchors_0 = cs_anchors_42;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_138)->g_scan_0 = &g_scan_42;
    if((params_42->shot_count_0) > 0U)
    {
        il_strike_1(id_39.x, &kernelContext_138);
    }
    return;
}

[[kernel]] void fr_il_mark(uint3 id_40 [[thread_position_in_grid]], FractureParams_0 constant* params_43 [[buffer(0)]], uint device* work_43 [[buffer(4)]], packed_float4 device* frames_43 [[buffer(6)]], packed_float4 device* bodies_43 [[buffer(7)]], packed_float4 device* chunks_43 [[buffer(1)]], uint device* sort_43 [[buffer(11)]], atomic<uint> device* atom_43 [[buffer(5)]], Record_natural_0 device* manifolds_43 [[buffer(10)]], packed_float4 device* loads_43 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_43 [[buffer(13)]], uint device* info_43 [[buffer(15)]], ScanParams_0 constant* scan_43 [[buffer(14)]], packed_float4 device* parts_43 [[buffer(3)]], packed_float4 device* colliders_43 [[buffer(8)]], packed_uint4 device* collider_info_43 [[buffer(9)]], packed_float4 device* view_chunks_43 [[buffer(17)]], ViewParams_0 constant* view_43 [[buffer(16)]], packed_float4 device* objects_43 [[buffer(18)]], uint device* shot_bodies_43 [[buffer(20)]], float device* lines_43 [[buffer(19)]], packed_float4 device* geometry_43 [[buffer(21)]], packed_float4 device* shots_43 [[buffer(22)]], packed_float4 device* lp_43 [[buffer(24)]], packed_float4 device* ext_45 [[buffer(23)]], packed_float4 device* lp_sides_43 [[buffer(28)]], uint device* engine_counters_43 [[buffer(26)]], packed_float4 device* engine_contacts_43 [[buffer(25)]], packed_float4 device* cs_anchors_43 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_139;
    (&kernelContext_139)->params_0 = params_43;
    (&kernelContext_139)->work_0 = work_43;
    (&kernelContext_139)->frames_0 = frames_43;
    (&kernelContext_139)->bodies_0 = bodies_43;
    (&kernelContext_139)->chunks_0 = chunks_43;
    (&kernelContext_139)->sort_0 = sort_43;
    (&kernelContext_139)->atom_0 = atom_43;
    (&kernelContext_139)->manifolds_0 = manifolds_43;
    (&kernelContext_139)->loads_0 = loads_43;
    (&kernelContext_139)->bonds_0 = bonds_43;
    (&kernelContext_139)->args_0 = args_43;
    (&kernelContext_139)->info_0 = info_43;
    (&kernelContext_139)->scan_0 = scan_43;
    (&kernelContext_139)->parts_0 = parts_43;
    (&kernelContext_139)->colliders_0 = colliders_43;
    (&kernelContext_139)->collider_info_0 = collider_info_43;
    (&kernelContext_139)->view_chunks_0 = view_chunks_43;
    (&kernelContext_139)->view_0 = view_43;
    (&kernelContext_139)->objects_0 = objects_43;
    (&kernelContext_139)->shot_bodies_0 = shot_bodies_43;
    (&kernelContext_139)->lines_0 = lines_43;
    (&kernelContext_139)->geometry_0 = geometry_43;
    (&kernelContext_139)->shots_0 = shots_43;
    (&kernelContext_139)->lp_0 = lp_43;
    (&kernelContext_139)->ext_0 = ext_45;
    (&kernelContext_139)->lp_sides_0 = lp_sides_43;
    (&kernelContext_139)->engine_counters_0 = engine_counters_43;
    (&kernelContext_139)->engine_contacts_0 = engine_contacts_43;
    (&kernelContext_139)->cs_anchors_0 = cs_anchors_43;
    threadgroup array<uint, int(1024)> g_scan_43;
    (&kernelContext_139)->g_scan_0 = &g_scan_43;
    if((params_43->shot_count_0) > 0U)
    {
        il_mark_1(id_40.x, &kernelContext_139);
    }
    return;
}

[[kernel]] void fr_il_carry(uint3 id_41 [[thread_position_in_grid]], FractureParams_0 constant* params_44 [[buffer(0)]], uint device* work_44 [[buffer(4)]], packed_float4 device* frames_44 [[buffer(6)]], packed_float4 device* bodies_44 [[buffer(7)]], packed_float4 device* chunks_44 [[buffer(1)]], uint device* sort_44 [[buffer(11)]], atomic<uint> device* atom_44 [[buffer(5)]], Record_natural_0 device* manifolds_44 [[buffer(10)]], packed_float4 device* loads_44 [[buffer(12)]], packed_uint4 device* bonds_44 [[buffer(2)]], uint device* args_44 [[buffer(13)]], uint device* info_44 [[buffer(15)]], ScanParams_0 constant* scan_44 [[buffer(14)]], packed_float4 device* parts_44 [[buffer(3)]], packed_float4 device* colliders_44 [[buffer(8)]], packed_uint4 device* collider_info_44 [[buffer(9)]], packed_float4 device* view_chunks_44 [[buffer(17)]], ViewParams_0 constant* view_44 [[buffer(16)]], packed_float4 device* objects_44 [[buffer(18)]], uint device* shot_bodies_44 [[buffer(20)]], float device* lines_44 [[buffer(19)]], packed_float4 device* geometry_44 [[buffer(21)]], packed_float4 device* shots_44 [[buffer(22)]], packed_float4 device* lp_44 [[buffer(24)]], packed_float4 device* ext_46 [[buffer(23)]], packed_float4 device* lp_sides_44 [[buffer(28)]], uint device* engine_counters_44 [[buffer(26)]], packed_float4 device* engine_contacts_44 [[buffer(25)]], packed_float4 device* cs_anchors_44 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_140;
    (&kernelContext_140)->params_0 = params_44;
    (&kernelContext_140)->work_0 = work_44;
    (&kernelContext_140)->frames_0 = frames_44;
    (&kernelContext_140)->bodies_0 = bodies_44;
    (&kernelContext_140)->chunks_0 = chunks_44;
    (&kernelContext_140)->sort_0 = sort_44;
    (&kernelContext_140)->atom_0 = atom_44;
    (&kernelContext_140)->manifolds_0 = manifolds_44;
    (&kernelContext_140)->loads_0 = loads_44;
    (&kernelContext_140)->bonds_0 = bonds_44;
    (&kernelContext_140)->args_0 = args_44;
    (&kernelContext_140)->info_0 = info_44;
    (&kernelContext_140)->scan_0 = scan_44;
    (&kernelContext_140)->parts_0 = parts_44;
    (&kernelContext_140)->colliders_0 = colliders_44;
    (&kernelContext_140)->collider_info_0 = collider_info_44;
    (&kernelContext_140)->view_chunks_0 = view_chunks_44;
    (&kernelContext_140)->view_0 = view_44;
    (&kernelContext_140)->objects_0 = objects_44;
    (&kernelContext_140)->shot_bodies_0 = shot_bodies_44;
    (&kernelContext_140)->lines_0 = lines_44;
    (&kernelContext_140)->geometry_0 = geometry_44;
    (&kernelContext_140)->shots_0 = shots_44;
    (&kernelContext_140)->lp_0 = lp_44;
    (&kernelContext_140)->ext_0 = ext_46;
    (&kernelContext_140)->lp_sides_0 = lp_sides_44;
    (&kernelContext_140)->engine_counters_0 = engine_counters_44;
    (&kernelContext_140)->engine_contacts_0 = engine_contacts_44;
    (&kernelContext_140)->cs_anchors_0 = cs_anchors_44;
    threadgroup array<uint, int(1024)> g_scan_44;
    (&kernelContext_140)->g_scan_0 = &g_scan_44;
    if((params_44->shot_count_0) > 0U)
    {
        il_carry_1(id_41.x, &kernelContext_140);
    }
    return;
}

[[kernel]] void fr_il_breaks(uint3 id_42 [[thread_position_in_grid]], FractureParams_0 constant* params_45 [[buffer(0)]], uint device* work_45 [[buffer(4)]], packed_float4 device* frames_45 [[buffer(6)]], packed_float4 device* bodies_45 [[buffer(7)]], packed_float4 device* chunks_45 [[buffer(1)]], uint device* sort_45 [[buffer(11)]], atomic<uint> device* atom_45 [[buffer(5)]], Record_natural_0 device* manifolds_45 [[buffer(10)]], packed_float4 device* loads_45 [[buffer(12)]], packed_uint4 device* bonds_45 [[buffer(2)]], uint device* args_45 [[buffer(13)]], uint device* info_45 [[buffer(15)]], ScanParams_0 constant* scan_45 [[buffer(14)]], packed_float4 device* parts_45 [[buffer(3)]], packed_float4 device* colliders_45 [[buffer(8)]], packed_uint4 device* collider_info_45 [[buffer(9)]], packed_float4 device* view_chunks_45 [[buffer(17)]], ViewParams_0 constant* view_45 [[buffer(16)]], packed_float4 device* objects_45 [[buffer(18)]], uint device* shot_bodies_45 [[buffer(20)]], float device* lines_45 [[buffer(19)]], packed_float4 device* geometry_45 [[buffer(21)]], packed_float4 device* shots_45 [[buffer(22)]], packed_float4 device* lp_45 [[buffer(24)]], packed_float4 device* ext_47 [[buffer(23)]], packed_float4 device* lp_sides_45 [[buffer(28)]], uint device* engine_counters_45 [[buffer(26)]], packed_float4 device* engine_contacts_45 [[buffer(25)]], packed_float4 device* cs_anchors_45 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_141;
    (&kernelContext_141)->params_0 = params_45;
    (&kernelContext_141)->work_0 = work_45;
    (&kernelContext_141)->frames_0 = frames_45;
    (&kernelContext_141)->bodies_0 = bodies_45;
    (&kernelContext_141)->chunks_0 = chunks_45;
    (&kernelContext_141)->sort_0 = sort_45;
    (&kernelContext_141)->atom_0 = atom_45;
    (&kernelContext_141)->manifolds_0 = manifolds_45;
    (&kernelContext_141)->loads_0 = loads_45;
    (&kernelContext_141)->bonds_0 = bonds_45;
    (&kernelContext_141)->args_0 = args_45;
    (&kernelContext_141)->info_0 = info_45;
    (&kernelContext_141)->scan_0 = scan_45;
    (&kernelContext_141)->parts_0 = parts_45;
    (&kernelContext_141)->colliders_0 = colliders_45;
    (&kernelContext_141)->collider_info_0 = collider_info_45;
    (&kernelContext_141)->view_chunks_0 = view_chunks_45;
    (&kernelContext_141)->view_0 = view_45;
    (&kernelContext_141)->objects_0 = objects_45;
    (&kernelContext_141)->shot_bodies_0 = shot_bodies_45;
    (&kernelContext_141)->lines_0 = lines_45;
    (&kernelContext_141)->geometry_0 = geometry_45;
    (&kernelContext_141)->shots_0 = shots_45;
    (&kernelContext_141)->lp_0 = lp_45;
    (&kernelContext_141)->ext_0 = ext_47;
    (&kernelContext_141)->lp_sides_0 = lp_sides_45;
    (&kernelContext_141)->engine_counters_0 = engine_counters_45;
    (&kernelContext_141)->engine_contacts_0 = engine_contacts_45;
    (&kernelContext_141)->cs_anchors_0 = cs_anchors_45;
    threadgroup array<uint, int(1024)> g_scan_45;
    (&kernelContext_141)->g_scan_0 = &g_scan_45;
    if((params_45->shot_count_0) > 0U)
    {
        il_breaks_1(id_42.x, &kernelContext_141);
    }
    return;
}

[[kernel]] void fr_csl_begin(uint3 id_43 [[thread_position_in_grid]], FractureParams_0 constant* params_46 [[buffer(0)]], uint device* work_46 [[buffer(4)]], packed_float4 device* frames_46 [[buffer(6)]], packed_float4 device* bodies_46 [[buffer(7)]], packed_float4 device* chunks_46 [[buffer(1)]], uint device* sort_46 [[buffer(11)]], atomic<uint> device* atom_46 [[buffer(5)]], Record_natural_0 device* manifolds_46 [[buffer(10)]], packed_float4 device* loads_46 [[buffer(12)]], packed_uint4 device* bonds_46 [[buffer(2)]], uint device* args_46 [[buffer(13)]], uint device* info_46 [[buffer(15)]], ScanParams_0 constant* scan_46 [[buffer(14)]], packed_float4 device* parts_46 [[buffer(3)]], packed_float4 device* colliders_46 [[buffer(8)]], packed_uint4 device* collider_info_46 [[buffer(9)]], packed_float4 device* view_chunks_46 [[buffer(17)]], ViewParams_0 constant* view_46 [[buffer(16)]], packed_float4 device* objects_46 [[buffer(18)]], uint device* shot_bodies_46 [[buffer(20)]], float device* lines_46 [[buffer(19)]], packed_float4 device* geometry_46 [[buffer(21)]], packed_float4 device* shots_46 [[buffer(22)]], packed_float4 device* lp_46 [[buffer(24)]], packed_float4 device* ext_48 [[buffer(23)]], packed_float4 device* lp_sides_46 [[buffer(28)]], uint device* engine_counters_46 [[buffer(26)]], packed_float4 device* engine_contacts_46 [[buffer(25)]], packed_float4 device* cs_anchors_46 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_142;
    (&kernelContext_142)->params_0 = params_46;
    (&kernelContext_142)->work_0 = work_46;
    (&kernelContext_142)->frames_0 = frames_46;
    (&kernelContext_142)->bodies_0 = bodies_46;
    (&kernelContext_142)->chunks_0 = chunks_46;
    (&kernelContext_142)->sort_0 = sort_46;
    (&kernelContext_142)->atom_0 = atom_46;
    (&kernelContext_142)->manifolds_0 = manifolds_46;
    (&kernelContext_142)->loads_0 = loads_46;
    (&kernelContext_142)->bonds_0 = bonds_46;
    (&kernelContext_142)->args_0 = args_46;
    (&kernelContext_142)->info_0 = info_46;
    (&kernelContext_142)->scan_0 = scan_46;
    (&kernelContext_142)->parts_0 = parts_46;
    (&kernelContext_142)->colliders_0 = colliders_46;
    (&kernelContext_142)->collider_info_0 = collider_info_46;
    (&kernelContext_142)->view_chunks_0 = view_chunks_46;
    (&kernelContext_142)->view_0 = view_46;
    (&kernelContext_142)->objects_0 = objects_46;
    (&kernelContext_142)->shot_bodies_0 = shot_bodies_46;
    (&kernelContext_142)->lines_0 = lines_46;
    (&kernelContext_142)->geometry_0 = geometry_46;
    (&kernelContext_142)->shots_0 = shots_46;
    (&kernelContext_142)->lp_0 = lp_46;
    (&kernelContext_142)->ext_0 = ext_48;
    (&kernelContext_142)->lp_sides_0 = lp_sides_46;
    (&kernelContext_142)->engine_counters_0 = engine_counters_46;
    (&kernelContext_142)->engine_contacts_0 = engine_contacts_46;
    (&kernelContext_142)->cs_anchors_0 = cs_anchors_46;
    threadgroup array<uint, int(1024)> g_scan_46;
    (&kernelContext_142)->g_scan_0 = &g_scan_46;
    il_begin_0(id_43.x, &kernelContext_142);
    return;
}

[[kernel]] void fr_csl_extend_even(uint3 id_44 [[thread_position_in_grid]], FractureParams_0 constant* params_47 [[buffer(0)]], uint device* work_47 [[buffer(4)]], packed_float4 device* frames_47 [[buffer(6)]], packed_float4 device* bodies_47 [[buffer(7)]], packed_float4 device* chunks_47 [[buffer(1)]], uint device* sort_47 [[buffer(11)]], atomic<uint> device* atom_47 [[buffer(5)]], Record_natural_0 device* manifolds_47 [[buffer(10)]], packed_float4 device* loads_47 [[buffer(12)]], packed_uint4 device* bonds_47 [[buffer(2)]], uint device* args_47 [[buffer(13)]], uint device* info_47 [[buffer(15)]], ScanParams_0 constant* scan_47 [[buffer(14)]], packed_float4 device* parts_47 [[buffer(3)]], packed_float4 device* colliders_47 [[buffer(8)]], packed_uint4 device* collider_info_47 [[buffer(9)]], packed_float4 device* view_chunks_47 [[buffer(17)]], ViewParams_0 constant* view_47 [[buffer(16)]], packed_float4 device* objects_47 [[buffer(18)]], uint device* shot_bodies_47 [[buffer(20)]], float device* lines_47 [[buffer(19)]], packed_float4 device* geometry_47 [[buffer(21)]], packed_float4 device* shots_47 [[buffer(22)]], packed_float4 device* lp_47 [[buffer(24)]], packed_float4 device* ext_49 [[buffer(23)]], packed_float4 device* lp_sides_47 [[buffer(28)]], uint device* engine_counters_47 [[buffer(26)]], packed_float4 device* engine_contacts_47 [[buffer(25)]], packed_float4 device* cs_anchors_47 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_143;
    (&kernelContext_143)->params_0 = params_47;
    (&kernelContext_143)->work_0 = work_47;
    (&kernelContext_143)->frames_0 = frames_47;
    (&kernelContext_143)->bodies_0 = bodies_47;
    (&kernelContext_143)->chunks_0 = chunks_47;
    (&kernelContext_143)->sort_0 = sort_47;
    (&kernelContext_143)->atom_0 = atom_47;
    (&kernelContext_143)->manifolds_0 = manifolds_47;
    (&kernelContext_143)->loads_0 = loads_47;
    (&kernelContext_143)->bonds_0 = bonds_47;
    (&kernelContext_143)->args_0 = args_47;
    (&kernelContext_143)->info_0 = info_47;
    (&kernelContext_143)->scan_0 = scan_47;
    (&kernelContext_143)->parts_0 = parts_47;
    (&kernelContext_143)->colliders_0 = colliders_47;
    (&kernelContext_143)->collider_info_0 = collider_info_47;
    (&kernelContext_143)->view_chunks_0 = view_chunks_47;
    (&kernelContext_143)->view_0 = view_47;
    (&kernelContext_143)->objects_0 = objects_47;
    (&kernelContext_143)->shot_bodies_0 = shot_bodies_47;
    (&kernelContext_143)->lines_0 = lines_47;
    (&kernelContext_143)->geometry_0 = geometry_47;
    (&kernelContext_143)->shots_0 = shots_47;
    (&kernelContext_143)->lp_0 = lp_47;
    (&kernelContext_143)->ext_0 = ext_49;
    (&kernelContext_143)->lp_sides_0 = lp_sides_47;
    (&kernelContext_143)->engine_counters_0 = engine_counters_47;
    (&kernelContext_143)->engine_contacts_0 = engine_contacts_47;
    (&kernelContext_143)->cs_anchors_0 = cs_anchors_47;
    threadgroup array<uint, int(1024)> g_scan_47;
    (&kernelContext_143)->g_scan_0 = &g_scan_47;
    il_extend_0(id_44.x, 0U, &kernelContext_143);
    return;
}

[[kernel]] void fr_csl_extend_odd(uint3 id_45 [[thread_position_in_grid]], FractureParams_0 constant* params_48 [[buffer(0)]], uint device* work_48 [[buffer(4)]], packed_float4 device* frames_48 [[buffer(6)]], packed_float4 device* bodies_48 [[buffer(7)]], packed_float4 device* chunks_48 [[buffer(1)]], uint device* sort_48 [[buffer(11)]], atomic<uint> device* atom_48 [[buffer(5)]], Record_natural_0 device* manifolds_48 [[buffer(10)]], packed_float4 device* loads_48 [[buffer(12)]], packed_uint4 device* bonds_48 [[buffer(2)]], uint device* args_48 [[buffer(13)]], uint device* info_48 [[buffer(15)]], ScanParams_0 constant* scan_48 [[buffer(14)]], packed_float4 device* parts_48 [[buffer(3)]], packed_float4 device* colliders_48 [[buffer(8)]], packed_uint4 device* collider_info_48 [[buffer(9)]], packed_float4 device* view_chunks_48 [[buffer(17)]], ViewParams_0 constant* view_48 [[buffer(16)]], packed_float4 device* objects_48 [[buffer(18)]], uint device* shot_bodies_48 [[buffer(20)]], float device* lines_48 [[buffer(19)]], packed_float4 device* geometry_48 [[buffer(21)]], packed_float4 device* shots_48 [[buffer(22)]], packed_float4 device* lp_48 [[buffer(24)]], packed_float4 device* ext_50 [[buffer(23)]], packed_float4 device* lp_sides_48 [[buffer(28)]], uint device* engine_counters_48 [[buffer(26)]], packed_float4 device* engine_contacts_48 [[buffer(25)]], packed_float4 device* cs_anchors_48 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_144;
    (&kernelContext_144)->params_0 = params_48;
    (&kernelContext_144)->work_0 = work_48;
    (&kernelContext_144)->frames_0 = frames_48;
    (&kernelContext_144)->bodies_0 = bodies_48;
    (&kernelContext_144)->chunks_0 = chunks_48;
    (&kernelContext_144)->sort_0 = sort_48;
    (&kernelContext_144)->atom_0 = atom_48;
    (&kernelContext_144)->manifolds_0 = manifolds_48;
    (&kernelContext_144)->loads_0 = loads_48;
    (&kernelContext_144)->bonds_0 = bonds_48;
    (&kernelContext_144)->args_0 = args_48;
    (&kernelContext_144)->info_0 = info_48;
    (&kernelContext_144)->scan_0 = scan_48;
    (&kernelContext_144)->parts_0 = parts_48;
    (&kernelContext_144)->colliders_0 = colliders_48;
    (&kernelContext_144)->collider_info_0 = collider_info_48;
    (&kernelContext_144)->view_chunks_0 = view_chunks_48;
    (&kernelContext_144)->view_0 = view_48;
    (&kernelContext_144)->objects_0 = objects_48;
    (&kernelContext_144)->shot_bodies_0 = shot_bodies_48;
    (&kernelContext_144)->lines_0 = lines_48;
    (&kernelContext_144)->geometry_0 = geometry_48;
    (&kernelContext_144)->shots_0 = shots_48;
    (&kernelContext_144)->lp_0 = lp_48;
    (&kernelContext_144)->ext_0 = ext_50;
    (&kernelContext_144)->lp_sides_0 = lp_sides_48;
    (&kernelContext_144)->engine_counters_0 = engine_counters_48;
    (&kernelContext_144)->engine_contacts_0 = engine_contacts_48;
    (&kernelContext_144)->cs_anchors_0 = cs_anchors_48;
    threadgroup array<uint, int(1024)> g_scan_48;
    (&kernelContext_144)->g_scan_0 = &g_scan_48;
    il_extend_0(id_45.x, 1U, &kernelContext_144);
    return;
}

[[kernel]] void fr_csl_strike(uint3 id_46 [[thread_position_in_grid]], FractureParams_0 constant* params_49 [[buffer(0)]], uint device* work_49 [[buffer(4)]], packed_float4 device* frames_49 [[buffer(6)]], packed_float4 device* bodies_49 [[buffer(7)]], packed_float4 device* chunks_49 [[buffer(1)]], uint device* sort_49 [[buffer(11)]], atomic<uint> device* atom_49 [[buffer(5)]], Record_natural_0 device* manifolds_49 [[buffer(10)]], packed_float4 device* loads_49 [[buffer(12)]], packed_uint4 device* bonds_49 [[buffer(2)]], uint device* args_49 [[buffer(13)]], uint device* info_49 [[buffer(15)]], ScanParams_0 constant* scan_49 [[buffer(14)]], packed_float4 device* parts_49 [[buffer(3)]], packed_float4 device* colliders_49 [[buffer(8)]], packed_uint4 device* collider_info_49 [[buffer(9)]], packed_float4 device* view_chunks_49 [[buffer(17)]], ViewParams_0 constant* view_49 [[buffer(16)]], packed_float4 device* objects_49 [[buffer(18)]], uint device* shot_bodies_49 [[buffer(20)]], float device* lines_49 [[buffer(19)]], packed_float4 device* geometry_49 [[buffer(21)]], packed_float4 device* shots_49 [[buffer(22)]], packed_float4 device* lp_49 [[buffer(24)]], packed_float4 device* ext_51 [[buffer(23)]], packed_float4 device* lp_sides_49 [[buffer(28)]], uint device* engine_counters_49 [[buffer(26)]], packed_float4 device* engine_contacts_49 [[buffer(25)]], packed_float4 device* cs_anchors_49 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_145;
    (&kernelContext_145)->params_0 = params_49;
    (&kernelContext_145)->work_0 = work_49;
    (&kernelContext_145)->frames_0 = frames_49;
    (&kernelContext_145)->bodies_0 = bodies_49;
    (&kernelContext_145)->chunks_0 = chunks_49;
    (&kernelContext_145)->sort_0 = sort_49;
    (&kernelContext_145)->atom_0 = atom_49;
    (&kernelContext_145)->manifolds_0 = manifolds_49;
    (&kernelContext_145)->loads_0 = loads_49;
    (&kernelContext_145)->bonds_0 = bonds_49;
    (&kernelContext_145)->args_0 = args_49;
    (&kernelContext_145)->info_0 = info_49;
    (&kernelContext_145)->scan_0 = scan_49;
    (&kernelContext_145)->parts_0 = parts_49;
    (&kernelContext_145)->colliders_0 = colliders_49;
    (&kernelContext_145)->collider_info_0 = collider_info_49;
    (&kernelContext_145)->view_chunks_0 = view_chunks_49;
    (&kernelContext_145)->view_0 = view_49;
    (&kernelContext_145)->objects_0 = objects_49;
    (&kernelContext_145)->shot_bodies_0 = shot_bodies_49;
    (&kernelContext_145)->lines_0 = lines_49;
    (&kernelContext_145)->geometry_0 = geometry_49;
    (&kernelContext_145)->shots_0 = shots_49;
    (&kernelContext_145)->lp_0 = lp_49;
    (&kernelContext_145)->ext_0 = ext_51;
    (&kernelContext_145)->lp_sides_0 = lp_sides_49;
    (&kernelContext_145)->engine_counters_0 = engine_counters_49;
    (&kernelContext_145)->engine_contacts_0 = engine_contacts_49;
    (&kernelContext_145)->cs_anchors_0 = cs_anchors_49;
    threadgroup array<uint, int(1024)> g_scan_49;
    (&kernelContext_145)->g_scan_0 = &g_scan_49;
    il_strike_0(id_46.x, &kernelContext_145);
    return;
}

[[kernel]] void fr_csl_mark(uint3 id_47 [[thread_position_in_grid]], FractureParams_0 constant* params_50 [[buffer(0)]], uint device* work_50 [[buffer(4)]], packed_float4 device* frames_50 [[buffer(6)]], packed_float4 device* bodies_50 [[buffer(7)]], packed_float4 device* chunks_50 [[buffer(1)]], uint device* sort_50 [[buffer(11)]], atomic<uint> device* atom_50 [[buffer(5)]], Record_natural_0 device* manifolds_50 [[buffer(10)]], packed_float4 device* loads_50 [[buffer(12)]], packed_uint4 device* bonds_50 [[buffer(2)]], uint device* args_50 [[buffer(13)]], uint device* info_50 [[buffer(15)]], ScanParams_0 constant* scan_50 [[buffer(14)]], packed_float4 device* parts_50 [[buffer(3)]], packed_float4 device* colliders_50 [[buffer(8)]], packed_uint4 device* collider_info_50 [[buffer(9)]], packed_float4 device* view_chunks_50 [[buffer(17)]], ViewParams_0 constant* view_50 [[buffer(16)]], packed_float4 device* objects_50 [[buffer(18)]], uint device* shot_bodies_50 [[buffer(20)]], float device* lines_50 [[buffer(19)]], packed_float4 device* geometry_50 [[buffer(21)]], packed_float4 device* shots_50 [[buffer(22)]], packed_float4 device* lp_50 [[buffer(24)]], packed_float4 device* ext_52 [[buffer(23)]], packed_float4 device* lp_sides_50 [[buffer(28)]], uint device* engine_counters_50 [[buffer(26)]], packed_float4 device* engine_contacts_50 [[buffer(25)]], packed_float4 device* cs_anchors_50 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_146;
    (&kernelContext_146)->params_0 = params_50;
    (&kernelContext_146)->work_0 = work_50;
    (&kernelContext_146)->frames_0 = frames_50;
    (&kernelContext_146)->bodies_0 = bodies_50;
    (&kernelContext_146)->chunks_0 = chunks_50;
    (&kernelContext_146)->sort_0 = sort_50;
    (&kernelContext_146)->atom_0 = atom_50;
    (&kernelContext_146)->manifolds_0 = manifolds_50;
    (&kernelContext_146)->loads_0 = loads_50;
    (&kernelContext_146)->bonds_0 = bonds_50;
    (&kernelContext_146)->args_0 = args_50;
    (&kernelContext_146)->info_0 = info_50;
    (&kernelContext_146)->scan_0 = scan_50;
    (&kernelContext_146)->parts_0 = parts_50;
    (&kernelContext_146)->colliders_0 = colliders_50;
    (&kernelContext_146)->collider_info_0 = collider_info_50;
    (&kernelContext_146)->view_chunks_0 = view_chunks_50;
    (&kernelContext_146)->view_0 = view_50;
    (&kernelContext_146)->objects_0 = objects_50;
    (&kernelContext_146)->shot_bodies_0 = shot_bodies_50;
    (&kernelContext_146)->lines_0 = lines_50;
    (&kernelContext_146)->geometry_0 = geometry_50;
    (&kernelContext_146)->shots_0 = shots_50;
    (&kernelContext_146)->lp_0 = lp_50;
    (&kernelContext_146)->ext_0 = ext_52;
    (&kernelContext_146)->lp_sides_0 = lp_sides_50;
    (&kernelContext_146)->engine_counters_0 = engine_counters_50;
    (&kernelContext_146)->engine_contacts_0 = engine_contacts_50;
    (&kernelContext_146)->cs_anchors_0 = cs_anchors_50;
    threadgroup array<uint, int(1024)> g_scan_50;
    (&kernelContext_146)->g_scan_0 = &g_scan_50;
    il_mark_0(id_47.x, &kernelContext_146);
    return;
}

[[kernel]] void fr_csl_carry(uint3 id_48 [[thread_position_in_grid]], FractureParams_0 constant* params_51 [[buffer(0)]], uint device* work_51 [[buffer(4)]], packed_float4 device* frames_51 [[buffer(6)]], packed_float4 device* bodies_51 [[buffer(7)]], packed_float4 device* chunks_51 [[buffer(1)]], uint device* sort_51 [[buffer(11)]], atomic<uint> device* atom_51 [[buffer(5)]], Record_natural_0 device* manifolds_51 [[buffer(10)]], packed_float4 device* loads_51 [[buffer(12)]], packed_uint4 device* bonds_51 [[buffer(2)]], uint device* args_51 [[buffer(13)]], uint device* info_51 [[buffer(15)]], ScanParams_0 constant* scan_51 [[buffer(14)]], packed_float4 device* parts_51 [[buffer(3)]], packed_float4 device* colliders_51 [[buffer(8)]], packed_uint4 device* collider_info_51 [[buffer(9)]], packed_float4 device* view_chunks_51 [[buffer(17)]], ViewParams_0 constant* view_51 [[buffer(16)]], packed_float4 device* objects_51 [[buffer(18)]], uint device* shot_bodies_51 [[buffer(20)]], float device* lines_51 [[buffer(19)]], packed_float4 device* geometry_51 [[buffer(21)]], packed_float4 device* shots_51 [[buffer(22)]], packed_float4 device* lp_51 [[buffer(24)]], packed_float4 device* ext_53 [[buffer(23)]], packed_float4 device* lp_sides_51 [[buffer(28)]], uint device* engine_counters_51 [[buffer(26)]], packed_float4 device* engine_contacts_51 [[buffer(25)]], packed_float4 device* cs_anchors_51 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_147;
    (&kernelContext_147)->params_0 = params_51;
    (&kernelContext_147)->work_0 = work_51;
    (&kernelContext_147)->frames_0 = frames_51;
    (&kernelContext_147)->bodies_0 = bodies_51;
    (&kernelContext_147)->chunks_0 = chunks_51;
    (&kernelContext_147)->sort_0 = sort_51;
    (&kernelContext_147)->atom_0 = atom_51;
    (&kernelContext_147)->manifolds_0 = manifolds_51;
    (&kernelContext_147)->loads_0 = loads_51;
    (&kernelContext_147)->bonds_0 = bonds_51;
    (&kernelContext_147)->args_0 = args_51;
    (&kernelContext_147)->info_0 = info_51;
    (&kernelContext_147)->scan_0 = scan_51;
    (&kernelContext_147)->parts_0 = parts_51;
    (&kernelContext_147)->colliders_0 = colliders_51;
    (&kernelContext_147)->collider_info_0 = collider_info_51;
    (&kernelContext_147)->view_chunks_0 = view_chunks_51;
    (&kernelContext_147)->view_0 = view_51;
    (&kernelContext_147)->objects_0 = objects_51;
    (&kernelContext_147)->shot_bodies_0 = shot_bodies_51;
    (&kernelContext_147)->lines_0 = lines_51;
    (&kernelContext_147)->geometry_0 = geometry_51;
    (&kernelContext_147)->shots_0 = shots_51;
    (&kernelContext_147)->lp_0 = lp_51;
    (&kernelContext_147)->ext_0 = ext_53;
    (&kernelContext_147)->lp_sides_0 = lp_sides_51;
    (&kernelContext_147)->engine_counters_0 = engine_counters_51;
    (&kernelContext_147)->engine_contacts_0 = engine_contacts_51;
    (&kernelContext_147)->cs_anchors_0 = cs_anchors_51;
    threadgroup array<uint, int(1024)> g_scan_51;
    (&kernelContext_147)->g_scan_0 = &g_scan_51;
    il_carry_0(id_48.x, &kernelContext_147);
    return;
}

[[kernel]] void fr_csl_breaks(uint3 id_49 [[thread_position_in_grid]], FractureParams_0 constant* params_52 [[buffer(0)]], uint device* work_52 [[buffer(4)]], packed_float4 device* frames_52 [[buffer(6)]], packed_float4 device* bodies_52 [[buffer(7)]], packed_float4 device* chunks_52 [[buffer(1)]], uint device* sort_52 [[buffer(11)]], atomic<uint> device* atom_52 [[buffer(5)]], Record_natural_0 device* manifolds_52 [[buffer(10)]], packed_float4 device* loads_52 [[buffer(12)]], packed_uint4 device* bonds_52 [[buffer(2)]], uint device* args_52 [[buffer(13)]], uint device* info_52 [[buffer(15)]], ScanParams_0 constant* scan_52 [[buffer(14)]], packed_float4 device* parts_52 [[buffer(3)]], packed_float4 device* colliders_52 [[buffer(8)]], packed_uint4 device* collider_info_52 [[buffer(9)]], packed_float4 device* view_chunks_52 [[buffer(17)]], ViewParams_0 constant* view_52 [[buffer(16)]], packed_float4 device* objects_52 [[buffer(18)]], uint device* shot_bodies_52 [[buffer(20)]], float device* lines_52 [[buffer(19)]], packed_float4 device* geometry_52 [[buffer(21)]], packed_float4 device* shots_52 [[buffer(22)]], packed_float4 device* lp_52 [[buffer(24)]], packed_float4 device* ext_54 [[buffer(23)]], packed_float4 device* lp_sides_52 [[buffer(28)]], uint device* engine_counters_52 [[buffer(26)]], packed_float4 device* engine_contacts_52 [[buffer(25)]], packed_float4 device* cs_anchors_52 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_148;
    (&kernelContext_148)->params_0 = params_52;
    (&kernelContext_148)->work_0 = work_52;
    (&kernelContext_148)->frames_0 = frames_52;
    (&kernelContext_148)->bodies_0 = bodies_52;
    (&kernelContext_148)->chunks_0 = chunks_52;
    (&kernelContext_148)->sort_0 = sort_52;
    (&kernelContext_148)->atom_0 = atom_52;
    (&kernelContext_148)->manifolds_0 = manifolds_52;
    (&kernelContext_148)->loads_0 = loads_52;
    (&kernelContext_148)->bonds_0 = bonds_52;
    (&kernelContext_148)->args_0 = args_52;
    (&kernelContext_148)->info_0 = info_52;
    (&kernelContext_148)->scan_0 = scan_52;
    (&kernelContext_148)->parts_0 = parts_52;
    (&kernelContext_148)->colliders_0 = colliders_52;
    (&kernelContext_148)->collider_info_0 = collider_info_52;
    (&kernelContext_148)->view_chunks_0 = view_chunks_52;
    (&kernelContext_148)->view_0 = view_52;
    (&kernelContext_148)->objects_0 = objects_52;
    (&kernelContext_148)->shot_bodies_0 = shot_bodies_52;
    (&kernelContext_148)->lines_0 = lines_52;
    (&kernelContext_148)->geometry_0 = geometry_52;
    (&kernelContext_148)->shots_0 = shots_52;
    (&kernelContext_148)->lp_0 = lp_52;
    (&kernelContext_148)->ext_0 = ext_54;
    (&kernelContext_148)->lp_sides_0 = lp_sides_52;
    (&kernelContext_148)->engine_counters_0 = engine_counters_52;
    (&kernelContext_148)->engine_contacts_0 = engine_contacts_52;
    (&kernelContext_148)->cs_anchors_0 = cs_anchors_52;
    threadgroup array<uint, int(1024)> g_scan_52;
    (&kernelContext_148)->g_scan_0 = &g_scan_52;
    il_breaks_0(id_49.x, &kernelContext_148);
    return;
}

[[kernel]] void fr_cs_begin(uint3 id_50 [[thread_position_in_grid]], FractureParams_0 constant* params_53 [[buffer(0)]], uint device* work_53 [[buffer(4)]], packed_float4 device* frames_53 [[buffer(6)]], packed_float4 device* bodies_53 [[buffer(7)]], packed_float4 device* chunks_53 [[buffer(1)]], uint device* sort_53 [[buffer(11)]], atomic<uint> device* atom_53 [[buffer(5)]], Record_natural_0 device* manifolds_53 [[buffer(10)]], packed_float4 device* loads_53 [[buffer(12)]], packed_uint4 device* bonds_53 [[buffer(2)]], uint device* args_53 [[buffer(13)]], uint device* info_53 [[buffer(15)]], ScanParams_0 constant* scan_53 [[buffer(14)]], packed_float4 device* parts_53 [[buffer(3)]], packed_float4 device* colliders_53 [[buffer(8)]], packed_uint4 device* collider_info_53 [[buffer(9)]], packed_float4 device* view_chunks_53 [[buffer(17)]], ViewParams_0 constant* view_53 [[buffer(16)]], packed_float4 device* objects_53 [[buffer(18)]], uint device* shot_bodies_53 [[buffer(20)]], float device* lines_53 [[buffer(19)]], packed_float4 device* geometry_53 [[buffer(21)]], packed_float4 device* shots_53 [[buffer(22)]], packed_float4 device* lp_53 [[buffer(24)]], packed_float4 device* ext_55 [[buffer(23)]], packed_float4 device* lp_sides_53 [[buffer(28)]], uint device* engine_counters_53 [[buffer(26)]], packed_float4 device* engine_contacts_53 [[buffer(25)]], packed_float4 device* cs_anchors_53 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_149;
    (&kernelContext_149)->params_0 = params_53;
    (&kernelContext_149)->work_0 = work_53;
    (&kernelContext_149)->frames_0 = frames_53;
    (&kernelContext_149)->bodies_0 = bodies_53;
    (&kernelContext_149)->chunks_0 = chunks_53;
    (&kernelContext_149)->sort_0 = sort_53;
    (&kernelContext_149)->atom_0 = atom_53;
    (&kernelContext_149)->manifolds_0 = manifolds_53;
    (&kernelContext_149)->loads_0 = loads_53;
    (&kernelContext_149)->bonds_0 = bonds_53;
    (&kernelContext_149)->args_0 = args_53;
    (&kernelContext_149)->info_0 = info_53;
    (&kernelContext_149)->scan_0 = scan_53;
    (&kernelContext_149)->parts_0 = parts_53;
    (&kernelContext_149)->colliders_0 = colliders_53;
    (&kernelContext_149)->collider_info_0 = collider_info_53;
    (&kernelContext_149)->view_chunks_0 = view_chunks_53;
    (&kernelContext_149)->view_0 = view_53;
    (&kernelContext_149)->objects_0 = objects_53;
    (&kernelContext_149)->shot_bodies_0 = shot_bodies_53;
    (&kernelContext_149)->lines_0 = lines_53;
    (&kernelContext_149)->geometry_0 = geometry_53;
    (&kernelContext_149)->shots_0 = shots_53;
    (&kernelContext_149)->lp_0 = lp_53;
    (&kernelContext_149)->ext_0 = ext_55;
    (&kernelContext_149)->lp_sides_0 = lp_sides_53;
    (&kernelContext_149)->engine_counters_0 = engine_counters_53;
    (&kernelContext_149)->engine_contacts_0 = engine_contacts_53;
    (&kernelContext_149)->cs_anchors_0 = cs_anchors_53;
    threadgroup array<uint, int(1024)> g_scan_53;
    (&kernelContext_149)->g_scan_0 = &g_scan_53;
    uint _S1107 = a_counters_1(&kernelContext_149);
    atomic_store_explicit((&kernelContext_149)->atom_0+(_S1107 + 14U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_149)->atom_0+(_S1107 + 15U), 0U, memory_order_relaxed);
    uint _S1108 = cs_rows_0(&kernelContext_149);
    uint _S1109 = groups_of_0(_S1108, 64U);
    set_args_0(7U, _S1109, &kernelContext_149);
    return;
}

uint w_cs_list_0(KernelContext_0 thread* kernelContext_150)
{
    uint _S1110 = w_carry_0(kernelContext_150);
    return _S1110 + kernelContext_150->params_0->chunk_count_0;
}

uint w_cs_bodies_0(KernelContext_0 thread* kernelContext_151)
{
    uint _S1111 = w_cs_list_0(kernelContext_151);
    return _S1111 + kernelContext_151->params_0->chunk_count_0;
}

[[kernel]] void fr_cs_contacts(uint3 id_51 [[thread_position_in_grid]], FractureParams_0 constant* params_54 [[buffer(0)]], uint device* work_54 [[buffer(4)]], packed_float4 device* frames_54 [[buffer(6)]], packed_float4 device* bodies_54 [[buffer(7)]], packed_float4 device* chunks_54 [[buffer(1)]], uint device* sort_54 [[buffer(11)]], atomic<uint> device* atom_54 [[buffer(5)]], Record_natural_0 device* manifolds_54 [[buffer(10)]], packed_float4 device* loads_54 [[buffer(12)]], packed_uint4 device* bonds_54 [[buffer(2)]], uint device* args_54 [[buffer(13)]], uint device* info_54 [[buffer(15)]], ScanParams_0 constant* scan_54 [[buffer(14)]], packed_float4 device* parts_54 [[buffer(3)]], packed_float4 device* colliders_54 [[buffer(8)]], packed_uint4 device* collider_info_54 [[buffer(9)]], packed_float4 device* view_chunks_54 [[buffer(17)]], ViewParams_0 constant* view_54 [[buffer(16)]], packed_float4 device* objects_54 [[buffer(18)]], uint device* shot_bodies_54 [[buffer(20)]], float device* lines_54 [[buffer(19)]], packed_float4 device* geometry_54 [[buffer(21)]], packed_float4 device* shots_54 [[buffer(22)]], packed_float4 device* lp_54 [[buffer(24)]], packed_float4 device* ext_56 [[buffer(23)]], packed_float4 device* lp_sides_54 [[buffer(28)]], uint device* engine_counters_54 [[buffer(26)]], packed_float4 device* engine_contacts_54 [[buffer(25)]], packed_float4 device* cs_anchors_54 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_152;
    (&kernelContext_152)->params_0 = params_54;
    (&kernelContext_152)->work_0 = work_54;
    (&kernelContext_152)->frames_0 = frames_54;
    (&kernelContext_152)->bodies_0 = bodies_54;
    (&kernelContext_152)->chunks_0 = chunks_54;
    (&kernelContext_152)->sort_0 = sort_54;
    (&kernelContext_152)->atom_0 = atom_54;
    (&kernelContext_152)->manifolds_0 = manifolds_54;
    (&kernelContext_152)->loads_0 = loads_54;
    (&kernelContext_152)->bonds_0 = bonds_54;
    (&kernelContext_152)->args_0 = args_54;
    (&kernelContext_152)->info_0 = info_54;
    (&kernelContext_152)->scan_0 = scan_54;
    (&kernelContext_152)->parts_0 = parts_54;
    (&kernelContext_152)->colliders_0 = colliders_54;
    (&kernelContext_152)->collider_info_0 = collider_info_54;
    (&kernelContext_152)->view_chunks_0 = view_chunks_54;
    (&kernelContext_152)->view_0 = view_54;
    (&kernelContext_152)->objects_0 = objects_54;
    (&kernelContext_152)->shot_bodies_0 = shot_bodies_54;
    (&kernelContext_152)->lines_0 = lines_54;
    (&kernelContext_152)->geometry_0 = geometry_54;
    (&kernelContext_152)->shots_0 = shots_54;
    (&kernelContext_152)->lp_0 = lp_54;
    (&kernelContext_152)->ext_0 = ext_56;
    (&kernelContext_152)->lp_sides_0 = lp_sides_54;
    (&kernelContext_152)->engine_counters_0 = engine_counters_54;
    (&kernelContext_152)->engine_contacts_0 = engine_contacts_54;
    (&kernelContext_152)->cs_anchors_0 = cs_anchors_54;
    threadgroup array<uint, int(1024)> g_scan_54;
    (&kernelContext_152)->g_scan_0 = &g_scan_54;
    uint r_25 = id_51.x;
    uint _S1112 = cs_rows_0(&kernelContext_152);
    if(r_25 >= _S1112)
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
        thread uint k_40;
        thread uint s_17;
        thread uint ob_5;
        thread uint co_5;
        thread float3 p_18;
        thread float3 u_9;
        thread float v_14;
        thread float m_eff_7;
        thread float t_in_8;
        bool _S1113 = cs_side_0(r_25, side_6, &k_40, &s_17, &ob_5, &co_5, &p_18, &u_9, &v_14, &m_eff_7, &t_in_8, &kernelContext_152);
        if(_S1113)
        {
            uint _S1114 = a_cs_0(s_17, &kernelContext_152);
            uint _S1115 = atomic_fetch_max_explicit((&kernelContext_152)->atom_0+_S1114, (as_type<uint>((m_eff_7 * v_14))), memory_order_relaxed);
            uint _S1116 = a_cs_0(s_17, &kernelContext_152);
            uint _S1117 = atomic_exchange_explicit((&kernelContext_152)->atom_0+(_S1116 + 8U), 1U, memory_order_relaxed);
            if(_S1117 == 0U)
            {
                uint _S1118 = w_cs_bodies_0(&kernelContext_152);
                uint _S1119 = a_counters_1(&kernelContext_152);
                uint _S1120 = atomic_fetch_add_explicit((&kernelContext_152)->atom_0+(_S1119 + 14U), 1U, memory_order_relaxed);
                *((&kernelContext_152)->work_0+(_S1118 + _S1120)) = s_17;
            }
        }
        side_6 = side_6 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_gate(uint3 id_52 [[thread_position_in_grid]], FractureParams_0 constant* params_55 [[buffer(0)]], uint device* work_55 [[buffer(4)]], packed_float4 device* frames_55 [[buffer(6)]], packed_float4 device* bodies_55 [[buffer(7)]], packed_float4 device* chunks_55 [[buffer(1)]], uint device* sort_55 [[buffer(11)]], atomic<uint> device* atom_55 [[buffer(5)]], Record_natural_0 device* manifolds_55 [[buffer(10)]], packed_float4 device* loads_55 [[buffer(12)]], packed_uint4 device* bonds_55 [[buffer(2)]], uint device* args_55 [[buffer(13)]], uint device* info_55 [[buffer(15)]], ScanParams_0 constant* scan_55 [[buffer(14)]], packed_float4 device* parts_55 [[buffer(3)]], packed_float4 device* colliders_55 [[buffer(8)]], packed_uint4 device* collider_info_55 [[buffer(9)]], packed_float4 device* view_chunks_55 [[buffer(17)]], ViewParams_0 constant* view_55 [[buffer(16)]], packed_float4 device* objects_55 [[buffer(18)]], uint device* shot_bodies_55 [[buffer(20)]], float device* lines_55 [[buffer(19)]], packed_float4 device* geometry_55 [[buffer(21)]], packed_float4 device* shots_55 [[buffer(22)]], packed_float4 device* lp_55 [[buffer(24)]], packed_float4 device* ext_57 [[buffer(23)]], packed_float4 device* lp_sides_55 [[buffer(28)]], uint device* engine_counters_55 [[buffer(26)]], packed_float4 device* engine_contacts_55 [[buffer(25)]], packed_float4 device* cs_anchors_55 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_153;
    (&kernelContext_153)->params_0 = params_55;
    (&kernelContext_153)->work_0 = work_55;
    (&kernelContext_153)->frames_0 = frames_55;
    (&kernelContext_153)->bodies_0 = bodies_55;
    (&kernelContext_153)->chunks_0 = chunks_55;
    (&kernelContext_153)->sort_0 = sort_55;
    (&kernelContext_153)->atom_0 = atom_55;
    (&kernelContext_153)->manifolds_0 = manifolds_55;
    (&kernelContext_153)->loads_0 = loads_55;
    (&kernelContext_153)->bonds_0 = bonds_55;
    (&kernelContext_153)->args_0 = args_55;
    (&kernelContext_153)->info_0 = info_55;
    (&kernelContext_153)->scan_0 = scan_55;
    (&kernelContext_153)->parts_0 = parts_55;
    (&kernelContext_153)->colliders_0 = colliders_55;
    (&kernelContext_153)->collider_info_0 = collider_info_55;
    (&kernelContext_153)->view_chunks_0 = view_chunks_55;
    (&kernelContext_153)->view_0 = view_55;
    (&kernelContext_153)->objects_0 = objects_55;
    (&kernelContext_153)->shot_bodies_0 = shot_bodies_55;
    (&kernelContext_153)->lines_0 = lines_55;
    (&kernelContext_153)->geometry_0 = geometry_55;
    (&kernelContext_153)->shots_0 = shots_55;
    (&kernelContext_153)->lp_0 = lp_55;
    (&kernelContext_153)->ext_0 = ext_57;
    (&kernelContext_153)->lp_sides_0 = lp_sides_55;
    (&kernelContext_153)->engine_counters_0 = engine_counters_55;
    (&kernelContext_153)->engine_contacts_0 = engine_contacts_55;
    (&kernelContext_153)->cs_anchors_0 = cs_anchors_55;
    threadgroup array<uint, int(1024)> g_scan_55;
    (&kernelContext_153)->g_scan_0 = &g_scan_55;
    uint _S1121 = a_counters_1(&kernelContext_153);
    uint struck_1 = atomic_load_explicit((&kernelContext_153)->atom_0+(_S1121 + 14U), memory_order_relaxed);
    bool any_0 = struck_1 > 0U;
    uint _S1122;
    if(any_0)
    {
        uint _S1123 = cs_rows_0(&kernelContext_153);
        uint _S1124 = groups_of_0(_S1123, 64U);
        _S1122 = _S1124;
    }
    else
    {
        _S1122 = 0U;
    }
    set_args_0(8U, _S1122, &kernelContext_153);
    uint _S1125 = groups_of_0(struck_1, 64U);
    set_args_0(9U, _S1125, &kernelContext_153);
    set_args_0(6U, struck_1, &kernelContext_153);
    if(any_0)
    {
        uint _S1126 = groups_of_0((&kernelContext_153)->params_0->chunk_count_0, 64U);
        _S1122 = _S1126;
    }
    else
    {
        _S1122 = 0U;
    }
    set_args_0(10U, _S1122, &kernelContext_153);
    if(any_0)
    {
        uint _S1127 = groups_of_0((&kernelContext_153)->params_0->bond_count_0, 64U);
        _S1122 = _S1127;
    }
    else
    {
        _S1122 = 0U;
    }
    set_args_0(11U, _S1122, &kernelContext_153);
    if(any_0)
    {
        uint _S1128 = groups_of_0((&kernelContext_153)->params_0->body_count_0, 64U);
        _S1122 = _S1128;
    }
    else
    {
        _S1122 = 0U;
    }
    set_args_0(12U, _S1122, &kernelContext_153);
    return;
}

[[kernel]] void fr_cs_pick_a(uint3 id_53 [[thread_position_in_grid]], FractureParams_0 constant* params_56 [[buffer(0)]], uint device* work_56 [[buffer(4)]], packed_float4 device* frames_56 [[buffer(6)]], packed_float4 device* bodies_56 [[buffer(7)]], packed_float4 device* chunks_56 [[buffer(1)]], uint device* sort_56 [[buffer(11)]], atomic<uint> device* atom_56 [[buffer(5)]], Record_natural_0 device* manifolds_56 [[buffer(10)]], packed_float4 device* loads_56 [[buffer(12)]], packed_uint4 device* bonds_56 [[buffer(2)]], uint device* args_56 [[buffer(13)]], uint device* info_56 [[buffer(15)]], ScanParams_0 constant* scan_56 [[buffer(14)]], packed_float4 device* parts_56 [[buffer(3)]], packed_float4 device* colliders_56 [[buffer(8)]], packed_uint4 device* collider_info_56 [[buffer(9)]], packed_float4 device* view_chunks_56 [[buffer(17)]], ViewParams_0 constant* view_56 [[buffer(16)]], packed_float4 device* objects_56 [[buffer(18)]], uint device* shot_bodies_56 [[buffer(20)]], float device* lines_56 [[buffer(19)]], packed_float4 device* geometry_56 [[buffer(21)]], packed_float4 device* shots_56 [[buffer(22)]], packed_float4 device* lp_56 [[buffer(24)]], packed_float4 device* ext_58 [[buffer(23)]], packed_float4 device* lp_sides_56 [[buffer(28)]], uint device* engine_counters_56 [[buffer(26)]], packed_float4 device* engine_contacts_56 [[buffer(25)]], packed_float4 device* cs_anchors_56 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_154;
    (&kernelContext_154)->params_0 = params_56;
    (&kernelContext_154)->work_0 = work_56;
    (&kernelContext_154)->frames_0 = frames_56;
    (&kernelContext_154)->bodies_0 = bodies_56;
    (&kernelContext_154)->chunks_0 = chunks_56;
    (&kernelContext_154)->sort_0 = sort_56;
    (&kernelContext_154)->atom_0 = atom_56;
    (&kernelContext_154)->manifolds_0 = manifolds_56;
    (&kernelContext_154)->loads_0 = loads_56;
    (&kernelContext_154)->bonds_0 = bonds_56;
    (&kernelContext_154)->args_0 = args_56;
    (&kernelContext_154)->info_0 = info_56;
    (&kernelContext_154)->scan_0 = scan_56;
    (&kernelContext_154)->parts_0 = parts_56;
    (&kernelContext_154)->colliders_0 = colliders_56;
    (&kernelContext_154)->collider_info_0 = collider_info_56;
    (&kernelContext_154)->view_chunks_0 = view_chunks_56;
    (&kernelContext_154)->view_0 = view_56;
    (&kernelContext_154)->objects_0 = objects_56;
    (&kernelContext_154)->shot_bodies_0 = shot_bodies_56;
    (&kernelContext_154)->lines_0 = lines_56;
    (&kernelContext_154)->geometry_0 = geometry_56;
    (&kernelContext_154)->shots_0 = shots_56;
    (&kernelContext_154)->lp_0 = lp_56;
    (&kernelContext_154)->ext_0 = ext_58;
    (&kernelContext_154)->lp_sides_0 = lp_sides_56;
    (&kernelContext_154)->engine_counters_0 = engine_counters_56;
    (&kernelContext_154)->engine_contacts_0 = engine_contacts_56;
    (&kernelContext_154)->cs_anchors_0 = cs_anchors_56;
    threadgroup array<uint, int(1024)> g_scan_56;
    (&kernelContext_154)->g_scan_0 = &g_scan_56;
    cs_pick_3(id_53.x, &kernelContext_154);
    return;
}

[[kernel]] void fr_cs_pick_b(uint3 id_54 [[thread_position_in_grid]], FractureParams_0 constant* params_57 [[buffer(0)]], uint device* work_57 [[buffer(4)]], packed_float4 device* frames_57 [[buffer(6)]], packed_float4 device* bodies_57 [[buffer(7)]], packed_float4 device* chunks_57 [[buffer(1)]], uint device* sort_57 [[buffer(11)]], atomic<uint> device* atom_57 [[buffer(5)]], Record_natural_0 device* manifolds_57 [[buffer(10)]], packed_float4 device* loads_57 [[buffer(12)]], packed_uint4 device* bonds_57 [[buffer(2)]], uint device* args_57 [[buffer(13)]], uint device* info_57 [[buffer(15)]], ScanParams_0 constant* scan_57 [[buffer(14)]], packed_float4 device* parts_57 [[buffer(3)]], packed_float4 device* colliders_57 [[buffer(8)]], packed_uint4 device* collider_info_57 [[buffer(9)]], packed_float4 device* view_chunks_57 [[buffer(17)]], ViewParams_0 constant* view_57 [[buffer(16)]], packed_float4 device* objects_57 [[buffer(18)]], uint device* shot_bodies_57 [[buffer(20)]], float device* lines_57 [[buffer(19)]], packed_float4 device* geometry_57 [[buffer(21)]], packed_float4 device* shots_57 [[buffer(22)]], packed_float4 device* lp_57 [[buffer(24)]], packed_float4 device* ext_59 [[buffer(23)]], packed_float4 device* lp_sides_57 [[buffer(28)]], uint device* engine_counters_57 [[buffer(26)]], packed_float4 device* engine_contacts_57 [[buffer(25)]], packed_float4 device* cs_anchors_57 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_155;
    (&kernelContext_155)->params_0 = params_57;
    (&kernelContext_155)->work_0 = work_57;
    (&kernelContext_155)->frames_0 = frames_57;
    (&kernelContext_155)->bodies_0 = bodies_57;
    (&kernelContext_155)->chunks_0 = chunks_57;
    (&kernelContext_155)->sort_0 = sort_57;
    (&kernelContext_155)->atom_0 = atom_57;
    (&kernelContext_155)->manifolds_0 = manifolds_57;
    (&kernelContext_155)->loads_0 = loads_57;
    (&kernelContext_155)->bonds_0 = bonds_57;
    (&kernelContext_155)->args_0 = args_57;
    (&kernelContext_155)->info_0 = info_57;
    (&kernelContext_155)->scan_0 = scan_57;
    (&kernelContext_155)->parts_0 = parts_57;
    (&kernelContext_155)->colliders_0 = colliders_57;
    (&kernelContext_155)->collider_info_0 = collider_info_57;
    (&kernelContext_155)->view_chunks_0 = view_chunks_57;
    (&kernelContext_155)->view_0 = view_57;
    (&kernelContext_155)->objects_0 = objects_57;
    (&kernelContext_155)->shot_bodies_0 = shot_bodies_57;
    (&kernelContext_155)->lines_0 = lines_57;
    (&kernelContext_155)->geometry_0 = geometry_57;
    (&kernelContext_155)->shots_0 = shots_57;
    (&kernelContext_155)->lp_0 = lp_57;
    (&kernelContext_155)->ext_0 = ext_59;
    (&kernelContext_155)->lp_sides_0 = lp_sides_57;
    (&kernelContext_155)->engine_counters_0 = engine_counters_57;
    (&kernelContext_155)->engine_contacts_0 = engine_contacts_57;
    (&kernelContext_155)->cs_anchors_0 = cs_anchors_57;
    threadgroup array<uint, int(1024)> g_scan_57;
    (&kernelContext_155)->g_scan_0 = &g_scan_57;
    cs_pick_2(id_54.x, &kernelContext_155);
    return;
}

[[kernel]] void fr_cs_pick_c(uint3 id_55 [[thread_position_in_grid]], FractureParams_0 constant* params_58 [[buffer(0)]], uint device* work_58 [[buffer(4)]], packed_float4 device* frames_58 [[buffer(6)]], packed_float4 device* bodies_58 [[buffer(7)]], packed_float4 device* chunks_58 [[buffer(1)]], uint device* sort_58 [[buffer(11)]], atomic<uint> device* atom_58 [[buffer(5)]], Record_natural_0 device* manifolds_58 [[buffer(10)]], packed_float4 device* loads_58 [[buffer(12)]], packed_uint4 device* bonds_58 [[buffer(2)]], uint device* args_58 [[buffer(13)]], uint device* info_58 [[buffer(15)]], ScanParams_0 constant* scan_58 [[buffer(14)]], packed_float4 device* parts_58 [[buffer(3)]], packed_float4 device* colliders_58 [[buffer(8)]], packed_uint4 device* collider_info_58 [[buffer(9)]], packed_float4 device* view_chunks_58 [[buffer(17)]], ViewParams_0 constant* view_58 [[buffer(16)]], packed_float4 device* objects_58 [[buffer(18)]], uint device* shot_bodies_58 [[buffer(20)]], float device* lines_58 [[buffer(19)]], packed_float4 device* geometry_58 [[buffer(21)]], packed_float4 device* shots_58 [[buffer(22)]], packed_float4 device* lp_58 [[buffer(24)]], packed_float4 device* ext_60 [[buffer(23)]], packed_float4 device* lp_sides_58 [[buffer(28)]], uint device* engine_counters_58 [[buffer(26)]], packed_float4 device* engine_contacts_58 [[buffer(25)]], packed_float4 device* cs_anchors_58 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_156;
    (&kernelContext_156)->params_0 = params_58;
    (&kernelContext_156)->work_0 = work_58;
    (&kernelContext_156)->frames_0 = frames_58;
    (&kernelContext_156)->bodies_0 = bodies_58;
    (&kernelContext_156)->chunks_0 = chunks_58;
    (&kernelContext_156)->sort_0 = sort_58;
    (&kernelContext_156)->atom_0 = atom_58;
    (&kernelContext_156)->manifolds_0 = manifolds_58;
    (&kernelContext_156)->loads_0 = loads_58;
    (&kernelContext_156)->bonds_0 = bonds_58;
    (&kernelContext_156)->args_0 = args_58;
    (&kernelContext_156)->info_0 = info_58;
    (&kernelContext_156)->scan_0 = scan_58;
    (&kernelContext_156)->parts_0 = parts_58;
    (&kernelContext_156)->colliders_0 = colliders_58;
    (&kernelContext_156)->collider_info_0 = collider_info_58;
    (&kernelContext_156)->view_chunks_0 = view_chunks_58;
    (&kernelContext_156)->view_0 = view_58;
    (&kernelContext_156)->objects_0 = objects_58;
    (&kernelContext_156)->shot_bodies_0 = shot_bodies_58;
    (&kernelContext_156)->lines_0 = lines_58;
    (&kernelContext_156)->geometry_0 = geometry_58;
    (&kernelContext_156)->shots_0 = shots_58;
    (&kernelContext_156)->lp_0 = lp_58;
    (&kernelContext_156)->ext_0 = ext_60;
    (&kernelContext_156)->lp_sides_0 = lp_sides_58;
    (&kernelContext_156)->engine_counters_0 = engine_counters_58;
    (&kernelContext_156)->engine_contacts_0 = engine_contacts_58;
    (&kernelContext_156)->cs_anchors_0 = cs_anchors_58;
    threadgroup array<uint, int(1024)> g_scan_58;
    (&kernelContext_156)->g_scan_0 = &g_scan_58;
    cs_pick_1(id_55.x, &kernelContext_156);
    return;
}

[[kernel]] void fr_cs_pick_d(uint3 id_56 [[thread_position_in_grid]], FractureParams_0 constant* params_59 [[buffer(0)]], uint device* work_59 [[buffer(4)]], packed_float4 device* frames_59 [[buffer(6)]], packed_float4 device* bodies_59 [[buffer(7)]], packed_float4 device* chunks_59 [[buffer(1)]], uint device* sort_59 [[buffer(11)]], atomic<uint> device* atom_59 [[buffer(5)]], Record_natural_0 device* manifolds_59 [[buffer(10)]], packed_float4 device* loads_59 [[buffer(12)]], packed_uint4 device* bonds_59 [[buffer(2)]], uint device* args_59 [[buffer(13)]], uint device* info_59 [[buffer(15)]], ScanParams_0 constant* scan_59 [[buffer(14)]], packed_float4 device* parts_59 [[buffer(3)]], packed_float4 device* colliders_59 [[buffer(8)]], packed_uint4 device* collider_info_59 [[buffer(9)]], packed_float4 device* view_chunks_59 [[buffer(17)]], ViewParams_0 constant* view_59 [[buffer(16)]], packed_float4 device* objects_59 [[buffer(18)]], uint device* shot_bodies_59 [[buffer(20)]], float device* lines_59 [[buffer(19)]], packed_float4 device* geometry_59 [[buffer(21)]], packed_float4 device* shots_59 [[buffer(22)]], packed_float4 device* lp_59 [[buffer(24)]], packed_float4 device* ext_61 [[buffer(23)]], packed_float4 device* lp_sides_59 [[buffer(28)]], uint device* engine_counters_59 [[buffer(26)]], packed_float4 device* engine_contacts_59 [[buffer(25)]], packed_float4 device* cs_anchors_59 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_157;
    (&kernelContext_157)->params_0 = params_59;
    (&kernelContext_157)->work_0 = work_59;
    (&kernelContext_157)->frames_0 = frames_59;
    (&kernelContext_157)->bodies_0 = bodies_59;
    (&kernelContext_157)->chunks_0 = chunks_59;
    (&kernelContext_157)->sort_0 = sort_59;
    (&kernelContext_157)->atom_0 = atom_59;
    (&kernelContext_157)->manifolds_0 = manifolds_59;
    (&kernelContext_157)->loads_0 = loads_59;
    (&kernelContext_157)->bonds_0 = bonds_59;
    (&kernelContext_157)->args_0 = args_59;
    (&kernelContext_157)->info_0 = info_59;
    (&kernelContext_157)->scan_0 = scan_59;
    (&kernelContext_157)->parts_0 = parts_59;
    (&kernelContext_157)->colliders_0 = colliders_59;
    (&kernelContext_157)->collider_info_0 = collider_info_59;
    (&kernelContext_157)->view_chunks_0 = view_chunks_59;
    (&kernelContext_157)->view_0 = view_59;
    (&kernelContext_157)->objects_0 = objects_59;
    (&kernelContext_157)->shot_bodies_0 = shot_bodies_59;
    (&kernelContext_157)->lines_0 = lines_59;
    (&kernelContext_157)->geometry_0 = geometry_59;
    (&kernelContext_157)->shots_0 = shots_59;
    (&kernelContext_157)->lp_0 = lp_59;
    (&kernelContext_157)->ext_0 = ext_61;
    (&kernelContext_157)->lp_sides_0 = lp_sides_59;
    (&kernelContext_157)->engine_counters_0 = engine_counters_59;
    (&kernelContext_157)->engine_contacts_0 = engine_contacts_59;
    (&kernelContext_157)->cs_anchors_0 = cs_anchors_59;
    threadgroup array<uint, int(1024)> g_scan_59;
    (&kernelContext_157)->g_scan_0 = &g_scan_59;
    cs_pick_0(id_56.x, &kernelContext_157);
    return;
}

[[kernel]] void fr_cs_record(uint3 id_57 [[thread_position_in_grid]], FractureParams_0 constant* params_60 [[buffer(0)]], uint device* work_60 [[buffer(4)]], packed_float4 device* frames_60 [[buffer(6)]], packed_float4 device* bodies_60 [[buffer(7)]], packed_float4 device* chunks_60 [[buffer(1)]], uint device* sort_60 [[buffer(11)]], atomic<uint> device* atom_60 [[buffer(5)]], Record_natural_0 device* manifolds_60 [[buffer(10)]], packed_float4 device* loads_60 [[buffer(12)]], packed_uint4 device* bonds_60 [[buffer(2)]], uint device* args_60 [[buffer(13)]], uint device* info_60 [[buffer(15)]], ScanParams_0 constant* scan_60 [[buffer(14)]], packed_float4 device* parts_60 [[buffer(3)]], packed_float4 device* colliders_60 [[buffer(8)]], packed_uint4 device* collider_info_60 [[buffer(9)]], packed_float4 device* view_chunks_60 [[buffer(17)]], ViewParams_0 constant* view_60 [[buffer(16)]], packed_float4 device* objects_60 [[buffer(18)]], uint device* shot_bodies_60 [[buffer(20)]], float device* lines_60 [[buffer(19)]], packed_float4 device* geometry_60 [[buffer(21)]], packed_float4 device* shots_60 [[buffer(22)]], packed_float4 device* lp_60 [[buffer(24)]], packed_float4 device* ext_62 [[buffer(23)]], packed_float4 device* lp_sides_60 [[buffer(28)]], uint device* engine_counters_60 [[buffer(26)]], packed_float4 device* engine_contacts_60 [[buffer(25)]], packed_float4 device* cs_anchors_60 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_158;
    (&kernelContext_158)->params_0 = params_60;
    (&kernelContext_158)->work_0 = work_60;
    (&kernelContext_158)->frames_0 = frames_60;
    (&kernelContext_158)->bodies_0 = bodies_60;
    (&kernelContext_158)->chunks_0 = chunks_60;
    (&kernelContext_158)->sort_0 = sort_60;
    (&kernelContext_158)->atom_0 = atom_60;
    (&kernelContext_158)->manifolds_0 = manifolds_60;
    (&kernelContext_158)->loads_0 = loads_60;
    (&kernelContext_158)->bonds_0 = bonds_60;
    (&kernelContext_158)->args_0 = args_60;
    (&kernelContext_158)->info_0 = info_60;
    (&kernelContext_158)->scan_0 = scan_60;
    (&kernelContext_158)->parts_0 = parts_60;
    (&kernelContext_158)->colliders_0 = colliders_60;
    (&kernelContext_158)->collider_info_0 = collider_info_60;
    (&kernelContext_158)->view_chunks_0 = view_chunks_60;
    (&kernelContext_158)->view_0 = view_60;
    (&kernelContext_158)->objects_0 = objects_60;
    (&kernelContext_158)->shot_bodies_0 = shot_bodies_60;
    (&kernelContext_158)->lines_0 = lines_60;
    (&kernelContext_158)->geometry_0 = geometry_60;
    (&kernelContext_158)->shots_0 = shots_60;
    (&kernelContext_158)->lp_0 = lp_60;
    (&kernelContext_158)->ext_0 = ext_62;
    (&kernelContext_158)->lp_sides_0 = lp_sides_60;
    (&kernelContext_158)->engine_counters_0 = engine_counters_60;
    (&kernelContext_158)->engine_contacts_0 = engine_contacts_60;
    (&kernelContext_158)->cs_anchors_0 = cs_anchors_60;
    threadgroup array<uint, int(1024)> g_scan_60;
    (&kernelContext_158)->g_scan_0 = &g_scan_60;
    uint _S1129 = id_57.x;
    uint _S1130 = a_counters_1(&kernelContext_158);
    uint _S1131 = atomic_load_explicit((&kernelContext_158)->atom_0+(_S1130 + 14U), memory_order_relaxed);
    if(_S1129 >= _S1131)
    {
        return;
    }
    uint _S1132 = w_cs_bodies_0(&kernelContext_158);
    uint device* _S1133 = (&kernelContext_158)->work_0+(_S1132 + _S1129);
    uint s_18 = *_S1133;
    uint _S1134 = a_cs_0(*_S1133, &kernelContext_158);
    uint _S1135 = atomic_load_explicit((&kernelContext_158)->atom_0+(_S1134 + 11U), memory_order_relaxed);
    uint key_2 = _S1135 - 1U;
    thread uint k_41;
    thread uint s2_0;
    thread uint ob_6;
    thread uint co_6;
    thread float3 p_19;
    thread float3 u_10;
    thread float v_15;
    thread float m_eff_8;
    thread float t_in_9;
    bool _S1136 = cs_side_0(key_2 / 2U, key_2 % 2U, &k_41, &s2_0, &ob_6, &co_6, &p_19, &u_10, &v_15, &m_eff_8, &t_in_9, &kernelContext_158);
    if(!_S1136)
    {
        return;
    }
    uint4 _S1137 = uint4(*((&kernelContext_158)->collider_info_0+(2U * co_6 + 1U))) ;
    float e_o_0 = (as_type<float>((_S1137.y)));
    float _S1138 = max((as_type<float>((_S1137.z))), 0.00100000004749745f);
    float j_7 = m_eff_8 * v_15;
    float tau_2 = hertz_duration_0(m_eff_8, _S1138, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / (float4(*((&kernelContext_158)->geometry_0+(3U * k_41 + 1U))) ).w), v_15);
    uint _S1139 = lp_body_0(s_18, &kernelContext_158);
    *((&kernelContext_158)->lp_0+(_S1139 + 2U)) = packed_float4(float4(from_rubble_0(p_19), 3.14159274101257324f * j_7 / (2.0f * tau_2))) ;
    *((&kernelContext_158)->lp_0+(_S1139 + 3U)) = packed_float4(float4(from_rubble_0(u_10), v_15)) ;
    *((&kernelContext_158)->lp_0+(_S1139 + 4U)) = packed_float4(float4(m_eff_8, _S1138, e_o_0, (as_type<float>((ob_6))))) ;
    *((&kernelContext_158)->lp_0+(_S1139 + 5U)) = packed_float4(float4((as_type<float>((k_41))), j_7 / tau_2, 1.0f / (float4(*((&kernelContext_158)->bodies_0+7U * ob_6)) ).w, t_in_9)) ;
    *((&kernelContext_158)->lp_0+(_S1139 + 6U)) = packed_float4(float4((as_type<float>((co_6))), (as_type<float>((_S1137.w))), 0.0f, 0.0f)) ;
    atomic_store_explicit((&kernelContext_158)->atom_0+(_S1134 + 1U), k_41 + 1U, memory_order_relaxed);
    uint _S1140 = atomic_fetch_add_explicit((&kernelContext_158)->atom_0+(_S1130 + 12U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_hertz(uint3 id_58 [[thread_position_in_grid]], FractureParams_0 constant* params_61 [[buffer(0)]], uint device* work_61 [[buffer(4)]], packed_float4 device* frames_61 [[buffer(6)]], packed_float4 device* bodies_61 [[buffer(7)]], packed_float4 device* chunks_61 [[buffer(1)]], uint device* sort_61 [[buffer(11)]], atomic<uint> device* atom_61 [[buffer(5)]], Record_natural_0 device* manifolds_61 [[buffer(10)]], packed_float4 device* loads_61 [[buffer(12)]], packed_uint4 device* bonds_61 [[buffer(2)]], uint device* args_61 [[buffer(13)]], uint device* info_61 [[buffer(15)]], ScanParams_0 constant* scan_61 [[buffer(14)]], packed_float4 device* parts_61 [[buffer(3)]], packed_float4 device* colliders_61 [[buffer(8)]], packed_uint4 device* collider_info_61 [[buffer(9)]], packed_float4 device* view_chunks_61 [[buffer(17)]], ViewParams_0 constant* view_61 [[buffer(16)]], packed_float4 device* objects_61 [[buffer(18)]], uint device* shot_bodies_61 [[buffer(20)]], float device* lines_61 [[buffer(19)]], packed_float4 device* geometry_61 [[buffer(21)]], packed_float4 device* shots_61 [[buffer(22)]], packed_float4 device* lp_61 [[buffer(24)]], packed_float4 device* ext_63 [[buffer(23)]], packed_float4 device* lp_sides_61 [[buffer(28)]], uint device* engine_counters_61 [[buffer(26)]], packed_float4 device* engine_contacts_61 [[buffer(25)]], packed_float4 device* cs_anchors_61 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_159;
    (&kernelContext_159)->params_0 = params_61;
    (&kernelContext_159)->work_0 = work_61;
    (&kernelContext_159)->frames_0 = frames_61;
    (&kernelContext_159)->bodies_0 = bodies_61;
    (&kernelContext_159)->chunks_0 = chunks_61;
    (&kernelContext_159)->sort_0 = sort_61;
    (&kernelContext_159)->atom_0 = atom_61;
    (&kernelContext_159)->manifolds_0 = manifolds_61;
    (&kernelContext_159)->loads_0 = loads_61;
    (&kernelContext_159)->bonds_0 = bonds_61;
    (&kernelContext_159)->args_0 = args_61;
    (&kernelContext_159)->info_0 = info_61;
    (&kernelContext_159)->scan_0 = scan_61;
    (&kernelContext_159)->parts_0 = parts_61;
    (&kernelContext_159)->colliders_0 = colliders_61;
    (&kernelContext_159)->collider_info_0 = collider_info_61;
    (&kernelContext_159)->view_chunks_0 = view_chunks_61;
    (&kernelContext_159)->view_0 = view_61;
    (&kernelContext_159)->objects_0 = objects_61;
    (&kernelContext_159)->shot_bodies_0 = shot_bodies_61;
    (&kernelContext_159)->lines_0 = lines_61;
    (&kernelContext_159)->geometry_0 = geometry_61;
    (&kernelContext_159)->shots_0 = shots_61;
    (&kernelContext_159)->lp_0 = lp_61;
    (&kernelContext_159)->ext_0 = ext_63;
    (&kernelContext_159)->lp_sides_0 = lp_sides_61;
    (&kernelContext_159)->engine_counters_0 = engine_counters_61;
    (&kernelContext_159)->engine_contacts_0 = engine_contacts_61;
    (&kernelContext_159)->cs_anchors_0 = cs_anchors_61;
    threadgroup array<uint, int(1024)> g_scan_61;
    (&kernelContext_159)->g_scan_0 = &g_scan_61;
    uint i_11 = id_58.x;
    bool _S1141;
    if(i_11 >= (params_61->bond_count_0))
    {
        _S1141 = true;
    }
    else
    {
        uint _S1142 = w_alive_0(&kernelContext_159);
        _S1141 = (*((&kernelContext_159)->work_0+(_S1142 + i_11))) == 0U;
    }
    if(_S1141)
    {
        return;
    }
    uint4 _S1143 = uint4(*((&kernelContext_159)->bonds_0+i_11)) ;
    uint _S1144 = body_of_0(_S1143.x, &kernelContext_159);
    bool _S1145 = cs_struck_1(_S1144, &kernelContext_159);
    if(!_S1145)
    {
        return;
    }
    uint _S1146 = lp_body_0(_S1144, &kernelContext_159);
    bool _S1147 = strike_share_breaks_0(_S1143, (float4(*((&kernelContext_159)->lp_0+(_S1146 + 2U))) ).xyz, (float4(*((&kernelContext_159)->lp_0+(_S1146 + 5U))) ).y, &kernelContext_159);
    if(_S1147)
    {
        uint _S1148 = w_alive_0(&kernelContext_159);
        *((&kernelContext_159)->work_0+(_S1148 + i_11)) = 0U;
        uint _S1149 = a_dirty_0(&kernelContext_159);
        uint _S1150 = atomic_fetch_or_explicit((&kernelContext_159)->atom_0+(_S1149 + _S1144), 1U, memory_order_relaxed);
        uint _S1151 = a_counters_1(&kernelContext_159);
        uint _S1152 = atomic_fetch_add_explicit((&kernelContext_159)->atom_0+_S1151, 1U, memory_order_relaxed);
        uint _S1153 = atomic_fetch_add_explicit((&kernelContext_159)->atom_0+(_S1151 + 1U), 1U, memory_order_relaxed);
        uint _S1154 = atomic_fetch_add_explicit((&kernelContext_159)->atom_0+(_S1151 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_gate2(uint3 id_59 [[thread_position_in_grid]], FractureParams_0 constant* params_62 [[buffer(0)]], uint device* work_62 [[buffer(4)]], packed_float4 device* frames_62 [[buffer(6)]], packed_float4 device* bodies_62 [[buffer(7)]], packed_float4 device* chunks_62 [[buffer(1)]], uint device* sort_62 [[buffer(11)]], atomic<uint> device* atom_62 [[buffer(5)]], Record_natural_0 device* manifolds_62 [[buffer(10)]], packed_float4 device* loads_62 [[buffer(12)]], packed_uint4 device* bonds_62 [[buffer(2)]], uint device* args_62 [[buffer(13)]], uint device* info_62 [[buffer(15)]], ScanParams_0 constant* scan_62 [[buffer(14)]], packed_float4 device* parts_62 [[buffer(3)]], packed_float4 device* colliders_62 [[buffer(8)]], packed_uint4 device* collider_info_62 [[buffer(9)]], packed_float4 device* view_chunks_62 [[buffer(17)]], ViewParams_0 constant* view_62 [[buffer(16)]], packed_float4 device* objects_62 [[buffer(18)]], uint device* shot_bodies_62 [[buffer(20)]], float device* lines_62 [[buffer(19)]], packed_float4 device* geometry_62 [[buffer(21)]], packed_float4 device* shots_62 [[buffer(22)]], packed_float4 device* lp_62 [[buffer(24)]], packed_float4 device* ext_64 [[buffer(23)]], packed_float4 device* lp_sides_62 [[buffer(28)]], uint device* engine_counters_62 [[buffer(26)]], packed_float4 device* engine_contacts_62 [[buffer(25)]], packed_float4 device* cs_anchors_62 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_160;
    (&kernelContext_160)->params_0 = params_62;
    (&kernelContext_160)->work_0 = work_62;
    (&kernelContext_160)->frames_0 = frames_62;
    (&kernelContext_160)->bodies_0 = bodies_62;
    (&kernelContext_160)->chunks_0 = chunks_62;
    (&kernelContext_160)->sort_0 = sort_62;
    (&kernelContext_160)->atom_0 = atom_62;
    (&kernelContext_160)->manifolds_0 = manifolds_62;
    (&kernelContext_160)->loads_0 = loads_62;
    (&kernelContext_160)->bonds_0 = bonds_62;
    (&kernelContext_160)->args_0 = args_62;
    (&kernelContext_160)->info_0 = info_62;
    (&kernelContext_160)->scan_0 = scan_62;
    (&kernelContext_160)->parts_0 = parts_62;
    (&kernelContext_160)->colliders_0 = colliders_62;
    (&kernelContext_160)->collider_info_0 = collider_info_62;
    (&kernelContext_160)->view_chunks_0 = view_chunks_62;
    (&kernelContext_160)->view_0 = view_62;
    (&kernelContext_160)->objects_0 = objects_62;
    (&kernelContext_160)->shot_bodies_0 = shot_bodies_62;
    (&kernelContext_160)->lines_0 = lines_62;
    (&kernelContext_160)->geometry_0 = geometry_62;
    (&kernelContext_160)->shots_0 = shots_62;
    (&kernelContext_160)->lp_0 = lp_62;
    (&kernelContext_160)->ext_0 = ext_64;
    (&kernelContext_160)->lp_sides_0 = lp_sides_62;
    (&kernelContext_160)->engine_counters_0 = engine_counters_62;
    (&kernelContext_160)->engine_contacts_0 = engine_contacts_62;
    (&kernelContext_160)->cs_anchors_0 = cs_anchors_62;
    threadgroup array<uint, int(1024)> g_scan_62;
    (&kernelContext_160)->g_scan_0 = &g_scan_62;
    uint _S1155 = a_counters_1(&kernelContext_160);
    uint _S1156 = atomic_load_explicit((&kernelContext_160)->atom_0+_S1155, memory_order_relaxed);
    uint _S1157;
    if(_S1156 > 0U)
    {
        uint _S1158 = cs_rows_0(&kernelContext_160);
        uint _S1159 = groups_of_0(_S1158, 64U);
        _S1157 = _S1159;
    }
    else
    {
        _S1157 = 0U;
    }
    set_args_0(13U, _S1157, &kernelContext_160);
    return;
}

[[kernel]] void fr_cs_save(uint3 id_60 [[thread_position_in_grid]], FractureParams_0 constant* params_63 [[buffer(0)]], uint device* work_63 [[buffer(4)]], packed_float4 device* frames_63 [[buffer(6)]], packed_float4 device* bodies_63 [[buffer(7)]], packed_float4 device* chunks_63 [[buffer(1)]], uint device* sort_63 [[buffer(11)]], atomic<uint> device* atom_63 [[buffer(5)]], Record_natural_0 device* manifolds_63 [[buffer(10)]], packed_float4 device* loads_63 [[buffer(12)]], packed_uint4 device* bonds_63 [[buffer(2)]], uint device* args_63 [[buffer(13)]], uint device* info_63 [[buffer(15)]], ScanParams_0 constant* scan_63 [[buffer(14)]], packed_float4 device* parts_63 [[buffer(3)]], packed_float4 device* colliders_63 [[buffer(8)]], packed_uint4 device* collider_info_63 [[buffer(9)]], packed_float4 device* view_chunks_63 [[buffer(17)]], ViewParams_0 constant* view_63 [[buffer(16)]], packed_float4 device* objects_63 [[buffer(18)]], uint device* shot_bodies_63 [[buffer(20)]], float device* lines_63 [[buffer(19)]], packed_float4 device* geometry_63 [[buffer(21)]], packed_float4 device* shots_63 [[buffer(22)]], packed_float4 device* lp_63 [[buffer(24)]], packed_float4 device* ext_65 [[buffer(23)]], packed_float4 device* lp_sides_63 [[buffer(28)]], uint device* engine_counters_63 [[buffer(26)]], packed_float4 device* engine_contacts_63 [[buffer(25)]], packed_float4 device* cs_anchors_63 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_161;
    (&kernelContext_161)->params_0 = params_63;
    (&kernelContext_161)->work_0 = work_63;
    (&kernelContext_161)->frames_0 = frames_63;
    (&kernelContext_161)->bodies_0 = bodies_63;
    (&kernelContext_161)->chunks_0 = chunks_63;
    (&kernelContext_161)->sort_0 = sort_63;
    (&kernelContext_161)->atom_0 = atom_63;
    (&kernelContext_161)->manifolds_0 = manifolds_63;
    (&kernelContext_161)->loads_0 = loads_63;
    (&kernelContext_161)->bonds_0 = bonds_63;
    (&kernelContext_161)->args_0 = args_63;
    (&kernelContext_161)->info_0 = info_63;
    (&kernelContext_161)->scan_0 = scan_63;
    (&kernelContext_161)->parts_0 = parts_63;
    (&kernelContext_161)->colliders_0 = colliders_63;
    (&kernelContext_161)->collider_info_0 = collider_info_63;
    (&kernelContext_161)->view_chunks_0 = view_chunks_63;
    (&kernelContext_161)->view_0 = view_63;
    (&kernelContext_161)->objects_0 = objects_63;
    (&kernelContext_161)->shot_bodies_0 = shot_bodies_63;
    (&kernelContext_161)->lines_0 = lines_63;
    (&kernelContext_161)->geometry_0 = geometry_63;
    (&kernelContext_161)->shots_0 = shots_63;
    (&kernelContext_161)->lp_0 = lp_63;
    (&kernelContext_161)->ext_0 = ext_65;
    (&kernelContext_161)->lp_sides_0 = lp_sides_63;
    (&kernelContext_161)->engine_counters_0 = engine_counters_63;
    (&kernelContext_161)->engine_contacts_0 = engine_contacts_63;
    (&kernelContext_161)->cs_anchors_0 = cs_anchors_63;
    threadgroup array<uint, int(1024)> g_scan_63;
    (&kernelContext_161)->g_scan_0 = &g_scan_63;
    uint r_26 = id_60.x;
    uint _S1160 = cs_rows_0(&kernelContext_161);
    if(r_26 >= _S1160)
    {
        return;
    }
    uint _S1161 = cs_row_0(r_26, &kernelContext_161);
    uint4 _S1162 = (as_type<uint4>((float4(*((&kernelContext_161)->engine_contacts_0+_S1161)) )));
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
        float4 _S1163 = float4(0.0f) ;
        bool _S1164 = side_7 == 0U;
        uint col_1;
        if(_S1164)
        {
            col_1 = _S1162.x;
        }
        else
        {
            col_1 = _S1162.y;
        }
        uint k_42;
        if((_S1162.w) != 0U)
        {
            k_42 = (uint4(*((&kernelContext_161)->collider_info_0+2U * col_1)) ).x;
        }
        else
        {
            k_42 = 4294967295U;
        }
        bool _S1165;
        if(k_42 != 4294967295U)
        {
            _S1165 = k_42 < ((&kernelContext_161)->params_0->chunk_count_0);
        }
        else
        {
            _S1165 = false;
        }
        bool _S1166;
        if(_S1165)
        {
            uint _S1167 = body_of_0(k_42, &kernelContext_161);
            bool _S1168 = dirty_0(_S1167, &kernelContext_161);
            _S1166 = _S1168;
        }
        else
        {
            _S1166 = false;
        }
        float4 saved_0;
        if(_S1166)
        {
            uint _S1169 = body_of_0(k_42, &kernelContext_161);
            uint at_16 = 7U * _S1169;
            float3 anchor_0;
            if(_S1164)
            {
                anchor_0 = (float4(*((&kernelContext_161)->engine_contacts_0+(_S1161 + 2U))) ).xyz;
            }
            else
            {
                anchor_0 = (float4(*((&kernelContext_161)->engine_contacts_0+(_S1161 + 3U))) ).xyz;
            }
            float3 _S1170 = (float4(*((&kernelContext_161)->bodies_0+at_16)) ).xyz;
            thread Quat_0 _S1171 = quat_of_0(float4(*((&kernelContext_161)->bodies_0+(at_16 + 1U))) );
            float3 _S1172 = rotate_0(&_S1171, anchor_0);
            saved_0 = float4(_S1170 + _S1172, 1.0f);
        }
        else
        {
            saved_0 = _S1163;
        }
        *((&kernelContext_161)->cs_anchors_0+(2U * r_26 + side_7)) = packed_float4(saved_0) ;
        side_7 = side_7 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_clean(uint3 id_61 [[thread_position_in_grid]], FractureParams_0 constant* params_64 [[buffer(0)]], uint device* work_64 [[buffer(4)]], packed_float4 device* frames_64 [[buffer(6)]], packed_float4 device* bodies_64 [[buffer(7)]], packed_float4 device* chunks_64 [[buffer(1)]], uint device* sort_64 [[buffer(11)]], atomic<uint> device* atom_64 [[buffer(5)]], Record_natural_0 device* manifolds_64 [[buffer(10)]], packed_float4 device* loads_64 [[buffer(12)]], packed_uint4 device* bonds_64 [[buffer(2)]], uint device* args_64 [[buffer(13)]], uint device* info_64 [[buffer(15)]], ScanParams_0 constant* scan_64 [[buffer(14)]], packed_float4 device* parts_64 [[buffer(3)]], packed_float4 device* colliders_64 [[buffer(8)]], packed_uint4 device* collider_info_64 [[buffer(9)]], packed_float4 device* view_chunks_64 [[buffer(17)]], ViewParams_0 constant* view_64 [[buffer(16)]], packed_float4 device* objects_64 [[buffer(18)]], uint device* shot_bodies_64 [[buffer(20)]], float device* lines_64 [[buffer(19)]], packed_float4 device* geometry_64 [[buffer(21)]], packed_float4 device* shots_64 [[buffer(22)]], packed_float4 device* lp_64 [[buffer(24)]], packed_float4 device* ext_66 [[buffer(23)]], packed_float4 device* lp_sides_64 [[buffer(28)]], uint device* engine_counters_64 [[buffer(26)]], packed_float4 device* engine_contacts_64 [[buffer(25)]], packed_float4 device* cs_anchors_64 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_162;
    (&kernelContext_162)->params_0 = params_64;
    (&kernelContext_162)->work_0 = work_64;
    (&kernelContext_162)->frames_0 = frames_64;
    (&kernelContext_162)->bodies_0 = bodies_64;
    (&kernelContext_162)->chunks_0 = chunks_64;
    (&kernelContext_162)->sort_0 = sort_64;
    (&kernelContext_162)->atom_0 = atom_64;
    (&kernelContext_162)->manifolds_0 = manifolds_64;
    (&kernelContext_162)->loads_0 = loads_64;
    (&kernelContext_162)->bonds_0 = bonds_64;
    (&kernelContext_162)->args_0 = args_64;
    (&kernelContext_162)->info_0 = info_64;
    (&kernelContext_162)->scan_0 = scan_64;
    (&kernelContext_162)->parts_0 = parts_64;
    (&kernelContext_162)->colliders_0 = colliders_64;
    (&kernelContext_162)->collider_info_0 = collider_info_64;
    (&kernelContext_162)->view_chunks_0 = view_chunks_64;
    (&kernelContext_162)->view_0 = view_64;
    (&kernelContext_162)->objects_0 = objects_64;
    (&kernelContext_162)->shot_bodies_0 = shot_bodies_64;
    (&kernelContext_162)->lines_0 = lines_64;
    (&kernelContext_162)->geometry_0 = geometry_64;
    (&kernelContext_162)->shots_0 = shots_64;
    (&kernelContext_162)->lp_0 = lp_64;
    (&kernelContext_162)->ext_0 = ext_66;
    (&kernelContext_162)->lp_sides_0 = lp_sides_64;
    (&kernelContext_162)->engine_counters_0 = engine_counters_64;
    (&kernelContext_162)->engine_contacts_0 = engine_contacts_64;
    (&kernelContext_162)->cs_anchors_0 = cs_anchors_64;
    threadgroup array<uint, int(1024)> g_scan_64;
    (&kernelContext_162)->g_scan_0 = &g_scan_64;
    uint _S1173 = id_61.x;
    uint _S1174 = a_counters_1(&kernelContext_162);
    uint _S1175 = atomic_load_explicit((&kernelContext_162)->atom_0+(_S1174 + 14U), memory_order_relaxed);
    if(_S1173 >= _S1175)
    {
        return;
    }
    uint _S1176 = w_cs_bodies_0(&kernelContext_162);
    uint _S1177 = a_cs_0(*((&kernelContext_162)->work_0+(_S1176 + _S1173)), &kernelContext_162);
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
        atomic_store_explicit((&kernelContext_162)->atom_0+(_S1177 + w_8), 0U, memory_order_relaxed);
        w_8 = w_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_fix(uint3 id_62 [[thread_position_in_grid]], FractureParams_0 constant* params_65 [[buffer(0)]], uint device* work_65 [[buffer(4)]], packed_float4 device* frames_65 [[buffer(6)]], packed_float4 device* bodies_65 [[buffer(7)]], packed_float4 device* chunks_65 [[buffer(1)]], uint device* sort_65 [[buffer(11)]], atomic<uint> device* atom_65 [[buffer(5)]], Record_natural_0 device* manifolds_65 [[buffer(10)]], packed_float4 device* loads_65 [[buffer(12)]], packed_uint4 device* bonds_65 [[buffer(2)]], uint device* args_65 [[buffer(13)]], uint device* info_65 [[buffer(15)]], ScanParams_0 constant* scan_65 [[buffer(14)]], packed_float4 device* parts_65 [[buffer(3)]], packed_float4 device* colliders_65 [[buffer(8)]], packed_uint4 device* collider_info_65 [[buffer(9)]], packed_float4 device* view_chunks_65 [[buffer(17)]], ViewParams_0 constant* view_65 [[buffer(16)]], packed_float4 device* objects_65 [[buffer(18)]], uint device* shot_bodies_65 [[buffer(20)]], float device* lines_65 [[buffer(19)]], packed_float4 device* geometry_65 [[buffer(21)]], packed_float4 device* shots_65 [[buffer(22)]], packed_float4 device* lp_65 [[buffer(24)]], packed_float4 device* ext_67 [[buffer(23)]], packed_float4 device* lp_sides_65 [[buffer(28)]], uint device* engine_counters_65 [[buffer(26)]], packed_float4 device* engine_contacts_65 [[buffer(25)]], packed_float4 device* cs_anchors_65 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_163;
    (&kernelContext_163)->params_0 = params_65;
    (&kernelContext_163)->work_0 = work_65;
    (&kernelContext_163)->frames_0 = frames_65;
    (&kernelContext_163)->bodies_0 = bodies_65;
    (&kernelContext_163)->chunks_0 = chunks_65;
    (&kernelContext_163)->sort_0 = sort_65;
    (&kernelContext_163)->atom_0 = atom_65;
    (&kernelContext_163)->manifolds_0 = manifolds_65;
    (&kernelContext_163)->loads_0 = loads_65;
    (&kernelContext_163)->bonds_0 = bonds_65;
    (&kernelContext_163)->args_0 = args_65;
    (&kernelContext_163)->info_0 = info_65;
    (&kernelContext_163)->scan_0 = scan_65;
    (&kernelContext_163)->parts_0 = parts_65;
    (&kernelContext_163)->colliders_0 = colliders_65;
    (&kernelContext_163)->collider_info_0 = collider_info_65;
    (&kernelContext_163)->view_chunks_0 = view_chunks_65;
    (&kernelContext_163)->view_0 = view_65;
    (&kernelContext_163)->objects_0 = objects_65;
    (&kernelContext_163)->shot_bodies_0 = shot_bodies_65;
    (&kernelContext_163)->lines_0 = lines_65;
    (&kernelContext_163)->geometry_0 = geometry_65;
    (&kernelContext_163)->shots_0 = shots_65;
    (&kernelContext_163)->lp_0 = lp_65;
    (&kernelContext_163)->ext_0 = ext_67;
    (&kernelContext_163)->lp_sides_0 = lp_sides_65;
    (&kernelContext_163)->engine_counters_0 = engine_counters_65;
    (&kernelContext_163)->engine_contacts_0 = engine_contacts_65;
    (&kernelContext_163)->cs_anchors_0 = cs_anchors_65;
    threadgroup array<uint, int(1024)> g_scan_65;
    (&kernelContext_163)->g_scan_0 = &g_scan_65;
    uint r_27 = id_62.x;
    uint _S1178 = cs_rows_0(&kernelContext_163);
    if(r_27 >= _S1178)
    {
        return;
    }
    uint _S1179 = cs_row_0(r_27, &kernelContext_163);
    uint4 _S1180 = (as_type<uint4>((float4(*((&kernelContext_163)->engine_contacts_0+_S1179)) )));
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
        float4 _S1181 = float4(*((&kernelContext_163)->cs_anchors_0+(2U * r_27 + side_8))) ;
        if((_S1181.w) == 0.0f)
        {
            side_8 = side_8 + 1U;
            continue;
        }
        uint col_2;
        if(side_8 == 0U)
        {
            col_2 = _S1180.x;
        }
        else
        {
            col_2 = _S1180.y;
        }
        uint at_17 = 7U * (as_type<uint>(((float4(*((&kernelContext_163)->colliders_0+4U * col_2)) ).x)));
        thread Quat_0 _S1182 = quat_of_0(float4(*((&kernelContext_163)->bodies_0+(at_17 + 1U))) );
        Quat_0 _S1183 = conj_0(&_S1182);
        float3 _S1184 = _S1181.xyz - (float4(*((&kernelContext_163)->bodies_0+at_17)) ).xyz;
        thread Quat_0 _S1185 = _S1183;
        float3 _S1186 = rotate_0(&_S1185, _S1184);
        (*((&kernelContext_163)->engine_contacts_0+(_S1179 + 2U + side_8))).xyz = _S1186;
        moved_0 = true;
        side_8 = side_8 + 1U;
    }
    if(moved_0)
    {
        (*((&kernelContext_163)->engine_contacts_0+(_S1179 + 4U)))[int(3)] = 0.0f;
        (*((&kernelContext_163)->engine_contacts_0+(_S1179 + 5U)))[int(3)] = 0.0f;
        uint _S1187 = _S1179 + 6U;
        *((&kernelContext_163)->engine_contacts_0+_S1187) = packed_float4(float4(1.0f, 1.0f, 0.0f, (float4(*((&kernelContext_163)->engine_contacts_0+_S1187)) ).w)) ;
        (*((&kernelContext_163)->engine_contacts_0+(_S1179 + 7U)))[int(1)] = 0.0f;
    }
    return;
}

[[kernel]] void fr_cs_count(uint3 id_63 [[thread_position_in_grid]], FractureParams_0 constant* params_66 [[buffer(0)]], uint device* work_66 [[buffer(4)]], packed_float4 device* frames_66 [[buffer(6)]], packed_float4 device* bodies_66 [[buffer(7)]], packed_float4 device* chunks_66 [[buffer(1)]], uint device* sort_66 [[buffer(11)]], atomic<uint> device* atom_66 [[buffer(5)]], Record_natural_0 device* manifolds_66 [[buffer(10)]], packed_float4 device* loads_66 [[buffer(12)]], packed_uint4 device* bonds_66 [[buffer(2)]], uint device* args_66 [[buffer(13)]], uint device* info_66 [[buffer(15)]], ScanParams_0 constant* scan_66 [[buffer(14)]], packed_float4 device* parts_66 [[buffer(3)]], packed_float4 device* colliders_66 [[buffer(8)]], packed_uint4 device* collider_info_66 [[buffer(9)]], packed_float4 device* view_chunks_66 [[buffer(17)]], ViewParams_0 constant* view_66 [[buffer(16)]], packed_float4 device* objects_66 [[buffer(18)]], uint device* shot_bodies_66 [[buffer(20)]], float device* lines_66 [[buffer(19)]], packed_float4 device* geometry_66 [[buffer(21)]], packed_float4 device* shots_66 [[buffer(22)]], packed_float4 device* lp_66 [[buffer(24)]], packed_float4 device* ext_68 [[buffer(23)]], packed_float4 device* lp_sides_66 [[buffer(28)]], uint device* engine_counters_66 [[buffer(26)]], packed_float4 device* engine_contacts_66 [[buffer(25)]], packed_float4 device* cs_anchors_66 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_164;
    (&kernelContext_164)->params_0 = params_66;
    (&kernelContext_164)->work_0 = work_66;
    (&kernelContext_164)->frames_0 = frames_66;
    (&kernelContext_164)->bodies_0 = bodies_66;
    (&kernelContext_164)->chunks_0 = chunks_66;
    (&kernelContext_164)->sort_0 = sort_66;
    (&kernelContext_164)->atom_0 = atom_66;
    (&kernelContext_164)->manifolds_0 = manifolds_66;
    (&kernelContext_164)->loads_0 = loads_66;
    (&kernelContext_164)->bonds_0 = bonds_66;
    (&kernelContext_164)->args_0 = args_66;
    (&kernelContext_164)->info_0 = info_66;
    (&kernelContext_164)->scan_0 = scan_66;
    (&kernelContext_164)->parts_0 = parts_66;
    (&kernelContext_164)->colliders_0 = colliders_66;
    (&kernelContext_164)->collider_info_0 = collider_info_66;
    (&kernelContext_164)->view_chunks_0 = view_chunks_66;
    (&kernelContext_164)->view_0 = view_66;
    (&kernelContext_164)->objects_0 = objects_66;
    (&kernelContext_164)->shot_bodies_0 = shot_bodies_66;
    (&kernelContext_164)->lines_0 = lines_66;
    (&kernelContext_164)->geometry_0 = geometry_66;
    (&kernelContext_164)->shots_0 = shots_66;
    (&kernelContext_164)->lp_0 = lp_66;
    (&kernelContext_164)->ext_0 = ext_68;
    (&kernelContext_164)->lp_sides_0 = lp_sides_66;
    (&kernelContext_164)->engine_counters_0 = engine_counters_66;
    (&kernelContext_164)->engine_contacts_0 = engine_contacts_66;
    (&kernelContext_164)->cs_anchors_0 = cs_anchors_66;
    threadgroup array<uint, int(1024)> g_scan_66;
    (&kernelContext_164)->g_scan_0 = &g_scan_66;
    uint k_43 = id_63.x;
    if(k_43 >= (params_66->chunk_count_0))
    {
        return;
    }
    uint _S1188 = body_of_0(k_43, &kernelContext_164);
    bool _S1189 = cs_struck_1(_S1188, &kernelContext_164);
    if(!_S1189)
    {
        return;
    }
    uint _S1190 = a_cs_0(_S1188, &kernelContext_164);
    uint _S1191 = atomic_fetch_max_explicit((&kernelContext_164)->atom_0+(_S1190 + 3U), ~k_43, memory_order_relaxed);
    if(((&kernelContext_164)->info_0[k_43] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_164)->atom_0+(_S1190 + 4U), 1U, memory_order_relaxed);
    }
    uint _S1192 = atomic_fetch_add_explicit((&kernelContext_164)->atom_0+(_S1190 + 5U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_base(uint3 id_64 [[thread_position_in_grid]], FractureParams_0 constant* params_67 [[buffer(0)]], uint device* work_67 [[buffer(4)]], packed_float4 device* frames_67 [[buffer(6)]], packed_float4 device* bodies_67 [[buffer(7)]], packed_float4 device* chunks_67 [[buffer(1)]], uint device* sort_67 [[buffer(11)]], atomic<uint> device* atom_67 [[buffer(5)]], Record_natural_0 device* manifolds_67 [[buffer(10)]], packed_float4 device* loads_67 [[buffer(12)]], packed_uint4 device* bonds_67 [[buffer(2)]], uint device* args_67 [[buffer(13)]], uint device* info_67 [[buffer(15)]], ScanParams_0 constant* scan_67 [[buffer(14)]], packed_float4 device* parts_67 [[buffer(3)]], packed_float4 device* colliders_67 [[buffer(8)]], packed_uint4 device* collider_info_67 [[buffer(9)]], packed_float4 device* view_chunks_67 [[buffer(17)]], ViewParams_0 constant* view_67 [[buffer(16)]], packed_float4 device* objects_67 [[buffer(18)]], uint device* shot_bodies_67 [[buffer(20)]], float device* lines_67 [[buffer(19)]], packed_float4 device* geometry_67 [[buffer(21)]], packed_float4 device* shots_67 [[buffer(22)]], packed_float4 device* lp_67 [[buffer(24)]], packed_float4 device* ext_69 [[buffer(23)]], packed_float4 device* lp_sides_67 [[buffer(28)]], uint device* engine_counters_67 [[buffer(26)]], packed_float4 device* engine_contacts_67 [[buffer(25)]], packed_float4 device* cs_anchors_67 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_165;
    (&kernelContext_165)->params_0 = params_67;
    (&kernelContext_165)->work_0 = work_67;
    (&kernelContext_165)->frames_0 = frames_67;
    (&kernelContext_165)->bodies_0 = bodies_67;
    (&kernelContext_165)->chunks_0 = chunks_67;
    (&kernelContext_165)->sort_0 = sort_67;
    (&kernelContext_165)->atom_0 = atom_67;
    (&kernelContext_165)->manifolds_0 = manifolds_67;
    (&kernelContext_165)->loads_0 = loads_67;
    (&kernelContext_165)->bonds_0 = bonds_67;
    (&kernelContext_165)->args_0 = args_67;
    (&kernelContext_165)->info_0 = info_67;
    (&kernelContext_165)->scan_0 = scan_67;
    (&kernelContext_165)->parts_0 = parts_67;
    (&kernelContext_165)->colliders_0 = colliders_67;
    (&kernelContext_165)->collider_info_0 = collider_info_67;
    (&kernelContext_165)->view_chunks_0 = view_chunks_67;
    (&kernelContext_165)->view_0 = view_67;
    (&kernelContext_165)->objects_0 = objects_67;
    (&kernelContext_165)->shot_bodies_0 = shot_bodies_67;
    (&kernelContext_165)->lines_0 = lines_67;
    (&kernelContext_165)->geometry_0 = geometry_67;
    (&kernelContext_165)->shots_0 = shots_67;
    (&kernelContext_165)->lp_0 = lp_67;
    (&kernelContext_165)->ext_0 = ext_69;
    (&kernelContext_165)->lp_sides_0 = lp_sides_67;
    (&kernelContext_165)->engine_counters_0 = engine_counters_67;
    (&kernelContext_165)->engine_contacts_0 = engine_contacts_67;
    (&kernelContext_165)->cs_anchors_0 = cs_anchors_67;
    threadgroup array<uint, int(1024)> g_scan_67;
    (&kernelContext_165)->g_scan_0 = &g_scan_67;
    uint _S1193 = id_64.x;
    uint _S1194 = a_counters_1(&kernelContext_165);
    uint _S1195 = atomic_load_explicit((&kernelContext_165)->atom_0+(_S1194 + 14U), memory_order_relaxed);
    if(_S1193 >= _S1195)
    {
        return;
    }
    uint _S1196 = w_cs_bodies_0(&kernelContext_165);
    uint _S1197 = a_cs_0(*((&kernelContext_165)->work_0+(_S1196 + _S1193)), &kernelContext_165);
    atomic<uint> device* _S1198 = (&kernelContext_165)->atom_0+(_S1197 + 6U);
    atomic<uint> device* _S1199 = (&kernelContext_165)->atom_0+(_S1194 + 15U);
    uint _S1200 = atomic_load_explicit((&kernelContext_165)->atom_0+(_S1197 + 5U), memory_order_relaxed);
    uint _S1201 = atomic_fetch_add_explicit(_S1199, _S1200, memory_order_relaxed);
    atomic_store_explicit(_S1198, _S1201, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_collect(uint3 id_65 [[thread_position_in_grid]], FractureParams_0 constant* params_68 [[buffer(0)]], uint device* work_68 [[buffer(4)]], packed_float4 device* frames_68 [[buffer(6)]], packed_float4 device* bodies_68 [[buffer(7)]], packed_float4 device* chunks_68 [[buffer(1)]], uint device* sort_68 [[buffer(11)]], atomic<uint> device* atom_68 [[buffer(5)]], Record_natural_0 device* manifolds_68 [[buffer(10)]], packed_float4 device* loads_68 [[buffer(12)]], packed_uint4 device* bonds_68 [[buffer(2)]], uint device* args_68 [[buffer(13)]], uint device* info_68 [[buffer(15)]], ScanParams_0 constant* scan_68 [[buffer(14)]], packed_float4 device* parts_68 [[buffer(3)]], packed_float4 device* colliders_68 [[buffer(8)]], packed_uint4 device* collider_info_68 [[buffer(9)]], packed_float4 device* view_chunks_68 [[buffer(17)]], ViewParams_0 constant* view_68 [[buffer(16)]], packed_float4 device* objects_68 [[buffer(18)]], uint device* shot_bodies_68 [[buffer(20)]], float device* lines_68 [[buffer(19)]], packed_float4 device* geometry_68 [[buffer(21)]], packed_float4 device* shots_68 [[buffer(22)]], packed_float4 device* lp_68 [[buffer(24)]], packed_float4 device* ext_70 [[buffer(23)]], packed_float4 device* lp_sides_68 [[buffer(28)]], uint device* engine_counters_68 [[buffer(26)]], packed_float4 device* engine_contacts_68 [[buffer(25)]], packed_float4 device* cs_anchors_68 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_166;
    (&kernelContext_166)->params_0 = params_68;
    (&kernelContext_166)->work_0 = work_68;
    (&kernelContext_166)->frames_0 = frames_68;
    (&kernelContext_166)->bodies_0 = bodies_68;
    (&kernelContext_166)->chunks_0 = chunks_68;
    (&kernelContext_166)->sort_0 = sort_68;
    (&kernelContext_166)->atom_0 = atom_68;
    (&kernelContext_166)->manifolds_0 = manifolds_68;
    (&kernelContext_166)->loads_0 = loads_68;
    (&kernelContext_166)->bonds_0 = bonds_68;
    (&kernelContext_166)->args_0 = args_68;
    (&kernelContext_166)->info_0 = info_68;
    (&kernelContext_166)->scan_0 = scan_68;
    (&kernelContext_166)->parts_0 = parts_68;
    (&kernelContext_166)->colliders_0 = colliders_68;
    (&kernelContext_166)->collider_info_0 = collider_info_68;
    (&kernelContext_166)->view_chunks_0 = view_chunks_68;
    (&kernelContext_166)->view_0 = view_68;
    (&kernelContext_166)->objects_0 = objects_68;
    (&kernelContext_166)->shot_bodies_0 = shot_bodies_68;
    (&kernelContext_166)->lines_0 = lines_68;
    (&kernelContext_166)->geometry_0 = geometry_68;
    (&kernelContext_166)->shots_0 = shots_68;
    (&kernelContext_166)->lp_0 = lp_68;
    (&kernelContext_166)->ext_0 = ext_70;
    (&kernelContext_166)->lp_sides_0 = lp_sides_68;
    (&kernelContext_166)->engine_counters_0 = engine_counters_68;
    (&kernelContext_166)->engine_contacts_0 = engine_contacts_68;
    (&kernelContext_166)->cs_anchors_0 = cs_anchors_68;
    threadgroup array<uint, int(1024)> g_scan_68;
    (&kernelContext_166)->g_scan_0 = &g_scan_68;
    uint k_44 = id_65.x;
    if(k_44 >= (params_68->chunk_count_0))
    {
        return;
    }
    uint _S1202 = body_of_0(k_44, &kernelContext_166);
    bool _S1203 = cs_struck_1(_S1202, &kernelContext_166);
    if(!_S1203)
    {
        return;
    }
    uint _S1204 = w_cs_list_0(&kernelContext_166);
    uint _S1205 = a_cs_0(_S1202, &kernelContext_166);
    uint _S1206 = atomic_load_explicit((&kernelContext_166)->atom_0+(_S1205 + 6U), memory_order_relaxed);
    uint _S1207 = _S1204 + _S1206;
    uint _S1208 = atomic_fetch_add_explicit((&kernelContext_166)->atom_0+(_S1205 + 7U), 1U, memory_order_relaxed);
    *((&kernelContext_166)->work_0+(_S1207 + _S1208)) = k_44;
    return;
}

void cs_load_0(uint k_45, uint s_19, KernelContext_0 thread* kernelContext_167)
{
    uint _S1209 = a_cs_0(s_19, kernelContext_167);
    uint _S1210 = atomic_load_explicit(kernelContext_167->atom_0+(_S1209 + 4U), memory_order_relaxed);
    bool supported_1 = _S1210 != 0U;
    bool fixed_2;
    if((kernelContext_167->info_0[k_45] & 1U) != 0U)
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            uint _S1211 = atomic_load_explicit(kernelContext_167->atom_0+(_S1209 + 3U), memory_order_relaxed);
            fixed_2 = (~_S1211) == k_45;
        }
        else
        {
            fixed_2 = false;
        }
    }
    uint _S1212 = lp_body_0(s_19, kernelContext_167);
    float4 _S1213 = float4(*(kernelContext_167->lp_0+(_S1212 + 2U))) ;
    float4 _S1214 = float4(*(kernelContext_167->lp_0+(_S1212 + 3U))) ;
    uint _S1215 = 7U * s_19;
    thread Quat_0 q_28;
    thread float3 origin_8;
    structure_pose_0(s_19, _S1215, &q_28, &origin_8, kernelContext_167);
    thread Quat_0 _S1216 = q_28;
    Quat_0 _S1217 = conj_0(&_S1216);
    float3 _S1218 = _S1214.xyz * float3(_S1213.w) ;
    thread Quat_0 _S1219 = _S1217;
    float3 _S1220 = rotate_0(&_S1219, _S1218);
    float3 _S1221 = _S1213.xyz - origin_8;
    thread Quat_0 _S1222 = _S1217;
    float3 _S1223 = rotate_0(&_S1222, _S1221);
    uint _S1224 = 4U * k_45;
    float4 _S1225 = float4(*(kernelContext_167->chunks_0+_S1224)) ;
    float3 _S1226 = float3(0.0f) ;
    uint _S1227 = atomic_load_explicit(kernelContext_167->atom_0+(_S1209 + 1U), memory_order_relaxed);
    float3 f_12;
    float3 t_7;
    if(_S1227 == (k_45 + 1U))
    {
        float3 _S1228 = cross(_S1223 - _S1225.xyz, _S1220);
        f_12 = _S1220;
        t_7 = _S1228;
    }
    else
    {
        f_12 = _S1226;
        t_7 = _S1226;
    }
    float inv_mass_7 = (float4(*(kernelContext_167->bodies_0+_S1215)) ).w;
    bool _S1229;
    if(!supported_1)
    {
        _S1229 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S1229 = false;
    }
    if(_S1229)
    {
        uint _S1230 = 2U * s_19;
        float3 com_2 = (float4(*(kernelContext_167->frames_0+_S1230)) ).xyz;
        Quat_0 p_20 = quat_of_0(float4(*(kernelContext_167->frames_0+(_S1230 + 1U))) );
        float3 _S1231 = (float4(*(kernelContext_167->bodies_0+(_S1215 + 2U))) ).xyz;
        thread Quat_0 _S1232 = p_20;
        Quat_0 _S1233 = conj_0(&_S1232);
        float3 _S1234 = cross(_S1223 - com_2, _S1220);
        thread Quat_0 _S1235 = _S1233;
        float3 _S1236 = rotate_0(&_S1235, _S1234);
        float3 _S1237 = _S1231 * _S1236;
        thread Quat_0 _S1238 = p_20;
        float3 _S1239 = rotate_0(&_S1238, _S1237);
        float4 _S1240 = float4(*(kernelContext_167->chunks_0+(_S1224 + 1U))) ;
        float4 _S1241 = float4(*(kernelContext_167->chunks_0+(_S1224 + 2U))) ;
        float _S1242 = _S1239.x;
        float _S1243 = _S1240.w;
        float _S1244 = _S1239.y;
        float _S1245 = _S1241.x;
        float _S1246 = _S1239.z;
        float _S1247 = _S1241.y;
        float3 t_8 = t_7 - float3(_S1240.x * _S1242 + _S1243 * _S1244 + _S1245 * _S1246, _S1243 * _S1242 + _S1240.y * _S1244 + _S1247 * _S1246, _S1245 * _S1242 + _S1247 * _S1244 + _S1240.z * _S1246);
        f_12 = f_12 - float3(_S1225.w)  * (_S1220 * float3(inv_mass_7)  + cross(_S1239, _S1225.xyz - com_2));
        t_7 = t_8;
    }
    uint _S1248 = st_f_0(k_45, 14U);
    packed_float4 device* _S1249 = kernelContext_167->lp_0+_S1248;
    float _S1250;
    if(fixed_2)
    {
        _S1250 = 1.0f;
    }
    else
    {
        _S1250 = 0.0f;
    }
    *_S1249 = packed_float4(float4(f_12, _S1250)) ;
    *(kernelContext_167->lp_0+(_S1248 + 1U)) = packed_float4(float4(t_7, 0.0f)) ;
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
        uint _S1251 = st_x_0(k_45, 14U, c_23);
        packed_float4 _S1252 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_167->lp_0+_S1251) = _S1252;
        *(kernelContext_167->lp_0+(_S1251 + 1U)) = _S1252;
        c_23 = c_23 + 1U;
    }
    uint _S1253 = st_d_0(k_45, 14U);
    packed_float4 _S1254 = packed_float4(float4(0.0f) ) ;
    *(kernelContext_167->lp_0+_S1253) = _S1254;
    *(kernelContext_167->lp_0+(_S1253 + 1U)) = _S1254;
    return;
}

void cs_break_0(uint i_12, uint s_20, KernelContext_0 thread* kernelContext_168)
{
    uint4 _S1255 = uint4(*(kernelContext_168->bonds_0+i_12)) ;
    LpBond_0 _S1256 = lp_bond_0(i_12, kernelContext_168);
    uint _S1257 = _S1255.x;
    uint _S1258 = _S1255.y;
    thread LpBond_0 _S1259 = _S1256;
    thread float3 q_lin_5;
    thread float3 q_ang_5;
    st_force_0(&_S1259, _S1257, _S1258, 14U, 0U, &q_lin_5, &q_ang_5, kernelContext_168);
    if((kernelContext_168->params_0->load_model_0) == 1U)
    {
        thread LpBond_0 _S1260 = _S1256;
        thread float3 b_lin_0;
        thread float3 b_ang_0;
        lp_force_0(&_S1260, _S1257, _S1258, 0U, &b_lin_0, &b_ang_0, kernelContext_168);
        q_lin_5 = q_lin_5 + b_lin_0;
        q_ang_5 = q_ang_5 + b_ang_0;
    }
    thread JointBond_0 _S1261 = _S1256.law_0;
    Measures_0 _S1262 = stress_measures_0(&_S1261, q_lin_5, q_ang_5);
    JointMaterial_0 _S1263 = lp_material_0(_S1256.law_0.ids_3.x, kernelContext_168);
    float _S1264 = _S1256.law_0.geom1_0.w;
    thread JointMaterial_0 _S1265 = _S1263;
    thread JointBond_0 _S1266 = _S1256.law_0;
    thread Measures_0 _S1267 = _S1262;
    float4 _S1268 = failure_indices_0(&_S1265, &_S1266, &_S1267, _S1264);
    if((max(max(_S1268.x, _S1268.y), max(_S1268.z, _S1268.w))) >= 1.0f)
    {
        uint _S1269 = w_alive_0(kernelContext_168);
        *(kernelContext_168->work_0+(_S1269 + i_12)) = 0U;
        uint _S1270 = a_dirty_0(kernelContext_168);
        uint _S1271 = atomic_fetch_or_explicit(kernelContext_168->atom_0+(_S1270 + s_20), 1U, memory_order_relaxed);
        uint _S1272 = a_counters_1(kernelContext_168);
        uint _S1273 = atomic_fetch_add_explicit(kernelContext_168->atom_0+_S1272, 1U, memory_order_relaxed);
        uint _S1274 = atomic_fetch_add_explicit(kernelContext_168->atom_0+(_S1272 + 1U), 1U, memory_order_relaxed);
        uint _S1275 = atomic_fetch_add_explicit(kernelContext_168->atom_0+(_S1272 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_solve(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_69 [[buffer(0)]], uint device* work_69 [[buffer(4)]], packed_float4 device* frames_69 [[buffer(6)]], packed_float4 device* bodies_69 [[buffer(7)]], packed_float4 device* chunks_69 [[buffer(1)]], uint device* sort_69 [[buffer(11)]], atomic<uint> device* atom_69 [[buffer(5)]], Record_natural_0 device* manifolds_69 [[buffer(10)]], packed_float4 device* loads_69 [[buffer(12)]], packed_uint4 device* bonds_69 [[buffer(2)]], uint device* args_69 [[buffer(13)]], uint device* info_69 [[buffer(15)]], ScanParams_0 constant* scan_69 [[buffer(14)]], packed_float4 device* parts_69 [[buffer(3)]], packed_float4 device* colliders_69 [[buffer(8)]], packed_uint4 device* collider_info_69 [[buffer(9)]], packed_float4 device* view_chunks_69 [[buffer(17)]], ViewParams_0 constant* view_69 [[buffer(16)]], packed_float4 device* objects_69 [[buffer(18)]], uint device* shot_bodies_69 [[buffer(20)]], float device* lines_69 [[buffer(19)]], packed_float4 device* geometry_69 [[buffer(21)]], packed_float4 device* shots_69 [[buffer(22)]], packed_float4 device* lp_69 [[buffer(24)]], packed_float4 device* ext_71 [[buffer(23)]], packed_float4 device* lp_sides_69 [[buffer(28)]], uint device* engine_counters_69 [[buffer(26)]], packed_float4 device* engine_contacts_69 [[buffer(25)]], packed_float4 device* cs_anchors_69 [[buffer(27)]])
{
    bool _S1276;
    uint j3_0;
    thread KernelContext_0 kernelContext_169;
    (&kernelContext_169)->params_0 = params_69;
    (&kernelContext_169)->work_0 = work_69;
    (&kernelContext_169)->frames_0 = frames_69;
    (&kernelContext_169)->bodies_0 = bodies_69;
    (&kernelContext_169)->chunks_0 = chunks_69;
    (&kernelContext_169)->sort_0 = sort_69;
    (&kernelContext_169)->atom_0 = atom_69;
    (&kernelContext_169)->manifolds_0 = manifolds_69;
    (&kernelContext_169)->loads_0 = loads_69;
    (&kernelContext_169)->bonds_0 = bonds_69;
    (&kernelContext_169)->args_0 = args_69;
    (&kernelContext_169)->info_0 = info_69;
    (&kernelContext_169)->scan_0 = scan_69;
    (&kernelContext_169)->parts_0 = parts_69;
    (&kernelContext_169)->colliders_0 = colliders_69;
    (&kernelContext_169)->collider_info_0 = collider_info_69;
    (&kernelContext_169)->view_chunks_0 = view_chunks_69;
    (&kernelContext_169)->view_0 = view_69;
    (&kernelContext_169)->objects_0 = objects_69;
    (&kernelContext_169)->shot_bodies_0 = shot_bodies_69;
    (&kernelContext_169)->lines_0 = lines_69;
    (&kernelContext_169)->geometry_0 = geometry_69;
    (&kernelContext_169)->shots_0 = shots_69;
    (&kernelContext_169)->lp_0 = lp_69;
    (&kernelContext_169)->ext_0 = ext_71;
    (&kernelContext_169)->lp_sides_0 = lp_sides_69;
    (&kernelContext_169)->engine_counters_0 = engine_counters_69;
    (&kernelContext_169)->engine_contacts_0 = engine_contacts_69;
    (&kernelContext_169)->cs_anchors_0 = cs_anchors_69;
    threadgroup array<uint, int(1024)> g_scan_69;
    (&kernelContext_169)->g_scan_0 = &g_scan_69;
    uint _S1277 = w_cs_bodies_0(&kernelContext_169);
    uint device* _S1278 = (&kernelContext_169)->work_0+(_S1277 + group_1.x);
    uint s_21 = *_S1278;
    uint t_9 = thread_2.x;
    uint _S1279 = a_cs_0(*_S1278, &kernelContext_169);
    uint _S1280 = atomic_load_explicit((&kernelContext_169)->atom_0+(_S1279 + 6U), memory_order_relaxed);
    uint _S1281 = atomic_load_explicit((&kernelContext_169)->atom_0+(_S1279 + 5U), memory_order_relaxed);
    uint j_8 = t_9;
    for(;;)
    {
        if(j_8 < _S1281)
        {
        }
        else
        {
            break;
        }
        uint _S1282 = w_cs_list_0(&kernelContext_169);
        cs_load_0(*((&kernelContext_169)->work_0+(_S1282 + _S1280 + j_8)), s_21, &kernelContext_169);
        j_8 = j_8 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint j2_0 = t_9;
    for(;;)
    {
        if(j2_0 < _S1281)
        {
        }
        else
        {
            break;
        }
        uint _S1283 = w_cs_list_0(&kernelContext_169);
        uint device* _S1284 = (&kernelContext_169)->work_0+(_S1283 + _S1280 + j2_0);
        uint k_46 = *_S1284;
        uint _S1285 = st_f_0(*_S1284, 14U);
        if(((float4(*((&kernelContext_169)->lp_0+_S1285)) ).w) == 0.0f)
        {
            bool _S1286 = lp_factor_0(k_46, &kernelContext_169);
            _S1276 = !_S1286;
        }
        else
        {
            _S1276 = false;
        }
        if(_S1276)
        {
            (*((&kernelContext_169)->lp_0+_S1285))[int(3)] = 1.0f;
            uint _S1287 = a_counters_1(&kernelContext_169);
            uint _S1288 = atomic_fetch_or_explicit((&kernelContext_169)->atom_0+(_S1287 + 3U), 32U, memory_order_relaxed);
        }
        j2_0 = j2_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint sweep_0 = 0U;
    for(;;)
    {
        if(sweep_0 < ((&kernelContext_169)->params_0->cs_sweeps_0))
        {
        }
        else
        {
            break;
        }
        j3_0 = t_9;
        for(;;)
        {
            if(j3_0 < _S1281)
            {
            }
            else
            {
                break;
            }
            uint _S1289 = w_cs_list_0(&kernelContext_169);
            st_sweep_0(*((&kernelContext_169)->work_0+(_S1289 + _S1280 + j3_0)), 14U, sweep_0 % 2U, (&kernelContext_169)->params_0->cs_c1_0, (&kernelContext_169)->params_0->cs_c2_0, &kernelContext_169);
            j3_0 = j3_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        sweep_0 = sweep_0 + 1U;
    }
    j3_0 = t_9;
    for(;;)
    {
        if(j3_0 < _S1281)
        {
        }
        else
        {
            break;
        }
        uint _S1290 = w_cs_list_0(&kernelContext_169);
        uint device* _S1291 = (&kernelContext_169)->work_0+(_S1290 + _S1280 + j3_0);
        uint k_47 = *_S1291;
        float4 _S1292 = float4(*((&kernelContext_169)->ext_0+((&kernelContext_169)->params_0->ext_chunks_0 + 4U * *_S1291))) ;
        uint _S1293 = (as_type<uint>((_S1292.x)));
        uint _S1294 = (as_type<uint>((_S1292.y)));
        uint e_6 = 0U;
        for(;;)
        {
            if(e_6 < _S1294)
            {
            }
            else
            {
                break;
            }
            uint _S1295 = adj_bond_0(_S1293 + e_6, &kernelContext_169);
            if(((uint4(*((&kernelContext_169)->bonds_0+_S1295)) ).x) == k_47)
            {
                uint _S1296 = w_alive_0(&kernelContext_169);
                _S1276 = (*((&kernelContext_169)->work_0+(_S1296 + _S1295))) != 0U;
            }
            else
            {
                _S1276 = false;
            }
            if(_S1276)
            {
                cs_break_0(_S1295, s_21, &kernelContext_169);
            }
            e_6 = e_6 + 1U;
        }
        j3_0 = j3_0 + 256U;
    }
    return;
}

