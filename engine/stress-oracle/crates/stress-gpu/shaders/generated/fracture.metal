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
    return 22U * kernelContext_0->params_0->chunk_count_0 + 6U * b_0;
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
    (&s_0)->bound_0 = _S12;
    (&s_0)->radius_0 = _S12;
    (&s_0)->mass_0 = _S11.z;
    (&s_0)->m_eff_0 = _S10.x;
    (&s_0)->modulus_0 = _S10.z;
    (&s_0)->launch_0 = 0U;
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

float cs_angular_0(uint b_15, float3 r_2, float3 u_0, KernelContext_0 thread* kernelContext_25)
{
    uint at_0 = 7U * b_15;
    thread Quat_0 _S35 = quat_of_0(float4(*(kernelContext_25->bodies_0+(at_0 + 1U))) );
    Quat_0 _S36 = conj_0(&_S35);
    float3 _S37 = cross(r_2, u_0);
    thread Quat_0 _S38 = _S36;
    float3 _S39 = rotate_0(&_S38, _S37);
    return dot(_S39 * (float4(*(kernelContext_25->bodies_0+(at_0 + 2U))) ).xyz, _S39);
}

bool cs_side_0(uint r_3, uint side_0, uint thread* k_1, uint thread* s_2, uint thread* ob_0, uint thread* co_0, float3 thread* p_0, float3 thread* u_1, float thread* v_1, float thread* m_eff_1, float thread* t_in_0, KernelContext_0 thread* kernelContext_26)
{
    *t_in_0 = 0.0f;
    *k_1 = 4294967295U;
    *s_2 = 4294967295U;
    *ob_0 = 4294967295U;
    *co_0 = 4294967295U;
    float3 _S40 = float3(0.0f) ;
    *p_0 = _S40;
    *u_1 = _S40;
    *v_1 = 0.0f;
    *m_eff_1 = 0.0f;
    uint _S41 = cs_row_0(r_3, kernelContext_26);
    uint4 ids_1 = (as_type<uint4>((float4(*(kernelContext_26->engine_contacts_0+_S41)) )));
    if((ids_1.w) == 0U)
    {
        return false;
    }
    bool _S42 = side_0 == 0U;
    uint cm_0;
    if(_S42)
    {
        cm_0 = ids_1.x;
    }
    else
    {
        cm_0 = ids_1.y;
    }
    uint _S43;
    if(_S42)
    {
        _S43 = ids_1.y;
    }
    else
    {
        _S43 = ids_1.x;
    }
    *co_0 = _S43;
    uint _S44 = (uint4(*(kernelContext_26->collider_info_0+2U * cm_0)) ).x;
    *k_1 = _S44;
    bool _S45;
    if(_S44 == 4294967295U)
    {
        _S45 = true;
    }
    else
    {
        _S45 = (*k_1) >= (kernelContext_26->params_0->chunk_count_0);
    }
    if(_S45)
    {
        _S45 = true;
    }
    else
    {
        _S45 = ((uint4(*(kernelContext_26->collider_info_0+2U * *co_0)) ).z) != 2U;
    }
    if(_S45)
    {
        return false;
    }
    uint _S46 = body_of_0(*k_1, kernelContext_26);
    *s_2 = _S46;
    uint _S47 = (as_type<uint>(((float4(*(kernelContext_26->colliders_0+4U * *co_0)) ).x)));
    *ob_0 = _S47;
    if(_S47 == (*s_2))
    {
        _S45 = true;
    }
    else
    {
        _S45 = (*ob_0) >= (kernelContext_26->params_0->max_bodies_0);
    }
    if(_S45)
    {
        return false;
    }
    uint bs_0 = 7U * *s_2;
    uint bo_0 = 7U * *ob_0;
    float inv_o_0 = (float4(*(kernelContext_26->bodies_0+bo_0)) ).w;
    if(!(inv_o_0 > 0.0f))
    {
        return false;
    }
    float3 n_1 = (float4(*(kernelContext_26->engine_contacts_0+(_S41 + 1U))) ).xyz;
    float3 _S48;
    if(_S42)
    {
        _S48 = - n_1;
    }
    else
    {
        _S48 = n_1;
    }
    *u_1 = _S48;
    Quat_0 _S49 = quat_of_0(float4(*(kernelContext_26->bodies_0+(bs_0 + 1U))) );
    if(_S42)
    {
        _S48 = (float4(*(kernelContext_26->engine_contacts_0+(_S41 + 2U))) ).xyz;
    }
    else
    {
        _S48 = (float4(*(kernelContext_26->engine_contacts_0+(_S41 + 3U))) ).xyz;
    }
    thread Quat_0 _S50 = _S49;
    float3 _S51 = rotate_0(&_S50, _S48);
    Quat_0 _S52 = quat_of_0(float4(*(kernelContext_26->bodies_0+(bo_0 + 1U))) );
    if(_S42)
    {
        _S48 = (float4(*(kernelContext_26->engine_contacts_0+(_S41 + 3U))) ).xyz;
    }
    else
    {
        _S48 = (float4(*(kernelContext_26->engine_contacts_0+(_S41 + 2U))) ).xyz;
    }
    thread Quat_0 _S53 = _S52;
    float3 _S54 = rotate_0(&_S53, _S48);
    float3 _S55 = (float4(*(kernelContext_26->bodies_0+bs_0)) ).xyz + _S51;
    *p_0 = _S55;
    float gap_0 = dot(_S55 - ((float4(*(kernelContext_26->bodies_0+bo_0)) ).xyz + _S54), *u_1);
    float _S56 = dot((float4(*(kernelContext_26->bodies_0+(bo_0 + 3U))) ).xyz + cross((float4(*(kernelContext_26->bodies_0+(bo_0 + 4U))) ).xyz, _S54) - ((float4(*(kernelContext_26->bodies_0+(bs_0 + 3U))) ).xyz + cross((float4(*(kernelContext_26->bodies_0+(bs_0 + 4U))) ).xyz, _S51)), *u_1);
    *v_1 = _S56;
    if(!(_S56 > (kernelContext_26->params_0->strike_speed_0)))
    {
        return false;
    }
    float _S57 = (float4(*(kernelContext_26->bodies_0+bs_0)) ).w;
    float _S58 = cs_angular_0(*s_2, _S51, *u_1, kernelContext_26);
    float _S59 = _S57 + _S58 + inv_o_0;
    float _S60 = cs_angular_0(*ob_0, _S54, *u_1, kernelContext_26);
    float w_1 = _S59 + _S60;
    if(!(w_1 > 0.0f))
    {
        return false;
    }
    *m_eff_1 = 1.0f / w_1;
    *t_in_0 = clamp(gap_0 / *v_1, 0.0f, kernelContext_26->params_0->dt_0);
    return true;
}

uint3 cs_identity_0(uint r_4, uint side_1, KernelContext_0 thread* kernelContext_27)
{
    uint _S61 = cs_row_0(r_4, kernelContext_27);
    uint4 ids_2 = (as_type<uint4>((float4(*(kernelContext_27->engine_contacts_0+_S61)) )));
    uint3 _S62;
    if(side_1 == 0U)
    {
        _S62 = uint3(ids_2.x, ids_2.y, 2U * ids_2.z);
    }
    else
    {
        _S62 = uint3(ids_2.y, ids_2.x, 2U * ids_2.z + 1U);
    }
    return _S62;
}

void cs_pick_0(uint r_5, KernelContext_0 thread* kernelContext_28)
{
    uint _S63 = cs_rows_0(kernelContext_28);
    if(r_5 >= _S63)
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
        bool _S64 = cs_side_0(r_5, side_2, &k_2, &s_3, &ob_1, &co_1, &p_1, &u_2, &v_2, &m_eff_2, &t_in_1, kernelContext_28);
        bool _S65;
        if(!_S64)
        {
            _S65 = true;
        }
        else
        {
            uint _S66 = (as_type<uint>((m_eff_2 * v_2)));
            uint _S67 = a_cs_0(s_3, kernelContext_28);
            uint _S68 = atomic_load_explicit(kernelContext_28->atom_0+_S67, memory_order_relaxed);
            _S65 = _S66 != _S68;
        }
        if(_S65)
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint3 _S69 = cs_identity_0(r_5, side_2, kernelContext_28);
        uint _S70 = a_cs_0(s_3, kernelContext_28);
        uint _S71 = atomic_load_explicit(kernelContext_28->atom_0+(_S70 + 2U), memory_order_relaxed);
        if((~_S71) != (_S69.x))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint _S72 = atomic_load_explicit(kernelContext_28->atom_0+(_S70 + 9U), memory_order_relaxed);
        if((~_S72) != (_S69.y))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        uint _S73 = atomic_load_explicit(kernelContext_28->atom_0+(_S70 + 10U), memory_order_relaxed);
        if((~_S73) != (_S69.z))
        {
            side_2 = side_2 + 1U;
            continue;
        }
        atomic_store_explicit(kernelContext_28->atom_0+(_S70 + 11U), 2U * r_5 + side_2 + 1U, memory_order_relaxed);
        side_2 = side_2 + 1U;
    }
    return;
}

void cs_pick_1(uint r_6, KernelContext_0 thread* kernelContext_29)
{
    uint _S74 = cs_rows_0(kernelContext_29);
    if(r_6 >= _S74)
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
        bool _S75 = cs_side_0(r_6, side_3, &k_3, &s_4, &ob_2, &co_2, &p_2, &u_3, &v_3, &m_eff_3, &t_in_2, kernelContext_29);
        bool _S76;
        if(!_S75)
        {
            _S76 = true;
        }
        else
        {
            uint _S77 = (as_type<uint>((m_eff_3 * v_3)));
            uint _S78 = a_cs_0(s_4, kernelContext_29);
            uint _S79 = atomic_load_explicit(kernelContext_29->atom_0+_S78, memory_order_relaxed);
            _S76 = _S77 != _S79;
        }
        if(_S76)
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint3 _S80 = cs_identity_0(r_6, side_3, kernelContext_29);
        uint _S81 = a_cs_0(s_4, kernelContext_29);
        uint _S82 = atomic_load_explicit(kernelContext_29->atom_0+(_S81 + 2U), memory_order_relaxed);
        if((~_S82) != (_S80.x))
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint _S83 = atomic_load_explicit(kernelContext_29->atom_0+(_S81 + 9U), memory_order_relaxed);
        if((~_S83) != (_S80.y))
        {
            side_3 = side_3 + 1U;
            continue;
        }
        uint _S84 = atomic_fetch_max_explicit(kernelContext_29->atom_0+(_S81 + 10U), ~_S80.z, memory_order_relaxed);
        side_3 = side_3 + 1U;
    }
    return;
}

void cs_pick_2(uint r_7, KernelContext_0 thread* kernelContext_30)
{
    uint _S85 = cs_rows_0(kernelContext_30);
    if(r_7 >= _S85)
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
        bool _S86 = cs_side_0(r_7, side_4, &k_4, &s_5, &ob_3, &co_3, &p_3, &u_4, &v_4, &m_eff_4, &t_in_3, kernelContext_30);
        bool _S87;
        if(!_S86)
        {
            _S87 = true;
        }
        else
        {
            uint _S88 = (as_type<uint>((m_eff_4 * v_4)));
            uint _S89 = a_cs_0(s_5, kernelContext_30);
            uint _S90 = atomic_load_explicit(kernelContext_30->atom_0+_S89, memory_order_relaxed);
            _S87 = _S88 != _S90;
        }
        if(_S87)
        {
            side_4 = side_4 + 1U;
            continue;
        }
        uint3 _S91 = cs_identity_0(r_7, side_4, kernelContext_30);
        uint _S92 = a_cs_0(s_5, kernelContext_30);
        uint _S93 = atomic_load_explicit(kernelContext_30->atom_0+(_S92 + 2U), memory_order_relaxed);
        if((~_S93) != (_S91.x))
        {
            side_4 = side_4 + 1U;
            continue;
        }
        uint _S94 = atomic_fetch_max_explicit(kernelContext_30->atom_0+(_S92 + 9U), ~_S91.y, memory_order_relaxed);
        side_4 = side_4 + 1U;
    }
    return;
}

void cs_pick_3(uint r_8, KernelContext_0 thread* kernelContext_31)
{
    uint _S95 = cs_rows_0(kernelContext_31);
    if(r_8 >= _S95)
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
        bool _S96 = cs_side_0(r_8, side_5, &k_5, &s_6, &ob_4, &co_4, &p_4, &u_5, &v_5, &m_eff_5, &t_in_4, kernelContext_31);
        bool _S97;
        if(!_S96)
        {
            _S97 = true;
        }
        else
        {
            uint _S98 = (as_type<uint>((m_eff_5 * v_5)));
            uint _S99 = a_cs_0(s_6, kernelContext_31);
            uint _S100 = atomic_load_explicit(kernelContext_31->atom_0+_S99, memory_order_relaxed);
            _S97 = _S98 != _S100;
        }
        if(_S97)
        {
            side_5 = side_5 + 1U;
            continue;
        }
        uint3 _S101 = cs_identity_0(r_8, side_5, kernelContext_31);
        uint _S102 = a_cs_0(s_6, kernelContext_31);
        uint _S103 = atomic_fetch_max_explicit(kernelContext_31->atom_0+(_S102 + 2U), ~_S101.x, memory_order_relaxed);
        side_5 = side_5 + 1U;
    }
    return;
}

uint w_alive_0(KernelContext_0 thread* kernelContext_32)
{
    return kernelContext_32->params_0->chunk_count_0;
}

struct IlPlace_0
{
    float along_0;
    float radial_0;
    bool entry_0;
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
    thread Quat_0 _S104 = axes_0;
    Quat_0 _S105 = quat_mul_0(&_S104, q_4);
    thread Quat_0 _S106 = _S105;
    Quat_0 _S107 = normalized_0(&_S106);
    return _S107;
}

float3 from_rubble_0(float3 v_6)
{
    return float3(v_6.x, - v_6.z, v_6.y);
}

void structure_pose_0(uint slot_0, uint b_16, Quat_0 thread* q_5, float3 thread* origin_0, KernelContext_0 thread* kernelContext_33)
{
    uint _S108 = 2U * slot_0;
    thread Quat_0 _S109 = quat_of_0(float4(*(kernelContext_33->frames_0+(_S108 + 1U))) );
    Quat_0 _S110 = conj_0(&_S109);
    float3 com_0 = (float4(*(kernelContext_33->frames_0+_S108)) ).xyz;
    thread Quat_0 _S111 = quat_of_0(float4(*(kernelContext_33->bodies_0+(b_16 + 1U))) );
    Quat_0 _S112 = rot_from_rubble_0(&_S111);
    thread Quat_0 _S113 = _S112;
    thread Quat_0 _S114 = _S110;
    Quat_0 _S115 = quat_mul_0(&_S113, &_S114);
    thread Quat_0 _S116 = _S115;
    Quat_0 _S117 = normalized_0(&_S116);
    *q_5 = _S117;
    float3 _S118 = from_rubble_0((float4(*(kernelContext_33->bodies_0+b_16)) ).xyz);
    thread Quat_0 _S119 = _S117;
    float3 _S120 = rotate_0(&_S119, com_0);
    *origin_0 = _S118 - _S120;
    return;
}

IlPlace_0 il_place_0(uint k_6, uint entry_1, float3 contact_2, float3 d_2, KernelContext_0 thread* kernelContext_34)
{
    uint _S121 = body_of_0(k_6, kernelContext_34);
    thread Quat_0 q_6;
    thread float3 origin_1;
    structure_pose_0(_S121, 7U * _S121, &q_6, &origin_1, kernelContext_34);
    float3 _S122 = (float4(*(kernelContext_34->chunks_0+4U * k_6)) ).xyz;
    thread Quat_0 _S123 = q_6;
    float3 _S124 = rotate_0(&_S123, _S122);
    float3 rel_1 = origin_1 + _S124 - contact_2;
    thread IlPlace_0 p_5;
    float _S125 = dot(rel_1, d_2);
    (&p_5)->along_0 = _S125;
    (&p_5)->radial_0 = length(rel_1 - d_2 * float3(_S125) );
    (&p_5)->entry_0 = k_6 == entry_1;
    return p_5;
}

bool il_plug_0(uint outcome_0, float r_11, float h_0, const IlPlace_0 thread* p_6)
{
    bool _S126;
    if((outcome_0 & 8U) != 0U)
    {
        if(p_6->entry_0)
        {
            _S126 = true;
        }
        else
        {
            float _S127 = p_6->along_0;
            if((p_6->along_0) >= 0.0f)
            {
                _S126 = _S127 <= h_0;
            }
            else
            {
                _S126 = false;
            }
            if(_S126)
            {
                _S126 = (p_6->radial_0) <= (r_11 + _S127);
            }
            else
            {
                _S126 = false;
            }
        }
    }
    else
    {
        _S126 = false;
    }
    return _S126;
}

bool il_scab_0(uint outcome_1, float r_12, float h_1, float scab_0, const IlPlace_0 thread* p_7)
{
    bool _S128;
    if((outcome_1 & 4U) != 0U)
    {
        _S128 = (p_7->along_0) >= (h_1 - scab_0);
    }
    else
    {
        _S128 = false;
    }
    if(_S128)
    {
        _S128 = (p_7->along_0) <= h_1;
    }
    else
    {
        _S128 = false;
    }
    if(_S128)
    {
        _S128 = (p_7->radial_0) <= (r_12 + h_1);
    }
    else
    {
        _S128 = false;
    }
    return _S128;
}

bool il_crater_0(uint outcome_2, float r_13, float depth_0, float a_1, const IlPlace_0 thread* p_8)
{
    bool _S129;
    if((outcome_2 & 2U) != 0U)
    {
        _S129 = (p_8->along_0) >= 0.0f;
    }
    else
    {
        _S129 = false;
    }
    if(_S129)
    {
        _S129 = (p_8->along_0) <= depth_0;
    }
    else
    {
        _S129 = false;
    }
    if(_S129)
    {
        _S129 = (p_8->radial_0) <= (max(r_13, a_1));
    }
    else
    {
        _S129 = false;
    }
    return _S129;
}

uint a_counters_1(KernelContext_0 thread* kernelContext_35)
{
    uint _S130 = a_best_root_0(kernelContext_35);
    return _S130 + kernelContext_35->params_0->body_count_0;
}

void il_breaks_0(uint i_0, KernelContext_0 thread* kernelContext_36)
{
    bool cut_0;
    if(i_0 >= (kernelContext_36->params_0->bond_count_0))
    {
        cut_0 = true;
    }
    else
    {
        uint _S131 = w_alive_0(kernelContext_36);
        cut_0 = (*(kernelContext_36->work_0+(_S131 + i_0))) == 0U;
    }
    if(cut_0)
    {
        return;
    }
    uint4 _S132 = uint4(*(kernelContext_36->bonds_0+i_0)) ;
    uint _S133 = _S132.x;
    uint _S134 = body_of_0(_S133, kernelContext_36);
    bool _S135 = il_struck_0(_S134, kernelContext_36);
    if(!_S135)
    {
        return;
    }
    uint _S136 = lp_body_0(_S134, kernelContext_36);
    float4 _S137 = float4(*(kernelContext_36->lp_0+_S136)) ;
    float4 _S138 = float4(*(kernelContext_36->lp_0+(_S136 + 1U))) ;
    uint outcome_3 = (as_type<uint>((_S137.w)));
    if((outcome_3 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_3;
    thread float3 d_3;
    thread float speed_2;
    il_path_0(_S134, &contact_3, &d_3, &speed_2, kernelContext_36);
    uint _S139 = il_entry_0(_S134, kernelContext_36);
    IlStriker_0 _S140 = il_striker_0(_S134, kernelContext_36);
    float _S141 = max(_S140.radius_0, 0.00100000004749745f);
    float h_2 = _S137.x;
    IlPlace_0 _S142 = il_place_0(_S133, _S139, contact_3, d_3, kernelContext_36);
    IlPlace_0 _S143 = il_place_0(_S132.y, _S139, contact_3, d_3, kernelContext_36);
    thread IlPlace_0 _S144 = _S142;
    bool _S145 = il_plug_0(outcome_3, _S141, h_2, &_S144);
    thread IlPlace_0 _S146 = _S143;
    bool _S147 = il_plug_0(outcome_3, _S141, h_2, &_S146);
    if(_S145 != _S147)
    {
        cut_0 = true;
    }
    else
    {
        float _S148 = _S138.y;
        thread IlPlace_0 _S149 = _S142;
        bool _S150 = il_scab_0(outcome_3, _S141, h_2, _S148, &_S149);
        thread IlPlace_0 _S151 = _S143;
        bool _S152 = il_scab_0(outcome_3, _S141, h_2, _S148, &_S151);
        cut_0 = _S150 != _S152;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S153 = _S138.x;
        float _S154 = _S138.z;
        thread IlPlace_0 _S155 = _S142;
        bool _S156 = il_crater_0(outcome_3, _S141, _S153, _S154, &_S155);
        cut_0 = _S156;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S157 = _S138.x;
        float _S158 = _S138.z;
        thread IlPlace_0 _S159 = _S143;
        bool _S160 = il_crater_0(outcome_3, _S141, _S157, _S158, &_S159);
        cut_0 = _S160;
    }
    if(cut_0)
    {
        uint _S161 = w_alive_0(kernelContext_36);
        *(kernelContext_36->work_0+(_S161 + i_0)) = 0U;
        uint _S162 = a_dirty_0(kernelContext_36);
        uint _S163 = atomic_fetch_or_explicit(kernelContext_36->atom_0+(_S162 + _S134), 1U, memory_order_relaxed);
        uint _S164 = a_counters_1(kernelContext_36);
        uint _S165 = atomic_fetch_add_explicit(kernelContext_36->atom_0+_S164, 1U, memory_order_relaxed);
        uint _S166 = atomic_fetch_add_explicit(kernelContext_36->atom_0+(_S164 + 1U), 1U, memory_order_relaxed);
        uint _S167 = atomic_fetch_add_explicit(kernelContext_36->atom_0+(_S164 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

uint w_label_0(KernelContext_0 thread* kernelContext_37)
{
    uint _S168 = w_alive_0(kernelContext_37);
    return _S168 + kernelContext_37->params_0->bond_count_0;
}

uint w_slot_0(KernelContext_0 thread* kernelContext_38)
{
    uint _S169 = w_label_0(kernelContext_38);
    return _S169 + kernelContext_38->params_0->chunk_count_0;
}

uint w_new_0(KernelContext_0 thread* kernelContext_39)
{
    uint _S170 = w_slot_0(kernelContext_39);
    return _S170 + kernelContext_39->params_0->chunk_count_0;
}

uint w_rank_0(KernelContext_0 thread* kernelContext_40)
{
    uint _S171 = w_new_0(kernelContext_40);
    return _S171 + kernelContext_40->params_0->chunk_count_0;
}

uint w_free_0(KernelContext_0 thread* kernelContext_41)
{
    uint _S172 = w_rank_0(kernelContext_41);
    return _S172 + kernelContext_41->params_0->chunk_count_0;
}

uint w_free_rank_0(KernelContext_0 thread* kernelContext_42)
{
    uint _S173 = w_free_0(kernelContext_42);
    return _S173 + kernelContext_42->params_0->body_count_0;
}

uint w_free_list_0(KernelContext_0 thread* kernelContext_43)
{
    uint _S174 = w_free_rank_0(kernelContext_43);
    return _S174 + kernelContext_43->params_0->body_count_0;
}

uint w_pulses_0(KernelContext_0 thread* kernelContext_44)
{
    uint _S175 = w_free_list_0(kernelContext_44);
    return _S175 + kernelContext_44->params_0->body_count_0 + 2050U;
}

uint w_marks_0(KernelContext_0 thread* kernelContext_45)
{
    uint _S176 = w_pulses_0(kernelContext_45);
    return _S176 + 4U * kernelContext_45->params_0->chunk_count_0;
}

uint w_carry_0(KernelContext_0 thread* kernelContext_46)
{
    uint _S177 = w_marks_0(kernelContext_46);
    return _S177 + kernelContext_46->params_0->chunk_count_0;
}

float3 to_rubble_0(float3 v_7)
{
    return float3(v_7.x, v_7.z, - v_7.y);
}

uint a_il_0(uint b_17, KernelContext_0 thread* kernelContext_47)
{
    uint _S178 = a_strike_0(kernelContext_47);
    return _S178 + 23U * kernelContext_47->params_0->body_count_0 + 4U * b_17;
}

float read64_0(uint at_1, float scale_0, KernelContext_0 thread* kernelContext_48)
{
    uint _S179 = atomic_load_explicit(kernelContext_48->atom_0+(at_1 + 1U), memory_order_relaxed);
    int hi_0 = (as_type<int>((_S179)));
    uint lo_0 = atomic_load_explicit(kernelContext_48->atom_0+at_1, memory_order_relaxed);
    return (float(hi_0) * 4.294967296e+09f + float(lo_0)) / scale_0;
}

void il_carry_0(uint k_7, KernelContext_0 thread* kernelContext_49)
{
    if(k_7 >= (kernelContext_49->params_0->chunk_count_0))
    {
        return;
    }
    uint _S180 = w_carry_0(kernelContext_49);
    uint _S181 = _S180 + k_7;
    uint device* _S182 = kernelContext_49->work_0+_S181;
    uint c_2 = *_S182;
    if((*_S182) == 0U)
    {
        return;
    }
    *(kernelContext_49->work_0+_S181) = 0U;
    uint b0_0 = c_2 - 1U;
    bool _S183 = il_struck_0(b0_0, kernelContext_49);
    if(!_S183)
    {
        return;
    }
    thread float3 contact_4;
    thread float3 d_4;
    thread float speed_3;
    il_path_0(b0_0, &contact_4, &d_4, &speed_3, kernelContext_49);
    float3 dr_0 = to_rubble_0(d_4);
    IlStriker_0 _S184 = il_striker_0(b0_0, kernelContext_49);
    uint _S185 = a_il_0(b0_0, kernelContext_49);
    float _S186 = read64_0(_S185 + 2U, kernelContext_49->params_0->scale_m_0, kernelContext_49);
    float3 v_wall_0 = (float4(*(kernelContext_49->bodies_0+(7U * b0_0 + 3U))) ).xyz;
    uint _S187 = 7U * _S184.body_0;
    float3 v_shot_0 = (float4(*(kernelContext_49->bodies_0+(_S187 + 3U))) ).xyz;
    float closing_0 = dot(v_shot_0 - v_wall_0, dr_0);
    float v_r_0 = _S184.mass_0 * closing_0 / (_S184.mass_0 + _S186);
    float3 _S188 = v_shot_0 + dr_0 * float3((v_r_0 - closing_0)) ;
    float3 _S189 = v_wall_0 + dr_0 * float3(v_r_0) ;
    float _S190 = il_t_in_0(b0_0, kernelContext_49);
    float3 _S191 = (float4(*(kernelContext_49->bodies_0+_S187)) ).xyz + dr_0 * float3(((closing_0 - v_r_0) * clamp(_S190, 0.0f, kernelContext_49->params_0->dt_0))) ;
    uint _S192 = body_of_0(k_7, kernelContext_49);
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
        uint s_at_0 = 7U * (_S184.body_0 + region_0 * kernelContext_49->params_0->max_bodies_0);
        (*(kernelContext_49->bodies_0+s_at_0)).xyz = _S191;
        (*(kernelContext_49->bodies_0+(s_at_0 + 3U))).xyz = _S188;
        (*(kernelContext_49->bodies_0+(7U * (_S192 + region_0 * kernelContext_49->params_0->max_bodies_0) + 3U))).xyz = _S189;
        region_0 = region_0 + 1U;
    }
    return;
}

void add64_0(uint at_2, float value_0, float scale_1, KernelContext_0 thread* kernelContext_50)
{
    float x_1 = round(value_0 * scale_1);
    float x_2;
    if(!((abs(x_1)) < 1.073741824e+09f))
    {
        uint _S193 = a_counters_1(kernelContext_50);
        uint _S194 = atomic_fetch_or_explicit(kernelContext_50->atom_0+(_S193 + 3U), 2U, memory_order_relaxed);
        x_2 = 0.0f;
    }
    else
    {
        x_2 = x_1;
    }
    int v_8 = int(x_2);
    uint lo_1 = (as_type<uint>((v_8)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_50->atom_0+at_2, lo_1, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_1) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S195;
    if(v_8 < int(0))
    {
        _S195 = 4294967295U;
    }
    else
    {
        _S195 = 0U;
    }
    uint hi_1 = _S195 + carry_0;
    if(hi_1 != 0U)
    {
        uint _S196 = atomic_fetch_add_explicit(kernelContext_50->atom_0+(at_2 + 1U), hi_1, memory_order_relaxed);
    }
    return;
}

void il_mark_0(uint k_8, KernelContext_0 thread* kernelContext_51)
{
    if(k_8 >= (kernelContext_51->params_0->chunk_count_0))
    {
        return;
    }
    uint _S197 = body_of_0(k_8, kernelContext_51);
    bool _S198 = il_struck_0(_S197, kernelContext_51);
    if(!_S198)
    {
        return;
    }
    uint _S199 = lp_body_0(_S197, kernelContext_51);
    float4 _S200 = float4(*(kernelContext_51->lp_0+_S199)) ;
    uint outcome_4 = (as_type<uint>((_S200.w)));
    if((outcome_4 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_5;
    thread float3 d_5;
    thread float speed_4;
    il_path_0(_S197, &contact_5, &d_5, &speed_4, kernelContext_51);
    uint _S201 = il_entry_0(_S197, kernelContext_51);
    IlStriker_0 _S202 = il_striker_0(_S197, kernelContext_51);
    float _S203 = max(_S202.radius_0, 0.00100000004749745f);
    IlPlace_0 _S204 = il_place_0(k_8, _S201, contact_5, d_5, kernelContext_51);
    float _S205 = _S200.x;
    thread IlPlace_0 _S206 = _S204;
    bool _S207 = il_plug_0(outcome_4, _S203, _S205, &_S206);
    if(_S207)
    {
        uint _S208 = w_carry_0(kernelContext_51);
        *(kernelContext_51->work_0+(_S208 + k_8)) = _S197 + 1U;
        uint _S209 = a_il_0(_S197, kernelContext_51);
        add64_0(_S209 + 2U, (float4(*(kernelContext_51->chunks_0+4U * k_8)) ).w, kernelContext_51->params_0->scale_m_0, kernelContext_51);
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
    float r_14 = abs(strain_rate_0);
    float4 _S210 = mat_0->dif_0;
    float ref_0 = mat_0->dif_0.x;
    if(r_14 <= ref_0)
    {
        return 1.0f;
    }
    float _S211 = _S210.z;
    float f_0;
    if(r_14 <= _S211)
    {
        float _S212 = (metal::fast::divide((r_14), (ref_0)));
        float _S213 = (metal::fast::pow((_S212), (_S210.y)));
        f_0 = _S213;
    }
    else
    {
        float _S214 = (metal::fast::divide((_S211), (ref_0)));
        float _S215 = (metal::fast::pow((_S214), (_S210.y)));
        float _S216 = (metal::fast::divide((r_14), (_S211)));
        float _S217 = (metal::fast::pow((_S216), (_S210.w)));
        f_0 = _S215 * _S217;
    }
    return clamp(f_0, 1.0f, mat_0->misc_0.x);
}

void il_impulse_0(uint b_18, uint sb_0, float m_shot_0, float3 contact_6, float3 d_6, float j_0, KernelContext_0 thread* kernelContext_52)
{
    float3 jr_0 = to_rubble_0(d_6) * float3(j_0) ;
    uint at_3 = 7U * b_18;
    float inv_mass_1 = (float4(*(kernelContext_52->bodies_0+at_3)) ).w;
    float3 _S218 = jr_0 * float3(inv_mass_1) ;
    float3 _S219 = float3(0.0f) ;
    float3 dw_0;
    if(inv_mass_1 > 0.0f)
    {
        Quat_0 q_7 = quat_of_0(float4(*(kernelContext_52->bodies_0+(at_3 + 1U))) );
        float3 r_15 = to_rubble_0(contact_6) - (float4(*(kernelContext_52->bodies_0+at_3)) ).xyz;
        float3 _S220 = (float4(*(kernelContext_52->bodies_0+(at_3 + 2U))) ).xyz;
        thread Quat_0 _S221 = q_7;
        Quat_0 _S222 = conj_0(&_S221);
        float3 _S223 = cross(r_15, jr_0);
        thread Quat_0 _S224 = _S222;
        float3 _S225 = rotate_0(&_S224, _S223);
        float3 _S226 = _S220 * _S225;
        thread Quat_0 _S227 = q_7;
        float3 _S228 = rotate_0(&_S227, _S226);
        dw_0 = _S228;
    }
    else
    {
        dw_0 = _S219;
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
        uint s_at_1 = 7U * (sb_0 + region_1 * kernelContext_52->params_0->max_bodies_0) + 3U;
        (*(kernelContext_52->bodies_0+s_at_1)).xyz = (float4(*(kernelContext_52->bodies_0+s_at_1)) ).xyz - jr_0 / float3(m_shot_0) ;
        uint w_at_0 = 7U * (b_18 + region_1 * kernelContext_52->params_0->max_bodies_0);
        uint _S229 = w_at_0 + 3U;
        (*(kernelContext_52->bodies_0+_S229)).xyz = (float4(*(kernelContext_52->bodies_0+_S229)) ).xyz + _S218;
        uint _S230 = w_at_0 + 4U;
        (*(kernelContext_52->bodies_0+_S230)).xyz = (float4(*(kernelContext_52->bodies_0+_S230)) ).xyz + dw_0;
        region_1 = region_1 + 1U;
    }
    return;
}

void il_strike_0(uint b_19, KernelContext_0 thread* kernelContext_53)
{
    bool damage_0;
    if(b_19 >= (kernelContext_53->params_0->body_count_0))
    {
        damage_0 = true;
    }
    else
    {
        bool _S231 = il_struck_0(b_19, kernelContext_53);
        damage_0 = !_S231;
    }
    if(damage_0)
    {
        return;
    }
    float _S232 = il_t_in_0(b_19, kernelContext_53);
    uint _S233 = il_entry_0(b_19, kernelContext_53);
    thread float3 contact_7;
    thread float3 d_7;
    thread float speed_5;
    il_path_0(b_19, &contact_7, &d_7, &speed_5, kernelContext_53);
    IlStriker_0 _S234 = il_striker_0(b_19, kernelContext_53);
    float _S235 = max(_S234.radius_0, 0.00100000004749745f);
    if((_S234.launch_0) != 0U)
    {
        uint _S236 = w_marks_0(kernelContext_53);
        damage_0 = (*(kernelContext_53->work_0+(_S236 + _S233))) == (_S234.launch_0);
    }
    else
    {
        damage_0 = false;
    }
    if(damage_0)
    {
        uint _S237 = lp_body_0(b_19, kernelContext_53);
        packed_float4 _S238 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_53->lp_0+_S237) = _S238;
        *(kernelContext_53->lp_0+(_S237 + 1U)) = _S238;
        return;
    }
    uint _S239 = a_il_0(b_19, kernelContext_53);
    uint _S240 = atomic_load_explicit(kernelContext_53->atom_0+_S239, memory_order_relaxed);
    uint _S241 = atomic_load_explicit(kernelContext_53->atom_0+(_S239 + 1U), memory_order_relaxed);
    if(_S240 != _S241)
    {
        uint _S242 = a_counters_1(kernelContext_53);
        uint _S243 = atomic_fetch_or_explicit(kernelContext_53->atom_0+(_S242 + 3U), 16U, memory_order_relaxed);
    }
    float _S244 = max((as_type<float>((_S240))) - _S235, 9.99999997475242708e-07f);
    uint o_1 = kernelContext_53->params_0->ext_chunks_0 + 4U * _S233;
    float4 _S245 = float4(*(kernelContext_53->ext_0+(o_1 + 1U))) ;
    thread JointMaterial_0 dm_0;
    float4 _S246 = float4(0.0f) ;
    (&dm_0)->strength_0 = _S246;
    (&dm_0)->energy_0 = _S246;
    (&dm_0)->dif_0 = float4(*(kernelContext_53->ext_0+(o_1 + 2U))) ;
    float4 _S247 = float4(*(kernelContext_53->ext_0+(o_1 + 3U))) ;
    float _S248 = _S247.x;
    (&dm_0)->misc_0 = float4(_S248, 0.0f, 0.0f, 0.0f);
    (&dm_0)->kind_flags_0 = uint4(0U) ;
    float nu_0 = _S247.y;
    float e_target_0 = _S247.z / (1.0f - nu_0 * nu_0);
    float e_star_0 = 1.0f / (1.0f / _S234.modulus_0 + 1.0f / e_target_0);
    float _S249 = max(speed_5, 0.00100000004749745f);
    float _S250 = sqrt(_S235);
    float h0_0 = pow(15.0f * _S234.m_eff_0 * _S249 * _S249 / (16.0f * e_star_0 * _S250), 0.40000000596046448f);
    float t_c_0 = 2.94000005722045898f * h0_0 / _S249;
    float force_0 = 1.33333337306976318f * e_star_0 * _S250 * pow(h0_0, 1.5f);
    float a_2 = sqrt(_S235 * h0_0);
    float p0_0 = 3.0f * force_0 / (6.28318548202514648f * a_2 * a_2);
    float ring_0 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    float k_9;
    if(_S248 > 0.0f)
    {
        float _S251 = _S249 / _S244;
        thread JointMaterial_0 _S252 = dm_0;
        float _S253 = dif_factor_0(&_S252, _S251);
        k_9 = _S253;
    }
    else
    {
        k_9 = 1.0f;
    }
    float ft_0 = k_9 * _S245.y;
    float fc_0 = k_9 * _S245.z;
    float tau_0 = k_9 * _S245.w;
    if(p0_0 > fc_0)
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ring_0 > ft_0;
    }
    float _S254 = fc_0 * (3.14159274101257324f * _S235 * _S235);
    float _S255 = min(_S254 * _S244, 1.57079637050628662f * (2.0f * _S235 + _S244) * _S244 * _S244 * tau_0);
    float _S256 = 2.0f * _S255;
    float v_bl_0 = sqrt(_S256 / _S234.m_eff_0);
    bool perforate_0;
    if(damage_0)
    {
        perforate_0 = _S249 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    float r2_0 = _S235 + _S244;
    float _S257 = _S245.x;
    float plug_0 = _S257 * 3.14159274101257324f * _S244 / 3.0f * (_S235 * _S235 + _S235 * r2_0 + r2_0 * r2_0);
    float v_res_0;
    if(perforate_0)
    {
        v_res_0 = _S234.mass_0 / (_S234.mass_0 + plug_0) * sqrt(max(_S249 * _S249 - v_bl_0 * v_bl_0, 0.0f));
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
        depth_1 = min(0.5f * _S234.m_eff_0 * _S249 * _S249 / _S254, _S244);
    }
    else
    {
        depth_1 = 0.0f;
    }
    bool crater_0 = depth_1 > a_2;
    float c_3 = sqrt(e_target_0 / _S257);
    float z_t_0 = _S257 * c_3;
    float rho_p_0 = _S234.mass_0 / (4.18879032135009766f * _S235 * _S235 * _S235);
    float z_p_0 = rho_p_0 * sqrt(_S234.modulus_0 / rho_p_0);
    float mean_pressure_0 = force_0 / (3.14159274101257324f * a_2 * a_2);
    float _S258 = min(min(z_t_0 * _S249 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_0);
    float source_radius_0;
    if(mean_pressure_0 > fc_0)
    {
        source_radius_0 = max(a_2, _S235);
    }
    else
    {
        source_radius_0 = a_2;
    }
    float pulse_0 = c_3 * t_c_0;
    float back_0 = _S258 * min(source_radius_0 / _S244, 1.0f) * min(2.0f * _S244 / pulse_0, 1.0f);
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
        scab_1 = 0.5f * min(pulse_0, _S244);
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
    uint _S259;
    if(crater_0)
    {
        _S259 = 2U;
    }
    else
    {
        _S259 = 0U;
    }
    uint _S260 = region_2 | _S259;
    if(spall_0)
    {
        region_2 = 4U;
    }
    else
    {
        region_2 = 0U;
    }
    uint _S261 = _S260 | region_2;
    if(perforate_0)
    {
        region_2 = 8U;
    }
    else
    {
        region_2 = 0U;
    }
    uint outcome_5 = _S261 | region_2;
    uint _S262 = lp_body_0(b_19, kernelContext_53);
    *(kernelContext_53->lp_0+_S262) = packed_float4(float4(_S244, v_bl_0, v_res_0, (as_type<float>((outcome_5))))) ;
    packed_float4 device* _S263 = kernelContext_53->lp_0+(_S262 + 1U);
    if(crater_0)
    {
        k_9 = depth_1;
    }
    else
    {
        k_9 = 0.0f;
    }
    *_S263 = packed_float4(float4(k_9, scab_1, a_2, _S255)) ;
    if(perforate_0)
    {
        uint _S264 = a_counters_1(kernelContext_53);
        uint _S265 = atomic_fetch_add_explicit(kernelContext_53->atom_0+(_S264 + 10U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_0)
        {
            uint _S266 = a_counters_1(kernelContext_53);
            uint _S267 = atomic_fetch_add_explicit(kernelContext_53->atom_0+(_S266 + 9U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_0)
            {
                uint _S268 = a_counters_1(kernelContext_53);
                uint _S269 = atomic_fetch_add_explicit(kernelContext_53->atom_0+(_S268 + 8U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S270 = a_counters_1(kernelContext_53);
                uint _S271 = atomic_fetch_add_explicit(kernelContext_53->atom_0+(_S270 + 7U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_0)
    {
        if((kernelContext_53->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_19, _S234.body_0, _S234.mass_0, contact_7, d_7, _S234.m_eff_0 * (_S249 - sqrt(max(_S249 * _S249 - _S256 / _S234.m_eff_0, 0.0f))), kernelContext_53);
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_19, _S234.body_0, _S234.mass_0, contact_7, d_7, _S234.m_eff_0 * _S249, kernelContext_53);
            float3 _S272 = to_rubble_0(d_7) * float3((_S249 * clamp(_S232, 0.0f, kernelContext_53->params_0->dt_0))) ;
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
                uint _S273 = 7U * (_S234.body_0 + region_2 * kernelContext_53->params_0->max_bodies_0);
                (*(kernelContext_53->bodies_0+_S273)).xyz = (float4(*(kernelContext_53->bodies_0+_S273)) ).xyz + _S272;
                region_2 = region_2 + 1U;
            }
        }
    }
    return;
}

float3 chunk_box_world_0(uint k_10, Quat_0 thread* q_8, KernelContext_0 thread* kernelContext_54)
{
    uint _S274 = body_of_0(k_10, kernelContext_54);
    thread float3 origin_2;
    structure_pose_0(_S274, 7U * _S274, q_8, &origin_2, kernelContext_54);
    float3 _S275 = (float4(*(kernelContext_54->geometry_0+3U * k_10)) ).xyz;
    thread Quat_0 _S276 = *q_8;
    float3 _S277 = rotate_0(&_S276, _S275);
    return origin_2 + _S277;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

float2 il_interval_0(uint k_11, float3 p_9, float3 d_8, float pad_0, KernelContext_0 thread* kernelContext_55)
{
    thread Quat_0 q_9;
    float3 _S278 = chunk_box_world_0(k_11, &q_9, kernelContext_55);
    uint _S279 = 3U * k_11;
    Quat_0 _S280 = quat_of_0(float4(*(kernelContext_55->geometry_0+(_S279 + 2U))) );
    thread Quat_0 _S281 = q_9;
    thread Quat_0 _S282 = _S280;
    Quat_0 _S283 = quat_mul_0(&_S281, &_S282);
    thread Quat_0 _S284 = _S283;
    Quat_0 _S285 = normalized_0(&_S284);
    float3 _S286 = (float4(*(kernelContext_55->geometry_0+(_S279 + 1U))) ).xyz;
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
        thread Quat_0 _S287 = _S285;
        float3 _S288 = rotate_0(&_S287, e_0);
        float o_2 = dot(p_9 - _S278, _S288);
        float dv_0 = dot(d_8, _S288);
        float ext_1 = _S286[ax_0] + pad_0;
        if((abs(dv_0)) < 9.99999971718068537e-10f)
        {
            if((abs(o_2)) > ext_1)
            {
                return float2(1.0f, 0.0f);
            }
            ax_0 = ax_0 + 1U;
            continue;
        }
        float ta_0 = (- ext_1 - o_2) / dv_0;
        float tb_0 = (ext_1 - o_2) / dv_0;
        float _S289 = min(t1_0, max(ta_0, tb_0));
        t0_0 = max(t0_0, min(ta_0, tb_0));
        t1_0 = _S289;
        ax_0 = ax_0 + 1U;
    }
    return float2(t0_0, t1_0);
}

void il_extend_0(uint k_12, uint from_0, KernelContext_0 thread* kernelContext_56)
{
    if(k_12 >= (kernelContext_56->params_0->chunk_count_0))
    {
        return;
    }
    uint _S290 = body_of_0(k_12, kernelContext_56);
    bool _S291 = il_struck_0(_S290, kernelContext_56);
    if(!_S291)
    {
        return;
    }
    thread float3 contact_8;
    thread float3 d_9;
    thread float speed_6;
    il_path_0(_S290, &contact_8, &d_9, &speed_6, kernelContext_56);
    IlStriker_0 _S292 = il_striker_0(_S290, kernelContext_56);
    uint _S293 = a_il_0(_S290, kernelContext_56);
    uint _S294 = atomic_load_explicit(kernelContext_56->atom_0+(_S293 + from_0), memory_order_relaxed);
    float end_0 = (as_type<float>((_S294)));
    uint to_0 = _S293 + 1U - from_0;
    uint _S295 = atomic_fetch_max_explicit(kernelContext_56->atom_0+to_0, (as_type<uint>((end_0))), memory_order_relaxed);
    float2 _S296 = il_interval_0(k_12, contact_8, d_9, _S292.radius_0, kernelContext_56);
    float _S297 = _S296.x;
    float _S298 = _S296.y;
    bool _S299;
    if(_S297 <= _S298)
    {
        _S299 = _S297 <= (end_0 + 0.00100000004749745f + 0.01999999955296516f * _S292.radius_0);
    }
    else
    {
        _S299 = false;
    }
    if(_S299)
    {
        _S299 = _S298 > end_0;
    }
    else
    {
        _S299 = false;
    }
    if(_S299)
    {
        uint _S300 = atomic_fetch_max_explicit(kernelContext_56->atom_0+to_0, (as_type<uint>((_S298))), memory_order_relaxed);
    }
    return;
}

void il_begin_0(uint b_20, KernelContext_0 thread* kernelContext_57)
{
    bool _S301;
    if(b_20 >= (kernelContext_57->params_0->body_count_0))
    {
        _S301 = true;
    }
    else
    {
        bool _S302 = il_struck_0(b_20, kernelContext_57);
        _S301 = !_S302;
    }
    if(_S301)
    {
        return;
    }
    thread float3 contact_9;
    thread float3 d_10;
    thread float speed_7;
    il_path_0(b_20, &contact_9, &d_10, &speed_7, kernelContext_57);
    uint _S303 = il_entry_0(b_20, kernelContext_57);
    IlStriker_0 _S304 = il_striker_0(b_20, kernelContext_57);
    float2 _S305 = il_interval_0(_S303, contact_9, d_10, _S304.radius_0, kernelContext_57);
    uint end_1 = (as_type<uint>((max(_S305.y, 0.0f))));
    uint _S306 = a_il_0(b_20, kernelContext_57);
    atomic_store_explicit(kernelContext_57->atom_0+_S306, end_1, memory_order_relaxed);
    atomic_store_explicit(kernelContext_57->atom_0+(_S306 + 1U), end_1, memory_order_relaxed);
    atomic_store_explicit(kernelContext_57->atom_0+(_S306 + 2U), 0U, memory_order_relaxed);
    atomic_store_explicit(kernelContext_57->atom_0+(_S306 + 3U), 0U, memory_order_relaxed);
    return;
}

void il_breaks_1(uint i_1, KernelContext_0 thread* kernelContext_58)
{
    bool cut_1;
    if(i_1 >= (kernelContext_58->params_0->bond_count_0))
    {
        cut_1 = true;
    }
    else
    {
        uint _S307 = w_alive_0(kernelContext_58);
        cut_1 = (*(kernelContext_58->work_0+(_S307 + i_1))) == 0U;
    }
    if(cut_1)
    {
        return;
    }
    uint4 _S308 = uint4(*(kernelContext_58->bonds_0+i_1)) ;
    uint _S309 = _S308.x;
    uint _S310 = body_of_0(_S309, kernelContext_58);
    bool _S311 = il_struck_1(_S310, kernelContext_58);
    if(!_S311)
    {
        return;
    }
    uint _S312 = lp_body_0(_S310, kernelContext_58);
    float4 _S313 = float4(*(kernelContext_58->lp_0+_S312)) ;
    float4 _S314 = float4(*(kernelContext_58->lp_0+(_S312 + 1U))) ;
    uint outcome_6 = (as_type<uint>((_S313.w)));
    if((outcome_6 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_10;
    thread float3 d_11;
    thread float speed_8;
    il_path_1(_S310, &contact_10, &d_11, &speed_8, kernelContext_58);
    uint _S315 = il_entry_1(_S310, kernelContext_58);
    IlStriker_0 _S316 = il_striker_1(_S310, kernelContext_58);
    float _S317 = max(_S316.radius_0, 0.00100000004749745f);
    float h_3 = _S313.x;
    IlPlace_0 _S318 = il_place_0(_S309, _S315, contact_10, d_11, kernelContext_58);
    IlPlace_0 _S319 = il_place_0(_S308.y, _S315, contact_10, d_11, kernelContext_58);
    thread IlPlace_0 _S320 = _S318;
    bool _S321 = il_plug_0(outcome_6, _S317, h_3, &_S320);
    thread IlPlace_0 _S322 = _S319;
    bool _S323 = il_plug_0(outcome_6, _S317, h_3, &_S322);
    if(_S321 != _S323)
    {
        cut_1 = true;
    }
    else
    {
        float _S324 = _S314.y;
        thread IlPlace_0 _S325 = _S318;
        bool _S326 = il_scab_0(outcome_6, _S317, h_3, _S324, &_S325);
        thread IlPlace_0 _S327 = _S319;
        bool _S328 = il_scab_0(outcome_6, _S317, h_3, _S324, &_S327);
        cut_1 = _S326 != _S328;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S329 = _S314.x;
        float _S330 = _S314.z;
        thread IlPlace_0 _S331 = _S318;
        bool _S332 = il_crater_0(outcome_6, _S317, _S329, _S330, &_S331);
        cut_1 = _S332;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S333 = _S314.x;
        float _S334 = _S314.z;
        thread IlPlace_0 _S335 = _S319;
        bool _S336 = il_crater_0(outcome_6, _S317, _S333, _S334, &_S335);
        cut_1 = _S336;
    }
    if(cut_1)
    {
        uint _S337 = w_alive_0(kernelContext_58);
        *(kernelContext_58->work_0+(_S337 + i_1)) = 0U;
        uint _S338 = a_dirty_0(kernelContext_58);
        uint _S339 = atomic_fetch_or_explicit(kernelContext_58->atom_0+(_S338 + _S310), 1U, memory_order_relaxed);
        uint _S340 = a_counters_1(kernelContext_58);
        uint _S341 = atomic_fetch_add_explicit(kernelContext_58->atom_0+_S340, 1U, memory_order_relaxed);
        uint _S342 = atomic_fetch_add_explicit(kernelContext_58->atom_0+(_S340 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

void il_carry_1(uint k_13, KernelContext_0 thread* kernelContext_59)
{
    if(k_13 >= (kernelContext_59->params_0->chunk_count_0))
    {
        return;
    }
    uint _S343 = w_carry_0(kernelContext_59);
    uint _S344 = _S343 + k_13;
    uint device* _S345 = kernelContext_59->work_0+_S344;
    uint c_4 = *_S345;
    if((*_S345) == 0U)
    {
        return;
    }
    *(kernelContext_59->work_0+_S344) = 0U;
    uint b0_1 = c_4 - 1U;
    bool _S346 = il_struck_1(b0_1, kernelContext_59);
    if(!_S346)
    {
        return;
    }
    thread float3 contact_11;
    thread float3 d_12;
    thread float speed_9;
    il_path_1(b0_1, &contact_11, &d_12, &speed_9, kernelContext_59);
    float3 dr_1 = to_rubble_0(d_12);
    IlStriker_0 _S347 = il_striker_1(b0_1, kernelContext_59);
    uint _S348 = a_il_0(b0_1, kernelContext_59);
    float _S349 = read64_0(_S348 + 2U, kernelContext_59->params_0->scale_m_0, kernelContext_59);
    float3 v_wall_1 = (float4(*(kernelContext_59->bodies_0+(7U * b0_1 + 3U))) ).xyz;
    uint _S350 = 7U * _S347.body_0;
    float3 v_shot_1 = (float4(*(kernelContext_59->bodies_0+(_S350 + 3U))) ).xyz;
    float closing_1 = dot(v_shot_1 - v_wall_1, dr_1);
    float v_r_1 = _S347.mass_0 * closing_1 / (_S347.mass_0 + _S349);
    float3 _S351 = v_shot_1 + dr_1 * float3((v_r_1 - closing_1)) ;
    float3 _S352 = v_wall_1 + dr_1 * float3(v_r_1) ;
    float _S353 = il_t_in_1(b0_1, kernelContext_59);
    float3 _S354 = (float4(*(kernelContext_59->bodies_0+_S350)) ).xyz + dr_1 * float3(((closing_1 - v_r_1) * clamp(_S353, 0.0f, kernelContext_59->params_0->dt_0))) ;
    uint _S355 = body_of_0(k_13, kernelContext_59);
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
        uint s_at_2 = 7U * (_S347.body_0 + region_3 * kernelContext_59->params_0->max_bodies_0);
        (*(kernelContext_59->bodies_0+s_at_2)).xyz = _S354;
        (*(kernelContext_59->bodies_0+(s_at_2 + 3U))).xyz = _S351;
        (*(kernelContext_59->bodies_0+(7U * (_S355 + region_3 * kernelContext_59->params_0->max_bodies_0) + 3U))).xyz = _S352;
        region_3 = region_3 + 1U;
    }
    return;
}

void il_mark_1(uint k_14, KernelContext_0 thread* kernelContext_60)
{
    if(k_14 >= (kernelContext_60->params_0->chunk_count_0))
    {
        return;
    }
    uint _S356 = body_of_0(k_14, kernelContext_60);
    bool _S357 = il_struck_1(_S356, kernelContext_60);
    if(!_S357)
    {
        return;
    }
    uint _S358 = lp_body_0(_S356, kernelContext_60);
    float4 _S359 = float4(*(kernelContext_60->lp_0+_S358)) ;
    float4 _S360 = float4(*(kernelContext_60->lp_0+(_S358 + 1U))) ;
    uint outcome_7 = (as_type<uint>((_S359.w)));
    if((outcome_7 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_12;
    thread float3 d_13;
    thread float speed_10;
    il_path_1(_S356, &contact_12, &d_13, &speed_10, kernelContext_60);
    uint _S361 = il_entry_1(_S356, kernelContext_60);
    IlStriker_0 _S362 = il_striker_1(_S356, kernelContext_60);
    float _S363 = max(_S362.radius_0, 0.00100000004749745f);
    IlPlace_0 _S364 = il_place_0(k_14, _S361, contact_12, d_13, kernelContext_60);
    float _S365 = _S359.x;
    thread IlPlace_0 _S366 = _S364;
    bool _S367 = il_plug_0(outcome_7, _S363, _S365, &_S366);
    float2 _S368 = il_interval_0(k_14, contact_12 - d_13 * float3(_S362.bound_0) , d_13, _S362.bound_0, kernelContext_60);
    float _S369 = _S368.x;
    float _S370 = _S368.y;
    bool path_0;
    if(_S369 <= _S370)
    {
        path_0 = _S369 <= (_S365 + 2.0f * _S362.bound_0);
    }
    else
    {
        path_0 = false;
    }
    if(path_0)
    {
        path_0 = _S370 >= 0.0f;
    }
    else
    {
        path_0 = false;
    }
    if(_S367)
    {
        path_0 = true;
    }
    if(path_0)
    {
        path_0 = true;
    }
    else
    {
        float _S371 = _S360.y;
        thread IlPlace_0 _S372 = _S364;
        bool _S373 = il_scab_0(outcome_7, _S363, _S365, _S371, &_S372);
        path_0 = _S373;
    }
    if(path_0)
    {
        path_0 = true;
    }
    else
    {
        float _S374 = _S360.x;
        float _S375 = _S360.z;
        thread IlPlace_0 _S376 = _S364;
        bool _S377 = il_crater_0(outcome_7, _S363, _S374, _S375, &_S376);
        path_0 = _S377;
    }
    if(path_0)
    {
        uint _S378 = w_marks_0(kernelContext_60);
        *(kernelContext_60->work_0+(_S378 + k_14)) = _S362.launch_0;
    }
    if(_S367)
    {
        uint _S379 = w_carry_0(kernelContext_60);
        *(kernelContext_60->work_0+(_S379 + k_14)) = _S356 + 1U;
        uint _S380 = a_il_0(_S356, kernelContext_60);
        add64_0(_S380 + 2U, (float4(*(kernelContext_60->chunks_0+4U * k_14)) ).w, kernelContext_60->params_0->scale_m_0, kernelContext_60);
    }
    return;
}

void il_strike_1(uint b_21, KernelContext_0 thread* kernelContext_61)
{
    bool damage_1;
    if(b_21 >= (kernelContext_61->params_0->body_count_0))
    {
        damage_1 = true;
    }
    else
    {
        bool _S381 = il_struck_1(b_21, kernelContext_61);
        damage_1 = !_S381;
    }
    if(damage_1)
    {
        return;
    }
    float _S382 = il_t_in_1(b_21, kernelContext_61);
    uint _S383 = il_entry_1(b_21, kernelContext_61);
    thread float3 contact_13;
    thread float3 d_14;
    thread float speed_11;
    il_path_1(b_21, &contact_13, &d_14, &speed_11, kernelContext_61);
    IlStriker_0 _S384 = il_striker_1(b_21, kernelContext_61);
    float _S385 = max(_S384.radius_0, 0.00100000004749745f);
    if((_S384.launch_0) != 0U)
    {
        uint _S386 = w_marks_0(kernelContext_61);
        damage_1 = (*(kernelContext_61->work_0+(_S386 + _S383))) == (_S384.launch_0);
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        uint _S387 = lp_body_0(b_21, kernelContext_61);
        packed_float4 _S388 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_61->lp_0+_S387) = _S388;
        *(kernelContext_61->lp_0+(_S387 + 1U)) = _S388;
        return;
    }
    uint _S389 = a_il_0(b_21, kernelContext_61);
    uint _S390 = atomic_load_explicit(kernelContext_61->atom_0+_S389, memory_order_relaxed);
    uint _S391 = atomic_load_explicit(kernelContext_61->atom_0+(_S389 + 1U), memory_order_relaxed);
    if(_S390 != _S391)
    {
        uint _S392 = a_counters_1(kernelContext_61);
        uint _S393 = atomic_fetch_or_explicit(kernelContext_61->atom_0+(_S392 + 3U), 16U, memory_order_relaxed);
    }
    float _S394 = max((as_type<float>((_S390))) - _S385, 9.99999997475242708e-07f);
    if(!(_S382 > 0.0f))
    {
        uint _S395 = lp_body_0(b_21, kernelContext_61);
        *(kernelContext_61->lp_0+_S395) = packed_float4(float4(_S394, 0.0f, 0.0f, 0.0f)) ;
        *(kernelContext_61->lp_0+(_S395 + 1U)) = packed_float4(float4(0.0f) ) ;
        return;
    }
    uint o_3 = kernelContext_61->params_0->ext_chunks_0 + 4U * _S383;
    float4 _S396 = float4(*(kernelContext_61->ext_0+(o_3 + 1U))) ;
    thread JointMaterial_0 dm_1;
    float4 _S397 = float4(0.0f) ;
    (&dm_1)->strength_0 = _S397;
    (&dm_1)->energy_0 = _S397;
    (&dm_1)->dif_0 = float4(*(kernelContext_61->ext_0+(o_3 + 2U))) ;
    float4 _S398 = float4(*(kernelContext_61->ext_0+(o_3 + 3U))) ;
    float _S399 = _S398.x;
    (&dm_1)->misc_0 = float4(_S399, 0.0f, 0.0f, 0.0f);
    (&dm_1)->kind_flags_0 = uint4(0U) ;
    float nu_1 = _S398.y;
    float e_target_1 = _S398.z / (1.0f - nu_1 * nu_1);
    float e_star_1 = 1.0f / (1.0f / _S384.modulus_0 + 1.0f / e_target_1);
    float _S400 = max(speed_11, 0.00100000004749745f);
    float _S401 = sqrt(_S385);
    float h0_1 = pow(15.0f * _S384.m_eff_0 * _S400 * _S400 / (16.0f * e_star_1 * _S401), 0.40000000596046448f);
    float t_c_1 = 2.94000005722045898f * h0_1 / _S400;
    float force_1 = 1.33333337306976318f * e_star_1 * _S401 * pow(h0_1, 1.5f);
    float a_3 = sqrt(_S385 * h0_1);
    float p0_1 = 3.0f * force_1 / (6.28318548202514648f * a_3 * a_3);
    float ring_1 = (1.0f - 2.0f * nu_1) * p0_1 / 3.0f;
    float k_15;
    if(_S399 > 0.0f)
    {
        float _S402 = _S400 / _S394;
        thread JointMaterial_0 _S403 = dm_1;
        float _S404 = dif_factor_0(&_S403, _S402);
        k_15 = _S404;
    }
    else
    {
        k_15 = 1.0f;
    }
    float ft_1 = k_15 * _S396.y;
    float fc_1 = k_15 * _S396.z;
    float tau_1 = k_15 * _S396.w;
    if(p0_1 > fc_1)
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = ring_1 > ft_1;
    }
    float _S405 = fc_1 * (3.14159274101257324f * _S385 * _S385);
    float _S406 = min(_S405 * _S394, 1.57079637050628662f * (2.0f * _S385 + _S394) * _S394 * _S394 * tau_1);
    float _S407 = 2.0f * _S406;
    float v_bl_1 = sqrt(_S407 / _S384.m_eff_0);
    bool perforate_1;
    if(damage_1)
    {
        perforate_1 = _S400 >= v_bl_1;
    }
    else
    {
        perforate_1 = false;
    }
    float r2_1 = _S385 + _S394;
    float _S408 = _S396.x;
    float plug_1 = _S408 * 3.14159274101257324f * _S394 / 3.0f * (_S385 * _S385 + _S385 * r2_1 + r2_1 * r2_1);
    float v_res_1;
    if(perforate_1)
    {
        v_res_1 = _S384.mass_0 / (_S384.mass_0 + plug_1) * sqrt(max(_S400 * _S400 - v_bl_1 * v_bl_1, 0.0f));
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
        depth_2 = min(0.5f * _S384.m_eff_0 * _S400 * _S400 / _S405, _S394);
    }
    else
    {
        depth_2 = 0.0f;
    }
    bool crater_1 = depth_2 > a_3;
    float c_5 = sqrt(e_target_1 / _S408);
    float z_t_1 = _S408 * c_5;
    float rho_p_1 = _S384.mass_0 / (4.18879032135009766f * _S385 * _S385 * _S385);
    float z_p_1 = rho_p_1 * sqrt(_S384.modulus_0 / rho_p_1);
    float mean_pressure_1 = force_1 / (3.14159274101257324f * a_3 * a_3);
    float _S409 = min(min(z_t_1 * _S400 * z_p_1 / (z_p_1 + z_t_1), mean_pressure_1), fc_1);
    float source_radius_1;
    if(mean_pressure_1 > fc_1)
    {
        source_radius_1 = max(a_3, _S385);
    }
    else
    {
        source_radius_1 = a_3;
    }
    float pulse_1 = c_5 * t_c_1;
    float back_1 = _S409 * min(source_radius_1 / _S394, 1.0f) * min(2.0f * _S394 / pulse_1, 1.0f);
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
        scab_2 = 0.5f * min(pulse_1, _S394);
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
    uint _S410;
    if(crater_1)
    {
        _S410 = 2U;
    }
    else
    {
        _S410 = 0U;
    }
    uint _S411 = region_4 | _S410;
    if(spall_1)
    {
        region_4 = 4U;
    }
    else
    {
        region_4 = 0U;
    }
    uint _S412 = _S411 | region_4;
    if(perforate_1)
    {
        region_4 = 8U;
    }
    else
    {
        region_4 = 0U;
    }
    uint outcome_8 = _S412 | region_4;
    uint _S413 = lp_body_0(b_21, kernelContext_61);
    *(kernelContext_61->lp_0+_S413) = packed_float4(float4(_S394, v_bl_1, v_res_1, (as_type<float>((outcome_8))))) ;
    packed_float4 device* _S414 = kernelContext_61->lp_0+(_S413 + 1U);
    if(crater_1)
    {
        k_15 = depth_2;
    }
    else
    {
        k_15 = 0.0f;
    }
    *_S414 = packed_float4(float4(k_15, scab_2, a_3, _S406)) ;
    if(perforate_1)
    {
        uint _S415 = a_counters_1(kernelContext_61);
        uint _S416 = atomic_fetch_add_explicit(kernelContext_61->atom_0+(_S415 + 10U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_1)
        {
            uint _S417 = a_counters_1(kernelContext_61);
            uint _S418 = atomic_fetch_add_explicit(kernelContext_61->atom_0+(_S417 + 9U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_1)
            {
                uint _S419 = a_counters_1(kernelContext_61);
                uint _S420 = atomic_fetch_add_explicit(kernelContext_61->atom_0+(_S419 + 8U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S421 = a_counters_1(kernelContext_61);
                uint _S422 = atomic_fetch_add_explicit(kernelContext_61->atom_0+(_S421 + 7U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_1)
    {
        if((kernelContext_61->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_21, _S384.body_0, _S384.mass_0, contact_13, d_14, _S384.m_eff_0 * (_S400 - sqrt(max(_S400 * _S400 - _S407 / _S384.m_eff_0, 0.0f))), kernelContext_61);
        }
    }
    else
    {
        if(damage_1)
        {
            il_impulse_0(b_21, _S384.body_0, _S384.mass_0, contact_13, d_14, _S384.m_eff_0 * _S400, kernelContext_61);
            float3 _S423 = to_rubble_0(d_14) * float3((_S400 * clamp(_S382, 0.0f, kernelContext_61->params_0->dt_0))) ;
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
                uint _S424 = 7U * (_S384.body_0 + region_4 * kernelContext_61->params_0->max_bodies_0);
                (*(kernelContext_61->bodies_0+_S424)).xyz = (float4(*(kernelContext_61->bodies_0+_S424)) ).xyz + _S423;
                region_4 = region_4 + 1U;
            }
        }
    }
    return;
}

void il_extend_1(uint k_16, uint from_1, KernelContext_0 thread* kernelContext_62)
{
    if(k_16 >= (kernelContext_62->params_0->chunk_count_0))
    {
        return;
    }
    uint _S425 = body_of_0(k_16, kernelContext_62);
    bool _S426 = il_struck_1(_S425, kernelContext_62);
    if(!_S426)
    {
        return;
    }
    thread float3 contact_14;
    thread float3 d_15;
    thread float speed_12;
    il_path_1(_S425, &contact_14, &d_15, &speed_12, kernelContext_62);
    IlStriker_0 _S427 = il_striker_1(_S425, kernelContext_62);
    uint _S428 = a_il_0(_S425, kernelContext_62);
    uint _S429 = atomic_load_explicit(kernelContext_62->atom_0+(_S428 + from_1), memory_order_relaxed);
    float end_2 = (as_type<float>((_S429)));
    uint to_1 = _S428 + 1U - from_1;
    uint _S430 = atomic_fetch_max_explicit(kernelContext_62->atom_0+to_1, (as_type<uint>((end_2))), memory_order_relaxed);
    float2 _S431 = il_interval_0(k_16, contact_14, d_15, _S427.radius_0, kernelContext_62);
    float _S432 = _S431.x;
    float _S433 = _S431.y;
    bool _S434;
    if(_S432 <= _S433)
    {
        _S434 = _S432 <= (end_2 + 0.00100000004749745f + 0.01999999955296516f * _S427.radius_0);
    }
    else
    {
        _S434 = false;
    }
    if(_S434)
    {
        _S434 = _S433 > end_2;
    }
    else
    {
        _S434 = false;
    }
    if(_S434)
    {
        uint _S435 = atomic_fetch_max_explicit(kernelContext_62->atom_0+to_1, (as_type<uint>((_S433))), memory_order_relaxed);
    }
    return;
}

uint a_first_1(uint shot_3, KernelContext_0 thread* kernelContext_63)
{
    uint _S436 = a_strike_0(kernelContext_63);
    return _S436 + 27U * kernelContext_63->params_0->body_count_0 + shot_3;
}

void il_begin_1(uint b_22, KernelContext_0 thread* kernelContext_64)
{
    bool _S437;
    if(b_22 >= (kernelContext_64->params_0->body_count_0))
    {
        _S437 = true;
    }
    else
    {
        bool _S438 = il_struck_1(b_22, kernelContext_64);
        _S437 = !_S438;
    }
    if(_S437)
    {
        return;
    }
    thread uint shot_4;
    bool _S439 = struck_0(b_22, &shot_4, kernelContext_64);
    uint _S440 = a_strike_0(kernelContext_64);
    uint _S441 = atomic_load_explicit(kernelContext_64->atom_0+(_S440 + 9U * b_22 + 6U), memory_order_relaxed);
    uint _S442 = a_first_1(shot_4, kernelContext_64);
    uint _S443 = atomic_load_explicit(kernelContext_64->atom_0+_S442, memory_order_relaxed);
    if(_S441 != _S443)
    {
        uint _S444 = strike_at_0(b_22, kernelContext_64);
        (*(kernelContext_64->frames_0+_S444))[int(3)] = 0.0f;
        return;
    }
    thread float3 contact_15;
    thread float3 d_16;
    thread float speed_13;
    il_path_1(b_22, &contact_15, &d_16, &speed_13, kernelContext_64);
    uint _S445 = il_entry_1(b_22, kernelContext_64);
    IlStriker_0 _S446 = il_striker_1(b_22, kernelContext_64);
    float2 _S447 = il_interval_0(_S445, contact_15, d_16, _S446.radius_0, kernelContext_64);
    uint end_3 = (as_type<uint>((max(_S447.y, 0.0f))));
    uint _S448 = a_il_0(b_22, kernelContext_64);
    atomic_store_explicit(kernelContext_64->atom_0+_S448, end_3, memory_order_relaxed);
    atomic_store_explicit(kernelContext_64->atom_0+(_S448 + 1U), end_3, memory_order_relaxed);
    atomic_store_explicit(kernelContext_64->atom_0+(_S448 + 2U), 0U, memory_order_relaxed);
    atomic_store_explicit(kernelContext_64->atom_0+(_S448 + 3U), 0U, memory_order_relaxed);
    return;
}

uint l_pool_0(uint half_0, KernelContext_0 thread* kernelContext_65)
{
    return kernelContext_65->params_0->chunk_count_0 + 3U * kernelContext_65->params_0->pool_cap_0 * half_0;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S449 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S449, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S450 = max(0.0f, _S449);
    float _S451 = min(dt_1, _S449 + duration_0);
    bool _S452;
    if(_S451 > _S450)
    {
        _S452 = duration_0 > 0.0f;
    }
    else
    {
        _S452 = false;
    }
    float sum_1;
    if(_S452)
    {
        float w_2 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_2 * (cos(w_2 * (_S450 - _S449)) - cos(w_2 * (_S451 - _S449)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

float hertz_duration_0(float m_0, float r_16, float e_1, float v_9)
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_16 * e_1 * e_1 * max(abs(v_9), 0.00100000004749745f)), 0.20000000298023224f);
}

uint l_vec_0(KernelContext_0 thread* kernelContext_66)
{
    return kernelContext_66->params_0->chunk_count_0 + 6U * kernelContext_66->params_0->pool_cap_0;
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
    thread KernelContext_0 kernelContext_67;
    (&kernelContext_67)->params_0 = params_1;
    (&kernelContext_67)->work_0 = work_1;
    (&kernelContext_67)->frames_0 = frames_1;
    (&kernelContext_67)->bodies_0 = bodies_1;
    (&kernelContext_67)->chunks_0 = chunks_1;
    (&kernelContext_67)->sort_0 = sort_1;
    (&kernelContext_67)->atom_0 = atom_1;
    (&kernelContext_67)->manifolds_0 = manifolds_1;
    (&kernelContext_67)->loads_0 = loads_1;
    (&kernelContext_67)->bonds_0 = bonds_1;
    (&kernelContext_67)->args_0 = args_1;
    (&kernelContext_67)->info_0 = info_1;
    (&kernelContext_67)->scan_0 = scan_1;
    (&kernelContext_67)->parts_0 = parts_1;
    (&kernelContext_67)->colliders_0 = colliders_1;
    (&kernelContext_67)->collider_info_0 = collider_info_1;
    (&kernelContext_67)->view_chunks_0 = view_chunks_1;
    (&kernelContext_67)->view_0 = view_1;
    (&kernelContext_67)->objects_0 = objects_1;
    (&kernelContext_67)->shot_bodies_0 = shot_bodies_1;
    (&kernelContext_67)->lines_0 = lines_1;
    (&kernelContext_67)->geometry_0 = geometry_1;
    (&kernelContext_67)->shots_0 = shots_1;
    (&kernelContext_67)->lp_0 = lp_1;
    (&kernelContext_67)->ext_0 = ext_2;
    (&kernelContext_67)->lp_sides_0 = lp_sides_1;
    (&kernelContext_67)->engine_counters_0 = engine_counters_1;
    (&kernelContext_67)->engine_contacts_0 = engine_contacts_1;
    (&kernelContext_67)->cs_anchors_0 = cs_anchors_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_67)->g_scan_0 = &g_scan_1;
    uint k_17 = id_0.x;
    if(k_17 >= (params_1->chunk_count_0))
    {
        return;
    }
    uint _S453 = body_of_0(k_17, &kernelContext_67);
    uint b_23 = 7U * _S453;
    thread Quat_0 q_10;
    thread float3 origin_3;
    structure_pose_0(_S453, b_23, &q_10, &origin_3, &kernelContext_67);
    uint _S454 = 4U * k_17;
    float4 _S455 = float4(*((&kernelContext_67)->chunks_0+_S454)) ;
    float3 _S456 = _S455.xyz;
    thread Quat_0 _S457 = q_10;
    float3 _S458 = rotate_0(&_S457, _S456);
    float3 centre_0 = origin_3 + _S458;
    float inv_mass_2 = (float4(*((&kernelContext_67)->bodies_0+b_23)) ).w;
    float m_body_0;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint from_2 = (&kernelContext_67)->params_0->parity_0;
    uint to_2 = 1U - (&kernelContext_67)->params_0->parity_0;
    uint _S459 = w_pulses_0(&kernelContext_67);
    uint _S460 = _S459 + _S454;
    uint _S461 = _S460 + 2U * from_2;
    uint off_0 = *((&kernelContext_67)->work_0+_S461);
    uint device* _S462 = (&kernelContext_67)->work_0+(_S461 + 1U);
    uint carried_0 = *_S462;
    uint room_0 = *_S462 + (&kernelContext_67)->sort_0[params_1->chunk_count_0 + 1U + k_17];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S463 = a_counters_1(&kernelContext_67);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_67)->atom_0+(_S463 + 5U + to_2), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_67)->params_0->pool_cap_0))
        {
            uint _S464 = atomic_fetch_or_explicit((&kernelContext_67)->atom_0+(_S463 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_67)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_67)->params_0->pool_cap_0 - base_1;
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
        uint _S465 = base_0;
        base_0 = base_1;
        room_1 = _S465;
    }
    else
    {
        base_0 = 0U;
        room_1 = room_0;
    }
    uint _S466 = params_1->chunk_count_0;
    float _S467 = _S455.w;
    float4 _S468 = float4(*((&kernelContext_67)->chunks_0+(_S454 + 3U))) ;
    float _S469 = (&kernelContext_67)->params_0->band_0;
    float _S470 = (&kernelContext_67)->params_0->dt_0;
    bool _S471 = ((&kernelContext_67)->params_0->impact_peak_0) != 0U;
    uint _S472 = l_pool_0(from_2, &kernelContext_67);
    uint _S473 = _S472 + 3U * off_0;
    uint _S474 = l_pool_0(to_2, &kernelContext_67);
    uint _S475 = _S474 + 3U * base_0;
    for(;;)
    {
        float3 _S476 = float3(0.0f) ;
        uint _S477 = (&kernelContext_67)->sort_0[k_17];
        uint _S478 = (&kernelContext_67)->sort_0[_S466 + 1U + k_17];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_1 = 0U;
        f_total_0 = _S476;
        t_total_0 = _S476;
        for(;;)
        {
            if(step_1 < _S478)
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
                if(q3_0 < _S478)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_67)->manifolds_0[_S477 + q3_0].ids_0) ).y + 1U;
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
            Record_natural_0 r_17 = (&kernelContext_67)->manifolds_0[_S477 + p_10];
            float4 _S479 = float4(r_17.impulse_0) ;
            float3 _S480 = _S479.xyz;
            float j_1 = length(_S480);
            if(j_1 == 0.0f)
            {
                uint _S481 = step_1 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S481;
                continue;
            }
            float m_other_0 = (float4(r_17.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S467;
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
            float _S482 = min(j_1, 2.0f * m_1 * max(_S479.w - _S469, 0.0f));
            if(_S482 < j_1)
            {
                float3 f_1 = _S480 / float3(j_1)  * float3(((j_1 - _S482) / _S470)) ;
                float3 _S483 = t_total_0 + cross((float4(r_17.point_0) ).xyz - centre_0, f_1);
                f_total_1 = f_total_0 + f_1;
                t_total_1 = _S483;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            uint _S481 = step_1 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S481;
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
            uint _S484 = _S473 + 3U * p_10;
            float4 _S485 = float4(*((&kernelContext_67)->loads_0+_S484)) ;
            thread float4 a_4 = _S485;
            float4 _S486 = float4(*((&kernelContext_67)->loads_0+(_S484 + 1U))) ;
            float4 _S487 = float4(*((&kernelContext_67)->loads_0+(_S484 + 2U))) ;
            float _S488 = _S486.w;
            float mean_0 = pulse_mean_0(_S485.x, _S485.y, _S485.z, _S485.w, _S488, _S470);
            if(mean_0 != 0.0f)
            {
                float3 f_2 = _S487.xyz * float3(mean_0) ;
                float3 _S489 = t_total_0 + cross(_S486.xyz - centre_0, f_2);
                f_total_0 = f_total_0 + f_2;
                t_total_0 = _S489;
            }
            if((a_4.x + a_4.z + _S488) > _S470)
            {
                a_4.x = a_4.x - _S470;
                uint _S490 = _S475 + 3U * kept_0;
                *((&kernelContext_67)->loads_0+_S490) = packed_float4(a_4) ;
                *((&kernelContext_67)->loads_0+(_S490 + 1U)) = packed_float4(_S486) ;
                *((&kernelContext_67)->loads_0+(_S490 + 2U)) = packed_float4(_S487) ;
                kept_0 = kept_0 + 1U;
            }
            p_10 = p_10 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S478)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_18 = (&kernelContext_67)->manifolds_0[_S477 + q3_0];
            float4 _S491 = float4(r_18.impulse_0) ;
            float3 _S492 = _S491.xyz;
            float j_2 = length(_S492);
            if(j_2 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S493 = float4(r_18.other_0) ;
            float m_other_1 = _S493.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S467;
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
            float _S494 = _S491.w;
            float _S495 = min(j_2, 2.0f * m_1 * max(_S494 - _S469, 0.0f));
            if(_S495 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S496 = _S468.y;
            float duration_1 = hertz_duration_0(m_1, min(_S493.z, _S468.z), 1.0f / ((1.0f - _S496 * _S496) / _S468.x + 1.0f / _S493.y), _S494);
            float peak_1 = 3.14159274101257324f * _S495 / (2.0f * duration_1);
            float3 dir_0 = _S492 / float3(j_2) ;
            float mean_1;
            if(_S471)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S470);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_3 = dir_0 * float3(mean_1) ;
                float3 _S497 = t_total_0 + cross((float4(r_18.point_0) ).xyz - centre_0, f_3);
                f_total_1 = f_total_0 + f_3;
                t_total_1 = _S497;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            uint kept_1;
            if(duration_1 > _S470)
            {
                if(kept_0 < room_1)
                {
                    uint _S498 = _S475 + 3U * kept_0;
                    *((&kernelContext_67)->loads_0+_S498) = packed_float4(float4(- _S470, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_67)->loads_0+(_S498 + 1U)) = packed_float4(float4((float4(r_18.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_67)->loads_0+(_S498 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
        uint _S499 = a_counters_1(&kernelContext_67);
        uint _S500 = atomic_fetch_or_explicit((&kernelContext_67)->atom_0+(_S499 + 3U), 8U, memory_order_relaxed);
    }
    uint _S501 = _S460 + 2U * to_2;
    *((&kernelContext_67)->work_0+_S501) = base_0;
    *((&kernelContext_67)->work_0+(_S501 + 1U)) = kept_0;
    *((&kernelContext_67)->loads_0+k_17) = packed_float4(float4(length(f_total_0), 0.0f, 0.0f, 0.0f)) ;
    if(((&kernelContext_67)->params_0->load_model_0) == 1U)
    {
        uint _S502 = l_vec_0(&kernelContext_67);
        uint _S503 = _S502 + 2U * k_17;
        *((&kernelContext_67)->loads_0+_S503) = packed_float4(float4(f_total_0, 0.0f)) ;
        *((&kernelContext_67)->loads_0+(_S503 + 1U)) = packed_float4(float4(t_total_0, 0.0f)) ;
    }
    return;
}

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], atomic<uint> device* atom_2 [[buffer(5)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]], packed_float4 device* geometry_2 [[buffer(21)]], packed_float4 device* shots_2 [[buffer(22)]], packed_float4 device* lp_2 [[buffer(24)]], packed_float4 device* ext_3 [[buffer(23)]], packed_float4 device* lp_sides_2 [[buffer(28)]], uint device* engine_counters_2 [[buffer(26)]], packed_float4 device* engine_contacts_2 [[buffer(25)]], packed_float4 device* cs_anchors_2 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_68;
    (&kernelContext_68)->params_0 = params_2;
    (&kernelContext_68)->work_0 = work_2;
    (&kernelContext_68)->frames_0 = frames_2;
    (&kernelContext_68)->bodies_0 = bodies_2;
    (&kernelContext_68)->chunks_0 = chunks_2;
    (&kernelContext_68)->sort_0 = sort_2;
    (&kernelContext_68)->atom_0 = atom_2;
    (&kernelContext_68)->manifolds_0 = manifolds_2;
    (&kernelContext_68)->loads_0 = loads_2;
    (&kernelContext_68)->bonds_0 = bonds_2;
    (&kernelContext_68)->args_0 = args_2;
    (&kernelContext_68)->info_0 = info_2;
    (&kernelContext_68)->scan_0 = scan_2;
    (&kernelContext_68)->parts_0 = parts_2;
    (&kernelContext_68)->colliders_0 = colliders_2;
    (&kernelContext_68)->collider_info_0 = collider_info_2;
    (&kernelContext_68)->view_chunks_0 = view_chunks_2;
    (&kernelContext_68)->view_0 = view_2;
    (&kernelContext_68)->objects_0 = objects_2;
    (&kernelContext_68)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_68)->lines_0 = lines_2;
    (&kernelContext_68)->geometry_0 = geometry_2;
    (&kernelContext_68)->shots_0 = shots_2;
    (&kernelContext_68)->lp_0 = lp_2;
    (&kernelContext_68)->ext_0 = ext_3;
    (&kernelContext_68)->lp_sides_0 = lp_sides_2;
    (&kernelContext_68)->engine_counters_0 = engine_counters_2;
    (&kernelContext_68)->engine_contacts_0 = engine_contacts_2;
    (&kernelContext_68)->cs_anchors_0 = cs_anchors_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_68)->g_scan_0 = &g_scan_2;
    uint i_2 = id_1.x;
    bool _S504;
    if(i_2 >= (params_2->bond_count_0))
    {
        _S504 = true;
    }
    else
    {
        uint _S505 = w_alive_0(&kernelContext_68);
        _S504 = (*((&kernelContext_68)->work_0+(_S505 + i_2))) == 0U;
    }
    if(_S504)
    {
        return;
    }
    uint4 _S506 = uint4(*((&kernelContext_68)->bonds_0+i_2)) ;
    uint _S507 = _S506.x;
    if((max((float4(*((&kernelContext_68)->loads_0+_S507)) ).x, (float4(*((&kernelContext_68)->loads_0+_S506.y)) ).x)) > (as_type<float>((_S506.z))))
    {
        uint _S508 = w_alive_0(&kernelContext_68);
        *((&kernelContext_68)->work_0+(_S508 + i_2)) = 0U;
        uint _S509 = a_dirty_0(&kernelContext_68);
        uint _S510 = body_of_0(_S507, &kernelContext_68);
        uint _S511 = atomic_fetch_or_explicit((&kernelContext_68)->atom_0+(_S509 + _S510), 1U, memory_order_relaxed);
        uint _S512 = a_counters_1(&kernelContext_68);
        uint _S513 = atomic_fetch_add_explicit((&kernelContext_68)->atom_0+_S512, 1U, memory_order_relaxed);
        uint _S514 = atomic_fetch_add_explicit((&kernelContext_68)->atom_0+(_S512 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint groups_of_0(uint n_3, uint size_0)
{
    uint _S515 = (n_3 + size_0 - 1U) / size_0;
    return _S515;
}

void set_args_0(uint at_4, uint x_3, KernelContext_0 thread* kernelContext_69)
{
    uint _S516 = 3U * at_4;
    *(kernelContext_69->args_0+_S516) = x_3;
    *(kernelContext_69->args_0+(_S516 + 1U)) = 1U;
    *(kernelContext_69->args_0+(_S516 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], packed_float4 device* lp_3 [[buffer(24)]], packed_float4 device* ext_4 [[buffer(23)]], packed_float4 device* lp_sides_3 [[buffer(28)]], uint device* engine_counters_3 [[buffer(26)]], packed_float4 device* engine_contacts_3 [[buffer(25)]], packed_float4 device* cs_anchors_3 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_70;
    (&kernelContext_70)->params_0 = params_3;
    (&kernelContext_70)->work_0 = work_3;
    (&kernelContext_70)->frames_0 = frames_3;
    (&kernelContext_70)->bodies_0 = bodies_3;
    (&kernelContext_70)->chunks_0 = chunks_3;
    (&kernelContext_70)->sort_0 = sort_3;
    (&kernelContext_70)->atom_0 = atom_3;
    (&kernelContext_70)->manifolds_0 = manifolds_3;
    (&kernelContext_70)->loads_0 = loads_3;
    (&kernelContext_70)->bonds_0 = bonds_3;
    (&kernelContext_70)->args_0 = args_3;
    (&kernelContext_70)->info_0 = info_3;
    (&kernelContext_70)->scan_0 = scan_3;
    (&kernelContext_70)->parts_0 = parts_3;
    (&kernelContext_70)->colliders_0 = colliders_3;
    (&kernelContext_70)->collider_info_0 = collider_info_3;
    (&kernelContext_70)->view_chunks_0 = view_chunks_3;
    (&kernelContext_70)->view_0 = view_3;
    (&kernelContext_70)->objects_0 = objects_3;
    (&kernelContext_70)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_70)->lines_0 = lines_3;
    (&kernelContext_70)->geometry_0 = geometry_3;
    (&kernelContext_70)->shots_0 = shots_3;
    (&kernelContext_70)->lp_0 = lp_3;
    (&kernelContext_70)->ext_0 = ext_4;
    (&kernelContext_70)->lp_sides_0 = lp_sides_3;
    (&kernelContext_70)->engine_counters_0 = engine_counters_3;
    (&kernelContext_70)->engine_contacts_0 = engine_contacts_3;
    (&kernelContext_70)->cs_anchors_0 = cs_anchors_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_70)->g_scan_0 = &g_scan_3;
    uint _S517 = a_counters_1(&kernelContext_70);
    uint _S518 = atomic_load_explicit((&kernelContext_70)->atom_0+_S517, memory_order_relaxed);
    bool run_0 = _S518 != 0U;
    atomic_store_explicit((&kernelContext_70)->atom_0+_S517, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_70)->atom_0+(_S517 + 5U + (&kernelContext_70)->params_0->parity_0), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S519 = atomic_fetch_add_explicit((&kernelContext_70)->atom_0+(_S517 + 4U), 1U, memory_order_relaxed);
    }
    uint _S520;
    if(run_0)
    {
        uint _S521 = groups_of_0((&kernelContext_70)->params_0->chunk_count_0, 64U);
        _S520 = _S521;
    }
    else
    {
        _S520 = 0U;
    }
    set_args_0(0U, _S520, &kernelContext_70);
    if(run_0)
    {
        uint _S522 = groups_of_0((&kernelContext_70)->params_0->bond_count_0, 64U);
        _S520 = _S522;
    }
    else
    {
        _S520 = 0U;
    }
    set_args_0(1U, _S520, &kernelContext_70);
    if(run_0)
    {
        uint _S523 = groups_of_0((&kernelContext_70)->params_0->body_count_0, 64U);
        _S520 = _S523;
    }
    else
    {
        _S520 = 0U;
    }
    set_args_0(2U, _S520, &kernelContext_70);
    if(run_0)
    {
        uint _S524 = groups_of_0((&kernelContext_70)->params_0->chunk_count_0, 1024U);
        _S520 = _S524;
    }
    else
    {
        _S520 = 0U;
    }
    set_args_0(3U, _S520, &kernelContext_70);
    if(run_0)
    {
        uint _S525 = groups_of_0((&kernelContext_70)->params_0->body_count_0, 1024U);
        _S520 = _S525;
    }
    else
    {
        _S520 = 0U;
    }
    set_args_0(4U, _S520, &kernelContext_70);
    int _S526;
    if(run_0)
    {
        _S526 = int(1);
    }
    else
    {
        _S526 = int(0);
    }
    set_args_0(5U, uint(_S526), &kernelContext_70);
    return;
}

bool dirty_0(uint body_1, KernelContext_0 thread* kernelContext_71)
{
    uint _S527 = a_dirty_0(kernelContext_71);
    uint _S528 = atomic_load_explicit(kernelContext_71->atom_0+(_S527 + body_1), memory_order_relaxed);
    return _S528 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_72)
{
    return kernelContext_72->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_73)
{
    return 2U * kernelContext_73->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], packed_float4 device* lp_4 [[buffer(24)]], packed_float4 device* ext_5 [[buffer(23)]], packed_float4 device* lp_sides_4 [[buffer(28)]], uint device* engine_counters_4 [[buffer(26)]], packed_float4 device* engine_contacts_4 [[buffer(25)]], packed_float4 device* cs_anchors_4 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_74;
    (&kernelContext_74)->params_0 = params_4;
    (&kernelContext_74)->work_0 = work_4;
    (&kernelContext_74)->frames_0 = frames_4;
    (&kernelContext_74)->bodies_0 = bodies_4;
    (&kernelContext_74)->chunks_0 = chunks_4;
    (&kernelContext_74)->sort_0 = sort_4;
    (&kernelContext_74)->atom_0 = atom_4;
    (&kernelContext_74)->manifolds_0 = manifolds_4;
    (&kernelContext_74)->loads_0 = loads_4;
    (&kernelContext_74)->bonds_0 = bonds_4;
    (&kernelContext_74)->args_0 = args_4;
    (&kernelContext_74)->info_0 = info_4;
    (&kernelContext_74)->scan_0 = scan_4;
    (&kernelContext_74)->parts_0 = parts_4;
    (&kernelContext_74)->colliders_0 = colliders_4;
    (&kernelContext_74)->collider_info_0 = collider_info_4;
    (&kernelContext_74)->view_chunks_0 = view_chunks_4;
    (&kernelContext_74)->view_0 = view_4;
    (&kernelContext_74)->objects_0 = objects_4;
    (&kernelContext_74)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_74)->lines_0 = lines_4;
    (&kernelContext_74)->geometry_0 = geometry_4;
    (&kernelContext_74)->shots_0 = shots_4;
    (&kernelContext_74)->lp_0 = lp_4;
    (&kernelContext_74)->ext_0 = ext_5;
    (&kernelContext_74)->lp_sides_0 = lp_sides_4;
    (&kernelContext_74)->engine_counters_0 = engine_counters_4;
    (&kernelContext_74)->engine_contacts_0 = engine_contacts_4;
    (&kernelContext_74)->cs_anchors_0 = cs_anchors_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_74)->g_scan_0 = &g_scan_4;
    uint c_6 = id_3.x;
    bool _S529;
    if(c_6 >= (params_4->chunk_count_0))
    {
        _S529 = true;
    }
    else
    {
        uint _S530 = body_of_0(c_6, &kernelContext_74);
        bool _S531 = dirty_0(_S530, &kernelContext_74);
        _S529 = !_S531;
    }
    if(_S529)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_74)->atom_0+(a_parent_0() + c_6), c_6, memory_order_relaxed);
    uint _S532 = a_count_0(&kernelContext_74);
    atomic_store_explicit((&kernelContext_74)->atom_0+(_S532 + c_6), 0U, memory_order_relaxed);
    uint _S533 = a_anchor_0(&kernelContext_74);
    atomic_store_explicit((&kernelContext_74)->atom_0+(_S533 + c_6), 0U, memory_order_relaxed);
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
        uint _S534 = a_sums_0(&kernelContext_74);
        atomic_store_explicit((&kernelContext_74)->atom_0+(_S534 + 20U * c_6 + w_3), 0U, memory_order_relaxed);
        w_3 = w_3 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], packed_float4 device* lp_5 [[buffer(24)]], packed_float4 device* ext_6 [[buffer(23)]], packed_float4 device* lp_sides_5 [[buffer(28)]], uint device* engine_counters_5 [[buffer(26)]], packed_float4 device* engine_contacts_5 [[buffer(25)]], packed_float4 device* cs_anchors_5 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_75;
    (&kernelContext_75)->params_0 = params_5;
    (&kernelContext_75)->work_0 = work_5;
    (&kernelContext_75)->frames_0 = frames_5;
    (&kernelContext_75)->bodies_0 = bodies_5;
    (&kernelContext_75)->chunks_0 = chunks_5;
    (&kernelContext_75)->sort_0 = sort_5;
    (&kernelContext_75)->atom_0 = atom_5;
    (&kernelContext_75)->manifolds_0 = manifolds_5;
    (&kernelContext_75)->loads_0 = loads_5;
    (&kernelContext_75)->bonds_0 = bonds_5;
    (&kernelContext_75)->args_0 = args_5;
    (&kernelContext_75)->info_0 = info_5;
    (&kernelContext_75)->scan_0 = scan_5;
    (&kernelContext_75)->parts_0 = parts_5;
    (&kernelContext_75)->colliders_0 = colliders_5;
    (&kernelContext_75)->collider_info_0 = collider_info_5;
    (&kernelContext_75)->view_chunks_0 = view_chunks_5;
    (&kernelContext_75)->view_0 = view_5;
    (&kernelContext_75)->objects_0 = objects_5;
    (&kernelContext_75)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_75)->lines_0 = lines_5;
    (&kernelContext_75)->geometry_0 = geometry_5;
    (&kernelContext_75)->shots_0 = shots_5;
    (&kernelContext_75)->lp_0 = lp_5;
    (&kernelContext_75)->ext_0 = ext_6;
    (&kernelContext_75)->lp_sides_0 = lp_sides_5;
    (&kernelContext_75)->engine_counters_0 = engine_counters_5;
    (&kernelContext_75)->engine_contacts_0 = engine_contacts_5;
    (&kernelContext_75)->cs_anchors_0 = cs_anchors_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_75)->g_scan_0 = &g_scan_5;
    uint s_7 = id_4.x;
    bool _S535;
    if(s_7 >= (params_5->body_count_0))
    {
        _S535 = true;
    }
    else
    {
        bool _S536 = dirty_0(s_7, &kernelContext_75);
        _S535 = !_S536;
    }
    if(_S535)
    {
        return;
    }
    uint _S537 = a_best_count_0(&kernelContext_75);
    atomic_store_explicit((&kernelContext_75)->atom_0+(_S537 + s_7), 0U, memory_order_relaxed);
    uint _S538 = a_best_root_0(&kernelContext_75);
    atomic_store_explicit((&kernelContext_75)->atom_0+(_S538 + s_7), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_4, KernelContext_0 thread* kernelContext_76)
{
    uint _S539 = a_parent_0();
    uint _S540 = atomic_load_explicit(kernelContext_76->atom_0+(_S539 + x_4), memory_order_relaxed);
    uint p_11 = _S540;
    uint _S541 = x_4;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S542;
        if(guard_0 < 1048576U)
        {
            _S542 = p_11 != _S541;
        }
        else
        {
            _S542 = false;
        }
        if(_S542)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_76->atom_0+(_S539 + p_11), memory_order_relaxed);
        if(gp_0 != p_11)
        {
            uint _S543;
            {
            uint _metal_cas_comparand = p_11;
            atomic_compare_exchange_weak_explicit(kernelContext_76->atom_0+(_S539 + _S541), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S543 = _metal_cas_comparand;
            }
        }
        uint _S544 = guard_0 + 1U;
        _S541 = p_11;
        p_11 = gp_0;
        guard_0 = _S544;
    }
    return _S541;
}

[[kernel]] void fr_uf_union(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], packed_float4 device* lp_6 [[buffer(24)]], packed_float4 device* ext_7 [[buffer(23)]], packed_float4 device* lp_sides_6 [[buffer(28)]], uint device* engine_counters_6 [[buffer(26)]], packed_float4 device* engine_contacts_6 [[buffer(25)]], packed_float4 device* cs_anchors_6 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_77;
    (&kernelContext_77)->params_0 = params_6;
    (&kernelContext_77)->work_0 = work_6;
    (&kernelContext_77)->frames_0 = frames_6;
    (&kernelContext_77)->bodies_0 = bodies_6;
    (&kernelContext_77)->chunks_0 = chunks_6;
    (&kernelContext_77)->sort_0 = sort_6;
    (&kernelContext_77)->atom_0 = atom_6;
    (&kernelContext_77)->manifolds_0 = manifolds_6;
    (&kernelContext_77)->loads_0 = loads_6;
    (&kernelContext_77)->bonds_0 = bonds_6;
    (&kernelContext_77)->args_0 = args_6;
    (&kernelContext_77)->info_0 = info_6;
    (&kernelContext_77)->scan_0 = scan_6;
    (&kernelContext_77)->parts_0 = parts_6;
    (&kernelContext_77)->colliders_0 = colliders_6;
    (&kernelContext_77)->collider_info_0 = collider_info_6;
    (&kernelContext_77)->view_chunks_0 = view_chunks_6;
    (&kernelContext_77)->view_0 = view_6;
    (&kernelContext_77)->objects_0 = objects_6;
    (&kernelContext_77)->shot_bodies_0 = shot_bodies_6;
    (&kernelContext_77)->lines_0 = lines_6;
    (&kernelContext_77)->geometry_0 = geometry_6;
    (&kernelContext_77)->shots_0 = shots_6;
    (&kernelContext_77)->lp_0 = lp_6;
    (&kernelContext_77)->ext_0 = ext_7;
    (&kernelContext_77)->lp_sides_0 = lp_sides_6;
    (&kernelContext_77)->engine_counters_0 = engine_counters_6;
    (&kernelContext_77)->engine_contacts_0 = engine_contacts_6;
    (&kernelContext_77)->cs_anchors_0 = cs_anchors_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_77)->g_scan_0 = &g_scan_6;
    uint i_3 = id_5.x;
    bool _S545;
    if(i_3 >= (params_6->bond_count_0))
    {
        _S545 = true;
    }
    else
    {
        uint _S546 = w_alive_0(&kernelContext_77);
        _S545 = (*((&kernelContext_77)->work_0+(_S546 + i_3))) == 0U;
    }
    if(_S545)
    {
        return;
    }
    uint4 _S547 = uint4(*((&kernelContext_77)->bonds_0+i_3)) ;
    uint _S548 = _S547.x;
    uint _S549 = body_of_0(_S548, &kernelContext_77);
    bool _S550 = dirty_0(_S549, &kernelContext_77);
    if(!_S550)
    {
        return;
    }
    uint _S551 = _S547.y;
    uint a_5 = _S548;
    uint b_24 = _S551;
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
        uint _S552 = find_0(a_5, &kernelContext_77);
        uint _S553 = find_0(b_24, &kernelContext_77);
        if(_S552 == _S553)
        {
            return;
        }
        uint _S554 = max(_S552, _S553);
        uint _S555 = min(_S552, _S553);
        uint was_0;
        {
        uint _metal_cas_comparand = _S554;
        atomic_compare_exchange_weak_explicit((&kernelContext_77)->atom_0+(a_parent_0() + _S554), &_metal_cas_comparand, _S555, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S554)
        {
            return;
        }
        uint _S556 = guard_1 + 1U;
        a_5 = _S554;
        b_24 = _S555;
        guard_1 = _S556;
    }
    uint _S557 = a_counters_1(&kernelContext_77);
    uint _S558 = atomic_fetch_or_explicit((&kernelContext_77)->atom_0+(_S557 + 3U), 4U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], packed_float4 device* lp_7 [[buffer(24)]], packed_float4 device* ext_8 [[buffer(23)]], packed_float4 device* lp_sides_7 [[buffer(28)]], uint device* engine_counters_7 [[buffer(26)]], packed_float4 device* engine_contacts_7 [[buffer(25)]], packed_float4 device* cs_anchors_7 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_78;
    (&kernelContext_78)->params_0 = params_7;
    (&kernelContext_78)->work_0 = work_7;
    (&kernelContext_78)->frames_0 = frames_7;
    (&kernelContext_78)->bodies_0 = bodies_7;
    (&kernelContext_78)->chunks_0 = chunks_7;
    (&kernelContext_78)->sort_0 = sort_7;
    (&kernelContext_78)->atom_0 = atom_7;
    (&kernelContext_78)->manifolds_0 = manifolds_7;
    (&kernelContext_78)->loads_0 = loads_7;
    (&kernelContext_78)->bonds_0 = bonds_7;
    (&kernelContext_78)->args_0 = args_7;
    (&kernelContext_78)->info_0 = info_7;
    (&kernelContext_78)->scan_0 = scan_7;
    (&kernelContext_78)->parts_0 = parts_7;
    (&kernelContext_78)->colliders_0 = colliders_7;
    (&kernelContext_78)->collider_info_0 = collider_info_7;
    (&kernelContext_78)->view_chunks_0 = view_chunks_7;
    (&kernelContext_78)->view_0 = view_7;
    (&kernelContext_78)->objects_0 = objects_7;
    (&kernelContext_78)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_78)->lines_0 = lines_7;
    (&kernelContext_78)->geometry_0 = geometry_7;
    (&kernelContext_78)->shots_0 = shots_7;
    (&kernelContext_78)->lp_0 = lp_7;
    (&kernelContext_78)->ext_0 = ext_8;
    (&kernelContext_78)->lp_sides_0 = lp_sides_7;
    (&kernelContext_78)->engine_counters_0 = engine_counters_7;
    (&kernelContext_78)->engine_contacts_0 = engine_contacts_7;
    (&kernelContext_78)->cs_anchors_0 = cs_anchors_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_78)->g_scan_0 = &g_scan_7;
    uint c_7 = id_6.x;
    bool _S559;
    if(c_7 >= (params_7->chunk_count_0))
    {
        _S559 = true;
    }
    else
    {
        uint _S560 = body_of_0(c_7, &kernelContext_78);
        bool _S561 = dirty_0(_S560, &kernelContext_78);
        _S559 = !_S561;
    }
    if(_S559)
    {
        return;
    }
    uint _S562 = find_0(c_7, &kernelContext_78);
    uint _S563 = w_label_0(&kernelContext_78);
    *((&kernelContext_78)->work_0+(_S563 + c_7)) = _S562;
    uint _S564 = a_count_0(&kernelContext_78);
    uint _S565 = atomic_fetch_add_explicit((&kernelContext_78)->atom_0+(_S564 + _S562), 1U, memory_order_relaxed);
    if(((&kernelContext_78)->info_0[c_7] & 1U) != 0U)
    {
        uint _S566 = a_anchor_0(&kernelContext_78);
        uint _S567 = atomic_fetch_or_explicit((&kernelContext_78)->atom_0+(_S566 + _S562), 1U, memory_order_relaxed);
    }
    uint _S568 = 4U * c_7;
    float4 _S569 = float4(*((&kernelContext_78)->chunks_0+_S568)) ;
    float4 _S570 = float4(*((&kernelContext_78)->chunks_0+(_S568 + 1U))) ;
    float4 _S571 = float4(*((&kernelContext_78)->chunks_0+(_S568 + 2U))) ;
    float m_2 = _S569.w;
    float3 d_17 = _S569.xyz - (float4(*((&kernelContext_78)->chunks_0+4U * _S562)) ).xyz;
    float dd_0 = dot(d_17, d_17);
    uint _S572 = a_sums_0(&kernelContext_78);
    uint at_5 = _S572 + 20U * _S562;
    add64_0(at_5, m_2, (&kernelContext_78)->params_0->scale_m_0, &kernelContext_78);
    float _S573 = d_17.x;
    float _S574 = m_2 * _S573;
    add64_0(at_5 + 2U, _S574, (&kernelContext_78)->params_0->scale_s_0, &kernelContext_78);
    float _S575 = d_17.y;
    float _S576 = m_2 * _S575;
    add64_0(at_5 + 4U, _S576, (&kernelContext_78)->params_0->scale_s_0, &kernelContext_78);
    float _S577 = d_17.z;
    add64_0(at_5 + 6U, m_2 * _S577, (&kernelContext_78)->params_0->scale_s_0, &kernelContext_78);
    add64_0(at_5 + 8U, _S570.x + m_2 * (dd_0 - _S573 * _S573), (&kernelContext_78)->params_0->scale_j_0, &kernelContext_78);
    add64_0(at_5 + 10U, _S570.y + m_2 * (dd_0 - _S575 * _S575), (&kernelContext_78)->params_0->scale_j_0, &kernelContext_78);
    add64_0(at_5 + 12U, _S570.z + m_2 * (dd_0 - _S577 * _S577), (&kernelContext_78)->params_0->scale_j_0, &kernelContext_78);
    add64_0(at_5 + 14U, _S570.w - _S574 * _S575, (&kernelContext_78)->params_0->scale_j_0, &kernelContext_78);
    add64_0(at_5 + 16U, _S571.x - _S574 * _S577, (&kernelContext_78)->params_0->scale_j_0, &kernelContext_78);
    add64_0(at_5 + 18U, _S571.y - _S576 * _S577, (&kernelContext_78)->params_0->scale_j_0, &kernelContext_78);
    return;
}

bool is_root_0(uint c_8, KernelContext_0 thread* kernelContext_79)
{
    bool _S578;
    if(c_8 < (kernelContext_79->params_0->chunk_count_0))
    {
        uint _S579 = body_of_0(c_8, kernelContext_79);
        bool _S580 = dirty_0(_S579, kernelContext_79);
        _S578 = _S580;
    }
    else
    {
        _S578 = false;
    }
    if(_S578)
    {
        uint _S581 = w_label_0(kernelContext_79);
        _S578 = (*(kernelContext_79->work_0+(_S581 + c_8))) == c_8;
    }
    else
    {
        _S578 = false;
    }
    return _S578;
}

[[kernel]] void fr_best_count(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], packed_float4 device* lp_8 [[buffer(24)]], packed_float4 device* ext_9 [[buffer(23)]], packed_float4 device* lp_sides_8 [[buffer(28)]], uint device* engine_counters_8 [[buffer(26)]], packed_float4 device* engine_contacts_8 [[buffer(25)]], packed_float4 device* cs_anchors_8 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_80;
    (&kernelContext_80)->params_0 = params_8;
    (&kernelContext_80)->work_0 = work_8;
    (&kernelContext_80)->frames_0 = frames_8;
    (&kernelContext_80)->bodies_0 = bodies_8;
    (&kernelContext_80)->chunks_0 = chunks_8;
    (&kernelContext_80)->sort_0 = sort_8;
    (&kernelContext_80)->atom_0 = atom_8;
    (&kernelContext_80)->manifolds_0 = manifolds_8;
    (&kernelContext_80)->loads_0 = loads_8;
    (&kernelContext_80)->bonds_0 = bonds_8;
    (&kernelContext_80)->args_0 = args_8;
    (&kernelContext_80)->info_0 = info_8;
    (&kernelContext_80)->scan_0 = scan_8;
    (&kernelContext_80)->parts_0 = parts_8;
    (&kernelContext_80)->colliders_0 = colliders_8;
    (&kernelContext_80)->collider_info_0 = collider_info_8;
    (&kernelContext_80)->view_chunks_0 = view_chunks_8;
    (&kernelContext_80)->view_0 = view_8;
    (&kernelContext_80)->objects_0 = objects_8;
    (&kernelContext_80)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_80)->lines_0 = lines_8;
    (&kernelContext_80)->geometry_0 = geometry_8;
    (&kernelContext_80)->shots_0 = shots_8;
    (&kernelContext_80)->lp_0 = lp_8;
    (&kernelContext_80)->ext_0 = ext_9;
    (&kernelContext_80)->lp_sides_0 = lp_sides_8;
    (&kernelContext_80)->engine_counters_0 = engine_counters_8;
    (&kernelContext_80)->engine_contacts_0 = engine_contacts_8;
    (&kernelContext_80)->cs_anchors_0 = cs_anchors_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_80)->g_scan_0 = &g_scan_8;
    uint c_9 = id_7.x;
    bool _S582 = is_root_0(c_9, &kernelContext_80);
    if(!_S582)
    {
        return;
    }
    uint _S583 = a_best_count_0(&kernelContext_80);
    uint _S584 = body_of_0(c_9, &kernelContext_80);
    atomic<uint> device* _S585 = (&kernelContext_80)->atom_0+(_S583 + _S584);
    uint _S586 = a_count_0(&kernelContext_80);
    uint _S587 = atomic_load_explicit((&kernelContext_80)->atom_0+(_S586 + c_9), memory_order_relaxed);
    uint _S588 = atomic_fetch_max_explicit(_S585, _S587, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], packed_float4 device* lp_9 [[buffer(24)]], packed_float4 device* ext_10 [[buffer(23)]], packed_float4 device* lp_sides_9 [[buffer(28)]], uint device* engine_counters_9 [[buffer(26)]], packed_float4 device* engine_contacts_9 [[buffer(25)]], packed_float4 device* cs_anchors_9 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_81;
    (&kernelContext_81)->params_0 = params_9;
    (&kernelContext_81)->work_0 = work_9;
    (&kernelContext_81)->frames_0 = frames_9;
    (&kernelContext_81)->bodies_0 = bodies_9;
    (&kernelContext_81)->chunks_0 = chunks_9;
    (&kernelContext_81)->sort_0 = sort_9;
    (&kernelContext_81)->atom_0 = atom_9;
    (&kernelContext_81)->manifolds_0 = manifolds_9;
    (&kernelContext_81)->loads_0 = loads_9;
    (&kernelContext_81)->bonds_0 = bonds_9;
    (&kernelContext_81)->args_0 = args_9;
    (&kernelContext_81)->info_0 = info_9;
    (&kernelContext_81)->scan_0 = scan_9;
    (&kernelContext_81)->parts_0 = parts_9;
    (&kernelContext_81)->colliders_0 = colliders_9;
    (&kernelContext_81)->collider_info_0 = collider_info_9;
    (&kernelContext_81)->view_chunks_0 = view_chunks_9;
    (&kernelContext_81)->view_0 = view_9;
    (&kernelContext_81)->objects_0 = objects_9;
    (&kernelContext_81)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_81)->lines_0 = lines_9;
    (&kernelContext_81)->geometry_0 = geometry_9;
    (&kernelContext_81)->shots_0 = shots_9;
    (&kernelContext_81)->lp_0 = lp_9;
    (&kernelContext_81)->ext_0 = ext_10;
    (&kernelContext_81)->lp_sides_0 = lp_sides_9;
    (&kernelContext_81)->engine_counters_0 = engine_counters_9;
    (&kernelContext_81)->engine_contacts_0 = engine_contacts_9;
    (&kernelContext_81)->cs_anchors_0 = cs_anchors_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_81)->g_scan_0 = &g_scan_9;
    uint c_10 = id_8.x;
    bool _S589 = is_root_0(c_10, &kernelContext_81);
    if(!_S589)
    {
        return;
    }
    uint _S590 = body_of_0(c_10, &kernelContext_81);
    uint _S591 = a_count_0(&kernelContext_81);
    uint _S592 = atomic_load_explicit((&kernelContext_81)->atom_0+(_S591 + c_10), memory_order_relaxed);
    uint _S593 = a_best_count_0(&kernelContext_81);
    uint _S594 = atomic_load_explicit((&kernelContext_81)->atom_0+(_S593 + _S590), memory_order_relaxed);
    if(_S592 == _S594)
    {
        uint _S595 = a_best_root_0(&kernelContext_81);
        uint _S596 = atomic_fetch_min_explicit((&kernelContext_81)->atom_0+(_S595 + _S590), c_10, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], packed_float4 device* lp_10 [[buffer(24)]], packed_float4 device* ext_11 [[buffer(23)]], packed_float4 device* lp_sides_10 [[buffer(28)]], uint device* engine_counters_10 [[buffer(26)]], packed_float4 device* engine_contacts_10 [[buffer(25)]], packed_float4 device* cs_anchors_10 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_82;
    (&kernelContext_82)->params_0 = params_10;
    (&kernelContext_82)->work_0 = work_10;
    (&kernelContext_82)->frames_0 = frames_10;
    (&kernelContext_82)->bodies_0 = bodies_10;
    (&kernelContext_82)->chunks_0 = chunks_10;
    (&kernelContext_82)->sort_0 = sort_10;
    (&kernelContext_82)->atom_0 = atom_10;
    (&kernelContext_82)->manifolds_0 = manifolds_10;
    (&kernelContext_82)->loads_0 = loads_10;
    (&kernelContext_82)->bonds_0 = bonds_10;
    (&kernelContext_82)->args_0 = args_10;
    (&kernelContext_82)->info_0 = info_10;
    (&kernelContext_82)->scan_0 = scan_10;
    (&kernelContext_82)->parts_0 = parts_10;
    (&kernelContext_82)->colliders_0 = colliders_10;
    (&kernelContext_82)->collider_info_0 = collider_info_10;
    (&kernelContext_82)->view_chunks_0 = view_chunks_10;
    (&kernelContext_82)->view_0 = view_10;
    (&kernelContext_82)->objects_0 = objects_10;
    (&kernelContext_82)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_82)->lines_0 = lines_10;
    (&kernelContext_82)->geometry_0 = geometry_10;
    (&kernelContext_82)->shots_0 = shots_10;
    (&kernelContext_82)->lp_0 = lp_10;
    (&kernelContext_82)->ext_0 = ext_11;
    (&kernelContext_82)->lp_sides_0 = lp_sides_10;
    (&kernelContext_82)->engine_counters_0 = engine_counters_10;
    (&kernelContext_82)->engine_contacts_0 = engine_contacts_10;
    (&kernelContext_82)->cs_anchors_0 = cs_anchors_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_82)->g_scan_0 = &g_scan_10;
    uint c_11 = id_9.x;
    if(c_11 >= (params_10->chunk_count_0))
    {
        return;
    }
    uint _S597 = w_new_0(&kernelContext_82);
    uint device* _S598 = (&kernelContext_82)->work_0+(_S597 + c_11);
    bool _S599 = is_root_0(c_11, &kernelContext_82);
    bool _S600;
    if(_S599)
    {
        uint _S601 = a_best_root_0(&kernelContext_82);
        uint _S602 = body_of_0(c_11, &kernelContext_82);
        uint _S603 = atomic_load_explicit((&kernelContext_82)->atom_0+(_S601 + _S602), memory_order_relaxed);
        _S600 = _S603 != c_11;
    }
    else
    {
        _S600 = false;
    }
    uint _S604;
    if(_S600)
    {
        _S604 = 1U;
    }
    else
    {
        _S604 = 0U;
    }
    *_S598 = _S604;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], packed_float4 device* lp_11 [[buffer(24)]], packed_float4 device* ext_12 [[buffer(23)]], packed_float4 device* lp_sides_11 [[buffer(28)]], uint device* engine_counters_11 [[buffer(26)]], packed_float4 device* engine_contacts_11 [[buffer(25)]], packed_float4 device* cs_anchors_11 [[buffer(27)]])
{
    uint step_2;
    uint add_0;
    thread KernelContext_0 kernelContext_83;
    (&kernelContext_83)->params_0 = params_11;
    (&kernelContext_83)->work_0 = work_11;
    (&kernelContext_83)->frames_0 = frames_11;
    (&kernelContext_83)->bodies_0 = bodies_11;
    (&kernelContext_83)->chunks_0 = chunks_11;
    (&kernelContext_83)->sort_0 = sort_11;
    (&kernelContext_83)->atom_0 = atom_11;
    (&kernelContext_83)->manifolds_0 = manifolds_11;
    (&kernelContext_83)->loads_0 = loads_11;
    (&kernelContext_83)->bonds_0 = bonds_11;
    (&kernelContext_83)->args_0 = args_11;
    (&kernelContext_83)->info_0 = info_11;
    (&kernelContext_83)->scan_0 = scan_11;
    (&kernelContext_83)->parts_0 = parts_11;
    (&kernelContext_83)->colliders_0 = colliders_11;
    (&kernelContext_83)->collider_info_0 = collider_info_11;
    (&kernelContext_83)->view_chunks_0 = view_chunks_11;
    (&kernelContext_83)->view_0 = view_11;
    (&kernelContext_83)->objects_0 = objects_11;
    (&kernelContext_83)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_83)->lines_0 = lines_11;
    (&kernelContext_83)->geometry_0 = geometry_11;
    (&kernelContext_83)->shots_0 = shots_11;
    (&kernelContext_83)->lp_0 = lp_11;
    (&kernelContext_83)->ext_0 = ext_12;
    (&kernelContext_83)->lp_sides_0 = lp_sides_11;
    (&kernelContext_83)->engine_counters_0 = engine_counters_11;
    (&kernelContext_83)->engine_contacts_0 = engine_contacts_11;
    (&kernelContext_83)->cs_anchors_0 = cs_anchors_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_83)->g_scan_0 = &g_scan_11;
    uint t_1 = thread_0.x;
    uint _S605 = group_0.x;
    uint _S606 = 1024U * _S605 + 4U * t_1;
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
        if((_S606 + j_3) < ((&kernelContext_83)->scan_0->n_0))
        {
            step_2 = *((&kernelContext_83)->work_0+((&kernelContext_83)->scan_0->in_base_0 + _S606 + j_3));
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
    (*(&kernelContext_83)->g_scan_0)[t_1] = sum_2;
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
            add_0 = (*(&kernelContext_83)->g_scan_0)[t_1 - step_2];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_83)->g_scan_0)[t_1] = (*(&kernelContext_83)->g_scan_0)[t_1] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_2 = step_2 << 1U;
    }
    uint _S607 = (*(&kernelContext_83)->g_scan_0)[t_1] - sum_2;
    j_3 = 0U;
    add_0 = _S607;
    for(;;)
    {
        if(j_3 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S606 + j_3) < ((&kernelContext_83)->scan_0->n_0))
        {
            *((&kernelContext_83)->work_0+((&kernelContext_83)->scan_0->out_base_0 + _S606 + j_3)) = add_0;
        }
        uint run_1 = add_0 + v_10[j_3];
        j_3 = j_3 + 1U;
        add_0 = run_1;
    }
    if(t_1 == 255U)
    {
        *((&kernelContext_83)->work_0+((&kernelContext_83)->scan_0->block_base_0 + _S605)) = (*(&kernelContext_83)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], packed_float4 device* lp_12 [[buffer(24)]], packed_float4 device* ext_13 [[buffer(23)]], packed_float4 device* lp_sides_12 [[buffer(28)]], uint device* engine_counters_12 [[buffer(26)]], packed_float4 device* engine_contacts_12 [[buffer(25)]], packed_float4 device* cs_anchors_12 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_84;
    (&kernelContext_84)->params_0 = params_12;
    (&kernelContext_84)->work_0 = work_12;
    (&kernelContext_84)->frames_0 = frames_12;
    (&kernelContext_84)->bodies_0 = bodies_12;
    (&kernelContext_84)->chunks_0 = chunks_12;
    (&kernelContext_84)->sort_0 = sort_12;
    (&kernelContext_84)->atom_0 = atom_12;
    (&kernelContext_84)->manifolds_0 = manifolds_12;
    (&kernelContext_84)->loads_0 = loads_12;
    (&kernelContext_84)->bonds_0 = bonds_12;
    (&kernelContext_84)->args_0 = args_12;
    (&kernelContext_84)->info_0 = info_12;
    (&kernelContext_84)->scan_0 = scan_12;
    (&kernelContext_84)->parts_0 = parts_12;
    (&kernelContext_84)->colliders_0 = colliders_12;
    (&kernelContext_84)->collider_info_0 = collider_info_12;
    (&kernelContext_84)->view_chunks_0 = view_chunks_12;
    (&kernelContext_84)->view_0 = view_12;
    (&kernelContext_84)->objects_0 = objects_12;
    (&kernelContext_84)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_84)->lines_0 = lines_12;
    (&kernelContext_84)->geometry_0 = geometry_12;
    (&kernelContext_84)->shots_0 = shots_12;
    (&kernelContext_84)->lp_0 = lp_12;
    (&kernelContext_84)->ext_0 = ext_13;
    (&kernelContext_84)->lp_sides_0 = lp_sides_12;
    (&kernelContext_84)->engine_counters_0 = engine_counters_12;
    (&kernelContext_84)->engine_contacts_0 = engine_contacts_12;
    (&kernelContext_84)->cs_anchors_0 = cs_anchors_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_84)->g_scan_0 = &g_scan_12;
    uint t_2 = thread_1.x;
    uint blocks_0 = (scan_12->n_0 + 1023U) / 1024U;
    bool _S608 = t_2 < blocks_0;
    uint v_11;
    if(_S608)
    {
        v_11 = *((&kernelContext_84)->work_0+((&kernelContext_84)->scan_0->block_base_0 + t_2));
    }
    else
    {
        v_11 = 0U;
    }
    (*(&kernelContext_84)->g_scan_0)[t_2] = v_11;
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
            add_1 = (*(&kernelContext_84)->g_scan_0)[t_2 - step_3];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_84)->g_scan_0)[t_2] = (*(&kernelContext_84)->g_scan_0)[t_2] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_3 = step_3 << 1U;
    }
    if(_S608)
    {
        *((&kernelContext_84)->work_0+((&kernelContext_84)->scan_0->block_base_0 + t_2)) = (*(&kernelContext_84)->g_scan_0)[t_2] - v_11;
    }
    if(t_2 == 1023U)
    {
        *((&kernelContext_84)->work_0+((&kernelContext_84)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_84)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], packed_float4 device* lp_13 [[buffer(24)]], packed_float4 device* ext_14 [[buffer(23)]], packed_float4 device* lp_sides_13 [[buffer(28)]], uint device* engine_counters_13 [[buffer(26)]], packed_float4 device* engine_contacts_13 [[buffer(25)]], packed_float4 device* cs_anchors_13 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_85;
    (&kernelContext_85)->params_0 = params_13;
    (&kernelContext_85)->work_0 = work_13;
    (&kernelContext_85)->frames_0 = frames_13;
    (&kernelContext_85)->bodies_0 = bodies_13;
    (&kernelContext_85)->chunks_0 = chunks_13;
    (&kernelContext_85)->sort_0 = sort_13;
    (&kernelContext_85)->atom_0 = atom_13;
    (&kernelContext_85)->manifolds_0 = manifolds_13;
    (&kernelContext_85)->loads_0 = loads_13;
    (&kernelContext_85)->bonds_0 = bonds_13;
    (&kernelContext_85)->args_0 = args_13;
    (&kernelContext_85)->info_0 = info_13;
    (&kernelContext_85)->scan_0 = scan_13;
    (&kernelContext_85)->parts_0 = parts_13;
    (&kernelContext_85)->colliders_0 = colliders_13;
    (&kernelContext_85)->collider_info_0 = collider_info_13;
    (&kernelContext_85)->view_chunks_0 = view_chunks_13;
    (&kernelContext_85)->view_0 = view_13;
    (&kernelContext_85)->objects_0 = objects_13;
    (&kernelContext_85)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_85)->lines_0 = lines_13;
    (&kernelContext_85)->geometry_0 = geometry_13;
    (&kernelContext_85)->shots_0 = shots_13;
    (&kernelContext_85)->lp_0 = lp_13;
    (&kernelContext_85)->ext_0 = ext_14;
    (&kernelContext_85)->lp_sides_0 = lp_sides_13;
    (&kernelContext_85)->engine_counters_0 = engine_counters_13;
    (&kernelContext_85)->engine_contacts_0 = engine_contacts_13;
    (&kernelContext_85)->cs_anchors_0 = cs_anchors_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_85)->g_scan_0 = &g_scan_13;
    uint i_4 = id_10.x;
    if(i_4 >= (scan_13->n_0))
    {
        return;
    }
    uint device* _S609 = (&kernelContext_85)->work_0+((&kernelContext_85)->scan_0->out_base_0 + i_4);
    *_S609 = *_S609 + *((&kernelContext_85)->work_0+((&kernelContext_85)->scan_0->block_base_0 + i_4 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], packed_float4 device* lp_14 [[buffer(24)]], packed_float4 device* ext_15 [[buffer(23)]], packed_float4 device* lp_sides_14 [[buffer(28)]], uint device* engine_counters_14 [[buffer(26)]], packed_float4 device* engine_contacts_14 [[buffer(25)]], packed_float4 device* cs_anchors_14 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_86;
    (&kernelContext_86)->params_0 = params_14;
    (&kernelContext_86)->work_0 = work_14;
    (&kernelContext_86)->frames_0 = frames_14;
    (&kernelContext_86)->bodies_0 = bodies_14;
    (&kernelContext_86)->chunks_0 = chunks_14;
    (&kernelContext_86)->sort_0 = sort_14;
    (&kernelContext_86)->atom_0 = atom_14;
    (&kernelContext_86)->manifolds_0 = manifolds_14;
    (&kernelContext_86)->loads_0 = loads_14;
    (&kernelContext_86)->bonds_0 = bonds_14;
    (&kernelContext_86)->args_0 = args_14;
    (&kernelContext_86)->info_0 = info_14;
    (&kernelContext_86)->scan_0 = scan_14;
    (&kernelContext_86)->parts_0 = parts_14;
    (&kernelContext_86)->colliders_0 = colliders_14;
    (&kernelContext_86)->collider_info_0 = collider_info_14;
    (&kernelContext_86)->view_chunks_0 = view_chunks_14;
    (&kernelContext_86)->view_0 = view_14;
    (&kernelContext_86)->objects_0 = objects_14;
    (&kernelContext_86)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_86)->lines_0 = lines_14;
    (&kernelContext_86)->geometry_0 = geometry_14;
    (&kernelContext_86)->shots_0 = shots_14;
    (&kernelContext_86)->lp_0 = lp_14;
    (&kernelContext_86)->ext_0 = ext_15;
    (&kernelContext_86)->lp_sides_0 = lp_sides_14;
    (&kernelContext_86)->engine_counters_0 = engine_counters_14;
    (&kernelContext_86)->engine_contacts_0 = engine_contacts_14;
    (&kernelContext_86)->cs_anchors_0 = cs_anchors_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_86)->g_scan_0 = &g_scan_14;
    uint s_8 = id_11.x;
    bool _S610;
    if(s_8 >= (params_14->body_count_0))
    {
        _S610 = true;
    }
    else
    {
        uint _S611 = w_free_0(&kernelContext_86);
        _S610 = (*((&kernelContext_86)->work_0+(_S611 + s_8))) == 0U;
    }
    if(_S610)
    {
        return;
    }
    uint _S612 = w_free_list_0(&kernelContext_86);
    uint _S613 = w_free_rank_0(&kernelContext_86);
    *((&kernelContext_86)->work_0+(_S612 + *((&kernelContext_86)->work_0+(_S613 + s_8)))) = s_8;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_87)
{
    uint _S614 = w_free_list_0(kernelContext_87);
    return _S614 + kernelContext_87->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_88)
{
    uint _S615 = scan_new_blocks_0(kernelContext_88);
    return _S615 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_89)
{
    uint blocks_1 = (kernelContext_89->params_0->body_count_0 + 1023U) / 1024U;
    uint _S616 = scan_free_blocks_0(kernelContext_89);
    return *(kernelContext_89->work_0+(_S616 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], packed_float4 device* lp_15 [[buffer(24)]], packed_float4 device* ext_16 [[buffer(23)]], packed_float4 device* lp_sides_15 [[buffer(28)]], uint device* engine_counters_15 [[buffer(26)]], packed_float4 device* engine_contacts_15 [[buffer(25)]], packed_float4 device* cs_anchors_15 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_90;
    (&kernelContext_90)->params_0 = params_15;
    (&kernelContext_90)->work_0 = work_15;
    (&kernelContext_90)->frames_0 = frames_15;
    (&kernelContext_90)->bodies_0 = bodies_15;
    (&kernelContext_90)->chunks_0 = chunks_15;
    (&kernelContext_90)->sort_0 = sort_15;
    (&kernelContext_90)->atom_0 = atom_15;
    (&kernelContext_90)->manifolds_0 = manifolds_15;
    (&kernelContext_90)->loads_0 = loads_15;
    (&kernelContext_90)->bonds_0 = bonds_15;
    (&kernelContext_90)->args_0 = args_15;
    (&kernelContext_90)->info_0 = info_15;
    (&kernelContext_90)->scan_0 = scan_15;
    (&kernelContext_90)->parts_0 = parts_15;
    (&kernelContext_90)->colliders_0 = colliders_15;
    (&kernelContext_90)->collider_info_0 = collider_info_15;
    (&kernelContext_90)->view_chunks_0 = view_chunks_15;
    (&kernelContext_90)->view_0 = view_15;
    (&kernelContext_90)->objects_0 = objects_15;
    (&kernelContext_90)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_90)->lines_0 = lines_15;
    (&kernelContext_90)->geometry_0 = geometry_15;
    (&kernelContext_90)->shots_0 = shots_15;
    (&kernelContext_90)->lp_0 = lp_15;
    (&kernelContext_90)->ext_0 = ext_16;
    (&kernelContext_90)->lp_sides_0 = lp_sides_15;
    (&kernelContext_90)->engine_counters_0 = engine_counters_15;
    (&kernelContext_90)->engine_contacts_0 = engine_contacts_15;
    (&kernelContext_90)->cs_anchors_0 = cs_anchors_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_90)->g_scan_0 = &g_scan_15;
    uint c_12 = id_12.x;
    bool _S617 = is_root_0(c_12, &kernelContext_90);
    if(!_S617)
    {
        return;
    }
    uint _S618 = body_of_0(c_12, &kernelContext_90);
    uint _S619 = a_best_root_0(&kernelContext_90);
    uint _S620 = atomic_load_explicit((&kernelContext_90)->atom_0+(_S619 + _S618), memory_order_relaxed);
    uint slot_1;
    if(_S620 != c_12)
    {
        uint _S621 = w_rank_0(&kernelContext_90);
        uint device* _S622 = (&kernelContext_90)->work_0+(_S621 + c_12);
        uint k_18 = *_S622;
        uint _S623 = free_total_0(&kernelContext_90);
        if((*_S622) < _S623)
        {
            uint _S624 = w_free_list_0(&kernelContext_90);
            uint device* _S625 = (&kernelContext_90)->work_0+(_S624 + k_18);
            uint slot_2 = *_S625;
            uint _S626 = w_free_0(&kernelContext_90);
            *((&kernelContext_90)->work_0+(_S626 + *_S625)) = 0U;
            uint _S627 = a_counters_1(&kernelContext_90);
            uint _S628 = atomic_fetch_add_explicit((&kernelContext_90)->atom_0+(_S627 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S629 = a_counters_1(&kernelContext_90);
            uint _S630 = atomic_fetch_or_explicit((&kernelContext_90)->atom_0+(_S629 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S618;
        }
    }
    else
    {
        slot_1 = _S618;
    }
    uint _S631 = w_slot_0(&kernelContext_90);
    *((&kernelContext_90)->work_0+(_S631 + c_12)) = slot_1;
    return;
}

float read64_1(uint at_6, float scale_2, KernelContext_0 thread* kernelContext_91)
{
    uint _S632 = atomic_load_explicit(kernelContext_91->atom_0+(at_6 + 1U), memory_order_relaxed);
    int hi_2 = (as_type<int>((_S632)));
    uint lo_2 = atomic_load_explicit(kernelContext_91->atom_0+at_6, memory_order_relaxed);
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
    thread Quat_0 q_11;
    if(m22_0 <= 0.0f)
    {
        float dif10_0 = m11_0 - m00_0;
        float omm22_0 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            float f_4 = omm22_0 - dif10_0;
            float inv_0 = 0.5f / sqrt(f_4);
            (&q_11)->x_0 = f_4 * inv_0;
            (&q_11)->y_0 = (m01_0 + m10_0) * inv_0;
            (&q_11)->z_0 = (m02_0 + m20_0) * inv_0;
            (&q_11)->w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            float f_5 = omm22_0 + dif10_0;
            float inv_1 = 0.5f / sqrt(f_5);
            (&q_11)->x_0 = (m01_0 + m10_0) * inv_1;
            (&q_11)->y_0 = f_5 * inv_1;
            (&q_11)->z_0 = (m12_0 + m21_0) * inv_1;
            (&q_11)->w_0 = (m20_0 - m02_0) * inv_1;
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
            (&q_11)->x_0 = (m02_0 + m20_0) * inv_2;
            (&q_11)->y_0 = (m12_0 + m21_0) * inv_2;
            (&q_11)->z_0 = f_6 * inv_2;
            (&q_11)->w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            float f_7 = opm22_0 + sum10_0;
            float inv_3 = 0.5f / sqrt(f_7);
            (&q_11)->x_0 = (m12_0 - m21_0) * inv_3;
            (&q_11)->y_0 = (m20_0 - m02_0) * inv_3;
            (&q_11)->z_0 = (m01_0 - m10_0) * inv_3;
            (&q_11)->w_0 = f_7 * inv_3;
        }
    }
    return q_11;
}

void principal_axes_0(float3 r0_0, float3 r1_0, float3 r2_2, Quat_0 thread* rot_0, float3 thread* moments_0)
{
    array<float, int(3)> _S633 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S634 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S635 = { { r2_2.x, r2_2.y, r2_2.z } };
    thread array<array<float, int(3)>, int(3)> a_6;
    a_6[int(0)] = _S633;
    a_6[int(1)] = _S634;
    a_6[int(2)] = _S635;
    array<float, int(3)> _S636 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S637 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S638 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_12;
    v_12[int(0)] = _S636;
    v_12[int(1)] = _S637;
    v_12[int(2)] = _S638;
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
        uint q_12;
        if((abs(a_6[int(0)][int(2)])) > (abs(a_6[0U][1U])))
        {
            q_12 = 2U;
        }
        else
        {
            q_12 = 1U;
        }
        uint p_12;
        uint q_13;
        if((abs(a_6[int(1)][int(2)])) > (abs(a_6[0U][q_12])))
        {
            p_12 = 1U;
            q_13 = 2U;
        }
        else
        {
            p_12 = 0U;
            q_13 = q_12;
        }
        if((abs(a_6[p_12][q_13])) <= (9.99999982451670044e-15f * max(max(abs(a_6[int(0)][int(0)]), abs(a_6[int(1)][int(1)])), max(abs(a_6[int(2)][int(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        float theta_0 = 0.5f * atan2(2.0f * a_6[p_12][q_13], a_6[q_13][q_13] - a_6[p_12][p_12]);
        float s_9 = sin(theta_0);
        float c_13 = cos(theta_0);
        float app_0 = a_6[p_12][p_12];
        float aqq_0 = a_6[q_13][q_13];
        float apq_0 = a_6[p_12][q_13];
        uint k_19 = 0U;
        for(;;)
        {
            if(k_19 < 3U)
            {
            }
            else
            {
                break;
            }
            bool _S639;
            if(k_19 != p_12)
            {
                _S639 = k_19 != q_13;
            }
            else
            {
                _S639 = false;
            }
            if(_S639)
            {
                float kp_0 = a_6[k_19][p_12];
                float kq_0 = a_6[k_19][q_13];
                float _S640 = c_13 * a_6[k_19][p_12] - s_9 * a_6[k_19][q_13];
                a_6[k_19][p_12] = _S640;
                a_6[p_12][k_19] = _S640;
                float _S641 = s_9 * kp_0 + c_13 * kq_0;
                a_6[k_19][q_13] = _S641;
                a_6[q_13][k_19] = _S641;
            }
            k_19 = k_19 + 1U;
        }
        float _S642 = c_13 * c_13;
        float _S643 = 2.0f * s_9 * c_13 * apq_0;
        float _S644 = s_9 * s_9;
        a_6[p_12][p_12] = _S642 * app_0 - _S643 + _S644 * aqq_0;
        a_6[q_13][q_13] = _S644 * app_0 + _S643 + _S642 * aqq_0;
        a_6[p_12][q_13] = 0.0f;
        a_6[q_13][p_12] = 0.0f;
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
            float kq_1 = v_12[row_0][q_13];
            v_12[row_0][p_12] = c_13 * v_12[row_0][p_12] - s_9 * v_12[row_0][q_13];
            v_12[row_0][q_13] = s_9 * kp_1 + c_13 * kq_1;
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
    thread Quat_0 _S645 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S646 = normalized_0(&_S645);
    *rot_0 = _S646;
    *moments_0 = float3(a_6[int(0)][int(0)], a_6[int(1)][int(1)], a_6[int(2)][int(2)]);
    return;
}

Quat_0 rot_to_rubble_0(const Quat_0 thread* q_14)
{
    thread Quat_0 axes_1;
    (&axes_1)->w_0 = 0.70710676908493042f;
    (&axes_1)->x_0 = -0.70710676908493042f;
    (&axes_1)->y_0 = 0.0f;
    (&axes_1)->z_0 = 0.0f;
    thread Quat_0 _S647 = axes_1;
    Quat_0 _S648 = quat_mul_0(&_S647, q_14);
    thread Quat_0 _S649 = _S648;
    Quat_0 _S650 = normalized_0(&_S649);
    return _S650;
}

uint stage_0(uint c_14, KernelContext_0 thread* kernelContext_92)
{
    return 2U * kernelContext_92->params_0->body_count_0 + 9U * c_14;
}

float4 quat_vec_0(const Quat_0 thread* q_15)
{
    return float4(q_15->x_0, q_15->y_0, q_15->z_0, q_15->w_0);
}

[[kernel]] void fr_piece(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], packed_float4 device* lp_16 [[buffer(24)]], packed_float4 device* ext_17 [[buffer(23)]], packed_float4 device* lp_sides_16 [[buffer(28)]], uint device* engine_counters_16 [[buffer(26)]], packed_float4 device* engine_contacts_16 [[buffer(25)]], packed_float4 device* cs_anchors_16 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_93;
    (&kernelContext_93)->params_0 = params_16;
    (&kernelContext_93)->work_0 = work_16;
    (&kernelContext_93)->frames_0 = frames_16;
    (&kernelContext_93)->bodies_0 = bodies_16;
    (&kernelContext_93)->chunks_0 = chunks_16;
    (&kernelContext_93)->sort_0 = sort_16;
    (&kernelContext_93)->atom_0 = atom_16;
    (&kernelContext_93)->manifolds_0 = manifolds_16;
    (&kernelContext_93)->loads_0 = loads_16;
    (&kernelContext_93)->bonds_0 = bonds_16;
    (&kernelContext_93)->args_0 = args_16;
    (&kernelContext_93)->info_0 = info_16;
    (&kernelContext_93)->scan_0 = scan_16;
    (&kernelContext_93)->parts_0 = parts_16;
    (&kernelContext_93)->colliders_0 = colliders_16;
    (&kernelContext_93)->collider_info_0 = collider_info_16;
    (&kernelContext_93)->view_chunks_0 = view_chunks_16;
    (&kernelContext_93)->view_0 = view_16;
    (&kernelContext_93)->objects_0 = objects_16;
    (&kernelContext_93)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_93)->lines_0 = lines_16;
    (&kernelContext_93)->geometry_0 = geometry_16;
    (&kernelContext_93)->shots_0 = shots_16;
    (&kernelContext_93)->lp_0 = lp_16;
    (&kernelContext_93)->ext_0 = ext_17;
    (&kernelContext_93)->lp_sides_0 = lp_sides_16;
    (&kernelContext_93)->engine_counters_0 = engine_counters_16;
    (&kernelContext_93)->engine_contacts_0 = engine_contacts_16;
    (&kernelContext_93)->cs_anchors_0 = cs_anchors_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_93)->g_scan_0 = &g_scan_16;
    uint c_15 = id_13.x;
    bool _S651 = is_root_0(c_15, &kernelContext_93);
    if(!_S651)
    {
        return;
    }
    uint _S652 = body_of_0(c_15, &kernelContext_93);
    uint pb_0 = 7U * _S652;
    uint _S653 = a_sums_0(&kernelContext_93);
    uint at_7 = _S653 + 20U * c_15;
    float _S654 = read64_0(at_7, (&kernelContext_93)->params_0->scale_m_0, &kernelContext_93);
    float _S655 = read64_0(at_7 + 2U, (&kernelContext_93)->params_0->scale_s_0, &kernelContext_93);
    float _S656 = read64_0(at_7 + 4U, (&kernelContext_93)->params_0->scale_s_0, &kernelContext_93);
    float _S657 = read64_0(at_7 + 6U, (&kernelContext_93)->params_0->scale_s_0, &kernelContext_93);
    float3 d_18 = float3(_S655, _S656, _S657) / float3(max(_S654, 1.00000000317107685e-30f)) ;
    float _S658 = read64_0(at_7 + 8U, (&kernelContext_93)->params_0->scale_j_0, &kernelContext_93);
    float _S659 = read64_0(at_7 + 10U, (&kernelContext_93)->params_0->scale_j_0, &kernelContext_93);
    float _S660 = read64_0(at_7 + 12U, (&kernelContext_93)->params_0->scale_j_0, &kernelContext_93);
    float _S661 = read64_0(at_7 + 14U, (&kernelContext_93)->params_0->scale_j_0, &kernelContext_93);
    float _S662 = read64_0(at_7 + 16U, (&kernelContext_93)->params_0->scale_j_0, &kernelContext_93);
    float _S663 = read64_0(at_7 + 18U, (&kernelContext_93)->params_0->scale_j_0, &kernelContext_93);
    float dd_1 = dot(d_18, d_18);
    float _S664 = d_18.x;
    float _S665 = _S654 * _S664;
    float _S666 = d_18.y;
    float _S667 = _S661 + _S665 * _S666;
    float _S668 = d_18.z;
    float _S669 = _S662 + _S665 * _S668;
    float _S670 = _S663 + _S654 * _S666 * _S668;
    float3 com_1 = (float4(*((&kernelContext_93)->chunks_0+4U * c_15)) ).xyz + d_18;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S658 - _S654 * (dd_1 - _S664 * _S664), _S667, _S669), float3(_S667, _S659 - _S654 * (dd_1 - _S666 * _S666), _S670), float3(_S669, _S670, _S660 - _S654 * (dd_1 - _S668 * _S668)), &principal_0, &moments_1);
    uint _S671 = a_anchor_0(&kernelContext_93);
    uint _S672 = atomic_load_explicit((&kernelContext_93)->atom_0+(_S671 + c_15), memory_order_relaxed);
    bool anchored_0 = _S672 != 0U;
    thread Quat_0 q_16;
    thread float3 origin_4;
    structure_pose_0(_S652, pb_0, &q_16, &origin_4, &kernelContext_93);
    float3 p_parent_0 = (float4(*((&kernelContext_93)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_93)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_93)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S673 = q_16;
    float3 _S674 = rotate_0(&_S673, com_1);
    float3 pos_0 = to_rubble_0(origin_4 + _S674);
    thread Quat_0 _S675 = q_16;
    thread Quat_0 _S676 = principal_0;
    Quat_0 _S677 = quat_mul_0(&_S675, &_S676);
    thread Quat_0 _S678 = _S677;
    Quat_0 _S679 = normalized_0(&_S678);
    thread Quat_0 _S680 = _S679;
    Quat_0 _S681 = rot_to_rubble_0(&_S680);
    bool _S682;
    if(anchored_0)
    {
        _S682 = true;
    }
    else
    {
        _S682 = !(_S654 > 0.0f);
    }
    float inv_mass_3;
    if(_S682)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / _S654;
    }
    float3 _S683 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S684;
        if((moments_1.x) > 0.0f)
        {
            _S684 = 1.0f / moments_1.x;
        }
        else
        {
            _S684 = 0.0f;
        }
        float _S685;
        if((moments_1.y) > 0.0f)
        {
            _S685 = 1.0f / moments_1.y;
        }
        else
        {
            _S685 = 0.0f;
        }
        float _S686;
        if((moments_1.z) > 0.0f)
        {
            _S686 = 1.0f / moments_1.z;
        }
        else
        {
            _S686 = 0.0f;
        }
        inv_inertia_0 = float3(_S684, _S685, _S686);
    }
    else
    {
        inv_inertia_0 = _S683;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S683;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S683;
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
    uint _S687 = stage_0(c_15, &kernelContext_93);
    *((&kernelContext_93)->frames_0+_S687) = packed_float4(float4(pos_0, inv_mass_3)) ;
    packed_float4 device* _S688 = (&kernelContext_93)->frames_0+(_S687 + 1U);
    thread Quat_0 _S689 = _S681;
    float4 _S690 = quat_vec_0(&_S689);
    *_S688 = packed_float4(_S690) ;
    *((&kernelContext_93)->frames_0+(_S687 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_93)->frames_0+(_S687 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_93)->frames_0+(_S687 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_93)->params_0->step_0))))) ;
    *((&kernelContext_93)->frames_0+(_S687 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_93)->frames_0+(_S687 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_93)->frames_0+(_S687 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_93)->frames_0+2U * _S652)) ).w)) ;
    packed_float4 device* _S691 = (&kernelContext_93)->frames_0+(_S687 + 8U);
    thread Quat_0 _S692 = principal_0;
    float4 _S693 = quat_vec_0(&_S692);
    *_S691 = packed_float4(_S693) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], packed_float4 device* lp_17 [[buffer(24)]], packed_float4 device* ext_18 [[buffer(23)]], packed_float4 device* lp_sides_17 [[buffer(28)]], uint device* engine_counters_17 [[buffer(26)]], packed_float4 device* engine_contacts_17 [[buffer(25)]], packed_float4 device* cs_anchors_17 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_94;
    (&kernelContext_94)->params_0 = params_17;
    (&kernelContext_94)->work_0 = work_17;
    (&kernelContext_94)->frames_0 = frames_17;
    (&kernelContext_94)->bodies_0 = bodies_17;
    (&kernelContext_94)->chunks_0 = chunks_17;
    (&kernelContext_94)->sort_0 = sort_17;
    (&kernelContext_94)->atom_0 = atom_17;
    (&kernelContext_94)->manifolds_0 = manifolds_17;
    (&kernelContext_94)->loads_0 = loads_17;
    (&kernelContext_94)->bonds_0 = bonds_17;
    (&kernelContext_94)->args_0 = args_17;
    (&kernelContext_94)->info_0 = info_17;
    (&kernelContext_94)->scan_0 = scan_17;
    (&kernelContext_94)->parts_0 = parts_17;
    (&kernelContext_94)->colliders_0 = colliders_17;
    (&kernelContext_94)->collider_info_0 = collider_info_17;
    (&kernelContext_94)->view_chunks_0 = view_chunks_17;
    (&kernelContext_94)->view_0 = view_17;
    (&kernelContext_94)->objects_0 = objects_17;
    (&kernelContext_94)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_94)->lines_0 = lines_17;
    (&kernelContext_94)->geometry_0 = geometry_17;
    (&kernelContext_94)->shots_0 = shots_17;
    (&kernelContext_94)->lp_0 = lp_17;
    (&kernelContext_94)->ext_0 = ext_18;
    (&kernelContext_94)->lp_sides_0 = lp_sides_17;
    (&kernelContext_94)->engine_counters_0 = engine_counters_17;
    (&kernelContext_94)->engine_contacts_0 = engine_contacts_17;
    (&kernelContext_94)->cs_anchors_0 = cs_anchors_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_94)->g_scan_0 = &g_scan_17;
    uint c_16 = id_14.x;
    bool _S694 = is_root_0(c_16, &kernelContext_94);
    if(!_S694)
    {
        return;
    }
    uint _S695 = w_slot_0(&kernelContext_94);
    uint slot_3 = *((&kernelContext_94)->work_0+(_S695 + c_16));
    uint _S696 = stage_0(c_16, &kernelContext_94);
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
            *((&kernelContext_94)->bodies_0+(7U * (slot_3 + region_5 * (&kernelContext_94)->params_0->max_bodies_0) + w_4)) = packed_float4(float4(*((&kernelContext_94)->frames_0+(_S696 + w_4))) ) ;
            w_4 = w_4 + 1U;
        }
        region_5 = region_5 + 1U;
    }
    uint _S697 = 2U * slot_3;
    *((&kernelContext_94)->frames_0+_S697) = packed_float4(float4(*((&kernelContext_94)->frames_0+(_S696 + 7U))) ) ;
    *((&kernelContext_94)->frames_0+(_S697 + 1U)) = packed_float4(float4(*((&kernelContext_94)->frames_0+(_S696 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], packed_float4 device* lp_18 [[buffer(24)]], packed_float4 device* ext_19 [[buffer(23)]], packed_float4 device* lp_sides_18 [[buffer(28)]], uint device* engine_counters_18 [[buffer(26)]], packed_float4 device* engine_contacts_18 [[buffer(25)]], packed_float4 device* cs_anchors_18 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_95;
    (&kernelContext_95)->params_0 = params_18;
    (&kernelContext_95)->work_0 = work_18;
    (&kernelContext_95)->frames_0 = frames_18;
    (&kernelContext_95)->bodies_0 = bodies_18;
    (&kernelContext_95)->chunks_0 = chunks_18;
    (&kernelContext_95)->sort_0 = sort_18;
    (&kernelContext_95)->atom_0 = atom_18;
    (&kernelContext_95)->manifolds_0 = manifolds_18;
    (&kernelContext_95)->loads_0 = loads_18;
    (&kernelContext_95)->bonds_0 = bonds_18;
    (&kernelContext_95)->args_0 = args_18;
    (&kernelContext_95)->info_0 = info_18;
    (&kernelContext_95)->scan_0 = scan_18;
    (&kernelContext_95)->parts_0 = parts_18;
    (&kernelContext_95)->colliders_0 = colliders_18;
    (&kernelContext_95)->collider_info_0 = collider_info_18;
    (&kernelContext_95)->view_chunks_0 = view_chunks_18;
    (&kernelContext_95)->view_0 = view_18;
    (&kernelContext_95)->objects_0 = objects_18;
    (&kernelContext_95)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_95)->lines_0 = lines_18;
    (&kernelContext_95)->geometry_0 = geometry_18;
    (&kernelContext_95)->shots_0 = shots_18;
    (&kernelContext_95)->lp_0 = lp_18;
    (&kernelContext_95)->ext_0 = ext_19;
    (&kernelContext_95)->lp_sides_0 = lp_sides_18;
    (&kernelContext_95)->engine_counters_0 = engine_counters_18;
    (&kernelContext_95)->engine_contacts_0 = engine_contacts_18;
    (&kernelContext_95)->cs_anchors_0 = cs_anchors_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_95)->g_scan_0 = &g_scan_18;
    uint c_17 = id_15.x;
    bool _S698;
    if(c_17 >= (params_18->chunk_count_0))
    {
        _S698 = true;
    }
    else
    {
        uint _S699 = body_of_0(c_17, &kernelContext_95);
        bool _S700 = dirty_0(_S699, &kernelContext_95);
        _S698 = !_S700;
    }
    if(_S698)
    {
        return;
    }
    uint _S701 = w_label_0(&kernelContext_95);
    uint device* _S702 = (&kernelContext_95)->work_0+(_S701 + c_17);
    uint r_19 = *_S702;
    uint _S703 = w_slot_0(&kernelContext_95);
    uint device* _S704 = (&kernelContext_95)->work_0+(_S703 + *_S702);
    uint slot_4 = *_S704;
    *((&kernelContext_95)->work_0+(w_body_0() + c_17)) = *_S704;
    uint _S705 = 2U * slot_4;
    float3 _S706 = (float4(*((&kernelContext_95)->frames_0+_S705)) ).xyz;
    thread Quat_0 _S707 = quat_of_0(float4(*((&kernelContext_95)->frames_0+(_S705 + 1U))) );
    Quat_0 _S708 = conj_0(&_S707);
    float inv_mass_4 = (float4(*((&kernelContext_95)->bodies_0+7U * slot_4)) ).w;
    float _S709;
    if(inv_mass_4 > 0.0f)
    {
        _S709 = 1.0f / inv_mass_4;
    }
    else
    {
        _S709 = float_bits_0(2139095040U);
    }
    float4 _S710 = float4(*((&kernelContext_95)->chunks_0+(4U * c_17 + 2U))) ;
    uint first_1 = (as_type<uint>((_S710.z)));
    uint _S711 = (as_type<uint>((_S710.w)));
    float4 _S712 = float4(*((&kernelContext_95)->chunks_0+(4U * r_19 + 3U))) ;
    uint p_13 = first_1;
    for(;;)
    {
        if(p_13 < (first_1 + _S711))
        {
        }
        else
        {
            break;
        }
        uint _S713 = 2U * p_13;
        float4 _S714 = float4(*((&kernelContext_95)->parts_0+_S713)) ;
        uint col_0 = (as_type<uint>((_S714.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_95)->parts_0+(_S713 + 1U))) );
        uint _S715 = 4U * col_0;
        float4 _S716 = float4(*((&kernelContext_95)->colliders_0+_S715)) ;
        *((&kernelContext_95)->colliders_0+_S715) = packed_float4(float4((as_type<float>((slot_4))), _S716.y, _S716.z, _S716.w)) ;
        uint _S717 = _S715 + 1U;
        packed_float4 device* _S718 = (&kernelContext_95)->colliders_0+_S717;
        float3 _S719 = _S714.xyz - _S706;
        thread Quat_0 _S720 = _S708;
        float3 _S721 = rotate_0(&_S720, _S719);
        *_S718 = packed_float4(float4(_S721, (float4(*((&kernelContext_95)->colliders_0+_S717)) ).w)) ;
        packed_float4 device* _S722 = (&kernelContext_95)->colliders_0+(_S715 + 2U);
        thread Quat_0 _S723 = _S708;
        thread Quat_0 _S724 = rot_1;
        Quat_0 _S725 = quat_mul_0(&_S723, &_S724);
        thread Quat_0 _S726 = _S725;
        Quat_0 _S727 = normalized_0(&_S726);
        thread Quat_0 _S728 = _S727;
        float4 _S729 = quat_vec_0(&_S728);
        *_S722 = packed_float4(_S729) ;
        uint _S730 = 2U * col_0;
        uint4 _S731 = uint4(*((&kernelContext_95)->collider_info_0+_S730)) ;
        *((&kernelContext_95)->collider_info_0+_S730) = packed_uint4(uint4(_S731.x, slot_4, _S731.z, _S731.w)) ;
        *((&kernelContext_95)->collider_info_0+(_S730 + 1U)) = packed_uint4(uint4((as_type<uint>((_S709))), (as_type<uint>((_S712.x))), (as_type<uint>((_S712.w))), 0U)) ;
        p_13 = p_13 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], packed_float4 device* lp_19 [[buffer(24)]], packed_float4 device* ext_20 [[buffer(23)]], packed_float4 device* lp_sides_19 [[buffer(28)]], uint device* engine_counters_19 [[buffer(26)]], packed_float4 device* engine_contacts_19 [[buffer(25)]], packed_float4 device* cs_anchors_19 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_96;
    (&kernelContext_96)->params_0 = params_19;
    (&kernelContext_96)->work_0 = work_19;
    (&kernelContext_96)->frames_0 = frames_19;
    (&kernelContext_96)->bodies_0 = bodies_19;
    (&kernelContext_96)->chunks_0 = chunks_19;
    (&kernelContext_96)->sort_0 = sort_19;
    (&kernelContext_96)->atom_0 = atom_19;
    (&kernelContext_96)->manifolds_0 = manifolds_19;
    (&kernelContext_96)->loads_0 = loads_19;
    (&kernelContext_96)->bonds_0 = bonds_19;
    (&kernelContext_96)->args_0 = args_19;
    (&kernelContext_96)->info_0 = info_19;
    (&kernelContext_96)->scan_0 = scan_19;
    (&kernelContext_96)->parts_0 = parts_19;
    (&kernelContext_96)->colliders_0 = colliders_19;
    (&kernelContext_96)->collider_info_0 = collider_info_19;
    (&kernelContext_96)->view_chunks_0 = view_chunks_19;
    (&kernelContext_96)->view_0 = view_19;
    (&kernelContext_96)->objects_0 = objects_19;
    (&kernelContext_96)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_96)->lines_0 = lines_19;
    (&kernelContext_96)->geometry_0 = geometry_19;
    (&kernelContext_96)->shots_0 = shots_19;
    (&kernelContext_96)->lp_0 = lp_19;
    (&kernelContext_96)->ext_0 = ext_20;
    (&kernelContext_96)->lp_sides_0 = lp_sides_19;
    (&kernelContext_96)->engine_counters_0 = engine_counters_19;
    (&kernelContext_96)->engine_contacts_0 = engine_contacts_19;
    (&kernelContext_96)->cs_anchors_0 = cs_anchors_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_96)->g_scan_0 = &g_scan_19;
    uint s_10 = id_16.x;
    if(s_10 >= (params_19->body_count_0))
    {
        return;
    }
    uint _S732 = a_dirty_0(&kernelContext_96);
    atomic_store_explicit((&kernelContext_96)->atom_0+(_S732 + s_10), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_4 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_8 = fract(h_4);
    float q_17 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_8);
    float t_3 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_8));
    uint k_20 = uint(h_4);
    float3 _S733;
    if(k_20 == 0U)
    {
        _S733 = float3(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_20 == 1U)
        {
            _S733 = float3(q_17, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_20 == 2U)
            {
                _S733 = float3(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_20 == 3U)
                {
                    _S733 = float3(0.34959998726844788f, q_17, 0.92000001668930054f);
                }
                else
                {
                    if(k_20 == 4U)
                    {
                        _S733 = float3(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S733 = float3(0.92000001668930054f, 0.34959998726844788f, q_17);
                    }
                }
            }
        }
    }
    return _S733;
}

void write_object_0(uint o_4, const Quat_0 thread* q_18, float3 t_4, float4 color_0, KernelContext_0 thread* kernelContext_97)
{
    float3 _S734 = rotate_0(q_18, float3(1.0f, 0.0f, 0.0f));
    float3 _S735 = rotate_0(q_18, float3(0.0f, 1.0f, 0.0f));
    float3 _S736 = rotate_0(q_18, float3(0.0f, 0.0f, 1.0f));
    uint _S737 = 4U * o_4;
    *(kernelContext_97->objects_0+_S737) = packed_float4(float4(_S734.x, _S735.x, _S736.x, t_4.x)) ;
    *(kernelContext_97->objects_0+(_S737 + 1U)) = packed_float4(float4(_S734.y, _S735.y, _S736.y, t_4.y)) ;
    *(kernelContext_97->objects_0+(_S737 + 2U)) = packed_float4(float4(_S734.z, _S735.z, _S736.z, t_4.z)) ;
    *(kernelContext_97->objects_0+(_S737 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], packed_float4 device* lp_20 [[buffer(24)]], packed_float4 device* ext_21 [[buffer(23)]], packed_float4 device* lp_sides_20 [[buffer(28)]], uint device* engine_counters_20 [[buffer(26)]], packed_float4 device* engine_contacts_20 [[buffer(25)]], packed_float4 device* cs_anchors_20 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_98;
    (&kernelContext_98)->params_0 = params_20;
    (&kernelContext_98)->work_0 = work_20;
    (&kernelContext_98)->frames_0 = frames_20;
    (&kernelContext_98)->bodies_0 = bodies_20;
    (&kernelContext_98)->chunks_0 = chunks_20;
    (&kernelContext_98)->sort_0 = sort_20;
    (&kernelContext_98)->atom_0 = atom_20;
    (&kernelContext_98)->manifolds_0 = manifolds_20;
    (&kernelContext_98)->loads_0 = loads_20;
    (&kernelContext_98)->bonds_0 = bonds_20;
    (&kernelContext_98)->args_0 = args_20;
    (&kernelContext_98)->info_0 = info_20;
    (&kernelContext_98)->scan_0 = scan_20;
    (&kernelContext_98)->parts_0 = parts_20;
    (&kernelContext_98)->colliders_0 = colliders_20;
    (&kernelContext_98)->collider_info_0 = collider_info_20;
    (&kernelContext_98)->view_chunks_0 = view_chunks_20;
    (&kernelContext_98)->view_0 = view_20;
    (&kernelContext_98)->objects_0 = objects_20;
    (&kernelContext_98)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_98)->lines_0 = lines_20;
    (&kernelContext_98)->geometry_0 = geometry_20;
    (&kernelContext_98)->shots_0 = shots_20;
    (&kernelContext_98)->lp_0 = lp_20;
    (&kernelContext_98)->ext_0 = ext_21;
    (&kernelContext_98)->lp_sides_0 = lp_sides_20;
    (&kernelContext_98)->engine_counters_0 = engine_counters_20;
    (&kernelContext_98)->engine_contacts_0 = engine_contacts_20;
    (&kernelContext_98)->cs_anchors_0 = cs_anchors_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_98)->g_scan_0 = &g_scan_20;
    uint k_21 = id_17.x;
    if(k_21 >= (params_20->chunk_count_0))
    {
        return;
    }
    uint _S738 = body_of_0(k_21, &kernelContext_98);
    uint b_25 = 7U * _S738;
    thread Quat_0 q_19;
    thread float3 origin_5;
    structure_pose_0(_S738, b_25, &q_19, &origin_5, &kernelContext_98);
    uint _S739 = 3U * k_21;
    float4 _S740 = float4(*((&kernelContext_98)->view_chunks_0+_S739)) ;
    float3 base_2 = (float4(*((&kernelContext_98)->view_chunks_0+(_S739 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_98)->bodies_0+b_25)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_98)->view_0->mode_0) == 1U)
    {
        float _S741 = max(length((float4(*((&kernelContext_98)->bodies_0+(b_25 + 3U))) ).xyz), length((float4(*((&kernelContext_98)->bodies_0+(b_25 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S741 > ((&kernelContext_98)->view_0->rest_speed_0))
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
        if(((&kernelContext_98)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S738);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S738);
            }
        }
    }
    uint _S742 = (as_type<uint>((_S740.w)));
    Quat_0 _S743 = quat_of_0(float4(*((&kernelContext_98)->view_chunks_0+(_S739 + 1U))) );
    thread Quat_0 _S744 = q_19;
    thread Quat_0 _S745 = _S743;
    Quat_0 _S746 = quat_mul_0(&_S744, &_S745);
    thread Quat_0 _S747 = _S746;
    Quat_0 _S748 = normalized_0(&_S747);
    float3 _S749 = _S740.xyz;
    thread Quat_0 _S750 = q_19;
    float3 _S751 = rotate_0(&_S750, _S749);
    float3 _S752 = origin_5 + _S751;
    float4 _S753 = float4(color_1, 1.0f);
    thread Quat_0 _S754 = _S748;
    write_object_0(_S742, &_S754, _S752, _S753, &kernelContext_98);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], packed_float4 device* lp_21 [[buffer(24)]], packed_float4 device* ext_22 [[buffer(23)]], packed_float4 device* lp_sides_21 [[buffer(28)]], uint device* engine_counters_21 [[buffer(26)]], packed_float4 device* engine_contacts_21 [[buffer(25)]], packed_float4 device* cs_anchors_21 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_21;
    (&kernelContext_99)->work_0 = work_21;
    (&kernelContext_99)->frames_0 = frames_21;
    (&kernelContext_99)->bodies_0 = bodies_21;
    (&kernelContext_99)->chunks_0 = chunks_21;
    (&kernelContext_99)->sort_0 = sort_21;
    (&kernelContext_99)->atom_0 = atom_21;
    (&kernelContext_99)->manifolds_0 = manifolds_21;
    (&kernelContext_99)->loads_0 = loads_21;
    (&kernelContext_99)->bonds_0 = bonds_21;
    (&kernelContext_99)->args_0 = args_21;
    (&kernelContext_99)->info_0 = info_21;
    (&kernelContext_99)->scan_0 = scan_21;
    (&kernelContext_99)->parts_0 = parts_21;
    (&kernelContext_99)->colliders_0 = colliders_21;
    (&kernelContext_99)->collider_info_0 = collider_info_21;
    (&kernelContext_99)->view_chunks_0 = view_chunks_21;
    (&kernelContext_99)->view_0 = view_21;
    (&kernelContext_99)->objects_0 = objects_21;
    (&kernelContext_99)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_99)->lines_0 = lines_21;
    (&kernelContext_99)->geometry_0 = geometry_21;
    (&kernelContext_99)->shots_0 = shots_21;
    (&kernelContext_99)->lp_0 = lp_21;
    (&kernelContext_99)->ext_0 = ext_22;
    (&kernelContext_99)->lp_sides_0 = lp_sides_21;
    (&kernelContext_99)->engine_counters_0 = engine_counters_21;
    (&kernelContext_99)->engine_contacts_0 = engine_contacts_21;
    (&kernelContext_99)->cs_anchors_0 = cs_anchors_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_99)->g_scan_0 = &g_scan_21;
    uint i_5 = id_18.x;
    if(i_5 >= (view_21->shot_count_1))
    {
        return;
    }
    uint b_26 = 7U * (&kernelContext_99)->shot_bodies_0[i_5];
    uint o_5 = (&kernelContext_99)->view_0->impactor_base_0 + i_5;
    if(((float4(*((&kernelContext_99)->bodies_0+b_26)) ).w) == 0.0f)
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
            *((&kernelContext_99)->objects_0+(4U * o_5 + w_5)) = packed_float4(float4(0.0f) ) ;
            w_5 = w_5 + 1U;
        }
        return;
    }
    thread Quat_0 _S755 = quat_of_0(float4(*((&kernelContext_99)->bodies_0+(b_26 + 1U))) );
    Quat_0 _S756 = rot_from_rubble_0(&_S755);
    float3 _S757 = from_rubble_0((float4(*((&kernelContext_99)->bodies_0+b_26)) ).xyz);
    float4 _S758 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S759 = _S756;
    write_object_0(o_5, &_S759, _S757, _S758, &kernelContext_99);
    return;
}

float3 chunk_world_0(uint k_22, KernelContext_0 thread* kernelContext_100)
{
    uint _S760 = body_of_0(k_22, kernelContext_100);
    thread Quat_0 q_20;
    thread float3 origin_6;
    structure_pose_0(_S760, 7U * _S760, &q_20, &origin_6, kernelContext_100);
    float3 _S761 = (float4(*(kernelContext_100->view_chunks_0+3U * k_22)) ).xyz;
    thread Quat_0 _S762 = q_20;
    float3 _S763 = rotate_0(&_S762, _S761);
    return origin_6 + _S763;
}

void write_line_vertex_0(uint v_13, float3 p_14, KernelContext_0 thread* kernelContext_101)
{
    uint _S764 = 7U * v_13;
    *(kernelContext_101->lines_0+_S764) = p_14.x;
    *(kernelContext_101->lines_0+(_S764 + 1U)) = p_14.y;
    *(kernelContext_101->lines_0+(_S764 + 2U)) = p_14.z;
    *(kernelContext_101->lines_0+(_S764 + 3U)) = 1.0f;
    *(kernelContext_101->lines_0+(_S764 + 4U)) = 0.86000001430511475f;
    *(kernelContext_101->lines_0+(_S764 + 5U)) = 0.25f;
    *(kernelContext_101->lines_0+(_S764 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], packed_float4 device* lp_22 [[buffer(24)]], packed_float4 device* ext_23 [[buffer(23)]], packed_float4 device* lp_sides_22 [[buffer(28)]], uint device* engine_counters_22 [[buffer(26)]], packed_float4 device* engine_contacts_22 [[buffer(25)]], packed_float4 device* cs_anchors_22 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_102;
    (&kernelContext_102)->params_0 = params_22;
    (&kernelContext_102)->work_0 = work_22;
    (&kernelContext_102)->frames_0 = frames_22;
    (&kernelContext_102)->bodies_0 = bodies_22;
    (&kernelContext_102)->chunks_0 = chunks_22;
    (&kernelContext_102)->sort_0 = sort_22;
    (&kernelContext_102)->atom_0 = atom_22;
    (&kernelContext_102)->manifolds_0 = manifolds_22;
    (&kernelContext_102)->loads_0 = loads_22;
    (&kernelContext_102)->bonds_0 = bonds_22;
    (&kernelContext_102)->args_0 = args_22;
    (&kernelContext_102)->info_0 = info_22;
    (&kernelContext_102)->scan_0 = scan_22;
    (&kernelContext_102)->parts_0 = parts_22;
    (&kernelContext_102)->colliders_0 = colliders_22;
    (&kernelContext_102)->collider_info_0 = collider_info_22;
    (&kernelContext_102)->view_chunks_0 = view_chunks_22;
    (&kernelContext_102)->view_0 = view_22;
    (&kernelContext_102)->objects_0 = objects_22;
    (&kernelContext_102)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_102)->lines_0 = lines_22;
    (&kernelContext_102)->geometry_0 = geometry_22;
    (&kernelContext_102)->shots_0 = shots_22;
    (&kernelContext_102)->lp_0 = lp_22;
    (&kernelContext_102)->ext_0 = ext_23;
    (&kernelContext_102)->lp_sides_0 = lp_sides_22;
    (&kernelContext_102)->engine_counters_0 = engine_counters_22;
    (&kernelContext_102)->engine_contacts_0 = engine_contacts_22;
    (&kernelContext_102)->cs_anchors_0 = cs_anchors_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_102)->g_scan_0 = &g_scan_22;
    uint i_6 = id_19.x;
    if(i_6 >= (params_22->bond_count_0))
    {
        return;
    }
    uint4 _S765 = uint4(*((&kernelContext_102)->bonds_0+i_6)) ;
    float3 _S766 = chunk_world_0(_S765.x, &kernelContext_102);
    uint _S767 = w_alive_0(&kernelContext_102);
    float3 b_27;
    if((*((&kernelContext_102)->work_0+(_S767 + i_6))) != 0U)
    {
        float3 _S768 = chunk_world_0(_S765.y, &kernelContext_102);
        b_27 = _S768;
    }
    else
    {
        b_27 = _S766;
    }
    uint _S769 = 2U * i_6;
    write_line_vertex_0(_S769, _S766, &kernelContext_102);
    write_line_vertex_0(_S769 + 1U, b_27, &kernelContext_102);
    return;
}

[[kernel]] void fr_strike_reset(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], atomic<uint> device* atom_23 [[buffer(5)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], packed_float4 device* lp_23 [[buffer(24)]], packed_float4 device* ext_24 [[buffer(23)]], packed_float4 device* lp_sides_23 [[buffer(28)]], uint device* engine_counters_23 [[buffer(26)]], packed_float4 device* engine_contacts_23 [[buffer(25)]], packed_float4 device* cs_anchors_23 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_103;
    (&kernelContext_103)->params_0 = params_23;
    (&kernelContext_103)->work_0 = work_23;
    (&kernelContext_103)->frames_0 = frames_23;
    (&kernelContext_103)->bodies_0 = bodies_23;
    (&kernelContext_103)->chunks_0 = chunks_23;
    (&kernelContext_103)->sort_0 = sort_23;
    (&kernelContext_103)->atom_0 = atom_23;
    (&kernelContext_103)->manifolds_0 = manifolds_23;
    (&kernelContext_103)->loads_0 = loads_23;
    (&kernelContext_103)->bonds_0 = bonds_23;
    (&kernelContext_103)->args_0 = args_23;
    (&kernelContext_103)->info_0 = info_23;
    (&kernelContext_103)->scan_0 = scan_23;
    (&kernelContext_103)->parts_0 = parts_23;
    (&kernelContext_103)->colliders_0 = colliders_23;
    (&kernelContext_103)->collider_info_0 = collider_info_23;
    (&kernelContext_103)->view_chunks_0 = view_chunks_23;
    (&kernelContext_103)->view_0 = view_23;
    (&kernelContext_103)->objects_0 = objects_23;
    (&kernelContext_103)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_103)->lines_0 = lines_23;
    (&kernelContext_103)->geometry_0 = geometry_23;
    (&kernelContext_103)->shots_0 = shots_23;
    (&kernelContext_103)->lp_0 = lp_23;
    (&kernelContext_103)->ext_0 = ext_24;
    (&kernelContext_103)->lp_sides_0 = lp_sides_23;
    (&kernelContext_103)->engine_counters_0 = engine_counters_23;
    (&kernelContext_103)->engine_contacts_0 = engine_contacts_23;
    (&kernelContext_103)->cs_anchors_0 = cs_anchors_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_103)->g_scan_0 = &g_scan_23;
    uint b_28 = id_20.x;
    if(b_28 >= (params_23->body_count_0))
    {
        return;
    }
    uint _S770 = a_strike_0(&kernelContext_103);
    uint at_8 = _S770 + 9U * b_28;
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
        atomic_store_explicit((&kernelContext_103)->atom_0+(at_8 + w_6), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_103)->atom_0+(at_8 + 3U + w_6), 0U, memory_order_relaxed);
        w_6 = w_6 + 1U;
    }
    atomic_store_explicit((&kernelContext_103)->atom_0+(at_8 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_103)->atom_0+(at_8 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_103)->atom_0+(at_8 + 8U), 4294967295U, memory_order_relaxed);
    uint _S771 = strike_at_0(b_28, &kernelContext_103);
    (*((&kernelContext_103)->frames_0+_S771))[int(3)] = 0.0f;
    return;
}

uint ordered_0(float f_9)
{
    uint u_6 = (as_type<uint>((f_9)));
    uint _S772;
    if((u_6 & 2147483648U) != 0U)
    {
        _S772 = ~u_6;
    }
    else
    {
        _S772 = u_6 | 2147483648U;
    }
    return _S772;
}

[[kernel]] void fr_strike_bounds(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], atomic<uint> device* atom_24 [[buffer(5)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], packed_float4 device* lp_24 [[buffer(24)]], packed_float4 device* ext_25 [[buffer(23)]], packed_float4 device* lp_sides_24 [[buffer(28)]], uint device* engine_counters_24 [[buffer(26)]], packed_float4 device* engine_contacts_24 [[buffer(25)]], packed_float4 device* cs_anchors_24 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_104;
    (&kernelContext_104)->params_0 = params_24;
    (&kernelContext_104)->work_0 = work_24;
    (&kernelContext_104)->frames_0 = frames_24;
    (&kernelContext_104)->bodies_0 = bodies_24;
    (&kernelContext_104)->chunks_0 = chunks_24;
    (&kernelContext_104)->sort_0 = sort_24;
    (&kernelContext_104)->atom_0 = atom_24;
    (&kernelContext_104)->manifolds_0 = manifolds_24;
    (&kernelContext_104)->loads_0 = loads_24;
    (&kernelContext_104)->bonds_0 = bonds_24;
    (&kernelContext_104)->args_0 = args_24;
    (&kernelContext_104)->info_0 = info_24;
    (&kernelContext_104)->scan_0 = scan_24;
    (&kernelContext_104)->parts_0 = parts_24;
    (&kernelContext_104)->colliders_0 = colliders_24;
    (&kernelContext_104)->collider_info_0 = collider_info_24;
    (&kernelContext_104)->view_chunks_0 = view_chunks_24;
    (&kernelContext_104)->view_0 = view_24;
    (&kernelContext_104)->objects_0 = objects_24;
    (&kernelContext_104)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_104)->lines_0 = lines_24;
    (&kernelContext_104)->geometry_0 = geometry_24;
    (&kernelContext_104)->shots_0 = shots_24;
    (&kernelContext_104)->lp_0 = lp_24;
    (&kernelContext_104)->ext_0 = ext_25;
    (&kernelContext_104)->lp_sides_0 = lp_sides_24;
    (&kernelContext_104)->engine_counters_0 = engine_counters_24;
    (&kernelContext_104)->engine_contacts_0 = engine_contacts_24;
    (&kernelContext_104)->cs_anchors_0 = cs_anchors_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_104)->g_scan_0 = &g_scan_24;
    uint k_23 = id_21.x;
    bool _S773;
    if(k_23 >= (params_24->chunk_count_0))
    {
        _S773 = true;
    }
    else
    {
        _S773 = ((&kernelContext_104)->params_0->shot_count_0) == 0U;
    }
    if(_S773)
    {
        return;
    }
    thread Quat_0 q_21;
    float3 _S774 = chunk_box_world_0(k_23, &q_21, &kernelContext_104);
    float _S775 = (float4(*((&kernelContext_104)->geometry_0+3U * k_23)) ).w;
    uint _S776 = a_strike_0(&kernelContext_104);
    uint _S777 = body_of_0(k_23, &kernelContext_104);
    uint _S778 = _S776 + 9U * _S777;
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
        uint _S779 = ax_1;
        uint _S780 = atomic_fetch_min_explicit((&kernelContext_104)->atom_0+(_S778 + ax_1), ordered_0(_S774[ax_1] - _S775), memory_order_relaxed);
        uint _S781 = atomic_fetch_max_explicit((&kernelContext_104)->atom_0+(_S778 + 3U + ax_1), ordered_0(_S774[_S779] + _S775), memory_order_relaxed);
        ax_1 = ax_1 + 1U;
    }
    return;
}

float unordered_0(uint u_7)
{
    uint _S782;
    if((u_7 & 2147483648U) != 0U)
    {
        _S782 = u_7 & 2147483647U;
    }
    else
    {
        _S782 = ~u_7;
    }
    return (as_type<float>((_S782)));
}

[[kernel]] void fr_strike_find(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], atomic<uint> device* atom_25 [[buffer(5)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], packed_float4 device* lp_25 [[buffer(24)]], packed_float4 device* ext_26 [[buffer(23)]], packed_float4 device* lp_sides_25 [[buffer(28)]], uint device* engine_counters_25 [[buffer(26)]], packed_float4 device* engine_contacts_25 [[buffer(25)]], packed_float4 device* cs_anchors_25 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_105;
    (&kernelContext_105)->params_0 = params_25;
    (&kernelContext_105)->work_0 = work_25;
    (&kernelContext_105)->frames_0 = frames_25;
    (&kernelContext_105)->bodies_0 = bodies_25;
    (&kernelContext_105)->chunks_0 = chunks_25;
    (&kernelContext_105)->sort_0 = sort_25;
    (&kernelContext_105)->atom_0 = atom_25;
    (&kernelContext_105)->manifolds_0 = manifolds_25;
    (&kernelContext_105)->loads_0 = loads_25;
    (&kernelContext_105)->bonds_0 = bonds_25;
    (&kernelContext_105)->args_0 = args_25;
    (&kernelContext_105)->info_0 = info_25;
    (&kernelContext_105)->scan_0 = scan_25;
    (&kernelContext_105)->parts_0 = parts_25;
    (&kernelContext_105)->colliders_0 = colliders_25;
    (&kernelContext_105)->collider_info_0 = collider_info_25;
    (&kernelContext_105)->view_chunks_0 = view_chunks_25;
    (&kernelContext_105)->view_0 = view_25;
    (&kernelContext_105)->objects_0 = objects_25;
    (&kernelContext_105)->shot_bodies_0 = shot_bodies_25;
    (&kernelContext_105)->lines_0 = lines_25;
    (&kernelContext_105)->geometry_0 = geometry_25;
    (&kernelContext_105)->shots_0 = shots_25;
    (&kernelContext_105)->lp_0 = lp_25;
    (&kernelContext_105)->ext_0 = ext_26;
    (&kernelContext_105)->lp_sides_0 = lp_sides_25;
    (&kernelContext_105)->engine_counters_0 = engine_counters_25;
    (&kernelContext_105)->engine_contacts_0 = engine_contacts_25;
    (&kernelContext_105)->cs_anchors_0 = cs_anchors_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_105)->g_scan_0 = &g_scan_25;
    uint b_29 = id_22.x;
    bool _S783;
    if(b_29 >= (params_25->body_count_0))
    {
        _S783 = true;
    }
    else
    {
        _S783 = ((&kernelContext_105)->params_0->shot_count_0) == 0U;
    }
    if(_S783)
    {
        return;
    }
    uint _S784 = a_strike_0(&kernelContext_105);
    uint at_9 = _S784 + 9U * b_29;
    uint _S785 = atomic_load_explicit((&kernelContext_105)->atom_0+at_9, memory_order_relaxed);
    if(_S785 == 4294967295U)
    {
        return;
    }
    uint _S786 = atomic_load_explicit((&kernelContext_105)->atom_0+at_9, memory_order_relaxed);
    float _S787 = unordered_0(_S786);
    uint _S788 = atomic_load_explicit((&kernelContext_105)->atom_0+(at_9 + 1U), memory_order_relaxed);
    float _S789 = unordered_0(_S788);
    uint _S790 = atomic_load_explicit((&kernelContext_105)->atom_0+(at_9 + 2U), memory_order_relaxed);
    float3 _S791 = float3(_S787, _S789, unordered_0(_S790));
    uint _S792 = atomic_load_explicit((&kernelContext_105)->atom_0+(at_9 + 3U), memory_order_relaxed);
    float _S793 = unordered_0(_S792);
    uint _S794 = atomic_load_explicit((&kernelContext_105)->atom_0+(at_9 + 4U), memory_order_relaxed);
    float _S795 = unordered_0(_S794);
    uint _S796 = atomic_load_explicit((&kernelContext_105)->atom_0+(at_9 + 5U), memory_order_relaxed);
    float3 _S797 = float3(_S793, _S795, unordered_0(_S796));
    float3 _S798 = from_rubble_0((float4(*((&kernelContext_105)->bodies_0+(7U * b_29 + 3U))) ).xyz);
    float _S799 = float_bits_0(2139095040U);
    uint k_24 = 0U;
    for(;;)
    {
        if(k_24 < ((&kernelContext_105)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S800 = float4(*((&kernelContext_105)->shots_0+2U * k_24)) ;
        uint sb_1 = 7U * (as_type<uint>((_S800.x)));
        if(((float4(*((&kernelContext_105)->bodies_0+sb_1)) ).w) == 0.0f)
        {
            k_24 = k_24 + 1U;
            continue;
        }
        float3 c_18 = from_rubble_0((float4(*((&kernelContext_105)->bodies_0+sb_1)) ).xyz);
        float3 rel_2 = from_rubble_0((float4(*((&kernelContext_105)->bodies_0+(sb_1 + 3U))) ).xyz) - _S798;
        if(!((length(rel_2)) > ((&kernelContext_105)->params_0->strike_speed_0)))
        {
            k_24 = k_24 + 1U;
            continue;
        }
        float3 _S801 = float3(_S800.y) ;
        float3 _S802 = _S791 - _S801;
        float3 _S803 = _S797 + _S801;
        float _S804 = (&kernelContext_105)->params_0->dt_0;
        float t0_1 = 0.0f;
        float t1_1 = _S804;
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
            uint _S805 = ax_2;
            if((abs(rel_2[ax_2])) < 9.999999960041972e-13f)
            {
                uint _S806 = ax_2;
                if((c_18[ax_2]) < (_S802[ax_2]))
                {
                    _S783 = true;
                }
                else
                {
                    _S783 = (c_18[_S806]) > (_S803[ax_2]);
                }
                float t0_2;
                if(_S783)
                {
                    t0_2 = _S799;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + 1U;
                continue;
            }
            float ta_1 = (_S802[ax_2] - c_18[ax_2]) / rel_2[_S805];
            float tb_1 = (_S803[ax_2] - c_18[ax_2]) / rel_2[_S805];
            float _S807 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S807;
            ax_2 = ax_2 + 1U;
        }
        if(t0_1 <= t1_1)
        {
            uint _S808 = strike_at_0(b_29, &kernelContext_105);
            *((&kernelContext_105)->frames_0+_S808) = packed_float4(float4(c_18, float(k_24 + 1U))) ;
            *((&kernelContext_105)->frames_0+(_S808 + 1U)) = packed_float4(float4(rel_2, t0_1)) ;
            return;
        }
        k_24 = k_24 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_entry(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], atomic<uint> device* atom_26 [[buffer(5)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], packed_float4 device* lp_26 [[buffer(24)]], packed_float4 device* ext_27 [[buffer(23)]], packed_float4 device* lp_sides_26 [[buffer(28)]], uint device* engine_counters_26 [[buffer(26)]], packed_float4 device* engine_contacts_26 [[buffer(25)]], packed_float4 device* cs_anchors_26 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_106;
    (&kernelContext_106)->params_0 = params_26;
    (&kernelContext_106)->work_0 = work_26;
    (&kernelContext_106)->frames_0 = frames_26;
    (&kernelContext_106)->bodies_0 = bodies_26;
    (&kernelContext_106)->chunks_0 = chunks_26;
    (&kernelContext_106)->sort_0 = sort_26;
    (&kernelContext_106)->atom_0 = atom_26;
    (&kernelContext_106)->manifolds_0 = manifolds_26;
    (&kernelContext_106)->loads_0 = loads_26;
    (&kernelContext_106)->bonds_0 = bonds_26;
    (&kernelContext_106)->args_0 = args_26;
    (&kernelContext_106)->info_0 = info_26;
    (&kernelContext_106)->scan_0 = scan_26;
    (&kernelContext_106)->parts_0 = parts_26;
    (&kernelContext_106)->colliders_0 = colliders_26;
    (&kernelContext_106)->collider_info_0 = collider_info_26;
    (&kernelContext_106)->view_chunks_0 = view_chunks_26;
    (&kernelContext_106)->view_0 = view_26;
    (&kernelContext_106)->objects_0 = objects_26;
    (&kernelContext_106)->shot_bodies_0 = shot_bodies_26;
    (&kernelContext_106)->lines_0 = lines_26;
    (&kernelContext_106)->geometry_0 = geometry_26;
    (&kernelContext_106)->shots_0 = shots_26;
    (&kernelContext_106)->lp_0 = lp_26;
    (&kernelContext_106)->ext_0 = ext_27;
    (&kernelContext_106)->lp_sides_0 = lp_sides_26;
    (&kernelContext_106)->engine_counters_0 = engine_counters_26;
    (&kernelContext_106)->engine_contacts_0 = engine_contacts_26;
    (&kernelContext_106)->cs_anchors_0 = cs_anchors_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_106)->g_scan_0 = &g_scan_26;
    uint k_25 = id_23.x;
    bool _S809;
    if(k_25 >= (params_26->chunk_count_0))
    {
        _S809 = true;
    }
    else
    {
        _S809 = ((&kernelContext_106)->params_0->shot_count_0) == 0U;
    }
    if(_S809)
    {
        return;
    }
    uint _S810 = body_of_0(k_25, &kernelContext_106);
    thread uint shot_5;
    bool _S811 = struck_0(_S810, &shot_5, &kernelContext_106);
    if(!_S811)
    {
        return;
    }
    uint _S812 = strike_at_0(_S810, &kernelContext_106);
    float4 _S813 = float4(*((&kernelContext_106)->frames_0+(_S812 + 1U))) ;
    float3 from_3 = (float4(*((&kernelContext_106)->frames_0+_S812)) ).xyz;
    float3 rel_3 = _S813.xyz;
    float3 d_19 = rel_3 / float3(max(length(rel_3), 9.999999960041972e-13f)) ;
    float _S814 = (float4(*((&kernelContext_106)->shots_0+2U * shot_5)) ).y;
    thread Quat_0 q_22;
    float3 _S815 = chunk_box_world_0(k_25, &q_22, &kernelContext_106);
    uint _S816 = 3U * k_25;
    Quat_0 _S817 = quat_of_0(float4(*((&kernelContext_106)->geometry_0+(_S816 + 2U))) );
    thread Quat_0 _S818 = q_22;
    thread Quat_0 _S819 = _S817;
    Quat_0 _S820 = quat_mul_0(&_S818, &_S819);
    thread Quat_0 _S821 = _S820;
    Quat_0 _S822 = normalized_0(&_S821);
    float3 _S823 = (float4(*((&kernelContext_106)->geometry_0+(_S816 + 1U))) ).xyz;
    float _S824 = float_bits_0(2139095040U);
    float _S825 = (&kernelContext_106)->params_0->dt_0;
    float t_in_5 = 0.0f;
    float t_out_0 = _S825;
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
        thread float3 e_2 = float3(0.0f) ;
        e_2[ax_3] = 1.0f;
        thread Quat_0 _S826 = _S822;
        float3 _S827 = rotate_0(&_S826, e_2);
        float o_6 = dot(from_3 - _S815, _S827);
        float dv_1 = dot(rel_3, _S827);
        float ext_28 = _S823[ax_3] + _S814;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            float t_in_6;
            if((abs(o_6)) > ext_28)
            {
                t_in_6 = _S824;
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
        float _S828 = min(t_out_0, max(ta_2, tb_2));
        t_in_5 = max(t_in_5, min(ta_2, tb_2));
        t_out_0 = _S828;
        ax_3 = ax_3 + 1U;
    }
    uint _S829 = a_strike_0(&kernelContext_106);
    uint at_10 = _S829 + 9U * _S810;
    if(t_in_5 <= t_out_0)
    {
        uint _S830 = atomic_fetch_min_explicit((&kernelContext_106)->atom_0+(at_10 + 6U), (as_type<uint>((t_in_5))), memory_order_relaxed);
    }
    float3 r_20 = _S815 - (from_3 + rel_3 * float3(_S813.w) );
    float along_1 = dot(r_20, d_19);
    uint _S831 = atomic_fetch_min_explicit((&kernelContext_106)->atom_0+(at_10 + 7U), (as_type<uint>((length(r_20 - d_19 * float3(along_1) ) + abs(along_1) * 0.00100000004749745f))), memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], atomic<uint> device* atom_27 [[buffer(5)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], packed_float4 device* lp_27 [[buffer(24)]], packed_float4 device* ext_29 [[buffer(23)]], packed_float4 device* lp_sides_27 [[buffer(28)]], uint device* engine_counters_27 [[buffer(26)]], packed_float4 device* engine_contacts_27 [[buffer(25)]], packed_float4 device* cs_anchors_27 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_107;
    (&kernelContext_107)->params_0 = params_27;
    (&kernelContext_107)->work_0 = work_27;
    (&kernelContext_107)->frames_0 = frames_27;
    (&kernelContext_107)->bodies_0 = bodies_27;
    (&kernelContext_107)->chunks_0 = chunks_27;
    (&kernelContext_107)->sort_0 = sort_27;
    (&kernelContext_107)->atom_0 = atom_27;
    (&kernelContext_107)->manifolds_0 = manifolds_27;
    (&kernelContext_107)->loads_0 = loads_27;
    (&kernelContext_107)->bonds_0 = bonds_27;
    (&kernelContext_107)->args_0 = args_27;
    (&kernelContext_107)->info_0 = info_27;
    (&kernelContext_107)->scan_0 = scan_27;
    (&kernelContext_107)->parts_0 = parts_27;
    (&kernelContext_107)->colliders_0 = colliders_27;
    (&kernelContext_107)->collider_info_0 = collider_info_27;
    (&kernelContext_107)->view_chunks_0 = view_chunks_27;
    (&kernelContext_107)->view_0 = view_27;
    (&kernelContext_107)->objects_0 = objects_27;
    (&kernelContext_107)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_107)->lines_0 = lines_27;
    (&kernelContext_107)->geometry_0 = geometry_27;
    (&kernelContext_107)->shots_0 = shots_27;
    (&kernelContext_107)->lp_0 = lp_27;
    (&kernelContext_107)->ext_0 = ext_29;
    (&kernelContext_107)->lp_sides_0 = lp_sides_27;
    (&kernelContext_107)->engine_counters_0 = engine_counters_27;
    (&kernelContext_107)->engine_contacts_0 = engine_contacts_27;
    (&kernelContext_107)->cs_anchors_0 = cs_anchors_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_107)->g_scan_0 = &g_scan_27;
    uint k_26 = id_24.x;
    bool _S832;
    if(k_26 >= (params_27->chunk_count_0))
    {
        _S832 = true;
    }
    else
    {
        _S832 = ((&kernelContext_107)->params_0->shot_count_0) == 0U;
    }
    if(_S832)
    {
        return;
    }
    uint _S833 = body_of_0(k_26, &kernelContext_107);
    thread uint shot_6;
    bool _S834 = struck_0(_S833, &shot_6, &kernelContext_107);
    if(!_S834)
    {
        return;
    }
    uint _S835 = strike_at_0(_S833, &kernelContext_107);
    float4 _S836 = float4(*((&kernelContext_107)->frames_0+_S835)) ;
    float4 _S837 = float4(*((&kernelContext_107)->frames_0+(_S835 + 1U))) ;
    float3 _S838 = _S837.xyz;
    float3 d_20 = _S838 / float3(max(length(_S838), 9.999999960041972e-13f)) ;
    thread Quat_0 q_23;
    float3 _S839 = chunk_box_world_0(k_26, &q_23, &kernelContext_107);
    float3 r_21 = _S839 - (_S836.xyz + _S838 * float3(_S837.w) );
    float along_2 = dot(r_21, d_20);
    float dist_0 = length(r_21 - d_20 * float3(along_2) ) + abs(along_2) * 0.00100000004749745f;
    uint _S840 = a_strike_0(&kernelContext_107);
    uint at_11 = _S840 + 9U * _S833;
    uint _S841 = (as_type<uint>((dist_0)));
    uint _S842 = atomic_load_explicit((&kernelContext_107)->atom_0+(at_11 + 7U), memory_order_relaxed);
    if(_S841 == _S842)
    {
        uint _S843 = atomic_fetch_min_explicit((&kernelContext_107)->atom_0+(at_11 + 8U), k_26, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], atomic<uint> device* atom_28 [[buffer(5)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], packed_float4 device* lp_28 [[buffer(24)]], packed_float4 device* ext_30 [[buffer(23)]], packed_float4 device* lp_sides_28 [[buffer(28)]], uint device* engine_counters_28 [[buffer(26)]], packed_float4 device* engine_contacts_28 [[buffer(25)]], packed_float4 device* cs_anchors_28 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_108;
    (&kernelContext_108)->params_0 = params_28;
    (&kernelContext_108)->work_0 = work_28;
    (&kernelContext_108)->frames_0 = frames_28;
    (&kernelContext_108)->bodies_0 = bodies_28;
    (&kernelContext_108)->chunks_0 = chunks_28;
    (&kernelContext_108)->sort_0 = sort_28;
    (&kernelContext_108)->atom_0 = atom_28;
    (&kernelContext_108)->manifolds_0 = manifolds_28;
    (&kernelContext_108)->loads_0 = loads_28;
    (&kernelContext_108)->bonds_0 = bonds_28;
    (&kernelContext_108)->args_0 = args_28;
    (&kernelContext_108)->info_0 = info_28;
    (&kernelContext_108)->scan_0 = scan_28;
    (&kernelContext_108)->parts_0 = parts_28;
    (&kernelContext_108)->colliders_0 = colliders_28;
    (&kernelContext_108)->collider_info_0 = collider_info_28;
    (&kernelContext_108)->view_chunks_0 = view_chunks_28;
    (&kernelContext_108)->view_0 = view_28;
    (&kernelContext_108)->objects_0 = objects_28;
    (&kernelContext_108)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_108)->lines_0 = lines_28;
    (&kernelContext_108)->geometry_0 = geometry_28;
    (&kernelContext_108)->shots_0 = shots_28;
    (&kernelContext_108)->lp_0 = lp_28;
    (&kernelContext_108)->ext_0 = ext_30;
    (&kernelContext_108)->lp_sides_0 = lp_sides_28;
    (&kernelContext_108)->engine_counters_0 = engine_counters_28;
    (&kernelContext_108)->engine_contacts_0 = engine_contacts_28;
    (&kernelContext_108)->cs_anchors_0 = cs_anchors_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_108)->g_scan_0 = &g_scan_28;
    uint b_30 = id_25.x;
    bool _S844;
    if(b_30 >= (params_28->body_count_0))
    {
        _S844 = true;
    }
    else
    {
        _S844 = ((&kernelContext_108)->params_0->shot_count_0) == 0U;
    }
    if(_S844)
    {
        return;
    }
    thread uint shot_7;
    bool _S845 = struck_0(b_30, &shot_7, &kernelContext_108);
    if(!_S845)
    {
        return;
    }
    uint _S846 = a_strike_0(&kernelContext_108);
    uint at_12 = _S846 + 9U * b_30;
    uint _S847 = atomic_load_explicit((&kernelContext_108)->atom_0+(at_12 + 6U), memory_order_relaxed);
    float t_in_7 = (as_type<float>((_S847)));
    uint entry_2 = atomic_load_explicit((&kernelContext_108)->atom_0+(at_12 + 8U), memory_order_relaxed);
    uint _S848 = strike_at_0(b_30, &kernelContext_108);
    if(!(t_in_7 <= ((&kernelContext_108)->params_0->dt_0)))
    {
        _S844 = true;
    }
    else
    {
        _S844 = entry_2 == 4294967295U;
    }
    if(_S844)
    {
        (*((&kernelContext_108)->frames_0+_S848))[int(3)] = 0.0f;
        return;
    }
    float3 rel_4 = (float4(*((&kernelContext_108)->frames_0+(_S848 + 1U))) ).xyz;
    float speed_14 = length(rel_4);
    float4 _S849 = float4(*((&kernelContext_108)->shots_0+2U * shot_7)) ;
    float _S850 = max(_S849.z, 0.00100000004749745f);
    float m_imp_0 = _S849.w;
    float e_imp_0 = (float4(*((&kernelContext_108)->shots_0+(2U * shot_7 + 1U))) ).x;
    float3 contact_16 = (float4(*((&kernelContext_108)->frames_0+_S848)) ).xyz + rel_4 * float3(t_in_7)  + rel_4 / float3(max(speed_14, 9.999999960041972e-13f))  * float3(_S849.y) ;
    float inv_mass_5 = (float4(*((&kernelContext_108)->bodies_0+7U * b_30)) ).w;
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
    float modulus_1 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_108)->geometry_0+(3U * entry_2 + 1U))) ).w);
    *((&kernelContext_108)->frames_0+(_S848 + 2U)) = packed_float4(float4(contact_16, m_eff_6 * speed_14 / (2.86999988555908203f * pow(m_eff_6 * m_eff_6 / (_S850 * modulus_1 * modulus_1 * max(speed_14, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

bool strike_share_breaks_0(uint4 bond_0, float3 contact_17, float force_2, KernelContext_0 thread* kernelContext_109)
{
    thread Quat_0 q_24;
    float3 _S851 = chunk_box_world_0(bond_0.x, &q_24, kernelContext_109);
    float3 _S852 = chunk_box_world_0(bond_0.y, &q_24, kernelContext_109);
    float dist_1 = length(float3(0.5f)  * (_S851 + _S852) - contact_17);
    return (force_2 * min(1.0f, (as_type<float>((bond_0.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((bond_0.z)));
}

[[kernel]] void fr_strike_breaks(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], atomic<uint> device* atom_29 [[buffer(5)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], packed_float4 device* lp_29 [[buffer(24)]], packed_float4 device* ext_31 [[buffer(23)]], packed_float4 device* lp_sides_29 [[buffer(28)]], uint device* engine_counters_29 [[buffer(26)]], packed_float4 device* engine_contacts_29 [[buffer(25)]], packed_float4 device* cs_anchors_29 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_110;
    (&kernelContext_110)->params_0 = params_29;
    (&kernelContext_110)->work_0 = work_29;
    (&kernelContext_110)->frames_0 = frames_29;
    (&kernelContext_110)->bodies_0 = bodies_29;
    (&kernelContext_110)->chunks_0 = chunks_29;
    (&kernelContext_110)->sort_0 = sort_29;
    (&kernelContext_110)->atom_0 = atom_29;
    (&kernelContext_110)->manifolds_0 = manifolds_29;
    (&kernelContext_110)->loads_0 = loads_29;
    (&kernelContext_110)->bonds_0 = bonds_29;
    (&kernelContext_110)->args_0 = args_29;
    (&kernelContext_110)->info_0 = info_29;
    (&kernelContext_110)->scan_0 = scan_29;
    (&kernelContext_110)->parts_0 = parts_29;
    (&kernelContext_110)->colliders_0 = colliders_29;
    (&kernelContext_110)->collider_info_0 = collider_info_29;
    (&kernelContext_110)->view_chunks_0 = view_chunks_29;
    (&kernelContext_110)->view_0 = view_29;
    (&kernelContext_110)->objects_0 = objects_29;
    (&kernelContext_110)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_110)->lines_0 = lines_29;
    (&kernelContext_110)->geometry_0 = geometry_29;
    (&kernelContext_110)->shots_0 = shots_29;
    (&kernelContext_110)->lp_0 = lp_29;
    (&kernelContext_110)->ext_0 = ext_31;
    (&kernelContext_110)->lp_sides_0 = lp_sides_29;
    (&kernelContext_110)->engine_counters_0 = engine_counters_29;
    (&kernelContext_110)->engine_contacts_0 = engine_contacts_29;
    (&kernelContext_110)->cs_anchors_0 = cs_anchors_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_110)->g_scan_0 = &g_scan_29;
    uint i_7 = id_26.x;
    bool _S853;
    if(i_7 >= (params_29->bond_count_0))
    {
        _S853 = true;
    }
    else
    {
        _S853 = ((&kernelContext_110)->params_0->shot_count_0) == 0U;
    }
    if(_S853)
    {
        _S853 = true;
    }
    else
    {
        uint _S854 = w_alive_0(&kernelContext_110);
        _S853 = (*((&kernelContext_110)->work_0+(_S854 + i_7))) == 0U;
    }
    if(_S853)
    {
        return;
    }
    uint4 _S855 = uint4(*((&kernelContext_110)->bonds_0+i_7)) ;
    uint _S856 = body_of_0(_S855.x, &kernelContext_110);
    thread uint shot_8;
    bool _S857 = struck_0(_S856, &shot_8, &kernelContext_110);
    if(!_S857)
    {
        return;
    }
    uint _S858 = strike_at_0(_S856, &kernelContext_110);
    float4 _S859 = float4(*((&kernelContext_110)->frames_0+(_S858 + 2U))) ;
    bool _S860 = strike_share_breaks_0(_S855, _S859.xyz, _S859.w, &kernelContext_110);
    if(_S860)
    {
        uint _S861 = w_alive_0(&kernelContext_110);
        *((&kernelContext_110)->work_0+(_S861 + i_7)) = 0U;
        uint _S862 = a_dirty_0(&kernelContext_110);
        uint _S863 = atomic_fetch_or_explicit((&kernelContext_110)->atom_0+(_S862 + _S856), 1U, memory_order_relaxed);
        uint _S864 = a_counters_1(&kernelContext_110);
        uint _S865 = atomic_fetch_add_explicit((&kernelContext_110)->atom_0+_S864, 1U, memory_order_relaxed);
        uint _S866 = atomic_fetch_add_explicit((&kernelContext_110)->atom_0+(_S864 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_lp_0(uint b_31, KernelContext_0 thread* kernelContext_111)
{
    uint _S867 = a_strike_0(kernelContext_111);
    return _S867 + 9U * kernelContext_111->params_0->body_count_0 + 14U * b_31;
}

[[kernel]] void fr_lp_clear(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], atomic<uint> device* atom_30 [[buffer(5)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], packed_float4 device* lp_30 [[buffer(24)]], packed_float4 device* ext_32 [[buffer(23)]], packed_float4 device* lp_sides_30 [[buffer(28)]], uint device* engine_counters_30 [[buffer(26)]], packed_float4 device* engine_contacts_30 [[buffer(25)]], packed_float4 device* cs_anchors_30 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_112;
    (&kernelContext_112)->params_0 = params_30;
    (&kernelContext_112)->work_0 = work_30;
    (&kernelContext_112)->frames_0 = frames_30;
    (&kernelContext_112)->bodies_0 = bodies_30;
    (&kernelContext_112)->chunks_0 = chunks_30;
    (&kernelContext_112)->sort_0 = sort_30;
    (&kernelContext_112)->atom_0 = atom_30;
    (&kernelContext_112)->manifolds_0 = manifolds_30;
    (&kernelContext_112)->loads_0 = loads_30;
    (&kernelContext_112)->bonds_0 = bonds_30;
    (&kernelContext_112)->args_0 = args_30;
    (&kernelContext_112)->info_0 = info_30;
    (&kernelContext_112)->scan_0 = scan_30;
    (&kernelContext_112)->parts_0 = parts_30;
    (&kernelContext_112)->colliders_0 = colliders_30;
    (&kernelContext_112)->collider_info_0 = collider_info_30;
    (&kernelContext_112)->view_chunks_0 = view_chunks_30;
    (&kernelContext_112)->view_0 = view_30;
    (&kernelContext_112)->objects_0 = objects_30;
    (&kernelContext_112)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_112)->lines_0 = lines_30;
    (&kernelContext_112)->geometry_0 = geometry_30;
    (&kernelContext_112)->shots_0 = shots_30;
    (&kernelContext_112)->lp_0 = lp_30;
    (&kernelContext_112)->ext_0 = ext_32;
    (&kernelContext_112)->lp_sides_0 = lp_sides_30;
    (&kernelContext_112)->engine_counters_0 = engine_counters_30;
    (&kernelContext_112)->engine_contacts_0 = engine_contacts_30;
    (&kernelContext_112)->cs_anchors_0 = cs_anchors_30;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_112)->g_scan_0 = &g_scan_30;
    uint b_32 = id_27.x;
    if(b_32 >= (params_30->body_count_0))
    {
        return;
    }
    uint _S868 = a_lp_0(b_32, &kernelContext_112);
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
        atomic_store_explicit((&kernelContext_112)->atom_0+(_S868 + w_7), 0U, memory_order_relaxed);
        w_7 = w_7 + 1U;
    }
    atomic_store_explicit((&kernelContext_112)->atom_0+(_S868 + 12U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_112)->atom_0+(_S868 + 13U), 0U, memory_order_relaxed);
    return;
}

uint st_f_0(uint k_27, uint state_0)
{
    return 22U * k_27 + state_0 + 6U;
}

uint lp_f_0(uint k_28)
{
    return st_f_0(k_28, 0U);
}

void add64w_0(uint at_13, float value_1, float scale_3, KernelContext_0 thread* kernelContext_113)
{
    float x_6 = round(value_1 * scale_3);
    float x_7;
    if(!((abs(x_6)) < 3.99999993722699776e+18f))
    {
        uint _S869 = a_counters_1(kernelContext_113);
        uint _S870 = atomic_fetch_or_explicit(kernelContext_113->atom_0+(_S869 + 3U), 2U, memory_order_relaxed);
        x_7 = 0.0f;
    }
    else
    {
        x_7 = x_6;
    }
    float u_8 = abs(x_7);
    float h_5 = floor(u_8 * 2.32830643653869629e-10f);
    uint lo_3 = uint(u_8 - h_5 * 4.294967296e+09f);
    uint hi_3 = uint(h_5);
    uint lo_4;
    uint hi_4;
    if(x_7 < 0.0f)
    {
        uint lo_5 = ~lo_3 + 1U;
        uint _S871 = ~hi_3;
        if(lo_5 == 0U)
        {
            lo_4 = 1U;
        }
        else
        {
            lo_4 = 0U;
        }
        uint _S872 = _S871 + lo_4;
        lo_4 = lo_5;
        hi_4 = _S872;
    }
    else
    {
        lo_4 = lo_3;
        hi_4 = hi_3;
    }
    uint old_1 = atomic_fetch_add_explicit(kernelContext_113->atom_0+at_13, lo_4, memory_order_relaxed);
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
        uint _S873 = atomic_fetch_add_explicit(kernelContext_113->atom_0+(at_13 + 1U), hi_5, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_lp_sum(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], atomic<uint> device* atom_31 [[buffer(5)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_31 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], packed_float4 device* lp_31 [[buffer(24)]], packed_float4 device* ext_33 [[buffer(23)]], packed_float4 device* lp_sides_31 [[buffer(28)]], uint device* engine_counters_31 [[buffer(26)]], packed_float4 device* engine_contacts_31 [[buffer(25)]], packed_float4 device* cs_anchors_31 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_114;
    (&kernelContext_114)->params_0 = params_31;
    (&kernelContext_114)->work_0 = work_31;
    (&kernelContext_114)->frames_0 = frames_31;
    (&kernelContext_114)->bodies_0 = bodies_31;
    (&kernelContext_114)->chunks_0 = chunks_31;
    (&kernelContext_114)->sort_0 = sort_31;
    (&kernelContext_114)->atom_0 = atom_31;
    (&kernelContext_114)->manifolds_0 = manifolds_31;
    (&kernelContext_114)->loads_0 = loads_31;
    (&kernelContext_114)->bonds_0 = bonds_31;
    (&kernelContext_114)->args_0 = args_31;
    (&kernelContext_114)->info_0 = info_31;
    (&kernelContext_114)->scan_0 = scan_31;
    (&kernelContext_114)->parts_0 = parts_31;
    (&kernelContext_114)->colliders_0 = colliders_31;
    (&kernelContext_114)->collider_info_0 = collider_info_31;
    (&kernelContext_114)->view_chunks_0 = view_chunks_31;
    (&kernelContext_114)->view_0 = view_31;
    (&kernelContext_114)->objects_0 = objects_31;
    (&kernelContext_114)->shot_bodies_0 = shot_bodies_31;
    (&kernelContext_114)->lines_0 = lines_31;
    (&kernelContext_114)->geometry_0 = geometry_31;
    (&kernelContext_114)->shots_0 = shots_31;
    (&kernelContext_114)->lp_0 = lp_31;
    (&kernelContext_114)->ext_0 = ext_33;
    (&kernelContext_114)->lp_sides_0 = lp_sides_31;
    (&kernelContext_114)->engine_counters_0 = engine_counters_31;
    (&kernelContext_114)->engine_contacts_0 = engine_contacts_31;
    (&kernelContext_114)->cs_anchors_0 = cs_anchors_31;
    threadgroup array<uint, int(1024)> g_scan_31;
    (&kernelContext_114)->g_scan_0 = &g_scan_31;
    uint k_29 = id_28.x;
    if(k_29 >= (params_31->chunk_count_0))
    {
        return;
    }
    uint _S874 = body_of_0(k_29, &kernelContext_114);
    thread Quat_0 q_25;
    thread float3 origin_7;
    structure_pose_0(_S874, 7U * _S874, &q_25, &origin_7, &kernelContext_114);
    thread Quat_0 _S875 = q_25;
    Quat_0 _S876 = conj_0(&_S875);
    float4 _S877 = float4(*((&kernelContext_114)->chunks_0+4U * k_29)) ;
    uint _S878 = l_vec_0(&kernelContext_114);
    uint _S879 = _S878 + 2U * k_29;
    float3 _S880 = (float4(*((&kernelContext_114)->loads_0+_S879)) ).xyz;
    thread Quat_0 _S881 = _S876;
    float3 _S882 = rotate_0(&_S881, _S880);
    float _S883 = _S877.w;
    float3 _S884 = float3((&kernelContext_114)->params_0->gx_0, (&kernelContext_114)->params_0->gy_0, (&kernelContext_114)->params_0->gz_0);
    thread Quat_0 _S885 = _S876;
    float3 _S886 = rotate_0(&_S885, _S884);
    float3 f_10 = _S882 + float3(_S883)  * _S886;
    float3 _S887 = (float4(*((&kernelContext_114)->loads_0+(_S879 + 1U))) ).xyz;
    thread Quat_0 _S888 = _S876;
    float3 _S889 = rotate_0(&_S888, _S887);
    uint _S890 = lp_f_0(k_29);
    *((&kernelContext_114)->lp_0+_S890) = packed_float4(float4(f_10, 0.0f)) ;
    *((&kernelContext_114)->lp_0+(_S890 + 1U)) = packed_float4(float4(_S889, 0.0f)) ;
    float3 _S891 = _S889 + cross(_S877.xyz - (float4(*((&kernelContext_114)->frames_0+2U * _S874)) ).xyz, f_10);
    uint _S892 = a_lp_0(_S874, &kernelContext_114);
    uint a_7 = 0U;
    for(;;)
    {
        if(a_7 < 3U)
        {
        }
        else
        {
            break;
        }
        uint _S893 = 2U * a_7;
        add64w_0(_S892 + _S893, f_10[a_7], (&kernelContext_114)->params_0->lp_scale_0, &kernelContext_114);
        add64w_0(_S892 + 6U + _S893, _S891[a_7], (&kernelContext_114)->params_0->lp_scale_0, &kernelContext_114);
        a_7 = a_7 + 1U;
    }
    uint _S894 = atomic_fetch_min_explicit((&kernelContext_114)->atom_0+(_S892 + 12U), k_29, memory_order_relaxed);
    if(((&kernelContext_114)->info_0[k_29] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_114)->atom_0+(_S892 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

float3 read64w3_0(uint at_14, float scale_4, KernelContext_0 thread* kernelContext_115)
{
    float _S895 = read64_1(at_14, scale_4, kernelContext_115);
    float _S896 = read64_1(at_14 + 2U, scale_4, kernelContext_115);
    float _S897 = read64_1(at_14 + 4U, scale_4, kernelContext_115);
    return float3(_S895, _S896, _S897);
}

[[kernel]] void fr_lp_relief(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], atomic<uint> device* atom_32 [[buffer(5)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_32 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], packed_float4 device* lp_32 [[buffer(24)]], packed_float4 device* ext_34 [[buffer(23)]], packed_float4 device* lp_sides_32 [[buffer(28)]], uint device* engine_counters_32 [[buffer(26)]], packed_float4 device* engine_contacts_32 [[buffer(25)]], packed_float4 device* cs_anchors_32 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_116;
    (&kernelContext_116)->params_0 = params_32;
    (&kernelContext_116)->work_0 = work_32;
    (&kernelContext_116)->frames_0 = frames_32;
    (&kernelContext_116)->bodies_0 = bodies_32;
    (&kernelContext_116)->chunks_0 = chunks_32;
    (&kernelContext_116)->sort_0 = sort_32;
    (&kernelContext_116)->atom_0 = atom_32;
    (&kernelContext_116)->manifolds_0 = manifolds_32;
    (&kernelContext_116)->loads_0 = loads_32;
    (&kernelContext_116)->bonds_0 = bonds_32;
    (&kernelContext_116)->args_0 = args_32;
    (&kernelContext_116)->info_0 = info_32;
    (&kernelContext_116)->scan_0 = scan_32;
    (&kernelContext_116)->parts_0 = parts_32;
    (&kernelContext_116)->colliders_0 = colliders_32;
    (&kernelContext_116)->collider_info_0 = collider_info_32;
    (&kernelContext_116)->view_chunks_0 = view_chunks_32;
    (&kernelContext_116)->view_0 = view_32;
    (&kernelContext_116)->objects_0 = objects_32;
    (&kernelContext_116)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_116)->lines_0 = lines_32;
    (&kernelContext_116)->geometry_0 = geometry_32;
    (&kernelContext_116)->shots_0 = shots_32;
    (&kernelContext_116)->lp_0 = lp_32;
    (&kernelContext_116)->ext_0 = ext_34;
    (&kernelContext_116)->lp_sides_0 = lp_sides_32;
    (&kernelContext_116)->engine_counters_0 = engine_counters_32;
    (&kernelContext_116)->engine_contacts_0 = engine_contacts_32;
    (&kernelContext_116)->cs_anchors_0 = cs_anchors_32;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_116)->g_scan_0 = &g_scan_32;
    uint k_30 = id_29.x;
    if(k_30 >= (params_32->chunk_count_0))
    {
        return;
    }
    uint _S898 = body_of_0(k_30, &kernelContext_116);
    uint _S899 = a_lp_0(_S898, &kernelContext_116);
    uint _S900 = atomic_load_explicit((&kernelContext_116)->atom_0+(_S899 + 13U), memory_order_relaxed);
    bool supported_0 = _S900 != 0U;
    bool fixed_1;
    if(((&kernelContext_116)->info_0[k_30] & 1U) != 0U)
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            uint _S901 = atomic_load_explicit((&kernelContext_116)->atom_0+(_S899 + 12U), memory_order_relaxed);
            fixed_1 = _S901 == k_30;
        }
        else
        {
            fixed_1 = false;
        }
    }
    uint _S902 = lp_f_0(k_30);
    thread float4 f_11 = float4(*((&kernelContext_116)->lp_0+_S902)) ;
    uint _S903 = _S902 + 1U;
    thread float4 t_5 = float4(*((&kernelContext_116)->lp_0+_S903)) ;
    uint b_33 = 7U * _S898;
    float inv_mass_6 = (float4(*((&kernelContext_116)->bodies_0+b_33)) ).w;
    bool _S904;
    if(!supported_0)
    {
        _S904 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S904 = false;
    }
    if(_S904)
    {
        float3 _S905 = read64w3_0(_S899, (&kernelContext_116)->params_0->lp_scale_0, &kernelContext_116);
        float3 _S906 = read64w3_0(_S899 + 6U, (&kernelContext_116)->params_0->lp_scale_0, &kernelContext_116);
        uint _S907 = 2U * _S898;
        Quat_0 p_15 = quat_of_0(float4(*((&kernelContext_116)->frames_0+(_S907 + 1U))) );
        float3 _S908 = (float4(*((&kernelContext_116)->bodies_0+(b_33 + 2U))) ).xyz;
        thread Quat_0 _S909 = p_15;
        Quat_0 _S910 = conj_0(&_S909);
        thread Quat_0 _S911 = _S910;
        float3 _S912 = rotate_0(&_S911, _S906);
        float3 _S913 = _S908 * _S912;
        thread Quat_0 _S914 = p_15;
        float3 _S915 = rotate_0(&_S914, _S913);
        uint _S916 = 4U * k_30;
        float4 _S917 = float4(*((&kernelContext_116)->chunks_0+_S916)) ;
        float4 _S918 = float4(*((&kernelContext_116)->chunks_0+(_S916 + 1U))) ;
        float4 _S919 = float4(*((&kernelContext_116)->chunks_0+(_S916 + 2U))) ;
        f_11.xyz = f_11.xyz - float3(_S917.w)  * (_S905 * float3(inv_mass_6)  + cross(_S915, _S917.xyz - (float4(*((&kernelContext_116)->frames_0+_S907)) ).xyz));
        float _S920 = _S915.x;
        float _S921 = _S918.w;
        float _S922 = _S915.y;
        float _S923 = _S919.x;
        float _S924 = _S915.z;
        float _S925 = _S919.y;
        t_5.xyz = t_5.xyz - float3(_S918.x * _S920 + _S921 * _S922 + _S923 * _S924, _S921 * _S920 + _S918.y * _S922 + _S925 * _S924, _S923 * _S920 + _S925 * _S922 + _S918.z * _S924);
    }
    float _S926;
    if(fixed_1)
    {
        _S926 = 1.0f;
    }
    else
    {
        _S926 = 0.0f;
    }
    f_11.w = _S926;
    *((&kernelContext_116)->lp_0+_S902) = packed_float4(f_11) ;
    *((&kernelContext_116)->lp_0+_S903) = packed_float4(t_5) ;
    return;
}

uint adj_bond_0(uint at_15, KernelContext_0 thread* kernelContext_117)
{
    return (as_type<uint>(((float4(*(kernelContext_117->ext_0+(kernelContext_117->params_0->ext_adj_0 + at_15 / 4U))) )[at_15 % 4U])));
}

uint lp_intact_0(uint k_31, KernelContext_0 thread* kernelContext_118)
{
    float4 _S927 = float4(*(kernelContext_118->ext_0+(kernelContext_118->params_0->ext_chunks_0 + 4U * k_31))) ;
    uint _S928 = (as_type<uint>((_S927.x)));
    uint _S929 = (as_type<uint>((_S927.y)));
    uint e_3 = 0U;
    uint intact_0 = 0U;
    for(;;)
    {
        if(e_3 < _S929)
        {
        }
        else
        {
            break;
        }
        uint _S930 = w_alive_0(kernelContext_118);
        uint _S931 = adj_bond_0(_S928 + e_3, kernelContext_118);
        uint _S932;
        if((*(kernelContext_118->work_0+(_S930 + _S931))) != 0U)
        {
            _S932 = 1U;
        }
        else
        {
            _S932 = 0U;
        }
        uint intact_1 = intact_0 + _S932;
        e_3 = e_3 + 1U;
        intact_0 = intact_1;
    }
    return intact_0;
}

uint lp_l_0(uint k_32)
{
    return 22U * k_32 + 8U;
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

LpBond_0 lp_bond_0(uint i_8, KernelContext_0 thread* kernelContext_119)
{
    uint o_7 = 10U * i_8;
    thread LpBond_0 b_34;
    (&b_34)->centroid_0 = (float4(*(kernelContext_119->ext_0+o_7)) ).xyz;
    float3 _S933 = (float4(*(kernelContext_119->ext_0+(o_7 + 1U))) ).xyz;
    (&b_34)->n_4 = _S933;
    float3 _S934 = (float4(*(kernelContext_119->ext_0+(o_7 + 2U))) ).xyz;
    (&b_34)->t1_2 = _S934;
    (&b_34)->t2_0 = cross(_S933, _S934);
    (&(&b_34)->law_0)->geom0_0 = float4(*(kernelContext_119->ext_0+(o_7 + 3U))) ;
    (&(&b_34)->law_0)->geom1_0 = float4(*(kernelContext_119->ext_0+(o_7 + 4U))) ;
    (&(&b_34)->law_0)->stiff0_0 = float4(*(kernelContext_119->ext_0+(o_7 + 5U))) ;
    (&(&b_34)->law_0)->stiff1_0 = float4(*(kernelContext_119->ext_0+(o_7 + 6U))) ;
    (&(&b_34)->law_0)->rebar0_0 = float4(*(kernelContext_119->ext_0+(o_7 + 7U))) ;
    (&(&b_34)->law_0)->rebar1_0 = float4(*(kernelContext_119->ext_0+(o_7 + 8U))) ;
    (&(&b_34)->law_0)->ids_3 = (as_type<uint4>((float4(*(kernelContext_119->ext_0+(o_7 + 9U))) )));
    return b_34;
}

uint tri_0(uint i_9, uint j_4)
{
    return i_9 * (i_9 + 1U) / 2U + j_4;
}

bool lp_factor_0(uint k_33, KernelContext_0 thread* kernelContext_120)
{
    uint q_26;
    uint a_8;
    uint p2_0;
    uint q2_0;
    uint a2_0;
    float s_11;
    uint _S935 = lp_intact_0(k_33, kernelContext_120);
    uint _S936 = lp_l_0(k_33);
    uint _S937 = _S935 + 1U;
    if((as_type<uint>(((float4(*(kernelContext_120->lp_0+(_S936 + 5U))) ).w))) == _S937)
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
    uint _S938 = 4U * k_33;
    float4 _S939 = float4(*(kernelContext_120->ext_0+(kernelContext_120->params_0->ext_chunks_0 + _S938))) ;
    uint _S940 = (as_type<uint>((_S939.x)));
    uint _S941 = (as_type<uint>((_S939.y)));
    q_26 = 0U;
    for(;;)
    {
        if(q_26 < _S941)
        {
        }
        else
        {
            break;
        }
        uint _S942 = adj_bond_0(_S940 + q_26, kernelContext_120);
        uint _S943 = w_alive_0(kernelContext_120);
        if((*(kernelContext_120->work_0+(_S943 + _S942))) == 0U)
        {
            q_26 = q_26 + 1U;
            continue;
        }
        LpBond_0 _S944 = lp_bond_0(_S942, kernelContext_120);
        float3 _S945 = _S944.centroid_0 - (float4(*(kernelContext_120->chunks_0+_S938)) ).xyz;
        float _S946 = _S944.law_0.stiff0_0.y;
        array<float, int(6)> _S947 = { { _S946, _S946, _S944.law_0.stiff0_0.x, _S944.law_0.stiff0_0.z, _S944.law_0.stiff0_0.w, _S944.law_0.stiff1_0.x } };
        array<float3, int(3)> _S948 = { { _S944.t1_2, _S944.t2_0, _S944.n_4 } };
        thread array<array<float, int(6)>, int(6)> j_5;
        a_8 = 0U;
        for(;;)
        {
            if(a_8 < 3U)
            {
            }
            else
            {
                break;
            }
            float3 c_19 = cross(_S945, _S948[a_8]);
            float _S949 = _S948[a_8].x;
            j_5[a_8][int(0)] = _S949;
            float _S950 = _S948[a_8].y;
            j_5[a_8][int(1)] = _S950;
            float _S951 = _S948[a_8].z;
            j_5[a_8][int(2)] = _S951;
            j_5[a_8][int(3)] = c_19.x;
            j_5[a_8][int(4)] = c_19.y;
            j_5[a_8][int(5)] = c_19.z;
            j_5[3U + a_8][int(0)] = 0.0f;
            j_5[3U + a_8][int(1)] = 0.0f;
            j_5[3U + a_8][int(2)] = 0.0f;
            j_5[3U + a_8][int(3)] = _S949;
            j_5[3U + a_8][int(4)] = _S950;
            j_5[3U + a_8][int(5)] = _S951;
            a_8 = a_8 + 1U;
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
                    float s_12 = s_11 + j_5[a2_0][p2_0] * _S947[a2_0] * j_5[a2_0][q2_0];
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
    a_8 = 0U;
    for(;;)
    {
        if(a_8 < 21U)
        {
        }
        else
        {
            break;
        }
        l_0[a_8] = 0.0f;
        a_8 = a_8 + 1U;
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
                bool _S952;
                if(ok_1)
                {
                    _S952 = s_11 > 0.0f;
                }
                else
                {
                    _S952 = false;
                }
                l_0[tri_0(p2_0, p2_0)] = sqrt(max(s_11, 1.00000000317107685e-30f));
                ok_1 = _S952;
            }
            else
            {
                l_0[tri_0(p2_0, q2_0)] = s_11 / l_0[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + 1U;
        }
        uint _S953 = p2_0 + 1U;
        ok_0 = ok_1;
        p2_0 = _S953;
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
        packed_float4 device* _S954 = kernelContext_120->lp_0+(_S936 + q2_0);
        uint _S955 = 4U * q2_0;
        float _S956 = l_0[_S955];
        uint _S957 = _S955 + 1U;
        if(_S957 < 21U)
        {
            s_11 = l_0[_S957];
        }
        else
        {
            s_11 = 0.0f;
        }
        uint _S958 = _S955 + 2U;
        float _S959;
        if(_S958 < 21U)
        {
            _S959 = l_0[_S958];
        }
        else
        {
            _S959 = 0.0f;
        }
        float _S960;
        if(q2_0 == 5U)
        {
            _S960 = (as_type<float>((_S937)));
        }
        else
        {
            _S960 = l_0[_S955 + 3U];
        }
        *_S954 = packed_float4(float4(_S956, s_11, _S959, _S960)) ;
        q2_0 = q2_0 + 1U;
    }
    return true;
}

[[kernel]] void fr_lp_block(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], atomic<uint> device* atom_33 [[buffer(5)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], packed_float4 device* lp_33 [[buffer(24)]], packed_float4 device* ext_35 [[buffer(23)]], packed_float4 device* lp_sides_33 [[buffer(28)]], uint device* engine_counters_33 [[buffer(26)]], packed_float4 device* engine_contacts_33 [[buffer(25)]], packed_float4 device* cs_anchors_33 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_121;
    (&kernelContext_121)->params_0 = params_33;
    (&kernelContext_121)->work_0 = work_33;
    (&kernelContext_121)->frames_0 = frames_33;
    (&kernelContext_121)->bodies_0 = bodies_33;
    (&kernelContext_121)->chunks_0 = chunks_33;
    (&kernelContext_121)->sort_0 = sort_33;
    (&kernelContext_121)->atom_0 = atom_33;
    (&kernelContext_121)->manifolds_0 = manifolds_33;
    (&kernelContext_121)->loads_0 = loads_33;
    (&kernelContext_121)->bonds_0 = bonds_33;
    (&kernelContext_121)->args_0 = args_33;
    (&kernelContext_121)->info_0 = info_33;
    (&kernelContext_121)->scan_0 = scan_33;
    (&kernelContext_121)->parts_0 = parts_33;
    (&kernelContext_121)->colliders_0 = colliders_33;
    (&kernelContext_121)->collider_info_0 = collider_info_33;
    (&kernelContext_121)->view_chunks_0 = view_chunks_33;
    (&kernelContext_121)->view_0 = view_33;
    (&kernelContext_121)->objects_0 = objects_33;
    (&kernelContext_121)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_121)->lines_0 = lines_33;
    (&kernelContext_121)->geometry_0 = geometry_33;
    (&kernelContext_121)->shots_0 = shots_33;
    (&kernelContext_121)->lp_0 = lp_33;
    (&kernelContext_121)->ext_0 = ext_35;
    (&kernelContext_121)->lp_sides_0 = lp_sides_33;
    (&kernelContext_121)->engine_counters_0 = engine_counters_33;
    (&kernelContext_121)->engine_contacts_0 = engine_contacts_33;
    (&kernelContext_121)->cs_anchors_0 = cs_anchors_33;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_121)->g_scan_0 = &g_scan_33;
    uint k_34 = id_30.x;
    bool _S961;
    if(k_34 >= (params_33->chunk_count_0))
    {
        _S961 = true;
    }
    else
    {
        _S961 = ((float4(*((&kernelContext_121)->lp_0+lp_f_0(k_34))) ).w) != 0.0f;
    }
    if(_S961)
    {
        return;
    }
    bool _S962 = lp_factor_0(k_34, &kernelContext_121);
    if(!_S962)
    {
        (*((&kernelContext_121)->lp_0+lp_f_0(k_34)))[int(3)] = 1.0f;
        uint _S963 = a_counters_1(&kernelContext_121);
        uint _S964 = atomic_fetch_or_explicit((&kernelContext_121)->atom_0+(_S963 + 3U), 32U, memory_order_relaxed);
    }
    return;
}

uint st_x_0(uint k_35, uint state_1, uint copy_0)
{
    return 22U * k_35 + state_1 + 2U * copy_0;
}

uint st_d_0(uint k_36, uint state_2)
{
    return 22U * k_36 + state_2 + 4U;
}

void st_sweep_0(uint k_37, uint state_3, uint from_4, float c1_1, float c2_1, KernelContext_0 thread* kernelContext_122)
{
    uint ka_0;
    uint kb_0;
    float s_14;
    uint to_3 = 1U - from_4;
    uint _S965 = st_x_0(k_37, state_3, from_4);
    float4 _S966 = float4(*(kernelContext_122->lp_0+_S965)) ;
    float4 _S967 = float4(*(kernelContext_122->lp_0+(_S965 + 1U))) ;
    uint _S968 = st_f_0(k_37, state_3);
    float4 _S969 = float4(*(kernelContext_122->lp_0+_S968)) ;
    if((_S969.w) != 0.0f)
    {
        uint _S970 = st_x_0(k_37, state_3, to_3);
        *(kernelContext_122->lp_0+_S970) = packed_float4(_S966) ;
        *(kernelContext_122->lp_0+(_S970 + 1U)) = packed_float4(_S967) ;
        uint _S971 = st_d_0(k_37, state_3);
        packed_float4 _S972 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_122->lp_0+_S971) = _S972;
        *(kernelContext_122->lp_0+(_S971 + 1U)) = _S972;
        return;
    }
    float _S973 = _S969.y;
    float _S974 = _S969.z;
    thread array<float, int(6)> r_22;
    r_22[int(0)] = _S969.x;
    r_22[int(1)] = _S973;
    r_22[int(2)] = _S974;
    r_22[int(3)] = 0.0f;
    r_22[int(4)] = 0.0f;
    r_22[int(5)] = 0.0f;
    float3 t_6 = (float4(*(kernelContext_122->lp_0+(_S968 + 1U))) ).xyz;
    r_22[int(3)] = t_6.x;
    r_22[int(4)] = t_6.y;
    r_22[int(5)] = t_6.z;
    float4 _S975 = float4(*(kernelContext_122->ext_0+(kernelContext_122->params_0->ext_chunks_0 + 4U * k_37))) ;
    uint _S976 = (as_type<uint>((_S975.x)));
    uint _S977 = (as_type<uint>((_S975.y)));
    uint e_4 = 0U;
    for(;;)
    {
        if(e_4 < _S977)
        {
        }
        else
        {
            break;
        }
        uint _S978 = 5U * (_S976 + e_4);
        float4 _S979 = float4(*(kernelContext_122->lp_sides_0+_S978)) ;
        uint _S980 = w_alive_0(kernelContext_122);
        if((*(kernelContext_122->work_0+(_S980 + (as_type<uint>((_S979.w)))))) == 0U)
        {
            e_4 = e_4 + 1U;
            continue;
        }
        float4 _S981 = float4(*(kernelContext_122->lp_sides_0+(_S978 + 1U))) ;
        float4 _S982 = float4(*(kernelContext_122->lp_sides_0+(_S978 + 2U))) ;
        float4 _S983 = float4(*(kernelContext_122->lp_sides_0+(_S978 + 3U))) ;
        float4 _S984 = float4(*(kernelContext_122->lp_sides_0+(_S978 + 4U))) ;
        bool first_side_0 = (_S984.w) != 0.0f;
        if(first_side_0)
        {
            ka_0 = k_37;
        }
        else
        {
            ka_0 = (as_type<uint>((_S981.w)));
        }
        if(first_side_0)
        {
            kb_0 = (as_type<uint>((_S981.w)));
        }
        else
        {
            kb_0 = k_37;
        }
        float3 ra_0 = _S979.xyz;
        float3 rb_0 = _S981.xyz;
        float3 t1_3 = _S982.xyz;
        float3 n_5 = _S983.xyz;
        float3 t2_1 = cross(n_5, t1_3);
        uint _S985 = st_x_0(ka_0, state_3, from_4);
        uint _S986 = st_x_0(kb_0, state_3, from_4);
        float3 _S987 = (float4(*(kernelContext_122->lp_0+(_S986 + 1U))) ).xyz;
        float3 _S988 = (float4(*(kernelContext_122->lp_0+(_S985 + 1U))) ).xyz;
        float3 delta_0 = (float4(*(kernelContext_122->lp_0+_S986)) ).xyz + cross(_S987, rb_0) - ((float4(*(kernelContext_122->lp_0+_S985)) ).xyz + cross(_S988, ra_0));
        float3 phi_0 = _S987 - _S988;
        float _S989 = _S982.w;
        float3 fb_0 = t1_3 * float3((_S989 * dot(delta_0, t1_3)))  + t2_1 * float3((_S989 * dot(delta_0, t2_1)))  + n_5 * float3((_S983.w * dot(delta_0, n_5))) ;
        float3 mb_0 = t1_3 * float3((_S984.x * dot(phi_0, t1_3)))  + t2_1 * float3((_S984.y * dot(phi_0, t2_1)))  + n_5 * float3((_S984.z * dot(phi_0, n_5))) ;
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
        float3 _S990;
        if(first_side_0)
        {
            _S990 = mb_0;
        }
        else
        {
            _S990 = - mb_0;
        }
        float3 lm_0 = _S990 + cross(arm_0, lf_0);
        r_22[int(0)] = r_22[int(0)] + lf_0.x;
        r_22[int(1)] = r_22[int(1)] + lf_0.y;
        r_22[int(2)] = r_22[int(2)] + lf_0.z;
        r_22[int(3)] = r_22[int(3)] + lm_0.x;
        r_22[int(4)] = r_22[int(4)] + lm_0.y;
        r_22[int(5)] = r_22[int(5)] + lm_0.z;
        e_4 = e_4 + 1U;
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
        float4 _S991 = float4(*(kernelContext_122->lp_0+(lp_l_0(k_37) + kb_0))) ;
        uint _S992 = 4U * kb_0;
        l_1[_S992] = _S991.x;
        uint _S993 = _S992 + 1U;
        if(_S993 < 21U)
        {
            l_1[_S993] = _S991.y;
        }
        uint _S994 = _S992 + 2U;
        if(_S994 < 21U)
        {
            l_1[_S994] = _S991.z;
        }
        uint _S995 = _S992 + 3U;
        if(_S995 < 21U)
        {
            l_1[_S995] = _S991.w;
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
        s_14 = r_22[i2_0];
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
        uint _S996 = uint(i3_0);
        z_3[i3_0] = s_14 / l_1[tri_0(_S996, _S996)];
        i3_0 = i3_0 - int(1);
    }
    uint _S997 = st_d_0(k_37, state_3);
    float4 _S998 = float4(*(kernelContext_122->lp_0+_S997)) ;
    thread float4 d0_0 = _S998;
    uint _S999 = _S997 + 1U;
    float4 _S1000 = float4(*(kernelContext_122->lp_0+_S999)) ;
    thread float4 d1_0 = _S1000;
    float3 _S1001 = float3(c1_1) ;
    float3 _S1002 = float3(c2_1) ;
    d0_0.xyz = _S1001 * _S998.xyz + _S1002 * float3(z_3[int(0)], z_3[int(1)], z_3[int(2)]);
    d1_0.xyz = _S1001 * _S1000.xyz + _S1002 * float3(z_3[int(3)], z_3[int(4)], z_3[int(5)]);
    *(kernelContext_122->lp_0+_S997) = packed_float4(d0_0) ;
    *(kernelContext_122->lp_0+_S999) = packed_float4(d1_0) ;
    uint _S1003 = st_x_0(k_37, state_3, to_3);
    *(kernelContext_122->lp_0+_S1003) = packed_float4(float4(_S966.xyz + d0_0.xyz, 0.0f)) ;
    *(kernelContext_122->lp_0+(_S1003 + 1U)) = packed_float4(float4(_S967.xyz + d1_0.xyz, 0.0f)) ;
    return;
}

void lp_sweep_0(uint k_38, uint from_5, KernelContext_0 thread* kernelContext_123)
{
    st_sweep_0(k_38, 0U, from_5, kernelContext_123->params_0->lp_c1_0, kernelContext_123->params_0->lp_c2_0, kernelContext_123);
    return;
}

[[kernel]] void fr_lp_sweep_even(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], atomic<uint> device* atom_34 [[buffer(5)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], packed_float4 device* lp_34 [[buffer(24)]], packed_float4 device* ext_36 [[buffer(23)]], packed_float4 device* lp_sides_34 [[buffer(28)]], uint device* engine_counters_34 [[buffer(26)]], packed_float4 device* engine_contacts_34 [[buffer(25)]], packed_float4 device* cs_anchors_34 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_124;
    (&kernelContext_124)->params_0 = params_34;
    (&kernelContext_124)->work_0 = work_34;
    (&kernelContext_124)->frames_0 = frames_34;
    (&kernelContext_124)->bodies_0 = bodies_34;
    (&kernelContext_124)->chunks_0 = chunks_34;
    (&kernelContext_124)->sort_0 = sort_34;
    (&kernelContext_124)->atom_0 = atom_34;
    (&kernelContext_124)->manifolds_0 = manifolds_34;
    (&kernelContext_124)->loads_0 = loads_34;
    (&kernelContext_124)->bonds_0 = bonds_34;
    (&kernelContext_124)->args_0 = args_34;
    (&kernelContext_124)->info_0 = info_34;
    (&kernelContext_124)->scan_0 = scan_34;
    (&kernelContext_124)->parts_0 = parts_34;
    (&kernelContext_124)->colliders_0 = colliders_34;
    (&kernelContext_124)->collider_info_0 = collider_info_34;
    (&kernelContext_124)->view_chunks_0 = view_chunks_34;
    (&kernelContext_124)->view_0 = view_34;
    (&kernelContext_124)->objects_0 = objects_34;
    (&kernelContext_124)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_124)->lines_0 = lines_34;
    (&kernelContext_124)->geometry_0 = geometry_34;
    (&kernelContext_124)->shots_0 = shots_34;
    (&kernelContext_124)->lp_0 = lp_34;
    (&kernelContext_124)->ext_0 = ext_36;
    (&kernelContext_124)->lp_sides_0 = lp_sides_34;
    (&kernelContext_124)->engine_counters_0 = engine_counters_34;
    (&kernelContext_124)->engine_contacts_0 = engine_contacts_34;
    (&kernelContext_124)->cs_anchors_0 = cs_anchors_34;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_124)->g_scan_0 = &g_scan_34;
    uint _S1004 = id_31.x;
    if(_S1004 < (params_34->chunk_count_0))
    {
        lp_sweep_0(_S1004, 0U, &kernelContext_124);
    }
    return;
}

[[kernel]] void fr_lp_sweep_odd(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], atomic<uint> device* atom_35 [[buffer(5)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], packed_float4 device* lp_35 [[buffer(24)]], packed_float4 device* ext_37 [[buffer(23)]], packed_float4 device* lp_sides_35 [[buffer(28)]], uint device* engine_counters_35 [[buffer(26)]], packed_float4 device* engine_contacts_35 [[buffer(25)]], packed_float4 device* cs_anchors_35 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_125;
    (&kernelContext_125)->params_0 = params_35;
    (&kernelContext_125)->work_0 = work_35;
    (&kernelContext_125)->frames_0 = frames_35;
    (&kernelContext_125)->bodies_0 = bodies_35;
    (&kernelContext_125)->chunks_0 = chunks_35;
    (&kernelContext_125)->sort_0 = sort_35;
    (&kernelContext_125)->atom_0 = atom_35;
    (&kernelContext_125)->manifolds_0 = manifolds_35;
    (&kernelContext_125)->loads_0 = loads_35;
    (&kernelContext_125)->bonds_0 = bonds_35;
    (&kernelContext_125)->args_0 = args_35;
    (&kernelContext_125)->info_0 = info_35;
    (&kernelContext_125)->scan_0 = scan_35;
    (&kernelContext_125)->parts_0 = parts_35;
    (&kernelContext_125)->colliders_0 = colliders_35;
    (&kernelContext_125)->collider_info_0 = collider_info_35;
    (&kernelContext_125)->view_chunks_0 = view_chunks_35;
    (&kernelContext_125)->view_0 = view_35;
    (&kernelContext_125)->objects_0 = objects_35;
    (&kernelContext_125)->shot_bodies_0 = shot_bodies_35;
    (&kernelContext_125)->lines_0 = lines_35;
    (&kernelContext_125)->geometry_0 = geometry_35;
    (&kernelContext_125)->shots_0 = shots_35;
    (&kernelContext_125)->lp_0 = lp_35;
    (&kernelContext_125)->ext_0 = ext_37;
    (&kernelContext_125)->lp_sides_0 = lp_sides_35;
    (&kernelContext_125)->engine_counters_0 = engine_counters_35;
    (&kernelContext_125)->engine_contacts_0 = engine_contacts_35;
    (&kernelContext_125)->cs_anchors_0 = cs_anchors_35;
    threadgroup array<uint, int(1024)> g_scan_35;
    (&kernelContext_125)->g_scan_0 = &g_scan_35;
    uint _S1005 = id_32.x;
    if(_S1005 < (params_35->chunk_count_0))
    {
        lp_sweep_0(_S1005, 1U, &kernelContext_125);
    }
    return;
}

void st_force_0(const LpBond_0 thread* b_35, uint ka_1, uint kb_1, uint state_4, uint c_20, float3 thread* q_lin_0, float3 thread* q_ang_0, KernelContext_0 thread* kernelContext_126)
{
    uint _S1006 = st_x_0(ka_1, state_4, c_20);
    uint _S1007 = st_x_0(kb_1, state_4, c_20);
    float3 _S1008 = (float4(*(kernelContext_126->lp_0+(_S1007 + 1U))) ).xyz;
    float3 _S1009 = (float4(*(kernelContext_126->lp_0+(_S1006 + 1U))) ).xyz;
    float3 delta_1 = (float4(*(kernelContext_126->lp_0+_S1007)) ).xyz + cross(_S1008, b_35->centroid_0 - (float4(*(kernelContext_126->chunks_0+4U * kb_1)) ).xyz) - ((float4(*(kernelContext_126->lp_0+_S1006)) ).xyz + cross(_S1009, b_35->centroid_0 - (float4(*(kernelContext_126->chunks_0+4U * ka_1)) ).xyz));
    float3 phi_1 = _S1008 - _S1009;
    float4 _S1010 = (&b_35->law_0)->stiff0_0;
    float kt_0 = (&b_35->law_0)->stiff1_0.x;
    float _S1011 = (&b_35->law_0)->stiff0_0.y;
    float3 _S1012 = b_35->t1_2;
    float3 _S1013 = b_35->t2_0;
    float3 _S1014 = b_35->n_4;
    *q_lin_0 = float3(_S1011 * dot(delta_1, b_35->t1_2), _S1011 * dot(delta_1, b_35->t2_0), (&b_35->law_0)->stiff0_0.x * dot(delta_1, b_35->n_4));
    *q_ang_0 = float3(_S1010.z * dot(phi_1, _S1012), _S1010.w * dot(phi_1, _S1013), kt_0 * dot(phi_1, _S1014));
    return;
}

void st_force_1(const LpBond_0 thread* b_36, uint ka_2, uint kb_2, uint state_5, uint c_21, float3 thread* q_lin_1, float3 thread* q_ang_1, KernelContext_0 thread* kernelContext_127)
{
    uint _S1015 = st_x_0(ka_2, state_5, c_21);
    uint _S1016 = st_x_0(kb_2, state_5, c_21);
    float3 _S1017 = (float4(*(kernelContext_127->lp_0+(_S1016 + 1U))) ).xyz;
    float3 _S1018 = (float4(*(kernelContext_127->lp_0+(_S1015 + 1U))) ).xyz;
    float3 delta_2 = (float4(*(kernelContext_127->lp_0+_S1016)) ).xyz + cross(_S1017, b_36->centroid_0 - (float4(*(kernelContext_127->chunks_0+4U * kb_2)) ).xyz) - ((float4(*(kernelContext_127->lp_0+_S1015)) ).xyz + cross(_S1018, b_36->centroid_0 - (float4(*(kernelContext_127->chunks_0+4U * ka_2)) ).xyz));
    float3 phi_2 = _S1017 - _S1018;
    float4 _S1019 = (&b_36->law_0)->stiff0_0;
    float kt_1 = (&b_36->law_0)->stiff1_0.x;
    float _S1020 = (&b_36->law_0)->stiff0_0.y;
    float3 _S1021 = b_36->t1_2;
    float3 _S1022 = b_36->t2_0;
    float3 _S1023 = b_36->n_4;
    *q_lin_1 = float3(_S1020 * dot(delta_2, b_36->t1_2), _S1020 * dot(delta_2, b_36->t2_0), (&b_36->law_0)->stiff0_0.x * dot(delta_2, b_36->n_4));
    *q_ang_1 = float3(_S1019.z * dot(phi_2, _S1021), _S1019.w * dot(phi_2, _S1022), kt_1 * dot(phi_2, _S1023));
    return;
}

void lp_force_0(const LpBond_0 thread* b_37, uint ka_3, uint kb_3, uint c_22, float3 thread* q_lin_2, float3 thread* q_ang_2, KernelContext_0 thread* kernelContext_128)
{
    st_force_1(b_37, ka_3, kb_3, 0U, c_22, q_lin_2, q_ang_2, kernelContext_128);
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
    float _S1024 = q_lin_3.z;
    float axial_0 = (metal::fast::divide((_S1024), (area_0)));
    float _S1025 = (metal::fast::divide((abs(q_ang_3.x)), (b_38->geom1_0.x)));
    float _S1026 = (metal::fast::divide((abs(q_ang_3.y)), (b_38->geom1_0.y)));
    float bending_0 = _S1025 + _S1026;
    float _S1027 = q_lin_3.x;
    float _S1028 = q_lin_3.y;
    float _S1029 = (metal::fast::sqrt((_S1027 * _S1027 + _S1028 * _S1028)));
    float _S1030 = (metal::fast::divide((_S1029), (area_0)));
    float _S1031 = (metal::fast::divide((abs(q_ang_3.z)), (b_38->geom0_0.w)));
    float shear_1 = _S1030 + _S1031;
    thread Measures_0 m_3;
    (&m_3)->tension_0 = axial_0 + bending_0;
    (&m_3)->shear_0 = shear_1;
    float _S1032 = - axial_0;
    (&m_3)->normal_compression_0 = max(_S1032, 0.0f);
    (&m_3)->compression_0 = _S1032 + bending_0;
    (&m_3)->compressive_force_0 = max(- _S1024, 0.0f);
    return m_3;
}

JointMaterial_0 lp_material_0(uint m_4, KernelContext_0 thread* kernelContext_129)
{
    uint o_8 = kernelContext_129->params_0->ext_mat_0 + 5U * m_4;
    thread JointMaterial_0 j_6;
    (&j_6)->strength_0 = float4(*(kernelContext_129->ext_0+o_8)) ;
    (&j_6)->energy_0 = float4(*(kernelContext_129->ext_0+(o_8 + 1U))) ;
    (&j_6)->dif_0 = float4(*(kernelContext_129->ext_0+(o_8 + 2U))) ;
    (&j_6)->misc_0 = float4(*(kernelContext_129->ext_0+(o_8 + 3U))) ;
    (&j_6)->kind_flags_0 = (as_type<uint4>((float4(*(kernelContext_129->ext_0+(o_8 + 4U))) )));
    return j_6;
}

float infinity_0()
{
    return float_bits_0(2139095040U);
}

float4 failure_indices_0(const JointMaterial_0 thread* mat_1, const JointBond_0 thread* b_39, const Measures_0 thread* m_5, float multiplier_0)
{
    float fc_2 = mat_1->strength_0.y * multiplier_0;
    float _S1033 = min(mat_1->strength_0.z * multiplier_0 + mat_1->strength_0.w * m_5->normal_compression_0, mat_1->energy_0.x * multiplier_0);
    thread float4 idx_0;
    float _S1034 = (metal::fast::divide((m_5->tension_0), (mat_1->strength_0.x * multiplier_0)));
    idx_0.x = max(_S1034, 0.0f);
    float _S1035;
    if(_S1033 > 0.0f)
    {
        float _S1036 = (metal::fast::divide((m_5->shear_0), (_S1033)));
        _S1035 = _S1036;
    }
    else
    {
        _S1035 = infinity_0();
    }
    idx_0.y = _S1035;
    float _S1037 = (metal::fast::divide((m_5->compression_0), (fc_2)));
    idx_0.z = max(_S1037, 0.0f);
    float _S1038 = b_39->stiff1_0.y;
    if(_S1038 > 0.0f)
    {
        float _S1039 = (metal::fast::divide((m_5->compressive_force_0), (_S1038)));
        _S1035 = _S1039;
    }
    else
    {
        _S1035 = 0.0f;
    }
    idx_0.w = _S1035;
    return idx_0;
}

[[kernel]] void fr_lp_breaks(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], atomic<uint> device* atom_36 [[buffer(5)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], packed_float4 device* lp_36 [[buffer(24)]], packed_float4 device* ext_38 [[buffer(23)]], packed_float4 device* lp_sides_36 [[buffer(28)]], uint device* engine_counters_36 [[buffer(26)]], packed_float4 device* engine_contacts_36 [[buffer(25)]], packed_float4 device* cs_anchors_36 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_130;
    (&kernelContext_130)->params_0 = params_36;
    (&kernelContext_130)->work_0 = work_36;
    (&kernelContext_130)->frames_0 = frames_36;
    (&kernelContext_130)->bodies_0 = bodies_36;
    (&kernelContext_130)->chunks_0 = chunks_36;
    (&kernelContext_130)->sort_0 = sort_36;
    (&kernelContext_130)->atom_0 = atom_36;
    (&kernelContext_130)->manifolds_0 = manifolds_36;
    (&kernelContext_130)->loads_0 = loads_36;
    (&kernelContext_130)->bonds_0 = bonds_36;
    (&kernelContext_130)->args_0 = args_36;
    (&kernelContext_130)->info_0 = info_36;
    (&kernelContext_130)->scan_0 = scan_36;
    (&kernelContext_130)->parts_0 = parts_36;
    (&kernelContext_130)->colliders_0 = colliders_36;
    (&kernelContext_130)->collider_info_0 = collider_info_36;
    (&kernelContext_130)->view_chunks_0 = view_chunks_36;
    (&kernelContext_130)->view_0 = view_36;
    (&kernelContext_130)->objects_0 = objects_36;
    (&kernelContext_130)->shot_bodies_0 = shot_bodies_36;
    (&kernelContext_130)->lines_0 = lines_36;
    (&kernelContext_130)->geometry_0 = geometry_36;
    (&kernelContext_130)->shots_0 = shots_36;
    (&kernelContext_130)->lp_0 = lp_36;
    (&kernelContext_130)->ext_0 = ext_38;
    (&kernelContext_130)->lp_sides_0 = lp_sides_36;
    (&kernelContext_130)->engine_counters_0 = engine_counters_36;
    (&kernelContext_130)->engine_contacts_0 = engine_contacts_36;
    (&kernelContext_130)->cs_anchors_0 = cs_anchors_36;
    threadgroup array<uint, int(1024)> g_scan_36;
    (&kernelContext_130)->g_scan_0 = &g_scan_36;
    uint i_10 = id_33.x;
    bool _S1040;
    if(i_10 >= (params_36->bond_count_0))
    {
        _S1040 = true;
    }
    else
    {
        uint _S1041 = w_alive_0(&kernelContext_130);
        _S1040 = (*((&kernelContext_130)->work_0+(_S1041 + i_10))) == 0U;
    }
    if(_S1040)
    {
        return;
    }
    uint4 _S1042 = uint4(*((&kernelContext_130)->bonds_0+i_10)) ;
    LpBond_0 _S1043 = lp_bond_0(i_10, &kernelContext_130);
    uint _S1044 = _S1042.x;
    uint _S1045 = _S1042.y;
    thread LpBond_0 _S1046 = _S1043;
    thread float3 q_lin_4;
    thread float3 q_ang_4;
    lp_force_0(&_S1046, _S1044, _S1045, 0U, &q_lin_4, &q_ang_4, &kernelContext_130);
    thread JointBond_0 _S1047 = _S1043.law_0;
    Measures_0 _S1048 = stress_measures_0(&_S1047, q_lin_4, q_ang_4);
    JointMaterial_0 _S1049 = lp_material_0(_S1043.law_0.ids_3.x, &kernelContext_130);
    float _S1050 = _S1043.law_0.geom1_0.w;
    thread JointMaterial_0 _S1051 = _S1049;
    thread JointBond_0 _S1052 = _S1043.law_0;
    thread Measures_0 _S1053 = _S1048;
    float4 _S1054 = failure_indices_0(&_S1051, &_S1052, &_S1053, _S1050);
    float _S1055 = max(max(_S1054.x, _S1054.y), max(_S1054.z, _S1054.w));
    uint _S1056 = a_counters_1(&kernelContext_130);
    uint _S1057 = atomic_fetch_max_explicit((&kernelContext_130)->atom_0+(_S1056 + 11U), (as_type<uint>((max(_S1055, 0.0f)))), memory_order_relaxed);
    if(_S1055 >= 1.0f)
    {
        uint _S1058 = w_alive_0(&kernelContext_130);
        *((&kernelContext_130)->work_0+(_S1058 + i_10)) = 0U;
        uint _S1059 = a_dirty_0(&kernelContext_130);
        uint _S1060 = body_of_0(_S1044, &kernelContext_130);
        uint _S1061 = atomic_fetch_or_explicit((&kernelContext_130)->atom_0+(_S1059 + _S1060), 1U, memory_order_relaxed);
        uint _S1062 = atomic_fetch_add_explicit((&kernelContext_130)->atom_0+_S1056, 1U, memory_order_relaxed);
        uint _S1063 = atomic_fetch_add_explicit((&kernelContext_130)->atom_0+(_S1056 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first_reset(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], atomic<uint> device* atom_37 [[buffer(5)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], packed_float4 device* lp_37 [[buffer(24)]], packed_float4 device* ext_39 [[buffer(23)]], packed_float4 device* lp_sides_37 [[buffer(28)]], uint device* engine_counters_37 [[buffer(26)]], packed_float4 device* engine_contacts_37 [[buffer(25)]], packed_float4 device* cs_anchors_37 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_131;
    (&kernelContext_131)->params_0 = params_37;
    (&kernelContext_131)->work_0 = work_37;
    (&kernelContext_131)->frames_0 = frames_37;
    (&kernelContext_131)->bodies_0 = bodies_37;
    (&kernelContext_131)->chunks_0 = chunks_37;
    (&kernelContext_131)->sort_0 = sort_37;
    (&kernelContext_131)->atom_0 = atom_37;
    (&kernelContext_131)->manifolds_0 = manifolds_37;
    (&kernelContext_131)->loads_0 = loads_37;
    (&kernelContext_131)->bonds_0 = bonds_37;
    (&kernelContext_131)->args_0 = args_37;
    (&kernelContext_131)->info_0 = info_37;
    (&kernelContext_131)->scan_0 = scan_37;
    (&kernelContext_131)->parts_0 = parts_37;
    (&kernelContext_131)->colliders_0 = colliders_37;
    (&kernelContext_131)->collider_info_0 = collider_info_37;
    (&kernelContext_131)->view_chunks_0 = view_chunks_37;
    (&kernelContext_131)->view_0 = view_37;
    (&kernelContext_131)->objects_0 = objects_37;
    (&kernelContext_131)->shot_bodies_0 = shot_bodies_37;
    (&kernelContext_131)->lines_0 = lines_37;
    (&kernelContext_131)->geometry_0 = geometry_37;
    (&kernelContext_131)->shots_0 = shots_37;
    (&kernelContext_131)->lp_0 = lp_37;
    (&kernelContext_131)->ext_0 = ext_39;
    (&kernelContext_131)->lp_sides_0 = lp_sides_37;
    (&kernelContext_131)->engine_counters_0 = engine_counters_37;
    (&kernelContext_131)->engine_contacts_0 = engine_contacts_37;
    (&kernelContext_131)->cs_anchors_0 = cs_anchors_37;
    threadgroup array<uint, int(1024)> g_scan_37;
    (&kernelContext_131)->g_scan_0 = &g_scan_37;
    uint _S1064 = id_34.x;
    if(_S1064 < (params_37->shot_count_0))
    {
        uint _S1065 = a_first_1(_S1064, &kernelContext_131);
        atomic_store_explicit((&kernelContext_131)->atom_0+_S1065, 2139095040U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], atomic<uint> device* atom_38 [[buffer(5)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], packed_float4 device* lp_38 [[buffer(24)]], packed_float4 device* ext_40 [[buffer(23)]], packed_float4 device* lp_sides_38 [[buffer(28)]], uint device* engine_counters_38 [[buffer(26)]], packed_float4 device* engine_contacts_38 [[buffer(25)]], packed_float4 device* cs_anchors_38 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_132;
    (&kernelContext_132)->params_0 = params_38;
    (&kernelContext_132)->work_0 = work_38;
    (&kernelContext_132)->frames_0 = frames_38;
    (&kernelContext_132)->bodies_0 = bodies_38;
    (&kernelContext_132)->chunks_0 = chunks_38;
    (&kernelContext_132)->sort_0 = sort_38;
    (&kernelContext_132)->atom_0 = atom_38;
    (&kernelContext_132)->manifolds_0 = manifolds_38;
    (&kernelContext_132)->loads_0 = loads_38;
    (&kernelContext_132)->bonds_0 = bonds_38;
    (&kernelContext_132)->args_0 = args_38;
    (&kernelContext_132)->info_0 = info_38;
    (&kernelContext_132)->scan_0 = scan_38;
    (&kernelContext_132)->parts_0 = parts_38;
    (&kernelContext_132)->colliders_0 = colliders_38;
    (&kernelContext_132)->collider_info_0 = collider_info_38;
    (&kernelContext_132)->view_chunks_0 = view_chunks_38;
    (&kernelContext_132)->view_0 = view_38;
    (&kernelContext_132)->objects_0 = objects_38;
    (&kernelContext_132)->shot_bodies_0 = shot_bodies_38;
    (&kernelContext_132)->lines_0 = lines_38;
    (&kernelContext_132)->geometry_0 = geometry_38;
    (&kernelContext_132)->shots_0 = shots_38;
    (&kernelContext_132)->lp_0 = lp_38;
    (&kernelContext_132)->ext_0 = ext_40;
    (&kernelContext_132)->lp_sides_0 = lp_sides_38;
    (&kernelContext_132)->engine_counters_0 = engine_counters_38;
    (&kernelContext_132)->engine_contacts_0 = engine_contacts_38;
    (&kernelContext_132)->cs_anchors_0 = cs_anchors_38;
    threadgroup array<uint, int(1024)> g_scan_38;
    (&kernelContext_132)->g_scan_0 = &g_scan_38;
    uint b_40 = id_35.x;
    bool _S1066;
    if(b_40 >= (params_38->body_count_0))
    {
        _S1066 = true;
    }
    else
    {
        _S1066 = ((&kernelContext_132)->params_0->shot_count_0) == 0U;
    }
    if(_S1066)
    {
        return;
    }
    thread uint shot_9;
    bool _S1067 = struck_0(b_40, &shot_9, &kernelContext_132);
    if(!_S1067)
    {
        return;
    }
    uint _S1068 = a_first_1(shot_9, &kernelContext_132);
    atomic<uint> device* _S1069 = (&kernelContext_132)->atom_0+_S1068;
    uint _S1070 = a_strike_0(&kernelContext_132);
    uint _S1071 = atomic_load_explicit((&kernelContext_132)->atom_0+(_S1070 + 9U * b_40 + 6U), memory_order_relaxed);
    uint _S1072 = atomic_fetch_min_explicit(_S1069, _S1071, memory_order_relaxed);
    return;
}

bool cs_struck_1(uint b_41, KernelContext_0 thread* kernelContext_133)
{
    bool _S1073;
    if(b_41 < (kernelContext_133->params_0->body_count_0))
    {
        uint _S1074 = a_cs_0(b_41, kernelContext_133);
        uint _S1075 = atomic_load_explicit(kernelContext_133->atom_0+(_S1074 + 1U), memory_order_relaxed);
        _S1073 = _S1075 != 0U;
    }
    else
    {
        _S1073 = false;
    }
    return _S1073;
}

[[kernel]] void fr_il_begin(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], atomic<uint> device* atom_39 [[buffer(5)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], packed_float4 device* lp_39 [[buffer(24)]], packed_float4 device* ext_41 [[buffer(23)]], packed_float4 device* lp_sides_39 [[buffer(28)]], uint device* engine_counters_39 [[buffer(26)]], packed_float4 device* engine_contacts_39 [[buffer(25)]], packed_float4 device* cs_anchors_39 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_134;
    (&kernelContext_134)->params_0 = params_39;
    (&kernelContext_134)->work_0 = work_39;
    (&kernelContext_134)->frames_0 = frames_39;
    (&kernelContext_134)->bodies_0 = bodies_39;
    (&kernelContext_134)->chunks_0 = chunks_39;
    (&kernelContext_134)->sort_0 = sort_39;
    (&kernelContext_134)->atom_0 = atom_39;
    (&kernelContext_134)->manifolds_0 = manifolds_39;
    (&kernelContext_134)->loads_0 = loads_39;
    (&kernelContext_134)->bonds_0 = bonds_39;
    (&kernelContext_134)->args_0 = args_39;
    (&kernelContext_134)->info_0 = info_39;
    (&kernelContext_134)->scan_0 = scan_39;
    (&kernelContext_134)->parts_0 = parts_39;
    (&kernelContext_134)->colliders_0 = colliders_39;
    (&kernelContext_134)->collider_info_0 = collider_info_39;
    (&kernelContext_134)->view_chunks_0 = view_chunks_39;
    (&kernelContext_134)->view_0 = view_39;
    (&kernelContext_134)->objects_0 = objects_39;
    (&kernelContext_134)->shot_bodies_0 = shot_bodies_39;
    (&kernelContext_134)->lines_0 = lines_39;
    (&kernelContext_134)->geometry_0 = geometry_39;
    (&kernelContext_134)->shots_0 = shots_39;
    (&kernelContext_134)->lp_0 = lp_39;
    (&kernelContext_134)->ext_0 = ext_41;
    (&kernelContext_134)->lp_sides_0 = lp_sides_39;
    (&kernelContext_134)->engine_counters_0 = engine_counters_39;
    (&kernelContext_134)->engine_contacts_0 = engine_contacts_39;
    (&kernelContext_134)->cs_anchors_0 = cs_anchors_39;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_134)->g_scan_0 = &g_scan_39;
    if((params_39->shot_count_0) > 0U)
    {
        il_begin_1(id_36.x, &kernelContext_134);
    }
    return;
}

[[kernel]] void fr_il_extend_even(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], atomic<uint> device* atom_40 [[buffer(5)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], packed_float4 device* lp_40 [[buffer(24)]], packed_float4 device* ext_42 [[buffer(23)]], packed_float4 device* lp_sides_40 [[buffer(28)]], uint device* engine_counters_40 [[buffer(26)]], packed_float4 device* engine_contacts_40 [[buffer(25)]], packed_float4 device* cs_anchors_40 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_135;
    (&kernelContext_135)->params_0 = params_40;
    (&kernelContext_135)->work_0 = work_40;
    (&kernelContext_135)->frames_0 = frames_40;
    (&kernelContext_135)->bodies_0 = bodies_40;
    (&kernelContext_135)->chunks_0 = chunks_40;
    (&kernelContext_135)->sort_0 = sort_40;
    (&kernelContext_135)->atom_0 = atom_40;
    (&kernelContext_135)->manifolds_0 = manifolds_40;
    (&kernelContext_135)->loads_0 = loads_40;
    (&kernelContext_135)->bonds_0 = bonds_40;
    (&kernelContext_135)->args_0 = args_40;
    (&kernelContext_135)->info_0 = info_40;
    (&kernelContext_135)->scan_0 = scan_40;
    (&kernelContext_135)->parts_0 = parts_40;
    (&kernelContext_135)->colliders_0 = colliders_40;
    (&kernelContext_135)->collider_info_0 = collider_info_40;
    (&kernelContext_135)->view_chunks_0 = view_chunks_40;
    (&kernelContext_135)->view_0 = view_40;
    (&kernelContext_135)->objects_0 = objects_40;
    (&kernelContext_135)->shot_bodies_0 = shot_bodies_40;
    (&kernelContext_135)->lines_0 = lines_40;
    (&kernelContext_135)->geometry_0 = geometry_40;
    (&kernelContext_135)->shots_0 = shots_40;
    (&kernelContext_135)->lp_0 = lp_40;
    (&kernelContext_135)->ext_0 = ext_42;
    (&kernelContext_135)->lp_sides_0 = lp_sides_40;
    (&kernelContext_135)->engine_counters_0 = engine_counters_40;
    (&kernelContext_135)->engine_contacts_0 = engine_contacts_40;
    (&kernelContext_135)->cs_anchors_0 = cs_anchors_40;
    threadgroup array<uint, int(1024)> g_scan_40;
    (&kernelContext_135)->g_scan_0 = &g_scan_40;
    if((params_40->shot_count_0) > 0U)
    {
        il_extend_1(id_37.x, 0U, &kernelContext_135);
    }
    return;
}

[[kernel]] void fr_il_extend_odd(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], atomic<uint> device* atom_41 [[buffer(5)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], packed_float4 device* lp_41 [[buffer(24)]], packed_float4 device* ext_43 [[buffer(23)]], packed_float4 device* lp_sides_41 [[buffer(28)]], uint device* engine_counters_41 [[buffer(26)]], packed_float4 device* engine_contacts_41 [[buffer(25)]], packed_float4 device* cs_anchors_41 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_136;
    (&kernelContext_136)->params_0 = params_41;
    (&kernelContext_136)->work_0 = work_41;
    (&kernelContext_136)->frames_0 = frames_41;
    (&kernelContext_136)->bodies_0 = bodies_41;
    (&kernelContext_136)->chunks_0 = chunks_41;
    (&kernelContext_136)->sort_0 = sort_41;
    (&kernelContext_136)->atom_0 = atom_41;
    (&kernelContext_136)->manifolds_0 = manifolds_41;
    (&kernelContext_136)->loads_0 = loads_41;
    (&kernelContext_136)->bonds_0 = bonds_41;
    (&kernelContext_136)->args_0 = args_41;
    (&kernelContext_136)->info_0 = info_41;
    (&kernelContext_136)->scan_0 = scan_41;
    (&kernelContext_136)->parts_0 = parts_41;
    (&kernelContext_136)->colliders_0 = colliders_41;
    (&kernelContext_136)->collider_info_0 = collider_info_41;
    (&kernelContext_136)->view_chunks_0 = view_chunks_41;
    (&kernelContext_136)->view_0 = view_41;
    (&kernelContext_136)->objects_0 = objects_41;
    (&kernelContext_136)->shot_bodies_0 = shot_bodies_41;
    (&kernelContext_136)->lines_0 = lines_41;
    (&kernelContext_136)->geometry_0 = geometry_41;
    (&kernelContext_136)->shots_0 = shots_41;
    (&kernelContext_136)->lp_0 = lp_41;
    (&kernelContext_136)->ext_0 = ext_43;
    (&kernelContext_136)->lp_sides_0 = lp_sides_41;
    (&kernelContext_136)->engine_counters_0 = engine_counters_41;
    (&kernelContext_136)->engine_contacts_0 = engine_contacts_41;
    (&kernelContext_136)->cs_anchors_0 = cs_anchors_41;
    threadgroup array<uint, int(1024)> g_scan_41;
    (&kernelContext_136)->g_scan_0 = &g_scan_41;
    if((params_41->shot_count_0) > 0U)
    {
        il_extend_1(id_38.x, 1U, &kernelContext_136);
    }
    return;
}

[[kernel]] void fr_il_strike(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], atomic<uint> device* atom_42 [[buffer(5)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], packed_float4 device* lp_42 [[buffer(24)]], packed_float4 device* ext_44 [[buffer(23)]], packed_float4 device* lp_sides_42 [[buffer(28)]], uint device* engine_counters_42 [[buffer(26)]], packed_float4 device* engine_contacts_42 [[buffer(25)]], packed_float4 device* cs_anchors_42 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_137;
    (&kernelContext_137)->params_0 = params_42;
    (&kernelContext_137)->work_0 = work_42;
    (&kernelContext_137)->frames_0 = frames_42;
    (&kernelContext_137)->bodies_0 = bodies_42;
    (&kernelContext_137)->chunks_0 = chunks_42;
    (&kernelContext_137)->sort_0 = sort_42;
    (&kernelContext_137)->atom_0 = atom_42;
    (&kernelContext_137)->manifolds_0 = manifolds_42;
    (&kernelContext_137)->loads_0 = loads_42;
    (&kernelContext_137)->bonds_0 = bonds_42;
    (&kernelContext_137)->args_0 = args_42;
    (&kernelContext_137)->info_0 = info_42;
    (&kernelContext_137)->scan_0 = scan_42;
    (&kernelContext_137)->parts_0 = parts_42;
    (&kernelContext_137)->colliders_0 = colliders_42;
    (&kernelContext_137)->collider_info_0 = collider_info_42;
    (&kernelContext_137)->view_chunks_0 = view_chunks_42;
    (&kernelContext_137)->view_0 = view_42;
    (&kernelContext_137)->objects_0 = objects_42;
    (&kernelContext_137)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_137)->lines_0 = lines_42;
    (&kernelContext_137)->geometry_0 = geometry_42;
    (&kernelContext_137)->shots_0 = shots_42;
    (&kernelContext_137)->lp_0 = lp_42;
    (&kernelContext_137)->ext_0 = ext_44;
    (&kernelContext_137)->lp_sides_0 = lp_sides_42;
    (&kernelContext_137)->engine_counters_0 = engine_counters_42;
    (&kernelContext_137)->engine_contacts_0 = engine_contacts_42;
    (&kernelContext_137)->cs_anchors_0 = cs_anchors_42;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_137)->g_scan_0 = &g_scan_42;
    if((params_42->shot_count_0) > 0U)
    {
        il_strike_1(id_39.x, &kernelContext_137);
    }
    return;
}

[[kernel]] void fr_il_mark(uint3 id_40 [[thread_position_in_grid]], FractureParams_0 constant* params_43 [[buffer(0)]], uint device* work_43 [[buffer(4)]], packed_float4 device* frames_43 [[buffer(6)]], packed_float4 device* bodies_43 [[buffer(7)]], packed_float4 device* chunks_43 [[buffer(1)]], uint device* sort_43 [[buffer(11)]], atomic<uint> device* atom_43 [[buffer(5)]], Record_natural_0 device* manifolds_43 [[buffer(10)]], packed_float4 device* loads_43 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_43 [[buffer(13)]], uint device* info_43 [[buffer(15)]], ScanParams_0 constant* scan_43 [[buffer(14)]], packed_float4 device* parts_43 [[buffer(3)]], packed_float4 device* colliders_43 [[buffer(8)]], packed_uint4 device* collider_info_43 [[buffer(9)]], packed_float4 device* view_chunks_43 [[buffer(17)]], ViewParams_0 constant* view_43 [[buffer(16)]], packed_float4 device* objects_43 [[buffer(18)]], uint device* shot_bodies_43 [[buffer(20)]], float device* lines_43 [[buffer(19)]], packed_float4 device* geometry_43 [[buffer(21)]], packed_float4 device* shots_43 [[buffer(22)]], packed_float4 device* lp_43 [[buffer(24)]], packed_float4 device* ext_45 [[buffer(23)]], packed_float4 device* lp_sides_43 [[buffer(28)]], uint device* engine_counters_43 [[buffer(26)]], packed_float4 device* engine_contacts_43 [[buffer(25)]], packed_float4 device* cs_anchors_43 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_138;
    (&kernelContext_138)->params_0 = params_43;
    (&kernelContext_138)->work_0 = work_43;
    (&kernelContext_138)->frames_0 = frames_43;
    (&kernelContext_138)->bodies_0 = bodies_43;
    (&kernelContext_138)->chunks_0 = chunks_43;
    (&kernelContext_138)->sort_0 = sort_43;
    (&kernelContext_138)->atom_0 = atom_43;
    (&kernelContext_138)->manifolds_0 = manifolds_43;
    (&kernelContext_138)->loads_0 = loads_43;
    (&kernelContext_138)->bonds_0 = bonds_43;
    (&kernelContext_138)->args_0 = args_43;
    (&kernelContext_138)->info_0 = info_43;
    (&kernelContext_138)->scan_0 = scan_43;
    (&kernelContext_138)->parts_0 = parts_43;
    (&kernelContext_138)->colliders_0 = colliders_43;
    (&kernelContext_138)->collider_info_0 = collider_info_43;
    (&kernelContext_138)->view_chunks_0 = view_chunks_43;
    (&kernelContext_138)->view_0 = view_43;
    (&kernelContext_138)->objects_0 = objects_43;
    (&kernelContext_138)->shot_bodies_0 = shot_bodies_43;
    (&kernelContext_138)->lines_0 = lines_43;
    (&kernelContext_138)->geometry_0 = geometry_43;
    (&kernelContext_138)->shots_0 = shots_43;
    (&kernelContext_138)->lp_0 = lp_43;
    (&kernelContext_138)->ext_0 = ext_45;
    (&kernelContext_138)->lp_sides_0 = lp_sides_43;
    (&kernelContext_138)->engine_counters_0 = engine_counters_43;
    (&kernelContext_138)->engine_contacts_0 = engine_contacts_43;
    (&kernelContext_138)->cs_anchors_0 = cs_anchors_43;
    threadgroup array<uint, int(1024)> g_scan_43;
    (&kernelContext_138)->g_scan_0 = &g_scan_43;
    if((params_43->shot_count_0) > 0U)
    {
        il_mark_1(id_40.x, &kernelContext_138);
    }
    return;
}

[[kernel]] void fr_il_carry(uint3 id_41 [[thread_position_in_grid]], FractureParams_0 constant* params_44 [[buffer(0)]], uint device* work_44 [[buffer(4)]], packed_float4 device* frames_44 [[buffer(6)]], packed_float4 device* bodies_44 [[buffer(7)]], packed_float4 device* chunks_44 [[buffer(1)]], uint device* sort_44 [[buffer(11)]], atomic<uint> device* atom_44 [[buffer(5)]], Record_natural_0 device* manifolds_44 [[buffer(10)]], packed_float4 device* loads_44 [[buffer(12)]], packed_uint4 device* bonds_44 [[buffer(2)]], uint device* args_44 [[buffer(13)]], uint device* info_44 [[buffer(15)]], ScanParams_0 constant* scan_44 [[buffer(14)]], packed_float4 device* parts_44 [[buffer(3)]], packed_float4 device* colliders_44 [[buffer(8)]], packed_uint4 device* collider_info_44 [[buffer(9)]], packed_float4 device* view_chunks_44 [[buffer(17)]], ViewParams_0 constant* view_44 [[buffer(16)]], packed_float4 device* objects_44 [[buffer(18)]], uint device* shot_bodies_44 [[buffer(20)]], float device* lines_44 [[buffer(19)]], packed_float4 device* geometry_44 [[buffer(21)]], packed_float4 device* shots_44 [[buffer(22)]], packed_float4 device* lp_44 [[buffer(24)]], packed_float4 device* ext_46 [[buffer(23)]], packed_float4 device* lp_sides_44 [[buffer(28)]], uint device* engine_counters_44 [[buffer(26)]], packed_float4 device* engine_contacts_44 [[buffer(25)]], packed_float4 device* cs_anchors_44 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_139;
    (&kernelContext_139)->params_0 = params_44;
    (&kernelContext_139)->work_0 = work_44;
    (&kernelContext_139)->frames_0 = frames_44;
    (&kernelContext_139)->bodies_0 = bodies_44;
    (&kernelContext_139)->chunks_0 = chunks_44;
    (&kernelContext_139)->sort_0 = sort_44;
    (&kernelContext_139)->atom_0 = atom_44;
    (&kernelContext_139)->manifolds_0 = manifolds_44;
    (&kernelContext_139)->loads_0 = loads_44;
    (&kernelContext_139)->bonds_0 = bonds_44;
    (&kernelContext_139)->args_0 = args_44;
    (&kernelContext_139)->info_0 = info_44;
    (&kernelContext_139)->scan_0 = scan_44;
    (&kernelContext_139)->parts_0 = parts_44;
    (&kernelContext_139)->colliders_0 = colliders_44;
    (&kernelContext_139)->collider_info_0 = collider_info_44;
    (&kernelContext_139)->view_chunks_0 = view_chunks_44;
    (&kernelContext_139)->view_0 = view_44;
    (&kernelContext_139)->objects_0 = objects_44;
    (&kernelContext_139)->shot_bodies_0 = shot_bodies_44;
    (&kernelContext_139)->lines_0 = lines_44;
    (&kernelContext_139)->geometry_0 = geometry_44;
    (&kernelContext_139)->shots_0 = shots_44;
    (&kernelContext_139)->lp_0 = lp_44;
    (&kernelContext_139)->ext_0 = ext_46;
    (&kernelContext_139)->lp_sides_0 = lp_sides_44;
    (&kernelContext_139)->engine_counters_0 = engine_counters_44;
    (&kernelContext_139)->engine_contacts_0 = engine_contacts_44;
    (&kernelContext_139)->cs_anchors_0 = cs_anchors_44;
    threadgroup array<uint, int(1024)> g_scan_44;
    (&kernelContext_139)->g_scan_0 = &g_scan_44;
    if((params_44->shot_count_0) > 0U)
    {
        il_carry_1(id_41.x, &kernelContext_139);
    }
    return;
}

[[kernel]] void fr_il_breaks(uint3 id_42 [[thread_position_in_grid]], FractureParams_0 constant* params_45 [[buffer(0)]], uint device* work_45 [[buffer(4)]], packed_float4 device* frames_45 [[buffer(6)]], packed_float4 device* bodies_45 [[buffer(7)]], packed_float4 device* chunks_45 [[buffer(1)]], uint device* sort_45 [[buffer(11)]], atomic<uint> device* atom_45 [[buffer(5)]], Record_natural_0 device* manifolds_45 [[buffer(10)]], packed_float4 device* loads_45 [[buffer(12)]], packed_uint4 device* bonds_45 [[buffer(2)]], uint device* args_45 [[buffer(13)]], uint device* info_45 [[buffer(15)]], ScanParams_0 constant* scan_45 [[buffer(14)]], packed_float4 device* parts_45 [[buffer(3)]], packed_float4 device* colliders_45 [[buffer(8)]], packed_uint4 device* collider_info_45 [[buffer(9)]], packed_float4 device* view_chunks_45 [[buffer(17)]], ViewParams_0 constant* view_45 [[buffer(16)]], packed_float4 device* objects_45 [[buffer(18)]], uint device* shot_bodies_45 [[buffer(20)]], float device* lines_45 [[buffer(19)]], packed_float4 device* geometry_45 [[buffer(21)]], packed_float4 device* shots_45 [[buffer(22)]], packed_float4 device* lp_45 [[buffer(24)]], packed_float4 device* ext_47 [[buffer(23)]], packed_float4 device* lp_sides_45 [[buffer(28)]], uint device* engine_counters_45 [[buffer(26)]], packed_float4 device* engine_contacts_45 [[buffer(25)]], packed_float4 device* cs_anchors_45 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_140;
    (&kernelContext_140)->params_0 = params_45;
    (&kernelContext_140)->work_0 = work_45;
    (&kernelContext_140)->frames_0 = frames_45;
    (&kernelContext_140)->bodies_0 = bodies_45;
    (&kernelContext_140)->chunks_0 = chunks_45;
    (&kernelContext_140)->sort_0 = sort_45;
    (&kernelContext_140)->atom_0 = atom_45;
    (&kernelContext_140)->manifolds_0 = manifolds_45;
    (&kernelContext_140)->loads_0 = loads_45;
    (&kernelContext_140)->bonds_0 = bonds_45;
    (&kernelContext_140)->args_0 = args_45;
    (&kernelContext_140)->info_0 = info_45;
    (&kernelContext_140)->scan_0 = scan_45;
    (&kernelContext_140)->parts_0 = parts_45;
    (&kernelContext_140)->colliders_0 = colliders_45;
    (&kernelContext_140)->collider_info_0 = collider_info_45;
    (&kernelContext_140)->view_chunks_0 = view_chunks_45;
    (&kernelContext_140)->view_0 = view_45;
    (&kernelContext_140)->objects_0 = objects_45;
    (&kernelContext_140)->shot_bodies_0 = shot_bodies_45;
    (&kernelContext_140)->lines_0 = lines_45;
    (&kernelContext_140)->geometry_0 = geometry_45;
    (&kernelContext_140)->shots_0 = shots_45;
    (&kernelContext_140)->lp_0 = lp_45;
    (&kernelContext_140)->ext_0 = ext_47;
    (&kernelContext_140)->lp_sides_0 = lp_sides_45;
    (&kernelContext_140)->engine_counters_0 = engine_counters_45;
    (&kernelContext_140)->engine_contacts_0 = engine_contacts_45;
    (&kernelContext_140)->cs_anchors_0 = cs_anchors_45;
    threadgroup array<uint, int(1024)> g_scan_45;
    (&kernelContext_140)->g_scan_0 = &g_scan_45;
    if((params_45->shot_count_0) > 0U)
    {
        il_breaks_1(id_42.x, &kernelContext_140);
    }
    return;
}

[[kernel]] void fr_csl_begin(uint3 id_43 [[thread_position_in_grid]], FractureParams_0 constant* params_46 [[buffer(0)]], uint device* work_46 [[buffer(4)]], packed_float4 device* frames_46 [[buffer(6)]], packed_float4 device* bodies_46 [[buffer(7)]], packed_float4 device* chunks_46 [[buffer(1)]], uint device* sort_46 [[buffer(11)]], atomic<uint> device* atom_46 [[buffer(5)]], Record_natural_0 device* manifolds_46 [[buffer(10)]], packed_float4 device* loads_46 [[buffer(12)]], packed_uint4 device* bonds_46 [[buffer(2)]], uint device* args_46 [[buffer(13)]], uint device* info_46 [[buffer(15)]], ScanParams_0 constant* scan_46 [[buffer(14)]], packed_float4 device* parts_46 [[buffer(3)]], packed_float4 device* colliders_46 [[buffer(8)]], packed_uint4 device* collider_info_46 [[buffer(9)]], packed_float4 device* view_chunks_46 [[buffer(17)]], ViewParams_0 constant* view_46 [[buffer(16)]], packed_float4 device* objects_46 [[buffer(18)]], uint device* shot_bodies_46 [[buffer(20)]], float device* lines_46 [[buffer(19)]], packed_float4 device* geometry_46 [[buffer(21)]], packed_float4 device* shots_46 [[buffer(22)]], packed_float4 device* lp_46 [[buffer(24)]], packed_float4 device* ext_48 [[buffer(23)]], packed_float4 device* lp_sides_46 [[buffer(28)]], uint device* engine_counters_46 [[buffer(26)]], packed_float4 device* engine_contacts_46 [[buffer(25)]], packed_float4 device* cs_anchors_46 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_141;
    (&kernelContext_141)->params_0 = params_46;
    (&kernelContext_141)->work_0 = work_46;
    (&kernelContext_141)->frames_0 = frames_46;
    (&kernelContext_141)->bodies_0 = bodies_46;
    (&kernelContext_141)->chunks_0 = chunks_46;
    (&kernelContext_141)->sort_0 = sort_46;
    (&kernelContext_141)->atom_0 = atom_46;
    (&kernelContext_141)->manifolds_0 = manifolds_46;
    (&kernelContext_141)->loads_0 = loads_46;
    (&kernelContext_141)->bonds_0 = bonds_46;
    (&kernelContext_141)->args_0 = args_46;
    (&kernelContext_141)->info_0 = info_46;
    (&kernelContext_141)->scan_0 = scan_46;
    (&kernelContext_141)->parts_0 = parts_46;
    (&kernelContext_141)->colliders_0 = colliders_46;
    (&kernelContext_141)->collider_info_0 = collider_info_46;
    (&kernelContext_141)->view_chunks_0 = view_chunks_46;
    (&kernelContext_141)->view_0 = view_46;
    (&kernelContext_141)->objects_0 = objects_46;
    (&kernelContext_141)->shot_bodies_0 = shot_bodies_46;
    (&kernelContext_141)->lines_0 = lines_46;
    (&kernelContext_141)->geometry_0 = geometry_46;
    (&kernelContext_141)->shots_0 = shots_46;
    (&kernelContext_141)->lp_0 = lp_46;
    (&kernelContext_141)->ext_0 = ext_48;
    (&kernelContext_141)->lp_sides_0 = lp_sides_46;
    (&kernelContext_141)->engine_counters_0 = engine_counters_46;
    (&kernelContext_141)->engine_contacts_0 = engine_contacts_46;
    (&kernelContext_141)->cs_anchors_0 = cs_anchors_46;
    threadgroup array<uint, int(1024)> g_scan_46;
    (&kernelContext_141)->g_scan_0 = &g_scan_46;
    il_begin_0(id_43.x, &kernelContext_141);
    return;
}

[[kernel]] void fr_csl_extend_even(uint3 id_44 [[thread_position_in_grid]], FractureParams_0 constant* params_47 [[buffer(0)]], uint device* work_47 [[buffer(4)]], packed_float4 device* frames_47 [[buffer(6)]], packed_float4 device* bodies_47 [[buffer(7)]], packed_float4 device* chunks_47 [[buffer(1)]], uint device* sort_47 [[buffer(11)]], atomic<uint> device* atom_47 [[buffer(5)]], Record_natural_0 device* manifolds_47 [[buffer(10)]], packed_float4 device* loads_47 [[buffer(12)]], packed_uint4 device* bonds_47 [[buffer(2)]], uint device* args_47 [[buffer(13)]], uint device* info_47 [[buffer(15)]], ScanParams_0 constant* scan_47 [[buffer(14)]], packed_float4 device* parts_47 [[buffer(3)]], packed_float4 device* colliders_47 [[buffer(8)]], packed_uint4 device* collider_info_47 [[buffer(9)]], packed_float4 device* view_chunks_47 [[buffer(17)]], ViewParams_0 constant* view_47 [[buffer(16)]], packed_float4 device* objects_47 [[buffer(18)]], uint device* shot_bodies_47 [[buffer(20)]], float device* lines_47 [[buffer(19)]], packed_float4 device* geometry_47 [[buffer(21)]], packed_float4 device* shots_47 [[buffer(22)]], packed_float4 device* lp_47 [[buffer(24)]], packed_float4 device* ext_49 [[buffer(23)]], packed_float4 device* lp_sides_47 [[buffer(28)]], uint device* engine_counters_47 [[buffer(26)]], packed_float4 device* engine_contacts_47 [[buffer(25)]], packed_float4 device* cs_anchors_47 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_142;
    (&kernelContext_142)->params_0 = params_47;
    (&kernelContext_142)->work_0 = work_47;
    (&kernelContext_142)->frames_0 = frames_47;
    (&kernelContext_142)->bodies_0 = bodies_47;
    (&kernelContext_142)->chunks_0 = chunks_47;
    (&kernelContext_142)->sort_0 = sort_47;
    (&kernelContext_142)->atom_0 = atom_47;
    (&kernelContext_142)->manifolds_0 = manifolds_47;
    (&kernelContext_142)->loads_0 = loads_47;
    (&kernelContext_142)->bonds_0 = bonds_47;
    (&kernelContext_142)->args_0 = args_47;
    (&kernelContext_142)->info_0 = info_47;
    (&kernelContext_142)->scan_0 = scan_47;
    (&kernelContext_142)->parts_0 = parts_47;
    (&kernelContext_142)->colliders_0 = colliders_47;
    (&kernelContext_142)->collider_info_0 = collider_info_47;
    (&kernelContext_142)->view_chunks_0 = view_chunks_47;
    (&kernelContext_142)->view_0 = view_47;
    (&kernelContext_142)->objects_0 = objects_47;
    (&kernelContext_142)->shot_bodies_0 = shot_bodies_47;
    (&kernelContext_142)->lines_0 = lines_47;
    (&kernelContext_142)->geometry_0 = geometry_47;
    (&kernelContext_142)->shots_0 = shots_47;
    (&kernelContext_142)->lp_0 = lp_47;
    (&kernelContext_142)->ext_0 = ext_49;
    (&kernelContext_142)->lp_sides_0 = lp_sides_47;
    (&kernelContext_142)->engine_counters_0 = engine_counters_47;
    (&kernelContext_142)->engine_contacts_0 = engine_contacts_47;
    (&kernelContext_142)->cs_anchors_0 = cs_anchors_47;
    threadgroup array<uint, int(1024)> g_scan_47;
    (&kernelContext_142)->g_scan_0 = &g_scan_47;
    il_extend_0(id_44.x, 0U, &kernelContext_142);
    return;
}

[[kernel]] void fr_csl_extend_odd(uint3 id_45 [[thread_position_in_grid]], FractureParams_0 constant* params_48 [[buffer(0)]], uint device* work_48 [[buffer(4)]], packed_float4 device* frames_48 [[buffer(6)]], packed_float4 device* bodies_48 [[buffer(7)]], packed_float4 device* chunks_48 [[buffer(1)]], uint device* sort_48 [[buffer(11)]], atomic<uint> device* atom_48 [[buffer(5)]], Record_natural_0 device* manifolds_48 [[buffer(10)]], packed_float4 device* loads_48 [[buffer(12)]], packed_uint4 device* bonds_48 [[buffer(2)]], uint device* args_48 [[buffer(13)]], uint device* info_48 [[buffer(15)]], ScanParams_0 constant* scan_48 [[buffer(14)]], packed_float4 device* parts_48 [[buffer(3)]], packed_float4 device* colliders_48 [[buffer(8)]], packed_uint4 device* collider_info_48 [[buffer(9)]], packed_float4 device* view_chunks_48 [[buffer(17)]], ViewParams_0 constant* view_48 [[buffer(16)]], packed_float4 device* objects_48 [[buffer(18)]], uint device* shot_bodies_48 [[buffer(20)]], float device* lines_48 [[buffer(19)]], packed_float4 device* geometry_48 [[buffer(21)]], packed_float4 device* shots_48 [[buffer(22)]], packed_float4 device* lp_48 [[buffer(24)]], packed_float4 device* ext_50 [[buffer(23)]], packed_float4 device* lp_sides_48 [[buffer(28)]], uint device* engine_counters_48 [[buffer(26)]], packed_float4 device* engine_contacts_48 [[buffer(25)]], packed_float4 device* cs_anchors_48 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_143;
    (&kernelContext_143)->params_0 = params_48;
    (&kernelContext_143)->work_0 = work_48;
    (&kernelContext_143)->frames_0 = frames_48;
    (&kernelContext_143)->bodies_0 = bodies_48;
    (&kernelContext_143)->chunks_0 = chunks_48;
    (&kernelContext_143)->sort_0 = sort_48;
    (&kernelContext_143)->atom_0 = atom_48;
    (&kernelContext_143)->manifolds_0 = manifolds_48;
    (&kernelContext_143)->loads_0 = loads_48;
    (&kernelContext_143)->bonds_0 = bonds_48;
    (&kernelContext_143)->args_0 = args_48;
    (&kernelContext_143)->info_0 = info_48;
    (&kernelContext_143)->scan_0 = scan_48;
    (&kernelContext_143)->parts_0 = parts_48;
    (&kernelContext_143)->colliders_0 = colliders_48;
    (&kernelContext_143)->collider_info_0 = collider_info_48;
    (&kernelContext_143)->view_chunks_0 = view_chunks_48;
    (&kernelContext_143)->view_0 = view_48;
    (&kernelContext_143)->objects_0 = objects_48;
    (&kernelContext_143)->shot_bodies_0 = shot_bodies_48;
    (&kernelContext_143)->lines_0 = lines_48;
    (&kernelContext_143)->geometry_0 = geometry_48;
    (&kernelContext_143)->shots_0 = shots_48;
    (&kernelContext_143)->lp_0 = lp_48;
    (&kernelContext_143)->ext_0 = ext_50;
    (&kernelContext_143)->lp_sides_0 = lp_sides_48;
    (&kernelContext_143)->engine_counters_0 = engine_counters_48;
    (&kernelContext_143)->engine_contacts_0 = engine_contacts_48;
    (&kernelContext_143)->cs_anchors_0 = cs_anchors_48;
    threadgroup array<uint, int(1024)> g_scan_48;
    (&kernelContext_143)->g_scan_0 = &g_scan_48;
    il_extend_0(id_45.x, 1U, &kernelContext_143);
    return;
}

[[kernel]] void fr_csl_strike(uint3 id_46 [[thread_position_in_grid]], FractureParams_0 constant* params_49 [[buffer(0)]], uint device* work_49 [[buffer(4)]], packed_float4 device* frames_49 [[buffer(6)]], packed_float4 device* bodies_49 [[buffer(7)]], packed_float4 device* chunks_49 [[buffer(1)]], uint device* sort_49 [[buffer(11)]], atomic<uint> device* atom_49 [[buffer(5)]], Record_natural_0 device* manifolds_49 [[buffer(10)]], packed_float4 device* loads_49 [[buffer(12)]], packed_uint4 device* bonds_49 [[buffer(2)]], uint device* args_49 [[buffer(13)]], uint device* info_49 [[buffer(15)]], ScanParams_0 constant* scan_49 [[buffer(14)]], packed_float4 device* parts_49 [[buffer(3)]], packed_float4 device* colliders_49 [[buffer(8)]], packed_uint4 device* collider_info_49 [[buffer(9)]], packed_float4 device* view_chunks_49 [[buffer(17)]], ViewParams_0 constant* view_49 [[buffer(16)]], packed_float4 device* objects_49 [[buffer(18)]], uint device* shot_bodies_49 [[buffer(20)]], float device* lines_49 [[buffer(19)]], packed_float4 device* geometry_49 [[buffer(21)]], packed_float4 device* shots_49 [[buffer(22)]], packed_float4 device* lp_49 [[buffer(24)]], packed_float4 device* ext_51 [[buffer(23)]], packed_float4 device* lp_sides_49 [[buffer(28)]], uint device* engine_counters_49 [[buffer(26)]], packed_float4 device* engine_contacts_49 [[buffer(25)]], packed_float4 device* cs_anchors_49 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_144;
    (&kernelContext_144)->params_0 = params_49;
    (&kernelContext_144)->work_0 = work_49;
    (&kernelContext_144)->frames_0 = frames_49;
    (&kernelContext_144)->bodies_0 = bodies_49;
    (&kernelContext_144)->chunks_0 = chunks_49;
    (&kernelContext_144)->sort_0 = sort_49;
    (&kernelContext_144)->atom_0 = atom_49;
    (&kernelContext_144)->manifolds_0 = manifolds_49;
    (&kernelContext_144)->loads_0 = loads_49;
    (&kernelContext_144)->bonds_0 = bonds_49;
    (&kernelContext_144)->args_0 = args_49;
    (&kernelContext_144)->info_0 = info_49;
    (&kernelContext_144)->scan_0 = scan_49;
    (&kernelContext_144)->parts_0 = parts_49;
    (&kernelContext_144)->colliders_0 = colliders_49;
    (&kernelContext_144)->collider_info_0 = collider_info_49;
    (&kernelContext_144)->view_chunks_0 = view_chunks_49;
    (&kernelContext_144)->view_0 = view_49;
    (&kernelContext_144)->objects_0 = objects_49;
    (&kernelContext_144)->shot_bodies_0 = shot_bodies_49;
    (&kernelContext_144)->lines_0 = lines_49;
    (&kernelContext_144)->geometry_0 = geometry_49;
    (&kernelContext_144)->shots_0 = shots_49;
    (&kernelContext_144)->lp_0 = lp_49;
    (&kernelContext_144)->ext_0 = ext_51;
    (&kernelContext_144)->lp_sides_0 = lp_sides_49;
    (&kernelContext_144)->engine_counters_0 = engine_counters_49;
    (&kernelContext_144)->engine_contacts_0 = engine_contacts_49;
    (&kernelContext_144)->cs_anchors_0 = cs_anchors_49;
    threadgroup array<uint, int(1024)> g_scan_49;
    (&kernelContext_144)->g_scan_0 = &g_scan_49;
    il_strike_0(id_46.x, &kernelContext_144);
    return;
}

[[kernel]] void fr_csl_mark(uint3 id_47 [[thread_position_in_grid]], FractureParams_0 constant* params_50 [[buffer(0)]], uint device* work_50 [[buffer(4)]], packed_float4 device* frames_50 [[buffer(6)]], packed_float4 device* bodies_50 [[buffer(7)]], packed_float4 device* chunks_50 [[buffer(1)]], uint device* sort_50 [[buffer(11)]], atomic<uint> device* atom_50 [[buffer(5)]], Record_natural_0 device* manifolds_50 [[buffer(10)]], packed_float4 device* loads_50 [[buffer(12)]], packed_uint4 device* bonds_50 [[buffer(2)]], uint device* args_50 [[buffer(13)]], uint device* info_50 [[buffer(15)]], ScanParams_0 constant* scan_50 [[buffer(14)]], packed_float4 device* parts_50 [[buffer(3)]], packed_float4 device* colliders_50 [[buffer(8)]], packed_uint4 device* collider_info_50 [[buffer(9)]], packed_float4 device* view_chunks_50 [[buffer(17)]], ViewParams_0 constant* view_50 [[buffer(16)]], packed_float4 device* objects_50 [[buffer(18)]], uint device* shot_bodies_50 [[buffer(20)]], float device* lines_50 [[buffer(19)]], packed_float4 device* geometry_50 [[buffer(21)]], packed_float4 device* shots_50 [[buffer(22)]], packed_float4 device* lp_50 [[buffer(24)]], packed_float4 device* ext_52 [[buffer(23)]], packed_float4 device* lp_sides_50 [[buffer(28)]], uint device* engine_counters_50 [[buffer(26)]], packed_float4 device* engine_contacts_50 [[buffer(25)]], packed_float4 device* cs_anchors_50 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_145;
    (&kernelContext_145)->params_0 = params_50;
    (&kernelContext_145)->work_0 = work_50;
    (&kernelContext_145)->frames_0 = frames_50;
    (&kernelContext_145)->bodies_0 = bodies_50;
    (&kernelContext_145)->chunks_0 = chunks_50;
    (&kernelContext_145)->sort_0 = sort_50;
    (&kernelContext_145)->atom_0 = atom_50;
    (&kernelContext_145)->manifolds_0 = manifolds_50;
    (&kernelContext_145)->loads_0 = loads_50;
    (&kernelContext_145)->bonds_0 = bonds_50;
    (&kernelContext_145)->args_0 = args_50;
    (&kernelContext_145)->info_0 = info_50;
    (&kernelContext_145)->scan_0 = scan_50;
    (&kernelContext_145)->parts_0 = parts_50;
    (&kernelContext_145)->colliders_0 = colliders_50;
    (&kernelContext_145)->collider_info_0 = collider_info_50;
    (&kernelContext_145)->view_chunks_0 = view_chunks_50;
    (&kernelContext_145)->view_0 = view_50;
    (&kernelContext_145)->objects_0 = objects_50;
    (&kernelContext_145)->shot_bodies_0 = shot_bodies_50;
    (&kernelContext_145)->lines_0 = lines_50;
    (&kernelContext_145)->geometry_0 = geometry_50;
    (&kernelContext_145)->shots_0 = shots_50;
    (&kernelContext_145)->lp_0 = lp_50;
    (&kernelContext_145)->ext_0 = ext_52;
    (&kernelContext_145)->lp_sides_0 = lp_sides_50;
    (&kernelContext_145)->engine_counters_0 = engine_counters_50;
    (&kernelContext_145)->engine_contacts_0 = engine_contacts_50;
    (&kernelContext_145)->cs_anchors_0 = cs_anchors_50;
    threadgroup array<uint, int(1024)> g_scan_50;
    (&kernelContext_145)->g_scan_0 = &g_scan_50;
    il_mark_0(id_47.x, &kernelContext_145);
    return;
}

[[kernel]] void fr_csl_carry(uint3 id_48 [[thread_position_in_grid]], FractureParams_0 constant* params_51 [[buffer(0)]], uint device* work_51 [[buffer(4)]], packed_float4 device* frames_51 [[buffer(6)]], packed_float4 device* bodies_51 [[buffer(7)]], packed_float4 device* chunks_51 [[buffer(1)]], uint device* sort_51 [[buffer(11)]], atomic<uint> device* atom_51 [[buffer(5)]], Record_natural_0 device* manifolds_51 [[buffer(10)]], packed_float4 device* loads_51 [[buffer(12)]], packed_uint4 device* bonds_51 [[buffer(2)]], uint device* args_51 [[buffer(13)]], uint device* info_51 [[buffer(15)]], ScanParams_0 constant* scan_51 [[buffer(14)]], packed_float4 device* parts_51 [[buffer(3)]], packed_float4 device* colliders_51 [[buffer(8)]], packed_uint4 device* collider_info_51 [[buffer(9)]], packed_float4 device* view_chunks_51 [[buffer(17)]], ViewParams_0 constant* view_51 [[buffer(16)]], packed_float4 device* objects_51 [[buffer(18)]], uint device* shot_bodies_51 [[buffer(20)]], float device* lines_51 [[buffer(19)]], packed_float4 device* geometry_51 [[buffer(21)]], packed_float4 device* shots_51 [[buffer(22)]], packed_float4 device* lp_51 [[buffer(24)]], packed_float4 device* ext_53 [[buffer(23)]], packed_float4 device* lp_sides_51 [[buffer(28)]], uint device* engine_counters_51 [[buffer(26)]], packed_float4 device* engine_contacts_51 [[buffer(25)]], packed_float4 device* cs_anchors_51 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_146;
    (&kernelContext_146)->params_0 = params_51;
    (&kernelContext_146)->work_0 = work_51;
    (&kernelContext_146)->frames_0 = frames_51;
    (&kernelContext_146)->bodies_0 = bodies_51;
    (&kernelContext_146)->chunks_0 = chunks_51;
    (&kernelContext_146)->sort_0 = sort_51;
    (&kernelContext_146)->atom_0 = atom_51;
    (&kernelContext_146)->manifolds_0 = manifolds_51;
    (&kernelContext_146)->loads_0 = loads_51;
    (&kernelContext_146)->bonds_0 = bonds_51;
    (&kernelContext_146)->args_0 = args_51;
    (&kernelContext_146)->info_0 = info_51;
    (&kernelContext_146)->scan_0 = scan_51;
    (&kernelContext_146)->parts_0 = parts_51;
    (&kernelContext_146)->colliders_0 = colliders_51;
    (&kernelContext_146)->collider_info_0 = collider_info_51;
    (&kernelContext_146)->view_chunks_0 = view_chunks_51;
    (&kernelContext_146)->view_0 = view_51;
    (&kernelContext_146)->objects_0 = objects_51;
    (&kernelContext_146)->shot_bodies_0 = shot_bodies_51;
    (&kernelContext_146)->lines_0 = lines_51;
    (&kernelContext_146)->geometry_0 = geometry_51;
    (&kernelContext_146)->shots_0 = shots_51;
    (&kernelContext_146)->lp_0 = lp_51;
    (&kernelContext_146)->ext_0 = ext_53;
    (&kernelContext_146)->lp_sides_0 = lp_sides_51;
    (&kernelContext_146)->engine_counters_0 = engine_counters_51;
    (&kernelContext_146)->engine_contacts_0 = engine_contacts_51;
    (&kernelContext_146)->cs_anchors_0 = cs_anchors_51;
    threadgroup array<uint, int(1024)> g_scan_51;
    (&kernelContext_146)->g_scan_0 = &g_scan_51;
    il_carry_0(id_48.x, &kernelContext_146);
    return;
}

[[kernel]] void fr_csl_breaks(uint3 id_49 [[thread_position_in_grid]], FractureParams_0 constant* params_52 [[buffer(0)]], uint device* work_52 [[buffer(4)]], packed_float4 device* frames_52 [[buffer(6)]], packed_float4 device* bodies_52 [[buffer(7)]], packed_float4 device* chunks_52 [[buffer(1)]], uint device* sort_52 [[buffer(11)]], atomic<uint> device* atom_52 [[buffer(5)]], Record_natural_0 device* manifolds_52 [[buffer(10)]], packed_float4 device* loads_52 [[buffer(12)]], packed_uint4 device* bonds_52 [[buffer(2)]], uint device* args_52 [[buffer(13)]], uint device* info_52 [[buffer(15)]], ScanParams_0 constant* scan_52 [[buffer(14)]], packed_float4 device* parts_52 [[buffer(3)]], packed_float4 device* colliders_52 [[buffer(8)]], packed_uint4 device* collider_info_52 [[buffer(9)]], packed_float4 device* view_chunks_52 [[buffer(17)]], ViewParams_0 constant* view_52 [[buffer(16)]], packed_float4 device* objects_52 [[buffer(18)]], uint device* shot_bodies_52 [[buffer(20)]], float device* lines_52 [[buffer(19)]], packed_float4 device* geometry_52 [[buffer(21)]], packed_float4 device* shots_52 [[buffer(22)]], packed_float4 device* lp_52 [[buffer(24)]], packed_float4 device* ext_54 [[buffer(23)]], packed_float4 device* lp_sides_52 [[buffer(28)]], uint device* engine_counters_52 [[buffer(26)]], packed_float4 device* engine_contacts_52 [[buffer(25)]], packed_float4 device* cs_anchors_52 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_147;
    (&kernelContext_147)->params_0 = params_52;
    (&kernelContext_147)->work_0 = work_52;
    (&kernelContext_147)->frames_0 = frames_52;
    (&kernelContext_147)->bodies_0 = bodies_52;
    (&kernelContext_147)->chunks_0 = chunks_52;
    (&kernelContext_147)->sort_0 = sort_52;
    (&kernelContext_147)->atom_0 = atom_52;
    (&kernelContext_147)->manifolds_0 = manifolds_52;
    (&kernelContext_147)->loads_0 = loads_52;
    (&kernelContext_147)->bonds_0 = bonds_52;
    (&kernelContext_147)->args_0 = args_52;
    (&kernelContext_147)->info_0 = info_52;
    (&kernelContext_147)->scan_0 = scan_52;
    (&kernelContext_147)->parts_0 = parts_52;
    (&kernelContext_147)->colliders_0 = colliders_52;
    (&kernelContext_147)->collider_info_0 = collider_info_52;
    (&kernelContext_147)->view_chunks_0 = view_chunks_52;
    (&kernelContext_147)->view_0 = view_52;
    (&kernelContext_147)->objects_0 = objects_52;
    (&kernelContext_147)->shot_bodies_0 = shot_bodies_52;
    (&kernelContext_147)->lines_0 = lines_52;
    (&kernelContext_147)->geometry_0 = geometry_52;
    (&kernelContext_147)->shots_0 = shots_52;
    (&kernelContext_147)->lp_0 = lp_52;
    (&kernelContext_147)->ext_0 = ext_54;
    (&kernelContext_147)->lp_sides_0 = lp_sides_52;
    (&kernelContext_147)->engine_counters_0 = engine_counters_52;
    (&kernelContext_147)->engine_contacts_0 = engine_contacts_52;
    (&kernelContext_147)->cs_anchors_0 = cs_anchors_52;
    threadgroup array<uint, int(1024)> g_scan_52;
    (&kernelContext_147)->g_scan_0 = &g_scan_52;
    il_breaks_0(id_49.x, &kernelContext_147);
    return;
}

[[kernel]] void fr_cs_begin(uint3 id_50 [[thread_position_in_grid]], FractureParams_0 constant* params_53 [[buffer(0)]], uint device* work_53 [[buffer(4)]], packed_float4 device* frames_53 [[buffer(6)]], packed_float4 device* bodies_53 [[buffer(7)]], packed_float4 device* chunks_53 [[buffer(1)]], uint device* sort_53 [[buffer(11)]], atomic<uint> device* atom_53 [[buffer(5)]], Record_natural_0 device* manifolds_53 [[buffer(10)]], packed_float4 device* loads_53 [[buffer(12)]], packed_uint4 device* bonds_53 [[buffer(2)]], uint device* args_53 [[buffer(13)]], uint device* info_53 [[buffer(15)]], ScanParams_0 constant* scan_53 [[buffer(14)]], packed_float4 device* parts_53 [[buffer(3)]], packed_float4 device* colliders_53 [[buffer(8)]], packed_uint4 device* collider_info_53 [[buffer(9)]], packed_float4 device* view_chunks_53 [[buffer(17)]], ViewParams_0 constant* view_53 [[buffer(16)]], packed_float4 device* objects_53 [[buffer(18)]], uint device* shot_bodies_53 [[buffer(20)]], float device* lines_53 [[buffer(19)]], packed_float4 device* geometry_53 [[buffer(21)]], packed_float4 device* shots_53 [[buffer(22)]], packed_float4 device* lp_53 [[buffer(24)]], packed_float4 device* ext_55 [[buffer(23)]], packed_float4 device* lp_sides_53 [[buffer(28)]], uint device* engine_counters_53 [[buffer(26)]], packed_float4 device* engine_contacts_53 [[buffer(25)]], packed_float4 device* cs_anchors_53 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_148;
    (&kernelContext_148)->params_0 = params_53;
    (&kernelContext_148)->work_0 = work_53;
    (&kernelContext_148)->frames_0 = frames_53;
    (&kernelContext_148)->bodies_0 = bodies_53;
    (&kernelContext_148)->chunks_0 = chunks_53;
    (&kernelContext_148)->sort_0 = sort_53;
    (&kernelContext_148)->atom_0 = atom_53;
    (&kernelContext_148)->manifolds_0 = manifolds_53;
    (&kernelContext_148)->loads_0 = loads_53;
    (&kernelContext_148)->bonds_0 = bonds_53;
    (&kernelContext_148)->args_0 = args_53;
    (&kernelContext_148)->info_0 = info_53;
    (&kernelContext_148)->scan_0 = scan_53;
    (&kernelContext_148)->parts_0 = parts_53;
    (&kernelContext_148)->colliders_0 = colliders_53;
    (&kernelContext_148)->collider_info_0 = collider_info_53;
    (&kernelContext_148)->view_chunks_0 = view_chunks_53;
    (&kernelContext_148)->view_0 = view_53;
    (&kernelContext_148)->objects_0 = objects_53;
    (&kernelContext_148)->shot_bodies_0 = shot_bodies_53;
    (&kernelContext_148)->lines_0 = lines_53;
    (&kernelContext_148)->geometry_0 = geometry_53;
    (&kernelContext_148)->shots_0 = shots_53;
    (&kernelContext_148)->lp_0 = lp_53;
    (&kernelContext_148)->ext_0 = ext_55;
    (&kernelContext_148)->lp_sides_0 = lp_sides_53;
    (&kernelContext_148)->engine_counters_0 = engine_counters_53;
    (&kernelContext_148)->engine_contacts_0 = engine_contacts_53;
    (&kernelContext_148)->cs_anchors_0 = cs_anchors_53;
    threadgroup array<uint, int(1024)> g_scan_53;
    (&kernelContext_148)->g_scan_0 = &g_scan_53;
    uint _S1076 = a_counters_1(&kernelContext_148);
    atomic_store_explicit((&kernelContext_148)->atom_0+(_S1076 + 14U), 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_148)->atom_0+(_S1076 + 15U), 0U, memory_order_relaxed);
    uint _S1077 = cs_rows_0(&kernelContext_148);
    uint _S1078 = groups_of_0(_S1077, 64U);
    set_args_0(7U, _S1078, &kernelContext_148);
    return;
}

uint w_cs_list_0(KernelContext_0 thread* kernelContext_149)
{
    uint _S1079 = w_carry_0(kernelContext_149);
    return _S1079 + kernelContext_149->params_0->chunk_count_0;
}

uint w_cs_bodies_0(KernelContext_0 thread* kernelContext_150)
{
    uint _S1080 = w_cs_list_0(kernelContext_150);
    return _S1080 + kernelContext_150->params_0->chunk_count_0;
}

[[kernel]] void fr_cs_contacts(uint3 id_51 [[thread_position_in_grid]], FractureParams_0 constant* params_54 [[buffer(0)]], uint device* work_54 [[buffer(4)]], packed_float4 device* frames_54 [[buffer(6)]], packed_float4 device* bodies_54 [[buffer(7)]], packed_float4 device* chunks_54 [[buffer(1)]], uint device* sort_54 [[buffer(11)]], atomic<uint> device* atom_54 [[buffer(5)]], Record_natural_0 device* manifolds_54 [[buffer(10)]], packed_float4 device* loads_54 [[buffer(12)]], packed_uint4 device* bonds_54 [[buffer(2)]], uint device* args_54 [[buffer(13)]], uint device* info_54 [[buffer(15)]], ScanParams_0 constant* scan_54 [[buffer(14)]], packed_float4 device* parts_54 [[buffer(3)]], packed_float4 device* colliders_54 [[buffer(8)]], packed_uint4 device* collider_info_54 [[buffer(9)]], packed_float4 device* view_chunks_54 [[buffer(17)]], ViewParams_0 constant* view_54 [[buffer(16)]], packed_float4 device* objects_54 [[buffer(18)]], uint device* shot_bodies_54 [[buffer(20)]], float device* lines_54 [[buffer(19)]], packed_float4 device* geometry_54 [[buffer(21)]], packed_float4 device* shots_54 [[buffer(22)]], packed_float4 device* lp_54 [[buffer(24)]], packed_float4 device* ext_56 [[buffer(23)]], packed_float4 device* lp_sides_54 [[buffer(28)]], uint device* engine_counters_54 [[buffer(26)]], packed_float4 device* engine_contacts_54 [[buffer(25)]], packed_float4 device* cs_anchors_54 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_151;
    (&kernelContext_151)->params_0 = params_54;
    (&kernelContext_151)->work_0 = work_54;
    (&kernelContext_151)->frames_0 = frames_54;
    (&kernelContext_151)->bodies_0 = bodies_54;
    (&kernelContext_151)->chunks_0 = chunks_54;
    (&kernelContext_151)->sort_0 = sort_54;
    (&kernelContext_151)->atom_0 = atom_54;
    (&kernelContext_151)->manifolds_0 = manifolds_54;
    (&kernelContext_151)->loads_0 = loads_54;
    (&kernelContext_151)->bonds_0 = bonds_54;
    (&kernelContext_151)->args_0 = args_54;
    (&kernelContext_151)->info_0 = info_54;
    (&kernelContext_151)->scan_0 = scan_54;
    (&kernelContext_151)->parts_0 = parts_54;
    (&kernelContext_151)->colliders_0 = colliders_54;
    (&kernelContext_151)->collider_info_0 = collider_info_54;
    (&kernelContext_151)->view_chunks_0 = view_chunks_54;
    (&kernelContext_151)->view_0 = view_54;
    (&kernelContext_151)->objects_0 = objects_54;
    (&kernelContext_151)->shot_bodies_0 = shot_bodies_54;
    (&kernelContext_151)->lines_0 = lines_54;
    (&kernelContext_151)->geometry_0 = geometry_54;
    (&kernelContext_151)->shots_0 = shots_54;
    (&kernelContext_151)->lp_0 = lp_54;
    (&kernelContext_151)->ext_0 = ext_56;
    (&kernelContext_151)->lp_sides_0 = lp_sides_54;
    (&kernelContext_151)->engine_counters_0 = engine_counters_54;
    (&kernelContext_151)->engine_contacts_0 = engine_contacts_54;
    (&kernelContext_151)->cs_anchors_0 = cs_anchors_54;
    threadgroup array<uint, int(1024)> g_scan_54;
    (&kernelContext_151)->g_scan_0 = &g_scan_54;
    uint r_23 = id_51.x;
    uint _S1081 = cs_rows_0(&kernelContext_151);
    if(r_23 >= _S1081)
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
        thread uint k_39;
        thread uint s_17;
        thread uint ob_5;
        thread uint co_5;
        thread float3 p_18;
        thread float3 u_9;
        thread float v_14;
        thread float m_eff_7;
        thread float t_in_8;
        bool _S1082 = cs_side_0(r_23, side_6, &k_39, &s_17, &ob_5, &co_5, &p_18, &u_9, &v_14, &m_eff_7, &t_in_8, &kernelContext_151);
        if(_S1082)
        {
            uint _S1083 = a_cs_0(s_17, &kernelContext_151);
            uint _S1084 = atomic_fetch_max_explicit((&kernelContext_151)->atom_0+_S1083, (as_type<uint>((m_eff_7 * v_14))), memory_order_relaxed);
            uint _S1085 = a_cs_0(s_17, &kernelContext_151);
            uint _S1086 = atomic_exchange_explicit((&kernelContext_151)->atom_0+(_S1085 + 8U), 1U, memory_order_relaxed);
            if(_S1086 == 0U)
            {
                uint _S1087 = w_cs_bodies_0(&kernelContext_151);
                uint _S1088 = a_counters_1(&kernelContext_151);
                uint _S1089 = atomic_fetch_add_explicit((&kernelContext_151)->atom_0+(_S1088 + 14U), 1U, memory_order_relaxed);
                *((&kernelContext_151)->work_0+(_S1087 + _S1089)) = s_17;
            }
        }
        side_6 = side_6 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_gate(uint3 id_52 [[thread_position_in_grid]], FractureParams_0 constant* params_55 [[buffer(0)]], uint device* work_55 [[buffer(4)]], packed_float4 device* frames_55 [[buffer(6)]], packed_float4 device* bodies_55 [[buffer(7)]], packed_float4 device* chunks_55 [[buffer(1)]], uint device* sort_55 [[buffer(11)]], atomic<uint> device* atom_55 [[buffer(5)]], Record_natural_0 device* manifolds_55 [[buffer(10)]], packed_float4 device* loads_55 [[buffer(12)]], packed_uint4 device* bonds_55 [[buffer(2)]], uint device* args_55 [[buffer(13)]], uint device* info_55 [[buffer(15)]], ScanParams_0 constant* scan_55 [[buffer(14)]], packed_float4 device* parts_55 [[buffer(3)]], packed_float4 device* colliders_55 [[buffer(8)]], packed_uint4 device* collider_info_55 [[buffer(9)]], packed_float4 device* view_chunks_55 [[buffer(17)]], ViewParams_0 constant* view_55 [[buffer(16)]], packed_float4 device* objects_55 [[buffer(18)]], uint device* shot_bodies_55 [[buffer(20)]], float device* lines_55 [[buffer(19)]], packed_float4 device* geometry_55 [[buffer(21)]], packed_float4 device* shots_55 [[buffer(22)]], packed_float4 device* lp_55 [[buffer(24)]], packed_float4 device* ext_57 [[buffer(23)]], packed_float4 device* lp_sides_55 [[buffer(28)]], uint device* engine_counters_55 [[buffer(26)]], packed_float4 device* engine_contacts_55 [[buffer(25)]], packed_float4 device* cs_anchors_55 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_152;
    (&kernelContext_152)->params_0 = params_55;
    (&kernelContext_152)->work_0 = work_55;
    (&kernelContext_152)->frames_0 = frames_55;
    (&kernelContext_152)->bodies_0 = bodies_55;
    (&kernelContext_152)->chunks_0 = chunks_55;
    (&kernelContext_152)->sort_0 = sort_55;
    (&kernelContext_152)->atom_0 = atom_55;
    (&kernelContext_152)->manifolds_0 = manifolds_55;
    (&kernelContext_152)->loads_0 = loads_55;
    (&kernelContext_152)->bonds_0 = bonds_55;
    (&kernelContext_152)->args_0 = args_55;
    (&kernelContext_152)->info_0 = info_55;
    (&kernelContext_152)->scan_0 = scan_55;
    (&kernelContext_152)->parts_0 = parts_55;
    (&kernelContext_152)->colliders_0 = colliders_55;
    (&kernelContext_152)->collider_info_0 = collider_info_55;
    (&kernelContext_152)->view_chunks_0 = view_chunks_55;
    (&kernelContext_152)->view_0 = view_55;
    (&kernelContext_152)->objects_0 = objects_55;
    (&kernelContext_152)->shot_bodies_0 = shot_bodies_55;
    (&kernelContext_152)->lines_0 = lines_55;
    (&kernelContext_152)->geometry_0 = geometry_55;
    (&kernelContext_152)->shots_0 = shots_55;
    (&kernelContext_152)->lp_0 = lp_55;
    (&kernelContext_152)->ext_0 = ext_57;
    (&kernelContext_152)->lp_sides_0 = lp_sides_55;
    (&kernelContext_152)->engine_counters_0 = engine_counters_55;
    (&kernelContext_152)->engine_contacts_0 = engine_contacts_55;
    (&kernelContext_152)->cs_anchors_0 = cs_anchors_55;
    threadgroup array<uint, int(1024)> g_scan_55;
    (&kernelContext_152)->g_scan_0 = &g_scan_55;
    uint _S1090 = a_counters_1(&kernelContext_152);
    uint struck_1 = atomic_load_explicit((&kernelContext_152)->atom_0+(_S1090 + 14U), memory_order_relaxed);
    bool any_0 = struck_1 > 0U;
    uint _S1091;
    if(any_0)
    {
        uint _S1092 = cs_rows_0(&kernelContext_152);
        uint _S1093 = groups_of_0(_S1092, 64U);
        _S1091 = _S1093;
    }
    else
    {
        _S1091 = 0U;
    }
    set_args_0(8U, _S1091, &kernelContext_152);
    uint _S1094 = groups_of_0(struck_1, 64U);
    set_args_0(9U, _S1094, &kernelContext_152);
    set_args_0(6U, struck_1, &kernelContext_152);
    if(any_0)
    {
        uint _S1095 = groups_of_0((&kernelContext_152)->params_0->chunk_count_0, 64U);
        _S1091 = _S1095;
    }
    else
    {
        _S1091 = 0U;
    }
    set_args_0(10U, _S1091, &kernelContext_152);
    if(any_0)
    {
        uint _S1096 = groups_of_0((&kernelContext_152)->params_0->bond_count_0, 64U);
        _S1091 = _S1096;
    }
    else
    {
        _S1091 = 0U;
    }
    set_args_0(11U, _S1091, &kernelContext_152);
    if(any_0)
    {
        uint _S1097 = groups_of_0((&kernelContext_152)->params_0->body_count_0, 64U);
        _S1091 = _S1097;
    }
    else
    {
        _S1091 = 0U;
    }
    set_args_0(12U, _S1091, &kernelContext_152);
    return;
}

[[kernel]] void fr_cs_pick_a(uint3 id_53 [[thread_position_in_grid]], FractureParams_0 constant* params_56 [[buffer(0)]], uint device* work_56 [[buffer(4)]], packed_float4 device* frames_56 [[buffer(6)]], packed_float4 device* bodies_56 [[buffer(7)]], packed_float4 device* chunks_56 [[buffer(1)]], uint device* sort_56 [[buffer(11)]], atomic<uint> device* atom_56 [[buffer(5)]], Record_natural_0 device* manifolds_56 [[buffer(10)]], packed_float4 device* loads_56 [[buffer(12)]], packed_uint4 device* bonds_56 [[buffer(2)]], uint device* args_56 [[buffer(13)]], uint device* info_56 [[buffer(15)]], ScanParams_0 constant* scan_56 [[buffer(14)]], packed_float4 device* parts_56 [[buffer(3)]], packed_float4 device* colliders_56 [[buffer(8)]], packed_uint4 device* collider_info_56 [[buffer(9)]], packed_float4 device* view_chunks_56 [[buffer(17)]], ViewParams_0 constant* view_56 [[buffer(16)]], packed_float4 device* objects_56 [[buffer(18)]], uint device* shot_bodies_56 [[buffer(20)]], float device* lines_56 [[buffer(19)]], packed_float4 device* geometry_56 [[buffer(21)]], packed_float4 device* shots_56 [[buffer(22)]], packed_float4 device* lp_56 [[buffer(24)]], packed_float4 device* ext_58 [[buffer(23)]], packed_float4 device* lp_sides_56 [[buffer(28)]], uint device* engine_counters_56 [[buffer(26)]], packed_float4 device* engine_contacts_56 [[buffer(25)]], packed_float4 device* cs_anchors_56 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_153;
    (&kernelContext_153)->params_0 = params_56;
    (&kernelContext_153)->work_0 = work_56;
    (&kernelContext_153)->frames_0 = frames_56;
    (&kernelContext_153)->bodies_0 = bodies_56;
    (&kernelContext_153)->chunks_0 = chunks_56;
    (&kernelContext_153)->sort_0 = sort_56;
    (&kernelContext_153)->atom_0 = atom_56;
    (&kernelContext_153)->manifolds_0 = manifolds_56;
    (&kernelContext_153)->loads_0 = loads_56;
    (&kernelContext_153)->bonds_0 = bonds_56;
    (&kernelContext_153)->args_0 = args_56;
    (&kernelContext_153)->info_0 = info_56;
    (&kernelContext_153)->scan_0 = scan_56;
    (&kernelContext_153)->parts_0 = parts_56;
    (&kernelContext_153)->colliders_0 = colliders_56;
    (&kernelContext_153)->collider_info_0 = collider_info_56;
    (&kernelContext_153)->view_chunks_0 = view_chunks_56;
    (&kernelContext_153)->view_0 = view_56;
    (&kernelContext_153)->objects_0 = objects_56;
    (&kernelContext_153)->shot_bodies_0 = shot_bodies_56;
    (&kernelContext_153)->lines_0 = lines_56;
    (&kernelContext_153)->geometry_0 = geometry_56;
    (&kernelContext_153)->shots_0 = shots_56;
    (&kernelContext_153)->lp_0 = lp_56;
    (&kernelContext_153)->ext_0 = ext_58;
    (&kernelContext_153)->lp_sides_0 = lp_sides_56;
    (&kernelContext_153)->engine_counters_0 = engine_counters_56;
    (&kernelContext_153)->engine_contacts_0 = engine_contacts_56;
    (&kernelContext_153)->cs_anchors_0 = cs_anchors_56;
    threadgroup array<uint, int(1024)> g_scan_56;
    (&kernelContext_153)->g_scan_0 = &g_scan_56;
    cs_pick_3(id_53.x, &kernelContext_153);
    return;
}

[[kernel]] void fr_cs_pick_b(uint3 id_54 [[thread_position_in_grid]], FractureParams_0 constant* params_57 [[buffer(0)]], uint device* work_57 [[buffer(4)]], packed_float4 device* frames_57 [[buffer(6)]], packed_float4 device* bodies_57 [[buffer(7)]], packed_float4 device* chunks_57 [[buffer(1)]], uint device* sort_57 [[buffer(11)]], atomic<uint> device* atom_57 [[buffer(5)]], Record_natural_0 device* manifolds_57 [[buffer(10)]], packed_float4 device* loads_57 [[buffer(12)]], packed_uint4 device* bonds_57 [[buffer(2)]], uint device* args_57 [[buffer(13)]], uint device* info_57 [[buffer(15)]], ScanParams_0 constant* scan_57 [[buffer(14)]], packed_float4 device* parts_57 [[buffer(3)]], packed_float4 device* colliders_57 [[buffer(8)]], packed_uint4 device* collider_info_57 [[buffer(9)]], packed_float4 device* view_chunks_57 [[buffer(17)]], ViewParams_0 constant* view_57 [[buffer(16)]], packed_float4 device* objects_57 [[buffer(18)]], uint device* shot_bodies_57 [[buffer(20)]], float device* lines_57 [[buffer(19)]], packed_float4 device* geometry_57 [[buffer(21)]], packed_float4 device* shots_57 [[buffer(22)]], packed_float4 device* lp_57 [[buffer(24)]], packed_float4 device* ext_59 [[buffer(23)]], packed_float4 device* lp_sides_57 [[buffer(28)]], uint device* engine_counters_57 [[buffer(26)]], packed_float4 device* engine_contacts_57 [[buffer(25)]], packed_float4 device* cs_anchors_57 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_154;
    (&kernelContext_154)->params_0 = params_57;
    (&kernelContext_154)->work_0 = work_57;
    (&kernelContext_154)->frames_0 = frames_57;
    (&kernelContext_154)->bodies_0 = bodies_57;
    (&kernelContext_154)->chunks_0 = chunks_57;
    (&kernelContext_154)->sort_0 = sort_57;
    (&kernelContext_154)->atom_0 = atom_57;
    (&kernelContext_154)->manifolds_0 = manifolds_57;
    (&kernelContext_154)->loads_0 = loads_57;
    (&kernelContext_154)->bonds_0 = bonds_57;
    (&kernelContext_154)->args_0 = args_57;
    (&kernelContext_154)->info_0 = info_57;
    (&kernelContext_154)->scan_0 = scan_57;
    (&kernelContext_154)->parts_0 = parts_57;
    (&kernelContext_154)->colliders_0 = colliders_57;
    (&kernelContext_154)->collider_info_0 = collider_info_57;
    (&kernelContext_154)->view_chunks_0 = view_chunks_57;
    (&kernelContext_154)->view_0 = view_57;
    (&kernelContext_154)->objects_0 = objects_57;
    (&kernelContext_154)->shot_bodies_0 = shot_bodies_57;
    (&kernelContext_154)->lines_0 = lines_57;
    (&kernelContext_154)->geometry_0 = geometry_57;
    (&kernelContext_154)->shots_0 = shots_57;
    (&kernelContext_154)->lp_0 = lp_57;
    (&kernelContext_154)->ext_0 = ext_59;
    (&kernelContext_154)->lp_sides_0 = lp_sides_57;
    (&kernelContext_154)->engine_counters_0 = engine_counters_57;
    (&kernelContext_154)->engine_contacts_0 = engine_contacts_57;
    (&kernelContext_154)->cs_anchors_0 = cs_anchors_57;
    threadgroup array<uint, int(1024)> g_scan_57;
    (&kernelContext_154)->g_scan_0 = &g_scan_57;
    cs_pick_2(id_54.x, &kernelContext_154);
    return;
}

[[kernel]] void fr_cs_pick_c(uint3 id_55 [[thread_position_in_grid]], FractureParams_0 constant* params_58 [[buffer(0)]], uint device* work_58 [[buffer(4)]], packed_float4 device* frames_58 [[buffer(6)]], packed_float4 device* bodies_58 [[buffer(7)]], packed_float4 device* chunks_58 [[buffer(1)]], uint device* sort_58 [[buffer(11)]], atomic<uint> device* atom_58 [[buffer(5)]], Record_natural_0 device* manifolds_58 [[buffer(10)]], packed_float4 device* loads_58 [[buffer(12)]], packed_uint4 device* bonds_58 [[buffer(2)]], uint device* args_58 [[buffer(13)]], uint device* info_58 [[buffer(15)]], ScanParams_0 constant* scan_58 [[buffer(14)]], packed_float4 device* parts_58 [[buffer(3)]], packed_float4 device* colliders_58 [[buffer(8)]], packed_uint4 device* collider_info_58 [[buffer(9)]], packed_float4 device* view_chunks_58 [[buffer(17)]], ViewParams_0 constant* view_58 [[buffer(16)]], packed_float4 device* objects_58 [[buffer(18)]], uint device* shot_bodies_58 [[buffer(20)]], float device* lines_58 [[buffer(19)]], packed_float4 device* geometry_58 [[buffer(21)]], packed_float4 device* shots_58 [[buffer(22)]], packed_float4 device* lp_58 [[buffer(24)]], packed_float4 device* ext_60 [[buffer(23)]], packed_float4 device* lp_sides_58 [[buffer(28)]], uint device* engine_counters_58 [[buffer(26)]], packed_float4 device* engine_contacts_58 [[buffer(25)]], packed_float4 device* cs_anchors_58 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_155;
    (&kernelContext_155)->params_0 = params_58;
    (&kernelContext_155)->work_0 = work_58;
    (&kernelContext_155)->frames_0 = frames_58;
    (&kernelContext_155)->bodies_0 = bodies_58;
    (&kernelContext_155)->chunks_0 = chunks_58;
    (&kernelContext_155)->sort_0 = sort_58;
    (&kernelContext_155)->atom_0 = atom_58;
    (&kernelContext_155)->manifolds_0 = manifolds_58;
    (&kernelContext_155)->loads_0 = loads_58;
    (&kernelContext_155)->bonds_0 = bonds_58;
    (&kernelContext_155)->args_0 = args_58;
    (&kernelContext_155)->info_0 = info_58;
    (&kernelContext_155)->scan_0 = scan_58;
    (&kernelContext_155)->parts_0 = parts_58;
    (&kernelContext_155)->colliders_0 = colliders_58;
    (&kernelContext_155)->collider_info_0 = collider_info_58;
    (&kernelContext_155)->view_chunks_0 = view_chunks_58;
    (&kernelContext_155)->view_0 = view_58;
    (&kernelContext_155)->objects_0 = objects_58;
    (&kernelContext_155)->shot_bodies_0 = shot_bodies_58;
    (&kernelContext_155)->lines_0 = lines_58;
    (&kernelContext_155)->geometry_0 = geometry_58;
    (&kernelContext_155)->shots_0 = shots_58;
    (&kernelContext_155)->lp_0 = lp_58;
    (&kernelContext_155)->ext_0 = ext_60;
    (&kernelContext_155)->lp_sides_0 = lp_sides_58;
    (&kernelContext_155)->engine_counters_0 = engine_counters_58;
    (&kernelContext_155)->engine_contacts_0 = engine_contacts_58;
    (&kernelContext_155)->cs_anchors_0 = cs_anchors_58;
    threadgroup array<uint, int(1024)> g_scan_58;
    (&kernelContext_155)->g_scan_0 = &g_scan_58;
    cs_pick_1(id_55.x, &kernelContext_155);
    return;
}

[[kernel]] void fr_cs_pick_d(uint3 id_56 [[thread_position_in_grid]], FractureParams_0 constant* params_59 [[buffer(0)]], uint device* work_59 [[buffer(4)]], packed_float4 device* frames_59 [[buffer(6)]], packed_float4 device* bodies_59 [[buffer(7)]], packed_float4 device* chunks_59 [[buffer(1)]], uint device* sort_59 [[buffer(11)]], atomic<uint> device* atom_59 [[buffer(5)]], Record_natural_0 device* manifolds_59 [[buffer(10)]], packed_float4 device* loads_59 [[buffer(12)]], packed_uint4 device* bonds_59 [[buffer(2)]], uint device* args_59 [[buffer(13)]], uint device* info_59 [[buffer(15)]], ScanParams_0 constant* scan_59 [[buffer(14)]], packed_float4 device* parts_59 [[buffer(3)]], packed_float4 device* colliders_59 [[buffer(8)]], packed_uint4 device* collider_info_59 [[buffer(9)]], packed_float4 device* view_chunks_59 [[buffer(17)]], ViewParams_0 constant* view_59 [[buffer(16)]], packed_float4 device* objects_59 [[buffer(18)]], uint device* shot_bodies_59 [[buffer(20)]], float device* lines_59 [[buffer(19)]], packed_float4 device* geometry_59 [[buffer(21)]], packed_float4 device* shots_59 [[buffer(22)]], packed_float4 device* lp_59 [[buffer(24)]], packed_float4 device* ext_61 [[buffer(23)]], packed_float4 device* lp_sides_59 [[buffer(28)]], uint device* engine_counters_59 [[buffer(26)]], packed_float4 device* engine_contacts_59 [[buffer(25)]], packed_float4 device* cs_anchors_59 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_156;
    (&kernelContext_156)->params_0 = params_59;
    (&kernelContext_156)->work_0 = work_59;
    (&kernelContext_156)->frames_0 = frames_59;
    (&kernelContext_156)->bodies_0 = bodies_59;
    (&kernelContext_156)->chunks_0 = chunks_59;
    (&kernelContext_156)->sort_0 = sort_59;
    (&kernelContext_156)->atom_0 = atom_59;
    (&kernelContext_156)->manifolds_0 = manifolds_59;
    (&kernelContext_156)->loads_0 = loads_59;
    (&kernelContext_156)->bonds_0 = bonds_59;
    (&kernelContext_156)->args_0 = args_59;
    (&kernelContext_156)->info_0 = info_59;
    (&kernelContext_156)->scan_0 = scan_59;
    (&kernelContext_156)->parts_0 = parts_59;
    (&kernelContext_156)->colliders_0 = colliders_59;
    (&kernelContext_156)->collider_info_0 = collider_info_59;
    (&kernelContext_156)->view_chunks_0 = view_chunks_59;
    (&kernelContext_156)->view_0 = view_59;
    (&kernelContext_156)->objects_0 = objects_59;
    (&kernelContext_156)->shot_bodies_0 = shot_bodies_59;
    (&kernelContext_156)->lines_0 = lines_59;
    (&kernelContext_156)->geometry_0 = geometry_59;
    (&kernelContext_156)->shots_0 = shots_59;
    (&kernelContext_156)->lp_0 = lp_59;
    (&kernelContext_156)->ext_0 = ext_61;
    (&kernelContext_156)->lp_sides_0 = lp_sides_59;
    (&kernelContext_156)->engine_counters_0 = engine_counters_59;
    (&kernelContext_156)->engine_contacts_0 = engine_contacts_59;
    (&kernelContext_156)->cs_anchors_0 = cs_anchors_59;
    threadgroup array<uint, int(1024)> g_scan_59;
    (&kernelContext_156)->g_scan_0 = &g_scan_59;
    cs_pick_0(id_56.x, &kernelContext_156);
    return;
}

[[kernel]] void fr_cs_record(uint3 id_57 [[thread_position_in_grid]], FractureParams_0 constant* params_60 [[buffer(0)]], uint device* work_60 [[buffer(4)]], packed_float4 device* frames_60 [[buffer(6)]], packed_float4 device* bodies_60 [[buffer(7)]], packed_float4 device* chunks_60 [[buffer(1)]], uint device* sort_60 [[buffer(11)]], atomic<uint> device* atom_60 [[buffer(5)]], Record_natural_0 device* manifolds_60 [[buffer(10)]], packed_float4 device* loads_60 [[buffer(12)]], packed_uint4 device* bonds_60 [[buffer(2)]], uint device* args_60 [[buffer(13)]], uint device* info_60 [[buffer(15)]], ScanParams_0 constant* scan_60 [[buffer(14)]], packed_float4 device* parts_60 [[buffer(3)]], packed_float4 device* colliders_60 [[buffer(8)]], packed_uint4 device* collider_info_60 [[buffer(9)]], packed_float4 device* view_chunks_60 [[buffer(17)]], ViewParams_0 constant* view_60 [[buffer(16)]], packed_float4 device* objects_60 [[buffer(18)]], uint device* shot_bodies_60 [[buffer(20)]], float device* lines_60 [[buffer(19)]], packed_float4 device* geometry_60 [[buffer(21)]], packed_float4 device* shots_60 [[buffer(22)]], packed_float4 device* lp_60 [[buffer(24)]], packed_float4 device* ext_62 [[buffer(23)]], packed_float4 device* lp_sides_60 [[buffer(28)]], uint device* engine_counters_60 [[buffer(26)]], packed_float4 device* engine_contacts_60 [[buffer(25)]], packed_float4 device* cs_anchors_60 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_157;
    (&kernelContext_157)->params_0 = params_60;
    (&kernelContext_157)->work_0 = work_60;
    (&kernelContext_157)->frames_0 = frames_60;
    (&kernelContext_157)->bodies_0 = bodies_60;
    (&kernelContext_157)->chunks_0 = chunks_60;
    (&kernelContext_157)->sort_0 = sort_60;
    (&kernelContext_157)->atom_0 = atom_60;
    (&kernelContext_157)->manifolds_0 = manifolds_60;
    (&kernelContext_157)->loads_0 = loads_60;
    (&kernelContext_157)->bonds_0 = bonds_60;
    (&kernelContext_157)->args_0 = args_60;
    (&kernelContext_157)->info_0 = info_60;
    (&kernelContext_157)->scan_0 = scan_60;
    (&kernelContext_157)->parts_0 = parts_60;
    (&kernelContext_157)->colliders_0 = colliders_60;
    (&kernelContext_157)->collider_info_0 = collider_info_60;
    (&kernelContext_157)->view_chunks_0 = view_chunks_60;
    (&kernelContext_157)->view_0 = view_60;
    (&kernelContext_157)->objects_0 = objects_60;
    (&kernelContext_157)->shot_bodies_0 = shot_bodies_60;
    (&kernelContext_157)->lines_0 = lines_60;
    (&kernelContext_157)->geometry_0 = geometry_60;
    (&kernelContext_157)->shots_0 = shots_60;
    (&kernelContext_157)->lp_0 = lp_60;
    (&kernelContext_157)->ext_0 = ext_62;
    (&kernelContext_157)->lp_sides_0 = lp_sides_60;
    (&kernelContext_157)->engine_counters_0 = engine_counters_60;
    (&kernelContext_157)->engine_contacts_0 = engine_contacts_60;
    (&kernelContext_157)->cs_anchors_0 = cs_anchors_60;
    threadgroup array<uint, int(1024)> g_scan_60;
    (&kernelContext_157)->g_scan_0 = &g_scan_60;
    uint _S1098 = id_57.x;
    uint _S1099 = a_counters_1(&kernelContext_157);
    uint _S1100 = atomic_load_explicit((&kernelContext_157)->atom_0+(_S1099 + 14U), memory_order_relaxed);
    if(_S1098 >= _S1100)
    {
        return;
    }
    uint _S1101 = w_cs_bodies_0(&kernelContext_157);
    uint device* _S1102 = (&kernelContext_157)->work_0+(_S1101 + _S1098);
    uint s_18 = *_S1102;
    uint _S1103 = a_cs_0(*_S1102, &kernelContext_157);
    uint _S1104 = atomic_load_explicit((&kernelContext_157)->atom_0+(_S1103 + 11U), memory_order_relaxed);
    uint key_2 = _S1104 - 1U;
    thread uint k_40;
    thread uint s2_0;
    thread uint ob_6;
    thread uint co_6;
    thread float3 p_19;
    thread float3 u_10;
    thread float v_15;
    thread float m_eff_8;
    thread float t_in_9;
    bool _S1105 = cs_side_0(key_2 / 2U, key_2 % 2U, &k_40, &s2_0, &ob_6, &co_6, &p_19, &u_10, &v_15, &m_eff_8, &t_in_9, &kernelContext_157);
    if(!_S1105)
    {
        return;
    }
    uint4 _S1106 = uint4(*((&kernelContext_157)->collider_info_0+(2U * co_6 + 1U))) ;
    float e_o_0 = (as_type<float>((_S1106.y)));
    float _S1107 = max((as_type<float>((_S1106.z))), 0.00100000004749745f);
    float j_7 = m_eff_8 * v_15;
    float tau_2 = hertz_duration_0(m_eff_8, _S1107, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / (float4(*((&kernelContext_157)->geometry_0+(3U * k_40 + 1U))) ).w), v_15);
    uint _S1108 = lp_body_0(s_18, &kernelContext_157);
    *((&kernelContext_157)->lp_0+(_S1108 + 2U)) = packed_float4(float4(from_rubble_0(p_19), 3.14159274101257324f * j_7 / (2.0f * tau_2))) ;
    *((&kernelContext_157)->lp_0+(_S1108 + 3U)) = packed_float4(float4(from_rubble_0(u_10), v_15)) ;
    *((&kernelContext_157)->lp_0+(_S1108 + 4U)) = packed_float4(float4(m_eff_8, _S1107, e_o_0, (as_type<float>((ob_6))))) ;
    *((&kernelContext_157)->lp_0+(_S1108 + 5U)) = packed_float4(float4((as_type<float>((k_40))), j_7 / tau_2, 1.0f / (float4(*((&kernelContext_157)->bodies_0+7U * ob_6)) ).w, t_in_9)) ;
    atomic_store_explicit((&kernelContext_157)->atom_0+(_S1103 + 1U), k_40 + 1U, memory_order_relaxed);
    uint _S1109 = atomic_fetch_add_explicit((&kernelContext_157)->atom_0+(_S1099 + 12U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_hertz(uint3 id_58 [[thread_position_in_grid]], FractureParams_0 constant* params_61 [[buffer(0)]], uint device* work_61 [[buffer(4)]], packed_float4 device* frames_61 [[buffer(6)]], packed_float4 device* bodies_61 [[buffer(7)]], packed_float4 device* chunks_61 [[buffer(1)]], uint device* sort_61 [[buffer(11)]], atomic<uint> device* atom_61 [[buffer(5)]], Record_natural_0 device* manifolds_61 [[buffer(10)]], packed_float4 device* loads_61 [[buffer(12)]], packed_uint4 device* bonds_61 [[buffer(2)]], uint device* args_61 [[buffer(13)]], uint device* info_61 [[buffer(15)]], ScanParams_0 constant* scan_61 [[buffer(14)]], packed_float4 device* parts_61 [[buffer(3)]], packed_float4 device* colliders_61 [[buffer(8)]], packed_uint4 device* collider_info_61 [[buffer(9)]], packed_float4 device* view_chunks_61 [[buffer(17)]], ViewParams_0 constant* view_61 [[buffer(16)]], packed_float4 device* objects_61 [[buffer(18)]], uint device* shot_bodies_61 [[buffer(20)]], float device* lines_61 [[buffer(19)]], packed_float4 device* geometry_61 [[buffer(21)]], packed_float4 device* shots_61 [[buffer(22)]], packed_float4 device* lp_61 [[buffer(24)]], packed_float4 device* ext_63 [[buffer(23)]], packed_float4 device* lp_sides_61 [[buffer(28)]], uint device* engine_counters_61 [[buffer(26)]], packed_float4 device* engine_contacts_61 [[buffer(25)]], packed_float4 device* cs_anchors_61 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_158;
    (&kernelContext_158)->params_0 = params_61;
    (&kernelContext_158)->work_0 = work_61;
    (&kernelContext_158)->frames_0 = frames_61;
    (&kernelContext_158)->bodies_0 = bodies_61;
    (&kernelContext_158)->chunks_0 = chunks_61;
    (&kernelContext_158)->sort_0 = sort_61;
    (&kernelContext_158)->atom_0 = atom_61;
    (&kernelContext_158)->manifolds_0 = manifolds_61;
    (&kernelContext_158)->loads_0 = loads_61;
    (&kernelContext_158)->bonds_0 = bonds_61;
    (&kernelContext_158)->args_0 = args_61;
    (&kernelContext_158)->info_0 = info_61;
    (&kernelContext_158)->scan_0 = scan_61;
    (&kernelContext_158)->parts_0 = parts_61;
    (&kernelContext_158)->colliders_0 = colliders_61;
    (&kernelContext_158)->collider_info_0 = collider_info_61;
    (&kernelContext_158)->view_chunks_0 = view_chunks_61;
    (&kernelContext_158)->view_0 = view_61;
    (&kernelContext_158)->objects_0 = objects_61;
    (&kernelContext_158)->shot_bodies_0 = shot_bodies_61;
    (&kernelContext_158)->lines_0 = lines_61;
    (&kernelContext_158)->geometry_0 = geometry_61;
    (&kernelContext_158)->shots_0 = shots_61;
    (&kernelContext_158)->lp_0 = lp_61;
    (&kernelContext_158)->ext_0 = ext_63;
    (&kernelContext_158)->lp_sides_0 = lp_sides_61;
    (&kernelContext_158)->engine_counters_0 = engine_counters_61;
    (&kernelContext_158)->engine_contacts_0 = engine_contacts_61;
    (&kernelContext_158)->cs_anchors_0 = cs_anchors_61;
    threadgroup array<uint, int(1024)> g_scan_61;
    (&kernelContext_158)->g_scan_0 = &g_scan_61;
    uint i_11 = id_58.x;
    bool _S1110;
    if(i_11 >= (params_61->bond_count_0))
    {
        _S1110 = true;
    }
    else
    {
        uint _S1111 = w_alive_0(&kernelContext_158);
        _S1110 = (*((&kernelContext_158)->work_0+(_S1111 + i_11))) == 0U;
    }
    if(_S1110)
    {
        return;
    }
    uint4 _S1112 = uint4(*((&kernelContext_158)->bonds_0+i_11)) ;
    uint _S1113 = body_of_0(_S1112.x, &kernelContext_158);
    bool _S1114 = cs_struck_1(_S1113, &kernelContext_158);
    if(!_S1114)
    {
        return;
    }
    uint _S1115 = lp_body_0(_S1113, &kernelContext_158);
    bool _S1116 = strike_share_breaks_0(_S1112, (float4(*((&kernelContext_158)->lp_0+(_S1115 + 2U))) ).xyz, (float4(*((&kernelContext_158)->lp_0+(_S1115 + 5U))) ).y, &kernelContext_158);
    if(_S1116)
    {
        uint _S1117 = w_alive_0(&kernelContext_158);
        *((&kernelContext_158)->work_0+(_S1117 + i_11)) = 0U;
        uint _S1118 = a_dirty_0(&kernelContext_158);
        uint _S1119 = atomic_fetch_or_explicit((&kernelContext_158)->atom_0+(_S1118 + _S1113), 1U, memory_order_relaxed);
        uint _S1120 = a_counters_1(&kernelContext_158);
        uint _S1121 = atomic_fetch_add_explicit((&kernelContext_158)->atom_0+_S1120, 1U, memory_order_relaxed);
        uint _S1122 = atomic_fetch_add_explicit((&kernelContext_158)->atom_0+(_S1120 + 1U), 1U, memory_order_relaxed);
        uint _S1123 = atomic_fetch_add_explicit((&kernelContext_158)->atom_0+(_S1120 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_gate2(uint3 id_59 [[thread_position_in_grid]], FractureParams_0 constant* params_62 [[buffer(0)]], uint device* work_62 [[buffer(4)]], packed_float4 device* frames_62 [[buffer(6)]], packed_float4 device* bodies_62 [[buffer(7)]], packed_float4 device* chunks_62 [[buffer(1)]], uint device* sort_62 [[buffer(11)]], atomic<uint> device* atom_62 [[buffer(5)]], Record_natural_0 device* manifolds_62 [[buffer(10)]], packed_float4 device* loads_62 [[buffer(12)]], packed_uint4 device* bonds_62 [[buffer(2)]], uint device* args_62 [[buffer(13)]], uint device* info_62 [[buffer(15)]], ScanParams_0 constant* scan_62 [[buffer(14)]], packed_float4 device* parts_62 [[buffer(3)]], packed_float4 device* colliders_62 [[buffer(8)]], packed_uint4 device* collider_info_62 [[buffer(9)]], packed_float4 device* view_chunks_62 [[buffer(17)]], ViewParams_0 constant* view_62 [[buffer(16)]], packed_float4 device* objects_62 [[buffer(18)]], uint device* shot_bodies_62 [[buffer(20)]], float device* lines_62 [[buffer(19)]], packed_float4 device* geometry_62 [[buffer(21)]], packed_float4 device* shots_62 [[buffer(22)]], packed_float4 device* lp_62 [[buffer(24)]], packed_float4 device* ext_64 [[buffer(23)]], packed_float4 device* lp_sides_62 [[buffer(28)]], uint device* engine_counters_62 [[buffer(26)]], packed_float4 device* engine_contacts_62 [[buffer(25)]], packed_float4 device* cs_anchors_62 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_159;
    (&kernelContext_159)->params_0 = params_62;
    (&kernelContext_159)->work_0 = work_62;
    (&kernelContext_159)->frames_0 = frames_62;
    (&kernelContext_159)->bodies_0 = bodies_62;
    (&kernelContext_159)->chunks_0 = chunks_62;
    (&kernelContext_159)->sort_0 = sort_62;
    (&kernelContext_159)->atom_0 = atom_62;
    (&kernelContext_159)->manifolds_0 = manifolds_62;
    (&kernelContext_159)->loads_0 = loads_62;
    (&kernelContext_159)->bonds_0 = bonds_62;
    (&kernelContext_159)->args_0 = args_62;
    (&kernelContext_159)->info_0 = info_62;
    (&kernelContext_159)->scan_0 = scan_62;
    (&kernelContext_159)->parts_0 = parts_62;
    (&kernelContext_159)->colliders_0 = colliders_62;
    (&kernelContext_159)->collider_info_0 = collider_info_62;
    (&kernelContext_159)->view_chunks_0 = view_chunks_62;
    (&kernelContext_159)->view_0 = view_62;
    (&kernelContext_159)->objects_0 = objects_62;
    (&kernelContext_159)->shot_bodies_0 = shot_bodies_62;
    (&kernelContext_159)->lines_0 = lines_62;
    (&kernelContext_159)->geometry_0 = geometry_62;
    (&kernelContext_159)->shots_0 = shots_62;
    (&kernelContext_159)->lp_0 = lp_62;
    (&kernelContext_159)->ext_0 = ext_64;
    (&kernelContext_159)->lp_sides_0 = lp_sides_62;
    (&kernelContext_159)->engine_counters_0 = engine_counters_62;
    (&kernelContext_159)->engine_contacts_0 = engine_contacts_62;
    (&kernelContext_159)->cs_anchors_0 = cs_anchors_62;
    threadgroup array<uint, int(1024)> g_scan_62;
    (&kernelContext_159)->g_scan_0 = &g_scan_62;
    uint _S1124 = a_counters_1(&kernelContext_159);
    uint _S1125 = atomic_load_explicit((&kernelContext_159)->atom_0+_S1124, memory_order_relaxed);
    uint _S1126;
    if(_S1125 > 0U)
    {
        uint _S1127 = cs_rows_0(&kernelContext_159);
        uint _S1128 = groups_of_0(_S1127, 64U);
        _S1126 = _S1128;
    }
    else
    {
        _S1126 = 0U;
    }
    set_args_0(13U, _S1126, &kernelContext_159);
    return;
}

[[kernel]] void fr_cs_save(uint3 id_60 [[thread_position_in_grid]], FractureParams_0 constant* params_63 [[buffer(0)]], uint device* work_63 [[buffer(4)]], packed_float4 device* frames_63 [[buffer(6)]], packed_float4 device* bodies_63 [[buffer(7)]], packed_float4 device* chunks_63 [[buffer(1)]], uint device* sort_63 [[buffer(11)]], atomic<uint> device* atom_63 [[buffer(5)]], Record_natural_0 device* manifolds_63 [[buffer(10)]], packed_float4 device* loads_63 [[buffer(12)]], packed_uint4 device* bonds_63 [[buffer(2)]], uint device* args_63 [[buffer(13)]], uint device* info_63 [[buffer(15)]], ScanParams_0 constant* scan_63 [[buffer(14)]], packed_float4 device* parts_63 [[buffer(3)]], packed_float4 device* colliders_63 [[buffer(8)]], packed_uint4 device* collider_info_63 [[buffer(9)]], packed_float4 device* view_chunks_63 [[buffer(17)]], ViewParams_0 constant* view_63 [[buffer(16)]], packed_float4 device* objects_63 [[buffer(18)]], uint device* shot_bodies_63 [[buffer(20)]], float device* lines_63 [[buffer(19)]], packed_float4 device* geometry_63 [[buffer(21)]], packed_float4 device* shots_63 [[buffer(22)]], packed_float4 device* lp_63 [[buffer(24)]], packed_float4 device* ext_65 [[buffer(23)]], packed_float4 device* lp_sides_63 [[buffer(28)]], uint device* engine_counters_63 [[buffer(26)]], packed_float4 device* engine_contacts_63 [[buffer(25)]], packed_float4 device* cs_anchors_63 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_160;
    (&kernelContext_160)->params_0 = params_63;
    (&kernelContext_160)->work_0 = work_63;
    (&kernelContext_160)->frames_0 = frames_63;
    (&kernelContext_160)->bodies_0 = bodies_63;
    (&kernelContext_160)->chunks_0 = chunks_63;
    (&kernelContext_160)->sort_0 = sort_63;
    (&kernelContext_160)->atom_0 = atom_63;
    (&kernelContext_160)->manifolds_0 = manifolds_63;
    (&kernelContext_160)->loads_0 = loads_63;
    (&kernelContext_160)->bonds_0 = bonds_63;
    (&kernelContext_160)->args_0 = args_63;
    (&kernelContext_160)->info_0 = info_63;
    (&kernelContext_160)->scan_0 = scan_63;
    (&kernelContext_160)->parts_0 = parts_63;
    (&kernelContext_160)->colliders_0 = colliders_63;
    (&kernelContext_160)->collider_info_0 = collider_info_63;
    (&kernelContext_160)->view_chunks_0 = view_chunks_63;
    (&kernelContext_160)->view_0 = view_63;
    (&kernelContext_160)->objects_0 = objects_63;
    (&kernelContext_160)->shot_bodies_0 = shot_bodies_63;
    (&kernelContext_160)->lines_0 = lines_63;
    (&kernelContext_160)->geometry_0 = geometry_63;
    (&kernelContext_160)->shots_0 = shots_63;
    (&kernelContext_160)->lp_0 = lp_63;
    (&kernelContext_160)->ext_0 = ext_65;
    (&kernelContext_160)->lp_sides_0 = lp_sides_63;
    (&kernelContext_160)->engine_counters_0 = engine_counters_63;
    (&kernelContext_160)->engine_contacts_0 = engine_contacts_63;
    (&kernelContext_160)->cs_anchors_0 = cs_anchors_63;
    threadgroup array<uint, int(1024)> g_scan_63;
    (&kernelContext_160)->g_scan_0 = &g_scan_63;
    uint r_24 = id_60.x;
    uint _S1129 = cs_rows_0(&kernelContext_160);
    if(r_24 >= _S1129)
    {
        return;
    }
    uint _S1130 = cs_row_0(r_24, &kernelContext_160);
    uint4 _S1131 = (as_type<uint4>((float4(*((&kernelContext_160)->engine_contacts_0+_S1130)) )));
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
        float4 _S1132 = float4(0.0f) ;
        bool _S1133 = side_7 == 0U;
        uint col_1;
        if(_S1133)
        {
            col_1 = _S1131.x;
        }
        else
        {
            col_1 = _S1131.y;
        }
        uint k_41;
        if((_S1131.w) != 0U)
        {
            k_41 = (uint4(*((&kernelContext_160)->collider_info_0+2U * col_1)) ).x;
        }
        else
        {
            k_41 = 4294967295U;
        }
        bool _S1134;
        if(k_41 != 4294967295U)
        {
            _S1134 = k_41 < ((&kernelContext_160)->params_0->chunk_count_0);
        }
        else
        {
            _S1134 = false;
        }
        bool _S1135;
        if(_S1134)
        {
            uint _S1136 = body_of_0(k_41, &kernelContext_160);
            bool _S1137 = dirty_0(_S1136, &kernelContext_160);
            _S1135 = _S1137;
        }
        else
        {
            _S1135 = false;
        }
        float4 saved_0;
        if(_S1135)
        {
            uint _S1138 = body_of_0(k_41, &kernelContext_160);
            uint at_16 = 7U * _S1138;
            float3 anchor_0;
            if(_S1133)
            {
                anchor_0 = (float4(*((&kernelContext_160)->engine_contacts_0+(_S1130 + 2U))) ).xyz;
            }
            else
            {
                anchor_0 = (float4(*((&kernelContext_160)->engine_contacts_0+(_S1130 + 3U))) ).xyz;
            }
            float3 _S1139 = (float4(*((&kernelContext_160)->bodies_0+at_16)) ).xyz;
            thread Quat_0 _S1140 = quat_of_0(float4(*((&kernelContext_160)->bodies_0+(at_16 + 1U))) );
            float3 _S1141 = rotate_0(&_S1140, anchor_0);
            saved_0 = float4(_S1139 + _S1141, 1.0f);
        }
        else
        {
            saved_0 = _S1132;
        }
        *((&kernelContext_160)->cs_anchors_0+(2U * r_24 + side_7)) = packed_float4(saved_0) ;
        side_7 = side_7 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_clean(uint3 id_61 [[thread_position_in_grid]], FractureParams_0 constant* params_64 [[buffer(0)]], uint device* work_64 [[buffer(4)]], packed_float4 device* frames_64 [[buffer(6)]], packed_float4 device* bodies_64 [[buffer(7)]], packed_float4 device* chunks_64 [[buffer(1)]], uint device* sort_64 [[buffer(11)]], atomic<uint> device* atom_64 [[buffer(5)]], Record_natural_0 device* manifolds_64 [[buffer(10)]], packed_float4 device* loads_64 [[buffer(12)]], packed_uint4 device* bonds_64 [[buffer(2)]], uint device* args_64 [[buffer(13)]], uint device* info_64 [[buffer(15)]], ScanParams_0 constant* scan_64 [[buffer(14)]], packed_float4 device* parts_64 [[buffer(3)]], packed_float4 device* colliders_64 [[buffer(8)]], packed_uint4 device* collider_info_64 [[buffer(9)]], packed_float4 device* view_chunks_64 [[buffer(17)]], ViewParams_0 constant* view_64 [[buffer(16)]], packed_float4 device* objects_64 [[buffer(18)]], uint device* shot_bodies_64 [[buffer(20)]], float device* lines_64 [[buffer(19)]], packed_float4 device* geometry_64 [[buffer(21)]], packed_float4 device* shots_64 [[buffer(22)]], packed_float4 device* lp_64 [[buffer(24)]], packed_float4 device* ext_66 [[buffer(23)]], packed_float4 device* lp_sides_64 [[buffer(28)]], uint device* engine_counters_64 [[buffer(26)]], packed_float4 device* engine_contacts_64 [[buffer(25)]], packed_float4 device* cs_anchors_64 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_161;
    (&kernelContext_161)->params_0 = params_64;
    (&kernelContext_161)->work_0 = work_64;
    (&kernelContext_161)->frames_0 = frames_64;
    (&kernelContext_161)->bodies_0 = bodies_64;
    (&kernelContext_161)->chunks_0 = chunks_64;
    (&kernelContext_161)->sort_0 = sort_64;
    (&kernelContext_161)->atom_0 = atom_64;
    (&kernelContext_161)->manifolds_0 = manifolds_64;
    (&kernelContext_161)->loads_0 = loads_64;
    (&kernelContext_161)->bonds_0 = bonds_64;
    (&kernelContext_161)->args_0 = args_64;
    (&kernelContext_161)->info_0 = info_64;
    (&kernelContext_161)->scan_0 = scan_64;
    (&kernelContext_161)->parts_0 = parts_64;
    (&kernelContext_161)->colliders_0 = colliders_64;
    (&kernelContext_161)->collider_info_0 = collider_info_64;
    (&kernelContext_161)->view_chunks_0 = view_chunks_64;
    (&kernelContext_161)->view_0 = view_64;
    (&kernelContext_161)->objects_0 = objects_64;
    (&kernelContext_161)->shot_bodies_0 = shot_bodies_64;
    (&kernelContext_161)->lines_0 = lines_64;
    (&kernelContext_161)->geometry_0 = geometry_64;
    (&kernelContext_161)->shots_0 = shots_64;
    (&kernelContext_161)->lp_0 = lp_64;
    (&kernelContext_161)->ext_0 = ext_66;
    (&kernelContext_161)->lp_sides_0 = lp_sides_64;
    (&kernelContext_161)->engine_counters_0 = engine_counters_64;
    (&kernelContext_161)->engine_contacts_0 = engine_contacts_64;
    (&kernelContext_161)->cs_anchors_0 = cs_anchors_64;
    threadgroup array<uint, int(1024)> g_scan_64;
    (&kernelContext_161)->g_scan_0 = &g_scan_64;
    uint _S1142 = id_61.x;
    uint _S1143 = a_counters_1(&kernelContext_161);
    uint _S1144 = atomic_load_explicit((&kernelContext_161)->atom_0+(_S1143 + 14U), memory_order_relaxed);
    if(_S1142 >= _S1144)
    {
        return;
    }
    uint _S1145 = w_cs_bodies_0(&kernelContext_161);
    uint _S1146 = a_cs_0(*((&kernelContext_161)->work_0+(_S1145 + _S1142)), &kernelContext_161);
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
        atomic_store_explicit((&kernelContext_161)->atom_0+(_S1146 + w_8), 0U, memory_order_relaxed);
        w_8 = w_8 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_fix(uint3 id_62 [[thread_position_in_grid]], FractureParams_0 constant* params_65 [[buffer(0)]], uint device* work_65 [[buffer(4)]], packed_float4 device* frames_65 [[buffer(6)]], packed_float4 device* bodies_65 [[buffer(7)]], packed_float4 device* chunks_65 [[buffer(1)]], uint device* sort_65 [[buffer(11)]], atomic<uint> device* atom_65 [[buffer(5)]], Record_natural_0 device* manifolds_65 [[buffer(10)]], packed_float4 device* loads_65 [[buffer(12)]], packed_uint4 device* bonds_65 [[buffer(2)]], uint device* args_65 [[buffer(13)]], uint device* info_65 [[buffer(15)]], ScanParams_0 constant* scan_65 [[buffer(14)]], packed_float4 device* parts_65 [[buffer(3)]], packed_float4 device* colliders_65 [[buffer(8)]], packed_uint4 device* collider_info_65 [[buffer(9)]], packed_float4 device* view_chunks_65 [[buffer(17)]], ViewParams_0 constant* view_65 [[buffer(16)]], packed_float4 device* objects_65 [[buffer(18)]], uint device* shot_bodies_65 [[buffer(20)]], float device* lines_65 [[buffer(19)]], packed_float4 device* geometry_65 [[buffer(21)]], packed_float4 device* shots_65 [[buffer(22)]], packed_float4 device* lp_65 [[buffer(24)]], packed_float4 device* ext_67 [[buffer(23)]], packed_float4 device* lp_sides_65 [[buffer(28)]], uint device* engine_counters_65 [[buffer(26)]], packed_float4 device* engine_contacts_65 [[buffer(25)]], packed_float4 device* cs_anchors_65 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_162;
    (&kernelContext_162)->params_0 = params_65;
    (&kernelContext_162)->work_0 = work_65;
    (&kernelContext_162)->frames_0 = frames_65;
    (&kernelContext_162)->bodies_0 = bodies_65;
    (&kernelContext_162)->chunks_0 = chunks_65;
    (&kernelContext_162)->sort_0 = sort_65;
    (&kernelContext_162)->atom_0 = atom_65;
    (&kernelContext_162)->manifolds_0 = manifolds_65;
    (&kernelContext_162)->loads_0 = loads_65;
    (&kernelContext_162)->bonds_0 = bonds_65;
    (&kernelContext_162)->args_0 = args_65;
    (&kernelContext_162)->info_0 = info_65;
    (&kernelContext_162)->scan_0 = scan_65;
    (&kernelContext_162)->parts_0 = parts_65;
    (&kernelContext_162)->colliders_0 = colliders_65;
    (&kernelContext_162)->collider_info_0 = collider_info_65;
    (&kernelContext_162)->view_chunks_0 = view_chunks_65;
    (&kernelContext_162)->view_0 = view_65;
    (&kernelContext_162)->objects_0 = objects_65;
    (&kernelContext_162)->shot_bodies_0 = shot_bodies_65;
    (&kernelContext_162)->lines_0 = lines_65;
    (&kernelContext_162)->geometry_0 = geometry_65;
    (&kernelContext_162)->shots_0 = shots_65;
    (&kernelContext_162)->lp_0 = lp_65;
    (&kernelContext_162)->ext_0 = ext_67;
    (&kernelContext_162)->lp_sides_0 = lp_sides_65;
    (&kernelContext_162)->engine_counters_0 = engine_counters_65;
    (&kernelContext_162)->engine_contacts_0 = engine_contacts_65;
    (&kernelContext_162)->cs_anchors_0 = cs_anchors_65;
    threadgroup array<uint, int(1024)> g_scan_65;
    (&kernelContext_162)->g_scan_0 = &g_scan_65;
    uint r_25 = id_62.x;
    uint _S1147 = cs_rows_0(&kernelContext_162);
    if(r_25 >= _S1147)
    {
        return;
    }
    uint _S1148 = cs_row_0(r_25, &kernelContext_162);
    uint4 _S1149 = (as_type<uint4>((float4(*((&kernelContext_162)->engine_contacts_0+_S1148)) )));
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
        float4 _S1150 = float4(*((&kernelContext_162)->cs_anchors_0+(2U * r_25 + side_8))) ;
        if((_S1150.w) == 0.0f)
        {
            side_8 = side_8 + 1U;
            continue;
        }
        uint col_2;
        if(side_8 == 0U)
        {
            col_2 = _S1149.x;
        }
        else
        {
            col_2 = _S1149.y;
        }
        uint at_17 = 7U * (as_type<uint>(((float4(*((&kernelContext_162)->colliders_0+4U * col_2)) ).x)));
        thread Quat_0 _S1151 = quat_of_0(float4(*((&kernelContext_162)->bodies_0+(at_17 + 1U))) );
        Quat_0 _S1152 = conj_0(&_S1151);
        float3 _S1153 = _S1150.xyz - (float4(*((&kernelContext_162)->bodies_0+at_17)) ).xyz;
        thread Quat_0 _S1154 = _S1152;
        float3 _S1155 = rotate_0(&_S1154, _S1153);
        (*((&kernelContext_162)->engine_contacts_0+(_S1148 + 2U + side_8))).xyz = _S1155;
        moved_0 = true;
        side_8 = side_8 + 1U;
    }
    if(moved_0)
    {
        (*((&kernelContext_162)->engine_contacts_0+(_S1148 + 4U)))[int(3)] = 0.0f;
        (*((&kernelContext_162)->engine_contacts_0+(_S1148 + 5U)))[int(3)] = 0.0f;
        uint _S1156 = _S1148 + 6U;
        *((&kernelContext_162)->engine_contacts_0+_S1156) = packed_float4(float4(1.0f, 1.0f, 0.0f, (float4(*((&kernelContext_162)->engine_contacts_0+_S1156)) ).w)) ;
        (*((&kernelContext_162)->engine_contacts_0+(_S1148 + 7U)))[int(1)] = 0.0f;
    }
    return;
}

[[kernel]] void fr_cs_count(uint3 id_63 [[thread_position_in_grid]], FractureParams_0 constant* params_66 [[buffer(0)]], uint device* work_66 [[buffer(4)]], packed_float4 device* frames_66 [[buffer(6)]], packed_float4 device* bodies_66 [[buffer(7)]], packed_float4 device* chunks_66 [[buffer(1)]], uint device* sort_66 [[buffer(11)]], atomic<uint> device* atom_66 [[buffer(5)]], Record_natural_0 device* manifolds_66 [[buffer(10)]], packed_float4 device* loads_66 [[buffer(12)]], packed_uint4 device* bonds_66 [[buffer(2)]], uint device* args_66 [[buffer(13)]], uint device* info_66 [[buffer(15)]], ScanParams_0 constant* scan_66 [[buffer(14)]], packed_float4 device* parts_66 [[buffer(3)]], packed_float4 device* colliders_66 [[buffer(8)]], packed_uint4 device* collider_info_66 [[buffer(9)]], packed_float4 device* view_chunks_66 [[buffer(17)]], ViewParams_0 constant* view_66 [[buffer(16)]], packed_float4 device* objects_66 [[buffer(18)]], uint device* shot_bodies_66 [[buffer(20)]], float device* lines_66 [[buffer(19)]], packed_float4 device* geometry_66 [[buffer(21)]], packed_float4 device* shots_66 [[buffer(22)]], packed_float4 device* lp_66 [[buffer(24)]], packed_float4 device* ext_68 [[buffer(23)]], packed_float4 device* lp_sides_66 [[buffer(28)]], uint device* engine_counters_66 [[buffer(26)]], packed_float4 device* engine_contacts_66 [[buffer(25)]], packed_float4 device* cs_anchors_66 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_163;
    (&kernelContext_163)->params_0 = params_66;
    (&kernelContext_163)->work_0 = work_66;
    (&kernelContext_163)->frames_0 = frames_66;
    (&kernelContext_163)->bodies_0 = bodies_66;
    (&kernelContext_163)->chunks_0 = chunks_66;
    (&kernelContext_163)->sort_0 = sort_66;
    (&kernelContext_163)->atom_0 = atom_66;
    (&kernelContext_163)->manifolds_0 = manifolds_66;
    (&kernelContext_163)->loads_0 = loads_66;
    (&kernelContext_163)->bonds_0 = bonds_66;
    (&kernelContext_163)->args_0 = args_66;
    (&kernelContext_163)->info_0 = info_66;
    (&kernelContext_163)->scan_0 = scan_66;
    (&kernelContext_163)->parts_0 = parts_66;
    (&kernelContext_163)->colliders_0 = colliders_66;
    (&kernelContext_163)->collider_info_0 = collider_info_66;
    (&kernelContext_163)->view_chunks_0 = view_chunks_66;
    (&kernelContext_163)->view_0 = view_66;
    (&kernelContext_163)->objects_0 = objects_66;
    (&kernelContext_163)->shot_bodies_0 = shot_bodies_66;
    (&kernelContext_163)->lines_0 = lines_66;
    (&kernelContext_163)->geometry_0 = geometry_66;
    (&kernelContext_163)->shots_0 = shots_66;
    (&kernelContext_163)->lp_0 = lp_66;
    (&kernelContext_163)->ext_0 = ext_68;
    (&kernelContext_163)->lp_sides_0 = lp_sides_66;
    (&kernelContext_163)->engine_counters_0 = engine_counters_66;
    (&kernelContext_163)->engine_contacts_0 = engine_contacts_66;
    (&kernelContext_163)->cs_anchors_0 = cs_anchors_66;
    threadgroup array<uint, int(1024)> g_scan_66;
    (&kernelContext_163)->g_scan_0 = &g_scan_66;
    uint k_42 = id_63.x;
    if(k_42 >= (params_66->chunk_count_0))
    {
        return;
    }
    uint _S1157 = body_of_0(k_42, &kernelContext_163);
    bool _S1158 = cs_struck_1(_S1157, &kernelContext_163);
    if(!_S1158)
    {
        return;
    }
    uint _S1159 = a_cs_0(_S1157, &kernelContext_163);
    uint _S1160 = atomic_fetch_max_explicit((&kernelContext_163)->atom_0+(_S1159 + 3U), ~k_42, memory_order_relaxed);
    if(((&kernelContext_163)->info_0[k_42] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_163)->atom_0+(_S1159 + 4U), 1U, memory_order_relaxed);
    }
    uint _S1161 = atomic_fetch_add_explicit((&kernelContext_163)->atom_0+(_S1159 + 5U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_base(uint3 id_64 [[thread_position_in_grid]], FractureParams_0 constant* params_67 [[buffer(0)]], uint device* work_67 [[buffer(4)]], packed_float4 device* frames_67 [[buffer(6)]], packed_float4 device* bodies_67 [[buffer(7)]], packed_float4 device* chunks_67 [[buffer(1)]], uint device* sort_67 [[buffer(11)]], atomic<uint> device* atom_67 [[buffer(5)]], Record_natural_0 device* manifolds_67 [[buffer(10)]], packed_float4 device* loads_67 [[buffer(12)]], packed_uint4 device* bonds_67 [[buffer(2)]], uint device* args_67 [[buffer(13)]], uint device* info_67 [[buffer(15)]], ScanParams_0 constant* scan_67 [[buffer(14)]], packed_float4 device* parts_67 [[buffer(3)]], packed_float4 device* colliders_67 [[buffer(8)]], packed_uint4 device* collider_info_67 [[buffer(9)]], packed_float4 device* view_chunks_67 [[buffer(17)]], ViewParams_0 constant* view_67 [[buffer(16)]], packed_float4 device* objects_67 [[buffer(18)]], uint device* shot_bodies_67 [[buffer(20)]], float device* lines_67 [[buffer(19)]], packed_float4 device* geometry_67 [[buffer(21)]], packed_float4 device* shots_67 [[buffer(22)]], packed_float4 device* lp_67 [[buffer(24)]], packed_float4 device* ext_69 [[buffer(23)]], packed_float4 device* lp_sides_67 [[buffer(28)]], uint device* engine_counters_67 [[buffer(26)]], packed_float4 device* engine_contacts_67 [[buffer(25)]], packed_float4 device* cs_anchors_67 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_164;
    (&kernelContext_164)->params_0 = params_67;
    (&kernelContext_164)->work_0 = work_67;
    (&kernelContext_164)->frames_0 = frames_67;
    (&kernelContext_164)->bodies_0 = bodies_67;
    (&kernelContext_164)->chunks_0 = chunks_67;
    (&kernelContext_164)->sort_0 = sort_67;
    (&kernelContext_164)->atom_0 = atom_67;
    (&kernelContext_164)->manifolds_0 = manifolds_67;
    (&kernelContext_164)->loads_0 = loads_67;
    (&kernelContext_164)->bonds_0 = bonds_67;
    (&kernelContext_164)->args_0 = args_67;
    (&kernelContext_164)->info_0 = info_67;
    (&kernelContext_164)->scan_0 = scan_67;
    (&kernelContext_164)->parts_0 = parts_67;
    (&kernelContext_164)->colliders_0 = colliders_67;
    (&kernelContext_164)->collider_info_0 = collider_info_67;
    (&kernelContext_164)->view_chunks_0 = view_chunks_67;
    (&kernelContext_164)->view_0 = view_67;
    (&kernelContext_164)->objects_0 = objects_67;
    (&kernelContext_164)->shot_bodies_0 = shot_bodies_67;
    (&kernelContext_164)->lines_0 = lines_67;
    (&kernelContext_164)->geometry_0 = geometry_67;
    (&kernelContext_164)->shots_0 = shots_67;
    (&kernelContext_164)->lp_0 = lp_67;
    (&kernelContext_164)->ext_0 = ext_69;
    (&kernelContext_164)->lp_sides_0 = lp_sides_67;
    (&kernelContext_164)->engine_counters_0 = engine_counters_67;
    (&kernelContext_164)->engine_contacts_0 = engine_contacts_67;
    (&kernelContext_164)->cs_anchors_0 = cs_anchors_67;
    threadgroup array<uint, int(1024)> g_scan_67;
    (&kernelContext_164)->g_scan_0 = &g_scan_67;
    uint _S1162 = id_64.x;
    uint _S1163 = a_counters_1(&kernelContext_164);
    uint _S1164 = atomic_load_explicit((&kernelContext_164)->atom_0+(_S1163 + 14U), memory_order_relaxed);
    if(_S1162 >= _S1164)
    {
        return;
    }
    uint _S1165 = w_cs_bodies_0(&kernelContext_164);
    uint _S1166 = a_cs_0(*((&kernelContext_164)->work_0+(_S1165 + _S1162)), &kernelContext_164);
    atomic<uint> device* _S1167 = (&kernelContext_164)->atom_0+(_S1166 + 6U);
    atomic<uint> device* _S1168 = (&kernelContext_164)->atom_0+(_S1163 + 15U);
    uint _S1169 = atomic_load_explicit((&kernelContext_164)->atom_0+(_S1166 + 5U), memory_order_relaxed);
    uint _S1170 = atomic_fetch_add_explicit(_S1168, _S1169, memory_order_relaxed);
    atomic_store_explicit(_S1167, _S1170, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_collect(uint3 id_65 [[thread_position_in_grid]], FractureParams_0 constant* params_68 [[buffer(0)]], uint device* work_68 [[buffer(4)]], packed_float4 device* frames_68 [[buffer(6)]], packed_float4 device* bodies_68 [[buffer(7)]], packed_float4 device* chunks_68 [[buffer(1)]], uint device* sort_68 [[buffer(11)]], atomic<uint> device* atom_68 [[buffer(5)]], Record_natural_0 device* manifolds_68 [[buffer(10)]], packed_float4 device* loads_68 [[buffer(12)]], packed_uint4 device* bonds_68 [[buffer(2)]], uint device* args_68 [[buffer(13)]], uint device* info_68 [[buffer(15)]], ScanParams_0 constant* scan_68 [[buffer(14)]], packed_float4 device* parts_68 [[buffer(3)]], packed_float4 device* colliders_68 [[buffer(8)]], packed_uint4 device* collider_info_68 [[buffer(9)]], packed_float4 device* view_chunks_68 [[buffer(17)]], ViewParams_0 constant* view_68 [[buffer(16)]], packed_float4 device* objects_68 [[buffer(18)]], uint device* shot_bodies_68 [[buffer(20)]], float device* lines_68 [[buffer(19)]], packed_float4 device* geometry_68 [[buffer(21)]], packed_float4 device* shots_68 [[buffer(22)]], packed_float4 device* lp_68 [[buffer(24)]], packed_float4 device* ext_70 [[buffer(23)]], packed_float4 device* lp_sides_68 [[buffer(28)]], uint device* engine_counters_68 [[buffer(26)]], packed_float4 device* engine_contacts_68 [[buffer(25)]], packed_float4 device* cs_anchors_68 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_165;
    (&kernelContext_165)->params_0 = params_68;
    (&kernelContext_165)->work_0 = work_68;
    (&kernelContext_165)->frames_0 = frames_68;
    (&kernelContext_165)->bodies_0 = bodies_68;
    (&kernelContext_165)->chunks_0 = chunks_68;
    (&kernelContext_165)->sort_0 = sort_68;
    (&kernelContext_165)->atom_0 = atom_68;
    (&kernelContext_165)->manifolds_0 = manifolds_68;
    (&kernelContext_165)->loads_0 = loads_68;
    (&kernelContext_165)->bonds_0 = bonds_68;
    (&kernelContext_165)->args_0 = args_68;
    (&kernelContext_165)->info_0 = info_68;
    (&kernelContext_165)->scan_0 = scan_68;
    (&kernelContext_165)->parts_0 = parts_68;
    (&kernelContext_165)->colliders_0 = colliders_68;
    (&kernelContext_165)->collider_info_0 = collider_info_68;
    (&kernelContext_165)->view_chunks_0 = view_chunks_68;
    (&kernelContext_165)->view_0 = view_68;
    (&kernelContext_165)->objects_0 = objects_68;
    (&kernelContext_165)->shot_bodies_0 = shot_bodies_68;
    (&kernelContext_165)->lines_0 = lines_68;
    (&kernelContext_165)->geometry_0 = geometry_68;
    (&kernelContext_165)->shots_0 = shots_68;
    (&kernelContext_165)->lp_0 = lp_68;
    (&kernelContext_165)->ext_0 = ext_70;
    (&kernelContext_165)->lp_sides_0 = lp_sides_68;
    (&kernelContext_165)->engine_counters_0 = engine_counters_68;
    (&kernelContext_165)->engine_contacts_0 = engine_contacts_68;
    (&kernelContext_165)->cs_anchors_0 = cs_anchors_68;
    threadgroup array<uint, int(1024)> g_scan_68;
    (&kernelContext_165)->g_scan_0 = &g_scan_68;
    uint k_43 = id_65.x;
    if(k_43 >= (params_68->chunk_count_0))
    {
        return;
    }
    uint _S1171 = body_of_0(k_43, &kernelContext_165);
    bool _S1172 = cs_struck_1(_S1171, &kernelContext_165);
    if(!_S1172)
    {
        return;
    }
    uint _S1173 = w_cs_list_0(&kernelContext_165);
    uint _S1174 = a_cs_0(_S1171, &kernelContext_165);
    uint _S1175 = atomic_load_explicit((&kernelContext_165)->atom_0+(_S1174 + 6U), memory_order_relaxed);
    uint _S1176 = _S1173 + _S1175;
    uint _S1177 = atomic_fetch_add_explicit((&kernelContext_165)->atom_0+(_S1174 + 7U), 1U, memory_order_relaxed);
    *((&kernelContext_165)->work_0+(_S1176 + _S1177)) = k_43;
    return;
}

void cs_load_0(uint k_44, uint s_19, KernelContext_0 thread* kernelContext_166)
{
    uint _S1178 = a_cs_0(s_19, kernelContext_166);
    uint _S1179 = atomic_load_explicit(kernelContext_166->atom_0+(_S1178 + 4U), memory_order_relaxed);
    bool supported_1 = _S1179 != 0U;
    bool fixed_2;
    if((kernelContext_166->info_0[k_44] & 1U) != 0U)
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            uint _S1180 = atomic_load_explicit(kernelContext_166->atom_0+(_S1178 + 3U), memory_order_relaxed);
            fixed_2 = (~_S1180) == k_44;
        }
        else
        {
            fixed_2 = false;
        }
    }
    uint _S1181 = lp_body_0(s_19, kernelContext_166);
    float4 _S1182 = float4(*(kernelContext_166->lp_0+(_S1181 + 2U))) ;
    float4 _S1183 = float4(*(kernelContext_166->lp_0+(_S1181 + 3U))) ;
    uint _S1184 = 7U * s_19;
    thread Quat_0 q_27;
    thread float3 origin_8;
    structure_pose_0(s_19, _S1184, &q_27, &origin_8, kernelContext_166);
    thread Quat_0 _S1185 = q_27;
    Quat_0 _S1186 = conj_0(&_S1185);
    float3 _S1187 = _S1183.xyz * float3(_S1182.w) ;
    thread Quat_0 _S1188 = _S1186;
    float3 _S1189 = rotate_0(&_S1188, _S1187);
    float3 _S1190 = _S1182.xyz - origin_8;
    thread Quat_0 _S1191 = _S1186;
    float3 _S1192 = rotate_0(&_S1191, _S1190);
    uint _S1193 = 4U * k_44;
    float4 _S1194 = float4(*(kernelContext_166->chunks_0+_S1193)) ;
    float3 _S1195 = float3(0.0f) ;
    uint _S1196 = atomic_load_explicit(kernelContext_166->atom_0+(_S1178 + 1U), memory_order_relaxed);
    float3 f_12;
    float3 t_7;
    if(_S1196 == (k_44 + 1U))
    {
        float3 _S1197 = cross(_S1192 - _S1194.xyz, _S1189);
        f_12 = _S1189;
        t_7 = _S1197;
    }
    else
    {
        f_12 = _S1195;
        t_7 = _S1195;
    }
    float inv_mass_7 = (float4(*(kernelContext_166->bodies_0+_S1184)) ).w;
    bool _S1198;
    if(!supported_1)
    {
        _S1198 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S1198 = false;
    }
    if(_S1198)
    {
        uint _S1199 = 2U * s_19;
        float3 com_2 = (float4(*(kernelContext_166->frames_0+_S1199)) ).xyz;
        Quat_0 p_20 = quat_of_0(float4(*(kernelContext_166->frames_0+(_S1199 + 1U))) );
        float3 _S1200 = (float4(*(kernelContext_166->bodies_0+(_S1184 + 2U))) ).xyz;
        thread Quat_0 _S1201 = p_20;
        Quat_0 _S1202 = conj_0(&_S1201);
        float3 _S1203 = cross(_S1192 - com_2, _S1189);
        thread Quat_0 _S1204 = _S1202;
        float3 _S1205 = rotate_0(&_S1204, _S1203);
        float3 _S1206 = _S1200 * _S1205;
        thread Quat_0 _S1207 = p_20;
        float3 _S1208 = rotate_0(&_S1207, _S1206);
        float4 _S1209 = float4(*(kernelContext_166->chunks_0+(_S1193 + 1U))) ;
        float4 _S1210 = float4(*(kernelContext_166->chunks_0+(_S1193 + 2U))) ;
        float _S1211 = _S1208.x;
        float _S1212 = _S1209.w;
        float _S1213 = _S1208.y;
        float _S1214 = _S1210.x;
        float _S1215 = _S1208.z;
        float _S1216 = _S1210.y;
        float3 t_8 = t_7 - float3(_S1209.x * _S1211 + _S1212 * _S1213 + _S1214 * _S1215, _S1212 * _S1211 + _S1209.y * _S1213 + _S1216 * _S1215, _S1214 * _S1211 + _S1216 * _S1213 + _S1209.z * _S1215);
        f_12 = f_12 - float3(_S1194.w)  * (_S1189 * float3(inv_mass_7)  + cross(_S1208, _S1194.xyz - com_2));
        t_7 = t_8;
    }
    uint _S1217 = st_f_0(k_44, 14U);
    packed_float4 device* _S1218 = kernelContext_166->lp_0+_S1217;
    float _S1219;
    if(fixed_2)
    {
        _S1219 = 1.0f;
    }
    else
    {
        _S1219 = 0.0f;
    }
    *_S1218 = packed_float4(float4(f_12, _S1219)) ;
    *(kernelContext_166->lp_0+(_S1217 + 1U)) = packed_float4(float4(t_7, 0.0f)) ;
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
        uint _S1220 = st_x_0(k_44, 14U, c_23);
        packed_float4 _S1221 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_166->lp_0+_S1220) = _S1221;
        *(kernelContext_166->lp_0+(_S1220 + 1U)) = _S1221;
        c_23 = c_23 + 1U;
    }
    uint _S1222 = st_d_0(k_44, 14U);
    packed_float4 _S1223 = packed_float4(float4(0.0f) ) ;
    *(kernelContext_166->lp_0+_S1222) = _S1223;
    *(kernelContext_166->lp_0+(_S1222 + 1U)) = _S1223;
    return;
}

void cs_break_0(uint i_12, uint s_20, KernelContext_0 thread* kernelContext_167)
{
    uint4 _S1224 = uint4(*(kernelContext_167->bonds_0+i_12)) ;
    LpBond_0 _S1225 = lp_bond_0(i_12, kernelContext_167);
    uint _S1226 = _S1224.x;
    uint _S1227 = _S1224.y;
    thread LpBond_0 _S1228 = _S1225;
    thread float3 q_lin_5;
    thread float3 q_ang_5;
    st_force_0(&_S1228, _S1226, _S1227, 14U, 0U, &q_lin_5, &q_ang_5, kernelContext_167);
    if((kernelContext_167->params_0->load_model_0) == 1U)
    {
        thread LpBond_0 _S1229 = _S1225;
        thread float3 b_lin_0;
        thread float3 b_ang_0;
        lp_force_0(&_S1229, _S1226, _S1227, 0U, &b_lin_0, &b_ang_0, kernelContext_167);
        q_lin_5 = q_lin_5 + b_lin_0;
        q_ang_5 = q_ang_5 + b_ang_0;
    }
    thread JointBond_0 _S1230 = _S1225.law_0;
    Measures_0 _S1231 = stress_measures_0(&_S1230, q_lin_5, q_ang_5);
    JointMaterial_0 _S1232 = lp_material_0(_S1225.law_0.ids_3.x, kernelContext_167);
    float _S1233 = _S1225.law_0.geom1_0.w;
    thread JointMaterial_0 _S1234 = _S1232;
    thread JointBond_0 _S1235 = _S1225.law_0;
    thread Measures_0 _S1236 = _S1231;
    float4 _S1237 = failure_indices_0(&_S1234, &_S1235, &_S1236, _S1233);
    if((max(max(_S1237.x, _S1237.y), max(_S1237.z, _S1237.w))) >= 1.0f)
    {
        uint _S1238 = w_alive_0(kernelContext_167);
        *(kernelContext_167->work_0+(_S1238 + i_12)) = 0U;
        uint _S1239 = a_dirty_0(kernelContext_167);
        uint _S1240 = atomic_fetch_or_explicit(kernelContext_167->atom_0+(_S1239 + s_20), 1U, memory_order_relaxed);
        uint _S1241 = a_counters_1(kernelContext_167);
        uint _S1242 = atomic_fetch_add_explicit(kernelContext_167->atom_0+_S1241, 1U, memory_order_relaxed);
        uint _S1243 = atomic_fetch_add_explicit(kernelContext_167->atom_0+(_S1241 + 1U), 1U, memory_order_relaxed);
        uint _S1244 = atomic_fetch_add_explicit(kernelContext_167->atom_0+(_S1241 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_solve(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_69 [[buffer(0)]], uint device* work_69 [[buffer(4)]], packed_float4 device* frames_69 [[buffer(6)]], packed_float4 device* bodies_69 [[buffer(7)]], packed_float4 device* chunks_69 [[buffer(1)]], uint device* sort_69 [[buffer(11)]], atomic<uint> device* atom_69 [[buffer(5)]], Record_natural_0 device* manifolds_69 [[buffer(10)]], packed_float4 device* loads_69 [[buffer(12)]], packed_uint4 device* bonds_69 [[buffer(2)]], uint device* args_69 [[buffer(13)]], uint device* info_69 [[buffer(15)]], ScanParams_0 constant* scan_69 [[buffer(14)]], packed_float4 device* parts_69 [[buffer(3)]], packed_float4 device* colliders_69 [[buffer(8)]], packed_uint4 device* collider_info_69 [[buffer(9)]], packed_float4 device* view_chunks_69 [[buffer(17)]], ViewParams_0 constant* view_69 [[buffer(16)]], packed_float4 device* objects_69 [[buffer(18)]], uint device* shot_bodies_69 [[buffer(20)]], float device* lines_69 [[buffer(19)]], packed_float4 device* geometry_69 [[buffer(21)]], packed_float4 device* shots_69 [[buffer(22)]], packed_float4 device* lp_69 [[buffer(24)]], packed_float4 device* ext_71 [[buffer(23)]], packed_float4 device* lp_sides_69 [[buffer(28)]], uint device* engine_counters_69 [[buffer(26)]], packed_float4 device* engine_contacts_69 [[buffer(25)]], packed_float4 device* cs_anchors_69 [[buffer(27)]])
{
    bool _S1245;
    uint j3_0;
    thread KernelContext_0 kernelContext_168;
    (&kernelContext_168)->params_0 = params_69;
    (&kernelContext_168)->work_0 = work_69;
    (&kernelContext_168)->frames_0 = frames_69;
    (&kernelContext_168)->bodies_0 = bodies_69;
    (&kernelContext_168)->chunks_0 = chunks_69;
    (&kernelContext_168)->sort_0 = sort_69;
    (&kernelContext_168)->atom_0 = atom_69;
    (&kernelContext_168)->manifolds_0 = manifolds_69;
    (&kernelContext_168)->loads_0 = loads_69;
    (&kernelContext_168)->bonds_0 = bonds_69;
    (&kernelContext_168)->args_0 = args_69;
    (&kernelContext_168)->info_0 = info_69;
    (&kernelContext_168)->scan_0 = scan_69;
    (&kernelContext_168)->parts_0 = parts_69;
    (&kernelContext_168)->colliders_0 = colliders_69;
    (&kernelContext_168)->collider_info_0 = collider_info_69;
    (&kernelContext_168)->view_chunks_0 = view_chunks_69;
    (&kernelContext_168)->view_0 = view_69;
    (&kernelContext_168)->objects_0 = objects_69;
    (&kernelContext_168)->shot_bodies_0 = shot_bodies_69;
    (&kernelContext_168)->lines_0 = lines_69;
    (&kernelContext_168)->geometry_0 = geometry_69;
    (&kernelContext_168)->shots_0 = shots_69;
    (&kernelContext_168)->lp_0 = lp_69;
    (&kernelContext_168)->ext_0 = ext_71;
    (&kernelContext_168)->lp_sides_0 = lp_sides_69;
    (&kernelContext_168)->engine_counters_0 = engine_counters_69;
    (&kernelContext_168)->engine_contacts_0 = engine_contacts_69;
    (&kernelContext_168)->cs_anchors_0 = cs_anchors_69;
    threadgroup array<uint, int(1024)> g_scan_69;
    (&kernelContext_168)->g_scan_0 = &g_scan_69;
    uint _S1246 = w_cs_bodies_0(&kernelContext_168);
    uint device* _S1247 = (&kernelContext_168)->work_0+(_S1246 + group_1.x);
    uint s_21 = *_S1247;
    uint t_9 = thread_2.x;
    uint _S1248 = a_cs_0(*_S1247, &kernelContext_168);
    uint _S1249 = atomic_load_explicit((&kernelContext_168)->atom_0+(_S1248 + 6U), memory_order_relaxed);
    uint _S1250 = atomic_load_explicit((&kernelContext_168)->atom_0+(_S1248 + 5U), memory_order_relaxed);
    uint j_8 = t_9;
    for(;;)
    {
        if(j_8 < _S1250)
        {
        }
        else
        {
            break;
        }
        uint _S1251 = w_cs_list_0(&kernelContext_168);
        cs_load_0(*((&kernelContext_168)->work_0+(_S1251 + _S1249 + j_8)), s_21, &kernelContext_168);
        j_8 = j_8 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint j2_0 = t_9;
    for(;;)
    {
        if(j2_0 < _S1250)
        {
        }
        else
        {
            break;
        }
        uint _S1252 = w_cs_list_0(&kernelContext_168);
        uint device* _S1253 = (&kernelContext_168)->work_0+(_S1252 + _S1249 + j2_0);
        uint k_45 = *_S1253;
        uint _S1254 = st_f_0(*_S1253, 14U);
        if(((float4(*((&kernelContext_168)->lp_0+_S1254)) ).w) == 0.0f)
        {
            bool _S1255 = lp_factor_0(k_45, &kernelContext_168);
            _S1245 = !_S1255;
        }
        else
        {
            _S1245 = false;
        }
        if(_S1245)
        {
            (*((&kernelContext_168)->lp_0+_S1254))[int(3)] = 1.0f;
            uint _S1256 = a_counters_1(&kernelContext_168);
            uint _S1257 = atomic_fetch_or_explicit((&kernelContext_168)->atom_0+(_S1256 + 3U), 32U, memory_order_relaxed);
        }
        j2_0 = j2_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint sweep_0 = 0U;
    for(;;)
    {
        if(sweep_0 < ((&kernelContext_168)->params_0->cs_sweeps_0))
        {
        }
        else
        {
            break;
        }
        j3_0 = t_9;
        for(;;)
        {
            if(j3_0 < _S1250)
            {
            }
            else
            {
                break;
            }
            uint _S1258 = w_cs_list_0(&kernelContext_168);
            st_sweep_0(*((&kernelContext_168)->work_0+(_S1258 + _S1249 + j3_0)), 14U, sweep_0 % 2U, (&kernelContext_168)->params_0->cs_c1_0, (&kernelContext_168)->params_0->cs_c2_0, &kernelContext_168);
            j3_0 = j3_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        sweep_0 = sweep_0 + 1U;
    }
    j3_0 = t_9;
    for(;;)
    {
        if(j3_0 < _S1250)
        {
        }
        else
        {
            break;
        }
        uint _S1259 = w_cs_list_0(&kernelContext_168);
        uint device* _S1260 = (&kernelContext_168)->work_0+(_S1259 + _S1249 + j3_0);
        uint k_46 = *_S1260;
        float4 _S1261 = float4(*((&kernelContext_168)->ext_0+((&kernelContext_168)->params_0->ext_chunks_0 + 4U * *_S1260))) ;
        uint _S1262 = (as_type<uint>((_S1261.x)));
        uint _S1263 = (as_type<uint>((_S1261.y)));
        uint e_5 = 0U;
        for(;;)
        {
            if(e_5 < _S1263)
            {
            }
            else
            {
                break;
            }
            uint _S1264 = adj_bond_0(_S1262 + e_5, &kernelContext_168);
            if(((uint4(*((&kernelContext_168)->bonds_0+_S1264)) ).x) == k_46)
            {
                uint _S1265 = w_alive_0(&kernelContext_168);
                _S1245 = (*((&kernelContext_168)->work_0+(_S1265 + _S1264))) != 0U;
            }
            else
            {
                _S1245 = false;
            }
            if(_S1245)
            {
                cs_break_0(_S1264, s_21, &kernelContext_168);
            }
            e_5 = e_5 + 1U;
        }
        j3_0 = j3_0 + 256U;
    }
    return;
}

