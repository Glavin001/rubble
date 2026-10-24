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
    return _S14 + 64U + 8U * b_4;
}

uint il_entry_0(uint b_5, KernelContext_0 thread* kernelContext_12)
{
    uint _S15 = a_cs_0(b_5, kernelContext_12);
    uint _S16 = atomic_load_explicit(kernelContext_12->atom_0+(_S15 + 1U), memory_order_relaxed);
    return _S16;
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
        _S19 = _S21 != 4294967295U;
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

uint w_alive_0(KernelContext_0 thread* kernelContext_22)
{
    return kernelContext_22->params_0->chunk_count_0;
}

uint w_body_0()
{
    return 0U;
}

uint body_of_0(uint c_0, KernelContext_0 thread* kernelContext_23)
{
    return *(kernelContext_23->work_0+(w_body_0() + c_0));
}

struct IlPlace_0
{
    float along_0;
    float radial_0;
    bool entry_0;
};

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
    thread Quat_0 _S35 = axes_0;
    Quat_0 _S36 = quat_mul_0(&_S35, q_3);
    thread Quat_0 _S37 = _S36;
    Quat_0 _S38 = normalized_0(&_S37);
    return _S38;
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

void structure_pose_0(uint slot_0, uint b_15, Quat_0 thread* q_5, float3 thread* origin_0, KernelContext_0 thread* kernelContext_24)
{
    uint _S39 = 2U * slot_0;
    thread Quat_0 _S40 = quat_of_0(float4(*(kernelContext_24->frames_0+(_S39 + 1U))) );
    Quat_0 _S41 = conj_0(&_S40);
    float3 com_0 = (float4(*(kernelContext_24->frames_0+_S39)) ).xyz;
    thread Quat_0 _S42 = quat_of_0(float4(*(kernelContext_24->bodies_0+(b_15 + 1U))) );
    Quat_0 _S43 = rot_from_rubble_0(&_S42);
    thread Quat_0 _S44 = _S43;
    thread Quat_0 _S45 = _S41;
    Quat_0 _S46 = quat_mul_0(&_S44, &_S45);
    thread Quat_0 _S47 = _S46;
    Quat_0 _S48 = normalized_0(&_S47);
    *q_5 = _S48;
    float3 _S49 = from_rubble_0((float4(*(kernelContext_24->bodies_0+b_15)) ).xyz);
    thread Quat_0 _S50 = _S48;
    float3 _S51 = rotate_0(&_S50, com_0);
    *origin_0 = _S49 - _S51;
    return;
}

IlPlace_0 il_place_0(uint k_1, uint entry_1, float3 contact_2, float3 d_2, KernelContext_0 thread* kernelContext_25)
{
    uint _S52 = body_of_0(k_1, kernelContext_25);
    thread Quat_0 q_6;
    thread float3 origin_1;
    structure_pose_0(_S52, 7U * _S52, &q_6, &origin_1, kernelContext_25);
    float3 _S53 = (float4(*(kernelContext_25->chunks_0+4U * k_1)) ).xyz;
    thread Quat_0 _S54 = q_6;
    float3 _S55 = rotate_0(&_S54, _S53);
    float3 rel_1 = origin_1 + _S55 - contact_2;
    thread IlPlace_0 p_0;
    float _S56 = dot(rel_1, d_2);
    (&p_0)->along_0 = _S56;
    (&p_0)->radial_0 = length(rel_1 - d_2 * float3(_S56) );
    (&p_0)->entry_0 = k_1 == entry_1;
    return p_0;
}

bool il_plug_0(uint outcome_0, float r_3, float h_0, const IlPlace_0 thread* p_1)
{
    bool _S57;
    if((outcome_0 & 8U) != 0U)
    {
        if(p_1->entry_0)
        {
            _S57 = true;
        }
        else
        {
            float _S58 = p_1->along_0;
            if((p_1->along_0) >= 0.0f)
            {
                _S57 = _S58 <= h_0;
            }
            else
            {
                _S57 = false;
            }
            if(_S57)
            {
                _S57 = (p_1->radial_0) <= (r_3 + _S58);
            }
            else
            {
                _S57 = false;
            }
        }
    }
    else
    {
        _S57 = false;
    }
    return _S57;
}

bool il_scab_0(uint outcome_1, float r_4, float h_1, float scab_0, const IlPlace_0 thread* p_2)
{
    bool _S59;
    if((outcome_1 & 4U) != 0U)
    {
        _S59 = (p_2->along_0) >= (h_1 - scab_0);
    }
    else
    {
        _S59 = false;
    }
    if(_S59)
    {
        _S59 = (p_2->along_0) <= h_1;
    }
    else
    {
        _S59 = false;
    }
    if(_S59)
    {
        _S59 = (p_2->radial_0) <= (r_4 + h_1);
    }
    else
    {
        _S59 = false;
    }
    return _S59;
}

bool il_crater_0(uint outcome_2, float r_5, float depth_0, float a_1, const IlPlace_0 thread* p_3)
{
    bool _S60;
    if((outcome_2 & 2U) != 0U)
    {
        _S60 = (p_3->along_0) >= 0.0f;
    }
    else
    {
        _S60 = false;
    }
    if(_S60)
    {
        _S60 = (p_3->along_0) <= depth_0;
    }
    else
    {
        _S60 = false;
    }
    if(_S60)
    {
        _S60 = (p_3->radial_0) <= (max(r_5, a_1));
    }
    else
    {
        _S60 = false;
    }
    return _S60;
}

uint a_counters_1(KernelContext_0 thread* kernelContext_26)
{
    uint _S61 = a_best_root_0(kernelContext_26);
    return _S61 + kernelContext_26->params_0->body_count_0;
}

void il_breaks_0(uint i_0, KernelContext_0 thread* kernelContext_27)
{
    bool cut_0;
    if(i_0 >= (kernelContext_27->params_0->bond_count_0))
    {
        cut_0 = true;
    }
    else
    {
        uint _S62 = w_alive_0(kernelContext_27);
        cut_0 = (*(kernelContext_27->work_0+(_S62 + i_0))) == 0U;
    }
    if(cut_0)
    {
        return;
    }
    uint4 _S63 = uint4(*(kernelContext_27->bonds_0+i_0)) ;
    uint _S64 = _S63.x;
    uint _S65 = body_of_0(_S64, kernelContext_27);
    bool _S66 = il_struck_0(_S65, kernelContext_27);
    if(!_S66)
    {
        return;
    }
    uint _S67 = lp_body_0(_S65, kernelContext_27);
    float4 _S68 = float4(*(kernelContext_27->lp_0+_S67)) ;
    float4 _S69 = float4(*(kernelContext_27->lp_0+(_S67 + 1U))) ;
    uint outcome_3 = (as_type<uint>((_S68.w)));
    if((outcome_3 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_3;
    thread float3 d_3;
    thread float speed_2;
    il_path_0(_S65, &contact_3, &d_3, &speed_2, kernelContext_27);
    uint _S70 = il_entry_0(_S65, kernelContext_27);
    IlStriker_0 _S71 = il_striker_0(_S65, kernelContext_27);
    float _S72 = max(_S71.radius_0, 0.00100000004749745f);
    float h_2 = _S68.x;
    IlPlace_0 _S73 = il_place_0(_S64, _S70, contact_3, d_3, kernelContext_27);
    IlPlace_0 _S74 = il_place_0(_S63.y, _S70, contact_3, d_3, kernelContext_27);
    thread IlPlace_0 _S75 = _S73;
    bool _S76 = il_plug_0(outcome_3, _S72, h_2, &_S75);
    thread IlPlace_0 _S77 = _S74;
    bool _S78 = il_plug_0(outcome_3, _S72, h_2, &_S77);
    if(_S76 != _S78)
    {
        cut_0 = true;
    }
    else
    {
        float _S79 = _S69.y;
        thread IlPlace_0 _S80 = _S73;
        bool _S81 = il_scab_0(outcome_3, _S72, h_2, _S79, &_S80);
        thread IlPlace_0 _S82 = _S74;
        bool _S83 = il_scab_0(outcome_3, _S72, h_2, _S79, &_S82);
        cut_0 = _S81 != _S83;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S84 = _S69.x;
        float _S85 = _S69.z;
        thread IlPlace_0 _S86 = _S73;
        bool _S87 = il_crater_0(outcome_3, _S72, _S84, _S85, &_S86);
        cut_0 = _S87;
    }
    if(cut_0)
    {
        cut_0 = true;
    }
    else
    {
        float _S88 = _S69.x;
        float _S89 = _S69.z;
        thread IlPlace_0 _S90 = _S74;
        bool _S91 = il_crater_0(outcome_3, _S72, _S88, _S89, &_S90);
        cut_0 = _S91;
    }
    if(cut_0)
    {
        uint _S92 = w_alive_0(kernelContext_27);
        *(kernelContext_27->work_0+(_S92 + i_0)) = 0U;
        uint _S93 = a_dirty_0(kernelContext_27);
        uint _S94 = atomic_fetch_or_explicit(kernelContext_27->atom_0+(_S93 + _S65), 1U, memory_order_relaxed);
        uint _S95 = a_counters_1(kernelContext_27);
        uint _S96 = atomic_fetch_add_explicit(kernelContext_27->atom_0+_S95, 1U, memory_order_relaxed);
        uint _S97 = atomic_fetch_add_explicit(kernelContext_27->atom_0+(_S95 + 1U), 1U, memory_order_relaxed);
        uint _S98 = atomic_fetch_add_explicit(kernelContext_27->atom_0+(_S95 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

uint w_label_0(KernelContext_0 thread* kernelContext_28)
{
    uint _S99 = w_alive_0(kernelContext_28);
    return _S99 + kernelContext_28->params_0->bond_count_0;
}

uint w_slot_0(KernelContext_0 thread* kernelContext_29)
{
    uint _S100 = w_label_0(kernelContext_29);
    return _S100 + kernelContext_29->params_0->chunk_count_0;
}

uint w_new_0(KernelContext_0 thread* kernelContext_30)
{
    uint _S101 = w_slot_0(kernelContext_30);
    return _S101 + kernelContext_30->params_0->chunk_count_0;
}

uint w_rank_0(KernelContext_0 thread* kernelContext_31)
{
    uint _S102 = w_new_0(kernelContext_31);
    return _S102 + kernelContext_31->params_0->chunk_count_0;
}

uint w_free_0(KernelContext_0 thread* kernelContext_32)
{
    uint _S103 = w_rank_0(kernelContext_32);
    return _S103 + kernelContext_32->params_0->chunk_count_0;
}

uint w_free_rank_0(KernelContext_0 thread* kernelContext_33)
{
    uint _S104 = w_free_0(kernelContext_33);
    return _S104 + kernelContext_33->params_0->body_count_0;
}

uint w_free_list_0(KernelContext_0 thread* kernelContext_34)
{
    uint _S105 = w_free_rank_0(kernelContext_34);
    return _S105 + kernelContext_34->params_0->body_count_0;
}

uint w_pulses_0(KernelContext_0 thread* kernelContext_35)
{
    uint _S106 = w_free_list_0(kernelContext_35);
    return _S106 + kernelContext_35->params_0->body_count_0 + 2050U;
}

uint w_marks_0(KernelContext_0 thread* kernelContext_36)
{
    uint _S107 = w_pulses_0(kernelContext_36);
    return _S107 + 4U * kernelContext_36->params_0->chunk_count_0;
}

uint w_carry_0(KernelContext_0 thread* kernelContext_37)
{
    uint _S108 = w_marks_0(kernelContext_37);
    return _S108 + kernelContext_37->params_0->chunk_count_0;
}

float3 to_rubble_0(float3 v_2)
{
    return float3(v_2.x, v_2.z, - v_2.y);
}

uint a_il_0(uint b_16, KernelContext_0 thread* kernelContext_38)
{
    uint _S109 = a_strike_0(kernelContext_38);
    return _S109 + 23U * kernelContext_38->params_0->body_count_0 + 4U * b_16;
}

float read64_0(uint at_0, float scale_0, KernelContext_0 thread* kernelContext_39)
{
    uint _S110 = atomic_load_explicit(kernelContext_39->atom_0+(at_0 + 1U), memory_order_relaxed);
    int hi_0 = (as_type<int>((_S110)));
    uint lo_0 = atomic_load_explicit(kernelContext_39->atom_0+at_0, memory_order_relaxed);
    return (float(hi_0) * 4.294967296e+09f + float(lo_0)) / scale_0;
}

void il_carry_0(uint k_2, KernelContext_0 thread* kernelContext_40)
{
    if(k_2 >= (kernelContext_40->params_0->chunk_count_0))
    {
        return;
    }
    uint _S111 = w_carry_0(kernelContext_40);
    uint _S112 = _S111 + k_2;
    uint device* _S113 = kernelContext_40->work_0+_S112;
    uint c_2 = *_S113;
    if((*_S113) == 0U)
    {
        return;
    }
    *(kernelContext_40->work_0+_S112) = 0U;
    uint b0_0 = c_2 - 1U;
    bool _S114 = il_struck_0(b0_0, kernelContext_40);
    if(!_S114)
    {
        return;
    }
    thread float3 contact_4;
    thread float3 d_4;
    thread float speed_3;
    il_path_0(b0_0, &contact_4, &d_4, &speed_3, kernelContext_40);
    float3 dr_0 = to_rubble_0(d_4);
    IlStriker_0 _S115 = il_striker_0(b0_0, kernelContext_40);
    uint _S116 = a_il_0(b0_0, kernelContext_40);
    float _S117 = read64_0(_S116 + 2U, kernelContext_40->params_0->scale_m_0, kernelContext_40);
    float3 v_wall_0 = (float4(*(kernelContext_40->bodies_0+(7U * b0_0 + 3U))) ).xyz;
    uint _S118 = 7U * _S115.body_0;
    float3 v_shot_0 = (float4(*(kernelContext_40->bodies_0+(_S118 + 3U))) ).xyz;
    float closing_0 = dot(v_shot_0 - v_wall_0, dr_0);
    float v_r_0 = _S115.mass_0 * closing_0 / (_S115.mass_0 + _S117);
    float3 _S119 = v_shot_0 + dr_0 * float3((v_r_0 - closing_0)) ;
    float3 _S120 = v_wall_0 + dr_0 * float3(v_r_0) ;
    float _S121 = il_t_in_0(b0_0, kernelContext_40);
    float3 _S122 = (float4(*(kernelContext_40->bodies_0+_S118)) ).xyz + dr_0 * float3(((closing_0 - v_r_0) * clamp(_S121, 0.0f, kernelContext_40->params_0->dt_0))) ;
    uint _S123 = body_of_0(k_2, kernelContext_40);
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
        uint s_at_0 = 7U * (_S115.body_0 + region_0 * kernelContext_40->params_0->max_bodies_0);
        (*(kernelContext_40->bodies_0+s_at_0)).xyz = _S122;
        (*(kernelContext_40->bodies_0+(s_at_0 + 3U))).xyz = _S119;
        (*(kernelContext_40->bodies_0+(7U * (_S123 + region_0 * kernelContext_40->params_0->max_bodies_0) + 3U))).xyz = _S120;
        region_0 = region_0 + 1U;
    }
    return;
}

void add64_0(uint at_1, float value_0, float scale_1, KernelContext_0 thread* kernelContext_41)
{
    float x_1 = round(value_0 * scale_1);
    float x_2;
    if(!((abs(x_1)) < 1.073741824e+09f))
    {
        uint _S124 = a_counters_1(kernelContext_41);
        uint _S125 = atomic_fetch_or_explicit(kernelContext_41->atom_0+(_S124 + 3U), 2U, memory_order_relaxed);
        x_2 = 0.0f;
    }
    else
    {
        x_2 = x_1;
    }
    int v_3 = int(x_2);
    uint lo_1 = (as_type<uint>((v_3)));
    uint old_0 = atomic_fetch_add_explicit(kernelContext_41->atom_0+at_1, lo_1, memory_order_relaxed);
    uint carry_0;
    if((old_0 + lo_1) < old_0)
    {
        carry_0 = 1U;
    }
    else
    {
        carry_0 = 0U;
    }
    uint _S126;
    if(v_3 < int(0))
    {
        _S126 = 4294967295U;
    }
    else
    {
        _S126 = 0U;
    }
    uint hi_1 = _S126 + carry_0;
    if(hi_1 != 0U)
    {
        uint _S127 = atomic_fetch_add_explicit(kernelContext_41->atom_0+(at_1 + 1U), hi_1, memory_order_relaxed);
    }
    return;
}

void il_mark_0(uint k_3, KernelContext_0 thread* kernelContext_42)
{
    if(k_3 >= (kernelContext_42->params_0->chunk_count_0))
    {
        return;
    }
    uint _S128 = body_of_0(k_3, kernelContext_42);
    bool _S129 = il_struck_0(_S128, kernelContext_42);
    if(!_S129)
    {
        return;
    }
    uint _S130 = lp_body_0(_S128, kernelContext_42);
    float4 _S131 = float4(*(kernelContext_42->lp_0+_S130)) ;
    uint outcome_4 = (as_type<uint>((_S131.w)));
    if((outcome_4 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_5;
    thread float3 d_5;
    thread float speed_4;
    il_path_0(_S128, &contact_5, &d_5, &speed_4, kernelContext_42);
    uint _S132 = il_entry_0(_S128, kernelContext_42);
    IlStriker_0 _S133 = il_striker_0(_S128, kernelContext_42);
    float _S134 = max(_S133.radius_0, 0.00100000004749745f);
    IlPlace_0 _S135 = il_place_0(k_3, _S132, contact_5, d_5, kernelContext_42);
    float _S136 = _S131.x;
    thread IlPlace_0 _S137 = _S135;
    bool _S138 = il_plug_0(outcome_4, _S134, _S136, &_S137);
    if(_S138)
    {
        uint _S139 = w_carry_0(kernelContext_42);
        *(kernelContext_42->work_0+(_S139 + k_3)) = _S128 + 1U;
        uint _S140 = a_il_0(_S128, kernelContext_42);
        add64_0(_S140 + 2U, (float4(*(kernelContext_42->chunks_0+4U * k_3)) ).w, kernelContext_42->params_0->scale_m_0, kernelContext_42);
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
    float r_6 = abs(strain_rate_0);
    float4 _S141 = mat_0->dif_0;
    float ref_0 = mat_0->dif_0.x;
    if(r_6 <= ref_0)
    {
        return 1.0f;
    }
    float _S142 = _S141.z;
    float f_0;
    if(r_6 <= _S142)
    {
        float _S143 = (metal::fast::divide((r_6), (ref_0)));
        float _S144 = (metal::fast::pow((_S143), (_S141.y)));
        f_0 = _S144;
    }
    else
    {
        float _S145 = (metal::fast::divide((_S142), (ref_0)));
        float _S146 = (metal::fast::pow((_S145), (_S141.y)));
        float _S147 = (metal::fast::divide((r_6), (_S142)));
        float _S148 = (metal::fast::pow((_S147), (_S141.w)));
        f_0 = _S146 * _S148;
    }
    return clamp(f_0, 1.0f, mat_0->misc_0.x);
}

void il_impulse_0(uint b_17, uint sb_0, float m_shot_0, float3 contact_6, float3 d_6, float j_0, KernelContext_0 thread* kernelContext_43)
{
    float3 jr_0 = to_rubble_0(d_6) * float3(j_0) ;
    uint at_2 = 7U * b_17;
    float inv_mass_1 = (float4(*(kernelContext_43->bodies_0+at_2)) ).w;
    float3 _S149 = jr_0 * float3(inv_mass_1) ;
    float3 _S150 = float3(0.0f) ;
    float3 dw_0;
    if(inv_mass_1 > 0.0f)
    {
        Quat_0 q_7 = quat_of_0(float4(*(kernelContext_43->bodies_0+(at_2 + 1U))) );
        float3 r_7 = to_rubble_0(contact_6) - (float4(*(kernelContext_43->bodies_0+at_2)) ).xyz;
        float3 _S151 = (float4(*(kernelContext_43->bodies_0+(at_2 + 2U))) ).xyz;
        thread Quat_0 _S152 = q_7;
        Quat_0 _S153 = conj_0(&_S152);
        float3 _S154 = cross(r_7, jr_0);
        thread Quat_0 _S155 = _S153;
        float3 _S156 = rotate_0(&_S155, _S154);
        float3 _S157 = _S151 * _S156;
        thread Quat_0 _S158 = q_7;
        float3 _S159 = rotate_0(&_S158, _S157);
        dw_0 = _S159;
    }
    else
    {
        dw_0 = _S150;
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
        uint s_at_1 = 7U * (sb_0 + region_1 * kernelContext_43->params_0->max_bodies_0) + 3U;
        (*(kernelContext_43->bodies_0+s_at_1)).xyz = (float4(*(kernelContext_43->bodies_0+s_at_1)) ).xyz - jr_0 / float3(m_shot_0) ;
        uint w_at_0 = 7U * (b_17 + region_1 * kernelContext_43->params_0->max_bodies_0);
        uint _S160 = w_at_0 + 3U;
        (*(kernelContext_43->bodies_0+_S160)).xyz = (float4(*(kernelContext_43->bodies_0+_S160)) ).xyz + _S149;
        uint _S161 = w_at_0 + 4U;
        (*(kernelContext_43->bodies_0+_S161)).xyz = (float4(*(kernelContext_43->bodies_0+_S161)) ).xyz + dw_0;
        region_1 = region_1 + 1U;
    }
    return;
}

void il_strike_0(uint b_18, KernelContext_0 thread* kernelContext_44)
{
    bool damage_0;
    if(b_18 >= (kernelContext_44->params_0->body_count_0))
    {
        damage_0 = true;
    }
    else
    {
        bool _S162 = il_struck_0(b_18, kernelContext_44);
        damage_0 = !_S162;
    }
    if(damage_0)
    {
        return;
    }
    float _S163 = il_t_in_0(b_18, kernelContext_44);
    uint _S164 = il_entry_0(b_18, kernelContext_44);
    thread float3 contact_7;
    thread float3 d_7;
    thread float speed_5;
    il_path_0(b_18, &contact_7, &d_7, &speed_5, kernelContext_44);
    IlStriker_0 _S165 = il_striker_0(b_18, kernelContext_44);
    float _S166 = max(_S165.radius_0, 0.00100000004749745f);
    if((_S165.launch_0) != 0U)
    {
        uint _S167 = w_marks_0(kernelContext_44);
        damage_0 = (*(kernelContext_44->work_0+(_S167 + _S164))) == (_S165.launch_0);
    }
    else
    {
        damage_0 = false;
    }
    if(damage_0)
    {
        uint _S168 = lp_body_0(b_18, kernelContext_44);
        packed_float4 _S169 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_44->lp_0+_S168) = _S169;
        *(kernelContext_44->lp_0+(_S168 + 1U)) = _S169;
        return;
    }
    uint _S170 = a_il_0(b_18, kernelContext_44);
    uint _S171 = atomic_load_explicit(kernelContext_44->atom_0+_S170, memory_order_relaxed);
    uint _S172 = atomic_load_explicit(kernelContext_44->atom_0+(_S170 + 1U), memory_order_relaxed);
    if(_S171 != _S172)
    {
        uint _S173 = a_counters_1(kernelContext_44);
        uint _S174 = atomic_fetch_or_explicit(kernelContext_44->atom_0+(_S173 + 3U), 16U, memory_order_relaxed);
    }
    float _S175 = max((as_type<float>((_S171))) - _S166, 9.99999997475242708e-07f);
    uint o_1 = kernelContext_44->params_0->ext_chunks_0 + 4U * _S164;
    float4 _S176 = float4(*(kernelContext_44->ext_0+(o_1 + 1U))) ;
    thread JointMaterial_0 dm_0;
    float4 _S177 = float4(0.0f) ;
    (&dm_0)->strength_0 = _S177;
    (&dm_0)->energy_0 = _S177;
    (&dm_0)->dif_0 = float4(*(kernelContext_44->ext_0+(o_1 + 2U))) ;
    float4 _S178 = float4(*(kernelContext_44->ext_0+(o_1 + 3U))) ;
    float _S179 = _S178.x;
    (&dm_0)->misc_0 = float4(_S179, 0.0f, 0.0f, 0.0f);
    (&dm_0)->kind_flags_0 = uint4(0U) ;
    float nu_0 = _S178.y;
    float e_target_0 = _S178.z / (1.0f - nu_0 * nu_0);
    float e_star_0 = 1.0f / (1.0f / _S165.modulus_0 + 1.0f / e_target_0);
    float _S180 = max(speed_5, 0.00100000004749745f);
    float _S181 = sqrt(_S166);
    float h0_0 = pow(15.0f * _S165.m_eff_0 * _S180 * _S180 / (16.0f * e_star_0 * _S181), 0.40000000596046448f);
    float t_c_0 = 2.94000005722045898f * h0_0 / _S180;
    float force_0 = 1.33333337306976318f * e_star_0 * _S181 * pow(h0_0, 1.5f);
    float a_2 = sqrt(_S166 * h0_0);
    float p0_0 = 3.0f * force_0 / (6.28318548202514648f * a_2 * a_2);
    float ring_0 = (1.0f - 2.0f * nu_0) * p0_0 / 3.0f;
    float k_4;
    if(_S179 > 0.0f)
    {
        float _S182 = _S180 / _S175;
        thread JointMaterial_0 _S183 = dm_0;
        float _S184 = dif_factor_0(&_S183, _S182);
        k_4 = _S184;
    }
    else
    {
        k_4 = 1.0f;
    }
    float ft_0 = k_4 * _S176.y;
    float fc_0 = k_4 * _S176.z;
    float tau_0 = k_4 * _S176.w;
    if(p0_0 > fc_0)
    {
        damage_0 = true;
    }
    else
    {
        damage_0 = ring_0 > ft_0;
    }
    float _S185 = fc_0 * (3.14159274101257324f * _S166 * _S166);
    float _S186 = min(_S185 * _S175, 1.57079637050628662f * (2.0f * _S166 + _S175) * _S175 * _S175 * tau_0);
    float _S187 = 2.0f * _S186;
    float v_bl_0 = sqrt(_S187 / _S165.m_eff_0);
    bool perforate_0;
    if(damage_0)
    {
        perforate_0 = _S180 >= v_bl_0;
    }
    else
    {
        perforate_0 = false;
    }
    float r2_0 = _S166 + _S175;
    float _S188 = _S176.x;
    float plug_0 = _S188 * 3.14159274101257324f * _S175 / 3.0f * (_S166 * _S166 + _S166 * r2_0 + r2_0 * r2_0);
    float v_res_0;
    if(perforate_0)
    {
        v_res_0 = _S165.mass_0 / (_S165.mass_0 + plug_0) * sqrt(max(_S180 * _S180 - v_bl_0 * v_bl_0, 0.0f));
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
        depth_1 = min(0.5f * _S165.m_eff_0 * _S180 * _S180 / _S185, _S175);
    }
    else
    {
        depth_1 = 0.0f;
    }
    bool crater_0 = depth_1 > a_2;
    float c_3 = sqrt(e_target_0 / _S188);
    float z_t_0 = _S188 * c_3;
    float rho_p_0 = _S165.mass_0 / (4.18879032135009766f * _S166 * _S166 * _S166);
    float z_p_0 = rho_p_0 * sqrt(_S165.modulus_0 / rho_p_0);
    float mean_pressure_0 = force_0 / (3.14159274101257324f * a_2 * a_2);
    float _S189 = min(min(z_t_0 * _S180 * z_p_0 / (z_p_0 + z_t_0), mean_pressure_0), fc_0);
    float source_radius_0;
    if(mean_pressure_0 > fc_0)
    {
        source_radius_0 = max(a_2, _S166);
    }
    else
    {
        source_radius_0 = a_2;
    }
    float pulse_0 = c_3 * t_c_0;
    float back_0 = _S189 * min(source_radius_0 / _S175, 1.0f) * min(2.0f * _S175 / pulse_0, 1.0f);
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
        scab_1 = 0.5f * min(pulse_0, _S175);
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
    uint _S190;
    if(crater_0)
    {
        _S190 = 2U;
    }
    else
    {
        _S190 = 0U;
    }
    uint _S191 = region_2 | _S190;
    if(spall_0)
    {
        region_2 = 4U;
    }
    else
    {
        region_2 = 0U;
    }
    uint _S192 = _S191 | region_2;
    if(perforate_0)
    {
        region_2 = 8U;
    }
    else
    {
        region_2 = 0U;
    }
    uint outcome_5 = _S192 | region_2;
    uint _S193 = lp_body_0(b_18, kernelContext_44);
    *(kernelContext_44->lp_0+_S193) = packed_float4(float4(_S175, v_bl_0, v_res_0, (as_type<float>((outcome_5))))) ;
    packed_float4 device* _S194 = kernelContext_44->lp_0+(_S193 + 1U);
    if(crater_0)
    {
        k_4 = depth_1;
    }
    else
    {
        k_4 = 0.0f;
    }
    *_S194 = packed_float4(float4(k_4, scab_1, a_2, _S186)) ;
    if(perforate_0)
    {
        uint _S195 = a_counters_1(kernelContext_44);
        uint _S196 = atomic_fetch_add_explicit(kernelContext_44->atom_0+(_S195 + 10U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_0)
        {
            uint _S197 = a_counters_1(kernelContext_44);
            uint _S198 = atomic_fetch_add_explicit(kernelContext_44->atom_0+(_S197 + 9U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_0)
            {
                uint _S199 = a_counters_1(kernelContext_44);
                uint _S200 = atomic_fetch_add_explicit(kernelContext_44->atom_0+(_S199 + 8U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S201 = a_counters_1(kernelContext_44);
                uint _S202 = atomic_fetch_add_explicit(kernelContext_44->atom_0+(_S201 + 7U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_0)
    {
        if((kernelContext_44->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_18, _S165.body_0, _S165.mass_0, contact_7, d_7, _S165.m_eff_0 * (_S180 - sqrt(max(_S180 * _S180 - _S187 / _S165.m_eff_0, 0.0f))), kernelContext_44);
        }
    }
    else
    {
        if(damage_0)
        {
            il_impulse_0(b_18, _S165.body_0, _S165.mass_0, contact_7, d_7, _S165.m_eff_0 * _S180, kernelContext_44);
            float3 _S203 = to_rubble_0(d_7) * float3((_S180 * clamp(_S163, 0.0f, kernelContext_44->params_0->dt_0))) ;
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
                uint _S204 = 7U * (_S165.body_0 + region_2 * kernelContext_44->params_0->max_bodies_0);
                (*(kernelContext_44->bodies_0+_S204)).xyz = (float4(*(kernelContext_44->bodies_0+_S204)) ).xyz + _S203;
                region_2 = region_2 + 1U;
            }
        }
    }
    return;
}

float3 chunk_box_world_0(uint k_5, Quat_0 thread* q_8, KernelContext_0 thread* kernelContext_45)
{
    uint _S205 = body_of_0(k_5, kernelContext_45);
    thread float3 origin_2;
    structure_pose_0(_S205, 7U * _S205, q_8, &origin_2, kernelContext_45);
    float3 _S206 = (float4(*(kernelContext_45->geometry_0+3U * k_5)) ).xyz;
    thread Quat_0 _S207 = *q_8;
    float3 _S208 = rotate_0(&_S207, _S206);
    return origin_2 + _S208;
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

float2 il_interval_0(uint k_6, float3 p_4, float3 d_8, float pad_0, KernelContext_0 thread* kernelContext_46)
{
    thread Quat_0 q_9;
    float3 _S209 = chunk_box_world_0(k_6, &q_9, kernelContext_46);
    uint _S210 = 3U * k_6;
    Quat_0 _S211 = quat_of_0(float4(*(kernelContext_46->geometry_0+(_S210 + 2U))) );
    thread Quat_0 _S212 = q_9;
    thread Quat_0 _S213 = _S211;
    Quat_0 _S214 = quat_mul_0(&_S212, &_S213);
    thread Quat_0 _S215 = _S214;
    Quat_0 _S216 = normalized_0(&_S215);
    float3 _S217 = (float4(*(kernelContext_46->geometry_0+(_S210 + 1U))) ).xyz;
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
        thread Quat_0 _S218 = _S216;
        float3 _S219 = rotate_0(&_S218, e_0);
        float o_2 = dot(p_4 - _S209, _S219);
        float dv_0 = dot(d_8, _S219);
        float ext_1 = _S217[ax_0] + pad_0;
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
        float _S220 = min(t1_0, max(ta_0, tb_0));
        t0_0 = max(t0_0, min(ta_0, tb_0));
        t1_0 = _S220;
        ax_0 = ax_0 + 1U;
    }
    return float2(t0_0, t1_0);
}

void il_extend_0(uint k_7, uint from_0, KernelContext_0 thread* kernelContext_47)
{
    if(k_7 >= (kernelContext_47->params_0->chunk_count_0))
    {
        return;
    }
    uint _S221 = body_of_0(k_7, kernelContext_47);
    bool _S222 = il_struck_0(_S221, kernelContext_47);
    if(!_S222)
    {
        return;
    }
    thread float3 contact_8;
    thread float3 d_9;
    thread float speed_6;
    il_path_0(_S221, &contact_8, &d_9, &speed_6, kernelContext_47);
    IlStriker_0 _S223 = il_striker_0(_S221, kernelContext_47);
    uint _S224 = a_il_0(_S221, kernelContext_47);
    uint _S225 = atomic_load_explicit(kernelContext_47->atom_0+(_S224 + from_0), memory_order_relaxed);
    float end_0 = (as_type<float>((_S225)));
    uint to_0 = _S224 + 1U - from_0;
    uint _S226 = atomic_fetch_max_explicit(kernelContext_47->atom_0+to_0, (as_type<uint>((end_0))), memory_order_relaxed);
    float2 _S227 = il_interval_0(k_7, contact_8, d_9, _S223.radius_0, kernelContext_47);
    float _S228 = _S227.x;
    float _S229 = _S227.y;
    bool _S230;
    if(_S228 <= _S229)
    {
        _S230 = _S228 <= (end_0 + 0.00100000004749745f + 0.01999999955296516f * _S223.radius_0);
    }
    else
    {
        _S230 = false;
    }
    if(_S230)
    {
        _S230 = _S229 > end_0;
    }
    else
    {
        _S230 = false;
    }
    if(_S230)
    {
        uint _S231 = atomic_fetch_max_explicit(kernelContext_47->atom_0+to_0, (as_type<uint>((_S229))), memory_order_relaxed);
    }
    return;
}

void il_begin_0(uint b_19, KernelContext_0 thread* kernelContext_48)
{
    bool _S232;
    if(b_19 >= (kernelContext_48->params_0->body_count_0))
    {
        _S232 = true;
    }
    else
    {
        bool _S233 = il_struck_0(b_19, kernelContext_48);
        _S232 = !_S233;
    }
    if(_S232)
    {
        return;
    }
    thread float3 contact_9;
    thread float3 d_10;
    thread float speed_7;
    il_path_0(b_19, &contact_9, &d_10, &speed_7, kernelContext_48);
    uint _S234 = il_entry_0(b_19, kernelContext_48);
    IlStriker_0 _S235 = il_striker_0(b_19, kernelContext_48);
    float2 _S236 = il_interval_0(_S234, contact_9, d_10, _S235.radius_0, kernelContext_48);
    uint end_1 = (as_type<uint>((max(_S236.y, 0.0f))));
    uint _S237 = a_il_0(b_19, kernelContext_48);
    atomic_store_explicit(kernelContext_48->atom_0+_S237, end_1, memory_order_relaxed);
    atomic_store_explicit(kernelContext_48->atom_0+(_S237 + 1U), end_1, memory_order_relaxed);
    atomic_store_explicit(kernelContext_48->atom_0+(_S237 + 2U), 0U, memory_order_relaxed);
    atomic_store_explicit(kernelContext_48->atom_0+(_S237 + 3U), 0U, memory_order_relaxed);
    return;
}

void il_breaks_1(uint i_1, KernelContext_0 thread* kernelContext_49)
{
    bool cut_1;
    if(i_1 >= (kernelContext_49->params_0->bond_count_0))
    {
        cut_1 = true;
    }
    else
    {
        uint _S238 = w_alive_0(kernelContext_49);
        cut_1 = (*(kernelContext_49->work_0+(_S238 + i_1))) == 0U;
    }
    if(cut_1)
    {
        return;
    }
    uint4 _S239 = uint4(*(kernelContext_49->bonds_0+i_1)) ;
    uint _S240 = _S239.x;
    uint _S241 = body_of_0(_S240, kernelContext_49);
    bool _S242 = il_struck_1(_S241, kernelContext_49);
    if(!_S242)
    {
        return;
    }
    uint _S243 = lp_body_0(_S241, kernelContext_49);
    float4 _S244 = float4(*(kernelContext_49->lp_0+_S243)) ;
    float4 _S245 = float4(*(kernelContext_49->lp_0+(_S243 + 1U))) ;
    uint outcome_6 = (as_type<uint>((_S244.w)));
    if((outcome_6 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_10;
    thread float3 d_11;
    thread float speed_8;
    il_path_1(_S241, &contact_10, &d_11, &speed_8, kernelContext_49);
    uint _S246 = il_entry_1(_S241, kernelContext_49);
    IlStriker_0 _S247 = il_striker_1(_S241, kernelContext_49);
    float _S248 = max(_S247.radius_0, 0.00100000004749745f);
    float h_3 = _S244.x;
    IlPlace_0 _S249 = il_place_0(_S240, _S246, contact_10, d_11, kernelContext_49);
    IlPlace_0 _S250 = il_place_0(_S239.y, _S246, contact_10, d_11, kernelContext_49);
    thread IlPlace_0 _S251 = _S249;
    bool _S252 = il_plug_0(outcome_6, _S248, h_3, &_S251);
    thread IlPlace_0 _S253 = _S250;
    bool _S254 = il_plug_0(outcome_6, _S248, h_3, &_S253);
    if(_S252 != _S254)
    {
        cut_1 = true;
    }
    else
    {
        float _S255 = _S245.y;
        thread IlPlace_0 _S256 = _S249;
        bool _S257 = il_scab_0(outcome_6, _S248, h_3, _S255, &_S256);
        thread IlPlace_0 _S258 = _S250;
        bool _S259 = il_scab_0(outcome_6, _S248, h_3, _S255, &_S258);
        cut_1 = _S257 != _S259;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S260 = _S245.x;
        float _S261 = _S245.z;
        thread IlPlace_0 _S262 = _S249;
        bool _S263 = il_crater_0(outcome_6, _S248, _S260, _S261, &_S262);
        cut_1 = _S263;
    }
    if(cut_1)
    {
        cut_1 = true;
    }
    else
    {
        float _S264 = _S245.x;
        float _S265 = _S245.z;
        thread IlPlace_0 _S266 = _S250;
        bool _S267 = il_crater_0(outcome_6, _S248, _S264, _S265, &_S266);
        cut_1 = _S267;
    }
    if(cut_1)
    {
        uint _S268 = w_alive_0(kernelContext_49);
        *(kernelContext_49->work_0+(_S268 + i_1)) = 0U;
        uint _S269 = a_dirty_0(kernelContext_49);
        uint _S270 = atomic_fetch_or_explicit(kernelContext_49->atom_0+(_S269 + _S241), 1U, memory_order_relaxed);
        uint _S271 = a_counters_1(kernelContext_49);
        uint _S272 = atomic_fetch_add_explicit(kernelContext_49->atom_0+_S271, 1U, memory_order_relaxed);
        uint _S273 = atomic_fetch_add_explicit(kernelContext_49->atom_0+(_S271 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

void il_carry_1(uint k_8, KernelContext_0 thread* kernelContext_50)
{
    if(k_8 >= (kernelContext_50->params_0->chunk_count_0))
    {
        return;
    }
    uint _S274 = w_carry_0(kernelContext_50);
    uint _S275 = _S274 + k_8;
    uint device* _S276 = kernelContext_50->work_0+_S275;
    uint c_4 = *_S276;
    if((*_S276) == 0U)
    {
        return;
    }
    *(kernelContext_50->work_0+_S275) = 0U;
    uint b0_1 = c_4 - 1U;
    bool _S277 = il_struck_1(b0_1, kernelContext_50);
    if(!_S277)
    {
        return;
    }
    thread float3 contact_11;
    thread float3 d_12;
    thread float speed_9;
    il_path_1(b0_1, &contact_11, &d_12, &speed_9, kernelContext_50);
    float3 dr_1 = to_rubble_0(d_12);
    IlStriker_0 _S278 = il_striker_1(b0_1, kernelContext_50);
    uint _S279 = a_il_0(b0_1, kernelContext_50);
    float _S280 = read64_0(_S279 + 2U, kernelContext_50->params_0->scale_m_0, kernelContext_50);
    float3 v_wall_1 = (float4(*(kernelContext_50->bodies_0+(7U * b0_1 + 3U))) ).xyz;
    uint _S281 = 7U * _S278.body_0;
    float3 v_shot_1 = (float4(*(kernelContext_50->bodies_0+(_S281 + 3U))) ).xyz;
    float closing_1 = dot(v_shot_1 - v_wall_1, dr_1);
    float v_r_1 = _S278.mass_0 * closing_1 / (_S278.mass_0 + _S280);
    float3 _S282 = v_shot_1 + dr_1 * float3((v_r_1 - closing_1)) ;
    float3 _S283 = v_wall_1 + dr_1 * float3(v_r_1) ;
    float _S284 = il_t_in_1(b0_1, kernelContext_50);
    float3 _S285 = (float4(*(kernelContext_50->bodies_0+_S281)) ).xyz + dr_1 * float3(((closing_1 - v_r_1) * clamp(_S284, 0.0f, kernelContext_50->params_0->dt_0))) ;
    uint _S286 = body_of_0(k_8, kernelContext_50);
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
        uint s_at_2 = 7U * (_S278.body_0 + region_3 * kernelContext_50->params_0->max_bodies_0);
        (*(kernelContext_50->bodies_0+s_at_2)).xyz = _S285;
        (*(kernelContext_50->bodies_0+(s_at_2 + 3U))).xyz = _S282;
        (*(kernelContext_50->bodies_0+(7U * (_S286 + region_3 * kernelContext_50->params_0->max_bodies_0) + 3U))).xyz = _S283;
        region_3 = region_3 + 1U;
    }
    return;
}

void il_mark_1(uint k_9, KernelContext_0 thread* kernelContext_51)
{
    if(k_9 >= (kernelContext_51->params_0->chunk_count_0))
    {
        return;
    }
    uint _S287 = body_of_0(k_9, kernelContext_51);
    bool _S288 = il_struck_1(_S287, kernelContext_51);
    if(!_S288)
    {
        return;
    }
    uint _S289 = lp_body_0(_S287, kernelContext_51);
    float4 _S290 = float4(*(kernelContext_51->lp_0+_S289)) ;
    float4 _S291 = float4(*(kernelContext_51->lp_0+(_S289 + 1U))) ;
    uint outcome_7 = (as_type<uint>((_S290.w)));
    if((outcome_7 & 14U) == 0U)
    {
        return;
    }
    thread float3 contact_12;
    thread float3 d_13;
    thread float speed_10;
    il_path_1(_S287, &contact_12, &d_13, &speed_10, kernelContext_51);
    uint _S292 = il_entry_1(_S287, kernelContext_51);
    IlStriker_0 _S293 = il_striker_1(_S287, kernelContext_51);
    float _S294 = max(_S293.radius_0, 0.00100000004749745f);
    IlPlace_0 _S295 = il_place_0(k_9, _S292, contact_12, d_13, kernelContext_51);
    float _S296 = _S290.x;
    thread IlPlace_0 _S297 = _S295;
    bool _S298 = il_plug_0(outcome_7, _S294, _S296, &_S297);
    float2 _S299 = il_interval_0(k_9, contact_12 - d_13 * float3(_S293.bound_0) , d_13, _S293.bound_0, kernelContext_51);
    float _S300 = _S299.x;
    float _S301 = _S299.y;
    bool path_0;
    if(_S300 <= _S301)
    {
        path_0 = _S300 <= (_S296 + 2.0f * _S293.bound_0);
    }
    else
    {
        path_0 = false;
    }
    if(path_0)
    {
        path_0 = _S301 >= 0.0f;
    }
    else
    {
        path_0 = false;
    }
    if(_S298)
    {
        path_0 = true;
    }
    if(path_0)
    {
        path_0 = true;
    }
    else
    {
        float _S302 = _S291.y;
        thread IlPlace_0 _S303 = _S295;
        bool _S304 = il_scab_0(outcome_7, _S294, _S296, _S302, &_S303);
        path_0 = _S304;
    }
    if(path_0)
    {
        path_0 = true;
    }
    else
    {
        float _S305 = _S291.x;
        float _S306 = _S291.z;
        thread IlPlace_0 _S307 = _S295;
        bool _S308 = il_crater_0(outcome_7, _S294, _S305, _S306, &_S307);
        path_0 = _S308;
    }
    if(path_0)
    {
        uint _S309 = w_marks_0(kernelContext_51);
        *(kernelContext_51->work_0+(_S309 + k_9)) = _S293.launch_0;
    }
    if(_S298)
    {
        uint _S310 = w_carry_0(kernelContext_51);
        *(kernelContext_51->work_0+(_S310 + k_9)) = _S287 + 1U;
        uint _S311 = a_il_0(_S287, kernelContext_51);
        add64_0(_S311 + 2U, (float4(*(kernelContext_51->chunks_0+4U * k_9)) ).w, kernelContext_51->params_0->scale_m_0, kernelContext_51);
    }
    return;
}

void il_strike_1(uint b_20, KernelContext_0 thread* kernelContext_52)
{
    bool damage_1;
    if(b_20 >= (kernelContext_52->params_0->body_count_0))
    {
        damage_1 = true;
    }
    else
    {
        bool _S312 = il_struck_1(b_20, kernelContext_52);
        damage_1 = !_S312;
    }
    if(damage_1)
    {
        return;
    }
    float _S313 = il_t_in_1(b_20, kernelContext_52);
    uint _S314 = il_entry_1(b_20, kernelContext_52);
    thread float3 contact_13;
    thread float3 d_14;
    thread float speed_11;
    il_path_1(b_20, &contact_13, &d_14, &speed_11, kernelContext_52);
    IlStriker_0 _S315 = il_striker_1(b_20, kernelContext_52);
    float _S316 = max(_S315.radius_0, 0.00100000004749745f);
    if((_S315.launch_0) != 0U)
    {
        uint _S317 = w_marks_0(kernelContext_52);
        damage_1 = (*(kernelContext_52->work_0+(_S317 + _S314))) == (_S315.launch_0);
    }
    else
    {
        damage_1 = false;
    }
    if(damage_1)
    {
        uint _S318 = lp_body_0(b_20, kernelContext_52);
        packed_float4 _S319 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_52->lp_0+_S318) = _S319;
        *(kernelContext_52->lp_0+(_S318 + 1U)) = _S319;
        return;
    }
    uint _S320 = a_il_0(b_20, kernelContext_52);
    uint _S321 = atomic_load_explicit(kernelContext_52->atom_0+_S320, memory_order_relaxed);
    uint _S322 = atomic_load_explicit(kernelContext_52->atom_0+(_S320 + 1U), memory_order_relaxed);
    if(_S321 != _S322)
    {
        uint _S323 = a_counters_1(kernelContext_52);
        uint _S324 = atomic_fetch_or_explicit(kernelContext_52->atom_0+(_S323 + 3U), 16U, memory_order_relaxed);
    }
    float _S325 = max((as_type<float>((_S321))) - _S316, 9.99999997475242708e-07f);
    if(!(_S313 > 0.0f))
    {
        uint _S326 = lp_body_0(b_20, kernelContext_52);
        *(kernelContext_52->lp_0+_S326) = packed_float4(float4(_S325, 0.0f, 0.0f, 0.0f)) ;
        *(kernelContext_52->lp_0+(_S326 + 1U)) = packed_float4(float4(0.0f) ) ;
        return;
    }
    uint o_3 = kernelContext_52->params_0->ext_chunks_0 + 4U * _S314;
    float4 _S327 = float4(*(kernelContext_52->ext_0+(o_3 + 1U))) ;
    thread JointMaterial_0 dm_1;
    float4 _S328 = float4(0.0f) ;
    (&dm_1)->strength_0 = _S328;
    (&dm_1)->energy_0 = _S328;
    (&dm_1)->dif_0 = float4(*(kernelContext_52->ext_0+(o_3 + 2U))) ;
    float4 _S329 = float4(*(kernelContext_52->ext_0+(o_3 + 3U))) ;
    float _S330 = _S329.x;
    (&dm_1)->misc_0 = float4(_S330, 0.0f, 0.0f, 0.0f);
    (&dm_1)->kind_flags_0 = uint4(0U) ;
    float nu_1 = _S329.y;
    float e_target_1 = _S329.z / (1.0f - nu_1 * nu_1);
    float e_star_1 = 1.0f / (1.0f / _S315.modulus_0 + 1.0f / e_target_1);
    float _S331 = max(speed_11, 0.00100000004749745f);
    float _S332 = sqrt(_S316);
    float h0_1 = pow(15.0f * _S315.m_eff_0 * _S331 * _S331 / (16.0f * e_star_1 * _S332), 0.40000000596046448f);
    float t_c_1 = 2.94000005722045898f * h0_1 / _S331;
    float force_1 = 1.33333337306976318f * e_star_1 * _S332 * pow(h0_1, 1.5f);
    float a_3 = sqrt(_S316 * h0_1);
    float p0_1 = 3.0f * force_1 / (6.28318548202514648f * a_3 * a_3);
    float ring_1 = (1.0f - 2.0f * nu_1) * p0_1 / 3.0f;
    float k_10;
    if(_S330 > 0.0f)
    {
        float _S333 = _S331 / _S325;
        thread JointMaterial_0 _S334 = dm_1;
        float _S335 = dif_factor_0(&_S334, _S333);
        k_10 = _S335;
    }
    else
    {
        k_10 = 1.0f;
    }
    float ft_1 = k_10 * _S327.y;
    float fc_1 = k_10 * _S327.z;
    float tau_1 = k_10 * _S327.w;
    if(p0_1 > fc_1)
    {
        damage_1 = true;
    }
    else
    {
        damage_1 = ring_1 > ft_1;
    }
    float _S336 = fc_1 * (3.14159274101257324f * _S316 * _S316);
    float _S337 = min(_S336 * _S325, 1.57079637050628662f * (2.0f * _S316 + _S325) * _S325 * _S325 * tau_1);
    float _S338 = 2.0f * _S337;
    float v_bl_1 = sqrt(_S338 / _S315.m_eff_0);
    bool perforate_1;
    if(damage_1)
    {
        perforate_1 = _S331 >= v_bl_1;
    }
    else
    {
        perforate_1 = false;
    }
    float r2_1 = _S316 + _S325;
    float _S339 = _S327.x;
    float plug_1 = _S339 * 3.14159274101257324f * _S325 / 3.0f * (_S316 * _S316 + _S316 * r2_1 + r2_1 * r2_1);
    float v_res_1;
    if(perforate_1)
    {
        v_res_1 = _S315.mass_0 / (_S315.mass_0 + plug_1) * sqrt(max(_S331 * _S331 - v_bl_1 * v_bl_1, 0.0f));
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
        depth_2 = min(0.5f * _S315.m_eff_0 * _S331 * _S331 / _S336, _S325);
    }
    else
    {
        depth_2 = 0.0f;
    }
    bool crater_1 = depth_2 > a_3;
    float c_5 = sqrt(e_target_1 / _S339);
    float z_t_1 = _S339 * c_5;
    float rho_p_1 = _S315.mass_0 / (4.18879032135009766f * _S316 * _S316 * _S316);
    float z_p_1 = rho_p_1 * sqrt(_S315.modulus_0 / rho_p_1);
    float mean_pressure_1 = force_1 / (3.14159274101257324f * a_3 * a_3);
    float _S340 = min(min(z_t_1 * _S331 * z_p_1 / (z_p_1 + z_t_1), mean_pressure_1), fc_1);
    float source_radius_1;
    if(mean_pressure_1 > fc_1)
    {
        source_radius_1 = max(a_3, _S316);
    }
    else
    {
        source_radius_1 = a_3;
    }
    float pulse_1 = c_5 * t_c_1;
    float back_1 = _S340 * min(source_radius_1 / _S325, 1.0f) * min(2.0f * _S325 / pulse_1, 1.0f);
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
        scab_2 = 0.5f * min(pulse_1, _S325);
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
    uint _S341;
    if(crater_1)
    {
        _S341 = 2U;
    }
    else
    {
        _S341 = 0U;
    }
    uint _S342 = region_4 | _S341;
    if(spall_1)
    {
        region_4 = 4U;
    }
    else
    {
        region_4 = 0U;
    }
    uint _S343 = _S342 | region_4;
    if(perforate_1)
    {
        region_4 = 8U;
    }
    else
    {
        region_4 = 0U;
    }
    uint outcome_8 = _S343 | region_4;
    uint _S344 = lp_body_0(b_20, kernelContext_52);
    *(kernelContext_52->lp_0+_S344) = packed_float4(float4(_S325, v_bl_1, v_res_1, (as_type<float>((outcome_8))))) ;
    packed_float4 device* _S345 = kernelContext_52->lp_0+(_S344 + 1U);
    if(crater_1)
    {
        k_10 = depth_2;
    }
    else
    {
        k_10 = 0.0f;
    }
    *_S345 = packed_float4(float4(k_10, scab_2, a_3, _S337)) ;
    if(perforate_1)
    {
        uint _S346 = a_counters_1(kernelContext_52);
        uint _S347 = atomic_fetch_add_explicit(kernelContext_52->atom_0+(_S346 + 10U), 1U, memory_order_relaxed);
    }
    else
    {
        if(spall_1)
        {
            uint _S348 = a_counters_1(kernelContext_52);
            uint _S349 = atomic_fetch_add_explicit(kernelContext_52->atom_0+(_S348 + 9U), 1U, memory_order_relaxed);
        }
        else
        {
            if(crater_1)
            {
                uint _S350 = a_counters_1(kernelContext_52);
                uint _S351 = atomic_fetch_add_explicit(kernelContext_52->atom_0+(_S350 + 8U), 1U, memory_order_relaxed);
            }
            else
            {
                uint _S352 = a_counters_1(kernelContext_52);
                uint _S353 = atomic_fetch_add_explicit(kernelContext_52->atom_0+(_S352 + 7U), 1U, memory_order_relaxed);
            }
        }
    }
    if(perforate_1)
    {
        if((kernelContext_52->params_0->fracture_work_0) != 0U)
        {
            il_impulse_0(b_20, _S315.body_0, _S315.mass_0, contact_13, d_14, _S315.m_eff_0 * (_S331 - sqrt(max(_S331 * _S331 - _S338 / _S315.m_eff_0, 0.0f))), kernelContext_52);
        }
    }
    else
    {
        if(damage_1)
        {
            il_impulse_0(b_20, _S315.body_0, _S315.mass_0, contact_13, d_14, _S315.m_eff_0 * _S331, kernelContext_52);
            float3 _S354 = to_rubble_0(d_14) * float3((_S331 * clamp(_S313, 0.0f, kernelContext_52->params_0->dt_0))) ;
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
                uint _S355 = 7U * (_S315.body_0 + region_4 * kernelContext_52->params_0->max_bodies_0);
                (*(kernelContext_52->bodies_0+_S355)).xyz = (float4(*(kernelContext_52->bodies_0+_S355)) ).xyz + _S354;
                region_4 = region_4 + 1U;
            }
        }
    }
    return;
}

void il_extend_1(uint k_11, uint from_1, KernelContext_0 thread* kernelContext_53)
{
    if(k_11 >= (kernelContext_53->params_0->chunk_count_0))
    {
        return;
    }
    uint _S356 = body_of_0(k_11, kernelContext_53);
    bool _S357 = il_struck_1(_S356, kernelContext_53);
    if(!_S357)
    {
        return;
    }
    thread float3 contact_14;
    thread float3 d_15;
    thread float speed_12;
    il_path_1(_S356, &contact_14, &d_15, &speed_12, kernelContext_53);
    IlStriker_0 _S358 = il_striker_1(_S356, kernelContext_53);
    uint _S359 = a_il_0(_S356, kernelContext_53);
    uint _S360 = atomic_load_explicit(kernelContext_53->atom_0+(_S359 + from_1), memory_order_relaxed);
    float end_2 = (as_type<float>((_S360)));
    uint to_1 = _S359 + 1U - from_1;
    uint _S361 = atomic_fetch_max_explicit(kernelContext_53->atom_0+to_1, (as_type<uint>((end_2))), memory_order_relaxed);
    float2 _S362 = il_interval_0(k_11, contact_14, d_15, _S358.radius_0, kernelContext_53);
    float _S363 = _S362.x;
    float _S364 = _S362.y;
    bool _S365;
    if(_S363 <= _S364)
    {
        _S365 = _S363 <= (end_2 + 0.00100000004749745f + 0.01999999955296516f * _S358.radius_0);
    }
    else
    {
        _S365 = false;
    }
    if(_S365)
    {
        _S365 = _S364 > end_2;
    }
    else
    {
        _S365 = false;
    }
    if(_S365)
    {
        uint _S366 = atomic_fetch_max_explicit(kernelContext_53->atom_0+to_1, (as_type<uint>((_S364))), memory_order_relaxed);
    }
    return;
}

uint a_first_1(uint shot_3, KernelContext_0 thread* kernelContext_54)
{
    uint _S367 = a_strike_0(kernelContext_54);
    return _S367 + 27U * kernelContext_54->params_0->body_count_0 + shot_3;
}

void il_begin_1(uint b_21, KernelContext_0 thread* kernelContext_55)
{
    bool _S368;
    if(b_21 >= (kernelContext_55->params_0->body_count_0))
    {
        _S368 = true;
    }
    else
    {
        bool _S369 = il_struck_1(b_21, kernelContext_55);
        _S368 = !_S369;
    }
    if(_S368)
    {
        return;
    }
    thread uint shot_4;
    bool _S370 = struck_0(b_21, &shot_4, kernelContext_55);
    uint _S371 = a_strike_0(kernelContext_55);
    uint _S372 = atomic_load_explicit(kernelContext_55->atom_0+(_S371 + 9U * b_21 + 6U), memory_order_relaxed);
    uint _S373 = a_first_1(shot_4, kernelContext_55);
    uint _S374 = atomic_load_explicit(kernelContext_55->atom_0+_S373, memory_order_relaxed);
    if(_S372 != _S374)
    {
        uint _S375 = strike_at_0(b_21, kernelContext_55);
        (*(kernelContext_55->frames_0+_S375))[int(3)] = 0.0f;
        return;
    }
    thread float3 contact_15;
    thread float3 d_16;
    thread float speed_13;
    il_path_1(b_21, &contact_15, &d_16, &speed_13, kernelContext_55);
    uint _S376 = il_entry_1(b_21, kernelContext_55);
    IlStriker_0 _S377 = il_striker_1(b_21, kernelContext_55);
    float2 _S378 = il_interval_0(_S376, contact_15, d_16, _S377.radius_0, kernelContext_55);
    uint end_3 = (as_type<uint>((max(_S378.y, 0.0f))));
    uint _S379 = a_il_0(b_21, kernelContext_55);
    atomic_store_explicit(kernelContext_55->atom_0+_S379, end_3, memory_order_relaxed);
    atomic_store_explicit(kernelContext_55->atom_0+(_S379 + 1U), end_3, memory_order_relaxed);
    atomic_store_explicit(kernelContext_55->atom_0+(_S379 + 2U), 0U, memory_order_relaxed);
    atomic_store_explicit(kernelContext_55->atom_0+(_S379 + 3U), 0U, memory_order_relaxed);
    return;
}

uint l_pool_0(uint half_0, KernelContext_0 thread* kernelContext_56)
{
    return kernelContext_56->params_0->chunk_count_0 + 3U * kernelContext_56->params_0->pool_cap_0 * half_0;
}

float pulse_mean_0(float start_0, float plateau_0, float plateau_time_0, float peak_0, float duration_0, float dt_1)
{
    float _S380 = start_0 + plateau_time_0;
    float sum_0 = plateau_0 * max(min(_S380, dt_1) - max(start_0, 0.0f), 0.0f);
    float _S381 = max(0.0f, _S380);
    float _S382 = min(dt_1, _S380 + duration_0);
    bool _S383;
    if(_S382 > _S381)
    {
        _S383 = duration_0 > 0.0f;
    }
    else
    {
        _S383 = false;
    }
    float sum_1;
    if(_S383)
    {
        float w_1 = 3.14159274101257324f / duration_0;
        sum_1 = sum_0 + peak_0 / w_1 * (cos(w_1 * (_S381 - _S380)) - cos(w_1 * (_S382 - _S380)));
    }
    else
    {
        sum_1 = sum_0;
    }
    return sum_1 / dt_1;
}

float hertz_duration_0(float m_0, float r_8, float e_1, float v_4)
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_8 * e_1 * e_1 * max(abs(v_4), 0.00100000004749745f)), 0.20000000298023224f);
}

uint l_vec_0(KernelContext_0 thread* kernelContext_57)
{
    return kernelContext_57->params_0->chunk_count_0 + 6U * kernelContext_57->params_0->pool_cap_0;
}

[[kernel]] void fr_loads(uint3 id_0 [[thread_position_in_grid]], FractureParams_0 constant* params_1 [[buffer(0)]], uint device* work_1 [[buffer(4)]], packed_float4 device* frames_1 [[buffer(6)]], packed_float4 device* bodies_1 [[buffer(7)]], packed_float4 device* chunks_1 [[buffer(1)]], uint device* sort_1 [[buffer(11)]], atomic<uint> device* atom_1 [[buffer(5)]], Record_natural_0 device* manifolds_1 [[buffer(10)]], packed_float4 device* loads_1 [[buffer(12)]], packed_uint4 device* bonds_1 [[buffer(2)]], uint device* args_1 [[buffer(13)]], uint device* info_1 [[buffer(15)]], ScanParams_0 constant* scan_1 [[buffer(14)]], packed_float4 device* parts_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(8)]], packed_uint4 device* collider_info_1 [[buffer(9)]], packed_float4 device* view_chunks_1 [[buffer(17)]], ViewParams_0 constant* view_1 [[buffer(16)]], packed_float4 device* objects_1 [[buffer(18)]], uint device* shot_bodies_1 [[buffer(20)]], float device* lines_1 [[buffer(19)]], packed_float4 device* geometry_1 [[buffer(21)]], packed_float4 device* shots_1 [[buffer(22)]], packed_float4 device* lp_1 [[buffer(24)]], packed_float4 device* ext_2 [[buffer(23)]], uint device* engine_counters_1 [[buffer(26)]], packed_float4 device* engine_contacts_1 [[buffer(25)]], packed_float4 device* cs_anchors_1 [[buffer(27)]])
{
    float3 f_total_0;
    float3 t_total_0;
    float m_1;
    uint p_5;
    uint kept_0;
    uint q3_0;
    bool overflow_0;
    bool overflow_1;
    float3 f_total_1;
    float3 t_total_1;
    thread KernelContext_0 kernelContext_58;
    (&kernelContext_58)->params_0 = params_1;
    (&kernelContext_58)->work_0 = work_1;
    (&kernelContext_58)->frames_0 = frames_1;
    (&kernelContext_58)->bodies_0 = bodies_1;
    (&kernelContext_58)->chunks_0 = chunks_1;
    (&kernelContext_58)->sort_0 = sort_1;
    (&kernelContext_58)->atom_0 = atom_1;
    (&kernelContext_58)->manifolds_0 = manifolds_1;
    (&kernelContext_58)->loads_0 = loads_1;
    (&kernelContext_58)->bonds_0 = bonds_1;
    (&kernelContext_58)->args_0 = args_1;
    (&kernelContext_58)->info_0 = info_1;
    (&kernelContext_58)->scan_0 = scan_1;
    (&kernelContext_58)->parts_0 = parts_1;
    (&kernelContext_58)->colliders_0 = colliders_1;
    (&kernelContext_58)->collider_info_0 = collider_info_1;
    (&kernelContext_58)->view_chunks_0 = view_chunks_1;
    (&kernelContext_58)->view_0 = view_1;
    (&kernelContext_58)->objects_0 = objects_1;
    (&kernelContext_58)->shot_bodies_0 = shot_bodies_1;
    (&kernelContext_58)->lines_0 = lines_1;
    (&kernelContext_58)->geometry_0 = geometry_1;
    (&kernelContext_58)->shots_0 = shots_1;
    (&kernelContext_58)->lp_0 = lp_1;
    (&kernelContext_58)->ext_0 = ext_2;
    (&kernelContext_58)->engine_counters_0 = engine_counters_1;
    (&kernelContext_58)->engine_contacts_0 = engine_contacts_1;
    (&kernelContext_58)->cs_anchors_0 = cs_anchors_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_58)->g_scan_0 = &g_scan_1;
    uint k_12 = id_0.x;
    if(k_12 >= (params_1->chunk_count_0))
    {
        return;
    }
    uint _S384 = body_of_0(k_12, &kernelContext_58);
    uint b_22 = 7U * _S384;
    thread Quat_0 q_10;
    thread float3 origin_3;
    structure_pose_0(_S384, b_22, &q_10, &origin_3, &kernelContext_58);
    uint _S385 = 4U * k_12;
    float4 _S386 = float4(*((&kernelContext_58)->chunks_0+_S385)) ;
    float3 _S387 = _S386.xyz;
    thread Quat_0 _S388 = q_10;
    float3 _S389 = rotate_0(&_S388, _S387);
    float3 centre_0 = origin_3 + _S389;
    float inv_mass_2 = (float4(*((&kernelContext_58)->bodies_0+b_22)) ).w;
    float m_body_0;
    if(inv_mass_2 > 0.0f)
    {
        m_body_0 = 1.0f / inv_mass_2;
    }
    else
    {
        m_body_0 = float_bits_0(2139095040U);
    }
    uint from_2 = (&kernelContext_58)->params_0->parity_0;
    uint to_2 = 1U - (&kernelContext_58)->params_0->parity_0;
    uint _S390 = w_pulses_0(&kernelContext_58);
    uint _S391 = _S390 + _S385;
    uint _S392 = _S391 + 2U * from_2;
    uint off_0 = *((&kernelContext_58)->work_0+_S392);
    uint device* _S393 = (&kernelContext_58)->work_0+(_S392 + 1U);
    uint carried_0 = *_S393;
    uint room_0 = *_S393 + (&kernelContext_58)->sort_0[params_1->chunk_count_0 + 1U + k_12];
    uint base_0;
    uint room_1;
    if(room_0 != 0U)
    {
        uint _S394 = a_counters_1(&kernelContext_58);
        uint base_1 = atomic_fetch_add_explicit((&kernelContext_58)->atom_0+(_S394 + 5U + to_2), room_0, memory_order_relaxed);
        if((base_1 + room_0) > ((&kernelContext_58)->params_0->pool_cap_0))
        {
            uint _S395 = atomic_fetch_or_explicit((&kernelContext_58)->atom_0+(_S394 + 3U), 8U, memory_order_relaxed);
            if(base_1 < ((&kernelContext_58)->params_0->pool_cap_0))
            {
                base_0 = (&kernelContext_58)->params_0->pool_cap_0 - base_1;
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
        uint _S396 = base_0;
        base_0 = base_1;
        room_1 = _S396;
    }
    else
    {
        base_0 = 0U;
        room_1 = room_0;
    }
    uint _S397 = params_1->chunk_count_0;
    float _S398 = _S386.w;
    float4 _S399 = float4(*((&kernelContext_58)->chunks_0+(_S385 + 3U))) ;
    float _S400 = (&kernelContext_58)->params_0->band_0;
    float _S401 = (&kernelContext_58)->params_0->dt_0;
    bool _S402 = ((&kernelContext_58)->params_0->impact_peak_0) != 0U;
    uint _S403 = l_pool_0(from_2, &kernelContext_58);
    uint _S404 = _S403 + 3U * off_0;
    uint _S405 = l_pool_0(to_2, &kernelContext_58);
    uint _S406 = _S405 + 3U * base_0;
    for(;;)
    {
        float3 _S407 = float3(0.0f) ;
        uint _S408 = (&kernelContext_58)->sort_0[k_12];
        uint _S409 = (&kernelContext_58)->sort_0[_S397 + 1U + k_12];
        bool first_0 = true;
        uint last_0 = 0U;
        uint step_1 = 0U;
        f_total_0 = _S407;
        t_total_0 = _S407;
        for(;;)
        {
            if(step_1 < _S409)
            {
            }
            else
            {
                break;
            }
            p_5 = 4294967295U;
            kept_0 = 0U;
            q3_0 = 0U;
            for(;;)
            {
                if(q3_0 < _S409)
                {
                }
                else
                {
                    break;
                }
                uint key_1 = (uint4((&kernelContext_58)->manifolds_0[_S408 + q3_0].ids_0) ).y + 1U;
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
                    if(p_5 == 4294967295U)
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
                    p_5 = q3_0;
                    kept_0 = key_1;
                }
                q3_0 = q3_0 + 1U;
            }
            if(p_5 == 4294967295U)
            {
                break;
            }
            Record_natural_0 r_9 = (&kernelContext_58)->manifolds_0[_S408 + p_5];
            float4 _S410 = float4(r_9.impulse_0) ;
            float3 _S411 = _S410.xyz;
            float j_1 = length(_S411);
            if(j_1 == 0.0f)
            {
                uint _S412 = step_1 + 1U;
                first_0 = false;
                last_0 = kept_0;
                step_1 = _S412;
                continue;
            }
            float m_other_0 = (float4(r_9.other_0) ).x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_0))
                {
                    m_1 = _S398;
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
            float _S413 = min(j_1, 2.0f * m_1 * max(_S410.w - _S400, 0.0f));
            if(_S413 < j_1)
            {
                float3 f_1 = _S411 / float3(j_1)  * float3(((j_1 - _S413) / _S401)) ;
                float3 _S414 = t_total_0 + cross((float4(r_9.point_0) ).xyz - centre_0, f_1);
                f_total_1 = f_total_0 + f_1;
                t_total_1 = _S414;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            f_total_0 = f_total_1;
            t_total_0 = t_total_1;
            uint _S412 = step_1 + 1U;
            first_0 = false;
            last_0 = kept_0;
            step_1 = _S412;
        }
        p_5 = 0U;
        kept_0 = 0U;
        for(;;)
        {
            if(p_5 < carried_0)
            {
            }
            else
            {
                break;
            }
            uint _S415 = _S404 + 3U * p_5;
            float4 _S416 = float4(*((&kernelContext_58)->loads_0+_S415)) ;
            thread float4 a_4 = _S416;
            float4 _S417 = float4(*((&kernelContext_58)->loads_0+(_S415 + 1U))) ;
            float4 _S418 = float4(*((&kernelContext_58)->loads_0+(_S415 + 2U))) ;
            float _S419 = _S417.w;
            float mean_0 = pulse_mean_0(_S416.x, _S416.y, _S416.z, _S416.w, _S419, _S401);
            if(mean_0 != 0.0f)
            {
                float3 f_2 = _S418.xyz * float3(mean_0) ;
                float3 _S420 = t_total_0 + cross(_S417.xyz - centre_0, f_2);
                f_total_0 = f_total_0 + f_2;
                t_total_0 = _S420;
            }
            if((a_4.x + a_4.z + _S419) > _S401)
            {
                a_4.x = a_4.x - _S401;
                uint _S421 = _S406 + 3U * kept_0;
                *((&kernelContext_58)->loads_0+_S421) = packed_float4(a_4) ;
                *((&kernelContext_58)->loads_0+(_S421 + 1U)) = packed_float4(_S417) ;
                *((&kernelContext_58)->loads_0+(_S421 + 2U)) = packed_float4(_S418) ;
                kept_0 = kept_0 + 1U;
            }
            p_5 = p_5 + 1U;
        }
        q3_0 = 0U;
        overflow_0 = false;
        for(;;)
        {
            if(q3_0 < _S409)
            {
            }
            else
            {
                break;
            }
            Record_natural_0 r_10 = (&kernelContext_58)->manifolds_0[_S408 + q3_0];
            float4 _S422 = float4(r_10.impulse_0) ;
            float3 _S423 = _S422.xyz;
            float j_2 = length(_S423);
            if(j_2 == 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float4 _S424 = float4(r_10.other_0) ;
            float m_other_1 = _S424.x;
            if(isinf(m_body_0))
            {
                if(isinf(m_other_1))
                {
                    m_1 = _S398;
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
            float _S425 = _S422.w;
            float _S426 = min(j_2, 2.0f * m_1 * max(_S425 - _S400, 0.0f));
            if(_S426 <= 0.0f)
            {
                overflow_1 = overflow_0;
                q3_0 = q3_0 + 1U;
                overflow_0 = overflow_1;
                continue;
            }
            float _S427 = _S399.y;
            float duration_1 = hertz_duration_0(m_1, min(_S424.z, _S399.z), 1.0f / ((1.0f - _S427 * _S427) / _S399.x + 1.0f / _S424.y), _S425);
            float peak_1 = 3.14159274101257324f * _S426 / (2.0f * duration_1);
            float3 dir_0 = _S423 / float3(j_2) ;
            float mean_1;
            if(_S402)
            {
                mean_1 = peak_1;
            }
            else
            {
                mean_1 = pulse_mean_0(0.0f, 0.0f, 0.0f, peak_1, duration_1, _S401);
            }
            if(mean_1 != 0.0f)
            {
                float3 f_3 = dir_0 * float3(mean_1) ;
                float3 _S428 = t_total_0 + cross((float4(r_10.point_0) ).xyz - centre_0, f_3);
                f_total_1 = f_total_0 + f_3;
                t_total_1 = _S428;
            }
            else
            {
                f_total_1 = f_total_0;
                t_total_1 = t_total_0;
            }
            uint kept_1;
            if(duration_1 > _S401)
            {
                if(kept_0 < room_1)
                {
                    uint _S429 = _S406 + 3U * kept_0;
                    *((&kernelContext_58)->loads_0+_S429) = packed_float4(float4(- _S401, 0.0f, 0.0f, peak_1)) ;
                    *((&kernelContext_58)->loads_0+(_S429 + 1U)) = packed_float4(float4((float4(r_10.point_0) ).xyz, duration_1)) ;
                    *((&kernelContext_58)->loads_0+(_S429 + 2U)) = packed_float4(float4(dir_0, 0.0f)) ;
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
        uint _S430 = a_counters_1(&kernelContext_58);
        uint _S431 = atomic_fetch_or_explicit((&kernelContext_58)->atom_0+(_S430 + 3U), 8U, memory_order_relaxed);
    }
    uint _S432 = _S391 + 2U * to_2;
    *((&kernelContext_58)->work_0+_S432) = base_0;
    *((&kernelContext_58)->work_0+(_S432 + 1U)) = kept_0;
    *((&kernelContext_58)->loads_0+k_12) = packed_float4(float4(length(f_total_0), 0.0f, 0.0f, 0.0f)) ;
    if(((&kernelContext_58)->params_0->load_model_0) == 1U)
    {
        uint _S433 = l_vec_0(&kernelContext_58);
        uint _S434 = _S433 + 2U * k_12;
        *((&kernelContext_58)->loads_0+_S434) = packed_float4(float4(f_total_0, 0.0f)) ;
        *((&kernelContext_58)->loads_0+(_S434 + 1U)) = packed_float4(float4(t_total_0, 0.0f)) ;
    }
    return;
}

[[kernel]] void fr_breaks(uint3 id_1 [[thread_position_in_grid]], FractureParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(4)]], packed_float4 device* frames_2 [[buffer(6)]], packed_float4 device* bodies_2 [[buffer(7)]], packed_float4 device* chunks_2 [[buffer(1)]], uint device* sort_2 [[buffer(11)]], atomic<uint> device* atom_2 [[buffer(5)]], Record_natural_0 device* manifolds_2 [[buffer(10)]], packed_float4 device* loads_2 [[buffer(12)]], packed_uint4 device* bonds_2 [[buffer(2)]], uint device* args_2 [[buffer(13)]], uint device* info_2 [[buffer(15)]], ScanParams_0 constant* scan_2 [[buffer(14)]], packed_float4 device* parts_2 [[buffer(3)]], packed_float4 device* colliders_2 [[buffer(8)]], packed_uint4 device* collider_info_2 [[buffer(9)]], packed_float4 device* view_chunks_2 [[buffer(17)]], ViewParams_0 constant* view_2 [[buffer(16)]], packed_float4 device* objects_2 [[buffer(18)]], uint device* shot_bodies_2 [[buffer(20)]], float device* lines_2 [[buffer(19)]], packed_float4 device* geometry_2 [[buffer(21)]], packed_float4 device* shots_2 [[buffer(22)]], packed_float4 device* lp_2 [[buffer(24)]], packed_float4 device* ext_3 [[buffer(23)]], uint device* engine_counters_2 [[buffer(26)]], packed_float4 device* engine_contacts_2 [[buffer(25)]], packed_float4 device* cs_anchors_2 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_59;
    (&kernelContext_59)->params_0 = params_2;
    (&kernelContext_59)->work_0 = work_2;
    (&kernelContext_59)->frames_0 = frames_2;
    (&kernelContext_59)->bodies_0 = bodies_2;
    (&kernelContext_59)->chunks_0 = chunks_2;
    (&kernelContext_59)->sort_0 = sort_2;
    (&kernelContext_59)->atom_0 = atom_2;
    (&kernelContext_59)->manifolds_0 = manifolds_2;
    (&kernelContext_59)->loads_0 = loads_2;
    (&kernelContext_59)->bonds_0 = bonds_2;
    (&kernelContext_59)->args_0 = args_2;
    (&kernelContext_59)->info_0 = info_2;
    (&kernelContext_59)->scan_0 = scan_2;
    (&kernelContext_59)->parts_0 = parts_2;
    (&kernelContext_59)->colliders_0 = colliders_2;
    (&kernelContext_59)->collider_info_0 = collider_info_2;
    (&kernelContext_59)->view_chunks_0 = view_chunks_2;
    (&kernelContext_59)->view_0 = view_2;
    (&kernelContext_59)->objects_0 = objects_2;
    (&kernelContext_59)->shot_bodies_0 = shot_bodies_2;
    (&kernelContext_59)->lines_0 = lines_2;
    (&kernelContext_59)->geometry_0 = geometry_2;
    (&kernelContext_59)->shots_0 = shots_2;
    (&kernelContext_59)->lp_0 = lp_2;
    (&kernelContext_59)->ext_0 = ext_3;
    (&kernelContext_59)->engine_counters_0 = engine_counters_2;
    (&kernelContext_59)->engine_contacts_0 = engine_contacts_2;
    (&kernelContext_59)->cs_anchors_0 = cs_anchors_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_59)->g_scan_0 = &g_scan_2;
    uint i_2 = id_1.x;
    bool _S435;
    if(i_2 >= (params_2->bond_count_0))
    {
        _S435 = true;
    }
    else
    {
        uint _S436 = w_alive_0(&kernelContext_59);
        _S435 = (*((&kernelContext_59)->work_0+(_S436 + i_2))) == 0U;
    }
    if(_S435)
    {
        return;
    }
    uint4 _S437 = uint4(*((&kernelContext_59)->bonds_0+i_2)) ;
    uint _S438 = _S437.x;
    if((max((float4(*((&kernelContext_59)->loads_0+_S438)) ).x, (float4(*((&kernelContext_59)->loads_0+_S437.y)) ).x)) > (as_type<float>((_S437.z))))
    {
        uint _S439 = w_alive_0(&kernelContext_59);
        *((&kernelContext_59)->work_0+(_S439 + i_2)) = 0U;
        uint _S440 = a_dirty_0(&kernelContext_59);
        uint _S441 = body_of_0(_S438, &kernelContext_59);
        uint _S442 = atomic_fetch_or_explicit((&kernelContext_59)->atom_0+(_S440 + _S441), 1U, memory_order_relaxed);
        uint _S443 = a_counters_1(&kernelContext_59);
        uint _S444 = atomic_fetch_add_explicit((&kernelContext_59)->atom_0+_S443, 1U, memory_order_relaxed);
        uint _S445 = atomic_fetch_add_explicit((&kernelContext_59)->atom_0+(_S443 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint groups_of_0(uint n_2, uint size_0)
{
    uint _S446 = (n_2 + size_0 - 1U) / size_0;
    return _S446;
}

void set_args_0(uint at_3, uint x_3, KernelContext_0 thread* kernelContext_60)
{
    uint _S447 = 3U * at_3;
    *(kernelContext_60->args_0+_S447) = x_3;
    *(kernelContext_60->args_0+(_S447 + 1U)) = 1U;
    *(kernelContext_60->args_0+(_S447 + 2U)) = 1U;
    return;
}

[[kernel]] void fr_gate(uint3 id_2 [[thread_position_in_grid]], FractureParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(4)]], packed_float4 device* frames_3 [[buffer(6)]], packed_float4 device* bodies_3 [[buffer(7)]], packed_float4 device* chunks_3 [[buffer(1)]], uint device* sort_3 [[buffer(11)]], atomic<uint> device* atom_3 [[buffer(5)]], Record_natural_0 device* manifolds_3 [[buffer(10)]], packed_float4 device* loads_3 [[buffer(12)]], packed_uint4 device* bonds_3 [[buffer(2)]], uint device* args_3 [[buffer(13)]], uint device* info_3 [[buffer(15)]], ScanParams_0 constant* scan_3 [[buffer(14)]], packed_float4 device* parts_3 [[buffer(3)]], packed_float4 device* colliders_3 [[buffer(8)]], packed_uint4 device* collider_info_3 [[buffer(9)]], packed_float4 device* view_chunks_3 [[buffer(17)]], ViewParams_0 constant* view_3 [[buffer(16)]], packed_float4 device* objects_3 [[buffer(18)]], uint device* shot_bodies_3 [[buffer(20)]], float device* lines_3 [[buffer(19)]], packed_float4 device* geometry_3 [[buffer(21)]], packed_float4 device* shots_3 [[buffer(22)]], packed_float4 device* lp_3 [[buffer(24)]], packed_float4 device* ext_4 [[buffer(23)]], uint device* engine_counters_3 [[buffer(26)]], packed_float4 device* engine_contacts_3 [[buffer(25)]], packed_float4 device* cs_anchors_3 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_61;
    (&kernelContext_61)->params_0 = params_3;
    (&kernelContext_61)->work_0 = work_3;
    (&kernelContext_61)->frames_0 = frames_3;
    (&kernelContext_61)->bodies_0 = bodies_3;
    (&kernelContext_61)->chunks_0 = chunks_3;
    (&kernelContext_61)->sort_0 = sort_3;
    (&kernelContext_61)->atom_0 = atom_3;
    (&kernelContext_61)->manifolds_0 = manifolds_3;
    (&kernelContext_61)->loads_0 = loads_3;
    (&kernelContext_61)->bonds_0 = bonds_3;
    (&kernelContext_61)->args_0 = args_3;
    (&kernelContext_61)->info_0 = info_3;
    (&kernelContext_61)->scan_0 = scan_3;
    (&kernelContext_61)->parts_0 = parts_3;
    (&kernelContext_61)->colliders_0 = colliders_3;
    (&kernelContext_61)->collider_info_0 = collider_info_3;
    (&kernelContext_61)->view_chunks_0 = view_chunks_3;
    (&kernelContext_61)->view_0 = view_3;
    (&kernelContext_61)->objects_0 = objects_3;
    (&kernelContext_61)->shot_bodies_0 = shot_bodies_3;
    (&kernelContext_61)->lines_0 = lines_3;
    (&kernelContext_61)->geometry_0 = geometry_3;
    (&kernelContext_61)->shots_0 = shots_3;
    (&kernelContext_61)->lp_0 = lp_3;
    (&kernelContext_61)->ext_0 = ext_4;
    (&kernelContext_61)->engine_counters_0 = engine_counters_3;
    (&kernelContext_61)->engine_contacts_0 = engine_contacts_3;
    (&kernelContext_61)->cs_anchors_0 = cs_anchors_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_61)->g_scan_0 = &g_scan_3;
    uint _S448 = a_counters_1(&kernelContext_61);
    uint _S449 = atomic_load_explicit((&kernelContext_61)->atom_0+_S448, memory_order_relaxed);
    bool run_0 = _S449 != 0U;
    atomic_store_explicit((&kernelContext_61)->atom_0+_S448, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_61)->atom_0+(_S448 + 5U + (&kernelContext_61)->params_0->parity_0), 0U, memory_order_relaxed);
    if(run_0)
    {
        uint _S450 = atomic_fetch_add_explicit((&kernelContext_61)->atom_0+(_S448 + 4U), 1U, memory_order_relaxed);
    }
    uint _S451;
    if(run_0)
    {
        uint _S452 = groups_of_0((&kernelContext_61)->params_0->chunk_count_0, 64U);
        _S451 = _S452;
    }
    else
    {
        _S451 = 0U;
    }
    set_args_0(0U, _S451, &kernelContext_61);
    if(run_0)
    {
        uint _S453 = groups_of_0((&kernelContext_61)->params_0->bond_count_0, 64U);
        _S451 = _S453;
    }
    else
    {
        _S451 = 0U;
    }
    set_args_0(1U, _S451, &kernelContext_61);
    if(run_0)
    {
        uint _S454 = groups_of_0((&kernelContext_61)->params_0->body_count_0, 64U);
        _S451 = _S454;
    }
    else
    {
        _S451 = 0U;
    }
    set_args_0(2U, _S451, &kernelContext_61);
    if(run_0)
    {
        uint _S455 = groups_of_0((&kernelContext_61)->params_0->chunk_count_0, 1024U);
        _S451 = _S455;
    }
    else
    {
        _S451 = 0U;
    }
    set_args_0(3U, _S451, &kernelContext_61);
    if(run_0)
    {
        uint _S456 = groups_of_0((&kernelContext_61)->params_0->body_count_0, 1024U);
        _S451 = _S456;
    }
    else
    {
        _S451 = 0U;
    }
    set_args_0(4U, _S451, &kernelContext_61);
    int _S457;
    if(run_0)
    {
        _S457 = int(1);
    }
    else
    {
        _S457 = int(0);
    }
    set_args_0(5U, uint(_S457), &kernelContext_61);
    return;
}

bool dirty_0(uint body_1, KernelContext_0 thread* kernelContext_62)
{
    uint _S458 = a_dirty_0(kernelContext_62);
    uint _S459 = atomic_load_explicit(kernelContext_62->atom_0+(_S458 + body_1), memory_order_relaxed);
    return _S459 != 0U;
}

uint a_parent_0()
{
    return 0U;
}

uint a_count_0(KernelContext_0 thread* kernelContext_63)
{
    return kernelContext_63->params_0->chunk_count_0;
}

uint a_anchor_0(KernelContext_0 thread* kernelContext_64)
{
    return 2U * kernelContext_64->params_0->chunk_count_0;
}

[[kernel]] void fr_uf_init(uint3 id_3 [[thread_position_in_grid]], FractureParams_0 constant* params_4 [[buffer(0)]], uint device* work_4 [[buffer(4)]], packed_float4 device* frames_4 [[buffer(6)]], packed_float4 device* bodies_4 [[buffer(7)]], packed_float4 device* chunks_4 [[buffer(1)]], uint device* sort_4 [[buffer(11)]], atomic<uint> device* atom_4 [[buffer(5)]], Record_natural_0 device* manifolds_4 [[buffer(10)]], packed_float4 device* loads_4 [[buffer(12)]], packed_uint4 device* bonds_4 [[buffer(2)]], uint device* args_4 [[buffer(13)]], uint device* info_4 [[buffer(15)]], ScanParams_0 constant* scan_4 [[buffer(14)]], packed_float4 device* parts_4 [[buffer(3)]], packed_float4 device* colliders_4 [[buffer(8)]], packed_uint4 device* collider_info_4 [[buffer(9)]], packed_float4 device* view_chunks_4 [[buffer(17)]], ViewParams_0 constant* view_4 [[buffer(16)]], packed_float4 device* objects_4 [[buffer(18)]], uint device* shot_bodies_4 [[buffer(20)]], float device* lines_4 [[buffer(19)]], packed_float4 device* geometry_4 [[buffer(21)]], packed_float4 device* shots_4 [[buffer(22)]], packed_float4 device* lp_4 [[buffer(24)]], packed_float4 device* ext_5 [[buffer(23)]], uint device* engine_counters_4 [[buffer(26)]], packed_float4 device* engine_contacts_4 [[buffer(25)]], packed_float4 device* cs_anchors_4 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_65;
    (&kernelContext_65)->params_0 = params_4;
    (&kernelContext_65)->work_0 = work_4;
    (&kernelContext_65)->frames_0 = frames_4;
    (&kernelContext_65)->bodies_0 = bodies_4;
    (&kernelContext_65)->chunks_0 = chunks_4;
    (&kernelContext_65)->sort_0 = sort_4;
    (&kernelContext_65)->atom_0 = atom_4;
    (&kernelContext_65)->manifolds_0 = manifolds_4;
    (&kernelContext_65)->loads_0 = loads_4;
    (&kernelContext_65)->bonds_0 = bonds_4;
    (&kernelContext_65)->args_0 = args_4;
    (&kernelContext_65)->info_0 = info_4;
    (&kernelContext_65)->scan_0 = scan_4;
    (&kernelContext_65)->parts_0 = parts_4;
    (&kernelContext_65)->colliders_0 = colliders_4;
    (&kernelContext_65)->collider_info_0 = collider_info_4;
    (&kernelContext_65)->view_chunks_0 = view_chunks_4;
    (&kernelContext_65)->view_0 = view_4;
    (&kernelContext_65)->objects_0 = objects_4;
    (&kernelContext_65)->shot_bodies_0 = shot_bodies_4;
    (&kernelContext_65)->lines_0 = lines_4;
    (&kernelContext_65)->geometry_0 = geometry_4;
    (&kernelContext_65)->shots_0 = shots_4;
    (&kernelContext_65)->lp_0 = lp_4;
    (&kernelContext_65)->ext_0 = ext_5;
    (&kernelContext_65)->engine_counters_0 = engine_counters_4;
    (&kernelContext_65)->engine_contacts_0 = engine_contacts_4;
    (&kernelContext_65)->cs_anchors_0 = cs_anchors_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_65)->g_scan_0 = &g_scan_4;
    uint c_6 = id_3.x;
    bool _S460;
    if(c_6 >= (params_4->chunk_count_0))
    {
        _S460 = true;
    }
    else
    {
        uint _S461 = body_of_0(c_6, &kernelContext_65);
        bool _S462 = dirty_0(_S461, &kernelContext_65);
        _S460 = !_S462;
    }
    if(_S460)
    {
        return;
    }
    atomic_store_explicit((&kernelContext_65)->atom_0+(a_parent_0() + c_6), c_6, memory_order_relaxed);
    uint _S463 = a_count_0(&kernelContext_65);
    atomic_store_explicit((&kernelContext_65)->atom_0+(_S463 + c_6), 0U, memory_order_relaxed);
    uint _S464 = a_anchor_0(&kernelContext_65);
    atomic_store_explicit((&kernelContext_65)->atom_0+(_S464 + c_6), 0U, memory_order_relaxed);
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
        uint _S465 = a_sums_0(&kernelContext_65);
        atomic_store_explicit((&kernelContext_65)->atom_0+(_S465 + 20U * c_6 + w_2), 0U, memory_order_relaxed);
        w_2 = w_2 + 1U;
    }
    return;
}

[[kernel]] void fr_body_init(uint3 id_4 [[thread_position_in_grid]], FractureParams_0 constant* params_5 [[buffer(0)]], uint device* work_5 [[buffer(4)]], packed_float4 device* frames_5 [[buffer(6)]], packed_float4 device* bodies_5 [[buffer(7)]], packed_float4 device* chunks_5 [[buffer(1)]], uint device* sort_5 [[buffer(11)]], atomic<uint> device* atom_5 [[buffer(5)]], Record_natural_0 device* manifolds_5 [[buffer(10)]], packed_float4 device* loads_5 [[buffer(12)]], packed_uint4 device* bonds_5 [[buffer(2)]], uint device* args_5 [[buffer(13)]], uint device* info_5 [[buffer(15)]], ScanParams_0 constant* scan_5 [[buffer(14)]], packed_float4 device* parts_5 [[buffer(3)]], packed_float4 device* colliders_5 [[buffer(8)]], packed_uint4 device* collider_info_5 [[buffer(9)]], packed_float4 device* view_chunks_5 [[buffer(17)]], ViewParams_0 constant* view_5 [[buffer(16)]], packed_float4 device* objects_5 [[buffer(18)]], uint device* shot_bodies_5 [[buffer(20)]], float device* lines_5 [[buffer(19)]], packed_float4 device* geometry_5 [[buffer(21)]], packed_float4 device* shots_5 [[buffer(22)]], packed_float4 device* lp_5 [[buffer(24)]], packed_float4 device* ext_6 [[buffer(23)]], uint device* engine_counters_5 [[buffer(26)]], packed_float4 device* engine_contacts_5 [[buffer(25)]], packed_float4 device* cs_anchors_5 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_66;
    (&kernelContext_66)->params_0 = params_5;
    (&kernelContext_66)->work_0 = work_5;
    (&kernelContext_66)->frames_0 = frames_5;
    (&kernelContext_66)->bodies_0 = bodies_5;
    (&kernelContext_66)->chunks_0 = chunks_5;
    (&kernelContext_66)->sort_0 = sort_5;
    (&kernelContext_66)->atom_0 = atom_5;
    (&kernelContext_66)->manifolds_0 = manifolds_5;
    (&kernelContext_66)->loads_0 = loads_5;
    (&kernelContext_66)->bonds_0 = bonds_5;
    (&kernelContext_66)->args_0 = args_5;
    (&kernelContext_66)->info_0 = info_5;
    (&kernelContext_66)->scan_0 = scan_5;
    (&kernelContext_66)->parts_0 = parts_5;
    (&kernelContext_66)->colliders_0 = colliders_5;
    (&kernelContext_66)->collider_info_0 = collider_info_5;
    (&kernelContext_66)->view_chunks_0 = view_chunks_5;
    (&kernelContext_66)->view_0 = view_5;
    (&kernelContext_66)->objects_0 = objects_5;
    (&kernelContext_66)->shot_bodies_0 = shot_bodies_5;
    (&kernelContext_66)->lines_0 = lines_5;
    (&kernelContext_66)->geometry_0 = geometry_5;
    (&kernelContext_66)->shots_0 = shots_5;
    (&kernelContext_66)->lp_0 = lp_5;
    (&kernelContext_66)->ext_0 = ext_6;
    (&kernelContext_66)->engine_counters_0 = engine_counters_5;
    (&kernelContext_66)->engine_contacts_0 = engine_contacts_5;
    (&kernelContext_66)->cs_anchors_0 = cs_anchors_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_66)->g_scan_0 = &g_scan_5;
    uint s_2 = id_4.x;
    bool _S466;
    if(s_2 >= (params_5->body_count_0))
    {
        _S466 = true;
    }
    else
    {
        bool _S467 = dirty_0(s_2, &kernelContext_66);
        _S466 = !_S467;
    }
    if(_S466)
    {
        return;
    }
    uint _S468 = a_best_count_0(&kernelContext_66);
    atomic_store_explicit((&kernelContext_66)->atom_0+(_S468 + s_2), 0U, memory_order_relaxed);
    uint _S469 = a_best_root_0(&kernelContext_66);
    atomic_store_explicit((&kernelContext_66)->atom_0+(_S469 + s_2), 4294967295U, memory_order_relaxed);
    return;
}

uint find_0(uint x_4, KernelContext_0 thread* kernelContext_67)
{
    uint _S470 = a_parent_0();
    uint _S471 = atomic_load_explicit(kernelContext_67->atom_0+(_S470 + x_4), memory_order_relaxed);
    uint p_6 = _S471;
    uint _S472 = x_4;
    uint guard_0 = 0U;
    for(;;)
    {
        bool _S473;
        if(guard_0 < 1048576U)
        {
            _S473 = p_6 != _S472;
        }
        else
        {
            _S473 = false;
        }
        if(_S473)
        {
        }
        else
        {
            break;
        }
        uint gp_0 = atomic_load_explicit(kernelContext_67->atom_0+(_S470 + p_6), memory_order_relaxed);
        if(gp_0 != p_6)
        {
            uint _S474;
            {
            uint _metal_cas_comparand = p_6;
            atomic_compare_exchange_weak_explicit(kernelContext_67->atom_0+(_S470 + _S472), &_metal_cas_comparand, gp_0, memory_order_relaxed, memory_order_relaxed);
            _S474 = _metal_cas_comparand;
            }
        }
        uint _S475 = guard_0 + 1U;
        _S472 = p_6;
        p_6 = gp_0;
        guard_0 = _S475;
    }
    return _S472;
}

[[kernel]] void fr_uf_union(uint3 id_5 [[thread_position_in_grid]], FractureParams_0 constant* params_6 [[buffer(0)]], uint device* work_6 [[buffer(4)]], packed_float4 device* frames_6 [[buffer(6)]], packed_float4 device* bodies_6 [[buffer(7)]], packed_float4 device* chunks_6 [[buffer(1)]], uint device* sort_6 [[buffer(11)]], atomic<uint> device* atom_6 [[buffer(5)]], Record_natural_0 device* manifolds_6 [[buffer(10)]], packed_float4 device* loads_6 [[buffer(12)]], packed_uint4 device* bonds_6 [[buffer(2)]], uint device* args_6 [[buffer(13)]], uint device* info_6 [[buffer(15)]], ScanParams_0 constant* scan_6 [[buffer(14)]], packed_float4 device* parts_6 [[buffer(3)]], packed_float4 device* colliders_6 [[buffer(8)]], packed_uint4 device* collider_info_6 [[buffer(9)]], packed_float4 device* view_chunks_6 [[buffer(17)]], ViewParams_0 constant* view_6 [[buffer(16)]], packed_float4 device* objects_6 [[buffer(18)]], uint device* shot_bodies_6 [[buffer(20)]], float device* lines_6 [[buffer(19)]], packed_float4 device* geometry_6 [[buffer(21)]], packed_float4 device* shots_6 [[buffer(22)]], packed_float4 device* lp_6 [[buffer(24)]], packed_float4 device* ext_7 [[buffer(23)]], uint device* engine_counters_6 [[buffer(26)]], packed_float4 device* engine_contacts_6 [[buffer(25)]], packed_float4 device* cs_anchors_6 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_68;
    (&kernelContext_68)->params_0 = params_6;
    (&kernelContext_68)->work_0 = work_6;
    (&kernelContext_68)->frames_0 = frames_6;
    (&kernelContext_68)->bodies_0 = bodies_6;
    (&kernelContext_68)->chunks_0 = chunks_6;
    (&kernelContext_68)->sort_0 = sort_6;
    (&kernelContext_68)->atom_0 = atom_6;
    (&kernelContext_68)->manifolds_0 = manifolds_6;
    (&kernelContext_68)->loads_0 = loads_6;
    (&kernelContext_68)->bonds_0 = bonds_6;
    (&kernelContext_68)->args_0 = args_6;
    (&kernelContext_68)->info_0 = info_6;
    (&kernelContext_68)->scan_0 = scan_6;
    (&kernelContext_68)->parts_0 = parts_6;
    (&kernelContext_68)->colliders_0 = colliders_6;
    (&kernelContext_68)->collider_info_0 = collider_info_6;
    (&kernelContext_68)->view_chunks_0 = view_chunks_6;
    (&kernelContext_68)->view_0 = view_6;
    (&kernelContext_68)->objects_0 = objects_6;
    (&kernelContext_68)->shot_bodies_0 = shot_bodies_6;
    (&kernelContext_68)->lines_0 = lines_6;
    (&kernelContext_68)->geometry_0 = geometry_6;
    (&kernelContext_68)->shots_0 = shots_6;
    (&kernelContext_68)->lp_0 = lp_6;
    (&kernelContext_68)->ext_0 = ext_7;
    (&kernelContext_68)->engine_counters_0 = engine_counters_6;
    (&kernelContext_68)->engine_contacts_0 = engine_contacts_6;
    (&kernelContext_68)->cs_anchors_0 = cs_anchors_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_68)->g_scan_0 = &g_scan_6;
    uint i_3 = id_5.x;
    bool _S476;
    if(i_3 >= (params_6->bond_count_0))
    {
        _S476 = true;
    }
    else
    {
        uint _S477 = w_alive_0(&kernelContext_68);
        _S476 = (*((&kernelContext_68)->work_0+(_S477 + i_3))) == 0U;
    }
    if(_S476)
    {
        return;
    }
    uint4 _S478 = uint4(*((&kernelContext_68)->bonds_0+i_3)) ;
    uint _S479 = _S478.x;
    uint _S480 = body_of_0(_S479, &kernelContext_68);
    bool _S481 = dirty_0(_S480, &kernelContext_68);
    if(!_S481)
    {
        return;
    }
    uint _S482 = _S478.y;
    uint a_5 = _S479;
    uint b_23 = _S482;
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
        uint _S483 = find_0(a_5, &kernelContext_68);
        uint _S484 = find_0(b_23, &kernelContext_68);
        if(_S483 == _S484)
        {
            return;
        }
        uint _S485 = max(_S483, _S484);
        uint _S486 = min(_S483, _S484);
        uint was_0;
        {
        uint _metal_cas_comparand = _S485;
        atomic_compare_exchange_weak_explicit((&kernelContext_68)->atom_0+(a_parent_0() + _S485), &_metal_cas_comparand, _S486, memory_order_relaxed, memory_order_relaxed);
        was_0 = _metal_cas_comparand;
        }
        if(was_0 == _S485)
        {
            return;
        }
        uint _S487 = guard_1 + 1U;
        a_5 = _S485;
        b_23 = _S486;
        guard_1 = _S487;
    }
    uint _S488 = a_counters_1(&kernelContext_68);
    uint _S489 = atomic_fetch_or_explicit((&kernelContext_68)->atom_0+(_S488 + 3U), 4U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_uf_flatten(uint3 id_6 [[thread_position_in_grid]], FractureParams_0 constant* params_7 [[buffer(0)]], uint device* work_7 [[buffer(4)]], packed_float4 device* frames_7 [[buffer(6)]], packed_float4 device* bodies_7 [[buffer(7)]], packed_float4 device* chunks_7 [[buffer(1)]], uint device* sort_7 [[buffer(11)]], atomic<uint> device* atom_7 [[buffer(5)]], Record_natural_0 device* manifolds_7 [[buffer(10)]], packed_float4 device* loads_7 [[buffer(12)]], packed_uint4 device* bonds_7 [[buffer(2)]], uint device* args_7 [[buffer(13)]], uint device* info_7 [[buffer(15)]], ScanParams_0 constant* scan_7 [[buffer(14)]], packed_float4 device* parts_7 [[buffer(3)]], packed_float4 device* colliders_7 [[buffer(8)]], packed_uint4 device* collider_info_7 [[buffer(9)]], packed_float4 device* view_chunks_7 [[buffer(17)]], ViewParams_0 constant* view_7 [[buffer(16)]], packed_float4 device* objects_7 [[buffer(18)]], uint device* shot_bodies_7 [[buffer(20)]], float device* lines_7 [[buffer(19)]], packed_float4 device* geometry_7 [[buffer(21)]], packed_float4 device* shots_7 [[buffer(22)]], packed_float4 device* lp_7 [[buffer(24)]], packed_float4 device* ext_8 [[buffer(23)]], uint device* engine_counters_7 [[buffer(26)]], packed_float4 device* engine_contacts_7 [[buffer(25)]], packed_float4 device* cs_anchors_7 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_69;
    (&kernelContext_69)->params_0 = params_7;
    (&kernelContext_69)->work_0 = work_7;
    (&kernelContext_69)->frames_0 = frames_7;
    (&kernelContext_69)->bodies_0 = bodies_7;
    (&kernelContext_69)->chunks_0 = chunks_7;
    (&kernelContext_69)->sort_0 = sort_7;
    (&kernelContext_69)->atom_0 = atom_7;
    (&kernelContext_69)->manifolds_0 = manifolds_7;
    (&kernelContext_69)->loads_0 = loads_7;
    (&kernelContext_69)->bonds_0 = bonds_7;
    (&kernelContext_69)->args_0 = args_7;
    (&kernelContext_69)->info_0 = info_7;
    (&kernelContext_69)->scan_0 = scan_7;
    (&kernelContext_69)->parts_0 = parts_7;
    (&kernelContext_69)->colliders_0 = colliders_7;
    (&kernelContext_69)->collider_info_0 = collider_info_7;
    (&kernelContext_69)->view_chunks_0 = view_chunks_7;
    (&kernelContext_69)->view_0 = view_7;
    (&kernelContext_69)->objects_0 = objects_7;
    (&kernelContext_69)->shot_bodies_0 = shot_bodies_7;
    (&kernelContext_69)->lines_0 = lines_7;
    (&kernelContext_69)->geometry_0 = geometry_7;
    (&kernelContext_69)->shots_0 = shots_7;
    (&kernelContext_69)->lp_0 = lp_7;
    (&kernelContext_69)->ext_0 = ext_8;
    (&kernelContext_69)->engine_counters_0 = engine_counters_7;
    (&kernelContext_69)->engine_contacts_0 = engine_contacts_7;
    (&kernelContext_69)->cs_anchors_0 = cs_anchors_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_69)->g_scan_0 = &g_scan_7;
    uint c_7 = id_6.x;
    bool _S490;
    if(c_7 >= (params_7->chunk_count_0))
    {
        _S490 = true;
    }
    else
    {
        uint _S491 = body_of_0(c_7, &kernelContext_69);
        bool _S492 = dirty_0(_S491, &kernelContext_69);
        _S490 = !_S492;
    }
    if(_S490)
    {
        return;
    }
    uint _S493 = find_0(c_7, &kernelContext_69);
    uint _S494 = w_label_0(&kernelContext_69);
    *((&kernelContext_69)->work_0+(_S494 + c_7)) = _S493;
    uint _S495 = a_count_0(&kernelContext_69);
    uint _S496 = atomic_fetch_add_explicit((&kernelContext_69)->atom_0+(_S495 + _S493), 1U, memory_order_relaxed);
    if(((&kernelContext_69)->info_0[c_7] & 1U) != 0U)
    {
        uint _S497 = a_anchor_0(&kernelContext_69);
        uint _S498 = atomic_fetch_or_explicit((&kernelContext_69)->atom_0+(_S497 + _S493), 1U, memory_order_relaxed);
    }
    uint _S499 = 4U * c_7;
    float4 _S500 = float4(*((&kernelContext_69)->chunks_0+_S499)) ;
    float4 _S501 = float4(*((&kernelContext_69)->chunks_0+(_S499 + 1U))) ;
    float4 _S502 = float4(*((&kernelContext_69)->chunks_0+(_S499 + 2U))) ;
    float m_2 = _S500.w;
    float3 d_17 = _S500.xyz - (float4(*((&kernelContext_69)->chunks_0+4U * _S493)) ).xyz;
    float dd_0 = dot(d_17, d_17);
    uint _S503 = a_sums_0(&kernelContext_69);
    uint at_4 = _S503 + 20U * _S493;
    add64_0(at_4, m_2, (&kernelContext_69)->params_0->scale_m_0, &kernelContext_69);
    float _S504 = d_17.x;
    float _S505 = m_2 * _S504;
    add64_0(at_4 + 2U, _S505, (&kernelContext_69)->params_0->scale_s_0, &kernelContext_69);
    float _S506 = d_17.y;
    float _S507 = m_2 * _S506;
    add64_0(at_4 + 4U, _S507, (&kernelContext_69)->params_0->scale_s_0, &kernelContext_69);
    float _S508 = d_17.z;
    add64_0(at_4 + 6U, m_2 * _S508, (&kernelContext_69)->params_0->scale_s_0, &kernelContext_69);
    add64_0(at_4 + 8U, _S501.x + m_2 * (dd_0 - _S504 * _S504), (&kernelContext_69)->params_0->scale_j_0, &kernelContext_69);
    add64_0(at_4 + 10U, _S501.y + m_2 * (dd_0 - _S506 * _S506), (&kernelContext_69)->params_0->scale_j_0, &kernelContext_69);
    add64_0(at_4 + 12U, _S501.z + m_2 * (dd_0 - _S508 * _S508), (&kernelContext_69)->params_0->scale_j_0, &kernelContext_69);
    add64_0(at_4 + 14U, _S501.w - _S505 * _S506, (&kernelContext_69)->params_0->scale_j_0, &kernelContext_69);
    add64_0(at_4 + 16U, _S502.x - _S505 * _S508, (&kernelContext_69)->params_0->scale_j_0, &kernelContext_69);
    add64_0(at_4 + 18U, _S502.y - _S507 * _S508, (&kernelContext_69)->params_0->scale_j_0, &kernelContext_69);
    return;
}

bool is_root_0(uint c_8, KernelContext_0 thread* kernelContext_70)
{
    bool _S509;
    if(c_8 < (kernelContext_70->params_0->chunk_count_0))
    {
        uint _S510 = body_of_0(c_8, kernelContext_70);
        bool _S511 = dirty_0(_S510, kernelContext_70);
        _S509 = _S511;
    }
    else
    {
        _S509 = false;
    }
    if(_S509)
    {
        uint _S512 = w_label_0(kernelContext_70);
        _S509 = (*(kernelContext_70->work_0+(_S512 + c_8))) == c_8;
    }
    else
    {
        _S509 = false;
    }
    return _S509;
}

[[kernel]] void fr_best_count(uint3 id_7 [[thread_position_in_grid]], FractureParams_0 constant* params_8 [[buffer(0)]], uint device* work_8 [[buffer(4)]], packed_float4 device* frames_8 [[buffer(6)]], packed_float4 device* bodies_8 [[buffer(7)]], packed_float4 device* chunks_8 [[buffer(1)]], uint device* sort_8 [[buffer(11)]], atomic<uint> device* atom_8 [[buffer(5)]], Record_natural_0 device* manifolds_8 [[buffer(10)]], packed_float4 device* loads_8 [[buffer(12)]], packed_uint4 device* bonds_8 [[buffer(2)]], uint device* args_8 [[buffer(13)]], uint device* info_8 [[buffer(15)]], ScanParams_0 constant* scan_8 [[buffer(14)]], packed_float4 device* parts_8 [[buffer(3)]], packed_float4 device* colliders_8 [[buffer(8)]], packed_uint4 device* collider_info_8 [[buffer(9)]], packed_float4 device* view_chunks_8 [[buffer(17)]], ViewParams_0 constant* view_8 [[buffer(16)]], packed_float4 device* objects_8 [[buffer(18)]], uint device* shot_bodies_8 [[buffer(20)]], float device* lines_8 [[buffer(19)]], packed_float4 device* geometry_8 [[buffer(21)]], packed_float4 device* shots_8 [[buffer(22)]], packed_float4 device* lp_8 [[buffer(24)]], packed_float4 device* ext_9 [[buffer(23)]], uint device* engine_counters_8 [[buffer(26)]], packed_float4 device* engine_contacts_8 [[buffer(25)]], packed_float4 device* cs_anchors_8 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_71;
    (&kernelContext_71)->params_0 = params_8;
    (&kernelContext_71)->work_0 = work_8;
    (&kernelContext_71)->frames_0 = frames_8;
    (&kernelContext_71)->bodies_0 = bodies_8;
    (&kernelContext_71)->chunks_0 = chunks_8;
    (&kernelContext_71)->sort_0 = sort_8;
    (&kernelContext_71)->atom_0 = atom_8;
    (&kernelContext_71)->manifolds_0 = manifolds_8;
    (&kernelContext_71)->loads_0 = loads_8;
    (&kernelContext_71)->bonds_0 = bonds_8;
    (&kernelContext_71)->args_0 = args_8;
    (&kernelContext_71)->info_0 = info_8;
    (&kernelContext_71)->scan_0 = scan_8;
    (&kernelContext_71)->parts_0 = parts_8;
    (&kernelContext_71)->colliders_0 = colliders_8;
    (&kernelContext_71)->collider_info_0 = collider_info_8;
    (&kernelContext_71)->view_chunks_0 = view_chunks_8;
    (&kernelContext_71)->view_0 = view_8;
    (&kernelContext_71)->objects_0 = objects_8;
    (&kernelContext_71)->shot_bodies_0 = shot_bodies_8;
    (&kernelContext_71)->lines_0 = lines_8;
    (&kernelContext_71)->geometry_0 = geometry_8;
    (&kernelContext_71)->shots_0 = shots_8;
    (&kernelContext_71)->lp_0 = lp_8;
    (&kernelContext_71)->ext_0 = ext_9;
    (&kernelContext_71)->engine_counters_0 = engine_counters_8;
    (&kernelContext_71)->engine_contacts_0 = engine_contacts_8;
    (&kernelContext_71)->cs_anchors_0 = cs_anchors_8;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_71)->g_scan_0 = &g_scan_8;
    uint c_9 = id_7.x;
    bool _S513 = is_root_0(c_9, &kernelContext_71);
    if(!_S513)
    {
        return;
    }
    uint _S514 = a_best_count_0(&kernelContext_71);
    uint _S515 = body_of_0(c_9, &kernelContext_71);
    atomic<uint> device* _S516 = (&kernelContext_71)->atom_0+(_S514 + _S515);
    uint _S517 = a_count_0(&kernelContext_71);
    uint _S518 = atomic_load_explicit((&kernelContext_71)->atom_0+(_S517 + c_9), memory_order_relaxed);
    uint _S519 = atomic_fetch_max_explicit(_S516, _S518, memory_order_relaxed);
    return;
}

[[kernel]] void fr_best_root(uint3 id_8 [[thread_position_in_grid]], FractureParams_0 constant* params_9 [[buffer(0)]], uint device* work_9 [[buffer(4)]], packed_float4 device* frames_9 [[buffer(6)]], packed_float4 device* bodies_9 [[buffer(7)]], packed_float4 device* chunks_9 [[buffer(1)]], uint device* sort_9 [[buffer(11)]], atomic<uint> device* atom_9 [[buffer(5)]], Record_natural_0 device* manifolds_9 [[buffer(10)]], packed_float4 device* loads_9 [[buffer(12)]], packed_uint4 device* bonds_9 [[buffer(2)]], uint device* args_9 [[buffer(13)]], uint device* info_9 [[buffer(15)]], ScanParams_0 constant* scan_9 [[buffer(14)]], packed_float4 device* parts_9 [[buffer(3)]], packed_float4 device* colliders_9 [[buffer(8)]], packed_uint4 device* collider_info_9 [[buffer(9)]], packed_float4 device* view_chunks_9 [[buffer(17)]], ViewParams_0 constant* view_9 [[buffer(16)]], packed_float4 device* objects_9 [[buffer(18)]], uint device* shot_bodies_9 [[buffer(20)]], float device* lines_9 [[buffer(19)]], packed_float4 device* geometry_9 [[buffer(21)]], packed_float4 device* shots_9 [[buffer(22)]], packed_float4 device* lp_9 [[buffer(24)]], packed_float4 device* ext_10 [[buffer(23)]], uint device* engine_counters_9 [[buffer(26)]], packed_float4 device* engine_contacts_9 [[buffer(25)]], packed_float4 device* cs_anchors_9 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_72;
    (&kernelContext_72)->params_0 = params_9;
    (&kernelContext_72)->work_0 = work_9;
    (&kernelContext_72)->frames_0 = frames_9;
    (&kernelContext_72)->bodies_0 = bodies_9;
    (&kernelContext_72)->chunks_0 = chunks_9;
    (&kernelContext_72)->sort_0 = sort_9;
    (&kernelContext_72)->atom_0 = atom_9;
    (&kernelContext_72)->manifolds_0 = manifolds_9;
    (&kernelContext_72)->loads_0 = loads_9;
    (&kernelContext_72)->bonds_0 = bonds_9;
    (&kernelContext_72)->args_0 = args_9;
    (&kernelContext_72)->info_0 = info_9;
    (&kernelContext_72)->scan_0 = scan_9;
    (&kernelContext_72)->parts_0 = parts_9;
    (&kernelContext_72)->colliders_0 = colliders_9;
    (&kernelContext_72)->collider_info_0 = collider_info_9;
    (&kernelContext_72)->view_chunks_0 = view_chunks_9;
    (&kernelContext_72)->view_0 = view_9;
    (&kernelContext_72)->objects_0 = objects_9;
    (&kernelContext_72)->shot_bodies_0 = shot_bodies_9;
    (&kernelContext_72)->lines_0 = lines_9;
    (&kernelContext_72)->geometry_0 = geometry_9;
    (&kernelContext_72)->shots_0 = shots_9;
    (&kernelContext_72)->lp_0 = lp_9;
    (&kernelContext_72)->ext_0 = ext_10;
    (&kernelContext_72)->engine_counters_0 = engine_counters_9;
    (&kernelContext_72)->engine_contacts_0 = engine_contacts_9;
    (&kernelContext_72)->cs_anchors_0 = cs_anchors_9;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_72)->g_scan_0 = &g_scan_9;
    uint c_10 = id_8.x;
    bool _S520 = is_root_0(c_10, &kernelContext_72);
    if(!_S520)
    {
        return;
    }
    uint _S521 = body_of_0(c_10, &kernelContext_72);
    uint _S522 = a_count_0(&kernelContext_72);
    uint _S523 = atomic_load_explicit((&kernelContext_72)->atom_0+(_S522 + c_10), memory_order_relaxed);
    uint _S524 = a_best_count_0(&kernelContext_72);
    uint _S525 = atomic_load_explicit((&kernelContext_72)->atom_0+(_S524 + _S521), memory_order_relaxed);
    if(_S523 == _S525)
    {
        uint _S526 = a_best_root_0(&kernelContext_72);
        uint _S527 = atomic_fetch_min_explicit((&kernelContext_72)->atom_0+(_S526 + _S521), c_10, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_mark(uint3 id_9 [[thread_position_in_grid]], FractureParams_0 constant* params_10 [[buffer(0)]], uint device* work_10 [[buffer(4)]], packed_float4 device* frames_10 [[buffer(6)]], packed_float4 device* bodies_10 [[buffer(7)]], packed_float4 device* chunks_10 [[buffer(1)]], uint device* sort_10 [[buffer(11)]], atomic<uint> device* atom_10 [[buffer(5)]], Record_natural_0 device* manifolds_10 [[buffer(10)]], packed_float4 device* loads_10 [[buffer(12)]], packed_uint4 device* bonds_10 [[buffer(2)]], uint device* args_10 [[buffer(13)]], uint device* info_10 [[buffer(15)]], ScanParams_0 constant* scan_10 [[buffer(14)]], packed_float4 device* parts_10 [[buffer(3)]], packed_float4 device* colliders_10 [[buffer(8)]], packed_uint4 device* collider_info_10 [[buffer(9)]], packed_float4 device* view_chunks_10 [[buffer(17)]], ViewParams_0 constant* view_10 [[buffer(16)]], packed_float4 device* objects_10 [[buffer(18)]], uint device* shot_bodies_10 [[buffer(20)]], float device* lines_10 [[buffer(19)]], packed_float4 device* geometry_10 [[buffer(21)]], packed_float4 device* shots_10 [[buffer(22)]], packed_float4 device* lp_10 [[buffer(24)]], packed_float4 device* ext_11 [[buffer(23)]], uint device* engine_counters_10 [[buffer(26)]], packed_float4 device* engine_contacts_10 [[buffer(25)]], packed_float4 device* cs_anchors_10 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_73;
    (&kernelContext_73)->params_0 = params_10;
    (&kernelContext_73)->work_0 = work_10;
    (&kernelContext_73)->frames_0 = frames_10;
    (&kernelContext_73)->bodies_0 = bodies_10;
    (&kernelContext_73)->chunks_0 = chunks_10;
    (&kernelContext_73)->sort_0 = sort_10;
    (&kernelContext_73)->atom_0 = atom_10;
    (&kernelContext_73)->manifolds_0 = manifolds_10;
    (&kernelContext_73)->loads_0 = loads_10;
    (&kernelContext_73)->bonds_0 = bonds_10;
    (&kernelContext_73)->args_0 = args_10;
    (&kernelContext_73)->info_0 = info_10;
    (&kernelContext_73)->scan_0 = scan_10;
    (&kernelContext_73)->parts_0 = parts_10;
    (&kernelContext_73)->colliders_0 = colliders_10;
    (&kernelContext_73)->collider_info_0 = collider_info_10;
    (&kernelContext_73)->view_chunks_0 = view_chunks_10;
    (&kernelContext_73)->view_0 = view_10;
    (&kernelContext_73)->objects_0 = objects_10;
    (&kernelContext_73)->shot_bodies_0 = shot_bodies_10;
    (&kernelContext_73)->lines_0 = lines_10;
    (&kernelContext_73)->geometry_0 = geometry_10;
    (&kernelContext_73)->shots_0 = shots_10;
    (&kernelContext_73)->lp_0 = lp_10;
    (&kernelContext_73)->ext_0 = ext_11;
    (&kernelContext_73)->engine_counters_0 = engine_counters_10;
    (&kernelContext_73)->engine_contacts_0 = engine_contacts_10;
    (&kernelContext_73)->cs_anchors_0 = cs_anchors_10;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_73)->g_scan_0 = &g_scan_10;
    uint c_11 = id_9.x;
    if(c_11 >= (params_10->chunk_count_0))
    {
        return;
    }
    uint _S528 = w_new_0(&kernelContext_73);
    uint device* _S529 = (&kernelContext_73)->work_0+(_S528 + c_11);
    bool _S530 = is_root_0(c_11, &kernelContext_73);
    bool _S531;
    if(_S530)
    {
        uint _S532 = a_best_root_0(&kernelContext_73);
        uint _S533 = body_of_0(c_11, &kernelContext_73);
        uint _S534 = atomic_load_explicit((&kernelContext_73)->atom_0+(_S532 + _S533), memory_order_relaxed);
        _S531 = _S534 != c_11;
    }
    else
    {
        _S531 = false;
    }
    uint _S535;
    if(_S531)
    {
        _S535 = 1U;
    }
    else
    {
        _S535 = 0U;
    }
    *_S529 = _S535;
    return;
}

[[kernel]] void fr_scan_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_11 [[buffer(0)]], uint device* work_11 [[buffer(4)]], packed_float4 device* frames_11 [[buffer(6)]], packed_float4 device* bodies_11 [[buffer(7)]], packed_float4 device* chunks_11 [[buffer(1)]], uint device* sort_11 [[buffer(11)]], atomic<uint> device* atom_11 [[buffer(5)]], Record_natural_0 device* manifolds_11 [[buffer(10)]], packed_float4 device* loads_11 [[buffer(12)]], packed_uint4 device* bonds_11 [[buffer(2)]], uint device* args_11 [[buffer(13)]], uint device* info_11 [[buffer(15)]], ScanParams_0 constant* scan_11 [[buffer(14)]], packed_float4 device* parts_11 [[buffer(3)]], packed_float4 device* colliders_11 [[buffer(8)]], packed_uint4 device* collider_info_11 [[buffer(9)]], packed_float4 device* view_chunks_11 [[buffer(17)]], ViewParams_0 constant* view_11 [[buffer(16)]], packed_float4 device* objects_11 [[buffer(18)]], uint device* shot_bodies_11 [[buffer(20)]], float device* lines_11 [[buffer(19)]], packed_float4 device* geometry_11 [[buffer(21)]], packed_float4 device* shots_11 [[buffer(22)]], packed_float4 device* lp_11 [[buffer(24)]], packed_float4 device* ext_12 [[buffer(23)]], uint device* engine_counters_11 [[buffer(26)]], packed_float4 device* engine_contacts_11 [[buffer(25)]], packed_float4 device* cs_anchors_11 [[buffer(27)]])
{
    uint step_2;
    uint add_0;
    thread KernelContext_0 kernelContext_74;
    (&kernelContext_74)->params_0 = params_11;
    (&kernelContext_74)->work_0 = work_11;
    (&kernelContext_74)->frames_0 = frames_11;
    (&kernelContext_74)->bodies_0 = bodies_11;
    (&kernelContext_74)->chunks_0 = chunks_11;
    (&kernelContext_74)->sort_0 = sort_11;
    (&kernelContext_74)->atom_0 = atom_11;
    (&kernelContext_74)->manifolds_0 = manifolds_11;
    (&kernelContext_74)->loads_0 = loads_11;
    (&kernelContext_74)->bonds_0 = bonds_11;
    (&kernelContext_74)->args_0 = args_11;
    (&kernelContext_74)->info_0 = info_11;
    (&kernelContext_74)->scan_0 = scan_11;
    (&kernelContext_74)->parts_0 = parts_11;
    (&kernelContext_74)->colliders_0 = colliders_11;
    (&kernelContext_74)->collider_info_0 = collider_info_11;
    (&kernelContext_74)->view_chunks_0 = view_chunks_11;
    (&kernelContext_74)->view_0 = view_11;
    (&kernelContext_74)->objects_0 = objects_11;
    (&kernelContext_74)->shot_bodies_0 = shot_bodies_11;
    (&kernelContext_74)->lines_0 = lines_11;
    (&kernelContext_74)->geometry_0 = geometry_11;
    (&kernelContext_74)->shots_0 = shots_11;
    (&kernelContext_74)->lp_0 = lp_11;
    (&kernelContext_74)->ext_0 = ext_12;
    (&kernelContext_74)->engine_counters_0 = engine_counters_11;
    (&kernelContext_74)->engine_contacts_0 = engine_contacts_11;
    (&kernelContext_74)->cs_anchors_0 = cs_anchors_11;
    threadgroup array<uint, int(1024)> g_scan_11;
    (&kernelContext_74)->g_scan_0 = &g_scan_11;
    uint t_1 = thread_0.x;
    uint _S536 = group_0.x;
    uint _S537 = 1024U * _S536 + 4U * t_1;
    thread array<uint, int(4)> v_5;
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
        if((_S537 + j_3) < ((&kernelContext_74)->scan_0->n_0))
        {
            step_2 = *((&kernelContext_74)->work_0+((&kernelContext_74)->scan_0->in_base_0 + _S537 + j_3));
        }
        else
        {
            step_2 = 0U;
        }
        v_5[j_3] = step_2;
        uint sum_3 = sum_2 + step_2;
        j_3 = j_3 + 1U;
        sum_2 = sum_3;
    }
    (*(&kernelContext_74)->g_scan_0)[t_1] = sum_2;
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
            add_0 = (*(&kernelContext_74)->g_scan_0)[t_1 - step_2];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_74)->g_scan_0)[t_1] = (*(&kernelContext_74)->g_scan_0)[t_1] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_2 = step_2 << 1U;
    }
    uint _S538 = (*(&kernelContext_74)->g_scan_0)[t_1] - sum_2;
    j_3 = 0U;
    add_0 = _S538;
    for(;;)
    {
        if(j_3 < 4U)
        {
        }
        else
        {
            break;
        }
        if((_S537 + j_3) < ((&kernelContext_74)->scan_0->n_0))
        {
            *((&kernelContext_74)->work_0+((&kernelContext_74)->scan_0->out_base_0 + _S537 + j_3)) = add_0;
        }
        uint run_1 = add_0 + v_5[j_3];
        j_3 = j_3 + 1U;
        add_0 = run_1;
    }
    if(t_1 == 255U)
    {
        *((&kernelContext_74)->work_0+((&kernelContext_74)->scan_0->block_base_0 + _S536)) = (*(&kernelContext_74)->g_scan_0)[int(255)];
    }
    return;
}

[[kernel]] void fr_scan_top(uint3 thread_1 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_12 [[buffer(0)]], uint device* work_12 [[buffer(4)]], packed_float4 device* frames_12 [[buffer(6)]], packed_float4 device* bodies_12 [[buffer(7)]], packed_float4 device* chunks_12 [[buffer(1)]], uint device* sort_12 [[buffer(11)]], atomic<uint> device* atom_12 [[buffer(5)]], Record_natural_0 device* manifolds_12 [[buffer(10)]], packed_float4 device* loads_12 [[buffer(12)]], packed_uint4 device* bonds_12 [[buffer(2)]], uint device* args_12 [[buffer(13)]], uint device* info_12 [[buffer(15)]], ScanParams_0 constant* scan_12 [[buffer(14)]], packed_float4 device* parts_12 [[buffer(3)]], packed_float4 device* colliders_12 [[buffer(8)]], packed_uint4 device* collider_info_12 [[buffer(9)]], packed_float4 device* view_chunks_12 [[buffer(17)]], ViewParams_0 constant* view_12 [[buffer(16)]], packed_float4 device* objects_12 [[buffer(18)]], uint device* shot_bodies_12 [[buffer(20)]], float device* lines_12 [[buffer(19)]], packed_float4 device* geometry_12 [[buffer(21)]], packed_float4 device* shots_12 [[buffer(22)]], packed_float4 device* lp_12 [[buffer(24)]], packed_float4 device* ext_13 [[buffer(23)]], uint device* engine_counters_12 [[buffer(26)]], packed_float4 device* engine_contacts_12 [[buffer(25)]], packed_float4 device* cs_anchors_12 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_75;
    (&kernelContext_75)->params_0 = params_12;
    (&kernelContext_75)->work_0 = work_12;
    (&kernelContext_75)->frames_0 = frames_12;
    (&kernelContext_75)->bodies_0 = bodies_12;
    (&kernelContext_75)->chunks_0 = chunks_12;
    (&kernelContext_75)->sort_0 = sort_12;
    (&kernelContext_75)->atom_0 = atom_12;
    (&kernelContext_75)->manifolds_0 = manifolds_12;
    (&kernelContext_75)->loads_0 = loads_12;
    (&kernelContext_75)->bonds_0 = bonds_12;
    (&kernelContext_75)->args_0 = args_12;
    (&kernelContext_75)->info_0 = info_12;
    (&kernelContext_75)->scan_0 = scan_12;
    (&kernelContext_75)->parts_0 = parts_12;
    (&kernelContext_75)->colliders_0 = colliders_12;
    (&kernelContext_75)->collider_info_0 = collider_info_12;
    (&kernelContext_75)->view_chunks_0 = view_chunks_12;
    (&kernelContext_75)->view_0 = view_12;
    (&kernelContext_75)->objects_0 = objects_12;
    (&kernelContext_75)->shot_bodies_0 = shot_bodies_12;
    (&kernelContext_75)->lines_0 = lines_12;
    (&kernelContext_75)->geometry_0 = geometry_12;
    (&kernelContext_75)->shots_0 = shots_12;
    (&kernelContext_75)->lp_0 = lp_12;
    (&kernelContext_75)->ext_0 = ext_13;
    (&kernelContext_75)->engine_counters_0 = engine_counters_12;
    (&kernelContext_75)->engine_contacts_0 = engine_contacts_12;
    (&kernelContext_75)->cs_anchors_0 = cs_anchors_12;
    threadgroup array<uint, int(1024)> g_scan_12;
    (&kernelContext_75)->g_scan_0 = &g_scan_12;
    uint t_2 = thread_1.x;
    uint blocks_0 = (scan_12->n_0 + 1023U) / 1024U;
    bool _S539 = t_2 < blocks_0;
    uint v_6;
    if(_S539)
    {
        v_6 = *((&kernelContext_75)->work_0+((&kernelContext_75)->scan_0->block_base_0 + t_2));
    }
    else
    {
        v_6 = 0U;
    }
    (*(&kernelContext_75)->g_scan_0)[t_2] = v_6;
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
            add_1 = (*(&kernelContext_75)->g_scan_0)[t_2 - step_3];
        }
        else
        {
            add_1 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_75)->g_scan_0)[t_2] = (*(&kernelContext_75)->g_scan_0)[t_2] + add_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_3 = step_3 << 1U;
    }
    if(_S539)
    {
        *((&kernelContext_75)->work_0+((&kernelContext_75)->scan_0->block_base_0 + t_2)) = (*(&kernelContext_75)->g_scan_0)[t_2] - v_6;
    }
    if(t_2 == 1023U)
    {
        *((&kernelContext_75)->work_0+((&kernelContext_75)->scan_0->block_base_0 + blocks_0)) = (*(&kernelContext_75)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void fr_scan_add(uint3 id_10 [[thread_position_in_grid]], FractureParams_0 constant* params_13 [[buffer(0)]], uint device* work_13 [[buffer(4)]], packed_float4 device* frames_13 [[buffer(6)]], packed_float4 device* bodies_13 [[buffer(7)]], packed_float4 device* chunks_13 [[buffer(1)]], uint device* sort_13 [[buffer(11)]], atomic<uint> device* atom_13 [[buffer(5)]], Record_natural_0 device* manifolds_13 [[buffer(10)]], packed_float4 device* loads_13 [[buffer(12)]], packed_uint4 device* bonds_13 [[buffer(2)]], uint device* args_13 [[buffer(13)]], uint device* info_13 [[buffer(15)]], ScanParams_0 constant* scan_13 [[buffer(14)]], packed_float4 device* parts_13 [[buffer(3)]], packed_float4 device* colliders_13 [[buffer(8)]], packed_uint4 device* collider_info_13 [[buffer(9)]], packed_float4 device* view_chunks_13 [[buffer(17)]], ViewParams_0 constant* view_13 [[buffer(16)]], packed_float4 device* objects_13 [[buffer(18)]], uint device* shot_bodies_13 [[buffer(20)]], float device* lines_13 [[buffer(19)]], packed_float4 device* geometry_13 [[buffer(21)]], packed_float4 device* shots_13 [[buffer(22)]], packed_float4 device* lp_13 [[buffer(24)]], packed_float4 device* ext_14 [[buffer(23)]], uint device* engine_counters_13 [[buffer(26)]], packed_float4 device* engine_contacts_13 [[buffer(25)]], packed_float4 device* cs_anchors_13 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_76;
    (&kernelContext_76)->params_0 = params_13;
    (&kernelContext_76)->work_0 = work_13;
    (&kernelContext_76)->frames_0 = frames_13;
    (&kernelContext_76)->bodies_0 = bodies_13;
    (&kernelContext_76)->chunks_0 = chunks_13;
    (&kernelContext_76)->sort_0 = sort_13;
    (&kernelContext_76)->atom_0 = atom_13;
    (&kernelContext_76)->manifolds_0 = manifolds_13;
    (&kernelContext_76)->loads_0 = loads_13;
    (&kernelContext_76)->bonds_0 = bonds_13;
    (&kernelContext_76)->args_0 = args_13;
    (&kernelContext_76)->info_0 = info_13;
    (&kernelContext_76)->scan_0 = scan_13;
    (&kernelContext_76)->parts_0 = parts_13;
    (&kernelContext_76)->colliders_0 = colliders_13;
    (&kernelContext_76)->collider_info_0 = collider_info_13;
    (&kernelContext_76)->view_chunks_0 = view_chunks_13;
    (&kernelContext_76)->view_0 = view_13;
    (&kernelContext_76)->objects_0 = objects_13;
    (&kernelContext_76)->shot_bodies_0 = shot_bodies_13;
    (&kernelContext_76)->lines_0 = lines_13;
    (&kernelContext_76)->geometry_0 = geometry_13;
    (&kernelContext_76)->shots_0 = shots_13;
    (&kernelContext_76)->lp_0 = lp_13;
    (&kernelContext_76)->ext_0 = ext_14;
    (&kernelContext_76)->engine_counters_0 = engine_counters_13;
    (&kernelContext_76)->engine_contacts_0 = engine_contacts_13;
    (&kernelContext_76)->cs_anchors_0 = cs_anchors_13;
    threadgroup array<uint, int(1024)> g_scan_13;
    (&kernelContext_76)->g_scan_0 = &g_scan_13;
    uint i_4 = id_10.x;
    if(i_4 >= (scan_13->n_0))
    {
        return;
    }
    uint device* _S540 = (&kernelContext_76)->work_0+((&kernelContext_76)->scan_0->out_base_0 + i_4);
    *_S540 = *_S540 + *((&kernelContext_76)->work_0+((&kernelContext_76)->scan_0->block_base_0 + i_4 / 1024U));
    return;
}

[[kernel]] void fr_freelist(uint3 id_11 [[thread_position_in_grid]], FractureParams_0 constant* params_14 [[buffer(0)]], uint device* work_14 [[buffer(4)]], packed_float4 device* frames_14 [[buffer(6)]], packed_float4 device* bodies_14 [[buffer(7)]], packed_float4 device* chunks_14 [[buffer(1)]], uint device* sort_14 [[buffer(11)]], atomic<uint> device* atom_14 [[buffer(5)]], Record_natural_0 device* manifolds_14 [[buffer(10)]], packed_float4 device* loads_14 [[buffer(12)]], packed_uint4 device* bonds_14 [[buffer(2)]], uint device* args_14 [[buffer(13)]], uint device* info_14 [[buffer(15)]], ScanParams_0 constant* scan_14 [[buffer(14)]], packed_float4 device* parts_14 [[buffer(3)]], packed_float4 device* colliders_14 [[buffer(8)]], packed_uint4 device* collider_info_14 [[buffer(9)]], packed_float4 device* view_chunks_14 [[buffer(17)]], ViewParams_0 constant* view_14 [[buffer(16)]], packed_float4 device* objects_14 [[buffer(18)]], uint device* shot_bodies_14 [[buffer(20)]], float device* lines_14 [[buffer(19)]], packed_float4 device* geometry_14 [[buffer(21)]], packed_float4 device* shots_14 [[buffer(22)]], packed_float4 device* lp_14 [[buffer(24)]], packed_float4 device* ext_15 [[buffer(23)]], uint device* engine_counters_14 [[buffer(26)]], packed_float4 device* engine_contacts_14 [[buffer(25)]], packed_float4 device* cs_anchors_14 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_77;
    (&kernelContext_77)->params_0 = params_14;
    (&kernelContext_77)->work_0 = work_14;
    (&kernelContext_77)->frames_0 = frames_14;
    (&kernelContext_77)->bodies_0 = bodies_14;
    (&kernelContext_77)->chunks_0 = chunks_14;
    (&kernelContext_77)->sort_0 = sort_14;
    (&kernelContext_77)->atom_0 = atom_14;
    (&kernelContext_77)->manifolds_0 = manifolds_14;
    (&kernelContext_77)->loads_0 = loads_14;
    (&kernelContext_77)->bonds_0 = bonds_14;
    (&kernelContext_77)->args_0 = args_14;
    (&kernelContext_77)->info_0 = info_14;
    (&kernelContext_77)->scan_0 = scan_14;
    (&kernelContext_77)->parts_0 = parts_14;
    (&kernelContext_77)->colliders_0 = colliders_14;
    (&kernelContext_77)->collider_info_0 = collider_info_14;
    (&kernelContext_77)->view_chunks_0 = view_chunks_14;
    (&kernelContext_77)->view_0 = view_14;
    (&kernelContext_77)->objects_0 = objects_14;
    (&kernelContext_77)->shot_bodies_0 = shot_bodies_14;
    (&kernelContext_77)->lines_0 = lines_14;
    (&kernelContext_77)->geometry_0 = geometry_14;
    (&kernelContext_77)->shots_0 = shots_14;
    (&kernelContext_77)->lp_0 = lp_14;
    (&kernelContext_77)->ext_0 = ext_15;
    (&kernelContext_77)->engine_counters_0 = engine_counters_14;
    (&kernelContext_77)->engine_contacts_0 = engine_contacts_14;
    (&kernelContext_77)->cs_anchors_0 = cs_anchors_14;
    threadgroup array<uint, int(1024)> g_scan_14;
    (&kernelContext_77)->g_scan_0 = &g_scan_14;
    uint s_3 = id_11.x;
    bool _S541;
    if(s_3 >= (params_14->body_count_0))
    {
        _S541 = true;
    }
    else
    {
        uint _S542 = w_free_0(&kernelContext_77);
        _S541 = (*((&kernelContext_77)->work_0+(_S542 + s_3))) == 0U;
    }
    if(_S541)
    {
        return;
    }
    uint _S543 = w_free_list_0(&kernelContext_77);
    uint _S544 = w_free_rank_0(&kernelContext_77);
    *((&kernelContext_77)->work_0+(_S543 + *((&kernelContext_77)->work_0+(_S544 + s_3)))) = s_3;
    return;
}

uint scan_new_blocks_0(KernelContext_0 thread* kernelContext_78)
{
    uint _S545 = w_free_list_0(kernelContext_78);
    return _S545 + kernelContext_78->params_0->body_count_0;
}

uint scan_free_blocks_0(KernelContext_0 thread* kernelContext_79)
{
    uint _S546 = scan_new_blocks_0(kernelContext_79);
    return _S546 + 1025U;
}

uint free_total_0(KernelContext_0 thread* kernelContext_80)
{
    uint blocks_1 = (kernelContext_80->params_0->body_count_0 + 1023U) / 1024U;
    uint _S547 = scan_free_blocks_0(kernelContext_80);
    return *(kernelContext_80->work_0+(_S547 + blocks_1));
}

[[kernel]] void fr_assign(uint3 id_12 [[thread_position_in_grid]], FractureParams_0 constant* params_15 [[buffer(0)]], uint device* work_15 [[buffer(4)]], packed_float4 device* frames_15 [[buffer(6)]], packed_float4 device* bodies_15 [[buffer(7)]], packed_float4 device* chunks_15 [[buffer(1)]], uint device* sort_15 [[buffer(11)]], atomic<uint> device* atom_15 [[buffer(5)]], Record_natural_0 device* manifolds_15 [[buffer(10)]], packed_float4 device* loads_15 [[buffer(12)]], packed_uint4 device* bonds_15 [[buffer(2)]], uint device* args_15 [[buffer(13)]], uint device* info_15 [[buffer(15)]], ScanParams_0 constant* scan_15 [[buffer(14)]], packed_float4 device* parts_15 [[buffer(3)]], packed_float4 device* colliders_15 [[buffer(8)]], packed_uint4 device* collider_info_15 [[buffer(9)]], packed_float4 device* view_chunks_15 [[buffer(17)]], ViewParams_0 constant* view_15 [[buffer(16)]], packed_float4 device* objects_15 [[buffer(18)]], uint device* shot_bodies_15 [[buffer(20)]], float device* lines_15 [[buffer(19)]], packed_float4 device* geometry_15 [[buffer(21)]], packed_float4 device* shots_15 [[buffer(22)]], packed_float4 device* lp_15 [[buffer(24)]], packed_float4 device* ext_16 [[buffer(23)]], uint device* engine_counters_15 [[buffer(26)]], packed_float4 device* engine_contacts_15 [[buffer(25)]], packed_float4 device* cs_anchors_15 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_81;
    (&kernelContext_81)->params_0 = params_15;
    (&kernelContext_81)->work_0 = work_15;
    (&kernelContext_81)->frames_0 = frames_15;
    (&kernelContext_81)->bodies_0 = bodies_15;
    (&kernelContext_81)->chunks_0 = chunks_15;
    (&kernelContext_81)->sort_0 = sort_15;
    (&kernelContext_81)->atom_0 = atom_15;
    (&kernelContext_81)->manifolds_0 = manifolds_15;
    (&kernelContext_81)->loads_0 = loads_15;
    (&kernelContext_81)->bonds_0 = bonds_15;
    (&kernelContext_81)->args_0 = args_15;
    (&kernelContext_81)->info_0 = info_15;
    (&kernelContext_81)->scan_0 = scan_15;
    (&kernelContext_81)->parts_0 = parts_15;
    (&kernelContext_81)->colliders_0 = colliders_15;
    (&kernelContext_81)->collider_info_0 = collider_info_15;
    (&kernelContext_81)->view_chunks_0 = view_chunks_15;
    (&kernelContext_81)->view_0 = view_15;
    (&kernelContext_81)->objects_0 = objects_15;
    (&kernelContext_81)->shot_bodies_0 = shot_bodies_15;
    (&kernelContext_81)->lines_0 = lines_15;
    (&kernelContext_81)->geometry_0 = geometry_15;
    (&kernelContext_81)->shots_0 = shots_15;
    (&kernelContext_81)->lp_0 = lp_15;
    (&kernelContext_81)->ext_0 = ext_16;
    (&kernelContext_81)->engine_counters_0 = engine_counters_15;
    (&kernelContext_81)->engine_contacts_0 = engine_contacts_15;
    (&kernelContext_81)->cs_anchors_0 = cs_anchors_15;
    threadgroup array<uint, int(1024)> g_scan_15;
    (&kernelContext_81)->g_scan_0 = &g_scan_15;
    uint c_12 = id_12.x;
    bool _S548 = is_root_0(c_12, &kernelContext_81);
    if(!_S548)
    {
        return;
    }
    uint _S549 = body_of_0(c_12, &kernelContext_81);
    uint _S550 = a_best_root_0(&kernelContext_81);
    uint _S551 = atomic_load_explicit((&kernelContext_81)->atom_0+(_S550 + _S549), memory_order_relaxed);
    uint slot_1;
    if(_S551 != c_12)
    {
        uint _S552 = w_rank_0(&kernelContext_81);
        uint device* _S553 = (&kernelContext_81)->work_0+(_S552 + c_12);
        uint k_13 = *_S553;
        uint _S554 = free_total_0(&kernelContext_81);
        if((*_S553) < _S554)
        {
            uint _S555 = w_free_list_0(&kernelContext_81);
            uint device* _S556 = (&kernelContext_81)->work_0+(_S555 + k_13);
            uint slot_2 = *_S556;
            uint _S557 = w_free_0(&kernelContext_81);
            *((&kernelContext_81)->work_0+(_S557 + *_S556)) = 0U;
            uint _S558 = a_counters_1(&kernelContext_81);
            uint _S559 = atomic_fetch_add_explicit((&kernelContext_81)->atom_0+(_S558 + 2U), 1U, memory_order_relaxed);
            slot_1 = slot_2;
        }
        else
        {
            uint _S560 = a_counters_1(&kernelContext_81);
            uint _S561 = atomic_fetch_or_explicit((&kernelContext_81)->atom_0+(_S560 + 3U), 1U, memory_order_relaxed);
            slot_1 = _S549;
        }
    }
    else
    {
        slot_1 = _S549;
    }
    uint _S562 = w_slot_0(&kernelContext_81);
    *((&kernelContext_81)->work_0+(_S562 + c_12)) = slot_1;
    return;
}

float read64_1(uint at_5, float scale_2, KernelContext_0 thread* kernelContext_82)
{
    uint _S563 = atomic_load_explicit(kernelContext_82->atom_0+(at_5 + 1U), memory_order_relaxed);
    int hi_2 = (as_type<int>((_S563)));
    uint lo_2 = atomic_load_explicit(kernelContext_82->atom_0+at_5, memory_order_relaxed);
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
    array<float, int(3)> _S564 = { { r0_0.x, r0_0.y, r0_0.z } };
    array<float, int(3)> _S565 = { { r1_0.x, r1_0.y, r1_0.z } };
    array<float, int(3)> _S566 = { { r2_2.x, r2_2.y, r2_2.z } };
    thread array<array<float, int(3)>, int(3)> a_6;
    a_6[int(0)] = _S564;
    a_6[int(1)] = _S565;
    a_6[int(2)] = _S566;
    array<float, int(3)> _S567 = { { 1.0f, 0.0f, 0.0f } };
    array<float, int(3)> _S568 = { { 0.0f, 1.0f, 0.0f } };
    array<float, int(3)> _S569 = { { 0.0f, 0.0f, 1.0f } };
    thread array<array<float, int(3)>, int(3)> v_7;
    v_7[int(0)] = _S567;
    v_7[int(1)] = _S568;
    v_7[int(2)] = _S569;
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
        uint p_7;
        uint q_13;
        if((abs(a_6[int(1)][int(2)])) > (abs(a_6[0U][q_12])))
        {
            p_7 = 1U;
            q_13 = 2U;
        }
        else
        {
            p_7 = 0U;
            q_13 = q_12;
        }
        if((abs(a_6[p_7][q_13])) <= (9.99999982451670044e-15f * max(max(abs(a_6[int(0)][int(0)]), abs(a_6[int(1)][int(1)])), max(abs(a_6[int(2)][int(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        float theta_0 = 0.5f * atan2(2.0f * a_6[p_7][q_13], a_6[q_13][q_13] - a_6[p_7][p_7]);
        float s_4 = sin(theta_0);
        float c_13 = cos(theta_0);
        float app_0 = a_6[p_7][p_7];
        float aqq_0 = a_6[q_13][q_13];
        float apq_0 = a_6[p_7][q_13];
        uint k_14 = 0U;
        for(;;)
        {
            if(k_14 < 3U)
            {
            }
            else
            {
                break;
            }
            bool _S570;
            if(k_14 != p_7)
            {
                _S570 = k_14 != q_13;
            }
            else
            {
                _S570 = false;
            }
            if(_S570)
            {
                float kp_0 = a_6[k_14][p_7];
                float kq_0 = a_6[k_14][q_13];
                float _S571 = c_13 * a_6[k_14][p_7] - s_4 * a_6[k_14][q_13];
                a_6[k_14][p_7] = _S571;
                a_6[p_7][k_14] = _S571;
                float _S572 = s_4 * kp_0 + c_13 * kq_0;
                a_6[k_14][q_13] = _S572;
                a_6[q_13][k_14] = _S572;
            }
            k_14 = k_14 + 1U;
        }
        float _S573 = c_13 * c_13;
        float _S574 = 2.0f * s_4 * c_13 * apq_0;
        float _S575 = s_4 * s_4;
        a_6[p_7][p_7] = _S573 * app_0 - _S574 + _S575 * aqq_0;
        a_6[q_13][q_13] = _S575 * app_0 + _S574 + _S573 * aqq_0;
        a_6[p_7][q_13] = 0.0f;
        a_6[q_13][p_7] = 0.0f;
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
            float kp_1 = v_7[row_0][p_7];
            float kq_1 = v_7[row_0][q_13];
            v_7[row_0][p_7] = c_13 * v_7[row_0][p_7] - s_4 * v_7[row_0][q_13];
            v_7[row_0][q_13] = s_4 * kp_1 + c_13 * kq_1;
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
    thread Quat_0 _S576 = quat_from_columns_0(x_5, y_1, z_2);
    Quat_0 _S577 = normalized_0(&_S576);
    *rot_0 = _S577;
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
    thread Quat_0 _S578 = axes_1;
    Quat_0 _S579 = quat_mul_0(&_S578, q_14);
    thread Quat_0 _S580 = _S579;
    Quat_0 _S581 = normalized_0(&_S580);
    return _S581;
}

uint stage_0(uint c_14, KernelContext_0 thread* kernelContext_83)
{
    return 2U * kernelContext_83->params_0->body_count_0 + 9U * c_14;
}

float4 quat_vec_0(const Quat_0 thread* q_15)
{
    return float4(q_15->x_0, q_15->y_0, q_15->z_0, q_15->w_0);
}

[[kernel]] void fr_piece(uint3 id_13 [[thread_position_in_grid]], FractureParams_0 constant* params_16 [[buffer(0)]], uint device* work_16 [[buffer(4)]], packed_float4 device* frames_16 [[buffer(6)]], packed_float4 device* bodies_16 [[buffer(7)]], packed_float4 device* chunks_16 [[buffer(1)]], uint device* sort_16 [[buffer(11)]], atomic<uint> device* atom_16 [[buffer(5)]], Record_natural_0 device* manifolds_16 [[buffer(10)]], packed_float4 device* loads_16 [[buffer(12)]], packed_uint4 device* bonds_16 [[buffer(2)]], uint device* args_16 [[buffer(13)]], uint device* info_16 [[buffer(15)]], ScanParams_0 constant* scan_16 [[buffer(14)]], packed_float4 device* parts_16 [[buffer(3)]], packed_float4 device* colliders_16 [[buffer(8)]], packed_uint4 device* collider_info_16 [[buffer(9)]], packed_float4 device* view_chunks_16 [[buffer(17)]], ViewParams_0 constant* view_16 [[buffer(16)]], packed_float4 device* objects_16 [[buffer(18)]], uint device* shot_bodies_16 [[buffer(20)]], float device* lines_16 [[buffer(19)]], packed_float4 device* geometry_16 [[buffer(21)]], packed_float4 device* shots_16 [[buffer(22)]], packed_float4 device* lp_16 [[buffer(24)]], packed_float4 device* ext_17 [[buffer(23)]], uint device* engine_counters_16 [[buffer(26)]], packed_float4 device* engine_contacts_16 [[buffer(25)]], packed_float4 device* cs_anchors_16 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_84;
    (&kernelContext_84)->params_0 = params_16;
    (&kernelContext_84)->work_0 = work_16;
    (&kernelContext_84)->frames_0 = frames_16;
    (&kernelContext_84)->bodies_0 = bodies_16;
    (&kernelContext_84)->chunks_0 = chunks_16;
    (&kernelContext_84)->sort_0 = sort_16;
    (&kernelContext_84)->atom_0 = atom_16;
    (&kernelContext_84)->manifolds_0 = manifolds_16;
    (&kernelContext_84)->loads_0 = loads_16;
    (&kernelContext_84)->bonds_0 = bonds_16;
    (&kernelContext_84)->args_0 = args_16;
    (&kernelContext_84)->info_0 = info_16;
    (&kernelContext_84)->scan_0 = scan_16;
    (&kernelContext_84)->parts_0 = parts_16;
    (&kernelContext_84)->colliders_0 = colliders_16;
    (&kernelContext_84)->collider_info_0 = collider_info_16;
    (&kernelContext_84)->view_chunks_0 = view_chunks_16;
    (&kernelContext_84)->view_0 = view_16;
    (&kernelContext_84)->objects_0 = objects_16;
    (&kernelContext_84)->shot_bodies_0 = shot_bodies_16;
    (&kernelContext_84)->lines_0 = lines_16;
    (&kernelContext_84)->geometry_0 = geometry_16;
    (&kernelContext_84)->shots_0 = shots_16;
    (&kernelContext_84)->lp_0 = lp_16;
    (&kernelContext_84)->ext_0 = ext_17;
    (&kernelContext_84)->engine_counters_0 = engine_counters_16;
    (&kernelContext_84)->engine_contacts_0 = engine_contacts_16;
    (&kernelContext_84)->cs_anchors_0 = cs_anchors_16;
    threadgroup array<uint, int(1024)> g_scan_16;
    (&kernelContext_84)->g_scan_0 = &g_scan_16;
    uint c_15 = id_13.x;
    bool _S582 = is_root_0(c_15, &kernelContext_84);
    if(!_S582)
    {
        return;
    }
    uint _S583 = body_of_0(c_15, &kernelContext_84);
    uint pb_0 = 7U * _S583;
    uint _S584 = a_sums_0(&kernelContext_84);
    uint at_6 = _S584 + 20U * c_15;
    float _S585 = read64_0(at_6, (&kernelContext_84)->params_0->scale_m_0, &kernelContext_84);
    float _S586 = read64_0(at_6 + 2U, (&kernelContext_84)->params_0->scale_s_0, &kernelContext_84);
    float _S587 = read64_0(at_6 + 4U, (&kernelContext_84)->params_0->scale_s_0, &kernelContext_84);
    float _S588 = read64_0(at_6 + 6U, (&kernelContext_84)->params_0->scale_s_0, &kernelContext_84);
    float3 d_18 = float3(_S586, _S587, _S588) / float3(max(_S585, 1.00000000317107685e-30f)) ;
    float _S589 = read64_0(at_6 + 8U, (&kernelContext_84)->params_0->scale_j_0, &kernelContext_84);
    float _S590 = read64_0(at_6 + 10U, (&kernelContext_84)->params_0->scale_j_0, &kernelContext_84);
    float _S591 = read64_0(at_6 + 12U, (&kernelContext_84)->params_0->scale_j_0, &kernelContext_84);
    float _S592 = read64_0(at_6 + 14U, (&kernelContext_84)->params_0->scale_j_0, &kernelContext_84);
    float _S593 = read64_0(at_6 + 16U, (&kernelContext_84)->params_0->scale_j_0, &kernelContext_84);
    float _S594 = read64_0(at_6 + 18U, (&kernelContext_84)->params_0->scale_j_0, &kernelContext_84);
    float dd_1 = dot(d_18, d_18);
    float _S595 = d_18.x;
    float _S596 = _S585 * _S595;
    float _S597 = d_18.y;
    float _S598 = _S592 + _S596 * _S597;
    float _S599 = d_18.z;
    float _S600 = _S593 + _S596 * _S599;
    float _S601 = _S594 + _S585 * _S597 * _S599;
    float3 com_1 = (float4(*((&kernelContext_84)->chunks_0+4U * c_15)) ).xyz + d_18;
    thread Quat_0 principal_0;
    thread float3 moments_1;
    principal_axes_0(float3(_S589 - _S585 * (dd_1 - _S595 * _S595), _S598, _S600), float3(_S598, _S590 - _S585 * (dd_1 - _S597 * _S597), _S601), float3(_S600, _S601, _S591 - _S585 * (dd_1 - _S599 * _S599)), &principal_0, &moments_1);
    uint _S602 = a_anchor_0(&kernelContext_84);
    uint _S603 = atomic_load_explicit((&kernelContext_84)->atom_0+(_S602 + c_15), memory_order_relaxed);
    bool anchored_0 = _S603 != 0U;
    thread Quat_0 q_16;
    thread float3 origin_4;
    structure_pose_0(_S583, pb_0, &q_16, &origin_4, &kernelContext_84);
    float3 p_parent_0 = (float4(*((&kernelContext_84)->bodies_0+pb_0)) ).xyz;
    float3 v_parent_0 = (float4(*((&kernelContext_84)->bodies_0+(pb_0 + 3U))) ).xyz;
    float3 w_parent_0 = (float4(*((&kernelContext_84)->bodies_0+(pb_0 + 4U))) ).xyz;
    thread Quat_0 _S604 = q_16;
    float3 _S605 = rotate_0(&_S604, com_1);
    float3 pos_0 = to_rubble_0(origin_4 + _S605);
    thread Quat_0 _S606 = q_16;
    thread Quat_0 _S607 = principal_0;
    Quat_0 _S608 = quat_mul_0(&_S606, &_S607);
    thread Quat_0 _S609 = _S608;
    Quat_0 _S610 = normalized_0(&_S609);
    thread Quat_0 _S611 = _S610;
    Quat_0 _S612 = rot_to_rubble_0(&_S611);
    bool _S613;
    if(anchored_0)
    {
        _S613 = true;
    }
    else
    {
        _S613 = !(_S585 > 0.0f);
    }
    float inv_mass_3;
    if(_S613)
    {
        inv_mass_3 = 0.0f;
    }
    else
    {
        inv_mass_3 = 1.0f / _S585;
    }
    float3 _S614 = float3(0.0f) ;
    float3 inv_inertia_0;
    if(!anchored_0)
    {
        float _S615;
        if((moments_1.x) > 0.0f)
        {
            _S615 = 1.0f / moments_1.x;
        }
        else
        {
            _S615 = 0.0f;
        }
        float _S616;
        if((moments_1.y) > 0.0f)
        {
            _S616 = 1.0f / moments_1.y;
        }
        else
        {
            _S616 = 0.0f;
        }
        float _S617;
        if((moments_1.z) > 0.0f)
        {
            _S617 = 1.0f / moments_1.z;
        }
        else
        {
            _S617 = 0.0f;
        }
        inv_inertia_0 = float3(_S615, _S616, _S617);
    }
    else
    {
        inv_inertia_0 = _S614;
    }
    float3 velocity_0;
    if(anchored_0)
    {
        velocity_0 = _S614;
    }
    else
    {
        velocity_0 = v_parent_0 + cross(w_parent_0, pos_0 - p_parent_0);
    }
    float3 spin_0;
    if(anchored_0)
    {
        spin_0 = _S614;
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
    uint _S618 = stage_0(c_15, &kernelContext_84);
    *((&kernelContext_84)->frames_0+_S618) = packed_float4(float4(pos_0, inv_mass_3)) ;
    packed_float4 device* _S619 = (&kernelContext_84)->frames_0+(_S618 + 1U);
    thread Quat_0 _S620 = _S612;
    float4 _S621 = quat_vec_0(&_S620);
    *_S619 = packed_float4(_S621) ;
    *((&kernelContext_84)->frames_0+(_S618 + 2U)) = packed_float4(float4(inv_inertia_0, (as_type<float>((flags_0))))) ;
    *((&kernelContext_84)->frames_0+(_S618 + 3U)) = packed_float4(float4(velocity_0, 0.0f)) ;
    *((&kernelContext_84)->frames_0+(_S618 + 4U)) = packed_float4(float4(spin_0, (as_type<float>(((&kernelContext_84)->params_0->step_0))))) ;
    *((&kernelContext_84)->frames_0+(_S618 + 5U)) = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_84)->frames_0+(_S618 + 6U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, 1.0f)) ;
    *((&kernelContext_84)->frames_0+(_S618 + 7U)) = packed_float4(float4(com_1, (float4(*((&kernelContext_84)->frames_0+2U * _S583)) ).w)) ;
    packed_float4 device* _S622 = (&kernelContext_84)->frames_0+(_S618 + 8U);
    thread Quat_0 _S623 = principal_0;
    float4 _S624 = quat_vec_0(&_S623);
    *_S622 = packed_float4(_S624) ;
    return;
}

[[kernel]] void fr_commit(uint3 id_14 [[thread_position_in_grid]], FractureParams_0 constant* params_17 [[buffer(0)]], uint device* work_17 [[buffer(4)]], packed_float4 device* frames_17 [[buffer(6)]], packed_float4 device* bodies_17 [[buffer(7)]], packed_float4 device* chunks_17 [[buffer(1)]], uint device* sort_17 [[buffer(11)]], atomic<uint> device* atom_17 [[buffer(5)]], Record_natural_0 device* manifolds_17 [[buffer(10)]], packed_float4 device* loads_17 [[buffer(12)]], packed_uint4 device* bonds_17 [[buffer(2)]], uint device* args_17 [[buffer(13)]], uint device* info_17 [[buffer(15)]], ScanParams_0 constant* scan_17 [[buffer(14)]], packed_float4 device* parts_17 [[buffer(3)]], packed_float4 device* colliders_17 [[buffer(8)]], packed_uint4 device* collider_info_17 [[buffer(9)]], packed_float4 device* view_chunks_17 [[buffer(17)]], ViewParams_0 constant* view_17 [[buffer(16)]], packed_float4 device* objects_17 [[buffer(18)]], uint device* shot_bodies_17 [[buffer(20)]], float device* lines_17 [[buffer(19)]], packed_float4 device* geometry_17 [[buffer(21)]], packed_float4 device* shots_17 [[buffer(22)]], packed_float4 device* lp_17 [[buffer(24)]], packed_float4 device* ext_18 [[buffer(23)]], uint device* engine_counters_17 [[buffer(26)]], packed_float4 device* engine_contacts_17 [[buffer(25)]], packed_float4 device* cs_anchors_17 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_85;
    (&kernelContext_85)->params_0 = params_17;
    (&kernelContext_85)->work_0 = work_17;
    (&kernelContext_85)->frames_0 = frames_17;
    (&kernelContext_85)->bodies_0 = bodies_17;
    (&kernelContext_85)->chunks_0 = chunks_17;
    (&kernelContext_85)->sort_0 = sort_17;
    (&kernelContext_85)->atom_0 = atom_17;
    (&kernelContext_85)->manifolds_0 = manifolds_17;
    (&kernelContext_85)->loads_0 = loads_17;
    (&kernelContext_85)->bonds_0 = bonds_17;
    (&kernelContext_85)->args_0 = args_17;
    (&kernelContext_85)->info_0 = info_17;
    (&kernelContext_85)->scan_0 = scan_17;
    (&kernelContext_85)->parts_0 = parts_17;
    (&kernelContext_85)->colliders_0 = colliders_17;
    (&kernelContext_85)->collider_info_0 = collider_info_17;
    (&kernelContext_85)->view_chunks_0 = view_chunks_17;
    (&kernelContext_85)->view_0 = view_17;
    (&kernelContext_85)->objects_0 = objects_17;
    (&kernelContext_85)->shot_bodies_0 = shot_bodies_17;
    (&kernelContext_85)->lines_0 = lines_17;
    (&kernelContext_85)->geometry_0 = geometry_17;
    (&kernelContext_85)->shots_0 = shots_17;
    (&kernelContext_85)->lp_0 = lp_17;
    (&kernelContext_85)->ext_0 = ext_18;
    (&kernelContext_85)->engine_counters_0 = engine_counters_17;
    (&kernelContext_85)->engine_contacts_0 = engine_contacts_17;
    (&kernelContext_85)->cs_anchors_0 = cs_anchors_17;
    threadgroup array<uint, int(1024)> g_scan_17;
    (&kernelContext_85)->g_scan_0 = &g_scan_17;
    uint c_16 = id_14.x;
    bool _S625 = is_root_0(c_16, &kernelContext_85);
    if(!_S625)
    {
        return;
    }
    uint _S626 = w_slot_0(&kernelContext_85);
    uint slot_3 = *((&kernelContext_85)->work_0+(_S626 + c_16));
    uint _S627 = stage_0(c_16, &kernelContext_85);
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
            *((&kernelContext_85)->bodies_0+(7U * (slot_3 + region_5 * (&kernelContext_85)->params_0->max_bodies_0) + w_3)) = packed_float4(float4(*((&kernelContext_85)->frames_0+(_S627 + w_3))) ) ;
            w_3 = w_3 + 1U;
        }
        region_5 = region_5 + 1U;
    }
    uint _S628 = 2U * slot_3;
    *((&kernelContext_85)->frames_0+_S628) = packed_float4(float4(*((&kernelContext_85)->frames_0+(_S627 + 7U))) ) ;
    *((&kernelContext_85)->frames_0+(_S628 + 1U)) = packed_float4(float4(*((&kernelContext_85)->frames_0+(_S627 + 8U))) ) ;
    return;
}

[[kernel]] void fr_reparent(uint3 id_15 [[thread_position_in_grid]], FractureParams_0 constant* params_18 [[buffer(0)]], uint device* work_18 [[buffer(4)]], packed_float4 device* frames_18 [[buffer(6)]], packed_float4 device* bodies_18 [[buffer(7)]], packed_float4 device* chunks_18 [[buffer(1)]], uint device* sort_18 [[buffer(11)]], atomic<uint> device* atom_18 [[buffer(5)]], Record_natural_0 device* manifolds_18 [[buffer(10)]], packed_float4 device* loads_18 [[buffer(12)]], packed_uint4 device* bonds_18 [[buffer(2)]], uint device* args_18 [[buffer(13)]], uint device* info_18 [[buffer(15)]], ScanParams_0 constant* scan_18 [[buffer(14)]], packed_float4 device* parts_18 [[buffer(3)]], packed_float4 device* colliders_18 [[buffer(8)]], packed_uint4 device* collider_info_18 [[buffer(9)]], packed_float4 device* view_chunks_18 [[buffer(17)]], ViewParams_0 constant* view_18 [[buffer(16)]], packed_float4 device* objects_18 [[buffer(18)]], uint device* shot_bodies_18 [[buffer(20)]], float device* lines_18 [[buffer(19)]], packed_float4 device* geometry_18 [[buffer(21)]], packed_float4 device* shots_18 [[buffer(22)]], packed_float4 device* lp_18 [[buffer(24)]], packed_float4 device* ext_19 [[buffer(23)]], uint device* engine_counters_18 [[buffer(26)]], packed_float4 device* engine_contacts_18 [[buffer(25)]], packed_float4 device* cs_anchors_18 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_86;
    (&kernelContext_86)->params_0 = params_18;
    (&kernelContext_86)->work_0 = work_18;
    (&kernelContext_86)->frames_0 = frames_18;
    (&kernelContext_86)->bodies_0 = bodies_18;
    (&kernelContext_86)->chunks_0 = chunks_18;
    (&kernelContext_86)->sort_0 = sort_18;
    (&kernelContext_86)->atom_0 = atom_18;
    (&kernelContext_86)->manifolds_0 = manifolds_18;
    (&kernelContext_86)->loads_0 = loads_18;
    (&kernelContext_86)->bonds_0 = bonds_18;
    (&kernelContext_86)->args_0 = args_18;
    (&kernelContext_86)->info_0 = info_18;
    (&kernelContext_86)->scan_0 = scan_18;
    (&kernelContext_86)->parts_0 = parts_18;
    (&kernelContext_86)->colliders_0 = colliders_18;
    (&kernelContext_86)->collider_info_0 = collider_info_18;
    (&kernelContext_86)->view_chunks_0 = view_chunks_18;
    (&kernelContext_86)->view_0 = view_18;
    (&kernelContext_86)->objects_0 = objects_18;
    (&kernelContext_86)->shot_bodies_0 = shot_bodies_18;
    (&kernelContext_86)->lines_0 = lines_18;
    (&kernelContext_86)->geometry_0 = geometry_18;
    (&kernelContext_86)->shots_0 = shots_18;
    (&kernelContext_86)->lp_0 = lp_18;
    (&kernelContext_86)->ext_0 = ext_19;
    (&kernelContext_86)->engine_counters_0 = engine_counters_18;
    (&kernelContext_86)->engine_contacts_0 = engine_contacts_18;
    (&kernelContext_86)->cs_anchors_0 = cs_anchors_18;
    threadgroup array<uint, int(1024)> g_scan_18;
    (&kernelContext_86)->g_scan_0 = &g_scan_18;
    uint c_17 = id_15.x;
    bool _S629;
    if(c_17 >= (params_18->chunk_count_0))
    {
        _S629 = true;
    }
    else
    {
        uint _S630 = body_of_0(c_17, &kernelContext_86);
        bool _S631 = dirty_0(_S630, &kernelContext_86);
        _S629 = !_S631;
    }
    if(_S629)
    {
        return;
    }
    uint _S632 = w_label_0(&kernelContext_86);
    uint device* _S633 = (&kernelContext_86)->work_0+(_S632 + c_17);
    uint r_11 = *_S633;
    uint _S634 = w_slot_0(&kernelContext_86);
    uint device* _S635 = (&kernelContext_86)->work_0+(_S634 + *_S633);
    uint slot_4 = *_S635;
    *((&kernelContext_86)->work_0+(w_body_0() + c_17)) = *_S635;
    uint _S636 = 2U * slot_4;
    float3 _S637 = (float4(*((&kernelContext_86)->frames_0+_S636)) ).xyz;
    thread Quat_0 _S638 = quat_of_0(float4(*((&kernelContext_86)->frames_0+(_S636 + 1U))) );
    Quat_0 _S639 = conj_0(&_S638);
    float inv_mass_4 = (float4(*((&kernelContext_86)->bodies_0+7U * slot_4)) ).w;
    float _S640;
    if(inv_mass_4 > 0.0f)
    {
        _S640 = 1.0f / inv_mass_4;
    }
    else
    {
        _S640 = float_bits_0(2139095040U);
    }
    float4 _S641 = float4(*((&kernelContext_86)->chunks_0+(4U * c_17 + 2U))) ;
    uint first_1 = (as_type<uint>((_S641.z)));
    uint _S642 = (as_type<uint>((_S641.w)));
    float4 _S643 = float4(*((&kernelContext_86)->chunks_0+(4U * r_11 + 3U))) ;
    uint p_8 = first_1;
    for(;;)
    {
        if(p_8 < (first_1 + _S642))
        {
        }
        else
        {
            break;
        }
        uint _S644 = 2U * p_8;
        float4 _S645 = float4(*((&kernelContext_86)->parts_0+_S644)) ;
        uint col_0 = (as_type<uint>((_S645.w)));
        Quat_0 rot_1 = quat_of_0(float4(*((&kernelContext_86)->parts_0+(_S644 + 1U))) );
        uint _S646 = 4U * col_0;
        float4 _S647 = float4(*((&kernelContext_86)->colliders_0+_S646)) ;
        *((&kernelContext_86)->colliders_0+_S646) = packed_float4(float4((as_type<float>((slot_4))), _S647.y, _S647.z, _S647.w)) ;
        uint _S648 = _S646 + 1U;
        packed_float4 device* _S649 = (&kernelContext_86)->colliders_0+_S648;
        float3 _S650 = _S645.xyz - _S637;
        thread Quat_0 _S651 = _S639;
        float3 _S652 = rotate_0(&_S651, _S650);
        *_S649 = packed_float4(float4(_S652, (float4(*((&kernelContext_86)->colliders_0+_S648)) ).w)) ;
        packed_float4 device* _S653 = (&kernelContext_86)->colliders_0+(_S646 + 2U);
        thread Quat_0 _S654 = _S639;
        thread Quat_0 _S655 = rot_1;
        Quat_0 _S656 = quat_mul_0(&_S654, &_S655);
        thread Quat_0 _S657 = _S656;
        Quat_0 _S658 = normalized_0(&_S657);
        thread Quat_0 _S659 = _S658;
        float4 _S660 = quat_vec_0(&_S659);
        *_S653 = packed_float4(_S660) ;
        uint _S661 = 2U * col_0;
        uint4 _S662 = uint4(*((&kernelContext_86)->collider_info_0+_S661)) ;
        *((&kernelContext_86)->collider_info_0+_S661) = packed_uint4(uint4(_S662.x, slot_4, _S662.z, _S662.w)) ;
        *((&kernelContext_86)->collider_info_0+(_S661 + 1U)) = packed_uint4(uint4((as_type<uint>((_S640))), (as_type<uint>((_S643.x))), (as_type<uint>((_S643.w))), 0U)) ;
        p_8 = p_8 + 1U;
    }
    return;
}

[[kernel]] void fr_body_clear(uint3 id_16 [[thread_position_in_grid]], FractureParams_0 constant* params_19 [[buffer(0)]], uint device* work_19 [[buffer(4)]], packed_float4 device* frames_19 [[buffer(6)]], packed_float4 device* bodies_19 [[buffer(7)]], packed_float4 device* chunks_19 [[buffer(1)]], uint device* sort_19 [[buffer(11)]], atomic<uint> device* atom_19 [[buffer(5)]], Record_natural_0 device* manifolds_19 [[buffer(10)]], packed_float4 device* loads_19 [[buffer(12)]], packed_uint4 device* bonds_19 [[buffer(2)]], uint device* args_19 [[buffer(13)]], uint device* info_19 [[buffer(15)]], ScanParams_0 constant* scan_19 [[buffer(14)]], packed_float4 device* parts_19 [[buffer(3)]], packed_float4 device* colliders_19 [[buffer(8)]], packed_uint4 device* collider_info_19 [[buffer(9)]], packed_float4 device* view_chunks_19 [[buffer(17)]], ViewParams_0 constant* view_19 [[buffer(16)]], packed_float4 device* objects_19 [[buffer(18)]], uint device* shot_bodies_19 [[buffer(20)]], float device* lines_19 [[buffer(19)]], packed_float4 device* geometry_19 [[buffer(21)]], packed_float4 device* shots_19 [[buffer(22)]], packed_float4 device* lp_19 [[buffer(24)]], packed_float4 device* ext_20 [[buffer(23)]], uint device* engine_counters_19 [[buffer(26)]], packed_float4 device* engine_contacts_19 [[buffer(25)]], packed_float4 device* cs_anchors_19 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_87;
    (&kernelContext_87)->params_0 = params_19;
    (&kernelContext_87)->work_0 = work_19;
    (&kernelContext_87)->frames_0 = frames_19;
    (&kernelContext_87)->bodies_0 = bodies_19;
    (&kernelContext_87)->chunks_0 = chunks_19;
    (&kernelContext_87)->sort_0 = sort_19;
    (&kernelContext_87)->atom_0 = atom_19;
    (&kernelContext_87)->manifolds_0 = manifolds_19;
    (&kernelContext_87)->loads_0 = loads_19;
    (&kernelContext_87)->bonds_0 = bonds_19;
    (&kernelContext_87)->args_0 = args_19;
    (&kernelContext_87)->info_0 = info_19;
    (&kernelContext_87)->scan_0 = scan_19;
    (&kernelContext_87)->parts_0 = parts_19;
    (&kernelContext_87)->colliders_0 = colliders_19;
    (&kernelContext_87)->collider_info_0 = collider_info_19;
    (&kernelContext_87)->view_chunks_0 = view_chunks_19;
    (&kernelContext_87)->view_0 = view_19;
    (&kernelContext_87)->objects_0 = objects_19;
    (&kernelContext_87)->shot_bodies_0 = shot_bodies_19;
    (&kernelContext_87)->lines_0 = lines_19;
    (&kernelContext_87)->geometry_0 = geometry_19;
    (&kernelContext_87)->shots_0 = shots_19;
    (&kernelContext_87)->lp_0 = lp_19;
    (&kernelContext_87)->ext_0 = ext_20;
    (&kernelContext_87)->engine_counters_0 = engine_counters_19;
    (&kernelContext_87)->engine_contacts_0 = engine_contacts_19;
    (&kernelContext_87)->cs_anchors_0 = cs_anchors_19;
    threadgroup array<uint, int(1024)> g_scan_19;
    (&kernelContext_87)->g_scan_0 = &g_scan_19;
    uint s_5 = id_16.x;
    if(s_5 >= (params_19->body_count_0))
    {
        return;
    }
    uint _S663 = a_dirty_0(&kernelContext_87);
    atomic_store_explicit((&kernelContext_87)->atom_0+(_S663 + s_5), 0U, memory_order_relaxed);
    return;
}

float3 slot_color_0(uint slot_5)
{
    float h_4 = fract(float(slot_5) * 0.6180340051651001f) * 6.0f;
    float f_8 = fract(h_4);
    float q_17 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * f_8);
    float t_3 = 0.92000001668930054f * (1.0f - 0.62000000476837158f * (1.0f - f_8));
    uint k_15 = uint(h_4);
    float3 _S664;
    if(k_15 == 0U)
    {
        _S664 = float3(0.92000001668930054f, t_3, 0.34959998726844788f);
    }
    else
    {
        if(k_15 == 1U)
        {
            _S664 = float3(q_17, 0.92000001668930054f, 0.34959998726844788f);
        }
        else
        {
            if(k_15 == 2U)
            {
                _S664 = float3(0.34959998726844788f, 0.92000001668930054f, t_3);
            }
            else
            {
                if(k_15 == 3U)
                {
                    _S664 = float3(0.34959998726844788f, q_17, 0.92000001668930054f);
                }
                else
                {
                    if(k_15 == 4U)
                    {
                        _S664 = float3(t_3, 0.34959998726844788f, 0.92000001668930054f);
                    }
                    else
                    {
                        _S664 = float3(0.92000001668930054f, 0.34959998726844788f, q_17);
                    }
                }
            }
        }
    }
    return _S664;
}

void write_object_0(uint o_4, const Quat_0 thread* q_18, float3 t_4, float4 color_0, KernelContext_0 thread* kernelContext_88)
{
    float3 _S665 = rotate_0(q_18, float3(1.0f, 0.0f, 0.0f));
    float3 _S666 = rotate_0(q_18, float3(0.0f, 1.0f, 0.0f));
    float3 _S667 = rotate_0(q_18, float3(0.0f, 0.0f, 1.0f));
    uint _S668 = 4U * o_4;
    *(kernelContext_88->objects_0+_S668) = packed_float4(float4(_S665.x, _S666.x, _S667.x, t_4.x)) ;
    *(kernelContext_88->objects_0+(_S668 + 1U)) = packed_float4(float4(_S665.y, _S666.y, _S667.y, t_4.y)) ;
    *(kernelContext_88->objects_0+(_S668 + 2U)) = packed_float4(float4(_S665.z, _S666.z, _S667.z, t_4.z)) ;
    *(kernelContext_88->objects_0+(_S668 + 3U)) = packed_float4(color_0) ;
    return;
}

[[kernel]] void fr_view_chunks(uint3 id_17 [[thread_position_in_grid]], FractureParams_0 constant* params_20 [[buffer(0)]], uint device* work_20 [[buffer(4)]], packed_float4 device* frames_20 [[buffer(6)]], packed_float4 device* bodies_20 [[buffer(7)]], packed_float4 device* chunks_20 [[buffer(1)]], uint device* sort_20 [[buffer(11)]], atomic<uint> device* atom_20 [[buffer(5)]], Record_natural_0 device* manifolds_20 [[buffer(10)]], packed_float4 device* loads_20 [[buffer(12)]], packed_uint4 device* bonds_20 [[buffer(2)]], uint device* args_20 [[buffer(13)]], uint device* info_20 [[buffer(15)]], ScanParams_0 constant* scan_20 [[buffer(14)]], packed_float4 device* parts_20 [[buffer(3)]], packed_float4 device* colliders_20 [[buffer(8)]], packed_uint4 device* collider_info_20 [[buffer(9)]], packed_float4 device* view_chunks_20 [[buffer(17)]], ViewParams_0 constant* view_20 [[buffer(16)]], packed_float4 device* objects_20 [[buffer(18)]], uint device* shot_bodies_20 [[buffer(20)]], float device* lines_20 [[buffer(19)]], packed_float4 device* geometry_20 [[buffer(21)]], packed_float4 device* shots_20 [[buffer(22)]], packed_float4 device* lp_20 [[buffer(24)]], packed_float4 device* ext_21 [[buffer(23)]], uint device* engine_counters_20 [[buffer(26)]], packed_float4 device* engine_contacts_20 [[buffer(25)]], packed_float4 device* cs_anchors_20 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_89;
    (&kernelContext_89)->params_0 = params_20;
    (&kernelContext_89)->work_0 = work_20;
    (&kernelContext_89)->frames_0 = frames_20;
    (&kernelContext_89)->bodies_0 = bodies_20;
    (&kernelContext_89)->chunks_0 = chunks_20;
    (&kernelContext_89)->sort_0 = sort_20;
    (&kernelContext_89)->atom_0 = atom_20;
    (&kernelContext_89)->manifolds_0 = manifolds_20;
    (&kernelContext_89)->loads_0 = loads_20;
    (&kernelContext_89)->bonds_0 = bonds_20;
    (&kernelContext_89)->args_0 = args_20;
    (&kernelContext_89)->info_0 = info_20;
    (&kernelContext_89)->scan_0 = scan_20;
    (&kernelContext_89)->parts_0 = parts_20;
    (&kernelContext_89)->colliders_0 = colliders_20;
    (&kernelContext_89)->collider_info_0 = collider_info_20;
    (&kernelContext_89)->view_chunks_0 = view_chunks_20;
    (&kernelContext_89)->view_0 = view_20;
    (&kernelContext_89)->objects_0 = objects_20;
    (&kernelContext_89)->shot_bodies_0 = shot_bodies_20;
    (&kernelContext_89)->lines_0 = lines_20;
    (&kernelContext_89)->geometry_0 = geometry_20;
    (&kernelContext_89)->shots_0 = shots_20;
    (&kernelContext_89)->lp_0 = lp_20;
    (&kernelContext_89)->ext_0 = ext_21;
    (&kernelContext_89)->engine_counters_0 = engine_counters_20;
    (&kernelContext_89)->engine_contacts_0 = engine_contacts_20;
    (&kernelContext_89)->cs_anchors_0 = cs_anchors_20;
    threadgroup array<uint, int(1024)> g_scan_20;
    (&kernelContext_89)->g_scan_0 = &g_scan_20;
    uint k_16 = id_17.x;
    if(k_16 >= (params_20->chunk_count_0))
    {
        return;
    }
    uint _S669 = body_of_0(k_16, &kernelContext_89);
    uint b_24 = 7U * _S669;
    thread Quat_0 q_19;
    thread float3 origin_5;
    structure_pose_0(_S669, b_24, &q_19, &origin_5, &kernelContext_89);
    uint _S670 = 3U * k_16;
    float4 _S671 = float4(*((&kernelContext_89)->view_chunks_0+_S670)) ;
    float3 base_2 = (float4(*((&kernelContext_89)->view_chunks_0+(_S670 + 2U))) ).xyz;
    bool fixed_0 = ((float4(*((&kernelContext_89)->bodies_0+b_24)) ).w) == 0.0f;
    float3 color_1;
    if(((&kernelContext_89)->view_0->mode_0) == 1U)
    {
        float _S672 = max(length((float4(*((&kernelContext_89)->bodies_0+(b_24 + 3U))) ).xyz), length((float4(*((&kernelContext_89)->bodies_0+(b_24 + 4U))) ).xyz));
        if(fixed_0)
        {
            color_1 = float3(0.44999998807907104f, 0.44999998807907104f, 0.47999998927116394f);
        }
        else
        {
            if(_S672 > ((&kernelContext_89)->view_0->rest_speed_0))
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
        if(((&kernelContext_89)->view_0->mode_0) == 2U)
        {
            color_1 = slot_color_0(_S669);
        }
        else
        {
            if(fixed_0)
            {
                color_1 = base_2;
            }
            else
            {
                color_1 = float3(0.44999998807907104f)  * base_2 + float3(0.55000001192092896f)  * slot_color_0(_S669);
            }
        }
    }
    uint _S673 = (as_type<uint>((_S671.w)));
    Quat_0 _S674 = quat_of_0(float4(*((&kernelContext_89)->view_chunks_0+(_S670 + 1U))) );
    thread Quat_0 _S675 = q_19;
    thread Quat_0 _S676 = _S674;
    Quat_0 _S677 = quat_mul_0(&_S675, &_S676);
    thread Quat_0 _S678 = _S677;
    Quat_0 _S679 = normalized_0(&_S678);
    float3 _S680 = _S671.xyz;
    thread Quat_0 _S681 = q_19;
    float3 _S682 = rotate_0(&_S681, _S680);
    float3 _S683 = origin_5 + _S682;
    float4 _S684 = float4(color_1, 1.0f);
    thread Quat_0 _S685 = _S679;
    write_object_0(_S673, &_S685, _S683, _S684, &kernelContext_89);
    return;
}

[[kernel]] void fr_view_shots(uint3 id_18 [[thread_position_in_grid]], FractureParams_0 constant* params_21 [[buffer(0)]], uint device* work_21 [[buffer(4)]], packed_float4 device* frames_21 [[buffer(6)]], packed_float4 device* bodies_21 [[buffer(7)]], packed_float4 device* chunks_21 [[buffer(1)]], uint device* sort_21 [[buffer(11)]], atomic<uint> device* atom_21 [[buffer(5)]], Record_natural_0 device* manifolds_21 [[buffer(10)]], packed_float4 device* loads_21 [[buffer(12)]], packed_uint4 device* bonds_21 [[buffer(2)]], uint device* args_21 [[buffer(13)]], uint device* info_21 [[buffer(15)]], ScanParams_0 constant* scan_21 [[buffer(14)]], packed_float4 device* parts_21 [[buffer(3)]], packed_float4 device* colliders_21 [[buffer(8)]], packed_uint4 device* collider_info_21 [[buffer(9)]], packed_float4 device* view_chunks_21 [[buffer(17)]], ViewParams_0 constant* view_21 [[buffer(16)]], packed_float4 device* objects_21 [[buffer(18)]], uint device* shot_bodies_21 [[buffer(20)]], float device* lines_21 [[buffer(19)]], packed_float4 device* geometry_21 [[buffer(21)]], packed_float4 device* shots_21 [[buffer(22)]], packed_float4 device* lp_21 [[buffer(24)]], packed_float4 device* ext_22 [[buffer(23)]], uint device* engine_counters_21 [[buffer(26)]], packed_float4 device* engine_contacts_21 [[buffer(25)]], packed_float4 device* cs_anchors_21 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_90;
    (&kernelContext_90)->params_0 = params_21;
    (&kernelContext_90)->work_0 = work_21;
    (&kernelContext_90)->frames_0 = frames_21;
    (&kernelContext_90)->bodies_0 = bodies_21;
    (&kernelContext_90)->chunks_0 = chunks_21;
    (&kernelContext_90)->sort_0 = sort_21;
    (&kernelContext_90)->atom_0 = atom_21;
    (&kernelContext_90)->manifolds_0 = manifolds_21;
    (&kernelContext_90)->loads_0 = loads_21;
    (&kernelContext_90)->bonds_0 = bonds_21;
    (&kernelContext_90)->args_0 = args_21;
    (&kernelContext_90)->info_0 = info_21;
    (&kernelContext_90)->scan_0 = scan_21;
    (&kernelContext_90)->parts_0 = parts_21;
    (&kernelContext_90)->colliders_0 = colliders_21;
    (&kernelContext_90)->collider_info_0 = collider_info_21;
    (&kernelContext_90)->view_chunks_0 = view_chunks_21;
    (&kernelContext_90)->view_0 = view_21;
    (&kernelContext_90)->objects_0 = objects_21;
    (&kernelContext_90)->shot_bodies_0 = shot_bodies_21;
    (&kernelContext_90)->lines_0 = lines_21;
    (&kernelContext_90)->geometry_0 = geometry_21;
    (&kernelContext_90)->shots_0 = shots_21;
    (&kernelContext_90)->lp_0 = lp_21;
    (&kernelContext_90)->ext_0 = ext_22;
    (&kernelContext_90)->engine_counters_0 = engine_counters_21;
    (&kernelContext_90)->engine_contacts_0 = engine_contacts_21;
    (&kernelContext_90)->cs_anchors_0 = cs_anchors_21;
    threadgroup array<uint, int(1024)> g_scan_21;
    (&kernelContext_90)->g_scan_0 = &g_scan_21;
    uint i_5 = id_18.x;
    if(i_5 >= (view_21->shot_count_1))
    {
        return;
    }
    uint b_25 = 7U * (&kernelContext_90)->shot_bodies_0[i_5];
    uint o_5 = (&kernelContext_90)->view_0->impactor_base_0 + i_5;
    if(((float4(*((&kernelContext_90)->bodies_0+b_25)) ).w) == 0.0f)
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
            *((&kernelContext_90)->objects_0+(4U * o_5 + w_4)) = packed_float4(float4(0.0f) ) ;
            w_4 = w_4 + 1U;
        }
        return;
    }
    thread Quat_0 _S686 = quat_of_0(float4(*((&kernelContext_90)->bodies_0+(b_25 + 1U))) );
    Quat_0 _S687 = rot_from_rubble_0(&_S686);
    float3 _S688 = from_rubble_0((float4(*((&kernelContext_90)->bodies_0+b_25)) ).xyz);
    float4 _S689 = float4(0.18000000715255737f, 0.18999999761581421f, 0.2199999988079071f, 1.0f);
    thread Quat_0 _S690 = _S687;
    write_object_0(o_5, &_S690, _S688, _S689, &kernelContext_90);
    return;
}

float3 chunk_world_0(uint k_17, KernelContext_0 thread* kernelContext_91)
{
    uint _S691 = body_of_0(k_17, kernelContext_91);
    thread Quat_0 q_20;
    thread float3 origin_6;
    structure_pose_0(_S691, 7U * _S691, &q_20, &origin_6, kernelContext_91);
    float3 _S692 = (float4(*(kernelContext_91->view_chunks_0+3U * k_17)) ).xyz;
    thread Quat_0 _S693 = q_20;
    float3 _S694 = rotate_0(&_S693, _S692);
    return origin_6 + _S694;
}

void write_line_vertex_0(uint v_8, float3 p_9, KernelContext_0 thread* kernelContext_92)
{
    uint _S695 = 7U * v_8;
    *(kernelContext_92->lines_0+_S695) = p_9.x;
    *(kernelContext_92->lines_0+(_S695 + 1U)) = p_9.y;
    *(kernelContext_92->lines_0+(_S695 + 2U)) = p_9.z;
    *(kernelContext_92->lines_0+(_S695 + 3U)) = 1.0f;
    *(kernelContext_92->lines_0+(_S695 + 4U)) = 0.86000001430511475f;
    *(kernelContext_92->lines_0+(_S695 + 5U)) = 0.25f;
    *(kernelContext_92->lines_0+(_S695 + 6U)) = 1.0f;
    return;
}

[[kernel]] void fr_view_bonds(uint3 id_19 [[thread_position_in_grid]], FractureParams_0 constant* params_22 [[buffer(0)]], uint device* work_22 [[buffer(4)]], packed_float4 device* frames_22 [[buffer(6)]], packed_float4 device* bodies_22 [[buffer(7)]], packed_float4 device* chunks_22 [[buffer(1)]], uint device* sort_22 [[buffer(11)]], atomic<uint> device* atom_22 [[buffer(5)]], Record_natural_0 device* manifolds_22 [[buffer(10)]], packed_float4 device* loads_22 [[buffer(12)]], packed_uint4 device* bonds_22 [[buffer(2)]], uint device* args_22 [[buffer(13)]], uint device* info_22 [[buffer(15)]], ScanParams_0 constant* scan_22 [[buffer(14)]], packed_float4 device* parts_22 [[buffer(3)]], packed_float4 device* colliders_22 [[buffer(8)]], packed_uint4 device* collider_info_22 [[buffer(9)]], packed_float4 device* view_chunks_22 [[buffer(17)]], ViewParams_0 constant* view_22 [[buffer(16)]], packed_float4 device* objects_22 [[buffer(18)]], uint device* shot_bodies_22 [[buffer(20)]], float device* lines_22 [[buffer(19)]], packed_float4 device* geometry_22 [[buffer(21)]], packed_float4 device* shots_22 [[buffer(22)]], packed_float4 device* lp_22 [[buffer(24)]], packed_float4 device* ext_23 [[buffer(23)]], uint device* engine_counters_22 [[buffer(26)]], packed_float4 device* engine_contacts_22 [[buffer(25)]], packed_float4 device* cs_anchors_22 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_93;
    (&kernelContext_93)->params_0 = params_22;
    (&kernelContext_93)->work_0 = work_22;
    (&kernelContext_93)->frames_0 = frames_22;
    (&kernelContext_93)->bodies_0 = bodies_22;
    (&kernelContext_93)->chunks_0 = chunks_22;
    (&kernelContext_93)->sort_0 = sort_22;
    (&kernelContext_93)->atom_0 = atom_22;
    (&kernelContext_93)->manifolds_0 = manifolds_22;
    (&kernelContext_93)->loads_0 = loads_22;
    (&kernelContext_93)->bonds_0 = bonds_22;
    (&kernelContext_93)->args_0 = args_22;
    (&kernelContext_93)->info_0 = info_22;
    (&kernelContext_93)->scan_0 = scan_22;
    (&kernelContext_93)->parts_0 = parts_22;
    (&kernelContext_93)->colliders_0 = colliders_22;
    (&kernelContext_93)->collider_info_0 = collider_info_22;
    (&kernelContext_93)->view_chunks_0 = view_chunks_22;
    (&kernelContext_93)->view_0 = view_22;
    (&kernelContext_93)->objects_0 = objects_22;
    (&kernelContext_93)->shot_bodies_0 = shot_bodies_22;
    (&kernelContext_93)->lines_0 = lines_22;
    (&kernelContext_93)->geometry_0 = geometry_22;
    (&kernelContext_93)->shots_0 = shots_22;
    (&kernelContext_93)->lp_0 = lp_22;
    (&kernelContext_93)->ext_0 = ext_23;
    (&kernelContext_93)->engine_counters_0 = engine_counters_22;
    (&kernelContext_93)->engine_contacts_0 = engine_contacts_22;
    (&kernelContext_93)->cs_anchors_0 = cs_anchors_22;
    threadgroup array<uint, int(1024)> g_scan_22;
    (&kernelContext_93)->g_scan_0 = &g_scan_22;
    uint i_6 = id_19.x;
    if(i_6 >= (params_22->bond_count_0))
    {
        return;
    }
    uint4 _S696 = uint4(*((&kernelContext_93)->bonds_0+i_6)) ;
    float3 _S697 = chunk_world_0(_S696.x, &kernelContext_93);
    uint _S698 = w_alive_0(&kernelContext_93);
    float3 b_26;
    if((*((&kernelContext_93)->work_0+(_S698 + i_6))) != 0U)
    {
        float3 _S699 = chunk_world_0(_S696.y, &kernelContext_93);
        b_26 = _S699;
    }
    else
    {
        b_26 = _S697;
    }
    uint _S700 = 2U * i_6;
    write_line_vertex_0(_S700, _S697, &kernelContext_93);
    write_line_vertex_0(_S700 + 1U, b_26, &kernelContext_93);
    return;
}

[[kernel]] void fr_strike_reset(uint3 id_20 [[thread_position_in_grid]], FractureParams_0 constant* params_23 [[buffer(0)]], uint device* work_23 [[buffer(4)]], packed_float4 device* frames_23 [[buffer(6)]], packed_float4 device* bodies_23 [[buffer(7)]], packed_float4 device* chunks_23 [[buffer(1)]], uint device* sort_23 [[buffer(11)]], atomic<uint> device* atom_23 [[buffer(5)]], Record_natural_0 device* manifolds_23 [[buffer(10)]], packed_float4 device* loads_23 [[buffer(12)]], packed_uint4 device* bonds_23 [[buffer(2)]], uint device* args_23 [[buffer(13)]], uint device* info_23 [[buffer(15)]], ScanParams_0 constant* scan_23 [[buffer(14)]], packed_float4 device* parts_23 [[buffer(3)]], packed_float4 device* colliders_23 [[buffer(8)]], packed_uint4 device* collider_info_23 [[buffer(9)]], packed_float4 device* view_chunks_23 [[buffer(17)]], ViewParams_0 constant* view_23 [[buffer(16)]], packed_float4 device* objects_23 [[buffer(18)]], uint device* shot_bodies_23 [[buffer(20)]], float device* lines_23 [[buffer(19)]], packed_float4 device* geometry_23 [[buffer(21)]], packed_float4 device* shots_23 [[buffer(22)]], packed_float4 device* lp_23 [[buffer(24)]], packed_float4 device* ext_24 [[buffer(23)]], uint device* engine_counters_23 [[buffer(26)]], packed_float4 device* engine_contacts_23 [[buffer(25)]], packed_float4 device* cs_anchors_23 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_94;
    (&kernelContext_94)->params_0 = params_23;
    (&kernelContext_94)->work_0 = work_23;
    (&kernelContext_94)->frames_0 = frames_23;
    (&kernelContext_94)->bodies_0 = bodies_23;
    (&kernelContext_94)->chunks_0 = chunks_23;
    (&kernelContext_94)->sort_0 = sort_23;
    (&kernelContext_94)->atom_0 = atom_23;
    (&kernelContext_94)->manifolds_0 = manifolds_23;
    (&kernelContext_94)->loads_0 = loads_23;
    (&kernelContext_94)->bonds_0 = bonds_23;
    (&kernelContext_94)->args_0 = args_23;
    (&kernelContext_94)->info_0 = info_23;
    (&kernelContext_94)->scan_0 = scan_23;
    (&kernelContext_94)->parts_0 = parts_23;
    (&kernelContext_94)->colliders_0 = colliders_23;
    (&kernelContext_94)->collider_info_0 = collider_info_23;
    (&kernelContext_94)->view_chunks_0 = view_chunks_23;
    (&kernelContext_94)->view_0 = view_23;
    (&kernelContext_94)->objects_0 = objects_23;
    (&kernelContext_94)->shot_bodies_0 = shot_bodies_23;
    (&kernelContext_94)->lines_0 = lines_23;
    (&kernelContext_94)->geometry_0 = geometry_23;
    (&kernelContext_94)->shots_0 = shots_23;
    (&kernelContext_94)->lp_0 = lp_23;
    (&kernelContext_94)->ext_0 = ext_24;
    (&kernelContext_94)->engine_counters_0 = engine_counters_23;
    (&kernelContext_94)->engine_contacts_0 = engine_contacts_23;
    (&kernelContext_94)->cs_anchors_0 = cs_anchors_23;
    threadgroup array<uint, int(1024)> g_scan_23;
    (&kernelContext_94)->g_scan_0 = &g_scan_23;
    uint b_27 = id_20.x;
    if(b_27 >= (params_23->body_count_0))
    {
        return;
    }
    uint _S701 = a_strike_0(&kernelContext_94);
    uint at_7 = _S701 + 9U * b_27;
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
        atomic_store_explicit((&kernelContext_94)->atom_0+(at_7 + w_5), 4294967295U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_94)->atom_0+(at_7 + 3U + w_5), 0U, memory_order_relaxed);
        w_5 = w_5 + 1U;
    }
    atomic_store_explicit((&kernelContext_94)->atom_0+(at_7 + 6U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_94)->atom_0+(at_7 + 7U), 2139095040U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_94)->atom_0+(at_7 + 8U), 4294967295U, memory_order_relaxed);
    uint _S702 = strike_at_0(b_27, &kernelContext_94);
    (*((&kernelContext_94)->frames_0+_S702))[int(3)] = 0.0f;
    return;
}

uint ordered_0(float f_9)
{
    uint u_0 = (as_type<uint>((f_9)));
    uint _S703;
    if((u_0 & 2147483648U) != 0U)
    {
        _S703 = ~u_0;
    }
    else
    {
        _S703 = u_0 | 2147483648U;
    }
    return _S703;
}

[[kernel]] void fr_strike_bounds(uint3 id_21 [[thread_position_in_grid]], FractureParams_0 constant* params_24 [[buffer(0)]], uint device* work_24 [[buffer(4)]], packed_float4 device* frames_24 [[buffer(6)]], packed_float4 device* bodies_24 [[buffer(7)]], packed_float4 device* chunks_24 [[buffer(1)]], uint device* sort_24 [[buffer(11)]], atomic<uint> device* atom_24 [[buffer(5)]], Record_natural_0 device* manifolds_24 [[buffer(10)]], packed_float4 device* loads_24 [[buffer(12)]], packed_uint4 device* bonds_24 [[buffer(2)]], uint device* args_24 [[buffer(13)]], uint device* info_24 [[buffer(15)]], ScanParams_0 constant* scan_24 [[buffer(14)]], packed_float4 device* parts_24 [[buffer(3)]], packed_float4 device* colliders_24 [[buffer(8)]], packed_uint4 device* collider_info_24 [[buffer(9)]], packed_float4 device* view_chunks_24 [[buffer(17)]], ViewParams_0 constant* view_24 [[buffer(16)]], packed_float4 device* objects_24 [[buffer(18)]], uint device* shot_bodies_24 [[buffer(20)]], float device* lines_24 [[buffer(19)]], packed_float4 device* geometry_24 [[buffer(21)]], packed_float4 device* shots_24 [[buffer(22)]], packed_float4 device* lp_24 [[buffer(24)]], packed_float4 device* ext_25 [[buffer(23)]], uint device* engine_counters_24 [[buffer(26)]], packed_float4 device* engine_contacts_24 [[buffer(25)]], packed_float4 device* cs_anchors_24 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_95;
    (&kernelContext_95)->params_0 = params_24;
    (&kernelContext_95)->work_0 = work_24;
    (&kernelContext_95)->frames_0 = frames_24;
    (&kernelContext_95)->bodies_0 = bodies_24;
    (&kernelContext_95)->chunks_0 = chunks_24;
    (&kernelContext_95)->sort_0 = sort_24;
    (&kernelContext_95)->atom_0 = atom_24;
    (&kernelContext_95)->manifolds_0 = manifolds_24;
    (&kernelContext_95)->loads_0 = loads_24;
    (&kernelContext_95)->bonds_0 = bonds_24;
    (&kernelContext_95)->args_0 = args_24;
    (&kernelContext_95)->info_0 = info_24;
    (&kernelContext_95)->scan_0 = scan_24;
    (&kernelContext_95)->parts_0 = parts_24;
    (&kernelContext_95)->colliders_0 = colliders_24;
    (&kernelContext_95)->collider_info_0 = collider_info_24;
    (&kernelContext_95)->view_chunks_0 = view_chunks_24;
    (&kernelContext_95)->view_0 = view_24;
    (&kernelContext_95)->objects_0 = objects_24;
    (&kernelContext_95)->shot_bodies_0 = shot_bodies_24;
    (&kernelContext_95)->lines_0 = lines_24;
    (&kernelContext_95)->geometry_0 = geometry_24;
    (&kernelContext_95)->shots_0 = shots_24;
    (&kernelContext_95)->lp_0 = lp_24;
    (&kernelContext_95)->ext_0 = ext_25;
    (&kernelContext_95)->engine_counters_0 = engine_counters_24;
    (&kernelContext_95)->engine_contacts_0 = engine_contacts_24;
    (&kernelContext_95)->cs_anchors_0 = cs_anchors_24;
    threadgroup array<uint, int(1024)> g_scan_24;
    (&kernelContext_95)->g_scan_0 = &g_scan_24;
    uint k_18 = id_21.x;
    bool _S704;
    if(k_18 >= (params_24->chunk_count_0))
    {
        _S704 = true;
    }
    else
    {
        _S704 = ((&kernelContext_95)->params_0->shot_count_0) == 0U;
    }
    if(_S704)
    {
        return;
    }
    thread Quat_0 q_21;
    float3 _S705 = chunk_box_world_0(k_18, &q_21, &kernelContext_95);
    float _S706 = (float4(*((&kernelContext_95)->geometry_0+3U * k_18)) ).w;
    uint _S707 = a_strike_0(&kernelContext_95);
    uint _S708 = body_of_0(k_18, &kernelContext_95);
    uint _S709 = _S707 + 9U * _S708;
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
        uint _S710 = ax_1;
        uint _S711 = atomic_fetch_min_explicit((&kernelContext_95)->atom_0+(_S709 + ax_1), ordered_0(_S705[ax_1] - _S706), memory_order_relaxed);
        uint _S712 = atomic_fetch_max_explicit((&kernelContext_95)->atom_0+(_S709 + 3U + ax_1), ordered_0(_S705[_S710] + _S706), memory_order_relaxed);
        ax_1 = ax_1 + 1U;
    }
    return;
}

float unordered_0(uint u_1)
{
    uint _S713;
    if((u_1 & 2147483648U) != 0U)
    {
        _S713 = u_1 & 2147483647U;
    }
    else
    {
        _S713 = ~u_1;
    }
    return (as_type<float>((_S713)));
}

[[kernel]] void fr_strike_find(uint3 id_22 [[thread_position_in_grid]], FractureParams_0 constant* params_25 [[buffer(0)]], uint device* work_25 [[buffer(4)]], packed_float4 device* frames_25 [[buffer(6)]], packed_float4 device* bodies_25 [[buffer(7)]], packed_float4 device* chunks_25 [[buffer(1)]], uint device* sort_25 [[buffer(11)]], atomic<uint> device* atom_25 [[buffer(5)]], Record_natural_0 device* manifolds_25 [[buffer(10)]], packed_float4 device* loads_25 [[buffer(12)]], packed_uint4 device* bonds_25 [[buffer(2)]], uint device* args_25 [[buffer(13)]], uint device* info_25 [[buffer(15)]], ScanParams_0 constant* scan_25 [[buffer(14)]], packed_float4 device* parts_25 [[buffer(3)]], packed_float4 device* colliders_25 [[buffer(8)]], packed_uint4 device* collider_info_25 [[buffer(9)]], packed_float4 device* view_chunks_25 [[buffer(17)]], ViewParams_0 constant* view_25 [[buffer(16)]], packed_float4 device* objects_25 [[buffer(18)]], uint device* shot_bodies_25 [[buffer(20)]], float device* lines_25 [[buffer(19)]], packed_float4 device* geometry_25 [[buffer(21)]], packed_float4 device* shots_25 [[buffer(22)]], packed_float4 device* lp_25 [[buffer(24)]], packed_float4 device* ext_26 [[buffer(23)]], uint device* engine_counters_25 [[buffer(26)]], packed_float4 device* engine_contacts_25 [[buffer(25)]], packed_float4 device* cs_anchors_25 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_96;
    (&kernelContext_96)->params_0 = params_25;
    (&kernelContext_96)->work_0 = work_25;
    (&kernelContext_96)->frames_0 = frames_25;
    (&kernelContext_96)->bodies_0 = bodies_25;
    (&kernelContext_96)->chunks_0 = chunks_25;
    (&kernelContext_96)->sort_0 = sort_25;
    (&kernelContext_96)->atom_0 = atom_25;
    (&kernelContext_96)->manifolds_0 = manifolds_25;
    (&kernelContext_96)->loads_0 = loads_25;
    (&kernelContext_96)->bonds_0 = bonds_25;
    (&kernelContext_96)->args_0 = args_25;
    (&kernelContext_96)->info_0 = info_25;
    (&kernelContext_96)->scan_0 = scan_25;
    (&kernelContext_96)->parts_0 = parts_25;
    (&kernelContext_96)->colliders_0 = colliders_25;
    (&kernelContext_96)->collider_info_0 = collider_info_25;
    (&kernelContext_96)->view_chunks_0 = view_chunks_25;
    (&kernelContext_96)->view_0 = view_25;
    (&kernelContext_96)->objects_0 = objects_25;
    (&kernelContext_96)->shot_bodies_0 = shot_bodies_25;
    (&kernelContext_96)->lines_0 = lines_25;
    (&kernelContext_96)->geometry_0 = geometry_25;
    (&kernelContext_96)->shots_0 = shots_25;
    (&kernelContext_96)->lp_0 = lp_25;
    (&kernelContext_96)->ext_0 = ext_26;
    (&kernelContext_96)->engine_counters_0 = engine_counters_25;
    (&kernelContext_96)->engine_contacts_0 = engine_contacts_25;
    (&kernelContext_96)->cs_anchors_0 = cs_anchors_25;
    threadgroup array<uint, int(1024)> g_scan_25;
    (&kernelContext_96)->g_scan_0 = &g_scan_25;
    uint b_28 = id_22.x;
    bool _S714;
    if(b_28 >= (params_25->body_count_0))
    {
        _S714 = true;
    }
    else
    {
        _S714 = ((&kernelContext_96)->params_0->shot_count_0) == 0U;
    }
    if(_S714)
    {
        return;
    }
    uint _S715 = a_strike_0(&kernelContext_96);
    uint at_8 = _S715 + 9U * b_28;
    uint _S716 = atomic_load_explicit((&kernelContext_96)->atom_0+at_8, memory_order_relaxed);
    if(_S716 == 4294967295U)
    {
        return;
    }
    uint _S717 = atomic_load_explicit((&kernelContext_96)->atom_0+at_8, memory_order_relaxed);
    float _S718 = unordered_0(_S717);
    uint _S719 = atomic_load_explicit((&kernelContext_96)->atom_0+(at_8 + 1U), memory_order_relaxed);
    float _S720 = unordered_0(_S719);
    uint _S721 = atomic_load_explicit((&kernelContext_96)->atom_0+(at_8 + 2U), memory_order_relaxed);
    float3 _S722 = float3(_S718, _S720, unordered_0(_S721));
    uint _S723 = atomic_load_explicit((&kernelContext_96)->atom_0+(at_8 + 3U), memory_order_relaxed);
    float _S724 = unordered_0(_S723);
    uint _S725 = atomic_load_explicit((&kernelContext_96)->atom_0+(at_8 + 4U), memory_order_relaxed);
    float _S726 = unordered_0(_S725);
    uint _S727 = atomic_load_explicit((&kernelContext_96)->atom_0+(at_8 + 5U), memory_order_relaxed);
    float3 _S728 = float3(_S724, _S726, unordered_0(_S727));
    float3 _S729 = from_rubble_0((float4(*((&kernelContext_96)->bodies_0+(7U * b_28 + 3U))) ).xyz);
    float _S730 = float_bits_0(2139095040U);
    uint k_19 = 0U;
    for(;;)
    {
        if(k_19 < ((&kernelContext_96)->params_0->shot_count_0))
        {
        }
        else
        {
            break;
        }
        float4 _S731 = float4(*((&kernelContext_96)->shots_0+2U * k_19)) ;
        uint sb_1 = 7U * (as_type<uint>((_S731.x)));
        if(((float4(*((&kernelContext_96)->bodies_0+sb_1)) ).w) == 0.0f)
        {
            k_19 = k_19 + 1U;
            continue;
        }
        float3 c_18 = from_rubble_0((float4(*((&kernelContext_96)->bodies_0+sb_1)) ).xyz);
        float3 rel_2 = from_rubble_0((float4(*((&kernelContext_96)->bodies_0+(sb_1 + 3U))) ).xyz) - _S729;
        if(!((length(rel_2)) > ((&kernelContext_96)->params_0->strike_speed_0)))
        {
            k_19 = k_19 + 1U;
            continue;
        }
        float3 _S732 = float3(_S731.y) ;
        float3 _S733 = _S722 - _S732;
        float3 _S734 = _S728 + _S732;
        float _S735 = (&kernelContext_96)->params_0->dt_0;
        float t0_1 = 0.0f;
        float t1_1 = _S735;
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
            uint _S736 = ax_2;
            if((abs(rel_2[ax_2])) < 9.999999960041972e-13f)
            {
                uint _S737 = ax_2;
                if((c_18[ax_2]) < (_S733[ax_2]))
                {
                    _S714 = true;
                }
                else
                {
                    _S714 = (c_18[_S737]) > (_S734[ax_2]);
                }
                float t0_2;
                if(_S714)
                {
                    t0_2 = _S730;
                }
                else
                {
                    t0_2 = t0_1;
                }
                t0_1 = t0_2;
                ax_2 = ax_2 + 1U;
                continue;
            }
            float ta_1 = (_S733[ax_2] - c_18[ax_2]) / rel_2[_S736];
            float tb_1 = (_S734[ax_2] - c_18[ax_2]) / rel_2[_S736];
            float _S738 = min(t1_1, max(ta_1, tb_1));
            t0_1 = max(t0_1, min(ta_1, tb_1));
            t1_1 = _S738;
            ax_2 = ax_2 + 1U;
        }
        if(t0_1 <= t1_1)
        {
            uint _S739 = strike_at_0(b_28, &kernelContext_96);
            *((&kernelContext_96)->frames_0+_S739) = packed_float4(float4(c_18, float(k_19 + 1U))) ;
            *((&kernelContext_96)->frames_0+(_S739 + 1U)) = packed_float4(float4(rel_2, t0_1)) ;
            return;
        }
        k_19 = k_19 + 1U;
    }
    return;
}

[[kernel]] void fr_strike_entry(uint3 id_23 [[thread_position_in_grid]], FractureParams_0 constant* params_26 [[buffer(0)]], uint device* work_26 [[buffer(4)]], packed_float4 device* frames_26 [[buffer(6)]], packed_float4 device* bodies_26 [[buffer(7)]], packed_float4 device* chunks_26 [[buffer(1)]], uint device* sort_26 [[buffer(11)]], atomic<uint> device* atom_26 [[buffer(5)]], Record_natural_0 device* manifolds_26 [[buffer(10)]], packed_float4 device* loads_26 [[buffer(12)]], packed_uint4 device* bonds_26 [[buffer(2)]], uint device* args_26 [[buffer(13)]], uint device* info_26 [[buffer(15)]], ScanParams_0 constant* scan_26 [[buffer(14)]], packed_float4 device* parts_26 [[buffer(3)]], packed_float4 device* colliders_26 [[buffer(8)]], packed_uint4 device* collider_info_26 [[buffer(9)]], packed_float4 device* view_chunks_26 [[buffer(17)]], ViewParams_0 constant* view_26 [[buffer(16)]], packed_float4 device* objects_26 [[buffer(18)]], uint device* shot_bodies_26 [[buffer(20)]], float device* lines_26 [[buffer(19)]], packed_float4 device* geometry_26 [[buffer(21)]], packed_float4 device* shots_26 [[buffer(22)]], packed_float4 device* lp_26 [[buffer(24)]], packed_float4 device* ext_27 [[buffer(23)]], uint device* engine_counters_26 [[buffer(26)]], packed_float4 device* engine_contacts_26 [[buffer(25)]], packed_float4 device* cs_anchors_26 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_97;
    (&kernelContext_97)->params_0 = params_26;
    (&kernelContext_97)->work_0 = work_26;
    (&kernelContext_97)->frames_0 = frames_26;
    (&kernelContext_97)->bodies_0 = bodies_26;
    (&kernelContext_97)->chunks_0 = chunks_26;
    (&kernelContext_97)->sort_0 = sort_26;
    (&kernelContext_97)->atom_0 = atom_26;
    (&kernelContext_97)->manifolds_0 = manifolds_26;
    (&kernelContext_97)->loads_0 = loads_26;
    (&kernelContext_97)->bonds_0 = bonds_26;
    (&kernelContext_97)->args_0 = args_26;
    (&kernelContext_97)->info_0 = info_26;
    (&kernelContext_97)->scan_0 = scan_26;
    (&kernelContext_97)->parts_0 = parts_26;
    (&kernelContext_97)->colliders_0 = colliders_26;
    (&kernelContext_97)->collider_info_0 = collider_info_26;
    (&kernelContext_97)->view_chunks_0 = view_chunks_26;
    (&kernelContext_97)->view_0 = view_26;
    (&kernelContext_97)->objects_0 = objects_26;
    (&kernelContext_97)->shot_bodies_0 = shot_bodies_26;
    (&kernelContext_97)->lines_0 = lines_26;
    (&kernelContext_97)->geometry_0 = geometry_26;
    (&kernelContext_97)->shots_0 = shots_26;
    (&kernelContext_97)->lp_0 = lp_26;
    (&kernelContext_97)->ext_0 = ext_27;
    (&kernelContext_97)->engine_counters_0 = engine_counters_26;
    (&kernelContext_97)->engine_contacts_0 = engine_contacts_26;
    (&kernelContext_97)->cs_anchors_0 = cs_anchors_26;
    threadgroup array<uint, int(1024)> g_scan_26;
    (&kernelContext_97)->g_scan_0 = &g_scan_26;
    uint k_20 = id_23.x;
    bool _S740;
    if(k_20 >= (params_26->chunk_count_0))
    {
        _S740 = true;
    }
    else
    {
        _S740 = ((&kernelContext_97)->params_0->shot_count_0) == 0U;
    }
    if(_S740)
    {
        return;
    }
    uint _S741 = body_of_0(k_20, &kernelContext_97);
    thread uint shot_5;
    bool _S742 = struck_0(_S741, &shot_5, &kernelContext_97);
    if(!_S742)
    {
        return;
    }
    uint _S743 = strike_at_0(_S741, &kernelContext_97);
    float4 _S744 = float4(*((&kernelContext_97)->frames_0+(_S743 + 1U))) ;
    float3 from_3 = (float4(*((&kernelContext_97)->frames_0+_S743)) ).xyz;
    float3 rel_3 = _S744.xyz;
    float3 d_19 = rel_3 / float3(max(length(rel_3), 9.999999960041972e-13f)) ;
    float _S745 = (float4(*((&kernelContext_97)->shots_0+2U * shot_5)) ).y;
    thread Quat_0 q_22;
    float3 _S746 = chunk_box_world_0(k_20, &q_22, &kernelContext_97);
    uint _S747 = 3U * k_20;
    Quat_0 _S748 = quat_of_0(float4(*((&kernelContext_97)->geometry_0+(_S747 + 2U))) );
    thread Quat_0 _S749 = q_22;
    thread Quat_0 _S750 = _S748;
    Quat_0 _S751 = quat_mul_0(&_S749, &_S750);
    thread Quat_0 _S752 = _S751;
    Quat_0 _S753 = normalized_0(&_S752);
    float3 _S754 = (float4(*((&kernelContext_97)->geometry_0+(_S747 + 1U))) ).xyz;
    float _S755 = float_bits_0(2139095040U);
    float _S756 = (&kernelContext_97)->params_0->dt_0;
    float t_in_0 = 0.0f;
    float t_out_0 = _S756;
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
        thread Quat_0 _S757 = _S753;
        float3 _S758 = rotate_0(&_S757, e_2);
        float o_6 = dot(from_3 - _S746, _S758);
        float dv_1 = dot(rel_3, _S758);
        float ext_28 = _S754[ax_3] + _S745;
        if((abs(dv_1)) < 9.999999960041972e-13f)
        {
            float t_in_1;
            if((abs(o_6)) > ext_28)
            {
                t_in_1 = _S755;
            }
            else
            {
                t_in_1 = t_in_0;
            }
            t_in_0 = t_in_1;
            ax_3 = ax_3 + 1U;
            continue;
        }
        float ta_2 = (- ext_28 - o_6) / dv_1;
        float tb_2 = (ext_28 - o_6) / dv_1;
        float _S759 = min(t_out_0, max(ta_2, tb_2));
        t_in_0 = max(t_in_0, min(ta_2, tb_2));
        t_out_0 = _S759;
        ax_3 = ax_3 + 1U;
    }
    uint _S760 = a_strike_0(&kernelContext_97);
    uint at_9 = _S760 + 9U * _S741;
    if(t_in_0 <= t_out_0)
    {
        uint _S761 = atomic_fetch_min_explicit((&kernelContext_97)->atom_0+(at_9 + 6U), (as_type<uint>((t_in_0))), memory_order_relaxed);
    }
    float3 r_12 = _S746 - (from_3 + rel_3 * float3(_S744.w) );
    float along_1 = dot(r_12, d_19);
    uint _S762 = atomic_fetch_min_explicit((&kernelContext_97)->atom_0+(at_9 + 7U), (as_type<uint>((length(r_12 - d_19 * float3(along_1) ) + abs(along_1) * 0.00100000004749745f))), memory_order_relaxed);
    return;
}

[[kernel]] void fr_strike_nearest(uint3 id_24 [[thread_position_in_grid]], FractureParams_0 constant* params_27 [[buffer(0)]], uint device* work_27 [[buffer(4)]], packed_float4 device* frames_27 [[buffer(6)]], packed_float4 device* bodies_27 [[buffer(7)]], packed_float4 device* chunks_27 [[buffer(1)]], uint device* sort_27 [[buffer(11)]], atomic<uint> device* atom_27 [[buffer(5)]], Record_natural_0 device* manifolds_27 [[buffer(10)]], packed_float4 device* loads_27 [[buffer(12)]], packed_uint4 device* bonds_27 [[buffer(2)]], uint device* args_27 [[buffer(13)]], uint device* info_27 [[buffer(15)]], ScanParams_0 constant* scan_27 [[buffer(14)]], packed_float4 device* parts_27 [[buffer(3)]], packed_float4 device* colliders_27 [[buffer(8)]], packed_uint4 device* collider_info_27 [[buffer(9)]], packed_float4 device* view_chunks_27 [[buffer(17)]], ViewParams_0 constant* view_27 [[buffer(16)]], packed_float4 device* objects_27 [[buffer(18)]], uint device* shot_bodies_27 [[buffer(20)]], float device* lines_27 [[buffer(19)]], packed_float4 device* geometry_27 [[buffer(21)]], packed_float4 device* shots_27 [[buffer(22)]], packed_float4 device* lp_27 [[buffer(24)]], packed_float4 device* ext_29 [[buffer(23)]], uint device* engine_counters_27 [[buffer(26)]], packed_float4 device* engine_contacts_27 [[buffer(25)]], packed_float4 device* cs_anchors_27 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_98;
    (&kernelContext_98)->params_0 = params_27;
    (&kernelContext_98)->work_0 = work_27;
    (&kernelContext_98)->frames_0 = frames_27;
    (&kernelContext_98)->bodies_0 = bodies_27;
    (&kernelContext_98)->chunks_0 = chunks_27;
    (&kernelContext_98)->sort_0 = sort_27;
    (&kernelContext_98)->atom_0 = atom_27;
    (&kernelContext_98)->manifolds_0 = manifolds_27;
    (&kernelContext_98)->loads_0 = loads_27;
    (&kernelContext_98)->bonds_0 = bonds_27;
    (&kernelContext_98)->args_0 = args_27;
    (&kernelContext_98)->info_0 = info_27;
    (&kernelContext_98)->scan_0 = scan_27;
    (&kernelContext_98)->parts_0 = parts_27;
    (&kernelContext_98)->colliders_0 = colliders_27;
    (&kernelContext_98)->collider_info_0 = collider_info_27;
    (&kernelContext_98)->view_chunks_0 = view_chunks_27;
    (&kernelContext_98)->view_0 = view_27;
    (&kernelContext_98)->objects_0 = objects_27;
    (&kernelContext_98)->shot_bodies_0 = shot_bodies_27;
    (&kernelContext_98)->lines_0 = lines_27;
    (&kernelContext_98)->geometry_0 = geometry_27;
    (&kernelContext_98)->shots_0 = shots_27;
    (&kernelContext_98)->lp_0 = lp_27;
    (&kernelContext_98)->ext_0 = ext_29;
    (&kernelContext_98)->engine_counters_0 = engine_counters_27;
    (&kernelContext_98)->engine_contacts_0 = engine_contacts_27;
    (&kernelContext_98)->cs_anchors_0 = cs_anchors_27;
    threadgroup array<uint, int(1024)> g_scan_27;
    (&kernelContext_98)->g_scan_0 = &g_scan_27;
    uint k_21 = id_24.x;
    bool _S763;
    if(k_21 >= (params_27->chunk_count_0))
    {
        _S763 = true;
    }
    else
    {
        _S763 = ((&kernelContext_98)->params_0->shot_count_0) == 0U;
    }
    if(_S763)
    {
        return;
    }
    uint _S764 = body_of_0(k_21, &kernelContext_98);
    thread uint shot_6;
    bool _S765 = struck_0(_S764, &shot_6, &kernelContext_98);
    if(!_S765)
    {
        return;
    }
    uint _S766 = strike_at_0(_S764, &kernelContext_98);
    float4 _S767 = float4(*((&kernelContext_98)->frames_0+_S766)) ;
    float4 _S768 = float4(*((&kernelContext_98)->frames_0+(_S766 + 1U))) ;
    float3 _S769 = _S768.xyz;
    float3 d_20 = _S769 / float3(max(length(_S769), 9.999999960041972e-13f)) ;
    thread Quat_0 q_23;
    float3 _S770 = chunk_box_world_0(k_21, &q_23, &kernelContext_98);
    float3 r_13 = _S770 - (_S767.xyz + _S769 * float3(_S768.w) );
    float along_2 = dot(r_13, d_20);
    float dist_0 = length(r_13 - d_20 * float3(along_2) ) + abs(along_2) * 0.00100000004749745f;
    uint _S771 = a_strike_0(&kernelContext_98);
    uint at_10 = _S771 + 9U * _S764;
    uint _S772 = (as_type<uint>((dist_0)));
    uint _S773 = atomic_load_explicit((&kernelContext_98)->atom_0+(at_10 + 7U), memory_order_relaxed);
    if(_S772 == _S773)
    {
        uint _S774 = atomic_fetch_min_explicit((&kernelContext_98)->atom_0+(at_10 + 8U), k_21, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_strike_force(uint3 id_25 [[thread_position_in_grid]], FractureParams_0 constant* params_28 [[buffer(0)]], uint device* work_28 [[buffer(4)]], packed_float4 device* frames_28 [[buffer(6)]], packed_float4 device* bodies_28 [[buffer(7)]], packed_float4 device* chunks_28 [[buffer(1)]], uint device* sort_28 [[buffer(11)]], atomic<uint> device* atom_28 [[buffer(5)]], Record_natural_0 device* manifolds_28 [[buffer(10)]], packed_float4 device* loads_28 [[buffer(12)]], packed_uint4 device* bonds_28 [[buffer(2)]], uint device* args_28 [[buffer(13)]], uint device* info_28 [[buffer(15)]], ScanParams_0 constant* scan_28 [[buffer(14)]], packed_float4 device* parts_28 [[buffer(3)]], packed_float4 device* colliders_28 [[buffer(8)]], packed_uint4 device* collider_info_28 [[buffer(9)]], packed_float4 device* view_chunks_28 [[buffer(17)]], ViewParams_0 constant* view_28 [[buffer(16)]], packed_float4 device* objects_28 [[buffer(18)]], uint device* shot_bodies_28 [[buffer(20)]], float device* lines_28 [[buffer(19)]], packed_float4 device* geometry_28 [[buffer(21)]], packed_float4 device* shots_28 [[buffer(22)]], packed_float4 device* lp_28 [[buffer(24)]], packed_float4 device* ext_30 [[buffer(23)]], uint device* engine_counters_28 [[buffer(26)]], packed_float4 device* engine_contacts_28 [[buffer(25)]], packed_float4 device* cs_anchors_28 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_28;
    (&kernelContext_99)->work_0 = work_28;
    (&kernelContext_99)->frames_0 = frames_28;
    (&kernelContext_99)->bodies_0 = bodies_28;
    (&kernelContext_99)->chunks_0 = chunks_28;
    (&kernelContext_99)->sort_0 = sort_28;
    (&kernelContext_99)->atom_0 = atom_28;
    (&kernelContext_99)->manifolds_0 = manifolds_28;
    (&kernelContext_99)->loads_0 = loads_28;
    (&kernelContext_99)->bonds_0 = bonds_28;
    (&kernelContext_99)->args_0 = args_28;
    (&kernelContext_99)->info_0 = info_28;
    (&kernelContext_99)->scan_0 = scan_28;
    (&kernelContext_99)->parts_0 = parts_28;
    (&kernelContext_99)->colliders_0 = colliders_28;
    (&kernelContext_99)->collider_info_0 = collider_info_28;
    (&kernelContext_99)->view_chunks_0 = view_chunks_28;
    (&kernelContext_99)->view_0 = view_28;
    (&kernelContext_99)->objects_0 = objects_28;
    (&kernelContext_99)->shot_bodies_0 = shot_bodies_28;
    (&kernelContext_99)->lines_0 = lines_28;
    (&kernelContext_99)->geometry_0 = geometry_28;
    (&kernelContext_99)->shots_0 = shots_28;
    (&kernelContext_99)->lp_0 = lp_28;
    (&kernelContext_99)->ext_0 = ext_30;
    (&kernelContext_99)->engine_counters_0 = engine_counters_28;
    (&kernelContext_99)->engine_contacts_0 = engine_contacts_28;
    (&kernelContext_99)->cs_anchors_0 = cs_anchors_28;
    threadgroup array<uint, int(1024)> g_scan_28;
    (&kernelContext_99)->g_scan_0 = &g_scan_28;
    uint b_29 = id_25.x;
    bool _S775;
    if(b_29 >= (params_28->body_count_0))
    {
        _S775 = true;
    }
    else
    {
        _S775 = ((&kernelContext_99)->params_0->shot_count_0) == 0U;
    }
    if(_S775)
    {
        return;
    }
    thread uint shot_7;
    bool _S776 = struck_0(b_29, &shot_7, &kernelContext_99);
    if(!_S776)
    {
        return;
    }
    uint _S777 = a_strike_0(&kernelContext_99);
    uint at_11 = _S777 + 9U * b_29;
    uint _S778 = atomic_load_explicit((&kernelContext_99)->atom_0+(at_11 + 6U), memory_order_relaxed);
    float t_in_2 = (as_type<float>((_S778)));
    uint entry_2 = atomic_load_explicit((&kernelContext_99)->atom_0+(at_11 + 8U), memory_order_relaxed);
    uint _S779 = strike_at_0(b_29, &kernelContext_99);
    if(!(t_in_2 <= ((&kernelContext_99)->params_0->dt_0)))
    {
        _S775 = true;
    }
    else
    {
        _S775 = entry_2 == 4294967295U;
    }
    if(_S775)
    {
        (*((&kernelContext_99)->frames_0+_S779))[int(3)] = 0.0f;
        return;
    }
    float3 rel_4 = (float4(*((&kernelContext_99)->frames_0+(_S779 + 1U))) ).xyz;
    float speed_14 = length(rel_4);
    float4 _S780 = float4(*((&kernelContext_99)->shots_0+2U * shot_7)) ;
    float _S781 = max(_S780.z, 0.00100000004749745f);
    float m_imp_0 = _S780.w;
    float e_imp_0 = (float4(*((&kernelContext_99)->shots_0+(2U * shot_7 + 1U))) ).x;
    float3 contact_16 = (float4(*((&kernelContext_99)->frames_0+_S779)) ).xyz + rel_4 * float3(t_in_2)  + rel_4 / float3(max(speed_14, 9.999999960041972e-13f))  * float3(_S780.y) ;
    float inv_mass_5 = (float4(*((&kernelContext_99)->bodies_0+7U * b_29)) ).w;
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
    float m_eff_1;
    if(anchored_1)
    {
        m_eff_1 = m_imp_0;
    }
    else
    {
        m_eff_1 = m_imp_0 * m_body_1 / (m_imp_0 + m_body_1);
    }
    float modulus_1 = 1.0f / (1.0f / e_imp_0 + 1.0f / (float4(*((&kernelContext_99)->geometry_0+(3U * entry_2 + 1U))) ).w);
    *((&kernelContext_99)->frames_0+(_S779 + 2U)) = packed_float4(float4(contact_16, m_eff_1 * speed_14 / (2.86999988555908203f * pow(m_eff_1 * m_eff_1 / (_S781 * modulus_1 * modulus_1 * max(speed_14, 0.00100000004749745f)), 0.20000000298023224f)))) ;
    return;
}

bool strike_share_breaks_0(uint4 bond_0, float3 contact_17, float force_2, KernelContext_0 thread* kernelContext_100)
{
    thread Quat_0 q_24;
    float3 _S782 = chunk_box_world_0(bond_0.x, &q_24, kernelContext_100);
    float3 _S783 = chunk_box_world_0(bond_0.y, &q_24, kernelContext_100);
    float dist_1 = length(float3(0.5f)  * (_S782 + _S783) - contact_17);
    return (force_2 * min(1.0f, (as_type<float>((bond_0.w))) / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (as_type<float>((bond_0.z)));
}

[[kernel]] void fr_strike_breaks(uint3 id_26 [[thread_position_in_grid]], FractureParams_0 constant* params_29 [[buffer(0)]], uint device* work_29 [[buffer(4)]], packed_float4 device* frames_29 [[buffer(6)]], packed_float4 device* bodies_29 [[buffer(7)]], packed_float4 device* chunks_29 [[buffer(1)]], uint device* sort_29 [[buffer(11)]], atomic<uint> device* atom_29 [[buffer(5)]], Record_natural_0 device* manifolds_29 [[buffer(10)]], packed_float4 device* loads_29 [[buffer(12)]], packed_uint4 device* bonds_29 [[buffer(2)]], uint device* args_29 [[buffer(13)]], uint device* info_29 [[buffer(15)]], ScanParams_0 constant* scan_29 [[buffer(14)]], packed_float4 device* parts_29 [[buffer(3)]], packed_float4 device* colliders_29 [[buffer(8)]], packed_uint4 device* collider_info_29 [[buffer(9)]], packed_float4 device* view_chunks_29 [[buffer(17)]], ViewParams_0 constant* view_29 [[buffer(16)]], packed_float4 device* objects_29 [[buffer(18)]], uint device* shot_bodies_29 [[buffer(20)]], float device* lines_29 [[buffer(19)]], packed_float4 device* geometry_29 [[buffer(21)]], packed_float4 device* shots_29 [[buffer(22)]], packed_float4 device* lp_29 [[buffer(24)]], packed_float4 device* ext_31 [[buffer(23)]], uint device* engine_counters_29 [[buffer(26)]], packed_float4 device* engine_contacts_29 [[buffer(25)]], packed_float4 device* cs_anchors_29 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_101;
    (&kernelContext_101)->params_0 = params_29;
    (&kernelContext_101)->work_0 = work_29;
    (&kernelContext_101)->frames_0 = frames_29;
    (&kernelContext_101)->bodies_0 = bodies_29;
    (&kernelContext_101)->chunks_0 = chunks_29;
    (&kernelContext_101)->sort_0 = sort_29;
    (&kernelContext_101)->atom_0 = atom_29;
    (&kernelContext_101)->manifolds_0 = manifolds_29;
    (&kernelContext_101)->loads_0 = loads_29;
    (&kernelContext_101)->bonds_0 = bonds_29;
    (&kernelContext_101)->args_0 = args_29;
    (&kernelContext_101)->info_0 = info_29;
    (&kernelContext_101)->scan_0 = scan_29;
    (&kernelContext_101)->parts_0 = parts_29;
    (&kernelContext_101)->colliders_0 = colliders_29;
    (&kernelContext_101)->collider_info_0 = collider_info_29;
    (&kernelContext_101)->view_chunks_0 = view_chunks_29;
    (&kernelContext_101)->view_0 = view_29;
    (&kernelContext_101)->objects_0 = objects_29;
    (&kernelContext_101)->shot_bodies_0 = shot_bodies_29;
    (&kernelContext_101)->lines_0 = lines_29;
    (&kernelContext_101)->geometry_0 = geometry_29;
    (&kernelContext_101)->shots_0 = shots_29;
    (&kernelContext_101)->lp_0 = lp_29;
    (&kernelContext_101)->ext_0 = ext_31;
    (&kernelContext_101)->engine_counters_0 = engine_counters_29;
    (&kernelContext_101)->engine_contacts_0 = engine_contacts_29;
    (&kernelContext_101)->cs_anchors_0 = cs_anchors_29;
    threadgroup array<uint, int(1024)> g_scan_29;
    (&kernelContext_101)->g_scan_0 = &g_scan_29;
    uint i_7 = id_26.x;
    bool _S784;
    if(i_7 >= (params_29->bond_count_0))
    {
        _S784 = true;
    }
    else
    {
        _S784 = ((&kernelContext_101)->params_0->shot_count_0) == 0U;
    }
    if(_S784)
    {
        _S784 = true;
    }
    else
    {
        uint _S785 = w_alive_0(&kernelContext_101);
        _S784 = (*((&kernelContext_101)->work_0+(_S785 + i_7))) == 0U;
    }
    if(_S784)
    {
        return;
    }
    uint4 _S786 = uint4(*((&kernelContext_101)->bonds_0+i_7)) ;
    uint _S787 = body_of_0(_S786.x, &kernelContext_101);
    thread uint shot_8;
    bool _S788 = struck_0(_S787, &shot_8, &kernelContext_101);
    if(!_S788)
    {
        return;
    }
    uint _S789 = strike_at_0(_S787, &kernelContext_101);
    float4 _S790 = float4(*((&kernelContext_101)->frames_0+(_S789 + 2U))) ;
    bool _S791 = strike_share_breaks_0(_S786, _S790.xyz, _S790.w, &kernelContext_101);
    if(_S791)
    {
        uint _S792 = w_alive_0(&kernelContext_101);
        *((&kernelContext_101)->work_0+(_S792 + i_7)) = 0U;
        uint _S793 = a_dirty_0(&kernelContext_101);
        uint _S794 = atomic_fetch_or_explicit((&kernelContext_101)->atom_0+(_S793 + _S787), 1U, memory_order_relaxed);
        uint _S795 = a_counters_1(&kernelContext_101);
        uint _S796 = atomic_fetch_add_explicit((&kernelContext_101)->atom_0+_S795, 1U, memory_order_relaxed);
        uint _S797 = atomic_fetch_add_explicit((&kernelContext_101)->atom_0+(_S795 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

uint a_lp_0(uint b_30, KernelContext_0 thread* kernelContext_102)
{
    uint _S798 = a_strike_0(kernelContext_102);
    return _S798 + 9U * kernelContext_102->params_0->body_count_0 + 14U * b_30;
}

[[kernel]] void fr_lp_clear(uint3 id_27 [[thread_position_in_grid]], FractureParams_0 constant* params_30 [[buffer(0)]], uint device* work_30 [[buffer(4)]], packed_float4 device* frames_30 [[buffer(6)]], packed_float4 device* bodies_30 [[buffer(7)]], packed_float4 device* chunks_30 [[buffer(1)]], uint device* sort_30 [[buffer(11)]], atomic<uint> device* atom_30 [[buffer(5)]], Record_natural_0 device* manifolds_30 [[buffer(10)]], packed_float4 device* loads_30 [[buffer(12)]], packed_uint4 device* bonds_30 [[buffer(2)]], uint device* args_30 [[buffer(13)]], uint device* info_30 [[buffer(15)]], ScanParams_0 constant* scan_30 [[buffer(14)]], packed_float4 device* parts_30 [[buffer(3)]], packed_float4 device* colliders_30 [[buffer(8)]], packed_uint4 device* collider_info_30 [[buffer(9)]], packed_float4 device* view_chunks_30 [[buffer(17)]], ViewParams_0 constant* view_30 [[buffer(16)]], packed_float4 device* objects_30 [[buffer(18)]], uint device* shot_bodies_30 [[buffer(20)]], float device* lines_30 [[buffer(19)]], packed_float4 device* geometry_30 [[buffer(21)]], packed_float4 device* shots_30 [[buffer(22)]], packed_float4 device* lp_30 [[buffer(24)]], packed_float4 device* ext_32 [[buffer(23)]], uint device* engine_counters_30 [[buffer(26)]], packed_float4 device* engine_contacts_30 [[buffer(25)]], packed_float4 device* cs_anchors_30 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_103;
    (&kernelContext_103)->params_0 = params_30;
    (&kernelContext_103)->work_0 = work_30;
    (&kernelContext_103)->frames_0 = frames_30;
    (&kernelContext_103)->bodies_0 = bodies_30;
    (&kernelContext_103)->chunks_0 = chunks_30;
    (&kernelContext_103)->sort_0 = sort_30;
    (&kernelContext_103)->atom_0 = atom_30;
    (&kernelContext_103)->manifolds_0 = manifolds_30;
    (&kernelContext_103)->loads_0 = loads_30;
    (&kernelContext_103)->bonds_0 = bonds_30;
    (&kernelContext_103)->args_0 = args_30;
    (&kernelContext_103)->info_0 = info_30;
    (&kernelContext_103)->scan_0 = scan_30;
    (&kernelContext_103)->parts_0 = parts_30;
    (&kernelContext_103)->colliders_0 = colliders_30;
    (&kernelContext_103)->collider_info_0 = collider_info_30;
    (&kernelContext_103)->view_chunks_0 = view_chunks_30;
    (&kernelContext_103)->view_0 = view_30;
    (&kernelContext_103)->objects_0 = objects_30;
    (&kernelContext_103)->shot_bodies_0 = shot_bodies_30;
    (&kernelContext_103)->lines_0 = lines_30;
    (&kernelContext_103)->geometry_0 = geometry_30;
    (&kernelContext_103)->shots_0 = shots_30;
    (&kernelContext_103)->lp_0 = lp_30;
    (&kernelContext_103)->ext_0 = ext_32;
    (&kernelContext_103)->engine_counters_0 = engine_counters_30;
    (&kernelContext_103)->engine_contacts_0 = engine_contacts_30;
    (&kernelContext_103)->cs_anchors_0 = cs_anchors_30;
    threadgroup array<uint, int(1024)> g_scan_30;
    (&kernelContext_103)->g_scan_0 = &g_scan_30;
    uint b_31 = id_27.x;
    if(b_31 >= (params_30->body_count_0))
    {
        return;
    }
    uint _S799 = a_lp_0(b_31, &kernelContext_103);
    uint w_6 = 0U;
    for(;;)
    {
        if(w_6 < 12U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit((&kernelContext_103)->atom_0+(_S799 + w_6), 0U, memory_order_relaxed);
        w_6 = w_6 + 1U;
    }
    atomic_store_explicit((&kernelContext_103)->atom_0+(_S799 + 12U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_103)->atom_0+(_S799 + 13U), 0U, memory_order_relaxed);
    return;
}

uint st_f_0(uint k_22, uint state_0)
{
    return 22U * k_22 + state_0 + 6U;
}

uint lp_f_0(uint k_23)
{
    return st_f_0(k_23, 0U);
}

void add64w_0(uint at_12, float value_1, float scale_3, KernelContext_0 thread* kernelContext_104)
{
    float x_6 = round(value_1 * scale_3);
    float x_7;
    if(!((abs(x_6)) < 3.99999993722699776e+18f))
    {
        uint _S800 = a_counters_1(kernelContext_104);
        uint _S801 = atomic_fetch_or_explicit(kernelContext_104->atom_0+(_S800 + 3U), 2U, memory_order_relaxed);
        x_7 = 0.0f;
    }
    else
    {
        x_7 = x_6;
    }
    float u_2 = abs(x_7);
    float h_5 = floor(u_2 * 2.32830643653869629e-10f);
    uint lo_3 = uint(u_2 - h_5 * 4.294967296e+09f);
    uint hi_3 = uint(h_5);
    uint lo_4;
    uint hi_4;
    if(x_7 < 0.0f)
    {
        uint lo_5 = ~lo_3 + 1U;
        uint _S802 = ~hi_3;
        if(lo_5 == 0U)
        {
            lo_4 = 1U;
        }
        else
        {
            lo_4 = 0U;
        }
        uint _S803 = _S802 + lo_4;
        lo_4 = lo_5;
        hi_4 = _S803;
    }
    else
    {
        lo_4 = lo_3;
        hi_4 = hi_3;
    }
    uint old_1 = atomic_fetch_add_explicit(kernelContext_104->atom_0+at_12, lo_4, memory_order_relaxed);
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
        uint _S804 = atomic_fetch_add_explicit(kernelContext_104->atom_0+(at_12 + 1U), hi_5, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_lp_sum(uint3 id_28 [[thread_position_in_grid]], FractureParams_0 constant* params_31 [[buffer(0)]], uint device* work_31 [[buffer(4)]], packed_float4 device* frames_31 [[buffer(6)]], packed_float4 device* bodies_31 [[buffer(7)]], packed_float4 device* chunks_31 [[buffer(1)]], uint device* sort_31 [[buffer(11)]], atomic<uint> device* atom_31 [[buffer(5)]], Record_natural_0 device* manifolds_31 [[buffer(10)]], packed_float4 device* loads_31 [[buffer(12)]], packed_uint4 device* bonds_31 [[buffer(2)]], uint device* args_31 [[buffer(13)]], uint device* info_31 [[buffer(15)]], ScanParams_0 constant* scan_31 [[buffer(14)]], packed_float4 device* parts_31 [[buffer(3)]], packed_float4 device* colliders_31 [[buffer(8)]], packed_uint4 device* collider_info_31 [[buffer(9)]], packed_float4 device* view_chunks_31 [[buffer(17)]], ViewParams_0 constant* view_31 [[buffer(16)]], packed_float4 device* objects_31 [[buffer(18)]], uint device* shot_bodies_31 [[buffer(20)]], float device* lines_31 [[buffer(19)]], packed_float4 device* geometry_31 [[buffer(21)]], packed_float4 device* shots_31 [[buffer(22)]], packed_float4 device* lp_31 [[buffer(24)]], packed_float4 device* ext_33 [[buffer(23)]], uint device* engine_counters_31 [[buffer(26)]], packed_float4 device* engine_contacts_31 [[buffer(25)]], packed_float4 device* cs_anchors_31 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_105;
    (&kernelContext_105)->params_0 = params_31;
    (&kernelContext_105)->work_0 = work_31;
    (&kernelContext_105)->frames_0 = frames_31;
    (&kernelContext_105)->bodies_0 = bodies_31;
    (&kernelContext_105)->chunks_0 = chunks_31;
    (&kernelContext_105)->sort_0 = sort_31;
    (&kernelContext_105)->atom_0 = atom_31;
    (&kernelContext_105)->manifolds_0 = manifolds_31;
    (&kernelContext_105)->loads_0 = loads_31;
    (&kernelContext_105)->bonds_0 = bonds_31;
    (&kernelContext_105)->args_0 = args_31;
    (&kernelContext_105)->info_0 = info_31;
    (&kernelContext_105)->scan_0 = scan_31;
    (&kernelContext_105)->parts_0 = parts_31;
    (&kernelContext_105)->colliders_0 = colliders_31;
    (&kernelContext_105)->collider_info_0 = collider_info_31;
    (&kernelContext_105)->view_chunks_0 = view_chunks_31;
    (&kernelContext_105)->view_0 = view_31;
    (&kernelContext_105)->objects_0 = objects_31;
    (&kernelContext_105)->shot_bodies_0 = shot_bodies_31;
    (&kernelContext_105)->lines_0 = lines_31;
    (&kernelContext_105)->geometry_0 = geometry_31;
    (&kernelContext_105)->shots_0 = shots_31;
    (&kernelContext_105)->lp_0 = lp_31;
    (&kernelContext_105)->ext_0 = ext_33;
    (&kernelContext_105)->engine_counters_0 = engine_counters_31;
    (&kernelContext_105)->engine_contacts_0 = engine_contacts_31;
    (&kernelContext_105)->cs_anchors_0 = cs_anchors_31;
    threadgroup array<uint, int(1024)> g_scan_31;
    (&kernelContext_105)->g_scan_0 = &g_scan_31;
    uint k_24 = id_28.x;
    if(k_24 >= (params_31->chunk_count_0))
    {
        return;
    }
    uint _S805 = body_of_0(k_24, &kernelContext_105);
    thread Quat_0 q_25;
    thread float3 origin_7;
    structure_pose_0(_S805, 7U * _S805, &q_25, &origin_7, &kernelContext_105);
    thread Quat_0 _S806 = q_25;
    Quat_0 _S807 = conj_0(&_S806);
    float4 _S808 = float4(*((&kernelContext_105)->chunks_0+4U * k_24)) ;
    uint _S809 = l_vec_0(&kernelContext_105);
    uint _S810 = _S809 + 2U * k_24;
    float3 _S811 = (float4(*((&kernelContext_105)->loads_0+_S810)) ).xyz;
    thread Quat_0 _S812 = _S807;
    float3 _S813 = rotate_0(&_S812, _S811);
    float _S814 = _S808.w;
    float3 _S815 = float3((&kernelContext_105)->params_0->gx_0, (&kernelContext_105)->params_0->gy_0, (&kernelContext_105)->params_0->gz_0);
    thread Quat_0 _S816 = _S807;
    float3 _S817 = rotate_0(&_S816, _S815);
    float3 f_10 = _S813 + float3(_S814)  * _S817;
    float3 _S818 = (float4(*((&kernelContext_105)->loads_0+(_S810 + 1U))) ).xyz;
    thread Quat_0 _S819 = _S807;
    float3 _S820 = rotate_0(&_S819, _S818);
    uint _S821 = lp_f_0(k_24);
    *((&kernelContext_105)->lp_0+_S821) = packed_float4(float4(f_10, 0.0f)) ;
    *((&kernelContext_105)->lp_0+(_S821 + 1U)) = packed_float4(float4(_S820, 0.0f)) ;
    float3 _S822 = _S820 + cross(_S808.xyz - (float4(*((&kernelContext_105)->frames_0+2U * _S805)) ).xyz, f_10);
    uint _S823 = a_lp_0(_S805, &kernelContext_105);
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
        uint _S824 = 2U * a_7;
        add64w_0(_S823 + _S824, f_10[a_7], (&kernelContext_105)->params_0->lp_scale_0, &kernelContext_105);
        add64w_0(_S823 + 6U + _S824, _S822[a_7], (&kernelContext_105)->params_0->lp_scale_0, &kernelContext_105);
        a_7 = a_7 + 1U;
    }
    uint _S825 = atomic_fetch_min_explicit((&kernelContext_105)->atom_0+(_S823 + 12U), k_24, memory_order_relaxed);
    if(((&kernelContext_105)->info_0[k_24] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_105)->atom_0+(_S823 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

float3 read64w3_0(uint at_13, float scale_4, KernelContext_0 thread* kernelContext_106)
{
    float _S826 = read64_1(at_13, scale_4, kernelContext_106);
    float _S827 = read64_1(at_13 + 2U, scale_4, kernelContext_106);
    float _S828 = read64_1(at_13 + 4U, scale_4, kernelContext_106);
    return float3(_S826, _S827, _S828);
}

[[kernel]] void fr_lp_relief(uint3 id_29 [[thread_position_in_grid]], FractureParams_0 constant* params_32 [[buffer(0)]], uint device* work_32 [[buffer(4)]], packed_float4 device* frames_32 [[buffer(6)]], packed_float4 device* bodies_32 [[buffer(7)]], packed_float4 device* chunks_32 [[buffer(1)]], uint device* sort_32 [[buffer(11)]], atomic<uint> device* atom_32 [[buffer(5)]], Record_natural_0 device* manifolds_32 [[buffer(10)]], packed_float4 device* loads_32 [[buffer(12)]], packed_uint4 device* bonds_32 [[buffer(2)]], uint device* args_32 [[buffer(13)]], uint device* info_32 [[buffer(15)]], ScanParams_0 constant* scan_32 [[buffer(14)]], packed_float4 device* parts_32 [[buffer(3)]], packed_float4 device* colliders_32 [[buffer(8)]], packed_uint4 device* collider_info_32 [[buffer(9)]], packed_float4 device* view_chunks_32 [[buffer(17)]], ViewParams_0 constant* view_32 [[buffer(16)]], packed_float4 device* objects_32 [[buffer(18)]], uint device* shot_bodies_32 [[buffer(20)]], float device* lines_32 [[buffer(19)]], packed_float4 device* geometry_32 [[buffer(21)]], packed_float4 device* shots_32 [[buffer(22)]], packed_float4 device* lp_32 [[buffer(24)]], packed_float4 device* ext_34 [[buffer(23)]], uint device* engine_counters_32 [[buffer(26)]], packed_float4 device* engine_contacts_32 [[buffer(25)]], packed_float4 device* cs_anchors_32 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_107;
    (&kernelContext_107)->params_0 = params_32;
    (&kernelContext_107)->work_0 = work_32;
    (&kernelContext_107)->frames_0 = frames_32;
    (&kernelContext_107)->bodies_0 = bodies_32;
    (&kernelContext_107)->chunks_0 = chunks_32;
    (&kernelContext_107)->sort_0 = sort_32;
    (&kernelContext_107)->atom_0 = atom_32;
    (&kernelContext_107)->manifolds_0 = manifolds_32;
    (&kernelContext_107)->loads_0 = loads_32;
    (&kernelContext_107)->bonds_0 = bonds_32;
    (&kernelContext_107)->args_0 = args_32;
    (&kernelContext_107)->info_0 = info_32;
    (&kernelContext_107)->scan_0 = scan_32;
    (&kernelContext_107)->parts_0 = parts_32;
    (&kernelContext_107)->colliders_0 = colliders_32;
    (&kernelContext_107)->collider_info_0 = collider_info_32;
    (&kernelContext_107)->view_chunks_0 = view_chunks_32;
    (&kernelContext_107)->view_0 = view_32;
    (&kernelContext_107)->objects_0 = objects_32;
    (&kernelContext_107)->shot_bodies_0 = shot_bodies_32;
    (&kernelContext_107)->lines_0 = lines_32;
    (&kernelContext_107)->geometry_0 = geometry_32;
    (&kernelContext_107)->shots_0 = shots_32;
    (&kernelContext_107)->lp_0 = lp_32;
    (&kernelContext_107)->ext_0 = ext_34;
    (&kernelContext_107)->engine_counters_0 = engine_counters_32;
    (&kernelContext_107)->engine_contacts_0 = engine_contacts_32;
    (&kernelContext_107)->cs_anchors_0 = cs_anchors_32;
    threadgroup array<uint, int(1024)> g_scan_32;
    (&kernelContext_107)->g_scan_0 = &g_scan_32;
    uint k_25 = id_29.x;
    if(k_25 >= (params_32->chunk_count_0))
    {
        return;
    }
    uint _S829 = body_of_0(k_25, &kernelContext_107);
    uint _S830 = a_lp_0(_S829, &kernelContext_107);
    uint _S831 = atomic_load_explicit((&kernelContext_107)->atom_0+(_S830 + 13U), memory_order_relaxed);
    bool supported_0 = _S831 != 0U;
    bool fixed_1;
    if(((&kernelContext_107)->info_0[k_25] & 1U) != 0U)
    {
        fixed_1 = true;
    }
    else
    {
        if(!supported_0)
        {
            uint _S832 = atomic_load_explicit((&kernelContext_107)->atom_0+(_S830 + 12U), memory_order_relaxed);
            fixed_1 = _S832 == k_25;
        }
        else
        {
            fixed_1 = false;
        }
    }
    uint _S833 = lp_f_0(k_25);
    thread float4 f_11 = float4(*((&kernelContext_107)->lp_0+_S833)) ;
    uint _S834 = _S833 + 1U;
    thread float4 t_5 = float4(*((&kernelContext_107)->lp_0+_S834)) ;
    uint b_32 = 7U * _S829;
    float inv_mass_6 = (float4(*((&kernelContext_107)->bodies_0+b_32)) ).w;
    bool _S835;
    if(!supported_0)
    {
        _S835 = inv_mass_6 > 0.0f;
    }
    else
    {
        _S835 = false;
    }
    if(_S835)
    {
        float3 _S836 = read64w3_0(_S830, (&kernelContext_107)->params_0->lp_scale_0, &kernelContext_107);
        float3 _S837 = read64w3_0(_S830 + 6U, (&kernelContext_107)->params_0->lp_scale_0, &kernelContext_107);
        uint _S838 = 2U * _S829;
        Quat_0 p_10 = quat_of_0(float4(*((&kernelContext_107)->frames_0+(_S838 + 1U))) );
        float3 _S839 = (float4(*((&kernelContext_107)->bodies_0+(b_32 + 2U))) ).xyz;
        thread Quat_0 _S840 = p_10;
        Quat_0 _S841 = conj_0(&_S840);
        thread Quat_0 _S842 = _S841;
        float3 _S843 = rotate_0(&_S842, _S837);
        float3 _S844 = _S839 * _S843;
        thread Quat_0 _S845 = p_10;
        float3 _S846 = rotate_0(&_S845, _S844);
        uint _S847 = 4U * k_25;
        float4 _S848 = float4(*((&kernelContext_107)->chunks_0+_S847)) ;
        float4 _S849 = float4(*((&kernelContext_107)->chunks_0+(_S847 + 1U))) ;
        float4 _S850 = float4(*((&kernelContext_107)->chunks_0+(_S847 + 2U))) ;
        f_11.xyz = f_11.xyz - float3(_S848.w)  * (_S836 * float3(inv_mass_6)  + cross(_S846, _S848.xyz - (float4(*((&kernelContext_107)->frames_0+_S838)) ).xyz));
        float _S851 = _S846.x;
        float _S852 = _S849.w;
        float _S853 = _S846.y;
        float _S854 = _S850.x;
        float _S855 = _S846.z;
        float _S856 = _S850.y;
        t_5.xyz = t_5.xyz - float3(_S849.x * _S851 + _S852 * _S853 + _S854 * _S855, _S852 * _S851 + _S849.y * _S853 + _S856 * _S855, _S854 * _S851 + _S856 * _S853 + _S849.z * _S855);
    }
    float _S857;
    if(fixed_1)
    {
        _S857 = 1.0f;
    }
    else
    {
        _S857 = 0.0f;
    }
    f_11.w = _S857;
    *((&kernelContext_107)->lp_0+_S833) = packed_float4(f_11) ;
    *((&kernelContext_107)->lp_0+_S834) = packed_float4(t_5) ;
    return;
}

uint adj_bond_0(uint at_14, KernelContext_0 thread* kernelContext_108)
{
    return (as_type<uint>(((float4(*(kernelContext_108->ext_0+(kernelContext_108->params_0->ext_adj_0 + at_14 / 4U))) )[at_14 % 4U])));
}

struct JointBond_0
{
    float4 geom0_0;
    float4 geom1_0;
    float4 stiff0_0;
    float4 stiff1_0;
    float4 rebar0_0;
    float4 rebar1_0;
    uint4 ids_1;
};

struct LpBond_0
{
    float3 centroid_0;
    float3 n_3;
    float3 t1_2;
    float3 t2_0;
    JointBond_0 law_0;
};

LpBond_0 lp_bond_0(uint i_8, KernelContext_0 thread* kernelContext_109)
{
    uint o_7 = 10U * i_8;
    thread LpBond_0 b_33;
    (&b_33)->centroid_0 = (float4(*(kernelContext_109->ext_0+o_7)) ).xyz;
    float3 _S858 = (float4(*(kernelContext_109->ext_0+(o_7 + 1U))) ).xyz;
    (&b_33)->n_3 = _S858;
    float3 _S859 = (float4(*(kernelContext_109->ext_0+(o_7 + 2U))) ).xyz;
    (&b_33)->t1_2 = _S859;
    (&b_33)->t2_0 = cross(_S858, _S859);
    (&(&b_33)->law_0)->geom0_0 = float4(*(kernelContext_109->ext_0+(o_7 + 3U))) ;
    (&(&b_33)->law_0)->geom1_0 = float4(*(kernelContext_109->ext_0+(o_7 + 4U))) ;
    (&(&b_33)->law_0)->stiff0_0 = float4(*(kernelContext_109->ext_0+(o_7 + 5U))) ;
    (&(&b_33)->law_0)->stiff1_0 = float4(*(kernelContext_109->ext_0+(o_7 + 6U))) ;
    (&(&b_33)->law_0)->rebar0_0 = float4(*(kernelContext_109->ext_0+(o_7 + 7U))) ;
    (&(&b_33)->law_0)->rebar1_0 = float4(*(kernelContext_109->ext_0+(o_7 + 8U))) ;
    (&(&b_33)->law_0)->ids_1 = (as_type<uint4>((float4(*(kernelContext_109->ext_0+(o_7 + 9U))) )));
    return b_33;
}

uint tri_0(uint i_9, uint j_4)
{
    return i_9 * (i_9 + 1U) / 2U + j_4;
}

uint lp_l_0(uint k_26)
{
    return 22U * k_26 + 8U;
}

bool lp_factor_0(uint k_27, KernelContext_0 thread* kernelContext_110)
{
    uint q_26;
    uint a_8;
    uint p2_0;
    uint q2_0;
    uint a2_0;
    float s_6;
    thread array<array<float, int(6)>, int(6)> d_21;
    uint p_11 = 0U;
    for(;;)
    {
        if(p_11 < 6U)
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
            d_21[p_11][q_26] = 0.0f;
            q_26 = q_26 + 1U;
        }
        p_11 = p_11 + 1U;
    }
    uint _S860 = 4U * k_27;
    float4 _S861 = float4(*(kernelContext_110->ext_0+(kernelContext_110->params_0->ext_chunks_0 + _S860))) ;
    uint _S862 = (as_type<uint>((_S861.x)));
    uint _S863 = (as_type<uint>((_S861.y)));
    q_26 = 0U;
    for(;;)
    {
        if(q_26 < _S863)
        {
        }
        else
        {
            break;
        }
        uint _S864 = adj_bond_0(_S862 + q_26, kernelContext_110);
        uint _S865 = w_alive_0(kernelContext_110);
        if((*(kernelContext_110->work_0+(_S865 + _S864))) == 0U)
        {
            q_26 = q_26 + 1U;
            continue;
        }
        LpBond_0 _S866 = lp_bond_0(_S864, kernelContext_110);
        float3 _S867 = _S866.centroid_0 - (float4(*(kernelContext_110->chunks_0+_S860)) ).xyz;
        float _S868 = _S866.law_0.stiff0_0.y;
        array<float, int(6)> _S869 = { { _S868, _S868, _S866.law_0.stiff0_0.x, _S866.law_0.stiff0_0.z, _S866.law_0.stiff0_0.w, _S866.law_0.stiff1_0.x } };
        array<float3, int(3)> _S870 = { { _S866.t1_2, _S866.t2_0, _S866.n_3 } };
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
            float3 c_19 = cross(_S867, _S870[a_8]);
            float _S871 = _S870[a_8].x;
            j_5[a_8][int(0)] = _S871;
            float _S872 = _S870[a_8].y;
            j_5[a_8][int(1)] = _S872;
            float _S873 = _S870[a_8].z;
            j_5[a_8][int(2)] = _S873;
            j_5[a_8][int(3)] = c_19.x;
            j_5[a_8][int(4)] = c_19.y;
            j_5[a_8][int(5)] = c_19.z;
            j_5[3U + a_8][int(0)] = 0.0f;
            j_5[3U + a_8][int(1)] = 0.0f;
            j_5[3U + a_8][int(2)] = 0.0f;
            j_5[3U + a_8][int(3)] = _S871;
            j_5[3U + a_8][int(4)] = _S872;
            j_5[3U + a_8][int(5)] = _S873;
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
                s_6 = 0.0f;
                for(;;)
                {
                    if(a2_0 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_7 = s_6 + j_5[a2_0][p2_0] * _S869[a2_0] * j_5[a2_0][q2_0];
                    a2_0 = a2_0 + 1U;
                    s_6 = s_7;
                }
                d_21[p2_0][q2_0] = d_21[p2_0][q2_0] + s_6;
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
            s_6 = d_21[p2_0][q2_0];
            for(;;)
            {
                if(a2_0 < q2_0)
                {
                }
                else
                {
                    break;
                }
                float s_8 = s_6 - l_0[tri_0(p2_0, a2_0)] * l_0[tri_0(q2_0, a2_0)];
                a2_0 = a2_0 + 1U;
                s_6 = s_8;
            }
            if(p2_0 == q2_0)
            {
                bool _S874;
                if(ok_1)
                {
                    _S874 = s_6 > 0.0f;
                }
                else
                {
                    _S874 = false;
                }
                l_0[tri_0(p2_0, p2_0)] = sqrt(max(s_6, 1.00000000317107685e-30f));
                ok_1 = _S874;
            }
            else
            {
                l_0[tri_0(p2_0, q2_0)] = s_6 / l_0[tri_0(q2_0, q2_0)];
            }
            q2_0 = q2_0 + 1U;
        }
        uint _S875 = p2_0 + 1U;
        ok_0 = ok_1;
        p2_0 = _S875;
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
        packed_float4 device* _S876 = kernelContext_110->lp_0+(lp_l_0(k_27) + q2_0);
        uint _S877 = 4U * q2_0;
        float _S878 = l_0[_S877];
        uint _S879 = _S877 + 1U;
        if(_S879 < 21U)
        {
            s_6 = l_0[_S879];
        }
        else
        {
            s_6 = 0.0f;
        }
        uint _S880 = _S877 + 2U;
        float _S881;
        if(_S880 < 21U)
        {
            _S881 = l_0[_S880];
        }
        else
        {
            _S881 = 0.0f;
        }
        uint _S882 = _S877 + 3U;
        float _S883;
        if(_S882 < 21U)
        {
            _S883 = l_0[_S882];
        }
        else
        {
            _S883 = 0.0f;
        }
        *_S876 = packed_float4(float4(_S878, s_6, _S881, _S883)) ;
        q2_0 = q2_0 + 1U;
    }
    return true;
}

[[kernel]] void fr_lp_block(uint3 id_30 [[thread_position_in_grid]], FractureParams_0 constant* params_33 [[buffer(0)]], uint device* work_33 [[buffer(4)]], packed_float4 device* frames_33 [[buffer(6)]], packed_float4 device* bodies_33 [[buffer(7)]], packed_float4 device* chunks_33 [[buffer(1)]], uint device* sort_33 [[buffer(11)]], atomic<uint> device* atom_33 [[buffer(5)]], Record_natural_0 device* manifolds_33 [[buffer(10)]], packed_float4 device* loads_33 [[buffer(12)]], packed_uint4 device* bonds_33 [[buffer(2)]], uint device* args_33 [[buffer(13)]], uint device* info_33 [[buffer(15)]], ScanParams_0 constant* scan_33 [[buffer(14)]], packed_float4 device* parts_33 [[buffer(3)]], packed_float4 device* colliders_33 [[buffer(8)]], packed_uint4 device* collider_info_33 [[buffer(9)]], packed_float4 device* view_chunks_33 [[buffer(17)]], ViewParams_0 constant* view_33 [[buffer(16)]], packed_float4 device* objects_33 [[buffer(18)]], uint device* shot_bodies_33 [[buffer(20)]], float device* lines_33 [[buffer(19)]], packed_float4 device* geometry_33 [[buffer(21)]], packed_float4 device* shots_33 [[buffer(22)]], packed_float4 device* lp_33 [[buffer(24)]], packed_float4 device* ext_35 [[buffer(23)]], uint device* engine_counters_33 [[buffer(26)]], packed_float4 device* engine_contacts_33 [[buffer(25)]], packed_float4 device* cs_anchors_33 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_111;
    (&kernelContext_111)->params_0 = params_33;
    (&kernelContext_111)->work_0 = work_33;
    (&kernelContext_111)->frames_0 = frames_33;
    (&kernelContext_111)->bodies_0 = bodies_33;
    (&kernelContext_111)->chunks_0 = chunks_33;
    (&kernelContext_111)->sort_0 = sort_33;
    (&kernelContext_111)->atom_0 = atom_33;
    (&kernelContext_111)->manifolds_0 = manifolds_33;
    (&kernelContext_111)->loads_0 = loads_33;
    (&kernelContext_111)->bonds_0 = bonds_33;
    (&kernelContext_111)->args_0 = args_33;
    (&kernelContext_111)->info_0 = info_33;
    (&kernelContext_111)->scan_0 = scan_33;
    (&kernelContext_111)->parts_0 = parts_33;
    (&kernelContext_111)->colliders_0 = colliders_33;
    (&kernelContext_111)->collider_info_0 = collider_info_33;
    (&kernelContext_111)->view_chunks_0 = view_chunks_33;
    (&kernelContext_111)->view_0 = view_33;
    (&kernelContext_111)->objects_0 = objects_33;
    (&kernelContext_111)->shot_bodies_0 = shot_bodies_33;
    (&kernelContext_111)->lines_0 = lines_33;
    (&kernelContext_111)->geometry_0 = geometry_33;
    (&kernelContext_111)->shots_0 = shots_33;
    (&kernelContext_111)->lp_0 = lp_33;
    (&kernelContext_111)->ext_0 = ext_35;
    (&kernelContext_111)->engine_counters_0 = engine_counters_33;
    (&kernelContext_111)->engine_contacts_0 = engine_contacts_33;
    (&kernelContext_111)->cs_anchors_0 = cs_anchors_33;
    threadgroup array<uint, int(1024)> g_scan_33;
    (&kernelContext_111)->g_scan_0 = &g_scan_33;
    uint k_28 = id_30.x;
    bool _S884;
    if(k_28 >= (params_33->chunk_count_0))
    {
        _S884 = true;
    }
    else
    {
        _S884 = ((float4(*((&kernelContext_111)->lp_0+lp_f_0(k_28))) ).w) != 0.0f;
    }
    if(_S884)
    {
        return;
    }
    bool _S885 = lp_factor_0(k_28, &kernelContext_111);
    if(!_S885)
    {
        (*((&kernelContext_111)->lp_0+lp_f_0(k_28)))[int(3)] = 1.0f;
        uint _S886 = a_counters_1(&kernelContext_111);
        uint _S887 = atomic_fetch_or_explicit((&kernelContext_111)->atom_0+(_S886 + 3U), 32U, memory_order_relaxed);
    }
    return;
}

uint st_x_0(uint k_29, uint state_1, uint copy_0)
{
    return 22U * k_29 + state_1 + 2U * copy_0;
}

uint st_d_0(uint k_30, uint state_2)
{
    return 22U * k_30 + state_2 + 4U;
}

void st_force_0(const LpBond_0 thread* b_34, uint ka_0, uint kb_0, uint state_3, uint c_20, float3 thread* q_lin_0, float3 thread* q_ang_0, KernelContext_0 thread* kernelContext_112)
{
    uint _S888 = st_x_0(ka_0, state_3, c_20);
    uint _S889 = st_x_0(kb_0, state_3, c_20);
    float3 _S890 = (float4(*(kernelContext_112->lp_0+(_S889 + 1U))) ).xyz;
    float3 _S891 = (float4(*(kernelContext_112->lp_0+(_S888 + 1U))) ).xyz;
    float3 delta_0 = (float4(*(kernelContext_112->lp_0+_S889)) ).xyz + cross(_S890, b_34->centroid_0 - (float4(*(kernelContext_112->chunks_0+4U * kb_0)) ).xyz) - ((float4(*(kernelContext_112->lp_0+_S888)) ).xyz + cross(_S891, b_34->centroid_0 - (float4(*(kernelContext_112->chunks_0+4U * ka_0)) ).xyz));
    float3 phi_0 = _S890 - _S891;
    float4 _S892 = (&b_34->law_0)->stiff0_0;
    float kt_0 = (&b_34->law_0)->stiff1_0.x;
    float _S893 = (&b_34->law_0)->stiff0_0.y;
    float3 _S894 = b_34->t1_2;
    float3 _S895 = b_34->t2_0;
    float3 _S896 = b_34->n_3;
    *q_lin_0 = float3(_S893 * dot(delta_0, b_34->t1_2), _S893 * dot(delta_0, b_34->t2_0), (&b_34->law_0)->stiff0_0.x * dot(delta_0, b_34->n_3));
    *q_ang_0 = float3(_S892.z * dot(phi_0, _S894), _S892.w * dot(phi_0, _S895), kt_0 * dot(phi_0, _S896));
    return;
}

void st_force_1(const LpBond_0 thread* b_35, uint ka_1, uint kb_1, uint state_4, uint c_21, float3 thread* q_lin_1, float3 thread* q_ang_1, KernelContext_0 thread* kernelContext_113)
{
    uint _S897 = st_x_0(ka_1, state_4, c_21);
    uint _S898 = st_x_0(kb_1, state_4, c_21);
    float3 _S899 = (float4(*(kernelContext_113->lp_0+(_S898 + 1U))) ).xyz;
    float3 _S900 = (float4(*(kernelContext_113->lp_0+(_S897 + 1U))) ).xyz;
    float3 delta_1 = (float4(*(kernelContext_113->lp_0+_S898)) ).xyz + cross(_S899, b_35->centroid_0 - (float4(*(kernelContext_113->chunks_0+4U * kb_1)) ).xyz) - ((float4(*(kernelContext_113->lp_0+_S897)) ).xyz + cross(_S900, b_35->centroid_0 - (float4(*(kernelContext_113->chunks_0+4U * ka_1)) ).xyz));
    float3 phi_1 = _S899 - _S900;
    float4 _S901 = (&b_35->law_0)->stiff0_0;
    float kt_1 = (&b_35->law_0)->stiff1_0.x;
    float _S902 = (&b_35->law_0)->stiff0_0.y;
    float3 _S903 = b_35->t1_2;
    float3 _S904 = b_35->t2_0;
    float3 _S905 = b_35->n_3;
    *q_lin_1 = float3(_S902 * dot(delta_1, b_35->t1_2), _S902 * dot(delta_1, b_35->t2_0), (&b_35->law_0)->stiff0_0.x * dot(delta_1, b_35->n_3));
    *q_ang_1 = float3(_S901.z * dot(phi_1, _S903), _S901.w * dot(phi_1, _S904), kt_1 * dot(phi_1, _S905));
    return;
}

void st_sweep_0(uint k_31, uint state_5, uint from_4, float c1_1, float c2_1, KernelContext_0 thread* kernelContext_114)
{
    float s_9;
    uint to_3 = 1U - from_4;
    uint _S906 = st_x_0(k_31, state_5, from_4);
    float4 _S907 = float4(*(kernelContext_114->lp_0+_S906)) ;
    float4 _S908 = float4(*(kernelContext_114->lp_0+(_S906 + 1U))) ;
    uint _S909 = st_f_0(k_31, state_5);
    float4 _S910 = float4(*(kernelContext_114->lp_0+_S909)) ;
    if((_S910.w) != 0.0f)
    {
        uint _S911 = st_x_0(k_31, state_5, to_3);
        *(kernelContext_114->lp_0+_S911) = packed_float4(_S907) ;
        *(kernelContext_114->lp_0+(_S911 + 1U)) = packed_float4(_S908) ;
        uint _S912 = st_d_0(k_31, state_5);
        packed_float4 _S913 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_114->lp_0+_S912) = _S913;
        *(kernelContext_114->lp_0+(_S912 + 1U)) = _S913;
        return;
    }
    float _S914 = _S910.y;
    float _S915 = _S910.z;
    thread array<float, int(6)> r_14;
    r_14[int(0)] = _S910.x;
    r_14[int(1)] = _S914;
    r_14[int(2)] = _S915;
    r_14[int(3)] = 0.0f;
    r_14[int(4)] = 0.0f;
    r_14[int(5)] = 0.0f;
    float3 t_6 = (float4(*(kernelContext_114->lp_0+(_S909 + 1U))) ).xyz;
    r_14[int(3)] = t_6.x;
    r_14[int(4)] = t_6.y;
    r_14[int(5)] = t_6.z;
    uint _S916 = 4U * k_31;
    float4 _S917 = float4(*(kernelContext_114->ext_0+(kernelContext_114->params_0->ext_chunks_0 + _S916))) ;
    uint _S918 = (as_type<uint>((_S917.x)));
    uint _S919 = (as_type<uint>((_S917.y)));
    uint e_3 = 0U;
    for(;;)
    {
        if(e_3 < _S919)
        {
        }
        else
        {
            break;
        }
        uint _S920 = adj_bond_0(_S918 + e_3, kernelContext_114);
        uint _S921 = w_alive_0(kernelContext_114);
        if((*(kernelContext_114->work_0+(_S921 + _S920))) == 0U)
        {
            e_3 = e_3 + 1U;
            continue;
        }
        uint4 _S922 = uint4(*(kernelContext_114->bonds_0+_S920)) ;
        LpBond_0 _S923 = lp_bond_0(_S920, kernelContext_114);
        uint _S924 = _S922.x;
        uint _S925 = _S922.y;
        thread LpBond_0 _S926 = _S923;
        thread float3 q_lin_2;
        thread float3 q_ang_2;
        st_force_0(&_S926, _S924, _S925, state_5, from_4, &q_lin_2, &q_ang_2, kernelContext_114);
        float3 fb_0 = _S923.t1_2 * float3(q_lin_2.x)  + _S923.t2_0 * float3(q_lin_2.y)  + _S923.n_3 * float3(q_lin_2.z) ;
        float3 mb_0 = _S923.t1_2 * float3(q_ang_2.x)  + _S923.t2_0 * float3(q_ang_2.y)  + _S923.n_3 * float3(q_ang_2.z) ;
        float3 arm_0 = _S923.centroid_0 - (float4(*(kernelContext_114->chunks_0+_S916)) ).xyz;
        bool _S927 = k_31 == _S924;
        float3 lf_0;
        if(_S927)
        {
            lf_0 = fb_0;
        }
        else
        {
            lf_0 = - fb_0;
        }
        float3 _S928;
        if(_S927)
        {
            _S928 = mb_0;
        }
        else
        {
            _S928 = - mb_0;
        }
        float3 lm_0 = _S928 + cross(arm_0, lf_0);
        r_14[int(0)] = r_14[int(0)] + lf_0.x;
        r_14[int(1)] = r_14[int(1)] + lf_0.y;
        r_14[int(2)] = r_14[int(2)] + lf_0.z;
        r_14[int(3)] = r_14[int(3)] + lm_0.x;
        r_14[int(4)] = r_14[int(4)] + lm_0.y;
        r_14[int(5)] = r_14[int(5)] + lm_0.z;
        e_3 = e_3 + 1U;
    }
    thread array<float, int(21)> l_1;
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
        l_1[z0_0] = 0.0f;
        z0_0 = z0_0 + 1U;
    }
    uint w_7 = 0U;
    for(;;)
    {
        if(w_7 < 6U)
        {
        }
        else
        {
            break;
        }
        float4 _S929 = float4(*(kernelContext_114->lp_0+(lp_l_0(k_31) + w_7))) ;
        uint _S930 = 4U * w_7;
        l_1[_S930] = _S929.x;
        uint _S931 = _S930 + 1U;
        if(_S931 < 21U)
        {
            l_1[_S931] = _S929.y;
        }
        uint _S932 = _S930 + 2U;
        if(_S932 < 21U)
        {
            l_1[_S932] = _S929.z;
        }
        uint _S933 = _S930 + 3U;
        if(_S933 < 21U)
        {
            l_1[_S933] = _S929.w;
        }
        w_7 = w_7 + 1U;
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
        uint p_12 = 0U;
        s_9 = r_14[i2_0];
        for(;;)
        {
            if(p_12 < i2_0)
            {
            }
            else
            {
                break;
            }
            float s_10 = s_9 - l_1[tri_0(i2_0, p_12)] * y_2[p_12];
            p_12 = p_12 + 1U;
            s_9 = s_10;
        }
        y_2[i2_0] = s_9 / l_1[tri_0(i2_0, i2_0)];
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
        s_9 = y_2[i3_0];
        for(;;)
        {
            if(p2_1 < int(6))
            {
            }
            else
            {
                break;
            }
            float s_11 = s_9 - l_1[tri_0(uint(p2_1), uint(i3_0))] * z_3[p2_1];
            p2_1 = p2_1 + int(1);
            s_9 = s_11;
        }
        uint _S934 = uint(i3_0);
        z_3[i3_0] = s_9 / l_1[tri_0(_S934, _S934)];
        i3_0 = i3_0 - int(1);
    }
    uint _S935 = st_d_0(k_31, state_5);
    float4 _S936 = float4(*(kernelContext_114->lp_0+_S935)) ;
    thread float4 d0_0 = _S936;
    uint _S937 = _S935 + 1U;
    float4 _S938 = float4(*(kernelContext_114->lp_0+_S937)) ;
    thread float4 d1_0 = _S938;
    float3 _S939 = float3(c1_1) ;
    float3 _S940 = float3(c2_1) ;
    d0_0.xyz = _S939 * _S936.xyz + _S940 * float3(z_3[int(0)], z_3[int(1)], z_3[int(2)]);
    d1_0.xyz = _S939 * _S938.xyz + _S940 * float3(z_3[int(3)], z_3[int(4)], z_3[int(5)]);
    *(kernelContext_114->lp_0+_S935) = packed_float4(d0_0) ;
    *(kernelContext_114->lp_0+_S937) = packed_float4(d1_0) ;
    uint _S941 = st_x_0(k_31, state_5, to_3);
    *(kernelContext_114->lp_0+_S941) = packed_float4(float4(_S907.xyz + d0_0.xyz, 0.0f)) ;
    *(kernelContext_114->lp_0+(_S941 + 1U)) = packed_float4(float4(_S908.xyz + d1_0.xyz, 0.0f)) ;
    return;
}

void lp_sweep_0(uint k_32, uint from_5, KernelContext_0 thread* kernelContext_115)
{
    st_sweep_0(k_32, 0U, from_5, kernelContext_115->params_0->lp_c1_0, kernelContext_115->params_0->lp_c2_0, kernelContext_115);
    return;
}

[[kernel]] void fr_lp_sweep_even(uint3 id_31 [[thread_position_in_grid]], FractureParams_0 constant* params_34 [[buffer(0)]], uint device* work_34 [[buffer(4)]], packed_float4 device* frames_34 [[buffer(6)]], packed_float4 device* bodies_34 [[buffer(7)]], packed_float4 device* chunks_34 [[buffer(1)]], uint device* sort_34 [[buffer(11)]], atomic<uint> device* atom_34 [[buffer(5)]], Record_natural_0 device* manifolds_34 [[buffer(10)]], packed_float4 device* loads_34 [[buffer(12)]], packed_uint4 device* bonds_34 [[buffer(2)]], uint device* args_34 [[buffer(13)]], uint device* info_34 [[buffer(15)]], ScanParams_0 constant* scan_34 [[buffer(14)]], packed_float4 device* parts_34 [[buffer(3)]], packed_float4 device* colliders_34 [[buffer(8)]], packed_uint4 device* collider_info_34 [[buffer(9)]], packed_float4 device* view_chunks_34 [[buffer(17)]], ViewParams_0 constant* view_34 [[buffer(16)]], packed_float4 device* objects_34 [[buffer(18)]], uint device* shot_bodies_34 [[buffer(20)]], float device* lines_34 [[buffer(19)]], packed_float4 device* geometry_34 [[buffer(21)]], packed_float4 device* shots_34 [[buffer(22)]], packed_float4 device* lp_34 [[buffer(24)]], packed_float4 device* ext_36 [[buffer(23)]], uint device* engine_counters_34 [[buffer(26)]], packed_float4 device* engine_contacts_34 [[buffer(25)]], packed_float4 device* cs_anchors_34 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_116;
    (&kernelContext_116)->params_0 = params_34;
    (&kernelContext_116)->work_0 = work_34;
    (&kernelContext_116)->frames_0 = frames_34;
    (&kernelContext_116)->bodies_0 = bodies_34;
    (&kernelContext_116)->chunks_0 = chunks_34;
    (&kernelContext_116)->sort_0 = sort_34;
    (&kernelContext_116)->atom_0 = atom_34;
    (&kernelContext_116)->manifolds_0 = manifolds_34;
    (&kernelContext_116)->loads_0 = loads_34;
    (&kernelContext_116)->bonds_0 = bonds_34;
    (&kernelContext_116)->args_0 = args_34;
    (&kernelContext_116)->info_0 = info_34;
    (&kernelContext_116)->scan_0 = scan_34;
    (&kernelContext_116)->parts_0 = parts_34;
    (&kernelContext_116)->colliders_0 = colliders_34;
    (&kernelContext_116)->collider_info_0 = collider_info_34;
    (&kernelContext_116)->view_chunks_0 = view_chunks_34;
    (&kernelContext_116)->view_0 = view_34;
    (&kernelContext_116)->objects_0 = objects_34;
    (&kernelContext_116)->shot_bodies_0 = shot_bodies_34;
    (&kernelContext_116)->lines_0 = lines_34;
    (&kernelContext_116)->geometry_0 = geometry_34;
    (&kernelContext_116)->shots_0 = shots_34;
    (&kernelContext_116)->lp_0 = lp_34;
    (&kernelContext_116)->ext_0 = ext_36;
    (&kernelContext_116)->engine_counters_0 = engine_counters_34;
    (&kernelContext_116)->engine_contacts_0 = engine_contacts_34;
    (&kernelContext_116)->cs_anchors_0 = cs_anchors_34;
    threadgroup array<uint, int(1024)> g_scan_34;
    (&kernelContext_116)->g_scan_0 = &g_scan_34;
    uint _S942 = id_31.x;
    if(_S942 < (params_34->chunk_count_0))
    {
        lp_sweep_0(_S942, 0U, &kernelContext_116);
    }
    return;
}

[[kernel]] void fr_lp_sweep_odd(uint3 id_32 [[thread_position_in_grid]], FractureParams_0 constant* params_35 [[buffer(0)]], uint device* work_35 [[buffer(4)]], packed_float4 device* frames_35 [[buffer(6)]], packed_float4 device* bodies_35 [[buffer(7)]], packed_float4 device* chunks_35 [[buffer(1)]], uint device* sort_35 [[buffer(11)]], atomic<uint> device* atom_35 [[buffer(5)]], Record_natural_0 device* manifolds_35 [[buffer(10)]], packed_float4 device* loads_35 [[buffer(12)]], packed_uint4 device* bonds_35 [[buffer(2)]], uint device* args_35 [[buffer(13)]], uint device* info_35 [[buffer(15)]], ScanParams_0 constant* scan_35 [[buffer(14)]], packed_float4 device* parts_35 [[buffer(3)]], packed_float4 device* colliders_35 [[buffer(8)]], packed_uint4 device* collider_info_35 [[buffer(9)]], packed_float4 device* view_chunks_35 [[buffer(17)]], ViewParams_0 constant* view_35 [[buffer(16)]], packed_float4 device* objects_35 [[buffer(18)]], uint device* shot_bodies_35 [[buffer(20)]], float device* lines_35 [[buffer(19)]], packed_float4 device* geometry_35 [[buffer(21)]], packed_float4 device* shots_35 [[buffer(22)]], packed_float4 device* lp_35 [[buffer(24)]], packed_float4 device* ext_37 [[buffer(23)]], uint device* engine_counters_35 [[buffer(26)]], packed_float4 device* engine_contacts_35 [[buffer(25)]], packed_float4 device* cs_anchors_35 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_117;
    (&kernelContext_117)->params_0 = params_35;
    (&kernelContext_117)->work_0 = work_35;
    (&kernelContext_117)->frames_0 = frames_35;
    (&kernelContext_117)->bodies_0 = bodies_35;
    (&kernelContext_117)->chunks_0 = chunks_35;
    (&kernelContext_117)->sort_0 = sort_35;
    (&kernelContext_117)->atom_0 = atom_35;
    (&kernelContext_117)->manifolds_0 = manifolds_35;
    (&kernelContext_117)->loads_0 = loads_35;
    (&kernelContext_117)->bonds_0 = bonds_35;
    (&kernelContext_117)->args_0 = args_35;
    (&kernelContext_117)->info_0 = info_35;
    (&kernelContext_117)->scan_0 = scan_35;
    (&kernelContext_117)->parts_0 = parts_35;
    (&kernelContext_117)->colliders_0 = colliders_35;
    (&kernelContext_117)->collider_info_0 = collider_info_35;
    (&kernelContext_117)->view_chunks_0 = view_chunks_35;
    (&kernelContext_117)->view_0 = view_35;
    (&kernelContext_117)->objects_0 = objects_35;
    (&kernelContext_117)->shot_bodies_0 = shot_bodies_35;
    (&kernelContext_117)->lines_0 = lines_35;
    (&kernelContext_117)->geometry_0 = geometry_35;
    (&kernelContext_117)->shots_0 = shots_35;
    (&kernelContext_117)->lp_0 = lp_35;
    (&kernelContext_117)->ext_0 = ext_37;
    (&kernelContext_117)->engine_counters_0 = engine_counters_35;
    (&kernelContext_117)->engine_contacts_0 = engine_contacts_35;
    (&kernelContext_117)->cs_anchors_0 = cs_anchors_35;
    threadgroup array<uint, int(1024)> g_scan_35;
    (&kernelContext_117)->g_scan_0 = &g_scan_35;
    uint _S943 = id_32.x;
    if(_S943 < (params_35->chunk_count_0))
    {
        lp_sweep_0(_S943, 1U, &kernelContext_117);
    }
    return;
}

void lp_force_0(const LpBond_0 thread* b_36, uint ka_2, uint kb_2, uint c_22, float3 thread* q_lin_3, float3 thread* q_ang_3, KernelContext_0 thread* kernelContext_118)
{
    st_force_1(b_36, ka_2, kb_2, 0U, c_22, q_lin_3, q_ang_3, kernelContext_118);
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

Measures_0 stress_measures_0(const JointBond_0 thread* b_37, float3 q_lin_4, float3 q_ang_4)
{
    float area_0 = b_37->geom0_0.x;
    float _S944 = q_lin_4.z;
    float axial_0 = (metal::fast::divide((_S944), (area_0)));
    float _S945 = (metal::fast::divide((abs(q_ang_4.x)), (b_37->geom1_0.x)));
    float _S946 = (metal::fast::divide((abs(q_ang_4.y)), (b_37->geom1_0.y)));
    float bending_0 = _S945 + _S946;
    float _S947 = q_lin_4.x;
    float _S948 = q_lin_4.y;
    float _S949 = (metal::fast::sqrt((_S947 * _S947 + _S948 * _S948)));
    float _S950 = (metal::fast::divide((_S949), (area_0)));
    float _S951 = (metal::fast::divide((abs(q_ang_4.z)), (b_37->geom0_0.w)));
    float shear_1 = _S950 + _S951;
    thread Measures_0 m_3;
    (&m_3)->tension_0 = axial_0 + bending_0;
    (&m_3)->shear_0 = shear_1;
    float _S952 = - axial_0;
    (&m_3)->normal_compression_0 = max(_S952, 0.0f);
    (&m_3)->compression_0 = _S952 + bending_0;
    (&m_3)->compressive_force_0 = max(- _S944, 0.0f);
    return m_3;
}

JointMaterial_0 lp_material_0(uint m_4, KernelContext_0 thread* kernelContext_119)
{
    uint o_8 = kernelContext_119->params_0->ext_mat_0 + 5U * m_4;
    thread JointMaterial_0 j_6;
    (&j_6)->strength_0 = float4(*(kernelContext_119->ext_0+o_8)) ;
    (&j_6)->energy_0 = float4(*(kernelContext_119->ext_0+(o_8 + 1U))) ;
    (&j_6)->dif_0 = float4(*(kernelContext_119->ext_0+(o_8 + 2U))) ;
    (&j_6)->misc_0 = float4(*(kernelContext_119->ext_0+(o_8 + 3U))) ;
    (&j_6)->kind_flags_0 = (as_type<uint4>((float4(*(kernelContext_119->ext_0+(o_8 + 4U))) )));
    return j_6;
}

float infinity_0()
{
    return float_bits_0(2139095040U);
}

float4 failure_indices_0(const JointMaterial_0 thread* mat_1, const JointBond_0 thread* b_38, const Measures_0 thread* m_5, float multiplier_0)
{
    float fc_2 = mat_1->strength_0.y * multiplier_0;
    float _S953 = min(mat_1->strength_0.z * multiplier_0 + mat_1->strength_0.w * m_5->normal_compression_0, mat_1->energy_0.x * multiplier_0);
    thread float4 idx_0;
    float _S954 = (metal::fast::divide((m_5->tension_0), (mat_1->strength_0.x * multiplier_0)));
    idx_0.x = max(_S954, 0.0f);
    float _S955;
    if(_S953 > 0.0f)
    {
        float _S956 = (metal::fast::divide((m_5->shear_0), (_S953)));
        _S955 = _S956;
    }
    else
    {
        _S955 = infinity_0();
    }
    idx_0.y = _S955;
    float _S957 = (metal::fast::divide((m_5->compression_0), (fc_2)));
    idx_0.z = max(_S957, 0.0f);
    float _S958 = b_38->stiff1_0.y;
    if(_S958 > 0.0f)
    {
        float _S959 = (metal::fast::divide((m_5->compressive_force_0), (_S958)));
        _S955 = _S959;
    }
    else
    {
        _S955 = 0.0f;
    }
    idx_0.w = _S955;
    return idx_0;
}

[[kernel]] void fr_lp_breaks(uint3 id_33 [[thread_position_in_grid]], FractureParams_0 constant* params_36 [[buffer(0)]], uint device* work_36 [[buffer(4)]], packed_float4 device* frames_36 [[buffer(6)]], packed_float4 device* bodies_36 [[buffer(7)]], packed_float4 device* chunks_36 [[buffer(1)]], uint device* sort_36 [[buffer(11)]], atomic<uint> device* atom_36 [[buffer(5)]], Record_natural_0 device* manifolds_36 [[buffer(10)]], packed_float4 device* loads_36 [[buffer(12)]], packed_uint4 device* bonds_36 [[buffer(2)]], uint device* args_36 [[buffer(13)]], uint device* info_36 [[buffer(15)]], ScanParams_0 constant* scan_36 [[buffer(14)]], packed_float4 device* parts_36 [[buffer(3)]], packed_float4 device* colliders_36 [[buffer(8)]], packed_uint4 device* collider_info_36 [[buffer(9)]], packed_float4 device* view_chunks_36 [[buffer(17)]], ViewParams_0 constant* view_36 [[buffer(16)]], packed_float4 device* objects_36 [[buffer(18)]], uint device* shot_bodies_36 [[buffer(20)]], float device* lines_36 [[buffer(19)]], packed_float4 device* geometry_36 [[buffer(21)]], packed_float4 device* shots_36 [[buffer(22)]], packed_float4 device* lp_36 [[buffer(24)]], packed_float4 device* ext_38 [[buffer(23)]], uint device* engine_counters_36 [[buffer(26)]], packed_float4 device* engine_contacts_36 [[buffer(25)]], packed_float4 device* cs_anchors_36 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_120;
    (&kernelContext_120)->params_0 = params_36;
    (&kernelContext_120)->work_0 = work_36;
    (&kernelContext_120)->frames_0 = frames_36;
    (&kernelContext_120)->bodies_0 = bodies_36;
    (&kernelContext_120)->chunks_0 = chunks_36;
    (&kernelContext_120)->sort_0 = sort_36;
    (&kernelContext_120)->atom_0 = atom_36;
    (&kernelContext_120)->manifolds_0 = manifolds_36;
    (&kernelContext_120)->loads_0 = loads_36;
    (&kernelContext_120)->bonds_0 = bonds_36;
    (&kernelContext_120)->args_0 = args_36;
    (&kernelContext_120)->info_0 = info_36;
    (&kernelContext_120)->scan_0 = scan_36;
    (&kernelContext_120)->parts_0 = parts_36;
    (&kernelContext_120)->colliders_0 = colliders_36;
    (&kernelContext_120)->collider_info_0 = collider_info_36;
    (&kernelContext_120)->view_chunks_0 = view_chunks_36;
    (&kernelContext_120)->view_0 = view_36;
    (&kernelContext_120)->objects_0 = objects_36;
    (&kernelContext_120)->shot_bodies_0 = shot_bodies_36;
    (&kernelContext_120)->lines_0 = lines_36;
    (&kernelContext_120)->geometry_0 = geometry_36;
    (&kernelContext_120)->shots_0 = shots_36;
    (&kernelContext_120)->lp_0 = lp_36;
    (&kernelContext_120)->ext_0 = ext_38;
    (&kernelContext_120)->engine_counters_0 = engine_counters_36;
    (&kernelContext_120)->engine_contacts_0 = engine_contacts_36;
    (&kernelContext_120)->cs_anchors_0 = cs_anchors_36;
    threadgroup array<uint, int(1024)> g_scan_36;
    (&kernelContext_120)->g_scan_0 = &g_scan_36;
    uint i_10 = id_33.x;
    bool _S960;
    if(i_10 >= (params_36->bond_count_0))
    {
        _S960 = true;
    }
    else
    {
        uint _S961 = w_alive_0(&kernelContext_120);
        _S960 = (*((&kernelContext_120)->work_0+(_S961 + i_10))) == 0U;
    }
    if(_S960)
    {
        return;
    }
    uint4 _S962 = uint4(*((&kernelContext_120)->bonds_0+i_10)) ;
    LpBond_0 _S963 = lp_bond_0(i_10, &kernelContext_120);
    uint _S964 = _S962.x;
    uint _S965 = _S962.y;
    thread LpBond_0 _S966 = _S963;
    thread float3 q_lin_5;
    thread float3 q_ang_5;
    lp_force_0(&_S966, _S964, _S965, 0U, &q_lin_5, &q_ang_5, &kernelContext_120);
    thread JointBond_0 _S967 = _S963.law_0;
    Measures_0 _S968 = stress_measures_0(&_S967, q_lin_5, q_ang_5);
    JointMaterial_0 _S969 = lp_material_0(_S963.law_0.ids_1.x, &kernelContext_120);
    float _S970 = _S963.law_0.geom1_0.w;
    thread JointMaterial_0 _S971 = _S969;
    thread JointBond_0 _S972 = _S963.law_0;
    thread Measures_0 _S973 = _S968;
    float4 _S974 = failure_indices_0(&_S971, &_S972, &_S973, _S970);
    float _S975 = max(max(_S974.x, _S974.y), max(_S974.z, _S974.w));
    uint _S976 = a_counters_1(&kernelContext_120);
    uint _S977 = atomic_fetch_max_explicit((&kernelContext_120)->atom_0+(_S976 + 11U), (as_type<uint>((max(_S975, 0.0f)))), memory_order_relaxed);
    if(_S975 >= 1.0f)
    {
        uint _S978 = w_alive_0(&kernelContext_120);
        *((&kernelContext_120)->work_0+(_S978 + i_10)) = 0U;
        uint _S979 = a_dirty_0(&kernelContext_120);
        uint _S980 = body_of_0(_S964, &kernelContext_120);
        uint _S981 = atomic_fetch_or_explicit((&kernelContext_120)->atom_0+(_S979 + _S980), 1U, memory_order_relaxed);
        uint _S982 = atomic_fetch_add_explicit((&kernelContext_120)->atom_0+_S976, 1U, memory_order_relaxed);
        uint _S983 = atomic_fetch_add_explicit((&kernelContext_120)->atom_0+(_S976 + 1U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first_reset(uint3 id_34 [[thread_position_in_grid]], FractureParams_0 constant* params_37 [[buffer(0)]], uint device* work_37 [[buffer(4)]], packed_float4 device* frames_37 [[buffer(6)]], packed_float4 device* bodies_37 [[buffer(7)]], packed_float4 device* chunks_37 [[buffer(1)]], uint device* sort_37 [[buffer(11)]], atomic<uint> device* atom_37 [[buffer(5)]], Record_natural_0 device* manifolds_37 [[buffer(10)]], packed_float4 device* loads_37 [[buffer(12)]], packed_uint4 device* bonds_37 [[buffer(2)]], uint device* args_37 [[buffer(13)]], uint device* info_37 [[buffer(15)]], ScanParams_0 constant* scan_37 [[buffer(14)]], packed_float4 device* parts_37 [[buffer(3)]], packed_float4 device* colliders_37 [[buffer(8)]], packed_uint4 device* collider_info_37 [[buffer(9)]], packed_float4 device* view_chunks_37 [[buffer(17)]], ViewParams_0 constant* view_37 [[buffer(16)]], packed_float4 device* objects_37 [[buffer(18)]], uint device* shot_bodies_37 [[buffer(20)]], float device* lines_37 [[buffer(19)]], packed_float4 device* geometry_37 [[buffer(21)]], packed_float4 device* shots_37 [[buffer(22)]], packed_float4 device* lp_37 [[buffer(24)]], packed_float4 device* ext_39 [[buffer(23)]], uint device* engine_counters_37 [[buffer(26)]], packed_float4 device* engine_contacts_37 [[buffer(25)]], packed_float4 device* cs_anchors_37 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_121;
    (&kernelContext_121)->params_0 = params_37;
    (&kernelContext_121)->work_0 = work_37;
    (&kernelContext_121)->frames_0 = frames_37;
    (&kernelContext_121)->bodies_0 = bodies_37;
    (&kernelContext_121)->chunks_0 = chunks_37;
    (&kernelContext_121)->sort_0 = sort_37;
    (&kernelContext_121)->atom_0 = atom_37;
    (&kernelContext_121)->manifolds_0 = manifolds_37;
    (&kernelContext_121)->loads_0 = loads_37;
    (&kernelContext_121)->bonds_0 = bonds_37;
    (&kernelContext_121)->args_0 = args_37;
    (&kernelContext_121)->info_0 = info_37;
    (&kernelContext_121)->scan_0 = scan_37;
    (&kernelContext_121)->parts_0 = parts_37;
    (&kernelContext_121)->colliders_0 = colliders_37;
    (&kernelContext_121)->collider_info_0 = collider_info_37;
    (&kernelContext_121)->view_chunks_0 = view_chunks_37;
    (&kernelContext_121)->view_0 = view_37;
    (&kernelContext_121)->objects_0 = objects_37;
    (&kernelContext_121)->shot_bodies_0 = shot_bodies_37;
    (&kernelContext_121)->lines_0 = lines_37;
    (&kernelContext_121)->geometry_0 = geometry_37;
    (&kernelContext_121)->shots_0 = shots_37;
    (&kernelContext_121)->lp_0 = lp_37;
    (&kernelContext_121)->ext_0 = ext_39;
    (&kernelContext_121)->engine_counters_0 = engine_counters_37;
    (&kernelContext_121)->engine_contacts_0 = engine_contacts_37;
    (&kernelContext_121)->cs_anchors_0 = cs_anchors_37;
    threadgroup array<uint, int(1024)> g_scan_37;
    (&kernelContext_121)->g_scan_0 = &g_scan_37;
    uint _S984 = id_34.x;
    if(_S984 < (params_37->shot_count_0))
    {
        uint _S985 = a_first_1(_S984, &kernelContext_121);
        atomic_store_explicit((&kernelContext_121)->atom_0+_S985, 2139095040U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_il_first(uint3 id_35 [[thread_position_in_grid]], FractureParams_0 constant* params_38 [[buffer(0)]], uint device* work_38 [[buffer(4)]], packed_float4 device* frames_38 [[buffer(6)]], packed_float4 device* bodies_38 [[buffer(7)]], packed_float4 device* chunks_38 [[buffer(1)]], uint device* sort_38 [[buffer(11)]], atomic<uint> device* atom_38 [[buffer(5)]], Record_natural_0 device* manifolds_38 [[buffer(10)]], packed_float4 device* loads_38 [[buffer(12)]], packed_uint4 device* bonds_38 [[buffer(2)]], uint device* args_38 [[buffer(13)]], uint device* info_38 [[buffer(15)]], ScanParams_0 constant* scan_38 [[buffer(14)]], packed_float4 device* parts_38 [[buffer(3)]], packed_float4 device* colliders_38 [[buffer(8)]], packed_uint4 device* collider_info_38 [[buffer(9)]], packed_float4 device* view_chunks_38 [[buffer(17)]], ViewParams_0 constant* view_38 [[buffer(16)]], packed_float4 device* objects_38 [[buffer(18)]], uint device* shot_bodies_38 [[buffer(20)]], float device* lines_38 [[buffer(19)]], packed_float4 device* geometry_38 [[buffer(21)]], packed_float4 device* shots_38 [[buffer(22)]], packed_float4 device* lp_38 [[buffer(24)]], packed_float4 device* ext_40 [[buffer(23)]], uint device* engine_counters_38 [[buffer(26)]], packed_float4 device* engine_contacts_38 [[buffer(25)]], packed_float4 device* cs_anchors_38 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_122;
    (&kernelContext_122)->params_0 = params_38;
    (&kernelContext_122)->work_0 = work_38;
    (&kernelContext_122)->frames_0 = frames_38;
    (&kernelContext_122)->bodies_0 = bodies_38;
    (&kernelContext_122)->chunks_0 = chunks_38;
    (&kernelContext_122)->sort_0 = sort_38;
    (&kernelContext_122)->atom_0 = atom_38;
    (&kernelContext_122)->manifolds_0 = manifolds_38;
    (&kernelContext_122)->loads_0 = loads_38;
    (&kernelContext_122)->bonds_0 = bonds_38;
    (&kernelContext_122)->args_0 = args_38;
    (&kernelContext_122)->info_0 = info_38;
    (&kernelContext_122)->scan_0 = scan_38;
    (&kernelContext_122)->parts_0 = parts_38;
    (&kernelContext_122)->colliders_0 = colliders_38;
    (&kernelContext_122)->collider_info_0 = collider_info_38;
    (&kernelContext_122)->view_chunks_0 = view_chunks_38;
    (&kernelContext_122)->view_0 = view_38;
    (&kernelContext_122)->objects_0 = objects_38;
    (&kernelContext_122)->shot_bodies_0 = shot_bodies_38;
    (&kernelContext_122)->lines_0 = lines_38;
    (&kernelContext_122)->geometry_0 = geometry_38;
    (&kernelContext_122)->shots_0 = shots_38;
    (&kernelContext_122)->lp_0 = lp_38;
    (&kernelContext_122)->ext_0 = ext_40;
    (&kernelContext_122)->engine_counters_0 = engine_counters_38;
    (&kernelContext_122)->engine_contacts_0 = engine_contacts_38;
    (&kernelContext_122)->cs_anchors_0 = cs_anchors_38;
    threadgroup array<uint, int(1024)> g_scan_38;
    (&kernelContext_122)->g_scan_0 = &g_scan_38;
    uint b_39 = id_35.x;
    bool _S986;
    if(b_39 >= (params_38->body_count_0))
    {
        _S986 = true;
    }
    else
    {
        _S986 = ((&kernelContext_122)->params_0->shot_count_0) == 0U;
    }
    if(_S986)
    {
        return;
    }
    thread uint shot_9;
    bool _S987 = struck_0(b_39, &shot_9, &kernelContext_122);
    if(!_S987)
    {
        return;
    }
    uint _S988 = a_first_1(shot_9, &kernelContext_122);
    atomic<uint> device* _S989 = (&kernelContext_122)->atom_0+_S988;
    uint _S990 = a_strike_0(&kernelContext_122);
    uint _S991 = atomic_load_explicit((&kernelContext_122)->atom_0+(_S990 + 9U * b_39 + 6U), memory_order_relaxed);
    uint _S992 = atomic_fetch_min_explicit(_S989, _S991, memory_order_relaxed);
    return;
}

bool cs_struck_1(uint b_40, KernelContext_0 thread* kernelContext_123)
{
    bool _S993;
    if(b_40 < (kernelContext_123->params_0->body_count_0))
    {
        uint _S994 = a_cs_0(b_40, kernelContext_123);
        uint _S995 = atomic_load_explicit(kernelContext_123->atom_0+(_S994 + 1U), memory_order_relaxed);
        _S993 = _S995 != 4294967295U;
    }
    else
    {
        _S993 = false;
    }
    return _S993;
}

[[kernel]] void fr_il_begin(uint3 id_36 [[thread_position_in_grid]], FractureParams_0 constant* params_39 [[buffer(0)]], uint device* work_39 [[buffer(4)]], packed_float4 device* frames_39 [[buffer(6)]], packed_float4 device* bodies_39 [[buffer(7)]], packed_float4 device* chunks_39 [[buffer(1)]], uint device* sort_39 [[buffer(11)]], atomic<uint> device* atom_39 [[buffer(5)]], Record_natural_0 device* manifolds_39 [[buffer(10)]], packed_float4 device* loads_39 [[buffer(12)]], packed_uint4 device* bonds_39 [[buffer(2)]], uint device* args_39 [[buffer(13)]], uint device* info_39 [[buffer(15)]], ScanParams_0 constant* scan_39 [[buffer(14)]], packed_float4 device* parts_39 [[buffer(3)]], packed_float4 device* colliders_39 [[buffer(8)]], packed_uint4 device* collider_info_39 [[buffer(9)]], packed_float4 device* view_chunks_39 [[buffer(17)]], ViewParams_0 constant* view_39 [[buffer(16)]], packed_float4 device* objects_39 [[buffer(18)]], uint device* shot_bodies_39 [[buffer(20)]], float device* lines_39 [[buffer(19)]], packed_float4 device* geometry_39 [[buffer(21)]], packed_float4 device* shots_39 [[buffer(22)]], packed_float4 device* lp_39 [[buffer(24)]], packed_float4 device* ext_41 [[buffer(23)]], uint device* engine_counters_39 [[buffer(26)]], packed_float4 device* engine_contacts_39 [[buffer(25)]], packed_float4 device* cs_anchors_39 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_124;
    (&kernelContext_124)->params_0 = params_39;
    (&kernelContext_124)->work_0 = work_39;
    (&kernelContext_124)->frames_0 = frames_39;
    (&kernelContext_124)->bodies_0 = bodies_39;
    (&kernelContext_124)->chunks_0 = chunks_39;
    (&kernelContext_124)->sort_0 = sort_39;
    (&kernelContext_124)->atom_0 = atom_39;
    (&kernelContext_124)->manifolds_0 = manifolds_39;
    (&kernelContext_124)->loads_0 = loads_39;
    (&kernelContext_124)->bonds_0 = bonds_39;
    (&kernelContext_124)->args_0 = args_39;
    (&kernelContext_124)->info_0 = info_39;
    (&kernelContext_124)->scan_0 = scan_39;
    (&kernelContext_124)->parts_0 = parts_39;
    (&kernelContext_124)->colliders_0 = colliders_39;
    (&kernelContext_124)->collider_info_0 = collider_info_39;
    (&kernelContext_124)->view_chunks_0 = view_chunks_39;
    (&kernelContext_124)->view_0 = view_39;
    (&kernelContext_124)->objects_0 = objects_39;
    (&kernelContext_124)->shot_bodies_0 = shot_bodies_39;
    (&kernelContext_124)->lines_0 = lines_39;
    (&kernelContext_124)->geometry_0 = geometry_39;
    (&kernelContext_124)->shots_0 = shots_39;
    (&kernelContext_124)->lp_0 = lp_39;
    (&kernelContext_124)->ext_0 = ext_41;
    (&kernelContext_124)->engine_counters_0 = engine_counters_39;
    (&kernelContext_124)->engine_contacts_0 = engine_contacts_39;
    (&kernelContext_124)->cs_anchors_0 = cs_anchors_39;
    threadgroup array<uint, int(1024)> g_scan_39;
    (&kernelContext_124)->g_scan_0 = &g_scan_39;
    if((params_39->shot_count_0) > 0U)
    {
        il_begin_1(id_36.x, &kernelContext_124);
    }
    return;
}

[[kernel]] void fr_il_extend_even(uint3 id_37 [[thread_position_in_grid]], FractureParams_0 constant* params_40 [[buffer(0)]], uint device* work_40 [[buffer(4)]], packed_float4 device* frames_40 [[buffer(6)]], packed_float4 device* bodies_40 [[buffer(7)]], packed_float4 device* chunks_40 [[buffer(1)]], uint device* sort_40 [[buffer(11)]], atomic<uint> device* atom_40 [[buffer(5)]], Record_natural_0 device* manifolds_40 [[buffer(10)]], packed_float4 device* loads_40 [[buffer(12)]], packed_uint4 device* bonds_40 [[buffer(2)]], uint device* args_40 [[buffer(13)]], uint device* info_40 [[buffer(15)]], ScanParams_0 constant* scan_40 [[buffer(14)]], packed_float4 device* parts_40 [[buffer(3)]], packed_float4 device* colliders_40 [[buffer(8)]], packed_uint4 device* collider_info_40 [[buffer(9)]], packed_float4 device* view_chunks_40 [[buffer(17)]], ViewParams_0 constant* view_40 [[buffer(16)]], packed_float4 device* objects_40 [[buffer(18)]], uint device* shot_bodies_40 [[buffer(20)]], float device* lines_40 [[buffer(19)]], packed_float4 device* geometry_40 [[buffer(21)]], packed_float4 device* shots_40 [[buffer(22)]], packed_float4 device* lp_40 [[buffer(24)]], packed_float4 device* ext_42 [[buffer(23)]], uint device* engine_counters_40 [[buffer(26)]], packed_float4 device* engine_contacts_40 [[buffer(25)]], packed_float4 device* cs_anchors_40 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_125;
    (&kernelContext_125)->params_0 = params_40;
    (&kernelContext_125)->work_0 = work_40;
    (&kernelContext_125)->frames_0 = frames_40;
    (&kernelContext_125)->bodies_0 = bodies_40;
    (&kernelContext_125)->chunks_0 = chunks_40;
    (&kernelContext_125)->sort_0 = sort_40;
    (&kernelContext_125)->atom_0 = atom_40;
    (&kernelContext_125)->manifolds_0 = manifolds_40;
    (&kernelContext_125)->loads_0 = loads_40;
    (&kernelContext_125)->bonds_0 = bonds_40;
    (&kernelContext_125)->args_0 = args_40;
    (&kernelContext_125)->info_0 = info_40;
    (&kernelContext_125)->scan_0 = scan_40;
    (&kernelContext_125)->parts_0 = parts_40;
    (&kernelContext_125)->colliders_0 = colliders_40;
    (&kernelContext_125)->collider_info_0 = collider_info_40;
    (&kernelContext_125)->view_chunks_0 = view_chunks_40;
    (&kernelContext_125)->view_0 = view_40;
    (&kernelContext_125)->objects_0 = objects_40;
    (&kernelContext_125)->shot_bodies_0 = shot_bodies_40;
    (&kernelContext_125)->lines_0 = lines_40;
    (&kernelContext_125)->geometry_0 = geometry_40;
    (&kernelContext_125)->shots_0 = shots_40;
    (&kernelContext_125)->lp_0 = lp_40;
    (&kernelContext_125)->ext_0 = ext_42;
    (&kernelContext_125)->engine_counters_0 = engine_counters_40;
    (&kernelContext_125)->engine_contacts_0 = engine_contacts_40;
    (&kernelContext_125)->cs_anchors_0 = cs_anchors_40;
    threadgroup array<uint, int(1024)> g_scan_40;
    (&kernelContext_125)->g_scan_0 = &g_scan_40;
    if((params_40->shot_count_0) > 0U)
    {
        il_extend_1(id_37.x, 0U, &kernelContext_125);
    }
    return;
}

[[kernel]] void fr_il_extend_odd(uint3 id_38 [[thread_position_in_grid]], FractureParams_0 constant* params_41 [[buffer(0)]], uint device* work_41 [[buffer(4)]], packed_float4 device* frames_41 [[buffer(6)]], packed_float4 device* bodies_41 [[buffer(7)]], packed_float4 device* chunks_41 [[buffer(1)]], uint device* sort_41 [[buffer(11)]], atomic<uint> device* atom_41 [[buffer(5)]], Record_natural_0 device* manifolds_41 [[buffer(10)]], packed_float4 device* loads_41 [[buffer(12)]], packed_uint4 device* bonds_41 [[buffer(2)]], uint device* args_41 [[buffer(13)]], uint device* info_41 [[buffer(15)]], ScanParams_0 constant* scan_41 [[buffer(14)]], packed_float4 device* parts_41 [[buffer(3)]], packed_float4 device* colliders_41 [[buffer(8)]], packed_uint4 device* collider_info_41 [[buffer(9)]], packed_float4 device* view_chunks_41 [[buffer(17)]], ViewParams_0 constant* view_41 [[buffer(16)]], packed_float4 device* objects_41 [[buffer(18)]], uint device* shot_bodies_41 [[buffer(20)]], float device* lines_41 [[buffer(19)]], packed_float4 device* geometry_41 [[buffer(21)]], packed_float4 device* shots_41 [[buffer(22)]], packed_float4 device* lp_41 [[buffer(24)]], packed_float4 device* ext_43 [[buffer(23)]], uint device* engine_counters_41 [[buffer(26)]], packed_float4 device* engine_contacts_41 [[buffer(25)]], packed_float4 device* cs_anchors_41 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_126;
    (&kernelContext_126)->params_0 = params_41;
    (&kernelContext_126)->work_0 = work_41;
    (&kernelContext_126)->frames_0 = frames_41;
    (&kernelContext_126)->bodies_0 = bodies_41;
    (&kernelContext_126)->chunks_0 = chunks_41;
    (&kernelContext_126)->sort_0 = sort_41;
    (&kernelContext_126)->atom_0 = atom_41;
    (&kernelContext_126)->manifolds_0 = manifolds_41;
    (&kernelContext_126)->loads_0 = loads_41;
    (&kernelContext_126)->bonds_0 = bonds_41;
    (&kernelContext_126)->args_0 = args_41;
    (&kernelContext_126)->info_0 = info_41;
    (&kernelContext_126)->scan_0 = scan_41;
    (&kernelContext_126)->parts_0 = parts_41;
    (&kernelContext_126)->colliders_0 = colliders_41;
    (&kernelContext_126)->collider_info_0 = collider_info_41;
    (&kernelContext_126)->view_chunks_0 = view_chunks_41;
    (&kernelContext_126)->view_0 = view_41;
    (&kernelContext_126)->objects_0 = objects_41;
    (&kernelContext_126)->shot_bodies_0 = shot_bodies_41;
    (&kernelContext_126)->lines_0 = lines_41;
    (&kernelContext_126)->geometry_0 = geometry_41;
    (&kernelContext_126)->shots_0 = shots_41;
    (&kernelContext_126)->lp_0 = lp_41;
    (&kernelContext_126)->ext_0 = ext_43;
    (&kernelContext_126)->engine_counters_0 = engine_counters_41;
    (&kernelContext_126)->engine_contacts_0 = engine_contacts_41;
    (&kernelContext_126)->cs_anchors_0 = cs_anchors_41;
    threadgroup array<uint, int(1024)> g_scan_41;
    (&kernelContext_126)->g_scan_0 = &g_scan_41;
    if((params_41->shot_count_0) > 0U)
    {
        il_extend_1(id_38.x, 1U, &kernelContext_126);
    }
    return;
}

[[kernel]] void fr_il_strike(uint3 id_39 [[thread_position_in_grid]], FractureParams_0 constant* params_42 [[buffer(0)]], uint device* work_42 [[buffer(4)]], packed_float4 device* frames_42 [[buffer(6)]], packed_float4 device* bodies_42 [[buffer(7)]], packed_float4 device* chunks_42 [[buffer(1)]], uint device* sort_42 [[buffer(11)]], atomic<uint> device* atom_42 [[buffer(5)]], Record_natural_0 device* manifolds_42 [[buffer(10)]], packed_float4 device* loads_42 [[buffer(12)]], packed_uint4 device* bonds_42 [[buffer(2)]], uint device* args_42 [[buffer(13)]], uint device* info_42 [[buffer(15)]], ScanParams_0 constant* scan_42 [[buffer(14)]], packed_float4 device* parts_42 [[buffer(3)]], packed_float4 device* colliders_42 [[buffer(8)]], packed_uint4 device* collider_info_42 [[buffer(9)]], packed_float4 device* view_chunks_42 [[buffer(17)]], ViewParams_0 constant* view_42 [[buffer(16)]], packed_float4 device* objects_42 [[buffer(18)]], uint device* shot_bodies_42 [[buffer(20)]], float device* lines_42 [[buffer(19)]], packed_float4 device* geometry_42 [[buffer(21)]], packed_float4 device* shots_42 [[buffer(22)]], packed_float4 device* lp_42 [[buffer(24)]], packed_float4 device* ext_44 [[buffer(23)]], uint device* engine_counters_42 [[buffer(26)]], packed_float4 device* engine_contacts_42 [[buffer(25)]], packed_float4 device* cs_anchors_42 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_127;
    (&kernelContext_127)->params_0 = params_42;
    (&kernelContext_127)->work_0 = work_42;
    (&kernelContext_127)->frames_0 = frames_42;
    (&kernelContext_127)->bodies_0 = bodies_42;
    (&kernelContext_127)->chunks_0 = chunks_42;
    (&kernelContext_127)->sort_0 = sort_42;
    (&kernelContext_127)->atom_0 = atom_42;
    (&kernelContext_127)->manifolds_0 = manifolds_42;
    (&kernelContext_127)->loads_0 = loads_42;
    (&kernelContext_127)->bonds_0 = bonds_42;
    (&kernelContext_127)->args_0 = args_42;
    (&kernelContext_127)->info_0 = info_42;
    (&kernelContext_127)->scan_0 = scan_42;
    (&kernelContext_127)->parts_0 = parts_42;
    (&kernelContext_127)->colliders_0 = colliders_42;
    (&kernelContext_127)->collider_info_0 = collider_info_42;
    (&kernelContext_127)->view_chunks_0 = view_chunks_42;
    (&kernelContext_127)->view_0 = view_42;
    (&kernelContext_127)->objects_0 = objects_42;
    (&kernelContext_127)->shot_bodies_0 = shot_bodies_42;
    (&kernelContext_127)->lines_0 = lines_42;
    (&kernelContext_127)->geometry_0 = geometry_42;
    (&kernelContext_127)->shots_0 = shots_42;
    (&kernelContext_127)->lp_0 = lp_42;
    (&kernelContext_127)->ext_0 = ext_44;
    (&kernelContext_127)->engine_counters_0 = engine_counters_42;
    (&kernelContext_127)->engine_contacts_0 = engine_contacts_42;
    (&kernelContext_127)->cs_anchors_0 = cs_anchors_42;
    threadgroup array<uint, int(1024)> g_scan_42;
    (&kernelContext_127)->g_scan_0 = &g_scan_42;
    if((params_42->shot_count_0) > 0U)
    {
        il_strike_1(id_39.x, &kernelContext_127);
    }
    return;
}

[[kernel]] void fr_il_mark(uint3 id_40 [[thread_position_in_grid]], FractureParams_0 constant* params_43 [[buffer(0)]], uint device* work_43 [[buffer(4)]], packed_float4 device* frames_43 [[buffer(6)]], packed_float4 device* bodies_43 [[buffer(7)]], packed_float4 device* chunks_43 [[buffer(1)]], uint device* sort_43 [[buffer(11)]], atomic<uint> device* atom_43 [[buffer(5)]], Record_natural_0 device* manifolds_43 [[buffer(10)]], packed_float4 device* loads_43 [[buffer(12)]], packed_uint4 device* bonds_43 [[buffer(2)]], uint device* args_43 [[buffer(13)]], uint device* info_43 [[buffer(15)]], ScanParams_0 constant* scan_43 [[buffer(14)]], packed_float4 device* parts_43 [[buffer(3)]], packed_float4 device* colliders_43 [[buffer(8)]], packed_uint4 device* collider_info_43 [[buffer(9)]], packed_float4 device* view_chunks_43 [[buffer(17)]], ViewParams_0 constant* view_43 [[buffer(16)]], packed_float4 device* objects_43 [[buffer(18)]], uint device* shot_bodies_43 [[buffer(20)]], float device* lines_43 [[buffer(19)]], packed_float4 device* geometry_43 [[buffer(21)]], packed_float4 device* shots_43 [[buffer(22)]], packed_float4 device* lp_43 [[buffer(24)]], packed_float4 device* ext_45 [[buffer(23)]], uint device* engine_counters_43 [[buffer(26)]], packed_float4 device* engine_contacts_43 [[buffer(25)]], packed_float4 device* cs_anchors_43 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_128;
    (&kernelContext_128)->params_0 = params_43;
    (&kernelContext_128)->work_0 = work_43;
    (&kernelContext_128)->frames_0 = frames_43;
    (&kernelContext_128)->bodies_0 = bodies_43;
    (&kernelContext_128)->chunks_0 = chunks_43;
    (&kernelContext_128)->sort_0 = sort_43;
    (&kernelContext_128)->atom_0 = atom_43;
    (&kernelContext_128)->manifolds_0 = manifolds_43;
    (&kernelContext_128)->loads_0 = loads_43;
    (&kernelContext_128)->bonds_0 = bonds_43;
    (&kernelContext_128)->args_0 = args_43;
    (&kernelContext_128)->info_0 = info_43;
    (&kernelContext_128)->scan_0 = scan_43;
    (&kernelContext_128)->parts_0 = parts_43;
    (&kernelContext_128)->colliders_0 = colliders_43;
    (&kernelContext_128)->collider_info_0 = collider_info_43;
    (&kernelContext_128)->view_chunks_0 = view_chunks_43;
    (&kernelContext_128)->view_0 = view_43;
    (&kernelContext_128)->objects_0 = objects_43;
    (&kernelContext_128)->shot_bodies_0 = shot_bodies_43;
    (&kernelContext_128)->lines_0 = lines_43;
    (&kernelContext_128)->geometry_0 = geometry_43;
    (&kernelContext_128)->shots_0 = shots_43;
    (&kernelContext_128)->lp_0 = lp_43;
    (&kernelContext_128)->ext_0 = ext_45;
    (&kernelContext_128)->engine_counters_0 = engine_counters_43;
    (&kernelContext_128)->engine_contacts_0 = engine_contacts_43;
    (&kernelContext_128)->cs_anchors_0 = cs_anchors_43;
    threadgroup array<uint, int(1024)> g_scan_43;
    (&kernelContext_128)->g_scan_0 = &g_scan_43;
    if((params_43->shot_count_0) > 0U)
    {
        il_mark_1(id_40.x, &kernelContext_128);
    }
    return;
}

[[kernel]] void fr_il_carry(uint3 id_41 [[thread_position_in_grid]], FractureParams_0 constant* params_44 [[buffer(0)]], uint device* work_44 [[buffer(4)]], packed_float4 device* frames_44 [[buffer(6)]], packed_float4 device* bodies_44 [[buffer(7)]], packed_float4 device* chunks_44 [[buffer(1)]], uint device* sort_44 [[buffer(11)]], atomic<uint> device* atom_44 [[buffer(5)]], Record_natural_0 device* manifolds_44 [[buffer(10)]], packed_float4 device* loads_44 [[buffer(12)]], packed_uint4 device* bonds_44 [[buffer(2)]], uint device* args_44 [[buffer(13)]], uint device* info_44 [[buffer(15)]], ScanParams_0 constant* scan_44 [[buffer(14)]], packed_float4 device* parts_44 [[buffer(3)]], packed_float4 device* colliders_44 [[buffer(8)]], packed_uint4 device* collider_info_44 [[buffer(9)]], packed_float4 device* view_chunks_44 [[buffer(17)]], ViewParams_0 constant* view_44 [[buffer(16)]], packed_float4 device* objects_44 [[buffer(18)]], uint device* shot_bodies_44 [[buffer(20)]], float device* lines_44 [[buffer(19)]], packed_float4 device* geometry_44 [[buffer(21)]], packed_float4 device* shots_44 [[buffer(22)]], packed_float4 device* lp_44 [[buffer(24)]], packed_float4 device* ext_46 [[buffer(23)]], uint device* engine_counters_44 [[buffer(26)]], packed_float4 device* engine_contacts_44 [[buffer(25)]], packed_float4 device* cs_anchors_44 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_129;
    (&kernelContext_129)->params_0 = params_44;
    (&kernelContext_129)->work_0 = work_44;
    (&kernelContext_129)->frames_0 = frames_44;
    (&kernelContext_129)->bodies_0 = bodies_44;
    (&kernelContext_129)->chunks_0 = chunks_44;
    (&kernelContext_129)->sort_0 = sort_44;
    (&kernelContext_129)->atom_0 = atom_44;
    (&kernelContext_129)->manifolds_0 = manifolds_44;
    (&kernelContext_129)->loads_0 = loads_44;
    (&kernelContext_129)->bonds_0 = bonds_44;
    (&kernelContext_129)->args_0 = args_44;
    (&kernelContext_129)->info_0 = info_44;
    (&kernelContext_129)->scan_0 = scan_44;
    (&kernelContext_129)->parts_0 = parts_44;
    (&kernelContext_129)->colliders_0 = colliders_44;
    (&kernelContext_129)->collider_info_0 = collider_info_44;
    (&kernelContext_129)->view_chunks_0 = view_chunks_44;
    (&kernelContext_129)->view_0 = view_44;
    (&kernelContext_129)->objects_0 = objects_44;
    (&kernelContext_129)->shot_bodies_0 = shot_bodies_44;
    (&kernelContext_129)->lines_0 = lines_44;
    (&kernelContext_129)->geometry_0 = geometry_44;
    (&kernelContext_129)->shots_0 = shots_44;
    (&kernelContext_129)->lp_0 = lp_44;
    (&kernelContext_129)->ext_0 = ext_46;
    (&kernelContext_129)->engine_counters_0 = engine_counters_44;
    (&kernelContext_129)->engine_contacts_0 = engine_contacts_44;
    (&kernelContext_129)->cs_anchors_0 = cs_anchors_44;
    threadgroup array<uint, int(1024)> g_scan_44;
    (&kernelContext_129)->g_scan_0 = &g_scan_44;
    if((params_44->shot_count_0) > 0U)
    {
        il_carry_1(id_41.x, &kernelContext_129);
    }
    return;
}

[[kernel]] void fr_il_breaks(uint3 id_42 [[thread_position_in_grid]], FractureParams_0 constant* params_45 [[buffer(0)]], uint device* work_45 [[buffer(4)]], packed_float4 device* frames_45 [[buffer(6)]], packed_float4 device* bodies_45 [[buffer(7)]], packed_float4 device* chunks_45 [[buffer(1)]], uint device* sort_45 [[buffer(11)]], atomic<uint> device* atom_45 [[buffer(5)]], Record_natural_0 device* manifolds_45 [[buffer(10)]], packed_float4 device* loads_45 [[buffer(12)]], packed_uint4 device* bonds_45 [[buffer(2)]], uint device* args_45 [[buffer(13)]], uint device* info_45 [[buffer(15)]], ScanParams_0 constant* scan_45 [[buffer(14)]], packed_float4 device* parts_45 [[buffer(3)]], packed_float4 device* colliders_45 [[buffer(8)]], packed_uint4 device* collider_info_45 [[buffer(9)]], packed_float4 device* view_chunks_45 [[buffer(17)]], ViewParams_0 constant* view_45 [[buffer(16)]], packed_float4 device* objects_45 [[buffer(18)]], uint device* shot_bodies_45 [[buffer(20)]], float device* lines_45 [[buffer(19)]], packed_float4 device* geometry_45 [[buffer(21)]], packed_float4 device* shots_45 [[buffer(22)]], packed_float4 device* lp_45 [[buffer(24)]], packed_float4 device* ext_47 [[buffer(23)]], uint device* engine_counters_45 [[buffer(26)]], packed_float4 device* engine_contacts_45 [[buffer(25)]], packed_float4 device* cs_anchors_45 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_130;
    (&kernelContext_130)->params_0 = params_45;
    (&kernelContext_130)->work_0 = work_45;
    (&kernelContext_130)->frames_0 = frames_45;
    (&kernelContext_130)->bodies_0 = bodies_45;
    (&kernelContext_130)->chunks_0 = chunks_45;
    (&kernelContext_130)->sort_0 = sort_45;
    (&kernelContext_130)->atom_0 = atom_45;
    (&kernelContext_130)->manifolds_0 = manifolds_45;
    (&kernelContext_130)->loads_0 = loads_45;
    (&kernelContext_130)->bonds_0 = bonds_45;
    (&kernelContext_130)->args_0 = args_45;
    (&kernelContext_130)->info_0 = info_45;
    (&kernelContext_130)->scan_0 = scan_45;
    (&kernelContext_130)->parts_0 = parts_45;
    (&kernelContext_130)->colliders_0 = colliders_45;
    (&kernelContext_130)->collider_info_0 = collider_info_45;
    (&kernelContext_130)->view_chunks_0 = view_chunks_45;
    (&kernelContext_130)->view_0 = view_45;
    (&kernelContext_130)->objects_0 = objects_45;
    (&kernelContext_130)->shot_bodies_0 = shot_bodies_45;
    (&kernelContext_130)->lines_0 = lines_45;
    (&kernelContext_130)->geometry_0 = geometry_45;
    (&kernelContext_130)->shots_0 = shots_45;
    (&kernelContext_130)->lp_0 = lp_45;
    (&kernelContext_130)->ext_0 = ext_47;
    (&kernelContext_130)->engine_counters_0 = engine_counters_45;
    (&kernelContext_130)->engine_contacts_0 = engine_contacts_45;
    (&kernelContext_130)->cs_anchors_0 = cs_anchors_45;
    threadgroup array<uint, int(1024)> g_scan_45;
    (&kernelContext_130)->g_scan_0 = &g_scan_45;
    if((params_45->shot_count_0) > 0U)
    {
        il_breaks_1(id_42.x, &kernelContext_130);
    }
    return;
}

[[kernel]] void fr_csl_begin(uint3 id_43 [[thread_position_in_grid]], FractureParams_0 constant* params_46 [[buffer(0)]], uint device* work_46 [[buffer(4)]], packed_float4 device* frames_46 [[buffer(6)]], packed_float4 device* bodies_46 [[buffer(7)]], packed_float4 device* chunks_46 [[buffer(1)]], uint device* sort_46 [[buffer(11)]], atomic<uint> device* atom_46 [[buffer(5)]], Record_natural_0 device* manifolds_46 [[buffer(10)]], packed_float4 device* loads_46 [[buffer(12)]], packed_uint4 device* bonds_46 [[buffer(2)]], uint device* args_46 [[buffer(13)]], uint device* info_46 [[buffer(15)]], ScanParams_0 constant* scan_46 [[buffer(14)]], packed_float4 device* parts_46 [[buffer(3)]], packed_float4 device* colliders_46 [[buffer(8)]], packed_uint4 device* collider_info_46 [[buffer(9)]], packed_float4 device* view_chunks_46 [[buffer(17)]], ViewParams_0 constant* view_46 [[buffer(16)]], packed_float4 device* objects_46 [[buffer(18)]], uint device* shot_bodies_46 [[buffer(20)]], float device* lines_46 [[buffer(19)]], packed_float4 device* geometry_46 [[buffer(21)]], packed_float4 device* shots_46 [[buffer(22)]], packed_float4 device* lp_46 [[buffer(24)]], packed_float4 device* ext_48 [[buffer(23)]], uint device* engine_counters_46 [[buffer(26)]], packed_float4 device* engine_contacts_46 [[buffer(25)]], packed_float4 device* cs_anchors_46 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_131;
    (&kernelContext_131)->params_0 = params_46;
    (&kernelContext_131)->work_0 = work_46;
    (&kernelContext_131)->frames_0 = frames_46;
    (&kernelContext_131)->bodies_0 = bodies_46;
    (&kernelContext_131)->chunks_0 = chunks_46;
    (&kernelContext_131)->sort_0 = sort_46;
    (&kernelContext_131)->atom_0 = atom_46;
    (&kernelContext_131)->manifolds_0 = manifolds_46;
    (&kernelContext_131)->loads_0 = loads_46;
    (&kernelContext_131)->bonds_0 = bonds_46;
    (&kernelContext_131)->args_0 = args_46;
    (&kernelContext_131)->info_0 = info_46;
    (&kernelContext_131)->scan_0 = scan_46;
    (&kernelContext_131)->parts_0 = parts_46;
    (&kernelContext_131)->colliders_0 = colliders_46;
    (&kernelContext_131)->collider_info_0 = collider_info_46;
    (&kernelContext_131)->view_chunks_0 = view_chunks_46;
    (&kernelContext_131)->view_0 = view_46;
    (&kernelContext_131)->objects_0 = objects_46;
    (&kernelContext_131)->shot_bodies_0 = shot_bodies_46;
    (&kernelContext_131)->lines_0 = lines_46;
    (&kernelContext_131)->geometry_0 = geometry_46;
    (&kernelContext_131)->shots_0 = shots_46;
    (&kernelContext_131)->lp_0 = lp_46;
    (&kernelContext_131)->ext_0 = ext_48;
    (&kernelContext_131)->engine_counters_0 = engine_counters_46;
    (&kernelContext_131)->engine_contacts_0 = engine_contacts_46;
    (&kernelContext_131)->cs_anchors_0 = cs_anchors_46;
    threadgroup array<uint, int(1024)> g_scan_46;
    (&kernelContext_131)->g_scan_0 = &g_scan_46;
    il_begin_0(id_43.x, &kernelContext_131);
    return;
}

[[kernel]] void fr_csl_extend_even(uint3 id_44 [[thread_position_in_grid]], FractureParams_0 constant* params_47 [[buffer(0)]], uint device* work_47 [[buffer(4)]], packed_float4 device* frames_47 [[buffer(6)]], packed_float4 device* bodies_47 [[buffer(7)]], packed_float4 device* chunks_47 [[buffer(1)]], uint device* sort_47 [[buffer(11)]], atomic<uint> device* atom_47 [[buffer(5)]], Record_natural_0 device* manifolds_47 [[buffer(10)]], packed_float4 device* loads_47 [[buffer(12)]], packed_uint4 device* bonds_47 [[buffer(2)]], uint device* args_47 [[buffer(13)]], uint device* info_47 [[buffer(15)]], ScanParams_0 constant* scan_47 [[buffer(14)]], packed_float4 device* parts_47 [[buffer(3)]], packed_float4 device* colliders_47 [[buffer(8)]], packed_uint4 device* collider_info_47 [[buffer(9)]], packed_float4 device* view_chunks_47 [[buffer(17)]], ViewParams_0 constant* view_47 [[buffer(16)]], packed_float4 device* objects_47 [[buffer(18)]], uint device* shot_bodies_47 [[buffer(20)]], float device* lines_47 [[buffer(19)]], packed_float4 device* geometry_47 [[buffer(21)]], packed_float4 device* shots_47 [[buffer(22)]], packed_float4 device* lp_47 [[buffer(24)]], packed_float4 device* ext_49 [[buffer(23)]], uint device* engine_counters_47 [[buffer(26)]], packed_float4 device* engine_contacts_47 [[buffer(25)]], packed_float4 device* cs_anchors_47 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_132;
    (&kernelContext_132)->params_0 = params_47;
    (&kernelContext_132)->work_0 = work_47;
    (&kernelContext_132)->frames_0 = frames_47;
    (&kernelContext_132)->bodies_0 = bodies_47;
    (&kernelContext_132)->chunks_0 = chunks_47;
    (&kernelContext_132)->sort_0 = sort_47;
    (&kernelContext_132)->atom_0 = atom_47;
    (&kernelContext_132)->manifolds_0 = manifolds_47;
    (&kernelContext_132)->loads_0 = loads_47;
    (&kernelContext_132)->bonds_0 = bonds_47;
    (&kernelContext_132)->args_0 = args_47;
    (&kernelContext_132)->info_0 = info_47;
    (&kernelContext_132)->scan_0 = scan_47;
    (&kernelContext_132)->parts_0 = parts_47;
    (&kernelContext_132)->colliders_0 = colliders_47;
    (&kernelContext_132)->collider_info_0 = collider_info_47;
    (&kernelContext_132)->view_chunks_0 = view_chunks_47;
    (&kernelContext_132)->view_0 = view_47;
    (&kernelContext_132)->objects_0 = objects_47;
    (&kernelContext_132)->shot_bodies_0 = shot_bodies_47;
    (&kernelContext_132)->lines_0 = lines_47;
    (&kernelContext_132)->geometry_0 = geometry_47;
    (&kernelContext_132)->shots_0 = shots_47;
    (&kernelContext_132)->lp_0 = lp_47;
    (&kernelContext_132)->ext_0 = ext_49;
    (&kernelContext_132)->engine_counters_0 = engine_counters_47;
    (&kernelContext_132)->engine_contacts_0 = engine_contacts_47;
    (&kernelContext_132)->cs_anchors_0 = cs_anchors_47;
    threadgroup array<uint, int(1024)> g_scan_47;
    (&kernelContext_132)->g_scan_0 = &g_scan_47;
    il_extend_0(id_44.x, 0U, &kernelContext_132);
    return;
}

[[kernel]] void fr_csl_extend_odd(uint3 id_45 [[thread_position_in_grid]], FractureParams_0 constant* params_48 [[buffer(0)]], uint device* work_48 [[buffer(4)]], packed_float4 device* frames_48 [[buffer(6)]], packed_float4 device* bodies_48 [[buffer(7)]], packed_float4 device* chunks_48 [[buffer(1)]], uint device* sort_48 [[buffer(11)]], atomic<uint> device* atom_48 [[buffer(5)]], Record_natural_0 device* manifolds_48 [[buffer(10)]], packed_float4 device* loads_48 [[buffer(12)]], packed_uint4 device* bonds_48 [[buffer(2)]], uint device* args_48 [[buffer(13)]], uint device* info_48 [[buffer(15)]], ScanParams_0 constant* scan_48 [[buffer(14)]], packed_float4 device* parts_48 [[buffer(3)]], packed_float4 device* colliders_48 [[buffer(8)]], packed_uint4 device* collider_info_48 [[buffer(9)]], packed_float4 device* view_chunks_48 [[buffer(17)]], ViewParams_0 constant* view_48 [[buffer(16)]], packed_float4 device* objects_48 [[buffer(18)]], uint device* shot_bodies_48 [[buffer(20)]], float device* lines_48 [[buffer(19)]], packed_float4 device* geometry_48 [[buffer(21)]], packed_float4 device* shots_48 [[buffer(22)]], packed_float4 device* lp_48 [[buffer(24)]], packed_float4 device* ext_50 [[buffer(23)]], uint device* engine_counters_48 [[buffer(26)]], packed_float4 device* engine_contacts_48 [[buffer(25)]], packed_float4 device* cs_anchors_48 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_133;
    (&kernelContext_133)->params_0 = params_48;
    (&kernelContext_133)->work_0 = work_48;
    (&kernelContext_133)->frames_0 = frames_48;
    (&kernelContext_133)->bodies_0 = bodies_48;
    (&kernelContext_133)->chunks_0 = chunks_48;
    (&kernelContext_133)->sort_0 = sort_48;
    (&kernelContext_133)->atom_0 = atom_48;
    (&kernelContext_133)->manifolds_0 = manifolds_48;
    (&kernelContext_133)->loads_0 = loads_48;
    (&kernelContext_133)->bonds_0 = bonds_48;
    (&kernelContext_133)->args_0 = args_48;
    (&kernelContext_133)->info_0 = info_48;
    (&kernelContext_133)->scan_0 = scan_48;
    (&kernelContext_133)->parts_0 = parts_48;
    (&kernelContext_133)->colliders_0 = colliders_48;
    (&kernelContext_133)->collider_info_0 = collider_info_48;
    (&kernelContext_133)->view_chunks_0 = view_chunks_48;
    (&kernelContext_133)->view_0 = view_48;
    (&kernelContext_133)->objects_0 = objects_48;
    (&kernelContext_133)->shot_bodies_0 = shot_bodies_48;
    (&kernelContext_133)->lines_0 = lines_48;
    (&kernelContext_133)->geometry_0 = geometry_48;
    (&kernelContext_133)->shots_0 = shots_48;
    (&kernelContext_133)->lp_0 = lp_48;
    (&kernelContext_133)->ext_0 = ext_50;
    (&kernelContext_133)->engine_counters_0 = engine_counters_48;
    (&kernelContext_133)->engine_contacts_0 = engine_contacts_48;
    (&kernelContext_133)->cs_anchors_0 = cs_anchors_48;
    threadgroup array<uint, int(1024)> g_scan_48;
    (&kernelContext_133)->g_scan_0 = &g_scan_48;
    il_extend_0(id_45.x, 1U, &kernelContext_133);
    return;
}

[[kernel]] void fr_csl_strike(uint3 id_46 [[thread_position_in_grid]], FractureParams_0 constant* params_49 [[buffer(0)]], uint device* work_49 [[buffer(4)]], packed_float4 device* frames_49 [[buffer(6)]], packed_float4 device* bodies_49 [[buffer(7)]], packed_float4 device* chunks_49 [[buffer(1)]], uint device* sort_49 [[buffer(11)]], atomic<uint> device* atom_49 [[buffer(5)]], Record_natural_0 device* manifolds_49 [[buffer(10)]], packed_float4 device* loads_49 [[buffer(12)]], packed_uint4 device* bonds_49 [[buffer(2)]], uint device* args_49 [[buffer(13)]], uint device* info_49 [[buffer(15)]], ScanParams_0 constant* scan_49 [[buffer(14)]], packed_float4 device* parts_49 [[buffer(3)]], packed_float4 device* colliders_49 [[buffer(8)]], packed_uint4 device* collider_info_49 [[buffer(9)]], packed_float4 device* view_chunks_49 [[buffer(17)]], ViewParams_0 constant* view_49 [[buffer(16)]], packed_float4 device* objects_49 [[buffer(18)]], uint device* shot_bodies_49 [[buffer(20)]], float device* lines_49 [[buffer(19)]], packed_float4 device* geometry_49 [[buffer(21)]], packed_float4 device* shots_49 [[buffer(22)]], packed_float4 device* lp_49 [[buffer(24)]], packed_float4 device* ext_51 [[buffer(23)]], uint device* engine_counters_49 [[buffer(26)]], packed_float4 device* engine_contacts_49 [[buffer(25)]], packed_float4 device* cs_anchors_49 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_134;
    (&kernelContext_134)->params_0 = params_49;
    (&kernelContext_134)->work_0 = work_49;
    (&kernelContext_134)->frames_0 = frames_49;
    (&kernelContext_134)->bodies_0 = bodies_49;
    (&kernelContext_134)->chunks_0 = chunks_49;
    (&kernelContext_134)->sort_0 = sort_49;
    (&kernelContext_134)->atom_0 = atom_49;
    (&kernelContext_134)->manifolds_0 = manifolds_49;
    (&kernelContext_134)->loads_0 = loads_49;
    (&kernelContext_134)->bonds_0 = bonds_49;
    (&kernelContext_134)->args_0 = args_49;
    (&kernelContext_134)->info_0 = info_49;
    (&kernelContext_134)->scan_0 = scan_49;
    (&kernelContext_134)->parts_0 = parts_49;
    (&kernelContext_134)->colliders_0 = colliders_49;
    (&kernelContext_134)->collider_info_0 = collider_info_49;
    (&kernelContext_134)->view_chunks_0 = view_chunks_49;
    (&kernelContext_134)->view_0 = view_49;
    (&kernelContext_134)->objects_0 = objects_49;
    (&kernelContext_134)->shot_bodies_0 = shot_bodies_49;
    (&kernelContext_134)->lines_0 = lines_49;
    (&kernelContext_134)->geometry_0 = geometry_49;
    (&kernelContext_134)->shots_0 = shots_49;
    (&kernelContext_134)->lp_0 = lp_49;
    (&kernelContext_134)->ext_0 = ext_51;
    (&kernelContext_134)->engine_counters_0 = engine_counters_49;
    (&kernelContext_134)->engine_contacts_0 = engine_contacts_49;
    (&kernelContext_134)->cs_anchors_0 = cs_anchors_49;
    threadgroup array<uint, int(1024)> g_scan_49;
    (&kernelContext_134)->g_scan_0 = &g_scan_49;
    il_strike_0(id_46.x, &kernelContext_134);
    return;
}

[[kernel]] void fr_csl_mark(uint3 id_47 [[thread_position_in_grid]], FractureParams_0 constant* params_50 [[buffer(0)]], uint device* work_50 [[buffer(4)]], packed_float4 device* frames_50 [[buffer(6)]], packed_float4 device* bodies_50 [[buffer(7)]], packed_float4 device* chunks_50 [[buffer(1)]], uint device* sort_50 [[buffer(11)]], atomic<uint> device* atom_50 [[buffer(5)]], Record_natural_0 device* manifolds_50 [[buffer(10)]], packed_float4 device* loads_50 [[buffer(12)]], packed_uint4 device* bonds_50 [[buffer(2)]], uint device* args_50 [[buffer(13)]], uint device* info_50 [[buffer(15)]], ScanParams_0 constant* scan_50 [[buffer(14)]], packed_float4 device* parts_50 [[buffer(3)]], packed_float4 device* colliders_50 [[buffer(8)]], packed_uint4 device* collider_info_50 [[buffer(9)]], packed_float4 device* view_chunks_50 [[buffer(17)]], ViewParams_0 constant* view_50 [[buffer(16)]], packed_float4 device* objects_50 [[buffer(18)]], uint device* shot_bodies_50 [[buffer(20)]], float device* lines_50 [[buffer(19)]], packed_float4 device* geometry_50 [[buffer(21)]], packed_float4 device* shots_50 [[buffer(22)]], packed_float4 device* lp_50 [[buffer(24)]], packed_float4 device* ext_52 [[buffer(23)]], uint device* engine_counters_50 [[buffer(26)]], packed_float4 device* engine_contacts_50 [[buffer(25)]], packed_float4 device* cs_anchors_50 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_135;
    (&kernelContext_135)->params_0 = params_50;
    (&kernelContext_135)->work_0 = work_50;
    (&kernelContext_135)->frames_0 = frames_50;
    (&kernelContext_135)->bodies_0 = bodies_50;
    (&kernelContext_135)->chunks_0 = chunks_50;
    (&kernelContext_135)->sort_0 = sort_50;
    (&kernelContext_135)->atom_0 = atom_50;
    (&kernelContext_135)->manifolds_0 = manifolds_50;
    (&kernelContext_135)->loads_0 = loads_50;
    (&kernelContext_135)->bonds_0 = bonds_50;
    (&kernelContext_135)->args_0 = args_50;
    (&kernelContext_135)->info_0 = info_50;
    (&kernelContext_135)->scan_0 = scan_50;
    (&kernelContext_135)->parts_0 = parts_50;
    (&kernelContext_135)->colliders_0 = colliders_50;
    (&kernelContext_135)->collider_info_0 = collider_info_50;
    (&kernelContext_135)->view_chunks_0 = view_chunks_50;
    (&kernelContext_135)->view_0 = view_50;
    (&kernelContext_135)->objects_0 = objects_50;
    (&kernelContext_135)->shot_bodies_0 = shot_bodies_50;
    (&kernelContext_135)->lines_0 = lines_50;
    (&kernelContext_135)->geometry_0 = geometry_50;
    (&kernelContext_135)->shots_0 = shots_50;
    (&kernelContext_135)->lp_0 = lp_50;
    (&kernelContext_135)->ext_0 = ext_52;
    (&kernelContext_135)->engine_counters_0 = engine_counters_50;
    (&kernelContext_135)->engine_contacts_0 = engine_contacts_50;
    (&kernelContext_135)->cs_anchors_0 = cs_anchors_50;
    threadgroup array<uint, int(1024)> g_scan_50;
    (&kernelContext_135)->g_scan_0 = &g_scan_50;
    il_mark_0(id_47.x, &kernelContext_135);
    return;
}

[[kernel]] void fr_csl_carry(uint3 id_48 [[thread_position_in_grid]], FractureParams_0 constant* params_51 [[buffer(0)]], uint device* work_51 [[buffer(4)]], packed_float4 device* frames_51 [[buffer(6)]], packed_float4 device* bodies_51 [[buffer(7)]], packed_float4 device* chunks_51 [[buffer(1)]], uint device* sort_51 [[buffer(11)]], atomic<uint> device* atom_51 [[buffer(5)]], Record_natural_0 device* manifolds_51 [[buffer(10)]], packed_float4 device* loads_51 [[buffer(12)]], packed_uint4 device* bonds_51 [[buffer(2)]], uint device* args_51 [[buffer(13)]], uint device* info_51 [[buffer(15)]], ScanParams_0 constant* scan_51 [[buffer(14)]], packed_float4 device* parts_51 [[buffer(3)]], packed_float4 device* colliders_51 [[buffer(8)]], packed_uint4 device* collider_info_51 [[buffer(9)]], packed_float4 device* view_chunks_51 [[buffer(17)]], ViewParams_0 constant* view_51 [[buffer(16)]], packed_float4 device* objects_51 [[buffer(18)]], uint device* shot_bodies_51 [[buffer(20)]], float device* lines_51 [[buffer(19)]], packed_float4 device* geometry_51 [[buffer(21)]], packed_float4 device* shots_51 [[buffer(22)]], packed_float4 device* lp_51 [[buffer(24)]], packed_float4 device* ext_53 [[buffer(23)]], uint device* engine_counters_51 [[buffer(26)]], packed_float4 device* engine_contacts_51 [[buffer(25)]], packed_float4 device* cs_anchors_51 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_136;
    (&kernelContext_136)->params_0 = params_51;
    (&kernelContext_136)->work_0 = work_51;
    (&kernelContext_136)->frames_0 = frames_51;
    (&kernelContext_136)->bodies_0 = bodies_51;
    (&kernelContext_136)->chunks_0 = chunks_51;
    (&kernelContext_136)->sort_0 = sort_51;
    (&kernelContext_136)->atom_0 = atom_51;
    (&kernelContext_136)->manifolds_0 = manifolds_51;
    (&kernelContext_136)->loads_0 = loads_51;
    (&kernelContext_136)->bonds_0 = bonds_51;
    (&kernelContext_136)->args_0 = args_51;
    (&kernelContext_136)->info_0 = info_51;
    (&kernelContext_136)->scan_0 = scan_51;
    (&kernelContext_136)->parts_0 = parts_51;
    (&kernelContext_136)->colliders_0 = colliders_51;
    (&kernelContext_136)->collider_info_0 = collider_info_51;
    (&kernelContext_136)->view_chunks_0 = view_chunks_51;
    (&kernelContext_136)->view_0 = view_51;
    (&kernelContext_136)->objects_0 = objects_51;
    (&kernelContext_136)->shot_bodies_0 = shot_bodies_51;
    (&kernelContext_136)->lines_0 = lines_51;
    (&kernelContext_136)->geometry_0 = geometry_51;
    (&kernelContext_136)->shots_0 = shots_51;
    (&kernelContext_136)->lp_0 = lp_51;
    (&kernelContext_136)->ext_0 = ext_53;
    (&kernelContext_136)->engine_counters_0 = engine_counters_51;
    (&kernelContext_136)->engine_contacts_0 = engine_contacts_51;
    (&kernelContext_136)->cs_anchors_0 = cs_anchors_51;
    threadgroup array<uint, int(1024)> g_scan_51;
    (&kernelContext_136)->g_scan_0 = &g_scan_51;
    il_carry_0(id_48.x, &kernelContext_136);
    return;
}

[[kernel]] void fr_csl_breaks(uint3 id_49 [[thread_position_in_grid]], FractureParams_0 constant* params_52 [[buffer(0)]], uint device* work_52 [[buffer(4)]], packed_float4 device* frames_52 [[buffer(6)]], packed_float4 device* bodies_52 [[buffer(7)]], packed_float4 device* chunks_52 [[buffer(1)]], uint device* sort_52 [[buffer(11)]], atomic<uint> device* atom_52 [[buffer(5)]], Record_natural_0 device* manifolds_52 [[buffer(10)]], packed_float4 device* loads_52 [[buffer(12)]], packed_uint4 device* bonds_52 [[buffer(2)]], uint device* args_52 [[buffer(13)]], uint device* info_52 [[buffer(15)]], ScanParams_0 constant* scan_52 [[buffer(14)]], packed_float4 device* parts_52 [[buffer(3)]], packed_float4 device* colliders_52 [[buffer(8)]], packed_uint4 device* collider_info_52 [[buffer(9)]], packed_float4 device* view_chunks_52 [[buffer(17)]], ViewParams_0 constant* view_52 [[buffer(16)]], packed_float4 device* objects_52 [[buffer(18)]], uint device* shot_bodies_52 [[buffer(20)]], float device* lines_52 [[buffer(19)]], packed_float4 device* geometry_52 [[buffer(21)]], packed_float4 device* shots_52 [[buffer(22)]], packed_float4 device* lp_52 [[buffer(24)]], packed_float4 device* ext_54 [[buffer(23)]], uint device* engine_counters_52 [[buffer(26)]], packed_float4 device* engine_contacts_52 [[buffer(25)]], packed_float4 device* cs_anchors_52 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_137;
    (&kernelContext_137)->params_0 = params_52;
    (&kernelContext_137)->work_0 = work_52;
    (&kernelContext_137)->frames_0 = frames_52;
    (&kernelContext_137)->bodies_0 = bodies_52;
    (&kernelContext_137)->chunks_0 = chunks_52;
    (&kernelContext_137)->sort_0 = sort_52;
    (&kernelContext_137)->atom_0 = atom_52;
    (&kernelContext_137)->manifolds_0 = manifolds_52;
    (&kernelContext_137)->loads_0 = loads_52;
    (&kernelContext_137)->bonds_0 = bonds_52;
    (&kernelContext_137)->args_0 = args_52;
    (&kernelContext_137)->info_0 = info_52;
    (&kernelContext_137)->scan_0 = scan_52;
    (&kernelContext_137)->parts_0 = parts_52;
    (&kernelContext_137)->colliders_0 = colliders_52;
    (&kernelContext_137)->collider_info_0 = collider_info_52;
    (&kernelContext_137)->view_chunks_0 = view_chunks_52;
    (&kernelContext_137)->view_0 = view_52;
    (&kernelContext_137)->objects_0 = objects_52;
    (&kernelContext_137)->shot_bodies_0 = shot_bodies_52;
    (&kernelContext_137)->lines_0 = lines_52;
    (&kernelContext_137)->geometry_0 = geometry_52;
    (&kernelContext_137)->shots_0 = shots_52;
    (&kernelContext_137)->lp_0 = lp_52;
    (&kernelContext_137)->ext_0 = ext_54;
    (&kernelContext_137)->engine_counters_0 = engine_counters_52;
    (&kernelContext_137)->engine_contacts_0 = engine_contacts_52;
    (&kernelContext_137)->cs_anchors_0 = cs_anchors_52;
    threadgroup array<uint, int(1024)> g_scan_52;
    (&kernelContext_137)->g_scan_0 = &g_scan_52;
    il_breaks_0(id_49.x, &kernelContext_137);
    return;
}

uint cs_rows_0(KernelContext_0 thread* kernelContext_138)
{
    return min(kernelContext_138->engine_counters_0[int(2)], kernelContext_138->params_0->cs_row_cap_0);
}

[[kernel]] void fr_cs_reset(uint3 id_50 [[thread_position_in_grid]], FractureParams_0 constant* params_53 [[buffer(0)]], uint device* work_53 [[buffer(4)]], packed_float4 device* frames_53 [[buffer(6)]], packed_float4 device* bodies_53 [[buffer(7)]], packed_float4 device* chunks_53 [[buffer(1)]], uint device* sort_53 [[buffer(11)]], atomic<uint> device* atom_53 [[buffer(5)]], Record_natural_0 device* manifolds_53 [[buffer(10)]], packed_float4 device* loads_53 [[buffer(12)]], packed_uint4 device* bonds_53 [[buffer(2)]], uint device* args_53 [[buffer(13)]], uint device* info_53 [[buffer(15)]], ScanParams_0 constant* scan_53 [[buffer(14)]], packed_float4 device* parts_53 [[buffer(3)]], packed_float4 device* colliders_53 [[buffer(8)]], packed_uint4 device* collider_info_53 [[buffer(9)]], packed_float4 device* view_chunks_53 [[buffer(17)]], ViewParams_0 constant* view_53 [[buffer(16)]], packed_float4 device* objects_53 [[buffer(18)]], uint device* shot_bodies_53 [[buffer(20)]], float device* lines_53 [[buffer(19)]], packed_float4 device* geometry_53 [[buffer(21)]], packed_float4 device* shots_53 [[buffer(22)]], packed_float4 device* lp_53 [[buffer(24)]], packed_float4 device* ext_55 [[buffer(23)]], uint device* engine_counters_53 [[buffer(26)]], packed_float4 device* engine_contacts_53 [[buffer(25)]], packed_float4 device* cs_anchors_53 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_139;
    (&kernelContext_139)->params_0 = params_53;
    (&kernelContext_139)->work_0 = work_53;
    (&kernelContext_139)->frames_0 = frames_53;
    (&kernelContext_139)->bodies_0 = bodies_53;
    (&kernelContext_139)->chunks_0 = chunks_53;
    (&kernelContext_139)->sort_0 = sort_53;
    (&kernelContext_139)->atom_0 = atom_53;
    (&kernelContext_139)->manifolds_0 = manifolds_53;
    (&kernelContext_139)->loads_0 = loads_53;
    (&kernelContext_139)->bonds_0 = bonds_53;
    (&kernelContext_139)->args_0 = args_53;
    (&kernelContext_139)->info_0 = info_53;
    (&kernelContext_139)->scan_0 = scan_53;
    (&kernelContext_139)->parts_0 = parts_53;
    (&kernelContext_139)->colliders_0 = colliders_53;
    (&kernelContext_139)->collider_info_0 = collider_info_53;
    (&kernelContext_139)->view_chunks_0 = view_chunks_53;
    (&kernelContext_139)->view_0 = view_53;
    (&kernelContext_139)->objects_0 = objects_53;
    (&kernelContext_139)->shot_bodies_0 = shot_bodies_53;
    (&kernelContext_139)->lines_0 = lines_53;
    (&kernelContext_139)->geometry_0 = geometry_53;
    (&kernelContext_139)->shots_0 = shots_53;
    (&kernelContext_139)->lp_0 = lp_53;
    (&kernelContext_139)->ext_0 = ext_55;
    (&kernelContext_139)->engine_counters_0 = engine_counters_53;
    (&kernelContext_139)->engine_contacts_0 = engine_contacts_53;
    (&kernelContext_139)->cs_anchors_0 = cs_anchors_53;
    threadgroup array<uint, int(1024)> g_scan_53;
    (&kernelContext_139)->g_scan_0 = &g_scan_53;
    uint b_41 = id_50.x;
    if(b_41 == 0U)
    {
        uint _S996 = a_counters_1(&kernelContext_139);
        atomic_store_explicit((&kernelContext_139)->atom_0+(_S996 + 14U), 0U, memory_order_relaxed);
        atomic_store_explicit((&kernelContext_139)->atom_0+(_S996 + 15U), 0U, memory_order_relaxed);
        uint _S997 = cs_rows_0(&kernelContext_139);
        uint _S998 = groups_of_0(_S997, 64U);
        set_args_0(7U, _S998, &kernelContext_139);
    }
    if(b_41 >= ((&kernelContext_139)->params_0->body_count_0))
    {
        return;
    }
    uint _S999 = a_cs_0(b_41, &kernelContext_139);
    atomic_store_explicit((&kernelContext_139)->atom_0+_S999, 0U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_139)->atom_0+(_S999 + 1U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_139)->atom_0+(_S999 + 2U), 4294967295U, memory_order_relaxed);
    atomic_store_explicit((&kernelContext_139)->atom_0+(_S999 + 3U), 4294967295U, memory_order_relaxed);
    uint w_8 = 4U;
    for(;;)
    {
        if(w_8 < 8U)
        {
        }
        else
        {
            break;
        }
        atomic_store_explicit((&kernelContext_139)->atom_0+(_S999 + w_8), 0U, memory_order_relaxed);
        w_8 = w_8 + 1U;
    }
    return;
}

uint cs_row_0(uint r_15, KernelContext_0 thread* kernelContext_140)
{
    return 8U * (kernelContext_140->params_0->cs_row_offset_0 + r_15);
}

float cs_angular_0(uint b_42, float3 r_16, float3 u_3, KernelContext_0 thread* kernelContext_141)
{
    uint at_15 = 7U * b_42;
    thread Quat_0 _S1000 = quat_of_0(float4(*(kernelContext_141->bodies_0+(at_15 + 1U))) );
    Quat_0 _S1001 = conj_0(&_S1000);
    float3 _S1002 = cross(r_16, u_3);
    thread Quat_0 _S1003 = _S1001;
    float3 _S1004 = rotate_0(&_S1003, _S1002);
    return dot(_S1004 * (float4(*(kernelContext_141->bodies_0+(at_15 + 2U))) ).xyz, _S1004);
}

bool cs_side_0(uint r_17, uint side_0, uint thread* k_33, uint thread* s_12, uint thread* ob_0, uint thread* co_0, float3 thread* p_13, float3 thread* u_4, float thread* v_9, float thread* m_eff_2, float thread* t_in_3, KernelContext_0 thread* kernelContext_142)
{
    *t_in_3 = 0.0f;
    *k_33 = 4294967295U;
    *s_12 = 4294967295U;
    *ob_0 = 4294967295U;
    *co_0 = 4294967295U;
    float3 _S1005 = float3(0.0f) ;
    *p_13 = _S1005;
    *u_4 = _S1005;
    *v_9 = 0.0f;
    *m_eff_2 = 0.0f;
    uint _S1006 = cs_row_0(r_17, kernelContext_142);
    uint4 ids_2 = (as_type<uint4>((float4(*(kernelContext_142->engine_contacts_0+_S1006)) )));
    if((ids_2.w) == 0U)
    {
        return false;
    }
    bool _S1007 = side_0 == 0U;
    uint cm_0;
    if(_S1007)
    {
        cm_0 = ids_2.x;
    }
    else
    {
        cm_0 = ids_2.y;
    }
    uint _S1008;
    if(_S1007)
    {
        _S1008 = ids_2.y;
    }
    else
    {
        _S1008 = ids_2.x;
    }
    *co_0 = _S1008;
    uint _S1009 = (uint4(*(kernelContext_142->collider_info_0+2U * cm_0)) ).x;
    *k_33 = _S1009;
    bool _S1010;
    if(_S1009 == 4294967295U)
    {
        _S1010 = true;
    }
    else
    {
        _S1010 = (*k_33) >= (kernelContext_142->params_0->chunk_count_0);
    }
    if(_S1010)
    {
        _S1010 = true;
    }
    else
    {
        _S1010 = ((uint4(*(kernelContext_142->collider_info_0+2U * *co_0)) ).z) != 2U;
    }
    if(_S1010)
    {
        return false;
    }
    uint _S1011 = body_of_0(*k_33, kernelContext_142);
    *s_12 = _S1011;
    uint _S1012 = (as_type<uint>(((float4(*(kernelContext_142->colliders_0+4U * *co_0)) ).x)));
    *ob_0 = _S1012;
    if(_S1012 == (*s_12))
    {
        _S1010 = true;
    }
    else
    {
        _S1010 = (*ob_0) >= (kernelContext_142->params_0->max_bodies_0);
    }
    if(_S1010)
    {
        return false;
    }
    uint bs_0 = 7U * *s_12;
    uint bo_0 = 7U * *ob_0;
    float inv_o_0 = (float4(*(kernelContext_142->bodies_0+bo_0)) ).w;
    if(!(inv_o_0 > 0.0f))
    {
        return false;
    }
    float3 n_4 = (float4(*(kernelContext_142->engine_contacts_0+(_S1006 + 1U))) ).xyz;
    float3 _S1013;
    if(_S1007)
    {
        _S1013 = - n_4;
    }
    else
    {
        _S1013 = n_4;
    }
    *u_4 = _S1013;
    Quat_0 _S1014 = quat_of_0(float4(*(kernelContext_142->bodies_0+(bs_0 + 1U))) );
    if(_S1007)
    {
        _S1013 = (float4(*(kernelContext_142->engine_contacts_0+(_S1006 + 2U))) ).xyz;
    }
    else
    {
        _S1013 = (float4(*(kernelContext_142->engine_contacts_0+(_S1006 + 3U))) ).xyz;
    }
    thread Quat_0 _S1015 = _S1014;
    float3 _S1016 = rotate_0(&_S1015, _S1013);
    Quat_0 _S1017 = quat_of_0(float4(*(kernelContext_142->bodies_0+(bo_0 + 1U))) );
    if(_S1007)
    {
        _S1013 = (float4(*(kernelContext_142->engine_contacts_0+(_S1006 + 3U))) ).xyz;
    }
    else
    {
        _S1013 = (float4(*(kernelContext_142->engine_contacts_0+(_S1006 + 2U))) ).xyz;
    }
    thread Quat_0 _S1018 = _S1017;
    float3 _S1019 = rotate_0(&_S1018, _S1013);
    float3 _S1020 = (float4(*(kernelContext_142->bodies_0+bs_0)) ).xyz + _S1016;
    *p_13 = _S1020;
    float gap_0 = dot(_S1020 - ((float4(*(kernelContext_142->bodies_0+bo_0)) ).xyz + _S1019), *u_4);
    float _S1021 = dot((float4(*(kernelContext_142->bodies_0+(bo_0 + 3U))) ).xyz + cross((float4(*(kernelContext_142->bodies_0+(bo_0 + 4U))) ).xyz, _S1019) - ((float4(*(kernelContext_142->bodies_0+(bs_0 + 3U))) ).xyz + cross((float4(*(kernelContext_142->bodies_0+(bs_0 + 4U))) ).xyz, _S1016)), *u_4);
    *v_9 = _S1021;
    if(!(_S1021 > (kernelContext_142->params_0->strike_speed_0)))
    {
        return false;
    }
    float _S1022 = (float4(*(kernelContext_142->bodies_0+bs_0)) ).w;
    float _S1023 = cs_angular_0(*s_12, _S1016, *u_4, kernelContext_142);
    float _S1024 = _S1022 + _S1023 + inv_o_0;
    float _S1025 = cs_angular_0(*ob_0, _S1019, *u_4, kernelContext_142);
    float w_9 = _S1024 + _S1025;
    if(!(w_9 > 0.0f))
    {
        return false;
    }
    *m_eff_2 = 1.0f / w_9;
    *t_in_3 = clamp(gap_0 / *v_9, 0.0f, kernelContext_142->params_0->dt_0);
    return true;
}

[[kernel]] void fr_cs_contacts(uint3 id_51 [[thread_position_in_grid]], FractureParams_0 constant* params_54 [[buffer(0)]], uint device* work_54 [[buffer(4)]], packed_float4 device* frames_54 [[buffer(6)]], packed_float4 device* bodies_54 [[buffer(7)]], packed_float4 device* chunks_54 [[buffer(1)]], uint device* sort_54 [[buffer(11)]], atomic<uint> device* atom_54 [[buffer(5)]], Record_natural_0 device* manifolds_54 [[buffer(10)]], packed_float4 device* loads_54 [[buffer(12)]], packed_uint4 device* bonds_54 [[buffer(2)]], uint device* args_54 [[buffer(13)]], uint device* info_54 [[buffer(15)]], ScanParams_0 constant* scan_54 [[buffer(14)]], packed_float4 device* parts_54 [[buffer(3)]], packed_float4 device* colliders_54 [[buffer(8)]], packed_uint4 device* collider_info_54 [[buffer(9)]], packed_float4 device* view_chunks_54 [[buffer(17)]], ViewParams_0 constant* view_54 [[buffer(16)]], packed_float4 device* objects_54 [[buffer(18)]], uint device* shot_bodies_54 [[buffer(20)]], float device* lines_54 [[buffer(19)]], packed_float4 device* geometry_54 [[buffer(21)]], packed_float4 device* shots_54 [[buffer(22)]], packed_float4 device* lp_54 [[buffer(24)]], packed_float4 device* ext_56 [[buffer(23)]], uint device* engine_counters_54 [[buffer(26)]], packed_float4 device* engine_contacts_54 [[buffer(25)]], packed_float4 device* cs_anchors_54 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_143;
    (&kernelContext_143)->params_0 = params_54;
    (&kernelContext_143)->work_0 = work_54;
    (&kernelContext_143)->frames_0 = frames_54;
    (&kernelContext_143)->bodies_0 = bodies_54;
    (&kernelContext_143)->chunks_0 = chunks_54;
    (&kernelContext_143)->sort_0 = sort_54;
    (&kernelContext_143)->atom_0 = atom_54;
    (&kernelContext_143)->manifolds_0 = manifolds_54;
    (&kernelContext_143)->loads_0 = loads_54;
    (&kernelContext_143)->bonds_0 = bonds_54;
    (&kernelContext_143)->args_0 = args_54;
    (&kernelContext_143)->info_0 = info_54;
    (&kernelContext_143)->scan_0 = scan_54;
    (&kernelContext_143)->parts_0 = parts_54;
    (&kernelContext_143)->colliders_0 = colliders_54;
    (&kernelContext_143)->collider_info_0 = collider_info_54;
    (&kernelContext_143)->view_chunks_0 = view_chunks_54;
    (&kernelContext_143)->view_0 = view_54;
    (&kernelContext_143)->objects_0 = objects_54;
    (&kernelContext_143)->shot_bodies_0 = shot_bodies_54;
    (&kernelContext_143)->lines_0 = lines_54;
    (&kernelContext_143)->geometry_0 = geometry_54;
    (&kernelContext_143)->shots_0 = shots_54;
    (&kernelContext_143)->lp_0 = lp_54;
    (&kernelContext_143)->ext_0 = ext_56;
    (&kernelContext_143)->engine_counters_0 = engine_counters_54;
    (&kernelContext_143)->engine_contacts_0 = engine_contacts_54;
    (&kernelContext_143)->cs_anchors_0 = cs_anchors_54;
    threadgroup array<uint, int(1024)> g_scan_54;
    (&kernelContext_143)->g_scan_0 = &g_scan_54;
    uint r_18 = id_51.x;
    uint _S1026 = cs_rows_0(&kernelContext_143);
    if(r_18 >= _S1026)
    {
        return;
    }
    uint side_1 = 0U;
    for(;;)
    {
        if(side_1 < 2U)
        {
        }
        else
        {
            break;
        }
        thread uint k_34;
        thread uint s_13;
        thread uint ob_1;
        thread uint co_1;
        thread float3 p_14;
        thread float3 u_5;
        thread float v_10;
        thread float m_eff_3;
        thread float t_in_4;
        bool _S1027 = cs_side_0(r_18, side_1, &k_34, &s_13, &ob_1, &co_1, &p_14, &u_5, &v_10, &m_eff_3, &t_in_4, &kernelContext_143);
        if(_S1027)
        {
            uint _S1028 = a_cs_0(s_13, &kernelContext_143);
            uint _S1029 = atomic_fetch_max_explicit((&kernelContext_143)->atom_0+_S1028, (as_type<uint>((m_eff_3 * v_10))), memory_order_relaxed);
        }
        side_1 = side_1 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_pick(uint3 id_52 [[thread_position_in_grid]], FractureParams_0 constant* params_55 [[buffer(0)]], uint device* work_55 [[buffer(4)]], packed_float4 device* frames_55 [[buffer(6)]], packed_float4 device* bodies_55 [[buffer(7)]], packed_float4 device* chunks_55 [[buffer(1)]], uint device* sort_55 [[buffer(11)]], atomic<uint> device* atom_55 [[buffer(5)]], Record_natural_0 device* manifolds_55 [[buffer(10)]], packed_float4 device* loads_55 [[buffer(12)]], packed_uint4 device* bonds_55 [[buffer(2)]], uint device* args_55 [[buffer(13)]], uint device* info_55 [[buffer(15)]], ScanParams_0 constant* scan_55 [[buffer(14)]], packed_float4 device* parts_55 [[buffer(3)]], packed_float4 device* colliders_55 [[buffer(8)]], packed_uint4 device* collider_info_55 [[buffer(9)]], packed_float4 device* view_chunks_55 [[buffer(17)]], ViewParams_0 constant* view_55 [[buffer(16)]], packed_float4 device* objects_55 [[buffer(18)]], uint device* shot_bodies_55 [[buffer(20)]], float device* lines_55 [[buffer(19)]], packed_float4 device* geometry_55 [[buffer(21)]], packed_float4 device* shots_55 [[buffer(22)]], packed_float4 device* lp_55 [[buffer(24)]], packed_float4 device* ext_57 [[buffer(23)]], uint device* engine_counters_55 [[buffer(26)]], packed_float4 device* engine_contacts_55 [[buffer(25)]], packed_float4 device* cs_anchors_55 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_144;
    (&kernelContext_144)->params_0 = params_55;
    (&kernelContext_144)->work_0 = work_55;
    (&kernelContext_144)->frames_0 = frames_55;
    (&kernelContext_144)->bodies_0 = bodies_55;
    (&kernelContext_144)->chunks_0 = chunks_55;
    (&kernelContext_144)->sort_0 = sort_55;
    (&kernelContext_144)->atom_0 = atom_55;
    (&kernelContext_144)->manifolds_0 = manifolds_55;
    (&kernelContext_144)->loads_0 = loads_55;
    (&kernelContext_144)->bonds_0 = bonds_55;
    (&kernelContext_144)->args_0 = args_55;
    (&kernelContext_144)->info_0 = info_55;
    (&kernelContext_144)->scan_0 = scan_55;
    (&kernelContext_144)->parts_0 = parts_55;
    (&kernelContext_144)->colliders_0 = colliders_55;
    (&kernelContext_144)->collider_info_0 = collider_info_55;
    (&kernelContext_144)->view_chunks_0 = view_chunks_55;
    (&kernelContext_144)->view_0 = view_55;
    (&kernelContext_144)->objects_0 = objects_55;
    (&kernelContext_144)->shot_bodies_0 = shot_bodies_55;
    (&kernelContext_144)->lines_0 = lines_55;
    (&kernelContext_144)->geometry_0 = geometry_55;
    (&kernelContext_144)->shots_0 = shots_55;
    (&kernelContext_144)->lp_0 = lp_55;
    (&kernelContext_144)->ext_0 = ext_57;
    (&kernelContext_144)->engine_counters_0 = engine_counters_55;
    (&kernelContext_144)->engine_contacts_0 = engine_contacts_55;
    (&kernelContext_144)->cs_anchors_0 = cs_anchors_55;
    threadgroup array<uint, int(1024)> g_scan_55;
    (&kernelContext_144)->g_scan_0 = &g_scan_55;
    uint r_19 = id_52.x;
    uint _S1030 = cs_rows_0(&kernelContext_144);
    if(r_19 >= _S1030)
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
        thread uint k_35;
        thread uint s_14;
        thread uint ob_2;
        thread uint co_2;
        thread float3 p_15;
        thread float3 u_6;
        thread float v_11;
        thread float m_eff_4;
        thread float t_in_5;
        bool _S1031 = cs_side_0(r_19, side_2, &k_35, &s_14, &ob_2, &co_2, &p_15, &u_6, &v_11, &m_eff_4, &t_in_5, &kernelContext_144);
        bool _S1032;
        if(_S1031)
        {
            uint _S1033 = (as_type<uint>((m_eff_4 * v_11)));
            uint _S1034 = a_cs_0(s_14, &kernelContext_144);
            uint _S1035 = atomic_load_explicit((&kernelContext_144)->atom_0+_S1034, memory_order_relaxed);
            _S1032 = _S1033 == _S1035;
        }
        else
        {
            _S1032 = false;
        }
        if(_S1032)
        {
            uint _S1036 = a_cs_0(s_14, &kernelContext_144);
            uint _S1037 = atomic_fetch_min_explicit((&kernelContext_144)->atom_0+(_S1036 + 2U), 2U * r_19 + side_2, memory_order_relaxed);
        }
        side_2 = side_2 + 1U;
    }
    return;
}

uint w_cs_list_0(KernelContext_0 thread* kernelContext_145)
{
    uint _S1038 = w_carry_0(kernelContext_145);
    return _S1038 + kernelContext_145->params_0->chunk_count_0;
}

uint w_cs_bodies_0(KernelContext_0 thread* kernelContext_146)
{
    uint _S1039 = w_cs_list_0(kernelContext_146);
    return _S1039 + kernelContext_146->params_0->chunk_count_0;
}

[[kernel]] void fr_cs_record(uint3 id_53 [[thread_position_in_grid]], FractureParams_0 constant* params_56 [[buffer(0)]], uint device* work_56 [[buffer(4)]], packed_float4 device* frames_56 [[buffer(6)]], packed_float4 device* bodies_56 [[buffer(7)]], packed_float4 device* chunks_56 [[buffer(1)]], uint device* sort_56 [[buffer(11)]], atomic<uint> device* atom_56 [[buffer(5)]], Record_natural_0 device* manifolds_56 [[buffer(10)]], packed_float4 device* loads_56 [[buffer(12)]], packed_uint4 device* bonds_56 [[buffer(2)]], uint device* args_56 [[buffer(13)]], uint device* info_56 [[buffer(15)]], ScanParams_0 constant* scan_56 [[buffer(14)]], packed_float4 device* parts_56 [[buffer(3)]], packed_float4 device* colliders_56 [[buffer(8)]], packed_uint4 device* collider_info_56 [[buffer(9)]], packed_float4 device* view_chunks_56 [[buffer(17)]], ViewParams_0 constant* view_56 [[buffer(16)]], packed_float4 device* objects_56 [[buffer(18)]], uint device* shot_bodies_56 [[buffer(20)]], float device* lines_56 [[buffer(19)]], packed_float4 device* geometry_56 [[buffer(21)]], packed_float4 device* shots_56 [[buffer(22)]], packed_float4 device* lp_56 [[buffer(24)]], packed_float4 device* ext_58 [[buffer(23)]], uint device* engine_counters_56 [[buffer(26)]], packed_float4 device* engine_contacts_56 [[buffer(25)]], packed_float4 device* cs_anchors_56 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_147;
    (&kernelContext_147)->params_0 = params_56;
    (&kernelContext_147)->work_0 = work_56;
    (&kernelContext_147)->frames_0 = frames_56;
    (&kernelContext_147)->bodies_0 = bodies_56;
    (&kernelContext_147)->chunks_0 = chunks_56;
    (&kernelContext_147)->sort_0 = sort_56;
    (&kernelContext_147)->atom_0 = atom_56;
    (&kernelContext_147)->manifolds_0 = manifolds_56;
    (&kernelContext_147)->loads_0 = loads_56;
    (&kernelContext_147)->bonds_0 = bonds_56;
    (&kernelContext_147)->args_0 = args_56;
    (&kernelContext_147)->info_0 = info_56;
    (&kernelContext_147)->scan_0 = scan_56;
    (&kernelContext_147)->parts_0 = parts_56;
    (&kernelContext_147)->colliders_0 = colliders_56;
    (&kernelContext_147)->collider_info_0 = collider_info_56;
    (&kernelContext_147)->view_chunks_0 = view_chunks_56;
    (&kernelContext_147)->view_0 = view_56;
    (&kernelContext_147)->objects_0 = objects_56;
    (&kernelContext_147)->shot_bodies_0 = shot_bodies_56;
    (&kernelContext_147)->lines_0 = lines_56;
    (&kernelContext_147)->geometry_0 = geometry_56;
    (&kernelContext_147)->shots_0 = shots_56;
    (&kernelContext_147)->lp_0 = lp_56;
    (&kernelContext_147)->ext_0 = ext_58;
    (&kernelContext_147)->engine_counters_0 = engine_counters_56;
    (&kernelContext_147)->engine_contacts_0 = engine_contacts_56;
    (&kernelContext_147)->cs_anchors_0 = cs_anchors_56;
    threadgroup array<uint, int(1024)> g_scan_56;
    (&kernelContext_147)->g_scan_0 = &g_scan_56;
    uint s_15 = id_53.x;
    if(s_15 >= (params_56->body_count_0))
    {
        return;
    }
    uint _S1040 = a_cs_0(s_15, &kernelContext_147);
    uint key_2 = atomic_load_explicit((&kernelContext_147)->atom_0+(_S1040 + 2U), memory_order_relaxed);
    if(key_2 == 4294967295U)
    {
        return;
    }
    thread uint k_36;
    thread uint s2_0;
    thread uint ob_3;
    thread uint co_3;
    thread float3 p_16;
    thread float3 u_7;
    thread float v_12;
    thread float m_eff_5;
    thread float t_in_6;
    bool _S1041 = cs_side_0(key_2 / 2U, key_2 % 2U, &k_36, &s2_0, &ob_3, &co_3, &p_16, &u_7, &v_12, &m_eff_5, &t_in_6, &kernelContext_147);
    if(!_S1041)
    {
        return;
    }
    uint4 _S1042 = uint4(*((&kernelContext_147)->collider_info_0+(2U * co_3 + 1U))) ;
    float e_o_0 = (as_type<float>((_S1042.y)));
    float _S1043 = max((as_type<float>((_S1042.z))), 0.00100000004749745f);
    float j_7 = m_eff_5 * v_12;
    float tau_2 = hertz_duration_0(m_eff_5, _S1043, 1.0f / (1.0f / max(e_o_0, 1.0f) + 1.0f / (float4(*((&kernelContext_147)->geometry_0+(3U * k_36 + 1U))) ).w), v_12);
    uint _S1044 = lp_body_0(s_15, &kernelContext_147);
    *((&kernelContext_147)->lp_0+(_S1044 + 2U)) = packed_float4(float4(from_rubble_0(p_16), 3.14159274101257324f * j_7 / (2.0f * tau_2))) ;
    *((&kernelContext_147)->lp_0+(_S1044 + 3U)) = packed_float4(float4(from_rubble_0(u_7), v_12)) ;
    *((&kernelContext_147)->lp_0+(_S1044 + 4U)) = packed_float4(float4(m_eff_5, _S1043, e_o_0, (as_type<float>((ob_3))))) ;
    *((&kernelContext_147)->lp_0+(_S1044 + 5U)) = packed_float4(float4((as_type<float>((k_36))), j_7 / tau_2, 1.0f / (float4(*((&kernelContext_147)->bodies_0+7U * ob_3)) ).w, t_in_6)) ;
    atomic_store_explicit((&kernelContext_147)->atom_0+(_S1040 + 1U), k_36, memory_order_relaxed);
    uint _S1045 = a_counters_1(&kernelContext_147);
    uint _S1046 = atomic_fetch_add_explicit((&kernelContext_147)->atom_0+(_S1045 + 12U), 1U, memory_order_relaxed);
    uint at_16 = atomic_fetch_add_explicit((&kernelContext_147)->atom_0+(_S1045 + 14U), 1U, memory_order_relaxed);
    uint _S1047 = w_cs_bodies_0(&kernelContext_147);
    *((&kernelContext_147)->work_0+(_S1047 + at_16)) = s_15;
    return;
}

[[kernel]] void fr_cs_hertz(uint3 id_54 [[thread_position_in_grid]], FractureParams_0 constant* params_57 [[buffer(0)]], uint device* work_57 [[buffer(4)]], packed_float4 device* frames_57 [[buffer(6)]], packed_float4 device* bodies_57 [[buffer(7)]], packed_float4 device* chunks_57 [[buffer(1)]], uint device* sort_57 [[buffer(11)]], atomic<uint> device* atom_57 [[buffer(5)]], Record_natural_0 device* manifolds_57 [[buffer(10)]], packed_float4 device* loads_57 [[buffer(12)]], packed_uint4 device* bonds_57 [[buffer(2)]], uint device* args_57 [[buffer(13)]], uint device* info_57 [[buffer(15)]], ScanParams_0 constant* scan_57 [[buffer(14)]], packed_float4 device* parts_57 [[buffer(3)]], packed_float4 device* colliders_57 [[buffer(8)]], packed_uint4 device* collider_info_57 [[buffer(9)]], packed_float4 device* view_chunks_57 [[buffer(17)]], ViewParams_0 constant* view_57 [[buffer(16)]], packed_float4 device* objects_57 [[buffer(18)]], uint device* shot_bodies_57 [[buffer(20)]], float device* lines_57 [[buffer(19)]], packed_float4 device* geometry_57 [[buffer(21)]], packed_float4 device* shots_57 [[buffer(22)]], packed_float4 device* lp_57 [[buffer(24)]], packed_float4 device* ext_59 [[buffer(23)]], uint device* engine_counters_57 [[buffer(26)]], packed_float4 device* engine_contacts_57 [[buffer(25)]], packed_float4 device* cs_anchors_57 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_148;
    (&kernelContext_148)->params_0 = params_57;
    (&kernelContext_148)->work_0 = work_57;
    (&kernelContext_148)->frames_0 = frames_57;
    (&kernelContext_148)->bodies_0 = bodies_57;
    (&kernelContext_148)->chunks_0 = chunks_57;
    (&kernelContext_148)->sort_0 = sort_57;
    (&kernelContext_148)->atom_0 = atom_57;
    (&kernelContext_148)->manifolds_0 = manifolds_57;
    (&kernelContext_148)->loads_0 = loads_57;
    (&kernelContext_148)->bonds_0 = bonds_57;
    (&kernelContext_148)->args_0 = args_57;
    (&kernelContext_148)->info_0 = info_57;
    (&kernelContext_148)->scan_0 = scan_57;
    (&kernelContext_148)->parts_0 = parts_57;
    (&kernelContext_148)->colliders_0 = colliders_57;
    (&kernelContext_148)->collider_info_0 = collider_info_57;
    (&kernelContext_148)->view_chunks_0 = view_chunks_57;
    (&kernelContext_148)->view_0 = view_57;
    (&kernelContext_148)->objects_0 = objects_57;
    (&kernelContext_148)->shot_bodies_0 = shot_bodies_57;
    (&kernelContext_148)->lines_0 = lines_57;
    (&kernelContext_148)->geometry_0 = geometry_57;
    (&kernelContext_148)->shots_0 = shots_57;
    (&kernelContext_148)->lp_0 = lp_57;
    (&kernelContext_148)->ext_0 = ext_59;
    (&kernelContext_148)->engine_counters_0 = engine_counters_57;
    (&kernelContext_148)->engine_contacts_0 = engine_contacts_57;
    (&kernelContext_148)->cs_anchors_0 = cs_anchors_57;
    threadgroup array<uint, int(1024)> g_scan_57;
    (&kernelContext_148)->g_scan_0 = &g_scan_57;
    uint i_11 = id_54.x;
    bool _S1048;
    if(i_11 >= (params_57->bond_count_0))
    {
        _S1048 = true;
    }
    else
    {
        uint _S1049 = w_alive_0(&kernelContext_148);
        _S1048 = (*((&kernelContext_148)->work_0+(_S1049 + i_11))) == 0U;
    }
    if(_S1048)
    {
        return;
    }
    uint4 _S1050 = uint4(*((&kernelContext_148)->bonds_0+i_11)) ;
    uint _S1051 = body_of_0(_S1050.x, &kernelContext_148);
    bool _S1052 = cs_struck_1(_S1051, &kernelContext_148);
    if(!_S1052)
    {
        return;
    }
    uint _S1053 = lp_body_0(_S1051, &kernelContext_148);
    bool _S1054 = strike_share_breaks_0(_S1050, (float4(*((&kernelContext_148)->lp_0+(_S1053 + 2U))) ).xyz, (float4(*((&kernelContext_148)->lp_0+(_S1053 + 5U))) ).y, &kernelContext_148);
    if(_S1054)
    {
        uint _S1055 = w_alive_0(&kernelContext_148);
        *((&kernelContext_148)->work_0+(_S1055 + i_11)) = 0U;
        uint _S1056 = a_dirty_0(&kernelContext_148);
        uint _S1057 = atomic_fetch_or_explicit((&kernelContext_148)->atom_0+(_S1056 + _S1051), 1U, memory_order_relaxed);
        uint _S1058 = a_counters_1(&kernelContext_148);
        uint _S1059 = atomic_fetch_add_explicit((&kernelContext_148)->atom_0+_S1058, 1U, memory_order_relaxed);
        uint _S1060 = atomic_fetch_add_explicit((&kernelContext_148)->atom_0+(_S1058 + 1U), 1U, memory_order_relaxed);
        uint _S1061 = atomic_fetch_add_explicit((&kernelContext_148)->atom_0+(_S1058 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_save(uint3 id_55 [[thread_position_in_grid]], FractureParams_0 constant* params_58 [[buffer(0)]], uint device* work_58 [[buffer(4)]], packed_float4 device* frames_58 [[buffer(6)]], packed_float4 device* bodies_58 [[buffer(7)]], packed_float4 device* chunks_58 [[buffer(1)]], uint device* sort_58 [[buffer(11)]], atomic<uint> device* atom_58 [[buffer(5)]], Record_natural_0 device* manifolds_58 [[buffer(10)]], packed_float4 device* loads_58 [[buffer(12)]], packed_uint4 device* bonds_58 [[buffer(2)]], uint device* args_58 [[buffer(13)]], uint device* info_58 [[buffer(15)]], ScanParams_0 constant* scan_58 [[buffer(14)]], packed_float4 device* parts_58 [[buffer(3)]], packed_float4 device* colliders_58 [[buffer(8)]], packed_uint4 device* collider_info_58 [[buffer(9)]], packed_float4 device* view_chunks_58 [[buffer(17)]], ViewParams_0 constant* view_58 [[buffer(16)]], packed_float4 device* objects_58 [[buffer(18)]], uint device* shot_bodies_58 [[buffer(20)]], float device* lines_58 [[buffer(19)]], packed_float4 device* geometry_58 [[buffer(21)]], packed_float4 device* shots_58 [[buffer(22)]], packed_float4 device* lp_58 [[buffer(24)]], packed_float4 device* ext_60 [[buffer(23)]], uint device* engine_counters_58 [[buffer(26)]], packed_float4 device* engine_contacts_58 [[buffer(25)]], packed_float4 device* cs_anchors_58 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_149;
    (&kernelContext_149)->params_0 = params_58;
    (&kernelContext_149)->work_0 = work_58;
    (&kernelContext_149)->frames_0 = frames_58;
    (&kernelContext_149)->bodies_0 = bodies_58;
    (&kernelContext_149)->chunks_0 = chunks_58;
    (&kernelContext_149)->sort_0 = sort_58;
    (&kernelContext_149)->atom_0 = atom_58;
    (&kernelContext_149)->manifolds_0 = manifolds_58;
    (&kernelContext_149)->loads_0 = loads_58;
    (&kernelContext_149)->bonds_0 = bonds_58;
    (&kernelContext_149)->args_0 = args_58;
    (&kernelContext_149)->info_0 = info_58;
    (&kernelContext_149)->scan_0 = scan_58;
    (&kernelContext_149)->parts_0 = parts_58;
    (&kernelContext_149)->colliders_0 = colliders_58;
    (&kernelContext_149)->collider_info_0 = collider_info_58;
    (&kernelContext_149)->view_chunks_0 = view_chunks_58;
    (&kernelContext_149)->view_0 = view_58;
    (&kernelContext_149)->objects_0 = objects_58;
    (&kernelContext_149)->shot_bodies_0 = shot_bodies_58;
    (&kernelContext_149)->lines_0 = lines_58;
    (&kernelContext_149)->geometry_0 = geometry_58;
    (&kernelContext_149)->shots_0 = shots_58;
    (&kernelContext_149)->lp_0 = lp_58;
    (&kernelContext_149)->ext_0 = ext_60;
    (&kernelContext_149)->engine_counters_0 = engine_counters_58;
    (&kernelContext_149)->engine_contacts_0 = engine_contacts_58;
    (&kernelContext_149)->cs_anchors_0 = cs_anchors_58;
    threadgroup array<uint, int(1024)> g_scan_58;
    (&kernelContext_149)->g_scan_0 = &g_scan_58;
    uint r_20 = id_55.x;
    uint _S1062 = cs_rows_0(&kernelContext_149);
    if(r_20 >= _S1062)
    {
        return;
    }
    uint _S1063 = cs_row_0(r_20, &kernelContext_149);
    uint4 _S1064 = (as_type<uint4>((float4(*((&kernelContext_149)->engine_contacts_0+_S1063)) )));
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
        float4 _S1065 = float4(0.0f) ;
        bool _S1066 = side_3 == 0U;
        uint col_1;
        if(_S1066)
        {
            col_1 = _S1064.x;
        }
        else
        {
            col_1 = _S1064.y;
        }
        uint k_37;
        if((_S1064.w) != 0U)
        {
            k_37 = (uint4(*((&kernelContext_149)->collider_info_0+2U * col_1)) ).x;
        }
        else
        {
            k_37 = 4294967295U;
        }
        bool _S1067;
        if(k_37 != 4294967295U)
        {
            _S1067 = k_37 < ((&kernelContext_149)->params_0->chunk_count_0);
        }
        else
        {
            _S1067 = false;
        }
        bool _S1068;
        if(_S1067)
        {
            uint _S1069 = body_of_0(k_37, &kernelContext_149);
            bool _S1070 = dirty_0(_S1069, &kernelContext_149);
            _S1068 = _S1070;
        }
        else
        {
            _S1068 = false;
        }
        float4 saved_0;
        if(_S1068)
        {
            uint _S1071 = body_of_0(k_37, &kernelContext_149);
            uint at_17 = 7U * _S1071;
            float3 anchor_0;
            if(_S1066)
            {
                anchor_0 = (float4(*((&kernelContext_149)->engine_contacts_0+(_S1063 + 2U))) ).xyz;
            }
            else
            {
                anchor_0 = (float4(*((&kernelContext_149)->engine_contacts_0+(_S1063 + 3U))) ).xyz;
            }
            float3 _S1072 = (float4(*((&kernelContext_149)->bodies_0+at_17)) ).xyz;
            thread Quat_0 _S1073 = quat_of_0(float4(*((&kernelContext_149)->bodies_0+(at_17 + 1U))) );
            float3 _S1074 = rotate_0(&_S1073, anchor_0);
            saved_0 = float4(_S1072 + _S1074, 1.0f);
        }
        else
        {
            saved_0 = _S1065;
        }
        *((&kernelContext_149)->cs_anchors_0+(2U * r_20 + side_3)) = packed_float4(saved_0) ;
        side_3 = side_3 + 1U;
    }
    return;
}

[[kernel]] void fr_cs_fix(uint3 id_56 [[thread_position_in_grid]], FractureParams_0 constant* params_59 [[buffer(0)]], uint device* work_59 [[buffer(4)]], packed_float4 device* frames_59 [[buffer(6)]], packed_float4 device* bodies_59 [[buffer(7)]], packed_float4 device* chunks_59 [[buffer(1)]], uint device* sort_59 [[buffer(11)]], atomic<uint> device* atom_59 [[buffer(5)]], Record_natural_0 device* manifolds_59 [[buffer(10)]], packed_float4 device* loads_59 [[buffer(12)]], packed_uint4 device* bonds_59 [[buffer(2)]], uint device* args_59 [[buffer(13)]], uint device* info_59 [[buffer(15)]], ScanParams_0 constant* scan_59 [[buffer(14)]], packed_float4 device* parts_59 [[buffer(3)]], packed_float4 device* colliders_59 [[buffer(8)]], packed_uint4 device* collider_info_59 [[buffer(9)]], packed_float4 device* view_chunks_59 [[buffer(17)]], ViewParams_0 constant* view_59 [[buffer(16)]], packed_float4 device* objects_59 [[buffer(18)]], uint device* shot_bodies_59 [[buffer(20)]], float device* lines_59 [[buffer(19)]], packed_float4 device* geometry_59 [[buffer(21)]], packed_float4 device* shots_59 [[buffer(22)]], packed_float4 device* lp_59 [[buffer(24)]], packed_float4 device* ext_61 [[buffer(23)]], uint device* engine_counters_59 [[buffer(26)]], packed_float4 device* engine_contacts_59 [[buffer(25)]], packed_float4 device* cs_anchors_59 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_150;
    (&kernelContext_150)->params_0 = params_59;
    (&kernelContext_150)->work_0 = work_59;
    (&kernelContext_150)->frames_0 = frames_59;
    (&kernelContext_150)->bodies_0 = bodies_59;
    (&kernelContext_150)->chunks_0 = chunks_59;
    (&kernelContext_150)->sort_0 = sort_59;
    (&kernelContext_150)->atom_0 = atom_59;
    (&kernelContext_150)->manifolds_0 = manifolds_59;
    (&kernelContext_150)->loads_0 = loads_59;
    (&kernelContext_150)->bonds_0 = bonds_59;
    (&kernelContext_150)->args_0 = args_59;
    (&kernelContext_150)->info_0 = info_59;
    (&kernelContext_150)->scan_0 = scan_59;
    (&kernelContext_150)->parts_0 = parts_59;
    (&kernelContext_150)->colliders_0 = colliders_59;
    (&kernelContext_150)->collider_info_0 = collider_info_59;
    (&kernelContext_150)->view_chunks_0 = view_chunks_59;
    (&kernelContext_150)->view_0 = view_59;
    (&kernelContext_150)->objects_0 = objects_59;
    (&kernelContext_150)->shot_bodies_0 = shot_bodies_59;
    (&kernelContext_150)->lines_0 = lines_59;
    (&kernelContext_150)->geometry_0 = geometry_59;
    (&kernelContext_150)->shots_0 = shots_59;
    (&kernelContext_150)->lp_0 = lp_59;
    (&kernelContext_150)->ext_0 = ext_61;
    (&kernelContext_150)->engine_counters_0 = engine_counters_59;
    (&kernelContext_150)->engine_contacts_0 = engine_contacts_59;
    (&kernelContext_150)->cs_anchors_0 = cs_anchors_59;
    threadgroup array<uint, int(1024)> g_scan_59;
    (&kernelContext_150)->g_scan_0 = &g_scan_59;
    uint r_21 = id_56.x;
    uint _S1075 = cs_rows_0(&kernelContext_150);
    if(r_21 >= _S1075)
    {
        return;
    }
    uint _S1076 = cs_row_0(r_21, &kernelContext_150);
    uint4 _S1077 = (as_type<uint4>((float4(*((&kernelContext_150)->engine_contacts_0+_S1076)) )));
    bool moved_0 = false;
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
        float4 _S1078 = float4(*((&kernelContext_150)->cs_anchors_0+(2U * r_21 + side_4))) ;
        if((_S1078.w) == 0.0f)
        {
            side_4 = side_4 + 1U;
            continue;
        }
        uint col_2;
        if(side_4 == 0U)
        {
            col_2 = _S1077.x;
        }
        else
        {
            col_2 = _S1077.y;
        }
        uint at_18 = 7U * (as_type<uint>(((float4(*((&kernelContext_150)->colliders_0+4U * col_2)) ).x)));
        thread Quat_0 _S1079 = quat_of_0(float4(*((&kernelContext_150)->bodies_0+(at_18 + 1U))) );
        Quat_0 _S1080 = conj_0(&_S1079);
        float3 _S1081 = _S1078.xyz - (float4(*((&kernelContext_150)->bodies_0+at_18)) ).xyz;
        thread Quat_0 _S1082 = _S1080;
        float3 _S1083 = rotate_0(&_S1082, _S1081);
        (*((&kernelContext_150)->engine_contacts_0+(_S1076 + 2U + side_4))).xyz = _S1083;
        moved_0 = true;
        side_4 = side_4 + 1U;
    }
    if(moved_0)
    {
        (*((&kernelContext_150)->engine_contacts_0+(_S1076 + 4U)))[int(3)] = 0.0f;
        (*((&kernelContext_150)->engine_contacts_0+(_S1076 + 5U)))[int(3)] = 0.0f;
        uint _S1084 = _S1076 + 6U;
        *((&kernelContext_150)->engine_contacts_0+_S1084) = packed_float4(float4(1.0f, 1.0f, 0.0f, (float4(*((&kernelContext_150)->engine_contacts_0+_S1084)) ).w)) ;
        (*((&kernelContext_150)->engine_contacts_0+(_S1076 + 7U)))[int(1)] = 0.0f;
    }
    return;
}

[[kernel]] void fr_cs_count(uint3 id_57 [[thread_position_in_grid]], FractureParams_0 constant* params_60 [[buffer(0)]], uint device* work_60 [[buffer(4)]], packed_float4 device* frames_60 [[buffer(6)]], packed_float4 device* bodies_60 [[buffer(7)]], packed_float4 device* chunks_60 [[buffer(1)]], uint device* sort_60 [[buffer(11)]], atomic<uint> device* atom_60 [[buffer(5)]], Record_natural_0 device* manifolds_60 [[buffer(10)]], packed_float4 device* loads_60 [[buffer(12)]], packed_uint4 device* bonds_60 [[buffer(2)]], uint device* args_60 [[buffer(13)]], uint device* info_60 [[buffer(15)]], ScanParams_0 constant* scan_60 [[buffer(14)]], packed_float4 device* parts_60 [[buffer(3)]], packed_float4 device* colliders_60 [[buffer(8)]], packed_uint4 device* collider_info_60 [[buffer(9)]], packed_float4 device* view_chunks_60 [[buffer(17)]], ViewParams_0 constant* view_60 [[buffer(16)]], packed_float4 device* objects_60 [[buffer(18)]], uint device* shot_bodies_60 [[buffer(20)]], float device* lines_60 [[buffer(19)]], packed_float4 device* geometry_60 [[buffer(21)]], packed_float4 device* shots_60 [[buffer(22)]], packed_float4 device* lp_60 [[buffer(24)]], packed_float4 device* ext_62 [[buffer(23)]], uint device* engine_counters_60 [[buffer(26)]], packed_float4 device* engine_contacts_60 [[buffer(25)]], packed_float4 device* cs_anchors_60 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_151;
    (&kernelContext_151)->params_0 = params_60;
    (&kernelContext_151)->work_0 = work_60;
    (&kernelContext_151)->frames_0 = frames_60;
    (&kernelContext_151)->bodies_0 = bodies_60;
    (&kernelContext_151)->chunks_0 = chunks_60;
    (&kernelContext_151)->sort_0 = sort_60;
    (&kernelContext_151)->atom_0 = atom_60;
    (&kernelContext_151)->manifolds_0 = manifolds_60;
    (&kernelContext_151)->loads_0 = loads_60;
    (&kernelContext_151)->bonds_0 = bonds_60;
    (&kernelContext_151)->args_0 = args_60;
    (&kernelContext_151)->info_0 = info_60;
    (&kernelContext_151)->scan_0 = scan_60;
    (&kernelContext_151)->parts_0 = parts_60;
    (&kernelContext_151)->colliders_0 = colliders_60;
    (&kernelContext_151)->collider_info_0 = collider_info_60;
    (&kernelContext_151)->view_chunks_0 = view_chunks_60;
    (&kernelContext_151)->view_0 = view_60;
    (&kernelContext_151)->objects_0 = objects_60;
    (&kernelContext_151)->shot_bodies_0 = shot_bodies_60;
    (&kernelContext_151)->lines_0 = lines_60;
    (&kernelContext_151)->geometry_0 = geometry_60;
    (&kernelContext_151)->shots_0 = shots_60;
    (&kernelContext_151)->lp_0 = lp_60;
    (&kernelContext_151)->ext_0 = ext_62;
    (&kernelContext_151)->engine_counters_0 = engine_counters_60;
    (&kernelContext_151)->engine_contacts_0 = engine_contacts_60;
    (&kernelContext_151)->cs_anchors_0 = cs_anchors_60;
    threadgroup array<uint, int(1024)> g_scan_60;
    (&kernelContext_151)->g_scan_0 = &g_scan_60;
    uint k_38 = id_57.x;
    if(k_38 >= (params_60->chunk_count_0))
    {
        return;
    }
    uint _S1085 = body_of_0(k_38, &kernelContext_151);
    bool _S1086 = cs_struck_1(_S1085, &kernelContext_151);
    if(!_S1086)
    {
        return;
    }
    uint _S1087 = a_cs_0(_S1085, &kernelContext_151);
    uint _S1088 = atomic_fetch_min_explicit((&kernelContext_151)->atom_0+(_S1087 + 3U), k_38, memory_order_relaxed);
    if(((&kernelContext_151)->info_0[k_38] & 1U) != 0U)
    {
        atomic_store_explicit((&kernelContext_151)->atom_0+(_S1087 + 4U), 1U, memory_order_relaxed);
    }
    uint _S1089 = atomic_fetch_add_explicit((&kernelContext_151)->atom_0+(_S1087 + 5U), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_base(uint3 id_58 [[thread_position_in_grid]], FractureParams_0 constant* params_61 [[buffer(0)]], uint device* work_61 [[buffer(4)]], packed_float4 device* frames_61 [[buffer(6)]], packed_float4 device* bodies_61 [[buffer(7)]], packed_float4 device* chunks_61 [[buffer(1)]], uint device* sort_61 [[buffer(11)]], atomic<uint> device* atom_61 [[buffer(5)]], Record_natural_0 device* manifolds_61 [[buffer(10)]], packed_float4 device* loads_61 [[buffer(12)]], packed_uint4 device* bonds_61 [[buffer(2)]], uint device* args_61 [[buffer(13)]], uint device* info_61 [[buffer(15)]], ScanParams_0 constant* scan_61 [[buffer(14)]], packed_float4 device* parts_61 [[buffer(3)]], packed_float4 device* colliders_61 [[buffer(8)]], packed_uint4 device* collider_info_61 [[buffer(9)]], packed_float4 device* view_chunks_61 [[buffer(17)]], ViewParams_0 constant* view_61 [[buffer(16)]], packed_float4 device* objects_61 [[buffer(18)]], uint device* shot_bodies_61 [[buffer(20)]], float device* lines_61 [[buffer(19)]], packed_float4 device* geometry_61 [[buffer(21)]], packed_float4 device* shots_61 [[buffer(22)]], packed_float4 device* lp_61 [[buffer(24)]], packed_float4 device* ext_63 [[buffer(23)]], uint device* engine_counters_61 [[buffer(26)]], packed_float4 device* engine_contacts_61 [[buffer(25)]], packed_float4 device* cs_anchors_61 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_152;
    (&kernelContext_152)->params_0 = params_61;
    (&kernelContext_152)->work_0 = work_61;
    (&kernelContext_152)->frames_0 = frames_61;
    (&kernelContext_152)->bodies_0 = bodies_61;
    (&kernelContext_152)->chunks_0 = chunks_61;
    (&kernelContext_152)->sort_0 = sort_61;
    (&kernelContext_152)->atom_0 = atom_61;
    (&kernelContext_152)->manifolds_0 = manifolds_61;
    (&kernelContext_152)->loads_0 = loads_61;
    (&kernelContext_152)->bonds_0 = bonds_61;
    (&kernelContext_152)->args_0 = args_61;
    (&kernelContext_152)->info_0 = info_61;
    (&kernelContext_152)->scan_0 = scan_61;
    (&kernelContext_152)->parts_0 = parts_61;
    (&kernelContext_152)->colliders_0 = colliders_61;
    (&kernelContext_152)->collider_info_0 = collider_info_61;
    (&kernelContext_152)->view_chunks_0 = view_chunks_61;
    (&kernelContext_152)->view_0 = view_61;
    (&kernelContext_152)->objects_0 = objects_61;
    (&kernelContext_152)->shot_bodies_0 = shot_bodies_61;
    (&kernelContext_152)->lines_0 = lines_61;
    (&kernelContext_152)->geometry_0 = geometry_61;
    (&kernelContext_152)->shots_0 = shots_61;
    (&kernelContext_152)->lp_0 = lp_61;
    (&kernelContext_152)->ext_0 = ext_63;
    (&kernelContext_152)->engine_counters_0 = engine_counters_61;
    (&kernelContext_152)->engine_contacts_0 = engine_contacts_61;
    (&kernelContext_152)->cs_anchors_0 = cs_anchors_61;
    threadgroup array<uint, int(1024)> g_scan_61;
    (&kernelContext_152)->g_scan_0 = &g_scan_61;
    uint s_16 = id_58.x;
    bool _S1090 = cs_struck_1(s_16, &kernelContext_152);
    if(!_S1090)
    {
        return;
    }
    uint _S1091 = a_cs_0(s_16, &kernelContext_152);
    atomic<uint> device* _S1092 = (&kernelContext_152)->atom_0+(_S1091 + 6U);
    uint _S1093 = a_counters_1(&kernelContext_152);
    atomic<uint> device* _S1094 = (&kernelContext_152)->atom_0+(_S1093 + 15U);
    uint _S1095 = atomic_load_explicit((&kernelContext_152)->atom_0+(_S1091 + 5U), memory_order_relaxed);
    uint _S1096 = atomic_fetch_add_explicit(_S1094, _S1095, memory_order_relaxed);
    atomic_store_explicit(_S1092, _S1096, memory_order_relaxed);
    return;
}

[[kernel]] void fr_cs_collect(uint3 id_59 [[thread_position_in_grid]], FractureParams_0 constant* params_62 [[buffer(0)]], uint device* work_62 [[buffer(4)]], packed_float4 device* frames_62 [[buffer(6)]], packed_float4 device* bodies_62 [[buffer(7)]], packed_float4 device* chunks_62 [[buffer(1)]], uint device* sort_62 [[buffer(11)]], atomic<uint> device* atom_62 [[buffer(5)]], Record_natural_0 device* manifolds_62 [[buffer(10)]], packed_float4 device* loads_62 [[buffer(12)]], packed_uint4 device* bonds_62 [[buffer(2)]], uint device* args_62 [[buffer(13)]], uint device* info_62 [[buffer(15)]], ScanParams_0 constant* scan_62 [[buffer(14)]], packed_float4 device* parts_62 [[buffer(3)]], packed_float4 device* colliders_62 [[buffer(8)]], packed_uint4 device* collider_info_62 [[buffer(9)]], packed_float4 device* view_chunks_62 [[buffer(17)]], ViewParams_0 constant* view_62 [[buffer(16)]], packed_float4 device* objects_62 [[buffer(18)]], uint device* shot_bodies_62 [[buffer(20)]], float device* lines_62 [[buffer(19)]], packed_float4 device* geometry_62 [[buffer(21)]], packed_float4 device* shots_62 [[buffer(22)]], packed_float4 device* lp_62 [[buffer(24)]], packed_float4 device* ext_64 [[buffer(23)]], uint device* engine_counters_62 [[buffer(26)]], packed_float4 device* engine_contacts_62 [[buffer(25)]], packed_float4 device* cs_anchors_62 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_153;
    (&kernelContext_153)->params_0 = params_62;
    (&kernelContext_153)->work_0 = work_62;
    (&kernelContext_153)->frames_0 = frames_62;
    (&kernelContext_153)->bodies_0 = bodies_62;
    (&kernelContext_153)->chunks_0 = chunks_62;
    (&kernelContext_153)->sort_0 = sort_62;
    (&kernelContext_153)->atom_0 = atom_62;
    (&kernelContext_153)->manifolds_0 = manifolds_62;
    (&kernelContext_153)->loads_0 = loads_62;
    (&kernelContext_153)->bonds_0 = bonds_62;
    (&kernelContext_153)->args_0 = args_62;
    (&kernelContext_153)->info_0 = info_62;
    (&kernelContext_153)->scan_0 = scan_62;
    (&kernelContext_153)->parts_0 = parts_62;
    (&kernelContext_153)->colliders_0 = colliders_62;
    (&kernelContext_153)->collider_info_0 = collider_info_62;
    (&kernelContext_153)->view_chunks_0 = view_chunks_62;
    (&kernelContext_153)->view_0 = view_62;
    (&kernelContext_153)->objects_0 = objects_62;
    (&kernelContext_153)->shot_bodies_0 = shot_bodies_62;
    (&kernelContext_153)->lines_0 = lines_62;
    (&kernelContext_153)->geometry_0 = geometry_62;
    (&kernelContext_153)->shots_0 = shots_62;
    (&kernelContext_153)->lp_0 = lp_62;
    (&kernelContext_153)->ext_0 = ext_64;
    (&kernelContext_153)->engine_counters_0 = engine_counters_62;
    (&kernelContext_153)->engine_contacts_0 = engine_contacts_62;
    (&kernelContext_153)->cs_anchors_0 = cs_anchors_62;
    threadgroup array<uint, int(1024)> g_scan_62;
    (&kernelContext_153)->g_scan_0 = &g_scan_62;
    uint k_39 = id_59.x;
    if(k_39 >= (params_62->chunk_count_0))
    {
        return;
    }
    uint _S1097 = body_of_0(k_39, &kernelContext_153);
    bool _S1098 = cs_struck_1(_S1097, &kernelContext_153);
    if(!_S1098)
    {
        return;
    }
    uint _S1099 = w_cs_list_0(&kernelContext_153);
    uint _S1100 = a_cs_0(_S1097, &kernelContext_153);
    uint _S1101 = atomic_load_explicit((&kernelContext_153)->atom_0+(_S1100 + 6U), memory_order_relaxed);
    uint _S1102 = _S1099 + _S1101;
    uint _S1103 = atomic_fetch_add_explicit((&kernelContext_153)->atom_0+(_S1100 + 7U), 1U, memory_order_relaxed);
    *((&kernelContext_153)->work_0+(_S1102 + _S1103)) = k_39;
    return;
}

[[kernel]] void fr_cs_gate(uint3 id_60 [[thread_position_in_grid]], FractureParams_0 constant* params_63 [[buffer(0)]], uint device* work_63 [[buffer(4)]], packed_float4 device* frames_63 [[buffer(6)]], packed_float4 device* bodies_63 [[buffer(7)]], packed_float4 device* chunks_63 [[buffer(1)]], uint device* sort_63 [[buffer(11)]], atomic<uint> device* atom_63 [[buffer(5)]], Record_natural_0 device* manifolds_63 [[buffer(10)]], packed_float4 device* loads_63 [[buffer(12)]], packed_uint4 device* bonds_63 [[buffer(2)]], uint device* args_63 [[buffer(13)]], uint device* info_63 [[buffer(15)]], ScanParams_0 constant* scan_63 [[buffer(14)]], packed_float4 device* parts_63 [[buffer(3)]], packed_float4 device* colliders_63 [[buffer(8)]], packed_uint4 device* collider_info_63 [[buffer(9)]], packed_float4 device* view_chunks_63 [[buffer(17)]], ViewParams_0 constant* view_63 [[buffer(16)]], packed_float4 device* objects_63 [[buffer(18)]], uint device* shot_bodies_63 [[buffer(20)]], float device* lines_63 [[buffer(19)]], packed_float4 device* geometry_63 [[buffer(21)]], packed_float4 device* shots_63 [[buffer(22)]], packed_float4 device* lp_63 [[buffer(24)]], packed_float4 device* ext_65 [[buffer(23)]], uint device* engine_counters_63 [[buffer(26)]], packed_float4 device* engine_contacts_63 [[buffer(25)]], packed_float4 device* cs_anchors_63 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_154;
    (&kernelContext_154)->params_0 = params_63;
    (&kernelContext_154)->work_0 = work_63;
    (&kernelContext_154)->frames_0 = frames_63;
    (&kernelContext_154)->bodies_0 = bodies_63;
    (&kernelContext_154)->chunks_0 = chunks_63;
    (&kernelContext_154)->sort_0 = sort_63;
    (&kernelContext_154)->atom_0 = atom_63;
    (&kernelContext_154)->manifolds_0 = manifolds_63;
    (&kernelContext_154)->loads_0 = loads_63;
    (&kernelContext_154)->bonds_0 = bonds_63;
    (&kernelContext_154)->args_0 = args_63;
    (&kernelContext_154)->info_0 = info_63;
    (&kernelContext_154)->scan_0 = scan_63;
    (&kernelContext_154)->parts_0 = parts_63;
    (&kernelContext_154)->colliders_0 = colliders_63;
    (&kernelContext_154)->collider_info_0 = collider_info_63;
    (&kernelContext_154)->view_chunks_0 = view_chunks_63;
    (&kernelContext_154)->view_0 = view_63;
    (&kernelContext_154)->objects_0 = objects_63;
    (&kernelContext_154)->shot_bodies_0 = shot_bodies_63;
    (&kernelContext_154)->lines_0 = lines_63;
    (&kernelContext_154)->geometry_0 = geometry_63;
    (&kernelContext_154)->shots_0 = shots_63;
    (&kernelContext_154)->lp_0 = lp_63;
    (&kernelContext_154)->ext_0 = ext_65;
    (&kernelContext_154)->engine_counters_0 = engine_counters_63;
    (&kernelContext_154)->engine_contacts_0 = engine_contacts_63;
    (&kernelContext_154)->cs_anchors_0 = cs_anchors_63;
    threadgroup array<uint, int(1024)> g_scan_63;
    (&kernelContext_154)->g_scan_0 = &g_scan_63;
    uint _S1104 = a_counters_1(&kernelContext_154);
    uint _S1105 = atomic_load_explicit((&kernelContext_154)->atom_0+(_S1104 + 14U), memory_order_relaxed);
    set_args_0(6U, _S1105, &kernelContext_154);
    return;
}

void cs_load_0(uint k_40, uint s_17, KernelContext_0 thread* kernelContext_155)
{
    uint _S1106 = a_cs_0(s_17, kernelContext_155);
    uint _S1107 = atomic_load_explicit(kernelContext_155->atom_0+(_S1106 + 4U), memory_order_relaxed);
    bool supported_1 = _S1107 != 0U;
    bool fixed_2;
    if((kernelContext_155->info_0[k_40] & 1U) != 0U)
    {
        fixed_2 = true;
    }
    else
    {
        if(!supported_1)
        {
            uint _S1108 = atomic_load_explicit(kernelContext_155->atom_0+(_S1106 + 3U), memory_order_relaxed);
            fixed_2 = _S1108 == k_40;
        }
        else
        {
            fixed_2 = false;
        }
    }
    uint _S1109 = lp_body_0(s_17, kernelContext_155);
    float4 _S1110 = float4(*(kernelContext_155->lp_0+(_S1109 + 2U))) ;
    float4 _S1111 = float4(*(kernelContext_155->lp_0+(_S1109 + 3U))) ;
    uint _S1112 = 7U * s_17;
    thread Quat_0 q_27;
    thread float3 origin_8;
    structure_pose_0(s_17, _S1112, &q_27, &origin_8, kernelContext_155);
    thread Quat_0 _S1113 = q_27;
    Quat_0 _S1114 = conj_0(&_S1113);
    float3 _S1115 = _S1111.xyz * float3(_S1110.w) ;
    thread Quat_0 _S1116 = _S1114;
    float3 _S1117 = rotate_0(&_S1116, _S1115);
    float3 _S1118 = _S1110.xyz - origin_8;
    thread Quat_0 _S1119 = _S1114;
    float3 _S1120 = rotate_0(&_S1119, _S1118);
    uint _S1121 = 4U * k_40;
    float4 _S1122 = float4(*(kernelContext_155->chunks_0+_S1121)) ;
    float3 _S1123 = float3(0.0f) ;
    uint _S1124 = atomic_load_explicit(kernelContext_155->atom_0+(_S1106 + 1U), memory_order_relaxed);
    float3 f_12;
    float3 t_7;
    if(_S1124 == k_40)
    {
        float3 _S1125 = cross(_S1120 - _S1122.xyz, _S1117);
        f_12 = _S1117;
        t_7 = _S1125;
    }
    else
    {
        f_12 = _S1123;
        t_7 = _S1123;
    }
    float inv_mass_7 = (float4(*(kernelContext_155->bodies_0+_S1112)) ).w;
    bool _S1126;
    if(!supported_1)
    {
        _S1126 = inv_mass_7 > 0.0f;
    }
    else
    {
        _S1126 = false;
    }
    if(_S1126)
    {
        uint _S1127 = 2U * s_17;
        float3 com_2 = (float4(*(kernelContext_155->frames_0+_S1127)) ).xyz;
        Quat_0 p_17 = quat_of_0(float4(*(kernelContext_155->frames_0+(_S1127 + 1U))) );
        float3 _S1128 = (float4(*(kernelContext_155->bodies_0+(_S1112 + 2U))) ).xyz;
        thread Quat_0 _S1129 = p_17;
        Quat_0 _S1130 = conj_0(&_S1129);
        float3 _S1131 = cross(_S1120 - com_2, _S1117);
        thread Quat_0 _S1132 = _S1130;
        float3 _S1133 = rotate_0(&_S1132, _S1131);
        float3 _S1134 = _S1128 * _S1133;
        thread Quat_0 _S1135 = p_17;
        float3 _S1136 = rotate_0(&_S1135, _S1134);
        float4 _S1137 = float4(*(kernelContext_155->chunks_0+(_S1121 + 1U))) ;
        float4 _S1138 = float4(*(kernelContext_155->chunks_0+(_S1121 + 2U))) ;
        float _S1139 = _S1136.x;
        float _S1140 = _S1137.w;
        float _S1141 = _S1136.y;
        float _S1142 = _S1138.x;
        float _S1143 = _S1136.z;
        float _S1144 = _S1138.y;
        float3 t_8 = t_7 - float3(_S1137.x * _S1139 + _S1140 * _S1141 + _S1142 * _S1143, _S1140 * _S1139 + _S1137.y * _S1141 + _S1144 * _S1143, _S1142 * _S1139 + _S1144 * _S1141 + _S1137.z * _S1143);
        f_12 = f_12 - float3(_S1122.w)  * (_S1117 * float3(inv_mass_7)  + cross(_S1136, _S1122.xyz - com_2));
        t_7 = t_8;
    }
    uint _S1145 = st_f_0(k_40, 14U);
    packed_float4 device* _S1146 = kernelContext_155->lp_0+_S1145;
    float _S1147;
    if(fixed_2)
    {
        _S1147 = 1.0f;
    }
    else
    {
        _S1147 = 0.0f;
    }
    *_S1146 = packed_float4(float4(f_12, _S1147)) ;
    *(kernelContext_155->lp_0+(_S1145 + 1U)) = packed_float4(float4(t_7, 0.0f)) ;
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
        uint _S1148 = st_x_0(k_40, 14U, c_23);
        packed_float4 _S1149 = packed_float4(float4(0.0f) ) ;
        *(kernelContext_155->lp_0+_S1148) = _S1149;
        *(kernelContext_155->lp_0+(_S1148 + 1U)) = _S1149;
        c_23 = c_23 + 1U;
    }
    uint _S1150 = st_d_0(k_40, 14U);
    packed_float4 _S1151 = packed_float4(float4(0.0f) ) ;
    *(kernelContext_155->lp_0+_S1150) = _S1151;
    *(kernelContext_155->lp_0+(_S1150 + 1U)) = _S1151;
    return;
}

void cs_break_0(uint i_12, uint s_18, KernelContext_0 thread* kernelContext_156)
{
    uint4 _S1152 = uint4(*(kernelContext_156->bonds_0+i_12)) ;
    LpBond_0 _S1153 = lp_bond_0(i_12, kernelContext_156);
    uint _S1154 = _S1152.x;
    uint _S1155 = _S1152.y;
    thread LpBond_0 _S1156 = _S1153;
    thread float3 q_lin_6;
    thread float3 q_ang_6;
    st_force_0(&_S1156, _S1154, _S1155, 14U, 0U, &q_lin_6, &q_ang_6, kernelContext_156);
    if((kernelContext_156->params_0->load_model_0) == 1U)
    {
        thread LpBond_0 _S1157 = _S1153;
        thread float3 b_lin_0;
        thread float3 b_ang_0;
        lp_force_0(&_S1157, _S1154, _S1155, 0U, &b_lin_0, &b_ang_0, kernelContext_156);
        q_lin_6 = q_lin_6 + b_lin_0;
        q_ang_6 = q_ang_6 + b_ang_0;
    }
    thread JointBond_0 _S1158 = _S1153.law_0;
    Measures_0 _S1159 = stress_measures_0(&_S1158, q_lin_6, q_ang_6);
    JointMaterial_0 _S1160 = lp_material_0(_S1153.law_0.ids_1.x, kernelContext_156);
    float _S1161 = _S1153.law_0.geom1_0.w;
    thread JointMaterial_0 _S1162 = _S1160;
    thread JointBond_0 _S1163 = _S1153.law_0;
    thread Measures_0 _S1164 = _S1159;
    float4 _S1165 = failure_indices_0(&_S1162, &_S1163, &_S1164, _S1161);
    if((max(max(_S1165.x, _S1165.y), max(_S1165.z, _S1165.w))) >= 1.0f)
    {
        uint _S1166 = w_alive_0(kernelContext_156);
        *(kernelContext_156->work_0+(_S1166 + i_12)) = 0U;
        uint _S1167 = a_dirty_0(kernelContext_156);
        uint _S1168 = atomic_fetch_or_explicit(kernelContext_156->atom_0+(_S1167 + s_18), 1U, memory_order_relaxed);
        uint _S1169 = a_counters_1(kernelContext_156);
        uint _S1170 = atomic_fetch_add_explicit(kernelContext_156->atom_0+_S1169, 1U, memory_order_relaxed);
        uint _S1171 = atomic_fetch_add_explicit(kernelContext_156->atom_0+(_S1169 + 1U), 1U, memory_order_relaxed);
        uint _S1172 = atomic_fetch_add_explicit(kernelContext_156->atom_0+(_S1169 + 13U), 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void fr_cs_solve(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], FractureParams_0 constant* params_64 [[buffer(0)]], uint device* work_64 [[buffer(4)]], packed_float4 device* frames_64 [[buffer(6)]], packed_float4 device* bodies_64 [[buffer(7)]], packed_float4 device* chunks_64 [[buffer(1)]], uint device* sort_64 [[buffer(11)]], atomic<uint> device* atom_64 [[buffer(5)]], Record_natural_0 device* manifolds_64 [[buffer(10)]], packed_float4 device* loads_64 [[buffer(12)]], packed_uint4 device* bonds_64 [[buffer(2)]], uint device* args_64 [[buffer(13)]], uint device* info_64 [[buffer(15)]], ScanParams_0 constant* scan_64 [[buffer(14)]], packed_float4 device* parts_64 [[buffer(3)]], packed_float4 device* colliders_64 [[buffer(8)]], packed_uint4 device* collider_info_64 [[buffer(9)]], packed_float4 device* view_chunks_64 [[buffer(17)]], ViewParams_0 constant* view_64 [[buffer(16)]], packed_float4 device* objects_64 [[buffer(18)]], uint device* shot_bodies_64 [[buffer(20)]], float device* lines_64 [[buffer(19)]], packed_float4 device* geometry_64 [[buffer(21)]], packed_float4 device* shots_64 [[buffer(22)]], packed_float4 device* lp_64 [[buffer(24)]], packed_float4 device* ext_66 [[buffer(23)]], uint device* engine_counters_64 [[buffer(26)]], packed_float4 device* engine_contacts_64 [[buffer(25)]], packed_float4 device* cs_anchors_64 [[buffer(27)]])
{
    bool _S1173;
    uint j3_0;
    thread KernelContext_0 kernelContext_157;
    (&kernelContext_157)->params_0 = params_64;
    (&kernelContext_157)->work_0 = work_64;
    (&kernelContext_157)->frames_0 = frames_64;
    (&kernelContext_157)->bodies_0 = bodies_64;
    (&kernelContext_157)->chunks_0 = chunks_64;
    (&kernelContext_157)->sort_0 = sort_64;
    (&kernelContext_157)->atom_0 = atom_64;
    (&kernelContext_157)->manifolds_0 = manifolds_64;
    (&kernelContext_157)->loads_0 = loads_64;
    (&kernelContext_157)->bonds_0 = bonds_64;
    (&kernelContext_157)->args_0 = args_64;
    (&kernelContext_157)->info_0 = info_64;
    (&kernelContext_157)->scan_0 = scan_64;
    (&kernelContext_157)->parts_0 = parts_64;
    (&kernelContext_157)->colliders_0 = colliders_64;
    (&kernelContext_157)->collider_info_0 = collider_info_64;
    (&kernelContext_157)->view_chunks_0 = view_chunks_64;
    (&kernelContext_157)->view_0 = view_64;
    (&kernelContext_157)->objects_0 = objects_64;
    (&kernelContext_157)->shot_bodies_0 = shot_bodies_64;
    (&kernelContext_157)->lines_0 = lines_64;
    (&kernelContext_157)->geometry_0 = geometry_64;
    (&kernelContext_157)->shots_0 = shots_64;
    (&kernelContext_157)->lp_0 = lp_64;
    (&kernelContext_157)->ext_0 = ext_66;
    (&kernelContext_157)->engine_counters_0 = engine_counters_64;
    (&kernelContext_157)->engine_contacts_0 = engine_contacts_64;
    (&kernelContext_157)->cs_anchors_0 = cs_anchors_64;
    threadgroup array<uint, int(1024)> g_scan_64;
    (&kernelContext_157)->g_scan_0 = &g_scan_64;
    uint _S1174 = w_cs_bodies_0(&kernelContext_157);
    uint device* _S1175 = (&kernelContext_157)->work_0+(_S1174 + group_1.x);
    uint s_19 = *_S1175;
    uint t_9 = thread_2.x;
    uint _S1176 = a_cs_0(*_S1175, &kernelContext_157);
    uint _S1177 = atomic_load_explicit((&kernelContext_157)->atom_0+(_S1176 + 6U), memory_order_relaxed);
    uint _S1178 = atomic_load_explicit((&kernelContext_157)->atom_0+(_S1176 + 5U), memory_order_relaxed);
    uint j_8 = t_9;
    for(;;)
    {
        if(j_8 < _S1178)
        {
        }
        else
        {
            break;
        }
        uint _S1179 = w_cs_list_0(&kernelContext_157);
        cs_load_0(*((&kernelContext_157)->work_0+(_S1179 + _S1177 + j_8)), s_19, &kernelContext_157);
        j_8 = j_8 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint j2_0 = t_9;
    for(;;)
    {
        if(j2_0 < _S1178)
        {
        }
        else
        {
            break;
        }
        uint _S1180 = w_cs_list_0(&kernelContext_157);
        uint device* _S1181 = (&kernelContext_157)->work_0+(_S1180 + _S1177 + j2_0);
        uint k_41 = *_S1181;
        uint _S1182 = st_f_0(*_S1181, 14U);
        if(((float4(*((&kernelContext_157)->lp_0+_S1182)) ).w) == 0.0f)
        {
            bool _S1183 = lp_factor_0(k_41, &kernelContext_157);
            _S1173 = !_S1183;
        }
        else
        {
            _S1173 = false;
        }
        if(_S1173)
        {
            (*((&kernelContext_157)->lp_0+_S1182))[int(3)] = 1.0f;
            uint _S1184 = a_counters_1(&kernelContext_157);
            uint _S1185 = atomic_fetch_or_explicit((&kernelContext_157)->atom_0+(_S1184 + 3U), 32U, memory_order_relaxed);
        }
        j2_0 = j2_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint sweep_0 = 0U;
    for(;;)
    {
        if(sweep_0 < ((&kernelContext_157)->params_0->cs_sweeps_0))
        {
        }
        else
        {
            break;
        }
        j3_0 = t_9;
        for(;;)
        {
            if(j3_0 < _S1178)
            {
            }
            else
            {
                break;
            }
            uint _S1186 = w_cs_list_0(&kernelContext_157);
            st_sweep_0(*((&kernelContext_157)->work_0+(_S1186 + _S1177 + j3_0)), 14U, sweep_0 % 2U, (&kernelContext_157)->params_0->cs_c1_0, (&kernelContext_157)->params_0->cs_c2_0, &kernelContext_157);
            j3_0 = j3_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        sweep_0 = sweep_0 + 1U;
    }
    j3_0 = t_9;
    for(;;)
    {
        if(j3_0 < _S1178)
        {
        }
        else
        {
            break;
        }
        uint _S1187 = w_cs_list_0(&kernelContext_157);
        uint device* _S1188 = (&kernelContext_157)->work_0+(_S1187 + _S1177 + j3_0);
        uint k_42 = *_S1188;
        float4 _S1189 = float4(*((&kernelContext_157)->ext_0+((&kernelContext_157)->params_0->ext_chunks_0 + 4U * *_S1188))) ;
        uint _S1190 = (as_type<uint>((_S1189.x)));
        uint _S1191 = (as_type<uint>((_S1189.y)));
        uint e_4 = 0U;
        for(;;)
        {
            if(e_4 < _S1191)
            {
            }
            else
            {
                break;
            }
            uint _S1192 = adj_bond_0(_S1190 + e_4, &kernelContext_157);
            if(((uint4(*((&kernelContext_157)->bonds_0+_S1192)) ).x) == k_42)
            {
                uint _S1193 = w_alive_0(&kernelContext_157);
                _S1173 = (*((&kernelContext_157)->work_0+(_S1193 + _S1192))) != 0U;
            }
            else
            {
                _S1173 = false;
            }
            if(_S1173)
            {
                cs_break_0(_S1192, s_19, &kernelContext_157);
            }
            e_4 = e_4 + 1U;
        }
        j3_0 = j3_0 + 256U;
    }
    return;
}

