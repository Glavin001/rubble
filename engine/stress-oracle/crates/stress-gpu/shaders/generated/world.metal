#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Params_0
{
    float4 gravity_0;
    float dt_0;
    uint fracture_0;
    uint rigid_motion_loads_0;
    uint step_start_0;
    float t_hi_0;
    float t_lo_0;
    uint max_steps_0;
    uint contact_mode_0;
    uint halt_index_0;
    uint chunk_count_0;
    uint pair_count_0;
    uint impactor_count_0;
    uint cand_count_0;
    uint probe_base_0;
    uint probe_stride_0;
    uint slot_base_0;
    uint ledger_base_0;
    uint record_base_0;
    uint record_stride_0;
    uint cand_base_0;
    uint pair_index_0;
    uint cand_index_0;
    float zeta_0;
    float pair_friction_0;
    float ground_hi_0;
    float ground_lo_0;
    float ground_friction_0;
    float ground_modulus_0;
    uint has_ground_0;
    uint wide_bond_groups_0;
    uint wide_bond_table_0;
    uint wide_chunk_table_0;
    uint wide_base_0;
    uint seg_index_0;
    uint seg_count_0;
    uint seg_base_0;
    uint solve_mode_0;
    uint cload_base_0;
    uint cframe_base_0;
    uint statics_base_0;
    uint statics_bonds_0;
    uint statics_newton_0;
    uint statics_cg_0;
    float statics_tol_0;
    uint pair_lanes_0;
    uint device_connect_0;
    uint cplan_base_0;
    uint iplan_base_0;
    uint contact_bond_groups_0;
    uint contact_bond_table_0;
};

struct Island_natural_0
{
    packed_uint4 range_0;
    packed_uint4 info_0;
    packed_float4 com_0;
    packed_float4 inertia0_0;
    packed_float4 inertia1_0;
    packed_float4 inertia2_0;
    packed_float4 inv0_0;
    packed_float4 inv1_0;
    packed_float4 inv2_0;
    packed_float4 wcom_0;
    packed_float4 winv0_0;
    packed_float4 winv1_0;
    packed_float4 winv2_0;
    packed_float4 rotation_0;
    packed_float4 position_0;
    packed_float4 position_err_0;
    packed_float4 velocity_0;
    packed_float4 velocity_err_0;
    packed_float4 angular_velocity_0;
    uint done_0;
    uint coverage_0;
    uint wide_groups_0;
    uint done_pad_0;
    packed_uint4 probes_0;
    packed_float4 energy_0;
    packed_float4 rotation_err_0;
    packed_float4 momentum_0;
    packed_float4 momentum_err_0;
};

struct ChunkStatic_natural_0
{
    packed_float4 center_0;
    packed_float4 inertia0_1;
    packed_float4 inertia1_1;
    packed_float4 inertia2_1;
    packed_float4 inv0_1;
    packed_float4 inv1_1;
    packed_float4 inv2_1;
    packed_float4 scale_0;
    packed_uint4 info_1;
    packed_uint4 load_range_0;
    packed_float4 half_0;
    packed_float4 crot0_0;
    packed_float4 crot1_0;
    packed_float4 crot2_0;
    packed_float4 cmat_0;
    packed_uint4 cinfo_0;
};

struct Impactor_natural_0
{
    packed_float4 position_1;
    packed_float4 position_err_1;
    packed_float4 velocity_1;
    packed_float4 velocity_err_1;
    packed_float4 angular_velocity_1;
    packed_float4 rotation_1;
    packed_float4 inertia0_2;
    packed_float4 inertia1_2;
    packed_float4 inertia2_2;
    packed_float4 inv0_2;
    packed_float4 inv1_2;
    packed_float4 inv2_2;
    packed_float4 shape_0;
    packed_float4 half_1;
    packed_float4 mat_0;
    packed_float4 crush_0;
    packed_float4 geom_0;
    packed_float4 rotation_err_1;
    packed_float4 ledger_0;
    packed_uint4 cand_0;
    packed_float4 momentum_1;
    packed_float4 momentum_err_1;
};

struct JointState_0
{
    float damage_0;
    float crush_1;
    float kappa_0;
    float kappa_c_0;
    float ductility_0;
    float ductility_c_0;
    float life_0;
    float plastic_x_0;
    float plastic_y_0;
    float plastic_t_0;
    float rebar_plastic_0;
    float rebar_slip0_0;
    float rebar_slip1_0;
    float rebar_work_0;
    float rebar_broken_0;
    float strain_rate_0;
    float governing_stress_0;
    float dissipated_0;
    float utilization_0;
    uint mode_0;
};

struct BondDyn_natural_0
{
    JointState_0 js_0;
    packed_float4 force_lin_0;
    packed_float4 force_ang_0;
    packed_float4 sums_0;
    packed_float4 comps_0;
    packed_uint4 events_0;
};

struct JointBond_natural_0
{
    packed_float4 geom0_0;
    packed_float4 geom1_0;
    packed_float4 stiff0_0;
    packed_float4 stiff1_0;
    packed_float4 rebar0_0;
    packed_float4 rebar1_0;
    packed_uint4 ids_0;
};

struct BondStatic_natural_0
{
    packed_float4 t1_0;
    packed_float4 t2_0;
    packed_float4 normal_0;
    packed_float4 ra_0;
    packed_float4 rb_0;
    packed_float4 centroid_0;
    packed_float4 c_lin_0;
    packed_float4 c_ang_0;
    JointBond_natural_0 law_0;
};

struct JointMaterial_0
{
    float4 strength_0;
    float4 energy_1;
    float4 dif_0;
    float4 misc_0;
    uint4 kind_flags_0;
};

struct MaterialTable_0
{
    array<JointMaterial_0, int(64)> m_0;
};

struct KernelContext_0
{
    Params_0 constant* params_0;
    Island_natural_0 device* islands_0;
    uint device* index_0;
    ChunkStatic_natural_0 device* chunks_0;
    packed_float4 device* state_0;
    packed_float4 device* scratch_0;
    packed_float4 device* contact_state_0;
    packed_float4 device* loads_0;
    Impactor_natural_0 device* impactors_0;
    BondDyn_natural_0 device* bond_dyn_0;
    BondStatic_natural_0 device* bonds_0;
    MaterialTable_0 constant* materials_0;
    array<float4, int(32)> threadgroup* g_part_a_0;
    array<float4, int(32)> threadgroup* g_part_b_0;
    array<uint, int(5)> threadgroup* g_seen_0;
    uint threadgroup* g_wide_run_0;
    array<uint, int(4096)> threadgroup* g_label_0;
    uint threadgroup* g_changed_0;
    uint threadgroup* g_halt_0;
    uint threadgroup* g_run_0;
    uint threadgroup* g_sums_0;
    array<uint, int(512)> threadgroup* g_label_narrow_0;
    uint threadgroup* g_changed_narrow_0;
};

struct Island_0
{
    uint4 range_0;
    uint4 info_0;
    float4 com_0;
    float4 inertia0_0;
    float4 inertia1_0;
    float4 inertia2_0;
    float4 inv0_0;
    float4 inv1_0;
    float4 inv2_0;
    float4 wcom_0;
    float4 winv0_0;
    float4 winv1_0;
    float4 winv2_0;
    float4 rotation_0;
    float4 position_0;
    float4 position_err_0;
    float4 velocity_0;
    float4 velocity_err_0;
    float4 angular_velocity_0;
    uint done_0;
    uint coverage_0;
    uint wide_groups_0;
    uint done_pad_0;
    uint4 probes_0;
    float4 energy_0;
    float4 rotation_err_0;
    float4 momentum_0;
    float4 momentum_err_0;
};

void seen_clear_0(uint tid_0, KernelContext_0 thread* kernelContext_0)
{
    if(tid_0 < 5U)
    {
        (*kernelContext_0->g_seen_0)[tid_0] = 0U;
    }
    return;
}

bool contact_stopped_0(const Island_0 thread* isl_0, KernelContext_0 thread* kernelContext_1)
{
    Island_natural_0 device* _S1 = kernelContext_1->islands_0+kernelContext_1->params_0->halt_index_0;
    uint4 _S2 = uint4(_S1->info_0) ;
    bool _S3;
    if(((_S2.z) & 1U) != 0U)
    {
        _S3 = true;
    }
    else
    {
        uint _S4 = _S2.y;
        if(_S4 != 0U)
        {
            _S3 = _S4 <= (isl_0->info_0.w);
        }
        else
        {
            _S3 = false;
        }
    }
    if(_S3)
    {
        _S3 = true;
    }
    else
    {
        uint _S5 = (uint4(_S1->range_0) ).x;
        if(_S5 != 0U)
        {
            _S3 = _S5 <= (isl_0->info_0.w + 1U);
        }
        else
        {
            _S3 = false;
        }
    }
    return _S3;
}

bool stop_genuine_0(uint _S6, uint _S7, KernelContext_0 thread* kernelContext_2)
{
    Island_natural_0 device* _S8 = kernelContext_2->islands_0+_S6;
    uint at_0 = (uint4(_S8->range_0) ).x;
    bool _S9;
    if(at_0 != 0U)
    {
        _S9 = at_0 <= _S7;
    }
    else
    {
        _S9 = false;
    }
    if(_S9)
    {
        uint _S10 = (uint4(_S8->info_0) ).y;
        if(_S10 == 0U)
        {
            _S9 = true;
        }
        else
        {
            _S9 = at_0 <= _S10;
        }
    }
    else
    {
        _S9 = false;
    }
    return _S9;
}

void cov_or_0(uint island_0, uint bits_0, KernelContext_0 thread* kernelContext_3)
{
    uint _S11 = atomic_fetch_or_explicit(((atomic_uint device*)(&(kernelContext_3->islands_0+island_0)->coverage_0)), bits_0, memory_order_relaxed);
    return;
}

void cov_mark_0(uint island_1, uint seen_0, uint bits_1, KernelContext_0 thread* kernelContext_4)
{
    if((bits_1 & (~seen_0)) != 0U)
    {
        cov_or_0(island_1, bits_1, kernelContext_4);
    }
    return;
}

void sentinel_mark_0(uint bits_2, KernelContext_0 thread* kernelContext_5)
{
    cov_mark_0(kernelContext_5->params_0->halt_index_0, (kernelContext_5->islands_0+kernelContext_5->params_0->halt_index_0)->coverage_0, bits_2, kernelContext_5);
    return;
}

void stop_seen_0(uint _S12, uint _S13, KernelContext_0 thread* kernelContext_6)
{
    bool _S14 = stop_genuine_0(_S12, _S13, kernelContext_6);
    if(_S14)
    {
        (kernelContext_6->islands_0+kernelContext_6->params_0->halt_index_0)->info_0[int(2)] = ((uint4((kernelContext_6->islands_0+kernelContext_6->params_0->halt_index_0)->info_0) ).z) | 1U;
        sentinel_mark_0(1048576U, kernelContext_6);
    }
    return;
}

struct Quat_0
{
    float w_0;
    float x_0;
    float y_0;
    float z_0;
};

struct Rigid_0
{
    Quat_0 rot_0;
    float4 rot_err_0;
    float3 pos_0;
    float3 pos_err_0;
    float3 vel_0;
    float3 vel_err_0;
    float3 w_1;
    float3 l_0;
    float3 l_err_0;
    float3 torque_0;
    float3 a_0;
    float3 alpha_0;
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

Rigid_0 rigid_of_0(const Island_0 thread* isl_1)
{
    thread Rigid_0 rg_0;
    (&rg_0)->rot_0 = quat_of_0(isl_1->rotation_0);
    (&rg_0)->rot_err_0 = isl_1->rotation_err_0;
    (&rg_0)->pos_0 = isl_1->position_0.xyz;
    (&rg_0)->pos_err_0 = isl_1->position_err_0.xyz;
    (&rg_0)->vel_0 = isl_1->velocity_0.xyz;
    (&rg_0)->vel_err_0 = isl_1->velocity_err_0.xyz;
    (&rg_0)->w_1 = isl_1->angular_velocity_0.xyz;
    (&rg_0)->l_0 = isl_1->momentum_0.xyz;
    (&rg_0)->l_err_0 = isl_1->momentum_err_0.xyz;
    float3 _S15 = float3(0.0f) ;
    (&rg_0)->torque_0 = _S15;
    (&rg_0)->a_0 = _S15;
    (&rg_0)->alpha_0 = _S15;
    return rg_0;
}

struct WorldPoint_0
{
    float3 hi_0;
    float3 lo_0;
    float3 rel_0;
};

float3 rotate_0(const Quat_0 thread* q_1, float3 v_0)
{
    float3 qv_0 = float3(q_1->x_0, q_1->y_0, q_1->z_0);
    float3 t_0 = cross(qv_0, v_0) * float3(2.0f) ;
    return v_0 + t_0 * float3(q_1->w_0)  + cross(qv_0, t_0);
}

WorldPoint_0 chunk_world_0(uint c_0, KernelContext_0 thread* kernelContext_7)
{
    ChunkStatic_natural_0 device* _S16 = kernelContext_7->chunks_0+c_0;
    Island_natural_0 device* _S17 = kernelContext_7->islands_0+(uint4(_S16->info_1) ).y;
    Quat_0 q_2 = quat_of_0(float4(_S17->rotation_0) );
    thread WorldPoint_0 w_2;
    (&w_2)->hi_0 = (float4(_S17->position_0) ).xyz;
    (&w_2)->lo_0 = (float4(_S17->position_err_0) ).xyz;
    float3 _S18 = (float4(_S16->center_0) ).xyz + (float4(*(kernelContext_7->state_0+4U * c_0)) ).xyz;
    thread Quat_0 _S19 = q_2;
    float3 _S20 = rotate_0(&_S19, _S18);
    (&w_2)->rel_0 = _S20;
    return w_2;
}

struct Box_0
{
    float3 center_1;
    float3 axis0_0;
    float3 axis1_0;
    float3 axis2_0;
    float3 half_2;
    uint hull_at_0;
    uint hull_v_0;
    uint hull_f_0;
};

float3 safe_normalize_0(float3 v_1)
{
    float n_0 = length(v_1);
    float3 _S21;
    if(n_0 > 1.00000000317107685e-30f)
    {
        _S21 = v_1 / float3(n_0) ;
    }
    else
    {
        _S21 = float3(0.0f) ;
    }
    return _S21;
}

Quat_0 from_axis_angle_0(float3 axis_0, float angle_0)
{
    float3 a_1 = safe_normalize_0(axis_0);
    float _S22 = 0.5f * angle_0;
    float s_0 = sin(_S22);
    thread Quat_0 q_3;
    (&q_3)->w_0 = cos(_S22);
    (&q_3)->x_0 = a_1.x * s_0;
    (&q_3)->y_0 = a_1.y * s_0;
    (&q_3)->z_0 = a_1.z * s_0;
    return q_3;
}

Box_0 chunk_box_0(uint c_1, float3 center_2, KernelContext_0 thread* kernelContext_8)
{
    ChunkStatic_natural_0 device* _S23 = kernelContext_8->chunks_0+c_1;
    Quat_0 q_4 = quat_of_0(float4((kernelContext_8->islands_0+(uint4(_S23->info_1) ).y)->rotation_0) );
    float3 th_0 = (float4(*(kernelContext_8->state_0+(4U * c_1 + 1U))) ).xyz;
    Quat_0 hidden_0 = from_axis_angle_0(th_0, length(th_0));
    thread Box_0 b_0;
    (&b_0)->center_1 = center_2;
    float4 _S24 = float4(_S23->crot0_0) ;
    float4 _S25 = float4(_S23->crot1_0) ;
    float4 _S26 = float4(_S23->crot2_0) ;
    float3 _S27 = float3(_S24.x, _S25.x, _S26.x);
    thread Quat_0 _S28 = hidden_0;
    float3 _S29 = rotate_0(&_S28, _S27);
    thread Quat_0 _S30 = q_4;
    float3 _S31 = rotate_0(&_S30, _S29);
    (&b_0)->axis0_0 = _S31;
    float3 _S32 = float3(_S24.y, _S25.y, _S26.y);
    thread Quat_0 _S33 = hidden_0;
    float3 _S34 = rotate_0(&_S33, _S32);
    thread Quat_0 _S35 = q_4;
    float3 _S36 = rotate_0(&_S35, _S34);
    (&b_0)->axis1_0 = _S36;
    float3 _S37 = float3(_S24.z, _S25.z, _S26.z);
    thread Quat_0 _S38 = hidden_0;
    float3 _S39 = rotate_0(&_S38, _S37);
    thread Quat_0 _S40 = q_4;
    float3 _S41 = rotate_0(&_S40, _S39);
    (&b_0)->axis2_0 = _S41;
    (&b_0)->half_2 = (float4(_S23->half_0) ).xyz;
    float4 _S42 = float4(_S23->cmat_0) ;
    (&b_0)->hull_at_0 = (as_type<uint>((_S42.z)));
    uint _S43 = (as_type<uint>((_S42.w)));
    (&b_0)->hull_v_0 = _S43 & 255U;
    (&b_0)->hull_f_0 = _S43 >> 8U;
    return b_0;
}

float3 box_axis_0(const Box_0 thread* b_1, uint k_0)
{
    float3 _S44;
    if(k_0 == 0U)
    {
        _S44 = b_1->axis0_0;
    }
    else
    {
        if(k_0 == 1U)
        {
            _S44 = b_1->axis1_0;
        }
        else
        {
            _S44 = b_1->axis2_0;
        }
    }
    return _S44;
}

float comp3_0(float3 v_2, uint k_1)
{
    float _S45;
    if(k_1 == 0U)
    {
        _S45 = v_2.x;
    }
    else
    {
        if(k_1 == 1U)
        {
            _S45 = v_2.y;
        }
        else
        {
            _S45 = v_2.z;
        }
    }
    return _S45;
}

float2 half_thickness_and_area_0(const Box_0 thread* b_2, float3 d_0)
{
    uint k_2 = 0U;
    float h_0 = 0.0f;
    float area_0 = 0.0f;
    for(;;)
    {
        if(k_2 < 3U)
        {
        }
        else
        {
            break;
        }
        float3 _S46 = box_axis_0(b_2, k_2);
        float c_2 = abs(dot(d_0, _S46));
        float h_1 = h_0 + c_2 * comp3_0(b_2->half_2, k_2);
        uint _S47 = k_2 + 1U;
        float area_1 = area_0 + c_2 * 4.0f * comp3_0(b_2->half_2, _S47 % 3U) * comp3_0(b_2->half_2, (k_2 + 2U) % 3U);
        k_2 = _S47;
        h_0 = h_1;
        area_0 = area_1;
    }
    return float2(h_0, area_0);
}

float contact_stiffness_0(float ea_0, const Box_0 thread* a_2, float eb_0, const Box_0 thread* b_3, float3 dir_0)
{
    float3 d_1 = safe_normalize_0(dir_0);
    float2 _S48 = half_thickness_and_area_0(a_2, d_1);
    float2 _S49 = half_thickness_and_area_0(b_3, d_1);
    return min(_S48.y, _S49.y) / (_S48.x / ea_0 + _S49.x / eb_0);
}

uint sample_count_0(const Box_0 thread* b_4)
{
    uint _S50 = b_4->hull_v_0;
    uint _S51;
    if((b_4->hull_v_0) == 0U)
    {
        _S51 = 14U;
    }
    else
    {
        _S51 = _S50 + b_4->hull_f_0;
    }
    return _S51;
}

float3 box_to_world_0(const Box_0 thread* b_5, float3 local_0)
{
    return b_5->axis0_0 * float3(local_0.x)  + b_5->axis1_0 * float3(local_0.y)  + b_5->axis2_0 * float3(local_0.z) ;
}

float3 sample_point_0(const Box_0 thread* b_6, uint i_0, KernelContext_0 thread* kernelContext_9)
{
    uint _S52 = b_6->hull_v_0;
    if((b_6->hull_v_0) != 0U)
    {
        float3 local_1;
        if(i_0 < _S52)
        {
            local_1 = (float4(*(kernelContext_9->loads_0+(b_6->hull_at_0 + i_0))) ).xyz * float3(0.89999997615814209f) ;
        }
        else
        {
            local_1 = (float4(*(kernelContext_9->loads_0+(b_6->hull_at_0 + _S52 + b_6->hull_f_0 + (i_0 - _S52)))) ).xyz;
        }
        float3 _S53 = b_6->center_1;
        float3 _S54 = box_to_world_0(b_6, local_1);
        return _S53 + _S54;
    }
    float sign_0;
    if(i_0 < 8U)
    {
        float3 h_2 = b_6->half_2 * float3(0.89999997615814209f) ;
        if((i_0 & 1U) == 0U)
        {
            sign_0 = - h_2.x;
        }
        else
        {
            sign_0 = h_2.x;
        }
        float _S55;
        if((i_0 & 2U) == 0U)
        {
            _S55 = - h_2.y;
        }
        else
        {
            _S55 = h_2.y;
        }
        float _S56;
        if((i_0 & 4U) == 0U)
        {
            _S56 = - h_2.z;
        }
        else
        {
            _S56 = h_2.z;
        }
        return b_6->center_1 + b_6->axis0_0 * float3(sign_0)  + b_6->axis1_0 * float3(_S55)  + b_6->axis2_0 * float3(_S56) ;
    }
    uint _S57 = i_0 - 8U;
    uint axis_1 = _S57 / 2U;
    if((_S57 % 2U) == 0U)
    {
        sign_0 = -1.0f;
    }
    else
    {
        sign_0 = 1.0f;
    }
    float3 _S58 = b_6->center_1;
    float3 _S59 = box_axis_0(b_6, axis_1);
    return _S58 + _S59 * float3((sign_0 * comp3_0(b_6->half_2, axis_1))) ;
}

void chunk_velocity_0(uint c_3, float3 thread* v_3, float3 thread* w_3, KernelContext_0 thread* kernelContext_10)
{
    ChunkStatic_natural_0 device* _S60 = kernelContext_10->chunks_0+c_3;
    Island_natural_0 device* _S61 = kernelContext_10->islands_0+(uint4(_S60->info_1) ).y;
    Quat_0 q_5 = quat_of_0(float4(_S61->rotation_0) );
    uint _S62 = 4U * c_3;
    float3 _S63 = (float4(_S60->center_0) ).xyz + (float4(*(kernelContext_10->state_0+_S62)) ).xyz - (float4(_S61->com_0) ).xyz;
    thread Quat_0 _S64 = q_5;
    float3 _S65 = rotate_0(&_S64, _S63);
    float3 _S66 = (float4(_S61->angular_velocity_0) ).xyz;
    float3 _S67 = (float4(_S61->velocity_0) ).xyz + (float4(_S61->velocity_err_0) ).xyz + cross(_S66, _S65);
    float3 _S68 = (float4(*(kernelContext_10->state_0+(_S62 + 2U))) ).xyz;
    thread Quat_0 _S69 = q_5;
    float3 _S70 = rotate_0(&_S69, _S68);
    *v_3 = _S67 + _S70;
    float3 _S71 = (float4(*(kernelContext_10->state_0+(_S62 + 3U))) ).xyz;
    thread Quat_0 _S72 = q_5;
    float3 _S73 = rotate_0(&_S72, _S71);
    *w_3 = _S66 + _S73;
    return;
}

float3 penalty_force_0(float k_3, float m_red_0, float friction_0, float depth_0, float3 normal_1, float3 rel_velocity_0, float dt_1, uint points_0, float thread* stored_0, float thread* dissipated_1, KernelContext_0 thread* kernelContext_11)
{
    float c_max_0 = 1.0f / max(float(points_0), 10.0f) * m_red_0 / dt_1;
    float vn_0 = dot(rel_velocity_0, normal_1);
    float _S74 = k_3 * depth_0;
    float _S75 = min(2.0f * kernelContext_11->params_0->zeta_0 * sqrt(k_3 * m_red_0), c_max_0) * vn_0;
    float _S76 = _S74 - _S75;
    float _S77 = max(_S76, 0.0f);
    float3 vt_0 = rel_velocity_0 - normal_1 * float3(vn_0) ;
    float vt_mag_0 = length(vt_0);
    float _S78 = friction_0 * _S77;
    float _S79 = min(_S78, min(c_max_0, _S78 / 0.00100000004749745f) * vt_mag_0);
    float3 ft_0;
    if(vt_mag_0 > 0.0f)
    {
        ft_0 = - vt_0 * float3((_S79 / vt_mag_0)) ;
    }
    else
    {
        ft_0 = float3(0.0f) ;
    }
    *stored_0 = 0.5f * k_3 * depth_0 * depth_0;
    float damping_power_0;
    if(_S76 > 0.0f)
    {
        damping_power_0 = _S75 * vn_0;
    }
    else
    {
        damping_power_0 = _S74 * max(vn_0, 0.0f);
    }
    *dissipated_1 = (damping_power_0 + length(ft_0) * vt_mag_0) * dt_1;
    return normal_1 * float3(_S77)  + ft_0;
}

void comp_add1_0(float thread* sum_0, float thread* err_0, float x_1)
{
    float t_1 = *sum_0 + x_1;
    if((abs(*sum_0)) >= (abs(x_1)))
    {
        *err_0 = *err_0 + (*sum_0 - t_1 + x_1);
    }
    else
    {
        *err_0 = *err_0 + (x_1 - t_1 + *sum_0);
    }
    *sum_0 = t_1;
    return;
}

void ground_contact_0(uint c_4, bool account_0, float3 thread* f_0, float3 thread* t_2, KernelContext_0 thread* kernelContext_12)
{
    ChunkStatic_natural_0 device* _S80 = kernelContext_12->chunks_0+c_4;
    WorldPoint_0 _S81 = chunk_world_0(c_4, kernelContext_12);
    float above_0 = _S81.hi_0.z - kernelContext_12->params_0->ground_hi_0 + (_S81.lo_0.z - kernelContext_12->params_0->ground_lo_0) + _S81.rel_0.z;
    if((above_0 - (float4(_S80->half_0) ).w) > 0.0f)
    {
        return;
    }
    Box_0 _S82 = chunk_box_0(c_4, float3(0.0f) , kernelContext_12);
    float _S83 = kernelContext_12->params_0->ground_modulus_0;
    float _S84 = (float4(_S80->cmat_0) ).x;
    float3 _S85 = float3(0.0f, 0.0f, 1.0f);
    thread Box_0 _S86 = _S82;
    thread Box_0 _S87 = _S82;
    float _S88 = contact_stiffness_0(_S83, &_S86, _S84, &_S87, _S85);
    thread Box_0 _S89 = _S82;
    uint _S90 = sample_count_0(&_S89);
    uint s_1 = 0U;
    uint n_1 = 0U;
    for(;;)
    {
        if(s_1 < _S90)
        {
        }
        else
        {
            break;
        }
        thread Box_0 _S91 = _S82;
        float3 _S92 = sample_point_0(&_S91, s_1, kernelContext_12);
        if((above_0 + _S92.z) < 0.0f)
        {
            n_1 = n_1 + 1U;
        }
        s_1 = s_1 + 1U;
    }
    if(n_1 == 0U)
    {
        return;
    }
    sentinel_mark_0(4U, kernelContext_12);
    thread float3 vc_0;
    thread float3 wc_0;
    chunk_velocity_0(c_4, &vc_0, &wc_0, kernelContext_12);
    thread float4 ledger_1 = float4(*(kernelContext_12->scratch_0+(kernelContext_12->params_0->ledger_base_0 + kernelContext_12->params_0->pair_count_0 + c_4))) ;
    s_1 = 0U;
    for(;;)
    {
        if(s_1 < _S90)
        {
        }
        else
        {
            break;
        }
        thread Box_0 _S93 = _S82;
        float3 _S94 = sample_point_0(&_S93, s_1, kernelContext_12);
        float _S95 = above_0 + _S94.z;
        if(!(_S95 < 0.0f))
        {
            s_1 = s_1 + 1U;
            continue;
        }
        thread float stored_1;
        thread float diss_0;
        float3 _S96 = penalty_force_0(_S88 / float(max(n_1, 5U)), (float4(_S80->center_0) ).w, kernelContext_12->params_0->ground_friction_0, - _S95, _S85, vc_0 + cross(wc_0, _S94), kernelContext_12->params_0->dt_0, n_1, &stored_1, &diss_0, kernelContext_12);
        *f_0 = *f_0 + _S96;
        *t_2 = *t_2 + cross(_S94, _S96);
        thread float _S97 = ledger_1.y;
        thread float _S98 = ledger_1.z;
        comp_add1_0(&_S97, &_S98, diss_0);
        ledger_1.z = _S98;
        ledger_1.y = _S97;
        s_1 = s_1 + 1U;
    }
    if(account_0)
    {
        *(kernelContext_12->scratch_0+(kernelContext_12->params_0->ledger_base_0 + kernelContext_12->params_0->pair_count_0 + c_4)) = packed_float4(ledger_1) ;
    }
    return;
}

void contact_segment_0(uint g_0, KernelContext_0 thread* kernelContext_13)
{
    uint _S99 = 3U * g_0;
    uint _S100 = kernelContext_13->index_0[kernelContext_13->params_0->seg_index_0 + _S99];
    uint begin_0 = kernelContext_13->index_0[kernelContext_13->params_0->seg_index_0 + _S99 + 1U];
    uint _S101 = kernelContext_13->index_0[kernelContext_13->params_0->seg_index_0 + _S99 + 2U];
    float3 _S102 = float3(0.0f) ;
    thread float3 f_1 = _S102;
    thread float3 t_3 = _S102;
    uint e_0 = begin_0;
    for(;;)
    {
        if(e_0 < _S101)
        {
        }
        else
        {
            break;
        }
        uint entry_0 = kernelContext_13->index_0[e_0];
        if(entry_0 == 2147483648U)
        {
            ground_contact_0(_S100, true, &f_1, &t_3, kernelContext_13);
            e_0 = e_0 + 1U;
            continue;
        }
        uint _S103 = 2U * entry_0;
        f_1 = f_1 + (float4(*(kernelContext_13->scratch_0+(kernelContext_13->params_0->slot_base_0 + _S103))) ).xyz;
        t_3 = t_3 + (float4(*(kernelContext_13->scratch_0+(kernelContext_13->params_0->slot_base_0 + _S103 + 1U))) ).xyz;
        e_0 = e_0 + 1U;
    }
    uint _S104 = 2U * g_0;
    *(kernelContext_13->scratch_0+(kernelContext_13->params_0->seg_base_0 + _S104)) = packed_float4(float4(f_1, 0.0f)) ;
    *(kernelContext_13->scratch_0+(kernelContext_13->params_0->seg_base_0 + _S104 + 1U)) = packed_float4(float4(t_3, 0.0f)) ;
    return;
}

float time_since_0(float4 origin_0, uint k_4, float dt_2, KernelContext_0 thread* kernelContext_14)
{
    return kernelContext_14->params_0->t_hi_0 - origin_0.x + (kernelContext_14->params_0->t_lo_0 - origin_0.y) + float(k_4) * dt_2;
}

float table_eval_0(uint offset_0, uint count_0, float tau_0, KernelContext_0 thread* kernelContext_15)
{
    float4 _S105 = float4(*(kernelContext_15->loads_0+offset_0)) ;
    if(tau_0 <= (_S105.x))
    {
        return _S105.y;
    }
    uint i_1 = 1U;
    for(;;)
    {
        if(i_1 < count_0)
        {
        }
        else
        {
            break;
        }
        uint _S106 = offset_0 + i_1;
        float4 _S107 = float4(*(kernelContext_15->loads_0+_S106)) ;
        float _S108 = _S107.x;
        if(tau_0 <= _S108)
        {
            float4 _S109 = float4(*(kernelContext_15->loads_0+(_S106 - 1U))) ;
            float _S110 = _S109.x;
            float _S111 = _S109.y;
            return _S111 + (tau_0 - _S110) / max(_S108 - _S110, 1.00000000317107685e-30f) * (_S107.y - _S111);
        }
        i_1 = i_1 + 1U;
    }
    return (float4(*(kernelContext_15->loads_0+(offset_0 + count_0 - 1U))) ).y;
}

float eval_function_0(uint term_0, uint k_5, float dt_3, float shift_0, KernelContext_0 thread* kernelContext_16)
{
    uint _S112 = 5U * term_0;
    uint4 info_2 = (as_type<uint4>((float4(*(kernelContext_16->loads_0+_S112)) )));
    float4 _S113 = float4(*(kernelContext_16->loads_0+(_S112 + 3U))) ;
    float4 _S114 = float4(*(kernelContext_16->loads_0+(_S112 + 4U))) ;
    uint kind_0 = info_2.z;
    if(kind_0 == 0U)
    {
        return _S113.z;
    }
    float _S115 = time_since_0(_S113, k_5, dt_3, kernelContext_16);
    float tau_1 = _S115 + shift_0;
    float shape_1;
    if(kind_0 == 1U)
    {
        if(tau_1 <= 0.0f)
        {
            shape_1 = 0.0f;
        }
        else
        {
            float _S116 = _S114.x;
            if(tau_1 >= _S116)
            {
                shape_1 = _S114.y;
            }
            else
            {
                shape_1 = _S114.y * tau_1 / _S116;
            }
        }
        return shape_1;
    }
    bool _S117;
    if(kind_0 == 2U)
    {
        if(tau_1 < 0.0f)
        {
            _S117 = true;
        }
        else
        {
            _S117 = tau_1 > (_S114.x);
        }
        if(_S117)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = _S114.y * sin(3.14159274101257324f * tau_1 / _S114.x);
        }
        return shape_1;
    }
    if(kind_0 == 3U)
    {
        float sn_0 = tau_1 / _S114.y;
        if(sn_0 < 0.0f)
        {
            _S117 = true;
        }
        else
        {
            _S117 = sn_0 > 1.0f;
        }
        if(_S117)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = _S114.x * (1.0f - sn_0) * exp(- _S114.z * sn_0);
        }
        return shape_1;
    }
    if(kind_0 == 4U)
    {
        float _S118 = table_eval_0(info_2.w, (as_type<uint>((_S114.x))), tau_1, kernelContext_16);
        return _S118;
    }
    if(kind_0 == 5U)
    {
        (*kernelContext_16->g_seen_0)[1U] = 1U;
        if(tau_1 < 0.0f)
        {
            return 0.0f;
        }
        float sn_1 = tau_1 / _S114.x;
        if(sn_1 < 0.0f)
        {
            _S117 = true;
        }
        else
        {
            _S117 = sn_1 > 1.0f;
        }
        if(_S117)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = (1.0f - sn_1) * exp(- _S114.y * sn_1);
        }
        float clearing_0 = _S113.w;
        float relax_0;
        if(clearing_0 > 0.0f)
        {
            relax_0 = max(1.0f - tau_1 / clearing_0, 0.0f);
        }
        else
        {
            relax_0 = 0.0f;
        }
        float _S119 = _S114.w;
        return (_S119 + (_S114.z - _S119) * relax_0) * shape_1;
    }
    if(kind_0 == 7U)
    {
        (*kernelContext_16->g_seen_0)[2U] = 1U;
        if(tau_1 < 0.0f)
        {
            return 0.0f;
        }
        float _S120 = _S114.y;
        if(tau_1 < _S120)
        {
            return _S114.x;
        }
        float s_2 = tau_1 - _S120;
        float _S121 = _S114.w;
        if(s_2 > _S121)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = _S114.z * sin(3.14159274101257324f * s_2 / _S121);
        }
        return shape_1;
    }
    float _S122 = _S114.x;
    if(_S122 <= 0.0f)
    {
        return 0.0f;
    }
    return clamp(1.0f - tau_1 / _S122, 0.0f, 1.0f);
}

void chunk_loads_0(uint _S123, uint _S124, const Quat_0 thread* _S125, uint _S126, float _S127, bool _S128, float _S129, float3 thread* _S130, float3 thread* _S131, float3 thread* _S132, float3 thread* _S133, KernelContext_0 thread* kernelContext_17)
{
    bool _S134;
    ChunkStatic_natural_0 device* _S135 = kernelContext_17->chunks_0+_S124;
    float3 _S136 = float3(0.0f) ;
    *_S130 = _S136;
    *_S131 = _S136;
    *_S132 = _S136;
    *_S133 = _S136;
    uint4 _S137 = uint4(_S135->load_range_0) ;
    uint term_1 = _S137.x;
    for(;;)
    {
        if(term_1 < (_S137.y))
        {
        }
        else
        {
            break;
        }
        (*kernelContext_17->g_seen_0)[0U] = 1U;
        uint _S138 = 5U * term_1;
        uint _S139 = (as_type<uint4>((float4(*(kernelContext_17->loads_0+_S138)) ))).y;
        bool replacement_0 = _S139 == 2U;
        float4 _S140 = float4(*(kernelContext_17->loads_0+(_S138 + 1U))) ;
        float4 _S141 = float4(*(kernelContext_17->loads_0+(_S138 + 2U))) ;
        float _S142;
        if(replacement_0)
        {
            _S142 = _S129;
        }
        else
        {
            _S142 = 0.0f;
        }
        float _S143 = eval_function_0(term_1, _S126, _S127, _S142, kernelContext_17);
        if(replacement_0)
        {
            float3 _S144 = float3(_S143) ;
            *_S132 = *_S132 + _S140.xyz * _S144;
            *_S133 = *_S133 + _S141.xyz * _S144;
            term_1 = term_1 + 1U;
            continue;
        }
        if(_S139 == 0U)
        {
            _S134 = true;
        }
        else
        {
            _S134 = _S139 == 3U;
        }
        float3 fw_0;
        if(_S134)
        {
            fw_0 = _S140.xyz * float3(_S143) ;
        }
        else
        {
            float3 _S145 = rotate_0(_S125, _S140.xyz);
            fw_0 = _S145 * float3((- _S143 * _S140.w)) ;
        }
        float3 lever_0;
        if(_S139 == 3U)
        {
            lever_0 = _S141.xyz - (float4(*(kernelContext_17->state_0+4U * _S123)) ).xyz;
        }
        else
        {
            lever_0 = _S141.xyz;
        }
        *_S130 = *_S130 + fw_0;
        float3 _S146 = rotate_0(_S125, lever_0);
        *_S131 = *_S131 + cross(_S146, fw_0);
        term_1 = term_1 + 1U;
    }
    if(_S128)
    {
        _S134 = ((uint4(_S135->cinfo_0) ).z) != 0U;
    }
    else
    {
        _S134 = false;
    }
    if(_S134)
    {
        uint4 _S147 = uint4(_S135->cinfo_0) ;
        uint g_1 = _S147.x;
        for(;;)
        {
            if(g_1 < (_S147.y))
            {
            }
            else
            {
                break;
            }
            uint _S148 = 2U * g_1;
            *_S130 = *_S130 + (float4(*(kernelContext_17->scratch_0+(kernelContext_17->params_0->seg_base_0 + _S148))) ).xyz;
            *_S131 = *_S131 + (float4(*(kernelContext_17->scratch_0+(kernelContext_17->params_0->seg_base_0 + _S148 + 1U))) ).xyz;
            g_1 = g_1 + 1U;
        }
    }
    return;
}

void chunk_external_0(uint _S149, uint _S150, const Quat_0 thread* _S151, uint _S152, float _S153, bool _S154, float3 thread* _S155, float3 thread* _S156, KernelContext_0 thread* kernelContext_18)
{
    thread float3 rf_0;
    thread float3 rm_0;
    chunk_loads_0(_S149, _S150, _S151, _S152, _S153, _S154, 0.0f, _S155, _S156, &rf_0, &rm_0, kernelContext_18);
    return;
}

void record_chunk_load_0(uint c_5, float3 f_2, float3 t_4, KernelContext_0 thread* kernelContext_19)
{
    if((kernelContext_19->params_0->solve_mode_0) == 0U)
    {
        return;
    }
    uint _S157 = 2U * c_5;
    *(kernelContext_19->scratch_0+(kernelContext_19->params_0->cload_base_0 + _S157)) = packed_float4(float4(f_2, 0.0f)) ;
    *(kernelContext_19->scratch_0+(kernelContext_19->params_0->cload_base_0 + _S157 + 1U)) = packed_float4(float4(t_4, 0.0f)) ;
    *(kernelContext_19->scratch_0+(kernelContext_19->params_0->cframe_base_0 + _S157)) = packed_float4(float4((float4(*(kernelContext_19->scratch_0+(kernelContext_19->params_0->cframe_base_0 + _S157))) ).xyz + f_2, 0.0f)) ;
    *(kernelContext_19->scratch_0+(kernelContext_19->params_0->cframe_base_0 + _S157 + 1U)) = packed_float4(float4((float4(*(kernelContext_19->scratch_0+(kernelContext_19->params_0->cframe_base_0 + _S157 + 1U))) ).xyz + t_4, 0.0f)) ;
    return;
}

float settled_chunk_load_0(uint c_6, const Quat_0 thread* rot_1, uint k_6, float dt_4, bool contact_0, KernelContext_0 thread* kernelContext_20)
{
    thread float3 f_3;
    thread float3 t_5;
    chunk_external_0(c_6, c_6, rot_1, k_6, dt_4, contact_0, &f_3, &t_5, kernelContext_20);
    record_chunk_load_0(c_6, f_3, t_5, kernelContext_20);
    return length(f_3);
}

void group_sum2_0(uint tid_1, float4 thread* a_3, float4 thread* b_7, KernelContext_0 thread* kernelContext_21)
{
    float4 wa_0 = simd_sum(*a_3);
    float4 wb_0 = simd_sum(*b_7);
    if((tid_1 % 32U) == 0U)
    {
        (*kernelContext_21->g_part_a_0)[tid_1 / 32U] = wa_0;
        (*kernelContext_21->g_part_b_0)[tid_1 / 32U] = wb_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 _S158 = (*kernelContext_21->g_part_a_0)[int(0)];
    float4 _S159 = (*kernelContext_21->g_part_b_0)[int(0)];
    uint w_4 = 1U;
    float4 sa_0 = _S158;
    float4 sb_0 = _S159;
    for(;;)
    {
        if(w_4 < 8U)
        {
        }
        else
        {
            break;
        }
        float4 sa_1 = sa_0 + (*kernelContext_21->g_part_a_0)[w_4];
        float4 sb_1 = sb_0 + (*kernelContext_21->g_part_b_0)[w_4];
        w_4 = w_4 + 1U;
        sa_0 = sa_1;
        sb_0 = sb_1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *a_3 = sa_0;
    *b_7 = sb_0;
    return;
}

void group_sum3_0(uint tid_2, float3 thread* a_4, float3 thread* b_8, KernelContext_0 thread* kernelContext_22)
{
    thread float4 x_2 = float4(*a_4, 0.0f);
    thread float4 y_1 = float4(*b_8, 0.0f);
    group_sum2_0(tid_2, &x_2, &y_1, kernelContext_22);
    *a_4 = x_2.xyz;
    *b_8 = y_1.xyz;
    return;
}

void net_load_0(uint c_7, const Island_0 thread* isl_2, const Rigid_0 thread* rg_1, uint k_7, float dt_5, bool contact_1, float3 thread* f_4, float3 thread* t_6, KernelContext_0 thread* kernelContext_23)
{
    ChunkStatic_natural_0 device* _S160 = kernelContext_23->chunks_0+c_7;
    thread float3 fl_0;
    thread float3 tl_0;
    thread float3 rep_f_0;
    thread float3 rep_m_0;
    chunk_loads_0(c_7, c_7, &rg_1->rot_0, k_7, dt_5, contact_1, 0.0f, &fl_0, &tl_0, &rep_f_0, &rep_m_0, kernelContext_23);
    float4 _S161 = float4(_S160->center_0) ;
    float3 fc_0 = fl_0 + kernelContext_23->params_0->gravity_0.xyz * float3(_S161.w) ;
    float3 _S162 = _S161.xyz;
    float3 _S163 = isl_2->com_0.xyz;
    float3 _S164 = rotate_0(&rg_1->rot_0, _S162 + (float4(*(kernelContext_23->state_0+4U * c_7)) ).xyz - _S163);
    *f_4 = *f_4 + fc_0;
    *t_6 = *t_6 + (cross(_S164, fc_0) + tl_0);
    uint4 _S165 = uint4(_S160->load_range_0) ;
    if((_S165.y) > (_S165.x))
    {
        float3 _S166 = rotate_0(&rg_1->rot_0, rep_f_0);
        *f_4 = *f_4 + _S166;
        float3 _S167 = rotate_0(&rg_1->rot_0, _S162 - _S163);
        float3 _S168 = cross(_S167, _S166);
        float3 _S169 = rotate_0(&rg_1->rot_0, rep_m_0);
        *t_6 = *t_6 + (_S168 + _S169);
    }
    return;
}

float3 inverse_rotate_0(const Quat_0 thread* q_6, float3 v_4)
{
    thread Quat_0 c_8;
    (&c_8)->w_0 = q_6->w_0;
    (&c_8)->x_0 = - q_6->x_0;
    (&c_8)->y_0 = - q_6->y_0;
    (&c_8)->z_0 = - q_6->z_0;
    thread Quat_0 _S170 = c_8;
    float3 _S171 = rotate_0(&_S170, v_4);
    return _S171;
}

float3 rows_mul_0(float4 r0_0, float4 r1_0, float4 r2_0, float3 v_5)
{
    return float3(dot(r0_0.xyz, v_5), dot(r1_0.xyz, v_5), dot(r2_0.xyz, v_5));
}

float3 rotate_1(const Quat_0 thread* q_7, float3 v_6)
{
    float3 qv_1 = float3(q_7->x_0, q_7->y_0, q_7->z_0);
    float3 t_7 = cross(qv_1, v_6) * float3(2.0f) ;
    return v_6 + t_7 * float3(q_7->w_0)  + cross(qv_1, t_7);
}

float3 world_mul_0(const Quat_0 thread* q_8, float4 r0_1, float4 r1_1, float4 r2_1, float3 v_7)
{
    float3 _S172 = inverse_rotate_0(q_8, v_7);
    float3 _S173 = rotate_1(q_8, rows_mul_0(r0_1, r1_1, r2_1, _S172));
    return _S173;
}

void rigid_acceleration_0(const Island_0 thread* isl_3, Rigid_0 thread* rg_2, float3 f_5, float3 t_8)
{
    thread Quat_0 _S174 = rg_2->rot_0;
    float3 _S175 = world_mul_0(&_S174, isl_3->inertia0_0, isl_3->inertia1_0, isl_3->inertia2_0, rg_2->w_1);
    rg_2->a_0 = f_5 / float3(isl_3->com_0.w) ;
    float3 _S176 = t_8 - cross(rg_2->w_1, _S175);
    thread Quat_0 _S177 = rg_2->rot_0;
    float3 _S178 = world_mul_0(&_S177, isl_3->inv0_0, isl_3->inv1_0, isl_3->inv2_0, _S176);
    rg_2->alpha_0 = _S178;
    rg_2->torque_0 = t_8;
    return;
}

float3 to_local_0(uint _S179, float3 _S180, KernelContext_0 thread* kernelContext_24)
{
    BondStatic_natural_0 device* _S181 = kernelContext_24->bonds_0+_S179;
    return float3(dot(_S180, (float4(_S181->t1_0) ).xyz), dot(_S180, (float4(_S181->t2_0) ).xyz), dot(_S180, (float4(_S181->normal_0) ).xyz));
}

bool is_damaged_0(const JointState_0 thread* st_0)
{
    bool _S182;
    if((st_0->damage_0) > 0.0f)
    {
        _S182 = true;
    }
    else
    {
        _S182 = (st_0->crush_1) > 0.0f;
    }
    return _S182;
}

struct Measures_0
{
    float tension_0;
    float shear_0;
    float normal_compression_0;
    float compression_0;
    float compressive_force_0;
};

struct JointResponse_0
{
    float3 force_lin_1;
    float3 force_ang_1;
    JointState_0 state_1;
    float dissipated_2;
    float overshoot_0;
    float stored_2;
    bool disconnected_0;
    float fatigue_before_0;
    Measures_0 measures_0;
};

bool connected_0(const JointState_0 thread* st_1, bool has_rebar_0)
{
    bool _S183;
    if((st_1->damage_0) < 1.0f)
    {
        _S183 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S183 = (st_1->rebar_broken_0) == 0.0f;
        }
        else
        {
            _S183 = false;
        }
    }
    return _S183;
}

Measures_0 stress_measures_0(const JointBond_natural_0 thread* b_9, float3 q_lin_0, float3 q_ang_0)
{
    float4 _S184 = float4(b_9->geom0_0) ;
    float area_2 = _S184.x;
    float _S185 = q_lin_0.z;
    float axial_0 = (metal::fast::divide((_S185), (area_2)));
    float4 _S186 = float4(b_9->geom1_0) ;
    float _S187 = (metal::fast::divide((abs(q_ang_0.x)), (_S186.x)));
    float _S188 = (metal::fast::divide((abs(q_ang_0.y)), (_S186.y)));
    float bending_0 = _S187 + _S188;
    float _S189 = q_lin_0.x;
    float _S190 = q_lin_0.y;
    float _S191 = (metal::fast::sqrt((_S189 * _S189 + _S190 * _S190)));
    float _S192 = (metal::fast::divide((_S191), (area_2)));
    float _S193 = (metal::fast::divide((abs(q_ang_0.z)), (_S184.w)));
    float shear_1 = _S192 + _S193;
    thread Measures_0 m_1;
    (&m_1)->tension_0 = axial_0 + bending_0;
    (&m_1)->shear_0 = shear_1;
    float _S194 = - axial_0;
    (&m_1)->normal_compression_0 = max(_S194, 0.0f);
    (&m_1)->compression_0 = _S194 + bending_0;
    (&m_1)->compressive_force_0 = max(- _S185, 0.0f);
    return m_1;
}

float expm1_accurate_0(float x_3)
{
    if((abs(x_3)) < 0.00100000004749745f)
    {
        return x_3 * (1.0f + x_3 * (0.5f + x_3 * 0.1666666716337204f));
    }
    return exp(x_3) - 1.0f;
}

float dif_factor_0(const JointMaterial_0 constant* mat_1, float strain_rate_1)
{
    float r_1 = abs(strain_rate_1);
    float4 _S195 = mat_1->dif_0;
    float ref_0 = mat_1->dif_0.x;
    if(r_1 <= ref_0)
    {
        return 1.0f;
    }
    float _S196 = _S195.z;
    float f_6;
    if(r_1 <= _S196)
    {
        float _S197 = (metal::fast::divide((r_1), (ref_0)));
        float _S198 = (metal::fast::pow((_S197), (_S195.y)));
        f_6 = _S198;
    }
    else
    {
        float _S199 = (metal::fast::divide((_S196), (ref_0)));
        float _S200 = (metal::fast::pow((_S199), (_S195.y)));
        float _S201 = (metal::fast::divide((r_1), (_S196)));
        float _S202 = (metal::fast::pow((_S201), (_S195.w)));
        f_6 = _S200 * _S202;
    }
    return clamp(f_6, 1.0f, mat_1->misc_0.x);
}

float fatigue_factor_0(const JointMaterial_0 constant* mat_2, float life_1)
{
    if(((mat_2->kind_flags_0.y) & 64U) == 0U)
    {
        return 1.0f;
    }
    float _S203 = clamp(life_1, 0.0f, 1.0f);
    float _S204 = (metal::fast::divide((1.0f), (mat_2->misc_0.y - 2.0f)));
    float _S205 = (metal::fast::pow((_S203), (_S204)));
    return _S205;
}

float infinity_0()
{
    return (as_type<float>((2139095040U)));
}

float4 failure_indices_0(const JointMaterial_0 constant* mat_3, const JointBond_natural_0 thread* b_10, const Measures_0 thread* m_2, float multiplier_0)
{
    float fc_1 = mat_3->strength_0.y * multiplier_0;
    float _S206 = min(mat_3->strength_0.z * multiplier_0 + mat_3->strength_0.w * m_2->normal_compression_0, mat_3->energy_1.x * multiplier_0);
    thread float4 idx_0;
    float _S207 = (metal::fast::divide((m_2->tension_0), (mat_3->strength_0.x * multiplier_0)));
    idx_0.x = max(_S207, 0.0f);
    float _S208;
    if(_S206 > 0.0f)
    {
        float _S209 = (metal::fast::divide((m_2->shear_0), (_S206)));
        _S208 = _S209;
    }
    else
    {
        _S208 = infinity_0();
    }
    idx_0.y = _S208;
    float _S210 = (metal::fast::divide((m_2->compression_0), (fc_1)));
    idx_0.z = max(_S210, 0.0f);
    float _S211 = (float4(b_10->stiff1_0) ).y;
    if(_S211 > 0.0f)
    {
        float _S212 = (metal::fast::divide((m_2->compressive_force_0), (_S211)));
        _S208 = _S212;
    }
    else
    {
        _S208 = 0.0f;
    }
    idx_0.w = _S208;
    return idx_0;
}

float sq_0(float x_4)
{
    return x_4 * x_4;
}

void compressed_region_0(float w0_0, float w1_0, float dz_0, float ax_0, float ay_0, array<float, int(6)> thread* region_0)
{
    uint count_1;
    float h0_0 = 0.5f * w0_0;
    float h1_0 = 0.5f * w1_0;
    float _S213 = - h0_0;
    float _S214 = - h1_0;
    array<float2, int(4)> _S215 = { { float2(_S213, _S214), float2(h0_0, _S214), float2(h0_0, h1_0), float2(_S213, h1_0) } };
    thread array<float2, int(8)> poly_0;
    uint i_2 = 0U;
    uint count_2 = 0U;
    for(;;)
    {
        if(i_2 < 4U)
        {
        }
        else
        {
            break;
        }
        uint _S216 = i_2;
        uint _S217 = i_2 + 1U;
        uint _S218 = _S217 % 4U;
        float _S219 = _S215[i_2].y;
        float _S220 = _S215[i_2].x;
        float fp_0 = dz_0 + ax_0 * _S219 - ay_0 * _S220;
        float _S221 = _S215[_S218].y;
        float _S222 = _S215[_S218].x;
        float fq_0 = dz_0 + ax_0 * _S221 - ay_0 * _S222;
        bool _S223 = fp_0 < 0.0f;
        if(_S223)
        {
            uint _S224 = count_2 + 1U;
            poly_0[count_2] = _S215[_S216];
            count_1 = _S224;
        }
        else
        {
            count_1 = count_2;
        }
        if(_S223 != (fq_0 < 0.0f))
        {
            float t_9 = fp_0 / (fp_0 - fq_0);
            uint _S225 = count_1 + 1U;
            poly_0[count_1] = float2(_S220 + t_9 * (_S222 - _S220), _S219 + t_9 * (_S221 - _S219));
            count_2 = _S225;
        }
        else
        {
            count_2 = count_1;
        }
        i_2 = _S217;
    }
    count_1 = 0U;
    for(;;)
    {
        if(count_1 < 6U)
        {
        }
        else
        {
            break;
        }
        (*region_0)[count_1] = 0.0f;
        count_1 = count_1 + 1U;
    }
    if(count_2 < 3U)
    {
        return;
    }
    float2 o_0 = poly_0[int(0)];
    i_2 = 0U;
    float a_5 = 0.0f;
    float sx_0 = 0.0f;
    float sy_0 = 0.0f;
    float ixx_0 = 0.0f;
    float iyy_0 = 0.0f;
    float ixy_0 = 0.0f;
    for(;;)
    {
        if(i_2 < count_2)
        {
        }
        else
        {
            break;
        }
        float _S226 = o_0.x;
        float x0_0 = poly_0[i_2].x - _S226;
        float _S227 = o_0.y;
        float y0_0 = poly_0[i_2].y - _S227;
        uint _S228 = i_2 + 1U;
        uint _S229 = _S228 % count_2;
        float x1_0 = poly_0[_S229].x - _S226;
        float y1_0 = poly_0[_S229].y - _S227;
        float _S230 = x0_0 * y1_0;
        float _S231 = x1_0 * y0_0;
        float cr_0 = _S230 - _S231;
        float a_6 = a_5 + cr_0 / 2.0f;
        float sx_1 = sx_0 + (x0_0 + x1_0) * cr_0 / 6.0f;
        float sy_1 = sy_0 + (y0_0 + y1_0) * cr_0 / 6.0f;
        float ixx_1 = ixx_0 + (x0_0 * x0_0 + x0_0 * x1_0 + x1_0 * x1_0) * cr_0 / 12.0f;
        float iyy_1 = iyy_0 + (y0_0 * y0_0 + y0_0 * y1_0 + y1_0 * y1_0) * cr_0 / 12.0f;
        float ixy_1 = ixy_0 + (_S230 + 2.0f * x0_0 * y0_0 + 2.0f * x1_0 * y1_0 + _S231) * cr_0 / 24.0f;
        i_2 = _S228;
        a_5 = a_6;
        sx_0 = sx_1;
        sy_0 = sy_1;
        ixx_0 = ixx_1;
        iyy_0 = iyy_1;
        ixy_0 = ixy_1;
    }
    if(a_5 <= 0.0f)
    {
        return;
    }
    float cx_0 = sx_0 / a_5;
    float cy_0 = sy_0 / a_5;
    (*region_0)[int(0)] = a_5;
    (*region_0)[int(1)] = o_0.x + cx_0;
    (*region_0)[int(2)] = o_0.y + cy_0;
    float _S232 = a_5 * cx_0;
    (*region_0)[int(3)] = ixx_0 - _S232 * cx_0;
    (*region_0)[int(4)] = iyy_0 - a_5 * cy_0 * cy_0;
    (*region_0)[int(5)] = ixy_0 - _S232 * cy_0;
    return;
}

__attribute__((noinline))
float4 no_tension_patch_0(float kn_0, float w0_1, float w1_1, float dz_1, float ax_1, float ay_1)
{
    thread array<float, int(6)> r_2;
    compressed_region_0(w0_1, w1_1, dz_1, ax_1, ay_1, &r_2);
    float a_7 = r_2[int(0)];
    if((r_2[int(0)]) == 0.0f)
    {
        return float4(0.0f) ;
    }
    float k_8 = kn_0 / (w0_1 * w1_1);
    float fc_2 = dz_1 + ax_1 * r_2[int(2)] - ay_1 * r_2[int(1)];
    float _S233 = a_7 * fc_2;
    float _S234 = - ay_1;
    return float4(k_8 * a_7 * fc_2, k_8 * (_S233 * r_2[int(2)] + (_S234 * r_2[int(5)] + ax_1 * r_2[int(4)])), - k_8 * (_S233 * r_2[int(1)] + (_S234 * r_2[int(3)] + ax_1 * r_2[int(5)])), 0.5f * k_8 * (_S233 * fc_2 + ay_1 * ay_1 * r_2[int(3)] + ax_1 * ax_1 * r_2[int(4)] - 2.0f * ax_1 * ay_1 * r_2[int(5)]));
}

constant array<float, int(6)> SPRING_AT_0 = { { -0.4166666567325592f, -0.25f, -0.0833333358168602f, 0.0833333358168602f, 0.25f, 0.4166666567325592f } };
float4 contact_normal_0(const JointMaterial_0 constant* mat_4, const JointBond_natural_0 thread* b_11, float crush_2, float3 d_lin_0, float3 d_ang_0)
{
    float kn_1 = (float4(b_11->stiff0_0) ).x;
    float4 _S235 = float4(b_11->geom0_0) ;
    float w0_2 = _S235.y;
    float w1_2 = _S235.z;
    float nc_sum_0;
    float m1_0;
    float m2_0;
    float energy_2;
    if(((mat_4->kind_flags_0.y) & 4U) != 0U)
    {
        float4 p_0 = no_tension_patch_0(kn_1 * (1.0f - crush_2), w0_2, w1_2, d_lin_0.z, d_ang_0.x, d_ang_0.y);
        float _S236 = p_0.y;
        float _S237 = p_0.z;
        float _S238 = p_0.w;
        nc_sum_0 = p_0.x;
        m1_0 = _S236;
        m2_0 = _S237;
        energy_2 = _S238;
    }
    else
    {
        float ki_0 = kn_1 * (1.0f - crush_2) / 36.0f;
        float _S239 = d_ang_0.x;
        float _S240 = d_ang_0.y;
        float spread_0 = abs(_S239) * 0.4166666567325592f * w1_2 + abs(_S240) * 0.4166666567325592f * w0_2;
        float _S241 = d_lin_0.z;
        float slack_0 = 9.99999997475242708e-07f * (abs(_S241) + spread_0);
        if((_S241 - spread_0) > slack_0)
        {
            nc_sum_0 = 0.0f;
            m1_0 = 0.0f;
            m2_0 = 0.0f;
            energy_2 = 0.0f;
        }
        else
        {
            if((_S241 + spread_0) < (- slack_0))
            {
                float i1_0 = 2.91666650772094727f * w0_2 * w0_2;
                float i2_0 = 2.91666650772094727f * w1_2 * w1_2;
                float _S242 = ki_0 * _S239 * i2_0;
                float _S243 = ki_0 * _S240 * i1_0;
                float _S244 = 0.5f * ki_0 * (36.0f * _S241 * _S241 + _S239 * _S239 * i2_0 + _S240 * _S240 * i1_0);
                nc_sum_0 = ki_0 * 36.0f * _S241;
                m1_0 = _S242;
                m2_0 = _S243;
                energy_2 = _S244;
            }
            else
            {
                uint i_3 = 0U;
                float nc_sum_1 = 0.0f;
                float m1_1 = 0.0f;
                float m2_1 = 0.0f;
                float energy_3 = 0.0f;
                for(;;)
                {
                    if(i_3 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float _S245 = SPRING_AT_0[i_3] * w0_2;
                    uint j_0 = 0U;
                    nc_sum_0 = nc_sum_1;
                    m1_0 = m1_1;
                    m2_0 = m2_1;
                    energy_2 = energy_3;
                    for(;;)
                    {
                        if(j_0 < 6U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        float s2_0 = SPRING_AT_0[j_0] * w1_2;
                        float di_0 = _S241 + _S239 * s2_0 - _S240 * _S245;
                        if(di_0 < 0.0f)
                        {
                            float f_7 = ki_0 * di_0;
                            float m1_2 = m1_0 + f_7 * s2_0;
                            float m2_2 = m2_0 - f_7 * _S245;
                            float energy_4 = energy_2 + 0.5f * ki_0 * di_0 * di_0;
                            nc_sum_0 = nc_sum_0 + f_7;
                            m1_0 = m1_2;
                            m2_0 = m2_2;
                            energy_2 = energy_4;
                        }
                        j_0 = j_0 + 1U;
                    }
                    i_3 = i_3 + 1U;
                    nc_sum_1 = nc_sum_0;
                    m1_1 = m1_0;
                    m2_1 = m2_0;
                    energy_3 = energy_2;
                }
                nc_sum_0 = nc_sum_1;
                m1_0 = m1_1;
                m2_0 = m2_1;
                energy_2 = energy_3;
            }
        }
    }
    return float4(nc_sum_0, m1_0, m2_0, energy_2);
}

float damage_law_0(uint kind_1, float kappa_1, float r_3)
{
    if(kappa_1 <= 1.0f)
    {
        return 0.0f;
    }
    if(kind_1 == 0U)
    {
        if(r_3 <= 1.0f)
        {
            return 1.0f;
        }
        float _S246 = (metal::fast::divide((r_3 * (kappa_1 - 1.0f)), (kappa_1 * (r_3 - 1.0f))));
        return min(_S246, 1.0f);
    }
    if(kappa_1 >= (0.5f * (r_3 + 1.0f)))
    {
        return 1.0f;
    }
    float _S247 = (metal::fast::divide((1.0f), (kappa_1)));
    return 1.0f - _S247;
}

__attribute__((noinline))
float2 damage_increment_0(uint kind_2, float kappa_old_0, float lambda_0, float r_4, float d_old_0, float psi_0)
{
    float _S248 = damage_law_0(kind_2, lambda_0, r_4);
    float _S249 = max(_S248, d_old_0);
    bool _S250;
    if(_S249 <= d_old_0)
    {
        _S250 = true;
    }
    else
    {
        _S250 = d_old_0 >= 1.0f;
    }
    if(_S250)
    {
        return float2(d_old_0, 0.0f);
    }
    float u0_0 = (metal::fast::divide((psi_0), (lambda_0 * lambda_0)));
    float _S251 = max(kappa_old_0, 1.0f);
    if(kind_2 == 0U)
    {
        if(r_4 > 1.0f)
        {
            float _S252 = (metal::fast::divide((u0_0 * r_4), (r_4 - 1.0f)));
            return float2(_S249, _S252 * max(min(lambda_0, r_4) - min(_S251, r_4), 0.0f));
        }
        return float2(_S249, (1.0f - d_old_0) * psi_0);
    }
    float ku_0 = 0.5f * (r_4 + 1.0f);
    float plateau_0 = u0_0 * max(min(lambda_0, ku_0) - min(_S251, ku_0), 0.0f);
    float snap_0;
    if(_S249 >= 1.0f)
    {
        snap_0 = u0_0 * ku_0;
    }
    else
    {
        snap_0 = 0.0f;
    }
    return float2(_S249, plateau_0 + snap_0);
}

struct Contact_0
{
    float3 q_lin_1;
    float3 q_ang_1;
    float energy_5;
    float diss_1;
    float3 plastic_0;
};

float signum_0(float x_5)
{
    float _S253;
    if(((as_type<uint>((x_5))) & 2147483648U) != 0U)
    {
        _S253 = -1.0f;
    }
    else
    {
        _S253 = 1.0f;
    }
    return _S253;
}

float2 return_map_0(float k_9, float total_0, float plastic_1, float cap_0)
{
    float trial_0 = k_9 * (total_0 - plastic_1);
    if((abs(trial_0)) <= cap_0)
    {
        return float2(trial_0, 0.0f);
    }
    float f_8 = cap_0 * signum_0(trial_0);
    float _S254 = (metal::fast::divide((trial_0 - f_8), (k_9)));
    return float2(f_8, _S254);
}

__attribute__((noinline))
Contact_0 contact_part_0(const JointMaterial_0 constant* mat_5, const JointBond_natural_0 thread* b_12, float crush_3, float3 plastic_2, float3 d_lin_1, float3 d_ang_1)
{
    thread Contact_0 c_9;
    float3 _S255 = float3(0.0f) ;
    (&c_9)->q_lin_1 = _S255;
    (&c_9)->q_ang_1 = _S255;
    (&c_9)->energy_5 = 0.0f;
    (&c_9)->diss_1 = 0.0f;
    (&c_9)->plastic_0 = plastic_2;
    if(((mat_5->kind_flags_0.y) & 2U) == 0U)
    {
        return c_9;
    }
    float ks_0 = (float4(b_12->stiff0_0) ).y;
    float kt_0 = (float4(b_12->stiff1_0) ).x;
    float4 _S256 = contact_normal_0(mat_5, b_12, crush_3, d_lin_1, d_ang_1);
    float nc_sum_2 = _S256.x;
    float m1_3 = _S256.y;
    float m2_3 = _S256.z;
    float energy_6 = _S256.w;
    float nc_0 = - nc_sum_2;
    thread float3 p_1 = plastic_2;
    (&c_9)->q_lin_1 = float3(0.0f, 0.0f, nc_sum_2);
    (&c_9)->q_ang_1 = float3(m1_3, m2_3, 0.0f);
    float slide_cap_0 = mat_5->strength_0.w * nc_0;
    float _S257 = ks_0 * (d_lin_1.x - plastic_2.x);
    float _S258 = ks_0 * (d_lin_1.y - plastic_2.y);
    float tn_0 = (metal::fast::sqrt((_S257 * _S257 + _S258 * _S258)));
    bool _S259;
    if(tn_0 > slide_cap_0)
    {
        _S259 = tn_0 > 0.0f;
    }
    else
    {
        _S259 = false;
    }
    float diss_2;
    if(_S259)
    {
        float _S260 = (metal::fast::divide((_S257), (tn_0)));
        float _S261 = (metal::fast::divide((_S258), (tn_0)));
        float dslip_0 = (metal::fast::divide((tn_0 - slide_cap_0), (ks_0)));
        p_1.x = p_1.x + _S260 * dslip_0;
        p_1.y = p_1.y + _S261 * dslip_0;
        (&c_9)->q_lin_1.x = _S260 * slide_cap_0;
        (&c_9)->q_lin_1.y = _S261 * slide_cap_0;
        diss_2 = slide_cap_0 * dslip_0;
    }
    else
    {
        (&c_9)->q_lin_1.x = _S257;
        (&c_9)->q_lin_1.y = _S258;
        diss_2 = 0.0f;
    }
    float2 tq_0 = return_map_0(kt_0, d_ang_1.z, p_1.z, slide_cap_0 * (float4(b_12->geom1_0) ).z);
    float _S262 = tq_0.x;
    float _S263 = tq_0.y;
    float diss_3 = diss_2 + abs(_S262) * abs(_S263);
    p_1.z = p_1.z + _S263;
    (&c_9)->q_ang_1.z = _S262;
    float _S264 = (metal::fast::divide((sq_0((&c_9)->q_lin_1.x)), (ks_0)));
    float _S265 = (metal::fast::divide((sq_0((&c_9)->q_lin_1.y)), (ks_0)));
    float _S266 = _S264 + _S265;
    float _S267 = (metal::fast::divide((sq_0(_S262)), (kt_0)));
    (&c_9)->energy_5 = energy_6 + 0.5f * (_S266 + _S267);
    (&c_9)->diss_1 = diss_3;
    (&c_9)->plastic_0 = p_1;
    return c_9;
}

float3 contact_offsets_0(const JointMaterial_0 constant* mat_6, const JointBond_natural_0 thread* b_13, float crush_4, float3 plastic_3, float3 d_lin_2, float3 d_ang_2)
{
    uint _S268 = mat_6->kind_flags_0.y;
    if((_S268 & 2U) == 0U)
    {
        return plastic_3;
    }
    float4 _S269 = float4(b_13->stiff0_0) ;
    float kn_2 = _S269.x;
    float ks_1 = _S269.y;
    float kt_1 = (float4(b_13->stiff1_0) ).x;
    float4 _S270 = float4(b_13->geom0_0) ;
    float w0_3 = _S270.y;
    float w1_3 = _S270.z;
    float nc_sum_3;
    if((_S268 & 4U) != 0U)
    {
        float4 _S271 = no_tension_patch_0(kn_2 * (1.0f - crush_4), w0_3, w1_3, d_lin_2.z, d_ang_2.x, d_ang_2.y);
        nc_sum_3 = _S271.x;
    }
    else
    {
        float ki_1 = kn_2 * (1.0f - crush_4) / 36.0f;
        float _S272 = d_ang_2.x;
        float _S273 = d_ang_2.y;
        float spread_1 = abs(_S272) * 0.4166666567325592f * w1_3 + abs(_S273) * 0.4166666567325592f * w0_3;
        float _S274 = d_lin_2.z;
        float slack_1 = 9.99999997475242708e-07f * (abs(_S274) + spread_1);
        if((_S274 - spread_1) > slack_1)
        {
            nc_sum_3 = 0.0f;
        }
        else
        {
            if((_S274 + spread_1) < (- slack_1))
            {
                nc_sum_3 = ki_1 * 36.0f * _S274;
            }
            else
            {
                uint i_4 = 0U;
                float nc_sum_4 = 0.0f;
                for(;;)
                {
                    if(i_4 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float _S275 = SPRING_AT_0[i_4] * w0_3;
                    uint j_1 = 0U;
                    nc_sum_3 = nc_sum_4;
                    for(;;)
                    {
                        if(j_1 < 6U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        float di_1 = _S274 + _S272 * (SPRING_AT_0[j_1] * w1_3) - _S273 * _S275;
                        if(di_1 < 0.0f)
                        {
                            nc_sum_3 = nc_sum_3 + ki_1 * di_1;
                        }
                        j_1 = j_1 + 1U;
                    }
                    i_4 = i_4 + 1U;
                    nc_sum_4 = nc_sum_3;
                }
                nc_sum_3 = nc_sum_4;
            }
        }
    }
    float nc_1 = - nc_sum_3;
    thread float3 p_2 = plastic_3;
    float slide_cap_1 = mat_6->strength_0.w * nc_1;
    float _S276 = ks_1 * (d_lin_2.x - plastic_3.x);
    float _S277 = ks_1 * (d_lin_2.y - plastic_3.y);
    float tn_1 = (metal::fast::sqrt((_S276 * _S276 + _S277 * _S277)));
    bool _S278;
    if(tn_1 > slide_cap_1)
    {
        _S278 = tn_1 > 0.0f;
    }
    else
    {
        _S278 = false;
    }
    if(_S278)
    {
        float _S279 = (metal::fast::divide((_S276), (tn_1)));
        float _S280 = (metal::fast::divide((_S277), (tn_1)));
        float dslip_1 = (metal::fast::divide((tn_1 - slide_cap_1), (ks_1)));
        p_2.x = p_2.x + _S279 * dslip_1;
        p_2.y = p_2.y + _S280 * dslip_1;
    }
    float2 tq_1 = return_map_0(kt_1, d_ang_2.z, p_2.z, slide_cap_1 * (float4(b_13->geom1_0) ).z);
    p_2.z = p_2.z + tq_1.y;
    return p_2;
}

float life_rate_0(const JointMaterial_0 constant* mat_7, float s_3)
{
    if(s_3 <= 0.0f)
    {
        return 0.0f;
    }
    float _S281 = mat_7->misc_0.y;
    float _S282 = _S281 + 1.0f;
    float _S283 = (metal::fast::pow((s_3), (_S281)));
    float _S284 = (metal::fast::divide((_S282 * _S283), (mat_7->misc_0.z)));
    return _S284;
}

JointResponse_0 joint_evaluate_0(const JointMaterial_0 constant* mat_8, const JointBond_natural_0 thread* b_14, const JointState_0 thread* state_2, float3 d_lin_3, float3 d_ang_3, float dt_6, bool fracture_1)
{
    float4 _S285 = float4(b_14->stiff0_0) ;
    float kn_3 = _S285.x;
    float ks_2 = _S285.y;
    float kb1_0 = _S285.z;
    float kb2_0 = _S285.w;
    float4 _S286 = float4(b_14->stiff1_0) ;
    float kt_2 = _S286.x;
    bool has_rebar_1 = (_S286.w) != 0.0f;
    uint kind_3 = mat_8->kind_flags_0.x;
    uint flags_0 = mat_8->kind_flags_0.y;
    bool softening_0 = (flags_0 & 1U) != 0U;
    thread JointState_0 st_2 = *state_2;
    bool _S287 = connected_0(state_2, has_rebar_1);
    float3 qe_lin_0 = d_lin_3 * float3(ks_2, ks_2, kn_3);
    float3 qe_ang_0 = d_ang_3 * float3(kb1_0, kb2_0, kt_2);
    Measures_0 _S288 = stress_measures_0(b_14, qe_lin_0, qe_ang_0);
    float _S289 = max(max(_S288.tension_0, _S288.shear_0), _S288.compression_0);
    bool _S290 = dt_6 > 0.0f;
    float dif_1;
    if(_S290)
    {
        float _S291 = (metal::fast::divide((_S289 - (&st_2)->governing_stress_0), (dt_6)));
        float raw_0 = (metal::fast::divide((max(_S291, 0.0f)), (mat_8->misc_0.w)));
        float tau_2 = _S286.z;
        if((flags_0 & 16U) != 0U)
        {
            float _S292 = (metal::fast::divide((dt_6), (tau_2)));
            dif_1 = - expm1_accurate_0(- _S292);
        }
        else
        {
            float _S293 = (metal::fast::divide((dt_6), (tau_2)));
            dif_1 = min(_S293, 1.0f);
        }
        (&st_2)->strain_rate_0 = (&st_2)->strain_rate_0 + (raw_0 - (&st_2)->strain_rate_0) * dif_1;
        (&st_2)->governing_stress_0 = _S289;
    }
    if((flags_0 & 32U) != 0U)
    {
        float _S294 = dif_factor_0(mat_8, (&st_2)->strain_rate_0);
        dif_1 = _S294;
    }
    else
    {
        dif_1 = 1.0f;
    }
    float weibull_0 = (float4(b_14->geom1_0) ).w;
    float _S295 = fatigue_factor_0(mat_8, (&st_2)->life_0);
    float multiplier_1 = weibull_0 * dif_1 * _S295;
    thread Measures_0 _S296 = _S288;
    float4 _S297 = failure_indices_0(mat_8, b_14, &_S296, multiplier_1);
    float _S298 = _S297.x;
    float _S299 = _S297.y;
    (&st_2)->utilization_0 = max(max(_S298, _S299), max(_S297.z, _S297.w));
    float _S300 = d_lin_3.x;
    float _S301 = d_lin_3.y;
    float _S302 = d_ang_3.z;
    float _S303 = ks_2 * (sq_0(_S300) + sq_0(_S301)) + kb1_0 * sq_0(d_ang_3.x) + kb2_0 * sq_0(d_ang_3.y) + kt_2 * sq_0(_S302);
    float _S304 = d_lin_3.z;
    bool _S305 = _S304 > 0.0f;
    if(_S305)
    {
        dif_1 = kn_3 * sq_0(_S304);
    }
    else
    {
        dif_1 = 0.0f;
    }
    float psi_ts_0 = 0.5f * (_S303 + dif_1);
    float psi_c_0;
    if(_S304 < 0.0f)
    {
        psi_c_0 = 0.5f * kn_3 * sq_0(_S304);
    }
    else
    {
        psi_c_0 = 0.0f;
    }
    float3 plastic_4 = float3((&st_2)->plastic_x_0, (&st_2)->plastic_y_0, (&st_2)->plastic_t_0);
    bool changed_0;
    if((flags_0 & 2U) != 0U)
    {
        changed_0 = (state_2->damage_0) != 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    float patch_0;
    if(changed_0)
    {
        float4 _S306 = contact_normal_0(mat_8, b_14, state_2->crush_1, d_lin_3, d_ang_3);
        patch_0 = _S306.w + 0.5f * (ks_2 * (sq_0(_S300 - state_2->plastic_x_0) + sq_0(_S301 - state_2->plastic_y_0)) + kt_2 * sq_0(_S302 - state_2->plastic_t_0));
    }
    else
    {
        patch_0 = 0.0f;
    }
    if(has_rebar_1)
    {
        changed_0 = (state_2->rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    float rebar_0;
    if(changed_0)
    {
        float4 _S307 = float4(b_14->rebar0_0) ;
        rebar_0 = 0.5f * (_S307.x * sq_0(_S304 - state_2->rebar_plastic_0) + _S307.y * (sq_0(_S300 - state_2->rebar_slip0_0) + sq_0(_S301 - state_2->rebar_slip1_0)));
    }
    else
    {
        rebar_0 = 0.0f;
    }
    float _S308 = state_2->damage_0;
    float _S309 = 1.0f - state_2->damage_0;
    float _S310 = state_2->crush_1;
    float frozen_0 = _S309 * (psi_ts_0 + (1.0f - state_2->crush_1) * psi_c_0) + state_2->damage_0 * patch_0 + rebar_0;
    float diss_contact_0;
    float psi_contact_0;
    float intact_normal_0;
    float dissipated_3;
    float3 qc_lin_0;
    if(fracture_1)
    {
        bool _S311 = _S298 >= _S299;
        if(_S311)
        {
            diss_contact_0 = _S298;
        }
        else
        {
            diss_contact_0 = _S299;
        }
        uint mode_ts_0;
        if(_S311)
        {
            mode_ts_0 = 1U;
        }
        else
        {
            mode_ts_0 = 2U;
        }
        if(diss_contact_0 > ((&st_2)->kappa_0))
        {
            changed_0 = diss_contact_0 > 1.0f;
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            changed_0 = psi_ts_0 > 0.0f;
        }
        else
        {
            changed_0 = false;
        }
        uint mode_c_0;
        if(changed_0)
        {
            if(mode_ts_0 == 1U)
            {
                psi_contact_0 = mat_8->energy_1.y;
            }
            else
            {
                psi_contact_0 = mat_8->energy_1.z;
            }
            if(softening_0)
            {
                float _S312 = (metal::fast::divide((psi_contact_0 * (float4(b_14->geom0_0) ).x * diss_contact_0 * diss_contact_0), (psi_ts_0)));
                intact_normal_0 = _S312;
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            (&st_2)->ductility_0 = intact_normal_0;
            if(softening_0)
            {
                mode_c_0 = kind_3;
            }
            else
            {
                mode_c_0 = 0U;
            }
            float2 inc_0 = damage_increment_0(mode_c_0, (&st_2)->kappa_0, diss_contact_0, intact_normal_0, (&st_2)->damage_0, psi_ts_0);
            float _S313 = inc_0.x;
            if(_S313 > ((&st_2)->damage_0))
            {
                Contact_0 _S314 = contact_part_0(mat_8, b_14, (&st_2)->crush_1, plastic_4, d_lin_3, d_ang_3);
                float _S315 = max(inc_0.y - max(_S314.energy_5 - (1.0f - (&st_2)->crush_1) * psi_c_0, 0.0f) * (_S313 - (&st_2)->damage_0), 0.0f);
                (&st_2)->damage_0 = _S313;
                (&st_2)->mode_0 = mode_ts_0;
                dissipated_3 = _S315;
            }
            else
            {
                dissipated_3 = 0.0f;
            }
        }
        else
        {
            dissipated_3 = 0.0f;
        }
        (&st_2)->kappa_0 = max((&st_2)->kappa_0, diss_contact_0);
        if(_S308 > 0.0f)
        {
            Contact_0 _S316 = contact_part_0(mat_8, b_14, _S310, float3(state_2->plastic_x_0, state_2->plastic_y_0, state_2->plastic_t_0), d_lin_3, d_ang_3);
            qc_lin_0 = qe_ang_0 * float3(_S309)  + _S316.q_ang_1 * float3(_S308) ;
        }
        else
        {
            qc_lin_0 = qe_ang_0;
        }
        Measures_0 _S317 = stress_measures_0(b_14, float3(0.0f, 0.0f, min(qe_lin_0.z, 0.0f)), qc_lin_0);
        thread Measures_0 _S318 = _S317;
        float4 _S319 = failure_indices_0(mat_8, b_14, &_S318, multiplier_1);
        float _S320 = _S319.z;
        float _S321 = _S319.w;
        bool _S322 = _S320 >= _S321;
        if(_S322)
        {
            psi_contact_0 = _S320;
        }
        else
        {
            psi_contact_0 = _S321;
        }
        if(_S322)
        {
            mode_c_0 = 3U;
        }
        else
        {
            mode_c_0 = 4U;
        }
        if(psi_contact_0 > ((&st_2)->kappa_c_0))
        {
            changed_0 = psi_contact_0 > 1.0f;
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            changed_0 = psi_c_0 > 0.0f;
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            if(softening_0)
            {
                float _S323 = (metal::fast::divide((mat_8->energy_1.w * (float4(b_14->geom0_0) ).x * psi_contact_0 * psi_contact_0), (psi_c_0)));
                intact_normal_0 = _S323;
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            (&st_2)->ductility_c_0 = intact_normal_0;
            uint law_1;
            if(!softening_0)
            {
                law_1 = 0U;
            }
            else
            {
                if(mode_c_0 == 4U)
                {
                    mode_ts_0 = 1U;
                }
                else
                {
                    mode_ts_0 = kind_3;
                }
                law_1 = mode_ts_0;
            }
            float2 inc_1 = damage_increment_0(law_1, (&st_2)->kappa_c_0, psi_contact_0, intact_normal_0, (&st_2)->crush_1, psi_c_0);
            float _S324 = inc_1.x;
            if(_S324 > ((&st_2)->crush_1))
            {
                float dissipated_4 = dissipated_3 + inc_1.y;
                (&st_2)->crush_1 = _S324;
                (&st_2)->mode_0 = mode_c_0;
                if(_S324 >= 1.0f)
                {
                    changed_0 = ((&st_2)->damage_0) < 1.0f;
                }
                else
                {
                    changed_0 = false;
                }
                if(changed_0)
                {
                    float dissipated_5 = dissipated_4 + psi_ts_0 * (1.0f - (&st_2)->damage_0);
                    (&st_2)->damage_0 = 1.0f;
                    dissipated_3 = dissipated_5;
                }
                else
                {
                    dissipated_3 = dissipated_4;
                }
            }
        }
        (&st_2)->kappa_c_0 = max((&st_2)->kappa_c_0, psi_contact_0);
    }
    else
    {
        dissipated_3 = 0.0f;
    }
    float dmg_0 = (&st_2)->damage_0;
    float3 _S325 = float3(0.0f) ;
    float3 qc_ang_0;
    if(((&st_2)->damage_0) == 0.0f)
    {
        if((flags_0 & 8U) == 0U)
        {
            float3 _S326 = contact_offsets_0(mat_8, b_14, (&st_2)->crush_1, plastic_4, d_lin_3, d_ang_3);
            (&st_2)->plastic_x_0 = _S326.x;
            (&st_2)->plastic_y_0 = _S326.y;
            (&st_2)->plastic_t_0 = _S326.z;
        }
        diss_contact_0 = 0.0f;
        qc_lin_0 = _S325;
        qc_ang_0 = _S325;
        psi_contact_0 = 0.0f;
    }
    else
    {
        Contact_0 _S327 = contact_part_0(mat_8, b_14, (&st_2)->crush_1, plastic_4, d_lin_3, d_ang_3);
        (&st_2)->plastic_x_0 = _S327.plastic_0.x;
        (&st_2)->plastic_y_0 = _S327.plastic_0.y;
        (&st_2)->plastic_t_0 = _S327.plastic_0.z;
        diss_contact_0 = _S327.diss_1;
        qc_lin_0 = _S327.q_lin_1;
        qc_ang_0 = _S327.q_ang_1;
        psi_contact_0 = _S327.energy_5;
    }
    float dissipated_6 = dissipated_3 + dmg_0 * diss_contact_0;
    if(_S305)
    {
        intact_normal_0 = kn_3 * _S304;
    }
    else
    {
        intact_normal_0 = (1.0f - (&st_2)->crush_1) * kn_3 * _S304;
    }
    float _S328 = 1.0f - dmg_0;
    float3 force_lin_2 = float3(_S328 * qe_lin_0.x + dmg_0 * qc_lin_0.x, _S328 * qe_lin_0.y + dmg_0 * qc_lin_0.y, _S328 * intact_normal_0 + dmg_0 * qc_lin_0.z);
    float3 force_ang_2 = qe_ang_0 * float3(_S328)  + qc_ang_0 * float3(dmg_0) ;
    float stored_3 = _S328 * (psi_ts_0 + (1.0f - (&st_2)->crush_1) * psi_c_0) + dmg_0 * psi_contact_0;
    if(has_rebar_1)
    {
        changed_0 = ((&st_2)->rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    float stored_4;
    float3 force_lin_3;
    if(changed_0)
    {
        float4 _S329 = float4(b_14->rebar0_0) ;
        float k_axial_0 = _S329.x;
        float k_dowel_0 = _S329.y;
        float yield_force_0 = _S329.z;
        float dowel_capacity_0 = _S329.w;
        float2 nr_0 = return_map_0(k_axial_0, _S304, (&st_2)->rebar_plastic_0, yield_force_0);
        float2 v1_0 = return_map_0(k_dowel_0, _S300, (&st_2)->rebar_slip0_0, dowel_capacity_0);
        float2 v2_0 = return_map_0(k_dowel_0, _S301, (&st_2)->rebar_slip1_0, dowel_capacity_0);
        float _S330 = nr_0.y;
        float _S331 = v1_0.y;
        float _S332 = v2_0.y;
        float work_0 = yield_force_0 * abs(_S330) + dowel_capacity_0 * (abs(_S331) + abs(_S332));
        (&st_2)->rebar_plastic_0 = (&st_2)->rebar_plastic_0 + _S330;
        (&st_2)->rebar_slip0_0 = (&st_2)->rebar_slip0_0 + _S331;
        (&st_2)->rebar_slip1_0 = (&st_2)->rebar_slip1_0 + _S332;
        (&st_2)->rebar_work_0 = (&st_2)->rebar_work_0 + work_0;
        float dissipated_7 = dissipated_6 + work_0;
        float _S333 = nr_0.x;
        float _S334 = (metal::fast::divide((sq_0(_S333)), (k_axial_0)));
        float _S335 = v1_0.x;
        float _S336 = v2_0.x;
        float _S337 = (metal::fast::divide((sq_0(_S335) + sq_0(_S336)), (k_dowel_0)));
        float elastic_0 = 0.5f * (_S334 + _S337);
        if(fracture_1)
        {
            changed_0 = ((&st_2)->rebar_work_0) >= ((float4(b_14->rebar1_0) ).x);
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            (&st_2)->rebar_broken_0 = 1.0f;
            float dissipated_8 = dissipated_7 + elastic_0;
            force_lin_3 = force_lin_2;
            stored_4 = stored_3;
            dissipated_3 = dissipated_8;
        }
        else
        {
            float stored_5 = stored_3 + elastic_0;
            force_lin_3 = force_lin_2 + float3(_S335, _S336, _S333);
            stored_4 = stored_5;
            dissipated_3 = dissipated_7;
        }
    }
    else
    {
        force_lin_3 = force_lin_2;
        stored_4 = stored_3;
        dissipated_3 = dissipated_6;
    }
    if(fracture_1)
    {
        changed_0 = _S290;
    }
    else
    {
        changed_0 = false;
    }
    if(changed_0)
    {
        changed_0 = (flags_0 & 64U) != 0U;
    }
    else
    {
        changed_0 = false;
    }
    if(changed_0)
    {
        Measures_0 _S338 = stress_measures_0(b_14, force_lin_3, force_ang_2);
        thread Measures_0 _S339 = _S338;
        float4 _S340 = failure_indices_0(mat_8, b_14, &_S339, weibull_0);
        float _S341 = life_rate_0(mat_8, max(max(_S340.x, _S340.y), _S340.z));
        (&st_2)->life_0 = max((&st_2)->life_0 - _S341 * dt_6, 0.0f);
    }
    if(((&st_2)->damage_0) != _S308)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = ((&st_2)->crush_1) != _S310;
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = ((&st_2)->plastic_x_0) != (state_2->plastic_x_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = ((&st_2)->plastic_y_0) != (state_2->plastic_y_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = ((&st_2)->plastic_t_0) != (state_2->plastic_t_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = ((&st_2)->rebar_plastic_0) != (state_2->rebar_plastic_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = ((&st_2)->rebar_slip0_0) != (state_2->rebar_slip0_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = ((&st_2)->rebar_slip1_0) != (state_2->rebar_slip1_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = ((&st_2)->rebar_broken_0) != (state_2->rebar_broken_0);
    }
    float released_0;
    if(changed_0)
    {
        released_0 = frozen_0 - stored_4;
    }
    else
    {
        released_0 = 0.0f;
    }
    float _S342 = max(min(dissipated_3, released_0), 0.0f);
    float overshoot_1 = released_0 - _S342;
    (&st_2)->dissipated_0 = (&st_2)->dissipated_0 + _S342;
    thread JointResponse_0 resp_0;
    (&resp_0)->force_lin_1 = force_lin_3;
    (&resp_0)->force_ang_1 = force_ang_2;
    (&resp_0)->state_1 = st_2;
    (&resp_0)->dissipated_2 = _S342;
    (&resp_0)->overshoot_0 = overshoot_1;
    (&resp_0)->stored_2 = stored_4;
    if(_S287)
    {
        thread JointState_0 _S343 = st_2;
        bool _S344 = connected_0(&_S343, has_rebar_1);
        changed_0 = !_S344;
    }
    else
    {
        changed_0 = false;
    }
    (&resp_0)->disconnected_0 = changed_0;
    (&resp_0)->fatigue_before_0 = _S295;
    (&resp_0)->measures_0 = _S288;
    return resp_0;
}

struct BondDyn_0
{
    JointState_0 js_0;
    float4 force_lin_0;
    float4 force_ang_0;
    float4 sums_0;
    float4 comps_0;
    uint4 events_0;
};

void secant_factors_0(const JointBond_natural_0 thread* b_15, const JointState_0 thread* st_3, float3 d_lin_4, float3 thread* f_lin_0, float3 thread* f_ang_0)
{
    float _S345 = st_3->damage_0;
    bool compressed_0 = (d_lin_4.z) < 0.0f;
    float contact_2;
    if(compressed_0)
    {
        contact_2 = _S345;
    }
    else
    {
        contact_2 = 0.0f;
    }
    float _S346 = 1.0f - _S345;
    float _S347 = max(_S346 + contact_2, 9.99999997475242708e-07f);
    float normal_2;
    if(compressed_0)
    {
        normal_2 = max(1.0f - st_3->crush_1, 9.99999997475242708e-07f);
    }
    else
    {
        normal_2 = max(_S346, 9.99999997475242708e-07f);
    }
    *f_lin_0 = float3(_S347, _S347, normal_2);
    *f_ang_0 = float3(_S347) ;
    bool _S348;
    if(((float4(b_15->stiff1_0) ).w) != 0.0f)
    {
        _S348 = (st_3->rebar_broken_0) == 0.0f;
    }
    else
    {
        _S348 = false;
    }
    if(_S348)
    {
        float4 _S349 = float4(b_15->rebar0_0) ;
        float4 _S350 = float4(b_15->stiff0_0) ;
        float _S351 = (metal::fast::divide((_S349.x), (_S350.x)));
        (*f_lin_0).z = (*f_lin_0).z + _S351;
        float _S352 = _S349.y;
        float _S353 = _S350.y;
        float _S354 = (metal::fast::divide((_S352), (_S353)));
        (*f_lin_0).x = (*f_lin_0).x + _S354;
        float _S355 = (metal::fast::divide((_S352), (_S353)));
        (*f_lin_0).y = (*f_lin_0).y + _S355;
    }
    return;
}

float3 to_body_0(uint _S356, float3 _S357, KernelContext_0 thread* kernelContext_25)
{
    BondStatic_natural_0 device* _S358 = kernelContext_25->bonds_0+_S356;
    return (float4(_S358->t1_0) ).xyz * float3(_S357.x)  + (float4(_S358->t2_0) ).xyz * float3(_S357.y)  + (float4(_S358->normal_0) ).xyz * float3(_S357.z) ;
}

void cov_or_1(uint island_2, uint bits_3, KernelContext_0 thread* kernelContext_26)
{
    uint _S359 = atomic_fetch_or_explicit(((atomic_uint device*)(&(kernelContext_26->islands_0+island_2)->coverage_0)), bits_3, memory_order_relaxed);
    return;
}

bool bond_update_0(uint i_5, float dt_7, bool fracture_2, uint abs_step_0, bool wide_0, KernelContext_0 thread* kernelContext_27)
{
    BondStatic_natural_0 device* _S360 = kernelContext_27->bonds_0+i_5;
    uint4 _S361 = uint4(_S360->law_0.ids_0) ;
    float3 ra_1 = (float4(_S360->ra_0) ).xyz;
    float3 rb_1 = (float4(_S360->rb_0) ).xyz;
    uint _S362 = 4U * _S361.y;
    float3 ta_0 = (float4(*(kernelContext_27->state_0+(_S362 + 1U))) ).xyz;
    float3 va_0 = (float4(*(kernelContext_27->state_0+(_S362 + 2U))) ).xyz;
    float3 wa_1 = (float4(*(kernelContext_27->state_0+(_S362 + 3U))) ).xyz;
    uint _S363 = 4U * _S361.z;
    float3 tb_0 = (float4(*(kernelContext_27->state_0+(_S363 + 1U))) ).xyz;
    float3 vb_0 = (float4(*(kernelContext_27->state_0+(_S363 + 2U))) ).xyz;
    float3 wb_1 = (float4(*(kernelContext_27->state_0+(_S363 + 3U))) ).xyz;
    float3 _S364 = to_local_0(i_5, (float4(*(kernelContext_27->state_0+_S363)) ).xyz + cross(tb_0, rb_1) - ((float4(*(kernelContext_27->state_0+_S362)) ).xyz + cross(ta_0, ra_1)), kernelContext_27);
    float3 _S365 = to_local_0(i_5, tb_0 - ta_0, kernelContext_27);
    float3 _S366 = to_local_0(i_5, vb_0 + cross(wb_1, rb_1) - (va_0 + cross(wa_1, ra_1)), kernelContext_27);
    float3 _S367 = to_local_0(i_5, wb_1 - wa_1, kernelContext_27);
    JointState_0 previous_0 = (kernelContext_27->bond_dyn_0+i_5)->js_0;
    thread JointState_0 _S368 = (kernelContext_27->bond_dyn_0+i_5)->js_0;
    bool _S369 = is_damaged_0(&_S368);
    thread JointBond_natural_0 _S370 = (kernelContext_27->bonds_0+i_5)->law_0;
    thread JointState_0 _S371 = (kernelContext_27->bond_dyn_0+i_5)->js_0;
    JointResponse_0 _S372 = joint_evaluate_0(&kernelContext_27->materials_0->m_0[(uint4((kernelContext_27->bonds_0+i_5)->law_0.ids_0) ).x], &_S370, &_S371, _S364, _S365, dt_7, fracture_2);
    BondStatic_natural_0 device* _S373 = kernelContext_27->bonds_0+i_5;
    BondDyn_natural_0 device* _S374 = kernelContext_27->bond_dyn_0+i_5;
    float4 _S375 = float4((*_S374).force_lin_0) ;
    float4 _S376 = float4((*_S374).force_ang_0) ;
    float4 _S377 = float4((*_S374).sums_0) ;
    float4 _S378 = float4((*_S374).comps_0) ;
    uint4 _S379 = uint4((*_S374).events_0) ;
    thread BondDyn_0 bd_0;
    (&bd_0)->js_0 = (*_S374).js_0;
    (&bd_0)->force_lin_0 = _S375;
    (&bd_0)->force_ang_0 = _S376;
    (&bd_0)->sums_0 = _S377;
    (&bd_0)->comps_0 = _S378;
    (&bd_0)->events_0 = _S379;
    float3 ra_2 = (float4(_S373->ra_0) ).xyz;
    float3 rb_2 = (float4(_S373->rb_0) ).xyz;
    thread JointBond_natural_0 _S380 = _S373->law_0;
    thread JointState_0 _S381 = _S372.state_1;
    thread float3 f_lin_1;
    thread float3 f_ang_1;
    secant_factors_0(&_S380, &_S381, _S364, &f_lin_1, &f_ang_1);
    float3 qd_lin_0 = _S366 * (float4(_S373->c_lin_0) ).xyz * f_lin_1;
    float3 qd_ang_0 = _S367 * (float4(_S373->c_ang_0) ).xyz * f_ang_1;
    float3 q_lin_2 = _S372.force_lin_1 + qd_lin_0;
    float3 q_ang_2 = _S372.force_ang_1 + qd_ang_0;
    float damped_0 = (dot(qd_lin_0, _S366) + dot(qd_ang_0, _S367)) * dt_7;
    float3 _S382 = to_body_0(i_5, q_lin_2, kernelContext_27);
    float3 _S383 = to_body_0(i_5, q_ang_2, kernelContext_27);
    uint _S384 = 3U * i_5;
    *(kernelContext_27->scratch_0+_S384) = packed_float4(float4(_S382, max(_S372.measures_0.tension_0, _S372.measures_0.compression_0))) ;
    *(kernelContext_27->scratch_0+(_S384 + 1U)) = packed_float4(float4(_S383 + cross(ra_2, _S382), 0.0f)) ;
    *(kernelContext_27->scratch_0+(_S384 + 2U)) = packed_float4(float4(- _S383 + cross(rb_2, - _S382), 0.0f)) ;
    thread float _S385 = (&bd_0)->sums_0.x;
    thread float _S386 = (&bd_0)->comps_0.x;
    comp_add1_0(&_S385, &_S386, _S372.dissipated_2);
    (&bd_0)->comps_0.x = _S386;
    (&bd_0)->sums_0.x = _S385;
    thread float _S387 = (&bd_0)->sums_0.y;
    thread float _S388 = (&bd_0)->comps_0.y;
    comp_add1_0(&_S387, &_S388, _S372.overshoot_0);
    (&bd_0)->comps_0.y = _S388;
    (&bd_0)->sums_0.y = _S387;
    thread float _S389 = (&bd_0)->sums_0.z;
    thread float _S390 = (&bd_0)->comps_0.z;
    comp_add1_0(&_S389, &_S390, damped_0);
    (&bd_0)->comps_0.z = _S390;
    (&bd_0)->sums_0.z = _S389;
    (&bd_0)->force_lin_0 = float4(q_lin_2, _S372.stored_2);
    (&bd_0)->force_ang_0 = float4(q_ang_2, max((&bd_0)->force_ang_0.w, _S372.state_1.utilization_0));
    bool _S391;
    if(!_S369)
    {
        thread JointState_0 _S392 = _S372.state_1;
        bool _S393 = is_damaged_0(&_S392);
        _S391 = _S393;
    }
    else
    {
        _S391 = false;
    }
    if(_S391)
    {
        _S391 = ((&bd_0)->events_0.x) == 0U;
    }
    else
    {
        _S391 = false;
    }
    if(_S391)
    {
        (&bd_0)->events_0.x = abs_step_0;
        (&bd_0)->events_0.w = _S372.state_1.mode_0;
        if(wide_0)
        {
            cov_or_1((uint4((&_S380)->ids_0) ).w, 32U, kernelContext_27);
        }
        else
        {
            (*kernelContext_27->g_seen_0)[3U] = 1U;
        }
    }
    if(((&bd_0)->events_0.y) == 0U)
    {
        _S391 = (_S372.state_1.life_0) != (previous_0.life_0);
    }
    else
    {
        _S391 = false;
    }
    if(_S391)
    {
        _S391 = (_S372.fatigue_before_0) > 0.99000000953674316f;
    }
    else
    {
        _S391 = false;
    }
    if(_S391)
    {
        float _S394 = fatigue_factor_0(&kernelContext_27->materials_0->m_0[(uint4((kernelContext_27->bonds_0+i_5)->law_0.ids_0) ).x], _S372.state_1.life_0);
        _S391 = _S394 <= 0.99000000953674316f;
    }
    else
    {
        _S391 = false;
    }
    if(_S391)
    {
        (&bd_0)->events_0.y = abs_step_0;
        if(wide_0)
        {
            cov_or_1((uint4((&_S380)->ids_0) ).w, 64U, kernelContext_27);
        }
        else
        {
            (*kernelContext_27->g_seen_0)[4U] = 1U;
        }
    }
    if(_S372.disconnected_0)
    {
        (&bd_0)->events_0.z = abs_step_0;
    }
    (&bd_0)->js_0 = _S372.state_1;
    BondDyn_natural_0 device* _S395 = kernelContext_27->bond_dyn_0+i_5;
    _S395->js_0 = bd_0.js_0;
    _S395->force_lin_0 = packed_float4(bd_0.force_lin_0) ;
    _S395->force_ang_0 = packed_float4(bd_0.force_ang_0) ;
    _S395->sums_0 = packed_float4(bd_0.sums_0) ;
    _S395->comps_0 = packed_float4(bd_0.comps_0) ;
    _S395->events_0 = packed_uint4(bd_0.events_0) ;
    return _S372.disconnected_0;
}

float3 inverse_rotate_1(const Quat_0 thread* q_9, float3 v_8)
{
    thread Quat_0 c_10;
    (&c_10)->w_0 = q_9->w_0;
    (&c_10)->x_0 = - q_9->x_0;
    (&c_10)->y_0 = - q_9->y_0;
    (&c_10)->z_0 = - q_9->z_0;
    thread Quat_0 _S396 = c_10;
    float3 _S397 = rotate_0(&_S396, v_8);
    return _S397;
}

void chunk_update_0(uint c_11, const Island_0 thread* isl_4, const Rigid_0 thread* rg_3, float dt_8, bool rml_0, uint step_0, bool contact_3, float thread* work_1, float thread* work_err_0, KernelContext_0 thread* kernelContext_28)
{
    ChunkStatic_natural_0 device* _S398 = kernelContext_28->chunks_0+c_11;
    float3 _S399 = float3(0.0f) ;
    uint _S400 = kernelContext_28->index_0[c_11];
    float peak_0 = 0.0f;
    uint e_1 = _S400;
    float3 fi_0 = _S399;
    float3 mi_0 = _S399;
    for(;;)
    {
        if(e_1 < (kernelContext_28->index_0)[c_11 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_1 = kernelContext_28->index_0[e_1];
        uint _S401 = 3U * (entry_1 >> 1U);
        float4 _S402 = float4(*(kernelContext_28->scratch_0+_S401)) ;
        if((entry_1 & 1U) == 0U)
        {
            float3 mi_1 = mi_0 + (float4(*(kernelContext_28->scratch_0+(_S401 + 1U))) ).xyz;
            fi_0 = fi_0 + _S402.xyz;
            mi_0 = mi_1;
        }
        else
        {
            float3 mi_2 = mi_0 + (float4(*(kernelContext_28->scratch_0+(_S401 + 2U))) ).xyz;
            fi_0 = fi_0 + - _S402.xyz;
            mi_0 = mi_2;
        }
        float _S403 = max(peak_0, _S402.w);
        uint _S404 = e_1 + 1U;
        peak_0 = _S403;
        e_1 = _S404;
    }
    uint _S405 = 4U * c_11;
    float3 u_0 = (float4(*(kernelContext_28->state_0+_S405)) ).xyz;
    uint _S406 = _S405 + 1U;
    float3 th_1 = (float4(*(kernelContext_28->state_0+_S406)) ).xyz;
    uint _S407 = _S405 + 2U;
    float3 v_9 = (float4(*(kernelContext_28->state_0+_S407)) ).xyz;
    uint _S408 = _S405 + 3U;
    float3 w_5 = (float4(*(kernelContext_28->state_0+_S408)) ).xyz;
    float4 _S409 = float4(_S398->center_0) ;
    float mass_0 = _S409.w;
    float3 _S410 = _S409.xyz;
    float3 _S411 = isl_4->com_0.xyz;
    float3 _S412 = rotate_0(&rg_3->rot_0, _S410 + u_0 - _S411);
    thread float3 f_load_0;
    thread float3 t_load_0;
    thread float3 rep_f_1;
    thread float3 rep_m_1;
    chunk_loads_0(c_11, c_11, &rg_3->rot_0, step_0, dt_8, contact_3, dt_8, &f_load_0, &t_load_0, &rep_f_1, &rep_m_1, kernelContext_28);
    record_chunk_load_0(c_11, f_load_0, t_load_0, kernelContext_28);
    float3 _S413 = float3(mass_0) ;
    float3 f_world_0 = f_load_0 + kernelContext_28->params_0->gravity_0.xyz * _S413;
    float3 t_world_0 = t_load_0;
    float3 f_world_1;
    float3 t_world_1;
    if(rml_0)
    {
        float3 f_world_2 = f_world_0 - (rg_3->a_0 + cross(rg_3->alpha_0, _S412) + cross(rg_3->w_1, cross(rg_3->w_1, _S412))) * _S413;
        float4 _S414 = float4(_S398->inertia0_1) ;
        float4 _S415 = float4(_S398->inertia1_1) ;
        float4 _S416 = float4(_S398->inertia2_1) ;
        float3 _S417 = world_mul_0(&rg_3->rot_0, _S414, _S415, _S416, rg_3->alpha_0);
        float3 _S418 = world_mul_0(&rg_3->rot_0, _S414, _S415, _S416, rg_3->w_1);
        float3 t_world_2 = t_world_0 - (_S417 + cross(rg_3->w_1, _S418));
        f_world_1 = f_world_2;
        t_world_1 = t_world_2;
    }
    else
    {
        f_world_1 = f_world_0;
        t_world_1 = t_world_0;
    }
    float3 _S419 = inverse_rotate_1(&rg_3->rot_0, f_world_1);
    float3 _S420 = inverse_rotate_1(&rg_3->rot_0, t_world_1);
    float3 f_ext_0;
    float3 m_ext_0;
    if(rml_0)
    {
        float3 _S421 = inverse_rotate_1(&rg_3->rot_0, rg_3->w_1);
        float4 _S422 = float4(_S398->inertia0_1) ;
        float4 _S423 = float4(_S398->inertia1_1) ;
        float4 _S424 = float4(_S398->inertia2_1) ;
        float3 i_w_0 = rows_mul_0(_S422, _S423, _S424, w_5);
        float3 m_ext_1 = _S420 - (cross(_S421, i_w_0) + cross(w_5, rows_mul_0(_S422, _S423, _S424, _S421)) + cross(w_5, i_w_0));
        f_ext_0 = _S419 - cross(_S421, v_9) * float3((2.0f * mass_0)) ;
        m_ext_0 = m_ext_1;
    }
    else
    {
        f_ext_0 = _S419;
        m_ext_0 = _S420;
    }
    float3 f_9 = f_ext_0 + rep_f_1 + fi_0;
    float3 m_3 = m_ext_0 + rep_m_1 + mi_0;
    uint support_0 = (uint4(_S398->info_1) ).x;
    float3 _S425 = float3((float4(*(kernelContext_28->state_0+_S406)) ).w, (float4(*(kernelContext_28->state_0+_S407)) ).w, (float4(*(kernelContext_28->state_0+_S408)) ).w);
    float3 reaction_0;
    float3 u_1;
    float3 th_2;
    float3 v_10;
    float3 w_6;
    if(support_0 == 1U)
    {
        reaction_0 = - f_9;
        u_1 = u_0;
        th_2 = th_1;
        v_10 = _S399;
        w_6 = _S399;
    }
    else
    {
        float4 _S426 = float4(_S398->scale_0) ;
        float3 w_7 = w_5 + rows_mul_0(float4(_S398->inv0_1) , float4(_S398->inv1_1) , float4(_S398->inv2_1) , m_3) * float3((dt_8 * _S426.z)) ;
        float3 _S427 = float3(dt_8) ;
        float3 th_3 = th_1 + w_7 * _S427;
        if(support_0 == 2U)
        {
            reaction_0 = - f_9;
            u_1 = u_0;
            th_2 = _S399;
        }
        else
        {
            float3 v_11 = v_9 + f_9 * float3((dt_8 * _S426.y)) ;
            float3 u_2 = u_0 + v_11 * _S427;
            reaction_0 = _S425;
            u_1 = u_2;
            th_2 = v_11;
        }
        float3 _S428 = th_2;
        th_2 = th_3;
        v_10 = _S428;
        w_6 = w_7;
    }
    *(kernelContext_28->state_0+_S405) = packed_float4(float4(u_1, peak_0)) ;
    *(kernelContext_28->state_0+_S406) = packed_float4(float4(th_2, reaction_0.x)) ;
    *(kernelContext_28->state_0+_S407) = packed_float4(float4(v_10, reaction_0.y)) ;
    *(kernelContext_28->state_0+_S408) = packed_float4(float4(w_6, reaction_0.z)) ;
    float3 _S429 = rotate_0(&rg_3->rot_0, _S410 + u_1 - _S411);
    float3 _S430 = rg_3->vel_0 + rg_3->vel_err_0 + cross(rg_3->w_1, _S429);
    float3 _S431 = rotate_0(&rg_3->rot_0, v_10);
    float3 v_world_0 = _S430 + _S431;
    float3 _S432 = rotate_0(&rg_3->rot_0, w_6);
    comp_add1_0(work_1, work_err_0, (dot(f_load_0, v_world_0) + dot(t_load_0, rg_3->w_1 + _S432)) * dt_8);
    return;
}

void comp_add_0(float3 thread* sum_1, float3 thread* err_1, float3 x_6)
{
    float3 t_10 = *sum_1 + x_6;
    float3 _S433 = abs(x_6);
    *err_1 = *err_1 + (select(x_6, *sum_1, (abs(*sum_1)) >= _S433) - t_10 + select(*sum_1, x_6, (abs(*sum_1)) >= _S433));
    *sum_1 = t_10;
    return;
}

float3 turn_difference_0(float3 omega_0, float dt_9, float3 v_12)
{
    float _S434 = length(omega_0);
    float angle_1 = _S434 * dt_9;
    if(angle_1 < 1.00000000317107685e-30f)
    {
        return float3(0.0f) ;
    }
    float3 a_8 = omega_0 / float3(_S434) ;
    float s_4 = sin(0.5f * angle_1);
    float3 av_0 = cross(a_8, v_12);
    return av_0 * float3(sin(angle_1))  + cross(a_8, av_0) * float3((2.0f * s_4 * s_4)) ;
}

float4 turn_minus_one_0(float3 axis_2, float angle_2)
{
    float s_5 = sin(0.25f * angle_2);
    return float4(axis_2 * float3(sin(0.5f * angle_2)) , -2.0f * s_5 * s_5);
}

Quat_0 quat_mul_0(const Quat_0 thread* a_9, const Quat_0 thread* o_1)
{
    thread Quat_0 r_5;
    (&r_5)->w_0 = a_9->w_0 * o_1->w_0 - a_9->x_0 * o_1->x_0 - a_9->y_0 * o_1->y_0 - a_9->z_0 * o_1->z_0;
    (&r_5)->x_0 = a_9->w_0 * o_1->x_0 + a_9->x_0 * o_1->w_0 + a_9->y_0 * o_1->z_0 - a_9->z_0 * o_1->y_0;
    (&r_5)->y_0 = a_9->w_0 * o_1->y_0 - a_9->x_0 * o_1->z_0 + a_9->y_0 * o_1->w_0 + a_9->z_0 * o_1->x_0;
    (&r_5)->z_0 = a_9->w_0 * o_1->z_0 + a_9->x_0 * o_1->y_0 - a_9->y_0 * o_1->x_0 + a_9->z_0 * o_1->w_0;
    return r_5;
}

float4 quat_vec_0(const Quat_0 thread* q_10)
{
    return float4(q_10->x_0, q_10->y_0, q_10->z_0, q_10->w_0);
}

void comp_add4_0(float4 thread* sum_2, float4 thread* err_2, float4 x_7)
{
    float4 t_11 = *sum_2 + x_7;
    float4 _S435 = abs(x_7);
    *err_2 = *err_2 + (select(x_7, *sum_2, (abs(*sum_2)) >= _S435) - t_11 + select(*sum_2, x_7, (abs(*sum_2)) >= _S435));
    *sum_2 = t_11;
    return;
}

void quat_accumulate_0(Quat_0 thread* hi_1, float4 thread* lo_1, float4 x_8)
{
    thread Quat_0 _S436 = *hi_1;
    float4 _S437 = quat_vec_0(&_S436);
    thread float4 sum_3 = _S437;
    comp_add4_0(&sum_3, lo_1, x_8);
    float4 t_12 = sum_3 + *lo_1;
    *lo_1 = *lo_1 - (t_12 - sum_3);
    *hi_1 = quat_of_0(t_12);
    return;
}

void turn_left_0(Quat_0 thread* hi_2, float4 thread* lo_2, float3 omega_1, float dt_10)
{
    float _S438 = length(omega_1);
    float angle_3 = _S438 * dt_10;
    if(angle_3 < 1.00000000317107685e-30f)
    {
        return;
    }
    float4 d_2 = turn_minus_one_0(omega_1 / float3(_S438) , angle_3);
    thread Quat_0 dq_0;
    (&dq_0)->x_0 = d_2.x;
    (&dq_0)->y_0 = d_2.y;
    (&dq_0)->z_0 = d_2.z;
    (&dq_0)->w_0 = d_2.w;
    thread Quat_0 _S439 = dq_0;
    thread Quat_0 _S440 = *hi_2;
    Quat_0 _S441 = quat_mul_0(&_S439, &_S440);
    thread Quat_0 _S442 = _S441;
    float4 _S443 = quat_vec_0(&_S442);
    quat_accumulate_0(hi_2, lo_2, _S443);
    return;
}

void integrate_rigid_0(const Island_0 thread* isl_5, Rigid_0 thread* rg_4, float dt_11)
{
    float3 _S444 = float3(dt_11) ;
    comp_add_0(&rg_4->l_0, &rg_4->l_err_0, rg_4->torque_0 * _S444);
    float3 l_1 = rg_4->l_0 + rg_4->l_err_0;
    comp_add_0(&rg_4->vel_0, &rg_4->vel_err_0, rg_4->a_0 * _S444);
    float3 vel_1 = rg_4->vel_0 + rg_4->vel_err_0;
    float4 _S445 = isl_5->inv0_0;
    float4 _S446 = isl_5->inv1_0;
    float4 _S447 = isl_5->inv2_0;
    thread Quat_0 _S448 = rg_4->rot_0;
    float3 _S449 = world_mul_0(&_S448, isl_5->inv0_0, isl_5->inv1_0, isl_5->inv2_0, l_1);
    float3 _S450 = vel_1 * _S444;
    float3 _S451 = isl_5->com_0.xyz;
    thread Quat_0 _S452 = rg_4->rot_0;
    float3 _S453 = rotate_0(&_S452, _S451);
    comp_add_0(&rg_4->pos_0, &rg_4->pos_err_0, _S450 - turn_difference_0(_S449, dt_11, _S453));
    turn_left_0(&rg_4->rot_0, &rg_4->rot_err_0, _S449, dt_11);
    thread Quat_0 _S454 = rg_4->rot_0;
    float3 _S455 = world_mul_0(&_S454, _S445, _S446, _S447, l_1);
    rg_4->w_1 = _S455;
    return;
}

void drift_moments_0(uint c_12, float3 thread* tu_0, float3 thread* pv_0, KernelContext_0 thread* kernelContext_29)
{
    ChunkStatic_natural_0 device* _S456 = kernelContext_29->chunks_0+c_12;
    uint _S457 = 4U * c_12;
    float3 _S458 = float3(((float4(_S456->center_0) ).w * (float4(_S456->scale_0) ).x)) ;
    *tu_0 = *tu_0 + (float4(*(kernelContext_29->state_0+_S457)) ).xyz * _S458;
    *pv_0 = *pv_0 + (float4(*(kernelContext_29->state_0+(_S457 + 2U))) ).xyz * _S458;
    return;
}

void drift_angular_0(uint c_13, float3 wcom_1, float3 tr_0, float3 dv_0, float3 thread* lu_0, float3 thread* lv_0, KernelContext_0 thread* kernelContext_30)
{
    ChunkStatic_natural_0 device* _S459 = kernelContext_30->chunks_0+c_13;
    float4 _S460 = float4(_S459->center_0) ;
    float3 r_6 = _S460.xyz - wcom_1;
    uint _S461 = 4U * c_13;
    float3 _S462 = float3(_S460.w) ;
    float4 _S463 = float4(_S459->inertia0_1) ;
    float4 _S464 = float4(_S459->inertia1_1) ;
    float4 _S465 = float4(_S459->inertia2_1) ;
    float3 _S466 = float3((float4(_S459->scale_0) ).x) ;
    *lu_0 = *lu_0 + (cross(r_6, (float4(*(kernelContext_30->state_0+_S461)) ).xyz - tr_0) * _S462 + rows_mul_0(_S463, _S464, _S465, (float4(*(kernelContext_30->state_0+(_S461 + 1U))) ).xyz)) * _S466;
    *lv_0 = *lv_0 + (cross(r_6, (float4(*(kernelContext_30->state_0+(_S461 + 2U))) ).xyz - dv_0) * _S462 + rows_mul_0(_S463, _S464, _S465, (float4(*(kernelContext_30->state_0+(_S461 + 3U))) ).xyz)) * _S466;
    return;
}

void drift_apply_0(uint c_14, float3 wcom_2, float3 tr_1, float3 phi_0, float3 dv_1, float3 dw_0, KernelContext_0 thread* kernelContext_31)
{
    float3 r_7 = (float4((kernelContext_31->chunks_0+c_14)->center_0) ).xyz - wcom_2;
    uint _S467 = 4U * c_14;
    *(kernelContext_31->state_0+_S467) = packed_float4(float4((float4(*(kernelContext_31->state_0+_S467)) ).xyz - (tr_1 + cross(phi_0, r_7)), (float4(*(kernelContext_31->state_0+_S467)) ).w)) ;
    uint _S468 = _S467 + 1U;
    *(kernelContext_31->state_0+_S468) = packed_float4(float4((float4(*(kernelContext_31->state_0+_S468)) ).xyz - phi_0, (float4(*(kernelContext_31->state_0+_S468)) ).w)) ;
    uint _S469 = _S467 + 2U;
    *(kernelContext_31->state_0+_S469) = packed_float4(float4((float4(*(kernelContext_31->state_0+_S469)) ).xyz - (dv_1 + cross(dw_0, r_7)), (float4(*(kernelContext_31->state_0+_S469)) ).w)) ;
    uint _S470 = _S467 + 3U;
    *(kernelContext_31->state_0+_S470) = packed_float4(float4((float4(*(kernelContext_31->state_0+_S470)) ).xyz - dw_0, (float4(*(kernelContext_31->state_0+_S470)) ).w)) ;
    return;
}

void turn_right_0(Quat_0 thread* hi_3, float4 thread* lo_3, float3 phi_1)
{
    float angle_4 = length(phi_1);
    if(angle_4 < 1.00000000317107685e-30f)
    {
        return;
    }
    float4 d_3 = turn_minus_one_0(phi_1 / float3(angle_4) , angle_4);
    thread Quat_0 dq_1;
    (&dq_1)->x_0 = d_3.x;
    (&dq_1)->y_0 = d_3.y;
    (&dq_1)->z_0 = d_3.z;
    (&dq_1)->w_0 = d_3.w;
    thread Quat_0 _S471 = *hi_3;
    thread Quat_0 _S472 = dq_1;
    Quat_0 _S473 = quat_mul_0(&_S471, &_S472);
    thread Quat_0 _S474 = _S473;
    float4 _S475 = quat_vec_0(&_S474);
    quat_accumulate_0(hi_3, lo_3, _S475);
    return;
}

void drift_rigid_0(const Island_0 thread* isl_6, Rigid_0 thread* rg_5, float3 tr_2, float3 phi_2, float3 dv_2, float3 dw_1)
{
    float3 wcom_3 = isl_6->wcom_0.xyz;
    Quat_0 rot_2 = rg_5->rot_0;
    float3 _S476 = tr_2 - turn_difference_0(phi_2, 1.0f, wcom_3);
    thread Quat_0 _S477 = rg_5->rot_0;
    float3 _S478 = rotate_0(&_S477, _S476);
    comp_add_0(&rg_5->pos_0, &rg_5->pos_err_0, _S478);
    thread Quat_0 _S479 = rg_5->rot_0;
    float3 _S480 = inverse_rotate_1(&_S479, rg_5->w_1);
    float4 _S481 = isl_6->inertia0_0;
    float4 _S482 = isl_6->inertia1_0;
    float4 _S483 = isl_6->inertia2_0;
    float3 _S484 = cross(phi_2, rows_mul_0(isl_6->inertia0_0, isl_6->inertia1_0, isl_6->inertia2_0, _S480)) - rows_mul_0(isl_6->inertia0_0, isl_6->inertia1_0, isl_6->inertia2_0, cross(phi_2, _S480));
    thread Quat_0 _S485 = rg_5->rot_0;
    float3 _S486 = rotate_0(&_S485, _S484);
    turn_right_0(&rg_5->rot_0, &rg_5->rot_err_0, phi_2);
    float3 _S487 = dv_2 + cross(dw_1, isl_6->com_0.xyz - wcom_3);
    thread Quat_0 _S488 = rot_2;
    float3 _S489 = rotate_0(&_S488, _S487);
    comp_add_0(&rg_5->vel_0, &rg_5->vel_err_0, _S489);
    thread Quat_0 _S490 = rot_2;
    float3 _S491 = rotate_0(&_S490, dw_1);
    rg_5->w_1 = rg_5->w_1 + _S491;
    thread Quat_0 _S492 = rg_5->rot_0;
    float3 _S493 = world_mul_0(&_S492, _S481, _S482, _S483, _S491);
    comp_add_0(&rg_5->l_0, &rg_5->l_err_0, _S486 + _S493);
    return;
}

void write_probe_0(uint slot_0, uint k_10, float value_0, KernelContext_0 thread* kernelContext_32)
{
    uint at_1 = kernelContext_32->params_0->probe_base_0 * 4U + slot_0 * kernelContext_32->params_0->probe_stride_0 + k_10;
    thread float4 v_13 = float4(*(kernelContext_32->scratch_0+at_1 / 4U)) ;
    v_13[at_1 % 4U] = value_0;
    *(kernelContext_32->scratch_0+at_1 / 4U) = packed_float4(v_13) ;
    return;
}

void record_probes_0(const Island_0 thread* isl_7, const Rigid_0 thread* rg_6, uint k_11, KernelContext_0 thread* kernelContext_33)
{
    uint4 _S494 = isl_7->probes_0;
    uint at_2 = isl_7->probes_0.x;
    for(;;)
    {
        if(at_2 < (_S494.y))
        {
        }
        else
        {
            break;
        }
        uint4 info_3 = (as_type<uint4>((float4(*(kernelContext_33->loads_0+at_2)) )));
        float4 _S495 = float4(*(kernelContext_33->loads_0+(at_2 + 1U))) ;
        float4 _S496 = float4(*(kernelContext_33->loads_0+(at_2 + 2U))) ;
        float4 _S497 = float4(*(kernelContext_33->loads_0+(at_2 + 3U))) ;
        uint kind_4 = info_3.x;
        uint i_6 = info_3.y;
        float value_1;
        if(kind_4 == 0U)
        {
            float3 small_0 = (float4(*(kernelContext_33->state_0+4U * i_6)) ).xyz + float3(_S495.w, _S496.w, _S497.w);
            float3 _S498 = rg_6->pos_0 - _S496.xyz;
            float3 _S499 = rotate_0(&rg_6->rot_0, (float4((kernelContext_33->chunks_0+i_6)->center_0) ).xyz);
            float3 _S500 = _S498 + _S499;
            float3 _S501 = rg_6->pos_err_0 - _S497.xyz;
            float3 _S502 = rotate_0(&rg_6->rot_0, small_0);
            value_1 = dot(_S500 + (_S501 + _S502), _S495.xyz);
        }
        else
        {
            if(kind_4 == 1U)
            {
                uint _S503 = 4U * i_6;
                float3 _S504 = rotate_0(&rg_6->rot_0, (float4((kernelContext_33->chunks_0+i_6)->center_0) ).xyz + (float4(*(kernelContext_33->state_0+_S503)) ).xyz - isl_7->com_0.xyz);
                float3 _S505 = rg_6->vel_0 + rg_6->vel_err_0 + cross(rg_6->w_1, _S504);
                float3 _S506 = rotate_0(&rg_6->rot_0, (float4(*(kernelContext_33->state_0+(_S503 + 2U))) ).xyz);
                value_1 = dot(_S505 + _S506, _S495.xyz);
            }
            else
            {
                if(kind_4 == 2U)
                {
                    uint _S507 = 3U * i_6;
                    float3 f_10 = (float4(*(kernelContext_33->scratch_0+_S507)) ).xyz;
                    bool _S508 = (info_3.z) == 0U;
                    float3 mc_0;
                    if(_S508)
                    {
                        mc_0 = (float4(*(kernelContext_33->scratch_0+(_S507 + 1U))) ).xyz;
                    }
                    else
                    {
                        mc_0 = (float4(*(kernelContext_33->scratch_0+(_S507 + 2U))) ).xyz;
                    }
                    float3 fc_3;
                    if(_S508)
                    {
                        fc_3 = f_10;
                    }
                    else
                    {
                        fc_3 = - f_10;
                    }
                    value_1 = dot(fc_3, _S495.xyz) + dot(mc_0, _S496.xyz);
                }
                else
                {
                    uint _S509 = 4U * i_6;
                    float3 _S510 = rotate_0(&rg_6->rot_0, float3((float4(*(kernelContext_33->state_0+(_S509 + 1U))) ).w, (float4(*(kernelContext_33->state_0+(_S509 + 2U))) ).w, (float4(*(kernelContext_33->state_0+(_S509 + 3U))) ).w));
                    value_1 = dot(_S510, _S495.xyz);
                }
            }
        }
        write_probe_0(info_3.w, k_11, value_1, kernelContext_33);
        at_2 = at_2 + 4U;
    }
    return;
}

bool island_connected_narrow_0(uint tid_3, uint ct_0, const Island_0 thread* isl_8, KernelContext_0 thread* kernelContext_34)
{
    uint _S511 = isl_8->range_0.x;
    uint n_2 = isl_8->range_0.y - _S511;
    bool converged_0;
    if(n_2 > 512U)
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = (kernelContext_34->params_0->device_connect_0) == 0U;
    }
    if(converged_0)
    {
        return false;
    }
    uint c_15 = ct_0;
    for(;;)
    {
        if(c_15 < n_2)
        {
        }
        else
        {
            break;
        }
        (*kernelContext_34->g_label_narrow_0)[c_15] = c_15;
        c_15 = c_15 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    converged_0 = false;
    uint pass_0 = 0U;
    for(;;)
    {
        if(pass_0 <= n_2)
        {
        }
        else
        {
            break;
        }
        if(tid_3 == 0U)
        {
            *kernelContext_34->g_changed_narrow_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c_15 = ct_0;
        for(;;)
        {
            if(c_15 < n_2)
            {
            }
            else
            {
                break;
            }
            uint gc_0 = _S511 + c_15;
            uint _S512 = kernelContext_34->index_0[gc_0];
            uint l_2 = (*kernelContext_34->g_label_narrow_0)[c_15];
            uint e_2 = _S512;
            for(;;)
            {
                if(e_2 < (kernelContext_34->index_0)[gc_0 + 1U])
                {
                }
                else
                {
                    break;
                }
                uint entry_2 = kernelContext_34->index_0[e_2];
                uint bond_0 = entry_2 >> 1U;
                bool _S513 = ((float4((kernelContext_34->bonds_0+bond_0)->law_0.stiff1_0) ).w) != 0.0f;
                thread JointState_0 _S514 = (kernelContext_34->bond_dyn_0+bond_0)->js_0;
                bool _S515 = connected_0(&_S514, _S513);
                if(!_S515)
                {
                    e_2 = e_2 + 1U;
                    continue;
                }
                uint other_0;
                if((entry_2 & 1U) == 0U)
                {
                    other_0 = (uint4((kernelContext_34->bonds_0+bond_0)->law_0.ids_0) ).z;
                }
                else
                {
                    other_0 = (uint4((kernelContext_34->bonds_0+bond_0)->law_0.ids_0) ).y;
                }
                l_2 = min(l_2, (*kernelContext_34->g_label_narrow_0)[other_0 - _S511]);
                e_2 = e_2 + 1U;
            }
            uint _S516 = min(l_2, (*kernelContext_34->g_label_narrow_0)[l_2]);
            if(_S516 < (*kernelContext_34->g_label_narrow_0)[c_15])
            {
                (*kernelContext_34->g_label_narrow_0)[c_15] = _S516;
                *kernelContext_34->g_changed_narrow_0 = 1U;
            }
            c_15 = c_15 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool converged_1 = (*kernelContext_34->g_changed_narrow_0) == 0U;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(converged_1)
        {
            converged_0 = converged_1;
            break;
        }
        uint _S517 = pass_0 + 1U;
        converged_0 = converged_1;
        pass_0 = _S517;
    }
    if(!converged_0)
    {
        return false;
    }
    if(tid_3 == 0U)
    {
        *kernelContext_34->g_changed_narrow_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    c_15 = ct_0;
    for(;;)
    {
        if(c_15 < n_2)
        {
        }
        else
        {
            break;
        }
        if(((*kernelContext_34->g_label_narrow_0)[c_15]) != 0U)
        {
            *kernelContext_34->g_changed_narrow_0 = 1U;
        }
        c_15 = c_15 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_34->g_changed_narrow_0) == 0U;
}

void contact_split_at_0(uint at_3, KernelContext_0 thread* kernelContext_35)
{
    uint previous_1 = (uint4((kernelContext_35->islands_0+kernelContext_35->params_0->halt_index_0)->info_0) ).y;
    uint _S518;
    if(previous_1 == 0U)
    {
        _S518 = at_3;
    }
    else
    {
        _S518 = min(previous_1, at_3);
    }
    (kernelContext_35->islands_0+kernelContext_35->params_0->halt_index_0)->info_0[int(1)] = _S518;
    return;
}

uint seen_bits_0(KernelContext_0 thread* kernelContext_36)
{
    uint _S519;
    if(((*kernelContext_36->g_seen_0)[0U]) != 0U)
    {
        _S519 = 32768U;
    }
    else
    {
        _S519 = 0U;
    }
    uint _S520;
    if(((*kernelContext_36->g_seen_0)[1U]) != 0U)
    {
        _S520 = 65536U;
    }
    else
    {
        _S520 = 0U;
    }
    uint _S521 = _S519 | _S520;
    if(((*kernelContext_36->g_seen_0)[2U]) != 0U)
    {
        _S519 = 131072U;
    }
    else
    {
        _S519 = 0U;
    }
    uint _S522 = _S521 | _S519;
    if(((*kernelContext_36->g_seen_0)[3U]) != 0U)
    {
        _S519 = 32U;
    }
    else
    {
        _S519 = 0U;
    }
    uint _S523 = _S522 | _S519;
    if(((*kernelContext_36->g_seen_0)[4U]) != 0U)
    {
        _S519 = 64U;
    }
    else
    {
        _S519 = 0U;
    }
    return _S523 | _S519;
}

void travel_stop_0(uint substep_0, bool fused_0, KernelContext_0 thread* kernelContext_37)
{
    if(fused_0)
    {
        (kernelContext_37->islands_0+kernelContext_37->params_0->halt_index_0)->range_0[int(0)] = substep_0;
        return;
    }
    (kernelContext_37->islands_0+kernelContext_37->params_0->halt_index_0)->info_0[int(2)] = ((uint4((kernelContext_37->islands_0+kernelContext_37->params_0->halt_index_0)->info_0) ).z) | 1U;
    sentinel_mark_0(1048576U, kernelContext_37);
    return;
}

void travel_check_0(uint c_16, bool fused_1, KernelContext_0 thread* kernelContext_38)
{
    ChunkStatic_natural_0 device* _S524 = kernelContext_38->chunks_0+c_16;
    uint4 _S525 = uint4(_S524->cinfo_0) ;
    if((_S525.z) == 0U)
    {
        return;
    }
    uint _S526 = 4U * c_16;
    uint at_4 = kernelContext_38->params_0->cplan_base_0 + _S526;
    float4 _S527 = float4(*(kernelContext_38->scratch_0+at_4)) ;
    float4 _S528 = float4(*(kernelContext_38->scratch_0+(at_4 + 1U))) ;
    float4 _S529 = float4(*(kernelContext_38->scratch_0+(at_4 + 2U))) ;
    float4 _S530 = float4(*(kernelContext_38->scratch_0+(at_4 + 3U))) ;
    WorldPoint_0 _S531 = chunk_world_0(c_16, kernelContext_38);
    float3 moved_0 = _S531.hi_0 - _S527.xyz + (_S531.lo_0 - _S528.xyz) + _S531.rel_0;
    uint _S532 = (uint4(_S524->info_1) ).y;
    float3 _S533 = float3((kernelContext_38->params_0->t_hi_0 - _S530.w + (kernelContext_38->params_0->t_lo_0 - _S528.w) + float((uint4((kernelContext_38->islands_0+_S532)->info_0) ).w - kernelContext_38->params_0->step_start_0) * kernelContext_38->params_0->dt_0)) ;
    float3 _S534 = _S530.xyz * _S533;
    uint _S535 = _S525.w;
    float3 _S536;
    if((_S535 & 2U) != 0U)
    {
        _S536 = float3(0.5f)  * kernelContext_38->params_0->gravity_0.xyz * _S533 * _S533;
    }
    else
    {
        _S536 = float3(0.0f) ;
    }
    float travel_0 = length(moved_0 - (_S534 + _S536));
    float travel_1;
    if((_S535 & 1U) != 0U)
    {
        float3 th_4 = (float4(*(kernelContext_38->state_0+(_S526 + 1U))) ).xyz;
        Quat_0 _S537 = from_axis_angle_0(th_4, length(th_4));
        thread Quat_0 _S538 = quat_of_0(float4((kernelContext_38->islands_0+_S532)->rotation_0) );
        thread Quat_0 _S539 = _S537;
        Quat_0 _S540 = quat_mul_0(&_S538, &_S539);
        thread Quat_0 _S541 = quat_of_0(float4(- _S529.xyz, _S529.w));
        thread Quat_0 _S542 = _S540;
        Quat_0 _S543 = quat_mul_0(&_S541, &_S542);
        travel_1 = travel_0 + 2.0f * atan2(length(float3(_S543.x_0, _S543.y_0, _S543.z_0)), abs(_S543.w_0)) * (float4(_S524->half_0) ).w;
    }
    else
    {
        travel_1 = travel_0;
    }
    if(travel_1 > (_S527.w))
    {
        travel_stop_0((uint4((kernelContext_38->islands_0+_S532)->info_0) ).w + 1U, fused_1, kernelContext_38);
    }
    return;
}

void island_frame_at_0(uint gi_0, uint tid_4, KernelContext_0 thread* kernelContext_39)
{
    uint c_17;
    bool woke_0;
    uint ct_1;
    if(tid_4 < 256U)
    {
        ct_1 = tid_4;
    }
    else
    {
        ct_1 = 1073741824U;
    }
    Island_natural_0 device* _S544 = kernelContext_39->islands_0+gi_0;
    uint4 _S545 = uint4((*_S544).info_0) ;
    float4 _S546 = float4((*_S544).com_0) ;
    float4 _S547 = float4((*_S544).inertia0_0) ;
    float4 _S548 = float4((*_S544).inertia1_0) ;
    float4 _S549 = float4((*_S544).inertia2_0) ;
    float4 _S550 = float4((*_S544).inv0_0) ;
    float4 _S551 = float4((*_S544).inv1_0) ;
    float4 _S552 = float4((*_S544).inv2_0) ;
    float4 _S553 = float4((*_S544).wcom_0) ;
    float4 _S554 = float4((*_S544).winv0_0) ;
    float4 _S555 = float4((*_S544).winv1_0) ;
    float4 _S556 = float4((*_S544).winv2_0) ;
    float4 _S557 = float4((*_S544).rotation_0) ;
    float4 _S558 = float4((*_S544).position_0) ;
    float4 _S559 = float4((*_S544).position_err_0) ;
    float4 _S560 = float4((*_S544).velocity_0) ;
    float4 _S561 = float4((*_S544).velocity_err_0) ;
    float4 _S562 = float4((*_S544).angular_velocity_0) ;
    uint4 _S563 = uint4((*_S544).probes_0) ;
    float4 _S564 = float4((*_S544).energy_0) ;
    float4 _S565 = float4((*_S544).rotation_err_0) ;
    float4 _S566 = float4((*_S544).momentum_0) ;
    float4 _S567 = float4((*_S544).momentum_err_0) ;
    thread Island_0 isl_9;
    (&isl_9)->range_0 = uint4((*_S544).range_0) ;
    (&isl_9)->info_0 = _S545;
    (&isl_9)->com_0 = _S546;
    (&isl_9)->inertia0_0 = _S547;
    (&isl_9)->inertia1_0 = _S548;
    (&isl_9)->inertia2_0 = _S549;
    (&isl_9)->inv0_0 = _S550;
    (&isl_9)->inv1_0 = _S551;
    (&isl_9)->inv2_0 = _S552;
    (&isl_9)->wcom_0 = _S553;
    (&isl_9)->winv0_0 = _S554;
    (&isl_9)->winv1_0 = _S555;
    (&isl_9)->winv2_0 = _S556;
    (&isl_9)->rotation_0 = _S557;
    (&isl_9)->position_0 = _S558;
    (&isl_9)->position_err_0 = _S559;
    (&isl_9)->velocity_0 = _S560;
    (&isl_9)->velocity_err_0 = _S561;
    (&isl_9)->angular_velocity_0 = _S562;
    (&isl_9)->done_0 = (*_S544).done_0;
    (&isl_9)->coverage_0 = (*_S544).coverage_0;
    (&isl_9)->wide_groups_0 = (*_S544).wide_groups_0;
    (&isl_9)->done_pad_0 = (*_S544).done_pad_0;
    (&isl_9)->probes_0 = _S563;
    (&isl_9)->energy_0 = _S564;
    (&isl_9)->rotation_err_0 = _S565;
    (&isl_9)->momentum_0 = _S566;
    (&isl_9)->momentum_err_0 = _S567;
    bool driven_0 = (((&isl_9)->info_0.x) & 2U) != 0U;
    bool _S568 = !((((&isl_9)->info_0.x) & 1U) != 0U);
    bool _S569;
    if(_S568)
    {
        _S569 = !driven_0;
    }
    else
    {
        _S569 = false;
    }
    bool contact_island_0 = (((&isl_9)->info_0.x) & 4U) != 0U;
    bool _S570 = (((&isl_9)->info_0.x) & 16U) != 0U;
    seen_clear_0(tid_4, kernelContext_39);
    bool _S571 = tid_4 == 0U;
    uint cov_0;
    bool settled_0;
    if(_S571)
    {
        if(contact_island_0 != ((kernelContext_39->params_0->contact_mode_0) == 1U))
        {
            settled_0 = true;
        }
        else
        {
            settled_0 = (((&isl_9)->info_0.x) & 8U) != 0U;
        }
        if(settled_0)
        {
            cov_0 = 0U;
        }
        else
        {
            cov_0 = 1U;
        }
        if(contact_island_0)
        {
            thread Island_0 _S572 = isl_9;
            bool _S573 = contact_stopped_0(&_S572, kernelContext_39);
            settled_0 = _S573;
        }
        else
        {
            settled_0 = false;
        }
        if(settled_0)
        {
            cov_0 = 0U;
        }
        *kernelContext_39->g_run_0 = cov_0;
        *kernelContext_39->g_halt_0 = 0U;
        if(contact_island_0)
        {
            Island_natural_0 sentinel_0 = *(kernelContext_39->islands_0+kernelContext_39->params_0->halt_index_0);
            stop_seen_0(kernelContext_39->params_0->halt_index_0, (&isl_9)->info_0.w + 1U, kernelContext_39);
            uint _S574 = (uint4(sentinel_0.info_0) ).y;
            if(_S574 != 0U)
            {
                settled_0 = _S574 <= ((&isl_9)->info_0.w);
            }
            else
            {
                settled_0 = false;
            }
            int _S575;
            if(settled_0)
            {
                _S575 = int(0);
            }
            else
            {
                _S575 = int(1);
            }
            *kernelContext_39->g_sums_0 = uint(_S575);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((((&isl_9)->info_0.z) & 1U) != 0U)
    {
        settled_0 = true;
    }
    else
    {
        settled_0 = (*kernelContext_39->g_run_0) == 0U;
    }
    if(settled_0)
    {
        cov_0 = 0U;
    }
    else
    {
        cov_0 = min((&isl_9)->info_0.y, kernelContext_39->params_0->max_steps_0);
    }
    float _S576 = kernelContext_39->params_0->dt_0;
    bool _S577 = (kernelContext_39->params_0->fracture_0) != 0U;
    bool _S578 = (kernelContext_39->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S579 = isl_9;
    Rigid_0 _S580 = rigid_of_0(&_S579);
    thread Rigid_0 rg_7 = _S580;
    thread float work_2 = 0.0f;
    thread float work_err_1 = 0.0f;
    settled_0 = _S570;
    bool woke_1 = false;
    uint done_1 = 0U;
    bool resumed_0 = false;
    for(;;)
    {
        if(done_1 < cov_0)
        {
        }
        else
        {
            woke_0 = woke_1;
            break;
        }
        uint abs_step_1 = (&isl_9)->info_0.w + done_1 + 1U;
        uint k_12 = abs_step_1 - 1U - kernelContext_39->params_0->step_start_0;
        bool quiet_0 = (((&isl_9)->info_0.x) & 32U) != 0U;
        bool _S581;
        if(contact_island_0)
        {
            _S581 = (*kernelContext_39->g_sums_0) != 0U;
        }
        else
        {
            _S581 = false;
        }
        if(_S581)
        {
            uint _S582 = (uint4((kernelContext_39->chunks_0+((&isl_9)->range_0.y - 1U))->cinfo_0) ).y;
            c_17 = (uint4((kernelContext_39->chunks_0+(&isl_9)->range_0.x)->cinfo_0) ).x + ct_1;
            for(;;)
            {
                if(c_17 < _S582)
                {
                }
                else
                {
                    break;
                }
                contact_segment_0(c_17, kernelContext_39);
                c_17 = c_17 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        bool _S583;
        if(settled_0)
        {
            _S583 = !quiet_0;
        }
        else
        {
            _S583 = false;
        }
        bool _S584;
        bool explicit_step_0;
        if(_S583)
        {
            float3 _S585 = float3(0.0f) ;
            thread float3 norm_0 = _S585;
            thread float3 unused0_0 = _S585;
            c_17 = (&isl_9)->range_0.x + ct_1;
            for(;;)
            {
                if(c_17 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S586 = (&rg_7)->rot_0;
                float _S587 = settled_chunk_load_0(c_17, &_S586, k_12, _S576, contact_island_0, kernelContext_39);
                norm_0.x = norm_0.x + _S587;
                c_17 = c_17 + 256U;
            }
            group_sum3_0(tid_4, &norm_0, &unused0_0, kernelContext_39);
            if((kernelContext_39->params_0->solve_mode_0) == 1U)
            {
                _S584 = (abs(norm_0.x - (&isl_9)->energy_0.z)) > ((&isl_9)->energy_0.w);
            }
            else
            {
                _S584 = false;
            }
            if(_S584)
            {
                explicit_step_0 = false;
                woke_0 = true;
            }
            else
            {
                explicit_step_0 = settled_0;
                woke_0 = woke_1;
            }
            settled_0 = explicit_step_0;
        }
        else
        {
            woke_0 = woke_1;
        }
        if(_S568)
        {
            _S584 = !quiet_0;
        }
        else
        {
            _S584 = false;
        }
        if(_S584)
        {
            float3 _S588 = float3(0.0f) ;
            thread float3 f_11 = _S588;
            thread float3 t_13 = _S588;
            c_17 = (&isl_9)->range_0.x + ct_1;
            for(;;)
            {
                if(c_17 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S589 = isl_9;
                thread Rigid_0 _S590 = rg_7;
                net_load_0(c_17, &_S589, &_S590, k_12, _S576, contact_island_0, &f_11, &t_13, kernelContext_39);
                c_17 = c_17 + 256U;
            }
            group_sum3_0(tid_4, &f_11, &t_13, kernelContext_39);
            thread Island_0 _S591 = isl_9;
            rigid_acceleration_0(&_S591, &rg_7, f_11, t_13);
        }
        if(!settled_0)
        {
            explicit_step_0 = !quiet_0;
        }
        else
        {
            explicit_step_0 = false;
        }
        bool _S592;
        if(explicit_step_0)
        {
            if(woke_0)
            {
                c_17 = (&isl_9)->range_0.z + tid_4;
                for(;;)
                {
                    if(c_17 < ((&isl_9)->range_0.w))
                    {
                    }
                    else
                    {
                        break;
                    }
                    bool _S593 = bond_update_0(c_17, _S576, _S577, abs_step_1, false, kernelContext_39);
                    if(_S593)
                    {
                        *kernelContext_39->g_halt_0 = 1U;
                    }
                    c_17 = c_17 + 256U;
                }
            }
            else
            {
                if(_S571)
                {
                    _S592 = (((uint4((kernelContext_39->islands_0+gi_0)->info_0) ).z) & 2U) != 0U;
                }
                else
                {
                    _S592 = false;
                }
                if(_S592)
                {
                    *kernelContext_39->g_halt_0 = 1U;
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            c_17 = (&isl_9)->range_0.x + ct_1;
            for(;;)
            {
                if(c_17 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S594 = isl_9;
                thread Rigid_0 _S595 = rg_7;
                chunk_update_0(c_17, &_S594, &_S595, _S576, _S578, k_12, contact_island_0, &work_2, &work_err_1, kernelContext_39);
                c_17 = c_17 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        if(_S569)
        {
            _S592 = !quiet_0;
        }
        else
        {
            _S592 = false;
        }
        if(_S592)
        {
            thread Island_0 _S596 = isl_9;
            integrate_rigid_0(&_S596, &rg_7, _S576);
        }
        bool _S597;
        if(_S568)
        {
            _S597 = explicit_step_0;
        }
        else
        {
            _S597 = false;
        }
        if(_S597)
        {
            float3 _S598 = (&isl_9)->wcom_0.xyz;
            float3 _S599 = float3(0.0f) ;
            thread float3 tu_1 = _S599;
            thread float3 pv_1 = _S599;
            c_17 = (&isl_9)->range_0.x + ct_1;
            for(;;)
            {
                if(c_17 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_17, &tu_1, &pv_1, kernelContext_39);
                c_17 = c_17 + 256U;
            }
            group_sum3_0(tid_4, &tu_1, &pv_1, kernelContext_39);
            float3 tr_3 = tu_1 / float3((&isl_9)->wcom_0.w) ;
            float3 dv_3 = pv_1 / float3((&isl_9)->wcom_0.w) ;
            thread float3 lu_1 = _S599;
            thread float3 lv_1 = _S599;
            uint c_18 = (&isl_9)->range_0.x + ct_1;
            for(;;)
            {
                if(c_18 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_18, _S598, tr_3, dv_3, &lu_1, &lv_1, kernelContext_39);
                c_18 = c_18 + 256U;
            }
            group_sum3_0(tid_4, &lu_1, &lv_1, kernelContext_39);
            float3 phi_3 = rows_mul_0((&isl_9)->winv0_0, (&isl_9)->winv1_0, (&isl_9)->winv2_0, lu_1);
            float3 dw_2 = rows_mul_0((&isl_9)->winv0_0, (&isl_9)->winv1_0, (&isl_9)->winv2_0, lv_1);
            uint c_19 = (&isl_9)->range_0.x + ct_1;
            for(;;)
            {
                if(c_19 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_19, _S598, tr_3, phi_3, dv_3, dw_2, kernelContext_39);
                c_19 = c_19 + 256U;
            }
            if(!driven_0)
            {
                thread Island_0 _S600 = isl_9;
                drift_rigid_0(&_S600, &rg_7, tr_3, phi_3, dv_3, dw_2);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        bool _S601;
        if(_S571)
        {
            _S601 = ((&isl_9)->probes_0.y) > ((&isl_9)->probes_0.x);
        }
        else
        {
            _S601 = false;
        }
        if(_S601)
        {
            thread Island_0 _S602 = isl_9;
            thread Rigid_0 _S603 = rg_7;
            record_probes_0(&_S602, &_S603, k_12, kernelContext_39);
        }
        uint _S604 = done_1 + 1U;
        bool _S605;
        if((*kernelContext_39->g_halt_0) != 0U)
        {
            thread Island_0 _S606 = isl_9;
            bool _S607 = island_connected_narrow_0(tid_4, ct_1, &_S606, kernelContext_39);
            _S605 = _S607;
        }
        else
        {
            _S605 = false;
        }
        bool resumed_1;
        if(_S605)
        {
            if(_S571)
            {
                *kernelContext_39->g_halt_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            resumed_1 = true;
        }
        else
        {
            resumed_1 = resumed_0;
        }
        if((*kernelContext_39->g_halt_0) != 0U)
        {
            done_1 = _S604;
            resumed_0 = resumed_1;
            break;
        }
        woke_1 = woke_0;
        done_1 = _S604;
        resumed_0 = resumed_1;
    }
    thread float3 wsum_0 = float3(work_2, work_err_1, 0.0f);
    thread float3 unused_0 = float3(0.0f) ;
    group_sum3_0(tid_4, &wsum_0, &unused_0, kernelContext_39);
    if(_S571)
    {
        thread Quat_0 _S608 = (&rg_7)->rot_0;
        float4 _S609 = quat_vec_0(&_S608);
        (&isl_9)->rotation_0 = _S609;
        (&isl_9)->rotation_err_0 = (&rg_7)->rot_err_0;
        (&isl_9)->position_0 = float4((&rg_7)->pos_0, 0.0f);
        (&isl_9)->position_err_0 = float4((&rg_7)->pos_err_0, 0.0f);
        (&isl_9)->velocity_0 = float4((&rg_7)->vel_0, 0.0f);
        (&isl_9)->velocity_err_0 = float4((&rg_7)->vel_err_0, 0.0f);
        (&isl_9)->angular_velocity_0 = float4((&rg_7)->w_1, 0.0f);
        (&isl_9)->momentum_0 = float4((&rg_7)->l_0, 0.0f);
        (&isl_9)->momentum_err_0 = float4((&rg_7)->l_err_0, 0.0f);
        (&isl_9)->done_0 = done_1;
        (&isl_9)->info_0.y = (&isl_9)->info_0.y - done_1;
        if((*kernelContext_39->g_halt_0) != 0U)
        {
            _S569 = contact_island_0;
        }
        else
        {
            _S569 = false;
        }
        if(_S569)
        {
            contact_split_at_0((&isl_9)->info_0.w + done_1, kernelContext_39);
        }
        float _S610 = wsum_0.x;
        thread float _S611 = (&isl_9)->energy_0.x;
        thread float _S612 = (&isl_9)->energy_0.y;
        comp_add1_0(&_S611, &_S612, _S610);
        (&isl_9)->energy_0.x = _S611;
        (&isl_9)->energy_0.y = _S612 + wsum_0.y;
        (&isl_9)->info_0.w = (&isl_9)->info_0.w + done_1;
        if((((&isl_9)->info_0.x) & 8U) == 0U)
        {
            (&isl_9)->info_0.z = ((&isl_9)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_39->g_halt_0) != 0U)
        {
            (&isl_9)->info_0.z = ((&isl_9)->info_0.z) | 1U;
        }
        uint _S613 = seen_bits_0(kernelContext_39);
        if((*kernelContext_39->g_halt_0) != 0U)
        {
            cov_0 = 128U;
        }
        else
        {
            cov_0 = 0U;
        }
        uint _S614 = _S613 | cov_0;
        if(woke_0)
        {
            cov_0 = 2048U;
        }
        else
        {
            cov_0 = 0U;
        }
        uint _S615 = _S614 | cov_0;
        if(resumed_0)
        {
            cov_0 = 2097280U;
        }
        else
        {
            cov_0 = 0U;
        }
        uint cov_1 = _S615 | cov_0;
        if(done_1 > 0U)
        {
            bool quiet_1 = (((&isl_9)->info_0.x) & 32U) != 0U;
            if(quiet_1)
            {
                cov_0 = 4096U;
            }
            else
            {
                cov_0 = 0U;
            }
            uint cov_2 = cov_1 | cov_0;
            if((((&isl_9)->info_0.x) & 16U) != 0U)
            {
                _S569 = !quiet_1;
            }
            else
            {
                _S569 = false;
            }
            if(_S569)
            {
                cov_0 = 512U;
            }
            else
            {
                cov_0 = 0U;
            }
            uint cov_3 = cov_2 | cov_0;
            if((kernelContext_39->params_0->contact_mode_0) == 1U)
            {
                cov_0 = 524288U;
            }
            else
            {
                cov_0 = 0U;
            }
            uint cov_4 = cov_3 | cov_0;
            if(((&isl_9)->probes_0.y) > ((&isl_9)->probes_0.x))
            {
                cov_0 = 262144U;
            }
            else
            {
                cov_0 = 0U;
            }
            cov_0 = cov_4 | cov_0;
        }
        else
        {
            cov_0 = cov_1;
        }
        (&isl_9)->coverage_0 = ((&isl_9)->coverage_0) | cov_0;
        if(woke_0)
        {
            (&isl_9)->info_0.x = ((&isl_9)->info_0.x) & 4294967279U;
            (&isl_9)->info_0.z = ((&isl_9)->info_0.z) | 4U;
        }
        Island_natural_0 device* _S616 = kernelContext_39->islands_0+gi_0;
        _S616->range_0 = packed_uint4(isl_9.range_0) ;
        _S616->info_0 = packed_uint4(isl_9.info_0) ;
        _S616->com_0 = packed_float4(isl_9.com_0) ;
        _S616->inertia0_0 = packed_float4(isl_9.inertia0_0) ;
        _S616->inertia1_0 = packed_float4(isl_9.inertia1_0) ;
        _S616->inertia2_0 = packed_float4(isl_9.inertia2_0) ;
        _S616->inv0_0 = packed_float4(isl_9.inv0_0) ;
        _S616->inv1_0 = packed_float4(isl_9.inv1_0) ;
        _S616->inv2_0 = packed_float4(isl_9.inv2_0) ;
        _S616->wcom_0 = packed_float4(isl_9.wcom_0) ;
        _S616->winv0_0 = packed_float4(isl_9.winv0_0) ;
        _S616->winv1_0 = packed_float4(isl_9.winv1_0) ;
        _S616->winv2_0 = packed_float4(isl_9.winv2_0) ;
        _S616->rotation_0 = packed_float4(isl_9.rotation_0) ;
        _S616->position_0 = packed_float4(isl_9.position_0) ;
        _S616->position_err_0 = packed_float4(isl_9.position_err_0) ;
        _S616->velocity_0 = packed_float4(isl_9.velocity_0) ;
        _S616->velocity_err_0 = packed_float4(isl_9.velocity_err_0) ;
        _S616->angular_velocity_0 = packed_float4(isl_9.angular_velocity_0) ;
        _S616->done_0 = isl_9.done_0;
        _S616->coverage_0 = isl_9.coverage_0;
        _S616->wide_groups_0 = isl_9.wide_groups_0;
        _S616->done_pad_0 = isl_9.done_pad_0;
        _S616->probes_0 = packed_uint4(isl_9.probes_0) ;
        _S616->energy_0 = packed_float4(isl_9.energy_0) ;
        _S616->rotation_err_0 = packed_float4(isl_9.rotation_err_0) ;
        _S616->momentum_0 = packed_float4(isl_9.momentum_0) ;
        _S616->momentum_err_0 = packed_float4(isl_9.momentum_err_0) ;
    }
    if(contact_island_0)
    {
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        if(done_1 > 0U)
        {
            c_17 = (&isl_9)->range_0.x + ct_1;
            for(;;)
            {
                if(c_17 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                travel_check_0(c_17, true, kernelContext_39);
                c_17 = c_17 + 256U;
            }
        }
    }
    return;
}

void island_frame_at_1(uint gi_1, uint tid_5, KernelContext_0 thread* kernelContext_40)
{
    bool woke_2;
    uint ct_2;
    if(tid_5 < 256U)
    {
        ct_2 = tid_5;
    }
    else
    {
        ct_2 = 1073741824U;
    }
    Island_natural_0 device* _S617 = kernelContext_40->islands_0+gi_1;
    uint4 _S618 = uint4((*_S617).info_0) ;
    float4 _S619 = float4((*_S617).com_0) ;
    float4 _S620 = float4((*_S617).inertia0_0) ;
    float4 _S621 = float4((*_S617).inertia1_0) ;
    float4 _S622 = float4((*_S617).inertia2_0) ;
    float4 _S623 = float4((*_S617).inv0_0) ;
    float4 _S624 = float4((*_S617).inv1_0) ;
    float4 _S625 = float4((*_S617).inv2_0) ;
    float4 _S626 = float4((*_S617).wcom_0) ;
    float4 _S627 = float4((*_S617).winv0_0) ;
    float4 _S628 = float4((*_S617).winv1_0) ;
    float4 _S629 = float4((*_S617).winv2_0) ;
    float4 _S630 = float4((*_S617).rotation_0) ;
    float4 _S631 = float4((*_S617).position_0) ;
    float4 _S632 = float4((*_S617).position_err_0) ;
    float4 _S633 = float4((*_S617).velocity_0) ;
    float4 _S634 = float4((*_S617).velocity_err_0) ;
    float4 _S635 = float4((*_S617).angular_velocity_0) ;
    uint4 _S636 = uint4((*_S617).probes_0) ;
    float4 _S637 = float4((*_S617).energy_0) ;
    float4 _S638 = float4((*_S617).rotation_err_0) ;
    float4 _S639 = float4((*_S617).momentum_0) ;
    float4 _S640 = float4((*_S617).momentum_err_0) ;
    thread Island_0 isl_10;
    (&isl_10)->range_0 = uint4((*_S617).range_0) ;
    (&isl_10)->info_0 = _S618;
    (&isl_10)->com_0 = _S619;
    (&isl_10)->inertia0_0 = _S620;
    (&isl_10)->inertia1_0 = _S621;
    (&isl_10)->inertia2_0 = _S622;
    (&isl_10)->inv0_0 = _S623;
    (&isl_10)->inv1_0 = _S624;
    (&isl_10)->inv2_0 = _S625;
    (&isl_10)->wcom_0 = _S626;
    (&isl_10)->winv0_0 = _S627;
    (&isl_10)->winv1_0 = _S628;
    (&isl_10)->winv2_0 = _S629;
    (&isl_10)->rotation_0 = _S630;
    (&isl_10)->position_0 = _S631;
    (&isl_10)->position_err_0 = _S632;
    (&isl_10)->velocity_0 = _S633;
    (&isl_10)->velocity_err_0 = _S634;
    (&isl_10)->angular_velocity_0 = _S635;
    (&isl_10)->done_0 = (*_S617).done_0;
    (&isl_10)->coverage_0 = (*_S617).coverage_0;
    (&isl_10)->wide_groups_0 = (*_S617).wide_groups_0;
    (&isl_10)->done_pad_0 = (*_S617).done_pad_0;
    (&isl_10)->probes_0 = _S636;
    (&isl_10)->energy_0 = _S637;
    (&isl_10)->rotation_err_0 = _S638;
    (&isl_10)->momentum_0 = _S639;
    (&isl_10)->momentum_err_0 = _S640;
    bool driven_1 = (((&isl_10)->info_0.x) & 2U) != 0U;
    bool _S641 = !((((&isl_10)->info_0.x) & 1U) != 0U);
    bool _S642;
    if(_S641)
    {
        _S642 = !driven_1;
    }
    else
    {
        _S642 = false;
    }
    bool contact_island_1 = (((&isl_10)->info_0.x) & 4U) != 0U;
    bool _S643 = (((&isl_10)->info_0.x) & 16U) != 0U;
    seen_clear_0(tid_5, kernelContext_40);
    bool _S644 = tid_5 == 0U;
    uint cov_5;
    bool settled_1;
    if(_S644)
    {
        if(contact_island_1 != ((kernelContext_40->params_0->contact_mode_0) == 1U))
        {
            settled_1 = true;
        }
        else
        {
            settled_1 = (((&isl_10)->info_0.x) & 8U) != 0U;
        }
        if(settled_1)
        {
            cov_5 = 0U;
        }
        else
        {
            cov_5 = 1U;
        }
        if(contact_island_1)
        {
            thread Island_0 _S645 = isl_10;
            bool _S646 = contact_stopped_0(&_S645, kernelContext_40);
            settled_1 = _S646;
        }
        else
        {
            settled_1 = false;
        }
        if(settled_1)
        {
            cov_5 = 0U;
        }
        *kernelContext_40->g_run_0 = cov_5;
        *kernelContext_40->g_halt_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((((&isl_10)->info_0.z) & 1U) != 0U)
    {
        settled_1 = true;
    }
    else
    {
        settled_1 = (*kernelContext_40->g_run_0) == 0U;
    }
    if(settled_1)
    {
        cov_5 = 0U;
    }
    else
    {
        cov_5 = min((&isl_10)->info_0.y, kernelContext_40->params_0->max_steps_0);
    }
    float _S647 = kernelContext_40->params_0->dt_0;
    bool _S648 = (kernelContext_40->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S649 = isl_10;
    Rigid_0 _S650 = rigid_of_0(&_S649);
    thread Rigid_0 rg_8 = _S650;
    thread float work_3 = 0.0f;
    thread float work_err_2 = 0.0f;
    settled_1 = _S643;
    bool woke_3 = false;
    uint done_2 = 0U;
    bool resumed_2 = false;
    for(;;)
    {
        if(done_2 < cov_5)
        {
        }
        else
        {
            woke_2 = woke_3;
            break;
        }
        uint k_13 = (&isl_10)->info_0.w + done_2 + 1U - 1U - kernelContext_40->params_0->step_start_0;
        bool quiet_2 = (((&isl_10)->info_0.x) & 32U) != 0U;
        bool _S651;
        if(settled_1)
        {
            _S651 = !quiet_2;
        }
        else
        {
            _S651 = false;
        }
        uint c_20;
        bool _S652;
        bool explicit_step_1;
        if(_S651)
        {
            float3 _S653 = float3(0.0f) ;
            thread float3 norm_1 = _S653;
            thread float3 unused0_1 = _S653;
            c_20 = (&isl_10)->range_0.x + ct_2;
            for(;;)
            {
                if(c_20 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S654 = (&rg_8)->rot_0;
                float _S655 = settled_chunk_load_0(c_20, &_S654, k_13, _S647, contact_island_1, kernelContext_40);
                norm_1.x = norm_1.x + _S655;
                c_20 = c_20 + 256U;
            }
            group_sum3_0(tid_5, &norm_1, &unused0_1, kernelContext_40);
            if((kernelContext_40->params_0->solve_mode_0) == 1U)
            {
                _S652 = (abs(norm_1.x - (&isl_10)->energy_0.z)) > ((&isl_10)->energy_0.w);
            }
            else
            {
                _S652 = false;
            }
            if(_S652)
            {
                explicit_step_1 = false;
                woke_2 = true;
            }
            else
            {
                explicit_step_1 = settled_1;
                woke_2 = woke_3;
            }
            settled_1 = explicit_step_1;
        }
        else
        {
            woke_2 = woke_3;
        }
        if(_S641)
        {
            _S652 = !quiet_2;
        }
        else
        {
            _S652 = false;
        }
        if(_S652)
        {
            float3 _S656 = float3(0.0f) ;
            thread float3 f_12 = _S656;
            thread float3 t_14 = _S656;
            c_20 = (&isl_10)->range_0.x + ct_2;
            for(;;)
            {
                if(c_20 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S657 = isl_10;
                thread Rigid_0 _S658 = rg_8;
                net_load_0(c_20, &_S657, &_S658, k_13, _S647, contact_island_1, &f_12, &t_14, kernelContext_40);
                c_20 = c_20 + 256U;
            }
            group_sum3_0(tid_5, &f_12, &t_14, kernelContext_40);
            thread Island_0 _S659 = isl_10;
            rigid_acceleration_0(&_S659, &rg_8, f_12, t_14);
        }
        if(!settled_1)
        {
            explicit_step_1 = !quiet_2;
        }
        else
        {
            explicit_step_1 = false;
        }
        bool _S660;
        if(explicit_step_1)
        {
            if(_S644)
            {
                _S660 = (((uint4((kernelContext_40->islands_0+gi_1)->info_0) ).z) & 2U) != 0U;
            }
            else
            {
                _S660 = false;
            }
            if(_S660)
            {
                *kernelContext_40->g_halt_0 = 1U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            c_20 = (&isl_10)->range_0.x + ct_2;
            for(;;)
            {
                if(c_20 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S661 = isl_10;
                thread Rigid_0 _S662 = rg_8;
                chunk_update_0(c_20, &_S661, &_S662, _S647, _S648, k_13, contact_island_1, &work_3, &work_err_2, kernelContext_40);
                c_20 = c_20 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        if(_S642)
        {
            _S660 = !quiet_2;
        }
        else
        {
            _S660 = false;
        }
        if(_S660)
        {
            thread Island_0 _S663 = isl_10;
            integrate_rigid_0(&_S663, &rg_8, _S647);
        }
        bool _S664;
        if(_S641)
        {
            _S664 = explicit_step_1;
        }
        else
        {
            _S664 = false;
        }
        if(_S664)
        {
            float3 _S665 = (&isl_10)->wcom_0.xyz;
            float3 _S666 = float3(0.0f) ;
            thread float3 tu_2 = _S666;
            thread float3 pv_2 = _S666;
            c_20 = (&isl_10)->range_0.x + ct_2;
            for(;;)
            {
                if(c_20 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_20, &tu_2, &pv_2, kernelContext_40);
                c_20 = c_20 + 256U;
            }
            group_sum3_0(tid_5, &tu_2, &pv_2, kernelContext_40);
            float3 tr_4 = tu_2 / float3((&isl_10)->wcom_0.w) ;
            float3 dv_4 = pv_2 / float3((&isl_10)->wcom_0.w) ;
            thread float3 lu_2 = _S666;
            thread float3 lv_2 = _S666;
            uint c_21 = (&isl_10)->range_0.x + ct_2;
            for(;;)
            {
                if(c_21 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_21, _S665, tr_4, dv_4, &lu_2, &lv_2, kernelContext_40);
                c_21 = c_21 + 256U;
            }
            group_sum3_0(tid_5, &lu_2, &lv_2, kernelContext_40);
            float3 phi_4 = rows_mul_0((&isl_10)->winv0_0, (&isl_10)->winv1_0, (&isl_10)->winv2_0, lu_2);
            float3 dw_3 = rows_mul_0((&isl_10)->winv0_0, (&isl_10)->winv1_0, (&isl_10)->winv2_0, lv_2);
            uint c_22 = (&isl_10)->range_0.x + ct_2;
            for(;;)
            {
                if(c_22 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_22, _S665, tr_4, phi_4, dv_4, dw_3, kernelContext_40);
                c_22 = c_22 + 256U;
            }
            if(!driven_1)
            {
                thread Island_0 _S667 = isl_10;
                drift_rigid_0(&_S667, &rg_8, tr_4, phi_4, dv_4, dw_3);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        bool _S668;
        if(_S644)
        {
            _S668 = ((&isl_10)->probes_0.y) > ((&isl_10)->probes_0.x);
        }
        else
        {
            _S668 = false;
        }
        if(_S668)
        {
            thread Island_0 _S669 = isl_10;
            thread Rigid_0 _S670 = rg_8;
            record_probes_0(&_S669, &_S670, k_13, kernelContext_40);
        }
        uint _S671 = done_2 + 1U;
        bool _S672;
        if((*kernelContext_40->g_halt_0) != 0U)
        {
            thread Island_0 _S673 = isl_10;
            bool _S674 = island_connected_narrow_0(tid_5, ct_2, &_S673, kernelContext_40);
            _S672 = _S674;
        }
        else
        {
            _S672 = false;
        }
        bool resumed_3;
        if(_S672)
        {
            if(_S644)
            {
                *kernelContext_40->g_halt_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            resumed_3 = true;
        }
        else
        {
            resumed_3 = resumed_2;
        }
        if((*kernelContext_40->g_halt_0) != 0U)
        {
            done_2 = _S671;
            resumed_2 = resumed_3;
            break;
        }
        woke_3 = woke_2;
        done_2 = _S671;
        resumed_2 = resumed_3;
    }
    thread float3 wsum_1 = float3(work_3, work_err_2, 0.0f);
    thread float3 unused_1 = float3(0.0f) ;
    group_sum3_0(tid_5, &wsum_1, &unused_1, kernelContext_40);
    if(_S644)
    {
        thread Quat_0 _S675 = (&rg_8)->rot_0;
        float4 _S676 = quat_vec_0(&_S675);
        (&isl_10)->rotation_0 = _S676;
        (&isl_10)->rotation_err_0 = (&rg_8)->rot_err_0;
        (&isl_10)->position_0 = float4((&rg_8)->pos_0, 0.0f);
        (&isl_10)->position_err_0 = float4((&rg_8)->pos_err_0, 0.0f);
        (&isl_10)->velocity_0 = float4((&rg_8)->vel_0, 0.0f);
        (&isl_10)->velocity_err_0 = float4((&rg_8)->vel_err_0, 0.0f);
        (&isl_10)->angular_velocity_0 = float4((&rg_8)->w_1, 0.0f);
        (&isl_10)->momentum_0 = float4((&rg_8)->l_0, 0.0f);
        (&isl_10)->momentum_err_0 = float4((&rg_8)->l_err_0, 0.0f);
        (&isl_10)->done_0 = done_2;
        (&isl_10)->info_0.y = (&isl_10)->info_0.y - done_2;
        if((*kernelContext_40->g_halt_0) != 0U)
        {
            _S642 = contact_island_1;
        }
        else
        {
            _S642 = false;
        }
        if(_S642)
        {
            contact_split_at_0((&isl_10)->info_0.w + done_2, kernelContext_40);
        }
        float _S677 = wsum_1.x;
        thread float _S678 = (&isl_10)->energy_0.x;
        thread float _S679 = (&isl_10)->energy_0.y;
        comp_add1_0(&_S678, &_S679, _S677);
        (&isl_10)->energy_0.x = _S678;
        (&isl_10)->energy_0.y = _S679 + wsum_1.y;
        (&isl_10)->info_0.w = (&isl_10)->info_0.w + done_2;
        if((((&isl_10)->info_0.x) & 8U) == 0U)
        {
            (&isl_10)->info_0.z = ((&isl_10)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_40->g_halt_0) != 0U)
        {
            (&isl_10)->info_0.z = ((&isl_10)->info_0.z) | 1U;
        }
        uint _S680 = seen_bits_0(kernelContext_40);
        if((*kernelContext_40->g_halt_0) != 0U)
        {
            ct_2 = 128U;
        }
        else
        {
            ct_2 = 0U;
        }
        uint _S681 = _S680 | ct_2;
        if(woke_2)
        {
            ct_2 = 2048U;
        }
        else
        {
            ct_2 = 0U;
        }
        uint _S682 = _S681 | ct_2;
        if(resumed_2)
        {
            ct_2 = 2097280U;
        }
        else
        {
            ct_2 = 0U;
        }
        uint cov_6 = _S682 | ct_2;
        if(done_2 > 0U)
        {
            bool quiet_3 = (((&isl_10)->info_0.x) & 32U) != 0U;
            if(quiet_3)
            {
                ct_2 = 4096U;
            }
            else
            {
                ct_2 = 0U;
            }
            uint cov_7 = cov_6 | ct_2;
            if((((&isl_10)->info_0.x) & 16U) != 0U)
            {
                _S642 = !quiet_3;
            }
            else
            {
                _S642 = false;
            }
            if(_S642)
            {
                ct_2 = 512U;
            }
            else
            {
                ct_2 = 0U;
            }
            uint cov_8 = cov_7 | ct_2;
            if((kernelContext_40->params_0->contact_mode_0) == 1U)
            {
                ct_2 = 524288U;
            }
            else
            {
                ct_2 = 0U;
            }
            uint cov_9 = cov_8 | ct_2;
            if(((&isl_10)->probes_0.y) > ((&isl_10)->probes_0.x))
            {
                ct_2 = 262144U;
            }
            else
            {
                ct_2 = 0U;
            }
            cov_5 = cov_9 | ct_2;
        }
        else
        {
            cov_5 = cov_6;
        }
        (&isl_10)->coverage_0 = ((&isl_10)->coverage_0) | cov_5;
        if(woke_2)
        {
            (&isl_10)->info_0.x = ((&isl_10)->info_0.x) & 4294967279U;
            (&isl_10)->info_0.z = ((&isl_10)->info_0.z) | 4U;
        }
        Island_natural_0 device* _S683 = kernelContext_40->islands_0+gi_1;
        _S683->range_0 = packed_uint4(isl_10.range_0) ;
        _S683->info_0 = packed_uint4(isl_10.info_0) ;
        _S683->com_0 = packed_float4(isl_10.com_0) ;
        _S683->inertia0_0 = packed_float4(isl_10.inertia0_0) ;
        _S683->inertia1_0 = packed_float4(isl_10.inertia1_0) ;
        _S683->inertia2_0 = packed_float4(isl_10.inertia2_0) ;
        _S683->inv0_0 = packed_float4(isl_10.inv0_0) ;
        _S683->inv1_0 = packed_float4(isl_10.inv1_0) ;
        _S683->inv2_0 = packed_float4(isl_10.inv2_0) ;
        _S683->wcom_0 = packed_float4(isl_10.wcom_0) ;
        _S683->winv0_0 = packed_float4(isl_10.winv0_0) ;
        _S683->winv1_0 = packed_float4(isl_10.winv1_0) ;
        _S683->winv2_0 = packed_float4(isl_10.winv2_0) ;
        _S683->rotation_0 = packed_float4(isl_10.rotation_0) ;
        _S683->position_0 = packed_float4(isl_10.position_0) ;
        _S683->position_err_0 = packed_float4(isl_10.position_err_0) ;
        _S683->velocity_0 = packed_float4(isl_10.velocity_0) ;
        _S683->velocity_err_0 = packed_float4(isl_10.velocity_err_0) ;
        _S683->angular_velocity_0 = packed_float4(isl_10.angular_velocity_0) ;
        _S683->done_0 = isl_10.done_0;
        _S683->coverage_0 = isl_10.coverage_0;
        _S683->wide_groups_0 = isl_10.wide_groups_0;
        _S683->done_pad_0 = isl_10.done_pad_0;
        _S683->probes_0 = packed_uint4(isl_10.probes_0) ;
        _S683->energy_0 = packed_float4(isl_10.energy_0) ;
        _S683->rotation_err_0 = packed_float4(isl_10.rotation_err_0) ;
        _S683->momentum_0 = packed_float4(isl_10.momentum_0) ;
        _S683->momentum_err_0 = packed_float4(isl_10.momentum_err_0) ;
    }
    return;
}

void island_frame_at_2(uint gi_2, uint tid_6, KernelContext_0 thread* kernelContext_41)
{
    bool woke_4;
    uint ct_3;
    if(tid_6 < 256U)
    {
        ct_3 = tid_6;
    }
    else
    {
        ct_3 = 1073741824U;
    }
    Island_natural_0 device* _S684 = kernelContext_41->islands_0+gi_2;
    uint4 _S685 = uint4((*_S684).info_0) ;
    float4 _S686 = float4((*_S684).com_0) ;
    float4 _S687 = float4((*_S684).inertia0_0) ;
    float4 _S688 = float4((*_S684).inertia1_0) ;
    float4 _S689 = float4((*_S684).inertia2_0) ;
    float4 _S690 = float4((*_S684).inv0_0) ;
    float4 _S691 = float4((*_S684).inv1_0) ;
    float4 _S692 = float4((*_S684).inv2_0) ;
    float4 _S693 = float4((*_S684).wcom_0) ;
    float4 _S694 = float4((*_S684).winv0_0) ;
    float4 _S695 = float4((*_S684).winv1_0) ;
    float4 _S696 = float4((*_S684).winv2_0) ;
    float4 _S697 = float4((*_S684).rotation_0) ;
    float4 _S698 = float4((*_S684).position_0) ;
    float4 _S699 = float4((*_S684).position_err_0) ;
    float4 _S700 = float4((*_S684).velocity_0) ;
    float4 _S701 = float4((*_S684).velocity_err_0) ;
    float4 _S702 = float4((*_S684).angular_velocity_0) ;
    uint4 _S703 = uint4((*_S684).probes_0) ;
    float4 _S704 = float4((*_S684).energy_0) ;
    float4 _S705 = float4((*_S684).rotation_err_0) ;
    float4 _S706 = float4((*_S684).momentum_0) ;
    float4 _S707 = float4((*_S684).momentum_err_0) ;
    thread Island_0 isl_11;
    (&isl_11)->range_0 = uint4((*_S684).range_0) ;
    (&isl_11)->info_0 = _S685;
    (&isl_11)->com_0 = _S686;
    (&isl_11)->inertia0_0 = _S687;
    (&isl_11)->inertia1_0 = _S688;
    (&isl_11)->inertia2_0 = _S689;
    (&isl_11)->inv0_0 = _S690;
    (&isl_11)->inv1_0 = _S691;
    (&isl_11)->inv2_0 = _S692;
    (&isl_11)->wcom_0 = _S693;
    (&isl_11)->winv0_0 = _S694;
    (&isl_11)->winv1_0 = _S695;
    (&isl_11)->winv2_0 = _S696;
    (&isl_11)->rotation_0 = _S697;
    (&isl_11)->position_0 = _S698;
    (&isl_11)->position_err_0 = _S699;
    (&isl_11)->velocity_0 = _S700;
    (&isl_11)->velocity_err_0 = _S701;
    (&isl_11)->angular_velocity_0 = _S702;
    (&isl_11)->done_0 = (*_S684).done_0;
    (&isl_11)->coverage_0 = (*_S684).coverage_0;
    (&isl_11)->wide_groups_0 = (*_S684).wide_groups_0;
    (&isl_11)->done_pad_0 = (*_S684).done_pad_0;
    (&isl_11)->probes_0 = _S703;
    (&isl_11)->energy_0 = _S704;
    (&isl_11)->rotation_err_0 = _S705;
    (&isl_11)->momentum_0 = _S706;
    (&isl_11)->momentum_err_0 = _S707;
    bool driven_2 = (((&isl_11)->info_0.x) & 2U) != 0U;
    bool _S708 = !((((&isl_11)->info_0.x) & 1U) != 0U);
    bool _S709;
    if(_S708)
    {
        _S709 = !driven_2;
    }
    else
    {
        _S709 = false;
    }
    bool contact_island_2 = (((&isl_11)->info_0.x) & 4U) != 0U;
    bool _S710 = (((&isl_11)->info_0.x) & 16U) != 0U;
    seen_clear_0(tid_6, kernelContext_41);
    bool _S711 = tid_6 == 0U;
    uint cov_10;
    bool settled_2;
    if(_S711)
    {
        if(contact_island_2 != ((kernelContext_41->params_0->contact_mode_0) == 1U))
        {
            settled_2 = true;
        }
        else
        {
            settled_2 = (((&isl_11)->info_0.x) & 8U) != 0U;
        }
        if(settled_2)
        {
            cov_10 = 0U;
        }
        else
        {
            cov_10 = 1U;
        }
        if(contact_island_2)
        {
            thread Island_0 _S712 = isl_11;
            bool _S713 = contact_stopped_0(&_S712, kernelContext_41);
            settled_2 = _S713;
        }
        else
        {
            settled_2 = false;
        }
        if(settled_2)
        {
            cov_10 = 0U;
        }
        *kernelContext_41->g_run_0 = cov_10;
        *kernelContext_41->g_halt_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((((&isl_11)->info_0.z) & 1U) != 0U)
    {
        settled_2 = true;
    }
    else
    {
        settled_2 = (*kernelContext_41->g_run_0) == 0U;
    }
    if(settled_2)
    {
        cov_10 = 0U;
    }
    else
    {
        cov_10 = min((&isl_11)->info_0.y, kernelContext_41->params_0->max_steps_0);
    }
    float _S714 = kernelContext_41->params_0->dt_0;
    bool _S715 = (kernelContext_41->params_0->fracture_0) != 0U;
    bool _S716 = (kernelContext_41->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S717 = isl_11;
    Rigid_0 _S718 = rigid_of_0(&_S717);
    thread Rigid_0 rg_9 = _S718;
    thread float work_4 = 0.0f;
    thread float work_err_3 = 0.0f;
    settled_2 = _S710;
    bool woke_5 = false;
    uint done_3 = 0U;
    bool resumed_4 = false;
    for(;;)
    {
        if(done_3 < cov_10)
        {
        }
        else
        {
            woke_4 = woke_5;
            break;
        }
        uint abs_step_2 = (&isl_11)->info_0.w + done_3 + 1U;
        uint k_14 = abs_step_2 - 1U - kernelContext_41->params_0->step_start_0;
        bool quiet_4 = (((&isl_11)->info_0.x) & 32U) != 0U;
        bool _S719;
        if(settled_2)
        {
            _S719 = !quiet_4;
        }
        else
        {
            _S719 = false;
        }
        uint c_23;
        bool _S720;
        bool explicit_step_2;
        if(_S719)
        {
            float3 _S721 = float3(0.0f) ;
            thread float3 norm_2 = _S721;
            thread float3 unused0_2 = _S721;
            c_23 = (&isl_11)->range_0.x + ct_3;
            for(;;)
            {
                if(c_23 < ((&isl_11)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S722 = (&rg_9)->rot_0;
                float _S723 = settled_chunk_load_0(c_23, &_S722, k_14, _S714, contact_island_2, kernelContext_41);
                norm_2.x = norm_2.x + _S723;
                c_23 = c_23 + 256U;
            }
            group_sum3_0(tid_6, &norm_2, &unused0_2, kernelContext_41);
            if((kernelContext_41->params_0->solve_mode_0) == 1U)
            {
                _S720 = (abs(norm_2.x - (&isl_11)->energy_0.z)) > ((&isl_11)->energy_0.w);
            }
            else
            {
                _S720 = false;
            }
            if(_S720)
            {
                explicit_step_2 = false;
                woke_4 = true;
            }
            else
            {
                explicit_step_2 = settled_2;
                woke_4 = woke_5;
            }
            settled_2 = explicit_step_2;
        }
        else
        {
            woke_4 = woke_5;
        }
        if(_S708)
        {
            _S720 = !quiet_4;
        }
        else
        {
            _S720 = false;
        }
        if(_S720)
        {
            float3 _S724 = float3(0.0f) ;
            thread float3 f_13 = _S724;
            thread float3 t_15 = _S724;
            c_23 = (&isl_11)->range_0.x + ct_3;
            for(;;)
            {
                if(c_23 < ((&isl_11)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S725 = isl_11;
                thread Rigid_0 _S726 = rg_9;
                net_load_0(c_23, &_S725, &_S726, k_14, _S714, contact_island_2, &f_13, &t_15, kernelContext_41);
                c_23 = c_23 + 256U;
            }
            group_sum3_0(tid_6, &f_13, &t_15, kernelContext_41);
            thread Island_0 _S727 = isl_11;
            rigid_acceleration_0(&_S727, &rg_9, f_13, t_15);
        }
        if(!settled_2)
        {
            explicit_step_2 = !quiet_4;
        }
        else
        {
            explicit_step_2 = false;
        }
        uint c_24;
        if(explicit_step_2)
        {
            c_23 = (&isl_11)->range_0.z + tid_6;
            for(;;)
            {
                if(c_23 < ((&isl_11)->range_0.w))
                {
                }
                else
                {
                    break;
                }
                bool _S728 = bond_update_0(c_23, _S714, _S715, abs_step_2, false, kernelContext_41);
                if(_S728)
                {
                    *kernelContext_41->g_halt_0 = 1U;
                }
                c_23 = c_23 + 512U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            c_24 = (&isl_11)->range_0.x + ct_3;
            for(;;)
            {
                if(c_24 < ((&isl_11)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S729 = isl_11;
                thread Rigid_0 _S730 = rg_9;
                chunk_update_0(c_24, &_S729, &_S730, _S714, _S716, k_14, contact_island_2, &work_4, &work_err_3, kernelContext_41);
                c_24 = c_24 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        bool _S731;
        if(_S709)
        {
            _S731 = !quiet_4;
        }
        else
        {
            _S731 = false;
        }
        if(_S731)
        {
            thread Island_0 _S732 = isl_11;
            integrate_rigid_0(&_S732, &rg_9, _S714);
        }
        bool _S733;
        if(_S708)
        {
            _S733 = explicit_step_2;
        }
        else
        {
            _S733 = false;
        }
        if(_S733)
        {
            float3 _S734 = (&isl_11)->wcom_0.xyz;
            float3 _S735 = float3(0.0f) ;
            thread float3 tu_3 = _S735;
            thread float3 pv_3 = _S735;
            c_23 = (&isl_11)->range_0.x + ct_3;
            for(;;)
            {
                if(c_23 < ((&isl_11)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_23, &tu_3, &pv_3, kernelContext_41);
                c_23 = c_23 + 256U;
            }
            group_sum3_0(tid_6, &tu_3, &pv_3, kernelContext_41);
            float3 tr_5 = tu_3 / float3((&isl_11)->wcom_0.w) ;
            float3 dv_5 = pv_3 / float3((&isl_11)->wcom_0.w) ;
            thread float3 lu_3 = _S735;
            thread float3 lv_3 = _S735;
            c_24 = (&isl_11)->range_0.x + ct_3;
            for(;;)
            {
                if(c_24 < ((&isl_11)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_24, _S734, tr_5, dv_5, &lu_3, &lv_3, kernelContext_41);
                c_24 = c_24 + 256U;
            }
            group_sum3_0(tid_6, &lu_3, &lv_3, kernelContext_41);
            float3 phi_5 = rows_mul_0((&isl_11)->winv0_0, (&isl_11)->winv1_0, (&isl_11)->winv2_0, lu_3);
            float3 dw_4 = rows_mul_0((&isl_11)->winv0_0, (&isl_11)->winv1_0, (&isl_11)->winv2_0, lv_3);
            uint c_25 = (&isl_11)->range_0.x + ct_3;
            for(;;)
            {
                if(c_25 < ((&isl_11)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_25, _S734, tr_5, phi_5, dv_5, dw_4, kernelContext_41);
                c_25 = c_25 + 256U;
            }
            if(!driven_2)
            {
                thread Island_0 _S736 = isl_11;
                drift_rigid_0(&_S736, &rg_9, tr_5, phi_5, dv_5, dw_4);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        bool _S737;
        if(_S711)
        {
            _S737 = ((&isl_11)->probes_0.y) > ((&isl_11)->probes_0.x);
        }
        else
        {
            _S737 = false;
        }
        if(_S737)
        {
            thread Island_0 _S738 = isl_11;
            thread Rigid_0 _S739 = rg_9;
            record_probes_0(&_S738, &_S739, k_14, kernelContext_41);
        }
        uint _S740 = done_3 + 1U;
        bool _S741;
        if((*kernelContext_41->g_halt_0) != 0U)
        {
            thread Island_0 _S742 = isl_11;
            bool _S743 = island_connected_narrow_0(tid_6, ct_3, &_S742, kernelContext_41);
            _S741 = _S743;
        }
        else
        {
            _S741 = false;
        }
        bool resumed_5;
        if(_S741)
        {
            if(_S711)
            {
                *kernelContext_41->g_halt_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            resumed_5 = true;
        }
        else
        {
            resumed_5 = resumed_4;
        }
        if((*kernelContext_41->g_halt_0) != 0U)
        {
            done_3 = _S740;
            resumed_4 = resumed_5;
            break;
        }
        woke_5 = woke_4;
        done_3 = _S740;
        resumed_4 = resumed_5;
    }
    thread float3 wsum_2 = float3(work_4, work_err_3, 0.0f);
    thread float3 unused_2 = float3(0.0f) ;
    group_sum3_0(tid_6, &wsum_2, &unused_2, kernelContext_41);
    if(_S711)
    {
        thread Quat_0 _S744 = (&rg_9)->rot_0;
        float4 _S745 = quat_vec_0(&_S744);
        (&isl_11)->rotation_0 = _S745;
        (&isl_11)->rotation_err_0 = (&rg_9)->rot_err_0;
        (&isl_11)->position_0 = float4((&rg_9)->pos_0, 0.0f);
        (&isl_11)->position_err_0 = float4((&rg_9)->pos_err_0, 0.0f);
        (&isl_11)->velocity_0 = float4((&rg_9)->vel_0, 0.0f);
        (&isl_11)->velocity_err_0 = float4((&rg_9)->vel_err_0, 0.0f);
        (&isl_11)->angular_velocity_0 = float4((&rg_9)->w_1, 0.0f);
        (&isl_11)->momentum_0 = float4((&rg_9)->l_0, 0.0f);
        (&isl_11)->momentum_err_0 = float4((&rg_9)->l_err_0, 0.0f);
        (&isl_11)->done_0 = done_3;
        (&isl_11)->info_0.y = (&isl_11)->info_0.y - done_3;
        if((*kernelContext_41->g_halt_0) != 0U)
        {
            _S709 = contact_island_2;
        }
        else
        {
            _S709 = false;
        }
        if(_S709)
        {
            contact_split_at_0((&isl_11)->info_0.w + done_3, kernelContext_41);
        }
        float _S746 = wsum_2.x;
        thread float _S747 = (&isl_11)->energy_0.x;
        thread float _S748 = (&isl_11)->energy_0.y;
        comp_add1_0(&_S747, &_S748, _S746);
        (&isl_11)->energy_0.x = _S747;
        (&isl_11)->energy_0.y = _S748 + wsum_2.y;
        (&isl_11)->info_0.w = (&isl_11)->info_0.w + done_3;
        if((((&isl_11)->info_0.x) & 8U) == 0U)
        {
            (&isl_11)->info_0.z = ((&isl_11)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_41->g_halt_0) != 0U)
        {
            (&isl_11)->info_0.z = ((&isl_11)->info_0.z) | 1U;
        }
        uint _S749 = seen_bits_0(kernelContext_41);
        if((*kernelContext_41->g_halt_0) != 0U)
        {
            ct_3 = 128U;
        }
        else
        {
            ct_3 = 0U;
        }
        uint _S750 = _S749 | ct_3;
        if(woke_4)
        {
            ct_3 = 2048U;
        }
        else
        {
            ct_3 = 0U;
        }
        uint _S751 = _S750 | ct_3;
        if(resumed_4)
        {
            ct_3 = 2097280U;
        }
        else
        {
            ct_3 = 0U;
        }
        uint cov_11 = _S751 | ct_3;
        if(done_3 > 0U)
        {
            bool quiet_5 = (((&isl_11)->info_0.x) & 32U) != 0U;
            if(quiet_5)
            {
                ct_3 = 4096U;
            }
            else
            {
                ct_3 = 0U;
            }
            uint cov_12 = cov_11 | ct_3;
            if((((&isl_11)->info_0.x) & 16U) != 0U)
            {
                _S709 = !quiet_5;
            }
            else
            {
                _S709 = false;
            }
            if(_S709)
            {
                ct_3 = 512U;
            }
            else
            {
                ct_3 = 0U;
            }
            uint cov_13 = cov_12 | ct_3;
            if((kernelContext_41->params_0->contact_mode_0) == 1U)
            {
                ct_3 = 524288U;
            }
            else
            {
                ct_3 = 0U;
            }
            uint cov_14 = cov_13 | ct_3;
            if(((&isl_11)->probes_0.y) > ((&isl_11)->probes_0.x))
            {
                ct_3 = 262144U;
            }
            else
            {
                ct_3 = 0U;
            }
            cov_10 = cov_14 | ct_3;
        }
        else
        {
            cov_10 = cov_11;
        }
        (&isl_11)->coverage_0 = ((&isl_11)->coverage_0) | cov_10;
        if(woke_4)
        {
            (&isl_11)->info_0.x = ((&isl_11)->info_0.x) & 4294967279U;
            (&isl_11)->info_0.z = ((&isl_11)->info_0.z) | 4U;
        }
        Island_natural_0 device* _S752 = kernelContext_41->islands_0+gi_2;
        _S752->range_0 = packed_uint4(isl_11.range_0) ;
        _S752->info_0 = packed_uint4(isl_11.info_0) ;
        _S752->com_0 = packed_float4(isl_11.com_0) ;
        _S752->inertia0_0 = packed_float4(isl_11.inertia0_0) ;
        _S752->inertia1_0 = packed_float4(isl_11.inertia1_0) ;
        _S752->inertia2_0 = packed_float4(isl_11.inertia2_0) ;
        _S752->inv0_0 = packed_float4(isl_11.inv0_0) ;
        _S752->inv1_0 = packed_float4(isl_11.inv1_0) ;
        _S752->inv2_0 = packed_float4(isl_11.inv2_0) ;
        _S752->wcom_0 = packed_float4(isl_11.wcom_0) ;
        _S752->winv0_0 = packed_float4(isl_11.winv0_0) ;
        _S752->winv1_0 = packed_float4(isl_11.winv1_0) ;
        _S752->winv2_0 = packed_float4(isl_11.winv2_0) ;
        _S752->rotation_0 = packed_float4(isl_11.rotation_0) ;
        _S752->position_0 = packed_float4(isl_11.position_0) ;
        _S752->position_err_0 = packed_float4(isl_11.position_err_0) ;
        _S752->velocity_0 = packed_float4(isl_11.velocity_0) ;
        _S752->velocity_err_0 = packed_float4(isl_11.velocity_err_0) ;
        _S752->angular_velocity_0 = packed_float4(isl_11.angular_velocity_0) ;
        _S752->done_0 = isl_11.done_0;
        _S752->coverage_0 = isl_11.coverage_0;
        _S752->wide_groups_0 = isl_11.wide_groups_0;
        _S752->done_pad_0 = isl_11.done_pad_0;
        _S752->probes_0 = packed_uint4(isl_11.probes_0) ;
        _S752->energy_0 = packed_float4(isl_11.energy_0) ;
        _S752->rotation_err_0 = packed_float4(isl_11.rotation_err_0) ;
        _S752->momentum_0 = packed_float4(isl_11.momentum_0) ;
        _S752->momentum_err_0 = packed_float4(isl_11.momentum_err_0) ;
    }
    return;
}

void island_frame_at_3(uint gi_3, uint tid_7, KernelContext_0 thread* kernelContext_42)
{
    bool woke_6;
    uint ct_4;
    if(tid_7 < 256U)
    {
        ct_4 = tid_7;
    }
    else
    {
        ct_4 = 1073741824U;
    }
    Island_natural_0 device* _S753 = kernelContext_42->islands_0+gi_3;
    uint4 _S754 = uint4((*_S753).info_0) ;
    float4 _S755 = float4((*_S753).com_0) ;
    float4 _S756 = float4((*_S753).inertia0_0) ;
    float4 _S757 = float4((*_S753).inertia1_0) ;
    float4 _S758 = float4((*_S753).inertia2_0) ;
    float4 _S759 = float4((*_S753).inv0_0) ;
    float4 _S760 = float4((*_S753).inv1_0) ;
    float4 _S761 = float4((*_S753).inv2_0) ;
    float4 _S762 = float4((*_S753).wcom_0) ;
    float4 _S763 = float4((*_S753).winv0_0) ;
    float4 _S764 = float4((*_S753).winv1_0) ;
    float4 _S765 = float4((*_S753).winv2_0) ;
    float4 _S766 = float4((*_S753).rotation_0) ;
    float4 _S767 = float4((*_S753).position_0) ;
    float4 _S768 = float4((*_S753).position_err_0) ;
    float4 _S769 = float4((*_S753).velocity_0) ;
    float4 _S770 = float4((*_S753).velocity_err_0) ;
    float4 _S771 = float4((*_S753).angular_velocity_0) ;
    uint4 _S772 = uint4((*_S753).probes_0) ;
    float4 _S773 = float4((*_S753).energy_0) ;
    float4 _S774 = float4((*_S753).rotation_err_0) ;
    float4 _S775 = float4((*_S753).momentum_0) ;
    float4 _S776 = float4((*_S753).momentum_err_0) ;
    thread Island_0 isl_12;
    (&isl_12)->range_0 = uint4((*_S753).range_0) ;
    (&isl_12)->info_0 = _S754;
    (&isl_12)->com_0 = _S755;
    (&isl_12)->inertia0_0 = _S756;
    (&isl_12)->inertia1_0 = _S757;
    (&isl_12)->inertia2_0 = _S758;
    (&isl_12)->inv0_0 = _S759;
    (&isl_12)->inv1_0 = _S760;
    (&isl_12)->inv2_0 = _S761;
    (&isl_12)->wcom_0 = _S762;
    (&isl_12)->winv0_0 = _S763;
    (&isl_12)->winv1_0 = _S764;
    (&isl_12)->winv2_0 = _S765;
    (&isl_12)->rotation_0 = _S766;
    (&isl_12)->position_0 = _S767;
    (&isl_12)->position_err_0 = _S768;
    (&isl_12)->velocity_0 = _S769;
    (&isl_12)->velocity_err_0 = _S770;
    (&isl_12)->angular_velocity_0 = _S771;
    (&isl_12)->done_0 = (*_S753).done_0;
    (&isl_12)->coverage_0 = (*_S753).coverage_0;
    (&isl_12)->wide_groups_0 = (*_S753).wide_groups_0;
    (&isl_12)->done_pad_0 = (*_S753).done_pad_0;
    (&isl_12)->probes_0 = _S772;
    (&isl_12)->energy_0 = _S773;
    (&isl_12)->rotation_err_0 = _S774;
    (&isl_12)->momentum_0 = _S775;
    (&isl_12)->momentum_err_0 = _S776;
    bool driven_3 = (((&isl_12)->info_0.x) & 2U) != 0U;
    bool _S777 = !((((&isl_12)->info_0.x) & 1U) != 0U);
    bool _S778;
    if(_S777)
    {
        _S778 = !driven_3;
    }
    else
    {
        _S778 = false;
    }
    bool contact_island_3 = (((&isl_12)->info_0.x) & 4U) != 0U;
    bool _S779 = (((&isl_12)->info_0.x) & 16U) != 0U;
    seen_clear_0(tid_7, kernelContext_42);
    bool _S780 = tid_7 == 0U;
    uint cov_15;
    bool settled_3;
    if(_S780)
    {
        if(contact_island_3 != ((kernelContext_42->params_0->contact_mode_0) == 1U))
        {
            settled_3 = true;
        }
        else
        {
            settled_3 = (((&isl_12)->info_0.x) & 8U) != 0U;
        }
        if(settled_3)
        {
            cov_15 = 0U;
        }
        else
        {
            cov_15 = 1U;
        }
        if(contact_island_3)
        {
            thread Island_0 _S781 = isl_12;
            bool _S782 = contact_stopped_0(&_S781, kernelContext_42);
            settled_3 = _S782;
        }
        else
        {
            settled_3 = false;
        }
        if(settled_3)
        {
            cov_15 = 0U;
        }
        *kernelContext_42->g_run_0 = cov_15;
        *kernelContext_42->g_halt_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((((&isl_12)->info_0.z) & 1U) != 0U)
    {
        settled_3 = true;
    }
    else
    {
        settled_3 = (*kernelContext_42->g_run_0) == 0U;
    }
    if(settled_3)
    {
        cov_15 = 0U;
    }
    else
    {
        cov_15 = min((&isl_12)->info_0.y, kernelContext_42->params_0->max_steps_0);
    }
    float _S783 = kernelContext_42->params_0->dt_0;
    bool _S784 = (kernelContext_42->params_0->fracture_0) != 0U;
    bool _S785 = (kernelContext_42->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S786 = isl_12;
    Rigid_0 _S787 = rigid_of_0(&_S786);
    thread Rigid_0 rg_10 = _S787;
    thread float work_5 = 0.0f;
    thread float work_err_4 = 0.0f;
    settled_3 = _S779;
    bool woke_7 = false;
    uint done_4 = 0U;
    bool resumed_6 = false;
    for(;;)
    {
        if(done_4 < cov_15)
        {
        }
        else
        {
            woke_6 = woke_7;
            break;
        }
        uint abs_step_3 = (&isl_12)->info_0.w + done_4 + 1U;
        uint k_15 = abs_step_3 - 1U - kernelContext_42->params_0->step_start_0;
        bool quiet_6 = (((&isl_12)->info_0.x) & 32U) != 0U;
        bool _S788;
        if(settled_3)
        {
            _S788 = !quiet_6;
        }
        else
        {
            _S788 = false;
        }
        uint c_26;
        bool _S789;
        bool explicit_step_3;
        if(_S788)
        {
            float3 _S790 = float3(0.0f) ;
            thread float3 norm_3 = _S790;
            thread float3 unused0_3 = _S790;
            c_26 = (&isl_12)->range_0.x + ct_4;
            for(;;)
            {
                if(c_26 < ((&isl_12)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S791 = (&rg_10)->rot_0;
                float _S792 = settled_chunk_load_0(c_26, &_S791, k_15, _S783, contact_island_3, kernelContext_42);
                norm_3.x = norm_3.x + _S792;
                c_26 = c_26 + 256U;
            }
            group_sum3_0(tid_7, &norm_3, &unused0_3, kernelContext_42);
            if((kernelContext_42->params_0->solve_mode_0) == 1U)
            {
                _S789 = (abs(norm_3.x - (&isl_12)->energy_0.z)) > ((&isl_12)->energy_0.w);
            }
            else
            {
                _S789 = false;
            }
            if(_S789)
            {
                explicit_step_3 = false;
                woke_6 = true;
            }
            else
            {
                explicit_step_3 = settled_3;
                woke_6 = woke_7;
            }
            settled_3 = explicit_step_3;
        }
        else
        {
            woke_6 = woke_7;
        }
        if(_S777)
        {
            _S789 = !quiet_6;
        }
        else
        {
            _S789 = false;
        }
        if(_S789)
        {
            float3 _S793 = float3(0.0f) ;
            thread float3 f_14 = _S793;
            thread float3 t_16 = _S793;
            c_26 = (&isl_12)->range_0.x + ct_4;
            for(;;)
            {
                if(c_26 < ((&isl_12)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S794 = isl_12;
                thread Rigid_0 _S795 = rg_10;
                net_load_0(c_26, &_S794, &_S795, k_15, _S783, contact_island_3, &f_14, &t_16, kernelContext_42);
                c_26 = c_26 + 256U;
            }
            group_sum3_0(tid_7, &f_14, &t_16, kernelContext_42);
            thread Island_0 _S796 = isl_12;
            rigid_acceleration_0(&_S796, &rg_10, f_14, t_16);
        }
        if(!settled_3)
        {
            explicit_step_3 = !quiet_6;
        }
        else
        {
            explicit_step_3 = false;
        }
        uint c_27;
        if(explicit_step_3)
        {
            c_26 = (&isl_12)->range_0.z + tid_7;
            for(;;)
            {
                if(c_26 < ((&isl_12)->range_0.w))
                {
                }
                else
                {
                    break;
                }
                bool _S797 = bond_update_0(c_26, _S783, _S784, abs_step_3, false, kernelContext_42);
                if(_S797)
                {
                    *kernelContext_42->g_halt_0 = 1U;
                }
                c_26 = c_26 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            c_27 = (&isl_12)->range_0.x + ct_4;
            for(;;)
            {
                if(c_27 < ((&isl_12)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S798 = isl_12;
                thread Rigid_0 _S799 = rg_10;
                chunk_update_0(c_27, &_S798, &_S799, _S783, _S785, k_15, contact_island_3, &work_5, &work_err_4, kernelContext_42);
                c_27 = c_27 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        bool _S800;
        if(_S778)
        {
            _S800 = !quiet_6;
        }
        else
        {
            _S800 = false;
        }
        if(_S800)
        {
            thread Island_0 _S801 = isl_12;
            integrate_rigid_0(&_S801, &rg_10, _S783);
        }
        bool _S802;
        if(_S777)
        {
            _S802 = explicit_step_3;
        }
        else
        {
            _S802 = false;
        }
        if(_S802)
        {
            float3 _S803 = (&isl_12)->wcom_0.xyz;
            float3 _S804 = float3(0.0f) ;
            thread float3 tu_4 = _S804;
            thread float3 pv_4 = _S804;
            c_26 = (&isl_12)->range_0.x + ct_4;
            for(;;)
            {
                if(c_26 < ((&isl_12)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_26, &tu_4, &pv_4, kernelContext_42);
                c_26 = c_26 + 256U;
            }
            group_sum3_0(tid_7, &tu_4, &pv_4, kernelContext_42);
            float3 tr_6 = tu_4 / float3((&isl_12)->wcom_0.w) ;
            float3 dv_6 = pv_4 / float3((&isl_12)->wcom_0.w) ;
            thread float3 lu_4 = _S804;
            thread float3 lv_4 = _S804;
            c_27 = (&isl_12)->range_0.x + ct_4;
            for(;;)
            {
                if(c_27 < ((&isl_12)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_27, _S803, tr_6, dv_6, &lu_4, &lv_4, kernelContext_42);
                c_27 = c_27 + 256U;
            }
            group_sum3_0(tid_7, &lu_4, &lv_4, kernelContext_42);
            float3 phi_6 = rows_mul_0((&isl_12)->winv0_0, (&isl_12)->winv1_0, (&isl_12)->winv2_0, lu_4);
            float3 dw_5 = rows_mul_0((&isl_12)->winv0_0, (&isl_12)->winv1_0, (&isl_12)->winv2_0, lv_4);
            uint c_28 = (&isl_12)->range_0.x + ct_4;
            for(;;)
            {
                if(c_28 < ((&isl_12)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_28, _S803, tr_6, phi_6, dv_6, dw_5, kernelContext_42);
                c_28 = c_28 + 256U;
            }
            if(!driven_3)
            {
                thread Island_0 _S805 = isl_12;
                drift_rigid_0(&_S805, &rg_10, tr_6, phi_6, dv_6, dw_5);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        bool _S806;
        if(_S780)
        {
            _S806 = ((&isl_12)->probes_0.y) > ((&isl_12)->probes_0.x);
        }
        else
        {
            _S806 = false;
        }
        if(_S806)
        {
            thread Island_0 _S807 = isl_12;
            thread Rigid_0 _S808 = rg_10;
            record_probes_0(&_S807, &_S808, k_15, kernelContext_42);
        }
        uint _S809 = done_4 + 1U;
        bool _S810;
        if((*kernelContext_42->g_halt_0) != 0U)
        {
            thread Island_0 _S811 = isl_12;
            bool _S812 = island_connected_narrow_0(tid_7, ct_4, &_S811, kernelContext_42);
            _S810 = _S812;
        }
        else
        {
            _S810 = false;
        }
        bool resumed_7;
        if(_S810)
        {
            if(_S780)
            {
                *kernelContext_42->g_halt_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            resumed_7 = true;
        }
        else
        {
            resumed_7 = resumed_6;
        }
        if((*kernelContext_42->g_halt_0) != 0U)
        {
            done_4 = _S809;
            resumed_6 = resumed_7;
            break;
        }
        woke_7 = woke_6;
        done_4 = _S809;
        resumed_6 = resumed_7;
    }
    thread float3 wsum_3 = float3(work_5, work_err_4, 0.0f);
    thread float3 unused_3 = float3(0.0f) ;
    group_sum3_0(tid_7, &wsum_3, &unused_3, kernelContext_42);
    if(_S780)
    {
        thread Quat_0 _S813 = (&rg_10)->rot_0;
        float4 _S814 = quat_vec_0(&_S813);
        (&isl_12)->rotation_0 = _S814;
        (&isl_12)->rotation_err_0 = (&rg_10)->rot_err_0;
        (&isl_12)->position_0 = float4((&rg_10)->pos_0, 0.0f);
        (&isl_12)->position_err_0 = float4((&rg_10)->pos_err_0, 0.0f);
        (&isl_12)->velocity_0 = float4((&rg_10)->vel_0, 0.0f);
        (&isl_12)->velocity_err_0 = float4((&rg_10)->vel_err_0, 0.0f);
        (&isl_12)->angular_velocity_0 = float4((&rg_10)->w_1, 0.0f);
        (&isl_12)->momentum_0 = float4((&rg_10)->l_0, 0.0f);
        (&isl_12)->momentum_err_0 = float4((&rg_10)->l_err_0, 0.0f);
        (&isl_12)->done_0 = done_4;
        (&isl_12)->info_0.y = (&isl_12)->info_0.y - done_4;
        if((*kernelContext_42->g_halt_0) != 0U)
        {
            _S778 = contact_island_3;
        }
        else
        {
            _S778 = false;
        }
        if(_S778)
        {
            contact_split_at_0((&isl_12)->info_0.w + done_4, kernelContext_42);
        }
        float _S815 = wsum_3.x;
        thread float _S816 = (&isl_12)->energy_0.x;
        thread float _S817 = (&isl_12)->energy_0.y;
        comp_add1_0(&_S816, &_S817, _S815);
        (&isl_12)->energy_0.x = _S816;
        (&isl_12)->energy_0.y = _S817 + wsum_3.y;
        (&isl_12)->info_0.w = (&isl_12)->info_0.w + done_4;
        if((((&isl_12)->info_0.x) & 8U) == 0U)
        {
            (&isl_12)->info_0.z = ((&isl_12)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_42->g_halt_0) != 0U)
        {
            (&isl_12)->info_0.z = ((&isl_12)->info_0.z) | 1U;
        }
        uint _S818 = seen_bits_0(kernelContext_42);
        if((*kernelContext_42->g_halt_0) != 0U)
        {
            ct_4 = 128U;
        }
        else
        {
            ct_4 = 0U;
        }
        uint _S819 = _S818 | ct_4;
        if(woke_6)
        {
            ct_4 = 2048U;
        }
        else
        {
            ct_4 = 0U;
        }
        uint _S820 = _S819 | ct_4;
        if(resumed_6)
        {
            ct_4 = 2097280U;
        }
        else
        {
            ct_4 = 0U;
        }
        uint cov_16 = _S820 | ct_4;
        if(done_4 > 0U)
        {
            bool quiet_7 = (((&isl_12)->info_0.x) & 32U) != 0U;
            if(quiet_7)
            {
                ct_4 = 4096U;
            }
            else
            {
                ct_4 = 0U;
            }
            uint cov_17 = cov_16 | ct_4;
            if((((&isl_12)->info_0.x) & 16U) != 0U)
            {
                _S778 = !quiet_7;
            }
            else
            {
                _S778 = false;
            }
            if(_S778)
            {
                ct_4 = 512U;
            }
            else
            {
                ct_4 = 0U;
            }
            uint cov_18 = cov_17 | ct_4;
            if((kernelContext_42->params_0->contact_mode_0) == 1U)
            {
                ct_4 = 524288U;
            }
            else
            {
                ct_4 = 0U;
            }
            uint cov_19 = cov_18 | ct_4;
            if(((&isl_12)->probes_0.y) > ((&isl_12)->probes_0.x))
            {
                ct_4 = 262144U;
            }
            else
            {
                ct_4 = 0U;
            }
            cov_15 = cov_19 | ct_4;
        }
        else
        {
            cov_15 = cov_16;
        }
        (&isl_12)->coverage_0 = ((&isl_12)->coverage_0) | cov_15;
        if(woke_6)
        {
            (&isl_12)->info_0.x = ((&isl_12)->info_0.x) & 4294967279U;
            (&isl_12)->info_0.z = ((&isl_12)->info_0.z) | 4U;
        }
        Island_natural_0 device* _S821 = kernelContext_42->islands_0+gi_3;
        _S821->range_0 = packed_uint4(isl_12.range_0) ;
        _S821->info_0 = packed_uint4(isl_12.info_0) ;
        _S821->com_0 = packed_float4(isl_12.com_0) ;
        _S821->inertia0_0 = packed_float4(isl_12.inertia0_0) ;
        _S821->inertia1_0 = packed_float4(isl_12.inertia1_0) ;
        _S821->inertia2_0 = packed_float4(isl_12.inertia2_0) ;
        _S821->inv0_0 = packed_float4(isl_12.inv0_0) ;
        _S821->inv1_0 = packed_float4(isl_12.inv1_0) ;
        _S821->inv2_0 = packed_float4(isl_12.inv2_0) ;
        _S821->wcom_0 = packed_float4(isl_12.wcom_0) ;
        _S821->winv0_0 = packed_float4(isl_12.winv0_0) ;
        _S821->winv1_0 = packed_float4(isl_12.winv1_0) ;
        _S821->winv2_0 = packed_float4(isl_12.winv2_0) ;
        _S821->rotation_0 = packed_float4(isl_12.rotation_0) ;
        _S821->position_0 = packed_float4(isl_12.position_0) ;
        _S821->position_err_0 = packed_float4(isl_12.position_err_0) ;
        _S821->velocity_0 = packed_float4(isl_12.velocity_0) ;
        _S821->velocity_err_0 = packed_float4(isl_12.velocity_err_0) ;
        _S821->angular_velocity_0 = packed_float4(isl_12.angular_velocity_0) ;
        _S821->done_0 = isl_12.done_0;
        _S821->coverage_0 = isl_12.coverage_0;
        _S821->wide_groups_0 = isl_12.wide_groups_0;
        _S821->done_pad_0 = isl_12.done_pad_0;
        _S821->probes_0 = packed_uint4(isl_12.probes_0) ;
        _S821->energy_0 = packed_float4(isl_12.energy_0) ;
        _S821->rotation_err_0 = packed_float4(isl_12.rotation_err_0) ;
        _S821->momentum_0 = packed_float4(isl_12.momentum_0) ;
        _S821->momentum_err_0 = packed_float4(isl_12.momentum_err_0) ;
    }
    return;
}

uint pair_lane_count_0(KernelContext_0 thread* kernelContext_43)
{
    return max(kernelContext_43->params_0->pair_lanes_0, 1U);
}

bool stopped_0(KernelContext_0 thread* kernelContext_44)
{
    uint4 _S822 = uint4((kernelContext_44->islands_0+kernelContext_44->params_0->halt_index_0)->info_0) ;
    bool _S823;
    if(((_S822.z) & 1U) != 0U)
    {
        _S823 = true;
    }
    else
    {
        _S823 = (_S822.y) != 0U;
    }
    return _S823;
}

bool pair_any_0(bool x_9, uint lanes_0)
{
    bool _S824;
    if(lanes_0 > 1U)
    {
        bool _S825 = simd_any(x_9);
        _S824 = _S825;
    }
    else
    {
        _S824 = x_9;
    }
    return _S824;
}

float3 world_diff_0(const WorldPoint_0 thread* a_10, const WorldPoint_0 thread* b_16)
{
    return a_10->hi_0 - b_16->hi_0 + (a_10->lo_0 - b_16->lo_0) + (a_10->rel_0 - b_16->rel_0);
}

bool may_overlap_lanes_0(const Box_0 thread* a_11, const Box_0 thread* b_17, uint lane_0, uint lanes_1)
{
    bool separated_0 = false;
    uint i_7 = lane_0;
    for(;;)
    {
        if(i_7 < 15U)
        {
        }
        else
        {
            break;
        }
        float3 d_4 = b_17->center_1 - a_11->center_1;
        float3 _S826 = a_11->half_2;
        float3 _S827 = b_17->half_2;
        float margin_0 = 0.00000999999974738f * (length(a_11->half_2) + length(b_17->half_2));
        float3 _S828 = a_11->axis0_0;
        float3 _S829 = a_11->axis1_0;
        float3 _S830 = a_11->axis2_0;
        float3 _S831 = b_17->axis0_0;
        float3 _S832 = b_17->axis1_0;
        float3 _S833 = b_17->axis2_0;
        array<float3, int(6)> axes_0 = { { a_11->axis0_0, a_11->axis1_0, a_11->axis2_0, b_17->axis0_0, b_17->axis1_0, b_17->axis2_0 } };
        float3 l_3;
        if(i_7 < 6U)
        {
            l_3 = axes_0[i_7];
        }
        else
        {
            uint _S834 = i_7 - 6U;
            l_3 = cross(axes_0[_S834 / 3U], axes_0[3U + _S834 % 3U]);
        }
        float len_0 = length(l_3);
        if(len_0 <= 9.99999997475242708e-07f)
        {
            i_7 = i_7 + lanes_1;
            continue;
        }
        bool separated_1;
        if((abs(dot(d_4, l_3))) > (_S826.x * abs(dot(_S828, l_3)) + _S826.y * abs(dot(_S829, l_3)) + _S826.z * abs(dot(_S830, l_3)) + (_S827.x * abs(dot(_S831, l_3)) + _S827.y * abs(dot(_S832, l_3)) + _S827.z * abs(dot(_S833, l_3))) + margin_0 * len_0))
        {
            separated_1 = true;
        }
        else
        {
            separated_1 = separated_0;
        }
        separated_0 = separated_1;
        i_7 = i_7 + lanes_1;
    }
    bool _S835 = pair_any_0(separated_0, lanes_1);
    return !_S835;
}

bool may_overlap_0(const Box_0 thread* a_12, const Box_0 thread* b_18)
{
    float3 _S836 = b_18->center_1 - a_12->center_1;
    float3 _S837 = a_12->half_2;
    float3 _S838 = b_18->half_2;
    float _S839 = 0.00000999999974738f * (length(a_12->half_2) + length(b_18->half_2));
    float3 _S840 = a_12->axis0_0;
    float3 _S841 = a_12->axis1_0;
    float3 _S842 = a_12->axis2_0;
    float3 _S843 = b_18->axis0_0;
    float3 _S844 = b_18->axis1_0;
    float3 _S845 = b_18->axis2_0;
    array<float3, int(6)> _S846 = { { a_12->axis0_0, a_12->axis1_0, a_12->axis2_0, b_18->axis0_0, b_18->axis1_0, b_18->axis2_0 } };
    uint i_8 = 0U;
    for(;;)
    {
        if(i_8 < 15U)
        {
        }
        else
        {
            break;
        }
        float3 l_4;
        if(i_8 < 6U)
        {
            l_4 = _S846[i_8];
        }
        else
        {
            uint _S847 = i_8 - 6U;
            l_4 = cross(_S846[_S847 / 3U], _S846[3U + _S847 % 3U]);
        }
        float len_1 = length(l_4);
        if(len_1 <= 9.99999997475242708e-07f)
        {
            i_8 = i_8 + 1U;
            continue;
        }
        if((abs(dot(_S836, l_4))) > (_S837.x * abs(dot(_S840, l_4)) + _S837.y * abs(dot(_S841, l_4)) + _S837.z * abs(dot(_S842, l_4)) + (_S838.x * abs(dot(_S843, l_4)) + _S838.y * abs(dot(_S844, l_4)) + _S838.z * abs(dot(_S845, l_4))) + _S839 * len_1))
        {
            return false;
        }
        i_8 = i_8 + 1U;
    }
    return true;
}

float3 sample_point_1(const Box_0 thread* b_19, uint i_9, KernelContext_0 thread* kernelContext_45)
{
    uint _S848 = b_19->hull_v_0;
    if((b_19->hull_v_0) != 0U)
    {
        float3 local_2;
        if(i_9 < _S848)
        {
            local_2 = (float4(*(kernelContext_45->loads_0+(b_19->hull_at_0 + i_9))) ).xyz * float3(0.89999997615814209f) ;
        }
        else
        {
            local_2 = (float4(*(kernelContext_45->loads_0+(b_19->hull_at_0 + _S848 + b_19->hull_f_0 + (i_9 - _S848)))) ).xyz;
        }
        float3 _S849 = b_19->center_1;
        float3 _S850 = box_to_world_0(b_19, local_2);
        return _S849 + _S850;
    }
    float sign_1;
    if(i_9 < 8U)
    {
        float3 h_3 = b_19->half_2 * float3(0.89999997615814209f) ;
        if((i_9 & 1U) == 0U)
        {
            sign_1 = - h_3.x;
        }
        else
        {
            sign_1 = h_3.x;
        }
        float _S851;
        if((i_9 & 2U) == 0U)
        {
            _S851 = - h_3.y;
        }
        else
        {
            _S851 = h_3.y;
        }
        float _S852;
        if((i_9 & 4U) == 0U)
        {
            _S852 = - h_3.z;
        }
        else
        {
            _S852 = h_3.z;
        }
        return b_19->center_1 + b_19->axis0_0 * float3(sign_1)  + b_19->axis1_0 * float3(_S851)  + b_19->axis2_0 * float3(_S852) ;
    }
    uint _S853 = i_9 - 8U;
    uint axis_3 = _S853 / 2U;
    if((_S853 % 2U) == 0U)
    {
        sign_1 = -1.0f;
    }
    else
    {
        sign_1 = 1.0f;
    }
    float3 _S854 = b_19->center_1;
    float3 _S855 = box_axis_0(b_19, axis_3);
    return _S854 + _S855 * float3((sign_1 * comp3_0(b_19->half_2, axis_3))) ;
}

float3 box_to_local_0(const Box_0 thread* b_20, float3 r_8)
{
    return float3(dot(r_8, b_20->axis0_0), dot(r_8, b_20->axis1_0), dot(r_8, b_20->axis2_0));
}

float hull_signed_distance_0(const Box_0 thread* b_21, float3 local_3, uint thread* face_0, KernelContext_0 thread* kernelContext_46)
{
    *face_0 = 0U;
    float best_0 = -1.00000001504746622e+30f;
    uint f_15 = 0U;
    for(;;)
    {
        if(f_15 < (b_21->hull_f_0))
        {
        }
        else
        {
            break;
        }
        float4 _S856 = float4(*(kernelContext_46->loads_0+(b_21->hull_at_0 + b_21->hull_v_0 + f_15))) ;
        float d_5 = dot(_S856.xyz, local_3) - _S856.w;
        if(d_5 > best_0)
        {
            *face_0 = f_15;
            best_0 = d_5;
        }
        f_15 = f_15 + 1U;
    }
    return best_0;
}

bool penetration_0(const Box_0 thread* b_22, float3 p_3, float thread* depth_1, float3 thread* normal_3, KernelContext_0 thread* kernelContext_47)
{
    *depth_1 = 0.0f;
    *normal_3 = float3(0.0f) ;
    float3 r_9 = p_3 - b_22->center_1;
    float3 _S857 = b_22->half_2;
    if((dot(r_9, r_9)) > (dot(b_22->half_2, b_22->half_2) * 1.00001001358032227f))
    {
        return false;
    }
    uint _S858 = b_22->hull_v_0;
    if((b_22->hull_v_0) != 0U)
    {
        float3 _S859 = box_to_local_0(b_22, r_9);
        thread uint face_1;
        float _S860 = hull_signed_distance_0(b_22, _S859, &face_1, kernelContext_47);
        if(!(_S860 < 0.0f))
        {
            return false;
        }
        *depth_1 = - _S860;
        float3 _S861 = box_to_world_0(b_22, (float4(*(kernelContext_47->loads_0+(b_22->hull_at_0 + _S858 + face_1))) ).xyz);
        *normal_3 = _S861;
        return true;
    }
    float best_1 = 1.00000001504746622e+30f;
    uint axis_4 = 0U;
    float side_0 = 1.0f;
    uint k_16 = 0U;
    for(;;)
    {
        if(k_16 < 3U)
        {
        }
        else
        {
            break;
        }
        float3 _S862 = box_axis_0(b_22, k_16);
        float local_4 = dot(r_9, _S862);
        float d_6 = comp3_0(_S857, k_16) - abs(local_4);
        if(d_6 <= 0.0f)
        {
            return false;
        }
        if(d_6 < best_1)
        {
            float _S863;
            if(local_4 >= 0.0f)
            {
                _S863 = 1.0f;
            }
            else
            {
                _S863 = -1.0f;
            }
            best_1 = d_6;
            axis_4 = k_16;
            side_0 = _S863;
        }
        k_16 = k_16 + 1U;
    }
    *depth_1 = best_1;
    float3 _S864 = box_axis_0(b_22, axis_4);
    *normal_3 = _S864 * float3(side_0) ;
    return true;
}

bool penetration_1(const Box_0 thread* b_23, float3 p_4, float thread* depth_2, float3 thread* normal_4, KernelContext_0 thread* kernelContext_48)
{
    *depth_2 = 0.0f;
    *normal_4 = float3(0.0f) ;
    float3 r_10 = p_4 - b_23->center_1;
    float3 _S865 = b_23->half_2;
    if((dot(r_10, r_10)) > (dot(b_23->half_2, b_23->half_2) * 1.00001001358032227f))
    {
        return false;
    }
    uint _S866 = b_23->hull_v_0;
    if((b_23->hull_v_0) != 0U)
    {
        float3 _S867 = box_to_local_0(b_23, r_10);
        thread uint face_2;
        float _S868 = hull_signed_distance_0(b_23, _S867, &face_2, kernelContext_48);
        if(!(_S868 < 0.0f))
        {
            return false;
        }
        *depth_2 = - _S868;
        float3 _S869 = box_to_world_0(b_23, (float4(*(kernelContext_48->loads_0+(b_23->hull_at_0 + _S866 + face_2))) ).xyz);
        *normal_4 = _S869;
        return true;
    }
    float best_2 = 1.00000001504746622e+30f;
    uint axis_5 = 0U;
    float side_1 = 1.0f;
    uint k_17 = 0U;
    for(;;)
    {
        if(k_17 < 3U)
        {
        }
        else
        {
            break;
        }
        float3 _S870 = box_axis_0(b_23, k_17);
        float local_5 = dot(r_10, _S870);
        float d_7 = comp3_0(_S865, k_17) - abs(local_5);
        if(d_7 <= 0.0f)
        {
            return false;
        }
        if(d_7 < best_2)
        {
            float _S871;
            if(local_5 >= 0.0f)
            {
                _S871 = 1.0f;
            }
            else
            {
                _S871 = -1.0f;
            }
            best_2 = d_7;
            axis_5 = k_17;
            side_1 = _S871;
        }
        k_17 = k_17 + 1U;
    }
    *depth_2 = best_2;
    float3 _S872 = box_axis_0(b_23, axis_5);
    *normal_4 = _S872 * float3(side_1) ;
    return true;
}

bool pair_point_0(const Box_0 thread* ba_0, const Box_0 thread* bb_0, uint na_0, uint e_3, float3 thread* p_5, float3 thread* n_3, float thread* d_8, KernelContext_0 thread* kernelContext_49)
{
    if(e_3 < na_0)
    {
        float3 _S873 = sample_point_1(ba_0, e_3, kernelContext_49);
        *p_5 = _S873;
        bool _S874 = penetration_1(bb_0, _S873, d_8, n_3, kernelContext_49);
        return _S874;
    }
    float3 _S875 = sample_point_1(bb_0, e_3 - na_0, kernelContext_49);
    *p_5 = _S875;
    bool _S876 = penetration_1(ba_0, _S875, d_8, n_3, kernelContext_49);
    if(!_S876)
    {
        return false;
    }
    *n_3 = - *n_3;
    return true;
}

bool is_nan_0(float x_10)
{
    return ((as_type<uint>((x_10))) & 2147483647U) > 2139095040U;
}

uint pair_count_sum_0(uint x_11, uint lanes_2)
{
    uint _S877;
    if(lanes_2 > 1U)
    {
        uint _S878 = simd_sum(x_11);
        _S877 = _S878;
    }
    else
    {
        _S877 = x_11;
    }
    return _S877;
}

float3 pair_sum3_0(float3 x_12, uint lanes_3)
{
    float3 _S879;
    if(lanes_3 > 1U)
    {
        float3 _S880 = simd_sum(x_12);
        _S879 = _S880;
    }
    else
    {
        _S879 = x_12;
    }
    return _S879;
}

float pair_sum_0(float x_13, uint lanes_4)
{
    float _S881;
    if(lanes_4 > 1U)
    {
        float _S882 = simd_sum(x_13);
        _S881 = _S882;
    }
    else
    {
        _S881 = x_13;
    }
    return _S881;
}

void pair_contact_0(uint i_10, uint lane_1, uint lanes_5, KernelContext_0 thread* kernelContext_50)
{
    uint at_5 = kernelContext_50->params_0->pair_index_0 + 6U * i_10;
    uint ca_0 = kernelContext_50->index_0[at_5];
    uint cb_0 = kernelContext_50->index_0[at_5 + 1U];
    uint _S883 = kernelContext_50->index_0[at_5 + 3U];
    float _S884 = (as_type<float>((kernelContext_50->index_0[at_5 + 4U])));
    float _S885 = (as_type<float>((kernelContext_50->index_0[at_5 + 5U])));
    float _S886 = kernelContext_50->params_0->dt_0;
    uint out_0 = kernelContext_50->params_0->slot_base_0 + 2U * kernelContext_50->index_0[at_5 + 2U];
    WorldPoint_0 _S887 = chunk_world_0(ca_0, kernelContext_50);
    WorldPoint_0 _S888 = chunk_world_0(cb_0, kernelContext_50);
    thread WorldPoint_0 _S889 = _S888;
    thread WorldPoint_0 _S890 = _S887;
    float3 _S891 = world_diff_0(&_S889, &_S890);
    bool touching_0 = !((length(_S891)) > ((float4((kernelContext_50->chunks_0+ca_0)->half_0) ).w + (float4((kernelContext_50->chunks_0+cb_0)->half_0) ).w));
    float4 _S892 = float4(*(kernelContext_50->scratch_0+(kernelContext_50->params_0->ledger_base_0 + i_10))) ;
    thread float4 ledger_2 = _S892;
    uint flags_1 = (as_type<uint>((_S892.w)));
    float3 _S893 = float3(0.0f) ;
    Box_0 _S894 = chunk_box_0(ca_0, _S893, kernelContext_50);
    Box_0 _S895 = chunk_box_0(cb_0, _S891, kernelContext_50);
    thread Box_0 _S896 = _S894;
    uint _S897 = sample_count_0(&_S896);
    thread Box_0 _S898 = _S895;
    uint _S899 = sample_count_0(&_S898);
    uint _S900 = _S897 + _S899;
    bool has_state_0;
    uint e_4;
    if(!touching_0)
    {
        if((flags_1 & 1U) != 0U)
        {
            e_4 = lane_1;
            for(;;)
            {
                if(e_4 < _S900)
                {
                }
                else
                {
                    break;
                }
                *(kernelContext_50->contact_state_0+(_S883 + e_4)) = packed_float4(float4((as_type<float>((2143289344U))), 0.0f, 0.0f, 0.0f)) ;
                e_4 = e_4 + lanes_5;
            }
        }
        if(lane_1 == 0U)
        {
            if((flags_1 & 2U) != 0U)
            {
                e_4 = 0U;
                for(;;)
                {
                    if(e_4 < 4U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    *(kernelContext_50->scratch_0+(out_0 + e_4)) = packed_float4(float4(0.0f) ) ;
                    e_4 = e_4 + 1U;
                }
            }
            if(flags_1 != 0U)
            {
                has_state_0 = true;
            }
            else
            {
                has_state_0 = (ledger_2.x) != 0.0f;
            }
            if(has_state_0)
            {
                ledger_2.x = 0.0f;
                ledger_2.w = (as_type<float>((0U)));
                *(kernelContext_50->scratch_0+(kernelContext_50->params_0->ledger_base_0 + i_10)) = packed_float4(ledger_2) ;
            }
        }
        return;
    }
    if(lanes_5 > 1U)
    {
        thread Box_0 _S901 = _S894;
        thread Box_0 _S902 = _S895;
        bool _S903 = may_overlap_lanes_0(&_S901, &_S902, lane_1, lanes_5);
        has_state_0 = _S903;
    }
    else
    {
        thread Box_0 _S904 = _S894;
        thread Box_0 _S905 = _S895;
        bool _S906 = may_overlap_0(&_S904, &_S905);
        has_state_0 = _S906;
    }
    uint count_3;
    float3 fa_0;
    float3 ta_1;
    float3 fb_0;
    float3 tb_1;
    float stored_sum_0;
    float diss_sum_0;
    if(has_state_0)
    {
        thread float3 va0_0;
        thread float3 wa0_0;
        chunk_velocity_0(ca_0, &va0_0, &wa0_0, kernelContext_50);
        thread float3 vb0_0;
        thread float3 wb0_0;
        chunk_velocity_0(cb_0, &vb0_0, &wb0_0, kernelContext_50);
        e_4 = lane_1;
        count_3 = 0U;
        uint engaged_0 = 0U;
        for(;;)
        {
            if(e_4 < _S900)
            {
            }
            else
            {
                break;
            }
            thread Box_0 _S907 = _S894;
            thread Box_0 _S908 = _S895;
            thread float3 p_6;
            thread float3 n_4;
            thread float d_9;
            bool _S909 = pair_point_0(&_S907, &_S908, _S897, e_4, &p_6, &n_4, &d_9, kernelContext_50);
            if(!_S909)
            {
                uint _S910 = _S883 + e_4;
                if(!is_nan_0((float4(*(kernelContext_50->contact_state_0+_S910)) ).x))
                {
                    *(kernelContext_50->contact_state_0+_S910) = packed_float4(float4((as_type<float>((2143289344U))), 0.0f, 0.0f, 0.0f)) ;
                }
                e_4 = e_4 + lanes_5;
                continue;
            }
            uint _S911 = count_3 + 1U;
            uint _S912 = _S883 + e_4;
            float4 _S913 = float4(*(kernelContext_50->contact_state_0+_S912)) ;
            thread float4 entry_3 = _S913;
            if(is_nan_0(_S913.x))
            {
                has_state_0 = true;
            }
            else
            {
                has_state_0 = (dot(entry_3.yzw, n_4)) < 0.99000000953674316f;
            }
            if(has_state_0)
            {
                if(d_9 > (2.0f * abs(dot(va0_0 + cross(wa0_0, p_6) - (vb0_0 + cross(wb0_0, p_6 - _S895.center_1)), n_4)) * _S886 + 9.99999971718068537e-10f))
                {
                    stored_sum_0 = d_9;
                }
                else
                {
                    stored_sum_0 = 0.0f;
                }
                entry_3 = float4(stored_sum_0, n_4);
            }
            entry_3.x = min(entry_3.x, d_9);
            *(kernelContext_50->contact_state_0+_S912) = packed_float4(entry_3) ;
            uint engaged_1;
            if((d_9 - entry_3.x) > 0.0f)
            {
                engaged_1 = engaged_0 + 1U;
            }
            else
            {
                engaged_1 = engaged_0;
            }
            count_3 = _S911;
            engaged_0 = engaged_1;
            e_4 = e_4 + lanes_5;
        }
        uint count_4 = pair_count_sum_0(count_3, lanes_5);
        uint engaged_2 = pair_count_sum_0(engaged_0, lanes_5);
        if(count_4 > 0U)
        {
            float _S914 = (float4((kernelContext_50->chunks_0+ca_0)->cmat_0) ).x;
            float _S915 = (float4((kernelContext_50->chunks_0+cb_0)->cmat_0) ).x;
            float3 _S916 = _S895.center_1 - _S894.center_1;
            thread Box_0 _S917 = _S894;
            thread Box_0 _S918 = _S895;
            float _S919 = contact_stiffness_0(_S914, &_S917, _S915, &_S918, _S916);
            float _S920 = _S919 / max(float(engaged_2), 10.0f);
            e_4 = lane_1;
            fa_0 = _S893;
            ta_1 = _S893;
            fb_0 = _S893;
            tb_1 = _S893;
            stored_sum_0 = 0.0f;
            diss_sum_0 = 0.0f;
            for(;;)
            {
                if(e_4 < _S900)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S921 = _S894;
                thread Box_0 _S922 = _S895;
                thread float3 p_7;
                thread float3 n_5;
                thread float d_10;
                bool _S923 = pair_point_0(&_S921, &_S922, _S897, e_4, &p_7, &n_5, &d_10, kernelContext_50);
                if(!_S923)
                {
                    e_4 = e_4 + lanes_5;
                    continue;
                }
                float eff_0 = d_10 - (float4(*(kernelContext_50->contact_state_0+(_S883 + e_4))) ).x;
                if(eff_0 <= 0.0f)
                {
                    e_4 = e_4 + lanes_5;
                    continue;
                }
                thread float stored_6;
                thread float diss_4;
                float3 _S924 = penalty_force_0(_S920, _S884, _S885, eff_0, n_5, va0_0 + cross(wa0_0, p_7) - (vb0_0 + cross(wb0_0, p_7 - _S895.center_1)), _S886, engaged_2, &stored_6, &diss_4, kernelContext_50);
                float3 ta_2 = ta_1 + cross(p_7, _S924);
                float3 _S925 = - _S924;
                float3 fb_1 = fb_0 + _S925;
                float3 tb_2 = tb_1 + cross(p_7 - _S895.center_1, _S925);
                float stored_sum_1 = stored_sum_0 + stored_6;
                float diss_sum_1 = diss_sum_0 + diss_4;
                fa_0 = fa_0 + _S924;
                ta_1 = ta_2;
                fb_0 = fb_1;
                tb_1 = tb_2;
                stored_sum_0 = stored_sum_1;
                diss_sum_0 = diss_sum_1;
                e_4 = e_4 + lanes_5;
            }
            float3 _S926 = pair_sum3_0(fa_0, lanes_5);
            float3 _S927 = pair_sum3_0(ta_1, lanes_5);
            float3 _S928 = pair_sum3_0(fb_0, lanes_5);
            float3 _S929 = pair_sum3_0(tb_1, lanes_5);
            float _S930 = pair_sum_0(stored_sum_0, lanes_5);
            float _S931 = pair_sum_0(diss_sum_0, lanes_5);
            has_state_0 = true;
            diss_sum_0 = _S931;
            fa_0 = _S926;
            ta_1 = _S927;
            fb_0 = _S928;
            tb_1 = _S929;
            stored_sum_0 = _S930;
        }
        else
        {
            has_state_0 = false;
            diss_sum_0 = 0.0f;
            fa_0 = _S893;
            ta_1 = _S893;
            fb_0 = _S893;
            tb_1 = _S893;
            stored_sum_0 = 0.0f;
        }
    }
    else
    {
        has_state_0 = false;
        diss_sum_0 = 0.0f;
        fa_0 = _S893;
        ta_1 = _S893;
        fb_0 = _S893;
        tb_1 = _S893;
        stored_sum_0 = 0.0f;
    }
    bool loaded_0;
    if(!has_state_0)
    {
        loaded_0 = (flags_1 & 1U) != 0U;
    }
    else
    {
        loaded_0 = false;
    }
    if(loaded_0)
    {
        e_4 = lane_1;
        for(;;)
        {
            if(e_4 < _S900)
            {
            }
            else
            {
                break;
            }
            *(kernelContext_50->contact_state_0+(_S883 + e_4)) = packed_float4(float4((as_type<float>((2143289344U))), 0.0f, 0.0f, 0.0f)) ;
            e_4 = e_4 + lanes_5;
        }
    }
    if(lane_1 != 0U)
    {
        return;
    }
    if(has_state_0)
    {
        thread float _S932 = ledger_2.y;
        thread float _S933 = ledger_2.z;
        comp_add1_0(&_S932, &_S933, diss_sum_0);
        ledger_2.z = _S933;
        ledger_2.y = _S932;
    }
    float3 _S934 = float3(0.0f) ;
    if(any(fa_0 != _S934))
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = any(ta_1 != _S934);
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = any(fb_0 != _S934);
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = any(tb_1 != _S934);
    }
    bool _S935;
    if(loaded_0)
    {
        _S935 = true;
    }
    else
    {
        _S935 = (flags_1 & 2U) != 0U;
    }
    if(_S935)
    {
        *(kernelContext_50->scratch_0+out_0) = packed_float4(float4(fa_0, 0.0f)) ;
        *(kernelContext_50->scratch_0+(out_0 + 1U)) = packed_float4(float4(ta_1, 0.0f)) ;
        *(kernelContext_50->scratch_0+(out_0 + 2U)) = packed_float4(float4(fb_0, 0.0f)) ;
        *(kernelContext_50->scratch_0+(out_0 + 3U)) = packed_float4(float4(tb_1, 0.0f)) ;
    }
    ledger_2.x = stored_sum_0;
    if(has_state_0)
    {
        e_4 = 1U;
    }
    else
    {
        e_4 = 0U;
    }
    if(loaded_0)
    {
        count_3 = 2U;
    }
    else
    {
        count_3 = 0U;
    }
    ledger_2.w = (as_type<float>((e_4 | count_3)));
    *(kernelContext_50->scratch_0+(kernelContext_50->params_0->ledger_base_0 + i_10)) = packed_float4(ledger_2) ;
    if(loaded_0)
    {
        sentinel_mark_0(1U, kernelContext_50);
    }
    return;
}

struct Impactor_0
{
    float4 position_1;
    float4 position_err_1;
    float4 velocity_1;
    float4 velocity_err_1;
    float4 angular_velocity_1;
    float4 rotation_1;
    float4 inertia0_2;
    float4 inertia1_2;
    float4 inertia2_2;
    float4 inv0_2;
    float4 inv1_2;
    float4 inv2_2;
    float4 shape_0;
    float4 half_1;
    float4 mat_0;
    float4 crush_0;
    float4 geom_0;
    float4 rotation_err_1;
    float4 ledger_0;
    uint4 cand_0;
    float4 momentum_1;
    float4 momentum_err_1;
};

Box_0 impactor_box_0(const Impactor_0 thread* imp_0, float3 center_3, float3 half_3)
{
    Quat_0 q_11 = quat_of_0(imp_0->rotation_1);
    thread Box_0 b_24;
    (&b_24)->center_1 = center_3;
    float3 _S936 = float3(1.0f, 0.0f, 0.0f);
    thread Quat_0 _S937 = q_11;
    float3 _S938 = rotate_0(&_S937, _S936);
    (&b_24)->axis0_0 = _S938;
    float3 _S939 = float3(0.0f, 1.0f, 0.0f);
    thread Quat_0 _S940 = q_11;
    float3 _S941 = rotate_0(&_S940, _S939);
    (&b_24)->axis1_0 = _S941;
    float3 _S942 = float3(0.0f, 0.0f, 1.0f);
    thread Quat_0 _S943 = q_11;
    float3 _S944 = rotate_0(&_S943, _S942);
    (&b_24)->axis2_0 = _S944;
    (&b_24)->half_2 = half_3;
    (&b_24)->hull_at_0 = 0U;
    (&b_24)->hull_v_0 = 0U;
    (&b_24)->hull_f_0 = 0U;
    return b_24;
}

uint impactor_slots_0(const Impactor_natural_0 thread* imp_1, const Box_0 thread* b_25)
{
    uint _S945;
    if(((float4(imp_1->shape_0) ).x) == 0.0f)
    {
        _S945 = 1U;
    }
    else
    {
        uint _S946 = sample_count_0(b_25);
        _S945 = _S946 + 14U;
    }
    return _S945;
}

uint impactor_slots_1(const Impactor_natural_0 thread* imp_2, const Box_0 thread* b_26)
{
    uint _S947;
    if(((float4(imp_2->shape_0) ).x) == 0.0f)
    {
        _S947 = 1U;
    }
    else
    {
        uint _S948 = sample_count_0(b_26);
        _S947 = _S948 + 14U;
    }
    return _S947;
}

bool sphere_contact_0(const Box_0 thread* b_27, float3 center_4, float radius_0, float3 thread* point_0, float3 thread* normal_5, float thread* depth_3, KernelContext_0 thread* kernelContext_51)
{
    float3 _S949 = float3(0.0f) ;
    *point_0 = _S949;
    *normal_5 = _S949;
    *depth_3 = 0.0f;
    float3 _S950 = b_27->center_1;
    float3 r_11 = center_4 - b_27->center_1;
    float3 _S951 = b_27->axis0_0;
    float3 _S952 = b_27->axis1_0;
    float3 _S953 = b_27->axis2_0;
    float3 local_6 = float3(dot(r_11, b_27->axis0_0), dot(r_11, b_27->axis1_0), dot(r_11, b_27->axis2_0));
    uint _S954 = b_27->hull_v_0;
    if((b_27->hull_v_0) != 0U)
    {
        thread uint face_3;
        float _S955 = hull_signed_distance_0(b_27, local_6, &face_3, kernelContext_51);
        if(_S955 >= radius_0)
        {
            return false;
        }
        float3 _S956 = box_to_world_0(b_27, (float4(*(kernelContext_51->loads_0+(b_27->hull_at_0 + _S954 + face_3))) ).xyz);
        *normal_5 = _S956;
        *point_0 = center_4 - _S956 * float3(max(_S955, 0.0f)) ;
        *depth_3 = radius_0 - _S955;
        return true;
    }
    float3 q_12 = clamp(local_6, - b_27->half_2, b_27->half_2);
    float3 d_11 = local_6 - q_12;
    float dist_0 = length(d_11);
    if(dist_0 > 9.999999960041972e-13f)
    {
        if(dist_0 >= radius_0)
        {
            return false;
        }
        float3 dn_0 = d_11 / float3(dist_0) ;
        *normal_5 = _S951 * float3(dn_0.x)  + _S952 * float3(dn_0.y)  + _S953 * float3(dn_0.z) ;
        *point_0 = _S950 + _S951 * float3(q_12.x)  + _S952 * float3(q_12.y)  + _S953 * float3(q_12.z) ;
        *depth_3 = radius_0 - dist_0;
        return true;
    }
    thread float inside_0;
    thread float3 n_6;
    bool _S957 = penetration_0(b_27, center_4, &inside_0, &n_6, kernelContext_51);
    if(!_S957)
    {
        return false;
    }
    *normal_5 = n_6;
    *point_0 = center_4 - n_6 * float3(min(radius_0, inside_0)) ;
    *depth_3 = radius_0 + inside_0;
    return true;
}

bool impactor_contact_0(const Impactor_natural_0 thread* imp_3, float crush_depth_0, const Box_0 thread* shrunk_0, const Box_0 thread* b_28, uint j_2, float3 thread* p_8, float3 thread* n_7, float thread* d_12, KernelContext_0 thread* kernelContext_52)
{
    float3 _S958 = float3(0.0f) ;
    *p_8 = _S958;
    *n_7 = _S958;
    *d_12 = 0.0f;
    float4 _S959 = float4(imp_3->shape_0) ;
    if((_S959.x) == 0.0f)
    {
        bool _S960 = sphere_contact_0(b_28, _S958, _S959.y - crush_depth_0, p_8, n_7, d_12, kernelContext_52);
        if(!_S960)
        {
            return false;
        }
        *n_7 = - *n_7;
        return true;
    }
    uint _S961 = sample_count_0(b_28);
    if(j_2 < _S961)
    {
        float3 _S962 = sample_point_0(b_28, j_2, kernelContext_52);
        *p_8 = _S962;
        bool _S963 = penetration_0(shrunk_0, _S962, d_12, n_7, kernelContext_52);
        return _S963;
    }
    float3 _S964 = sample_point_0(shrunk_0, j_2 - _S961, kernelContext_52);
    *p_8 = _S964;
    bool _S965 = penetration_0(b_28, _S964, d_12, n_7, kernelContext_52);
    if(!_S965)
    {
        return false;
    }
    *n_7 = - *n_7;
    return true;
}

bool impactor_contact_1(const Impactor_natural_0 thread* imp_4, float crush_depth_1, const Box_0 thread* shrunk_1, const Box_0 thread* b_29, uint j_3, float3 thread* p_9, float3 thread* n_8, float thread* d_13, KernelContext_0 thread* kernelContext_53)
{
    float3 _S966 = float3(0.0f) ;
    *p_9 = _S966;
    *n_8 = _S966;
    *d_13 = 0.0f;
    float4 _S967 = float4(imp_4->shape_0) ;
    if((_S967.x) == 0.0f)
    {
        bool _S968 = sphere_contact_0(b_29, _S966, _S967.y - crush_depth_1, p_9, n_8, d_13, kernelContext_53);
        if(!_S968)
        {
            return false;
        }
        *n_8 = - *n_8;
        return true;
    }
    uint _S969 = sample_count_0(b_29);
    if(j_3 < _S969)
    {
        float3 _S970 = sample_point_0(b_29, j_3, kernelContext_53);
        *p_9 = _S970;
        bool _S971 = penetration_0(shrunk_1, _S970, d_13, n_8, kernelContext_53);
        return _S971;
    }
    float3 _S972 = sample_point_0(shrunk_1, j_3 - _S969, kernelContext_53);
    *p_9 = _S972;
    bool _S973 = penetration_0(b_29, _S972, d_13, n_8, kernelContext_53);
    if(!_S973)
    {
        return false;
    }
    *n_8 = - *n_8;
    return true;
}

WorldPoint_0 impactor_point_0(uint _S974, KernelContext_0 thread* kernelContext_54)
{
    Impactor_natural_0 device* _S975 = kernelContext_54->impactors_0+_S974;
    thread WorldPoint_0 wi_0;
    (&wi_0)->hi_0 = (float4(_S975->position_1) ).xyz;
    (&wi_0)->lo_0 = (float4(_S975->position_err_1) ).xyz;
    (&wi_0)->rel_0 = float3(0.0f) ;
    return wi_0;
}

Box_0 impactor_box_1(uint _S976, float3 _S977, float3 _S978, KernelContext_0 thread* kernelContext_55)
{
    Quat_0 q_13 = quat_of_0(float4((kernelContext_55->impactors_0+_S976)->rotation_1) );
    thread Box_0 b_30;
    (&b_30)->center_1 = _S977;
    float3 _S979 = float3(1.0f, 0.0f, 0.0f);
    thread Quat_0 _S980 = q_13;
    float3 _S981 = rotate_0(&_S980, _S979);
    (&b_30)->axis0_0 = _S981;
    float3 _S982 = float3(0.0f, 1.0f, 0.0f);
    thread Quat_0 _S983 = q_13;
    float3 _S984 = rotate_0(&_S983, _S982);
    (&b_30)->axis1_0 = _S984;
    float3 _S985 = float3(0.0f, 0.0f, 1.0f);
    thread Quat_0 _S986 = q_13;
    float3 _S987 = rotate_0(&_S986, _S985);
    (&b_30)->axis2_0 = _S987;
    (&b_30)->half_2 = _S978;
    (&b_30)->hull_at_0 = 0U;
    (&b_30)->hull_v_0 = 0U;
    (&b_30)->hull_f_0 = 0U;
    return b_30;
}

Box_0 impactor_shrunk_0(uint _S988, float _S989, const Box_0 thread* _S990)
{
    thread Box_0 shrunk_2 = *_S990;
    (&shrunk_2)->half_2 = _S990->half_2 - min(float3(_S989) , _S990->half_2 * float3(0.5f) );
    return shrunk_2;
}

bool impactor_contact_2(uint _S991, float _S992, const Box_0 thread* _S993, const Box_0 thread* _S994, uint _S995, float3 thread* _S996, float3 thread* _S997, float thread* _S998, KernelContext_0 thread* kernelContext_56)
{
    Impactor_natural_0 _S999 = *(kernelContext_56->impactors_0+_S991);
    float3 _S1000 = float3(0.0f) ;
    *_S996 = _S1000;
    *_S997 = _S1000;
    *_S998 = 0.0f;
    float4 _S1001 = float4(_S999.shape_0) ;
    if((_S1001.x) == 0.0f)
    {
        bool _S1002 = sphere_contact_0(_S994, _S1000, _S1001.y - _S992, _S996, _S997, _S998, kernelContext_56);
        if(!_S1002)
        {
            return false;
        }
        *_S997 = - *_S997;
        return true;
    }
    uint _S1003 = sample_count_0(_S994);
    if(_S995 < _S1003)
    {
        float3 _S1004 = sample_point_0(_S994, _S995, kernelContext_56);
        *_S996 = _S1004;
        bool _S1005 = penetration_0(_S993, _S1004, _S998, _S997, kernelContext_56);
        return _S1005;
    }
    float3 _S1006 = sample_point_0(_S993, _S995 - _S1003, kernelContext_56);
    *_S996 = _S1006;
    bool _S1007 = penetration_0(_S994, _S1006, _S998, _S997, kernelContext_56);
    if(!_S1007)
    {
        return false;
    }
    *_S997 = - *_S997;
    return true;
}

void impactor_candidate_forces_0(uint k_18, uint lane_2, uint lanes_6, KernelContext_0 thread* kernelContext_57)
{
    uint _S1008;
    bool _S1009;
    uint _S1010 = 3U * k_18;
    uint at_6 = kernelContext_57->params_0->cand_index_0 + _S1010;
    uint c_29 = kernelContext_57->index_0[at_6];
    uint slot_1 = kernelContext_57->index_0[at_6 + 1U];
    uint _S1011 = kernelContext_57->index_0[at_6 + 2U];
    thread Impactor_natural_0 _S1012 = *(kernelContext_57->impactors_0+_S1011);
    float _S1013 = kernelContext_57->params_0->dt_0;
    float3 _S1014 = float3(0.0f) ;
    thread float4 data_0 = float4(*(kernelContext_57->scratch_0+(kernelContext_57->params_0->cand_base_0 + _S1010))) ;
    bool hit_0;
    float3 f_sum_0;
    float3 t_sum_0;
    float3 imp_f_0;
    float3 imp_t_0;
    if(((uint4((&_S1012)->cand_0) ).z) == 0U)
    {
        WorldPoint_0 _S1015 = chunk_world_0(c_29, kernelContext_57);
        WorldPoint_0 _S1016 = impactor_point_0(_S1011, kernelContext_57);
        thread WorldPoint_0 _S1017 = _S1015;
        thread WorldPoint_0 _S1018 = _S1016;
        float3 _S1019 = world_diff_0(&_S1017, &_S1018);
        float4 _S1020 = float4((&_S1012)->half_1) ;
        if(!((length(_S1019)) > (_S1020.w + (float4((kernelContext_57->chunks_0+c_29)->half_0) ).w)))
        {
            Box_0 _S1021 = impactor_box_1(_S1011, _S1014, _S1020.xyz, kernelContext_57);
            Box_0 _S1022 = chunk_box_0(c_29, _S1019, kernelContext_57);
            float4 _S1023 = float4((&_S1012)->mat_0) ;
            float _S1024 = _S1023.x;
            float _S1025 = (float4((kernelContext_57->chunks_0+c_29)->cmat_0) ).x;
            float3 _S1026 = _S1022.center_1 - _S1021.center_1;
            thread Box_0 _S1027 = _S1021;
            thread Box_0 _S1028 = _S1022;
            float _S1029 = contact_stiffness_0(_S1024, &_S1027, _S1025, &_S1028, _S1026);
            float4 _S1030 = float4((&_S1012)->geom_0) ;
            float _S1031 = _S1030.y;
            thread Box_0 _S1032 = _S1021;
            Box_0 _S1033 = impactor_shrunk_0(_S1011, _S1031, &_S1032);
            uint j_4 = lane_2;
            uint n_9 = 0U;
            for(;;)
            {
                thread Box_0 _S1034 = _S1022;
                uint _S1035 = impactor_slots_0(&_S1012, &_S1034);
                _S1008 = _S1035;
                if(j_4 < _S1035)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S1036 = _S1033;
                thread Box_0 _S1037 = _S1022;
                thread float3 p_10;
                thread float3 nrm_0;
                thread float dep_0;
                bool _S1038 = impactor_contact_2(_S1011, _S1031, &_S1036, &_S1037, j_4, &p_10, &nrm_0, &dep_0, kernelContext_57);
                if(_S1038)
                {
                    n_9 = n_9 + 1U;
                }
                j_4 = j_4 + lanes_6;
            }
            uint n_10 = pair_count_sum_0(n_9, lanes_6);
            bool _S1039 = ((float4((&_S1012)->shape_0) ).x) == 0.0f;
            float _S1040;
            if(_S1039)
            {
                _S1040 = _S1029;
            }
            else
            {
                _S1040 = _S1029 / max(float(n_10), 10.0f);
            }
            if(_S1039)
            {
                j_4 = 1U;
            }
            else
            {
                j_4 = n_10;
            }
            float m_4 = (float4((kernelContext_57->chunks_0+c_29)->center_0) ).w;
            float _S1041 = _S1023.z;
            float _S1042 = m_4 * _S1041 / (m_4 + _S1041);
            float _S1043;
            if((kernelContext_57->params_0->pair_friction_0) >= 0.0f)
            {
                _S1043 = kernelContext_57->params_0->pair_friction_0;
            }
            else
            {
                _S1043 = min(_S1023.y, (float4((kernelContext_57->chunks_0+c_29)->cmat_0) ).y);
            }
            float3 _S1044 = (float4((&_S1012)->velocity_1) ).xyz + (float4((&_S1012)->velocity_err_1) ).xyz;
            thread float3 vc_1;
            thread float3 wc_1;
            chunk_velocity_0(c_29, &vc_1, &wc_1, kernelContext_57);
            hit_0 = false;
            uint j_5 = lane_2;
            f_sum_0 = _S1014;
            t_sum_0 = _S1014;
            imp_f_0 = _S1014;
            imp_t_0 = _S1014;
            float diss_sum_2 = 0.0f;
            for(;;)
            {
                bool _S1045 = n_10 > 0U;
                _S1009 = _S1045;
                bool _S1046;
                if(_S1045)
                {
                    _S1046 = j_5 < _S1008;
                }
                else
                {
                    _S1046 = false;
                }
                if(_S1046)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S1047 = _S1033;
                thread Box_0 _S1048 = _S1022;
                thread float3 p_11;
                thread float3 nrm_1;
                thread float dep_1;
                bool _S1049 = impactor_contact_0(&_S1012, _S1031, &_S1047, &_S1048, j_5, &p_11, &nrm_1, &dep_1, kernelContext_57);
                if(!_S1049)
                {
                    j_5 = j_5 + lanes_6;
                    continue;
                }
                thread float stored_7;
                thread float diss_5;
                float3 _S1050 = penalty_force_0(_S1040, _S1042, _S1043, dep_1 * _S1030.x, nrm_1, vc_1 + cross(wc_1, p_11 - _S1022.center_1) - (_S1044 + cross((float4((&_S1012)->angular_velocity_1) ).xyz, p_11)), _S1013, j_4, &stored_7, &diss_5, kernelContext_57);
                float3 f_sum_1 = f_sum_0 + _S1050;
                float3 t_sum_1 = t_sum_0 + cross(p_11 - _S1022.center_1, _S1050);
                float3 imp_f_1 = imp_f_0 - _S1050;
                float3 imp_t_1 = imp_t_0 - cross(p_11, _S1050);
                float diss_sum_3 = diss_sum_2 + diss_5;
                hit_0 = true;
                f_sum_0 = f_sum_1;
                t_sum_0 = t_sum_1;
                imp_f_0 = imp_f_1;
                imp_t_0 = imp_t_1;
                diss_sum_2 = diss_sum_3;
                j_5 = j_5 + lanes_6;
            }
            float3 _S1051 = pair_sum3_0(f_sum_0, lanes_6);
            float3 _S1052 = pair_sum3_0(t_sum_0, lanes_6);
            float3 _S1053 = pair_sum3_0(imp_f_0, lanes_6);
            float3 _S1054 = pair_sum3_0(imp_t_0, lanes_6);
            float diss_sum_4 = pair_sum_0(diss_sum_2, lanes_6);
            bool _S1055 = pair_any_0(hit_0, lanes_6);
            if(_S1009)
            {
                thread float _S1056 = data_0.z;
                thread float _S1057 = data_0.w;
                comp_add1_0(&_S1056, &_S1057, diss_sum_4);
                data_0.w = _S1057;
                data_0.z = _S1056;
            }
            hit_0 = _S1055;
            f_sum_0 = _S1051;
            t_sum_0 = _S1052;
            imp_f_0 = _S1053;
            imp_t_0 = _S1054;
        }
        else
        {
            hit_0 = false;
            f_sum_0 = _S1014;
            t_sum_0 = _S1014;
            imp_f_0 = _S1014;
            imp_t_0 = _S1014;
        }
    }
    else
    {
        hit_0 = false;
        f_sum_0 = _S1014;
        t_sum_0 = _S1014;
        imp_f_0 = _S1014;
        imp_t_0 = _S1014;
    }
    if(lane_2 != 0U)
    {
        return;
    }
    uint _S1058 = 2U * slot_1;
    *(kernelContext_57->scratch_0+(kernelContext_57->params_0->slot_base_0 + _S1058)) = packed_float4(float4(f_sum_0, 0.0f)) ;
    *(kernelContext_57->scratch_0+(kernelContext_57->params_0->slot_base_0 + _S1058 + 1U)) = packed_float4(float4(t_sum_0, 0.0f)) ;
    *(kernelContext_57->scratch_0+(kernelContext_57->params_0->cand_base_0 + _S1010)) = packed_float4(data_0) ;
    *(kernelContext_57->scratch_0+(kernelContext_57->params_0->cand_base_0 + _S1010 + 1U)) = packed_float4(float4(imp_f_0, 0.0f)) ;
    *(kernelContext_57->scratch_0+(kernelContext_57->params_0->cand_base_0 + _S1010 + 2U)) = packed_float4(float4(imp_t_0, 0.0f)) ;
    if(hit_0)
    {
        sentinel_mark_0(2U, kernelContext_57);
    }
    return;
}

void impactor_travel_check_0(uint ii_0, bool fused_2, KernelContext_0 thread* kernelContext_58)
{
    Impactor_natural_0 device* _S1059 = kernelContext_58->impactors_0+ii_0;
    uint at_7 = kernelContext_58->params_0->iplan_base_0 + 4U * ii_0;
    float4 _S1060 = float4(*(kernelContext_58->scratch_0+at_7)) ;
    uint4 _S1061 = uint4(_S1059->cand_0) ;
    bool _S1062;
    if((_S1061.z) != 0U)
    {
        _S1062 = true;
    }
    else
    {
        _S1062 = (_S1060.w) <= 0.0f;
    }
    if(_S1062)
    {
        return;
    }
    float4 _S1063 = float4(*(kernelContext_58->scratch_0+(at_7 + 1U))) ;
    float4 _S1064 = float4(*(kernelContext_58->scratch_0+(at_7 + 2U))) ;
    float4 _S1065 = float4(*(kernelContext_58->scratch_0+(at_7 + 3U))) ;
    uint _S1066 = _S1061.w;
    float3 moved_1 = (float4(_S1059->position_1) ).xyz - _S1060.xyz + ((float4(_S1059->position_err_1) ).xyz - _S1063.xyz);
    float3 _S1067 = float3((kernelContext_58->params_0->t_hi_0 - _S1065.w + (kernelContext_58->params_0->t_lo_0 - _S1063.w) + float(_S1066 - kernelContext_58->params_0->step_start_0) * kernelContext_58->params_0->dt_0)) ;
    float3 expected_0 = _S1065.xyz * _S1067 + float3(0.5f)  * kernelContext_58->params_0->gravity_0.xyz * _S1067 * _S1067;
    Quat_0 _S1068 = quat_of_0(float4(_S1059->rotation_1) );
    thread Quat_0 _S1069 = quat_of_0(float4(- _S1064.xyz, _S1064.w));
    thread Quat_0 _S1070 = _S1068;
    Quat_0 _S1071 = quat_mul_0(&_S1069, &_S1070);
    if((length(moved_1 - expected_0) + 2.0f * atan2(length(float3(_S1071.x_0, _S1071.y_0, _S1071.z_0)), abs(_S1071.w_0)) * (float4(_S1059->half_1) ).w) > (_S1060.w))
    {
        travel_stop_0(_S1066 + 1U, fused_2, kernelContext_58);
    }
    return;
}

void contact_forces_at_0(uint i_11, bool fused_3, KernelContext_0 thread* kernelContext_59)
{
    uint _S1072 = pair_lane_count_0(kernelContext_59);
    uint pair_threads_0 = kernelContext_59->params_0->pair_count_0 * _S1072;
    if(i_11 < pair_threads_0)
    {
        bool _S1073 = stopped_0(kernelContext_59);
        bool _S1074 = pair_any_0(_S1073, _S1072);
        if(!_S1074)
        {
            uint _S1075 = i_11 / _S1072;
            uint _S1076 = i_11 % _S1072;
            pair_contact_0(_S1075, _S1076, _S1072, kernelContext_59);
        }
        return;
    }
    uint j_6 = i_11 - pair_threads_0;
    uint cand_threads_0 = kernelContext_59->params_0->cand_count_0 * _S1072;
    if(j_6 < cand_threads_0)
    {
        bool _S1077 = stopped_0(kernelContext_59);
        bool _S1078 = pair_any_0(_S1077, _S1072);
        if(!_S1078)
        {
            uint _S1079 = j_6 / _S1072;
            uint _S1080 = j_6 % _S1072;
            impactor_candidate_forces_0(_S1079, _S1080, _S1072, kernelContext_59);
        }
        return;
    }
    if(fused_3)
    {
        return;
    }
    uint _S1081 = j_6 - cand_threads_0;
    bool _S1082;
    if(_S1081 < (kernelContext_59->params_0->chunk_count_0))
    {
        bool _S1083 = stopped_0(kernelContext_59);
        _S1082 = !_S1083;
    }
    else
    {
        _S1082 = false;
    }
    if(_S1082)
    {
        travel_check_0(_S1081, false, kernelContext_59);
    }
    else
    {
        if((_S1081 - kernelContext_59->params_0->chunk_count_0) < (kernelContext_59->params_0->impactor_count_0))
        {
            bool _S1084 = stopped_0(kernelContext_59);
            _S1082 = !_S1084;
        }
        else
        {
            _S1082 = false;
        }
        if(_S1082)
        {
            impactor_travel_check_0(_S1081 - kernelContext_59->params_0->chunk_count_0, false, kernelContext_59);
        }
    }
    return;
}

[[kernel]] void contact_forces(uint3 id_0 [[thread_position_in_grid]], Params_0 constant* params_1 [[buffer(0)]], Island_natural_0 device* islands_1 [[buffer(9)]], uint device* index_1 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(3)]], packed_float4 device* state_3 [[buffer(6)]], packed_float4 device* scratch_1 [[buffer(8)]], packed_float4 device* contact_state_1 [[buffer(11)]], packed_float4 device* loads_1 [[buffer(5)]], Impactor_natural_0 device* impactors_1 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(7)]], BondStatic_natural_0 device* bonds_1 [[buffer(2)]], MaterialTable_0 constant* materials_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_60;
    (&kernelContext_60)->params_0 = params_1;
    (&kernelContext_60)->islands_0 = islands_1;
    (&kernelContext_60)->index_0 = index_1;
    (&kernelContext_60)->chunks_0 = chunks_1;
    (&kernelContext_60)->state_0 = state_3;
    (&kernelContext_60)->scratch_0 = scratch_1;
    (&kernelContext_60)->contact_state_0 = contact_state_1;
    (&kernelContext_60)->loads_0 = loads_1;
    (&kernelContext_60)->impactors_0 = impactors_1;
    (&kernelContext_60)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_60)->bonds_0 = bonds_1;
    (&kernelContext_60)->materials_0 = materials_1;
    threadgroup array<float4, int(32)> g_part_a_1;
    (&kernelContext_60)->g_part_a_0 = &g_part_a_1;
    threadgroup array<float4, int(32)> g_part_b_1;
    (&kernelContext_60)->g_part_b_0 = &g_part_b_1;
    threadgroup array<uint, int(5)> g_seen_1;
    (&kernelContext_60)->g_seen_0 = &g_seen_1;
    threadgroup uint g_wide_run_1;
    (&kernelContext_60)->g_wide_run_0 = &g_wide_run_1;
    threadgroup array<uint, int(4096)> g_label_1;
    (&kernelContext_60)->g_label_0 = &g_label_1;
    threadgroup uint g_changed_1;
    (&kernelContext_60)->g_changed_0 = &g_changed_1;
    threadgroup uint g_halt_1;
    (&kernelContext_60)->g_halt_0 = &g_halt_1;
    threadgroup uint g_run_1;
    (&kernelContext_60)->g_run_0 = &g_run_1;
    threadgroup uint g_sums_1;
    (&kernelContext_60)->g_sums_0 = &g_sums_1;
    threadgroup array<uint, int(512)> g_label_narrow_1;
    (&kernelContext_60)->g_label_narrow_0 = &g_label_narrow_1;
    threadgroup uint g_changed_narrow_1;
    (&kernelContext_60)->g_changed_narrow_0 = &g_changed_narrow_1;
    contact_forces_at_0(id_0.x, false, &kernelContext_60);
    return;
}

uint impactor_contact_count_0(uint _S1085, float _S1086, const Box_0 thread* _S1087, const Box_0 thread* _S1088, KernelContext_0 thread* kernelContext_61)
{
    thread Impactor_natural_0 _S1089 = *(kernelContext_61->impactors_0+_S1085);
    uint j_7 = 0U;
    uint count_5 = 0U;
    for(;;)
    {
        uint _S1090 = impactor_slots_1(&_S1089, _S1088);
        if(j_7 < _S1090)
        {
        }
        else
        {
            break;
        }
        thread float3 p_12;
        thread float3 n_11;
        thread float d_14;
        bool _S1091 = impactor_contact_2(_S1085, _S1086, _S1087, _S1088, j_7, &p_12, &n_11, &d_14, kernelContext_61);
        if(_S1091)
        {
            count_5 = count_5 + 1U;
        }
        j_7 = j_7 + 1U;
    }
    return count_5;
}

[[kernel]] void impactor_shares(uint3 id_1 [[thread_position_in_grid]], Params_0 constant* params_2 [[buffer(0)]], Island_natural_0 device* islands_2 [[buffer(9)]], uint device* index_2 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(3)]], packed_float4 device* state_4 [[buffer(6)]], packed_float4 device* scratch_2 [[buffer(8)]], packed_float4 device* contact_state_2 [[buffer(11)]], packed_float4 device* loads_2 [[buffer(5)]], Impactor_natural_0 device* impactors_2 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(7)]], BondStatic_natural_0 device* bonds_2 [[buffer(2)]], MaterialTable_0 constant* materials_2 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_62;
    (&kernelContext_62)->params_0 = params_2;
    (&kernelContext_62)->islands_0 = islands_2;
    (&kernelContext_62)->index_0 = index_2;
    (&kernelContext_62)->chunks_0 = chunks_2;
    (&kernelContext_62)->state_0 = state_4;
    (&kernelContext_62)->scratch_0 = scratch_2;
    (&kernelContext_62)->contact_state_0 = contact_state_2;
    (&kernelContext_62)->loads_0 = loads_2;
    (&kernelContext_62)->impactors_0 = impactors_2;
    (&kernelContext_62)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_62)->bonds_0 = bonds_2;
    (&kernelContext_62)->materials_0 = materials_2;
    threadgroup array<float4, int(32)> g_part_a_2;
    (&kernelContext_62)->g_part_a_0 = &g_part_a_2;
    threadgroup array<float4, int(32)> g_part_b_2;
    (&kernelContext_62)->g_part_b_0 = &g_part_b_2;
    threadgroup array<uint, int(5)> g_seen_2;
    (&kernelContext_62)->g_seen_0 = &g_seen_2;
    threadgroup uint g_wide_run_2;
    (&kernelContext_62)->g_wide_run_0 = &g_wide_run_2;
    threadgroup array<uint, int(4096)> g_label_2;
    (&kernelContext_62)->g_label_0 = &g_label_2;
    threadgroup uint g_changed_2;
    (&kernelContext_62)->g_changed_0 = &g_changed_2;
    threadgroup uint g_halt_2;
    (&kernelContext_62)->g_halt_0 = &g_halt_2;
    threadgroup uint g_run_2;
    (&kernelContext_62)->g_run_0 = &g_run_2;
    threadgroup uint g_sums_2;
    (&kernelContext_62)->g_sums_0 = &g_sums_2;
    threadgroup array<uint, int(512)> g_label_narrow_2;
    (&kernelContext_62)->g_label_narrow_0 = &g_label_narrow_2;
    threadgroup uint g_changed_narrow_2;
    (&kernelContext_62)->g_changed_narrow_0 = &g_changed_narrow_2;
    uint k_19 = id_1.x;
    bool _S1092;
    if(k_19 >= (params_2->cand_count_0))
    {
        _S1092 = true;
    }
    else
    {
        bool _S1093 = stopped_0(&kernelContext_62);
        _S1092 = _S1093;
    }
    if(_S1092)
    {
        return;
    }
    uint _S1094 = 3U * k_19;
    uint at_8 = (&kernelContext_62)->params_0->cand_index_0 + _S1094;
    uint c_30 = (&kernelContext_62)->index_0[at_8];
    uint _S1095 = (&kernelContext_62)->index_0[at_8 + 2U];
    thread Impactor_natural_0 _S1096 = *((&kernelContext_62)->impactors_0+_S1095);
    if(((uint4((&_S1096)->cand_0) ).z) == 0U)
    {
        _S1092 = ((float4((&_S1096)->crush_0) ).x) > 0.0f;
    }
    else
    {
        _S1092 = false;
    }
    float total_1;
    float ksum_0;
    if(_S1092)
    {
        WorldPoint_0 _S1097 = chunk_world_0(c_30, &kernelContext_62);
        WorldPoint_0 _S1098 = impactor_point_0(_S1095, &kernelContext_62);
        thread WorldPoint_0 _S1099 = _S1097;
        thread WorldPoint_0 _S1100 = _S1098;
        float3 _S1101 = world_diff_0(&_S1099, &_S1100);
        float4 _S1102 = float4((&_S1096)->half_1) ;
        if(!((length(_S1101)) > (_S1102.w + (float4(((&kernelContext_62)->chunks_0+c_30)->half_0) ).w)))
        {
            Box_0 _S1103 = impactor_box_1(_S1095, float3(0.0f) , _S1102.xyz, &kernelContext_62);
            Box_0 _S1104 = chunk_box_0(c_30, _S1101, &kernelContext_62);
            float _S1105 = (float4((&_S1096)->mat_0) ).x;
            float _S1106 = (float4(((&kernelContext_62)->chunks_0+c_30)->cmat_0) ).x;
            float3 _S1107 = _S1104.center_1 - _S1103.center_1;
            thread Box_0 _S1108 = _S1103;
            thread Box_0 _S1109 = _S1104;
            float _S1110 = contact_stiffness_0(_S1105, &_S1108, _S1106, &_S1109, _S1107);
            float _S1111 = (float4((&_S1096)->crush_0) ).w;
            thread Box_0 _S1112 = _S1103;
            Box_0 _S1113 = impactor_shrunk_0(_S1095, _S1111, &_S1112);
            thread Box_0 _S1114 = _S1113;
            thread Box_0 _S1115 = _S1104;
            uint _S1116 = impactor_contact_count_0(_S1095, _S1111, &_S1114, &_S1115, &kernelContext_62);
            float _S1117;
            if(((float4((&_S1096)->shape_0) ).x) == 0.0f)
            {
                _S1117 = _S1110;
            }
            else
            {
                _S1117 = _S1110 / max(float(_S1116), 10.0f);
            }
            uint j_8 = 0U;
            total_1 = 0.0f;
            ksum_0 = 0.0f;
            for(;;)
            {
                if(_S1116 > 0U)
                {
                    thread Box_0 _S1118 = _S1104;
                    uint _S1119 = impactor_slots_0(&_S1096, &_S1118);
                    _S1092 = j_8 < _S1119;
                }
                else
                {
                    _S1092 = false;
                }
                if(_S1092)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S1120 = _S1113;
                thread Box_0 _S1121 = _S1104;
                thread float3 p_13;
                thread float3 nrm_2;
                thread float dep_2;
                bool _S1122 = impactor_contact_1(&_S1096, _S1111, &_S1120, &_S1121, j_8, &p_13, &nrm_2, &dep_2, &kernelContext_62);
                if(!_S1122)
                {
                    j_8 = j_8 + 1U;
                    continue;
                }
                float ksum_1 = ksum_0 + _S1117;
                total_1 = total_1 + _S1117 * dep_2;
                ksum_0 = ksum_1;
                j_8 = j_8 + 1U;
            }
        }
        else
        {
            total_1 = 0.0f;
            ksum_0 = 0.0f;
        }
    }
    else
    {
        total_1 = 0.0f;
        ksum_0 = 0.0f;
    }
    float4 _S1123 = float4(*((&kernelContext_62)->scratch_0+((&kernelContext_62)->params_0->cand_base_0 + _S1094))) ;
    *((&kernelContext_62)->scratch_0+((&kernelContext_62)->params_0->cand_base_0 + _S1094)) = packed_float4(float4(total_1, ksum_0, _S1123.z, _S1123.w)) ;
    return;
}

[[kernel]] void impactor_crush(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], Params_0 constant* params_3 [[buffer(0)]], Island_natural_0 device* islands_3 [[buffer(9)]], uint device* index_3 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(3)]], packed_float4 device* state_5 [[buffer(6)]], packed_float4 device* scratch_3 [[buffer(8)]], packed_float4 device* contact_state_3 [[buffer(11)]], packed_float4 device* loads_3 [[buffer(5)]], Impactor_natural_0 device* impactors_3 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_3 [[buffer(7)]], BondStatic_natural_0 device* bonds_3 [[buffer(2)]], MaterialTable_0 constant* materials_3 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_63;
    (&kernelContext_63)->params_0 = params_3;
    (&kernelContext_63)->islands_0 = islands_3;
    (&kernelContext_63)->index_0 = index_3;
    (&kernelContext_63)->chunks_0 = chunks_3;
    (&kernelContext_63)->state_0 = state_5;
    (&kernelContext_63)->scratch_0 = scratch_3;
    (&kernelContext_63)->contact_state_0 = contact_state_3;
    (&kernelContext_63)->loads_0 = loads_3;
    (&kernelContext_63)->impactors_0 = impactors_3;
    (&kernelContext_63)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_63)->bonds_0 = bonds_3;
    (&kernelContext_63)->materials_0 = materials_3;
    threadgroup array<float4, int(32)> g_part_a_3;
    (&kernelContext_63)->g_part_a_0 = &g_part_a_3;
    threadgroup array<float4, int(32)> g_part_b_3;
    (&kernelContext_63)->g_part_b_0 = &g_part_b_3;
    threadgroup array<uint, int(5)> g_seen_3;
    (&kernelContext_63)->g_seen_0 = &g_seen_3;
    threadgroup uint g_wide_run_3;
    (&kernelContext_63)->g_wide_run_0 = &g_wide_run_3;
    threadgroup array<uint, int(4096)> g_label_3;
    (&kernelContext_63)->g_label_0 = &g_label_3;
    threadgroup uint g_changed_3;
    (&kernelContext_63)->g_changed_0 = &g_changed_3;
    threadgroup uint g_halt_3;
    (&kernelContext_63)->g_halt_0 = &g_halt_3;
    threadgroup uint g_run_3;
    (&kernelContext_63)->g_run_0 = &g_run_3;
    threadgroup uint g_sums_3;
    (&kernelContext_63)->g_sums_0 = &g_sums_3;
    threadgroup array<uint, int(512)> g_label_narrow_3;
    (&kernelContext_63)->g_label_narrow_0 = &g_label_narrow_3;
    threadgroup uint g_changed_narrow_3;
    (&kernelContext_63)->g_changed_narrow_0 = &g_changed_narrow_3;
    uint ii_1 = group_0.x;
    uint tid_8 = thread_0.x;
    bool _S1124;
    if(ii_1 >= (params_3->impactor_count_0))
    {
        _S1124 = true;
    }
    else
    {
        bool _S1125 = stopped_0(&kernelContext_63);
        _S1124 = _S1125;
    }
    if(_S1124)
    {
        return;
    }
    Impactor_natural_0 device* _S1126 = (&kernelContext_63)->impactors_0+ii_1;
    float4 _S1127 = float4((*_S1126).position_err_1) ;
    float4 _S1128 = float4((*_S1126).velocity_1) ;
    float4 _S1129 = float4((*_S1126).velocity_err_1) ;
    float4 _S1130 = float4((*_S1126).angular_velocity_1) ;
    float4 _S1131 = float4((*_S1126).rotation_1) ;
    float4 _S1132 = float4((*_S1126).inertia0_2) ;
    float4 _S1133 = float4((*_S1126).inertia1_2) ;
    float4 _S1134 = float4((*_S1126).inertia2_2) ;
    float4 _S1135 = float4((*_S1126).inv0_2) ;
    float4 _S1136 = float4((*_S1126).inv1_2) ;
    float4 _S1137 = float4((*_S1126).inv2_2) ;
    float4 _S1138 = float4((*_S1126).shape_0) ;
    float4 _S1139 = float4((*_S1126).half_1) ;
    float4 _S1140 = float4((*_S1126).mat_0) ;
    float4 _S1141 = float4((*_S1126).crush_0) ;
    float4 _S1142 = float4((*_S1126).geom_0) ;
    float4 _S1143 = float4((*_S1126).rotation_err_1) ;
    float4 _S1144 = float4((*_S1126).ledger_0) ;
    uint4 _S1145 = uint4((*_S1126).cand_0) ;
    float4 _S1146 = float4((*_S1126).momentum_1) ;
    float4 _S1147 = float4((*_S1126).momentum_err_1) ;
    thread Impactor_0 imp_5;
    (&imp_5)->position_1 = float4((*_S1126).position_1) ;
    (&imp_5)->position_err_1 = _S1127;
    (&imp_5)->velocity_1 = _S1128;
    (&imp_5)->velocity_err_1 = _S1129;
    (&imp_5)->angular_velocity_1 = _S1130;
    (&imp_5)->rotation_1 = _S1131;
    (&imp_5)->inertia0_2 = _S1132;
    (&imp_5)->inertia1_2 = _S1133;
    (&imp_5)->inertia2_2 = _S1134;
    (&imp_5)->inv0_2 = _S1135;
    (&imp_5)->inv1_2 = _S1136;
    (&imp_5)->inv2_2 = _S1137;
    (&imp_5)->shape_0 = _S1138;
    (&imp_5)->half_1 = _S1139;
    (&imp_5)->mat_0 = _S1140;
    (&imp_5)->crush_0 = _S1141;
    (&imp_5)->geom_0 = _S1142;
    (&imp_5)->rotation_err_1 = _S1143;
    (&imp_5)->ledger_0 = _S1144;
    (&imp_5)->cand_0 = _S1145;
    (&imp_5)->momentum_1 = _S1146;
    (&imp_5)->momentum_err_1 = _S1147;
    float4 _S1148 = float4(0.0f) ;
    thread float4 shares_0 = _S1148;
    thread float4 unused_4 = _S1148;
    uint k_20 = (&imp_5)->cand_0.x + tid_8;
    for(;;)
    {
        if(k_20 < ((&imp_5)->cand_0.y))
        {
        }
        else
        {
            break;
        }
        shares_0 = shares_0 + float4(*((&kernelContext_63)->scratch_0+((&kernelContext_63)->params_0->cand_base_0 + 3U * k_20))) ;
        k_20 = k_20 + 256U;
    }
    group_sum2_0(tid_8, &shares_0, &unused_4, &kernelContext_63);
    if(tid_8 != 0U)
    {
        _S1124 = true;
    }
    else
    {
        _S1124 = ((&imp_5)->cand_0.z) != 0U;
    }
    if(_S1124)
    {
        return;
    }
    (&imp_5)->geom_0 = float4(1.0f, (&imp_5)->crush_0.w, 0.0f, 0.0f);
    float total_2 = shares_0.x;
    if(((&imp_5)->crush_0.x) > 0.0f)
    {
        _S1124 = ((&imp_5)->crush_0.z) < ((&imp_5)->crush_0.y);
    }
    else
    {
        _S1124 = false;
    }
    if(_S1124)
    {
        _S1124 = total_2 > ((&imp_5)->crush_0.x);
    }
    else
    {
        _S1124 = false;
    }
    if(_S1124)
    {
        float extra_0 = (total_2 - (&imp_5)->crush_0.x) / shares_0.y;
        (&imp_5)->crush_0.w = (&imp_5)->crush_0.w + extra_0;
        (&imp_5)->crush_0.z = (&imp_5)->crush_0.z + (&imp_5)->crush_0.x * extra_0;
        float _S1149 = (&imp_5)->crush_0.x * extra_0;
        thread float _S1150 = (&imp_5)->ledger_0.z;
        thread float _S1151 = (&imp_5)->ledger_0.w;
        comp_add1_0(&_S1150, &_S1151, _S1149);
        (&imp_5)->ledger_0.w = _S1151;
        (&imp_5)->ledger_0.z = _S1150;
        float _S1152 = (&imp_5)->crush_0.x * extra_0;
        thread float _S1153 = (&imp_5)->ledger_0.x;
        thread float _S1154 = (&imp_5)->ledger_0.y;
        comp_add1_0(&_S1153, &_S1154, _S1152);
        (&imp_5)->ledger_0.y = _S1154;
        (&imp_5)->ledger_0.x = _S1153;
        (&imp_5)->geom_0.x = (&imp_5)->crush_0.x / total_2;
    }
    Impactor_natural_0 device* _S1155 = (&kernelContext_63)->impactors_0+ii_1;
    _S1155->position_1 = packed_float4(imp_5.position_1) ;
    _S1155->position_err_1 = packed_float4(imp_5.position_err_1) ;
    _S1155->velocity_1 = packed_float4(imp_5.velocity_1) ;
    _S1155->velocity_err_1 = packed_float4(imp_5.velocity_err_1) ;
    _S1155->angular_velocity_1 = packed_float4(imp_5.angular_velocity_1) ;
    _S1155->rotation_1 = packed_float4(imp_5.rotation_1) ;
    _S1155->inertia0_2 = packed_float4(imp_5.inertia0_2) ;
    _S1155->inertia1_2 = packed_float4(imp_5.inertia1_2) ;
    _S1155->inertia2_2 = packed_float4(imp_5.inertia2_2) ;
    _S1155->inv0_2 = packed_float4(imp_5.inv0_2) ;
    _S1155->inv1_2 = packed_float4(imp_5.inv1_2) ;
    _S1155->inv2_2 = packed_float4(imp_5.inv2_2) ;
    _S1155->shape_0 = packed_float4(imp_5.shape_0) ;
    _S1155->half_1 = packed_float4(imp_5.half_1) ;
    _S1155->mat_0 = packed_float4(imp_5.mat_0) ;
    _S1155->crush_0 = packed_float4(imp_5.crush_0) ;
    _S1155->geom_0 = packed_float4(imp_5.geom_0) ;
    _S1155->rotation_err_1 = packed_float4(imp_5.rotation_err_1) ;
    _S1155->ledger_0 = packed_float4(imp_5.ledger_0) ;
    _S1155->cand_0 = packed_uint4(imp_5.cand_0) ;
    _S1155->momentum_1 = packed_float4(imp_5.momentum_1) ;
    _S1155->momentum_err_1 = packed_float4(imp_5.momentum_err_1) ;
    return;
}

struct WideGroup_0
{
    uint island_3;
    uint begin_1;
    uint end_0;
    uint first_0;
};

WideGroup_0 wide_group_0(uint table_0, uint g_2, KernelContext_0 thread* kernelContext_64)
{
    thread WideGroup_0 w_8;
    uint _S1156 = table_0 + 4U * g_2;
    (&w_8)->island_3 = kernelContext_64->index_0[_S1156];
    (&w_8)->begin_1 = kernelContext_64->index_0[_S1156 + 1U];
    (&w_8)->end_0 = kernelContext_64->index_0[_S1156 + 2U];
    (&w_8)->first_0 = kernelContext_64->index_0[_S1156 + 3U];
    return w_8;
}

bool contact_stopped_1(const Island_natural_0 thread* isl_13, KernelContext_0 thread* kernelContext_65)
{
    Island_natural_0 device* _S1157 = kernelContext_65->islands_0+kernelContext_65->params_0->halt_index_0;
    uint4 _S1158 = uint4(_S1157->info_0) ;
    bool _S1159;
    if(((_S1158.z) & 1U) != 0U)
    {
        _S1159 = true;
    }
    else
    {
        uint _S1160 = _S1158.y;
        if(_S1160 != 0U)
        {
            _S1159 = _S1160 <= ((uint4(isl_13->info_0) ).w);
        }
        else
        {
            _S1159 = false;
        }
    }
    if(_S1159)
    {
        _S1159 = true;
    }
    else
    {
        uint _S1161 = (uint4(_S1157->range_0) ).x;
        if(_S1161 != 0U)
        {
            _S1159 = _S1161 <= ((uint4(isl_13->info_0) ).w + 1U);
        }
        else
        {
            _S1159 = false;
        }
    }
    return _S1159;
}

bool wide_runs_0(const Island_natural_0 thread* isl_14, KernelContext_0 thread* kernelContext_66)
{
    uint4 _S1162 = uint4(isl_14->info_0) ;
    bool _S1163;
    if(((_S1162.z) & 1U) != 0U)
    {
        _S1163 = true;
    }
    else
    {
        _S1163 = (_S1162.y) == 0U;
    }
    if(_S1163)
    {
        return false;
    }
    if(((_S1162.x) & 4U) == 0U)
    {
        _S1163 = true;
    }
    else
    {
        bool _S1164 = contact_stopped_1(isl_14, kernelContext_66);
        _S1163 = !_S1164;
    }
    return _S1163;
}

bool wide_enter_0(uint tid_9, const Island_natural_0 thread* isl_15, KernelContext_0 thread* kernelContext_67)
{
    seen_clear_0(tid_9, kernelContext_67);
    if(tid_9 == 0U)
    {
        bool _S1165 = wide_runs_0(isl_15, kernelContext_67);
        int _S1166;
        if(_S1165)
        {
            _S1166 = int(1);
        }
        else
        {
            _S1166 = int(0);
        }
        *kernelContext_67->g_wide_run_0 = uint(_S1166);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_67->g_wide_run_0) != 0U;
}

uint wide_step_0(const Island_natural_0 thread* isl_16, KernelContext_0 thread* kernelContext_68)
{
    return (uint4(isl_16->info_0) ).w - kernelContext_68->params_0->step_start_0;
}

uint wide_step_1(const Island_natural_0 thread* isl_17, KernelContext_0 thread* kernelContext_69)
{
    return (uint4(isl_17->info_0) ).w - kernelContext_69->params_0->step_start_0;
}

uint wide_step_2(const Island_0 thread* isl_18, KernelContext_0 thread* kernelContext_70)
{
    return isl_18->info_0.w - kernelContext_70->params_0->step_start_0;
}

bool wide_enter_1(uint _S1167, uint _S1168, KernelContext_0 thread* kernelContext_71)
{
    thread Island_natural_0 _S1169 = *(kernelContext_71->islands_0+_S1168);
    seen_clear_0(_S1167, kernelContext_71);
    if(_S1167 == 0U)
    {
        bool _S1170 = wide_runs_0(&_S1169, kernelContext_71);
        int _S1171;
        if(_S1170)
        {
            _S1171 = int(1);
        }
        else
        {
            _S1171 = int(0);
        }
        *kernelContext_71->g_wide_run_0 = uint(_S1171);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_71->g_wide_run_0) != 0U;
}

[[kernel]] void wide_wake(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], Params_0 constant* params_4 [[buffer(0)]], Island_natural_0 device* islands_4 [[buffer(9)]], uint device* index_4 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_4 [[buffer(3)]], packed_float4 device* state_6 [[buffer(6)]], packed_float4 device* scratch_4 [[buffer(8)]], packed_float4 device* contact_state_4 [[buffer(11)]], packed_float4 device* loads_4 [[buffer(5)]], Impactor_natural_0 device* impactors_4 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_4 [[buffer(7)]], BondStatic_natural_0 device* bonds_4 [[buffer(2)]], MaterialTable_0 constant* materials_4 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_72;
    (&kernelContext_72)->params_0 = params_4;
    (&kernelContext_72)->islands_0 = islands_4;
    (&kernelContext_72)->index_0 = index_4;
    (&kernelContext_72)->chunks_0 = chunks_4;
    (&kernelContext_72)->state_0 = state_6;
    (&kernelContext_72)->scratch_0 = scratch_4;
    (&kernelContext_72)->contact_state_0 = contact_state_4;
    (&kernelContext_72)->loads_0 = loads_4;
    (&kernelContext_72)->impactors_0 = impactors_4;
    (&kernelContext_72)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_72)->bonds_0 = bonds_4;
    (&kernelContext_72)->materials_0 = materials_4;
    threadgroup array<float4, int(32)> g_part_a_4;
    (&kernelContext_72)->g_part_a_0 = &g_part_a_4;
    threadgroup array<float4, int(32)> g_part_b_4;
    (&kernelContext_72)->g_part_b_0 = &g_part_b_4;
    threadgroup array<uint, int(5)> g_seen_4;
    (&kernelContext_72)->g_seen_0 = &g_seen_4;
    threadgroup uint g_wide_run_4;
    (&kernelContext_72)->g_wide_run_0 = &g_wide_run_4;
    threadgroup array<uint, int(4096)> g_label_4;
    (&kernelContext_72)->g_label_0 = &g_label_4;
    threadgroup uint g_changed_4;
    (&kernelContext_72)->g_changed_0 = &g_changed_4;
    threadgroup uint g_halt_4;
    (&kernelContext_72)->g_halt_0 = &g_halt_4;
    threadgroup uint g_run_4;
    (&kernelContext_72)->g_run_0 = &g_run_4;
    threadgroup uint g_sums_4;
    (&kernelContext_72)->g_sums_0 = &g_sums_4;
    threadgroup array<uint, int(512)> g_label_narrow_4;
    (&kernelContext_72)->g_label_narrow_0 = &g_label_narrow_4;
    threadgroup uint g_changed_narrow_4;
    (&kernelContext_72)->g_changed_narrow_0 = &g_changed_narrow_4;
    uint tid_10 = thread_1.x;
    uint _S1172 = group_1.x;
    WideGroup_0 _S1173 = wide_group_0(params_4->wide_chunk_table_0, _S1172, &kernelContext_72);
    if(_S1172 != (_S1173.first_0))
    {
        return;
    }
    thread Island_natural_0 _S1174 = *((&kernelContext_72)->islands_0+_S1173.island_3);
    uint4 _S1175 = uint4((&_S1174)->info_0) ;
    uint _S1176 = _S1175.x;
    bool _S1177;
    if((_S1176 & 16U) == 0U)
    {
        _S1177 = true;
    }
    else
    {
        bool _S1178 = wide_enter_1(tid_10, _S1173.island_3, &kernelContext_72);
        _S1177 = !_S1178;
    }
    if(_S1177)
    {
        return;
    }
    Quat_0 _S1179 = quat_of_0(float4((&_S1174)->rotation_0) );
    bool _S1180 = (_S1176 & 4U) != 0U;
    float3 _S1181 = float3(0.0f) ;
    thread float3 norm_4 = _S1181;
    thread float3 unused_5 = _S1181;
    uint4 _S1182 = uint4((&_S1174)->range_0) ;
    uint c_31 = _S1182.x + tid_10;
    for(;;)
    {
        if(c_31 < (_S1182.y))
        {
        }
        else
        {
            break;
        }
        uint _S1183 = wide_step_0(&_S1174, &kernelContext_72);
        float _S1184 = (&kernelContext_72)->params_0->dt_0;
        thread Quat_0 _S1185 = _S1179;
        float _S1186 = settled_chunk_load_0(c_31, &_S1185, _S1183, _S1184, _S1180, &kernelContext_72);
        norm_4.x = norm_4.x + _S1186;
        c_31 = c_31 + 256U;
    }
    group_sum3_0(tid_10, &norm_4, &unused_5, &kernelContext_72);
    if(tid_10 != 0U)
    {
        return;
    }
    uint _S1187 = seen_bits_0(&kernelContext_72);
    uint cov_20 = 1024U | _S1187;
    if(((&kernelContext_72)->params_0->solve_mode_0) == 1U)
    {
        float4 _S1188 = float4((&_S1174)->energy_0) ;
        _S1177 = (abs(norm_4.x - _S1188.z)) > (_S1188.w);
    }
    else
    {
        _S1177 = false;
    }
    uint cov_21;
    if(_S1177)
    {
        ((&kernelContext_72)->islands_0+_S1173.island_3)->info_0[int(0)] = _S1176 & 4294967279U;
        ((&kernelContext_72)->islands_0+_S1173.island_3)->info_0[int(2)] = (_S1175.z) | 4U;
        cov_21 = cov_20 | 2048U;
    }
    else
    {
        cov_21 = cov_20;
    }
    cov_mark_0(_S1173.island_3, (&_S1174)->coverage_0, cov_21, &kernelContext_72);
    return;
}

void wide_partials_0(uint tid_11, uint first_1, uint count_6, uint p_14, float3 thread* a_13, float3 thread* b_31, KernelContext_0 thread* kernelContext_73)
{
    float4 _S1189 = float4(0.0f) ;
    thread float4 x_14 = _S1189;
    thread float4 y_2 = _S1189;
    uint s_6 = tid_11;
    for(;;)
    {
        if(s_6 < count_6)
        {
        }
        else
        {
            break;
        }
        uint _S1190 = 8U * (first_1 + s_6);
        x_14 = x_14 + float4(*(kernelContext_73->scratch_0+(kernelContext_73->params_0->wide_base_0 + _S1190 + p_14))) ;
        y_2 = y_2 + float4(*(kernelContext_73->scratch_0+(kernelContext_73->params_0->wide_base_0 + _S1190 + p_14 + 1U))) ;
        s_6 = s_6 + 256U;
    }
    group_sum2_0(tid_11, &x_14, &y_2, kernelContext_73);
    *a_13 = x_14.xyz;
    *b_31 = y_2.xyz;
    return;
}

void wide_store_0(uint slot_2, uint p_15, float3 a_14, float3 b_32, KernelContext_0 thread* kernelContext_74)
{
    uint _S1191 = 8U * slot_2;
    *(kernelContext_74->scratch_0+(kernelContext_74->params_0->wide_base_0 + _S1191 + p_15)) = packed_float4(float4(a_14, 0.0f)) ;
    *(kernelContext_74->scratch_0+(kernelContext_74->params_0->wide_base_0 + _S1191 + p_15 + 1U)) = packed_float4(float4(b_32, 0.0f)) ;
    return;
}

[[kernel]] void wide_drift(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], Params_0 constant* params_5 [[buffer(0)]], Island_natural_0 device* islands_5 [[buffer(9)]], uint device* index_5 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_5 [[buffer(3)]], packed_float4 device* state_7 [[buffer(6)]], packed_float4 device* scratch_5 [[buffer(8)]], packed_float4 device* contact_state_5 [[buffer(11)]], packed_float4 device* loads_5 [[buffer(5)]], Impactor_natural_0 device* impactors_5 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_5 [[buffer(7)]], BondStatic_natural_0 device* bonds_5 [[buffer(2)]], MaterialTable_0 constant* materials_5 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_75;
    (&kernelContext_75)->params_0 = params_5;
    (&kernelContext_75)->islands_0 = islands_5;
    (&kernelContext_75)->index_0 = index_5;
    (&kernelContext_75)->chunks_0 = chunks_5;
    (&kernelContext_75)->state_0 = state_7;
    (&kernelContext_75)->scratch_0 = scratch_5;
    (&kernelContext_75)->contact_state_0 = contact_state_5;
    (&kernelContext_75)->loads_0 = loads_5;
    (&kernelContext_75)->impactors_0 = impactors_5;
    (&kernelContext_75)->bond_dyn_0 = bond_dyn_5;
    (&kernelContext_75)->bonds_0 = bonds_5;
    (&kernelContext_75)->materials_0 = materials_5;
    threadgroup array<float4, int(32)> g_part_a_5;
    (&kernelContext_75)->g_part_a_0 = &g_part_a_5;
    threadgroup array<float4, int(32)> g_part_b_5;
    (&kernelContext_75)->g_part_b_0 = &g_part_b_5;
    threadgroup array<uint, int(5)> g_seen_5;
    (&kernelContext_75)->g_seen_0 = &g_seen_5;
    threadgroup uint g_wide_run_5;
    (&kernelContext_75)->g_wide_run_0 = &g_wide_run_5;
    threadgroup array<uint, int(4096)> g_label_5;
    (&kernelContext_75)->g_label_0 = &g_label_5;
    threadgroup uint g_changed_5;
    (&kernelContext_75)->g_changed_0 = &g_changed_5;
    threadgroup uint g_halt_5;
    (&kernelContext_75)->g_halt_0 = &g_halt_5;
    threadgroup uint g_run_5;
    (&kernelContext_75)->g_run_0 = &g_run_5;
    threadgroup uint g_sums_5;
    (&kernelContext_75)->g_sums_0 = &g_sums_5;
    threadgroup array<uint, int(512)> g_label_narrow_5;
    (&kernelContext_75)->g_label_narrow_0 = &g_label_narrow_5;
    threadgroup uint g_changed_narrow_5;
    (&kernelContext_75)->g_changed_narrow_0 = &g_changed_narrow_5;
    uint tid_12 = thread_2.x;
    uint _S1192 = group_2.x;
    WideGroup_0 _S1193 = wide_group_0(params_5->wide_chunk_table_0, _S1192, &kernelContext_75);
    Island_natural_0 device* _S1194 = (&kernelContext_75)->islands_0+_S1193.island_3;
    Island_natural_0 isl_19 = *_S1194;
    bool _S1195;
    if((((uint4((*_S1194).info_0) ).x) & 17U) != 0U)
    {
        _S1195 = true;
    }
    else
    {
        bool _S1196 = wide_enter_1(tid_12, _S1193.island_3, &kernelContext_75);
        _S1195 = !_S1196;
    }
    if(_S1195)
    {
        return;
    }
    thread float3 tu_5;
    thread float3 pv_5;
    wide_partials_0(tid_12, _S1193.first_0, isl_19.wide_groups_0, 2U, &tu_5, &pv_5, &kernelContext_75);
    float4 _S1197 = float4(isl_19.wcom_0) ;
    float3 _S1198 = float3(_S1197.w) ;
    float3 tr_7 = tu_5 / _S1198;
    float3 dv_7 = pv_5 / _S1198;
    float3 _S1199 = float3(0.0f) ;
    thread float3 lu_5 = _S1199;
    thread float3 lv_5 = _S1199;
    uint c_32 = _S1193.begin_1 + tid_12;
    if(c_32 < (_S1193.end_0))
    {
        drift_angular_0(c_32, _S1197.xyz, tr_7, dv_7, &lu_5, &lv_5, &kernelContext_75);
    }
    group_sum3_0(tid_12, &lu_5, &lv_5, &kernelContext_75);
    if(tid_12 == 0U)
    {
        wide_store_0(_S1192, 4U, lu_5, lv_5, &kernelContext_75);
    }
    return;
}

Rigid_0 rigid_of_1(const Island_natural_0 thread* isl_20)
{
    thread Rigid_0 rg_11;
    (&rg_11)->rot_0 = quat_of_0(float4(isl_20->rotation_0) );
    (&rg_11)->rot_err_0 = float4(isl_20->rotation_err_0) ;
    (&rg_11)->pos_0 = (float4(isl_20->position_0) ).xyz;
    (&rg_11)->pos_err_0 = (float4(isl_20->position_err_0) ).xyz;
    (&rg_11)->vel_0 = (float4(isl_20->velocity_0) ).xyz;
    (&rg_11)->vel_err_0 = (float4(isl_20->velocity_err_0) ).xyz;
    (&rg_11)->w_1 = (float4(isl_20->angular_velocity_0) ).xyz;
    (&rg_11)->l_0 = (float4(isl_20->momentum_0) ).xyz;
    (&rg_11)->l_err_0 = (float4(isl_20->momentum_err_0) ).xyz;
    float3 _S1200 = float3(0.0f) ;
    (&rg_11)->torque_0 = _S1200;
    (&rg_11)->a_0 = _S1200;
    (&rg_11)->alpha_0 = _S1200;
    return rg_11;
}

void rigid_acceleration_1(const Island_natural_0 thread* isl_21, Rigid_0 thread* rg_12, float3 f_16, float3 t_17)
{
    float4 _S1201 = float4(isl_21->inertia0_0) ;
    float4 _S1202 = float4(isl_21->inertia1_0) ;
    float4 _S1203 = float4(isl_21->inertia2_0) ;
    thread Quat_0 _S1204 = rg_12->rot_0;
    float3 _S1205 = world_mul_0(&_S1204, _S1201, _S1202, _S1203, rg_12->w_1);
    rg_12->a_0 = f_16 / float3((float4(isl_21->com_0) ).w) ;
    float4 _S1206 = float4(isl_21->inv0_0) ;
    float4 _S1207 = float4(isl_21->inv1_0) ;
    float4 _S1208 = float4(isl_21->inv2_0) ;
    float3 _S1209 = t_17 - cross(rg_12->w_1, _S1205);
    thread Quat_0 _S1210 = rg_12->rot_0;
    float3 _S1211 = world_mul_0(&_S1210, _S1206, _S1207, _S1208, _S1209);
    rg_12->alpha_0 = _S1211;
    rg_12->torque_0 = t_17;
    return;
}

Rigid_0 wide_rigid_frame_0(uint tid_13, const Island_natural_0 thread* isl_22, const WideGroup_0 thread* wg_0, KernelContext_0 thread* kernelContext_76)
{
    Rigid_0 _S1212 = rigid_of_1(isl_22);
    thread Rigid_0 rg_13 = _S1212;
    if((((uint4(isl_22->info_0) ).x) & 1U) == 0U)
    {
        thread float3 f_17;
        thread float3 t_18;
        wide_partials_0(tid_13, wg_0->first_0, isl_22->wide_groups_0, 0U, &f_17, &t_18, kernelContext_76);
        rigid_acceleration_1(isl_22, &rg_13, f_17, t_18);
    }
    return rg_13;
}

Rigid_0 wide_rigid_frame_1(uint tid_14, const Island_natural_0 thread* isl_23, const WideGroup_0 thread* wg_1, KernelContext_0 thread* kernelContext_77)
{
    Rigid_0 _S1213 = rigid_of_1(isl_23);
    thread Rigid_0 rg_14 = _S1213;
    if((((uint4(isl_23->info_0) ).x) & 1U) == 0U)
    {
        thread float3 f_18;
        thread float3 t_19;
        wide_partials_0(tid_14, wg_1->first_0, isl_23->wide_groups_0, 0U, &f_18, &t_19, kernelContext_77);
        rigid_acceleration_1(isl_23, &rg_14, f_18, t_19);
    }
    return rg_14;
}

void integrate_rigid_1(const Island_natural_0 thread* isl_24, Rigid_0 thread* rg_15, float dt_12)
{
    float3 _S1214 = float3(dt_12) ;
    comp_add_0(&rg_15->l_0, &rg_15->l_err_0, rg_15->torque_0 * _S1214);
    float3 l_5 = rg_15->l_0 + rg_15->l_err_0;
    comp_add_0(&rg_15->vel_0, &rg_15->vel_err_0, rg_15->a_0 * _S1214);
    float3 vel_2 = rg_15->vel_0 + rg_15->vel_err_0;
    float4 _S1215 = float4(isl_24->inv0_0) ;
    float4 _S1216 = float4(isl_24->inv1_0) ;
    float4 _S1217 = float4(isl_24->inv2_0) ;
    thread Quat_0 _S1218 = rg_15->rot_0;
    float3 _S1219 = world_mul_0(&_S1218, _S1215, _S1216, _S1217, l_5);
    float3 _S1220 = vel_2 * _S1214;
    float3 _S1221 = (float4(isl_24->com_0) ).xyz;
    thread Quat_0 _S1222 = rg_15->rot_0;
    float3 _S1223 = rotate_0(&_S1222, _S1221);
    comp_add_0(&rg_15->pos_0, &rg_15->pos_err_0, _S1220 - turn_difference_0(_S1219, dt_12, _S1223));
    turn_left_0(&rg_15->rot_0, &rg_15->rot_err_0, _S1219, dt_12);
    thread Quat_0 _S1224 = rg_15->rot_0;
    float3 _S1225 = world_mul_0(&_S1224, _S1215, _S1216, _S1217, l_5);
    rg_15->w_1 = _S1225;
    return;
}

void drift_rigid_1(const Island_natural_0 thread* isl_25, Rigid_0 thread* rg_16, float3 tr_8, float3 phi_7, float3 dv_8, float3 dw_6)
{
    float3 wcom_4 = (float4(isl_25->wcom_0) ).xyz;
    Quat_0 rot_3 = rg_16->rot_0;
    float3 _S1226 = tr_8 - turn_difference_0(phi_7, 1.0f, wcom_4);
    thread Quat_0 _S1227 = rg_16->rot_0;
    float3 _S1228 = rotate_0(&_S1227, _S1226);
    comp_add_0(&rg_16->pos_0, &rg_16->pos_err_0, _S1228);
    thread Quat_0 _S1229 = rg_16->rot_0;
    float3 _S1230 = inverse_rotate_1(&_S1229, rg_16->w_1);
    float4 _S1231 = float4(isl_25->inertia0_0) ;
    float4 _S1232 = float4(isl_25->inertia1_0) ;
    float4 _S1233 = float4(isl_25->inertia2_0) ;
    float3 _S1234 = cross(phi_7, rows_mul_0(_S1231, _S1232, _S1233, _S1230)) - rows_mul_0(_S1231, _S1232, _S1233, cross(phi_7, _S1230));
    thread Quat_0 _S1235 = rg_16->rot_0;
    float3 _S1236 = rotate_0(&_S1235, _S1234);
    turn_right_0(&rg_16->rot_0, &rg_16->rot_err_0, phi_7);
    float3 _S1237 = dv_8 + cross(dw_6, (float4(isl_25->com_0) ).xyz - wcom_4);
    thread Quat_0 _S1238 = rot_3;
    float3 _S1239 = rotate_0(&_S1238, _S1237);
    comp_add_0(&rg_16->vel_0, &rg_16->vel_err_0, _S1239);
    thread Quat_0 _S1240 = rot_3;
    float3 _S1241 = rotate_0(&_S1240, dw_6);
    rg_16->w_1 = rg_16->w_1 + _S1241;
    thread Quat_0 _S1242 = rg_16->rot_0;
    float3 _S1243 = world_mul_0(&_S1242, _S1231, _S1232, _S1233, _S1241);
    comp_add_0(&rg_16->l_0, &rg_16->l_err_0, _S1236 + _S1243);
    return;
}

[[kernel]] void wide_rigid(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_3 [[thread_position_in_threadgroup]], Params_0 constant* params_6 [[buffer(0)]], Island_natural_0 device* islands_6 [[buffer(9)]], uint device* index_6 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_6 [[buffer(3)]], packed_float4 device* state_8 [[buffer(6)]], packed_float4 device* scratch_6 [[buffer(8)]], packed_float4 device* contact_state_6 [[buffer(11)]], packed_float4 device* loads_6 [[buffer(5)]], Impactor_natural_0 device* impactors_6 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_6 [[buffer(7)]], BondStatic_natural_0 device* bonds_6 [[buffer(2)]], MaterialTable_0 constant* materials_6 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_78;
    (&kernelContext_78)->params_0 = params_6;
    (&kernelContext_78)->islands_0 = islands_6;
    (&kernelContext_78)->index_0 = index_6;
    (&kernelContext_78)->chunks_0 = chunks_6;
    (&kernelContext_78)->state_0 = state_8;
    (&kernelContext_78)->scratch_0 = scratch_6;
    (&kernelContext_78)->contact_state_0 = contact_state_6;
    (&kernelContext_78)->loads_0 = loads_6;
    (&kernelContext_78)->impactors_0 = impactors_6;
    (&kernelContext_78)->bond_dyn_0 = bond_dyn_6;
    (&kernelContext_78)->bonds_0 = bonds_6;
    (&kernelContext_78)->materials_0 = materials_6;
    threadgroup array<float4, int(32)> g_part_a_6;
    (&kernelContext_78)->g_part_a_0 = &g_part_a_6;
    threadgroup array<float4, int(32)> g_part_b_6;
    (&kernelContext_78)->g_part_b_0 = &g_part_b_6;
    threadgroup array<uint, int(5)> g_seen_6;
    (&kernelContext_78)->g_seen_0 = &g_seen_6;
    threadgroup uint g_wide_run_6;
    (&kernelContext_78)->g_wide_run_0 = &g_wide_run_6;
    threadgroup array<uint, int(4096)> g_label_6;
    (&kernelContext_78)->g_label_0 = &g_label_6;
    threadgroup uint g_changed_6;
    (&kernelContext_78)->g_changed_0 = &g_changed_6;
    threadgroup uint g_halt_6;
    (&kernelContext_78)->g_halt_0 = &g_halt_6;
    threadgroup uint g_run_6;
    (&kernelContext_78)->g_run_0 = &g_run_6;
    threadgroup uint g_sums_6;
    (&kernelContext_78)->g_sums_0 = &g_sums_6;
    threadgroup array<uint, int(512)> g_label_narrow_6;
    (&kernelContext_78)->g_label_narrow_0 = &g_label_narrow_6;
    threadgroup uint g_changed_narrow_6;
    (&kernelContext_78)->g_changed_narrow_0 = &g_changed_narrow_6;
    uint tid_15 = thread_3.x;
    uint _S1244 = group_3.x;
    WideGroup_0 _S1245 = wide_group_0(params_6->wide_chunk_table_0, _S1244, &kernelContext_78);
    thread Island_natural_0 _S1246 = *((&kernelContext_78)->islands_0+_S1245.island_3);
    uint _S1247 = (uint4((&_S1246)->info_0) ).x;
    bool _S1248;
    if((_S1247 & 1U) != 0U)
    {
        _S1248 = true;
    }
    else
    {
        bool _S1249 = wide_enter_1(tid_15, _S1245.island_3, &kernelContext_78);
        _S1248 = !_S1249;
    }
    if(_S1248)
    {
        return;
    }
    if((_S1247 & 16U) != 0U)
    {
        if(_S1244 != (_S1245.first_0))
        {
            return;
        }
        thread WideGroup_0 _S1250 = _S1245;
        Rigid_0 _S1251 = wide_rigid_frame_0(tid_15, &_S1246, &_S1250, &kernelContext_78);
        thread Rigid_0 rs_0 = _S1251;
        if(tid_15 != 0U)
        {
            _S1248 = true;
        }
        else
        {
            _S1248 = (_S1247 & 2U) != 0U;
        }
        if(_S1248)
        {
            return;
        }
        integrate_rigid_1(&_S1246, &rs_0, (&kernelContext_78)->params_0->dt_0);
        thread Quat_0 _S1252 = (&rs_0)->rot_0;
        float4 _S1253 = quat_vec_0(&_S1252);
        ((&kernelContext_78)->islands_0+_S1245.island_3)->rotation_0 = packed_float4(_S1253) ;
        ((&kernelContext_78)->islands_0+_S1245.island_3)->rotation_err_0 = packed_float4((&rs_0)->rot_err_0) ;
        ((&kernelContext_78)->islands_0+_S1245.island_3)->position_0 = packed_float4(float4((&rs_0)->pos_0, 0.0f)) ;
        ((&kernelContext_78)->islands_0+_S1245.island_3)->position_err_0 = packed_float4(float4((&rs_0)->pos_err_0, 0.0f)) ;
        ((&kernelContext_78)->islands_0+_S1245.island_3)->velocity_0 = packed_float4(float4((&rs_0)->vel_0, 0.0f)) ;
        ((&kernelContext_78)->islands_0+_S1245.island_3)->velocity_err_0 = packed_float4(float4((&rs_0)->vel_err_0, 0.0f)) ;
        ((&kernelContext_78)->islands_0+_S1245.island_3)->angular_velocity_0 = packed_float4(float4((&rs_0)->w_1, 0.0f)) ;
        ((&kernelContext_78)->islands_0+_S1245.island_3)->momentum_0 = packed_float4(float4((&rs_0)->l_0, 0.0f)) ;
        ((&kernelContext_78)->islands_0+_S1245.island_3)->momentum_err_0 = packed_float4(float4((&rs_0)->l_err_0, 0.0f)) ;
        return;
    }
    thread float3 tu_6;
    thread float3 pv_6;
    wide_partials_0(tid_15, _S1245.first_0, (&_S1246)->wide_groups_0, 2U, &tu_6, &pv_6, &kernelContext_78);
    thread float3 lu_6;
    thread float3 lv_6;
    wide_partials_0(tid_15, _S1245.first_0, (&_S1246)->wide_groups_0, 4U, &lu_6, &lv_6, &kernelContext_78);
    float4 _S1254 = float4((&_S1246)->wcom_0) ;
    float3 _S1255 = float3(_S1254.w) ;
    float3 tr_9 = tu_6 / _S1255;
    float3 dv_9 = pv_6 / _S1255;
    float4 _S1256 = float4((&_S1246)->winv0_0) ;
    float4 _S1257 = float4((&_S1246)->winv1_0) ;
    float4 _S1258 = float4((&_S1246)->winv2_0) ;
    float3 phi_8 = rows_mul_0(_S1256, _S1257, _S1258, lu_6);
    float3 dw_7 = rows_mul_0(_S1256, _S1257, _S1258, lv_6);
    uint c_33 = _S1245.begin_1 + tid_15;
    if(c_33 < (_S1245.end_0))
    {
        drift_apply_0(c_33, _S1254.xyz, tr_9, phi_8, dv_9, dw_7, &kernelContext_78);
    }
    if(_S1244 != (_S1245.first_0))
    {
        return;
    }
    thread WideGroup_0 _S1259 = _S1245;
    Rigid_0 _S1260 = wide_rigid_frame_0(tid_15, &_S1246, &_S1259, &kernelContext_78);
    thread Rigid_0 rg_17 = _S1260;
    if(tid_15 != 0U)
    {
        return;
    }
    if(!((_S1247 & 2U) != 0U))
    {
        integrate_rigid_1(&_S1246, &rg_17, (&kernelContext_78)->params_0->dt_0);
        drift_rigid_1(&_S1246, &rg_17, tr_9, phi_8, dv_9, dw_7);
    }
    thread Quat_0 _S1261 = (&rg_17)->rot_0;
    float4 _S1262 = quat_vec_0(&_S1261);
    ((&kernelContext_78)->islands_0+_S1245.island_3)->rotation_0 = packed_float4(_S1262) ;
    ((&kernelContext_78)->islands_0+_S1245.island_3)->rotation_err_0 = packed_float4((&rg_17)->rot_err_0) ;
    ((&kernelContext_78)->islands_0+_S1245.island_3)->position_0 = packed_float4(float4((&rg_17)->pos_0, 0.0f)) ;
    ((&kernelContext_78)->islands_0+_S1245.island_3)->position_err_0 = packed_float4(float4((&rg_17)->pos_err_0, 0.0f)) ;
    ((&kernelContext_78)->islands_0+_S1245.island_3)->velocity_0 = packed_float4(float4((&rg_17)->vel_0, 0.0f)) ;
    ((&kernelContext_78)->islands_0+_S1245.island_3)->velocity_err_0 = packed_float4(float4((&rg_17)->vel_err_0, 0.0f)) ;
    ((&kernelContext_78)->islands_0+_S1245.island_3)->angular_velocity_0 = packed_float4(float4((&rg_17)->w_1, 0.0f)) ;
    ((&kernelContext_78)->islands_0+_S1245.island_3)->momentum_0 = packed_float4(float4((&rg_17)->l_0, 0.0f)) ;
    ((&kernelContext_78)->islands_0+_S1245.island_3)->momentum_err_0 = packed_float4(float4((&rg_17)->l_err_0, 0.0f)) ;
    return;
}

bool island_connected_0(uint tid_16, const Island_0 thread* isl_26, KernelContext_0 thread* kernelContext_79)
{
    uint _S1263 = isl_26->range_0.x;
    uint n_12 = isl_26->range_0.y - _S1263;
    bool converged_2;
    if(n_12 > 4096U)
    {
        converged_2 = true;
    }
    else
    {
        converged_2 = (kernelContext_79->params_0->device_connect_0) == 0U;
    }
    if(converged_2)
    {
        return false;
    }
    uint c_34 = tid_16;
    for(;;)
    {
        if(c_34 < n_12)
        {
        }
        else
        {
            break;
        }
        (*kernelContext_79->g_label_0)[c_34] = c_34;
        c_34 = c_34 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    converged_2 = false;
    uint pass_1 = 0U;
    for(;;)
    {
        if(pass_1 <= n_12)
        {
        }
        else
        {
            break;
        }
        if(tid_16 == 0U)
        {
            *kernelContext_79->g_changed_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c_34 = tid_16;
        for(;;)
        {
            if(c_34 < n_12)
            {
            }
            else
            {
                break;
            }
            uint gc_1 = _S1263 + c_34;
            uint _S1264 = kernelContext_79->index_0[gc_1];
            uint l_6 = (*kernelContext_79->g_label_0)[c_34];
            uint e_5 = _S1264;
            for(;;)
            {
                if(e_5 < (kernelContext_79->index_0)[gc_1 + 1U])
                {
                }
                else
                {
                    break;
                }
                uint entry_4 = kernelContext_79->index_0[e_5];
                uint bond_1 = entry_4 >> 1U;
                bool _S1265 = ((float4((kernelContext_79->bonds_0+bond_1)->law_0.stiff1_0) ).w) != 0.0f;
                thread JointState_0 _S1266 = (kernelContext_79->bond_dyn_0+bond_1)->js_0;
                bool _S1267 = connected_0(&_S1266, _S1265);
                if(!_S1267)
                {
                    e_5 = e_5 + 1U;
                    continue;
                }
                uint other_1;
                if((entry_4 & 1U) == 0U)
                {
                    other_1 = (uint4((kernelContext_79->bonds_0+bond_1)->law_0.ids_0) ).z;
                }
                else
                {
                    other_1 = (uint4((kernelContext_79->bonds_0+bond_1)->law_0.ids_0) ).y;
                }
                l_6 = min(l_6, (*kernelContext_79->g_label_0)[other_1 - _S1263]);
                e_5 = e_5 + 1U;
            }
            uint _S1268 = min(l_6, (*kernelContext_79->g_label_0)[l_6]);
            if(_S1268 < (*kernelContext_79->g_label_0)[c_34])
            {
                (*kernelContext_79->g_label_0)[c_34] = _S1268;
                *kernelContext_79->g_changed_0 = 1U;
            }
            c_34 = c_34 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool converged_3 = (*kernelContext_79->g_changed_0) == 0U;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(converged_3)
        {
            converged_2 = converged_3;
            break;
        }
        uint _S1269 = pass_1 + 1U;
        converged_2 = converged_3;
        pass_1 = _S1269;
    }
    if(!converged_2)
    {
        return false;
    }
    if(tid_16 == 0U)
    {
        *kernelContext_79->g_changed_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    c_34 = tid_16;
    for(;;)
    {
        if(c_34 < n_12)
        {
        }
        else
        {
            break;
        }
        if(((*kernelContext_79->g_label_0)[c_34]) != 0U)
        {
            *kernelContext_79->g_changed_0 = 1U;
        }
        c_34 = c_34 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_79->g_changed_0) == 0U;
}

[[kernel]] void wide_end(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], Params_0 constant* params_7 [[buffer(0)]], Island_natural_0 device* islands_7 [[buffer(9)]], uint device* index_7 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_7 [[buffer(3)]], packed_float4 device* state_9 [[buffer(6)]], packed_float4 device* scratch_7 [[buffer(8)]], packed_float4 device* contact_state_7 [[buffer(11)]], packed_float4 device* loads_7 [[buffer(5)]], Impactor_natural_0 device* impactors_7 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_7 [[buffer(7)]], BondStatic_natural_0 device* bonds_7 [[buffer(2)]], MaterialTable_0 constant* materials_7 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_80;
    (&kernelContext_80)->params_0 = params_7;
    (&kernelContext_80)->islands_0 = islands_7;
    (&kernelContext_80)->index_0 = index_7;
    (&kernelContext_80)->chunks_0 = chunks_7;
    (&kernelContext_80)->state_0 = state_9;
    (&kernelContext_80)->scratch_0 = scratch_7;
    (&kernelContext_80)->contact_state_0 = contact_state_7;
    (&kernelContext_80)->loads_0 = loads_7;
    (&kernelContext_80)->impactors_0 = impactors_7;
    (&kernelContext_80)->bond_dyn_0 = bond_dyn_7;
    (&kernelContext_80)->bonds_0 = bonds_7;
    (&kernelContext_80)->materials_0 = materials_7;
    threadgroup array<float4, int(32)> g_part_a_7;
    (&kernelContext_80)->g_part_a_0 = &g_part_a_7;
    threadgroup array<float4, int(32)> g_part_b_7;
    (&kernelContext_80)->g_part_b_0 = &g_part_b_7;
    threadgroup array<uint, int(5)> g_seen_7;
    (&kernelContext_80)->g_seen_0 = &g_seen_7;
    threadgroup uint g_wide_run_7;
    (&kernelContext_80)->g_wide_run_0 = &g_wide_run_7;
    threadgroup array<uint, int(4096)> g_label_7;
    (&kernelContext_80)->g_label_0 = &g_label_7;
    threadgroup uint g_changed_7;
    (&kernelContext_80)->g_changed_0 = &g_changed_7;
    threadgroup uint g_halt_7;
    (&kernelContext_80)->g_halt_0 = &g_halt_7;
    threadgroup uint g_run_7;
    (&kernelContext_80)->g_run_0 = &g_run_7;
    threadgroup uint g_sums_7;
    (&kernelContext_80)->g_sums_0 = &g_sums_7;
    threadgroup array<uint, int(512)> g_label_narrow_7;
    (&kernelContext_80)->g_label_narrow_0 = &g_label_narrow_7;
    threadgroup uint g_changed_narrow_7;
    (&kernelContext_80)->g_changed_narrow_0 = &g_changed_narrow_7;
    uint tid_17 = thread_4.x;
    uint _S1270 = group_4.x;
    WideGroup_0 _S1271 = wide_group_0(params_7->wide_chunk_table_0, _S1270, &kernelContext_80);
    if(_S1270 != (_S1271.first_0))
    {
        return;
    }
    Island_natural_0 device* _S1272 = (&kernelContext_80)->islands_0+_S1271.island_3;
    thread Island_natural_0 _S1273 = *_S1272;
    uint4 _S1274 = uint4((&_S1273)->info_0) ;
    float4 _S1275 = float4((&_S1273)->com_0) ;
    float4 _S1276 = float4((&_S1273)->inertia0_0) ;
    float4 _S1277 = float4((&_S1273)->inertia1_0) ;
    float4 _S1278 = float4((&_S1273)->inertia2_0) ;
    float4 _S1279 = float4((&_S1273)->inv0_0) ;
    float4 _S1280 = float4((&_S1273)->inv1_0) ;
    float4 _S1281 = float4((&_S1273)->inv2_0) ;
    float4 _S1282 = float4((&_S1273)->wcom_0) ;
    float4 _S1283 = float4((&_S1273)->winv0_0) ;
    float4 _S1284 = float4((&_S1273)->winv1_0) ;
    float4 _S1285 = float4((&_S1273)->winv2_0) ;
    float4 _S1286 = float4((&_S1273)->rotation_0) ;
    float4 _S1287 = float4((&_S1273)->position_0) ;
    float4 _S1288 = float4((&_S1273)->position_err_0) ;
    float4 _S1289 = float4((&_S1273)->velocity_0) ;
    float4 _S1290 = float4((&_S1273)->velocity_err_0) ;
    float4 _S1291 = float4((&_S1273)->angular_velocity_0) ;
    uint4 _S1292 = uint4((&_S1273)->probes_0) ;
    float4 _S1293 = float4((&_S1273)->energy_0) ;
    float4 _S1294 = float4((&_S1273)->rotation_err_0) ;
    float4 _S1295 = float4((&_S1273)->momentum_0) ;
    float4 _S1296 = float4((&_S1273)->momentum_err_0) ;
    thread Island_0 isl_27;
    (&isl_27)->range_0 = uint4((&_S1273)->range_0) ;
    (&isl_27)->info_0 = _S1274;
    (&isl_27)->com_0 = _S1275;
    (&isl_27)->inertia0_0 = _S1276;
    (&isl_27)->inertia1_0 = _S1277;
    (&isl_27)->inertia2_0 = _S1278;
    (&isl_27)->inv0_0 = _S1279;
    (&isl_27)->inv1_0 = _S1280;
    (&isl_27)->inv2_0 = _S1281;
    (&isl_27)->wcom_0 = _S1282;
    (&isl_27)->winv0_0 = _S1283;
    (&isl_27)->winv1_0 = _S1284;
    (&isl_27)->winv2_0 = _S1285;
    (&isl_27)->rotation_0 = _S1286;
    (&isl_27)->position_0 = _S1287;
    (&isl_27)->position_err_0 = _S1288;
    (&isl_27)->velocity_0 = _S1289;
    (&isl_27)->velocity_err_0 = _S1290;
    (&isl_27)->angular_velocity_0 = _S1291;
    (&isl_27)->done_0 = (&_S1273)->done_0;
    (&isl_27)->coverage_0 = (&_S1273)->coverage_0;
    (&isl_27)->wide_groups_0 = (&_S1273)->wide_groups_0;
    (&isl_27)->done_pad_0 = (&_S1273)->done_pad_0;
    (&isl_27)->probes_0 = _S1292;
    (&isl_27)->energy_0 = _S1293;
    (&isl_27)->rotation_err_0 = _S1294;
    (&isl_27)->momentum_0 = _S1295;
    (&isl_27)->momentum_err_0 = _S1296;
    _S1273 = *_S1272;
    bool _S1297 = wide_enter_0(tid_17, &_S1273, &kernelContext_80);
    if(!_S1297)
    {
        return;
    }
    thread float3 work_6;
    thread float3 flags_2;
    wide_partials_0(tid_17, _S1271.first_0, (&isl_27)->wide_groups_0, 6U, &work_6, &flags_2, &kernelContext_80);
    bool halt_0 = (((&isl_27)->info_0.z) & 2U) != 0U;
    bool whole_0;
    if(halt_0)
    {
        thread Island_0 _S1298 = isl_27;
        bool _S1299 = island_connected_0(tid_17, &_S1298, &kernelContext_80);
        whole_0 = _S1299;
    }
    else
    {
        whole_0 = false;
    }
    if(tid_17 != 0U)
    {
        return;
    }
    bool halt_1;
    if(whole_0)
    {
        (&isl_27)->info_0.z = ((&isl_27)->info_0.z) & 4294967293U;
        (&isl_27)->coverage_0 = ((&isl_27)->coverage_0) | 2097280U;
        halt_1 = false;
    }
    else
    {
        halt_1 = halt_0;
    }
    thread Island_0 _S1300 = isl_27;
    uint _S1301 = wide_step_2(&_S1300, &kernelContext_80);
    if(((&isl_27)->probes_0.y) > ((&isl_27)->probes_0.x))
    {
        thread Island_0 _S1302 = isl_27;
        Rigid_0 _S1303 = rigid_of_0(&_S1302);
        thread Island_0 _S1304 = isl_27;
        thread Rigid_0 _S1305 = _S1303;
        record_probes_0(&_S1304, &_S1305, _S1301, &kernelContext_80);
    }
    if(halt_1)
    {
        whole_0 = (((&isl_27)->info_0.x) & 4U) != 0U;
    }
    else
    {
        whole_0 = false;
    }
    if(whole_0)
    {
        contact_split_at_0((&isl_27)->info_0.w + 1U, &kernelContext_80);
    }
    float _S1306 = work_6.x;
    thread float _S1307 = (&isl_27)->energy_0.x;
    thread float _S1308 = (&isl_27)->energy_0.y;
    comp_add1_0(&_S1307, &_S1308, _S1306);
    (&isl_27)->energy_0.x = _S1307;
    (&isl_27)->energy_0.y = _S1308 + work_6.y;
    (&isl_27)->done_0 = (&isl_27)->done_0 + 1U;
    (&isl_27)->info_0.y = (&isl_27)->info_0.y - 1U;
    (&isl_27)->info_0.w = (&isl_27)->info_0.w + 1U;
    if(halt_1)
    {
        (&isl_27)->info_0.z = (((&isl_27)->info_0.z) & 4294967293U) | 1U;
    }
    uint _S1309;
    if(halt_1)
    {
        _S1309 = 128U;
    }
    else
    {
        _S1309 = 0U;
    }
    uint _S1310 = 256U | _S1309;
    if(((&isl_27)->probes_0.y) > ((&isl_27)->probes_0.x))
    {
        _S1309 = 262144U;
    }
    else
    {
        _S1309 = 0U;
    }
    uint _S1311 = _S1310 | _S1309;
    if((((&isl_27)->info_0.x) & 4U) != 0U)
    {
        _S1309 = 524288U;
    }
    else
    {
        _S1309 = 0U;
    }
    uint _S1312 = _S1311 | _S1309;
    if((flags_2.x) > 0.0f)
    {
        _S1309 = 32768U;
    }
    else
    {
        _S1309 = 0U;
    }
    uint _S1313 = _S1312 | _S1309;
    if((flags_2.y) > 0.0f)
    {
        _S1309 = 65536U;
    }
    else
    {
        _S1309 = 0U;
    }
    uint _S1314 = _S1313 | _S1309;
    if((flags_2.z) > 0.0f)
    {
        _S1309 = 131072U;
    }
    else
    {
        _S1309 = 0U;
    }
    (&isl_27)->coverage_0 = ((&isl_27)->coverage_0) | (_S1314 | _S1309);
    Island_natural_0 device* _S1315 = (&kernelContext_80)->islands_0+_S1271.island_3;
    _S1315->range_0 = packed_uint4(isl_27.range_0) ;
    _S1315->info_0 = packed_uint4(isl_27.info_0) ;
    _S1315->com_0 = packed_float4(isl_27.com_0) ;
    _S1315->inertia0_0 = packed_float4(isl_27.inertia0_0) ;
    _S1315->inertia1_0 = packed_float4(isl_27.inertia1_0) ;
    _S1315->inertia2_0 = packed_float4(isl_27.inertia2_0) ;
    _S1315->inv0_0 = packed_float4(isl_27.inv0_0) ;
    _S1315->inv1_0 = packed_float4(isl_27.inv1_0) ;
    _S1315->inv2_0 = packed_float4(isl_27.inv2_0) ;
    _S1315->wcom_0 = packed_float4(isl_27.wcom_0) ;
    _S1315->winv0_0 = packed_float4(isl_27.winv0_0) ;
    _S1315->winv1_0 = packed_float4(isl_27.winv1_0) ;
    _S1315->winv2_0 = packed_float4(isl_27.winv2_0) ;
    _S1315->rotation_0 = packed_float4(isl_27.rotation_0) ;
    _S1315->position_0 = packed_float4(isl_27.position_0) ;
    _S1315->position_err_0 = packed_float4(isl_27.position_err_0) ;
    _S1315->velocity_0 = packed_float4(isl_27.velocity_0) ;
    _S1315->velocity_err_0 = packed_float4(isl_27.velocity_err_0) ;
    _S1315->angular_velocity_0 = packed_float4(isl_27.angular_velocity_0) ;
    _S1315->done_0 = isl_27.done_0;
    _S1315->coverage_0 = isl_27.coverage_0;
    _S1315->wide_groups_0 = isl_27.wide_groups_0;
    _S1315->done_pad_0 = isl_27.done_pad_0;
    _S1315->probes_0 = packed_uint4(isl_27.probes_0) ;
    _S1315->energy_0 = packed_float4(isl_27.energy_0) ;
    _S1315->rotation_err_0 = packed_float4(isl_27.rotation_err_0) ;
    _S1315->momentum_0 = packed_float4(isl_27.momentum_0) ;
    _S1315->momentum_err_0 = packed_float4(isl_27.momentum_err_0) ;
    return;
}

uint sv_0(uint c_35, uint slot_3, KernelContext_0 thread* kernelContext_81)
{
    return kernelContext_81->params_0->statics_base_0 + 23U * c_35 + slot_3;
}

void project_load_slot_0(uint tid_18, const Island_natural_0 thread* isl_28, uint slot_4, KernelContext_0 thread* kernelContext_82)
{
    uint _S1316;
    float3 _S1317 = float3(0.0f) ;
    thread float3 net_f_0 = _S1317;
    thread float3 net_m_0 = _S1317;
    uint4 _S1318 = uint4(isl_28->range_0) ;
    uint _S1319 = _S1318.x + tid_18;
    uint c_36 = _S1319;
    for(;;)
    {
        uint _S1320 = _S1318.y;
        _S1316 = _S1320;
        if(c_36 < _S1320)
        {
        }
        else
        {
            break;
        }
        uint _S1321 = sv_0(c_36, slot_4, kernelContext_82);
        float3 fi_1 = (float4(*(kernelContext_82->scratch_0+_S1321)) ).xyz;
        net_f_0 = net_f_0 + fi_1;
        float3 _S1322 = cross((float4((kernelContext_82->chunks_0+c_36)->center_0) ).xyz - (float4(isl_28->com_0) ).xyz, fi_1);
        uint _S1323 = sv_0(c_36, slot_4 + 1U, kernelContext_82);
        net_m_0 = net_m_0 + (_S1322 + (float4(*(kernelContext_82->scratch_0+_S1323)) ).xyz);
        c_36 = c_36 + 256U;
    }
    group_sum3_0(tid_18, &net_f_0, &net_m_0, kernelContext_82);
    float4 _S1324 = float4(isl_28->com_0) ;
    float3 _S1325 = net_f_0 / float3(_S1324.w) ;
    float3 _S1326 = rows_mul_0(float4(isl_28->inv0_0) , float4(isl_28->inv1_0) , float4(isl_28->inv2_0) , net_m_0);
    c_36 = _S1319;
    for(;;)
    {
        if(c_36 < _S1316)
        {
        }
        else
        {
            break;
        }
        ChunkStatic_natural_0 device* _S1327 = kernelContext_82->chunks_0+c_36;
        float4 _S1328 = float4(_S1327->center_0) ;
        float3 r_12 = _S1328.xyz - _S1324.xyz;
        uint _S1329 = sv_0(c_36, slot_4, kernelContext_82);
        *(kernelContext_82->scratch_0+_S1329) = packed_float4(float4((float4(*(kernelContext_82->scratch_0+_S1329)) ).xyz - (_S1325 + cross(_S1326, r_12)) * float3(_S1328.w) , 0.0f)) ;
        uint _S1330 = sv_0(c_36, slot_4 + 1U, kernelContext_82);
        *(kernelContext_82->scratch_0+_S1330) = packed_float4(float4((float4(*(kernelContext_82->scratch_0+_S1330)) ).xyz - rows_mul_0(float4(_S1327->inertia0_1) , float4(_S1327->inertia1_1) , float4(_S1327->inertia2_1) , _S1326), 0.0f)) ;
        c_36 = c_36 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    return;
}

float island_dot_0(uint tid_19, uint c0_0, uint c1_0, uint sa_2, uint sb_2, KernelContext_0 thread* kernelContext_83)
{
    float4 _S1331 = float4(0.0f) ;
    thread float4 acc_0 = _S1331;
    thread float4 unused_6 = _S1331;
    uint c_37 = c0_0 + tid_19;
    for(;;)
    {
        if(c_37 < c1_0)
        {
        }
        else
        {
            break;
        }
        uint _S1332 = sv_0(c_37, sa_2, kernelContext_83);
        float3 _S1333 = (float4(*(kernelContext_83->scratch_0+_S1332)) ).xyz;
        uint _S1334 = sv_0(c_37, sb_2, kernelContext_83);
        float _S1335 = dot(_S1333, (float4(*(kernelContext_83->scratch_0+_S1334)) ).xyz);
        uint _S1336 = sv_0(c_37, sa_2 + 1U, kernelContext_83);
        float3 _S1337 = (float4(*(kernelContext_83->scratch_0+_S1336)) ).xyz;
        uint _S1338 = sv_0(c_37, sb_2 + 1U, kernelContext_83);
        acc_0.x = acc_0.x + (_S1335 + dot(_S1337, (float4(*(kernelContext_83->scratch_0+_S1338)) ).xyz));
        c_37 = c_37 + 256U;
    }
    group_sum2_0(tid_19, &acc_0, &unused_6, kernelContext_83);
    return acc_0.x;
}

void static_kinematics_0(uint _S1339, float3 thread* _S1340, float3 thread* _S1341, KernelContext_0 thread* kernelContext_84)
{
    BondStatic_natural_0 device* _S1342 = kernelContext_84->bonds_0+_S1339;
    uint4 _S1343 = uint4(_S1342->law_0.ids_0) ;
    uint ca_1 = _S1343.y;
    uint cb_1 = _S1343.z;
    uint _S1344 = 4U * cb_1;
    uint _S1345 = 4U * ca_1;
    float3 _S1346 = (float4(*(kernelContext_84->state_0+_S1344)) ).xyz - (float4(*(kernelContext_84->state_0+_S1345)) ).xyz;
    uint _S1347 = sv_0(cb_1, 21U, kernelContext_84);
    float3 _S1348 = (float4(*(kernelContext_84->scratch_0+_S1347)) ).xyz;
    uint _S1349 = sv_0(ca_1, 21U, kernelContext_84);
    float3 du_0 = _S1346 + (_S1348 - (float4(*(kernelContext_84->scratch_0+_S1349)) ).xyz);
    uint _S1350 = _S1344 + 1U;
    uint _S1351 = _S1345 + 1U;
    float3 _S1352 = (float4(*(kernelContext_84->state_0+_S1350)) ).xyz - (float4(*(kernelContext_84->state_0+_S1351)) ).xyz;
    uint _S1353 = sv_0(cb_1, 22U, kernelContext_84);
    float3 _S1354 = (float4(*(kernelContext_84->scratch_0+_S1353)) ).xyz;
    uint _S1355 = sv_0(ca_1, 22U, kernelContext_84);
    float3 dth_0 = _S1352 + (_S1354 - (float4(*(kernelContext_84->scratch_0+_S1355)) ).xyz);
    float3 _S1356 = to_local_0(_S1339, du_0 + (cross((float4(*(kernelContext_84->state_0+_S1350)) ).xyz + (float4(*(kernelContext_84->scratch_0+_S1353)) ).xyz, (float4(_S1342->rb_0) ).xyz) - cross((float4(*(kernelContext_84->state_0+_S1351)) ).xyz + (float4(*(kernelContext_84->scratch_0+_S1355)) ).xyz, (float4(_S1342->ra_0) ).xyz)), kernelContext_84);
    *_S1340 = _S1356;
    float3 _S1357 = to_local_0(_S1339, dth_0, kernelContext_84);
    *_S1341 = _S1357;
    return;
}

JointResponse_0 static_response_0(uint i_12, KernelContext_0 thread* kernelContext_85)
{
    BondStatic_natural_0 device* _S1358 = kernelContext_85->bonds_0+i_12;
    thread float3 d_lin_5;
    thread float3 d_ang_4;
    static_kinematics_0(i_12, &d_lin_5, &d_ang_4, kernelContext_85);
    thread JointBond_natural_0 _S1359 = _S1358->law_0;
    _S1359 = _S1358->law_0;
    thread JointState_0 _S1360 = (kernelContext_85->bond_dyn_0+i_12)->js_0;
    JointResponse_0 _S1361 = joint_evaluate_0(&kernelContext_85->materials_0->m_0[(uint4((&_S1359)->ids_0) ).x], &_S1359, &_S1360, d_lin_5, d_ang_4, 0.0f, false);
    return _S1361;
}

void gather_loads_0(uint c_38, float3 thread* fi_2, float3 thread* mi_3, KernelContext_0 thread* kernelContext_86)
{
    float3 _S1362 = float3(0.0f) ;
    *fi_2 = _S1362;
    *mi_3 = _S1362;
    uint e_6 = kernelContext_86->index_0[c_38];
    for(;;)
    {
        if(e_6 < (kernelContext_86->index_0)[c_38 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_5 = kernelContext_86->index_0[e_6];
        uint bond_2 = entry_5 >> 1U;
        if((entry_5 & 1U) == 0U)
        {
            uint _S1363 = 3U * bond_2;
            *fi_2 = *fi_2 + (float4(*(kernelContext_86->scratch_0+_S1363)) ).xyz;
            *mi_3 = *mi_3 + (float4(*(kernelContext_86->scratch_0+(_S1363 + 1U))) ).xyz;
        }
        else
        {
            uint _S1364 = 3U * bond_2;
            *fi_2 = *fi_2 - (float4(*(kernelContext_86->scratch_0+_S1364)) ).xyz;
            *mi_3 = *mi_3 + (float4(*(kernelContext_86->scratch_0+(_S1364 + 2U))) ).xyz;
        }
        e_6 = e_6 + 1U;
    }
    return;
}

float bond_load_magnitude2_0(uint c_39, KernelContext_0 thread* kernelContext_87)
{
    uint e_7 = kernelContext_87->index_0[c_39];
    float m_5 = 0.0f;
    for(;;)
    {
        if(e_7 < (kernelContext_87->index_0)[c_39 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_6 = kernelContext_87->index_0[e_7];
        uint _S1365 = 3U * (entry_6 >> 1U);
        float3 f_19 = (float4(*(kernelContext_87->scratch_0+_S1365)) ).xyz;
        float3 t_20;
        if((entry_6 & 1U) == 0U)
        {
            t_20 = (float4(*(kernelContext_87->scratch_0+(_S1365 + 1U))) ).xyz;
        }
        else
        {
            t_20 = (float4(*(kernelContext_87->scratch_0+(_S1365 + 2U))) ).xyz;
        }
        float m_6 = m_5 + (dot(f_19, f_19) + dot(t_20, t_20));
        e_7 = e_7 + 1U;
        m_5 = m_6;
    }
    return m_5;
}

uint fixed_mask_0(uint c_40, KernelContext_0 thread* kernelContext_88)
{
    uint support_1 = (uint4((kernelContext_88->chunks_0+c_40)->info_1) ).x;
    uint _S1366;
    if(support_1 == 1U)
    {
        _S1366 = 63U;
    }
    else
    {
        if(support_1 == 2U)
        {
            _S1366 = 7U;
        }
        else
        {
            _S1366 = 0U;
        }
    }
    return _S1366;
}

void hold_0(uint mask_0, float4 thread* lin_0, float4 thread* ang_0, float4 keep_lin_0, float4 keep_ang_0)
{
    uint d_15 = 0U;
    for(;;)
    {
        if(d_15 < 3U)
        {
        }
        else
        {
            break;
        }
        if((mask_0 & (1U << d_15)) != 0U)
        {
            (*lin_0)[d_15] = keep_lin_0[d_15];
        }
        if((mask_0 & (1U << (d_15 + 3U))) != 0U)
        {
            (*ang_0)[d_15] = keep_ang_0[d_15];
        }
        d_15 = d_15 + 1U;
    }
    return;
}

bool is_finite_0(float x_15)
{
    return ((as_type<uint>((x_15))) & 2147483647U) < 2139095040U;
}

uint statics_bond_slot_0(uint i_13, KernelContext_0 thread* kernelContext_89)
{
    return kernelContext_89->params_0->statics_base_0 + 23U * kernelContext_89->params_0->chunk_count_0 + 2U * i_13;
}

void store_inverse_0(uint c_41, const array<float, int(36)> thread* a_15, KernelContext_0 thread* kernelContext_90)
{
    uint j_9;
    float sum_4;
    thread array<float, int(36)> l_7;
    uint k_21 = 0U;
    for(;;)
    {
        if(k_21 < 36U)
        {
        }
        else
        {
            break;
        }
        l_7[k_21] = 0.0f;
        k_21 = k_21 + 1U;
    }
    bool spd_0 = true;
    uint i_14 = 0U;
    for(;;)
    {
        bool _S1367;
        if(i_14 < 6U)
        {
            _S1367 = spd_0;
        }
        else
        {
            _S1367 = false;
        }
        if(_S1367)
        {
        }
        else
        {
            break;
        }
        j_9 = 0U;
        for(;;)
        {
            if(j_9 <= i_14)
            {
            }
            else
            {
                break;
            }
            uint _S1368 = i_14 * 6U;
            uint _S1369 = _S1368 + j_9;
            k_21 = 0U;
            sum_4 = (*a_15)[_S1369];
            for(;;)
            {
                if(k_21 < j_9)
                {
                }
                else
                {
                    break;
                }
                float sum_5 = sum_4 - l_7[_S1368 + k_21] * l_7[j_9 * 6U + k_21];
                k_21 = k_21 + 1U;
                sum_4 = sum_5;
            }
            if(i_14 == j_9)
            {
                if(sum_4 <= 0.0f)
                {
                    spd_0 = false;
                    break;
                }
                l_7[_S1368 + i_14] = sqrt(sum_4);
            }
            else
            {
                l_7[_S1369] = sum_4 / l_7[j_9 * 6U + j_9];
            }
            j_9 = j_9 + 1U;
        }
        i_14 = i_14 + 1U;
    }
    thread array<float, int(36)> inv_0;
    if(!spd_0)
    {
        k_21 = 0U;
        for(;;)
        {
            if(k_21 < 36U)
            {
            }
            else
            {
                break;
            }
            inv_0[k_21] = 0.0f;
            k_21 = k_21 + 1U;
        }
        k_21 = 0U;
        for(;;)
        {
            if(k_21 < 6U)
            {
            }
            else
            {
                break;
            }
            uint _S1370 = k_21 * 6U + k_21;
            float _S1371 = (*a_15)[_S1370];
            if(((*a_15)[_S1370]) > 0.0f)
            {
                sum_4 = 1.0f / _S1371;
            }
            else
            {
                sum_4 = 0.0f;
            }
            inv_0[_S1370] = sum_4;
            k_21 = k_21 + 1U;
        }
    }
    else
    {
        j_9 = 0U;
        for(;;)
        {
            if(j_9 < 6U)
            {
            }
            else
            {
                break;
            }
            thread array<float, int(6)> y_3;
            y_3[int(0)] = 0.0f;
            y_3[int(1)] = 0.0f;
            y_3[int(2)] = 0.0f;
            y_3[int(3)] = 0.0f;
            y_3[int(4)] = 0.0f;
            y_3[int(5)] = 0.0f;
            i_14 = 0U;
            for(;;)
            {
                if(i_14 < 6U)
                {
                }
                else
                {
                    break;
                }
                if(i_14 == j_9)
                {
                    sum_4 = 1.0f;
                }
                else
                {
                    sum_4 = 0.0f;
                }
                k_21 = 0U;
                float s_7 = sum_4;
                for(;;)
                {
                    if(k_21 < i_14)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_8 = s_7 - l_7[i_14 * 6U + k_21] * y_3[k_21];
                    k_21 = k_21 + 1U;
                    s_7 = s_8;
                }
                y_3[i_14] = s_7 / l_7[i_14 * 6U + i_14];
                i_14 = i_14 + 1U;
            }
            thread array<float, int(6)> x_16;
            x_16[int(0)] = 0.0f;
            x_16[int(1)] = 0.0f;
            x_16[int(2)] = 0.0f;
            x_16[int(3)] = 0.0f;
            x_16[int(4)] = 0.0f;
            x_16[int(5)] = 0.0f;
            uint ii_2 = 0U;
            for(;;)
            {
                if(ii_2 < 6U)
                {
                }
                else
                {
                    break;
                }
                uint i_15 = 5U - ii_2;
                k_21 = i_15 + 1U;
                sum_4 = y_3[i_15];
                for(;;)
                {
                    if(k_21 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_9 = sum_4 - l_7[k_21 * 6U + i_15] * x_16[k_21];
                    k_21 = k_21 + 1U;
                    sum_4 = s_9;
                }
                x_16[i_15] = sum_4 / l_7[i_15 * 6U + i_15];
                ii_2 = ii_2 + 1U;
            }
            uint i_16 = 0U;
            for(;;)
            {
                if(i_16 < 6U)
                {
                }
                else
                {
                    break;
                }
                inv_0[i_16 * 6U + j_9] = x_16[i_16];
                i_16 = i_16 + 1U;
            }
            j_9 = j_9 + 1U;
        }
    }
    j_9 = 0U;
    for(;;)
    {
        if(j_9 < 9U)
        {
        }
        else
        {
            break;
        }
        uint _S1372 = sv_0(c_41, 12U + j_9, kernelContext_90);
        uint _S1373 = 4U * j_9;
        *(kernelContext_90->scratch_0+_S1372) = packed_float4(float4(inv_0[_S1373], inv_0[_S1373 + 1U], inv_0[_S1373 + 2U], inv_0[_S1373 + 3U])) ;
        j_9 = j_9 + 1U;
    }
    return;
}

void assemble_block_0(uint c_42, KernelContext_0 thread* kernelContext_91)
{
    uint p_16;
    uint r_13;
    thread array<float, int(36)> a_16;
    uint k_22 = 0U;
    for(;;)
    {
        if(k_22 < 36U)
        {
        }
        else
        {
            break;
        }
        a_16[k_22] = 0.0f;
        k_22 = k_22 + 1U;
    }
    uint e_8 = kernelContext_91->index_0[c_42];
    for(;;)
    {
        if(e_8 < (kernelContext_91->index_0)[c_42 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_7 = kernelContext_91->index_0[e_8];
        uint i_17 = entry_7 >> 1U;
        bool _S1374 = (entry_7 & 1U) != 0U;
        BondStatic_natural_0 device* _S1375 = kernelContext_91->bonds_0+i_17;
        uint _S1376 = statics_bond_slot_0(i_17, kernelContext_91);
        float4 _S1377 = float4(*(kernelContext_91->scratch_0+_S1376)) ;
        float4 _S1378 = float4(*(kernelContext_91->scratch_0+(_S1376 + 1U))) ;
        p_16 = 0U;
        for(;;)
        {
            if(p_16 < 6U)
            {
            }
            else
            {
                break;
            }
            uint _S1379 = p_16 % 3U;
            float3 t_21;
            if(_S1379 == 0U)
            {
                t_21 = (float4(_S1375->t1_0) ).xyz;
            }
            else
            {
                if(_S1379 == 1U)
                {
                    t_21 = (float4(_S1375->t2_0) ).xyz;
                }
                else
                {
                    t_21 = (float4(_S1375->normal_0) ).xyz;
                }
            }
            bool _S1380 = p_16 < 3U;
            float3 row_u_0;
            float3 row_t_0;
            if(_S1380)
            {
                if(_S1374)
                {
                    row_u_0 = t_21;
                }
                else
                {
                    row_u_0 = - t_21;
                }
                if(_S1374)
                {
                    row_t_0 = cross((float4(_S1375->rb_0) ).xyz, t_21);
                }
                else
                {
                    row_t_0 = - cross((float4(_S1375->ra_0) ).xyz, t_21);
                }
            }
            else
            {
                float3 _S1381 = float3(0.0f) ;
                if(_S1374)
                {
                    row_u_0 = t_21;
                }
                else
                {
                    row_u_0 = - t_21;
                }
                float3 _S1382 = row_u_0;
                row_u_0 = _S1381;
                row_t_0 = _S1382;
            }
            float kp_0;
            if(_S1380)
            {
                kp_0 = _S1377[p_16];
            }
            else
            {
                kp_0 = _S1378[p_16 - 3U];
            }
            if(kp_0 == 0.0f)
            {
                p_16 = p_16 + 1U;
                continue;
            }
            array<float, int(6)> _S1383 = { { row_u_0.x, row_u_0.y, row_u_0.z, row_t_0.x, row_t_0.y, row_t_0.z } };
            r_13 = 0U;
            for(;;)
            {
                if(r_13 < 6U)
                {
                }
                else
                {
                    break;
                }
                uint q_14 = 0U;
                for(;;)
                {
                    if(q_14 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    a_16[r_13 * 6U + q_14] = a_16[r_13 * 6U + q_14] + kp_0 * _S1383[r_13] * _S1383[q_14];
                    q_14 = q_14 + 1U;
                }
                r_13 = r_13 + 1U;
            }
            p_16 = p_16 + 1U;
        }
        e_8 = e_8 + 1U;
    }
    uint _S1384 = fixed_mask_0(c_42, kernelContext_91);
    p_16 = 0U;
    for(;;)
    {
        if(p_16 < 6U)
        {
        }
        else
        {
            break;
        }
        if((_S1384 & (1U << p_16)) != 0U)
        {
            r_13 = 0U;
            for(;;)
            {
                if(r_13 < 6U)
                {
                }
                else
                {
                    break;
                }
                a_16[p_16 * 6U + r_13] = 0.0f;
                a_16[r_13 * 6U + p_16] = 0.0f;
                r_13 = r_13 + 1U;
            }
            a_16[p_16 * 6U + p_16] = 1.0f;
        }
        p_16 = p_16 + 1U;
    }
    p_16 = 0U;
    for(;;)
    {
        if(p_16 < 6U)
        {
        }
        else
        {
            break;
        }
        if((a_16[p_16 * 6U + p_16]) == 0.0f)
        {
            a_16[p_16 * 6U + p_16] = 1.0f;
        }
        p_16 = p_16 + 1U;
    }
    thread array<float, int(36)> _S1385 = a_16;
    store_inverse_0(c_42, &_S1385, kernelContext_91);
    return;
}

float block_get_0(uint c_43, uint i_18, uint j_10, KernelContext_0 thread* kernelContext_92)
{
    uint k_23 = i_18 * 6U + j_10;
    uint _S1386 = sv_0(c_43, 12U + k_23 / 4U, kernelContext_92);
    return (*(kernelContext_92->scratch_0+_S1386))[k_23 % 4U];
}

void precondition_0(uint c_44, KernelContext_0 thread* kernelContext_93)
{
    uint _S1387 = sv_0(c_44, 4U, kernelContext_93);
    float4 _S1388 = float4(*(kernelContext_93->scratch_0+_S1387)) ;
    uint _S1389 = sv_0(c_44, 5U, kernelContext_93);
    float4 _S1390 = float4(*(kernelContext_93->scratch_0+_S1389)) ;
    array<float, int(6)> _S1391 = { { _S1388.x, _S1388.y, _S1388.z, _S1390.x, _S1390.y, _S1390.z } };
    thread array<float, int(6)> z_1;
    uint i_19 = 0U;
    for(;;)
    {
        if(i_19 < 6U)
        {
        }
        else
        {
            break;
        }
        uint j_11 = 0U;
        float s_10 = 0.0f;
        for(;;)
        {
            if(j_11 < 6U)
            {
            }
            else
            {
                break;
            }
            float _S1392 = block_get_0(c_44, i_19, j_11, kernelContext_93);
            float s_11 = s_10 + _S1392 * _S1391[j_11];
            j_11 = j_11 + 1U;
            s_10 = s_11;
        }
        z_1[i_19] = s_10;
        i_19 = i_19 + 1U;
    }
    uint _S1393 = sv_0(c_44, 6U, kernelContext_93);
    *(kernelContext_93->scratch_0+_S1393) = packed_float4(float4(z_1[int(0)], z_1[int(1)], z_1[int(2)], 0.0f)) ;
    uint _S1394 = sv_0(c_44, 7U, kernelContext_93);
    *(kernelContext_93->scratch_0+_S1394) = packed_float4(float4(z_1[int(3)], z_1[int(4)], z_1[int(5)], 0.0f)) ;
    return;
}

void project_displacement_slot_0(uint tid_20, const Island_natural_0 thread* isl_29, uint slot_5, KernelContext_0 thread* kernelContext_94)
{
    uint _S1395;
    float3 _S1396 = float3(0.0f) ;
    thread float3 p_17 = _S1396;
    thread float3 l_8 = _S1396;
    uint4 _S1397 = uint4(isl_29->range_0) ;
    uint _S1398 = _S1397.x + tid_20;
    uint c_45 = _S1398;
    for(;;)
    {
        uint _S1399 = _S1397.y;
        _S1395 = _S1399;
        if(c_45 < _S1399)
        {
        }
        else
        {
            break;
        }
        ChunkStatic_natural_0 device* _S1400 = kernelContext_94->chunks_0+c_45;
        uint _S1401 = sv_0(c_45, slot_5, kernelContext_94);
        float3 u_3 = (float4(*(kernelContext_94->scratch_0+_S1401)) ).xyz;
        uint _S1402 = sv_0(c_45, slot_5 + 1U, kernelContext_94);
        float3 th_5 = (float4(*(kernelContext_94->scratch_0+_S1402)) ).xyz;
        float4 _S1403 = float4(_S1400->center_0) ;
        float3 r_14 = _S1403.xyz - (float4(isl_29->com_0) ).xyz;
        float3 _S1404 = float3(_S1403.w) ;
        p_17 = p_17 + u_3 * _S1404;
        l_8 = l_8 + (cross(r_14, u_3) * _S1404 + rows_mul_0(float4(_S1400->inertia0_1) , float4(_S1400->inertia1_1) , float4(_S1400->inertia2_1) , th_5));
        c_45 = c_45 + 256U;
    }
    group_sum3_0(tid_20, &p_17, &l_8, kernelContext_94);
    float4 _S1405 = float4(isl_29->com_0) ;
    float3 _S1406 = p_17 / float3(_S1405.w) ;
    float3 _S1407 = rows_mul_0(float4(isl_29->inv0_0) , float4(isl_29->inv1_0) , float4(isl_29->inv2_0) , l_8);
    c_45 = _S1398;
    for(;;)
    {
        if(c_45 < _S1395)
        {
        }
        else
        {
            break;
        }
        float3 r_15 = (float4((kernelContext_94->chunks_0+c_45)->center_0) ).xyz - _S1405.xyz;
        uint _S1408 = sv_0(c_45, slot_5, kernelContext_94);
        *(kernelContext_94->scratch_0+_S1408) = packed_float4(float4((float4(*(kernelContext_94->scratch_0+_S1408)) ).xyz - _S1406 - cross(_S1407, r_15), 0.0f)) ;
        uint _S1409 = sv_0(c_45, slot_5 + 1U, kernelContext_94);
        *(kernelContext_94->scratch_0+_S1409) = packed_float4(float4((float4(*(kernelContext_94->scratch_0+_S1409)) ).xyz - _S1407, 0.0f)) ;
        c_45 = c_45 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    return;
}

uint statics_result_slot_0(uint island_4, KernelContext_0 thread* kernelContext_95)
{
    return kernelContext_95->params_0->statics_base_0 + 23U * kernelContext_95->params_0->chunk_count_0 + 2U * kernelContext_95->params_0->statics_bonds_0 + island_4;
}

void write_bond_loads_0(uint _S1410, uint _S1411, float3 _S1412, float3 _S1413, float _S1414, KernelContext_0 thread* kernelContext_96)
{
    BondStatic_natural_0 device* _S1415 = kernelContext_96->bonds_0+_S1411;
    float3 _S1416 = to_body_0(_S1411, _S1412, kernelContext_96);
    float3 _S1417 = to_body_0(_S1411, _S1413, kernelContext_96);
    uint _S1418 = 3U * _S1410;
    *(kernelContext_96->scratch_0+_S1418) = packed_float4(float4(_S1416, _S1414)) ;
    *(kernelContext_96->scratch_0+(_S1418 + 1U)) = packed_float4(float4(_S1417 + cross((float4(_S1415->ra_0) ).xyz, _S1416), 0.0f)) ;
    *(kernelContext_96->scratch_0+(_S1418 + 2U)) = packed_float4(float4(- _S1417 + cross((float4(_S1415->rb_0) ).xyz, - _S1416), 0.0f)) ;
    return;
}

void bond_kinematics_0(uint _S1419, float3 _S1420, float3 _S1421, float3 _S1422, float3 _S1423, float3 thread* _S1424, float3 thread* _S1425, KernelContext_0 thread* kernelContext_97)
{
    BondStatic_natural_0 device* _S1426 = kernelContext_97->bonds_0+_S1419;
    float3 _S1427 = to_local_0(_S1419, _S1422 + cross(_S1423, (float4(_S1426->rb_0) ).xyz) - (_S1420 + cross(_S1421, (float4(_S1426->ra_0) ).xyz)), kernelContext_97);
    *_S1424 = _S1427;
    float3 _S1428 = to_local_0(_S1419, _S1423 - _S1421, kernelContext_97);
    *_S1425 = _S1428;
    return;
}

[[kernel]] void island_statics(uint3 group_5 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], Params_0 constant* params_8 [[buffer(0)]], Island_natural_0 device* islands_8 [[buffer(9)]], uint device* index_8 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_8 [[buffer(3)]], packed_float4 device* state_10 [[buffer(6)]], packed_float4 device* scratch_8 [[buffer(8)]], packed_float4 device* contact_state_8 [[buffer(11)]], packed_float4 device* loads_8 [[buffer(5)]], Impactor_natural_0 device* impactors_8 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_8 [[buffer(7)]], BondStatic_natural_0 device* bonds_8 [[buffer(2)]], MaterialTable_0 constant* materials_8 [[buffer(1)]])
{
    uint i_20;
    bool converged_4;
    uint c_46;
    thread KernelContext_0 kernelContext_98;
    (&kernelContext_98)->params_0 = params_8;
    (&kernelContext_98)->islands_0 = islands_8;
    (&kernelContext_98)->index_0 = index_8;
    (&kernelContext_98)->chunks_0 = chunks_8;
    (&kernelContext_98)->state_0 = state_10;
    (&kernelContext_98)->scratch_0 = scratch_8;
    (&kernelContext_98)->contact_state_0 = contact_state_8;
    (&kernelContext_98)->loads_0 = loads_8;
    (&kernelContext_98)->impactors_0 = impactors_8;
    (&kernelContext_98)->bond_dyn_0 = bond_dyn_8;
    (&kernelContext_98)->bonds_0 = bonds_8;
    (&kernelContext_98)->materials_0 = materials_8;
    threadgroup array<float4, int(32)> g_part_a_8;
    (&kernelContext_98)->g_part_a_0 = &g_part_a_8;
    threadgroup array<float4, int(32)> g_part_b_8;
    (&kernelContext_98)->g_part_b_0 = &g_part_b_8;
    threadgroup array<uint, int(5)> g_seen_8;
    (&kernelContext_98)->g_seen_0 = &g_seen_8;
    threadgroup uint g_wide_run_8;
    (&kernelContext_98)->g_wide_run_0 = &g_wide_run_8;
    threadgroup array<uint, int(4096)> g_label_8;
    (&kernelContext_98)->g_label_0 = &g_label_8;
    threadgroup uint g_changed_8;
    (&kernelContext_98)->g_changed_0 = &g_changed_8;
    threadgroup uint g_halt_8;
    (&kernelContext_98)->g_halt_0 = &g_halt_8;
    threadgroup uint g_run_8;
    (&kernelContext_98)->g_run_0 = &g_run_8;
    threadgroup uint g_sums_8;
    (&kernelContext_98)->g_sums_0 = &g_sums_8;
    threadgroup array<uint, int(512)> g_label_narrow_8;
    (&kernelContext_98)->g_label_narrow_0 = &g_label_narrow_8;
    threadgroup uint g_changed_narrow_8;
    (&kernelContext_98)->g_changed_narrow_0 = &g_changed_narrow_8;
    uint tid_21 = thread_5.x;
    uint _S1429 = group_5.x;
    thread Island_natural_0 _S1430 = *(islands_8+_S1429);
    uint4 _S1431 = uint4((&_S1430)->info_0) ;
    uint _S1432 = _S1431.z;
    if((_S1432 & 8U) == 0U)
    {
        return;
    }
    bool free_0 = ((_S1431.x) & 1U) == 0U;
    uint4 _S1433 = uint4((&_S1430)->range_0) ;
    uint c0_1 = _S1433.x;
    uint c1_1 = _S1433.y;
    uint b0_0 = _S1433.z;
    uint _S1434 = _S1433.w;
    uint _S1435 = c0_1 + tid_21;
    uint c_47 = _S1435;
    for(;;)
    {
        if(c_47 < c1_1)
        {
        }
        else
        {
            break;
        }
        uint _S1436 = sv_0(c_47, 21U, &kernelContext_98);
        packed_float4 _S1437 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_98)->scratch_0+_S1436) = _S1437;
        uint _S1438 = sv_0(c_47, 22U, &kernelContext_98);
        *((&kernelContext_98)->scratch_0+_S1438) = _S1437;
        c_47 = c_47 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    if(free_0)
    {
        project_load_slot_0(tid_21, &_S1430, 0U, &kernelContext_98);
    }
    float _S1439 = island_dot_0(tid_21, c0_1, c1_1, 0U, 0U, &kernelContext_98);
    float _S1440 = max(sqrt(_S1439), 1.00000000317107685e-30f);
    float _S1441 = (&kernelContext_98)->params_0->statics_tol_0;
    uint _S1442 = min((&kernelContext_98)->params_0->statics_cg_0, 20U * (c1_1 - c0_1) * 6U + 200U);
    float previous_2 = 1.00000001504746622e+30f;
    float start_0 = 1.00000001504746622e+30f;
    float residual_0 = 0.0f;
    uint newton_0 = 0U;
    uint cg_total_0 = 0U;
    for(;;)
    {
        if(newton_0 < ((&kernelContext_98)->params_0->statics_newton_0))
        {
        }
        else
        {
            converged_4 = false;
            break;
        }
        uint _S1443 = b0_0 + tid_21;
        i_20 = _S1443;
        for(;;)
        {
            if(i_20 < _S1434)
            {
            }
            else
            {
                break;
            }
            JointResponse_0 _S1444 = static_response_0(i_20, &kernelContext_98);
            write_bond_loads_0(i_20, i_20, _S1444.force_lin_1, _S1444.force_ang_1, 0.0f, &kernelContext_98);
            i_20 = i_20 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        float4 _S1445 = float4(0.0f) ;
        thread float4 magnitude_0 = _S1445;
        thread float4 unused_m_0 = _S1445;
        c_47 = _S1435;
        for(;;)
        {
            if(c_47 < c1_1)
            {
            }
            else
            {
                break;
            }
            thread float3 fi_3;
            thread float3 mi_4;
            gather_loads_0(c_47, &fi_3, &mi_4, &kernelContext_98);
            float _S1446 = bond_load_magnitude2_0(c_47, &kernelContext_98);
            magnitude_0.x = magnitude_0.x + _S1446;
            uint _S1447 = sv_0(c_47, 0U, &kernelContext_98);
            thread float4 r_lin_0 = float4((float4(*((&kernelContext_98)->scratch_0+_S1447)) ).xyz + fi_3, 0.0f);
            uint _S1448 = sv_0(c_47, 1U, &kernelContext_98);
            thread float4 r_ang_0 = float4((float4(*((&kernelContext_98)->scratch_0+_S1448)) ).xyz + mi_4, 0.0f);
            uint _S1449 = fixed_mask_0(c_47, &kernelContext_98);
            hold_0(_S1449, &r_lin_0, &r_ang_0, _S1445, _S1445);
            uint _S1450 = sv_0(c_47, 4U, &kernelContext_98);
            *((&kernelContext_98)->scratch_0+_S1450) = packed_float4(r_lin_0) ;
            uint _S1451 = sv_0(c_47, 5U, &kernelContext_98);
            *((&kernelContext_98)->scratch_0+_S1451) = packed_float4(r_ang_0) ;
            c_47 = c_47 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        group_sum2_0(tid_21, &magnitude_0, &unused_m_0, &kernelContext_98);
        if(free_0)
        {
            project_load_slot_0(tid_21, &_S1430, 4U, &kernelContext_98);
        }
        float _S1452 = island_dot_0(tid_21, c0_1, c1_1, 4U, 4U, &kernelContext_98);
        float residual_1 = sqrt(_S1452) / _S1440;
        float _S1453 = max(0.00100000004749745f, 3.83999986297567375e-06f * sqrt(magnitude_0.x) / _S1440);
        if(newton_0 == 0U)
        {
            start_0 = residual_1;
        }
        bool _S1454 = is_finite_0(residual_1);
        bool _S1455;
        if(_S1454)
        {
            if(residual_1 <= _S1441)
            {
                converged_4 = true;
            }
            else
            {
                if(residual_1 <= _S1453)
                {
                    converged_4 = residual_1 > (0.5f * previous_2);
                }
                else
                {
                    converged_4 = false;
                }
                if(converged_4)
                {
                    _S1455 = residual_1 <= start_0;
                }
                else
                {
                    _S1455 = false;
                }
                converged_4 = _S1455;
            }
        }
        else
        {
            converged_4 = false;
        }
        if(converged_4)
        {
            residual_0 = residual_1;
            converged_4 = true;
            break;
        }
        if(!_S1454)
        {
            residual_0 = residual_1;
            converged_4 = false;
            break;
        }
        uint i_21 = _S1443;
        for(;;)
        {
            if(i_21 < _S1434)
            {
            }
            else
            {
                break;
            }
            BondStatic_natural_0 device* _S1456 = (&kernelContext_98)->bonds_0+i_21;
            thread float3 d_lin_6;
            thread float3 d_ang_5;
            static_kinematics_0(i_21, &d_lin_6, &d_ang_5, &kernelContext_98);
            thread JointBond_natural_0 _S1457 = _S1456->law_0;
            thread JointState_0 _S1458 = ((&kernelContext_98)->bond_dyn_0+i_21)->js_0;
            thread float3 f_lin_2;
            thread float3 f_ang_2;
            secant_factors_0(&_S1457, &_S1458, d_lin_6, &f_lin_2, &f_ang_2);
            float4 _S1459 = float4((&_S1457)->stiff0_0) ;
            uint _S1460 = statics_bond_slot_0(i_21, &kernelContext_98);
            float _S1461 = _S1459.y;
            *((&kernelContext_98)->scratch_0+_S1460) = packed_float4(float4(_S1461 * f_lin_2.x, _S1461 * f_lin_2.y, _S1459.x * f_lin_2.z, 0.0f)) ;
            *((&kernelContext_98)->scratch_0+(_S1460 + 1U)) = packed_float4(float4(_S1459.z * f_ang_2.x, _S1459.w * f_ang_2.y, (float4((&_S1457)->stiff1_0) ).x * f_ang_2.z, 0.0f)) ;
            i_21 = i_21 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        uint c_48 = _S1435;
        for(;;)
        {
            if(c_48 < c1_1)
            {
            }
            else
            {
                break;
            }
            assemble_block_0(c_48, &kernelContext_98);
            uint _S1462 = sv_0(c_48, 2U, &kernelContext_98);
            packed_float4 _S1463 = packed_float4(_S1445) ;
            *((&kernelContext_98)->scratch_0+_S1462) = _S1463;
            uint _S1464 = sv_0(c_48, 3U, &kernelContext_98);
            *((&kernelContext_98)->scratch_0+_S1464) = _S1463;
            c_48 = c_48 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        uint c_49 = _S1435;
        for(;;)
        {
            if(c_49 < c1_1)
            {
            }
            else
            {
                break;
            }
            precondition_0(c_49, &kernelContext_98);
            c_49 = c_49 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        if(free_0)
        {
            project_displacement_slot_0(tid_21, &_S1430, 6U, &kernelContext_98);
        }
        uint c_50 = _S1435;
        for(;;)
        {
            if(c_50 < c1_1)
            {
            }
            else
            {
                break;
            }
            uint _S1465 = sv_0(c_50, 8U, &kernelContext_98);
            packed_float4 device* _S1466 = (&kernelContext_98)->scratch_0+_S1465;
            uint _S1467 = sv_0(c_50, 6U, &kernelContext_98);
            *_S1466 = packed_float4(float4(*((&kernelContext_98)->scratch_0+_S1467)) ) ;
            uint _S1468 = sv_0(c_50, 9U, &kernelContext_98);
            packed_float4 device* _S1469 = (&kernelContext_98)->scratch_0+_S1468;
            uint _S1470 = sv_0(c_50, 7U, &kernelContext_98);
            *_S1469 = packed_float4(float4(*((&kernelContext_98)->scratch_0+_S1470)) ) ;
            c_50 = c_50 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        float _S1471 = island_dot_0(tid_21, c0_1, c1_1, 4U, 6U, &kernelContext_98);
        float _S1472 = island_dot_0(tid_21, c0_1, c1_1, 4U, 4U, &kernelContext_98);
        float _S1473 = sqrt(_S1472);
        float rz_0 = _S1471;
        uint k_24 = 0U;
        uint cg_total_1 = cg_total_0;
        for(;;)
        {
            if(k_24 < _S1442)
            {
                _S1455 = _S1473 > 0.0f;
            }
            else
            {
                _S1455 = false;
            }
            if(_S1455)
            {
            }
            else
            {
                cg_total_0 = cg_total_1;
                break;
            }
            uint i_22 = _S1443;
            for(;;)
            {
                if(i_22 < _S1434)
                {
                }
                else
                {
                    break;
                }
                uint4 _S1474 = uint4(((&kernelContext_98)->bonds_0+i_22)->law_0.ids_0) ;
                uint ca_2 = _S1474.y;
                uint cb_2 = _S1474.z;
                uint _S1475 = sv_0(ca_2, 8U, &kernelContext_98);
                float3 _S1476 = (float4(*((&kernelContext_98)->scratch_0+_S1475)) ).xyz;
                uint _S1477 = sv_0(ca_2, 9U, &kernelContext_98);
                float3 _S1478 = (float4(*((&kernelContext_98)->scratch_0+_S1477)) ).xyz;
                uint _S1479 = sv_0(cb_2, 8U, &kernelContext_98);
                float3 _S1480 = (float4(*((&kernelContext_98)->scratch_0+_S1479)) ).xyz;
                uint _S1481 = sv_0(cb_2, 9U, &kernelContext_98);
                thread float3 d_lin_7;
                thread float3 d_ang_6;
                bond_kinematics_0(i_22, _S1476, _S1478, _S1480, (float4(*((&kernelContext_98)->scratch_0+_S1481)) ).xyz, &d_lin_7, &d_ang_6, &kernelContext_98);
                uint _S1482 = statics_bond_slot_0(i_22, &kernelContext_98);
                write_bond_loads_0(i_22, i_22, d_lin_7 * (float4(*((&kernelContext_98)->scratch_0+_S1482)) ).xyz, d_ang_6 * (float4(*((&kernelContext_98)->scratch_0+(_S1482 + 1U))) ).xyz, 0.0f, &kernelContext_98);
                i_22 = i_22 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            c_46 = _S1435;
            for(;;)
            {
                if(c_46 < c1_1)
                {
                }
                else
                {
                    break;
                }
                thread float3 fi_4;
                thread float3 mi_5;
                gather_loads_0(c_46, &fi_4, &mi_5, &kernelContext_98);
                thread float4 ap_lin_0 = float4(- fi_4, 0.0f);
                thread float4 ap_ang_0 = float4(- mi_5, 0.0f);
                uint _S1483 = fixed_mask_0(c_46, &kernelContext_98);
                uint _S1484 = sv_0(c_46, 8U, &kernelContext_98);
                float4 _S1485 = float4(*((&kernelContext_98)->scratch_0+_S1484)) ;
                uint _S1486 = sv_0(c_46, 9U, &kernelContext_98);
                hold_0(_S1483, &ap_lin_0, &ap_ang_0, _S1485, float4(*((&kernelContext_98)->scratch_0+_S1486)) );
                uint _S1487 = sv_0(c_46, 10U, &kernelContext_98);
                *((&kernelContext_98)->scratch_0+_S1487) = packed_float4(ap_lin_0) ;
                uint _S1488 = sv_0(c_46, 11U, &kernelContext_98);
                *((&kernelContext_98)->scratch_0+_S1488) = packed_float4(ap_ang_0) ;
                c_46 = c_46 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            float _S1489 = island_dot_0(tid_21, c0_1, c1_1, 8U, 10U, &kernelContext_98);
            uint _S1490 = cg_total_1 + 1U;
            if(_S1489 <= 0.0f)
            {
                cg_total_0 = _S1490;
                break;
            }
            float _S1491 = rz_0 / _S1489;
            uint c_51 = _S1435;
            for(;;)
            {
                if(c_51 < c1_1)
                {
                }
                else
                {
                    break;
                }
                uint _S1492 = sv_0(c_51, 2U, &kernelContext_98);
                packed_float4 device* _S1493 = (&kernelContext_98)->scratch_0+_S1492;
                float3 _S1494 = (float4(*((&kernelContext_98)->scratch_0+_S1492)) ).xyz;
                uint _S1495 = sv_0(c_51, 8U, &kernelContext_98);
                float3 _S1496 = float3(_S1491) ;
                *_S1493 = packed_float4(float4(_S1494 + _S1496 * (float4(*((&kernelContext_98)->scratch_0+_S1495)) ).xyz, 0.0f)) ;
                uint _S1497 = sv_0(c_51, 3U, &kernelContext_98);
                packed_float4 device* _S1498 = (&kernelContext_98)->scratch_0+_S1497;
                float3 _S1499 = (float4(*((&kernelContext_98)->scratch_0+_S1497)) ).xyz;
                uint _S1500 = sv_0(c_51, 9U, &kernelContext_98);
                *_S1498 = packed_float4(float4(_S1499 + _S1496 * (float4(*((&kernelContext_98)->scratch_0+_S1500)) ).xyz, 0.0f)) ;
                uint _S1501 = sv_0(c_51, 4U, &kernelContext_98);
                packed_float4 device* _S1502 = (&kernelContext_98)->scratch_0+_S1501;
                float3 _S1503 = (float4(*((&kernelContext_98)->scratch_0+_S1501)) ).xyz;
                uint _S1504 = sv_0(c_51, 10U, &kernelContext_98);
                *_S1502 = packed_float4(float4(_S1503 - _S1496 * (float4(*((&kernelContext_98)->scratch_0+_S1504)) ).xyz, 0.0f)) ;
                uint _S1505 = sv_0(c_51, 5U, &kernelContext_98);
                packed_float4 device* _S1506 = (&kernelContext_98)->scratch_0+_S1505;
                float3 _S1507 = (float4(*((&kernelContext_98)->scratch_0+_S1505)) ).xyz;
                uint _S1508 = sv_0(c_51, 11U, &kernelContext_98);
                *_S1506 = packed_float4(float4(_S1507 - _S1496 * (float4(*((&kernelContext_98)->scratch_0+_S1508)) ).xyz, 0.0f)) ;
                c_51 = c_51 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            float _S1509 = island_dot_0(tid_21, c0_1, c1_1, 4U, 4U, &kernelContext_98);
            if((sqrt(_S1509)) <= (0.00009999999747379f * _S1473))
            {
                cg_total_0 = _S1490;
                break;
            }
            uint c_52 = _S1435;
            for(;;)
            {
                if(c_52 < c1_1)
                {
                }
                else
                {
                    break;
                }
                precondition_0(c_52, &kernelContext_98);
                c_52 = c_52 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            if(free_0)
            {
                project_displacement_slot_0(tid_21, &_S1430, 6U, &kernelContext_98);
            }
            float _S1510 = island_dot_0(tid_21, c0_1, c1_1, 4U, 6U, &kernelContext_98);
            float _S1511 = _S1510 / rz_0;
            uint c_53 = _S1435;
            for(;;)
            {
                if(c_53 < c1_1)
                {
                }
                else
                {
                    break;
                }
                uint _S1512 = sv_0(c_53, 8U, &kernelContext_98);
                packed_float4 device* _S1513 = (&kernelContext_98)->scratch_0+_S1512;
                uint _S1514 = sv_0(c_53, 6U, &kernelContext_98);
                float3 _S1515 = float3(_S1511) ;
                *_S1513 = packed_float4(float4((float4(*((&kernelContext_98)->scratch_0+_S1514)) ).xyz + _S1515 * (float4(*((&kernelContext_98)->scratch_0+_S1512)) ).xyz, 0.0f)) ;
                uint _S1516 = sv_0(c_53, 9U, &kernelContext_98);
                packed_float4 device* _S1517 = (&kernelContext_98)->scratch_0+_S1516;
                uint _S1518 = sv_0(c_53, 7U, &kernelContext_98);
                *_S1517 = packed_float4(float4((float4(*((&kernelContext_98)->scratch_0+_S1518)) ).xyz + _S1515 * (float4(*((&kernelContext_98)->scratch_0+_S1516)) ).xyz, 0.0f)) ;
                c_53 = c_53 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            uint _S1519 = k_24 + 1U;
            rz_0 = _S1510;
            k_24 = _S1519;
            cg_total_1 = _S1490;
        }
        c_46 = _S1435;
        for(;;)
        {
            if(c_46 < c1_1)
            {
            }
            else
            {
                break;
            }
            uint _S1520 = 4U * c_46;
            thread float3 u_4 = (float4(*((&kernelContext_98)->state_0+_S1520)) ).xyz;
            uint _S1521 = sv_0(c_46, 21U, &kernelContext_98);
            thread float3 u_lo_0 = (float4(*((&kernelContext_98)->scratch_0+_S1521)) ).xyz;
            uint _S1522 = _S1520 + 1U;
            thread float3 th_6 = (float4(*((&kernelContext_98)->state_0+_S1522)) ).xyz;
            uint _S1523 = sv_0(c_46, 22U, &kernelContext_98);
            thread float3 th_lo_0 = (float4(*((&kernelContext_98)->scratch_0+_S1523)) ).xyz;
            uint _S1524 = sv_0(c_46, 2U, &kernelContext_98);
            comp_add_0(&u_4, &u_lo_0, (float4(*((&kernelContext_98)->scratch_0+_S1524)) ).xyz);
            uint _S1525 = sv_0(c_46, 3U, &kernelContext_98);
            comp_add_0(&th_6, &th_lo_0, (float4(*((&kernelContext_98)->scratch_0+_S1525)) ).xyz);
            *((&kernelContext_98)->state_0+_S1520) = packed_float4(float4(u_4, (float4(*((&kernelContext_98)->state_0+_S1520)) ).w)) ;
            *((&kernelContext_98)->state_0+_S1522) = packed_float4(float4(th_6, (float4(*((&kernelContext_98)->state_0+_S1522)) ).w)) ;
            *((&kernelContext_98)->scratch_0+_S1521) = packed_float4(float4(u_lo_0, 0.0f)) ;
            *((&kernelContext_98)->scratch_0+_S1523) = packed_float4(float4(th_lo_0, 0.0f)) ;
            c_46 = c_46 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        uint _S1526 = newton_0 + 1U;
        previous_2 = residual_1;
        residual_0 = residual_1;
        newton_0 = _S1526;
    }
    i_20 = b0_0 + tid_21;
    for(;;)
    {
        if(i_20 < _S1434)
        {
        }
        else
        {
            break;
        }
        JointResponse_0 _S1527 = static_response_0(i_20, &kernelContext_98);
        BondDyn_natural_0 device* _S1528 = (&kernelContext_98)->bond_dyn_0+i_20;
        float4 _S1529 = float4((*_S1528).force_lin_0) ;
        float4 _S1530 = float4((*_S1528).force_ang_0) ;
        float4 _S1531 = float4((*_S1528).sums_0) ;
        float4 _S1532 = float4((*_S1528).comps_0) ;
        uint4 _S1533 = uint4((*_S1528).events_0) ;
        thread BondDyn_0 bd_1;
        (&bd_1)->js_0 = (*_S1528).js_0;
        (&bd_1)->force_lin_0 = _S1529;
        (&bd_1)->force_ang_0 = _S1530;
        (&bd_1)->sums_0 = _S1531;
        (&bd_1)->comps_0 = _S1532;
        (&bd_1)->events_0 = _S1533;
        (&bd_1)->force_lin_0 = float4(_S1527.force_lin_1, _S1527.stored_2);
        (&bd_1)->force_ang_0 = float4(_S1527.force_ang_1, (&bd_1)->force_ang_0.w);
        BondDyn_natural_0 device* _S1534 = (&kernelContext_98)->bond_dyn_0+i_20;
        _S1534->js_0 = bd_1.js_0;
        _S1534->force_lin_0 = packed_float4(bd_1.force_lin_0) ;
        _S1534->force_ang_0 = packed_float4(bd_1.force_ang_0) ;
        _S1534->sums_0 = packed_float4(bd_1.sums_0) ;
        _S1534->comps_0 = packed_float4(bd_1.comps_0) ;
        _S1534->events_0 = packed_uint4(bd_1.events_0) ;
        write_bond_loads_0(i_20, i_20, _S1527.force_lin_1, _S1527.force_ang_1, max(_S1527.measures_0.tension_0, _S1527.measures_0.compression_0), &kernelContext_98);
        i_20 = i_20 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    c_47 = _S1435;
    for(;;)
    {
        if(c_47 < c1_1)
        {
        }
        else
        {
            break;
        }
        thread float3 fi_5;
        thread float3 mi_6;
        gather_loads_0(c_47, &fi_5, &mi_6, &kernelContext_98);
        uint _S1535 = fixed_mask_0(c_47, &kernelContext_98);
        float3 reaction_1;
        if(_S1535 != 0U)
        {
            uint _S1536 = sv_0(c_47, 0U, &kernelContext_98);
            reaction_1 = - ((float4(*((&kernelContext_98)->scratch_0+_S1536)) ).xyz + fi_5);
        }
        else
        {
            uint _S1537 = 4U * c_47;
            reaction_1 = float3((float4(*((&kernelContext_98)->state_0+(_S1537 + 1U))) ).w, (float4(*((&kernelContext_98)->state_0+(_S1537 + 2U))) ).w, (float4(*((&kernelContext_98)->state_0+(_S1537 + 3U))) ).w);
        }
        uint _S1538 = 4U * c_47;
        uint _S1539 = _S1538 + 1U;
        *((&kernelContext_98)->state_0+_S1539) = packed_float4(float4((float4(*((&kernelContext_98)->state_0+_S1539)) ).xyz, reaction_1.x)) ;
        *((&kernelContext_98)->state_0+(_S1538 + 2U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, reaction_1.y)) ;
        *((&kernelContext_98)->state_0+(_S1538 + 3U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, reaction_1.z)) ;
        c_47 = c_47 + 256U;
    }
    if(tid_21 == 0U)
    {
        uint _S1540 = statics_result_slot_0(_S1429, &kernelContext_98);
        packed_float4 device* _S1541 = (&kernelContext_98)->scratch_0+_S1540;
        float _S1542 = (as_type<float>((newton_0)));
        float _S1543 = (as_type<float>((cg_total_0)));
        if(converged_4)
        {
            previous_2 = 1.0f;
        }
        else
        {
            previous_2 = 0.0f;
        }
        *_S1541 = packed_float4(float4(residual_0, _S1542, _S1543, previous_2)) ;
        ((&kernelContext_98)->islands_0+_S1429)->info_0[int(2)] = _S1432 & 4294967287U;
    }
    return;
}

[[kernel]] void settled_fatigue(uint3 group_6 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], Params_0 constant* params_9 [[buffer(0)]], Island_natural_0 device* islands_9 [[buffer(9)]], uint device* index_9 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_9 [[buffer(3)]], packed_float4 device* state_11 [[buffer(6)]], packed_float4 device* scratch_9 [[buffer(8)]], packed_float4 device* contact_state_9 [[buffer(11)]], packed_float4 device* loads_9 [[buffer(5)]], Impactor_natural_0 device* impactors_9 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_9 [[buffer(7)]], BondStatic_natural_0 device* bonds_9 [[buffer(2)]], MaterialTable_0 constant* materials_9 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_9;
    (&kernelContext_99)->islands_0 = islands_9;
    (&kernelContext_99)->index_0 = index_9;
    (&kernelContext_99)->chunks_0 = chunks_9;
    (&kernelContext_99)->state_0 = state_11;
    (&kernelContext_99)->scratch_0 = scratch_9;
    (&kernelContext_99)->contact_state_0 = contact_state_9;
    (&kernelContext_99)->loads_0 = loads_9;
    (&kernelContext_99)->impactors_0 = impactors_9;
    (&kernelContext_99)->bond_dyn_0 = bond_dyn_9;
    (&kernelContext_99)->bonds_0 = bonds_9;
    (&kernelContext_99)->materials_0 = materials_9;
    threadgroup array<float4, int(32)> g_part_a_9;
    (&kernelContext_99)->g_part_a_0 = &g_part_a_9;
    threadgroup array<float4, int(32)> g_part_b_9;
    (&kernelContext_99)->g_part_b_0 = &g_part_b_9;
    threadgroup array<uint, int(5)> g_seen_9;
    (&kernelContext_99)->g_seen_0 = &g_seen_9;
    threadgroup uint g_wide_run_9;
    (&kernelContext_99)->g_wide_run_0 = &g_wide_run_9;
    threadgroup array<uint, int(4096)> g_label_9;
    (&kernelContext_99)->g_label_0 = &g_label_9;
    threadgroup uint g_changed_9;
    (&kernelContext_99)->g_changed_0 = &g_changed_9;
    threadgroup uint g_halt_9;
    (&kernelContext_99)->g_halt_0 = &g_halt_9;
    threadgroup uint g_run_9;
    (&kernelContext_99)->g_run_0 = &g_run_9;
    threadgroup uint g_sums_9;
    (&kernelContext_99)->g_sums_0 = &g_sums_9;
    threadgroup array<uint, int(512)> g_label_narrow_9;
    (&kernelContext_99)->g_label_narrow_0 = &g_label_narrow_9;
    threadgroup uint g_changed_narrow_9;
    (&kernelContext_99)->g_changed_narrow_0 = &g_changed_narrow_9;
    uint tid_22 = thread_6.x;
    uint _S1544 = group_6.x;
    Island_natural_0 device* _S1545 = islands_9+_S1544;
    Island_natural_0 isl_30 = *_S1545;
    uint4 _S1546 = uint4((*_S1545).info_0) ;
    bool _S1547;
    if(((_S1546.x) & 16U) == 0U)
    {
        _S1547 = true;
    }
    else
    {
        uint4 _S1548 = uint4(isl_30.range_0) ;
        _S1547 = (_S1548.w) == (_S1548.z);
    }
    if(_S1547)
    {
        return;
    }
    seen_clear_0(tid_22, &kernelContext_99);
    if(tid_22 == 0U)
    {
        *(&kernelContext_99)->g_halt_0 = 0U;
        *(&kernelContext_99)->g_run_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S1549 = _S1546.w;
    uint4 _S1550 = uint4(isl_30.range_0) ;
    uint i_23 = _S1550.z + tid_22;
    for(;;)
    {
        if(i_23 < (_S1550.w))
        {
        }
        else
        {
            break;
        }
        BondStatic_natural_0 device* _S1551 = (&kernelContext_99)->bonds_0+i_23;
        BondDyn_natural_0 device* _S1552 = (&kernelContext_99)->bond_dyn_0+i_23;
        float4 _S1553 = float4((*_S1552).force_lin_0) ;
        float4 _S1554 = float4((*_S1552).force_ang_0) ;
        float4 _S1555 = float4((*_S1552).sums_0) ;
        float4 _S1556 = float4((*_S1552).comps_0) ;
        uint4 _S1557 = uint4((*_S1552).events_0) ;
        thread BondDyn_0 bd_2;
        (&bd_2)->js_0 = (*_S1552).js_0;
        (&bd_2)->force_lin_0 = _S1553;
        (&bd_2)->force_ang_0 = _S1554;
        (&bd_2)->sums_0 = _S1555;
        (&bd_2)->comps_0 = _S1556;
        (&bd_2)->events_0 = _S1557;
        thread JointBond_natural_0 _S1558 = _S1551->law_0;
        uint4 _S1559 = uint4((&_S1558)->ids_0) ;
        uint _S1560 = 4U * _S1559.y;
        uint _S1561 = 4U * _S1559.z;
        thread float3 d_lin_8;
        thread float3 d_ang_7;
        bond_kinematics_0(i_23, (float4(*((&kernelContext_99)->state_0+_S1560)) ).xyz, (float4(*((&kernelContext_99)->state_0+(_S1560 + 1U))) ).xyz, (float4(*((&kernelContext_99)->state_0+_S1561)) ).xyz, (float4(*((&kernelContext_99)->state_0+(_S1561 + 1U))) ).xyz, &d_lin_8, &d_ang_7, &kernelContext_99);
        JointState_0 previous_3 = (&bd_2)->js_0;
        float _S1562 = (&kernelContext_99)->params_0->dt_0;
        bool _S1563 = ((&kernelContext_99)->params_0->fracture_0) != 0U;
        _S1558 = _S1551->law_0;
        thread JointState_0 _S1564 = (&bd_2)->js_0;
        JointResponse_0 _S1565 = joint_evaluate_0(&(&kernelContext_99)->materials_0->m_0[_S1559.x], &_S1558, &_S1564, d_lin_8, d_ang_7, _S1562, _S1563);
        if((_S1565.state_1.damage_0) > (previous_3.damage_0 + 9.99999971718068537e-10f))
        {
            _S1547 = true;
        }
        else
        {
            _S1547 = (_S1565.state_1.crush_1) > (previous_3.crush_1 + 9.99999971718068537e-10f);
        }
        uint flags_3;
        if(_S1547)
        {
            flags_3 = 16U;
        }
        else
        {
            flags_3 = 0U;
        }
        thread float _S1566 = (&bd_2)->sums_0.x;
        thread float _S1567 = (&bd_2)->comps_0.x;
        comp_add1_0(&_S1566, &_S1567, _S1565.dissipated_2);
        (&bd_2)->comps_0.x = _S1567;
        (&bd_2)->sums_0.x = _S1566;
        thread float _S1568 = (&bd_2)->sums_0.y;
        thread float _S1569 = (&bd_2)->comps_0.y;
        comp_add1_0(&_S1568, &_S1569, _S1565.overshoot_0);
        (&bd_2)->comps_0.y = _S1569;
        (&bd_2)->sums_0.y = _S1568;
        (&bd_2)->force_lin_0 = float4(_S1565.force_lin_1, _S1565.stored_2);
        (&bd_2)->force_ang_0 = float4(_S1565.force_ang_1, max((&bd_2)->force_ang_0.w, _S1565.state_1.utilization_0));
        thread JointState_0 _S1570 = previous_3;
        bool _S1571 = is_damaged_0(&_S1570);
        bool _S1572;
        if(!_S1571)
        {
            thread JointState_0 _S1573 = _S1565.state_1;
            bool _S1574 = is_damaged_0(&_S1573);
            _S1572 = _S1574;
        }
        else
        {
            _S1572 = false;
        }
        bool _S1575;
        if(_S1572)
        {
            _S1575 = ((&bd_2)->events_0.x) == 0U;
        }
        else
        {
            _S1575 = false;
        }
        if(_S1575)
        {
            (&bd_2)->events_0.x = _S1549;
            (&bd_2)->events_0.w = _S1565.state_1.mode_0;
            (*(&kernelContext_99)->g_seen_0)[3U] = 1U;
        }
        bool _S1576;
        if(((&bd_2)->events_0.y) == 0U)
        {
            _S1576 = (_S1565.state_1.life_0) != (previous_3.life_0);
        }
        else
        {
            _S1576 = false;
        }
        bool _S1577;
        if(_S1576)
        {
            _S1577 = (_S1565.fatigue_before_0) > 0.99000000953674316f;
        }
        else
        {
            _S1577 = false;
        }
        bool _S1578;
        if(_S1577)
        {
            float _S1579 = fatigue_factor_0(&(&kernelContext_99)->materials_0->m_0[_S1559.x], _S1565.state_1.life_0);
            _S1578 = _S1579 <= 0.99000000953674316f;
        }
        else
        {
            _S1578 = false;
        }
        if(_S1578)
        {
            (&bd_2)->events_0.y = _S1549;
            (*(&kernelContext_99)->g_seen_0)[4U] = 1U;
        }
        uint flags_4;
        if(_S1565.disconnected_0)
        {
            (&bd_2)->events_0.z = _S1549;
            flags_4 = flags_3 | 32U;
        }
        else
        {
            flags_4 = flags_3;
        }
        (&bd_2)->js_0 = _S1565.state_1;
        BondDyn_natural_0 device* _S1580 = (&kernelContext_99)->bond_dyn_0+i_23;
        _S1580->js_0 = bd_2.js_0;
        _S1580->force_lin_0 = packed_float4(bd_2.force_lin_0) ;
        _S1580->force_ang_0 = packed_float4(bd_2.force_ang_0) ;
        _S1580->sums_0 = packed_float4(bd_2.sums_0) ;
        _S1580->comps_0 = packed_float4(bd_2.comps_0) ;
        _S1580->events_0 = packed_uint4(bd_2.events_0) ;
        write_bond_loads_0(i_23, i_23, _S1565.force_lin_1, _S1565.force_ang_1, max(_S1565.measures_0.tension_0, _S1565.measures_0.compression_0), &kernelContext_99);
        if((flags_4 & 16U) != 0U)
        {
            *(&kernelContext_99)->g_halt_0 = 1U;
        }
        if((flags_4 & 32U) != 0U)
        {
            *(&kernelContext_99)->g_run_0 = 1U;
        }
        i_23 = i_23 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid_22 != 0U)
    {
        return;
    }
    if(((*(&kernelContext_99)->g_halt_0) | (*(&kernelContext_99)->g_run_0)) != 0U)
    {
        uint _S1581 = _S1546.z;
        if((*(&kernelContext_99)->g_halt_0) != 0U)
        {
            i_23 = 16U;
        }
        else
        {
            i_23 = 0U;
        }
        uint _S1582 = _S1581 | i_23;
        if((*(&kernelContext_99)->g_run_0) != 0U)
        {
            i_23 = 32U;
        }
        else
        {
            i_23 = 0U;
        }
        ((&kernelContext_99)->islands_0+_S1544)->info_0[int(2)] = _S1582 | i_23;
    }
    uint _S1583 = (isl_30.coverage_0) | 16384U;
    uint _S1584 = seen_bits_0(&kernelContext_99);
    uint _S1585 = _S1583 | _S1584;
    if((*(&kernelContext_99)->g_run_0) != 0U)
    {
        i_23 = 128U;
    }
    else
    {
        i_23 = 0U;
    }
    ((&kernelContext_99)->islands_0+_S1544)->coverage_0 = _S1585 | i_23;
    return;
}

void impactor_integrate_at_0(uint ii_3, uint tid_23, bool fused_4, KernelContext_0 thread* kernelContext_100)
{
    float3 p_18;
    if(ii_3 >= (kernelContext_100->params_0->impactor_count_0))
    {
        return;
    }
    uint _S1586 = kernelContext_100->params_0->halt_index_0;
    Island_natural_0 device* _S1587 = kernelContext_100->islands_0+kernelContext_100->params_0->halt_index_0;
    Impactor_natural_0 device* _S1588 = kernelContext_100->impactors_0+ii_3;
    float4 _S1589 = float4((*_S1588).position_err_1) ;
    float4 _S1590 = float4((*_S1588).velocity_1) ;
    float4 _S1591 = float4((*_S1588).velocity_err_1) ;
    float4 _S1592 = float4((*_S1588).angular_velocity_1) ;
    float4 _S1593 = float4((*_S1588).rotation_1) ;
    float4 _S1594 = float4((*_S1588).inertia0_2) ;
    float4 _S1595 = float4((*_S1588).inertia1_2) ;
    float4 _S1596 = float4((*_S1588).inertia2_2) ;
    float4 _S1597 = float4((*_S1588).inv0_2) ;
    float4 _S1598 = float4((*_S1588).inv1_2) ;
    float4 _S1599 = float4((*_S1588).inv2_2) ;
    float4 _S1600 = float4((*_S1588).shape_0) ;
    float4 _S1601 = float4((*_S1588).half_1) ;
    float4 _S1602 = float4((*_S1588).mat_0) ;
    float4 _S1603 = float4((*_S1588).crush_0) ;
    float4 _S1604 = float4((*_S1588).geom_0) ;
    float4 _S1605 = float4((*_S1588).rotation_err_1) ;
    float4 _S1606 = float4((*_S1588).ledger_0) ;
    uint4 _S1607 = uint4((*_S1588).cand_0) ;
    float4 _S1608 = float4((*_S1588).momentum_1) ;
    float4 _S1609 = float4((*_S1588).momentum_err_1) ;
    thread Impactor_0 imp_6;
    (&imp_6)->position_1 = float4((*_S1588).position_1) ;
    (&imp_6)->position_err_1 = _S1589;
    (&imp_6)->velocity_1 = _S1590;
    (&imp_6)->velocity_err_1 = _S1591;
    (&imp_6)->angular_velocity_1 = _S1592;
    (&imp_6)->rotation_1 = _S1593;
    (&imp_6)->inertia0_2 = _S1594;
    (&imp_6)->inertia1_2 = _S1595;
    (&imp_6)->inertia2_2 = _S1596;
    (&imp_6)->inv0_2 = _S1597;
    (&imp_6)->inv1_2 = _S1598;
    (&imp_6)->inv2_2 = _S1599;
    (&imp_6)->shape_0 = _S1600;
    (&imp_6)->half_1 = _S1601;
    (&imp_6)->mat_0 = _S1602;
    (&imp_6)->crush_0 = _S1603;
    (&imp_6)->geom_0 = _S1604;
    (&imp_6)->rotation_err_1 = _S1605;
    (&imp_6)->ledger_0 = _S1606;
    (&imp_6)->cand_0 = _S1607;
    (&imp_6)->momentum_1 = _S1608;
    (&imp_6)->momentum_err_1 = _S1609;
    uint4 _S1610 = uint4(_S1587->info_0) ;
    bool stop_0;
    if(((_S1610.z) & 1U) != 0U)
    {
        stop_0 = true;
    }
    else
    {
        uint _S1611 = _S1610.y;
        if(_S1611 != 0U)
        {
            stop_0 = ((&imp_6)->cand_0.w) >= _S1611;
        }
        else
        {
            stop_0 = false;
        }
    }
    if(fused_4)
    {
        if(stop_0)
        {
            stop_0 = true;
        }
        else
        {
            uint _S1612 = (uint4(_S1587->range_0) ).x;
            if(_S1612 != 0U)
            {
                stop_0 = _S1612 <= ((&imp_6)->cand_0.w + 1U);
            }
            else
            {
                stop_0 = false;
            }
        }
        if(tid_23 == 0U)
        {
            stop_seen_0(_S1586, (&imp_6)->cand_0.w + 1U, kernelContext_100);
        }
    }
    if(stop_0)
    {
        return;
    }
    if(((&imp_6)->cand_0.z) != 0U)
    {
        if(tid_23 == 0U)
        {
            (kernelContext_100->impactors_0+ii_3)->cand_0[int(3)] = (&imp_6)->cand_0.w + 1U;
        }
        return;
    }
    float4 _S1613 = float4(0.0f) ;
    thread float4 rf_1 = _S1613;
    thread float4 rt_0 = _S1613;
    uint k_25 = (&imp_6)->cand_0.x + tid_23;
    for(;;)
    {
        if(k_25 < ((&imp_6)->cand_0.y))
        {
        }
        else
        {
            break;
        }
        uint _S1614 = 3U * k_25;
        rf_1 = rf_1 + float4(*(kernelContext_100->scratch_0+(kernelContext_100->params_0->cand_base_0 + _S1614 + 1U))) ;
        rt_0 = rt_0 + float4(*(kernelContext_100->scratch_0+(kernelContext_100->params_0->cand_base_0 + _S1614 + 2U))) ;
        k_25 = k_25 + 256U;
    }
    group_sum2_0(tid_23, &rf_1, &rt_0, kernelContext_100);
    if(tid_23 != 0U)
    {
        return;
    }
    float dt_13 = kernelContext_100->params_0->dt_0;
    float3 _S1615 = float3(0.0f) ;
    float3 load_f_0;
    float3 load_t_0;
    if((kernelContext_100->params_0->has_ground_0) != 0U)
    {
        float3 _S1616 = (&imp_6)->half_1.xyz;
        thread Impactor_0 _S1617 = imp_6;
        Box_0 _S1618 = impactor_box_0(&_S1617, _S1615, _S1616);
        float3 _S1619 = (&imp_6)->velocity_1.xyz + (&imp_6)->velocity_err_1.xyz;
        float _S1620 = kernelContext_100->params_0->ground_modulus_0;
        float _S1621 = (&imp_6)->mat_0.x;
        float3 _S1622 = float3(0.0f, 0.0f, 1.0f);
        thread Box_0 _S1623 = _S1618;
        thread Box_0 _S1624 = _S1618;
        float _S1625 = contact_stiffness_0(_S1620, &_S1623, _S1621, &_S1624, _S1622);
        float _S1626 = (&imp_6)->position_1.z - kernelContext_100->params_0->ground_hi_0 + ((&imp_6)->position_err_1.z - kernelContext_100->params_0->ground_lo_0);
        uint total_points_0;
        if(((&imp_6)->shape_0.x) == 0.0f)
        {
            total_points_0 = 1U;
        }
        else
        {
            total_points_0 = 14U;
        }
        float _S1627 = _S1625 / float(min(total_points_0, 5U));
        uint s_12 = 0U;
        uint below_0 = 0U;
        for(;;)
        {
            if(s_12 < total_points_0)
            {
            }
            else
            {
                break;
            }
            if(((&imp_6)->shape_0.x) == 0.0f)
            {
                p_18 = float3(0.0f, 0.0f, - (&imp_6)->shape_0.y);
            }
            else
            {
                thread Box_0 _S1628 = _S1618;
                float3 _S1629 = sample_point_0(&_S1628, s_12, kernelContext_100);
                p_18 = _S1629;
            }
            if((_S1626 + p_18.z) < 0.0f)
            {
                below_0 = below_0 + 1U;
            }
            s_12 = s_12 + 1U;
        }
        if(below_0 > 0U)
        {
            sentinel_mark_0(8U, kernelContext_100);
        }
        s_12 = 0U;
        load_f_0 = _S1615;
        load_t_0 = _S1615;
        for(;;)
        {
            if(s_12 < total_points_0)
            {
            }
            else
            {
                break;
            }
            if(((&imp_6)->shape_0.x) == 0.0f)
            {
                p_18 = float3(0.0f, 0.0f, - (&imp_6)->shape_0.y);
            }
            else
            {
                thread Box_0 _S1630 = _S1618;
                float3 _S1631 = sample_point_0(&_S1630, s_12, kernelContext_100);
                p_18 = _S1631;
            }
            float depth_4 = - (_S1626 + p_18.z);
            if(depth_4 <= 0.0f)
            {
                s_12 = s_12 + 1U;
                continue;
            }
            thread float stored_8;
            thread float diss_6;
            float3 _S1632 = penalty_force_0(_S1627, (&imp_6)->mat_0.z, kernelContext_100->params_0->ground_friction_0, depth_4, _S1622, _S1619 + cross((&imp_6)->angular_velocity_1.xyz, p_18), dt_13, below_0, &stored_8, &diss_6, kernelContext_100);
            float3 load_f_1 = load_f_0 + _S1632;
            float3 load_t_1 = load_t_0 + cross(p_18, _S1632);
            thread float _S1633 = (&imp_6)->ledger_0.x;
            thread float _S1634 = (&imp_6)->ledger_0.y;
            comp_add1_0(&_S1633, &_S1634, diss_6);
            (&imp_6)->ledger_0.y = _S1634;
            (&imp_6)->ledger_0.x = _S1633;
            load_f_0 = load_f_1;
            load_t_0 = load_t_1;
            s_12 = s_12 + 1U;
        }
    }
    else
    {
        load_f_0 = _S1615;
        load_t_0 = _S1615;
    }
    float3 load_f_2 = rf_1.xyz + load_f_0;
    float3 load_t_2 = rt_0.xyz + load_t_0;
    float m_7 = (&imp_6)->mat_0.z;
    thread float3 vel_3 = (&imp_6)->velocity_1.xyz;
    thread float3 vel_err_1 = (&imp_6)->velocity_err_1.xyz;
    float3 _S1635 = float3(dt_13) ;
    comp_add_0(&vel_3, &vel_err_1, (load_f_2 / float3(m_7)  + kernelContext_100->params_0->gravity_0.xyz) * _S1635);
    Quat_0 _S1636 = quat_of_0((&imp_6)->rotation_1);
    thread Quat_0 q_15 = _S1636;
    thread float4 q_err_0 = (&imp_6)->rotation_err_1;
    thread float3 l_hi_0 = (&imp_6)->momentum_1.xyz;
    thread float3 l_err_1 = (&imp_6)->momentum_err_1.xyz;
    comp_add_0(&l_hi_0, &l_err_1, load_t_2 * _S1635);
    float3 l_9 = l_hi_0 + l_err_1;
    thread Quat_0 _S1637 = _S1636;
    float3 _S1638 = world_mul_0(&_S1637, (&imp_6)->inv0_2, (&imp_6)->inv1_2, (&imp_6)->inv2_2, l_9);
    thread float3 pos_1 = (&imp_6)->position_1.xyz;
    thread float3 pos_err_1 = (&imp_6)->position_err_1.xyz;
    comp_add_0(&pos_1, &pos_err_1, (vel_3 + vel_err_1) * _S1635);
    turn_left_0(&q_15, &q_err_0, _S1638, dt_13);
    thread Quat_0 _S1639 = q_15;
    float3 _S1640 = world_mul_0(&_S1639, (&imp_6)->inv0_2, (&imp_6)->inv1_2, (&imp_6)->inv2_2, l_9);
    (&imp_6)->angular_velocity_1 = float4(_S1640, 0.0f);
    thread Quat_0 _S1641 = q_15;
    float4 _S1642 = quat_vec_0(&_S1641);
    (&imp_6)->rotation_1 = _S1642;
    (&imp_6)->rotation_err_1 = q_err_0;
    (&imp_6)->momentum_1 = float4(l_hi_0, 0.0f);
    (&imp_6)->momentum_err_1 = float4(l_err_1, 0.0f);
    (&imp_6)->position_1 = float4(pos_1, 0.0f);
    (&imp_6)->position_err_1 = float4(pos_err_1, 0.0f);
    (&imp_6)->velocity_1 = float4(vel_3, 0.0f);
    (&imp_6)->velocity_err_1 = float4(vel_err_1, 0.0f);
    (&imp_6)->cand_0.w = (&imp_6)->cand_0.w + 1U;
    (&imp_6)->geom_0 = float4(1.0f, (&imp_6)->crush_0.w, 0.0f, 0.0f);
    Impactor_natural_0 device* _S1643 = kernelContext_100->impactors_0+ii_3;
    _S1643->position_1 = packed_float4(imp_6.position_1) ;
    _S1643->position_err_1 = packed_float4(imp_6.position_err_1) ;
    _S1643->velocity_1 = packed_float4(imp_6.velocity_1) ;
    _S1643->velocity_err_1 = packed_float4(imp_6.velocity_err_1) ;
    _S1643->angular_velocity_1 = packed_float4(imp_6.angular_velocity_1) ;
    _S1643->rotation_1 = packed_float4(imp_6.rotation_1) ;
    _S1643->inertia0_2 = packed_float4(imp_6.inertia0_2) ;
    _S1643->inertia1_2 = packed_float4(imp_6.inertia1_2) ;
    _S1643->inertia2_2 = packed_float4(imp_6.inertia2_2) ;
    _S1643->inv0_2 = packed_float4(imp_6.inv0_2) ;
    _S1643->inv1_2 = packed_float4(imp_6.inv1_2) ;
    _S1643->inv2_2 = packed_float4(imp_6.inv2_2) ;
    _S1643->shape_0 = packed_float4(imp_6.shape_0) ;
    _S1643->half_1 = packed_float4(imp_6.half_1) ;
    _S1643->mat_0 = packed_float4(imp_6.mat_0) ;
    _S1643->crush_0 = packed_float4(imp_6.crush_0) ;
    _S1643->geom_0 = packed_float4(imp_6.geom_0) ;
    _S1643->rotation_err_1 = packed_float4(imp_6.rotation_err_1) ;
    _S1643->ledger_0 = packed_float4(imp_6.ledger_0) ;
    _S1643->cand_0 = packed_uint4(imp_6.cand_0) ;
    _S1643->momentum_1 = packed_float4(imp_6.momentum_1) ;
    _S1643->momentum_err_1 = packed_float4(imp_6.momentum_err_1) ;
    uint k_26 = (&imp_6)->cand_0.w - 1U - kernelContext_100->params_0->step_start_0;
    if(k_26 < (kernelContext_100->params_0->record_stride_0))
    {
        uint at_9 = kernelContext_100->params_0->record_base_0 + 2U * (ii_3 * kernelContext_100->params_0->record_stride_0 + k_26);
        *(kernelContext_100->scratch_0+at_9) = packed_float4(float4(vel_3 + vel_err_1, 0.0f)) ;
        *(kernelContext_100->scratch_0+(at_9 + 1U)) = packed_float4(float4(pos_1 + pos_err_1, 0.0f)) ;
    }
    if(fused_4)
    {
        impactor_travel_check_0(ii_3, true, kernelContext_100);
    }
    return;
}

[[kernel]] void impactor_integrate(uint3 group_7 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], Params_0 constant* params_10 [[buffer(0)]], Island_natural_0 device* islands_10 [[buffer(9)]], uint device* index_10 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_10 [[buffer(3)]], packed_float4 device* state_12 [[buffer(6)]], packed_float4 device* scratch_10 [[buffer(8)]], packed_float4 device* contact_state_10 [[buffer(11)]], packed_float4 device* loads_10 [[buffer(5)]], Impactor_natural_0 device* impactors_10 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_10 [[buffer(7)]], BondStatic_natural_0 device* bonds_10 [[buffer(2)]], MaterialTable_0 constant* materials_10 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_101;
    (&kernelContext_101)->params_0 = params_10;
    (&kernelContext_101)->islands_0 = islands_10;
    (&kernelContext_101)->index_0 = index_10;
    (&kernelContext_101)->chunks_0 = chunks_10;
    (&kernelContext_101)->state_0 = state_12;
    (&kernelContext_101)->scratch_0 = scratch_10;
    (&kernelContext_101)->contact_state_0 = contact_state_10;
    (&kernelContext_101)->loads_0 = loads_10;
    (&kernelContext_101)->impactors_0 = impactors_10;
    (&kernelContext_101)->bond_dyn_0 = bond_dyn_10;
    (&kernelContext_101)->bonds_0 = bonds_10;
    (&kernelContext_101)->materials_0 = materials_10;
    threadgroup array<float4, int(32)> g_part_a_10;
    (&kernelContext_101)->g_part_a_0 = &g_part_a_10;
    threadgroup array<float4, int(32)> g_part_b_10;
    (&kernelContext_101)->g_part_b_0 = &g_part_b_10;
    threadgroup array<uint, int(5)> g_seen_10;
    (&kernelContext_101)->g_seen_0 = &g_seen_10;
    threadgroup uint g_wide_run_10;
    (&kernelContext_101)->g_wide_run_0 = &g_wide_run_10;
    threadgroup array<uint, int(4096)> g_label_10;
    (&kernelContext_101)->g_label_0 = &g_label_10;
    threadgroup uint g_changed_10;
    (&kernelContext_101)->g_changed_0 = &g_changed_10;
    threadgroup uint g_halt_10;
    (&kernelContext_101)->g_halt_0 = &g_halt_10;
    threadgroup uint g_run_10;
    (&kernelContext_101)->g_run_0 = &g_run_10;
    threadgroup uint g_sums_10;
    (&kernelContext_101)->g_sums_0 = &g_sums_10;
    threadgroup array<uint, int(512)> g_label_narrow_10;
    (&kernelContext_101)->g_label_narrow_0 = &g_label_narrow_10;
    threadgroup uint g_changed_narrow_10;
    (&kernelContext_101)->g_changed_narrow_0 = &g_changed_narrow_10;
    impactor_integrate_at_0(group_7.x, thread_7.x, false, &kernelContext_101);
    return;
}

void contact_sum_at_0(uint g_3, KernelContext_0 thread* kernelContext_102)
{
    bool _S1644;
    if(g_3 >= (kernelContext_102->params_0->seg_count_0))
    {
        _S1644 = true;
    }
    else
    {
        bool _S1645 = stopped_0(kernelContext_102);
        _S1644 = _S1645;
    }
    if(_S1644)
    {
        return;
    }
    contact_segment_0(g_3, kernelContext_102);
    return;
}

[[kernel]] void contact_sums(uint3 id_2 [[thread_position_in_grid]], Params_0 constant* params_11 [[buffer(0)]], Island_natural_0 device* islands_11 [[buffer(9)]], uint device* index_11 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_11 [[buffer(3)]], packed_float4 device* state_13 [[buffer(6)]], packed_float4 device* scratch_11 [[buffer(8)]], packed_float4 device* contact_state_11 [[buffer(11)]], packed_float4 device* loads_11 [[buffer(5)]], Impactor_natural_0 device* impactors_11 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_11 [[buffer(7)]], BondStatic_natural_0 device* bonds_11 [[buffer(2)]], MaterialTable_0 constant* materials_11 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_103;
    (&kernelContext_103)->params_0 = params_11;
    (&kernelContext_103)->islands_0 = islands_11;
    (&kernelContext_103)->index_0 = index_11;
    (&kernelContext_103)->chunks_0 = chunks_11;
    (&kernelContext_103)->state_0 = state_13;
    (&kernelContext_103)->scratch_0 = scratch_11;
    (&kernelContext_103)->contact_state_0 = contact_state_11;
    (&kernelContext_103)->loads_0 = loads_11;
    (&kernelContext_103)->impactors_0 = impactors_11;
    (&kernelContext_103)->bond_dyn_0 = bond_dyn_11;
    (&kernelContext_103)->bonds_0 = bonds_11;
    (&kernelContext_103)->materials_0 = materials_11;
    threadgroup array<float4, int(32)> g_part_a_11;
    (&kernelContext_103)->g_part_a_0 = &g_part_a_11;
    threadgroup array<float4, int(32)> g_part_b_11;
    (&kernelContext_103)->g_part_b_0 = &g_part_b_11;
    threadgroup array<uint, int(5)> g_seen_11;
    (&kernelContext_103)->g_seen_0 = &g_seen_11;
    threadgroup uint g_wide_run_11;
    (&kernelContext_103)->g_wide_run_0 = &g_wide_run_11;
    threadgroup array<uint, int(4096)> g_label_11;
    (&kernelContext_103)->g_label_0 = &g_label_11;
    threadgroup uint g_changed_11;
    (&kernelContext_103)->g_changed_0 = &g_changed_11;
    threadgroup uint g_halt_11;
    (&kernelContext_103)->g_halt_0 = &g_halt_11;
    threadgroup uint g_run_11;
    (&kernelContext_103)->g_run_0 = &g_run_11;
    threadgroup uint g_sums_11;
    (&kernelContext_103)->g_sums_0 = &g_sums_11;
    threadgroup array<uint, int(512)> g_label_narrow_11;
    (&kernelContext_103)->g_label_narrow_0 = &g_label_narrow_11;
    threadgroup uint g_changed_narrow_11;
    (&kernelContext_103)->g_changed_narrow_0 = &g_changed_narrow_11;
    contact_sum_at_0(id_2.x, &kernelContext_103);
    return;
}

void net_load_1(uint c_54, const Island_natural_0 thread* isl_31, const Rigid_0 thread* rg_18, uint k_27, float dt_14, bool contact_4, float3 thread* f_20, float3 thread* t_22, KernelContext_0 thread* kernelContext_104)
{
    ChunkStatic_natural_0 device* _S1646 = kernelContext_104->chunks_0+c_54;
    thread float3 fl_1;
    thread float3 tl_1;
    thread float3 rep_f_2;
    thread float3 rep_m_2;
    chunk_loads_0(c_54, c_54, &rg_18->rot_0, k_27, dt_14, contact_4, 0.0f, &fl_1, &tl_1, &rep_f_2, &rep_m_2, kernelContext_104);
    float4 _S1647 = float4(_S1646->center_0) ;
    float3 fc_4 = fl_1 + kernelContext_104->params_0->gravity_0.xyz * float3(_S1647.w) ;
    float3 _S1648 = _S1647.xyz;
    float3 _S1649 = (float4(isl_31->com_0) ).xyz;
    float3 _S1650 = rotate_0(&rg_18->rot_0, _S1648 + (float4(*(kernelContext_104->state_0+4U * c_54)) ).xyz - _S1649);
    *f_20 = *f_20 + fc_4;
    *t_22 = *t_22 + (cross(_S1650, fc_4) + tl_1);
    uint4 _S1651 = uint4(_S1646->load_range_0) ;
    if((_S1651.y) > (_S1651.x))
    {
        float3 _S1652 = rotate_0(&rg_18->rot_0, rep_f_2);
        *f_20 = *f_20 + _S1652;
        float3 _S1653 = rotate_0(&rg_18->rot_0, _S1648 - _S1649);
        float3 _S1654 = cross(_S1653, _S1652);
        float3 _S1655 = rotate_0(&rg_18->rot_0, rep_m_2);
        *t_22 = *t_22 + (_S1654 + _S1655);
    }
    return;
}

void chunk_update_1(uint c_55, const Island_natural_0 thread* isl_32, const Rigid_0 thread* rg_19, float dt_15, bool rml_1, uint step_1, bool contact_5, float thread* work_7, float thread* work_err_5, KernelContext_0 thread* kernelContext_105)
{
    ChunkStatic_natural_0 device* _S1656 = kernelContext_105->chunks_0+c_55;
    float3 _S1657 = float3(0.0f) ;
    uint _S1658 = kernelContext_105->index_0[c_55];
    float peak_1 = 0.0f;
    uint e_9 = _S1658;
    float3 fi_6 = _S1657;
    float3 mi_7 = _S1657;
    for(;;)
    {
        if(e_9 < (kernelContext_105->index_0)[c_55 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_8 = kernelContext_105->index_0[e_9];
        uint _S1659 = 3U * (entry_8 >> 1U);
        float4 _S1660 = float4(*(kernelContext_105->scratch_0+_S1659)) ;
        if((entry_8 & 1U) == 0U)
        {
            float3 mi_8 = mi_7 + (float4(*(kernelContext_105->scratch_0+(_S1659 + 1U))) ).xyz;
            fi_6 = fi_6 + _S1660.xyz;
            mi_7 = mi_8;
        }
        else
        {
            float3 mi_9 = mi_7 + (float4(*(kernelContext_105->scratch_0+(_S1659 + 2U))) ).xyz;
            fi_6 = fi_6 + - _S1660.xyz;
            mi_7 = mi_9;
        }
        float _S1661 = max(peak_1, _S1660.w);
        uint _S1662 = e_9 + 1U;
        peak_1 = _S1661;
        e_9 = _S1662;
    }
    uint _S1663 = 4U * c_55;
    float3 u_5 = (float4(*(kernelContext_105->state_0+_S1663)) ).xyz;
    uint _S1664 = _S1663 + 1U;
    float3 th_7 = (float4(*(kernelContext_105->state_0+_S1664)) ).xyz;
    uint _S1665 = _S1663 + 2U;
    float3 v_14 = (float4(*(kernelContext_105->state_0+_S1665)) ).xyz;
    uint _S1666 = _S1663 + 3U;
    float3 w_9 = (float4(*(kernelContext_105->state_0+_S1666)) ).xyz;
    float4 _S1667 = float4(_S1656->center_0) ;
    float mass_1 = _S1667.w;
    float3 _S1668 = _S1667.xyz;
    float3 _S1669 = (float4(isl_32->com_0) ).xyz;
    float3 _S1670 = rotate_0(&rg_19->rot_0, _S1668 + u_5 - _S1669);
    thread float3 f_load_1;
    thread float3 t_load_1;
    thread float3 rep_f_3;
    thread float3 rep_m_3;
    chunk_loads_0(c_55, c_55, &rg_19->rot_0, step_1, dt_15, contact_5, dt_15, &f_load_1, &t_load_1, &rep_f_3, &rep_m_3, kernelContext_105);
    record_chunk_load_0(c_55, f_load_1, t_load_1, kernelContext_105);
    float3 _S1671 = float3(mass_1) ;
    float3 f_world_3 = f_load_1 + kernelContext_105->params_0->gravity_0.xyz * _S1671;
    float3 t_world_3 = t_load_1;
    float3 f_world_4;
    float3 t_world_4;
    if(rml_1)
    {
        float3 f_world_5 = f_world_3 - (rg_19->a_0 + cross(rg_19->alpha_0, _S1670) + cross(rg_19->w_1, cross(rg_19->w_1, _S1670))) * _S1671;
        float4 _S1672 = float4(_S1656->inertia0_1) ;
        float4 _S1673 = float4(_S1656->inertia1_1) ;
        float4 _S1674 = float4(_S1656->inertia2_1) ;
        float3 _S1675 = world_mul_0(&rg_19->rot_0, _S1672, _S1673, _S1674, rg_19->alpha_0);
        float3 _S1676 = world_mul_0(&rg_19->rot_0, _S1672, _S1673, _S1674, rg_19->w_1);
        float3 t_world_5 = t_world_3 - (_S1675 + cross(rg_19->w_1, _S1676));
        f_world_4 = f_world_5;
        t_world_4 = t_world_5;
    }
    else
    {
        f_world_4 = f_world_3;
        t_world_4 = t_world_3;
    }
    float3 _S1677 = inverse_rotate_1(&rg_19->rot_0, f_world_4);
    float3 _S1678 = inverse_rotate_1(&rg_19->rot_0, t_world_4);
    float3 f_ext_1;
    float3 m_ext_2;
    if(rml_1)
    {
        float3 _S1679 = inverse_rotate_1(&rg_19->rot_0, rg_19->w_1);
        float4 _S1680 = float4(_S1656->inertia0_1) ;
        float4 _S1681 = float4(_S1656->inertia1_1) ;
        float4 _S1682 = float4(_S1656->inertia2_1) ;
        float3 i_w_1 = rows_mul_0(_S1680, _S1681, _S1682, w_9);
        float3 m_ext_3 = _S1678 - (cross(_S1679, i_w_1) + cross(w_9, rows_mul_0(_S1680, _S1681, _S1682, _S1679)) + cross(w_9, i_w_1));
        f_ext_1 = _S1677 - cross(_S1679, v_14) * float3((2.0f * mass_1)) ;
        m_ext_2 = m_ext_3;
    }
    else
    {
        f_ext_1 = _S1677;
        m_ext_2 = _S1678;
    }
    float3 f_21 = f_ext_1 + rep_f_3 + fi_6;
    float3 m_8 = m_ext_2 + rep_m_3 + mi_7;
    uint support_2 = (uint4(_S1656->info_1) ).x;
    float3 _S1683 = float3((float4(*(kernelContext_105->state_0+_S1664)) ).w, (float4(*(kernelContext_105->state_0+_S1665)) ).w, (float4(*(kernelContext_105->state_0+_S1666)) ).w);
    float3 reaction_2;
    float3 u_6;
    float3 th_8;
    float3 v_15;
    float3 w_10;
    if(support_2 == 1U)
    {
        reaction_2 = - f_21;
        u_6 = u_5;
        th_8 = th_7;
        v_15 = _S1657;
        w_10 = _S1657;
    }
    else
    {
        float4 _S1684 = float4(_S1656->scale_0) ;
        float3 w_11 = w_9 + rows_mul_0(float4(_S1656->inv0_1) , float4(_S1656->inv1_1) , float4(_S1656->inv2_1) , m_8) * float3((dt_15 * _S1684.z)) ;
        float3 _S1685 = float3(dt_15) ;
        float3 th_9 = th_7 + w_11 * _S1685;
        if(support_2 == 2U)
        {
            reaction_2 = - f_21;
            u_6 = u_5;
            th_8 = _S1657;
        }
        else
        {
            float3 v_16 = v_14 + f_21 * float3((dt_15 * _S1684.y)) ;
            float3 u_7 = u_5 + v_16 * _S1685;
            reaction_2 = _S1683;
            u_6 = u_7;
            th_8 = v_16;
        }
        float3 _S1686 = th_8;
        th_8 = th_9;
        v_15 = _S1686;
        w_10 = w_11;
    }
    *(kernelContext_105->state_0+_S1663) = packed_float4(float4(u_6, peak_1)) ;
    *(kernelContext_105->state_0+_S1664) = packed_float4(float4(th_8, reaction_2.x)) ;
    *(kernelContext_105->state_0+_S1665) = packed_float4(float4(v_15, reaction_2.y)) ;
    *(kernelContext_105->state_0+_S1666) = packed_float4(float4(w_10, reaction_2.z)) ;
    float3 _S1687 = rotate_0(&rg_19->rot_0, _S1668 + u_6 - _S1669);
    float3 _S1688 = rg_19->vel_0 + rg_19->vel_err_0 + cross(rg_19->w_1, _S1687);
    float3 _S1689 = rotate_0(&rg_19->rot_0, v_15);
    float3 v_world_1 = _S1688 + _S1689;
    float3 _S1690 = rotate_0(&rg_19->rot_0, w_10);
    comp_add1_0(work_7, work_err_5, (dot(f_load_1, v_world_1) + dot(t_load_1, rg_19->w_1 + _S1690)) * dt_15);
    return;
}

[[kernel]] void island_frame(uint3 group_8 [[threadgroup_position_in_grid]], uint3 thread_8 [[thread_position_in_threadgroup]], Params_0 constant* params_12 [[buffer(0)]], Island_natural_0 device* islands_12 [[buffer(9)]], uint device* index_12 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_12 [[buffer(3)]], packed_float4 device* state_14 [[buffer(6)]], packed_float4 device* scratch_12 [[buffer(8)]], packed_float4 device* contact_state_12 [[buffer(11)]], packed_float4 device* loads_12 [[buffer(5)]], Impactor_natural_0 device* impactors_12 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_12 [[buffer(7)]], BondStatic_natural_0 device* bonds_12 [[buffer(2)]], MaterialTable_0 constant* materials_12 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_106;
    (&kernelContext_106)->params_0 = params_12;
    (&kernelContext_106)->islands_0 = islands_12;
    (&kernelContext_106)->index_0 = index_12;
    (&kernelContext_106)->chunks_0 = chunks_12;
    (&kernelContext_106)->state_0 = state_14;
    (&kernelContext_106)->scratch_0 = scratch_12;
    (&kernelContext_106)->contact_state_0 = contact_state_12;
    (&kernelContext_106)->loads_0 = loads_12;
    (&kernelContext_106)->impactors_0 = impactors_12;
    (&kernelContext_106)->bond_dyn_0 = bond_dyn_12;
    (&kernelContext_106)->bonds_0 = bonds_12;
    (&kernelContext_106)->materials_0 = materials_12;
    threadgroup array<float4, int(32)> g_part_a_12;
    (&kernelContext_106)->g_part_a_0 = &g_part_a_12;
    threadgroup array<float4, int(32)> g_part_b_12;
    (&kernelContext_106)->g_part_b_0 = &g_part_b_12;
    threadgroup array<uint, int(5)> g_seen_12;
    (&kernelContext_106)->g_seen_0 = &g_seen_12;
    threadgroup uint g_wide_run_12;
    (&kernelContext_106)->g_wide_run_0 = &g_wide_run_12;
    threadgroup array<uint, int(4096)> g_label_12;
    (&kernelContext_106)->g_label_0 = &g_label_12;
    threadgroup uint g_changed_12;
    (&kernelContext_106)->g_changed_0 = &g_changed_12;
    threadgroup uint g_halt_12;
    (&kernelContext_106)->g_halt_0 = &g_halt_12;
    threadgroup uint g_run_12;
    (&kernelContext_106)->g_run_0 = &g_run_12;
    threadgroup uint g_sums_12;
    (&kernelContext_106)->g_sums_0 = &g_sums_12;
    threadgroup array<uint, int(512)> g_label_narrow_12;
    (&kernelContext_106)->g_label_narrow_0 = &g_label_narrow_12;
    threadgroup uint g_changed_narrow_12;
    (&kernelContext_106)->g_changed_narrow_0 = &g_changed_narrow_12;
    island_frame_at_3(group_8.x, thread_8.x, &kernelContext_106);
    return;
}

[[kernel]] void island_frame_bonds_wide(uint3 group_9 [[threadgroup_position_in_grid]], uint3 thread_9 [[thread_position_in_threadgroup]], Params_0 constant* params_13 [[buffer(0)]], Island_natural_0 device* islands_13 [[buffer(9)]], uint device* index_13 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_13 [[buffer(3)]], packed_float4 device* state_15 [[buffer(6)]], packed_float4 device* scratch_13 [[buffer(8)]], packed_float4 device* contact_state_13 [[buffer(11)]], packed_float4 device* loads_13 [[buffer(5)]], Impactor_natural_0 device* impactors_13 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_13 [[buffer(7)]], BondStatic_natural_0 device* bonds_13 [[buffer(2)]], MaterialTable_0 constant* materials_13 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_107;
    (&kernelContext_107)->params_0 = params_13;
    (&kernelContext_107)->islands_0 = islands_13;
    (&kernelContext_107)->index_0 = index_13;
    (&kernelContext_107)->chunks_0 = chunks_13;
    (&kernelContext_107)->state_0 = state_15;
    (&kernelContext_107)->scratch_0 = scratch_13;
    (&kernelContext_107)->contact_state_0 = contact_state_13;
    (&kernelContext_107)->loads_0 = loads_13;
    (&kernelContext_107)->impactors_0 = impactors_13;
    (&kernelContext_107)->bond_dyn_0 = bond_dyn_13;
    (&kernelContext_107)->bonds_0 = bonds_13;
    (&kernelContext_107)->materials_0 = materials_13;
    threadgroup array<float4, int(32)> g_part_a_13;
    (&kernelContext_107)->g_part_a_0 = &g_part_a_13;
    threadgroup array<float4, int(32)> g_part_b_13;
    (&kernelContext_107)->g_part_b_0 = &g_part_b_13;
    threadgroup array<uint, int(5)> g_seen_13;
    (&kernelContext_107)->g_seen_0 = &g_seen_13;
    threadgroup uint g_wide_run_13;
    (&kernelContext_107)->g_wide_run_0 = &g_wide_run_13;
    threadgroup array<uint, int(4096)> g_label_13;
    (&kernelContext_107)->g_label_0 = &g_label_13;
    threadgroup uint g_changed_13;
    (&kernelContext_107)->g_changed_0 = &g_changed_13;
    threadgroup uint g_halt_13;
    (&kernelContext_107)->g_halt_0 = &g_halt_13;
    threadgroup uint g_run_13;
    (&kernelContext_107)->g_run_0 = &g_run_13;
    threadgroup uint g_sums_13;
    (&kernelContext_107)->g_sums_0 = &g_sums_13;
    threadgroup array<uint, int(512)> g_label_narrow_13;
    (&kernelContext_107)->g_label_narrow_0 = &g_label_narrow_13;
    threadgroup uint g_changed_narrow_13;
    (&kernelContext_107)->g_changed_narrow_0 = &g_changed_narrow_13;
    island_frame_at_2(group_9.x, thread_9.x, &kernelContext_107);
    return;
}

[[kernel]] void contact_islands(uint3 group_10 [[threadgroup_position_in_grid]], uint3 thread_10 [[thread_position_in_threadgroup]], Params_0 constant* params_14 [[buffer(0)]], Island_natural_0 device* islands_14 [[buffer(9)]], uint device* index_14 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_14 [[buffer(3)]], packed_float4 device* state_16 [[buffer(6)]], packed_float4 device* scratch_14 [[buffer(8)]], packed_float4 device* contact_state_14 [[buffer(11)]], packed_float4 device* loads_14 [[buffer(5)]], Impactor_natural_0 device* impactors_14 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_14 [[buffer(7)]], BondStatic_natural_0 device* bonds_14 [[buffer(2)]], MaterialTable_0 constant* materials_14 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_108;
    (&kernelContext_108)->params_0 = params_14;
    (&kernelContext_108)->islands_0 = islands_14;
    (&kernelContext_108)->index_0 = index_14;
    (&kernelContext_108)->chunks_0 = chunks_14;
    (&kernelContext_108)->state_0 = state_16;
    (&kernelContext_108)->scratch_0 = scratch_14;
    (&kernelContext_108)->contact_state_0 = contact_state_14;
    (&kernelContext_108)->loads_0 = loads_14;
    (&kernelContext_108)->impactors_0 = impactors_14;
    (&kernelContext_108)->bond_dyn_0 = bond_dyn_14;
    (&kernelContext_108)->bonds_0 = bonds_14;
    (&kernelContext_108)->materials_0 = materials_14;
    threadgroup array<float4, int(32)> g_part_a_14;
    (&kernelContext_108)->g_part_a_0 = &g_part_a_14;
    threadgroup array<float4, int(32)> g_part_b_14;
    (&kernelContext_108)->g_part_b_0 = &g_part_b_14;
    threadgroup array<uint, int(5)> g_seen_14;
    (&kernelContext_108)->g_seen_0 = &g_seen_14;
    threadgroup uint g_wide_run_14;
    (&kernelContext_108)->g_wide_run_0 = &g_wide_run_14;
    threadgroup array<uint, int(4096)> g_label_14;
    (&kernelContext_108)->g_label_0 = &g_label_14;
    threadgroup uint g_changed_14;
    (&kernelContext_108)->g_changed_0 = &g_changed_14;
    threadgroup uint g_halt_14;
    (&kernelContext_108)->g_halt_0 = &g_halt_14;
    threadgroup uint g_run_14;
    (&kernelContext_108)->g_run_0 = &g_run_14;
    threadgroup uint g_sums_14;
    (&kernelContext_108)->g_sums_0 = &g_sums_14;
    threadgroup array<uint, int(512)> g_label_narrow_14;
    (&kernelContext_108)->g_label_narrow_0 = &g_label_narrow_14;
    threadgroup uint g_changed_narrow_14;
    (&kernelContext_108)->g_changed_narrow_0 = &g_changed_narrow_14;
    island_frame_at_1(group_10.x, thread_10.x, &kernelContext_108);
    return;
}

void wide_chunks_at_0(uint gx_0, uint tid_24, KernelContext_0 thread* kernelContext_109)
{
    WideGroup_0 _S1691 = wide_group_0(kernelContext_109->params_0->wide_chunk_table_0, gx_0, kernelContext_109);
    thread Island_natural_0 _S1692 = *(kernelContext_109->islands_0+_S1691.island_3);
    bool _S1693 = wide_enter_1(tid_24, _S1691.island_3, kernelContext_109);
    if(!_S1693)
    {
        return;
    }
    uint _S1694 = (uint4((&_S1692)->info_0) ).x;
    bool anchored_0 = (_S1694 & 1U) != 0U;
    if((_S1694 & 16U) != 0U)
    {
        if(tid_24 == 0U)
        {
            *(kernelContext_109->scratch_0+(kernelContext_109->params_0->wide_base_0 + 8U * gx_0 + 6U)) = packed_float4(float4(0.0f) ) ;
        }
        return;
    }
    thread WideGroup_0 _S1695 = _S1691;
    Rigid_0 _S1696 = wide_rigid_frame_1(tid_24, &_S1692, &_S1695, kernelContext_109);
    thread float work_8 = 0.0f;
    thread float work_err_6 = 0.0f;
    float3 _S1697 = float3(0.0f) ;
    thread float3 tu_7 = _S1697;
    thread float3 pv_7 = _S1697;
    uint c_56 = _S1691.begin_1 + tid_24;
    if(c_56 < (_S1691.end_0))
    {
        float _S1698 = kernelContext_109->params_0->dt_0;
        bool _S1699 = (kernelContext_109->params_0->rigid_motion_loads_0) != 0U;
        uint _S1700 = wide_step_1(&_S1692, kernelContext_109);
        bool _S1701 = (_S1694 & 4U) != 0U;
        thread Rigid_0 _S1702 = _S1696;
        chunk_update_1(c_56, &_S1692, &_S1702, _S1698, _S1699, _S1700, _S1701, &work_8, &work_err_6, kernelContext_109);
        if(!anchored_0)
        {
            drift_moments_0(c_56, &tu_7, &pv_7, kernelContext_109);
        }
    }
    thread float3 wsum_4 = float3(work_8, work_err_6, 0.0f);
    thread float3 unused_7 = _S1697;
    group_sum3_0(tid_24, &wsum_4, &unused_7, kernelContext_109);
    bool _S1703 = !anchored_0;
    if(_S1703)
    {
        group_sum3_0(tid_24, &tu_7, &pv_7, kernelContext_109);
    }
    if(tid_24 == 0U)
    {
        uint _S1704 = 8U * gx_0;
        *(kernelContext_109->scratch_0+(kernelContext_109->params_0->wide_base_0 + _S1704 + 6U)) = packed_float4(float4(wsum_4, 0.0f)) ;
        if(_S1703)
        {
            wide_store_0(gx_0, 2U, tu_7, pv_7, kernelContext_109);
        }
        *(kernelContext_109->scratch_0+(kernelContext_109->params_0->wide_base_0 + _S1704 + 7U)) = packed_float4(float4(float((*kernelContext_109->g_seen_0)[0U]), float((*kernelContext_109->g_seen_0)[1U]), float((*kernelContext_109->g_seen_0)[2U]), 0.0f)) ;
    }
    return;
}

[[kernel]] void wide_chunks(uint3 group_11 [[threadgroup_position_in_grid]], uint3 thread_11 [[thread_position_in_threadgroup]], Params_0 constant* params_15 [[buffer(0)]], Island_natural_0 device* islands_15 [[buffer(9)]], uint device* index_15 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_15 [[buffer(3)]], packed_float4 device* state_17 [[buffer(6)]], packed_float4 device* scratch_15 [[buffer(8)]], packed_float4 device* contact_state_15 [[buffer(11)]], packed_float4 device* loads_15 [[buffer(5)]], Impactor_natural_0 device* impactors_15 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_15 [[buffer(7)]], BondStatic_natural_0 device* bonds_15 [[buffer(2)]], MaterialTable_0 constant* materials_15 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_110;
    (&kernelContext_110)->params_0 = params_15;
    (&kernelContext_110)->islands_0 = islands_15;
    (&kernelContext_110)->index_0 = index_15;
    (&kernelContext_110)->chunks_0 = chunks_15;
    (&kernelContext_110)->state_0 = state_17;
    (&kernelContext_110)->scratch_0 = scratch_15;
    (&kernelContext_110)->contact_state_0 = contact_state_15;
    (&kernelContext_110)->loads_0 = loads_15;
    (&kernelContext_110)->impactors_0 = impactors_15;
    (&kernelContext_110)->bond_dyn_0 = bond_dyn_15;
    (&kernelContext_110)->bonds_0 = bonds_15;
    (&kernelContext_110)->materials_0 = materials_15;
    threadgroup array<float4, int(32)> g_part_a_15;
    (&kernelContext_110)->g_part_a_0 = &g_part_a_15;
    threadgroup array<float4, int(32)> g_part_b_15;
    (&kernelContext_110)->g_part_b_0 = &g_part_b_15;
    threadgroup array<uint, int(5)> g_seen_15;
    (&kernelContext_110)->g_seen_0 = &g_seen_15;
    threadgroup uint g_wide_run_15;
    (&kernelContext_110)->g_wide_run_0 = &g_wide_run_15;
    threadgroup array<uint, int(4096)> g_label_15;
    (&kernelContext_110)->g_label_0 = &g_label_15;
    threadgroup uint g_changed_15;
    (&kernelContext_110)->g_changed_0 = &g_changed_15;
    threadgroup uint g_halt_15;
    (&kernelContext_110)->g_halt_0 = &g_halt_15;
    threadgroup uint g_run_15;
    (&kernelContext_110)->g_run_0 = &g_run_15;
    threadgroup uint g_sums_15;
    (&kernelContext_110)->g_sums_0 = &g_sums_15;
    threadgroup array<uint, int(512)> g_label_narrow_15;
    (&kernelContext_110)->g_label_narrow_0 = &g_label_narrow_15;
    threadgroup uint g_changed_narrow_15;
    (&kernelContext_110)->g_changed_narrow_0 = &g_changed_narrow_15;
    wide_chunks_at_0(group_11.x, thread_11.x, &kernelContext_110);
    return;
}

[[kernel]] void sums_and_impactors(uint3 group_12 [[threadgroup_position_in_grid]], uint3 thread_12 [[thread_position_in_threadgroup]], Params_0 constant* params_16 [[buffer(0)]], Island_natural_0 device* islands_16 [[buffer(9)]], uint device* index_16 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_16 [[buffer(3)]], packed_float4 device* state_18 [[buffer(6)]], packed_float4 device* scratch_16 [[buffer(8)]], packed_float4 device* contact_state_16 [[buffer(11)]], packed_float4 device* loads_16 [[buffer(5)]], Impactor_natural_0 device* impactors_16 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_16 [[buffer(7)]], BondStatic_natural_0 device* bonds_16 [[buffer(2)]], MaterialTable_0 constant* materials_16 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_111;
    (&kernelContext_111)->params_0 = params_16;
    (&kernelContext_111)->islands_0 = islands_16;
    (&kernelContext_111)->index_0 = index_16;
    (&kernelContext_111)->chunks_0 = chunks_16;
    (&kernelContext_111)->state_0 = state_18;
    (&kernelContext_111)->scratch_0 = scratch_16;
    (&kernelContext_111)->contact_state_0 = contact_state_16;
    (&kernelContext_111)->loads_0 = loads_16;
    (&kernelContext_111)->impactors_0 = impactors_16;
    (&kernelContext_111)->bond_dyn_0 = bond_dyn_16;
    (&kernelContext_111)->bonds_0 = bonds_16;
    (&kernelContext_111)->materials_0 = materials_16;
    threadgroup array<float4, int(32)> g_part_a_16;
    (&kernelContext_111)->g_part_a_0 = &g_part_a_16;
    threadgroup array<float4, int(32)> g_part_b_16;
    (&kernelContext_111)->g_part_b_0 = &g_part_b_16;
    threadgroup array<uint, int(5)> g_seen_16;
    (&kernelContext_111)->g_seen_0 = &g_seen_16;
    threadgroup uint g_wide_run_16;
    (&kernelContext_111)->g_wide_run_0 = &g_wide_run_16;
    threadgroup array<uint, int(4096)> g_label_16;
    (&kernelContext_111)->g_label_0 = &g_label_16;
    threadgroup uint g_changed_16;
    (&kernelContext_111)->g_changed_0 = &g_changed_16;
    threadgroup uint g_halt_16;
    (&kernelContext_111)->g_halt_0 = &g_halt_16;
    threadgroup uint g_run_16;
    (&kernelContext_111)->g_run_0 = &g_run_16;
    threadgroup uint g_sums_16;
    (&kernelContext_111)->g_sums_0 = &g_sums_16;
    threadgroup array<uint, int(512)> g_label_narrow_16;
    (&kernelContext_111)->g_label_narrow_0 = &g_label_narrow_16;
    threadgroup uint g_changed_narrow_16;
    (&kernelContext_111)->g_changed_narrow_0 = &g_changed_narrow_16;
    uint sums_groups_0 = (params_16->seg_count_0 + 255U) / 256U;
    uint _S1705 = group_12.x;
    if(_S1705 < sums_groups_0)
    {
        contact_sum_at_0(_S1705 * 256U + thread_12.x, &kernelContext_111);
    }
    else
    {
        impactor_integrate_at_0(_S1705 - sums_groups_0, thread_12.x, false, &kernelContext_111);
    }
    return;
}

[[kernel]] void islands_and_wide_chunks(uint3 group_13 [[threadgroup_position_in_grid]], uint3 thread_13 [[thread_position_in_threadgroup]], Params_0 constant* params_17 [[buffer(0)]], Island_natural_0 device* islands_17 [[buffer(9)]], uint device* index_17 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_17 [[buffer(3)]], packed_float4 device* state_19 [[buffer(6)]], packed_float4 device* scratch_17 [[buffer(8)]], packed_float4 device* contact_state_17 [[buffer(11)]], packed_float4 device* loads_17 [[buffer(5)]], Impactor_natural_0 device* impactors_17 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_17 [[buffer(7)]], BondStatic_natural_0 device* bonds_17 [[buffer(2)]], MaterialTable_0 constant* materials_17 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_112;
    (&kernelContext_112)->params_0 = params_17;
    (&kernelContext_112)->islands_0 = islands_17;
    (&kernelContext_112)->index_0 = index_17;
    (&kernelContext_112)->chunks_0 = chunks_17;
    (&kernelContext_112)->state_0 = state_19;
    (&kernelContext_112)->scratch_0 = scratch_17;
    (&kernelContext_112)->contact_state_0 = contact_state_17;
    (&kernelContext_112)->loads_0 = loads_17;
    (&kernelContext_112)->impactors_0 = impactors_17;
    (&kernelContext_112)->bond_dyn_0 = bond_dyn_17;
    (&kernelContext_112)->bonds_0 = bonds_17;
    (&kernelContext_112)->materials_0 = materials_17;
    threadgroup array<float4, int(32)> g_part_a_17;
    (&kernelContext_112)->g_part_a_0 = &g_part_a_17;
    threadgroup array<float4, int(32)> g_part_b_17;
    (&kernelContext_112)->g_part_b_0 = &g_part_b_17;
    threadgroup array<uint, int(5)> g_seen_17;
    (&kernelContext_112)->g_seen_0 = &g_seen_17;
    threadgroup uint g_wide_run_17;
    (&kernelContext_112)->g_wide_run_0 = &g_wide_run_17;
    threadgroup array<uint, int(4096)> g_label_17;
    (&kernelContext_112)->g_label_0 = &g_label_17;
    threadgroup uint g_changed_17;
    (&kernelContext_112)->g_changed_0 = &g_changed_17;
    threadgroup uint g_halt_17;
    (&kernelContext_112)->g_halt_0 = &g_halt_17;
    threadgroup uint g_run_17;
    (&kernelContext_112)->g_run_0 = &g_run_17;
    threadgroup uint g_sums_17;
    (&kernelContext_112)->g_sums_0 = &g_sums_17;
    threadgroup array<uint, int(512)> g_label_narrow_17;
    (&kernelContext_112)->g_label_narrow_0 = &g_label_narrow_17;
    threadgroup uint g_changed_narrow_17;
    (&kernelContext_112)->g_changed_narrow_0 = &g_changed_narrow_17;
    uint _S1706 = group_13.x;
    if(_S1706 < (params_17->halt_index_0))
    {
        island_frame_at_3(_S1706, thread_13.x, &kernelContext_112);
    }
    else
    {
        wide_chunks_at_0(_S1706 - params_17->halt_index_0, thread_13.x, &kernelContext_112);
    }
    return;
}

[[kernel]] void contact_islands_and_wide_chunks(uint3 group_14 [[threadgroup_position_in_grid]], uint3 thread_14 [[thread_position_in_threadgroup]], Params_0 constant* params_18 [[buffer(0)]], Island_natural_0 device* islands_18 [[buffer(9)]], uint device* index_18 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_18 [[buffer(3)]], packed_float4 device* state_20 [[buffer(6)]], packed_float4 device* scratch_18 [[buffer(8)]], packed_float4 device* contact_state_18 [[buffer(11)]], packed_float4 device* loads_18 [[buffer(5)]], Impactor_natural_0 device* impactors_18 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_18 [[buffer(7)]], BondStatic_natural_0 device* bonds_18 [[buffer(2)]], MaterialTable_0 constant* materials_18 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_113;
    (&kernelContext_113)->params_0 = params_18;
    (&kernelContext_113)->islands_0 = islands_18;
    (&kernelContext_113)->index_0 = index_18;
    (&kernelContext_113)->chunks_0 = chunks_18;
    (&kernelContext_113)->state_0 = state_20;
    (&kernelContext_113)->scratch_0 = scratch_18;
    (&kernelContext_113)->contact_state_0 = contact_state_18;
    (&kernelContext_113)->loads_0 = loads_18;
    (&kernelContext_113)->impactors_0 = impactors_18;
    (&kernelContext_113)->bond_dyn_0 = bond_dyn_18;
    (&kernelContext_113)->bonds_0 = bonds_18;
    (&kernelContext_113)->materials_0 = materials_18;
    threadgroup array<float4, int(32)> g_part_a_18;
    (&kernelContext_113)->g_part_a_0 = &g_part_a_18;
    threadgroup array<float4, int(32)> g_part_b_18;
    (&kernelContext_113)->g_part_b_0 = &g_part_b_18;
    threadgroup array<uint, int(5)> g_seen_18;
    (&kernelContext_113)->g_seen_0 = &g_seen_18;
    threadgroup uint g_wide_run_18;
    (&kernelContext_113)->g_wide_run_0 = &g_wide_run_18;
    threadgroup array<uint, int(4096)> g_label_18;
    (&kernelContext_113)->g_label_0 = &g_label_18;
    threadgroup uint g_changed_18;
    (&kernelContext_113)->g_changed_0 = &g_changed_18;
    threadgroup uint g_halt_18;
    (&kernelContext_113)->g_halt_0 = &g_halt_18;
    threadgroup uint g_run_18;
    (&kernelContext_113)->g_run_0 = &g_run_18;
    threadgroup uint g_sums_18;
    (&kernelContext_113)->g_sums_0 = &g_sums_18;
    threadgroup array<uint, int(512)> g_label_narrow_18;
    (&kernelContext_113)->g_label_narrow_0 = &g_label_narrow_18;
    threadgroup uint g_changed_narrow_18;
    (&kernelContext_113)->g_changed_narrow_0 = &g_changed_narrow_18;
    uint _S1707 = group_14.x;
    if(_S1707 < (params_18->halt_index_0))
    {
        island_frame_at_1(_S1707, thread_14.x, &kernelContext_113);
    }
    else
    {
        wide_chunks_at_0(_S1707 - params_18->halt_index_0, thread_14.x, &kernelContext_113);
    }
    return;
}

void wide_bonds_at_0(uint gx_1, uint tid_25, KernelContext_0 thread* kernelContext_114)
{
    bool bond_group_0 = gx_1 < (kernelContext_114->params_0->wide_bond_groups_0);
    WideGroup_0 wg_2;
    if(bond_group_0)
    {
        WideGroup_0 _S1708 = wide_group_0(kernelContext_114->params_0->wide_bond_table_0, gx_1, kernelContext_114);
        wg_2 = _S1708;
    }
    else
    {
        WideGroup_0 _S1709 = wide_group_0(kernelContext_114->params_0->wide_chunk_table_0, gx_1 - kernelContext_114->params_0->wide_bond_groups_0, kernelContext_114);
        wg_2 = _S1709;
    }
    WideGroup_0 _S1710 = wg_2;
    thread Island_natural_0 _S1711 = *(kernelContext_114->islands_0+wg_2.island_3);
    bool _S1712 = wide_enter_1(tid_25, wg_2.island_3, kernelContext_114);
    if(!_S1712)
    {
        return;
    }
    uint _S1713 = wide_step_1(&_S1711, kernelContext_114);
    if(bond_group_0)
    {
        uint4 _S1714 = uint4((&_S1711)->info_0) ;
        if(((_S1714.x) & 16U) != 0U)
        {
            return;
        }
        uint i_24 = wg_2.begin_1 + tid_25;
        bool _S1715;
        if(i_24 < (wg_2.end_0))
        {
            bool _S1716 = bond_update_0(i_24, kernelContext_114->params_0->dt_0, (kernelContext_114->params_0->fracture_0) != 0U, _S1714.w + 1U, true, kernelContext_114);
            _S1715 = _S1716;
        }
        else
        {
            _S1715 = false;
        }
        if(_S1715)
        {
            (kernelContext_114->islands_0+_S1710.island_3)->info_0[int(2)] = (_S1714.z) | 2U;
        }
        return;
    }
    uint _S1717 = (uint4((&_S1711)->info_0) ).x;
    if((_S1717 & 1U) != 0U)
    {
        return;
    }
    float3 _S1718 = float3(0.0f) ;
    thread float3 f_22 = _S1718;
    thread float3 t_23 = _S1718;
    uint c_57 = wg_2.begin_1 + tid_25;
    if(c_57 < (wg_2.end_0))
    {
        Rigid_0 _S1719 = rigid_of_1(&_S1711);
        float _S1720 = kernelContext_114->params_0->dt_0;
        bool _S1721 = (_S1717 & 4U) != 0U;
        thread Rigid_0 _S1722 = _S1719;
        net_load_1(c_57, &_S1711, &_S1722, _S1713, _S1720, _S1721, &f_22, &t_23, kernelContext_114);
    }
    group_sum3_0(tid_25, &f_22, &t_23, kernelContext_114);
    if(tid_25 == 0U)
    {
        wide_store_0(gx_1 - kernelContext_114->params_0->wide_bond_groups_0, 0U, f_22, t_23, kernelContext_114);
    }
    return;
}

[[kernel]] void wide_bonds(uint3 group_15 [[threadgroup_position_in_grid]], uint3 thread_15 [[thread_position_in_threadgroup]], Params_0 constant* params_19 [[buffer(0)]], Island_natural_0 device* islands_19 [[buffer(9)]], uint device* index_19 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_19 [[buffer(3)]], packed_float4 device* state_21 [[buffer(6)]], packed_float4 device* scratch_19 [[buffer(8)]], packed_float4 device* contact_state_19 [[buffer(11)]], packed_float4 device* loads_19 [[buffer(5)]], Impactor_natural_0 device* impactors_19 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_19 [[buffer(7)]], BondStatic_natural_0 device* bonds_19 [[buffer(2)]], MaterialTable_0 constant* materials_19 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_115;
    (&kernelContext_115)->params_0 = params_19;
    (&kernelContext_115)->islands_0 = islands_19;
    (&kernelContext_115)->index_0 = index_19;
    (&kernelContext_115)->chunks_0 = chunks_19;
    (&kernelContext_115)->state_0 = state_21;
    (&kernelContext_115)->scratch_0 = scratch_19;
    (&kernelContext_115)->contact_state_0 = contact_state_19;
    (&kernelContext_115)->loads_0 = loads_19;
    (&kernelContext_115)->impactors_0 = impactors_19;
    (&kernelContext_115)->bond_dyn_0 = bond_dyn_19;
    (&kernelContext_115)->bonds_0 = bonds_19;
    (&kernelContext_115)->materials_0 = materials_19;
    threadgroup array<float4, int(32)> g_part_a_19;
    (&kernelContext_115)->g_part_a_0 = &g_part_a_19;
    threadgroup array<float4, int(32)> g_part_b_19;
    (&kernelContext_115)->g_part_b_0 = &g_part_b_19;
    threadgroup array<uint, int(5)> g_seen_19;
    (&kernelContext_115)->g_seen_0 = &g_seen_19;
    threadgroup uint g_wide_run_19;
    (&kernelContext_115)->g_wide_run_0 = &g_wide_run_19;
    threadgroup array<uint, int(4096)> g_label_19;
    (&kernelContext_115)->g_label_0 = &g_label_19;
    threadgroup uint g_changed_19;
    (&kernelContext_115)->g_changed_0 = &g_changed_19;
    threadgroup uint g_halt_19;
    (&kernelContext_115)->g_halt_0 = &g_halt_19;
    threadgroup uint g_run_19;
    (&kernelContext_115)->g_run_0 = &g_run_19;
    threadgroup uint g_sums_19;
    (&kernelContext_115)->g_sums_0 = &g_sums_19;
    threadgroup array<uint, int(512)> g_label_narrow_19;
    (&kernelContext_115)->g_label_narrow_0 = &g_label_narrow_19;
    threadgroup uint g_changed_narrow_19;
    (&kernelContext_115)->g_changed_narrow_0 = &g_changed_narrow_19;
    wide_bonds_at_0(group_15.x, thread_15.x, &kernelContext_115);
    return;
}

void bond_group_at_0(uint table_1, uint gx_2, uint tid_26, KernelContext_0 thread* kernelContext_116)
{
    WideGroup_0 _S1723 = wide_group_0(table_1, gx_2, kernelContext_116);
    Island_natural_0 isl_33 = *(kernelContext_116->islands_0+_S1723.island_3);
    bool _S1724 = wide_enter_1(tid_26, _S1723.island_3, kernelContext_116);
    bool _S1725;
    if(!_S1724)
    {
        _S1725 = true;
    }
    else
    {
        _S1725 = (((uint4(isl_33.info_0) ).x) & 16U) != 0U;
    }
    if(_S1725)
    {
        return;
    }
    uint i_25 = _S1723.begin_1 + tid_26;
    if(i_25 < (_S1723.end_0))
    {
        bool _S1726 = bond_update_0(i_25, kernelContext_116->params_0->dt_0, (kernelContext_116->params_0->fracture_0) != 0U, (uint4(isl_33.info_0) ).w + 1U, true, kernelContext_116);
        _S1725 = _S1726;
    }
    else
    {
        _S1725 = false;
    }
    if(_S1725)
    {
        (kernelContext_116->islands_0+_S1723.island_3)->info_0[int(2)] = ((uint4(isl_33.info_0) ).z) | 2U;
    }
    return;
}

[[kernel]] void sums_impactors_bonds(uint3 group_16 [[threadgroup_position_in_grid]], uint3 thread_16 [[thread_position_in_threadgroup]], Params_0 constant* params_20 [[buffer(0)]], Island_natural_0 device* islands_20 [[buffer(9)]], uint device* index_20 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_20 [[buffer(3)]], packed_float4 device* state_22 [[buffer(6)]], packed_float4 device* scratch_20 [[buffer(8)]], packed_float4 device* contact_state_20 [[buffer(11)]], packed_float4 device* loads_20 [[buffer(5)]], Impactor_natural_0 device* impactors_20 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_20 [[buffer(7)]], BondStatic_natural_0 device* bonds_20 [[buffer(2)]], MaterialTable_0 constant* materials_20 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_117;
    (&kernelContext_117)->params_0 = params_20;
    (&kernelContext_117)->islands_0 = islands_20;
    (&kernelContext_117)->index_0 = index_20;
    (&kernelContext_117)->chunks_0 = chunks_20;
    (&kernelContext_117)->state_0 = state_22;
    (&kernelContext_117)->scratch_0 = scratch_20;
    (&kernelContext_117)->contact_state_0 = contact_state_20;
    (&kernelContext_117)->loads_0 = loads_20;
    (&kernelContext_117)->impactors_0 = impactors_20;
    (&kernelContext_117)->bond_dyn_0 = bond_dyn_20;
    (&kernelContext_117)->bonds_0 = bonds_20;
    (&kernelContext_117)->materials_0 = materials_20;
    threadgroup array<float4, int(32)> g_part_a_20;
    (&kernelContext_117)->g_part_a_0 = &g_part_a_20;
    threadgroup array<float4, int(32)> g_part_b_20;
    (&kernelContext_117)->g_part_b_0 = &g_part_b_20;
    threadgroup array<uint, int(5)> g_seen_20;
    (&kernelContext_117)->g_seen_0 = &g_seen_20;
    threadgroup uint g_wide_run_20;
    (&kernelContext_117)->g_wide_run_0 = &g_wide_run_20;
    threadgroup array<uint, int(4096)> g_label_20;
    (&kernelContext_117)->g_label_0 = &g_label_20;
    threadgroup uint g_changed_20;
    (&kernelContext_117)->g_changed_0 = &g_changed_20;
    threadgroup uint g_halt_20;
    (&kernelContext_117)->g_halt_0 = &g_halt_20;
    threadgroup uint g_run_20;
    (&kernelContext_117)->g_run_0 = &g_run_20;
    threadgroup uint g_sums_20;
    (&kernelContext_117)->g_sums_0 = &g_sums_20;
    threadgroup array<uint, int(512)> g_label_narrow_20;
    (&kernelContext_117)->g_label_narrow_0 = &g_label_narrow_20;
    threadgroup uint g_changed_narrow_20;
    (&kernelContext_117)->g_changed_narrow_0 = &g_changed_narrow_20;
    uint sums_groups_1 = (params_20->seg_count_0 + 255U) / 256U;
    uint _S1727 = max(params_20->impactor_count_0, 1U);
    uint _S1728 = group_16.x;
    if(_S1728 < sums_groups_1)
    {
        contact_sum_at_0(_S1728 * 256U + thread_16.x, &kernelContext_117);
    }
    else
    {
        if(_S1728 < (sums_groups_1 + _S1727))
        {
            impactor_integrate_at_0(_S1728 - sums_groups_1, thread_16.x, false, &kernelContext_117);
        }
        else
        {
            uint gb_0 = _S1728 - sums_groups_1 - _S1727;
            bool narrow_0 = gb_0 < ((&kernelContext_117)->params_0->contact_bond_groups_0);
            uint _S1729;
            if(narrow_0)
            {
                _S1729 = (&kernelContext_117)->params_0->contact_bond_table_0;
            }
            else
            {
                _S1729 = (&kernelContext_117)->params_0->wide_bond_table_0;
            }
            uint _S1730;
            if(narrow_0)
            {
                _S1730 = gb_0;
            }
            else
            {
                _S1730 = gb_0 - (&kernelContext_117)->params_0->contact_bond_groups_0;
            }
            bond_group_at_0(_S1729, _S1730, thread_16.x, &kernelContext_117);
        }
    }
    return;
}

[[kernel]] void forces_and_bonds(uint3 group_17 [[threadgroup_position_in_grid]], uint3 thread_17 [[thread_position_in_threadgroup]], Params_0 constant* params_21 [[buffer(0)]], Island_natural_0 device* islands_21 [[buffer(9)]], uint device* index_21 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_21 [[buffer(3)]], packed_float4 device* state_23 [[buffer(6)]], packed_float4 device* scratch_21 [[buffer(8)]], packed_float4 device* contact_state_21 [[buffer(11)]], packed_float4 device* loads_21 [[buffer(5)]], Impactor_natural_0 device* impactors_21 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_21 [[buffer(7)]], BondStatic_natural_0 device* bonds_21 [[buffer(2)]], MaterialTable_0 constant* materials_21 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_118;
    (&kernelContext_118)->params_0 = params_21;
    (&kernelContext_118)->islands_0 = islands_21;
    (&kernelContext_118)->index_0 = index_21;
    (&kernelContext_118)->chunks_0 = chunks_21;
    (&kernelContext_118)->state_0 = state_23;
    (&kernelContext_118)->scratch_0 = scratch_21;
    (&kernelContext_118)->contact_state_0 = contact_state_21;
    (&kernelContext_118)->loads_0 = loads_21;
    (&kernelContext_118)->impactors_0 = impactors_21;
    (&kernelContext_118)->bond_dyn_0 = bond_dyn_21;
    (&kernelContext_118)->bonds_0 = bonds_21;
    (&kernelContext_118)->materials_0 = materials_21;
    threadgroup array<float4, int(32)> g_part_a_21;
    (&kernelContext_118)->g_part_a_0 = &g_part_a_21;
    threadgroup array<float4, int(32)> g_part_b_21;
    (&kernelContext_118)->g_part_b_0 = &g_part_b_21;
    threadgroup array<uint, int(5)> g_seen_21;
    (&kernelContext_118)->g_seen_0 = &g_seen_21;
    threadgroup uint g_wide_run_21;
    (&kernelContext_118)->g_wide_run_0 = &g_wide_run_21;
    threadgroup array<uint, int(4096)> g_label_21;
    (&kernelContext_118)->g_label_0 = &g_label_21;
    threadgroup uint g_changed_21;
    (&kernelContext_118)->g_changed_0 = &g_changed_21;
    threadgroup uint g_halt_21;
    (&kernelContext_118)->g_halt_0 = &g_halt_21;
    threadgroup uint g_run_21;
    (&kernelContext_118)->g_run_0 = &g_run_21;
    threadgroup uint g_sums_21;
    (&kernelContext_118)->g_sums_0 = &g_sums_21;
    threadgroup array<uint, int(512)> g_label_narrow_21;
    (&kernelContext_118)->g_label_narrow_0 = &g_label_narrow_21;
    threadgroup uint g_changed_narrow_21;
    (&kernelContext_118)->g_changed_narrow_0 = &g_changed_narrow_21;
    uint _S1731 = params_21->pair_count_0 + params_21->cand_count_0;
    uint _S1732 = pair_lane_count_0(&kernelContext_118);
    uint force_groups_0 = (_S1731 * _S1732 + 255U) / 256U;
    uint _S1733 = group_17.x;
    if(_S1733 < force_groups_0)
    {
        contact_forces_at_0(_S1733 * 256U + thread_17.x, true, &kernelContext_118);
    }
    else
    {
        bond_group_at_0((&kernelContext_118)->params_0->contact_bond_table_0, _S1733 - force_groups_0, thread_17.x, &kernelContext_118);
    }
    return;
}

[[kernel]] void islands_sums_impactors(uint3 group_18 [[threadgroup_position_in_grid]], uint3 thread_18 [[thread_position_in_threadgroup]], Params_0 constant* params_22 [[buffer(0)]], Island_natural_0 device* islands_22 [[buffer(9)]], uint device* index_22 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_22 [[buffer(3)]], packed_float4 device* state_24 [[buffer(6)]], packed_float4 device* scratch_22 [[buffer(8)]], packed_float4 device* contact_state_22 [[buffer(11)]], packed_float4 device* loads_22 [[buffer(5)]], Impactor_natural_0 device* impactors_22 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_22 [[buffer(7)]], BondStatic_natural_0 device* bonds_22 [[buffer(2)]], MaterialTable_0 constant* materials_22 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_119;
    (&kernelContext_119)->params_0 = params_22;
    (&kernelContext_119)->islands_0 = islands_22;
    (&kernelContext_119)->index_0 = index_22;
    (&kernelContext_119)->chunks_0 = chunks_22;
    (&kernelContext_119)->state_0 = state_24;
    (&kernelContext_119)->scratch_0 = scratch_22;
    (&kernelContext_119)->contact_state_0 = contact_state_22;
    (&kernelContext_119)->loads_0 = loads_22;
    (&kernelContext_119)->impactors_0 = impactors_22;
    (&kernelContext_119)->bond_dyn_0 = bond_dyn_22;
    (&kernelContext_119)->bonds_0 = bonds_22;
    (&kernelContext_119)->materials_0 = materials_22;
    threadgroup array<float4, int(32)> g_part_a_22;
    (&kernelContext_119)->g_part_a_0 = &g_part_a_22;
    threadgroup array<float4, int(32)> g_part_b_22;
    (&kernelContext_119)->g_part_b_0 = &g_part_b_22;
    threadgroup array<uint, int(5)> g_seen_22;
    (&kernelContext_119)->g_seen_0 = &g_seen_22;
    threadgroup uint g_wide_run_22;
    (&kernelContext_119)->g_wide_run_0 = &g_wide_run_22;
    threadgroup array<uint, int(4096)> g_label_22;
    (&kernelContext_119)->g_label_0 = &g_label_22;
    threadgroup uint g_changed_22;
    (&kernelContext_119)->g_changed_0 = &g_changed_22;
    threadgroup uint g_halt_22;
    (&kernelContext_119)->g_halt_0 = &g_halt_22;
    threadgroup uint g_run_22;
    (&kernelContext_119)->g_run_0 = &g_run_22;
    threadgroup uint g_sums_22;
    (&kernelContext_119)->g_sums_0 = &g_sums_22;
    threadgroup array<uint, int(512)> g_label_narrow_22;
    (&kernelContext_119)->g_label_narrow_0 = &g_label_narrow_22;
    threadgroup uint g_changed_narrow_22;
    (&kernelContext_119)->g_changed_narrow_0 = &g_changed_narrow_22;
    uint _S1734 = group_18.x;
    if(_S1734 < (params_22->halt_index_0))
    {
        island_frame_at_0(_S1734, thread_18.x, &kernelContext_119);
    }
    else
    {
        impactor_integrate_at_0(_S1734 - params_22->halt_index_0, thread_18.x, true, &kernelContext_119);
    }
    return;
}

[[kernel]] void travel_start(uint3 id_3 [[thread_position_in_grid]], Params_0 constant* params_23 [[buffer(0)]], Island_natural_0 device* islands_23 [[buffer(9)]], uint device* index_23 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_23 [[buffer(3)]], packed_float4 device* state_25 [[buffer(6)]], packed_float4 device* scratch_23 [[buffer(8)]], packed_float4 device* contact_state_23 [[buffer(11)]], packed_float4 device* loads_23 [[buffer(5)]], Impactor_natural_0 device* impactors_23 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_23 [[buffer(7)]], BondStatic_natural_0 device* bonds_23 [[buffer(2)]], MaterialTable_0 constant* materials_23 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_120;
    (&kernelContext_120)->params_0 = params_23;
    (&kernelContext_120)->islands_0 = islands_23;
    (&kernelContext_120)->index_0 = index_23;
    (&kernelContext_120)->chunks_0 = chunks_23;
    (&kernelContext_120)->state_0 = state_25;
    (&kernelContext_120)->scratch_0 = scratch_23;
    (&kernelContext_120)->contact_state_0 = contact_state_23;
    (&kernelContext_120)->loads_0 = loads_23;
    (&kernelContext_120)->impactors_0 = impactors_23;
    (&kernelContext_120)->bond_dyn_0 = bond_dyn_23;
    (&kernelContext_120)->bonds_0 = bonds_23;
    (&kernelContext_120)->materials_0 = materials_23;
    threadgroup array<float4, int(32)> g_part_a_23;
    (&kernelContext_120)->g_part_a_0 = &g_part_a_23;
    threadgroup array<float4, int(32)> g_part_b_23;
    (&kernelContext_120)->g_part_b_0 = &g_part_b_23;
    threadgroup array<uint, int(5)> g_seen_23;
    (&kernelContext_120)->g_seen_0 = &g_seen_23;
    threadgroup uint g_wide_run_23;
    (&kernelContext_120)->g_wide_run_0 = &g_wide_run_23;
    threadgroup array<uint, int(4096)> g_label_23;
    (&kernelContext_120)->g_label_0 = &g_label_23;
    threadgroup uint g_changed_23;
    (&kernelContext_120)->g_changed_0 = &g_changed_23;
    threadgroup uint g_halt_23;
    (&kernelContext_120)->g_halt_0 = &g_halt_23;
    threadgroup uint g_run_23;
    (&kernelContext_120)->g_run_0 = &g_run_23;
    threadgroup uint g_sums_23;
    (&kernelContext_120)->g_sums_0 = &g_sums_23;
    threadgroup array<uint, int(512)> g_label_narrow_23;
    (&kernelContext_120)->g_label_narrow_0 = &g_label_narrow_23;
    threadgroup uint g_changed_narrow_23;
    (&kernelContext_120)->g_changed_narrow_0 = &g_changed_narrow_23;
    uint i_26 = id_3.x;
    if(i_26 < (params_23->chunk_count_0))
    {
        travel_check_0(i_26, true, &kernelContext_120);
    }
    else
    {
        if((i_26 - params_23->chunk_count_0) < ((&kernelContext_120)->params_0->impactor_count_0))
        {
            impactor_travel_check_0(i_26 - params_23->chunk_count_0, true, &kernelContext_120);
        }
    }
    return;
}

