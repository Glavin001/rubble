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
    uint sload_base_0;
    uint implicit_contact_0;
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

uint4 group_uniform4_0(uint4 value_0, uint thread_0)
{
    return value_0;
}

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

uint group_uniform_0(uint value_1, uint thread_1)
{
    return value_1;
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
    float _S74 = max(float(points_0), 10.0f);
    float vn_0 = dot(rel_velocity_0, normal_1);
    float c_4 = 2.0f * kernelContext_11->params_0->zeta_0 * sqrt(k_3 * m_red_0);
    float _S75 = k_3 * depth_0;
    float _S76 = k_3 * dt_1;
    float _S77 = max((_S75 - (_S76 + c_4) * vn_0) / (1.0f + (_S76 * dt_1 + c_4 * dt_1) / (m_red_0 / _S74)), 0.0f);
    float3 vt_0 = rel_velocity_0 - normal_1 * float3(vn_0) ;
    float vt_mag_0 = length(vt_0);
    float _S78 = friction_0 * _S77;
    float _S79 = min(_S78, min(1.0f / _S74 * m_red_0 / dt_1, _S78 / 0.00100000004749745f) * vt_mag_0);
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
    float _S80 = c_4 * vn_0;
    float damping_power_0;
    if((_S75 - _S80) > 0.0f)
    {
        damping_power_0 = _S80 * vn_0;
    }
    else
    {
        damping_power_0 = _S75 * max(vn_0, 0.0f);
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

void ground_contact_0(uint c_5, bool account_0, float3 thread* f_0, float3 thread* t_2, KernelContext_0 thread* kernelContext_12)
{
    ChunkStatic_natural_0 device* _S81 = kernelContext_12->chunks_0+c_5;
    WorldPoint_0 _S82 = chunk_world_0(c_5, kernelContext_12);
    float above_0 = _S82.hi_0.z - kernelContext_12->params_0->ground_hi_0 + (_S82.lo_0.z - kernelContext_12->params_0->ground_lo_0) + _S82.rel_0.z;
    if((above_0 - (float4(_S81->half_0) ).w) > 0.0f)
    {
        return;
    }
    Box_0 _S83 = chunk_box_0(c_5, float3(0.0f) , kernelContext_12);
    float _S84 = kernelContext_12->params_0->ground_modulus_0;
    float _S85 = (float4(_S81->cmat_0) ).x;
    float3 _S86 = float3(0.0f, 0.0f, 1.0f);
    thread Box_0 _S87 = _S83;
    thread Box_0 _S88 = _S83;
    float _S89 = contact_stiffness_0(_S84, &_S87, _S85, &_S88, _S86);
    thread Box_0 _S90 = _S83;
    uint _S91 = sample_count_0(&_S90);
    uint s_1 = 0U;
    uint n_1 = 0U;
    for(;;)
    {
        if(s_1 < _S91)
        {
        }
        else
        {
            break;
        }
        thread Box_0 _S92 = _S83;
        float3 _S93 = sample_point_0(&_S92, s_1, kernelContext_12);
        if((above_0 + _S93.z) < 0.0f)
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
    chunk_velocity_0(c_5, &vc_0, &wc_0, kernelContext_12);
    thread float4 ledger_1 = float4(*(kernelContext_12->scratch_0+(kernelContext_12->params_0->ledger_base_0 + kernelContext_12->params_0->pair_count_0 + c_5))) ;
    s_1 = 0U;
    for(;;)
    {
        if(s_1 < _S91)
        {
        }
        else
        {
            break;
        }
        thread Box_0 _S94 = _S83;
        float3 _S95 = sample_point_0(&_S94, s_1, kernelContext_12);
        float _S96 = above_0 + _S95.z;
        if(!(_S96 < 0.0f))
        {
            s_1 = s_1 + 1U;
            continue;
        }
        thread float stored_1;
        thread float diss_0;
        float3 _S97 = penalty_force_0(_S89 / float(max(n_1, 5U)), (float4(_S81->center_0) ).w, kernelContext_12->params_0->ground_friction_0, - _S96, _S86, vc_0 + cross(wc_0, _S95), kernelContext_12->params_0->dt_0, n_1, &stored_1, &diss_0, kernelContext_12);
        *f_0 = *f_0 + _S97;
        *t_2 = *t_2 + cross(_S95, _S97);
        thread float _S98 = ledger_1.y;
        thread float _S99 = ledger_1.z;
        comp_add1_0(&_S98, &_S99, diss_0);
        ledger_1.z = _S99;
        ledger_1.y = _S98;
        s_1 = s_1 + 1U;
    }
    if(account_0)
    {
        *(kernelContext_12->scratch_0+(kernelContext_12->params_0->ledger_base_0 + kernelContext_12->params_0->pair_count_0 + c_5)) = packed_float4(ledger_1) ;
    }
    return;
}

void contact_segment_0(uint g_0, KernelContext_0 thread* kernelContext_13)
{
    uint _S100 = 3U * g_0;
    uint _S101 = kernelContext_13->index_0[kernelContext_13->params_0->seg_index_0 + _S100];
    uint begin_0 = kernelContext_13->index_0[kernelContext_13->params_0->seg_index_0 + _S100 + 1U];
    uint _S102 = kernelContext_13->index_0[kernelContext_13->params_0->seg_index_0 + _S100 + 2U];
    float3 _S103 = float3(0.0f) ;
    thread float3 f_1 = _S103;
    thread float3 t_3 = _S103;
    uint e_0 = begin_0;
    for(;;)
    {
        if(e_0 < _S102)
        {
        }
        else
        {
            break;
        }
        uint entry_0 = kernelContext_13->index_0[e_0];
        if(entry_0 == 2147483648U)
        {
            ground_contact_0(_S101, true, &f_1, &t_3, kernelContext_13);
            e_0 = e_0 + 1U;
            continue;
        }
        uint _S104 = 2U * entry_0;
        f_1 = f_1 + (float4(*(kernelContext_13->scratch_0+(kernelContext_13->params_0->slot_base_0 + _S104))) ).xyz;
        t_3 = t_3 + (float4(*(kernelContext_13->scratch_0+(kernelContext_13->params_0->slot_base_0 + _S104 + 1U))) ).xyz;
        e_0 = e_0 + 1U;
    }
    uint _S105 = 2U * g_0;
    *(kernelContext_13->scratch_0+(kernelContext_13->params_0->seg_base_0 + _S105)) = packed_float4(float4(f_1, 0.0f)) ;
    *(kernelContext_13->scratch_0+(kernelContext_13->params_0->seg_base_0 + _S105 + 1U)) = packed_float4(float4(t_3, 0.0f)) ;
    return;
}

float time_since_0(float4 origin_0, uint k_4, float dt_2, KernelContext_0 thread* kernelContext_14)
{
    return kernelContext_14->params_0->t_hi_0 - origin_0.x + (kernelContext_14->params_0->t_lo_0 - origin_0.y) + float(k_4) * dt_2;
}

float table_eval_0(uint offset_0, uint count_0, float tau_0, KernelContext_0 thread* kernelContext_15)
{
    float4 _S106 = float4(*(kernelContext_15->loads_0+offset_0)) ;
    if(tau_0 <= (_S106.x))
    {
        return _S106.y;
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
        uint _S107 = offset_0 + i_1;
        float4 _S108 = float4(*(kernelContext_15->loads_0+_S107)) ;
        float _S109 = _S108.x;
        if(tau_0 <= _S109)
        {
            float4 _S110 = float4(*(kernelContext_15->loads_0+(_S107 - 1U))) ;
            float _S111 = _S110.x;
            float _S112 = _S110.y;
            return _S112 + (tau_0 - _S111) / max(_S109 - _S111, 1.00000000317107685e-30f) * (_S108.y - _S112);
        }
        i_1 = i_1 + 1U;
    }
    return (float4(*(kernelContext_15->loads_0+(offset_0 + count_0 - 1U))) ).y;
}

float eval_function_0(uint term_0, uint k_5, float dt_3, float shift_0, KernelContext_0 thread* kernelContext_16)
{
    uint _S113 = 5U * term_0;
    uint4 info_2 = (as_type<uint4>((float4(*(kernelContext_16->loads_0+_S113)) )));
    float4 _S114 = float4(*(kernelContext_16->loads_0+(_S113 + 3U))) ;
    float4 _S115 = float4(*(kernelContext_16->loads_0+(_S113 + 4U))) ;
    uint kind_0 = info_2.z;
    if(kind_0 == 0U)
    {
        return _S114.z;
    }
    float _S116 = time_since_0(_S114, k_5, dt_3, kernelContext_16);
    float tau_1 = _S116 + shift_0;
    float shape_1;
    if(kind_0 == 1U)
    {
        if(tau_1 <= 0.0f)
        {
            shape_1 = 0.0f;
        }
        else
        {
            float _S117 = _S115.x;
            if(tau_1 >= _S117)
            {
                shape_1 = _S115.y;
            }
            else
            {
                shape_1 = _S115.y * tau_1 / _S117;
            }
        }
        return shape_1;
    }
    bool _S118;
    if(kind_0 == 2U)
    {
        if(tau_1 < 0.0f)
        {
            _S118 = true;
        }
        else
        {
            _S118 = tau_1 > (_S115.x);
        }
        if(_S118)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = _S115.y * sin(3.14159274101257324f * tau_1 / _S115.x);
        }
        return shape_1;
    }
    if(kind_0 == 3U)
    {
        float sn_0 = tau_1 / _S115.y;
        if(sn_0 < 0.0f)
        {
            _S118 = true;
        }
        else
        {
            _S118 = sn_0 > 1.0f;
        }
        if(_S118)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = _S115.x * (1.0f - sn_0) * exp(- _S115.z * sn_0);
        }
        return shape_1;
    }
    if(kind_0 == 4U)
    {
        float _S119 = table_eval_0(info_2.w, (as_type<uint>((_S115.x))), tau_1, kernelContext_16);
        return _S119;
    }
    if(kind_0 == 5U)
    {
        (*kernelContext_16->g_seen_0)[1U] = 1U;
        if(tau_1 < 0.0f)
        {
            return 0.0f;
        }
        float sn_1 = tau_1 / _S115.x;
        if(sn_1 < 0.0f)
        {
            _S118 = true;
        }
        else
        {
            _S118 = sn_1 > 1.0f;
        }
        if(_S118)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = (1.0f - sn_1) * exp(- _S115.y * sn_1);
        }
        float clearing_0 = _S114.w;
        float relax_0;
        if(clearing_0 > 0.0f)
        {
            relax_0 = max(1.0f - tau_1 / clearing_0, 0.0f);
        }
        else
        {
            relax_0 = 0.0f;
        }
        float _S120 = _S115.w;
        return (_S120 + (_S115.z - _S120) * relax_0) * shape_1;
    }
    if(kind_0 == 7U)
    {
        (*kernelContext_16->g_seen_0)[2U] = 1U;
        if(tau_1 < 0.0f)
        {
            return 0.0f;
        }
        float _S121 = _S115.y;
        if(tau_1 < _S121)
        {
            return _S115.x;
        }
        float s_2 = tau_1 - _S121;
        float _S122 = _S115.w;
        if(s_2 > _S122)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = _S115.z * sin(3.14159274101257324f * s_2 / _S122);
        }
        return shape_1;
    }
    float _S123 = _S115.x;
    if(_S123 <= 0.0f)
    {
        return 0.0f;
    }
    return clamp(1.0f - tau_1 / _S123, 0.0f, 1.0f);
}

void chunk_loads_0(uint _S124, uint _S125, const Quat_0 thread* _S126, uint _S127, float _S128, bool _S129, float _S130, float3 thread* _S131, float3 thread* _S132, float3 thread* _S133, float3 thread* _S134, KernelContext_0 thread* kernelContext_17)
{
    bool _S135;
    ChunkStatic_natural_0 device* _S136 = kernelContext_17->chunks_0+_S125;
    float3 _S137 = float3(0.0f) ;
    *_S131 = _S137;
    *_S132 = _S137;
    *_S133 = _S137;
    *_S134 = _S137;
    uint4 _S138 = uint4(_S136->load_range_0) ;
    uint term_1 = _S138.x;
    for(;;)
    {
        if(term_1 < (_S138.y))
        {
        }
        else
        {
            break;
        }
        (*kernelContext_17->g_seen_0)[0U] = 1U;
        uint _S139 = 5U * term_1;
        uint _S140 = (as_type<uint4>((float4(*(kernelContext_17->loads_0+_S139)) ))).y;
        bool replacement_0 = _S140 == 2U;
        float4 _S141 = float4(*(kernelContext_17->loads_0+(_S139 + 1U))) ;
        float4 _S142 = float4(*(kernelContext_17->loads_0+(_S139 + 2U))) ;
        float _S143;
        if(replacement_0)
        {
            _S143 = _S130;
        }
        else
        {
            _S143 = 0.0f;
        }
        float _S144 = eval_function_0(term_1, _S127, _S128, _S143, kernelContext_17);
        if(replacement_0)
        {
            float3 _S145 = float3(_S144) ;
            *_S133 = *_S133 + _S141.xyz * _S145;
            *_S134 = *_S134 + _S142.xyz * _S145;
            term_1 = term_1 + 1U;
            continue;
        }
        if(_S140 == 0U)
        {
            _S135 = true;
        }
        else
        {
            _S135 = _S140 == 3U;
        }
        float3 fw_0;
        if(_S135)
        {
            fw_0 = _S141.xyz * float3(_S144) ;
        }
        else
        {
            float3 _S146 = rotate_0(_S126, _S141.xyz);
            fw_0 = _S146 * float3((- _S144 * _S141.w)) ;
        }
        float3 lever_0;
        if(_S140 == 3U)
        {
            lever_0 = _S142.xyz - (float4(*(kernelContext_17->state_0+4U * _S124)) ).xyz;
        }
        else
        {
            lever_0 = _S142.xyz;
        }
        *_S131 = *_S131 + fw_0;
        float3 _S147 = rotate_0(_S126, lever_0);
        *_S132 = *_S132 + cross(_S147, fw_0);
        term_1 = term_1 + 1U;
    }
    if(_S129)
    {
        _S135 = ((uint4(_S136->cinfo_0) ).z) != 0U;
    }
    else
    {
        _S135 = false;
    }
    if(_S135)
    {
        uint4 _S148 = uint4(_S136->cinfo_0) ;
        uint g_1 = _S148.x;
        for(;;)
        {
            if(g_1 < (_S148.y))
            {
            }
            else
            {
                break;
            }
            uint _S149 = 2U * g_1;
            *_S131 = *_S131 + (float4(*(kernelContext_17->scratch_0+(kernelContext_17->params_0->seg_base_0 + _S149))) ).xyz;
            *_S132 = *_S132 + (float4(*(kernelContext_17->scratch_0+(kernelContext_17->params_0->seg_base_0 + _S149 + 1U))) ).xyz;
            g_1 = g_1 + 1U;
        }
    }
    return;
}

void chunk_external_0(uint _S150, uint _S151, const Quat_0 thread* _S152, uint _S153, float _S154, bool _S155, float3 thread* _S156, float3 thread* _S157, KernelContext_0 thread* kernelContext_18)
{
    thread float3 rf_0;
    thread float3 rm_0;
    chunk_loads_0(_S150, _S151, _S152, _S153, _S154, _S155, 0.0f, _S156, _S157, &rf_0, &rm_0, kernelContext_18);
    return;
}

void record_chunk_load_0(uint c_6, float3 f_2, float3 t_4, KernelContext_0 thread* kernelContext_19)
{
    if((kernelContext_19->params_0->solve_mode_0) == 0U)
    {
        return;
    }
    uint _S158 = 2U * c_6;
    *(kernelContext_19->scratch_0+(kernelContext_19->params_0->cload_base_0 + _S158)) = packed_float4(float4(f_2, 0.0f)) ;
    *(kernelContext_19->scratch_0+(kernelContext_19->params_0->cload_base_0 + _S158 + 1U)) = packed_float4(float4(t_4, 0.0f)) ;
    *(kernelContext_19->scratch_0+(kernelContext_19->params_0->cframe_base_0 + _S158)) = packed_float4(float4((float4(*(kernelContext_19->scratch_0+(kernelContext_19->params_0->cframe_base_0 + _S158))) ).xyz + f_2, 0.0f)) ;
    *(kernelContext_19->scratch_0+(kernelContext_19->params_0->cframe_base_0 + _S158 + 1U)) = packed_float4(float4((float4(*(kernelContext_19->scratch_0+(kernelContext_19->params_0->cframe_base_0 + _S158 + 1U))) ).xyz + t_4, 0.0f)) ;
    return;
}

float settled_chunk_load_0(uint c_7, const Quat_0 thread* rot_1, uint k_6, float dt_4, bool contact_0, KernelContext_0 thread* kernelContext_20)
{
    thread float3 f_3;
    thread float3 t_5;
    chunk_external_0(c_7, c_7, rot_1, k_6, dt_4, contact_0, &f_3, &t_5, kernelContext_20);
    record_chunk_load_0(c_7, f_3, t_5, kernelContext_20);
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
    float4 _S159 = (*kernelContext_21->g_part_a_0)[int(0)];
    float4 _S160 = (*kernelContext_21->g_part_b_0)[int(0)];
    uint w_4 = 1U;
    float4 sa_0 = _S159;
    float4 sb_0 = _S160;
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

void net_load_0(uint c_8, const Island_0 thread* isl_2, const Rigid_0 thread* rg_1, uint k_7, float dt_5, bool contact_1, float3 thread* f_4, float3 thread* t_6, KernelContext_0 thread* kernelContext_23)
{
    ChunkStatic_natural_0 device* _S161 = kernelContext_23->chunks_0+c_8;
    thread float3 fl_0;
    thread float3 tl_0;
    thread float3 rep_f_0;
    thread float3 rep_m_0;
    chunk_loads_0(c_8, c_8, &rg_1->rot_0, k_7, dt_5, contact_1, 0.0f, &fl_0, &tl_0, &rep_f_0, &rep_m_0, kernelContext_23);
    float4 _S162 = float4(_S161->center_0) ;
    float3 fc_0 = fl_0 + kernelContext_23->params_0->gravity_0.xyz * float3(_S162.w) ;
    float3 _S163 = _S162.xyz;
    float3 _S164 = isl_2->com_0.xyz;
    float3 _S165 = rotate_0(&rg_1->rot_0, _S163 + (float4(*(kernelContext_23->state_0+4U * c_8)) ).xyz - _S164);
    *f_4 = *f_4 + fc_0;
    *t_6 = *t_6 + (cross(_S165, fc_0) + tl_0);
    uint4 _S166 = uint4(_S161->load_range_0) ;
    if((_S166.y) > (_S166.x))
    {
        float3 _S167 = rotate_0(&rg_1->rot_0, rep_f_0);
        *f_4 = *f_4 + _S167;
        float3 _S168 = rotate_0(&rg_1->rot_0, _S163 - _S164);
        float3 _S169 = cross(_S168, _S167);
        float3 _S170 = rotate_0(&rg_1->rot_0, rep_m_0);
        *t_6 = *t_6 + (_S169 + _S170);
    }
    return;
}

float3 inverse_rotate_0(const Quat_0 thread* q_6, float3 v_4)
{
    thread Quat_0 c_9;
    (&c_9)->w_0 = q_6->w_0;
    (&c_9)->x_0 = - q_6->x_0;
    (&c_9)->y_0 = - q_6->y_0;
    (&c_9)->z_0 = - q_6->z_0;
    thread Quat_0 _S171 = c_9;
    float3 _S172 = rotate_0(&_S171, v_4);
    return _S172;
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
    float3 _S173 = inverse_rotate_0(q_8, v_7);
    float3 _S174 = rotate_1(q_8, rows_mul_0(r0_1, r1_1, r2_1, _S173));
    return _S174;
}

void rigid_acceleration_0(const Island_0 thread* isl_3, Rigid_0 thread* rg_2, float3 f_5, float3 t_8)
{
    thread Quat_0 _S175 = rg_2->rot_0;
    float3 _S176 = world_mul_0(&_S175, isl_3->inertia0_0, isl_3->inertia1_0, isl_3->inertia2_0, rg_2->w_1);
    rg_2->a_0 = f_5 / float3(isl_3->com_0.w) ;
    float3 _S177 = t_8 - cross(rg_2->w_1, _S176);
    thread Quat_0 _S178 = rg_2->rot_0;
    float3 _S179 = world_mul_0(&_S178, isl_3->inv0_0, isl_3->inv1_0, isl_3->inv2_0, _S177);
    rg_2->alpha_0 = _S179;
    rg_2->torque_0 = t_8;
    return;
}

float3 to_local_0(uint _S180, float3 _S181, KernelContext_0 thread* kernelContext_24)
{
    BondStatic_natural_0 device* _S182 = kernelContext_24->bonds_0+_S180;
    return float3(dot(_S181, (float4(_S182->t1_0) ).xyz), dot(_S181, (float4(_S182->t2_0) ).xyz), dot(_S181, (float4(_S182->normal_0) ).xyz));
}

bool is_damaged_0(const JointState_0 thread* st_0)
{
    bool _S183;
    if((st_0->damage_0) > 0.0f)
    {
        _S183 = true;
    }
    else
    {
        _S183 = (st_0->crush_1) > 0.0f;
    }
    return _S183;
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
    bool _S184;
    if((st_1->damage_0) < 1.0f)
    {
        _S184 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S184 = (st_1->rebar_broken_0) == 0.0f;
        }
        else
        {
            _S184 = false;
        }
    }
    return _S184;
}

Measures_0 stress_measures_0(const JointBond_natural_0 thread* b_9, float3 q_lin_0, float3 q_ang_0)
{
    float4 _S185 = float4(b_9->geom0_0) ;
    float area_2 = _S185.x;
    float _S186 = q_lin_0.z;
    float axial_0 = (metal::fast::divide((_S186), (area_2)));
    float4 _S187 = float4(b_9->geom1_0) ;
    float _S188 = (metal::fast::divide((abs(q_ang_0.x)), (_S187.x)));
    float _S189 = (metal::fast::divide((abs(q_ang_0.y)), (_S187.y)));
    float bending_0 = _S188 + _S189;
    float _S190 = q_lin_0.x;
    float _S191 = q_lin_0.y;
    float _S192 = (metal::fast::sqrt((_S190 * _S190 + _S191 * _S191)));
    float _S193 = (metal::fast::divide((_S192), (area_2)));
    float _S194 = (metal::fast::divide((abs(q_ang_0.z)), (_S185.w)));
    float shear_1 = _S193 + _S194;
    thread Measures_0 m_1;
    (&m_1)->tension_0 = axial_0 + bending_0;
    (&m_1)->shear_0 = shear_1;
    float _S195 = - axial_0;
    (&m_1)->normal_compression_0 = max(_S195, 0.0f);
    (&m_1)->compression_0 = _S195 + bending_0;
    (&m_1)->compressive_force_0 = max(- _S186, 0.0f);
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
    float4 _S196 = mat_1->dif_0;
    float ref_0 = mat_1->dif_0.x;
    if(r_1 <= ref_0)
    {
        return 1.0f;
    }
    float _S197 = _S196.z;
    float f_6;
    if(r_1 <= _S197)
    {
        float _S198 = (metal::fast::divide((r_1), (ref_0)));
        float _S199 = (metal::fast::pow((_S198), (_S196.y)));
        f_6 = _S199;
    }
    else
    {
        float _S200 = (metal::fast::divide((_S197), (ref_0)));
        float _S201 = (metal::fast::pow((_S200), (_S196.y)));
        float _S202 = (metal::fast::divide((r_1), (_S197)));
        float _S203 = (metal::fast::pow((_S202), (_S196.w)));
        f_6 = _S201 * _S203;
    }
    return clamp(f_6, 1.0f, mat_1->misc_0.x);
}

float fatigue_factor_0(const JointMaterial_0 constant* mat_2, float life_1)
{
    if(((mat_2->kind_flags_0.y) & 64U) == 0U)
    {
        return 1.0f;
    }
    float _S204 = clamp(life_1, 0.0f, 1.0f);
    float _S205 = (metal::fast::divide((1.0f), (mat_2->misc_0.y - 2.0f)));
    float _S206 = (metal::fast::pow((_S204), (_S205)));
    return _S206;
}

float float_bits_0(uint bits_3)
{
    return (as_type<float>((bits_3)));
}

float infinity_0()
{
    return float_bits_0(2139095040U);
}

float4 failure_indices_0(const JointMaterial_0 constant* mat_3, const JointBond_natural_0 thread* b_10, const Measures_0 thread* m_2, float multiplier_0)
{
    float fc_1 = mat_3->strength_0.y * multiplier_0;
    float _S207 = min(mat_3->strength_0.z * multiplier_0 + mat_3->strength_0.w * m_2->normal_compression_0, mat_3->energy_1.x * multiplier_0);
    thread float4 idx_0;
    float _S208 = (metal::fast::divide((m_2->tension_0), (mat_3->strength_0.x * multiplier_0)));
    idx_0.x = max(_S208, 0.0f);
    float _S209;
    if(_S207 > 0.0f)
    {
        float _S210 = (metal::fast::divide((m_2->shear_0), (_S207)));
        _S209 = _S210;
    }
    else
    {
        _S209 = infinity_0();
    }
    idx_0.y = _S209;
    float _S211 = (metal::fast::divide((m_2->compression_0), (fc_1)));
    idx_0.z = max(_S211, 0.0f);
    float _S212 = (float4(b_10->stiff1_0) ).y;
    if(_S212 > 0.0f)
    {
        float _S213 = (metal::fast::divide((m_2->compressive_force_0), (_S212)));
        _S209 = _S213;
    }
    else
    {
        _S209 = 0.0f;
    }
    idx_0.w = _S209;
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
    float _S214 = - h0_0;
    float _S215 = - h1_0;
    array<float2, int(4)> _S216 = { { float2(_S214, _S215), float2(h0_0, _S215), float2(h0_0, h1_0), float2(_S214, h1_0) } };
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
        uint _S217 = i_2;
        uint _S218 = i_2 + 1U;
        uint _S219 = _S218 % 4U;
        float _S220 = _S216[i_2].y;
        float _S221 = _S216[i_2].x;
        float fp_0 = dz_0 + ax_0 * _S220 - ay_0 * _S221;
        float _S222 = _S216[_S219].y;
        float _S223 = _S216[_S219].x;
        float fq_0 = dz_0 + ax_0 * _S222 - ay_0 * _S223;
        bool _S224 = fp_0 < 0.0f;
        if(_S224)
        {
            uint _S225 = count_2 + 1U;
            poly_0[count_2] = _S216[_S217];
            count_1 = _S225;
        }
        else
        {
            count_1 = count_2;
        }
        if(_S224 != (fq_0 < 0.0f))
        {
            float t_9 = fp_0 / (fp_0 - fq_0);
            uint _S226 = count_1 + 1U;
            poly_0[count_1] = float2(_S221 + t_9 * (_S223 - _S221), _S220 + t_9 * (_S222 - _S220));
            count_2 = _S226;
        }
        else
        {
            count_2 = count_1;
        }
        i_2 = _S218;
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
        float _S227 = o_0.x;
        float x0_0 = poly_0[i_2].x - _S227;
        float _S228 = o_0.y;
        float y0_0 = poly_0[i_2].y - _S228;
        uint _S229 = i_2 + 1U;
        uint _S230 = _S229 % count_2;
        float x1_0 = poly_0[_S230].x - _S227;
        float y1_0 = poly_0[_S230].y - _S228;
        float _S231 = x0_0 * y1_0;
        float _S232 = x1_0 * y0_0;
        float cr_0 = _S231 - _S232;
        float a_6 = a_5 + cr_0 / 2.0f;
        float sx_1 = sx_0 + (x0_0 + x1_0) * cr_0 / 6.0f;
        float sy_1 = sy_0 + (y0_0 + y1_0) * cr_0 / 6.0f;
        float ixx_1 = ixx_0 + (x0_0 * x0_0 + x0_0 * x1_0 + x1_0 * x1_0) * cr_0 / 12.0f;
        float iyy_1 = iyy_0 + (y0_0 * y0_0 + y0_0 * y1_0 + y1_0 * y1_0) * cr_0 / 12.0f;
        float ixy_1 = ixy_0 + (_S231 + 2.0f * x0_0 * y0_0 + 2.0f * x1_0 * y1_0 + _S232) * cr_0 / 24.0f;
        i_2 = _S229;
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
    float _S233 = a_5 * cx_0;
    (*region_0)[int(3)] = ixx_0 - _S233 * cx_0;
    (*region_0)[int(4)] = iyy_0 - a_5 * cy_0 * cy_0;
    (*region_0)[int(5)] = ixy_0 - _S233 * cy_0;
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
    float _S234 = a_7 * fc_2;
    float _S235 = - ay_1;
    return float4(k_8 * a_7 * fc_2, k_8 * (_S234 * r_2[int(2)] + (_S235 * r_2[int(5)] + ax_1 * r_2[int(4)])), - k_8 * (_S234 * r_2[int(1)] + (_S235 * r_2[int(3)] + ax_1 * r_2[int(5)])), 0.5f * k_8 * (_S234 * fc_2 + ay_1 * ay_1 * r_2[int(3)] + ax_1 * ax_1 * r_2[int(4)] - 2.0f * ax_1 * ay_1 * r_2[int(5)]));
}

constant array<float, int(6)> SPRING_AT_0 = { { -0.4166666567325592f, -0.25f, -0.0833333358168602f, 0.0833333358168602f, 0.25f, 0.4166666567325592f } };
float4 contact_normal_0(const JointMaterial_0 constant* mat_4, const JointBond_natural_0 thread* b_11, float crush_2, float3 d_lin_0, float3 d_ang_0)
{
    float kn_1 = (float4(b_11->stiff0_0) ).x;
    float4 _S236 = float4(b_11->geom0_0) ;
    float w0_2 = _S236.y;
    float w1_2 = _S236.z;
    float nc_sum_0;
    float m1_0;
    float m2_0;
    float energy_2;
    if(((mat_4->kind_flags_0.y) & 4U) != 0U)
    {
        float4 p_0 = no_tension_patch_0(kn_1 * (1.0f - crush_2), w0_2, w1_2, d_lin_0.z, d_ang_0.x, d_ang_0.y);
        float _S237 = p_0.y;
        float _S238 = p_0.z;
        float _S239 = p_0.w;
        nc_sum_0 = p_0.x;
        m1_0 = _S237;
        m2_0 = _S238;
        energy_2 = _S239;
    }
    else
    {
        float ki_0 = kn_1 * (1.0f - crush_2) / 36.0f;
        float _S240 = d_ang_0.x;
        float _S241 = d_ang_0.y;
        float spread_0 = abs(_S240) * 0.4166666567325592f * w1_2 + abs(_S241) * 0.4166666567325592f * w0_2;
        float _S242 = d_lin_0.z;
        float slack_0 = 9.99999997475242708e-07f * (abs(_S242) + spread_0);
        if((_S242 - spread_0) > slack_0)
        {
            nc_sum_0 = 0.0f;
            m1_0 = 0.0f;
            m2_0 = 0.0f;
            energy_2 = 0.0f;
        }
        else
        {
            if((_S242 + spread_0) < (- slack_0))
            {
                float i1_0 = 2.91666650772094727f * w0_2 * w0_2;
                float i2_0 = 2.91666650772094727f * w1_2 * w1_2;
                float _S243 = ki_0 * _S240 * i2_0;
                float _S244 = ki_0 * _S241 * i1_0;
                float _S245 = 0.5f * ki_0 * (36.0f * _S242 * _S242 + _S240 * _S240 * i2_0 + _S241 * _S241 * i1_0);
                nc_sum_0 = ki_0 * 36.0f * _S242;
                m1_0 = _S243;
                m2_0 = _S244;
                energy_2 = _S245;
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
                    float _S246 = SPRING_AT_0[i_3] * w0_2;
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
                        float di_0 = _S242 + _S240 * s2_0 - _S241 * _S246;
                        if(di_0 < 0.0f)
                        {
                            float f_7 = ki_0 * di_0;
                            float m1_2 = m1_0 + f_7 * s2_0;
                            float m2_2 = m2_0 - f_7 * _S246;
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
        float _S247 = (metal::fast::divide((r_3 * (kappa_1 - 1.0f)), (kappa_1 * (r_3 - 1.0f))));
        return min(_S247, 1.0f);
    }
    if(kappa_1 >= (0.5f * (r_3 + 1.0f)))
    {
        return 1.0f;
    }
    float _S248 = (metal::fast::divide((1.0f), (kappa_1)));
    return 1.0f - _S248;
}

__attribute__((noinline))
float2 damage_increment_0(uint kind_2, float kappa_old_0, float lambda_0, float r_4, float d_old_0, float psi_0)
{
    float _S249 = damage_law_0(kind_2, lambda_0, r_4);
    float _S250 = max(_S249, d_old_0);
    bool _S251;
    if(_S250 <= d_old_0)
    {
        _S251 = true;
    }
    else
    {
        _S251 = d_old_0 >= 1.0f;
    }
    if(_S251)
    {
        return float2(d_old_0, 0.0f);
    }
    float u0_0 = (metal::fast::divide((psi_0), (lambda_0 * lambda_0)));
    float _S252 = max(kappa_old_0, 1.0f);
    if(kind_2 == 0U)
    {
        if(r_4 > 1.0f)
        {
            float _S253 = (metal::fast::divide((u0_0 * r_4), (r_4 - 1.0f)));
            return float2(_S250, _S253 * max(min(lambda_0, r_4) - min(_S252, r_4), 0.0f));
        }
        return float2(_S250, (1.0f - d_old_0) * psi_0);
    }
    float ku_0 = 0.5f * (r_4 + 1.0f);
    float plateau_0 = u0_0 * max(min(lambda_0, ku_0) - min(_S252, ku_0), 0.0f);
    float snap_0;
    if(_S250 >= 1.0f)
    {
        snap_0 = u0_0 * ku_0;
    }
    else
    {
        snap_0 = 0.0f;
    }
    return float2(_S250, plateau_0 + snap_0);
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
    float _S254;
    if(((as_type<uint>((x_5))) & 2147483648U) != 0U)
    {
        _S254 = -1.0f;
    }
    else
    {
        _S254 = 1.0f;
    }
    return _S254;
}

float2 return_map_0(float k_9, float total_0, float plastic_1, float cap_0)
{
    float trial_0 = k_9 * (total_0 - plastic_1);
    if((abs(trial_0)) <= cap_0)
    {
        return float2(trial_0, 0.0f);
    }
    float f_8 = cap_0 * signum_0(trial_0);
    float _S255 = (metal::fast::divide((trial_0 - f_8), (k_9)));
    return float2(f_8, _S255);
}

__attribute__((noinline))
Contact_0 contact_part_0(const JointMaterial_0 constant* mat_5, const JointBond_natural_0 thread* b_12, float crush_3, float3 plastic_2, float3 d_lin_1, float3 d_ang_1)
{
    thread Contact_0 c_10;
    float3 _S256 = float3(0.0f) ;
    (&c_10)->q_lin_1 = _S256;
    (&c_10)->q_ang_1 = _S256;
    (&c_10)->energy_5 = 0.0f;
    (&c_10)->diss_1 = 0.0f;
    (&c_10)->plastic_0 = plastic_2;
    if(((mat_5->kind_flags_0.y) & 2U) == 0U)
    {
        return c_10;
    }
    float ks_0 = (float4(b_12->stiff0_0) ).y;
    float kt_0 = (float4(b_12->stiff1_0) ).x;
    float4 _S257 = contact_normal_0(mat_5, b_12, crush_3, d_lin_1, d_ang_1);
    float nc_sum_2 = _S257.x;
    float m1_3 = _S257.y;
    float m2_3 = _S257.z;
    float energy_6 = _S257.w;
    float nc_0 = - nc_sum_2;
    thread float3 p_1 = plastic_2;
    (&c_10)->q_lin_1 = float3(0.0f, 0.0f, nc_sum_2);
    (&c_10)->q_ang_1 = float3(m1_3, m2_3, 0.0f);
    float slide_cap_0 = mat_5->strength_0.w * nc_0;
    float _S258 = ks_0 * (d_lin_1.x - plastic_2.x);
    float _S259 = ks_0 * (d_lin_1.y - plastic_2.y);
    float tn_0 = (metal::fast::sqrt((_S258 * _S258 + _S259 * _S259)));
    bool _S260;
    if(tn_0 > slide_cap_0)
    {
        _S260 = tn_0 > 0.0f;
    }
    else
    {
        _S260 = false;
    }
    float diss_2;
    if(_S260)
    {
        float _S261 = (metal::fast::divide((_S258), (tn_0)));
        float _S262 = (metal::fast::divide((_S259), (tn_0)));
        float dslip_0 = (metal::fast::divide((tn_0 - slide_cap_0), (ks_0)));
        p_1.x = p_1.x + _S261 * dslip_0;
        p_1.y = p_1.y + _S262 * dslip_0;
        (&c_10)->q_lin_1.x = _S261 * slide_cap_0;
        (&c_10)->q_lin_1.y = _S262 * slide_cap_0;
        diss_2 = slide_cap_0 * dslip_0;
    }
    else
    {
        (&c_10)->q_lin_1.x = _S258;
        (&c_10)->q_lin_1.y = _S259;
        diss_2 = 0.0f;
    }
    float2 tq_0 = return_map_0(kt_0, d_ang_1.z, p_1.z, slide_cap_0 * (float4(b_12->geom1_0) ).z);
    float _S263 = tq_0.x;
    float _S264 = tq_0.y;
    float diss_3 = diss_2 + abs(_S263) * abs(_S264);
    p_1.z = p_1.z + _S264;
    (&c_10)->q_ang_1.z = _S263;
    float _S265 = (metal::fast::divide((sq_0((&c_10)->q_lin_1.x)), (ks_0)));
    float _S266 = (metal::fast::divide((sq_0((&c_10)->q_lin_1.y)), (ks_0)));
    float _S267 = _S265 + _S266;
    float _S268 = (metal::fast::divide((sq_0(_S263)), (kt_0)));
    (&c_10)->energy_5 = energy_6 + 0.5f * (_S267 + _S268);
    (&c_10)->diss_1 = diss_3;
    (&c_10)->plastic_0 = p_1;
    return c_10;
}

float3 contact_offsets_0(const JointMaterial_0 constant* mat_6, const JointBond_natural_0 thread* b_13, float crush_4, float3 plastic_3, float3 d_lin_2, float3 d_ang_2)
{
    uint _S269 = mat_6->kind_flags_0.y;
    if((_S269 & 2U) == 0U)
    {
        return plastic_3;
    }
    float4 _S270 = float4(b_13->stiff0_0) ;
    float kn_2 = _S270.x;
    float ks_1 = _S270.y;
    float kt_1 = (float4(b_13->stiff1_0) ).x;
    float4 _S271 = float4(b_13->geom0_0) ;
    float w0_3 = _S271.y;
    float w1_3 = _S271.z;
    float nc_sum_3;
    if((_S269 & 4U) != 0U)
    {
        float4 _S272 = no_tension_patch_0(kn_2 * (1.0f - crush_4), w0_3, w1_3, d_lin_2.z, d_ang_2.x, d_ang_2.y);
        nc_sum_3 = _S272.x;
    }
    else
    {
        float ki_1 = kn_2 * (1.0f - crush_4) / 36.0f;
        float _S273 = d_ang_2.x;
        float _S274 = d_ang_2.y;
        float spread_1 = abs(_S273) * 0.4166666567325592f * w1_3 + abs(_S274) * 0.4166666567325592f * w0_3;
        float _S275 = d_lin_2.z;
        float slack_1 = 9.99999997475242708e-07f * (abs(_S275) + spread_1);
        if((_S275 - spread_1) > slack_1)
        {
            nc_sum_3 = 0.0f;
        }
        else
        {
            if((_S275 + spread_1) < (- slack_1))
            {
                nc_sum_3 = ki_1 * 36.0f * _S275;
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
                    float _S276 = SPRING_AT_0[i_4] * w0_3;
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
                        float di_1 = _S275 + _S273 * (SPRING_AT_0[j_1] * w1_3) - _S274 * _S276;
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
    float _S277 = ks_1 * (d_lin_2.x - plastic_3.x);
    float _S278 = ks_1 * (d_lin_2.y - plastic_3.y);
    float tn_1 = (metal::fast::sqrt((_S277 * _S277 + _S278 * _S278)));
    bool _S279;
    if(tn_1 > slide_cap_1)
    {
        _S279 = tn_1 > 0.0f;
    }
    else
    {
        _S279 = false;
    }
    if(_S279)
    {
        float _S280 = (metal::fast::divide((_S277), (tn_1)));
        float _S281 = (metal::fast::divide((_S278), (tn_1)));
        float dslip_1 = (metal::fast::divide((tn_1 - slide_cap_1), (ks_1)));
        p_2.x = p_2.x + _S280 * dslip_1;
        p_2.y = p_2.y + _S281 * dslip_1;
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
    float _S282 = mat_7->misc_0.y;
    float _S283 = _S282 + 1.0f;
    float _S284 = (metal::fast::pow((s_3), (_S282)));
    float _S285 = (metal::fast::divide((_S283 * _S284), (mat_7->misc_0.z)));
    return _S285;
}

JointResponse_0 joint_evaluate_0(const JointMaterial_0 constant* mat_8, const JointBond_natural_0 thread* b_14, const JointState_0 thread* state_2, float3 d_lin_3, float3 d_ang_3, float dt_6, bool fracture_1)
{
    float4 _S286 = float4(b_14->stiff0_0) ;
    float kn_3 = _S286.x;
    float ks_2 = _S286.y;
    float kb1_0 = _S286.z;
    float kb2_0 = _S286.w;
    float4 _S287 = float4(b_14->stiff1_0) ;
    float kt_2 = _S287.x;
    bool has_rebar_1 = (_S287.w) != 0.0f;
    uint kind_3 = mat_8->kind_flags_0.x;
    uint flags_0 = mat_8->kind_flags_0.y;
    bool softening_0 = (flags_0 & 1U) != 0U;
    thread JointState_0 st_2 = *state_2;
    bool _S288 = connected_0(state_2, has_rebar_1);
    float3 qe_lin_0 = d_lin_3 * float3(ks_2, ks_2, kn_3);
    float3 qe_ang_0 = d_ang_3 * float3(kb1_0, kb2_0, kt_2);
    Measures_0 _S289 = stress_measures_0(b_14, qe_lin_0, qe_ang_0);
    float _S290 = max(max(_S289.tension_0, _S289.shear_0), _S289.compression_0);
    bool _S291 = dt_6 > 0.0f;
    float dif_1;
    if(_S291)
    {
        float _S292 = (metal::fast::divide((_S290 - (&st_2)->governing_stress_0), (dt_6)));
        float raw_0 = (metal::fast::divide((max(_S292, 0.0f)), (mat_8->misc_0.w)));
        float tau_2 = _S287.z;
        if((flags_0 & 16U) != 0U)
        {
            float _S293 = (metal::fast::divide((dt_6), (tau_2)));
            dif_1 = - expm1_accurate_0(- _S293);
        }
        else
        {
            float _S294 = (metal::fast::divide((dt_6), (tau_2)));
            dif_1 = min(_S294, 1.0f);
        }
        (&st_2)->strain_rate_0 = (&st_2)->strain_rate_0 + (raw_0 - (&st_2)->strain_rate_0) * dif_1;
        (&st_2)->governing_stress_0 = _S290;
    }
    if((flags_0 & 32U) != 0U)
    {
        float _S295 = dif_factor_0(mat_8, (&st_2)->strain_rate_0);
        dif_1 = _S295;
    }
    else
    {
        dif_1 = 1.0f;
    }
    float weibull_0 = (float4(b_14->geom1_0) ).w;
    float _S296 = fatigue_factor_0(mat_8, (&st_2)->life_0);
    float multiplier_1 = weibull_0 * dif_1 * _S296;
    thread Measures_0 _S297 = _S289;
    float4 _S298 = failure_indices_0(mat_8, b_14, &_S297, multiplier_1);
    float _S299 = _S298.x;
    float _S300 = _S298.y;
    (&st_2)->utilization_0 = max(max(_S299, _S300), max(_S298.z, _S298.w));
    float _S301 = d_lin_3.x;
    float _S302 = d_lin_3.y;
    float _S303 = d_ang_3.z;
    float _S304 = ks_2 * (sq_0(_S301) + sq_0(_S302)) + kb1_0 * sq_0(d_ang_3.x) + kb2_0 * sq_0(d_ang_3.y) + kt_2 * sq_0(_S303);
    float _S305 = d_lin_3.z;
    bool _S306 = _S305 > 0.0f;
    if(_S306)
    {
        dif_1 = kn_3 * sq_0(_S305);
    }
    else
    {
        dif_1 = 0.0f;
    }
    float psi_ts_0 = 0.5f * (_S304 + dif_1);
    float psi_c_0;
    if(_S305 < 0.0f)
    {
        psi_c_0 = 0.5f * kn_3 * sq_0(_S305);
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
        float4 _S307 = contact_normal_0(mat_8, b_14, state_2->crush_1, d_lin_3, d_ang_3);
        patch_0 = _S307.w + 0.5f * (ks_2 * (sq_0(_S301 - state_2->plastic_x_0) + sq_0(_S302 - state_2->plastic_y_0)) + kt_2 * sq_0(_S303 - state_2->plastic_t_0));
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
        float4 _S308 = float4(b_14->rebar0_0) ;
        rebar_0 = 0.5f * (_S308.x * sq_0(_S305 - state_2->rebar_plastic_0) + _S308.y * (sq_0(_S301 - state_2->rebar_slip0_0) + sq_0(_S302 - state_2->rebar_slip1_0)));
    }
    else
    {
        rebar_0 = 0.0f;
    }
    float _S309 = state_2->damage_0;
    float _S310 = 1.0f - state_2->damage_0;
    float _S311 = state_2->crush_1;
    float frozen_0 = _S310 * (psi_ts_0 + (1.0f - state_2->crush_1) * psi_c_0) + state_2->damage_0 * patch_0 + rebar_0;
    float diss_contact_0;
    float psi_contact_0;
    float intact_normal_0;
    float dissipated_3;
    float3 qc_lin_0;
    if(fracture_1)
    {
        bool _S312 = _S299 >= _S300;
        if(_S312)
        {
            diss_contact_0 = _S299;
        }
        else
        {
            diss_contact_0 = _S300;
        }
        uint mode_ts_0;
        if(_S312)
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
                float _S313 = (metal::fast::divide((psi_contact_0 * (float4(b_14->geom0_0) ).x * diss_contact_0 * diss_contact_0), (psi_ts_0)));
                intact_normal_0 = _S313;
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
            float _S314 = inc_0.x;
            if(_S314 > ((&st_2)->damage_0))
            {
                Contact_0 _S315 = contact_part_0(mat_8, b_14, (&st_2)->crush_1, plastic_4, d_lin_3, d_ang_3);
                float _S316 = max(inc_0.y - max(_S315.energy_5 - (1.0f - (&st_2)->crush_1) * psi_c_0, 0.0f) * (_S314 - (&st_2)->damage_0), 0.0f);
                (&st_2)->damage_0 = _S314;
                (&st_2)->mode_0 = mode_ts_0;
                dissipated_3 = _S316;
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
        if(_S309 > 0.0f)
        {
            Contact_0 _S317 = contact_part_0(mat_8, b_14, _S311, float3(state_2->plastic_x_0, state_2->plastic_y_0, state_2->plastic_t_0), d_lin_3, d_ang_3);
            qc_lin_0 = qe_ang_0 * float3(_S310)  + _S317.q_ang_1 * float3(_S309) ;
        }
        else
        {
            qc_lin_0 = qe_ang_0;
        }
        Measures_0 _S318 = stress_measures_0(b_14, float3(0.0f, 0.0f, min(qe_lin_0.z, 0.0f)), qc_lin_0);
        thread Measures_0 _S319 = _S318;
        float4 _S320 = failure_indices_0(mat_8, b_14, &_S319, multiplier_1);
        float _S321 = _S320.z;
        float _S322 = _S320.w;
        bool _S323 = _S321 >= _S322;
        if(_S323)
        {
            psi_contact_0 = _S321;
        }
        else
        {
            psi_contact_0 = _S322;
        }
        if(_S323)
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
                float _S324 = (metal::fast::divide((mat_8->energy_1.w * (float4(b_14->geom0_0) ).x * psi_contact_0 * psi_contact_0), (psi_c_0)));
                intact_normal_0 = _S324;
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
            float _S325 = inc_1.x;
            if(_S325 > ((&st_2)->crush_1))
            {
                float dissipated_4 = dissipated_3 + inc_1.y;
                (&st_2)->crush_1 = _S325;
                (&st_2)->mode_0 = mode_c_0;
                if(_S325 >= 1.0f)
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
    float3 _S326 = float3(0.0f) ;
    float3 qc_ang_0;
    if(((&st_2)->damage_0) == 0.0f)
    {
        if((flags_0 & 8U) == 0U)
        {
            float3 _S327 = contact_offsets_0(mat_8, b_14, (&st_2)->crush_1, plastic_4, d_lin_3, d_ang_3);
            (&st_2)->plastic_x_0 = _S327.x;
            (&st_2)->plastic_y_0 = _S327.y;
            (&st_2)->plastic_t_0 = _S327.z;
        }
        diss_contact_0 = 0.0f;
        qc_lin_0 = _S326;
        qc_ang_0 = _S326;
        psi_contact_0 = 0.0f;
    }
    else
    {
        Contact_0 _S328 = contact_part_0(mat_8, b_14, (&st_2)->crush_1, plastic_4, d_lin_3, d_ang_3);
        (&st_2)->plastic_x_0 = _S328.plastic_0.x;
        (&st_2)->plastic_y_0 = _S328.plastic_0.y;
        (&st_2)->plastic_t_0 = _S328.plastic_0.z;
        diss_contact_0 = _S328.diss_1;
        qc_lin_0 = _S328.q_lin_1;
        qc_ang_0 = _S328.q_ang_1;
        psi_contact_0 = _S328.energy_5;
    }
    float dissipated_6 = dissipated_3 + dmg_0 * diss_contact_0;
    if(_S306)
    {
        intact_normal_0 = kn_3 * _S305;
    }
    else
    {
        intact_normal_0 = (1.0f - (&st_2)->crush_1) * kn_3 * _S305;
    }
    float _S329 = 1.0f - dmg_0;
    float3 force_lin_2 = float3(_S329 * qe_lin_0.x + dmg_0 * qc_lin_0.x, _S329 * qe_lin_0.y + dmg_0 * qc_lin_0.y, _S329 * intact_normal_0 + dmg_0 * qc_lin_0.z);
    float3 force_ang_2 = qe_ang_0 * float3(_S329)  + qc_ang_0 * float3(dmg_0) ;
    float stored_3 = _S329 * (psi_ts_0 + (1.0f - (&st_2)->crush_1) * psi_c_0) + dmg_0 * psi_contact_0;
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
        float4 _S330 = float4(b_14->rebar0_0) ;
        float k_axial_0 = _S330.x;
        float k_dowel_0 = _S330.y;
        float yield_force_0 = _S330.z;
        float dowel_capacity_0 = _S330.w;
        float2 nr_0 = return_map_0(k_axial_0, _S305, (&st_2)->rebar_plastic_0, yield_force_0);
        float2 v1_0 = return_map_0(k_dowel_0, _S301, (&st_2)->rebar_slip0_0, dowel_capacity_0);
        float2 v2_0 = return_map_0(k_dowel_0, _S302, (&st_2)->rebar_slip1_0, dowel_capacity_0);
        float _S331 = nr_0.y;
        float _S332 = v1_0.y;
        float _S333 = v2_0.y;
        float work_0 = yield_force_0 * abs(_S331) + dowel_capacity_0 * (abs(_S332) + abs(_S333));
        (&st_2)->rebar_plastic_0 = (&st_2)->rebar_plastic_0 + _S331;
        (&st_2)->rebar_slip0_0 = (&st_2)->rebar_slip0_0 + _S332;
        (&st_2)->rebar_slip1_0 = (&st_2)->rebar_slip1_0 + _S333;
        (&st_2)->rebar_work_0 = (&st_2)->rebar_work_0 + work_0;
        float dissipated_7 = dissipated_6 + work_0;
        float _S334 = nr_0.x;
        float _S335 = (metal::fast::divide((sq_0(_S334)), (k_axial_0)));
        float _S336 = v1_0.x;
        float _S337 = v2_0.x;
        float _S338 = (metal::fast::divide((sq_0(_S336) + sq_0(_S337)), (k_dowel_0)));
        float elastic_0 = 0.5f * (_S335 + _S338);
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
            force_lin_3 = force_lin_2 + float3(_S336, _S337, _S334);
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
        changed_0 = _S291;
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
        Measures_0 _S339 = stress_measures_0(b_14, force_lin_3, force_ang_2);
        thread Measures_0 _S340 = _S339;
        float4 _S341 = failure_indices_0(mat_8, b_14, &_S340, weibull_0);
        float _S342 = life_rate_0(mat_8, max(max(_S341.x, _S341.y), _S341.z));
        (&st_2)->life_0 = max((&st_2)->life_0 - _S342 * dt_6, 0.0f);
    }
    if(((&st_2)->damage_0) != _S309)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = ((&st_2)->crush_1) != _S311;
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
    float _S343 = max(min(dissipated_3, released_0), 0.0f);
    float overshoot_1 = released_0 - _S343;
    (&st_2)->dissipated_0 = (&st_2)->dissipated_0 + _S343;
    thread JointResponse_0 resp_0;
    (&resp_0)->force_lin_1 = force_lin_3;
    (&resp_0)->force_ang_1 = force_ang_2;
    (&resp_0)->state_1 = st_2;
    (&resp_0)->dissipated_2 = _S343;
    (&resp_0)->overshoot_0 = overshoot_1;
    (&resp_0)->stored_2 = stored_4;
    if(_S288)
    {
        thread JointState_0 _S344 = st_2;
        bool _S345 = connected_0(&_S344, has_rebar_1);
        changed_0 = !_S345;
    }
    else
    {
        changed_0 = false;
    }
    (&resp_0)->disconnected_0 = changed_0;
    (&resp_0)->fatigue_before_0 = _S296;
    (&resp_0)->measures_0 = _S289;
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
    float _S346 = st_3->damage_0;
    bool compressed_0 = (d_lin_4.z) < 0.0f;
    float contact_2;
    if(compressed_0)
    {
        contact_2 = _S346;
    }
    else
    {
        contact_2 = 0.0f;
    }
    float _S347 = 1.0f - _S346;
    float _S348 = max(_S347 + contact_2, 9.99999997475242708e-07f);
    float normal_2;
    if(compressed_0)
    {
        normal_2 = max(1.0f - st_3->crush_1, 9.99999997475242708e-07f);
    }
    else
    {
        normal_2 = max(_S347, 9.99999997475242708e-07f);
    }
    *f_lin_0 = float3(_S348, _S348, normal_2);
    *f_ang_0 = float3(_S348) ;
    bool _S349;
    if(((float4(b_15->stiff1_0) ).w) != 0.0f)
    {
        _S349 = (st_3->rebar_broken_0) == 0.0f;
    }
    else
    {
        _S349 = false;
    }
    if(_S349)
    {
        float4 _S350 = float4(b_15->rebar0_0) ;
        float4 _S351 = float4(b_15->stiff0_0) ;
        float _S352 = (metal::fast::divide((_S350.x), (_S351.x)));
        (*f_lin_0).z = (*f_lin_0).z + _S352;
        float _S353 = _S350.y;
        float _S354 = _S351.y;
        float _S355 = (metal::fast::divide((_S353), (_S354)));
        (*f_lin_0).x = (*f_lin_0).x + _S355;
        float _S356 = (metal::fast::divide((_S353), (_S354)));
        (*f_lin_0).y = (*f_lin_0).y + _S356;
    }
    return;
}

float3 to_body_0(uint _S357, float3 _S358, KernelContext_0 thread* kernelContext_25)
{
    BondStatic_natural_0 device* _S359 = kernelContext_25->bonds_0+_S357;
    return (float4(_S359->t1_0) ).xyz * float3(_S358.x)  + (float4(_S359->t2_0) ).xyz * float3(_S358.y)  + (float4(_S359->normal_0) ).xyz * float3(_S358.z) ;
}

void cov_or_1(uint island_2, uint bits_4, KernelContext_0 thread* kernelContext_26)
{
    uint _S360 = atomic_fetch_or_explicit(((atomic_uint device*)(&(kernelContext_26->islands_0+island_2)->coverage_0)), bits_4, memory_order_relaxed);
    return;
}

bool bond_update_0(uint i_5, float dt_7, bool fracture_2, uint abs_step_0, bool wide_0, KernelContext_0 thread* kernelContext_27)
{
    BondStatic_natural_0 device* _S361 = kernelContext_27->bonds_0+i_5;
    uint4 _S362 = uint4(_S361->law_0.ids_0) ;
    float3 ra_1 = (float4(_S361->ra_0) ).xyz;
    float3 rb_1 = (float4(_S361->rb_0) ).xyz;
    uint _S363 = 4U * _S362.y;
    float3 ta_0 = (float4(*(kernelContext_27->state_0+(_S363 + 1U))) ).xyz;
    float3 va_0 = (float4(*(kernelContext_27->state_0+(_S363 + 2U))) ).xyz;
    float3 wa_1 = (float4(*(kernelContext_27->state_0+(_S363 + 3U))) ).xyz;
    uint _S364 = 4U * _S362.z;
    float3 tb_0 = (float4(*(kernelContext_27->state_0+(_S364 + 1U))) ).xyz;
    float3 vb_0 = (float4(*(kernelContext_27->state_0+(_S364 + 2U))) ).xyz;
    float3 wb_1 = (float4(*(kernelContext_27->state_0+(_S364 + 3U))) ).xyz;
    float3 _S365 = to_local_0(i_5, (float4(*(kernelContext_27->state_0+_S364)) ).xyz + cross(tb_0, rb_1) - ((float4(*(kernelContext_27->state_0+_S363)) ).xyz + cross(ta_0, ra_1)), kernelContext_27);
    float3 _S366 = to_local_0(i_5, tb_0 - ta_0, kernelContext_27);
    float3 _S367 = to_local_0(i_5, vb_0 + cross(wb_1, rb_1) - (va_0 + cross(wa_1, ra_1)), kernelContext_27);
    float3 _S368 = to_local_0(i_5, wb_1 - wa_1, kernelContext_27);
    JointState_0 previous_0 = (kernelContext_27->bond_dyn_0+i_5)->js_0;
    thread JointState_0 _S369 = (kernelContext_27->bond_dyn_0+i_5)->js_0;
    bool _S370 = is_damaged_0(&_S369);
    thread JointBond_natural_0 _S371 = (kernelContext_27->bonds_0+i_5)->law_0;
    thread JointState_0 _S372 = (kernelContext_27->bond_dyn_0+i_5)->js_0;
    JointResponse_0 _S373 = joint_evaluate_0(&kernelContext_27->materials_0->m_0[(uint4((kernelContext_27->bonds_0+i_5)->law_0.ids_0) ).x], &_S371, &_S372, _S365, _S366, dt_7, fracture_2);
    BondStatic_natural_0 device* _S374 = kernelContext_27->bonds_0+i_5;
    BondDyn_natural_0 device* _S375 = kernelContext_27->bond_dyn_0+i_5;
    float4 _S376 = float4((*_S375).force_lin_0) ;
    float4 _S377 = float4((*_S375).force_ang_0) ;
    float4 _S378 = float4((*_S375).sums_0) ;
    float4 _S379 = float4((*_S375).comps_0) ;
    uint4 _S380 = uint4((*_S375).events_0) ;
    thread BondDyn_0 bd_0;
    (&bd_0)->js_0 = (*_S375).js_0;
    (&bd_0)->force_lin_0 = _S376;
    (&bd_0)->force_ang_0 = _S377;
    (&bd_0)->sums_0 = _S378;
    (&bd_0)->comps_0 = _S379;
    (&bd_0)->events_0 = _S380;
    float3 ra_2 = (float4(_S374->ra_0) ).xyz;
    float3 rb_2 = (float4(_S374->rb_0) ).xyz;
    thread JointBond_natural_0 _S381 = _S374->law_0;
    thread JointState_0 _S382 = _S373.state_1;
    thread float3 f_lin_1;
    thread float3 f_ang_1;
    secant_factors_0(&_S381, &_S382, _S365, &f_lin_1, &f_ang_1);
    float3 qd_lin_0 = _S367 * (float4(_S374->c_lin_0) ).xyz * f_lin_1;
    float3 qd_ang_0 = _S368 * (float4(_S374->c_ang_0) ).xyz * f_ang_1;
    float3 q_lin_2 = _S373.force_lin_1 + qd_lin_0;
    float3 q_ang_2 = _S373.force_ang_1 + qd_ang_0;
    float damped_0 = (dot(qd_lin_0, _S367) + dot(qd_ang_0, _S368)) * dt_7;
    float3 _S383 = to_body_0(i_5, q_lin_2, kernelContext_27);
    float3 _S384 = to_body_0(i_5, q_ang_2, kernelContext_27);
    uint _S385 = 3U * i_5;
    *(kernelContext_27->scratch_0+_S385) = packed_float4(float4(_S383, max(_S373.measures_0.tension_0, _S373.measures_0.compression_0))) ;
    *(kernelContext_27->scratch_0+(_S385 + 1U)) = packed_float4(float4(_S384 + cross(ra_2, _S383), 0.0f)) ;
    *(kernelContext_27->scratch_0+(_S385 + 2U)) = packed_float4(float4(- _S384 + cross(rb_2, - _S383), 0.0f)) ;
    thread float _S386 = (&bd_0)->sums_0.x;
    thread float _S387 = (&bd_0)->comps_0.x;
    comp_add1_0(&_S386, &_S387, _S373.dissipated_2);
    (&bd_0)->comps_0.x = _S387;
    (&bd_0)->sums_0.x = _S386;
    thread float _S388 = (&bd_0)->sums_0.y;
    thread float _S389 = (&bd_0)->comps_0.y;
    comp_add1_0(&_S388, &_S389, _S373.overshoot_0);
    (&bd_0)->comps_0.y = _S389;
    (&bd_0)->sums_0.y = _S388;
    thread float _S390 = (&bd_0)->sums_0.z;
    thread float _S391 = (&bd_0)->comps_0.z;
    comp_add1_0(&_S390, &_S391, damped_0);
    (&bd_0)->comps_0.z = _S391;
    (&bd_0)->sums_0.z = _S390;
    (&bd_0)->force_lin_0 = float4(q_lin_2, _S373.stored_2);
    (&bd_0)->force_ang_0 = float4(q_ang_2, max((&bd_0)->force_ang_0.w, _S373.state_1.utilization_0));
    bool _S392;
    if(!_S370)
    {
        thread JointState_0 _S393 = _S373.state_1;
        bool _S394 = is_damaged_0(&_S393);
        _S392 = _S394;
    }
    else
    {
        _S392 = false;
    }
    if(_S392)
    {
        _S392 = ((&bd_0)->events_0.x) == 0U;
    }
    else
    {
        _S392 = false;
    }
    if(_S392)
    {
        (&bd_0)->events_0.x = abs_step_0;
        (&bd_0)->events_0.w = _S373.state_1.mode_0;
        if(wide_0)
        {
            cov_or_1((uint4((&_S381)->ids_0) ).w, 32U, kernelContext_27);
        }
        else
        {
            (*kernelContext_27->g_seen_0)[3U] = 1U;
        }
    }
    if(((&bd_0)->events_0.y) == 0U)
    {
        _S392 = (_S373.state_1.life_0) != (previous_0.life_0);
    }
    else
    {
        _S392 = false;
    }
    if(_S392)
    {
        _S392 = (_S373.fatigue_before_0) > 0.99000000953674316f;
    }
    else
    {
        _S392 = false;
    }
    if(_S392)
    {
        float _S395 = fatigue_factor_0(&kernelContext_27->materials_0->m_0[(uint4((kernelContext_27->bonds_0+i_5)->law_0.ids_0) ).x], _S373.state_1.life_0);
        _S392 = _S395 <= 0.99000000953674316f;
    }
    else
    {
        _S392 = false;
    }
    if(_S392)
    {
        (&bd_0)->events_0.y = abs_step_0;
        if(wide_0)
        {
            cov_or_1((uint4((&_S381)->ids_0) ).w, 64U, kernelContext_27);
        }
        else
        {
            (*kernelContext_27->g_seen_0)[4U] = 1U;
        }
    }
    if(_S373.disconnected_0)
    {
        (&bd_0)->events_0.z = abs_step_0;
    }
    (&bd_0)->js_0 = _S373.state_1;
    BondDyn_natural_0 device* _S396 = kernelContext_27->bond_dyn_0+i_5;
    _S396->js_0 = bd_0.js_0;
    _S396->force_lin_0 = packed_float4(bd_0.force_lin_0) ;
    _S396->force_ang_0 = packed_float4(bd_0.force_ang_0) ;
    _S396->sums_0 = packed_float4(bd_0.sums_0) ;
    _S396->comps_0 = packed_float4(bd_0.comps_0) ;
    _S396->events_0 = packed_uint4(bd_0.events_0) ;
    return _S373.disconnected_0;
}

float3 inverse_rotate_1(const Quat_0 thread* q_9, float3 v_8)
{
    thread Quat_0 c_11;
    (&c_11)->w_0 = q_9->w_0;
    (&c_11)->x_0 = - q_9->x_0;
    (&c_11)->y_0 = - q_9->y_0;
    (&c_11)->z_0 = - q_9->z_0;
    thread Quat_0 _S397 = c_11;
    float3 _S398 = rotate_0(&_S397, v_8);
    return _S398;
}

void chunk_update_0(uint c_12, const Island_0 thread* isl_4, const Rigid_0 thread* rg_3, float dt_8, bool rml_0, uint step_0, bool contact_3, float thread* work_1, float thread* work_err_0, KernelContext_0 thread* kernelContext_28)
{
    ChunkStatic_natural_0 device* _S399 = kernelContext_28->chunks_0+c_12;
    float3 _S400 = float3(0.0f) ;
    uint _S401 = kernelContext_28->index_0[c_12];
    float peak_0 = 0.0f;
    uint e_1 = _S401;
    float3 fi_0 = _S400;
    float3 mi_0 = _S400;
    for(;;)
    {
        if(e_1 < (kernelContext_28->index_0)[c_12 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_1 = kernelContext_28->index_0[e_1];
        uint _S402 = 3U * (entry_1 >> 1U);
        float4 _S403 = float4(*(kernelContext_28->scratch_0+_S402)) ;
        if((entry_1 & 1U) == 0U)
        {
            float3 mi_1 = mi_0 + (float4(*(kernelContext_28->scratch_0+(_S402 + 1U))) ).xyz;
            fi_0 = fi_0 + _S403.xyz;
            mi_0 = mi_1;
        }
        else
        {
            float3 mi_2 = mi_0 + (float4(*(kernelContext_28->scratch_0+(_S402 + 2U))) ).xyz;
            fi_0 = fi_0 + - _S403.xyz;
            mi_0 = mi_2;
        }
        float _S404 = max(peak_0, _S403.w);
        uint _S405 = e_1 + 1U;
        peak_0 = _S404;
        e_1 = _S405;
    }
    uint _S406 = 4U * c_12;
    float3 u_0 = (float4(*(kernelContext_28->state_0+_S406)) ).xyz;
    uint _S407 = _S406 + 1U;
    float3 th_1 = (float4(*(kernelContext_28->state_0+_S407)) ).xyz;
    uint _S408 = _S406 + 2U;
    float3 v_9 = (float4(*(kernelContext_28->state_0+_S408)) ).xyz;
    uint _S409 = _S406 + 3U;
    float3 w_5 = (float4(*(kernelContext_28->state_0+_S409)) ).xyz;
    float4 _S410 = float4(_S399->center_0) ;
    float mass_0 = _S410.w;
    float3 _S411 = _S410.xyz;
    float3 _S412 = isl_4->com_0.xyz;
    float3 _S413 = rotate_0(&rg_3->rot_0, _S411 + u_0 - _S412);
    thread float3 f_load_0;
    thread float3 t_load_0;
    thread float3 rep_f_1;
    thread float3 rep_m_1;
    chunk_loads_0(c_12, c_12, &rg_3->rot_0, step_0, dt_8, contact_3, dt_8, &f_load_0, &t_load_0, &rep_f_1, &rep_m_1, kernelContext_28);
    record_chunk_load_0(c_12, f_load_0, t_load_0, kernelContext_28);
    float3 _S414 = float3(mass_0) ;
    float3 f_world_0 = f_load_0 + kernelContext_28->params_0->gravity_0.xyz * _S414;
    float3 t_world_0 = t_load_0;
    float3 f_world_1;
    float3 t_world_1;
    if(rml_0)
    {
        float3 f_world_2 = f_world_0 - (rg_3->a_0 + cross(rg_3->alpha_0, _S413) + cross(rg_3->w_1, cross(rg_3->w_1, _S413))) * _S414;
        float4 _S415 = float4(_S399->inertia0_1) ;
        float4 _S416 = float4(_S399->inertia1_1) ;
        float4 _S417 = float4(_S399->inertia2_1) ;
        float3 _S418 = world_mul_0(&rg_3->rot_0, _S415, _S416, _S417, rg_3->alpha_0);
        float3 _S419 = world_mul_0(&rg_3->rot_0, _S415, _S416, _S417, rg_3->w_1);
        float3 t_world_2 = t_world_0 - (_S418 + cross(rg_3->w_1, _S419));
        f_world_1 = f_world_2;
        t_world_1 = t_world_2;
    }
    else
    {
        f_world_1 = f_world_0;
        t_world_1 = t_world_0;
    }
    float3 _S420 = inverse_rotate_1(&rg_3->rot_0, f_world_1);
    float3 _S421 = inverse_rotate_1(&rg_3->rot_0, t_world_1);
    float3 f_ext_0;
    float3 m_ext_0;
    if(rml_0)
    {
        float3 _S422 = inverse_rotate_1(&rg_3->rot_0, rg_3->w_1);
        float4 _S423 = float4(_S399->inertia0_1) ;
        float4 _S424 = float4(_S399->inertia1_1) ;
        float4 _S425 = float4(_S399->inertia2_1) ;
        float3 i_w_0 = rows_mul_0(_S423, _S424, _S425, w_5);
        float3 m_ext_1 = _S421 - (cross(_S422, i_w_0) + cross(w_5, rows_mul_0(_S423, _S424, _S425, _S422)) + cross(w_5, i_w_0));
        f_ext_0 = _S420 - cross(_S422, v_9) * float3((2.0f * mass_0)) ;
        m_ext_0 = m_ext_1;
    }
    else
    {
        f_ext_0 = _S420;
        m_ext_0 = _S421;
    }
    float3 f_9 = f_ext_0 + rep_f_1 + fi_0;
    float3 m_3 = m_ext_0 + rep_m_1 + mi_0;
    uint support_0 = (uint4(_S399->info_1) ).x;
    float3 _S426 = float3((float4(*(kernelContext_28->state_0+_S407)) ).w, (float4(*(kernelContext_28->state_0+_S408)) ).w, (float4(*(kernelContext_28->state_0+_S409)) ).w);
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
        v_10 = _S400;
        w_6 = _S400;
    }
    else
    {
        float4 _S427 = float4(_S399->scale_0) ;
        float3 w_7 = w_5 + rows_mul_0(float4(_S399->inv0_1) , float4(_S399->inv1_1) , float4(_S399->inv2_1) , m_3) * float3((dt_8 * _S427.z)) ;
        float3 _S428 = float3(dt_8) ;
        float3 th_3 = th_1 + w_7 * _S428;
        if(support_0 == 2U)
        {
            reaction_0 = - f_9;
            u_1 = u_0;
            th_2 = _S400;
        }
        else
        {
            float3 v_11 = v_9 + f_9 * float3((dt_8 * _S427.y)) ;
            float3 u_2 = u_0 + v_11 * _S428;
            reaction_0 = _S426;
            u_1 = u_2;
            th_2 = v_11;
        }
        float3 _S429 = th_2;
        th_2 = th_3;
        v_10 = _S429;
        w_6 = w_7;
    }
    *(kernelContext_28->state_0+_S406) = packed_float4(float4(u_1, peak_0)) ;
    *(kernelContext_28->state_0+_S407) = packed_float4(float4(th_2, reaction_0.x)) ;
    *(kernelContext_28->state_0+_S408) = packed_float4(float4(v_10, reaction_0.y)) ;
    *(kernelContext_28->state_0+_S409) = packed_float4(float4(w_6, reaction_0.z)) ;
    float3 _S430 = rotate_0(&rg_3->rot_0, _S411 + u_1 - _S412);
    float3 _S431 = rg_3->vel_0 + rg_3->vel_err_0 + cross(rg_3->w_1, _S430);
    float3 _S432 = rotate_0(&rg_3->rot_0, v_10);
    float3 v_world_0 = _S431 + _S432;
    float3 _S433 = rotate_0(&rg_3->rot_0, w_6);
    comp_add1_0(work_1, work_err_0, (dot(f_load_0, v_world_0) + dot(t_load_0, rg_3->w_1 + _S433)) * dt_8);
    return;
}

void comp_add_0(float3 thread* sum_1, float3 thread* err_1, float3 x_6)
{
    float3 t_10 = *sum_1 + x_6;
    float3 _S434 = abs(x_6);
    *err_1 = *err_1 + (select(x_6, *sum_1, (abs(*sum_1)) >= _S434) - t_10 + select(*sum_1, x_6, (abs(*sum_1)) >= _S434));
    *sum_1 = t_10;
    return;
}

float3 turn_difference_0(float3 omega_0, float dt_9, float3 v_12)
{
    float _S435 = length(omega_0);
    float angle_1 = _S435 * dt_9;
    if(angle_1 < 1.00000000317107685e-30f)
    {
        return float3(0.0f) ;
    }
    float3 a_8 = omega_0 / float3(_S435) ;
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
    float4 _S436 = abs(x_7);
    *err_2 = *err_2 + (select(x_7, *sum_2, (abs(*sum_2)) >= _S436) - t_11 + select(*sum_2, x_7, (abs(*sum_2)) >= _S436));
    *sum_2 = t_11;
    return;
}

void quat_accumulate_0(Quat_0 thread* hi_1, float4 thread* lo_1, float4 x_8)
{
    thread Quat_0 _S437 = *hi_1;
    float4 _S438 = quat_vec_0(&_S437);
    thread float4 sum_3 = _S438;
    comp_add4_0(&sum_3, lo_1, x_8);
    float4 t_12 = sum_3 + *lo_1;
    *lo_1 = *lo_1 - (t_12 - sum_3);
    *hi_1 = quat_of_0(t_12);
    return;
}

void turn_left_0(Quat_0 thread* hi_2, float4 thread* lo_2, float3 omega_1, float dt_10)
{
    float _S439 = length(omega_1);
    float angle_3 = _S439 * dt_10;
    if(angle_3 < 1.00000000317107685e-30f)
    {
        return;
    }
    float4 d_2 = turn_minus_one_0(omega_1 / float3(_S439) , angle_3);
    thread Quat_0 dq_0;
    (&dq_0)->x_0 = d_2.x;
    (&dq_0)->y_0 = d_2.y;
    (&dq_0)->z_0 = d_2.z;
    (&dq_0)->w_0 = d_2.w;
    thread Quat_0 _S440 = dq_0;
    thread Quat_0 _S441 = *hi_2;
    Quat_0 _S442 = quat_mul_0(&_S440, &_S441);
    thread Quat_0 _S443 = _S442;
    float4 _S444 = quat_vec_0(&_S443);
    quat_accumulate_0(hi_2, lo_2, _S444);
    return;
}

void integrate_rigid_0(const Island_0 thread* isl_5, Rigid_0 thread* rg_4, float dt_11)
{
    float3 _S445 = float3(dt_11) ;
    comp_add_0(&rg_4->l_0, &rg_4->l_err_0, rg_4->torque_0 * _S445);
    float3 l_1 = rg_4->l_0 + rg_4->l_err_0;
    comp_add_0(&rg_4->vel_0, &rg_4->vel_err_0, rg_4->a_0 * _S445);
    float3 vel_1 = rg_4->vel_0 + rg_4->vel_err_0;
    float4 _S446 = isl_5->inv0_0;
    float4 _S447 = isl_5->inv1_0;
    float4 _S448 = isl_5->inv2_0;
    thread Quat_0 _S449 = rg_4->rot_0;
    float3 _S450 = world_mul_0(&_S449, isl_5->inv0_0, isl_5->inv1_0, isl_5->inv2_0, l_1);
    float3 _S451 = vel_1 * _S445;
    float3 _S452 = isl_5->com_0.xyz;
    thread Quat_0 _S453 = rg_4->rot_0;
    float3 _S454 = rotate_0(&_S453, _S452);
    comp_add_0(&rg_4->pos_0, &rg_4->pos_err_0, _S451 - turn_difference_0(_S450, dt_11, _S454));
    turn_left_0(&rg_4->rot_0, &rg_4->rot_err_0, _S450, dt_11);
    thread Quat_0 _S455 = rg_4->rot_0;
    float3 _S456 = world_mul_0(&_S455, _S446, _S447, _S448, l_1);
    rg_4->w_1 = _S456;
    return;
}

void drift_moments_0(uint c_13, float3 thread* tu_0, float3 thread* pv_0, KernelContext_0 thread* kernelContext_29)
{
    ChunkStatic_natural_0 device* _S457 = kernelContext_29->chunks_0+c_13;
    uint _S458 = 4U * c_13;
    float3 _S459 = float3(((float4(_S457->center_0) ).w * (float4(_S457->scale_0) ).x)) ;
    *tu_0 = *tu_0 + (float4(*(kernelContext_29->state_0+_S458)) ).xyz * _S459;
    *pv_0 = *pv_0 + (float4(*(kernelContext_29->state_0+(_S458 + 2U))) ).xyz * _S459;
    return;
}

void drift_angular_0(uint c_14, float3 wcom_1, float3 tr_0, float3 dv_0, float3 thread* lu_0, float3 thread* lv_0, KernelContext_0 thread* kernelContext_30)
{
    ChunkStatic_natural_0 device* _S460 = kernelContext_30->chunks_0+c_14;
    float4 _S461 = float4(_S460->center_0) ;
    float3 r_6 = _S461.xyz - wcom_1;
    uint _S462 = 4U * c_14;
    float3 _S463 = float3(_S461.w) ;
    float4 _S464 = float4(_S460->inertia0_1) ;
    float4 _S465 = float4(_S460->inertia1_1) ;
    float4 _S466 = float4(_S460->inertia2_1) ;
    float3 _S467 = float3((float4(_S460->scale_0) ).x) ;
    *lu_0 = *lu_0 + (cross(r_6, (float4(*(kernelContext_30->state_0+_S462)) ).xyz - tr_0) * _S463 + rows_mul_0(_S464, _S465, _S466, (float4(*(kernelContext_30->state_0+(_S462 + 1U))) ).xyz)) * _S467;
    *lv_0 = *lv_0 + (cross(r_6, (float4(*(kernelContext_30->state_0+(_S462 + 2U))) ).xyz - dv_0) * _S463 + rows_mul_0(_S464, _S465, _S466, (float4(*(kernelContext_30->state_0+(_S462 + 3U))) ).xyz)) * _S467;
    return;
}

void drift_apply_0(uint c_15, float3 wcom_2, float3 tr_1, float3 phi_0, float3 dv_1, float3 dw_0, KernelContext_0 thread* kernelContext_31)
{
    float3 r_7 = (float4((kernelContext_31->chunks_0+c_15)->center_0) ).xyz - wcom_2;
    uint _S468 = 4U * c_15;
    *(kernelContext_31->state_0+_S468) = packed_float4(float4((float4(*(kernelContext_31->state_0+_S468)) ).xyz - (tr_1 + cross(phi_0, r_7)), (float4(*(kernelContext_31->state_0+_S468)) ).w)) ;
    uint _S469 = _S468 + 1U;
    *(kernelContext_31->state_0+_S469) = packed_float4(float4((float4(*(kernelContext_31->state_0+_S469)) ).xyz - phi_0, (float4(*(kernelContext_31->state_0+_S469)) ).w)) ;
    uint _S470 = _S468 + 2U;
    *(kernelContext_31->state_0+_S470) = packed_float4(float4((float4(*(kernelContext_31->state_0+_S470)) ).xyz - (dv_1 + cross(dw_0, r_7)), (float4(*(kernelContext_31->state_0+_S470)) ).w)) ;
    uint _S471 = _S468 + 3U;
    *(kernelContext_31->state_0+_S471) = packed_float4(float4((float4(*(kernelContext_31->state_0+_S471)) ).xyz - dw_0, (float4(*(kernelContext_31->state_0+_S471)) ).w)) ;
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
    thread Quat_0 _S472 = *hi_3;
    thread Quat_0 _S473 = dq_1;
    Quat_0 _S474 = quat_mul_0(&_S472, &_S473);
    thread Quat_0 _S475 = _S474;
    float4 _S476 = quat_vec_0(&_S475);
    quat_accumulate_0(hi_3, lo_3, _S476);
    return;
}

void drift_rigid_0(const Island_0 thread* isl_6, Rigid_0 thread* rg_5, float3 tr_2, float3 phi_2, float3 dv_2, float3 dw_1)
{
    float3 wcom_3 = isl_6->wcom_0.xyz;
    Quat_0 rot_2 = rg_5->rot_0;
    float3 _S477 = tr_2 - turn_difference_0(phi_2, 1.0f, wcom_3);
    thread Quat_0 _S478 = rg_5->rot_0;
    float3 _S479 = rotate_0(&_S478, _S477);
    comp_add_0(&rg_5->pos_0, &rg_5->pos_err_0, _S479);
    thread Quat_0 _S480 = rg_5->rot_0;
    float3 _S481 = inverse_rotate_1(&_S480, rg_5->w_1);
    float4 _S482 = isl_6->inertia0_0;
    float4 _S483 = isl_6->inertia1_0;
    float4 _S484 = isl_6->inertia2_0;
    float3 _S485 = cross(phi_2, rows_mul_0(isl_6->inertia0_0, isl_6->inertia1_0, isl_6->inertia2_0, _S481)) - rows_mul_0(isl_6->inertia0_0, isl_6->inertia1_0, isl_6->inertia2_0, cross(phi_2, _S481));
    thread Quat_0 _S486 = rg_5->rot_0;
    float3 _S487 = rotate_0(&_S486, _S485);
    turn_right_0(&rg_5->rot_0, &rg_5->rot_err_0, phi_2);
    float3 _S488 = dv_2 + cross(dw_1, isl_6->com_0.xyz - wcom_3);
    thread Quat_0 _S489 = rot_2;
    float3 _S490 = rotate_0(&_S489, _S488);
    comp_add_0(&rg_5->vel_0, &rg_5->vel_err_0, _S490);
    thread Quat_0 _S491 = rot_2;
    float3 _S492 = rotate_0(&_S491, dw_1);
    rg_5->w_1 = rg_5->w_1 + _S492;
    thread Quat_0 _S493 = rg_5->rot_0;
    float3 _S494 = world_mul_0(&_S493, _S482, _S483, _S484, _S492);
    comp_add_0(&rg_5->l_0, &rg_5->l_err_0, _S487 + _S494);
    return;
}

void write_probe_0(uint slot_0, uint k_10, float value_2, KernelContext_0 thread* kernelContext_32)
{
    uint at_1 = kernelContext_32->params_0->probe_base_0 * 4U + slot_0 * kernelContext_32->params_0->probe_stride_0 + k_10;
    thread float4 v_13 = float4(*(kernelContext_32->scratch_0+at_1 / 4U)) ;
    v_13[at_1 % 4U] = value_2;
    *(kernelContext_32->scratch_0+at_1 / 4U) = packed_float4(v_13) ;
    return;
}

void record_probes_0(const Island_0 thread* isl_7, const Rigid_0 thread* rg_6, uint k_11, KernelContext_0 thread* kernelContext_33)
{
    uint4 _S495 = isl_7->probes_0;
    uint at_2 = isl_7->probes_0.x;
    for(;;)
    {
        if(at_2 < (_S495.y))
        {
        }
        else
        {
            break;
        }
        uint4 info_3 = (as_type<uint4>((float4(*(kernelContext_33->loads_0+at_2)) )));
        float4 _S496 = float4(*(kernelContext_33->loads_0+(at_2 + 1U))) ;
        float4 _S497 = float4(*(kernelContext_33->loads_0+(at_2 + 2U))) ;
        float4 _S498 = float4(*(kernelContext_33->loads_0+(at_2 + 3U))) ;
        uint kind_4 = info_3.x;
        uint i_6 = info_3.y;
        float value_3;
        if(kind_4 == 0U)
        {
            float3 small_0 = (float4(*(kernelContext_33->state_0+4U * i_6)) ).xyz + float3(_S496.w, _S497.w, _S498.w);
            float3 _S499 = rg_6->pos_0 - _S497.xyz;
            float3 _S500 = rotate_0(&rg_6->rot_0, (float4((kernelContext_33->chunks_0+i_6)->center_0) ).xyz);
            float3 _S501 = _S499 + _S500;
            float3 _S502 = rg_6->pos_err_0 - _S498.xyz;
            float3 _S503 = rotate_0(&rg_6->rot_0, small_0);
            value_3 = dot(_S501 + (_S502 + _S503), _S496.xyz);
        }
        else
        {
            if(kind_4 == 1U)
            {
                uint _S504 = 4U * i_6;
                float3 _S505 = rotate_0(&rg_6->rot_0, (float4((kernelContext_33->chunks_0+i_6)->center_0) ).xyz + (float4(*(kernelContext_33->state_0+_S504)) ).xyz - isl_7->com_0.xyz);
                float3 _S506 = rg_6->vel_0 + rg_6->vel_err_0 + cross(rg_6->w_1, _S505);
                float3 _S507 = rotate_0(&rg_6->rot_0, (float4(*(kernelContext_33->state_0+(_S504 + 2U))) ).xyz);
                value_3 = dot(_S506 + _S507, _S496.xyz);
            }
            else
            {
                if(kind_4 == 2U)
                {
                    uint _S508 = 3U * i_6;
                    float3 f_10 = (float4(*(kernelContext_33->scratch_0+_S508)) ).xyz;
                    bool _S509 = (info_3.z) == 0U;
                    float3 mc_0;
                    if(_S509)
                    {
                        mc_0 = (float4(*(kernelContext_33->scratch_0+(_S508 + 1U))) ).xyz;
                    }
                    else
                    {
                        mc_0 = (float4(*(kernelContext_33->scratch_0+(_S508 + 2U))) ).xyz;
                    }
                    float3 fc_3;
                    if(_S509)
                    {
                        fc_3 = f_10;
                    }
                    else
                    {
                        fc_3 = - f_10;
                    }
                    value_3 = dot(fc_3, _S496.xyz) + dot(mc_0, _S497.xyz);
                }
                else
                {
                    uint _S510 = 4U * i_6;
                    float3 _S511 = rotate_0(&rg_6->rot_0, float3((float4(*(kernelContext_33->state_0+(_S510 + 1U))) ).w, (float4(*(kernelContext_33->state_0+(_S510 + 2U))) ).w, (float4(*(kernelContext_33->state_0+(_S510 + 3U))) ).w));
                    value_3 = dot(_S511, _S496.xyz);
                }
            }
        }
        write_probe_0(info_3.w, k_11, value_3, kernelContext_33);
        at_2 = at_2 + 4U;
    }
    return;
}

bool island_connected_narrow_0(uint tid_3, uint ct_0, const Island_0 thread* isl_8, KernelContext_0 thread* kernelContext_34)
{
    uint _S512 = isl_8->range_0.x;
    uint n_2 = group_uniform_0(isl_8->range_0.y - _S512, tid_3);
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
    uint c_16 = ct_0;
    for(;;)
    {
        if(c_16 < n_2)
        {
        }
        else
        {
            break;
        }
        (*kernelContext_34->g_label_narrow_0)[c_16] = c_16;
        c_16 = c_16 + 256U;
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
        c_16 = ct_0;
        for(;;)
        {
            if(c_16 < n_2)
            {
            }
            else
            {
                break;
            }
            uint gc_0 = _S512 + c_16;
            uint _S513 = kernelContext_34->index_0[gc_0];
            uint l_2 = (*kernelContext_34->g_label_narrow_0)[c_16];
            uint e_2 = _S513;
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
                bool _S514 = ((float4((kernelContext_34->bonds_0+bond_0)->law_0.stiff1_0) ).w) != 0.0f;
                thread JointState_0 _S515 = (kernelContext_34->bond_dyn_0+bond_0)->js_0;
                bool _S516 = connected_0(&_S515, _S514);
                if(!_S516)
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
                l_2 = min(l_2, (*kernelContext_34->g_label_narrow_0)[other_0 - _S512]);
                e_2 = e_2 + 1U;
            }
            uint _S517 = min(l_2, (*kernelContext_34->g_label_narrow_0)[l_2]);
            if(_S517 < (*kernelContext_34->g_label_narrow_0)[c_16])
            {
                (*kernelContext_34->g_label_narrow_0)[c_16] = _S517;
                *kernelContext_34->g_changed_narrow_0 = 1U;
            }
            c_16 = c_16 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool converged_1 = (group_uniform_0(*kernelContext_34->g_changed_narrow_0, tid_3)) == 0U;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(converged_1)
        {
            converged_0 = converged_1;
            break;
        }
        uint _S518 = pass_0 + 1U;
        converged_0 = converged_1;
        pass_0 = _S518;
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
    c_16 = ct_0;
    for(;;)
    {
        if(c_16 < n_2)
        {
        }
        else
        {
            break;
        }
        if(((*kernelContext_34->g_label_narrow_0)[c_16]) != 0U)
        {
            *kernelContext_34->g_changed_narrow_0 = 1U;
        }
        c_16 = c_16 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (group_uniform_0(*kernelContext_34->g_changed_narrow_0, tid_3)) == 0U;
}

void contact_split_at_0(uint at_3, KernelContext_0 thread* kernelContext_35)
{
    uint previous_1 = (uint4((kernelContext_35->islands_0+kernelContext_35->params_0->halt_index_0)->info_0) ).y;
    uint _S519;
    if(previous_1 == 0U)
    {
        _S519 = at_3;
    }
    else
    {
        _S519 = min(previous_1, at_3);
    }
    (kernelContext_35->islands_0+kernelContext_35->params_0->halt_index_0)->info_0[int(1)] = _S519;
    return;
}

uint seen_bits_0(KernelContext_0 thread* kernelContext_36)
{
    uint _S520;
    if(((*kernelContext_36->g_seen_0)[0U]) != 0U)
    {
        _S520 = 32768U;
    }
    else
    {
        _S520 = 0U;
    }
    uint _S521;
    if(((*kernelContext_36->g_seen_0)[1U]) != 0U)
    {
        _S521 = 65536U;
    }
    else
    {
        _S521 = 0U;
    }
    uint _S522 = _S520 | _S521;
    if(((*kernelContext_36->g_seen_0)[2U]) != 0U)
    {
        _S520 = 131072U;
    }
    else
    {
        _S520 = 0U;
    }
    uint _S523 = _S522 | _S520;
    if(((*kernelContext_36->g_seen_0)[3U]) != 0U)
    {
        _S520 = 32U;
    }
    else
    {
        _S520 = 0U;
    }
    uint _S524 = _S523 | _S520;
    if(((*kernelContext_36->g_seen_0)[4U]) != 0U)
    {
        _S520 = 64U;
    }
    else
    {
        _S520 = 0U;
    }
    return _S524 | _S520;
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

void travel_check_0(uint c_17, bool fused_1, KernelContext_0 thread* kernelContext_38)
{
    ChunkStatic_natural_0 device* _S525 = kernelContext_38->chunks_0+c_17;
    uint4 _S526 = uint4(_S525->cinfo_0) ;
    if((_S526.z) == 0U)
    {
        return;
    }
    uint _S527 = 4U * c_17;
    uint at_4 = kernelContext_38->params_0->cplan_base_0 + _S527;
    float4 _S528 = float4(*(kernelContext_38->scratch_0+at_4)) ;
    float4 _S529 = float4(*(kernelContext_38->scratch_0+(at_4 + 1U))) ;
    float4 _S530 = float4(*(kernelContext_38->scratch_0+(at_4 + 2U))) ;
    float4 _S531 = float4(*(kernelContext_38->scratch_0+(at_4 + 3U))) ;
    WorldPoint_0 _S532 = chunk_world_0(c_17, kernelContext_38);
    float3 moved_0 = _S532.hi_0 - _S528.xyz + (_S532.lo_0 - _S529.xyz) + _S532.rel_0;
    uint _S533 = (uint4(_S525->info_1) ).y;
    float3 _S534 = float3((kernelContext_38->params_0->t_hi_0 - _S531.w + (kernelContext_38->params_0->t_lo_0 - _S529.w) + float((uint4((kernelContext_38->islands_0+_S533)->info_0) ).w - kernelContext_38->params_0->step_start_0) * kernelContext_38->params_0->dt_0)) ;
    float3 _S535 = _S531.xyz * _S534;
    uint _S536 = _S526.w;
    float3 _S537;
    if((_S536 & 2U) != 0U)
    {
        _S537 = float3(0.5f)  * kernelContext_38->params_0->gravity_0.xyz * _S534 * _S534;
    }
    else
    {
        _S537 = float3(0.0f) ;
    }
    float travel_0 = length(moved_0 - (_S535 + _S537));
    float travel_1;
    if((_S536 & 1U) != 0U)
    {
        float3 th_4 = (float4(*(kernelContext_38->state_0+(_S527 + 1U))) ).xyz;
        Quat_0 _S538 = from_axis_angle_0(th_4, length(th_4));
        thread Quat_0 _S539 = quat_of_0(float4((kernelContext_38->islands_0+_S533)->rotation_0) );
        thread Quat_0 _S540 = _S538;
        Quat_0 _S541 = quat_mul_0(&_S539, &_S540);
        thread Quat_0 _S542 = quat_of_0(float4(- _S530.xyz, _S530.w));
        thread Quat_0 _S543 = _S541;
        Quat_0 _S544 = quat_mul_0(&_S542, &_S543);
        travel_1 = travel_0 + 2.0f * atan2(length(float3(_S544.x_0, _S544.y_0, _S544.z_0)), abs(_S544.w_0)) * (float4(_S525->half_0) ).w;
    }
    else
    {
        travel_1 = travel_0;
    }
    if(travel_1 > (_S528.w))
    {
        travel_stop_0((uint4((kernelContext_38->islands_0+_S533)->info_0) ).w + 1U, fused_1, kernelContext_38);
    }
    return;
}

void island_frame_at_0(uint gi_0, uint tid_4, KernelContext_0 thread* kernelContext_39)
{
    uint c_18;
    bool woke_0;
    uint _S545;
    uint ct_1;
    if(tid_4 < 256U)
    {
        ct_1 = tid_4;
    }
    else
    {
        ct_1 = 1073741824U;
    }
    Island_natural_0 device* _S546 = kernelContext_39->islands_0+gi_0;
    uint4 _S547 = uint4((*_S546).info_0) ;
    float4 _S548 = float4((*_S546).com_0) ;
    float4 _S549 = float4((*_S546).inertia0_0) ;
    float4 _S550 = float4((*_S546).inertia1_0) ;
    float4 _S551 = float4((*_S546).inertia2_0) ;
    float4 _S552 = float4((*_S546).inv0_0) ;
    float4 _S553 = float4((*_S546).inv1_0) ;
    float4 _S554 = float4((*_S546).inv2_0) ;
    float4 _S555 = float4((*_S546).wcom_0) ;
    float4 _S556 = float4((*_S546).winv0_0) ;
    float4 _S557 = float4((*_S546).winv1_0) ;
    float4 _S558 = float4((*_S546).winv2_0) ;
    float4 _S559 = float4((*_S546).rotation_0) ;
    float4 _S560 = float4((*_S546).position_0) ;
    float4 _S561 = float4((*_S546).position_err_0) ;
    float4 _S562 = float4((*_S546).velocity_0) ;
    float4 _S563 = float4((*_S546).velocity_err_0) ;
    float4 _S564 = float4((*_S546).angular_velocity_0) ;
    uint4 _S565 = uint4((*_S546).probes_0) ;
    float4 _S566 = float4((*_S546).energy_0) ;
    float4 _S567 = float4((*_S546).rotation_err_0) ;
    float4 _S568 = float4((*_S546).momentum_0) ;
    float4 _S569 = float4((*_S546).momentum_err_0) ;
    thread Island_0 isl_9;
    (&isl_9)->range_0 = uint4((*_S546).range_0) ;
    (&isl_9)->info_0 = _S547;
    (&isl_9)->com_0 = _S548;
    (&isl_9)->inertia0_0 = _S549;
    (&isl_9)->inertia1_0 = _S550;
    (&isl_9)->inertia2_0 = _S551;
    (&isl_9)->inv0_0 = _S552;
    (&isl_9)->inv1_0 = _S553;
    (&isl_9)->inv2_0 = _S554;
    (&isl_9)->wcom_0 = _S555;
    (&isl_9)->winv0_0 = _S556;
    (&isl_9)->winv1_0 = _S557;
    (&isl_9)->winv2_0 = _S558;
    (&isl_9)->rotation_0 = _S559;
    (&isl_9)->position_0 = _S560;
    (&isl_9)->position_err_0 = _S561;
    (&isl_9)->velocity_0 = _S562;
    (&isl_9)->velocity_err_0 = _S563;
    (&isl_9)->angular_velocity_0 = _S564;
    (&isl_9)->done_0 = (*_S546).done_0;
    (&isl_9)->coverage_0 = (*_S546).coverage_0;
    (&isl_9)->wide_groups_0 = (*_S546).wide_groups_0;
    (&isl_9)->done_pad_0 = (*_S546).done_pad_0;
    (&isl_9)->probes_0 = _S565;
    (&isl_9)->energy_0 = _S566;
    (&isl_9)->rotation_err_0 = _S567;
    (&isl_9)->momentum_0 = _S568;
    (&isl_9)->momentum_err_0 = _S569;
    uint4 info_4 = group_uniform4_0((&isl_9)->info_0, tid_4);
    uint4 _S570 = group_uniform4_0((&isl_9)->range_0, tid_4);
    uint _S571 = info_4.x;
    bool driven_0 = (_S571 & 2U) != 0U;
    bool _S572 = !((_S571 & 1U) != 0U);
    bool _S573;
    if(_S572)
    {
        _S573 = !driven_0;
    }
    else
    {
        _S573 = false;
    }
    bool contact_island_0 = (_S571 & 4U) != 0U;
    bool _S574 = (_S571 & 16U) != 0U;
    seen_clear_0(tid_4, kernelContext_39);
    bool _S575 = tid_4 == 0U;
    uint cov_0;
    bool settled_0;
    if(_S575)
    {
        if(contact_island_0 != ((kernelContext_39->params_0->contact_mode_0) == 1U))
        {
            settled_0 = true;
        }
        else
        {
            settled_0 = (_S571 & 8U) != 0U;
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
            thread Island_0 _S576 = isl_9;
            bool _S577 = contact_stopped_0(&_S576, kernelContext_39);
            settled_0 = _S577;
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
            uint _S578 = info_4.w;
            stop_seen_0(kernelContext_39->params_0->halt_index_0, _S578 + 1U, kernelContext_39);
            uint _S579 = (uint4(sentinel_0.info_0) ).y;
            if(_S579 != 0U)
            {
                settled_0 = _S579 <= _S578;
            }
            else
            {
                settled_0 = false;
            }
            int _S580;
            if(settled_0)
            {
                _S580 = int(0);
            }
            else
            {
                _S580 = int(1);
            }
            *kernelContext_39->g_sums_0 = uint(_S580);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(((info_4.z) & 1U) != 0U)
    {
        settled_0 = true;
    }
    else
    {
        settled_0 = (group_uniform_0(*kernelContext_39->g_run_0, tid_4)) == 0U;
    }
    if(settled_0)
    {
        cov_0 = 0U;
    }
    else
    {
        cov_0 = min(info_4.y, kernelContext_39->params_0->max_steps_0);
    }
    float _S581 = kernelContext_39->params_0->dt_0;
    bool _S582 = (kernelContext_39->params_0->fracture_0) != 0U;
    bool _S583 = (kernelContext_39->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S584 = isl_9;
    Rigid_0 _S585 = rigid_of_0(&_S584);
    thread Rigid_0 rg_7 = _S585;
    thread float work_2 = 0.0f;
    thread float work_err_1 = 0.0f;
    settled_0 = _S574;
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
        uint abs_step_1 = info_4.w + done_1 + 1U;
        uint k_12 = abs_step_1 - 1U - kernelContext_39->params_0->step_start_0;
        bool quiet_0 = (_S571 & 32U) != 0U;
        bool _S586;
        if(contact_island_0)
        {
            _S586 = (group_uniform_0(*kernelContext_39->g_sums_0, tid_4)) != 0U;
        }
        else
        {
            _S586 = false;
        }
        if(_S586)
        {
            uint _S587 = (uint4((kernelContext_39->chunks_0+(_S570.y - 1U))->cinfo_0) ).y;
            c_18 = (uint4((kernelContext_39->chunks_0+_S570.x)->cinfo_0) ).x + ct_1;
            for(;;)
            {
                if(c_18 < _S587)
                {
                }
                else
                {
                    break;
                }
                contact_segment_0(c_18, kernelContext_39);
                c_18 = c_18 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        bool _S588;
        if(settled_0)
        {
            _S588 = !quiet_0;
        }
        else
        {
            _S588 = false;
        }
        uint c_19;
        bool _S589;
        bool explicit_step_0;
        if(_S588)
        {
            float3 _S590 = float3(0.0f) ;
            thread float3 norm_0 = _S590;
            thread float3 unused0_0 = _S590;
            c_18 = _S570.x + ct_1;
            for(;;)
            {
                if(c_18 < (_S570.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S591 = (&rg_7)->rot_0;
                float _S592 = settled_chunk_load_0(c_18, &_S591, k_12, _S581, contact_island_0, kernelContext_39);
                norm_0.x = norm_0.x + _S592;
                c_18 = c_18 + 256U;
            }
            group_sum3_0(tid_4, &norm_0, &unused0_0, kernelContext_39);
            if((kernelContext_39->params_0->solve_mode_0) == 1U)
            {
                _S589 = (abs(norm_0.x - (&isl_9)->energy_0.z)) > ((&isl_9)->energy_0.w);
            }
            else
            {
                _S589 = false;
            }
            if(_S589)
            {
                c_19 = 1U;
            }
            else
            {
                c_19 = 0U;
            }
            if((group_uniform_0(c_19, tid_4)) != 0U)
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
        if(_S572)
        {
            _S589 = !quiet_0;
        }
        else
        {
            _S589 = false;
        }
        if(_S589)
        {
            float3 _S593 = float3(0.0f) ;
            thread float3 f_11 = _S593;
            thread float3 t_13 = _S593;
            c_18 = _S570.x + ct_1;
            for(;;)
            {
                if(c_18 < (_S570.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S594 = isl_9;
                thread Rigid_0 _S595 = rg_7;
                net_load_0(c_18, &_S594, &_S595, k_12, _S581, contact_island_0, &f_11, &t_13, kernelContext_39);
                c_18 = c_18 + 256U;
            }
            group_sum3_0(tid_4, &f_11, &t_13, kernelContext_39);
            thread Island_0 _S596 = isl_9;
            rigid_acceleration_0(&_S596, &rg_7, f_11, t_13);
        }
        if(!settled_0)
        {
            explicit_step_0 = !quiet_0;
        }
        else
        {
            explicit_step_0 = false;
        }
        bool _S597;
        if(explicit_step_0)
        {
            if(woke_0)
            {
                c_18 = _S570.z + tid_4;
                for(;;)
                {
                    if(c_18 < (_S570.w))
                    {
                    }
                    else
                    {
                        break;
                    }
                    bool _S598 = bond_update_0(c_18, _S581, _S582, abs_step_1, false, kernelContext_39);
                    if(_S598)
                    {
                        *kernelContext_39->g_halt_0 = 1U;
                    }
                    c_18 = c_18 + 256U;
                }
            }
            else
            {
                if(_S575)
                {
                    _S597 = (((uint4((kernelContext_39->islands_0+gi_0)->info_0) ).z) & 2U) != 0U;
                }
                else
                {
                    _S597 = false;
                }
                if(_S597)
                {
                    *kernelContext_39->g_halt_0 = 1U;
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            c_18 = _S570.x + ct_1;
            for(;;)
            {
                if(c_18 < (_S570.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S599 = isl_9;
                thread Rigid_0 _S600 = rg_7;
                chunk_update_0(c_18, &_S599, &_S600, _S581, _S583, k_12, contact_island_0, &work_2, &work_err_1, kernelContext_39);
                c_18 = c_18 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        if(_S573)
        {
            _S597 = !quiet_0;
        }
        else
        {
            _S597 = false;
        }
        if(_S597)
        {
            thread Island_0 _S601 = isl_9;
            integrate_rigid_0(&_S601, &rg_7, _S581);
        }
        bool _S602;
        if(_S572)
        {
            _S602 = explicit_step_0;
        }
        else
        {
            _S602 = false;
        }
        if(_S602)
        {
            float3 _S603 = (&isl_9)->wcom_0.xyz;
            float3 _S604 = float3(0.0f) ;
            thread float3 tu_1 = _S604;
            thread float3 pv_1 = _S604;
            uint _S605 = _S570.x + ct_1;
            c_18 = _S605;
            for(;;)
            {
                uint _S606 = _S570.y;
                _S545 = _S606;
                if(c_18 < _S606)
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_18, &tu_1, &pv_1, kernelContext_39);
                c_18 = c_18 + 256U;
            }
            group_sum3_0(tid_4, &tu_1, &pv_1, kernelContext_39);
            float3 tr_3 = tu_1 / float3((&isl_9)->wcom_0.w) ;
            float3 dv_3 = pv_1 / float3((&isl_9)->wcom_0.w) ;
            thread float3 lu_1 = _S604;
            thread float3 lv_1 = _S604;
            c_19 = _S605;
            for(;;)
            {
                if(c_19 < _S545)
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_19, _S603, tr_3, dv_3, &lu_1, &lv_1, kernelContext_39);
                c_19 = c_19 + 256U;
            }
            group_sum3_0(tid_4, &lu_1, &lv_1, kernelContext_39);
            float3 phi_3 = rows_mul_0((&isl_9)->winv0_0, (&isl_9)->winv1_0, (&isl_9)->winv2_0, lu_1);
            float3 dw_2 = rows_mul_0((&isl_9)->winv0_0, (&isl_9)->winv1_0, (&isl_9)->winv2_0, lv_1);
            uint c_20 = _S605;
            for(;;)
            {
                if(c_20 < _S545)
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_20, _S603, tr_3, phi_3, dv_3, dw_2, kernelContext_39);
                c_20 = c_20 + 256U;
            }
            if(!driven_0)
            {
                thread Island_0 _S607 = isl_9;
                drift_rigid_0(&_S607, &rg_7, tr_3, phi_3, dv_3, dw_2);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        bool _S608;
        if(_S575)
        {
            _S608 = ((&isl_9)->probes_0.y) > ((&isl_9)->probes_0.x);
        }
        else
        {
            _S608 = false;
        }
        if(_S608)
        {
            thread Island_0 _S609 = isl_9;
            thread Rigid_0 _S610 = rg_7;
            record_probes_0(&_S609, &_S610, k_12, kernelContext_39);
        }
        uint _S611 = done_1 + 1U;
        bool _S612;
        if((group_uniform_0(*kernelContext_39->g_halt_0, tid_4)) != 0U)
        {
            thread Island_0 _S613 = isl_9;
            bool _S614 = island_connected_narrow_0(tid_4, ct_1, &_S613, kernelContext_39);
            _S612 = _S614;
        }
        else
        {
            _S612 = false;
        }
        bool resumed_1;
        if(_S612)
        {
            if(_S575)
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
        if((group_uniform_0(*kernelContext_39->g_halt_0, tid_4)) != 0U)
        {
            done_1 = _S611;
            resumed_0 = resumed_1;
            break;
        }
        woke_1 = woke_0;
        done_1 = _S611;
        resumed_0 = resumed_1;
    }
    thread float3 wsum_0 = float3(work_2, work_err_1, 0.0f);
    thread float3 unused_0 = float3(0.0f) ;
    group_sum3_0(tid_4, &wsum_0, &unused_0, kernelContext_39);
    if(_S575)
    {
        thread Quat_0 _S615 = (&rg_7)->rot_0;
        float4 _S616 = quat_vec_0(&_S615);
        (&isl_9)->rotation_0 = _S616;
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
            _S573 = contact_island_0;
        }
        else
        {
            _S573 = false;
        }
        if(_S573)
        {
            contact_split_at_0((&isl_9)->info_0.w + done_1, kernelContext_39);
        }
        float _S617 = wsum_0.x;
        thread float _S618 = (&isl_9)->energy_0.x;
        thread float _S619 = (&isl_9)->energy_0.y;
        comp_add1_0(&_S618, &_S619, _S617);
        (&isl_9)->energy_0.x = _S618;
        (&isl_9)->energy_0.y = _S619 + wsum_0.y;
        (&isl_9)->info_0.w = (&isl_9)->info_0.w + done_1;
        if((((&isl_9)->info_0.x) & 8U) == 0U)
        {
            (&isl_9)->info_0.z = ((&isl_9)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_39->g_halt_0) != 0U)
        {
            (&isl_9)->info_0.z = ((&isl_9)->info_0.z) | 1U;
        }
        uint _S620 = seen_bits_0(kernelContext_39);
        if((*kernelContext_39->g_halt_0) != 0U)
        {
            cov_0 = 128U;
        }
        else
        {
            cov_0 = 0U;
        }
        uint _S621 = _S620 | cov_0;
        if(woke_0)
        {
            cov_0 = 2048U;
        }
        else
        {
            cov_0 = 0U;
        }
        uint _S622 = _S621 | cov_0;
        if(resumed_0)
        {
            cov_0 = 2097280U;
        }
        else
        {
            cov_0 = 0U;
        }
        uint cov_1 = _S622 | cov_0;
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
                _S573 = !quiet_1;
            }
            else
            {
                _S573 = false;
            }
            if(_S573)
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
        Island_natural_0 device* _S623 = kernelContext_39->islands_0+gi_0;
        _S623->range_0 = packed_uint4(isl_9.range_0) ;
        _S623->info_0 = packed_uint4(isl_9.info_0) ;
        _S623->com_0 = packed_float4(isl_9.com_0) ;
        _S623->inertia0_0 = packed_float4(isl_9.inertia0_0) ;
        _S623->inertia1_0 = packed_float4(isl_9.inertia1_0) ;
        _S623->inertia2_0 = packed_float4(isl_9.inertia2_0) ;
        _S623->inv0_0 = packed_float4(isl_9.inv0_0) ;
        _S623->inv1_0 = packed_float4(isl_9.inv1_0) ;
        _S623->inv2_0 = packed_float4(isl_9.inv2_0) ;
        _S623->wcom_0 = packed_float4(isl_9.wcom_0) ;
        _S623->winv0_0 = packed_float4(isl_9.winv0_0) ;
        _S623->winv1_0 = packed_float4(isl_9.winv1_0) ;
        _S623->winv2_0 = packed_float4(isl_9.winv2_0) ;
        _S623->rotation_0 = packed_float4(isl_9.rotation_0) ;
        _S623->position_0 = packed_float4(isl_9.position_0) ;
        _S623->position_err_0 = packed_float4(isl_9.position_err_0) ;
        _S623->velocity_0 = packed_float4(isl_9.velocity_0) ;
        _S623->velocity_err_0 = packed_float4(isl_9.velocity_err_0) ;
        _S623->angular_velocity_0 = packed_float4(isl_9.angular_velocity_0) ;
        _S623->done_0 = isl_9.done_0;
        _S623->coverage_0 = isl_9.coverage_0;
        _S623->wide_groups_0 = isl_9.wide_groups_0;
        _S623->done_pad_0 = isl_9.done_pad_0;
        _S623->probes_0 = packed_uint4(isl_9.probes_0) ;
        _S623->energy_0 = packed_float4(isl_9.energy_0) ;
        _S623->rotation_err_0 = packed_float4(isl_9.rotation_err_0) ;
        _S623->momentum_0 = packed_float4(isl_9.momentum_0) ;
        _S623->momentum_err_0 = packed_float4(isl_9.momentum_err_0) ;
    }
    if(contact_island_0)
    {
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if(done_1 > 0U)
        {
            c_18 = (&isl_9)->range_0.x + ct_1;
            for(;;)
            {
                if(c_18 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                travel_check_0(c_18, true, kernelContext_39);
                c_18 = c_18 + 256U;
            }
        }
    }
    return;
}

void island_frame_at_1(uint gi_1, uint tid_5, KernelContext_0 thread* kernelContext_40)
{
    uint c_21;
    bool woke_2;
    uint _S624;
    uint ct_2;
    if(tid_5 < 256U)
    {
        ct_2 = tid_5;
    }
    else
    {
        ct_2 = 1073741824U;
    }
    Island_natural_0 device* _S625 = kernelContext_40->islands_0+gi_1;
    uint4 _S626 = uint4((*_S625).info_0) ;
    float4 _S627 = float4((*_S625).com_0) ;
    float4 _S628 = float4((*_S625).inertia0_0) ;
    float4 _S629 = float4((*_S625).inertia1_0) ;
    float4 _S630 = float4((*_S625).inertia2_0) ;
    float4 _S631 = float4((*_S625).inv0_0) ;
    float4 _S632 = float4((*_S625).inv1_0) ;
    float4 _S633 = float4((*_S625).inv2_0) ;
    float4 _S634 = float4((*_S625).wcom_0) ;
    float4 _S635 = float4((*_S625).winv0_0) ;
    float4 _S636 = float4((*_S625).winv1_0) ;
    float4 _S637 = float4((*_S625).winv2_0) ;
    float4 _S638 = float4((*_S625).rotation_0) ;
    float4 _S639 = float4((*_S625).position_0) ;
    float4 _S640 = float4((*_S625).position_err_0) ;
    float4 _S641 = float4((*_S625).velocity_0) ;
    float4 _S642 = float4((*_S625).velocity_err_0) ;
    float4 _S643 = float4((*_S625).angular_velocity_0) ;
    uint4 _S644 = uint4((*_S625).probes_0) ;
    float4 _S645 = float4((*_S625).energy_0) ;
    float4 _S646 = float4((*_S625).rotation_err_0) ;
    float4 _S647 = float4((*_S625).momentum_0) ;
    float4 _S648 = float4((*_S625).momentum_err_0) ;
    thread Island_0 isl_10;
    (&isl_10)->range_0 = uint4((*_S625).range_0) ;
    (&isl_10)->info_0 = _S626;
    (&isl_10)->com_0 = _S627;
    (&isl_10)->inertia0_0 = _S628;
    (&isl_10)->inertia1_0 = _S629;
    (&isl_10)->inertia2_0 = _S630;
    (&isl_10)->inv0_0 = _S631;
    (&isl_10)->inv1_0 = _S632;
    (&isl_10)->inv2_0 = _S633;
    (&isl_10)->wcom_0 = _S634;
    (&isl_10)->winv0_0 = _S635;
    (&isl_10)->winv1_0 = _S636;
    (&isl_10)->winv2_0 = _S637;
    (&isl_10)->rotation_0 = _S638;
    (&isl_10)->position_0 = _S639;
    (&isl_10)->position_err_0 = _S640;
    (&isl_10)->velocity_0 = _S641;
    (&isl_10)->velocity_err_0 = _S642;
    (&isl_10)->angular_velocity_0 = _S643;
    (&isl_10)->done_0 = (*_S625).done_0;
    (&isl_10)->coverage_0 = (*_S625).coverage_0;
    (&isl_10)->wide_groups_0 = (*_S625).wide_groups_0;
    (&isl_10)->done_pad_0 = (*_S625).done_pad_0;
    (&isl_10)->probes_0 = _S644;
    (&isl_10)->energy_0 = _S645;
    (&isl_10)->rotation_err_0 = _S646;
    (&isl_10)->momentum_0 = _S647;
    (&isl_10)->momentum_err_0 = _S648;
    uint4 info_5 = group_uniform4_0((&isl_10)->info_0, tid_5);
    uint4 _S649 = group_uniform4_0((&isl_10)->range_0, tid_5);
    uint _S650 = info_5.x;
    bool driven_1 = (_S650 & 2U) != 0U;
    bool _S651 = !((_S650 & 1U) != 0U);
    bool _S652;
    if(_S651)
    {
        _S652 = !driven_1;
    }
    else
    {
        _S652 = false;
    }
    bool contact_island_1 = (_S650 & 4U) != 0U;
    bool _S653 = (_S650 & 16U) != 0U;
    seen_clear_0(tid_5, kernelContext_40);
    bool _S654 = tid_5 == 0U;
    uint cov_5;
    bool settled_1;
    if(_S654)
    {
        if(contact_island_1 != ((kernelContext_40->params_0->contact_mode_0) == 1U))
        {
            settled_1 = true;
        }
        else
        {
            settled_1 = (_S650 & 8U) != 0U;
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
            thread Island_0 _S655 = isl_10;
            bool _S656 = contact_stopped_0(&_S655, kernelContext_40);
            settled_1 = _S656;
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
        if(contact_island_1)
        {
            Island_natural_0 sentinel_1 = *(kernelContext_40->islands_0+kernelContext_40->params_0->halt_index_0);
            uint _S657 = info_5.w;
            stop_seen_0(kernelContext_40->params_0->halt_index_0, _S657 + 1U, kernelContext_40);
            uint _S658 = (uint4(sentinel_1.info_0) ).y;
            if(_S658 != 0U)
            {
                settled_1 = _S658 <= _S657;
            }
            else
            {
                settled_1 = false;
            }
            int _S659;
            if(settled_1)
            {
                _S659 = int(0);
            }
            else
            {
                _S659 = int(1);
            }
            *kernelContext_40->g_sums_0 = uint(_S659);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(((info_5.z) & 1U) != 0U)
    {
        settled_1 = true;
    }
    else
    {
        settled_1 = (group_uniform_0(*kernelContext_40->g_run_0, tid_5)) == 0U;
    }
    if(settled_1)
    {
        cov_5 = 0U;
    }
    else
    {
        cov_5 = min(info_5.y, kernelContext_40->params_0->max_steps_0);
    }
    float _S660 = kernelContext_40->params_0->dt_0;
    bool _S661 = (kernelContext_40->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S662 = isl_10;
    Rigid_0 _S663 = rigid_of_0(&_S662);
    thread Rigid_0 rg_8 = _S663;
    thread float work_3 = 0.0f;
    thread float work_err_2 = 0.0f;
    settled_1 = _S653;
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
        uint k_13 = info_5.w + done_2 + 1U - 1U - kernelContext_40->params_0->step_start_0;
        bool quiet_2 = (_S650 & 32U) != 0U;
        bool _S664;
        if(contact_island_1)
        {
            _S664 = (group_uniform_0(*kernelContext_40->g_sums_0, tid_5)) != 0U;
        }
        else
        {
            _S664 = false;
        }
        if(_S664)
        {
            uint _S665 = (uint4((kernelContext_40->chunks_0+(_S649.y - 1U))->cinfo_0) ).y;
            c_21 = (uint4((kernelContext_40->chunks_0+_S649.x)->cinfo_0) ).x + ct_2;
            for(;;)
            {
                if(c_21 < _S665)
                {
                }
                else
                {
                    break;
                }
                contact_segment_0(c_21, kernelContext_40);
                c_21 = c_21 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        bool _S666;
        if(settled_1)
        {
            _S666 = !quiet_2;
        }
        else
        {
            _S666 = false;
        }
        uint c_22;
        bool _S667;
        bool explicit_step_1;
        if(_S666)
        {
            float3 _S668 = float3(0.0f) ;
            thread float3 norm_1 = _S668;
            thread float3 unused0_1 = _S668;
            c_21 = _S649.x + ct_2;
            for(;;)
            {
                if(c_21 < (_S649.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S669 = (&rg_8)->rot_0;
                float _S670 = settled_chunk_load_0(c_21, &_S669, k_13, _S660, contact_island_1, kernelContext_40);
                norm_1.x = norm_1.x + _S670;
                c_21 = c_21 + 256U;
            }
            group_sum3_0(tid_5, &norm_1, &unused0_1, kernelContext_40);
            if((kernelContext_40->params_0->solve_mode_0) == 1U)
            {
                _S667 = (abs(norm_1.x - (&isl_10)->energy_0.z)) > ((&isl_10)->energy_0.w);
            }
            else
            {
                _S667 = false;
            }
            if(_S667)
            {
                c_22 = 1U;
            }
            else
            {
                c_22 = 0U;
            }
            if((group_uniform_0(c_22, tid_5)) != 0U)
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
        if(_S651)
        {
            _S667 = !quiet_2;
        }
        else
        {
            _S667 = false;
        }
        if(_S667)
        {
            float3 _S671 = float3(0.0f) ;
            thread float3 f_12 = _S671;
            thread float3 t_14 = _S671;
            c_21 = _S649.x + ct_2;
            for(;;)
            {
                if(c_21 < (_S649.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S672 = isl_10;
                thread Rigid_0 _S673 = rg_8;
                net_load_0(c_21, &_S672, &_S673, k_13, _S660, contact_island_1, &f_12, &t_14, kernelContext_40);
                c_21 = c_21 + 256U;
            }
            group_sum3_0(tid_5, &f_12, &t_14, kernelContext_40);
            thread Island_0 _S674 = isl_10;
            rigid_acceleration_0(&_S674, &rg_8, f_12, t_14);
        }
        if(!settled_1)
        {
            explicit_step_1 = !quiet_2;
        }
        else
        {
            explicit_step_1 = false;
        }
        bool _S675;
        if(explicit_step_1)
        {
            if(_S654)
            {
                _S675 = (((uint4((kernelContext_40->islands_0+gi_1)->info_0) ).z) & 2U) != 0U;
            }
            else
            {
                _S675 = false;
            }
            if(_S675)
            {
                *kernelContext_40->g_halt_0 = 1U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            c_21 = _S649.x + ct_2;
            for(;;)
            {
                if(c_21 < (_S649.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S676 = isl_10;
                thread Rigid_0 _S677 = rg_8;
                chunk_update_0(c_21, &_S676, &_S677, _S660, _S661, k_13, contact_island_1, &work_3, &work_err_2, kernelContext_40);
                c_21 = c_21 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        if(_S652)
        {
            _S675 = !quiet_2;
        }
        else
        {
            _S675 = false;
        }
        if(_S675)
        {
            thread Island_0 _S678 = isl_10;
            integrate_rigid_0(&_S678, &rg_8, _S660);
        }
        bool _S679;
        if(_S651)
        {
            _S679 = explicit_step_1;
        }
        else
        {
            _S679 = false;
        }
        if(_S679)
        {
            float3 _S680 = (&isl_10)->wcom_0.xyz;
            float3 _S681 = float3(0.0f) ;
            thread float3 tu_2 = _S681;
            thread float3 pv_2 = _S681;
            uint _S682 = _S649.x + ct_2;
            c_21 = _S682;
            for(;;)
            {
                uint _S683 = _S649.y;
                _S624 = _S683;
                if(c_21 < _S683)
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_21, &tu_2, &pv_2, kernelContext_40);
                c_21 = c_21 + 256U;
            }
            group_sum3_0(tid_5, &tu_2, &pv_2, kernelContext_40);
            float3 tr_4 = tu_2 / float3((&isl_10)->wcom_0.w) ;
            float3 dv_4 = pv_2 / float3((&isl_10)->wcom_0.w) ;
            thread float3 lu_2 = _S681;
            thread float3 lv_2 = _S681;
            c_22 = _S682;
            for(;;)
            {
                if(c_22 < _S624)
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_22, _S680, tr_4, dv_4, &lu_2, &lv_2, kernelContext_40);
                c_22 = c_22 + 256U;
            }
            group_sum3_0(tid_5, &lu_2, &lv_2, kernelContext_40);
            float3 phi_4 = rows_mul_0((&isl_10)->winv0_0, (&isl_10)->winv1_0, (&isl_10)->winv2_0, lu_2);
            float3 dw_3 = rows_mul_0((&isl_10)->winv0_0, (&isl_10)->winv1_0, (&isl_10)->winv2_0, lv_2);
            uint c_23 = _S682;
            for(;;)
            {
                if(c_23 < _S624)
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_23, _S680, tr_4, phi_4, dv_4, dw_3, kernelContext_40);
                c_23 = c_23 + 256U;
            }
            if(!driven_1)
            {
                thread Island_0 _S684 = isl_10;
                drift_rigid_0(&_S684, &rg_8, tr_4, phi_4, dv_4, dw_3);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        bool _S685;
        if(_S654)
        {
            _S685 = ((&isl_10)->probes_0.y) > ((&isl_10)->probes_0.x);
        }
        else
        {
            _S685 = false;
        }
        if(_S685)
        {
            thread Island_0 _S686 = isl_10;
            thread Rigid_0 _S687 = rg_8;
            record_probes_0(&_S686, &_S687, k_13, kernelContext_40);
        }
        uint _S688 = done_2 + 1U;
        bool _S689;
        if((group_uniform_0(*kernelContext_40->g_halt_0, tid_5)) != 0U)
        {
            thread Island_0 _S690 = isl_10;
            bool _S691 = island_connected_narrow_0(tid_5, ct_2, &_S690, kernelContext_40);
            _S689 = _S691;
        }
        else
        {
            _S689 = false;
        }
        bool resumed_3;
        if(_S689)
        {
            if(_S654)
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
        if((group_uniform_0(*kernelContext_40->g_halt_0, tid_5)) != 0U)
        {
            done_2 = _S688;
            resumed_2 = resumed_3;
            break;
        }
        woke_3 = woke_2;
        done_2 = _S688;
        resumed_2 = resumed_3;
    }
    thread float3 wsum_1 = float3(work_3, work_err_2, 0.0f);
    thread float3 unused_1 = float3(0.0f) ;
    group_sum3_0(tid_5, &wsum_1, &unused_1, kernelContext_40);
    if(_S654)
    {
        thread Quat_0 _S692 = (&rg_8)->rot_0;
        float4 _S693 = quat_vec_0(&_S692);
        (&isl_10)->rotation_0 = _S693;
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
            _S652 = contact_island_1;
        }
        else
        {
            _S652 = false;
        }
        if(_S652)
        {
            contact_split_at_0((&isl_10)->info_0.w + done_2, kernelContext_40);
        }
        float _S694 = wsum_1.x;
        thread float _S695 = (&isl_10)->energy_0.x;
        thread float _S696 = (&isl_10)->energy_0.y;
        comp_add1_0(&_S695, &_S696, _S694);
        (&isl_10)->energy_0.x = _S695;
        (&isl_10)->energy_0.y = _S696 + wsum_1.y;
        (&isl_10)->info_0.w = (&isl_10)->info_0.w + done_2;
        if((((&isl_10)->info_0.x) & 8U) == 0U)
        {
            (&isl_10)->info_0.z = ((&isl_10)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_40->g_halt_0) != 0U)
        {
            (&isl_10)->info_0.z = ((&isl_10)->info_0.z) | 1U;
        }
        uint _S697 = seen_bits_0(kernelContext_40);
        if((*kernelContext_40->g_halt_0) != 0U)
        {
            cov_5 = 128U;
        }
        else
        {
            cov_5 = 0U;
        }
        uint _S698 = _S697 | cov_5;
        if(woke_2)
        {
            cov_5 = 2048U;
        }
        else
        {
            cov_5 = 0U;
        }
        uint _S699 = _S698 | cov_5;
        if(resumed_2)
        {
            cov_5 = 2097280U;
        }
        else
        {
            cov_5 = 0U;
        }
        uint cov_6 = _S699 | cov_5;
        if(done_2 > 0U)
        {
            bool quiet_3 = (((&isl_10)->info_0.x) & 32U) != 0U;
            if(quiet_3)
            {
                cov_5 = 4096U;
            }
            else
            {
                cov_5 = 0U;
            }
            uint cov_7 = cov_6 | cov_5;
            if((((&isl_10)->info_0.x) & 16U) != 0U)
            {
                _S652 = !quiet_3;
            }
            else
            {
                _S652 = false;
            }
            if(_S652)
            {
                cov_5 = 512U;
            }
            else
            {
                cov_5 = 0U;
            }
            uint cov_8 = cov_7 | cov_5;
            if((kernelContext_40->params_0->contact_mode_0) == 1U)
            {
                cov_5 = 524288U;
            }
            else
            {
                cov_5 = 0U;
            }
            uint cov_9 = cov_8 | cov_5;
            if(((&isl_10)->probes_0.y) > ((&isl_10)->probes_0.x))
            {
                cov_5 = 262144U;
            }
            else
            {
                cov_5 = 0U;
            }
            cov_5 = cov_9 | cov_5;
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
        Island_natural_0 device* _S700 = kernelContext_40->islands_0+gi_1;
        _S700->range_0 = packed_uint4(isl_10.range_0) ;
        _S700->info_0 = packed_uint4(isl_10.info_0) ;
        _S700->com_0 = packed_float4(isl_10.com_0) ;
        _S700->inertia0_0 = packed_float4(isl_10.inertia0_0) ;
        _S700->inertia1_0 = packed_float4(isl_10.inertia1_0) ;
        _S700->inertia2_0 = packed_float4(isl_10.inertia2_0) ;
        _S700->inv0_0 = packed_float4(isl_10.inv0_0) ;
        _S700->inv1_0 = packed_float4(isl_10.inv1_0) ;
        _S700->inv2_0 = packed_float4(isl_10.inv2_0) ;
        _S700->wcom_0 = packed_float4(isl_10.wcom_0) ;
        _S700->winv0_0 = packed_float4(isl_10.winv0_0) ;
        _S700->winv1_0 = packed_float4(isl_10.winv1_0) ;
        _S700->winv2_0 = packed_float4(isl_10.winv2_0) ;
        _S700->rotation_0 = packed_float4(isl_10.rotation_0) ;
        _S700->position_0 = packed_float4(isl_10.position_0) ;
        _S700->position_err_0 = packed_float4(isl_10.position_err_0) ;
        _S700->velocity_0 = packed_float4(isl_10.velocity_0) ;
        _S700->velocity_err_0 = packed_float4(isl_10.velocity_err_0) ;
        _S700->angular_velocity_0 = packed_float4(isl_10.angular_velocity_0) ;
        _S700->done_0 = isl_10.done_0;
        _S700->coverage_0 = isl_10.coverage_0;
        _S700->wide_groups_0 = isl_10.wide_groups_0;
        _S700->done_pad_0 = isl_10.done_pad_0;
        _S700->probes_0 = packed_uint4(isl_10.probes_0) ;
        _S700->energy_0 = packed_float4(isl_10.energy_0) ;
        _S700->rotation_err_0 = packed_float4(isl_10.rotation_err_0) ;
        _S700->momentum_0 = packed_float4(isl_10.momentum_0) ;
        _S700->momentum_err_0 = packed_float4(isl_10.momentum_err_0) ;
    }
    if(contact_island_1)
    {
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if(done_2 > 0U)
        {
            c_21 = (&isl_10)->range_0.x + ct_2;
            for(;;)
            {
                if(c_21 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                travel_check_0(c_21, true, kernelContext_40);
                c_21 = c_21 + 256U;
            }
        }
    }
    return;
}

void island_frame_at_2(uint gi_2, uint tid_6, KernelContext_0 thread* kernelContext_41)
{
    bool woke_4;
    uint _S701;
    uint ct_3;
    if(tid_6 < 256U)
    {
        ct_3 = tid_6;
    }
    else
    {
        ct_3 = 1073741824U;
    }
    Island_natural_0 device* _S702 = kernelContext_41->islands_0+gi_2;
    uint4 _S703 = uint4((*_S702).info_0) ;
    float4 _S704 = float4((*_S702).com_0) ;
    float4 _S705 = float4((*_S702).inertia0_0) ;
    float4 _S706 = float4((*_S702).inertia1_0) ;
    float4 _S707 = float4((*_S702).inertia2_0) ;
    float4 _S708 = float4((*_S702).inv0_0) ;
    float4 _S709 = float4((*_S702).inv1_0) ;
    float4 _S710 = float4((*_S702).inv2_0) ;
    float4 _S711 = float4((*_S702).wcom_0) ;
    float4 _S712 = float4((*_S702).winv0_0) ;
    float4 _S713 = float4((*_S702).winv1_0) ;
    float4 _S714 = float4((*_S702).winv2_0) ;
    float4 _S715 = float4((*_S702).rotation_0) ;
    float4 _S716 = float4((*_S702).position_0) ;
    float4 _S717 = float4((*_S702).position_err_0) ;
    float4 _S718 = float4((*_S702).velocity_0) ;
    float4 _S719 = float4((*_S702).velocity_err_0) ;
    float4 _S720 = float4((*_S702).angular_velocity_0) ;
    uint4 _S721 = uint4((*_S702).probes_0) ;
    float4 _S722 = float4((*_S702).energy_0) ;
    float4 _S723 = float4((*_S702).rotation_err_0) ;
    float4 _S724 = float4((*_S702).momentum_0) ;
    float4 _S725 = float4((*_S702).momentum_err_0) ;
    thread Island_0 isl_11;
    (&isl_11)->range_0 = uint4((*_S702).range_0) ;
    (&isl_11)->info_0 = _S703;
    (&isl_11)->com_0 = _S704;
    (&isl_11)->inertia0_0 = _S705;
    (&isl_11)->inertia1_0 = _S706;
    (&isl_11)->inertia2_0 = _S707;
    (&isl_11)->inv0_0 = _S708;
    (&isl_11)->inv1_0 = _S709;
    (&isl_11)->inv2_0 = _S710;
    (&isl_11)->wcom_0 = _S711;
    (&isl_11)->winv0_0 = _S712;
    (&isl_11)->winv1_0 = _S713;
    (&isl_11)->winv2_0 = _S714;
    (&isl_11)->rotation_0 = _S715;
    (&isl_11)->position_0 = _S716;
    (&isl_11)->position_err_0 = _S717;
    (&isl_11)->velocity_0 = _S718;
    (&isl_11)->velocity_err_0 = _S719;
    (&isl_11)->angular_velocity_0 = _S720;
    (&isl_11)->done_0 = (*_S702).done_0;
    (&isl_11)->coverage_0 = (*_S702).coverage_0;
    (&isl_11)->wide_groups_0 = (*_S702).wide_groups_0;
    (&isl_11)->done_pad_0 = (*_S702).done_pad_0;
    (&isl_11)->probes_0 = _S721;
    (&isl_11)->energy_0 = _S722;
    (&isl_11)->rotation_err_0 = _S723;
    (&isl_11)->momentum_0 = _S724;
    (&isl_11)->momentum_err_0 = _S725;
    uint4 info_6 = group_uniform4_0((&isl_11)->info_0, tid_6);
    uint4 _S726 = group_uniform4_0((&isl_11)->range_0, tid_6);
    uint _S727 = info_6.x;
    bool driven_2 = (_S727 & 2U) != 0U;
    bool _S728 = !((_S727 & 1U) != 0U);
    bool _S729;
    if(_S728)
    {
        _S729 = !driven_2;
    }
    else
    {
        _S729 = false;
    }
    bool contact_island_2 = (_S727 & 4U) != 0U;
    bool _S730 = (_S727 & 16U) != 0U;
    seen_clear_0(tid_6, kernelContext_41);
    bool _S731 = tid_6 == 0U;
    uint cov_10;
    bool settled_2;
    if(_S731)
    {
        if(contact_island_2 != ((kernelContext_41->params_0->contact_mode_0) == 1U))
        {
            settled_2 = true;
        }
        else
        {
            settled_2 = (_S727 & 8U) != 0U;
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
            thread Island_0 _S732 = isl_11;
            bool _S733 = contact_stopped_0(&_S732, kernelContext_41);
            settled_2 = _S733;
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
    if(((info_6.z) & 1U) != 0U)
    {
        settled_2 = true;
    }
    else
    {
        settled_2 = (group_uniform_0(*kernelContext_41->g_run_0, tid_6)) == 0U;
    }
    if(settled_2)
    {
        cov_10 = 0U;
    }
    else
    {
        cov_10 = min(info_6.y, kernelContext_41->params_0->max_steps_0);
    }
    float _S734 = kernelContext_41->params_0->dt_0;
    bool _S735 = (kernelContext_41->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S736 = isl_11;
    Rigid_0 _S737 = rigid_of_0(&_S736);
    thread Rigid_0 rg_9 = _S737;
    thread float work_4 = 0.0f;
    thread float work_err_3 = 0.0f;
    settled_2 = _S730;
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
        uint k_14 = info_6.w + done_3 + 1U - 1U - kernelContext_41->params_0->step_start_0;
        bool quiet_4 = (_S727 & 32U) != 0U;
        bool _S738;
        if(settled_2)
        {
            _S738 = !quiet_4;
        }
        else
        {
            _S738 = false;
        }
        uint c_24;
        uint c_25;
        bool _S739;
        bool explicit_step_2;
        if(_S738)
        {
            float3 _S740 = float3(0.0f) ;
            thread float3 norm_2 = _S740;
            thread float3 unused0_2 = _S740;
            c_24 = _S726.x + ct_3;
            for(;;)
            {
                if(c_24 < (_S726.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S741 = (&rg_9)->rot_0;
                float _S742 = settled_chunk_load_0(c_24, &_S741, k_14, _S734, contact_island_2, kernelContext_41);
                norm_2.x = norm_2.x + _S742;
                c_24 = c_24 + 256U;
            }
            group_sum3_0(tid_6, &norm_2, &unused0_2, kernelContext_41);
            if((kernelContext_41->params_0->solve_mode_0) == 1U)
            {
                _S739 = (abs(norm_2.x - (&isl_11)->energy_0.z)) > ((&isl_11)->energy_0.w);
            }
            else
            {
                _S739 = false;
            }
            if(_S739)
            {
                c_25 = 1U;
            }
            else
            {
                c_25 = 0U;
            }
            if((group_uniform_0(c_25, tid_6)) != 0U)
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
        if(_S728)
        {
            _S739 = !quiet_4;
        }
        else
        {
            _S739 = false;
        }
        if(_S739)
        {
            float3 _S743 = float3(0.0f) ;
            thread float3 f_13 = _S743;
            thread float3 t_15 = _S743;
            c_24 = _S726.x + ct_3;
            for(;;)
            {
                if(c_24 < (_S726.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S744 = isl_11;
                thread Rigid_0 _S745 = rg_9;
                net_load_0(c_24, &_S744, &_S745, k_14, _S734, contact_island_2, &f_13, &t_15, kernelContext_41);
                c_24 = c_24 + 256U;
            }
            group_sum3_0(tid_6, &f_13, &t_15, kernelContext_41);
            thread Island_0 _S746 = isl_11;
            rigid_acceleration_0(&_S746, &rg_9, f_13, t_15);
        }
        if(!settled_2)
        {
            explicit_step_2 = !quiet_4;
        }
        else
        {
            explicit_step_2 = false;
        }
        bool _S747;
        if(explicit_step_2)
        {
            if(_S731)
            {
                _S747 = (((uint4((kernelContext_41->islands_0+gi_2)->info_0) ).z) & 2U) != 0U;
            }
            else
            {
                _S747 = false;
            }
            if(_S747)
            {
                *kernelContext_41->g_halt_0 = 1U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            c_24 = _S726.x + ct_3;
            for(;;)
            {
                if(c_24 < (_S726.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S748 = isl_11;
                thread Rigid_0 _S749 = rg_9;
                chunk_update_0(c_24, &_S748, &_S749, _S734, _S735, k_14, contact_island_2, &work_4, &work_err_3, kernelContext_41);
                c_24 = c_24 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        if(_S729)
        {
            _S747 = !quiet_4;
        }
        else
        {
            _S747 = false;
        }
        if(_S747)
        {
            thread Island_0 _S750 = isl_11;
            integrate_rigid_0(&_S750, &rg_9, _S734);
        }
        bool _S751;
        if(_S728)
        {
            _S751 = explicit_step_2;
        }
        else
        {
            _S751 = false;
        }
        if(_S751)
        {
            float3 _S752 = (&isl_11)->wcom_0.xyz;
            float3 _S753 = float3(0.0f) ;
            thread float3 tu_3 = _S753;
            thread float3 pv_3 = _S753;
            uint _S754 = _S726.x + ct_3;
            c_24 = _S754;
            for(;;)
            {
                uint _S755 = _S726.y;
                _S701 = _S755;
                if(c_24 < _S755)
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_24, &tu_3, &pv_3, kernelContext_41);
                c_24 = c_24 + 256U;
            }
            group_sum3_0(tid_6, &tu_3, &pv_3, kernelContext_41);
            float3 tr_5 = tu_3 / float3((&isl_11)->wcom_0.w) ;
            float3 dv_5 = pv_3 / float3((&isl_11)->wcom_0.w) ;
            thread float3 lu_3 = _S753;
            thread float3 lv_3 = _S753;
            c_25 = _S754;
            for(;;)
            {
                if(c_25 < _S701)
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_25, _S752, tr_5, dv_5, &lu_3, &lv_3, kernelContext_41);
                c_25 = c_25 + 256U;
            }
            group_sum3_0(tid_6, &lu_3, &lv_3, kernelContext_41);
            float3 phi_5 = rows_mul_0((&isl_11)->winv0_0, (&isl_11)->winv1_0, (&isl_11)->winv2_0, lu_3);
            float3 dw_4 = rows_mul_0((&isl_11)->winv0_0, (&isl_11)->winv1_0, (&isl_11)->winv2_0, lv_3);
            uint c_26 = _S754;
            for(;;)
            {
                if(c_26 < _S701)
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_26, _S752, tr_5, phi_5, dv_5, dw_4, kernelContext_41);
                c_26 = c_26 + 256U;
            }
            if(!driven_2)
            {
                thread Island_0 _S756 = isl_11;
                drift_rigid_0(&_S756, &rg_9, tr_5, phi_5, dv_5, dw_4);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        bool _S757;
        if(_S731)
        {
            _S757 = ((&isl_11)->probes_0.y) > ((&isl_11)->probes_0.x);
        }
        else
        {
            _S757 = false;
        }
        if(_S757)
        {
            thread Island_0 _S758 = isl_11;
            thread Rigid_0 _S759 = rg_9;
            record_probes_0(&_S758, &_S759, k_14, kernelContext_41);
        }
        uint _S760 = done_3 + 1U;
        bool _S761;
        if((group_uniform_0(*kernelContext_41->g_halt_0, tid_6)) != 0U)
        {
            thread Island_0 _S762 = isl_11;
            bool _S763 = island_connected_narrow_0(tid_6, ct_3, &_S762, kernelContext_41);
            _S761 = _S763;
        }
        else
        {
            _S761 = false;
        }
        bool resumed_5;
        if(_S761)
        {
            if(_S731)
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
        if((group_uniform_0(*kernelContext_41->g_halt_0, tid_6)) != 0U)
        {
            done_3 = _S760;
            resumed_4 = resumed_5;
            break;
        }
        woke_5 = woke_4;
        done_3 = _S760;
        resumed_4 = resumed_5;
    }
    thread float3 wsum_2 = float3(work_4, work_err_3, 0.0f);
    thread float3 unused_2 = float3(0.0f) ;
    group_sum3_0(tid_6, &wsum_2, &unused_2, kernelContext_41);
    if(_S731)
    {
        thread Quat_0 _S764 = (&rg_9)->rot_0;
        float4 _S765 = quat_vec_0(&_S764);
        (&isl_11)->rotation_0 = _S765;
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
            _S729 = contact_island_2;
        }
        else
        {
            _S729 = false;
        }
        if(_S729)
        {
            contact_split_at_0((&isl_11)->info_0.w + done_3, kernelContext_41);
        }
        float _S766 = wsum_2.x;
        thread float _S767 = (&isl_11)->energy_0.x;
        thread float _S768 = (&isl_11)->energy_0.y;
        comp_add1_0(&_S767, &_S768, _S766);
        (&isl_11)->energy_0.x = _S767;
        (&isl_11)->energy_0.y = _S768 + wsum_2.y;
        (&isl_11)->info_0.w = (&isl_11)->info_0.w + done_3;
        if((((&isl_11)->info_0.x) & 8U) == 0U)
        {
            (&isl_11)->info_0.z = ((&isl_11)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_41->g_halt_0) != 0U)
        {
            (&isl_11)->info_0.z = ((&isl_11)->info_0.z) | 1U;
        }
        uint _S769 = seen_bits_0(kernelContext_41);
        if((*kernelContext_41->g_halt_0) != 0U)
        {
            ct_3 = 128U;
        }
        else
        {
            ct_3 = 0U;
        }
        uint _S770 = _S769 | ct_3;
        if(woke_4)
        {
            ct_3 = 2048U;
        }
        else
        {
            ct_3 = 0U;
        }
        uint _S771 = _S770 | ct_3;
        if(resumed_4)
        {
            ct_3 = 2097280U;
        }
        else
        {
            ct_3 = 0U;
        }
        uint cov_11 = _S771 | ct_3;
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
                _S729 = !quiet_5;
            }
            else
            {
                _S729 = false;
            }
            if(_S729)
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
        Island_natural_0 device* _S772 = kernelContext_41->islands_0+gi_2;
        _S772->range_0 = packed_uint4(isl_11.range_0) ;
        _S772->info_0 = packed_uint4(isl_11.info_0) ;
        _S772->com_0 = packed_float4(isl_11.com_0) ;
        _S772->inertia0_0 = packed_float4(isl_11.inertia0_0) ;
        _S772->inertia1_0 = packed_float4(isl_11.inertia1_0) ;
        _S772->inertia2_0 = packed_float4(isl_11.inertia2_0) ;
        _S772->inv0_0 = packed_float4(isl_11.inv0_0) ;
        _S772->inv1_0 = packed_float4(isl_11.inv1_0) ;
        _S772->inv2_0 = packed_float4(isl_11.inv2_0) ;
        _S772->wcom_0 = packed_float4(isl_11.wcom_0) ;
        _S772->winv0_0 = packed_float4(isl_11.winv0_0) ;
        _S772->winv1_0 = packed_float4(isl_11.winv1_0) ;
        _S772->winv2_0 = packed_float4(isl_11.winv2_0) ;
        _S772->rotation_0 = packed_float4(isl_11.rotation_0) ;
        _S772->position_0 = packed_float4(isl_11.position_0) ;
        _S772->position_err_0 = packed_float4(isl_11.position_err_0) ;
        _S772->velocity_0 = packed_float4(isl_11.velocity_0) ;
        _S772->velocity_err_0 = packed_float4(isl_11.velocity_err_0) ;
        _S772->angular_velocity_0 = packed_float4(isl_11.angular_velocity_0) ;
        _S772->done_0 = isl_11.done_0;
        _S772->coverage_0 = isl_11.coverage_0;
        _S772->wide_groups_0 = isl_11.wide_groups_0;
        _S772->done_pad_0 = isl_11.done_pad_0;
        _S772->probes_0 = packed_uint4(isl_11.probes_0) ;
        _S772->energy_0 = packed_float4(isl_11.energy_0) ;
        _S772->rotation_err_0 = packed_float4(isl_11.rotation_err_0) ;
        _S772->momentum_0 = packed_float4(isl_11.momentum_0) ;
        _S772->momentum_err_0 = packed_float4(isl_11.momentum_err_0) ;
    }
    return;
}

void island_frame_at_3(uint gi_3, uint tid_7, KernelContext_0 thread* kernelContext_42)
{
    bool woke_6;
    uint _S773;
    uint ct_4;
    if(tid_7 < 256U)
    {
        ct_4 = tid_7;
    }
    else
    {
        ct_4 = 1073741824U;
    }
    Island_natural_0 device* _S774 = kernelContext_42->islands_0+gi_3;
    uint4 _S775 = uint4((*_S774).info_0) ;
    float4 _S776 = float4((*_S774).com_0) ;
    float4 _S777 = float4((*_S774).inertia0_0) ;
    float4 _S778 = float4((*_S774).inertia1_0) ;
    float4 _S779 = float4((*_S774).inertia2_0) ;
    float4 _S780 = float4((*_S774).inv0_0) ;
    float4 _S781 = float4((*_S774).inv1_0) ;
    float4 _S782 = float4((*_S774).inv2_0) ;
    float4 _S783 = float4((*_S774).wcom_0) ;
    float4 _S784 = float4((*_S774).winv0_0) ;
    float4 _S785 = float4((*_S774).winv1_0) ;
    float4 _S786 = float4((*_S774).winv2_0) ;
    float4 _S787 = float4((*_S774).rotation_0) ;
    float4 _S788 = float4((*_S774).position_0) ;
    float4 _S789 = float4((*_S774).position_err_0) ;
    float4 _S790 = float4((*_S774).velocity_0) ;
    float4 _S791 = float4((*_S774).velocity_err_0) ;
    float4 _S792 = float4((*_S774).angular_velocity_0) ;
    uint4 _S793 = uint4((*_S774).probes_0) ;
    float4 _S794 = float4((*_S774).energy_0) ;
    float4 _S795 = float4((*_S774).rotation_err_0) ;
    float4 _S796 = float4((*_S774).momentum_0) ;
    float4 _S797 = float4((*_S774).momentum_err_0) ;
    thread Island_0 isl_12;
    (&isl_12)->range_0 = uint4((*_S774).range_0) ;
    (&isl_12)->info_0 = _S775;
    (&isl_12)->com_0 = _S776;
    (&isl_12)->inertia0_0 = _S777;
    (&isl_12)->inertia1_0 = _S778;
    (&isl_12)->inertia2_0 = _S779;
    (&isl_12)->inv0_0 = _S780;
    (&isl_12)->inv1_0 = _S781;
    (&isl_12)->inv2_0 = _S782;
    (&isl_12)->wcom_0 = _S783;
    (&isl_12)->winv0_0 = _S784;
    (&isl_12)->winv1_0 = _S785;
    (&isl_12)->winv2_0 = _S786;
    (&isl_12)->rotation_0 = _S787;
    (&isl_12)->position_0 = _S788;
    (&isl_12)->position_err_0 = _S789;
    (&isl_12)->velocity_0 = _S790;
    (&isl_12)->velocity_err_0 = _S791;
    (&isl_12)->angular_velocity_0 = _S792;
    (&isl_12)->done_0 = (*_S774).done_0;
    (&isl_12)->coverage_0 = (*_S774).coverage_0;
    (&isl_12)->wide_groups_0 = (*_S774).wide_groups_0;
    (&isl_12)->done_pad_0 = (*_S774).done_pad_0;
    (&isl_12)->probes_0 = _S793;
    (&isl_12)->energy_0 = _S794;
    (&isl_12)->rotation_err_0 = _S795;
    (&isl_12)->momentum_0 = _S796;
    (&isl_12)->momentum_err_0 = _S797;
    uint4 info_7 = group_uniform4_0((&isl_12)->info_0, tid_7);
    uint4 _S798 = group_uniform4_0((&isl_12)->range_0, tid_7);
    uint _S799 = info_7.x;
    bool driven_3 = (_S799 & 2U) != 0U;
    bool _S800 = !((_S799 & 1U) != 0U);
    bool _S801;
    if(_S800)
    {
        _S801 = !driven_3;
    }
    else
    {
        _S801 = false;
    }
    bool contact_island_3 = (_S799 & 4U) != 0U;
    bool _S802 = (_S799 & 16U) != 0U;
    seen_clear_0(tid_7, kernelContext_42);
    bool _S803 = tid_7 == 0U;
    uint cov_15;
    bool settled_3;
    if(_S803)
    {
        if(contact_island_3 != ((kernelContext_42->params_0->contact_mode_0) == 1U))
        {
            settled_3 = true;
        }
        else
        {
            settled_3 = (_S799 & 8U) != 0U;
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
            thread Island_0 _S804 = isl_12;
            bool _S805 = contact_stopped_0(&_S804, kernelContext_42);
            settled_3 = _S805;
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
    if(((info_7.z) & 1U) != 0U)
    {
        settled_3 = true;
    }
    else
    {
        settled_3 = (group_uniform_0(*kernelContext_42->g_run_0, tid_7)) == 0U;
    }
    if(settled_3)
    {
        cov_15 = 0U;
    }
    else
    {
        cov_15 = min(info_7.y, kernelContext_42->params_0->max_steps_0);
    }
    float _S806 = kernelContext_42->params_0->dt_0;
    bool _S807 = (kernelContext_42->params_0->fracture_0) != 0U;
    bool _S808 = (kernelContext_42->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S809 = isl_12;
    Rigid_0 _S810 = rigid_of_0(&_S809);
    thread Rigid_0 rg_10 = _S810;
    thread float work_5 = 0.0f;
    thread float work_err_4 = 0.0f;
    settled_3 = _S802;
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
        uint abs_step_2 = info_7.w + done_4 + 1U;
        uint k_15 = abs_step_2 - 1U - kernelContext_42->params_0->step_start_0;
        bool quiet_6 = (_S799 & 32U) != 0U;
        bool _S811;
        if(settled_3)
        {
            _S811 = !quiet_6;
        }
        else
        {
            _S811 = false;
        }
        uint c_27;
        uint c_28;
        bool _S812;
        bool explicit_step_3;
        if(_S811)
        {
            float3 _S813 = float3(0.0f) ;
            thread float3 norm_3 = _S813;
            thread float3 unused0_3 = _S813;
            c_27 = _S798.x + ct_4;
            for(;;)
            {
                if(c_27 < (_S798.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S814 = (&rg_10)->rot_0;
                float _S815 = settled_chunk_load_0(c_27, &_S814, k_15, _S806, contact_island_3, kernelContext_42);
                norm_3.x = norm_3.x + _S815;
                c_27 = c_27 + 256U;
            }
            group_sum3_0(tid_7, &norm_3, &unused0_3, kernelContext_42);
            if((kernelContext_42->params_0->solve_mode_0) == 1U)
            {
                _S812 = (abs(norm_3.x - (&isl_12)->energy_0.z)) > ((&isl_12)->energy_0.w);
            }
            else
            {
                _S812 = false;
            }
            if(_S812)
            {
                c_28 = 1U;
            }
            else
            {
                c_28 = 0U;
            }
            if((group_uniform_0(c_28, tid_7)) != 0U)
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
        if(_S800)
        {
            _S812 = !quiet_6;
        }
        else
        {
            _S812 = false;
        }
        if(_S812)
        {
            float3 _S816 = float3(0.0f) ;
            thread float3 f_14 = _S816;
            thread float3 t_16 = _S816;
            c_27 = _S798.x + ct_4;
            for(;;)
            {
                if(c_27 < (_S798.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S817 = isl_12;
                thread Rigid_0 _S818 = rg_10;
                net_load_0(c_27, &_S817, &_S818, k_15, _S806, contact_island_3, &f_14, &t_16, kernelContext_42);
                c_27 = c_27 + 256U;
            }
            group_sum3_0(tid_7, &f_14, &t_16, kernelContext_42);
            thread Island_0 _S819 = isl_12;
            rigid_acceleration_0(&_S819, &rg_10, f_14, t_16);
        }
        if(!settled_3)
        {
            explicit_step_3 = !quiet_6;
        }
        else
        {
            explicit_step_3 = false;
        }
        if(explicit_step_3)
        {
            c_27 = _S798.z + tid_7;
            for(;;)
            {
                if(c_27 < (_S798.w))
                {
                }
                else
                {
                    break;
                }
                bool _S820 = bond_update_0(c_27, _S806, _S807, abs_step_2, false, kernelContext_42);
                if(_S820)
                {
                    *kernelContext_42->g_halt_0 = 1U;
                }
                c_27 = c_27 + 512U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            c_28 = _S798.x + ct_4;
            for(;;)
            {
                if(c_28 < (_S798.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S821 = isl_12;
                thread Rigid_0 _S822 = rg_10;
                chunk_update_0(c_28, &_S821, &_S822, _S806, _S808, k_15, contact_island_3, &work_5, &work_err_4, kernelContext_42);
                c_28 = c_28 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        bool _S823;
        if(_S801)
        {
            _S823 = !quiet_6;
        }
        else
        {
            _S823 = false;
        }
        if(_S823)
        {
            thread Island_0 _S824 = isl_12;
            integrate_rigid_0(&_S824, &rg_10, _S806);
        }
        bool _S825;
        if(_S800)
        {
            _S825 = explicit_step_3;
        }
        else
        {
            _S825 = false;
        }
        if(_S825)
        {
            float3 _S826 = (&isl_12)->wcom_0.xyz;
            float3 _S827 = float3(0.0f) ;
            thread float3 tu_4 = _S827;
            thread float3 pv_4 = _S827;
            uint _S828 = _S798.x + ct_4;
            c_27 = _S828;
            for(;;)
            {
                uint _S829 = _S798.y;
                _S773 = _S829;
                if(c_27 < _S829)
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_27, &tu_4, &pv_4, kernelContext_42);
                c_27 = c_27 + 256U;
            }
            group_sum3_0(tid_7, &tu_4, &pv_4, kernelContext_42);
            float3 tr_6 = tu_4 / float3((&isl_12)->wcom_0.w) ;
            float3 dv_6 = pv_4 / float3((&isl_12)->wcom_0.w) ;
            thread float3 lu_4 = _S827;
            thread float3 lv_4 = _S827;
            c_28 = _S828;
            for(;;)
            {
                if(c_28 < _S773)
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_28, _S826, tr_6, dv_6, &lu_4, &lv_4, kernelContext_42);
                c_28 = c_28 + 256U;
            }
            group_sum3_0(tid_7, &lu_4, &lv_4, kernelContext_42);
            float3 phi_6 = rows_mul_0((&isl_12)->winv0_0, (&isl_12)->winv1_0, (&isl_12)->winv2_0, lu_4);
            float3 dw_5 = rows_mul_0((&isl_12)->winv0_0, (&isl_12)->winv1_0, (&isl_12)->winv2_0, lv_4);
            uint c_29 = _S828;
            for(;;)
            {
                if(c_29 < _S773)
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_29, _S826, tr_6, phi_6, dv_6, dw_5, kernelContext_42);
                c_29 = c_29 + 256U;
            }
            if(!driven_3)
            {
                thread Island_0 _S830 = isl_12;
                drift_rigid_0(&_S830, &rg_10, tr_6, phi_6, dv_6, dw_5);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        bool _S831;
        if(_S803)
        {
            _S831 = ((&isl_12)->probes_0.y) > ((&isl_12)->probes_0.x);
        }
        else
        {
            _S831 = false;
        }
        if(_S831)
        {
            thread Island_0 _S832 = isl_12;
            thread Rigid_0 _S833 = rg_10;
            record_probes_0(&_S832, &_S833, k_15, kernelContext_42);
        }
        uint _S834 = done_4 + 1U;
        bool _S835;
        if((group_uniform_0(*kernelContext_42->g_halt_0, tid_7)) != 0U)
        {
            thread Island_0 _S836 = isl_12;
            bool _S837 = island_connected_narrow_0(tid_7, ct_4, &_S836, kernelContext_42);
            _S835 = _S837;
        }
        else
        {
            _S835 = false;
        }
        bool resumed_7;
        if(_S835)
        {
            if(_S803)
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
        if((group_uniform_0(*kernelContext_42->g_halt_0, tid_7)) != 0U)
        {
            done_4 = _S834;
            resumed_6 = resumed_7;
            break;
        }
        woke_7 = woke_6;
        done_4 = _S834;
        resumed_6 = resumed_7;
    }
    thread float3 wsum_3 = float3(work_5, work_err_4, 0.0f);
    thread float3 unused_3 = float3(0.0f) ;
    group_sum3_0(tid_7, &wsum_3, &unused_3, kernelContext_42);
    if(_S803)
    {
        thread Quat_0 _S838 = (&rg_10)->rot_0;
        float4 _S839 = quat_vec_0(&_S838);
        (&isl_12)->rotation_0 = _S839;
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
            _S801 = contact_island_3;
        }
        else
        {
            _S801 = false;
        }
        if(_S801)
        {
            contact_split_at_0((&isl_12)->info_0.w + done_4, kernelContext_42);
        }
        float _S840 = wsum_3.x;
        thread float _S841 = (&isl_12)->energy_0.x;
        thread float _S842 = (&isl_12)->energy_0.y;
        comp_add1_0(&_S841, &_S842, _S840);
        (&isl_12)->energy_0.x = _S841;
        (&isl_12)->energy_0.y = _S842 + wsum_3.y;
        (&isl_12)->info_0.w = (&isl_12)->info_0.w + done_4;
        if((((&isl_12)->info_0.x) & 8U) == 0U)
        {
            (&isl_12)->info_0.z = ((&isl_12)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_42->g_halt_0) != 0U)
        {
            (&isl_12)->info_0.z = ((&isl_12)->info_0.z) | 1U;
        }
        uint _S843 = seen_bits_0(kernelContext_42);
        if((*kernelContext_42->g_halt_0) != 0U)
        {
            ct_4 = 128U;
        }
        else
        {
            ct_4 = 0U;
        }
        uint _S844 = _S843 | ct_4;
        if(woke_6)
        {
            ct_4 = 2048U;
        }
        else
        {
            ct_4 = 0U;
        }
        uint _S845 = _S844 | ct_4;
        if(resumed_6)
        {
            ct_4 = 2097280U;
        }
        else
        {
            ct_4 = 0U;
        }
        uint cov_16 = _S845 | ct_4;
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
                _S801 = !quiet_7;
            }
            else
            {
                _S801 = false;
            }
            if(_S801)
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
        Island_natural_0 device* _S846 = kernelContext_42->islands_0+gi_3;
        _S846->range_0 = packed_uint4(isl_12.range_0) ;
        _S846->info_0 = packed_uint4(isl_12.info_0) ;
        _S846->com_0 = packed_float4(isl_12.com_0) ;
        _S846->inertia0_0 = packed_float4(isl_12.inertia0_0) ;
        _S846->inertia1_0 = packed_float4(isl_12.inertia1_0) ;
        _S846->inertia2_0 = packed_float4(isl_12.inertia2_0) ;
        _S846->inv0_0 = packed_float4(isl_12.inv0_0) ;
        _S846->inv1_0 = packed_float4(isl_12.inv1_0) ;
        _S846->inv2_0 = packed_float4(isl_12.inv2_0) ;
        _S846->wcom_0 = packed_float4(isl_12.wcom_0) ;
        _S846->winv0_0 = packed_float4(isl_12.winv0_0) ;
        _S846->winv1_0 = packed_float4(isl_12.winv1_0) ;
        _S846->winv2_0 = packed_float4(isl_12.winv2_0) ;
        _S846->rotation_0 = packed_float4(isl_12.rotation_0) ;
        _S846->position_0 = packed_float4(isl_12.position_0) ;
        _S846->position_err_0 = packed_float4(isl_12.position_err_0) ;
        _S846->velocity_0 = packed_float4(isl_12.velocity_0) ;
        _S846->velocity_err_0 = packed_float4(isl_12.velocity_err_0) ;
        _S846->angular_velocity_0 = packed_float4(isl_12.angular_velocity_0) ;
        _S846->done_0 = isl_12.done_0;
        _S846->coverage_0 = isl_12.coverage_0;
        _S846->wide_groups_0 = isl_12.wide_groups_0;
        _S846->done_pad_0 = isl_12.done_pad_0;
        _S846->probes_0 = packed_uint4(isl_12.probes_0) ;
        _S846->energy_0 = packed_float4(isl_12.energy_0) ;
        _S846->rotation_err_0 = packed_float4(isl_12.rotation_err_0) ;
        _S846->momentum_0 = packed_float4(isl_12.momentum_0) ;
        _S846->momentum_err_0 = packed_float4(isl_12.momentum_err_0) ;
    }
    return;
}

void island_frame_at_4(uint gi_4, uint tid_8, KernelContext_0 thread* kernelContext_43)
{
    bool woke_8;
    uint _S847;
    uint ct_5;
    if(tid_8 < 256U)
    {
        ct_5 = tid_8;
    }
    else
    {
        ct_5 = 1073741824U;
    }
    Island_natural_0 device* _S848 = kernelContext_43->islands_0+gi_4;
    uint4 _S849 = uint4((*_S848).info_0) ;
    float4 _S850 = float4((*_S848).com_0) ;
    float4 _S851 = float4((*_S848).inertia0_0) ;
    float4 _S852 = float4((*_S848).inertia1_0) ;
    float4 _S853 = float4((*_S848).inertia2_0) ;
    float4 _S854 = float4((*_S848).inv0_0) ;
    float4 _S855 = float4((*_S848).inv1_0) ;
    float4 _S856 = float4((*_S848).inv2_0) ;
    float4 _S857 = float4((*_S848).wcom_0) ;
    float4 _S858 = float4((*_S848).winv0_0) ;
    float4 _S859 = float4((*_S848).winv1_0) ;
    float4 _S860 = float4((*_S848).winv2_0) ;
    float4 _S861 = float4((*_S848).rotation_0) ;
    float4 _S862 = float4((*_S848).position_0) ;
    float4 _S863 = float4((*_S848).position_err_0) ;
    float4 _S864 = float4((*_S848).velocity_0) ;
    float4 _S865 = float4((*_S848).velocity_err_0) ;
    float4 _S866 = float4((*_S848).angular_velocity_0) ;
    uint4 _S867 = uint4((*_S848).probes_0) ;
    float4 _S868 = float4((*_S848).energy_0) ;
    float4 _S869 = float4((*_S848).rotation_err_0) ;
    float4 _S870 = float4((*_S848).momentum_0) ;
    float4 _S871 = float4((*_S848).momentum_err_0) ;
    thread Island_0 isl_13;
    (&isl_13)->range_0 = uint4((*_S848).range_0) ;
    (&isl_13)->info_0 = _S849;
    (&isl_13)->com_0 = _S850;
    (&isl_13)->inertia0_0 = _S851;
    (&isl_13)->inertia1_0 = _S852;
    (&isl_13)->inertia2_0 = _S853;
    (&isl_13)->inv0_0 = _S854;
    (&isl_13)->inv1_0 = _S855;
    (&isl_13)->inv2_0 = _S856;
    (&isl_13)->wcom_0 = _S857;
    (&isl_13)->winv0_0 = _S858;
    (&isl_13)->winv1_0 = _S859;
    (&isl_13)->winv2_0 = _S860;
    (&isl_13)->rotation_0 = _S861;
    (&isl_13)->position_0 = _S862;
    (&isl_13)->position_err_0 = _S863;
    (&isl_13)->velocity_0 = _S864;
    (&isl_13)->velocity_err_0 = _S865;
    (&isl_13)->angular_velocity_0 = _S866;
    (&isl_13)->done_0 = (*_S848).done_0;
    (&isl_13)->coverage_0 = (*_S848).coverage_0;
    (&isl_13)->wide_groups_0 = (*_S848).wide_groups_0;
    (&isl_13)->done_pad_0 = (*_S848).done_pad_0;
    (&isl_13)->probes_0 = _S867;
    (&isl_13)->energy_0 = _S868;
    (&isl_13)->rotation_err_0 = _S869;
    (&isl_13)->momentum_0 = _S870;
    (&isl_13)->momentum_err_0 = _S871;
    uint4 info_8 = group_uniform4_0((&isl_13)->info_0, tid_8);
    uint4 _S872 = group_uniform4_0((&isl_13)->range_0, tid_8);
    uint _S873 = info_8.x;
    bool driven_4 = (_S873 & 2U) != 0U;
    bool _S874 = !((_S873 & 1U) != 0U);
    bool _S875;
    if(_S874)
    {
        _S875 = !driven_4;
    }
    else
    {
        _S875 = false;
    }
    bool contact_island_4 = (_S873 & 4U) != 0U;
    bool _S876 = (_S873 & 16U) != 0U;
    seen_clear_0(tid_8, kernelContext_43);
    bool _S877 = tid_8 == 0U;
    uint cov_20;
    bool settled_4;
    if(_S877)
    {
        if(contact_island_4 != ((kernelContext_43->params_0->contact_mode_0) == 1U))
        {
            settled_4 = true;
        }
        else
        {
            settled_4 = (_S873 & 8U) != 0U;
        }
        if(settled_4)
        {
            cov_20 = 0U;
        }
        else
        {
            cov_20 = 1U;
        }
        if(contact_island_4)
        {
            thread Island_0 _S878 = isl_13;
            bool _S879 = contact_stopped_0(&_S878, kernelContext_43);
            settled_4 = _S879;
        }
        else
        {
            settled_4 = false;
        }
        if(settled_4)
        {
            cov_20 = 0U;
        }
        *kernelContext_43->g_run_0 = cov_20;
        *kernelContext_43->g_halt_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(((info_8.z) & 1U) != 0U)
    {
        settled_4 = true;
    }
    else
    {
        settled_4 = (group_uniform_0(*kernelContext_43->g_run_0, tid_8)) == 0U;
    }
    if(settled_4)
    {
        cov_20 = 0U;
    }
    else
    {
        cov_20 = min(info_8.y, kernelContext_43->params_0->max_steps_0);
    }
    float _S880 = kernelContext_43->params_0->dt_0;
    bool _S881 = (kernelContext_43->params_0->fracture_0) != 0U;
    bool _S882 = (kernelContext_43->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S883 = isl_13;
    Rigid_0 _S884 = rigid_of_0(&_S883);
    thread Rigid_0 rg_11 = _S884;
    thread float work_6 = 0.0f;
    thread float work_err_5 = 0.0f;
    settled_4 = _S876;
    bool woke_9 = false;
    uint done_5 = 0U;
    bool resumed_8 = false;
    for(;;)
    {
        if(done_5 < cov_20)
        {
        }
        else
        {
            woke_8 = woke_9;
            break;
        }
        uint abs_step_3 = info_8.w + done_5 + 1U;
        uint k_16 = abs_step_3 - 1U - kernelContext_43->params_0->step_start_0;
        bool quiet_8 = (_S873 & 32U) != 0U;
        bool _S885;
        if(settled_4)
        {
            _S885 = !quiet_8;
        }
        else
        {
            _S885 = false;
        }
        uint c_30;
        uint c_31;
        bool _S886;
        bool explicit_step_4;
        if(_S885)
        {
            float3 _S887 = float3(0.0f) ;
            thread float3 norm_4 = _S887;
            thread float3 unused0_4 = _S887;
            c_30 = _S872.x + ct_5;
            for(;;)
            {
                if(c_30 < (_S872.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S888 = (&rg_11)->rot_0;
                float _S889 = settled_chunk_load_0(c_30, &_S888, k_16, _S880, contact_island_4, kernelContext_43);
                norm_4.x = norm_4.x + _S889;
                c_30 = c_30 + 256U;
            }
            group_sum3_0(tid_8, &norm_4, &unused0_4, kernelContext_43);
            if((kernelContext_43->params_0->solve_mode_0) == 1U)
            {
                _S886 = (abs(norm_4.x - (&isl_13)->energy_0.z)) > ((&isl_13)->energy_0.w);
            }
            else
            {
                _S886 = false;
            }
            if(_S886)
            {
                c_31 = 1U;
            }
            else
            {
                c_31 = 0U;
            }
            if((group_uniform_0(c_31, tid_8)) != 0U)
            {
                explicit_step_4 = false;
                woke_8 = true;
            }
            else
            {
                explicit_step_4 = settled_4;
                woke_8 = woke_9;
            }
            settled_4 = explicit_step_4;
        }
        else
        {
            woke_8 = woke_9;
        }
        if(_S874)
        {
            _S886 = !quiet_8;
        }
        else
        {
            _S886 = false;
        }
        if(_S886)
        {
            float3 _S890 = float3(0.0f) ;
            thread float3 f_15 = _S890;
            thread float3 t_17 = _S890;
            c_30 = _S872.x + ct_5;
            for(;;)
            {
                if(c_30 < (_S872.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S891 = isl_13;
                thread Rigid_0 _S892 = rg_11;
                net_load_0(c_30, &_S891, &_S892, k_16, _S880, contact_island_4, &f_15, &t_17, kernelContext_43);
                c_30 = c_30 + 256U;
            }
            group_sum3_0(tid_8, &f_15, &t_17, kernelContext_43);
            thread Island_0 _S893 = isl_13;
            rigid_acceleration_0(&_S893, &rg_11, f_15, t_17);
        }
        if(!settled_4)
        {
            explicit_step_4 = !quiet_8;
        }
        else
        {
            explicit_step_4 = false;
        }
        if(explicit_step_4)
        {
            c_30 = _S872.z + tid_8;
            for(;;)
            {
                if(c_30 < (_S872.w))
                {
                }
                else
                {
                    break;
                }
                bool _S894 = bond_update_0(c_30, _S880, _S881, abs_step_3, false, kernelContext_43);
                if(_S894)
                {
                    *kernelContext_43->g_halt_0 = 1U;
                }
                c_30 = c_30 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            c_31 = _S872.x + ct_5;
            for(;;)
            {
                if(c_31 < (_S872.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S895 = isl_13;
                thread Rigid_0 _S896 = rg_11;
                chunk_update_0(c_31, &_S895, &_S896, _S880, _S882, k_16, contact_island_4, &work_6, &work_err_5, kernelContext_43);
                c_31 = c_31 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        bool _S897;
        if(_S875)
        {
            _S897 = !quiet_8;
        }
        else
        {
            _S897 = false;
        }
        if(_S897)
        {
            thread Island_0 _S898 = isl_13;
            integrate_rigid_0(&_S898, &rg_11, _S880);
        }
        bool _S899;
        if(_S874)
        {
            _S899 = explicit_step_4;
        }
        else
        {
            _S899 = false;
        }
        if(_S899)
        {
            float3 _S900 = (&isl_13)->wcom_0.xyz;
            float3 _S901 = float3(0.0f) ;
            thread float3 tu_5 = _S901;
            thread float3 pv_5 = _S901;
            uint _S902 = _S872.x + ct_5;
            c_30 = _S902;
            for(;;)
            {
                uint _S903 = _S872.y;
                _S847 = _S903;
                if(c_30 < _S903)
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_30, &tu_5, &pv_5, kernelContext_43);
                c_30 = c_30 + 256U;
            }
            group_sum3_0(tid_8, &tu_5, &pv_5, kernelContext_43);
            float3 tr_7 = tu_5 / float3((&isl_13)->wcom_0.w) ;
            float3 dv_7 = pv_5 / float3((&isl_13)->wcom_0.w) ;
            thread float3 lu_5 = _S901;
            thread float3 lv_5 = _S901;
            c_31 = _S902;
            for(;;)
            {
                if(c_31 < _S847)
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_31, _S900, tr_7, dv_7, &lu_5, &lv_5, kernelContext_43);
                c_31 = c_31 + 256U;
            }
            group_sum3_0(tid_8, &lu_5, &lv_5, kernelContext_43);
            float3 phi_7 = rows_mul_0((&isl_13)->winv0_0, (&isl_13)->winv1_0, (&isl_13)->winv2_0, lu_5);
            float3 dw_6 = rows_mul_0((&isl_13)->winv0_0, (&isl_13)->winv1_0, (&isl_13)->winv2_0, lv_5);
            uint c_32 = _S902;
            for(;;)
            {
                if(c_32 < _S847)
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_32, _S900, tr_7, phi_7, dv_7, dw_6, kernelContext_43);
                c_32 = c_32 + 256U;
            }
            if(!driven_4)
            {
                thread Island_0 _S904 = isl_13;
                drift_rigid_0(&_S904, &rg_11, tr_7, phi_7, dv_7, dw_6);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        bool _S905;
        if(_S877)
        {
            _S905 = ((&isl_13)->probes_0.y) > ((&isl_13)->probes_0.x);
        }
        else
        {
            _S905 = false;
        }
        if(_S905)
        {
            thread Island_0 _S906 = isl_13;
            thread Rigid_0 _S907 = rg_11;
            record_probes_0(&_S906, &_S907, k_16, kernelContext_43);
        }
        uint _S908 = done_5 + 1U;
        bool _S909;
        if((group_uniform_0(*kernelContext_43->g_halt_0, tid_8)) != 0U)
        {
            thread Island_0 _S910 = isl_13;
            bool _S911 = island_connected_narrow_0(tid_8, ct_5, &_S910, kernelContext_43);
            _S909 = _S911;
        }
        else
        {
            _S909 = false;
        }
        bool resumed_9;
        if(_S909)
        {
            if(_S877)
            {
                *kernelContext_43->g_halt_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            resumed_9 = true;
        }
        else
        {
            resumed_9 = resumed_8;
        }
        if((group_uniform_0(*kernelContext_43->g_halt_0, tid_8)) != 0U)
        {
            done_5 = _S908;
            resumed_8 = resumed_9;
            break;
        }
        woke_9 = woke_8;
        done_5 = _S908;
        resumed_8 = resumed_9;
    }
    thread float3 wsum_4 = float3(work_6, work_err_5, 0.0f);
    thread float3 unused_4 = float3(0.0f) ;
    group_sum3_0(tid_8, &wsum_4, &unused_4, kernelContext_43);
    if(_S877)
    {
        thread Quat_0 _S912 = (&rg_11)->rot_0;
        float4 _S913 = quat_vec_0(&_S912);
        (&isl_13)->rotation_0 = _S913;
        (&isl_13)->rotation_err_0 = (&rg_11)->rot_err_0;
        (&isl_13)->position_0 = float4((&rg_11)->pos_0, 0.0f);
        (&isl_13)->position_err_0 = float4((&rg_11)->pos_err_0, 0.0f);
        (&isl_13)->velocity_0 = float4((&rg_11)->vel_0, 0.0f);
        (&isl_13)->velocity_err_0 = float4((&rg_11)->vel_err_0, 0.0f);
        (&isl_13)->angular_velocity_0 = float4((&rg_11)->w_1, 0.0f);
        (&isl_13)->momentum_0 = float4((&rg_11)->l_0, 0.0f);
        (&isl_13)->momentum_err_0 = float4((&rg_11)->l_err_0, 0.0f);
        (&isl_13)->done_0 = done_5;
        (&isl_13)->info_0.y = (&isl_13)->info_0.y - done_5;
        if((*kernelContext_43->g_halt_0) != 0U)
        {
            _S875 = contact_island_4;
        }
        else
        {
            _S875 = false;
        }
        if(_S875)
        {
            contact_split_at_0((&isl_13)->info_0.w + done_5, kernelContext_43);
        }
        float _S914 = wsum_4.x;
        thread float _S915 = (&isl_13)->energy_0.x;
        thread float _S916 = (&isl_13)->energy_0.y;
        comp_add1_0(&_S915, &_S916, _S914);
        (&isl_13)->energy_0.x = _S915;
        (&isl_13)->energy_0.y = _S916 + wsum_4.y;
        (&isl_13)->info_0.w = (&isl_13)->info_0.w + done_5;
        if((((&isl_13)->info_0.x) & 8U) == 0U)
        {
            (&isl_13)->info_0.z = ((&isl_13)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_43->g_halt_0) != 0U)
        {
            (&isl_13)->info_0.z = ((&isl_13)->info_0.z) | 1U;
        }
        uint _S917 = seen_bits_0(kernelContext_43);
        if((*kernelContext_43->g_halt_0) != 0U)
        {
            ct_5 = 128U;
        }
        else
        {
            ct_5 = 0U;
        }
        uint _S918 = _S917 | ct_5;
        if(woke_8)
        {
            ct_5 = 2048U;
        }
        else
        {
            ct_5 = 0U;
        }
        uint _S919 = _S918 | ct_5;
        if(resumed_8)
        {
            ct_5 = 2097280U;
        }
        else
        {
            ct_5 = 0U;
        }
        uint cov_21 = _S919 | ct_5;
        if(done_5 > 0U)
        {
            bool quiet_9 = (((&isl_13)->info_0.x) & 32U) != 0U;
            if(quiet_9)
            {
                ct_5 = 4096U;
            }
            else
            {
                ct_5 = 0U;
            }
            uint cov_22 = cov_21 | ct_5;
            if((((&isl_13)->info_0.x) & 16U) != 0U)
            {
                _S875 = !quiet_9;
            }
            else
            {
                _S875 = false;
            }
            if(_S875)
            {
                ct_5 = 512U;
            }
            else
            {
                ct_5 = 0U;
            }
            uint cov_23 = cov_22 | ct_5;
            if((kernelContext_43->params_0->contact_mode_0) == 1U)
            {
                ct_5 = 524288U;
            }
            else
            {
                ct_5 = 0U;
            }
            uint cov_24 = cov_23 | ct_5;
            if(((&isl_13)->probes_0.y) > ((&isl_13)->probes_0.x))
            {
                ct_5 = 262144U;
            }
            else
            {
                ct_5 = 0U;
            }
            cov_20 = cov_24 | ct_5;
        }
        else
        {
            cov_20 = cov_21;
        }
        (&isl_13)->coverage_0 = ((&isl_13)->coverage_0) | cov_20;
        if(woke_8)
        {
            (&isl_13)->info_0.x = ((&isl_13)->info_0.x) & 4294967279U;
            (&isl_13)->info_0.z = ((&isl_13)->info_0.z) | 4U;
        }
        Island_natural_0 device* _S920 = kernelContext_43->islands_0+gi_4;
        _S920->range_0 = packed_uint4(isl_13.range_0) ;
        _S920->info_0 = packed_uint4(isl_13.info_0) ;
        _S920->com_0 = packed_float4(isl_13.com_0) ;
        _S920->inertia0_0 = packed_float4(isl_13.inertia0_0) ;
        _S920->inertia1_0 = packed_float4(isl_13.inertia1_0) ;
        _S920->inertia2_0 = packed_float4(isl_13.inertia2_0) ;
        _S920->inv0_0 = packed_float4(isl_13.inv0_0) ;
        _S920->inv1_0 = packed_float4(isl_13.inv1_0) ;
        _S920->inv2_0 = packed_float4(isl_13.inv2_0) ;
        _S920->wcom_0 = packed_float4(isl_13.wcom_0) ;
        _S920->winv0_0 = packed_float4(isl_13.winv0_0) ;
        _S920->winv1_0 = packed_float4(isl_13.winv1_0) ;
        _S920->winv2_0 = packed_float4(isl_13.winv2_0) ;
        _S920->rotation_0 = packed_float4(isl_13.rotation_0) ;
        _S920->position_0 = packed_float4(isl_13.position_0) ;
        _S920->position_err_0 = packed_float4(isl_13.position_err_0) ;
        _S920->velocity_0 = packed_float4(isl_13.velocity_0) ;
        _S920->velocity_err_0 = packed_float4(isl_13.velocity_err_0) ;
        _S920->angular_velocity_0 = packed_float4(isl_13.angular_velocity_0) ;
        _S920->done_0 = isl_13.done_0;
        _S920->coverage_0 = isl_13.coverage_0;
        _S920->wide_groups_0 = isl_13.wide_groups_0;
        _S920->done_pad_0 = isl_13.done_pad_0;
        _S920->probes_0 = packed_uint4(isl_13.probes_0) ;
        _S920->energy_0 = packed_float4(isl_13.energy_0) ;
        _S920->rotation_err_0 = packed_float4(isl_13.rotation_err_0) ;
        _S920->momentum_0 = packed_float4(isl_13.momentum_0) ;
        _S920->momentum_err_0 = packed_float4(isl_13.momentum_err_0) ;
    }
    return;
}

uint pair_lane_count_0(KernelContext_0 thread* kernelContext_44)
{
    return max(kernelContext_44->params_0->pair_lanes_0, 1U);
}

bool stopped_0(KernelContext_0 thread* kernelContext_45)
{
    uint4 _S921 = uint4((kernelContext_45->islands_0+kernelContext_45->params_0->halt_index_0)->info_0) ;
    bool _S922;
    if(((_S921.z) & 1U) != 0U)
    {
        _S922 = true;
    }
    else
    {
        _S922 = (_S921.y) != 0U;
    }
    return _S922;
}

bool stopped_1(KernelContext_0 thread* kernelContext_46)
{
    uint4 _S923 = uint4((kernelContext_46->islands_0+kernelContext_46->params_0->halt_index_0)->info_0) ;
    bool _S924;
    if(((_S923.z) & 1U) != 0U)
    {
        _S924 = true;
    }
    else
    {
        _S924 = (_S923.y) != 0U;
    }
    return _S924;
}

bool pair_any_0(bool x_9, uint lanes_0)
{
    bool _S925;
    if(lanes_0 > 1U)
    {
        bool _S926 = simd_any(x_9);
        _S925 = _S926;
    }
    else
    {
        _S925 = x_9;
    }
    return _S925;
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
        float3 _S927 = a_11->half_2;
        float3 _S928 = b_17->half_2;
        float margin_0 = 0.00000999999974738f * (length(a_11->half_2) + length(b_17->half_2));
        float3 _S929 = a_11->axis0_0;
        float3 _S930 = a_11->axis1_0;
        float3 _S931 = a_11->axis2_0;
        float3 _S932 = b_17->axis0_0;
        float3 _S933 = b_17->axis1_0;
        float3 _S934 = b_17->axis2_0;
        array<float3, int(6)> axes_0 = { { a_11->axis0_0, a_11->axis1_0, a_11->axis2_0, b_17->axis0_0, b_17->axis1_0, b_17->axis2_0 } };
        float3 l_3;
        if(i_7 < 6U)
        {
            l_3 = axes_0[i_7];
        }
        else
        {
            uint _S935 = i_7 - 6U;
            l_3 = cross(axes_0[_S935 / 3U], axes_0[3U + _S935 % 3U]);
        }
        float len_0 = length(l_3);
        if(len_0 <= 9.99999997475242708e-07f)
        {
            i_7 = i_7 + lanes_1;
            continue;
        }
        bool separated_1;
        if((abs(dot(d_4, l_3))) > (_S927.x * abs(dot(_S929, l_3)) + _S927.y * abs(dot(_S930, l_3)) + _S927.z * abs(dot(_S931, l_3)) + (_S928.x * abs(dot(_S932, l_3)) + _S928.y * abs(dot(_S933, l_3)) + _S928.z * abs(dot(_S934, l_3))) + margin_0 * len_0))
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
    bool _S936 = pair_any_0(separated_0, lanes_1);
    return !_S936;
}

bool may_overlap_0(const Box_0 thread* a_12, const Box_0 thread* b_18)
{
    float3 _S937 = b_18->center_1 - a_12->center_1;
    float3 _S938 = a_12->half_2;
    float3 _S939 = b_18->half_2;
    float _S940 = 0.00000999999974738f * (length(a_12->half_2) + length(b_18->half_2));
    float3 _S941 = a_12->axis0_0;
    float3 _S942 = a_12->axis1_0;
    float3 _S943 = a_12->axis2_0;
    float3 _S944 = b_18->axis0_0;
    float3 _S945 = b_18->axis1_0;
    float3 _S946 = b_18->axis2_0;
    array<float3, int(6)> _S947 = { { a_12->axis0_0, a_12->axis1_0, a_12->axis2_0, b_18->axis0_0, b_18->axis1_0, b_18->axis2_0 } };
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
            l_4 = _S947[i_8];
        }
        else
        {
            uint _S948 = i_8 - 6U;
            l_4 = cross(_S947[_S948 / 3U], _S947[3U + _S948 % 3U]);
        }
        float len_1 = length(l_4);
        if(len_1 <= 9.99999997475242708e-07f)
        {
            i_8 = i_8 + 1U;
            continue;
        }
        if((abs(dot(_S937, l_4))) > (_S938.x * abs(dot(_S941, l_4)) + _S938.y * abs(dot(_S942, l_4)) + _S938.z * abs(dot(_S943, l_4)) + (_S939.x * abs(dot(_S944, l_4)) + _S939.y * abs(dot(_S945, l_4)) + _S939.z * abs(dot(_S946, l_4))) + _S940 * len_1))
        {
            return false;
        }
        i_8 = i_8 + 1U;
    }
    return true;
}

float3 box_to_local_0(const Box_0 thread* b_19, float3 r_8)
{
    return float3(dot(r_8, b_19->axis0_0), dot(r_8, b_19->axis1_0), dot(r_8, b_19->axis2_0));
}

float hull_signed_distance_0(const Box_0 thread* b_20, float3 local_2, uint thread* face_0, KernelContext_0 thread* kernelContext_47)
{
    *face_0 = 0U;
    float best_0 = -1.00000001504746622e+30f;
    uint f_16 = 0U;
    for(;;)
    {
        if(f_16 < (b_20->hull_f_0))
        {
        }
        else
        {
            break;
        }
        float4 _S949 = float4(*(kernelContext_47->loads_0+(b_20->hull_at_0 + b_20->hull_v_0 + f_16))) ;
        float d_5 = dot(_S949.xyz, local_2) - _S949.w;
        if(d_5 > best_0)
        {
            *face_0 = f_16;
            best_0 = d_5;
        }
        f_16 = f_16 + 1U;
    }
    return best_0;
}

bool penetration_0(const Box_0 thread* b_21, float3 p_3, float thread* depth_1, float3 thread* normal_3, KernelContext_0 thread* kernelContext_48)
{
    *depth_1 = 0.0f;
    *normal_3 = float3(0.0f) ;
    float3 r_9 = p_3 - b_21->center_1;
    float3 _S950 = b_21->half_2;
    if((dot(r_9, r_9)) > (dot(b_21->half_2, b_21->half_2) * 1.00001001358032227f))
    {
        return false;
    }
    uint _S951 = b_21->hull_v_0;
    if((b_21->hull_v_0) != 0U)
    {
        float3 _S952 = box_to_local_0(b_21, r_9);
        thread uint face_1;
        float _S953 = hull_signed_distance_0(b_21, _S952, &face_1, kernelContext_48);
        if(!(_S953 < 0.0f))
        {
            return false;
        }
        *depth_1 = - _S953;
        float3 _S954 = box_to_world_0(b_21, (float4(*(kernelContext_48->loads_0+(b_21->hull_at_0 + _S951 + face_1))) ).xyz);
        *normal_3 = _S954;
        return true;
    }
    float best_1 = 1.00000001504746622e+30f;
    uint axis_3 = 0U;
    float side_0 = 1.0f;
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
        float3 _S955 = box_axis_0(b_21, k_17);
        float local_3 = dot(r_9, _S955);
        float d_6 = comp3_0(_S950, k_17) - abs(local_3);
        if(d_6 <= 0.0f)
        {
            return false;
        }
        if(d_6 < best_1)
        {
            float _S956;
            if(local_3 >= 0.0f)
            {
                _S956 = 1.0f;
            }
            else
            {
                _S956 = -1.0f;
            }
            best_1 = d_6;
            axis_3 = k_17;
            side_0 = _S956;
        }
        k_17 = k_17 + 1U;
    }
    *depth_1 = best_1;
    float3 _S957 = box_axis_0(b_21, axis_3);
    *normal_3 = _S957 * float3(side_0) ;
    return true;
}

bool pair_point_0(const Box_0 thread* ba_0, const Box_0 thread* bb_0, uint na_0, uint e_3, float3 thread* p_4, float3 thread* n_3, float thread* d_7, KernelContext_0 thread* kernelContext_49)
{
    bool from_a_0 = e_3 < na_0;
    Box_0 _S958;
    if(from_a_0)
    {
        _S958 = *ba_0;
    }
    else
    {
        _S958 = *bb_0;
    }
    uint _S959;
    if(from_a_0)
    {
        _S959 = e_3;
    }
    else
    {
        _S959 = e_3 - na_0;
    }
    thread Box_0 _S960 = _S958;
    float3 _S961 = sample_point_0(&_S960, _S959, kernelContext_49);
    *p_4 = _S961;
    if(from_a_0)
    {
        _S958 = *bb_0;
    }
    else
    {
        _S958 = *ba_0;
    }
    thread Box_0 _S962 = _S958;
    bool _S963 = penetration_0(&_S962, *p_4, d_7, n_3, kernelContext_49);
    if(!_S963)
    {
        return false;
    }
    if(!from_a_0)
    {
        *n_3 = - *n_3;
    }
    return true;
}

bool is_nan_0(float x_10)
{
    return ((as_type<uint>((x_10))) & 2147483647U) > 2139095040U;
}

uint pair_count_sum_0(uint x_11, uint lanes_2)
{
    uint _S964;
    if(lanes_2 > 1U)
    {
        uint _S965 = simd_sum(x_11);
        _S964 = _S965;
    }
    else
    {
        _S964 = x_11;
    }
    return _S964;
}

float3 pair_sum3_0(float3 x_12, uint lanes_3)
{
    float3 _S966;
    if(lanes_3 > 1U)
    {
        float3 _S967 = simd_sum(x_12);
        _S966 = _S967;
    }
    else
    {
        _S966 = x_12;
    }
    return _S966;
}

float pair_sum_0(float x_13, uint lanes_4)
{
    float _S968;
    if(lanes_4 > 1U)
    {
        float _S969 = simd_sum(x_13);
        _S968 = _S969;
    }
    else
    {
        _S968 = x_13;
    }
    return _S968;
}

void pair_contact_0(uint i_9, uint lane_1, uint lanes_5, KernelContext_0 thread* kernelContext_50)
{
    uint at_5 = kernelContext_50->params_0->pair_index_0 + 6U * i_9;
    uint ca_0 = kernelContext_50->index_0[at_5];
    uint cb_0 = kernelContext_50->index_0[at_5 + 1U];
    uint _S970 = kernelContext_50->index_0[at_5 + 3U];
    float _S971 = (as_type<float>((kernelContext_50->index_0[at_5 + 4U])));
    float _S972 = (as_type<float>((kernelContext_50->index_0[at_5 + 5U])));
    float _S973 = kernelContext_50->params_0->dt_0;
    uint out_0 = kernelContext_50->params_0->slot_base_0 + 2U * kernelContext_50->index_0[at_5 + 2U];
    WorldPoint_0 _S974 = chunk_world_0(ca_0, kernelContext_50);
    WorldPoint_0 _S975 = chunk_world_0(cb_0, kernelContext_50);
    thread WorldPoint_0 _S976 = _S975;
    thread WorldPoint_0 _S977 = _S974;
    float3 _S978 = world_diff_0(&_S976, &_S977);
    bool touching_0 = !((length(_S978)) > ((float4((kernelContext_50->chunks_0+ca_0)->half_0) ).w + (float4((kernelContext_50->chunks_0+cb_0)->half_0) ).w));
    float4 _S979 = float4(*(kernelContext_50->scratch_0+(kernelContext_50->params_0->ledger_base_0 + i_9))) ;
    thread float4 ledger_2 = _S979;
    uint flags_1 = (as_type<uint>((_S979.w)));
    float3 _S980 = float3(0.0f) ;
    Box_0 _S981 = chunk_box_0(ca_0, _S980, kernelContext_50);
    Box_0 _S982 = chunk_box_0(cb_0, _S978, kernelContext_50);
    thread Box_0 _S983 = _S981;
    uint _S984 = sample_count_0(&_S983);
    thread Box_0 _S985 = _S982;
    uint _S986 = sample_count_0(&_S985);
    uint _S987 = _S984 + _S986;
    bool has_state_0;
    uint e_4;
    if(!touching_0)
    {
        if((flags_1 & 1U) != 0U)
        {
            e_4 = lane_1;
            for(;;)
            {
                if(e_4 < _S987)
                {
                }
                else
                {
                    break;
                }
                *(kernelContext_50->contact_state_0+(_S970 + e_4)) = packed_float4(float4(float_bits_0(2143289344U), 0.0f, 0.0f, 0.0f)) ;
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
                *(kernelContext_50->scratch_0+(kernelContext_50->params_0->ledger_base_0 + i_9)) = packed_float4(ledger_2) ;
            }
        }
        return;
    }
    if(lanes_5 > 1U)
    {
        thread Box_0 _S988 = _S981;
        thread Box_0 _S989 = _S982;
        bool _S990 = may_overlap_lanes_0(&_S988, &_S989, lane_1, lanes_5);
        has_state_0 = _S990;
    }
    else
    {
        thread Box_0 _S991 = _S981;
        thread Box_0 _S992 = _S982;
        bool _S993 = may_overlap_0(&_S991, &_S992);
        has_state_0 = _S993;
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
            if(e_4 < _S987)
            {
            }
            else
            {
                break;
            }
            thread Box_0 _S994 = _S981;
            thread Box_0 _S995 = _S982;
            thread float3 p_5;
            thread float3 n_4;
            thread float d_8;
            bool _S996 = pair_point_0(&_S994, &_S995, _S984, e_4, &p_5, &n_4, &d_8, kernelContext_50);
            if(!_S996)
            {
                uint _S997 = _S970 + e_4;
                if(!is_nan_0((float4(*(kernelContext_50->contact_state_0+_S997)) ).x))
                {
                    *(kernelContext_50->contact_state_0+_S997) = packed_float4(float4(float_bits_0(2143289344U), 0.0f, 0.0f, 0.0f)) ;
                }
                e_4 = e_4 + lanes_5;
                continue;
            }
            uint _S998 = count_3 + 1U;
            uint _S999 = _S970 + e_4;
            float4 _S1000 = float4(*(kernelContext_50->contact_state_0+_S999)) ;
            thread float4 entry_3 = _S1000;
            if(is_nan_0(_S1000.x))
            {
                has_state_0 = true;
            }
            else
            {
                has_state_0 = (dot(entry_3.yzw, n_4)) < 0.99000000953674316f;
            }
            if(has_state_0)
            {
                if(d_8 > (2.0f * abs(dot(va0_0 + cross(wa0_0, p_5) - (vb0_0 + cross(wb0_0, p_5 - _S982.center_1)), n_4)) * _S973 + 9.99999971718068537e-10f))
                {
                    stored_sum_0 = d_8;
                }
                else
                {
                    stored_sum_0 = 0.0f;
                }
                entry_3 = float4(stored_sum_0, n_4);
            }
            entry_3.x = min(entry_3.x, d_8);
            *(kernelContext_50->contact_state_0+_S999) = packed_float4(entry_3) ;
            uint engaged_1;
            if((d_8 - entry_3.x) > 0.0f)
            {
                engaged_1 = engaged_0 + 1U;
            }
            else
            {
                engaged_1 = engaged_0;
            }
            count_3 = _S998;
            engaged_0 = engaged_1;
            e_4 = e_4 + lanes_5;
        }
        uint count_4 = pair_count_sum_0(count_3, lanes_5);
        uint engaged_2 = pair_count_sum_0(engaged_0, lanes_5);
        if(count_4 > 0U)
        {
            float _S1001 = (float4((kernelContext_50->chunks_0+ca_0)->cmat_0) ).x;
            float _S1002 = (float4((kernelContext_50->chunks_0+cb_0)->cmat_0) ).x;
            float3 _S1003 = _S982.center_1 - _S981.center_1;
            thread Box_0 _S1004 = _S981;
            thread Box_0 _S1005 = _S982;
            float _S1006 = contact_stiffness_0(_S1001, &_S1004, _S1002, &_S1005, _S1003);
            float _S1007 = _S1006 / max(float(engaged_2), 10.0f);
            e_4 = lane_1;
            fa_0 = _S980;
            ta_1 = _S980;
            fb_0 = _S980;
            tb_1 = _S980;
            stored_sum_0 = 0.0f;
            diss_sum_0 = 0.0f;
            for(;;)
            {
                if(e_4 < _S987)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S1008 = _S981;
                thread Box_0 _S1009 = _S982;
                thread float3 p_6;
                thread float3 n_5;
                thread float d_9;
                bool _S1010 = pair_point_0(&_S1008, &_S1009, _S984, e_4, &p_6, &n_5, &d_9, kernelContext_50);
                if(!_S1010)
                {
                    e_4 = e_4 + lanes_5;
                    continue;
                }
                float eff_0 = d_9 - (float4(*(kernelContext_50->contact_state_0+(_S970 + e_4))) ).x;
                if(eff_0 <= 0.0f)
                {
                    e_4 = e_4 + lanes_5;
                    continue;
                }
                thread float stored_6;
                thread float diss_4;
                float3 _S1011 = penalty_force_0(_S1007, _S971, _S972, eff_0, n_5, va0_0 + cross(wa0_0, p_6) - (vb0_0 + cross(wb0_0, p_6 - _S982.center_1)), _S973, engaged_2, &stored_6, &diss_4, kernelContext_50);
                float3 ta_2 = ta_1 + cross(p_6, _S1011);
                float3 _S1012 = - _S1011;
                float3 fb_1 = fb_0 + _S1012;
                float3 tb_2 = tb_1 + cross(p_6 - _S982.center_1, _S1012);
                float stored_sum_1 = stored_sum_0 + stored_6;
                float diss_sum_1 = diss_sum_0 + diss_4;
                fa_0 = fa_0 + _S1011;
                ta_1 = ta_2;
                fb_0 = fb_1;
                tb_1 = tb_2;
                stored_sum_0 = stored_sum_1;
                diss_sum_0 = diss_sum_1;
                e_4 = e_4 + lanes_5;
            }
            float3 _S1013 = pair_sum3_0(fa_0, lanes_5);
            float3 _S1014 = pair_sum3_0(ta_1, lanes_5);
            float3 _S1015 = pair_sum3_0(fb_0, lanes_5);
            float3 _S1016 = pair_sum3_0(tb_1, lanes_5);
            float _S1017 = pair_sum_0(stored_sum_0, lanes_5);
            float _S1018 = pair_sum_0(diss_sum_0, lanes_5);
            has_state_0 = true;
            diss_sum_0 = _S1018;
            fa_0 = _S1013;
            ta_1 = _S1014;
            fb_0 = _S1015;
            tb_1 = _S1016;
            stored_sum_0 = _S1017;
        }
        else
        {
            has_state_0 = false;
            diss_sum_0 = 0.0f;
            fa_0 = _S980;
            ta_1 = _S980;
            fb_0 = _S980;
            tb_1 = _S980;
            stored_sum_0 = 0.0f;
        }
    }
    else
    {
        has_state_0 = false;
        diss_sum_0 = 0.0f;
        fa_0 = _S980;
        ta_1 = _S980;
        fb_0 = _S980;
        tb_1 = _S980;
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
            if(e_4 < _S987)
            {
            }
            else
            {
                break;
            }
            *(kernelContext_50->contact_state_0+(_S970 + e_4)) = packed_float4(float4(float_bits_0(2143289344U), 0.0f, 0.0f, 0.0f)) ;
            e_4 = e_4 + lanes_5;
        }
    }
    if(lane_1 != 0U)
    {
        return;
    }
    if(has_state_0)
    {
        thread float _S1019 = ledger_2.y;
        thread float _S1020 = ledger_2.z;
        comp_add1_0(&_S1019, &_S1020, diss_sum_0);
        ledger_2.z = _S1020;
        ledger_2.y = _S1019;
    }
    float3 _S1021 = float3(0.0f) ;
    if(any(fa_0 != _S1021))
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = any(ta_1 != _S1021);
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = any(fb_0 != _S1021);
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = any(tb_1 != _S1021);
    }
    bool _S1022;
    if(loaded_0)
    {
        _S1022 = true;
    }
    else
    {
        _S1022 = (flags_1 & 2U) != 0U;
    }
    if(_S1022)
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
    *(kernelContext_50->scratch_0+(kernelContext_50->params_0->ledger_base_0 + i_9)) = packed_float4(ledger_2) ;
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
    thread Box_0 b_22;
    (&b_22)->center_1 = center_3;
    float3 _S1023 = float3(1.0f, 0.0f, 0.0f);
    thread Quat_0 _S1024 = q_11;
    float3 _S1025 = rotate_0(&_S1024, _S1023);
    (&b_22)->axis0_0 = _S1025;
    float3 _S1026 = float3(0.0f, 1.0f, 0.0f);
    thread Quat_0 _S1027 = q_11;
    float3 _S1028 = rotate_0(&_S1027, _S1026);
    (&b_22)->axis1_0 = _S1028;
    float3 _S1029 = float3(0.0f, 0.0f, 1.0f);
    thread Quat_0 _S1030 = q_11;
    float3 _S1031 = rotate_0(&_S1030, _S1029);
    (&b_22)->axis2_0 = _S1031;
    (&b_22)->half_2 = half_3;
    (&b_22)->hull_at_0 = 0U;
    (&b_22)->hull_v_0 = 0U;
    (&b_22)->hull_f_0 = 0U;
    return b_22;
}

uint impactor_slots_0(const Impactor_natural_0 thread* imp_1, const Box_0 thread* b_23)
{
    uint _S1032;
    if(((float4(imp_1->shape_0) ).x) == 0.0f)
    {
        _S1032 = 1U;
    }
    else
    {
        uint _S1033 = sample_count_0(b_23);
        _S1032 = _S1033 + 14U;
    }
    return _S1032;
}

uint impactor_slots_1(const Impactor_natural_0 thread* imp_2, const Box_0 thread* b_24)
{
    uint _S1034;
    if(((float4(imp_2->shape_0) ).x) == 0.0f)
    {
        _S1034 = 1U;
    }
    else
    {
        uint _S1035 = sample_count_0(b_24);
        _S1034 = _S1035 + 14U;
    }
    return _S1034;
}

bool sphere_contact_0(const Box_0 thread* b_25, float3 center_4, float radius_0, float3 thread* point_0, float3 thread* normal_4, float thread* depth_2, KernelContext_0 thread* kernelContext_51)
{
    float3 _S1036 = float3(0.0f) ;
    *point_0 = _S1036;
    *normal_4 = _S1036;
    *depth_2 = 0.0f;
    float3 _S1037 = b_25->center_1;
    float3 r_10 = center_4 - b_25->center_1;
    float3 _S1038 = b_25->axis0_0;
    float3 _S1039 = b_25->axis1_0;
    float3 _S1040 = b_25->axis2_0;
    float3 local_4 = float3(dot(r_10, b_25->axis0_0), dot(r_10, b_25->axis1_0), dot(r_10, b_25->axis2_0));
    uint _S1041 = b_25->hull_v_0;
    if((b_25->hull_v_0) != 0U)
    {
        thread uint face_2;
        float _S1042 = hull_signed_distance_0(b_25, local_4, &face_2, kernelContext_51);
        if(_S1042 >= radius_0)
        {
            return false;
        }
        float3 _S1043 = box_to_world_0(b_25, (float4(*(kernelContext_51->loads_0+(b_25->hull_at_0 + _S1041 + face_2))) ).xyz);
        *normal_4 = _S1043;
        *point_0 = center_4 - _S1043 * float3(max(_S1042, 0.0f)) ;
        *depth_2 = radius_0 - _S1042;
        return true;
    }
    float3 q_12 = clamp(local_4, - b_25->half_2, b_25->half_2);
    float3 d_10 = local_4 - q_12;
    float dist_0 = length(d_10);
    if(dist_0 > 9.999999960041972e-13f)
    {
        if(dist_0 >= radius_0)
        {
            return false;
        }
        float3 dn_0 = d_10 / float3(dist_0) ;
        *normal_4 = _S1038 * float3(dn_0.x)  + _S1039 * float3(dn_0.y)  + _S1040 * float3(dn_0.z) ;
        *point_0 = _S1037 + _S1038 * float3(q_12.x)  + _S1039 * float3(q_12.y)  + _S1040 * float3(q_12.z) ;
        *depth_2 = radius_0 - dist_0;
        return true;
    }
    thread float inside_0;
    thread float3 n_6;
    bool _S1044 = penetration_0(b_25, center_4, &inside_0, &n_6, kernelContext_51);
    if(!_S1044)
    {
        return false;
    }
    *normal_4 = n_6;
    *point_0 = center_4 - n_6 * float3(min(radius_0, inside_0)) ;
    *depth_2 = radius_0 + inside_0;
    return true;
}

bool impactor_contact_0(const Impactor_natural_0 thread* imp_3, float crush_depth_0, const Box_0 thread* shrunk_0, const Box_0 thread* b_26, uint j_2, float3 thread* p_7, float3 thread* n_7, float thread* d_11, KernelContext_0 thread* kernelContext_52)
{
    float3 _S1045 = float3(0.0f) ;
    *p_7 = _S1045;
    *n_7 = _S1045;
    *d_11 = 0.0f;
    float4 _S1046 = float4(imp_3->shape_0) ;
    if((_S1046.x) == 0.0f)
    {
        bool _S1047 = sphere_contact_0(b_26, _S1045, _S1046.y - crush_depth_0, p_7, n_7, d_11, kernelContext_52);
        if(!_S1047)
        {
            return false;
        }
        *n_7 = - *n_7;
        return true;
    }
    uint _S1048 = sample_count_0(b_26);
    if(j_2 < _S1048)
    {
        float3 _S1049 = sample_point_0(b_26, j_2, kernelContext_52);
        *p_7 = _S1049;
        bool _S1050 = penetration_0(shrunk_0, _S1049, d_11, n_7, kernelContext_52);
        return _S1050;
    }
    float3 _S1051 = sample_point_0(shrunk_0, j_2 - _S1048, kernelContext_52);
    *p_7 = _S1051;
    bool _S1052 = penetration_0(b_26, _S1051, d_11, n_7, kernelContext_52);
    if(!_S1052)
    {
        return false;
    }
    *n_7 = - *n_7;
    return true;
}

bool impactor_contact_1(const Impactor_natural_0 thread* imp_4, float crush_depth_1, const Box_0 thread* shrunk_1, const Box_0 thread* b_27, uint j_3, float3 thread* p_8, float3 thread* n_8, float thread* d_12, KernelContext_0 thread* kernelContext_53)
{
    float3 _S1053 = float3(0.0f) ;
    *p_8 = _S1053;
    *n_8 = _S1053;
    *d_12 = 0.0f;
    float4 _S1054 = float4(imp_4->shape_0) ;
    if((_S1054.x) == 0.0f)
    {
        bool _S1055 = sphere_contact_0(b_27, _S1053, _S1054.y - crush_depth_1, p_8, n_8, d_12, kernelContext_53);
        if(!_S1055)
        {
            return false;
        }
        *n_8 = - *n_8;
        return true;
    }
    uint _S1056 = sample_count_0(b_27);
    if(j_3 < _S1056)
    {
        float3 _S1057 = sample_point_0(b_27, j_3, kernelContext_53);
        *p_8 = _S1057;
        bool _S1058 = penetration_0(shrunk_1, _S1057, d_12, n_8, kernelContext_53);
        return _S1058;
    }
    float3 _S1059 = sample_point_0(shrunk_1, j_3 - _S1056, kernelContext_53);
    *p_8 = _S1059;
    bool _S1060 = penetration_0(b_27, _S1059, d_12, n_8, kernelContext_53);
    if(!_S1060)
    {
        return false;
    }
    *n_8 = - *n_8;
    return true;
}

WorldPoint_0 impactor_point_0(uint _S1061, KernelContext_0 thread* kernelContext_54)
{
    Impactor_natural_0 device* _S1062 = kernelContext_54->impactors_0+_S1061;
    thread WorldPoint_0 wi_0;
    (&wi_0)->hi_0 = (float4(_S1062->position_1) ).xyz;
    (&wi_0)->lo_0 = (float4(_S1062->position_err_1) ).xyz;
    (&wi_0)->rel_0 = float3(0.0f) ;
    return wi_0;
}

Box_0 impactor_box_1(uint _S1063, float3 _S1064, float3 _S1065, KernelContext_0 thread* kernelContext_55)
{
    Quat_0 q_13 = quat_of_0(float4((kernelContext_55->impactors_0+_S1063)->rotation_1) );
    thread Box_0 b_28;
    (&b_28)->center_1 = _S1064;
    float3 _S1066 = float3(1.0f, 0.0f, 0.0f);
    thread Quat_0 _S1067 = q_13;
    float3 _S1068 = rotate_0(&_S1067, _S1066);
    (&b_28)->axis0_0 = _S1068;
    float3 _S1069 = float3(0.0f, 1.0f, 0.0f);
    thread Quat_0 _S1070 = q_13;
    float3 _S1071 = rotate_0(&_S1070, _S1069);
    (&b_28)->axis1_0 = _S1071;
    float3 _S1072 = float3(0.0f, 0.0f, 1.0f);
    thread Quat_0 _S1073 = q_13;
    float3 _S1074 = rotate_0(&_S1073, _S1072);
    (&b_28)->axis2_0 = _S1074;
    (&b_28)->half_2 = _S1065;
    (&b_28)->hull_at_0 = 0U;
    (&b_28)->hull_v_0 = 0U;
    (&b_28)->hull_f_0 = 0U;
    return b_28;
}

Box_0 impactor_shrunk_0(uint _S1075, float _S1076, const Box_0 thread* _S1077)
{
    thread Box_0 shrunk_2 = *_S1077;
    (&shrunk_2)->half_2 = _S1077->half_2 - min(float3(_S1076) , _S1077->half_2 * float3(0.5f) );
    return shrunk_2;
}

bool impactor_contact_2(uint _S1078, float _S1079, const Box_0 thread* _S1080, const Box_0 thread* _S1081, uint _S1082, float3 thread* _S1083, float3 thread* _S1084, float thread* _S1085, KernelContext_0 thread* kernelContext_56)
{
    Impactor_natural_0 _S1086 = *(kernelContext_56->impactors_0+_S1078);
    float3 _S1087 = float3(0.0f) ;
    *_S1083 = _S1087;
    *_S1084 = _S1087;
    *_S1085 = 0.0f;
    float4 _S1088 = float4(_S1086.shape_0) ;
    if((_S1088.x) == 0.0f)
    {
        bool _S1089 = sphere_contact_0(_S1081, _S1087, _S1088.y - _S1079, _S1083, _S1084, _S1085, kernelContext_56);
        if(!_S1089)
        {
            return false;
        }
        *_S1084 = - *_S1084;
        return true;
    }
    uint _S1090 = sample_count_0(_S1081);
    if(_S1082 < _S1090)
    {
        float3 _S1091 = sample_point_0(_S1081, _S1082, kernelContext_56);
        *_S1083 = _S1091;
        bool _S1092 = penetration_0(_S1080, _S1091, _S1085, _S1084, kernelContext_56);
        return _S1092;
    }
    float3 _S1093 = sample_point_0(_S1080, _S1082 - _S1090, kernelContext_56);
    *_S1083 = _S1093;
    bool _S1094 = penetration_0(_S1081, _S1093, _S1085, _S1084, kernelContext_56);
    if(!_S1094)
    {
        return false;
    }
    *_S1084 = - *_S1084;
    return true;
}

void impactor_candidate_forces_0(uint k_18, uint lane_2, uint lanes_6, KernelContext_0 thread* kernelContext_57)
{
    uint _S1095;
    bool _S1096;
    uint _S1097 = 3U * k_18;
    uint at_6 = kernelContext_57->params_0->cand_index_0 + _S1097;
    uint c_33 = kernelContext_57->index_0[at_6];
    uint slot_1 = kernelContext_57->index_0[at_6 + 1U];
    uint _S1098 = kernelContext_57->index_0[at_6 + 2U];
    thread Impactor_natural_0 _S1099 = *(kernelContext_57->impactors_0+_S1098);
    float _S1100 = kernelContext_57->params_0->dt_0;
    float3 _S1101 = float3(0.0f) ;
    thread float4 data_0 = float4(*(kernelContext_57->scratch_0+(kernelContext_57->params_0->cand_base_0 + _S1097))) ;
    bool hit_0;
    float3 f_sum_0;
    float3 t_sum_0;
    float3 imp_f_0;
    float3 imp_t_0;
    if(((uint4((&_S1099)->cand_0) ).z) == 0U)
    {
        WorldPoint_0 _S1102 = chunk_world_0(c_33, kernelContext_57);
        WorldPoint_0 _S1103 = impactor_point_0(_S1098, kernelContext_57);
        thread WorldPoint_0 _S1104 = _S1102;
        thread WorldPoint_0 _S1105 = _S1103;
        float3 _S1106 = world_diff_0(&_S1104, &_S1105);
        float4 _S1107 = float4((&_S1099)->half_1) ;
        if(!((length(_S1106)) > (_S1107.w + (float4((kernelContext_57->chunks_0+c_33)->half_0) ).w)))
        {
            Box_0 _S1108 = impactor_box_1(_S1098, _S1101, _S1107.xyz, kernelContext_57);
            Box_0 _S1109 = chunk_box_0(c_33, _S1106, kernelContext_57);
            float4 _S1110 = float4((&_S1099)->mat_0) ;
            float _S1111 = _S1110.x;
            float _S1112 = (float4((kernelContext_57->chunks_0+c_33)->cmat_0) ).x;
            float3 _S1113 = _S1109.center_1 - _S1108.center_1;
            thread Box_0 _S1114 = _S1108;
            thread Box_0 _S1115 = _S1109;
            float _S1116 = contact_stiffness_0(_S1111, &_S1114, _S1112, &_S1115, _S1113);
            float4 _S1117 = float4((&_S1099)->geom_0) ;
            float _S1118 = _S1117.y;
            thread Box_0 _S1119 = _S1108;
            Box_0 _S1120 = impactor_shrunk_0(_S1098, _S1118, &_S1119);
            uint j_4 = lane_2;
            uint n_9 = 0U;
            for(;;)
            {
                thread Box_0 _S1121 = _S1109;
                uint _S1122 = impactor_slots_0(&_S1099, &_S1121);
                _S1095 = _S1122;
                if(j_4 < _S1122)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S1123 = _S1120;
                thread Box_0 _S1124 = _S1109;
                thread float3 p_9;
                thread float3 nrm_0;
                thread float dep_0;
                bool _S1125 = impactor_contact_2(_S1098, _S1118, &_S1123, &_S1124, j_4, &p_9, &nrm_0, &dep_0, kernelContext_57);
                if(_S1125)
                {
                    n_9 = n_9 + 1U;
                }
                j_4 = j_4 + lanes_6;
            }
            uint n_10 = pair_count_sum_0(n_9, lanes_6);
            bool _S1126 = ((float4((&_S1099)->shape_0) ).x) == 0.0f;
            float _S1127;
            if(_S1126)
            {
                _S1127 = _S1116;
            }
            else
            {
                _S1127 = _S1116 / max(float(n_10), 10.0f);
            }
            if(_S1126)
            {
                j_4 = 1U;
            }
            else
            {
                j_4 = n_10;
            }
            float m_4 = (float4((kernelContext_57->chunks_0+c_33)->center_0) ).w;
            float _S1128 = _S1110.z;
            float _S1129 = m_4 * _S1128 / (m_4 + _S1128);
            float _S1130;
            if((kernelContext_57->params_0->pair_friction_0) >= 0.0f)
            {
                _S1130 = kernelContext_57->params_0->pair_friction_0;
            }
            else
            {
                _S1130 = min(_S1110.y, (float4((kernelContext_57->chunks_0+c_33)->cmat_0) ).y);
            }
            float3 _S1131 = (float4((&_S1099)->velocity_1) ).xyz + (float4((&_S1099)->velocity_err_1) ).xyz;
            thread float3 vc_1;
            thread float3 wc_1;
            chunk_velocity_0(c_33, &vc_1, &wc_1, kernelContext_57);
            hit_0 = false;
            uint j_5 = lane_2;
            f_sum_0 = _S1101;
            t_sum_0 = _S1101;
            imp_f_0 = _S1101;
            imp_t_0 = _S1101;
            float diss_sum_2 = 0.0f;
            for(;;)
            {
                bool _S1132 = n_10 > 0U;
                _S1096 = _S1132;
                bool _S1133;
                if(_S1132)
                {
                    _S1133 = j_5 < _S1095;
                }
                else
                {
                    _S1133 = false;
                }
                if(_S1133)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S1134 = _S1120;
                thread Box_0 _S1135 = _S1109;
                thread float3 p_10;
                thread float3 nrm_1;
                thread float dep_1;
                bool _S1136 = impactor_contact_0(&_S1099, _S1118, &_S1134, &_S1135, j_5, &p_10, &nrm_1, &dep_1, kernelContext_57);
                if(!_S1136)
                {
                    j_5 = j_5 + lanes_6;
                    continue;
                }
                thread float stored_7;
                thread float diss_5;
                float3 _S1137 = penalty_force_0(_S1127, _S1129, _S1130, dep_1 * _S1117.x, nrm_1, vc_1 + cross(wc_1, p_10 - _S1109.center_1) - (_S1131 + cross((float4((&_S1099)->angular_velocity_1) ).xyz, p_10)), _S1100, j_4, &stored_7, &diss_5, kernelContext_57);
                float3 f_sum_1 = f_sum_0 + _S1137;
                float3 t_sum_1 = t_sum_0 + cross(p_10 - _S1109.center_1, _S1137);
                float3 imp_f_1 = imp_f_0 - _S1137;
                float3 imp_t_1 = imp_t_0 - cross(p_10, _S1137);
                float diss_sum_3 = diss_sum_2 + diss_5;
                hit_0 = true;
                f_sum_0 = f_sum_1;
                t_sum_0 = t_sum_1;
                imp_f_0 = imp_f_1;
                imp_t_0 = imp_t_1;
                diss_sum_2 = diss_sum_3;
                j_5 = j_5 + lanes_6;
            }
            float3 _S1138 = pair_sum3_0(f_sum_0, lanes_6);
            float3 _S1139 = pair_sum3_0(t_sum_0, lanes_6);
            float3 _S1140 = pair_sum3_0(imp_f_0, lanes_6);
            float3 _S1141 = pair_sum3_0(imp_t_0, lanes_6);
            float diss_sum_4 = pair_sum_0(diss_sum_2, lanes_6);
            bool _S1142 = pair_any_0(hit_0, lanes_6);
            if(_S1096)
            {
                thread float _S1143 = data_0.z;
                thread float _S1144 = data_0.w;
                comp_add1_0(&_S1143, &_S1144, diss_sum_4);
                data_0.w = _S1144;
                data_0.z = _S1143;
            }
            hit_0 = _S1142;
            f_sum_0 = _S1138;
            t_sum_0 = _S1139;
            imp_f_0 = _S1140;
            imp_t_0 = _S1141;
        }
        else
        {
            hit_0 = false;
            f_sum_0 = _S1101;
            t_sum_0 = _S1101;
            imp_f_0 = _S1101;
            imp_t_0 = _S1101;
        }
    }
    else
    {
        hit_0 = false;
        f_sum_0 = _S1101;
        t_sum_0 = _S1101;
        imp_f_0 = _S1101;
        imp_t_0 = _S1101;
    }
    if(lane_2 != 0U)
    {
        return;
    }
    uint _S1145 = 2U * slot_1;
    *(kernelContext_57->scratch_0+(kernelContext_57->params_0->slot_base_0 + _S1145)) = packed_float4(float4(f_sum_0, 0.0f)) ;
    *(kernelContext_57->scratch_0+(kernelContext_57->params_0->slot_base_0 + _S1145 + 1U)) = packed_float4(float4(t_sum_0, 0.0f)) ;
    *(kernelContext_57->scratch_0+(kernelContext_57->params_0->cand_base_0 + _S1097)) = packed_float4(data_0) ;
    *(kernelContext_57->scratch_0+(kernelContext_57->params_0->cand_base_0 + _S1097 + 1U)) = packed_float4(float4(imp_f_0, 0.0f)) ;
    *(kernelContext_57->scratch_0+(kernelContext_57->params_0->cand_base_0 + _S1097 + 2U)) = packed_float4(float4(imp_t_0, 0.0f)) ;
    if(hit_0)
    {
        sentinel_mark_0(2U, kernelContext_57);
    }
    return;
}

void impactor_travel_check_0(uint ii_0, bool fused_2, KernelContext_0 thread* kernelContext_58)
{
    Impactor_natural_0 device* _S1146 = kernelContext_58->impactors_0+ii_0;
    uint at_7 = kernelContext_58->params_0->iplan_base_0 + 4U * ii_0;
    float4 _S1147 = float4(*(kernelContext_58->scratch_0+at_7)) ;
    uint4 _S1148 = uint4(_S1146->cand_0) ;
    bool _S1149;
    if((_S1148.z) != 0U)
    {
        _S1149 = true;
    }
    else
    {
        _S1149 = (_S1147.w) <= 0.0f;
    }
    if(_S1149)
    {
        return;
    }
    float4 _S1150 = float4(*(kernelContext_58->scratch_0+(at_7 + 1U))) ;
    float4 _S1151 = float4(*(kernelContext_58->scratch_0+(at_7 + 2U))) ;
    float4 _S1152 = float4(*(kernelContext_58->scratch_0+(at_7 + 3U))) ;
    uint _S1153 = _S1148.w;
    float3 moved_1 = (float4(_S1146->position_1) ).xyz - _S1147.xyz + ((float4(_S1146->position_err_1) ).xyz - _S1150.xyz);
    float3 _S1154 = float3((kernelContext_58->params_0->t_hi_0 - _S1152.w + (kernelContext_58->params_0->t_lo_0 - _S1150.w) + float(_S1153 - kernelContext_58->params_0->step_start_0) * kernelContext_58->params_0->dt_0)) ;
    float3 expected_0 = _S1152.xyz * _S1154 + float3(0.5f)  * kernelContext_58->params_0->gravity_0.xyz * _S1154 * _S1154;
    Quat_0 _S1155 = quat_of_0(float4(_S1146->rotation_1) );
    thread Quat_0 _S1156 = quat_of_0(float4(- _S1151.xyz, _S1151.w));
    thread Quat_0 _S1157 = _S1155;
    Quat_0 _S1158 = quat_mul_0(&_S1156, &_S1157);
    if((length(moved_1 - expected_0) + 2.0f * atan2(length(float3(_S1158.x_0, _S1158.y_0, _S1158.z_0)), abs(_S1158.w_0)) * (float4(_S1146->half_1) ).w) > (_S1147.w))
    {
        travel_stop_0(_S1153 + 1U, fused_2, kernelContext_58);
    }
    return;
}

void contact_forces_at_0(uint i_10, bool fused_3, KernelContext_0 thread* kernelContext_59)
{
    uint _S1159 = pair_lane_count_0(kernelContext_59);
    uint pair_threads_0 = kernelContext_59->params_0->pair_count_0 * _S1159;
    if(i_10 < pair_threads_0)
    {
        bool _S1160 = stopped_0(kernelContext_59);
        bool _S1161 = pair_any_0(_S1160, _S1159);
        if(!_S1161)
        {
            uint _S1162 = i_10 / _S1159;
            uint _S1163 = i_10 % _S1159;
            pair_contact_0(_S1162, _S1163, _S1159, kernelContext_59);
        }
        return;
    }
    uint j_6 = i_10 - pair_threads_0;
    uint cand_threads_0 = kernelContext_59->params_0->cand_count_0 * _S1159;
    if(j_6 < cand_threads_0)
    {
        bool _S1164 = stopped_0(kernelContext_59);
        bool _S1165 = pair_any_0(_S1164, _S1159);
        if(!_S1165)
        {
            uint _S1166 = j_6 / _S1159;
            uint _S1167 = j_6 % _S1159;
            impactor_candidate_forces_0(_S1166, _S1167, _S1159, kernelContext_59);
        }
        return;
    }
    if(fused_3)
    {
        return;
    }
    uint _S1168 = j_6 - cand_threads_0;
    bool _S1169;
    if(_S1168 < (kernelContext_59->params_0->chunk_count_0))
    {
        bool _S1170 = stopped_0(kernelContext_59);
        _S1169 = !_S1170;
    }
    else
    {
        _S1169 = false;
    }
    if(_S1169)
    {
        travel_check_0(_S1168, false, kernelContext_59);
    }
    else
    {
        if((_S1168 - kernelContext_59->params_0->chunk_count_0) < (kernelContext_59->params_0->impactor_count_0))
        {
            bool _S1171 = stopped_0(kernelContext_59);
            _S1169 = !_S1171;
        }
        else
        {
            _S1169 = false;
        }
        if(_S1169)
        {
            impactor_travel_check_0(_S1168 - kernelContext_59->params_0->chunk_count_0, false, kernelContext_59);
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

uint impactor_contact_count_0(uint _S1172, float _S1173, const Box_0 thread* _S1174, const Box_0 thread* _S1175, KernelContext_0 thread* kernelContext_61)
{
    thread Impactor_natural_0 _S1176 = *(kernelContext_61->impactors_0+_S1172);
    uint j_7 = 0U;
    uint count_5 = 0U;
    for(;;)
    {
        uint _S1177 = impactor_slots_1(&_S1176, _S1175);
        if(j_7 < _S1177)
        {
        }
        else
        {
            break;
        }
        thread float3 p_11;
        thread float3 n_11;
        thread float d_13;
        bool _S1178 = impactor_contact_2(_S1172, _S1173, _S1174, _S1175, j_7, &p_11, &n_11, &d_13, kernelContext_61);
        if(_S1178)
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
    bool _S1179;
    if(k_19 >= (params_2->cand_count_0))
    {
        _S1179 = true;
    }
    else
    {
        bool _S1180 = stopped_0(&kernelContext_62);
        _S1179 = _S1180;
    }
    if(_S1179)
    {
        return;
    }
    uint _S1181 = 3U * k_19;
    uint at_8 = (&kernelContext_62)->params_0->cand_index_0 + _S1181;
    uint c_34 = (&kernelContext_62)->index_0[at_8];
    uint _S1182 = (&kernelContext_62)->index_0[at_8 + 2U];
    thread Impactor_natural_0 _S1183 = *((&kernelContext_62)->impactors_0+_S1182);
    if(((uint4((&_S1183)->cand_0) ).z) == 0U)
    {
        _S1179 = ((float4((&_S1183)->crush_0) ).x) > 0.0f;
    }
    else
    {
        _S1179 = false;
    }
    float total_1;
    float ksum_0;
    if(_S1179)
    {
        WorldPoint_0 _S1184 = chunk_world_0(c_34, &kernelContext_62);
        WorldPoint_0 _S1185 = impactor_point_0(_S1182, &kernelContext_62);
        thread WorldPoint_0 _S1186 = _S1184;
        thread WorldPoint_0 _S1187 = _S1185;
        float3 _S1188 = world_diff_0(&_S1186, &_S1187);
        float4 _S1189 = float4((&_S1183)->half_1) ;
        if(!((length(_S1188)) > (_S1189.w + (float4(((&kernelContext_62)->chunks_0+c_34)->half_0) ).w)))
        {
            Box_0 _S1190 = impactor_box_1(_S1182, float3(0.0f) , _S1189.xyz, &kernelContext_62);
            Box_0 _S1191 = chunk_box_0(c_34, _S1188, &kernelContext_62);
            float _S1192 = (float4((&_S1183)->mat_0) ).x;
            float _S1193 = (float4(((&kernelContext_62)->chunks_0+c_34)->cmat_0) ).x;
            float3 _S1194 = _S1191.center_1 - _S1190.center_1;
            thread Box_0 _S1195 = _S1190;
            thread Box_0 _S1196 = _S1191;
            float _S1197 = contact_stiffness_0(_S1192, &_S1195, _S1193, &_S1196, _S1194);
            float _S1198 = (float4((&_S1183)->crush_0) ).w;
            thread Box_0 _S1199 = _S1190;
            Box_0 _S1200 = impactor_shrunk_0(_S1182, _S1198, &_S1199);
            thread Box_0 _S1201 = _S1200;
            thread Box_0 _S1202 = _S1191;
            uint _S1203 = impactor_contact_count_0(_S1182, _S1198, &_S1201, &_S1202, &kernelContext_62);
            float _S1204;
            if(((float4((&_S1183)->shape_0) ).x) == 0.0f)
            {
                _S1204 = _S1197;
            }
            else
            {
                _S1204 = _S1197 / max(float(_S1203), 10.0f);
            }
            uint j_8 = 0U;
            total_1 = 0.0f;
            ksum_0 = 0.0f;
            for(;;)
            {
                if(_S1203 > 0U)
                {
                    thread Box_0 _S1205 = _S1191;
                    uint _S1206 = impactor_slots_0(&_S1183, &_S1205);
                    _S1179 = j_8 < _S1206;
                }
                else
                {
                    _S1179 = false;
                }
                if(_S1179)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S1207 = _S1200;
                thread Box_0 _S1208 = _S1191;
                thread float3 p_12;
                thread float3 nrm_2;
                thread float dep_2;
                bool _S1209 = impactor_contact_1(&_S1183, _S1198, &_S1207, &_S1208, j_8, &p_12, &nrm_2, &dep_2, &kernelContext_62);
                if(!_S1209)
                {
                    j_8 = j_8 + 1U;
                    continue;
                }
                float ksum_1 = ksum_0 + _S1204;
                total_1 = total_1 + _S1204 * dep_2;
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
    float4 _S1210 = float4(*((&kernelContext_62)->scratch_0+((&kernelContext_62)->params_0->cand_base_0 + _S1181))) ;
    *((&kernelContext_62)->scratch_0+((&kernelContext_62)->params_0->cand_base_0 + _S1181)) = packed_float4(float4(total_1, ksum_0, _S1210.z, _S1210.w)) ;
    return;
}

bool stopped_uniform_0(uint tid_9, KernelContext_0 thread* kernelContext_63)
{
    bool _S1211 = stopped_1(kernelContext_63);
    uint _S1212;
    if(_S1211)
    {
        _S1212 = 1U;
    }
    else
    {
        _S1212 = 0U;
    }
    return (group_uniform_0(_S1212, tid_9)) != 0U;
}

[[kernel]] void impactor_crush(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], Params_0 constant* params_3 [[buffer(0)]], Island_natural_0 device* islands_3 [[buffer(9)]], uint device* index_3 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(3)]], packed_float4 device* state_5 [[buffer(6)]], packed_float4 device* scratch_3 [[buffer(8)]], packed_float4 device* contact_state_3 [[buffer(11)]], packed_float4 device* loads_3 [[buffer(5)]], Impactor_natural_0 device* impactors_3 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_3 [[buffer(7)]], BondStatic_natural_0 device* bonds_3 [[buffer(2)]], MaterialTable_0 constant* materials_3 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_64;
    (&kernelContext_64)->params_0 = params_3;
    (&kernelContext_64)->islands_0 = islands_3;
    (&kernelContext_64)->index_0 = index_3;
    (&kernelContext_64)->chunks_0 = chunks_3;
    (&kernelContext_64)->state_0 = state_5;
    (&kernelContext_64)->scratch_0 = scratch_3;
    (&kernelContext_64)->contact_state_0 = contact_state_3;
    (&kernelContext_64)->loads_0 = loads_3;
    (&kernelContext_64)->impactors_0 = impactors_3;
    (&kernelContext_64)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_64)->bonds_0 = bonds_3;
    (&kernelContext_64)->materials_0 = materials_3;
    threadgroup array<float4, int(32)> g_part_a_3;
    (&kernelContext_64)->g_part_a_0 = &g_part_a_3;
    threadgroup array<float4, int(32)> g_part_b_3;
    (&kernelContext_64)->g_part_b_0 = &g_part_b_3;
    threadgroup array<uint, int(5)> g_seen_3;
    (&kernelContext_64)->g_seen_0 = &g_seen_3;
    threadgroup uint g_wide_run_3;
    (&kernelContext_64)->g_wide_run_0 = &g_wide_run_3;
    threadgroup array<uint, int(4096)> g_label_3;
    (&kernelContext_64)->g_label_0 = &g_label_3;
    threadgroup uint g_changed_3;
    (&kernelContext_64)->g_changed_0 = &g_changed_3;
    threadgroup uint g_halt_3;
    (&kernelContext_64)->g_halt_0 = &g_halt_3;
    threadgroup uint g_run_3;
    (&kernelContext_64)->g_run_0 = &g_run_3;
    threadgroup uint g_sums_3;
    (&kernelContext_64)->g_sums_0 = &g_sums_3;
    threadgroup array<uint, int(512)> g_label_narrow_3;
    (&kernelContext_64)->g_label_narrow_0 = &g_label_narrow_3;
    threadgroup uint g_changed_narrow_3;
    (&kernelContext_64)->g_changed_narrow_0 = &g_changed_narrow_3;
    uint ii_1 = group_0.x;
    uint tid_10 = thread_2.x;
    bool _S1213;
    if(ii_1 >= (params_3->impactor_count_0))
    {
        _S1213 = true;
    }
    else
    {
        bool _S1214 = stopped_uniform_0(tid_10, &kernelContext_64);
        _S1213 = _S1214;
    }
    if(_S1213)
    {
        return;
    }
    Impactor_natural_0 device* _S1215 = (&kernelContext_64)->impactors_0+ii_1;
    float4 _S1216 = float4((*_S1215).position_err_1) ;
    float4 _S1217 = float4((*_S1215).velocity_1) ;
    float4 _S1218 = float4((*_S1215).velocity_err_1) ;
    float4 _S1219 = float4((*_S1215).angular_velocity_1) ;
    float4 _S1220 = float4((*_S1215).rotation_1) ;
    float4 _S1221 = float4((*_S1215).inertia0_2) ;
    float4 _S1222 = float4((*_S1215).inertia1_2) ;
    float4 _S1223 = float4((*_S1215).inertia2_2) ;
    float4 _S1224 = float4((*_S1215).inv0_2) ;
    float4 _S1225 = float4((*_S1215).inv1_2) ;
    float4 _S1226 = float4((*_S1215).inv2_2) ;
    float4 _S1227 = float4((*_S1215).shape_0) ;
    float4 _S1228 = float4((*_S1215).half_1) ;
    float4 _S1229 = float4((*_S1215).mat_0) ;
    float4 _S1230 = float4((*_S1215).crush_0) ;
    float4 _S1231 = float4((*_S1215).geom_0) ;
    float4 _S1232 = float4((*_S1215).rotation_err_1) ;
    float4 _S1233 = float4((*_S1215).ledger_0) ;
    uint4 _S1234 = uint4((*_S1215).cand_0) ;
    float4 _S1235 = float4((*_S1215).momentum_1) ;
    float4 _S1236 = float4((*_S1215).momentum_err_1) ;
    thread Impactor_0 imp_5;
    (&imp_5)->position_1 = float4((*_S1215).position_1) ;
    (&imp_5)->position_err_1 = _S1216;
    (&imp_5)->velocity_1 = _S1217;
    (&imp_5)->velocity_err_1 = _S1218;
    (&imp_5)->angular_velocity_1 = _S1219;
    (&imp_5)->rotation_1 = _S1220;
    (&imp_5)->inertia0_2 = _S1221;
    (&imp_5)->inertia1_2 = _S1222;
    (&imp_5)->inertia2_2 = _S1223;
    (&imp_5)->inv0_2 = _S1224;
    (&imp_5)->inv1_2 = _S1225;
    (&imp_5)->inv2_2 = _S1226;
    (&imp_5)->shape_0 = _S1227;
    (&imp_5)->half_1 = _S1228;
    (&imp_5)->mat_0 = _S1229;
    (&imp_5)->crush_0 = _S1230;
    (&imp_5)->geom_0 = _S1231;
    (&imp_5)->rotation_err_1 = _S1232;
    (&imp_5)->ledger_0 = _S1233;
    (&imp_5)->cand_0 = _S1234;
    (&imp_5)->momentum_1 = _S1235;
    (&imp_5)->momentum_err_1 = _S1236;
    float4 _S1237 = float4(0.0f) ;
    thread float4 shares_0 = _S1237;
    thread float4 unused_5 = _S1237;
    uint k_20 = (&imp_5)->cand_0.x + tid_10;
    for(;;)
    {
        if(k_20 < ((&imp_5)->cand_0.y))
        {
        }
        else
        {
            break;
        }
        shares_0 = shares_0 + float4(*((&kernelContext_64)->scratch_0+((&kernelContext_64)->params_0->cand_base_0 + 3U * k_20))) ;
        k_20 = k_20 + 256U;
    }
    group_sum2_0(tid_10, &shares_0, &unused_5, &kernelContext_64);
    if(tid_10 != 0U)
    {
        _S1213 = true;
    }
    else
    {
        _S1213 = ((&imp_5)->cand_0.z) != 0U;
    }
    if(_S1213)
    {
        return;
    }
    (&imp_5)->geom_0 = float4(1.0f, (&imp_5)->crush_0.w, 0.0f, 0.0f);
    float total_2 = shares_0.x;
    if(((&imp_5)->crush_0.x) > 0.0f)
    {
        _S1213 = ((&imp_5)->crush_0.z) < ((&imp_5)->crush_0.y);
    }
    else
    {
        _S1213 = false;
    }
    if(_S1213)
    {
        _S1213 = total_2 > ((&imp_5)->crush_0.x);
    }
    else
    {
        _S1213 = false;
    }
    if(_S1213)
    {
        float extra_0 = (total_2 - (&imp_5)->crush_0.x) / shares_0.y;
        (&imp_5)->crush_0.w = (&imp_5)->crush_0.w + extra_0;
        (&imp_5)->crush_0.z = (&imp_5)->crush_0.z + (&imp_5)->crush_0.x * extra_0;
        float _S1238 = (&imp_5)->crush_0.x * extra_0;
        thread float _S1239 = (&imp_5)->ledger_0.z;
        thread float _S1240 = (&imp_5)->ledger_0.w;
        comp_add1_0(&_S1239, &_S1240, _S1238);
        (&imp_5)->ledger_0.w = _S1240;
        (&imp_5)->ledger_0.z = _S1239;
        float _S1241 = (&imp_5)->crush_0.x * extra_0;
        thread float _S1242 = (&imp_5)->ledger_0.x;
        thread float _S1243 = (&imp_5)->ledger_0.y;
        comp_add1_0(&_S1242, &_S1243, _S1241);
        (&imp_5)->ledger_0.y = _S1243;
        (&imp_5)->ledger_0.x = _S1242;
        (&imp_5)->geom_0.x = (&imp_5)->crush_0.x / total_2;
    }
    Impactor_natural_0 device* _S1244 = (&kernelContext_64)->impactors_0+ii_1;
    _S1244->position_1 = packed_float4(imp_5.position_1) ;
    _S1244->position_err_1 = packed_float4(imp_5.position_err_1) ;
    _S1244->velocity_1 = packed_float4(imp_5.velocity_1) ;
    _S1244->velocity_err_1 = packed_float4(imp_5.velocity_err_1) ;
    _S1244->angular_velocity_1 = packed_float4(imp_5.angular_velocity_1) ;
    _S1244->rotation_1 = packed_float4(imp_5.rotation_1) ;
    _S1244->inertia0_2 = packed_float4(imp_5.inertia0_2) ;
    _S1244->inertia1_2 = packed_float4(imp_5.inertia1_2) ;
    _S1244->inertia2_2 = packed_float4(imp_5.inertia2_2) ;
    _S1244->inv0_2 = packed_float4(imp_5.inv0_2) ;
    _S1244->inv1_2 = packed_float4(imp_5.inv1_2) ;
    _S1244->inv2_2 = packed_float4(imp_5.inv2_2) ;
    _S1244->shape_0 = packed_float4(imp_5.shape_0) ;
    _S1244->half_1 = packed_float4(imp_5.half_1) ;
    _S1244->mat_0 = packed_float4(imp_5.mat_0) ;
    _S1244->crush_0 = packed_float4(imp_5.crush_0) ;
    _S1244->geom_0 = packed_float4(imp_5.geom_0) ;
    _S1244->rotation_err_1 = packed_float4(imp_5.rotation_err_1) ;
    _S1244->ledger_0 = packed_float4(imp_5.ledger_0) ;
    _S1244->cand_0 = packed_uint4(imp_5.cand_0) ;
    _S1244->momentum_1 = packed_float4(imp_5.momentum_1) ;
    _S1244->momentum_err_1 = packed_float4(imp_5.momentum_err_1) ;
    return;
}

struct WideGroup_0
{
    uint island_3;
    uint begin_1;
    uint end_0;
    uint first_0;
};

WideGroup_0 wide_group_0(uint table_0, uint g_2, KernelContext_0 thread* kernelContext_65)
{
    thread WideGroup_0 w_8;
    uint _S1245 = table_0 + 4U * g_2;
    (&w_8)->island_3 = kernelContext_65->index_0[_S1245];
    (&w_8)->begin_1 = kernelContext_65->index_0[_S1245 + 1U];
    (&w_8)->end_0 = kernelContext_65->index_0[_S1245 + 2U];
    (&w_8)->first_0 = kernelContext_65->index_0[_S1245 + 3U];
    return w_8;
}

WideGroup_0 wide_group_uniform_0(uint table_1, uint g_3, uint tid_11, KernelContext_0 thread* kernelContext_66)
{
    WideGroup_0 _S1246 = wide_group_0(table_1, g_3, kernelContext_66);
    thread WideGroup_0 w_9 = _S1246;
    uint4 u_3 = group_uniform4_0(uint4((&w_9)->island_3, (&w_9)->begin_1, (&w_9)->end_0, (&w_9)->first_0), tid_11);
    (&w_9)->island_3 = u_3.x;
    (&w_9)->begin_1 = u_3.y;
    (&w_9)->end_0 = u_3.z;
    (&w_9)->first_0 = u_3.w;
    return w_9;
}

bool wide_runs_0(const Island_0 thread* isl_14, KernelContext_0 thread* kernelContext_67)
{
    uint4 _S1247 = isl_14->info_0;
    bool _S1248;
    if(((isl_14->info_0.z) & 1U) != 0U)
    {
        _S1248 = true;
    }
    else
    {
        _S1248 = (_S1247.y) == 0U;
    }
    if(_S1248)
    {
        return false;
    }
    if(((_S1247.x) & 4U) == 0U)
    {
        _S1248 = true;
    }
    else
    {
        bool _S1249 = contact_stopped_0(isl_14, kernelContext_67);
        _S1248 = !_S1249;
    }
    return _S1248;
}

bool wide_enter_0(uint tid_12, const Island_0 thread* isl_15, KernelContext_0 thread* kernelContext_68)
{
    seen_clear_0(tid_12, kernelContext_68);
    if(tid_12 == 0U)
    {
        bool _S1250 = wide_runs_0(isl_15, kernelContext_68);
        int _S1251;
        if(_S1250)
        {
            _S1251 = int(1);
        }
        else
        {
            _S1251 = int(0);
        }
        *kernelContext_68->g_wide_run_0 = uint(_S1251);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (group_uniform_0(*kernelContext_68->g_wide_run_0, tid_12)) != 0U;
}

uint wide_step_0(const Island_0 thread* isl_16, KernelContext_0 thread* kernelContext_69)
{
    return isl_16->info_0.w - kernelContext_69->params_0->step_start_0;
}

Island_0 group_uniform_island_0(uint _S1252, uint _S1253, KernelContext_0 thread* kernelContext_70)
{
    Island_natural_0 device* _S1254 = kernelContext_70->islands_0+_S1252;
    Island_0 _S1255 = { uint4((*_S1254).range_0) , uint4((*_S1254).info_0) , float4((*_S1254).com_0) , float4((*_S1254).inertia0_0) , float4((*_S1254).inertia1_0) , float4((*_S1254).inertia2_0) , float4((*_S1254).inv0_0) , float4((*_S1254).inv1_0) , float4((*_S1254).inv2_0) , float4((*_S1254).wcom_0) , float4((*_S1254).winv0_0) , float4((*_S1254).winv1_0) , float4((*_S1254).winv2_0) , float4((*_S1254).rotation_0) , float4((*_S1254).position_0) , float4((*_S1254).position_err_0) , float4((*_S1254).velocity_0) , float4((*_S1254).velocity_err_0) , float4((*_S1254).angular_velocity_0) , (*_S1254).done_0, (*_S1254).coverage_0, (*_S1254).wide_groups_0, (*_S1254).done_pad_0, uint4((*_S1254).probes_0) , float4((*_S1254).energy_0) , float4((*_S1254).rotation_err_0) , float4((*_S1254).momentum_0) , float4((*_S1254).momentum_err_0)  };
    return _S1255;
}

[[kernel]] void wide_wake(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_3 [[thread_position_in_threadgroup]], Params_0 constant* params_4 [[buffer(0)]], Island_natural_0 device* islands_4 [[buffer(9)]], uint device* index_4 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_4 [[buffer(3)]], packed_float4 device* state_6 [[buffer(6)]], packed_float4 device* scratch_4 [[buffer(8)]], packed_float4 device* contact_state_4 [[buffer(11)]], packed_float4 device* loads_4 [[buffer(5)]], Impactor_natural_0 device* impactors_4 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_4 [[buffer(7)]], BondStatic_natural_0 device* bonds_4 [[buffer(2)]], MaterialTable_0 constant* materials_4 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_71;
    (&kernelContext_71)->params_0 = params_4;
    (&kernelContext_71)->islands_0 = islands_4;
    (&kernelContext_71)->index_0 = index_4;
    (&kernelContext_71)->chunks_0 = chunks_4;
    (&kernelContext_71)->state_0 = state_6;
    (&kernelContext_71)->scratch_0 = scratch_4;
    (&kernelContext_71)->contact_state_0 = contact_state_4;
    (&kernelContext_71)->loads_0 = loads_4;
    (&kernelContext_71)->impactors_0 = impactors_4;
    (&kernelContext_71)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_71)->bonds_0 = bonds_4;
    (&kernelContext_71)->materials_0 = materials_4;
    threadgroup array<float4, int(32)> g_part_a_4;
    (&kernelContext_71)->g_part_a_0 = &g_part_a_4;
    threadgroup array<float4, int(32)> g_part_b_4;
    (&kernelContext_71)->g_part_b_0 = &g_part_b_4;
    threadgroup array<uint, int(5)> g_seen_4;
    (&kernelContext_71)->g_seen_0 = &g_seen_4;
    threadgroup uint g_wide_run_4;
    (&kernelContext_71)->g_wide_run_0 = &g_wide_run_4;
    threadgroup array<uint, int(4096)> g_label_4;
    (&kernelContext_71)->g_label_0 = &g_label_4;
    threadgroup uint g_changed_4;
    (&kernelContext_71)->g_changed_0 = &g_changed_4;
    threadgroup uint g_halt_4;
    (&kernelContext_71)->g_halt_0 = &g_halt_4;
    threadgroup uint g_run_4;
    (&kernelContext_71)->g_run_0 = &g_run_4;
    threadgroup uint g_sums_4;
    (&kernelContext_71)->g_sums_0 = &g_sums_4;
    threadgroup array<uint, int(512)> g_label_narrow_4;
    (&kernelContext_71)->g_label_narrow_0 = &g_label_narrow_4;
    threadgroup uint g_changed_narrow_4;
    (&kernelContext_71)->g_changed_narrow_0 = &g_changed_narrow_4;
    uint tid_13 = thread_3.x;
    uint _S1256 = group_1.x;
    WideGroup_0 _S1257 = wide_group_uniform_0(params_4->wide_chunk_table_0, _S1256, tid_13, &kernelContext_71);
    if(_S1256 != (_S1257.first_0))
    {
        return;
    }
    Island_0 _S1258 = group_uniform_island_0(_S1257.island_3, tid_13, &kernelContext_71);
    uint _S1259 = _S1258.info_0.x;
    bool _S1260;
    if((_S1259 & 16U) == 0U)
    {
        _S1260 = true;
    }
    else
    {
        thread Island_0 _S1261 = _S1258;
        bool _S1262 = wide_enter_0(tid_13, &_S1261, &kernelContext_71);
        _S1260 = !_S1262;
    }
    if(_S1260)
    {
        return;
    }
    Quat_0 _S1263 = quat_of_0(_S1258.rotation_0);
    bool _S1264 = (_S1259 & 4U) != 0U;
    float3 _S1265 = float3(0.0f) ;
    thread float3 norm_5 = _S1265;
    thread float3 unused_6 = _S1265;
    uint c_35 = _S1258.range_0.x + tid_13;
    for(;;)
    {
        if(c_35 < (_S1258.range_0.y))
        {
        }
        else
        {
            break;
        }
        thread Island_0 _S1266 = _S1258;
        uint _S1267 = wide_step_0(&_S1266, &kernelContext_71);
        float _S1268 = (&kernelContext_71)->params_0->dt_0;
        thread Quat_0 _S1269 = _S1263;
        float _S1270 = settled_chunk_load_0(c_35, &_S1269, _S1267, _S1268, _S1264, &kernelContext_71);
        norm_5.x = norm_5.x + _S1270;
        c_35 = c_35 + 256U;
    }
    group_sum3_0(tid_13, &norm_5, &unused_6, &kernelContext_71);
    if(tid_13 != 0U)
    {
        return;
    }
    uint _S1271 = seen_bits_0(&kernelContext_71);
    uint cov_25 = 1024U | _S1271;
    if(((&kernelContext_71)->params_0->solve_mode_0) == 1U)
    {
        _S1260 = (abs(norm_5.x - _S1258.energy_0.z)) > (_S1258.energy_0.w);
    }
    else
    {
        _S1260 = false;
    }
    uint cov_26;
    if(_S1260)
    {
        ((&kernelContext_71)->islands_0+_S1257.island_3)->info_0[int(0)] = _S1259 & 4294967279U;
        ((&kernelContext_71)->islands_0+_S1257.island_3)->info_0[int(2)] = (_S1258.info_0.z) | 4U;
        cov_26 = cov_25 | 2048U;
    }
    else
    {
        cov_26 = cov_25;
    }
    cov_mark_0(_S1257.island_3, _S1258.coverage_0, cov_26, &kernelContext_71);
    return;
}

void wide_partials_0(uint tid_14, uint first_1, uint count_6, uint p_13, float3 thread* a_13, float3 thread* b_29, KernelContext_0 thread* kernelContext_72)
{
    float4 _S1272 = float4(0.0f) ;
    thread float4 x_14 = _S1272;
    thread float4 y_2 = _S1272;
    uint s_6 = tid_14;
    for(;;)
    {
        if(s_6 < count_6)
        {
        }
        else
        {
            break;
        }
        uint _S1273 = 8U * (first_1 + s_6);
        x_14 = x_14 + float4(*(kernelContext_72->scratch_0+(kernelContext_72->params_0->wide_base_0 + _S1273 + p_13))) ;
        y_2 = y_2 + float4(*(kernelContext_72->scratch_0+(kernelContext_72->params_0->wide_base_0 + _S1273 + p_13 + 1U))) ;
        s_6 = s_6 + 256U;
    }
    group_sum2_0(tid_14, &x_14, &y_2, kernelContext_72);
    *a_13 = x_14.xyz;
    *b_29 = y_2.xyz;
    return;
}

void wide_store_0(uint slot_2, uint p_14, float3 a_14, float3 b_30, KernelContext_0 thread* kernelContext_73)
{
    uint _S1274 = 8U * slot_2;
    *(kernelContext_73->scratch_0+(kernelContext_73->params_0->wide_base_0 + _S1274 + p_14)) = packed_float4(float4(a_14, 0.0f)) ;
    *(kernelContext_73->scratch_0+(kernelContext_73->params_0->wide_base_0 + _S1274 + p_14 + 1U)) = packed_float4(float4(b_30, 0.0f)) ;
    return;
}

[[kernel]] void wide_drift(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], Params_0 constant* params_5 [[buffer(0)]], Island_natural_0 device* islands_5 [[buffer(9)]], uint device* index_5 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_5 [[buffer(3)]], packed_float4 device* state_7 [[buffer(6)]], packed_float4 device* scratch_5 [[buffer(8)]], packed_float4 device* contact_state_5 [[buffer(11)]], packed_float4 device* loads_5 [[buffer(5)]], Impactor_natural_0 device* impactors_5 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_5 [[buffer(7)]], BondStatic_natural_0 device* bonds_5 [[buffer(2)]], MaterialTable_0 constant* materials_5 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_74;
    (&kernelContext_74)->params_0 = params_5;
    (&kernelContext_74)->islands_0 = islands_5;
    (&kernelContext_74)->index_0 = index_5;
    (&kernelContext_74)->chunks_0 = chunks_5;
    (&kernelContext_74)->state_0 = state_7;
    (&kernelContext_74)->scratch_0 = scratch_5;
    (&kernelContext_74)->contact_state_0 = contact_state_5;
    (&kernelContext_74)->loads_0 = loads_5;
    (&kernelContext_74)->impactors_0 = impactors_5;
    (&kernelContext_74)->bond_dyn_0 = bond_dyn_5;
    (&kernelContext_74)->bonds_0 = bonds_5;
    (&kernelContext_74)->materials_0 = materials_5;
    threadgroup array<float4, int(32)> g_part_a_5;
    (&kernelContext_74)->g_part_a_0 = &g_part_a_5;
    threadgroup array<float4, int(32)> g_part_b_5;
    (&kernelContext_74)->g_part_b_0 = &g_part_b_5;
    threadgroup array<uint, int(5)> g_seen_5;
    (&kernelContext_74)->g_seen_0 = &g_seen_5;
    threadgroup uint g_wide_run_5;
    (&kernelContext_74)->g_wide_run_0 = &g_wide_run_5;
    threadgroup array<uint, int(4096)> g_label_5;
    (&kernelContext_74)->g_label_0 = &g_label_5;
    threadgroup uint g_changed_5;
    (&kernelContext_74)->g_changed_0 = &g_changed_5;
    threadgroup uint g_halt_5;
    (&kernelContext_74)->g_halt_0 = &g_halt_5;
    threadgroup uint g_run_5;
    (&kernelContext_74)->g_run_0 = &g_run_5;
    threadgroup uint g_sums_5;
    (&kernelContext_74)->g_sums_0 = &g_sums_5;
    threadgroup array<uint, int(512)> g_label_narrow_5;
    (&kernelContext_74)->g_label_narrow_0 = &g_label_narrow_5;
    threadgroup uint g_changed_narrow_5;
    (&kernelContext_74)->g_changed_narrow_0 = &g_changed_narrow_5;
    uint tid_15 = thread_4.x;
    uint _S1275 = group_2.x;
    WideGroup_0 _S1276 = wide_group_uniform_0(params_5->wide_chunk_table_0, _S1275, tid_15, &kernelContext_74);
    Island_0 _S1277 = group_uniform_island_0(_S1276.island_3, tid_15, &kernelContext_74);
    bool _S1278;
    if(((_S1277.info_0.x) & 17U) != 0U)
    {
        _S1278 = true;
    }
    else
    {
        thread Island_0 _S1279 = _S1277;
        bool _S1280 = wide_enter_0(tid_15, &_S1279, &kernelContext_74);
        _S1278 = !_S1280;
    }
    if(_S1278)
    {
        return;
    }
    thread float3 tu_6;
    thread float3 pv_6;
    wide_partials_0(tid_15, _S1276.first_0, _S1277.wide_groups_0, 2U, &tu_6, &pv_6, &kernelContext_74);
    float3 _S1281 = float3(_S1277.wcom_0.w) ;
    float3 tr_8 = tu_6 / _S1281;
    float3 dv_8 = pv_6 / _S1281;
    float3 _S1282 = float3(0.0f) ;
    thread float3 lu_6 = _S1282;
    thread float3 lv_6 = _S1282;
    uint c_36 = _S1276.begin_1 + tid_15;
    if(c_36 < (_S1276.end_0))
    {
        drift_angular_0(c_36, _S1277.wcom_0.xyz, tr_8, dv_8, &lu_6, &lv_6, &kernelContext_74);
    }
    group_sum3_0(tid_15, &lu_6, &lv_6, &kernelContext_74);
    if(tid_15 == 0U)
    {
        wide_store_0(_S1275, 4U, lu_6, lv_6, &kernelContext_74);
    }
    return;
}

Rigid_0 wide_rigid_frame_0(uint tid_16, const Island_0 thread* isl_17, const WideGroup_0 thread* wg_0, KernelContext_0 thread* kernelContext_75)
{
    Rigid_0 _S1283 = rigid_of_0(isl_17);
    thread Rigid_0 rg_12 = _S1283;
    if(((isl_17->info_0.x) & 1U) == 0U)
    {
        thread float3 f_17;
        thread float3 t_18;
        wide_partials_0(tid_16, wg_0->first_0, isl_17->wide_groups_0, 0U, &f_17, &t_18, kernelContext_75);
        rigid_acceleration_0(isl_17, &rg_12, f_17, t_18);
    }
    return rg_12;
}

[[kernel]] void wide_rigid(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], Params_0 constant* params_6 [[buffer(0)]], Island_natural_0 device* islands_6 [[buffer(9)]], uint device* index_6 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_6 [[buffer(3)]], packed_float4 device* state_8 [[buffer(6)]], packed_float4 device* scratch_6 [[buffer(8)]], packed_float4 device* contact_state_6 [[buffer(11)]], packed_float4 device* loads_6 [[buffer(5)]], Impactor_natural_0 device* impactors_6 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_6 [[buffer(7)]], BondStatic_natural_0 device* bonds_6 [[buffer(2)]], MaterialTable_0 constant* materials_6 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_76;
    (&kernelContext_76)->params_0 = params_6;
    (&kernelContext_76)->islands_0 = islands_6;
    (&kernelContext_76)->index_0 = index_6;
    (&kernelContext_76)->chunks_0 = chunks_6;
    (&kernelContext_76)->state_0 = state_8;
    (&kernelContext_76)->scratch_0 = scratch_6;
    (&kernelContext_76)->contact_state_0 = contact_state_6;
    (&kernelContext_76)->loads_0 = loads_6;
    (&kernelContext_76)->impactors_0 = impactors_6;
    (&kernelContext_76)->bond_dyn_0 = bond_dyn_6;
    (&kernelContext_76)->bonds_0 = bonds_6;
    (&kernelContext_76)->materials_0 = materials_6;
    threadgroup array<float4, int(32)> g_part_a_6;
    (&kernelContext_76)->g_part_a_0 = &g_part_a_6;
    threadgroup array<float4, int(32)> g_part_b_6;
    (&kernelContext_76)->g_part_b_0 = &g_part_b_6;
    threadgroup array<uint, int(5)> g_seen_6;
    (&kernelContext_76)->g_seen_0 = &g_seen_6;
    threadgroup uint g_wide_run_6;
    (&kernelContext_76)->g_wide_run_0 = &g_wide_run_6;
    threadgroup array<uint, int(4096)> g_label_6;
    (&kernelContext_76)->g_label_0 = &g_label_6;
    threadgroup uint g_changed_6;
    (&kernelContext_76)->g_changed_0 = &g_changed_6;
    threadgroup uint g_halt_6;
    (&kernelContext_76)->g_halt_0 = &g_halt_6;
    threadgroup uint g_run_6;
    (&kernelContext_76)->g_run_0 = &g_run_6;
    threadgroup uint g_sums_6;
    (&kernelContext_76)->g_sums_0 = &g_sums_6;
    threadgroup array<uint, int(512)> g_label_narrow_6;
    (&kernelContext_76)->g_label_narrow_0 = &g_label_narrow_6;
    threadgroup uint g_changed_narrow_6;
    (&kernelContext_76)->g_changed_narrow_0 = &g_changed_narrow_6;
    uint tid_17 = thread_5.x;
    uint _S1284 = group_3.x;
    WideGroup_0 _S1285 = wide_group_uniform_0(params_6->wide_chunk_table_0, _S1284, tid_17, &kernelContext_76);
    Island_0 _S1286 = group_uniform_island_0(_S1285.island_3, tid_17, &kernelContext_76);
    uint _S1287 = _S1286.info_0.x;
    bool _S1288;
    if((_S1287 & 1U) != 0U)
    {
        _S1288 = true;
    }
    else
    {
        thread Island_0 _S1289 = _S1286;
        bool _S1290 = wide_enter_0(tid_17, &_S1289, &kernelContext_76);
        _S1288 = !_S1290;
    }
    if(_S1288)
    {
        return;
    }
    if((_S1287 & 16U) != 0U)
    {
        if(_S1284 != (_S1285.first_0))
        {
            return;
        }
        thread Island_0 _S1291 = _S1286;
        thread WideGroup_0 _S1292 = _S1285;
        Rigid_0 _S1293 = wide_rigid_frame_0(tid_17, &_S1291, &_S1292, &kernelContext_76);
        thread Rigid_0 rs_0 = _S1293;
        if(tid_17 == 0U)
        {
            _S1288 = (_S1287 & 2U) == 0U;
        }
        else
        {
            _S1288 = false;
        }
        if(_S1288)
        {
            float _S1294 = (&kernelContext_76)->params_0->dt_0;
            thread Island_0 _S1295 = _S1286;
            integrate_rigid_0(&_S1295, &rs_0, _S1294);
            thread Quat_0 _S1296 = (&rs_0)->rot_0;
            float4 _S1297 = quat_vec_0(&_S1296);
            ((&kernelContext_76)->islands_0+_S1285.island_3)->rotation_0 = packed_float4(_S1297) ;
            ((&kernelContext_76)->islands_0+_S1285.island_3)->rotation_err_0 = packed_float4((&rs_0)->rot_err_0) ;
            ((&kernelContext_76)->islands_0+_S1285.island_3)->position_0 = packed_float4(float4((&rs_0)->pos_0, 0.0f)) ;
            ((&kernelContext_76)->islands_0+_S1285.island_3)->position_err_0 = packed_float4(float4((&rs_0)->pos_err_0, 0.0f)) ;
            ((&kernelContext_76)->islands_0+_S1285.island_3)->velocity_0 = packed_float4(float4((&rs_0)->vel_0, 0.0f)) ;
            ((&kernelContext_76)->islands_0+_S1285.island_3)->velocity_err_0 = packed_float4(float4((&rs_0)->vel_err_0, 0.0f)) ;
            ((&kernelContext_76)->islands_0+_S1285.island_3)->angular_velocity_0 = packed_float4(float4((&rs_0)->w_1, 0.0f)) ;
            ((&kernelContext_76)->islands_0+_S1285.island_3)->momentum_0 = packed_float4(float4((&rs_0)->l_0, 0.0f)) ;
            ((&kernelContext_76)->islands_0+_S1285.island_3)->momentum_err_0 = packed_float4(float4((&rs_0)->l_err_0, 0.0f)) ;
        }
        return;
    }
    thread float3 tu_7;
    thread float3 pv_7;
    wide_partials_0(tid_17, _S1285.first_0, _S1286.wide_groups_0, 2U, &tu_7, &pv_7, &kernelContext_76);
    thread float3 lu_7;
    thread float3 lv_7;
    wide_partials_0(tid_17, _S1285.first_0, _S1286.wide_groups_0, 4U, &lu_7, &lv_7, &kernelContext_76);
    float3 _S1298 = float3(_S1286.wcom_0.w) ;
    float3 tr_9 = tu_7 / _S1298;
    float3 dv_9 = pv_7 / _S1298;
    float3 phi_8 = rows_mul_0(_S1286.winv0_0, _S1286.winv1_0, _S1286.winv2_0, lu_7);
    float3 dw_7 = rows_mul_0(_S1286.winv0_0, _S1286.winv1_0, _S1286.winv2_0, lv_7);
    uint c_37 = _S1285.begin_1 + tid_17;
    if(c_37 < (_S1285.end_0))
    {
        drift_apply_0(c_37, _S1286.wcom_0.xyz, tr_9, phi_8, dv_9, dw_7, &kernelContext_76);
    }
    if(_S1284 != (_S1285.first_0))
    {
        return;
    }
    thread Island_0 _S1299 = _S1286;
    thread WideGroup_0 _S1300 = _S1285;
    Rigid_0 _S1301 = wide_rigid_frame_0(tid_17, &_S1299, &_S1300, &kernelContext_76);
    thread Rigid_0 rg_13 = _S1301;
    if(tid_17 != 0U)
    {
        return;
    }
    if(!((_S1287 & 2U) != 0U))
    {
        float _S1302 = (&kernelContext_76)->params_0->dt_0;
        thread Island_0 _S1303 = _S1286;
        integrate_rigid_0(&_S1303, &rg_13, _S1302);
        thread Island_0 _S1304 = _S1286;
        drift_rigid_0(&_S1304, &rg_13, tr_9, phi_8, dv_9, dw_7);
    }
    thread Quat_0 _S1305 = (&rg_13)->rot_0;
    float4 _S1306 = quat_vec_0(&_S1305);
    ((&kernelContext_76)->islands_0+_S1285.island_3)->rotation_0 = packed_float4(_S1306) ;
    ((&kernelContext_76)->islands_0+_S1285.island_3)->rotation_err_0 = packed_float4((&rg_13)->rot_err_0) ;
    ((&kernelContext_76)->islands_0+_S1285.island_3)->position_0 = packed_float4(float4((&rg_13)->pos_0, 0.0f)) ;
    ((&kernelContext_76)->islands_0+_S1285.island_3)->position_err_0 = packed_float4(float4((&rg_13)->pos_err_0, 0.0f)) ;
    ((&kernelContext_76)->islands_0+_S1285.island_3)->velocity_0 = packed_float4(float4((&rg_13)->vel_0, 0.0f)) ;
    ((&kernelContext_76)->islands_0+_S1285.island_3)->velocity_err_0 = packed_float4(float4((&rg_13)->vel_err_0, 0.0f)) ;
    ((&kernelContext_76)->islands_0+_S1285.island_3)->angular_velocity_0 = packed_float4(float4((&rg_13)->w_1, 0.0f)) ;
    ((&kernelContext_76)->islands_0+_S1285.island_3)->momentum_0 = packed_float4(float4((&rg_13)->l_0, 0.0f)) ;
    ((&kernelContext_76)->islands_0+_S1285.island_3)->momentum_err_0 = packed_float4(float4((&rg_13)->l_err_0, 0.0f)) ;
    return;
}

bool island_connected_0(uint tid_18, const Island_0 thread* isl_18, KernelContext_0 thread* kernelContext_77)
{
    uint _S1307 = isl_18->range_0.x;
    uint n_12 = group_uniform_0(isl_18->range_0.y - _S1307, tid_18);
    bool converged_2;
    if(n_12 > 4096U)
    {
        converged_2 = true;
    }
    else
    {
        converged_2 = (kernelContext_77->params_0->device_connect_0) == 0U;
    }
    if(converged_2)
    {
        return false;
    }
    uint c_38 = tid_18;
    for(;;)
    {
        if(c_38 < n_12)
        {
        }
        else
        {
            break;
        }
        (*kernelContext_77->g_label_0)[c_38] = c_38;
        c_38 = c_38 + 256U;
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
        if(tid_18 == 0U)
        {
            *kernelContext_77->g_changed_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c_38 = tid_18;
        for(;;)
        {
            if(c_38 < n_12)
            {
            }
            else
            {
                break;
            }
            uint gc_1 = _S1307 + c_38;
            uint _S1308 = kernelContext_77->index_0[gc_1];
            uint l_5 = (*kernelContext_77->g_label_0)[c_38];
            uint e_5 = _S1308;
            for(;;)
            {
                if(e_5 < (kernelContext_77->index_0)[gc_1 + 1U])
                {
                }
                else
                {
                    break;
                }
                uint entry_4 = kernelContext_77->index_0[e_5];
                uint bond_1 = entry_4 >> 1U;
                bool _S1309 = ((float4((kernelContext_77->bonds_0+bond_1)->law_0.stiff1_0) ).w) != 0.0f;
                thread JointState_0 _S1310 = (kernelContext_77->bond_dyn_0+bond_1)->js_0;
                bool _S1311 = connected_0(&_S1310, _S1309);
                if(!_S1311)
                {
                    e_5 = e_5 + 1U;
                    continue;
                }
                uint other_1;
                if((entry_4 & 1U) == 0U)
                {
                    other_1 = (uint4((kernelContext_77->bonds_0+bond_1)->law_0.ids_0) ).z;
                }
                else
                {
                    other_1 = (uint4((kernelContext_77->bonds_0+bond_1)->law_0.ids_0) ).y;
                }
                l_5 = min(l_5, (*kernelContext_77->g_label_0)[other_1 - _S1307]);
                e_5 = e_5 + 1U;
            }
            uint _S1312 = min(l_5, (*kernelContext_77->g_label_0)[l_5]);
            if(_S1312 < (*kernelContext_77->g_label_0)[c_38])
            {
                (*kernelContext_77->g_label_0)[c_38] = _S1312;
                *kernelContext_77->g_changed_0 = 1U;
            }
            c_38 = c_38 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool converged_3 = (group_uniform_0(*kernelContext_77->g_changed_0, tid_18)) == 0U;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(converged_3)
        {
            converged_2 = converged_3;
            break;
        }
        uint _S1313 = pass_1 + 1U;
        converged_2 = converged_3;
        pass_1 = _S1313;
    }
    if(!converged_2)
    {
        return false;
    }
    if(tid_18 == 0U)
    {
        *kernelContext_77->g_changed_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    c_38 = tid_18;
    for(;;)
    {
        if(c_38 < n_12)
        {
        }
        else
        {
            break;
        }
        if(((*kernelContext_77->g_label_0)[c_38]) != 0U)
        {
            *kernelContext_77->g_changed_0 = 1U;
        }
        c_38 = c_38 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (group_uniform_0(*kernelContext_77->g_changed_0, tid_18)) == 0U;
}

[[kernel]] void wide_end(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], Params_0 constant* params_7 [[buffer(0)]], Island_natural_0 device* islands_7 [[buffer(9)]], uint device* index_7 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_7 [[buffer(3)]], packed_float4 device* state_9 [[buffer(6)]], packed_float4 device* scratch_7 [[buffer(8)]], packed_float4 device* contact_state_7 [[buffer(11)]], packed_float4 device* loads_7 [[buffer(5)]], Impactor_natural_0 device* impactors_7 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_7 [[buffer(7)]], BondStatic_natural_0 device* bonds_7 [[buffer(2)]], MaterialTable_0 constant* materials_7 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_78;
    (&kernelContext_78)->params_0 = params_7;
    (&kernelContext_78)->islands_0 = islands_7;
    (&kernelContext_78)->index_0 = index_7;
    (&kernelContext_78)->chunks_0 = chunks_7;
    (&kernelContext_78)->state_0 = state_9;
    (&kernelContext_78)->scratch_0 = scratch_7;
    (&kernelContext_78)->contact_state_0 = contact_state_7;
    (&kernelContext_78)->loads_0 = loads_7;
    (&kernelContext_78)->impactors_0 = impactors_7;
    (&kernelContext_78)->bond_dyn_0 = bond_dyn_7;
    (&kernelContext_78)->bonds_0 = bonds_7;
    (&kernelContext_78)->materials_0 = materials_7;
    threadgroup array<float4, int(32)> g_part_a_7;
    (&kernelContext_78)->g_part_a_0 = &g_part_a_7;
    threadgroup array<float4, int(32)> g_part_b_7;
    (&kernelContext_78)->g_part_b_0 = &g_part_b_7;
    threadgroup array<uint, int(5)> g_seen_7;
    (&kernelContext_78)->g_seen_0 = &g_seen_7;
    threadgroup uint g_wide_run_7;
    (&kernelContext_78)->g_wide_run_0 = &g_wide_run_7;
    threadgroup array<uint, int(4096)> g_label_7;
    (&kernelContext_78)->g_label_0 = &g_label_7;
    threadgroup uint g_changed_7;
    (&kernelContext_78)->g_changed_0 = &g_changed_7;
    threadgroup uint g_halt_7;
    (&kernelContext_78)->g_halt_0 = &g_halt_7;
    threadgroup uint g_run_7;
    (&kernelContext_78)->g_run_0 = &g_run_7;
    threadgroup uint g_sums_7;
    (&kernelContext_78)->g_sums_0 = &g_sums_7;
    threadgroup array<uint, int(512)> g_label_narrow_7;
    (&kernelContext_78)->g_label_narrow_0 = &g_label_narrow_7;
    threadgroup uint g_changed_narrow_7;
    (&kernelContext_78)->g_changed_narrow_0 = &g_changed_narrow_7;
    uint tid_19 = thread_6.x;
    uint _S1314 = group_4.x;
    WideGroup_0 _S1315 = wide_group_uniform_0(params_7->wide_chunk_table_0, _S1314, tid_19, &kernelContext_78);
    if(_S1314 != (_S1315.first_0))
    {
        return;
    }
    Island_0 _S1316 = group_uniform_island_0(_S1315.island_3, tid_19, &kernelContext_78);
    thread Island_0 isl_19 = _S1316;
    thread Island_0 _S1317 = _S1316;
    bool _S1318 = wide_enter_0(tid_19, &_S1317, &kernelContext_78);
    if(!_S1318)
    {
        return;
    }
    thread float3 work_7;
    thread float3 flags_2;
    wide_partials_0(tid_19, _S1315.first_0, (&isl_19)->wide_groups_0, 6U, &work_7, &flags_2, &kernelContext_78);
    bool halt_0 = (((&isl_19)->info_0.z) & 2U) != 0U;
    bool whole_0;
    if(halt_0)
    {
        thread Island_0 _S1319 = isl_19;
        bool _S1320 = island_connected_0(tid_19, &_S1319, &kernelContext_78);
        whole_0 = _S1320;
    }
    else
    {
        whole_0 = false;
    }
    if(tid_19 != 0U)
    {
        return;
    }
    bool halt_1;
    if(whole_0)
    {
        (&isl_19)->info_0.z = ((&isl_19)->info_0.z) & 4294967293U;
        (&isl_19)->coverage_0 = ((&isl_19)->coverage_0) | 2097280U;
        halt_1 = false;
    }
    else
    {
        halt_1 = halt_0;
    }
    thread Island_0 _S1321 = isl_19;
    uint _S1322 = wide_step_0(&_S1321, &kernelContext_78);
    if(((&isl_19)->probes_0.y) > ((&isl_19)->probes_0.x))
    {
        thread Island_0 _S1323 = isl_19;
        Rigid_0 _S1324 = rigid_of_0(&_S1323);
        thread Island_0 _S1325 = isl_19;
        thread Rigid_0 _S1326 = _S1324;
        record_probes_0(&_S1325, &_S1326, _S1322, &kernelContext_78);
    }
    if(halt_1)
    {
        whole_0 = (((&isl_19)->info_0.x) & 4U) != 0U;
    }
    else
    {
        whole_0 = false;
    }
    if(whole_0)
    {
        contact_split_at_0((&isl_19)->info_0.w + 1U, &kernelContext_78);
    }
    float _S1327 = work_7.x;
    thread float _S1328 = (&isl_19)->energy_0.x;
    thread float _S1329 = (&isl_19)->energy_0.y;
    comp_add1_0(&_S1328, &_S1329, _S1327);
    (&isl_19)->energy_0.x = _S1328;
    (&isl_19)->energy_0.y = _S1329 + work_7.y;
    (&isl_19)->done_0 = (&isl_19)->done_0 + 1U;
    (&isl_19)->info_0.y = (&isl_19)->info_0.y - 1U;
    (&isl_19)->info_0.w = (&isl_19)->info_0.w + 1U;
    if(halt_1)
    {
        (&isl_19)->info_0.z = (((&isl_19)->info_0.z) & 4294967293U) | 1U;
    }
    uint _S1330;
    if(halt_1)
    {
        _S1330 = 128U;
    }
    else
    {
        _S1330 = 0U;
    }
    uint _S1331 = 256U | _S1330;
    if(((&isl_19)->probes_0.y) > ((&isl_19)->probes_0.x))
    {
        _S1330 = 262144U;
    }
    else
    {
        _S1330 = 0U;
    }
    uint _S1332 = _S1331 | _S1330;
    if((((&isl_19)->info_0.x) & 4U) != 0U)
    {
        _S1330 = 524288U;
    }
    else
    {
        _S1330 = 0U;
    }
    uint _S1333 = _S1332 | _S1330;
    if((flags_2.x) > 0.0f)
    {
        _S1330 = 32768U;
    }
    else
    {
        _S1330 = 0U;
    }
    uint _S1334 = _S1333 | _S1330;
    if((flags_2.y) > 0.0f)
    {
        _S1330 = 65536U;
    }
    else
    {
        _S1330 = 0U;
    }
    uint _S1335 = _S1334 | _S1330;
    if((flags_2.z) > 0.0f)
    {
        _S1330 = 131072U;
    }
    else
    {
        _S1330 = 0U;
    }
    (&isl_19)->coverage_0 = ((&isl_19)->coverage_0) | (_S1335 | _S1330);
    Island_natural_0 device* _S1336 = (&kernelContext_78)->islands_0+_S1315.island_3;
    _S1336->range_0 = packed_uint4(isl_19.range_0) ;
    _S1336->info_0 = packed_uint4(isl_19.info_0) ;
    _S1336->com_0 = packed_float4(isl_19.com_0) ;
    _S1336->inertia0_0 = packed_float4(isl_19.inertia0_0) ;
    _S1336->inertia1_0 = packed_float4(isl_19.inertia1_0) ;
    _S1336->inertia2_0 = packed_float4(isl_19.inertia2_0) ;
    _S1336->inv0_0 = packed_float4(isl_19.inv0_0) ;
    _S1336->inv1_0 = packed_float4(isl_19.inv1_0) ;
    _S1336->inv2_0 = packed_float4(isl_19.inv2_0) ;
    _S1336->wcom_0 = packed_float4(isl_19.wcom_0) ;
    _S1336->winv0_0 = packed_float4(isl_19.winv0_0) ;
    _S1336->winv1_0 = packed_float4(isl_19.winv1_0) ;
    _S1336->winv2_0 = packed_float4(isl_19.winv2_0) ;
    _S1336->rotation_0 = packed_float4(isl_19.rotation_0) ;
    _S1336->position_0 = packed_float4(isl_19.position_0) ;
    _S1336->position_err_0 = packed_float4(isl_19.position_err_0) ;
    _S1336->velocity_0 = packed_float4(isl_19.velocity_0) ;
    _S1336->velocity_err_0 = packed_float4(isl_19.velocity_err_0) ;
    _S1336->angular_velocity_0 = packed_float4(isl_19.angular_velocity_0) ;
    _S1336->done_0 = isl_19.done_0;
    _S1336->coverage_0 = isl_19.coverage_0;
    _S1336->wide_groups_0 = isl_19.wide_groups_0;
    _S1336->done_pad_0 = isl_19.done_pad_0;
    _S1336->probes_0 = packed_uint4(isl_19.probes_0) ;
    _S1336->energy_0 = packed_float4(isl_19.energy_0) ;
    _S1336->rotation_err_0 = packed_float4(isl_19.rotation_err_0) ;
    _S1336->momentum_0 = packed_float4(isl_19.momentum_0) ;
    _S1336->momentum_err_0 = packed_float4(isl_19.momentum_err_0) ;
    return;
}

uint sv_0(uint c_39, uint slot_3, KernelContext_0 thread* kernelContext_79)
{
    return kernelContext_79->params_0->statics_base_0 + 23U * c_39 + slot_3;
}

void project_load_slot_0(uint tid_20, const Island_0 thread* isl_20, uint slot_4, KernelContext_0 thread* kernelContext_80)
{
    uint _S1337;
    float3 _S1338 = float3(0.0f) ;
    thread float3 net_f_0 = _S1338;
    thread float3 net_m_0 = _S1338;
    uint4 _S1339 = isl_20->range_0;
    uint _S1340 = isl_20->range_0.x + tid_20;
    uint c_40 = _S1340;
    for(;;)
    {
        uint _S1341 = _S1339.y;
        _S1337 = _S1341;
        if(c_40 < _S1341)
        {
        }
        else
        {
            break;
        }
        uint _S1342 = sv_0(c_40, slot_4, kernelContext_80);
        float3 fi_1 = (float4(*(kernelContext_80->scratch_0+_S1342)) ).xyz;
        net_f_0 = net_f_0 + fi_1;
        float3 _S1343 = cross((float4((kernelContext_80->chunks_0+c_40)->center_0) ).xyz - isl_20->com_0.xyz, fi_1);
        uint _S1344 = sv_0(c_40, slot_4 + 1U, kernelContext_80);
        net_m_0 = net_m_0 + (_S1343 + (float4(*(kernelContext_80->scratch_0+_S1344)) ).xyz);
        c_40 = c_40 + 256U;
    }
    group_sum3_0(tid_20, &net_f_0, &net_m_0, kernelContext_80);
    float4 _S1345 = isl_20->com_0;
    float3 _S1346 = net_f_0 / float3(isl_20->com_0.w) ;
    float3 _S1347 = rows_mul_0(isl_20->inv0_0, isl_20->inv1_0, isl_20->inv2_0, net_m_0);
    c_40 = _S1340;
    for(;;)
    {
        if(c_40 < _S1337)
        {
        }
        else
        {
            break;
        }
        ChunkStatic_natural_0 device* _S1348 = kernelContext_80->chunks_0+c_40;
        float4 _S1349 = float4(_S1348->center_0) ;
        float3 r_11 = _S1349.xyz - _S1345.xyz;
        uint _S1350 = sv_0(c_40, slot_4, kernelContext_80);
        *(kernelContext_80->scratch_0+_S1350) = packed_float4(float4((float4(*(kernelContext_80->scratch_0+_S1350)) ).xyz - (_S1346 + cross(_S1347, r_11)) * float3(_S1349.w) , 0.0f)) ;
        uint _S1351 = sv_0(c_40, slot_4 + 1U, kernelContext_80);
        *(kernelContext_80->scratch_0+_S1351) = packed_float4(float4((float4(*(kernelContext_80->scratch_0+_S1351)) ).xyz - rows_mul_0(float4(_S1348->inertia0_1) , float4(_S1348->inertia1_1) , float4(_S1348->inertia2_1) , _S1347), 0.0f)) ;
        c_40 = c_40 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return;
}

float island_dot_0(uint tid_21, uint c0_0, uint c1_0, uint sa_2, uint sb_2, KernelContext_0 thread* kernelContext_81)
{
    float4 _S1352 = float4(0.0f) ;
    thread float4 acc_0 = _S1352;
    thread float4 unused_7 = _S1352;
    uint c_41 = c0_0 + tid_21;
    for(;;)
    {
        if(c_41 < c1_0)
        {
        }
        else
        {
            break;
        }
        uint _S1353 = sv_0(c_41, sa_2, kernelContext_81);
        float3 _S1354 = (float4(*(kernelContext_81->scratch_0+_S1353)) ).xyz;
        uint _S1355 = sv_0(c_41, sb_2, kernelContext_81);
        float _S1356 = dot(_S1354, (float4(*(kernelContext_81->scratch_0+_S1355)) ).xyz);
        uint _S1357 = sv_0(c_41, sa_2 + 1U, kernelContext_81);
        float3 _S1358 = (float4(*(kernelContext_81->scratch_0+_S1357)) ).xyz;
        uint _S1359 = sv_0(c_41, sb_2 + 1U, kernelContext_81);
        acc_0.x = acc_0.x + (_S1356 + dot(_S1358, (float4(*(kernelContext_81->scratch_0+_S1359)) ).xyz));
        c_41 = c_41 + 256U;
    }
    group_sum2_0(tid_21, &acc_0, &unused_7, kernelContext_81);
    return acc_0.x;
}

void static_kinematics_0(uint _S1360, float3 thread* _S1361, float3 thread* _S1362, KernelContext_0 thread* kernelContext_82)
{
    BondStatic_natural_0 device* _S1363 = kernelContext_82->bonds_0+_S1360;
    uint4 _S1364 = uint4(_S1363->law_0.ids_0) ;
    uint ca_1 = _S1364.y;
    uint cb_1 = _S1364.z;
    uint _S1365 = 4U * cb_1;
    uint _S1366 = 4U * ca_1;
    float3 _S1367 = (float4(*(kernelContext_82->state_0+_S1365)) ).xyz - (float4(*(kernelContext_82->state_0+_S1366)) ).xyz;
    uint _S1368 = sv_0(cb_1, 21U, kernelContext_82);
    float3 _S1369 = (float4(*(kernelContext_82->scratch_0+_S1368)) ).xyz;
    uint _S1370 = sv_0(ca_1, 21U, kernelContext_82);
    float3 du_0 = _S1367 + (_S1369 - (float4(*(kernelContext_82->scratch_0+_S1370)) ).xyz);
    uint _S1371 = _S1365 + 1U;
    uint _S1372 = _S1366 + 1U;
    float3 _S1373 = (float4(*(kernelContext_82->state_0+_S1371)) ).xyz - (float4(*(kernelContext_82->state_0+_S1372)) ).xyz;
    uint _S1374 = sv_0(cb_1, 22U, kernelContext_82);
    float3 _S1375 = (float4(*(kernelContext_82->scratch_0+_S1374)) ).xyz;
    uint _S1376 = sv_0(ca_1, 22U, kernelContext_82);
    float3 dth_0 = _S1373 + (_S1375 - (float4(*(kernelContext_82->scratch_0+_S1376)) ).xyz);
    float3 _S1377 = to_local_0(_S1360, du_0 + (cross((float4(*(kernelContext_82->state_0+_S1371)) ).xyz + (float4(*(kernelContext_82->scratch_0+_S1374)) ).xyz, (float4(_S1363->rb_0) ).xyz) - cross((float4(*(kernelContext_82->state_0+_S1372)) ).xyz + (float4(*(kernelContext_82->scratch_0+_S1376)) ).xyz, (float4(_S1363->ra_0) ).xyz)), kernelContext_82);
    *_S1361 = _S1377;
    float3 _S1378 = to_local_0(_S1360, dth_0, kernelContext_82);
    *_S1362 = _S1378;
    return;
}

JointResponse_0 static_response_0(uint i_11, KernelContext_0 thread* kernelContext_83)
{
    BondStatic_natural_0 device* _S1379 = kernelContext_83->bonds_0+i_11;
    thread float3 d_lin_5;
    thread float3 d_ang_4;
    static_kinematics_0(i_11, &d_lin_5, &d_ang_4, kernelContext_83);
    thread JointBond_natural_0 _S1380 = _S1379->law_0;
    _S1380 = _S1379->law_0;
    thread JointState_0 _S1381 = (kernelContext_83->bond_dyn_0+i_11)->js_0;
    JointResponse_0 _S1382 = joint_evaluate_0(&kernelContext_83->materials_0->m_0[(uint4((&_S1380)->ids_0) ).x], &_S1380, &_S1381, d_lin_5, d_ang_4, 0.0f, false);
    return _S1382;
}

void gather_loads_0(uint c_42, float3 thread* fi_2, float3 thread* mi_3, KernelContext_0 thread* kernelContext_84)
{
    float3 _S1383 = float3(0.0f) ;
    *fi_2 = _S1383;
    *mi_3 = _S1383;
    uint e_6 = kernelContext_84->index_0[c_42];
    for(;;)
    {
        if(e_6 < (kernelContext_84->index_0)[c_42 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_5 = kernelContext_84->index_0[e_6];
        uint bond_2 = entry_5 >> 1U;
        if((entry_5 & 1U) == 0U)
        {
            uint _S1384 = 3U * bond_2;
            *fi_2 = *fi_2 + (float4(*(kernelContext_84->scratch_0+_S1384)) ).xyz;
            *mi_3 = *mi_3 + (float4(*(kernelContext_84->scratch_0+(_S1384 + 1U))) ).xyz;
        }
        else
        {
            uint _S1385 = 3U * bond_2;
            *fi_2 = *fi_2 - (float4(*(kernelContext_84->scratch_0+_S1385)) ).xyz;
            *mi_3 = *mi_3 + (float4(*(kernelContext_84->scratch_0+(_S1385 + 2U))) ).xyz;
        }
        e_6 = e_6 + 1U;
    }
    return;
}

float bond_load_magnitude2_0(uint c_43, KernelContext_0 thread* kernelContext_85)
{
    uint e_7 = kernelContext_85->index_0[c_43];
    float m_5 = 0.0f;
    for(;;)
    {
        if(e_7 < (kernelContext_85->index_0)[c_43 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_6 = kernelContext_85->index_0[e_7];
        uint _S1386 = 3U * (entry_6 >> 1U);
        float3 f_18 = (float4(*(kernelContext_85->scratch_0+_S1386)) ).xyz;
        float3 t_19;
        if((entry_6 & 1U) == 0U)
        {
            t_19 = (float4(*(kernelContext_85->scratch_0+(_S1386 + 1U))) ).xyz;
        }
        else
        {
            t_19 = (float4(*(kernelContext_85->scratch_0+(_S1386 + 2U))) ).xyz;
        }
        float m_6 = m_5 + (dot(f_18, f_18) + dot(t_19, t_19));
        e_7 = e_7 + 1U;
        m_5 = m_6;
    }
    return m_5;
}

uint fixed_mask_0(uint c_44, KernelContext_0 thread* kernelContext_86)
{
    uint support_1 = (uint4((kernelContext_86->chunks_0+c_44)->info_1) ).x;
    uint _S1387;
    if(support_1 == 1U)
    {
        _S1387 = 63U;
    }
    else
    {
        if(support_1 == 2U)
        {
            _S1387 = 7U;
        }
        else
        {
            _S1387 = 0U;
        }
    }
    return _S1387;
}

void hold_0(uint mask_0, float4 thread* lin_0, float4 thread* ang_0, float4 keep_lin_0, float4 keep_ang_0)
{
    uint d_14 = 0U;
    for(;;)
    {
        if(d_14 < 3U)
        {
        }
        else
        {
            break;
        }
        if((mask_0 & (1U << d_14)) != 0U)
        {
            (*lin_0)[d_14] = keep_lin_0[d_14];
        }
        if((mask_0 & (1U << (d_14 + 3U))) != 0U)
        {
            (*ang_0)[d_14] = keep_ang_0[d_14];
        }
        d_14 = d_14 + 1U;
    }
    return;
}

bool is_finite_0(float x_15)
{
    return ((as_type<uint>((x_15))) & 2147483647U) < 2139095040U;
}

uint statics_bond_slot_0(uint i_12, KernelContext_0 thread* kernelContext_87)
{
    return kernelContext_87->params_0->statics_base_0 + 23U * kernelContext_87->params_0->chunk_count_0 + 2U * i_12;
}

void store_inverse_0(uint c_45, const array<float, int(36)> thread* a_15, KernelContext_0 thread* kernelContext_88)
{
    uint j_9;
    float sum_4;
    thread array<float, int(36)> l_6;
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
        l_6[k_21] = 0.0f;
        k_21 = k_21 + 1U;
    }
    bool spd_0 = true;
    uint i_13 = 0U;
    for(;;)
    {
        bool _S1388;
        if(i_13 < 6U)
        {
            _S1388 = spd_0;
        }
        else
        {
            _S1388 = false;
        }
        if(_S1388)
        {
        }
        else
        {
            break;
        }
        j_9 = 0U;
        for(;;)
        {
            if(j_9 <= i_13)
            {
            }
            else
            {
                break;
            }
            uint _S1389 = i_13 * 6U;
            uint _S1390 = _S1389 + j_9;
            k_21 = 0U;
            sum_4 = (*a_15)[_S1390];
            for(;;)
            {
                if(k_21 < j_9)
                {
                }
                else
                {
                    break;
                }
                float sum_5 = sum_4 - l_6[_S1389 + k_21] * l_6[j_9 * 6U + k_21];
                k_21 = k_21 + 1U;
                sum_4 = sum_5;
            }
            if(i_13 == j_9)
            {
                if(sum_4 <= 0.0f)
                {
                    spd_0 = false;
                    break;
                }
                l_6[_S1389 + i_13] = sqrt(sum_4);
            }
            else
            {
                l_6[_S1390] = sum_4 / l_6[j_9 * 6U + j_9];
            }
            j_9 = j_9 + 1U;
        }
        i_13 = i_13 + 1U;
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
            uint _S1391 = k_21 * 6U + k_21;
            float _S1392 = (*a_15)[_S1391];
            if(((*a_15)[_S1391]) > 0.0f)
            {
                sum_4 = 1.0f / _S1392;
            }
            else
            {
                sum_4 = 0.0f;
            }
            inv_0[_S1391] = sum_4;
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
            i_13 = 0U;
            for(;;)
            {
                if(i_13 < 6U)
                {
                }
                else
                {
                    break;
                }
                if(i_13 == j_9)
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
                    if(k_21 < i_13)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_8 = s_7 - l_6[i_13 * 6U + k_21] * y_3[k_21];
                    k_21 = k_21 + 1U;
                    s_7 = s_8;
                }
                y_3[i_13] = s_7 / l_6[i_13 * 6U + i_13];
                i_13 = i_13 + 1U;
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
                uint i_14 = 5U - ii_2;
                k_21 = i_14 + 1U;
                sum_4 = y_3[i_14];
                for(;;)
                {
                    if(k_21 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_9 = sum_4 - l_6[k_21 * 6U + i_14] * x_16[k_21];
                    k_21 = k_21 + 1U;
                    sum_4 = s_9;
                }
                x_16[i_14] = sum_4 / l_6[i_14 * 6U + i_14];
                ii_2 = ii_2 + 1U;
            }
            uint i_15 = 0U;
            for(;;)
            {
                if(i_15 < 6U)
                {
                }
                else
                {
                    break;
                }
                inv_0[i_15 * 6U + j_9] = x_16[i_15];
                i_15 = i_15 + 1U;
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
        uint _S1393 = sv_0(c_45, 12U + j_9, kernelContext_88);
        uint _S1394 = 4U * j_9;
        *(kernelContext_88->scratch_0+_S1393) = packed_float4(float4(inv_0[_S1394], inv_0[_S1394 + 1U], inv_0[_S1394 + 2U], inv_0[_S1394 + 3U])) ;
        j_9 = j_9 + 1U;
    }
    return;
}

void assemble_block_0(uint c_46, KernelContext_0 thread* kernelContext_89)
{
    uint p_15;
    uint r_12;
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
    uint e_8 = kernelContext_89->index_0[c_46];
    for(;;)
    {
        if(e_8 < (kernelContext_89->index_0)[c_46 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_7 = kernelContext_89->index_0[e_8];
        uint i_16 = entry_7 >> 1U;
        bool _S1395 = (entry_7 & 1U) != 0U;
        BondStatic_natural_0 device* _S1396 = kernelContext_89->bonds_0+i_16;
        uint _S1397 = statics_bond_slot_0(i_16, kernelContext_89);
        float4 _S1398 = float4(*(kernelContext_89->scratch_0+_S1397)) ;
        float4 _S1399 = float4(*(kernelContext_89->scratch_0+(_S1397 + 1U))) ;
        p_15 = 0U;
        for(;;)
        {
            if(p_15 < 6U)
            {
            }
            else
            {
                break;
            }
            uint _S1400 = p_15 % 3U;
            float3 t_20;
            if(_S1400 == 0U)
            {
                t_20 = (float4(_S1396->t1_0) ).xyz;
            }
            else
            {
                if(_S1400 == 1U)
                {
                    t_20 = (float4(_S1396->t2_0) ).xyz;
                }
                else
                {
                    t_20 = (float4(_S1396->normal_0) ).xyz;
                }
            }
            bool _S1401 = p_15 < 3U;
            float3 row_u_0;
            float3 row_t_0;
            if(_S1401)
            {
                if(_S1395)
                {
                    row_u_0 = t_20;
                }
                else
                {
                    row_u_0 = - t_20;
                }
                if(_S1395)
                {
                    row_t_0 = cross((float4(_S1396->rb_0) ).xyz, t_20);
                }
                else
                {
                    row_t_0 = - cross((float4(_S1396->ra_0) ).xyz, t_20);
                }
            }
            else
            {
                float3 _S1402 = float3(0.0f) ;
                if(_S1395)
                {
                    row_u_0 = t_20;
                }
                else
                {
                    row_u_0 = - t_20;
                }
                float3 _S1403 = row_u_0;
                row_u_0 = _S1402;
                row_t_0 = _S1403;
            }
            float kp_0;
            if(_S1401)
            {
                kp_0 = _S1398[p_15];
            }
            else
            {
                kp_0 = _S1399[p_15 - 3U];
            }
            if(kp_0 == 0.0f)
            {
                p_15 = p_15 + 1U;
                continue;
            }
            array<float, int(6)> _S1404 = { { row_u_0.x, row_u_0.y, row_u_0.z, row_t_0.x, row_t_0.y, row_t_0.z } };
            r_12 = 0U;
            for(;;)
            {
                if(r_12 < 6U)
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
                    a_16[r_12 * 6U + q_14] = a_16[r_12 * 6U + q_14] + kp_0 * _S1404[r_12] * _S1404[q_14];
                    q_14 = q_14 + 1U;
                }
                r_12 = r_12 + 1U;
            }
            p_15 = p_15 + 1U;
        }
        e_8 = e_8 + 1U;
    }
    uint _S1405 = fixed_mask_0(c_46, kernelContext_89);
    p_15 = 0U;
    for(;;)
    {
        if(p_15 < 6U)
        {
        }
        else
        {
            break;
        }
        if((_S1405 & (1U << p_15)) != 0U)
        {
            r_12 = 0U;
            for(;;)
            {
                if(r_12 < 6U)
                {
                }
                else
                {
                    break;
                }
                a_16[p_15 * 6U + r_12] = 0.0f;
                a_16[r_12 * 6U + p_15] = 0.0f;
                r_12 = r_12 + 1U;
            }
            a_16[p_15 * 6U + p_15] = 1.0f;
        }
        p_15 = p_15 + 1U;
    }
    p_15 = 0U;
    for(;;)
    {
        if(p_15 < 6U)
        {
        }
        else
        {
            break;
        }
        if((a_16[p_15 * 6U + p_15]) == 0.0f)
        {
            a_16[p_15 * 6U + p_15] = 1.0f;
        }
        p_15 = p_15 + 1U;
    }
    thread array<float, int(36)> _S1406 = a_16;
    store_inverse_0(c_46, &_S1406, kernelContext_89);
    return;
}

float block_get_0(uint c_47, uint i_17, uint j_10, KernelContext_0 thread* kernelContext_90)
{
    uint k_23 = i_17 * 6U + j_10;
    uint _S1407 = sv_0(c_47, 12U + k_23 / 4U, kernelContext_90);
    return (*(kernelContext_90->scratch_0+_S1407))[k_23 % 4U];
}

void precondition_0(uint c_48, KernelContext_0 thread* kernelContext_91)
{
    uint _S1408 = sv_0(c_48, 4U, kernelContext_91);
    float4 _S1409 = float4(*(kernelContext_91->scratch_0+_S1408)) ;
    uint _S1410 = sv_0(c_48, 5U, kernelContext_91);
    float4 _S1411 = float4(*(kernelContext_91->scratch_0+_S1410)) ;
    array<float, int(6)> _S1412 = { { _S1409.x, _S1409.y, _S1409.z, _S1411.x, _S1411.y, _S1411.z } };
    thread array<float, int(6)> z_1;
    uint i_18 = 0U;
    for(;;)
    {
        if(i_18 < 6U)
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
            float _S1413 = block_get_0(c_48, i_18, j_11, kernelContext_91);
            float s_11 = s_10 + _S1413 * _S1412[j_11];
            j_11 = j_11 + 1U;
            s_10 = s_11;
        }
        z_1[i_18] = s_10;
        i_18 = i_18 + 1U;
    }
    uint _S1414 = sv_0(c_48, 6U, kernelContext_91);
    *(kernelContext_91->scratch_0+_S1414) = packed_float4(float4(z_1[int(0)], z_1[int(1)], z_1[int(2)], 0.0f)) ;
    uint _S1415 = sv_0(c_48, 7U, kernelContext_91);
    *(kernelContext_91->scratch_0+_S1415) = packed_float4(float4(z_1[int(3)], z_1[int(4)], z_1[int(5)], 0.0f)) ;
    return;
}

void project_displacement_slot_0(uint tid_22, const Island_0 thread* isl_21, uint slot_5, KernelContext_0 thread* kernelContext_92)
{
    uint _S1416;
    float3 _S1417 = float3(0.0f) ;
    thread float3 p_16 = _S1417;
    thread float3 l_7 = _S1417;
    uint4 _S1418 = isl_21->range_0;
    uint _S1419 = isl_21->range_0.x + tid_22;
    uint c_49 = _S1419;
    for(;;)
    {
        uint _S1420 = _S1418.y;
        _S1416 = _S1420;
        if(c_49 < _S1420)
        {
        }
        else
        {
            break;
        }
        ChunkStatic_natural_0 device* _S1421 = kernelContext_92->chunks_0+c_49;
        uint _S1422 = sv_0(c_49, slot_5, kernelContext_92);
        float3 u_4 = (float4(*(kernelContext_92->scratch_0+_S1422)) ).xyz;
        uint _S1423 = sv_0(c_49, slot_5 + 1U, kernelContext_92);
        float3 th_5 = (float4(*(kernelContext_92->scratch_0+_S1423)) ).xyz;
        float4 _S1424 = float4(_S1421->center_0) ;
        float3 r_13 = _S1424.xyz - isl_21->com_0.xyz;
        float3 _S1425 = float3(_S1424.w) ;
        p_16 = p_16 + u_4 * _S1425;
        l_7 = l_7 + (cross(r_13, u_4) * _S1425 + rows_mul_0(float4(_S1421->inertia0_1) , float4(_S1421->inertia1_1) , float4(_S1421->inertia2_1) , th_5));
        c_49 = c_49 + 256U;
    }
    group_sum3_0(tid_22, &p_16, &l_7, kernelContext_92);
    float4 _S1426 = isl_21->com_0;
    float3 _S1427 = p_16 / float3(isl_21->com_0.w) ;
    float3 _S1428 = rows_mul_0(isl_21->inv0_0, isl_21->inv1_0, isl_21->inv2_0, l_7);
    c_49 = _S1419;
    for(;;)
    {
        if(c_49 < _S1416)
        {
        }
        else
        {
            break;
        }
        float3 r_14 = (float4((kernelContext_92->chunks_0+c_49)->center_0) ).xyz - _S1426.xyz;
        uint _S1429 = sv_0(c_49, slot_5, kernelContext_92);
        *(kernelContext_92->scratch_0+_S1429) = packed_float4(float4((float4(*(kernelContext_92->scratch_0+_S1429)) ).xyz - _S1427 - cross(_S1428, r_14), 0.0f)) ;
        uint _S1430 = sv_0(c_49, slot_5 + 1U, kernelContext_92);
        *(kernelContext_92->scratch_0+_S1430) = packed_float4(float4((float4(*(kernelContext_92->scratch_0+_S1430)) ).xyz - _S1428, 0.0f)) ;
        c_49 = c_49 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return;
}

uint statics_result_slot_0(uint island_4, KernelContext_0 thread* kernelContext_93)
{
    return kernelContext_93->params_0->statics_base_0 + 23U * kernelContext_93->params_0->chunk_count_0 + 2U * kernelContext_93->params_0->statics_bonds_0 + island_4;
}

void write_bond_loads_0(uint _S1431, uint _S1432, float3 _S1433, float3 _S1434, float _S1435, KernelContext_0 thread* kernelContext_94)
{
    BondStatic_natural_0 device* _S1436 = kernelContext_94->bonds_0+_S1432;
    float3 _S1437 = to_body_0(_S1432, _S1433, kernelContext_94);
    float3 _S1438 = to_body_0(_S1432, _S1434, kernelContext_94);
    uint _S1439 = 3U * _S1431;
    *(kernelContext_94->scratch_0+_S1439) = packed_float4(float4(_S1437, _S1435)) ;
    *(kernelContext_94->scratch_0+(_S1439 + 1U)) = packed_float4(float4(_S1438 + cross((float4(_S1436->ra_0) ).xyz, _S1437), 0.0f)) ;
    *(kernelContext_94->scratch_0+(_S1439 + 2U)) = packed_float4(float4(- _S1438 + cross((float4(_S1436->rb_0) ).xyz, - _S1437), 0.0f)) ;
    return;
}

void bond_kinematics_0(uint _S1440, float3 _S1441, float3 _S1442, float3 _S1443, float3 _S1444, float3 thread* _S1445, float3 thread* _S1446, KernelContext_0 thread* kernelContext_95)
{
    BondStatic_natural_0 device* _S1447 = kernelContext_95->bonds_0+_S1440;
    float3 _S1448 = to_local_0(_S1440, _S1443 + cross(_S1444, (float4(_S1447->rb_0) ).xyz) - (_S1441 + cross(_S1442, (float4(_S1447->ra_0) ).xyz)), kernelContext_95);
    *_S1445 = _S1448;
    float3 _S1449 = to_local_0(_S1440, _S1444 - _S1442, kernelContext_95);
    *_S1446 = _S1449;
    return;
}

[[kernel]] void island_statics(uint3 group_5 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], Params_0 constant* params_8 [[buffer(0)]], Island_natural_0 device* islands_8 [[buffer(9)]], uint device* index_8 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_8 [[buffer(3)]], packed_float4 device* state_10 [[buffer(6)]], packed_float4 device* scratch_8 [[buffer(8)]], packed_float4 device* contact_state_8 [[buffer(11)]], packed_float4 device* loads_8 [[buffer(5)]], Impactor_natural_0 device* impactors_8 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_8 [[buffer(7)]], BondStatic_natural_0 device* bonds_8 [[buffer(2)]], MaterialTable_0 constant* materials_8 [[buffer(1)]])
{
    uint i_19;
    bool converged_4;
    uint c_50;
    thread KernelContext_0 kernelContext_96;
    (&kernelContext_96)->params_0 = params_8;
    (&kernelContext_96)->islands_0 = islands_8;
    (&kernelContext_96)->index_0 = index_8;
    (&kernelContext_96)->chunks_0 = chunks_8;
    (&kernelContext_96)->state_0 = state_10;
    (&kernelContext_96)->scratch_0 = scratch_8;
    (&kernelContext_96)->contact_state_0 = contact_state_8;
    (&kernelContext_96)->loads_0 = loads_8;
    (&kernelContext_96)->impactors_0 = impactors_8;
    (&kernelContext_96)->bond_dyn_0 = bond_dyn_8;
    (&kernelContext_96)->bonds_0 = bonds_8;
    (&kernelContext_96)->materials_0 = materials_8;
    threadgroup array<float4, int(32)> g_part_a_8;
    (&kernelContext_96)->g_part_a_0 = &g_part_a_8;
    threadgroup array<float4, int(32)> g_part_b_8;
    (&kernelContext_96)->g_part_b_0 = &g_part_b_8;
    threadgroup array<uint, int(5)> g_seen_8;
    (&kernelContext_96)->g_seen_0 = &g_seen_8;
    threadgroup uint g_wide_run_8;
    (&kernelContext_96)->g_wide_run_0 = &g_wide_run_8;
    threadgroup array<uint, int(4096)> g_label_8;
    (&kernelContext_96)->g_label_0 = &g_label_8;
    threadgroup uint g_changed_8;
    (&kernelContext_96)->g_changed_0 = &g_changed_8;
    threadgroup uint g_halt_8;
    (&kernelContext_96)->g_halt_0 = &g_halt_8;
    threadgroup uint g_run_8;
    (&kernelContext_96)->g_run_0 = &g_run_8;
    threadgroup uint g_sums_8;
    (&kernelContext_96)->g_sums_0 = &g_sums_8;
    threadgroup array<uint, int(512)> g_label_narrow_8;
    (&kernelContext_96)->g_label_narrow_0 = &g_label_narrow_8;
    threadgroup uint g_changed_narrow_8;
    (&kernelContext_96)->g_changed_narrow_0 = &g_changed_narrow_8;
    uint tid_23 = thread_7.x;
    uint _S1450 = group_5.x;
    Island_0 _S1451 = group_uniform_island_0(_S1450, tid_23, &kernelContext_96);
    uint _S1452 = _S1451.info_0.z;
    if((_S1452 & 8U) == 0U)
    {
        return;
    }
    bool free_0 = ((_S1451.info_0.x) & 1U) == 0U;
    uint c0_1 = _S1451.range_0.x;
    uint c1_1 = _S1451.range_0.y;
    uint b0_0 = _S1451.range_0.z;
    uint _S1453 = _S1451.range_0.w;
    uint _S1454 = c0_1 + tid_23;
    uint c_51 = _S1454;
    for(;;)
    {
        if(c_51 < c1_1)
        {
        }
        else
        {
            break;
        }
        uint _S1455 = sv_0(c_51, 0U, &kernelContext_96);
        uint _S1456 = 2U * c_51;
        *((&kernelContext_96)->scratch_0+_S1455) = packed_float4(float4(*((&kernelContext_96)->scratch_0+((&kernelContext_96)->params_0->sload_base_0 + _S1456))) ) ;
        uint _S1457 = sv_0(c_51, 1U, &kernelContext_96);
        *((&kernelContext_96)->scratch_0+_S1457) = packed_float4(float4(*((&kernelContext_96)->scratch_0+((&kernelContext_96)->params_0->sload_base_0 + _S1456 + 1U))) ) ;
        uint _S1458 = sv_0(c_51, 21U, &kernelContext_96);
        packed_float4 _S1459 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_96)->scratch_0+_S1458) = _S1459;
        uint _S1460 = sv_0(c_51, 22U, &kernelContext_96);
        *((&kernelContext_96)->scratch_0+_S1460) = _S1459;
        c_51 = c_51 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if(free_0)
    {
        thread Island_0 _S1461 = _S1451;
        project_load_slot_0(tid_23, &_S1461, 0U, &kernelContext_96);
    }
    float _S1462 = island_dot_0(tid_23, c0_1, c1_1, 0U, 0U, &kernelContext_96);
    float _S1463 = max(sqrt(_S1462), 1.00000000317107685e-30f);
    float _S1464 = (&kernelContext_96)->params_0->statics_tol_0;
    uint _S1465 = min((&kernelContext_96)->params_0->statics_cg_0, 20U * (c1_1 - c0_1) * 6U + 200U);
    float previous_2 = 1.00000001504746622e+30f;
    float start_0 = 1.00000001504746622e+30f;
    float residual_0 = 0.0f;
    uint newton_0 = 0U;
    uint cg_total_0 = 0U;
    for(;;)
    {
        if(newton_0 < ((&kernelContext_96)->params_0->statics_newton_0))
        {
        }
        else
        {
            converged_4 = false;
            break;
        }
        uint _S1466 = b0_0 + tid_23;
        i_19 = _S1466;
        for(;;)
        {
            if(i_19 < _S1453)
            {
            }
            else
            {
                break;
            }
            JointResponse_0 _S1467 = static_response_0(i_19, &kernelContext_96);
            write_bond_loads_0(i_19, i_19, _S1467.force_lin_1, _S1467.force_ang_1, 0.0f, &kernelContext_96);
            i_19 = i_19 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float4 _S1468 = float4(0.0f) ;
        thread float4 magnitude_0 = _S1468;
        thread float4 unused_m_0 = _S1468;
        c_51 = _S1454;
        for(;;)
        {
            if(c_51 < c1_1)
            {
            }
            else
            {
                break;
            }
            thread float3 fi_3;
            thread float3 mi_4;
            gather_loads_0(c_51, &fi_3, &mi_4, &kernelContext_96);
            float _S1469 = bond_load_magnitude2_0(c_51, &kernelContext_96);
            magnitude_0.x = magnitude_0.x + _S1469;
            uint _S1470 = sv_0(c_51, 0U, &kernelContext_96);
            thread float4 r_lin_0 = float4((float4(*((&kernelContext_96)->scratch_0+_S1470)) ).xyz + fi_3, 0.0f);
            uint _S1471 = sv_0(c_51, 1U, &kernelContext_96);
            thread float4 r_ang_0 = float4((float4(*((&kernelContext_96)->scratch_0+_S1471)) ).xyz + mi_4, 0.0f);
            uint _S1472 = fixed_mask_0(c_51, &kernelContext_96);
            hold_0(_S1472, &r_lin_0, &r_ang_0, _S1468, _S1468);
            uint _S1473 = sv_0(c_51, 4U, &kernelContext_96);
            *((&kernelContext_96)->scratch_0+_S1473) = packed_float4(r_lin_0) ;
            uint _S1474 = sv_0(c_51, 5U, &kernelContext_96);
            *((&kernelContext_96)->scratch_0+_S1474) = packed_float4(r_ang_0) ;
            c_51 = c_51 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        group_sum2_0(tid_23, &magnitude_0, &unused_m_0, &kernelContext_96);
        if(free_0)
        {
            thread Island_0 _S1475 = _S1451;
            project_load_slot_0(tid_23, &_S1475, 4U, &kernelContext_96);
        }
        float _S1476 = island_dot_0(tid_23, c0_1, c1_1, 4U, 4U, &kernelContext_96);
        float residual_1 = sqrt(_S1476) / _S1463;
        float _S1477 = max(0.00100000004749745f, 3.83999986297567375e-06f * sqrt(magnitude_0.x) / _S1463);
        if(newton_0 == 0U)
        {
            start_0 = residual_1;
        }
        bool _S1478 = is_finite_0(residual_1);
        bool _S1479;
        if(_S1478)
        {
            if(residual_1 <= _S1464)
            {
                converged_4 = true;
            }
            else
            {
                if(residual_1 <= _S1477)
                {
                    converged_4 = residual_1 > (0.5f * previous_2);
                }
                else
                {
                    converged_4 = false;
                }
                if(converged_4)
                {
                    _S1479 = residual_1 <= start_0;
                }
                else
                {
                    _S1479 = false;
                }
                converged_4 = _S1479;
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
        if(!_S1478)
        {
            residual_0 = residual_1;
            converged_4 = false;
            break;
        }
        uint i_20 = _S1466;
        for(;;)
        {
            if(i_20 < _S1453)
            {
            }
            else
            {
                break;
            }
            BondStatic_natural_0 device* _S1480 = (&kernelContext_96)->bonds_0+i_20;
            thread float3 d_lin_6;
            thread float3 d_ang_5;
            static_kinematics_0(i_20, &d_lin_6, &d_ang_5, &kernelContext_96);
            thread JointBond_natural_0 _S1481 = _S1480->law_0;
            thread JointState_0 _S1482 = ((&kernelContext_96)->bond_dyn_0+i_20)->js_0;
            thread float3 f_lin_2;
            thread float3 f_ang_2;
            secant_factors_0(&_S1481, &_S1482, d_lin_6, &f_lin_2, &f_ang_2);
            float4 _S1483 = float4((&_S1481)->stiff0_0) ;
            uint _S1484 = statics_bond_slot_0(i_20, &kernelContext_96);
            float _S1485 = _S1483.y;
            *((&kernelContext_96)->scratch_0+_S1484) = packed_float4(float4(_S1485 * f_lin_2.x, _S1485 * f_lin_2.y, _S1483.x * f_lin_2.z, 0.0f)) ;
            *((&kernelContext_96)->scratch_0+(_S1484 + 1U)) = packed_float4(float4(_S1483.z * f_ang_2.x, _S1483.w * f_ang_2.y, (float4((&_S1481)->stiff1_0) ).x * f_ang_2.z, 0.0f)) ;
            i_20 = i_20 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        uint c_52 = _S1454;
        for(;;)
        {
            if(c_52 < c1_1)
            {
            }
            else
            {
                break;
            }
            assemble_block_0(c_52, &kernelContext_96);
            uint _S1486 = sv_0(c_52, 2U, &kernelContext_96);
            packed_float4 _S1487 = packed_float4(_S1468) ;
            *((&kernelContext_96)->scratch_0+_S1486) = _S1487;
            uint _S1488 = sv_0(c_52, 3U, &kernelContext_96);
            *((&kernelContext_96)->scratch_0+_S1488) = _S1487;
            c_52 = c_52 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        uint c_53 = _S1454;
        for(;;)
        {
            if(c_53 < c1_1)
            {
            }
            else
            {
                break;
            }
            precondition_0(c_53, &kernelContext_96);
            c_53 = c_53 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if(free_0)
        {
            thread Island_0 _S1489 = _S1451;
            project_displacement_slot_0(tid_23, &_S1489, 6U, &kernelContext_96);
        }
        uint c_54 = _S1454;
        for(;;)
        {
            if(c_54 < c1_1)
            {
            }
            else
            {
                break;
            }
            uint _S1490 = sv_0(c_54, 8U, &kernelContext_96);
            packed_float4 device* _S1491 = (&kernelContext_96)->scratch_0+_S1490;
            uint _S1492 = sv_0(c_54, 6U, &kernelContext_96);
            *_S1491 = packed_float4(float4(*((&kernelContext_96)->scratch_0+_S1492)) ) ;
            uint _S1493 = sv_0(c_54, 9U, &kernelContext_96);
            packed_float4 device* _S1494 = (&kernelContext_96)->scratch_0+_S1493;
            uint _S1495 = sv_0(c_54, 7U, &kernelContext_96);
            *_S1494 = packed_float4(float4(*((&kernelContext_96)->scratch_0+_S1495)) ) ;
            c_54 = c_54 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float _S1496 = island_dot_0(tid_23, c0_1, c1_1, 4U, 6U, &kernelContext_96);
        float _S1497 = island_dot_0(tid_23, c0_1, c1_1, 4U, 4U, &kernelContext_96);
        float _S1498 = sqrt(_S1497);
        float rz_0 = _S1496;
        uint k_24 = 0U;
        uint cg_total_1 = cg_total_0;
        for(;;)
        {
            if(k_24 < _S1465)
            {
                _S1479 = _S1498 > 0.0f;
            }
            else
            {
                _S1479 = false;
            }
            if(_S1479)
            {
            }
            else
            {
                cg_total_0 = cg_total_1;
                break;
            }
            uint i_21 = _S1466;
            for(;;)
            {
                if(i_21 < _S1453)
                {
                }
                else
                {
                    break;
                }
                uint4 _S1499 = uint4(((&kernelContext_96)->bonds_0+i_21)->law_0.ids_0) ;
                uint ca_2 = _S1499.y;
                uint cb_2 = _S1499.z;
                uint _S1500 = sv_0(ca_2, 8U, &kernelContext_96);
                float3 _S1501 = (float4(*((&kernelContext_96)->scratch_0+_S1500)) ).xyz;
                uint _S1502 = sv_0(ca_2, 9U, &kernelContext_96);
                float3 _S1503 = (float4(*((&kernelContext_96)->scratch_0+_S1502)) ).xyz;
                uint _S1504 = sv_0(cb_2, 8U, &kernelContext_96);
                float3 _S1505 = (float4(*((&kernelContext_96)->scratch_0+_S1504)) ).xyz;
                uint _S1506 = sv_0(cb_2, 9U, &kernelContext_96);
                thread float3 d_lin_7;
                thread float3 d_ang_6;
                bond_kinematics_0(i_21, _S1501, _S1503, _S1505, (float4(*((&kernelContext_96)->scratch_0+_S1506)) ).xyz, &d_lin_7, &d_ang_6, &kernelContext_96);
                uint _S1507 = statics_bond_slot_0(i_21, &kernelContext_96);
                write_bond_loads_0(i_21, i_21, d_lin_7 * (float4(*((&kernelContext_96)->scratch_0+_S1507)) ).xyz, d_ang_6 * (float4(*((&kernelContext_96)->scratch_0+(_S1507 + 1U))) ).xyz, 0.0f, &kernelContext_96);
                i_21 = i_21 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            c_50 = _S1454;
            for(;;)
            {
                if(c_50 < c1_1)
                {
                }
                else
                {
                    break;
                }
                thread float3 fi_4;
                thread float3 mi_5;
                gather_loads_0(c_50, &fi_4, &mi_5, &kernelContext_96);
                thread float4 ap_lin_0 = float4(- fi_4, 0.0f);
                thread float4 ap_ang_0 = float4(- mi_5, 0.0f);
                uint _S1508 = fixed_mask_0(c_50, &kernelContext_96);
                uint _S1509 = sv_0(c_50, 8U, &kernelContext_96);
                float4 _S1510 = float4(*((&kernelContext_96)->scratch_0+_S1509)) ;
                uint _S1511 = sv_0(c_50, 9U, &kernelContext_96);
                hold_0(_S1508, &ap_lin_0, &ap_ang_0, _S1510, float4(*((&kernelContext_96)->scratch_0+_S1511)) );
                uint _S1512 = sv_0(c_50, 10U, &kernelContext_96);
                *((&kernelContext_96)->scratch_0+_S1512) = packed_float4(ap_lin_0) ;
                uint _S1513 = sv_0(c_50, 11U, &kernelContext_96);
                *((&kernelContext_96)->scratch_0+_S1513) = packed_float4(ap_ang_0) ;
                c_50 = c_50 + 256U;
            }
            float _S1514 = island_dot_0(tid_23, c0_1, c1_1, 8U, 10U, &kernelContext_96);
            uint _S1515 = cg_total_1 + 1U;
            if(_S1514 <= 0.0f)
            {
                cg_total_0 = _S1515;
                break;
            }
            float _S1516 = rz_0 / _S1514;
            uint c_55 = _S1454;
            for(;;)
            {
                if(c_55 < c1_1)
                {
                }
                else
                {
                    break;
                }
                uint _S1517 = sv_0(c_55, 2U, &kernelContext_96);
                packed_float4 device* _S1518 = (&kernelContext_96)->scratch_0+_S1517;
                float3 _S1519 = (float4(*((&kernelContext_96)->scratch_0+_S1517)) ).xyz;
                uint _S1520 = sv_0(c_55, 8U, &kernelContext_96);
                float3 _S1521 = float3(_S1516) ;
                *_S1518 = packed_float4(float4(_S1519 + _S1521 * (float4(*((&kernelContext_96)->scratch_0+_S1520)) ).xyz, 0.0f)) ;
                uint _S1522 = sv_0(c_55, 3U, &kernelContext_96);
                packed_float4 device* _S1523 = (&kernelContext_96)->scratch_0+_S1522;
                float3 _S1524 = (float4(*((&kernelContext_96)->scratch_0+_S1522)) ).xyz;
                uint _S1525 = sv_0(c_55, 9U, &kernelContext_96);
                *_S1523 = packed_float4(float4(_S1524 + _S1521 * (float4(*((&kernelContext_96)->scratch_0+_S1525)) ).xyz, 0.0f)) ;
                uint _S1526 = sv_0(c_55, 4U, &kernelContext_96);
                packed_float4 device* _S1527 = (&kernelContext_96)->scratch_0+_S1526;
                float3 _S1528 = (float4(*((&kernelContext_96)->scratch_0+_S1526)) ).xyz;
                uint _S1529 = sv_0(c_55, 10U, &kernelContext_96);
                *_S1527 = packed_float4(float4(_S1528 - _S1521 * (float4(*((&kernelContext_96)->scratch_0+_S1529)) ).xyz, 0.0f)) ;
                uint _S1530 = sv_0(c_55, 5U, &kernelContext_96);
                packed_float4 device* _S1531 = (&kernelContext_96)->scratch_0+_S1530;
                float3 _S1532 = (float4(*((&kernelContext_96)->scratch_0+_S1530)) ).xyz;
                uint _S1533 = sv_0(c_55, 11U, &kernelContext_96);
                *_S1531 = packed_float4(float4(_S1532 - _S1521 * (float4(*((&kernelContext_96)->scratch_0+_S1533)) ).xyz, 0.0f)) ;
                c_55 = c_55 + 256U;
            }
            float _S1534 = island_dot_0(tid_23, c0_1, c1_1, 4U, 4U, &kernelContext_96);
            if((sqrt(_S1534)) <= (0.00009999999747379f * _S1498))
            {
                cg_total_0 = _S1515;
                break;
            }
            uint c_56 = _S1454;
            for(;;)
            {
                if(c_56 < c1_1)
                {
                }
                else
                {
                    break;
                }
                precondition_0(c_56, &kernelContext_96);
                c_56 = c_56 + 256U;
            }
            if(free_0)
            {
                threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                thread Island_0 _S1535 = _S1451;
                project_displacement_slot_0(tid_23, &_S1535, 6U, &kernelContext_96);
            }
            float _S1536 = island_dot_0(tid_23, c0_1, c1_1, 4U, 6U, &kernelContext_96);
            float _S1537 = _S1536 / rz_0;
            uint c_57 = _S1454;
            for(;;)
            {
                if(c_57 < c1_1)
                {
                }
                else
                {
                    break;
                }
                uint _S1538 = sv_0(c_57, 8U, &kernelContext_96);
                packed_float4 device* _S1539 = (&kernelContext_96)->scratch_0+_S1538;
                uint _S1540 = sv_0(c_57, 6U, &kernelContext_96);
                float3 _S1541 = float3(_S1537) ;
                *_S1539 = packed_float4(float4((float4(*((&kernelContext_96)->scratch_0+_S1540)) ).xyz + _S1541 * (float4(*((&kernelContext_96)->scratch_0+_S1538)) ).xyz, 0.0f)) ;
                uint _S1542 = sv_0(c_57, 9U, &kernelContext_96);
                packed_float4 device* _S1543 = (&kernelContext_96)->scratch_0+_S1542;
                uint _S1544 = sv_0(c_57, 7U, &kernelContext_96);
                *_S1543 = packed_float4(float4((float4(*((&kernelContext_96)->scratch_0+_S1544)) ).xyz + _S1541 * (float4(*((&kernelContext_96)->scratch_0+_S1542)) ).xyz, 0.0f)) ;
                c_57 = c_57 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            uint _S1545 = k_24 + 1U;
            rz_0 = _S1536;
            k_24 = _S1545;
            cg_total_1 = _S1515;
        }
        c_50 = _S1454;
        for(;;)
        {
            if(c_50 < c1_1)
            {
            }
            else
            {
                break;
            }
            uint _S1546 = 4U * c_50;
            thread float3 u_5 = (float4(*((&kernelContext_96)->state_0+_S1546)) ).xyz;
            uint _S1547 = sv_0(c_50, 21U, &kernelContext_96);
            thread float3 u_lo_0 = (float4(*((&kernelContext_96)->scratch_0+_S1547)) ).xyz;
            uint _S1548 = _S1546 + 1U;
            thread float3 th_6 = (float4(*((&kernelContext_96)->state_0+_S1548)) ).xyz;
            uint _S1549 = sv_0(c_50, 22U, &kernelContext_96);
            thread float3 th_lo_0 = (float4(*((&kernelContext_96)->scratch_0+_S1549)) ).xyz;
            uint _S1550 = sv_0(c_50, 2U, &kernelContext_96);
            comp_add_0(&u_5, &u_lo_0, (float4(*((&kernelContext_96)->scratch_0+_S1550)) ).xyz);
            uint _S1551 = sv_0(c_50, 3U, &kernelContext_96);
            comp_add_0(&th_6, &th_lo_0, (float4(*((&kernelContext_96)->scratch_0+_S1551)) ).xyz);
            *((&kernelContext_96)->state_0+_S1546) = packed_float4(float4(u_5, (float4(*((&kernelContext_96)->state_0+_S1546)) ).w)) ;
            *((&kernelContext_96)->state_0+_S1548) = packed_float4(float4(th_6, (float4(*((&kernelContext_96)->state_0+_S1548)) ).w)) ;
            *((&kernelContext_96)->scratch_0+_S1547) = packed_float4(float4(u_lo_0, 0.0f)) ;
            *((&kernelContext_96)->scratch_0+_S1549) = packed_float4(float4(th_lo_0, 0.0f)) ;
            c_50 = c_50 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        uint _S1552 = newton_0 + 1U;
        previous_2 = residual_1;
        residual_0 = residual_1;
        newton_0 = _S1552;
    }
    i_19 = b0_0 + tid_23;
    for(;;)
    {
        if(i_19 < _S1453)
        {
        }
        else
        {
            break;
        }
        JointResponse_0 _S1553 = static_response_0(i_19, &kernelContext_96);
        BondDyn_natural_0 device* _S1554 = (&kernelContext_96)->bond_dyn_0+i_19;
        float4 _S1555 = float4((*_S1554).force_lin_0) ;
        float4 _S1556 = float4((*_S1554).force_ang_0) ;
        float4 _S1557 = float4((*_S1554).sums_0) ;
        float4 _S1558 = float4((*_S1554).comps_0) ;
        uint4 _S1559 = uint4((*_S1554).events_0) ;
        thread BondDyn_0 bd_1;
        (&bd_1)->js_0 = (*_S1554).js_0;
        (&bd_1)->force_lin_0 = _S1555;
        (&bd_1)->force_ang_0 = _S1556;
        (&bd_1)->sums_0 = _S1557;
        (&bd_1)->comps_0 = _S1558;
        (&bd_1)->events_0 = _S1559;
        (&bd_1)->force_lin_0 = float4(_S1553.force_lin_1, _S1553.stored_2);
        (&bd_1)->force_ang_0 = float4(_S1553.force_ang_1, (&bd_1)->force_ang_0.w);
        BondDyn_natural_0 device* _S1560 = (&kernelContext_96)->bond_dyn_0+i_19;
        _S1560->js_0 = bd_1.js_0;
        _S1560->force_lin_0 = packed_float4(bd_1.force_lin_0) ;
        _S1560->force_ang_0 = packed_float4(bd_1.force_ang_0) ;
        _S1560->sums_0 = packed_float4(bd_1.sums_0) ;
        _S1560->comps_0 = packed_float4(bd_1.comps_0) ;
        _S1560->events_0 = packed_uint4(bd_1.events_0) ;
        write_bond_loads_0(i_19, i_19, _S1553.force_lin_1, _S1553.force_ang_1, max(_S1553.measures_0.tension_0, _S1553.measures_0.compression_0), &kernelContext_96);
        i_19 = i_19 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    c_51 = _S1454;
    for(;;)
    {
        if(c_51 < c1_1)
        {
        }
        else
        {
            break;
        }
        thread float3 fi_5;
        thread float3 mi_6;
        gather_loads_0(c_51, &fi_5, &mi_6, &kernelContext_96);
        uint _S1561 = fixed_mask_0(c_51, &kernelContext_96);
        float3 reaction_1;
        if(_S1561 != 0U)
        {
            uint _S1562 = sv_0(c_51, 0U, &kernelContext_96);
            reaction_1 = - ((float4(*((&kernelContext_96)->scratch_0+_S1562)) ).xyz + fi_5);
        }
        else
        {
            uint _S1563 = 4U * c_51;
            reaction_1 = float3((float4(*((&kernelContext_96)->state_0+(_S1563 + 1U))) ).w, (float4(*((&kernelContext_96)->state_0+(_S1563 + 2U))) ).w, (float4(*((&kernelContext_96)->state_0+(_S1563 + 3U))) ).w);
        }
        uint _S1564 = 4U * c_51;
        uint _S1565 = _S1564 + 1U;
        *((&kernelContext_96)->state_0+_S1565) = packed_float4(float4((float4(*((&kernelContext_96)->state_0+_S1565)) ).xyz, reaction_1.x)) ;
        *((&kernelContext_96)->state_0+(_S1564 + 2U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, reaction_1.y)) ;
        *((&kernelContext_96)->state_0+(_S1564 + 3U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, reaction_1.z)) ;
        c_51 = c_51 + 256U;
    }
    if(tid_23 == 0U)
    {
        uint _S1566 = statics_result_slot_0(_S1450, &kernelContext_96);
        packed_float4 device* _S1567 = (&kernelContext_96)->scratch_0+_S1566;
        float _S1568 = (as_type<float>((newton_0)));
        float _S1569 = (as_type<float>((cg_total_0)));
        if(converged_4)
        {
            previous_2 = 1.0f;
        }
        else
        {
            previous_2 = 0.0f;
        }
        *_S1567 = packed_float4(float4(residual_0, _S1568, _S1569, previous_2)) ;
        ((&kernelContext_96)->islands_0+_S1450)->info_0[int(2)] = _S1452 & 4294967287U;
    }
    return;
}

[[kernel]] void settled_fatigue(uint3 group_6 [[threadgroup_position_in_grid]], uint3 thread_8 [[thread_position_in_threadgroup]], Params_0 constant* params_9 [[buffer(0)]], Island_natural_0 device* islands_9 [[buffer(9)]], uint device* index_9 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_9 [[buffer(3)]], packed_float4 device* state_11 [[buffer(6)]], packed_float4 device* scratch_9 [[buffer(8)]], packed_float4 device* contact_state_9 [[buffer(11)]], packed_float4 device* loads_9 [[buffer(5)]], Impactor_natural_0 device* impactors_9 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_9 [[buffer(7)]], BondStatic_natural_0 device* bonds_9 [[buffer(2)]], MaterialTable_0 constant* materials_9 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_97;
    (&kernelContext_97)->params_0 = params_9;
    (&kernelContext_97)->islands_0 = islands_9;
    (&kernelContext_97)->index_0 = index_9;
    (&kernelContext_97)->chunks_0 = chunks_9;
    (&kernelContext_97)->state_0 = state_11;
    (&kernelContext_97)->scratch_0 = scratch_9;
    (&kernelContext_97)->contact_state_0 = contact_state_9;
    (&kernelContext_97)->loads_0 = loads_9;
    (&kernelContext_97)->impactors_0 = impactors_9;
    (&kernelContext_97)->bond_dyn_0 = bond_dyn_9;
    (&kernelContext_97)->bonds_0 = bonds_9;
    (&kernelContext_97)->materials_0 = materials_9;
    threadgroup array<float4, int(32)> g_part_a_9;
    (&kernelContext_97)->g_part_a_0 = &g_part_a_9;
    threadgroup array<float4, int(32)> g_part_b_9;
    (&kernelContext_97)->g_part_b_0 = &g_part_b_9;
    threadgroup array<uint, int(5)> g_seen_9;
    (&kernelContext_97)->g_seen_0 = &g_seen_9;
    threadgroup uint g_wide_run_9;
    (&kernelContext_97)->g_wide_run_0 = &g_wide_run_9;
    threadgroup array<uint, int(4096)> g_label_9;
    (&kernelContext_97)->g_label_0 = &g_label_9;
    threadgroup uint g_changed_9;
    (&kernelContext_97)->g_changed_0 = &g_changed_9;
    threadgroup uint g_halt_9;
    (&kernelContext_97)->g_halt_0 = &g_halt_9;
    threadgroup uint g_run_9;
    (&kernelContext_97)->g_run_0 = &g_run_9;
    threadgroup uint g_sums_9;
    (&kernelContext_97)->g_sums_0 = &g_sums_9;
    threadgroup array<uint, int(512)> g_label_narrow_9;
    (&kernelContext_97)->g_label_narrow_0 = &g_label_narrow_9;
    threadgroup uint g_changed_narrow_9;
    (&kernelContext_97)->g_changed_narrow_0 = &g_changed_narrow_9;
    uint tid_24 = thread_8.x;
    uint _S1570 = group_6.x;
    Island_0 _S1571 = group_uniform_island_0(_S1570, tid_24, &kernelContext_97);
    bool _S1572;
    if(((_S1571.info_0.x) & 16U) == 0U)
    {
        _S1572 = true;
    }
    else
    {
        _S1572 = (_S1571.range_0.w) == (_S1571.range_0.z);
    }
    if(_S1572)
    {
        return;
    }
    seen_clear_0(tid_24, &kernelContext_97);
    if(tid_24 == 0U)
    {
        *(&kernelContext_97)->g_halt_0 = 0U;
        *(&kernelContext_97)->g_run_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S1573 = _S1571.info_0.w;
    uint i_22 = _S1571.range_0.z + tid_24;
    for(;;)
    {
        if(i_22 < (_S1571.range_0.w))
        {
        }
        else
        {
            break;
        }
        BondStatic_natural_0 device* _S1574 = (&kernelContext_97)->bonds_0+i_22;
        BondDyn_natural_0 device* _S1575 = (&kernelContext_97)->bond_dyn_0+i_22;
        float4 _S1576 = float4((*_S1575).force_lin_0) ;
        float4 _S1577 = float4((*_S1575).force_ang_0) ;
        float4 _S1578 = float4((*_S1575).sums_0) ;
        float4 _S1579 = float4((*_S1575).comps_0) ;
        uint4 _S1580 = uint4((*_S1575).events_0) ;
        thread BondDyn_0 bd_2;
        (&bd_2)->js_0 = (*_S1575).js_0;
        (&bd_2)->force_lin_0 = _S1576;
        (&bd_2)->force_ang_0 = _S1577;
        (&bd_2)->sums_0 = _S1578;
        (&bd_2)->comps_0 = _S1579;
        (&bd_2)->events_0 = _S1580;
        thread JointBond_natural_0 _S1581 = _S1574->law_0;
        uint4 _S1582 = uint4((&_S1581)->ids_0) ;
        uint _S1583 = 4U * _S1582.y;
        uint _S1584 = 4U * _S1582.z;
        thread float3 d_lin_8;
        thread float3 d_ang_7;
        bond_kinematics_0(i_22, (float4(*((&kernelContext_97)->state_0+_S1583)) ).xyz, (float4(*((&kernelContext_97)->state_0+(_S1583 + 1U))) ).xyz, (float4(*((&kernelContext_97)->state_0+_S1584)) ).xyz, (float4(*((&kernelContext_97)->state_0+(_S1584 + 1U))) ).xyz, &d_lin_8, &d_ang_7, &kernelContext_97);
        JointState_0 previous_3 = (&bd_2)->js_0;
        float _S1585 = (&kernelContext_97)->params_0->dt_0;
        bool _S1586 = ((&kernelContext_97)->params_0->fracture_0) != 0U;
        _S1581 = _S1574->law_0;
        thread JointState_0 _S1587 = (&bd_2)->js_0;
        JointResponse_0 _S1588 = joint_evaluate_0(&(&kernelContext_97)->materials_0->m_0[_S1582.x], &_S1581, &_S1587, d_lin_8, d_ang_7, _S1585, _S1586);
        if((_S1588.state_1.damage_0) > (previous_3.damage_0 + 9.99999971718068537e-10f))
        {
            _S1572 = true;
        }
        else
        {
            _S1572 = (_S1588.state_1.crush_1) > (previous_3.crush_1 + 9.99999971718068537e-10f);
        }
        uint flags_3;
        if(_S1572)
        {
            flags_3 = 16U;
        }
        else
        {
            flags_3 = 0U;
        }
        thread float _S1589 = (&bd_2)->sums_0.x;
        thread float _S1590 = (&bd_2)->comps_0.x;
        comp_add1_0(&_S1589, &_S1590, _S1588.dissipated_2);
        (&bd_2)->comps_0.x = _S1590;
        (&bd_2)->sums_0.x = _S1589;
        thread float _S1591 = (&bd_2)->sums_0.y;
        thread float _S1592 = (&bd_2)->comps_0.y;
        comp_add1_0(&_S1591, &_S1592, _S1588.overshoot_0);
        (&bd_2)->comps_0.y = _S1592;
        (&bd_2)->sums_0.y = _S1591;
        (&bd_2)->force_lin_0 = float4(_S1588.force_lin_1, _S1588.stored_2);
        (&bd_2)->force_ang_0 = float4(_S1588.force_ang_1, max((&bd_2)->force_ang_0.w, _S1588.state_1.utilization_0));
        thread JointState_0 _S1593 = previous_3;
        bool _S1594 = is_damaged_0(&_S1593);
        bool _S1595;
        if(!_S1594)
        {
            thread JointState_0 _S1596 = _S1588.state_1;
            bool _S1597 = is_damaged_0(&_S1596);
            _S1595 = _S1597;
        }
        else
        {
            _S1595 = false;
        }
        bool _S1598;
        if(_S1595)
        {
            _S1598 = ((&bd_2)->events_0.x) == 0U;
        }
        else
        {
            _S1598 = false;
        }
        if(_S1598)
        {
            (&bd_2)->events_0.x = _S1573;
            (&bd_2)->events_0.w = _S1588.state_1.mode_0;
            (*(&kernelContext_97)->g_seen_0)[3U] = 1U;
        }
        bool _S1599;
        if(((&bd_2)->events_0.y) == 0U)
        {
            _S1599 = (_S1588.state_1.life_0) != (previous_3.life_0);
        }
        else
        {
            _S1599 = false;
        }
        bool _S1600;
        if(_S1599)
        {
            _S1600 = (_S1588.fatigue_before_0) > 0.99000000953674316f;
        }
        else
        {
            _S1600 = false;
        }
        bool _S1601;
        if(_S1600)
        {
            float _S1602 = fatigue_factor_0(&(&kernelContext_97)->materials_0->m_0[_S1582.x], _S1588.state_1.life_0);
            _S1601 = _S1602 <= 0.99000000953674316f;
        }
        else
        {
            _S1601 = false;
        }
        if(_S1601)
        {
            (&bd_2)->events_0.y = _S1573;
            (*(&kernelContext_97)->g_seen_0)[4U] = 1U;
        }
        uint flags_4;
        if(_S1588.disconnected_0)
        {
            (&bd_2)->events_0.z = _S1573;
            flags_4 = flags_3 | 32U;
        }
        else
        {
            flags_4 = flags_3;
        }
        (&bd_2)->js_0 = _S1588.state_1;
        BondDyn_natural_0 device* _S1603 = (&kernelContext_97)->bond_dyn_0+i_22;
        _S1603->js_0 = bd_2.js_0;
        _S1603->force_lin_0 = packed_float4(bd_2.force_lin_0) ;
        _S1603->force_ang_0 = packed_float4(bd_2.force_ang_0) ;
        _S1603->sums_0 = packed_float4(bd_2.sums_0) ;
        _S1603->comps_0 = packed_float4(bd_2.comps_0) ;
        _S1603->events_0 = packed_uint4(bd_2.events_0) ;
        write_bond_loads_0(i_22, i_22, _S1588.force_lin_1, _S1588.force_ang_1, max(_S1588.measures_0.tension_0, _S1588.measures_0.compression_0), &kernelContext_97);
        if((flags_4 & 16U) != 0U)
        {
            *(&kernelContext_97)->g_halt_0 = 1U;
        }
        if((flags_4 & 32U) != 0U)
        {
            *(&kernelContext_97)->g_run_0 = 1U;
        }
        i_22 = i_22 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid_24 != 0U)
    {
        return;
    }
    if(((*(&kernelContext_97)->g_halt_0) | (*(&kernelContext_97)->g_run_0)) != 0U)
    {
        uint _S1604 = _S1571.info_0.z;
        if((*(&kernelContext_97)->g_halt_0) != 0U)
        {
            i_22 = 16U;
        }
        else
        {
            i_22 = 0U;
        }
        uint _S1605 = _S1604 | i_22;
        if((*(&kernelContext_97)->g_run_0) != 0U)
        {
            i_22 = 32U;
        }
        else
        {
            i_22 = 0U;
        }
        ((&kernelContext_97)->islands_0+_S1570)->info_0[int(2)] = _S1605 | i_22;
    }
    uint _S1606 = (_S1571.coverage_0) | 16384U;
    uint _S1607 = seen_bits_0(&kernelContext_97);
    uint _S1608 = _S1606 | _S1607;
    if((*(&kernelContext_97)->g_run_0) != 0U)
    {
        i_22 = 128U;
    }
    else
    {
        i_22 = 0U;
    }
    ((&kernelContext_97)->islands_0+_S1570)->coverage_0 = _S1608 | i_22;
    return;
}

void impactor_integrate_at_0(uint ii_3, uint tid_25, bool fused_4, KernelContext_0 thread* kernelContext_98)
{
    float3 p_17;
    if(ii_3 >= (kernelContext_98->params_0->impactor_count_0))
    {
        return;
    }
    uint _S1609 = kernelContext_98->params_0->halt_index_0;
    Island_natural_0 device* _S1610 = kernelContext_98->islands_0+kernelContext_98->params_0->halt_index_0;
    Impactor_natural_0 device* _S1611 = kernelContext_98->impactors_0+ii_3;
    float4 _S1612 = float4((*_S1611).position_err_1) ;
    float4 _S1613 = float4((*_S1611).velocity_1) ;
    float4 _S1614 = float4((*_S1611).velocity_err_1) ;
    float4 _S1615 = float4((*_S1611).angular_velocity_1) ;
    float4 _S1616 = float4((*_S1611).rotation_1) ;
    float4 _S1617 = float4((*_S1611).inertia0_2) ;
    float4 _S1618 = float4((*_S1611).inertia1_2) ;
    float4 _S1619 = float4((*_S1611).inertia2_2) ;
    float4 _S1620 = float4((*_S1611).inv0_2) ;
    float4 _S1621 = float4((*_S1611).inv1_2) ;
    float4 _S1622 = float4((*_S1611).inv2_2) ;
    float4 _S1623 = float4((*_S1611).shape_0) ;
    float4 _S1624 = float4((*_S1611).half_1) ;
    float4 _S1625 = float4((*_S1611).mat_0) ;
    float4 _S1626 = float4((*_S1611).crush_0) ;
    float4 _S1627 = float4((*_S1611).geom_0) ;
    float4 _S1628 = float4((*_S1611).rotation_err_1) ;
    float4 _S1629 = float4((*_S1611).ledger_0) ;
    uint4 _S1630 = uint4((*_S1611).cand_0) ;
    float4 _S1631 = float4((*_S1611).momentum_1) ;
    float4 _S1632 = float4((*_S1611).momentum_err_1) ;
    thread Impactor_0 imp_6;
    (&imp_6)->position_1 = float4((*_S1611).position_1) ;
    (&imp_6)->position_err_1 = _S1612;
    (&imp_6)->velocity_1 = _S1613;
    (&imp_6)->velocity_err_1 = _S1614;
    (&imp_6)->angular_velocity_1 = _S1615;
    (&imp_6)->rotation_1 = _S1616;
    (&imp_6)->inertia0_2 = _S1617;
    (&imp_6)->inertia1_2 = _S1618;
    (&imp_6)->inertia2_2 = _S1619;
    (&imp_6)->inv0_2 = _S1620;
    (&imp_6)->inv1_2 = _S1621;
    (&imp_6)->inv2_2 = _S1622;
    (&imp_6)->shape_0 = _S1623;
    (&imp_6)->half_1 = _S1624;
    (&imp_6)->mat_0 = _S1625;
    (&imp_6)->crush_0 = _S1626;
    (&imp_6)->geom_0 = _S1627;
    (&imp_6)->rotation_err_1 = _S1628;
    (&imp_6)->ledger_0 = _S1629;
    (&imp_6)->cand_0 = _S1630;
    (&imp_6)->momentum_1 = _S1631;
    (&imp_6)->momentum_err_1 = _S1632;
    uint4 _S1633 = uint4(_S1610->info_0) ;
    bool stop_0;
    if(((_S1633.z) & 1U) != 0U)
    {
        stop_0 = true;
    }
    else
    {
        uint _S1634 = _S1633.y;
        if(_S1634 != 0U)
        {
            stop_0 = ((&imp_6)->cand_0.w) >= _S1634;
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
            uint _S1635 = (uint4(_S1610->range_0) ).x;
            if(_S1635 != 0U)
            {
                stop_0 = _S1635 <= ((&imp_6)->cand_0.w + 1U);
            }
            else
            {
                stop_0 = false;
            }
        }
        if(tid_25 == 0U)
        {
            stop_seen_0(_S1609, (&imp_6)->cand_0.w + 1U, kernelContext_98);
        }
    }
    uint k_25;
    if(stop_0)
    {
        k_25 = 1U;
    }
    else
    {
        k_25 = 0U;
    }
    if((group_uniform_0(k_25, tid_25)) != 0U)
    {
        return;
    }
    if((group_uniform_0((&imp_6)->cand_0.z, tid_25)) != 0U)
    {
        if(tid_25 == 0U)
        {
            (kernelContext_98->impactors_0+ii_3)->cand_0[int(3)] = (&imp_6)->cand_0.w + 1U;
        }
        return;
    }
    float4 _S1636 = float4(0.0f) ;
    thread float4 rf_1 = _S1636;
    thread float4 rt_0 = _S1636;
    k_25 = (&imp_6)->cand_0.x + tid_25;
    for(;;)
    {
        if(k_25 < ((&imp_6)->cand_0.y))
        {
        }
        else
        {
            break;
        }
        uint _S1637 = 3U * k_25;
        rf_1 = rf_1 + float4(*(kernelContext_98->scratch_0+(kernelContext_98->params_0->cand_base_0 + _S1637 + 1U))) ;
        rt_0 = rt_0 + float4(*(kernelContext_98->scratch_0+(kernelContext_98->params_0->cand_base_0 + _S1637 + 2U))) ;
        k_25 = k_25 + 256U;
    }
    group_sum2_0(tid_25, &rf_1, &rt_0, kernelContext_98);
    if(tid_25 != 0U)
    {
        return;
    }
    float dt_12 = kernelContext_98->params_0->dt_0;
    float3 _S1638 = float3(0.0f) ;
    float3 load_f_0;
    float3 load_t_0;
    if((kernelContext_98->params_0->has_ground_0) != 0U)
    {
        float3 _S1639 = (&imp_6)->half_1.xyz;
        thread Impactor_0 _S1640 = imp_6;
        Box_0 _S1641 = impactor_box_0(&_S1640, _S1638, _S1639);
        float3 _S1642 = (&imp_6)->velocity_1.xyz + (&imp_6)->velocity_err_1.xyz;
        float _S1643 = kernelContext_98->params_0->ground_modulus_0;
        float _S1644 = (&imp_6)->mat_0.x;
        float3 _S1645 = float3(0.0f, 0.0f, 1.0f);
        thread Box_0 _S1646 = _S1641;
        thread Box_0 _S1647 = _S1641;
        float _S1648 = contact_stiffness_0(_S1643, &_S1646, _S1644, &_S1647, _S1645);
        float _S1649 = (&imp_6)->position_1.z - kernelContext_98->params_0->ground_hi_0 + ((&imp_6)->position_err_1.z - kernelContext_98->params_0->ground_lo_0);
        uint total_points_0;
        if(((&imp_6)->shape_0.x) == 0.0f)
        {
            total_points_0 = 1U;
        }
        else
        {
            total_points_0 = 14U;
        }
        float _S1650 = _S1648 / float(min(total_points_0, 5U));
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
                p_17 = float3(0.0f, 0.0f, - (&imp_6)->shape_0.y);
            }
            else
            {
                thread Box_0 _S1651 = _S1641;
                float3 _S1652 = sample_point_0(&_S1651, s_12, kernelContext_98);
                p_17 = _S1652;
            }
            if((_S1649 + p_17.z) < 0.0f)
            {
                below_0 = below_0 + 1U;
            }
            s_12 = s_12 + 1U;
        }
        if(below_0 > 0U)
        {
            sentinel_mark_0(8U, kernelContext_98);
        }
        s_12 = 0U;
        load_f_0 = _S1638;
        load_t_0 = _S1638;
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
                p_17 = float3(0.0f, 0.0f, - (&imp_6)->shape_0.y);
            }
            else
            {
                thread Box_0 _S1653 = _S1641;
                float3 _S1654 = sample_point_0(&_S1653, s_12, kernelContext_98);
                p_17 = _S1654;
            }
            float depth_3 = - (_S1649 + p_17.z);
            if(depth_3 <= 0.0f)
            {
                s_12 = s_12 + 1U;
                continue;
            }
            thread float stored_8;
            thread float diss_6;
            float3 _S1655 = penalty_force_0(_S1650, (&imp_6)->mat_0.z, kernelContext_98->params_0->ground_friction_0, depth_3, _S1645, _S1642 + cross((&imp_6)->angular_velocity_1.xyz, p_17), dt_12, below_0, &stored_8, &diss_6, kernelContext_98);
            float3 load_f_1 = load_f_0 + _S1655;
            float3 load_t_1 = load_t_0 + cross(p_17, _S1655);
            thread float _S1656 = (&imp_6)->ledger_0.x;
            thread float _S1657 = (&imp_6)->ledger_0.y;
            comp_add1_0(&_S1656, &_S1657, diss_6);
            (&imp_6)->ledger_0.y = _S1657;
            (&imp_6)->ledger_0.x = _S1656;
            load_f_0 = load_f_1;
            load_t_0 = load_t_1;
            s_12 = s_12 + 1U;
        }
    }
    else
    {
        load_f_0 = _S1638;
        load_t_0 = _S1638;
    }
    float3 load_f_2 = rf_1.xyz + load_f_0;
    float3 load_t_2 = rt_0.xyz + load_t_0;
    float m_7 = (&imp_6)->mat_0.z;
    thread float3 vel_2 = (&imp_6)->velocity_1.xyz;
    thread float3 vel_err_1 = (&imp_6)->velocity_err_1.xyz;
    float3 _S1658 = float3(dt_12) ;
    comp_add_0(&vel_2, &vel_err_1, (load_f_2 / float3(m_7)  + kernelContext_98->params_0->gravity_0.xyz) * _S1658);
    Quat_0 _S1659 = quat_of_0((&imp_6)->rotation_1);
    thread Quat_0 q_15 = _S1659;
    thread float4 q_err_0 = (&imp_6)->rotation_err_1;
    thread float3 l_hi_0 = (&imp_6)->momentum_1.xyz;
    thread float3 l_err_1 = (&imp_6)->momentum_err_1.xyz;
    comp_add_0(&l_hi_0, &l_err_1, load_t_2 * _S1658);
    float3 l_8 = l_hi_0 + l_err_1;
    thread Quat_0 _S1660 = _S1659;
    float3 _S1661 = world_mul_0(&_S1660, (&imp_6)->inv0_2, (&imp_6)->inv1_2, (&imp_6)->inv2_2, l_8);
    thread float3 pos_1 = (&imp_6)->position_1.xyz;
    thread float3 pos_err_1 = (&imp_6)->position_err_1.xyz;
    comp_add_0(&pos_1, &pos_err_1, (vel_2 + vel_err_1) * _S1658);
    turn_left_0(&q_15, &q_err_0, _S1661, dt_12);
    thread Quat_0 _S1662 = q_15;
    float3 _S1663 = world_mul_0(&_S1662, (&imp_6)->inv0_2, (&imp_6)->inv1_2, (&imp_6)->inv2_2, l_8);
    (&imp_6)->angular_velocity_1 = float4(_S1663, 0.0f);
    thread Quat_0 _S1664 = q_15;
    float4 _S1665 = quat_vec_0(&_S1664);
    (&imp_6)->rotation_1 = _S1665;
    (&imp_6)->rotation_err_1 = q_err_0;
    (&imp_6)->momentum_1 = float4(l_hi_0, 0.0f);
    (&imp_6)->momentum_err_1 = float4(l_err_1, 0.0f);
    (&imp_6)->position_1 = float4(pos_1, 0.0f);
    (&imp_6)->position_err_1 = float4(pos_err_1, 0.0f);
    (&imp_6)->velocity_1 = float4(vel_2, 0.0f);
    (&imp_6)->velocity_err_1 = float4(vel_err_1, 0.0f);
    (&imp_6)->cand_0.w = (&imp_6)->cand_0.w + 1U;
    (&imp_6)->geom_0 = float4(1.0f, (&imp_6)->crush_0.w, 0.0f, 0.0f);
    Impactor_natural_0 device* _S1666 = kernelContext_98->impactors_0+ii_3;
    _S1666->position_1 = packed_float4(imp_6.position_1) ;
    _S1666->position_err_1 = packed_float4(imp_6.position_err_1) ;
    _S1666->velocity_1 = packed_float4(imp_6.velocity_1) ;
    _S1666->velocity_err_1 = packed_float4(imp_6.velocity_err_1) ;
    _S1666->angular_velocity_1 = packed_float4(imp_6.angular_velocity_1) ;
    _S1666->rotation_1 = packed_float4(imp_6.rotation_1) ;
    _S1666->inertia0_2 = packed_float4(imp_6.inertia0_2) ;
    _S1666->inertia1_2 = packed_float4(imp_6.inertia1_2) ;
    _S1666->inertia2_2 = packed_float4(imp_6.inertia2_2) ;
    _S1666->inv0_2 = packed_float4(imp_6.inv0_2) ;
    _S1666->inv1_2 = packed_float4(imp_6.inv1_2) ;
    _S1666->inv2_2 = packed_float4(imp_6.inv2_2) ;
    _S1666->shape_0 = packed_float4(imp_6.shape_0) ;
    _S1666->half_1 = packed_float4(imp_6.half_1) ;
    _S1666->mat_0 = packed_float4(imp_6.mat_0) ;
    _S1666->crush_0 = packed_float4(imp_6.crush_0) ;
    _S1666->geom_0 = packed_float4(imp_6.geom_0) ;
    _S1666->rotation_err_1 = packed_float4(imp_6.rotation_err_1) ;
    _S1666->ledger_0 = packed_float4(imp_6.ledger_0) ;
    _S1666->cand_0 = packed_uint4(imp_6.cand_0) ;
    _S1666->momentum_1 = packed_float4(imp_6.momentum_1) ;
    _S1666->momentum_err_1 = packed_float4(imp_6.momentum_err_1) ;
    uint k_26 = (&imp_6)->cand_0.w - 1U - kernelContext_98->params_0->step_start_0;
    if(k_26 < (kernelContext_98->params_0->record_stride_0))
    {
        uint at_9 = kernelContext_98->params_0->record_base_0 + 2U * (ii_3 * kernelContext_98->params_0->record_stride_0 + k_26);
        *(kernelContext_98->scratch_0+at_9) = packed_float4(float4(vel_2 + vel_err_1, 0.0f)) ;
        *(kernelContext_98->scratch_0+(at_9 + 1U)) = packed_float4(float4(pos_1 + pos_err_1, 0.0f)) ;
    }
    if(fused_4)
    {
        impactor_travel_check_0(ii_3, true, kernelContext_98);
    }
    return;
}

[[kernel]] void impactor_integrate(uint3 group_7 [[threadgroup_position_in_grid]], uint3 thread_9 [[thread_position_in_threadgroup]], Params_0 constant* params_10 [[buffer(0)]], Island_natural_0 device* islands_10 [[buffer(9)]], uint device* index_10 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_10 [[buffer(3)]], packed_float4 device* state_12 [[buffer(6)]], packed_float4 device* scratch_10 [[buffer(8)]], packed_float4 device* contact_state_10 [[buffer(11)]], packed_float4 device* loads_10 [[buffer(5)]], Impactor_natural_0 device* impactors_10 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_10 [[buffer(7)]], BondStatic_natural_0 device* bonds_10 [[buffer(2)]], MaterialTable_0 constant* materials_10 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_10;
    (&kernelContext_99)->islands_0 = islands_10;
    (&kernelContext_99)->index_0 = index_10;
    (&kernelContext_99)->chunks_0 = chunks_10;
    (&kernelContext_99)->state_0 = state_12;
    (&kernelContext_99)->scratch_0 = scratch_10;
    (&kernelContext_99)->contact_state_0 = contact_state_10;
    (&kernelContext_99)->loads_0 = loads_10;
    (&kernelContext_99)->impactors_0 = impactors_10;
    (&kernelContext_99)->bond_dyn_0 = bond_dyn_10;
    (&kernelContext_99)->bonds_0 = bonds_10;
    (&kernelContext_99)->materials_0 = materials_10;
    threadgroup array<float4, int(32)> g_part_a_10;
    (&kernelContext_99)->g_part_a_0 = &g_part_a_10;
    threadgroup array<float4, int(32)> g_part_b_10;
    (&kernelContext_99)->g_part_b_0 = &g_part_b_10;
    threadgroup array<uint, int(5)> g_seen_10;
    (&kernelContext_99)->g_seen_0 = &g_seen_10;
    threadgroup uint g_wide_run_10;
    (&kernelContext_99)->g_wide_run_0 = &g_wide_run_10;
    threadgroup array<uint, int(4096)> g_label_10;
    (&kernelContext_99)->g_label_0 = &g_label_10;
    threadgroup uint g_changed_10;
    (&kernelContext_99)->g_changed_0 = &g_changed_10;
    threadgroup uint g_halt_10;
    (&kernelContext_99)->g_halt_0 = &g_halt_10;
    threadgroup uint g_run_10;
    (&kernelContext_99)->g_run_0 = &g_run_10;
    threadgroup uint g_sums_10;
    (&kernelContext_99)->g_sums_0 = &g_sums_10;
    threadgroup array<uint, int(512)> g_label_narrow_10;
    (&kernelContext_99)->g_label_narrow_0 = &g_label_narrow_10;
    threadgroup uint g_changed_narrow_10;
    (&kernelContext_99)->g_changed_narrow_0 = &g_changed_narrow_10;
    impactor_integrate_at_0(group_7.x, thread_9.x, false, &kernelContext_99);
    return;
}

void contact_sum_at_0(uint g_4, KernelContext_0 thread* kernelContext_100)
{
    bool _S1667;
    if(g_4 >= (kernelContext_100->params_0->seg_count_0))
    {
        _S1667 = true;
    }
    else
    {
        bool _S1668 = stopped_0(kernelContext_100);
        _S1667 = _S1668;
    }
    if(_S1667)
    {
        return;
    }
    contact_segment_0(g_4, kernelContext_100);
    return;
}

[[kernel]] void contact_sums(uint3 id_2 [[thread_position_in_grid]], Params_0 constant* params_11 [[buffer(0)]], Island_natural_0 device* islands_11 [[buffer(9)]], uint device* index_11 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_11 [[buffer(3)]], packed_float4 device* state_13 [[buffer(6)]], packed_float4 device* scratch_11 [[buffer(8)]], packed_float4 device* contact_state_11 [[buffer(11)]], packed_float4 device* loads_11 [[buffer(5)]], Impactor_natural_0 device* impactors_11 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_11 [[buffer(7)]], BondStatic_natural_0 device* bonds_11 [[buffer(2)]], MaterialTable_0 constant* materials_11 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_101;
    (&kernelContext_101)->params_0 = params_11;
    (&kernelContext_101)->islands_0 = islands_11;
    (&kernelContext_101)->index_0 = index_11;
    (&kernelContext_101)->chunks_0 = chunks_11;
    (&kernelContext_101)->state_0 = state_13;
    (&kernelContext_101)->scratch_0 = scratch_11;
    (&kernelContext_101)->contact_state_0 = contact_state_11;
    (&kernelContext_101)->loads_0 = loads_11;
    (&kernelContext_101)->impactors_0 = impactors_11;
    (&kernelContext_101)->bond_dyn_0 = bond_dyn_11;
    (&kernelContext_101)->bonds_0 = bonds_11;
    (&kernelContext_101)->materials_0 = materials_11;
    threadgroup array<float4, int(32)> g_part_a_11;
    (&kernelContext_101)->g_part_a_0 = &g_part_a_11;
    threadgroup array<float4, int(32)> g_part_b_11;
    (&kernelContext_101)->g_part_b_0 = &g_part_b_11;
    threadgroup array<uint, int(5)> g_seen_11;
    (&kernelContext_101)->g_seen_0 = &g_seen_11;
    threadgroup uint g_wide_run_11;
    (&kernelContext_101)->g_wide_run_0 = &g_wide_run_11;
    threadgroup array<uint, int(4096)> g_label_11;
    (&kernelContext_101)->g_label_0 = &g_label_11;
    threadgroup uint g_changed_11;
    (&kernelContext_101)->g_changed_0 = &g_changed_11;
    threadgroup uint g_halt_11;
    (&kernelContext_101)->g_halt_0 = &g_halt_11;
    threadgroup uint g_run_11;
    (&kernelContext_101)->g_run_0 = &g_run_11;
    threadgroup uint g_sums_11;
    (&kernelContext_101)->g_sums_0 = &g_sums_11;
    threadgroup array<uint, int(512)> g_label_narrow_11;
    (&kernelContext_101)->g_label_narrow_0 = &g_label_narrow_11;
    threadgroup uint g_changed_narrow_11;
    (&kernelContext_101)->g_changed_narrow_0 = &g_changed_narrow_11;
    contact_sum_at_0(id_2.x, &kernelContext_101);
    return;
}

[[kernel]] void island_frame(uint3 group_8 [[threadgroup_position_in_grid]], uint3 thread_10 [[thread_position_in_threadgroup]], Params_0 constant* params_12 [[buffer(0)]], Island_natural_0 device* islands_12 [[buffer(9)]], uint device* index_12 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_12 [[buffer(3)]], packed_float4 device* state_14 [[buffer(6)]], packed_float4 device* scratch_12 [[buffer(8)]], packed_float4 device* contact_state_12 [[buffer(11)]], packed_float4 device* loads_12 [[buffer(5)]], Impactor_natural_0 device* impactors_12 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_12 [[buffer(7)]], BondStatic_natural_0 device* bonds_12 [[buffer(2)]], MaterialTable_0 constant* materials_12 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_102;
    (&kernelContext_102)->params_0 = params_12;
    (&kernelContext_102)->islands_0 = islands_12;
    (&kernelContext_102)->index_0 = index_12;
    (&kernelContext_102)->chunks_0 = chunks_12;
    (&kernelContext_102)->state_0 = state_14;
    (&kernelContext_102)->scratch_0 = scratch_12;
    (&kernelContext_102)->contact_state_0 = contact_state_12;
    (&kernelContext_102)->loads_0 = loads_12;
    (&kernelContext_102)->impactors_0 = impactors_12;
    (&kernelContext_102)->bond_dyn_0 = bond_dyn_12;
    (&kernelContext_102)->bonds_0 = bonds_12;
    (&kernelContext_102)->materials_0 = materials_12;
    threadgroup array<float4, int(32)> g_part_a_12;
    (&kernelContext_102)->g_part_a_0 = &g_part_a_12;
    threadgroup array<float4, int(32)> g_part_b_12;
    (&kernelContext_102)->g_part_b_0 = &g_part_b_12;
    threadgroup array<uint, int(5)> g_seen_12;
    (&kernelContext_102)->g_seen_0 = &g_seen_12;
    threadgroup uint g_wide_run_12;
    (&kernelContext_102)->g_wide_run_0 = &g_wide_run_12;
    threadgroup array<uint, int(4096)> g_label_12;
    (&kernelContext_102)->g_label_0 = &g_label_12;
    threadgroup uint g_changed_12;
    (&kernelContext_102)->g_changed_0 = &g_changed_12;
    threadgroup uint g_halt_12;
    (&kernelContext_102)->g_halt_0 = &g_halt_12;
    threadgroup uint g_run_12;
    (&kernelContext_102)->g_run_0 = &g_run_12;
    threadgroup uint g_sums_12;
    (&kernelContext_102)->g_sums_0 = &g_sums_12;
    threadgroup array<uint, int(512)> g_label_narrow_12;
    (&kernelContext_102)->g_label_narrow_0 = &g_label_narrow_12;
    threadgroup uint g_changed_narrow_12;
    (&kernelContext_102)->g_changed_narrow_0 = &g_changed_narrow_12;
    island_frame_at_4(group_8.x, thread_10.x, &kernelContext_102);
    return;
}

[[kernel]] void island_frame_bonds_wide(uint3 group_9 [[threadgroup_position_in_grid]], uint3 thread_11 [[thread_position_in_threadgroup]], Params_0 constant* params_13 [[buffer(0)]], Island_natural_0 device* islands_13 [[buffer(9)]], uint device* index_13 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_13 [[buffer(3)]], packed_float4 device* state_15 [[buffer(6)]], packed_float4 device* scratch_13 [[buffer(8)]], packed_float4 device* contact_state_13 [[buffer(11)]], packed_float4 device* loads_13 [[buffer(5)]], Impactor_natural_0 device* impactors_13 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_13 [[buffer(7)]], BondStatic_natural_0 device* bonds_13 [[buffer(2)]], MaterialTable_0 constant* materials_13 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_103;
    (&kernelContext_103)->params_0 = params_13;
    (&kernelContext_103)->islands_0 = islands_13;
    (&kernelContext_103)->index_0 = index_13;
    (&kernelContext_103)->chunks_0 = chunks_13;
    (&kernelContext_103)->state_0 = state_15;
    (&kernelContext_103)->scratch_0 = scratch_13;
    (&kernelContext_103)->contact_state_0 = contact_state_13;
    (&kernelContext_103)->loads_0 = loads_13;
    (&kernelContext_103)->impactors_0 = impactors_13;
    (&kernelContext_103)->bond_dyn_0 = bond_dyn_13;
    (&kernelContext_103)->bonds_0 = bonds_13;
    (&kernelContext_103)->materials_0 = materials_13;
    threadgroup array<float4, int(32)> g_part_a_13;
    (&kernelContext_103)->g_part_a_0 = &g_part_a_13;
    threadgroup array<float4, int(32)> g_part_b_13;
    (&kernelContext_103)->g_part_b_0 = &g_part_b_13;
    threadgroup array<uint, int(5)> g_seen_13;
    (&kernelContext_103)->g_seen_0 = &g_seen_13;
    threadgroup uint g_wide_run_13;
    (&kernelContext_103)->g_wide_run_0 = &g_wide_run_13;
    threadgroup array<uint, int(4096)> g_label_13;
    (&kernelContext_103)->g_label_0 = &g_label_13;
    threadgroup uint g_changed_13;
    (&kernelContext_103)->g_changed_0 = &g_changed_13;
    threadgroup uint g_halt_13;
    (&kernelContext_103)->g_halt_0 = &g_halt_13;
    threadgroup uint g_run_13;
    (&kernelContext_103)->g_run_0 = &g_run_13;
    threadgroup uint g_sums_13;
    (&kernelContext_103)->g_sums_0 = &g_sums_13;
    threadgroup array<uint, int(512)> g_label_narrow_13;
    (&kernelContext_103)->g_label_narrow_0 = &g_label_narrow_13;
    threadgroup uint g_changed_narrow_13;
    (&kernelContext_103)->g_changed_narrow_0 = &g_changed_narrow_13;
    island_frame_at_3(group_9.x, thread_11.x, &kernelContext_103);
    return;
}

[[kernel]] void contact_islands(uint3 group_10 [[threadgroup_position_in_grid]], uint3 thread_12 [[thread_position_in_threadgroup]], Params_0 constant* params_14 [[buffer(0)]], Island_natural_0 device* islands_14 [[buffer(9)]], uint device* index_14 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_14 [[buffer(3)]], packed_float4 device* state_16 [[buffer(6)]], packed_float4 device* scratch_14 [[buffer(8)]], packed_float4 device* contact_state_14 [[buffer(11)]], packed_float4 device* loads_14 [[buffer(5)]], Impactor_natural_0 device* impactors_14 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_14 [[buffer(7)]], BondStatic_natural_0 device* bonds_14 [[buffer(2)]], MaterialTable_0 constant* materials_14 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_104;
    (&kernelContext_104)->params_0 = params_14;
    (&kernelContext_104)->islands_0 = islands_14;
    (&kernelContext_104)->index_0 = index_14;
    (&kernelContext_104)->chunks_0 = chunks_14;
    (&kernelContext_104)->state_0 = state_16;
    (&kernelContext_104)->scratch_0 = scratch_14;
    (&kernelContext_104)->contact_state_0 = contact_state_14;
    (&kernelContext_104)->loads_0 = loads_14;
    (&kernelContext_104)->impactors_0 = impactors_14;
    (&kernelContext_104)->bond_dyn_0 = bond_dyn_14;
    (&kernelContext_104)->bonds_0 = bonds_14;
    (&kernelContext_104)->materials_0 = materials_14;
    threadgroup array<float4, int(32)> g_part_a_14;
    (&kernelContext_104)->g_part_a_0 = &g_part_a_14;
    threadgroup array<float4, int(32)> g_part_b_14;
    (&kernelContext_104)->g_part_b_0 = &g_part_b_14;
    threadgroup array<uint, int(5)> g_seen_14;
    (&kernelContext_104)->g_seen_0 = &g_seen_14;
    threadgroup uint g_wide_run_14;
    (&kernelContext_104)->g_wide_run_0 = &g_wide_run_14;
    threadgroup array<uint, int(4096)> g_label_14;
    (&kernelContext_104)->g_label_0 = &g_label_14;
    threadgroup uint g_changed_14;
    (&kernelContext_104)->g_changed_0 = &g_changed_14;
    threadgroup uint g_halt_14;
    (&kernelContext_104)->g_halt_0 = &g_halt_14;
    threadgroup uint g_run_14;
    (&kernelContext_104)->g_run_0 = &g_run_14;
    threadgroup uint g_sums_14;
    (&kernelContext_104)->g_sums_0 = &g_sums_14;
    threadgroup array<uint, int(512)> g_label_narrow_14;
    (&kernelContext_104)->g_label_narrow_0 = &g_label_narrow_14;
    threadgroup uint g_changed_narrow_14;
    (&kernelContext_104)->g_changed_narrow_0 = &g_changed_narrow_14;
    island_frame_at_2(group_10.x, thread_12.x, &kernelContext_104);
    return;
}

void wide_chunks_at_0(uint gx_0, uint tid_26, KernelContext_0 thread* kernelContext_105)
{
    WideGroup_0 _S1669 = wide_group_uniform_0(kernelContext_105->params_0->wide_chunk_table_0, gx_0, tid_26, kernelContext_105);
    Island_0 _S1670 = group_uniform_island_0(_S1669.island_3, tid_26, kernelContext_105);
    thread Island_0 _S1671 = _S1670;
    bool _S1672 = wide_enter_0(tid_26, &_S1671, kernelContext_105);
    if(!_S1672)
    {
        return;
    }
    uint _S1673 = _S1670.info_0.x;
    bool anchored_0 = (_S1673 & 1U) != 0U;
    if((_S1673 & 16U) != 0U)
    {
        if(tid_26 == 0U)
        {
            *(kernelContext_105->scratch_0+(kernelContext_105->params_0->wide_base_0 + 8U * gx_0 + 6U)) = packed_float4(float4(0.0f) ) ;
        }
        return;
    }
    thread Island_0 _S1674 = _S1670;
    thread WideGroup_0 _S1675 = _S1669;
    Rigid_0 _S1676 = wide_rigid_frame_0(tid_26, &_S1674, &_S1675, kernelContext_105);
    thread float work_8 = 0.0f;
    thread float work_err_6 = 0.0f;
    float3 _S1677 = float3(0.0f) ;
    thread float3 tu_8 = _S1677;
    thread float3 pv_8 = _S1677;
    uint c_58 = _S1669.begin_1 + tid_26;
    if(c_58 < (_S1669.end_0))
    {
        float _S1678 = kernelContext_105->params_0->dt_0;
        bool _S1679 = (kernelContext_105->params_0->rigid_motion_loads_0) != 0U;
        thread Island_0 _S1680 = _S1670;
        uint _S1681 = wide_step_0(&_S1680, kernelContext_105);
        bool _S1682 = (_S1673 & 4U) != 0U;
        thread Island_0 _S1683 = _S1670;
        thread Rigid_0 _S1684 = _S1676;
        chunk_update_0(c_58, &_S1683, &_S1684, _S1678, _S1679, _S1681, _S1682, &work_8, &work_err_6, kernelContext_105);
        if(!anchored_0)
        {
            drift_moments_0(c_58, &tu_8, &pv_8, kernelContext_105);
        }
    }
    thread float3 wsum_5 = float3(work_8, work_err_6, 0.0f);
    thread float3 unused_8 = _S1677;
    group_sum3_0(tid_26, &wsum_5, &unused_8, kernelContext_105);
    bool _S1685 = !anchored_0;
    if(_S1685)
    {
        group_sum3_0(tid_26, &tu_8, &pv_8, kernelContext_105);
    }
    if(tid_26 == 0U)
    {
        uint _S1686 = 8U * gx_0;
        *(kernelContext_105->scratch_0+(kernelContext_105->params_0->wide_base_0 + _S1686 + 6U)) = packed_float4(float4(wsum_5, 0.0f)) ;
        if(_S1685)
        {
            wide_store_0(gx_0, 2U, tu_8, pv_8, kernelContext_105);
        }
        *(kernelContext_105->scratch_0+(kernelContext_105->params_0->wide_base_0 + _S1686 + 7U)) = packed_float4(float4(float((*kernelContext_105->g_seen_0)[0U]), float((*kernelContext_105->g_seen_0)[1U]), float((*kernelContext_105->g_seen_0)[2U]), 0.0f)) ;
    }
    return;
}

[[kernel]] void wide_chunks(uint3 group_11 [[threadgroup_position_in_grid]], uint3 thread_13 [[thread_position_in_threadgroup]], Params_0 constant* params_15 [[buffer(0)]], Island_natural_0 device* islands_15 [[buffer(9)]], uint device* index_15 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_15 [[buffer(3)]], packed_float4 device* state_17 [[buffer(6)]], packed_float4 device* scratch_15 [[buffer(8)]], packed_float4 device* contact_state_15 [[buffer(11)]], packed_float4 device* loads_15 [[buffer(5)]], Impactor_natural_0 device* impactors_15 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_15 [[buffer(7)]], BondStatic_natural_0 device* bonds_15 [[buffer(2)]], MaterialTable_0 constant* materials_15 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_106;
    (&kernelContext_106)->params_0 = params_15;
    (&kernelContext_106)->islands_0 = islands_15;
    (&kernelContext_106)->index_0 = index_15;
    (&kernelContext_106)->chunks_0 = chunks_15;
    (&kernelContext_106)->state_0 = state_17;
    (&kernelContext_106)->scratch_0 = scratch_15;
    (&kernelContext_106)->contact_state_0 = contact_state_15;
    (&kernelContext_106)->loads_0 = loads_15;
    (&kernelContext_106)->impactors_0 = impactors_15;
    (&kernelContext_106)->bond_dyn_0 = bond_dyn_15;
    (&kernelContext_106)->bonds_0 = bonds_15;
    (&kernelContext_106)->materials_0 = materials_15;
    threadgroup array<float4, int(32)> g_part_a_15;
    (&kernelContext_106)->g_part_a_0 = &g_part_a_15;
    threadgroup array<float4, int(32)> g_part_b_15;
    (&kernelContext_106)->g_part_b_0 = &g_part_b_15;
    threadgroup array<uint, int(5)> g_seen_15;
    (&kernelContext_106)->g_seen_0 = &g_seen_15;
    threadgroup uint g_wide_run_15;
    (&kernelContext_106)->g_wide_run_0 = &g_wide_run_15;
    threadgroup array<uint, int(4096)> g_label_15;
    (&kernelContext_106)->g_label_0 = &g_label_15;
    threadgroup uint g_changed_15;
    (&kernelContext_106)->g_changed_0 = &g_changed_15;
    threadgroup uint g_halt_15;
    (&kernelContext_106)->g_halt_0 = &g_halt_15;
    threadgroup uint g_run_15;
    (&kernelContext_106)->g_run_0 = &g_run_15;
    threadgroup uint g_sums_15;
    (&kernelContext_106)->g_sums_0 = &g_sums_15;
    threadgroup array<uint, int(512)> g_label_narrow_15;
    (&kernelContext_106)->g_label_narrow_0 = &g_label_narrow_15;
    threadgroup uint g_changed_narrow_15;
    (&kernelContext_106)->g_changed_narrow_0 = &g_changed_narrow_15;
    wide_chunks_at_0(group_11.x, thread_13.x, &kernelContext_106);
    return;
}

[[kernel]] void sums_and_impactors(uint3 group_12 [[threadgroup_position_in_grid]], uint3 thread_14 [[thread_position_in_threadgroup]], Params_0 constant* params_16 [[buffer(0)]], Island_natural_0 device* islands_16 [[buffer(9)]], uint device* index_16 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_16 [[buffer(3)]], packed_float4 device* state_18 [[buffer(6)]], packed_float4 device* scratch_16 [[buffer(8)]], packed_float4 device* contact_state_16 [[buffer(11)]], packed_float4 device* loads_16 [[buffer(5)]], Impactor_natural_0 device* impactors_16 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_16 [[buffer(7)]], BondStatic_natural_0 device* bonds_16 [[buffer(2)]], MaterialTable_0 constant* materials_16 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_107;
    (&kernelContext_107)->params_0 = params_16;
    (&kernelContext_107)->islands_0 = islands_16;
    (&kernelContext_107)->index_0 = index_16;
    (&kernelContext_107)->chunks_0 = chunks_16;
    (&kernelContext_107)->state_0 = state_18;
    (&kernelContext_107)->scratch_0 = scratch_16;
    (&kernelContext_107)->contact_state_0 = contact_state_16;
    (&kernelContext_107)->loads_0 = loads_16;
    (&kernelContext_107)->impactors_0 = impactors_16;
    (&kernelContext_107)->bond_dyn_0 = bond_dyn_16;
    (&kernelContext_107)->bonds_0 = bonds_16;
    (&kernelContext_107)->materials_0 = materials_16;
    threadgroup array<float4, int(32)> g_part_a_16;
    (&kernelContext_107)->g_part_a_0 = &g_part_a_16;
    threadgroup array<float4, int(32)> g_part_b_16;
    (&kernelContext_107)->g_part_b_0 = &g_part_b_16;
    threadgroup array<uint, int(5)> g_seen_16;
    (&kernelContext_107)->g_seen_0 = &g_seen_16;
    threadgroup uint g_wide_run_16;
    (&kernelContext_107)->g_wide_run_0 = &g_wide_run_16;
    threadgroup array<uint, int(4096)> g_label_16;
    (&kernelContext_107)->g_label_0 = &g_label_16;
    threadgroup uint g_changed_16;
    (&kernelContext_107)->g_changed_0 = &g_changed_16;
    threadgroup uint g_halt_16;
    (&kernelContext_107)->g_halt_0 = &g_halt_16;
    threadgroup uint g_run_16;
    (&kernelContext_107)->g_run_0 = &g_run_16;
    threadgroup uint g_sums_16;
    (&kernelContext_107)->g_sums_0 = &g_sums_16;
    threadgroup array<uint, int(512)> g_label_narrow_16;
    (&kernelContext_107)->g_label_narrow_0 = &g_label_narrow_16;
    threadgroup uint g_changed_narrow_16;
    (&kernelContext_107)->g_changed_narrow_0 = &g_changed_narrow_16;
    uint sums_groups_0 = (params_16->seg_count_0 + 255U) / 256U;
    uint _S1687 = group_12.x;
    if(_S1687 < sums_groups_0)
    {
        contact_sum_at_0(_S1687 * 256U + thread_14.x, &kernelContext_107);
    }
    else
    {
        impactor_integrate_at_0(_S1687 - sums_groups_0, thread_14.x, false, &kernelContext_107);
    }
    return;
}

[[kernel]] void islands_and_wide_chunks(uint3 group_13 [[threadgroup_position_in_grid]], uint3 thread_15 [[thread_position_in_threadgroup]], Params_0 constant* params_17 [[buffer(0)]], Island_natural_0 device* islands_17 [[buffer(9)]], uint device* index_17 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_17 [[buffer(3)]], packed_float4 device* state_19 [[buffer(6)]], packed_float4 device* scratch_17 [[buffer(8)]], packed_float4 device* contact_state_17 [[buffer(11)]], packed_float4 device* loads_17 [[buffer(5)]], Impactor_natural_0 device* impactors_17 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_17 [[buffer(7)]], BondStatic_natural_0 device* bonds_17 [[buffer(2)]], MaterialTable_0 constant* materials_17 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_108;
    (&kernelContext_108)->params_0 = params_17;
    (&kernelContext_108)->islands_0 = islands_17;
    (&kernelContext_108)->index_0 = index_17;
    (&kernelContext_108)->chunks_0 = chunks_17;
    (&kernelContext_108)->state_0 = state_19;
    (&kernelContext_108)->scratch_0 = scratch_17;
    (&kernelContext_108)->contact_state_0 = contact_state_17;
    (&kernelContext_108)->loads_0 = loads_17;
    (&kernelContext_108)->impactors_0 = impactors_17;
    (&kernelContext_108)->bond_dyn_0 = bond_dyn_17;
    (&kernelContext_108)->bonds_0 = bonds_17;
    (&kernelContext_108)->materials_0 = materials_17;
    threadgroup array<float4, int(32)> g_part_a_17;
    (&kernelContext_108)->g_part_a_0 = &g_part_a_17;
    threadgroup array<float4, int(32)> g_part_b_17;
    (&kernelContext_108)->g_part_b_0 = &g_part_b_17;
    threadgroup array<uint, int(5)> g_seen_17;
    (&kernelContext_108)->g_seen_0 = &g_seen_17;
    threadgroup uint g_wide_run_17;
    (&kernelContext_108)->g_wide_run_0 = &g_wide_run_17;
    threadgroup array<uint, int(4096)> g_label_17;
    (&kernelContext_108)->g_label_0 = &g_label_17;
    threadgroup uint g_changed_17;
    (&kernelContext_108)->g_changed_0 = &g_changed_17;
    threadgroup uint g_halt_17;
    (&kernelContext_108)->g_halt_0 = &g_halt_17;
    threadgroup uint g_run_17;
    (&kernelContext_108)->g_run_0 = &g_run_17;
    threadgroup uint g_sums_17;
    (&kernelContext_108)->g_sums_0 = &g_sums_17;
    threadgroup array<uint, int(512)> g_label_narrow_17;
    (&kernelContext_108)->g_label_narrow_0 = &g_label_narrow_17;
    threadgroup uint g_changed_narrow_17;
    (&kernelContext_108)->g_changed_narrow_0 = &g_changed_narrow_17;
    uint _S1688 = group_13.x;
    if(_S1688 < (params_17->halt_index_0))
    {
        island_frame_at_4(_S1688, thread_15.x, &kernelContext_108);
    }
    else
    {
        wide_chunks_at_0(_S1688 - params_17->halt_index_0, thread_15.x, &kernelContext_108);
    }
    return;
}

[[kernel]] void contact_islands_and_wide_chunks(uint3 group_14 [[threadgroup_position_in_grid]], uint3 thread_16 [[thread_position_in_threadgroup]], Params_0 constant* params_18 [[buffer(0)]], Island_natural_0 device* islands_18 [[buffer(9)]], uint device* index_18 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_18 [[buffer(3)]], packed_float4 device* state_20 [[buffer(6)]], packed_float4 device* scratch_18 [[buffer(8)]], packed_float4 device* contact_state_18 [[buffer(11)]], packed_float4 device* loads_18 [[buffer(5)]], Impactor_natural_0 device* impactors_18 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_18 [[buffer(7)]], BondStatic_natural_0 device* bonds_18 [[buffer(2)]], MaterialTable_0 constant* materials_18 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_109;
    (&kernelContext_109)->params_0 = params_18;
    (&kernelContext_109)->islands_0 = islands_18;
    (&kernelContext_109)->index_0 = index_18;
    (&kernelContext_109)->chunks_0 = chunks_18;
    (&kernelContext_109)->state_0 = state_20;
    (&kernelContext_109)->scratch_0 = scratch_18;
    (&kernelContext_109)->contact_state_0 = contact_state_18;
    (&kernelContext_109)->loads_0 = loads_18;
    (&kernelContext_109)->impactors_0 = impactors_18;
    (&kernelContext_109)->bond_dyn_0 = bond_dyn_18;
    (&kernelContext_109)->bonds_0 = bonds_18;
    (&kernelContext_109)->materials_0 = materials_18;
    threadgroup array<float4, int(32)> g_part_a_18;
    (&kernelContext_109)->g_part_a_0 = &g_part_a_18;
    threadgroup array<float4, int(32)> g_part_b_18;
    (&kernelContext_109)->g_part_b_0 = &g_part_b_18;
    threadgroup array<uint, int(5)> g_seen_18;
    (&kernelContext_109)->g_seen_0 = &g_seen_18;
    threadgroup uint g_wide_run_18;
    (&kernelContext_109)->g_wide_run_0 = &g_wide_run_18;
    threadgroup array<uint, int(4096)> g_label_18;
    (&kernelContext_109)->g_label_0 = &g_label_18;
    threadgroup uint g_changed_18;
    (&kernelContext_109)->g_changed_0 = &g_changed_18;
    threadgroup uint g_halt_18;
    (&kernelContext_109)->g_halt_0 = &g_halt_18;
    threadgroup uint g_run_18;
    (&kernelContext_109)->g_run_0 = &g_run_18;
    threadgroup uint g_sums_18;
    (&kernelContext_109)->g_sums_0 = &g_sums_18;
    threadgroup array<uint, int(512)> g_label_narrow_18;
    (&kernelContext_109)->g_label_narrow_0 = &g_label_narrow_18;
    threadgroup uint g_changed_narrow_18;
    (&kernelContext_109)->g_changed_narrow_0 = &g_changed_narrow_18;
    uint _S1689 = group_14.x;
    if(_S1689 < (params_18->halt_index_0))
    {
        island_frame_at_2(_S1689, thread_16.x, &kernelContext_109);
    }
    else
    {
        wide_chunks_at_0(_S1689 - params_18->halt_index_0, thread_16.x, &kernelContext_109);
    }
    return;
}

void wide_bonds_at_0(uint gx_1, uint tid_27, KernelContext_0 thread* kernelContext_110)
{
    bool bond_group_0 = gx_1 < (kernelContext_110->params_0->wide_bond_groups_0);
    WideGroup_0 wg_1;
    if(bond_group_0)
    {
        WideGroup_0 _S1690 = wide_group_uniform_0(kernelContext_110->params_0->wide_bond_table_0, gx_1, tid_27, kernelContext_110);
        wg_1 = _S1690;
    }
    else
    {
        WideGroup_0 _S1691 = wide_group_uniform_0(kernelContext_110->params_0->wide_chunk_table_0, gx_1 - kernelContext_110->params_0->wide_bond_groups_0, tid_27, kernelContext_110);
        wg_1 = _S1691;
    }
    WideGroup_0 _S1692 = wg_1;
    Island_0 _S1693 = group_uniform_island_0(wg_1.island_3, tid_27, kernelContext_110);
    thread Island_0 _S1694 = _S1693;
    bool _S1695 = wide_enter_0(tid_27, &_S1694, kernelContext_110);
    if(!_S1695)
    {
        return;
    }
    thread Island_0 _S1696 = _S1693;
    uint _S1697 = wide_step_0(&_S1696, kernelContext_110);
    if(bond_group_0)
    {
        if(((_S1693.info_0.x) & 16U) != 0U)
        {
            return;
        }
        uint i_23 = wg_1.begin_1 + tid_27;
        bool _S1698;
        if(i_23 < (wg_1.end_0))
        {
            bool _S1699 = bond_update_0(i_23, kernelContext_110->params_0->dt_0, (kernelContext_110->params_0->fracture_0) != 0U, _S1693.info_0.w + 1U, true, kernelContext_110);
            _S1698 = _S1699;
        }
        else
        {
            _S1698 = false;
        }
        if(_S1698)
        {
            (kernelContext_110->islands_0+_S1692.island_3)->info_0[int(2)] = (_S1693.info_0.z) | 2U;
        }
        return;
    }
    uint _S1700 = _S1693.info_0.x;
    if((_S1700 & 1U) != 0U)
    {
        return;
    }
    float3 _S1701 = float3(0.0f) ;
    thread float3 f_19 = _S1701;
    thread float3 t_21 = _S1701;
    uint c_59 = wg_1.begin_1 + tid_27;
    if(c_59 < (wg_1.end_0))
    {
        thread Island_0 _S1702 = _S1693;
        Rigid_0 _S1703 = rigid_of_0(&_S1702);
        float _S1704 = kernelContext_110->params_0->dt_0;
        bool _S1705 = (_S1700 & 4U) != 0U;
        thread Island_0 _S1706 = _S1693;
        thread Rigid_0 _S1707 = _S1703;
        net_load_0(c_59, &_S1706, &_S1707, _S1697, _S1704, _S1705, &f_19, &t_21, kernelContext_110);
    }
    group_sum3_0(tid_27, &f_19, &t_21, kernelContext_110);
    if(tid_27 == 0U)
    {
        wide_store_0(gx_1 - kernelContext_110->params_0->wide_bond_groups_0, 0U, f_19, t_21, kernelContext_110);
    }
    return;
}

[[kernel]] void wide_bonds(uint3 group_15 [[threadgroup_position_in_grid]], uint3 thread_17 [[thread_position_in_threadgroup]], Params_0 constant* params_19 [[buffer(0)]], Island_natural_0 device* islands_19 [[buffer(9)]], uint device* index_19 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_19 [[buffer(3)]], packed_float4 device* state_21 [[buffer(6)]], packed_float4 device* scratch_19 [[buffer(8)]], packed_float4 device* contact_state_19 [[buffer(11)]], packed_float4 device* loads_19 [[buffer(5)]], Impactor_natural_0 device* impactors_19 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_19 [[buffer(7)]], BondStatic_natural_0 device* bonds_19 [[buffer(2)]], MaterialTable_0 constant* materials_19 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_111;
    (&kernelContext_111)->params_0 = params_19;
    (&kernelContext_111)->islands_0 = islands_19;
    (&kernelContext_111)->index_0 = index_19;
    (&kernelContext_111)->chunks_0 = chunks_19;
    (&kernelContext_111)->state_0 = state_21;
    (&kernelContext_111)->scratch_0 = scratch_19;
    (&kernelContext_111)->contact_state_0 = contact_state_19;
    (&kernelContext_111)->loads_0 = loads_19;
    (&kernelContext_111)->impactors_0 = impactors_19;
    (&kernelContext_111)->bond_dyn_0 = bond_dyn_19;
    (&kernelContext_111)->bonds_0 = bonds_19;
    (&kernelContext_111)->materials_0 = materials_19;
    threadgroup array<float4, int(32)> g_part_a_19;
    (&kernelContext_111)->g_part_a_0 = &g_part_a_19;
    threadgroup array<float4, int(32)> g_part_b_19;
    (&kernelContext_111)->g_part_b_0 = &g_part_b_19;
    threadgroup array<uint, int(5)> g_seen_19;
    (&kernelContext_111)->g_seen_0 = &g_seen_19;
    threadgroup uint g_wide_run_19;
    (&kernelContext_111)->g_wide_run_0 = &g_wide_run_19;
    threadgroup array<uint, int(4096)> g_label_19;
    (&kernelContext_111)->g_label_0 = &g_label_19;
    threadgroup uint g_changed_19;
    (&kernelContext_111)->g_changed_0 = &g_changed_19;
    threadgroup uint g_halt_19;
    (&kernelContext_111)->g_halt_0 = &g_halt_19;
    threadgroup uint g_run_19;
    (&kernelContext_111)->g_run_0 = &g_run_19;
    threadgroup uint g_sums_19;
    (&kernelContext_111)->g_sums_0 = &g_sums_19;
    threadgroup array<uint, int(512)> g_label_narrow_19;
    (&kernelContext_111)->g_label_narrow_0 = &g_label_narrow_19;
    threadgroup uint g_changed_narrow_19;
    (&kernelContext_111)->g_changed_narrow_0 = &g_changed_narrow_19;
    wide_bonds_at_0(group_15.x, thread_17.x, &kernelContext_111);
    return;
}

void bond_group_at_0(uint table_2, uint gx_2, uint tid_28, KernelContext_0 thread* kernelContext_112)
{
    WideGroup_0 _S1708 = wide_group_uniform_0(table_2, gx_2, tid_28, kernelContext_112);
    Island_0 _S1709 = group_uniform_island_0(_S1708.island_3, tid_28, kernelContext_112);
    thread Island_0 _S1710 = _S1709;
    bool _S1711 = wide_enter_0(tid_28, &_S1710, kernelContext_112);
    bool _S1712;
    if(!_S1711)
    {
        _S1712 = true;
    }
    else
    {
        _S1712 = ((_S1709.info_0.x) & 16U) != 0U;
    }
    if(_S1712)
    {
        return;
    }
    uint i_24 = _S1708.begin_1 + tid_28;
    if(i_24 < (_S1708.end_0))
    {
        bool _S1713 = bond_update_0(i_24, kernelContext_112->params_0->dt_0, (kernelContext_112->params_0->fracture_0) != 0U, _S1709.info_0.w + 1U, true, kernelContext_112);
        _S1712 = _S1713;
    }
    else
    {
        _S1712 = false;
    }
    if(_S1712)
    {
        (kernelContext_112->islands_0+_S1708.island_3)->info_0[int(2)] = (_S1709.info_0.z) | 2U;
    }
    return;
}

[[kernel]] void sums_impactors_bonds(uint3 group_16 [[threadgroup_position_in_grid]], uint3 thread_18 [[thread_position_in_threadgroup]], Params_0 constant* params_20 [[buffer(0)]], Island_natural_0 device* islands_20 [[buffer(9)]], uint device* index_20 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_20 [[buffer(3)]], packed_float4 device* state_22 [[buffer(6)]], packed_float4 device* scratch_20 [[buffer(8)]], packed_float4 device* contact_state_20 [[buffer(11)]], packed_float4 device* loads_20 [[buffer(5)]], Impactor_natural_0 device* impactors_20 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_20 [[buffer(7)]], BondStatic_natural_0 device* bonds_20 [[buffer(2)]], MaterialTable_0 constant* materials_20 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_113;
    (&kernelContext_113)->params_0 = params_20;
    (&kernelContext_113)->islands_0 = islands_20;
    (&kernelContext_113)->index_0 = index_20;
    (&kernelContext_113)->chunks_0 = chunks_20;
    (&kernelContext_113)->state_0 = state_22;
    (&kernelContext_113)->scratch_0 = scratch_20;
    (&kernelContext_113)->contact_state_0 = contact_state_20;
    (&kernelContext_113)->loads_0 = loads_20;
    (&kernelContext_113)->impactors_0 = impactors_20;
    (&kernelContext_113)->bond_dyn_0 = bond_dyn_20;
    (&kernelContext_113)->bonds_0 = bonds_20;
    (&kernelContext_113)->materials_0 = materials_20;
    threadgroup array<float4, int(32)> g_part_a_20;
    (&kernelContext_113)->g_part_a_0 = &g_part_a_20;
    threadgroup array<float4, int(32)> g_part_b_20;
    (&kernelContext_113)->g_part_b_0 = &g_part_b_20;
    threadgroup array<uint, int(5)> g_seen_20;
    (&kernelContext_113)->g_seen_0 = &g_seen_20;
    threadgroup uint g_wide_run_20;
    (&kernelContext_113)->g_wide_run_0 = &g_wide_run_20;
    threadgroup array<uint, int(4096)> g_label_20;
    (&kernelContext_113)->g_label_0 = &g_label_20;
    threadgroup uint g_changed_20;
    (&kernelContext_113)->g_changed_0 = &g_changed_20;
    threadgroup uint g_halt_20;
    (&kernelContext_113)->g_halt_0 = &g_halt_20;
    threadgroup uint g_run_20;
    (&kernelContext_113)->g_run_0 = &g_run_20;
    threadgroup uint g_sums_20;
    (&kernelContext_113)->g_sums_0 = &g_sums_20;
    threadgroup array<uint, int(512)> g_label_narrow_20;
    (&kernelContext_113)->g_label_narrow_0 = &g_label_narrow_20;
    threadgroup uint g_changed_narrow_20;
    (&kernelContext_113)->g_changed_narrow_0 = &g_changed_narrow_20;
    uint sums_groups_1 = (params_20->seg_count_0 + 255U) / 256U;
    uint _S1714 = max(params_20->impactor_count_0, 1U);
    uint _S1715 = group_16.x;
    if(_S1715 < sums_groups_1)
    {
        contact_sum_at_0(_S1715 * 256U + thread_18.x, &kernelContext_113);
    }
    else
    {
        if(_S1715 < (sums_groups_1 + _S1714))
        {
            impactor_integrate_at_0(_S1715 - sums_groups_1, thread_18.x, false, &kernelContext_113);
        }
        else
        {
            uint gb_0 = _S1715 - sums_groups_1 - _S1714;
            bool narrow_0 = gb_0 < ((&kernelContext_113)->params_0->contact_bond_groups_0);
            uint _S1716;
            if(narrow_0)
            {
                _S1716 = (&kernelContext_113)->params_0->contact_bond_table_0;
            }
            else
            {
                _S1716 = (&kernelContext_113)->params_0->wide_bond_table_0;
            }
            uint _S1717;
            if(narrow_0)
            {
                _S1717 = gb_0;
            }
            else
            {
                _S1717 = gb_0 - (&kernelContext_113)->params_0->contact_bond_groups_0;
            }
            bond_group_at_0(_S1716, _S1717, thread_18.x, &kernelContext_113);
        }
    }
    return;
}

[[kernel]] void forces_and_bonds(uint3 group_17 [[threadgroup_position_in_grid]], uint3 thread_19 [[thread_position_in_threadgroup]], Params_0 constant* params_21 [[buffer(0)]], Island_natural_0 device* islands_21 [[buffer(9)]], uint device* index_21 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_21 [[buffer(3)]], packed_float4 device* state_23 [[buffer(6)]], packed_float4 device* scratch_21 [[buffer(8)]], packed_float4 device* contact_state_21 [[buffer(11)]], packed_float4 device* loads_21 [[buffer(5)]], Impactor_natural_0 device* impactors_21 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_21 [[buffer(7)]], BondStatic_natural_0 device* bonds_21 [[buffer(2)]], MaterialTable_0 constant* materials_21 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_114;
    (&kernelContext_114)->params_0 = params_21;
    (&kernelContext_114)->islands_0 = islands_21;
    (&kernelContext_114)->index_0 = index_21;
    (&kernelContext_114)->chunks_0 = chunks_21;
    (&kernelContext_114)->state_0 = state_23;
    (&kernelContext_114)->scratch_0 = scratch_21;
    (&kernelContext_114)->contact_state_0 = contact_state_21;
    (&kernelContext_114)->loads_0 = loads_21;
    (&kernelContext_114)->impactors_0 = impactors_21;
    (&kernelContext_114)->bond_dyn_0 = bond_dyn_21;
    (&kernelContext_114)->bonds_0 = bonds_21;
    (&kernelContext_114)->materials_0 = materials_21;
    threadgroup array<float4, int(32)> g_part_a_21;
    (&kernelContext_114)->g_part_a_0 = &g_part_a_21;
    threadgroup array<float4, int(32)> g_part_b_21;
    (&kernelContext_114)->g_part_b_0 = &g_part_b_21;
    threadgroup array<uint, int(5)> g_seen_21;
    (&kernelContext_114)->g_seen_0 = &g_seen_21;
    threadgroup uint g_wide_run_21;
    (&kernelContext_114)->g_wide_run_0 = &g_wide_run_21;
    threadgroup array<uint, int(4096)> g_label_21;
    (&kernelContext_114)->g_label_0 = &g_label_21;
    threadgroup uint g_changed_21;
    (&kernelContext_114)->g_changed_0 = &g_changed_21;
    threadgroup uint g_halt_21;
    (&kernelContext_114)->g_halt_0 = &g_halt_21;
    threadgroup uint g_run_21;
    (&kernelContext_114)->g_run_0 = &g_run_21;
    threadgroup uint g_sums_21;
    (&kernelContext_114)->g_sums_0 = &g_sums_21;
    threadgroup array<uint, int(512)> g_label_narrow_21;
    (&kernelContext_114)->g_label_narrow_0 = &g_label_narrow_21;
    threadgroup uint g_changed_narrow_21;
    (&kernelContext_114)->g_changed_narrow_0 = &g_changed_narrow_21;
    uint _S1718 = params_21->pair_count_0 + params_21->cand_count_0;
    uint _S1719 = pair_lane_count_0(&kernelContext_114);
    uint force_groups_0 = (_S1718 * _S1719 + 255U) / 256U;
    uint _S1720 = group_17.x;
    if(_S1720 < force_groups_0)
    {
        contact_forces_at_0(_S1720 * 256U + thread_19.x, true, &kernelContext_114);
    }
    else
    {
        bond_group_at_0((&kernelContext_114)->params_0->contact_bond_table_0, _S1720 - force_groups_0, thread_19.x, &kernelContext_114);
    }
    return;
}

[[kernel]] void islands_sums_impactors(uint3 group_18 [[threadgroup_position_in_grid]], uint3 thread_20 [[thread_position_in_threadgroup]], Params_0 constant* params_22 [[buffer(0)]], Island_natural_0 device* islands_22 [[buffer(9)]], uint device* index_22 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_22 [[buffer(3)]], packed_float4 device* state_24 [[buffer(6)]], packed_float4 device* scratch_22 [[buffer(8)]], packed_float4 device* contact_state_22 [[buffer(11)]], packed_float4 device* loads_22 [[buffer(5)]], Impactor_natural_0 device* impactors_22 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_22 [[buffer(7)]], BondStatic_natural_0 device* bonds_22 [[buffer(2)]], MaterialTable_0 constant* materials_22 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_115;
    (&kernelContext_115)->params_0 = params_22;
    (&kernelContext_115)->islands_0 = islands_22;
    (&kernelContext_115)->index_0 = index_22;
    (&kernelContext_115)->chunks_0 = chunks_22;
    (&kernelContext_115)->state_0 = state_24;
    (&kernelContext_115)->scratch_0 = scratch_22;
    (&kernelContext_115)->contact_state_0 = contact_state_22;
    (&kernelContext_115)->loads_0 = loads_22;
    (&kernelContext_115)->impactors_0 = impactors_22;
    (&kernelContext_115)->bond_dyn_0 = bond_dyn_22;
    (&kernelContext_115)->bonds_0 = bonds_22;
    (&kernelContext_115)->materials_0 = materials_22;
    threadgroup array<float4, int(32)> g_part_a_22;
    (&kernelContext_115)->g_part_a_0 = &g_part_a_22;
    threadgroup array<float4, int(32)> g_part_b_22;
    (&kernelContext_115)->g_part_b_0 = &g_part_b_22;
    threadgroup array<uint, int(5)> g_seen_22;
    (&kernelContext_115)->g_seen_0 = &g_seen_22;
    threadgroup uint g_wide_run_22;
    (&kernelContext_115)->g_wide_run_0 = &g_wide_run_22;
    threadgroup array<uint, int(4096)> g_label_22;
    (&kernelContext_115)->g_label_0 = &g_label_22;
    threadgroup uint g_changed_22;
    (&kernelContext_115)->g_changed_0 = &g_changed_22;
    threadgroup uint g_halt_22;
    (&kernelContext_115)->g_halt_0 = &g_halt_22;
    threadgroup uint g_run_22;
    (&kernelContext_115)->g_run_0 = &g_run_22;
    threadgroup uint g_sums_22;
    (&kernelContext_115)->g_sums_0 = &g_sums_22;
    threadgroup array<uint, int(512)> g_label_narrow_22;
    (&kernelContext_115)->g_label_narrow_0 = &g_label_narrow_22;
    threadgroup uint g_changed_narrow_22;
    (&kernelContext_115)->g_changed_narrow_0 = &g_changed_narrow_22;
    uint _S1721 = group_18.x;
    if(_S1721 < (params_22->halt_index_0))
    {
        island_frame_at_1(_S1721, thread_20.x, &kernelContext_115);
    }
    else
    {
        impactor_integrate_at_0(_S1721 - params_22->halt_index_0, thread_20.x, true, &kernelContext_115);
    }
    return;
}

[[kernel]] void islands_sums_impactors_wake(uint3 group_19 [[threadgroup_position_in_grid]], uint3 thread_21 [[thread_position_in_threadgroup]], Params_0 constant* params_23 [[buffer(0)]], Island_natural_0 device* islands_23 [[buffer(9)]], uint device* index_23 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_23 [[buffer(3)]], packed_float4 device* state_25 [[buffer(6)]], packed_float4 device* scratch_23 [[buffer(8)]], packed_float4 device* contact_state_23 [[buffer(11)]], packed_float4 device* loads_23 [[buffer(5)]], Impactor_natural_0 device* impactors_23 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_23 [[buffer(7)]], BondStatic_natural_0 device* bonds_23 [[buffer(2)]], MaterialTable_0 constant* materials_23 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_116;
    (&kernelContext_116)->params_0 = params_23;
    (&kernelContext_116)->islands_0 = islands_23;
    (&kernelContext_116)->index_0 = index_23;
    (&kernelContext_116)->chunks_0 = chunks_23;
    (&kernelContext_116)->state_0 = state_25;
    (&kernelContext_116)->scratch_0 = scratch_23;
    (&kernelContext_116)->contact_state_0 = contact_state_23;
    (&kernelContext_116)->loads_0 = loads_23;
    (&kernelContext_116)->impactors_0 = impactors_23;
    (&kernelContext_116)->bond_dyn_0 = bond_dyn_23;
    (&kernelContext_116)->bonds_0 = bonds_23;
    (&kernelContext_116)->materials_0 = materials_23;
    threadgroup array<float4, int(32)> g_part_a_23;
    (&kernelContext_116)->g_part_a_0 = &g_part_a_23;
    threadgroup array<float4, int(32)> g_part_b_23;
    (&kernelContext_116)->g_part_b_0 = &g_part_b_23;
    threadgroup array<uint, int(5)> g_seen_23;
    (&kernelContext_116)->g_seen_0 = &g_seen_23;
    threadgroup uint g_wide_run_23;
    (&kernelContext_116)->g_wide_run_0 = &g_wide_run_23;
    threadgroup array<uint, int(4096)> g_label_23;
    (&kernelContext_116)->g_label_0 = &g_label_23;
    threadgroup uint g_changed_23;
    (&kernelContext_116)->g_changed_0 = &g_changed_23;
    threadgroup uint g_halt_23;
    (&kernelContext_116)->g_halt_0 = &g_halt_23;
    threadgroup uint g_run_23;
    (&kernelContext_116)->g_run_0 = &g_run_23;
    threadgroup uint g_sums_23;
    (&kernelContext_116)->g_sums_0 = &g_sums_23;
    threadgroup array<uint, int(512)> g_label_narrow_23;
    (&kernelContext_116)->g_label_narrow_0 = &g_label_narrow_23;
    threadgroup uint g_changed_narrow_23;
    (&kernelContext_116)->g_changed_narrow_0 = &g_changed_narrow_23;
    uint _S1722 = group_19.x;
    if(_S1722 < (params_23->halt_index_0))
    {
        island_frame_at_0(_S1722, thread_21.x, &kernelContext_116);
    }
    else
    {
        impactor_integrate_at_0(_S1722 - params_23->halt_index_0, thread_21.x, true, &kernelContext_116);
    }
    return;
}

[[kernel]] void travel_start(uint3 id_3 [[thread_position_in_grid]], Params_0 constant* params_24 [[buffer(0)]], Island_natural_0 device* islands_24 [[buffer(9)]], uint device* index_24 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_24 [[buffer(3)]], packed_float4 device* state_26 [[buffer(6)]], packed_float4 device* scratch_24 [[buffer(8)]], packed_float4 device* contact_state_24 [[buffer(11)]], packed_float4 device* loads_24 [[buffer(5)]], Impactor_natural_0 device* impactors_24 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_24 [[buffer(7)]], BondStatic_natural_0 device* bonds_24 [[buffer(2)]], MaterialTable_0 constant* materials_24 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_117;
    (&kernelContext_117)->params_0 = params_24;
    (&kernelContext_117)->islands_0 = islands_24;
    (&kernelContext_117)->index_0 = index_24;
    (&kernelContext_117)->chunks_0 = chunks_24;
    (&kernelContext_117)->state_0 = state_26;
    (&kernelContext_117)->scratch_0 = scratch_24;
    (&kernelContext_117)->contact_state_0 = contact_state_24;
    (&kernelContext_117)->loads_0 = loads_24;
    (&kernelContext_117)->impactors_0 = impactors_24;
    (&kernelContext_117)->bond_dyn_0 = bond_dyn_24;
    (&kernelContext_117)->bonds_0 = bonds_24;
    (&kernelContext_117)->materials_0 = materials_24;
    threadgroup array<float4, int(32)> g_part_a_24;
    (&kernelContext_117)->g_part_a_0 = &g_part_a_24;
    threadgroup array<float4, int(32)> g_part_b_24;
    (&kernelContext_117)->g_part_b_0 = &g_part_b_24;
    threadgroup array<uint, int(5)> g_seen_24;
    (&kernelContext_117)->g_seen_0 = &g_seen_24;
    threadgroup uint g_wide_run_24;
    (&kernelContext_117)->g_wide_run_0 = &g_wide_run_24;
    threadgroup array<uint, int(4096)> g_label_24;
    (&kernelContext_117)->g_label_0 = &g_label_24;
    threadgroup uint g_changed_24;
    (&kernelContext_117)->g_changed_0 = &g_changed_24;
    threadgroup uint g_halt_24;
    (&kernelContext_117)->g_halt_0 = &g_halt_24;
    threadgroup uint g_run_24;
    (&kernelContext_117)->g_run_0 = &g_run_24;
    threadgroup uint g_sums_24;
    (&kernelContext_117)->g_sums_0 = &g_sums_24;
    threadgroup array<uint, int(512)> g_label_narrow_24;
    (&kernelContext_117)->g_label_narrow_0 = &g_label_narrow_24;
    threadgroup uint g_changed_narrow_24;
    (&kernelContext_117)->g_changed_narrow_0 = &g_changed_narrow_24;
    uint i_25 = id_3.x;
    if(i_25 < (params_24->chunk_count_0))
    {
        travel_check_0(i_25, true, &kernelContext_117);
    }
    else
    {
        if((i_25 - params_24->chunk_count_0) < ((&kernelContext_117)->params_0->impactor_count_0))
        {
            impactor_travel_check_0(i_25 - params_24->chunk_count_0, true, &kernelContext_117);
        }
    }
    return;
}

